"""T1 SceneFlow training entry point.

Reads the offline Phase-4.5 cache, drives `FlowFeatureAssembler` per step, and
computes a rectified-flow-style loss against a `SceneFlowMatching` module.

DDP scaffolding follows `train_tokenizer.py`. Visualization every `--vis_every`
steps dumps the same image set as `inference_scene_editor.py --dump_features`.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from datasets.waymo_flow_cache_dataset import SUPPORTED_CACHE_SCHEMA_VERSIONS, WaymoFlowCacheDataset
from dggt.losses.flow_losses import (
    boundary_mask_from_edit_mask,
    build_hard_edit_domain,
    build_masked_rectified_flow_target,
    compute_total_loss,
    masked_flow_euler_step,
    project_masked_flow_state,
    rae_t_grid,
)
from dggt.losses.rgb_render_loss import (
    compute_rgb_render_loss,
    decode_generated_dggt_geometry,
    rgb_render_loss_enabled,
    rgb_render_loss_ramp,
    rgb_render_sigma_weight,
    setup_lpips_for_rgb_loss,
    should_apply_rgb_render_loss,
)
from dggt.models.flow_feature_assembler import FlowFeatureAssembler
from dggt.models.asset_pass import (
    build_asset_condition_slots,
    require_asset_patch_valid_mask,
)
from dggt.models.joint_scene_tokenizer import JointSceneTokenizer
from dggt.models.embedders.text_encoder import TextEncoder
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.feature_quant import QuantizedTokens, dequantize_tokens
from dggt.utils.feature_stats import (
    DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH,
    FEATURE_STATS_SCHEMA,
    FEATURE_STATS_SCHEMA_VERSION,
    checkpoint_sha256,
    load_all_stats_into_buffers,
    validate_production_stats_coverage,
)
from dggt.utils.camera_condition import (
    camera_condition_from_waymo_metric_target,
    normalize_front_image_hw,
)
from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_STATS_VERSION,
    CAMERA_TARGET_SOURCE,
    CAMERA_TARGET_SPACE,
)
from dggt.utils.factorized_asset_condition import (
    FACTORIZED_ASSET_CONDITION_VERSION,
    PLACEMENT_STATE_DIM,
)
from dggt.utils.flow_cache_io import (
    is_chunked_flow_cache,
    load_chunked_flow_cache_subset,
    load_chunked_flow_cache_summary,
    load_flow_cache,
)
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.flow_schedule import (
    build_flow_schedule_config,
    validate_checkpoint_flow_schedule,
)
from dggt.utils.rae_optim import build_rae_optimizer, build_rae_scheduler
from dggt.utils.scene_gauge import (
    PULLBACK_RENDER_BOUNDARY,
    PULLBACK_RUNTIME_CONTRACT_VERSION,
    PullbackCalibration,
    SCENE_GAUGE_DIM,
    SCENE_GAUGE_REPRESENTATION,
    SCENE_GAUGE_STATS_VERSION,
    apply_pullback_calibration,
    load_pullback_calibration,
)
from dggt.utils.sliding_window import cosine_window, default_window_stride, window_slices
from dggt.utils.tokens import reattach_special_tokens, replace_selected_levels, select_patch_pyramid
from dggt.utils.tokenizer_checkpoint import load_scene_tokenizer_checkpoint_strict
from dggt.utils.tokenizer_window import (
    decode_tokenizer_windowed,
    encode_tokenizer_windowed,
)
from dggt.utils.validation_cache_naming import validation_asset_condition_kind
from dggt.utils.validation_rng import make_validation_generator, preserve_validation_rng_state
from diffusers.training_utils import EMAModel
from train_scene_flow_pretrain import (
    PRETRAIN_FEATURE_STATS_CONTRACT_KEY,
    TOKENIZER_LEVELS,
    DEFAULT_SKY_GRID,
    SKY_TOKEN_DIM,
    _image_grid,
    _latent_pca_grid,
    _mask_grid,
    _normalized_mask_grid,
    _render_gs_map_rgb,
    _semantic_logits_to_sky_mask,
    _sky_mask_image_grid,
    build_sky_tokens_from_images,
    load_dggt_aggregator_and_tokenizer,
    sky_grid_shape,
    split_image_tokens_for_heads,
    validate_pretrain_feature_stats_contract,
)


FORMAL_FLOW_DOMAIN_VERSION = "hard_binary_edit_domain_v1"
# All formal caches currently come from the 10 Hz Waymo camera stream. Keep
# mRoPE time coordinates identical to the factorized pretraining stage.
FORMAL_SCENE_FPS = 10.0
FORMAL_DGGT_CONTEXT_LENGTH = 29
FORMAL_TOKENIZER_WINDOW_LEN = 10
DEFAULT_FORMAL_PULLBACK_CALIBRATION = "data/scene_gauge/pullback_d63b34f7.json"
FORMAL_METRIC_GAUGE_CONTRACT_SCHEMA = "formal_scene_flow_metric_gauge_contract"
FORMAL_METRIC_GAUGE_CONTRACT_VERSION = "1.0.0"

METRIC_GAUGE_PROVENANCE_FIELDS = frozenset(
    {
        "scene_gauge_representation",
        "scene_gauge_stats_version",
        "gauge_table_sha256",
        "tokenizer_sha256",
        "dggt_checkpoint_sha256",
        "pullback_artifact_sha256",
        "pullback_runtime_contract_version",
        "pullback_window_len",
        "pullback_patch_grid_hw",
        "camera_generation_representation",
        "camera_target_space",
        "camera_target_source",
    }
)

FORMAL_METRIC_GAUGE_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "feature_stats_sha256",
        "dggt_context_length",
        "camera_gen_dim",
        "gauge_gen_dim",
        "placement_state_dim",
        "factorized_asset_condition_version",
        "dggt_checkpoint_sha256",
        "tokenizer_sha256",
        "gauge_table_sha256",
        "pullback_artifact_sha256",
        "pullback_window_len",
        "pullback_patch_grid_hw",
    }
)


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a lowercase 64-character SHA-256, got {value!r}"
        )
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(
            f"{name} must be a lowercase 64-character SHA-256, got {value!r}"
        )
    return value


def _config_value(config: Any, field: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _require_patch_grid(value: Any, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must contain two positive integers, got {value!r}")
    return int(value[0]), int(value[1])


def validate_metric_gauge_provenance(
    provenance: Any,
    *,
    config: Any,
    expected_dggt_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_pullback_runtime_contract_version: str | None = None,
    expected_window_len: int = FORMAL_TOKENIZER_WINDOW_LEN,
    expected_patch_grid: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate the unchanged 12-field pretrain metric/gauge provenance."""

    if not isinstance(provenance, Mapping):
        raise ValueError(
            "checkpoint is missing metric_gauge_provenance; legacy "
            "camera_dggt_provenance checkpoints are deliberately rejected"
        )
    actual_fields = set(provenance)
    if actual_fields != METRIC_GAUGE_PROVENANCE_FIELDS:
        raise ValueError(
            "metric_gauge_provenance does not match the strict 12-field schema: "
            f"missing={sorted(METRIC_GAUGE_PROVENANCE_FIELDS - actual_fields)}, "
            f"unknown={sorted(actual_fields - METRIC_GAUGE_PROVENANCE_FIELDS)}"
        )
    window_len = _require_positive_int(
        provenance["pullback_window_len"],
        name="metric_gauge_provenance.pullback_window_len",
    )
    expected_window = _require_positive_int(
        int(expected_window_len), name="expected_window_len"
    )
    artifact_grid = _require_patch_grid(
        provenance["pullback_patch_grid_hw"],
        name="metric_gauge_provenance.pullback_patch_grid_hw",
    )
    config_grid = _require_patch_grid(
        _config_value(config, "patch_grid"), name="SceneFlow config patch_grid"
    )
    runtime_grid = (
        config_grid
        if expected_patch_grid is None
        else _require_patch_grid(tuple(int(item) for item in expected_patch_grid), name="expected_patch_grid")
    )
    if config_grid != runtime_grid:
        raise ValueError(
            f"SceneFlow config patch_grid={config_grid} != runtime patch_grid={runtime_grid}"
        )
    runtime_contract_version = provenance["pullback_runtime_contract_version"]
    if runtime_contract_version != PULLBACK_RUNTIME_CONTRACT_VERSION:
        raise ValueError(
            "metric_gauge_provenance.pullback_runtime_contract_version is unsupported: "
            f"{runtime_contract_version!r}"
        )
    if (
        expected_pullback_runtime_contract_version is not None
        and expected_pullback_runtime_contract_version
        != PULLBACK_RUNTIME_CONTRACT_VERSION
    ):
        raise ValueError(
            "expected_pullback_runtime_contract_version is unsupported: "
            f"{expected_pullback_runtime_contract_version!r}"
        )

    expected_values = {
        "scene_gauge_representation": SCENE_GAUGE_REPRESENTATION,
        "scene_gauge_stats_version": SCENE_GAUGE_STATS_VERSION,
        "pullback_window_len": expected_window,
        "pullback_patch_grid_hw": list(runtime_grid),
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_target_space": CAMERA_TARGET_SPACE,
        "camera_target_source": CAMERA_TARGET_SOURCE,
    }
    if expected_pullback_runtime_contract_version is not None:
        expected_values["pullback_runtime_contract_version"] = (
            expected_pullback_runtime_contract_version
        )
    if expected_dggt_sha256 is not None:
        expected_values["dggt_checkpoint_sha256"] = _require_sha256(
            expected_dggt_sha256, name="runtime DGGT checkpoint SHA-256"
        )
    if expected_tokenizer_sha256 is not None:
        expected_values["tokenizer_sha256"] = _require_sha256(
            expected_tokenizer_sha256, name="runtime tokenizer checkpoint SHA-256"
        )
    for field, expected in expected_values.items():
        if provenance[field] != expected:
            raise ValueError(
                f"metric_gauge_provenance.{field} mismatch: "
                f"checkpoint={provenance[field]!r}, expected={expected!r}"
            )
    if window_len != expected_window or artifact_grid != runtime_grid:
        raise AssertionError("unreachable pullback window/grid mismatch")
    for field in (
        "gauge_table_sha256",
        "tokenizer_sha256",
        "dggt_checkpoint_sha256",
        "pullback_artifact_sha256",
    ):
        _require_sha256(
            provenance[field], name=f"metric_gauge_provenance.{field}"
        )

    config_expectations = {
        "camera_gen_dim": CAMERA_GENERATION_DIM,
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_stats_version": CAMERA_STATS_VERSION,
        "gauge_gen_dim": SCENE_GAUGE_DIM,
        "scene_gauge_representation": SCENE_GAUGE_REPRESENTATION,
        "scene_gauge_stats_version": SCENE_GAUGE_STATS_VERSION,
        "asset_condition_protocol": "factorized_v1",
        "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
    }
    for field, expected in config_expectations.items():
        actual = _config_value(config, field)
        if actual != expected:
            raise ValueError(
                f"SceneFlow config {field}={actual!r} does not satisfy the v4 "
                f"metric/gauge contract {expected!r}"
            )
    for field in ("placement_mean", "placement_std"):
        value = _config_value(config, field)
        if not isinstance(value, (list, tuple)) or len(value) != PLACEMENT_STATE_DIM:
            raise ValueError(
                f"SceneFlow config {field} must contain {PLACEMENT_STATE_DIM} values"
            )
    return dict(provenance)


def build_formal_metric_gauge_contract(
    provenance: Mapping[str, Any],
    *,
    feature_stats_sha256: str,
) -> dict[str, Any]:
    """Cross-bind formal-only metadata without changing base provenance."""

    return {
        "schema": FORMAL_METRIC_GAUGE_CONTRACT_SCHEMA,
        "version": FORMAL_METRIC_GAUGE_CONTRACT_VERSION,
        "feature_stats_sha256": _require_sha256(
            feature_stats_sha256, name="feature-stats SHA-256"
        ),
        "dggt_context_length": FORMAL_DGGT_CONTEXT_LENGTH,
        "camera_gen_dim": CAMERA_GENERATION_DIM,
        "gauge_gen_dim": SCENE_GAUGE_DIM,
        "placement_state_dim": PLACEMENT_STATE_DIM,
        "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
        "dggt_checkpoint_sha256": provenance["dggt_checkpoint_sha256"],
        "tokenizer_sha256": provenance["tokenizer_sha256"],
        "gauge_table_sha256": provenance["gauge_table_sha256"],
        "pullback_artifact_sha256": provenance["pullback_artifact_sha256"],
        "pullback_window_len": provenance["pullback_window_len"],
        "pullback_patch_grid_hw": list(provenance["pullback_patch_grid_hw"]),
    }


def validate_formal_metric_gauge_contract(
    contract: Any,
    *,
    provenance: Mapping[str, Any],
    expected_feature_stats_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the formal-only cross-binding block with an exact schema."""

    if not isinstance(contract, Mapping):
        raise ValueError(
            "formal checkpoint is missing formal_metric_gauge_contract; old formal "
            "checkpoints are deliberately rejected"
        )
    actual_fields = set(contract)
    if actual_fields != FORMAL_METRIC_GAUGE_CONTRACT_FIELDS:
        raise ValueError(
            "formal_metric_gauge_contract does not match the strict schema: "
            f"missing={sorted(FORMAL_METRIC_GAUGE_CONTRACT_FIELDS - actual_fields)}, "
            f"unknown={sorted(actual_fields - FORMAL_METRIC_GAUGE_CONTRACT_FIELDS)}"
        )
    expected = build_formal_metric_gauge_contract(
        provenance,
        feature_stats_sha256=(
            contract["feature_stats_sha256"]
            if expected_feature_stats_sha256 is None
            else expected_feature_stats_sha256
        ),
    )
    for field, expected_value in expected.items():
        if contract[field] != expected_value:
            raise ValueError(
                f"formal_metric_gauge_contract.{field} mismatch: "
                f"checkpoint={contract[field]!r}, expected={expected_value!r}"
            )
    _require_sha256(
        contract["feature_stats_sha256"],
        name="formal_metric_gauge_contract.feature_stats_sha256",
    )
    return dict(contract)


def validate_formal_feature_stats_artifact(
    path: str | Path,
    *,
    provenance: Mapping[str, Any],
    expected_window_len: int,
    expected_patch_grid: Sequence[int],
) -> str:
    """Validate and hash the exact production feature-stats artifact."""

    stats_path = Path(path)
    payload = torch.load(stats_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Feature stats at {stats_path} must be a dict")
    validate_production_stats_coverage(payload)
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Feature stats are missing source provenance")
    expected_source = {
        "sequence_length": int(expected_window_len),
        "dggt_context_length": FORMAL_DGGT_CONTEXT_LENGTH,
        "patch_grid": list(_require_patch_grid(tuple(int(v) for v in expected_patch_grid), name="expected_patch_grid")),
        "tokenizer_checkpoint_sha256": provenance["tokenizer_sha256"],
        "scene_gauge_table_sha256": provenance["gauge_table_sha256"],
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise ValueError(
                f"Feature stats source.{field} mismatch: "
                f"stats={source.get(field)!r}, expected={expected!r}"
            )
    top_level_expected = {
        "stats_schema": FEATURE_STATS_SCHEMA,
        "stats_schema_version": FEATURE_STATS_SCHEMA_VERSION,
        "dggt_checkpoint_sha256": provenance["dggt_checkpoint_sha256"],
        "tokenizer_checkpoint_sha256": provenance["tokenizer_sha256"],
        "gauge_table_sha256": provenance["gauge_table_sha256"],
        "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
        "placement_dim": PLACEMENT_STATE_DIM,
    }
    for field, expected in top_level_expected.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"Feature stats {field} mismatch: "
                f"stats={payload.get(field)!r}, expected={expected!r}"
            )
    return checkpoint_sha256(stats_path)


def validate_metric_gauge_checkpoint_payload(
    payload: Any,
    *,
    path: str | Path,
    config: Any,
    require_formal_contract: bool,
    expected_dggt_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_pullback_runtime_contract_version: str | None = None,
    expected_feature_stats_sha256: str | None = None,
    expected_window_len: int = FORMAL_TOKENIZER_WINDOW_LEN,
    expected_patch_grid: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate base provenance and, for formal checkpoints, its cross-binding."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} is not a versioned metric/gauge checkpoint")
    if "camera_dggt_provenance" in payload:
        raise ValueError(
            f"{path} contains legacy camera_dggt_provenance; old checkpoints are deliberately rejected"
        )
    provenance = validate_metric_gauge_provenance(
        payload.get("metric_gauge_provenance"),
        config=config,
        expected_dggt_sha256=expected_dggt_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        expected_pullback_runtime_contract_version=(
            expected_pullback_runtime_contract_version
        ),
        expected_window_len=expected_window_len,
        expected_patch_grid=expected_patch_grid,
    )
    raw_contract = payload.get("formal_metric_gauge_contract")
    if raw_contract is None:
        if require_formal_contract:
            raise ValueError(
                f"{path} is missing formal_metric_gauge_contract; old formal checkpoints are rejected"
            )
        return provenance, None
    contract = validate_formal_metric_gauge_contract(
        raw_contract,
        provenance=provenance,
        expected_feature_stats_sha256=expected_feature_stats_sha256,
    )
    return provenance, contract


def _asset_condition_kind_from_item(item: dict[str, Any], mode_kind: str) -> str:
    variant = item.get("validation_variant")
    if variant not in (None, ""):
        return validation_asset_condition_kind(str(variant))
    return (
        "mode_b_empty"
        if str(mode_kind) in ("mode_b", "mode_b_empty", "empty")
        else "mode_a"
    )


def formal_flow_domain_config(args) -> dict[str, Any]:
    return {
        "version": FORMAL_FLOW_DOMAIN_VERSION,
        "threshold": float(getattr(args, "edit_domain_threshold", 1e-4)),
        "dilation": int(getattr(args, "edit_domain_dilation", 1)),
    }


def validate_formal_flow_domain_config(payload: Any, args, path: str | Path) -> None:
    saved = payload.get("formal_flow_domain_config") if isinstance(payload, dict) else None
    expected = formal_flow_domain_config(args)
    if not isinstance(saved, dict):
        raise ValueError(f"{path} has no formal_flow_domain_config; expected {expected!r}")
    same = (
        saved.get("version") == expected["version"]
        and abs(float(saved.get("threshold", -1.0)) - expected["threshold"]) <= 1e-12
        and int(saved.get("dilation", -1)) == expected["dilation"]
    )
    if not same:
        raise ValueError(
            f"{path} formal flow-domain config {saved!r} does not match runtime {expected!r}. "
            "Training, validation, and offline inference must use the same binary edit domain."
        )


# ---------------------------------------------------------------------- #
# DDP + misc utilities (mirrored from train_tokenizer.py)                #
# ---------------------------------------------------------------------- #
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args) -> tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device, local_rank, world_size


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def init_wandb(args: argparse.Namespace, log_dir: Path):
    if not args.wandb or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or remove --wandb.") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        dir=str(log_dir),
        config=vars(args),
    )


def log_wandb(run, metrics: dict[str, float], step: int, prefix: str) -> None:
    if run is None:
        return
    run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)


def setup_text_encoder(args: argparse.Namespace, device: torch.device) -> nn.Module | None:
    if bool(getattr(args, "no_text_condition", False)):
        return None
    if not getattr(args, "caption_root", None):
        return None
    encoder = TextEncoder(
        model_name=str(args.text_encoder_path),
        max_length=int(args.text_max_length),
    ).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_text_condition(
    text_encoder: nn.Module | None,
    captions: list[str] | tuple[str, ...] | None,
    drop_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if text_encoder is None:
        return None, None
    if captions is None:
        raise RuntimeError("Text encoder is enabled but the batch has no captions.")
    clean = [str(c) if c is not None else "" for c in captions]
    if drop_mask is not None and int(drop_mask.numel()) != len(clean):
        raise ValueError(f"drop_mask has {int(drop_mask.numel())} rows, captions has {len(clean)}")
    out = text_encoder(clean)
    tokens = out["tokens"]
    attention_mask = out["attention_mask"]
    if drop_mask is None or not bool(drop_mask.to(dtype=torch.bool).any().item()):
        return tokens, attention_mask
    null_out = text_encoder([""] * len(clean))
    drop = drop_mask.to(device=tokens.device, dtype=torch.bool).view(len(clean), 1, 1)
    tokens = torch.where(drop, null_out["tokens"].to(device=tokens.device, dtype=tokens.dtype), tokens)
    mask_drop = drop_mask.to(device=attention_mask.device, dtype=torch.bool).view(len(clean), 1)
    attention_mask = torch.where(
        mask_drop,
        null_out["attention_mask"].to(device=attention_mask.device, dtype=attention_mask.dtype),
        attention_mask,
    )
    return tokens, attention_mask


def build_camera_condition_from_waymo_gt(
    camera_to_world: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    *,
    device: torch.device,
    image_hw: tuple[int, int] | None = None,
    trajectory_anchor_to_world: torch.Tensor | None = None,
    previous_camera_to_world: torch.Tensor | None = None,
    anchor_mask: torch.Tensor,
    scene_flow: nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not torch.is_tensor(camera_to_world) or not torch.is_tensor(intrinsics):
        return None, None
    if not torch.is_tensor(trajectory_anchor_to_world):
        raise ValueError(
            "Formal metric camera conditioning requires trajectory_anchor_to_world"
        )
    camera_metric = camera_to_world.to(device=device, dtype=torch.float32)
    intrinsics_metric = intrinsics.to(device=device, dtype=torch.float32)
    # A formal cache item is unbatched.  Its common multi-view layout
    # [S,V,4,4] / [S,V,3,3] is rank-identical to a batched front-camera
    # [B,S,...] tensor, so resolve that ambiguity at this single-item adapter
    # before entering the shared helper.
    if (
        camera_metric.ndim == 4
        and intrinsics_metric.ndim == 4
        and tuple(camera_metric.shape[:2]) == tuple(intrinsics_metric.shape[:2])
        and int(camera_metric.shape[0]) > 1
    ):
        camera_metric = camera_metric[:, 0].unsqueeze(0)
        intrinsics_metric = intrinsics_metric[:, 0].unsqueeze(0)
    condition, valid, _, returned_anchor_mask = (
        camera_condition_from_waymo_metric_target(
            camera_metric,
            intrinsics_metric,
            image_hw=image_hw,
            trajectory_anchor_to_world=trajectory_anchor_to_world.to(
                device=device, dtype=torch.float32
            ),
            previous_camera_to_world=(
                None
                if not torch.is_tensor(previous_camera_to_world)
                else previous_camera_to_world.to(device=device, dtype=torch.float32)
            ),
            anchor_mask=anchor_mask.to(device=device, dtype=torch.bool),
            normalize_camera=unwrap_ddp(scene_flow).normalize_camera,
        )
    )
    expected_anchor_mask = anchor_mask.to(
        device=returned_anchor_mask.device, dtype=torch.bool
    )
    if not torch.equal(returned_anchor_mask, expected_anchor_mask):
        raise RuntimeError("formal metric camera condition changed global anchor roles")
    return condition, valid


def build_camera_condition_from_sample(
    sample: dict[str, Any],
    *,
    device: torch.device,
    anchor_mask: torch.Tensor,
    scene_flow: nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image_hw = normalize_front_image_hw(sample.get("raw_image_size_hw"))
    return build_camera_condition_from_waymo_gt(
        sample.get("camera_to_world_corrected"),
        sample.get("intrinsics"),
        device=device,
        image_hw=image_hw,
        trajectory_anchor_to_world=sample.get("camera_trajectory_anchor_to_world_corrected"),
        previous_camera_to_world=sample.get("camera_previous_to_world_corrected"),
        anchor_mask=anchor_mask,
        scene_flow=scene_flow,
    )


TRAIN_PROGRESS_KEYS = (
    "lr",
    ("l", "loss"),
    ("l_flow", "loss_flow"),
    ("l_preserve", "loss_preserve"),
    ("l_boundary", "loss_boundary"),
    ("l_repa", "loss_repa"),
    ("l_identity", "loss_identity"),
    ("data_s", "data_wait_s"),
    ("train_s", "train_wall_s"),
    "optim_s",
    ("step_s", "step_wall_s"),
    ("data_frac", "data_wait_frac"),
    ("ips", "items_per_s_per_rank"),
)


def _format_train_progress_metrics(metrics: dict[str, float]) -> dict[str, str]:
    """Compact terminal progress; keep verbose metrics available for wandb."""
    out: dict[str, str] = {}
    for spec in TRAIN_PROGRESS_KEYS:
        display_key, metric_key = spec if isinstance(spec, tuple) else (spec, spec)
        if metric_key not in metrics:
            continue
        value = float(metrics[metric_key])
        if display_key == "lr":
            out[display_key] = f"{value:.2e}"
        elif display_key.endswith("_s") or display_key == "data_frac":
            out[display_key] = f"{value:.3f}"
        elif display_key == "ips":
            out[display_key] = f"{value:.2f}"
        else:
            out[display_key] = f"{value:.4f}"
    return out


def _format_train_progress_line(metrics: dict[str, float]) -> str:
    return " | ".join(
        f"{key}={value}" for key, value in _format_train_progress_metrics(metrics).items()
    )


def autocast_context(args, device: torch.device):
    if args.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def sample_uncond_drop_mask(
    batch_size: int,
    prob: float,
    *,
    device: torch.device,
    training: bool = True,
) -> torch.Tensor | None:
    if not bool(training) or float(prob) <= 0.0:
        return None
    return torch.rand(int(batch_size), device=device) < float(prob)


def formal_sky_generation_enabled(args) -> bool:
    """T1 editing keeps sky from GT render context, so do not train sky gen tokens."""
    del args
    return False


@torch.no_grad()
def materialize_ema_state_dict(scene_flow: nn.Module, ema: EMAModel) -> dict[str, torch.Tensor]:
    sf = unwrap_ddp(scene_flow)
    params = list(sf.parameters())
    ema.store(params)
    ema.copy_to(params)
    try:
        return {key: value.detach().cpu().clone() for key, value in sf.state_dict().items()}
    finally:
        ema.restore(params)


@torch.no_grad()
def sync_ema_shadow_from_model(scene_flow: nn.Module, ema: EMAModel) -> None:
    """Initialize EMAModel shadow params from the currently loaded model."""
    params = list(unwrap_ddp(scene_flow).parameters())
    if len(params) != len(ema.shadow_params):
        raise ValueError(
            f"EMA shadow param count {len(ema.shadow_params)} != model param count {len(params)}"
        )
    ema.shadow_params = [p.detach().clone() for p in params]


def split_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "scale_shift_table" in name
            or name.endswith("null_kv")
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_training_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> LambdaLR:
    decay_end_steps = int(args.decay_end_steps) if int(args.decay_end_steps) > 0 else int(args.max_steps)
    return build_rae_scheduler(
        optimizer,
        scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
        decay_end_steps=decay_end_steps,
        base_lr=args.lr,
        final_lr=args.final_lr,
        warmup_from_zero=args.warmup_from_zero,
    )


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def _load_state_dict_checked(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    path: str,
    label: str,
) -> tuple[int, int]:
    missing, unexpected = module.load_state_dict(_strip_module_prefix(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} from {path} is incompatible: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    return len(missing), len(unexpected)


def _scene_flow_prediction_type_from_module(scene_flow: nn.Module) -> str:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return str(getattr(cfg, "prediction_type", "x"))


def _checkpoint_prediction_type(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    cfg = payload.get("scene_flow_config")
    if isinstance(cfg, dict) and "prediction_type" in cfg:
        return str(cfg["prediction_type"])
    args = payload.get("args")
    if isinstance(args, dict) and "prediction_type" in args:
        return str(args["prediction_type"])
    return None


SCENE_FLOW_CONFIG_COMPAT_FIELDS = (
    "rope_layout_version",
    "rope_theta",
    "encoder_rope_theta",
    "ddt_rope_theta",
    "encoder_mrope_section",
    "ddt_mrope_section",
    "patch_grid",
    "out_channels",
    "sky_grid",
    "sky_representation_version",
    "sky_atlas_hw",
    "sky_token_dim",
    "rope_max_position",
    "camera_gen_dim",
    "camera_generation_representation",
    "camera_stats_version",
    "gauge_gen_dim",
    "scene_gauge_representation",
    "scene_gauge_stats_version",
    "camera_condition_representation",
    "mask_compositing_version",
    "asset_position_mode",
    "asset_condition_protocol",
    "factorized_asset_condition_version",
    "placement_mean",
    "placement_std",
    "sky_rope_temporal_offset",
    "camera_rope_spatial_mode",
    "sky_mask_head_version",
    "sky_mask_refine_scale",
    "sky_mask_refine_channels",
)


def build_scene_flow_from_checkpoint_config(
    checkpoint_path: str | Path,
    *,
    patch_grid: tuple[int, int],
    latent_dim: int,
    device: torch.device,
) -> WanSceneFlow:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("scene_flow_config"), dict):
        raise ValueError(
            f"{checkpoint_path} has no scene_flow_config. Formal training must construct the exact "
            "pretrained architecture from a versioned checkpoint."
        )
    config = dict(payload["scene_flow_config"])
    is_formal_checkpoint = bool(
        payload.get("formal_flow_domain_version") is not None
        or payload.get("formal_metric_gauge_contract") is not None
    )
    validate_metric_gauge_checkpoint_payload(
        payload,
        path=checkpoint_path,
        config=config,
        require_formal_contract=is_formal_checkpoint,
        expected_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=patch_grid,
    )
    if not is_formal_checkpoint:
        validate_pretrain_feature_stats_contract(
            payload.get(PRETRAIN_FEATURE_STATS_CONTRACT_KEY),
            path=checkpoint_path,
            expected_sequence_length=FORMAL_TOKENIZER_WINDOW_LEN,
            expected_patch_grid=patch_grid,
        )
    if tuple(config.get("patch_grid", ())) != tuple(patch_grid):
        raise ValueError(f"checkpoint patch_grid={config.get('patch_grid')} != cache patch_grid={patch_grid}")
    if int(config.get("out_channels", -1)) != int(latent_dim):
        raise ValueError(f"checkpoint out_channels={config.get('out_channels')} != --latent_dim={latent_dim}")
    return WanSceneFlow(**config).to(device)


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_config_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_normalize_config_value(v) for v in value)
    return value


def _config_values_match(current: Any, saved: Any) -> bool:
    current = _normalize_config_value(current)
    saved = _normalize_config_value(saved)
    if isinstance(current, tuple) and isinstance(saved, tuple):
        return len(current) == len(saved) and all(_config_values_match(c, s) for c, s in zip(current, saved))
    if isinstance(current, float) or isinstance(saved, float):
        try:
            return abs(float(current) - float(saved)) <= 1e-6
        except (TypeError, ValueError):
            return False
    return current == saved


def _validate_scene_flow_checkpoint_config(scene_flow: nn.Module, payload: Any, path: str | Path) -> None:
    _validate_scene_flow_prediction_type(scene_flow, payload, path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a versioned metric/gauge SceneFlow checkpoint")
    saved_cfg = payload.get("scene_flow_config")
    if not isinstance(saved_cfg, dict):
        raise ValueError(f"{path} is missing scene_flow_config; old checkpoints are rejected")
    require_formal_contract = bool(
        payload.get("formal_flow_domain_version") is not None
        or payload.get("formal_metric_gauge_contract") is not None
    )
    provenance, formal_contract = validate_metric_gauge_checkpoint_payload(
        payload,
        path=path,
        config=saved_cfg,
        require_formal_contract=require_formal_contract,
        expected_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=saved_cfg.get("patch_grid"),
    )
    current_provenance = getattr(unwrap_ddp(scene_flow), "_metric_gauge_provenance", None)
    if current_provenance is not None and dict(current_provenance) != provenance:
        raise ValueError(
            f"{path} metric_gauge_provenance does not match the active formal coordinate contract"
        )
    current_formal_contract = getattr(
        unwrap_ddp(scene_flow), "_formal_metric_gauge_contract", None
    )
    if current_formal_contract is not None and require_formal_contract:
        if formal_contract is None or dict(current_formal_contract) != formal_contract:
            raise ValueError(
                f"{path} formal_metric_gauge_contract does not match the active formal coordinate contract"
            )
    if saved_cfg.get("sky_representation_version") != "rgb_patch_teacher_anchor_v3":
        raise ValueError(
            f"{path} is not an rgb_patch_teacher_anchor_v3 sky checkpoint and cannot be resumed directly."
        )
    if "rope_layout_version" not in saved_cfg and "mrope_temporal_margin" in saved_cfg:
        raise ValueError(
            f"{path} was saved with the legacy global mrope_temporal_margin RoPE layout. "
            "The current SceneFlow model uses the fixed A3 layout "
            "(video/asset/camera shared video time, camera center, spherical sky coordinates near 15000); "
            "do not resume/warm-start across these incompatible position semantics."
        )
    current_cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    mismatches: list[str] = []
    for field in SCENE_FLOW_CONFIG_COMPAT_FIELDS:
        if field not in saved_cfg or not hasattr(current_cfg, field):
            continue
        current_value = getattr(current_cfg, field)
        saved_value = saved_cfg[field]
        if not _config_values_match(current_value, saved_value):
            mismatches.append(f"{field}: checkpoint={saved_value!r}, current={current_value!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            f"{path} SceneFlow config does not match the current model: {joined}. "
            "Do not resume/warm-start across incompatible RoPE/model geometry settings."
        )


def _validate_scene_flow_prediction_type(scene_flow: nn.Module, payload: Any, path: str | Path) -> None:
    current = _scene_flow_prediction_type_from_module(scene_flow)
    saved = _checkpoint_prediction_type(payload)
    if saved is None:
        if current == "v":
            raise ValueError(
                f"{path} does not record SceneFlow prediction_type. Refusing to load it into "
                "a velocity-prediction model because legacy checkpoints were x-prediction by default. "
                "Use --prediction_type x for that checkpoint or warm-start from a checkpoint saved with scene_flow_config."
            )
        return
    if saved != current:
        raise ValueError(
            f"{path} prediction_type={saved!r} does not match current model prediction_type={current!r}. "
            "Do not warm-start/resume across x-prediction and velocity-prediction checkpoints."
        )


def load_scene_flow_warm_start(
    scene_flow: nn.Module,
    pretrain_path: str | None,
    *,
    use_ema: bool = True,
    args: argparse.Namespace | None = None,
) -> dict[str, Any] | None:
    if not pretrain_path:
        return None

    sf = unwrap_ddp(scene_flow)
    payload = torch.load(pretrain_path, map_location="cpu")
    _validate_scene_flow_checkpoint_config(scene_flow, payload, pretrain_path)
    if args is None:
        raise ValueError("SceneFlow warm-start requires runtime args for flow-schedule validation")
    stats_contract = validate_pretrain_feature_stats_contract(
        payload.get(PRETRAIN_FEATURE_STATS_CONTRACT_KEY),
        path=pretrain_path,
        expected_feature_stats_sha256=getattr(args, "feature_stats_sha256", None),
        expected_sequence_length=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=getattr(unwrap_ddp(scene_flow).config, "patch_grid", None),
    )
    # The pretrain checkpoint owns the frozen metric/gauge coordinate system.
    # Keep the exact validated 12-field block on the formal model so every
    # later save inherits it verbatim rather than reconstructing provenance
    # from runtime arguments.
    sf._metric_gauge_provenance = copy.deepcopy(
        payload["metric_gauge_provenance"]
    )
    sf._pretrain_feature_stats_contract = copy.deepcopy(stats_contract)
    validate_checkpoint_flow_schedule(
        payload,
        args,
        pretrain_path,
        prediction_type=_scene_flow_prediction_type_from_module(scene_flow),
        t_eps=scene_flow_t_eps(scene_flow),
    )
    info: dict[str, Any] = {
        "path": pretrain_path,
        "step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "ema_used": False,
    }

    if use_ema:
        if isinstance(payload, dict) and "ema_scene_flow_state_dict" in payload:
            _load_state_dict_checked(
                sf,
                payload["ema_scene_flow_state_dict"],
                path=pretrain_path,
                label="EMA SceneFlow state_dict",
            )
            info["ema_used"] = True
            info["source"] = "ema_scene_flow_state_dict"
            return info

        if isinstance(payload, dict) and payload.get("is_ema_weights") and "scene_flow" in payload:
            _load_state_dict_checked(
                sf,
                payload["scene_flow"],
                path=pretrain_path,
                label="EMA SceneFlow weights-only state_dict",
            )
            info["ema_used"] = True
            info["source"] = "ema_weights_only"
            return info

        if isinstance(payload, dict) and "ema_scene_flow" in payload:
            if "scene_flow" not in payload:
                raise ValueError(f"{pretrain_path} has ema_scene_flow but no scene_flow buffers to initialize.")
            _load_state_dict_checked(
                sf,
                payload["scene_flow"],
                path=pretrain_path,
                label="raw SceneFlow state_dict",
            )
            ema = EMAModel(sf.parameters())
            ema.load_state_dict(payload["ema_scene_flow"])
            ema.copy_to(sf.parameters())
            info["ema_used"] = True
            info["source"] = "ema_scene_flow"
            return info

        raise ValueError(
            f"{pretrain_path} does not contain EMA SceneFlow weights. "
            "Use the full pretrain_step{N}.pt checkpoint, a new "
            "pretrain_step{N}_ema_weights_only.pt export, or pass "
            "--no_scene_flow_pretrain_ema to explicitly load raw weights."
        )

    if isinstance(payload, dict) and "scene_flow" in payload:
        state = payload["scene_flow"]
        info["source"] = "scene_flow"
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        info["source"] = "state_dict"
    else:
        state = payload
        info["source"] = "raw_state_dict"
    _load_state_dict_checked(sf, state, path=pretrain_path, label="SceneFlow state_dict")
    return info


def _infer_cache_patch_grid(dataset: WaymoFlowCacheDataset) -> tuple[int, int]:
    if len(dataset.entries) == 0:
        raise RuntimeError("Cannot infer patch grid from an empty cache dataset.")

    def _load_patch_grid(idx: int) -> tuple[int, int]:
        entry = dataset.entries[idx]
        cache_path = entry.get("cache_path")
        if cache_path is None:
            raise KeyError("Cache dataset entry is missing 'cache_path'.")
        if is_chunked_flow_cache(cache_path):
            summary = load_chunked_flow_cache_summary(cache_path)
            patch_grid = summary.get("patch_grid")
            if patch_grid is None or len(patch_grid) != 2:
                raise KeyError(f"Chunked cache payload {cache_path} is missing patch_grid=(H,W).")
            return (int(patch_grid[0]), int(patch_grid[1]))
        payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
        WaymoFlowCacheDataset._validate_v6_payload(
            payload,
            cache_path=Path(cache_path),
            entry=entry,
        )
        patch_grid = payload.get("meta", {}).get("patch_grid")
        if patch_grid is None or len(patch_grid) != 2:
            raise KeyError(f"Cache payload {cache_path} is missing meta.patch_grid=(H,W).")
        out = (int(patch_grid[0]), int(patch_grid[1]))
        if out[0] <= 0 or out[1] <= 0:
            raise ValueError(f"Invalid cache patch_grid {out} in {cache_path}.")
        return out

    return dataset._getitem_with_cache_read_retry(0, _load_patch_grid)


def split_train_val_entries(
    dataset: WaymoFlowCacheDataset,
    *,
    val_fraction: float,
    seed: int,
) -> WaymoFlowCacheDataset | None:
    entries = list(dataset.entries)
    if len(entries) < 2 or float(val_fraction) <= 0.0:
        return None
    val_count = int(round(len(entries) * float(val_fraction)))
    val_count = max(1, min(val_count, len(entries) - 1))
    indices = list(range(len(entries)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_indices = set(indices[:val_count])
    train_entries = [entry for idx, entry in enumerate(entries) if idx not in val_indices]
    val_entries = [entry for idx, entry in enumerate(entries) if idx in val_indices]

    dataset.entries = train_entries
    dataset._window_seed = int(seed)
    dataset._rng = random.Random(int(seed))
    dataset._rng_worker_seed = None
    val_dataset = copy.copy(dataset)
    val_dataset.entries = val_entries
    val_dataset._window_seed = int(seed) + 1
    val_dataset._rng = random.Random(int(seed) + 1)
    val_dataset._rng_worker_seed = None
    val_dataset.deterministic_windows = True
    return val_dataset


def _validate_item_patch_grid(
    asset_pass_result,
    assembler: FlowFeatureAssembler,
    cache_path: str | None = None,
) -> None:
    item_grid = tuple(int(v) for v in asset_pass_result.patch_grid)
    if item_grid != assembler.patch_grid:
        where = f" for {cache_path}" if cache_path else ""
        raise ValueError(
            f"Cache patch_grid{where} is {item_grid}, but assembler was initialized "
            f"with {assembler.patch_grid}. Use one training run per image geometry."
        )


# ---------------------------------------------------------------------- #
# CLI                                                                     #
# ---------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T1 SceneFlow training (Phase 9).")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Base DGGT checkpoint for tokenizer.")
    parser.add_argument(
        "--tokenizer_ckpt_path",
        type=str,
        default=None,
        help=(
            "Tokenizer checkpoint used for SceneFlow latents. It may be omitted only when "
            "--ckpt_path embeds a complete scene_tokenizer state; otherwise startup fails."
        ),
    )
    parser.add_argument("--feature_stats_path", type=str, default=str(DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH),
                        help=(
                            "Metric-gauge v4 latent/camera/gauge/placement stats contract; "
                            "defaults to the full-pass tokenizer-v2 v5 stats artifact. "
                            "The file must exactly match the "
                            "buffers stored in the warm-start/resume checkpoint; mismatches fail fast."
                        ))
    parser.add_argument(
        "--pullback_calibration_path",
        type=str,
        default=DEFAULT_FORMAL_PULLBACK_CALIBRATION,
        help=(
            "Strict checkpoint-bound tokenizer pullback artifact. Formal render/decode "
            "always executes its explicit identity boundary and rejects a hash, tokenizer, "
            "DGGT, window, or patch-grid mismatch."
        ),
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=1024,
        help=(
            "Tokenizer latent channels; must match pretrain warm-start and feature stats. "
            "SceneFlow DDT visual embedders consume z_t directly; legacy in_channels "
            "still records 3 * latent_dim + 3 packed control channels."
        ),
    )
    parser.add_argument("--scene_flow_pretrain_path", type=str, default=None,
                        help="Optional SceneFlow pretrain checkpoint for warm-start.")
    parser.add_argument("--asset_position_mode", choices=("localized", "canonical"), default="localized")
    parser.add_argument("--scene_flow_pretrain_ema", dest="scene_flow_pretrain_ema",
                        action="store_true", default=True,
                        help="Load EMA weights from --scene_flow_pretrain_path. Enabled by default.")
    parser.add_argument("--no_scene_flow_pretrain_ema", dest="scene_flow_pretrain_ema",
                        action="store_false",
                        help="Load raw scene_flow weights from --scene_flow_pretrain_path.")
    parser.add_argument("--cache_root", type=str, default=None,
                        help="Offline feature cache root (Phase 4.5 output). Mutually exclusive with --manifest_path.")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Merged Mode A/B JSONL manifest from tools/build_flow_train_manifest.py.")
    parser.add_argument("--mode_filter", type=str, default=None,
                        help="When using --manifest_path, restrict to comma-sep modes (e.g. 'mode_a,mode_b').")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument(
        "--caption_root",
        type=str,
        default="/data/disk2/lyy_dataset/waymo_processed_dggt/training_captions",
    )
    parser.add_argument("--val_manifest_path", type=str, default=None,
                        help="Optional independent validation manifest. Enables --val_caption_root.")
    parser.add_argument("--val_cache_root", type=str, default=None,
                        help="Optional independent validation cache root. Enables --val_caption_root.")
    parser.add_argument("--val_caption_root", type=str, default=None,
                        help="Caption root for an independent validation manifest/cache.")
    parser.add_argument(
        "--val_scene_gauge_sha256",
        type=str,
        default=None,
        help=(
            "Trusted scene-gauge table SHA-256 for an independent validation cache. "
            "Required with --val_manifest_path/--val_cache_root; internal holdout uses "
            "the training checkpoint provenance."
        ),
    )
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--text_encoder_path", type=str, default="/home/dancer/model/Qwen/Qwen3-0.6B")
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--no_text_condition", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Hold out this fraction of the training cache entries for validation.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dggt-flow")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=50)

    parser.add_argument("--sequence_length", type=int, default=10,
                        help="Fixed number of frames sampled from each cache clip.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-process cache items per micro-batch.")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    gradient_checkpointing_group = parser.add_mutually_exclusive_group()
    gradient_checkpointing_group.add_argument(
        "--gradient_checkpointing",
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable SceneFlow activation checkpointing to reduce training memory.",
    )
    gradient_checkpointing_group.add_argument(
        "--no_gradient_checkpointing",
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable SceneFlow activation checkpointing to avoid backward recomputation.",
    )
    gradient_checkpointing_group.add_argument(
        "--half_gradient_checkpointing",
        "--half-gradient-checkpointing",
        dest="half_gradient_checkpointing",
        action="store_true",
        help="Checkpoint alternating SceneFlow encoder and DDT blocks.",
    )
    gradient_checkpointing_group.add_argument(
        "--three_quarter_gradient_checkpointing",
        "--three-quarter-gradient-checkpointing",
        dest="three_quarter_gradient_checkpointing",
        action="store_true",
        help="Checkpoint three of every four SceneFlow encoder blocks and no DDT blocks.",
    )
    parser.set_defaults(
        gradient_checkpointing=True,
        half_gradient_checkpointing=False,
        three_quarter_gradient_checkpointing=False,
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=1,
                        help="DataLoader batches prefetched per worker. Keep low because each cache item is large.")
    parser.add_argument("--no_persistent_workers", action="store_true",
                        help="Disable persistent DataLoader workers.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable DataLoader pin_memory. Disabled by default because cache items are GB-scale.")
    parser.add_argument("--mp_sharing_strategy", type=str, default="file_system",
                        choices=("file_system", "file_descriptor"),
                        help="Torch multiprocessing tensor sharing strategy for DataLoader workers.")
    parser.add_argument("--no_mmap_plain_cache", action="store_true",
                        help="Disable mmap=True when reading uncompressed torch cache files.")
    parser.add_argument("--no_batch_scene_flow", action="store_true",
                        help="Process cache items in a micro-batch serially instead of batching WanSceneFlow.")
    parser.add_argument(
        "--full_asset_lut_cache",
        action="store_true",
        help=(
            "Compatibility flag. RAE-style SceneFlow always loads all cached "
            "asset LUT levels because asset conditioning is encoded before "
            "frame-level compression."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--vis_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--val_log_images", type=int, default=10)
    parser.add_argument("--val_sample_steps", type=int, default=50)
    parser.add_argument(
        "--val_sliding_window",
        type=int,
        default=10,
        help="Validation CFG sampling window. 0 disables sliding; use the training sequence length for long clips.",
    )
    parser.add_argument(
        "--val_sliding_stride",
        type=int,
        default=7,
        help="Validation CFG sampling stride. 0 defaults to a three-frame overlap; overlap is mandatory.",
    )
    parser.add_argument("--no_val_render_rgb", action="store_true",
                        help="Skip validation 3DGS RGB renders and log latent/mask diagnostics only.")
    parser.add_argument("--no_val_ema", action="store_true",
                        help="Disable EMA weights for validation. Default matches pretrain: validate with EMA.")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--final_lr", type=float, default=2e-5)
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=("linear", "cosine"))
    parser.add_argument("--decay_end_steps", type=int, default=0,
                        help="LR decay end step. 0 means --max_steps, matching step-based RAE training.")
    parser.add_argument("--warmup_from_zero", action="store_true",
                        help="RAEv2 t2i keeps warmup_from_zero=false by default.")
    parser.add_argument("--optimizer_type", type=str, default="gmuon", choices=("gmuon", "adamw"))
    parser.add_argument("--gmuon_momentum", type=float, default=0.95)
    parser.add_argument("--gmuon_nesterov", action="store_true", default=True)
    parser.add_argument("--no_gmuon_nesterov", dest="gmuon_nesterov", action="store_false")
    parser.add_argument("--gmuon_ns_coefficients_preset", type=str, default="POLAR_EXPRESS_COEFFICIENTS")
    parser.add_argument("--gmuon_ns_use_kernels", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument(
        "--shift",
        type=float,
        default=10.0,
        help="Manually specified FlowMatch / RAE time-distribution shift.",
    )

    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_preserve", type=float, default=1.0)
    parser.add_argument("--lambda_repa", type=float, default=0.0)
    parser.add_argument("--base_model_coeff", type=float, default=0.1)
    parser.add_argument("--lambda_boundary", type=float, default=0.25)
    parser.add_argument("--lambda_identity", type=float, default=1.0)
    parser.add_argument("--preserve_floor", type=float, default=0.2)
    parser.add_argument(
        "--lambda_rgb_render",
        type=float,
        default=0.1,
        help="Deployment-aligned RGB loss with generated depth and fixed input-DGGT camera.",
    )
    parser.add_argument(
        "--lambda_level_consistency",
        type=float,
        default=0.1,
        help="Four-level tokenizer-decoder consistency weight, evaluated on RGB render steps.",
    )
    parser.add_argument(
        "--lambda_head_consistency",
        type=float,
        default=0.1,
        help="Frozen depth/GS/dynamic-head consistency weight, evaluated on RGB render steps.",
    )
    parser.add_argument("--rgb_render_every", type=int, default=1)
    parser.add_argument("--rgb_render_start_step", type=int, default=5000)
    parser.add_argument("--rgb_render_warmup_steps", type=int, default=5000)
    parser.add_argument(
        "--rgb_render_sigma_power",
        type=float,
        default=2.0,
        help=(
            "Continuously attenuate RGB reconstruction at noisy timesteps with "
            "w(sigma)=(1-sigma)^power; 0 disables sigma weighting."
        ),
    )
    parser.add_argument(
        "--feedback_conf_weight_power",
        type=float,
        default=1.0,
        help=(
            "Teacher depth-confidence exponent for reconstruction feedback; "
            "0 disables confidence weighting."
        ),
    )
    parser.add_argument(
        "--feedback_conf_weight_floor",
        type=float,
        default=0.05,
        help="Lower clamp for teacher depth confidence before weighting.",
    )
    parser.add_argument("--rgb_render_max_samples", type=int, default=1)
    parser.add_argument("--rgb_render_max_frames", type=int, default=0)
    parser.add_argument("--rgb_render_stride", type=int, default=1)
    parser.add_argument("--rgb_render_lpips_weight", type=float, default=0.01)
    parser.add_argument("--rgb_render_lpips_net", type=str, default="alex")
    parser.add_argument(
        "--edit_domain_threshold",
        type=float,
        default=1e-4,
        help="Threshold soft source+destination coverage into the binary formal-edit flow domain.",
    )
    parser.add_argument(
        "--edit_domain_dilation",
        type=int,
        default=1,
        help="Patch-grid dilation radius for the binary formal-edit flow domain.",
    )
    parser.add_argument(
        "--no_sky_generation",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Formal T1 training no longer packs "
            "generated-sky tokens or computes sky flow loss; RGB rendering preserves "
            "the original input RGB exactly inside the GT sky mask."
        ),
    )
    parser.add_argument("--sky_grid_h", type=int, default=DEFAULT_SKY_GRID[0])
    parser.add_argument("--sky_grid_w", type=int, default=DEFAULT_SKY_GRID[1])
    parser.add_argument(
        "--lambda_sky_flow",
        type=float,
        default=0.1,
        help="Deprecated no-op in formal T1 training; sky flow loss is pretrain-only.",
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--asset_control_guidance_scale", type=float, default=1.0)
    parser.add_argument("--uncond_drop_prob", type=float, default=0.1)
    parser.add_argument("--val_guidance_scales", type=str, default="")
    parser.add_argument("--weighting_scheme", type=str, default="waver")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--loss_weighting_scheme", type=str, default="none")
    parser.add_argument(
        "--prediction_type",
        type=str,
        choices=("v", "x"),
        default="x",
        help=(
            "SceneFlow model output parameterization. Default 'x' follows RAEv2 T2I "
            "by predicting the clean latent and converting it to RF velocity for loss/sampling."
        ),
    )
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    return parser


# ---------------------------------------------------------------------- #
# Model setup                                                             #
# ---------------------------------------------------------------------- #
def _load_tokenizer(ckpt_path: str, device: torch.device) -> nn.Module:
    # Formal cached training only needs the tokenizer.  Loading it directly is
    # both cheaper than constructing full VGGT and, crucially, lets us require
    # an exact tokenizer match instead of silently retaining random weights.
    tokenizer = JointSceneTokenizer().to(device=device, dtype=torch.float32)
    load_scene_tokenizer_checkpoint_strict(tokenizer, ckpt_path)
    tokenizer.eval()
    return tokenizer


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def scene_flow_prediction_type(scene_flow: nn.Module) -> str:
    return _scene_flow_prediction_type_from_module(scene_flow)


def model_prediction_to_velocity(scene_flow: nn.Module, prediction: torch.Tensor, target) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        # Match RAEv2 Transport.convert_model_pred: x-pred is converted to
        # velocity with the same t_eps clamp used to build the RF target.
        return (target.z_t - prediction) / target.sigmas4.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).clamp_min(float(getattr(target, "t_eps", scene_flow_t_eps(scene_flow))))
    return prediction


def scene_flow_t_eps(scene_flow: nn.Module) -> float:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return float(getattr(cfg, "t_eps", 0.05))


def model_prediction_to_clean(scene_flow: nn.Module, prediction: torch.Tensor, target) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        return prediction
    # RAEv2 trains velocity against (z_t - z_clean) / max(sigma, t_eps), so
    # recovering the clean endpoint must invert that same clamped denominator.
    sigma_safe = target.sigmas4.to(device=prediction.device, dtype=prediction.dtype).clamp_min(
        float(getattr(target, "t_eps", scene_flow_t_eps(scene_flow)))
    )
    return target.z_t - sigma_safe * prediction


def sampler_prediction_to_velocity(scene_flow: nn.Module, prediction: torch.Tensor, z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        while sigma.ndim < z.ndim:
            sigma = sigma.view(*sigma.shape, 1)
        return (z - prediction) / sigma.to(device=z.device, dtype=z.dtype).clamp_min(scene_flow_t_eps(scene_flow))
    return prediction


def build_formal_edit_domains(
    bundle,
    args,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft semantics, binary edit/keep domains, and an inner boundary ring."""
    patch_grid = getattr(bundle, "patch_grid", getattr(args, "patch_grid", (25, 37)))
    threshold = float(getattr(args, "edit_domain_threshold", 1e-4))
    dilation = int(getattr(args, "edit_domain_dilation", 1))
    soft_edit = (bundle.M_source.float() + bundle.M_dest.float()).clamp(0.0, 1.0)
    core = build_hard_edit_domain(
        bundle.M_source,
        bundle.M_dest,
        patch_grid,
        threshold=threshold,
        dilation_radius=0,
    )
    edit_domain = build_hard_edit_domain(
        bundle.M_source,
        bundle.M_dest,
        patch_grid,
        threshold=threshold,
        dilation_radius=dilation,
    )
    # The boundary target must be inside the generated domain.  Using the old
    # outer ring without first expanding the flow domain made boundary and
    # preserve losses request different clean endpoints for the same token.
    if dilation > 0:
        boundary = boundary_mask_from_edit_mask(core, patch_grid, radius=dilation) * edit_domain
    else:
        boundary = torch.zeros_like(edit_domain)
    edit_domain = edit_domain.to(device=device, dtype=dtype)
    keep_domain = 1.0 - edit_domain
    return (
        soft_edit.to(device=device, dtype=dtype),
        edit_domain,
        keep_domain,
        boundary.to(device=device, dtype=dtype),
    )


def normalize_asset_latents(scene_flow_root: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    cfg = getattr(scene_flow_root, "config", None)
    latent_dim = int(getattr(cfg, "out_channels", tokens.shape[-1]))
    if torch.is_tensor(tokens) and tokens.ndim >= 3 and int(tokens.shape[-1]) == latent_dim:
        return scene_flow_root.normalize(tokens.float()).to(dtype=tokens.dtype)
    return tokens


def estimate_sparse_asset_token_count(
    scene_flow_root: nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor | None,
) -> float:
    if not torch.is_tensor(tokens):
        return 0.0
    cfg = getattr(scene_flow_root, "config", None)
    max_patch = int(getattr(cfg, "max_asset_patch_tokens_per_asset_frame", 32))
    max_total = int(getattr(cfg, "max_asset_tokens", 4096))
    if tokens.ndim == 5:
        if mask is None:
            valid = torch.ones(tokens.shape[:-1], device=tokens.device, dtype=torch.bool)
        else:
            valid = mask.to(device=tokens.device, dtype=torch.bool)
        counts = valid.sum(dim=-1)
        per_asset_frame = torch.where(
            counts > 0,
            counts.clamp_max(max_patch) + 1,
            torch.zeros_like(counts),
        )
        return float(per_asset_frame.sum(dim=(1, 2)).clamp_max(max_total).float().mean().item())
    if tokens.ndim == 3:
        if mask is None:
            return float(tokens.shape[1])
        return float(mask.to(device=tokens.device, dtype=torch.bool).sum(dim=1).float().mean().item())
    return 0.0


def estimate_control_token_count(
    scene_flow_root: nn.Module,
    M_source: torch.Tensor,
    M_dest: torch.Tensor,
    patch_grid: tuple[int, int],
) -> float:
    cfg = getattr(scene_flow_root, "config", None)
    max_per_frame = int(getattr(cfg, "max_control_tokens_per_frame", 128))
    max_total = int(getattr(cfg, "max_control_tokens", 1024))
    M_edit = (M_source.float() + M_dest.float()).clamp(0.0, 1.0)
    b, s, p, _ = M_edit.shape
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    grid = M_edit.reshape(b * s, gh, gw, 1).permute(0, 3, 1, 2)
    support = torch.nn.functional.max_pool2d(grid, kernel_size=5, stride=1, padding=2).gt(0.0)
    counts = support.reshape(b, s, p).sum(dim=-1)
    return float(counts.clamp_max(max_per_frame).sum(dim=1).clamp_max(max_total).float().mean().item())


def _move_v6_fast_path_inputs(
    item: dict[str, Any],
    mode_kind: str,
    device: torch.device,
) -> tuple[dict[str, Any] | None, list[torch.Tensor]]:
    schema_version = int(item.get("cache_schema_version", 0))
    if schema_version not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"Training item {item.get('cache_path', '<unknown>')} has "
            f"cache_schema_version={schema_version}; supported versions are "
            f"{SUPPORTED_CACHE_SCHEMA_VERSIONS}."
        )

    if mode_kind not in ("mode_a", "mode_b"):
        raise RuntimeError(
            f"Training item {item.get('cache_path', '<unknown>')} has "
            f"invalid mode_kind={mode_kind!r}."
        )

    phase1_localized_lite = item.get("phase1_localized")
    if mode_kind == "mode_a":
        if phase1_localized_lite is None:
            raise RuntimeError(
                f"Mode-A cache item {item.get('cache_path', '<unknown>')} missing phase1_localized."
            )
        phase1_localized_lite = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in phase1_localized_lite.items()
        }
    else:
        if phase1_localized_lite is not None:
            raise RuntimeError(
                f"Mode-B v6 item {item.get('cache_path', '<unknown>')} unexpectedly has phase1_localized."
            )
        phase1_localized_lite = None

    splatted_tok_low_cached = item.get("splatted_tok_low_cached")
    if splatted_tok_low_cached is None:
        raise RuntimeError(
            f"v6 item {item.get('cache_path', '<unknown>')} missing splatted_tok_low_cached."
        )
    return phase1_localized_lite, [t.to(device) for t in splatted_tok_low_cached]


def _flatten_fast_asset_kv(asset_pass_result, flow_inputs: dict[str, Any], device: torch.device) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    coverage = flow_inputs.get("phase1_coverage")
    if torch.is_tensor(coverage):
        coverage = coverage.to(device=device, dtype=torch.bool)
    phase4_slots = {int(v) for v in flow_inputs.get("phase4_slots", [])}
    for obj_key in sorted(int(k) for k in asset_pass_result.F_g_lut_asset.keys()):
        levels = asset_pass_result.F_g_lut_asset[obj_key]
        if not levels:
            continue
        lvl = levels[-1].to(device)
        patch_valid = require_asset_patch_valid_mask(
            asset_pass_result,
            obj_key,
            expected_shape=(int(lvl.shape[0]), int(lvl.shape[1]), int(lvl.shape[2])),
            device=device,
        )
        if torch.is_tensor(coverage) and obj_key < int(coverage.shape[0]):
            cov = coverage[obj_key].to(device=device, dtype=lvl.dtype)
            if cov.ndim != 1 or int(cov.numel()) != int(lvl.shape[1]):
                raise ValueError(
                    f"Asset {obj_key} coverage must be [S]={int(lvl.shape[1])}, "
                    f"got {tuple(cov.shape)} for LUT {tuple(lvl.shape)}."
                )
            patch_valid = patch_valid & cov.to(dtype=torch.bool).reshape(1, -1, 1)
        if phase4_slots and obj_key not in phase4_slots:
            patch_valid = torch.zeros_like(patch_valid)
        lvl = lvl * patch_valid.unsqueeze(-1).to(dtype=lvl.dtype)
        B, S, P, C = lvl.shape
        chunks.append(lvl.reshape(B, S * P, C))
    if not chunks:
        return torch.zeros((1, 0, 3072), dtype=torch.float32, device=device)
    return torch.cat(chunks, dim=1)


def _encode_fast_asset_conditions(
    asset_pass_result,
    flow_inputs: dict[str, Any],
    *,
    scene_tokenizer: nn.Module,
    patch_grid: tuple[int, int],
    reference: torch.Tensor,
    device: torch.device,
    mode_kind: str = "mode_a",
    tokenizer_window_len: int = FORMAL_TOKENIZER_WINDOW_LEN,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode each asset's 4-level LUT through the tokenizer.

    Returns `[B,K,S,P,C_latent]` plus a `[B,K,S,P]` valid mask. K is capped at 5.
    """
    B, S, P, C = reference.shape
    out = reference.new_zeros((B, 5, S, P, C))
    mask = torch.zeros((B, 5, S, P), device=device, dtype=torch.bool)
    if str(mode_kind) == "mode_b":
        return out, mask

    coverage = flow_inputs.get("phase1_coverage")
    phase4_slots = flow_inputs.get("phase4_slots")
    out, mask = build_asset_condition_slots(
        asset_pass_result,
        phase1_coverage=coverage if torch.is_tensor(coverage) else None,
        phase4_slots=phase4_slots,
        scene_tokenizer=scene_tokenizer,
        patch_grid=patch_grid,
        reference=reference,
        max_assets=5,
        expected_num_levels=4,
        tokenizer_window_len=int(tokenizer_window_len),
    )
    return out, mask


def _split_nplc_levels_for_train(x: torch.Tensor) -> list[torch.Tensor]:
    return [x[:, :, level, :].unsqueeze(0).contiguous() for level in range(int(x.shape[2]))]


def _dequantize_nplc_levels_on_device(
    payload: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    q = QuantizedTokens(
        data=payload["data"].to(device, non_blocking=True),
        scale=payload["scale"].to(device, non_blocking=True),
        layout=str(payload.get("layout", "NPLC")),
    )
    return _split_nplc_levels_for_train(dequantize_tokens(q, dtype=dtype))


def _dequantize_stacked_nplc_levels_on_device(
    payloads: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    data = torch.stack([p["data"] for p in payloads], dim=0)
    scale = torch.stack([p["scale"] for p in payloads], dim=0)
    b, s, p_count, levels, channels = data.shape
    q = QuantizedTokens(
        data=data.reshape(b * s, p_count, levels, channels).to(device, non_blocking=True),
        scale=scale.reshape(b * s, levels).to(device, non_blocking=True),
        layout=str(payloads[0].get("layout", "NPLC")),
    )
    x = dequantize_tokens(q, dtype=dtype).reshape(b, s, p_count, levels, channels)
    return [x[:, :, :, level, :].contiguous() for level in range(int(levels))]


def _build_sky_tokens_from_training_source(
    source: dict[str, Any] | None,
    args,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if args is None or not formal_sky_generation_enabled(args) or source is None:
        return None, None
    images = source.get("images", source.get("images_clean"))
    masks = source.get("masks", source.get("sky_mask"))
    if not torch.is_tensor(images) or not torch.is_tensor(masks):
        return None, None
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if masks.ndim == 4:
        masks = masks.unsqueeze(0)
    sky_h, sky_w = sky_grid_shape(args)
    return build_sky_tokens_from_images(
        images,
        masks,
        grid_h=sky_h,
        grid_w=sky_w,
    )


def _build_sky_tokens_from_item(
    item: dict[str, Any],
    args,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    source = item.get("sky_training")
    if source is None:
        source = item.get("sample")
    return _build_sky_tokens_from_training_source(source, args, device)


def _attach_sky_tokens_to_bundle(
    bundle: Any,
    item: dict[str, Any],
    args,
    device: torch.device,
) -> Any:
    del item, args, device
    # Formal T1 sampling and offline inference do not provide generated-sky
    # state to SceneFlow. Keep training on the same conditioning surface; pretrain
    # owns the generated-sky branch.
    bundle.sky_gen_clean = None
    bundle.sky_gen_attention_mask = None
    return bundle


def _frame_ids_from_sources(
    *,
    seq_len: int,
    device: torch.device,
    sources: tuple[Any, ...],
) -> torch.Tensor:
    for raw in sources:
        if raw is None:
            continue
        if torch.is_tensor(raw):
            ids = raw.detach().to(device=device, dtype=torch.long)
        else:
            ids = torch.as_tensor(raw, device=device, dtype=torch.long)
        ids = ids.reshape(-1)
        if int(ids.numel()) == int(seq_len):
            return ids.view(1, int(seq_len)).contiguous()
    return torch.arange(int(seq_len), device=device, dtype=torch.long).view(1, int(seq_len))


def _frame_ids_from_item(
    item: dict[str, Any],
    *,
    seq_len: int,
    device: torch.device,
    flow_inputs: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
) -> torch.Tensor:
    mode_b = item.get("mode_b")
    return _frame_ids_from_sources(
        seq_len=seq_len,
        device=device,
        sources=(
            item.get("subset_frames"),
            None if flow_inputs is None else flow_inputs.get("subset_frames"),
            None if not isinstance(mode_b, dict) else mode_b.get("subset_frames"),
            None if sample is None else sample.get("frame_ids"),
            None if sample is None else sample.get("frame_indices"),
        ),
    )


def _formal_rgb_context(
    item: dict[str, Any],
    *,
    device: torch.device,
    strict: bool,
) -> dict[str, torch.Tensor | int] | None:
    """Load only RGB targets/GT sky and the fixed input-DGGT camera.

    Depth is intentionally absent: the primary RGB path must decode depth from
    the SceneFlow-generated video tokens.
    """
    source = item.get("rgb_training")
    if not isinstance(source, dict):
        sample = item.get("sample")
        source = sample if isinstance(sample, dict) else None
    predictions = item.get("predictions")
    predictions = predictions if isinstance(predictions, dict) else {}
    where = str(item.get("cache_path", "<unknown>"))
    if not isinstance(source, dict):
        if strict:
            raise RuntimeError(f"{where} has no RGB training payload.")
        return None
    images = source.get("images", source.get("images_clean"))
    if not torch.is_tensor(images):
        if strict:
            raise RuntimeError(f"{where} RGB payload is missing images.")
        return None
    images = images.to(device=device, dtype=torch.float32)
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5 or int(images.shape[2]) != 3:
        raise ValueError(f"{where} RGB images must be [B,S,3,H,W], got {images.shape}")
    masks = source.get("masks", source.get("sky_mask"))
    if torch.is_tensor(masks):
        masks = masks.to(device=device, dtype=torch.float32)
        if masks.ndim == 4:
            masks = masks.unsqueeze(0)
    else:
        masks = None
    timestamps = source.get("timestamps", item.get("subset_frames"))
    if torch.is_tensor(timestamps):
        timestamps = timestamps.to(device=device, dtype=torch.float32)
    else:
        timestamps = torch.arange(int(images.shape[1]), device=device, dtype=torch.float32)
    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)
    pose = predictions.get("pose_enc")
    if not torch.is_tensor(pose):
        if strict:
            raise RuntimeError(
                f"{where} is missing frozen input-DGGT pose_enc required by formal RGB loss."
            )
        return None
    pose = pose.to(device=device, dtype=torch.float32)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    if pose.ndim != 3 or int(pose.shape[-1]) != 9:
        raise ValueError(f"{where} DGGT pose_enc must be [B,S,9], got {pose.shape}")
    seq_len = int(images.shape[1])
    if int(pose.shape[1]) < seq_len:
        raise ValueError(f"{where} pose has {pose.shape[1]} frames, RGB has {seq_len}")
    return {
        "rgb_render_images": images.contiguous(),
        "rgb_render_masks": None if masks is None else masks[:, :seq_len].contiguous(),
        "rgb_render_timestamps": timestamps[:, :seq_len].contiguous(),
        "rgb_render_pose_enc_dggt": pose[:, :seq_len].contiguous(),
        "rgb_render_patch_start_idx": int(predictions.get("patch_start_idx", 5)),
    }


def _attach_rgb_context(bundle: Any, context: dict[str, Any] | None) -> Any:
    if context is not None:
        for key, value in context.items():
            setattr(bundle, key, value)
    return bundle


def _merge_rgb_contexts(bundles: list[Any]) -> dict[str, Any] | None:
    if not bundles or not all(torch.is_tensor(getattr(b, "rgb_render_images", None)) for b in bundles):
        return None
    masks = [getattr(b, "rgb_render_masks", None) for b in bundles]
    return {
        "rgb_render_images": torch.cat([b.rgb_render_images for b in bundles], dim=0),
        "rgb_render_masks": torch.cat(masks, dim=0) if all(torch.is_tensor(m) for m in masks) else None,
        "rgb_render_timestamps": torch.cat([b.rgb_render_timestamps for b in bundles], dim=0),
        "rgb_render_pose_enc_dggt": torch.cat([b.rgb_render_pose_enc_dggt for b in bundles], dim=0),
        "rgb_render_patch_start_idx": int(bundles[0].rgb_render_patch_start_idx),
    }


def build_cached_flow_bundle(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    device: torch.device,
    args=None,
    *,
    scene_flow: nn.Module,
    include_rgb_render_context: bool = False,
) -> Any:
    flow_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in item["flow_inputs_cached"].items()
    }
    predictions_raw = item["predictions"]
    cache_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if isinstance(predictions_raw.get("image_tokens_quantized"), dict):
        F_g_lut_scene = _dequantize_nplc_levels_on_device(
            predictions_raw["image_tokens_quantized"],
            device=device,
            dtype=cache_dtype,
        )
    else:
        predictions = _move_predictions(predictions_raw, device)
        F_g_lut_scene = assembler._select_lut_scene(predictions)

    splatted_tok_low_quantized = item.get("splatted_tok_low_quantized")
    if isinstance(splatted_tok_low_quantized, dict):
        splatted_tok_low = _dequantize_nplc_levels_on_device(
            splatted_tok_low_quantized,
            device=device,
            dtype=F_g_lut_scene[0].dtype,
        )
    else:
        splatted_tok_low_cached = item.get("splatted_tok_low_cached")
        if splatted_tok_low_cached is None:
            raise RuntimeError(
                f"Fast cache item {item.get('cache_path', '<unknown>')} missing splatted_tok_low_cached."
            )
        splatted_tok_low = [t.to(device=device, dtype=F_g_lut_scene[0].dtype) for t in splatted_tok_low_cached]

    if assembler.scene_tokenizer is None:
        raise RuntimeError("FlowFeatureAssembler needs scene_tokenizer for cached SceneFlow inputs.")
    z_clean = encode_tokenizer_windowed(
        assembler.scene_tokenizer,
        F_g_lut_scene,
        patch_grid=assembler.patch_grid,
        window_len=FORMAL_TOKENIZER_WINDOW_LEN,
    )
    z_splat = encode_tokenizer_windowed(
        assembler.scene_tokenizer,
        splatted_tok_low,
        patch_grid=assembler.patch_grid,
        window_len=FORMAL_TOKENIZER_WINDOW_LEN,
    )

    M_preserve = flow_inputs["M_preserve"].to(device=device, dtype=torch.float32)
    M_source = flow_inputs["M_source"].to(device=device, dtype=torch.float32)
    M_dest = flow_inputs["M_dest"].to(device=device, dtype=torch.float32)
    scaffold_pooled = flow_inputs["scaffold_pooled"].to(device=device, dtype=torch.float32)
    # Keep the cached fast path inside ScaffoldPacker.forward so a DDP-wrapped
    # packer installs and executes its gradient-reduction hooks.
    scaffold_tok = assembler.scaffold_packer(scaffold_pooled, already_pooled=True)

    asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
    mode_kind = str(item.get("mode_kind", flow_inputs.get("mode_kind", "mode_a")))
    F_asset_tokens, asset_mask = _encode_fast_asset_conditions(
        asset_pass_result,
        flow_inputs,
        scene_tokenizer=assembler.scene_tokenizer,
        patch_grid=assembler.patch_grid,
        reference=z_clean,
        device=device,
        mode_kind=mode_kind,
        tokenizer_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
    )
    frame_ids = _frame_ids_from_item(
        item,
        seq_len=int(z_clean.shape[1]),
        device=device,
        flow_inputs=flow_inputs,
    )
    camera_gt = item.get("camera_gt") or {}
    camera_condition_tokens, camera_attention_mask = build_camera_condition_from_waymo_gt(
        camera_gt.get("camera_to_world_corrected"),
        camera_gt.get("intrinsics"),
        device=device,
        image_hw=camera_gt.get("raw_image_size_hw"),
        trajectory_anchor_to_world=camera_gt.get("trajectory_anchor_to_world"),
        previous_camera_to_world=camera_gt.get("previous_camera_to_world"),
        anchor_mask=frame_ids.eq(0),
        scene_flow=scene_flow,
    )
    if camera_condition_tokens is None:
        raise RuntimeError(
            f"Formal SceneFlow cache item {item.get('cache_path', '<unknown>')} "
            "is missing Waymo camera_gt; camera conditioning must use Waymo camera parameters."
        )
    bundle = SimpleNamespace(
        z_clean=z_clean,
        z_splat=z_splat,
        scaffold_tok=scaffold_tok,
        M_preserve=M_preserve,
        M_source=M_source,
        M_dest=M_dest,
        F_asset_tokens=F_asset_tokens,
        encoder_attention_mask=asset_mask,
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        asset_condition_kind=_asset_condition_kind_from_item(item, mode_kind),
        phase4_slots=list(flow_inputs.get("phase4_slots", [])),
        captions=[str(item.get("caption", ""))],
        patch_grid=assembler.patch_grid,
        patch_start_idx=assembler.patch_start_idx,
        splatted_tok_low=splatted_tok_low,
        F_g_lut_scene=F_g_lut_scene,
        frame_ids=frame_ids,
    )
    bundle = _attach_sky_tokens_to_bundle(bundle, item, args, device)
    if include_rgb_render_context:
        bundle = _attach_rgb_context(bundle, _formal_rgb_context(item, device=device, strict=True))
    return bundle


def build_cached_flow_batch_bundle(
    items: list[dict[str, Any]],
    assembler: FlowFeatureAssembler,
    device: torch.device,
    args=None,
    *,
    scene_flow: nn.Module,
    include_rgb_render_context: bool = False,
) -> tuple[Any, list[int], float]:
    if len(items) == 0:
        raise ValueError("Cannot build an empty cached flow batch.")
    if not all(item.get("flow_inputs_cached") is not None for item in items):
        raise ValueError("build_cached_flow_batch_bundle requires fast cache items.")

    cache_dtype = torch.float16 if device.type == "cuda" else torch.float32
    prediction_payloads = [item["predictions"]["image_tokens_quantized"] for item in items]
    splat_payloads = [item["splatted_tok_low_quantized"] for item in items]
    F_g_lut_scene = _dequantize_stacked_nplc_levels_on_device(
        prediction_payloads,
        device=device,
        dtype=cache_dtype,
    )
    splatted_tok_low = _dequantize_stacked_nplc_levels_on_device(
        splat_payloads,
        device=device,
        dtype=cache_dtype,
    )

    if assembler.scene_tokenizer is None:
        raise RuntimeError("FlowFeatureAssembler needs scene_tokenizer for cached SceneFlow inputs.")
    z_clean = encode_tokenizer_windowed(
        assembler.scene_tokenizer,
        F_g_lut_scene,
        patch_grid=assembler.patch_grid,
        window_len=FORMAL_TOKENIZER_WINDOW_LEN,
    )
    z_splat = encode_tokenizer_windowed(
        assembler.scene_tokenizer,
        splatted_tok_low,
        patch_grid=assembler.patch_grid,
        window_len=FORMAL_TOKENIZER_WINDOW_LEN,
    )

    flow_inputs = {
        key: torch.cat(
            [item["flow_inputs_cached"][key].to(device, non_blocking=True) for item in items],
            dim=0,
        )
        for key in ("M_preserve", "M_source", "M_dest", "scaffold_pooled")
    }
    M_preserve = flow_inputs["M_preserve"].to(device=device, dtype=torch.float32)
    M_source = flow_inputs["M_source"].to(device=device, dtype=torch.float32)
    M_dest = flow_inputs["M_dest"].to(device=device, dtype=torch.float32)
    # Do not unwrap here: the packer is independently DDP-wrapped in formal
    # training and its forward must run through that wrapper on every rank.
    scaffold_tok = assembler.scaffold_packer(
        flow_inputs["scaffold_pooled"].to(device=device, dtype=torch.float32),
        already_pooled=True,
    )

    asset_bundles: list[Any] = []
    num_objects = 0.0
    for item in items:
        mode_kind = str(item.get("mode_kind", "mode_a"))
        item_flow_inputs = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in item["flow_inputs_cached"].items()
        }
        asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
        F_asset_tokens_i, asset_mask_i = _encode_fast_asset_conditions(
            asset_pass_result,
            item_flow_inputs,
            scene_tokenizer=assembler.scene_tokenizer,
            patch_grid=assembler.patch_grid,
            reference=z_clean[len(asset_bundles) : len(asset_bundles) + 1],
            device=device,
            mode_kind=mode_kind,
            tokenizer_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        )
        asset_bundles.append(
            SimpleNamespace(
                F_asset_tokens=F_asset_tokens_i,
                encoder_attention_mask=asset_mask_i,
                asset_condition_kind=_asset_condition_kind_from_item(item, mode_kind),
            )
        )
        num_objects += float(len(item_flow_inputs.get("phase4_slots", [])))
    F_asset_tokens, asset_mask = _pad_asset_tokens_for_batch(asset_bundles)
    asset_lengths = [int(bundle.F_asset_tokens.shape[1]) for bundle in asset_bundles]
    frame_id_rows = [
        _frame_ids_from_item(
            item,
            seq_len=int(z_clean.shape[1]),
            device=device,
            flow_inputs=item.get("flow_inputs_cached"),
        )
        for item in items
    ]
    camera_token_rows: list[torch.Tensor] = []
    camera_mask_rows: list[torch.Tensor] = []
    for item, item_frame_ids in zip(items, frame_id_rows):
        camera_gt = item.get("camera_gt") or {}
        camera_tokens_i, camera_mask_i = build_camera_condition_from_waymo_gt(
            camera_gt.get("camera_to_world_corrected"),
            camera_gt.get("intrinsics"),
            device=device,
            image_hw=camera_gt.get("raw_image_size_hw"),
            trajectory_anchor_to_world=camera_gt.get("trajectory_anchor_to_world"),
            previous_camera_to_world=camera_gt.get("previous_camera_to_world"),
            anchor_mask=item_frame_ids.eq(0),
            scene_flow=scene_flow,
        )
        if camera_tokens_i is None:
            raise RuntimeError(
                f"Formal SceneFlow cache item {item.get('cache_path', '<unknown>')} "
                "is missing Waymo camera_gt; camera conditioning must use Waymo camera parameters."
            )
        if camera_tokens_i is not None:
            camera_token_rows.append(camera_tokens_i)
            camera_mask_rows.append(
                camera_mask_i
                if camera_mask_i is not None
                else torch.ones(camera_tokens_i.shape[:2], device=device, dtype=torch.bool)
            )
    if len(camera_token_rows) == len(items):
        camera_condition_tokens = torch.cat(camera_token_rows, dim=0)
        camera_attention_mask = torch.cat(camera_mask_rows, dim=0)
    else:
        camera_condition_tokens, camera_attention_mask = None, None
    # Keep formal T1 train conditioning identical to validation/offline sampling:
    # no generated-sky tokens are packed into the SceneFlow sequence.
    sky_gen_clean = None
    sky_gen_attention_mask = None
    frame_ids = torch.cat(frame_id_rows, dim=0)

    merged = SimpleNamespace(
        z_clean=z_clean,
        z_splat=z_splat,
        scaffold_tok=scaffold_tok,
        M_preserve=M_preserve,
        M_source=M_source,
        M_dest=M_dest,
        F_asset_tokens=F_asset_tokens,
        encoder_attention_mask=asset_mask,
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        sky_gen_clean=sky_gen_clean,
        sky_gen_attention_mask=sky_gen_attention_mask,
        asset_condition_kind=[
            _asset_condition_kind_from_item(
                item,
                str(item.get("mode_kind", "mode_a")),
            )
            for item in items
        ],
        phase4_slots=[],
        captions=[str(item.get("caption", "")) for item in items],
        patch_grid=assembler.patch_grid,
        patch_start_idx=assembler.patch_start_idx,
        splatted_tok_low=splatted_tok_low,
        F_g_lut_scene=F_g_lut_scene,
        frame_ids=frame_ids,
    )
    if include_rgb_render_context:
        contexts = []
        for item in items:
            holder = SimpleNamespace()
            _attach_rgb_context(holder, _formal_rgb_context(item, device=device, strict=True))
            contexts.append(holder)
        merged = _attach_rgb_context(merged, _merge_rgb_contexts(contexts))
    return merged, asset_lengths, num_objects / float(len(items))


# ---------------------------------------------------------------------- #
# Train step                                                              #
# ---------------------------------------------------------------------- #
def build_flow_bundle(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    device: torch.device,
    args=None,
    *,
    scene_flow: nn.Module,
    include_rgb_render_context: bool = False,
) -> Any:
    if item.get("flow_inputs_cached") is not None:
        return build_cached_flow_bundle(
            item,
            assembler,
            device,
            args=args,
            scene_flow=scene_flow,
            include_rgb_render_context=include_rgb_render_context,
        )

    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["sample"].items()}
    predictions = _move_predictions(item["predictions"], device)
    asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
    _validate_item_patch_grid(asset_pass_result, assembler, item.get("cache_path"))
    cameras_dggt = {k: v.to(device) for k, v in item["cameras_dggt"].items()}
    mode_kind = str(item.get("mode_kind", sample.get("mode_kind", "mode_a")))
    mode_b_payload = item.get("mode_b")
    if mode_b_payload is not None:
        mode_b_payload = _move_mode_b(mode_b_payload, device)

    phase1_localized_lite, splatted_tok_low_cached = _move_v6_fast_path_inputs(
        item, mode_kind, device
    )

    bundle = assembler(
        sample=sample,
        predictions=predictions,
        asset_pass_result=asset_pass_result,
        cameras_dggt=cameras_dggt,
        object_slots_spec="all",
        device=device,
        mode_kind=mode_kind,
        mode_b=mode_b_payload,
        phase1_localized_lite=phase1_localized_lite,
        splatted_tok_low_cached=splatted_tok_low_cached,
    )
    frame_ids = _frame_ids_from_item(
        item,
        seq_len=int(bundle.z_clean.shape[1]),
        device=device,
        sample=sample,
    )
    camera_condition_tokens, camera_attention_mask = build_camera_condition_from_sample(
        sample,
        device=device,
        anchor_mask=frame_ids.eq(0),
        scene_flow=scene_flow,
    )
    if camera_condition_tokens is None:
        raise RuntimeError(
            f"Formal SceneFlow item {item.get('cache_path', '<unknown>')} "
            "sample is missing camera_to_world_corrected/intrinsics; camera conditioning must use Waymo camera parameters."
        )
    bundle.camera_condition_tokens = camera_condition_tokens
    bundle.camera_attention_mask = camera_attention_mask
    bundle.captions = [str(item.get("caption", ""))]
    bundle.frame_ids = frame_ids
    kind = str(getattr(bundle, "extras", {}).get("asset_condition_kind", mode_kind))
    bundle.asset_condition_kind = _asset_condition_kind_from_item(item, kind)
    bundle = _attach_sky_tokens_to_bundle(bundle, item, args, device)
    if include_rgb_render_context:
        bundle = _attach_rgb_context(bundle, _formal_rgb_context(item, device=device, strict=True))
    return bundle


def _asset_condition_kinds(bundle, batch_size: int) -> list[str]:
    kind = getattr(bundle, "asset_condition_kind", None)
    if kind is None:
        extras = getattr(bundle, "extras", {})
        if isinstance(extras, dict):
            kind = extras.get("asset_condition_kind") or extras.get("mode_kind")
    if kind is None:
        return ["mode_a"] * int(batch_size)
    if isinstance(kind, str):
        normalized = "mode_b_empty" if kind in ("mode_b", "mode_b_empty", "empty") else str(kind)
        return [normalized] * int(batch_size)
    values = list(kind)
    if len(values) != int(batch_size):
        raise ValueError(f"asset_condition_kind length {len(values)} != batch size {batch_size}")
    return [
        "mode_b_empty" if str(v) in ("mode_b", "mode_b_empty", "empty") else str(v)
        for v in values
    ]


def _asset_condition_kind_for_model(bundle, batch_size: int) -> list[str]:
    return _asset_condition_kinds(bundle, batch_size)


def _maybe_drop_asset_kv(
    bundle,
    drop_prob: float,
    drop_mask: torch.Tensor | None = None,
) -> None:
    """Mask asset conditions for CFG-unconditional training rows.

    This intentionally does not insert the learned empty-asset token. That token
    is a conditional Mode-B signal: a target hole exists, but no target asset is
    provided. CFG uncond uses the learned asset-null token, which is distinct
    from the user-facing formal-edit path where asset input is required.
    """
    if drop_mask is None and float(drop_prob) <= 0.0:
        return
    if not torch.is_tensor(bundle.F_asset_tokens):
        return
    if bundle.F_asset_tokens.shape[1] == 0:
        return
    B = int(bundle.F_asset_tokens.shape[0])
    if drop_mask is None:
        drop = torch.rand(B, device=bundle.F_asset_tokens.device) < float(drop_prob)
    else:
        drop = drop_mask.to(device=bundle.F_asset_tokens.device, dtype=torch.bool).view(-1)
        if int(drop.numel()) != B:
            raise ValueError(f"drop_mask has {int(drop.numel())} rows, asset batch has {B}")
    if not bool(drop.any().item()):
        return
    mask = getattr(bundle, "encoder_attention_mask", None)
    if bundle.F_asset_tokens.ndim == 5:
        if mask is None:
            mask = torch.ones(
                bundle.F_asset_tokens.shape[:-1],
                device=bundle.F_asset_tokens.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.clone().to(device=bundle.F_asset_tokens.device, dtype=torch.bool)
        mask[drop] = False
        bundle.encoder_attention_mask = mask
    else:
        if mask is None:
            mask = torch.ones(
                bundle.F_asset_tokens.shape[:2],
                device=bundle.F_asset_tokens.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.clone().to(device=bundle.F_asset_tokens.device, dtype=torch.bool)
        mask[drop] = False
        bundle.encoder_attention_mask = mask

    kinds = _asset_condition_kinds(bundle, B)
    for idx, should_drop in enumerate(drop.detach().cpu().tolist()):
        if should_drop:
            kinds[idx] = "asset_uncond"
    bundle.asset_condition_kind = kinds


def _pad_asset_tokens_for_batch(bundles: list[Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if len(bundles) == 0:
        raise ValueError("Cannot pad an empty bundle list.")
    token_lists = [bundle.F_asset_tokens for bundle in bundles]
    device = token_lists[0].device
    dtype = token_lists[0].dtype
    if token_lists[0].ndim == 5:
        _, _, S, P, C = token_lists[0].shape
        lengths = [int(tokens.shape[1]) for tokens in token_lists]
        max_len = max(lengths)
        batch = len(token_lists)
        if max_len == 0:
            return token_lists[0].new_zeros((batch, 0, S, P, C)), None
        out = torch.zeros((batch, max_len, S, P, C), device=device, dtype=dtype)
        mask = torch.zeros((batch, max_len, S, P), device=device, dtype=torch.bool)
        for row, bundle in enumerate(bundles):
            tokens = bundle.F_asset_tokens
            n = int(tokens.shape[1])
            if n == 0:
                continue
            out[row : row + 1, :n] = tokens
            row_mask = getattr(bundle, "encoder_attention_mask", None)
            if row_mask is None:
                mask[row : row + 1, :n] = True
            else:
                mask[row : row + 1, :n] = row_mask.to(device=device, dtype=torch.bool)
        return out, mask

    dim = int(token_lists[0].shape[-1])
    lengths = [int(tokens.shape[1]) for tokens in token_lists]
    max_len = max(lengths)
    batch = len(token_lists)
    if max_len == 0:
        return token_lists[0].new_zeros((batch, 0, dim)), None
    out = torch.zeros((batch, max_len, dim), device=device, dtype=dtype)
    mask = torch.zeros((batch, max_len), device=device, dtype=torch.bool)
    for row, tokens in enumerate(token_lists):
        n = int(tokens.shape[1])
        if n == 0:
            continue
        out[row, :n] = tokens.squeeze(0)
        mask[row, :n] = True
    if all(n == max_len for n in lengths):
        return out, None
    return out, mask


def _merge_bundles_for_scene_flow(bundles: list[Any]) -> tuple[Any, torch.Tensor | None, list[int]]:
    asset_tokens, asset_mask = _pad_asset_tokens_for_batch(bundles)
    lengths = [int(bundle.F_asset_tokens.shape[1]) for bundle in bundles]
    camera_tokens_list = [getattr(bundle, "camera_condition_tokens", None) for bundle in bundles]
    if all(torch.is_tensor(tokens) for tokens in camera_tokens_list):
        camera_condition_tokens = torch.cat(camera_tokens_list, dim=0)
        camera_masks = [getattr(bundle, "camera_attention_mask", None) for bundle in bundles]
        camera_attention_mask = (
            torch.cat(camera_masks, dim=0)
            if all(torch.is_tensor(mask) for mask in camera_masks)
            else None
        )
    else:
        camera_condition_tokens = None
        camera_attention_mask = None
    sky_tokens_list = [getattr(bundle, "sky_gen_clean", None) for bundle in bundles]
    if all(torch.is_tensor(tokens) for tokens in sky_tokens_list):
        sky_gen_clean = torch.cat(sky_tokens_list, dim=0)
        sky_masks = [getattr(bundle, "sky_gen_attention_mask", None) for bundle in bundles]
        sky_gen_attention_mask = (
            torch.cat(sky_masks, dim=0)
            if all(torch.is_tensor(mask) for mask in sky_masks)
            else None
        )
    else:
        sky_gen_clean = None
        sky_gen_attention_mask = None
    frame_ids_list = [getattr(bundle, "frame_ids", None) for bundle in bundles]
    if all(torch.is_tensor(ids) for ids in frame_ids_list):
        frame_ids = torch.cat(frame_ids_list, dim=0)
    else:
        frame_ids = None
    merged = SimpleNamespace(
        z_clean=torch.cat([bundle.z_clean for bundle in bundles], dim=0),
        z_splat=torch.cat([bundle.z_splat for bundle in bundles], dim=0),
        scaffold_tok=torch.cat([bundle.scaffold_tok for bundle in bundles], dim=0),
        M_preserve=torch.cat([bundle.M_preserve for bundle in bundles], dim=0),
        M_source=torch.cat([bundle.M_source for bundle in bundles], dim=0),
        M_dest=torch.cat([bundle.M_dest for bundle in bundles], dim=0),
        F_asset_tokens=asset_tokens,
        encoder_attention_mask=asset_mask,
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        sky_gen_clean=sky_gen_clean,
        sky_gen_attention_mask=sky_gen_attention_mask,
        frame_ids=frame_ids,
        asset_condition_kind=[kind for bundle in bundles for kind in _asset_condition_kinds(bundle, 1)],
        phase4_slots=[],
        captions=[caption for bundle in bundles for caption in getattr(bundle, "captions", [""])],
    )
    merged = _attach_rgb_context(merged, _merge_rgb_contexts(bundles))
    return merged, asset_mask, lengths


def require_formal_pullback_calibration(scene_flow: nn.Module) -> PullbackCalibration:
    """Return the validated formal pullback contract or fail closed.

    Formal editing uses the v2 render boundary, which is numerically identity.
    Requiring the artifact prevents a tokenizer or calibration swap from
    silently bypassing the checkpoint-bound contract merely because the
    accepted Scheme-A coefficients happen to be one.
    """

    calibration = getattr(unwrap_ddp(scene_flow), "_pullback_calibration", None)
    if not isinstance(calibration, PullbackCalibration):
        raise RuntimeError(
            "Formal SceneFlow decode/render requires a validated "
            "PullbackCalibration attached at startup"
        )
    return calibration


def _add_formal_rgb_render_loss(
    loss: torch.Tensor,
    metrics: dict[str, float],
    *,
    args: argparse.Namespace,
    global_step: int | None,
    active: bool,
    render_vggt_model: nn.Module | None,
    scene_flow_root: nn.Module,
    z_pred: torch.Tensor,
    bundle: Any,
    target: Any,
    lpips_model: nn.Module | None,
) -> torch.Tensor:
    if not active:
        metrics["rgb_render_active"] = 0.0
        metrics["loss_level_consistency"] = 0.0
        metrics["loss_head_consistency"] = 0.0
        metrics["loss_level_consistency_weighted"] = 0.0
        metrics["loss_head_consistency_weighted"] = 0.0
        return loss
    if render_vggt_model is None:
        raise RuntimeError("Formal RGB loss requires the frozen DGGT decode/render model.")
    images = getattr(bundle, "rgb_render_images", None)
    masks = getattr(bundle, "rgb_render_masks", None)
    pose = getattr(bundle, "rgb_render_pose_enc_dggt", None)
    timestamps = getattr(bundle, "rgb_render_timestamps", None)
    if not all(torch.is_tensor(value) for value in (images, masks, pose, timestamps)):
        raise RuntimeError(
            "Formal RGB loss requires RGB, GT sky mask, timestamps, and input-DGGT pose; "
            "teacher depth is deliberately not part of this contract."
        )
    available = min(int(images.shape[0]), int(z_pred.shape[0]), int(target.sigmas.shape[0]))
    render_samples = (
        available
        if int(args.rgb_render_max_samples) <= 0
        else min(int(args.rgb_render_max_samples), available)
    )
    sigma = target.sigmas[:render_samples]
    sigma_weights = rgb_render_sigma_weight(
        sigma,
        float(getattr(args, "rgb_render_sigma_power", 2.0)),
    )
    pullback_calibration = require_formal_pullback_calibration(scene_flow_root)
    result = compute_rgb_render_loss(
        vggt_model=unwrap_ddp(render_vggt_model),
        scene_flow_root=scene_flow_root,
        z_clean_pred_n=z_pred,
        z_clean_target_n=getattr(bundle, "z_clean_n", None),
        images=images,
        timestamps=timestamps,
        render_pose_enc_dggt=pose,
        render_sky_probability=masks,
        loss_sky_mask_gt=masks,
        patch_grid=getattr(bundle, "patch_grid", getattr(args, "patch_grid", (25, 37))),
        patch_start_idx=int(
            getattr(bundle, "rgb_render_patch_start_idx", getattr(bundle, "patch_start_idx", 5))
        ),
        max_samples=int(args.rgb_render_max_samples),
        max_frames=int(args.rgb_render_max_frames),
        render_stride=int(args.rgb_render_stride),
        background_mode="gt_sky",
        sky_tokens=None,
        sky_grid=tuple(getattr(args, "sky_grid", DEFAULT_SKY_GRID)),
        patch_weight_mask=target.M_edit,
        sky_weight=0.0,
        camera_grad_scale=0.0,
        sky_mask_grad_scale=0.0,
        lpips_model=lpips_model,
        lpips_weight=float(args.rgb_render_lpips_weight),
        loss_sample_weight=sigma_weights,
        conf_weight_power=float(
            getattr(args, "feedback_conf_weight_power", 1.0)
        ),
        conf_weight_floor=float(
            getattr(args, "feedback_conf_weight_floor", 0.05)
        ),
        pullback_calibration=pullback_calibration,
    )
    ramp = rgb_render_loss_ramp(args, global_step)
    weighted = float(args.lambda_rgb_render) * float(ramp) * result.loss
    result_level_loss = getattr(result, "level_loss", result.loss * 0.0)
    result_head_loss = getattr(result, "head_loss", result.loss * 0.0)
    weighted_level = (
        float(getattr(args, "lambda_level_consistency", 0.0))
        * float(ramp)
        * result_level_loss
    )
    weighted_head = (
        float(getattr(args, "lambda_head_consistency", 0.0))
        * float(ramp)
        * result_head_loss
    )
    metrics.update(result.logs)
    metrics["rgb_render_sigma_mean"] = float(sigma.float().mean().detach().item())
    metrics["rgb_render_sigma_weight_mean"] = float(sigma_weights.mean().detach().item())
    metrics["loss_rgb_render_sigma_weighted"] = float(result.loss.detach().item())
    metrics["loss_rgb_render_weighted"] = float(weighted.detach().item())
    metrics["loss_level_consistency_weighted"] = float(weighted_level.detach().item())
    metrics["loss_head_consistency_weighted"] = float(weighted_head.detach().item())
    metrics["rgb_render_ramp"] = float(ramp)
    metrics["rgb_render_active"] = 1.0
    return loss + weighted + weighted_level + weighted_head


def train_step(
    item: dict[str, Any] | list[dict[str, Any]],
    assembler: FlowFeatureAssembler,
    scene_flow: nn.Module,
    scheduler: Any,
    device: torch.device,
    args,
    text_encoder: nn.Module | None = None,
    *,
    global_step: int | None = None,
    render_vggt_model: nn.Module | None = None,
    lpips_model: nn.Module | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    rgb_render_active = should_apply_rgb_render_loss(
        args,
        global_step,
        training=unwrap_ddp(scene_flow).training,
    )
    if isinstance(item, list):
        if len(item) == 0:
            raise ValueError("Received an empty training micro-batch.")
        if len(item) > 1 and not bool(getattr(args, "no_batch_scene_flow", False)):
            if all(single.get("flow_inputs_cached") is not None for single in item):
                bundle, asset_lengths, mean_num_objects = build_cached_flow_batch_bundle(
                    item,
                    assembler,
                    device,
                    args=args,
                    scene_flow=scene_flow,
                    include_rgb_render_context=rgb_render_active,
                )
                asset_mask = getattr(bundle, "encoder_attention_mask", None)
            else:
                bundles = [
                    build_flow_bundle(
                        single,
                        assembler,
                        device,
                        args=args,
                        scene_flow=scene_flow,
                        include_rgb_render_context=rgb_render_active,
                    )
                    for single in item
                ]
                bundle, asset_mask, asset_lengths = _merge_bundles_for_scene_flow(bundles)
                mean_num_objects = sum(float(len(b.phase4_slots)) for b in bundles) / float(len(bundles))
            text_drop_mask = sample_uncond_drop_mask(
                int(bundle.z_clean.shape[0]),
                args.uncond_drop_prob,
                device=bundle.z_clean.device,
                training=unwrap_ddp(scene_flow).training,
            )
            asset_control_drop_mask = sample_uncond_drop_mask(
                int(bundle.z_clean.shape[0]),
                args.uncond_drop_prob,
                device=bundle.z_clean.device,
                training=unwrap_ddp(scene_flow).training,
            )
            if asset_control_drop_mask is not None:
                _maybe_drop_asset_kv(bundle, args.uncond_drop_prob, drop_mask=asset_control_drop_mask)
                asset_mask = getattr(bundle, "encoder_attention_mask", None)
            sf = unwrap_ddp(scene_flow)
            z_clean_n = sf.normalize(bundle.z_clean)
            z_splat_n = sf.normalize(bundle.z_splat)
            bundle.F_asset_tokens = normalize_asset_latents(sf, bundle.F_asset_tokens)
            bundle.z_clean_n = z_clean_n

            M_edit_soft, M_edit, M_keep, boundary = build_formal_edit_domains(
                bundle,
                args,
                device=z_clean_n.device,
                dtype=z_clean_n.dtype,
            )
            bundle.M_edit = M_edit
            target = build_masked_rectified_flow_target(
                scheduler,
                z_clean_n,
                z_splat_n,
                M_edit,
                weighting_scheme=args.weighting_scheme,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
                loss_weighting_scheme=args.loss_weighting_scheme,
                time_shift=float(args.shift),
                t_eps=scene_flow_t_eps(scene_flow),
                generator=generator,
            )
            use_repa = float(args.lambda_repa) != 0.0
            text_tokens, text_mask = encode_text_condition(
                text_encoder,
                getattr(bundle, "captions", None),
                drop_mask=text_drop_mask,
            )
            out = scene_flow(
                target.z_t,
                target.sigmas,
                z_splat_n,
                bundle.scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                bundle.F_asset_tokens,
                encoder_attention_mask=asset_mask,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                return_mid=use_repa,
                return_base=float(args.base_model_coeff) != 0.0,
                asset_condition_kind=_asset_condition_kind_for_model(bundle, int(z_clean_n.shape[0])),
                control_drop_mask=asset_control_drop_mask,
                frame_ids=getattr(bundle, "frame_ids", None),
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit,
            )
            if use_repa:
                model_out, mid_repa = out
                if isinstance(model_out, tuple):
                    pred_clean, pred_base = model_out
                else:
                    pred_clean, pred_base = model_out, None
            else:
                model_out, mid_repa = out, None
                if isinstance(model_out, tuple):
                    pred_clean, pred_base = model_out
                else:
                    pred_clean, pred_base = model_out, None
            v_pred = model_prediction_to_velocity(scene_flow, pred_clean, target)
            z_pred = model_prediction_to_clean(scene_flow, pred_clean, target)
            v_base_pred = (
                model_prediction_to_velocity(scene_flow, pred_base, target)
                if pred_base is not None
                else None
            )
            loss, metrics = compute_total_loss(
                v_pred=v_pred,
                v_gt=target.v_gt,
                eps=target.eps,
                bundle=bundle,
                sd3_weights=target.weights,
                mid_repa=mid_repa,
                repa_target=target.z_clean_target,
                z_pred=z_pred,
                z_preserve_target=target.z_cond,
                M_edit=target.M_edit,
                M_preserve_loss=M_keep,
                boundary_mask=boundary,
                v_base_pred=v_base_pred,
                base_model_coeff=args.base_model_coeff,
                lambda_flow=args.lambda_flow,
                lambda_preserve=args.lambda_preserve,
                lambda_boundary=args.lambda_boundary,
                lambda_repa=args.lambda_repa,
                lambda_identity=args.lambda_identity,
                identity_batch=~M_edit.detach().to(torch.bool).flatten(1).any(dim=1),
                preserve_floor=args.preserve_floor,
            )
            metrics.update({
                "edit_weight_mean": float(M_edit_soft.mean().item()),
                "edit_frac": float(M_edit.mean().item()),
                "num_objects": float(mean_num_objects),
                "kv_tokens": sum(float(n) for n in asset_lengths) / float(len(asset_lengths)),
                "asset_token_count": estimate_sparse_asset_token_count(sf, bundle.F_asset_tokens, asset_mask),
                "control_token_count": estimate_control_token_count(
                    sf,
                    bundle.M_source,
                    bundle.M_dest,
                    getattr(bundle, "patch_grid", getattr(args, "patch_grid", (25, 37))),
                ),
                "sigma_mean": float(target.sigmas.float().mean().item()),
                "micro_batch_size": float(len(item)),
            })
            loss = _add_formal_rgb_render_loss(
                loss,
                metrics,
                args=args,
                global_step=global_step,
                active=rgb_render_active,
                render_vggt_model=render_vggt_model,
                scene_flow_root=sf,
                z_pred=z_pred,
                bundle=bundle,
                target=target,
                lpips_model=lpips_model,
            )
            metrics["loss"] = float(loss.detach().item())
            return loss, metrics

        losses: list[torch.Tensor] = []
        metric_sums: dict[str, float] = {}
        for single in item:
            loss_i, metrics_i = train_step(
                single,
                assembler,
                scene_flow,
                scheduler,
                device,
                args,
                text_encoder,
                global_step=global_step,
                render_vggt_model=render_vggt_model,
                lpips_model=lpips_model,
                generator=generator,
            )
            losses.append(loss_i)
            for key, value in metrics_i.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        scale = 1.0 / float(len(item))
        metrics = {key: value * scale for key, value in metric_sums.items()}
        metrics["micro_batch_size"] = float(len(item))
        return torch.stack(losses).mean(), metrics

    bundle = build_flow_bundle(
        item,
        assembler,
        device,
        args=args,
        scene_flow=scene_flow,
        include_rgb_render_context=rgb_render_active,
    )
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean)
    z_splat_n = sf.normalize(bundle.z_splat)
    bundle.F_asset_tokens = normalize_asset_latents(sf, bundle.F_asset_tokens)
    bundle.z_clean_n = z_clean_n
    text_drop_mask = sample_uncond_drop_mask(
        int(bundle.z_clean.shape[0]),
        args.uncond_drop_prob,
        device=bundle.z_clean.device,
        training=unwrap_ddp(scene_flow).training,
    )
    asset_control_drop_mask = sample_uncond_drop_mask(
        int(bundle.z_clean.shape[0]),
        args.uncond_drop_prob,
        device=bundle.z_clean.device,
        training=unwrap_ddp(scene_flow).training,
    )
    if asset_control_drop_mask is not None:
        _maybe_drop_asset_kv(bundle, args.uncond_drop_prob, drop_mask=asset_control_drop_mask)

    M_edit_soft, M_edit, M_keep, boundary = build_formal_edit_domains(
        bundle,
        args,
        device=z_clean_n.device,
        dtype=z_clean_n.dtype,
    )
    bundle.M_edit = M_edit
    target = build_masked_rectified_flow_target(
        scheduler,
        z_clean_n,
        z_splat_n,
        M_edit,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
        loss_weighting_scheme=args.loss_weighting_scheme,
        time_shift=float(args.shift),
        t_eps=scene_flow_t_eps(scene_flow),
        generator=generator,
    )
    use_repa = float(args.lambda_repa) != 0.0
    text_tokens, text_mask = encode_text_condition(
        text_encoder,
        getattr(bundle, "captions", None),
        drop_mask=text_drop_mask,
    )
    out = scene_flow(
        target.z_t,
        target.sigmas,
        z_splat_n,
        bundle.scaffold_tok,
        bundle.M_preserve,
        bundle.M_source,
        bundle.M_dest,
        bundle.F_asset_tokens,
        encoder_attention_mask=getattr(bundle, "encoder_attention_mask", None),
        text_tokens=text_tokens,
        text_attention_mask=text_mask,
        camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
        camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
        return_mid=use_repa,
        return_base=float(args.base_model_coeff) != 0.0,
        asset_condition_kind=_asset_condition_kind_for_model(bundle, int(z_clean_n.shape[0])),
        control_drop_mask=asset_control_drop_mask,
        frame_ids=getattr(bundle, "frame_ids", None),
        fps=FORMAL_SCENE_FPS,
        flow_edit_mask=M_edit,
    )
    if use_repa:
        model_out, mid_repa = out
        if isinstance(model_out, tuple):
            pred_clean, pred_base = model_out
        else:
            pred_clean, pred_base = model_out, None
    else:
        model_out, mid_repa = out, None
        if isinstance(model_out, tuple):
            pred_clean, pred_base = model_out
        else:
            pred_clean, pred_base = model_out, None
    v_pred = model_prediction_to_velocity(scene_flow, pred_clean, target)
    z_pred = model_prediction_to_clean(scene_flow, pred_clean, target)
    v_base_pred = (
        model_prediction_to_velocity(scene_flow, pred_base, target)
        if pred_base is not None
        else None
    )
    loss, metrics = compute_total_loss(
        v_pred=v_pred,
        v_gt=target.v_gt,
        eps=target.eps,
        bundle=bundle,
        sd3_weights=target.weights,
        mid_repa=mid_repa,
        repa_target=target.z_clean_target,
        z_pred=z_pred,
        z_preserve_target=target.z_cond,
        M_edit=target.M_edit,
        M_preserve_loss=M_keep,
        boundary_mask=boundary,
        v_base_pred=v_base_pred,
        base_model_coeff=args.base_model_coeff,
        lambda_flow=args.lambda_flow,
        lambda_preserve=args.lambda_preserve,
        lambda_boundary=args.lambda_boundary,
        lambda_repa=args.lambda_repa,
        lambda_identity=args.lambda_identity,
        identity_batch=~M_edit.detach().to(torch.bool).flatten(1).any(dim=1),
        preserve_floor=args.preserve_floor,
    )
    metrics.update({
        "edit_weight_mean": float(M_edit_soft.mean().item()),
        "edit_frac": float(M_edit.mean().item()),
        "num_objects": float(len(bundle.phase4_slots)),
        "kv_tokens": float(bundle.F_asset_tokens.shape[1]),
        "asset_token_count": estimate_sparse_asset_token_count(
            sf,
            bundle.F_asset_tokens,
            getattr(bundle, "encoder_attention_mask", None),
        ),
        "control_token_count": estimate_control_token_count(
            sf,
            bundle.M_source,
            bundle.M_dest,
            getattr(bundle, "patch_grid", getattr(args, "patch_grid", (25, 37))),
        ),
        "sigma_mean": float(target.sigmas.float().mean().item()),
    })
    loss = _add_formal_rgb_render_loss(
        loss,
        metrics,
        args=args,
        global_step=global_step,
        active=rgb_render_active,
        render_vggt_model=render_vggt_model,
        scene_flow_root=sf,
        z_pred=z_pred,
        bundle=bundle,
        target=target,
        lpips_model=lpips_model,
    )
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


def _move_predictions(predictions: dict, device: torch.device) -> dict:
    out: dict[str, Any] = {}
    for k, v in predictions.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if torch.is_tensor(x) else x for x in v] if v is not None else v
        elif v is None:
            out[k] = None
        else:
            out[k] = v
    return out


def _move_mode_b(mode_b: dict, device: torch.device) -> dict:
    out = dict(mode_b)
    for k in ("delete_mask", "delete_mask_per_frame_subset", "subset_frames",
              "delete_core_indices", "delete_shell_indices"):
        v = out.get(k)
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def _move_asset_pass(apr, device: torch.device):
    from dggt.models.asset_pass import AssetPassResult

    return AssetPassResult(
        patch_grid=apr.patch_grid,
        patch_start_idx=apr.patch_start_idx,
        object_keys=list(apr.object_keys),
        cameras_waymo={k: v.to(device) for k, v in apr.cameras_waymo.items()} if apr.cameras_waymo else {},
        F_g_lut_asset={k: [lv.to(device) for lv in v] for k, v in apr.F_g_lut_asset.items()},
        ptr_asset={k: [p.to(device) for p in v] for k, v in apr.ptr_asset.items()},
        G_asset_waymo={
            k: [{kk: vv.to(device) for kk, vv in g.items()} for g in v]
            for k, v in apr.G_asset_waymo.items()
        },
        G_asset_dggt=None
        if apr.G_asset_dggt is None
        else {
            k: [{kk: vv.to(device) for kk, vv in g.items()} for g in v]
            for k, v in apr.G_asset_dggt.items()
        },
        I_asset={k: v.to(device) for k, v in apr.I_asset.items()},
        A_asset={k: v.to(device) for k, v in apr.A_asset.items()},
        asset_patch_valid_mask={
            k: v.to(device) for k, v in apr.asset_patch_valid_mask.items()
        },
        asset_pass_space=apr.asset_pass_space,
        fit_metrics=apr.fit_metrics,
    )


def _bundle_frame_ids(
    bundle: Any,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    frame_ids = getattr(bundle, "frame_ids", None)
    if frame_ids is None:
        return torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
    if torch.is_tensor(frame_ids):
        frame_ids_t = frame_ids.to(device=device, dtype=torch.long)
    else:
        frame_ids_t = torch.as_tensor(frame_ids, device=device, dtype=torch.long)
    if frame_ids_t.ndim == 1:
        frame_ids_t = frame_ids_t.view(1, -1).expand(batch_size, -1)
    if frame_ids_t.shape != (batch_size, seq_len):
        raise ValueError(f"bundle.frame_ids must be [S] or [B,S], got {tuple(frame_ids_t.shape)}")
    return frame_ids_t.contiguous()


def _slice_time(tensor: torch.Tensor | None, start: int, end: int, seq_len: int) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if tensor.ndim >= 2 and int(tensor.shape[1]) == int(seq_len):
        return tensor[:, start:end]
    return tensor


def _slice_asset_time(tensor: torch.Tensor | None, start: int, end: int, seq_len: int) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if tensor.ndim == 5 and int(tensor.shape[2]) == int(seq_len):
        return tensor[:, :, start:end]
    if tensor.ndim == 4 and int(tensor.shape[2]) == int(seq_len):
        return tensor[:, :, start:end]
    if tensor.ndim == 4 and int(tensor.shape[1]) == int(seq_len):
        return tensor[:, start:end]
    return tensor


def _validation_sliding_params(args: argparse.Namespace, seq_len: int) -> tuple[int, int] | None:
    window = int(getattr(args, "val_sliding_window", 0) or 0)
    if window <= 0 or int(seq_len) <= window:
        return None
    stride = int(getattr(args, "val_sliding_stride", 0) or 0)
    if stride <= 0:
        stride = default_window_stride(window)
    return min(window, int(seq_len)), max(1, stride)


@torch.no_grad()
def _cfg_sample_edit_latents_sliding(
    scene_flow: nn.Module,
    bundle,
    args,
    step: int,
    device: torch.device,
    guidance_scale: float,
    text_encoder: nn.Module | None,
    *,
    window: int,
    stride: int,
) -> torch.Tensor:
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean.float())
    t_steps = rae_t_grid(
        num_steps=int(args.val_sample_steps),
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
        t_eps=scene_flow_t_eps(scene_flow),
    )
    z_splat_n = sf.normalize(bundle.z_splat.float())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))
    _, M_edit, _, _ = build_formal_edit_domains(
        bundle,
        args,
        device=device,
        dtype=z_clean_n.dtype,
    )
    z = torch.empty_like(z_clean_n)
    z.normal_(generator=generator)
    z = project_masked_flow_state(z, z_splat_n, M_edit)
    batch_size = int(z.shape[0])
    seq_len = int(z.shape[1])
    frame_ids = _bundle_frame_ids(bundle, batch_size=batch_size, seq_len=seq_len, device=device)
    windows = window_slices(seq_len, window, stride)

    F_asset = normalize_asset_latents(sf, bundle.F_asset_tokens)
    if F_asset.ndim in (4, 5):
        F_uncond = torch.zeros_like(F_asset)
        uncond_asset_mask = torch.zeros(F_asset.shape[:-1], device=F_asset.device, dtype=torch.bool)
    else:
        F_uncond = F_asset.new_zeros((batch_size, 0, F_asset.shape[-1]))
        uncond_asset_mask = None
    encoder_attention_mask = getattr(bundle, "encoder_attention_mask", None)
    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(text_encoder, [""] * batch_size if text_tokens is not None else None)
    asset_kinds = _asset_condition_kind_for_model(bundle, batch_size)
    asset_control_scale = float(getattr(args, "asset_control_guidance_scale", 1.0))
    do_cfg = (
        abs(float(guidance_scale) - 1.0) > 1e-6
        or abs(asset_control_scale - 1.0) > 1e-6
    )
    drop_all_control = torch.ones((batch_size,), device=device, dtype=torch.bool)
    camera_condition_tokens = getattr(bundle, "camera_condition_tokens", None)
    camera_attention_mask = getattr(bundle, "camera_attention_mask", None)

    for i in range(int(args.val_sample_steps)):
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)
        v_acc = torch.zeros_like(z)
        v_weight = torch.zeros((1, seq_len, 1, 1), device=device, dtype=z.dtype)

        for start, end in windows:
            actual = int(end - start)
            w = cosine_window(actual, device=device, dtype=z.dtype).view(1, actual, 1, 1)
            z_w = z[:, start:end]
            z_splat_w = z_splat_n[:, start:end]
            scaffold_w = bundle.scaffold_tok[:, start:end]
            M_preserve_w = bundle.M_preserve[:, start:end]
            M_source_w = bundle.M_source[:, start:end]
            M_dest_w = bundle.M_dest[:, start:end]
            M_edit_w = M_edit[:, start:end]
            frame_ids_w = frame_ids[:, start:end]
            camera_tokens_w = _slice_time(camera_condition_tokens, start, end, seq_len)
            camera_mask_w = _slice_time(camera_attention_mask, start, end, seq_len)
            F_asset_w = _slice_asset_time(F_asset, start, end, seq_len)
            asset_mask_w = _slice_asset_time(encoder_attention_mask, start, end, seq_len)
            F_uncond_w = _slice_asset_time(F_uncond, start, end, seq_len)
            uncond_mask_w = _slice_asset_time(uncond_asset_mask, start, end, seq_len)

            v_full = sf(
                z_w,
                sigma,
                z_splat_w,
                scaffold_w,
                M_preserve_w,
                M_source_w,
                M_dest_w,
                F_asset_w,
                encoder_attention_mask=asset_mask_w,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=camera_tokens_w,
                camera_attention_mask=camera_mask_w,
                asset_condition_kind=asset_kinds,
                return_mid=False,
                frame_ids=frame_ids_w,
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit_w,
            )
            if do_cfg:
                v_text = sf(
                    z_w,
                    sigma,
                    z_splat_w,
                    scaffold_w,
                    M_preserve_w,
                    M_source_w,
                    M_dest_w,
                    F_uncond_w,
                    encoder_attention_mask=uncond_mask_w,
                    text_tokens=text_tokens,
                    text_attention_mask=text_mask,
                    camera_condition_tokens=camera_tokens_w,
                    camera_attention_mask=camera_mask_w,
                    asset_condition_kind=["asset_uncond"] * batch_size,
                    return_mid=False,
                    control_drop_mask=drop_all_control,
                    frame_ids=frame_ids_w,
                    fps=FORMAL_SCENE_FPS,
                    flow_edit_mask=M_edit_w,
                )
                v_uncond = sf(
                    z_w,
                    sigma,
                    z_splat_w,
                    scaffold_w,
                    M_preserve_w,
                    M_source_w,
                    M_dest_w,
                    F_uncond_w,
                    encoder_attention_mask=uncond_mask_w,
                    text_tokens=text_null,
                    text_attention_mask=text_null_mask,
                    camera_condition_tokens=camera_tokens_w,
                    camera_attention_mask=camera_mask_w,
                    asset_condition_kind=["asset_uncond"] * batch_size,
                    return_mid=False,
                    control_drop_mask=drop_all_control,
                    frame_ids=frame_ids_w,
                    fps=FORMAL_SCENE_FPS,
                    flow_edit_mask=M_edit_w,
                )
                v_pred = (
                    v_uncond
                    + float(guidance_scale) * (v_text - v_uncond)
                    + asset_control_scale * (v_full - v_text)
                )
            else:
                v_pred = v_full
            v = sampler_prediction_to_velocity(sf, v_pred, z_w, sigma)
            v_acc[:, start:end] += v * w
            v_weight[:, start:end] += w

        v = v_acc / v_weight.clamp_min(1e-6)
        z = masked_flow_euler_step(z, v, step_h, z_splat_n, M_edit)

    return project_masked_flow_state(z, z_splat_n, M_edit)


@torch.no_grad()
def cfg_sample_edit_latents(
    scene_flow: nn.Module,
    bundle,
    args,
    step: int,
    device: torch.device,
    guidance_scale: float,
    text_encoder: nn.Module | None = None,
) -> torch.Tensor:
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean.float())
    sliding = _validation_sliding_params(args, int(z_clean_n.shape[1]))
    if sliding is not None:
        return _cfg_sample_edit_latents_sliding(
            scene_flow,
            bundle,
            args,
            step,
            device,
            guidance_scale,
            text_encoder,
            window=sliding[0],
            stride=sliding[1],
        )
    t_steps = rae_t_grid(
        num_steps=int(args.val_sample_steps),
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
        t_eps=scene_flow_t_eps(scene_flow),
    )
    z_splat_n = sf.normalize(bundle.z_splat.float())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))
    _, M_edit, _, _ = build_formal_edit_domains(
        bundle,
        args,
        device=device,
        dtype=z_clean_n.dtype,
    )
    z = torch.empty_like(z_clean_n)
    z.normal_(generator=generator)
    z = project_masked_flow_state(z, z_splat_n, M_edit)
    batch_size = int(z.shape[0])
    frame_ids = _bundle_frame_ids(bundle, batch_size=batch_size, seq_len=int(z.shape[1]), device=device)
    F_asset = normalize_asset_latents(sf, bundle.F_asset_tokens)
    if F_asset.ndim in (4, 5):
        F_uncond = torch.zeros_like(F_asset)
        uncond_asset_mask = torch.zeros(F_asset.shape[:-1], device=F_asset.device, dtype=torch.bool)
    else:
        F_uncond = F_asset.new_zeros((batch_size, 0, F_asset.shape[-1]))
        uncond_asset_mask = None
    encoder_attention_mask = getattr(bundle, "encoder_attention_mask", None)
    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(text_encoder, [""] * batch_size if text_tokens is not None else None)
    asset_kinds = _asset_condition_kind_for_model(bundle, batch_size)
    asset_control_scale = float(getattr(args, "asset_control_guidance_scale", 1.0))
    do_cfg = (
        abs(float(guidance_scale) - 1.0) > 1e-6
        or abs(asset_control_scale - 1.0) > 1e-6
    )
    drop_all_control = torch.ones((batch_size,), device=device, dtype=torch.bool)

    for i in range(int(args.val_sample_steps)):
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)
        v_full = sf(
            z,
            sigma,
            z_splat_n,
            bundle.scaffold_tok,
            bundle.M_preserve,
            bundle.M_source,
            bundle.M_dest,
            F_asset,
            encoder_attention_mask=encoder_attention_mask,
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
            camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
            asset_condition_kind=asset_kinds,
            return_mid=False,
            frame_ids=frame_ids,
            fps=FORMAL_SCENE_FPS,
            flow_edit_mask=M_edit,
        )
        if do_cfg:
            v_text = sf(
                z,
                sigma,
                z_splat_n,
                bundle.scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                F_uncond,
                encoder_attention_mask=uncond_asset_mask,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                asset_condition_kind=["asset_uncond"] * batch_size,
                return_mid=False,
                control_drop_mask=drop_all_control,
                frame_ids=frame_ids,
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit,
            )
            v_uncond = sf(
                z,
                sigma,
                z_splat_n,
                bundle.scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                F_uncond,
                encoder_attention_mask=uncond_asset_mask,
                text_tokens=text_null,
                text_attention_mask=text_null_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                asset_condition_kind=["asset_uncond"] * batch_size,
                return_mid=False,
                control_drop_mask=drop_all_control,
                frame_ids=frame_ids,
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit,
            )
            v = (
                v_uncond
                + float(guidance_scale) * (v_text - v_uncond)
                + asset_control_scale * (v_full - v_text)
            )
        else:
            v = v_full
        v = sampler_prediction_to_velocity(sf, v, z, sigma)
        z = masked_flow_euler_step(z, v, step_h, z_splat_n, M_edit)

    return project_masked_flow_state(z, z_splat_n, M_edit)


def _validation_scales(args) -> list[float]:
    scales = [float(args.guidance_scale)]
    for raw in str(args.val_guidance_scales).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if all(abs(value - seen) > 1e-6 for seen in scales):
            scales.append(value)
    return scales


def _first_item(item: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(item, list):
        if len(item) == 0:
            raise ValueError("Received an empty collated batch.")
        return item[0]
    return item


def dataloader_runtime_kwargs(args) -> dict[str, Any]:
    if int(args.num_workers) <= 0:
        return {}
    return {
        "prefetch_factor": max(1, int(args.prefetch_factor)),
        "persistent_workers": not bool(args.no_persistent_workers),
    }


def _validate_cached_render_pose(
    pose: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    where: str,
) -> torch.Tensor:
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    expected = (int(batch_size), int(seq_len), 9)
    if tuple(pose.shape) != expected:
        raise ValueError(f"{where} DGGT pose_enc must be {expected}, got {tuple(pose.shape)}")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError(f"{where} DGGT pose_enc contains non-finite values")
    return pose.contiguous()


def cached_render_pose_from_item(item: dict[str, Any]) -> torch.Tensor:
    """Return the cached full-context DGGT pose already sliced to this item."""
    predictions = item.get("predictions")
    pose = predictions.get("pose_enc") if isinstance(predictions, dict) else None
    if not torch.is_tensor(pose):
        raise RuntimeError(
            f"{item.get('cache_path', '<unknown>')} is missing cached full-context predictions.pose_enc"
        )
    subset = item.get("subset_frames")
    seq_len = int(torch.as_tensor(subset).numel()) if subset is not None else int(pose.shape[-2])
    batch_size = int(pose.shape[0]) if pose.ndim == 3 else 1
    return _validate_cached_render_pose(
        pose,
        batch_size=batch_size,
        seq_len=seq_len,
        where=str(item.get("cache_path", "validation item")),
    )


def _cached_render_pose_from_payload(payload: dict[str, Any], subset: torch.Tensor) -> torch.Tensor:
    pass1 = payload.get("pass1")
    pose = pass1.get("pose_enc") if isinstance(pass1, dict) else None
    if not torch.is_tensor(pose):
        raise RuntimeError("Validation cache payload is missing full-context pass1.pose_enc")
    subset = subset.detach().cpu().to(torch.long).reshape(-1)
    if pose.ndim == 2:
        selected = pose.index_select(0, subset).unsqueeze(0)
    elif pose.ndim == 3 and int(pose.shape[0]) == 1:
        selected = pose.index_select(1, subset)
    else:
        raise ValueError(f"Cached pass1.pose_enc must be [S,9] or [1,S,9], got {tuple(pose.shape)}")
    return _validate_cached_render_pose(
        selected,
        batch_size=1,
        seq_len=int(subset.numel()),
        where="validation cache payload",
    )


def _prepare_visualization_batch(
    sample: dict[str, Any],
    *,
    render_pose_enc_dggt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    images = sample.get("images", sample.get("images_clean"))
    masks = sample.get("masks", sample.get("sky_mask"))
    timestamps = sample.get("timestamps")
    if not torch.is_tensor(images):
        raise RuntimeError("Validation visualization sample is missing tensor images/images_clean.")
    if not torch.is_tensor(masks):
        raise RuntimeError("Validation visualization sample is missing tensor masks/sky_mask.")
    if not torch.is_tensor(timestamps):
        raise RuntimeError("Validation visualization sample is missing tensor timestamps.")

    if images.ndim == 4:
        images = images.unsqueeze(0)
    if masks.ndim == 4:
        masks = masks.unsqueeze(0)
    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"Expected visualization images [B,S,3,H,W], got {tuple(images.shape)}")
    if masks.ndim != 5:
        raise ValueError(f"Expected visualization masks [B,S,3,H,W], got {tuple(masks.shape)}")
    if timestamps.ndim != 2:
        raise ValueError(f"Expected visualization timestamps [B,S], got {tuple(timestamps.shape)}")
    render_pose_enc_dggt = _validate_cached_render_pose(
        render_pose_enc_dggt,
        batch_size=int(images.shape[0]),
        seq_len=int(images.shape[1]),
        where="validation visualization",
    )
    return {
        "images": images.contiguous(),
        "masks": masks.contiguous(),
        "timestamps": timestamps.contiguous(),
        "render_pose_enc_dggt": render_pose_enc_dggt,
    }


def load_validation_visualization_batch(
    item: dict[str, Any],
    dataset: WaymoFlowCacheDataset,
) -> dict[str, torch.Tensor]:
    """Return the raw batch fields needed by the 3DGS validation renderer."""
    if item.get("sample") is not None:
        return _prepare_visualization_batch(
            item["sample"],
            render_pose_enc_dggt=cached_render_pose_from_item(item),
        )

    cache_path_raw = item.get("cache_path")
    if cache_path_raw is None:
        raise RuntimeError("Fast validation item is missing cache_path; cannot load RGB render inputs.")
    subset = item.get("subset_frames")
    if not torch.is_tensor(subset):
        subset = torch.as_tensor(subset, dtype=torch.long)
    subset = subset.detach().cpu().to(torch.long).contiguous()
    cache_path = Path(cache_path_raw)
    entry = {
        "cache_path": str(cache_path),
        "mode_kind": item.get("mode_kind", "unknown"),
    }

    if is_chunked_flow_cache(cache_path):
        payload = load_chunked_flow_cache_subset(
            cache_path,
            subset,
            consumer="tokenizer_stage_b",
        )
        subset_payload = torch.arange(int(subset.numel()), dtype=torch.long)
    else:
        payload = load_flow_cache(
            cache_path,
            map_location="cpu",
            weights_only=False,
            mmap=bool(getattr(dataset, "mmap_plain_cache", True)),
        )
        subset_payload = subset

    if not is_chunked_flow_cache(cache_path):
        dataset._validate_loaded_payload(payload, cache_path=cache_path, entry=entry)
    sample = dataset._build_sample(payload, subset_payload)
    return _prepare_visualization_batch(
        sample,
        render_pose_enc_dggt=_cached_render_pose_from_payload(payload, subset_payload),
    )


def _save_rgb_validation_images(
    rgb_images: dict[str, torch.Tensor],
    out_dir: Path,
    paths: dict[str, Path],
    frames: int,
    suffix: str,
    *,
    only_generated: bool,
) -> None:
    skip_for_extra = {"input_rgb_gt", "tokenizer_recon_3dgs_rgb", "dggt_clean_3dgs_rgb"}
    for name, tensor in rgb_images.items():
        if only_generated and name in skip_for_extra:
            continue
        key = f"{name}{suffix}" if name.startswith("generated_") else name
        filename = f"{key}.jpg"
        path = out_dir / filename
        save_image_grid(tensor, path, nrow=frames)
        paths[key] = path


def _cuda_empty_cache_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cached_render_pose_from_batch(
    batch: dict[str, torch.Tensor],
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pose = batch.get("render_pose_enc_dggt")
    if not torch.is_tensor(pose):
        raise RuntimeError(
            "Formal RGB rendering requires cached full-context render_pose_enc_dggt; "
            "do not recompute CameraHead on a validation window."
        )
    pose = _validate_cached_render_pose(
        pose,
        batch_size=int(images.shape[0]),
        seq_len=int(images.shape[1]),
        where="formal RGB render batch",
    )
    return pose.to(device=device, dtype=torch.float32, non_blocking=True)


@torch.no_grad()
def render_validation_rgb_gt_sky(
    batch: dict[str, torch.Tensor],
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render formal validation with the cached full-context DGGT camera and GT sky."""
    images = batch["images"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)
    render_pose_enc_dggt = _cached_render_pose_from_batch(batch, images, device)
    frames = min(int(args.val_log_images), int(images.shape[1]))
    sf = unwrap_ddp(scene_flow)
    pullback_calibration = require_formal_pullback_calibration(sf)
    timestamps_raw = batch["timestamps"]
    timestamps = timestamps_raw[0] if torch.is_tensor(timestamps_raw) else torch.as_tensor(timestamps_raw[0])

    result: dict[str, torch.Tensor] = {}

    with autocast_context(args, device):
        outputs = vggt_model.get_aggregator_token_outputs(images)
        aggregated_tokens_list = outputs["aggregated_tokens_list"]
        image_tokens_list = outputs["image_tokens_list"]
        dino_token_list = outputs["dino_token_list"]
        patch_start_idx = int(outputs["patch_start_idx"])
        del outputs

        tokens_4 = select_patch_pyramid(image_tokens_list, TOKENIZER_LEVELS, patch_start_idx)
        z_recon = encode_tokenizer_windowed(
            vggt_model.scene_tokenizer,
            tokens_4,
            patch_grid=args.patch_grid,
            window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        )
        del tokens_4
        recon_patch_tokens = decode_tokenizer_windowed(
            vggt_model.scene_tokenizer,
            z_recon,
            patch_grid=args.patch_grid,
            window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        )
        del z_recon
        recon_full_tokens = reattach_special_tokens(
            image_tokens_list,
            TOKENIZER_LEVELS,
            patch_start_idx,
            recon_patch_tokens,
        )
        del recon_patch_tokens
        recon_image_tokens = replace_selected_levels(
            image_tokens_list,
            TOKENIZER_LEVELS,
            recon_full_tokens,
        )
        del recon_full_tokens

    with autocast_context(args, device):
        with torch.amp.autocast(device_type=device.type, enabled=False):
            depth, _ = vggt_model.depth_head(aggregated_tokens_list, images, patch_start_idx)
            dynamic_conf, _ = vggt_model.instance_head(dino_token_list, images, patch_start_idx)
            clean_gs_map, clean_gs_conf = vggt_model.gs_head(image_tokens_list, images, patch_start_idx)

    del aggregated_tokens_list, dino_token_list, image_tokens_list

    result["dggt_clean_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model,
        images,
        masks,
        timestamps,
        render_pose_enc_dggt,
        depth,
        clean_gs_map,
        clean_gs_conf,
        dynamic_conf,
        device,
        frames,
        background_mode="gt_sky",
        use_sky_mask=True,
    )
    del depth, dynamic_conf, clean_gs_map, clean_gs_conf
    _cuda_empty_cache_if_available()

    with autocast_context(args, device):
        geometry = decode_generated_dggt_geometry(
            vggt_model=vggt_model,
            scene_flow_root=sf,
            z_clean_pred_n=z_generated_raw_n,
            patch_grid=args.patch_grid,
            patch_start_idx=patch_start_idx,
            image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            tokenizer_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
            pullback_calibration=pullback_calibration,
        )
        with torch.amp.autocast(device_type=device.type, enabled=False):
            generated_semantic_logits, _ = vggt_model.semantic_head(
                geometry.dino_tokens,
                None,
                patch_start_idx,
                image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            )
            generated_sky_mask = _semantic_logits_to_sky_mask(generated_semantic_logits)
    raw_gs_map, raw_gs_conf = geometry.gs_map, geometry.gs_conf
    generated_depth, generated_dynamic_conf = geometry.depth, geometry.dynamic_conf

    # Diagnostic only: formal rendering below still composites with the GT sky
    # mask passed as `masks`, not this DGGT semantic-head prediction.
    result["generated_pred_sky_mask"] = _sky_mask_image_grid(generated_sky_mask, frames)
    result["generated_raw_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model,
        images,
        masks,
        timestamps,
        render_pose_enc_dggt,
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        device,
        frames,
        background_mode="gt_sky",
        use_sky_mask=True,
    )
    del (
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        generated_semantic_logits,
        generated_sky_mask,
    )
    _cuda_empty_cache_if_available()

    with autocast_context(args, device):
        recon_agg, recon_dino = split_image_tokens_for_heads(recon_image_tokens)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            recon_depth, _ = vggt_model.depth_head(recon_agg, images, patch_start_idx)
            recon_dynamic_conf, _ = vggt_model.instance_head(recon_dino, images, patch_start_idx)
            recon_gs_map, recon_gs_conf = vggt_model.gs_head(recon_image_tokens, images, patch_start_idx)

    recon_pullback = apply_pullback_calibration(
        recon_depth,
        recon_gs_map,
        log_metric_scale=0.0,
        calibration=pullback_calibration,
        boundary=PULLBACK_RENDER_BOUNDARY,
    )
    if recon_pullback.depth_dggt is not recon_depth or recon_pullback.gs_map_dggt is not recon_gs_map:
        raise AssertionError("formal tokenizer-recon render pullback must be an exact identity")
    recon_depth = recon_pullback.depth_dggt
    recon_gs_map = recon_pullback.gs_map_dggt

    del recon_image_tokens, recon_agg, recon_dino

    result["tokenizer_recon_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model,
        images,
        masks,
        timestamps,
        render_pose_enc_dggt,
        recon_depth,
        recon_gs_map,
        recon_gs_conf,
        recon_dynamic_conf,
        device,
        frames,
        background_mode="gt_sky",
        use_sky_mask=True,
    )
    del render_pose_enc_dggt, recon_depth, recon_dynamic_conf, recon_gs_map, recon_gs_conf
    _cuda_empty_cache_if_available()

    result["input_rgb_gt"] = _image_grid(images, frames)
    return result


@torch.no_grad()
def render_validation_generated_rgb_gt_sky(
    batch: dict[str, torch.Tensor],
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render a generated branch with cached full-context DGGT camera over GT sky."""
    images = batch["images"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)
    render_pose_enc_dggt = _cached_render_pose_from_batch(batch, images, device)
    frames = min(int(args.val_log_images), int(images.shape[1]))
    sf = unwrap_ddp(scene_flow)
    pullback_calibration = require_formal_pullback_calibration(sf)
    timestamps_raw = batch["timestamps"]
    timestamps = timestamps_raw[0] if torch.is_tensor(timestamps_raw) else torch.as_tensor(timestamps_raw[0])

    with autocast_context(args, device):
        outputs = vggt_model.get_aggregator_token_outputs(images)
        patch_start_idx = int(outputs["patch_start_idx"])
        del outputs
        geometry = decode_generated_dggt_geometry(
            vggt_model=vggt_model,
            scene_flow_root=sf,
            z_clean_pred_n=z_generated_raw_n,
            patch_grid=args.patch_grid,
            patch_start_idx=patch_start_idx,
            image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            tokenizer_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
            pullback_calibration=pullback_calibration,
        )
        with torch.amp.autocast(device_type=device.type, enabled=False):
            generated_semantic_logits, _ = vggt_model.semantic_head(
                geometry.dino_tokens,
                None,
                patch_start_idx,
                image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            )
            generated_sky_mask = _semantic_logits_to_sky_mask(generated_semantic_logits)
    raw_gs_map, raw_gs_conf = geometry.gs_map, geometry.gs_conf
    generated_depth, generated_dynamic_conf = geometry.depth, geometry.dynamic_conf

    result = {
        # Diagnostic only; `_render_gs_map_rgb` below receives GT `masks`.
        "generated_pred_sky_mask": _sky_mask_image_grid(generated_sky_mask, frames),
        "generated_raw_3dgs_rgb": _render_gs_map_rgb(
            vggt_model,
            images,
            masks,
            timestamps,
            render_pose_enc_dggt,
            generated_depth,
            raw_gs_map,
            raw_gs_conf,
            generated_dynamic_conf,
            device,
            frames,
            background_mode="gt_sky",
            use_sky_mask=True,
        ),
    }
    del (
        render_pose_enc_dggt,
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        generated_semantic_logits,
        generated_sky_mask,
    )
    _cuda_empty_cache_if_available()
    return result


@torch.no_grad()
def save_validation_images(
    bundle,
    scene_flow: nn.Module,
    log_dir: Path,
    step: int,
    args,
    device: torch.device,
    text_encoder: nn.Module | None = None,
    *,
    visualization_batch: dict[str, torch.Tensor] | None = None,
    vggt_model: nn.Module | None = None,
) -> dict[str, Path]:
    out_dir = log_dir / "validation" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean.float())
    frames = min(int(args.val_log_images), int(z_clean_n.shape[1]))
    paths: dict[str, Path] = {}

    base_images = {
        "target_latent_pca": _latent_pca_grid(z_clean_n, bundle.patch_grid, frames),
        "M_preserve": _mask_grid(bundle.M_preserve, bundle.patch_grid, frames),
        "M_source": _mask_grid(bundle.M_source, bundle.patch_grid, frames),
        "M_dest": _mask_grid(bundle.M_dest, bundle.patch_grid, frames),
    }
    if visualization_batch is not None:
        gt_images = visualization_batch.get("images")
        if torch.is_tensor(gt_images) and gt_images.ndim == 5:
            base_images["input_rgb_gt"] = _image_grid(gt_images, frames)
    for name, image in base_images.items():
        path = out_dir / f"{name}.jpg"
        save_image_grid(image, path, nrow=frames)
        paths[name] = path

    render_rgb = (
        vggt_model is not None
        and visualization_batch is not None
        and not bool(getattr(args, "no_val_render_rgb", False))
    )
    for scale_idx, scale in enumerate(_validation_scales(args)):
        z_generated_raw = cfg_sample_edit_latents(scene_flow, bundle, args, step, device, scale, text_encoder)
        z_generated_preserve_blend = bundle.M_preserve * z_clean_n + (1.0 - bundle.M_preserve) * z_generated_raw
        suffix = f"__cfg{scale:g}"
        images = {
            f"generated_raw_latent_pca{suffix}": _latent_pca_grid(z_generated_raw, bundle.patch_grid, frames),
            f"generated_preserve_blend_latent_pca{suffix}": _latent_pca_grid(
                z_generated_preserve_blend,
                bundle.patch_grid,
                frames,
            ),
            f"abs_error_raw{suffix}": _normalized_mask_grid(
                (z_generated_raw - z_clean_n).abs().mean(dim=-1, keepdim=True),
                bundle.patch_grid,
                frames,
            ),
            f"abs_error_preserve_blend{suffix}": _normalized_mask_grid(
                (z_generated_preserve_blend - z_clean_n).abs().mean(dim=-1, keepdim=True),
                bundle.patch_grid,
                frames,
            ),
        }
        for name, image in images.items():
            path = out_dir / f"{name}.jpg"
            save_image_grid(image, path, nrow=frames)
            paths[name] = path
        if render_rgb:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rgb_images = (
                    render_validation_rgb_gt_sky(
                        visualization_batch,
                        vggt_model,
                        scene_flow,
                        z_generated_raw,
                        args,
                        device,
                    )
                    if scale_idx == 0
                    else render_validation_generated_rgb_gt_sky(
                        visualization_batch,
                        vggt_model,
                        scene_flow,
                        z_generated_raw,
                        args,
                        device,
                    )
                )
                _save_rgb_validation_images(
                    rgb_images,
                    out_dir,
                    paths,
                    frames,
                    suffix,
                    only_generated=scale_idx != 0,
                )
            except Exception as exc:
                print(
                    f"[validation {step:06d}] warning: failed to render 3DGS RGB "
                    f"for cfg={scale:g}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                render_rgb = False
    return paths


@torch.no_grad()
def run_validation(
    loader: DataLoader,
    assembler: FlowFeatureAssembler,
    scene_flow: nn.Module,
    scheduler: Any,
    device: torch.device,
    args,
    step: int,
    log_dir: Path,
    wandb_run,
    ema: EMAModel | None = None,
    text_encoder: nn.Module | None = None,
    vggt_model: nn.Module | None = None,
) -> dict[str, float]:
    with preserve_validation_rng_state(device):
        return _run_validation_impl(
            loader,
            assembler,
            scene_flow,
            scheduler,
            device,
            args,
            step,
            log_dir,
            wandb_run,
            ema,
            text_encoder,
            vggt_model,
        )


@torch.no_grad()
def _run_validation_impl(
    loader: DataLoader,
    assembler: FlowFeatureAssembler,
    scene_flow: nn.Module,
    scheduler: Any,
    device: torch.device,
    args,
    step: int,
    log_dir: Path,
    wandb_run,
    ema: EMAModel | None = None,
    text_encoder: nn.Module | None = None,
    vggt_model: nn.Module | None = None,
) -> dict[str, float]:
    was_training = scene_flow.training
    scene_flow.eval()
    assembler.eval()
    use_val_ema = ema is not None and not args.no_val_ema
    ema_params = list(unwrap_ddp(scene_flow).parameters()) if use_val_ema else None
    if use_val_ema:
        ema.store(ema_params)
        ema.copy_to(ema_params)

    sums: dict[str, float] = {}
    count = 0
    first_item: dict[str, Any] | None = None
    validation_generator = make_validation_generator(device, int(args.seed))
    iterator = loader
    if is_main_process() and not args.no_tqdm:
        iterator = tqdm(loader, total=args.val_batches, desc=f"val {step:06d}", dynamic_ncols=True, leave=False)

    for item in iterator:
        if count >= args.val_batches:
            break
        if first_item is None and is_main_process():
            first_item = _first_item(item)
        with autocast_context(args, device):
            loss, logs = train_step(
                item,
                assembler,
                scene_flow,
                scheduler,
                device,
                args,
                text_encoder,
                generator=validation_generator,
            )
        logs = dict(logs)
        logs["loss"] = float(loss.detach().item())
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["batches"] = float(count)

    if is_main_process():
        image_paths: dict[str, Path] = {}
        if first_item is not None and args.val_log_images > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            visualization_batch = None
            try:
                visualization_batch = load_validation_visualization_batch(first_item, loader.dataset)
            except Exception as exc:
                print(
                    f"[validation {step:06d}] warning: failed to load RGB GT inputs: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            first_bundle = build_flow_bundle(
                first_item,
                assembler,
                device,
                args=args,
                scene_flow=scene_flow,
            )
            image_paths = save_validation_images(
                first_bundle,
                scene_flow,
                log_dir,
                step,
                args,
                device,
                text_encoder,
                visualization_batch=visualization_batch,
                vggt_model=vggt_model,
            )
        metrics_text = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"[validation {step:06d}] {metrics_text}", flush=True)
        log_wandb(wandb_run, metrics, step, "validation")
        if wandb_run is not None and image_paths:
            import wandb

            wandb_run.log(
                {f"validation/{name}": wandb.Image(str(path)) for name, path in image_paths.items()},
                step=step,
            )

    if use_val_ema:
        ema.restore(ema_params)
    if was_training:
        scene_flow.train()
        assembler.scaffold_packer.train()
    if is_distributed():
        dist.barrier()
    return metrics


def load_resume_checkpoint(
    scene_flow: nn.Module,
    assembler: FlowFeatureAssembler,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    resume_path: str | None,
    device: torch.device,
    args,
) -> int:
    if not resume_path:
        return 0
    payload = torch.load(resume_path, map_location=device)
    if not isinstance(payload, dict) or "scene_flow" not in payload:
        raise ValueError(f"Unsupported resume checkpoint format: {resume_path}")
    saved_flow_domain = payload.get("formal_flow_domain_version")
    if saved_flow_domain != FORMAL_FLOW_DOMAIN_VERSION:
        raise ValueError(
            f"{resume_path} formal_flow_domain_version={saved_flow_domain!r}, expected "
            f"{FORMAL_FLOW_DOMAIN_VERSION!r}. Legacy formal checkpoints used an inconsistent "
            "soft-mask flow path and cannot be resumed as mathematically equivalent training."
        )
    validate_formal_flow_domain_config(payload, args, resume_path)
    _validate_scene_flow_checkpoint_config(scene_flow, payload, resume_path)
    validate_checkpoint_flow_schedule(
        payload,
        args,
        resume_path,
        prediction_type=_scene_flow_prediction_type_from_module(scene_flow),
        t_eps=scene_flow_t_eps(scene_flow),
    )
    required_keys = {"step", "scene_flow", "scaffold_packer", "ema_scene_flow", "optimizer", "lr_scheduler"}
    missing_keys = sorted(required_keys.difference(payload.keys()))
    if missing_keys:
        raise ValueError(
            f"`--resume_path` requires a full training checkpoint, but {resume_path} "
            f"is missing keys: {missing_keys}. Do not pass *_weights_only.pt or "
            f"*_ema_weights_only.pt to --resume_path; those files are for warm-start "
            f"or inference, not exact training resume."
        )
    unwrap_ddp(scene_flow).load_state_dict(payload["scene_flow"], strict=True)
    unwrap_ddp(assembler.scaffold_packer).load_state_dict(payload["scaffold_packer"], strict=True)
    ema.load_state_dict(payload["ema_scene_flow"])
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    step = int(payload.get("step", 0))
    if is_main_process():
        print(f"[resume] loaded {resume_path} at step={step}", flush=True)
    return step


# ---------------------------------------------------------------------- #
# Main loop                                                               #
# ---------------------------------------------------------------------- #
def main() -> None:
    args = build_argparser().parse_args()
    args.sky_grid = sky_grid_shape(args)
    if not args.tokenizer_ckpt_path:
        raise ValueError(
            "Formal metric/gauge training requires an explicit --tokenizer_ckpt_path "
            "for content-hash verification"
        )
    if not args.feature_stats_path:
        raise ValueError(
            "Formal metric/gauge training requires --feature_stats_path"
        )
    if not args.pullback_calibration_path:
        raise ValueError(
            "Formal metric/gauge training requires --pullback_calibration_path"
        )
    if int(args.sequence_length) != FORMAL_TOKENIZER_WINDOW_LEN:
        raise ValueError(
            "Formal metric/gauge training must use the checkpoint-bound tokenizer "
            f"window {FORMAL_TOKENIZER_WINDOW_LEN}, got {args.sequence_length}"
        )
    if float(args.lambda_rgb_render) < 0.0:
        raise ValueError("--lambda_rgb_render must be non-negative.")
    if float(args.lambda_level_consistency) < 0.0:
        raise ValueError("--lambda_level_consistency must be non-negative.")
    if float(args.lambda_head_consistency) < 0.0:
        raise ValueError("--lambda_head_consistency must be non-negative.")
    if (
        float(args.lambda_level_consistency) > 0.0
        or float(args.lambda_head_consistency) > 0.0
    ) and not rgb_render_loss_enabled(args):
        raise ValueError(
            "Reconstruction feedback shares the RGB render schedule and requires "
            "--lambda_rgb_render > 0 with --rgb_render_every > 0. Set both feedback "
            "weights to zero when disabling the render path."
        )
    if int(args.rgb_render_every) < 0:
        raise ValueError("--rgb_render_every must be non-negative.")
    if int(args.rgb_render_start_step) < 0 or int(args.rgb_render_warmup_steps) < 0:
        raise ValueError("RGB render start/warmup steps must be non-negative.")
    if not math.isfinite(float(args.rgb_render_sigma_power)) or float(args.rgb_render_sigma_power) < 0.0:
        raise ValueError("--rgb_render_sigma_power must be finite and non-negative.")
    if (
        not math.isfinite(float(args.feedback_conf_weight_power))
        or float(args.feedback_conf_weight_power) < 0.0
    ):
        raise ValueError("--feedback_conf_weight_power must be finite and non-negative.")
    if (
        not math.isfinite(float(args.feedback_conf_weight_floor))
        or not 0.0 < float(args.feedback_conf_weight_floor) <= 1.0
    ):
        raise ValueError("--feedback_conf_weight_floor must be finite and in (0, 1].")
    if int(args.rgb_render_max_samples) < 0 or int(args.rgb_render_max_frames) < 0:
        raise ValueError("RGB render sample/frame limits must be non-negative.")
    if int(args.rgb_render_stride) <= 0:
        raise ValueError("--rgb_render_stride must be positive.")
    device, local_rank, world_size = setup_distributed(args)
    if int(args.num_workers) > 0:
        torch.multiprocessing.set_sharing_strategy(str(args.mp_sharing_strategy))
    seed_everything(args.seed + get_rank())

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest_path is None and args.cache_root is None:
        raise ValueError("Provide either --cache_root or --manifest_path.")

    mode_filter = (
        [m.strip() for m in args.mode_filter.split(",") if m.strip()]
        if args.mode_filter else None
    )
    enable_rgb_render_loss = rgb_render_loss_enabled(args)
    train_ds = WaymoFlowCacheDataset(
        cache_root=args.cache_root,
        manifest_path=args.manifest_path,
        mode_filter=mode_filter,
        split=args.split,
        min_frames=args.sequence_length,
        max_frames=args.sequence_length,
        seed=args.seed,
        mmap_plain_cache=not bool(args.no_mmap_plain_cache),
        asset_lut_level_indices=None,
        caption_root=args.caption_root,
        include_sky_training_data=False,
        include_rgb_training_data=enable_rgb_render_loss,
        require_edit_window=True,
        edit_domain_threshold=args.edit_domain_threshold,
        require_metric_gauge_provenance=True,
    )
    independent_val = args.val_manifest_path is not None or args.val_cache_root is not None
    val_caption_root = args.val_caption_root
    if independent_val:
        val_caption_root = val_caption_root if val_caption_root is not None else args.caption_root
        val_ds = WaymoFlowCacheDataset(
            cache_root=args.val_cache_root,
            manifest_path=args.val_manifest_path,
            mode_filter=mode_filter,
            split=args.val_split,
            min_frames=args.sequence_length,
            max_frames=args.sequence_length,
            seed=args.seed + 1,
            mmap_plain_cache=not bool(args.no_mmap_plain_cache),
            asset_lut_level_indices=None,
            caption_root=val_caption_root,
            include_sky_training_data=False,
            include_rgb_training_data=False,
            require_edit_window=True,
            edit_domain_threshold=args.edit_domain_threshold,
            deterministic_windows=True,
            require_metric_gauge_provenance=True,
        )
    else:
        if args.val_caption_root is not None:
            train_caption = Path(str(args.caption_root)).expanduser().resolve()
            val_caption = Path(str(args.val_caption_root)).expanduser().resolve()
            if val_caption != train_caption:
                raise ValueError(
                    "--val_caption_root can only differ from --caption_root when using "
                    "--val_manifest_path or --val_cache_root. Internal --val_fraction holdout "
                    "uses training cache entries and must use --caption_root captions."
                )
        val_ds = split_train_val_entries(
            train_ds,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    if is_main_process():
        print(
            f"[data] train_entries={len(train_ds.entries)} "
            f"val_entries={0 if val_ds is None else len(val_ds.entries)} "
            f"val_source={'independent' if independent_val else 'holdout'} "
            f"val_fraction={0.0 if independent_val else float(args.val_fraction):.3f} "
            "asset_lut_levels=all",
            flush=True,
        )
    val_loader = None
    if val_ds is not None and args.val_every > 0 and args.val_batches > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: batch,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            **dataloader_runtime_kwargs(args),
        )
    patch_grid = _infer_cache_patch_grid(train_ds)
    h_splat = patch_grid[0] * 4
    w_splat = patch_grid[1] * 4
    args.patch_grid = list(patch_grid)
    args.H_splat = int(h_splat)
    args.W_splat = int(w_splat)
    if is_main_process():
        (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(
            f"[train] cache patch_grid={patch_grid}, H_splat={h_splat}, W_splat={w_splat}",
            flush=True,
        )
    wandb_run = init_wandb(args, log_dir)

    render_vggt = None
    enable_val_rgb_render = (
        not bool(args.no_val_render_rgb)
        and val_loader is not None
        and int(args.val_log_images) > 0
    )
    if enable_rgb_render_loss or (is_main_process() and enable_val_rgb_render):
        render_vggt = load_dggt_aggregator_and_tokenizer(
            args.ckpt_path,
            args.tokenizer_ckpt_path,
            device,
        )
        render_vggt.scene_tokenizer.float()
        if is_main_process() and enable_val_rgb_render:
            print("[validation] 3DGS RGB rendering enabled on rank 0.", flush=True)
        if is_main_process() and enable_rgb_render_loss:
            print(
                "[train] deployment-aligned generated-depth RGB supervision enabled.",
                flush=True,
            )
    lpips_model = setup_lpips_for_rgb_loss(args, device)

    tokenizer = (
        render_vggt.scene_tokenizer
        if render_vggt is not None
        else _load_tokenizer(args.tokenizer_ckpt_path or args.ckpt_path, device)
    )
    freeze_module(tokenizer)  # T1: encoder frozen; decoder layer_heads/local_refine can be unfrozen later.

    # Assembler: scaffold_packer + feature_splatter + soft_mask + noise_scheduler trainable.
    assembler = FlowFeatureAssembler(
        scene_tokenizer=tokenizer,
        patch_grid=patch_grid,
        H_splat=h_splat,
        W_splat=w_splat,
        scaffold_out_dim=int(args.latent_dim),
        tokenizer_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    # Freeze inner editor / soft_mask (no params), scaffold packer trainable.
    freeze_module(assembler.editor)
    freeze_module(assembler.soft_mask)  # no params but safe.
    freeze_module(assembler.feature_splatter)

    architecture_checkpoint = args.resume_path or args.scene_flow_pretrain_path
    if not architecture_checkpoint:
        raise ValueError(
            "Formal SceneFlow training requires --scene_flow_pretrain_path (new pretrain checkpoint) "
            "or --resume_path; constructing a default camera architecture is not supported."
        )
    scene_flow = build_scene_flow_from_checkpoint_config(
        architecture_checkpoint,
        patch_grid=patch_grid,
        latent_dim=int(args.latent_dim),
        device=device,
    )
    args.dggt_checkpoint_sha256 = checkpoint_sha256(args.ckpt_path)
    args.tokenizer_checkpoint_sha256 = checkpoint_sha256(args.tokenizer_ckpt_path)
    architecture_payload = torch.load(architecture_checkpoint, map_location="cpu")
    base_provenance = validate_metric_gauge_provenance(
        architecture_payload.get("metric_gauge_provenance")
        if isinstance(architecture_payload, Mapping)
        else None,
        config=scene_flow.config,
        expected_dggt_sha256=args.dggt_checkpoint_sha256,
        expected_tokenizer_sha256=args.tokenizer_checkpoint_sha256,
        expected_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=patch_grid,
    )
    args.feature_stats_sha256 = validate_formal_feature_stats_artifact(
        args.feature_stats_path,
        provenance=base_provenance,
        expected_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=patch_grid,
    )
    base_provenance, existing_formal_contract = validate_metric_gauge_checkpoint_payload(
        architecture_payload,
        path=architecture_checkpoint,
        config=scene_flow.config,
        require_formal_contract=bool(args.resume_path),
        expected_dggt_sha256=args.dggt_checkpoint_sha256,
        expected_tokenizer_sha256=args.tokenizer_checkpoint_sha256,
        expected_feature_stats_sha256=args.feature_stats_sha256,
        expected_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=patch_grid,
    )
    train_ds.bind_metric_gauge_provenance(
        expected_scene_gauge_sha256=base_provenance["gauge_table_sha256"],
        expected_dggt_sha256=args.dggt_checkpoint_sha256,
    )
    if val_ds is not None:
        if independent_val:
            if args.val_scene_gauge_sha256 is None:
                raise ValueError(
                    "Independent formal validation requires --val_scene_gauge_sha256 "
                    "so its cache provenance is checked against a trusted table."
                )
            val_gauge_sha256 = args.val_scene_gauge_sha256
        else:
            val_gauge_sha256 = base_provenance["gauge_table_sha256"]
        val_ds.bind_metric_gauge_provenance(
            expected_scene_gauge_sha256=val_gauge_sha256,
            expected_dggt_sha256=args.dggt_checkpoint_sha256,
        )
    formal_metric_gauge_contract = (
        existing_formal_contract
        if existing_formal_contract is not None
        else build_formal_metric_gauge_contract(
            base_provenance,
            feature_stats_sha256=args.feature_stats_sha256,
        )
    )
    pullback_calibration = load_pullback_calibration(
        args.pullback_calibration_path,
        tokenizer_checkpoint_path=args.tokenizer_ckpt_path,
        dggt_checkpoint_path=args.ckpt_path,
        expected_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=patch_grid,
        expected_artifact_sha256=base_provenance["pullback_artifact_sha256"],
    )
    if (
        base_provenance["pullback_runtime_contract_version"]
        != pullback_calibration.runtime_contract_version
    ):
        raise ValueError(
            "checkpoint pullback runtime contract does not match the loaded artifact: "
            f"checkpoint={base_provenance['pullback_runtime_contract_version']!r}, "
            f"artifact={pullback_calibration.runtime_contract_version!r}"
        )
    scene_flow._metric_gauge_provenance = copy.deepcopy(base_provenance)
    scene_flow._formal_metric_gauge_contract = copy.deepcopy(
        formal_metric_gauge_contract
    )
    scene_flow._pullback_calibration = pullback_calibration
    if is_main_process():
        print(
            "[pullback] verified formal render identity contract "
            f"{pullback_calibration.path} "
            f"(sha256={pullback_calibration.artifact_sha256})",
            flush=True,
        )
    if str(scene_flow.config.asset_position_mode) != str(args.asset_position_mode):
        raise ValueError(
            f"checkpoint asset_position_mode={scene_flow.config.asset_position_mode!r} "
            f"!= --asset_position_mode={args.asset_position_mode!r}"
        )
    if bool(args.three_quarter_gradient_checkpointing):
        scene_flow.enable_three_quarter_gradient_checkpointing()
    elif bool(args.half_gradient_checkpointing):
        scene_flow.enable_half_gradient_checkpointing()
    elif bool(args.gradient_checkpointing):
        scene_flow.enable_gradient_checkpointing()
    else:
        scene_flow.disable_gradient_checkpointing()
    if is_main_process():
        print(
            "[memory] SceneFlow gradient checkpointing "
            f"mode={scene_flow.gradient_checkpointing_mode} "
            f"encoder_blocks={len(scene_flow.checkpointed_block_indices(len(scene_flow.blocks)))}/{len(scene_flow.blocks)} "
            f"ddt_blocks={len(scene_flow.checkpointed_block_indices(len(scene_flow.ddt_head), block_group='ddt'))}/{len(scene_flow.ddt_head)}",
            flush=True,
        )
    text_encoder = setup_text_encoder(args, device)
    warm_start_info = load_scene_flow_warm_start(
        scene_flow,
        args.scene_flow_pretrain_path,
        use_ema=bool(args.scene_flow_pretrain_ema),
        args=args,
    )
    if is_main_process() and warm_start_info is not None:
        print(f"[warm-start] {warm_start_info}", flush=True)

    ema = EMAModel(scene_flow.parameters(), decay=args.ema_decay)
    ema.to(device)
    # DDP broadcasts rank-0 module parameters in its constructor.  Non-resume
    # runs must rebuild the EMA after that broadcast so rank-local random
    # initialization cannot survive in EMA shadow params.  Exact resume loads a
    # checkpointed EMA below and must preserve it.
    sync_ema_after_ddp_initial_broadcast = not bool(args.resume_path)

    if world_size > 1:
        scene_flow = DistributedDataParallel(
            scene_flow,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
        assembler.scaffold_packer = DistributedDataParallel(
            assembler.scaffold_packer,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
    if sync_ema_after_ddp_initial_broadcast:
        sync_ema_shadow_from_model(scene_flow, ema)

    clip_params = list(unwrap_ddp(scene_flow).parameters()) + list(
        unwrap_ddp(assembler.scaffold_packer).parameters()
    )
    scene_decay, scene_no_decay = split_param_groups(unwrap_ddp(scene_flow))
    scaffold_decay, scaffold_no_decay = split_param_groups(unwrap_ddp(assembler.scaffold_packer))
    optimizer, optimizer_msg = build_rae_optimizer(
        [
            {"params": scene_decay + scaffold_decay, "weight_decay": args.weight_decay},
            {"params": scene_no_decay + scaffold_no_decay, "weight_decay": 0.0},
        ],
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.gmuon_momentum,
        nesterov=args.gmuon_nesterov,
        ns_coefficients_preset=args.gmuon_ns_coefficients_preset,
        ns_use_kernels=args.gmuon_ns_use_kernels,
    )
    lr_scheduler = build_training_scheduler(optimizer, args)
    if is_main_process():
        decay_end_steps = int(args.decay_end_steps) if int(args.decay_end_steps) > 0 else int(args.max_steps)
        print(
            f"[optim] {optimizer_msg}; scheduler={args.scheduler_type} "
            f"warmup={args.warmup_steps} decay_end={decay_end_steps} "
            f"lr={args.lr}->{args.final_lr}",
            flush=True,
        )
    flow_scheduler = None
    global_step = load_resume_checkpoint(
        scene_flow,
        assembler,
        ema,
        optimizer,
        lr_scheduler,
        args.resume_path,
        device,
        args,
    )
    load_all_stats_into_buffers(
        unwrap_ddp(scene_flow),
        args.feature_stats_path,
        token_dim=int(args.latent_dim),
        expected_dggt_sha256=args.dggt_checkpoint_sha256,
        expected_tokenizer_sha256=args.tokenizer_checkpoint_sha256,
        expected_scene_gauge_sha256=base_provenance["gauge_table_sha256"],
        expected_sequence_length=FORMAL_TOKENIZER_WINDOW_LEN,
        require_existing_match=True,
    )
    if is_main_process():
        checkpoint_kind = "resume" if args.resume_path else "warm-start"
        print(
            f"[stats] verified latent+camera+gauge+placement stats against the effective "
            f"{checkpoint_kind} checkpoint and loaded {args.feature_stats_path} "
            f"(sha256={args.feature_stats_sha256})",
            flush=True,
        )

    sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        drop_last=True,
        **dataloader_runtime_kwargs(args),
    )

    accum_count = 0
    scene_flow.train()
    assembler.scaffold_packer.train()
    optimizer.zero_grad(set_to_none=True)
    wandb_sums: dict[str, float] = {}
    wandb_count = 0
    accum_data_wait_s = 0.0
    accum_train_wall_s = 0.0
    progress = None
    if is_main_process() and not args.no_tqdm:
        progress = tqdm(total=args.max_steps, initial=global_step, desc="train", dynamic_ncols=True)
    try:
        while global_step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(global_step)
            data_wait_t0 = time.perf_counter()
            for item in loader:
                data_wait_s = time.perf_counter() - data_wait_t0
                if global_step >= args.max_steps:
                    break
                micro_t0 = time.perf_counter()
                sync_grad = (accum_count + 1) % max(1, args.grad_accum_steps) == 0
                with ExitStack() as stack:
                    if isinstance(scene_flow, DistributedDataParallel) and not sync_grad:
                        stack.enter_context(scene_flow.no_sync())
                    if isinstance(assembler.scaffold_packer, DistributedDataParallel) and not sync_grad:
                        stack.enter_context(assembler.scaffold_packer.no_sync())
                    with autocast_context(args, device):
                        loss, metrics = train_step(
                            item,
                            assembler,
                            scene_flow,
                            flow_scheduler,
                            device,
                            args,
                            text_encoder,
                            global_step=global_step,
                            render_vggt_model=render_vggt,
                            lpips_model=lpips_model,
                        )
                        loss = loss / max(1, args.grad_accum_steps)
                    loss.backward()
                micro_wall_s = time.perf_counter() - micro_t0
                accum_data_wait_s += float(data_wait_s)
                accum_train_wall_s += float(micro_wall_s)
                accum_count += 1
                data_wait_t0 = time.perf_counter()
                if not sync_grad:
                    continue

                optim_t0 = time.perf_counter()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.step(unwrap_ddp(scene_flow).parameters())
                optim_s = time.perf_counter() - optim_t0
                data_wait_step_s = accum_data_wait_s
                train_wall_step_s = accum_train_wall_s
                step_wall_s = data_wait_step_s + train_wall_step_s + float(optim_s)
                accum_count = 0
                accum_data_wait_s = 0.0
                accum_train_wall_s = 0.0
                global_step += 1

                if is_main_process():
                    lr_now = float(optimizer.param_groups[0]["lr"])
                    train_metrics = dict(metrics)
                    train_metrics["lr"] = lr_now
                    train_metrics["data_wait_s"] = float(data_wait_step_s)
                    train_metrics["train_wall_s"] = float(train_wall_step_s)
                    train_metrics["optim_s"] = float(optim_s)
                    train_metrics["step_wall_s"] = float(step_wall_s)
                    train_metrics["data_wait_frac"] = (
                        float(data_wait_step_s / step_wall_s) if step_wall_s > 0.0 else 0.0
                    )
                    micro_bs = float(metrics.get("micro_batch_size", args.batch_size))
                    train_metrics["items_per_s_per_rank"] = (
                        micro_bs * max(1, int(args.grad_accum_steps)) / step_wall_s
                        if step_wall_s > 0.0
                        else 0.0
                    )
                    if progress is not None:
                        postfix = _format_train_progress_metrics(train_metrics)
                        progress.set_postfix(postfix, refresh=False)
                    elif global_step % max(1, int(args.log_every)) == 0:
                        metrics_str = _format_train_progress_line(train_metrics)
                        print(f"[step {global_step:06d}] {metrics_str}", flush=True)
                    for key, value in train_metrics.items():
                        wandb_sums[key] = wandb_sums.get(key, 0.0) + float(value)
                    wandb_count += 1
                    if wandb_run is not None and wandb_count >= max(1, int(args.wandb_log_every)):
                        averaged = {key: value / wandb_count for key, value in wandb_sums.items()}
                        log_wandb(wandb_run, averaged, global_step, "train")
                        wandb_sums = {}
                        wandb_count = 0

                if is_main_process() and args.vis_every > 0 and (global_step % args.vis_every == 0):
                    _dump_vis(_first_item(item), assembler, log_dir, global_step, device, args)

                if (
                    val_loader is not None
                    and args.val_every > 0
                    and args.val_batches > 0
                    and global_step % args.val_every == 0
                ):
                    run_validation(
                        loader=val_loader,
                        assembler=assembler,
                        scene_flow=scene_flow,
                        scheduler=flow_scheduler,
                        device=device,
                        args=args,
                        step=global_step,
                        log_dir=log_dir,
                        wandb_run=wandb_run,
                        ema=ema,
                        text_encoder=text_encoder,
                        vggt_model=render_vggt,
                    )

                if global_step > 0 and global_step % args.save_every == 0:
                    if is_distributed():
                        dist.barrier()
                    if is_main_process():
                        _save_checkpoint(scene_flow, assembler, ema, optimizer, lr_scheduler, global_step, log_dir, args)
                    if is_distributed():
                        dist.barrier()

                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if is_distributed():
        dist.barrier()
    if is_main_process():
        _save_checkpoint(scene_flow, assembler, ema, optimizer, lr_scheduler, global_step, log_dir, args)
        if wandb_run is not None:
            wandb_run.finish()
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def _dump_vis(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    log_dir: Path,
    step: int,
    device: torch.device,
    args,
) -> None:
    from dggt.utils.flow_viz import dump_flow_features

    if item.get("flow_inputs_cached") is not None:
        if is_main_process():
            print(
                f"[vis] skipping full flow feature dump for fast cache item at step={step}; "
                "fast items do not load raw heads or asset Gaussians.",
                flush=True,
            )
        return

    vis_dir = log_dir / "vis" / f"step_{step:06d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["sample"].items()}
        predictions = _move_predictions(item["predictions"], device)
        apr = _move_asset_pass(item["asset_pass_result"], device)
        _validate_item_patch_grid(apr, assembler, item.get("cache_path"))
        cams = {k: v.to(device) for k, v in item["cameras_dggt"].items()}
        mode_kind = str(item.get("mode_kind", sample.get("mode_kind", "mode_a")))
        mode_b_payload = item.get("mode_b")
        if mode_b_payload is not None:
            mode_b_payload = _move_mode_b(mode_b_payload, device)
        phase1_localized_lite, splatted_tok_low_cached = _move_v6_fast_path_inputs(
            item, mode_kind, device
        )
        bundle = assembler(
            sample=sample,
            predictions=predictions,
            asset_pass_result=apr,
            cameras_dggt=cams,
            object_slots_spec="all",
            device=device,
            mode_kind=mode_kind,
            mode_b=mode_b_payload,
            phase1_localized_lite=phase1_localized_lite,
            splatted_tok_low_cached=splatted_tok_low_cached,
        )
    dump_flow_features(bundle, vis_dir, save_splat_pca=False)


def _save_checkpoint(
    scene_flow: nn.Module,
    assembler: FlowFeatureAssembler,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    step: int,
    log_dir: Path,
    args,
) -> None:
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sf = unwrap_ddp(scene_flow)
    scene_flow_state = sf.state_dict()
    ema_scene_flow_state = materialize_ema_state_dict(scene_flow, ema)
    # ScaffoldPacker is optimized during formal training, but is intentionally
    # not part of the SceneFlow EMA (matching validation, which uses EMA
    # SceneFlow weights together with the current trained packer).  Every
    # inference-oriented export must therefore carry this state explicitly;
    # otherwise a weights-only checkpoint silently falls back to a freshly
    # initialized packer and changes the edit-control conditioning.
    scaffold_packer_state = unwrap_ddp(assembler.scaffold_packer).state_dict()
    scene_flow_config = sf.config.to_dict() if hasattr(sf, "config") and hasattr(sf.config, "to_dict") else {}
    flow_schedule_config = build_flow_schedule_config(
        args,
        prediction_type=_scene_flow_prediction_type_from_module(sf),
        t_eps=scene_flow_t_eps(sf),
    )
    flow_domain_config = formal_flow_domain_config(args)
    provenance = copy.deepcopy(getattr(sf, "_metric_gauge_provenance", None))
    pullback_calibration = require_formal_pullback_calibration(sf)
    formal_metric_gauge_contract = copy.deepcopy(
        getattr(sf, "_formal_metric_gauge_contract", None)
    )
    provenance = validate_metric_gauge_provenance(
        provenance,
        config=scene_flow_config,
        expected_dggt_sha256=getattr(args, "dggt_checkpoint_sha256", None),
        expected_tokenizer_sha256=getattr(
            args, "tokenizer_checkpoint_sha256", None
        ),
        expected_pullback_runtime_contract_version=(
            pullback_calibration.runtime_contract_version
        ),
        expected_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        expected_patch_grid=scene_flow_config.get("patch_grid"),
    )
    formal_metric_gauge_contract = validate_formal_metric_gauge_contract(
        formal_metric_gauge_contract,
        provenance=provenance,
        expected_feature_stats_sha256=getattr(
            args, "feature_stats_sha256", None
        ),
    )
    state = {
        "step": int(step),
        "scene_flow": scene_flow_state,
        "scene_flow_config": scene_flow_config,
        "flow_schedule_config": flow_schedule_config,
        "ema_scene_flow": ema.state_dict(),
        "ema_scene_flow_state_dict": ema_scene_flow_state,
        "scaffold_packer": scaffold_packer_state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": vars(args),
        "metric_gauge_provenance": provenance,
        "formal_metric_gauge_contract": formal_metric_gauge_contract,
        "formal_flow_domain_version": FORMAL_FLOW_DOMAIN_VERSION,
        "formal_flow_domain_config": flow_domain_config,
    }
    torch.save(state, ckpt_dir / f"flow_step{step:06d}.pt")
    torch.save(
        {
            "scene_flow": scene_flow_state,
            "scaffold_packer": scaffold_packer_state,
            "scene_flow_config": scene_flow_config,
            "flow_schedule_config": flow_schedule_config,
            "metric_gauge_provenance": provenance,
            "formal_metric_gauge_contract": formal_metric_gauge_contract,
            "formal_flow_domain_version": FORMAL_FLOW_DOMAIN_VERSION,
            "formal_flow_domain_config": flow_domain_config,
        },
        ckpt_dir / f"flow_step{step:06d}_weights_only.pt",
    )
    torch.save(
        {
            "scene_flow": ema_scene_flow_state,
            "scaffold_packer": scaffold_packer_state,
            "scene_flow_config": scene_flow_config,
            "flow_schedule_config": flow_schedule_config,
            "step": int(step),
            "is_ema_weights": True,
            "metric_gauge_provenance": provenance,
            "formal_metric_gauge_contract": formal_metric_gauge_contract,
            "formal_flow_domain_version": FORMAL_FLOW_DOMAIN_VERSION,
            "formal_flow_domain_config": flow_domain_config,
        },
        ckpt_dir / f"flow_step{step:06d}_ema_weights_only.pt",
    )


if __name__ == "__main__":
    main()
