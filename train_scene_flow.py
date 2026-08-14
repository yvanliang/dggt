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
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
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
from dggt.models.joint_scene_tokenizer import JointSceneTokenizer
from dggt.models.embedders.text_encoder import TextEncoder
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.feature_quant import QuantizedTokens, dequantize_tokens
from dggt.utils.feature_stats import (
    DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH,
    load_feature_stats,
)
from dggt.utils.camera_condition import (
    camera_condition_from_waymo_request,
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
from dggt.utils.sliding_window import cosine_window, default_window_stride, window_slices
from dggt.utils.tokens import reattach_special_tokens, replace_selected_levels, select_patch_pyramid
from dggt.utils.tokenizer_checkpoint import load_scene_tokenizer_checkpoint_strict
from dggt.utils.tokenizer_window import (
    decode_tokenizer_windowed,
    encode_tokenizer_windowed,
)
from dggt.utils.validation_rng import make_validation_generator, preserve_validation_rng_state
from diffusers.training_utils import EMAModel
from train_scene_flow_pretrain import (
    TOKENIZER_LEVELS,
    _image_grid,
    _latent_pca_grid,
    _mask_grid,
    _normalized_mask_grid,
    _render_gs_map_rgb,
    _semantic_logits_to_sky_mask,
    _sky_mask_image_grid,
    load_dggt_aggregator_and_tokenizer,
    split_image_tokens_for_heads,
)


FORMAL_FLOW_DOMAIN_VERSION = "hard_binary_edit_domain_v1"
# All formal caches currently come from the 10 Hz Waymo camera stream. Keep
# mRoPE time coordinates identical to pretraining.
FORMAL_SCENE_FPS = 10.0
FORMAL_DGGT_CONTEXT_LENGTH = 29
FORMAL_TOKENIZER_WINDOW_LEN = 10
FORMAL_LAYOUT_CONDITION_VERSION = "none"
FORMAL_LAYOUT_DISABLED_CONFIG = {
    "layout_map_injection": False,
    "layout_actor_injection": False,
    "layout_map_metric_injection": False,
    "layout_actor_metric_injection": False,
    "appearance_context_injection": False,
    "layout_to_gauge_grad_scale": 0.0,
}


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


def build_camera_condition_from_waymo_request(
    camera_to_world: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    *,
    device: torch.device,
    image_hw: tuple[int, int] | None = None,
    trajectory_anchor_to_world: torch.Tensor | None = None,
    previous_camera_to_world: torch.Tensor | None = None,
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
    condition, valid = camera_condition_from_waymo_request(
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
    )
    return condition, valid


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


def _scene_flow_prediction_type_from_module(scene_flow: nn.Module) -> str:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return str(getattr(cfg, "prediction_type", "x"))


def load_formal_latent_stats(
    scene_flow: nn.Module,
    path: str | Path,
    *,
    token_dim: int,
    require_existing_match: bool,
) -> None:
    """Load the only feature statistics consumed by the layout-free editor."""
    mean, std = load_feature_stats(path, token_dim=int(token_dim))
    root = unwrap_ddp(scene_flow)
    if require_existing_match:
        current_mean = mean.to(device=root.mu_z.device, dtype=root.mu_z.dtype)
        current_std = std.to(
            device=root.sigma_z.device, dtype=root.sigma_z.dtype
        ).clamp_min(1.0e-4)
        if not torch.equal(root.mu_z, current_mean) or not torch.equal(root.sigma_z, current_std):
            raise ValueError(
                f"{path} latent statistics do not match the exact-resume checkpoint buffers"
            )
    root.set_latent_stats(mean, std)


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
            f"{checkpoint_path} has no scene_flow_config; only current formal checkpoints are accepted."
        )
    config = dict(payload["scene_flow_config"])
    if config.get("layout_condition_version") != FORMAL_LAYOUT_CONDITION_VERSION:
        raise ValueError(
            f"{checkpoint_path} layout_condition_version={config.get('layout_condition_version')!r}; "
            "the first formal editor requires its independent from-scratch 'none' model and "
            "must not silently ignore a layout_v2 checkpoint."
        )
    for field, expected in FORMAL_LAYOUT_DISABLED_CONFIG.items():
        if config.get(field) != expected:
            raise ValueError(
                f"{checkpoint_path} {field}={config.get(field)!r}; "
                f"layout-free formal checkpoints require {expected!r}"
            )
    if tuple(config.get("patch_grid", ())) != tuple(patch_grid):
        raise ValueError(f"checkpoint patch_grid={config.get('patch_grid')} != cache patch_grid={patch_grid}")
    if int(config.get("out_channels", -1)) != int(latent_dim):
        raise ValueError(f"checkpoint out_channels={config.get('out_channels')} != --latent_dim={latent_dim}")
    for derived in (
        "hidden_size",
        "rope_layout_version",
        "sky_rope_temporal_offset",
        "camera_rope_spatial_mode",
    ):
        config.pop(derived, None)
    return WanSceneFlow(**config).to(device)


def _validate_scene_flow_checkpoint_config(scene_flow: nn.Module, payload: Any, path: str | Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a current formal SceneFlow checkpoint")
    saved_cfg = payload.get("scene_flow_config")
    if not isinstance(saved_cfg, dict):
        raise ValueError(f"{path} is missing scene_flow_config")
    if saved_cfg.get("layout_condition_version") != FORMAL_LAYOUT_CONDITION_VERSION:
        raise ValueError(
            f"{path} is not a layout-free formal-editor checkpoint; refusing cross-task loading"
        )
    current_cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    for field in (
        "layout_condition_version",
        *FORMAL_LAYOUT_DISABLED_CONFIG,
        "patch_grid",
        "out_channels",
        "prediction_type",
    ):
        current_value = getattr(current_cfg, field, None)
        saved_value = saved_cfg.get(field)
        same = (
            tuple(current_value) == tuple(saved_value)
            if field == "patch_grid"
            else current_value == saved_value
        )
        if not same:
            raise ValueError(
                f"{path} config {field}={saved_value!r} != active model {current_value!r}"
            )
    if payload.get("formal_flow_domain_version") != FORMAL_FLOW_DOMAIN_VERSION:
        raise ValueError(
            f"{path} formal_flow_domain_version is not {FORMAL_FLOW_DOMAIN_VERSION!r}"
        )


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
                            "Tokenizer latent mean/std. Exact resume requires these values "
                            "to match the checkpoint buffers."
                        ))
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=1024,
        help=(
            "Tokenizer latent channels. SceneFlow consumes z_t directly."
        ),
    )
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
    parser.add_argument("--guidance_scale", type=float, default=1.0)
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
    *,
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

    frame_ids = _frame_ids_from_item(
        item,
        seq_len=int(z_clean.shape[1]),
        device=device,
        flow_inputs=flow_inputs,
    )
    camera_gt = item.get("camera_gt") or {}
    camera_condition_tokens, camera_attention_mask = build_camera_condition_from_waymo_request(
        camera_gt.get("camera_to_world_corrected"),
        camera_gt.get("intrinsics"),
        device=device,
        image_hw=camera_gt.get("raw_image_size_hw"),
        trajectory_anchor_to_world=camera_gt.get("trajectory_anchor_to_world"),
        previous_camera_to_world=camera_gt.get("previous_camera_to_world"),
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
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        captions=[str(item.get("caption", ""))],
        patch_grid=assembler.patch_grid,
        patch_start_idx=assembler.patch_start_idx,
        splatted_tok_low=splatted_tok_low,
        F_g_lut_scene=F_g_lut_scene,
        frame_ids=frame_ids,
    )
    if include_rgb_render_context:
        bundle = _attach_rgb_context(bundle, _formal_rgb_context(item, device=device, strict=True))
    return bundle


def build_cached_flow_batch_bundle(
    items: list[dict[str, Any]],
    assembler: FlowFeatureAssembler,
    device: torch.device,
    *,
    include_rgb_render_context: bool = False,
) -> Any:
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
    for item in items:
        camera_gt = item.get("camera_gt") or {}
        camera_tokens_i, camera_mask_i = build_camera_condition_from_waymo_request(
            camera_gt.get("camera_to_world_corrected"),
            camera_gt.get("intrinsics"),
            device=device,
            image_hw=camera_gt.get("raw_image_size_hw"),
            trajectory_anchor_to_world=camera_gt.get("trajectory_anchor_to_world"),
            previous_camera_to_world=camera_gt.get("previous_camera_to_world"),
        )
        if camera_tokens_i is None:
            raise RuntimeError(
                f"Formal SceneFlow cache item {item.get('cache_path', '<unknown>')} "
                "is missing Waymo camera_gt; camera conditioning must use Waymo camera parameters."
            )
        camera_token_rows.append(camera_tokens_i)
        camera_mask_rows.append(
            camera_mask_i
            if camera_mask_i is not None
            else torch.ones(camera_tokens_i.shape[:2], device=device, dtype=torch.bool)
        )
    camera_condition_tokens = torch.cat(camera_token_rows, dim=0)
    camera_attention_mask = torch.cat(camera_mask_rows, dim=0)
    frame_ids = torch.cat(frame_id_rows, dim=0)

    merged = SimpleNamespace(
        z_clean=z_clean,
        z_splat=z_splat,
        scaffold_tok=scaffold_tok,
        M_preserve=M_preserve,
        M_source=M_source,
        M_dest=M_dest,
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
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
    return merged


# ---------------------------------------------------------------------- #
# Train step                                                              #
# ---------------------------------------------------------------------- #
def build_flow_bundle(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    device: torch.device,
    *,
    include_rgb_render_context: bool = False,
) -> Any:
    if item.get("flow_inputs_cached") is not None:
        return build_cached_flow_bundle(
            item,
            assembler,
            device,
            include_rgb_render_context=include_rgb_render_context,
        )
    raise RuntimeError(
        "The clean-cut formal editor requires flow_inputs_cached; rebuild this cache item "
        "with the current offline feature pipeline."
    )


def _merge_bundles_for_scene_flow(bundles: list[Any]) -> Any:
    if not bundles:
        raise ValueError("Cannot merge an empty formal-editor batch")
    camera_tokens_list = [getattr(bundle, "camera_condition_tokens", None) for bundle in bundles]
    camera_masks = [getattr(bundle, "camera_attention_mask", None) for bundle in bundles]
    if not all(torch.is_tensor(tokens) for tokens in camera_tokens_list):
        raise ValueError("Every formal-editor item must carry requested camera tokens")
    if not all(torch.is_tensor(mask) for mask in camera_masks):
        raise ValueError("Every formal-editor item must carry a requested camera mask")
    camera_condition_tokens = torch.cat(camera_tokens_list, dim=0)
    camera_attention_mask = torch.cat(camera_masks, dim=0)
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
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        frame_ids=frame_ids,
        captions=[caption for bundle in bundles for caption in getattr(bundle, "captions", [""])],
        patch_grid=bundles[0].patch_grid,
        patch_start_idx=bundles[0].patch_start_idx,
    )
    merged = _attach_rgb_context(merged, _merge_rgb_contexts(bundles))
    return merged


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
        sky_grid=(16, 32),
        patch_weight_mask=target.M_edit,
        lpips_model=lpips_model,
        lpips_weight=float(args.rgb_render_lpips_weight),
        loss_sample_weight=sigma_weights,
        conf_weight_power=float(
            getattr(args, "feedback_conf_weight_power", 1.0)
        ),
        conf_weight_floor=float(
            getattr(args, "feedback_conf_weight_floor", 0.05)
        ),
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
        if not item:
            raise ValueError("Received an empty training micro-batch.")
        if len(item) > 1 and not bool(getattr(args, "no_batch_scene_flow", False)):
            if all(single.get("flow_inputs_cached") is not None for single in item):
                bundle = build_cached_flow_batch_bundle(
                    item,
                    assembler,
                    device,
                    include_rgb_render_context=rgb_render_active,
                )
            else:
                bundle = _merge_bundles_for_scene_flow([
                    build_flow_bundle(
                        single,
                        assembler,
                        device,
                        include_rgb_render_context=rgb_render_active,
                    )
                    for single in item
                ])
        else:
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
    else:
        bundle = build_flow_bundle(
            item,
            assembler,
            device,
            include_rgb_render_context=rgb_render_active,
        )

    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean)
    z_splat_n = sf.normalize(bundle.z_splat)
    bundle.z_clean_n = z_clean_n
    text_drop_mask = sample_uncond_drop_mask(
        int(bundle.z_clean.shape[0]),
        args.uncond_drop_prob,
        device=bundle.z_clean.device,
        training=unwrap_ddp(scene_flow).training,
    )

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
        text_tokens=text_tokens,
        text_attention_mask=text_mask,
        camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
        camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
        return_mid=use_repa,
        return_base=float(args.base_model_coeff) != 0.0,
        return_dict=True,
        frame_ids=getattr(bundle, "frame_ids", None),
        fps=FORMAL_SCENE_FPS,
        flow_edit_mask=M_edit,
    )
    pred_clean = out["video"]
    pred_base = out.get("video_base")
    mid_repa = out.get("mid_repa") if use_repa else None
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
        "control_token_count": estimate_control_token_count(
            sf,
            bundle.M_source,
            bundle.M_dest,
            getattr(bundle, "patch_grid", getattr(args, "patch_grid", (25, 37))),
        ),
        "sigma_mean": float(target.sigmas.float().mean().item()),
    })
    if isinstance(item, list):
        metrics["micro_batch_size"] = float(len(item))
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

    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(text_encoder, [""] * batch_size if text_tokens is not None else None)
    do_cfg = abs(float(guidance_scale) - 1.0) > 1e-6
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
            out_full = sf(
                z_w,
                sigma,
                z_splat_w,
                scaffold_w,
                M_preserve_w,
                M_source_w,
                M_dest_w,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=camera_tokens_w,
                camera_attention_mask=camera_mask_w,
                return_mid=False,
                return_dict=True,
                frame_ids=frame_ids_w,
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit_w,
            )
            pred = out_full["video"]
            if do_cfg:
                out_null = sf(
                    z_w,
                    sigma,
                    z_splat_w,
                    scaffold_w,
                    M_preserve_w,
                    M_source_w,
                    M_dest_w,
                    text_tokens=text_null,
                    text_attention_mask=text_null_mask,
                    camera_condition_tokens=camera_tokens_w,
                    camera_attention_mask=camera_mask_w,
                    return_mid=False,
                    return_dict=True,
                    frame_ids=frame_ids_w,
                    fps=FORMAL_SCENE_FPS,
                    flow_edit_mask=M_edit_w,
                )
                pred_null = out_null["video"]
                pred = pred_null + float(guidance_scale) * (pred - pred_null)
            v = sampler_prediction_to_velocity(sf, pred, z_w, sigma)
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
    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(text_encoder, [""] * batch_size if text_tokens is not None else None)
    do_cfg = abs(float(guidance_scale) - 1.0) > 1e-6

    for i in range(int(args.val_sample_steps)):
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)
        out_full = sf(
            z,
            sigma,
            z_splat_n,
            bundle.scaffold_tok,
            bundle.M_preserve,
            bundle.M_source,
            bundle.M_dest,
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
            camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
            return_mid=False,
            return_dict=True,
            frame_ids=frame_ids,
            fps=FORMAL_SCENE_FPS,
            flow_edit_mask=M_edit,
        )
        pred = out_full["video"]
        if do_cfg:
            out_null = sf(
                z,
                sigma,
                z_splat_n,
                bundle.scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                text_tokens=text_null,
                text_attention_mask=text_null_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                return_mid=False,
                return_dict=True,
                frame_ids=frame_ids,
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit,
            )
            pred_null = out_null["video"]
            pred = pred_null + float(guidance_scale) * (pred - pred_null)
        v = sampler_prediction_to_velocity(sf, pred, z, sigma)
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
            f"{FORMAL_FLOW_DOMAIN_VERSION!r}. Earlier formal checkpoints used an inconsistent "
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
            f"*_ema_weights_only.pt to --resume_path; those files are for inference, "
            "not exact training resume."
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
    if not args.tokenizer_ckpt_path:
        raise ValueError(
            "Formal editor training requires an explicit --tokenizer_ckpt_path"
        )
    if not args.feature_stats_path:
        raise ValueError("Formal editor training requires --feature_stats_path")
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
        caption_root=args.caption_root,
        include_sky_training_data=False,
        include_rgb_training_data=enable_rgb_render_loss,
        require_edit_window=True,
        edit_domain_threshold=args.edit_domain_threshold,
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
            caption_root=val_caption_root,
            include_sky_training_data=False,
            include_rgb_training_data=False,
            require_edit_window=True,
            edit_domain_threshold=args.edit_domain_threshold,
            deterministic_windows=True,
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
            f"val_fraction={0.0 if independent_val else float(args.val_fraction):.3f}",
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

    if args.resume_path:
        scene_flow = build_scene_flow_from_checkpoint_config(
            args.resume_path,
            patch_grid=patch_grid,
            latent_dim=int(args.latent_dim),
            device=device,
        )
    else:
        scene_flow = WanSceneFlow.from_scene_config(
            patch_grid=patch_grid,
            out_channels=int(args.latent_dim),
            prediction_type=str(args.prediction_type),
            layout_condition_version=FORMAL_LAYOUT_CONDITION_VERSION,
            **FORMAL_LAYOUT_DISABLED_CONFIG,
        ).to(device)
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
    load_formal_latent_stats(
        scene_flow,
        args.feature_stats_path,
        token_dim=int(args.latent_dim),
        require_existing_match=bool(args.resume_path),
    )
    if is_main_process():
        print(
            f"[stats] loaded tokenizer latent statistics from {args.feature_stats_path}",
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
                    _dump_vis(_first_item(item), assembler, log_dir, global_step, device)

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
) -> None:
    vis_dir = log_dir / "vis" / f"step_{step:06d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        bundle = build_flow_bundle(item, assembler, device)
        frames = min(10, int(bundle.z_clean.shape[1]))
        grids = {
            "target_latent_pca": _latent_pca_grid(bundle.z_clean.float(), bundle.patch_grid, frames),
            "M_preserve": _mask_grid(bundle.M_preserve, bundle.patch_grid, frames),
            "M_source": _mask_grid(bundle.M_source, bundle.patch_grid, frames),
            "M_dest": _mask_grid(bundle.M_dest, bundle.patch_grid, frames),
        }
        for name, image in grids.items():
            save_image_grid(image, vis_dir / f"{name}.jpg", nrow=frames)


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
            "formal_flow_domain_version": FORMAL_FLOW_DOMAIN_VERSION,
            "formal_flow_domain_config": flow_domain_config,
        },
        ckpt_dir / f"flow_step{step:06d}_ema_weights_only.pt",
    )


if __name__ == "__main__":
    main()
