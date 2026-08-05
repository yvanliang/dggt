#!/usr/bin/env python3
"""Offline inference for the SceneFlow full-scene pretrain model.

The data path intentionally matches ``train_scene_flow_pretrain.py`` validation:

* raw ``WaymoOpenDataset(mode=1, views=1)`` validation clips;
* trunk-major sample ordering (all scenes' trunk 0, then trunk 1, ...);
* pure-noise RAE/FlowMatch sampling through ``cfg_sample_pretrain_latents``;
* generated camera/sky/sky-mask states and the no-image DGGT head/render path.

Condition modes rotate by global trunk-major validation row:

    none -> cam -> asset_cam -> ...

``none`` means text-only; text remains the required base condition.  Missing
asset/camera modalities use the learned null-condition tokens used by pretrain
condition-task sampling. ``--cfg`` controls text only; camera and asset use
their explicit hierarchical scales. Factorized asset placement is exposed only
together with its matching camera condition.

Example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow_pretrain.py \
        --cfg 1.0,2.0,4.0 \
        --num_frames 10 \
        --start 0 --end 8 \
        --output_dir runs/scene_flow_pretrain_inference

Each sample gets one directory whose name includes its condition mode.  In
addition to training-validation image grids (except ``abs_error``), each CFG
result contains:

* ``generated_raw_gaussians__cfg*_frame*.ply``: per-frame DGGT/3DGS Gaussian
  PLY schema in the selected export units (metric by default);
* ``generated_raw_points__cfg*_frame*.ply``: per-frame xyz + uchar RGB,
  directly viewable in MeshLab.

The PLYs are built by DGGT's canonical ``build_clean_scene_state`` and written
by its shared PLY writers.  Consequently they use the same non-sky + valid-depth
selection, world-coordinate unprojection, Gaussian activations and serialization
as ``inference_scene_editor.py``.  As in DGGT, frames are exported separately
so overlapping input frames do not visually over-densify a single point cloud;
no unrequested fusion heuristic is applied.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ``datasets.dataset`` imports Open3D at module scope, but the raw pretrain
# dataset path used here never calls it.  Match the repository's lightweight
# dataloader inspection utility so ``--help`` and inference remain usable in
# the training environment where Open3D is intentionally not installed.
try:
    import open3d as _open3d  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open3d"] = types.ModuleType("open3d")

try:
    import gsplat as _gsplat  # noqa: F401
except ModuleNotFoundError:
    # Keep argument parsing/import-time unit tests available.  Actual RGB
    # rendering still fails with an explicit dependency error if invoked.
    gsplat_stub = types.ModuleType("gsplat")
    gsplat_rendering_stub = types.ModuleType("gsplat.rendering")

    def _missing_gsplat(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "gsplat is required for SceneFlow 3DGS rendering; run this script in the dggt training environment."
        )

    gsplat_stub.rasterization = _missing_gsplat
    gsplat_rendering_stub.rasterization = _missing_gsplat
    sys.modules["gsplat"] = gsplat_stub
    sys.modules["gsplat.rendering"] = gsplat_rendering_stub

from datasets.dataset import WaymoOpenDataset
from dggt.models.scene_flow import WanSceneFlow
from dggt.models.canonical_asset_encoder import CanonicalAssetEncoder
from dggt.losses.rgb_render_loss import decode_generated_dggt_geometry
from dggt.utils.feature_stats import (
    DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH,
    checkpoint_sha256,
    load_all_stats_into_buffers,
)
from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_STATS_VERSION,
    CAMERA_TARGET_SOURCE,
    CAMERA_TARGET_SPACE,
)
from dggt.utils.camera_condition import camera_condition_from_waymo_metric_target
from dggt.utils.camera_geometry_flow_consistency import (
    CAMERA_GEOMETRY_FLOW_DIAGNOSTIC_SCHEMA,
    camera_geometry_flow_consistency,
)
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.flow_schedule import resolve_inference_flow_schedule
from dggt.utils.gaussian_edit import CleanSceneState, build_clean_scene_state
from dggt.utils.gaussian_ply import write_gaussian_ply, write_point_ply
from dggt.utils.sliding_window import resolve_offline_window
from dggt.utils.scene_gauge import (
    PULLBACK_METRIC_BOUNDARY,
    PULLBACK_RENDER_BOUNDARY,
    PULLBACK_RUNTIME_CONTRACT_VERSION,
    SCENE_GAUGE_DIM,
    SCENE_GAUGE_REPRESENTATION,
    SCENE_GAUGE_STATS_VERSION,
    PullbackCalibration,
    PullbackResult,
    apply_pullback_calibration,
    assemble_dggt_pose_encoding,
    gauge_to_pose_enc_fov,
    load_pullback_calibration,
    metric_c2w_to_teacher_anchor_dggt,
)
from dggt.utils.factorized_asset_condition import (
    FACTORIZED_ASSET_CONDITION_VERSION,
    PLACEMENT_STATE_DIM,
    FactorizedAssetCondition,
    build_factorized_asset_condition,
    canonicalize_asset_reference,
    interpolate_box_keyframes,
    object_to_anchor_from_center_yaw,
    resize_crop_intrinsics_to_model_canvas,
)
from train_scene_flow_pretrain import (
    DEFAULT_SKY_GRID,
    PRETRAIN_FEATURE_STATS_CONTRACT_KEY,
    SKY_TOKEN_DIM,
    CyclicSequentialSampler,
    _align_sliding_asset_payload_slots,
    _fixed_render_hw,
    _image_grid,
    _latent_pca_grid,
    _mask_grid,
    _predict_camera_mats,
    _render_gs_map_rgb,
    _sliding_intrinsics_window,
    _sky_background_image_grid,
    _sky_mask_image_grid,
    _sky_mask_patch_to_image,
    _timestamps_for_generated_render,
    attach_training_equivalent_sliding_asset_conditions,
    apply_pretrain_condition_task,
    autocast_context,
    build_pretrain_bundle_from_batch,
    build_full_scene_bundle,
    cfg_sample_pretrain_latents,
    decode_metric_camera_from_features,
    decode_sky_patch_tokens,
    discover_scene_names,
    load_dggt_aggregator_and_tokenizer,
    render_validation_generated_rgb,
    seed_everything,
    setup_text_encoder,
    sky_generation_enabled,
    sky_grid_shape,
    sky_tokens_to_background,
    validate_pretrain_feature_stats_contract,
    validate_scene_flow_checkpoint_config,
    unwrap_ddp,
)


DEFAULT_WEIGHTS = None
DEFAULT_VAL_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt/validation"
DEFAULT_CAPTION_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions"
DEFAULT_DGGT_CKPT = "/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt"
DEFAULT_TOKENIZER_CKPT = "logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt"
DEFAULT_VAL_SCENE_GAUGE = "data/scene_gauge/validation.json"
DEFAULT_PULLBACK_CALIBRATION = "data/scene_gauge/pullback_d63b34f7.json"
DEFAULT_TEXT_ENCODER = "/home/dancer/model/Qwen/Qwen3-0.6B"
CONDITION_MODES = ("none", "cam", "asset_cam")
GENERATED_METRIC_CAMERA_SCHEMA = "generated_metric_camera_trajectory_v1"
CAMERA_GAUGE_ATTRIBUTION_ARMS = (
    "teacher_camera__teacher_gauge",
    "generated_camera__teacher_gauge",
    "teacher_camera__generated_gauge",
    "generated_camera__generated_gauge",
)

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


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256, got {value!r}.")
    result = value
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256, got {value!r}.")
    return result


def validate_metric_gauge_provenance(
    provenance: Any,
    *,
    scene_flow: nn.Module,
    dggt_sha256: str,
    tokenizer_sha256: str,
    expected_pullback_runtime_contract_version: str | None = None,
    expected_window_len: int,
    expected_patch_grid: Sequence[int],
) -> dict[str, Any]:
    """Reject anything except the clean-cut metric/gauge checkpoint contract."""
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "SceneFlow checkpoint is missing the required `metric_gauge_provenance` block; "
            "legacy camera/checkpoint formats are intentionally unsupported."
        )
    actual_fields = set(provenance)
    if actual_fields != METRIC_GAUGE_PROVENANCE_FIELDS:
        raise ValueError(
            "metric_gauge_provenance does not match the strict schema: "
            f"missing={sorted(METRIC_GAUGE_PROVENANCE_FIELDS - actual_fields)}, "
            f"unknown={sorted(actual_fields - METRIC_GAUGE_PROVENANCE_FIELDS)}."
        )
    if isinstance(expected_window_len, bool) or int(expected_window_len) <= 0:
        raise ValueError("expected_window_len must be a positive integer")
    expected_grid = tuple(int(value) for value in expected_patch_grid)
    if len(expected_grid) != 2 or any(value <= 0 for value in expected_grid):
        raise ValueError("expected_patch_grid must contain two positive integers")
    if isinstance(provenance["pullback_window_len"], bool) or not isinstance(
        provenance["pullback_window_len"], int
    ):
        raise ValueError("metric_gauge_provenance.pullback_window_len must be an integer")
    artifact_grid = provenance["pullback_patch_grid_hw"]
    if not isinstance(artifact_grid, list) or len(artifact_grid) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in artifact_grid
    ):
        raise ValueError(
            "metric_gauge_provenance.pullback_patch_grid_hw must be a two-item positive integer list"
        )
    config = getattr(unwrap_ddp(scene_flow), "config", None)
    runtime_contract_version = provenance["pullback_runtime_contract_version"]
    if runtime_contract_version != PULLBACK_RUNTIME_CONTRACT_VERSION:
        raise ValueError(
            "metric_gauge_provenance.pullback_runtime_contract_version is unsupported: "
            f"{runtime_contract_version!r}."
        )
    if (
        expected_pullback_runtime_contract_version is not None
        and expected_pullback_runtime_contract_version
        != PULLBACK_RUNTIME_CONTRACT_VERSION
    ):
        raise ValueError(
            "expected_pullback_runtime_contract_version is unsupported: "
            f"{expected_pullback_runtime_contract_version!r}."
        )

    expected_values = {
        "scene_gauge_representation": SCENE_GAUGE_REPRESENTATION,
        "scene_gauge_stats_version": SCENE_GAUGE_STATS_VERSION,
        "tokenizer_sha256": _require_sha256(tokenizer_sha256, name="tokenizer checkpoint SHA-256"),
        "dggt_checkpoint_sha256": _require_sha256(dggt_sha256, name="DGGT checkpoint SHA-256"),
        "pullback_window_len": int(expected_window_len),
        "pullback_patch_grid_hw": list(expected_grid),
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_target_space": CAMERA_TARGET_SPACE,
        "camera_target_source": CAMERA_TARGET_SOURCE,
    }
    if expected_pullback_runtime_contract_version is not None:
        expected_values["pullback_runtime_contract_version"] = (
            expected_pullback_runtime_contract_version
        )
    for field, expected in expected_values.items():
        if provenance[field] != expected:
            raise ValueError(
                f"metric_gauge_provenance.{field} mismatch: "
                f"checkpoint={provenance[field]!r}, current={expected!r}."
            )
    for field in (
        "gauge_table_sha256",
        "pullback_artifact_sha256",
        "tokenizer_sha256",
        "dggt_checkpoint_sha256",
    ):
        _require_sha256(provenance[field], name=f"metric_gauge_provenance.{field}")
    config_expectations = {
        "gauge_gen_dim": SCENE_GAUGE_DIM,
        "scene_gauge_representation": SCENE_GAUGE_REPRESENTATION,
        "scene_gauge_stats_version": SCENE_GAUGE_STATS_VERSION,
        "camera_gen_dim": CAMERA_GENERATION_DIM,
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "asset_condition_protocol": "factorized_v1",
        "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
    }
    for field, expected in config_expectations.items():
        actual = getattr(config, field, None)
        if actual != expected:
            raise ValueError(
                f"SceneFlow config {field}={actual!r} does not satisfy the metric/gauge "
                f"inference contract {expected!r}."
            )
    for field in ("placement_mean", "placement_std"):
        values = getattr(config, field, None)
        if not isinstance(values, (list, tuple)) or len(values) != PLACEMENT_STATE_DIM:
            raise ValueError(
                f"SceneFlow config {field} must contain exactly {PLACEMENT_STATE_DIM} "
                "factorized-v3 values."
            )
        try:
            numeric_values = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SceneFlow config {field} must contain numeric values.") from exc
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"SceneFlow config {field} must contain finite values.")
        if field == "placement_std" and not all(value > 0.0 for value in numeric_values):
            raise ValueError("SceneFlow config placement_std must contain positive values.")
    return dict(provenance)


def parse_cfg_scales(value: str | Sequence[str]) -> list[float]:
    """Parse comma/space separated CFG values while preserving user order."""
    raw_values = [value] if isinstance(value, str) else list(value)
    scales: list[float] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            scale = float(part)
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(f"CFG scales must be finite and non-negative, got {part!r}.")
            if scale not in scales:
                scales.append(scale)
    if not scales:
        raise ValueError("--cfg must contain at least one scale.")
    return scales


def cfg_tag(scale: float) -> str:
    return f"{float(scale):g}".replace("-", "m").replace(".", "p")


def condition_mode_for_position(position: int) -> str:
    return CONDITION_MODES[int(position) % len(CONDITION_MODES)]


def _batch_size_from_bundle(bundle: Any) -> int:
    return int(bundle.z_clean_n.shape[0])


def apply_condition_mode(bundle: Any, mode: str) -> Any:
    """Hide optional conditions with the same learned-null semantics as training."""
    if mode not in CONDITION_MODES:
        raise ValueError(f"Unknown condition mode {mode!r}; expected one of {CONDITION_MODES}.")
    task = {
        "none": "joint_generation",
        "cam": "camera_controlled",
        "asset_cam": "asset_camera_controlled",
    }[mode]
    return apply_pretrain_condition_task(bundle, task)


def _row_has_any(mask: torch.Tensor | None, batch_size: int) -> list[bool]:
    if not torch.is_tensor(mask):
        return [False] * int(batch_size)
    return mask.detach().to(torch.bool).reshape(int(batch_size), -1).any(dim=1).cpu().tolist()


def actual_condition_rows(bundle: Any) -> dict[str, list[bool]]:
    batch_size = _batch_size_from_bundle(bundle)
    factorized = getattr(bundle, "factorized_asset_condition", None)
    if isinstance(factorized, FactorizedAssetCondition):
        asset_rows = (
            factorized.appearance_mask.any(dim=(1, 2))
            & factorized.track_valid.any(dim=(1, 2))
        ).detach().cpu().tolist()
    else:
        asset_rows = _row_has_any(getattr(bundle, "encoder_attention_mask", None), batch_size)
    by_window = getattr(bundle, "factorized_asset_conditions_by_window", None)
    if by_window:
        window_rows = [False] * batch_size
        for condition in by_window.values():
            rows = (
                condition.appearance_mask.any(dim=(1, 2))
                & condition.track_valid.any(dim=(1, 2))
            ).detach().cpu().tolist()
            window_rows = [
                bool(window_rows[row]) or bool(rows[row])
                for row in range(batch_size)
            ]
        asset_rows = [
            bool(asset_rows[row]) or bool(window_rows[row])
            for row in range(batch_size)
        ]
    asset_kinds = getattr(bundle, "asset_condition_kind", None)
    if isinstance(asset_kinds, str):
        asset_kinds = [asset_kinds] * batch_size
    if asset_kinds is not None:
        asset_rows = [
            bool(row) and str(asset_kinds[idx]).lower() not in {"none", "asset_uncond", "asset_null"}
            for idx, row in enumerate(asset_rows)
        ]
    camera_tokens = getattr(bundle, "camera_condition_tokens", None)
    camera_rows = (
        _row_has_any(getattr(bundle, "camera_attention_mask", None), batch_size)
        if torch.is_tensor(camera_tokens)
        else [False] * batch_size
    )
    camera_kinds = getattr(bundle, "camera_condition_kind", None)
    if isinstance(camera_kinds, str):
        camera_kinds = [camera_kinds] * batch_size
    if camera_kinds is not None:
        camera_rows = [
            bool(row) and str(camera_kinds[idx]).lower() not in {"camera_uncond", "camera_null"}
            for idx, row in enumerate(camera_rows)
        ]
    return {"asset": asset_rows, "camera": camera_rows}


def _load_external_rgba(
    image_path: Path,
    mask_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).contiguous()
    embedded_alpha = torch.from_numpy(rgba[..., 3]).unsqueeze(0).contiguous()
    if mask_path is not None:
        mask_array = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        alpha = torch.from_numpy(mask_array).unsqueeze(0).contiguous()
    elif image.mode in ("RGBA", "LA") or "transparency" in image.info:
        alpha = embedded_alpha
    else:
        raise ValueError(
            f"RGB asset {image_path} requires an explicit `mask`; "
            "automatic foreground extraction is outside the model protocol."
        )
    if tuple(alpha.shape[-2:]) != tuple(rgb.shape[-2:]):
        raise ValueError(f"asset mask {mask_path} does not match image {image_path}")
    return rgb, alpha


def _object_to_anchor_from_center_yaw(
    centers: torch.Tensor,
    yaws: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper around the shared training-coordinate builder."""
    if centers.ndim != 3:
        raise ValueError("centers/yaws must be [K,S,3] and [K,S]")
    return object_to_anchor_from_center_yaw(centers, yaws)


def build_external_factorized_pretrain_bundle(
    manifest_path: str | Path,
    *,
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    device: torch.device,
    patch_grid: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    """Build an open-inference bundle without a target video or target mask."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    if payload.get("coordinate_frame") != "camera_anchor":
        raise ValueError("external asset manifest coordinate_frame must be 'camera_anchor'")
    fps = float(payload.get("fps", 10.0))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("manifest fps must be finite and positive")
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("manifest requires a camera object")
    camera_to_anchor = torch.as_tensor(camera.get("camera_to_anchor"), dtype=torch.float32)
    if camera_to_anchor.ndim != 3 or camera_to_anchor.shape[-2:] != (4, 4):
        raise ValueError("camera.camera_to_anchor must be [S,4,4]")
    seq_len = int(camera_to_anchor.shape[0])
    intrinsics = torch.as_tensor(camera.get("intrinsics"), dtype=torch.float32)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("camera.intrinsics must be [1/S,3,3]")
    if int(intrinsics.shape[0]) == 1:
        intrinsics = intrinsics.expand(seq_len, -1, -1).contiguous()
    if int(intrinsics.shape[0]) != seq_len:
        raise ValueError("camera intrinsics length must be one or match camera_to_anchor")
    image_size_hw = torch.as_tensor(camera.get("image_size_hw"), dtype=torch.long)
    if tuple(image_size_hw.shape) != (2,) or bool((image_size_hw <= 0).any()):
        raise ValueError("camera.image_size_hw must contain positive [height,width]")

    sf = unwrap_ddp(scene_flow)
    if str(getattr(sf.config, "asset_condition_protocol", "")) != "factorized_v1":
        raise RuntimeError(
            "External factorized asset inference requires a checkpoint trained with "
            "asset_condition_protocol='factorized_v1'; legacy checkpoints do not "
            "contain trained placement-adapter weights."
        )
    max_assets = int(sf.config.max_assets)
    canvas_hw = (int(patch_grid[0]) * 14, int(patch_grid[1]) * 14)
    reference_rgb = torch.zeros((max_assets, 3, *canvas_hw), dtype=torch.float32)
    reference_alpha = torch.zeros((max_assets, 1, *canvas_hw), dtype=torch.float32)
    centers = torch.zeros((max_assets, seq_len, 3), dtype=torch.float32)
    sizes = torch.ones((max_assets, seq_len, 3), dtype=torch.float32)
    yaws = torch.zeros((max_assets, seq_len), dtype=torch.float32)
    velocities = torch.zeros((max_assets, seq_len, 3), dtype=torch.float32)
    track_valid = torch.zeros((max_assets, seq_len), dtype=torch.bool)
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("manifest objects must be a list")
    if len(objects) > max_assets:
        raise ValueError(f"manifest has {len(objects)} objects, model supports {max_assets}")
    object_ids = []
    for slot, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError("each manifest object must be a JSON object")
        image_value = item.get("image")
        if not image_value:
            raise ValueError("each manifest object requires `image`")
        image_path = (path.parent / str(image_value)).resolve()
        mask_value = item.get("mask")
        mask_path = None if not mask_value else (path.parent / str(mask_value)).resolve()
        rgb, alpha = _load_external_rgba(image_path, mask_path)
        reference_rgb[slot], reference_alpha[slot] = canonicalize_asset_reference(
            rgb, alpha, canvas_hw
        )
        center, size, yaw, velocity, valid = interpolate_box_keyframes(
            item.get("keyframes", []),
            seq_len,
            fps,
        )
        centers[slot] = center
        sizes[slot] = size
        yaws[slot] = yaw
        velocities[slot] = velocity
        track_valid[slot] = valid
        object_ids.append(str(item.get("id", f"asset_{slot}")))

    encoder = CanonicalAssetEncoder(
        vggt_model.aggregator,
        vggt_model.scene_tokenizer,
        sf,
        patch_grid=patch_grid,
        max_tokens=32,
    ).to(device)
    appearance = encoder(
        reference_rgb.to(device).reshape(max_assets, 1, 3, *canvas_hw),
        reference_alpha.to(device).reshape(max_assets, 1, 1, *canvas_hw),
        batch_size=1,
        num_assets=max_assets,
    )
    object_to_anchor = _object_to_anchor_from_center_yaw(centers, yaws).unsqueeze(0).to(device)
    reference_frame_id = torch.full((1, max_assets), -1, device=device, dtype=torch.long)
    projection_intrinsics, projection_image_hw = (
        resize_crop_intrinsics_to_model_canvas(
            intrinsics.to(device),
            image_size_hw.to(device),
            target_width=int(patch_grid[1]) * 14,
            patch_size=14,
        )
    )
    condition = build_factorized_asset_condition(
        appearance_tokens=appearance.appearance_tokens,
        appearance_mask=appearance.appearance_mask,
        canonical_uv=appearance.canonical_uv,
        object_to_anchor=object_to_anchor,
        center_anchor=centers.unsqueeze(0).to(device),
        box_size_lwh=sizes.unsqueeze(0).to(device),
        yaw=yaws.unsqueeze(0).to(device),
        velocity_anchor=velocities.unsqueeze(0).to(device),
        track_valid=track_valid.unsqueeze(0).to(device),
        camera_to_anchor=camera_to_anchor.unsqueeze(0).to(device),
        intrinsics=projection_intrinsics.unsqueeze(0),
        image_size_hw=projection_image_hw,
        patch_grid=patch_grid,
        reference_frame_id=reference_frame_id,
    )
    # Anchor coordinates are a valid metric world.  Use the same role-aware
    # normalized 9-D condition channels as raw pretraining and formal T1.
    frame_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1)
    camera_tokens, camera_mask, _, returned_anchor_mask = (
        camera_condition_from_waymo_metric_target(
            camera_to_anchor.unsqueeze(0).to(device),
            intrinsics.unsqueeze(0).to(device),
            image_hw=tuple(int(value) for value in image_size_hw.tolist()),
            trajectory_anchor_to_world=torch.eye(4, device=device).view(1, 1, 4, 4),
            previous_camera_to_world=None,
            anchor_mask=frame_ids.eq(0),
            normalize_camera=sf.normalize_camera,
        )
    )
    if not torch.equal(returned_anchor_mask, frame_ids.eq(0)):
        raise RuntimeError("external metric camera condition returned inconsistent anchor roles")
    latent_dim = int(sf.config.out_channels)
    dummy_endpoint = torch.zeros(
        (1, seq_len, int(patch_grid[0]) * int(patch_grid[1]), latent_dim),
        device=device,
    )
    bundle = build_full_scene_bundle(
        dummy_endpoint,
        kv_dim=latent_dim,
        camera_condition_tokens=camera_tokens,
        camera_attention_mask=camera_mask,
        camera_condition_kind=["camera"],
        frame_ids=frame_ids,
    )
    bundle.F_asset_tokens = dummy_endpoint.new_zeros((1, 0, latent_dim))
    bundle.encoder_attention_mask = None
    bundle.factorized_asset_condition = condition
    bundle.F_asset_lengths = condition.appearance_mask.any(dim=-1).sum(dim=-1).long()
    bundle.asset_condition_kind = [
        "factorized_asset" if bool(condition.appearance_mask.any()) else "none"
    ]
    bundle.asset_condition_source_kind = ["external_manifest"]
    bundle.captions = [str(payload.get("text", ""))]
    bundle.fps = torch.tensor([fps], device=device)
    bundle.camera_gen_anchor_mask = frame_ids.eq(0)
    return bundle, {
        "manifest": str(path),
        "condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
        "object_ids": object_ids,
        "num_frames": seq_len,
        "fps": fps,
        "image_size_hw": image_size_hw.tolist(),
    }


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def _checkpoint_config(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    config = payload.get("scene_flow_config")
    return dict(config) if isinstance(config, dict) else None


def build_scene_flow_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    no_ema: bool,
    fallback_args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the exact saved architecture and load EMA weights strictly."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SceneFlow checkpoint not found: {path}. The requested default is intentionally allowed "
            "to be absent until training writes it; pass --weights when it becomes available."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "camera_dggt_provenance" in payload:
        raise ValueError(
            f"{path} contains legacy camera_dggt_provenance; mixed old/new provenance is rejected."
        )
    checkpoint_provenance = (
        payload.get("metric_gauge_provenance") if isinstance(payload, dict) else None
    )
    if not isinstance(payload, dict) or not isinstance(checkpoint_provenance, Mapping):
        raise ValueError(
            f"{path} is not a metric/gauge SceneFlow checkpoint: the strict "
            "`metric_gauge_provenance` block is missing. Legacy checkpoints are rejected."
        )
    provenance_fields = set(checkpoint_provenance)
    if provenance_fields != METRIC_GAUGE_PROVENANCE_FIELDS:
        raise ValueError(
            f"{path} metric_gauge_provenance violates the strict schema before model "
            f"construction: missing={sorted(METRIC_GAUGE_PROVENANCE_FIELDS - provenance_fields)}, "
            f"unknown={sorted(provenance_fields - METRIC_GAUGE_PROVENANCE_FIELDS)}."
        )
    checkpoint_stats_contract = validate_pretrain_feature_stats_contract(
        payload.get(PRETRAIN_FEATURE_STATS_CONTRACT_KEY),
        path=path,
    )
    config = _checkpoint_config(payload)
    if config is None:
        raise ValueError(
            f"{path} has no versioned scene_flow_config; metric/gauge v4 inference "
            "does not construct a fallback architecture."
        )
    required_config = {
        "camera_gen_dim": CAMERA_GENERATION_DIM,
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_stats_version": CAMERA_STATS_VERSION,
        "gauge_gen_dim": SCENE_GAUGE_DIM,
        "scene_gauge_representation": SCENE_GAUGE_REPRESENTATION,
        "scene_gauge_stats_version": SCENE_GAUGE_STATS_VERSION,
        "asset_condition_protocol": "factorized_v1",
        "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
    }
    required_shape_config = {"patch_grid", "placement_mean", "placement_std"}
    missing_config = sorted((set(required_config) | required_shape_config) - set(config))
    mismatched_config = {
        field: {"checkpoint": config.get(field), "required": expected}
        for field, expected in required_config.items()
        if field in config and config[field] != expected
    }
    if missing_config or mismatched_config:
        raise ValueError(
            f"{path} does not satisfy the clean-cut Waymo metric camera/gauge v4 config: "
            f"missing={missing_config}, mismatched={mismatched_config}."
        )
    patch_grid = config["patch_grid"]
    if (
        not isinstance(patch_grid, (list, tuple))
        or len(patch_grid) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in patch_grid)
    ):
        raise ValueError(f"{path} scene_flow_config.patch_grid must contain two positive integers.")
    for field in ("placement_mean", "placement_std"):
        values = config[field]
        if not isinstance(values, (list, tuple)) or len(values) != PLACEMENT_STATE_DIM:
            raise ValueError(
                f"{path} scene_flow_config.{field} must contain exactly "
                f"{PLACEMENT_STATE_DIM} factorized-v3 values."
            )
        try:
            numeric_values = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} scene_flow_config.{field} must be numeric.") from exc
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"{path} scene_flow_config.{field} must be finite.")
        if field == "placement_std" and not all(value > 0.0 for value in numeric_values):
            raise ValueError(f"{path} scene_flow_config.placement_std must be positive.")
    validate_pretrain_feature_stats_contract(
        checkpoint_stats_contract,
        path=path,
        expected_patch_grid=patch_grid,
    )
    flow_schedule = resolve_inference_flow_schedule(payload, fallback_args, path)
    scene_flow = WanSceneFlow(**config)
    validate_scene_flow_checkpoint_config(scene_flow, payload, path)

    source = ""
    if no_ema:
        if not isinstance(payload, dict) or "scene_flow" not in payload:
            raise ValueError(f"{path} has no raw scene_flow state for --no_ema.")
        if bool(payload.get("is_ema_weights")):
            raise ValueError(f"{path} is EMA-only; remove --no_ema or use a full/raw checkpoint.")
        state = payload["scene_flow"]
        source = "raw scene_flow"
    elif isinstance(payload, dict) and "ema_scene_flow_state_dict" in payload:
        state = payload["ema_scene_flow_state_dict"]
        source = "ema_scene_flow_state_dict"
    elif isinstance(payload, dict) and bool(payload.get("is_ema_weights")) and "scene_flow" in payload:
        state = payload["scene_flow"]
        source = "EMA-only scene_flow"
    elif isinstance(payload, dict) and "scene_flow" in payload:
        state = payload["scene_flow"]
        source = "raw scene_flow (EMA unavailable)"
        print(f"[warn] {path} does not contain EMA weights; falling back to raw weights.", flush=True)
    else:
        raise ValueError(
            f"Unsupported metric/gauge SceneFlow checkpoint weight container: {path}"
        )

    if state is not None:
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint state {source} is not a state_dict.")
        scene_flow.load_state_dict(_strip_module_prefix(state), strict=True)
    scene_flow.to(device).eval()
    for parameter in scene_flow.parameters():
        parameter.requires_grad_(False)
    info = {
        "path": str(path),
        "weight_source": source,
        "step": int(payload.get("step", 0)) if isinstance(payload, dict) else 0,
        "scene_flow_config": config,
        "metric_gauge_provenance": payload.get("metric_gauge_provenance")
        if isinstance(payload, dict)
        else None,
        PRETRAIN_FEATURE_STATS_CONTRACT_KEY: checkpoint_stats_contract,
        "flow_schedule_config": flow_schedule,
    }
    del payload
    return scene_flow, info


def sync_args_from_model(args: argparse.Namespace, scene_flow: nn.Module) -> None:
    config = scene_flow.config
    args.patch_grid = tuple(int(v) for v in config.patch_grid)
    args.latent_dim = int(config.out_channels)
    args.prediction_type = str(config.prediction_type)
    args.sky_grid = tuple(int(v) for v in config.sky_grid)
    args.sky_grid_h, args.sky_grid_w = args.sky_grid
    args.sky_atlas_hw = tuple(int(v) for v in getattr(config, "sky_atlas_hw", (32, 64)))
    args.sky_mask_refine_scale = int(config.sky_mask_refine_scale)
    args.sky_mask_refine_channels = int(config.sky_mask_refine_channels)
    args.asset_position_mode = str(getattr(config, "asset_position_mode", "localized"))
    args.camera_gen_dim = int(config.camera_gen_dim)
    args.camera_generation_representation = str(config.camera_generation_representation)
    args.gauge_gen_dim = int(config.gauge_gen_dim)
    args.scene_gauge_representation = str(config.scene_gauge_representation)
    args.scene_gauge_stats_version = str(config.scene_gauge_stats_version)


def build_generated_dggt_scene_state(
    *,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
    sky_mask: torch.Tensor,
) -> CleanSceneState:
    """Build the generated scene through DGGT's canonical scene-state path.

    Only ``images_clean`` metadata is synthetic because generated inference has
    no source RGB image.  DGGT uses that tensor solely for ``[S,H,W]`` shape and
    stores it in the returned state; geometry and PLY fields come exclusively
    from pose/depth/GS/dynamic predictions plus the generated sky mask.
    """
    if int(pose_enc.shape[0]) != 1:
        raise ValueError(f"PLY export currently expects batch size 1, got pose_enc={tuple(pose_enc.shape)}")
    if (
        sky_mask.ndim != 5
        or int(sky_mask.shape[0]) != 1
        or int(sky_mask.shape[2]) not in (1, 3)
    ):
        raise ValueError(f"sky_mask must be [1,S,1/3,H,W], got {tuple(sky_mask.shape)}")
    seq_len, height, width = int(sky_mask.shape[1]), int(sky_mask.shape[-2]), int(sky_mask.shape[-1])
    images_clean = torch.zeros((seq_len, 3, height, width), dtype=torch.float32)
    sky_probability = sky_mask[0].detach().cpu().float().mean(dim=1).clamp(0.0, 1.0)
    non_sky_probability = 1.0 - sky_probability
    # Keep depth-valid points in the canonical state, then apply one explicit
    # hard sky threshold to both PLY formats at export time.
    sky_mask_cpu = torch.zeros_like(non_sky_probability).unsqueeze(1)
    sample = {
        "images_clean": images_clean,
        # build_clean_scene_state's fallback expression accesses `masks`
        # eagerly, so provide both aliases exactly as normal DGGT samples do.
        "sky_mask": sky_mask_cpu,
        "masks": sky_mask_cpu,
        "cam_ids": torch.tensor([0], dtype=torch.long),
    }
    predictions = {
        "pose_enc": pose_enc,
        "depth": depth,
        "gs_map": gs_map,
        "gs_conf": gs_conf,
        "dynamic_conf": dynamic_conf,
    }
    state = build_clean_scene_state(sample, predictions)
    point_non_sky = non_sky_probability[
        state.source_image_ids,
        state.source_y,
        state.source_x,
    ].reshape(-1, 1)
    state.sky_probabilities = 1.0 - point_non_sky
    state.opacities = state.opacities * point_non_sky
    return state


def prepare_generated_geometry_boundaries(
    *,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gauge: torch.Tensor,
    calibration: PullbackCalibration,
    export_units: str,
) -> tuple[PullbackResult, PullbackResult]:
    """Keep native rendering exact and calibrate only a requested metric export."""
    if export_units not in {"dggt", "metric"}:
        raise ValueError(f"export_units must be 'dggt' or 'metric', got {export_units!r}")
    if (
        not torch.is_tensor(gauge)
        or gauge.ndim != 3
        or tuple(gauge.shape[-2:]) != (1, SCENE_GAUGE_DIM)
    ):
        raise ValueError(f"generated gauge must be [B,1,{SCENE_GAUGE_DIM}], got {getattr(gauge, 'shape', None)}")
    if float(calibration.c_gs) != 1.0:
        raise ValueError(f"v2 metric inference requires c_gs=1.0, got {calibration.c_gs!r}")
    log_metric_scale = gauge[..., 0]
    render_geometry = apply_pullback_calibration(
        depth,
        gs_map,
        log_metric_scale=log_metric_scale,
        calibration=calibration,
        boundary=PULLBACK_RENDER_BOUNDARY,
    )
    # Rendering must remain bitwise in the frozen tokenizer's native geometry.
    if render_geometry.depth_dggt is not depth or render_geometry.gs_map_dggt is not gs_map:
        raise AssertionError("render pullback boundary must return the original depth and GS tensors")
    if not bool(torch.equal(render_geometry.c_depth_factor, torch.ones_like(render_geometry.c_depth_factor))):
        raise AssertionError("render pullback boundary must have unit depth factors")
    if export_units == "dggt":
        return render_geometry, render_geometry
    metric_geometry = apply_pullback_calibration(
        depth,
        gs_map,
        log_metric_scale=log_metric_scale,
        calibration=calibration,
        boundary=PULLBACK_METRIC_BOUNDARY,
    )
    return render_geometry, metric_geometry


def _single_sample_gauge_summary(gauge: torch.Tensor) -> dict[str, Any]:
    value = torch.as_tensor(gauge).detach().float()
    if value.ndim == 1:
        value = value.view(1, 1, -1)
    elif value.ndim == 2:
        value = value.unsqueeze(1)
    if tuple(value.shape) != (1, 1, SCENE_GAUGE_DIM) or not bool(torch.isfinite(value).all()):
        raise ValueError(
            f"PLY export requires one finite scene gauge [1,1,{SCENE_GAUGE_DIM}], got {tuple(value.shape)}"
        )
    fov_yx = gauge_to_pose_enc_fov(value, 1)[0, 0]
    log_metric_scale = float(value[0, 0, 0].item())
    metres_per_unit = math.exp(log_metric_scale)
    if not math.isfinite(metres_per_unit) or metres_per_unit <= 0.0:
        raise ValueError(f"generated metres_per_unit is invalid: {metres_per_unit!r}")
    return {
        "representation": SCENE_GAUGE_REPRESENTATION,
        "values": [float(channel) for channel in value[0, 0].cpu().tolist()],
        "log_metric_scale": log_metric_scale,
        "metres_per_unit": metres_per_unit,
        "fov_deg": {
            "x": math.degrees(float(fov_yx[1].item())),
            "y": math.degrees(float(fov_yx[0].item())),
        },
    }


def _single_camera_trajectory(
    value: torch.Tensor | Any,
    *,
    name: str,
    allow_front_view_axis: bool = False,
) -> torch.Tensor:
    """Canonicalize one metric camera trajectory to CPU ``[S,4,4]``.

    Raw Waymo batches may carry an explicit view axis, whereas decoded
    generated trajectories do not.  Offline inference is deliberately
    batch-size one, so accepting a larger batch here would make the JSON
    artifact ambiguous and is rejected.
    """

    tensor = torch.as_tensor(value).detach().float()
    if allow_front_view_axis and tensor.ndim == 5:
        if int(tensor.shape[2]) < 1:
            raise ValueError(f"{name} has an empty view axis")
        tensor = tensor[:, :, 0]
    if tensor.ndim == 3 and tuple(tensor.shape[-2:]) == (4, 4):
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (4, 4):
        raise ValueError(
            f"{name} must be [1,S,4,4]"
            + (" or [1,S,V,4,4]" if allow_front_view_axis else "")
            + f", got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) != 1 or int(tensor.shape[1]) < 1:
        raise ValueError(f"{name} must contain exactly one non-empty trajectory, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor[0].cpu()


def condition_vs_generated_camera_translation_metrics(
    generated_camera_to_world_metric: torch.Tensor,
    waymo_camera_to_world_metric: torch.Tensor,
) -> dict[str, float | int]:
    """Compare generated and requested Waymo camera centers in metres.

    ``translation_l2_m_rmse`` is the plan's direct controllability measure:
    the root mean squared Euclidean camera-center error over frames.  The
    relative-to-first variant removes a common global translation and exposes
    trajectory-shape drift separately.
    """

    generated = _single_camera_trajectory(
        generated_camera_to_world_metric,
        name="generated_camera_to_world_metric",
    )
    waymo = _single_camera_trajectory(
        waymo_camera_to_world_metric,
        name="waymo_camera_to_world_metric",
        allow_front_view_axis=True,
    )
    if generated.shape != waymo.shape:
        raise ValueError(
            "generated and Waymo camera trajectories must have identical shape, "
            f"got {tuple(generated.shape)} and {tuple(waymo.shape)}"
        )
    generated_t = generated[:, :3, 3]
    waymo_t = waymo[:, :3, 3]
    absolute_error = torch.linalg.vector_norm(generated_t - waymo_t, dim=-1)
    generated_relative = generated_t - generated_t[:1]
    waymo_relative = waymo_t - waymo_t[:1]
    relative_error = torch.linalg.vector_norm(
        generated_relative - waymo_relative, dim=-1
    )

    def summarize(error: torch.Tensor, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(error.mean().item()),
            f"{prefix}_rmse": float(error.square().mean().sqrt().item()),
            f"{prefix}_max": float(error.max().item()),
        }

    return {
        "frame_count": int(generated.shape[0]),
        **summarize(absolute_error, "translation_l2_m"),
        **summarize(relative_error, "relative_to_first_translation_l2_m"),
    }


def build_generated_metric_camera_summary(
    *,
    camera_state_metric_9d: torch.Tensor,
    camera_to_world_metric: torch.Tensor,
    camera_anchor_mask: torch.Tensor,
    gauge: torch.Tensor,
    waymo_camera_to_world_metric: torch.Tensor | None,
    camera_condition_active: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build JSON and tensor artifacts for 10/29-frame camera comparison."""

    state = torch.as_tensor(camera_state_metric_9d).detach().float()
    if state.ndim == 2:
        state = state.unsqueeze(0)
    if state.ndim != 3 or int(state.shape[0]) != 1 or int(state.shape[-1]) != CAMERA_GENERATION_DIM:
        raise ValueError(
            "camera_state_metric_9d must be [1,S,9], "
            f"got {tuple(state.shape)}"
        )
    if int(state.shape[1]) < 1 or not bool(torch.isfinite(state).all()):
        raise ValueError("camera_state_metric_9d must be non-empty and finite")
    generated_c2w = _single_camera_trajectory(
        camera_to_world_metric,
        name="camera_to_world_metric",
    )
    if int(generated_c2w.shape[0]) != int(state.shape[1]):
        raise ValueError(
            "camera state and decoded trajectory lengths differ: "
            f"{int(state.shape[1])} vs {int(generated_c2w.shape[0])}"
        )
    anchors = torch.as_tensor(camera_anchor_mask).detach().to(dtype=torch.bool)
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    if tuple(anchors.shape) != tuple(state.shape[:2]):
        raise ValueError(
            f"camera_anchor_mask must be {tuple(state.shape[:2])}, got {tuple(anchors.shape)}"
        )
    gauge_summary = _single_sample_gauge_summary(gauge)
    gauge_tensor = torch.as_tensor(gauge).detach().float().reshape(1, 1, SCENE_GAUGE_DIM)
    generated_translations = generated_c2w[:, :3, 3]
    first_ten = min(10, int(state.shape[1]))

    comparison: dict[str, Any]
    waymo_c2w: torch.Tensor | None = None
    if waymo_camera_to_world_metric is None:
        comparison = {
            "status": "unavailable",
            "reason": "waymo_camera_to_world_metric_missing",
            "camera_condition_active": bool(camera_condition_active),
            "eligible_for_camera_controllability_gate": False,
        }
    else:
        waymo_c2w = _single_camera_trajectory(
            waymo_camera_to_world_metric,
            name="waymo_camera_to_world_metric",
            allow_front_view_axis=True,
        )
        metrics = condition_vs_generated_camera_translation_metrics(
            generated_c2w,
            waymo_c2w,
        )
        comparison = {
            "status": (
                "computed_camera_condition_active"
                if camera_condition_active
                else "reference_only_camera_condition_inactive"
            ),
            "camera_condition_active": bool(camera_condition_active),
            "eligible_for_camera_controllability_gate": bool(camera_condition_active),
            "target": "Waymo front-camera camera_to_world_corrected",
            **metrics,
            "waymo_translations_m": waymo_c2w[:, :3, 3].tolist(),
            "first_10_waymo_translations_m": waymo_c2w[:first_ten, :3, 3].tolist(),
        }

    summary = {
        "schema": GENERATED_METRIC_CAMERA_SCHEMA,
        "units": "metres",
        "camera_state_representation": CAMERA_GENERATION_REPRESENTATION,
        "frame_count": int(state.shape[1]),
        "camera_state_metric_9d": state[0].cpu().tolist(),
        "camera_to_world_metric": generated_c2w.tolist(),
        "translations_m": generated_translations.tolist(),
        "first_10_camera_state_metric_9d": state[0, :first_ten].cpu().tolist(),
        "first_10_translations_m": generated_translations[:first_ten].tolist(),
        "camera_anchor_mask": anchors[0].cpu().tolist(),
        "scene_global_gauge": gauge_summary,
        "scene_global_gauge_shape": [1, 1, SCENE_GAUGE_DIM],
        "condition_vs_generated": comparison,
    }
    tensor_artifact = {
        "schema": GENERATED_METRIC_CAMERA_SCHEMA,
        "units": "metres",
        "camera_state_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_state_metric_9d": state.cpu(),
        "camera_to_world_metric": generated_c2w.unsqueeze(0),
        "translations_m": generated_translations.unsqueeze(0),
        "camera_anchor_mask": anchors.cpu(),
        "scene_global_gauge": gauge_tensor.cpu(),
        "waymo_camera_to_world_metric": (
            None if waymo_c2w is None else waymo_c2w.unsqueeze(0)
        ),
        "condition_vs_generated": comparison,
    }
    return summary, tensor_artifact


def export_generated_pointclouds(
    *,
    scene_state: CleanSceneState,
    output_dir: Path,
    suffix: str,
    stride: int,
    sky_probability_threshold: float = 0.5,
    min_effective_opacity: float = 0.01,
    export_units: str = "dggt",
    gauge: torch.Tensor | None = None,
    c_depth_factor: torch.Tensor | None = None,
    calibration: PullbackCalibration | None = None,
) -> dict[str, Any]:
    """Write one canonical Gaussian/point PLY pair per generated frame.

    DGGT's inspection/export utilities write frame-local point clouds such as
    ``sample00_frame00.ply``.  Keeping that convention matters for MeshLab:
    opening an all-frame merged cloud makes the per-pixel samples from adjacent
    camera poses overlap densely, which visually reads as oversized points even
    though the PLY schema has no point-size field.
    """
    if export_units not in {"dggt", "metric"}:
        raise ValueError(f"export_units must be 'dggt' or 'metric', got {export_units!r}")
    gauge_summary = None if gauge is None else _single_sample_gauge_summary(gauge)
    if export_units == "metric" and gauge_summary is None:
        raise ValueError("metric PLY export requires the generated scene gauge")
    if export_units == "metric" and calibration is None:
        raise ValueError("metric PLY export requires a validated pullback calibration")
    metres_per_unit = 1.0 if export_units == "dggt" else float(gauge_summary["metres_per_unit"])
    stride = max(1, int(stride))
    seq_len = int(scene_state.images.shape[0])
    frame_summaries: list[dict[str, Any]] = []
    frame_counts: list[int] = []
    total_points = 0
    total_before = 0
    total_sky_removed = 0
    total_opacity_removed = 0
    sky_probabilities = getattr(scene_state, "sky_probabilities", None)
    if not torch.is_tensor(sky_probabilities):
        raise RuntimeError("generated scene state is missing per-point sky probabilities")
    for frame_idx in range(seq_len):
        frame_keep = scene_state.source_image_ids == int(frame_idx)
        if stride > 1:
            frame_keep &= torch.remainder(scene_state.source_y, stride) == 0
            frame_keep &= torch.remainder(scene_state.source_x, stride) == 0
        before = int(frame_keep.sum().item())
        sky_keep = sky_probabilities.reshape(-1) < float(sky_probability_threshold)
        opacity_keep = scene_state.opacities.reshape(-1) >= float(min_effective_opacity)
        sky_removed = int((frame_keep & ~sky_keep).sum().item())
        opacity_removed = int((frame_keep & sky_keep & ~opacity_keep).sum().item())
        keep = frame_keep & sky_keep & opacity_keep
        total_before += before
        total_sky_removed += sky_removed
        total_opacity_removed += opacity_removed

        points = scene_state.means[keep] * metres_per_unit
        colors = scene_state.colors[keep]
        opacities = scene_state.opacities[keep]
        scales = scene_state.scales[keep] * metres_per_unit
        quats = scene_state.quats[keep]
        count = int(points.shape[0])
        frame_counts.append(count)
        total_points += count

        frame_suffix = f"{suffix}_frame{frame_idx:02d}"
        gaussian_path = output_dir / f"generated_raw_gaussians__{frame_suffix}.ply"
        point_path = output_dir / f"generated_raw_points__{frame_suffix}.ply"
        write_gaussian_ply(
            {
                "means": points,
                "features_dc_rgb": colors,
                "opacities": opacities,
                "scales": scales,
                "quats": quats,
            },
            gaussian_path,
        )
        write_point_ply(points, colors, point_path, opacities=opacities)
        frame_summaries.append(
            {
                "frame": int(frame_idx),
                "gaussian_ply": str(gaussian_path),
                "point_ply": str(point_path),
                "num_points": count,
                "num_points_before_filter": before,
                "sky_removed": sky_removed,
                "low_opacity_removed": opacity_removed,
            }
        )

    c_depth_summary: dict[str, Any] | None = None
    if torch.is_tensor(c_depth_factor):
        factors = c_depth_factor.detach().float().reshape(-1)
        if factors.numel() == 0 or not bool(torch.isfinite(factors).all()):
            raise ValueError("c_depth_factor must be finite and non-empty")
        c_depth_summary = {
            "form": (
                calibration.depth_form
                if export_units == "metric" and calibration is not None
                else "identity"
            ),
            "factor_min": float(factors.min().item()),
            "factor_median": float(factors.median().item()),
            "factor_max": float(factors.max().item()),
        }
        if export_units == "metric" and calibration is not None:
            c_depth_summary.update(
                {
                    "a": float(calibration.depth_a),
                    "b": float(calibration.depth_b),
                    "reference_depth_m": float(calibration.reference_depth_m),
                    "runtime_depth_clamp_m": list(calibration.runtime_depth_clamp_m),
                }
            )
    summary = {
        "frames": frame_summaries,
        "num_points": int(total_points),
        "num_points_before_filter": int(total_before),
        "sky_removed": int(total_sky_removed),
        "low_opacity_removed": int(total_opacity_removed),
        "sky_probability_threshold": float(sky_probability_threshold),
        "min_effective_opacity": float(min_effective_opacity),
        "frame_counts": frame_counts,
        "stride": stride,
        "schema": "Per-frame DGGT Gaussian PLY + MeshLab xyz/uchar-RGB PLY",
        "scene_builder": "dggt.utils.gaussian_edit.build_clean_scene_state",
        "validity_rule": "generated depth > 1e-4 AND p_sky < threshold AND effective opacity >= threshold",
        "export_units": export_units,
        "coordinates": (
            "metres in generated-camera world coordinates"
            if export_units == "metric"
            else "DGGT generated-camera world coordinates"
        ),
        "metres_per_unit_applied": float(metres_per_unit),
        "gauge": gauge_summary,
        "c_depth": c_depth_summary,
        "c_gs": 1.0 if calibration is None else float(calibration.c_gs),
        "pullback_boundary": (
            PULLBACK_METRIC_BOUNDARY if export_units == "metric" else PULLBACK_RENDER_BOUNDARY
        ),
        "pullback_artifact_sha256": None if calibration is None else calibration.artifact_sha256,
        "pullback_runtime_contract_version": (
            None if calibration is None else calibration.runtime_contract_version
        ),
        "tokenizer_sha256": None if calibration is None else calibration.tokenizer_sha256,
        "merged_ply_saved": False,
        "meshlab_note": "Open one frame PLY at a time; merged multi-frame clouds visually over-densify points.",
    }
    if gauge_summary is not None:
        summary.update(
            {
                "log_metric_scale": gauge_summary["log_metric_scale"],
                "metres_per_unit": gauge_summary["metres_per_unit"],
                "fov_deg": gauge_summary["fov_deg"],
            }
        )
    return summary


@torch.no_grad()
def render_and_export_generated(
    *,
    batch: dict[str, Any],
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    generated_sample: Any,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    suffix: str,
    camera_condition_active: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any], dict[str, Any]]:
    z_generated = generated_sample.video
    camera_generated = generated_sample.camera_state_metric
    camera_anchor_mask = generated_sample.camera_anchor_mask
    camera_initial_c2w = generated_sample.camera_initial_c2w_metric
    camera_trajectory_anchor_to_world = (
        generated_sample.camera_trajectory_anchor_to_world_metric
    )
    gauge_generated = generated_sample.gauge
    sky_generated = generated_sample.sky
    sky_mask_patch = generated_sample.sky_mask_patch
    sky_mask_refined = generated_sample.sky_mask_refined
    if camera_generated is None:
        raise RuntimeError("Pretrain sampling did not generate camera tokens.")
    if gauge_generated is None:
        raise RuntimeError("Pretrain sampling did not generate the required scene gauge.")
    if sky_mask_patch is None:
        raise RuntimeError("Pretrain sampling did not generate a sky mask.")
    pullback_calibration = getattr(args, "pullback_calibration", None)
    if not isinstance(pullback_calibration, PullbackCalibration):
        raise RuntimeError("Inference is missing a validated checkpoint-bound pullback calibration.")

    seq_len = int(z_generated.shape[1])
    height, width = _fixed_render_hw(args)
    frames = min(int(args.val_log_images), seq_len)
    timestamps = _timestamps_for_generated_render(batch, seq_len=seq_len, device=device)

    patch_start_idx = int(getattr(vggt_model.aggregator, "patch_start_idx", 5))
    with autocast_context(args, device):
        geometry = decode_generated_dggt_geometry(
            vggt_model=vggt_model,
            scene_flow_root=unwrap_ddp(scene_flow),
            z_clean_pred_n=z_generated,
            patch_grid=args.patch_grid,
            patch_start_idx=patch_start_idx,
            image_hw=(height, width),
            pullback_calibration=pullback_calibration,
        )
        generated_sky_mask = _sky_mask_patch_to_image(
            sky_mask_refined if sky_mask_refined is not None else sky_mask_patch,
            patch_grid=args.patch_grid,
            height=height,
            width=width,
            device=device,
        )
    with torch.amp.autocast(device_type=device.type, enabled=False):
        generated_camera_trajectory = decode_metric_camera_from_features(
            camera_generated.to(device=device, dtype=torch.float32),
            camera_anchor_mask=camera_anchor_mask,
            initial_camera_to_world=camera_initial_c2w,
            trajectory_anchor_to_world=camera_trajectory_anchor_to_world,
        )
        camera_to_world_dggt = metric_c2w_to_teacher_anchor_dggt(
            generated_camera_trajectory.camera_to_world,
            camera_trajectory_anchor_to_world,
            gauge_generated[..., 0].to(device=device, dtype=torch.float32),
        )
        generated_pose = assemble_dggt_pose_encoding(
            camera_to_world_dggt,
            gauge_generated.to(device=device, dtype=torch.float32),
        )
    camera_summary, camera_tensor_artifact = build_generated_metric_camera_summary(
        camera_state_metric_9d=camera_generated,
        camera_to_world_metric=generated_camera_trajectory.camera_to_world,
        camera_anchor_mask=camera_anchor_mask,
        gauge=gauge_generated,
        waymo_camera_to_world_metric=batch.get("camera_to_world_corrected"),
        camera_condition_active=bool(camera_condition_active),
    )
    camera_artifact_path = output_dir / f"generated_metric_camera__{suffix}.pt"
    torch.save(camera_tensor_artifact, camera_artifact_path)
    camera_summary["tensor_artifact"] = str(camera_artifact_path)
    gs_map, gs_conf = geometry.gs_map, geometry.gs_conf
    generated_depth, generated_dynamic = geometry.depth, geometry.dynamic_conf
    render_geometry, export_geometry = prepare_generated_geometry_boundaries(
        depth=generated_depth,
        gs_map=gs_map,
        gauge=gauge_generated.to(device),
        calibration=pullback_calibration,
        export_units=str(args.export_units),
    )
    flow_consistency = camera_geometry_flow_consistency(
        render_geometry.depth_dggt,
        generated_pose,
        sky_probability=generated_sky_mask,
        dynamic_logits=generated_dynamic,
        sample_stride=int(args.flow_consistency_stride),
    )

    sky_background = None
    sky_grid_image = None
    if sky_generated is not None:
        sky_h, sky_w = args.sky_atlas_hw
        extrinsic, intrinsic = _predict_camera_mats(generated_pose, (height, width), device)
        sky_background = sky_tokens_to_background(
            decode_sky_patch_tokens(sky_generated.to(device)),
            seq_len=seq_len,
            height=height,
            width=width,
            grid_h=sky_h,
            grid_w=sky_w,
            extrinsics=extrinsic,
            intrinsics=intrinsic,
        )
        sky_grid_image = _sky_background_image_grid(sky_background, frames)

    images = {
        "generated_pred_sky_mask": _sky_mask_image_grid(generated_sky_mask, frames),
        "generated_raw_3dgs_rgb": _render_gs_map_rgb(
            vggt_model,
            None,
            generated_sky_mask,
            timestamps,
            generated_pose,
            render_geometry.depth_dggt,
            render_geometry.gs_map_dggt,
            gs_conf,
            generated_dynamic,
            device,
            frames,
            background_mode="black",
            use_sky_mask=True,
            background_override=sky_background,
            image_hw=(height, width),
            soft_sky_mask=True,
        ),
    }
    if sky_grid_image is not None:
        images["generated_sky_rgb"] = sky_grid_image

    scene_state = build_generated_dggt_scene_state(
        pose_enc=generated_pose,
        depth=export_geometry.depth_dggt,
        gs_map=export_geometry.gs_map_dggt,
        gs_conf=gs_conf,
        dynamic_conf=generated_dynamic,
        sky_mask=generated_sky_mask,
    )
    ply_summary = export_generated_pointclouds(
        scene_state=scene_state,
        output_dir=output_dir,
        suffix=suffix,
        stride=args.ply_stride,
        sky_probability_threshold=float(args.ply_sky_probability_threshold),
        min_effective_opacity=float(args.ply_min_effective_opacity),
        export_units=str(args.export_units),
        gauge=gauge_generated,
        c_depth_factor=export_geometry.c_depth_factor,
        calibration=pullback_calibration,
    )
    del (
        generated_camera_trajectory,
        camera_to_world_dggt,
        generated_pose,
        generated_depth,
        generated_dynamic,
        generated_sky_mask,
        gs_map,
        gs_conf,
        render_geometry,
        export_geometry,
        scene_state,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images, ply_summary, flow_consistency, camera_summary


def _first(value: Any, default: Any = None) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        result = value.reshape(-1)[0].item()
        return result
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def save_common_validation_images(
    bundle: Any,
    batch: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    frames = min(int(args.val_log_images), int(bundle.z_clean_n.shape[1]))
    images: dict[str, torch.Tensor] = {
        "target_latent_pca": _latent_pca_grid(bundle.z_clean_n, args.patch_grid, frames),
        "input_rgb_gt": _image_grid(batch["images"], frames),
    }
    sky_patch = getattr(bundle, "sky_mask_clean", None)
    if torch.is_tensor(sky_patch):
        images["target_sky_mask_patch"] = _mask_grid(sky_patch, args.patch_grid, frames)
    sky_refined = getattr(bundle, "sky_mask_refined_clean", None)
    if torch.is_tensor(sky_refined):
        images["target_sky_mask_refined"] = _sky_mask_image_grid(sky_refined, frames)

    paths: dict[str, str] = {}
    for name, tensor in images.items():
        path = output_dir / f"{name}.jpg"
        save_image_grid(tensor, path, nrow=frames)
        paths[name] = str(path)
    caption = str(_first(batch.get("caption"), ""))
    (output_dir / "caption.txt").write_text(caption + "\n")
    return paths


def save_cfg_images(
    z_generated: torch.Tensor,
    rgb_images: dict[str, torch.Tensor],
    output_dir: Path,
    args: argparse.Namespace,
    suffix: str,
) -> dict[str, str]:
    frames = min(int(args.val_log_images), int(z_generated.shape[1]))
    images = {
        f"generated_raw_latent_pca__{suffix}": _latent_pca_grid(
            z_generated, args.patch_grid, frames
        )
    }
    images.update({f"{name}__{suffix}": value for name, value in rgb_images.items()})
    paths: dict[str, str] = {}
    for name, tensor in images.items():
        path = output_dir / f"{name}.jpg"
        save_image_grid(tensor, path, nrow=frames)
        paths[name] = str(path)
    return paths


def camera_gauge_attribution_arm_inputs(
    *,
    teacher_camera: torch.Tensor,
    generated_camera: torch.Tensor,
    teacher_gauge: torch.Tensor,
    generated_gauge: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return the fixed 2x2 camera/gauge intervention design."""

    if teacher_camera.shape != generated_camera.shape:
        raise ValueError("teacher/generated camera tensors must have identical shapes")
    if teacher_gauge.shape != generated_gauge.shape:
        raise ValueError("teacher/generated gauge tensors must have identical shapes")
    return {
        "teacher_camera__teacher_gauge": (teacher_camera, teacher_gauge),
        "generated_camera__teacher_gauge": (generated_camera, teacher_gauge),
        "teacher_camera__generated_gauge": (teacher_camera, generated_gauge),
        "generated_camera__generated_gauge": (generated_camera, generated_gauge),
    }


@torch.no_grad()
def run_camera_gauge_attribution(
    *,
    batch: dict[str, Any],
    bundle: Any,
    generated_sample: Any,
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    suffix: str,
) -> dict[str, Any]:
    """Render the four camera/gauge arms with one fixed generated geometry.

    All four calls receive the same generated latent, sky, sky mask and camera
    integration anchors. Only the camera trajectory and the scene-global gauge
    change. This makes the 2x2 differences an actual intervention rather than
    a comparison across unrelated diffusion samples.
    """

    teacher_camera = getattr(bundle, "camera_target_state_metric", None)
    teacher_gauge = getattr(bundle, "scene_gauge_clean", None)
    generated_camera = generated_sample.camera_state_metric
    generated_gauge = generated_sample.gauge
    if not all(
        torch.is_tensor(value)
        for value in (teacher_camera, teacher_gauge, generated_camera, generated_gauge)
    ):
        raise RuntimeError("four-arm attribution requires teacher/generated camera and gauge tensors")
    arms = camera_gauge_attribution_arm_inputs(
        teacher_camera=teacher_camera,
        generated_camera=generated_camera,
        teacher_gauge=teacher_gauge,
        generated_gauge=generated_gauge,
    )
    frames = min(int(args.val_log_images), int(generated_sample.video.shape[1]))
    gt = _image_grid(batch["images"], frames).float()
    results: dict[str, dict[str, Any]] = {}
    l1_values: dict[str, float] = {}
    for arm_name, (camera_features, gauge) in arms.items():
        rendered = render_validation_generated_rgb(
            batch,
            vggt_model,
            scene_flow,
            generated_sample.video,
            args,
            device,
            generated_camera_features=camera_features,
            generated_camera_anchor_mask=generated_sample.camera_anchor_mask,
            generated_camera_initial_c2w=generated_sample.camera_initial_c2w_metric,
            generated_camera_anchor_c2w=(
                generated_sample.camera_trajectory_anchor_to_world_metric
            ),
            generated_gauge=gauge,
            generated_sky_tokens=generated_sample.sky,
            generated_sky_mask_patch=generated_sample.sky_mask_patch,
            generated_sky_mask_refined=generated_sample.sky_mask_refined,
        )
        rgb = rendered["generated_raw_3dgs_rgb"].detach().float().cpu()
        if rgb.shape != gt.shape:
            raise RuntimeError(
                f"attribution render shape {tuple(rgb.shape)} != GT {tuple(gt.shape)}"
            )
        error = rgb - gt
        l1 = float(error.abs().mean().item())
        mse = float(error.square().mean().item())
        psnr = float(-10.0 * math.log10(max(mse, 1.0e-12)))
        path = output_dir / f"camera_gauge_arm__{arm_name}__{suffix}.jpg"
        save_image_grid(rgb, path, nrow=frames)
        results[arm_name] = {
            "l1": l1,
            "mse": mse,
            "psnr_db": psnr,
            "image": str(path),
        }
        l1_values[arm_name] = l1
        del rendered, rgb, error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tt, gt_arm, tg, gg = (
        l1_values[name] for name in CAMERA_GAUGE_ATTRIBUTION_ARMS
    )
    effects = {
        "camera_l1_effect_at_teacher_gauge": gt_arm - tt,
        "gauge_l1_effect_at_teacher_camera": tg - tt,
        "camera_gauge_l1_interaction": gg - gt_arm - tg + tt,
        "end_to_end_l1_gap_from_teacher_teacher": gg - tt,
    }
    payload = {
        "schema": "scene_flow_camera_gauge_four_arm_v1",
        "fixed": ["generated_video_latent", "generated_sky", "generated_sky_mask", "noise_seed"],
        "arms": results,
        "effects": effects,
        "interpretation": (
            "positive L1 effects are degradations; interaction is the residual beyond "
            "the additive camera-only and gauge-only effects"
        ),
    }
    json_path = output_dir / f"camera_gauge_attribution__{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    payload["json"] = str(json_path)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-scene SceneFlow pretrain inference on raw Waymo validation clips."
    )
    parser.add_argument(
        "--weights",
        "--scene_flow_ckpt_path",
        dest="weights",
        default=DEFAULT_WEIGHTS,
        help="Required v2-only pretrain full/EMA checkpoint; v1-bound checkpoints are rejected.",
    )
    parser.add_argument("--dggt_ckpt_path", default=DEFAULT_DGGT_CKPT)
    parser.add_argument("--tokenizer_ckpt_path", default=DEFAULT_TOKENIZER_CKPT)
    parser.add_argument(
        "--pullback_calibration_path",
        default=DEFAULT_PULLBACK_CALIBRATION,
        help="Strict checkpoint-bound tokenizer pullback artifact.",
    )
    parser.add_argument(
        "--feature_stats_path",
        default=str(DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH),
        help=(
            "Tokenizer-v2 v5 stats contract; it must exactly match the SceneFlow "
            "checkpoint buffers."
        ),
    )
    parser.add_argument("--val_image_dir", "--image_dir", dest="val_image_dir", default=DEFAULT_VAL_ROOT)
    parser.add_argument(
        "--val_scene_gauge_path",
        default=DEFAULT_VAL_SCENE_GAUGE,
        help="Offline full-29-frame validation gauge table (raw Waymo mode only).",
    )
    parser.add_argument("--val_caption_root", "--caption_root", dest="val_caption_root", default=DEFAULT_CAPTION_ROOT)
    parser.add_argument("--text_encoder_path", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--no_text_condition", action="store_true")
    parser.add_argument("--output_dir", default="runs/scene_flow_pretrain_inference")
    parser.add_argument(
        "--camera_gauge_attribution",
        action="store_true",
        help=(
            "Run the fixed-geometry four-arm render audit: teacher/generated camera "
            "crossed with teacher/generated gauge. Intended for early-checkpoint diagnosis."
        ),
    )
    parser.add_argument(
        "--asset_manifest",
        default=None,
        help=(
            "External factorized asset manifest (RGBA or RGB+mask, box keyframes, "
            "camera_to_anchor, intrinsics and FPS). This mode does not read a target video."
        ),
    )

    parser.add_argument("--val_scene_start", type=int, default=0)
    parser.add_argument("--val_scene_end", type=int, default=100)
    parser.add_argument("--start", type=int, default=0, help="First trunk-major validation dataset row.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive row; default processes the selected dataset.")
    parser.add_argument("--index", type=int, default=None, help="Process one trunk-major dataset row.")
    parser.add_argument(
        "--num_frames",
        "--frames",
        dest="num_frames",
        type=int,
        default=10,
        help="Contiguous frames per validation sample (default: 10).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument(
        "--cfg",
        "--cfg_scales",
        "--guidance_scales",
        dest="cfg_values",
        nargs="+",
        default=["1.0"],
        help="One or more comma/space-separated CFG scales (default: 1.0).",
    )
    parser.add_argument(
        "--asset_control_guidance_scale",
        type=float,
        default=1.0,
        help=(
            "Independent asset-condition guidance scale. Default 1.0 keeps the supplied visual "
            "condition fixed while --cfg sweeps text guidance, matching Cosmos conditional generation."
        ),
    )
    parser.add_argument(
        "--camera_guidance_scale",
        type=float,
        default=1.0,
        help=(
            "Independent camera-condition guidance scale. Default 1.0 keeps the requested camera "
            "trajectory fixed while --cfg sweeps text guidance."
        ),
    )
    parser.add_argument(
        "--camera_text_guidance_scale",
        type=float,
        default=1.0,
        help="Independent text-CFG scale for generated camera state.",
    )
    parser.add_argument("--sample_steps", "--val_sample_steps", dest="val_sample_steps", type=int, default=35)
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Optional assertion for the flow shift stored in the checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_ema", action="store_true", help="Use raw weights from a full checkpoint.")
    parser.add_argument("--no_sky_generation", action="store_true")
    parser.add_argument(
        "--export_units",
        choices=("dggt", "metric"),
        default="metric",
        help="Point/Gaussian PLY coordinate units; metric applies the validated pullback and gauge.",
    )
    parser.add_argument(
        "--ply_stride",
        type=int,
        default=1,
        help=(
            "Pixel-grid stride applied after DGGT canonical scene construction. "
            "Default 1 preserves the exact DGGT scene; >1 is explicit export-only downsampling."
        ),
    )
    parser.add_argument("--ply_sky_probability_threshold", type=float, default=0.5)
    parser.add_argument("--ply_min_effective_opacity", type=float, default=0.01)
    parser.add_argument(
        "--flow_consistency_stride",
        type=int,
        default=4,
        help=(
            "Pixel sampling stride for the generated static-geometry reprojection/cycle "
            "diagnostic. Coordinates and errors remain in full-resolution pixels."
        ),
    )
    parser.add_argument(
        "--val_sliding_window",
        type=int,
        default=10,
        help=(
            "Offline SceneFlow window size, capped at 10. Values <=0 select automatic mode; "
            "requests longer than 10 frames always use overlapping sliding windows."
        ),
    )
    parser.add_argument(
        "--val_sliding_stride",
        type=int,
        default=7,
        help="Sliding stride; it must be smaller than the effective window for long clips.",
    )

    # Fallback architecture only; saved scene_flow_config takes precedence.
    parser.add_argument("--patch_grid_h", type=int, default=25)
    parser.add_argument("--patch_grid_w", type=int, default=37)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--prediction_type", choices=("x", "v"), default="x")
    parser.add_argument("--sky_grid_h", type=int, default=DEFAULT_SKY_GRID[0])
    parser.add_argument("--sky_grid_w", type=int, default=DEFAULT_SKY_GRID[1])
    parser.add_argument("--sky_mask_refine_scale", type=int, default=4)
    parser.add_argument("--sky_mask_refine_channels", type=int, default=256)
    parser.add_argument("--asset_position_mode", choices=("localized", "canonical"), default="localized")
    parser.add_argument("--sky_unobserved_loss_weight", type=float, default=0.0)
    return parser


def validate_args(args: argparse.Namespace) -> list[float]:
    if bool(args.camera_gauge_attribution) and args.asset_manifest:
        raise ValueError(
            "--camera_gauge_attribution requires raw Waymo teacher camera/gauge targets; "
            "it is unavailable with --asset_manifest."
        )
    if int(args.num_frames) < 2:
        raise ValueError("--num_frames must be >= 2 (the raw dataset timestamp path and pretrain require it).")
    if int(args.num_frames) > 29:
        raise ValueError("--num_frames cannot exceed the 29-frame validation caption trunk.")
    if int(args.val_sample_steps) <= 0:
        raise ValueError("--sample_steps must be positive.")
    if int(args.ply_stride) <= 0:
        raise ValueError("--ply_stride must be positive.")
    if int(args.flow_consistency_stride) <= 0:
        raise ValueError("--flow_consistency_stride must be positive.")
    if not 0.0 <= float(args.ply_sky_probability_threshold) <= 1.0:
        raise ValueError("--ply_sky_probability_threshold must be in [0,1].")
    if not 0.0 <= float(args.ply_min_effective_opacity) <= 1.0:
        raise ValueError("--ply_min_effective_opacity must be in [0,1].")
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be non-negative.")
    if int(args.val_scene_end) <= int(args.val_scene_start):
        raise ValueError("--val_scene_end must be greater than --val_scene_start.")
    return parse_cfg_scales(args.cfg_values)


def main() -> None:
    args = build_argparser().parse_args()
    cfg_scales = validate_args(args)
    args.val_sliding_window, args.val_sliding_stride, offline_sliding = resolve_offline_window(
        int(args.num_frames),
        int(args.val_sliding_window),
        int(args.val_sliding_stride),
    )
    args.patch_grid = (int(args.patch_grid_h), int(args.patch_grid_w))
    args.sky_grid = (int(args.sky_grid_h), int(args.sky_grid_w))
    # Architecture construction is checkpoint-only; this value exists for
    # argument/config reporting before the strict checkpoint sync.
    args.camera_gen_dim = CAMERA_GENERATION_DIM
    args.caption_root = args.val_caption_root
    args.val_log_images = int(args.num_frames)
    # `--cfg` is text guidance.  As in Cosmos conditional video generation,
    # clean structural conditions remain present in both CFG branches instead
    # of being amplified together with text. Asset/camera can still be swept
    # explicitly through their independent flags.
    args.guidance_scale = 1.0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    seed_everything(int(args.seed))

    if not args.weights:
        raise ValueError(
            "--weights is required; no v1-bound SceneFlow checkpoint is accepted as a default"
        )

    scene_flow, checkpoint_info = build_scene_flow_from_checkpoint(
        args.weights,
        device=device,
        no_ema=bool(args.no_ema),
        fallback_args=args,
    )
    sync_args_from_model(args, scene_flow)
    raw_provenance = checkpoint_info.get("metric_gauge_provenance")
    recorded_tokenizer_sha256 = (
        raw_provenance.get("tokenizer_sha256")
        if isinstance(raw_provenance, Mapping)
        else ""
    )
    recorded_dggt_sha256 = (
        raw_provenance.get("dggt_checkpoint_sha256")
        if isinstance(raw_provenance, Mapping)
        else ""
    )
    provenance = validate_metric_gauge_provenance(
        raw_provenance,
        scene_flow=scene_flow,
        dggt_sha256=recorded_dggt_sha256,
        tokenizer_sha256=recorded_tokenizer_sha256,
        expected_window_len=int(args.val_sliding_window),
        expected_patch_grid=args.patch_grid,
    )
    pullback_calibration = load_pullback_calibration(
        args.pullback_calibration_path,
        tokenizer_checkpoint_path=args.tokenizer_ckpt_path,
        dggt_checkpoint_path=args.dggt_ckpt_path,
        expected_window_len=int(args.val_sliding_window),
        expected_patch_grid=args.patch_grid,
        expected_artifact_sha256=provenance["pullback_artifact_sha256"],
    )
    provenance = validate_metric_gauge_provenance(
        provenance,
        scene_flow=scene_flow,
        dggt_sha256=pullback_calibration.dggt_sha256,
        tokenizer_sha256=pullback_calibration.tokenizer_sha256,
        expected_pullback_runtime_contract_version=(
            pullback_calibration.runtime_contract_version
        ),
        expected_window_len=int(args.val_sliding_window),
        expected_patch_grid=args.patch_grid,
    )
    checkpoint_info["metric_gauge_provenance"] = provenance
    stats_contract = validate_pretrain_feature_stats_contract(
        checkpoint_info.get(PRETRAIN_FEATURE_STATS_CONTRACT_KEY),
        path=args.weights,
        expected_sequence_length=int(args.val_sliding_window),
        expected_patch_grid=args.patch_grid,
    )
    checkpoint_info[PRETRAIN_FEATURE_STATS_CONTRACT_KEY] = stats_contract
    args.pullback_calibration = pullback_calibration
    scene_flow._pullback_calibration = pullback_calibration
    if args.feature_stats_path:
        runtime_stats_sha256 = checkpoint_sha256(args.feature_stats_path)
        validate_pretrain_feature_stats_contract(
            stats_contract,
            path=args.weights,
            expected_feature_stats_sha256=runtime_stats_sha256,
            expected_sequence_length=int(args.val_sliding_window),
            expected_patch_grid=args.patch_grid,
        )
        load_all_stats_into_buffers(
            scene_flow,
            args.feature_stats_path,
            token_dim=int(args.latent_dim),
            dggt_ckpt_path=args.dggt_ckpt_path,
            expected_tokenizer_sha256=pullback_calibration.tokenizer_sha256,
            expected_sequence_length=int(args.val_sliding_window),
            expected_patch_grid=args.patch_grid,
            expected_scene_gauge_sha256=provenance["gauge_table_sha256"],
            require_existing_match=True,
        )
    scene_flow.require_camera_stats()
    scene_flow.require_gauge_stats()

    print(
        f"[checkpoint] {checkpoint_info['weight_source']} step={checkpoint_info['step']} "
        f"grid={args.patch_grid} latent_dim={args.latent_dim} prediction={args.prediction_type}",
        flush=True,
    )
    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path, args.tokenizer_ckpt_path, device
    )
    if (
        str(scene_flow.config.camera_generation_representation) == CAMERA_GENERATION_REPRESENTATION
        and int(scene_flow.config.camera_gen_dim) != CAMERA_GENERATION_DIM
    ):
        raise ValueError(
            "Waymo metric camera v4 SceneFlow "
            f"camera_gen_dim={scene_flow.config.camera_gen_dim} must be {CAMERA_GENERATION_DIM}."
        )
    text_encoder = setup_text_encoder(args, device)

    if args.asset_manifest:
        bundle, manifest_info = build_external_factorized_pretrain_bundle(
            args.asset_manifest,
            vggt_model=vggt_model,
            scene_flow=scene_flow,
            device=device,
            patch_grid=args.patch_grid,
        )
        args.num_frames = int(bundle.z_clean_n.shape[1])
        args.val_log_images = int(args.num_frames)
        args.val_sliding_window, args.val_sliding_stride, offline_sliding = resolve_offline_window(
            int(args.num_frames),
            int(args.val_sliding_window),
            int(args.val_sliding_stride),
        )
        if int(args.val_sliding_window) != int(pullback_calibration.window_len):
            raise ValueError(
                "External manifest changed the effective inference window after pullback validation: "
                f"window={args.val_sliding_window}, artifact={pullback_calibration.window_len}."
            )
        apply_condition_mode(bundle, "asset_cam")
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        results = []
        for scale in cfg_scales:
            args.guidance_scale = float(scale)
            generated = cfg_sample_pretrain_latents(
                scene_flow,
                bundle,
                args,
                step=0,
                device=device,
                guidance_scale=float(scale),
                text_encoder=text_encoder,
                return_camera=False,
                return_sky=sky_generation_enabled(args),
                return_gauge=True,
                return_sky_mask=True,
            )
            if generated.gauge is None:
                raise RuntimeError("External factorized sampling did not return the required scene gauge.")
            gauge_summary = _single_sample_gauge_summary(generated.gauge)
            output_path = output_root / f"external_factorized__cfg{cfg_tag(scale)}.pt"
            torch.save(
                {
                    "video_latent_normalized": generated.video.detach().cpu(),
                    "sky_tokens": None if generated.sky is None else generated.sky.detach().cpu(),
                    "sky_mask_patch": None
                    if generated.sky_mask_patch is None
                    else generated.sky_mask_patch.detach().cpu(),
                    "gauge": generated.gauge.detach().cpu(),
                    "frame_ids": bundle.frame_ids.detach().cpu(),
                    "fps": bundle.fps.detach().cpu(),
                    "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
                },
                output_path,
            )
            results.append(
                {"cfg": float(scale), "output": str(output_path), "gauge": gauge_summary}
            )
        summary = {
            "mode": "external_factorized_asset",
            "target_video_read": False,
            "target_dynamic_mask_read": False,
            "manifest": manifest_info,
            "checkpoint": checkpoint_info,
            "pullback": {
                "artifact_sha256": pullback_calibration.artifact_sha256,
                "tokenizer_sha256": pullback_calibration.tokenizer_sha256,
                "dggt_checkpoint_sha256": pullback_calibration.dggt_sha256,
                "runtime_contract_version": (
                    pullback_calibration.runtime_contract_version
                ),
                "c_depth": {
                    "form": pullback_calibration.depth_form,
                    "a": pullback_calibration.depth_a,
                    "b": pullback_calibration.depth_b,
                },
                "c_gs": pullback_calibration.c_gs,
            },
            "camera_geometry_flow_consistency": {
                "schema": CAMERA_GEOMETRY_FLOW_DIAGNOSTIC_SCHEMA,
                "status": "not_computed",
                "reason": (
                    "external factorized mode saves generated latents/gauge without decoding "
                    "camera and geometry"
                ),
                "requires_gt_images": False,
            },
            "window": int(args.val_sliding_window),
            "window_stride": int(args.val_sliding_stride),
            "sliding_window_active": bool(offline_sliding),
            "results": results,
        }
        (output_root / "external_factorized_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print(f"[done] wrote external factorized outputs under {output_root}", flush=True)
        return

    scene_names = discover_scene_names(args.val_image_dir, args.val_scene_start, args.val_scene_end)
    dataset = WaymoOpenDataset(
        image_dir=args.val_image_dir,
        scene_names=scene_names,
        sequence_length=int(args.num_frames),
        start_idx=0,
        mode=1,
        views=1,
        caption_root=None if bool(args.no_text_condition) else args.val_caption_root,
        pretrain_patch_grid=args.patch_grid,
        trunk_major_samples=True,
        trunk_frames=29,
        return_full_dggt_context=True,
        load_dynamic_masks=False,
        binary_mask_channels=1,
        image_output_dtype="uint8",
        scene_gauge_path=args.val_scene_gauge_path,
        expected_scene_gauge_dggt_sha256=pullback_calibration.dggt_sha256,
        expected_scene_gauge_split=Path(args.val_image_dir).name,
    )
    if len(dataset) == 0:
        raise RuntimeError("The selected validation dataset has no full caption-trunk samples.")

    if args.index is not None:
        start = int(args.index)
        end = start + 1
    else:
        start = int(args.start)
        end = len(dataset) if args.end is None else int(args.end)
    start = max(0, min(start, len(dataset)))
    end = max(start, min(end, len(dataset)))
    if start == end:
        raise RuntimeError(f"No validation rows selected: [{start}, {end}) of {len(dataset)}.")

    sampler = CyclicSequentialSampler(dataset)
    sampler.set_offset(start)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        drop_last=False,
    )
    selected_count = end - start
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[data] scenes={len(scene_names)} rows={len(dataset)} selected=[{start},{end}) "
        f"order=trunk-major modes={','.join(CONDITION_MODES)} frames={args.num_frames}",
        flush=True,
    )
    print(
        f"[sampling] cfg={cfg_scales} steps={args.val_sample_steps} shift={args.shift} "
        f"window={args.val_sliding_window} stride={args.val_sliding_stride} "
        f"sliding={offline_sliding}",
        flush=True,
    )

    all_summaries: list[dict[str, Any]] = []
    run_flow_diagnostics: list[dict[str, Any]] = []
    iterator = tqdm(loader, total=selected_count, desc="pretrain inference", dynamic_ncols=True)
    for position, batch in enumerate(iterator):
        if position >= selected_count:
            break
        dataset_index = start + position
        # Bind the condition cycle to the global trunk-major row, not to the
        # current --start offset.  Sub-runs therefore keep the same mode for a
        # given validation sample.
        mode = condition_mode_for_position(dataset_index)
        scene_name = str(_first(batch.get("scene_name"), "unknown"))
        start_idx = int(_first(batch.get("start_idx"), 0))
        clip_index = int(_first(batch.get("clip_index"), start_idx // 29))
        sample_dir = output_root / (
            f"{dataset_index:06d}_scene{scene_name}_clip{clip_index:02d}_start{start_idx:03d}_{mode}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        bundle = build_pretrain_bundle_from_batch(batch, vggt_model, scene_flow, device, args)
        sliding_asset_info = {"active": False, "windows": []}
        if offline_sliding:
            sliding_asset_info = attach_training_equivalent_sliding_asset_conditions(
                bundle,
                dataset=dataset,
                dataset_index=dataset_index,
                batch=batch,
                vggt_model=vggt_model,
                scene_flow=scene_flow,
                device=device,
                patch_grid=args.patch_grid,
                window=int(args.val_sliding_window),
                stride=int(args.val_sliding_stride),
            )
        apply_condition_mode(bundle, mode)
        condition_rows = actual_condition_rows(bundle)
        common_paths = save_common_validation_images(bundle, batch, sample_dir, args)

        per_cfg: list[dict[str, Any]] = []
        for scale in cfg_scales:
            # Same sample step across CFG values => identical initial video,
            # camera and sky noise, matching training validation comparisons.
            args.guidance_scale = float(scale)
            generated = cfg_sample_pretrain_latents(
                scene_flow,
                bundle,
                args,
                step=dataset_index,
                device=device,
                guidance_scale=float(scale),
                text_encoder=text_encoder,
                return_camera=True,
                return_sky=sky_generation_enabled(args),
                return_gauge=True,
                return_sky_mask=True,
            )
            suffix = f"cfg{cfg_tag(scale)}"
            (
                rgb_images,
                ply_summary,
                flow_consistency,
                generated_metric_camera,
            ) = render_and_export_generated(
                batch=batch,
                vggt_model=vggt_model,
                scene_flow=scene_flow,
                generated_sample=generated,
                args=args,
                device=device,
                output_dir=sample_dir,
                suffix=suffix,
                camera_condition_active=bool(condition_rows["camera"][0]),
            )
            image_paths = save_cfg_images(
                generated.video, rgb_images, sample_dir, args, suffix
            )
            attribution = None
            if bool(args.camera_gauge_attribution):
                attribution = run_camera_gauge_attribution(
                    batch=batch,
                    bundle=bundle,
                    generated_sample=generated,
                    vggt_model=vggt_model,
                    scene_flow=scene_flow,
                    args=args,
                    device=device,
                    output_dir=sample_dir,
                    suffix=suffix,
                )
            per_cfg.append(
                {
                    "cfg": float(scale),
                    "noise_seed": int(args.seed) + dataset_index,
                    "factored_scales": {
                        "text": float(scale),
                        "asset": float(args.asset_control_guidance_scale),
                        "camera": float(args.camera_guidance_scale),
                        "camera_text": float(args.camera_text_guidance_scale),
                    },
                    "images": image_paths,
                    "pointcloud": ply_summary,
                    "generated_metric_camera": generated_metric_camera,
                    "camera_geometry_flow_consistency": flow_consistency,
                    "camera_gauge_attribution": attribution,
                }
            )
            run_flow_diagnostics.append(
                {
                    "dataset_index": dataset_index,
                    "scene_name": scene_name,
                    "clip_index": clip_index,
                    "start_idx": start_idx,
                    "cfg": float(scale),
                    "status": flow_consistency["status"],
                    "pair_count": flow_consistency["pair_count"],
                    "informative_pair_count": flow_consistency[
                        "informative_pair_count"
                    ],
                    "pair_status_counts": flow_consistency["pair_status_counts"],
                    "support": flow_consistency["support"],
                    "all_supported_metrics": flow_consistency[
                        "all_supported_metrics"
                    ],
                    "informative_metrics": flow_consistency[
                        "informative_metrics"
                    ],
                }
            )
            del generated, rgb_images, flow_consistency
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = {
            "dataset_index": dataset_index,
            "dataset_order": "trunk-major: every scene trunk 0, then every scene trunk 1, ...",
            "scene_name": scene_name,
            "clip_index": clip_index,
            "start_idx": start_idx,
            "frame_ids": batch["frame_ids"].detach().cpu().tolist()
            if torch.is_tensor(batch.get("frame_ids"))
            else batch.get("frame_ids"),
            "image_paths": batch.get("image_paths"),
            "caption": str(_first(batch.get("caption"), "")),
            "caption_path": str(_first(batch.get("caption_path"), "")),
            "condition_mode": mode,
            "condition_policy": "text always; optional asset/camera rotate none,asset,cam,asset_cam",
            "actual_condition_rows": condition_rows,
            "num_frames": int(args.num_frames),
            "window": int(args.val_sliding_window),
            "window_stride": int(args.val_sliding_stride),
            "sliding_window_active": bool(offline_sliding),
            "sliding_asset_reference_policy": sliding_asset_info,
            "sample_steps": int(args.val_sample_steps),
            "shift": float(args.shift),
            "checkpoint": checkpoint_info,
            "export_units": str(args.export_units),
            "validation_gauge_table_sha256": dataset.scene_gauge_sha256,
            "generated_gauges": [item["pointcloud"]["gauge"] for item in per_cfg],
            "generated_metric_cameras": [
                item["generated_metric_camera"] for item in per_cfg
            ],
            "pullback": {
                "artifact_sha256": pullback_calibration.artifact_sha256,
                "tokenizer_sha256": pullback_calibration.tokenizer_sha256,
                "dggt_checkpoint_sha256": pullback_calibration.dggt_sha256,
                "runtime_contract_version": (
                    pullback_calibration.runtime_contract_version
                ),
                "c_depth": {
                    "form": (
                        pullback_calibration.depth_form
                        if args.export_units == "metric"
                        else "identity"
                    ),
                    "a": pullback_calibration.depth_a,
                    "b": pullback_calibration.depth_b,
                },
                "c_gs": pullback_calibration.c_gs,
            },
            "common_images": common_paths,
            "cfg_results": per_cfg,
            "abs_error_saved": False,
            "rgb_compositing": "gsplat_premultiplied_over_background",
        }
        (sample_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        all_summaries.append(summary)
        iterator.set_postfix(scene=scene_name, mode=mode)
        del bundle, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_summary = {
        "weights": str(args.weights),
        "checkpoint": checkpoint_info,
        "validation_root": str(args.val_image_dir),
        "caption_root": None if args.no_text_condition else str(args.val_caption_root),
        "validation_gauge_table": str(args.val_scene_gauge_path),
        "validation_gauge_table_sha256": dataset.scene_gauge_sha256,
        "gauge_table_sha256": provenance["gauge_table_sha256"],
        "tokenizer_sha256": pullback_calibration.tokenizer_sha256,
        "dggt_checkpoint_sha256": pullback_calibration.dggt_sha256,
        "pullback_artifact_sha256": pullback_calibration.artifact_sha256,
        "pullback_runtime_contract_version": (
            pullback_calibration.runtime_contract_version
        ),
        "c_depth": {
            "form": (
                pullback_calibration.depth_form
                if args.export_units == "metric"
                else "identity"
            ),
            "a": pullback_calibration.depth_a,
            "b": pullback_calibration.depth_b,
        },
        "c_gs": pullback_calibration.c_gs,
        "export_units": str(args.export_units),
        "rgb_compositing": "gsplat_premultiplied_over_background",
        "selected_range": [start, end],
        "cfg_scales": cfg_scales,
        "asset_control_guidance_scale": float(args.asset_control_guidance_scale),
        "camera_guidance_scale": float(args.camera_guidance_scale),
        "camera_text_guidance_scale": float(args.camera_text_guidance_scale),
        "generated_metric_camera_schema": GENERATED_METRIC_CAMERA_SCHEMA,
        "condition_cycle": list(CONDITION_MODES),
        "num_frames": int(args.num_frames),
        "window": int(args.val_sliding_window),
        "window_stride": int(args.val_sliding_stride),
        "sliding_window_active": bool(offline_sliding),
        "sample_steps": int(args.val_sample_steps),
        "shift": float(args.shift),
        "ply_stride": int(args.ply_stride),
        "camera_geometry_flow_consistency": {
            "schema": CAMERA_GEOMETRY_FLOW_DIAGNOSTIC_SCHEMA,
            "name": "generated static-geometry reprojection/cycle diagnostic",
            "is_independently_predicted_optical_flow": False,
            "requires_gt_images": False,
            "sample_stride": int(args.flow_consistency_stride),
            "results": run_flow_diagnostics,
        },
        "samples": all_summaries,
    }
    (output_root / "all_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))
    print(f"[done] wrote {len(all_summaries)} sample folders under {output_root}", flush=True)


if __name__ == "__main__":
    main()
