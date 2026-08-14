#!/usr/bin/env python3
"""Offline layout-v2 SceneFlow inference on requested Waymo cameras.

The runtime contract is intentionally narrow: C is always supplied by the
request, M/G are projected atomically for that C, A only binds appearance to G,
and the sampled gauge remains scene-global.  Long clips use the training
sampler's strict LayoutConditionBatch slicing and cosine overlap merge.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Subset, default_collate
from tqdm.auto import tqdm

# The raw pretrain path does not call Open3D.  Keeping imports lightweight also
# makes --help and the pure P9 contract tests usable on CPU-only hosts.
try:
    import open3d as _open3d  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open3d"] = types.ModuleType("open3d")

try:
    import gsplat as _gsplat  # noqa: F401
except ModuleNotFoundError:
    gsplat_stub = types.ModuleType("gsplat")
    gsplat_rendering_stub = types.ModuleType("gsplat.rendering")

    def _missing_gsplat(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "gsplat is required for SceneFlow RGB rendering in the training environment"
        )

    gsplat_stub.rasterization = _missing_gsplat
    gsplat_rendering_stub.rasterization = _missing_gsplat
    sys.modules["gsplat"] = gsplat_stub
    sys.modules["gsplat.rendering"] = gsplat_rendering_stub

from datasets.dataset import WaymoOpenDataset
from datasets.tools.hdmap_schema import RASTER_SCHEMA_HASH
from dggt.utils.layout_raster import STATIC_FAR_PLANE_M
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.actor_geometry_condition import LayoutMode
from dggt.utils.flow_schedule import (
    FLOW_SCHEDULE_VERSION,
    resolve_inference_flow_schedule,
)
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.gaussian_edit import CleanSceneState, build_clean_scene_state
from dggt.utils.gaussian_ply import write_gaussian_ply, write_point_ply
from dggt.utils.layout_condition import (
    LAYOUT_CONDITION_VERSION,
    LayoutConditionBatch,
    MapMode,
    assert_layout_overlap_consistent,
    neutralize_raster_rows,
    required_cfg_branches,
)
from dggt.utils.scene_gauge import (
    PULLBACK_METRIC_BOUNDARY,
    PULLBACK_RENDER_BOUNDARY,
    SCENE_GAUGE_DIM,
    SCENE_GAUGE_REPRESENTATION,
    PullbackCalibration,
    PullbackResult,
    apply_pullback_calibration,
    gauge_to_pose_enc_fov,
)
from dggt.utils.sliding_window import (
    OFFLINE_MAX_SINGLE_WINDOW,
    default_window_stride,
    window_slices,
)
from train_scene_flow_pretrain import (
    DEFAULT_SKY_GRID,
    T59_VALIDATION_SAMPLE_STEPS,
    build_layout_condition_from_batch,
    build_pretrain_bundle_from_batch,
    cfg_sample_pretrain_latents,
    discover_scene_names,
    load_dggt_aggregator_and_tokenizer,
    load_scene_flow_state_dict_strict,
    render_validation_generated_rgb,
    requested_render_pose_encoding,
    seed_everything,
    setup_text_encoder,
    sky_generation_enabled,
    validate_scene_flow_checkpoint_config,
)


DEFAULT_WEIGHTS = None
DEFAULT_VAL_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt/validation"
DEFAULT_HDMAP_ROOT = (
    "/data/disk2/lyy_dataset/waymo_processed_dggt/validation_hdmap"
)
DEFAULT_CAPTION_ROOT = (
    "/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions"
)
DEFAULT_DGGT_CKPT = (
    "/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt"
)
DEFAULT_TOKENIZER_CKPT = (
    "logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt"
)
DEFAULT_TEXT_ENCODER = "/home/dancer/model/Qwen/Qwen3-0.6B"

CONDITION_MODES = ("tc", "tcmg", "tcmga")
LAYOUT_MODES = ("full", "empty", "null")
SCENE_FLOW_DERIVED_CONFIG_KEYS = (
    "hidden_size",
    "rope_layout_version",
    "sky_rope_temporal_offset",
    "camera_rope_spatial_mode",
)


def parse_cfg_scales(values: Sequence[str]) -> list[float]:
    """Parse comma/space separated text guidance scales without deduplicating."""

    result: list[float] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                scale = float(token)
                if not math.isfinite(scale) or scale < 0.0:
                    raise ValueError("CFG scales must be finite and non-negative")
                result.append(scale)
    if not result:
        raise ValueError("at least one CFG scale is required")
    return result


def cfg_tag(scale: float) -> str:
    return f"{float(scale):g}".replace("-", "m").replace(".", "p")


def _first(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return default if not value else _first(value[0], default)
    return value


def _single_string(value: Any, default: str = "") -> str:
    return str(_first(value, default))


def _collate_layout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the same one-row collation used by the inference DataLoader."""

    return default_collate([payload])


def _replace_batch_layout_payload(
    batch: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    collated = _collate_layout_payload(payload)
    for key, value in collated.items():
        batch[key] = value


def _actor_raster_empty(raster: torch.Tensor) -> torch.Tensor:
    return neutralize_raster_rows(
        raster,
        torch.ones(int(raster.shape[0]), dtype=torch.bool, device=raster.device),
        channel_start=22,
        channel_end=int(raster.shape[2]),
    )


def apply_requested_layout_mode(
    layout: LayoutConditionBatch,
    appearance_class_id: torch.Tensor,
    mode: str,
) -> tuple[LayoutConditionBatch, torch.Tensor]:
    """Apply the requested G semantics while preserving the M/G task contract."""

    mode = str(mode)
    if mode not in LAYOUT_MODES:
        raise ValueError(f"unknown layout mode {mode!r}")
    if mode == "full":
        return layout, appearance_class_id
    if mode == "null":
        return layout.without_layout(), torch.full_like(appearance_class_id, -1)
    geometry = layout.actor_geometry.empty_like()
    empty_layout = replace(
        layout,
        raster=_actor_raster_empty(layout.raster),
        actor_geometry=geometry,
        projected_actor_geometry=layout.projected_actor_geometry.null_like(),
        appearance=layout.appearance.null_like(),
    )
    return empty_layout, torch.full_like(appearance_class_id, -1)


def apply_condition_mode(
    bundle: Any,
    mode: str,
    *,
    layout_mode: str,
) -> dict[str, bool]:
    """Apply one explicit TC, TCMG, or TCMGA inference task."""

    if mode not in CONDITION_MODES:
        raise ValueError(f"unknown condition mode {mode!r}")
    layout, class_id = apply_requested_layout_mode(
        bundle.layout_condition,
        bundle.appearance_class_id,
        layout_mode,
    )
    if mode == "tc":
        layout = layout.without_layout()
        class_id = torch.full_like(class_id, -1)
    elif mode == "tcmg":
        layout = layout.without_appearance()
        class_id = torch.full_like(class_id, -1)
    bundle.layout_condition = layout
    bundle.appearance_class_id = class_id
    has_layout = _layout_present(layout)
    has_appearance = bool(layout.appearance.binding_valid.any().item())
    return {
        "camera": True,
        "layout": has_layout,
        "appearance": has_appearance,
    }


def inference_cfg_branches(
    bundle: Any,
    *,
    text_scale: float,
    layout_scale: float,
    appearance_scale: float,
) -> tuple[str, ...]:
    """Expose the sampler's exact lazy branch plan for summaries and tests."""

    layout = bundle.layout_condition
    if not isinstance(layout, LayoutConditionBatch):
        raise TypeError("bundle.layout_condition must be LayoutConditionBatch")
    layout_present = _layout_present(layout)
    appearance_present = bool(layout.appearance.binding_valid.any().item())
    return required_cfg_branches(
        text_scale=float(text_scale),
        layout_scale=float(layout_scale) if layout_present else 1.0,
        appearance_scale=float(appearance_scale),
        appearance_present=appearance_present,
    )


def effective_layout_guidance_scale(bundle: Any, requested: float) -> float:
    layout = bundle.layout_condition
    present = _layout_present(layout)
    return float(requested) if present else 1.0


def effective_appearance_guidance_scale(bundle: Any, requested: float) -> float:
    layout = bundle.layout_condition
    present = bool(layout.appearance.binding_valid.any().item())
    return float(requested) if present else 1.0


def _layout_present(layout: LayoutConditionBatch) -> bool:
    map_present = bool((layout.map_mode != int(MapMode.NULL)).any().item())
    actor_present = bool(
        (layout.actor_geometry.layout_mode != int(LayoutMode.NULL)).any().item()
    )
    return map_present or actor_present


def strict_layout_slices(
    layout: LayoutConditionBatch,
    windows: Sequence[tuple[int, int]],
) -> tuple[LayoutConditionBatch, ...]:
    """Slice all time-varying fields under the factual per-window mode contract.

    A remains scene-global for rows whose G has support.  ``slice_frames``
    intentionally nulls bindings on a FULL row that becomes factual G=EMPTY in
    the selected window, preserving the rule that A cannot outlive G.
    """

    return tuple(layout.slice_frames(slice(start, end)) for start, end in windows)


def audit_strict_layout_windows(
    full_layout: LayoutConditionBatch,
    windows: Sequence[tuple[int, int]],
) -> list[dict[str, int]]:
    """Audit the exact layout slices consumed by sliding-window sampling.

    Layout projection is atomic for the full requested camera trajectory.  A
    later window must therefore slice that result, not re-project with a new
    first-frame ``map_pose_offset`` anchor.  Adjacent slices are compared by
    scene-local track id so a corrupted overlap fails before sampling.
    """

    built = strict_layout_slices(full_layout, windows)
    summary: list[dict[str, int]] = []
    for index, ((start, end), layout) in enumerate(zip(windows, built)):
        full_frames = range(int(start), int(end))
        local_frames = range(int(end) - int(start))
        assert_layout_overlap_consistent(
            full_layout,
            layout,
            left_frames=full_frames,
            right_frames=local_frames,
        )
        if index:
            previous_start, previous_end = windows[index - 1]
            previous = built[index - 1]
            overlap_start = max(previous_start, int(start))
            overlap_end = min(previous_end, int(end))
            if overlap_end > overlap_start:
                assert_layout_overlap_consistent(
                    previous,
                    layout,
                    left_frames=list(
                        range(
                            overlap_start - previous_start,
                            overlap_end - previous_start,
                        )
                    ),
                    right_frames=list(
                        range(overlap_start - int(start), overlap_end - int(start))
                    ),
                )
        summary.append(
            {
                "start": int(start),
                "end": int(end),
                "actor_slots": int(
                    layout.actor_geometry.slot_valid.sum().item()
                ),
            }
        )
    return summary


def _ratio_dict(
    counts: Any,
    names: Sequence[str],
) -> dict[str, float]:
    value = torch.as_tensor(counts).detach().reshape(-1).to(dtype=torch.float64)
    if int(value.numel()) != len(names):
        raise ValueError("appearance bucket count width does not match its labels")
    total = float(value.sum().item())
    if total <= 0.0:
        return {str(name): 0.0 for name in names}
    return {
        str(name): float(value[index].item()) / total
        for index, name in enumerate(names)
    }


def layout_run_provenance(
    batch: dict[str, Any],
    *,
    bundle: Any,
    args: argparse.Namespace,
    condition_mode: str,
    windows: Sequence[dict[str, int]],
) -> dict[str, Any]:
    """Build the required P9 provenance without content digests."""

    count_fields = (
        "annotated_count",
        "supported_count",
        "eligible_count",
        "ignored_unsupported_count",
        "ignored_invalid_geometry_count",
        "ignored_outside_requested_view_count",
    )
    counts = {
        name: int(_first(batch.get(f"actor_geometry_{name}"), 0))
        for name in count_fields
    }
    geometry_mode = LayoutMode(
        int(bundle.layout_condition.actor_geometry.layout_mode[0].item())
    ).name.lower()
    return {
        "hdmap_root": str(args.hdmap_root),
        "hdmap_schema_version": _single_string(
            batch.get("hdmap_schema_version")
        ),
        "raster_schema_hash": _single_string(batch.get("raster_schema_hash")),
        "attribute_source": _single_string(batch.get("attribute_source")),
        "counts": counts,
        # Non-zero means the projector had to omit HD-map polygons it could not
        # triangulate.  ``counts`` records what was written, not what was lost.
        "map_dropped_primitives": int(
            _first(batch.get("map_dropped_primitives"), 0)
        ),
        "condition_mode": str(condition_mode),
        "layout_mode": geometry_mode,
        "requested_layout_mode": str(args.layout_mode),
        "pre_cap_count": int(
            _first(batch.get("actor_geometry_pre_cap_count"), 0)
        ),
        "post_cap_count": int(
            _first(batch.get("actor_geometry_post_cap_count"), 0)
        ),
        "overflow": int(_first(batch.get("actor_geometry_overflow"), 0)),
        "layout_max_actors": int(args.layout_max_actors),
        "static_far_plane_m": float(
            _first(batch.get("static_far_plane_m"), STATIC_FAR_PLANE_M)
        ),
        "layout_guidance_scale_requested": float(args.layout_guidance_scale),
        "layout_guidance_scale_effective": effective_layout_guidance_scale(
            bundle, float(args.layout_guidance_scale)
        ),
        "appearance_guidance_scale_requested": float(
            args.asset_control_guidance_scale
        ),
        "appearance_guidance_scale_effective": (
            effective_appearance_guidance_scale(
                bundle, float(args.asset_control_guidance_scale)
            )
        ),
        "appearance_bucket_ratios": {
            "class": _ratio_dict(
                batch["appearance_class_bucket_counts"],
                ("vehicle", "pedestrian", "cyclist"),
            ),
            "distance": _ratio_dict(
                batch["appearance_distance_bucket_counts"],
                ("near", "mid", "far"),
            ),
            "area": _ratio_dict(
                batch["appearance_area_bucket_counts"],
                ("small", "large"),
            ),
            "motion": _ratio_dict(
                batch["appearance_motion_bucket_counts"],
                ("stationary", "moving"),
            ),
        },
        "windows": list(windows),
        "gauge_scope": "scene_global",
        "render_camera_source": "requested_C",
        "render_gauge_source": "predicted_gauge",
    }


def _require_current_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    use_ema: bool,
    args: argparse.Namespace,
) -> tuple[WanSceneFlow, dict[str, Any]]:
    """Load one exact layout-v2 checkpoint with no migration path."""

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint_path} must contain a versioned mapping")
    config = payload.get("scene_flow_config")
    schedule = payload.get("flow_schedule_config")
    if not isinstance(config, dict):
        raise ValueError(f"{checkpoint_path} is missing scene_flow_config")
    if not isinstance(schedule, dict):
        raise ValueError(f"{checkpoint_path} is missing flow_schedule_config")
    if config.get("layout_condition_version") != LAYOUT_CONDITION_VERSION:
        raise ValueError("inference requires the layout-v2 model contract")
    if config.get("raster_schema_hash") != RASTER_SCHEMA_HASH:
        raise ValueError("checkpoint raster schema does not match the runtime schema")
    if float(config.get("static_far_plane_m", -1.0)) != STATIC_FAR_PLANE_M:
        raise ValueError(
            f"checkpoint static_far_plane_m must be {STATIC_FAR_PLANE_M:g}"
        )
    if int(config.get("layout_max_actors", -1)) <= 0:
        raise ValueError("checkpoint layout_max_actors must be positive")
    if schedule.get("version") != FLOW_SCHEDULE_VERSION:
        raise ValueError("checkpoint flow schedule is not the current version")
    flow_schedule = resolve_inference_flow_schedule(
        payload, args, checkpoint_path
    )

    constructor_config = dict(config)
    missing_derived = sorted(
        key for key in SCENE_FLOW_DERIVED_CONFIG_KEYS if key not in constructor_config
    )
    if missing_derived:
        raise ValueError(
            f"{checkpoint_path} is missing derived SceneFlow config keys "
            f"{missing_derived}"
        )
    for key in SCENE_FLOW_DERIVED_CONFIG_KEYS:
        constructor_config.pop(key)
    model = WanSceneFlow(**constructor_config)
    validate_scene_flow_checkpoint_config(model, payload, checkpoint_path)

    ema_used = False
    ema_note: str | None = None
    if use_ema and "ema_scene_flow_state_dict" in payload:
        state = payload["ema_scene_flow_state_dict"]
        if not isinstance(state, dict):
            raise ValueError(
                f"{checkpoint_path} ema_scene_flow_state_dict must be a tensor mapping"
            )
        source = "ema_scene_flow_state_dict"
        ema_used = True
    elif bool(payload.get("is_ema_weights", False)):
        if not use_ema:
            raise ValueError("--no_ema cannot be used with an EMA-only checkpoint")
        state = payload.get("scene_flow")
        if not isinstance(state, dict):
            raise ValueError(f"{checkpoint_path} is missing EMA-only scene_flow weights")
        source = "ema_weights_only"
        ema_used = True
    elif isinstance(payload.get("scene_flow"), dict):
        state = payload["scene_flow"]
        source = "scene_flow"
        ema_note = (
            "--no_ema set; using raw scene_flow weights"
            if not use_ema
            else "checkpoint carries only raw scene_flow weights"
        )
    else:
        raise ValueError(f"{checkpoint_path} is missing scene_flow weights")
    load_scene_flow_state_dict_strict(
        model,
        state,
        path=checkpoint_path,
        source=source,
    )
    model.to(device).eval()
    model.require_gauge_stats()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(checkpoint_path),
        "weight_source": source,
        "source": source,
        "ema_used": ema_used,
        "ema_note": ema_note,
        "step": int(payload.get("step", -1)),
        "flow_schedule": dict(flow_schedule),
        "flow_schedule_config": dict(flow_schedule),
        "prediction_type": str(model.config.prediction_type),
        "checkpoint_prediction_type": str(config["prediction_type"]),
    }


def _sync_args_from_model(
    args: argparse.Namespace,
    scene_flow: WanSceneFlow,
    checkpoint_info: dict[str, Any],
) -> None:
    config = scene_flow.config
    checkpoint_layout_max_actors = int(config.layout_max_actors)
    if args.layout_max_actors is None:
        args.layout_max_actors = checkpoint_layout_max_actors
    elif int(args.layout_max_actors) != checkpoint_layout_max_actors:
        raise ValueError(
            "--layout_max_actors must exactly match the checkpoint model config"
        )
    if float(args.static_far_plane_m) != float(config.static_far_plane_m):
        raise ValueError(
            "--static_far_plane_m must exactly match the checkpoint model config"
        )
    args.patch_grid = tuple(int(value) for value in config.patch_grid)
    args.latent_dim = int(config.out_channels)
    args.prediction_type = str(config.prediction_type)
    args.sequence_length = int(args.num_frames)
    args.sky_grid = tuple(int(value) for value in config.sky_grid)
    args.sky_grid_h, args.sky_grid_w = args.sky_grid
    args.sky_atlas_hw = tuple(
        int(value) for value in getattr(config, "sky_atlas_hw", (32, 64))
    )
    args.sky_mask_refine_scale = int(config.sky_mask_refine_scale)
    args.sky_mask_refine_channels = int(config.sky_mask_refine_channels)
    args.caption_root = None if args.no_text_condition else args.val_caption_root
    saved_shift = float(checkpoint_info["flow_schedule"]["shift"])
    if args.shift is not None and abs(float(args.shift) - saved_shift) > 1.0e-12:
        raise ValueError("--shift must match the checkpoint flow schedule")
    args.shift = saved_shift


def _resolve_windows(
    sequence_length: int,
    requested_window: int,
    requested_stride: int,
) -> tuple[tuple[int, int], ...]:
    sequence_length = int(sequence_length)
    window = min(
        sequence_length,
        OFFLINE_MAX_SINGLE_WINDOW,
        int(requested_window) if int(requested_window) > 0 else OFFLINE_MAX_SINGLE_WINDOW,
    )
    if sequence_length <= window:
        return ((0, sequence_length),)
    stride = int(requested_stride)
    if stride <= 0:
        stride = default_window_stride(window)
    if stride >= window:
        raise ValueError("sliding stride must be smaller than the window")
    return tuple(window_slices(sequence_length, window, stride))


def _inject_inference_gauge_placeholder(
    batch: dict[str, Any],
    *,
    batch_size: int,
) -> None:
    """Supply an invalid target row; sampling uses the checkpoint gauge prior."""

    batch["scene_gauge"] = torch.zeros(
        int(batch_size), SCENE_GAUGE_DIM, dtype=torch.float32
    )
    batch["scene_gauge_valid"] = torch.zeros(
        int(batch_size), SCENE_GAUGE_DIM, dtype=torch.bool
    )


def _dataset_scene_index(
    dataset: WaymoOpenDataset,
    dataset_index: int,
) -> int:
    entry = dataset.trunk_major_index[int(dataset_index)]
    return int(entry[0])


def _refresh_layout_for_requested_camera(
    *,
    dataset: WaymoOpenDataset,
    dataset_index: int,
    batch: dict[str, Any],
) -> None:
    sequence_length = int(batch["camera_to_world_corrected"].shape[1])
    start_idx = int(_first(batch.get("start_idx"), 0))
    frame_indices = list(range(start_idx, start_idx + sequence_length))
    trunk_frames = int(dataset.trunk_frames)
    context_base = (start_idx // trunk_frames) * trunk_frames
    outer_frame_indices = list(range(context_base, context_base + trunk_frames))
    requested = batch["camera_to_world_corrected"][0]
    payload = dataset.build_layout_payload_for_camera(
        _dataset_scene_index(dataset, dataset_index),
        frame_indices,
        requested_camera_to_world=requested,
        outer_frame_indices=outer_frame_indices,
        deterministic_reference=True,
    )
    _replace_batch_layout_payload(batch, payload)


def _save_render_images(
    images: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    suffix: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, value in images.items():
        path = output_dir / f"{name}__{suffix}.jpg"
        save_image_grid(value.detach().float().cpu(), path)
        paths[name] = str(path)
    return paths


def _save_requested_input(
    batch: dict[str, Any],
    output_dir: Path,
    *,
    max_frames: int,
) -> str:
    images = batch["images"]
    value = images[0, : int(max_frames)].detach().float().cpu()
    if value.max().item() > 1.5:
        value = value / 255.0
    path = output_dir / "requested_input.jpg"
    save_image_grid(value, path)
    return str(path)


def offline_sample_tensor_payload(
    bundle: Any,
    sampled: Any,
    requested_pose: torch.Tensor,
) -> dict[str, Any]:
    """Build the exact portable tensor artifact written by offline inference."""

    required_bundle_tensors = (
        "camera_to_world_requested_metric",
        "camera_intrinsics_requested_canvas_metric",
        "camera_intrinsics_requested_raw_metric",
        "camera_requested_raw_image_size_hw",
        "frame_ids",
    )
    for name in required_bundle_tensors:
        if not torch.is_tensor(getattr(bundle, name, None)):
            raise RuntimeError(f"offline bundle is missing tensor {name}")
    required_sample_tensors = (
        "video",
        "sky_mask_patch",
        "sky_mask_refined",
        "gauge",
    )
    for name in required_sample_tensors:
        if not torch.is_tensor(getattr(sampled, name, None)):
            raise RuntimeError(f"offline sample is missing tensor {name}")
    if not torch.is_tensor(requested_pose):
        raise TypeError("requested_pose must be a tensor")
    canvas_hw = tuple(int(value) for value in bundle.camera_requested_canvas_image_size_hw)
    if len(canvas_hw) != 2 or any(value <= 0 for value in canvas_hw):
        raise ValueError("camera_requested_canvas_image_size_hw must be positive [H,W]")
    return {
        "layout_condition_version": LAYOUT_CONDITION_VERSION,
        "render_camera_source": "requested_C",
        "render_gauge_source": "predicted_gauge",
        "video_latent_normalized": sampled.video.detach().cpu(),
        "sky_tokens": (
            None if sampled.sky is None else sampled.sky.detach().cpu()
        ),
        "sky_mask_patch": sampled.sky_mask_patch.detach().cpu(),
        "sky_mask_refined": sampled.sky_mask_refined.detach().cpu(),
        "gauge": sampled.gauge.detach().cpu(),
        "requested_camera_to_world_metric": (
            bundle.camera_to_world_requested_metric.detach().cpu()
        ),
        "requested_camera_intrinsics_canvas_metric": (
            bundle.camera_intrinsics_requested_canvas_metric.detach().cpu()
        ),
        "requested_camera_intrinsics_raw_metric": (
            bundle.camera_intrinsics_requested_raw_metric.detach().cpu()
        ),
        "requested_camera_canvas_image_size_hw": list(canvas_hw),
        "requested_camera_raw_image_size_hw": (
            bundle.camera_requested_raw_image_size_hw.detach().cpu()
        ),
        "requested_render_pose_encoding": requested_pose.detach().cpu(),
        "frame_ids": bundle.frame_ids.detach().cpu(),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run layout-v2 SceneFlow inference with requested Waymo C"
    )
    parser.add_argument("--weights", required=DEFAULT_WEIGHTS is None, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dggt_ckpt_path", default=DEFAULT_DGGT_CKPT)
    parser.add_argument("--tokenizer_ckpt_path", default=DEFAULT_TOKENIZER_CKPT)
    parser.add_argument("--val_image_dir", "--image_dir", dest="val_image_dir", default=DEFAULT_VAL_ROOT)
    parser.add_argument("--hdmap_root", default=DEFAULT_HDMAP_ROOT)
    parser.add_argument("--val_caption_root", "--caption_root", dest="val_caption_root", default=DEFAULT_CAPTION_ROOT)
    parser.add_argument("--text_encoder_path", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--no_text_condition", action="store_true")
    parser.add_argument("--output_dir", default="runs/scene_flow_pretrain_inference_v6")

    parser.add_argument("--val_scene_start", type=int, default=0)
    parser.add_argument("--val_scene_end", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--num_frames", "--frames", dest="num_frames", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument(
        "--cfg",
        "--cfg_scales",
        dest="cfg_values",
        nargs="+",
        default=["1.0"],
    )
    parser.add_argument(
        "--condition_mode",
        choices=CONDITION_MODES,
        default="tcmga",
        help="Explicit training-task condition set; it never cycles by dataset index.",
    )
    parser.add_argument("--layout_mode", choices=LAYOUT_MODES, default="full")
    parser.add_argument("--layout_guidance_scale", type=float, default=1.0)
    parser.add_argument("--asset_control_guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--layout_max_actors",
        type=int,
        default=None,
        help=(
            "Optional strict assertion. When omitted, use the positive value "
            "stored in the checkpoint model config."
        ),
    )
    parser.add_argument(
        "--static_far_plane_m",
        type=float,
        default=STATIC_FAR_PLANE_M,
        help="Frozen camera optical-depth far plane for static HD-map geometry.",
    )
    parser.add_argument(
        "--sample_steps",
        "--val_sample_steps",
        dest="val_sample_steps",
        type=int,
        default=T59_VALIDATION_SAMPLE_STEPS,
    )
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--no_sky_generation", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--val_log_images", type=int, default=10)
    parser.add_argument("--sky_unobserved_loss_weight", type=float, default=0.0)
    parser.add_argument("--val_sliding_window", type=int, default=10)
    parser.add_argument("--val_sliding_stride", type=int, default=7)
    return parser


def validate_args(args: argparse.Namespace) -> list[float]:
    if int(args.num_frames) < 2 or int(args.num_frames) > 29:
        raise ValueError("--num_frames must be in [2,29]")
    if int(args.val_sample_steps) != T59_VALIDATION_SAMPLE_STEPS:
        raise ValueError(
            "--sample_steps is frozen to "
            f"{T59_VALIDATION_SAMPLE_STEPS} by the accepted T59 decision"
        )
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be non-negative")
    if int(args.val_scene_end) <= int(args.val_scene_start):
        raise ValueError("--val_scene_end must be greater than --val_scene_start")
    if args.layout_max_actors is not None and int(args.layout_max_actors) <= 0:
        raise ValueError("--layout_max_actors must be positive when specified")
    if float(args.static_far_plane_m) != STATIC_FAR_PLANE_M:
        raise ValueError(
            f"--static_far_plane_m is frozen to {STATIC_FAR_PLANE_M:g}"
        )
    for name in ("layout_guidance_scale", "asset_control_guidance_scale"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} must be finite and non-negative")
    if args.no_text_condition:
        scales = parse_cfg_scales(args.cfg_values)
        if any(scale != 1.0 for scale in scales):
            raise ValueError("text CFG must remain 1 when text conditioning is disabled")
        return scales
    return parse_cfg_scales(args.cfg_values)


@torch.no_grad()
def run_inference(args: argparse.Namespace, cfg_scales: Sequence[float]) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(args.seed))

    scene_flow, checkpoint_info = _require_current_checkpoint(
        args.weights,
        device=device,
        use_ema=not bool(args.no_ema),
        args=args,
    )
    _sync_args_from_model(args, scene_flow, checkpoint_info)
    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path,
        args.tokenizer_ckpt_path,
        device,
    )
    text_encoder = setup_text_encoder(args, device)

    scene_names = discover_scene_names(
        args.val_image_dir,
        args.val_scene_start,
        args.val_scene_end,
    )
    dataset = WaymoOpenDataset(
        image_dir=args.val_image_dir,
        scene_names=scene_names,
        sequence_length=int(args.num_frames),
        start_idx=0,
        mode=1,
        views=1,
        caption_root=None if args.no_text_condition else args.val_caption_root,
        pretrain_patch_grid=args.patch_grid,
        hdmap_root=args.hdmap_root,
        layout_max_actors=int(args.layout_max_actors),
        static_far_plane_m=float(args.static_far_plane_m),
        trunk_major_samples=True,
        trunk_frames=29,
        return_full_dggt_context=True,
        load_dynamic_masks=False,
        binary_mask_channels=1,
        image_output_dtype="uint8",
        scene_gauge_path=None,
    )
    if not dataset:
        raise RuntimeError("the selected validation dataset is empty")

    if args.index is not None:
        start = int(args.index)
        end = start + 1
    else:
        start = int(args.start)
        end = len(dataset) if args.end is None else int(args.end)
    start = max(0, min(start, len(dataset)))
    end = max(start, min(end, len(dataset)))
    if start == end:
        raise RuntimeError(f"no rows selected from [{start},{end})")

    selected_indices = list(range(start, end))
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        drop_last=False,
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    windows = _resolve_windows(
        int(args.num_frames),
        int(args.val_sliding_window),
        int(args.val_sliding_stride),
    )
    args.val_sliding_window = (
        int(windows[0][1] - windows[0][0]) if len(windows) > 1 else 0
    )
    args.val_sliding_stride = (
        int(windows[1][0] - windows[0][0])
        if len(windows) > 1
        else int(args.val_sliding_stride)
    )

    samples: list[dict[str, Any]] = []
    iterator = tqdm(
        zip(selected_indices, loader),
        total=len(selected_indices),
        desc="layout-v2 inference",
        dynamic_ncols=True,
    )
    for dataset_index, batch in iterator:
        mode = str(args.condition_mode)
        scene_name = _single_string(batch.get("scene_name"), "unknown")
        start_idx = int(_first(batch.get("start_idx"), 0))
        clip_index = int(_first(batch.get("clip_index"), start_idx // 29))
        sample_dir = output_root / (
            f"{dataset_index:06d}_scene{scene_name}_clip{clip_index:02d}_"
            f"start{start_idx:03d}_{mode}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        _refresh_layout_for_requested_camera(
            dataset=dataset,
            dataset_index=dataset_index,
            batch=batch,
        )
        _inject_inference_gauge_placeholder(batch, batch_size=1)
        bundle = build_pretrain_bundle_from_batch(
            batch,
            vggt_model,
            scene_flow,
            device,
            args,
            include_metric_depth_diagnostic=False,
        )
        window_summary = audit_strict_layout_windows(
            bundle.layout_condition,
            windows,
        )
        condition_rows = apply_condition_mode(
            bundle,
            mode,
            layout_mode=args.layout_mode,
        )
        input_image = _save_requested_input(
            batch,
            sample_dir,
            max_frames=int(args.val_log_images),
        )
        provenance = layout_run_provenance(
            batch,
            bundle=bundle,
            args=args,
            condition_mode=mode,
            windows=window_summary,
        )

        cfg_results: list[dict[str, Any]] = []
        for scale in cfg_scales:
            sample_args = argparse.Namespace(**vars(args))
            sample_args.guidance_scale = float(scale)
            sample_args.layout_guidance_scale = effective_layout_guidance_scale(
                bundle,
                float(args.layout_guidance_scale),
            )
            sample_args.asset_control_guidance_scale = (
                effective_appearance_guidance_scale(
                    bundle,
                    float(args.asset_control_guidance_scale),
                )
            )
            branches = inference_cfg_branches(
                bundle,
                text_scale=float(scale),
                layout_scale=float(sample_args.layout_guidance_scale),
                appearance_scale=float(sample_args.asset_control_guidance_scale),
            )
            sampled = cfg_sample_pretrain_latents(
                scene_flow,
                bundle,
                sample_args,
                step=int(dataset_index),
                device=device,
                guidance_scale=float(scale),
                text_encoder=text_encoder,
                return_sky=sky_generation_enabled(sample_args),
                return_gauge=True,
                return_sky_mask=True,
            )
            if sampled.gauge is None:
                raise RuntimeError("SceneFlow sampling did not return its scene gauge")
            requested_pose = requested_render_pose_encoding(
                bundle.camera_to_world_requested_metric,
                bundle.camera_trajectory_anchor_to_world_metric,
                sampled.gauge,
            )
            render_paths: dict[str, str] = {}
            if not args.skip_render:
                rendered = render_validation_generated_rgb(
                    batch,
                    vggt_model,
                    scene_flow,
                    sampled.video,
                    sample_args,
                    device,
                    bundle.camera_to_world_requested_metric,
                    bundle.camera_trajectory_anchor_to_world_metric,
                    sampled.gauge,
                    sampled.sky,
                    sampled.sky_mask_patch,
                    sampled.sky_mask_refined,
                )
                render_paths = _save_render_images(
                    rendered,
                    sample_dir,
                    suffix=f"cfg{cfg_tag(scale)}",
                )
            tensor_path = sample_dir / f"sample__cfg{cfg_tag(scale)}.pt"
            torch.save(
                offline_sample_tensor_payload(bundle, sampled, requested_pose),
                tensor_path,
            )
            cfg_results.append(
                {
                    "text_guidance_scale_requested": float(scale),
                    "text_guidance_scale_effective": float(scale),
                    "layout_guidance_scale_requested": float(
                        args.layout_guidance_scale
                    ),
                    "layout_guidance_scale_effective": float(
                        sample_args.layout_guidance_scale
                    ),
                    "appearance_guidance_scale_requested": float(
                        args.asset_control_guidance_scale
                    ),
                    "appearance_guidance_scale_effective": float(
                        sample_args.asset_control_guidance_scale
                    ),
                    "branches": list(branches),
                    "tensor_path": str(tensor_path),
                    "render_images": render_paths,
                }
            )
            del sampled, requested_pose
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = {
            "dataset_index": int(dataset_index),
            "scene_name": scene_name,
            "clip_index": clip_index,
            "start_idx": start_idx,
            "frame_ids": bundle.frame_ids.detach().cpu().tolist(),
            "caption": _single_string(batch.get("caption")),
            "requested_input": input_image,
            "condition_rows": condition_rows,
            "layout": provenance,
            "sample_steps": int(args.val_sample_steps),
            "flow_shift": float(args.shift),
            "sliding_window_active": len(windows) > 1,
            "window": int(windows[0][1] - windows[0][0]),
            "window_stride": int(args.val_sliding_stride),
            "checkpoint": checkpoint_info,
            "cfg_results": cfg_results,
        }
        summary_path = sample_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        samples.append(summary)
        iterator.set_postfix(scene=scene_name, mode=mode)
        del batch, bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layout_effective_values = sorted(
        {
            float(result["layout_guidance_scale_effective"])
            for sample in samples
            for result in sample["cfg_results"]
        }
    )
    appearance_effective_values = sorted(
        {
            float(result["appearance_guidance_scale_effective"])
            for sample in samples
            for result in sample["cfg_results"]
        }
    )
    run_summary = {
        "contract": LAYOUT_CONDITION_VERSION,
        "condition_mode": str(args.condition_mode),
        "selected_range": [start, end],
        "hdmap_root": str(args.hdmap_root),
        "layout_mode": str(args.layout_mode),
        "layout_max_actors": int(args.layout_max_actors),
        "static_far_plane_m": float(args.static_far_plane_m),
        "text_guidance_scales_requested": [float(value) for value in cfg_scales],
        "text_guidance_scales_effective": [float(value) for value in cfg_scales],
        "layout_guidance_scale_requested": float(args.layout_guidance_scale),
        "layout_guidance_scale_effective": (
            layout_effective_values[0]
            if len(layout_effective_values) == 1
            else None
        ),
        "layout_guidance_scale_effective_values": layout_effective_values,
        "appearance_guidance_scale_requested": float(
            args.asset_control_guidance_scale
        ),
        "appearance_guidance_scale_effective": (
            appearance_effective_values[0]
            if len(appearance_effective_values) == 1
            else None
        ),
        "appearance_guidance_scale_effective_values": appearance_effective_values,
        "window": int(windows[0][1] - windows[0][0]),
        "window_stride": int(args.val_sliding_stride),
        "sliding_window_active": len(windows) > 1,
        "checkpoint": checkpoint_info,
        "samples": samples,
    }
    (output_root / "all_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=str)
    )
    return run_summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg_scales = validate_args(args)
    summary = run_inference(args, cfg_scales)
    print(
        f"[done] wrote {len(summary['samples'])} sample folders under {args.output_dir}",
        flush=True,
    )


# Pure DGGT geometry helpers remain public for non-P9 export tests.  They do
# not participate in camera control or checkpoint validation.
def build_generated_dggt_scene_state(
    *,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
    sky_mask: torch.Tensor,
) -> CleanSceneState:
    if int(pose_enc.shape[0]) != 1:
        raise ValueError("point export currently expects batch size one")
    if (
        sky_mask.ndim != 5
        or int(sky_mask.shape[0]) != 1
        or int(sky_mask.shape[2]) not in (1, 3)
    ):
        raise ValueError("sky_mask must be [1,S,1/3,H,W]")
    seq_len = int(sky_mask.shape[1])
    height, width = int(sky_mask.shape[-2]), int(sky_mask.shape[-1])
    images_clean = torch.zeros((seq_len, 3, height, width), dtype=torch.float32)
    sky_probability = (
        sky_mask[0].detach().cpu().float().mean(dim=1).clamp(0.0, 1.0)
    )
    non_sky_probability = 1.0 - sky_probability
    sky_mask_cpu = torch.zeros_like(non_sky_probability).unsqueeze(1)
    state = build_clean_scene_state(
        {
            "images_clean": images_clean,
            "sky_mask": sky_mask_cpu,
            "masks": sky_mask_cpu,
            "cam_ids": torch.tensor([0], dtype=torch.long),
        },
        {
            "pose_enc": pose_enc,
            "depth": depth,
            "gs_map": gs_map,
            "gs_conf": gs_conf,
            "dynamic_conf": dynamic_conf,
        },
    )
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
    if export_units not in {"dggt", "metric"}:
        raise ValueError("export_units must be dggt or metric")
    if (
        not torch.is_tensor(gauge)
        or gauge.ndim != 3
        or tuple(gauge.shape[-2:]) != (1, SCENE_GAUGE_DIM)
    ):
        raise ValueError(f"gauge must be [B,1,{SCENE_GAUGE_DIM}]")
    if float(calibration.c_gs) != 1.0:
        raise ValueError("metric inference requires c_gs=1")
    render_geometry = apply_pullback_calibration(
        depth,
        gs_map,
        log_metric_scale=gauge[..., 0],
        calibration=calibration,
        boundary=PULLBACK_RENDER_BOUNDARY,
    )
    if (
        render_geometry.depth_dggt is not depth
        or render_geometry.gs_map_dggt is not gs_map
    ):
        raise AssertionError("native render geometry must remain unchanged")
    if export_units == "dggt":
        return render_geometry, render_geometry
    metric_geometry = apply_pullback_calibration(
        depth,
        gs_map,
        log_metric_scale=gauge[..., 0],
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
    if tuple(value.shape) != (1, 1, SCENE_GAUGE_DIM):
        raise ValueError(f"gauge must be [1,1,{SCENE_GAUGE_DIM}]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("gauge must be finite")
    fov_yx = gauge_to_pose_enc_fov(value, 1)[0, 0]
    metres_per_unit = math.exp(float(value[0, 0, 0].item()))
    return {
        "representation": SCENE_GAUGE_REPRESENTATION,
        "values": [float(channel) for channel in value[0, 0].cpu().tolist()],
        "log_metric_scale": float(value[0, 0, 0].item()),
        "metres_per_unit": metres_per_unit,
        "fov_deg": {
            "x": math.degrees(float(fov_yx[1].item())),
            "y": math.degrees(float(fov_yx[0].item())),
        },
    }


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
    if export_units not in {"dggt", "metric"}:
        raise ValueError("export_units must be dggt or metric")
    gauge_summary = None if gauge is None else _single_sample_gauge_summary(gauge)
    if export_units == "metric" and (
        gauge_summary is None or calibration is None
    ):
        raise ValueError("metric point export requires gauge and calibration")
    metres_per_unit = (
        1.0
        if export_units == "dggt"
        else float(gauge_summary["metres_per_unit"])
    )
    stride = max(1, int(stride))
    output_dir.mkdir(parents=True, exist_ok=True)
    sky_probabilities = getattr(scene_state, "sky_probabilities", None)
    if not torch.is_tensor(sky_probabilities):
        raise RuntimeError("scene state is missing sky probabilities")

    frames: list[dict[str, Any]] = []
    for frame_idx in range(int(scene_state.images.shape[0])):
        keep = scene_state.source_image_ids == frame_idx
        if stride > 1:
            keep &= torch.remainder(scene_state.source_y, stride) == 0
            keep &= torch.remainder(scene_state.source_x, stride) == 0
        keep &= sky_probabilities.reshape(-1) < float(
            sky_probability_threshold
        )
        keep &= scene_state.opacities.reshape(-1) >= float(
            min_effective_opacity
        )
        points = scene_state.means[keep] * metres_per_unit
        colors = scene_state.colors[keep]
        opacities = scene_state.opacities[keep]
        scales = scene_state.scales[keep] * metres_per_unit
        quats = scene_state.quats[keep]
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
        frames.append(
            {
                "frame": frame_idx,
                "gaussian_ply": str(gaussian_path),
                "point_ply": str(point_path),
                "num_points": int(points.shape[0]),
            }
        )

    factor_summary = None
    if torch.is_tensor(c_depth_factor):
        factors = c_depth_factor.detach().float().reshape(-1)
        factor_summary = {
            "factor_min": float(factors.min().item()),
            "factor_median": float(factors.median().item()),
            "factor_max": float(factors.max().item()),
        }
    return {
        "frames": frames,
        "num_points": sum(int(frame["num_points"]) for frame in frames),
        "export_units": export_units,
        "metres_per_unit_applied": metres_per_unit,
        "gauge": gauge_summary,
        "c_depth": factor_summary,
        "c_gs": 1.0 if calibration is None else float(calibration.c_gs),
    }


if __name__ == "__main__":
    main()
