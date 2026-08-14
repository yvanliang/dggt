#!/usr/bin/env python3
"""Offline diagnostics for the online HD-map/layout projection contract.

The tool consumes an already-built HD-map sidecar plus an explicit requested
camera snapshot.  It never converts or mutates HD-map data.  Real-data runs
record projection coverage, thin-line survival, actor counts, camera
equivariance, and the observable effect of the window's map-pose offset.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.tools.hdmap_schema import read_scene_npz
from dggt.utils.actor_geometry_condition import (
    ActorGeometryCondition,
    CameraSpec,
    LayoutMode,
    project_world_points,
)
from dggt.utils.layout_raster import (
    dequantize_layout_raster,
    project_layout,
    thin_line_projection_theory,
)


def _as_batched(value: Any, trailing_rank: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim == trailing_rank:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != trailing_rank + 1:
        raise ValueError(f"{name} has an unsupported shape {tuple(tensor.shape)}")
    return tensor


def load_camera_npz(path: str | Path) -> CameraSpec:
    """Load the explicit C-sideflow required by the production projector."""

    with np.load(Path(path), allow_pickle=False) as payload:
        world_to_anchor = _as_batched(payload["world_to_anchor"], 2, "world_to_anchor")
        anchor_to_camera = _as_batched(
            payload["anchor_to_camera"], 3, "anchor_to_camera"
        )
        intrinsics = _as_batched(payload["intrinsics"], 3, "intrinsics")
        raw_to_canvas = _as_batched(payload["raw_to_canvas"], 3, "raw_to_canvas")
        offsets = torch.as_tensor(payload["map_pose_offset"], dtype=torch.float64)
        if offsets.ndim == 2:
            offsets = offsets.unsqueeze(0)
        canvas_hw = tuple(int(v) for v in payload["canvas_hw"].reshape(-1).tolist())
        patch_grid = (
            tuple(int(v) for v in payload["patch_grid"].reshape(-1).tolist())
            if "patch_grid" in payload
            else (25, 37)
        )
    if len(canvas_hw) != 2 or len(patch_grid) != 2:
        raise ValueError("canvas_hw and patch_grid must each contain two integers")
    return CameraSpec.from_window(
        world_to_anchor=world_to_anchor,
        anchor_to_camera=anchor_to_camera,
        intrinsics=intrinsics,
        raw_to_canvas=raw_to_canvas,
        map_pose_offsets=offsets,
        canvas_hw=canvas_hw,
        patch_grid=patch_grid,
    )


def empty_actor_geometry(cam: CameraSpec, layout_max_actors: int) -> ActorGeometryCondition:
    b, s, kg = cam.batch_size, cam.num_frames, int(layout_max_actors)
    if kg <= 0:
        raise ValueError("layout_max_actors must be positive")
    return ActorGeometryCondition(
        slot_track_id=torch.full((b, kg), -1, dtype=torch.int64),
        class_id=torch.full((b, kg), -1, dtype=torch.int8),
        corners_world=torch.zeros((b, kg, s, 8, 3), dtype=torch.float64),
        velocity_world=torch.zeros((b, kg, s, 3), dtype=torch.float32),
        box_size=torch.zeros((b, kg, s, 3), dtype=torch.float32),
        yaw=torch.zeros((b, kg, s), dtype=torch.float32),
        is_moving=torch.zeros((b, kg, s), dtype=torch.bool),
        track_valid=torch.zeros((b, kg, s), dtype=torch.bool),
        slot_valid=torch.zeros((b, kg), dtype=torch.bool),
        layout_mode=torch.full((b,), int(LayoutMode.EMPTY), dtype=torch.int8),
        raw_track_key=[[""] * kg for _ in range(b)],
    )


def load_actor_geometry(
    path: str | Path | None,
    cam: CameraSpec,
    layout_max_actors: int,
) -> ActorGeometryCondition:
    if path is None:
        return empty_actor_geometry(cam, layout_max_actors)
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, ActorGeometryCondition):
        geometry = payload
    elif isinstance(payload, dict):
        names = {field.name for field in fields(ActorGeometryCondition)}
        missing = names.difference(payload)
        if missing:
            raise KeyError(f"actor geometry payload is missing {sorted(missing)}")
        geometry = ActorGeometryCondition(**{name: payload[name] for name in names})
    else:
        raise TypeError("actor geometry payload must be a dataclass or field dictionary")
    return geometry.validate(layout_max_actors=layout_max_actors)


def _coverage_centroid(mask: torch.Tensor) -> tuple[float, float] | None:
    weights = mask.float().sum(dim=(0, 1))
    total = float(weights.sum().item())
    if total <= 0.0:
        return None
    yy, xx = torch.meshgrid(
        torch.arange(weights.shape[0], dtype=torch.float32),
        torch.arange(weights.shape[1], dtype=torch.float32),
        indexing="ij",
    )
    return (
        float((xx * weights).sum().item() / total),
        float((yy * weights).sum().item() / total),
    )


def shifted_camera_x(cam: CameraSpec, metres: float) -> CameraSpec:
    """Translate requested C in anchor +x without editing map or actors."""

    transform = torch.eye(4, dtype=torch.float64).expand(
        cam.batch_size, cam.num_frames, 4, 4
    ).clone()
    transform[..., 0, 3] = -float(metres)
    return CameraSpec(
        world_to_anchor=cam.world_to_anchor,
        anchor_to_camera=cam.anchor_to_camera @ transform,
        intrinsics=cam.intrinsics,
        raw_to_canvas=cam.raw_to_canvas,
        map_pose_offset=cam.map_pose_offset,
        canvas_hw=cam.canvas_hw,
        patch_grid=cam.patch_grid,
        near_plane_m=cam.near_plane_m,
    )


def _reference_projection_error(path: str | Path, cam: CameraSpec) -> dict[str, float]:
    with np.load(Path(path), allow_pickle=False) as payload:
        points = torch.as_tensor(payload["points_world"], dtype=torch.float64)
        expected_uv = torch.as_tensor(payload["uv_canvas_normalized"], dtype=torch.float64)
        frame_index = int(np.asarray(payload.get("frame_index", 0)).reshape(-1)[0])
    if points.ndim == 2:
        points = points.unsqueeze(0)
    _points_camera, uv_canvas = project_world_points(points, cam)
    actual = uv_canvas[:, frame_index]
    if expected_uv.ndim == 2:
        expected_uv = expected_uv.unsqueeze(0)
    if actual.shape != expected_uv.shape:
        raise ValueError(
            f"reference UV shape {tuple(expected_uv.shape)} != projected {tuple(actual.shape)}"
        )
    error = (actual.double() - expected_uv).abs()
    return {
        "max_abs_pixel_normalized": float(error.max().item()),
        "mean_abs_pixel_normalized": float(error.mean().item()),
    }


def run_diagnostics(
    *,
    scene: Any,
    geometry: ActorGeometryCondition,
    camera: CameraSpec,
    layout_max_actors: int,
    reference_projection: str | Path | None = None,
) -> dict[str, Any]:
    base = project_layout(
        scene,
        geometry,
        camera,
        layout_max_actors=layout_max_actors,
    )
    raster = dequantize_layout_raster(
        base.map_layout.layout_raster,
        raster_schema_hash=base.map_layout.raster_schema_hash,
    )
    static_valid = raster[:, :, 21] > 0.0
    actor_valid = raster[:, :, 32] > 0.0
    thin = raster[:, :, (0, 1, 2)]
    thin_patch = torch.nn.functional.avg_pool2d(
        thin.flatten(0, 1), kernel_size=4, stride=4
    )
    thin_nonzero_patch_count = int((thin_patch > 0.0).sum().item())
    thin_theory = thin_line_projection_theory(scene, camera)
    thin_expected_patch_count = int(thin_theory.expected_patch_count)
    thin_survival_ratio = (
        1.0
        if thin_expected_patch_count == 0
        else float(thin_nonzero_patch_count) / float(thin_expected_patch_count)
    )
    if thin_expected_patch_count > 0 and thin_survival_ratio < 0.8:
        raise AssertionError(
            "T23 thin-line survival ratio "
            f"{thin_survival_ratio:.6f} is below the frozen 0.8 threshold"
        )

    shifted = shifted_camera_x(camera, 2.0)
    shifted_projection = project_layout(
        scene,
        geometry,
        shifted,
        layout_max_actors=layout_max_actors,
    )
    shifted_raster = dequantize_layout_raster(
        shifted_projection.map_layout.layout_raster,
        raster_schema_hash=shifted_projection.map_layout.raster_schema_hash,
    )
    base_centroid = _coverage_centroid(static_valid)
    shifted_centroid = _coverage_centroid(shifted_raster[:, :, 21] > 0.0)

    zero_offset = CameraSpec(
        world_to_anchor=camera.world_to_anchor,
        anchor_to_camera=camera.anchor_to_camera,
        intrinsics=camera.intrinsics,
        raw_to_canvas=camera.raw_to_canvas,
        map_pose_offset=torch.zeros_like(camera.map_pose_offset),
        canvas_hw=camera.canvas_hw,
        patch_grid=camera.patch_grid,
        near_plane_m=camera.near_plane_m,
    )
    zero_projection = project_layout(
        scene,
        geometry,
        zero_offset,
        layout_max_actors=layout_max_actors,
    )
    offset_changed = torch.count_nonzero(
        base.map_layout.layout_raster != zero_projection.map_layout.layout_raster
    ).item()

    summary: dict[str, Any] = {
        "scene_id": str(scene.scene_id),
        "frames": int(camera.num_frames),
        "primitive_count": int(len(scene.features)),
        "layout_max_actors": int(layout_max_actors),
        "actor_slot_count": int(geometry.slot_valid.sum().item()),
        "actor_visible_track_frames": int(base.actor_geometry.in_frustum.sum().item()),
        "static_valid_raster_pixels": int(static_valid.sum().item()),
        "actor_valid_raster_pixels": int(actor_valid.sum().item()),
        "thin_line_nonzero_patch_count": thin_nonzero_patch_count,
        "thin_line_theoretical_patch_count": thin_expected_patch_count,
        "thin_line_survival_ratio": thin_survival_ratio,
        "thin_line_projected_length_patch_units": float(
            thin_theory.projected_length_patch_units
        ),
        "thin_line_visible_segment_count": int(
            thin_theory.visible_segment_count
        ),
        "thin_line_survival_threshold": 0.8,
        "camera_shift_x_m": 2.0,
        "static_centroid_before_xy": base_centroid,
        "static_centroid_after_xy": shifted_centroid,
        "static_centroid_delta_xy": (
            None
            if base_centroid is None or shifted_centroid is None
            else [
                shifted_centroid[0] - base_centroid[0],
                shifted_centroid[1] - base_centroid[1],
            ]
        ),
        "map_pose_offset_first_frame_m": camera.map_pose_offset[0].tolist(),
        "offset_zeroed_changed_raster_values": int(offset_changed),
    }
    if reference_projection is not None:
        summary["production_projection_reference"] = _reference_projection_error(
            reference_projection, camera
        )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdmap_root", type=Path, required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--camera_npz", type=Path, required=True)
    parser.add_argument("--actor_geometry", type=Path)
    parser.add_argument("--reference_projection", type=Path)
    parser.add_argument("--layout_max_actors", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    camera = load_camera_npz(args.camera_npz)
    scene = read_scene_npz(args.hdmap_root, str(args.scene_id))
    geometry = load_actor_geometry(
        args.actor_geometry,
        camera,
        int(args.layout_max_actors),
    )
    summary = run_diagnostics(
        scene=scene,
        geometry=geometry,
        camera=camera,
        layout_max_actors=int(args.layout_max_actors),
        reference_projection=args.reference_projection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
