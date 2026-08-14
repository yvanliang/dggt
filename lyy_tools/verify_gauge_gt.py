"""Verify the DGGT metric gauge with three metric references.

This Phase-0 diagnostic deliberately reports two protocols:

``full_29f`` (the protocol used for the redesign)
    The frozen DGGT aggregator is run once on a complete 29-frame trunk.  A
    depth scale is first estimated independently in every frame, frame
    outliers are rejected with MAD, and the trunk target is the median of the
    retained frame medians.  Camera Umeyama and actor-box estimates use the
    same complete trunk.

``legacy_10f`` (a reproduction-only auxiliary)
    The first dataset window is gathered from the 29-frame prediction and the
    old pooled-pixel depth median / 10-camera-centre Umeyama are recomputed.
    The ``ego_motion_m > 2`` split in this block is the definition that should
    reproduce the previously measured 24 stationary windows out of 90.

All three reported scales have the same direction and units::

    s = DGGT units / metric metre

The actor ruler does *not* read lidar.  For every class-matched semantic actor
pixel, it intersects the Waymo camera ray with the annotated oriented cuboid
(``obj_to_world`` + metric ``lwh``).  Each pixel provides a log-scale interval
from ``DGGT_depth / metric_exit_depth`` to
``DGGT_depth / metric_entry_depth``.  Each actor/frame estimate uses a
maximum-consensus interval (default support >= 0.6); a point is exposed only
when that interval is sufficiently narrow.  The entry-depth ratio is retained
only as a clearly named surface proxy.  Strict consensus is built per actor
and with equal actor weight per frame.  Across frames, valid frame points use
MAD rejection and a median; the strict trunk interval intersection is retained
only as a diagnostic because normal frame depth noise shifts narrow intervals.
Thus neither large boxes nor noisy frames dominate.  Because a cuboid is only
a tight proxy for the true mesh, this actor estimate is diagnostic and never
offline gauge GT.

Example (the full 90-trunk run is intentionally explicit)::

    conda activate dggt
    GAUGE_DEVICE=cuda:0 python lyy_tools/verify_gauge_gt.py \
      --scenes 300-329 --trunks 0,1,2 \
      --output-json lyy_tools/verify_gauge_gt_results.json

For a cheap wiring check, add ``--max-trunks 1``.  Pure geometry and robust
aggregation tests live in ``tests/test_verify_gauge_gt.py`` and do not require
the dataset, checkpoint, or CUDA.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset import (  # noqa: E402
    WaymoOpenDataset,
    _load_waymo_semantic_labels_model,
    _waymo_semantic_values_for_class,
    load_and_preprocess_flow,
)
from dggt.models.vggt import VGGT  # noqa: E402
from dggt.utils.actor_geometry_condition import (  # noqa: E402
    raw_to_model_canvas_homography,
)
from dggt.utils.scene_gauge import dggt_pose_encoding_to_camera_to_world  # noqa: E402


DEFAULT_IMAGE_DIR = "/data/lyy_dataset/waymo_processed_dggt/training"
DEFAULT_CHECKPOINT = "/data/lyy_dataset/model/dggt/model_latest_waymo.pt"
TRUNK_FRAMES = 29
LEGACY_WINDOW_FRAMES = 10
SCALE_DEFINITION = "DGGT units per metric metre"
RESULT_SCHEMA_VERSION = "verify_gauge_gt_v1"


def _is_finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


@dataclass(frozen=True)
class ScaleAggregate:
    """MAD-filtered scalar aggregation result."""

    value: float | None
    n_total: int
    n_inlier: int
    median_before_filter: float | None
    mad_before_filter: float | None
    std_inlier: float | None
    cv_inlier: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mad_inlier_mask(
    array: np.ndarray,
    *,
    mad_z: float = 3.5,
    reject_outliers: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Return the exact raw-space MAD mask used by scalar aggregation."""

    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros((0,), dtype=np.bool_), float("nan"), float("nan")
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    keep = np.ones(values.shape, dtype=np.bool_)
    if reject_outliers and values.size >= 4:
        if mad <= np.finfo(np.float64).eps:
            tolerance = max(abs(median) * 1.0e-6, 1.0e-12)
        else:
            tolerance = float(mad_z) * 1.4826 * mad
        keep = deviations <= tolerance
        if not bool(keep.any()):
            keep = np.ones(values.shape, dtype=np.bool_)
    return keep, median, mad


def aggregate_scale_values(
    values: Iterable[float | int | None],
    *,
    mad_z: float = 3.5,
    reject_outliers: bool = True,
) -> ScaleAggregate:
    """Return a raw-space median after robust MAD rejection.

    The redesign plan explicitly calls for frame median -> MAD -> trunk
    median, so filtering is intentionally performed in raw scale space rather
    than silently changing the target to a log-space estimator.
    """

    array = np.asarray(
        [float(value) for value in values if _is_finite_positive(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return ScaleAggregate(None, 0, 0, None, None, None, None)

    keep, median, mad = _mad_inlier_mask(
        array,
        mad_z=mad_z,
        reject_outliers=reject_outliers,
    )

    retained = array[keep]
    value = float(np.median(retained))
    std = float(np.std(retained))
    cv = std / abs(float(np.mean(retained))) if retained.size else float("nan")
    return ScaleAggregate(
        value=value,
        n_total=int(array.size),
        n_inlier=int(retained.size),
        median_before_filter=median,
        mad_before_filter=mad,
        std_inlier=std,
        cv_inlier=float(cv),
    )


def maximum_consensus_log_interval(
    lower_log_scale: Iterable[float],
    upper_log_scale: Iterable[float],
    *,
    min_support_fraction: float = 0.6,
    max_log_width: float = 0.25,
) -> dict[str, Any]:
    """Find a maximum-overlap interval from per-observation log-scale bounds.

    Every actor pixel supplies an interval, not an exact surface location.  A
    maximum-consensus subset is found without consulting lidar or s_depth.  A
    point estimate is exposed only when that subset has enough support and its
    common intersection is sufficiently narrow.
    """

    lower = np.asarray(list(lower_log_scale), dtype=np.float64)
    upper = np.asarray(list(upper_log_scale), dtype=np.float64)
    if lower.shape != upper.shape:
        raise ValueError("lower/upper log-scale arrays must match")
    finite = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    lower, upper = lower[finite], upper[finite]
    count = int(lower.size)
    if count == 0:
        return {
            "valid": False,
            "point_log_scale": None,
            "low_log_scale": None,
            "high_log_scale": None,
            "log_width": None,
            "support": 0,
            "total": 0,
            "support_fraction": 0.0,
        }

    # A maximum overlap always begins at one of the lower endpoints.  Count
    # interval membership at all such endpoints, choose the median maximizer,
    # then intersect the complete consensus subset containing that point.
    candidates = np.unique(lower)
    sorted_lower = np.sort(lower)
    sorted_upper = np.sort(upper)
    support_counts = (
        np.searchsorted(sorted_lower, candidates, side="right")
        - np.searchsorted(sorted_upper, candidates, side="left")
    )
    max_support = int(support_counts.max())
    maximizing = candidates[support_counts == max_support]
    # Pick an actual maximizing endpoint.  The arithmetic median of two
    # disconnected maxima can fall in an uncovered gap.
    seed = float(maximizing[(len(maximizing) - 1) // 2])
    consensus = (lower <= seed) & (upper >= seed)
    low = float(lower[consensus].max())
    high = float(upper[consensus].min())
    width = max(0.0, high - low)
    support_fraction = max_support / float(count)
    valid = (
        support_fraction >= float(min_support_fraction)
        and width <= float(max_log_width)
    )
    return {
        "valid": bool(valid),
        "point_log_scale": 0.5 * (low + high) if valid else None,
        "low_log_scale": low,
        "high_log_scale": high,
        "log_width": width,
        "support": max_support,
        "total": count,
        "support_fraction": support_fraction,
    }


def estimate_depth_ruler(
    dggt_depth: np.ndarray,
    lidar_depth_m: np.ndarray,
    *,
    min_pixels_per_frame: int = 64,
    min_pixels_per_trunk: int = 5000,
    min_valid_frames: int = 15,
    min_depth_m: float = 1.0,
    max_depth_m: float = 80.0,
) -> dict[str, Any]:
    """Estimate ``s_depth`` using equal-weighted frame medians."""

    predicted = np.asarray(dggt_depth, dtype=np.float64)
    metric = np.asarray(lidar_depth_m, dtype=np.float64)
    if predicted.shape != metric.shape or predicted.ndim != 3:
        raise ValueError(
            "dggt_depth and lidar_depth_m must be matching [T,H,W] arrays, "
            f"got {predicted.shape} and {metric.shape}"
        )

    frame_rows: list[dict[str, Any]] = []
    eligible_scales: list[float] = []
    total_valid = 0
    for frame_index in range(int(predicted.shape[0])):
        pred_frame = predicted[frame_index]
        metric_frame = metric[frame_index]
        valid = (
            np.isfinite(pred_frame)
            & np.isfinite(metric_frame)
            & (pred_frame > 1.0e-6)
            & (metric_frame > float(min_depth_m))
            & (metric_frame < float(max_depth_m))
        )
        n_valid = int(valid.sum())
        total_valid += n_valid
        frame_scale = (
            float(np.median(pred_frame[valid] / metric_frame[valid]))
            if n_valid > 0
            else None
        )
        eligible = n_valid >= int(min_pixels_per_frame) and _is_finite_positive(frame_scale)
        if eligible:
            eligible_scales.append(float(frame_scale))
        frame_rows.append(
            {
                "frame_index": frame_index,
                "scale": frame_scale,
                "n_valid_px": n_valid,
                "eligible": bool(eligible),
            }
        )

    aggregate = aggregate_scale_values(eligible_scales)
    valid_trunk = (
        total_valid >= int(min_pixels_per_trunk)
        and aggregate.n_inlier >= int(min_valid_frames)
        and aggregate.value is not None
    )
    midpoint = int(predicted.shape[0]) // 2
    first_half = aggregate_scale_values(
        row["scale"]
        for row in frame_rows
        if row["eligible"] and int(row["frame_index"]) < midpoint
    )
    second_half = aggregate_scale_values(
        row["scale"]
        for row in frame_rows
        if row["eligible"] and int(row["frame_index"]) > midpoint
    )
    half_log_delta = (
        math.log(float(second_half.value) / float(first_half.value))
        if _is_finite_positive(first_half.value) and _is_finite_positive(second_half.value)
        else None
    )
    eligible_array = np.asarray(eligible_scales, dtype=np.float64)
    frame_max_min = (
        float(eligible_array.max() / eligible_array.min())
        if eligible_array.size and float(eligible_array.min()) > 0.0
        else None
    )
    return {
        "scale": aggregate.value if valid_trunk else None,
        "candidate_scale": aggregate.value,
        "valid": bool(valid_trunk),
        "n_valid_px": total_valid,
        "n_valid_frames": aggregate.n_inlier,
        "n_candidate_frames": aggregate.n_total,
        "aggregate": aggregate.to_dict(),
        "first_half_aggregate": first_half.to_dict(),
        "second_half_aggregate": second_half.to_dict(),
        "second_over_first_log_scale": half_log_delta,
        "frame_max_over_min": frame_max_min,
        "per_frame": frame_rows,
    }


def estimate_pooled_depth_scale(
    dggt_depth: np.ndarray,
    lidar_depth_m: np.ndarray,
    *,
    min_depth_m: float = 1.0,
    max_depth_m: float = 80.0,
) -> tuple[float | None, int]:
    """Reproduce the old pooled-pixel depth statistic on a selected window."""

    predicted = np.asarray(dggt_depth, dtype=np.float64)
    metric = np.asarray(lidar_depth_m, dtype=np.float64)
    if predicted.shape != metric.shape:
        raise ValueError("pooled depth arrays must have matching shapes")
    valid = (
        np.isfinite(predicted)
        & np.isfinite(metric)
        & (predicted > 1.0e-6)
        & (metric > float(min_depth_m))
        & (metric < float(max_depth_m))
    )
    count = int(valid.sum())
    if count == 0:
        return None, 0
    return float(np.median(predicted[valid] / metric[valid])), count


def umeyama_scale(src_metric: np.ndarray, dst_dggt: np.ndarray) -> float:
    """Fit the Sim(3) scale mapping metric centres to DGGT centres."""

    src = np.asarray(src_metric, dtype=np.float64)
    dst = np.asarray(dst_dggt, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be matching [N,3], got {src.shape}/{dst.shape}")
    if src.shape[0] < 2 or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Umeyama inputs must contain at least two finite points")
    src_centered = src - src.mean(axis=0, keepdims=True)
    dst_centered = dst - dst.mean(axis=0, keepdims=True)
    variance = float(np.square(src_centered).sum() / src.shape[0])
    if variance <= 1.0e-12:
        raise ValueError("metric camera centres are degenerate")
    covariance = dst_centered.T @ src_centered / src.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)
    signs = np.ones((3,), dtype=np.float64)
    if np.linalg.det(u @ vh) < 0.0:
        signs[-1] = -1.0
    scale = float(np.sum(singular_values * signs) / variance)
    if not _is_finite_positive(scale):
        raise ValueError(f"Umeyama produced invalid scale {scale}")
    return scale


def camera_path_length(camera_centres: np.ndarray) -> float:
    centres = np.asarray(camera_centres, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError("camera_centres must be [N,3]")
    if centres.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centres, axis=0), axis=-1).sum())


def camera_motion_diagnostics(camera_centres: np.ndarray) -> dict[str, float]:
    """Report path and observability diagnostics for a camera trajectory."""

    centres = np.asarray(camera_centres, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError("camera_centres must be [N,3]")
    path = camera_path_length(centres)
    if centres.shape[0] == 0:
        return {"path_m": 0.0, "endpoint_m": 0.0, "span_m": 0.0, "centered_rms_m": 0.0}
    endpoint = float(np.linalg.norm(centres[-1] - centres[0]))
    pairwise = centres[:, None, :] - centres[None, :, :]
    span = float(np.linalg.norm(pairwise, axis=-1).max())
    centered = centres - centres.mean(axis=0, keepdims=True)
    centered_rms = float(np.sqrt(np.square(centered).sum(axis=-1).mean()))
    return {
        "path_m": path,
        "endpoint_m": endpoint,
        "span_m": span,
        "centered_rms_m": centered_rms,
    }


@dataclass(frozen=True)
class ActorBox:
    object_id: str
    class_name: str
    object_to_camera: np.ndarray
    size_lwh_m: np.ndarray
    semantic_values: tuple[int, ...]


@dataclass(frozen=True)
class RayBoxPatch:
    x0: int
    x1: int
    y0: int
    y1: int
    front_depth_m: np.ndarray
    back_depth_m: np.ndarray
    hit_mask: np.ndarray
    truncated: bool


def oriented_box_ray_depth_patch(
    intrinsics: np.ndarray,
    object_to_camera: np.ndarray,
    size_lwh_m: np.ndarray,
    image_hw: Sequence[int],
    *,
    near_plane_m: float = 1.0e-3,
    edge_margin_px: int = 2,
) -> RayBoxPatch | None:
    """Rasterize metric ray entry/exit depths for one oriented cuboid.

    Rays use OpenCV camera coordinates and are parameterized as
    ``[(u-cx)/fx, (v-cy)/fy, 1] * z``; the slab-intersection parameter is
    therefore camera-z depth in metres, matching DGGT's depth convention.
    """

    K = np.asarray(intrinsics, dtype=np.float64)
    o2c = np.asarray(object_to_camera, dtype=np.float64)
    size = np.asarray(size_lwh_m, dtype=np.float64).reshape(-1)
    height, width = int(image_hw[0]), int(image_hw[1])
    if K.shape != (3, 3) or o2c.shape != (4, 4) or size.shape != (3,):
        raise ValueError("K/o2c/size must have shapes [3,3], [4,4], [3]")
    if height <= 0 or width <= 0 or not np.isfinite(K).all() or not np.isfinite(o2c).all():
        raise ValueError("invalid camera geometry")
    if not np.isfinite(size).all() or bool((size <= 0.0).any()):
        raise ValueError("box lwh must be finite and positive")

    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners_object = 0.5 * signs * size[None]
    corners_camera = corners_object @ o2c[:3, :3].T + o2c[:3, 3]
    if bool((corners_camera[:, 2] <= float(near_plane_m)).any()):
        # Near-plane clipping is possible, but such boxes are a poor metric
        # ruler and are deliberately excluded from this diagnostic.
        return None
    projected = corners_camera @ K.T
    xy = projected[:, :2] / projected[:, 2:3]
    raw_x0, raw_y0 = np.min(xy, axis=0)
    raw_x1, raw_y1 = np.max(xy, axis=0)
    if raw_x1 < 0.0 or raw_y1 < 0.0 or raw_x0 > width - 1 or raw_y0 > height - 1:
        return None
    truncated = bool(
        raw_x0 < float(edge_margin_px)
        or raw_y0 < float(edge_margin_px)
        or raw_x1 > float(width - 1 - edge_margin_px)
        or raw_y1 > float(height - 1 - edge_margin_px)
    )
    x0 = max(0, int(math.floor(float(raw_x0))))
    y0 = max(0, int(math.floor(float(raw_y0))))
    x1 = min(width, int(math.ceil(float(raw_x1))) + 1)
    y1 = min(height, int(math.ceil(float(raw_y1))) + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64),
        np.arange(x0, x1, dtype=np.float64),
        indexing="ij",
    )
    rays_camera = np.stack(
        (
            (xx - K[0, 2]) / K[0, 0],
            (yy - K[1, 2]) / K[1, 1],
            np.ones_like(xx),
        ),
        axis=-1,
    )
    camera_to_object = np.linalg.inv(o2c)
    ray_origin_object = camera_to_object[:3, 3]
    rays_object = rays_camera @ camera_to_object[:3, :3].T
    lower = -0.5 * size
    upper = 0.5 * size
    entry = np.full(xx.shape, -np.inf, dtype=np.float64)
    exit_ = np.full(xx.shape, np.inf, dtype=np.float64)
    possible = np.ones(xx.shape, dtype=np.bool_)
    epsilon = 1.0e-12
    for axis in range(3):
        direction = rays_object[..., axis]
        origin = float(ray_origin_object[axis])
        nonparallel = np.abs(direction) > epsilon
        outside_parallel = (~nonparallel) & ((origin < lower[axis]) | (origin > upper[axis]))
        possible &= ~outside_parallel
        safe_direction = np.where(nonparallel, direction, 1.0)
        t0 = (lower[axis] - origin) / safe_direction
        t1 = (upper[axis] - origin) / safe_direction
        axis_entry = np.where(nonparallel, np.minimum(t0, t1), -np.inf)
        axis_exit = np.where(nonparallel, np.maximum(t0, t1), np.inf)
        entry = np.maximum(entry, axis_entry)
        exit_ = np.minimum(exit_, axis_exit)
    hit = (
        possible
        & np.isfinite(entry)
        & np.isfinite(exit_)
        & (entry > float(near_plane_m))
        & (exit_ >= entry)
    )
    return RayBoxPatch(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        front_depth_m=entry.astype(np.float32),
        back_depth_m=exit_.astype(np.float32),
        hit_mask=hit,
        truncated=truncated,
    )


def estimate_actor_frame_scale(
    dggt_depth: np.ndarray,
    semantic_labels: np.ndarray,
    intrinsics: np.ndarray,
    boxes: Sequence[ActorBox],
    *,
    min_pixels_per_actor: int = 32,
    edge_margin_px: int = 2,
    min_consensus_fraction: float = 0.6,
    max_log_interval_width: float = 0.25,
) -> dict[str, Any]:
    """Estimate one actor scale interval for a frame without reading lidar."""

    depth = np.asarray(dggt_depth, dtype=np.float64)
    labels = np.asarray(semantic_labels)
    if depth.ndim != 2 or labels.shape != depth.shape:
        raise ValueError("depth and semantic_labels must be matching [H,W] arrays")
    height, width = depth.shape
    prepared: list[tuple[ActorBox, RayBoxPatch]] = []
    rejected_geometry = 0
    rejected_truncated = 0
    for box in boxes:
        patch = oriented_box_ray_depth_patch(
            intrinsics,
            box.object_to_camera,
            box.size_lwh_m,
            (height, width),
            edge_margin_px=edge_margin_px,
        )
        if patch is None or not bool(patch.hit_mask.any()):
            rejected_geometry += 1
            continue
        if patch.truncated:
            rejected_truncated += 1
            continue
        prepared.append((box, patch))

    # An annotation z-buffer assigns overlapping same-class pixels to the
    # nearest cuboid before semantic filtering.  This avoids counting one car
    # in every overlapping vehicle box.
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int32)
    for box_index, (_, patch) in enumerate(prepared):
        target_depth = nearest[patch.y0 : patch.y1, patch.x0 : patch.x1]
        target_owner = owner[patch.y0 : patch.y1, patch.x0 : patch.x1]
        update = patch.hit_mask & (patch.front_depth_m < target_depth)
        target_depth[update] = patch.front_depth_m[update]
        target_owner[update] = int(box_index)

    observations: list[dict[str, Any]] = []
    rejected_pixels = 0
    for box_index, (box, patch) in enumerate(prepared):
        ys = slice(patch.y0, patch.y1)
        xs = slice(patch.x0, patch.x1)
        local_depth = depth[ys, xs]
        semantic = np.isin(labels[ys, xs], np.asarray(box.semantic_values))
        valid = (
            patch.hit_mask
            & (owner[ys, xs] == int(box_index))
            & semantic
            & np.isfinite(local_depth)
            & (local_depth > 1.0e-6)
        )
        n_pixels = int(valid.sum())
        if n_pixels < int(min_pixels_per_actor):
            rejected_pixels += 1
            continue
        surface_ratios = local_depth[valid] / patch.front_depth_m[valid]
        surface_aggregate = aggregate_scale_values(surface_ratios)
        lower_log = np.log(local_depth[valid]) - np.log(patch.back_depth_m[valid])
        upper_log = np.log(local_depth[valid]) - np.log(patch.front_depth_m[valid])
        consensus = maximum_consensus_log_interval(
            lower_log,
            upper_log,
            min_support_fraction=min_consensus_fraction,
            max_log_width=max_log_interval_width,
        )
        if surface_aggregate.value is None or consensus["low_log_scale"] is None:
            continue
        point_scale = (
            math.exp(float(consensus["point_log_scale"]))
            if consensus["point_log_scale"] is not None
            else None
        )
        observations.append(
            {
                "object_id": str(box.object_id),
                "class_name": str(box.class_name),
                "scale": point_scale,
                "scale_surface_proxy": float(surface_aggregate.value),
                "scale_interval_low": math.exp(float(consensus["low_log_scale"])),
                "scale_interval_high": math.exp(float(consensus["high_log_scale"])),
                "consensus": consensus,
                "n_pixels": n_pixels,
                "surface_proxy_pixel_aggregate": surface_aggregate.to_dict(),
            }
        )

    # Each actor contributes one interval irrespective of its pixel area.
    valid_observations = [
        obs for obs in observations if bool(obs.get("consensus", {}).get("valid", False))
    ]
    frame_consensus = maximum_consensus_log_interval(
        (math.log(float(obs["scale_interval_low"])) for obs in valid_observations),
        (math.log(float(obs["scale_interval_high"])) for obs in valid_observations),
        min_support_fraction=min_consensus_fraction,
        max_log_width=max_log_interval_width,
    )
    frame_scale = (
        math.exp(float(frame_consensus["point_log_scale"]))
        if frame_consensus["point_log_scale"] is not None
        else None
    )
    surface_proxy = aggregate_scale_values(
        obs["scale_surface_proxy"] for obs in observations
    )
    return {
        "scale": frame_scale,
        "scale_surface_proxy": surface_proxy.value,
        "scale_interval_low": (
            math.exp(float(frame_consensus["low_log_scale"]))
            if frame_consensus["low_log_scale"] is not None
            else None
        ),
        "scale_interval_high": (
            math.exp(float(frame_consensus["high_log_scale"]))
            if frame_consensus["high_log_scale"] is not None
            else None
        ),
        "consensus": frame_consensus,
        "n_actors": len(valid_observations),
        "n_actor_candidates": len(observations),
        "n_pixels": int(sum(int(obs["n_pixels"]) for obs in observations)),
        "aggregate": aggregate_scale_values(obs["scale"] for obs in observations).to_dict(),
        "surface_proxy_aggregate": surface_proxy.to_dict(),
        "coverage": {
            "n_input_boxes": len(boxes),
            "n_prepared_boxes": len(prepared),
            "n_rejected_geometry": rejected_geometry,
            "n_rejected_truncated": rejected_truncated,
            "n_rejected_insufficient_pixels": rejected_pixels,
            "n_rejected_consensus": len(observations) - len(valid_observations),
        },
        "observations": observations,
    }


def aggregate_actor_trunk(
    per_frame: Sequence[Mapping[str, Any]],
    *,
    min_valid_frames: int = 3,
    min_consensus_fraction: float = 0.6,
    max_log_interval_width: float = 0.25,
) -> dict[str, Any]:
    """MAD-filter equal-weighted frame points into one actor trunk scale.

    Pixel/object/frame levels retain their strict ray-interval consensus.  At
    the trunk level, however, ordinary frame-to-frame depth noise shifts those
    already-narrow intervals enough that requiring a literal common
    intersection is over-strict.  The trunk point is therefore the MAD-filtered
    median of valid frame points.  The strict interval intersection is retained
    as a diagnostic only.
    """

    point_frames = [
        frame
        for frame in per_frame
        if bool(frame.get("consensus", {}).get("valid", False))
        and _is_finite_positive(frame.get("scale"))
    ]
    point_values = np.asarray(
        [float(frame["scale"]) for frame in point_frames],
        dtype=np.float64,
    )
    point_aggregate = aggregate_scale_values(point_values)
    point_keep, _, _ = _mad_inlier_mask(point_values)
    inlier_frames = [
        frame for frame, keep in zip(point_frames, point_keep.tolist()) if bool(keep)
    ]
    surface_proxy = aggregate_scale_values(
        frame.get("scale_surface_proxy") for frame in per_frame
    )
    interval_frames = [
        frame
        for frame in per_frame
        if bool(frame.get("consensus", {}).get("valid", False))
        and _is_finite_positive(frame.get("scale_interval_low"))
        and _is_finite_positive(frame.get("scale_interval_high"))
    ]
    trunk_consensus = maximum_consensus_log_interval(
        (math.log(float(frame["scale_interval_low"])) for frame in interval_frames),
        (math.log(float(frame["scale_interval_high"])) for frame in interval_frames),
        min_support_fraction=min_consensus_fraction,
        max_log_width=max_log_interval_width,
    )
    strict_interval_low = (
        math.exp(float(trunk_consensus["low_log_scale"]))
        if trunk_consensus["low_log_scale"] is not None
        else None
    )
    strict_interval_high = (
        math.exp(float(trunk_consensus["high_log_scale"]))
        if trunk_consensus["high_log_scale"] is not None
        else None
    )
    point = point_aggregate.value

    # Build a reference-free robust uncertainty band.  The noise term comes
    # from dispersion of the retained frame log-points.  The geometric term
    # prevents the trunk band from becoming narrower than a typical valid
    # per-frame ray/box interval.  Neither term reads lidar or s_depth.
    inlier_log_points = np.asarray(
        [math.log(float(frame["scale"])) for frame in inlier_frames],
        dtype=np.float64,
    )
    if inlier_log_points.size and _is_finite_positive(point):
        log_center = math.log(float(point))
        log_mad = float(np.median(np.abs(inlier_log_points - np.median(inlier_log_points))))
        robust_sigma_log = 1.4826 * log_mad
        noise_radius_log = 3.5 * robust_sigma_log
        frame_half_widths = [
            0.5
            * (
                math.log(float(frame["scale_interval_high"]))
                - math.log(float(frame["scale_interval_low"]))
            )
            for frame in inlier_frames
            if _is_finite_positive(frame.get("scale_interval_low"))
            and _is_finite_positive(frame.get("scale_interval_high"))
            and float(frame["scale_interval_low"]) <= float(frame["scale_interval_high"])
        ]
        median_frame_half_width_log = (
            float(np.median(np.asarray(frame_half_widths, dtype=np.float64)))
            if frame_half_widths
            else 0.0
        )
        radius_log = max(noise_radius_log, median_frame_half_width_log)
        interval_low = math.exp(log_center - radius_log)
        interval_high = math.exp(log_center + radius_log)
    else:
        log_center = None
        log_mad = None
        robust_sigma_log = None
        noise_radius_log = None
        median_frame_half_width_log = None
        radius_log = None
        interval_low = None
        interval_high = None

    uncertainty_valid = (
        point_aggregate.n_inlier >= int(min_valid_frames)
        and _is_finite_positive(interval_low)
        and _is_finite_positive(interval_high)
    )
    point_in_interval = (
        _is_finite_positive(point)
        and _is_finite_positive(interval_low)
        and _is_finite_positive(interval_high)
        and float(interval_low) <= float(point) <= float(interval_high)
    )
    valid = bool(uncertainty_valid and point_in_interval)
    robust_uncertainty = {
        "valid": bool(uncertainty_valid),
        "method": (
            "inlier frame log-point median +/- max(3.5*1.4826*MAD, "
            "median per-frame log half-width)"
        ),
        "scale_interval_low": interval_low,
        "scale_interval_high": interval_high,
        "log_center": log_center,
        "log_mad": log_mad,
        "robust_sigma_log": robust_sigma_log,
        "noise_radius_log": noise_radius_log,
        "median_frame_half_width_log": median_frame_half_width_log,
        "radius_log": radius_log,
        "n_candidate_frames": int(point_values.size),
        "n_inlier_frames": point_aggregate.n_inlier,
    }
    strict_consensus_diagnostic = dict(trunk_consensus)
    strict_consensus_diagnostic["scale_interval_low"] = strict_interval_low
    strict_consensus_diagnostic["scale_interval_high"] = strict_interval_high
    return {
        "scale": point if valid else None,
        "candidate_scale": point,
        "frame_median_candidate_scale": point_aggregate.value,
        "scale_surface_proxy": surface_proxy.value,
        "scale_interval_low": interval_low,
        "scale_interval_high": interval_high,
        "valid": bool(valid),
        "point_in_interval": bool(point_in_interval),
        "n_valid_frames": point_aggregate.n_inlier,
        "n_actor_observations": int(
            sum(int(frame.get("n_actors", 0)) for frame in per_frame)
        ),
        "n_pixels": int(sum(int(frame.get("n_pixels", 0)) for frame in per_frame)),
        "aggregate": point_aggregate.to_dict(),
        "surface_proxy_aggregate": surface_proxy.to_dict(),
        "robust_uncertainty": robust_uncertainty,
        "strict_interval_consensus_diagnostic": strict_consensus_diagnostic,
        "per_frame": [dict(frame) for frame in per_frame],
    }


def _actor_class_allowed(class_name: str, requested: set[str]) -> bool:
    if "all" in requested:
        return bool(_waymo_semantic_values_for_class(class_name))
    name = str(class_name).lower()
    return any(token in name for token in requested)


def index_actor_annotations(
    metadata: Mapping[str, Any] | None,
    frame_indices: Sequence[int],
    *,
    requested_classes: set[str],
) -> dict[int, list[dict[str, Any]]]:
    """Index raw Waymo actor annotations without appearance/reference filters."""

    wanted = {int(frame) for frame in frame_indices}
    indexed: dict[int, list[dict[str, Any]]] = {frame: [] for frame in wanted}
    if not metadata:
        return indexed
    instances = metadata.get("instances_info", {})
    if not isinstance(instances, Mapping):
        return indexed
    for metadata_id, info in instances.items():
        if not isinstance(info, Mapping):
            continue
        class_name = str(info.get("class_name", ""))
        semantic_values = tuple(int(v) for v in _waymo_semantic_values_for_class(class_name))
        if not semantic_values or not _actor_class_allowed(class_name, requested_classes):
            continue
        annotations = info.get("frame_annotations", {})
        if not isinstance(annotations, Mapping):
            continue
        frames = annotations.get("frame_idx", [])
        poses = annotations.get("obj_to_world", [])
        sizes = annotations.get("box_size", [])
        for position, frame_value in enumerate(frames):
            frame = int(frame_value)
            if frame not in wanted or position >= len(poses) or position >= len(sizes):
                continue
            try:
                object_to_world = np.asarray(poses[position], dtype=np.float64).reshape(4, 4)
                size_lwh = np.asarray(sizes[position], dtype=np.float64).reshape(3)
            except (TypeError, ValueError):
                continue
            if (
                not np.isfinite(object_to_world).all()
                or not np.isfinite(size_lwh).all()
                or bool((size_lwh <= 0.0).any())
            ):
                continue
            indexed[frame].append(
                {
                    "object_id": str(info.get("raw_object_id", info.get("id", metadata_id))),
                    "class_name": class_name,
                    "semantic_values": semantic_values,
                    "object_to_world": object_to_world,
                    "size_lwh_m": size_lwh,
                }
            )
    return indexed


def _front_semantic_path(scene_root: Path, frame_index: int) -> Path | None:
    for suffix in ("png", "jpg"):
        candidate = scene_root / "custom_masks" / f"{int(frame_index):03d}_0.{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _load_model(checkpoint_path: str, device: torch.device) -> VGGT:
    model = VGGT().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a state dict")
    clean_state = {
        key[len("module.") :] if str(key).startswith("module.") else str(key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(clean_state, strict=False)
    important_missing = [
        key
        for key in incompatible.missing_keys
        if key.startswith(("aggregator.", "camera_head.", "depth_head."))
    ]
    if important_missing:
        raise RuntimeError(
            "checkpoint is missing aggregator/camera/depth weights: "
            + ", ".join(important_missing[:20])
        )
    model.eval()
    return model


def _predict_full_trunk(
    model: VGGT,
    context_images: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    images = context_images.unsqueeze(0) if context_images.ndim == 4 else context_images
    images = images.to(device)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        output = model.get_aggregator_token_outputs(images)
        aggregated = output["aggregated_tokens_list"]
        pose_encoding = model.camera_head(aggregated)[-1].float()
        with torch.autocast(device_type=device.type, enabled=False):
            depth, _ = model.depth_head(
                [tokens.float() for tokens in aggregated],
                images=images,
                patch_start_idx=int(output["patch_start_idx"]),
            )
    c2w = dggt_pose_encoding_to_camera_to_world(pose_encoding)[0].detach().cpu().numpy()
    depth_array = depth[0].detach().float().cpu().numpy()
    if depth_array.ndim == 4 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if depth_array.ndim != 3:
        raise RuntimeError(f"unexpected DGGT depth shape {depth_array.shape}")
    del output, aggregated, pose_encoding, depth, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return c2w.astype(np.float64), depth_array.astype(np.float32)


def _load_lidar_depth_trunk(
    image_dir: str,
    scene: str,
    frame_indices: Sequence[int],
    image_hw: Sequence[int],
) -> tuple[np.ndarray | None, list[str]]:
    paths = [
        Path(image_dir) / scene / "depth_flows_4" / f"{int(frame):03d}_0.npy"
        for frame in frame_indices
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return None, missing
    tensor = load_and_preprocess_flow(
        [str(path) for path in paths],
        None,
        None,
        int(image_hw[0]),
        int(image_hw[1]),
    ).float()
    array = tensor.cpu().numpy()
    if array.ndim == 4:
        array = array[..., 0]
    if array.ndim != 3:
        raise RuntimeError(f"unexpected lidar depth shape {array.shape}")
    return array.astype(np.float32), []


def _evaluate_actor_ruler(
    *,
    dggt_depth: np.ndarray,
    intrinsics_model: np.ndarray,
    camera_to_world_absolute: Mapping[int, np.ndarray],
    annotations_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    scene_root: Path,
    frame_indices: Sequence[int],
    min_pixels_per_actor: int,
    min_valid_frames: int,
    edge_margin_px: int,
    min_consensus_fraction: float,
    max_log_interval_width: float,
) -> dict[str, Any]:
    per_frame: list[dict[str, Any]] = []
    for local_index, frame in enumerate(frame_indices):
        semantic_path = _front_semantic_path(scene_root, int(frame))
        labels = (
            _load_waymo_semantic_labels_model(str(semantic_path), target_width=518)
            if semantic_path is not None
            else None
        )
        if labels is None or tuple(labels.shape) != tuple(dggt_depth[local_index].shape):
            per_frame.append(
                {
                    "frame_index": int(frame),
                    "scale": None,
                    "scale_interval_low": None,
                    "scale_interval_high": None,
                    "n_actors": 0,
                    "n_pixels": 0,
                    "aggregate": aggregate_scale_values([]).to_dict(),
                    "observations": [],
                    "reason": "semantic_mask_missing_or_shape_mismatch",
                }
            )
            continue
        world_to_camera = np.linalg.inv(camera_to_world_absolute[int(frame)])
        boxes: list[ActorBox] = []
        for record in annotations_by_frame.get(int(frame), []):
            boxes.append(
                ActorBox(
                    object_id=str(record["object_id"]),
                    class_name=str(record["class_name"]),
                    object_to_camera=world_to_camera
                    @ np.asarray(record["object_to_world"], dtype=np.float64),
                    size_lwh_m=np.asarray(record["size_lwh_m"], dtype=np.float64),
                    semantic_values=tuple(int(v) for v in record["semantic_values"]),
                )
            )
        frame_result = estimate_actor_frame_scale(
            dggt_depth[local_index],
            labels,
            intrinsics_model,
            boxes,
            min_pixels_per_actor=min_pixels_per_actor,
            edge_margin_px=edge_margin_px,
            min_consensus_fraction=min_consensus_fraction,
            max_log_interval_width=max_log_interval_width,
        )
        frame_result["frame_index"] = int(frame)
        frame_result["n_candidate_boxes"] = len(boxes)
        per_frame.append(frame_result)
    return aggregate_actor_trunk(
        per_frame,
        min_valid_frames=min_valid_frames,
        min_consensus_fraction=min_consensus_fraction,
        max_log_interval_width=max_log_interval_width,
    )


def evaluate_trunk(
    *,
    model: VGGT,
    dataset: WaymoOpenDataset,
    image_dir: str,
    scene: str,
    trunk_index: int,
    device: torch.device,
    requested_actor_classes: set[str],
    min_depth_pixels_per_frame: int,
    min_depth_pixels_per_trunk: int,
    min_depth_frames: int,
    actor_min_pixels: int,
    actor_min_frames: int,
    actor_edge_margin: int,
    actor_min_consensus_fraction: float,
    actor_max_log_interval_width: float,
    run_actor: bool,
) -> dict[str, Any]:
    base = int(trunk_index) * TRUNK_FRAMES
    frames = list(range(base, base + TRUNK_FRAMES))
    dataset.start_idx = base
    item = dataset[0]
    context = item["dggt_context_images"]
    if int(context.shape[0]) != TRUNK_FRAMES:
        raise RuntimeError(f"expected 29 context frames, got {tuple(context.shape)}")
    dggt_c2w, dggt_depth = _predict_full_trunk(model, context, device)
    image_hw = tuple(int(v) for v in dggt_depth.shape[-2:])

    camera_metadata = dataset._load_waymo_camera_metadata(0, frames)
    camera_to_ego = np.asarray(camera_metadata["camera_to_ego_dataset"], dtype=np.float64)
    ego_to_world = camera_metadata["ego_to_world"]
    camera_to_world_absolute = {
        frame: np.asarray(ego_to_world[frame], dtype=np.float64) @ camera_to_ego
        for frame in frames
    }
    anchor_inverse = np.linalg.inv(camera_to_world_absolute[base])
    metric_c2a = np.stack(
        [anchor_inverse @ camera_to_world_absolute[frame] for frame in frames],
        axis=0,
    )
    metric_centres = metric_c2a[:, :3, 3]
    dggt_centres = dggt_c2w[:, :3, 3]
    motion_29 = camera_motion_diagnostics(metric_centres)
    try:
        s_cam_raw = umeyama_scale(metric_centres, dggt_centres)
    except ValueError:
        s_cam_raw = None
    # Span, unlike accumulated path, cannot be inflated by stationary jitter
    # and directly measures the baseline that makes a scale identifiable.
    s_cam = s_cam_raw if motion_29["span_m"] > 2.0 else None

    lidar_depth, missing_lidar = _load_lidar_depth_trunk(
        image_dir,
        scene,
        frames,
        image_hw,
    )
    if lidar_depth is None:
        depth_result = {
            "scale": None,
            "candidate_scale": None,
            "valid": False,
            "n_valid_px": 0,
            "n_valid_frames": 0,
            "n_candidate_frames": 0,
            "aggregate": aggregate_scale_values([]).to_dict(),
            "first_half_aggregate": aggregate_scale_values([]).to_dict(),
            "second_half_aggregate": aggregate_scale_values([]).to_dict(),
            "second_over_first_log_scale": None,
            "frame_max_over_min": None,
            "per_frame": [],
            "missing_files": missing_lidar,
        }
    else:
        depth_result = estimate_depth_ruler(
            dggt_depth,
            lidar_depth,
            min_pixels_per_frame=min_depth_pixels_per_frame,
            min_pixels_per_trunk=min_depth_pixels_per_trunk,
            min_valid_frames=min_depth_frames,
        )
        depth_result["missing_files"] = []

    K_raw = torch.as_tensor(camera_metadata["intrinsics"], dtype=torch.float32)
    raw_to_model, model_hw_tuple = raw_to_model_canvas_homography(
        camera_metadata["raw_hw"],
        target_width=518,
        patch_size=14,
    )
    K_model = raw_to_model.to(dtype=K_raw.dtype) @ K_raw
    if model_hw_tuple != image_hw:
        raise RuntimeError(
            f"Waymo/model canvas mismatch: intrinsics helper gives {model_hw_tuple}, depth is {image_hw}"
        )

    if run_actor:
        raw_actor_metadata = dataset._load_instance_metadata(0)
        annotations = index_actor_annotations(
            raw_actor_metadata,
            frames,
            requested_classes=requested_actor_classes,
        )
        actor_result = _evaluate_actor_ruler(
            dggt_depth=dggt_depth,
            intrinsics_model=K_model.detach().cpu().numpy(),
            camera_to_world_absolute=camera_to_world_absolute,
            annotations_by_frame=annotations,
            scene_root=Path(image_dir) / scene,
            frame_indices=frames,
            min_pixels_per_actor=actor_min_pixels,
            min_valid_frames=actor_min_frames,
            edge_margin_px=actor_edge_margin,
            min_consensus_fraction=actor_min_consensus_fraction,
            max_log_interval_width=actor_max_log_interval_width,
        )
    else:
        actor_result = aggregate_actor_trunk(
            [],
            min_valid_frames=actor_min_frames,
            min_consensus_fraction=actor_min_consensus_fraction,
            max_log_interval_width=actor_max_log_interval_width,
        )
        actor_result["disabled"] = True

    selected = np.asarray(item["dggt_window_indices"], dtype=np.int64).reshape(-1)
    if selected.size != LEGACY_WINDOW_FRAMES:
        raise RuntimeError(f"legacy auxiliary expected 10 selected frames, got {selected.tolist()}")
    legacy_metric_centres = metric_centres[selected]
    legacy_dggt_centres = dggt_centres[selected]
    legacy_motion = camera_path_length(legacy_metric_centres)
    try:
        legacy_s_cam = umeyama_scale(legacy_metric_centres, legacy_dggt_centres)
    except ValueError:
        legacy_s_cam = None
    if lidar_depth is None:
        legacy_s_depth, legacy_valid_px = None, 0
    else:
        legacy_s_depth, legacy_valid_px = estimate_pooled_depth_scale(
            dggt_depth[selected],
            lidar_depth[selected],
        )

    return {
        "scene": str(scene),
        "trunk": int(trunk_index),
        "frame_start": base,
        "frame_end": base + TRUNK_FRAMES - 1,
        "s_cam": s_cam,
        "s_depth": depth_result.get("scale"),
        "s_actor": actor_result.get("scale"),
        "metres_per_dggt_unit": (
            1.0 / float(depth_result["scale"])
            if _is_finite_positive(depth_result.get("scale"))
            else None
        ),
        "camera_29f": {
            "scale": s_cam,
            "candidate_scale": s_cam_raw,
            "valid": bool(s_cam is not None),
            "stationary": bool(motion_29["span_m"] <= 2.0),
            "motion": motion_29,
            "validity_rule": "maximum pairwise camera-centre span over 29 frames > 2 metres",
        },
        "depth_29f": depth_result,
        "actor_29f": actor_result,
        "legacy_10f": {
            "window_local_indices": selected.tolist(),
            "global_frames": [base + int(index) for index in selected],
            "ego_motion_m": legacy_motion,
            "stationary": bool(legacy_motion <= 2.0),
            "s_cam": legacy_s_cam,
            "s_depth": legacy_s_depth,
            "n_valid_depth_px": legacy_valid_px,
            "cam_depth_ratio": (
                float(legacy_s_cam) / float(legacy_s_depth)
                if _is_finite_positive(legacy_s_cam) and _is_finite_positive(legacy_s_depth)
                else None
            ),
        },
    }


def _scalar_summary(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _pair_summary(pairs: Iterable[tuple[float | None, float | None]]) -> dict[str, Any]:
    valid = [
        (float(first), float(second))
        for first, second in pairs
        if _is_finite_positive(first) and _is_finite_positive(second)
    ]
    if not valid:
        return {
            "n": 0,
            "ratio_mean": None,
            "ratio_std": None,
            "ratio_median": None,
            "correlation": None,
        }
    first = np.asarray([pair[0] for pair in valid], dtype=np.float64)
    second = np.asarray([pair[1] for pair in valid], dtype=np.float64)
    ratios = first / second
    log_ratios = np.log(ratios)
    correlation = (
        float(np.corrcoef(first, second)[0, 1])
        if first.size >= 2 and float(first.std()) > 0.0 and float(second.std()) > 0.0
        else None
    )
    return {
        "n": int(ratios.size),
        "ratio_mean": float(ratios.mean()),
        "ratio_std": float(ratios.std()),
        "ratio_median": float(np.median(ratios)),
        "ratio_mad": float(np.median(np.abs(ratios - np.median(ratios)))),
        "log_ratio_mean": float(log_ratios.mean()),
        "log_ratio_std": float(log_ratios.std()),
        "log_ratio_median": float(np.median(log_ratios)),
        "log_ratio_mad": float(
            np.median(np.abs(log_ratios - np.median(log_ratios)))
        ),
        "within_5pct": float(np.mean(np.abs(ratios - 1.0) <= 0.05)),
        "within_10pct": float(np.mean(np.abs(ratios - 1.0) <= 0.10)),
        "correlation": correlation,
    }


def _actor_interval_reference_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_key: str,
) -> dict[str, Any]:
    records: list[tuple[float, float, float]] = []
    any_interval_count = 0
    for row in rows:
        actor = row.get("actor_29f", {})
        low = actor.get("scale_interval_low")
        high = actor.get("scale_interval_high")
        reference = row.get(reference_key)
        has_interval = (
            _is_finite_positive(low)
            and _is_finite_positive(high)
            and _is_finite_positive(reference)
            and float(low) <= float(high)
        )
        if has_interval:
            any_interval_count += 1
        if has_interval and bool(actor.get("valid", False)) and bool(
            actor.get("robust_uncertainty", {}).get("valid", False)
        ):
            records.append((float(low), float(high), float(reference)))
    if not records:
        return {
            "n": 0,
            "n_any_interval": any_interval_count,
            "coverage": None,
            "log_interval_width": _scalar_summary([]),
            "reference_minus_interval_mid_log": _scalar_summary([]),
        }
    covered = [low <= reference <= high for low, high, reference in records]
    widths = [math.log(high) - math.log(low) for low, high, _ in records]
    offsets = [
        math.log(reference) - 0.5 * (math.log(low) + math.log(high))
        for low, high, reference in records
    ]
    return {
        "n": len(records),
        "n_any_interval": any_interval_count,
        "coverage": float(np.mean(covered)),
        "log_interval_width": _scalar_summary(widths),
        "reference_minus_interval_mid_log": _scalar_summary(offsets),
    }


def _group_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_rows = len(rows)
    valid_depth = [row for row in rows if bool(row.get("depth_29f", {}).get("valid", False))]
    return {
        "n": n_rows,
        "depth_valid_count": len(valid_depth),
        "depth_valid_fraction": len(valid_depth) / float(n_rows) if n_rows else None,
        "depth_frame_robust_cv": _scalar_summary(
            row.get("depth_29f", {}).get("aggregate", {}).get("cv_inlier")
            for row in rows
        ),
        "depth_second_over_first_log_scale": _scalar_summary(
            row.get("depth_29f", {}).get("second_over_first_log_scale")
            for row in rows
        ),
        "depth_frame_max_over_min": _scalar_summary(
            row.get("depth_29f", {}).get("frame_max_over_min") for row in rows
        ),
        "actor_over_depth": _pair_summary(
            (row.get("s_actor"), row.get("s_depth")) for row in rows
        ),
        "actor_depth_interval": _actor_interval_reference_summary(
            rows,
            reference_key="s_depth",
        ),
        "actor_over_camera": _pair_summary(
            (row.get("s_actor"), row.get("s_cam")) for row in rows
        ),
        "actor_camera_interval": _actor_interval_reference_summary(
            rows,
            reference_key="s_cam",
        ),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stationary_legacy = [
        row for row in rows if bool(row.get("legacy_10f", {}).get("stationary", False))
    ]
    moving_legacy = [row for row in rows if row not in stationary_legacy]
    stationary_29 = [
        row for row in rows if bool(row.get("camera_29f", {}).get("stationary", False))
    ]
    moving_29 = [row for row in rows if row not in stationary_29]
    by_scene: dict[str, int] = Counter(str(row.get("scene", "")) for row in rows)
    return {
        "n_trunks": len(rows),
        "trunks_per_scene": dict(sorted(by_scene.items())),
        "legacy_stationary_count": len(stationary_legacy),
        "legacy_moving_count": len(moving_legacy),
        "legacy_stationary_definition": "10-frame Waymo camera path length <= 2 metres",
        "full29_stationary_count": len(stationary_29),
        "full29_moving_count": len(moving_29),
        "full29_stationary_definition": "29-frame maximum pairwise camera-centre span <= 2 metres",
        "legacy_cam_over_depth": _pair_summary(
            (
                row.get("legacy_10f", {}).get("s_cam"),
                row.get("legacy_10f", {}).get("s_depth"),
            )
            for row in moving_legacy
        ),
        "full_cam_over_depth": _pair_summary(
            (row.get("s_cam"), row.get("s_depth")) for row in rows
        ),
        "actor_over_depth": _pair_summary(
            (row.get("s_actor"), row.get("s_depth")) for row in rows
        ),
        "actor_over_camera": _pair_summary(
            (row.get("s_actor"), row.get("s_cam")) for row in rows
        ),
        "actor_depth_interval_all": _actor_interval_reference_summary(
            rows,
            reference_key="s_depth",
        ),
        "depth_scale_all": _scalar_summary(row.get("s_depth") for row in rows),
        "depth_scale_stationary_10f_group": _scalar_summary(
            row.get("s_depth") for row in stationary_legacy
        ),
        "depth_frame_cv_stationary_10f_group": _scalar_summary(
            row.get("depth_29f", {}).get("aggregate", {}).get("cv_inlier")
            for row in stationary_legacy
        ),
        "actor_scale_all": _scalar_summary(row.get("s_actor") for row in rows),
        "full29_stationary_group": _group_diagnostics(stationary_29),
        "full29_moving_group": _group_diagnostics(moving_29),
    }


def result_definitions() -> dict[str, str]:
    return {
        "scale_direction": SCALE_DEFINITION,
        "s_depth": (
            "per frame median(DGGT depth / lidar metric depth) for 1m < lidar < 80m; "
            "MAD frame rejection; median retained frame scale"
        ),
        "s_cam": (
            "29-frame Umeyama Sim(3) scale from Waymo metric camera centres to DGGT "
            "camera centres; valid only when 29-frame maximum pairwise span > 2m"
        ),
        "s_actor": (
            "NO LIDAR: each visible class-semantic actor pixel bounds log s between "
            "DGGT-depth/metric-cuboid-exit and DGGT-depth/metric-cuboid-entry; maximum "
            "consensus per object/frame, then MAD+median over equal-weighted frame points"
        ),
        "s_actor_interval": (
            "trunk uncertainty is frame log-point median +/- the larger of robust MAD noise "
            "and median frame half-width; strict trunk intersection is diagnostic only; "
            "the annotated cuboid is not an exact mesh, so s_actor is not offline GT"
        ),
        "ruler_independence": (
            "camera, lidar, and actor boxes are distinct metric references; actor and lidar "
            "are not statistically independent because both ratios share DGGT depth"
        ),
        "legacy_10f": (
            "reproduction-only pooled depth median and 10-centre Umeyama; stationarity is "
            "10-frame camera path <= 2m"
        ),
    }


def parse_integer_spec(specification: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for raw_part in str(specification).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            candidates = range(start, end + step, step)
        else:
            candidates = (int(part),)
        for value in candidates:
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError("integer specification is empty")
    return values


def _print_pair(label: str, payload: Mapping[str, Any]) -> None:
    if int(payload.get("n", 0)) == 0:
        print(f"{label:<28}: n=0")
        return
    correlation = payload.get("correlation")
    corr_text = "n/a" if correlation is None else f"{float(correlation):.4f}"
    message = (
        f"{label:<28}: n={int(payload['n'])}  "
        f"ratio={float(payload['ratio_mean']):.4f}±{float(payload['ratio_std']):.4f}  "
        f"median={float(payload['ratio_median']):.4f}  corr={corr_text}"
    )
    print(message)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically checkpoint a diagnostic JSON after each completed trunk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        default=os.environ.get("GAUGE_IMAGE_DIR", DEFAULT_IMAGE_DIR),
        help="Waymo processed split root (default: GAUGE_IMAGE_DIR or %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("GAUGE_CKPT", DEFAULT_CHECKPOINT),
        help="DGGT checkpoint (default: GAUGE_CKPT or %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("GAUGE_DEVICE", "cuda:0"),
        help="Torch device (default: GAUGE_DEVICE or cuda:0)",
    )
    parser.add_argument("--scenes", default="300-329", help="Comma/range scene ids")
    parser.add_argument("--trunks", default="0,1,2", help="Comma/range trunk indices")
    parser.add_argument("--max-trunks", type=int, default=0, help="Stop after N trunks; 0 means all")
    parser.add_argument("--output-json", default="", help="Optional result JSON path")
    parser.add_argument(
        "--actor-classes",
        default="vehicle",
        help="Comma-separated class-name substrings, or 'all' (default: vehicle)",
    )
    parser.add_argument("--actor-min-pixels", type=int, default=32)
    parser.add_argument("--actor-min-frames", type=int, default=3)
    parser.add_argument("--actor-edge-margin", type=int, default=2)
    parser.add_argument("--actor-min-consensus-fraction", type=float, default=0.6)
    parser.add_argument("--actor-max-log-interval-width", type=float, default=0.25)
    parser.add_argument("--skip-actor", action="store_true", help="Disable the actor ruler")
    parser.add_argument("--min-depth-pixels-per-frame", type=int, default=64)
    parser.add_argument("--min-depth-pixels-per-trunk", type=int, default=5000)
    parser.add_argument("--min-depth-frames", type=int, default=15)
    parser.add_argument(
        "--min-success-fraction",
        type=float,
        default=0.9,
        help="Return nonzero when fewer than this fraction of requested trunks succeed",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on the first bad trunk")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    scenes = [f"{value:03d}" for value in parse_integer_spec(args.scenes)]
    trunks = parse_integer_spec(args.trunks)
    requested_actor_classes = {
        token.strip().lower() for token in str(args.actor_classes).split(",") if token.strip()
    }
    if not requested_actor_classes:
        raise ValueError("--actor-classes must not be empty")
    if not (0.0 < float(args.actor_min_consensus_fraction) <= 1.0):
        raise ValueError("--actor-min-consensus-fraction must be in (0,1]")
    if float(args.actor_max_log_interval_width) <= 0.0:
        raise ValueError("--actor-max-log-interval-width must be positive")
    if not (0.0 <= float(args.min_success_fraction) <= 1.0):
        raise ValueError("--min-success-fraction must be in [0,1]")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.set_device(device)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    requested_total = len(scenes) * len(trunks)
    if args.max_trunks > 0:
        requested_total = min(requested_total, int(args.max_trunks))
    config = {
        "image_dir": str(args.image_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "scenes": scenes,
        "trunks": trunks,
        "requested_trunks": requested_total,
        "trunk_frames": TRUNK_FRAMES,
        "legacy_window_frames": LEGACY_WINDOW_FRAMES,
        "head_precision": "aggregator+camera under bf16 autocast; depth head forced fp32",
        "actor_classes": sorted(requested_actor_classes),
        "actor_enabled": not args.skip_actor,
        "min_depth_pixels_per_frame": args.min_depth_pixels_per_frame,
        "min_depth_pixels_per_trunk": args.min_depth_pixels_per_trunk,
        "min_depth_frames": args.min_depth_frames,
        "actor_min_pixels": args.actor_min_pixels,
        "actor_min_frames": args.actor_min_frames,
        "actor_edge_margin": args.actor_edge_margin,
        "actor_min_consensus_fraction": args.actor_min_consensus_fraction,
        "actor_max_log_interval_width": args.actor_max_log_interval_width,
        "min_success_fraction": args.min_success_fraction,
    }
    output_path = Path(args.output_json) if args.output_json else None

    def build_result(status: str) -> dict[str, Any]:
        current_summary = summarize_rows(rows)
        success_fraction = len(rows) / float(requested_total) if requested_total else 0.0
        current_summary["requested_trunks"] = requested_total
        current_summary["success_fraction"] = success_fraction
        current_summary["min_success_fraction"] = float(args.min_success_fraction)
        current_summary["coverage_ok"] = success_fraction >= float(args.min_success_fraction)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": status,
            "definitions": result_definitions(),
            "config": config,
            "summary": current_summary,
            "rows": rows,
            "errors": errors,
            "skip_counts": dict(sorted(skip_reasons.items())),
        }

    def checkpoint_progress() -> None:
        if output_path is not None:
            _write_json_atomic(output_path, build_result("running"))

    model = _load_model(args.checkpoint, device)
    attempted = 0
    stop = False
    for scene in scenes:
        scene_root = Path(args.image_dir) / scene
        if not scene_root.is_dir():
            skip_reasons["scene_missing"] += len(trunks)
            continue
        dataset = WaymoOpenDataset(
            image_dir=args.image_dir,
            scene_names=[scene],
            sequence_length=LEGACY_WINDOW_FRAMES,
            mode=1,
            views=1,
            start_idx=0,
            pretrain_patch_grid=(25, 37),
            pretrain_max_objects=0,
            pretrain_instance_cache_size=2,
            trunk_frames=TRUNK_FRAMES,
            return_full_dggt_context=True,
            load_dynamic_masks=False,
            binary_mask_channels=1,
            image_output_dtype="uint8",
        )
        for trunk in trunks:
            if args.max_trunks > 0 and attempted >= int(args.max_trunks):
                stop = True
                break
            attempted += 1
            try:
                row = evaluate_trunk(
                    model=model,
                    dataset=dataset,
                    image_dir=args.image_dir,
                    scene=scene,
                    trunk_index=trunk,
                    device=device,
                    requested_actor_classes=requested_actor_classes,
                    min_depth_pixels_per_frame=args.min_depth_pixels_per_frame,
                    min_depth_pixels_per_trunk=args.min_depth_pixels_per_trunk,
                    min_depth_frames=args.min_depth_frames,
                    actor_min_pixels=args.actor_min_pixels,
                    actor_min_frames=args.actor_min_frames,
                    actor_edge_margin=args.actor_edge_margin,
                    actor_min_consensus_fraction=args.actor_min_consensus_fraction,
                    actor_max_log_interval_width=args.actor_max_log_interval_width,
                    run_actor=not args.skip_actor,
                )
            except Exception as error:  # diagnostics must retain coverage failures
                if args.strict:
                    raise
                reason = type(error).__name__
                skip_reasons[reason] += 1
                errors.append(
                    {
                        "scene": scene,
                        "trunk": int(trunk),
                        "error_type": reason,
                        "message": str(error),
                    }
                )
                print(f"[skip] scene={scene} trunk={trunk}: {reason}: {error}", file=sys.stderr)
                checkpoint_progress()
                continue
            rows.append(row)
            checkpoint_progress()
            legacy = row["legacy_10f"]
            actor_text = "n/a" if row["s_actor"] is None else f"{float(row['s_actor']):.5f}"
            print(
                f"scene={scene} trunk={trunk} motion10={float(legacy['ego_motion_m']):6.2f}m "
                f"stationary={str(bool(legacy['stationary'])):<5} "
                f"s_cam29={row['s_cam'] if row['s_cam'] is not None else 'n/a'} "
                f"s_depth29={row['s_depth'] if row['s_depth'] is not None else 'n/a'} "
                f"s_actor29={actor_text}",
                flush=True,
            )
        if stop:
            break

    result = build_result("complete")
    summary = result["summary"]

    print("\n=== gauge GT summary ===")
    print(
        f"trunks={summary['n_trunks']}  legacy stationary={summary['legacy_stationary_count']} "
        f"moving={summary['legacy_moving_count']}"
    )
    _print_pair("legacy s_cam / s_depth", summary["legacy_cam_over_depth"])
    _print_pair("full29 s_cam / s_depth", summary["full_cam_over_depth"])
    _print_pair("actor / depth", summary["actor_over_depth"])
    _print_pair("actor / camera", summary["actor_over_camera"])
    if errors:
        print(f"coverage errors={len(errors)}  reasons={dict(skip_reasons)}")

    if output_path is not None:
        _write_json_atomic(output_path, result)
        print(f"wrote {output_path}")
    if not rows:
        return 2
    return 0 if bool(summary["coverage_ok"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
