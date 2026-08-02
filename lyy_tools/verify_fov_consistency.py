"""Diagnose which intrinsics are consistent with frozen DGGT geometry.

This is the Phase-0 D1/D3 diagnostic from
``docs/metric_scale_camera_redesign_plan.md``.  Every sample is one complete
29-frame caption trunk.  The frozen aggregator is run once, then CameraHead,
DepthHead, PointHead, and GaussianHead are evaluated from the same tokens.

The decisive render is deliberately *primitive-level leave-one-frame-out*.
For a target frame ``t``, all Gaussians whose means came from frame ``t`` are
removed.  A candidate intrinsic matrix is used both to unproject source-frame
depth and to render the held-out target view.  This avoids two invalid
comparisons:

* unprojecting and rendering the same frame with the same K makes K cancel;
* building means with predicted K and only changing rasterizer K mechanically
  favours predicted K.

This is not an encoder-level held-out experiment: the 29-frame aggregator has
still seen every RGB frame, including the render target.  The diagnostic asks
whether its decoded *geometry* is cross-view self-consistent; it does not claim
novel-view generalization from target-masked input.

The module keeps statistics and branch selection as pure functions so that
the CPU unit tests do not load a 5 GB checkpoint or import gsplat.

D3 reuses the exact D1 leave-one-out primitives and the realizable
``mean(log(tan(FOV/2)))`` trunk K.  It changes only the target view matrix:
the reference arm uses the native frozen-DGGT camera, while the replacement
arm reanchors the metric Waymo c2w trajectory and scales only its translation
with the 29-frame LiDAR teacher ruler.  Thus K, source depth, Gaussian
attributes, source support, and world-space means are identical across arms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_IMAGE_DIR = "/data/lyy_dataset/waymo_processed_dggt/training"
DEFAULT_CHECKPOINT = "/data/lyy_dataset/model/dggt/model_latest_waymo.pt"
DEFAULT_OUTPUT = "runs/verify_fov_consistency/results.json"
TRUNK_FRAMES = 29
D3_FOCUS_SCENES = ("312", "314", "325")
D3_DEFAULT_MAX_LOSS_DB = 0.3
CRITICAL_CHECKPOINT_PREFIXES = (
    "aggregator.",
    "camera_head.",
    "depth_head.",
    "point_head.",
    "gs_head.",
)


def parse_index_spec(spec: str, *, zero_pad: int = 0) -> tuple[str, ...]:
    """Parse ``"0,2,5-7"`` into an ordered, duplicate-free tuple."""

    values: list[int] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-", 1)
            if len(pieces) != 2 or not pieces[0].strip() or not pieces[1].strip():
                raise ValueError(f"invalid integer range {part!r}")
            start, end = int(pieces[0]), int(pieces[1])
            if end < start:
                raise ValueError(f"descending integer range {part!r} is not supported")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("index specification is empty")
    unique = tuple(dict.fromkeys(values))
    if any(value < 0 for value in unique):
        raise ValueError(f"indices must be non-negative, got {unique}")
    if int(zero_pad) > 0:
        return tuple(f"{value:0{int(zero_pad)}d}" for value in unique)
    return tuple(str(value) for value in unique)


def scale_intrinsics(intrinsics: torch.Tensor, stride: int) -> torch.Tensor:
    """Scale pinhole intrinsics for ``image[:, ::stride, ::stride]``."""

    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    value = torch.as_tensor(intrinsics)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must end in [3,3], got {tuple(value.shape)}")
    if stride == 1:
        return value.clone()
    out = value.clone()
    out[..., 0, 0] /= float(stride)
    out[..., 1, 1] /= float(stride)
    out[..., 0, 2] /= float(stride)
    out[..., 1, 2] /= float(stride)
    return out


def stride_sample_images_nhwc(images: torch.Tensor, stride: int) -> torch.Tensor:
    """Sample RGB at exactly the pixel lattice represented by ``K / stride``."""

    value = torch.as_tensor(images)
    if value.ndim != 4 or int(value.shape[1]) != 3:
        raise ValueError(f"images must be [S,3,H,W], got {tuple(value.shape)}")
    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return value[:, :, ::stride, ::stride].permute(0, 2, 3, 1).contiguous()


def critical_missing_checkpoint_keys(missing_keys: Iterable[str]) -> list[str]:
    """Select missing state keys that make the D1 result invalid."""

    return [
        str(key)
        for key in missing_keys
        if str(key).startswith(CRITICAL_CHECKPOINT_PREFIXES)
    ]


def torch_unproject_depth(
    depth: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Unproject OpenCV z-depth to world coordinates, all in torch."""

    if depth.ndim != 3:
        raise ValueError(f"depth must be [S,H,W], got {tuple(depth.shape)}")
    if world_to_camera.shape != (depth.shape[0], 4, 4):
        raise ValueError(
            "world_to_camera must be [S,4,4] aligned with depth, got "
            f"{tuple(world_to_camera.shape)}"
        )
    if intrinsics.shape != (depth.shape[0], 3, 3):
        raise ValueError(
            f"intrinsics must be [S,3,3], got {tuple(intrinsics.shape)}"
        )
    seq_len, height, width = (int(v) for v in depth.shape)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0].to(depth).clamp_min(1.0e-6).view(seq_len, 1, 1)
    fy = intrinsics[:, 1, 1].to(depth).clamp_min(1.0e-6).view(seq_len, 1, 1)
    cx = intrinsics[:, 0, 2].to(depth).view(seq_len, 1, 1)
    cy = intrinsics[:, 1, 2].to(depth).view(seq_len, 1, 1)
    z = depth.float()
    x = (xx.view(1, height, width) - cx) * z / fx
    y = (yy.view(1, height, width) - cy) * z / fy
    camera = torch.stack((x, y, z), dim=-1)
    rotation = world_to_camera[:, :3, :3].to(camera)
    translation = world_to_camera[:, :3, 3].to(camera)
    return torch.einsum(
        "sij,shwj->shwi",
        rotation.transpose(-1, -2),
        camera - translation[:, None, None, :],
    )


def world_to_camera_points(
    world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> torch.Tensor:
    if world_points.ndim != 4 or int(world_points.shape[-1]) != 3:
        raise ValueError(
            f"world_points must be [S,H,W,3], got {tuple(world_points.shape)}"
        )
    if world_to_camera.shape != (world_points.shape[0], 4, 4):
        raise ValueError(
            f"world_to_camera shape {tuple(world_to_camera.shape)} is not aligned"
        )
    rotation = world_to_camera[:, :3, :3].to(world_points)
    translation = world_to_camera[:, :3, 3].to(world_points)
    return (
        torch.einsum("sij,shwj->shwi", rotation, world_points)
        + translation[:, None, None, :]
    )


def _as_scalar_map(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 4 and int(tensor.shape[-1]) == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be [S,H,W] or [S,H,W,1], got {tuple(tensor.shape)}")
    return tensor


def _finite_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        raise ValueError("cannot compute a quantile from no finite values")
    return torch.quantile(finite.float(), float(q))


def point_geometry_metrics(
    *,
    world_points: torch.Tensor,
    point_confidence: torch.Tensor,
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    static_mask: torch.Tensor,
    confidence_quantile: float = 0.5,
    pixel_scale: float = 1.0,
) -> dict[str, float | int]:
    """Compare candidate-K depth unprojection with independent PointHead XYZ."""

    points = torch.as_tensor(world_points).float()
    if points.ndim != 4 or int(points.shape[-1]) != 3:
        raise ValueError(f"world_points must be [S,H,W,3], got {tuple(points.shape)}")
    depth_map = _as_scalar_map(depth, "depth").float()
    point_conf = _as_scalar_map(point_confidence, "point_confidence").float()
    depth_conf = _as_scalar_map(depth_confidence, "depth_confidence").float()
    mask = torch.as_tensor(static_mask, dtype=torch.bool, device=points.device)
    expected = tuple(points.shape[:-1])
    for name, value in (
        ("depth", depth_map),
        ("point_confidence", point_conf),
        ("depth_confidence", depth_conf),
        ("static_mask", mask),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {expected}")
    q = float(confidence_quantile)
    if not 0.0 <= q < 1.0:
        raise ValueError(f"confidence_quantile must be in [0,1), got {q}")

    w2c = torch.as_tensor(world_to_camera, device=points.device, dtype=points.dtype)
    candidate_k = torch.as_tensor(intrinsics, device=points.device, dtype=points.dtype)
    point_camera = world_to_camera_points(points, w2c)
    valid = (
        mask
        & torch.isfinite(points).all(dim=-1)
        & torch.isfinite(point_camera).all(dim=-1)
        & torch.isfinite(depth_map)
        & torch.isfinite(point_conf)
        & torch.isfinite(depth_conf)
        & (depth_map > 1.0e-6)
        # PointHead is an independent diagnostic head; do not assume that its
        # camera-space z sign matches DepthHead's OpenCV z convention.  The
        # relative XYZ and ray-angle metrics only require a nonzero vector,
        # while reprojection only requires z to be away from zero.
        & (point_camera.norm(dim=-1) > 1.0e-6)
        & (point_camera[..., 2].abs() > 1.0e-6)
    )
    n_base = int(valid.sum().item())
    if n_base == 0:
        raise ValueError("point/depth geometry comparison has no valid static pixels")
    if q > 0.0:
        point_threshold = _finite_quantile(point_conf[valid], q)
        depth_threshold = _finite_quantile(depth_conf[valid], q)
        valid = valid & (point_conf >= point_threshold) & (depth_conf >= depth_threshold)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        raise ValueError("confidence filtering removed every geometry pixel")

    unprojected = torch_unproject_depth(depth_map, w2c, candidate_k)
    unprojected_camera = world_to_camera_points(unprojected, w2c)
    point_distance = (unprojected - points).norm(dim=-1)
    point_range = point_camera.norm(dim=-1).clamp_min(1.0e-6)
    relative_l2 = point_distance / point_range

    point_ray = F.normalize(point_camera, dim=-1, eps=1.0e-8)
    depth_ray = F.normalize(unprojected_camera, dim=-1, eps=1.0e-8)
    cosine = (point_ray * depth_ray).sum(dim=-1).clamp(-1.0, 1.0)
    angular_deg = torch.rad2deg(torch.acos(cosine))

    seq_len, height, width = expected
    yy, xx = torch.meshgrid(
        torch.arange(height, device=points.device, dtype=points.dtype),
        torch.arange(width, device=points.device, dtype=points.dtype),
        indexing="ij",
    )
    z_raw = point_camera[..., 2]
    z = torch.where(
        z_raw.abs() > 1.0e-6,
        z_raw,
        torch.where(z_raw >= 0.0, z_raw.new_tensor(1.0e-6), z_raw.new_tensor(-1.0e-6)),
    )
    projected_u = (
        candidate_k[:, 0, 0].view(seq_len, 1, 1) * point_camera[..., 0] / z
        + candidate_k[:, 0, 2].view(seq_len, 1, 1)
    )
    projected_v = (
        candidate_k[:, 1, 1].view(seq_len, 1, 1) * point_camera[..., 1] / z
        + candidate_k[:, 1, 2].view(seq_len, 1, 1)
    )
    reprojection_px = torch.sqrt(
        (projected_u - xx.view(1, height, width)).square()
        + (projected_v - yy.view(1, height, width)).square()
    ) * float(pixel_scale)

    def stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
        selected = value[valid].float()
        return {
            f"{prefix}_mean": float(selected.mean().item()),
            f"{prefix}_median": float(selected.median().item()),
            f"{prefix}_p90": float(torch.quantile(selected, 0.9).item()),
        }

    result: dict[str, float | int] = {
        "n_base_valid": n_base,
        "n_valid": n_valid,
        "confidence_quantile": q,
    }
    result.update(stats(relative_l2, "relative_l2"))
    result.update(stats(angular_deg, "angular_deg"))
    result.update(stats(reprojection_px, "reprojection_px"))
    return result


def assess_point_head_coordinate_compatibility(
    geometry_by_candidate: dict[str, dict[str, float | int]],
    image_hw: tuple[int, int],
    *,
    max_reprojection_fraction_of_diagonal: float = 0.1,
    max_angular_deg: float = 30.0,
) -> dict[str, Any]:
    """Say whether PointHead and CameraHead appear to share a usable convention.

    PointHead is only a supporting D1 ruler.  Its output can use a different
    axis convention (or be unusable in this checkpoint), in which case even
    the best candidate K cannot reproject the per-pixel XYZ near its source
    pixel.  That failure must be reported instead of being interpreted as
    evidence for either FOV branch.
    """

    height, width = (int(value) for value in image_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"image_hw must be positive, got {image_hw}")
    if not geometry_by_candidate:
        raise ValueError("geometry_by_candidate must not be empty")
    reprojection_limit = float(max_reprojection_fraction_of_diagonal) * math.hypot(
        height, width
    )
    angular_limit = float(max_angular_deg)
    if not math.isfinite(reprojection_limit) or reprojection_limit <= 0.0:
        raise ValueError("reprojection compatibility threshold must be positive")
    if not math.isfinite(angular_limit) or angular_limit <= 0.0:
        raise ValueError("angular compatibility threshold must be positive")

    candidate_status: dict[str, dict[str, float | bool]] = {}
    for name, metrics in geometry_by_candidate.items():
        reprojection = float(metrics["reprojection_px_median"])
        angular = float(metrics["angular_deg_median"])
        compatible = (
            math.isfinite(reprojection)
            and math.isfinite(angular)
            and reprojection <= reprojection_limit
            and angular <= angular_limit
        )
        candidate_status[str(name)] = {
            "reprojection_px_median": reprojection,
            "angular_deg_median": angular,
            "compatible": bool(compatible),
        }
    compatible_names = [
        name for name, status in candidate_status.items() if bool(status["compatible"])
    ]
    best_name = min(
        candidate_status,
        key=lambda name: float(candidate_status[name]["reprojection_px_median"]),
    )
    compatible = bool(compatible_names)
    return {
        "status": "compatible" if compatible else "coordinate_incompatible",
        "reason": (
            "at least one K reprojects PointHead XYZ near its source pixels"
            if compatible
            else "no K reprojects PointHead XYZ near its source pixels; metrics are diagnostic only"
        ),
        "used_for_branch_decision": False,
        "best_candidate_by_reprojection": best_name,
        "compatible_candidates": compatible_names,
        "max_reprojection_px_median": reprojection_limit,
        "max_angular_deg_median": angular_limit,
        "candidates": candidate_status,
    }


def summarize_fov_trunk(pose_encoding: torch.Tensor) -> dict[str, Any]:
    """Return 29-frame DGGT FOV statistics in external ``[x,y]`` order."""

    pose = torch.as_tensor(pose_encoding).float()
    if pose.ndim == 3:
        if int(pose.shape[0]) != 1:
            raise ValueError(f"expected one trunk, got pose shape {tuple(pose.shape)}")
        pose = pose[0]
    if pose.ndim != 2 or int(pose.shape[-1]) != 9:
        raise ValueError(f"pose_encoding must be [S,9] or [1,S,9], got {tuple(pose.shape)}")
    if int(pose.shape[0]) != TRUNK_FRAMES:
        raise ValueError(
            f"FOV diagnostic requires {TRUNK_FRAMES} frames, got {int(pose.shape[0])}"
        )
    # DGGT pose is [translation, quaternion, FOVy, FOVx].
    fov_xy_deg = torch.rad2deg(torch.stack((pose[:, 8], pose[:, 7]), dim=-1))
    if not bool(torch.isfinite(fov_xy_deg).all()):
        raise ValueError("predicted FOV contains non-finite values")
    mean = fov_xy_deg.mean(dim=0)
    std = fov_xy_deg.std(dim=0, unbiased=False)
    ranges = fov_xy_deg.amax(dim=0) - fov_xy_deg.amin(dim=0)
    mae = (fov_xy_deg - mean).abs().mean(dim=0)
    return {
        "mean_xy_deg": [float(v) for v in mean.tolist()],
        "std_xy_deg": [float(v) for v in std.tolist()],
        "range_xy_deg": [float(v) for v in ranges.tolist()],
        "mae_from_trunk_mean_xy_deg": [float(v) for v in mae.tolist()],
        "per_frame_xy_deg": [[float(v) for v in row] for row in fov_xy_deg.tolist()],
    }


def pose_with_trunk_mean_logtan_fov(pose_encoding: torch.Tensor) -> torch.Tensor:
    """Replace every frame FOV by the scene-gauge trunk representation.

    Phase 2 stores ``mean(log(tan(FOV / 2)))`` in its single scene-global
    token.  Averaging focal length, FOV degrees, or FOV radians would therefore
    test a different camera than the one the redesigned architecture can emit.
    DGGT pose channels are ``[..., FOVy, FOVx]`` and stay in that order here.
    """

    pose = torch.as_tensor(pose_encoding).float()
    if pose.ndim != 2 or tuple(pose.shape) != (TRUNK_FRAMES, 9):
        raise ValueError(
            f"pose_encoding must be [{TRUNK_FRAMES},9], got {tuple(pose.shape)}"
        )
    fov_yx = pose[:, 7:9]
    if not bool(torch.isfinite(fov_yx).all()):
        raise ValueError("predicted FOV contains non-finite values")
    if bool(((fov_yx <= 0.0) | (fov_yx >= math.pi)).any()):
        raise ValueError("predicted FOV must lie in (0, pi)")
    mean_log_tan_half = torch.log(torch.tan(0.5 * fov_yx)).mean(dim=0)
    trunk_fov_yx = 2.0 * torch.atan(torch.exp(mean_log_tan_half))
    result = pose.clone()
    result[:, 7:9] = trunk_fov_yx
    return result


def leave_one_out_source_mask(static_mask: torch.Tensor, target_index: int) -> torch.Tensor:
    """Return source support with the target frame removed completely."""

    mask = torch.as_tensor(static_mask, dtype=torch.bool)
    if mask.ndim != 3:
        raise ValueError(f"static_mask must be [S,H,W], got {tuple(mask.shape)}")
    target_index = int(target_index)
    if not 0 <= target_index < int(mask.shape[0]):
        raise IndexError(
            f"target_index {target_index} is outside [0,{int(mask.shape[0])})"
        )
    result = mask.clone()
    result[target_index] = False
    return result


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> dict[str, float | int]:
    """PSNR over a fixed spatial mask; RGB channels receive equal weight."""

    pred = torch.as_tensor(prediction).float()
    truth = torch.as_tensor(target, device=pred.device).float()
    if pred.shape != truth.shape or pred.ndim != 3 or int(pred.shape[-1]) != 3:
        raise ValueError(
            "prediction and target must match [H,W,3], got "
            f"{tuple(pred.shape)} and {tuple(truth.shape)}"
        )
    valid = torch.as_tensor(mask, device=pred.device, dtype=torch.bool)
    if valid.shape != pred.shape[:2]:
        raise ValueError(f"mask shape {tuple(valid.shape)} != {tuple(pred.shape[:2])}")
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("masked PSNR has no valid pixels")
    squared = (pred - truth).square()
    mse = squared[valid].mean()
    max_value = float(data_range)
    if not math.isfinite(max_value) or max_value <= 0.0:
        raise ValueError(f"data_range must be finite and positive, got {data_range}")
    psnr = 10.0 * torch.log10(
        mse.new_tensor(max_value * max_value) / mse.clamp_min(1.0e-12)
    )
    return {
        "mse": float(mse.item()),
        "psnr_db": float(psnr.item()),
        "n_valid_pixels": count,
    }


def shared_alpha_support_scores(
    first_artifact: dict[str, torch.Tensor],
    second_artifact: dict[str, torch.Tensor],
    *,
    alpha_threshold: float = 0.05,
    first_name: str = "predicted",
    second_name: str = "waymo",
) -> dict[str, Any]:
    """Paired PSNR on the candidates' symmetric common rendered support."""

    threshold = float(alpha_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"alpha_threshold must be in [0,1], got {threshold}")
    first_name = str(first_name)
    second_name = str(second_name)
    if not first_name or not second_name or first_name == second_name:
        raise ValueError("candidate names must be non-empty and distinct")
    required = ("render", "alpha", "target", "eval_mask")
    for name, artifact in ((first_name, first_artifact), (second_name, second_artifact)):
        missing = [key for key in required if key not in artifact]
        if missing:
            raise KeyError(f"{name} artifact is missing {missing}")
    first_eval = torch.as_tensor(first_artifact["eval_mask"], dtype=torch.bool)
    second_eval = torch.as_tensor(second_artifact["eval_mask"], dtype=torch.bool)
    if first_eval.shape != second_eval.shape or not torch.equal(first_eval, second_eval):
        raise ValueError("candidate artifacts must use the identical fixed eval mask")
    first_target = torch.as_tensor(first_artifact["target"]).float()
    second_target = torch.as_tensor(second_artifact["target"]).float()
    if first_target.shape != second_target.shape or not torch.allclose(
        first_target, second_target, atol=0.0, rtol=0.0
    ):
        raise ValueError("candidate artifacts must use the identical RGB target")
    first_alpha = torch.as_tensor(first_artifact["alpha"]).float()
    second_alpha = torch.as_tensor(second_artifact["alpha"]).float()
    if first_alpha.shape != first_eval.shape or second_alpha.shape != first_eval.shape:
        raise ValueError("candidate alpha maps must align with eval_mask")
    shared = first_eval & (first_alpha > threshold) & (second_alpha > threshold)
    fixed_count = int(first_eval.sum().item())
    shared_count = int(shared.sum().item())
    if fixed_count == 0 or shared_count == 0:
        raise ValueError("shared-alpha PSNR has no valid common support")
    first_score = masked_psnr(
        torch.as_tensor(first_artifact["render"]), first_target, shared
    )
    second_score = masked_psnr(
        torch.as_tensor(second_artifact["render"]), first_target, shared
    )
    delta_key = f"delta_{first_name}_minus_{second_name}_db"
    return {
        "alpha_threshold": threshold,
        "n_fixed_static_pixels": fixed_count,
        "n_shared_alpha_pixels": shared_count,
        "shared_fraction_of_fixed_static": float(shared_count / fixed_count),
        first_name: first_score,
        second_name: second_score,
        delta_key: float(first_score["psnr_db"]) - float(second_score["psnr_db"]),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Deterministic non-parametric CI for a trunk-level paired mean."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    samples = int(samples)
    if samples <= 0:
        raise ValueError(f"bootstrap samples must be positive, got {samples}")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence)
    low, high = np.quantile(means, (alpha, 1.0 - alpha))
    return float(low), float(high)


def scene_cluster_means(
    values: Sequence[float],
    scene_ids: Sequence[str | int],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Average trunks within scene before treating scenes as independent."""

    array = np.asarray(list(values), dtype=np.float64)
    labels = tuple(str(value) for value in scene_ids)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cluster values must be a non-empty finite vector")
    if len(labels) != int(array.size):
        raise ValueError(
            f"scene_ids length {len(labels)} != values length {int(array.size)}"
        )
    grouped: dict[str, list[float]] = {}
    for label, value in zip(labels, array.tolist()):
        grouped.setdefault(label, []).append(float(value))
    ordered_scenes = tuple(grouped)
    means = np.asarray(
        [np.mean(grouped[scene]) for scene in ordered_scenes],
        dtype=np.float64,
    )
    return means, ordered_scenes


def decide_fov_branch(
    psnr_delta_pred_minus_waymo_db: Sequence[float],
    *,
    scene_ids: Sequence[str | int] | None = None,
    min_effect_db: float = 0.2,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Three-state decision using scene-clustered paired PSNR differences."""

    values = np.asarray(list(psnr_delta_pred_minus_waymo_db), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("PSNR deltas must be a non-empty finite vector")
    effect = float(min_effect_db)
    if not math.isfinite(effect) or effect < 0.0:
        raise ValueError(f"min_effect_db must be finite and non-negative, got {effect}")
    if scene_ids is None:
        independent_values = values
        scene_labels = tuple(str(index) for index in range(int(values.size)))
        resampling_unit = "input_row"
    else:
        independent_values, scene_labels = scene_cluster_means(values, scene_ids)
        resampling_unit = "scene_mean_after_averaging_trunks"
    ci_low, ci_high = bootstrap_mean_ci(
        independent_values,
        samples=int(bootstrap_samples),
        confidence=0.95,
        seed=int(seed),
    )
    mean = float(independent_values.mean())
    clear_predicted = mean >= effect and ci_low > 0.0
    clear_waymo = mean <= -effect and ci_high < 0.0
    equivalent = ci_low >= -effect and ci_high <= effect
    if clear_predicted:
        branch = "A"
        reason = "predicted K has a clear practically relevant gain"
    elif clear_waymo:
        branch = "B"
        reason = "Waymo K has a clear practically relevant gain"
    elif equivalent:
        branch = "B"
        reason = "the entire 95% CI is inside the practical-equivalence band"
    else:
        branch = "INCONCLUSIVE"
        reason = "the CI is too wide or straddles a practically relevant boundary"
    return {
        "branch": branch,
        "rule": (
            "A for a clear predicted-K gain; B for a clear Waymo-K gain or a 95% CI "
            "fully inside +/-min_effect_db; otherwise INCONCLUSIVE"
        ),
        "reason": reason,
        "resampling_unit": resampling_unit,
        "n_trunks": int(values.size),
        "n_scenes": int(independent_values.size),
        "scene_ids": list(scene_labels),
        "mean_delta_db": mean,
        "median_scene_mean_delta_db": float(np.median(independent_values)),
        "fraction_scene_means_pred_better": float(np.mean(independent_values > 0.0)),
        "ci95_mean_delta_db": [ci_low, ci_high],
        "min_effect_db": effect,
    }


def decide_scene_cluster_noninferiority(
    candidate_minus_reference_db: Sequence[float],
    *,
    scene_ids: Sequence[str | int],
    margin_db: float = 0.2,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Test whether a scene-global K loses less than ``margin_db`` PSNR.

    The candidate is the realizable trunk-mean FOV and the reference is the
    native per-frame CameraHead FOV.  A significantly better trunk-mean K is
    allowed: this is intentionally a one-sided non-inferiority test.
    """

    values = np.asarray(list(candidate_minus_reference_db), dtype=np.float64)
    margin = float(margin_db)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("non-inferiority deltas must be a non-empty finite vector")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"margin_db must be finite and non-negative, got {margin}")
    scene_values, labels = scene_cluster_means(values, scene_ids)
    ci_low, ci_high = bootstrap_mean_ci(
        scene_values,
        samples=int(bootstrap_samples),
        confidence=0.95,
        seed=int(seed),
    )
    passed = ci_low > -margin
    return {
        "passed": bool(passed),
        "rule": "pass when the scene-cluster bootstrap CI lower bound is > -margin_db",
        "reason": (
            "trunk-mean K is non-inferior to native per-frame K"
            if passed
            else "trunk-mean K may lose more than the allowed PSNR margin"
        ),
        "resampling_unit": "scene_mean_after_averaging_trunks",
        "n_trunks": int(values.size),
        "n_scenes": int(scene_values.size),
        "scene_ids": list(labels),
        "mean_delta_db": float(scene_values.mean()),
        "ci95_mean_delta_db": [ci_low, ci_high],
        "margin_db": margin,
    }


def decide_metric_camera_render_space(
    native_minus_metric_psnr_db: Sequence[float],
    *,
    scene_ids: Sequence[str | int],
    max_loss_db: float = D3_DEFAULT_MAX_LOSS_DB,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Choose the D3 render camera from a scene-clustered PSNR-loss CI.

    Each input is ``PSNR(native DGGT camera) - PSNR(metric-converted
    camera)`` for one trunk, so positive values are the cost of replacing the
    teacher camera.  The plan's strict gate is implemented on the full 95%
    confidence interval: metric is accepted only when its upper bound is
    below ``max_loss_db``; teacher is selected only when the lower bound is at
    or above the gate.  A CI crossing the gate is honestly inconclusive.
    """

    values = np.asarray(list(native_minus_metric_psnr_db), dtype=np.float64)
    threshold = float(max_loss_db)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("D3 PSNR losses must be a non-empty finite vector")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"max_loss_db must be finite and non-negative, got {threshold}")
    scene_values, labels = scene_cluster_means(values, scene_ids)
    ci_low, ci_high = bootstrap_mean_ci(
        scene_values,
        samples=int(bootstrap_samples),
        confidence=0.95,
        seed=int(seed),
    )
    if ci_high < threshold:
        render_camera_space = "metric"
        reason = "the entire 95% CI is below the maximum allowed replacement loss"
    elif ci_low >= threshold:
        render_camera_space = "teacher"
        reason = "the entire 95% CI is at or above the maximum allowed replacement loss"
    else:
        render_camera_space = "INCONCLUSIVE"
        reason = "the 95% CI crosses the metric-camera replacement-loss gate"
    return {
        "render_camera_space": render_camera_space,
        "rule": (
            "metric when CI upper < max_loss_db; teacher when CI lower >= "
            "max_loss_db; otherwise INCONCLUSIVE"
        ),
        "reason": reason,
        "loss_definition": "PSNR(native_DGGT_camera) - PSNR(metric_converted_camera)",
        "resampling_unit": "scene_mean_after_averaging_trunks",
        "n_trunks": int(values.size),
        "n_scenes": int(scene_values.size),
        "scene_ids": list(labels),
        "mean_loss_db": float(scene_values.mean()),
        "median_scene_mean_loss_db": float(np.median(scene_values)),
        "fraction_scene_means_metric_worse": float(np.mean(scene_values > 0.0)),
        "ci95_mean_loss_db": [ci_low, ci_high],
        "max_loss_db": threshold,
    }


def combine_metric_camera_render_decisions(
    fixed_static_mask: dict[str, Any],
    shared_alpha_support: dict[str, Any],
) -> dict[str, Any]:
    """Require both D3 support definitions to select the same camera space."""

    fixed = str(fixed_static_mask.get("render_camera_space"))
    shared = str(shared_alpha_support.get("render_camera_space"))
    if fixed == shared and fixed in ("metric", "teacher"):
        camera_space = fixed
        reason = f"fixed-static and shared-alpha D3 gates both select {fixed}"
    else:
        camera_space = "INCONCLUSIVE"
        reason = (
            "fixed-static and shared-alpha D3 gates do not give the same "
            f"conclusive result ({fixed} vs {shared})"
        )
    return {
        "render_camera_space": camera_space,
        "reason": reason,
        "fixed_static_mask": fixed_static_mask,
        "shared_alpha_support": shared_alpha_support,
    }


def metric_camera_scene_breakdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    focus_scenes: Sequence[str | int] = D3_FOCUS_SCENES,
) -> dict[str, Any]:
    """Aggregate D3 losses by scene and explicitly surface known outliers."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scene = str(row["scene"])
        grouped.setdefault(scene, []).append(row)
    records: list[dict[str, Any]] = []
    for scene in sorted(
        grouped,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        scene_rows = grouped[scene]
        fixed = np.asarray(
            [float(row["fixed_static_loss_db"]) for row in scene_rows],
            dtype=np.float64,
        )
        shared = np.asarray(
            [float(row["shared_alpha_loss_db"]) for row in scene_rows],
            dtype=np.float64,
        )
        if not np.isfinite(fixed).all() or not np.isfinite(shared).all():
            raise ValueError(f"scene {scene} contains non-finite D3 losses")
        records.append(
            {
                "scene": scene,
                "n_trunks": len(scene_rows),
                "trunks": sorted(int(row["trunk"]) for row in scene_rows),
                "mean_fixed_static_loss_db": float(fixed.mean()),
                "mean_shared_alpha_loss_db": float(shared.mean()),
                "max_fixed_static_loss_db": float(fixed.max()),
                "max_shared_alpha_loss_db": float(shared.max()),
            }
        )
    ranked = sorted(
        records,
        key=lambda record: float(record["mean_fixed_static_loss_db"]),
        reverse=True,
    )
    rank_by_scene = {str(record["scene"]): index for index, record in enumerate(ranked, 1)}
    by_scene = {str(record["scene"]): record for record in records}
    focus: dict[str, Any] = {}
    for raw_scene in focus_scenes:
        scene = str(raw_scene)
        record = by_scene.get(scene)
        focus[scene] = {
            "present": record is not None,
            "rank_by_mean_fixed_static_loss_desc": rank_by_scene.get(scene),
            "metrics": record,
        }
    return {
        "n_scenes": len(records),
        "by_scene": by_scene,
        "ranked_by_mean_fixed_static_loss_desc": ranked,
        "focus_scenes": focus,
    }


def combine_render_decisions(
    full_mask_decision: dict[str, Any],
    shared_alpha_decision: dict[str, Any],
) -> dict[str, Any]:
    """Require fixed-mask and shared-alpha PSNR to agree before choosing A/B."""

    full_branch = str(full_mask_decision.get("branch"))
    shared_branch = str(shared_alpha_decision.get("branch"))
    if full_branch == shared_branch and full_branch in ("A", "B"):
        branch = full_branch
        reason = f"full static mask and shared-alpha support both choose {branch}"
    else:
        branch = "INCONCLUSIVE"
        reason = (
            "full static mask and shared-alpha support do not give the same "
            f"conclusive branch ({full_branch} vs {shared_branch})"
        )
    return {
        "branch": branch,
        "reason": reason,
        "full_static_mask": full_mask_decision,
        "shared_alpha_support": shared_alpha_decision,
    }


def combine_architecture_decisions(
    framewise_decision: dict[str, Any],
    trunk_mean_decision: dict[str, Any],
    full_mask_noninferiority: dict[str, Any],
    shared_alpha_noninferiority: dict[str, Any],
) -> dict[str, Any]:
    """Choose from the K that the one-token gauge can actually represent.

    Native per-frame K is a directionality sanity check.  The decisive A
    evidence is the direct trunk-mean-vs-Waymo render; trunk-mean-vs-native
    non-inferiority is reported but cannot overrule that direct comparison.
    """

    framewise_branch = str(framewise_decision.get("branch"))
    trunk_mean_branch = str(trunk_mean_decision.get("branch"))
    if framewise_branch == trunk_mean_branch == "B":
        branch = "B"
        reason = "native and trunk-mean predicted K both choose Waymo K"
    elif framewise_branch == trunk_mean_branch == "A":
        branch = "A"
        reason = (
            "the realizable trunk-mean predicted K directly beats Waymo K and "
            "the native per-frame comparison agrees"
        )
    else:
        branch = "INCONCLUSIVE"
        reason = (
            "native and realizable trunk-mean comparisons do not give the same "
            "conclusive winner"
        )
    return {
        "branch": branch,
        "reason": reason,
        "framewise_predicted_vs_waymo": framewise_decision,
        "trunk_mean_predicted_vs_waymo": trunk_mean_decision,
        "trunk_mean_minus_framewise_noninferiority": {
            "used_for_branch_decision": False,
            "full_static_mask": full_mask_noninferiority,
            "shared_alpha_support": shared_alpha_noninferiority,
        },
    }


def _split_gaussian_map(
    gaussian_map: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        gaussian_map[..., :3][support].reshape(-1, 3),
        gaussian_map[..., 3][support].reshape(-1),
        gaussian_map[..., 4:7][support].reshape(-1, 3),
        gaussian_map[..., 7:11][support].reshape(-1, 4),
    )


@torch.inference_mode()
def render_leave_one_out_candidate(
    *,
    images: torch.Tensor,
    depth: torch.Tensor,
    gaussian_map: torch.Tensor,
    gaussian_confidence: torch.Tensor,
    static_mask: torch.Tensor,
    timestamps: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    render_world_to_camera: torch.Tensor | None = None,
    target_indices: Sequence[int],
    stride: int,
    device: torch.device,
    min_eval_pixels: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    """Render static source frames into held-out target views under one K.

    ``world_to_camera`` always defines the fixed world-space Gaussian means.
    When ``render_world_to_camera`` is supplied, only rasterizer view matrices
    change.  D3 uses this explicit split to guarantee that native and
    metric-converted camera arms share identical primitives and intrinsics.
    """

    try:
        from gsplat.rendering import rasterization
    except ImportError as exc:  # pragma: no cover - exercised in the real env
        raise RuntimeError(
            "verify_fov_consistency rendering requires gsplat in the dggt environment"
        ) from exc

    rgb = torch.as_tensor(images).float()
    if rgb.ndim != 4 or int(rgb.shape[1]) != 3:
        raise ValueError(f"images must be [S,3,H,W], got {tuple(rgb.shape)}")
    seq_len, _, height, width = rgb.shape
    if int(seq_len) != TRUNK_FRAMES:
        raise ValueError(f"render requires a {TRUNK_FRAMES}-frame trunk, got {seq_len}")
    depth_map = _as_scalar_map(depth, "depth").float()
    gs = torch.as_tensor(gaussian_map).float()
    gs_conf = _as_scalar_map(gaussian_confidence, "gaussian_confidence").float()
    support_full = torch.as_tensor(static_mask, dtype=torch.bool)
    expected_map = (seq_len, height, width)
    if tuple(depth_map.shape) != expected_map:
        raise ValueError(f"depth shape {tuple(depth_map.shape)} != {expected_map}")
    if tuple(gs.shape) != expected_map + (11,):
        raise ValueError(f"gaussian_map shape {tuple(gs.shape)} != {expected_map + (11,)}")
    if tuple(gs_conf.shape) != expected_map or tuple(support_full.shape) != expected_map:
        raise ValueError("gaussian confidence/static mask are not aligned with images")
    if world_to_camera.shape != (seq_len, 4, 4) or intrinsics.shape != (seq_len, 3, 3):
        raise ValueError("camera matrices are not aligned with the trunk")
    if render_world_to_camera is not None and tuple(render_world_to_camera.shape) != (
        seq_len,
        4,
        4,
    ):
        raise ValueError("render_world_to_camera is not aligned with the trunk")

    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"render stride must be positive, got {stride}")
    render_h = int(math.ceil(height / float(stride)))
    render_w = int(math.ceil(width / float(stride)))
    depth_low = depth_map[:, ::stride, ::stride].to(device=device, dtype=torch.float32)
    gs_low = gs[:, ::stride, ::stride].to(device=device, dtype=torch.float32)
    conf_low = gs_conf[:, ::stride, ::stride].to(device=device, dtype=torch.float32)
    support_low = support_full[:, ::stride, ::stride].to(device=device)
    if tuple(depth_low.shape[-2:]) != (render_h, render_w):
        raise RuntimeError(
            f"strided depth shape {tuple(depth_low.shape[-2:])} != {(render_h, render_w)}"
        )
    geometry_w2c = torch.as_tensor(
        world_to_camera, device=device, dtype=torch.float32
    )
    render_w2c = (
        geometry_w2c
        if render_world_to_camera is None
        else torch.as_tensor(
            render_world_to_camera, device=device, dtype=torch.float32
        )
    )
    candidate_k = scale_intrinsics(
        torch.as_tensor(intrinsics, device=device, dtype=torch.float32), stride
    )
    # The same candidate K is used here and in rasterization below.
    world_points = torch_unproject_depth(depth_low, geometry_w2c, candidate_k)
    finite_source = (
        torch.isfinite(world_points).all(dim=-1)
        & torch.isfinite(depth_low)
        & (depth_low > 1.0e-6)
    )
    # Use the exact same 0,stride,2*stride,... pixel lattice as depth, masks,
    # and K/stride.  Area interpolation uses different sample centres when H
    # or W is not divisible by stride (350/4 and 518/4 here), which would add a
    # small candidate-independent but avoidable PSNR alignment error.
    target_low = stride_sample_images_nhwc(
        rgb.to(device=device, dtype=torch.float32),
        stride,
    )
    time_values = torch.as_tensor(timestamps, device=device, dtype=torch.float32).reshape(-1)
    if int(time_values.numel()) != seq_len:
        raise ValueError(f"timestamps contain {time_values.numel()} values, expected {seq_len}")

    rows: list[dict[str, Any]] = []
    artifacts: dict[int, dict[str, torch.Tensor]] = {}
    log_point_one_tenth = math.log(0.1)
    for target_index_raw in target_indices:
        target_index = int(target_index_raw)
        source_support = leave_one_out_source_mask(
            support_low & finite_source,
            target_index,
        )
        source_count = int(source_support.sum().item())
        if source_count == 0:
            raise ValueError(f"target {target_index} has no leave-one-out source Gaussians")
        means = world_points[source_support].reshape(-1, 3)
        colors, opacities, scales, rotations = _split_gaussian_map(gs_low, source_support)
        source_frame = torch.nonzero(source_support, as_tuple=False)[:, 0]
        source_time = time_values[source_frame]
        lifespan = conf_low[source_support].reshape(-1)
        decay = torch.exp(
            log_point_one_tenth
            * (source_time - time_values[target_index]).square()
            / (lifespan.square() + 1.0e-6)
        )
        opacities = opacities * decay
        rendered, alpha, _ = rasterization(
            means=means.float(),
            quats=rotations.float(),
            scales=scales.float().clamp_min(1.0e-5),
            opacities=opacities.float().view(-1),
            colors=colors.float().clamp(0.0, 1.0),
            viewmats=render_w2c[target_index : target_index + 1],
            Ks=candidate_k[target_index : target_index + 1],
            width=render_w,
            height=render_h,
            render_mode="RGB",
        )
        # gsplat without a supplied background returns premultiplied RGB.  A
        # black background is therefore exactly rendered[..., :3]; applying
        # alpha a second time would be incorrect.
        prediction = rendered[0, ..., :3].float().clamp(0.0, 1.0)
        alpha_map = alpha[0, ..., 0].float().clamp(0.0, 1.0)
        eval_mask = support_low[target_index]
        if int(eval_mask.sum().item()) < int(min_eval_pixels):
            raise ValueError(
                f"target {target_index} has only {int(eval_mask.sum())} static eval pixels "
                f"(< {int(min_eval_pixels)})"
            )
        score = masked_psnr(prediction, target_low[target_index], eval_mask)
        score.update(
            {
                "target_index": target_index,
                "n_source_gaussians": source_count,
                "alpha_mean_on_eval": float(alpha_map[eval_mask].mean().item()),
                "alpha_gt_005_fraction_on_eval": float(
                    (alpha_map[eval_mask] > 0.05).float().mean().item()
                ),
            }
        )
        rows.append(score)
        artifacts[target_index] = {
            "render": prediction.detach().cpu(),
            "alpha": alpha_map.detach().cpu(),
            "target": target_low[target_index].detach().cpu(),
            "eval_mask": eval_mask.detach().cpu(),
        }
    return rows, artifacts


def _mean_metric(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"cannot average invalid metric {key!r}")
    return float(np.mean(values))


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    if not np.isfinite(array).all():
        raise ValueError("summary values must be finite")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically replace a JSON result so interrupted runs retain prior trunks."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _load_model(checkpoint_path: str, device: torch.device):
    from dggt.models.vggt import VGGT

    model = VGGT().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state must be a mapping, got {type(state).__name__}")
    state = {
        key[len("module.") :] if str(key).startswith("module.") else str(key): value
        for key, value in state.items()
    }
    incompat = model.load_state_dict(state, strict=False)
    critical_missing = critical_missing_checkpoint_keys(incompat.missing_keys)
    if critical_missing:
        preview = ", ".join(critical_missing[:20])
        raise RuntimeError(f"checkpoint is missing critical D1 keys: {preview}")
    if incompat.unexpected_keys:
        print(f"checkpoint unexpected keys: {len(incompat.unexpected_keys)}", flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del checkpoint, state
    return model


@torch.inference_mode()
def _run_frozen_heads(
    model,
    context_images: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = context_images
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5 or int(images.shape[0]) != 1 or int(images.shape[1]) != TRUNK_FRAMES:
        raise ValueError(
            f"context_images must be [1,{TRUNK_FRAMES},3,H,W], got {tuple(images.shape)}"
        )
    images = images.to(device=device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
    else:
        images = images.float()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        token_outputs = model.get_aggregator_token_outputs(images)
    patch_start_idx = int(token_outputs["patch_start_idx"])

    aggregated = token_outputs.pop("aggregated_tokens_list")
    aggregated_fp32 = [tokens.float() for tokens in aggregated]
    del aggregated
    with torch.autocast(device_type="cuda", enabled=False):
        pose = model.camera_head(aggregated_fp32)[-1].float()
        depth, depth_conf = model.depth_head(
            aggregated_fp32,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        points, point_conf = model.point_head(
            aggregated_fp32,
            images=images,
            patch_start_idx=patch_start_idx,
        )
    del aggregated_fp32

    image_tokens = token_outputs.pop("image_tokens_list")
    image_tokens_fp32 = [tokens.float() for tokens in image_tokens]
    del image_tokens
    with torch.autocast(device_type="cuda", enabled=False):
        gaussian_map, gaussian_conf = model.gs_head(
            image_tokens_fp32,
            images=images,
            patch_start_idx=patch_start_idx,
        )
    del image_tokens_fp32, token_outputs, images
    result = {
        "pose": pose[0].detach().cpu().float(),
        "depth": depth[0].detach().cpu().float(),
        "depth_conf": depth_conf[0].detach().cpu().float(),
        "points": points[0].detach().cpu().float(),
        "point_conf": point_conf[0].detach().cpu().float(),
        "gaussian_map": gaussian_map[0].detach().cpu().float(),
        "gaussian_conf": gaussian_conf[0].detach().cpu().float(),
    }
    del pose, depth, depth_conf, points, point_conf, gaussian_map, gaussian_conf
    torch.cuda.empty_cache()
    return result


def _camera_candidates(
    pose: torch.Tensor,
    raw_intrinsics: torch.Tensor,
    raw_image_hw: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from dggt.utils.factorized_asset_condition import (
        resize_crop_intrinsics_to_model_canvas,
    )
    from dggt.utils.pose_enc import pose_encoding_to_extri_intri

    height, width = image_hw
    extrinsics3, predicted_k_b = pose_encoding_to_extri_intri(
        pose.unsqueeze(0).float(),
        (height, width),
    )
    trunk_mean_pose = pose_with_trunk_mean_logtan_fov(pose)
    trunk_mean_extrinsics3, predicted_trunk_mean_k_b = pose_encoding_to_extri_intri(
        trunk_mean_pose.unsqueeze(0),
        (height, width),
    )
    if not torch.allclose(
        extrinsics3, trunk_mean_extrinsics3, atol=1.0e-6, rtol=1.0e-6
    ):
        raise RuntimeError("replacing FOV unexpectedly changed CameraHead extrinsics")
    bottom = extrinsics3.new_tensor((0.0, 0.0, 0.0, 1.0)).view(1, 1, 1, 4)
    world_to_camera = torch.cat(
        (extrinsics3, bottom.expand(1, TRUNK_FRAMES, -1, -1)),
        dim=-2,
    )[0]
    waymo_k, model_hw = resize_crop_intrinsics_to_model_canvas(
        torch.as_tensor(raw_intrinsics).float(),
        torch.as_tensor(raw_image_hw),
        target_width=int(width),
        patch_size=14,
    )
    if model_hw.reshape(-1, 2).shape[0] != 1:
        unique_hw = torch.unique(model_hw.reshape(-1, 2), dim=0)
    else:
        unique_hw = model_hw.reshape(-1, 2)
    if unique_hw.shape != (1, 2) or tuple(int(v) for v in unique_hw[0].tolist()) != (
        height,
        width,
    ):
        raise RuntimeError(
            f"Waymo model canvas {model_hw.tolist()} does not match DGGT {(height, width)}"
        )
    waymo_k = waymo_k.reshape(-1, 3, 3)
    if int(waymo_k.shape[0]) == 1:
        waymo_k = waymo_k.expand(TRUNK_FRAMES, -1, -1).clone()
    elif int(waymo_k.shape[0]) != TRUNK_FRAMES:
        raise ValueError(
            f"Waymo intrinsics have {int(waymo_k.shape[0])} rows, expected 1 or {TRUNK_FRAMES}"
        )
    return world_to_camera.float(), {
        "predicted": predicted_k_b[0].float(),
        "predicted_trunk_mean": predicted_trunk_mean_k_b[0].float(),
        "waymo": waymo_k.float(),
    }


def _waymo_fov_xy_deg(
    waymo_k: torch.Tensor,
    image_hw: tuple[int, int],
) -> list[float]:
    from dggt.utils.camera_condition import fov_from_intrinsics

    fov = fov_from_intrinsics(waymo_k[:1], image_hw).reshape(-1, 2)[0]
    return [float(value) for value in torch.rad2deg(fov).tolist()]


def _make_dataset(image_dir: str, scene: str, trunk: int):
    from datasets.dataset import WaymoOpenDataset

    return WaymoOpenDataset(
        image_dir=image_dir,
        scene_names=[scene],
        sequence_length=TRUNK_FRAMES,
        mode=1,
        views=1,
        start_idx=int(trunk) * TRUNK_FRAMES,
        pretrain_patch_grid=(25, 37),
        pretrain_max_objects=0,
        pretrain_instance_cache_size=0,
        trunk_frames=TRUNK_FRAMES,
        camera_anchor_window_probability=1.0,
        return_full_dggt_context=True,
        load_dynamic_masks=True,
        binary_mask_channels=1,
        image_output_dtype="uint8",
    )


def _prepare_static_mask(item: dict[str, Any]) -> tuple[torch.Tensor, bool]:
    sky = torch.as_tensor(item["masks"]).float()
    if sky.ndim != 4:
        raise ValueError(f"sky masks must be [S,C,H,W], got {tuple(sky.shape)}")
    sky_mask = sky.mean(dim=1) > 0.5
    has_dynamic = "dynamic_mask" in item
    if has_dynamic:
        dynamic = torch.as_tensor(item["dynamic_mask"]).float()
        if dynamic.ndim != 4:
            raise ValueError(
                f"dynamic masks must be [S,C,H,W], got {tuple(dynamic.shape)}"
            )
        dynamic_mask = dynamic.mean(dim=1) > 0.5
    else:
        dynamic_mask = torch.zeros_like(sky_mask)
    return (~sky_mask) & (~dynamic_mask), bool(has_dynamic)


def camera_centre_motion_metrics(camera_to_world: torch.Tensor) -> dict[str, float]:
    """Measure 29-frame baseline without accumulating stationary jitter."""

    c2w = torch.as_tensor(camera_to_world).float()
    if c2w.ndim == 4 and int(c2w.shape[1]) == 1:
        c2w = c2w[:, 0]
    if c2w.shape != (TRUNK_FRAMES, 4, 4):
        raise ValueError(f"Waymo c2w shape is {tuple(c2w.shape)}, expected [29,4,4]")
    centres = c2w[:, :3, 3]
    path_length = (centres[1:] - centres[:-1]).norm(dim=-1).sum()
    max_pairwise_span = torch.pdist(centres).max()
    return {
        "path_length_m": float(path_length.item()),
        "max_pairwise_span_m": float(max_pairwise_span.item()),
    }


def metric_c2w_to_dggt_with_teacher_gauge(
    camera_to_world_metric: torch.Tensor,
    anchor_to_world_metric: torch.Tensor,
    log_metric_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Reanchor Waymo c2w and convert metric translation to teacher units.

    ``log_metric_scale = log(metres / DGGT_unit) = log(1 / s_lidar)``.
    Therefore converting metres to DGGT teacher units divides translation by
    ``exp(log_metric_scale)`` (equivalently multiplies it by ``s_lidar``).
    Rotation is copied exactly; no Sim(3) rotation/translation fit is allowed
    in D3 because that would hide the camera-target replacement residual.
    """

    c2w = torch.as_tensor(camera_to_world_metric).float()
    if c2w.ndim == 4 and int(c2w.shape[1]) == 1:
        c2w = c2w[:, 0]
    if c2w.shape != (TRUNK_FRAMES, 4, 4):
        raise ValueError(
            f"metric camera_to_world must be [{TRUNK_FRAMES},4,4], got {tuple(c2w.shape)}"
        )
    anchor = torch.as_tensor(anchor_to_world_metric, device=c2w.device).float()
    if anchor.ndim == 3 and int(anchor.shape[0]) == 1:
        anchor = anchor[0]
    if anchor.shape != (4, 4):
        raise ValueError(f"metric anchor must be [4,4], got {tuple(anchor.shape)}")
    log_scale = torch.as_tensor(log_metric_scale, device=c2w.device, dtype=c2w.dtype)
    if log_scale.numel() != 1 or not bool(torch.isfinite(log_scale).all()):
        raise ValueError("log_metric_scale must be one finite scalar")
    metres_per_unit = torch.exp(log_scale.reshape(()))
    if not bool(torch.isfinite(metres_per_unit)) or float(metres_per_unit) <= 0.0:
        raise ValueError("exp(log_metric_scale) must be finite and positive")
    relative = torch.linalg.inv(anchor) @ c2w
    converted = relative.clone()
    converted[:, :3, 3] /= metres_per_unit
    if not bool(torch.isfinite(converted).all()):
        raise ValueError("metric-to-DGGT camera conversion produced non-finite values")
    identity = torch.eye(4, device=converted.device, dtype=converted.dtype)
    if not torch.allclose(relative[0], identity, atol=2.0e-4, rtol=0.0):
        error = float((relative[0] - identity).abs().max().item())
        raise ValueError(f"Waymo camera does not reanchor to identity (max error {error:.3e})")
    if not torch.allclose(
        converted[:, :3, :3], relative[:, :3, :3], atol=0.0, rtol=0.0
    ):
        raise RuntimeError("metric camera conversion unexpectedly changed rotation")
    return converted


def camera_to_world_to_world_to_camera(camera_to_world: torch.Tensor) -> torch.Tensor:
    """Invert a validated 29-frame homogeneous camera trajectory."""

    c2w = torch.as_tensor(camera_to_world).float()
    if c2w.shape != (TRUNK_FRAMES, 4, 4):
        raise ValueError(f"camera_to_world must be [{TRUNK_FRAMES},4,4]")
    if not bool(torch.isfinite(c2w).all()):
        raise ValueError("camera_to_world contains non-finite values")
    return torch.linalg.inv(c2w)


def _estimate_d3_lidar_teacher_gauge(
    *,
    image_dir: str,
    scene: str,
    trunk: int,
    dggt_depth: torch.Tensor,
    image_hw: tuple[int, int],
) -> dict[str, Any]:
    """Run the already-validated D2 29-frame LiDAR ruler for D3."""

    from lyy_tools.verify_gauge_gt import (
        _load_lidar_depth_trunk,
        estimate_depth_ruler,
    )

    frames = tuple(
        int(trunk) * TRUNK_FRAMES + local for local in range(TRUNK_FRAMES)
    )
    lidar_depth, missing = _load_lidar_depth_trunk(
        image_dir,
        str(scene),
        frames,
        image_hw,
    )
    if lidar_depth is None:
        return {
            "valid": False,
            "reason": "missing_lidar_depth",
            "missing_files": list(missing),
            "s_lidar": None,
            "log_metric_scale": None,
        }
    depth = torch.as_tensor(dggt_depth).detach().cpu().float().numpy()
    if depth.ndim == 4 and int(depth.shape[-1]) == 1:
        depth = depth[..., 0]
    ruler = estimate_depth_ruler(depth, lidar_depth)
    scale = ruler.get("scale")
    if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0.0:
        return {
            "valid": False,
            "reason": "D2_29f_lidar_ruler_invalid",
            "missing_files": [],
            "s_lidar": None,
            "log_metric_scale": None,
            "ruler": ruler,
        }
    s_lidar = float(scale)
    return {
        "valid": True,
        "reason": "valid_D2_29f_lidar_teacher_ruler",
        "missing_files": [],
        "s_lidar": s_lidar,
        "log_metric_scale": float(math.log(1.0 / s_lidar)),
        "metres_per_dggt_unit": float(1.0 / s_lidar),
        "n_valid_px": int(ruler["n_valid_px"]),
        "n_valid_frames": int(ruler["n_valid_frames"]),
        "n_candidate_frames": int(ruler["n_candidate_frames"]),
        "frame_cv": ruler["aggregate"]["cv_inlier"],
        "second_over_first_log_scale": ruler["second_over_first_log_scale"],
        "frame_max_over_min": ruler["frame_max_over_min"],
    }


def _process_trunk(
    *,
    model,
    image_dir: str,
    scene: str,
    trunk: int,
    device: torch.device,
    render_targets: Sequence[int],
    render_stride: int,
    geometry_stride: int,
    confidence_quantile: float,
    min_eval_pixels: int,
) -> dict[str, Any]:
    dataset = _make_dataset(image_dir, scene, trunk)
    if len(dataset) == 0:
        raise FileNotFoundError(f"scene {scene} is absent from {image_dir}")
    item = dataset[0]
    context = torch.as_tensor(item["dggt_context_images"])
    if int(context.shape[0]) != TRUNK_FRAMES:
        raise RuntimeError(
            f"scene {scene} trunk {trunk} returned {int(context.shape[0])} context frames"
        )
    images = context.float().div(255.0) if context.dtype == torch.uint8 else context.float()
    height, width = int(images.shape[-2]), int(images.shape[-1])
    static_mask, has_dynamic_mask = _prepare_static_mask(item)
    heads = _run_frozen_heads(model, context, device)
    pose = heads["pose"]
    world_to_camera, candidates = _camera_candidates(
        pose,
        torch.as_tensor(item["intrinsics"]),
        torch.as_tensor(item["raw_image_size_hw"]),
        (height, width),
    )

    geometry: dict[str, dict[str, float | int]] = {}
    geometry_stride = int(geometry_stride)
    geometry_slices = (slice(None), slice(None, None, geometry_stride), slice(None, None, geometry_stride))
    points_geom = heads["points"][geometry_slices]
    point_conf_geom = heads["point_conf"][geometry_slices]
    depth_geom = heads["depth"][geometry_slices]
    depth_conf_geom = heads["depth_conf"][geometry_slices]
    static_geom = static_mask[geometry_slices]
    for name, candidate in candidates.items():
        geometry[name] = point_geometry_metrics(
            world_points=points_geom,
            point_confidence=point_conf_geom,
            depth=depth_geom,
            depth_confidence=depth_conf_geom,
            world_to_camera=world_to_camera,
            intrinsics=scale_intrinsics(candidate, geometry_stride),
            static_mask=static_geom,
            confidence_quantile=confidence_quantile,
            pixel_scale=float(geometry_stride),
        )
    geometry_compatibility = assess_point_head_coordinate_compatibility(
        geometry, (height, width)
    )
    motion = camera_centre_motion_metrics(item["camera_to_world_corrected"])
    d3_gauge = _estimate_d3_lidar_teacher_gauge(
        image_dir=image_dir,
        scene=scene,
        trunk=trunk,
        dggt_depth=heads["depth"],
        image_hw=(height, width),
    )
    metric_render_w2c: torch.Tensor | None = None
    if bool(d3_gauge["valid"]):
        metric_c2w_dggt = metric_c2w_to_dggt_with_teacher_gauge(
            torch.as_tensor(item["camera_to_world_corrected"]),
            torch.as_tensor(item["camera_trajectory_anchor_to_world_corrected"]),
            float(d3_gauge["log_metric_scale"]),
        )
        metric_render_w2c = camera_to_world_to_world_to_camera(metric_c2w_dggt)

    render: dict[str, Any] = {}
    render_artifacts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for name, candidate in candidates.items():
        frame_rows, artifacts = render_leave_one_out_candidate(
            images=images,
            depth=heads["depth"],
            gaussian_map=heads["gaussian_map"],
            gaussian_confidence=heads["gaussian_conf"],
            static_mask=static_mask,
            timestamps=torch.as_tensor(item["timestamps"]),
            world_to_camera=world_to_camera,
            intrinsics=candidate,
            target_indices=render_targets,
            stride=render_stride,
            device=device,
            min_eval_pixels=min_eval_pixels,
        )
        render[name] = {
            "per_target": frame_rows,
            "mean_psnr_db": _mean_metric(frame_rows, "psnr_db"),
            "mean_alpha_on_eval": _mean_metric(frame_rows, "alpha_mean_on_eval"),
            "mean_alpha_gt_005_fraction_on_eval": _mean_metric(
                frame_rows, "alpha_gt_005_fraction_on_eval"
            ),
        }
        render_artifacts[name] = artifacts
    d3: dict[str, Any] = {
        "valid": bool(d3_gauge["valid"]),
        "gauge": d3_gauge,
        "protocol": {
            "primitive_camera": "native_frozen_DGGT_CameraHead",
            "native_render_camera": "native_frozen_DGGT_CameraHead",
            "metric_render_camera": (
                "inv(Waymo_anchor_c2w) @ Waymo_c2w; translation divided by "
                "exp(log_metric_scale), rotation unchanged"
            ),
            "shared_intrinsics": "DGGT_gauge_trunk_mean_log_tan_half_FOV",
            "same_world_space_gaussian_primitives": True,
            "only_render_view_matrix_changes": True,
        },
    }
    if metric_render_w2c is not None:
        metric_rows, metric_artifacts = render_leave_one_out_candidate(
            images=images,
            depth=heads["depth"],
            gaussian_map=heads["gaussian_map"],
            gaussian_confidence=heads["gaussian_conf"],
            static_mask=static_mask,
            timestamps=torch.as_tensor(item["timestamps"]),
            # Fixed primitive frame and fixed gauge K across both D3 arms.
            world_to_camera=world_to_camera,
            render_world_to_camera=metric_render_w2c,
            intrinsics=candidates["predicted_trunk_mean"],
            target_indices=render_targets,
            stride=render_stride,
            device=device,
            min_eval_pixels=min_eval_pixels,
        )
        native_rows = render["predicted_trunk_mean"]["per_target"]
        native_artifacts = render_artifacts["predicted_trunk_mean"]
        shared_rows: list[dict[str, Any]] = []
        for target_index in render_targets:
            shared = shared_alpha_support_scores(
                native_artifacts[int(target_index)],
                metric_artifacts[int(target_index)],
                alpha_threshold=0.05,
                first_name="native",
                second_name="metric",
            )
            shared["target_index"] = int(target_index)
            shared_rows.append(shared)
        native_mean = _mean_metric(native_rows, "psnr_db")
        metric_mean = _mean_metric(metric_rows, "psnr_db")
        shared_native_mean = float(
            np.mean([row["native"]["psnr_db"] for row in shared_rows])
        )
        shared_metric_mean = float(
            np.mean([row["metric"]["psnr_db"] for row in shared_rows])
        )
        d3["render_leave_one_out_static"] = {
            "native_dggt_camera": {
                "per_target": native_rows,
                "mean_psnr_db": native_mean,
            },
            "metric_converted_camera": {
                "per_target": metric_rows,
                "mean_psnr_db": metric_mean,
            },
            "loss_native_minus_metric_db": float(native_mean - metric_mean),
            "shared_alpha_support": {
                "alpha_threshold": 0.05,
                "per_target": shared_rows,
                "mean_native_psnr_db": shared_native_mean,
                "mean_metric_psnr_db": shared_metric_mean,
                "loss_native_minus_metric_db": float(
                    shared_native_mean - shared_metric_mean
                ),
                "mean_shared_fraction_of_fixed_static": float(
                    np.mean(
                        [
                            row["shared_fraction_of_fixed_static"]
                            for row in shared_rows
                        ]
                    )
                ),
                "min_shared_fraction_of_fixed_static": float(
                    np.min(
                        [
                            row["shared_fraction_of_fixed_static"]
                            for row in shared_rows
                        ]
                    )
                ),
            },
        }
        del metric_artifacts
    pair_specs = (
        ("shared_alpha_support", "predicted", "waymo"),
        (
            "shared_alpha_support_trunk_mean_vs_waymo",
            "predicted_trunk_mean",
            "waymo",
        ),
        (
            "shared_alpha_support_trunk_mean_vs_predicted",
            "predicted_trunk_mean",
            "predicted",
        ),
    )
    for output_name, first_name, second_name in pair_specs:
        shared_rows: list[dict[str, Any]] = []
        delta_key = f"delta_{first_name}_minus_{second_name}_db"
        for target_index in render_targets:
            shared = shared_alpha_support_scores(
                render_artifacts[first_name][int(target_index)],
                render_artifacts[second_name][int(target_index)],
                alpha_threshold=0.05,
                first_name=first_name,
                second_name=second_name,
            )
            shared["target_index"] = int(target_index)
            shared_rows.append(shared)
        render[output_name] = {
            "candidate_names": [first_name, second_name],
            "alpha_threshold": 0.05,
            "per_target": shared_rows,
            f"mean_{first_name}_psnr_db": float(
                np.mean([row[first_name]["psnr_db"] for row in shared_rows])
            ),
            f"mean_{second_name}_psnr_db": float(
                np.mean([row[second_name]["psnr_db"] for row in shared_rows])
            ),
            "mean_shared_fraction_of_fixed_static": float(
                np.mean([row["shared_fraction_of_fixed_static"] for row in shared_rows])
            ),
            "min_shared_fraction_of_fixed_static": float(
                np.min([row["shared_fraction_of_fixed_static"] for row in shared_rows])
            ),
            delta_key: float(np.mean([row[delta_key] for row in shared_rows])),
        }
    render["minimum_shared_support_fraction_across_comparisons"] = float(
        min(
            render[output_name]["min_shared_fraction_of_fixed_static"]
            for output_name, _, _ in pair_specs
        )
    )
    # Keep tensors out of JSON after computing the symmetric shared-support
    # score.  Every candidate's own alpha coverage remains in per_target rows.
    del render_artifacts
    render["delta_predicted_minus_waymo_db"] = (
        float(render["predicted"]["mean_psnr_db"])
        - float(render["waymo"]["mean_psnr_db"])
    )
    render["delta_predicted_trunk_mean_minus_waymo_db"] = (
        float(render["predicted_trunk_mean"]["mean_psnr_db"])
        - float(render["waymo"]["mean_psnr_db"])
    )
    render["delta_predicted_trunk_mean_minus_predicted_db"] = (
        float(render["predicted_trunk_mean"]["mean_psnr_db"])
        - float(render["predicted"]["mean_psnr_db"])
    )

    row = {
        "scene": scene,
        "trunk": int(trunk),
        "global_frame_range": [
            int(trunk) * TRUNK_FRAMES,
            int(trunk) * TRUNK_FRAMES + TRUNK_FRAMES - 1,
        ],
        "image_hw": [height, width],
        # Keep the historical key as an alias, but its corrected semantics are
        # now max pairwise baseline rather than accumulated path length.
        "ego_motion_m": motion["max_pairwise_span_m"],
        "ego_motion_max_pairwise_span_m": motion["max_pairwise_span_m"],
        "ego_motion_path_length_m": motion["path_length_m"],
        "has_dynamic_mask": has_dynamic_mask,
        "static_pixel_fraction": float(static_mask.float().mean().item()),
        "fov_dggt": summarize_fov_trunk(pose),
        "fov_waymo_xy_deg": _waymo_fov_xy_deg(candidates["waymo"], (height, width)),
        "geometry_point_head": geometry,
        "geometry_point_head_compatibility": geometry_compatibility,
        "render_leave_one_out_static": render,
        "d3_metric_camera": d3,
    }
    del heads, item, context, images, static_mask, candidates, world_to_camera
    torch.cuda.empty_cache()
    return row


def _build_d3_summary(
    rows: Sequence[dict[str, Any]],
    *,
    render_stride: int,
    max_loss_db: float,
    bootstrap_samples: int,
    seed: int,
    min_decision_trunks: int,
    min_decision_scenes: int,
    min_shared_alpha_fraction: float,
    min_success_fraction: float,
) -> dict[str, Any]:
    """Build the D3 metric-camera render gate without changing D1 semantics."""

    valid_rows = [
        row
        for row in rows
        if "error" not in row
        and bool(row.get("d3_metric_camera", {}).get("valid"))
        and "render_leave_one_out_static" in row.get("d3_metric_camera", {})
    ]
    requested_count = len(rows)
    success_fraction = (
        float(len(valid_rows) / requested_count) if requested_count else 0.0
    )

    def loss_record(row: Mapping[str, Any]) -> dict[str, Any]:
        render = row["d3_metric_camera"]["render_leave_one_out_static"]
        return {
            "scene": str(row["scene"]),
            "trunk": int(row["trunk"]),
            "fixed_static_loss_db": float(render["loss_native_minus_metric_db"]),
            "shared_alpha_loss_db": float(
                render["shared_alpha_support"]["loss_native_minus_metric_db"]
            ),
        }

    all_records = [loss_record(row) for row in valid_rows]

    def motion_subset_summary(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        records = [loss_record(row) for row in subset]
        return {
            "n_trunks": len(records),
            "n_scenes": len({record["scene"] for record in records}),
            "fixed_static_loss_db": _summary(
                record["fixed_static_loss_db"] for record in records
            ),
            "shared_alpha_loss_db": _summary(
                record["shared_alpha_loss_db"] for record in records
            ),
        }

    moving_rows = [
        row
        for row in valid_rows
        if float(row["ego_motion_max_pairwise_span_m"]) > 2.0
    ]
    stationary_rows = [
        row
        for row in valid_rows
        if float(row["ego_motion_max_pairwise_span_m"]) <= 2.0
    ]
    decision_rows = [
        row
        for row in valid_rows
        if bool(row["has_dynamic_mask"])
        and float(
            row["d3_metric_camera"]["render_leave_one_out_static"][
                "shared_alpha_support"
            ]["min_shared_fraction_of_fixed_static"]
        )
        >= float(min_shared_alpha_fraction)
    ]
    decision_records = [loss_record(row) for row in decision_rows]
    scene_ids = [record["scene"] for record in decision_records]
    n_scenes = len(set(scene_ids))
    if (
        len(decision_records) >= int(min_decision_trunks)
        and n_scenes >= int(min_decision_scenes)
    ):
        fixed = decide_metric_camera_render_space(
            [record["fixed_static_loss_db"] for record in decision_records],
            scene_ids=scene_ids,
            max_loss_db=max_loss_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        shared = decide_metric_camera_render_space(
            [record["shared_alpha_loss_db"] for record in decision_records],
            scene_ids=scene_ids,
            max_loss_db=max_loss_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        decision = combine_metric_camera_render_decisions(fixed, shared)
    else:
        decision = {
            "render_camera_space": "INCONCLUSIVE",
            "reason": (
                f"only {len(decision_records)} eligible trunks across {n_scenes} "
                "scenes after motion/dynamic-mask/shared-alpha gates; need at "
                f"least {int(min_decision_trunks)} trunks and "
                f"{int(min_decision_scenes)} scenes"
            ),
        }
    if int(render_stride) > 2 and decision["render_camera_space"] in (
        "metric",
        "teacher",
    ):
        provisional = decision
        decision = {
            "render_camera_space": "INCONCLUSIVE",
            "reason": (
                f"render_stride={int(render_stride)} is screening-only; stride <= 2 "
                "is required for the decisive D3 gate"
            ),
            "provisional_stride_gt2_decision": provisional,
        }
    if success_fraction < float(min_success_fraction):
        provisional = decision
        decision = {
            "render_camera_space": "INCONCLUSIVE",
            "reason": (
                f"only {len(valid_rows)}/{requested_count} requested trunks have a "
                f"valid D3 LiDAR gauge and render ({success_fraction:.3f}); require "
                f">= {float(min_success_fraction):.3f}"
            ),
            "provisional_before_success_fraction_gate": provisional,
        }
    return {
        "gate_name": "D3_metric_camera_render_replacement",
        "max_allowed_loss_db": float(max_loss_db),
        "loss_definition": "PSNR(native_DGGT_camera) - PSNR(metric_converted_camera)",
        "n_valid_trunks": len(valid_rows),
        "n_requested_trunks": requested_count,
        "success_fraction": success_fraction,
        "min_success_fraction": float(min_success_fraction),
        "fixed_static_loss_db_all": _summary(
            record["fixed_static_loss_db"] for record in all_records
        ),
        "shared_alpha_loss_db_all": _summary(
            record["shared_alpha_loss_db"] for record in all_records
        ),
        "native_psnr_db_all": _summary(
            row["d3_metric_camera"]["render_leave_one_out_static"][
                "native_dggt_camera"
            ]["mean_psnr_db"]
            for row in valid_rows
        ),
        "metric_psnr_db_all": _summary(
            row["d3_metric_camera"]["render_leave_one_out_static"][
                "metric_converted_camera"
            ]["mean_psnr_db"]
            for row in valid_rows
        ),
        "scene_breakdown_all_valid": metric_camera_scene_breakdown(all_records),
        "scene_breakdown_decision_population": metric_camera_scene_breakdown(
            decision_records
        ),
        "motion_sensitivity_descriptive": {
            "split_rule": "29-frame max pairwise Waymo camera-centre span > 2m",
            "not_used_for_primary_gate": True,
            "moving": motion_subset_summary(moving_rows),
            "stationary": motion_subset_summary(stationary_rows),
        },
        "decision_population": {
            "n_trunks": len(decision_rows),
            "n_scenes": n_scenes,
            "motion_gate": None,
            "includes_stationary_trunks": True,
            "motion_gate_reason": (
                "D3 uses the LiDAR teacher ruler, so camera-scale identifiability "
                "is irrelevant and rotational replacement cost remains measurable "
                "for stationary trunks"
            ),
            "requires_dynamic_mask": True,
            "min_shared_alpha_fraction": float(min_shared_alpha_fraction),
            "excluded_invalid_lidar_or_render": requested_count - len(valid_rows),
            "excluded_without_dynamic_mask": sum(
                not bool(row["has_dynamic_mask"])
                for row in valid_rows
            ),
            "excluded_low_shared_alpha_coverage": sum(
                bool(row["has_dynamic_mask"])
                and float(
                    row["d3_metric_camera"]["render_leave_one_out_static"][
                        "shared_alpha_support"
                    ]["min_shared_fraction_of_fixed_static"]
                )
                < float(min_shared_alpha_fraction)
                for row in valid_rows
            ),
        },
        "decision": decision,
    }


def _build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    render_stride: int,
    min_decision_motion_m: float,
    min_effect_db: float,
    bootstrap_samples: int,
    seed: int,
    min_decision_trunks: int,
    min_decision_scenes: int,
    min_shared_alpha_fraction: float,
    min_success_fraction: float,
    metric_camera_max_loss_db: float = D3_DEFAULT_MAX_LOSS_DB,
) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    if not valid:
        raise RuntimeError("no valid trunks were processed")
    if not 0.0 <= float(min_success_fraction) <= 1.0:
        raise ValueError("min_success_fraction must be in [0,1]")
    success_fraction = float(len(valid) / len(rows))
    x_std = [row["fov_dggt"]["std_xy_deg"][0] for row in valid]
    y_std = [row["fov_dggt"]["std_xy_deg"][1] for row in valid]
    x_means = [row["fov_dggt"]["mean_xy_deg"][0] for row in valid]
    y_means = [row["fov_dggt"]["mean_xy_deg"][1] for row in valid]
    decision_rows = [
        row
        for row in valid
        if float(row["ego_motion_max_pairwise_span_m"])
        > float(min_decision_motion_m)
        and bool(row["has_dynamic_mask"])
        and float(
            row["render_leave_one_out_static"][
                "minimum_shared_support_fraction_across_comparisons"
            ]
        )
        >= float(min_shared_alpha_fraction)
    ]
    scene_ids = [str(row["scene"]) for row in decision_rows]
    n_decision_scenes = len(set(scene_ids))
    framewise_full_deltas = [
        float(row["render_leave_one_out_static"]["delta_predicted_minus_waymo_db"])
        for row in decision_rows
    ]
    framewise_shared_deltas = [
        float(
            row["render_leave_one_out_static"]["shared_alpha_support"][
                "delta_predicted_minus_waymo_db"
            ]
        )
        for row in decision_rows
    ]
    trunk_mean_full_deltas = [
        float(
            row["render_leave_one_out_static"][
                "delta_predicted_trunk_mean_minus_waymo_db"
            ]
        )
        for row in decision_rows
    ]
    trunk_mean_shared_deltas = [
        float(
            row["render_leave_one_out_static"][
                "shared_alpha_support_trunk_mean_vs_waymo"
            ]["delta_predicted_trunk_mean_minus_waymo_db"]
        )
        for row in decision_rows
    ]
    trunk_mean_minus_framewise_full_deltas = [
        float(
            row["render_leave_one_out_static"][
                "delta_predicted_trunk_mean_minus_predicted_db"
            ]
        )
        for row in decision_rows
    ]
    trunk_mean_minus_framewise_shared_deltas = [
        float(
            row["render_leave_one_out_static"][
                "shared_alpha_support_trunk_mean_vs_predicted"
            ]["delta_predicted_trunk_mean_minus_predicted_db"]
        )
        for row in decision_rows
    ]
    if (
        len(framewise_full_deltas) >= int(min_decision_trunks)
        and n_decision_scenes >= int(min_decision_scenes)
    ):
        framewise_full_decision = decide_fov_branch(
            framewise_full_deltas,
            scene_ids=scene_ids,
            min_effect_db=min_effect_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        framewise_shared_decision = decide_fov_branch(
            framewise_shared_deltas,
            scene_ids=scene_ids,
            min_effect_db=min_effect_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        framewise_decision = combine_render_decisions(
            framewise_full_decision, framewise_shared_decision
        )
        trunk_mean_full_decision = decide_fov_branch(
            trunk_mean_full_deltas,
            scene_ids=scene_ids,
            min_effect_db=min_effect_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        trunk_mean_shared_decision = decide_fov_branch(
            trunk_mean_shared_deltas,
            scene_ids=scene_ids,
            min_effect_db=min_effect_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        trunk_mean_decision = combine_render_decisions(
            trunk_mean_full_decision, trunk_mean_shared_decision
        )
        full_noninferiority = decide_scene_cluster_noninferiority(
            trunk_mean_minus_framewise_full_deltas,
            scene_ids=scene_ids,
            margin_db=min_effect_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        shared_noninferiority = decide_scene_cluster_noninferiority(
            trunk_mean_minus_framewise_shared_deltas,
            scene_ids=scene_ids,
            margin_db=min_effect_db,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        decision = combine_architecture_decisions(
            framewise_decision,
            trunk_mean_decision,
            full_noninferiority,
            shared_noninferiority,
        )
        decision["min_ego_motion_m_exclusive"] = float(min_decision_motion_m)
        decision["requires_dynamic_mask"] = True
        decision["min_shared_alpha_fraction"] = float(min_shared_alpha_fraction)
    else:
        decision = {
            "branch": "INCONCLUSIVE",
            "reason": (
                f"only {len(framewise_full_deltas)} eligible trunks across "
                f"{n_decision_scenes} scenes "
                "after motion/dynamic-mask/shared-alpha-coverage gates; need at least "
                f"{int(min_decision_trunks)} trunks and {int(min_decision_scenes)} scenes"
            ),
            "n_trunks": len(framewise_full_deltas),
            "n_scenes": n_decision_scenes,
            "min_ego_motion_m_exclusive": float(min_decision_motion_m),
            "requires_dynamic_mask": True,
            "min_shared_alpha_fraction": float(min_shared_alpha_fraction),
        }
    if int(render_stride) > 2 and decision["branch"] in ("A", "B"):
        provisional = decision
        decision = {
            "branch": "INCONCLUSIVE",
            "reason": (
                f"render_stride={int(render_stride)} is screening-only; a stride <= 2 "
                "run is required for the decisive D1 branch"
            ),
            "provisional_stride_gt2_decision": provisional,
            "required_max_decisive_render_stride": 2,
        }
    if success_fraction < float(min_success_fraction):
        provisional = decision
        decision = {
            "branch": "INCONCLUSIVE",
            "reason": (
                f"only {len(valid)}/{len(rows)} requested trunks succeeded "
                f"({success_fraction:.3f}); require >= {float(min_success_fraction):.3f}"
            ),
            "provisional_before_success_fraction_gate": provisional,
            "success_fraction": success_fraction,
            "min_success_fraction": float(min_success_fraction),
        }
    geometry_summary: dict[str, Any] = {}
    for candidate in ("predicted", "predicted_trunk_mean", "waymo"):
        geometry_summary[candidate] = {
            metric: _summary(
                row["geometry_point_head"][candidate][metric] for row in valid
            )
            for metric in (
                "relative_l2_median",
                "angular_deg_median",
                "reprojection_px_median",
            )
        }
    compatibility_statuses = [
        str(row["geometry_point_head_compatibility"]["status"]) for row in valid
    ]
    compatibility_counts = {
        status: compatibility_statuses.count(status)
        for status in sorted(set(compatibility_statuses))
    }
    return {
        "n_valid_trunks": len(valid),
        "n_error_trunks": len(rows) - len(valid),
        "n_requested_trunks": len(rows),
        "success_fraction": success_fraction,
        "min_success_fraction_for_decision": float(min_success_fraction),
        "fov_within_trunk_std_deg": {"x": _summary(x_std), "y": _summary(y_std)},
        "fov_between_trunk_mean_std_deg": {
            "x": float(np.std(np.asarray(x_means, dtype=np.float64))),
            "y": float(np.std(np.asarray(y_means, dtype=np.float64))),
        },
        "geometry_point_head": {
            "candidates": geometry_summary,
            "coordinate_compatibility_status_counts": compatibility_counts,
            "n_coordinate_compatible_trunks": compatibility_statuses.count("compatible"),
            "used_for_branch_decision": False,
        },
        "render_predicted_psnr_db": _summary(
            row["render_leave_one_out_static"]["predicted"]["mean_psnr_db"]
            for row in valid
        ),
        "render_predicted_trunk_mean_psnr_db": _summary(
            row["render_leave_one_out_static"]["predicted_trunk_mean"]["mean_psnr_db"]
            for row in valid
        ),
        "render_waymo_psnr_db": _summary(
            row["render_leave_one_out_static"]["waymo"]["mean_psnr_db"]
            for row in valid
        ),
        "render_delta_predicted_trunk_mean_minus_waymo_db_all": _summary(
            row["render_leave_one_out_static"][
                "delta_predicted_trunk_mean_minus_waymo_db"
            ]
            for row in valid
        ),
        "render_delta_predicted_trunk_mean_minus_predicted_db_all": _summary(
            row["render_leave_one_out_static"][
                "delta_predicted_trunk_mean_minus_predicted_db"
            ]
            for row in valid
        ),
        "render_delta_predicted_minus_waymo_db_all": _summary(
            row["render_leave_one_out_static"]["delta_predicted_minus_waymo_db"]
            for row in valid
        ),
        "render_shared_alpha_delta_predicted_trunk_mean_minus_waymo_db_all": _summary(
            row["render_leave_one_out_static"][
                "shared_alpha_support_trunk_mean_vs_waymo"
            ]["delta_predicted_trunk_mean_minus_waymo_db"]
            for row in valid
        ),
        "render_shared_alpha_delta_predicted_trunk_mean_minus_predicted_db_all": _summary(
            row["render_leave_one_out_static"][
                "shared_alpha_support_trunk_mean_vs_predicted"
            ]["delta_predicted_trunk_mean_minus_predicted_db"]
            for row in valid
        ),
        "render_shared_alpha_delta_predicted_minus_waymo_db_all": _summary(
            row["render_leave_one_out_static"]["shared_alpha_support"][
                "delta_predicted_minus_waymo_db"
            ]
            for row in valid
        ),
        "render_shared_alpha_psnr_db": {
            "predicted": _summary(
                row["render_leave_one_out_static"]["shared_alpha_support"][
                    "mean_predicted_psnr_db"
                ]
                for row in valid
            ),
            "waymo": _summary(
                row["render_leave_one_out_static"]["shared_alpha_support"][
                    "mean_waymo_psnr_db"
                ]
                for row in valid
            ),
        },
        "render_alpha_gt_005_fraction_on_eval": {
            candidate: _summary(
                row["render_leave_one_out_static"][candidate][
                    "mean_alpha_gt_005_fraction_on_eval"
                ]
                for row in valid
            )
            for candidate in ("predicted", "predicted_trunk_mean", "waymo")
        },
        "render_shared_alpha_fraction_of_fixed_static": _summary(
            row["render_leave_one_out_static"]["shared_alpha_support"][
                "mean_shared_fraction_of_fixed_static"
            ]
            for row in valid
        ),
        "render_shared_alpha_min_fraction_across_targets": _summary(
            row["render_leave_one_out_static"]["shared_alpha_support"][
                "min_shared_fraction_of_fixed_static"
            ]
            for row in valid
        ),
        "render_minimum_shared_support_fraction_across_comparisons": _summary(
            row["render_leave_one_out_static"][
                "minimum_shared_support_fraction_across_comparisons"
            ]
            for row in valid
        ),
        "decision_population": {
            "n_trunks": len(decision_rows),
            "n_scenes": n_decision_scenes,
            "excluded_without_dynamic_mask": sum(
                not bool(row["has_dynamic_mask"])
                and float(row["ego_motion_max_pairwise_span_m"])
                > float(min_decision_motion_m)
                for row in valid
            ),
            "excluded_low_motion": sum(
                float(row["ego_motion_max_pairwise_span_m"])
                <= float(min_decision_motion_m)
                for row in valid
            ),
            "excluded_low_shared_alpha_coverage": sum(
                float(row["ego_motion_max_pairwise_span_m"])
                > float(min_decision_motion_m)
                and bool(row["has_dynamic_mask"])
                and float(
                    row["render_leave_one_out_static"][
                        "minimum_shared_support_fraction_across_comparisons"
                    ]
                )
                < float(min_shared_alpha_fraction)
                for row in valid
            ),
        },
        "decision": decision,
        "d3_metric_camera": _build_d3_summary(
            rows,
            render_stride=int(render_stride),
            max_loss_db=float(metric_camera_max_loss_db),
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed),
            min_decision_trunks=int(min_decision_trunks),
            min_decision_scenes=int(min_decision_scenes),
            min_shared_alpha_fraction=float(min_shared_alpha_fraction),
            min_success_fraction=float(min_success_fraction),
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=os.environ.get("GAUGE_DEVICE", "cuda:0"))
    parser.add_argument("--scenes", default="300-329", help="e.g. 300-329 or 300,302")
    parser.add_argument("--trunks", default="0,1,2")
    parser.add_argument("--render-targets", default="0,7,14,21,28")
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--geometry-stride", type=int, default=2)
    parser.add_argument("--confidence-quantile", type=float, default=0.5)
    parser.add_argument("--min-eval-pixels", type=int, default=500)
    parser.add_argument("--min-decision-motion-m", type=float, default=2.0)
    parser.add_argument("--min-effect-db", type=float, default=0.2)
    parser.add_argument(
        "--metric-camera-max-loss-db",
        type=float,
        default=D3_DEFAULT_MAX_LOSS_DB,
        help="D3 native-minus-metric PSNR loss gate",
    )
    parser.add_argument("--min-decision-trunks", type=int, default=10)
    parser.add_argument("--min-decision-scenes", type=int, default=10)
    parser.add_argument("--min-shared-alpha-fraction", type=float, default=0.2)
    parser.add_argument("--min-success-fraction", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trunks", type=int, default=0, help="0 means no limit")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _run_config(
    args: argparse.Namespace,
    *,
    scenes: Sequence[str],
    trunks: Sequence[int],
    targets: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "image_dir": os.path.abspath(args.image_dir),
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "scenes": list(scenes),
        "trunks": [int(value) for value in trunks],
        "render_targets": [int(value) for value in targets],
        "render_stride": int(args.render_stride),
        "geometry_stride": int(args.geometry_stride),
        "confidence_quantile": float(args.confidence_quantile),
        "min_eval_pixels": int(args.min_eval_pixels),
        "min_decision_motion_m": float(args.min_decision_motion_m),
        "min_effect_db": float(args.min_effect_db),
        "metric_camera_max_loss_db": float(args.metric_camera_max_loss_db),
        "min_decision_trunks": int(args.min_decision_trunks),
        "min_decision_scenes": int(args.min_decision_scenes),
        "min_shared_alpha_fraction": float(args.min_shared_alpha_fraction),
        "min_success_fraction": float(args.min_success_fraction),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "method_notes": [
            "primitive-level leave-one-out: target-frame means are excluded, but the 29-frame aggregator saw target RGB",
            "candidate K is used for both source depth unprojection and target rasterization",
            "predicted_trunk_mean uses mean(log(tan(FOV/2))) exactly as the scene-global gauge representation",
            "the realizable trunk-mean-vs-Waymo render is decisive; native per-frame K must choose the same winner",
            "trunk-mean-vs-native non-inferiority is reported as a diagnostic and does not override the direct winner",
            "branch bootstrap unit is scene after averaging trunks within scene",
            "decision motion gate uses 29-frame camera-centre max pairwise span, not accumulated path length",
            "requested trunk success fraction must reach the configured threshold before A/B is allowed",
            "full-static-mask and shared-alpha-support decisions must agree",
            "render_stride subsamples Gaussians without rescaling their 3D covariance; stride > 2 is screening-only and cannot return A/B",
            "PointHead metrics are excluded from branch selection when its coordinate convention fails a loose reprojection sanity check",
            "D3 uses the D2 29-frame LiDAR teacher ruler and changes only the render view matrix; Gaussian primitives and trunk-mean gauge K are shared",
            "D3 includes stationary trunks because LiDAR fixes scale and camera-rotation replacement cost remains measurable; D1 alone uses the >2m camera-motion gate",
            "D3 loss is native-DGGT-camera PSNR minus metric-converted-camera PSNR, bootstrapped after averaging trunks within scene",
            "D3 requires fixed-static and shared-alpha 95% CIs to agree relative to the configured 0.3 dB gate",
            "D3 scene breakdown always surfaces scenes 312, 314, and 325",
        ],
    }


def phase0_d1_d3_exit_code(summary: Mapping[str, Any]) -> int:
    """Return a stable CLI exit code for the two independent Phase-0 gates."""

    d1_branch = str(summary["decision"]["branch"])
    if d1_branch == "INCONCLUSIVE":
        return 2
    if d1_branch not in ("A", "B"):
        raise ValueError(f"unexpected D1 branch {d1_branch!r}")
    d3_space = str(
        summary["d3_metric_camera"]["decision"]["render_camera_space"]
    )
    if d3_space == "INCONCLUSIVE":
        return 3
    if d3_space not in ("metric", "teacher"):
        raise ValueError(f"unexpected D3 render camera space {d3_space!r}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scenes = parse_index_spec(args.scenes, zero_pad=3)
    trunks = tuple(int(value) for value in parse_index_spec(args.trunks))
    targets = tuple(int(value) for value in parse_index_spec(args.render_targets))
    if any(target >= TRUNK_FRAMES for target in targets):
        raise ValueError(f"render targets must be in [0,{TRUNK_FRAMES}), got {targets}")
    if int(args.render_stride) <= 0 or int(args.geometry_stride) <= 0:
        raise ValueError("render/geometry strides must be positive")
    if int(args.min_eval_pixels) <= 0:
        raise ValueError("min_eval_pixels must be positive")
    if int(args.min_decision_trunks) <= 0 or int(args.min_decision_scenes) <= 0:
        raise ValueError("min_decision_trunks/min_decision_scenes must be positive")
    if not 0.0 <= float(args.min_shared_alpha_fraction) <= 1.0:
        raise ValueError("min_shared_alpha_fraction must be in [0,1]")
    if not 0.0 <= float(args.min_success_fraction) <= 1.0:
        raise ValueError("min_success_fraction must be in [0,1]")
    if (
        not math.isfinite(float(args.metric_camera_max_loss_db))
        or float(args.metric_camera_max_loss_db) < 0.0
    ):
        raise ValueError("metric_camera_max_loss_db must be finite and non-negative")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the real D1 diagnostic requires CUDA because gsplat is CUDA-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_path = Path(args.output).expanduser().resolve()
    run_config = _run_config(
        args,
        scenes=scenes,
        trunks=trunks,
        targets=targets,
        device=device,
    )
    start_time = time.time()
    atomic_write_json(
        output_path,
        {
            "schema_version": "verify_fov_consistency_v3_d3_metric_camera",
            "status": "initializing",
            "config": run_config,
            "trunks": [],
            "elapsed_seconds": 0.0,
        },
    )
    model = _load_model(args.checkpoint, device)
    requested = [(scene, trunk) for scene in scenes for trunk in trunks]
    if int(args.max_trunks) > 0:
        requested = requested[: int(args.max_trunks)]
    rows: list[dict[str, Any]] = []
    for index, (scene, trunk) in enumerate(requested, start=1):
        label = f"scene={scene} trunk={trunk}"
        print(f"[{index}/{len(requested)}] {label}", flush=True)
        try:
            row = _process_trunk(
                model=model,
                image_dir=args.image_dir,
                scene=scene,
                trunk=trunk,
                device=device,
                render_targets=targets,
                render_stride=int(args.render_stride),
                geometry_stride=int(args.geometry_stride),
                confidence_quantile=float(args.confidence_quantile),
                min_eval_pixels=int(args.min_eval_pixels),
            )
            rows.append(row)
            delta = row["render_leave_one_out_static"]["delta_predicted_minus_waymo_db"]
            fov_std = row["fov_dggt"]["std_xy_deg"]
            d3_render = row["d3_metric_camera"].get("render_leave_one_out_static")
            d3_text = (
                "n/a"
                if d3_render is None
                else f"{float(d3_render['loss_native_minus_metric_db']):+.3f}dB"
            )
            print(
                f"  motion={row['ego_motion_m']:.2f}m  "
                f"FOV std xy=({fov_std[0]:.3f},{fov_std[1]:.3f})deg  "
                f"delta PSNR(pred-waymo)={delta:+.3f}dB  "
                f"D3 loss(native-metric)={d3_text}",
                flush=True,
            )
        except Exception as exc:
            error_row = {
                "scene": scene,
                "trunk": int(trunk),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            rows.append(error_row)
            print(f"  ERROR: {error_row['error']}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
        finally:
            torch.cuda.empty_cache()
            # Preserve every completed/error row before advancing.  If the
            # process is interrupted later, this valid JSON is the recovery
            # record; os.replace prevents readers from observing a partial
            # write.
            atomic_write_json(
                output_path,
                {
                    "schema_version": "verify_fov_consistency_v3_d3_metric_camera",
                    "status": "running",
                    "config": run_config,
                    "completed_requests": len(rows),
                    "total_requests": len(requested),
                    "trunks": rows,
                    "elapsed_seconds": float(time.time() - start_time),
                },
            )

    summary = _build_summary(
        rows,
        render_stride=int(args.render_stride),
        min_decision_motion_m=float(args.min_decision_motion_m),
        min_effect_db=float(args.min_effect_db),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
        min_decision_trunks=int(args.min_decision_trunks),
        min_decision_scenes=int(args.min_decision_scenes),
        min_shared_alpha_fraction=float(args.min_shared_alpha_fraction),
        min_success_fraction=float(args.min_success_fraction),
        metric_camera_max_loss_db=float(args.metric_camera_max_loss_db),
    )
    payload = {
        "schema_version": "verify_fov_consistency_v3_d3_metric_camera",
        "status": "complete",
        "config": run_config,
        "summary": summary,
        "trunks": rows,
        "elapsed_seconds": float(time.time() - start_time),
    }
    atomic_write_json(output_path, payload)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print(f"wrote {output_path}", flush=True)
    exit_code = phase0_d1_d3_exit_code(summary)
    if exit_code == 2:
        print(
            "D1 branch is inconclusive; run enough moving trunks before changing Phase-1 values.",
            file=sys.stderr,
        )
        return 2
    if exit_code == 3:
        print(
            "D3 metric-camera render gate is inconclusive; do not choose the "
            "Phase-3 render camera space yet.",
            file=sys.stderr,
        )
        return 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
