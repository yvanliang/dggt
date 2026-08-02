"""Generated camera/static-geometry reprojection-flow consistency diagnostic.

The current generator has no independent optical-flow, scene-flow, or point
correspondence head.  Consequently it is not possible to compare a predicted
"geometry-only flow" with camera flow without inventing correspondences.  This
module implements the strongest observable alternative from generated outputs:

1. unproject generated depth in frame ``t``;
2. transport it with the generated camera motion and project into ``t+1``;
3. sample generated target depth at that landing point;
4. transport that target-depth point back and measure the flow-cycle error.

The paired projected-vs-sampled z-depth residual catches pure-rotation cases,
where optical flow itself is depth-independent.  Sky, predicted-dynamic,
out-of-frustum, invalid-depth, and occluded support is reported explicitly.
No input/GT image is read or required.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from dggt.utils.pose_enc import pose_encoding_to_extri_intri


CAMERA_GEOMETRY_FLOW_DIAGNOSTIC_SCHEMA = (
    "generated_static_geometry_reprojection_cycle_v1"
)


def _as_depth(depth: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(depth).float()
    if value.ndim == 5 and int(value.shape[-1]) == 1:
        value = value[..., 0]
    if value.ndim != 4:
        raise ValueError(
            "depth must be [B,S,H,W] or [B,S,H,W,1], got "
            f"{tuple(value.shape)}"
        )
    return value


def _as_probability_map(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    height: int,
    width: int,
    name: str,
    logits: bool,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).float()
    if tensor.ndim == 5 and int(tensor.shape[-1]) == 1:
        tensor = tensor[..., 0]
    elif tensor.ndim == 5 and int(tensor.shape[2]) in (1, 3):
        tensor = tensor.mean(dim=2)
    if tuple(tensor.shape) != (batch_size, seq_len, height, width):
        raise ValueError(
            f"{name} must map to [B,S,H,W]={(batch_size, seq_len, height, width)}, "
            f"got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return torch.sigmoid(tensor) if logits else tensor.clamp(0.0, 1.0)


def _world_to_camera_4x4(pose_enc: torch.Tensor, image_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsic_3x4, intrinsics = pose_encoding_to_extri_intri(
        pose_enc.float(), image_hw
    )
    if intrinsics is None:
        raise RuntimeError("pose encoding did not produce intrinsics")
    batch_size, seq_len = int(pose_enc.shape[0]), int(pose_enc.shape[1])
    world_to_camera = torch.zeros(
        (batch_size, seq_len, 4, 4),
        device=pose_enc.device,
        dtype=torch.float32,
    )
    world_to_camera[..., :3, :] = extrinsic_3x4.float()
    world_to_camera[..., 3, 3] = 1.0
    return world_to_camera, intrinsics.float()


def _rigid_inverse(matrix: torch.Tensor) -> torch.Tensor:
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    result = torch.zeros_like(matrix)
    result[..., :3, :3] = rotation.transpose(-1, -2)
    result[..., :3, 3] = -torch.matmul(
        rotation.transpose(-1, -2), translation.unsqueeze(-1)
    ).squeeze(-1)
    result[..., 3, 3] = 1.0
    return result


def _sample_bilinear(image: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    grid_x = 2.0 * x / float(width - 1) - 1.0
    grid_y = 2.0 * y / float(height - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image.view(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def _distribution(values: torch.Tensor) -> dict[str, int | float | None]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
    }


def _cat_or_empty(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    return torch.cat(values) if values else reference.new_zeros((0,), dtype=torch.float32)


@torch.no_grad()
def camera_geometry_flow_consistency(
    depth_dggt: torch.Tensor,
    pose_enc_dggt: torch.Tensor,
    *,
    sky_probability: torch.Tensor | None = None,
    dynamic_logits: torch.Tensor | None = None,
    sample_stride: int = 4,
    depth_min: float = 1.0e-4,
    sky_probability_threshold: float = 0.5,
    dynamic_probability_threshold: float = 0.5,
    occlusion_relative_tolerance: float = 0.05,
    min_pair_support_pixels: int = 64,
    min_informative_motion_px: float = 0.25,
) -> dict[str, Any]:
    """Measure adjacent-frame static depth reprojection/cycle consistency.

    This is a generated-output diagnostic, not an independently predicted
    optical-flow metric.  Its primary errors are full-resolution pixels and
    absolute log z-depth.  Metrics are aggregated separately over all valid
    support and over motion-informative pairs.
    """

    depth = _as_depth(depth_dggt)
    pose = torch.as_tensor(
        pose_enc_dggt, device=depth.device, dtype=torch.float32
    )
    if pose.ndim != 3 or int(pose.shape[-1]) != 9:
        raise ValueError(f"pose_enc_dggt must be [B,S,9], got {tuple(pose.shape)}")
    batch_size, seq_len, height, width = (int(value) for value in depth.shape)
    if tuple(pose.shape[:2]) != (batch_size, seq_len):
        raise ValueError(
            f"pose batch/sequence {tuple(pose.shape[:2])} != depth {(batch_size, seq_len)}"
        )
    if height <= 1 or width <= 1:
        raise ValueError("flow consistency requires image height and width greater than one")
    if isinstance(sample_stride, bool) or int(sample_stride) <= 0:
        raise ValueError("sample_stride must be a positive integer")
    if not 0.0 < float(depth_min):
        raise ValueError("depth_min must be positive")
    for name, threshold in (
        ("sky_probability_threshold", sky_probability_threshold),
        ("dynamic_probability_threshold", dynamic_probability_threshold),
    ):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if float(occlusion_relative_tolerance) < 0.0:
        raise ValueError("occlusion_relative_tolerance must be non-negative")
    if isinstance(min_pair_support_pixels, bool) or int(min_pair_support_pixels) <= 0:
        raise ValueError("min_pair_support_pixels must be a positive integer")
    if float(min_informative_motion_px) < 0.0:
        raise ValueError("min_informative_motion_px must be non-negative")

    sky = _as_probability_map(
        sky_probability,
        batch_size=batch_size,
        seq_len=seq_len,
        height=height,
        width=width,
        name="sky_probability",
        logits=False,
    )
    dynamic = _as_probability_map(
        dynamic_logits,
        batch_size=batch_size,
        seq_len=seq_len,
        height=height,
        width=width,
        name="dynamic_logits",
        logits=True,
    )
    valid = torch.isfinite(depth) & (depth > float(depth_min))
    if sky is not None:
        valid &= sky <= float(sky_probability_threshold)
    if dynamic is not None:
        valid &= dynamic <= float(dynamic_probability_threshold)
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))

    world_to_camera, intrinsics = _world_to_camera_4x4(
        pose, (height, width)
    )
    if not bool(torch.isfinite(world_to_camera).all()) or not bool(
        torch.isfinite(intrinsics).all()
    ):
        raise ValueError("generated camera matrices contain non-finite values")
    if bool((intrinsics[..., 0, 0] <= 0.0).any()) or bool(
        (intrinsics[..., 1, 1] <= 0.0).any()
    ):
        raise ValueError("generated camera focal lengths must be positive")
    camera_to_world = _rigid_inverse(world_to_camera)

    stride = int(sample_stride)
    yy, xx = torch.meshgrid(
        torch.arange(0, height, stride, device=depth.device, dtype=torch.float32),
        torch.arange(0, width, stride, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    source_x = xx.reshape(-1)
    source_y = yy.reshape(-1)
    sampled_pixels_per_pair = int(source_x.numel())

    all_cycle: list[torch.Tensor] = []
    all_depth_error: list[torch.Tensor] = []
    all_raw_depth_error: list[torch.Tensor] = []
    all_flow: list[torch.Tensor] = []
    informative_cycle: list[torch.Tensor] = []
    informative_depth_error: list[torch.Tensor] = []
    informative_flow: list[torch.Tensor] = []
    pairs: list[dict[str, Any]] = []
    total_source = total_projectable = total_raw = total_visible = total_metric = 0
    total_occluded = 0
    reason_counts: dict[str, int] = {}

    for batch_index in range(batch_size):
        for frame_index in range(max(0, seq_len - 1)):
            source_mask = valid[batch_index, frame_index, ::stride, ::stride].reshape(-1)
            source_z = safe_depth[batch_index, frame_index, ::stride, ::stride].reshape(-1)
            k_source = intrinsics[batch_index, frame_index]
            k_target = intrinsics[batch_index, frame_index + 1]
            source_xyz = torch.stack(
                (
                    (source_x - k_source[0, 2]) * source_z / k_source[0, 0],
                    (source_y - k_source[1, 2]) * source_z / k_source[1, 1],
                    source_z,
                ),
                dim=-1,
            )
            source_to_target = (
                world_to_camera[batch_index, frame_index + 1]
                @ camera_to_world[batch_index, frame_index]
            )
            target_xyz_expected = (
                source_xyz @ source_to_target[:3, :3].transpose(0, 1)
                + source_to_target[:3, 3]
            )
            expected_target_z = target_xyz_expected[:, 2]
            safe_expected_z = expected_target_z.clamp_min(float(depth_min))
            target_x = (
                k_target[0, 0] * target_xyz_expected[:, 0] / safe_expected_z
                + k_target[0, 2]
            )
            target_y = (
                k_target[1, 1] * target_xyz_expected[:, 1] / safe_expected_z
                + k_target[1, 2]
            )
            finite_projection = (
                torch.isfinite(target_x)
                & torch.isfinite(target_y)
                & torch.isfinite(expected_target_z)
            )
            projectable = (
                source_mask
                & finite_projection
                & (expected_target_z > float(depth_min))
                & (target_x >= 0.0)
                & (target_x <= float(width - 1))
                & (target_y >= 0.0)
                & (target_y <= float(height - 1))
            )
            sampled_target_depth = _sample_bilinear(
                safe_depth[batch_index, frame_index + 1], target_x, target_y
            )
            sampled_target_support = _sample_bilinear(
                valid[batch_index, frame_index + 1].float(), target_x, target_y
            )
            raw_support = (
                projectable
                & (sampled_target_support >= 0.999)
                & torch.isfinite(sampled_target_depth)
                & (sampled_target_depth > float(depth_min))
            )
            visible = raw_support & (
                expected_target_z
                <= sampled_target_depth * (1.0 + float(occlusion_relative_tolerance))
            )

            safe_sampled_z = sampled_target_depth.clamp_min(float(depth_min))
            target_xyz_sampled = torch.stack(
                (
                    (target_x - k_target[0, 2])
                    * safe_sampled_z
                    / k_target[0, 0],
                    (target_y - k_target[1, 2])
                    * safe_sampled_z
                    / k_target[1, 1],
                    safe_sampled_z,
                ),
                dim=-1,
            )
            target_to_source = (
                world_to_camera[batch_index, frame_index]
                @ camera_to_world[batch_index, frame_index + 1]
            )
            source_xyz_cycle = (
                target_xyz_sampled @ target_to_source[:3, :3].transpose(0, 1)
                + target_to_source[:3, 3]
            )
            cycle_z = source_xyz_cycle[:, 2]
            safe_cycle_z = cycle_z.clamp_min(float(depth_min))
            cycle_x = (
                k_source[0, 0] * source_xyz_cycle[:, 0] / safe_cycle_z
                + k_source[0, 2]
            )
            cycle_y = (
                k_source[1, 1] * source_xyz_cycle[:, 1] / safe_cycle_z
                + k_source[1, 2]
            )
            cycle_finite = (
                torch.isfinite(cycle_x)
                & torch.isfinite(cycle_y)
                & torch.isfinite(cycle_z)
                & (cycle_z > float(depth_min))
            )
            metric_support = visible & cycle_finite
            flow_magnitude = torch.sqrt(
                (target_x - source_x).square() + (target_y - source_y).square()
            )
            cycle_error = torch.sqrt(
                (cycle_x - source_x).square() + (cycle_y - source_y).square()
            )
            depth_log_error = torch.abs(
                torch.log(safe_sampled_z) - torch.log(safe_expected_z)
            )

            source_count = int(source_mask.sum().item())
            projectable_count = int(projectable.sum().item())
            raw_count = int(raw_support.sum().item())
            visible_count = int(visible.sum().item())
            metric_count = int(metric_support.sum().item())
            occluded_count = int((raw_support & ~visible).sum().item())
            pair_flow = flow_magnitude[metric_support]
            pair_cycle = cycle_error[metric_support]
            pair_depth_error = depth_log_error[metric_support]
            raw_depth_error = depth_log_error[raw_support]
            mean_motion = (
                float(pair_flow.mean().item()) if pair_flow.numel() else None
            )
            if source_count == 0:
                reason = "no_static_source_support"
            elif projectable_count == 0:
                reason = "no_in_frustum_projection"
            elif metric_count < int(min_pair_support_pixels):
                reason = "insufficient_visible_support"
            elif mean_motion is None or mean_motion < float(min_informative_motion_px):
                reason = "low_motion"
            else:
                reason = "informative"
            informative = reason == "informative"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if metric_count:
                all_cycle.append(pair_cycle)
                all_depth_error.append(pair_depth_error)
                all_flow.append(pair_flow)
            if raw_count:
                all_raw_depth_error.append(raw_depth_error)
            if informative:
                informative_cycle.append(pair_cycle)
                informative_depth_error.append(pair_depth_error)
                informative_flow.append(pair_flow)
            total_source += source_count
            total_projectable += projectable_count
            total_raw += raw_count
            total_visible += visible_count
            total_metric += metric_count
            total_occluded += occluded_count
            pairs.append(
                {
                    "batch_index": batch_index,
                    "source_frame": frame_index,
                    "target_frame": frame_index + 1,
                    "status": reason,
                    "informative": informative,
                    "sampled_pixel_count": sampled_pixels_per_pair,
                    "source_static_valid_count": source_count,
                    "projectable_count": projectable_count,
                    "target_valid_count_pre_occlusion": raw_count,
                    "occluded_count": occluded_count,
                    "visible_count": visible_count,
                    "metric_support_count": metric_count,
                    "metric_support_fraction_of_sampled": metric_count
                    / float(sampled_pixels_per_pair),
                    "camera_flow_magnitude_px": _distribution(pair_flow),
                    "flow_cycle_epe_px": _distribution(pair_cycle),
                    "z_depth_abs_log_error": _distribution(pair_depth_error),
                    "z_depth_abs_log_error_pre_occlusion": _distribution(
                        raw_depth_error
                    ),
                }
            )

    all_pair_slots = batch_size * max(0, seq_len - 1)
    sampled_total = all_pair_slots * sampled_pixels_per_pair
    supported_pair_count = sum(
        int(pair["metric_support_count"] > 0) for pair in pairs
    )
    informative_pair_count = int(reason_counts.get("informative", 0))
    if seq_len < 2:
        status = "single_frame"
    elif informative_pair_count > 0:
        status = "ok"
    elif supported_pair_count > 0:
        status = "degenerate"
    else:
        status = "insufficient_support"

    all_metrics = {
        "camera_flow_magnitude_px": _distribution(
            _cat_or_empty(all_flow, depth)
        ),
        "flow_cycle_epe_px": _distribution(_cat_or_empty(all_cycle, depth)),
        "z_depth_abs_log_error": _distribution(
            _cat_or_empty(all_depth_error, depth)
        ),
        "z_depth_abs_log_error_pre_occlusion": _distribution(
            _cat_or_empty(all_raw_depth_error, depth)
        ),
    }
    informative_metrics = {
        "camera_flow_magnitude_px": _distribution(
            _cat_or_empty(informative_flow, depth)
        ),
        "flow_cycle_epe_px": _distribution(
            _cat_or_empty(informative_cycle, depth)
        ),
        "z_depth_abs_log_error": _distribution(
            _cat_or_empty(informative_depth_error, depth)
        ),
    }
    return {
        "schema": CAMERA_GEOMETRY_FLOW_DIAGNOSTIC_SCHEMA,
        "status": status,
        "name": "generated static-geometry reprojection/cycle diagnostic",
        "is_independently_predicted_optical_flow": False,
        "requires_gt_images": False,
        "coordinate_space": "DGGT native render geometry and generated gauge intrinsics",
        "depth_boundary": "render_identity",
        "semantics": (
            "camera-induced forward reprojection from generated D_t; generated D_t+1 "
            "supports the inverse reprojection flow; their cycle and z-depth residual are measured"
        ),
        "limitation": (
            "No independent optical-flow/scene-flow/correspondence head exists; this diagnoses "
            "static generated-depth/camera consistency and does not score dynamic-object motion."
        ),
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "image_hw": [height, width],
        "sample_stride": stride,
        "sampled_pixels_per_pair": sampled_pixels_per_pair,
        "pair_count": all_pair_slots,
        "supported_pair_count": supported_pair_count,
        "informative_pair_count": informative_pair_count,
        "degenerate_pair_count": all_pair_slots - informative_pair_count,
        "pair_status_counts": reason_counts,
        "support": {
            "sampled_count": sampled_total,
            "source_static_valid_count": total_source,
            "projectable_count": total_projectable,
            "target_valid_count_pre_occlusion": total_raw,
            "occluded_count": total_occluded,
            "visible_count": total_visible,
            "metric_support_count": total_metric,
            "metric_support_fraction_of_sampled": (
                total_metric / float(sampled_total) if sampled_total else 0.0
            ),
            "metric_support_fraction_of_source_valid": (
                total_metric / float(total_source) if total_source else 0.0
            ),
            "occluded_fraction_of_target_valid": (
                total_occluded / float(total_raw) if total_raw else 0.0
            ),
        },
        "thresholds": {
            "depth_min": float(depth_min),
            "sky_probability": float(sky_probability_threshold),
            "dynamic_probability": float(dynamic_probability_threshold),
            "occlusion_relative_tolerance": float(occlusion_relative_tolerance),
            "min_pair_support_pixels": int(min_pair_support_pixels),
            "min_informative_motion_px": float(min_informative_motion_px),
        },
        "all_supported_metrics": all_metrics,
        "informative_metrics": informative_metrics,
        "pairs": pairs,
    }


__all__ = [
    "CAMERA_GEOMETRY_FLOW_DIAGNOSTIC_SCHEMA",
    "camera_geometry_flow_consistency",
]
