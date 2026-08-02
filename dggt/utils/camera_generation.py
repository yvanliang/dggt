"""Metric Waymo geometry for Cosmos-style SceneFlow camera generation.

The generated token is deliberately small and interpretable.  Frame zero is an
absolute camera-to-trajectory-anchor pose; every later frame is the adjacent
camera motion ``inv(c2w[t-1]) @ c2w[t]``.  Translations are metres.  FOV belongs
to the separate scene-global gauge token.  Rotations use the continuous
six-dimensional representation from Zhou et al. and are projected onto SO(3)
during decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from dggt.utils.rotation import quat_to_mat


CAMERA_GENERATION_REPRESENTATION = "waymo_metric_relative_se3_rot6d_v4"
CAMERA_GENERATION_DIM = 9
CAMERA_STATS_VERSION = "waymo_metric_camera_anchor_delta_per_channel_v5_global_context"
CAMERA_TARGET_SPACE = "waymo_metric_camera_to_world"
CAMERA_TARGET_SOURCE = "waymo_gt_extrinsics"
CAMERA_STATS_STD_FLOOR = 1.0e-4


def invert_se3(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"SE(3) matrix must end in [4,4], got {tuple(matrix.shape)}")
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    out = torch.zeros_like(matrix)
    out[..., :3, :3] = rotation.transpose(-1, -2)
    out[..., :3, 3] = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def rotation_matrix_to_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Encode the first two matrix columns as ``[r1_xyz, r2_xyz]``."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in [3,3], got {tuple(matrix.shape)}")
    return matrix[..., :3, :2].transpose(-1, -2).reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(rotation_6d: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Project any finite 6D code to a finite, right-handed SO(3) matrix.

    In particular, the all-zero code maps to identity.  A plain normalized
    Gram--Schmidt implementation leaves its first column at zero, which makes
    the generated matrix singular precisely at the flow model's common zero
    initialization.
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"rotation_6d must end in 6 channels, got {tuple(rotation_6d.shape)}")
    if not bool(torch.isfinite(rotation_6d).all()):
        raise ValueError("rotation_6d must contain only finite values")
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    e1 = torch.zeros_like(a1)
    e1[..., 0] = 1.0
    b1 = F.normalize(torch.where(a1.norm(dim=-1, keepdim=True) < eps, e1, a1), dim=-1, eps=eps)
    b2_residual = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    # Degenerate network outputs still need a finite, right-handed projection.
    fallback_axis = torch.zeros_like(b1)
    min_axis = b1.abs().argmin(dim=-1, keepdim=True)
    fallback_axis.scatter_(-1, min_axis, 1.0)
    fallback = fallback_axis - (fallback_axis * b1).sum(dim=-1, keepdim=True) * b1
    use_fallback = b2_residual.norm(dim=-1, keepdim=True) < eps
    b2 = F.normalize(torch.where(use_fallback, fallback, b2_residual), dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _as_batched_sequence(x: torch.Tensor, trailing_ndim: int, name: str) -> tuple[torch.Tensor, bool]:
    if x.ndim == trailing_ndim + 1:
        return x, False
    if x.ndim == trailing_ndim:
        return x.unsqueeze(0), True
    raise ValueError(f"{name} has unsupported shape {tuple(x.shape)}")


def camera_anchor_mask(batch_size: int, seq_len: int, *, device: torch.device | None = None) -> torch.Tensor:
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    mask = torch.zeros((int(batch_size), int(seq_len)), device=device, dtype=torch.bool)
    mask[:, 0] = True
    return mask


def camera_to_world_from_dggt_pose_enc(pose_enc_dggt: torch.Tensor) -> torch.Tensor:
    """Convert DGGT CameraHead pose encodings to camera-to-world matrices."""
    pose, squeezed = _as_batched_sequence(pose_enc_dggt.float(), 2, "pose_enc_dggt")
    if pose.shape[-1] != 9:
        raise ValueError(f"DGGT pose_enc must be [B,S,9], got {tuple(pose.shape)}")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError("DGGT pose_enc contains non-finite values")
    quat = pose[..., 3:7]
    if bool((quat.norm(dim=-1) < 1.0e-8).any()):
        raise ValueError("DGGT pose_enc contains a zero quaternion")
    b, s = int(pose.shape[0]), int(pose.shape[1])
    w2c = torch.zeros((b, s, 4, 4), device=pose.device, dtype=pose.dtype)
    w2c[..., :3, :3] = quat_to_mat(F.normalize(quat, dim=-1))
    w2c[..., :3, 3] = pose[..., :3]
    w2c[..., 3, 3] = 1.0
    c2w = invert_se3(w2c)
    return c2w.squeeze(0) if squeezed else c2w


def _as_batched_pose(value: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:
    pose = torch.as_tensor(value).float()
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    elif pose.ndim == 4 and int(pose.shape[1]) == 1:
        pose = pose[:, 0]
    if pose.ndim != 3 or tuple(pose.shape[-2:]) != (4, 4):
        raise ValueError(f"{name} must be [4,4], [B,4,4], or [B,1,4,4], got {tuple(pose.shape)}")
    if int(pose.shape[0]) == 1 and batch_size > 1:
        pose = pose.expand(batch_size, -1, -1)
    if int(pose.shape[0]) != batch_size:
        raise ValueError(f"{name} batch {pose.shape[0]} != camera batch {batch_size}")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError(f"{name} contains non-finite values")
    return pose


def camera_state_from_waymo_c2w(
    camera_to_world: torch.Tensor,
    anchor_to_world: torch.Tensor,
    *,
    previous_camera_to_world: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode deterministic Waymo camera GT as a 9D metric trajectory.

    The first token is ``inv(anchor_to_world) @ camera_to_world[0]`` and every
    later token is the adjacent transform.  This makes the target exactly the
    metric delta half of the Waymo camera condition; no frozen DGGT gauge enters
    the target construction.
    """

    c2w, squeezed = _as_batched_sequence(camera_to_world.float(), 3, "camera_to_world")
    if tuple(c2w.shape[-2:]) != (4, 4):
        raise ValueError(f"camera_to_world must end in [4,4], got {tuple(c2w.shape)}")
    if not bool(torch.isfinite(c2w).all()):
        raise ValueError("camera_to_world contains non-finite values")
    b, s = int(c2w.shape[0]), int(c2w.shape[1])
    anchor = _as_batched_pose(anchor_to_world, batch_size=b, name="anchor_to_world").to(
        device=c2w.device, dtype=c2w.dtype
    )
    anchor_relative = invert_se3(anchor).unsqueeze(1) @ c2w
    relative = anchor_relative.clone()
    if s > 1:
        relative[:, 1:] = invert_se3(c2w[:, :-1]) @ c2w[:, 1:]
    anchors = camera_anchor_mask(b, s, device=c2w.device) if anchor_mask is None else torch.as_tensor(
        anchor_mask, device=c2w.device, dtype=torch.bool
    )
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    if tuple(anchors.shape) != (b, s):
        raise ValueError(f"anchor_mask shape {tuple(anchors.shape)} != {(b, s)}")
    if not bool(((anchors.sum(dim=1) == 0) | ((anchors.sum(dim=1) == 1) & anchors[:, 0])).all()):
        raise ValueError("each camera window must be complete (anchor at local 0) or delta-only")
    delta_only = anchors.sum(dim=1) == 0
    if bool(delta_only.any()):
        if previous_camera_to_world is None:
            raise ValueError("delta-only camera targets require previous_camera_to_world")
        previous = torch.as_tensor(
            previous_camera_to_world, device=c2w.device, dtype=c2w.dtype
        )
        if previous.ndim == 4 and tuple(previous.shape[:2]) == (b, s):
            previous = previous[:, 0]
        elif squeezed and previous.ndim == 3 and tuple(previous.shape) == (s, 4, 4):
            previous = previous[0]
        previous = _as_batched_pose(
            previous, batch_size=b, name="previous_camera_to_world"
        ).to(device=c2w.device, dtype=c2w.dtype)
        first_delta = invert_se3(previous) @ c2w[:, 0]
        relative[:, 0] = torch.where(delta_only.view(b, 1, 1), first_delta, relative[:, 0])
    state = torch.cat(
        (relative[..., :3, 3], rotation_matrix_to_6d(relative[..., :3, :3])),
        dim=-1,
    )
    if squeezed:
        return state.squeeze(0), anchors.squeeze(0)
    return state, anchors


@dataclass
class DecodedCameraTrajectory:
    camera_to_world: torch.Tensor
    world_to_camera: torch.Tensor


def decode_camera_trajectory(
    state: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    initial_camera_to_world: torch.Tensor | None = None,
    trajectory_anchor_to_world: torch.Tensor | None = None,
) -> DecodedCameraTrajectory:
    """Integrate a complete or globally sliced v4 metric camera trajectory.

    A complete clip has its unique anchor at local token zero.  A window that
    starts later contains deltas only and must provide the global previous
    frame's camera-to-world matrix.  Local window starts are never promoted to
    anchors.
    """
    tokens, squeezed = _as_batched_sequence(state.float(), 2, "camera state")
    if tokens.shape[-1] != CAMERA_GENERATION_DIM:
        raise ValueError(f"camera state dim must be {CAMERA_GENERATION_DIM}, got {tokens.shape[-1]}")
    b, s = int(tokens.shape[0]), int(tokens.shape[1])
    if anchor_mask is None:
        raise ValueError("global camera_anchor_mask is required to decode a metric camera trajectory")
    anchors = torch.as_tensor(anchor_mask, device=tokens.device, dtype=torch.bool)
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    if anchors.shape != (b, s):
        raise ValueError(f"camera anchor mask shape {tuple(anchors.shape)} != {(b, s)}")
    anchor_counts = anchors.sum(dim=1)
    complete_rows = (anchor_counts == 1) & anchors[:, 0]
    delta_only_rows = anchor_counts == 0
    if not bool((complete_rows | delta_only_rows).all()):
        raise ValueError(
            "camera windows must either contain the one global anchor at local frame 0 "
            "or contain deltas only"
        )
    anchor = None
    if trajectory_anchor_to_world is not None:
        anchor = _as_batched_pose(
            trajectory_anchor_to_world,
            batch_size=b,
            name="trajectory_anchor_to_world",
        ).to(device=tokens.device, dtype=tokens.dtype)
    initial = None
    if bool(delta_only_rows.any()):
        if initial_camera_to_world is None:
            raise ValueError(
                "delta-only camera windows require initial_camera_to_world from the global previous frame"
            )
        initial = _as_batched_pose(
            initial_camera_to_world,
            batch_size=b,
            name="initial_camera_to_world",
        ).to(device=tokens.device, dtype=tokens.dtype)
        if anchor is not None:
            initial = invert_se3(anchor) @ initial
    relative = torch.zeros((b, s, 4, 4), device=tokens.device, dtype=tokens.dtype)
    relative[..., :3, :3] = rotation_6d_to_matrix(tokens[..., 3:9])
    relative[..., :3, 3] = tokens[..., :3]
    relative[..., 3, 3] = 1
    first = relative[:, 0]
    if initial is not None:
        first = torch.where(
            delta_only_rows.view(b, 1, 1),
            initial @ relative[:, 0],
            first,
        )
    c2w_frames = [first]
    for index in range(1, s):
        c2w_frames.append(c2w_frames[-1] @ relative[:, index])
    camera_to_reference = torch.stack(c2w_frames, dim=1)
    c2w = camera_to_reference if anchor is None else anchor.unsqueeze(1) @ camera_to_reference
    w2c = invert_se3(c2w)
    if squeezed:
        return DecodedCameraTrajectory(c2w.squeeze(0), w2c.squeeze(0))
    return DecodedCameraTrajectory(c2w, w2c)


def so3_geodesic_angle(rotation_a: torch.Tensor, rotation_b: torch.Tensor) -> torch.Tensor:
    if rotation_a.shape != rotation_b.shape or rotation_a.shape[-2:] != (3, 3):
        raise ValueError(f"rotation shapes must match [...,3,3], got {rotation_a.shape} and {rotation_b.shape}")
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    # atan2 is accurate and has a useful gradient near zero.
    skew = torch.stack(
        (relative[..., 2, 1] - relative[..., 1, 2],
         relative[..., 0, 2] - relative[..., 2, 0],
         relative[..., 1, 0] - relative[..., 0, 1]),
        dim=-1,
    )
    sine = 0.5 * skew.norm(dim=-1)
    return torch.atan2(sine, cosine)


def _differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def camera_geometry_loss(
    predicted_state: torch.Tensor,
    target_state: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    initial_camera_to_world: torch.Tensor | None = None,
    trajectory_anchor_to_world: torch.Tensor | None = None,
    absolute_weight: float = 1.0,
    relative_weight: float = 1.0,
    smoothness_weight: float = 0.25,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pose/trajectory loss on the fully reconstructed camera clip.

    The absolute/relative/smoothness weights select the trajectory objective,
    while the translation/rotation weights select physical components
    consistently across those objectives.
    """
    if predicted_state.shape != target_state.shape or predicted_state.shape[-1] != CAMERA_GENERATION_DIM:
        raise ValueError(f"camera states must match [B,S,{CAMERA_GENERATION_DIM}]")
    pred = decode_camera_trajectory(
        predicted_state,
        anchor_mask,
        initial_camera_to_world=initial_camera_to_world,
        trajectory_anchor_to_world=trajectory_anchor_to_world,
    )
    target = decode_camera_trajectory(
        target_state,
        anchor_mask,
        initial_camera_to_world=initial_camera_to_world,
        trajectory_anchor_to_world=trajectory_anchor_to_world,
    )
    p_c2w, t_c2w = pred.camera_to_world, target.camera_to_world
    if p_c2w.ndim == 3:
        p_c2w, t_c2w = p_c2w.unsqueeze(0), t_c2w.unsqueeze(0)
    zero = _differentiable_zero(predicted_state)
    abs_t = F.smooth_l1_loss(p_c2w[..., :3, 3], t_c2w[..., :3, 3])
    abs_r = so3_geodesic_angle(p_c2w[..., :3, :3], t_c2w[..., :3, :3]).mean()
    anchors = torch.as_tensor(anchor_mask, device=predicted_state.device, dtype=torch.bool)
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    delta_mask = ~anchors
    if bool(delta_mask.any()):
        rel_t_error = F.smooth_l1_loss(
            predicted_state[..., :3], target_state[..., :3], reduction="none"
        ).mean(dim=-1)
        rel_t = rel_t_error[delta_mask].mean()
        pred_rel_r = rotation_6d_to_matrix(predicted_state[..., 3:9])
        target_rel_r = rotation_6d_to_matrix(target_state[..., 3:9])
        rel_r = so3_geodesic_angle(pred_rel_r, target_rel_r)[delta_mask].mean()
    else:
        rel_t = rel_r = zero
    if predicted_state.shape[-2] >= 3:
        pred_dt = p_c2w[..., 1:, :3, 3] - p_c2w[..., :-1, :3, 3]
        target_dt = t_c2w[..., 1:, :3, 3] - t_c2w[..., :-1, :3, 3]
        accel_t = F.smooth_l1_loss(pred_dt[..., 1:, :] - pred_dt[..., :-1, :], target_dt[..., 1:, :] - target_dt[..., :-1, :])
        pred_dr = p_c2w[..., :-1, :3, :3].transpose(-1, -2) @ p_c2w[..., 1:, :3, :3]
        target_dr = t_c2w[..., :-1, :3, :3].transpose(-1, -2) @ t_c2w[..., 1:, :3, :3]
        pred_dr2 = pred_dr[..., :-1, :, :].transpose(-1, -2) @ pred_dr[..., 1:, :, :]
        target_dr2 = target_dr[..., :-1, :, :].transpose(-1, -2) @ target_dr[..., 1:, :, :]
        accel_r = so3_geodesic_angle(pred_dr2, target_dr2).mean()
    else:
        accel_t = accel_r = zero
    absolute = (
        float(translation_weight) * abs_t
        + float(rotation_weight) * abs_r
    )
    relative = float(translation_weight) * rel_t + float(rotation_weight) * rel_r
    smoothness = (
        float(translation_weight) * accel_t
        + float(rotation_weight) * accel_r
    )
    total = float(absolute_weight) * absolute + float(relative_weight) * relative + float(smoothness_weight) * smoothness
    metrics = {
        "camera_absolute_translation": abs_t,
        "camera_absolute_rotation_rad": abs_r,
        "camera_relative_translation": rel_t,
        "camera_relative_rotation_rad": rel_r,
        "camera_acceleration_translation": accel_t,
        "camera_acceleration_rotation_rad": accel_r,
    }
    return total, metrics


def normalize_camera_state(
    state: torch.Tensor,
    anchor_mask: torch.Tensor,
    anchor_mean: torch.Tensor,
    anchor_std: torch.Tensor,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
) -> torch.Tensor:
    mask = anchor_mask.to(device=state.device, dtype=torch.bool).unsqueeze(-1)
    am = anchor_mean.to(device=state.device, dtype=state.dtype)
    ast = anchor_std.to(device=state.device, dtype=state.dtype).clamp_min(CAMERA_STATS_STD_FLOOR)
    dm = delta_mean.to(device=state.device, dtype=state.dtype)
    dst = delta_std.to(device=state.device, dtype=state.dtype).clamp_min(CAMERA_STATS_STD_FLOOR)
    return torch.where(mask, (state - am) / ast, (state - dm) / dst)


def denormalize_camera_state(
    normalized: torch.Tensor,
    anchor_mask: torch.Tensor,
    anchor_mean: torch.Tensor,
    anchor_std: torch.Tensor,
    delta_mean: torch.Tensor,
    delta_std: torch.Tensor,
) -> torch.Tensor:
    mask = anchor_mask.to(device=normalized.device, dtype=torch.bool).unsqueeze(-1)
    am = anchor_mean.to(device=normalized.device, dtype=normalized.dtype)
    ast = anchor_std.to(device=normalized.device, dtype=normalized.dtype).clamp_min(CAMERA_STATS_STD_FLOOR)
    dm = delta_mean.to(device=normalized.device, dtype=normalized.dtype)
    dst = delta_std.to(device=normalized.device, dtype=normalized.dtype).clamp_min(CAMERA_STATS_STD_FLOOR)
    return torch.where(mask, normalized * ast + am, normalized * dst + dm)
