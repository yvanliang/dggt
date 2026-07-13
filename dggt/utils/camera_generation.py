"""DGGT-space geometry for Cosmos-style SceneFlow camera generation.

The generated token is deliberately small and interpretable.  Frame zero is an
absolute camera-to-world pose; every later frame is the adjacent camera motion
``inv(c2w[t-1]) @ c2w[t]``.  FOV remains absolute for every frame.  Rotations
use the continuous six-dimensional representation from Zhou et al. and are
projected onto SO(3) during decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from dggt.utils.rotation import mat_to_quat, quat_to_mat


CAMERA_GENERATION_REPRESENTATION = "dggt_relative_se3_rot6d_logfov_v3"
CAMERA_GENERATION_DIM = 11
CAMERA_STATS_VERSION = "dggt_camera_anchor_delta_per_channel_v3"
CAMERA_TARGET_SPACE = "dggt_camera_head_pose_enc"
CAMERA_TARGET_SOURCE = "frozen_dggt_camera_head"
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


def log_tan_half_fov(fov_xy: torch.Tensor) -> torch.Tensor:
    if fov_xy.shape[-1] != 2:
        raise ValueError(f"FOV must end in two channels, got {tuple(fov_xy.shape)}")
    if not bool(torch.isfinite(fov_xy).all()) or bool(((fov_xy <= 0) | (fov_xy >= torch.pi)).any()):
        raise ValueError("horizontal/vertical FOV must be finite and in (0, pi)")
    return torch.log(torch.tan(0.5 * fov_xy))


def fov_from_log_tan(log_fov_xy: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.atan(torch.exp(log_fov_xy))


def camera_anchor_mask(batch_size: int, seq_len: int, *, device: torch.device | None = None) -> torch.Tensor:
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    mask = torch.zeros((int(batch_size), int(seq_len)), device=device, dtype=torch.bool)
    mask[:, 0] = True
    return mask


def camera_state_from_dggt_pose_enc(pose_enc_dggt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode frozen DGGT CameraHead ``[t_w2c,q_xyzw,FOVy,FOVx]`` to 11D.

    This is intentionally the sole public target-construction entry point.
    Waymo extrinsics/intrinsics belong to the independent 20D conditioning
    path and must never be converted into this target space.
    """
    pose, squeezed = _as_batched_sequence(pose_enc_dggt.float(), 2, "pose_enc_dggt")
    if pose.shape[-1] != 9:
        raise ValueError(f"DGGT pose_enc must be [B,S,9], got {tuple(pose.shape)}")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError("DGGT pose_enc contains non-finite values")
    if bool(((pose[..., 7:9] <= 0) | (pose[..., 7:9] >= torch.pi)).any()):
        raise ValueError("DGGT FOVy/FOVx must be in (0, pi)")
    quat = pose[..., 3:7]
    if bool((quat.norm(dim=-1) < 1.0e-8).any()):
        raise ValueError("DGGT pose_enc contains a zero quaternion")
    b, s = int(pose.shape[0]), int(pose.shape[1])
    w2c = torch.zeros((b, s, 4, 4), device=pose.device, dtype=pose.dtype)
    w2c[..., :3, :3] = quat_to_mat(F.normalize(quat, dim=-1))
    w2c[..., :3, 3] = pose[..., :3]
    w2c[..., 3, 3] = 1.0
    c2w = invert_se3(w2c)
    relative = c2w.clone()
    if s > 1:
        relative[:, 1:] = invert_se3(c2w[:, :-1]) @ c2w[:, 1:]
    # DGGT stores [FOVy,FOVx], while the state stores [log FOVx, log FOVy].
    fov_xy = torch.stack((pose[..., 8], pose[..., 7]), dim=-1)
    state = torch.cat(
        (
            relative[..., :3, 3],
            rotation_matrix_to_6d(relative[..., :3, :3]),
            log_tan_half_fov(fov_xy),
        ),
        dim=-1,
    )
    anchors = camera_anchor_mask(b, s, device=c2w.device)
    if squeezed:
        return state.squeeze(0), anchors.squeeze(0)
    return state, anchors


@dataclass
class DecodedCameraTrajectory:
    camera_to_world: torch.Tensor
    world_to_camera: torch.Tensor
    pose_encoding: torch.Tensor
    fov_xy: torch.Tensor


def decode_camera_trajectory(
    state: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> DecodedCameraTrajectory:
    """Integrate a complete v3 trajectory and restore DGGT-space pose_enc."""
    tokens, squeezed = _as_batched_sequence(state.float(), 2, "camera state")
    if tokens.shape[-1] != CAMERA_GENERATION_DIM:
        raise ValueError(f"camera state dim must be {CAMERA_GENERATION_DIM}, got {tokens.shape[-1]}")
    b, s = int(tokens.shape[0]), int(tokens.shape[1])
    anchors = torch.as_tensor(anchor_mask, device=tokens.device, dtype=torch.bool)
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0)
    if anchors.shape != (b, s):
        raise ValueError(f"camera anchor mask shape {tuple(anchors.shape)} != {(b, s)}")
    expected = camera_anchor_mask(b, s, device=tokens.device)
    if not torch.equal(anchors, expected):
        raise ValueError(
            "a complete camera trajectory must have exactly one global anchor at frame 0; "
            "do not promote sliding-window first frames to anchors"
        )
    relative = torch.zeros((b, s, 4, 4), device=tokens.device, dtype=tokens.dtype)
    relative[..., :3, :3] = rotation_6d_to_matrix(tokens[..., 3:9])
    relative[..., :3, 3] = tokens[..., :3]
    relative[..., 3, 3] = 1
    c2w_frames = [relative[:, 0]]
    for index in range(1, s):
        c2w_frames.append(c2w_frames[-1] @ relative[:, index])
    c2w = torch.stack(c2w_frames, dim=1)
    w2c = invert_se3(c2w)
    fov_xy = fov_from_log_tan(tokens[..., 9:11])
    # DGGT pose encoding is [w2c translation, xyzw quaternion, vertical FOV, horizontal FOV].
    quaternion = mat_to_quat(w2c[..., :3, :3])
    pose = torch.cat((w2c[..., :3, 3], quaternion, fov_xy[..., 1:2], fov_xy[..., 0:1]), dim=-1)
    if squeezed:
        return DecodedCameraTrajectory(c2w.squeeze(0), w2c.squeeze(0), pose.squeeze(0), fov_xy.squeeze(0))
    return DecodedCameraTrajectory(c2w, w2c, pose, fov_xy)


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
    absolute_weight: float = 1.0,
    relative_weight: float = 1.0,
    smoothness_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pose/trajectory loss on the fully reconstructed camera clip."""
    if predicted_state.shape != target_state.shape or predicted_state.shape[-1] != CAMERA_GENERATION_DIM:
        raise ValueError(f"camera states must match [B,S,{CAMERA_GENERATION_DIM}]")
    pred = decode_camera_trajectory(predicted_state, anchor_mask)
    target = decode_camera_trajectory(target_state, anchor_mask)
    p_c2w, t_c2w = pred.camera_to_world, target.camera_to_world
    if p_c2w.ndim == 3:
        p_c2w, t_c2w = p_c2w.unsqueeze(0), t_c2w.unsqueeze(0)
    zero = _differentiable_zero(predicted_state)
    abs_t = F.smooth_l1_loss(p_c2w[..., :3, 3], t_c2w[..., :3, 3])
    abs_r = so3_geodesic_angle(p_c2w[..., :3, :3], t_c2w[..., :3, :3]).mean()
    abs_fov = F.smooth_l1_loss(predicted_state[..., 9:11], target_state[..., 9:11])
    if predicted_state.shape[-2] >= 2:
        rel_t = F.smooth_l1_loss(predicted_state[..., 1:, :3], target_state[..., 1:, :3])
        pred_rel_r = rotation_6d_to_matrix(predicted_state[..., 1:, 3:9])
        target_rel_r = rotation_6d_to_matrix(target_state[..., 1:, 3:9])
        rel_r = so3_geodesic_angle(pred_rel_r, target_rel_r).mean()
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
        pred_fov_d = predicted_state[..., 1:, 9:11] - predicted_state[..., :-1, 9:11]
        target_fov_d = target_state[..., 1:, 9:11] - target_state[..., :-1, 9:11]
        accel_fov = F.smooth_l1_loss(pred_fov_d[..., 1:, :] - pred_fov_d[..., :-1, :], target_fov_d[..., 1:, :] - target_fov_d[..., :-1, :])
    else:
        accel_t = accel_r = accel_fov = zero
    absolute = abs_t + abs_r + abs_fov
    relative = rel_t + rel_r
    smoothness = accel_t + accel_r + accel_fov
    total = float(absolute_weight) * absolute + float(relative_weight) * relative + float(smoothness_weight) * smoothness
    metrics = {
        "camera_absolute_translation": abs_t,
        "camera_absolute_rotation_rad": abs_r,
        "camera_log_fov": abs_fov,
        "camera_relative_translation": rel_t,
        "camera_relative_rotation_rad": rel_r,
        "camera_acceleration_translation": accel_t,
        "camera_acceleration_rotation_rad": accel_r,
        "camera_acceleration_fov": accel_fov,
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
