from __future__ import annotations

from collections.abc import Callable

import torch

from dggt.utils.camera_generation import camera_state_from_waymo_c2w

CAMERA_POSE_SUMMARY_DIM = 20
CAMERA_CONDITION_REPRESENTATION = "waymo_metric_rel_delta_rot6d_fov20d_stats_v3"


def fov_from_intrinsics(intrinsics: torch.Tensor, image_size_hw) -> torch.Tensor:
    """Waymo-only principal-point-aware ``[FOVx,FOVy]`` in radians."""
    if image_size_hw is None:
        raise ValueError("raw image size is required for Waymo camera FOV")
    Ks = intrinsics.float()
    squeezed = Ks.ndim == 3
    if squeezed:
        Ks = Ks.unsqueeze(0)
    if Ks.ndim != 4 or Ks.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must be [B,S,3,3] or [S,3,3], got {tuple(intrinsics.shape)}")
    b, s = Ks.shape[:2]
    hw = torch.as_tensor(image_size_hw, device=Ks.device, dtype=Ks.dtype)
    if hw.shape == (2,):
        hw = hw.view(1, 1, 2).expand(b, s, 2)
    elif hw.ndim == 2 and hw.shape == (b, 2):
        hw = hw[:, None].expand(b, s, 2)
    elif hw.ndim == 2 and hw.shape == (s, 2):
        hw = hw[None].expand(b, s, 2)
    elif hw.ndim == 3 and hw.shape[-1] == 2:
        hw = hw.expand(b, s, 2)
    else:
        raise ValueError(f"image_size_hw cannot map to B={b}, S={s}: {tuple(hw.shape)}")
    height, width = hw.unbind(-1)
    fx, fy = Ks[..., 0, 0], Ks[..., 1, 1]
    cx, cy = Ks[..., 0, 2], Ks[..., 1, 2]
    values = torch.stack((fx, fy, cx, cy, height, width), dim=-1)
    if not bool(torch.isfinite(values).all()) or bool((fx <= 0).any()) or bool((fy <= 0).any()):
        raise ValueError("Waymo intrinsics/image size must be finite with positive focal lengths")
    if bool(((cx < 0) | (cx > width) | (cy < 0) | (cy > height)).any()):
        raise ValueError("Waymo principal point must lie inside the raw image bounds")
    result = torch.stack(
        (torch.atan2(cx, fx) + torch.atan2(width - cx, fx),
         torch.atan2(cy, fy) + torch.atan2(height - cy, fy)),
        dim=-1,
    )
    return result.squeeze(0) if squeezed else result


def normalize_front_image_hw(image_hw) -> tuple[int, int] | None:
    if image_hw is None:
        return None
    if torch.is_tensor(image_hw):
        if int(image_hw.numel()) < 2:
            return None
        x = image_hw.detach().cpu().to(dtype=torch.long)
        if x.ndim == 1:
            return int(x[0].item()), int(x[1].item())
        row = x.reshape(-1, x.shape[-1])[0]
        return int(row[0].item()), int(row[1].item())
    if isinstance(image_hw, (list, tuple)):
        if len(image_hw) < 2:
            return None
        first = image_hw[0]
        if isinstance(first, (list, tuple)) or torch.is_tensor(first):
            return normalize_front_image_hw(first)
        return int(image_hw[0]), int(image_hw[1])
    return None


def _rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Return the first two rotation columns as a continuous 6D rotation code."""
    return matrix[..., :3, :2].transpose(-1, -2).reshape(matrix.shape[:-2] + (6,))


def _invert_se3(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected SE3 matrices [...,4,4], got {tuple(matrix.shape)}")
    rot = matrix[..., :3, :3]
    trans = matrix[..., :3, 3]
    rot_inv = rot.transpose(-1, -2)
    trans_inv = -(rot_inv @ trans.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(matrix)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = trans_inv
    out[..., 3, 3] = 1.0
    return out


def _to_batched_sequence(x: torch.Tensor, *, last_dims: int, name: str) -> torch.Tensor:
    if x.ndim == last_dims:
        return x.unsqueeze(0)
    if x.ndim == last_dims + 1:
        return x
    raise ValueError(f"{name} must be [S,...] or [B,S,...], got {tuple(x.shape)}")


def _to_batched_pose(
    value: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    pose = torch.as_tensor(value, device=device, dtype=dtype)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    elif pose.ndim == 4 and int(pose.shape[1]) == 1:
        pose = pose[:, 0]
    if pose.ndim != 3 or tuple(pose.shape[-2:]) != (4, 4):
        raise ValueError(
            f"{name} must be [4,4], [B,4,4], or [B,1,4,4], got {tuple(pose.shape)}"
        )
    if int(pose.shape[0]) == 1 and batch_size > 1:
        pose = pose.expand(batch_size, -1, -1)
    if int(pose.shape[0]) != batch_size:
        raise ValueError(f"{name} batch {pose.shape[0]} != camera batch {batch_size}")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError(f"{name} contains non-finite values")
    return pose


def _camera_summary_from_c2w(
    camera_to_world: torch.Tensor,
    fov: torch.Tensor,
    *,
    trajectory_anchor_to_world: torch.Tensor | None = None,
    previous_camera_to_world: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = _to_batched_sequence(camera_to_world.float(), last_dims=3, name="camera_to_world")
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must end in [4,4], got {tuple(c2w.shape)}")
    b, s = int(c2w.shape[0]), int(c2w.shape[1])
    fov = _to_batched_sequence(fov.float(), last_dims=2, name="fov")
    if fov.shape != (b, s, 2):
        raise ValueError(f"fov must be [B,S,2], got {tuple(fov.shape)} for B={b}, S={s}")

    if trajectory_anchor_to_world is None:
        anchor = c2w[:, :1]
    else:
        anchor = _to_batched_pose(
            trajectory_anchor_to_world,
            batch_size=b,
            device=c2w.device,
            dtype=c2w.dtype,
            name="trajectory_anchor_to_world",
        ).unsqueeze(1)
    c0_inv = _invert_se3(anchor).expand(-1, s, -1, -1)
    rel = c0_inv @ c2w

    if previous_camera_to_world is None:
        prev = torch.cat([c2w[:, :1], c2w[:, :-1]], dim=1)
    else:
        prev = _to_batched_sequence(
            previous_camera_to_world.float(), last_dims=3, name="previous_camera_to_world"
        )
        if int(prev.shape[0]) == 1 and b > 1:
            prev = prev.expand(b, -1, -1, -1)
        if prev.shape != c2w.shape:
            raise ValueError(
                f"previous_camera_to_world shape {tuple(prev.shape)} != camera shape {tuple(c2w.shape)}"
            )
    delta = _invert_se3(prev) @ c2w

    rel_t = rel[..., :3, 3]
    delta_t = delta[..., :3, 3]
    features = torch.cat(
        [
            rel_t,
            _rotation_6d(rel[..., :3, :3]),
            delta_t,
            _rotation_6d(delta[..., :3, :3]),
            fov,
        ],
        dim=-1,
    )
    finite = torch.isfinite(features).all(dim=-1)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, finite


def _select_front_camera_to_world(camera_to_world: torch.Tensor, intrinsics: torch.Tensor | None = None) -> torch.Tensor:
    c2w = camera_to_world.float()
    if c2w.ndim == 5 and c2w.shape[-2:] == (4, 4):
        return c2w[:, :, 0]
    if c2w.ndim == 4 and c2w.shape[-2:] == (4, 4):
        if torch.is_tensor(intrinsics) and intrinsics.ndim >= 4 and intrinsics.shape[-2:] == (3, 3):
            if int(intrinsics.shape[0]) == int(c2w.shape[0]) and int(intrinsics.shape[1]) == int(c2w.shape[1]):
                return c2w
        if int(c2w.shape[0]) == 1 and int(c2w.shape[1]) != 1:
            return c2w
        return c2w[:, 0].unsqueeze(0)
    if c2w.ndim == 3 and c2w.shape[-2:] == (4, 4):
        return c2w.unsqueeze(0)
    raise ValueError(f"Unsupported camera_to_world shape {tuple(c2w.shape)}")


def _select_front_intrinsics(intrinsics: torch.Tensor, *, batch_size: int, seq_len: int) -> torch.Tensor:
    Ks = intrinsics.float()
    if Ks.ndim == 5 and Ks.shape[-2:] == (3, 3):
        Ks = Ks[:, :, 0]
    if Ks.ndim == 4 and Ks.shape[-2:] == (3, 3):
        if int(Ks.shape[0]) == batch_size and int(Ks.shape[1]) == seq_len:
            pass
        elif int(Ks.shape[0]) == batch_size and int(Ks.shape[1]) == 1:
            Ks = Ks[:, 0].unsqueeze(1).expand(-1, seq_len, -1, -1)
        elif int(Ks.shape[0]) == 1 and int(Ks.shape[1]) == seq_len:
            Ks = Ks.expand(batch_size, -1, -1, -1)
        elif int(Ks.shape[0]) == seq_len:
            Ks = Ks[:, 0].unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif int(Ks.shape[0]) == batch_size:
            Ks = Ks[:, 0].unsqueeze(1).expand(-1, seq_len, -1, -1)
        else:
            Ks = Ks[0, 0].view(1, 1, 3, 3).expand(batch_size, seq_len, -1, -1)
    elif Ks.ndim == 3 and Ks.shape[-2:] == (3, 3):
        if int(Ks.shape[0]) == batch_size and batch_size != 1:
            Ks = Ks.unsqueeze(1).expand(-1, seq_len, -1, -1)
        elif int(Ks.shape[0]) == seq_len:
            Ks = Ks.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            Ks = Ks[0].view(1, 1, 3, 3).expand(batch_size, seq_len, -1, -1)
    elif Ks.ndim == 2 and Ks.shape == (3, 3):
        Ks = Ks.view(1, 1, 3, 3).expand(batch_size, seq_len, -1, -1)
    else:
        raise ValueError(f"Unsupported intrinsics shape {tuple(Ks.shape)}")
    if Ks.shape[:2] != (batch_size, seq_len):
        raise ValueError(f"Selected intrinsics shape {tuple(Ks.shape)} != {(batch_size, seq_len, 3, 3)}")
    return Ks


def camera_summary_from_waymo_gt(
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_hw: tuple[int, int] | None = None,
    trajectory_anchor_to_world: torch.Tensor | None = None,
    previous_camera_to_world: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build camera summary tokens from dataset GT camera trajectory.

    Waymo edit/precompute caches store cameras with an explicit view dimension;
    SceneFlow currently trains on the front camera, so this helper selects view 0
    and returns one camera token per video frame.
    """
    c2w = _select_front_camera_to_world(camera_to_world, intrinsics)
    b, s = int(c2w.shape[0]), int(c2w.shape[1])
    Ks = _select_front_intrinsics(intrinsics, batch_size=b, seq_len=s)
    image_hw = normalize_front_image_hw(image_hw)
    fov_xy = fov_from_intrinsics(Ks, image_hw)
    fov = torch.stack((fov_xy[..., 1], fov_xy[..., 0]), dim=-1)
    return _camera_summary_from_c2w(
        c2w,
        fov,
        trajectory_anchor_to_world=trajectory_anchor_to_world,
        previous_camera_to_world=previous_camera_to_world,
    )


def camera_condition_from_waymo_metric_target(
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_hw: tuple[int, int] | None,
    trajectory_anchor_to_world: torch.Tensor,
    previous_camera_to_world: torch.Tensor | None,
    anchor_mask: torch.Tensor,
    normalize_camera: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the v3 condition and its matching v4 metric generation target.

    The 20-D condition keeps its raw anchor-relative pose and Waymo FOV, but
    channels ``9:18`` are the *same* role-aware normalized 9-D state used by
    camera generation.  Keeping this assembly in one helper prevents raw
    pretraining, formal T1, and offline/external inference from silently using
    different camera-condition coordinates.

    ``anchor_mask`` is global: a sliced window that does not contain trunk
    frame zero must be delta-only and must receive the preceding camera pose.
    """

    if not callable(normalize_camera):
        raise TypeError("normalize_camera must be callable")
    c2w = _select_front_camera_to_world(camera_to_world, intrinsics)
    batch_size, seq_len = int(c2w.shape[0]), int(c2w.shape[1])
    previous_front = None
    if previous_camera_to_world is not None:
        previous_raw = torch.as_tensor(previous_camera_to_world).float()
        if (
            previous_raw.ndim == 4
            and int(previous_raw.shape[0]) == batch_size
            and int(previous_raw.shape[1]) in (1, seq_len)
        ):
            previous_front = previous_raw
        elif previous_raw.ndim == 3 and int(previous_raw.shape[0]) == batch_size:
            previous_front = previous_raw.unsqueeze(1)
        else:
            previous_front = _select_front_camera_to_world(
                previous_raw,
                intrinsics,
            )
        if int(previous_front.shape[0]) != batch_size or int(previous_front.shape[1]) not in (
            1,
            seq_len,
        ):
            raise ValueError(
                "previous_camera_to_world must provide one preceding pose or one pose per frame: "
                f"got {tuple(previous_front.shape)} for B={batch_size}, S={seq_len}"
            )
    target_state, returned_anchor_mask = camera_state_from_waymo_c2w(
        c2w,
        trajectory_anchor_to_world,
        previous_camera_to_world=previous_front,
        anchor_mask=anchor_mask,
    )
    normalized_target = normalize_camera(target_state, returned_anchor_mask)
    if normalized_target.shape != target_state.shape:
        raise ValueError(
            "normalize_camera must preserve the metric target shape: "
            f"got {tuple(normalized_target.shape)} for {tuple(target_state.shape)}"
        )
    if not bool(torch.isfinite(normalized_target).all()):
        raise ValueError("normalized camera target contains non-finite values")
    summary_previous = previous_front
    if summary_previous is not None and int(summary_previous.shape[1]) == 1 and seq_len > 1:
        summary_previous = torch.cat((summary_previous, c2w[:, :-1]), dim=1)
    condition, valid = camera_summary_from_waymo_gt(
        c2w,
        intrinsics,
        image_hw=image_hw,
        trajectory_anchor_to_world=trajectory_anchor_to_world,
        previous_camera_to_world=summary_previous,
    )
    if condition.shape[:-1] != normalized_target.shape[:-1]:
        raise ValueError(
            "camera summary and generation target disagree on batch/time shape: "
            f"{tuple(condition.shape)} vs {tuple(normalized_target.shape)}"
        )
    condition = condition.clone()
    condition[..., 9:18] = normalized_target.to(
        device=condition.device,
        dtype=condition.dtype,
    )
    return condition, valid, target_state, returned_anchor_mask
