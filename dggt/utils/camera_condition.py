from __future__ import annotations

import torch

CAMERA_POSE_SUMMARY_DIM = 20
CAMERA_CONDITION_REPRESENTATION = "waymo_rel_delta_rot6d_fov20d_direct_v2"


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


def _camera_summary_from_c2w(
    camera_to_world: torch.Tensor,
    fov: torch.Tensor,
    *,
    translation_scale: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = _to_batched_sequence(camera_to_world.float(), last_dims=3, name="camera_to_world")
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must end in [4,4], got {tuple(c2w.shape)}")
    b, s = int(c2w.shape[0]), int(c2w.shape[1])
    fov = _to_batched_sequence(fov.float(), last_dims=2, name="fov")
    if fov.shape != (b, s, 2):
        raise ValueError(f"fov must be [B,S,2], got {tuple(fov.shape)} for B={b}, S={s}")

    c0_inv = _invert_se3(c2w[:, :1]).expand(-1, s, -1, -1)
    rel = c0_inv @ c2w

    prev = torch.cat([c2w[:, :1], c2w[:, :-1]], dim=1)
    delta = _invert_se3(prev) @ c2w

    scale = max(float(translation_scale), 1e-6)
    rel_t = rel[..., :3, 3] / scale
    delta_t = delta[..., :3, 3] / scale
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
    translation_scale: float = 10.0,
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
    return _camera_summary_from_c2w(c2w, fov, translation_scale=translation_scale)
