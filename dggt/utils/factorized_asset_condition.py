"""Leak-free, factorized asset conditioning shared by training and inference.

This module deliberately has no dependency on target video latents or target
segmentation.  Appearance is represented by a canonical reference, while every
destination quantity is derived from explicit 3D boxes and cameras.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


PLACEMENT_STATE_DIM = 12
CANONICAL_CROP_MARGIN = 0.15
CANONICAL_LONG_SIDE_FRACTION = 0.80
CANONICAL_ALPHA_PATCH_THRESHOLD = 0.05
MAX_CANONICAL_APPEARANCE_TOKENS = 32
FACTORIZED_ASSET_CONDITION_VERSION = "factorized_asset_v1"
BOX_PROJECTION_NEAR_PLANE = 1.0e-3


def resize_crop_intrinsics_to_model_canvas(
    intrinsics: torch.Tensor,
    image_size_hw: torch.Tensor | Sequence[int],
    *,
    target_width: int = 518,
    patch_size: int = 14,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw pinhole intrinsics into the resized/cropped model canvas."""
    Ks = torch.as_tensor(intrinsics)
    if Ks.ndim < 2 or tuple(Ks.shape[-2:]) != (3, 3):
        raise ValueError(
            f"intrinsics must end in [3,3], got {tuple(Ks.shape)}"
        )
    if not Ks.is_floating_point():
        Ks = Ks.to(dtype=torch.float32)
    hw = torch.as_tensor(image_size_hw, device=Ks.device, dtype=Ks.dtype)
    if hw.ndim < 1 or int(hw.shape[-1]) != 2:
        raise ValueError(
            f"image_size_hw must end in [height,width], got {tuple(hw.shape)}"
        )
    if bool((hw <= 0).any()):
        raise ValueError("image_size_hw entries must be positive")
    target_width = int(target_width)
    patch_size = int(patch_size)
    if target_width <= 0 or patch_size <= 0:
        raise ValueError("target_width and patch_size must be positive")

    # A batch-level [B,2] image size naturally broadcasts across per-frame
    # intrinsics [B,S,3,3].
    while hw.ndim - 1 < Ks.ndim - 2:
        hw = hw.unsqueeze(-2)
    leading_shape = torch.broadcast_shapes(Ks.shape[:-2], hw.shape[:-1])
    Ks = Ks.expand(leading_shape + (3, 3))
    hw = hw.expand(leading_shape + (2,))

    raw_h, raw_w = hw.unbind(-1)
    model_w = torch.full_like(raw_w, float(target_width))
    model_h_before_crop = (
        torch.round(raw_h * (model_w / raw_w) / float(patch_size))
        * float(patch_size)
    )
    crop_top = torch.where(
        model_h_before_crop > model_w,
        torch.floor((model_h_before_crop - model_w) / 2.0),
        torch.zeros_like(model_h_before_crop),
    )
    model_h = torch.minimum(model_h_before_crop, model_w)

    raw_to_model = torch.zeros(
        leading_shape + (3, 3),
        device=Ks.device,
        dtype=Ks.dtype,
    )
    raw_to_model[..., 0, 0] = model_w / raw_w
    raw_to_model[..., 1, 1] = model_h_before_crop / raw_h
    raw_to_model[..., 1, 2] = -crop_top
    raw_to_model[..., 2, 2] = 1.0
    model_intrinsics = torch.matmul(raw_to_model, Ks)
    model_hw = torch.stack((model_h, model_w), dim=-1).to(dtype=torch.long)
    return model_intrinsics, model_hw


@dataclass(frozen=True)
class FactorizedAssetCondition:
    """The only asset-condition object accepted by SceneFlow pretraining."""

    appearance_tokens: torch.Tensor
    appearance_mask: torch.Tensor
    canonical_uv: torch.Tensor
    placement_state: torch.Tensor
    target_bbox_patch: torch.Tensor
    track_valid: torch.Tensor
    reference_frame_id: torch.Tensor | None = None

    def validate(self) -> "FactorizedAssetCondition":
        if self.appearance_tokens.ndim != 4:
            raise ValueError(
                "appearance_tokens must be [B,K,Q,Ca], got "
                f"{tuple(self.appearance_tokens.shape)}"
            )
        b, k, q, _ = self.appearance_tokens.shape
        expected = (b, k, q)
        if tuple(self.appearance_mask.shape) != expected:
            raise ValueError(
                f"appearance_mask shape {tuple(self.appearance_mask.shape)} != {expected}"
            )
        if self.appearance_mask.dtype != torch.bool:
            raise TypeError("appearance_mask must have bool dtype")
        if tuple(self.canonical_uv.shape) != expected + (2,):
            raise ValueError(
                f"canonical_uv shape {tuple(self.canonical_uv.shape)} != {expected + (2,)}"
            )
        if self.placement_state.ndim != 4 or tuple(self.placement_state.shape[:2]) != (b, k):
            raise ValueError(
                "placement_state must be [B,K,S,12], got "
                f"{tuple(self.placement_state.shape)}"
            )
        if int(self.placement_state.shape[-1]) != PLACEMENT_STATE_DIM:
            raise ValueError(
                f"placement_state last dim must be {PLACEMENT_STATE_DIM}, "
                f"got {self.placement_state.shape[-1]}"
            )
        s = int(self.placement_state.shape[2])
        if tuple(self.target_bbox_patch.shape) != (b, k, s, 4):
            raise ValueError(
                "target_bbox_patch shape "
                f"{tuple(self.target_bbox_patch.shape)} != {(b, k, s, 4)}"
            )
        if tuple(self.track_valid.shape) != (b, k, s):
            raise ValueError(
                f"track_valid shape {tuple(self.track_valid.shape)} != {(b, k, s)}"
            )
        if self.track_valid.dtype != torch.bool:
            raise TypeError("track_valid must have bool dtype")
        if self.reference_frame_id is not None and tuple(self.reference_frame_id.shape) != (b, k):
            raise ValueError(
                "reference_frame_id shape "
                f"{tuple(self.reference_frame_id.shape)} != {(b, k)}"
            )
        if self.reference_frame_id is not None and self.reference_frame_id.dtype != torch.long:
            raise TypeError("reference_frame_id must have int64 dtype")
        for name, value in (
            ("appearance_tokens", self.appearance_tokens),
            ("canonical_uv", self.canonical_uv),
            ("placement_state", self.placement_state),
            ("target_bbox_patch", self.target_bbox_patch),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if bool(((self.canonical_uv < 0.0) | (self.canonical_uv > 1.0)).any()):
            raise ValueError("canonical_uv must lie in [0,1]")
        return self

    @property
    def batch_size(self) -> int:
        return int(self.appearance_tokens.shape[0])

    @property
    def num_assets(self) -> int:
        return int(self.appearance_tokens.shape[1])

    @property
    def seq_len(self) -> int:
        return int(self.placement_state.shape[2])

    def to(self, *args: Any, **kwargs: Any) -> "FactorizedAssetCondition":
        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(*args, **kwargs)

        moved_reference = move(self.reference_frame_id)
        return FactorizedAssetCondition(
            appearance_tokens=move(self.appearance_tokens),
            appearance_mask=move(self.appearance_mask).to(dtype=torch.bool),
            canonical_uv=move(self.canonical_uv),
            placement_state=move(self.placement_state),
            target_bbox_patch=move(self.target_bbox_patch),
            track_valid=move(self.track_valid).to(dtype=torch.bool),
            reference_frame_id=(
                None if moved_reference is None else moved_reference.to(dtype=torch.long)
            ),
        ).validate()

    def slice_time(self, start: int, end: int) -> "FactorizedAssetCondition":
        start, end = int(start), int(end)
        if start < 0 or end < start or end > self.seq_len:
            raise IndexError(f"invalid factorized-condition slice [{start}:{end}] for S={self.seq_len}")
        return replace(
            self,
            placement_state=self.placement_state[:, :, start:end],
            target_bbox_patch=self.target_bbox_patch[:, :, start:end],
            track_valid=self.track_valid[:, :, start:end],
        ).validate()

    def drop_rows(self, rows: torch.Tensor) -> "FactorizedAssetCondition":
        rows = torch.as_tensor(rows, device=self.appearance_mask.device, dtype=torch.bool).reshape(-1)
        if tuple(rows.shape) != (self.batch_size,):
            raise ValueError(f"drop rows must be [B]={self.batch_size}, got {tuple(rows.shape)}")
        mask = self.appearance_mask.clone()
        track = self.track_valid.clone()
        mask[rows] = False
        track[rows] = False
        return replace(self, appearance_mask=mask, track_valid=track).validate()


def canonicalize_asset_reference(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    canvas_hw: tuple[int, int] | Sequence[int],
    *,
    crop_margin: float = CANONICAL_CROP_MARGIN,
    long_side_fraction: float = CANONICAL_LONG_SIDE_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isolate, tightly crop, resize and center one asset on a black canvas."""
    if rgb.ndim != 3 or int(rgb.shape[0]) != 3:
        raise ValueError(f"rgb must be [3,H,W], got {tuple(rgb.shape)}")
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    if alpha.ndim != 3 or int(alpha.shape[0]) != 1 or alpha.shape[-2:] != rgb.shape[-2:]:
        raise ValueError(
            f"alpha must be [1,H,W] matching rgb, got {tuple(alpha.shape)} and {tuple(rgb.shape)}"
        )
    canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
    if canvas_h <= 0 or canvas_w <= 0:
        raise ValueError(f"canvas_hw must be positive, got {canvas_hw}")
    if not 0.0 <= float(crop_margin):
        raise ValueError(f"crop_margin must be non-negative, got {crop_margin}")
    if not 0.0 < float(long_side_fraction) <= 1.0:
        raise ValueError(f"long_side_fraction must be in (0,1], got {long_side_fraction}")

    rgb = rgb.float()
    alpha = alpha.float().clamp(0.0, 1.0)
    foreground = alpha[0] > 0.0
    out_rgb = rgb.new_zeros((3, canvas_h, canvas_w))
    out_alpha = alpha.new_zeros((1, canvas_h, canvas_w))
    if not bool(foreground.any()):
        return out_rgb, out_alpha

    ys, xs = torch.where(foreground)
    y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
    x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
    box_h, box_w = y1 - y0, x1 - x0
    pad_y = int(math.ceil(float(box_h) * float(crop_margin)))
    pad_x = int(math.ceil(float(box_w) * float(crop_margin)))
    y0, y1 = max(0, y0 - pad_y), min(int(rgb.shape[-2]), y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(int(rgb.shape[-1]), x1 + pad_x)

    crop_alpha = alpha[:, y0:y1, x0:x1]
    # Premultiplication makes reference-mask-outside background provably
    # irrelevant to the appearance encoder.
    crop_rgb = rgb[:, y0:y1, x0:x1] * crop_alpha
    crop_h, crop_w = int(crop_rgb.shape[-2]), int(crop_rgb.shape[-1])
    available_h = max(1, int(round(canvas_h * float(long_side_fraction))))
    available_w = max(1, int(round(canvas_w * float(long_side_fraction))))
    scale = min(available_h / float(crop_h), available_w / float(crop_w))
    new_h = max(1, min(canvas_h, int(round(crop_h * scale))))
    new_w = max(1, min(canvas_w, int(round(crop_w * scale))))
    resized_rgb = F.interpolate(
        crop_rgb.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    resized_alpha = F.interpolate(
        crop_alpha.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )[0].clamp(0.0, 1.0)
    top = (canvas_h - new_h) // 2
    left = (canvas_w - new_w) // 2
    out_alpha[:, top : top + new_h, left : left + new_w] = resized_alpha
    # `resized_rgb` is already premultiplied before interpolation. Multiplying
    # alpha a second time would darken anti-aliased boundary pixels.
    out_rgb[:, top : top + new_h, left : left + new_w] = resized_rgb
    return out_rgb, out_alpha


def alpha_to_patch_mask(
    alpha: torch.Tensor,
    patch_grid: tuple[int, int] | Sequence[int],
    *,
    coverage_threshold: float = CANONICAL_ALPHA_PATCH_THRESHOLD,
) -> torch.Tensor:
    """Return patches whose alpha coverage (not max alpha) reaches 5%."""
    squeeze = alpha.ndim == 3
    if squeeze:
        alpha = alpha.unsqueeze(0)
    if alpha.ndim != 4 or int(alpha.shape[1]) != 1:
        raise ValueError(f"alpha must be [N,1,H,W] or [1,H,W], got {tuple(alpha.shape)}")
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    pooled = F.adaptive_avg_pool2d(alpha.float().clamp(0.0, 1.0), (gh, gw))
    result = pooled[:, 0].reshape(int(alpha.shape[0]), gh * gw).ge(float(coverage_threshold))
    return result[0] if squeeze else result


def canonical_patch_uv(patch_grid: tuple[int, int] | Sequence[int], *, device=None) -> torch.Tensor:
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    y, x = torch.meshgrid(
        (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) / float(gh),
        (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) / float(gw),
        indexing="ij",
    )
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def sample_canonical_tokens(
    tokens: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_grid: tuple[int, int] | Sequence[int],
    *,
    max_tokens: int = MAX_CANONICAL_APPEARANCE_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniformly sample canonical patches once, independent of target frames."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [N,P,C], got {tuple(tokens.shape)}")
    n, p, c = tokens.shape
    if tuple(patch_mask.shape) != (n, p):
        raise ValueError(f"patch_mask shape {tuple(patch_mask.shape)} != {(n, p)}")
    q = int(max_tokens)
    uv_all = canonical_patch_uv(patch_grid, device=tokens.device)
    if int(uv_all.shape[0]) != p:
        raise ValueError(f"patch_grid={patch_grid} has {uv_all.shape[0]} patches, tokens have {p}")
    out = tokens.new_zeros((n, q, c))
    out_mask = torch.zeros((n, q), device=tokens.device, dtype=torch.bool)
    out_uv = tokens.new_zeros((n, q, 2), dtype=torch.float32)
    for row in range(n):
        valid = torch.nonzero(patch_mask[row], as_tuple=False).flatten()
        count = min(int(valid.numel()), q)
        if count <= 0:
            continue
        valid_uv = uv_all.index_select(0, valid)
        uv_min = valid_uv.amin(dim=0)
        uv_max = valid_uv.amax(dim=0)
        uv_span = uv_max - uv_min
        if count == 1:
            selected = valid[valid.numel() // 2 : valid.numel() // 2 + 1]
        else:
            ranks = torch.linspace(0, int(valid.numel()) - 1, count, device=tokens.device).round().long()
            selected = valid.index_select(0, ranks)
        out[row, :count] = tokens[row].index_select(0, selected)
        selected_uv = uv_all.index_select(0, selected)
        # ``canonicalize_asset_reference`` centers a variable-aspect-ratio crop
        # on a fixed canvas. Destination bboxes describe the object, not that
        # canvas, so positions must be relative to the occupied alpha extent.
        # A one-patch-wide extent has no resolvable coordinate on that axis and
        # is placed at its destination midpoint.
        normalized_uv = torch.where(
            uv_span > 1.0e-8,
            (selected_uv - uv_min) / uv_span.clamp_min(1.0e-8),
            torch.full_like(selected_uv, 0.5),
        )
        out_uv[row, :count] = normalized_uv.clamp(0.0, 1.0)
        out_mask[row, :count] = True
    return out, out_mask, out_uv


def finite_difference_velocity(
    centers: torch.Tensor,
    timestamps: torch.Tensor,
    track_valid: torch.Tensor,
) -> torch.Tensor:
    """Differentiate centers using real timestamps and valid neighboring track samples."""
    if centers.ndim < 3 or int(centers.shape[-1]) != 3:
        raise ValueError(f"centers must be [...,S,3], got {tuple(centers.shape)}")
    if tuple(track_valid.shape) != tuple(centers.shape[:-1]):
        raise ValueError(
            f"track_valid shape {tuple(track_valid.shape)} != {tuple(centers.shape[:-1])}"
        )
    s = int(centers.shape[-2])
    times = torch.as_tensor(timestamps, device=centers.device, dtype=centers.dtype)
    if times.ndim == 1:
        times = times.view((1,) * (centers.ndim - 2) + (s,)).expand(centers.shape[:-1])
    if tuple(times.shape) != tuple(centers.shape[:-1]):
        raise ValueError(f"timestamps shape {tuple(times.shape)} cannot map to {tuple(centers.shape[:-1])}")
    velocity = torch.zeros_like(centers)
    flat_center = centers.reshape(-1, s, 3)
    flat_times = times.reshape(-1, s)
    flat_valid = track_valid.reshape(-1, s)
    flat_velocity = velocity.reshape(-1, s, 3)
    for row in range(int(flat_center.shape[0])):
        valid_indices = torch.nonzero(flat_valid[row], as_tuple=False).flatten().tolist()
        if len(valid_indices) < 2:
            continue
        for rank, frame in enumerate(valid_indices):
            left = valid_indices[max(0, rank - 1)]
            right = valid_indices[min(len(valid_indices) - 1, rank + 1)]
            if left == right:
                continue
            dt = (flat_times[row, right] - flat_times[row, left]).clamp_min(1.0e-6)
            flat_velocity[row, frame] = (
                flat_center[row, right] - flat_center[row, left]
            ) / dt
    return velocity


def interpolate_box_keyframes(
    keyframes: Sequence[Mapping[str, Any]],
    num_frames: int,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate center/size/yaw, then derive velocity from timestamps."""
    if int(num_frames) <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    if not keyframes:
        raise ValueError("each object requires at least one box keyframe")
    ordered = sorted(keyframes, key=lambda item: int(item["frame_id"]))
    ids = torch.tensor([int(item["frame_id"]) for item in ordered], dtype=torch.long)
    if bool((ids < 0).any()) or bool((ids >= int(num_frames)).any()):
        raise ValueError(f"keyframe ids must lie in [0,{int(num_frames) - 1}]")
    if int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError("object keyframe ids must be unique")
    center_k = torch.tensor([item["center"] for item in ordered], dtype=torch.float32)
    size_k = torch.tensor([item["size"] for item in ordered], dtype=torch.float32)
    yaw_k = torch.tensor([float(item["yaw"]) for item in ordered], dtype=torch.float32)
    if tuple(center_k.shape[1:]) != (3,) or tuple(size_k.shape[1:]) != (3,):
        raise ValueError("keyframe center and size must each have three values")
    if bool((size_k <= 0.0).any()):
        raise ValueError("box sizes must be positive")

    centers = torch.zeros((num_frames, 3), dtype=torch.float32)
    sizes = torch.zeros_like(centers)
    yaws = torch.zeros((num_frames,), dtype=torch.float32)
    for frame in range(int(num_frames)):
        if len(ordered) == 1:
            left = right = 0
            weight = 0.0
        elif frame <= int(ids[0]):
            left = right = 0
            weight = 0.0
        elif frame >= int(ids[-1]):
            left = right = len(ordered) - 1
            weight = 0.0
        else:
            right = int(torch.searchsorted(ids, torch.tensor(frame), right=False).item())
            left = right - 1
            weight = (frame - int(ids[left])) / float(int(ids[right]) - int(ids[left]))
        centers[frame] = torch.lerp(center_k[left], center_k[right], float(weight))
        sizes[frame] = torch.lerp(size_k[left], size_k[right], float(weight))
        delta = torch.atan2(torch.sin(yaw_k[right] - yaw_k[left]), torch.cos(yaw_k[right] - yaw_k[left]))
        yaws[frame] = yaw_k[left] + float(weight) * delta
    track_valid = torch.ones((num_frames,), dtype=torch.bool)
    timestamps = torch.arange(num_frames, dtype=torch.float32) / float(fps)
    velocity = finite_difference_velocity(
        centers.unsqueeze(0), timestamps, track_valid.unsqueeze(0)
    )[0]
    return centers, sizes, yaws, velocity, track_valid


def _box_corners_object(size_lwh: torch.Tensor) -> torch.Tensor:
    signs = size_lwh.new_tensor(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )
    return 0.5 * signs * size_lwh.unsqueeze(-2)


def object_to_anchor_from_center_yaw(
    centers: torch.Tensor,
    yaws: torch.Tensor,
) -> torch.Tensor:
    """Build Waymo-object-to-camera-anchor transforms from center and yaw."""
    if centers.ndim < 2 or centers.shape[-1] != 3:
        raise ValueError("centers must end in [...,S,3]")
    if tuple(yaws.shape) != tuple(centers.shape[:-1]):
        raise ValueError("yaws shape must match centers without the xyz dimension")
    result = torch.eye(
        4,
        dtype=centers.dtype,
        device=centers.device,
    ).expand(*centers.shape[:-1], 4, 4).clone()
    cosine, sine = torch.cos(yaws), torch.sin(yaws)
    # Waymo object axes are x=length/forward, y=width/left, z=height/up.
    # Camera-anchor axes are x=right, y=down, z=forward.
    result[..., 0, 0] = sine
    result[..., 2, 0] = cosine
    result[..., 0, 1] = -cosine
    result[..., 1, 1] = 0.0
    result[..., 2, 1] = sine
    result[..., 0, 2] = 0.0
    result[..., 1, 2] = -1.0
    result[..., 2, 2] = 0.0
    result[..., :3, 3] = centers
    return result


def project_anchor_boxes_to_patch_bboxes(
    object_to_anchor: torch.Tensor,
    box_size_lwh: torch.Tensor,
    track_valid: torch.Tensor,
    camera_to_anchor: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: torch.Tensor | Sequence[int],
    patch_grid: tuple[int, int] | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project boxes with geometry only; no silhouette or GT visibility input."""
    if object_to_anchor.ndim != 5 or object_to_anchor.shape[-2:] != (4, 4):
        raise ValueError(f"object_to_anchor must be [B,K,S,4,4], got {tuple(object_to_anchor.shape)}")
    b, k, s = (int(v) for v in object_to_anchor.shape[:3])
    if tuple(box_size_lwh.shape) != (b, k, s, 3):
        raise ValueError(f"box_size_lwh shape {tuple(box_size_lwh.shape)} != {(b, k, s, 3)}")
    if tuple(track_valid.shape) != (b, k, s):
        raise ValueError(f"track_valid shape {tuple(track_valid.shape)} != {(b, k, s)}")
    c2a = torch.as_tensor(camera_to_anchor, device=object_to_anchor.device, dtype=object_to_anchor.dtype)
    if c2a.ndim == 3:
        c2a = c2a.unsqueeze(0)
    if tuple(c2a.shape) != (b, s, 4, 4):
        raise ValueError(f"camera_to_anchor shape {tuple(c2a.shape)} != {(b, s, 4, 4)}")
    Ks = torch.as_tensor(intrinsics, device=object_to_anchor.device, dtype=object_to_anchor.dtype)
    if Ks.ndim == 3:
        Ks = Ks.unsqueeze(0)
    if int(Ks.shape[1]) == 1 and s > 1:
        Ks = Ks.expand(b, s, -1, -1)
    if tuple(Ks.shape) != (b, s, 3, 3):
        raise ValueError(f"intrinsics shape {tuple(Ks.shape)} != {(b, s, 3, 3)}")
    hw = torch.as_tensor(image_size_hw, device=object_to_anchor.device, dtype=object_to_anchor.dtype)
    if tuple(hw.shape) == (2,):
        hw = hw.view(1, 1, 2).expand(b, s, 2)
    elif tuple(hw.shape) == (b, 2):
        hw = hw[:, None].expand(b, s, 2)
    elif tuple(hw.shape) == (s, 2):
        hw = hw[None].expand(b, s, 2)
    elif tuple(hw.shape) == (b, 1, 2):
        hw = hw.expand(b, s, 2)
    if tuple(hw.shape) != (b, s, 2):
        raise ValueError(f"image_size_hw shape {tuple(hw.shape)} cannot map to {(b, s, 2)}")

    corners_obj = _box_corners_object(box_size_lwh)
    ones = torch.ones(corners_obj.shape[:-1] + (1,), device=corners_obj.device, dtype=corners_obj.dtype)
    corners_anchor = torch.matmul(
        object_to_anchor.unsqueeze(-3),
        torch.cat((corners_obj, ones), dim=-1).unsqueeze(-1),
    ).squeeze(-1)[..., :3]
    anchor_to_camera = torch.linalg.inv(c2a)
    corners_h = torch.cat((corners_anchor, ones), dim=-1)
    corners_cam = torch.matmul(
        anchor_to_camera[:, None, :, None],
        corners_h.unsqueeze(-1),
    ).squeeze(-1)[..., :3]
    depth = corners_cam[..., 2]
    near = float(BOX_PROJECTION_NEAR_PLANE)
    positive = depth >= near

    # A cuboid can cross the camera near plane even when none of its visible
    # corner projections describe the visible silhouette. Clip all 12 edges
    # against that plane and include the intersections in the 2D bounds.
    edges = torch.tensor(
        (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ),
        device=corners_cam.device,
        dtype=torch.long,
    )
    edge_start = corners_cam.index_select(-2, edges[:, 0])
    edge_end = corners_cam.index_select(-2, edges[:, 1])
    start_depth = edge_start[..., 2]
    end_depth = edge_end[..., 2]
    edge_crosses = (start_depth >= near) != (end_depth >= near)
    depth_delta = end_depth - start_depth
    safe_depth_delta = torch.where(
        depth_delta.abs() > torch.finfo(corners_cam.dtype).eps,
        depth_delta,
        torch.ones_like(depth_delta),
    )
    interpolation = (near - start_depth) / safe_depth_delta
    intersections = edge_start + interpolation.unsqueeze(-1) * (edge_end - edge_start)
    candidates_cam = torch.cat((corners_cam, intersections), dim=-2)
    candidates_valid = torch.cat((positive, edge_crosses), dim=-1)
    candidate_depth = candidates_cam[..., 2]
    proj_h = torch.matmul(
        Ks[:, None, :, None],
        candidates_cam.unsqueeze(-1),
    ).squeeze(-1)
    xy = proj_h[..., :2] / candidate_depth.clamp_min(near).unsqueeze(-1)
    inf = torch.full_like(xy[..., 0], float("inf"))
    ninf = torch.full_like(xy[..., 0], float("-inf"))
    x0 = torch.where(candidates_valid, xy[..., 0], inf).amin(dim=-1)
    y0 = torch.where(candidates_valid, xy[..., 1], inf).amin(dim=-1)
    x1 = torch.where(candidates_valid, xy[..., 0], ninf).amax(dim=-1)
    y1 = torch.where(candidates_valid, xy[..., 1], ninf).amax(dim=-1)
    has_depth = candidates_valid.any(dim=-1)
    height = hw[:, None, :, 0]
    width = hw[:, None, :, 1]
    crosses_near = positive.any(dim=-1) & (~positive).any(dim=-1)
    # Near-plane intersections can project arbitrarily far away. For boxes
    # crossing the plane, the useful 2D support is their visible image-space
    # intersection; keeping unbounded coordinates would collapse many RoPE
    # positions onto the canvas edges downstream.
    clipped_x0 = torch.minimum(x0.clamp_min(0.0), width)
    clipped_y0 = torch.minimum(y0.clamp_min(0.0), height)
    clipped_x1 = torch.minimum(x1.clamp_min(0.0), width)
    clipped_y1 = torch.minimum(y1.clamp_min(0.0), height)
    x0 = torch.where(crosses_near, clipped_x0, x0)
    y0 = torch.where(crosses_near, clipped_y0, y0)
    x1 = torch.where(crosses_near, clipped_x1, x1)
    y1 = torch.where(crosses_near, clipped_y1, y1)
    in_frustum = (
        track_valid
        & has_depth
        & (x1 > 0.0)
        & (y1 > 0.0)
        & (x0 < width)
        & (y0 < height)
        & (x1 > x0)
        & (y1 > y0)
    )
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    bbox = torch.stack(
        (
            x0 / width * float(gw),
            y0 / height * float(gh),
            x1 / width * float(gw),
            y1 / height * float(gh),
        ),
        dim=-1,
    )
    fallback = bbox.new_tensor([-1.0, -1.0, -1.0, -1.0])
    bbox = torch.where((track_valid & has_depth).unsqueeze(-1), bbox, fallback)
    return torch.nan_to_num(bbox, nan=-1.0, posinf=-1.0, neginf=-1.0), in_frustum


def bbox_patch_mask(
    target_bbox_patch: torch.Tensor,
    in_frustum: torch.Tensor,
    patch_grid: tuple[int, int] | Sequence[int],
) -> torch.Tensor:
    """Rasterize geometry-projected boxes into destination patch cells."""
    if target_bbox_patch.ndim != 4 or int(target_bbox_patch.shape[-1]) != 4:
        raise ValueError(
            f"target_bbox_patch must be [B,K,S,4], got {tuple(target_bbox_patch.shape)}"
        )
    if tuple(in_frustum.shape) != tuple(target_bbox_patch.shape[:-1]):
        raise ValueError("in_frustum must match target_bbox_patch leading dimensions")
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    top, left = torch.meshgrid(
        torch.arange(gh, device=target_bbox_patch.device, dtype=torch.float32),
        torch.arange(gw, device=target_bbox_patch.device, dtype=torch.float32),
        indexing="ij",
    )
    x0, y0, x1, y1 = target_bbox_patch.unbind(-1)
    mask = (
        ((left + 1.0) > x0.unsqueeze(-1).unsqueeze(-1))
        & (left < x1.unsqueeze(-1).unsqueeze(-1))
        & ((top + 1.0) > y0.unsqueeze(-1).unsqueeze(-1))
        & (top < y1.unsqueeze(-1).unsqueeze(-1))
        & in_frustum.unsqueeze(-1).unsqueeze(-1)
    )
    return mask.reshape(target_bbox_patch.shape[:-1] + (gh * gw,))


def build_placement_state(
    center_anchor: torch.Tensor,
    box_size_lwh: torch.Tensor,
    yaw: torch.Tensor,
    velocity_anchor: torch.Tensor,
    in_frustum: torch.Tensor,
) -> torch.Tensor:
    expected = center_anchor.shape[:-1]
    if int(center_anchor.shape[-1]) != 3 or tuple(box_size_lwh.shape) != expected + (3,):
        raise ValueError("center_anchor and box_size_lwh must be [...,3]")
    if tuple(yaw.shape) != expected or tuple(velocity_anchor.shape) != expected + (3,):
        raise ValueError("yaw must be [...] and velocity_anchor [...,3]")
    if tuple(in_frustum.shape) != expected:
        raise ValueError("in_frustum must match placement leading dimensions")
    return torch.cat(
        (
            center_anchor,
            box_size_lwh.clamp_min(1.0e-6).log(),
            torch.sin(yaw).unsqueeze(-1),
            torch.cos(yaw).unsqueeze(-1),
            velocity_anchor,
            in_frustum.to(dtype=center_anchor.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )


def build_factorized_asset_condition(
    *,
    appearance_tokens: torch.Tensor,
    appearance_mask: torch.Tensor,
    canonical_uv: torch.Tensor,
    object_to_anchor: torch.Tensor,
    center_anchor: torch.Tensor,
    box_size_lwh: torch.Tensor,
    yaw: torch.Tensor,
    velocity_anchor: torch.Tensor,
    track_valid: torch.Tensor,
    camera_to_anchor: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: torch.Tensor | Sequence[int],
    patch_grid: tuple[int, int] | Sequence[int],
    reference_frame_id: torch.Tensor | None = None,
) -> FactorizedAssetCondition:
    """Shared training/inference builder for destination geometry."""
    expected_center = object_to_anchor[..., :3, 3]
    if tuple(expected_center.shape) != tuple(center_anchor.shape):
        raise ValueError(
            "object_to_anchor translation and center_anchor must have matching shapes"
        )
    valid = track_valid.to(device=expected_center.device, dtype=torch.bool)
    center_error = (expected_center - center_anchor.to(expected_center)).abs().amax(
        dim=-1
    )
    if bool((center_error[valid] > 1.0e-4).any()):
        raise ValueError(
            "center_anchor disagrees with object_to_anchor translation on a valid track"
        )
    heading = object_to_anchor[..., :3, 0]
    transform_yaw = torch.atan2(heading[..., 0], heading[..., 2])
    yaw_error = torch.atan2(
        torch.sin(transform_yaw - yaw.to(transform_yaw)),
        torch.cos(transform_yaw - yaw.to(transform_yaw)),
    ).abs()
    if bool((yaw_error[valid] > 1.0e-4).any()):
        raise ValueError(
            "yaw disagrees with object_to_anchor local-x heading on a valid track; "
            "yaw=0 must point along anchor +z"
        )
    height_axis = object_to_anchor[..., :3, 2]
    height_axis_norm = torch.linalg.vector_norm(height_axis, dim=-1)
    height_alignment = -height_axis[..., 1] / height_axis_norm.clamp_min(1.0e-8)
    if bool(
        (
            (height_axis_norm - 1.0).abs().gt(1.0e-3)
            | height_alignment.lt(0.95)
        )[valid].any()
    ):
        raise ValueError(
            "object_to_anchor local-z height axis must be unit length and point "
            "approximately along anchor -y on a valid track"
        )
    bbox, in_frustum = project_anchor_boxes_to_patch_bboxes(
        object_to_anchor,
        box_size_lwh,
        track_valid,
        camera_to_anchor,
        intrinsics,
        image_size_hw,
        patch_grid,
    )
    placement = build_placement_state(
        center_anchor,
        box_size_lwh,
        yaw,
        velocity_anchor,
        in_frustum,
    )
    # A slot without a source reference is closed even when a target track is
    # present; there is intentionally no target-window fallback.
    source_valid = appearance_mask.any(dim=-1)
    closed_track = track_valid & source_valid.unsqueeze(-1)
    return FactorizedAssetCondition(
        appearance_tokens=appearance_tokens,
        appearance_mask=appearance_mask,
        canonical_uv=canonical_uv,
        placement_state=placement,
        target_bbox_patch=bbox,
        track_valid=closed_track,
        reference_frame_id=reference_frame_id,
    ).validate()
