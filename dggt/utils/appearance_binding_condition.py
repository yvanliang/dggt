"""Appearance-to-geometry binding without a second source of position.

The six fields of :class:`AppearanceBindingCondition` are the complete v2
contract.  Boxes, placement features, per-slot position embeddings, and metric
geometry are intentionally absent: every spatial address is gathered from
``ProjectedActorGeometry`` through ``geometry_idx``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from dggt.utils.actor_geometry_condition import (
    ActorGeometryCondition,
    LayoutMode,
    ProjectedActorGeometry,
    gather_actor_slots,
)


MAX_APPEARANCE_BINDINGS = 5
MAX_APPEARANCE_TOKENS = 32
APPEARANCE_TOKEN_DIM = 1024
CANONICAL_CROP_MARGIN = 0.15
CANONICAL_LONG_SIDE_FRACTION = 0.80
CANONICAL_ALPHA_PATCH_THRESHOLD = 0.05


class AppearanceMode(IntEnum):
    NULL = 0
    REAL = 1


def canonicalize_appearance_reference(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    canvas_hw: tuple[int, int] | Sequence[int],
    *,
    crop_margin: float = CANONICAL_CROP_MARGIN,
    long_side_fraction: float = CANONICAL_LONG_SIDE_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop and center one premultiplied reference without target geometry."""

    if rgb.ndim != 3 or int(rgb.shape[0]) != 3:
        raise ValueError(f"rgb must be [3,H,W], got {tuple(rgb.shape)}")
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    if (
        alpha.ndim != 3
        or int(alpha.shape[0]) != 1
        or alpha.shape[-2:] != rgb.shape[-2:]
    ):
        raise ValueError("alpha must be [1,H,W] and match rgb")
    canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
    if canvas_h <= 0 or canvas_w <= 0:
        raise ValueError("canvas dimensions must be positive")
    if float(crop_margin) < 0.0:
        raise ValueError("crop_margin must be non-negative")
    if not 0.0 < float(long_side_fraction) <= 1.0:
        raise ValueError("long_side_fraction must be in (0,1]")

    rgb = rgb.float()
    alpha = alpha.float().clamp(0.0, 1.0)
    foreground = alpha[0] > 0.0
    output_rgb = rgb.new_zeros((3, canvas_h, canvas_w))
    output_alpha = alpha.new_zeros((1, canvas_h, canvas_w))
    if not bool(foreground.any()):
        return output_rgb, output_alpha
    ys, xs = torch.where(foreground)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(math.ceil((y1 - y0) * float(crop_margin)))
    pad_x = int(math.ceil((x1 - x0) * float(crop_margin)))
    y0, y1 = max(0, y0 - pad_y), min(int(rgb.shape[-2]), y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(int(rgb.shape[-1]), x1 + pad_x)
    crop_alpha = alpha[:, y0:y1, x0:x1]
    crop_rgb = rgb[:, y0:y1, x0:x1] * crop_alpha
    crop_h, crop_w = int(crop_rgb.shape[-2]), int(crop_rgb.shape[-1])
    scale = min(
        max(1, round(canvas_h * float(long_side_fraction))) / float(crop_h),
        max(1, round(canvas_w * float(long_side_fraction))) / float(crop_w),
    )
    new_h = max(1, min(canvas_h, int(round(crop_h * scale))))
    new_w = max(1, min(canvas_w, int(round(crop_w * scale))))
    resized_rgb = F.interpolate(
        crop_rgb.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0]
    resized_alpha = F.interpolate(
        crop_alpha.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0].clamp(0.0, 1.0)
    top, left = (canvas_h - new_h) // 2, (canvas_w - new_w) // 2
    output_rgb[:, top : top + new_h, left : left + new_w] = resized_rgb
    output_alpha[:, top : top + new_h, left : left + new_w] = resized_alpha
    return output_rgb, output_alpha


def appearance_alpha_to_patch_mask(
    alpha: torch.Tensor,
    patch_grid: tuple[int, int] | Sequence[int],
    *,
    coverage_threshold: float = CANONICAL_ALPHA_PATCH_THRESHOLD,
) -> torch.Tensor:
    """Select canonical patches by alpha coverage, not maximum alpha."""

    squeeze = alpha.ndim == 3
    if squeeze:
        alpha = alpha.unsqueeze(0)
    if alpha.ndim != 4 or int(alpha.shape[1]) != 1:
        raise ValueError("alpha must be [N,1,H,W] or [1,H,W]")
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    pooled = F.adaptive_avg_pool2d(alpha.float().clamp(0.0, 1.0), (gh, gw))
    mask = pooled[:, 0].reshape(int(alpha.shape[0]), gh * gw).ge(
        float(coverage_threshold)
    )
    return mask[0] if squeeze else mask


def canonical_patch_uv(
    patch_grid: tuple[int, int] | Sequence[int], *, device=None
) -> torch.Tensor:
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    y, x = torch.meshgrid(
        (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) / gh,
        (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) / gw,
        indexing="ij",
    )
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def sample_appearance_tokens(
    tokens: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_grid: tuple[int, int] | Sequence[int],
    *,
    max_tokens: int = MAX_APPEARANCE_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniformly sample Q appearance tokens and canonical, non-target UVs."""

    if tokens.ndim != 3:
        raise ValueError("tokens must be [N,P,C]")
    rows, patches, channels = map(int, tokens.shape)
    if tuple(patch_mask.shape) != (rows, patches):
        raise ValueError("patch_mask must match the first two token axes")
    count_max = int(max_tokens)
    if not 0 < count_max <= MAX_APPEARANCE_TOKENS:
        raise ValueError(f"max_tokens must be in [1,{MAX_APPEARANCE_TOKENS}]")
    all_uv = canonical_patch_uv(patch_grid, device=tokens.device)
    if int(all_uv.shape[0]) != patches:
        raise ValueError("patch_grid does not match appearance token count")
    output = tokens.new_zeros((rows, count_max, channels))
    output_mask = torch.zeros((rows, count_max), device=tokens.device, dtype=torch.bool)
    output_uv = torch.zeros(
        (rows, count_max, 2), device=tokens.device, dtype=torch.float32
    )
    for row in range(rows):
        valid = torch.nonzero(patch_mask[row], as_tuple=False).flatten()
        selected_count = min(int(valid.numel()), count_max)
        if selected_count == 0:
            continue
        ranks = torch.linspace(
            0, int(valid.numel()) - 1, selected_count, device=tokens.device
        ).round().long()
        selected = valid.index_select(0, ranks)
        selected_uv = all_uv.index_select(0, selected)
        uv_min, uv_max = selected_uv.amin(dim=0), selected_uv.amax(dim=0)
        uv_span = uv_max - uv_min
        normalized_uv = torch.where(
            uv_span > 1.0e-8,
            (selected_uv - uv_min) / uv_span.clamp_min(1.0e-8),
            torch.full_like(selected_uv, 0.5),
        )
        output[row, :selected_count] = tokens[row].index_select(0, selected)
        output_mask[row, :selected_count] = True
        output_uv[row, :selected_count] = normalized_uv.clamp(0.0, 1.0)
    return output, output_mask, output_uv


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _require_shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {expected}")


@dataclass(frozen=True)
class AppearanceBindingCondition:
    """An unordered set of appearance payloads bound to slots in ``G``.

    Do not add fields here without changing the layout-v2 protocol.  In
    particular, this dataclass must never own a target bbox, placement state,
    target-frame track validity, metric corners/velocity, or a slot embedding.
    """

    appearance_tokens: torch.Tensor
    appearance_mask: torch.Tensor
    canonical_uv: torch.Tensor
    geometry_idx: torch.Tensor
    binding_valid: torch.Tensor
    appearance_mode: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.geometry_idx.shape[0])

    @property
    def num_bindings(self) -> int:
        return int(self.geometry_idx.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.appearance_tokens.shape[2])

    def validate(self) -> "AppearanceBindingCondition":
        if self.appearance_tokens.ndim != 4:
            raise ValueError(
                "appearance_tokens must be [B,Ka,Q,1024], got "
                f"{tuple(self.appearance_tokens.shape)}"
            )
        b, ka, q, channels = (int(v) for v in self.appearance_tokens.shape)
        if not 0 < ka <= MAX_APPEARANCE_BINDINGS:
            raise ValueError(f"Ka must be in [1,{MAX_APPEARANCE_BINDINGS}], got {ka}")
        if not 0 < q <= MAX_APPEARANCE_TOKENS:
            raise ValueError(f"Q must be in [1,{MAX_APPEARANCE_TOKENS}], got {q}")
        if channels != APPEARANCE_TOKEN_DIM:
            raise ValueError(
                f"appearance token dim must be {APPEARANCE_TOKEN_DIM}, got {channels}"
            )
        _require_shape("appearance_mask", self.appearance_mask, (b, ka, q))
        _require_shape("canonical_uv", self.canonical_uv, (b, ka, q, 2))
        _require_shape("geometry_idx", self.geometry_idx, (b, ka))
        _require_shape("binding_valid", self.binding_valid, (b, ka))
        _require_shape("appearance_mode", self.appearance_mode, (b,))

        if not self.appearance_tokens.is_floating_point():
            raise TypeError("appearance_tokens must have a floating dtype")
        if self.canonical_uv.dtype != torch.float32:
            raise TypeError("canonical_uv must have float32 dtype")
        if self.appearance_mask.dtype != torch.bool:
            raise TypeError("appearance_mask must have bool dtype")
        if self.geometry_idx.dtype != torch.int64:
            raise TypeError("geometry_idx must have int64 dtype")
        if self.binding_valid.dtype != torch.bool:
            raise TypeError("binding_valid must have bool dtype")
        if self.appearance_mode.dtype != torch.int8:
            raise TypeError("appearance_mode must have int8 dtype")
        tensors = (
            self.appearance_tokens,
            self.appearance_mask,
            self.canonical_uv,
            self.geometry_idx,
            self.binding_valid,
            self.appearance_mode,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all AppearanceBindingCondition tensors must share one device")
        if not bool(torch.isfinite(self.appearance_tokens).all()):
            raise ValueError("appearance_tokens contains NaN or Inf")
        if not bool(torch.isfinite(self.canonical_uv).all()):
            raise ValueError("canonical_uv contains NaN or Inf")
        if bool(((self.appearance_mode < int(AppearanceMode.NULL)) | (
            self.appearance_mode > int(AppearanceMode.REAL)
        )).any()):
            raise ValueError("appearance_mode contains an unknown value")

        null_rows = self.appearance_mode == int(AppearanceMode.NULL)
        if bool((self.binding_valid & null_rows[:, None]).any()):
            raise ValueError("A NULL rows cannot contain a valid binding")
        if bool((self.appearance_mask & null_rows[:, None, None]).any()):
            raise ValueError("A NULL rows must remove every token from the attention mask")
        if bool((self.geometry_idx[~self.binding_valid] != -1).any()):
            raise ValueError("invalid/padded A slots must use geometry_idx=-1")
        if bool((self.geometry_idx[self.binding_valid] < 0).any()):
            raise ValueError("binding-valid A slots require a non-negative geometry_idx")

        token_present = self.appearance_mask.any(dim=-1)
        if bool((token_present != self.binding_valid).any()):
            raise ValueError(
                "each binding-valid A slot needs at least one appearance token, and "
                "padded slots must have none"
            )
        valid_uv = self.canonical_uv[self.appearance_mask]
        if valid_uv.numel() and bool(((valid_uv < 0.0) | (valid_uv > 1.0)).any()):
            raise ValueError("canonical_uv must lie in [0,1] at valid tokens")
        return self

    def validate_against_geometry(
        self,
        geometry: ActorGeometryCondition,
        *,
        appearance_class_id: torch.Tensor | None = None,
        retained_slot_mask: torch.Tensor | None = None,
    ) -> "AppearanceBindingCondition":
        """Apply all five model-entry fail-fast checks from the v2 contract.

        ``appearance_class_id`` is supplied alongside the reference-image
        metadata rather than stored in ``A``; keeping it out of the dataclass is
        necessary for the frozen six-field payload.  ``retained_slot_mask`` is
        the proposed post-dropout G mask and lets callers reject a partial
        dropout before deleting an actor bound by A.
        """

        self.validate()
        geometry.validate()
        if self.batch_size != geometry.batch_size:
            raise ValueError("A and G batch sizes differ")
        if self.geometry_idx.device != geometry.slot_valid.device:
            raise ValueError("A and G tensors must share one device")

        geometry_absent = (
            (geometry.layout_mode == int(LayoutMode.NULL))
            | (geometry.layout_mode == int(LayoutMode.EMPTY))
        )
        appearance_real = self.appearance_mode == int(AppearanceMode.REAL)
        if bool((geometry_absent & appearance_real).any()):
            raise ValueError("G EMPTY/NULL requires A NULL")

        bound = self.binding_valid & appearance_real[:, None]
        gathered_slot_valid, in_range = gather_actor_slots(
            geometry.slot_valid,
            self.geometry_idx,
            selection_mask=bound,
            fill_value=False,
        )
        missing_geometry = bound & (~in_range | ~gathered_slot_valid)
        if bool(missing_geometry.any()):
            raise ValueError(
                "every binding-valid A slot in REAL mode must reference a slot-valid G actor"
            )

        # A is an unordered set, but it may contain each G identity at most once.
        pair_bound = bound[:, :, None] & bound[:, None, :]
        same_geometry = self.geometry_idx[:, :, None] == self.geometry_idx[:, None, :]
        upper = torch.triu(
            torch.ones(
                self.num_bindings,
                self.num_bindings,
                dtype=torch.bool,
                device=self.geometry_idx.device,
            ),
            diagonal=1,
        )
        if bool((pair_bound & same_geometry & upper[None]).any()):
            raise ValueError("two A slots cannot bind the same G slot")

        if bool(bound.any()):
            if appearance_class_id is None:
                raise ValueError(
                    "appearance_class_id is required to verify A/G class agreement"
                )
            _require_shape(
                "appearance_class_id",
                appearance_class_id,
                tuple(self.geometry_idx.shape),
            )
            if appearance_class_id.dtype not in _INTEGER_DTYPES:
                raise TypeError("appearance_class_id must have an integer dtype")
            if appearance_class_id.device != self.geometry_idx.device:
                raise ValueError("appearance_class_id and A must share one device")
            gathered_class, _ = gather_actor_slots(
                geometry.class_id,
                self.geometry_idx,
                selection_mask=bound,
                fill_value=-1,
            )
            if bool((bound & (appearance_class_id != gathered_class)).any()):
                raise ValueError("A class must match the class of its bound G slot")

        if retained_slot_mask is not None:
            _require_shape(
                "retained_slot_mask",
                retained_slot_mask,
                tuple(geometry.slot_valid.shape),
            )
            if retained_slot_mask.dtype != torch.bool:
                raise TypeError("retained_slot_mask must have bool dtype")
            if retained_slot_mask.device != self.geometry_idx.device:
                raise ValueError("retained_slot_mask and A must share one device")
            retained, _ = gather_actor_slots(
                retained_slot_mask,
                self.geometry_idx,
                selection_mask=bound,
                fill_value=False,
            )
            if bool((bound & ~retained).any()):
                raise ValueError(
                    "G partial dropout cannot delete an A-bound actor; invalidate "
                    "the corresponding A slot in the same operation"
                )
        return self

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AppearanceBindingCondition":
        updates = {
            name: getattr(self, name).to(device=device, non_blocking=non_blocking)
            for name in self.__dataclass_fields__
        }
        return replace(self, **updates)

    def null_like(self) -> "AppearanceBindingCondition":
        """Return a leak-free A=NULL condition for TCMG/TC branches."""

        return replace(
            self,
            appearance_tokens=torch.zeros_like(self.appearance_tokens),
            appearance_mask=torch.zeros_like(self.appearance_mask),
            canonical_uv=torch.zeros_like(self.canonical_uv),
            geometry_idx=torch.full_like(self.geometry_idx, -1),
            binding_valid=torch.zeros_like(self.binding_valid),
            appearance_mode=torch.full_like(
                self.appearance_mode, int(AppearanceMode.NULL)
            ),
        )

    def invalidate_bindings(
        self,
        invalid: torch.Tensor,
    ) -> "AppearanceBindingCondition":
        """Synchronously invalidate selected A slots and clear their payload."""

        _require_shape("invalid", invalid, tuple(self.binding_valid.shape))
        if invalid.dtype != torch.bool:
            raise TypeError("invalid must have bool dtype")
        if invalid.device != self.binding_valid.device:
            raise ValueError("invalid and A must share one device")
        invalid = invalid & self.binding_valid
        remaining = self.binding_valid & ~invalid
        row_mode = torch.where(
            remaining.any(dim=-1),
            torch.full_like(self.appearance_mode, int(AppearanceMode.REAL)),
            torch.full_like(self.appearance_mode, int(AppearanceMode.NULL)),
        )
        return replace(
            self,
            appearance_tokens=torch.where(
                invalid[..., None, None],
                torch.zeros_like(self.appearance_tokens),
                self.appearance_tokens,
            ),
            appearance_mask=self.appearance_mask & ~invalid[..., None],
            canonical_uv=torch.where(
                invalid[..., None, None],
                torch.zeros_like(self.canonical_uv),
                self.canonical_uv,
            ),
            geometry_idx=torch.where(
                invalid,
                torch.full_like(self.geometry_idx, -1),
                self.geometry_idx,
            ),
            binding_valid=remaining,
            appearance_mode=row_mode,
        )

    def permute_bindings(
        self,
        permutation: Sequence[int] | torch.Tensor,
    ) -> "AppearanceBindingCondition":
        """Permute the unordered A axis; no slot embedding exists to observe it."""

        permutation = torch.as_tensor(
            permutation,
            dtype=torch.int64,
            device=self.geometry_idx.device,
        )
        if tuple(permutation.shape) != (self.num_bindings,):
            raise ValueError("permutation must have shape [Ka]")
        if sorted(permutation.cpu().tolist()) != list(range(self.num_bindings)):
            raise ValueError("permutation must contain every A slot exactly once")
        return replace(
            self,
            appearance_tokens=self.appearance_tokens[:, permutation],
            appearance_mask=self.appearance_mask[:, permutation],
            canonical_uv=self.canonical_uv[:, permutation],
            geometry_idx=self.geometry_idx[:, permutation],
            binding_valid=self.binding_valid[:, permutation],
        )

    def canonicalized(self) -> "AppearanceBindingCondition":
        """Return a deterministic geometry-id order for the unordered A set.

        Attention is mathematically permutation invariant, but reducing the
        same values in a different slot order can still change floating-point
        roundoff.  Canonicalizing at the model boundary makes D15's strict
        output-invariance contract literal, while adding no slot identity or
        spatial information to A.
        """

        sentinel = torch.iinfo(self.geometry_idx.dtype).max
        keys = torch.where(
            self.binding_valid,
            self.geometry_idx,
            torch.full_like(self.geometry_idx, sentinel),
        )
        order = torch.argsort(keys, dim=1, stable=True)

        def gather_axis(value: torch.Tensor) -> torch.Tensor:
            index = order.reshape(
                self.batch_size,
                self.num_bindings,
                *([1] * (value.ndim - 2)),
            ).expand_as(value)
            return torch.gather(value, 1, index)

        return replace(
            self,
            appearance_tokens=gather_axis(self.appearance_tokens),
            appearance_mask=gather_axis(self.appearance_mask),
            canonical_uv=gather_axis(self.canonical_uv),
            geometry_idx=gather_axis(self.geometry_idx),
            binding_valid=gather_axis(self.binding_valid),
        )


@dataclass(frozen=True)
class GatheredAppearanceGeometry:
    """Spatial addresses and masks derived exclusively from projected G."""

    bbox_patch: torch.Tensor
    token_patch_xy: torch.Tensor
    pooled_patch_xy: torch.Tensor
    addr_ok: torch.Tensor
    token_attention_mask: torch.Tensor
    effective_binding_valid: torch.Tensor
    all_window_out_of_frustum: torch.Tensor
    invalid_all_window_count: torch.Tensor


def gather_appearance_geometry(
    appearance: AppearanceBindingCondition,
    projected: ProjectedActorGeometry,
) -> GatheredAppearanceGeometry:
    """Resolve A token addresses while enforcing all three gather invariants.

    1. Indices are clamped before gather and masked afterwards.
    2. Tokens without a valid in-frustum address are removed from attention.
    3. Bindings that have no usable frame in the whole window are marked
       invalid in ``effective_binding_valid`` and counted per batch row.
    """

    appearance.validate()
    projected.validate()
    if appearance.batch_size != projected.batch_size:
        raise ValueError("A and ProjectedActorGeometry batch sizes differ")
    if appearance.geometry_idx.device != projected.valid.device:
        raise ValueError("A and ProjectedActorGeometry must share one device")

    bound = appearance.binding_valid & (
        appearance.appearance_mode == int(AppearanceMode.REAL)
    )[:, None]
    bbox, in_range = gather_actor_slots(
        projected.bbox_patch,
        appearance.geometry_idx,
        selection_mask=bound,
        fill_value=0.0,
    )
    if bool((bound & ~in_range).any()):
        raise ValueError("binding-valid geometry_idx exceeds projected G actor axis")
    projected_valid, _ = gather_actor_slots(
        projected.valid,
        appearance.geometry_idx,
        selection_mask=bound,
        fill_value=False,
    )
    frame_support, _ = gather_actor_slots(
        projected.frame_support,
        appearance.geometry_idx,
        selection_mask=bound,
        fill_value=False,
    )
    # A tokens may acquire a target-frame address wherever the clipped
    # silhouette used by early G exists.  The full-corner late A context has
    # the stricter ``metric_support`` gate indirectly through the actor
    # z-buffer weights, because its 27-D reader cannot represent clipped
    # near/far-crossing corners without fabricating log-depths.
    addr_ok = projected_valid & frame_support & in_range[..., None]
    all_window_out = bound & ~addr_ok.any(dim=-1)
    effective_binding_valid = bound & ~all_window_out
    addr_ok = addr_ok & effective_binding_valid[..., None]

    bbox = torch.where(addr_ok[..., None], bbox, torch.zeros_like(bbox))
    x0, y0, x1, y1 = bbox.unbind(dim=-1)
    canonical_u = appearance.canonical_uv[..., 0][:, :, None, :]
    canonical_v = appearance.canonical_uv[..., 1][:, :, None, :]
    token_x = x0[..., None] + canonical_u * (x1 - x0)[..., None]
    token_y = y0[..., None] + canonical_v * (y1 - y0)[..., None]
    token_patch_xy = torch.stack((token_x, token_y), dim=-1)
    token_attention_mask = (
        appearance.appearance_mask[:, :, None, :] & addr_ok[..., None]
    )
    token_patch_xy = torch.where(
        token_attention_mask[..., None],
        token_patch_xy,
        torch.zeros_like(token_patch_xy),
    )
    pooled_patch_xy = torch.stack(
        ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
        dim=-1,
    )
    pooled_patch_xy = torch.where(
        addr_ok[..., None], pooled_patch_xy, torch.zeros_like(pooled_patch_xy)
    )
    return GatheredAppearanceGeometry(
        bbox_patch=bbox,
        token_patch_xy=token_patch_xy,
        pooled_patch_xy=pooled_patch_xy,
        addr_ok=addr_ok,
        token_attention_mask=token_attention_mask,
        effective_binding_valid=effective_binding_valid,
        all_window_out_of_frustum=all_window_out,
        invalid_all_window_count=all_window_out.sum(dim=-1, dtype=torch.int64),
    )


__all__ = [
    "APPEARANCE_TOKEN_DIM",
    "CANONICAL_ALPHA_PATCH_THRESHOLD",
    "CANONICAL_CROP_MARGIN",
    "CANONICAL_LONG_SIDE_FRACTION",
    "MAX_APPEARANCE_BINDINGS",
    "MAX_APPEARANCE_TOKENS",
    "AppearanceBindingCondition",
    "AppearanceMode",
    "GatheredAppearanceGeometry",
    "appearance_alpha_to_patch_mask",
    "canonical_patch_uv",
    "canonicalize_appearance_reference",
    "gather_appearance_geometry",
    "sample_appearance_tokens",
]
