"""Layout-v2 task states, leak-free branches, and chained CFG algebra.

This module contains no dataset or model policy.  It only freezes the three
legal training tasks and the four inference branches from the implementation
contract so training, validation, and offline inference cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Mapping, Sequence

import torch

from datasets.tools.hdmap_schema import (
    MAP_METRIC_RESERVED_ZERO_GROUPS,
    RASTER_ACTOR_CHANNELS,
    RASTER_MAP_CHANNELS,
    RASTER_RESERVED_ZERO_CHANNELS,
    RASTER_SCHEMA_HASH,
    RASTER_SIGNED_CHANNELS,
    RASTER_STATIC_VALID_CHANNEL,
)
from dggt.utils.actor_geometry_condition import (
    ActorGeometryCondition,
    LayoutMode,
    ProjectedActorGeometry,
)
from dggt.utils.appearance_binding_condition import AppearanceBindingCondition


class MapMode(IntEnum):
    NULL = 0
    EMPTY = 1
    PRESENT = 2


class LayoutTask(IntEnum):
    TC = 0
    TCMG = 1
    TCMGA = 2


LAYOUT_TASK_PROBABILITIES = (0.10, 0.50, 0.40)
LAYOUT_CONDITION_VERSION = "layout_v2"


def assert_neutral_raster_rows(
    raster: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    channel_start: int,
    channel_end: int,
    label: str,
) -> None:
    """Assert exact uint8 zero points for absent layout rows."""

    if raster.dtype != torch.uint8 or raster.ndim != 5:
        raise TypeError("layout raster must be uint8 [B,S,C,H,W]")
    mask = torch.as_tensor(row_mask, device=raster.device, dtype=torch.bool)
    if tuple(mask.shape) != (int(raster.shape[0]),):
        raise ValueError("absent-row mask must be bool [B]")
    if not bool(mask.any()):
        return
    start, end = int(channel_start), int(channel_end)
    if not (0 <= start < end <= int(raster.shape[2])):
        raise ValueError("invalid raster channel interval")
    signed = tuple(
        channel for channel in RASTER_SIGNED_CHANNELS if start <= channel < end
    )
    unsigned = tuple(
        channel for channel in range(start, end) if channel not in signed
    )
    rows = raster[mask]
    if signed and bool(rows[:, :, signed].ne(127).any()):
        raise ValueError(f"{label} rows require raw 127 in signed zero-point channels")
    if unsigned and bool(rows[:, :, unsigned].ne(0).any()):
        raise ValueError(f"{label} rows must not leak raster values")


def neutralize_raster_rows(
    raster: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    channel_start: int,
    channel_end: int,
) -> torch.Tensor:
    """Return a copy with selected rows set to the schema wire zero point."""

    mask = torch.as_tensor(row_mask, device=raster.device, dtype=torch.bool)
    if tuple(mask.shape) != (int(raster.shape[0]),):
        raise ValueError("neutral-row mask must be bool [B]")
    if not bool(mask.any()):
        return raster
    result = raster.clone()
    start, end = int(channel_start), int(channel_end)
    result[mask, :, start:end] = 0
    for channel in RASTER_SIGNED_CHANNELS:
        if start <= int(channel) < end:
            result[mask, :, int(channel)] = 127
    return result


def map_support_rows(
    raster: torch.Tensor,
    map_metric: torch.Tensor,
) -> torch.Tensor:
    """Return per-row factual static-map support for one window.

    This is the single definition of "does this window actually carry HD-map
    evidence".  The dataloader's outer-request packing and the sampler's window
    slicing both recompute a factual ``MapMode`` and must agree bit for bit, so
    neither may spell the predicate out a second time.
    """

    support = raster[:, :, RASTER_STATIC_VALID_CHANNEL].ne(0).flatten(1).any(dim=1)
    valid_group_start = max(MAP_METRIC_RESERVED_ZERO_GROUPS) + 1
    return support | (
        map_metric[..., valid_group_start:, 3].ne(0).flatten(1).any(dim=1)
    )


def factual_map_mode(
    outer_map_mode: torch.Tensor,
    raster: torch.Tensor,
    map_metric: torch.Tensor,
) -> torch.Tensor:
    """Downgrade ``PRESENT`` to factual ``EMPTY`` where a window has no support.

    ``NULL`` is preserved: the absence of information never becomes the factual
    assertion that the scene has no map.
    """

    support = map_support_rows(raster, map_metric)
    map_null = outer_map_mode == int(MapMode.NULL)
    return torch.where(
        map_null,
        torch.full_like(outer_map_mode, int(MapMode.NULL)),
        torch.where(
            support,
            torch.full_like(outer_map_mode, int(MapMode.PRESENT)),
            torch.full_like(outer_map_mode, int(MapMode.EMPTY)),
        ),
    )


def sample_layout_tasks(
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample only ``TC/TCMG/TCMGA`` with the frozen 0.10/0.50/0.40 law."""

    batch_size = int(batch_size)
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    probabilities = torch.tensor(
        LAYOUT_TASK_PROBABILITIES, device=device, dtype=torch.float32
    )
    return torch.multinomial(
        probabilities,
        batch_size,
        replacement=True,
        generator=generator,
    ).to(dtype=torch.int8)


def layout_to_gauge_scale(global_step: int, *, upper: float = 1.0) -> float:
    """Return the mandatory 0 -> 1 layout-to-gauge gradient ramp."""

    step = int(global_step)
    upper = float(upper)
    if not 0.0 <= upper <= 1.0:
        raise ValueError("layout_to_gauge_grad_scale upper bound must be in [0,1]")
    if step < 5_000:
        return 0.0
    if step < 15_000:
        return upper * float(step - 5_000) / 10_000.0
    return upper


@dataclass(frozen=True)
class LayoutConditionBatch:
    """All runtime layout inputs carried together to prevent partial refreshes."""

    raster: torch.Tensor
    map_metric: torch.Tensor
    actor_geometry: ActorGeometryCondition
    projected_actor_geometry: ProjectedActorGeometry
    appearance: AppearanceBindingCondition
    map_mode: torch.Tensor
    raster_schema_hash: str

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.raster.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.raster.shape[1])

    def validate(self) -> "LayoutConditionBatch":
        if self.raster.ndim != 5 or int(self.raster.shape[2]) != 33:
            raise ValueError(
                "layout raster must be [B,S,33,H,W], got "
                f"{tuple(self.raster.shape)}"
            )
        if self.raster.dtype != torch.uint8:
            raise TypeError("layout raster must retain its uint8 wire dtype")
        b, s = int(self.raster.shape[0]), int(self.raster.shape[1])
        if (
            self.map_metric.ndim != 5
            or tuple(self.map_metric.shape[:2]) != (b, s)
            or int(self.map_metric.shape[2]) != 25 * 37
        ):
            raise ValueError("map_metric must be [B,S,P,Gm,4]")
        if int(self.map_metric.shape[-2]) != 5 or int(self.map_metric.shape[-1]) != 4:
            raise ValueError("map_metric must use five groups and four fields")
        if self.map_metric.dtype != torch.float32:
            raise TypeError("map_metric must be float32")
        if tuple(self.map_mode.shape) != (b,) or self.map_mode.dtype != torch.int8:
            raise ValueError("map_mode must be int8 [B]")
        if bool(((self.map_mode < int(MapMode.NULL)) | (
            self.map_mode > int(MapMode.PRESENT)
        )).any()):
            raise ValueError("map_mode contains an unknown state")
        if self.actor_geometry.batch_size != b or self.actor_geometry.num_frames != s:
            raise ValueError("actor geometry does not match raster batch/time axes")
        projected = self.projected_actor_geometry
        if projected.batch_size != b or projected.num_frames != s:
            raise ValueError("projected actor cache does not match raster batch/time axes")
        if projected.num_slots != self.actor_geometry.num_slots:
            raise ValueError("projected actor cache changed the fixed G slot axis")
        if self.appearance.batch_size != b:
            raise ValueError("appearance condition does not match layout batch")
        if self.raster_schema_hash != RASTER_SCHEMA_HASH:
            raise ValueError("raster_schema_hash does not match the frozen schema")
        if bool(self.raster[:, :, RASTER_RESERVED_ZERO_CHANNELS].any()):
            raise ValueError("layout raster contains excluded static-map features")
        if bool(self.map_metric[..., MAP_METRIC_RESERVED_ZERO_GROUPS, :].any()):
            raise ValueError("map_metric contains excluded lane-centerline features")
        devices = {
            self.raster.device,
            self.map_metric.device,
            self.map_mode.device,
            self.actor_geometry.slot_valid.device,
            projected.valid.device,
            self.appearance.binding_valid.device,
        }
        if len(devices) != 1:
            raise ValueError("all layout-condition tensors must share one device")
        absent_map = (self.map_mode == int(MapMode.NULL)) | (
            self.map_mode == int(MapMode.EMPTY)
        )
        if bool(self.map_metric[absent_map].any()):
            raise ValueError("M=NULL rows must not leak metric values")
        if bool(absent_map.any()):
            assert_neutral_raster_rows(
                self.raster,
                absent_map,
                channel_start=0,
                channel_end=22,
                label="M=NULL/EMPTY",
            )
        layout_mode = self.actor_geometry.layout_mode
        absent_actor = (layout_mode == int(LayoutMode.NULL)) | (
            layout_mode == int(LayoutMode.EMPTY)
        )
        if bool(absent_actor.any()):
            assert_neutral_raster_rows(
                self.raster,
                absent_actor,
                channel_start=22,
                channel_end=33,
                label="G=NULL/EMPTY",
            )
        return self

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "LayoutConditionBatch":
        return replace(
            self,
            raster=self.raster.to(device=device, non_blocking=non_blocking),
            map_metric=self.map_metric.to(device=device, non_blocking=non_blocking),
            actor_geometry=self.actor_geometry.to(device, non_blocking=non_blocking),
            projected_actor_geometry=self.projected_actor_geometry.to(
                device, non_blocking=non_blocking
            ),
            appearance=self.appearance.to(device, non_blocking=non_blocking),
            map_mode=self.map_mode.to(device=device, non_blocking=non_blocking),
        )

    def slice_frames(self, frame_index) -> "LayoutConditionBatch":
        """Slice one outer-request projection and recompute factual modes.

        Internal sampler windows never reproject or re-anchor geometry.  They
        inherit the outer request's stable actor slots and merely slice the
        shared caches.  A FULL row whose slice has no actor support becomes
        factual EMPTY; PARTIAL remains PARTIAL because an incomplete supplied
        set cannot establish that the scene is empty.  Appearance stays
        scene-global except when G becomes factual EMPTY, where the D3 contract
        requires A=NULL.
        """

        if isinstance(frame_index, int):
            frame_index = [int(frame_index)]
        elif torch.is_tensor(frame_index) and frame_index.ndim == 0:
            frame_index = [int(frame_index.item())]

        raster = self.raster[:, frame_index]
        map_metric = self.map_metric[:, frame_index]
        geometry = self.actor_geometry.slice_frames(frame_index)
        projected = self.projected_actor_geometry.slice_frames(frame_index)

        map_mode = factual_map_mode(self.map_mode, raster, map_metric)
        absent_map = map_mode != int(MapMode.PRESENT)
        raster = neutralize_raster_rows(
            raster,
            absent_map,
            channel_start=RASTER_MAP_CHANNELS[0],
            channel_end=RASTER_MAP_CHANNELS[1],
        )
        map_metric = torch.where(
            absent_map.view(-1, 1, 1, 1, 1),
            torch.zeros_like(map_metric),
            map_metric,
        )

        actor_support = projected.frame_support.flatten(1).any(dim=1)
        outer_layout_mode = geometry.layout_mode
        full_became_empty = (
            outer_layout_mode == int(LayoutMode.FULL)
        ) & ~actor_support
        layout_mode = torch.where(
            full_became_empty,
            torch.full_like(outer_layout_mode, int(LayoutMode.EMPTY)),
            outer_layout_mode,
        )
        actor_empty = layout_mode == int(LayoutMode.EMPTY)
        if bool(actor_empty.any()):
            def clear_actor(value: torch.Tensor, fill_value=0) -> torch.Tensor:
                row_mask = actor_empty.view(
                    actor_empty.shape[0], *([1] * (value.ndim - 1))
                )
                return torch.where(
                    row_mask,
                    torch.full_like(value, fill_value),
                    value,
                )

            raw_track_key = [list(row) for row in geometry.raw_track_key]
            for row in actor_empty.nonzero(as_tuple=False).flatten().cpu().tolist():
                raw_track_key[row] = ["" for _ in range(geometry.num_slots)]
            geometry = replace(
                geometry,
                slot_track_id=clear_actor(geometry.slot_track_id, -1),
                class_id=clear_actor(geometry.class_id, -1),
                corners_world=clear_actor(geometry.corners_world),
                velocity_world=clear_actor(geometry.velocity_world),
                box_size=clear_actor(geometry.box_size),
                yaw=clear_actor(geometry.yaw),
                is_moving=clear_actor(geometry.is_moving),
                track_valid=clear_actor(geometry.track_valid),
                slot_valid=clear_actor(geometry.slot_valid),
                layout_mode=layout_mode,
                raw_track_key=raw_track_key,
            )
            projected = replace(
                projected,
                **{
                    name: clear_actor(getattr(projected, name))
                    for name in projected.__dataclass_fields__
                },
            )
            raster = neutralize_raster_rows(
                raster,
                actor_empty,
                channel_start=RASTER_ACTOR_CHANNELS[0],
                channel_end=RASTER_ACTOR_CHANNELS[1],
            )
            appearance = self.appearance.invalidate_bindings(
                actor_empty[:, None].expand_as(self.appearance.binding_valid)
            )
        else:
            geometry = replace(geometry, layout_mode=layout_mode)
            appearance = self.appearance

        return replace(
            self,
            raster=raster,
            map_metric=map_metric,
            actor_geometry=geometry,
            projected_actor_geometry=projected,
            appearance=appearance,
            map_mode=map_mode,
        )

    def without_appearance(self) -> "LayoutConditionBatch":
        return replace(self, appearance=self.appearance.null_like())

    def without_layout(self) -> "LayoutConditionBatch":
        """Construct the leak-free ``TC`` branch (NULL is not factual EMPTY)."""

        raster = neutralize_raster_rows(
            self.raster,
            torch.ones(
                self.batch_size,
                dtype=torch.bool,
                device=self.raster.device,
            ),
            channel_start=RASTER_MAP_CHANNELS[0],
            channel_end=RASTER_ACTOR_CHANNELS[1],
        )
        return replace(
            self,
            raster=raster,
            map_metric=torch.zeros_like(self.map_metric),
            actor_geometry=self.actor_geometry.null_like(),
            projected_actor_geometry=self.projected_actor_geometry.null_like(),
            appearance=self.appearance.null_like(),
            map_mode=torch.full_like(self.map_mode, int(MapMode.NULL)),
        )


def required_cfg_branches(
    *,
    text_scale: float,
    layout_scale: float,
    appearance_scale: float,
    appearance_present: bool,
) -> tuple[str, ...]:
    """Return the exact lazy branch set required by chained layout CFG."""

    branches = ["full"]
    if float(text_scale) != 1.0:
        branches.append("no_text_full")
    if float(layout_scale) != 1.0:
        branches.extend(("appearance_dropped", "layout_dropped"))
    if appearance_present and float(appearance_scale) != 1.0:
        branches.append("appearance_dropped")
    # Preserve semantic ordering while avoiding a duplicate TCMG evaluation.
    return tuple(dict.fromkeys(branches))


LAYOUT_GUIDANCE_PROTECTED_KEYS = frozenset(
    {"sky", "gauge", "sky_mask_logits", "sky_mask_refined_logits"}
)
"""Outputs that keep ``s_L = s_A = 1`` (§8.3).

``sky`` and ``gauge`` are there because amplifying layout must not reshape the
sky dome or move the predicted metric scale.  The two mask logits are there for
a different reason: they are threshold-sensitive geometry gates, not a velocity
field.  Export binarizes them at ``sky_probability < 0.5``, i.e. at logit zero,
so a linear extrapolation of a logit from ``-0.05`` to ``+0.03`` flips a
Gaussian from kept to culled and changes the point-cloud topology.  Protecting
them does not blind the mask to layout: the ``full`` branch already carries the
complete M/G/A condition, so the mask stays conditioned — it is simply the
``s_L = s_A = 1`` mask, rather than an extrapolation of an auxiliary head whose
decision threshold was never calibrated for it.
"""


def combine_chained_cfg(
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    text_scale: float,
    layout_scale: float,
    appearance_scale: float,
    appearance_present: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply §8.1 while forcing layout/appearance scale 1 for the §8.3 set."""

    if "full" not in outputs:
        raise KeyError("chained CFG requires the full branch")
    full = outputs["full"]
    result: dict[str, torch.Tensor] = {}
    for key, value in full.items():
        if not torch.is_tensor(value):
            continue
        combined = value
        if float(text_scale) != 1.0:
            no_text = outputs.get("no_text_full")
            if no_text is None or key not in no_text:
                raise KeyError(f"missing no_text_full output for {key}")
            combined = combined + (float(text_scale) - 1.0) * (
                value - no_text[key]
            )
        # Layout and appearance guidance are forbidden for the protected set.
        if key not in LAYOUT_GUIDANCE_PROTECTED_KEYS:
            if float(layout_scale) != 1.0:
                tcmg = outputs.get("appearance_dropped")
                tc = outputs.get("layout_dropped")
                if tcmg is None or tc is None or key not in tcmg or key not in tc:
                    raise KeyError(f"missing TCMG/TC output for {key}")
                combined = combined + (float(layout_scale) - 1.0) * (
                    tcmg[key] - tc[key]
                )
            if bool(appearance_present) and float(appearance_scale) != 1.0:
                tcmg = outputs.get("appearance_dropped")
                if tcmg is None or key not in tcmg:
                    raise KeyError(f"missing TCMG output for {key}")
                combined = combined + (float(appearance_scale) - 1.0) * (
                    value - tcmg[key]
                )
        result[key] = combined
    return result


def assert_layout_overlap_consistent(
    left: LayoutConditionBatch,
    right: LayoutConditionBatch,
    *,
    left_frames: Sequence[int] | torch.Tensor,
    right_frames: Sequence[int] | torch.Tensor,
) -> None:
    """Fail fast when independently built sliding windows disagree in overlap.

    Actor slots are matched by scene-local ``slot_track_id`` rather than by
    their padded axis position.  Raster/map fields must be byte/value exact;
    shared actors must carry identical world geometry and projected boxes.
    """

    left.validate()
    right.validate()
    if left.batch_size != right.batch_size:
        raise ValueError("overlapping layout windows have different batch sizes")
    left_index = torch.as_tensor(left_frames, dtype=torch.long).reshape(-1)
    right_index = torch.as_tensor(right_frames, dtype=torch.long).reshape(-1)
    if int(left_index.numel()) != int(right_index.numel()) or int(left_index.numel()) == 0:
        raise ValueError("overlap frame index lists must be non-empty and equally sized")
    if bool((left_index < 0).any()) or bool((left_index >= left.num_frames).any()):
        raise IndexError("left overlap frame index is out of range")
    if bool((right_index < 0).any()) or bool((right_index >= right.num_frames).any()):
        raise IndexError("right overlap frame index is out of range")
    left_index = left_index.to(device=left.raster.device)
    right_index = right_index.to(device=right.raster.device)
    if not torch.equal(
        left.raster.index_select(1, left_index),
        right.raster.index_select(1, right_index),
    ):
        raise ValueError("overlap layout_raster values differ")
    if not torch.equal(
        left.map_metric.index_select(1, left_index),
        right.map_metric.index_select(1, right_index),
    ):
        raise ValueError("overlap map_metric values differ")

    left_g, right_g = left.actor_geometry, right.actor_geometry
    left_p, right_p = left.projected_actor_geometry, right.projected_actor_geometry
    actor_fields = (
        "corners_world",
        "velocity_world",
        "box_size",
        "yaw",
        "is_moving",
        "track_valid",
    )
    projected_fields = (
        "bbox_patch",
        "patch_weight",
        "log_z_patch",
        "silhouette_uv",
        "silhouette_vertex_valid",
        "corners_camera",
        "uv_corners",
        "velocity_camera",
        "uv_center",
        "log_z_w",
        "center_depth_valid",
        "frame_support",
        "metric_support",
        "in_frustum",
        "valid",
    )
    for row in range(left.batch_size):
        left_slots = {
            int(left_g.slot_track_id[row, slot].item()): slot
            for slot in left_g.slot_valid[row].nonzero(as_tuple=False).flatten().tolist()
        }
        right_slots = {
            int(right_g.slot_track_id[row, slot].item()): slot
            for slot in right_g.slot_valid[row].nonzero(as_tuple=False).flatten().tolist()
        }
        # A window that downgraded FULL to factual EMPTY clears every slot, so
        # the intersection below goes empty and the field comparison would pass
        # vacuously.  That is precisely the case worth checking: a track the
        # other side still supports inside the overlap proves the downgrade was
        # wrong.  PARTIAL is exempt because an incomplete supplied set may
        # legitimately omit tracks.
        partial = int(LayoutMode.PARTIAL)
        if (
            int(left_g.layout_mode[row].item()) != partial
            and int(right_g.layout_mode[row].item()) != partial
        ):
            for slots, projected, index, side in (
                (left_slots, left_p, left_index, "left"),
                (right_slots, right_p, right_index, "right"),
            ):
                other = right_slots if side == "left" else left_slots
                for track_id, slot in slots.items():
                    if track_id in other:
                        continue
                    if bool(
                        projected.frame_support[row, slot]
                        .index_select(0, index)
                        .any()
                    ):
                        raise ValueError(
                            f"track {track_id} is supported in the {side} overlap "
                            "but absent from the other window"
                        )
        for track_id in sorted(set(left_slots).intersection(right_slots)):
            ls, rs = left_slots[track_id], right_slots[track_id]
            for name in actor_fields:
                left_value = getattr(left_g, name)[row, ls].index_select(0, left_index)
                right_value = getattr(right_g, name)[row, rs].index_select(0, right_index)
                if not torch.equal(left_value, right_value):
                    raise ValueError(
                        f"shared track {track_id} has inconsistent overlap {name}"
                    )
            for name in projected_fields:
                left_value = getattr(left_p, name)[row, ls].index_select(0, left_index)
                right_value = getattr(right_p, name)[row, rs].index_select(0, right_index)
                if not torch.equal(left_value, right_value):
                    raise ValueError(
                        f"shared track {track_id} has inconsistent projected overlap {name}"
                    )


__all__ = [
    "LAYOUT_CONDITION_VERSION",
    "LAYOUT_TASK_PROBABILITIES",
    "LayoutConditionBatch",
    "LayoutTask",
    "MapMode",
    "assert_neutral_raster_rows",
    "assert_layout_overlap_consistent",
    "combine_chained_cfg",
    "layout_to_gauge_scale",
    "neutralize_raster_rows",
    "required_cfg_branches",
    "sample_layout_tasks",
]
