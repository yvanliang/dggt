"""Dataset-side construction of full actor geometry and appearance bindings.

This module deliberately does not read HD-map files.  It converts the existing
``instances_info.json`` payload into the fixed-padded :class:`ActorGeometryCondition`
used by training and inference.  Geometry is built before appearance references
are considered, so missing reference imagery can never remove an actor from G.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from dggt.utils.actor_geometry_condition import (
    ActorGeometryCondition,
    CameraSpec,
    LayoutMode,
    ProjectedActorGeometry,
    project_actor_geometry,
)


SUPPORTED_ACTOR_CLASS_IDS: dict[str, int] = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
}
ACTOR_CLASS_NAMES: tuple[str, ...] = ("vehicle", "pedestrian", "cyclist")
DEFAULT_LAYOUT_MAX_ACTORS = 96
MOVING_SPEED_THRESHOLD_MPS = 0.5


@dataclass(frozen=True)
class ActorBuildStats:
    """Non-tensor diagnostics carried into the run summary."""

    annotated_count: int
    supported_count: int
    eligible_count: int
    pre_cap_count: int
    post_cap_count: int
    overflow: int
    ignored_unsupported_count: int
    ignored_invalid_geometry_count: int
    ignored_outside_requested_view_count: int
    layout_mode: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class AppearanceSamplingStats:
    selected_count: int
    candidate_count: int
    class_buckets: dict[str, int]
    distance_buckets: dict[str, int]
    area_buckets: dict[str, int]
    motion_buckets: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_count": int(self.selected_count),
            "candidate_count": int(self.candidate_count),
            "class_buckets": dict(self.class_buckets),
            "distance_buckets": dict(self.distance_buckets),
            "area_buckets": dict(self.area_buckets),
            "motion_buckets": dict(self.motion_buckets),
        }


def _normalize_class_name(value: object) -> str | None:
    name = str(value).strip().lower()
    aliases = {
        "car": "vehicle",
        "vehicles": "vehicle",
        "pedestrians": "pedestrian",
        "person": "pedestrian",
        "bicycle": "cyclist",
        "bicyclist": "cyclist",
        "cyclists": "cyclist",
    }
    name = aliases.get(name, name)
    return name if name in SUPPORTED_ACTOR_CLASS_IDS else None


def cuboid_corners_world(
    object_to_world: np.ndarray,
    size_lwh: np.ndarray,
) -> np.ndarray:
    """Return the fixed binary-axis corner order consumed by the projector."""

    half = 0.5 * size_lwh
    signs = np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, +1.0),
            (-1.0, +1.0, -1.0),
            (-1.0, +1.0, +1.0),
            (+1.0, -1.0, -1.0),
            (+1.0, -1.0, +1.0),
            (+1.0, +1.0, -1.0),
            (+1.0, +1.0, +1.0),
        ],
        dtype=np.float64,
    )
    local = signs * half[None]
    return (
        local @ object_to_world[:3, :3].T
        + object_to_world[:3, 3][None]
    )


def _track_velocity_world(
    frames: Sequence[int],
    centers: np.ndarray,
    frame_index: int,
    *,
    frames_per_second: float,
) -> np.ndarray:
    """Central finite difference of actor centres, excluding ego motion."""

    if len(frames) < 2:
        return np.zeros((3,), dtype=np.float32)
    rank = frames.index(int(frame_index))
    left = max(0, rank - 1)
    right = min(len(frames) - 1, rank + 1)
    if left == right:
        return np.zeros((3,), dtype=np.float32)
    delta_frames = int(frames[right]) - int(frames[left])
    if delta_frames <= 0:
        return np.zeros((3,), dtype=np.float32)
    dt = float(delta_frames) / float(frames_per_second)
    return ((centers[right] - centers[left]) / dt).astype(np.float32)


def _empty_geometry(
    *,
    num_frames: int,
    layout_max_actors: int,
    mode: LayoutMode,
    device: torch.device | str = "cpu",
) -> ActorGeometryCondition:
    kg, s = int(layout_max_actors), int(num_frames)
    return ActorGeometryCondition(
        slot_track_id=torch.full((1, kg), -1, dtype=torch.int64, device=device),
        class_id=torch.full((1, kg), -1, dtype=torch.int8, device=device),
        corners_world=torch.zeros((1, kg, s, 8, 3), dtype=torch.float64, device=device),
        velocity_world=torch.zeros((1, kg, s, 3), dtype=torch.float32, device=device),
        box_size=torch.zeros((1, kg, s, 3), dtype=torch.float32, device=device),
        yaw=torch.zeros((1, kg, s), dtype=torch.float32, device=device),
        is_moving=torch.zeros((1, kg, s), dtype=torch.bool, device=device),
        track_valid=torch.zeros((1, kg, s), dtype=torch.bool, device=device),
        slot_valid=torch.zeros((1, kg), dtype=torch.bool, device=device),
        layout_mode=torch.full((1,), int(mode), dtype=torch.int8, device=device),
        raw_track_key=[[""] * kg],
    )


def _candidate_geometry(
    candidates: Sequence[dict[str, Any]],
    *,
    num_frames: int,
    world_to_anchor: torch.Tensor,
) -> ActorGeometryCondition:
    """Materialize an unpadded temporary G for projection-based eligibility."""

    kg, s = len(candidates), int(num_frames)
    corners = torch.zeros((1, kg, s, 8, 3), dtype=torch.float64)
    velocity = torch.zeros((1, kg, s, 3), dtype=torch.float32)
    box_size = torch.zeros((1, kg, s, 3), dtype=torch.float32)
    yaw = torch.zeros((1, kg, s), dtype=torch.float32)
    moving = torch.zeros((1, kg, s), dtype=torch.bool)
    track_valid = torch.zeros((1, kg, s), dtype=torch.bool)
    slot_ids = torch.full((1, kg), -1, dtype=torch.int64)
    class_ids = torch.full((1, kg), -1, dtype=torch.int8)
    raw_keys: list[str] = []
    anchor_rotation = world_to_anchor[0, :3, :3].detach().cpu().double().numpy()
    for slot, candidate in enumerate(candidates):
        slot_ids[0, slot] = int(candidate["slot_track_id"])
        class_ids[0, slot] = int(candidate["class_id"])
        raw_keys.append(str(candidate["raw_track_key"]))
        for local_index, frame in enumerate(candidate["frames"]):
            if frame is None:
                continue
            corners[0, slot, local_index] = torch.from_numpy(frame["corners"])
            velocity[0, slot, local_index] = torch.from_numpy(frame["velocity"])
            box_size[0, slot, local_index] = torch.from_numpy(frame["size"].astype(np.float32))
            actor_rotation_anchor = anchor_rotation @ frame["object_to_world"][:3, :3]
            yaw[0, slot, local_index] = float(
                math.atan2(actor_rotation_anchor[1, 0], actor_rotation_anchor[0, 0])
            )
            speed = float(np.linalg.norm(frame["velocity"]))
            moving[0, slot, local_index] = speed > MOVING_SPEED_THRESHOLD_MPS
            track_valid[0, slot, local_index] = True
    device = world_to_anchor.device
    return ActorGeometryCondition(
        slot_track_id=slot_ids.to(device),
        class_id=class_ids.to(device),
        corners_world=corners.to(device),
        velocity_world=velocity.to(device),
        box_size=box_size.to(device),
        yaw=yaw.to(device),
        is_moving=moving.to(device),
        track_valid=track_valid.to(device),
        slot_valid=torch.ones((1, kg), dtype=torch.bool, device=device),
        layout_mode=torch.full((1,), int(LayoutMode.FULL), dtype=torch.int8, device=device),
        raw_track_key=[raw_keys],
    )


def build_actor_geometry_from_instances(
    instances_info: Mapping[str, Any],
    frame_indices: Sequence[int],
    camera: CameraSpec,
    *,
    layout_max_actors: int = DEFAULT_LAYOUT_MAX_ACTORS,
    bound_track_keys: Iterable[str] = (),
    frames_per_second: float = 10.0,
) -> tuple[ActorGeometryCondition, ProjectedActorGeometry, ActorBuildStats]:
    """Build every eligible actor before applying the fixed actor-axis cap.

    Eligibility is exactly the production definition: supported class, finite
    positive geometry, centre optical depth in ``[0.5,120]`` metres, and at
    least one target-window frame in the requested front-camera frustum.
    Appearance availability is intentionally not an input.
    """

    if camera.batch_size != 1:
        raise ValueError("dataset actor construction accepts one scene at a time")
    indices = [int(value) for value in frame_indices]
    if not indices:
        raise ValueError("frame_indices must be non-empty")
    if camera.num_frames != len(indices):
        raise ValueError("CameraSpec frame count must match frame_indices")
    cap = int(layout_max_actors)
    if cap <= 0:
        raise ValueError("layout_max_actors must be positive")
    if not math.isfinite(float(frames_per_second)) or float(frames_per_second) <= 0.0:
        raise ValueError("frames_per_second must be finite and positive")

    annotated_count = len(instances_info)
    supported_count = 0
    invalid_count = 0
    unsupported_count = 0
    candidates: list[dict[str, Any]] = []
    for fallback_slot, (contiguous_key, raw_info) in enumerate(instances_info.items()):
        if not isinstance(raw_info, Mapping):
            invalid_count += 1
            continue
        class_name = _normalize_class_name(raw_info.get("class_name", ""))
        if class_name is None:
            unsupported_count += 1
            continue
        supported_count += 1
        annotations = raw_info.get("frame_annotations")
        if not isinstance(annotations, Mapping):
            invalid_count += 1
            continue
        raw_frames = annotations.get("frame_idx", [])
        raw_poses = annotations.get("obj_to_world", [])
        raw_sizes = annotations.get("box_size", [])
        if not (
            isinstance(raw_frames, Sequence)
            and isinstance(raw_poses, Sequence)
            and isinstance(raw_sizes, Sequence)
            and len(raw_frames) == len(raw_poses) == len(raw_sizes)
        ):
            invalid_count += 1
            continue
        try:
            frames = [int(value) for value in raw_frames]
            poses = np.asarray(raw_poses, dtype=np.float64)
            sizes = np.asarray(raw_sizes, dtype=np.float64)
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        if (
            poses.shape != (len(frames), 4, 4)
            or sizes.shape != (len(frames), 3)
            or not np.isfinite(poses).all()
            or not np.isfinite(sizes).all()
            or bool((sizes <= 0.0).any())
            or len(set(frames)) != len(frames)
        ):
            invalid_count += 1
            continue
        centers = poses[:, :3, 3]
        lookup = {frame: rank for rank, frame in enumerate(frames)}
        target_frames: list[dict[str, Any] | None] = []
        for frame_index in indices:
            rank = lookup.get(frame_index)
            if rank is None:
                target_frames.append(None)
                continue
            velocity = _track_velocity_world(
                frames,
                centers,
                frame_index,
                frames_per_second=float(frames_per_second),
            )
            target_frames.append(
                {
                    "object_to_world": poses[rank],
                    "size": sizes[rank],
                    "corners": cuboid_corners_world(poses[rank], sizes[rank]),
                    "velocity": velocity,
                }
            )
        if not any(frame is not None for frame in target_frames):
            continue
        try:
            slot_track_id = int(contiguous_key)
        except (TypeError, ValueError):
            slot_track_id = int(fallback_slot)
        candidates.append(
            {
                "slot_track_id": slot_track_id,
                "raw_track_key": str(
                    raw_info.get("raw_object_id", raw_info.get("id", contiguous_key))
                ),
                "class_id": SUPPORTED_ACTOR_CLASS_IDS[class_name],
                "frames": target_frames,
            }
        )

    if not candidates:
        geometry = _empty_geometry(
            num_frames=len(indices), layout_max_actors=cap, mode=LayoutMode.EMPTY,
            device=camera.world_to_anchor.device,
        )
        projected = project_actor_geometry(geometry, camera)
        stats = ActorBuildStats(
            annotated_count=annotated_count,
            supported_count=supported_count,
            eligible_count=0,
            pre_cap_count=0,
            post_cap_count=0,
            overflow=0,
            ignored_unsupported_count=unsupported_count,
            ignored_invalid_geometry_count=invalid_count,
            ignored_outside_requested_view_count=0,
            layout_mode=LayoutMode.EMPTY.name,
        )
        return geometry, projected, stats

    temporary = _candidate_geometry(
        candidates,
        num_frames=len(indices),
        world_to_anchor=camera.world_to_anchor,
    )
    temporary_projected = project_actor_geometry(temporary, camera)
    # ``frame_support`` is the sole per-frame eligibility mask.  It combines
    # track/slot validity, centre optical-z in the inclusive [near,120 m]
    # interval, and non-empty clipped cuboid/canvas overlap.  Never recover a
    # mask from the zero-padded log-depth values.
    eligible_mask = temporary_projected.frame_support.any(dim=-1)[0]
    eligible_indices = eligible_mask.nonzero(as_tuple=False).flatten().tolist()
    outside_count = len(candidates) - len(eligible_indices)
    if not eligible_indices:
        geometry = _empty_geometry(
            num_frames=len(indices), layout_max_actors=cap, mode=LayoutMode.EMPTY,
            device=camera.world_to_anchor.device,
        )
        projected = project_actor_geometry(geometry, camera)
        stats = ActorBuildStats(
            annotated_count=annotated_count,
            supported_count=supported_count,
            eligible_count=0,
            pre_cap_count=0,
            post_cap_count=0,
            overflow=0,
            ignored_unsupported_count=unsupported_count,
            ignored_invalid_geometry_count=invalid_count,
            ignored_outside_requested_view_count=outside_count,
            layout_mode=LayoutMode.EMPTY.name,
        )
        return geometry, projected, stats

    required = {str(value) for value in bound_track_keys}
    visible_area = temporary_projected.patch_weight.sum(dim=(2, 3))[0]
    eligible_indices.sort(
        key=lambda slot: (
            str(candidates[slot]["raw_track_key"]) in required,
            float(visible_area[slot].item()),
        ),
        reverse=True,
    )
    pre_cap_count = len(eligible_indices)
    selected_indices = eligible_indices[:cap]
    overflow = max(0, pre_cap_count - cap)
    if required:
        selected_keys = {str(candidates[slot]["raw_track_key"]) for slot in selected_indices}
        missing_required = sorted(required.difference(selected_keys))
        if missing_required:
            raise ValueError(
                "appearance-bound actor is not eligible for requested G: "
                f"{missing_required[:5]}"
            )
    mode = LayoutMode.PARTIAL if overflow else LayoutMode.FULL
    selected_candidates = [candidates[slot] for slot in selected_indices]
    selected = _candidate_geometry(
        selected_candidates,
        num_frames=len(indices),
        world_to_anchor=camera.world_to_anchor,
    )
    padded = _empty_geometry(
        num_frames=len(indices), layout_max_actors=cap, mode=mode,
        device=camera.world_to_anchor.device,
    )
    count = len(selected_candidates)
    geometry = ActorGeometryCondition(
        slot_track_id=padded.slot_track_id.clone(),
        class_id=padded.class_id.clone(),
        corners_world=padded.corners_world.clone(),
        velocity_world=padded.velocity_world.clone(),
        box_size=padded.box_size.clone(),
        yaw=padded.yaw.clone(),
        is_moving=padded.is_moving.clone(),
        track_valid=padded.track_valid.clone(),
        slot_valid=padded.slot_valid.clone(),
        layout_mode=torch.full_like(padded.layout_mode, int(mode)),
        raw_track_key=[[""] * cap],
    )
    # Frozen dataclasses own mutable tensors; populate clones before rebuilding
    # the validated final value.
    tensors = {
        "slot_track_id": geometry.slot_track_id.clone(),
        "class_id": geometry.class_id.clone(),
        "corners_world": geometry.corners_world.clone(),
        "velocity_world": geometry.velocity_world.clone(),
        "box_size": geometry.box_size.clone(),
        "yaw": geometry.yaw.clone(),
        "is_moving": geometry.is_moving.clone(),
        "track_valid": geometry.track_valid.clone(),
        "slot_valid": geometry.slot_valid.clone(),
    }
    for name in tensors:
        tensors[name][:, :count] = getattr(selected, name)
    raw_keys = [str(item["raw_track_key"]) for item in selected_candidates] + [""] * (cap - count)
    geometry = ActorGeometryCondition(
        **tensors,
        layout_mode=torch.full((1,), int(mode), dtype=torch.int8, device=camera.world_to_anchor.device),
        raw_track_key=[raw_keys],
    )
    projected = project_actor_geometry(geometry, camera)
    stats = ActorBuildStats(
        annotated_count=annotated_count,
        supported_count=supported_count,
        eligible_count=pre_cap_count,
        pre_cap_count=pre_cap_count,
        post_cap_count=count,
        overflow=overflow,
        ignored_unsupported_count=unsupported_count,
        ignored_invalid_geometry_count=invalid_count,
        ignored_outside_requested_view_count=outside_count,
        layout_mode=mode.name,
    )
    return geometry, projected, stats


def sample_appearance_geometry_indices(
    geometry: ActorGeometryCondition,
    projected: ProjectedActorGeometry,
    reference_available: torch.Tensor,
    *,
    max_bindings: int = 5,
    rng: random.Random | None = None,
) -> tuple[list[int], AppearanceSamplingStats]:
    """Randomly sample A from G while balancing all four required axes.

    The returned integers are G-slot indices.  Reference pixels/tokens are
    intentionally outside this function; resolving track keys into G happens
    before any tensor ``geometry_idx`` is constructed.
    """

    geometry.validate()
    projected.validate()
    if geometry.batch_size != 1:
        raise ValueError("appearance sampling accepts one scene at a time")
    if tuple(reference_available.shape) != (geometry.num_slots,) or reference_available.dtype != torch.bool:
        raise TypeError("reference_available must be bool [Kg]")
    limit = min(5, max(0, int(max_bindings)))
    generator = rng if rng is not None else random
    candidates = (
        geometry.slot_valid[0]
        & reference_available.to(device=geometry.slot_valid.device)
        & projected.frame_support[0].any(dim=-1)
    ).nonzero(as_tuple=False).flatten().tolist()
    target_count = generator.randint(0, min(limit, len(candidates))) if candidates else 0

    area = projected.patch_weight[0].sum(dim=(1, 2)).detach().cpu()
    visible = projected.frame_support[0]
    depth_values: dict[int, float] = {}
    for slot in candidates:
        valid_depth = projected.log_z_w[0, slot][visible[slot]]
        depth_values[slot] = (
            float(torch.exp(valid_depth).median().item()) if valid_depth.numel() else float("inf")
        )
    finite_depths = sorted(value for value in depth_values.values() if math.isfinite(value))
    near_cut = finite_depths[len(finite_depths) // 3] if finite_depths else 0.0
    far_cut = finite_depths[(2 * len(finite_depths)) // 3] if finite_depths else 0.0
    area_values = sorted(float(area[slot].item()) for slot in candidates)
    area_cut = area_values[len(area_values) // 2] if area_values else 0.0

    def buckets(slot: int) -> tuple[str, str, str, str]:
        class_name = ACTOR_CLASS_NAMES[int(geometry.class_id[0, slot].item())]
        distance = depth_values[slot]
        distance_bucket = "near" if distance <= near_cut else "mid" if distance <= far_cut else "far"
        area_bucket = "large" if float(area[slot].item()) >= area_cut else "small"
        motion_bucket = "moving" if bool(geometry.is_moving[0, slot].any()) else "stationary"
        return class_name, distance_bucket, area_bucket, motion_bucket

    selected: list[int] = []
    remaining = list(candidates)
    counts: list[dict[str, int]] = [{}, {}, {}, {}]
    for _ in range(target_count):
        weighted: list[tuple[int, float]] = []
        for slot in remaining:
            slot_buckets = buckets(slot)
            # Inverse-frequency across all four axes, with a random tie breaker.
            balance = sum(1.0 / (1.0 + counts[axis].get(value, 0)) for axis, value in enumerate(slot_buckets))
            weighted.append((slot, balance * (0.75 + 0.5 * generator.random())))
        chosen = max(weighted, key=lambda item: item[1])[0]
        selected.append(chosen)
        remaining.remove(chosen)
        for axis, value in enumerate(buckets(chosen)):
            counts[axis][value] = counts[axis].get(value, 0) + 1

    stats = AppearanceSamplingStats(
        selected_count=len(selected),
        candidate_count=len(candidates),
        class_buckets=counts[0],
        distance_buckets=counts[1],
        area_buckets=counts[2],
        motion_buckets=counts[3],
    )
    return selected, stats
