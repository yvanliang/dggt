"""Pure data contracts for full-scene actor geometry.

``ActorGeometryCondition`` is the only owner of actor position.  Appearance
conditioning may refer to a geometry slot, but it must never carry a second
copy of a box, trajectory, or spatial address.

The projection path below is the single owner of every camera-space actor
cache.  Actor boxes and the requested camera are both derived from the same
unadjusted Waymo ``Frame.pose`` coordinate system, so ``map_pose_offset`` is
deliberately *not* part of this path.  Coordinates remain float64 through the
anchor-origin subtraction, requested-camera transform, and real
intrinsics/crop transform; only the derived cache is cast to float32.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import math
from typing import Sequence

import torch


class LayoutMode(IntEnum):
    """Completeness state of the actor-layout condition."""

    NULL = 0
    EMPTY = 1
    PARTIAL = 2
    FULL = 3


_BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)

ACTOR_FAR_PLANE_M = 120.0
ACTOR_NEAR_PLANE_M = 0.5
MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES = 32


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


def _normalize_frame_index(
    frame_index: slice | Sequence[int] | torch.Tensor,
) -> slice | Sequence[int] | torch.Tensor:
    if isinstance(frame_index, torch.Tensor):
        if frame_index.ndim != 1:
            raise ValueError("frame_index tensor must be one-dimensional")
        if frame_index.dtype not in _INTEGER_DTYPES and frame_index.dtype != torch.bool:
            raise TypeError("frame_index tensor must have integer or bool dtype")
    return frame_index


def raw_to_model_canvas_homography(
    raw_hw: Sequence[int] | torch.Tensor,
    *,
    target_width: int = 518,
    patch_size: int = 14,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reproduce ``load_and_preprocess_images`` crop geometry exactly.

    The raw image is resized to ``target_width``; its aspect-preserving height
    is rounded to a multiple of ``patch_size`` with Python's built-in ``round``
    (the production preprocessing rule), then center-cropped vertically when
    it exceeds ``target_width``.  The returned float64 homography maps raw
    pixel coordinates directly into that final canvas.
    """

    if isinstance(raw_hw, torch.Tensor):
        if raw_hw.numel() != 2:
            raise ValueError("raw_hw must contain exactly (height,width)")
        raw_values = raw_hw.detach().cpu().reshape(-1).tolist()
    else:
        raw_values = list(raw_hw)
    if len(raw_values) != 2:
        raise ValueError("raw_hw must contain exactly (height,width)")
    raw_h, raw_w = (int(v) for v in raw_values)
    target_width = int(target_width)
    patch_size = int(patch_size)
    if min(raw_h, raw_w, target_width, patch_size) <= 0:
        raise ValueError("raw size, target_width, and patch_size must be positive")
    new_height = round(
        raw_h * (float(target_width) / float(raw_w)) / float(patch_size)
    ) * patch_size
    if new_height <= 0:
        raise ValueError("raw aspect ratio rounds to a zero-height model canvas")
    crop_top = (
        (new_height - target_width) // 2 if new_height > target_width else 0
    )
    canvas_h = target_width if new_height > target_width else new_height
    homography = torch.tensor(
        (
            (float(target_width) / float(raw_w), 0.0, 0.0),
            (0.0, float(new_height) / float(raw_h), -float(crop_top)),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    return homography, (int(canvas_h), int(target_width))


@dataclass(frozen=True)
class ActorGeometryCondition:
    """Fixed-padded geometry for every eligible actor in a target window.

    The actor axis is always ``Kg == layout_max_actors``.  ``slot_valid`` is the
    padding mask; ``track_valid`` is per frame and never depends on appearance.
    World-space corners remain float64 until an external projection function
    subtracts the anchor origin.
    """

    slot_track_id: torch.Tensor
    class_id: torch.Tensor
    corners_world: torch.Tensor
    velocity_world: torch.Tensor
    box_size: torch.Tensor
    yaw: torch.Tensor
    is_moving: torch.Tensor
    track_valid: torch.Tensor
    slot_valid: torch.Tensor
    layout_mode: torch.Tensor
    raw_track_key: list[list[str]]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.slot_track_id.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.slot_track_id.shape[1])

    @property
    def num_frames(self) -> int:
        return int(self.track_valid.shape[2])

    def validate(
        self,
        *,
        layout_max_actors: int | None = None,
    ) -> "ActorGeometryCondition":
        if self.slot_track_id.ndim != 2:
            raise ValueError(
                "slot_track_id must be [B,Kg], got "
                f"{tuple(self.slot_track_id.shape)}"
            )
        b, kg = (int(v) for v in self.slot_track_id.shape)
        if kg <= 0:
            raise ValueError("the fixed-padded actor axis Kg must be positive")
        if layout_max_actors is not None and kg != int(layout_max_actors):
            raise ValueError(
                f"Kg={kg} must equal layout_max_actors={int(layout_max_actors)}; "
                "batch-dependent padding is forbidden"
            )
        if self.corners_world.ndim != 5:
            raise ValueError(
                "corners_world must be [B,Kg,S,8,3], got "
                f"{tuple(self.corners_world.shape)}"
            )
        s = int(self.corners_world.shape[2])
        _require_shape("class_id", self.class_id, (b, kg))
        _require_shape("corners_world", self.corners_world, (b, kg, s, 8, 3))
        _require_shape("velocity_world", self.velocity_world, (b, kg, s, 3))
        _require_shape("box_size", self.box_size, (b, kg, s, 3))
        _require_shape("yaw", self.yaw, (b, kg, s))
        _require_shape("is_moving", self.is_moving, (b, kg, s))
        _require_shape("track_valid", self.track_valid, (b, kg, s))
        _require_shape("slot_valid", self.slot_valid, (b, kg))
        _require_shape("layout_mode", self.layout_mode, (b,))

        expected_dtypes = {
            "slot_track_id": torch.int64,
            "class_id": torch.int8,
            "corners_world": torch.float64,
            "velocity_world": torch.float32,
            "box_size": torch.float32,
            "yaw": torch.float32,
            "is_moving": torch.bool,
            "track_valid": torch.bool,
            "slot_valid": torch.bool,
            "layout_mode": torch.int8,
        }
        for name, expected_dtype in expected_dtypes.items():
            value = getattr(self, name)
            if value.dtype != expected_dtype:
                raise TypeError(f"{name} must have {expected_dtype} dtype")

        tensor_values = (
            self.slot_track_id,
            self.class_id,
            self.corners_world,
            self.velocity_world,
            self.box_size,
            self.yaw,
            self.is_moving,
            self.track_valid,
            self.slot_valid,
            self.layout_mode,
        )
        devices = {value.device for value in tensor_values}
        if len(devices) != 1:
            raise ValueError("all ActorGeometryCondition tensors must share one device")

        if bool(((self.layout_mode < int(LayoutMode.NULL)) | (
            self.layout_mode > int(LayoutMode.FULL)
        )).any()):
            raise ValueError("layout_mode contains an unknown value")
        if bool((self.track_valid & ~self.slot_valid[..., None]).any()):
            raise ValueError("track_valid cannot be true for a padded G slot")
        absent = (self.layout_mode == int(LayoutMode.NULL)) | (
            self.layout_mode == int(LayoutMode.EMPTY)
        )
        if bool((self.slot_valid & absent[:, None]).any()):
            raise ValueError("G NULL/EMPTY rows cannot contain a valid actor slot")

        for name in ("corners_world", "velocity_world", "box_size", "yaw"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if bool((self.box_size[self.track_valid] <= 0.0).any()):
            raise ValueError("box_size must be positive at every track-valid frame")
        if bool((self.slot_track_id[self.slot_valid] < 0).any()):
            raise ValueError("valid G slots require non-negative scene-local slot_track_id")
        if bool((self.class_id[self.slot_valid] < 0).any()):
            raise ValueError("valid G slots require a non-negative class_id")

        if len(self.raw_track_key) != b:
            raise ValueError(
                f"raw_track_key batch length {len(self.raw_track_key)} != {b}"
            )
        for batch_index, keys in enumerate(self.raw_track_key):
            if not isinstance(keys, list) or len(keys) != kg:
                raise ValueError(
                    "raw_track_key must be a fixed-padded list[list[str]] with "
                    f"row length Kg={kg}; row {batch_index} is invalid"
                )
            if any(not isinstance(key, str) for key in keys):
                raise TypeError("every raw_track_key entry must be a string")

        # Scene-local ids and raw keys are identities, so duplicates among
        # live slots would make A -> G binding ambiguous.
        for batch_index in range(b):
            valid = self.slot_valid[batch_index]
            ids = self.slot_track_id[batch_index, valid]
            if int(torch.unique(ids).numel()) != int(ids.numel()):
                raise ValueError(
                    f"slot_track_id must be unique among valid slots in batch {batch_index}"
                )
            live_keys = [
                self.raw_track_key[batch_index][slot]
                for slot in valid.nonzero(as_tuple=False).flatten().cpu().tolist()
            ]
            if any(not key for key in live_keys):
                raise ValueError("valid G slots require a non-empty raw_track_key")
            if len(set(live_keys)) != len(live_keys):
                raise ValueError(
                    f"raw_track_key must be unique among valid slots in batch {batch_index}"
                )
        return self

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "ActorGeometryCondition":
        """Move tensors without changing their contract dtypes."""

        tensor_names = (
            "slot_track_id",
            "class_id",
            "corners_world",
            "velocity_world",
            "box_size",
            "yaw",
            "is_moving",
            "track_valid",
            "slot_valid",
            "layout_mode",
        )
        updates = {
            name: getattr(self, name).to(device=device, non_blocking=non_blocking)
            for name in tensor_names
        }
        updates["raw_track_key"] = [list(row) for row in self.raw_track_key]
        return replace(self, **updates)

    def slice_frames(
        self,
        frame_index: slice | Sequence[int] | torch.Tensor,
    ) -> "ActorGeometryCondition":
        """Slice the frame axis while preserving the fixed actor-slot axis."""

        frame_index = _normalize_frame_index(frame_index)
        return replace(
            self,
            corners_world=self.corners_world[:, :, frame_index],
            velocity_world=self.velocity_world[:, :, frame_index],
            box_size=self.box_size[:, :, frame_index],
            yaw=self.yaw[:, :, frame_index],
            is_moving=self.is_moving[:, :, frame_index],
            track_valid=self.track_valid[:, :, frame_index],
            raw_track_key=[list(row) for row in self.raw_track_key],
        )

    def _absent_like(self, mode: LayoutMode) -> "ActorGeometryCondition":
        if mode not in (LayoutMode.NULL, LayoutMode.EMPTY):
            raise ValueError("absent G helper only accepts NULL or EMPTY")
        return replace(
            self,
            slot_track_id=torch.full_like(self.slot_track_id, -1),
            class_id=torch.full_like(self.class_id, -1),
            corners_world=torch.zeros_like(self.corners_world),
            velocity_world=torch.zeros_like(self.velocity_world),
            box_size=torch.zeros_like(self.box_size),
            yaw=torch.zeros_like(self.yaw),
            is_moving=torch.zeros_like(self.is_moving),
            track_valid=torch.zeros_like(self.track_valid),
            slot_valid=torch.zeros_like(self.slot_valid),
            layout_mode=torch.full_like(self.layout_mode, int(mode)),
            raw_track_key=[[
                "" for _ in range(self.num_slots)
            ] for _ in range(self.batch_size)],
        )

    def null_like(self) -> "ActorGeometryCondition":
        """Return a leak-free CFG-negative G condition."""

        return self._absent_like(LayoutMode.NULL)

    def empty_like(self) -> "ActorGeometryCondition":
        """Return a factual EMPTY G condition, distinct from NULL."""

        return self._absent_like(LayoutMode.EMPTY)


@dataclass(frozen=True)
class CameraSpec:
    """Requested metric camera used by both map and actor projection.

    ``intrinsics`` is the real Waymo pixel-space matrix and
    ``raw_to_canvas`` is the explicit crop/resize homography.  Their product,
    rather than a predicted DGGT intrinsic, is the only matrix used for image
    placement.  ``map_pose_offset`` is one ``[B,3]`` value used only when
    projecting static map coordinates.  Actor geometry and requested cameras
    share the same raw ``Frame.pose`` world and must never consume it.  Callers
    freeze the value from the first frame of the requested window, as required
    by the v2.1 map contract.
    """

    world_to_anchor: torch.Tensor
    anchor_to_camera: torch.Tensor
    intrinsics: torch.Tensor
    raw_to_canvas: torch.Tensor
    map_pose_offset: torch.Tensor
    canvas_hw: tuple[int, int]
    patch_grid: tuple[int, int] = (25, 37)
    near_plane_m: float = 0.5

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.world_to_anchor.shape[0])

    @property
    def num_frames(self) -> int:
        return int(self.anchor_to_camera.shape[1])

    @property
    def canvas_intrinsics(self) -> torch.Tensor:
        """Return ``raw_to_canvas @ K_W`` without changing precision."""

        return torch.matmul(self.raw_to_canvas, self.intrinsics)

    @property
    def normalized_canvas_intrinsics(self) -> torch.Tensor:
        """Return real ``K_W`` normalized to the requested model canvas."""

        canvas_h, canvas_w = self.canvas_hw
        scale = self.intrinsics.new_zeros(
            (self.batch_size, self.num_frames, 3, 3)
        )
        scale[..., 0, 0] = 1.0 / float(canvas_w)
        scale[..., 1, 1] = 1.0 / float(canvas_h)
        scale[..., 2, 2] = 1.0
        return torch.matmul(scale, self.canvas_intrinsics)

    def validate(self) -> "CameraSpec":
        if self.world_to_anchor.ndim != 3 or tuple(
            self.world_to_anchor.shape[-2:]
        ) != (4, 4):
            raise ValueError(
                "world_to_anchor must be [B,4,4], got "
                f"{tuple(self.world_to_anchor.shape)}"
            )
        b = int(self.world_to_anchor.shape[0])
        if self.anchor_to_camera.ndim != 4 or tuple(
            self.anchor_to_camera.shape[-2:]
        ) != (4, 4):
            raise ValueError(
                "anchor_to_camera must be [B,S,4,4], got "
                f"{tuple(self.anchor_to_camera.shape)}"
            )
        s = int(self.anchor_to_camera.shape[1])
        _require_shape("anchor_to_camera", self.anchor_to_camera, (b, s, 4, 4))
        _require_shape("intrinsics", self.intrinsics, (b, s, 3, 3))
        _require_shape("raw_to_canvas", self.raw_to_canvas, (b, s, 3, 3))
        _require_shape("map_pose_offset", self.map_pose_offset, (b, 3))
        tensors = (
            self.world_to_anchor,
            self.anchor_to_camera,
            self.intrinsics,
            self.raw_to_canvas,
            self.map_pose_offset,
        )
        if any(value.dtype != torch.float64 for value in tensors):
            raise TypeError("all CameraSpec tensors must have float64 dtype")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all CameraSpec tensors must share one device")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("CameraSpec contains NaN or Inf")
        if s <= 0:
            raise ValueError("CameraSpec requires at least one frame")
        canvas_h, canvas_w = (int(v) for v in self.canvas_hw)
        patch_h, patch_w = (int(v) for v in self.patch_grid)
        if min(canvas_h, canvas_w, patch_h, patch_w) <= 0:
            raise ValueError("canvas_hw and patch_grid must be positive")
        if not (float(self.near_plane_m) > 0.0):
            raise ValueError("near_plane_m must be positive")
        return self

    @classmethod
    def from_window(
        cls,
        *,
        world_to_anchor: torch.Tensor,
        anchor_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        raw_to_canvas: torch.Tensor,
        map_pose_offsets: torch.Tensor,
        canvas_hw: tuple[int, int],
        patch_grid: tuple[int, int] = (25, 37),
        near_plane_m: float = 0.5,
    ) -> "CameraSpec":
        """Build a camera while freezing the map offset to the first frame.

        ``map_pose_offsets`` may be ``[B,S,3]`` or already-frozen ``[B,3]``.
        Selecting ``[:, 0]`` here gives every static-map reader one explicit
        interpretation instead of allowing readers to choose different frames.
        Actor projection intentionally ignores this field.
        """

        offsets = torch.as_tensor(map_pose_offsets)
        if offsets.ndim == 3:
            offsets = offsets[:, 0]
        elif offsets.ndim != 2:
            raise ValueError("map_pose_offsets must be [B,S,3] or [B,3]")
        return cls(
            world_to_anchor=world_to_anchor,
            anchor_to_camera=anchor_to_camera,
            intrinsics=intrinsics,
            raw_to_canvas=raw_to_canvas,
            map_pose_offset=offsets,
            canvas_hw=canvas_hw,
            patch_grid=patch_grid,
            near_plane_m=near_plane_m,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "CameraSpec":
        return replace(
            self,
            world_to_anchor=self.world_to_anchor.to(
                device=device, non_blocking=non_blocking
            ),
            anchor_to_camera=self.anchor_to_camera.to(
                device=device, non_blocking=non_blocking
            ),
            intrinsics=self.intrinsics.to(device=device, non_blocking=non_blocking),
            raw_to_canvas=self.raw_to_canvas.to(
                device=device, non_blocking=non_blocking
            ),
            map_pose_offset=self.map_pose_offset.to(
                device=device, non_blocking=non_blocking
            ),
        )


@dataclass(frozen=True)
class ProjectedActorGeometry:
    """Read-only derived cache produced from ``(G_world, requested C)``.

    Every downstream reader must consume this single object rather than
    independently re-projecting geometry.
    """

    bbox_patch: torch.Tensor
    patch_weight: torch.Tensor
    log_z_patch: torch.Tensor
    silhouette_uv: torch.Tensor
    silhouette_vertex_valid: torch.Tensor
    corners_camera: torch.Tensor
    uv_corners: torch.Tensor
    velocity_camera: torch.Tensor
    uv_center: torch.Tensor
    log_z_w: torch.Tensor
    center_depth_valid: torch.Tensor
    frame_support: torch.Tensor
    metric_support: torch.Tensor
    in_frustum: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.valid.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.valid.shape[1])

    @property
    def num_frames(self) -> int:
        return int(self.valid.shape[2])

    @property
    def track_valid(self) -> torch.Tensor:
        """Compatibility-free semantic alias used by the gather contract."""

        return self.valid

    @property
    def patch_support(self) -> torch.Tensor:
        """Explicit mask for valid per-patch surface depth values."""

        return self.frame_support[..., None] & self.patch_weight.gt(0.0)

    def validate(self) -> "ProjectedActorGeometry":
        if self.valid.ndim != 3:
            raise ValueError(
                f"valid must be [B,Kg,S], got {tuple(self.valid.shape)}"
            )
        b, kg, s = (int(v) for v in self.valid.shape)
        if kg <= 0:
            raise ValueError("ProjectedActorGeometry requires a positive Kg axis")
        if self.patch_weight.ndim != 4 or tuple(self.patch_weight.shape[:3]) != (
            b,
            kg,
            s,
        ):
            raise ValueError(
                "patch_weight must be [B,Kg,S,P], got "
                f"{tuple(self.patch_weight.shape)}"
            )
        p = int(self.patch_weight.shape[-1])
        if p <= 0:
            raise ValueError("ProjectedActorGeometry requires a positive patch axis")
        _require_shape("bbox_patch", self.bbox_patch, (b, kg, s, 4))
        _require_shape("log_z_patch", self.log_z_patch, (b, kg, s, p))
        _require_shape(
            "silhouette_uv",
            self.silhouette_uv,
            (b, kg, s, MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES, 2),
        )
        _require_shape(
            "silhouette_vertex_valid",
            self.silhouette_vertex_valid,
            (b, kg, s, MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES),
        )
        _require_shape("corners_camera", self.corners_camera, (b, kg, s, 8, 3))
        _require_shape("uv_corners", self.uv_corners, (b, kg, s, 8, 2))
        _require_shape("velocity_camera", self.velocity_camera, (b, kg, s, 3))
        _require_shape("uv_center", self.uv_center, (b, kg, s, 2))
        _require_shape("log_z_w", self.log_z_w, (b, kg, s))
        _require_shape("center_depth_valid", self.center_depth_valid, (b, kg, s))
        _require_shape("frame_support", self.frame_support, (b, kg, s))
        _require_shape("metric_support", self.metric_support, (b, kg, s))
        _require_shape("in_frustum", self.in_frustum, (b, kg, s))
        bool_names = (
            "silhouette_vertex_valid",
            "center_depth_valid",
            "frame_support",
            "metric_support",
            "in_frustum",
            "valid",
        )
        for name in bool_names:
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
        float_names = (
            "bbox_patch",
            "patch_weight",
            "log_z_patch",
            "silhouette_uv",
            "corners_camera",
            "uv_corners",
            "velocity_camera",
            "uv_center",
            "log_z_w",
        )
        for name in float_names:
            value = getattr(self, name)
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must have float32 dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        devices = {
            getattr(self, name).device for name in (*float_names, *bool_names)
        }
        if len(devices) != 1:
            raise ValueError("all ProjectedActorGeometry tensors must share one device")
        if not torch.equal(self.in_frustum, self.frame_support):
            raise ValueError("in_frustum is a compatibility alias and must equal frame_support")
        if bool((self.in_frustum & ~self.valid).any()):
            raise ValueError("in_frustum cannot be true where projected geometry is invalid")
        if bool((self.center_depth_valid & ~self.valid).any()):
            raise ValueError("center_depth_valid cannot be true where geometry is invalid")
        if bool((self.frame_support & ~self.center_depth_valid).any()):
            raise ValueError("frame_support requires a valid in-range actor centre")
        if bool((self.metric_support & ~self.frame_support).any()):
            raise ValueError("metric_support must be a subset of frame_support")
        corner_z = self.corners_camera[..., 2]
        expected_metric_support = (
            self.frame_support
            & corner_z.ge(float(ACTOR_NEAR_PLANE_M)).all(dim=-1)
            & corner_z.le(float(ACTOR_FAR_PLANE_M)).all(dim=-1)
        )
        if not torch.equal(self.metric_support, expected_metric_support):
            raise ValueError(
                "metric_support must equal frame_support with all eight original "
                "corner depths in [0.5,120] m"
            )
        if bool((self.patch_weight < 0.0).any()) or bool(
            (self.patch_weight > 1.0 + 1.0e-5).any()
        ):
            raise ValueError("patch_weight must be fractional coverage in [0,1]")
        unsupported = ~self.frame_support[..., None]
        if bool(self.patch_weight.masked_select(unsupported).ne(0.0).any()):
            raise ValueError("patch_weight must be zero outside frame_support")
        if bool(self.log_z_patch.masked_select(~self.patch_support).ne(0.0).any()):
            raise ValueError("log_z_patch padding must be zero outside patch_support")
        if bool(self.log_z_w.masked_select(~self.center_depth_valid).ne(0.0).any()):
            raise ValueError("log_z_w padding must be zero outside center_depth_valid")
        if bool(
            self.uv_center.masked_select(
                (~self.center_depth_valid)[..., None].expand_as(self.uv_center)
            ).ne(0.0).any()
        ):
            raise ValueError("uv_center padding must be zero outside center_depth_valid")
        vertex_count = self.silhouette_vertex_valid.sum(dim=-1)
        if bool((self.frame_support & vertex_count.lt(3)).any()) or bool(
            ((~self.frame_support) & vertex_count.ne(0)).any()
        ):
            raise ValueError("silhouette must have >=3 vertices iff frame_support is true")
        padded_vertices = ~self.silhouette_vertex_valid[..., None]
        if bool(self.silhouette_uv.masked_select(padded_vertices).ne(0.0).any()):
            raise ValueError("silhouette_uv padded vertices must be exactly zero")
        active_vertices = self.silhouette_uv.masked_select(
            self.silhouette_vertex_valid[..., None].expand_as(self.silhouette_uv)
        )
        if active_vertices.numel() and bool(
            ((active_vertices < -1.0e-6) | (active_vertices > 1.0 + 1.0e-6)).any()
        ):
            raise ValueError("silhouette_uv active vertices must lie on the model canvas")
        covered_frames = self.patch_weight.sum(dim=-1).gt(0.0)
        if not torch.equal(covered_frames, self.frame_support):
            raise ValueError("frame_support must exactly match nonzero silhouette coverage")
        return self

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "ProjectedActorGeometry":
        updates = {
            name: getattr(self, name).to(device=device, non_blocking=non_blocking)
            for name in self.__dataclass_fields__
        }
        return replace(self, **updates)

    def slice_frames(
        self,
        frame_index: slice | Sequence[int] | torch.Tensor,
    ) -> "ProjectedActorGeometry":
        frame_index = _normalize_frame_index(frame_index)
        return replace(
            self,
            bbox_patch=self.bbox_patch[:, :, frame_index],
            patch_weight=self.patch_weight[:, :, frame_index],
            log_z_patch=self.log_z_patch[:, :, frame_index],
            silhouette_uv=self.silhouette_uv[:, :, frame_index],
            silhouette_vertex_valid=self.silhouette_vertex_valid[:, :, frame_index],
            corners_camera=self.corners_camera[:, :, frame_index],
            uv_corners=self.uv_corners[:, :, frame_index],
            velocity_camera=self.velocity_camera[:, :, frame_index],
            uv_center=self.uv_center[:, :, frame_index],
            log_z_w=self.log_z_w[:, :, frame_index],
            center_depth_valid=self.center_depth_valid[:, :, frame_index],
            frame_support=self.frame_support[:, :, frame_index],
            metric_support=self.metric_support[:, :, frame_index],
            in_frustum=self.in_frustum[:, :, frame_index],
            valid=self.valid[:, :, frame_index],
        )

    def null_like(self) -> "ProjectedActorGeometry":
        return replace(
            self,
            bbox_patch=torch.zeros_like(self.bbox_patch),
            patch_weight=torch.zeros_like(self.patch_weight),
            log_z_patch=torch.zeros_like(self.log_z_patch),
            silhouette_uv=torch.zeros_like(self.silhouette_uv),
            silhouette_vertex_valid=torch.zeros_like(self.silhouette_vertex_valid),
            corners_camera=torch.zeros_like(self.corners_camera),
            uv_corners=torch.zeros_like(self.uv_corners),
            velocity_camera=torch.zeros_like(self.velocity_camera),
            uv_center=torch.zeros_like(self.uv_center),
            log_z_w=torch.zeros_like(self.log_z_w),
            center_depth_valid=torch.zeros_like(self.center_depth_valid),
            frame_support=torch.zeros_like(self.frame_support),
            metric_support=torch.zeros_like(self.metric_support),
            in_frustum=torch.zeros_like(self.in_frustum),
            valid=torch.zeros_like(self.valid),
        )


def project_world_points(
    points_world: torch.Tensor,
    cam: CameraSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project static world points through the one v2.1 coordinate chain.

    Args:
        points_world: Float64 points shaped ``[B,N,3]``.  The same map points
            are viewed from every frame of ``cam``.
        cam: Requested camera and first-frame window offset.

    Returns:
        ``(points_camera, uv_canvas)`` shaped ``[B,S,N,3]`` and
        ``[B,S,N,2]``.  ``uv_canvas`` is normalized to ``[0,1]`` at the model
        canvas; it is intentionally not clamped at the image boundary.
    """

    if points_world.ndim != 3 or int(points_world.shape[-1]) != 3:
        raise ValueError("points_world must be [B,N,3]")
    if points_world.dtype != torch.float64:
        raise TypeError("points_world must remain float64 until anchor subtraction")
    if int(points_world.shape[0]) != cam.batch_size:
        raise ValueError("points_world and CameraSpec batch sizes differ")
    if points_world.device != cam.world_to_anchor.device:
        raise ValueError("points_world and CameraSpec must share one device")

    # The ordering is deliberate: subtract the window-first Waymo offset in
    # float64, then remove the large anchor origin, and only cast at the final
    # cache boundary.
    corrected = points_world - cam.map_pose_offset[:, None, :]
    ones = torch.ones_like(corrected[..., :1])
    corrected_h = torch.cat((corrected, ones), dim=-1)
    anchor_h = torch.einsum("bij,bnj->bni", cam.world_to_anchor, corrected_h)
    anchor_h = anchor_h[:, None].expand(-1, cam.num_frames, -1, -1)
    camera_h = torch.einsum("bsij,bsnj->bsni", cam.anchor_to_camera, anchor_h)
    camera = camera_h[..., :3]
    projected_h = torch.einsum("bsij,bsnj->bsni", cam.canvas_intrinsics, camera)
    z = projected_h[..., 2:3]
    finite_nonzero = torch.isfinite(projected_h).all(dim=-1, keepdim=True) & z.abs().gt(
        torch.finfo(torch.float64).eps
    )
    safe_z = torch.where(finite_nonzero, z, torch.ones_like(z))
    pixel = projected_h[..., :2] / safe_z
    canvas_h, canvas_w = cam.canvas_hw
    normalizer = pixel.new_tensor((float(canvas_w), float(canvas_h)))
    uv = torch.where(finite_nonzero.expand_as(pixel), pixel / normalizer, 0.0)
    return camera, uv


def project_actor_world_points(
    points_world: torch.Tensor,
    cam: CameraSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project actor points without applying the map-only pose offset.

    ``points_world`` is ``[B,N,3]`` in the same unadjusted Waymo segment world
    as the requested camera.  The returned camera points and normalized canvas
    coordinates are ``[B,S,N,3]`` and ``[B,S,N,2]``.  This public entry point is
    also the canonical projector for dataset-side appearance references.
    """

    if points_world.ndim != 3 or int(points_world.shape[-1]) != 3:
        raise ValueError("points_world must be [B,N,3]")
    if points_world.dtype != torch.float64:
        raise TypeError("actor points must remain float64 until anchor subtraction")
    if int(points_world.shape[0]) != cam.batch_size:
        raise ValueError("points_world and CameraSpec batch sizes differ")
    if points_world.device != cam.world_to_anchor.device:
        raise ValueError("points_world and CameraSpec must share one device")

    ones = torch.ones_like(points_world[..., :1])
    points_h = torch.cat((points_world, ones), dim=-1)
    anchor_h = torch.einsum("bij,bnj->bni", cam.world_to_anchor, points_h)
    anchor_h = anchor_h[:, None].expand(-1, cam.num_frames, -1, -1)
    camera_h = torch.einsum("bsij,bsnj->bsni", cam.anchor_to_camera, anchor_h)
    camera = camera_h[..., :3]
    projected_h = torch.einsum("bsij,bsnj->bsni", cam.canvas_intrinsics, camera)
    z = projected_h[..., 2:3]
    finite_nonzero = torch.isfinite(projected_h).all(dim=-1, keepdim=True) & z.abs().gt(
        torch.finfo(torch.float64).eps
    )
    safe_z = torch.where(finite_nonzero, z, torch.ones_like(z))
    pixel = projected_h[..., :2] / safe_z
    canvas_h, canvas_w = cam.canvas_hw
    normalizer = pixel.new_tensor((float(canvas_w), float(canvas_h)))
    uv = torch.where(finite_nonzero.expand_as(pixel), pixel / normalizer, 0.0)
    return camera, uv


def _project_actor_points(
    points_world: torch.Tensor,
    cam: CameraSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame-aligned no-offset variant for ``[B,K,S,Q,3]`` actor points."""

    ones = torch.ones_like(points_world[..., :1])
    points_h = torch.cat((points_world, ones), dim=-1)
    anchor_h = torch.einsum("bij,bksqj->bksqi", cam.world_to_anchor, points_h)
    camera_h = torch.einsum("bsij,bksqj->bksqi", cam.anchor_to_camera, anchor_h)
    camera = camera_h[..., :3]
    projected_h = torch.einsum(
        "bsij,bksqj->bksqi", cam.canvas_intrinsics, camera
    )
    z = projected_h[..., 2:3]
    finite_nonzero = torch.isfinite(projected_h).all(dim=-1, keepdim=True) & z.abs().gt(
        torch.finfo(torch.float64).eps
    )
    safe_z = torch.where(finite_nonzero, z, torch.ones_like(z))
    pixel = projected_h[..., :2] / safe_z
    canvas_h, canvas_w = cam.canvas_hw
    normalizer = pixel.new_tensor((float(canvas_w), float(canvas_h)))
    uv = torch.where(finite_nonzero.expand_as(pixel), pixel / normalizer, 0.0)
    return camera, uv


def _clipped_cuboid_vertices(
    corners_camera: torch.Tensor,
    *,
    near_plane_m: float,
    far_plane_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-padded vertices of a cuboid clipped to an optical-z slab."""

    z = corners_camera[..., 2]
    finite_corner = torch.isfinite(corners_camera).all(dim=-1)
    inside = (
        finite_corner
        & z.ge(float(near_plane_m))
        & z.le(float(far_plane_m))
    )
    edge_index_0 = torch.tensor(
        [edge[0] for edge in _BOX_EDGES], device=z.device, dtype=torch.int64
    )
    edge_index_1 = torch.tensor(
        [edge[1] for edge in _BOX_EDGES], device=z.device, dtype=torch.int64
    )
    edge_start = corners_camera.index_select(-2, edge_index_0)
    edge_end = corners_camera.index_select(-2, edge_index_1)
    start_z = edge_start[..., 2]
    end_z = edge_end[..., 2]
    finite_edge = torch.isfinite(edge_start).all(dim=-1) & torch.isfinite(edge_end).all(
        dim=-1
    )

    intersections: list[torch.Tensor] = []
    intersection_valid: list[torch.Tensor] = []
    for depth in (float(near_plane_m), float(far_plane_m)):
        crossing = finite_edge & (
            (start_z.lt(depth) & end_z.gt(depth))
            | (end_z.lt(depth) & start_z.gt(depth))
        )
        denominator = end_z - start_z
        safe_denominator = torch.where(
            crossing, denominator, torch.ones_like(denominator)
        )
        t = ((depth - start_z) / safe_denominator).clamp(0.0, 1.0)
        point = edge_start + t[..., None] * (edge_end - edge_start)
        intersections.append(point)
        intersection_valid.append(crossing & torch.isfinite(point).all(dim=-1))

    candidates = torch.cat((corners_camera, *intersections), dim=-2)
    candidate_valid = torch.cat((inside, *intersection_valid), dim=-1)
    if int(candidates.shape[-2]) != MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES:
        raise AssertionError("clipped cuboid vertex padding contract changed")
    return candidates, candidate_valid


def _cross_2d(
    origin: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (
        left[1] - origin[1]
    ) * (right[0] - origin[0])


def _convex_hull_2d(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Andrew monotone-chain hull in deterministic counter-clockwise order."""

    ordered = sorted(set((float(x), float(y)) for x, y in points))
    if len(ordered) <= 1:
        return ordered
    epsilon = 1.0e-12

    def build(values: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        side: list[tuple[float, float]] = []
        for point in values:
            while len(side) >= 2 and _cross_2d(side[-2], side[-1], point) <= epsilon:
                side.pop()
            side.append(point)
        return side

    lower = build(ordered)
    upper = build(tuple(reversed(ordered)))
    return lower[:-1] + upper[:-1]


def _clip_polygon_boundary(
    polygon: Sequence[tuple[float, float]],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    result: list[tuple[float, float]] = []
    epsilon = 1.0e-12

    def inside(point: tuple[float, float]) -> bool:
        value = point[axis]
        return (
            value >= boundary - epsilon
            if keep_greater
            else value <= boundary + epsilon
        )

    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(denominator) > epsilon:
                t = (boundary - previous[axis]) / denominator
                intersection = (
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                )
                result.append(intersection)
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _clip_polygon_to_rectangle(
    polygon: Sequence[tuple[float, float]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    clipped = list(polygon)
    clipped = _clip_polygon_boundary(
        clipped, axis=0, boundary=float(x_min), keep_greater=True
    )
    clipped = _clip_polygon_boundary(
        clipped, axis=0, boundary=float(x_max), keep_greater=False
    )
    clipped = _clip_polygon_boundary(
        clipped, axis=1, boundary=float(y_min), keep_greater=True
    )
    clipped = _clip_polygon_boundary(
        clipped, axis=1, boundary=float(y_max), keep_greater=False
    )
    # Boundary classification intentionally has a tiny tolerance so a
    # mathematically on-edge vertex is not lost to round-off.  Do not leak
    # that tolerance into downstream spatial addresses, though: values such
    # as -2e-16 are enough to trip the fail-closed RoPE range check after an
    # appearance token gathers its position from this silhouette bbox.
    # Snapping here keeps the canonical projected geometry in the exact
    # closed rectangle while preserving strict rejection of real excursions.
    return [
        (
            min(max(float(x), float(x_min)), float(x_max)),
            min(max(float(y), float(y_min)), float(y_max)),
        )
        for x, y in clipped
    ]


def _polygon_area_and_centroid(
    polygon: Sequence[tuple[float, float]],
) -> tuple[float, tuple[float, float]]:
    if len(polygon) < 3:
        return 0.0, (0.0, 0.0)
    # Translate before the shoelace sum.  Patch/silhouette intersections can
    # be tiny slivers next to coordinates O(10); evaluating the unshifted
    # formula loses enough significant bits to put the computed centroid just
    # outside the polygon (and therefore just outside the cuboid ray slab).
    origin_x, origin_y = polygon[0]
    twice_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        current_x = current[0] - origin_x
        current_y = current[1] - origin_y
        following_x = following[0] - origin_x
        following_y = following[1] - origin_y
        cross = current_x * following_y - following_x * current_y
        twice_area += cross
        centroid_x += (current_x + following_x) * cross
        centroid_y += (current_y + following_y) * cross
    if abs(twice_area) <= 1.0e-14:
        return 0.0, (0.0, 0.0)
    centroid = (
        origin_x + centroid_x / (3.0 * twice_area),
        origin_y + centroid_y / (3.0 * twice_area),
    )
    return 0.5 * abs(twice_area), centroid


def _matrix_vector_3x3(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _cuboid_ray_entry_depth(
    *,
    patch_point: tuple[float, float],
    patch_grid: tuple[int, int],
    canvas_hw: tuple[int, int],
    inverse_canvas_intrinsic: Sequence[Sequence[float]],
    inverse_half_axes: Sequence[Sequence[float]],
    local_camera_origin: Sequence[float],
    near_plane_m: float,
    far_plane_m: float,
) -> float | None:
    """Nearest cuboid surface z along the representative ray of one patch."""

    grid_h, grid_w = patch_grid
    canvas_h, canvas_w = canvas_hw
    pixel = (
        float(patch_point[0]) * float(canvas_w) / float(grid_w),
        float(patch_point[1]) * float(canvas_h) / float(grid_h),
        1.0,
    )
    ray = _matrix_vector_3x3(inverse_canvas_intrinsic, pixel)
    if not all(math.isfinite(value) for value in ray):
        return None
    if abs(ray[2]) <= 1.0e-12:
        return None
    direction = tuple(value / ray[2] for value in ray)
    local_direction = _matrix_vector_3x3(inverse_half_axes, direction)

    entry = float(near_plane_m)
    exit_ = float(far_plane_m)
    tolerance = 1.0e-9
    for origin_value, direction_value in zip(local_camera_origin, local_direction):
        if abs(direction_value) <= 1.0e-12:
            if origin_value < -1.0 - tolerance or origin_value > 1.0 + tolerance:
                return None
            continue
        first = (-1.0 - origin_value) / direction_value
        second = (1.0 - origin_value) / direction_value
        entry = max(entry, min(first, second))
        exit_ = min(exit_, max(first, second))
        if entry > exit_ + tolerance:
            return None
    if entry < float(near_plane_m) - tolerance or entry > float(far_plane_m) + tolerance:
        return None
    return min(max(entry, float(near_plane_m)), float(far_plane_m))


def _cuboid_patch_surface_depth(
    *,
    overlap_polygon: Sequence[tuple[float, float]],
    overlap_centroid: tuple[float, float],
    patch_grid: tuple[int, int],
    canvas_hw: tuple[int, int],
    inverse_canvas_intrinsic: Sequence[Sequence[float]],
    inverse_half_axes: Sequence[Sequence[float]],
    local_camera_origin: Sequence[float],
    near_plane_m: float,
    far_plane_m: float,
) -> float | None:
    """Return a robust local ray/cuboid entry depth for one covered patch.

    The stable area centroid and the arithmetic vertex mean are both interior
    representatives of the convex overlap polygon.  If round-off still puts
    both rays outside the analytic slab, retry vertices pulled strictly toward
    the interior.  Only finite intersections participate, and the nearest is
    retained.  A genuinely unsolved patch returns ``None`` so its coverage can
    be removed without discarding other valid patches or the whole actor.
    """

    if len(overlap_polygon) < 3:
        return None
    vertex_mean = (
        sum(point[0] for point in overlap_polygon) / float(len(overlap_polygon)),
        sum(point[1] for point in overlap_polygon) / float(len(overlap_polygon)),
    )
    primary_candidates = (overlap_centroid, vertex_mean)

    def depths_for(
        candidates: Sequence[tuple[float, float]],
    ) -> list[float]:
        depths: list[float] = []
        seen: set[tuple[float, float]] = set()
        for point in candidates:
            point = (float(point[0]), float(point[1]))
            if point in seen:
                continue
            seen.add(point)
            depth = _cuboid_ray_entry_depth(
                patch_point=point,
                patch_grid=patch_grid,
                canvas_hw=canvas_hw,
                inverse_canvas_intrinsic=inverse_canvas_intrinsic,
                inverse_half_axes=inverse_half_axes,
                local_camera_origin=local_camera_origin,
                near_plane_m=near_plane_m,
                far_plane_m=far_plane_m,
            )
            if depth is not None and math.isfinite(depth) and depth > 0.0:
                depths.append(float(depth))
        return depths

    depths = depths_for(primary_candidates)
    if depths:
        return min(depths)

    # Boundary vertices are valid silhouette locations but can sit exactly on
    # two slab planes.  Pull them toward a guaranteed convex interior point so
    # the fallback never depends on an arbitrary global centre or fake depth.
    inset_fraction = 1.0e-3
    fallback_candidates = [
        (
            (1.0 - inset_fraction) * point[0]
            + inset_fraction * vertex_mean[0],
            (1.0 - inset_fraction) * point[1]
            + inset_fraction * vertex_mean[1],
        )
        for point in overlap_polygon
    ]
    depths = depths_for(fallback_candidates)
    return min(depths) if depths else None


def project_actor_cuboid_corners(
    corners_world: torch.Tensor,
    cam: CameraSpec,
    *,
    track_valid: torch.Tensor | None = None,
    far_plane_m: float = ACTOR_FAR_PLANE_M,
) -> dict[str, torch.Tensor]:
    """Project cuboids through the canonical no-offset actor coordinate chain.

    Args:
        corners_world: ``[B,K,S,8,3]`` float64 corners in raw Waymo segment
            world coordinates.
        cam: Requested camera in that same raw world.
        track_valid: Optional ``[B,K,S]`` mask.  Omitted means every cuboid is
            geometrically present.
        far_plane_m: Inclusive optical-z far plane; frozen to 120 m in the
            layout-v2 production contract.

    The fixed-padded CCW ``silhouette_uv`` is the sole 2-D outline source for
    early rasterization.  ``patch_weight`` is exact polygon/cell overlap area.
    ``log_z_patch`` is the nearest valid local ray/cuboid entry depth for each
    non-empty patch/silhouette intersection; its zero padding is meaningful only
    together with ``frame_support & (patch_weight > 0)``.
    """

    cam.validate()
    if corners_world.ndim != 5 or tuple(corners_world.shape[-2:]) != (8, 3):
        raise ValueError("corners_world must be [B,K,S,8,3]")
    if corners_world.dtype != torch.float64:
        raise TypeError("corners_world must remain float64 until camera projection")
    b, k, s = (int(value) for value in corners_world.shape[:3])
    if b != cam.batch_size or s != cam.num_frames:
        raise ValueError("corners_world and CameraSpec batch/frame dimensions differ")
    if corners_world.device != cam.world_to_anchor.device:
        raise ValueError("corners_world and CameraSpec must share one device")
    if not bool(torch.isfinite(corners_world).all()):
        raise ValueError("corners_world contains NaN or Inf")
    if track_valid is None:
        track_valid = torch.ones((b, k, s), dtype=torch.bool, device=corners_world.device)
    else:
        _require_shape("track_valid", track_valid, (b, k, s))
        if track_valid.dtype != torch.bool:
            raise TypeError("track_valid must have bool dtype")
        if track_valid.device != corners_world.device:
            raise ValueError("track_valid and corners_world must share one device")

    near = float(cam.near_plane_m)
    far = float(far_plane_m)
    if not math.isfinite(far) or far <= near:
        raise ValueError("far_plane_m must be finite and exceed CameraSpec.near_plane_m")

    corners_camera64, uv_corners64 = _project_actor_points(corners_world, cam)
    centers_world = corners_world.mean(dim=-2, keepdim=True)
    centers_camera64, uv_center64 = _project_actor_points(centers_world, cam)
    centers_camera64 = centers_camera64.squeeze(-2)
    uv_center64 = uv_center64.squeeze(-2)
    center_z = centers_camera64[..., 2]
    center_depth_valid = (
        track_valid
        & torch.isfinite(centers_camera64).all(dim=-1)
        & center_z.ge(near)
        & center_z.le(far)
    )
    safe_center_z = torch.where(
        center_depth_valid, center_z, torch.ones_like(center_z)
    )
    log_z_w64 = torch.where(center_depth_valid, torch.log(safe_center_z), 0.0)
    uv_center64 = torch.where(center_depth_valid[..., None], uv_center64, 0.0)

    clipped_vertices, clipped_vertex_valid = _clipped_cuboid_vertices(
        corners_camera64, near_plane_m=near, far_plane_m=far
    )
    projected_h = torch.einsum(
        "bsij,bksqj->bksqi", cam.canvas_intrinsics, clipped_vertices
    )
    candidate_z = projected_h[..., 2:3]
    safe_candidate_z = torch.where(
        clipped_vertex_valid[..., None], candidate_z, torch.ones_like(candidate_z)
    )
    candidate_pixel = projected_h[..., :2] / safe_candidate_z
    canvas_h, canvas_w = cam.canvas_hw
    uv_candidates = candidate_pixel / candidate_pixel.new_tensor(
        (float(canvas_w), float(canvas_h))
    )
    clipped_vertex_valid = clipped_vertex_valid & torch.isfinite(uv_candidates).all(
        dim=-1
    )

    grid_h, grid_w = (int(value) for value in cam.patch_grid)
    patch_count = grid_h * grid_w
    bbox_patch64 = torch.full(
        (b, k, s, 4), -1.0, dtype=torch.float64, device=corners_world.device
    )
    patch_weight64 = torch.zeros(
        (b, k, s, patch_count), dtype=torch.float64, device=corners_world.device
    )
    log_z_patch64 = torch.zeros_like(patch_weight64)
    silhouette_uv64 = torch.zeros(
        (b, k, s, MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES, 2),
        dtype=torch.float64,
        device=corners_world.device,
    )
    silhouette_vertex_valid = torch.zeros(
        (b, k, s, MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES),
        dtype=torch.bool,
        device=corners_world.device,
    )
    frame_support = torch.zeros((b, k, s), dtype=torch.bool, device=corners_world.device)

    inverse_canvas_intrinsics = torch.linalg.inv(cam.canvas_intrinsics)
    candidate_rows = center_depth_valid.nonzero(as_tuple=False).detach().cpu().tolist()
    for batch_index, slot, frame_index in candidate_rows:
        candidate_mask = clipped_vertex_valid[batch_index, slot, frame_index]
        candidate_uv = uv_candidates[batch_index, slot, frame_index][candidate_mask]
        if int(candidate_uv.shape[0]) < 3:
            continue
        points = [
            (float(point[0]), float(point[1]))
            for point in candidate_uv.detach().cpu().tolist()
        ]
        hull = _convex_hull_2d(points)
        hull = _clip_polygon_to_rectangle(
            hull, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0
        )
        hull = _convex_hull_2d(hull)
        hull_area, _ = _polygon_area_and_centroid(hull)
        if len(hull) < 3 or hull_area <= 1.0e-14:
            continue
        if len(hull) > MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES:
            raise RuntimeError(
                "projected cuboid silhouette exceeds its fixed 32-vertex contract"
            )

        frame_corners = corners_camera64[batch_index, slot, frame_index]
        center = frame_corners.mean(dim=0)
        half_axis_x = 0.5 * (
            frame_corners[4:8].mean(dim=0) - frame_corners[0:4].mean(dim=0)
        )
        half_axis_y = 0.5 * (
            frame_corners[[2, 3, 6, 7]].mean(dim=0)
            - frame_corners[[0, 1, 4, 5]].mean(dim=0)
        )
        half_axis_z = 0.5 * (
            frame_corners[[1, 3, 5, 7]].mean(dim=0)
            - frame_corners[[0, 2, 4, 6]].mean(dim=0)
        )
        half_axes = torch.stack((half_axis_x, half_axis_y, half_axis_z), dim=-1)
        determinant = torch.linalg.det(half_axes)
        if not bool(torch.isfinite(determinant)) or abs(float(determinant)) <= 1.0e-12:
            continue
        inverse_half_axes_tensor = torch.linalg.inv(half_axes)
        local_camera_origin_tensor = inverse_half_axes_tensor @ (-center)
        inverse_half_axes = inverse_half_axes_tensor.detach().cpu().tolist()
        local_camera_origin = local_camera_origin_tensor.detach().cpu().tolist()
        inverse_intrinsic = inverse_canvas_intrinsics[
            batch_index, frame_index
        ].detach().cpu().tolist()

        patch_polygon = [
            (point[0] * float(grid_w), point[1] * float(grid_h)) for point in hull
        ]
        min_x = max(0, int(math.floor(min(point[0] for point in patch_polygon))))
        max_x = min(grid_w, int(math.ceil(max(point[0] for point in patch_polygon))))
        min_y = max(0, int(math.floor(min(point[1] for point in patch_polygon))))
        max_y = min(grid_h, int(math.ceil(max(point[1] for point in patch_polygon))))
        live_patch_count = 0
        for patch_y in range(min_y, max_y):
            for patch_x in range(min_x, max_x):
                overlap_polygon = _clip_polygon_to_rectangle(
                    patch_polygon,
                    x_min=float(patch_x),
                    x_max=float(patch_x + 1),
                    y_min=float(patch_y),
                    y_max=float(patch_y + 1),
                )
                overlap_area, overlap_centroid = _polygon_area_and_centroid(
                    overlap_polygon
                )
                if overlap_area <= 1.0e-12:
                    continue
                depth = _cuboid_patch_surface_depth(
                    overlap_polygon=overlap_polygon,
                    overlap_centroid=overlap_centroid,
                    patch_grid=(grid_h, grid_w),
                    canvas_hw=(canvas_h, canvas_w),
                    inverse_canvas_intrinsic=inverse_intrinsic,
                    inverse_half_axes=inverse_half_axes,
                    local_camera_origin=local_camera_origin,
                    near_plane_m=near,
                    far_plane_m=far,
                )
                if depth is None or not math.isfinite(depth) or depth <= 0.0:
                    # The polygon coverage and ray/slab intersection are
                    # independently rounded.  If no strictly local candidate
                    # has a mathematical hit, drop only this patch.  Remaining
                    # patches still determine frame_support below.
                    continue
                patch_index = patch_y * grid_w + patch_x
                patch_weight64[batch_index, slot, frame_index, patch_index] = min(
                    1.0, max(0.0, overlap_area)
                )
                log_z_patch64[batch_index, slot, frame_index, patch_index] = math.log(
                    depth
                )
                live_patch_count += 1
        if live_patch_count == 0:
            continue

        frame_support[batch_index, slot, frame_index] = True
        vertex_count = len(hull)
        silhouette_uv64[batch_index, slot, frame_index, :vertex_count] = torch.tensor(
            hull, dtype=torch.float64, device=corners_world.device
        )
        silhouette_vertex_valid[
            batch_index, slot, frame_index, :vertex_count
        ] = True
        u_values = [point[0] for point in patch_polygon]
        v_values = [point[1] for point in patch_polygon]
        bbox_patch64[batch_index, slot, frame_index] = torch.tensor(
            (min(u_values), min(v_values), max(u_values), max(v_values)),
            dtype=torch.float64,
            device=corners_world.device,
        )

    corners_camera64 = torch.where(
        track_valid[..., None, None], corners_camera64, 0.0
    )
    uv_corners64 = torch.where(
        track_valid[..., None, None], uv_corners64, 0.0
    )
    corners_camera = corners_camera64.float()
    # Freeze the support decision at the same float32 cache boundary that the
    # dataclass validator and late reader consume.  This avoids opposite
    # classifications from float64→float32 rounding exactly at 0.5/120 m.
    corner_z = corners_camera[..., 2]
    metric_support = (
        frame_support
        & corner_z.ge(near).all(dim=-1)
        & corner_z.le(far).all(dim=-1)
    )
    return {
        "bbox_patch": bbox_patch64.float(),
        "patch_weight": patch_weight64.float(),
        "log_z_patch": log_z_patch64.float(),
        "silhouette_uv": silhouette_uv64.float(),
        "silhouette_vertex_valid": silhouette_vertex_valid,
        "corners_camera": corners_camera,
        "uv_corners": uv_corners64.float(),
        "uv_center": uv_center64.float(),
        "log_z_w": log_z_w64.float(),
        "center_depth_valid": center_depth_valid,
        "frame_support": frame_support,
        "metric_support": metric_support,
    }


def project_actor_geometry(
    g: ActorGeometryCondition,
    cam: CameraSpec,
) -> ProjectedActorGeometry:
    """Build the only camera-space cache from ``(G_world, requested C)``.

    The visible cuboid is clipped to the inclusive optical-z interval
    ``[near_plane_m, 120 m]`` before its canvas silhouette, fractional patch
    coverage, and per-patch nearest-surface depth are derived.  Original metric
    corners remain unclamped for the late full-K reader.
    """

    g.validate()
    cam.validate()
    if g.batch_size != cam.batch_size or g.num_frames != cam.num_frames:
        raise ValueError(
            "ActorGeometryCondition and CameraSpec batch/frame dimensions differ"
        )
    if g.corners_world.device != cam.world_to_anchor.device:
        raise ValueError("ActorGeometryCondition and CameraSpec must share one device")

    track_valid = g.track_valid & g.slot_valid[..., None]
    cuboid = project_actor_cuboid_corners(
        g.corners_world,
        cam,
        track_valid=track_valid,
        far_plane_m=ACTOR_FAR_PLANE_M,
    )
    rotation_world_to_camera = torch.matmul(
        cam.anchor_to_camera[..., :3, :3],
        cam.world_to_anchor[:, None, :3, :3],
    )
    velocity_camera64 = torch.einsum(
        "bsij,bksj->bksi", rotation_world_to_camera, g.velocity_world.double()
    )
    velocity_camera64 = torch.where(
        track_valid[..., None], velocity_camera64, 0.0
    )

    return ProjectedActorGeometry(
        bbox_patch=cuboid["bbox_patch"],
        patch_weight=cuboid["patch_weight"],
        log_z_patch=cuboid["log_z_patch"],
        silhouette_uv=cuboid["silhouette_uv"],
        silhouette_vertex_valid=cuboid["silhouette_vertex_valid"],
        corners_camera=cuboid["corners_camera"],
        uv_corners=cuboid["uv_corners"],
        velocity_camera=velocity_camera64.float(),
        uv_center=cuboid["uv_center"],
        log_z_w=cuboid["log_z_w"],
        center_depth_valid=cuboid["center_depth_valid"],
        frame_support=cuboid["frame_support"],
        metric_support=cuboid["metric_support"],
        in_frustum=cuboid["frame_support"],
        valid=track_valid,
    )


def gather_actor_slots(
    actor_values: torch.Tensor,
    geometry_idx: torch.Tensor,
    *,
    selection_mask: torch.Tensor | None = None,
    fill_value: float | int | bool = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Safely gather ``[B,Kg,...]`` values with ``[B,Ka]`` geometry indices.

    Invalid indices are clamped *before* ``torch.gather`` and then masked from
    the result.  In particular, ``geometry_idx == -1`` can never select the
    final actor slot.  The returned bool tensor is the effective in-range
    selection mask with shape ``[B,Ka]``.
    """

    if actor_values.ndim < 2:
        raise ValueError("actor_values must be [B,Kg,...]")
    if geometry_idx.ndim != 2:
        raise ValueError("geometry_idx must be [B,Ka]")
    if geometry_idx.dtype != torch.int64:
        raise TypeError("geometry_idx must have int64 dtype")
    b, kg = (int(v) for v in actor_values.shape[:2])
    if kg <= 0:
        raise ValueError("cannot gather from an empty actor axis")
    if int(geometry_idx.shape[0]) != b:
        raise ValueError("actor_values and geometry_idx batch sizes differ")
    if actor_values.device != geometry_idx.device:
        raise ValueError("actor_values and geometry_idx must share one device")
    in_range = geometry_idx.ge(0) & geometry_idx.lt(kg)
    if selection_mask is not None:
        _require_shape("selection_mask", selection_mask, tuple(geometry_idx.shape))
        if selection_mask.dtype != torch.bool:
            raise TypeError("selection_mask must have bool dtype")
        if selection_mask.device != geometry_idx.device:
            raise ValueError("selection_mask and geometry_idx must share one device")
        in_range = in_range & selection_mask

    # This ordering is the critical D3 gather invariant: clamp, gather, mask.
    safe_idx = geometry_idx.clamp(min=0, max=kg - 1)
    trailing = tuple(int(v) for v in actor_values.shape[2:])
    gather_idx = safe_idx.reshape(
        b, int(safe_idx.shape[1]), *([1] * len(trailing))
    ).expand(b, int(safe_idx.shape[1]), *trailing)
    gathered = torch.gather(actor_values, dim=1, index=gather_idx)
    expanded_mask = in_range.reshape(
        b, int(in_range.shape[1]), *([1] * len(trailing))
    ).expand_as(gathered)
    fill = torch.as_tensor(fill_value, dtype=gathered.dtype, device=gathered.device)
    gathered = torch.where(expanded_mask, gathered, fill)
    return gathered, in_range


__all__ = [
    "ACTOR_FAR_PLANE_M",
    "ACTOR_NEAR_PLANE_M",
    "ActorGeometryCondition",
    "CameraSpec",
    "LayoutMode",
    "MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES",
    "ProjectedActorGeometry",
    "gather_actor_slots",
    "project_actor_cuboid_corners",
    "project_actor_geometry",
    "project_actor_world_points",
    "project_world_points",
    "raw_to_model_canvas_homography",
]
