"""Online HD-map/actor projection and gauge re-read primitives.

The vector sidecar and :class:`ActorGeometryCondition` are camera-independent
truth.  This module is the one online path that turns them into the requested
camera's image-space layout.  It deliberately keeps three domains separate:

* Waymo world/anchor/camera geometry stays float64 until the large anchor
  origin has been removed;
* raster channels contain only image-space, dimensionless quantities and are
  quantized with the frozen 33-channel schema;
* metric depth is exposed only through ``map_metric`` and the FP32 three-scalar
  gauge re-read helpers.

No content digest is computed here.  ``RASTER_SCHEMA_HASH`` is the explicit
frozen schema identifier owned by ``datasets.tools.hdmap_schema``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

# OpenCV defaults to the host's full core count (128 on the training node).
# Eight DataLoader workers would otherwise create roughly one thousand native
# threads for tiny per-feature draws and collapse throughput through
# oversubscription.  Rasterization is parallelized at the sample/worker level.
cv2.setNumThreads(1)

from datasets.tools.hdmap_schema import (
    ATTRIBUTE_SOURCE_NONE,
    CLASS_IDS,
    HDMapFeature,
    HDMapScene,
    MAP_METRIC_RESERVED_ZERO_GROUPS,
    RASTER_ACTOR_CHANNELS,
    RASTER_MAP_CHANNELS,
    RASTER_RESERVED_ZERO_CHANNELS,
    RASTER_SOURCE_EXCLUDED_CLASSES,
    RASTER_CHANNEL_COUNT,
    RASTER_FLAG_CHANNELS,
    RASTER_QUANTIZATION,
    RASTER_SCHEMA_HASH,
    RASTER_STATIC_VALID_CHANNEL,
)
from dggt.utils.actor_geometry_condition import (
    ACTOR_FAR_PLANE_M,
    ActorGeometryCondition,
    CameraSpec,
    ProjectedActorGeometry,
    project_actor_geometry,
)
from dggt.utils.layout_condition import MapMode


LAYOUT_RASTER_HW: tuple[int, int] = (100, 148)
LAYOUT_OVERSAMPLE = 4
LAYOUT_MAP_GROUPS = 5
STATIC_FAR_PLANE_M = 120.0
MAP_GROUP_BY_CLASS: dict[str, int] = {
    "road_line": 1,
    "road_edge": 2,
    "crosswalk": 3,
    "speed_bump": 3,
    "stop_sign": 4,
}


@dataclass(frozen=True)
class GaugeRereadResult:
    """FP32, fail-closed result of the three-scalar gauge identity."""

    features: torch.Tensor
    valid: torch.Tensor
    x_over_z: torch.Tensor
    y_over_z: torch.Tensor
    log_z_d: torch.Tensor


@dataclass(frozen=True)
class ThinLineProjectionTheory:
    """Projected thin-line denominator for the real-geometry T23 check."""

    expected_patch_count: int
    projected_length_patch_units: float
    visible_segment_count: int


@dataclass(frozen=True)
class CameraProjectionSnapshot:
    """Exact, non-hashed snapshot used only to reject stale derived caches."""

    world_to_anchor: torch.Tensor
    anchor_to_camera: torch.Tensor
    intrinsics: torch.Tensor
    raw_to_canvas: torch.Tensor
    map_pose_offset: torch.Tensor
    canvas_hw: tuple[int, int]
    patch_grid: tuple[int, int]
    near_plane_m: float

    @classmethod
    def from_camera(cls, cam: CameraSpec) -> "CameraProjectionSnapshot":
        return cls(
            world_to_anchor=cam.world_to_anchor.detach().cpu().clone(),
            anchor_to_camera=cam.anchor_to_camera.detach().cpu().clone(),
            intrinsics=cam.intrinsics.detach().cpu().clone(),
            raw_to_canvas=cam.raw_to_canvas.detach().cpu().clone(),
            map_pose_offset=cam.map_pose_offset.detach().cpu().clone(),
            canvas_hw=tuple(int(v) for v in cam.canvas_hw),
            patch_grid=tuple(int(v) for v in cam.patch_grid),
            near_plane_m=float(cam.near_plane_m),
        )

    def matches(self, cam: CameraSpec) -> bool:
        other = CameraProjectionSnapshot.from_camera(cam)
        return (
            self.canvas_hw == other.canvas_hw
            and self.patch_grid == other.patch_grid
            and self.near_plane_m == other.near_plane_m
            and torch.equal(self.world_to_anchor, other.world_to_anchor)
            and torch.equal(self.anchor_to_camera, other.anchor_to_camera)
            and torch.equal(self.intrinsics, other.intrinsics)
            and torch.equal(self.raw_to_canvas, other.raw_to_canvas)
            and torch.equal(self.map_pose_offset, other.map_pose_offset)
        )


@dataclass(frozen=True)
class ProjectedMapLayout:
    """Static/dynamic image layout and metric map sidecar for one camera."""

    layout_raster: torch.Tensor
    map_metric: torch.Tensor
    map_mode: torch.Tensor
    raster_schema_hash: str
    static_far_plane_m: float
    camera_snapshot: CameraProjectionSnapshot
    # HD-map polygons this projection had to drop because they could not be
    # triangulated.  Dropping is fail-closed, but a segment that drops many of
    # them degrades to a factual EMPTY map, so the count belongs in the run
    # summary rather than in silence.
    dropped_primitives: int = 0

    def __post_init__(self) -> None:
        if self.layout_raster.ndim != 5:
            raise ValueError("layout_raster must be [B,S,33,H,W]")
        b, s, channels = (int(v) for v in self.layout_raster.shape[:3])
        if channels != RASTER_CHANNEL_COUNT:
            raise ValueError(f"layout_raster requires {RASTER_CHANNEL_COUNT} channels")
        if self.layout_raster.dtype != torch.uint8:
            raise TypeError("layout_raster must have uint8 dtype")
        height, width = (int(v) for v in self.layout_raster.shape[-2:])
        expected_patches = (height // 4) * (width // 4)
        if self.map_metric.ndim != 5 or tuple(self.map_metric.shape[:3]) != (
            b,
            s,
            expected_patches,
        ):
            raise ValueError("map_metric must be [B,S,P,Gm,4]")
        if tuple(self.map_metric.shape[-2:]) != (LAYOUT_MAP_GROUPS, 4):
            raise ValueError("map_metric must have five (u,v,log_z_w,valid) groups")
        if self.map_metric.dtype != torch.float32:
            raise TypeError("map_metric must have float32 dtype")
        if tuple(self.map_mode.shape) != (b,) or self.map_mode.dtype != torch.int8:
            raise TypeError("map_mode must be int8 [B]")
        if self.raster_schema_hash != RASTER_SCHEMA_HASH:
            raise ValueError("layout raster schema identifier mismatch")
        if float(self.static_far_plane_m) != STATIC_FAR_PLANE_M:
            raise ValueError(
                "static_far_plane_m must match the frozen 120 m layout contract"
            )
        if not bool(torch.isfinite(self.map_metric).all()):
            raise ValueError("map_metric contains NaN or Inf")
        if bool(self.layout_raster[:, :, RASTER_RESERVED_ZERO_CHANNELS].any()):
            raise ValueError("projected raster contains excluded static-map features")
        if bool(self.map_metric[..., MAP_METRIC_RESERVED_ZERO_GROUPS, :].any()):
            raise ValueError(
                "projected map_metric contains excluded lane-centerline features"
            )

    def assert_current(self, cam: CameraSpec) -> None:
        if not self.camera_snapshot.matches(cam):
            raise ValueError(
                "stale layout projection: requested camera/crop/offset changed; "
                "recompute M and G together"
            )


@dataclass(frozen=True)
class LayoutProjection:
    """Atomic M+G projection result; consumers must not update one half."""

    map_layout: ProjectedMapLayout
    actor_geometry: ProjectedActorGeometry

    def assert_current(self, cam: CameraSpec) -> None:
        self.map_layout.assert_current(cam)


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)  # type: ignore[attr-defined]
        return value

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (scale,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return grad_output * scale, None


def scale_gradient(
    value: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:
    """Identity in the forward pass and ``scale`` in the backward pass."""

    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError("value must be a floating-point tensor")
    scale_tensor = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
    if scale_tensor.numel() != 1 or not bool(torch.isfinite(scale_tensor).all()):
        raise ValueError("gradient scale must be one finite scalar")
    if bool((scale_tensor < 0.0).any()):
        raise ValueError("gradient scale must be non-negative")
    return _ScaleGradient.apply(value, scale_tensor)


def reread_metric_geometry(
    uv: torch.Tensor,
    log_z_w: torch.Tensor,
    gauge_physical: torch.Tensor,
    valid: torch.Tensor,
    *,
    gauge_valid: torch.Tensor | None = None,
    near_plane_m: float = 0.5,
    ray_abs_max: float = 16.0,
    log_depth_abs_max: float = 20.0,
) -> GaugeRereadResult:
    """Apply the v2.1 three-scalar identity in a strict FP32 island.

    ``gauge_physical`` is scene-global ``[ell, ax, ay]`` with shape ``[B,3]``.
    Invalid rows are replaced *before* exp/log/division.  Returned invalid
    features are bitwise zero, including for non-finite or over-support input.
    """

    if uv.ndim < 2 or int(uv.shape[-1]) != 2:
        raise ValueError("uv must be [B,...,2]")
    if tuple(log_z_w.shape) != tuple(uv.shape[:-1]):
        raise ValueError("log_z_w must match uv leading dimensions")
    if tuple(valid.shape) != tuple(log_z_w.shape) or valid.dtype != torch.bool:
        raise TypeError("valid must be bool and match log_z_w")
    if gauge_physical.ndim != 2 or tuple(gauge_physical.shape) != (
        int(uv.shape[0]),
        3,
    ):
        raise ValueError("gauge_physical must be [B,3]")
    if (
        not uv.is_floating_point()
        or not log_z_w.is_floating_point()
        or not gauge_physical.is_floating_point()
    ):
        raise TypeError("uv, log_z_w, and gauge_physical must be floating point")
    if len({uv.device, log_z_w.device, gauge_physical.device, valid.device}) != 1:
        raise ValueError("gauge re-read inputs must share one device")
    if not (float(near_plane_m) > 0.0):
        raise ValueError("near_plane_m must be positive")
    if not (float(ray_abs_max) > 0.0 and float(log_depth_abs_max) > 0.0):
        raise ValueError("support limits must be positive")
    batch_size = int(uv.shape[0])
    if gauge_valid is None:
        gauge_valid = torch.ones((batch_size,), device=uv.device, dtype=torch.bool)
    elif tuple(gauge_valid.shape) != (batch_size,) or gauge_valid.dtype != torch.bool:
        raise TypeError("gauge_valid must be bool [B]")

    device_type = uv.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        uv32 = uv.float()
        log_z_w32 = log_z_w.float()
        gauge32 = gauge_physical.float()
        finite_gauge = gauge32.isfinite().all(dim=-1)
        # Reserve headroom before exp so a masked overflow can never poison
        # backward with 0*Inf.  The test is detached; only supported rows enter
        # the live exponential graph.
        exp_limit = math.log(torch.finfo(torch.float32).max) - 4.0
        ell_diag, ax_diag, ay_diag = gauge32.detach().double().unbind(dim=-1)
        exponent_args = torch.stack((-ell_diag, ax_diag, ay_diag), dim=-1)
        exponent_safe = exponent_args.abs().le(exp_limit).all(dim=-1)
        row_valid = gauge_valid & finite_gauge & exponent_safe
        safe_gauge = torch.where(row_valid[:, None], gauge32, torch.zeros_like(gauge32))
        ell, ax, ay = safe_gauge.unbind(dim=-1)
        expand_shape = (batch_size,) + (1,) * (uv32.ndim - 2)
        exp_ax = torch.exp(ax).reshape(expand_shape)
        exp_ay = torch.exp(ay).reshape(expand_shape)
        ell = ell.reshape(expand_shape)
        u, v = uv32.unbind(dim=-1)
        x_over_z = 2.0 * exp_ax * (u - 0.5)
        y_over_z = 2.0 * exp_ay * (v - 0.5)
        log_z_d = log_z_w32 - ell

        finite_input = uv32.isfinite().all(dim=-1) & log_z_w32.isfinite()
        metric_front = log_z_w32.ge(math.log(float(near_plane_m)))
        support = (
            x_over_z.isfinite()
            & y_over_z.isfinite()
            & log_z_d.isfinite()
            & x_over_z.abs().le(float(ray_abs_max))
            & y_over_z.abs().le(float(ray_abs_max))
            & log_z_d.abs().le(float(log_depth_abs_max))
        )
        row_shape = (batch_size,) + (1,) * (valid.ndim - 1)
        effective_valid = (
            valid & finite_input & metric_front & support & row_valid.reshape(row_shape)
        )
        zero = torch.zeros_like(x_over_z)
        x_over_z = torch.where(effective_valid, x_over_z, zero)
        y_over_z = torch.where(effective_valid, y_over_z, zero)
        log_z_d = torch.where(effective_valid, log_z_d, zero)
        features = torch.stack((x_over_z, y_over_z, log_z_d), dim=-1)
    return GaugeRereadResult(
        features=features,
        valid=effective_valid,
        x_over_z=x_over_z,
        y_over_z=y_over_z,
        log_z_d=log_z_d,
    )


def build_map_metric_features(
    map_metric: torch.Tensor,
    gauge_physical: torch.Tensor,
    *,
    gauge_valid: torch.Tensor | None = None,
    grad_scale: float | torch.Tensor = 1.0,
    near_plane_m: float = 0.5,
    ray_abs_max: float = 16.0,
    log_depth_abs_max: float = 20.0,
) -> torch.Tensor:
    """Return the corrected ``Gm*3+Gm`` late-map adapter input."""

    if map_metric.ndim != 5 or int(map_metric.shape[-1]) != 4:
        raise ValueError("map_metric must be [B,S,P,Gm,4]")
    if not map_metric.is_floating_point():
        raise TypeError("map_metric must be floating point")
    gauge_for_layout = scale_gradient(gauge_physical, grad_scale)
    u, v, log_z_w, valid_float = map_metric.unbind(dim=-1)
    valid = valid_float.gt(0.5)
    reread = reread_metric_geometry(
        torch.stack((u, v), dim=-1),
        log_z_w,
        gauge_for_layout,
        valid,
        gauge_valid=gauge_valid,
        near_plane_m=near_plane_m,
        ray_abs_max=ray_abs_max,
        log_depth_abs_max=log_depth_abs_max,
    )
    features = reread.features * reread.valid[..., None].float()
    return torch.cat((features.flatten(start_dim=-2), reread.valid.float()), dim=-1)


def quantize_layout_raster(
    raster: np.ndarray | torch.Tensor,
    *,
    raster_schema_hash: str = RASTER_SCHEMA_HASH,
) -> torch.Tensor:
    """Quantize ``[...,33,H,W]`` according to the frozen channel table."""

    if raster_schema_hash != RASTER_SCHEMA_HASH:
        raise ValueError("raster_schema_hash does not match the frozen schema")
    value = torch.as_tensor(raster)
    if value.ndim < 3 or int(value.shape[-3]) != RASTER_CHANNEL_COUNT:
        raise ValueError("raster must have channel axis [...,33,H,W]")
    if not value.is_floating_point():
        raise TypeError("unquantized raster must be floating point")
    value32 = value.float()
    if not bool(torch.isfinite(value32).all()):
        raise ValueError("unquantized raster contains NaN or Inf")
    output = torch.empty_like(value32, dtype=torch.uint8)
    for channel, (scale, zero_point, clip_lo, clip_hi) in enumerate(
        RASTER_QUANTIZATION
    ):
        channel_value = value32[..., channel, :, :].clamp(
            float(clip_lo), float(clip_hi)
        )
        quantized = torch.round(channel_value / float(scale) + int(zero_point))
        output[..., channel, :, :] = quantized.clamp(0.0, 255.0).to(torch.uint8)
    return output


def dequantize_layout_raster(
    raster: np.ndarray | torch.Tensor,
    *,
    raster_schema_hash: str = RASTER_SCHEMA_HASH,
) -> torch.Tensor:
    """Dequantize uint8 layout while asserting the schema in both directions."""

    if raster_schema_hash != RASTER_SCHEMA_HASH:
        raise ValueError("raster_schema_hash does not match the frozen schema")
    value = torch.as_tensor(raster)
    if value.ndim < 3 or int(value.shape[-3]) != RASTER_CHANNEL_COUNT:
        raise ValueError("raster must have channel axis [...,33,H,W]")
    if value.dtype != torch.uint8:
        raise TypeError("quantized raster must have uint8 dtype")
    output = torch.empty_like(value, dtype=torch.float32)
    for channel, (scale, zero_point, _clip_lo, _clip_hi) in enumerate(
        RASTER_QUANTIZATION
    ):
        output[..., channel, :, :] = (
            value[..., channel, :, :].float() - int(zero_point)
        ) * float(scale)
    return output


def clip_segment_to_depth_range(
    start_camera: np.ndarray | Sequence[float],
    end_camera: np.ndarray | Sequence[float],
    *,
    near_plane_m: float = 0.5,
    far_plane_m: float = STATIC_FAR_PLANE_M,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip one camera-space segment to ``near <= z <= far`` exactly."""

    start = np.asarray(start_camera, dtype=np.float64).reshape(3)
    end = np.asarray(end_camera, dtype=np.float64).reshape(3)
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return None
    near, far = float(near_plane_m), float(far_plane_m)
    if not math.isfinite(near) or not math.isfinite(far) or not 0.0 < near < far:
        raise ValueError(
            "depth range must satisfy finite 0 < near_plane_m < far_plane_m"
        )

    delta = end - start
    delta_z = float(delta[2])
    if delta_z == 0.0:
        if near <= float(start[2]) <= far:
            return start.copy(), end.copy()
        return None
    t_near = (near - float(start[2])) / delta_z
    t_far = (far - float(start[2])) / delta_z
    t0 = max(0.0, min(t_near, t_far))
    t1 = min(1.0, max(t_near, t_far))
    if t0 > t1:
        return None
    clipped_start = start + t0 * delta
    clipped_end = start + t1 * delta
    clipped_start[2] = float(np.clip(clipped_start[2], near, far))
    clipped_end[2] = float(np.clip(clipped_end[2], near, far))
    return clipped_start, clipped_end


def _clip_polygon_to_z_half_space(
    vertices_camera: np.ndarray,
    plane_m: float,
    *,
    keep_greater: bool,
) -> np.ndarray:
    """Sutherland-Hodgman clip against one optical-z half-space."""

    vertices = np.asarray(vertices_camera, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        return np.zeros((0, 3), dtype=np.float64)
    output: list[np.ndarray] = []
    previous = vertices[-1]
    previous_inside = bool(
        previous[2] >= plane_m if keep_greater else previous[2] <= plane_m
    )
    for current in vertices:
        current_inside = bool(
            current[2] >= plane_m if keep_greater else current[2] <= plane_m
        )
        if current_inside != previous_inside:
            denominator = float(current[2] - previous[2])
            if denominator != 0.0:
                t = (float(plane_m) - float(previous[2])) / denominator
                intersection = previous + np.clip(t, 0.0, 1.0) * (current - previous)
                intersection[2] = float(plane_m)
                output.append(intersection)
        if current_inside:
            output.append(current.copy())
        previous = current
        previous_inside = current_inside
    if len(output) < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(output, axis=0)


def _clip_polygon_to_depth_range(
    vertices_camera: np.ndarray,
    near_plane_m: float,
    far_plane_m: float,
) -> np.ndarray:
    """Clip one 3-D polygon to the closed optical-depth slab."""

    near, far = float(near_plane_m), float(far_plane_m)
    if not math.isfinite(near) or not math.isfinite(far) or not 0.0 < near < far:
        raise ValueError(
            "depth range must satisfy finite 0 < near_plane_m < far_plane_m"
        )
    clipped = _clip_polygon_to_z_half_space(
        vertices_camera,
        near,
        keep_greater=True,
    )
    if len(clipped) < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return _clip_polygon_to_z_half_space(clipped, far, keep_greater=False)


@dataclass(frozen=True)
class _NumpyFrameProjector:
    """One frame's float64 coordinate chain cached on the CPU.

    Static map rasterization used to read CameraSpec tensors and recompute
    raw_to_canvas @ K for every polyline segment.  Real Waymo scenes contain
    tens of thousands of segments, so that crossed the Torch/NumPy boundary
    hundreds of thousands of times per sample.  This immutable frame cache
    keeps the exact two-stage anchor transform while making projection a
    batched NumPy matrix multiply.
    """

    map_pose_offset: np.ndarray
    world_to_anchor: np.ndarray
    anchor_to_camera: np.ndarray
    canvas_intrinsic: np.ndarray
    canvas_scale: np.ndarray
    near_plane_m: float
    static_far_plane_m: float
    patch_grid: tuple[int, int]

    @classmethod
    def from_camera(
        cls,
        cam: CameraSpec,
        batch_index: int,
        frame_index: int,
        *,
        static_far_plane_m: float,
    ) -> "_NumpyFrameProjector":
        raw_to_canvas = (
            cam.raw_to_canvas[batch_index, frame_index].detach().cpu().numpy()
        )
        intrinsic = cam.intrinsics[batch_index, frame_index].detach().cpu().numpy()
        canvas_h, canvas_w = cam.canvas_hw
        far = float(static_far_plane_m)
        if not math.isfinite(far) or far <= float(cam.near_plane_m):
            raise ValueError(
                "static_far_plane_m must be finite and exceed near_plane_m"
            )
        return cls(
            map_pose_offset=(cam.map_pose_offset[batch_index].detach().cpu().numpy()),
            world_to_anchor=(cam.world_to_anchor[batch_index].detach().cpu().numpy()),
            anchor_to_camera=(
                cam.anchor_to_camera[batch_index, frame_index].detach().cpu().numpy()
            ),
            canvas_intrinsic=raw_to_canvas @ intrinsic,
            canvas_scale=np.asarray((canvas_w, canvas_h), dtype=np.float64),
            near_plane_m=float(cam.near_plane_m),
            static_far_plane_m=far,
            patch_grid=tuple(int(value) for value in cam.patch_grid),
        )

    def world_to_camera(self, vertices_world: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices_world, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("world vertices must be [N,3]")
        # Keep float64 through the large-world subtraction and preserve the
        # specified world->anchor->camera operation order.
        corrected = vertices - self.map_pose_offset[None]
        corrected_h = np.concatenate(
            (
                corrected,
                np.ones((len(corrected), 1), dtype=np.float64),
            ),
            axis=-1,
        )
        anchor_h = (self.world_to_anchor @ corrected_h.T).T
        camera_h = (self.anchor_to_camera @ anchor_h.T).T
        return camera_h[:, :3]

    def project(self, vertices_camera: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices_camera, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("camera vertices must be [N,3]")
        if len(vertices) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        projected_h = (self.canvas_intrinsic @ vertices.T).T
        uv_pixel = projected_h[:, :2] / projected_h[:, 2:3]
        return uv_pixel / self.canvas_scale


def _world_to_camera_numpy(
    vertices_world: np.ndarray,
    projector: _NumpyFrameProjector,
) -> np.ndarray:
    return projector.world_to_camera(vertices_world)


def _project_camera_numpy(
    vertices_camera: np.ndarray,
    projector: _NumpyFrameProjector,
) -> np.ndarray:
    return projector.project(vertices_camera)


def _clip_line_to_canvas(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky clip in finite pixel coordinates."""

    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return None
    delta = end - start
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-delta[0], start[0]),
        (delta[0], float(width - 1) - start[0]),
        (-delta[1], start[1]),
        (delta[1], float(height - 1) - start[1]),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            if ratio > t1:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0:
                return None
            t1 = min(t1, ratio)
    return start + t0 * delta, start + t1 * delta


def _clip_polyline_segments_to_depth_range(
    vertices_camera: np.ndarray,
    near_plane_m: float,
    far_plane_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized clipping of every adjacent segment to optical-depth range."""

    vertices = np.asarray(vertices_camera, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty.copy()
    near, far = float(near_plane_m), float(far_plane_m)
    if not math.isfinite(near) or not math.isfinite(far) or not 0.0 < near < far:
        raise ValueError(
            "depth range must satisfy finite 0 < near_plane_m < far_plane_m"
        )
    start = vertices[:-1]
    end = vertices[1:]
    finite = np.isfinite(start).all(axis=-1) & np.isfinite(end).all(axis=-1)
    segment_min = np.minimum(start[:, 2], end[:, 2])
    segment_max = np.maximum(start[:, 2], end[:, 2])
    keep = finite & (segment_max >= near) & (segment_min <= far)
    if not bool(keep.any()):
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty.copy()
    start = start[keep].copy()
    end = end[keep].copy()
    delta = end - start
    delta_z = delta[:, 2]
    parallel = delta_z == 0.0
    safe_delta_z = np.where(parallel, 1.0, delta_z)
    t_near = (near - start[:, 2]) / safe_delta_z
    t_far = (far - start[:, 2]) / safe_delta_z
    t0 = np.maximum(0.0, np.minimum(t_near, t_far))
    t1 = np.minimum(1.0, np.maximum(t_near, t_far))
    t0 = np.where(parallel, 0.0, t0)
    t1 = np.where(parallel, 1.0, t1)
    valid = t0 <= t1
    clipped_start = start + t0[:, None] * delta
    clipped_end = start + t1[:, None] * delta
    clipped_start[:, 2] = np.clip(clipped_start[:, 2], near, far)
    clipped_end[:, 2] = np.clip(clipped_end[:, 2], near, far)
    return clipped_start[valid], clipped_end[valid]


def _clip_lines_to_canvas(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Liang-Barsky clipping for independent two-point curves."""

    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    if start.ndim != 2 or end.shape != start.shape or int(start.shape[1]) != 2:
        raise ValueError("line endpoints must both be [N,2]")
    count = int(start.shape[0])
    if count == 0:
        return start.copy(), end.copy(), np.zeros((0,), dtype=np.bool_)
    delta = end - start
    t0 = np.zeros((count,), dtype=np.float64)
    t1 = np.ones((count,), dtype=np.float64)
    valid = np.isfinite(start).all(axis=-1) & np.isfinite(end).all(axis=-1)
    for p, q in (
        (-delta[:, 0], start[:, 0]),
        (delta[:, 0], float(width - 1) - start[:, 0]),
        (-delta[:, 1], start[:, 1]),
        (delta[:, 1], float(height - 1) - start[:, 1]),
    ):
        parallel = p == 0.0
        valid &= ~(parallel & (q < 0.0))
        ratio = np.zeros_like(q)
        np.divide(q, p, out=ratio, where=~parallel)
        entering = (~parallel) & (p < 0.0)
        leaving = (~parallel) & (p > 0.0)
        valid &= ~(entering & (ratio > t1))
        valid &= ~(leaving & (ratio < t0))
        t0 = np.where(entering, np.maximum(t0, ratio), t0)
        t1 = np.where(leaving, np.minimum(t1, ratio), t1)
    valid &= t0 <= t1
    return (
        start + t0[:, None] * delta,
        start + t1[:, None] * delta,
        valid,
    )


@dataclass(frozen=True)
class _LocalGridGeometry:
    """Nearest visible primitive geometry for every supported image cell."""

    support: np.ndarray  # [H,W] bool, derived from the exact supersampled mask
    depth: np.ndarray  # [H,W] float64, inf outside geometrically resolved support
    uv: np.ndarray  # [H,W,2] float64 normalized canvas coordinates
    tangent: np.ndarray  # [H,W,2] float32 image-plane unit tangent


@dataclass(frozen=True)
class _FeatureLocalGeometry:
    """One clipped primitive rasterized once for both early and late readers."""

    high_mask: np.ndarray
    raster: _LocalGridGeometry
    patch: _LocalGridGeometry
    # Polygon components that survived depth clipping but could not be
    # triangulated.  Dropping them is fail-closed, but a silent drop would let a
    # segment degrade to a factual EMPTY map with nobody noticing, so the count
    # is carried up into the projection for the run summary.
    dropped_primitives: int = 0


def _empty_local_grid(support: np.ndarray) -> _LocalGridGeometry:
    support = np.asarray(support, dtype=np.bool_)
    return _LocalGridGeometry(
        support=support,
        depth=np.full(support.shape, np.inf, dtype=np.float64),
        uv=np.zeros(support.shape + (2,), dtype=np.float64),
        tangent=np.zeros(support.shape + (2,), dtype=np.float32),
    )


def _segment_rect_intervals(
    start_uv: np.ndarray,
    end_uv: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Liang-Barsky intervals for one segment against many axis-aligned cells."""

    start = np.asarray(start_uv, dtype=np.float64).reshape(2)
    end = np.asarray(end_uv, dtype=np.float64).reshape(2)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.ndim != 2 or lower.shape[1] != 2 or upper.shape != lower.shape:
        raise ValueError("cell lower/upper bounds must both be [N,2]")
    count = int(lower.shape[0])
    t0 = np.zeros((count,), dtype=np.float64)
    t1 = np.ones((count,), dtype=np.float64)
    valid = np.isfinite(start).all() & np.isfinite(end).all()
    valid = np.full((count,), bool(valid), dtype=np.bool_)
    delta = end - start
    for p, q in (
        (-delta[0], start[0] - lower[:, 0]),
        (delta[0], upper[:, 0] - start[0]),
        (-delta[1], start[1] - lower[:, 1]),
        (delta[1], upper[:, 1] - start[1]),
    ):
        if abs(float(p)) <= np.finfo(np.float64).eps:
            valid &= q >= 0.0
            continue
        ratio = q / float(p)
        if p < 0.0:
            valid &= ratio <= t1
            t0 = np.maximum(t0, ratio)
        else:
            valid &= ratio >= t0
            t1 = np.minimum(t1, ratio)
    valid &= t0 <= t1
    return valid, t0, t1


def _update_line_grid(
    grid: _LocalGridGeometry,
    uv_start: np.ndarray,
    uv_end: np.ndarray,
    depth_start: np.ndarray,
    depth_end: np.ndarray,
    *,
    canvas_scale: np.ndarray,
    margin_uv: tuple[float, float],
) -> None:
    """Resolve local line depth/tangent analytically in every covered cell."""

    cells = np.argwhere(grid.support)
    if not len(cells):
        return
    grid_h, grid_w = grid.support.shape
    y = cells[:, 0]
    x = cells[:, 1]
    margin_u, margin_v = (float(value) for value in margin_uv)
    lower = np.stack(
        (x / float(grid_w) - margin_u, y / float(grid_h) - margin_v),
        axis=-1,
    )
    upper = np.stack(
        (
            (x + 1) / float(grid_w) + margin_u,
            (y + 1) / float(grid_h) + margin_v,
        ),
        axis=-1,
    )
    for start, end, z0, z1 in zip(uv_start, uv_end, depth_start, depth_end):
        if (
            not np.isfinite(start).all()
            or not np.isfinite(end).all()
            or not math.isfinite(float(z0))
            or not math.isfinite(float(z1))
            or float(z0) <= 0.0
            or float(z1) <= 0.0
        ):
            continue
        valid, t0, t1 = _segment_rect_intervals(start, end, lower, upper)
        if not bool(valid.any()):
            continue
        # Optical depth is harmonic along the projected 2-D segment.  It is
        # monotonic, so the shallower clipped endpoint is the exact cell min.
        choose_t0 = float(z0) <= float(z1)
        t = np.where(choose_t0, t0, t1)
        inv_depth = (1.0 - t) / float(z0) + t / float(z1)
        candidate_depth = np.divide(
            1.0,
            inv_depth,
            out=np.full_like(inv_depth, np.inf),
            where=inv_depth > 0.0,
        )
        current = grid.depth[y, x]
        update = valid & np.isfinite(candidate_depth) & (candidate_depth < current)
        if not bool(update.any()):
            continue
        selected_y = y[update]
        selected_x = x[update]
        selected_t = t[update]
        grid.depth[selected_y, selected_x] = candidate_depth[update]
        grid.uv[selected_y, selected_x] = (
            start[None] + selected_t[:, None] * (end - start)[None]
        )
        tangent_pixel = (end - start) * np.asarray(canvas_scale, dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent_pixel))
        if tangent_norm > 1.0e-12:
            grid.tangent[selected_y, selected_x] = (
                tangent_pixel / tangent_norm
            ).astype(np.float32)


def _clip_polygon_to_rect_2d(
    vertices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Clip a convex 2-D polygon to one closed cell rectangle."""

    output = np.asarray(vertices, dtype=np.float64)
    if output.ndim != 2 or output.shape[1] != 2 or len(output) < 3:
        return np.zeros((0, 2), dtype=np.float64)

    def clip_axis(
        polygon: np.ndarray,
        axis: int,
        boundary: float,
        *,
        keep_greater: bool,
    ) -> np.ndarray:
        if len(polygon) == 0:
            return polygon
        values: list[np.ndarray] = []
        previous = polygon[-1]
        previous_inside = bool(
            previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        )
        for current in polygon:
            current_inside = bool(
                current[axis] >= boundary if keep_greater else current[axis] <= boundary
            )
            if current_inside != previous_inside:
                denominator = float(current[axis] - previous[axis])
                if abs(denominator) > np.finfo(np.float64).eps:
                    t = (float(boundary) - float(previous[axis])) / denominator
                    intersection = previous + np.clip(t, 0.0, 1.0) * (
                        current - previous
                    )
                    intersection[axis] = float(boundary)
                    values.append(intersection)
            if current_inside:
                values.append(current.copy())
            previous = current
            previous_inside = current_inside
        if not values:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack(values, axis=0)

    output = clip_axis(output, 0, float(lower[0]), keep_greater=True)
    output = clip_axis(output, 0, float(upper[0]), keep_greater=False)
    output = clip_axis(output, 1, float(lower[1]), keep_greater=True)
    return clip_axis(output, 1, float(upper[1]), keep_greater=False)


def _clean_projected_polygon(
    uv: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove adjacent duplicate/collinear vertices without changing the shape."""

    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or depth.shape != (len(uv),):
        raise ValueError("projected polygon uv/depth shapes differ")
    keep: list[int] = []
    for index, point in enumerate(uv):
        if not keep or float(np.linalg.norm(point - uv[keep[-1]])) > 1.0e-12:
            keep.append(index)
    if len(keep) > 1 and float(np.linalg.norm(uv[keep[0]] - uv[keep[-1]])) <= 1.0e-12:
        keep.pop()
    uv = uv[keep]
    depth = depth[keep]
    changed = True
    while changed and len(uv) > 3:
        changed = False
        for index in range(len(uv)):
            previous = uv[(index - 1) % len(uv)]
            current = uv[index]
            following = uv[(index + 1) % len(uv)]
            first = current - previous
            second = following - current
            cross = float(first[0] * second[1] - first[1] * second[0])
            if abs(cross) <= 1.0e-14:
                uv = np.delete(uv, index, axis=0)
                depth = np.delete(depth, index, axis=0)
                changed = True
                break
    return uv, depth


def _triangulate_polygon(uv: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    """Deterministically ear-clip a simple projected polygon.

    Triangulation is topological, so perform its orientation predicates after
    whitening both principal axes. A valid ground-plane polygon can be
    almost edge-on to the requested camera: in normalized canvas coordinates
    one span may then be O(1) while the other is O(1e-12). Fixed absolute
    cross-product tolerances on the original coordinates incorrectly classify
    every ear as collinear/occupied in that case.

    Returning no triangles is the fail-closed representation for a genuinely
    degenerate or non-simple feature. One malformed map primitive must not
    terminate the complete DataLoader sample.
    """

    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) < 3:
        return ()
    center = uv.mean(axis=0)
    centered = uv - center[None]
    try:
        _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return ()
    extent = max(
        np.finfo(np.float64).tiny,
        float(np.linalg.norm(centered, axis=-1).max(initial=0.0)),
    )
    minimum_span = 64.0 * np.finfo(np.float64).eps * extent
    if (
        len(singular_values) != 2
        or not bool(np.isfinite(singular_values).all())
        or bool((singular_values <= minimum_span).any())
    ):
        return ()
    predicate_uv = (centered @ axes.T) / singular_values[None]
    signed_area = 0.5 * float(
        np.sum(
            predicate_uv[:, 0] * np.roll(predicate_uv[:, 1], -1)
            - predicate_uv[:, 1] * np.roll(predicate_uv[:, 0], -1)
        )
    )
    if abs(signed_area) <= 1.0e-14:
        return ()
    orientation = 1.0 if signed_area > 0.0 else -1.0
    remaining = list(range(len(uv)))
    triangles: list[tuple[int, int, int]] = []

    def cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        first = b - a
        second = c - a
        return float(first[0] * second[1] - first[1] * second[0])

    def inside_triangle(
        point: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> bool:
        values = (
            np.asarray((cross(a, b, point), cross(b, c, point), cross(c, a, point)))
            * orientation
        )
        return bool((values >= -1.0e-13).all())

    while len(remaining) > 3:
        ear_found = False
        for local_index, current in enumerate(remaining):
            previous = remaining[(local_index - 1) % len(remaining)]
            following = remaining[(local_index + 1) % len(remaining)]
            if (
                cross(
                    predicate_uv[previous],
                    predicate_uv[current],
                    predicate_uv[following],
                )
                * orientation
                <= 1.0e-14
            ):
                continue
            if any(
                inside_triangle(
                    predicate_uv[candidate],
                    predicate_uv[previous],
                    predicate_uv[current],
                    predicate_uv[following],
                )
                for candidate in remaining
                if candidate not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[local_index]
            ear_found = True
            break
        if not ear_found:
            return ()
    triangles.append(tuple(remaining))
    return tuple(triangles)


def _nearest_polygon_boundary_geometry(
    query_uv: np.ndarray,
    polygon_uv: np.ndarray,
    canvas_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest boundary UV, unit tangent, and pixel distance per query."""

    query = np.asarray(query_uv, dtype=np.float64)
    polygon = np.asarray(polygon_uv, dtype=np.float64)
    if not len(query):
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
        )
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 2:
        raise ValueError("polygon_uv must contain at least two 2-D vertices")
    scale = np.asarray(canvas_scale, dtype=np.float64).reshape(2)
    if bool((scale <= 0.0).any()) or not bool(np.isfinite(scale).all()):
        raise ValueError("canvas_scale must contain two finite positive values")
    start = polygon * scale
    end = np.roll(polygon, -1, axis=0) * scale
    delta = end - start
    norm2 = np.square(delta).sum(axis=-1)
    safe_norm2 = np.where(norm2 > 1.0e-20, norm2, 1.0)
    query_pixel = query * scale
    relative = query_pixel[:, None, :] - start[None]
    t = np.clip(
        np.sum(relative * delta[None], axis=-1) / safe_norm2[None],
        0.0,
        1.0,
    )
    closest = start[None] + t[..., None] * delta[None]
    squared_distance = np.square(query_pixel[:, None] - closest).sum(axis=-1)
    edge_index = squared_distance.argmin(axis=-1)
    row_index = np.arange(len(query), dtype=np.int64)
    nearest_pixel = closest[row_index, edge_index]
    selected = delta[edge_index]
    norm = np.linalg.norm(selected, axis=-1, keepdims=True)
    tangent = np.divide(
        selected,
        norm,
        out=np.zeros_like(selected),
        where=norm > 1.0e-12,
    ).astype(np.float32)
    return (
        nearest_pixel / scale,
        tangent,
        np.sqrt(squared_distance[row_index, edge_index]),
    )


def _nearest_polygon_edge_tangent(
    query_uv: np.ndarray,
    polygon_uv: np.ndarray,
    canvas_scale: np.ndarray,
) -> np.ndarray:
    """Return the local image-plane boundary tangent nearest each query."""

    _, tangent, _ = _nearest_polygon_boundary_geometry(
        query_uv, polygon_uv, canvas_scale
    )
    return tangent


def _update_polygon_grid(
    grid: _LocalGridGeometry,
    polygon_uv: np.ndarray,
    polygon_depth: np.ndarray,
    triangles: tuple[tuple[int, int, int], ...],
) -> None:
    """Intersect every covered cell with one polygon component and resolve z."""

    cells = np.argwhere(grid.support)
    if not len(cells):
        return
    if not triangles:
        return
    grid_h, grid_w = grid.support.shape
    for triangle_index in triangles:
        triangle = polygon_uv[list(triangle_index)]
        inverse_depth = 1.0 / polygon_depth[list(triangle_index)]
        triangle_center = triangle.mean(axis=0)
        triangle_centered = triangle - triangle_center[None]
        try:
            _, triangle_singular_values, triangle_axes = np.linalg.svd(
                triangle_centered, full_matrices=False
            )
        except np.linalg.LinAlgError:
            continue
        triangle_extent = max(
            np.finfo(np.float64).tiny,
            float(
                np.linalg.norm(triangle_centered, axis=-1).max(initial=0.0)
            ),
        )
        minimum_span = 64.0 * np.finfo(np.float64).eps * triangle_extent
        if (
            len(triangle_singular_values) != 2
            or not bool(np.isfinite(triangle_singular_values).all())
            or bool((triangle_singular_values <= minimum_span).any())
        ):
            continue
        normalized_triangle = (
            triangle_centered @ triangle_axes.T
        ) / triangle_singular_values[None]
        system = np.concatenate(
            (normalized_triangle, np.ones((3, 1), dtype=np.float64)),
            axis=-1,
        )
        try:
            inverse_depth_plane = np.linalg.solve(system, inverse_depth)
        except np.linalg.LinAlgError:
            continue
        min_uv = triangle.min(axis=0)
        max_uv = triangle.max(axis=0)
        candidate = cells[
            (cells[:, 1] + 1 > min_uv[0] * grid_w)
            & (cells[:, 1] < max_uv[0] * grid_w)
            & (cells[:, 0] + 1 > min_uv[1] * grid_h)
            & (cells[:, 0] < max_uv[1] * grid_h)
        ]
        for y, x in candidate:
            lower = np.asarray((x / grid_w, y / grid_h), dtype=np.float64)
            upper = np.asarray(((x + 1) / grid_w, (y + 1) / grid_h), dtype=np.float64)
            intersection = _clip_polygon_to_rect_2d(triangle, lower, upper)
            if len(intersection) < 3:
                continue
            normalized_intersection = (
                (intersection - triangle_center[None]) @ triangle_axes.T
            ) / triangle_singular_values[None]
            inv_z = (
                normalized_intersection @ inverse_depth_plane[:2]
                + inverse_depth_plane[2]
            )
            finite = np.isfinite(inv_z) & (inv_z > 0.0)
            if not bool(finite.any()):
                continue
            finite_indices = np.flatnonzero(finite)
            finite_depth = 1.0 / inv_z[finite_indices]
            minimum_depth = float(finite_depth.min())
            selection_tolerance = 1.0e-10 * max(1.0, abs(minimum_depth))
            tied_indices = finite_indices[
                np.abs(finite_depth - minimum_depth) <= selection_tolerance
            ]
            selected = min(
                (int(value) for value in tied_indices),
                key=lambda value: (
                    float(intersection[value, 0]),
                    float(intersection[value, 1]),
                ),
            )
            depth = 1.0 / float(inv_z[selected])
            candidate_uv = intersection[selected]
            current_depth = float(grid.depth[y, x])
            depth_tolerance = 1.0e-10 * max(
                1.0,
                abs(depth),
                abs(current_depth) if math.isfinite(current_depth) else 1.0,
            )
            strictly_nearer = depth < current_depth - depth_tolerance
            tied = math.isfinite(current_depth) and abs(
                depth - current_depth
            ) <= depth_tolerance
            lexicographically_earlier = tied and (
                float(candidate_uv[0]), float(candidate_uv[1])
            ) < (
                float(grid.uv[y, x, 0]), float(grid.uv[y, x, 1])
            )
            if not strictly_nearer and not lexicographically_earlier:
                continue
            grid.depth[y, x] = depth
            grid.uv[y, x] = candidate_uv


def _rasterize_feature_local_geometry(
    feature: HDMapFeature,
    vertices_camera: np.ndarray,
    projector: _NumpyFrameProjector,
    *,
    raster_hw: tuple[int, int],
    oversample: int,
) -> _FeatureLocalGeometry:
    """Build one clipped local-geometry field shared by early and late M."""

    height, width = (int(value) for value in raster_hw)
    high_h, high_w = height * int(oversample), width * int(oversample)
    mask = np.zeros((high_h, high_w), dtype=np.uint8)
    near = float(projector.near_plane_m)
    far = float(projector.static_far_plane_m)
    point_camera: np.ndarray | None = None
    segment_start = np.zeros((0, 3), dtype=np.float64)
    segment_end = np.zeros((0, 3), dtype=np.float64)
    polygon_components: list[
        tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, int], ...]]
    ] = []
    polygon_boundary_uv = np.zeros((0, 2), dtype=np.float64)
    dropped_primitives = 0

    if feature.geometry == "point" or (
        feature.geometry != "polygon" and len(vertices_camera) == 1
    ):
        point = np.asarray(vertices_camera[0], dtype=np.float64)
        if bool(np.isfinite(point).all()) and near <= float(point[2]) <= far:
            uv = _project_camera_numpy(point[None], projector)[0]
            pixel = uv * np.asarray((high_w, high_h), dtype=np.float64)
            if np.isfinite(pixel).all() and (
                -oversample <= pixel[0] <= high_w + oversample
                and -oversample <= pixel[1] <= high_h + oversample
            ):
                cv2.circle(
                    mask,
                    tuple(np.rint(pixel).astype(np.int32)),
                    max(1, int(oversample) // 2),
                    255,
                    thickness=-1,
                    lineType=cv2.LINE_8,
                )
                point_camera = point
    elif feature.geometry == "polygon":
        # A concave source polygon can split into disconnected components when
        # clipped by the near/far slab. Clipping the complete concave ring can
        # insert bridge edges and yield a self-intersection. Triangulate in
        # stable Waymo world XY first, then clip each convex triangle; their
        # union is exact and never needs a geometry-changing convex hull.
        source_triangles = _triangulate_polygon(
            np.asarray(feature.vertices, dtype=np.float64)[:, :2]
        )
        if not source_triangles and len(feature.vertices) >= 3:
            dropped_primitives += 1
        boundary_camera = _clip_polygon_to_depth_range(
            vertices_camera, near, far
        )
        if len(boundary_camera) >= 3:
            polygon_boundary_uv, _ = _clean_projected_polygon(
                _project_camera_numpy(boundary_camera, projector),
                boundary_camera[:, 2],
            )
        for source_triangle in source_triangles:
            component_camera = _clip_polygon_to_depth_range(
                vertices_camera[list(source_triangle)], near, far
            )
            if len(component_camera) < 3:
                continue
            component_uv, component_depth = _clean_projected_polygon(
                _project_camera_numpy(component_camera, projector),
                component_camera[:, 2],
            )
            component_triangles = _triangulate_polygon(component_uv)
            if not component_triangles:
                dropped_primitives += 1
                continue
            polygon_components.append(
                (component_uv, component_depth, component_triangles)
            )
            pixel = component_uv * np.asarray(
                (high_w, high_h), dtype=np.float64
            )
            pixel[:, 0] = np.clip(pixel[:, 0], -high_w, 2 * high_w)
            pixel[:, 1] = np.clip(pixel[:, 1], -high_h, 2 * high_h)
            if np.isfinite(pixel).all():
                cv2.fillPoly(
                    mask,
                    [np.rint(pixel).astype(np.int32)],
                    255,
                    lineType=cv2.LINE_8,
                )
    else:
        segment_start, segment_end = _clip_polyline_segments_to_depth_range(
            vertices_camera, near, far
        )
        if len(segment_start):
            projected = _project_camera_numpy(
                np.concatenate((segment_start, segment_end), axis=0), projector
            )
            count = len(segment_start)
            scale = np.asarray((high_w, high_h), dtype=np.float64)
            pixel_start, pixel_end, visible = _clip_lines_to_canvas(
                projected[:count] * scale,
                projected[count:] * scale,
                high_w,
                high_h,
            )
            curves = np.rint(
                np.stack((pixel_start[visible], pixel_end[visible]), axis=1)
            ).astype(np.int32)
            if len(curves):
                cv2.polylines(
                    mask,
                    curves,
                    False,
                    255,
                    thickness=max(1, int(oversample)),
                    lineType=cv2.LINE_8,
                )

    raster = _empty_local_grid(_mask_any_downsample(mask, (height, width)))
    patch = _empty_local_grid(_mask_to_patch_support(mask, projector.patch_grid))
    if not bool(mask.any()):
        return _FeatureLocalGeometry(mask, raster, patch, dropped_primitives)

    if point_camera is not None:
        uv = _project_camera_numpy(point_camera[None], projector)[0]
        for grid in (raster, patch):
            grid.depth[grid.support] = float(point_camera[2])
            grid.uv[grid.support] = uv
    elif polygon_components:
        for component_uv, component_depth, component_triangles in polygon_components:
            for grid in (raster, patch):
                _update_polygon_grid(
                    grid,
                    component_uv,
                    component_depth,
                    component_triangles,
                )
        # Resolve tangent only after all components have competed for nearest
        # depth. Use the clipped source-ring boundary so internal source
        # triangulation diagonals never leak into ch10/11.
        if len(polygon_boundary_uv) >= 2:
            for grid in (raster, patch):
                resolved = grid.support & np.isfinite(grid.depth)
                if bool(resolved.any()):
                    grid.tangent[resolved] = _nearest_polygon_edge_tangent(
                        grid.uv[resolved],
                        polygon_boundary_uv,
                        projector.canvas_scale,
                    )
    elif len(segment_start):
        projected = _project_camera_numpy(
            np.concatenate((segment_start, segment_end), axis=0), projector
        )
        count = len(segment_start)
        # The centerline is drawn one raster pixel wide.  Expanding each cell by
        # half that stroke resolves the local point for edge-touching cells
        # without ever falling back to a primitive-global centroid/depth.
        margin_uv = (
            (0.5 * int(oversample) + 1.0) / float(high_w),
            (0.5 * int(oversample) + 1.0) / float(high_h),
        )
        for grid in (raster, patch):
            _update_line_grid(
                grid,
                projected[:count],
                projected[count:],
                segment_start[:, 2],
                segment_end[:, 2],
                canvas_scale=projector.canvas_scale,
                margin_uv=margin_uv,
            )
    return _FeatureLocalGeometry(mask, raster, patch, dropped_primitives)


def _coverage_downsample(mask: np.ndarray, oversample: int) -> np.ndarray:
    height = int(mask.shape[0]) // int(oversample)
    width = int(mask.shape[1]) // int(oversample)
    return (
        mask.reshape(height, oversample, width, oversample)
        .astype(np.float32)
        .mean(axis=(1, 3))
        / 255.0
    )


def _mask_any_downsample(
    mask: np.ndarray,
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Fast exact-positive block reduction for uint8 binary masks."""

    output_h, output_w = (int(value) for value in output_hw)
    if output_h <= 0 or output_w <= 0:
        raise ValueError("output_hw must be positive")
    if mask.shape[0] % output_h or mask.shape[1] % output_w:
        raise ValueError("mask shape must be integer-divisible by output_hw")
    # INTER_AREA over an integer block is its average.  The smallest possible
    # positive input here is 255 / (16*16), which rounds to uint8 value 1, so
    # >0 is exactly equivalent to a blockwise any without a Python-side reduce.
    reduced = cv2.resize(
        mask,
        (output_w, output_h),
        interpolation=cv2.INTER_AREA,
    )
    return reduced > 0


def _mask_to_patch_support(
    mask: np.ndarray,
    patch_grid: tuple[int, int],
) -> np.ndarray:
    gh, gw = patch_grid
    height, width = mask.shape
    if height % gh or width % gw:
        raise ValueError("high-resolution mask is not divisible by patch grid")
    return _mask_any_downsample(mask, (gh, gw))


def _feature_flags(feature: HDMapFeature) -> tuple[float, ...]:
    """Channels 15--20 for an attribute-known feature."""

    flags = [0.0] * 6
    subtype = int(feature.subtype)
    if feature.cls == "road_line" and subtype >= 0:
        flags[3] = float(subtype in (2, 3, 6, 7, 8))
        flags[4] = float(subtype in (4, 5, 6, 7, 8))
    elif feature.cls == "road_edge" and subtype >= 0:
        flags[5] = float(subtype == 2)
    return tuple(flags)


def thin_line_projection_theory(
    scenes: HDMapScene | None | Sequence[HDMapScene | None],
    cam: CameraSpec,
    *,
    static_far_plane_m: float = STATIC_FAR_PLANE_M,
) -> ThinLineProjectionTheory:
    """Count ideal patch-grid traversal of projected thin map polylines.

    The denominator is independent of the 4x raster implementation: each
    near-plane/canvas-clipped road-line or road-edge segment is drawn
    once on its semantic patch-grid plane with unit width.  The companion
    projected length makes the real-data diagnostic auditable instead of
    reporting only an observed nonzero count.
    """

    cam.validate()
    if isinstance(scenes, HDMapScene) or scenes is None:
        scene_rows: list[HDMapScene | None] = [scenes]
    else:
        scene_rows = list(scenes)
    if len(scene_rows) != cam.batch_size:
        raise ValueError("one HDMapScene (or NULL) is required per batch row")
    gh, gw = cam.patch_grid
    expected_patch_count = 0
    projected_length = 0.0
    visible_segment_count = 0
    thin_classes = ("road_line", "road_edge")

    for batch_index, scene in enumerate(scene_rows):
        if scene is None:
            continue
        scene.validate()
        for frame_index in range(cam.num_frames):
            projector = _NumpyFrameProjector.from_camera(
                cam,
                batch_index,
                frame_index,
                static_far_plane_m=float(static_far_plane_m),
            )
            masks = np.zeros((len(thin_classes), gh, gw), dtype=np.uint8)
            for feature in scene.features:
                if feature.cls not in thin_classes:
                    continue
                class_index = thin_classes.index(feature.cls)
                vertices_camera = projector.world_to_camera(feature.vertices)
                if len(vertices_camera) == 1:
                    point = vertices_camera[0]
                    if (
                        bool(np.isfinite(point).all())
                        and projector.near_plane_m
                        <= float(point[2])
                        <= projector.static_far_plane_m
                    ):
                        uv = projector.project(point[None])[0]
                        xy = np.floor(
                            uv * np.asarray((gw, gh), dtype=np.float64)
                        ).astype(np.int64)
                        if 0 <= xy[0] < gw and 0 <= xy[1] < gh:
                            masks[class_index, xy[1], xy[0]] = 1
                    continue
                start, end = _clip_polyline_segments_to_depth_range(
                    vertices_camera,
                    projector.near_plane_m,
                    projector.static_far_plane_m,
                )
                if not len(start):
                    continue
                projected = projector.project(np.concatenate((start, end), axis=0))
                count = len(start)
                scale = np.asarray((gw, gh), dtype=np.float64)
                clipped_start, clipped_end, visible = _clip_lines_to_canvas(
                    projected[:count] * scale,
                    projected[count:] * scale,
                    gw,
                    gh,
                )
                curves = np.rint(
                    np.stack(
                        (clipped_start[visible], clipped_end[visible]),
                        axis=1,
                    )
                ).astype(np.int32)
                if not len(curves):
                    continue
                cv2.polylines(
                    masks[class_index],
                    curves,
                    False,
                    1,
                    thickness=1,
                    lineType=cv2.LINE_8,
                )
                projected_length += float(
                    np.linalg.norm(
                        clipped_end[visible] - clipped_start[visible],
                        axis=-1,
                    ).sum()
                )
                visible_segment_count += int(visible.sum())
            expected_patch_count += int(np.count_nonzero(masks))
    return ThinLineProjectionTheory(
        expected_patch_count=expected_patch_count,
        projected_length_patch_units=projected_length,
        visible_segment_count=visible_segment_count,
    )


def _static_frame_raster(
    scene: HDMapScene | None,
    cam: CameraSpec,
    batch_index: int,
    frame_index: int,
    *,
    raster_hw: tuple[int, int],
    oversample: int,
    static_far_plane_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    height, width = raster_hw
    gh, gw = cam.patch_grid
    raster = np.zeros((RASTER_CHANNEL_COUNT, height, width), dtype=np.float32)
    map_metric = np.zeros((gh * gw, LAYOUT_MAP_GROUPS, 4), dtype=np.float32)
    dropped_primitives = 0
    if scene is None or not scene.features:
        return raster, map_metric, dropped_primitives

    high_h, high_w = height * oversample, width * oversample
    class_masks = np.zeros((7, high_h, high_w), dtype=np.uint8)
    nearest_depth = np.full((height, width), np.inf, dtype=np.float64)
    nearest_metric_depth = np.full(
        (gh * gw, LAYOUT_MAP_GROUPS), np.inf, dtype=np.float64
    )
    attribute_known = scene.attribute_source != ATTRIBUTE_SOURCE_NONE
    projector = _NumpyFrameProjector.from_camera(
        cam,
        batch_index,
        frame_index,
        static_far_plane_m=float(static_far_plane_m),
    )

    for feature in scene.features:
        if feature.cls in RASTER_SOURCE_EXCLUDED_CLASSES:
            continue
        vertices_camera = _world_to_camera_numpy(
            feature.vertices,
            projector,
        )
        local = _rasterize_feature_local_geometry(
            feature,
            vertices_camera,
            projector,
            raster_hw=raster_hw,
            oversample=oversample,
        )
        dropped_primitives += int(local.dropped_primitives)
        if not bool(local.high_mask.any()):
            continue
        class_index = CLASS_IDS[feature.cls]
        cv2.bitwise_or(
            class_masks[class_index],
            local.high_mask,
            dst=class_masks[class_index],
        )
        local_raster_valid = local.raster.support & np.isfinite(local.raster.depth)
        closer = local_raster_valid & (local.raster.depth < nearest_depth)
        if bool(closer.any()):
            raster[10, closer] = local.raster.tangent[..., 0][closer]
            raster[11, closer] = local.raster.tangent[..., 1][closer]
            center_patch = local.raster.uv * np.asarray((gw, gh))
            subpatch = center_patch - np.floor(center_patch)
            raster[12, closer] = np.clip(subpatch[..., 0][closer], 0.0, 1.0)
            raster[13, closer] = np.clip(subpatch[..., 1][closer], 0.0, 1.0)
            raster[14, closer] = float(attribute_known)
            if attribute_known:
                for offset, value in enumerate(_feature_flags(feature)):
                    raster[15 + offset, closer] = value
            nearest_depth[closer] = local.raster.depth[closer]

        group = MAP_GROUP_BY_CLASS[feature.cls]
        patch_depth = local.patch.depth.reshape(-1)
        patch_uv = local.patch.uv.reshape(-1, 2)
        patch_valid = local.patch.support.reshape(-1) & np.isfinite(patch_depth)
        patch_closer = patch_valid & (patch_depth < nearest_metric_depth[:, group])
        patch_indices = np.flatnonzero(patch_closer)
        if len(patch_indices):
            map_metric[patch_indices, group, 0:2] = patch_uv[patch_indices]
            map_metric[patch_indices, group, 2] = np.log(
                np.maximum(patch_depth[patch_indices], float(cam.near_plane_m))
            )
            map_metric[patch_indices, group, 3] = 1.0
            nearest_metric_depth[patch_indices, group] = patch_depth[patch_indices]

    # ch0 is a frozen zero slot: LaneCenter is source-only and excluded from M.
    for class_index in range(1, 7):
        raster[class_index] = _coverage_downsample(class_masks[class_index], oversample)
    # ch7 is the matching frozen-zero distance slot.
    for channel, class_index in zip((8, 9), (1, 2)):
        foreground = raster[class_index] > 0.0
        if bool(foreground.any()):
            # Four raster pixels form one 25x37 patch; normalize a distance of
            # eight patches to one.
            distance_patch = distance_transform_edt(~foreground) / 4.0
            raster[channel] = np.minimum(distance_patch, 8.0) / 8.0
    raster[RASTER_STATIC_VALID_CHANNEL] = np.maximum.reduce(raster[:7]) > 0.0
    return raster, map_metric, dropped_primitives


def _actor_frame_raster(
    g: ActorGeometryCondition,
    projected: ProjectedActorGeometry,
    cam: CameraSpec,
    batch_index: int,
    frame_index: int,
    *,
    raster_hw: tuple[int, int],
    oversample: int,
    layout_max_actors: int,
) -> np.ndarray:
    height, width = (int(value) for value in raster_hw)
    gh, gw = (int(value) for value in cam.patch_grid)
    if height != gh * 4 or width != gw * 4:
        raise ValueError("actor raster must be exactly 4x the patch grid")
    high_h, high_w = height * int(oversample), width * int(oversample)
    class_masks = np.zeros((3, high_h, high_w), dtype=np.uint8)
    edge_mask = np.zeros((high_h, high_w), dtype=np.uint8)
    actor_masks: list[tuple[int, np.ndarray, np.ndarray]] = []
    fixed_shift = 8
    fixed_scale = float(1 << fixed_shift)

    for slot in range(g.num_slots):
        # frame_support is the semantic gate.  In particular, no padded
        # log-depth value may be exponentiated before this check.
        if not bool(projected.frame_support[batch_index, slot, frame_index]):
            continue
        class_id = int(g.class_id[batch_index, slot])
        if class_id not in (0, 1, 2):
            continue
        vertex_valid = projected.silhouette_vertex_valid[batch_index, slot, frame_index]
        polygon_uv = (
            projected.silhouette_uv[batch_index, slot, frame_index][vertex_valid]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if len(polygon_uv) < 3 or not bool(np.isfinite(polygon_uv).all()):
            raise ValueError("supported actor requires a finite silhouette polygon")

        # The projected cache owns the clipped CCW silhouette.  Draw that
        # polygon itself--never its axis-aligned bbox--with subpixel vertices.
        polygon_pixel = polygon_uv * np.asarray(
            (max(1, high_w - 1), max(1, high_h - 1)), dtype=np.float64
        )
        fixed_polygon = np.rint(polygon_pixel * fixed_scale).astype(np.int32)
        mask = np.zeros((high_h, high_w), dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [fixed_polygon],
            255,
            lineType=cv2.LINE_8,
            shift=fixed_shift,
        )
        if not bool(mask.any()):
            raise RuntimeError(
                "supported actor silhouette vanished during rasterization"
            )
        cv2.polylines(
            edge_mask,
            [fixed_polygon],
            True,
            255,
            thickness=1,
            lineType=cv2.LINE_8,
            shift=fixed_shift,
        )
        cv2.bitwise_or(
            class_masks[class_id],
            mask,
            dst=class_masks[class_id],
        )
        actor_masks.append((slot, mask, polygon_uv))

    raster = np.zeros((11, height, width), dtype=np.float32)
    for class_id in range(3):
        raster[class_id] = _coverage_downsample(class_masks[class_id], oversample)
    union = (
        np.maximum.reduce(class_masks)
        if actor_masks
        else np.zeros((high_h, high_w), dtype=np.uint8)
    )
    foreground = _coverage_downsample(union, oversample) > 0.0
    if bool(edge_mask.any()):
        # ch25 is unsigned distance to every actual cuboid silhouette edge.
        # Unlike EDT(~filled_union), this is positive both inside and outside
        # a box and preserves the internal outlines of overlapping actors.
        distance_high = distance_transform_edt(edge_mask == 0)
        distance_raster = distance_high.reshape(
            height, oversample, width, oversample
        ).min(axis=(1, 3))
        distance_patch = distance_raster / float(4 * int(oversample))
        raster[3] = np.minimum(distance_patch, 8.0) / 8.0

    nearest_depth = np.full((height, width), np.inf, dtype=np.float64)
    count_high = np.zeros((high_h, high_w), dtype=np.float32)
    canvas_k = (
        cam.canvas_intrinsics[batch_index, frame_index].cpu().numpy().astype(np.float64)
    )
    inverse_canvas_k = np.linalg.inv(canvas_k)
    canvas_scale = np.asarray((cam.canvas_hw[1], cam.canvas_hw[0]), dtype=np.float64)
    near = float(cam.near_plane_m)
    for slot, mask, polygon_uv in actor_masks:
        count_high += mask.astype(np.float32) / 255.0
        low_mask = _mask_any_downsample(mask, (height, width))

        patch_support = (
            projected.patch_support[batch_index, slot, frame_index].cpu().numpy()
        )
        log_z_patch = (
            projected.log_z_patch[batch_index, slot, frame_index]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        patch_depth = np.full((gh * gw,), np.inf, dtype=np.float64)
        # This ordering is deliberate: the fail-closed padding log(1 m)=0 is
        # never read as geometry because only explicitly supported entries are
        # exponentiated.
        patch_depth[patch_support] = np.exp(log_z_patch[patch_support])
        active_depth = patch_depth[patch_support]
        if len(active_depth) and (
            not bool(np.isfinite(active_depth).all())
            or bool((active_depth < near - 1.0e-5).any())
            or bool((active_depth > ACTOR_FAR_PLANE_M + 1.0e-4).any())
        ):
            raise ValueError("supported actor patch depth is outside [near, 120] m")
        support_grid = patch_support.reshape(gh, gw)
        depth_grid = patch_depth.reshape(gh, gw)
        support_raster = np.repeat(np.repeat(support_grid, 4, axis=0), 4, axis=1)
        depth_raster = np.repeat(np.repeat(depth_grid, 4, axis=0), 4, axis=1)
        candidate = low_mask & support_raster & np.isfinite(depth_raster)
        closer = candidate & (depth_raster < nearest_depth)
        if not bool(closer.any()):
            continue

        cells = np.argwhere(closer)
        cell_y = cells[:, 0]
        cell_x = cells[:, 1]
        query_uv = np.stack(
            (
                (cell_x.astype(np.float64) + 0.5) / float(width),
                (cell_y.astype(np.float64) + 0.5) / float(height),
            ),
            axis=-1,
        )
        boundary_uv, _, _ = _nearest_polygon_boundary_geometry(
            query_uv, polygon_uv, canvas_scale
        )
        boundary_patch = boundary_uv * np.asarray((gw, gh), dtype=np.float64)
        subpatch = boundary_patch - np.floor(boundary_patch)
        raster[4, cell_y, cell_x] = np.clip(subpatch[:, 0], 0.0, 1.0)
        raster[5, cell_y, cell_x] = np.clip(subpatch[:, 1], 0.0, 1.0)

        # Re-read image-plane velocity on the same local ray and per-patch
        # surface depth that selected this actor.  This avoids a global centre
        # proxy whose projection can lie in another patch entirely.
        query_pixel = query_uv * canvas_scale
        homogeneous = np.concatenate(
            (query_pixel, np.ones((len(query_pixel), 1), dtype=np.float64)),
            axis=-1,
        )
        ray = homogeneous @ inverse_canvas_k.T
        ray_valid = np.isfinite(ray).all(axis=-1) & (np.abs(ray[:, 2]) > 1.0e-12)
        ray = np.divide(
            ray,
            ray[:, 2:3],
            out=np.zeros_like(ray),
            where=np.abs(ray[:, 2:3]) > 1.0e-12,
        )
        local_depth = depth_raster[cell_y, cell_x]
        local_camera = ray * local_depth[:, None]
        velocity = (
            projected.velocity_camera[batch_index, slot, frame_index]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        endpoint = local_camera + velocity[None]
        endpoint_h = endpoint @ canvas_k.T
        endpoint_valid = (
            ray_valid & np.isfinite(endpoint_h).all(axis=-1) & (endpoint_h[:, 2] > near)
        )
        endpoint_pixel = np.divide(
            endpoint_h[:, :2],
            endpoint_h[:, 2:3],
            out=np.zeros_like(endpoint_h[:, :2]),
            where=endpoint_valid[:, None],
        )
        delta = endpoint_pixel - query_pixel
        norm = np.linalg.norm(delta, axis=-1)
        direction_valid = endpoint_valid & (norm > 1.0e-9)
        direction = np.divide(
            delta,
            norm[:, None],
            out=np.zeros_like(delta),
            where=direction_valid[:, None],
        ).astype(np.float32)
        if float(np.linalg.norm(velocity)) <= 1.0e-6:
            direction.fill(0.0)
        raster[6, cell_y, cell_x] = direction[:, 0]
        raster[7, cell_y, cell_x] = direction[:, 1]
        raster[8, cell_y, cell_x] = float(g.is_moving[batch_index, slot, frame_index])
        nearest_depth[cell_y, cell_x] = local_depth

    raster[9] = np.clip(
        count_high.reshape(height, oversample, width, oversample).mean(axis=(1, 3))
        / float(max(1, layout_max_actors)),
        0.0,
        1.0,
    )
    raster[10] = foreground.astype(np.float32)
    return raster


def project_layout(
    scenes: HDMapScene | None | Sequence[HDMapScene | None],
    g: ActorGeometryCondition,
    cam: CameraSpec,
    *,
    raster_hw: tuple[int, int] = LAYOUT_RASTER_HW,
    oversample: int = LAYOUT_OVERSAMPLE,
    layout_max_actors: int | None = None,
    static_far_plane_m: float = STATIC_FAR_PLANE_M,
) -> LayoutProjection:
    """Atomically project M and G for the requested camera.

    This is a pure, deterministic online operation.  Call it again after any
    edit to ``C`` or ``G``; never splice a new actor cache into an old map
    raster (or vice versa).
    """

    cam.validate()
    g.validate(layout_max_actors=layout_max_actors)
    if g.batch_size != cam.batch_size or g.num_frames != cam.num_frames:
        raise ValueError("G and C batch/frame dimensions differ")
    if cam.world_to_anchor.device.type != "cpu":
        raise ValueError("online CPU rasterization requires a CPU CameraSpec")
    height, width = (int(v) for v in raster_hw)
    gh, gw = cam.patch_grid
    if (height, width) != (gh * 4, gw * 4):
        raise ValueError("raster_hw must be exactly 4x the patch grid")
    if int(oversample) != LAYOUT_OVERSAMPLE:
        raise ValueError("the frozen raster contract requires 4x supersampling")
    if float(static_far_plane_m) != STATIC_FAR_PLANE_M:
        raise ValueError(
            f"static_far_plane_m is frozen to {STATIC_FAR_PLANE_M:g} metres"
        )
    max_actors = g.num_slots if layout_max_actors is None else int(layout_max_actors)

    if isinstance(scenes, HDMapScene) or scenes is None:
        scene_rows: list[HDMapScene | None] = [scenes]
    else:
        scene_rows = list(scenes)
    if len(scene_rows) != cam.batch_size:
        raise ValueError("one HDMapScene (or NULL) is required per batch row")
    for scene in scene_rows:
        if scene is not None:
            scene.validate()

    projected_actor = project_actor_geometry(g, cam)
    float_raster = np.zeros(
        (cam.batch_size, cam.num_frames, RASTER_CHANNEL_COUNT, height, width),
        dtype=np.float32,
    )
    metric = np.zeros(
        (
            cam.batch_size,
            cam.num_frames,
            gh * gw,
            LAYOUT_MAP_GROUPS,
            4,
        ),
        dtype=np.float32,
    )
    map_mode = torch.empty((cam.batch_size,), dtype=torch.int8)
    dropped_primitives = 0
    for batch_index, scene in enumerate(scene_rows):
        for frame_index in range(cam.num_frames):
            static, frame_metric, frame_dropped = _static_frame_raster(
                scene,
                cam,
                batch_index,
                frame_index,
                raster_hw=(height, width),
                oversample=int(oversample),
                static_far_plane_m=float(static_far_plane_m),
            )
            dynamic = _actor_frame_raster(
                g,
                projected_actor,
                cam,
                batch_index,
                frame_index,
                raster_hw=(height, width),
                oversample=int(oversample),
                layout_max_actors=max_actors,
            )
            dropped_primitives += int(frame_dropped)
            map_lo, map_hi = RASTER_MAP_CHANNELS
            actor_lo, actor_hi = RASTER_ACTOR_CHANNELS
            float_raster[batch_index, frame_index, map_lo:map_hi] = static[map_lo:map_hi]
            float_raster[batch_index, frame_index, actor_lo:actor_hi] = dynamic
            metric[batch_index, frame_index] = frame_metric
        map_mode[batch_index] = int(
            MapMode.NULL
            if scene is None
            else (
                MapMode.PRESENT
                if bool(
                    float_raster[batch_index, :, RASTER_STATIC_VALID_CHANNEL].any()
                )
                else MapMode.EMPTY
            )
        )

    map_layout = ProjectedMapLayout(
        layout_raster=quantize_layout_raster(float_raster),
        map_metric=torch.from_numpy(metric),
        map_mode=map_mode,
        raster_schema_hash=RASTER_SCHEMA_HASH,
        static_far_plane_m=float(static_far_plane_m),
        camera_snapshot=CameraProjectionSnapshot.from_camera(cam),
        dropped_primitives=int(dropped_primitives),
    )
    return LayoutProjection(map_layout=map_layout, actor_geometry=projected_actor)


def assert_projection_consistent(
    projection: LayoutProjection,
    scenes: HDMapScene | None | Sequence[HDMapScene | None],
    g: ActorGeometryCondition,
    cam: CameraSpec,
) -> None:
    """Expensive validation hook used at edit/test boundaries (T29)."""

    projection.assert_current(cam)
    expected = project_layout(
        scenes,
        g,
        cam,
        raster_hw=tuple(int(v) for v in projection.map_layout.layout_raster.shape[-2:]),
        layout_max_actors=g.num_slots,
        static_far_plane_m=projection.map_layout.static_far_plane_m,
    )
    if not torch.equal(
        projection.map_layout.layout_raster, expected.map_layout.layout_raster
    ) or not torch.equal(
        projection.map_layout.map_metric, expected.map_layout.map_metric
    ):
        raise ValueError("stale map projection: M was not recomputed from requested C")
    for name in projection.actor_geometry.__dataclass_fields__:
        if not torch.equal(
            getattr(projection.actor_geometry, name),
            getattr(expected.actor_geometry, name),
        ):
            raise ValueError(
                "stale actor projection: G was not recomputed from requested C"
            )


__all__ = [
    "CameraProjectionSnapshot",
    "GaugeRereadResult",
    "LAYOUT_MAP_GROUPS",
    "LAYOUT_OVERSAMPLE",
    "LAYOUT_RASTER_HW",
    "STATIC_FAR_PLANE_M",
    "LayoutProjection",
    "MapMode",
    "ProjectedMapLayout",
    "ThinLineProjectionTheory",
    "assert_projection_consistent",
    "build_map_metric_features",
    "clip_segment_to_depth_range",
    "dequantize_layout_raster",
    "project_layout",
    "quantize_layout_raster",
    "reread_metric_geometry",
    "scale_gradient",
    "thin_line_projection_theory",
]
