from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from dggt.utils.gaussian_edit import (
    CleanSceneState,
    build_box_corners,
    compute_bbox_from_projected_points,
    points_in_box,
    project_world_points,
)
from dggt.utils.ground_plane import estimate_ground_plane_per_frame


@dataclass
class ImaginedObject:
    slot: int
    motion_mode: str
    size_dggt: torch.Tensor
    center_dggt_per_frame: torch.Tensor
    yaw_dggt_per_frame: torch.Tensor
    visible_in_frame_per_view: torch.Tensor
    bbox_2d_per_view: torch.Tensor
    semantic_overlap_px: int = 0
    existing_box_iou_3d: float = 0.0
    occluder_profile: dict[str, float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": int(self.slot),
            "motion_mode": str(self.motion_mode),
            "size_dggt": [float(v) for v in self.size_dggt.detach().cpu().tolist()],
            "center_dggt_per_frame": [
                [float(v) for v in row] for row in self.center_dggt_per_frame.detach().cpu().tolist()
            ],
            "yaw_dggt_per_frame": [float(v) for v in self.yaw_dggt_per_frame.detach().cpu().tolist()],
            "visible_in_frame_per_view": self.visible_in_frame_per_view.detach().cpu().bool().tolist(),
            "bbox_2d_per_view": self.bbox_2d_per_view.detach().cpu().float().tolist(),
            "semantic_overlap_px": int(self.semantic_overlap_px),
            "existing_box_iou_3d": float(self.existing_box_iou_3d),
            "occluder_profile": dict(self.occluder_profile),
        }


@dataclass
class ModeBPlan:
    scene_name: str
    clip_name: str
    clip_index: int
    views: int
    imagined_objects: list[ImaginedObject]
    rng_seed: int
    eligible: bool
    rejection_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def num_imagined_objects(self) -> int:
        return len(self.imagined_objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_name": str(self.scene_name),
            "clip_name": str(self.clip_name),
            "clip_index": int(self.clip_index),
            "views": int(self.views),
            "num_imagined_objects": int(self.num_imagined_objects),
            "imagined_objects": [obj.to_dict() for obj in self.imagined_objects],
            "rng_seed": int(self.rng_seed),
            "eligible": bool(self.eligible),
            "rejection_reason": str(self.rejection_reason),
            "metrics": self.metrics,
        }


@dataclass
class ModeBDeletionResult:
    delete_mask: torch.Tensor
    shell_mask: torch.Tensor
    delete_mask_per_frame: torch.Tensor
    delete_core_indices: torch.Tensor
    delete_shell_indices: torch.Tensor

    def to_payload(self) -> dict[str, torch.Tensor]:
        return {
            "delete_mask": self.delete_mask,
            "shell_mask": self.shell_mask,
            "delete_mask_per_frame": self.delete_mask_per_frame,
            "delete_core_indices": self.delete_core_indices,
            "delete_shell_indices": self.delete_shell_indices,
        }


def _as_float_tensor(value: Any, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    if shape is not None:
        tensor = tensor.reshape(shape)
    return tensor.contiguous()


def _rotation_from_yaw(yaw: float, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Local axes are [length/front, width/left, height/up]. DGGT camera up is -Y.
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    front = torch.tensor([s, 0.0, c], dtype=dtype)
    up = torch.tensor([0.0, -1.0, 0.0], dtype=dtype)
    left = torch.linalg.cross(up, front)
    left = left / left.norm().clamp_min(1e-6)
    return torch.stack([front, left, up], dim=1)


def _yaw_from_forward(forward: torch.Tensor) -> float:
    forward = forward.detach().cpu().float()
    x = float(forward[0].item())
    z = float(forward[2].item())
    if abs(x) + abs(z) < 1e-6:
        return 0.0
    return math.atan2(x, z)


def _center_y_from_ground(ground_y: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return ground_y.to(dtype=torch.float32) - 0.5 * size.detach().cpu().float()[2]


def _infer_num_views(clean_state: CleanSceneState, explicit_views: int | None = None) -> int:
    if explicit_views is not None:
        views = int(explicit_views)
        if views <= 0:
            raise ValueError(f"views must be positive, got {views}")
        return views
    if clean_state.source_view_ids.numel() > 0:
        return int(clean_state.source_view_ids.max().item()) + 1
    return 1


def _infer_num_frames(clean_state: CleanSceneState, num_views: int) -> int:
    if clean_state.source_frame_ids.numel() > 0:
        return int(clean_state.source_frame_ids.max().item()) + 1
    return int(math.ceil(float(clean_state.images.shape[0]) / float(max(num_views, 1))))


def _image_index(frame_idx: int, view_idx: int, num_views: int, num_images: int) -> int | None:
    idx = int(frame_idx) * int(num_views) + int(view_idx)
    if 0 <= idx < int(num_images):
        return idx
    return None


def _convex_hull(points: torch.Tensor) -> list[tuple[float, float]]:
    pts = sorted({(float(x), float(y)) for x, y in points.detach().cpu().float().tolist()})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_to_mask(points: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    height, width = image_hw
    if points.numel() == 0:
        return torch.zeros((height, width), dtype=torch.bool)
    hull = _convex_hull(points)
    if len(hull) < 3:
        return torch.zeros((height, width), dtype=torch.bool)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(hull, fill=1)
    return torch.from_numpy(np.array(mask, dtype=np.uint8, copy=True)).bool()


def _aabb_from_corners(corners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return corners.min(dim=0).values, corners.max(dim=0).values


def _aabb_iou(min_a: torch.Tensor, max_a: torch.Tensor, min_b: torch.Tensor, max_b: torch.Tensor) -> float:
    inter_min = torch.maximum(min_a, min_b)
    inter_max = torch.minimum(max_a, max_b)
    inter_size = (inter_max - inter_min).clamp_min(0.0)
    inter_vol = float(inter_size.prod().item())
    if inter_vol <= 0.0:
        return 0.0
    vol_a = float((max_a - min_a).clamp_min(0.0).prod().item())
    vol_b = float((max_b - min_b).clamp_min(0.0).prod().item())
    denom = vol_a + vol_b - inter_vol
    if denom <= 0.0:
        return 0.0
    return inter_vol / denom


def _existing_box_at_frame(obj: dict[str, Any], frame_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    present_mask = obj.get("present_mask")
    if present_mask is not None:
        present_values = list(present_mask)
        if frame_idx >= len(present_values) or not bool(present_values[frame_idx]):
            return None

    center_value = None
    for key in ("center_dggt_per_frame", "center_per_frame"):
        if key in obj:
            seq = obj[key]
            if frame_idx < len(seq):
                center_value = seq[frame_idx]
            break
    if center_value is None:
        for key in ("center_dggt", "refined_center", "center"):
            if key in obj:
                center_value = obj[key]
                break
    if center_value is None:
        return None

    size_value = None
    for key in ("size_dggt_per_frame", "size_per_frame", "box_size_dggt_per_frame"):
        if key in obj:
            seq = obj[key]
            if frame_idx < len(seq):
                size_value = seq[frame_idx]
            break
    for key in ("size_dggt", "refined_size", "size", "box_size_dggt"):
        if size_value is None and key in obj:
            size_value = obj[key]
            break
    if size_value is None:
        return None

    rotation_value = None
    for key in ("rotation_dggt_per_frame", "rotation_per_frame"):
        if key in obj:
            seq = obj[key]
            if frame_idx < len(seq):
                rotation_value = seq[frame_idx]
            break
    if rotation_value is None:
        for key in ("rotation_dggt", "refined_rotation", "rotation"):
            if key in obj:
                rotation_value = obj[key]
                break

    if rotation_value is not None:
        rotation = _as_float_tensor(rotation_value, shape=(3, 3))
    else:
        yaw_value = None
        for key in ("yaw_dggt_per_frame", "yaw_per_frame"):
            if key in obj:
                seq = obj[key]
                if frame_idx < len(seq):
                    yaw_value = seq[frame_idx]
                break
        if yaw_value is None:
            yaw_value = obj.get("yaw_dggt", obj.get("yaw", 0.0))
        rotation = _rotation_from_yaw(float(yaw_value))
    return _as_float_tensor(center_value, shape=(3,)), _as_float_tensor(size_value, shape=(3,)), rotation


class ModeBPlanner:
    def __init__(
        self,
        *,
        min_visible_frames: int = 15,
        max_semantic_overlap_px: int = 0,
        max_trials_per_object: int = 80,
        canonical_size: tuple[float, float, float] = (4.05, 1.71, 1.44),
        canonical_size_jitter: float = 0.15,
        yaw_jitter_deg: float = 30.0,
        motion_probs: tuple[float, float, float] = (0.5, 0.3, 0.2),
        core_scale: float = 1.0,
        shell_scale: float = 1.05,
        min_projected_area_px: float = 64.0,
        min_projected_transfer_size_px: float = 128.0,
        transfer_image_hw: tuple[int, int] = (704, 1280),
        max_projected_area_ratio: float = 0.12,
        max_projected_width_ratio: float = 0.45,
        max_projected_height_ratio: float = 0.52,
        min_projected_top_y_ratio: float = 0.20,
        min_projected_center_y_ratio: float = 0.35,
        min_projected_bottom_y_ratio: float = 0.50,
        max_projected_bottom_y_ratio: float = 0.92,
        min_ground_support_ratio: float = 0.18,
        require_first_frame_visible: bool = False,
        fast_camera_step_ratio: float = 0.018,
        slow_camera_step_ratio: float = 0.006,
        rng_seed: int = 0,
    ) -> None:
        self.min_visible_frames = int(min_visible_frames)
        self.max_semantic_overlap_px = int(max_semantic_overlap_px)
        self.max_trials_per_object = int(max_trials_per_object)
        self.canonical_size = torch.tensor(canonical_size, dtype=torch.float32)
        self.canonical_size_jitter = float(canonical_size_jitter)
        self.yaw_jitter_deg = float(yaw_jitter_deg)
        self.motion_probs = tuple(float(v) for v in motion_probs)
        self.core_scale = float(core_scale)
        self.shell_scale = float(shell_scale)
        self.min_projected_area_px = float(min_projected_area_px)
        self.min_projected_transfer_size_px = float(min_projected_transfer_size_px)
        self.transfer_image_hw = tuple(int(v) for v in transfer_image_hw)
        self.max_projected_area_ratio = float(max_projected_area_ratio)
        self.max_projected_width_ratio = float(max_projected_width_ratio)
        self.max_projected_height_ratio = float(max_projected_height_ratio)
        self.min_projected_top_y_ratio = float(min_projected_top_y_ratio)
        self.min_projected_center_y_ratio = float(min_projected_center_y_ratio)
        self.min_projected_bottom_y_ratio = float(min_projected_bottom_y_ratio)
        self.max_projected_bottom_y_ratio = float(max_projected_bottom_y_ratio)
        self.min_ground_support_ratio = float(min_ground_support_ratio)
        self.require_first_frame_visible = bool(require_first_frame_visible)
        self.fast_camera_step_ratio = float(fast_camera_step_ratio)
        self.slow_camera_step_ratio = float(slow_camera_step_ratio)
        self.rng_seed = int(rng_seed)
        self.rng = random.Random(self.rng_seed)
        self._scene_point_candidate_cache: dict[int, torch.Tensor] = {}

    def _sample_target_count(self, views: int) -> int:
        if int(views) == 1:
            return self.rng.choice([1, 2, 3])
        return self.rng.choice([3, 4, 5])

    def _sample_motion_mode(self, camera_motion_level: str = "medium") -> str:
        modes = ("static", "slow", "ego_matched")
        if camera_motion_level == "fast":
            probs = (0.08, 0.22, 0.70)
        elif camera_motion_level == "slow":
            probs = (0.55, 0.35, 0.10)
        else:
            probs = self.motion_probs
        total = sum(probs)
        if total <= 0.0:
            return "static"
        r = self.rng.random() * total
        acc = 0.0
        for mode, prob in zip(modes, probs):
            acc += float(prob)
            if r <= acc:
                return mode
        return modes[-1]

    def _camera_motion_metrics(self, clean_state: CleanSceneState, num_frames: int, num_views: int) -> dict[str, float | str]:
        ego = []
        for frame_idx in range(max(int(num_frames), 1)):
            image_idx = _image_index(frame_idx, 0, num_views, clean_state.camera_to_world.shape[0])
            if image_idx is None:
                image_idx = min(frame_idx, clean_state.camera_to_world.shape[0] - 1)
            ego.append(clean_state.camera_to_world[image_idx, :3, 3].detach().cpu().float())
        if len(ego) < 2:
            return {"median_step": 0.0, "total": 0.0, "level": "slow"}
        ego_pose = torch.stack(ego, dim=0)
        steps = torch.norm(ego_pose[1:] - ego_pose[:-1], dim=-1)
        median_step = float(torch.median(steps).item()) if steps.numel() > 0 else 0.0
        total = float(torch.norm(ego_pose[-1] - ego_pose[0]).item())
        if median_step >= self.fast_camera_step_ratio:
            level = "fast"
        elif median_step <= self.slow_camera_step_ratio:
            level = "slow"
        else:
            level = "medium"
        return {"median_step": median_step, "total": total, "level": level}

    def _sample_occluder_profile(self) -> dict[str, float | str]:
        variant = self.rng.choices(("wide", "balanced", "tall"), weights=(0.45, 0.35, 0.20), k=1)[0]
        if variant == "wide":
            profile: dict[str, float | str] = {
                "variant": variant,
                "bbox_width_mult": self.rng.uniform(2.205, 2.790),
                "bbox_height_mult": self.rng.uniform(1.485, 1.845),
                "shape_width_scale": self.rng.uniform(1.00, 1.08),
                "shape_height_scale": self.rng.uniform(0.92, 1.00),
                "shape_center_y": self.rng.uniform(0.48, 0.52),
                "top_taper": self.rng.uniform(0.16, 0.25),
                "bottom_boost": self.rng.uniform(0.04, 0.10),
                "exponent": self.rng.uniform(3.0, 3.8),
            }
        elif variant == "tall":
            profile = {
                "variant": variant,
                "bbox_width_mult": self.rng.uniform(1.755, 2.205),
                "bbox_height_mult": self.rng.uniform(1.890, 2.295),
                "shape_width_scale": self.rng.uniform(0.88, 0.98),
                "shape_height_scale": self.rng.uniform(1.00, 1.08),
                "shape_center_y": self.rng.uniform(0.46, 0.50),
                "top_taper": self.rng.uniform(0.10, 0.20),
                "bottom_boost": self.rng.uniform(0.02, 0.08),
                "exponent": self.rng.uniform(2.6, 3.4),
            }
        else:
            profile = {
                "variant": variant,
                "bbox_width_mult": self.rng.uniform(1.980, 2.475),
                "bbox_height_mult": self.rng.uniform(1.665, 2.025),
                "shape_width_scale": self.rng.uniform(0.96, 1.04),
                "shape_height_scale": self.rng.uniform(0.96, 1.04),
                "shape_center_y": self.rng.uniform(0.47, 0.51),
                "top_taper": self.rng.uniform(0.14, 0.24),
                "bottom_boost": self.rng.uniform(0.03, 0.09),
                "exponent": self.rng.uniform(2.8, 3.6),
            }
        return profile

    def _sample_size(self, existing_objects: list[dict[str, Any]]) -> torch.Tensor:
        sizes = []
        for obj in existing_objects:
            for key in ("size_dggt_per_frame", "size_per_frame", "box_size_dggt_per_frame"):
                if key not in obj:
                    continue
                try:
                    seq = _as_float_tensor(obj[key]).view(-1, 3)
                    valid = (seq > 0).all(dim=1)
                    if valid.any():
                        sizes.append(seq[valid].mean(dim=0))
                except Exception:
                    pass
                break
            for key in ("size_dggt", "refined_size", "size", "box_size_dggt"):
                if key in obj:
                    try:
                        sizes.append(_as_float_tensor(obj[key], shape=(3,)))
                    except Exception:
                        pass
                    break
        base = torch.stack(sizes).mean(dim=0) if sizes else self.canonical_size.clone()
        lo = max(0.0, 1.0 - self.canonical_size_jitter)
        hi = 1.0 + self.canonical_size_jitter
        jitter = torch.tensor([self.rng.uniform(lo, hi) for _ in range(3)], dtype=torch.float32)
        return (base * jitter).clamp_min(1e-3)

    def _sample_base_center(
        self,
        clean_state: CleanSceneState,
        size: torch.Tensor,
        ground_y: torch.Tensor,
        num_frames: int,
        num_views: int,
        view_hint: int,
    ) -> tuple[torch.Tensor, int, int]:
        frame_idx = 0
        view_idx = int(view_hint) % max(num_views, 1)
        image_idx = _image_index(frame_idx, view_idx, num_views, clean_state.camera_to_world.shape[0])
        if image_idx is None:
            image_idx = 0
        camera_center = self._sample_base_center_from_camera_xy_shift(
            clean_state,
            size,
            ground_y,
            frame_idx,
            image_idx,
        )
        if camera_center is not None:
            return camera_center, frame_idx, image_idx

        frame_idx = self.rng.randrange(max(num_frames, 1))
        image_idx = _image_index(frame_idx, view_idx, num_views, clean_state.camera_to_world.shape[0])
        if image_idx is None:
            image_idx = 0
        scene_center = self._sample_base_center_from_scene_points(
            clean_state,
            size,
            ground_y,
            frame_idx,
            image_idx,
        )
        if scene_center is not None:
            return scene_center, frame_idx, image_idx

        camera = clean_state.camera_to_world[image_idx].detach().cpu().float()
        origin = camera[:3, 3]
        right = camera[:3, 0]
        forward = camera[:3, 2]
        right = right / right.norm().clamp_min(1e-6)
        forward = forward / forward.norm().clamp_min(1e-6)

        length = float(size[0].item())
        if clean_state.means.numel() > 0:
            distances = (clean_state.means.detach().cpu().float() - origin.view(1, 3)).norm(dim=-1)
            near = float(torch.quantile(distances, 0.25).item())
            far = float(torch.quantile(distances, 0.80).item())
            min_dist = max(length * 1.5, near)
            max_dist = max(min_dist + length, far)
        else:
            min_dist = length * 2.0
            max_dist = length * 6.0
        distance = self.rng.uniform(min_dist, max_dist)
        lateral = self.rng.uniform(-2.0 * length, 2.0 * length)
        center = origin + forward * distance + right * lateral
        if ground_y.numel() > 0:
            center[1] = _center_y_from_ground(ground_y[min(frame_idx, ground_y.numel() - 1)], size)
        return center, frame_idx, image_idx

    def _sample_base_center_from_camera_xy_shift(
        self,
        clean_state: CleanSceneState,
        size: torch.Tensor,
        ground_y: torch.Tensor,
        frame_idx: int,
        image_idx: int,
    ) -> torch.Tensor | None:
        if image_idx >= clean_state.point_map_world.shape[0] or image_idx >= clean_state.valid_mask.shape[0]:
            return None
        point_map = clean_state.point_map_world[image_idx].detach().cpu().float()
        valid_mask = clean_state.valid_mask[image_idx].detach().cpu().bool()
        if point_map.numel() == 0 or valid_mask.numel() == 0:
            return None

        height, width = valid_mask.shape
        candidate_mask = valid_mask.clone()
        if image_idx < clean_state.depth.shape[0]:
            depth = clean_state.depth[image_idx].detach().cpu().float()
            candidate_mask &= torch.isfinite(depth) & (depth > 1e-4)
        if clean_state.semantic_vehicle_mask.numel() > 0 and image_idx < clean_state.semantic_vehicle_mask.shape[0]:
            candidate_mask &= ~clean_state.semantic_vehicle_mask[image_idx].detach().cpu().bool()
        if ground_y.numel() > 0:
            gy = float(ground_y[min(int(frame_idx), ground_y.numel() - 1)].item())
            y_margin = max(float(size[2].item()) * 0.75, 0.12)
            candidate_mask &= (point_map[..., 1] - gy).abs() <= y_margin

        if int(candidate_mask.sum().item()) == 0:
            return None

        search_regions = (
            (0.54, 0.76, 0.28, 0.72),
            (0.50, 0.82, 0.18, 0.82),
            (0.44, 0.88, 0.08, 0.92),
            (0.36, 0.92, 0.02, 0.98),
        )
        for y0, y1, x0, x1 in search_regions:
            y_lo = int(round(float(height) * y0))
            y_hi = int(round(float(height) * y1))
            x_lo = int(round(float(width) * x0))
            x_hi = int(round(float(width) * x1))
            if y_hi <= y_lo or x_hi <= x_lo:
                continue
            region = candidate_mask[y_lo:y_hi, x_lo:x_hi]
            if int(region.sum().item()) == 0:
                continue
            ys, xs = torch.nonzero(region, as_tuple=True)
            if ys.numel() == 0:
                continue
            pick = self.rng.randrange(int(ys.numel()))
            y_full = int((ys[pick] + y_lo).item())
            x_full = int((xs[pick] + x_lo).item())
            ground_point = point_map[y_full, x_full].clone()
            if not bool(torch.isfinite(ground_point).all().item()):
                continue
            center = ground_point
            center[1] = _center_y_from_ground(ground_point[1], size)
            return center
        return None

    def _sample_base_center_from_scene_points(
        self,
        clean_state: CleanSceneState,
        size: torch.Tensor,
        ground_y: torch.Tensor,
        frame_idx: int,
        image_idx: int,
    ) -> torch.Tensor | None:
        if clean_state.means.numel() == 0 or clean_state.source_image_ids.numel() == 0:
            return None
        candidate_indices = self._scene_candidate_indices(clean_state, image_idx)
        if candidate_indices.numel() == 0:
            return None
        points = clean_state.means.detach().cpu().float()
        if ground_y.numel() > 0:
            gy = float(ground_y[min(int(frame_idx), ground_y.numel() - 1)].item())
            y_margin = max(float(size[2].item()) * 1.0, 0.25)
            candidate_y = points[candidate_indices, 1]
            near_ground = (candidate_y - gy).abs() <= y_margin
            if int(near_ground.sum().item()) >= 32:
                candidate_indices = candidate_indices[near_ground]
        pick = candidate_indices[self.rng.randrange(int(candidate_indices.numel()))]
        center = points[pick].clone()
        center[1] = _center_y_from_ground(center[1], size)
        return center

    def _scene_candidate_indices(self, clean_state: CleanSceneState, image_idx: int) -> torch.Tensor:
        image_idx = int(image_idx)
        cached = self._scene_point_candidate_cache.get(image_idx)
        if cached is not None:
            return cached

        source_image_ids = clean_state.source_image_ids.detach().cpu().long()
        mask = source_image_ids == image_idx
        if clean_state.source_y.numel() == clean_state.means.shape[0]:
            image_h = int(clean_state.images.shape[-2])
            source_y = clean_state.source_y.detach().cpu().long()
            mid_ground_mask = (
                mask
                & (source_y >= int(round(image_h * 0.45)))
                & (source_y <= int(round(image_h * 0.84)))
            )
            if int(mid_ground_mask.sum().item()) >= 32:
                mask = mid_ground_mask
        if clean_state.dynamic_prob.numel() == clean_state.means.shape[0]:
            mask &= clean_state.dynamic_prob.detach().cpu().float() < 0.5
        if clean_state.semantic_vehicle_mask.numel() > 0 and image_idx < clean_state.semantic_vehicle_mask.shape[0]:
            semantic = clean_state.semantic_vehicle_mask[image_idx].detach().cpu().bool()
            ys = clean_state.source_y.detach().cpu().long().clamp(0, semantic.shape[0] - 1)
            xs = clean_state.source_x.detach().cpu().long().clamp(0, semantic.shape[1] - 1)
            mask &= ~semantic[ys, xs]

        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        self._scene_point_candidate_cache[image_idx] = candidate_indices
        return candidate_indices

    def _build_motion(
        self,
        clean_state: CleanSceneState,
        base_center: torch.Tensor,
        base_frame_idx: int,
        base_yaw: float,
        size: torch.Tensor,
        motion_mode: str,
        ground_y: torch.Tensor,
        num_frames: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = base_center.view(1, 3).repeat(num_frames, 1)
        yaws = torch.full((num_frames,), float(base_yaw), dtype=torch.float32)
        length = float(size[0].item())
        if motion_mode == "slow":
            step = self.rng.uniform(0.3, 1.2) * length / float(max(num_frames, 1))
            direction = _rotation_from_yaw(base_yaw)[:, 0]
            for frame_idx in range(num_frames):
                centers[frame_idx] = base_center + direction * (step * frame_idx)
        elif motion_mode == "ego_matched":
            cameras = []
            for frame_idx in range(num_frames):
                image_idx = _image_index(frame_idx, 0, num_views, clean_state.camera_to_world.shape[0])
                if image_idx is None:
                    image_idx = min(frame_idx, clean_state.camera_to_world.shape[0] - 1)
                cameras.append(clean_state.camera_to_world[image_idx].detach().cpu().float())
            camera_to_world = torch.stack(cameras, dim=0)
            ref_idx = max(0, min(int(base_frame_idx), camera_to_world.shape[0] - 1))
            ref_camera = camera_to_world[ref_idx]
            ref_rotation = ref_camera[:3, :3]
            ref_translation = ref_camera[:3, 3]
            relative_cam = ref_rotation.T @ (base_center.detach().cpu().float() - ref_translation)
            drift_x_total = self.rng.uniform(-0.45, 0.45) * length
            drift_z_total = self.rng.uniform(-0.35, 0.35) * length
            wobble_phase = self.rng.uniform(0.0, 2.0 * math.pi)
            drift = torch.zeros((num_frames, 3), dtype=torch.float32)
            denom = float(max(num_frames - 1, 1))
            for frame_idx in range(num_frames):
                t = float(frame_idx) / denom
                ease = t * t * (3.0 - 2.0 * t)
                wobble = math.sin(2.0 * math.pi * t + wobble_phase)
                drift[frame_idx, 0] = drift_x_total * ease + 0.05 * length * wobble
                drift[frame_idx, 2] = drift_z_total * ease
            rel = relative_cam.view(1, 3) + drift
            centers = torch.einsum("nij,nj->ni", camera_to_world[:, :3, :3], rel) + camera_to_world[:, :3, 3]

        if ground_y.numel() > 0:
            for frame_idx in range(num_frames):
                centers[frame_idx, 1] = _center_y_from_ground(ground_y[min(frame_idx, ground_y.numel() - 1)], size)

        return centers.contiguous(), yaws.contiguous()

    def _project_box(
        self,
        clean_state: CleanSceneState,
        center: torch.Tensor,
        size: torch.Tensor,
        yaw: float,
        image_idx: int,
        image_hw: tuple[int, int],
    ) -> dict[str, Any]:
        rotation = _rotation_from_yaw(float(yaw), dtype=center.dtype)
        corners = build_box_corners(center, size, rotation)
        uv, depths, valid = project_world_points(
            corners,
            clean_state.camera_to_world[image_idx],
            clean_state.intrinsics[image_idx],
            image_hw,
        )
        in_front = torch.isfinite(uv).all(dim=-1) & torch.isfinite(depths) & (depths > 1e-4)
        clipped_uv = uv.clone()
        clipped_uv[:, 0] = clipped_uv[:, 0].clamp(0.0, float(image_hw[1]))
        clipped_uv[:, 1] = clipped_uv[:, 1].clamp(0.0, float(image_hw[0]))
        bbox = compute_bbox_from_projected_points(clipped_uv, in_front)
        visible = False
        area = 0.0
        if bbox is not None:
            area = float(((bbox[2] - bbox[0]).clamp_min(0.0) * (bbox[3] - bbox[1]).clamp_min(0.0)).item())
        center_uv, center_depth, center_valid = project_world_points(
            center.view(1, 3),
            clean_state.camera_to_world[image_idx],
            clean_state.intrinsics[image_idx],
            image_hw,
        )
        if bbox is not None:
            visible = bool(center_valid[0].item()) and int(valid.sum().item()) >= 1 and area >= self.min_projected_area_px
        return {
            "rotation": rotation,
            "corners": corners,
            "uv": uv,
            "valid": valid,
            "bbox": bbox,
            "visible": bool(visible),
            "area": float(area),
            "center_uv": center_uv[0],
            "center_depth": center_depth[0],
            "center_valid": bool(center_valid[0].item()),
        }

    def _depth_ok(
        self,
        clean_state: CleanSceneState,
        projection: dict[str, Any],
        image_idx: int,
        length: float,
    ) -> bool:
        if not projection["center_valid"]:
            return True
        image_hw = clean_state.depth.shape[-2:]
        u = int(round(float(projection["center_uv"][0].item())))
        v = int(round(float(projection["center_uv"][1].item())))
        if not (0 <= v < image_hw[0] and 0 <= u < image_hw[1]):
            return True
        depth_at_center = clean_state.depth[image_idx, v, u].detach().cpu().float()
        if not torch.isfinite(depth_at_center) or float(depth_at_center.item()) <= 1e-4:
            return True
        return float(depth_at_center.item()) + float(length) > float(projection["center_depth"].item())

    def _semantic_overlap(
        self,
        clean_state: CleanSceneState,
        bbox: torch.Tensor,
        image_idx: int,
        image_hw: tuple[int, int],
        *,
        profile: dict[str, float | str] | None = None,
    ) -> int:
        if clean_state.semantic_vehicle_mask.numel() == 0 or image_idx >= clean_state.semantic_vehicle_mask.shape[0]:
            return 0
        poly_mask = _vehicle_occluder_mask_for_image_box(bbox, image_hw, profile=profile)
        semantic = clean_state.semantic_vehicle_mask[image_idx].detach().cpu().bool()
        if semantic.shape != poly_mask.shape:
            return 0
        return int((poly_mask & semantic).sum().item())

    def _ground_support_ok(
        self,
        clean_state: CleanSceneState,
        bbox: torch.Tensor,
        image_idx: int,
        image_hw: tuple[int, int],
    ) -> bool:
        if image_idx >= clean_state.valid_mask.shape[0]:
            return True
        height, width = image_hw
        x1 = max(0, min(width, int(math.floor(float(bbox[0].item())))))
        y1 = max(0, min(height, int(math.floor(float(bbox[1].item())))))
        x2 = max(0, min(width, int(math.ceil(float(bbox[2].item())))))
        y2 = max(0, min(height, int(math.ceil(float(bbox[3].item())))))
        if x2 <= x1 or y2 <= y1:
            return False

        box_h = max(1, y2 - y1)
        support_y1 = max(y1, y2 - max(4, int(round(box_h * 0.25))))
        support = clean_state.valid_mask[image_idx, support_y1:y2, x1:x2].detach().cpu().bool()
        if support.numel() == 0:
            return False
        support_ratio = float(support.float().mean().item())
        return support_ratio >= self.min_ground_support_ratio

    def _validate_candidate(
        self,
        clean_state: CleanSceneState,
        centers: torch.Tensor,
        yaws: torch.Tensor,
        size: torch.Tensor,
        existing_objects: list[dict[str, Any]],
        accepted: list[ImaginedObject],
        num_views: int,
        occluder_profile: dict[str, float | str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        num_frames = centers.shape[0]
        image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
        visible = torch.zeros((num_frames, num_views), dtype=torch.bool)
        bboxes = torch.zeros((num_frames, num_views, 4), dtype=torch.float32)
        semantic_overlap = 0
        max_frame_semantic_overlap = 0
        max_existing_iou = 0.0
        length = float(size[0].item())
        frame_skip_counts: dict[str, int] = {}

        def count_skip(reason: str) -> None:
            frame_skip_counts[reason] = frame_skip_counts.get(reason, 0) + 1

        for frame_idx in range(num_frames):
            rotation = _rotation_from_yaw(float(yaws[frame_idx].item()), dtype=centers.dtype)
            candidate_corners = build_box_corners(centers[frame_idx], size, rotation)
            cand_min, cand_max = _aabb_from_corners(candidate_corners)

            for obj in existing_objects:
                existing = _existing_box_at_frame(obj, frame_idx)
                if existing is None:
                    continue
                ex_center, ex_size, ex_rotation = existing
                ex_corners = build_box_corners(ex_center, ex_size, ex_rotation)
                ex_min, ex_max = _aabb_from_corners(ex_corners)
                iou = _aabb_iou(cand_min, cand_max, ex_min, ex_max)
                max_existing_iou = max(max_existing_iou, iou)
                if iou > 0.0:
                    return False, {"reason": "existing_3d_box_overlap", "existing_box_iou_3d": max_existing_iou}

            for obj in accepted:
                other_center = obj.center_dggt_per_frame[min(frame_idx, obj.center_dggt_per_frame.shape[0] - 1)]
                other_size = obj.size_dggt
                other_yaw = float(obj.yaw_dggt_per_frame[min(frame_idx, obj.yaw_dggt_per_frame.shape[0] - 1)].item())
                other_corners = build_box_corners(other_center, other_size, _rotation_from_yaw(other_yaw))
                other_min, other_max = _aabb_from_corners(other_corners)
                if _aabb_iou(cand_min, cand_max, other_min, other_max) > 0.0:
                    return False, {"reason": "imagined_3d_box_overlap", "existing_box_iou_3d": max_existing_iou}

            for view_idx in range(num_views):
                image_idx = _image_index(frame_idx, view_idx, num_views, clean_state.images.shape[0])
                if image_idx is None:
                    continue
                projection = self._project_box(
                    clean_state,
                    centers[frame_idx],
                    size,
                    float(yaws[frame_idx].item()),
                    image_idx,
                    image_hw,
                )
                if not projection["visible"]:
                    count_skip("projected_box_not_visible")
                    continue

                bbox = projection["bbox"]
                box_w = float((bbox[2] - bbox[0]).clamp_min(0.0).item())
                box_h = float((bbox[3] - bbox[1]).clamp_min(0.0).item())
                min_w = self.min_projected_transfer_size_px * float(image_hw[1]) / float(self.transfer_image_hw[1])
                min_h = self.min_projected_transfer_size_px * float(image_hw[0]) / float(self.transfer_image_hw[0])
                ground_contact_y = _estimate_ground_contact_y(
                    clean_state,
                    image_idx,
                    projection["center_uv"],
                    float((centers[frame_idx, 1] + 0.5 * size[2]).item()),
                    image_hw,
                    width_hint=max(box_w, min_w),
                )
                bbox = _normalize_grounded_edit_box(
                    bbox,
                    projection["center_uv"],
                    image_hw,
                    min_w=min_w,
                    min_h=min_h,
                    profile=occluder_profile,
                    ground_contact_y=ground_contact_y,
                )
                box_w = float((bbox[2] - bbox[0]).clamp_min(0.0).item())
                box_h = float((bbox[3] - bbox[1]).clamp_min(0.0).item())
                box_area = box_w * box_h
                box_cy = 0.5 * float((bbox[1] + bbox[3]).item())
                if box_w < min_w or box_h < min_h:
                    count_skip("projected_box_too_small")
                    continue
                if float(bbox[1].item()) < float(image_hw[0]) * self.min_projected_top_y_ratio:
                    count_skip("projected_box_top_too_high")
                    continue
                if box_cy < float(image_hw[0]) * self.min_projected_center_y_ratio:
                    count_skip("projected_box_too_high")
                    continue
                if float(bbox[3].item()) < float(image_hw[0]) * self.min_projected_bottom_y_ratio:
                    count_skip("projected_box_bottom_too_high")
                    continue
                if float(bbox[3].item()) > float(image_hw[0]) * self.max_projected_bottom_y_ratio:
                    count_skip("projected_box_bottom_too_low")
                    continue
                max_area = float(image_hw[0] * image_hw[1]) * self.max_projected_area_ratio
                if (
                    box_area > max_area
                    or box_w > float(image_hw[1]) * self.max_projected_width_ratio
                    or box_h > float(image_hw[0]) * self.max_projected_height_ratio
                ):
                    count_skip("projected_box_too_large")
                    continue
                if not self._depth_ok(clean_state, projection, image_idx, length):
                    count_skip("depth_conflict")
                    continue
                if not self._ground_support_ok(clean_state, bbox, image_idx, image_hw):
                    count_skip("insufficient_ground_support")
                    continue
                bboxes[frame_idx, view_idx] = bbox
                visible[frame_idx, view_idx] = True
                overlap_bbox = _scale_box_xyxy(bbox, 0.90, image_hw)
                frame_semantic_overlap = self._semantic_overlap(
                    clean_state,
                    overlap_bbox,
                    image_idx,
                    image_hw,
                    profile=occluder_profile,
                )
                semantic_overlap += frame_semantic_overlap
                max_frame_semantic_overlap = max(max_frame_semantic_overlap, int(frame_semantic_overlap))
                if semantic_overlap > self.max_semantic_overlap_px:
                    return False, {
                        "reason": "semantic_overlap",
                        "semantic_overlap_px": int(semantic_overlap),
                        "max_frame_semantic_overlap_px": int(max_frame_semantic_overlap),
                        "existing_box_iou_3d": max_existing_iou,
                    }

        keep_frames = _longest_visible_run(visible.any(dim=1))
        if keep_frames.numel() == 0 or not bool(keep_frames.any().item()):
            return False, {
                "reason": "no_visible_frames",
                "visible_frames": 0,
                "frame_skip_counts": frame_skip_counts,
                "existing_box_iou_3d": max_existing_iou,
            }
        if self.require_first_frame_visible and not bool(keep_frames[0].item()):
            return False, {
                "reason": "first_frame_not_visible",
                "visible_frames": int(keep_frames.sum().item()),
                "frame_skip_counts": frame_skip_counts,
                "existing_box_iou_3d": max_existing_iou,
            }
        visible &= keep_frames.view(-1, 1)
        bboxes = bboxes * keep_frames.view(-1, 1, 1).to(dtype=bboxes.dtype)
        visible_frames = int(keep_frames.sum().item())
        if visible_frames < self.min_visible_frames:
            return False, {
                "reason": "too_few_visible_frames",
                "visible_frames": visible_frames,
                "frame_skip_counts": frame_skip_counts,
                "existing_box_iou_3d": max_existing_iou,
            }

        return True, {
            "visible": visible,
            "bboxes": bboxes,
            "semantic_overlap_px": int(semantic_overlap),
            "max_frame_semantic_overlap_px": int(max_frame_semantic_overlap),
            "existing_box_iou_3d": float(max_existing_iou),
            "visible_frames": visible_frames,
        }

    def _view_coverage_ok(self, objects: list[ImaginedObject], num_views: int) -> bool:
        if num_views < 3:
            return True
        for view_idx in range(num_views):
            if not any(int(obj.visible_in_frame_per_view[view_idx].sum().item()) >= self.min_visible_frames for obj in objects):
                return False
        return True

    def plan(
        self,
        clean_state: CleanSceneState,
        *,
        existing_objects: list[dict[str, Any]] | None = None,
        num_objects_target: int | None = None,
        views: int | None = None,
        scene_name: str = "",
        clip_name: str = "",
        clip_index: int = 0,
    ) -> ModeBPlan:
        existing_objects = list(existing_objects or [])
        num_views = _infer_num_views(clean_state, views)
        num_frames = _infer_num_frames(clean_state, num_views)
        target_count = int(num_objects_target) if num_objects_target is not None else self._sample_target_count(num_views)
        camera_motion = self._camera_motion_metrics(clean_state, num_frames, num_views)
        camera_motion_level = str(camera_motion["level"])
        ground_y = estimate_ground_plane_per_frame(clean_state)
        if ground_y.numel() == 0:
            ground_y = torch.zeros((num_frames,), dtype=torch.float32)

        accepted: list[ImaginedObject] = []
        rejection_counts: dict[str, int] = {}
        rejection_counts_by_shrink: dict[str, dict[str, int]] = {}
        frame_skip_counts_by_rejection: dict[str, dict[str, int]] = {}
        best_too_few_visible_frames = 0
        best_too_few_skip_counts: dict[str, int] = {}
        shrink_factors = (1.0, 0.7, 0.5, 0.35, 0.25, 0.18)
        for slot in range(target_count):
            accepted_obj = None
            for shrink in shrink_factors:
                shrink_key = f"{shrink:.2f}"
                rejection_counts_by_shrink.setdefault(shrink_key, {})
                for trial in range(self.max_trials_per_object):
                    size = self._sample_size(existing_objects) * float(shrink)
                    base_center, base_frame_idx, image_idx = self._sample_base_center(
                        clean_state,
                        size,
                        ground_y,
                        num_frames,
                        num_views,
                        view_hint=slot + trial,
                    )
                    forward = clean_state.camera_to_world[image_idx, :3, 2].detach().cpu().float()
                    yaw = _yaw_from_forward(forward)
                    yaw += math.radians(self.rng.uniform(-self.yaw_jitter_deg, self.yaw_jitter_deg))
                    if self.rng.random() < 0.5:
                        yaw += math.pi
                    motion_mode = self._sample_motion_mode(camera_motion_level)
                    occluder_profile = self._sample_occluder_profile()
                    centers, yaws = self._build_motion(
                        clean_state,
                        base_center,
                        base_frame_idx,
                        yaw,
                        size,
                        motion_mode,
                        ground_y,
                        num_frames,
                        num_views,
                    )
                    ok, info = self._validate_candidate(
                        clean_state,
                        centers,
                        yaws,
                        size,
                        existing_objects,
                        accepted,
                        num_views,
                        occluder_profile=occluder_profile,
                    )
                    if not ok:
                        reason = str(info.get("reason", "unknown"))
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                        shrink_counts = rejection_counts_by_shrink[shrink_key]
                        shrink_counts[reason] = shrink_counts.get(reason, 0) + 1
                        frame_skip_counts = info.get("frame_skip_counts", {})
                        if isinstance(frame_skip_counts, dict):
                            reason_frame_counts = frame_skip_counts_by_rejection.setdefault(reason, {})
                            for skip_reason, count in frame_skip_counts.items():
                                skip_key = str(skip_reason)
                                reason_frame_counts[skip_key] = reason_frame_counts.get(skip_key, 0) + int(count)
                        if reason == "too_few_visible_frames":
                            visible_frames = int(info.get("visible_frames", 0))
                            if visible_frames >= best_too_few_visible_frames:
                                best_too_few_visible_frames = visible_frames
                                best_too_few_skip_counts = dict(info.get("frame_skip_counts", {}))
                        continue
                    accepted_obj = ImaginedObject(
                        slot=slot,
                        motion_mode=motion_mode,
                        size_dggt=size.detach().cpu().float(),
                        center_dggt_per_frame=centers.detach().cpu().float(),
                        yaw_dggt_per_frame=yaws.detach().cpu().float(),
                        visible_in_frame_per_view=info["visible"].T.contiguous(),
                        bbox_2d_per_view=info["bboxes"].contiguous(),
                        semantic_overlap_px=int(info["semantic_overlap_px"]),
                        existing_box_iou_3d=float(info["existing_box_iou_3d"]),
                        occluder_profile=occluder_profile,
                    )
                    break
                if accepted_obj is not None:
                    break
            if accepted_obj is not None:
                accepted.append(accepted_obj)

        min_count = 1 if num_views == 1 else 3
        eligible = len(accepted) >= min_count and self._view_coverage_ok(accepted, num_views)
        rejection_reason = "" if eligible else "too_few_accepted_objects"
        if len(accepted) >= min_count and not self._view_coverage_ok(accepted, num_views):
            rejection_reason = "view_coverage_failed"

        return ModeBPlan(
            scene_name=scene_name,
            clip_name=clip_name,
            clip_index=int(clip_index),
            views=int(num_views),
            imagined_objects=accepted,
            rng_seed=int(self.rng_seed),
            eligible=bool(eligible),
            rejection_reason=rejection_reason,
            metrics={
                "target_count": int(target_count),
                "accepted_count": int(len(accepted)),
                "min_count": int(min_count),
                "rejection_counts": rejection_counts,
                "camera_motion": camera_motion,
                "min_visible_frames": int(self.min_visible_frames),
                "max_semantic_overlap_px": int(self.max_semantic_overlap_px),
                "require_first_frame_visible": bool(self.require_first_frame_visible),
                "min_projected_transfer_size_px": float(self.min_projected_transfer_size_px),
                "transfer_image_hw": [int(v) for v in self.transfer_image_hw],
                "max_projected_area_ratio": float(self.max_projected_area_ratio),
                "max_projected_width_ratio": float(self.max_projected_width_ratio),
                "max_projected_height_ratio": float(self.max_projected_height_ratio),
                "min_projected_top_y_ratio": float(self.min_projected_top_y_ratio),
                "min_projected_center_y_ratio": float(self.min_projected_center_y_ratio),
                "min_projected_bottom_y_ratio": float(self.min_projected_bottom_y_ratio),
                "max_projected_bottom_y_ratio": float(self.max_projected_bottom_y_ratio),
                "min_ground_support_ratio": float(self.min_ground_support_ratio),
                "fast_camera_step_ratio": float(self.fast_camera_step_ratio),
                "slow_camera_step_ratio": float(self.slow_camera_step_ratio),
                "best_rejected_visible_frames": int(best_too_few_visible_frames),
                "best_rejected_frame_skip_counts": best_too_few_skip_counts,
                "rejection_counts_by_shrink": rejection_counts_by_shrink,
                "frame_skip_counts_by_rejection": frame_skip_counts_by_rejection,
            },
        )


def apply_mode_b(
    clean_state: CleanSceneState,
    imagined_objects: ModeBPlan | list[ImaginedObject],
    *,
    core_scale: float = 1.0,
    shell_scale: float = 1.05,
    depth_tolerance: float = 0.05,
    shell_depth_slack_ratio: float = 0.25,
) -> ModeBDeletionResult:
    objects = imagined_objects.imagined_objects if isinstance(imagined_objects, ModeBPlan) else list(imagined_objects)
    num_points = int(clean_state.means.shape[0])
    num_frames = max((int(obj.center_dggt_per_frame.shape[0]) for obj in objects), default=0)
    delete_mask_per_frame = torch.zeros((num_frames, num_points), dtype=torch.bool)
    core_mask = torch.zeros((num_points,), dtype=torch.bool)
    shell_mask = torch.zeros((num_points,), dtype=torch.bool)
    means = clean_state.means.detach().cpu().float()
    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    num_views = _infer_num_views(clean_state, None)
    num_images = int(clean_state.images.shape[0])

    for frame_idx in range(num_frames):
        frame_core_mask = torch.zeros((num_points,), dtype=torch.bool)
        frame_shell_mask = torch.zeros((num_points,), dtype=torch.bool)
        frame_protected_mask = torch.zeros((num_points,), dtype=torch.bool)
        projection_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for obj in objects:
            local_frame_idx = min(frame_idx, obj.center_dggt_per_frame.shape[0] - 1)
            center = obj.center_dggt_per_frame[local_frame_idx].detach().cpu().float()
            size = obj.size_dggt.detach().cpu().float()
            yaw = float(obj.yaw_dggt_per_frame[local_frame_idx].item())
            rotation = _rotation_from_yaw(yaw)
            box_corners = build_box_corners(center, size, rotation)

            for view_idx in range(num_views):
                if view_idx >= obj.visible_in_frame_per_view.shape[0]:
                    continue
                if not bool(obj.visible_in_frame_per_view[view_idx, local_frame_idx].item()):
                    continue
                image_idx = _image_index(frame_idx, view_idx, num_views, num_images)
                if image_idx is None:
                    continue
                bbox = obj.bbox_2d_per_view[local_frame_idx, view_idx].detach().cpu().float()
                if float((bbox[2] - bbox[0]).item()) <= 0.0 or float((bbox[3] - bbox[1]).item()) <= 0.0:
                    continue

                cached = projection_cache.get(int(image_idx))
                if cached is None:
                    cached = project_world_points(
                        means,
                        clean_state.camera_to_world[image_idx],
                        clean_state.intrinsics[image_idx],
                        image_hw,
                    )
                    projection_cache[int(image_idx)] = cached
                uv, depths, valid = cached
                frame_protected_mask |= _points_on_semantic_vehicle_mask(clean_state, uv, valid, image_idx)
                _, corner_depths, corner_valid = project_world_points(
                    box_corners,
                    clean_state.camera_to_world[image_idx],
                    clean_state.intrinsics[image_idx],
                    image_hw,
                )
                if bool(corner_valid.any().item()):
                    min_object_depth = float(corner_depths[corner_valid].min().item())
                else:
                    _, center_depth, center_valid = project_world_points(
                        center.view(1, 3),
                        clean_state.camera_to_world[image_idx],
                        clean_state.intrinsics[image_idx],
                        image_hw,
                    )
                    min_object_depth = float(center_depth[0].item()) if bool(center_valid[0].item()) else 0.0

                core_bbox = _scale_box_xyxy(bbox, float(core_scale), image_hw)
                shell_bbox = _scale_box_xyxy(bbox, float(shell_scale), image_hw)
                shell_depth_slack = max(float(depth_tolerance), float(size.max().item()) * float(shell_depth_slack_ratio))
                shell_depth_keep = depths >= (min_object_depth - shell_depth_slack)
                profile = getattr(obj, "occluder_profile", {}) or {}
                core_here = _points_in_vehicle_occluder(uv, valid, core_bbox, profile)
                shell_full = _points_in_vehicle_occluder(uv, valid & shell_depth_keep, shell_bbox, profile)
                shell_here = shell_full & ~core_here
                frame_core_mask |= core_here
                frame_shell_mask |= shell_here
        frame_core_mask &= ~frame_protected_mask
        frame_shell_mask &= ~frame_protected_mask
        core_mask |= frame_core_mask
        shell_mask |= frame_shell_mask
        frame_mask = frame_core_mask | frame_shell_mask
        delete_mask_per_frame[frame_idx] = frame_mask

    delete_mask = core_mask | shell_mask
    delete_core_indices = torch.nonzero(core_mask, as_tuple=False).flatten().to(torch.int64)
    delete_shell_indices = torch.nonzero(shell_mask & ~core_mask, as_tuple=False).flatten().to(torch.int64)
    return ModeBDeletionResult(
        delete_mask=delete_mask,
        shell_mask=shell_mask & ~core_mask,
        delete_mask_per_frame=delete_mask_per_frame,
        delete_core_indices=delete_core_indices,
        delete_shell_indices=delete_shell_indices,
    )


def _scale_box_xyxy(box: torch.Tensor, scale: float, image_hw: tuple[int, int]) -> torch.Tensor:
    height, width = image_hw
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = max(0.0, x2 - x1) * 0.5 * float(scale)
    half_h = max(0.0, y2 - y1) * 0.5 * float(scale)
    return torch.tensor(
        [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(float(width), cx + half_w),
            min(float(height), cy + half_h),
        ],
        dtype=torch.float32,
    )


def _normalize_grounded_edit_box(
    box: torch.Tensor,
    center_uv: torch.Tensor,
    image_hw: tuple[int, int],
    *,
    min_w: float,
    min_h: float,
    profile: dict[str, float | str] | None = None,
    ground_contact_y: float | None = None,
) -> torch.Tensor:
    height, width = image_hw
    profile = profile or {}
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    cx = float(center_uv[0].item()) if torch.isfinite(center_uv[0]) else 0.5 * (x1 + x2)
    bottom = y2 if ground_contact_y is None else max(y2, float(ground_contact_y))
    raw_w = max(0.0, x2 - x1)
    raw_h = max(0.0, y2 - y1)
    width_mult = float(profile.get("bbox_width_mult", 2.45))
    height_mult = float(profile.get("bbox_height_mult", 1.95))
    target_w = float(min_w) * width_mult
    target_h = float(min_h) * height_mult
    edit_w = max(float(min_w), min(max(raw_w, target_w), float(width) * 0.42))
    edit_h = max(float(min_h), min(max(raw_h, target_h), float(height) * 0.44))
    x1_new = max(0.0, min(float(width) - edit_w, cx - 0.5 * edit_w))
    x2_new = min(float(width), x1_new + edit_w)
    y2_new = max(edit_h, min(float(height), bottom))
    y1_new = max(0.0, y2_new - edit_h)
    return torch.tensor([x1_new, y1_new, x2_new, y2_new], dtype=torch.float32)


def _estimate_ground_contact_y(
    clean_state: CleanSceneState,
    image_idx: int,
    center_uv: torch.Tensor,
    ground_world_y: float,
    image_hw: tuple[int, int],
    *,
    width_hint: float,
) -> float | None:
    if image_idx >= clean_state.point_map_world.shape[0] or image_idx >= clean_state.valid_mask.shape[0]:
        return None
    point_map = clean_state.point_map_world[image_idx].detach().cpu().float()
    valid_mask = clean_state.valid_mask[image_idx].detach().cpu().bool()
    if point_map.numel() == 0 or valid_mask.numel() == 0:
        return None

    height, width = image_hw
    cx = int(round(float(center_uv[0].item())))
    cy = int(round(float(center_uv[1].item())))
    if not (0 <= cx < width and 0 <= cy < height):
        return None

    x_radius = max(10, int(round(float(width_hint) * 0.22)))
    x1 = max(0, cx - x_radius)
    x2 = min(width, cx + x_radius + 1)
    y1 = max(0, cy)
    y2 = height

    patch_valid = valid_mask[y1:y2, x1:x2].clone()
    if clean_state.semantic_vehicle_mask.numel() > 0 and image_idx < clean_state.semantic_vehicle_mask.shape[0]:
        patch_valid &= ~clean_state.semantic_vehicle_mask[image_idx, y1:y2, x1:x2].detach().cpu().bool()
    if int(patch_valid.sum().item()) == 0:
        return None

    patch_points = point_map[y1:y2, x1:x2]
    y_margin = max(0.10, float(width_hint) * 0.0025)
    ground_mask = patch_valid & ((patch_points[..., 1] - float(ground_world_y)).abs() <= y_margin)
    if int(ground_mask.sum().item()) == 0:
        y_margin = max(0.20, y_margin * 2.0)
        ground_mask = patch_valid & ((patch_points[..., 1] - float(ground_world_y)).abs() <= y_margin)
    matched_y = None
    if int(ground_mask.sum().item()) > 0:
        ys, xs = torch.nonzero(ground_mask, as_tuple=True)
        if ys.numel() > 0:
            score = ys.float() + 0.15 * (xs.float() - float(cx - x1)).abs()
            pick = int(torch.argmin(score).item())
            matched_y = float((ys[pick] + y1).item())

    # Fallback: estimate a local road-contact row from valid support below the object.
    support_ys, _ = torch.nonzero(patch_valid, as_tuple=True)
    if support_ys.numel() == 0:
        return matched_y
    support_rows = support_ys.float() + float(y1)
    quantile_y = float(torch.quantile(support_rows, 0.45).item())
    if matched_y is None:
        return quantile_y
    return max(matched_y, quantile_y)


def _points_in_image_box(uv: torch.Tensor, valid: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    return (
        valid
        & (uv[:, 0] >= box[0])
        & (uv[:, 0] <= box[2])
        & (uv[:, 1] >= box[1])
        & (uv[:, 1] <= box[3])
    )


def _points_on_semantic_vehicle_mask(
    clean_state: CleanSceneState,
    uv: torch.Tensor,
    valid: torch.Tensor,
    image_idx: int,
) -> torch.Tensor:
    protected = torch.zeros((uv.shape[0],), dtype=torch.bool)
    if clean_state.semantic_vehicle_mask.numel() == 0 or image_idx >= clean_state.semantic_vehicle_mask.shape[0]:
        return protected
    semantic = clean_state.semantic_vehicle_mask[image_idx].detach().cpu().bool()
    if semantic.numel() == 0:
        return protected
    height, width = semantic.shape
    finite = valid.detach().cpu().bool() & torch.isfinite(uv.detach().cpu()).all(dim=-1)
    if int(finite.sum().item()) == 0:
        return protected
    xs = uv.detach().cpu()[:, 0].round().long().clamp(0, width - 1)
    ys = uv.detach().cpu()[:, 1].round().long().clamp(0, height - 1)
    protected[finite] = semantic[ys[finite], xs[finite]]
    return protected


def _longest_visible_run(visible_by_frame: torch.Tensor) -> torch.Tensor:
    visible_list = [bool(v) for v in visible_by_frame.detach().cpu().bool().tolist()]
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for idx, is_visible in enumerate(visible_list):
        if is_visible:
            if cur_len == 0:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_len = 0
    keep = torch.zeros_like(visible_by_frame, dtype=torch.bool)
    if best_len > 0:
        keep[best_start : best_start + best_len] = True
    return keep


def _vehicle_occluder_mask_for_image_box(
    box: torch.Tensor,
    image_hw: tuple[int, int],
    profile: dict[str, float | str] | None = None,
) -> torch.Tensor:
    height, width = image_hw
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    uv = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    valid = torch.ones((height * width,), dtype=torch.bool)
    return _points_in_vehicle_occluder(uv, valid, box.detach().cpu().float(), profile).view(height, width)


def _points_in_vehicle_occluder(
    uv: torch.Tensor,
    valid: torch.Tensor,
    box: torch.Tensor,
    profile: dict[str, float | str] | None = None,
) -> torch.Tensor:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    profile = profile or {}
    shape_width_scale = float(profile.get("shape_width_scale", 1.0))
    shape_height_scale = float(profile.get("shape_height_scale", 1.0))
    shape_center_y = float(profile.get("shape_center_y", 0.5))
    half_w = max(1e-6, 0.5 * (x2 - x1) * shape_width_scale)
    half_h = max(1e-6, 0.5 * (y2 - y1) * shape_height_scale)
    cx = 0.5 * (x1 + x2)
    cy = y1 + (y2 - y1) * shape_center_y
    nx = (uv[:, 0] - cx) / half_w
    ny = (uv[:, 1] - cy) / half_h
    top_taper = ((-ny - 0.05) / 0.95).clamp(0.0, 1.0)
    bottom = ((ny - 0.15) / 0.85).clamp(0.0, 1.0)
    taper = float(profile.get("top_taper", 0.22))
    bottom_boost = float(profile.get("bottom_boost", 0.06))
    width_profile = (1.0 - taper * top_taper + bottom_boost * bottom).clamp(0.70, 1.12)
    exponent = float(profile.get("exponent", 2.8))
    shape = (nx.abs() / width_profile).pow(exponent) + ny.abs().pow(exponent) <= 1.0
    return _points_in_image_box(uv, valid, box) & shape
