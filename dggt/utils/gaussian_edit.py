from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dggt.utils.gaussian_ply import GAUSSIAN_SH_C0, read_gaussian_ply
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.rotation import mat_to_quat, quat_to_mat

_GS_LWH_TO_XYZ_SCALE = (0.90, 0.90, 0.88)
WAYMO_DYNAMIC_SPEED_THRESH_MPS = 1.0
_DYNAMIC_VOXEL_SCALE = (0.10, 0.10, 0.20)
_STATIC_VOXEL_SCALE = (0.08, 0.08, 0.16)
_MAX_POSE_REFINE_YAW_DEG = 15.0
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


@dataclass
class Sim3Transform:
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    mean_alignment_error: float

    def apply_points(self, points: torch.Tensor) -> torch.Tensor:
        return self.scale * (points @ self.rotation.T) + self.translation

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation": [[float(v) for v in row] for row in self.rotation.tolist()],
            "translation": [float(v) for v in self.translation.tolist()],
            "mean_alignment_error": float(self.mean_alignment_error),
        }


@dataclass
class CleanSceneState:
    images: torch.Tensor
    means: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    gs_conf: torch.Tensor
    dynamic_prob: torch.Tensor
    source_image_ids: torch.Tensor
    source_frame_ids: torch.Tensor
    source_view_ids: torch.Tensor
    source_y: torch.Tensor
    source_x: torch.Tensor
    point_map_world: torch.Tensor
    valid_mask: torch.Tensor
    depth: torch.Tensor
    semantic_vehicle_prob: torch.Tensor
    semantic_vehicle_mask: torch.Tensor
    camera_to_world: torch.Tensor
    world_to_camera: torch.Tensor
    intrinsics: torch.Tensor


@dataclass
class LocalizedFrameObject:
    slot_idx: int
    frame_idx: int
    source_front_index: int
    asset_object_id: str
    scene_raw_object_id: str
    asset_path: str
    match_score: float
    delete_motion_mode: str
    waymo_frame_dynamic: bool
    waymo_frame_speed_mps: float
    waymo_max_speed_mps: float
    waymo_mean_speed_mps: float
    render_dynamic_ratio: float
    gt_center: torch.Tensor
    gt_size: torch.Tensor
    gt_rotation: torch.Tensor
    proposal_center: torch.Tensor
    proposal_size: torch.Tensor
    proposal_rotation: torch.Tensor
    refined_center: torch.Tensor
    refined_size: torch.Tensor
    refined_rotation: torch.Tensor
    asset_rotation: torch.Tensor
    asset_scale: float
    asset_bottom_center: torch.Tensor
    delete_core_indices: torch.Tensor
    delete_shell_indices: torch.Tensor
    candidate_count: int
    seed_point_count: int
    candidate_pool_count: int
    cluster_kept_count: int
    target_delete_coverage: float
    outside_box_leak_ratio: float
    target_bbox_model: torch.Tensor | None
    projected_asset_bbox: torch.Tensor | None
    seed_pixel_mask: torch.Tensor | None
    delete_component_pixel_mask: torch.Tensor | None
    asset_means_world: torch.Tensor
    asset_colors: torch.Tensor
    asset_opacities: torch.Tensor
    asset_scales: torch.Tensor
    asset_quats: torch.Tensor
    asset_means_local: torch.Tensor
    asset_scales_local: torch.Tensor
    asset_quats_local: torch.Tensor
    asset_scale_factors: torch.Tensor
    asset_object_to_world: torch.Tensor
    asset_local_yaw_deg: float = 0.0
    pose_refine_diagnostics: dict[str, Any] | None = None
    waymo_box_corners_model: torch.Tensor | None = None
    waymo_box_corners_valid: torch.Tensor | None = None
    initial_box_corners_model: torch.Tensor | None = None
    initial_box_corners_valid: torch.Tensor | None = None
    refined_box_corners_model: torch.Tensor | None = None
    refined_box_corners_valid: torch.Tensor | None = None


@dataclass
class EditedSceneState:
    clean: dict[str, torch.Tensor]
    deleted: dict[str, torch.Tensor]
    asset_only: dict[str, torch.Tensor]
    edited: dict[str, torch.Tensor]
    localized_objects: list[LocalizedFrameObject]
    delete_mask: torch.Tensor
    shell_mask: torch.Tensor


def _to_cpu_float_tensor(value: Any, *, shape_last: int | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().to(torch.float32).contiguous()
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    if shape_last is not None and tensor.dim() == 1:
        tensor = tensor.view(-1, shape_last)
    return tensor.contiguous()


def _rigid_inverse_from_world_to_camera(extrinsics: torch.Tensor) -> torch.Tensor:
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    inv_rot = rotations.transpose(1, 2)
    inv_t = -torch.bmm(inv_rot, translations.unsqueeze(-1)).squeeze(-1)
    camera_to_world = torch.eye(4, dtype=extrinsics.dtype).unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    camera_to_world[:, :3, :3] = inv_rot
    camera_to_world[:, :3, 3] = inv_t
    return camera_to_world


def _unproject_depth_to_world(
    depth: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    seq_len, height, width = depth.shape
    device = depth.device

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    xx = xx.view(1, height, width).expand(seq_len, -1, -1)
    yy = yy.view(1, height, width).expand(seq_len, -1, -1)

    fx = intrinsics[:, 0, 0].view(seq_len, 1, 1)
    fy = intrinsics[:, 1, 1].view(seq_len, 1, 1)
    cx = intrinsics[:, 0, 2].view(seq_len, 1, 1)
    cy = intrinsics[:, 1, 2].view(seq_len, 1, 1)

    x_cam = (xx - cx) * depth / fx.clamp_min(1e-6)
    y_cam = (yy - cy) * depth / fy.clamp_min(1e-6)
    cam_h = torch.stack([x_cam, y_cam, depth, torch.ones_like(depth)], dim=-1)
    camera_to_world = _rigid_inverse_from_world_to_camera(world_to_camera)
    world_h = torch.matmul(cam_h.view(seq_len, -1, 4), camera_to_world.transpose(1, 2))
    return world_h.view(seq_len, height, width, 4)[..., :3]


def build_clean_scene_state(sample: dict[str, Any], predictions: dict[str, torch.Tensor]) -> CleanSceneState:
    images = sample["images_clean"].detach().cpu().float()
    seq_len, _, height, width = images.shape

    pose_enc = predictions["pose_enc"].detach().cpu().float()
    world_to_camera, intrinsics = pose_encoding_to_extri_intri(pose_enc, (height, width))
    world_to_camera = world_to_camera[0].detach().cpu().float().contiguous()
    intrinsics = intrinsics[0].detach().cpu().float().contiguous()
    camera_to_world = _rigid_inverse_from_world_to_camera(world_to_camera)

    depth = predictions["depth"][0].detach().cpu().float()
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    point_map_world = _unproject_depth_to_world(depth, world_to_camera, intrinsics)

    gs_map = predictions["gs_map"][0].detach().cpu().float()
    dynamic_logits = predictions["dynamic_conf"][0].detach().cpu().float()
    if dynamic_logits.shape[-1] == 1:
        dynamic_logits = dynamic_logits[..., 0]
    dynamic_prob = torch.sigmoid(dynamic_logits)
    gs_conf = predictions["gs_conf"][0].detach().cpu().float()
    semantic_logits = predictions.get("semantic_logits")
    semantic_vehicle_prob = torch.zeros((seq_len, height, width), dtype=torch.float32)
    semantic_vehicle_mask = torch.zeros((seq_len, height, width), dtype=torch.bool)
    if semantic_logits is not None:
        semantic_logits = semantic_logits[0].detach().cpu().float()
        semantic_probs = torch.softmax(semantic_logits, dim=-1)
        if semantic_probs.shape[-1] > 4:
            semantic_vehicle_prob = semantic_probs[..., 4]
            semantic_vehicle_mask = semantic_probs.argmax(dim=-1) == 4

    sky_mask = sample.get("sky_mask", sample["masks"]).detach().cpu().float()
    sky_mask_hw = sky_mask.permute(0, 2, 3, 1)
    non_sky_mask = (sky_mask_hw < 0.5).any(dim=-1)
    valid_mask = non_sky_mask & (depth > 1e-4)

    means = point_map_world[valid_mask].reshape(-1, 3)
    colors = gs_map[..., :3][valid_mask].reshape(-1, 3).clamp(0.0, 1.0)
    opacities = gs_map[..., 3:4][valid_mask].reshape(-1, 1).clamp(1e-6, 1.0 - 1e-6)
    scales = gs_map[..., 4:7][valid_mask].reshape(-1, 3).clamp_min(1e-6)
    quats = F.normalize(gs_map[..., 7:11][valid_mask].reshape(-1, 4), dim=-1)
    gs_conf_flat = gs_conf[valid_mask].reshape(-1)
    dynamic_flat = dynamic_prob[valid_mask].reshape(-1)

    source_image_ids = torch.arange(seq_len, dtype=torch.long).view(seq_len, 1, 1).expand(seq_len, height, width)
    source_image_ids = source_image_ids[valid_mask].reshape(-1)
    num_views = int(sample["cam_ids"].numel())
    source_frame_ids = torch.div(source_image_ids, num_views, rounding_mode="floor")
    source_view_ids = torch.remainder(source_image_ids, num_views)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.long),
        torch.arange(width, dtype=torch.long),
        indexing="ij",
    )
    source_y = yy.view(1, height, width).expand(seq_len, -1, -1)[valid_mask].reshape(-1)
    source_x = xx.view(1, height, width).expand(seq_len, -1, -1)[valid_mask].reshape(-1)

    return CleanSceneState(
        images=images,
        means=means,
        colors=colors,
        opacities=opacities,
        scales=scales,
        quats=quats,
        gs_conf=gs_conf_flat,
        dynamic_prob=dynamic_flat,
        source_image_ids=source_image_ids,
        source_frame_ids=source_frame_ids,
        source_view_ids=source_view_ids,
        source_y=source_y,
        source_x=source_x,
        point_map_world=point_map_world,
        valid_mask=valid_mask,
        depth=depth,
        semantic_vehicle_prob=semantic_vehicle_prob,
        semantic_vehicle_mask=semantic_vehicle_mask,
        camera_to_world=camera_to_world,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
    )


def _orthonormalize_rotation(rotation: torch.Tensor) -> torch.Tensor:
    # Preserve the semantic column layout [front, left, up]. SVD is unstable
    # here because true rotation matrices have repeated singular values.
    front = rotation[:, 0]
    up = rotation[:, 2]

    front = front / front.norm().clamp_min(1e-6)
    up = up - front * torch.dot(up, front)
    if float(up.norm()) < 1e-6:
        left_seed = rotation[:, 1]
        up = torch.linalg.cross(front, left_seed)
    up = up / up.norm().clamp_min(1e-6)

    left = torch.linalg.cross(up, front)
    left = left / left.norm().clamp_min(1e-6)
    up = torch.linalg.cross(front, left)
    up = up / up.norm().clamp_min(1e-6)
    return torch.stack([front, left, up], dim=1)


def _estimate_sim3_umeyama(source: torch.Tensor, target: torch.Tensor) -> Sim3Transform:
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError(f"Expected matched Nx3 points, got {tuple(source.shape)} and {tuple(target.shape)}")

    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / float(source.shape[0])
    u, singular_values, v = torch.linalg.svd(covariance)
    det_sign = torch.det(u @ v)
    diag = torch.eye(3, dtype=source.dtype)
    if det_sign < 0:
        diag[-1, -1] = -1.0

    rotation = u @ diag @ v
    var_source = (source_centered.pow(2).sum() / float(source.shape[0])).clamp_min(1e-8)
    scale = float((singular_values * diag.diagonal()).sum() / var_source)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (source @ rotation.T) + translation
    error = float((aligned - target).norm(dim=1).mean())

    return Sim3Transform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        mean_alignment_error=error,
    )


def _estimate_rotation_wahba(source_vectors: torch.Tensor, target_vectors: torch.Tensor) -> torch.Tensor:
    if source_vectors.shape != target_vectors.shape or source_vectors.shape[-1] != 3:
        raise ValueError(
            f"Expected matched Nx3 direction vectors, got {tuple(source_vectors.shape)} and {tuple(target_vectors.shape)}"
        )
    finite = torch.isfinite(source_vectors).all(dim=-1) & torch.isfinite(target_vectors).all(dim=-1)
    source_vectors = source_vectors[finite].float()
    target_vectors = target_vectors[finite].float()
    if source_vectors.shape[0] < 2:
        return torch.eye(3, dtype=torch.float32)

    source_vectors = source_vectors / source_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target_vectors = target_vectors / target_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    covariance = target_vectors.T @ source_vectors / float(source_vectors.shape[0])
    u, _, vh = torch.linalg.svd(covariance)
    diag = torch.eye(3, dtype=torch.float32)
    if torch.det(u @ vh) < 0:
        diag[-1, -1] = -1.0
    return (u @ diag @ vh).float()


def _robust_scale_from_camera_centers(source_centers: torch.Tensor, target_centers: torch.Tensor) -> float:
    if source_centers.shape[0] < 2:
        return 1.0

    pair_i, pair_j = torch.triu_indices(source_centers.shape[0], source_centers.shape[0], offset=1)
    source_dist = (source_centers[pair_i] - source_centers[pair_j]).norm(dim=-1)
    target_dist = (target_centers[pair_i] - target_centers[pair_j]).norm(dim=-1)
    finite = torch.isfinite(source_dist) & torch.isfinite(target_dist)
    valid = finite & (source_dist > 1e-4) & (target_dist > 1e-6)
    ratios = target_dist[valid] / source_dist[valid].clamp_min(1e-6)
    if ratios.numel() > 0:
        scale = float(torch.median(ratios).item())
        if math.isfinite(scale) and scale > 0.0:
            return scale

    source_var = ((source_centers - source_centers.mean(0)) ** 2).sum()
    target_var = ((target_centers - target_centers.mean(0)) ** 2).sum()
    scale_fallback = torch.sqrt(target_var / source_var.clamp_min(1e-6))
    if torch.isfinite(scale_fallback) and float(scale_fallback.item()) > 0.0:
        return float(scale_fallback.item())
    return 1.0


def estimate_scene_alignment(sample: dict[str, Any], clean_state: CleanSceneState) -> Sim3Transform:
    gt_camera_to_world = sample["camera_to_world_corrected"].detach().cpu().float().view(-1, 4, 4)
    pred_camera_to_world = clean_state.camera_to_world.detach().cpu().float().view(-1, 4, 4)
    if gt_camera_to_world.shape[0] != pred_camera_to_world.shape[0]:
        raise ValueError(
            f"Camera count mismatch between GT and prediction: "
            f"{gt_camera_to_world.shape[0]} vs {pred_camera_to_world.shape[0]}"
        )

    gt_centers = gt_camera_to_world[:, :3, 3]
    pred_centers = pred_camera_to_world[:, :3, 3]
    if gt_centers.shape[0] > 1:
        gt_var = ((gt_centers - gt_centers.mean(0)) ** 2).sum()
        pred_var = ((pred_centers - pred_centers.mean(0)) ** 2).sum()
        scale_t = torch.sqrt(pred_var / gt_var.clamp_min(1e-6))
        if not torch.isfinite(scale_t) or float(scale_t.item()) <= 0.0:
            scale_t = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale_t = torch.tensor(1.0, dtype=torch.float32)
    scale = float(scale_t.item())

    # DGGT's reconstructed world gauge is anchored by the input view order.  In
    # practice, averaging camera rotations across the clip can rotate the whole
    # scene away from the first-frame gauge and visibly bias object headings.
    R_gt = gt_camera_to_world[0, :3, :3]
    R_pred = pred_camera_to_world[0, :3, :3]
    rotation = _orthonormalize_rotation(R_pred @ R_gt.T)

    translation = pred_centers[0] - scale * (rotation @ gt_centers[0])
    aligned = scale * (gt_centers @ rotation.T) + translation
    error = float((aligned - pred_centers).norm(dim=1).mean().item())

    return Sim3Transform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        mean_alignment_error=error,
    )


def build_box_corners(center: torch.Tensor, size: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    half = size * 0.5
    local = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=center.dtype,
        device=center.device,
    )
    local = local * half
    return local @ rotation.T + center


def project_world_points(
    points_world: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image_hw
    rotation_wc = camera_to_world[:3, :3]
    translation_wc = camera_to_world[:3, 3]
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ translation_wc

    points_cam = points_world @ rotation_cw.T + translation_cw
    depths = points_cam[:, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = fx * points_cam[:, 0] / depths.clamp_min(1e-6) + cx
    v = fy * points_cam[:, 1] / depths.clamp_min(1e-6) + cy
    valid = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(depths)
        & (depths > 1e-4)
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )
    return torch.stack([u, v], dim=-1), depths, valid


def _project_world_points_unclipped(
    points_world: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points_cam = _world_to_camera_points(points_world, camera_to_world)
    depths = points_cam[:, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    z = depths.clamp_min(1e-6)
    u = fx * points_cam[:, 0] / z + cx
    v = fy * points_cam[:, 1] / z + cy
    valid = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(depths) & (depths > eps)
    return torch.stack([u, v], dim=-1), depths, valid


def _model_resize_geometry(
    raw_hw: tuple[int, int],
    model_hw: tuple[int, int],
    target_width: int | None = None,
) -> dict[str, int | float]:
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    if raw_h <= 0 or raw_w <= 0 or model_h <= 0 or model_w <= 0:
        raise ValueError(f"Invalid resize geometry raw_hw={raw_hw}, model_hw={model_hw}")
    new_w = int(model_w if target_width is None else target_width)
    new_h = round(raw_h * (new_w / float(raw_w)) / 14.0) * 14
    crop_top = max((new_h - new_w) // 2, 0) if new_h > new_w else 0
    out_h = new_w if new_h > new_w else new_h
    out_w = new_w
    pad_top = max((model_h - out_h) // 2, 0)
    pad_left = max((model_w - out_w) // 2, 0)
    return {
        "scale_x": new_w / float(raw_w),
        "scale_y": new_h / float(raw_h),
        "crop_top": int(crop_top),
        "out_h": int(out_h),
        "out_w": int(out_w),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
    }


def _raw_intrinsics_to_model(
    intrinsics_raw: torch.Tensor,
    raw_hw: tuple[int, int],
    model_hw: tuple[int, int],
) -> torch.Tensor:
    intrinsics_raw = intrinsics_raw.detach().float()
    geom = _model_resize_geometry(raw_hw, model_hw, target_width=int(model_hw[1]))
    intrinsics_model = intrinsics_raw.clone()
    intrinsics_model[0, 0] = intrinsics_raw[0, 0] * float(geom["scale_x"])
    intrinsics_model[1, 1] = intrinsics_raw[1, 1] * float(geom["scale_y"])
    intrinsics_model[0, 2] = intrinsics_raw[0, 2] * float(geom["scale_x"]) + float(geom["pad_left"])
    intrinsics_model[1, 2] = (
        intrinsics_raw[1, 2] * float(geom["scale_y"])
        - float(geom["crop_top"])
        + float(geom["pad_top"])
    )
    return intrinsics_model


def project_waymo_box_corners_model(
    box_corners_world: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics_raw: torch.Tensor,
    raw_image_size_hw: torch.Tensor | tuple[int, int] | list[int],
    model_image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_hw_tensor = torch.as_tensor(raw_image_size_hw).view(-1)
    raw_hw = (int(raw_hw_tensor[0].item()), int(raw_hw_tensor[1].item()))
    intrinsics_model = _raw_intrinsics_to_model(intrinsics_raw, raw_hw, model_image_hw)
    return _project_world_points_unclipped(
        box_corners_world.detach().to(device=intrinsics_model.device, dtype=intrinsics_model.dtype),
        camera_to_world.detach().to(device=intrinsics_model.device, dtype=intrinsics_model.dtype),
        intrinsics_model,
    )


def project_dggt_box_corners_model(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    corners = build_box_corners(object_center, object_size, object_rotation)
    return _project_world_points_unclipped(corners, camera_to_world, intrinsics)


def _edge_midpoints(points: torch.Tensor) -> torch.Tensor:
    edge_indices = torch.tensor(_BOX_EDGES, dtype=torch.long, device=points.device)
    return 0.5 * (points.index_select(0, edge_indices[:, 0]) + points.index_select(0, edge_indices[:, 1]))


def project_world_box_corners(
    box_corners_world: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    uv, depths, valid = project_world_points(box_corners_world, camera_to_world, intrinsics, image_hw)
    return uv, valid & (depths > 1e-4)


def compute_bbox_from_projected_points(uv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor | None:
    if valid.sum().item() == 0:
        return None
    valid_uv = uv[valid]
    x1 = valid_uv[:, 0].min()
    y1 = valid_uv[:, 1].min()
    x2 = valid_uv[:, 0].max()
    y2 = valid_uv[:, 1].max()
    if bool((x2 <= x1).item()) or bool((y2 <= y1).item()):
        return None
    return torch.stack([x1, y1, x2, y2], dim=0).detach().float()


def box_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a.tolist()]
    bx1, by1, bx2, by2 = [float(v) for v in box_b.tolist()]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _score_projected_bbox(box_pred: torch.Tensor, box_target: torch.Tensor) -> float:
    iou = box_iou_xyxy(box_pred, box_target)
    pred_center = 0.5 * (box_pred[:2] + box_pred[2:])
    tgt_center = 0.5 * (box_target[:2] + box_target[2:])
    pred_size = (box_pred[2:] - box_pred[:2]).clamp_min(1.0)
    tgt_size = (box_target[2:] - box_target[:2]).clamp_min(1.0)
    center_penalty = float(((pred_center - tgt_center).abs() / tgt_size).mean())
    size_penalty = float((torch.log(pred_size / tgt_size)).abs().mean())
    return iou - 0.35 * center_penalty - 0.35 * size_penalty


def points_in_box(
    points: torch.Tensor,
    center: torch.Tensor,
    rotation: torch.Tensor,
    size: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    local = (points - center) @ rotation
    half = size * (0.5 * scale)
    return (local.abs() <= (half + 1e-5)).all(dim=-1)


def _points_in_box(
    points: torch.Tensor,
    center: torch.Tensor,
    rotation: torch.Tensor,
    size: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    return points_in_box(points, center, rotation, size, scale=scale)


def _transform_track_box(
    obj_to_world: torch.Tensor,
    box_size: torch.Tensor,
    transform: Sim3Transform,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj_to_world = _to_cpu_float_tensor(obj_to_world)
    box_size = _to_cpu_float_tensor(box_size).view(3)
    center_waymo = obj_to_world[:3, 3].view(1, 3)
    center_dggt = transform.apply_points(center_waymo)[0]
    rotation_waymo = obj_to_world[:3, :3]
    rotation_dggt = _orthonormalize_rotation(transform.rotation @ rotation_waymo)
    size_dggt = box_size * float(transform.scale)
    return center_dggt, size_dggt, rotation_dggt


def _extract_foreground_seed(
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
    box_xyxy: torch.Tensor,
    depth_quantile: float = 0.35,
    shrink_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = point_map_world.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 = max(0, min(width - 1, int(math.floor(x1))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x2))))
    y1 = max(0, min(height - 1, int(math.floor(y1))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y2))))

    sx1 = max(x1, min(width - 1, int(math.floor(x1 + bw * shrink_ratio))))
    sx2 = max(sx1 + 1, min(width, int(math.ceil(x2 - bw * shrink_ratio))))
    sy1 = max(y1, min(height - 1, int(math.floor(y1 + bh * shrink_ratio))))
    sy2 = max(sy1 + 1, min(height, int(math.ceil(y2 - bh * shrink_ratio))))

    inner_valid = valid_mask[sy1:sy2, sx1:sx2]
    inner_depth = depth_map[sy1:sy2, sx1:sx2]
    inner_values = inner_depth[inner_valid]
    if inner_values.numel() == 0:
        full_mask = torch.zeros((height, width), dtype=torch.bool)
        return torch.zeros((0, 3), dtype=torch.float32), full_mask

    depth_threshold = torch.quantile(inner_values, depth_quantile)
    box_mask = torch.zeros((height, width), dtype=torch.bool)
    box_mask[y1:y2, x1:x2] = True
    seed_mask = box_mask & valid_mask & (depth_map <= depth_threshold)
    if int(seed_mask.sum().item()) < 24:
        depth_threshold = torch.quantile(inner_values, min(0.6, max(depth_quantile, 0.5)))
        seed_mask = box_mask & valid_mask & (depth_map <= depth_threshold)
    if int(seed_mask.sum().item()) < 24:
        seed_mask = box_mask & valid_mask

    seed_points = point_map_world[seed_mask].reshape(-1, 3)
    return seed_points, seed_mask


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    mask_f = mask.to(torch.float32).view(1, 1, mask.shape[0], mask.shape[1])
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(mask_f, kernel_size=kernel, stride=1, padding=radius)
    return dilated[0, 0] > 0.5


def _erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    mask_f = mask.to(torch.float32).view(1, 1, mask.shape[0], mask.shape[1])
    kernel = radius * 2 + 1
    eroded = 1.0 - F.max_pool2d(1.0 - mask_f, kernel_size=kernel, stride=1, padding=radius)
    return eroded[0, 0] > 0.5


def _close_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return _erode_mask(_dilate_mask(mask, radius), radius)


def _open_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return _dilate_mask(_erode_mask(mask, radius), radius)


def _select_connected_component(mask: torch.Tensor, seed_mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().cpu().bool()
    seed_mask = seed_mask.detach().cpu().bool() & mask
    if int(mask.sum().item()) == 0:
        return mask
    height, width = mask.shape
    visited = torch.zeros_like(mask)
    best_mask = torch.zeros_like(mask)
    best_score = -1.0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seed_center = None
    if int(seed_mask.sum().item()) > 0:
        seed_pixels = torch.nonzero(seed_mask, as_tuple=False).float()
        seed_center = seed_pixels.mean(dim=0)

    for start_y, start_x in torch.nonzero(mask, as_tuple=False).tolist():
        if visited[start_y, start_x]:
            continue
        queue = deque([(start_y, start_x)])
        component_pixels: list[tuple[int, int]] = []
        visited[start_y, start_x] = True
        overlap = 0
        while queue:
            y, x = queue.popleft()
            component_pixels.append((y, x))
            if seed_mask[y, x]:
                overlap += 1
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

        component_mask = torch.zeros_like(mask)
        ys = torch.tensor([p[0] for p in component_pixels], dtype=torch.long)
        xs = torch.tensor([p[1] for p in component_pixels], dtype=torch.long)
        component_mask[ys, xs] = True
        area = float(len(component_pixels))
        if overlap > 0:
            score = 1e6 + overlap * 1000.0 + area
        elif seed_center is not None:
            component_center = torch.stack([ys.float().mean(), xs.float().mean()])
            score = area - float(torch.norm(component_center - seed_center).item()) * 100.0
        else:
            score = area
        if score > best_score:
            best_score = score
            best_mask = component_mask
    return best_mask


def _points_in_protected_boxes(
    points: torch.Tensor,
    protected_boxes: list[dict[str, torch.Tensor]],
    scale: float = 1.05,
) -> torch.Tensor:
    if points.shape[0] == 0 or len(protected_boxes) == 0:
        return torch.zeros((points.shape[0],), dtype=torch.bool)
    inside = torch.zeros((points.shape[0],), dtype=torch.bool)
    for box in protected_boxes:
        inside |= _points_in_box(
            points,
            box["center"],
            box["rotation"],
            box["size"],
            scale=scale,
        )
    return inside


def _collect_protected_boxes(
    sample: dict[str, Any],
    alignment: Sim3Transform,
    target_slot_idx: int,
    frame_idx: int,
    view_offset: int,
) -> list[dict[str, torch.Tensor]]:
    protected_boxes: list[dict[str, torch.Tensor]] = []
    total_objects = int(sample["object_track_valid_mask_selected"].shape[0])
    bbox_present = sample["object_bbox_present_mask_selected"]
    bbox_model = sample.get("object_bbox_model_selected")

    for slot_idx in range(total_objects):
        if slot_idx == target_slot_idx:
            continue
        if not bool(sample["object_valid_mask"][slot_idx].item()):
            continue
        if not bool(sample["object_track_valid_mask_selected"][slot_idx, frame_idx].item()):
            continue
        bbox_model_view = None
        if isinstance(bbox_present, torch.Tensor) and isinstance(bbox_model, torch.Tensor):
            if bool(bbox_present[slot_idx, frame_idx, view_offset].item()):
                bbox_model_view = bbox_model[slot_idx, frame_idx, view_offset].detach().cpu().float()
        center, size, rotation = _transform_track_box(
            sample["object_obj_to_world_selected"][slot_idx, frame_idx],
            sample["object_box_size_selected"][slot_idx, frame_idx],
            alignment,
        )
        protected_boxes.append(
            {
                "slot_idx": torch.tensor(slot_idx, dtype=torch.long),
                "scene_raw_object_id": str(sample["object_scene_raw_ids"][slot_idx]),
                "center": center,
                "size": size,
                "rotation": rotation,
                "bbox_model": bbox_model_view,
            }
        )
    return protected_boxes


def _extract_semantic_object_component_3d(
    clean_state: CleanSceneState,
    source_front_index: int,
    target_bbox_model: torch.Tensor,
    proposal_center: torch.Tensor,
    proposal_rotation: torch.Tensor,
    proposal_size: torch.Tensor,
    protected_boxes: list[dict[str, torch.Tensor]],
    proposal_scale: float,
    dynamic_mode: bool,
) -> dict[str, torch.Tensor]:
    semantic_vehicle_mask = clean_state.semantic_vehicle_mask[source_front_index]
    semantic_vehicle_prob = clean_state.semantic_vehicle_prob[source_front_index]
    valid_mask = clean_state.valid_mask[source_front_index]
    if int((semantic_vehicle_mask & valid_mask).sum().item()) == 0 and float(semantic_vehicle_prob.max().item()) < 0.10:
        empty_pixel = torch.zeros_like(valid_mask)
        empty_points = torch.zeros((0, 3), dtype=torch.float32)
        empty_local = torch.zeros((0,), dtype=torch.bool)
        return {"points": empty_points, "pixel_mask": empty_pixel, "local_mask": empty_local}

    image_hw = tuple(valid_mask.shape)
    bbox_inner_mask, bbox_outer_mask = _build_bbox_masks(
        target_bbox_model,
        image_hw,
        inner_ratio=0.08,
        outer_ratio=0.18,
    )
    semantic_raw = ((semantic_vehicle_mask | (semantic_vehicle_prob >= 0.22)) & valid_mask & bbox_outer_mask)
    semantic_raw = _close_mask(semantic_raw, radius=1)
    semantic_raw = _open_mask(semantic_raw, radius=1)

    seed_mask = semantic_raw & bbox_inner_mask
    if int(seed_mask.sum().item()) < 12:
        seed_mask = (semantic_vehicle_prob >= 0.30) & bbox_inner_mask & valid_mask
    if int(seed_mask.sum().item()) < 6:
        seed_mask = bbox_inner_mask & valid_mask

    semantic_component = _select_connected_component(semantic_raw | seed_mask, seed_mask)
    semantic_component = _close_mask(semantic_component, radius=1) & bbox_outer_mask & valid_mask

    in_view_mask = clean_state.source_image_ids == int(source_front_index)
    local_indices = torch.nonzero(in_view_mask, as_tuple=False).flatten()
    if local_indices.numel() == 0:
        empty_pixel = torch.zeros_like(valid_mask)
        empty_points = torch.zeros((0, 3), dtype=torch.float32)
        empty_local = torch.zeros((0,), dtype=torch.bool)
        return {"points": empty_points, "pixel_mask": empty_pixel, "local_mask": empty_local}

    local_points = clean_state.means[in_view_mask]
    local_y = clean_state.source_y[in_view_mask]
    local_x = clean_state.source_x[in_view_mask]
    local_depth = clean_state.depth[source_front_index][local_y, local_x]
    semantic_local = semantic_component[local_y, local_x] & torch.isfinite(local_points).all(dim=-1)
    if int(semantic_local.sum().item()) == 0:
        empty_pixel = torch.zeros_like(valid_mask)
        empty_points = torch.zeros((0, 3), dtype=torch.float32)
        return {"points": empty_points, "pixel_mask": empty_pixel, "local_mask": semantic_local}

    proposal_box_local = _points_in_box(
        local_points,
        proposal_center,
        proposal_rotation,
        proposal_size,
        scale=max(1.30, proposal_scale + 0.05),
    )
    box_corners = build_box_corners(
        proposal_center,
        proposal_size * max(1.0, proposal_scale),
        proposal_rotation,
    )
    box_cam = _world_to_camera_points(box_corners, clean_state.camera_to_world[source_front_index])
    depth_low = float(box_cam[:, 2].min().item()) - 0.08
    depth_high = float(box_cam[:, 2].max().item()) + 0.08
    depth_local = (local_depth >= depth_low) & (local_depth <= depth_high)

    protected_local = _points_in_protected_boxes(local_points, protected_boxes, scale=1.08)
    target_core_local = _points_in_box(local_points, proposal_center, proposal_rotation, proposal_size, scale=1.05)

    semantic_local = semantic_local & proposal_box_local & depth_local
    semantic_local = semantic_local & (~protected_local | target_core_local)
    if int(semantic_local.sum().item()) == 0:
        empty_pixel = torch.zeros_like(valid_mask)
        empty_points = torch.zeros((0, 3), dtype=torch.float32)
        return {"points": empty_points, "pixel_mask": empty_pixel, "local_mask": semantic_local}

    seed_local = semantic_local & bbox_inner_mask[local_y, local_x]
    if int(seed_local.sum().item()) < 6:
        seed_local = semantic_local
    voxel_scale = proposal_size * torch.tensor(
        _DYNAMIC_VOXEL_SCALE if dynamic_mode else _STATIC_VOXEL_SCALE,
        dtype=proposal_size.dtype,
    )
    voxel_scale = voxel_scale.clamp_min(0.03)
    semantic_points = local_points[semantic_local]
    semantic_seed = seed_local[semantic_local]
    keep_mask, _ = _voxel_connected_component(
        semantic_points,
        semantic_seed,
        voxel_scale,
        connectivity=6,
    )
    if int(keep_mask.sum().item()) == 0:
        empty_pixel = torch.zeros_like(valid_mask)
        empty_points = torch.zeros((0, 3), dtype=torch.float32)
        return {"points": empty_points, "pixel_mask": empty_pixel, "local_mask": semantic_local.new_zeros(semantic_local.shape)}

    semantic_keep_local = torch.zeros_like(semantic_local)
    semantic_keep_local[semantic_local] = keep_mask
    semantic_pixel_mask = torch.zeros_like(valid_mask)
    kept_global = local_indices[semantic_keep_local]
    semantic_pixel_mask[clean_state.source_y[kept_global], clean_state.source_x[kept_global]] = True
    return {
        "points": local_points[semantic_keep_local],
        "pixel_mask": semantic_pixel_mask,
        "local_mask": semantic_keep_local,
    }


def _build_bbox_masks(
    box_xyxy: torch.Tensor,
    image_hw: tuple[int, int],
    inner_ratio: float = 0.08,
    outer_ratio: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_hw
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    bbox_inner_mask = torch.zeros((height, width), dtype=torch.bool)
    bbox_outer_mask = torch.zeros((height, width), dtype=torch.bool)
    ix1 = max(0, min(width - 1, int(math.floor(x1 + inner_ratio * bw))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2 - inner_ratio * bw))))
    iy1 = max(0, min(height - 1, int(math.floor(y1 + inner_ratio * bh))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2 - inner_ratio * bh))))
    ox1 = max(0, min(width - 1, int(math.floor(x1 - outer_ratio * bw))))
    ox2 = max(ox1 + 1, min(width, int(math.ceil(x2 + outer_ratio * bw))))
    oy1 = max(0, min(height - 1, int(math.floor(y1 - outer_ratio * bh))))
    oy2 = max(oy1 + 1, min(height, int(math.ceil(y2 + outer_ratio * bh))))
    bbox_inner_mask[iy1:iy2, ix1:ix2] = True
    bbox_outer_mask[oy1:oy2, ox1:ox2] = True
    return bbox_inner_mask, bbox_outer_mask


def _grow_pixel_component_from_seed(
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    allowed_mask: torch.Tensor,
    seed_mask: torch.Tensor,
    xyz_thresh: float,
    depth_thresh: float,
) -> torch.Tensor:
    allowed_mask = allowed_mask.detach().cpu().bool()
    seed_mask = (seed_mask.detach().cpu().bool()) & allowed_mask
    if int(seed_mask.sum().item()) == 0:
        return torch.zeros_like(allowed_mask)

    point_map_world = point_map_world.detach().cpu().float()
    depth_map = depth_map.detach().cpu().float()
    height, width = allowed_mask.shape
    out_mask = torch.zeros_like(allowed_mask)
    queue = deque((int(y), int(x)) for y, x in torch.nonzero(seed_mask, as_tuple=False).tolist())
    out_mask[seed_mask] = True
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()
        point = point_map_world[y, x]
        depth = float(depth_map[y, x].item())
        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            if out_mask[ny, nx] or not allowed_mask[ny, nx]:
                continue
            neighbor_depth = float(depth_map[ny, nx].item())
            if abs(neighbor_depth - depth) > depth_thresh:
                continue
            if float(torch.norm(point_map_world[ny, nx] - point).item()) > xyz_thresh:
                continue
            out_mask[ny, nx] = True
            queue.append((ny, nx))
    return out_mask


def _extract_object_pixels_from_bbox_component(
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
    box_xyxy: torch.Tensor,
    dynamic_image_mask: torch.Tensor | None = None,
    dynamic_mode: bool = False,
    coarse_center: torch.Tensor | None = None,
    coarse_rotation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_hw = tuple(point_map_world.shape[:2])
    bbox_inner_mask, bbox_outer_mask = _build_bbox_masks(box_xyxy, image_hw)
    seed_points, seed_mask = _extract_foreground_seed(
        point_map_world,
        depth_map,
        valid_mask,
        box_xyxy,
        depth_quantile=0.30 if dynamic_mode else 0.22,
        shrink_ratio=0.10 if dynamic_mode else 0.16,
    )
    height, width = image_hw
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    sx1 = max(0, min(width - 1, int(math.floor(x1 + 0.12 * bw))))
    sx2 = max(sx1 + 1, min(width, int(math.ceil(x2 - 0.12 * bw))))
    sy1 = max(0, min(height - 1, int(math.floor(y1 + 0.08 * bh))))
    sy2 = max(sy1 + 1, min(height, int(math.ceil(y2 - 0.28 * bh))))
    seed_region_mask = torch.zeros((height, width), dtype=torch.bool)
    seed_region_mask[sy1:sy2, sx1:sx2] = True

    allowed_mask = bbox_outer_mask & valid_mask
    if dynamic_mode and dynamic_image_mask is not None:
        dynamic_outer = _dilate_mask(dynamic_image_mask & bbox_outer_mask, radius=2)
        allowed_mask = allowed_mask & (dynamic_outer | bbox_inner_mask | _dilate_mask(seed_mask, radius=2))
    region_valid = seed_region_mask & valid_mask
    region_depths = depth_map[region_valid]
    if region_depths.numel() > 0:
        region_q = 0.40 if dynamic_mode else 0.32
        region_depth_thresh = torch.quantile(region_depths, region_q)
        biased_seed_mask = region_valid & (depth_map <= region_depth_thresh) & allowed_mask
    else:
        biased_seed_mask = seed_mask & seed_region_mask & allowed_mask
    if int(biased_seed_mask.sum().item()) >= 16:
        seed_mask = biased_seed_mask
    else:
        seed_mask = seed_mask & allowed_mask
    if int(seed_mask.sum().item()) < 24:
        seed_mask = seed_region_mask & valid_mask
    if int(seed_mask.sum().item()) < 24:
        seed_mask = bbox_inner_mask & valid_mask
        if dynamic_mode and dynamic_image_mask is not None:
            seed_mask = seed_mask & (_dilate_mask(dynamic_image_mask, radius=1) | bbox_inner_mask)
    if int(seed_mask.sum().item()) < 24:
        fallback_mask = bbox_inner_mask & valid_mask
        return point_map_world[fallback_mask].reshape(-1, 3), fallback_mask

    if seed_points.shape[0] > 0:
        visible_extent = (seed_points.max(dim=0).values - seed_points.min(dim=0).values).clamp_min(1e-4)
        xyz_thresh = float(visible_extent.max().item()) * (1.6 if dynamic_mode else 1.35)
        depth_thresh = float(visible_extent.norm().item()) * (0.9 if dynamic_mode else 0.7)
    else:
        xyz_thresh = 0.04
        depth_thresh = 0.06
    xyz_thresh = max(0.02, min(0.12, xyz_thresh))
    depth_thresh = max(0.03, min(0.16, depth_thresh))

    object_mask = _grow_pixel_component_from_seed(
        point_map_world=point_map_world,
        depth_map=depth_map,
        allowed_mask=allowed_mask,
        seed_mask=seed_mask,
        xyz_thresh=xyz_thresh,
        depth_thresh=depth_thresh,
    )

    max_growth_ratio = 5.0 if dynamic_mode else 4.0
    if int(object_mask.sum().item()) > int(max(seed_mask.sum().item(), 1) * max_growth_ratio):
        object_mask = _dilate_mask(seed_mask, radius=1) & allowed_mask
    eroded_mask = _erode_mask(object_mask, radius=1)
    if int(eroded_mask.sum().item()) >= 16:
        selected_eroded = _select_connected_component(eroded_mask, seed_mask)
        if int(selected_eroded.sum().item()) >= 16:
            object_mask = _select_connected_component(object_mask, selected_eroded)
    if (
        coarse_center is not None
        and coarse_rotation is not None
        and int(object_mask.sum().item()) >= 24
    ):
        object_points = point_map_world[object_mask].reshape(-1, 3)
        local_coords = (object_points - coarse_center) @ coarse_rotation
        local_up = local_coords[:, 2]
        up_low = torch.quantile(local_up, 0.05)
        up_high = torch.quantile(local_up, 0.95)
        up_margin = max(0.006, float((up_high - up_low).item()) * 0.12)
        keep_object = local_up >= (up_low + up_margin)
        if int(keep_object.sum().item()) >= max(24, int(object_points.shape[0] * 0.55)):
            filtered_mask = torch.zeros_like(object_mask)
            object_pixels = torch.nonzero(object_mask, as_tuple=False)
            kept_pixels = object_pixels[keep_object]
            filtered_mask[kept_pixels[:, 0], kept_pixels[:, 1]] = True
            object_mask = filtered_mask
    if int(object_mask.sum().item()) < 24:
        fallback_mask = bbox_inner_mask & valid_mask
        return point_map_world[fallback_mask].reshape(-1, 3), fallback_mask
    return point_map_world[object_mask].reshape(-1, 3), object_mask


def _extract_object_pixels_from_3d_box(
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
    box_xyxy: torch.Tensor,
    box_center: torch.Tensor,
    box_rotation: torch.Tensor,
    box_size: torch.Tensor,
    box_scale: float = 1.15,
    expand_ratio: float = 0.08,
    min_points: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = point_map_world.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 = max(0, min(width - 1, int(math.floor(x1 - bw * expand_ratio))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x2 + bw * expand_ratio))))
    y1 = max(0, min(height - 1, int(math.floor(y1 - bh * expand_ratio))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y2 + bh * expand_ratio))))

    pixel_mask = torch.zeros((height, width), dtype=torch.bool)
    patch_valid = valid_mask[y1:y2, x1:x2]
    if patch_valid.any():
        patch_points = point_map_world[y1:y2, x1:x2]
        valid_points = patch_points[patch_valid].reshape(-1, 3)
        inside = _points_in_box(
            valid_points,
            box_center,
            box_rotation,
            box_size,
            scale=box_scale,
        )
        patch_mask = torch.zeros_like(patch_valid)
        patch_mask[patch_valid] = inside
        pixel_mask[y1:y2, x1:x2] = patch_mask

    pixel_count_before_fallback = int(pixel_mask.sum().item())
    if pixel_count_before_fallback < min_points:
        print("[gaussian_edit][object_pixels] fallback_to_foreground_seed")
        seed_points, seed_mask = _extract_foreground_seed(
            point_map_world,
            depth_map,
            valid_mask,
            box_xyxy,
            depth_quantile=0.5,
            shrink_ratio=0.0,
        )
        pixel_mask |= seed_mask
    else:
        seed_points = point_map_world[pixel_mask].reshape(-1, 3)

    pixel_mask = _dilate_mask(pixel_mask, radius=2)
    return point_map_world[pixel_mask].reshape(-1, 3), pixel_mask


def _rotation_z(angle_rad: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def _rotation_z_tensor(angle_rad: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    zero = torch.zeros((), dtype=dtype, device=angle_rad.device)
    one = torch.ones((), dtype=dtype, device=angle_rad.device)
    return torch.stack(
        [
            torch.stack([c.to(dtype), -s.to(dtype), zero]),
            torch.stack([s.to(dtype), c.to(dtype), zero]),
            torch.stack([zero, zero, one]),
        ],
        dim=0,
    )


def _so3_exp_map(rotvec: torch.Tensor) -> torch.Tensor:
    dtype = rotvec.dtype
    device = rotvec.device
    theta = torch.linalg.norm(rotvec).clamp_min(1e-8)
    wx, wy, wz = rotvec[0], rotvec[1], rotvec[2]
    zero = torch.zeros((), dtype=dtype, device=device)
    skew = torch.stack(
        [
            torch.stack([zero, -wz, wy]),
            torch.stack([wz, zero, -wx]),
            torch.stack([-wy, wx, zero]),
        ],
        dim=0,
    )
    identity = torch.eye(3, dtype=dtype, device=device)
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / theta.pow(2)
    return identity + a * skew + b * (skew @ skew)


def _rotation_delta_magnitude_rad(base_rotation: torch.Tensor, rotation: torch.Tensor) -> float:
    relative = base_rotation.T @ rotation
    trace = torch.trace(relative)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.acos(cos_theta).item())


def _build_upright_rotation(front_dir: torch.Tensor, up_dir: torch.Tensor) -> torch.Tensor:
    up = up_dir / up_dir.norm().clamp_min(1e-6)
    front = front_dir - up * torch.dot(front_dir, up)
    if float(front.norm()) < 1e-6:
        front = torch.tensor([1.0, 0.0, 0.0], dtype=up.dtype)
        front = front - up * torch.dot(front, up)
    front = front / front.norm().clamp_min(1e-6)
    left = torch.linalg.cross(up, front)
    left = left / left.norm().clamp_min(1e-6)
    front = torch.linalg.cross(left, up)
    front = front / front.norm().clamp_min(1e-6)
    return torch.stack([front, left, up], dim=1)


def _build_label_track_rotation(gt_rotation: torch.Tensor, scene_up: torch.Tensor) -> torch.Tensor:
    front = gt_rotation[:, 0]
    front = front - scene_up * torch.dot(front, scene_up)
    if float(front.norm()) < 1e-6:
        front = gt_rotation[:, 0]
    up = gt_rotation[:, 2]
    if float(up.norm()) < 1e-6:
        up = scene_up
    return _build_upright_rotation(front, up)


def _yaw_delta_between_rotations(base_rotation: torch.Tensor, solved_rotation: torch.Tensor) -> float:
    relative = base_rotation.T @ solved_rotation
    return float(torch.atan2(relative[1, 0], relative[0, 0]).item())


def _robust_shared_yaw_delta(yaw_deltas: list[float], scores: list[float]) -> float:
    if len(yaw_deltas) == 0:
        return 0.0
    if len(yaw_deltas) == 1:
        return float(yaw_deltas[0])

    angles = torch.tensor(yaw_deltas, dtype=torch.float32)
    angles = _wrap_angle_rad(angles)
    anchor = torch.median(angles)
    diffs = _wrap_angle_rad(angles - anchor).abs()
    keep_count = max(1, math.ceil(0.75 * len(yaw_deltas)))
    keep_indices = torch.argsort(diffs)[:keep_count]
    kept_angles = angles[keep_indices]

    if len(scores) == len(yaw_deltas):
        weights = torch.tensor(scores, dtype=torch.float32)[keep_indices]
        weights = torch.softmax(weights, dim=0)
    else:
        weights = torch.full((keep_indices.numel(),), 1.0 / float(max(1, keep_indices.numel())), dtype=torch.float32)

    sin_sum = torch.sum(weights * torch.sin(kept_angles))
    cos_sum = torch.sum(weights * torch.cos(kept_angles))
    return float(torch.atan2(sin_sum, cos_sum).item())


def _rotation_heading_angle(rotation: torch.Tensor) -> float:
    front = rotation[:, 0]
    return float(torch.atan2(front[1], front[0]).item())


def _shared_track_rotation_candidate(
    rotations: list[torch.Tensor],
    scores: list[float],
    max_spread_deg: float = 8.0,
) -> torch.Tensor | None:
    if len(rotations) < 2:
        return None
    headings = torch.tensor([_rotation_heading_angle(rot) for rot in rotations], dtype=torch.float32)
    shared_heading = _robust_shared_yaw_delta(headings.tolist(), scores)
    diffs = _wrap_angle_rad(headings - shared_heading).abs()
    spread_deg = float(torch.rad2deg(diffs.max()).item())
    if spread_deg > max_spread_deg:
        return None
    inlier_indices = torch.nonzero(diffs <= math.radians(max_spread_deg), as_tuple=False).flatten()
    if inlier_indices.numel() == 0:
        return None
    best_local = int(torch.tensor(scores, dtype=torch.float32)[inlier_indices].argmax().item())
    best_index = int(inlier_indices[best_local].item())
    return rotations[best_index].clone()


def _shared_track_yaw_delta(
    yaw_deltas: list[float],
    scores: list[float],
    max_yaw_rad: float,
    max_spread_deg: float = 18.0,
) -> float:
    if max_yaw_rad <= 0.0 or len(yaw_deltas) == 0:
        return 0.0
    finite_items = [
        (float(yaw), float(score))
        for yaw, score in zip(yaw_deltas, scores)
        if math.isfinite(float(yaw)) and math.isfinite(float(score))
    ]
    if len(finite_items) == 0:
        return 0.0
    if len(finite_items) == 1:
        yaw = finite_items[0][0]
        return max(-max_yaw_rad, min(max_yaw_rad, yaw))

    yaws = [item[0] for item in finite_items]
    finite_scores = [item[1] for item in finite_items]
    shared = _robust_shared_yaw_delta(yaws, finite_scores)
    angles = _wrap_angle_rad(torch.tensor(yaws, dtype=torch.float32))
    diffs = _wrap_angle_rad(angles - float(shared)).abs()
    inlier_thresh = math.radians(float(max_spread_deg))
    inlier_count = int((diffs <= inlier_thresh).sum().item())
    min_inliers = max(2, math.ceil(0.5 * len(yaws)))
    if inlier_count < min_inliers:
        return 0.0
    return max(-max_yaw_rad, min(max_yaw_rad, float(shared)))


def _world_to_camera_points(points_world: torch.Tensor, camera_to_world: torch.Tensor) -> torch.Tensor:
    rotation_wc = camera_to_world[:3, :3]
    translation_wc = camera_to_world[:3, 3]
    rotation_cw = rotation_wc.T
    translation_cw = -rotation_cw @ translation_wc
    return points_world @ rotation_cw.T + translation_cw


def _camera_to_world_points(points_cam: torch.Tensor, camera_to_world: torch.Tensor) -> torch.Tensor:
    rotation_wc = camera_to_world[:3, :3]
    translation_wc = camera_to_world[:3, 3]
    return points_cam @ rotation_wc.T + translation_wc


def _unproject_center_xy(target_bbox_model: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    target_center_2d = 0.5 * (target_bbox_model[:2] + target_bbox_model[2:])
    fx = intrinsics[0, 0].clamp_min(1e-6)
    fy = intrinsics[1, 1].clamp_min(1e-6)
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    return torch.stack(
        [
            (target_center_2d[0] - cx) * depth / fx,
            (target_center_2d[1] - cy) * depth / fy,
        ],
        dim=0,
    )


def _project_box_bbox_soft(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    corners = build_box_corners(object_center, object_size, object_rotation)
    uv, depths, _ = project_world_points(corners, camera_to_world, intrinsics, image_hw)
    bbox = torch.stack(
        [
            uv[:, 0].min(),
            uv[:, 1].min(),
            uv[:, 0].max(),
            uv[:, 1].max(),
        ],
        dim=0,
    )
    invalid_penalty = F.relu(1e-3 - depths).mean()
    return bbox, invalid_penalty


def _bbox_alignment_loss(pred_bbox: torch.Tensor, target_bbox: torch.Tensor) -> torch.Tensor:
    target_size = (target_bbox[2:] - target_bbox[:2]).clamp_min(1.0)
    pred_size = (pred_bbox[2:] - pred_bbox[:2]).clamp_min(1.0)
    pred_center = 0.5 * (pred_bbox[:2] + pred_bbox[2:])
    target_center = 0.5 * (target_bbox[:2] + target_bbox[2:])
    edge_scale = torch.stack([target_size[0], target_size[1], target_size[0], target_size[1]])
    edge_loss = ((pred_bbox - target_bbox) / edge_scale).abs().mean()
    center_loss = ((pred_center - target_center) / target_size).abs().mean()
    size_loss = (torch.log(pred_size / target_size)).abs().mean()
    return edge_loss + 0.5 * center_loss + 0.35 * size_loss


def _project_box_bbox(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor | None:
    projected_corners, projected_valid = project_world_box_corners(
        build_box_corners(object_center, object_size, object_rotation),
        camera_to_world,
        intrinsics,
        image_hw,
    )
    return compute_bbox_from_projected_points(projected_corners, projected_valid)


def _project_box_edge_midpoints_model(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    corners = build_box_corners(object_center, object_size, object_rotation)
    return _project_world_points_unclipped(_edge_midpoints(corners), camera_to_world, intrinsics)


def _safe_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _box_area_xyxy(box: torch.Tensor | None) -> float | None:
    if box is None:
        return None
    width = max(0.0, float((box[2] - box[0]).item()))
    height = max(0.0, float((box[3] - box[1]).item()))
    return width * height


def _box_size_xyxy(box: torch.Tensor | None) -> tuple[float, float] | None:
    if box is None:
        return None
    return (
        max(0.0, float((box[2] - box[0]).item())),
        max(0.0, float((box[3] - box[1]).item())),
    )


def _corner_projection_rms(
    pred_uv: torch.Tensor,
    pred_valid: torch.Tensor,
    target_uv: torch.Tensor,
    target_valid: torch.Tensor,
    min_count: int = 4,
) -> tuple[float, int]:
    common = pred_valid.bool() & target_valid.bool() & torch.isfinite(pred_uv).all(dim=-1) & torch.isfinite(target_uv).all(dim=-1)
    count = int(common.sum().item())
    if count < min_count:
        return float("inf"), count
    residual = pred_uv[common] - target_uv[common]
    return float(torch.sqrt(residual.pow(2).sum(dim=-1).mean()).item()), count


def _view_half_extent(
    view_dir_world: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
) -> torch.Tensor:
    view_dir = view_dir_world / view_dir_world.norm().clamp_min(1e-6)
    view_local = torch.matmul(view_dir.view(1, 3), object_rotation).view(3)
    return torch.sum(view_local.abs() * (0.5 * object_size)).clamp_min(1e-4)


def _delete_support_depth_prior(
    support_points_world: torch.Tensor | None,
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    center_cam = _world_to_camera_points(object_center.view(1, 3), camera_to_world)[0]
    fallback = center_cam[2].clamp_min(1e-3)
    if support_points_world is None or support_points_world.shape[0] < 8:
        return fallback, None

    support = support_points_world.detach().to(device=object_center.device, dtype=object_center.dtype)
    support_cam = _world_to_camera_points(support, camera_to_world)
    finite = torch.isfinite(support_cam).all(dim=-1) & (support_cam[:, 2] > 1e-4)
    if int(finite.sum().item()) < 8:
        return fallback, None

    support = support[finite]
    support_cam = support_cam[finite]
    visible_depth = torch.median(support_cam[:, 2]).clamp_min(1e-3)
    support_mean = support.mean(dim=0)
    camera_origin = camera_to_world[:3, 3]
    view_dir = support_mean - camera_origin
    half_extent = _view_half_extent(view_dir, object_size, object_rotation)
    center_depth_prior = (visible_depth + half_extent).clamp_min(1e-3)
    support_center_prior = support_mean + view_dir / view_dir.norm().clamp_min(1e-6) * half_extent
    return center_depth_prior, support_center_prior


def _corner_pose_metrics(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    waymo_corner_uv: torch.Tensor,
    waymo_corner_valid: torch.Tensor,
    waymo_edge_uv: torch.Tensor,
    waymo_edge_valid: torch.Tensor,
    target_bbox_model: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
) -> dict[str, Any]:
    corner_uv, corner_depths, corner_valid = project_dggt_box_corners_model(
        object_center,
        object_size,
        object_rotation,
        camera_to_world,
        intrinsics,
    )
    edge_uv, edge_depths, edge_valid = _project_box_edge_midpoints_model(
        object_center,
        object_size,
        object_rotation,
        camera_to_world,
        intrinsics,
    )
    corner_rms, corner_count = _corner_projection_rms(
        corner_uv,
        corner_valid,
        waymo_corner_uv,
        waymo_corner_valid,
        min_count=4,
    )
    edge_rms, edge_count = _corner_projection_rms(
        edge_uv,
        edge_valid,
        waymo_edge_uv,
        waymo_edge_valid,
        min_count=4,
    )
    bbox = compute_bbox_from_projected_points(corner_uv, corner_valid)
    bbox_iou = 0.0 if bbox is None else box_iou_xyxy(bbox, target_bbox_model)
    center_depth = _world_to_camera_points(object_center.view(1, 3), camera_to_world)[0, 2]
    return {
        "corner_uv": corner_uv,
        "corner_depths": corner_depths,
        "corner_valid": corner_valid,
        "edge_uv": edge_uv,
        "edge_depths": edge_depths,
        "edge_valid": edge_valid,
        "target_corner_uv": waymo_corner_uv,
        "target_corner_valid": waymo_corner_valid,
        "corner_rms": corner_rms,
        "corner_count": corner_count,
        "edge_rms": edge_rms,
        "edge_count": edge_count,
        "bbox": bbox,
        "bbox_iou": float(bbox_iou),
        "center_depth": float(center_depth.item()),
    }


def _serialize_corner_pose_diagnostics(
    *,
    status: str,
    accepted: bool,
    reason: str,
    before: dict[str, Any],
    after: dict[str, Any],
    candidate_yaw_delta: float,
    used_yaw_delta: float,
    base_center: torch.Tensor,
    used_center: torch.Tensor,
    candidate_center: torch.Tensor,
    base_size: torch.Tensor | None = None,
    used_size: torch.Tensor | None = None,
    candidate_size: torch.Tensor | None = None,
    shared_yaw_delta: float | None = None,
) -> dict[str, Any]:
    depth_before = _safe_float(before.get("center_depth"))
    depth_after = _safe_float(after.get("center_depth"))
    depth_ratio = None
    if depth_before is not None and depth_after is not None and abs(depth_before) > 1e-6:
        depth_ratio = depth_after / depth_before
    center_shift = float((used_center - base_center).norm().item())
    candidate_center_shift = float((candidate_center - base_center).norm().item())
    bbox_area_before = _box_area_xyxy(before.get("bbox"))
    bbox_area_after = _box_area_xyxy(after.get("bbox"))
    bbox_area_ratio = None
    if bbox_area_before is not None and bbox_area_after is not None and bbox_area_before > 1e-6:
        bbox_area_ratio = bbox_area_after / bbox_area_before
    diagnostics = {
        "corner_refine_status": status,
        "corner_refine_accepted": bool(accepted),
        "corner_refine_reason": reason,
        "corner_rms_before": _safe_float(before.get("corner_rms")),
        "corner_rms_after": _safe_float(after.get("corner_rms")),
        "edge_rms_before": _safe_float(before.get("edge_rms")),
        "edge_rms_after": _safe_float(after.get("edge_rms")),
        "bbox_iou_before": _safe_float(before.get("bbox_iou")),
        "bbox_iou_after": _safe_float(after.get("bbox_iou")),
        "yaw_delta_rad": float(used_yaw_delta),
        "yaw_delta_deg": float(math.degrees(used_yaw_delta)),
        "candidate_yaw_delta_rad": float(candidate_yaw_delta),
        "candidate_yaw_delta_deg": float(math.degrees(candidate_yaw_delta)),
        "center_shift": center_shift,
        "candidate_center_shift": candidate_center_shift,
        "depth_before": depth_before,
        "depth_after": depth_after,
        "depth_ratio": depth_ratio,
        "bbox_area_before": bbox_area_before,
        "bbox_area_after": bbox_area_after,
        "bbox_area_ratio": bbox_area_ratio,
        "valid_corner_count": int(after.get("corner_count", 0)),
        "valid_edge_count": int(after.get("edge_count", 0)),
    }
    before_corner_uv = before.get("corner_uv")
    after_corner_uv = after.get("corner_uv")
    waymo_corner_uv = after.get("target_corner_uv")
    before_corner_valid = before.get("corner_valid")
    after_corner_valid = after.get("corner_valid")
    waymo_corner_valid = after.get("target_corner_valid")
    if (
        isinstance(before_corner_uv, torch.Tensor)
        and isinstance(after_corner_uv, torch.Tensor)
        and isinstance(waymo_corner_uv, torch.Tensor)
        and isinstance(before_corner_valid, torch.Tensor)
        and isinstance(after_corner_valid, torch.Tensor)
        and isinstance(waymo_corner_valid, torch.Tensor)
    ):
        before_common = before_corner_valid.bool() & waymo_corner_valid.bool()
        after_common = after_corner_valid.bool() & waymo_corner_valid.bool()
        before_residual = before_corner_uv.detach().cpu().float() - waymo_corner_uv.detach().cpu().float()
        after_residual = after_corner_uv.detach().cpu().float() - waymo_corner_uv.detach().cpu().float()
        before_err = torch.linalg.norm(before_residual, dim=-1)
        after_err = torch.linalg.norm(after_residual, dim=-1)
        if bool(before_common.any().item()):
            diagnostics["corner_residual_max_before"] = float(before_err[before_common.cpu()].max().item())
            diagnostics["corner_residual_mean_xy_before"] = [
                float(v) for v in before_residual[before_common.cpu()].mean(dim=0).tolist()
            ]
        if bool(after_common.any().item()):
            diagnostics["corner_residual_max_after"] = float(after_err[after_common.cpu()].max().item())
            diagnostics["corner_residual_mean_xy_after"] = [
                float(v) for v in after_residual[after_common.cpu()].mean(dim=0).tolist()
            ]
            diagnostics["corner_residual_px_after"] = [float(v) for v in after_err.tolist()]
            diagnostics["corner_residual_xy_after"] = [
                [float(x), float(y)] for x, y in after_residual.tolist()
            ]
    if shared_yaw_delta is not None:
        diagnostics["shared_yaw_delta_rad"] = float(shared_yaw_delta)
        diagnostics["shared_yaw_delta_deg"] = float(math.degrees(shared_yaw_delta))
    if base_size is not None and used_size is not None and candidate_size is not None:
        base_size_cpu = base_size.detach().cpu().float().clamp_min(1e-6)
        used_size_cpu = used_size.detach().cpu().float()
        candidate_size_cpu = candidate_size.detach().cpu().float()
        diagnostics["size_scale"] = [float(v) for v in (used_size_cpu / base_size_cpu).tolist()]
        diagnostics["candidate_size_scale"] = [float(v) for v in (candidate_size_cpu / base_size_cpu).tolist()]
    return diagnostics


def _solve_corner_projection_pose(
    *,
    base_center: torch.Tensor,
    initial_center: torch.Tensor,
    object_size: torch.Tensor,
    base_rotation: torch.Tensor,
    waymo_corner_uv: torch.Tensor,
    waymo_corner_depths: torch.Tensor,
    waymo_corner_valid: torch.Tensor,
    waymo_edge_uv: torch.Tensor,
    waymo_edge_depths: torch.Tensor,
    waymo_edge_valid: torch.Tensor,
    target_bbox_model: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    support_points_world: torch.Tensor | None,
    max_yaw_rad: float,
    optimize_yaw: bool,
    fixed_yaw_delta: float = 0.0,
    iterations: int = 96,
) -> dict[str, Any]:
    del waymo_corner_depths, waymo_edge_depths, image_hw, initial_center, fixed_yaw_delta

    device = camera_to_world.device
    dtype = camera_to_world.dtype
    base_center = base_center.to(device=device, dtype=dtype)
    object_size = object_size.to(device=device, dtype=dtype)
    base_rotation = base_rotation.to(device=device, dtype=dtype)
    waymo_corner_uv = waymo_corner_uv.to(device=device, dtype=dtype)
    waymo_corner_valid = waymo_corner_valid.to(device=device)
    waymo_edge_uv = waymo_edge_uv.to(device=device, dtype=dtype)
    waymo_edge_valid = waymo_edge_valid.to(device=device)
    target_bbox_model = target_bbox_model.to(device=device, dtype=dtype)
    camera_to_world = camera_to_world.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)

    target_corner_valid = waymo_corner_valid.bool() & torch.isfinite(waymo_corner_uv).all(dim=-1)
    target_edge_valid = waymo_edge_valid.bool() & torch.isfinite(waymo_edge_uv).all(dim=-1)

    before = _corner_pose_metrics(
        base_center,
        object_size,
        base_rotation,
        waymo_corner_uv,
        target_corner_valid,
        waymo_edge_uv,
        target_edge_valid,
        target_bbox_model,
        camera_to_world,
        intrinsics,
    )
    if int(target_corner_valid.sum().item()) < 4:
        diagnostics = _serialize_corner_pose_diagnostics(
            status="insufficient_corners",
            accepted=False,
            reason="fewer_than_four_valid_waymo_corners",
            before=before,
            after=before,
            candidate_yaw_delta=0.0,
            used_yaw_delta=0.0,
            base_center=base_center,
            used_center=base_center,
            candidate_center=base_center,
            base_size=object_size,
            used_size=object_size,
            candidate_size=object_size,
        )
        return {
            "center": base_center.clone(),
            "size": object_size.clone(),
            "rotation": base_rotation.clone(),
            "bbox": before["bbox"],
            "yaw_delta": 0.0,
            "candidate_yaw_delta": 0.0,
            "accepted": False,
            "diagnostics": diagnostics,
            "waymo_uv": waymo_corner_uv.detach().cpu().float(),
            "waymo_valid": target_corner_valid.detach().cpu(),
            "initial_uv": before["corner_uv"].detach().cpu().float(),
            "initial_valid": before["corner_valid"].detach().cpu(),
            "refined_uv": before["corner_uv"].detach().cpu().float(),
            "refined_valid": before["corner_valid"].detach().cpu(),
        }

    rot_limit = max(0.0, float(max_yaw_rad))
    optimize_rotation_eff = bool(optimize_yaw and rot_limit > 1e-7)
    target_bbox_size = (target_bbox_model[2:] - target_bbox_model[:2]).clamp_min(1.0)
    uv_scale = torch.linalg.norm(target_bbox_size).clamp_min(16.0)
    size_norm = torch.linalg.norm(object_size).clamp_min(1e-3)
    base_center_cam = _world_to_camera_points(base_center.view(1, 3), camera_to_world)[0]
    base_depth = base_center_cam[2].clamp_min(1e-3)
    max_lateral_shift = max(0.35, float(size_norm.item()) * 0.35)
    max_depth_shift = max(0.75, float(size_norm.item()) * 0.80)
    delta_cam_limits = torch.tensor(
        [max_lateral_shift, max_lateral_shift, max_depth_shift],
        dtype=dtype,
        device=device,
    )

    support_depth_prior, support_center_prior = _delete_support_depth_prior(
        support_points_world,
        base_center,
        object_size,
        base_rotation,
        camera_to_world,
    )
    support_depth_prior = support_depth_prior.to(device=device, dtype=dtype)
    if support_center_prior is not None:
        support_center_prior = support_center_prior.to(device=device, dtype=dtype)

    center_param = torch.zeros((3,), dtype=dtype, device=device, requires_grad=True)
    rot_param = torch.zeros((3,), dtype=dtype, device=device, requires_grad=optimize_rotation_eff)
    size_param = torch.zeros((3,), dtype=dtype, device=device, requires_grad=True)
    max_log_size_delta = math.log(1.35)
    params = [center_param, size_param]
    if optimize_rotation_eff:
        params.append(rot_param)

    optimizer = torch.optim.Adam(params, lr=0.045)

    def build_candidate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_cam = delta_cam_limits * torch.tanh(center_param)
        delta_world = delta_cam @ camera_to_world[:3, :3].T
        center_world = base_center + delta_world
        log_size_delta = max_log_size_delta * torch.tanh(size_param)
        size = object_size * torch.exp(log_size_delta)
        if optimize_rotation_eff:
            rotvec = rot_limit * torch.tanh(rot_param)
            rotation = base_rotation @ _so3_exp_map(rotvec)
        else:
            rotvec = torch.zeros((3,), dtype=dtype, device=device)
            rotation = base_rotation
        return center_world, size, rotation, delta_cam, rotvec, log_size_delta

    with torch.enable_grad():
        for _ in range(max(1, int(iterations))):
            optimizer.zero_grad(set_to_none=True)
            center_world, candidate_size, rotation, delta_cam, rotvec, log_size_delta = build_candidate()
            corners = build_box_corners(center_world, candidate_size, rotation)
            corner_uv, corner_depths, corner_valid = _project_world_points_unclipped(
                corners,
                camera_to_world,
                intrinsics,
            )
            common = target_corner_valid & corner_valid & torch.isfinite(corner_uv).all(dim=-1)
            corner_weight = common.to(dtype)
            corner_loss_raw = F.huber_loss(
                corner_uv / uv_scale,
                waymo_corner_uv / uv_scale,
                reduction="none",
                delta=0.14,
            ).sum(dim=-1)
            corner_l2_raw = ((corner_uv - waymo_corner_uv) / uv_scale).pow(2).sum(dim=-1)
            corner_loss = (
                (corner_loss_raw * corner_weight).sum()
                / corner_weight.sum().clamp_min(1.0)
                + 0.18 * (corner_l2_raw * corner_weight).sum() / corner_weight.sum().clamp_min(1.0)
            )

            edges_world = _edge_midpoints(corners)
            edge_uv, edge_depths, edge_valid = _project_world_points_unclipped(
                edges_world,
                camera_to_world,
                intrinsics,
            )
            edge_common = target_edge_valid & edge_valid & torch.isfinite(edge_uv).all(dim=-1)
            edge_weight = edge_common.to(dtype)
            edge_loss_raw = F.huber_loss(
                edge_uv / uv_scale,
                waymo_edge_uv / uv_scale,
                reduction="none",
                delta=0.12,
            ).sum(dim=-1)
            edge_l2_raw = ((edge_uv - waymo_edge_uv) / uv_scale).pow(2).sum(dim=-1)
            edge_loss = (
                (edge_loss_raw * edge_weight).sum()
                / edge_weight.sum().clamp_min(1.0)
                + 0.12 * (edge_l2_raw * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)
            )

            bbox_common = target_corner_valid & torch.isfinite(corner_uv).all(dim=-1)
            large = torch.tensor(1.0e6, dtype=dtype, device=device)
            small = torch.tensor(-1.0e6, dtype=dtype, device=device)
            uv_min = torch.where(bbox_common[:, None], corner_uv, large).min(dim=0).values
            uv_max = torch.where(bbox_common[:, None], corner_uv, small).max(dim=0).values
            pred_bbox = torch.stack([uv_min[0], uv_min[1], uv_max[0], uv_max[1]], dim=0)
            bbox_loss = _bbox_alignment_loss(pred_bbox, target_bbox_model)

            center_cam = _world_to_camera_points(center_world.view(1, 3), camera_to_world)[0]
            depth_scale = base_depth.clamp_min(1.0)
            base_depth_loss = F.huber_loss(
                center_cam[2:3] / depth_scale,
                base_depth.view(1) / depth_scale,
                reduction="mean",
                delta=0.15,
            )
            support_depth_loss = F.huber_loss(
                center_cam[2:3] / support_depth_prior.clamp_min(1.0),
                support_depth_prior.view(1) / support_depth_prior.clamp_min(1.0),
                reduction="mean",
                delta=0.12,
            )
            support_center_loss = torch.tensor(0.0, dtype=dtype, device=device)
            if support_center_prior is not None:
                support_center_loss = F.huber_loss(
                    (center_world - support_center_prior) / size_norm,
                    torch.zeros((3,), dtype=dtype, device=device),
                    reduction="mean",
                    delta=0.35,
                )
            center_prior_loss = (delta_cam / delta_cam_limits).pow(2).mean()
            rotation_prior_loss = (rotvec / max(rot_limit, 1e-6)).pow(2).mean() if optimize_rotation_eff else torch.tensor(0.0, dtype=dtype, device=device)
            size_prior_loss = (log_size_delta / max_log_size_delta).pow(2).mean()
            invalid_loss = F.relu(1e-4 - corner_depths).mean() + 0.5 * F.relu(1e-4 - edge_depths).mean()

            loss = (
                corner_loss
                + 0.35 * edge_loss
                + 0.08 * bbox_loss
                + 0.025 * center_prior_loss
                + 0.018 * base_depth_loss
                + 0.035 * support_depth_loss
                + 0.018 * support_center_loss
                + 0.006 * rotation_prior_loss
                + 0.004 * size_prior_loss
                + 3.0 * invalid_loss
            )
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        best_center, best_size, best_rotation, best_delta_cam, best_rotvec, best_log_size_delta = build_candidate()
        best_rotation = _orthonormalize_rotation(best_rotation)
        best_yaw_delta = _yaw_delta_between_rotations(base_rotation, best_rotation)
        rotation_delta_mag = _rotation_delta_magnitude_rad(base_rotation, best_rotation)

    del optimizer, center_param, rot_param, size_param, params

    after_candidate = _corner_pose_metrics(
        best_center,
        best_size,
        best_rotation,
        waymo_corner_uv,
        target_corner_valid,
        waymo_edge_uv,
        target_edge_valid,
        target_bbox_model,
        camera_to_world,
        intrinsics,
    )
    before_rms = _safe_float(before.get("corner_rms"))
    after_rms = _safe_float(after_candidate.get("corner_rms"))
    before_iou = _safe_float(before.get("bbox_iou")) or 0.0
    after_iou = _safe_float(after_candidate.get("bbox_iou")) or 0.0
    base_depth = _safe_float(before.get("center_depth"))
    after_depth = _safe_float(after_candidate.get("center_depth"))
    depth_ratio = None
    if base_depth is not None and after_depth is not None and abs(base_depth) > 1e-6:
        depth_ratio = after_depth / base_depth
    center_shift = float((best_center - base_center).norm().item())
    support_depth = _safe_float(support_depth_prior)
    support_depth_ratio = None
    if support_depth is not None and after_depth is not None and abs(support_depth) > 1e-6:
        support_depth_ratio = after_depth / support_depth
    target_area = _box_area_xyxy(target_bbox_model)
    after_area = _box_area_xyxy(after_candidate.get("bbox"))
    target_box_size = _box_size_xyxy(target_bbox_model)
    after_box_size = _box_size_xyxy(after_candidate.get("bbox"))
    target_area_ratio = None
    target_width_ratio = None
    target_height_ratio = None
    if target_area is not None and after_area is not None and target_area > 1e-6:
        target_area_ratio = after_area / target_area
    if target_box_size is not None and after_box_size is not None:
        target_w_diag, target_h_diag = target_box_size
        after_w_diag, after_h_diag = after_box_size
        if target_w_diag > 1e-6:
            target_width_ratio = after_w_diag / target_w_diag
        if target_h_diag > 1e-6:
            target_height_ratio = after_h_diag / target_h_diag

    accepted = True
    reason = "accepted"
    if after_rms is None:
        accepted = False
        reason = "corner_rms_not_finite"
    elif before_rms is not None:
        min_drop = max(1.0, 0.08 * before_rms)
        if after_rms > before_rms - min_drop:
            accepted = False
            reason = "corner_rms_not_improved"
    if accepted and before_iou > 0.02 and after_iou + 0.04 < before_iou:
        accepted = False
        reason = "bbox_iou_degraded"
    if accepted and (depth_ratio is None or depth_ratio < 0.78 or depth_ratio > 1.28):
        accepted = False
        reason = "depth_ratio_out_of_bounds"
    support_depth_outlier = (
        support_depth_ratio is not None
        and (support_depth_ratio < 0.55 or support_depth_ratio > 2.20)
    )
    max_shift = float(torch.linalg.norm(delta_cam_limits).item())
    if accepted and center_shift > max_shift:
        accepted = False
        reason = "center_shift_out_of_bounds"
    if accepted and optimize_rotation_eff and rotation_delta_mag > max(0.10, rot_limit * 1.45):
        accepted = False
        reason = "rotation_delta_out_of_bounds"
    if accepted and target_area is not None and after_area is not None and target_area > 1e-6:
        if after_area / target_area > 1.22:
            accepted = False
            reason = "projected_box_too_large"
    if accepted and target_box_size is not None and after_box_size is not None:
        target_w, target_h = target_box_size
        after_w, after_h = after_box_size
        if target_w > 1e-6 and after_w / target_w > 1.18:
            accepted = False
            reason = "projected_box_width_too_large"
        if accepted and target_h > 1e-6 and after_h / target_h > 1.18:
            accepted = False
            reason = "projected_box_height_too_large"

    used_center = best_center if accepted else base_center.clone()
    used_rotation = best_rotation if accepted else base_rotation.clone()
    used_yaw_delta = best_yaw_delta if accepted else 0.0
    used_metrics = after_candidate if accepted else before
    status = "accepted" if accepted else "rejected"
    diagnostics = _serialize_corner_pose_diagnostics(
        status=status,
        accepted=accepted,
        reason=reason,
        before=before,
        after=after_candidate,
        candidate_yaw_delta=best_yaw_delta,
        used_yaw_delta=used_yaw_delta,
        base_center=base_center,
            used_center=used_center,
            candidate_center=best_center,
            base_size=object_size,
            used_size=best_size if accepted else object_size,
            candidate_size=best_size,
        )
    diagnostics["support_depth_prior"] = support_depth
    diagnostics["support_depth_ratio"] = support_depth_ratio
    diagnostics["support_depth_outlier"] = bool(support_depth_outlier)
    diagnostics["max_center_shift"] = float(max_shift)
    diagnostics["rotation_delta_rad"] = float(rotation_delta_mag)
    diagnostics["rotation_delta_deg"] = float(math.degrees(rotation_delta_mag))
    diagnostics["center_delta_cam"] = [float(v) for v in best_delta_cam.detach().cpu().tolist()]
    diagnostics["log_size_delta"] = [float(v) for v in best_log_size_delta.detach().cpu().tolist()]
    diagnostics["target_bbox_area_ratio"] = target_area_ratio
    diagnostics["target_bbox_width_ratio"] = target_width_ratio
    diagnostics["target_bbox_height_ratio"] = target_height_ratio
    diagnostics["solver_iterations"] = int(iterations)
    diagnostics["solver_loss_profile"] = "corner_l2_v2_support_soft"
    return {
        "center": used_center.clone(),
        "size": (best_size if accepted else object_size).clone(),
        "rotation": _orthonormalize_rotation(used_rotation),
        "bbox": used_metrics["bbox"],
        "yaw_delta": float(used_yaw_delta),
        "candidate_yaw_delta": float(best_yaw_delta),
        "accepted": bool(accepted),
        "diagnostics": diagnostics,
        "waymo_uv": waymo_corner_uv.detach().cpu().float(),
        "waymo_valid": target_corner_valid.detach().cpu(),
        "initial_uv": before["corner_uv"].detach().cpu().float(),
        "initial_valid": before["corner_valid"].detach().cpu(),
        "refined_uv": used_metrics["corner_uv"].detach().cpu().float(),
        "refined_valid": used_metrics["corner_valid"].detach().cpu(),
    }


def _base_corner_projection_result(
    *,
    base_center: torch.Tensor,
    object_size: torch.Tensor,
    base_rotation: torch.Tensor,
    waymo_corner_uv: torch.Tensor,
    waymo_corner_valid: torch.Tensor,
    waymo_edge_uv: torch.Tensor,
    waymo_edge_valid: torch.Tensor,
    target_bbox_model: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    status: str,
    reason: str,
    shared_yaw_delta: float | None = None,
) -> dict[str, Any]:
    metrics = _corner_pose_metrics(
        base_center,
        object_size,
        base_rotation,
        waymo_corner_uv,
        waymo_corner_valid,
        waymo_edge_uv,
        waymo_edge_valid,
        target_bbox_model,
        camera_to_world,
        intrinsics,
    )
    diagnostics = _serialize_corner_pose_diagnostics(
        status=status,
        accepted=False,
        reason=reason,
        before=metrics,
        after=metrics,
        candidate_yaw_delta=0.0,
        used_yaw_delta=0.0,
        base_center=base_center,
        used_center=base_center,
        candidate_center=base_center,
        base_size=object_size,
        used_size=object_size,
        candidate_size=object_size,
        shared_yaw_delta=shared_yaw_delta,
    )
    return {
        "center": base_center.clone(),
        "size": object_size.clone(),
        "rotation": base_rotation.clone(),
        "bbox": metrics["bbox"],
        "yaw_delta": 0.0,
        "candidate_yaw_delta": 0.0,
        "accepted": False,
        "diagnostics": diagnostics,
        "waymo_uv": waymo_corner_uv.detach().cpu().float(),
        "waymo_valid": waymo_corner_valid.detach().cpu(),
        "initial_uv": metrics["corner_uv"].detach().cpu().float(),
        "initial_valid": metrics["corner_valid"].detach().cpu(),
        "refined_uv": metrics["corner_uv"].detach().cpu().float(),
        "refined_valid": metrics["corner_valid"].detach().cpu(),
    }


def _corner_projection_result_for_pose(
    *,
    initial_center: torch.Tensor,
    initial_rotation: torch.Tensor,
    refined_center: torch.Tensor,
    refined_rotation: torch.Tensor,
    object_size: torch.Tensor,
    waymo_corner_uv: torch.Tensor,
    waymo_corner_valid: torch.Tensor,
    waymo_edge_uv: torch.Tensor,
    waymo_edge_valid: torch.Tensor,
    target_bbox_model: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    status: str,
    reason: str,
    accepted: bool,
    shared_yaw_delta: float,
    track_yaw_candidate_count: int,
) -> dict[str, Any]:
    target_corner_valid = waymo_corner_valid.bool() & torch.isfinite(waymo_corner_uv).all(dim=-1)
    target_edge_valid = waymo_edge_valid.bool() & torch.isfinite(waymo_edge_uv).all(dim=-1)
    before = _corner_pose_metrics(
        initial_center,
        object_size,
        initial_rotation,
        waymo_corner_uv,
        target_corner_valid,
        waymo_edge_uv,
        target_edge_valid,
        target_bbox_model,
        camera_to_world,
        intrinsics,
    )
    after = _corner_pose_metrics(
        refined_center,
        object_size,
        refined_rotation,
        waymo_corner_uv,
        target_corner_valid,
        waymo_edge_uv,
        target_edge_valid,
        target_bbox_model,
        camera_to_world,
        intrinsics,
    )
    diagnostics = _serialize_corner_pose_diagnostics(
        status=status,
        accepted=accepted,
        reason=reason,
        before=before,
        after=after,
        candidate_yaw_delta=shared_yaw_delta,
        used_yaw_delta=shared_yaw_delta,
        base_center=initial_center,
        used_center=refined_center,
        candidate_center=refined_center,
        base_size=object_size,
        used_size=object_size,
        candidate_size=object_size,
        shared_yaw_delta=shared_yaw_delta,
    )
    diagnostics["track_yaw_candidate_count"] = int(track_yaw_candidate_count)
    return {
        "center": refined_center.clone(),
        "size": object_size.clone(),
        "rotation": _orthonormalize_rotation(refined_rotation),
        "bbox": after["bbox"],
        "yaw_delta": float(shared_yaw_delta),
        "candidate_yaw_delta": float(shared_yaw_delta),
        "accepted": bool(accepted),
        "diagnostics": diagnostics,
        "waymo_uv": waymo_corner_uv.detach().cpu().float(),
        "waymo_valid": target_corner_valid.detach().cpu(),
        "initial_uv": before["corner_uv"].detach().cpu().float(),
        "initial_valid": before["corner_valid"].detach().cpu(),
        "refined_uv": after["corner_uv"].detach().cpu().float(),
        "refined_valid": after["corner_valid"].detach().cpu(),
    }


def _shared_corner_yaw_gate(
    frame_specs: list[dict[str, Any]],
    shared_yaw_delta: float,
    clean_state: CleanSceneState,
) -> tuple[bool, str]:
    if len(frame_specs) == 0 or abs(float(shared_yaw_delta)) < 1e-7:
        return False, "zero_shared_yaw"
    before_values: list[float] = []
    after_values: list[float] = []
    iou_deltas: list[float] = []
    improved_count = 0
    evaluated_count = 0

    for frame_spec in frame_specs:
        target_corner_bbox = frame_spec.get("waymo_bbox_model")
        if target_corner_bbox is None:
            target_corner_bbox = frame_spec["target_bbox_model"]
        source_front_index = int(frame_spec["source_front_index"])
        yaw_tensor = torch.tensor(float(shared_yaw_delta), dtype=frame_spec["gt_size"].dtype)
        after_rotation = _orthonormalize_rotation(
            frame_spec["proposal_rotation_base"] @ _rotation_z_tensor(yaw_tensor, dtype=frame_spec["gt_size"].dtype)
        )
        before = _corner_pose_metrics(
            frame_spec["proposal_center_initial"],
            frame_spec["gt_size"],
            frame_spec["proposal_rotation_base"],
            frame_spec["waymo_corner_uv"],
            frame_spec["waymo_corner_valid"],
            frame_spec["waymo_edge_uv"],
            frame_spec["waymo_edge_valid"],
            target_corner_bbox,
            clean_state.camera_to_world[source_front_index],
            clean_state.intrinsics[source_front_index],
        )
        after = _corner_pose_metrics(
            frame_spec["proposal_center_initial"],
            frame_spec["gt_size"],
            after_rotation,
            frame_spec["waymo_corner_uv"],
            frame_spec["waymo_corner_valid"],
            frame_spec["waymo_edge_uv"],
            frame_spec["waymo_edge_valid"],
            target_corner_bbox,
            clean_state.camera_to_world[source_front_index],
            clean_state.intrinsics[source_front_index],
        )
        before_rms = _safe_float(before.get("corner_rms"))
        after_rms = _safe_float(after.get("corner_rms"))
        if before_rms is None or after_rms is None:
            continue
        evaluated_count += 1
        before_values.append(before_rms)
        after_values.append(after_rms)
        iou_deltas.append(float(after.get("bbox_iou", 0.0)) - float(before.get("bbox_iou", 0.0)))
        if after_rms < before_rms - max(0.5, 0.03 * before_rms):
            improved_count += 1
        if after_rms > before_rms + max(1.0, 0.10 * before_rms):
            return False, "shared_yaw_degrades_frame_corner_rms"
        if iou_deltas[-1] < -0.04:
            return False, "shared_yaw_degrades_frame_bbox_iou"

    if evaluated_count == 0:
        return False, "shared_yaw_no_evaluable_frames"
    before_t = torch.tensor(before_values, dtype=torch.float32)
    after_t = torch.tensor(after_values, dtype=torch.float32)
    median_before = float(torch.median(before_t).item())
    median_after = float(torch.median(after_t).item())
    if median_after > median_before - max(0.5, 0.05 * median_before):
        return False, "shared_yaw_track_corner_rms_not_improved"
    min_improved = max(1, math.ceil(0.75 * evaluated_count))
    if improved_count < min_improved:
        return False, "shared_yaw_too_few_improved_frames"
    return True, "accepted"


def _wrap_angle_rad(angle_rad: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


def _proposal_depth_candidates(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    camera_to_world: torch.Tensor,
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
    target_bbox_model: torch.Tensor,
) -> list[torch.Tensor]:
    seed_points, _ = _extract_foreground_seed(
        point_map_world,
        depth_map,
        valid_mask,
        target_bbox_model,
        depth_quantile=0.35,
        shrink_ratio=0.05,
    )
    if seed_points.shape[0] > 0:
        seed_points_cam = _world_to_camera_points(seed_points, camera_to_world)
        finite_seed = torch.isfinite(seed_points_cam).all(dim=-1) & (seed_points_cam[:, 2] > 1e-3)
        seed_points_cam = seed_points_cam[finite_seed]
    else:
        seed_points_cam = torch.zeros((0, 3), dtype=object_center.dtype)

    object_center_cam = _world_to_camera_points(object_center.view(1, 3), camera_to_world)[0]
    depth_prior = object_center_cam[2].clamp_min(1e-2)
    depth_candidates: list[torch.Tensor] = [depth_prior]
    if seed_points_cam.shape[0] > 0:
        seed_depth = torch.quantile(seed_points_cam[:, 2], 0.5).clamp_min(1e-2)
        view_dir = seed_points.mean(dim=0) - camera_to_world[:3, 3]
        view_norm = view_dir.norm().clamp_min(1e-6)
        view_dir = view_dir / view_norm
        view_local = torch.matmul(view_dir.view(1, 3), object_rotation).view(3)
        support = torch.sum(view_local.abs() * (object_size * 0.5)).clamp_min(0.05)
        depth_candidates.append((seed_depth + support).clamp_min(1e-2))
        depth_candidates.append((seed_depth + 0.5 * support).clamp_min(1e-2))
    return depth_candidates


def _solve_proposal_pose_from_target_bbox(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    target_bbox_model: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    depth_candidates = _proposal_depth_candidates(
        object_center=object_center,
        object_size=object_size,
        object_rotation=object_rotation,
        camera_to_world=camera_to_world,
        point_map_world=point_map_world,
        depth_map=depth_map,
        valid_mask=valid_mask,
        target_bbox_model=target_bbox_model,
    )
    depth_prior = depth_candidates[0]

    base_rotation = object_rotation.clone()
    base_bbox = _project_box_bbox(object_center, object_size, base_rotation, camera_to_world, intrinsics, image_hw)
    best_bbox = base_bbox
    best_score = -1e9 if base_bbox is None else _score_projected_bbox(base_bbox, target_bbox_model)
    best_center = object_center.clone()
    best_rotation = base_rotation.clone()

    for depth_init in depth_candidates:
        with torch.enable_grad():
            center_xy_init = _unproject_center_xy(target_bbox_model, depth_init, intrinsics)
            center_xy = center_xy_init.clone().detach().requires_grad_(True)
            log_depth = depth_init.log().clone().detach().requires_grad_(True)
            yaw_param = torch.zeros(
                (),
                dtype=object_center.dtype,
                device=object_center.device,
                requires_grad=True,
            )
            optimizer = torch.optim.Adam([center_xy, log_depth, yaw_param], lr=0.05)

            for _ in range(60):
                optimizer.zero_grad()
                depth = torch.exp(log_depth).clamp_min(1e-3)
                center_cam = torch.cat([center_xy, depth.view(1)], dim=0)
                yaw = math.pi * torch.tanh(yaw_param)
                rotation = base_rotation @ _rotation_z_tensor(yaw, dtype=object_center.dtype)
                center_world = _camera_to_world_points(center_cam.view(1, 3), camera_to_world)[0]
                pred_bbox, invalid_penalty = _project_box_bbox_soft(
                    center_world,
                    object_size,
                    rotation,
                    camera_to_world,
                    intrinsics,
                    image_hw,
                )
                loss = _bbox_alignment_loss(pred_bbox, target_bbox_model)
                loss = loss + 2.0 * invalid_penalty
                loss = loss + 0.02 * (log_depth - depth_prior.log()).pow(2)
                loss = loss + 0.01 * yaw.pow(2)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            depth = torch.exp(log_depth).clamp_min(1e-3)
            center_cam = torch.cat([center_xy, depth.view(1)], dim=0)
            yaw = math.pi * torch.tanh(yaw_param)
            rotation = base_rotation @ _rotation_z_tensor(yaw, dtype=object_center.dtype)
            center_world = _camera_to_world_points(center_cam.view(1, 3), camera_to_world)[0]
            bbox = _project_box_bbox(center_world, object_size, rotation, camera_to_world, intrinsics, image_hw)
            score = -1e9 if bbox is None else _score_projected_bbox(bbox, target_bbox_model)
            if score > best_score:
                best_score = score
                best_center = center_world.clone()
                best_rotation = rotation.clone()
                best_bbox = bbox

    return best_center, _orthonormalize_rotation(best_rotation), best_bbox


def _solve_proposal_center_with_fixed_rotation(
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
    target_bbox_model: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    depth_candidates = _proposal_depth_candidates(
        object_center=object_center,
        object_size=object_size,
        object_rotation=object_rotation,
        camera_to_world=camera_to_world,
        point_map_world=point_map_world,
        depth_map=depth_map,
        valid_mask=valid_mask,
        target_bbox_model=target_bbox_model,
    )
    depth_prior = depth_candidates[0]

    best_bbox = _project_box_bbox(object_center, object_size, object_rotation, camera_to_world, intrinsics, image_hw)
    best_score = -1e9 if best_bbox is None else _score_projected_bbox(best_bbox, target_bbox_model)
    best_center = object_center.clone()

    for depth_init in depth_candidates:
        with torch.enable_grad():
            center_xy_init = _unproject_center_xy(target_bbox_model, depth_init, intrinsics)
            center_xy = center_xy_init.clone().detach().requires_grad_(True)
            log_depth = depth_init.log().clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([center_xy, log_depth], lr=0.05)

            for _ in range(50):
                optimizer.zero_grad()
                depth = torch.exp(log_depth).clamp_min(1e-3)
                center_cam = torch.cat([center_xy, depth.view(1)], dim=0)
                center_world = _camera_to_world_points(center_cam.view(1, 3), camera_to_world)[0]
                pred_bbox, invalid_penalty = _project_box_bbox_soft(
                    center_world,
                    object_size,
                    object_rotation,
                    camera_to_world,
                    intrinsics,
                    image_hw,
                )
                loss = _bbox_alignment_loss(pred_bbox, target_bbox_model)
                loss = loss + 2.0 * invalid_penalty
                loss = loss + 0.02 * (log_depth - depth_prior.log()).pow(2)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            depth = torch.exp(log_depth).clamp_min(1e-3)
            center_cam = torch.cat([center_xy, depth.view(1)], dim=0)
            center_world = _camera_to_world_points(center_cam.view(1, 3), camera_to_world)[0]
            bbox = _project_box_bbox(center_world, object_size, object_rotation, camera_to_world, intrinsics, image_hw)
            score = -1e9 if bbox is None else _score_projected_bbox(bbox, target_bbox_model)
            if score > best_score:
                best_score = score
                best_center = center_world.clone()
                best_bbox = bbox

    return best_center, best_bbox


def _refine_proposal_box_from_foreground_seed(
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
    box_xyxy: torch.Tensor,
    object_center: torch.Tensor,
    object_size: torch.Tensor,
    object_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_points, _ = _extract_foreground_seed(
        point_map_world,
        depth_map,
        valid_mask,
        box_xyxy,
        depth_quantile=0.5,
        shrink_ratio=0.0,
    )
    if seed_points.shape[0] < 24:
        return object_center, object_size

    local_seed = (seed_points - object_center) @ object_rotation
    q_low = torch.quantile(local_seed, 0.05, dim=0)
    q_high = torch.quantile(local_seed, 0.95, dim=0)
    local_center = 0.5 * (q_low + q_high)
    visible_extent = (q_high - q_low).clamp_min(1e-4)
    refined_center = object_center + local_center @ object_rotation.T
    refined_size = torch.maximum(object_size, visible_extent * 1.15)
    return refined_center, refined_size


def _load_asset_gaussians(path: str, cache: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if path in cache:
        return cache[path]

    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".ply":
        asset = read_gaussian_ply(path)
        means = torch.tensor(asset["means"].tolist(), dtype=torch.float32)
        colors = torch.tensor(asset["rgb"].tolist(), dtype=torch.float32).clamp(0.0, 1.0)
        opacities = (
            torch.tensor(asset["opacities"].tolist(), dtype=torch.float32)
            .view(-1, 1)
            .clamp(1e-6, 1.0 - 1e-6)
        )
        scales = torch.tensor(asset["scales"].tolist(), dtype=torch.float32).clamp_min(1e-6)
        quats = torch.tensor(asset["quats"].tolist(), dtype=torch.float32)
    elif suffix == ".spz":
        try:
            import spz
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading .spz assets requires the `spz` Python package to be installed."
            ) from exc

        unpack_options = spz.UnpackOptions()
        unpack_options.to_coord = spz.CoordinateSystem.UNSPECIFIED
        cloud = spz.load_spz(str(path_obj), unpack_options)
        means = _to_cpu_float_tensor(getattr(cloud, "positions"), shape_last=3)
        features_dc = _to_cpu_float_tensor(getattr(cloud, "colors"), shape_last=3)
        colors = (GAUSSIAN_SH_C0 * features_dc + 0.5).clamp(0.0, 1.0)

        alpha_values = getattr(cloud, "alphas", None)
        if alpha_values is None:
            alpha_values = getattr(cloud, "alpha", None)
        if alpha_values is None:
            raise ValueError(f"SPZ cloud does not expose alpha values: {path}")
        alpha_raw = _to_cpu_float_tensor(alpha_values, shape_last=1)
        opacities = torch.sigmoid(alpha_raw).clamp(1e-6, 1.0 - 1e-6)

        scale_raw = _to_cpu_float_tensor(getattr(cloud, "scales"), shape_last=3)
        scales = torch.exp(scale_raw).clamp_min(1e-6)
        rotations_xyzw = _to_cpu_float_tensor(getattr(cloud, "rotations"), shape_last=4)
        quats = F.normalize(rotations_xyzw[:, [3, 0, 1, 2]], dim=-1)
    else:
        raise ValueError(f"Unsupported asset format for {path}; expected .ply or .spz")

    cache[path] = {
        "means_raw": means,
        "colors": colors,
        "opacities": opacities,
        "scales": scales,
        "quats": quats,
        "vertex_count": torch.tensor([means.shape[0]], dtype=torch.long),
    }
    return cache[path]


def _quat_wxyz_to_mat(quats_wxyz: torch.Tensor) -> torch.Tensor:
    quats_xyzw = quats_wxyz[..., [1, 2, 3, 0]]
    return quat_to_mat(quats_xyzw)


def _mat_to_quat_wxyz(rotation_mats: torch.Tensor) -> torch.Tensor:
    quats_xyzw = mat_to_quat(rotation_mats)
    return quats_xyzw[..., [3, 0, 1, 2]]


def _compute_asset_scale_factors(
    asset_local: dict[str, torch.Tensor],
    target_lwh: torch.Tensor,
    opacity_threshold: float = 0.01,
) -> torch.Tensor:
    target_lwh = target_lwh.float()
    opacities = asset_local["opacities"].squeeze(-1).to(target_lwh.device)
    visible = opacities > opacity_threshold
    means_raw = asset_local["means_raw"].to(target_lwh.device)
    visible_xyz = means_raw[visible] if torch.any(visible) else means_raw
    if visible_xyz.numel() == 0:
        current_lwh = torch.ones(3, dtype=torch.float32, device=target_lwh.device)
    else:
        min_xyz = visible_xyz.min(dim=0).values
        max_xyz = visible_xyz.max(dim=0).values
        current_lwh = (max_xyz - min_xyz).clamp_min(1e-6)
    scale_wlh = torch.tensor(_GS_LWH_TO_XYZ_SCALE, dtype=torch.float32, device=target_lwh.device)
    return (target_lwh / current_lwh) * scale_wlh


def _asset_object_to_world_matrix(
    object_rotation: torch.Tensor,
    object_center: torch.Tensor,
) -> torch.Tensor:
    transform = torch.eye(4, dtype=object_rotation.dtype)
    transform[:3, :3] = object_rotation
    transform[:3, 3] = object_center
    return transform


def _transform_asset_gaussians_simple(
    asset_local: dict[str, torch.Tensor],
    target_lwh: torch.Tensor,
    object_rotation: torch.Tensor,
    object_center: torch.Tensor,
    opacity_threshold: float = 0.01,
    asset_yaw_correction_deg: float = 180.0,
) -> dict[str, torch.Tensor]:
    scale_factors = _compute_asset_scale_factors(asset_local, target_lwh, opacity_threshold=opacity_threshold)
    asset_yaw = _rotation_z(
        math.radians(float(asset_yaw_correction_deg)),
        dtype=object_rotation.dtype,
    ).to(object_rotation.device)
    world_rot = object_rotation @ asset_yaw

    means_scaled = asset_local["means_raw"].to(target_lwh.device) * scale_factors.view(1, 3)
    means_world = means_scaled @ world_rot.T + object_center

    quats = asset_local["quats"].to(target_lwh.device)
    scales = asset_local["scales"].to(target_lwh.device)
    local_rot = _quat_wxyz_to_mat(F.normalize(quats, dim=-1))
    world_rot_batch = world_rot.unsqueeze(0).expand(local_rot.shape[0], -1, -1)
    world_quats = F.normalize(_mat_to_quat_wxyz(world_rot_batch @ local_rot), dim=-1)
    scales_new = scales * scale_factors.view(1, 3)

    return {
        "means": means_world,
        "colors": asset_local["colors"],
        "opacities": asset_local["opacities"],
        "scales": scales_new,
        "quats": world_quats,
    }


def _project_asset_bbox_simple(
    asset_local: dict[str, torch.Tensor],
    target_lwh: torch.Tensor,
    object_rotation: torch.Tensor,
    object_center: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    opacity_threshold: float = 0.01,
    asset_yaw_correction_deg: float = 180.0,
) -> torch.Tensor | None:
    opacities = asset_local["opacities"].squeeze(-1)
    visible = opacities > opacity_threshold
    if not torch.any(visible):
        return None

    visible_xyz = asset_local["means_raw"][visible]
    if visible_xyz.shape[0] > 4096:
        stride = max(1, visible_xyz.shape[0] // 4096)
        visible_xyz = visible_xyz[::stride]

    scale_factors = _compute_asset_scale_factors(asset_local, target_lwh, opacity_threshold=opacity_threshold)
    asset_yaw = _rotation_z(
        math.radians(float(asset_yaw_correction_deg)),
        dtype=object_rotation.dtype,
    ).to(object_rotation.device)
    world_rot = object_rotation @ asset_yaw
    means_world = visible_xyz.to(target_lwh.device) * scale_factors.view(1, 3)
    means_world = means_world @ world_rot.T + object_center
    uv, _, valid = project_world_points(means_world, camera_to_world, intrinsics, image_hw)
    return compute_bbox_from_projected_points(uv, valid)


def _static_frame_segments(
    items: list[LocalizedFrameObject],
    motion_speed_thresh: float,
) -> list[list[LocalizedFrameObject]]:
    ordered = sorted(items, key=lambda item: int(item.frame_idx))
    segments: list[list[LocalizedFrameObject]] = []
    current: list[LocalizedFrameObject] = []
    prev_frame: int | None = None
    for item in ordered:
        is_static = (
            not bool(getattr(item, "waymo_frame_dynamic", False))
            or float(item.waymo_frame_speed_mps) <= float(motion_speed_thresh)
        )
        frame_idx = int(item.frame_idx)
        if is_static and (prev_frame is None or frame_idx == prev_frame + 1):
            current.append(item)
        else:
            if len(current) >= 2:
                segments.append(current)
            current = [item] if is_static else []
        prev_frame = frame_idx if is_static else None
    if len(current) >= 2:
        segments.append(current)
    return segments


def _choose_shared_static_rotation(segment: list[LocalizedFrameObject]) -> torch.Tensor:
    rotations = [item.proposal_rotation.detach().cpu().float() for item in segment]
    scores: list[float] = []
    for item in segment:
        diag = item.pose_refine_diagnostics or {}
        rms = diag.get("corner_rms_after")
        try:
            score = -float(rms)
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
    shared = _shared_track_rotation_candidate(rotations, scores, max_spread_deg=12.0)
    if shared is not None:
        return shared
    best_idx = max(range(len(segment)), key=lambda idx: scores[idx])
    return rotations[best_idx].clone()


def _localized_asset_local_payload(item: LocalizedFrameObject) -> dict[str, torch.Tensor] | None:
    if item.asset_means_local.numel() == 0:
        return None
    return {
        "means_raw": item.asset_means_local,
        "colors": item.asset_colors,
        "opacities": item.asset_opacities,
        "scales": item.asset_scales_local,
        "quats": item.asset_quats_local,
    }


def _update_localized_item_static_pose(
    item: LocalizedFrameObject,
    *,
    center: torch.Tensor,
    size: torch.Tensor,
    rotation: torch.Tensor,
    segment_start: int,
    segment_end: int,
    segment_length: int,
    clean_state: CleanSceneState,
    image_hw: tuple[int, int],
    asset_yaw_correction_deg: float,
) -> None:
    center = center.to(dtype=item.proposal_center.dtype)
    size = size.to(dtype=item.proposal_size.dtype)
    rotation = _orthonormalize_rotation(rotation.to(dtype=item.proposal_rotation.dtype))
    source_front_index = int(item.source_front_index)
    camera_to_world = clean_state.camera_to_world[source_front_index].to(device=center.device, dtype=center.dtype)
    intrinsics = clean_state.intrinsics[source_front_index].to(device=center.device, dtype=center.dtype)
    item.proposal_center = center.clone()
    item.proposal_size = size.clone()
    item.proposal_rotation = rotation.clone()
    item.refined_rotation = rotation.clone()
    item.asset_rotation = rotation.clone()
    item.asset_bottom_center = center.clone()
    item.asset_object_to_world = _asset_object_to_world_matrix(rotation, center)

    asset_local = _localized_asset_local_payload(item)
    if asset_local is not None:
        item.asset_scale_factors = _compute_asset_scale_factors(asset_local, size)
        asset_world = _transform_asset_gaussians_simple(
            asset_local,
            size,
            rotation,
            center,
            asset_yaw_correction_deg=asset_yaw_correction_deg,
        )
        item.asset_means_world = asset_world["means"]
        item.asset_colors = asset_world["colors"]
        item.asset_opacities = asset_world["opacities"]
        item.asset_scales = asset_world["scales"]
        item.asset_quats = asset_world["quats"]
        item.asset_scale = float(
            torch.mean(
                size
                / (asset_local["means_raw"].max(dim=0).values - asset_local["means_raw"].min(dim=0).values).clamp_min(1e-6)
            ).item()
        )
        item.projected_asset_bbox = _project_asset_bbox_simple(
            asset_local=asset_local,
            target_lwh=size,
            object_rotation=rotation,
            object_center=center,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            image_hw=image_hw,
            asset_yaw_correction_deg=asset_yaw_correction_deg,
        )

    uv, _, valid = project_dggt_box_corners_model(
        center,
        size,
        rotation,
        camera_to_world,
        intrinsics,
    )
    item.refined_box_corners_model = uv.detach().cpu().float()
    item.refined_box_corners_valid = valid.detach().cpu()

    diag = dict(item.pose_refine_diagnostics or {})
    waymo_uv = item.waymo_box_corners_model
    waymo_valid = item.waymo_box_corners_valid
    if waymo_uv is not None and waymo_valid is not None:
        waymo_uv_dev = waymo_uv.to(device=uv.device, dtype=uv.dtype)
        waymo_valid_dev = waymo_valid.to(device=uv.device)
        corner_rms, corner_count = _corner_projection_rms(
            uv,
            valid,
            waymo_uv_dev,
            waymo_valid_dev,
            min_count=4,
        )
        diag["corner_rms_after"] = _safe_float(corner_rms)
        diag["valid_corner_count"] = int(corner_count)
        target_bbox = compute_bbox_from_projected_points(waymo_uv_dev, waymo_valid_dev.bool())
        refined_bbox = compute_bbox_from_projected_points(uv, valid)
        if target_bbox is not None and refined_bbox is not None:
            diag["bbox_iou_after"] = float(box_iou_xyxy(refined_bbox.detach().cpu(), target_bbox.detach().cpu()))
            target_area = _box_area_xyxy(target_bbox)
            refined_area = _box_area_xyxy(refined_bbox)
            target_size = _box_size_xyxy(target_bbox)
            refined_size = _box_size_xyxy(refined_bbox)
            if target_area is not None and refined_area is not None and target_area > 1e-6:
                diag["target_bbox_area_ratio"] = refined_area / target_area
            if target_size is not None and refined_size is not None:
                target_w, target_h = target_size
                refined_w, refined_h = refined_size
                if target_w > 1e-6:
                    diag["target_bbox_width_ratio"] = refined_w / target_w
                if target_h > 1e-6:
                    diag["target_bbox_height_ratio"] = refined_h / target_h
        residual = item.refined_box_corners_model - waymo_uv.detach().cpu().float()
        err = torch.linalg.norm(residual, dim=-1)
        diag["corner_residual_max_after"] = float(err.max().item())
        diag["corner_residual_mean_xy_after"] = [float(v) for v in residual.mean(dim=0).tolist()]
        diag["corner_residual_px_after"] = [float(v) for v in err.tolist()]
        diag["corner_residual_xy_after"] = [[float(x), float(y)] for x, y in residual.tolist()]

    diag["static_segment_stabilized"] = True
    diag["static_segment_start_frame"] = int(segment_start)
    diag["static_segment_end_frame"] = int(segment_end)
    diag["static_segment_length"] = int(segment_length)
    diag["static_segment_policy"] = "waymo_static_contiguous_shared_dggt_pose_v1"
    item.pose_refine_diagnostics = diag


def _stabilize_static_localized_segments(
    items: list[LocalizedFrameObject],
    *,
    motion_speed_thresh: float,
    clean_state: CleanSceneState,
    image_hw: tuple[int, int],
    asset_yaw_correction_deg: float,
) -> None:
    for segment in _static_frame_segments(items, motion_speed_thresh):
        center_stack = torch.stack([item.proposal_center.detach().cpu().float() for item in segment], dim=0)
        size_stack = torch.stack([item.proposal_size.detach().cpu().float() for item in segment], dim=0)
        shared_center = torch.median(center_stack, dim=0).values
        shared_size = torch.median(size_stack, dim=0).values
        shared_rotation = _choose_shared_static_rotation(segment)
        start_frame = min(int(item.frame_idx) for item in segment)
        end_frame = max(int(item.frame_idx) for item in segment)
        for item in segment:
            _update_localized_item_static_pose(
                item,
                center=shared_center,
                size=shared_size,
                rotation=shared_rotation,
                segment_start=start_frame,
                segment_end=end_frame,
                segment_length=len(segment),
                clean_state=clean_state,
                image_hw=image_hw,
                asset_yaw_correction_deg=asset_yaw_correction_deg,
            )


def _empty_gaussian_dict() -> dict[str, torch.Tensor]:
    return {
        "means": torch.zeros((0, 3), dtype=torch.float32),
        "colors": torch.zeros((0, 3), dtype=torch.float32),
        "opacities": torch.zeros((0, 1), dtype=torch.float32),
        "scales": torch.zeros((0, 3), dtype=torch.float32),
        "quats": torch.zeros((0, 4), dtype=torch.float32),
    }


def _subset_gaussians(scene: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "means": scene["means"][mask],
        "colors": scene["colors"][mask],
        "opacities": scene["opacities"][mask],
        "scales": scene["scales"][mask],
        "quats": scene["quats"][mask],
    }


def _concat_gaussians(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(chunks) == 0:
        return _empty_gaussian_dict()
    return {
        "means": torch.cat([chunk["means"] for chunk in chunks], dim=0),
        "colors": torch.cat([chunk["colors"] for chunk in chunks], dim=0),
        "opacities": torch.cat([chunk["opacities"] for chunk in chunks], dim=0),
        "scales": torch.cat([chunk["scales"] for chunk in chunks], dim=0),
        "quats": torch.cat([chunk["quats"] for chunk in chunks], dim=0),
    }


def empty_gaussian_dict() -> dict[str, torch.Tensor]:
    return _empty_gaussian_dict()


def load_asset_gaussians(path: str, cache: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return _load_asset_gaussians(path, cache)


def transform_asset_gaussians(
    asset_local: dict[str, torch.Tensor],
    target_lwh: torch.Tensor,
    object_rotation: torch.Tensor,
    object_center: torch.Tensor,
    opacity_threshold: float = 0.01,
    asset_yaw_correction_deg: float = 180.0,
) -> dict[str, torch.Tensor]:
    return _transform_asset_gaussians_simple(
        asset_local,
        target_lwh,
        object_rotation,
        object_center,
        opacity_threshold=opacity_threshold,
        asset_yaw_correction_deg=asset_yaw_correction_deg,
    )


def apply_sim3_to_gaussian_dict(
    gaussians: dict[str, torch.Tensor],
    transform: Sim3Transform,
) -> dict[str, torch.Tensor]:
    means = _to_cpu_float_tensor(gaussians["means"], shape_last=3)
    colors = _to_cpu_float_tensor(gaussians["colors"], shape_last=3)
    opacities = _to_cpu_float_tensor(gaussians["opacities"], shape_last=1)
    scales = _to_cpu_float_tensor(gaussians["scales"], shape_last=3)
    quats = _to_cpu_float_tensor(gaussians["quats"], shape_last=4)

    means_out = transform.apply_points(means)
    scales_out = scales * float(transform.scale)
    local_rot = _quat_wxyz_to_mat(F.normalize(quats, dim=-1))
    world_rot = transform.rotation.detach().cpu().float().view(1, 3, 3)
    world_rot = world_rot.expand(local_rot.shape[0], -1, -1)
    quats_out = F.normalize(_mat_to_quat_wxyz(world_rot @ local_rot), dim=-1)

    return {
        "means": means_out.contiguous(),
        "colors": colors.contiguous(),
        "opacities": opacities.contiguous(),
        "scales": scales_out.contiguous(),
        "quats": quats_out.contiguous(),
    }


def _binary_mask_from_image(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().cpu().float()
    if mask.dim() == 3:
        return (mask > 0.5).any(dim=0)
    return mask > 0.5


def _compute_render_dynamic_ratio(
    dynamic_prob: torch.Tensor,
    support_mask: torch.Tensor,
    dynamic_prob_thresh: float,
) -> float:
    if int(support_mask.sum().item()) == 0:
        return 0.0
    return float((dynamic_prob[support_mask] >= dynamic_prob_thresh).float().mean().item())


def _refine_box_from_target_support(
    local_points: torch.Tensor,
    local_y: torch.Tensor,
    local_x: torch.Tensor,
    target_pixel_mask: torch.Tensor,
    coarse_center: torch.Tensor,
    coarse_rotation: torch.Tensor,
    coarse_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_local = target_pixel_mask[local_y, local_x] & torch.isfinite(local_points).all(dim=-1)
    if int(target_local.sum().item()) < 24:
        return coarse_center.clone(), coarse_size.clone(), target_local

    target_points = local_points[target_local]
    target_local_coords = (target_points - coarse_center) @ coarse_rotation
    q_low = torch.quantile(target_local_coords, 0.05, dim=0)
    q_high = torch.quantile(target_local_coords, 0.95, dim=0)
    visible_extent = (q_high - q_low).clamp_min(1e-4)
    local_center = 0.5 * (q_low + q_high)
    refined_center = coarse_center + local_center @ coarse_rotation.T
    margin = torch.maximum(visible_extent * 0.18, torch.full_like(visible_extent, 0.025))
    refined_size = (visible_extent + 2.0 * margin).clamp_min(0.05)
    return refined_center, refined_size, target_local


def _fit_box_from_support_points(
    support_points: torch.Tensor,
    coarse_center: torch.Tensor,
    coarse_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if support_points.shape[0] < 24:
        return coarse_center.clone(), torch.full((3,), 0.12, dtype=coarse_center.dtype)
    local = (support_points - coarse_center) @ coarse_rotation
    q_low = torch.quantile(local, 0.05, dim=0)
    q_high = torch.quantile(local, 0.95, dim=0)
    visible_extent = (q_high - q_low).clamp_min(1e-4)
    local_center = 0.5 * (q_low + q_high)
    refined_center = coarse_center + local_center @ coarse_rotation.T
    margin = torch.maximum(visible_extent * 0.18, torch.full_like(visible_extent, 0.025))
    refined_size = (visible_extent + 2.0 * margin).clamp_min(0.05)
    return refined_center, refined_size


def _voxel_connected_component(
    points: torch.Tensor,
    seed_mask: torch.Tensor,
    voxel_size: torch.Tensor,
    connectivity: int = 26,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] == 0:
        empty = torch.zeros((0,), dtype=torch.bool)
        return empty, empty
    if int(seed_mask.sum().item()) == 0:
        empty = torch.zeros((points.shape[0],), dtype=torch.bool)
        return empty, empty

    voxel_size = voxel_size.detach().cpu().float().clamp_min(1e-3).view(1, 3)
    points = points.detach().cpu().float()
    seed_mask = seed_mask.detach().cpu().bool()
    origin = points.min(dim=0).values
    coords = torch.floor((points - origin) / voxel_size).to(torch.int64)
    unique_coords, inverse = torch.unique(coords, sorted=True, return_inverse=True, dim=0)

    coord_list = [tuple(int(v) for v in coord) for coord in unique_coords.tolist()]
    coord_to_index = {coord: idx for idx, coord in enumerate(coord_list)}
    seed_voxels = sorted(set(int(v) for v in inverse[seed_mask].tolist()))
    keep_voxel = [False] * len(coord_list)
    queue = list(seed_voxels)
    if int(connectivity) == 6:
        neighbor_offsets = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
    else:
        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
    while queue:
        voxel_idx = queue.pop()
        if keep_voxel[voxel_idx]:
            continue
        keep_voxel[voxel_idx] = True
        vx, vy, vz = coord_list[voxel_idx]
        for dx, dy, dz in neighbor_offsets:
            neighbor_idx = coord_to_index.get((vx + dx, vy + dy, vz + dz))
            if neighbor_idx is not None and not keep_voxel[neighbor_idx]:
                queue.append(neighbor_idx)

    keep_voxel_mask = torch.tensor(keep_voxel, dtype=torch.bool)
    keep_point_mask = keep_voxel_mask[inverse]
    if int(keep_point_mask.sum().item()) == 0:
        empty = torch.zeros((points.shape[0],), dtype=torch.bool)
        return empty, empty

    keep_coords = {coord_list[idx] for idx, flag in enumerate(keep_voxel) if flag}
    ring_voxel = [False] * len(coord_list)
    for voxel_idx, coord in enumerate(coord_list):
        if keep_voxel[voxel_idx]:
            continue
        vx, vy, vz = coord
        found_neighbor = False
        for dx, dy, dz in neighbor_offsets:
            if (vx + dx, vy + dy, vz + dz) in keep_coords:
                found_neighbor = True
                if found_neighbor:
                    break
        ring_voxel[voxel_idx] = found_neighbor
    ring_voxel_mask = torch.tensor(ring_voxel, dtype=torch.bool)
    ring_point_mask = ring_voxel_mask[inverse] & (~keep_point_mask)
    return keep_point_mask, ring_point_mask


def _extract_delete_component(
    clean_state: CleanSceneState,
    source_front_index: int,
    gt_center: torch.Tensor,
    gt_rotation: torch.Tensor,
    gt_size: torch.Tensor,
    box_xyxy: torch.Tensor,
    object_pixel_mask: torch.Tensor,
    dynamic_image_mask: torch.Tensor,
    protected_boxes: list[dict[str, torch.Tensor]] | None,
    delete_motion_mode: str,
    dynamic_prob_thresh: float,
    core_scale: float,
    shell_scale: float,
    proposal_scale: float,
) -> dict[str, Any]:
    in_view_mask = clean_state.source_image_ids == int(source_front_index)
    local_indices = torch.nonzero(in_view_mask, as_tuple=False).flatten()
    if local_indices.numel() == 0:
        return {
            "delete_core_indices": torch.zeros((0,), dtype=torch.long),
            "delete_shell_indices": torch.zeros((0,), dtype=torch.long),
            "seed_point_count": 0,
            "candidate_pool_count": 0,
            "cluster_kept_count": 0,
            "render_dynamic_ratio": 0.0,
            "target_delete_coverage": 0.0,
            "outside_box_leak_ratio": 0.0,
            "seed_pixel_mask": object_pixel_mask,
            "delete_component_pixel_mask": torch.zeros_like(object_pixel_mask),
        }

    local_points = clean_state.means[in_view_mask]
    local_dynamic_prob = clean_state.dynamic_prob[in_view_mask]
    local_y = clean_state.source_y[in_view_mask]
    local_x = clean_state.source_x[in_view_mask]
    local_depth = clean_state.depth[source_front_index][local_y, local_x]
    finite_local = torch.isfinite(local_points).all(dim=-1)

    height, width = object_pixel_mask.shape
    bbox_inner_mask, bbox_outer_mask = _build_bbox_masks(
        box_xyxy,
        (height, width),
        inner_ratio=0.08,
        outer_ratio=0.04,
    )

    _, foreground_seed_mask = _extract_foreground_seed(
        clean_state.point_map_world[source_front_index],
        clean_state.depth[source_front_index],
        clean_state.valid_mask[source_front_index],
        box_xyxy,
        depth_quantile=0.55,
        shrink_ratio=0.05,
    )
    support_pixel_mask = _dilate_mask(object_pixel_mask | foreground_seed_mask, radius=2)
    core_pixel_mask = _dilate_mask(object_pixel_mask | foreground_seed_mask, radius=1)
    shell_pixel_mask = _dilate_mask(support_pixel_mask, radius=6)
    support_local = support_pixel_mask[local_y, local_x]
    core_local = core_pixel_mask[local_y, local_x]
    shell_local = shell_pixel_mask[local_y, local_x]
    foreground_local = foreground_seed_mask[local_y, local_x]
    bbox_inner_local = bbox_inner_mask[local_y, local_x]
    bbox_outer_local = bbox_outer_mask[local_y, local_x]
    dynamic_local = dynamic_image_mask[local_y, local_x]
    protected_local = _points_in_protected_boxes(local_points, protected_boxes or [], scale=1.08)
    refined_center, refined_size, target_local = _refine_box_from_target_support(
        local_points=local_points,
        local_y=local_y,
        local_x=local_x,
        target_pixel_mask=object_pixel_mask,
        coarse_center=gt_center,
        coarse_rotation=gt_rotation,
        coarse_size=gt_size,
    )
    box_corners = build_box_corners(
        refined_center,
        refined_size * max(1.0, proposal_scale),
        gt_rotation,
    )
    box_cam = _world_to_camera_points(box_corners, clean_state.camera_to_world[source_front_index])
    box_depth_low = float(box_cam[:, 2].min().item()) - 0.05
    box_depth_high = float(box_cam[:, 2].max().item()) + 0.08

    core_box_local = _points_in_box(
        local_points,
        refined_center,
        gt_rotation,
        refined_size,
        scale=max(1.00, core_scale + 0.20),
    )
    proposal_box_local = _points_in_box(
        local_points,
        refined_center,
        gt_rotation,
        refined_size,
        scale=max(1.35, proposal_scale + 0.10),
    )
    shell_box_local = _points_in_box(
        local_points,
        refined_center,
        gt_rotation,
        refined_size,
        scale=max(1.60, shell_scale + 0.45),
    )

    ratio_support = target_local & proposal_box_local
    if int(ratio_support.sum().item()) == 0:
        ratio_support = support_local & proposal_box_local
    if int(ratio_support.sum().item()) == 0:
        ratio_support = target_local | bbox_inner_local
    render_dynamic_ratio = _compute_render_dynamic_ratio(local_dynamic_prob, ratio_support, dynamic_prob_thresh)

    depth_seed_local = target_local | foreground_local | core_local
    if int(depth_seed_local.sum().item()) == 0:
        depth_seed_local = bbox_inner_local
    if int(depth_seed_local.sum().item()) > 0:
        seed_depths = local_depth[depth_seed_local]
        depth_low = float(torch.quantile(seed_depths, 0.05).item()) - 0.15
        depth_high = float(torch.quantile(seed_depths, 0.95).item()) + 0.25
        depth_band_local = (local_depth >= depth_low) & (local_depth <= depth_high)
    else:
        depth_band_local = torch.ones_like(local_depth, dtype=torch.bool)
    box_depth_local = (local_depth >= box_depth_low) & (local_depth <= box_depth_high)
    depth_band_local = depth_band_local & box_depth_local

    dynamic_support_local = dynamic_local | (local_dynamic_prob >= dynamic_prob_thresh)
    if delete_motion_mode == "dynamic":
        candidate_local = shell_box_local & depth_band_local & bbox_outer_local & dynamic_support_local
        seed_local = target_local & dynamic_support_local
        if int(seed_local.sum().item()) < 12:
            seed_local = support_local & proposal_box_local & dynamic_support_local
        if int(seed_local.sum().item()) < 12:
            dynamic_region = proposal_box_local & depth_band_local & dynamic_support_local
            if int(dynamic_region.sum().item()) > 0:
                dynamic_indices = torch.nonzero(dynamic_region, as_tuple=False).flatten()
                top_k = min(48, int(dynamic_indices.numel()))
                top_values, top_order = torch.topk(local_dynamic_prob[dynamic_indices], k=top_k)
                _ = top_values
                seed_local = torch.zeros_like(dynamic_region)
                seed_local[dynamic_indices[top_order]] = True
        voxel_scale = refined_size * torch.tensor(_DYNAMIC_VOXEL_SCALE, dtype=gt_size.dtype)
        voxel_scale = voxel_scale.clamp_min(0.008)
        connectivity = 26
    else:
        candidate_local = proposal_box_local & depth_band_local & bbox_outer_local
        seed_local = target_local | (foreground_local & proposal_box_local)
        if int(seed_local.sum().item()) < 12:
            seed_local = support_local & proposal_box_local
        if int(seed_local.sum().item()) < 12:
            seed_local = support_local & core_box_local
        voxel_scale = refined_size * torch.tensor(_STATIC_VOXEL_SCALE, dtype=gt_size.dtype)
        voxel_scale = voxel_scale.clamp_min(0.006)
        connectivity = 6

    candidate_local = candidate_local & (~protected_local | core_box_local)
    seed_local = seed_local & (~protected_local | core_box_local)

    candidate_local = candidate_local & finite_local
    candidate_pool_count = int(candidate_local.sum().item())
    seed_point_count = int(seed_local.sum().item())

    if candidate_pool_count == 0:
        return {
            "delete_core_indices": torch.zeros((0,), dtype=torch.long),
            "delete_shell_indices": torch.zeros((0,), dtype=torch.long),
            "seed_point_count": seed_point_count,
            "candidate_pool_count": 0,
            "cluster_kept_count": 0,
            "render_dynamic_ratio": render_dynamic_ratio,
            "target_delete_coverage": 0.0,
            "outside_box_leak_ratio": 0.0,
            "seed_pixel_mask": seed_local.new_zeros(object_pixel_mask.shape),
            "delete_component_pixel_mask": torch.zeros_like(object_pixel_mask),
            "refined_center": refined_center,
            "refined_size": refined_size,
        }

    candidate_indices = local_indices[candidate_local]
    candidate_points = local_points[candidate_local]
    candidate_seed_mask = seed_local[candidate_local]
    if int(candidate_seed_mask.sum().item()) == 0:
        nearest_idx = int(torch.argmin((candidate_points - gt_center).pow(2).sum(dim=-1)).item())
        candidate_seed_mask = torch.zeros((candidate_points.shape[0],), dtype=torch.bool)
        candidate_seed_mask[nearest_idx] = True
        seed_point_count = 1

    cluster_core_mask, cluster_shell_mask = _voxel_connected_component(
        candidate_points,
        candidate_seed_mask,
        voxel_scale,
        connectivity=connectivity,
    )
    if int(cluster_core_mask.sum().item()) == 0:
        return {
            "delete_core_indices": torch.zeros((0,), dtype=torch.long),
            "delete_shell_indices": torch.zeros((0,), dtype=torch.long),
            "seed_point_count": seed_point_count,
            "candidate_pool_count": candidate_pool_count,
            "cluster_kept_count": 0,
            "render_dynamic_ratio": render_dynamic_ratio,
            "target_delete_coverage": 0.0,
            "outside_box_leak_ratio": 0.0,
            "seed_pixel_mask": torch.zeros_like(object_pixel_mask),
            "delete_component_pixel_mask": torch.zeros_like(object_pixel_mask),
            "refined_center": refined_center,
            "refined_size": refined_size,
        }

    delete_core_indices = candidate_indices[cluster_core_mask]
    delete_shell_indices = candidate_indices[cluster_shell_mask]
    cluster_kept_count = int(cluster_core_mask.sum().item())

    seed_pixel_mask = torch.zeros_like(object_pixel_mask)
    seed_points_global = local_indices[seed_local]
    seed_pixel_mask[clean_state.source_y[seed_points_global], clean_state.source_x[seed_points_global]] = True

    delete_union_local = torch.zeros((local_indices.shape[0],), dtype=torch.bool)
    delete_union_local[candidate_local] = cluster_core_mask | cluster_shell_mask
    delete_component_pixel_mask = torch.zeros_like(object_pixel_mask)
    delete_points_global = local_indices[delete_union_local]
    delete_component_pixel_mask[clean_state.source_y[delete_points_global], clean_state.source_x[delete_points_global]] = True

    target_pixels = max(1, int(object_pixel_mask.sum().item()))
    deleted_pixels = int(delete_component_pixel_mask.sum().item())
    target_delete_coverage = float((delete_component_pixel_mask & object_pixel_mask).sum().item()) / float(target_pixels)
    if deleted_pixels == 0:
        outside_box_leak_ratio = 0.0
    else:
        outside_box_leak_ratio = float((delete_component_pixel_mask & (~shell_pixel_mask)).sum().item()) / float(
            deleted_pixels
        )

    return {
        "delete_core_indices": delete_core_indices,
        "delete_shell_indices": delete_shell_indices,
        "seed_point_count": seed_point_count,
        "candidate_pool_count": candidate_pool_count,
        "cluster_kept_count": cluster_kept_count,
        "render_dynamic_ratio": render_dynamic_ratio,
        "target_delete_coverage": target_delete_coverage,
        "outside_box_leak_ratio": outside_box_leak_ratio,
        "seed_pixel_mask": seed_pixel_mask,
        "delete_component_pixel_mask": delete_component_pixel_mask,
        "refined_center": refined_center,
        "refined_size": refined_size,
    }


def parse_object_slots(sample: dict[str, Any], object_slots: str | None) -> list[int]:
    editable_count = int(sample["editable_object_count"].item())
    editable_indices = sample["editable_object_indices"][:editable_count].tolist()
    if object_slots is None or object_slots == "" or object_slots == "all":
        return [int(idx) for idx in editable_indices]
    slots = []
    for token in object_slots.split(","):
        token = token.strip()
        if token == "":
            continue
        slots.append(int(token))
    return slots


def _select_localization_view(
    sample: dict[str, Any],
    slot_idx: int,
    frame_idx: int,
) -> tuple[int | None, torch.Tensor | None]:
    cam_ids = sample["cam_ids"]
    bbox_present = sample["object_bbox_present_mask_selected"]
    bbox_model = sample.get("object_bbox_model_selected")

    if isinstance(bbox_present, torch.Tensor) and isinstance(bbox_model, torch.Tensor):
        valid_view_offsets = torch.nonzero(bbox_present[slot_idx, frame_idx], as_tuple=False).flatten()
        if valid_view_offsets.numel() == 0:
            return None, None

        front_offsets = torch.nonzero(cam_ids == 0, as_tuple=False).flatten()
        if front_offsets.numel() > 0:
            front_offset = int(front_offsets[0].item())
            if bool(bbox_present[slot_idx, frame_idx, front_offset].item()):
                return (
                    front_offset,
                    bbox_model[slot_idx, frame_idx, front_offset].detach().cpu().float(),
                )

        best_offset = None
        best_box = None
        best_area = None
        for candidate_offset in valid_view_offsets.tolist():
            candidate_box = bbox_model[slot_idx, frame_idx, candidate_offset].detach().cpu().float()
            width = max(0.0, float(candidate_box[2] - candidate_box[0]))
            height = max(0.0, float(candidate_box[3] - candidate_box[1]))
            area = width * height
            if best_area is None or area > best_area:
                best_offset = int(candidate_offset)
                best_box = candidate_box
                best_area = area
        return best_offset, best_box

    front_offsets = torch.nonzero(cam_ids == 0, as_tuple=False).flatten()
    front_offset = int(front_offsets[0].item()) if front_offsets.numel() > 0 else 0
    front_bbox_present = sample["object_front_bbox_present_mask_selected"]
    if not bool(front_bbox_present[slot_idx, frame_idx].item()):
        return None, None
    return (
        front_offset,
        sample["object_front_bbox_model_selected"][slot_idx, frame_idx].detach().cpu().float(),
    )


def _morphology_opening(mask_2d: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask_2d.bool()
    return _dilate_mask(_erode_mask(mask_2d.bool(), radius), radius)


def _connected_components_4(mask_2d: torch.Tensor) -> list[dict[str, Any]]:
    mask = mask_2d.detach().cpu().bool()
    height, width = mask.shape
    visited = torch.zeros_like(mask)
    components: list[dict[str, Any]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for start_y, start_x in torch.nonzero(mask, as_tuple=False).tolist():
        if visited[start_y, start_x]:
            continue
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        ys_list: list[int] = []
        xs_list: list[int] = []
        while queue:
            y, x = queue.popleft()
            ys_list.append(y)
            xs_list.append(x)
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
        ys = torch.tensor(ys_list, dtype=torch.long)
        xs = torch.tensor(xs_list, dtype=torch.long)
        comp_mask = torch.zeros_like(mask)
        comp_mask[ys, xs] = True
        components.append(
            {
                "mask": comp_mask,
                "ys": ys,
                "xs": xs,
                "bbox_xyxy": torch.tensor(
                    [int(xs.min().item()), int(ys.min().item()), int(xs.max().item()) + 1, int(ys.max().item()) + 1],
                    dtype=torch.float32,
                ),
                "area": int(ys.numel()),
            }
        )
    return components


def _geodesic_reconstruction(seed_mask: torch.Tensor, raw_mask: torch.Tensor) -> torch.Tensor:
    seed = (seed_mask.bool() & raw_mask.bool()).to(torch.float32)
    raw = raw_mask.bool().to(torch.float32)
    if seed.sum().item() == 0:
        return torch.zeros_like(raw_mask, dtype=torch.bool)
    current = seed
    for _ in range(256):
        dilated = F.max_pool2d(current.view(1, 1, *current.shape), kernel_size=3, stride=1, padding=1)[0, 0]
        nxt = dilated * raw
        if torch.equal(nxt, current):
            break
        current = nxt
    return current > 0.5


def _component_in_bbox_ratio(component: dict[str, Any], bbox_xyxy: torch.Tensor | None) -> float:
    if bbox_xyxy is None:
        return 0.0
    x1 = float(bbox_xyxy[0].item())
    y1 = float(bbox_xyxy[1].item())
    x2 = float(bbox_xyxy[2].item())
    y2 = float(bbox_xyxy[3].item())
    ys = component["ys"].float()
    xs = component["xs"].float()
    inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    box_area = box_w * box_h
    return float(inside.sum().item()) / float(box_area)


def _box_to_mask(
    bbox_xyxy: torch.Tensor | None,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = image_hw
    mask = torch.zeros((height, width), dtype=torch.bool)
    if bbox_xyxy is None:
        return mask
    x1 = max(0, min(width - 1, int(math.floor(float(bbox_xyxy[0].item())))))
    y1 = max(0, min(height - 1, int(math.floor(float(bbox_xyxy[1].item())))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(bbox_xyxy[2].item())))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(bbox_xyxy[3].item())))))
    mask[y1:y2, x1:x2] = True
    return mask


def _count_mask_pixels_in_bbox(mask_2d: torch.Tensor, bbox_xyxy: torch.Tensor | None) -> int:
    if bbox_xyxy is None:
        return 0
    height, width = mask_2d.shape
    x1 = max(0, min(width - 1, int(math.floor(float(bbox_xyxy[0].item())))))
    y1 = max(0, min(height - 1, int(math.floor(float(bbox_xyxy[1].item())))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(bbox_xyxy[2].item())))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(bbox_xyxy[3].item())))))
    return int(mask_2d[y1:y2, x1:x2].sum().item())


def _estimate_mask_3d_thresholds(
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
    seed_mask: torch.Tensor,
    coarse_size: torch.Tensor,
) -> tuple[float, float]:
    support = seed_mask.bool() & valid_mask.bool()
    if int(support.sum().item()) == 0:
        base = max(0.006, float(coarse_size.max().item()) * 0.22)
        return min(base, 0.08), min(base * 1.4, 0.10)

    point_map_world = point_map_world.detach().cpu().float()
    depth_map = depth_map.detach().cpu().float()
    support = support.detach().cpu().bool()

    xyz_samples: list[torch.Tensor] = []
    depth_samples: list[torch.Tensor] = []
    vertical = support[:-1, :] & support[1:, :]
    if vertical.any():
        xyz_samples.append(torch.norm(point_map_world[:-1, :][vertical] - point_map_world[1:, :][vertical], dim=-1))
        depth_samples.append((depth_map[:-1, :][vertical] - depth_map[1:, :][vertical]).abs())
    horizontal = support[:, :-1] & support[:, 1:]
    if horizontal.any():
        xyz_samples.append(torch.norm(point_map_world[:, :-1][horizontal] - point_map_world[:, 1:][horizontal], dim=-1))
        depth_samples.append((depth_map[:, :-1][horizontal] - depth_map[:, 1:][horizontal]).abs())

    if len(xyz_samples) == 0:
        base = max(0.006, float(coarse_size.max().item()) * 0.22)
        return min(base, 0.08), min(base * 1.4, 0.10)

    xyz_values = torch.cat(xyz_samples, dim=0)
    depth_values = torch.cat(depth_samples, dim=0)
    xyz_values = xyz_values[torch.isfinite(xyz_values)]
    depth_values = depth_values[torch.isfinite(depth_values)]
    if xyz_values.numel() == 0 or depth_values.numel() == 0:
        base = max(0.006, float(coarse_size.max().item()) * 0.22)
        return min(base, 0.08), min(base * 1.4, 0.10)

    xyz_thresh = float(torch.quantile(xyz_values, 0.90).item()) * 2.5
    depth_thresh = float(torch.quantile(depth_values, 0.90).item()) * 2.5
    xyz_floor = max(0.004, float(coarse_size.max().item()) * 0.03)
    depth_floor = max(0.004, float(coarse_size.max().item()) * 0.04)
    xyz_thresh = min(max(xyz_thresh, xyz_floor), max(0.08, float(coarse_size.max().item()) * 0.45))
    depth_thresh = min(max(depth_thresh, depth_floor), max(0.10, float(coarse_size.max().item()) * 0.60))
    return xyz_thresh, depth_thresh


def _connected_components_3d(
    mask_2d: torch.Tensor,
    point_map_world: torch.Tensor,
    depth_map: torch.Tensor,
    valid_mask: torch.Tensor,
    xyz_thresh: float,
    depth_thresh: float,
) -> list[dict[str, Any]]:
    mask = (mask_2d.bool() & valid_mask.bool()).detach().cpu()
    if int(mask.sum().item()) == 0:
        return []

    point_map_world = point_map_world.detach().cpu().float()
    depth_map = depth_map.detach().cpu().float()
    height, width = mask.shape
    visited = torch.zeros_like(mask)
    components: list[dict[str, Any]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for start_y, start_x in torch.nonzero(mask, as_tuple=False).tolist():
        if visited[start_y, start_x]:
            continue
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            point = point_map_world[y, x]
            depth = float(depth_map[y, x].item())
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                if abs(float(depth_map[ny, nx].item()) - depth) > depth_thresh:
                    continue
                if float(torch.norm(point_map_world[ny, nx] - point).item()) > xyz_thresh:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

        ys = torch.tensor([p[0] for p in pixels], dtype=torch.long)
        xs = torch.tensor([p[1] for p in pixels], dtype=torch.long)
        comp_mask = torch.zeros_like(mask)
        comp_mask[ys, xs] = True
        comp_points = point_map_world[ys, xs]
        components.append(
            {
                "mask": comp_mask,
                "ys": ys,
                "xs": xs,
                "points": comp_points,
                "area": int(ys.numel()),
            }
        )
    return components


def _select_target_semantic_mask(
    raw_mask: torch.Tensor,
    semantic_prob: torch.Tensor,
    point_map_world_view: torch.Tensor,
    depth_map_view: torch.Tensor,
    valid_mask_view: torch.Tensor,
    target_bbox_model: torch.Tensor,
    other_boxes_2d: list[torch.Tensor],
    coarse_center: torch.Tensor,
    coarse_rotation: torch.Tensor,
    coarse_size: torch.Tensor,
) -> torch.Tensor:
    image_hw = tuple(raw_mask.shape)
    target_box_mask = _box_to_mask(target_bbox_model, image_hw)
    bbox_inner_mask, bbox_outer_mask = _build_bbox_masks(
        target_bbox_model,
        image_hw,
        inner_ratio=0.10,
        outer_ratio=0.20,
    )

    raw_vehicle = raw_mask.bool() & valid_mask_view.bool() & bbox_outer_mask
    clean_vehicle = _close_mask(_open_mask(raw_vehicle, radius=1), radius=1)

    candidate_clean = torch.zeros_like(clean_vehicle)
    for comp in _connected_components_4(clean_vehicle):
        if comp["area"] < 6:
            continue
        if _count_mask_pixels_in_bbox(comp["mask"], target_bbox_model) > 0:
            candidate_clean |= comp["mask"]
    if int(candidate_clean.sum().item()) == 0:
        candidate_clean = clean_vehicle.clone()

    candidate_mask = _geodesic_reconstruction(candidate_clean & bbox_outer_mask, raw_vehicle)
    candidate_mask &= raw_vehicle
    if int(candidate_mask.sum().item()) == 0:
        return torch.zeros_like(raw_mask, dtype=torch.bool)

    seed_mask = candidate_mask & bbox_inner_mask
    if int(seed_mask.sum().item()) < 12:
        prob_seed = (semantic_prob >= 0.22) & valid_mask_view.bool() & bbox_inner_mask
        seed_mask |= prob_seed
    if int(seed_mask.sum().item()) < 12:
        seed_mask = candidate_mask & target_box_mask
    if int(seed_mask.sum().item()) == 0:
        return _select_connected_component(candidate_mask, target_box_mask & candidate_mask)

    xyz_thresh, depth_thresh = _estimate_mask_3d_thresholds(
        point_map_world_view,
        depth_map_view,
        valid_mask_view,
        seed_mask,
        coarse_size,
    )
    components_3d = _connected_components_3d(
        candidate_mask,
        point_map_world_view,
        depth_map_view,
        valid_mask_view,
        xyz_thresh=xyz_thresh,
        depth_thresh=depth_thresh,
    )
    if len(components_3d) == 0:
        return _select_connected_component(candidate_mask, seed_mask)

    best_mask = torch.zeros_like(raw_mask, dtype=torch.bool)
    best_score = -1e18
    for comp in components_3d:
        if comp["area"] < 8:
            continue
        comp_mask = comp["mask"]
        comp_points = comp["points"]
        seed_overlap = int((comp_mask & seed_mask).sum().item())
        target_overlap = _count_mask_pixels_in_bbox(comp_mask, target_bbox_model)
        other_overlap = 0
        for other_box in other_boxes_2d:
            other_overlap = max(other_overlap, _count_mask_pixels_in_bbox(comp_mask, other_box))
        box_dist = _normalized_box_distance(
            comp_points,
            coarse_center,
            coarse_rotation,
            coarse_size,
            scale=2.0,
        )
        box_dist_med = float(torch.median(box_dist).item()) if box_dist.numel() > 0 else 1e6
        score = (
            seed_overlap * 100000.0
            + target_overlap * 6.0
            - other_overlap * 4.0
            - box_dist_med * 120.0
            + float(comp["area"]) * 0.05
        )
        if score > best_score:
            best_score = score
            best_mask = comp_mask

    if int(best_mask.sum().item()) == 0:
        return _select_connected_component(candidate_mask, seed_mask)
    return best_mask


def _fit_delete_box_from_component_points(
    support_points: torch.Tensor,
    coarse_center: torch.Tensor,
    coarse_rotation: torch.Tensor,
    coarse_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if support_points.shape[0] < 24:
        return coarse_center.clone(), coarse_size.clone()
    local = (support_points - coarse_center) @ coarse_rotation
    q_low = torch.quantile(local, 0.05, dim=0)
    q_high = torch.quantile(local, 0.95, dim=0)
    axis_margin = torch.maximum(
        coarse_size * torch.tensor([0.08, 0.08, 0.06], dtype=coarse_size.dtype),
        torch.full_like(coarse_size, 0.004),
    )
    low = torch.minimum(q_low - axis_margin, -0.5 * coarse_size)
    high = torch.maximum(q_high + axis_margin, 0.5 * coarse_size)
    refined_center = coarse_center + (0.5 * (low + high)) @ coarse_rotation.T
    refined_size = (high - low).clamp_min(coarse_size * 0.85)
    return refined_center, refined_size


def _expand_delete_cluster_to_scene(
    clean_state: CleanSceneState,
    seed_indices: torch.Tensor,
    support_points_world: torch.Tensor,
    refined_center: torch.Tensor,
    refined_rotation: torch.Tensor,
    refined_size: torch.Tensor,
    source_front_index: int,
    target_bbox_model: torch.Tensor,
    target_mask_2d: torch.Tensor,
    protected_boxes: list[dict[str, torch.Tensor]],
    delete_motion_mode: str,
    dynamic_prob_thresh: float,
) -> dict[str, Any]:
    empty = {
        "delete_core_indices": torch.zeros((0,), dtype=torch.long),
        "delete_shell_indices": torch.zeros((0,), dtype=torch.long),
        "candidate_pool_count": 0,
        "cluster_kept_count": 0,
        "delete_motion_mode": delete_motion_mode,
    }
    if seed_indices.numel() == 0:
        return empty

    points = clean_state.means.detach().cpu().float()
    finite_mask = torch.isfinite(points).all(dim=-1)
    protected_mask = _points_in_protected_boxes(points, protected_boxes, scale=1.02)
    candidate_mask = finite_mask & (~protected_mask)
    candidate_mask &= _points_in_box(
        points,
        refined_center,
        refined_rotation,
        refined_size,
        scale=1.05,
    )
    if int(candidate_mask.sum().item()) == 0:
        return empty

    camera_to_world = clean_state.camera_to_world[source_front_index]
    seed_points = points[seed_indices]
    if support_points_world.shape[0] >= 16:
        support_points = support_points_world.detach().cpu().float()
    else:
        support_points = seed_points
    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return empty
    candidate_points = points[candidate_indices]
    projected_uv, _, projected_valid = project_world_points(
        candidate_points,
        camera_to_world,
        clean_state.intrinsics[source_front_index],
        image_hw,
    )
    x1, y1, x2, y2 = [float(v) for v in target_bbox_model.tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    gate_radius = max(4, int(round(max(bw, bh) * 0.05)))
    gate_mask = _dilate_mask(target_mask_2d.bool(), radius=gate_radius)
    gate_x1 = max(0, min(image_hw[1] - 1, int(math.floor(x1 - bw * 0.03))))
    gate_x2 = max(gate_x1 + 1, min(image_hw[1], int(math.ceil(x2 + bw * 0.03))))
    gate_y1 = max(0, min(image_hw[0] - 1, int(math.floor(y1 - bh * 0.03))))
    gate_y2 = max(gate_y1 + 1, min(image_hw[0], int(math.ceil(y2 + bh * 0.03))))
    bbox_keep = (
        projected_valid
        & (projected_uv[:, 0] >= gate_x1)
        & (projected_uv[:, 0] < gate_x2)
        & (projected_uv[:, 1] >= gate_y1)
        & (projected_uv[:, 1] < gate_y2)
    )
    if int(bbox_keep.sum().item()) > 0:
        gate_uv = projected_uv[bbox_keep]
        gate_x = gate_uv[:, 0].long().clamp(0, image_hw[1] - 1)
        gate_y = gate_uv[:, 1].long().clamp(0, image_hw[0] - 1)
        silhouette_keep = gate_mask[gate_y, gate_x]
        gate_selected = torch.zeros_like(bbox_keep)
        gate_selected[torch.nonzero(bbox_keep, as_tuple=False).flatten()[silhouette_keep]] = True
        candidate_mask = torch.zeros_like(candidate_mask)
        candidate_mask[candidate_indices[gate_selected]] = True

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return empty

    seed_local = (support_points - refined_center) @ refined_rotation
    z_low = float(torch.quantile(seed_local[:, 2], 0.05).item()) - max(0.004, float(refined_size[2].item()) * 0.15)
    z_high = float(torch.quantile(seed_local[:, 2], 0.95).item()) + max(0.004, float(refined_size[2].item()) * 0.12)
    candidate_local = (points[candidate_indices] - refined_center) @ refined_rotation
    z_keep = (candidate_local[:, 2] >= z_low) & (candidate_local[:, 2] <= z_high)
    if int(z_keep.sum().item()) > 0:
        candidate_mask = torch.zeros_like(candidate_mask)
        candidate_mask[candidate_indices[z_keep]] = True

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return empty

    seed_cam = _world_to_camera_points(support_points, camera_to_world)
    seed_depth = seed_cam[:, 2]
    if seed_depth.numel() > 0:
        depth_q05 = float(torch.quantile(seed_depth, 0.05).item())
        depth_q95 = float(torch.quantile(seed_depth, 0.95).item())
        depth_margin = max(0.006, (depth_q95 - depth_q05) * 0.35)
        depth_low = depth_q05 - depth_margin
        depth_high = depth_q95 + depth_margin
        candidate_cam = _world_to_camera_points(points[candidate_indices], camera_to_world)
        depth_keep = (candidate_cam[:, 2] >= depth_low) & (candidate_cam[:, 2] <= depth_high)
        if int(depth_keep.sum().item()) > 0:
            candidate_mask = torch.zeros_like(candidate_mask)
            candidate_mask[candidate_indices[depth_keep]] = True
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return empty

    candidate_seed_mask = torch.zeros((candidate_indices.shape[0],), dtype=torch.bool)
    seed_lookup = torch.zeros((points.shape[0],), dtype=torch.bool)
    seed_lookup[seed_indices] = True
    candidate_seed_mask = seed_lookup[candidate_indices]

    final_motion_mode = delete_motion_mode
    if delete_motion_mode == "dynamic":
        dynamic_keep = clean_state.dynamic_prob[candidate_indices] >= dynamic_prob_thresh
        if int(dynamic_keep.sum().item()) >= max(12, int(candidate_seed_mask.sum().item())):
            candidate_indices = candidate_indices[dynamic_keep]
            candidate_seed_mask = candidate_seed_mask[dynamic_keep]
        else:
            final_motion_mode = "dynamic_waymo_static_render"
    if candidate_indices.numel() == 0:
        return empty
    if int(candidate_seed_mask.sum().item()) == 0:
        candidate_points = points[candidate_indices]
        nearest_idx = int(torch.argmin((candidate_points - refined_center).pow(2).sum(dim=-1)).item())
        candidate_seed_mask = torch.zeros((candidate_indices.shape[0],), dtype=torch.bool)
        candidate_seed_mask[nearest_idx] = True

    voxel_const = _DYNAMIC_VOXEL_SCALE if final_motion_mode == "dynamic" else _STATIC_VOXEL_SCALE
    voxel_floor = 0.008 if final_motion_mode == "dynamic" else 0.006
    voxel_scale = refined_size * torch.tensor(voxel_const, dtype=refined_size.dtype)
    voxel_scale = voxel_scale.clamp_min(voxel_floor)
    connectivity = 26 if final_motion_mode == "dynamic" else 6
    cluster_core_mask, cluster_shell_mask = _voxel_connected_component(
        points[candidate_indices],
        candidate_seed_mask,
        voxel_scale,
        connectivity=connectivity,
    )
    if int(cluster_core_mask.sum().item()) == 0:
        return empty
    return {
        "delete_core_indices": candidate_indices[cluster_core_mask],
        "delete_shell_indices": candidate_indices[cluster_shell_mask],
        "candidate_pool_count": int(candidate_indices.shape[0]),
        "cluster_kept_count": int(cluster_core_mask.sum().item()),
        "delete_motion_mode": final_motion_mode,
    }


def _normalized_box_distance(
    points: torch.Tensor,
    center: torch.Tensor,
    rotation: torch.Tensor,
    size: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    if points.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.float32)
    local = (points - center) @ rotation
    half = (size * (0.5 * scale)).clamp_min(1e-4)
    return (local.abs() / half).amax(dim=-1)


def _filter_pixel_mask_by_box(
    pixel_mask_2d: torch.Tensor,
    point_map_world_view: torch.Tensor,
    valid_mask_view: torch.Tensor,
    box_center: torch.Tensor,
    box_rotation: torch.Tensor,
    box_size: torch.Tensor,
    scale: float = 1.10,
) -> torch.Tensor:
    resolved = pixel_mask_2d.bool() & valid_mask_view.bool()
    if int(resolved.sum().item()) == 0:
        return torch.zeros_like(pixel_mask_2d, dtype=torch.bool)
    pts3d = point_map_world_view[resolved].reshape(-1, 3)
    inside = _points_in_box(pts3d, box_center, box_rotation, box_size, scale=scale)
    if int(inside.sum().item()) == 0:
        return torch.zeros_like(pixel_mask_2d, dtype=torch.bool)
    out = torch.zeros_like(pixel_mask_2d, dtype=torch.bool)
    coords = torch.nonzero(resolved, as_tuple=False)
    kept_coords = coords[inside]
    out[kept_coords[:, 0], kept_coords[:, 1]] = True
    return out


def _select_delete_box_candidate(
    pixel_mask_2d: torch.Tensor,
    point_map_world_view: torch.Tensor,
    valid_mask_view: torch.Tensor,
    gt_center: torch.Tensor,
    gt_rotation: torch.Tensor,
    gt_size: torch.Tensor,
    proposal_center: torch.Tensor,
    proposal_rotation: torch.Tensor,
    proposal_size: torch.Tensor,
    filter_scale: float = 1.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_mask = _filter_pixel_mask_by_box(
        pixel_mask_2d,
        point_map_world_view,
        valid_mask_view,
        gt_center,
        gt_rotation,
        gt_size,
        scale=filter_scale,
    )
    gt_support = int(gt_mask.sum().item())

    proposal_shift_local = ((proposal_center - gt_center).view(1, 3) @ gt_rotation).abs()[0]
    proposal_shift_norm = proposal_shift_local / gt_size.clamp_min(1e-4)
    proposal_plausible = bool(
        torch.isfinite(proposal_shift_norm).all().item()
        and float(proposal_shift_norm.max().item()) <= 1.5
    )
    if not proposal_plausible:
        return gt_center, gt_rotation, gt_size

    proposal_mask = _filter_pixel_mask_by_box(
        pixel_mask_2d,
        point_map_world_view,
        valid_mask_view,
        proposal_center,
        proposal_rotation,
        proposal_size,
        scale=filter_scale,
    )
    proposal_support = int(proposal_mask.sum().item())
    if proposal_support <= 0:
        return gt_center, gt_rotation, gt_size
    if gt_support <= 0:
        return proposal_center, proposal_rotation, proposal_size

    if proposal_support >= max(gt_support + 16, int(math.ceil(gt_support * 1.25))):
        return proposal_center, proposal_rotation, proposal_size
    return gt_center, gt_rotation, gt_size


def _assign_shared_pixels_by_box_distance(
    shared_mask: torch.Tensor,
    point_map_world_view: torch.Tensor,
    valid_mask_view: torch.Tensor,
    involved_boxes: list[dict[str, torch.Tensor]],
    current_box_idx: int,
    scale: float = 1.10,
) -> torch.Tensor:
    if len(involved_boxes) == 0 or int(shared_mask.sum().item()) == 0:
        return torch.zeros_like(shared_mask, dtype=torch.bool)
    resolved = shared_mask.bool() & valid_mask_view.bool()
    if int(resolved.sum().item()) == 0:
        return torch.zeros_like(shared_mask, dtype=torch.bool)
    pts3d = point_map_world_view[resolved].reshape(-1, 3)
    costs = []
    for box in involved_boxes:
        costs.append(
            _normalized_box_distance(
                pts3d,
                box["center"],
                box["rotation"],
                box["size"],
                scale=scale,
            )
        )
    cost_stack = torch.stack(costs, dim=1)
    inside_any = (cost_stack <= 1.0).any(dim=1)
    nearest = cost_stack.argmin(dim=1)
    keep = inside_any & (nearest == int(current_box_idx)) & (cost_stack[:, int(current_box_idx)] <= 1.0)
    out = torch.zeros_like(shared_mask, dtype=torch.bool)
    coords = torch.nonzero(resolved, as_tuple=False)
    if keep.any():
        kept_coords = coords[keep]
        out[kept_coords[:, 0], kept_coords[:, 1]] = True
    return out


def _gaussian_indices_from_pixel_mask(
    clean_state: CleanSceneState,
    pixel_mask_2d: torch.Tensor,
    view_image_idx: int,
) -> torch.Tensor:
    view_match = clean_state.source_image_ids == int(view_image_idx)
    if not view_match.any():
        return torch.zeros_like(view_match)
    mask_2d = pixel_mask_2d.detach().cpu().bool()
    ys = clean_state.source_y
    xs = clean_state.source_x
    result = torch.zeros_like(view_match)
    idx = torch.nonzero(view_match, as_tuple=False).flatten()
    if idx.numel() == 0:
        return result
    sampled = mask_2d[ys[idx], xs[idx]]
    result[idx[sampled]] = True
    return result


def _sample_camera_to_world_view(sample: dict[str, Any], frame_idx: int, view_offset: int) -> torch.Tensor:
    camera_to_world = sample["camera_to_world_corrected"].detach().cpu().float()
    if camera_to_world.dim() == 4:
        return camera_to_world[frame_idx, view_offset]
    return camera_to_world[frame_idx]


def _sample_intrinsics_view(sample: dict[str, Any], frame_idx: int, view_offset: int) -> torch.Tensor:
    intrinsics = sample["intrinsics"].detach().cpu().float()
    if intrinsics.dim() == 4:
        return intrinsics[frame_idx, view_offset]
    if intrinsics.dim() == 3:
        if intrinsics.shape[0] == int(sample["cam_ids"].numel()):
            return intrinsics[view_offset]
        return intrinsics[frame_idx]
    return intrinsics


def _sample_raw_hw_view(sample: dict[str, Any], frame_idx: int, view_offset: int) -> torch.Tensor:
    raw_hw = sample["raw_image_size_hw"].detach().cpu().long()
    if raw_hw.dim() == 3:
        return raw_hw[frame_idx, view_offset]
    if raw_hw.dim() == 2:
        if raw_hw.shape[0] == int(sample["cam_ids"].numel()):
            return raw_hw[view_offset]
        return raw_hw[frame_idx]
    return raw_hw.view(2)


def localize_objects(
    sample: dict[str, Any],
    clean_state: CleanSceneState,
    alignment: Sim3Transform,
    object_slots: list[int],
    min_match_score: float = 0.1,
    dynamic_thresh: float = 0.5,
    core_scale: float = 0.85,
    shell_scale: float = 1.05,
    proposal_scale: float = 1.25,
    min_candidate_points: int = 64,
    motion_speed_thresh: float = WAYMO_DYNAMIC_SPEED_THRESH_MPS,
    dynamic_prob_thresh: float | None = None,
    dynamic_ratio_thresh: float = 0.35,
    use_pose_refine: bool = True,
    max_pose_refine_yaw_deg: float = _MAX_POSE_REFINE_YAW_DEG,
    asset_yaw_correction_deg: float = 180.0,
    asset_cache: dict[str, dict[str, torch.Tensor]] | None = None,
    load_asset: bool = True,
) -> list[LocalizedFrameObject]:
    if dynamic_prob_thresh is None:
        dynamic_prob_thresh = dynamic_thresh
    num_views = int(sample["cam_ids"].numel())
    if num_views != 1:
        raise NotImplementedError(
            f"localize_objects currently supports views=1 only; got views={num_views}"
        )
    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    if asset_cache is None:
        asset_cache = {}
    asset_root = sample.get("asset_meta", {}).get("asset_root", "")
    track_valid = sample["object_track_valid_mask_selected"]
    object_obj_to_world = sample["object_obj_to_world_selected"]
    object_box_size = sample["object_box_size_selected"]
    object_box_corners_world = sample.get("object_box_corners_world_selected")
    object_max_speed_mps = sample.get("object_max_speed_mps")
    object_mean_speed_mps = sample.get("object_mean_speed_mps")
    object_is_moving_track = sample.get("object_is_moving_track")
    object_speed_mps = sample.get("object_speed_mps_selected")
    object_is_moving_frame = sample.get("object_is_moving_frame_selected")
    scene_up = alignment.rotation @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    scene_up = scene_up / scene_up.norm().clamp_min(1e-6)

    slot_meta: dict[int, dict[str, Any]] = {}
    slot_frame_geom: dict[int, dict[int, dict[str, Any]]] = {}

    for slot_idx in object_slots:
        if slot_idx < 0 or slot_idx >= track_valid.shape[0]:
            continue
        if not bool(sample["object_asset_valid_mask"][slot_idx].item()):
            continue
        match_score = float(sample["object_scene_match_scores"][slot_idx].item())
        if match_score < min_match_score:
            continue

        scene_raw_object_id = str(sample["object_scene_raw_ids"][slot_idx])
        asset_path = sample["object_asset_paths"][slot_idx]
        if asset_root and scene_raw_object_id and (not asset_path or not Path(str(asset_path)).is_file()):
            self_asset_path = f"{asset_root}/{scene_raw_object_id}.ply"
            try:
                import os
                if os.path.isfile(self_asset_path):
                    asset_path = self_asset_path
            except Exception:
                pass
        if not asset_path:
            continue
        asset_local = _load_asset_gaussians(asset_path, asset_cache) if load_asset else None
        waymo_max_speed = (
            float(object_max_speed_mps[slot_idx].item())
            if isinstance(object_max_speed_mps, torch.Tensor)
            else 0.0
        )
        waymo_mean_speed = (
            float(object_mean_speed_mps[slot_idx].item())
            if isinstance(object_mean_speed_mps, torch.Tensor)
            else 0.0
        )
        waymo_dynamic = (
            bool(object_is_moving_track[slot_idx].item())
            if isinstance(object_is_moving_track, torch.Tensor)
            else waymo_max_speed > motion_speed_thresh
        )

        frame_specs: list[dict[str, Any]] = []
        for frame_idx in range(track_valid.shape[1]):
            if not bool(track_valid[slot_idx, frame_idx].item()):
                continue
            view_offset, target_bbox_model = _select_localization_view(sample, slot_idx, frame_idx)
            if view_offset is None or target_bbox_model is None:
                continue

            gt_center, gt_size, gt_rotation = _transform_track_box(
                object_obj_to_world[slot_idx, frame_idx],
                object_box_size[slot_idx, frame_idx],
                alignment,
            )
            source_front_index = frame_idx * num_views + int(view_offset)
            label_track_rotation = _build_label_track_rotation(gt_rotation, scene_up)
            # Use the exact Sim3-transformed Waymo box rotation for projection
            # matching. Rebuilding an upright label rotation changes the cuboid's
            # projected shape and can make the DGGT/Waymo 3D boxes disagree even
            # before any yaw refinement is applied.
            proposal_rotation_base = _orthonormalize_rotation(gt_rotation)

            proposal_rotation_init = proposal_rotation_base.clone()
            proposal_center_init = gt_center.clone()
            proposal_bbox = _project_box_bbox(
                gt_center,
                gt_size,
                proposal_rotation_base,
                clean_state.camera_to_world[source_front_index],
                clean_state.intrinsics[source_front_index],
                image_hw,
            )
            if use_pose_refine:
                proposal_center_init, proposal_bbox = _solve_proposal_center_with_fixed_rotation(
                    object_center=gt_center,
                    object_size=gt_size,
                    object_rotation=proposal_rotation_base,
                    target_bbox_model=target_bbox_model,
                    camera_to_world=clean_state.camera_to_world[source_front_index],
                    intrinsics=clean_state.intrinsics[source_front_index],
                    image_hw=image_hw,
                    point_map_world=clean_state.point_map_world[source_front_index],
                    depth_map=clean_state.depth[source_front_index],
                    valid_mask=clean_state.valid_mask[source_front_index],
                )
            proposal_yaw_delta = 0.0
            proposal_score = -1e9 if proposal_bbox is None else _score_projected_bbox(proposal_bbox, target_bbox_model)

            waymo_corner_uv = torch.zeros((8, 2), dtype=torch.float32)
            waymo_corner_depths = torch.zeros((8,), dtype=torch.float32)
            waymo_corner_valid = torch.zeros((8,), dtype=torch.bool)
            waymo_edge_uv = torch.zeros((len(_BOX_EDGES), 2), dtype=torch.float32)
            waymo_edge_depths = torch.zeros((len(_BOX_EDGES),), dtype=torch.float32)
            waymo_edge_valid = torch.zeros((len(_BOX_EDGES),), dtype=torch.bool)
            waymo_bbox_model = None
            if isinstance(object_box_corners_world, torch.Tensor):
                projection_device = clean_state.camera_to_world[source_front_index].device
                projection_dtype = clean_state.camera_to_world[source_front_index].dtype
                corners_waymo_world = object_box_corners_world[slot_idx, frame_idx].detach().to(
                    device=projection_device,
                    dtype=projection_dtype,
                )
                if corners_waymo_world.shape == (8, 3) and torch.isfinite(corners_waymo_world).all():
                    camera_waymo = _sample_camera_to_world_view(sample, int(frame_idx), int(view_offset)).to(
                        device=projection_device,
                        dtype=projection_dtype,
                    )
                    intrinsics_waymo = _sample_intrinsics_view(sample, int(frame_idx), int(view_offset)).to(
                        device=projection_device,
                        dtype=projection_dtype,
                    )
                    raw_hw_waymo = _sample_raw_hw_view(sample, int(frame_idx), int(view_offset))
                    waymo_corner_uv, waymo_corner_depths, waymo_corner_valid = project_waymo_box_corners_model(
                        corners_waymo_world,
                        camera_waymo,
                        intrinsics_waymo,
                        raw_hw_waymo,
                        image_hw,
                    )
                    waymo_bbox_model = compute_bbox_from_projected_points(
                        waymo_corner_uv,
                        waymo_corner_valid,
                    )
                    waymo_edge_uv, waymo_edge_depths, waymo_edge_valid = project_waymo_box_corners_model(
                        _edge_midpoints(corners_waymo_world),
                        camera_waymo,
                        intrinsics_waymo,
                        raw_hw_waymo,
                        image_hw,
                    )

            waymo_frame_dynamic = (
                bool(object_is_moving_frame[slot_idx, frame_idx].item())
                if isinstance(object_is_moving_frame, torch.Tensor)
                else waymo_dynamic
            )
            waymo_frame_speed = (
                float(object_speed_mps[slot_idx, frame_idx].item())
                if isinstance(object_speed_mps, torch.Tensor)
                else waymo_mean_speed
            )
            frame_specs.append(
                {
                    "frame_idx": int(frame_idx),
                    "source_front_index": int(source_front_index),
                    "view_offset": int(view_offset),
                    "gt_center": gt_center,
                    "gt_size": gt_size,
                    "gt_rotation": gt_rotation,
                    "label_track_rotation": label_track_rotation,
                    "proposal_rotation_base": proposal_rotation_base,
                    "proposal_center_initial": proposal_center_init,
                    "proposal_rotation_initial": proposal_rotation_init,
                    "proposal_yaw_delta": float(proposal_yaw_delta),
                    "proposal_score_initial": proposal_score,
                    "waymo_frame_dynamic": bool(waymo_frame_dynamic),
                    "waymo_frame_speed": waymo_frame_speed,
                    "target_bbox_model": target_bbox_model,
                    "waymo_corner_uv": waymo_corner_uv,
                    "waymo_corner_depths": waymo_corner_depths,
                    "waymo_corner_valid": waymo_corner_valid,
                    "waymo_edge_uv": waymo_edge_uv,
                    "waymo_edge_depths": waymo_edge_depths,
                    "waymo_edge_valid": waymo_edge_valid,
                    "waymo_bbox_model": waymo_bbox_model,
                }
            )

        if len(frame_specs) == 0:
            continue

        corner_yaw_deltas: list[float] = []
        corner_yaw_scores: list[float] = []
        max_yaw_rad = max(0.0, math.radians(float(max_pose_refine_yaw_deg)))
        shared_yaw_delta = 0.0
        shared_yaw_ok = False
        shared_yaw_reason = "corner_refine_deferred_until_delete_support"

        slot_meta[int(slot_idx)] = {
            "match_score": match_score,
            "scene_raw_object_id": scene_raw_object_id,
            "asset_path": asset_path,
            "asset_local": asset_local,
            "waymo_max_speed": waymo_max_speed,
            "waymo_mean_speed": waymo_mean_speed,
            "shared_yaw_delta": shared_yaw_delta,
            "corner_yaw_candidate_count": len(corner_yaw_deltas),
            "corner_yaw_refine_reason": shared_yaw_reason,
        }
        frame_geom: dict[int, dict[str, Any]] = {}
        for frame_spec in frame_specs:
            frame_idx = frame_spec["frame_idx"]
            source_front_index = frame_spec["source_front_index"]
            gt_center = frame_spec["gt_center"]
            gt_size = frame_spec["gt_size"]
            if use_pose_refine:
                yaw_delta = float(shared_yaw_delta)
                yaw_tensor = torch.tensor(yaw_delta, dtype=gt_center.dtype)
                proposal_rotation = _orthonormalize_rotation(
                    frame_spec["proposal_rotation_base"] @ _rotation_z_tensor(yaw_tensor, dtype=gt_center.dtype)
                )
                proposal_center, _ = _solve_proposal_center_with_fixed_rotation(
                    object_center=frame_spec["proposal_center_initial"],
                    object_size=gt_size,
                    object_rotation=proposal_rotation,
                    target_bbox_model=frame_spec["target_bbox_model"],
                    camera_to_world=clean_state.camera_to_world[source_front_index],
                    intrinsics=clean_state.intrinsics[source_front_index],
                    image_hw=image_hw,
                    point_map_world=clean_state.point_map_world[source_front_index],
                    depth_map=clean_state.depth[source_front_index],
                    valid_mask=clean_state.valid_mask[source_front_index],
                )
            else:
                proposal_rotation = frame_spec["proposal_rotation_base"].clone()
                proposal_center = gt_center.clone()
            proposal_size = gt_size.clone()
            target_bbox_2d = frame_spec["target_bbox_model"].clone()
            target_corner_bbox = frame_spec.get("waymo_bbox_model")
            if target_corner_bbox is None:
                target_corner_bbox = frame_spec["target_bbox_model"]
            corner_pose_result = _corner_projection_result_for_pose(
                initial_center=frame_spec["proposal_center_initial"],
                initial_rotation=frame_spec["proposal_rotation_base"],
                refined_center=proposal_center,
                refined_rotation=proposal_rotation,
                object_size=gt_size,
                waymo_corner_uv=frame_spec["waymo_corner_uv"],
                waymo_corner_valid=frame_spec["waymo_corner_valid"],
                waymo_edge_uv=frame_spec["waymo_edge_uv"],
                waymo_edge_valid=frame_spec["waymo_edge_valid"],
                target_bbox_model=target_corner_bbox,
                camera_to_world=clean_state.camera_to_world[source_front_index],
                intrinsics=clean_state.intrinsics[source_front_index],
                status="accepted"
                if shared_yaw_ok
                else ("rejected" if len(corner_yaw_deltas) > 0 else "no_track_yaw_inliers"),
                reason=shared_yaw_reason if len(corner_yaw_deltas) > 0 else "corner_refine_had_no_accepted_track_yaw_candidates",
                accepted=shared_yaw_ok,
                shared_yaw_delta=float(shared_yaw_delta),
                track_yaw_candidate_count=len(corner_yaw_deltas),
            )

            frame_geom[int(frame_idx)] = {
                **frame_spec,
                "proposal_center": proposal_center,
                "proposal_center_raw": proposal_center.clone(),
                "proposal_rotation": proposal_rotation,
                "proposal_size": proposal_size,
                "target_bbox_2d": target_bbox_2d,
                "shared_yaw_delta": float(shared_yaw_delta),
                "corner_pose_result": corner_pose_result,
            }
        if use_pose_refine and len(frame_geom) > 1:
            deltas = torch.stack(
                [geom["proposal_center_raw"] - geom["gt_center"] for geom in frame_geom.values()],
                dim=0,
            )
            finite_delta = torch.isfinite(deltas).all(dim=-1)
            if bool(finite_delta.any().item()):
                shared_delta = torch.median(deltas[finite_delta], dim=0).values
                for geom in frame_geom.values():
                    geom["proposal_center"] = geom["gt_center"] + shared_delta.to(geom["gt_center"].dtype)
                    geom["proposal_center_shared_delta"] = shared_delta.clone()
                    target_corner_bbox = geom.get("waymo_bbox_model")
                    if target_corner_bbox is None:
                        target_corner_bbox = geom["target_bbox_model"]
                    geom["corner_pose_result"] = _corner_projection_result_for_pose(
                        initial_center=geom["proposal_center_initial"],
                        initial_rotation=geom["proposal_rotation_base"],
                        refined_center=geom["proposal_center"],
                        refined_rotation=geom["proposal_rotation"],
                        object_size=geom["gt_size"],
                        waymo_corner_uv=geom["waymo_corner_uv"],
                        waymo_corner_valid=geom["waymo_corner_valid"],
                        waymo_edge_uv=geom["waymo_edge_uv"],
                        waymo_edge_valid=geom["waymo_edge_valid"],
                        target_bbox_model=target_corner_bbox,
                        camera_to_world=clean_state.camera_to_world[geom["source_front_index"]],
                        intrinsics=clean_state.intrinsics[geom["source_front_index"]],
                        status="accepted"
                        if shared_yaw_ok
                        else ("rejected" if len(corner_yaw_deltas) > 0 else "no_track_yaw_inliers"),
                        reason=shared_yaw_reason if len(corner_yaw_deltas) > 0 else "corner_refine_had_no_accepted_track_yaw_candidates",
                        accepted=shared_yaw_ok,
                        shared_yaw_delta=float(shared_yaw_delta),
                        track_yaw_candidate_count=len(corner_yaw_deltas),
                    )
        slot_frame_geom[int(slot_idx)] = frame_geom

    if not slot_frame_geom:
        return []

    per_frame_components: dict[int, list[dict[str, Any]]] = {}
    total_frames = clean_state.semantic_vehicle_mask.shape[0]
    for img_idx in range(total_frames):
        raw_mask = clean_state.semantic_vehicle_mask[img_idx]
        if int(raw_mask.sum().item()) == 0:
            per_frame_components[img_idx] = []
            continue
        clean_mask = _morphology_opening(raw_mask, radius=1)
        per_frame_components[img_idx] = _connected_components_4(clean_mask)

    localized: list[LocalizedFrameObject] = []

    for slot_idx, frames in slot_frame_geom.items():
        meta = slot_meta[slot_idx]
        asset_local = meta["asset_local"]
        scene_raw_object_id = meta["scene_raw_object_id"]
        asset_path = meta["asset_path"]
        match_score = meta["match_score"]
        waymo_max_speed = meta["waymo_max_speed"]
        waymo_mean_speed = meta["waymo_mean_speed"]

        frame_records: list[dict[str, Any]] = []
        shared_yaw_delta = float(meta.get("shared_yaw_delta", 0.0))
        corner_yaw_candidate_count = int(meta.get("corner_yaw_candidate_count", 0))

        for frame_idx, geom in frames.items():
            source_front_index = int(geom["source_front_index"])
            view_offset = int(geom["view_offset"])
            gt_center = geom["gt_center"]
            gt_size = geom["gt_size"]
            gt_rotation = geom["gt_rotation"]
            proposal_center = geom["proposal_center"]
            proposal_rotation = geom["proposal_rotation"]
            proposal_size = geom["proposal_size"]
            target_bbox_model = geom["target_bbox_model"]
            target_bbox_2d = geom["target_bbox_2d"]
            waymo_frame_dynamic = bool(geom["waymo_frame_dynamic"])
            waymo_frame_speed = float(geom["waymo_frame_speed"])

            raw_mask = clean_state.semantic_vehicle_mask[source_front_index]
            valid_mask_view = clean_state.valid_mask[source_front_index]
            semantic_prob_view = clean_state.semantic_vehicle_prob[source_front_index]
            if target_bbox_2d is None:
                continue
            if int((raw_mask & valid_mask_view).sum().item()) < 16 and float(semantic_prob_view.max().item()) < 0.18:
                continue

            protected_boxes = _collect_protected_boxes(
                sample,
                alignment,
                target_slot_idx=int(slot_idx),
                frame_idx=int(frame_idx),
                view_offset=view_offset,
            )

            other_boxes_2d: list[torch.Tensor] = []
            for prot in protected_boxes:
                if prot.get("bbox_model") is not None:
                    other_boxes_2d.append(prot["bbox_model"])

            matched_mask = _select_target_semantic_mask(
                raw_mask=raw_mask,
                semantic_prob=semantic_prob_view,
                point_map_world_view=clean_state.point_map_world[source_front_index],
                depth_map_view=clean_state.depth[source_front_index],
                valid_mask_view=valid_mask_view,
                target_bbox_model=target_bbox_model,
                other_boxes_2d=other_boxes_2d,
                coarse_center=gt_center,
                coarse_rotation=gt_rotation,
                coarse_size=gt_size,
            )
            matched_pixel_count = int(matched_mask.sum().item())
            if matched_pixel_count < 16:
                continue

            matched_points = clean_state.point_map_world[source_front_index][matched_mask & valid_mask_view].reshape(-1, 3)
            if matched_points.shape[0] < 16:
                continue
            refined_center, refined_size = _fit_delete_box_from_component_points(
                matched_points,
                gt_center,
                gt_rotation,
                gt_size,
            )

            visible_seed_mask = _gaussian_indices_from_pixel_mask(
                clean_state,
                matched_mask,
                view_image_idx=source_front_index,
            )
            visible_seed_indices = torch.nonzero(visible_seed_mask, as_tuple=False).flatten().to(torch.long)
            if visible_seed_indices.numel() == 0:
                continue

            delete_motion_mode = "dynamic" if waymo_frame_dynamic else "static"

            scene_delete_info = _expand_delete_cluster_to_scene(
                clean_state=clean_state,
                seed_indices=visible_seed_indices,
                support_points_world=matched_points,
                refined_center=refined_center,
                refined_rotation=gt_rotation,
                refined_size=refined_size,
                source_front_index=source_front_index,
                target_bbox_model=target_bbox_model,
                target_mask_2d=matched_mask,
                protected_boxes=protected_boxes,
                delete_motion_mode=delete_motion_mode,
                dynamic_prob_thresh=dynamic_prob_thresh,
            )
            if waymo_frame_dynamic and int(scene_delete_info["cluster_kept_count"]) < 24:
                static_scene_delete_info = _expand_delete_cluster_to_scene(
                    clean_state=clean_state,
                    seed_indices=visible_seed_indices,
                    support_points_world=matched_points,
                    refined_center=refined_center,
                    refined_rotation=gt_rotation,
                    refined_size=refined_size,
                    source_front_index=source_front_index,
                    target_bbox_model=target_bbox_model,
                    target_mask_2d=matched_mask,
                    protected_boxes=protected_boxes,
                    delete_motion_mode="static",
                    dynamic_prob_thresh=dynamic_prob_thresh,
                )
                if int(static_scene_delete_info["cluster_kept_count"]) > int(scene_delete_info["cluster_kept_count"]):
                    scene_delete_info = static_scene_delete_info
                    scene_delete_info["delete_motion_mode"] = "dynamic_waymo_static_render"
            if int(scene_delete_info["cluster_kept_count"]) == 0:
                continue

            frame_records.append(
                {
                    "frame_idx": int(frame_idx),
                    "geom": geom,
                    "matched_mask": matched_mask,
                    "matched_points": matched_points,
                    "matched_pixel_count": matched_pixel_count,
                    "visible_seed_indices": visible_seed_indices,
                    "scene_delete_info": scene_delete_info,
                    "refined_center": refined_center,
                    "refined_size": refined_size,
                    "waymo_frame_speed": waymo_frame_speed,
                }
            )

        if len(frame_records) == 0:
            continue

        localized_start = len(localized)
        for record in frame_records:
            frame_idx = int(record["frame_idx"])
            geom = record["geom"]
            source_front_index = int(geom["source_front_index"])
            gt_center = geom["gt_center"]
            gt_size = geom["gt_size"]
            gt_rotation = geom["gt_rotation"]
            proposal_rotation_final = geom["proposal_rotation"]
            proposal_center_final = geom["proposal_center"]
            proposal_size_final = geom["proposal_size"]
            target_bbox_model = geom["target_bbox_model"]
            matched_mask = record["matched_mask"]
            matched_points = record["matched_points"]
            matched_pixel_count = int(record["matched_pixel_count"])
            visible_seed_indices = record["visible_seed_indices"]
            scene_delete_info = record["scene_delete_info"]
            refined_center = record["refined_center"]
            refined_size = record["refined_size"]
            waymo_frame_speed = float(record["waymo_frame_speed"])

            pose_result = geom.get("corner_pose_result")
            target_corner_bbox = geom.get("waymo_bbox_model")
            if target_corner_bbox is None:
                target_corner_bbox = target_bbox_model
            if use_pose_refine and max_yaw_rad > 0.0:
                pose_result = _solve_corner_projection_pose(
                    base_center=proposal_center_final,
                    initial_center=proposal_center_final,
                    object_size=proposal_size_final,
                    base_rotation=proposal_rotation_final,
                    waymo_corner_uv=geom["waymo_corner_uv"],
                    waymo_corner_depths=geom["waymo_corner_depths"],
                    waymo_corner_valid=geom["waymo_corner_valid"],
                    waymo_edge_uv=geom["waymo_edge_uv"],
                    waymo_edge_depths=geom["waymo_edge_depths"],
                    waymo_edge_valid=geom["waymo_edge_valid"],
                    target_bbox_model=target_corner_bbox,
                    camera_to_world=clean_state.camera_to_world[source_front_index],
                    intrinsics=clean_state.intrinsics[source_front_index],
                    image_hw=image_hw,
                    support_points_world=matched_points,
                    max_yaw_rad=max_yaw_rad,
                    optimize_yaw=True,
                )
                geom["corner_pose_result"] = pose_result
                if bool(pose_result.get("accepted", False)):
                    proposal_center_final = pose_result["center"].to(device=gt_center.device, dtype=gt_center.dtype)
                    proposal_size_final = pose_result["size"].to(device=gt_size.device, dtype=gt_size.dtype)
                    proposal_rotation_final = pose_result["rotation"].to(device=gt_rotation.device, dtype=gt_rotation.dtype)
                    geom["proposal_center"] = proposal_center_final
                    geom["proposal_size"] = proposal_size_final
                    geom["proposal_rotation"] = proposal_rotation_final
            if pose_result is None:
                pose_result = _base_corner_projection_result(
                    base_center=proposal_center_final,
                    object_size=gt_size,
                    base_rotation=proposal_rotation_final,
                    waymo_corner_uv=geom["waymo_corner_uv"],
                    waymo_corner_valid=geom["waymo_corner_valid"],
                    waymo_edge_uv=geom["waymo_edge_uv"],
                    waymo_edge_valid=geom["waymo_edge_valid"],
                    target_bbox_model=target_bbox_model,
                    camera_to_world=clean_state.camera_to_world[source_front_index],
                    intrinsics=clean_state.intrinsics[source_front_index],
                    status="disabled",
                    reason="pose_refine_disabled",
                )
            pose_result["diagnostics"]["track_yaw_candidate_count"] = corner_yaw_candidate_count

            delete_core_indices = scene_delete_info["delete_core_indices"]
            delete_shell_indices = scene_delete_info["delete_shell_indices"]
            candidate_count = int(scene_delete_info["candidate_pool_count"])
            cluster_kept_count = int(scene_delete_info["cluster_kept_count"])
            delete_motion_mode = str(scene_delete_info["delete_motion_mode"])
            if delete_core_indices.numel() > 0:
                render_dynamic_ratio = float(
                    (clean_state.dynamic_prob[delete_core_indices] >= dynamic_prob_thresh).float().mean().item()
                )
            else:
                render_dynamic_ratio = 0.0

            frame_rotation = proposal_rotation_final
            insert_size = proposal_size_final.clone()
            asset_center = proposal_center_final.clone()
            proposal_center = asset_center.clone()
            proposal_rotation = frame_rotation.clone()
            proposal_size = insert_size.clone()
            used_yaw_delta = float(pose_result.get("yaw_delta", 0.0))
            asset_object_to_world = _asset_object_to_world_matrix(frame_rotation, asset_center)
            if asset_local is not None:
                asset_scale_factors = _compute_asset_scale_factors(asset_local, insert_size)
                asset_world = _transform_asset_gaussians_simple(
                    asset_local,
                    insert_size,
                    frame_rotation,
                    asset_center,
                    asset_yaw_correction_deg=asset_yaw_correction_deg,
                )
                projected_asset_bbox = _project_asset_bbox_simple(
                    asset_local=asset_local,
                    target_lwh=insert_size,
                    object_rotation=frame_rotation,
                    object_center=asset_center,
                    camera_to_world=clean_state.camera_to_world[source_front_index],
                    intrinsics=clean_state.intrinsics[source_front_index],
                    image_hw=image_hw,
                    asset_yaw_correction_deg=asset_yaw_correction_deg,
                )
                asset_scale = float(
                    torch.mean(
                        insert_size
                        / (asset_local["means_raw"].max(dim=0).values - asset_local["means_raw"].min(dim=0).values).clamp_min(1e-6)
                    ).item()
                )
                asset_means_world = asset_world["means"]
                asset_colors = asset_world["colors"]
                asset_opacities = asset_world["opacities"]
                asset_scales_world = asset_world["scales"]
                asset_quats_world = asset_world["quats"]
                asset_means_local = asset_local["means_raw"]
                asset_scales_local = asset_local["scales"]
                asset_quats_local = asset_local["quats"]
            else:
                asset_scale_factors = torch.ones(3, dtype=torch.float32)
                projected_asset_bbox = None
                asset_scale = 1.0
                asset_means_world = torch.zeros((0, 3), dtype=torch.float32)
                asset_colors = torch.zeros((0, 3), dtype=torch.float32)
                asset_opacities = torch.zeros((0, 1), dtype=torch.float32)
                asset_scales_world = torch.zeros((0, 3), dtype=torch.float32)
                asset_quats_world = torch.zeros((0, 4), dtype=torch.float32)
                asset_means_local = torch.zeros((0, 3), dtype=torch.float32)
                asset_scales_local = torch.zeros((0, 3), dtype=torch.float32)
                asset_quats_local = torch.zeros((0, 4), dtype=torch.float32)

            localized.append(
                LocalizedFrameObject(
                    slot_idx=int(slot_idx),
                    frame_idx=int(frame_idx),
                    source_front_index=int(source_front_index),
                    asset_object_id=str(sample["object_asset_ids"][slot_idx]),
                    scene_raw_object_id=scene_raw_object_id,
                    asset_path=str(asset_path),
                    match_score=match_score,
                    delete_motion_mode=delete_motion_mode,
                    waymo_frame_dynamic=bool(waymo_frame_dynamic),
                    waymo_frame_speed_mps=waymo_frame_speed,
                    waymo_max_speed_mps=waymo_max_speed,
                    waymo_mean_speed_mps=waymo_mean_speed,
                    render_dynamic_ratio=render_dynamic_ratio,
                    gt_center=gt_center,
                    gt_size=gt_size,
                    gt_rotation=gt_rotation,
                    proposal_center=proposal_center,
                    proposal_size=proposal_size,
                    proposal_rotation=proposal_rotation,
                    refined_center=refined_center,
                    refined_size=refined_size,
                    refined_rotation=frame_rotation,
                    asset_rotation=frame_rotation,
                    asset_scale=asset_scale,
                    asset_bottom_center=asset_center,
                    delete_core_indices=delete_core_indices,
                    delete_shell_indices=delete_shell_indices,
                    candidate_count=matched_pixel_count,
                    seed_point_count=int(visible_seed_indices.numel()),
                    candidate_pool_count=candidate_count,
                    cluster_kept_count=cluster_kept_count,
                    target_delete_coverage=1.0,
                    outside_box_leak_ratio=0.0,
                    target_bbox_model=target_bbox_model,
                    projected_asset_bbox=projected_asset_bbox,
                    seed_pixel_mask=matched_mask.clone(),
                    delete_component_pixel_mask=matched_mask.clone(),
                    asset_means_world=asset_means_world,
                    asset_colors=asset_colors,
                    asset_opacities=asset_opacities,
                    asset_scales=asset_scales_world,
                    asset_quats=asset_quats_world,
                    asset_means_local=asset_means_local,
                    asset_scales_local=asset_scales_local,
                    asset_quats_local=asset_quats_local,
                    asset_scale_factors=asset_scale_factors,
                    asset_object_to_world=asset_object_to_world,
                    asset_local_yaw_deg=float(asset_yaw_correction_deg) + float(math.degrees(used_yaw_delta)),
                    pose_refine_diagnostics=pose_result["diagnostics"],
                    waymo_box_corners_model=pose_result["waymo_uv"],
                    waymo_box_corners_valid=pose_result["waymo_valid"],
                    initial_box_corners_model=pose_result["initial_uv"],
                    initial_box_corners_valid=pose_result["initial_valid"],
                    refined_box_corners_model=pose_result["refined_uv"],
                    refined_box_corners_valid=pose_result["refined_valid"],
                )
            )
        _stabilize_static_localized_segments(
            localized[localized_start:],
            motion_speed_thresh=motion_speed_thresh,
            clean_state=clean_state,
            image_hw=image_hw,
            asset_yaw_correction_deg=asset_yaw_correction_deg,
        )

    return localized

def apply_mode_a(clean_state: CleanSceneState, localized_objects: list[LocalizedFrameObject]) -> EditedSceneState:
    clean = {
        "means": clean_state.means,
        "colors": clean_state.colors,
        "opacities": clean_state.opacities,
        "scales": clean_state.scales,
        "quats": clean_state.quats,
    }
    delete_mask = torch.zeros((clean_state.means.shape[0],), dtype=torch.bool)
    shell_mask = torch.zeros((clean_state.means.shape[0],), dtype=torch.bool)

    for item in localized_objects:
        if item.delete_core_indices.numel() > 0:
            delete_mask[item.delete_core_indices] = True
        if item.delete_shell_indices.numel() > 0:
            shell_mask[item.delete_shell_indices] = True
            delete_mask[item.delete_shell_indices] = True
    deleted = _subset_gaussians(clean, ~delete_mask)
    asset_only = _concat_gaussians([])
    edited = deleted

    return EditedSceneState(
        clean=clean,
        deleted=deleted,
        asset_only=asset_only,
        edited=edited,
        localized_objects=localized_objects,
        delete_mask=delete_mask,
        shell_mask=shell_mask,
    )


def render_gaussians_as_points(
    scene: dict[str, torch.Tensor],
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    max_points: int = 250000,
) -> torch.Tensor:
    means = scene["means"].detach().cpu().float()
    colors = scene["colors"].detach().cpu().float().clamp(0.0, 1.0)
    opacities = scene["opacities"].detach().cpu().float().view(-1).clamp(0.0, 1.0)

    seq_len = camera_to_world.shape[0]
    height, width = image_hw
    renders = []

    if means.shape[0] == 0:
        return torch.zeros((seq_len, 3, height, width), dtype=torch.float32)

    if means.shape[0] > max_points:
        keep = torch.topk(opacities, k=max_points, largest=True).indices
        means = means[keep]
        colors = colors[keep]
        opacities = opacities[keep]

    for view_idx in range(seq_len):
        uv, depths, valid = project_world_points(means, camera_to_world[view_idx], intrinsics[view_idx], image_hw)
        if valid.sum().item() == 0:
            renders.append(torch.zeros((3, height, width), dtype=torch.float32))
            continue

        uv = uv[valid]
        depths = depths[valid]
        colors_valid = colors[valid]
        alpha_valid = opacities[valid]
        u = uv[:, 0].long().clamp(0, width - 1)
        v = uv[:, 1].long().clamp(0, height - 1)
        flat_idx = v * width + u

        order = torch.argsort(depths, descending=True)
        flat_idx = flat_idx[order]
        colors_valid = colors_valid[order]
        alpha_valid = alpha_valid[order]

        canvas = torch.zeros((3, height * width), dtype=torch.float32)
        canvas[:, flat_idx] = (colors_valid * alpha_valid.unsqueeze(-1)).T
        image = canvas.view(3, height, width)
        image = F.avg_pool2d(image.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
        renders.append(image.clamp(0.0, 1.0))

    return torch.stack(renders, dim=0)
