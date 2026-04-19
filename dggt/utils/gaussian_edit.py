from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F

from dggt.utils.gaussian_ply import read_gaussian_ply
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.rotation import mat_to_quat, quat_to_mat

_GS_LWH_TO_XYZ_SCALE = (0.90, 0.90, 0.88)
WAYMO_DYNAMIC_SPEED_THRESH_MPS = 1.0
_DYNAMIC_VOXEL_SCALE = (0.10, 0.10, 0.20)
_STATIC_VOXEL_SCALE = (0.08, 0.08, 0.16)


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
    gt_right = gt_camera_to_world[:, :3, 0]
    pred_right = pred_camera_to_world[:, :3, 0]
    gt_forward = gt_camera_to_world[:, :3, 2]
    pred_forward = pred_camera_to_world[:, :3, 2]

    source = torch.cat([gt_centers, gt_centers + 0.5 * gt_right, gt_centers + 0.5 * gt_forward], dim=0)
    target = torch.cat(
        [pred_centers, pred_centers + 0.5 * pred_right, pred_centers + 0.5 * pred_forward],
        dim=0,
    )
    return _estimate_sim3_umeyama(source, target)


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
    x1 = float(valid_uv[:, 0].min())
    y1 = float(valid_uv[:, 1].min())
    x2 = float(valid_uv[:, 0].max())
    y2 = float(valid_uv[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


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


def _points_in_box(
    points: torch.Tensor,
    center: torch.Tensor,
    rotation: torch.Tensor,
    size: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    local = (points - center) @ rotation
    half = size * (0.5 * scale)
    return (local.abs() <= (half + 1e-5)).all(dim=-1)


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
    protected_count = int(sample.get("protected_object_count", torch.tensor(0)).item())
    if protected_count > 0:
        candidate_slots = sample["protected_object_indices"][:protected_count].tolist()
    else:
        candidate_slots = []

    if len(candidate_slots) == 0:
        total_objects = int(sample["object_track_valid_mask_selected"].shape[0])
        candidate_slots = [slot for slot in range(total_objects) if slot != target_slot_idx]

    for slot_idx in candidate_slots:
        if slot_idx == target_slot_idx:
            continue
        if not bool(sample["object_valid_mask"][slot_idx].item()):
            continue
        if not bool(sample["object_track_valid_mask_selected"][slot_idx, frame_idx].item()):
            continue
        if "object_bbox_valid_mask_selected" in sample:
            if not bool(sample["object_bbox_valid_mask_selected"][slot_idx, frame_idx, view_offset].item()):
                continue
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


def _rotation_fix_z_180(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


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
        center_xy_init = _unproject_center_xy(target_bbox_model, depth_init, intrinsics)
        center_xy = center_xy_init.clone().detach().requires_grad_(True)
        log_depth = depth_init.log().clone().detach().requires_grad_(True)
        yaw_param = torch.zeros((), dtype=object_center.dtype, requires_grad=True)
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

    asset = read_gaussian_ply(path)
    means = torch.tensor(asset["means"].tolist(), dtype=torch.float32)
    colors = torch.tensor(asset["rgb"].tolist(), dtype=torch.float32).clamp(0.0, 1.0)
    opacities = torch.tensor(asset["opacities"].tolist(), dtype=torch.float32).view(-1, 1).clamp(1e-6, 1.0 - 1e-6)
    scales = torch.tensor(asset["scales"].tolist(), dtype=torch.float32).clamp_min(1e-6)
    quats = torch.tensor(asset["quats"].tolist(), dtype=torch.float32)

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
    opacities = asset_local["opacities"].squeeze(-1)
    visible = opacities > opacity_threshold
    visible_xyz = asset_local["means_raw"][visible] if torch.any(visible) else asset_local["means_raw"]
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
    opacity_threshold: float = 0.01
) -> dict[str, torch.Tensor]:
    scale_factors = _compute_asset_scale_factors(asset_local, target_lwh, opacity_threshold=opacity_threshold)
    world_rot = object_rotation @ _rotation_fix_z_180(dtype=object_rotation.dtype).to(object_rotation.device)

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
    world_rot = object_rotation @ _rotation_fix_z_180(dtype=object_rotation.dtype).to(object_rotation.device)
    means_world = visible_xyz.to(target_lwh.device) * scale_factors.view(1, 3)
    means_world = means_world @ world_rot.T + object_center
    uv, _, valid = project_world_points(means_world, camera_to_world, intrinsics, image_hw)
    return compute_bbox_from_projected_points(uv, valid)


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
        voxel_scale = (refined_size * torch.tensor(_DYNAMIC_VOXEL_SCALE, dtype=gt_size.dtype)).clamp_min(0.035)
        connectivity = 26
    else:
        candidate_local = proposal_box_local & depth_band_local & bbox_outer_local
        seed_local = target_local | (foreground_local & proposal_box_local)
        if int(seed_local.sum().item()) < 12:
            seed_local = support_local & proposal_box_local
        if int(seed_local.sum().item()) < 12:
            seed_local = support_local & core_box_local
        voxel_scale = (refined_size * torch.tensor(_STATIC_VOXEL_SCALE, dtype=gt_size.dtype)).clamp_min(0.04)
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
    bbox_valid = sample.get("object_bbox_valid_mask_selected")
    bbox_model = sample.get("object_bbox_model_selected")

    if isinstance(bbox_valid, torch.Tensor) and isinstance(bbox_model, torch.Tensor):
        valid_view_offsets = torch.nonzero(bbox_valid[slot_idx, frame_idx], as_tuple=False).flatten()
        if valid_view_offsets.numel() == 0:
            return None, None

        front_offsets = torch.nonzero(cam_ids == 0, as_tuple=False).flatten()
        if front_offsets.numel() > 0:
            front_offset = int(front_offsets[0].item())
            if bool(bbox_valid[slot_idx, frame_idx, front_offset].item()):
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
    if not bool(sample["object_front_bbox_valid_mask_selected"][slot_idx, frame_idx].item()):
        return None, None
    return (
        front_offset,
        sample["object_front_bbox_model_selected"][slot_idx, frame_idx].detach().cpu().float(),
    )


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
) -> list[LocalizedFrameObject]:
    if dynamic_prob_thresh is None:
        dynamic_prob_thresh = dynamic_thresh
    num_views = int(sample["cam_ids"].numel())
    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    asset_cache: dict[str, dict[str, torch.Tensor]] = {}
    asset_root = sample.get("asset_meta", {}).get("asset_root", "")
    localized: list[LocalizedFrameObject] = []
    track_valid = sample["object_track_valid_mask_selected"]
    object_obj_to_world = sample["object_obj_to_world_selected"]
    object_box_size = sample["object_box_size_selected"]
    object_max_speed_mps = sample.get("object_max_speed_mps")
    object_mean_speed_mps = sample.get("object_mean_speed_mps")
    object_is_moving_track = sample.get("object_is_moving_track")
    object_speed_mps = sample.get("object_speed_mps_selected")
    object_is_moving_frame = sample.get("object_is_moving_frame_selected")
    scene_up = alignment.rotation @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    scene_up = scene_up / scene_up.norm().clamp_min(1e-6)

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
        if asset_root and scene_raw_object_id:
            self_asset_path = f"{asset_root}/{scene_raw_object_id}.ply"
            try:
                import os
                if os.path.isfile(self_asset_path):
                    asset_path = self_asset_path
            except Exception:
                pass
        if not asset_path:
            continue
        asset_local = _load_asset_gaussians(asset_path, asset_cache)
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
            proposal_rotation_base = _build_label_track_rotation(gt_rotation, scene_up)
            proposal_center, proposal_rotation, proposal_bbox = _solve_proposal_pose_from_target_bbox(
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
            proposal_score = -1e9 if proposal_bbox is None else _score_projected_bbox(proposal_bbox, target_bbox_model)
            waymo_frame_dynamic = (
                bool(object_is_moving_frame[slot_idx, frame_idx].item())
                if isinstance(object_is_moving_frame, torch.Tensor)
                else waymo_dynamic
            )
            frame_specs.append(
                {
                    "frame_idx": int(frame_idx),
                    "source_front_index": int(source_front_index),
                    "gt_center": gt_center,
                    "gt_size": gt_size,
                    "gt_rotation": gt_rotation,
                    "proposal_rotation_base": proposal_rotation_base,
                    "proposal_center_initial": proposal_center,
                    "proposal_rotation_initial": proposal_rotation,
                    "proposal_score_initial": proposal_score,
                    "waymo_frame_dynamic": bool(waymo_frame_dynamic),
                    "target_bbox_model": target_bbox_model,
                }
            )

        if len(frame_specs) == 0:
            continue

        shared_yaw_delta = _robust_shared_yaw_delta(
            [
                _yaw_delta_between_rotations(spec["proposal_rotation_base"], spec["proposal_rotation_initial"])
                for spec in frame_specs
            ],
            [float(spec["proposal_score_initial"]) for spec in frame_specs],
        )
        shared_track_rotation = _shared_track_rotation_candidate(
            [spec["proposal_rotation_initial"] for spec in frame_specs],
            [float(spec["proposal_score_initial"]) for spec in frame_specs],
        )

        for frame_spec_idx, frame_spec in enumerate(frame_specs):
            frame_idx = frame_spec["frame_idx"]
            source_front_index = frame_spec["source_front_index"]
            view_offset = int(source_front_index % num_views)
            gt_center = frame_spec["gt_center"]
            gt_size = frame_spec["gt_size"]
            gt_rotation = frame_spec["gt_rotation"]
            if shared_track_rotation is not None:
                proposal_rotation = shared_track_rotation.clone()
            else:
                proposal_rotation_base = frame_spec["proposal_rotation_base"]
                shared_yaw_tensor = torch.tensor(shared_yaw_delta, dtype=gt_center.dtype)
                proposal_rotation = _orthonormalize_rotation(
                    proposal_rotation_base @ _rotation_z_tensor(shared_yaw_tensor, dtype=gt_center.dtype)
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
            delete_center, proposal_size = _refine_proposal_box_from_foreground_seed(
                point_map_world=clean_state.point_map_world[source_front_index],
                depth_map=clean_state.depth[source_front_index],
                valid_mask=clean_state.valid_mask[source_front_index],
                box_xyxy=frame_spec["target_bbox_model"],
                object_center=proposal_center,
                object_size=gt_size,
                object_rotation=proposal_rotation,
            )
            frame_rotation = proposal_rotation
            target_bbox_model_raw = frame_spec["target_bbox_model"]
            target_bbox_model = target_bbox_model_raw
            waymo_frame_speed = (
                float(object_speed_mps[slot_idx, frame_idx].item())
                if isinstance(object_speed_mps, torch.Tensor)
                else waymo_mean_speed
            )
            waymo_frame_dynamic = bool(frame_spec.get("waymo_frame_dynamic", waymo_dynamic))
            dynamic_image_mask = _binary_mask_from_image(sample["dynamic_mask"][source_front_index])
            protected_boxes = _collect_protected_boxes(
                sample,
                alignment,
                target_slot_idx=int(slot_idx),
                frame_idx=int(frame_idx),
                view_offset=view_offset,
            )
            candidate_points, object_pixel_mask = _extract_object_pixels_from_bbox_component(
                point_map_world=clean_state.point_map_world[source_front_index],
                depth_map=clean_state.depth[source_front_index],
                valid_mask=clean_state.valid_mask[source_front_index],
                box_xyxy=target_bbox_model,
                dynamic_image_mask=dynamic_image_mask,
                dynamic_mode=bool(waymo_frame_dynamic),
                coarse_center=delete_center,
                coarse_rotation=proposal_rotation,
            )
            semantic_support = _extract_semantic_object_component_3d(
                clean_state=clean_state,
                source_front_index=source_front_index,
                target_bbox_model=target_bbox_model,
                proposal_center=delete_center,
                proposal_rotation=proposal_rotation,
                proposal_size=proposal_size,
                protected_boxes=protected_boxes,
                proposal_scale=proposal_scale,
                dynamic_mode=bool(waymo_frame_dynamic),
            )
            if semantic_support["points"].shape[0] >= 24:
                semantic_pixel_mask = semantic_support["pixel_mask"]
                object_pixel_mask = semantic_pixel_mask | (
                    object_pixel_mask & _dilate_mask(semantic_pixel_mask, radius=2)
                )
                support_points_for_fit = clean_state.point_map_world[source_front_index][object_pixel_mask].reshape(-1, 3)
            else:
                support_points_for_fit = candidate_points

            delete_center_fit, delete_size_fit = _fit_box_from_support_points(
                support_points_for_fit,
                coarse_center=delete_center,
                coarse_rotation=proposal_rotation,
            )
            if support_points_for_fit.shape[0] < max(24, min_candidate_points // 2):
                continue
            delete_center = delete_center_fit
            proposal_size = delete_size_fit
            delete_motion_mode = "dynamic" if waymo_frame_dynamic else "static"
            delete_result = _extract_delete_component(
                clean_state=clean_state,
                source_front_index=source_front_index,
                gt_center=delete_center,
                gt_rotation=proposal_rotation,
                gt_size=proposal_size,
                box_xyxy=target_bbox_model,
                object_pixel_mask=object_pixel_mask,
                dynamic_image_mask=dynamic_image_mask,
                protected_boxes=protected_boxes,
                delete_motion_mode=delete_motion_mode,
                dynamic_prob_thresh=dynamic_prob_thresh,
                core_scale=core_scale,
                shell_scale=shell_scale,
                proposal_scale=proposal_scale,
            )
            if waymo_frame_dynamic and (
                delete_result["render_dynamic_ratio"] < dynamic_ratio_thresh
                or delete_result["cluster_kept_count"] < max(24, min_candidate_points // 4)
            ):
                delete_motion_mode = "dynamic_waymo_static_render"
                delete_result = _extract_delete_component(
                    clean_state=clean_state,
                    source_front_index=source_front_index,
                    gt_center=delete_center,
                    gt_rotation=proposal_rotation,
                    gt_size=proposal_size,
                    box_xyxy=target_bbox_model,
                    object_pixel_mask=object_pixel_mask,
                    dynamic_image_mask=dynamic_image_mask,
                    protected_boxes=protected_boxes,
                    delete_motion_mode="static",
                    dynamic_prob_thresh=dynamic_prob_thresh,
                    core_scale=core_scale,
                    shell_scale=shell_scale,
                    proposal_scale=proposal_scale,
                )

            if delete_result["cluster_kept_count"] == 0:
                continue

            refined_center = delete_result["refined_center"].clone()
            refined_size = delete_result["refined_size"].clone()
            insert_size = gt_size.clone()
            asset_center = proposal_center.clone()
            asset_scale_factors = _compute_asset_scale_factors(asset_local, insert_size)
            asset_object_to_world = _asset_object_to_world_matrix(frame_rotation, asset_center)
            asset_world = _transform_asset_gaussians_simple(
                asset_local,
                insert_size,
                frame_rotation,
                asset_center,
            )
            projected_asset_bbox = _project_asset_bbox_simple(
                asset_local=asset_local,
                target_lwh=insert_size,
                object_rotation=frame_rotation,
                object_center=asset_center,
                camera_to_world=clean_state.camera_to_world[source_front_index],
                intrinsics=clean_state.intrinsics[source_front_index],
                image_hw=image_hw,
            )
            asset_scale = float(
                torch.mean(
                    insert_size
                    / (asset_local["means_raw"].max(dim=0).values - asset_local["means_raw"].min(dim=0).values).clamp_min(1e-6)
                ).item()
            )

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
                    waymo_frame_speed_mps=waymo_frame_speed,
                    waymo_max_speed_mps=waymo_max_speed,
                    waymo_mean_speed_mps=waymo_mean_speed,
                    render_dynamic_ratio=float(delete_result["render_dynamic_ratio"]),
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
                    delete_core_indices=delete_result["delete_core_indices"],
                    delete_shell_indices=delete_result["delete_shell_indices"],
                    candidate_count=int(delete_result["candidate_pool_count"]),
                    seed_point_count=int(delete_result["seed_point_count"]),
                    candidate_pool_count=int(delete_result["candidate_pool_count"]),
                    cluster_kept_count=int(delete_result["cluster_kept_count"]),
                    target_delete_coverage=float(delete_result["target_delete_coverage"]),
                    outside_box_leak_ratio=float(delete_result["outside_box_leak_ratio"]),
                    target_bbox_model=target_bbox_model,
                    projected_asset_bbox=projected_asset_bbox,
                    seed_pixel_mask=delete_result["seed_pixel_mask"],
                    delete_component_pixel_mask=delete_result["delete_component_pixel_mask"],
                    asset_means_world=asset_world["means"],
                    asset_colors=asset_world["colors"],
                    asset_opacities=asset_world["opacities"],
                    asset_scales=asset_world["scales"],
                    asset_quats=asset_world["quats"],
                    asset_means_local=asset_local["means_raw"],
                    asset_scales_local=asset_local["scales"],
                    asset_quats_local=asset_local["quats"],
                    asset_scale_factors=asset_scale_factors,
                    asset_object_to_world=asset_object_to_world,
                )
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
    asset_chunks: list[dict[str, torch.Tensor]] = []

    for item in localized_objects:
        if item.delete_core_indices.numel() > 0:
            delete_mask[item.delete_core_indices] = True
        if item.delete_shell_indices.numel() > 0:
            shell_mask[item.delete_shell_indices] = True
            delete_mask[item.delete_shell_indices] = True
        asset_chunks.append(
            {
                "means": item.asset_means_world,
                "colors": item.asset_colors,
                "opacities": item.asset_opacities,
                "scales": item.asset_scales,
                "quats": item.asset_quats,
            }
        )

    deleted = _subset_gaussians(clean, ~delete_mask)
    asset_only = _concat_gaussians(asset_chunks)
    edited = _concat_gaussians([deleted, asset_only])

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
