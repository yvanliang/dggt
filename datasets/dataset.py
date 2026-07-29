import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
import os
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import random
import open3d as o3d
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import json

from dggt.utils.gaussian_time import gaussian_timestamps_from_frame_ids
from dggt.utils.factorized_asset_condition import (
    FACTORIZED_ASSET_CONDITION_VERSION,
    alpha_to_patch_mask,
    bbox_patch_mask,
    canonicalize_asset_reference,
    project_anchor_boxes_to_patch_bboxes,
    resize_crop_intrinsics_to_model_canvas,
)


WAYMO_OPENCV2DATASET = np.array(
    [
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def _normalize_waymo_caption_base(name):
    base = os.path.basename(str(name).lstrip("\ufeff").rstrip("/"))
    if base.endswith(".tfrecord"):
        base = base[: -len(".tfrecord")]
    if base.startswith("segment-"):
        base = base[len("segment-"):]
    suffix = "_with_camera_labels"
    if base.endswith(suffix):
        base = base[:-len(suffix)]
    return base


def _load_waymo_matrix4(path, name):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing Waymo {name}: {path}")
    values = np.loadtxt(path, dtype=np.float32)
    if values.size != 16:
        raise ValueError(f"Waymo {name} must contain 16 values, got shape {values.shape}: {path}")
    return values.reshape(4, 4).astype(np.float32)


def _load_waymo_intrinsics_matrix(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing Waymo intrinsics: {path}")
    values = np.loadtxt(path, dtype=np.float32)
    if values.shape == (3, 3):
        return values.astype(np.float32)
    flat = values.reshape(-1)
    if flat.size < 4:
        raise ValueError(f"Waymo intrinsics must contain at least fx, fy, cx, cy: {path}")
    fx, fy, cx, cy = [float(v) for v in flat[:4]]
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=target_size, mode="nearest")
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    return flow.squeeze()

    
def load_and_preprocess_flow(flow_path_list, extrinsic_paths, intrinsic_path, height, width):
    if len(flow_path_list) == 0:
        raise ValueError("At least 1 image is required")

    flows = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # print(f"[DEBUG] flow_path_list: {flow_path_list}")
    for i, flow_path in enumerate(flow_path_list):
        # print(f"[DEBUG] Processing flow_path[{i}]: {flow_path}")
        # print(f"[DEBUG] Is file: {os.path.isfile(flow_path) if flow_path else 'Empty/None'}")
        # print(f"[DEBUG] Is dir: {os.path.isdir(flow_path) if flow_path else 'Empty/None'}")
        depth_and_flow = np.load(flow_path)
        flow = depth_and_flow
        flow = torch.tensor(flow).float()
        flow = resize_flow(flow, (height, width))
        flows.append(flow)
    
    return torch.stack(flows)


def load_and_preprocess_images(
    image_path_list,
    mode="crop",
    *,
    output_dtype="float32",
):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    images = []
    shapes = set()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        with Image.open(image_path) as source:
            # If there's an alpha channel, blend onto white background.
            if source.mode == "RGBA":
                background = Image.new(
                    "RGBA",
                    source.size,
                    (255, 255, 255, 255),
                )
                img = Image.alpha_composite(background, source)
            else:
                img = source
            img = img.convert("RGB")
            width, height = img.size

            # Original behavior: set width to 518px.
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            img = img.resize(
                (new_width, new_height),
                Image.Resampling.BICUBIC,
            )
            # NumPy performs this bulk conversion substantially faster than
            # torchvision's per-image ToTensor path in DataLoader workers.
            if str(output_dtype) == "uint8":
                image_array = np.ascontiguousarray(
                    np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
                )
            elif str(output_dtype) == "float32":
                image_array = np.ascontiguousarray(
                    np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
                )
            else:
                raise ValueError(
                    "output_dtype must be 'float32' or 'uint8', got "
                    f"{output_dtype!r}"
                )
            img = torch.from_numpy(image_array)
            if str(output_dtype) == "float32":
                # Preserve torchvision ToTensor's exact arithmetic order:
                # uint8 -> float32 followed by division, rather than a NumPy
                # float multiply that differs by one ULP for many pixel values.
                img = img.to(dtype=torch.float32).div(255.0)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=255 if img.dtype == torch.uint8 else 1.0,
                )
            padded_images.append(img)
        images = padded_images
    images = torch.stack(images)  # concatenate images
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_and_preprocess_binary_masks(
    mask_path_list,
    mode="crop",
    threshold=0.5,
    *,
    channels=3,
):
    # Check for empty list
    if len(mask_path_list) == 0:
        raise ValueError("At least 1 mask is required")

    masks = []
    shapes = set()
    target_size = 518

    for mask_path in mask_path_list:
        with Image.open(mask_path) as source:
            mask = source.convert("L")
            width, height = mask.size
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            mask = mask.resize(
                (new_width, new_height),
                Image.Resampling.NEAREST,
            )
            mask_array = np.asarray(mask, dtype=np.uint8).copy()
        mask = torch.from_numpy(mask_array).unsqueeze(0)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            mask = mask[:, start_y : start_y + target_size, :]

        mask = mask.gt(float(threshold) * 255.0).to(torch.float32)
        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if channels > 1:
            mask = mask.expand(channels, -1, -1).contiguous()

        shapes.add((mask.shape[1], mask.shape[2]))
        masks.append(mask)

    if len(shapes) > 1:
        print(f"Warning: Found masks with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_masks = []
        for mask in masks:
            h_padding = max_height - mask.shape[1]
            w_padding = max_width - mask.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                mask = torch.nn.functional.pad(
                    mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )
            padded_masks.append(mask)
        masks = padded_masks

    masks = torch.stack(masks)
    if len(mask_path_list) == 1 and masks.dim() == 3:
        masks = masks.unsqueeze(0)

    return masks


def _waymo_resize_geometry(image_hw, target_width=518):
    image_h, image_w = [int(v) for v in image_hw]
    new_width = int(target_width)
    new_height = round(image_h * (new_width / image_w) / 14) * 14
    crop_top = max((new_height - new_width) // 2, 0) if new_height > new_width else 0
    out_height = new_width if new_height > new_width else new_height
    return {
        "scale_x": new_width / float(image_w),
        "scale_y": new_height / float(image_h),
        "crop_top": crop_top,
        "out_hw": (int(out_height), int(new_width)),
    }


def _transform_waymo_box_to_model(box_xyxy, image_hw, target_width=518):
    geom = _waymo_resize_geometry(image_hw, target_width=target_width)
    box = np.asarray(box_xyxy, dtype=np.float32).copy()
    box[[0, 2]] *= float(geom["scale_x"])
    box[[1, 3]] = box[[1, 3]] * float(geom["scale_y"]) - float(geom["crop_top"])
    out_h, out_w = geom["out_hw"]
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(out_w))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(out_h))
    return box.astype(np.float32), geom


def _build_waymo_box_corners_world(obj_to_world, box_size):
    length, width, height = np.asarray(box_size, dtype=np.float32).reshape(3).tolist()
    local = np.array(
        [
            [-length / 2, -width / 2, -height / 2, 1.0],
            [-length / 2, -width / 2, height / 2, 1.0],
            [-length / 2, width / 2, -height / 2, 1.0],
            [-length / 2, width / 2, height / 2, 1.0],
            [length / 2, -width / 2, -height / 2, 1.0],
            [length / 2, -width / 2, height / 2, 1.0],
            [length / 2, width / 2, -height / 2, 1.0],
            [length / 2, width / 2, height / 2, 1.0],
        ],
        dtype=np.float32,
    )
    return (np.asarray(obj_to_world, dtype=np.float32) @ local.T).T[:, :3].astype(np.float32)


def _project_world_points_to_raw(
    points_world,
    camera_to_world,
    intrinsics,
    image_hw,
    eps=1e-6,
    *,
    world_to_camera=None,
):
    points_world = np.asarray(points_world, dtype=np.float32)
    if world_to_camera is None:
        world_to_camera = np.linalg.inv(
            np.asarray(camera_to_world, dtype=np.float32)
        )
    else:
        world_to_camera = np.asarray(world_to_camera, dtype=np.float32)
    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    points_cam = (world_to_camera @ points_h.T).T[:, :3]
    depths = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (depths > float(eps))
    if not np.any(valid):
        return None, points_cam
    projected_h = (np.asarray(intrinsics, dtype=np.float32) @ points_cam[valid].T).T
    if not np.isfinite(projected_h).all():
        return None, points_cam
    projected = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], float(eps))
    image_h, image_w = [int(v) for v in image_hw]
    x1 = float(np.clip(projected[:, 0].min(), 0.0, float(image_w)))
    x2 = float(np.clip(projected[:, 0].max(), 0.0, float(image_w)))
    y1 = float(np.clip(projected[:, 1].min(), 0.0, float(image_h)))
    y2 = float(np.clip(projected[:, 1].max(), 0.0, float(image_h)))
    if x2 <= x1 or y2 <= y1:
        return None, points_cam
    return np.array([x1, y1, x2, y2], dtype=np.float32), points_cam


def _model_box_to_patch_mask(box_xyxy, model_hw, patch_grid):
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    box = np.asarray(box_xyxy, dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    if x2 <= x1 or y2 <= y1 or model_h <= 0 or model_w <= 0 or gh <= 0 or gw <= 0:
        return np.zeros((gh * gw,), dtype=np.bool_), 0.0
    patch_w = float(model_w) / float(gw)
    patch_h = float(model_h) / float(gh)
    px1 = max(0, min(gw, int(np.floor(x1 / patch_w))))
    px2 = max(0, min(gw, int(np.ceil(x2 / patch_w))))
    py1 = max(0, min(gh, int(np.floor(y1 / patch_h))))
    py2 = max(0, min(gh, int(np.ceil(y2 / patch_h))))
    mask = np.zeros((gh, gw), dtype=np.bool_)
    if px2 > px1 and py2 > py1:
        mask[py1:py2, px1:px2] = True
    return mask.reshape(gh * gw), float(mask.sum())


def _load_waymo_dynamic_mask_model(mask_path, target_width=518, threshold=0.5):
    if not mask_path or not os.path.exists(mask_path):
        return None
    mask = Image.open(mask_path).convert("L")
    width, height = mask.size
    new_width = int(target_width)
    new_height = round(height * (new_width / width) / 14) * 14
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
    if new_height > target_width:
        start_y = (new_height - target_width) // 2
        mask = mask.crop((0, start_y, new_width, start_y + target_width))
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return arr > float(threshold)


def _waymo_semantic_values_for_class(class_name):
    name = str(class_name).lower()
    if any(value in name for value in ("vehicle", "car", "truck", "bus")):
        return (40,)
    if any(value in name for value in ("pedestrian", "person")):
        return (50,)
    if any(value in name for value in ("cyclist", "bicycle", "motorcycle", "rider")):
        return (60,)
    return ()


def _load_waymo_semantic_labels_model(mask_path, target_width=518):
    """Decode one semantic label image on the DGGT model canvas."""
    if not mask_path or not os.path.exists(mask_path):
        return None
    with Image.open(mask_path) as source:
        mask = source.convert("L")
        width, height = mask.size
        new_width = int(target_width)
        new_height = round(height * (new_width / width) / 14) * 14
        mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        if new_height > target_width:
            start_y = (new_height - target_width) // 2
            mask = mask.crop((0, start_y, new_width, start_y + target_width))
        return np.asarray(mask, dtype=np.uint8).copy()


def _load_waymo_semantic_foreground_model(mask_path, class_name, target_width=518):
    """Load class-matched semantic foreground on the DGGT model canvas."""
    semantic_values = _waymo_semantic_values_for_class(class_name)
    if not semantic_values:
        return None
    labels = _load_waymo_semantic_labels_model(
        mask_path,
        target_width=target_width,
    )
    if labels is None:
        return None
    return np.isin(labels, semantic_values)


def _boxed_binary_mask_patch_support(
    foreground,
    box_xyxy,
    patch_grid,
    *,
    coverage_threshold=0.05,
    integral=None,
):
    """Pool ``foreground ∩ box`` with adaptive-average-pool bin semantics.

    An integral image makes this O(patches) without allocating a full-resolution
    alpha tensor or crossing the NumPy/Torch boundary for every candidate.
    """
    foreground = np.asarray(foreground, dtype=np.bool_)
    if foreground.ndim != 2:
        raise ValueError(
            f"foreground must be a 2D binary mask, got {foreground.shape}"
        )
    height, width = foreground.shape
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    empty = np.zeros((gh * gw,), dtype=np.bool_)
    if height <= 0 or width <= 0 or gh <= 0 or gw <= 0:
        return empty, 0

    x0, y0, x1, y1 = [
        int(round(float(value)))
        for value in np.asarray(box_xyxy, dtype=np.float32).reshape(4)
    ]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return empty, 0

    if integral is None:
        integral = np.pad(
            foreground.astype(np.int32, copy=False).cumsum(0).cumsum(1),
            ((1, 0), (1, 0)),
        )
    else:
        integral = np.asarray(integral)
        if integral.shape != (height + 1, width + 1):
            raise ValueError(
                f"integral shape {integral.shape} != {(height + 1, width + 1)}"
            )
    pool_y0 = np.floor(np.arange(gh) * height / gh).astype(np.int64)
    pool_y1 = np.ceil((np.arange(gh) + 1) * height / gh).astype(np.int64)
    pool_x0 = np.floor(np.arange(gw) * width / gw).astype(np.int64)
    pool_x1 = np.ceil((np.arange(gw) + 1) * width / gw).astype(np.int64)
    iy0 = np.maximum(pool_y0, y0)
    iy1 = np.minimum(pool_y1, y1)
    ix0 = np.maximum(pool_x0, x0)
    ix1 = np.minimum(pool_x1, x1)
    valid_y = iy1 > iy0
    valid_x = ix1 > ix0
    counts = (
        integral[iy1[:, None], ix1[None, :]]
        - integral[iy0[:, None], ix1[None, :]]
        - integral[iy1[:, None], ix0[None, :]]
        + integral[iy0[:, None], ix0[None, :]]
    )
    counts *= valid_y[:, None] & valid_x[None, :]
    pool_area = (
        (pool_y1 - pool_y0)[:, None]
        * (pool_x1 - pool_x0)[None, :]
    )
    support = counts >= float(coverage_threshold) * pool_area
    foreground_area = int(
        integral[y1, x1] - integral[y0, x1]
        - integral[y1, x0] + integral[y0, x0]
    )
    return support.reshape(gh * gw), foreground_area


def _select_pretrain_reference_candidate(reference_candidates, deterministic=False):
    if not reference_candidates:
        raise ValueError("reference_candidates must be non-empty")
    if not deterministic:
        return random.choice(reference_candidates)
    # All candidates already pass the same geometry, occlusion and minimum
    # patch-support gates. Prefer the cleanest/largest visible foreground, then
    # use frame id as a stable tie-breaker. This order is shared by every
    # sliding window, so a legal candidate stays canonical across windows.
    return max(
        reference_candidates,
        key=lambda item: (int(item[2]), float(item[3]), -int(item[0])),
    )


def _largest_connected_component_4n(mask_grid):
    gh, gw = int(mask_grid.shape[0]), int(mask_grid.shape[1])
    visited = np.zeros((gh, gw), dtype=np.bool_)
    best = []
    for y in range(gh):
        for x in range(gw):
            if not bool(mask_grid[y, x]) or bool(visited[y, x]):
                continue
            stack = [(y, x)]
            visited[y, x] = True
            comp = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= gh or nx < 0 or nx >= gw:
                        continue
                    if bool(mask_grid[ny, nx]) and not bool(visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    out = np.zeros((gh, gw), dtype=np.bool_)
    for y, x in best:
        out[y, x] = True
    return out


def _dynamic_mask_box_to_patch_mask(dynamic_model_mask, box_xyxy, model_hw, patch_grid, padding_px=4):
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    empty = np.zeros((gh * gw,), dtype=np.bool_)
    if dynamic_model_mask is None:
        return empty, 0.0
    if model_h <= 0 or model_w <= 0 or gh <= 0 or gw <= 0:
        return empty, 0.0
    dyn = np.asarray(dynamic_model_mask, dtype=np.bool_)
    if dyn.shape[0] != model_h or dyn.shape[1] != model_w:
        return empty, 0.0

    x1, y1, x2, y2 = [float(v) for v in np.asarray(box_xyxy, dtype=np.float32).tolist()]
    if x2 <= x1 or y2 <= y1:
        return empty, 0.0
    pad = int(max(0, padding_px))
    ix1 = max(0, min(model_w, int(np.floor(x1)) - pad))
    ix2 = max(0, min(model_w, int(np.ceil(x2)) + pad))
    iy1 = max(0, min(model_h, int(np.floor(y1)) - pad))
    iy2 = max(0, min(model_h, int(np.ceil(y2)) + pad))
    if ix2 <= ix1 or iy2 <= iy1:
        return empty, 0.0

    roi = dyn[iy1:iy2, ix1:ix2]
    if not np.any(roi):
        return empty, 0.0

    ys, xs = np.nonzero(roi)
    xs = xs.astype(np.float32) + float(ix1)
    ys = ys.astype(np.float32) + float(iy1)
    patch_w = float(model_w) / float(gw)
    patch_h = float(model_h) / float(gh)
    patch_x = np.clip(np.floor(xs / patch_w).astype(np.int64), 0, gw - 1)
    patch_y = np.clip(np.floor(ys / patch_h).astype(np.int64), 0, gh - 1)
    grid = np.zeros((gh, gw), dtype=np.bool_)
    grid[patch_y, patch_x] = True
    grid = _largest_connected_component_4n(grid)
    return grid.reshape(gh * gw), float(grid.sum())


def _dynamic_mask_to_patch_grid(dynamic_model_mask, model_hw, patch_grid):
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    empty = np.zeros((gh * gw,), dtype=np.bool_)
    if dynamic_model_mask is None or model_h <= 0 or model_w <= 0 or gh <= 0 or gw <= 0:
        return empty
    dyn = np.asarray(dynamic_model_mask, dtype=np.bool_)
    if dyn.shape[0] != model_h or dyn.shape[1] != model_w:
        return empty
    out = np.zeros((gh, gw), dtype=np.bool_)
    for y in range(gh):
        y1 = int(round(y * model_h / gh))
        y2 = int(round((y + 1) * model_h / gh))
        for x in range(gw):
            x1 = int(round(x * model_w / gw))
            x2 = int(round((x + 1) * model_w / gw))
            out[y, x] = bool(np.any(dyn[y1:y2, x1:x2]))
    return out.reshape(gh * gw)


def _patch_center_xy(model_hw, patch_grid):
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    yy = (np.arange(gh, dtype=np.float32) + 0.5) * (float(model_h) / float(gh))
    xx = (np.arange(gw, dtype=np.float32) + 0.5) * (float(model_w) / float(gw))
    grid_x, grid_y = np.meshgrid(xx, yy)
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)


def _assign_dynamic_patch_masks_to_boxes(dynamic_model_mask, boxes_xyxy, model_hw, patch_grid, padding_px=6):
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] != 4:
        return np.zeros((0, int(patch_grid[0]) * int(patch_grid[1])), dtype=np.bool_)
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    dynamic_patch = _dynamic_mask_to_patch_grid(dynamic_model_mask, model_hw, (gh, gw))
    assigned = np.zeros((boxes.shape[0], gh * gw), dtype=np.bool_)
    if not np.any(dynamic_patch):
        return assigned

    centers = _patch_center_xy(model_hw, (gh, gw))
    scores = np.full((boxes.shape[0], gh * gw), np.inf, dtype=np.float32)
    pad = float(max(0, padding_px))
    for obj_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        x1p = max(0.0, x1 - pad)
        y1p = max(0.0, y1 - pad)
        x2p = min(float(model_w), x2 + pad)
        y2p = min(float(model_h), y2 + pad)
        inside = (
            (centers[:, 0] >= x1p)
            & (centers[:, 0] <= x2p)
            & (centers[:, 1] >= y1p)
            & (centers[:, 1] <= y2p)
            & dynamic_patch
        )
        if not np.any(inside):
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = max(0.5 * (x2 - x1), 1.0)
        half_h = max(0.5 * (y2 - y1), 1.0)
        dx = (centers[:, 0] - cx) / half_w
        dy = (centers[:, 1] - cy) / half_h
        # Lower is better. The normalized distance makes overlapping boxes
        # compete for each dynamic patch instead of duplicating connected masks.
        scores[obj_idx, inside] = dx[inside] * dx[inside] + dy[inside] * dy[inside]

    eligible = np.isfinite(scores).any(axis=0)
    if not np.any(eligible):
        return assigned
    owner = np.argmin(scores[:, eligible], axis=0)
    patch_ids = np.nonzero(eligible)[0]
    assigned[owner, patch_ids] = True
    return assigned


class WaymoOpenDataset(Dataset):
    def __init__(
        self,
        image_dir,
        scene_names=None,
        sequence_length=None,
        start_idx=-1,
        mode=1,
        views=1,
        intervals=2,
        caption_root=None,
        scene_name_to_index_path="/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/metadata/training/scene_name_to_index.json",
        waymo_train_list_path=None,
        waymo_val_list_path=None,
        pretrain_patch_grid=(25, 37),
        pretrain_max_objects=5,
        pretrain_instance_cache_size=8,
        trunk_major_samples=False,
        trunk_frames=29,
        camera_anchor_window_probability=0.0,
        return_full_dggt_context=False,
        trunk_major_window_offsets=None,
        load_dynamic_masks=True,
        binary_mask_channels=3,
        image_output_dtype="float32",
    ):
        #mode 1 : train
        #mode 2 : pure reconstruction
        #mode 3 : interplation
        
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        if mode == 1:
            interval = 1
        elif mode == 2:
            interval = 1
        elif mode == 3:
            interval = intervals
        else:
            interval = 1
        self.interval =  interval
        self.mode = mode
        if mode == 1:
            test_mode = False
            load_flow = False
        elif mode == 2:
            test_mode = True
            load_flow = False
        elif mode == 3:
            test_mode = True
            load_flow = True
        else:
            pass
        self.test_mode = test_mode
        self.load_flow = load_flow
        self.views = views
        self.caption_root = caption_root
        self.pretrain_patch_grid = (int(pretrain_patch_grid[0]), int(pretrain_patch_grid[1]))
        self.pretrain_max_objects = int(pretrain_max_objects)
        self.pretrain_instance_cache_size = int(pretrain_instance_cache_size)
        if self.pretrain_instance_cache_size < 0:
            raise ValueError(
                "pretrain_instance_cache_size must be non-negative, got "
                f"{pretrain_instance_cache_size}"
            )
        self.trunk_major_samples = bool(trunk_major_samples)
        self.trunk_frames = int(trunk_frames)
        if self.trunk_frames <= 0:
            raise ValueError(f"trunk_frames must be positive, got {trunk_frames}")
        self.camera_anchor_window_probability = float(camera_anchor_window_probability)
        if not 0.0 <= self.camera_anchor_window_probability <= 1.0:
            raise ValueError(
                "camera_anchor_window_probability must be in [0, 1], got "
                f"{camera_anchor_window_probability}"
            )
        self.return_full_dggt_context = bool(return_full_dggt_context)
        self.load_dynamic_masks = bool(load_dynamic_masks)
        self.binary_mask_channels = int(binary_mask_channels)
        if self.binary_mask_channels <= 0:
            raise ValueError(
                "binary_mask_channels must be positive, got "
                f"{binary_mask_channels}"
            )
        self.image_output_dtype = str(image_output_dtype)
        if self.image_output_dtype not in ("float32", "uint8"):
            raise ValueError(
                "image_output_dtype must be 'float32' or 'uint8', got "
                f"{image_output_dtype!r}"
            )
        if self.return_full_dggt_context and self.views != 1:
            raise ValueError("Full DGGT context is currently supported only for views=1.")
        if trunk_major_window_offsets is None:
            self.trunk_major_window_offsets = None
        else:
            offsets = tuple(int(offset) for offset in trunk_major_window_offsets)
            if not offsets or any(offset < 0 for offset in offsets):
                raise ValueError(
                    "trunk_major_window_offsets must contain non-negative offsets, got "
                    f"{trunk_major_window_offsets}"
                )
            self.trunk_major_window_offsets = tuple(dict.fromkeys(offsets))
        self.scene_name_to_index_path = scene_name_to_index_path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.waymo_train_list_path = waymo_train_list_path or os.path.join(repo_root, "data", "waymo_train_list_full.txt")
        self.waymo_val_list_path = waymo_val_list_path or os.path.join(repo_root, "data", "waymo_val_list_full.txt")
        self.scene_name_to_base = self._build_scene_name_to_base(
            scene_name_to_index_path=scene_name_to_index_path,
            image_dir=image_dir,
        )

        # Scan all scene folders and collect image paths
        if scene_names is None:
            scene_names = [] 
            scene_names_ = [f"{i:03d}" for i in range(0, 99)]
            scene_names = scene_names + scene_names_
        self.scenes = scene_names
        self.image_paths = []
        self.sky_mask_paths = []
        self.dynamic_mask_path = []
        self.extrinsic_paths = []
        self.intrinsic_paths = []
        self.semantic_mask_path = []
        self.depth_flow_paths = []
        self.ego_paths = []
        self.scene_roots = []
        self._instance_metadata_cache = {}
        self._camera_metadata_cache = {}

        self.start_idx = start_idx

        for scene_name in scene_names:
            scene_path = os.path.join(image_dir, scene_name, "images")
            if os.path.isdir(scene_path):
                self.scene_roots.append(os.path.join(image_dir, scene_name))
                # image
                if self.views == 1:
                    image_paths = sorted(
                        [
                            os.path.join(scene_path, f)
                            for f in os.listdir(scene_path)
                            if f.endswith(("_0.jpg", "_0.png"))
                        ]
                    )
                    self.image_paths.append(image_paths)
                elif self.views == 3:
                    views_image_lists = []
                    for v in range(3):
                        suffixes = (f"_{v}.jpg", f"_{v}.png")
                        files_v = sorted(
                            [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith(suffixes)]
                        )
                        views_image_lists.append(files_v)
                    lengths = [len(l) for l in views_image_lists]
                    if len(set(lengths)) != 1:
                        raise RuntimeError(f"Inconsistent number of images across views in scene {scene_name}, lengths: {lengths}")
                    self.image_paths.append(views_image_lists)

                # sky_mask
                sky_mask_path = os.path.join(image_dir, scene_name, "sky_masks")
                if os.path.isdir(sky_mask_path):
                    if self.views == 1:
                        sky_mask_paths = sorted(
                            [os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.sky_mask_paths.append(sky_mask_paths)
                    elif self.views == 3:
                        views_sky_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(sky_mask_path, f) for f in os.listdir(sky_mask_path) if f.endswith(suffixes)])
                            views_sky_lists.append(files_v)
                        self.sky_mask_paths.append(views_sky_lists)
                else:
                    self.sky_mask_paths.append([] if self.views == 1 else [[] for _ in range(3)])

                # extrinsic
                extrinsic_path = os.path.join(image_dir, scene_name, "ego_pose")
                if os.path.isdir(extrinsic_path):
                    extrinsic_paths = sorted([
                        os.path.join(extrinsic_path, f)
                        for f in os.listdir(extrinsic_path)
                        if f.endswith(".txt")
                    ])
                    self.extrinsic_paths.append(extrinsic_paths)
                else:
                    self.extrinsic_paths.append([])

                # extrinsic
                ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                # ego_path = os.path.join(image_dir, scene_name, "extrinsics")
                if os.path.isdir(ego_path):
                    ego_path =  os.path.join(ego_path, "0.txt")
                    self.ego_paths.append(ego_path)
                else:
                    self.ego_paths.append("")

                # intrinsic
                intrinsic_path = os.path.join(image_dir, scene_name, "intrinsics")
                if os.path.isdir(intrinsic_path):
                    if self.views == 1:
                        intrinsic_paths = os.path.join(intrinsic_path, "0.txt")
                        self.intrinsic_paths.append(intrinsic_paths)
                    elif self.views == 3:
                        intrinsics_views = []
                        for v in range(3):
                            p = os.path.join(intrinsic_path, f"{v}.txt")
                            intrinsics_views.append(p if os.path.exists(p) else "")
                        self.intrinsic_paths.append(intrinsics_views)
                else:
                    self.intrinsic_paths.append("" if self.views == 1 else ["" for _ in range(3)])

                # dynamic mask
                dynamic_mask_path = os.path.join(image_dir, scene_name, "fine_dynamic_masks/all")
                if os.path.isdir(dynamic_mask_path):
                    if self.views == 1:
                        dynamic_mask_paths = sorted(
                            [os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.dynamic_mask_path.append(dynamic_mask_paths)
                    elif self.views == 3:
                        views_dyn_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(dynamic_mask_path, f) for f in os.listdir(dynamic_mask_path) if f.endswith(suffixes)])
                            views_dyn_lists.append(files_v)
                        self.dynamic_mask_path.append(views_dyn_lists)
                else:
                    self.dynamic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])
                # depth
                depth_path = os.path.join(image_dir, scene_name, "depth_flows_4")
                if os.path.isdir(depth_path):
                    if self.views == 1:
                        depth_paths = sorted(
                            [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith("_0.npy")]
                        )
                        self.depth_flow_paths.append(depth_paths)
                    elif self.views == 3:
                        views_depth_lists = []
                        for v in range(3):
                            suffix = f"_{v}.npy"
                            files_v = sorted(
                                [os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith(suffix)]
                            )
                            views_depth_lists.append(files_v)
                        self.depth_flow_paths.append(views_depth_lists)
                else:
                    self.depth_flow_paths.append([] if self.views == 1 else [[] for _ in range(3)])
                # semantic mask
                semantic_mask_path = os.path.join(image_dir, scene_name, "custom_masks")
                if os.path.isdir(semantic_mask_path):
                    if self.views == 1:
                        semantic_mask_paths = sorted(
                            [os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(("_0.jpg", "_0.png"))]
                        )
                        self.semantic_mask_path.append(semantic_mask_paths)
                    elif self.views == 3:
                        views_sem_lists = []
                        for v in range(3):
                            suffixes = (f"_{v}.jpg", f"_{v}.png")
                            files_v = sorted([os.path.join(semantic_mask_path, f) for f in os.listdir(semantic_mask_path) if f.endswith(suffixes)])
                            views_sem_lists.append(files_v)
                        self.semantic_mask_path.append(views_sem_lists)
                else:
                    self.semantic_mask_path.append([] if self.views == 1 else [[] for _ in range(3)])

        # Validation/inference order: every scene's trunk 0, then every scene's
        # trunk 1, etc. Only trunks that can provide a full sequence are used.
        self.trunk_major_index = []
        if self.trunk_major_samples:
            max_trunks = max(
                (
                    len(paths[0] if self.views == 3 else paths) // self.trunk_frames
                    for paths in self.image_paths
                ),
                default=0,
            )
            for trunk_idx in range(max_trunks):
                trunk_base = trunk_idx * self.trunk_frames
                for scene_idx, paths in enumerate(self.image_paths):
                    total_frames = len(paths[0] if self.views == 3 else paths)
                    available = min(trunk_base + self.trunk_frames, total_frames) - trunk_base
                    if available >= int(self.sequence_length):
                        if self.trunk_major_window_offsets is None:
                            self.trunk_major_index.append((scene_idx, trunk_idx))
                        else:
                            max_offset = available - int(self.sequence_length)
                            for offset in self.trunk_major_window_offsets:
                                if offset <= max_offset:
                                    self.trunk_major_index.append((scene_idx, trunk_idx, offset))


    def _build_scene_name_to_base(self, scene_name_to_index_path, image_dir):
        mapping = self._load_scene_name_to_base(scene_name_to_index_path)
        list_paths = self._get_preferred_waymo_list_paths(image_dir)
        list_mapping = self._load_scene_name_to_base_from_lists(list_paths)
        # Prefer explicit train/val list indexing for numeric scene ids (000, 001, ...).
        mapping.update(list_mapping)
        return mapping

    def _get_preferred_waymo_list_paths(self, image_dir):
        image_dir_l = str(image_dir).lower()
        has_train = os.path.exists(self.waymo_train_list_path)
        has_val = os.path.exists(self.waymo_val_list_path)
        if "validation" in image_dir_l or "/val" in image_dir_l:
            return [p for p in (self.waymo_val_list_path, self.waymo_train_list_path) if os.path.exists(p)]
        if "training" in image_dir_l or "/train" in image_dir_l:
            return [p for p in (self.waymo_train_list_path, self.waymo_val_list_path) if os.path.exists(p)]
        if has_train and has_val:
            return [self.waymo_train_list_path, self.waymo_val_list_path]
        if has_train:
            return [self.waymo_train_list_path]
        if has_val:
            return [self.waymo_val_list_path]
        return []

    def _load_scene_name_to_base_from_lists(self, paths):
        mapping = {}
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    lines = [line.strip().lstrip("\ufeff") for line in f if line.strip()]
            except Exception as exc:
                if self.caption_root:
                    raise RuntimeError(f"Failed to read Waymo list file: {path}") from exc
                continue

            # The processed Waymo folders are indexed by lexicographically
            # sorted segment name.  The training list already follows that
            # order, but the validation list contains the same segments in a
            # shuffled order.  Enumerating the raw validation file therefore
            # attached every numeric scene folder to the wrong caption.
            bases = sorted(_normalize_waymo_caption_base(item) for item in lines)
            if len(bases) != len(set(bases)):
                raise ValueError(f"Waymo list contains duplicate segment names: {path}")
            for idx, base in enumerate(bases):
                key = f"{idx:03d}"
                if key not in mapping:
                    mapping[key] = base
        return mapping


    def _load_scene_name_to_base(self, path):
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            if self.caption_root:
                raise RuntimeError(f"Failed to read scene_name_to_index mapping: {path}") from exc
            return {}
        if not isinstance(data, dict):
            if self.caption_root:
                raise ValueError(f"scene_name_to_index mapping must be a JSON object: {path}")
            return {}
        mapping = {}
        for key, value in data.items():
            key_s = str(key)
            value_s = str(value)
            if value_s.isdigit():
                mapping[value_s.zfill(3)] = _normalize_waymo_caption_base(key_s)
            if key_s.isdigit():
                mapping[key_s.zfill(3)] = _normalize_waymo_caption_base(value_s)
        return mapping

    def _caption_path(self, scene_name, start_idx):
        if not self.caption_root:
            return None
        scene_key = str(scene_name).zfill(3) if str(scene_name).isdigit() else str(scene_name)
        scene_base = self.scene_name_to_base.get(scene_key, str(scene_name))
        scene_base = _normalize_waymo_caption_base(scene_base)
        clip_index = int(start_idx) // 29
        return os.path.join(str(self.caption_root), "pinhole_front", f"{scene_base}_{clip_index}.json")

    def _load_caption(self, scene_name, start_idx):
        path = self._caption_path(scene_name, start_idx)
        if path is None:
            return None
        if not os.path.exists(path):
            clip_index = int(start_idx) // 29
            raise FileNotFoundError(
                f"Caption file not found for scene={scene_name!r}, clip_index={clip_index}: {path}"
            )
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to read caption JSON: {path}") from exc
        if isinstance(payload, dict):
            caption = payload.get("caption")
            if caption is not None:
                return str(caption)
            raise KeyError(f"Caption JSON lacks 'caption': {path}")
        raise ValueError(f"Caption JSON must be an object with key 'caption': {path}")


    def __len__(self):
        if self.trunk_major_samples:
            return len(self.trunk_major_index)
        return len(self.scenes)

    def _sample_start_in_trunk(self, total_frames, trunk_idx):
        """Sample a contiguous sequence inside one explicitly selected trunk."""
        base = int(trunk_idx) * self.trunk_frames
        end = min(base + self.trunk_frames, int(total_frames))
        if end - base < int(self.sequence_length):
            raise IndexError(
                f"trunk {trunk_idx} has only {end - base} frames, "
                f"fewer than {self.sequence_length}"
            )
        return self._sample_balanced_camera_start(base, end)

    def _fixed_start_in_trunk(self, total_frames, trunk_idx, window_offset=None):
        """Select the deterministic offset requested by ``start_idx`` in a trunk."""
        base = int(trunk_idx) * self.trunk_frames
        end = min(base + self.trunk_frames, int(total_frames))
        if end - base < int(self.sequence_length):
            raise IndexError(
                f"trunk {trunk_idx} has only {end - base} frames, "
                f"fewer than {self.sequence_length}"
            )
        requested_offset = int(self.start_idx) if window_offset is None else int(window_offset)
        offset = min(max(0, requested_offset), end - base - int(self.sequence_length))
        return base + offset

    def _sample_balanced_camera_start(self, base, end):
        """Balance clip-anchor and delta-only windows without changing token roles.

        Natural uniform sampling sees the clip anchor with probability
        ``1 / (trunk_frames - sequence_length + 1)``.  Instead, first choose
        whether this sample contains the one global camera anchor, then sample
        uniformly among the non-anchor starts.  This avoids both anchor
        starvation and accidentally promoting a local window start to anchor.
        """
        base = int(base)
        end = int(end)
        last_start = end - int(self.sequence_length)
        if last_start <= base:
            return base
        if random.random() < self.camera_anchor_window_probability:
            return base
        return random.randint(base + 1, last_start)

    def _sample_start_within_caption_trunk(self, total_frames, trunk_frames=29):
        """Sample a contiguous raw-pretrain window without crossing caption trunks."""
        sequence_length = int(self.sequence_length)
        total_frames = int(total_frames)
        trunk_frames = int(trunk_frames)
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        if total_frames <= sequence_length:
            return 0
        valid_trunks = []
        for base in range(0, total_frames - trunk_frames + 1, trunk_frames):
            end = base + trunk_frames
            if end - base >= sequence_length:
                valid_trunks.append((base, end))
        if not valid_trunks:
            return random.randint(0, max(0, total_frames - sequence_length))
        base, end = random.choice(valid_trunks)
        return self._sample_balanced_camera_start(base, end)

    def _fixed_start_within_caption_trunk(self, total_frames, trunk_frames=29):
        """Return a deterministic start index without crossing caption trunks."""
        sequence_length = int(self.sequence_length)
        total_frames = int(total_frames)
        trunk_frames = int(trunk_frames)
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        max_start = max(0, total_frames - sequence_length)
        start = min(max(0, int(self.start_idx)), max_start)
        if total_frames <= sequence_length:
            return 0

        trunk_base = (start // trunk_frames) * trunk_frames
        trunk_end = min(trunk_base + trunk_frames, total_frames)
        if trunk_end - trunk_base < sequence_length:
            return min(trunk_base, max_start)
        if start + sequence_length > trunk_end:
            start = trunk_end - sequence_length
        return min(max(trunk_base, start), max_start)

    def _load_waymo_camera_metadata(self, idx, frame_indices=()):
        """Load calibration once and lazily cache requested ego poses per worker."""
        scene_name = str(self.scenes[idx])
        cache = self._camera_metadata_cache
        if scene_name in cache:
            payload = cache.pop(scene_name)
            cache[scene_name] = payload
        else:
            camera_extrinsic_path = (
                self.ego_paths[idx] if idx < len(self.ego_paths) else ""
            )
            intrinsic_path = self.intrinsic_paths[idx]
            if isinstance(intrinsic_path, (list, tuple)):
                intrinsic_path = intrinsic_path[0] if intrinsic_path else ""
            camera_to_ego = _load_waymo_matrix4(
                camera_extrinsic_path,
                "front camera extrinsics",
            )
            intrinsics = _load_waymo_intrinsics_matrix(intrinsic_path)
            with Image.open(self.image_paths[idx][0]) as image:
                raw_hw = (int(image.height), int(image.width))
            payload = {
                "camera_to_ego": camera_to_ego,
                "camera_to_ego_dataset": (
                    camera_to_ego @ WAYMO_OPENCV2DATASET
                ).astype(np.float32),
                "intrinsics": intrinsics,
                "raw_hw": raw_hw,
                "ego_to_world": {},
            }
            if self.pretrain_instance_cache_size > 0:
                cache[scene_name] = payload
                while len(cache) > self.pretrain_instance_cache_size:
                    oldest_scene = next(iter(cache))
                    del cache[oldest_scene]

        ego_pose_paths = self.extrinsic_paths[idx]
        poses = payload["ego_to_world"]
        for frame_idx in dict.fromkeys(int(value) for value in frame_indices):
            if frame_idx not in poses:
                poses[frame_idx] = _load_waymo_matrix4(
                    ego_pose_paths[frame_idx],
                    "ego pose",
                )
        return payload

    def _load_front_waymo_camera_gt(self, idx, indices, image_seq):
        if self.views != 1:
            raise ValueError("Raw Waymo camera GT loading currently expects front-camera views=1 clips.")
        ego_pose_paths = self.extrinsic_paths[idx]
        if len(ego_pose_paths) == 0:
            raise RuntimeError(f"Scene {self.scenes[idx]} is missing ego_pose/*.txt files.")
        if max(indices) >= len(ego_pose_paths):
            raise RuntimeError(
                f"Scene {self.scenes[idx]} has {len(ego_pose_paths)} ego poses, "
                f"but clip needs frame {max(indices)}."
            )

        # Keep every sampled window in one clip-global Waymo coordinate frame.
        # Sliding inference constructs the full trajectory before slicing; using
        # the sampled window start here would give training a different camera
        # condition representation.
        clip_start = int(indices[0] // 29) * 29
        context_indices = sorted(set(
            [clip_start]
            + [max(clip_start, int(frame_idx) - 1) for frame_idx in indices]
            + list(indices)
        ))
        camera_metadata = self._load_waymo_camera_metadata(
            idx,
            context_indices,
        )
        cam_to_ego = camera_metadata["camera_to_ego_dataset"]
        ego_to_world_by_frame = camera_metadata["ego_to_world"]
        ego_to_world_start = ego_to_world_by_frame[clip_start]
        ego_start_inv = np.linalg.inv(ego_to_world_start).astype(np.float32)
        camera_by_frame = {}
        for frame_idx in context_indices:
            ego_to_world = ego_to_world_by_frame[frame_idx]
            camera_by_frame[int(frame_idx)] = (ego_start_inv @ ego_to_world @ cam_to_ego).astype(np.float32)
        camera_to_world = [camera_by_frame[int(frame_idx)] for frame_idx in indices]
        camera_to_world = np.stack(camera_to_world, axis=0)[:, None]
        camera_anchor_to_world = camera_by_frame[clip_start][None]
        previous_camera_to_world = np.stack(
            [camera_by_frame[max(clip_start, int(frame_idx) - 1)] for frame_idx in indices], axis=0
        )

        intrinsics = camera_metadata["intrinsics"][None]
        raw_height, raw_width = camera_metadata["raw_hw"]
        raw_image_size_hw = torch.tensor([int(raw_height), int(raw_width)], dtype=torch.long)
        return (
            torch.from_numpy(camera_to_world),
            torch.from_numpy(intrinsics),
            raw_image_size_hw,
            torch.from_numpy(camera_anchor_to_world),
            torch.from_numpy(previous_camera_to_world),
        )

    def _load_instance_metadata(self, idx):
        scene_name = str(self.scenes[idx])
        if scene_name in self._instance_metadata_cache:
            # Plain dicts preserve insertion order.  Pop/reinsert makes this a
            # small per-worker LRU without adding another dependency or lock.
            payload = self._instance_metadata_cache.pop(scene_name)
            self._instance_metadata_cache[scene_name] = payload
            return payload
        scene_root = self.scene_roots[idx] if idx < len(self.scene_roots) else os.path.join(self.image_dir, scene_name)
        instances_root = os.path.join(scene_root, "instances")
        paths = {
            "instances_info": os.path.join(instances_root, "instances_info.json"),
            "frame_instances": os.path.join(instances_root, "frame_instances.json"),
            "object_id_map": os.path.join(instances_root, "object_id_map.json"),
        }
        if not os.path.exists(paths["instances_info"]):
            payload = None
        else:
            try:
                with open(paths["instances_info"], "r") as f:
                    instances_info = json.load(f)
                frame_instances = {}
                if os.path.exists(paths["frame_instances"]):
                    with open(paths["frame_instances"], "r") as f:
                        frame_instances = json.load(f)
                object_id_map = {}
                if os.path.exists(paths["object_id_map"]):
                    with open(paths["object_id_map"], "r") as f:
                        object_id_map = json.load(f)
                payload = {
                    "instances_info": instances_info if isinstance(instances_info, dict) else {},
                    "frame_instances": frame_instances if isinstance(frame_instances, dict) else {},
                    "object_id_map": object_id_map if isinstance(object_id_map, dict) else {},
                }
            except Exception:
                payload = None
        # Parsed Waymo instance JSON is much larger than the source file.  An
        # unbounded cache is replicated in every persistent DataLoader worker
        # and eventually retains nearly every scene after enough shuffled
        # epochs.  Keep only a small hot set; random scene sampling has little
        # long-range locality, while a bounded cache prevents per-worker RAM
        # from growing with the complete dataset.
        if self.pretrain_instance_cache_size > 0:
            self._instance_metadata_cache[scene_name] = payload
            while len(self._instance_metadata_cache) > self.pretrain_instance_cache_size:
                oldest_scene = next(iter(self._instance_metadata_cache))
                del self._instance_metadata_cache[oldest_scene]
        return payload

    def _candidate_instance_ids_for_clip(self, metadata, indices):
        instances_info = metadata.get("instances_info", {})
        frame_instances = metadata.get("frame_instances", {})
        candidate_ids = []
        seen = set()
        for frame_idx in indices:
            frame_key = str(int(frame_idx))
            for object_id in frame_instances.get(frame_key, []):
                object_key = str(object_id)
                if object_key in instances_info and object_key not in seen:
                    candidate_ids.append(object_key)
                    seen.add(object_key)
        if candidate_ids:
            return candidate_ids
        return [str(k) for k in instances_info.keys()]

    def _class_priority_bonus(self, class_name):
        name = str(class_name).lower()
        if "vehicle" in name or "car" in name or "truck" in name or "bus" in name:
            return 50.0
        if "pedestrian" in name or "cyclist" in name or "person" in name:
            return 25.0
        return 0.0

    def _project_pretrain_object_slots_legacy_leaky(self, idx, indices, image_seq):
        """Legacy target-dynamic-mask oracle retained for offline ablations only."""
        max_objects = max(0, int(self.pretrain_max_objects))
        gh, gw = self.pretrain_patch_grid
        empty = {
            "pretrain_object_ids": [""] * max_objects,
            "pretrain_object_class_names": [""] * max_objects,
            "pretrain_object_bbox_model": torch.zeros((max_objects, len(indices), 4), dtype=torch.float32),
            "pretrain_object_patch_mask": torch.zeros((max_objects, len(indices), gh * gw), dtype=torch.bool),
            "pretrain_object_valid_mask": torch.zeros((max_objects, len(indices)), dtype=torch.bool),
            "pretrain_object_scores": torch.zeros((max_objects,), dtype=torch.float32),
            "pretrain_asset_source_kind": "legacy_fallback",
        }
        if max_objects <= 0 or self.views != 1:
            return empty
        metadata = self._load_instance_metadata(idx)
        if metadata is None:
            return empty
        instances_info = metadata.get("instances_info", {})
        if not instances_info:
            empty["pretrain_asset_source_kind"] = "instances_empty"
            return empty

        ego_pose_paths = self.extrinsic_paths[idx]
        camera_extrinsic_path = self.ego_paths[idx] if idx < len(self.ego_paths) else ""
        intrinsic_path = self.intrinsic_paths[idx]
        if isinstance(intrinsic_path, (list, tuple)):
            intrinsic_path = intrinsic_path[0] if len(intrinsic_path) > 0 else ""
        try:
            camera_to_ego_front = _load_waymo_matrix4(camera_extrinsic_path, "front camera extrinsics")
            intrinsics = _load_waymo_intrinsics_matrix(intrinsic_path)
        except Exception:
            return empty
        if len(ego_pose_paths) == 0 or max(indices) >= len(ego_pose_paths):
            return empty
        with Image.open(image_seq[0]) as img:
            raw_width, raw_height = img.size
        raw_hw = (int(raw_height), int(raw_width))
        model_hw = _waymo_resize_geometry(raw_hw, target_width=518)["out_hw"]
        dynamic_mask_paths = self.dynamic_mask_path[idx] if idx < len(self.dynamic_mask_path) else []
        dynamic_model_masks_by_frame = {}
        if isinstance(dynamic_mask_paths, list) and len(dynamic_mask_paths) > 0:
            for frame_idx in indices:
                if int(frame_idx) >= len(dynamic_mask_paths):
                    continue
                try:
                    dynamic_model_masks_by_frame[int(frame_idx)] = _load_waymo_dynamic_mask_model(
                        dynamic_mask_paths[int(frame_idx)],
                        target_width=518,
                        threshold=0.5,
                    )
                except Exception:
                    dynamic_model_masks_by_frame[int(frame_idx)] = None
        has_dynamic_model_mask = any(
            mask is not None and bool(np.any(mask))
            for mask in dynamic_model_masks_by_frame.values()
        )
        camera_to_world_by_frame = {}
        for frame_idx in indices:
            try:
                ego_to_world = _load_waymo_matrix4(ego_pose_paths[frame_idx], "ego pose")
            except Exception:
                return empty
            camera_to_world_by_frame[int(frame_idx)] = (
                ego_to_world @ camera_to_ego_front @ WAYMO_OPENCV2DATASET
            ).astype(np.float32)

        projected_candidates = []
        for object_id in self._candidate_instance_ids_for_clip(metadata, indices):
            instance_info = instances_info.get(str(object_id))
            if not isinstance(instance_info, dict):
                continue
            frame_annotations = instance_info.get("frame_annotations", {})
            frame_indices = [int(v) for v in frame_annotations.get("frame_idx", [])]
            obj_to_world_seq = frame_annotations.get("obj_to_world", [])
            box_size_seq = frame_annotations.get("box_size", [])
            if len(frame_indices) == 0 or len(obj_to_world_seq) == 0 or len(box_size_seq) == 0:
                continue
            track_lookup = {frame_idx: pos for pos, frame_idx in enumerate(frame_indices)}
            bbox_model = np.zeros((len(indices), 4), dtype=np.float32)
            box_valid_mask = np.zeros((len(indices),), dtype=np.bool_)
            depth_by_frame = np.zeros((len(indices),), dtype=np.float32)
            for local_idx, frame_idx in enumerate(indices):
                track_idx = track_lookup.get(int(frame_idx))
                if track_idx is None:
                    continue
                try:
                    obj_to_world = np.asarray(obj_to_world_seq[track_idx], dtype=np.float32).reshape(4, 4)
                    box_size = np.asarray(box_size_seq[track_idx], dtype=np.float32).reshape(3)
                except Exception:
                    continue
                camera_to_world = camera_to_world_by_frame[int(frame_idx)]
                world_to_camera = np.linalg.inv(camera_to_world)
                center_world = obj_to_world[:3, 3]
                center_cam = world_to_camera[:3, :3] @ center_world + world_to_camera[:3, 3]
                depth_z = float(center_cam[2])
                if not np.isfinite(depth_z) or depth_z <= 1e-6:
                    continue
                corners_world = _build_waymo_box_corners_world(obj_to_world, box_size)
                raw_box, _ = _project_world_points_to_raw(corners_world, camera_to_world, intrinsics, raw_hw)
                if raw_box is None:
                    continue
                model_box, _ = _transform_waymo_box_to_model(raw_box, raw_hw, target_width=518)
                if float(model_box[2] - model_box[0]) <= 0.0 or float(model_box[3] - model_box[1]) <= 0.0:
                    continue
                bbox_model[local_idx] = model_box
                box_valid_mask[local_idx] = True
                depth_by_frame[local_idx] = float(depth_z)
            if not bool(box_valid_mask.any()):
                continue
            projected_candidates.append(
                {
                    "object_id": str(instance_info.get("raw_object_id", instance_info.get("id", object_id))),
                    "class_name": str(instance_info.get("class_name", "")),
                    "is_moving_track": bool(instance_info.get("is_moving_track", False)),
                    "bbox_model": bbox_model,
                    "box_valid_mask": box_valid_mask,
                    "depth_by_frame": depth_by_frame,
                    "patch_mask": np.zeros((len(indices), gh * gw), dtype=np.bool_),
                    "valid_mask": np.zeros((len(indices),), dtype=np.bool_),
                    "patch_areas": [],
                    "inverse_depths": [],
                }
            )

        for local_idx, frame_idx in enumerate(indices):
            frame_items = [
                item
                for item in projected_candidates
                if bool(item["box_valid_mask"][local_idx])
            ]
            if not frame_items:
                continue
            boxes = np.stack([item["bbox_model"][local_idx] for item in frame_items], axis=0)
            assigned_masks = _assign_dynamic_patch_masks_to_boxes(
                dynamic_model_masks_by_frame.get(int(frame_idx)),
                boxes,
                model_hw,
                (gh, gw),
            )
            for item, frame_patch_mask in zip(frame_items, assigned_masks):
                patch_area = float(np.asarray(frame_patch_mask, dtype=np.bool_).sum())
                if patch_area <= 0.0:
                    continue
                item["patch_mask"][local_idx] = frame_patch_mask
                item["valid_mask"][local_idx] = True
                item["patch_areas"].append(patch_area)
                depth_z = float(item["depth_by_frame"][local_idx])
                item["inverse_depths"].append(1.0 / max(depth_z, 1e-6))

        candidates = []
        for item in projected_candidates:
            valid_mask = item["valid_mask"]
            patch_areas = item["patch_areas"]
            inverse_depths = item["inverse_depths"]
            coverage_frames = int(valid_mask.sum())
            if coverage_frames <= 0:
                continue
            median_area_patch = float(np.median(np.asarray(patch_areas, dtype=np.float32))) if patch_areas else 0.0
            if median_area_patch <= 0.0:
                continue
            foreground_score = float(np.mean(np.asarray(inverse_depths, dtype=np.float32))) if inverse_depths else 0.0
            class_name = str(item["class_name"])
            moving_bonus = 100.0 if bool(item["is_moving_track"]) else 0.0
            score = (
                float(coverage_frames) * 1000.0
                + median_area_patch * 20.0
                + foreground_score * 20000.0
                + moving_bonus
                + self._class_priority_bonus(class_name)
            )
            candidates.append(
                {
                    "object_id": str(item["object_id"]),
                    "class_name": class_name,
                    "bbox_model": item["bbox_model"],
                    "patch_mask": item["patch_mask"],
                    "valid_mask": valid_mask,
                    "score": float(score),
                }
            )

        if not candidates:
            empty["pretrain_asset_source_kind"] = (
                "instances_no_dynamic_projection" if has_dynamic_model_mask else "instances_no_dynamic_mask"
            )
            return empty
        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected = candidates[:max_objects]
        object_ids = [""] * max_objects
        class_names = [""] * max_objects
        bbox_model = np.zeros((max_objects, len(indices), 4), dtype=np.float32)
        patch_mask = np.zeros((max_objects, len(indices), gh * gw), dtype=np.bool_)
        valid_mask = np.zeros((max_objects, len(indices)), dtype=np.bool_)
        scores = np.zeros((max_objects,), dtype=np.float32)
        for slot, item in enumerate(selected):
            object_ids[slot] = str(item["object_id"])
            class_names[slot] = str(item["class_name"])
            bbox_model[slot] = item["bbox_model"]
            patch_mask[slot] = item["patch_mask"]
            valid_mask[slot] = item["valid_mask"]
            scores[slot] = float(item["score"])
        return {
            "pretrain_object_ids": object_ids,
            "pretrain_object_class_names": class_names,
            "pretrain_object_bbox_model": torch.tensor(bbox_model.tolist(), dtype=torch.float32),
            "pretrain_object_patch_mask": torch.tensor(patch_mask.tolist(), dtype=torch.bool),
            "pretrain_object_valid_mask": torch.tensor(valid_mask.tolist(), dtype=torch.bool),
            "pretrain_object_scores": torch.tensor(scores.tolist(), dtype=torch.float32),
            "pretrain_asset_source_kind": "instances_projected",
        }

    def _project_pretrain_object_slots(
        self,
        idx,
        indices,
        image_seq,
        *,
        deterministic_reference=False,
    ):
        """Build leak-free references and target placement from raw Waymo metadata.

        Target boxes, validity and patch support below use only 3D tracks and
        cameras. Class-matched semantic foreground is loaded exclusively for
        the one source reference frame selected outside ``indices``.
        """
        del image_seq
        max_objects = max(0, int(self.pretrain_max_objects))
        seq_len = len(indices)
        gh, gw = self.pretrain_patch_grid
        canvas_hw = (gh * 14, gw * 14)

        def empty_payload(source_kind="factorized_empty"):
            eye = torch.eye(4, dtype=torch.float32)
            return {
                "pretrain_object_ids": [""] * max_objects,
                "pretrain_object_class_names": [""] * max_objects,
                "pretrain_object_obj_to_anchor": eye.view(1, 1, 4, 4).repeat(max_objects, seq_len, 1, 1),
                "pretrain_object_center_anchor": torch.zeros((max_objects, seq_len, 3), dtype=torch.float32),
                "pretrain_object_box_size": torch.zeros((max_objects, seq_len, 3), dtype=torch.float32),
                "pretrain_object_yaw": torch.zeros((max_objects, seq_len), dtype=torch.float32),
                "pretrain_object_yaw_sincos": torch.zeros((max_objects, seq_len, 2), dtype=torch.float32),
                "pretrain_object_velocity_anchor": torch.zeros((max_objects, seq_len, 3), dtype=torch.float32),
                "pretrain_object_track_valid": torch.zeros((max_objects, seq_len), dtype=torch.bool),
                "pretrain_object_in_frustum": torch.zeros((max_objects, seq_len), dtype=torch.bool),
                "pretrain_object_bbox_model": torch.zeros((max_objects, seq_len, 4), dtype=torch.float32),
                "pretrain_object_bbox_patch": torch.full((max_objects, seq_len, 4), -1.0, dtype=torch.float32),
                "pretrain_object_patch_mask": torch.zeros((max_objects, seq_len, gh * gw), dtype=torch.bool),
                "pretrain_object_valid_mask": torch.zeros((max_objects, seq_len), dtype=torch.bool),
                "pretrain_object_scores": torch.zeros((max_objects,), dtype=torch.float32),
                "pretrain_reference_rgb": torch.zeros((max_objects, 3, *canvas_hw), dtype=torch.float32),
                "pretrain_reference_alpha": torch.zeros((max_objects, 1, *canvas_hw), dtype=torch.float32),
                "pretrain_reference_patch_mask": torch.zeros((max_objects, gh * gw), dtype=torch.bool),
                "pretrain_reference_frame_id": torch.full((max_objects,), -1, dtype=torch.long),
                "pretrain_camera_to_anchor": eye.view(1, 4, 4).repeat(seq_len, 1, 1),
                "pretrain_asset_source_kind": source_kind,
                "pretrain_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
                "pretrain_fps": torch.tensor(10.0, dtype=torch.float32),
            }

        empty = empty_payload()
        if max_objects <= 0 or self.views != 1:
            return empty
        metadata = self._load_instance_metadata(idx)
        if metadata is None or not metadata.get("instances_info"):
            return empty_payload("factorized_instances_missing")
        ego_pose_paths = self.extrinsic_paths[idx]
        if not ego_pose_paths:
            return empty_payload("factorized_camera_missing")
        try:
            camera_metadata = self._load_waymo_camera_metadata(idx)
            camera_to_ego = camera_metadata["camera_to_ego"]
            intrinsics = camera_metadata["intrinsics"]
        except Exception:
            return empty_payload("factorized_camera_missing")

        total_frames = len(self.image_paths[idx])
        trunk_base = (int(indices[0]) // int(self.trunk_frames)) * int(self.trunk_frames)
        trunk_end = min(trunk_base + int(self.trunk_frames), total_frames)
        trunk_indices = list(range(trunk_base, trunk_end))
        target_set = {int(value) for value in indices}
        reference_indices = [value for value in trunk_indices if value not in target_set]
        if not reference_indices:
            # In particular, a complete 29-frame target window cannot source a
            # reference from itself.
            return empty_payload("factorized_no_external_reference")

        try:
            camera_metadata = self._load_waymo_camera_metadata(
                idx,
                trunk_indices,
            )
            ego_to_world_by_frame = camera_metadata["ego_to_world"]
            anchor_ego_to_world = ego_to_world_by_frame[trunk_base]
            anchor_to_world = (
                anchor_ego_to_world @ camera_to_ego @ WAYMO_OPENCV2DATASET
            ).astype(np.float32)
            world_to_anchor = np.linalg.inv(anchor_to_world).astype(np.float32)
        except Exception:
            return empty_payload("factorized_anchor_missing")

        camera_to_anchor = []
        camera_to_world = {}
        world_to_camera = {}
        for frame_idx in trunk_indices:
            ego_to_world = ego_to_world_by_frame[frame_idx]
            c2w = (ego_to_world @ camera_to_ego @ WAYMO_OPENCV2DATASET).astype(np.float32)
            camera_to_world[frame_idx] = c2w
            world_to_camera[frame_idx] = np.linalg.inv(c2w).astype(np.float32)
            if frame_idx in target_set:
                camera_to_anchor.append((world_to_anchor @ c2w).astype(np.float32))
        empty["pretrain_camera_to_anchor"] = torch.from_numpy(
            np.stack(camera_to_anchor, axis=0)
        )

        raw_hw = camera_metadata["raw_hw"]
        model_hw = _waymo_resize_geometry(raw_hw, target_width=518)["out_hw"]
        if tuple(model_hw) != tuple(canvas_hw):
            # The tokenizer patch grid must describe the actual DGGT canvas.
            canvas_hw = tuple(int(value) for value in model_hw)
            empty["pretrain_reference_rgb"] = torch.zeros((max_objects, 3, *canvas_hw), dtype=torch.float32)
            empty["pretrain_reference_alpha"] = torch.zeros((max_objects, 1, *canvas_hw), dtype=torch.float32)

        semantic_paths = self.semantic_mask_path[idx] if idx < len(self.semantic_mask_path) else []
        instances_info = metadata["instances_info"]
        frame_instances = metadata.get("frame_instances", {})

        geometry_by_object = {}
        object_records = []
        for object_id in self._candidate_instance_ids_for_clip(
            metadata,
            trunk_indices,
        ):
            object_key = str(object_id)
            info = instances_info.get(object_key)
            if not isinstance(info, dict):
                continue
            annotations = info.get("frame_annotations", {})
            frame_seq = [
                int(value) for value in annotations.get("frame_idx", [])
            ]
            poses = annotations.get("obj_to_world", [])
            sizes = annotations.get("box_size", [])
            lookup = {frame: pos for pos, frame in enumerate(frame_seq)}
            if not lookup:
                continue

            per_frame = {}
            for frame_idx in trunk_indices:
                pos = lookup.get(int(frame_idx))
                if pos is None or pos >= len(poses) or pos >= len(sizes):
                    continue
                try:
                    o2w = np.asarray(
                        poses[pos],
                        dtype=np.float32,
                    ).reshape(4, 4)
                    size = np.asarray(
                        sizes[pos],
                        dtype=np.float32,
                    ).reshape(3)
                except Exception:
                    continue
                if (
                    not np.isfinite(o2w).all()
                    or not np.isfinite(size).all()
                    or np.any(size <= 0.0)
                ):
                    continue
                raw_box, points_cam = _project_world_points_to_raw(
                    _build_waymo_box_corners_world(o2w, size),
                    camera_to_world[frame_idx],
                    intrinsics,
                    raw_hw,
                    world_to_camera=world_to_camera[frame_idx],
                )
                in_frustum = raw_box is not None
                model_box = np.zeros((4,), dtype=np.float32)
                patch_mask = np.zeros((gh * gw,), dtype=np.bool_)
                bbox_patch = np.full((4,), -1.0, dtype=np.float32)
                if raw_box is not None:
                    model_box, _ = _transform_waymo_box_to_model(
                        raw_box,
                        raw_hw,
                        target_width=518,
                    )
                    patch_mask, _ = _model_box_to_patch_mask(
                        model_box,
                        model_hw,
                        (gh, gw),
                    )
                    bbox_patch = np.array(
                        [
                            model_box[0] / float(model_hw[1]) * gw,
                            model_box[1] / float(model_hw[0]) * gh,
                            model_box[2] / float(model_hw[1]) * gw,
                            model_box[3] / float(model_hw[0]) * gh,
                        ],
                        dtype=np.float32,
                    )
                o2a = (world_to_anchor @ o2w).astype(np.float32)
                heading = o2a[:3, 0]
                yaw = float(
                    np.arctan2(
                        float(heading[0]),
                        float(heading[2]) + 1.0e-8,
                    )
                )
                positive_depth = points_cam[:, 2][
                    points_cam[:, 2] > 1.0e-6
                ]
                per_frame[frame_idx] = {
                    "o2a": o2a,
                    "center": o2a[:3, 3].copy(),
                    "size": size,
                    "yaw": yaw,
                    "model_box": model_box,
                    "bbox_patch": bbox_patch,
                    "patch_mask": patch_mask,
                    "in_frustum": bool(
                        in_frustum and positive_depth.size > 0
                    ),
                    "depth": (
                        float(np.median(positive_depth))
                        if positive_depth.size > 0
                        else float("inf")
                    ),
                }
            geometry_by_object[object_key] = per_frame
            object_records.append((object_key, info, per_frame))

        def reference_overlap_is_acceptable(
            frame_idx,
            current_key,
            current_box,
            current_depth,
        ):
            current = np.asarray(current_box, dtype=np.float32)
            current_area = max(0.0, float(current[2] - current[0])) * max(
                0.0, float(current[3] - current[1])
            )
            if current_area <= 0.0:
                return False
            for other_id in frame_instances.get(str(int(frame_idx)), []):
                other_key = str(other_id)
                if other_key == str(current_key):
                    continue
                other_geom = geometry_by_object.get(other_key, {}).get(
                    int(frame_idx)
                )
                if other_geom is None or not other_geom["in_frustum"]:
                    continue
                other_box = other_geom["model_box"]
                ix0 = max(float(current[0]), float(other_box[0]))
                iy0 = max(float(current[1]), float(other_box[1]))
                ix1 = min(float(current[2]), float(other_box[2]))
                iy1 = min(float(current[3]), float(other_box[3]))
                intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if intersection <= 0.0:
                    continue
                other_area = max(0.0, float(other_box[2] - other_box[0])) * max(
                    0.0, float(other_box[3] - other_box[1])
                )
                union = max(current_area + other_area - intersection, 1.0e-6)
                iou = intersection / union
                other_depth = float(other_geom["depth"])
                if iou > 0.5 or (
                    other_depth < float(current_depth)
                    and intersection / current_area > 0.25
                ):
                    return False
            return True

        semantic_labels_by_frame = {}
        semantic_foreground_cache = {}
        semantic_integral_cache = {}

        def semantic_foreground(frame_idx, class_name):
            semantic_values = _waymo_semantic_values_for_class(class_name)
            if (
                not semantic_values
                or not isinstance(semantic_paths, list)
                or frame_idx >= len(semantic_paths)
            ):
                return None
            key = (int(frame_idx), semantic_values)
            if key in semantic_foreground_cache:
                return semantic_foreground_cache[key]
            if frame_idx not in semantic_labels_by_frame:
                semantic_labels_by_frame[frame_idx] = (
                    _load_waymo_semantic_labels_model(
                        semantic_paths[frame_idx],
                        target_width=518,
                    )
                )
            labels = semantic_labels_by_frame[frame_idx]
            foreground = (
                None
                if labels is None
                else np.isin(labels, semantic_values)
            )
            semantic_foreground_cache[key] = foreground
            return foreground

        def semantic_integral(frame_idx, class_name, foreground):
            key = (
                int(frame_idx),
                _waymo_semantic_values_for_class(class_name),
            )
            if key not in semantic_integral_cache:
                semantic_integral_cache[key] = np.pad(
                    foreground.astype(
                        np.int32,
                        copy=False,
                    ).cumsum(0).cumsum(1),
                    ((1, 0), (1, 0)),
                )
            return semantic_integral_cache[key]

        projected = []
        for object_id, info, per_frame in object_records:
            reference_candidates = []
            for frame_idx in reference_indices:
                geom = per_frame.get(frame_idx)
                if geom is None or not geom["in_frustum"]:
                    continue
                if not reference_overlap_is_acceptable(
                    frame_idx,
                    object_id,
                    geom["model_box"],
                    geom["depth"],
                ):
                    continue
                semantic = semantic_foreground(
                    frame_idx,
                    info.get("class_name", ""),
                )
                if semantic is None:
                    continue
                reference_patch_mask, foreground_area = (
                    _boxed_binary_mask_patch_support(
                        semantic,
                        geom["model_box"],
                        (gh, gw),
                        integral=semantic_integral(
                            frame_idx,
                            info.get("class_name", ""),
                            semantic,
                        ),
                    )
                )
                patch_count = int(reference_patch_mask.sum())
                if patch_count < 4:
                    continue
                reference_candidates.append(
                    (
                        frame_idx,
                        reference_patch_mask,
                        patch_count,
                        float(foreground_area),
                    )
                )
            if not reference_candidates:
                continue
            reference_frame, _, _, _ = (
                _select_pretrain_reference_candidate(
                    reference_candidates,
                    deterministic=bool(deterministic_reference),
                )
            )

            target_valid = np.array([frame in per_frame for frame in indices], dtype=np.bool_)
            target_centers = np.zeros((seq_len, 3), dtype=np.float32)
            target_sizes = np.zeros((seq_len, 3), dtype=np.float32)
            target_yaw = np.zeros((seq_len,), dtype=np.float32)
            target_o2a = np.tile(np.eye(4, dtype=np.float32), (seq_len, 1, 1))
            target_box_model = np.zeros((seq_len, 4), dtype=np.float32)
            target_bbox_patch = np.full((seq_len, 4), -1.0, dtype=np.float32)
            target_patch_mask = np.zeros((seq_len, gh * gw), dtype=np.bool_)
            target_in_frustum = np.zeros((seq_len,), dtype=np.bool_)
            for local_idx, frame_idx in enumerate(indices):
                geom = per_frame.get(frame_idx)
                if geom is None:
                    continue
                target_o2a[local_idx] = geom["o2a"]
                target_centers[local_idx] = geom["center"]
                target_sizes[local_idx] = geom["size"]
                target_yaw[local_idx] = geom["yaw"]
                target_box_model[local_idx] = geom["model_box"]
                target_bbox_patch[local_idx] = geom["bbox_patch"]
                target_patch_mask[local_idx] = geom["patch_mask"]
                target_in_frustum[local_idx] = geom["in_frustum"]
            velocity = np.zeros_like(target_centers)
            valid_frames = sorted(int(frame) for frame in per_frame)
            valid_rank = {frame: rank for rank, frame in enumerate(valid_frames)}
            for local_idx, frame_idx in enumerate(indices):
                if not target_valid[local_idx] or len(valid_frames) < 2:
                    continue
                rank = valid_rank[int(frame_idx)]
                left_frame = valid_frames[max(0, rank - 1)]
                right_frame = valid_frames[min(len(valid_frames) - 1, rank + 1)]
                if left_frame == right_frame:
                    continue
                dt = float(right_frame - left_frame) / 10.0
                velocity[local_idx] = (
                    per_frame[right_frame]["center"]
                    - per_frame[left_frame]["center"]
                ) / max(dt, 1.0e-6)
            coverage = int(target_valid.sum())
            area = float(target_patch_mask.sum())
            score = coverage * 1000.0 + area + self._class_priority_bonus(info.get("class_name", ""))
            projected.append(
                {
                    "object_id": str(info.get("raw_object_id", info.get("id", object_id))),
                    "class_name": str(info.get("class_name", "")),
                    "o2a": target_o2a,
                    "center": target_centers,
                    "size": target_sizes,
                    "yaw": target_yaw,
                    "velocity": velocity,
                    "track_valid": target_valid,
                    "in_frustum": target_in_frustum,
                    "bbox_model": target_box_model,
                    "bbox_patch": target_bbox_patch,
                    "patch_mask": target_patch_mask,
                    "reference_frame": int(reference_frame - trunk_base),
                    "reference_frame_absolute": int(reference_frame),
                    "reference_box_model": per_frame[reference_frame][
                        "model_box"
                    ],
                    "score": score,
                }
            )

        if not projected:
            return empty_payload("factorized_no_external_reference")
        projected.sort(key=lambda item: item["score"], reverse=True)
        selected = []
        reference_rgb_cache = {}
        for item in projected:
            reference_frame = item["reference_frame_absolute"]
            semantic = semantic_foreground(
                reference_frame,
                item["class_name"],
            )
            if semantic is None:
                continue
            x0, y0, x1, y1 = [
                int(round(float(value)))
                for value in item["reference_box_model"]
            ]
            x0, x1 = max(0, x0), min(int(model_hw[1]), x1)
            y0, y1 = max(0, y0), min(int(model_hw[0]), y1)
            reference_alpha_np = np.zeros(model_hw, dtype=np.float32)
            if x1 > x0 and y1 > y0:
                reference_alpha_np[y0:y1, x0:x1] = semantic[
                    y0:y1,
                    x0:x1,
                ]
            reference_alpha = torch.from_numpy(
                reference_alpha_np
            ).unsqueeze(0)
            if reference_frame not in reference_rgb_cache:
                reference_rgb_cache[reference_frame] = (
                    load_and_preprocess_images(
                        [self.image_paths[idx][reference_frame]]
                    )[0]
                )
            canonical_rgb, canonical_alpha = canonicalize_asset_reference(
                reference_rgb_cache[reference_frame],
                reference_alpha,
                canvas_hw,
            )
            canonical_patch_mask = alpha_to_patch_mask(
                canonical_alpha,
                (gh, gw),
            )
            if int(canonical_patch_mask.sum().item()) < 4:
                continue
            item["reference_rgb"] = canonical_rgb
            item["reference_alpha"] = canonical_alpha
            item["reference_patch_mask"] = canonical_patch_mask
            selected.append(item)
            if len(selected) >= max_objects:
                break
        if not selected:
            return empty_payload("factorized_no_external_reference")
        for slot, item in enumerate(selected):
            empty["pretrain_object_ids"][slot] = item["object_id"]
            empty["pretrain_object_class_names"][slot] = item["class_name"]
            empty["pretrain_object_obj_to_anchor"][slot] = torch.from_numpy(item["o2a"])
            empty["pretrain_object_center_anchor"][slot] = torch.from_numpy(item["center"])
            empty["pretrain_object_box_size"][slot] = torch.from_numpy(item["size"])
            empty["pretrain_object_yaw"][slot] = torch.from_numpy(item["yaw"])
            empty["pretrain_object_yaw_sincos"][slot, :, 0] = torch.sin(empty["pretrain_object_yaw"][slot])
            empty["pretrain_object_yaw_sincos"][slot, :, 1] = torch.cos(empty["pretrain_object_yaw"][slot])
            empty["pretrain_object_velocity_anchor"][slot] = torch.from_numpy(item["velocity"])
            empty["pretrain_object_track_valid"][slot] = torch.from_numpy(item["track_valid"])
            empty["pretrain_object_in_frustum"][slot] = torch.from_numpy(item["in_frustum"])
            empty["pretrain_object_bbox_model"][slot] = torch.from_numpy(item["bbox_model"])
            empty["pretrain_object_bbox_patch"][slot] = torch.from_numpy(item["bbox_patch"])
            empty["pretrain_object_patch_mask"][slot] = torch.from_numpy(item["patch_mask"])
            empty["pretrain_object_valid_mask"][slot] = torch.from_numpy(item["in_frustum"])
            empty["pretrain_object_scores"][slot] = float(item["score"])
            empty["pretrain_reference_rgb"][slot] = item["reference_rgb"]
            empty["pretrain_reference_alpha"][slot] = item["reference_alpha"]
            empty["pretrain_reference_patch_mask"][slot] = item["reference_patch_mask"]
            empty["pretrain_reference_frame_id"][slot] = int(item["reference_frame"])
        projection_intrinsics, projection_image_hw = (
            resize_crop_intrinsics_to_model_canvas(
                torch.from_numpy(intrinsics),
                raw_hw,
                target_width=518,
                patch_size=14,
            )
        )
        projected_bbox, projected_in_frustum = project_anchor_boxes_to_patch_bboxes(
            empty["pretrain_object_obj_to_anchor"].unsqueeze(0),
            empty["pretrain_object_box_size"].unsqueeze(0),
            empty["pretrain_object_track_valid"].unsqueeze(0),
            empty["pretrain_camera_to_anchor"].unsqueeze(0),
            projection_intrinsics
            .view(1, 1, 3, 3)
            .repeat(1, seq_len, 1, 1),
            projection_image_hw,
            (gh, gw),
        )
        empty["pretrain_object_bbox_patch"] = projected_bbox[0]
        empty["pretrain_object_in_frustum"] = projected_in_frustum[0]
        empty["pretrain_object_valid_mask"] = projected_in_frustum[0].clone()
        empty["pretrain_object_patch_mask"] = bbox_patch_mask(
            projected_bbox,
            projected_in_frustum,
            (gh, gw),
        )[0]
        empty["pretrain_asset_source_kind"] = "instances_projected"
        return empty

    def build_pretrain_asset_payload_for_sample_window(
        self,
        sample_index,
        window_start,
        window_end,
    ):
        """Build leak-free asset metadata for one inference window.

        This is intentionally a thin public wrapper around the same projector
        used by ``__getitem__``.  It is used only when raw-video inference
        generates a long trunk through sliding windows: each window must obtain
        its reference from the frames outside that window, exactly as training
        does. Unlike training augmentation, inference deterministically prefers
        the highest-quality legal reference so overlapping windows reuse the
        same appearance whenever possible.
        """
        if not self.trunk_major_samples:
            raise RuntimeError(
                "window-specific pretrain asset payloads require trunk_major_samples=True"
            )
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= len(self.trunk_major_index):
            raise IndexError(
                f"sample index {sample_index} is outside [0,{len(self.trunk_major_index)})"
            )
        window_start, window_end = int(window_start), int(window_end)
        if window_start < 0 or window_end <= window_start:
            raise ValueError(
                f"invalid local inference window [{window_start},{window_end})"
            )
        entry = self.trunk_major_index[sample_index]
        scene_idx, trunk_idx = int(entry[0]), int(entry[1])
        paths = self.image_paths[scene_idx]
        total_frames = len(paths[0] if self.views == 3 else paths)
        if len(entry) == 3:
            sample_start = self._fixed_start_in_trunk(
                total_frames,
                trunk_idx,
                window_offset=int(entry[2]),
            )
        else:
            sample_start = self._fixed_start_in_trunk(total_frames, trunk_idx)
        if window_end > int(self.sequence_length):
            raise ValueError(
                f"local inference window [{window_start},{window_end}) exceeds "
                f"sample sequence length {self.sequence_length}"
            )
        indices = list(
            range(sample_start + window_start, sample_start + window_end)
        )
        image_seq = [paths[index] for index in indices]
        return self._project_pretrain_object_slots(
            scene_idx,
            indices,
            image_seq,
            deterministic_reference=True,
        )

    def __getitem__(self, idx):
        trunk_idx = None
        trunk_window_offset = None
        if self.trunk_major_samples:
            trunk_entry = self.trunk_major_index[idx]
            if len(trunk_entry) == 3:
                idx, trunk_idx, trunk_window_offset = trunk_entry
            else:
                idx, trunk_idx = trunk_entry
        image_paths = self.image_paths[idx]
        sky_mask_paths = self.sky_mask_paths[idx]
        dynamic_mask_paths = self.dynamic_mask_path[idx]
        semantic_mask_paths = self.semantic_mask_path[idx]

        total_frames = len(image_paths[0] if self.views == 3 else image_paths)
        if trunk_idx is not None:
            if trunk_window_offset is not None:
                start_idx = self._fixed_start_in_trunk(
                    total_frames, trunk_idx, window_offset=trunk_window_offset
                )
            elif self.mode == 1 and int(self.start_idx) >= 0:
                start_idx = self._fixed_start_in_trunk(total_frames, trunk_idx)
            else:
                start_idx = self._sample_start_in_trunk(total_frames, trunk_idx)
        elif self.mode == 1 and int(self.start_idx) >= 0:
            start_idx = self._fixed_start_within_caption_trunk(total_frames)
        elif self.mode == 1:
            start_idx = self._sample_start_within_caption_trunk(total_frames)
        else:
            start_idx = 0

        if self.mode == 1:
            indices = list(range(start_idx, start_idx + self.sequence_length))

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                if self.return_full_dggt_context:
                    context_base = (int(start_idx) // self.trunk_frames) * self.trunk_frames
                    context_end = context_base + self.trunk_frames
                    if context_end > total_frames:
                        raise RuntimeError(
                            "Full DGGT context requires a complete caption trunk: "
                            f"scene={self.scenes[idx]} base={context_base} end={context_end} "
                            f"total_frames={total_frames}"
                        )
                    context_indices = list(range(context_base, context_end))
                    context_seq = [image_paths[i] for i in context_indices]
                    dggt_context_images = load_and_preprocess_images(
                        context_seq,
                        output_dtype=self.image_output_dtype,
                    )
                    window_context_indices = torch.tensor(
                        [i - context_base for i in indices], dtype=torch.long
                    )
                    images = dggt_context_images.index_select(0, window_context_indices)
                else:
                    dggt_context_images = None
                    window_context_indices = None
                    images = load_and_preprocess_images(
                        seq,
                        output_dtype=self.image_output_dtype,
                    )  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(
                    seq,
                    output_dtype=self.image_output_dtype,
                )  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_binary_masks(
                    mask_seq,
                    channels=self.binary_mask_channels,
                )  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_binary_masks(
                    mask_seq,
                    channels=self.binary_mask_channels,
                )  # [S*3, C, H, W]

            frame_ids = np.array(indices, dtype=np.int64) - int(start_idx // 29) * 29
            timestamps = gaussian_timestamps_from_frame_ids(frame_ids)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
                frame_ids = np.repeat(frame_ids, 3)

            input_dict = {
                "images": images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                "frame_ids": frame_ids,
                "interval": [1] * (self.sequence_length - 1),
                "scene_name": self.scenes[idx],
                "start_idx": start_idx,
                "clip_index": start_idx // 29,
            }
            if self.return_full_dggt_context:
                input_dict["dggt_context_images"] = dggt_context_images
                input_dict["dggt_window_indices"] = window_context_indices
                input_dict["dggt_context_frame_ids"] = torch.arange(
                    self.trunk_frames, dtype=torch.long
                )
                input_dict["window_contains_camera_anchor"] = bool(frame_ids[0] == 0)
            caption = self._load_caption(self.scenes[idx], start_idx)
            if caption is not None:
                if self.views == 1:
                    caption = caption.replace("multi-camera", "front-camera")
                input_dict["caption"] = caption
                input_dict["caption_path"] = self._caption_path(self.scenes[idx], start_idx)

            if self.views == 1:
                (
                    camera_to_world,
                    intrinsics,
                    raw_image_size_hw,
                    camera_anchor_to_world,
                    previous_camera_to_world,
                ) = self._load_front_waymo_camera_gt(idx, indices, seq)
                input_dict["camera_to_world_corrected"] = camera_to_world
                input_dict["intrinsics"] = intrinsics
                input_dict["raw_image_size_hw"] = raw_image_size_hw
                input_dict["camera_trajectory_anchor_to_world_corrected"] = camera_anchor_to_world
                input_dict["camera_previous_to_world_corrected"] = previous_camera_to_world
        
            if self.load_dynamic_masks and len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_binary_masks(
                        dy_mask_seq,
                        channels=self.binary_mask_channels,
                    )  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_binary_masks(
                        dy_mask_seq,
                        channels=self.binary_mask_channels,
                    )  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            if self.views == 1:
                input_dict.update(self._project_pretrain_object_slots(idx, indices, seq))

            
            # if len(semantic_mask_paths) > 0:
            #     if self.views == 1:
            #         sem_mask_seq = [semantic_mask_paths[i] for i in indices]
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S, C, H, W]
            #     elif self.views == 3:
            #         sem_mask_seq = []
            #         for i in indices:
            #             for v in range(3):
            #                 sem_mask_seq.append(semantic_mask_paths[v][i])
            #         semantic_mask = load_and_preprocess_images(sem_mask_seq)  # [S*3, C, H, W]
            #     semantic_mask = semantic_mask * 255 / 10
            #     semantic_mask = semantic_mask.int()
            #     semantic_mask[semantic_mask > 9] = 255
            #     input_dict["semantic_mask"] = semantic_mask

            return input_dict

        elif self.mode == 2: 
            start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            
            frame_ids = np.array(indices, dtype=np.int64) - int(start_idx // 29) * 29
            timestamps = gaussian_timestamps_from_frame_ids(frame_ids)
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
                frame_ids = np.repeat(frame_ids, 3)

            #images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

            #sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S*3, C, H, W]
                


            input_dict = {
                "images": images,
                "image_paths": seq,
                "masks": masks,
                "timestamps": timestamps,
                "frame_ids": frame_ids,
                "interval": intervals,
                "scene_name": self.scenes[idx],
                "start_idx": start_idx,
                "clip_index": start_idx // 29,
            }
            caption = self._load_caption(self.scenes[idx], start_idx)
            if caption is not None:
                input_dict["caption"] = caption
                input_dict["caption_path"] = self._caption_path(self.scenes[idx], start_idx)
            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S*3, C, H, W]
                input_dict["dynamic_mask"] = dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    depth_data = torch.zeros(len(indices), images.shape[2], images.shape[3])
                else:
                    depth_data = torch.zeros(len(indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = depth_data

            return input_dict

        else:  # self.mode == 3
            start_idx = 0
            indices = [start_idx + i * self.interval for i in range(self.sequence_length)]
            intervals = [self.interval for _ in range(self.sequence_length - 1)]
            target_indices = [start_idx + i for i in range(self.sequence_length * self.interval - (self.interval - 1))]

            timestamps = gaussian_timestamps_from_frame_ids(np.array(indices, dtype=np.int64))
            if self.views == 3:
                timestamps = np.repeat(timestamps, 3)
            
            # images
            if self.views == 1:
                seq = [image_paths[i] for i in indices]
                images = load_and_preprocess_images(seq)  # [S, C, H, W]
                target_seq = [image_paths[i] for i in target_indices]
                target_images = load_and_preprocess_images(target_seq)  # [T, C, H, W]
            elif self.views == 3:
                seq = []
                for i in indices:
                    for v in range(3):
                        seq.append(image_paths[v][i])
                images = load_and_preprocess_images(seq)  # [S*3, C, H, W]

                target_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_seq.append(image_paths[v][i])
                target_images = load_and_preprocess_images(target_seq)  # [T*3, C, H, W]

            # sky masks
            if self.views == 1:
                mask_seq = [sky_mask_paths[i] for i in indices]
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S, C, H, W]
                target_mask_seq = [sky_mask_paths[i] for i in target_indices]
                target_masks = load_and_preprocess_binary_masks(target_mask_seq)  # [T, C, H, W]
            elif self.views == 3:
                mask_seq = []
                for i in indices:
                    for v in range(3):
                        mask_seq.append(sky_mask_paths[v][i])
                masks = load_and_preprocess_binary_masks(mask_seq)  # [S*3, C, H, W]

                target_mask_seq = []
                for i in target_indices:
                    for v in range(3):
                        target_mask_seq.append(sky_mask_paths[v][i])
                target_masks = load_and_preprocess_binary_masks(target_mask_seq)  # [T*3, C, H, W]

            input_dict = {
                "images": images,
                "targets": target_images,
                "masks": masks,
                "image_paths": seq,
                "timestamps": timestamps,
                # "target_timestamps": target_timestamps,
                "interval": intervals,
                "target_masks": target_masks,
                "scene_name": self.scenes[idx],
                "start_idx": start_idx,
                "clip_index": start_idx // 29,
            }
            caption = self._load_caption(self.scenes[idx], start_idx)
            if caption is not None:
                input_dict["caption"] = caption

            if len(dynamic_mask_paths) > 0:
                if self.views == 1:
                    dy_mask_seq = [dynamic_mask_paths[i] for i in indices]
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)  # [S, C, H, W]
                    target_dy_mask_seq = [dynamic_mask_paths[i] for i in target_indices]
                    target_dynamic_mask = load_and_preprocess_binary_masks(target_dy_mask_seq)  # [T, C, H, W]
                elif self.views == 3:
                    dy_mask_seq = []
                    target_dy_mask_seq = []
                    for i in indices:
                        for v in range(3):
                            dy_mask_seq.append(dynamic_mask_paths[v][i])
                    for i in target_indices:
                        for v in range(3):
                            target_dy_mask_seq.append(dynamic_mask_paths[v][i])
                    dynamic_mask = load_and_preprocess_binary_masks(dy_mask_seq)         # [S*3, C, H, W]
                    target_dynamic_mask = load_and_preprocess_binary_masks(target_dy_mask_seq)  # [T*3, C, H, W]
                input_dict["dynamic_mask"] = target_dynamic_mask

            if len(self.depth_flow_paths) > 0 and len(self.depth_flow_paths[idx]) > 0:
                if self.views == 1:
                    if len(self.depth_flow_paths[idx]) > 0:
                        depth_seq = [self.depth_flow_paths[idx][i] for i in indices]
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_seq = [self.depth_flow_paths[idx][i] for i in target_indices]
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
                elif self.views == 3:
                    # Check if all views have depth paths
                    if all(len(self.depth_flow_paths[idx][v]) > 0 for v in range(3)):
                        depth_seq = []
                        target_depth_seq = []
                        for i in indices:
                            for v in range(3):
                                depth_seq.append(self.depth_flow_paths[idx][v][i])
                        for i in target_indices:
                            for v in range(3):
                                target_depth_seq.append(self.depth_flow_paths[idx][v][i])
                        depth_data = load_and_preprocess_flow(depth_seq, None, None, images.shape[2], images.shape[3])
                        target_depth_data = load_and_preprocess_flow(target_depth_seq, None, None, images.shape[2], images.shape[3])
                    else:
                        target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                    input_dict["gt_depth"] = target_depth_data
            else:
                # No depth data available, create zero tensor with same shape as images
                if self.views == 1:
                    target_depth_data = torch.zeros(len(target_indices), images.shape[2], images.shape[3])
                else:
                    target_depth_data = torch.zeros(len(target_indices) * 3, images.shape[2], images.shape[3])
                input_dict["gt_depth"] = target_depth_data
            return input_dict
