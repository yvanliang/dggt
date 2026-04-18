from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


CAM_NAME_TO_ID = {
    "pinhole_front": 0,
    "pinhole_front_left": 1,
    "pinhole_front_right": 2,
    "pinhole_side_left": 3,
    "pinhole_side_right": 4,
}
CAM_ID_TO_NAME = {cam_id: cam_name for cam_name, cam_id in CAM_NAME_TO_ID.items()}

DEFAULT_TRANSFER_HW = (704, 1280)
DEFAULT_PROCESSED_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt"
DEFAULT_TRANSFER_ROOT = "/data/disk2/lyy_dataset/waymo_transfer"
DEFAULT_RAW_ROOT = "/data/disk2/lyy_dataset/waymo"
DEFAULT_ASSET_ROOT = "/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed"
WAYMO_DYNAMIC_SPEED_THRESH_MPS = 1.0
WAYMO_OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
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
        flow = F.interpolate(
            flow.permute(0, 3, 1, 2),
            size=target_size,
            mode="nearest",
        )
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    return flow.squeeze()


def load_and_preprocess_flow(flow_path_list, height, width):
    if len(flow_path_list) == 0:
        raise ValueError("At least 1 flow file is required")

    flows = []
    for flow_path in flow_path_list:
        depth_and_flow = np.load(flow_path)
        flow = torch.tensor(depth_and_flow.tolist(), dtype=torch.float32)
        flow = resize_flow(flow, (height, width))
        flows.append(flow)
    return torch.stack(flows)


def pil_to_tensor_without_numpy(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    channels = len(img.getbands())
    buffer = torch.ByteStorage.from_buffer(img.tobytes())
    tensor = torch.ByteTensor(buffer)
    tensor = tensor.view(img.size[1], img.size[0], channels).permute(2, 0, 1).float()
    return tensor / 255.0


def load_and_preprocess_images(image_path_list):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    target_size = 518

    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = pil_to_tensor_without_numpy(img)

        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
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
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)
    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)
    return images


def load_json(path, default=None):
    path = Path(path)
    if path.is_file():
        with path.open("r") as f:
            return json.load(f)
    return {} if default is None else default


def read_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def scene_base_from_clip_name(clip_name):
    return clip_name.rsplit("_", 1)[0]


def scene_name_from_base(scene_base):
    return f"segment-{scene_base}_with_camera_labels"


def resolve_default_manifest_path(processed_root, split, views):
    manifest_root = Path(processed_root) / "waymo_edit_cache" / "manifests" / split
    candidates = [
        manifest_root / f"{split}_mode_a_views{views}.jsonl",
    ]
    if views == 3:
        candidates.append(manifest_root / f"{split}_mode_a.jsonl")
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def resolve_default_candidate_path(processed_root, split):
    return Path(processed_root) / "waymo_edit_cache" / "metadata" / split / "mode_a_candidates.jsonl"


def _list_scene_dirs(processed_root, split_name):
    split_root = Path(processed_root) / split_name
    if not split_root.is_dir():
        return []
    return sorted([path for path in split_root.iterdir() if path.is_dir()])


def _list_frame_indices(scene_root, cam_id=0):
    image_root = scene_root / "images"
    frame_indices = []
    for path in sorted(image_root.glob(f"*_{cam_id}.jpg")) + sorted(image_root.glob(f"*_{cam_id}.png")):
        stem = path.stem
        try:
            frame_indices.append(int(stem.split("_")[0]))
        except Exception:
            continue
    return sorted(frame_indices)


def build_clean_clip_records(processed_root, views, split_seed=0, train_ratio=0.9, max_clips_per_scene=6, clip_length=29):
    del views
    all_records = []
    for source_split in ("training", "validation"):
        for scene_dir_path in _list_scene_dirs(processed_root, source_split):
            frame_indices = _list_frame_indices(scene_dir_path, cam_id=0)
            if len(frame_indices) < clip_length:
                continue

            num_clips = min(max_clips_per_scene, len(frame_indices) // clip_length)
            for clip_index in range(num_clips):
                start = clip_index * clip_length
                end = start + clip_length
                clip_frame_indices = frame_indices[start:end]
                if len(clip_frame_indices) != clip_length:
                    continue

                scene_dir = str(scene_dir_path.name)
                scene_base = f"{source_split}_{scene_dir}"
                clip_name = f"{scene_base}_{clip_index}"
                all_records.append(
                    {
                        "record_index": len(all_records),
                        "scene_id": len(all_records),
                        "scene_name": scene_base,
                        "scene_dir": scene_dir,
                        "scene_base": scene_base,
                        "clip_name": clip_name,
                        "clip_index": clip_index,
                        "scene_frame_indices": clip_frame_indices,
                        "source_split": source_split,
                        "edit_mode": "clean",
                        "objects": [],
                    }
                )

    rng = random.Random(int(split_seed))
    rng.shuffle(all_records)

    total = len(all_records)
    if total == 0:
        return [], [], {"clean_total": 0, "edited_total": 0, "train_total": 0, "val_total": 0}

    train_count = int(total * float(train_ratio))
    if total > 1:
        train_count = min(max(train_count, 1), total - 1)
    else:
        train_count = total

    train_records = all_records[:train_count]
    val_records = all_records[train_count:]
    stats = {
        "clean_total": total,
        "edited_total": 0,
        "train_total": len(train_records),
        "val_total": len(val_records),
    }
    return train_records, val_records, stats


def resolve_image_path(root, frame_idx, cam_id):
    for ext in (".jpg", ".png"):
        path = root / f"{frame_idx:03d}_{cam_id}{ext}"
        if path.is_file():
            return str(path)
    return ""


def read_image_size(path):
    with Image.open(path) as img:
        width, height = img.size
    return height, width


def build_intrinsic_matrix(normalized_intrinsics, image_hw):
    image_h, image_w = image_hw
    fx_n, fy_n, cx_n, cy_n = normalized_intrinsics
    fx = fx_n * image_w
    fy = fy_n * image_h
    cx = cx_n * image_w
    cy = cy_n * image_h
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def compute_resize_geometry(image_hw, target_width=518):
    image_h, image_w = image_hw
    new_width = target_width
    new_height = round(image_h * (new_width / image_w) / 14) * 14
    crop_top = max((new_height - target_width) // 2, 0) if new_height > target_width else 0
    out_height = target_width if new_height > target_width else new_height
    return {
        "scale_x": new_width / image_w,
        "scale_y": new_height / image_h,
        "crop_top": crop_top,
        "out_hw": (out_height, new_width),
    }


def transform_box_xyxy(box_xyxy, image_hw, target_width=518):
    box = np.asarray(box_xyxy, dtype=np.float32)
    geom = compute_resize_geometry(image_hw, target_width=target_width)
    scale_x = geom["scale_x"]
    scale_y = geom["scale_y"]
    crop_top = geom["crop_top"]
    out_h, out_w = geom["out_hw"]

    out = box.copy()
    out[[0, 2]] *= scale_x
    out[[1, 3]] = out[[1, 3]] * scale_y - crop_top
    out[[0, 2]] = np.clip(out[[0, 2]], 0, out_w)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, out_h)
    return out.astype(np.float32), geom["out_hw"]


def center_crop_offset(raw_hw, crop_hw):
    raw_h, raw_w = raw_hw
    crop_h, crop_w = crop_hw
    top = max((raw_h - crop_h) // 2, 0)
    left = max((raw_w - crop_w) // 2, 0)
    return top, left


def transfer_box_to_raw_box(box_xyxy, raw_hw, transfer_hw):
    box = np.asarray(box_xyxy, dtype=np.float32)
    raw_h, raw_w = raw_hw
    transfer_h, transfer_w = transfer_hw
    if transfer_h <= raw_h and transfer_w <= raw_w:
        top, left = center_crop_offset(raw_hw, transfer_hw)
        offset = np.array([left, top, left, top], dtype=np.float32)
        return box + offset

    scale_x = raw_w / transfer_w
    scale_y = raw_h / transfer_h
    out = box.copy()
    out[[0, 2]] *= scale_x
    out[[1, 3]] *= scale_y
    return out.astype(np.float32)


def normalize_box_xyxy(box):
    default_box = np.zeros((4,), dtype=np.float32)
    if box is None:
        return default_box, False
    try:
        box_arr = np.asarray(box, dtype=np.float32).reshape(-1)
    except Exception:
        return default_box, False
    if box_arr.shape[0] != 4:
        return default_box, False
    if not np.all(np.isfinite(box_arr)):
        return default_box, False
    return box_arr.astype(np.float32), True


def normalize_box_sequence(box_sequence, expected_length=None):
    if expected_length is None:
        expected_length = len(box_sequence) if box_sequence is not None else 0

    boxes = np.zeros((expected_length, 4), dtype=np.float32)
    valid_mask = np.zeros((expected_length,), dtype=bool)
    if box_sequence is None:
        return boxes, valid_mask

    for frame_idx, box in enumerate(box_sequence[:expected_length]):
        box_arr, is_valid = normalize_box_xyxy(box)
        if is_valid:
            boxes[frame_idx] = box_arr
            valid_mask[frame_idx] = True
    return boxes, valid_mask


def build_box_corners_world(obj_to_world, box_size):
    length, width, height = box_size.tolist()
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
    world = (obj_to_world @ local.T).T[:, :3]
    return world.astype(np.float32)


def compose_waymo_camera_to_world(ego_pose, camera_to_ego):
    ego_pose = np.asarray(ego_pose, dtype=np.float32)
    camera_to_ego = np.asarray(camera_to_ego, dtype=np.float32)
    return (ego_pose @ camera_to_ego @ WAYMO_OPENCV2DATASET).astype(np.float32)


def numpy_like_to_torch(value, dtype=None):
    if isinstance(value, np.ndarray):
        return torch.tensor(value.tolist(), dtype=dtype)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def compute_track_speed_profile(frame_annotations, speed_values=None):
    frame_indices = [int(v) for v in frame_annotations.get("frame_idx", [])]
    obj_to_world = frame_annotations.get("obj_to_world", [])
    num_frames = min(len(frame_indices), len(obj_to_world))
    if num_frames == 0:
        return []
    if speed_values is None or len(speed_values) < num_frames:
        track_id = frame_annotations.get("object_id", "unknown")
        raise ValueError(
            f"Missing speed_mps annotations for track {track_id}: "
            f"need {num_frames}, got {0 if speed_values is None else len(speed_values)}"
        )

    profile = []
    for idx in range(num_frames):
        profile.append(max(0.0, float(speed_values[idx])))
    return profile


def compute_track_speed_stats(frame_annotations, speed_values=None):
    profile = compute_track_speed_profile(frame_annotations, speed_values)
    if len(profile) == 0:
        return 0.0, 0.0
    return float(max(profile)), float(sum(profile) / len(profile))


def tensor_to_uint8_image(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
    tensor = (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return np.array(tensor.tolist(), dtype=np.uint8)


def make_image_grid(images, cols=3, pad_value=255):
    if len(images) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    height, width = images[0].shape[:2]
    canvas = np.full((rows * height, cols * width, 3), pad_value, dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * height
        x0 = col * width
        canvas[y0 : y0 + height, x0 : x0 + width] = image
    return canvas


def draw_box_xyxy(image, box_xyxy, color, label=None, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(x1, image.shape[1] - 1))
    x2 = max(0, min(x2, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    y2 = max(0, min(y2, image.shape[0] - 1))
    if x2 <= x1 or y2 <= y1:
        return image

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def project_world_points_to_model_image(points_world, camera_to_world, intrinsic, raw_hw, target_width=518, eps=1e-6):
    points_world = np.asarray(points_world, dtype=np.float32)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)

    world_to_camera = np.linalg.inv(camera_to_world)
    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    points_cam = (world_to_camera @ points_h.T).T[:, :3]
    points_img = (intrinsic @ points_cam.T).T
    depths = points_img[:, 2]

    valid = np.isfinite(points_img).all(axis=1) & (depths > eps)
    points_2d = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        points_2d[valid] = points_img[valid, :2] / depths[valid, None]
        geom = compute_resize_geometry(tuple(int(v) for v in raw_hw), target_width=target_width)
        points_2d[valid, 0] *= geom["scale_x"]
        points_2d[valid, 1] = points_2d[valid, 1] * geom["scale_y"] - geom["crop_top"]
    return points_2d, valid


def draw_projected_3d_box(image, corners_2d, valid_mask, color, label=None, thickness=2):
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    corners_2d = np.asarray(corners_2d, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.sum() < 2:
        return image

    for start_idx, end_idx in edges:
        if not (valid_mask[start_idx] and valid_mask[end_idx]):
            continue
        pt1 = tuple(int(round(v)) for v in corners_2d[start_idx])
        pt2 = tuple(int(round(v)) for v in corners_2d[end_idx])
        cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)

    if label:
        valid_points = corners_2d[valid_mask]
        anchor = valid_points[np.argmin(valid_points[:, 1] + valid_points[:, 0])]
        anchor_xy = (int(round(anchor[0])), int(round(anchor[1])) - 4)
        cv2.putText(
            image,
            label,
            anchor_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


class WaymoEditDataset(Dataset):
    def __init__(
        self,
        processed_root=DEFAULT_PROCESSED_ROOT,
        transfer_root=DEFAULT_TRANSFER_ROOT,
        raw_root=DEFAULT_RAW_ROOT,
        asset_root=DEFAULT_ASSET_ROOT,
        split="training",
        final_info_path=None,
        manifest_path=None,
        candidate_path=None,
        sequence_length=4,
        mode=1,
        views=1,
        sample_window=20,
        transfer_camera="pinhole_front",
        alignment_hw=None,
        alignment_num_anchors=None,
        alignment_patch_size=None,
        alignment_cache_path=None,
        persist_alignment_cache=False,
        tokenizer=None,
        clean_only=False,
        clean_caption_template="a clean driving scene",
        clean_split_seed=0,
        clean_train_ratio=0.9,
    ):
        super().__init__()
        del transfer_root, transfer_camera, alignment_hw, alignment_num_anchors
        del alignment_patch_size, alignment_cache_path, persist_alignment_cache

        if views not in (1, 3):
            raise ValueError(f"Only views=1 or views=3 are supported, got {views}")
        if mode not in (1, 2):
            raise ValueError(f"Only mode=1 or mode=2 are supported, got {mode}")

        self.processed_root = Path(processed_root)
        self.raw_root = Path(raw_root)
        self.asset_root = Path(asset_root)
        self.split = split
        self.sequence_length = sequence_length
        self.mode = mode
        self.views = views
        self.sample_window = sample_window
        self.camera_ids = [0] if views == 1 else [0, 1, 2]
        self.tokenizer = tokenizer
        self.clean_only = bool(clean_only)
        self.clean_caption_template = str(clean_caption_template)
        self.clean_split_seed = int(clean_split_seed)
        self.clean_train_ratio = float(clean_train_ratio)

        self.clean_sample_stats = {
            "clean_total": 0,
            "edited_total": 0,
            "train_total": 0,
            "val_total": 0,
        }
        self.dataset_split_name = "train" if split in ("training", "train") else "val"

        if self.clean_only:
            train_records, val_records, clean_stats = build_clean_clip_records(
                self.processed_root,
                self.views,
                split_seed=self.clean_split_seed,
                train_ratio=self.clean_train_ratio,
            )
            self.clean_sample_stats = clean_stats
            self.samples = train_records if self.dataset_split_name == "train" else val_records
            self.max_objects = 0
            self.processed_split_root = None
            self.annotations_root = None
            self._scene_cache = {}
            return

        if manifest_path is None and final_info_path is not None:
            final_info_path = Path(final_info_path)
            raise ValueError(
                "WaymoEditDataset now reads manifest jsonl, not final_info.json. "
                f"Use manifest_path instead of final_info_path: got {final_info_path}"
            )
        if manifest_path is None:
            manifest_path = resolve_default_manifest_path(self.processed_root, split, views)
        if candidate_path is None:
            candidate_path = resolve_default_candidate_path(self.processed_root, split)

        self.manifest_path = Path(manifest_path)
        self.candidate_path = Path(candidate_path)

        manifest_entries = read_jsonl(self.manifest_path)
        candidate_records = read_jsonl(self.candidate_path)
        if len(candidate_records) == 0:
            raise FileNotFoundError(
                f"Candidate metadata not found at {self.candidate_path}. "
                "Run datasets/tools/build_edit_metadata.py first."
            )

        candidate_by_record_index = {
            int(record["record_index"]): record for record in candidate_records
        }
        self.samples = []
        for entry in manifest_entries:
            record_index = int(entry["record_index"])
            record = candidate_by_record_index.get(record_index)
            if record is not None:
                self.samples.append(record)
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.manifest_path}")

        self.max_objects = max((len(sample.get("objects", [])) for sample in self.samples), default=0)
        self.processed_split_root = self.processed_root / split
        if not self.processed_split_root.is_dir():
            raise FileNotFoundError(f"Processed split root not found: {self.processed_split_root}")

        self.annotations_root = self.processed_root / "waymo_edit_cache" / "annotations" / split
        self._scene_cache = {}

    def _get_scene_cache(self, record):
        scene_dir = str(record["scene_dir"])
        if scene_dir in self._scene_cache:
            return self._scene_cache[scene_dir]

        scene_root = self.processed_split_root / scene_dir
        annotation_path = self.annotations_root / f"{record['scene_name']}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation json not found: {annotation_path}")

        annotation = load_json(annotation_path, default={})
        instances_info = load_json(scene_root / "instances" / "instances_info.json", default={})

        raw_to_instance = {}
        for contig_id, instance_info in instances_info.items():
            raw_object_id = str(instance_info.get("raw_object_id", instance_info.get("id", "")))
            raw_to_instance[raw_object_id] = {
                "contig_instance_id": int(contig_id),
                "instance_info": instance_info,
            }

        image_size_by_cam = {}
        for cam_id in range(5):
            img_path = resolve_image_path(scene_root / "images", 0, cam_id)
            if img_path:
                image_size_by_cam[cam_id] = read_image_size(img_path)

        scene_cache = {
            "scene_root": scene_root,
            "annotation": annotation,
            "instances_info": instances_info,
            "raw_to_instance": raw_to_instance,
            "image_size_by_cam": image_size_by_cam,
        }
        self._scene_cache[scene_dir] = scene_cache
        return scene_cache

    def _pick_candidate(self, candidates):
        if len(candidates) == 0:
            return None
        if self.mode == 1:
            return random.choice(candidates)
        return sorted(
            candidates,
            key=lambda item: (-item["num_visible_total"], item["start_rel"], item["num_non_visible_required"]),
        )[0]

    def _pick_indices(self, indices, count):
        indices = list(sorted(indices))
        if count <= 0:
            return []
        if len(indices) < count:
            raise ValueError(f"Cannot pick {count} indices from only {len(indices)} candidates")
        if self.mode == 1:
            return sorted(random.sample(indices, count))
        return indices[:count]

    def _sample_local_indices(
        self,
        segment_start,
        segment_end,
        frame_has_preferred_editable_object,
        frame_has_editable_object,
        sample_num_frames=None,
    ):
        sample_num_frames = self.sequence_length if sample_num_frames is None else int(sample_num_frames)
        sample_window = max(int(self.sample_window), sample_num_frames)
        segment_length = segment_end - segment_start
        if segment_length < sample_num_frames:
            raise ValueError(
                f"segment length {segment_length} is smaller than requested num_frames {sample_num_frames}"
            )

        segment_visible_editable = frame_has_preferred_editable_object[segment_start:segment_end]
        segment_editable = frame_has_editable_object[segment_start:segment_end]
        if isinstance(segment_visible_editable, torch.Tensor):
            segment_visible_editable = segment_visible_editable.tolist()
        if isinstance(segment_editable, torch.Tensor):
            segment_editable = segment_editable.tolist()

        all_start_candidates = list(range(0, segment_length - sample_num_frames + 1))
        if segment_length >= sample_window:
            preferred_start_candidates = list(range(0, segment_length - sample_window + 1))
        else:
            preferred_start_candidates = all_start_candidates

        def build_candidate(start_rel, visible_flags):
            window_end = min(segment_length, start_rel + sample_window)
            future_indices = list(range(start_rel + 1, window_end))
            if len(future_indices) < sample_num_frames - 1:
                return None
            visible_future = [idx for idx in future_indices if visible_flags[idx]]
            start_visible = bool(visible_flags[start_rel])
            num_visible_total = len(visible_future) + int(start_visible)
            num_non_visible_required = max(0, sample_num_frames - num_visible_total)
            return {
                "start_rel": start_rel,
                "future_indices": future_indices,
                "visible_future": visible_future,
                "start_visible": start_visible,
                "num_visible_total": num_visible_total,
                "num_non_visible_required": num_non_visible_required,
            }

        preferred_all_visible = []
        preferred_mixed_start_visible = []
        preferred_mixed_any = []
        fallback_all_visible = []
        fallback_mixed_start_visible = []
        fallback_mixed_any = []

        for start_rel in preferred_start_candidates:
            candidate = build_candidate(start_rel, segment_visible_editable)
            if candidate is None:
                continue
            if candidate["start_visible"] and len(candidate["visible_future"]) >= sample_num_frames - 1:
                preferred_all_visible.append(candidate)
            elif candidate["start_visible"]:
                preferred_mixed_start_visible.append(candidate)
            else:
                preferred_mixed_any.append(candidate)

        for start_rel in all_start_candidates:
            candidate = build_candidate(start_rel, segment_editable)
            if candidate is None:
                continue
            if candidate["start_visible"] and len(candidate["visible_future"]) >= sample_num_frames - 1:
                fallback_all_visible.append(candidate)
            elif candidate["start_visible"]:
                fallback_mixed_start_visible.append(candidate)
            else:
                fallback_mixed_any.append(candidate)

        candidate = (
            self._pick_candidate(preferred_all_visible)
            or self._pick_candidate(preferred_mixed_start_visible)
            or self._pick_candidate(preferred_mixed_any)
            or self._pick_candidate(fallback_all_visible)
            or self._pick_candidate(fallback_mixed_start_visible)
            or self._pick_candidate(fallback_mixed_any)
        )
        if candidate is None:
            raise RuntimeError(
                f"Failed to sample frames inside segment [{segment_start}, {segment_end}) with length {segment_length}"
            )

        num_required_future = sample_num_frames - 1
        chosen_future = self._pick_indices(
            candidate["visible_future"],
            min(num_required_future, len(candidate["visible_future"])),
        )
        num_remaining = num_required_future - len(chosen_future)
        if num_remaining > 0:
            remaining_future_pool = [idx for idx in candidate["future_indices"] if idx not in chosen_future]
            chosen_future.extend(self._pick_indices(remaining_future_pool, num_remaining))
        chosen_future = sorted(chosen_future)

        selected_segment_indices = [candidate["start_rel"]] + chosen_future
        intervals = [idx - candidate["start_rel"] for idx in chosen_future]
        return selected_segment_indices, intervals

    def _parse_sample_index(self, idx):
        sample_num_frames = self.sequence_length
        tuple_override = False

        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise ValueError(f"WaymoEditDataset tuple index must be (idx, num_frames), got {idx}")
            idx, sample_num_frames = idx
            tuple_override = True

        sample_index = int(idx)
        sample_num_frames = int(sample_num_frames)
        if tuple_override and not 4 <= sample_num_frames <= 8:
            raise ValueError(f"WaymoEditDataset tuple num_frames must be in [4, 8], got {sample_num_frames}")
        if sample_num_frames <= 0:
            raise ValueError(f"WaymoEditDataset num_frames must be positive, got {sample_num_frames}")
        return sample_index, sample_num_frames

    def _coerce_bool_sequence(self, values, expected_length):
        if values is None:
            return None
        values = list(values)
        if len(values) < expected_length:
            values = values + [False] * (expected_length - len(values))
        else:
            values = values[:expected_length]
        return torch.tensor([bool(v) for v in values], dtype=torch.bool)

    def _build_lightweight_sampling_flags(self, record):
        clip_len = len(record["scene_frame_indices"])
        frame_has_visible_editable_object = torch.zeros((clip_len,), dtype=torch.bool)

        for object_record in record.get("objects", []):
            asset_object_id = str(object_record.get("asset_object_id", ""))
            default_asset_path = self.asset_root / f"{asset_object_id}.ply" if asset_object_id else self.asset_root
            asset_path = Path(object_record.get("asset_path", default_asset_path))
            if not asset_path.is_file():
                continue

            object_visible = torch.zeros((clip_len,), dtype=torch.bool)
            visibility_by_view = object_record.get("visibility_by_view", {})
            for cam_id in self.camera_ids:
                cam_name = CAM_ID_TO_NAME[cam_id]
                view_flags = self._coerce_bool_sequence(visibility_by_view.get(cam_name), clip_len)
                if view_flags is not None:
                    object_visible |= view_flags
            frame_has_visible_editable_object |= object_visible

        if not bool(frame_has_visible_editable_object.any().item()):
            top_level_key = "frame_has_front_visible_object" if self.views == 1 else "frame_has_front3_visible_object"
            top_level_flags = self._coerce_bool_sequence(record.get(top_level_key), clip_len)
            if top_level_flags is not None:
                frame_has_visible_editable_object = top_level_flags

        frame_has_editable_object = frame_has_visible_editable_object.clone()
        if not bool(frame_has_editable_object.any().item()):
            frame_has_editable_object = torch.ones((clip_len,), dtype=torch.bool)
        return frame_has_visible_editable_object, frame_has_editable_object

    def _build_normalized_timestamps(self, local_indices):
        timestamps = np.array(local_indices, dtype=np.float32)
        if len(timestamps) == 0:
            return timestamps
        timestamps = timestamps - float(timestamps[0])
        if len(timestamps) > 1 and float(timestamps[-1]) > 0:
            timestamps = timestamps / float(timestamps[-1])
        else:
            timestamps = np.zeros_like(timestamps, dtype=np.float32)
        if len(self.camera_ids) > 1:
            timestamps = np.repeat(timestamps, len(self.camera_ids))
        return timestamps.astype(np.float32, copy=False)

    def _build_base_sample(
        self,
        record,
        sample_index,
        sample_num_frames,
        clip_frame_indices,
        local_indices,
        scene_frame_indices,
        image_paths,
        images,
        timestamps,
        intervals,
    ):
        return {
            "sample_index": int(sample_index),
            "num_frames": torch.tensor(int(sample_num_frames), dtype=torch.long),
            "images": images,
            "images_clean": images,
            "image_paths": image_paths,
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
            "interval": torch.tensor(intervals, dtype=torch.long),
            "scene_id": int(record["scene_id"]),
            "scene_name": str(record["scene_name"]),
            "scene_dir": str(record["scene_dir"]),
            "scene_base": str(record["scene_base"]),
            "clip_name": str(record["clip_name"]),
            "clip_index": torch.tensor(int(record["clip_index"]), dtype=torch.long),
            "cam_ids": torch.tensor(self.camera_ids, dtype=torch.long),
            "frame_indices": torch.tensor(scene_frame_indices, dtype=torch.long),
            "local_frame_indices": torch.tensor(local_indices, dtype=torch.long),
            "clip_frame_indices": torch.tensor(clip_frame_indices, dtype=torch.long),
            "edit_mode": str(record.get("edit_mode", "replace")),
            "edit_spec": {
                "clip_name": str(record["clip_name"]),
                "scene_base": str(record["scene_base"]),
                "clip_index": int(record["clip_index"]),
                "scene_frame_indices": clip_frame_indices,
                "selected_scene_frame_indices": scene_frame_indices,
                "selected_local_indices": local_indices,
                "sample_num_frames": int(sample_num_frames),
            },
        }

    def _build_clean_caption(self, record, sample_num_frames):
        template_context: dict[str, Any] = {
            "scene_name": str(record["scene_name"]),
            "scene_base": str(record["scene_base"]),
            "clip_name": str(record["clip_name"]),
            "clip_index": int(record["clip_index"]),
            "edit_mode": str(record.get("edit_mode", "replace")),
            "num_frames": int(sample_num_frames),
            "num_views": int(len(self.camera_ids)),
        }
        try:
            return self.clean_caption_template.format(**template_context)
        except Exception:
            return self.clean_caption_template

    def _build_tokenizer_payload(self, record, sample_num_frames, images):
        if self.tokenizer is None:
            return {}

        caption = self._build_clean_caption(record, sample_num_frames)
        pixel_values = images.mul(2.0).sub(1.0)
        payload = {
            "caption": caption,
            "conditioning_pixel_values": pixel_values,
            "output_pixel_values": pixel_values,
        }
        tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if model_max_length is not None:
            tokenizer_kwargs["max_length"] = model_max_length
        encoded = self.tokenizer(caption, **tokenizer_kwargs)

        if hasattr(encoded, "input_ids"):
            payload["input_ids"] = encoded.input_ids
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            payload["input_ids"] = encoded["input_ids"]

        if hasattr(encoded, "attention_mask"):
            payload["attention_mask"] = encoded.attention_mask
        elif isinstance(encoded, dict) and "attention_mask" in encoded:
            payload["attention_mask"] = encoded["attention_mask"]
        return payload

    def _resolve_mask_root(self, scene_root):
        candidates = [
            scene_root / "fine_dynamic_masks" / "vehicle",
            scene_root / "fine_dynamic_masks" / "all",
            scene_root / "dynamic_masks" / "vehicle",
            scene_root / "dynamic_masks",
        ]
        for root in candidates:
            if root.is_dir():
                return root
        return None

    def _load_optional_image_stack(self, path_list, like_tensor):
        valid = [path for path in path_list if path and Path(path).is_file()]
        if len(valid) == len(path_list) and len(valid) > 0:
            return load_and_preprocess_images(path_list)
        raise ValueError(f"Failed to load images from {path_list}")

    def _load_optional_depth_stack(self, path_list, like_tensor):
        valid = [path for path in path_list if path and Path(path).is_file()]
        if len(valid) == len(path_list) and len(valid) > 0:
            return load_and_preprocess_flow(path_list, like_tensor.shape[2], like_tensor.shape[3])
        return torch.zeros(
            (len(path_list), like_tensor.shape[2], like_tensor.shape[3]),
            dtype=like_tensor.dtype,
        )

    def _prepare_object_data(self, record, scene_cache):
        clip_frame_indices = [int(v) for v in record["scene_frame_indices"]]
        clip_len = len(clip_frame_indices)
        num_views = len(self.camera_ids)
        front_view_offset = self.camera_ids.index(0) if 0 in self.camera_ids else None

        object_valid_mask = torch.zeros((self.max_objects,), dtype=torch.bool)
        object_track_valid_mask = torch.zeros((self.max_objects, clip_len), dtype=torch.bool)
        object_bbox_valid_mask = torch.zeros((self.max_objects, clip_len, num_views), dtype=torch.bool)
        object_contig_ids = torch.full((self.max_objects,), -1, dtype=torch.long)
        object_scene_match_scores = torch.zeros((self.max_objects,), dtype=torch.float32)
        object_track_range = torch.full((self.max_objects, 2), -1, dtype=torch.long)
        object_track_frame_count = torch.zeros((self.max_objects,), dtype=torch.long)
        object_max_speed_mps = torch.zeros((self.max_objects,), dtype=torch.float32)
        object_mean_speed_mps = torch.zeros((self.max_objects,), dtype=torch.float32)
        object_is_moving_track = torch.zeros((self.max_objects,), dtype=torch.bool)
        object_asset_valid_mask = torch.zeros((self.max_objects,), dtype=torch.bool)

        object_bbox_transfer = torch.zeros((self.max_objects, clip_len, num_views, 4), dtype=torch.float32)
        object_bbox_raw = torch.zeros((self.max_objects, clip_len, num_views, 4), dtype=torch.float32)
        object_bbox_model = torch.zeros((self.max_objects, clip_len, num_views, 4), dtype=torch.float32)
        object_bbox_patch = torch.zeros((self.max_objects, clip_len, num_views, 4), dtype=torch.float32)

        object_obj_to_world = torch.zeros((self.max_objects, clip_len, 4, 4), dtype=torch.float32)
        object_box_size = torch.zeros((self.max_objects, clip_len, 3), dtype=torch.float32)
        object_box_corners_world = torch.zeros((self.max_objects, clip_len, 8, 3), dtype=torch.float32)
        object_centers_world = torch.zeros((self.max_objects, clip_len, 3), dtype=torch.float32)
        object_speed_mps = torch.zeros((self.max_objects, clip_len), dtype=torch.float32)
        object_is_moving_frame = torch.zeros((self.max_objects, clip_len), dtype=torch.bool)

        object_asset_ids = [""] * self.max_objects
        object_scene_raw_ids = [""] * self.max_objects
        object_class_names = [""] * self.max_objects
        object_asset_paths = [""] * self.max_objects

        for object_slot, object_record in enumerate(record.get("objects", [])):
            object_valid_mask[object_slot] = True

            asset_object_id = str(object_record["asset_object_id"])
            scene_raw_id = str(object_record.get("scene_raw_object_id", asset_object_id))
            asset_path = Path(object_record.get("asset_path", self.asset_root / f"{asset_object_id}.ply"))
            match_score = float(object_record.get("match_score", 1.0))

            object_asset_ids[object_slot] = asset_object_id
            object_scene_raw_ids[object_slot] = scene_raw_id
            object_asset_paths[object_slot] = str(asset_path)
            object_scene_match_scores[object_slot] = match_score
            object_asset_valid_mask[object_slot] = asset_path.is_file()

            instance_entry = scene_cache["raw_to_instance"].get(scene_raw_id)
            if instance_entry is None:
                continue

            contig_instance_id = int(instance_entry["contig_instance_id"])
            instance_info = instance_entry["instance_info"]
            frame_annotations = instance_info["frame_annotations"]
            track_frame_indices = [int(v) for v in frame_annotations["frame_idx"]]
            track_lookup = {frame_idx: idx for idx, frame_idx in enumerate(track_frame_indices)}
            track_speed_profile = compute_track_speed_profile(
                frame_annotations,
                frame_annotations.get("speed_mps"),
            )

            object_contig_ids[object_slot] = contig_instance_id
            object_class_names[object_slot] = str(instance_info.get("class_name", ""))
            object_track_range[object_slot, 0] = int(min(track_frame_indices)) if track_frame_indices else -1
            object_track_range[object_slot, 1] = int(max(track_frame_indices)) if track_frame_indices else -1
            object_track_frame_count[object_slot] = len(track_frame_indices)

            max_speed_mps = instance_info.get("max_speed_mps")
            mean_speed_mps = instance_info.get("mean_speed_mps")
            if max_speed_mps is None or mean_speed_mps is None:
                max_speed_mps, mean_speed_mps = compute_track_speed_stats(
                    frame_annotations,
                    frame_annotations.get("speed_mps"),
                )
            object_max_speed_mps[object_slot] = float(max_speed_mps)
            object_mean_speed_mps[object_slot] = float(mean_speed_mps)
            object_is_moving_track[object_slot] = bool(
                instance_info.get(
                    "is_moving_track",
                    float(max_speed_mps) > WAYMO_DYNAMIC_SPEED_THRESH_MPS,
                )
            )

            view_sequences = {}
            transfer_boxes_by_view = object_record.get("boxes_by_view_transfer", object_record.get("boxes_by_view", {}))
            raw_boxes_by_view = object_record.get("boxes_by_view_raw", {})
            model_boxes_by_view = object_record.get("boxes_by_view_model", {})
            if len(raw_boxes_by_view) == 0 or len(model_boxes_by_view) == 0:
                raise KeyError(
                    "Metadata record is missing boxes_by_view_raw/model. "
                    "Regenerate metadata with datasets/tools/build_edit_metadata.py."
                )
            for view_offset, cam_id in enumerate(self.camera_ids):
                cam_name = CAM_ID_TO_NAME[cam_id]
                transfer_boxes, transfer_valid_mask = normalize_box_sequence(
                    transfer_boxes_by_view.get(cam_name),
                    expected_length=clip_len,
                )
                raw_boxes, raw_valid_mask = normalize_box_sequence(
                    raw_boxes_by_view.get(cam_name),
                    expected_length=clip_len,
                )
                model_boxes, model_valid_mask = normalize_box_sequence(
                    model_boxes_by_view.get(cam_name),
                    expected_length=clip_len,
                )
                view_sequences[view_offset] = {
                    "cam_id": cam_id,
                    "transfer_boxes": transfer_boxes,
                    "transfer_valid_mask": transfer_valid_mask,
                    "raw_boxes": raw_boxes,
                    "raw_valid_mask": raw_valid_mask,
                    "model_boxes": model_boxes,
                    "model_valid_mask": model_valid_mask,
                }

            for local_idx, scene_frame_idx in enumerate(clip_frame_indices):
                for view_offset, view_data in view_sequences.items():
                    transfer_valid = bool(view_data["transfer_valid_mask"][local_idx])
                    raw_valid = bool(view_data["raw_valid_mask"][local_idx])
                    model_valid = bool(view_data["model_valid_mask"][local_idx])
                    if not transfer_valid:
                        continue
                    transfer_box = view_data["transfer_boxes"][local_idx]
                    if not (raw_valid and model_valid):
                        raise ValueError(
                            f"Metadata boxes are invalid for object_slot={object_slot}, frame={local_idx}, cam={view_data['cam_id']}. "
                            "Regenerate metadata with datasets/tools/build_edit_metadata.py."
                        )
                    raw_box = view_data["raw_boxes"][local_idx]
                    model_box = view_data["model_boxes"][local_idx]

                    object_bbox_valid_mask[object_slot, local_idx, view_offset] = True
                    object_bbox_transfer[object_slot, local_idx, view_offset] = numpy_like_to_torch(
                        transfer_box,
                        dtype=torch.float32,
                    )
                    object_bbox_raw[object_slot, local_idx, view_offset] = numpy_like_to_torch(
                        raw_box,
                        dtype=torch.float32,
                    )
                    object_bbox_model[object_slot, local_idx, view_offset] = numpy_like_to_torch(
                        model_box,
                        dtype=torch.float32,
                    )
                    object_bbox_patch[object_slot, local_idx, view_offset] = numpy_like_to_torch(
                        model_box / 14.0,
                        dtype=torch.float32,
                    )

                if scene_frame_idx not in track_lookup:
                    continue

                track_idx = track_lookup[scene_frame_idx]
                obj_to_world = np.asarray(frame_annotations["obj_to_world"][track_idx], dtype=np.float32)
                box_size = np.asarray(frame_annotations["box_size"][track_idx], dtype=np.float32)
                box_corners_world = build_box_corners_world(obj_to_world, box_size)

                object_track_valid_mask[object_slot, local_idx] = True
                object_obj_to_world[object_slot, local_idx] = numpy_like_to_torch(
                    obj_to_world,
                    dtype=torch.float32,
                )
                object_box_size[object_slot, local_idx] = numpy_like_to_torch(
                    box_size,
                    dtype=torch.float32,
                )
                object_box_corners_world[object_slot, local_idx] = numpy_like_to_torch(
                    box_corners_world,
                    dtype=torch.float32,
                )
                object_centers_world[object_slot, local_idx] = numpy_like_to_torch(
                    obj_to_world[:3, 3],
                    dtype=torch.float32,
                )

                if track_idx < len(track_speed_profile):
                    frame_speed = float(track_speed_profile[track_idx])
                    object_speed_mps[object_slot, local_idx] = frame_speed
                    object_is_moving_frame[object_slot, local_idx] = bool(
                        frame_speed > WAYMO_DYNAMIC_SPEED_THRESH_MPS
                    )

        if front_view_offset is not None:
            object_front_bbox_valid_mask = object_bbox_valid_mask[:, :, front_view_offset]
            object_front_bbox_transfer = object_bbox_transfer[:, :, front_view_offset]
            object_front_bbox_raw = object_bbox_raw[:, :, front_view_offset]
            object_front_bbox_model = object_bbox_model[:, :, front_view_offset]
            object_front_bbox_patch = object_bbox_patch[:, :, front_view_offset]
        else:
            object_front_bbox_valid_mask = torch.zeros((self.max_objects, clip_len), dtype=torch.bool)
            object_front_bbox_transfer = torch.zeros((self.max_objects, clip_len, 4), dtype=torch.float32)
            object_front_bbox_raw = torch.zeros((self.max_objects, clip_len, 4), dtype=torch.float32)
            object_front_bbox_model = torch.zeros((self.max_objects, clip_len, 4), dtype=torch.float32)
            object_front_bbox_patch = torch.zeros((self.max_objects, clip_len, 4), dtype=torch.float32)

        object_visible_editable_mask = (
            object_track_valid_mask[:, :, None]
            & object_bbox_valid_mask
            & object_asset_valid_mask[:, None, None]
        )
        object_projected_editable_mask = object_visible_editable_mask.clone()
        frame_has_editable_object = (
            object_track_valid_mask
            & object_asset_valid_mask[:, None]
        ).any(dim=0)
        frame_has_visible_editable_object = object_visible_editable_mask.any(dim=(0, 2))
        frame_has_projected_editable_object = object_projected_editable_mask.any(dim=(0, 2))
        frame_has_front_visible_editable_object = (
            object_track_valid_mask
            & object_front_bbox_valid_mask
            & object_asset_valid_mask[:, None]
        ).any(dim=0)

        reference_cam_id = 0 if 0 in scene_cache["image_size_by_cam"] else self.camera_ids[0]
        model_h, model_w = compute_resize_geometry(scene_cache["image_size_by_cam"][reference_cam_id])["out_hw"]

        return {
            "object_valid_mask": object_valid_mask,
            "object_track_valid_mask": object_track_valid_mask,
            "object_bbox_valid_mask": object_bbox_valid_mask,
            "object_contig_ids": object_contig_ids,
            "object_scene_match_scores": object_scene_match_scores,
            "object_track_range": object_track_range,
            "object_track_frame_count": object_track_frame_count,
            "object_max_speed_mps": object_max_speed_mps,
            "object_mean_speed_mps": object_mean_speed_mps,
            "object_is_moving_track": object_is_moving_track,
            "object_asset_valid_mask": object_asset_valid_mask,
            "object_bbox_transfer": object_bbox_transfer,
            "object_bbox_raw": object_bbox_raw,
            "object_bbox_model": object_bbox_model,
            "object_bbox_patch": object_bbox_patch,
            "object_front_bbox_valid_mask": object_front_bbox_valid_mask,
            "object_front_bbox_transfer": object_front_bbox_transfer,
            "object_front_bbox_raw": object_front_bbox_raw,
            "object_front_bbox_model": object_front_bbox_model,
            "object_front_bbox_patch": object_front_bbox_patch,
            "object_obj_to_world": object_obj_to_world,
            "object_box_size": object_box_size,
            "object_box_corners_world": object_box_corners_world,
            "object_centers_world": object_centers_world,
            "object_speed_mps": object_speed_mps,
            "object_is_moving_frame": object_is_moving_frame,
            "object_asset_ids": object_asset_ids,
            "object_scene_raw_ids": object_scene_raw_ids,
            "object_class_names": object_class_names,
            "object_asset_paths": object_asset_paths,
            "object_visible_editable_mask": object_visible_editable_mask,
            "object_projected_editable_mask": object_projected_editable_mask,
            "frame_has_editable_object": frame_has_editable_object,
            "frame_has_visible_editable_object": frame_has_visible_editable_object,
            "frame_has_front_visible_editable_object": frame_has_front_visible_editable_object,
            "frame_has_projected_editable_object": frame_has_projected_editable_object,
            "front_model_image_hw": torch.tensor([model_h, model_w], dtype=torch.long),
        }

    def _select_object_tensors(self, object_data, selected_indices):
        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        object_tensors = {}

        for key in [
            "object_valid_mask",
            "object_contig_ids",
            "object_scene_match_scores",
            "object_track_range",
            "object_track_frame_count",
            "object_max_speed_mps",
            "object_mean_speed_mps",
            "object_is_moving_track",
            "object_asset_valid_mask",
            "object_asset_ids",
            "object_scene_raw_ids",
            "object_class_names",
            "object_asset_paths",
            "front_model_image_hw",
        ]:
            object_tensors[key] = object_data[key]

        for key in [
            "object_track_valid_mask",
            "object_bbox_valid_mask",
            "object_bbox_transfer",
            "object_bbox_raw",
            "object_bbox_model",
            "object_bbox_patch",
            "object_front_bbox_valid_mask",
            "object_front_bbox_transfer",
            "object_front_bbox_raw",
            "object_front_bbox_model",
            "object_front_bbox_patch",
            "object_obj_to_world",
            "object_box_size",
            "object_box_corners_world",
            "object_centers_world",
            "object_speed_mps",
            "object_is_moving_frame",
            "object_visible_editable_mask",
            "object_projected_editable_mask",
        ]:
            object_tensors[f"{key}_selected"] = object_data[key].index_select(1, selected_indices)

        for key in [
            "frame_has_editable_object",
            "frame_has_visible_editable_object",
            "frame_has_front_visible_editable_object",
            "frame_has_projected_editable_object",
        ]:
            object_tensors[f"{key}_selected"] = object_data[key].index_select(0, selected_indices)

        return object_tensors

    def _build_asset_metadata(self, object_tensors):
        editable_object_mask = object_tensors["object_visible_editable_mask_selected"].any(dim=(1, 2))
        editable_indices = torch.nonzero(editable_object_mask, as_tuple=False).flatten()
        editable_count = int(editable_indices.numel())

        editable_object_indices = torch.full((self.max_objects,), -1, dtype=torch.long)
        if editable_count > 0:
            editable_object_indices[:editable_count] = editable_indices

        editable_slots_list = [int(idx) for idx in editable_indices.tolist()]
        editable_asset_object_ids = [object_tensors["object_asset_ids"][idx] for idx in editable_slots_list]
        editable_scene_raw_object_ids = [object_tensors["object_scene_raw_ids"][idx] for idx in editable_slots_list]
        editable_asset_paths = [object_tensors["object_asset_paths"][idx] for idx in editable_slots_list]
        editable_scene_match_scores = [
            float(object_tensors["object_scene_match_scores"][idx].item()) for idx in editable_slots_list
        ]

        edit_object_slot = int(editable_slots_list[0]) if editable_count > 0 else -1
        selected_asset_path = editable_asset_paths[0] if editable_count > 0 else ""
        return {
            "editable_object_indices": editable_object_indices,
            "editable_object_count": torch.tensor(editable_count, dtype=torch.long),
            "edit_object_slot": torch.tensor(edit_object_slot, dtype=torch.long),
            "asset_meta": {
                "asset_root": str(self.asset_root),
                "editable_object_slots": editable_slots_list,
                "editable_asset_object_ids": editable_asset_object_ids,
                "editable_scene_raw_object_ids": editable_scene_raw_object_ids,
                "editable_asset_paths": editable_asset_paths,
                "editable_scene_match_scores": editable_scene_match_scores,
                "selected_asset_path": selected_asset_path,
            },
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_index, sample_num_frames = self._parse_sample_index(idx)
        record = self.samples[sample_index]
        clip_frame_indices = [int(v) for v in record["scene_frame_indices"]]

        if self.clean_only:
            frame_has_visible_editable_object, frame_has_editable_object = self._build_lightweight_sampling_flags(record)
            local_indices, intervals = self._sample_local_indices(
                0,
                len(clip_frame_indices),
                frame_has_visible_editable_object,
                frame_has_editable_object,
                sample_num_frames=sample_num_frames,
            )
            scene_frame_indices = [clip_frame_indices[local_idx] for local_idx in local_indices]

            source_split = str(record.get("source_split", self.split))
            scene_root = self.processed_root / source_split / str(record["scene_dir"])
            image_paths = []
            sky_mask_paths = []
            for frame_idx in scene_frame_indices:
                for cam_id in self.camera_ids:
                    image_paths.append(resolve_image_path(scene_root / "images", frame_idx, cam_id))
                    sky_mask_paths.append(resolve_image_path(scene_root / "sky_masks", frame_idx, cam_id))
            images = load_and_preprocess_images(image_paths)
            sky_masks = self._load_optional_image_stack(sky_mask_paths, images)
            sample = self._build_base_sample(
                record=record,
                sample_index=sample_index,
                sample_num_frames=sample_num_frames,
                clip_frame_indices=clip_frame_indices,
                local_indices=local_indices,
                scene_frame_indices=scene_frame_indices,
                image_paths=image_paths,
                images=images,
                timestamps=self._build_normalized_timestamps(local_indices),
                intervals=intervals,
            )
            sample["masks"] = sky_masks
            sample["sky_mask"] = sky_masks
            sample.update(self._build_tokenizer_payload(record, sample_num_frames, images))
            return sample

        scene_cache = self._get_scene_cache(record)

        object_data = self._prepare_object_data(record, scene_cache)
        local_indices, intervals = self._sample_local_indices(
            0,
            len(clip_frame_indices),
            object_data["frame_has_visible_editable_object"],
            object_data["frame_has_editable_object"],
            sample_num_frames=sample_num_frames,
        )
        scene_frame_indices = [clip_frame_indices[local_idx] for local_idx in local_indices]

        scene_root = scene_cache["scene_root"]
        image_paths = []
        sky_mask_paths = []
        dynamic_mask_paths = []
        depth_flow_paths = []
        dynamic_root = self._resolve_mask_root(scene_root)
        for frame_idx in scene_frame_indices:
            for cam_id in self.camera_ids:
                image_paths.append(resolve_image_path(scene_root / "images", frame_idx, cam_id))
                sky_mask_paths.append(resolve_image_path(scene_root / "sky_masks", frame_idx, cam_id))
                dynamic_mask_paths.append(
                    resolve_image_path(dynamic_root, frame_idx, cam_id) if dynamic_root is not None else ""
                )
                depth_path = scene_root / "depth_flows_4" / f"{frame_idx:03d}_{cam_id}.npy"
                depth_flow_paths.append(str(depth_path) if depth_path.is_file() else "")

        images = load_and_preprocess_images(image_paths)
        sky_masks = self._load_optional_image_stack(sky_mask_paths, images)
        dynamic_masks = self._load_optional_image_stack(dynamic_mask_paths, images)
        gt_depth = self._load_optional_depth_stack(depth_flow_paths, images)

        timestamps = self._build_normalized_timestamps(local_indices)

        annotation = scene_cache["annotation"]
        camera_to_world = []
        camera_to_world_corrected = []
        intrinsics = []
        raw_image_size_hw = []
        ego_pose_selected = np.asarray(annotation["ego_pose"], dtype=np.float32)[scene_frame_indices]
        for cam_id in self.camera_ids:
            cam_to_world_full = np.asarray(annotation["camera_to_world"][str(cam_id)], dtype=np.float32)
            camera_to_world.append(cam_to_world_full[scene_frame_indices])
            cam_to_ego = np.asarray(annotation["camera_to_ego"][str(cam_id)], dtype=np.float32)
            corrected = np.stack(
                [compose_waymo_camera_to_world(ego_pose, cam_to_ego) for ego_pose in ego_pose_selected],
                axis=0,
            )
            camera_to_world_corrected.append(corrected)
            cam_hw = scene_cache["image_size_by_cam"][cam_id]
            raw_image_size_hw.append(cam_hw)
            intrinsics.append(
                build_intrinsic_matrix(
                    annotation["normalized_intrinsics"][str(cam_id)],
                    cam_hw,
                )
            )

        camera_to_world = numpy_like_to_torch(np.stack(camera_to_world, axis=1), dtype=torch.float32)
        camera_to_world_corrected = numpy_like_to_torch(
            np.stack(camera_to_world_corrected, axis=1),
            dtype=torch.float32,
        )
        intrinsics = numpy_like_to_torch(np.stack(intrinsics, axis=0), dtype=torch.float32)
        camera_to_ego = numpy_like_to_torch(
            np.stack(
                [np.asarray(annotation["camera_to_ego"][str(cam_id)], dtype=np.float32) for cam_id in self.camera_ids],
                axis=0,
            ),
            dtype=torch.float32,
        )
        ego_pose = numpy_like_to_torch(
            np.asarray(annotation["ego_pose"], dtype=np.float32)[scene_frame_indices],
            dtype=torch.float32,
        )

        object_tensors = self._select_object_tensors(object_data, local_indices)
        asset_payload = self._build_asset_metadata(object_tensors)

        sample = self._build_base_sample(
            record=record,
            sample_index=sample_index,
            sample_num_frames=sample_num_frames,
            clip_frame_indices=clip_frame_indices,
            local_indices=local_indices,
            scene_frame_indices=scene_frame_indices,
            image_paths=image_paths,
            images=images,
            timestamps=timestamps,
            intervals=intervals,
        )
        sample.update(
            {
            "masks": sky_masks,
            "sky_mask": sky_masks,
            "dynamic_mask": dynamic_masks,
            "gt_depth": gt_depth,
            "camera_to_world": camera_to_world,
            "camera_to_world_corrected": camera_to_world_corrected,
            "camera_to_ego": camera_to_ego,
            "ego_pose": ego_pose,
            "intrinsics": intrinsics,
            "raw_image_size_hw": torch.tensor(raw_image_size_hw, dtype=torch.long),
            "transfer_image_size_hw": torch.tensor(DEFAULT_TRANSFER_HW, dtype=torch.long),
            }
        )
        sample.update(object_tensors)
        sample.update(asset_payload)
        sample.update(self._build_tokenizer_payload(record, sample_num_frames, images))
        return sample


def summarise_value(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(value.keys()),
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "preview": value[:3],
        }
    return {
        "type": type(value).__name__,
        "value": value,
    }


def save_debug_visuals(sample, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [tensor_to_uint8_image(img) for img in sample["images_clean"]]
    sky_masks = [tensor_to_uint8_image(mask) for mask in sample["sky_mask"]]
    dynamic_masks = [tensor_to_uint8_image(mask) for mask in sample["dynamic_mask"]]

    cols = int(sample["cam_ids"].numel())
    Image.fromarray(make_image_grid(images, cols=cols)).save(output_dir / "images_clean_grid.png")
    Image.fromarray(make_image_grid(sky_masks, cols=cols)).save(output_dir / "sky_mask_grid.png")
    Image.fromarray(make_image_grid(dynamic_masks, cols=cols)).save(output_dir / "dynamic_mask_grid.png")

    overlay_images = [img.copy() for img in images]
    editable_count = int(sample["editable_object_count"].item())
    editable_slots = sample["editable_object_indices"][:editable_count].tolist()
    colors = [
        (0, 255, 0),
        (255, 200, 0),
        (255, 0, 0),
        (0, 200, 255),
        (255, 0, 255),
        (160, 255, 0),
    ]
    num_views = cols
    for color_idx, object_slot in enumerate(editable_slots):
        color = colors[color_idx % len(colors)]
        object_tag = sample["object_scene_raw_ids"][object_slot] or sample["object_asset_ids"][object_slot]
        for frame_idx in range(sample["frame_indices"].numel()):
            for view_offset, cam_id in enumerate(sample["cam_ids"].tolist()):
                if not bool(sample["object_bbox_valid_mask_selected"][object_slot, frame_idx, view_offset].item()):
                    continue
                image_idx = frame_idx * num_views + view_offset
                label = f"{object_tag[:8]} f{int(sample['frame_indices'][frame_idx].item())} c{cam_id}"
                overlay_images[image_idx] = draw_box_xyxy(
                    overlay_images[image_idx],
                    sample["object_bbox_model_selected"][object_slot, frame_idx, view_offset].tolist(),
                    color=color,
                    label=label,
                )

    Image.fromarray(make_image_grid(overlay_images, cols=cols)).save(output_dir / "images_with_boxes.png")

    summary = {
        "scene_name": sample["scene_name"],
        "clip_name": sample["clip_name"],
        "frame_indices": sample["frame_indices"].tolist(),
        "cam_ids": sample["cam_ids"].tolist(),
        "editable_object_count": editable_count,
        "editable_object_indices": editable_slots,
        "editable_asset_object_ids": sample["asset_meta"]["editable_asset_object_ids"],
        "editable_scene_raw_object_ids": sample["asset_meta"]["editable_scene_raw_object_ids"],
        "selected_visible_editable_frames": sample["frame_has_visible_editable_object_selected"].tolist(),
        "selected_front_visible_frames": sample["frame_has_front_visible_editable_object_selected"].tolist(),
    }
    save_json(output_dir / "sample_summary.json", summary)


def print_sample_summary(sample):
    print(f"scene_name: {sample['scene_name']}")
    print(f"clip_name: {sample['clip_name']}")
    print(f"frame_indices: {sample['frame_indices'].tolist()}")
    print(f"cam_ids: {sample['cam_ids'].tolist()}")
    print(f"editable_object_count: {int(sample['editable_object_count'].item())}")
    print(f"editable_asset_object_ids: {sample['asset_meta']['editable_asset_object_ids']}")
    print(f"editable_scene_raw_object_ids: {sample['asset_meta']['editable_scene_raw_object_ids']}")
    print(f"selected_visible_editable_frames: {sample['frame_has_visible_editable_object_selected'].tolist()}")
    print(f"selected_front_visible_frames: {sample['frame_has_front_visible_editable_object_selected'].tolist()}")
    print("key_shapes:")
    for key in [
        "images_clean",
        "sky_mask",
        "dynamic_mask",
        "gt_depth",
        "camera_to_world",
        "camera_to_world_corrected",
        "intrinsics",
        "object_bbox_model_selected",
        "object_bbox_valid_mask_selected",
        "object_obj_to_world_selected",
        "object_box_size_selected",
    ]:
        print(f"  {key}: {summarise_value(sample[key])}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Smoke-test WaymoEditDataset loading and export debug visuals.")
    parser.add_argument("--processed_root", type=str, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--asset_root", type=str, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--split", type=str, default="training", choices=["training", "validation"])
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--candidate_path", type=str, default=None)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--mode", type=int, default=2, choices=[1, 2])
    parser.add_argument("--views", type=int, default=1, choices=[1, 3])
    parser.add_argument("--sample_window", type=int, default=20)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/waymo_edit_dataset_debug",
        help="Directory for saved debug grids and summary json.",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = WaymoEditDataset(
        processed_root=args.processed_root,
        asset_root=args.asset_root,
        split=args.split,
        manifest_path=args.manifest_path,
        candidate_path=args.candidate_path,
        sequence_length=args.sequence_length,
        mode=args.mode,
        views=args.views,
        sample_window=args.sample_window,
    )

    sample = dataset[args.index]
    print_sample_summary(sample)
    save_debug_visuals(sample, args.output_dir)

    print("")
    print(f"manifest_path: {dataset.manifest_path}")
    print(f"candidate_path: {dataset.candidate_path}")
    print(f"num_samples: {len(dataset)}")


if __name__ == "__main__":
    main()
