from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CAMERA_NAMES = (
    "pinhole_front",
    "pinhole_front_left",
    "pinhole_front_right",
)
CAM_NAME_TO_ID = {
    "pinhole_front": 0,
    "pinhole_front_left": 1,
    "pinhole_front_right": 2,
    "pinhole_side_left": 3,
    "pinhole_side_right": 4,
}
DEFAULT_CLIP_LENGTH = 29
DEFAULT_FINAL_INFO_PATH = Path(__file__).resolve().parents[2] / "data" / "final_info.json"
DEFAULT_TRANSFER_HW = (704, 1280)
TRANSFER_BOX_LONG_EDGE_PX = float(max(DEFAULT_TRANSFER_HW))
MIN_EDIT_BOX_SIZE_PX = TRANSFER_BOX_LONG_EDGE_PX / 10.0
EDIT_OCCLUSION_IOU_THRESHOLD = 0.8
DEPTH_TIE_EPS = 1e-4
VEHICLE_CLASS_KEYWORDS = (
    "vehicle",
    "car",
    "truck",
    "bus",
    "van",
    "pickup",
    "trailer",
)
WAYMO_OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def load_json(path: Path, default=None):
    if path.exists():
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


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def scene_base_from_clip_name(clip_name: str) -> str:
    return clip_name.rsplit("_", 1)[0]


def scene_name_from_base(scene_base: str) -> str:
    return f"segment-{scene_base}_with_camera_labels"


def parse_clip_index(clip_name: str) -> int:
    return int(clip_name.rsplit("_", 1)[1])


def normalize_box_xyxy(box):
    if box is None:
        return None
    try:
        box_arr = [float(v) for v in box]
    except Exception:
        return None
    if len(box_arr) != 4:
        return None
    return box_arr


def normalize_box_sequence(box_sequence, expected_length: int):
    if box_sequence is None:
        return [None] * expected_length
    normalized = []
    for box in list(box_sequence)[:expected_length]:
        normalized.append(normalize_box_xyxy(box))
    if len(normalized) < expected_length:
        normalized.extend([None] * (expected_length - len(normalized)))
    return normalized


def build_present_flags_from_boxes(box_sequence):
    return [box is not None for box in box_sequence]


def build_filter_stats():
    return {
        "num_boxes_total": 0,
        "num_boxes_filtered_small": 0,
        "num_high_iou_pairs": 0,
        "num_boxes_filtered_occluded": 0,
        "num_objects_filtered_non_vehicle": 0,
        "kept_vehicle_class_counts": {},
        "filtered_non_vehicle_class_counts": {},
    }


def accumulate_filter_stats(target: dict, source: dict):
    for key, value in source.items():
        if isinstance(value, dict):
            target.setdefault(key, {})
            for child_key, child_value in value.items():
                target[key][child_key] = int(target[key].get(child_key, 0)) + int(child_value)
            continue
        target[key] = int(target.get(key, 0)) + int(value)


def normalize_class_name(class_name):
    return str(class_name).strip()


def is_vehicle_related_class(class_name):
    normalized = normalize_class_name(class_name).lower().replace("-", "_").replace(" ", "_")
    if normalized == "":
        return False
    return any(keyword in normalized for keyword in VEHICLE_CLASS_KEYWORDS)


def increment_counter(counter_dict, key):
    key = normalize_class_name(key) or "__EMPTY__"
    counter_dict[key] = int(counter_dict.get(key, 0)) + 1


def box_size_xyxy(box_xyxy):
    box = np.asarray(box_xyxy, dtype=np.float32)
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    return width, height


def is_transfer_box_large_enough(box_xyxy, min_box_size_px=MIN_EDIT_BOX_SIZE_PX):
    width, height = box_size_xyxy(box_xyxy)
    return width >= float(min_box_size_px) and height >= float(min_box_size_px)


def build_size_editable_from_boxes(box_sequence, stats=None, min_box_size_px=MIN_EDIT_BOX_SIZE_PX):
    editable = []
    for box in box_sequence:
        if box is None:
            editable.append(False)
            continue
        if stats is not None:
            stats["num_boxes_total"] += 1
        keep = is_transfer_box_large_enough(box, min_box_size_px=min_box_size_px)
        if stats is not None and not keep:
            stats["num_boxes_filtered_small"] += 1
        editable.append(bool(keep))
    return editable


def resolve_image_path(root: Path, frame_idx: int, cam_id: int):
    for ext in (".jpg", ".png"):
        path = root / f"{frame_idx:03d}_{cam_id}{ext}"
        if path.is_file():
            return path
    return None


def read_image_size(path: Path):
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
    out = box.copy()
    out[[0, 2]] *= geom["scale_x"]
    out[[1, 3]] = out[[1, 3]] * geom["scale_y"] - geom["crop_top"]
    out_h, out_w = geom["out_hw"]
    out[[0, 2]] = np.clip(out[[0, 2]], 0, out_w)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, out_h)
    return out.astype(np.float32)


def transfer_box_to_raw_box(box_xyxy, raw_hw, transfer_hw):
    box = np.asarray(box_xyxy, dtype=np.float32)
    raw_h, raw_w = raw_hw
    transfer_h, transfer_w = transfer_hw
    out = box.copy()
    out[[0, 2]] *= raw_w / transfer_w
    out[[1, 3]] *= raw_h / transfer_h
    return out.astype(np.float32)


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = np.asarray(box_a, dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(box_b, dtype=np.float32)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def box_overlap_ratios_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = np.asarray(box_a, dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(box_b, dtype=np.float32)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    overlap_a = 0.0 if area_a <= 0.0 else float(inter_area / area_a)
    overlap_b = 0.0 if area_b <= 0.0 else float(inter_area / area_b)
    return overlap_a, overlap_b


def compute_object_camera_depth_z(obj_to_world, camera_to_world):
    obj_to_world = np.asarray(obj_to_world, dtype=np.float32)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    world_to_camera = np.linalg.inv(camera_to_world)
    center_world = obj_to_world[:3, 3]
    center_cam = world_to_camera[:3, :3] @ center_world + world_to_camera[:3, 3]
    depth_z = float(center_cam[2])
    if not np.isfinite(depth_z) or depth_z <= 0.0:
        return None
    return depth_z


def resolve_occluded_slots(frame_boxes_by_slot, frame_depth_by_slot, iou_threshold=EDIT_OCCLUSION_IOU_THRESHOLD):
    slots = sorted(frame_boxes_by_slot.keys())
    occluded_slots = set()
    high_iou_pairs = 0
    for left_idx in range(len(slots)):
        slot_a = slots[left_idx]
        box_a = frame_boxes_by_slot[slot_a]
        for right_idx in range(left_idx + 1, len(slots)):
            slot_b = slots[right_idx]
            box_b = frame_boxes_by_slot[slot_b]
            overlap_a, overlap_b = box_overlap_ratios_xyxy(box_a, box_b)
            if max(overlap_a, overlap_b) <= float(iou_threshold):
                continue
            high_iou_pairs += 1
            depth_a = frame_depth_by_slot.get(slot_a)
            depth_b = frame_depth_by_slot.get(slot_b)
            if depth_a is None or depth_b is None:
                continue
            depth_delta = float(depth_a - depth_b)
            if abs(depth_delta) <= DEPTH_TIE_EPS:
                continue
            occluded_slots.add(slot_a if depth_delta > 0.0 else slot_b)
    return occluded_slots, high_iou_pairs


def compose_waymo_camera_to_world(ego_pose, camera_to_ego):
    return (np.asarray(ego_pose, dtype=np.float32) @ np.asarray(camera_to_ego, dtype=np.float32) @ WAYMO_OPENCV2DATASET).astype(np.float32)


def build_box_corners_world(obj_to_world, box_size):
    length, width, height = np.asarray(box_size, dtype=np.float32).tolist()
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


def project_world_box_to_raw_box(box_corners_world, camera_to_world, intrinsics, image_hw, eps=1e-6):
    corners_world = np.asarray(box_corners_world, dtype=np.float32)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float32))
    corners_cam = (world_to_camera[:3, :3] @ corners_world.T + world_to_camera[:3, 3:4]).T
    points_img = (np.asarray(intrinsics, dtype=np.float32) @ corners_cam.T).T
    depths = points_img[:, 2]
    if not np.all(np.isfinite(points_img)):
        return None
    if np.any(depths <= eps):
        return None
    projected = points_img[:, :2] / (depths[:, None] + eps)
    image_h, image_w = image_hw
    x1 = float(np.clip(projected[:, 0].min(), 0.0, image_w))
    x2 = float(np.clip(projected[:, 0].max(), 0.0, image_w))
    y1 = float(np.clip(projected[:, 1].min(), 0.0, image_h))
    y2 = float(np.clip(projected[:, 1].max(), 0.0, image_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_scene_index(processed_root: Path, split: str):
    annotation_root = processed_root / "waymo_edit_cache" / "annotations" / split
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Annotations root not found: {annotation_root}")

    processed_split_root = processed_root / split
    if not processed_split_root.is_dir():
        raise FileNotFoundError(f"Processed split root not found: {processed_split_root}")

    scene_index = {}
    for ann_path in sorted(annotation_root.glob("*.json")):
        payload = load_json(ann_path, {})
        if not payload:
            continue
        scene_id = int(payload["scene_id"])
        scene_dir = f"{scene_id:03d}"
        scene_root = processed_split_root / scene_dir
        if not scene_root.is_dir():
            continue
        scene_name = str(payload["scene_name"])
        scene_base = scene_name.removeprefix("segment-").removesuffix("_with_camera_labels")
        scene_index[scene_base] = {
            "scene_id": scene_id,
            "scene_dir": scene_dir,
            "scene_name": scene_name,
            "scene_base": scene_base,
            "scene_root": str(scene_root),
            "annotation_path": str(ann_path),
            "num_timesteps": int(payload.get("num_timesteps", 0)),
        }
    return scene_index


def load_scene_object_index(scene_root: Path):
    instances_info = load_json(scene_root / "instances" / "instances_info.json", {})
    object_index = {}
    for contig_instance_id, instance_info in instances_info.items():
        raw_object_id = str(instance_info.get("raw_object_id", instance_info.get("id", "")))
        if not raw_object_id:
            continue
        object_index[raw_object_id] = {
            "contig_instance_id": int(contig_instance_id),
            "class_name": str(instance_info.get("class_name", "")),
            "instance_info": instance_info,
        }
    return object_index


def load_scene_context(scene_info: dict):
    scene_root = Path(scene_info["scene_root"])
    annotation = load_json(Path(scene_info["annotation_path"]), {})
    object_index = load_scene_object_index(scene_root)
    image_size_by_cam = {}
    for camera_name in CAMERA_NAMES:
        cam_id = CAM_NAME_TO_ID[camera_name]
        image_path = resolve_image_path(scene_root / "images", 0, cam_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing image for scene {scene_info['scene_name']} cam {cam_id}")
        image_size_by_cam[cam_id] = read_image_size(image_path)
    return {
        "annotation": annotation,
        "object_index": object_index,
        "image_size_by_cam": image_size_by_cam,
        "scene_root": scene_root,
    }


def build_clip_candidate(
    record_idx: int,
    record: dict,
    scene_info: dict,
    scene_context: dict,
    asset_root: Path,
):
    filter_stats = build_filter_stats()
    clip_name = str(record["clip_name"])
    clip_index = parse_clip_index(clip_name)
    clip_frame_indices = list(
        range(
            clip_index * DEFAULT_CLIP_LENGTH,
            (clip_index + 1) * DEFAULT_CLIP_LENGTH,
        )
    )
    if len(clip_frame_indices) != DEFAULT_CLIP_LENGTH:
        return None, "invalid_clip_range", filter_stats
    if clip_frame_indices[-1] >= int(scene_info["num_timesteps"]):
        return None, "clip_out_of_range", filter_stats

    object_list = list(record.get("object_list", []))
    object_position = list(record.get("object_position", []))
    if len(object_list) == 0 or len(object_list) != len(object_position):
        return None, "invalid_object_payload", filter_stats

    annotation = scene_context["annotation"]
    scene_object_index = scene_context["object_index"]
    ego_pose_full = np.asarray(annotation["ego_pose"], dtype=np.float32)
    camera_to_ego_by_cam = {
        camera_name: np.asarray(annotation["camera_to_ego"][str(CAM_NAME_TO_ID[camera_name])], dtype=np.float32)
        for camera_name in CAMERA_NAMES
    }
    camera_to_world_clip_by_cam = {
        camera_name: [
            compose_waymo_camera_to_world(
                ego_pose_full[scene_frame_idx],
                camera_to_ego_by_cam[camera_name],
            )
            for scene_frame_idx in clip_frame_indices
        ]
        for camera_name in CAMERA_NAMES
    }

    filtered_objects = []
    for asset_object_id, position_payload in zip(object_list, object_position):
        asset_object_id = str(asset_object_id)
        scene_object = scene_object_index.get(asset_object_id)
        if scene_object is None:
            return None, "missing_scene_object", filter_stats
        class_name = str(scene_object["class_name"])
        if not is_vehicle_related_class(class_name):
            filter_stats["num_objects_filtered_non_vehicle"] += 1
            increment_counter(filter_stats["filtered_non_vehicle_class_counts"], class_name)
            continue
        increment_counter(filter_stats["kept_vehicle_class_counts"], class_name)
        filtered_objects.append((asset_object_id, position_payload, scene_object))

    if len(filtered_objects) == 0:
        return None, "no_vehicle_object", filter_stats

    objects = []
    object_track_contexts = []
    for slot, (asset_object_id, position_payload, scene_object) in enumerate(filtered_objects):
        asset_path = asset_root / f"{asset_object_id}.ply"
        if not asset_path.is_file():
            return None, "missing_asset", filter_stats
        instance_info = scene_object["instance_info"]
        frame_annotations = instance_info["frame_annotations"]
        track_frame_indices = [int(v) for v in frame_annotations.get("frame_idx", [])]
        track_lookup = {frame_idx: idx for idx, frame_idx in enumerate(track_frame_indices)}
        object_track_contexts.append(
            {
                "track_lookup": track_lookup,
                "frame_annotations": frame_annotations,
            }
        )

        boxes_by_view_transfer = {}
        boxes_by_view_raw = {}
        boxes_by_view_model = {}
        box_mapping_mode_by_view = {}
        bbox_present_by_view = {}
        bbox_present_count_by_view = {}
        for camera_name in CAMERA_NAMES:
            transfer_sequence = normalize_box_sequence(
                position_payload.get(camera_name),
                expected_length=DEFAULT_CLIP_LENGTH,
            )
            raw_sequence = []
            model_sequence = []
            mode_sequence = []
            cam_id = CAM_NAME_TO_ID[camera_name]
            raw_hw = scene_context["image_size_by_cam"][cam_id]
            for local_idx, transfer_box in enumerate(transfer_sequence):
                if transfer_box is None:
                    raw_sequence.append(None)
                    model_sequence.append(None)
                    mode_sequence.append(None)
                    continue
                raw_box = transfer_box_to_raw_box(
                    transfer_box,
                    raw_hw=raw_hw,
                    transfer_hw=DEFAULT_TRANSFER_HW,
                )
                mapping_mode = "resize"
                model_box = transform_box_xyxy(raw_box, raw_hw).tolist()
                raw_sequence.append([float(v) for v in raw_box.tolist()])
                model_sequence.append([float(v) for v in model_box])
                mode_sequence.append(mapping_mode)

            bbox_present = build_present_flags_from_boxes(transfer_sequence)
            boxes_by_view_transfer[camera_name] = transfer_sequence
            boxes_by_view_raw[camera_name] = raw_sequence
            boxes_by_view_model[camera_name] = model_sequence
            box_mapping_mode_by_view[camera_name] = mode_sequence
            bbox_present_by_view[camera_name] = bbox_present
            bbox_present_count_by_view[camera_name] = int(sum(bbox_present))

        objects.append(
            {
                "slot": int(slot),
                "asset_object_id": asset_object_id,
                "scene_raw_object_id": asset_object_id,
                "asset_path": str(asset_path),
                "contig_instance_id": int(scene_object["contig_instance_id"]),
                "class_name": str(scene_object["class_name"]),
                "match_score": 1.0,
                "boxes_by_view_transfer": boxes_by_view_transfer,
                "boxes_by_view_raw": boxes_by_view_raw,
                "boxes_by_view_model": boxes_by_view_model,
                "box_mapping_mode_by_view": box_mapping_mode_by_view,
                "bbox_present_by_view": bbox_present_by_view,
                "bbox_present_count_by_view": bbox_present_count_by_view,
            }
        )

    frame_has_front_present_object = []
    frame_has_front3_present_object = []
    for obj in objects:
        bbox_editable_by_view = {}
        bbox_editable_count_by_view = {}
        for camera_name in CAMERA_NAMES:
            editable_sequence = build_size_editable_from_boxes(
                obj["boxes_by_view_transfer"][camera_name],
                stats=filter_stats,
            )
            bbox_editable_by_view[camera_name] = editable_sequence
            bbox_editable_count_by_view[camera_name] = int(sum(editable_sequence))
        obj["bbox_editable_by_view"] = bbox_editable_by_view
        obj["bbox_editable_count_by_view"] = bbox_editable_count_by_view

    for local_idx, scene_frame_idx in enumerate(clip_frame_indices):
        for camera_name in CAMERA_NAMES:
            frame_boxes_by_slot = {}
            frame_depth_by_slot = {}
            for obj in objects:
                slot = int(obj["slot"])
                if not obj["bbox_editable_by_view"][camera_name][local_idx]:
                    continue
                frame_boxes_by_slot[slot] = obj["boxes_by_view_transfer"][camera_name][local_idx]
                track_context = object_track_contexts[slot]
                track_idx = track_context["track_lookup"].get(scene_frame_idx)
                if track_idx is None:
                    continue
                obj_to_world = np.asarray(
                    track_context["frame_annotations"]["obj_to_world"][track_idx],
                    dtype=np.float32,
                )
                depth_z = compute_object_camera_depth_z(
                    obj_to_world,
                    camera_to_world_clip_by_cam[camera_name][local_idx],
                )
                if depth_z is not None:
                    frame_depth_by_slot[slot] = depth_z
            occluded_slots, high_iou_pairs = resolve_occluded_slots(
                frame_boxes_by_slot,
                frame_depth_by_slot,
            )
            filter_stats["num_high_iou_pairs"] += int(high_iou_pairs)
            filter_stats["num_boxes_filtered_occluded"] += int(len(occluded_slots))
            for slot in occluded_slots:
                objects[slot]["bbox_editable_by_view"][camera_name][local_idx] = False

    editable_object_slots_by_frame_front = []
    editable_object_slots_by_frame_front3 = []
    for frame_idx in range(DEFAULT_CLIP_LENGTH):
        front_present = any(
            obj["bbox_present_by_view"]["pinhole_front"][frame_idx]
            for obj in objects
        )
        front3_present = any(
            obj["bbox_present_by_view"][camera_name][frame_idx]
            for obj in objects
            for camera_name in CAMERA_NAMES
        )
        frame_has_front_present_object.append(bool(front_present))
        frame_has_front3_present_object.append(bool(front3_present))
        front_slots = [
            int(obj["slot"])
            for obj in objects
            if obj["bbox_editable_by_view"]["pinhole_front"][frame_idx]
        ]
        front3_slots = [
            int(obj["slot"])
            for obj in objects
            if any(obj["bbox_editable_by_view"][camera_name][frame_idx] for camera_name in CAMERA_NAMES)
        ]
        editable_object_slots_by_frame_front.append(front_slots)
        editable_object_slots_by_frame_front3.append(front3_slots)

    for obj in objects:
        obj["bbox_editable_count_by_view"] = {
            camera_name: int(sum(obj["bbox_editable_by_view"][camera_name]))
            for camera_name in CAMERA_NAMES
        }

    frame_has_front_editable_object = [len(slots) > 0 for slots in editable_object_slots_by_frame_front]
    frame_has_front3_editable_object = [len(slots) > 0 for slots in editable_object_slots_by_frame_front3]

    candidate = {
        "record_index": int(record_idx),
        "scene_id": int(scene_info["scene_id"]),
        "scene_dir": str(scene_info["scene_dir"]),
        "scene_name": str(scene_info["scene_name"]),
        "scene_base": str(scene_info["scene_base"]),
        "clip_name": clip_name,
        "clip_index": int(clip_index),
        "clip_length": int(DEFAULT_CLIP_LENGTH),
        "scene_frame_indices": clip_frame_indices,
        "edit_mode": "replace",
        "num_objects": int(len(objects)),
        "object_ids": [obj["asset_object_id"] for obj in objects],
        "frame_has_front_present_object": frame_has_front_present_object,
        "frame_has_front3_present_object": frame_has_front3_present_object,
        "num_front_present_frames": int(sum(frame_has_front_present_object)),
        "num_front3_present_frames": int(sum(frame_has_front3_present_object)),
        "frame_has_front_editable_object": frame_has_front_editable_object,
        "frame_has_front3_editable_object": frame_has_front3_editable_object,
        "editable_object_slots_by_frame_front": editable_object_slots_by_frame_front,
        "editable_object_slots_by_frame_front3": editable_object_slots_by_frame_front3,
        "num_front_editable_frames": int(sum(frame_has_front_editable_object)),
        "num_front3_editable_frames": int(sum(frame_has_front3_editable_object)),
        "objects": objects,
    }
    return candidate, None, filter_stats


def build_manifest_entry(candidate: dict, *, views: int):
    editable_frames = (
        candidate["num_front_editable_frames"]
        if views == 1
        else candidate["num_front3_editable_frames"]
    )
    present_frames = (
        candidate["num_front_present_frames"]
        if views == 1
        else candidate["num_front3_present_frames"]
    )
    return {
        "record_index": int(candidate["record_index"]),
        "scene_id": int(candidate["scene_id"]),
        "scene_dir": str(candidate["scene_dir"]),
        "scene_name": str(candidate["scene_name"]),
        "clip_name": str(candidate["clip_name"]),
        "clip_index": int(candidate["clip_index"]),
        "num_objects": int(candidate["num_objects"]),
        "editable_frames": int(editable_frames),
        "present_frames": int(present_frames),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build clip-level metadata and view-specific manifests from final_info.json.",
    )
    parser.add_argument("--processed_root", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory that will receive metadata/<split> and manifests/<split>.",
    )
    parser.add_argument("--split", type=str, required=True, choices=["training", "validation"])
    parser.add_argument("--final_info_path", type=str, default=str(DEFAULT_FINAL_INFO_PATH))
    parser.add_argument("--asset_root", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    metadata_root = output_root / "metadata" / args.split
    manifest_root = output_root / "manifests" / args.split
    final_info_path = Path(args.final_info_path)
    asset_root = Path(args.asset_root)

    records = load_json(final_info_path, default=[])
    if not isinstance(records, list):
        raise ValueError(f"{final_info_path} does not contain a list")

    scene_index = load_scene_index(processed_root, args.split)
    scene_context_cache = {}

    scene_name_to_index = {}
    object_scene_index = {}
    clip_name_to_record_index = {}
    asset_candidate_ids = set()
    mode_a_candidates = []
    manifest_views1 = []
    manifest_views3 = []

    summary = {
        "split": args.split,
        "final_info_path": str(final_info_path),
        "asset_root": str(asset_root),
        "processed_root": str(processed_root),
        "box_coordinate_space": f"transfer_{DEFAULT_TRANSFER_HW[0]}x{DEFAULT_TRANSFER_HW[1]}",
        "transfer_image_size_hw": list(DEFAULT_TRANSFER_HW),
        "size_filter_long_edge_px": float(MIN_EDIT_BOX_SIZE_PX),
        "occlusion_iou_threshold": float(EDIT_OCCLUSION_IOU_THRESHOLD),
        "num_input_records": int(len(records)),
        "num_candidates": 0,
        "num_views1_manifest": 0,
        "num_views3_manifest": 0,
        "num_front_editable_manifest": 0,
        "num_front3_editable_manifest": 0,
        "skipped_missing_scene": 0,
        "skipped_invalid_payload": 0,
        "skipped_clip_out_of_range": 0,
        "skipped_no_vehicle_object": 0,
        "skipped_missing_asset": 0,
        "skipped_missing_scene_object": 0,
        "num_boxes_total": 0,
        "num_boxes_filtered_small": 0,
        "num_high_iou_pairs": 0,
        "num_boxes_filtered_occluded": 0,
        "num_objects_filtered_non_vehicle": 0,
        "kept_vehicle_class_counts": {},
        "filtered_non_vehicle_class_counts": {},
    }

    for record_idx, record in enumerate(records):
        clip_name = str(record.get("clip_name", ""))
        if clip_name == "":
            summary["skipped_invalid_payload"] += 1
            continue
        scene_base = scene_base_from_clip_name(clip_name)
        scene_info = scene_index.get(scene_base)
        if scene_info is None:
            summary["skipped_missing_scene"] += 1
            continue

        scene_name_to_index[scene_info["scene_name"]] = int(scene_info["scene_id"])

        scene_dir = scene_info["scene_dir"]
        if scene_dir not in scene_context_cache:
            scene_context_cache[scene_dir] = load_scene_context(scene_info)

        candidate, skip_reason, filter_stats = build_clip_candidate(
            record_idx=record_idx,
            record=record,
            scene_info=scene_info,
            scene_context=scene_context_cache[scene_dir],
            asset_root=asset_root,
        )
        accumulate_filter_stats(summary, filter_stats)
        if candidate is None:
            if skip_reason == "clip_out_of_range":
                summary["skipped_clip_out_of_range"] += 1
            elif skip_reason == "no_vehicle_object":
                summary["skipped_no_vehicle_object"] += 1
            elif skip_reason == "missing_asset":
                summary["skipped_missing_asset"] += 1
            elif skip_reason == "missing_scene_object":
                summary["skipped_missing_scene_object"] += 1
            else:
                summary["skipped_invalid_payload"] += 1
            continue

        mode_a_candidates.append(candidate)
        clip_name_to_record_index[candidate["clip_name"]] = int(candidate["record_index"])

        if candidate["num_front3_editable_frames"] > 0:
            manifest_views3.append(build_manifest_entry(candidate, views=3))
        if candidate["num_front_editable_frames"] > 0:
            manifest_views1.append(build_manifest_entry(candidate, views=1))

        for obj in candidate["objects"]:
            raw_object_id = obj["scene_raw_object_id"]
            asset_candidate_ids.add(raw_object_id)
            object_scene_index.setdefault(
                raw_object_id,
                {
                    "scene_id": int(candidate["scene_id"]),
                    "scene_dir": str(candidate["scene_dir"]),
                    "scene_name": str(candidate["scene_name"]),
                    "contig_instance_id": int(obj["contig_instance_id"]),
                    "class_name": str(obj["class_name"]),
                    "clip_names": [],
                },
            )
            object_scene_index[raw_object_id]["clip_names"].append(candidate["clip_name"])

    summary["num_candidates"] = int(len(mode_a_candidates))
    summary["num_views1_manifest"] = int(len(manifest_views1))
    summary["num_views3_manifest"] = int(len(manifest_views3))
    summary["num_front_editable_manifest"] = int(len(manifest_views1))
    summary["num_front3_editable_manifest"] = int(len(manifest_views3))

    write_json(metadata_root / "scene_name_to_index.json", scene_name_to_index)
    write_json(metadata_root / "object_scene_index.json", object_scene_index)
    write_json(metadata_root / "asset_candidate_ids.json", sorted(asset_candidate_ids))
    write_json(metadata_root / "clip_name_to_record_index.json", clip_name_to_record_index)
    write_jsonl(metadata_root / "mode_a_candidates.jsonl", mode_a_candidates)
    write_json(metadata_root / "metadata_summary.json", summary)

    write_jsonl(manifest_root / f"{args.split}_mode_a_views1.jsonl", manifest_views1)
    write_jsonl(manifest_root / f"{args.split}_mode_a_views3.jsonl", manifest_views3)
    write_json(
        manifest_root / "manifest_summary.json",
        {
            "split": args.split,
            "num_views1": int(len(manifest_views1)),
            "num_views3": int(len(manifest_views3)),
            "size_filter_long_edge_px": float(MIN_EDIT_BOX_SIZE_PX),
            "occlusion_iou_threshold": float(EDIT_OCCLUSION_IOU_THRESHOLD),
        },
    )


if __name__ == "__main__":
    main()
