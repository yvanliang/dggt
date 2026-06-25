"""Export metric Waymo validation annotations for the ChatSim benchmark.

The validation tars already store object poses in the absolute Waymo world
frame and dimensions in metres. ChatSim is aligned to the same metric geometry
(with its world origin changed to the scene's frame-0 vehicle pose), so this
exporter intentionally does not run DGGT, estimate a Sim3, or use
``localize_validation_objects``.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_validation.toolkits.waymo_name_index import val_name2index

CLIP_LENGTH = 29
WAYMO_OPENCV_TO_EGO = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


class TarFrames:
    def __init__(self, path: Path):
        self.path = path
        with tarfile.open(path) as archive:
            names = sorted(name for name in archive.getnames() if name.endswith(".json"))
        self.names: dict[int, str] = {}
        for name in names:
            source_frame = int(Path(name).name.split(".")[-3])
            if source_frame % 3:
                raise ValueError(f"unexpected tar frame number in {name}")
            self.names[source_frame // 3] = name

    def frame(self, scene_frame_index: int) -> dict[str, Any]:
        name = self.names.get(int(scene_frame_index))
        if name is None:
            return {}
        with tarfile.open(self.path) as archive:
            member = archive.extractfile(name)
            return json.load(member) if member is not None else {}


def _list_front_frames(scene_root: Path) -> list[int]:
    frames = set()
    for suffix in ("jpg", "png"):
        for path in (scene_root / "images").glob(f"*_0.{suffix}"):
            frames.add(int(path.stem.split("_")[0]))
    return sorted(frames)


def _intrinsics(normalized: list[float], image_hw: list[int]) -> np.ndarray:
    height, width = image_hw
    fx, fy, cx, cy = normalized
    return np.array(
        [[fx * width, 0, cx * width], [0, fy * height, cy * height], [0, 0, 1]],
        dtype=np.float64,
    )


def _camera_to_world_opencv(ego_pose: Any, camera_to_ego: Any) -> np.ndarray:
    return (
        np.asarray(ego_pose, dtype=np.float64)
        @ np.asarray(camera_to_ego, dtype=np.float64)
        @ WAYMO_OPENCV_TO_EGO
    )


def _box_corners(pose: Any, lwh: Any) -> np.ndarray:
    length, width, height = np.asarray(lwh, dtype=np.float64)
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float64,
    )
    local = signs * np.array([length, width, height]) / 2.0
    transform = np.asarray(pose, dtype=np.float64)
    return (transform[:3, :3] @ local.T).T + transform[:3, 3]


def _project_bbox_raw(
    corners_world: np.ndarray,
    camera_to_world_opencv: np.ndarray,
    intrinsics: np.ndarray,
) -> list[float] | None:
    world_h = np.concatenate(
        (np.asarray(corners_world), np.ones((len(corners_world), 1))), axis=1
    )
    camera = (np.linalg.inv(camera_to_world_opencv) @ world_h.T).T[:, :3]
    depth = camera[:, 2]
    projected = (intrinsics @ camera.T).T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
    if not np.any(valid):
        return None
    points = uv[valid]
    return [
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    ]


def _box_payload(raw_id: str, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_id": str(raw_id),
        "object_to_world_waymo": info["object_to_world"],
        "lwh_m": info["object_lwh"],
        "is_moving": bool(info.get("object_is_moving", False)),
    }


def build_edit_spec(
    entry: dict[str, Any],
    *,
    entry_index: int,
    processed_root: Path,
    split: str,
    all_object_info_root: Path,
    insertion_root: Path,
    replacement_root: Path,
    reposition_root: Path,
) -> dict[str, Any]:
    segment, clip_text = str(entry["clip_name"]).rsplit("_", 1)
    clip_index = int(clip_text)
    scene_index = int(val_name2index[segment])
    scene_root = processed_root / split / f"{scene_index:03d}"
    all_frames = _list_front_frames(scene_root)
    start = clip_index * CLIP_LENGTH
    scene_frame_indices = all_frames[start : start + CLIP_LENGTH]
    if len(scene_frame_indices) != CLIP_LENGTH:
        raise ValueError(
            f"{entry['clip_name']}: expected {CLIP_LENGTH} frames, "
            f"found {len(scene_frame_indices)}"
        )

    annotation_path = (
        processed_root
        / "waymo_edit_cache"
        / "annotations"
        / split
        / f"segment-{segment}_with_camera_labels.json"
    )
    annotation = json.loads(annotation_path.read_text())
    raw_hw = annotation["original_image_size"]["0"]
    intrinsics = _intrinsics(annotation["normalized_intrinsics"]["0"], raw_hw)
    camera_to_ego = annotation["camera_to_ego"]["0"]

    originals = TarFrames(all_object_info_root / f"{segment}.tar")
    insertions = TarFrames(insertion_root / f"{segment}.tar")
    replacements = TarFrames(replacement_root / f"{segment}.tar")
    repositions = TarFrames(reposition_root / f"{segment}.tar")
    source_ids = {
        role: str(entry["origin_object_dict"][role])
        for role in ("deletion", "replacement", "repositioning")
    }
    asset_ids = {
        "insertion": str(entry["insertion_candidates"]),
        "replacement": str(entry["replacement_candidates"]),
        "repositioning": source_ids["repositioning"],
    }

    frames = []
    for frame_in_clip, scene_frame_index in enumerate(scene_frame_indices):
        c2w = _camera_to_world_opencv(
            annotation["ego_pose"][scene_frame_index], camera_to_ego
        )
        original_frame = originals.frame(scene_frame_index)
        delete_tracks = {}
        for role, raw_id in source_ids.items():
            info = original_frame.get(raw_id)
            if info is not None:
                delete_tracks[role] = _box_payload(raw_id, info)

        target_info = {
            "insertion": insertions.frame(scene_frame_index).get("insertion_0"),
            "replacement": replacements.frame(scene_frame_index).get(
                source_ids["replacement"]
            ),
            "repositioning": repositions.frame(scene_frame_index).get(
                source_ids["repositioning"]
            ),
        }
        assets = {}
        for role, info in target_info.items():
            if info is None:
                continue
            pose = info["object_to_world"]
            lwh = info["object_lwh"]
            assets[role] = {
                "asset_id": asset_ids[role],
                "object_to_world_waymo": pose,
                "lwh_m": lwh,
                "target_bbox_waymo_raw": _project_bbox_raw(
                    _box_corners(pose, lwh), c2w, intrinsics
                ),
            }

        frames.append(
            {
                "frame_in_clip": frame_in_clip,
                "scene_frame_index": scene_frame_index,
                "camera_to_world_waymo_opencv": c2w.tolist(),
                "delete_tracks": delete_tracks,
                "assets": assets,
            }
        )

    return {
        "schema_version": 2,
        "coordinate_system": {
            "object_poses": "absolute_waymo_world_flu",
            "camera_poses": "absolute_waymo_world_opencv_rdf",
            "dimensions": "metres_lwh",
            "chatsim_conversion": (
                "inv(waymo_front_camera_image_vehicle_pose_frame0) "
                "@ object_to_world_waymo"
            ),
            "scale_conversion": "none",
        },
        "entry_index": int(entry.get("index", entry_index)),
        "segment": segment,
        "clip_index": clip_index,
        "scene_frame_indices": scene_frame_indices,
        "source_ids": source_ids,
        "asset_ids": asset_ids,
        "raw_image_size_hw": raw_hw,
        "frames": frames,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_out_root", required=True)
    parser.add_argument(
        "--processed_root",
        default="/data/disk2/lyy_dataset/waymo_processed_dggt",
    )
    parser.add_argument("--final_info_path", default="data/final_info_validation.json")
    parser.add_argument(
        "--all_object_info_root",
        default="data/validation_info/all_object_info",
    )
    parser.add_argument(
        "--all_object_info_insertion_root",
        default="data/validation_info/all_object_info_insertion",
    )
    parser.add_argument(
        "--all_object_info_replacement_root",
        default="data/validation_info/all_object_info_replacement",
    )
    parser.add_argument(
        "--all_object_info_reposition_root",
        default="data/validation_info/all_object_info_reposition",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start", type=int, default=0, help="entry index, inclusive")
    parser.add_argument("--end", type=int, default=None, help="entry index, exclusive")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    entries = json.loads(Path(args.final_info_path).read_text())
    start = max(0, int(args.start))
    end = min(len(entries), int(args.end if args.end is not None else len(entries)))
    for index in range(start, end):
        spec = build_edit_spec(
            entries[index],
            entry_index=index,
            processed_root=Path(args.processed_root),
            split=args.split,
            all_object_info_root=Path(args.all_object_info_root),
            insertion_root=Path(args.all_object_info_insertion_root),
            replacement_root=Path(args.all_object_info_replacement_root),
            reposition_root=Path(args.all_object_info_reposition_root),
        )
        out = Path(args.spec_out_root) / f"entry_{spec['entry_index']:03d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "projection_debug").mkdir(exist_ok=True)
        (out / "edit_spec.json").write_text(
            json.dumps(spec, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
