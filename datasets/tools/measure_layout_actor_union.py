#!/usr/bin/env python
"""Measure the full eligible-actor union in production target windows (T13).

This deliberately bypasses every appearance-reference gate.  An actor is
eligible exactly when its class is vehicle/pedestrian/cyclist, its annotation
is finite with a positive box size, its optical center depth is in
``[0.5, 120]`` metres, and its cuboid intersects the front-camera image in at
least one frame of the target window.

Only complete 29-frame DGGT trunks are measured.  For ``S < 29`` every
contiguous window inside each complete trunk is included, matching the set of
starts available to production training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


WAYMO_OPENCV2DATASET = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
TRUNK_FRAMES = 29
DEFAULT_WINDOW_LENGTHS = (10, 29)
DEPTH_RANGE_METRES = (0.5, 120.0)


def _supported_class(class_name: Any) -> str | None:
    name = str(class_name).lower()
    if any(token in name for token in ("vehicle", "car", "truck", "bus")):
        return "vehicle"
    if any(token in name for token in ("pedestrian", "person")):
        return "pedestrian"
    if any(token in name for token in ("cyclist", "bicycle", "motorcycle", "rider")):
        return "cyclist"
    return None


def _load_matrix4(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"expected 16 matrix values in {path}, got {values.shape}")
    return values.reshape(4, 4)


def _load_intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.shape == (3, 3):
        return values
    flat = values.reshape(-1)
    if flat.size < 4:
        raise ValueError(f"expected at least fx/fy/cx/cy in {path}")
    fx, fy, cx, cy = (float(value) for value in flat[:4])
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _box_corners_world(obj_to_world: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    length, width, height = box_size.tolist()
    local = np.asarray(
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
        dtype=np.float64,
    )
    return (obj_to_world @ local.T).T[:, :3]


def _cuboid_intersects_image(
    corners_world: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
) -> bool:
    points_h = np.concatenate(
        [corners_world, np.ones((corners_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_camera = (world_to_camera @ points_h.T).T[:, :3]
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1.0e-6)
    if not bool(valid.any()):
        return False
    projected_h = (intrinsics @ points_camera[valid].T).T
    if not np.isfinite(projected_h).all():
        return False
    uv = projected_h[:, :2] / projected_h[:, 2:3]
    height, width = image_hw
    x0 = float(np.clip(uv[:, 0].min(), 0.0, float(width)))
    x1 = float(np.clip(uv[:, 0].max(), 0.0, float(width)))
    y0 = float(np.clip(uv[:, 1].min(), 0.0, float(height)))
    y1 = float(np.clip(uv[:, 1].max(), 0.0, float(height)))
    return x1 > x0 and y1 > y0


def _front_image_hw(scene_root: Path) -> tuple[int, int]:
    images = sorted((scene_root / "images").glob("*_0.jpg"))
    if not images:
        images = sorted((scene_root / "images").glob("*_0.png"))
    if not images:
        raise FileNotFoundError(f"no front-camera image under {scene_root / 'images'}")
    with Image.open(images[0]) as image:
        width, height = image.size
    return int(height), int(width)


def _window_counts(
    visible_by_frame: list[set[str]],
    window_length: int,
    trunk_frames: int = TRUNK_FRAMES,
) -> list[int]:
    if not 1 <= int(window_length) <= int(trunk_frames):
        raise ValueError("window_length must be within one trunk")
    counts: list[int] = []
    complete_frames = len(visible_by_frame) // trunk_frames * trunk_frames
    for trunk_start in range(0, complete_frames, trunk_frames):
        last_start = trunk_start + trunk_frames - window_length
        for start in range(trunk_start, last_start + 1):
            union: set[str] = set()
            for frame_tracks in visible_by_frame[start : start + window_length]:
                union.update(frame_tracks)
            counts.append(len(union))
    return counts


def measure_scene(
    scene_root_value: str,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
) -> dict[str, Any]:
    scene_root = Path(scene_root_value)
    scene_id = scene_root.name
    instances_path = scene_root / "instances" / "instances_info.json"
    with open(instances_path, "r", encoding="utf-8") as handle:
        instances_info = json.load(handle)

    pose_paths = sorted((scene_root / "ego_pose").glob("*.txt"))
    num_frames = len(pose_paths)
    if num_frames < TRUNK_FRAMES:
        raise ValueError(f"{scene_root} has only {num_frames} ego poses")
    camera_to_ego = _load_matrix4(scene_root / "extrinsics" / "0.txt")
    intrinsics = _load_intrinsics(scene_root / "intrinsics" / "0.txt")
    image_hw = _front_image_hw(scene_root)

    world_to_camera = []
    for pose_path in pose_paths:
        ego_to_world = _load_matrix4(pose_path)
        camera_to_world = ego_to_world @ camera_to_ego @ WAYMO_OPENCV2DATASET
        world_to_camera.append(np.linalg.inv(camera_to_world))

    visible_by_frame: list[set[str]] = [set() for _ in range(num_frames)]
    unsupported_tracks = 0
    malformed_annotations = 0
    for instance_key, info in instances_info.items():
        if not isinstance(info, dict) or _supported_class(info.get("class_name")) is None:
            unsupported_tracks += 1
            continue
        annotations = info.get("frame_annotations", {})
        frames = annotations.get("frame_idx", [])
        poses = annotations.get("obj_to_world", [])
        sizes = annotations.get("box_size", [])
        track_key = str(info.get("raw_object_id", info.get("id", instance_key)))
        for position, frame_value in enumerate(frames):
            frame_index = int(frame_value)
            if not 0 <= frame_index < num_frames or position >= len(poses) or position >= len(sizes):
                malformed_annotations += 1
                continue
            try:
                obj_to_world = np.asarray(poses[position], dtype=np.float64).reshape(4, 4)
                box_size = np.asarray(sizes[position], dtype=np.float64).reshape(3)
            except (TypeError, ValueError):
                malformed_annotations += 1
                continue
            if (
                not np.isfinite(obj_to_world).all()
                or not np.isfinite(box_size).all()
                or bool((box_size <= 0.0).any())
            ):
                malformed_annotations += 1
                continue
            center_world_h = np.concatenate([obj_to_world[:3, 3], [1.0]])
            center_camera = world_to_camera[frame_index] @ center_world_h
            depth = float(center_camera[2])
            if not DEPTH_RANGE_METRES[0] <= depth <= DEPTH_RANGE_METRES[1]:
                continue
            corners_world = _box_corners_world(obj_to_world, box_size)
            if _cuboid_intersects_image(
                corners_world,
                world_to_camera[frame_index],
                intrinsics,
                image_hw,
            ):
                visible_by_frame[frame_index].add(track_key)

    return {
        "scene_id": scene_id,
        "num_frames": num_frames,
        "num_complete_trunks": num_frames // TRUNK_FRAMES,
        "unsupported_tracks": unsupported_tracks,
        "malformed_annotations": malformed_annotations,
        "counts": {
            str(length): _window_counts(visible_by_frame, length)
            for length in window_lengths
        },
    }


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.int64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty window-count distribution")
    return {
        "num_windows": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def _discover_scenes(split_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(split_root.iterdir())
        if path.is_dir() and (path / "instances" / "instances_info.json").is_file()
    ]


def measure_dataset(
    roots: Iterable[str | Path],
    *,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    workers: int = 1,
) -> dict[str, Any]:
    split_roots = [Path(root) for root in roots]
    jobs: list[tuple[str, str]] = []
    for split_root in split_roots:
        jobs.extend((split_root.name, str(scene)) for scene in _discover_scenes(split_root))
    if not jobs:
        raise RuntimeError("no processed Waymo scenes with instances_info.json found")

    results: list[tuple[str, dict[str, Any]]] = []
    if workers <= 1:
        for split, scene in jobs:
            results.append((split, measure_scene(scene, window_lengths)))
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [
                (split, executor.submit(measure_scene, scene, window_lengths))
                for split, scene in jobs
            ]
            for split, future in futures:
                results.append((split, future.result()))

    summary: dict[str, Any] = {
        "contract": {
            "supported_classes": ["vehicle", "pedestrian", "cyclist"],
            "depth_metres": list(DEPTH_RANGE_METRES),
            "trunk_frames": TRUNK_FRAMES,
            "window_lengths": list(window_lengths),
            "appearance_gate": False,
        },
        "splits": {},
    }
    all_counts = {str(length): [] for length in window_lengths}
    for split_root in split_roots:
        split = split_root.name
        split_results = [result for result_split, result in results if result_split == split]
        split_counts = {
            str(length): [
                count
                for result in split_results
                for count in result["counts"][str(length)]
            ]
            for length in window_lengths
        }
        summary["splits"][split] = {
            "num_scenes": len(split_results),
            "num_complete_trunks": sum(r["num_complete_trunks"] for r in split_results),
            "unsupported_tracks": sum(r["unsupported_tracks"] for r in split_results),
            "malformed_annotations": sum(r["malformed_annotations"] for r in split_results),
            "windows": {
                key: _distribution(values) for key, values in split_counts.items()
            },
        }
        for key, values in split_counts.items():
            all_counts[key].extend(values)
    summary["all"] = {
        "num_scenes": len(results),
        "windows": {key: _distribution(values) for key, values in all_counts.items()},
    }
    summary["recommended_layout_max_actors"] = int(
        math.ceil(max(stats["max"] for stats in summary["all"]["windows"].values()) / 8.0)
        * 8
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "/data/disk2/lyy_dataset/waymo_processed_dggt/training",
            "/data/disk2/lyy_dataset/waymo_processed_dggt/validation",
        ],
    )
    parser.add_argument("--window_lengths", nargs="+", type=int, default=list(DEFAULT_WINDOW_LENGTHS))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = measure_dataset(
        args.roots,
        window_lengths=tuple(int(value) for value in args.window_lengths),
        workers=int(args.workers),
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(rendered + "\n", encoding="utf-8")
        tmp.replace(args.output)


if __name__ == "__main__":
    main()
