#!/usr/bin/env python
"""Build authoritative Waymo hdmap JSON/NPZ pairs from cached frame 0.

All seven geometry classes and every published attribute come from the same
``MapFeature`` object.  RDS is consulted only for the non-blocking T16 geometry
cross-check implemented in ``build_hdmap_dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.tools.build_hdmap_dataset import cross_validate_tfrecord_geometry  # noqa: E402
from datasets.tools.fetch_waymo_map_frame0 import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    _frame_from_payload,
    read_first_record,
)
from datasets.tools.hdmap_schema import (  # noqa: E402
    ATTRIBUTE_SOURCE_TFRECORD,
    CLASS_NAMES,
    GEOMETRY_SOURCE_TFRECORD,
    HDMapFeature,
    HDMapScene,
    normalize_segment_name,
    read_scene_json,
    read_scene_npz,
    scene_id_map,
    summarize,
    write_scene,
)


DEFAULT_OUT_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt"
DEFAULT_RDS_ROOT = "/data/disk2/lyy_dataset/waymo_rds"
DEFAULT_LISTS = {
    "training": "data/waymo_train_list_full.txt",
    "validation": "data/waymo_val_list_full.txt",
}


class EmptyMapFeaturesError(ValueError):
    pass


def _point(point: Any) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def _points(points: Iterable[Any]) -> np.ndarray:
    return np.asarray([_point(point) for point in points], dtype=np.float64)


def _boundary(boundary: Any) -> dict[str, int]:
    return {
        "lane_start_index": int(boundary.lane_start_index),
        "lane_end_index": int(boundary.lane_end_index),
        "boundary_feature_id": int(boundary.boundary_feature_id),
        "boundary_type": int(boundary.boundary_type),
    }


def _neighbor(neighbor: Any) -> dict[str, Any]:
    return {
        "feature_id": int(neighbor.feature_id),
        "self_start_index": int(neighbor.self_start_index),
        "self_end_index": int(neighbor.self_end_index),
        "neighbor_start_index": int(neighbor.neighbor_start_index),
        "neighbor_end_index": int(neighbor.neighbor_end_index),
        "boundaries": [_boundary(boundary) for boundary in neighbor.boundaries],
    }


def features_from_frame(frame: Any) -> list[HDMapFeature]:
    """Translate frame-0 MapFeature messages without matching or inference."""
    if len(frame.map_features) == 0:
        raise EmptyMapFeaturesError(f"segment {frame.context.name!r} has no map_features")
    grouped: dict[str, list[HDMapFeature]] = {name: [] for name in CLASS_NAMES}
    for map_feature in frame.map_features:
        cls = map_feature.WhichOneof("feature_data")
        if cls not in grouped:
            raise ValueError(
                f"map feature id={int(map_feature.id)} has unsupported payload {cls!r}"
            )
        feature_id = int(map_feature.id)
        if cls == "lane":
            lane = map_feature.lane
            feature = HDMapFeature(
                cls=cls,
                vertices=_points(lane.polyline),
                feature_id=feature_id,
                subtype=int(lane.type),
                speed_limit_mph=float(lane.speed_limit_mph),
                interpolating=bool(lane.interpolating),
                attributes={
                    "entry_lanes": [int(value) for value in lane.entry_lanes],
                    "exit_lanes": [int(value) for value in lane.exit_lanes],
                    "left_boundaries": [_boundary(value) for value in lane.left_boundaries],
                    "right_boundaries": [_boundary(value) for value in lane.right_boundaries],
                    "left_neighbors": [_neighbor(value) for value in lane.left_neighbors],
                    "right_neighbors": [_neighbor(value) for value in lane.right_neighbors],
                },
            )
        elif cls == "road_line":
            feature = HDMapFeature(
                cls=cls,
                vertices=_points(map_feature.road_line.polyline),
                feature_id=feature_id,
                subtype=int(map_feature.road_line.type),
            )
        elif cls == "road_edge":
            feature = HDMapFeature(
                cls=cls,
                vertices=_points(map_feature.road_edge.polyline),
                feature_id=feature_id,
                subtype=int(map_feature.road_edge.type),
            )
        elif cls in {"crosswalk", "speed_bump", "driveway"}:
            feature = HDMapFeature(
                cls=cls,
                vertices=_points(getattr(map_feature, cls).polygon),
                feature_id=feature_id,
            )
        elif cls == "stop_sign":
            feature = HDMapFeature(
                cls=cls,
                vertices=np.asarray([_point(map_feature.stop_sign.position)], dtype=np.float64),
                feature_id=feature_id,
                attributes={"lane": [int(value) for value in map_feature.stop_sign.lane]},
            )
        else:  # pragma: no cover - guarded by the class membership check
            raise AssertionError(cls)
        grouped[cls].append(feature)
    # Freeze class-major order while retaining the original index order inside
    # each class.  RDS emits this same per-class order, which makes T16 direct.
    return [feature for cls in CLASS_NAMES for feature in grouped[cls]]


def _load_offset(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    offsets = np.asarray(payload.get("offsets"), dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[0] < 1 or offsets.shape[1] != 3:
        raise ValueError(f"{path}: offsets must be non-empty [S,3], got {offsets.shape}")
    if not np.isfinite(offsets).all():
        raise ValueError(f"{path}: offsets contain NaN or Inf")
    return offsets


def build_scene_from_cache(
    *,
    head_path: str | Path,
    offset_path: str | Path,
    split: str,
    scene_id: str,
    expected_segment: str,
) -> HDMapScene:
    frame = _frame_from_payload(read_first_record(head_path))
    segment = str(frame.context.name)
    if segment != expected_segment:
        raise ValueError(
            f"{head_path}: frame context {segment!r} != list mapping {expected_segment!r}"
        )
    offsets = _load_offset(offset_path)
    frame0_offset = np.asarray(
        [[frame.map_pose_offset.x, frame.map_pose_offset.y, frame.map_pose_offset.z]],
        dtype=np.float64,
    )
    if not np.array_equal(offsets[:1], frame0_offset):
        raise ValueError(f"{segment}: remote offset frame 0 differs from parsed frame 0")
    return HDMapScene(
        segment=segment,
        scene_id=scene_id,
        split=split,
        features=features_from_frame(frame),
        geometry_source=GEOMETRY_SOURCE_TFRECORD,
        attribute_source=ATTRIBUTE_SOURCE_TFRECORD,
        map_pose_offset=offsets,
    ).validate()


def _assert_scene_equal(expected: HDMapScene, actual: HDMapScene, source: str) -> None:
    scalar_fields = (
        "segment",
        "scene_id",
        "split",
        "geometry_source",
        "attribute_source",
        "schema_version",
    )
    for name in scalar_fields:
        if getattr(expected, name) != getattr(actual, name):
            raise AssertionError(f"{source}: scene field {name} differs")
    if not np.array_equal(expected.map_pose_offset, actual.map_pose_offset):
        raise AssertionError(f"{source}: map_pose_offset differs")
    if len(expected.features) != len(actual.features):
        raise AssertionError(f"{source}: feature count differs")
    for index, (lhs, rhs) in enumerate(zip(expected.features, actual.features)):
        if (
            lhs.cls != rhs.cls
            or lhs.feature_id != rhs.feature_id
            or lhs.subtype != rhs.subtype
            or lhs.interpolating != rhs.interpolating
            or lhs.attributes != rhs.attributes
            or not np.array_equal(lhs.vertices, rhs.vertices)
            or not (
                (np.isnan(lhs.speed_limit_mph) and np.isnan(rhs.speed_limit_mph))
                or lhs.speed_limit_mph == rhs.speed_limit_mph
            )
        ):
            raise AssertionError(f"{source}: feature {index} differs")


def verify_written_scene(root: str | Path, scene: HDMapScene) -> None:
    _assert_scene_equal(scene, read_scene_json(root, scene.scene_id), "json")
    _assert_scene_equal(scene, read_scene_npz(root, scene.scene_id), "npz")


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    try:
        scene = build_scene_from_cache(
            head_path=task["head_path"],
            offset_path=task["offset_path"],
            split=task["split"],
            scene_id=task["scene_id"],
            expected_segment=task["segment"],
        )
        write_scene(task["out_root"], scene)
        verify_written_scene(task["out_root"], scene)
        t16 = None
        if task["rds_validate"]:
            t16 = cross_validate_tfrecord_geometry(scene, task["rds_root"])
        return {
            "ok": True,
            "scene_id": scene.scene_id,
            "segment": scene.segment,
            "counts": scene.counts(),
            "vertices": int(sum(len(feature.vertices) for feature in scene.features)),
            "frames": int(scene.map_pose_offset.shape[0]),
            "summary": summarize(scene),
            "t16": t16,
        }
    except EmptyMapFeaturesError as exc:
        return {
            "ok": True,
            "skipped_empty_map_features": True,
            "scene_id": task["scene_id"],
            "segment": task["segment"],
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - isolate/report each segment
        return {
            "ok": False,
            "scene_id": task.get("scene_id"),
            "segment": task.get("segment"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def _cache_segment(path: Path) -> str:
    name = path.name
    if name.endswith(".head"):
        name = name[: -len(".head")]
    return normalize_segment_name(name)


def _index_heads(cache_root: Path, split: str) -> dict[str, Path]:
    paths = sorted((cache_root / split).glob("segment-*.tfrecord.head"))
    indexed = {_cache_segment(path): path for path in paths}
    if len(indexed) != len(paths):
        raise ValueError(f"duplicate normalized segment in {cache_root / split}")
    return indexed


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--rds_root", default=DEFAULT_RDS_ROOT)
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--train_list")
    parser.add_argument("--val_list")
    parser.add_argument("--splits", nargs="+", choices=("training", "validation"), default=["training", "validation"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_rds_validate", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root)
    cache_root = Path(args.cache_root)
    out_root = Path(args.out_root)
    list_paths = {
        "training": Path(args.train_list) if args.train_list else repo_root / DEFAULT_LISTS["training"],
        "validation": Path(args.val_list) if args.val_list else repo_root / DEFAULT_LISTS["validation"],
    }
    run_summary: dict[str, Any] = {
        "cache_root": str(cache_root),
        "out_root": str(out_root),
        "rds_root": str(args.rds_root),
        "splits": {},
    }
    exit_code = 0
    for split in args.splits:
        mapping = scene_id_map(list_paths[split])
        heads = _index_heads(cache_root, split)
        segments = sorted(mapping)
        if args.limit:
            segments = segments[: args.limit]
        tasks = []
        skipped_existing = []
        for segment in segments:
            scene_id = mapping[segment]
            head_path = heads.get(segment)
            offset_path = cache_root / "map_pose_offset" / split / f"{segment}.json"
            split_out = out_root / f"{split}_hdmap"
            if args.skip_existing and all(
                (split_out / scene_id / filename).is_file()
                for filename in ("hdmap.json", "hdmap.npz")
            ):
                skipped_existing.append(scene_id)
                continue
            tasks.append(
                {
                    "head_path": None if head_path is None else str(head_path),
                    "offset_path": str(offset_path),
                    "out_root": str(split_out),
                    "rds_root": str(args.rds_root),
                    "rds_validate": not args.no_rds_validate,
                    "split": split,
                    "scene_id": scene_id,
                    "segment": segment,
                }
            )
        print(f"[{split}] list={len(mapping)} tasks={len(tasks)} existing={len(skipped_existing)}")
        started = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(_worker, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                if result.get("ok") and not result.get("skipped_empty_map_features"):
                    if done <= 3 or done % 200 == 0:
                        print(f"  [{done}/{len(tasks)}] {result['summary']}")
        failures = [result for result in results if not result.get("ok")]
        empty = [result for result in results if result.get("skipped_empty_map_features")]
        t16_mismatches = [
            {"scene_id": result["scene_id"], "segment": result["segment"], "report": result["t16"]}
            for result in results
            if result.get("t16") is not None and not result["t16"].get("ok")
        ]
        split_summary = {
            "listed": len(mapping),
            "processed": len(results) - len(failures) - len(empty),
            "skipped_existing": skipped_existing,
            "empty_map_features_segments": [result["segment"] for result in empty],
            "failures": failures,
            "t16_mismatch_count": len(t16_mismatches),
            "t16_mismatches": t16_mismatches,
            "elapsed_seconds": time.time() - started,
            "class_totals": {
                name: sum(result.get("counts", {}).get(name, 0) for result in results)
                for name in CLASS_NAMES
            },
        }
        run_summary["splits"][split] = split_summary
        print(
            f"[{split}] wrote={split_summary['processed']} empty={len(empty)} "
            f"failures={len(failures)} t16_mismatches={len(t16_mismatches)}"
        )
        if failures:
            exit_code = 1
            for failure in failures[:10]:
                print(f"  FAIL {failure['scene_id']} {failure['segment']}: {failure['error']}")

    summary_path = args.summary or cache_root / "build_summary.json"
    _write_summary(summary_path, run_summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
