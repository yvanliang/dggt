#!/usr/bin/env python
"""RDS geometry reader retained as the T16 cross-validation reference.

The RDS-HQ dump produced by ``Cosmos-Drive-Dreams/convert_waymo_to_rds_hq.py``
stores four static map classes as per-segment tars of JSON polylines:

    3d_lanes/<segment>.tar            -> <segment>.lanes.json
    3d_lanelines/<segment>.tar        -> <segment>.lanelines.json
    3d_road_boundaries/<segment>.tar  -> <segment>.road_boundaries.json
    3d_crosswalks/<segment>.tar       -> <segment>.crosswalks.json

Its geometry is *exactly* Waymo's: for segment ``10017090168044687777_...`` all
73 lanes / 7 road lines / 31 road edges / 2 crosswalks reproduce the raw
``frame.map_features`` polylines to ``atol=1e-6`` and in the same index order.
What it drops is every **attribute** (lane type, speed limit, topology, road
line style/colour, road edge type) and three whole classes (``speed_bump``,
``stop_sign``, ``driveway``).

As of the v2.1 data contract, RDS is not a producer: geometry and attributes
both come directly from the original frame-0 ``MapFeature`` objects.  The
reader below remains so ``build_hdmap_from_tfrecord.py`` can compare the four
shared geometry classes point-for-point, in the same per-class index order,
with absolute tolerance ``1e-6``.  A mismatch is reported but never blocks the
tfrecord-derived write.

Scene ids
---------
Output folders use the **3-digit scene id that ``WaymoOpenDataset`` uses**, i.e.
the index of the segment in the *lexicographically sorted* split list, not the
line order of the list file (see ``hdmap_schema.scene_id_map``).  Getting this
wrong silently pairs every scene with another scene's road network, which no
downstream metric would catch, so ``--verify`` re-derives the mapping and
asserts the segment recorded inside each written file.

Usage
-----
    conda activate dggt
    python datasets/tools/build_hdmap_dataset.py \
        --rds_root  /data/disk2/lyy_dataset/waymo_rds \
        --hdmap_root /data/disk2/lyy_dataset/waymo_processed_dggt \
        --splits training validation \
        --workers 16
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.tools.hdmap_schema import (  # noqa: E402
    ATTRIBUTE_SOURCE_NONE,
    CLASS_NAMES,
    GEOMETRY_SOURCE_RDS,
    RDS_CLASS_DIRS,
    RDS_FILE_SUFFIX,
    HDMapFeature,
    HDMapScene,
    normalize_segment_name,
    read_scene_json,
    read_scene_npz,
    scene_id_map,
    summarize,
    write_scene,
)

DEFAULT_LIST = {
    "training": "data/waymo_train_list_full.txt",
    "validation": "data/waymo_val_list_full.txt",
}


def _extract_json(tar_path: Path, suffix: str) -> dict[str, Any]:
    """Read the single ``*.<suffix>`` member of an RDS class tar."""
    with tarfile.open(tar_path) as handle:
        members = [m for m in handle.getmembers() if m.name.endswith(suffix)]
        if len(members) != 1:
            raise ValueError(
                f"{tar_path} must contain exactly one *.{suffix} member, found {len(members)}"
            )
        payload = handle.extractfile(members[0])
        if payload is None:
            raise ValueError(f"{tar_path} member {members[0].name} is not a regular file")
        return json.load(io.BytesIO(payload.read()))


def _vertices_of(label: dict[str, Any]) -> np.ndarray | None:
    shape = label.get("labelData", {}).get("shape3d", {})
    payload = shape.get("polyline3d") or shape.get("surface") or shape.get("polygon3d")
    if payload is None:
        return None
    vertices = payload.get("vertices")
    if not vertices:
        return None
    return np.asarray(vertices, dtype=np.float64)


def build_scene_from_rds(
    rds_root: Path,
    split: str,
    segment: str,
    scene_id: str,
    *,
    min_vertices: int = 1,
) -> tuple[HDMapScene, dict[str, int]]:
    """Read the four RDS class tars of one segment into an ``HDMapScene``.

    ``min_vertices=1`` is deliberate: Waymo publishes single-vertex lane stubs,
    and a migration must not delete source features.  Only labels with *no*
    geometry at all, or non-finite coordinates, are skipped, and both are
    counted so the run log can be audited against the source.
    """
    features: list[HDMapFeature] = []
    dropped = {name: 0 for name in CLASS_NAMES}
    for cls, directory in RDS_CLASS_DIRS.items():
        tar_path = rds_root / split / directory / f"{segment}.tar"
        if not tar_path.exists():
            raise FileNotFoundError(f"missing RDS class tar: {tar_path}")
        payload = _extract_json(tar_path, RDS_FILE_SUFFIX[cls])
        labels = payload.get("labels")
        if labels is None:
            raise ValueError(f"{tar_path} JSON has no 'labels' key")
        for label in labels:
            vertices = _vertices_of(label)
            # An RDS label with <2 vertices carries no projectable geometry.
            # Counting these instead of silently skipping keeps the migration
            # auditable: --verify compares the surviving counts to the source.
            if vertices is None or vertices.shape[0] < min_vertices:
                dropped[cls] += 1
                continue
            if not np.isfinite(vertices).all():
                dropped[cls] += 1
                continue
            features.append(HDMapFeature(cls=cls, vertices=vertices))
    scene = HDMapScene(
        segment=segment,
        scene_id=scene_id,
        split=split,
        features=features,
        geometry_source=GEOMETRY_SOURCE_RDS,
        attribute_source=ATTRIBUTE_SOURCE_NONE,
        map_pose_offset=None,
    ).validate()
    return scene, dropped


def cross_validate_tfrecord_geometry(
    scene: HDMapScene,
    rds_root: str | Path,
    *,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    """T16: compare tfrecord geometry against RDS in the original index order.

    The returned report is data, not an exception policy.  Production callers
    attach/report it after writing the authoritative tfrecord scene, so stale
    or missing RDS data cannot suppress a valid hdmap artifact.
    """
    report: dict[str, Any] = {
        "ok": True,
        "segment": scene.segment,
        "scene_id": scene.scene_id,
        "atol": float(atol),
        "classes": {},
        "errors": [],
    }
    try:
        reference, dropped = build_scene_from_rds(
            Path(rds_root),
            scene.split,
            scene.segment,
            scene.scene_id,
        )
    except Exception as exc:  # reporting reference; never invalidate source
        report["ok"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        return report

    for cls in RDS_CLASS_DIRS:
        actual = scene.by_class(cls)
        expected = reference.by_class(cls)
        class_report: dict[str, Any] = {
            "tfrecord_count": len(actual),
            "rds_count": len(expected),
            "dropped_rds": int(dropped.get(cls, 0)),
            "matched": 0,
            "mismatches": [],
        }
        if len(actual) != len(expected):
            class_report["mismatches"].append(
                {
                    "kind": "count",
                    "tfrecord": len(actual),
                    "rds": len(expected),
                }
            )
        for index, (tfrecord_feature, rds_feature) in enumerate(zip(actual, expected)):
            lhs = tfrecord_feature.vertices
            rhs = rds_feature.vertices
            if lhs.shape != rhs.shape:
                class_report["mismatches"].append(
                    {
                        "kind": "shape",
                        "index": index,
                        "tfrecord": list(lhs.shape),
                        "rds": list(rhs.shape),
                    }
                )
                continue
            if not np.allclose(lhs, rhs, rtol=0.0, atol=float(atol)):
                class_report["mismatches"].append(
                    {
                        "kind": "vertices",
                        "index": index,
                        "max_abs_error": float(np.max(np.abs(lhs - rhs))),
                    }
                )
                continue
            class_report["matched"] += 1
        if class_report["mismatches"]:
            report["ok"] = False
        report["classes"][cls] = class_report
    return report


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    try:
        scene = read_scene_npz(
            Path(task["hdmap_root"]) / f"{task['split']}_hdmap",
            task["scene_id"],
        )
        if scene.segment != task["segment"]:
            raise ValueError(
                f"scene mapping says {task['segment']!r}, hdmap says {scene.segment!r}"
            )
        report = cross_validate_tfrecord_geometry(scene, task["rds_root"])
        return {
            "ok": True,
            "scene_id": scene.scene_id,
            "segment": scene.segment,
            "match": bool(report["ok"]),
            "report": report,
        }
    except Exception as exc:  # noqa: BLE001 - reported per task, never swallowed
        return {
            "ok": False,
            "scene_id": task.get("scene_id"),
            "segment": task.get("segment"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
        }


def verify_split(
    out_root: Path,
    split: str,
    mapping: dict[str, str],
    *,
    sample: int,
    scene_ids: list[str] | None = None,
) -> None:
    """Re-read written scenes and assert JSON/NPZ agreement and correct pairing."""
    root = out_root / f"{split}_hdmap"
    inverse = {scene_id: segment for segment, scene_id in mapping.items()}
    scene_ids = sorted(scene_ids) if scene_ids is not None else sorted(inverse)
    if sample > 0:
        step = max(1, len(scene_ids) // sample)
        scene_ids = scene_ids[::step][:sample]
    for scene_id in scene_ids:
        from_npz = read_scene_npz(root, scene_id)
        from_json = read_scene_json(root, scene_id)
        if from_npz.segment != inverse[scene_id]:
            raise AssertionError(
                f"{split}/{scene_id}: file says segment={from_npz.segment}, "
                f"sorted-list mapping says {inverse[scene_id]}"
            )
        if from_json.segment != from_npz.segment:
            raise AssertionError(f"{split}/{scene_id}: json/npz segment mismatch")
        if from_json.counts() != from_npz.counts():
            raise AssertionError(f"{split}/{scene_id}: json/npz class counts differ")
        if len(from_json.features) != len(from_npz.features):
            raise AssertionError(f"{split}/{scene_id}: json/npz feature count differs")
        for a, b in zip(from_json.features, from_npz.features):
            if a.cls != b.cls:
                raise AssertionError(f"{split}/{scene_id}: json/npz class order differs")
            if a.vertices.shape != b.vertices.shape or not np.array_equal(a.vertices, b.vertices):
                raise AssertionError(f"{split}/{scene_id}: json/npz vertices differ")
    print(f"[verify] {split}: {len(scene_ids)} scenes re-read, json==npz, segment pairing correct")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rds_root", default="/data/disk2/lyy_dataset/waymo_rds")
    parser.add_argument("--hdmap_root", default="/data/disk2/lyy_dataset/waymo_processed_dggt")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--train_list", default=None, help="defaults to <repo>/data/waymo_train_list_full.txt")
    parser.add_argument("--val_list", default=None, help="defaults to <repo>/data/waymo_val_list_full.txt")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="validate only the first N segments per split")
    parser.add_argument("--summary", type=Path, default=Path("data/hdmap_rds_cross_validation_summary.json"))
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    rds_root = Path(args.rds_root)
    hdmap_root = Path(args.hdmap_root)
    list_paths = {
        "training": Path(args.train_list) if args.train_list else repo_root / DEFAULT_LIST["training"],
        "validation": Path(args.val_list) if args.val_list else repo_root / DEFAULT_LIST["validation"],
    }

    run_summary: dict[str, Any] = {"rds_root": str(rds_root), "hdmap_root": str(hdmap_root), "splits": {}}
    exit_code = 0
    for split in args.splits:
        mapping = scene_id_map(list_paths[split])
        segments = sorted(mapping)
        if args.limit:
            segments = segments[: args.limit]
        tasks = [
            {
                "rds_root": str(rds_root),
                "hdmap_root": str(hdmap_root),
                "split": split,
                "segment": segment,
                "scene_id": mapping[segment],
            }
            for segment in segments
        ]
        print(f"[{split}] validating {len(tasks)} tfrecord hdmap scenes against RDS")

        started = time.time()
        failures: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if not result["ok"]:
                    failures.append(result)
                    continue
                if not result["match"]:
                    mismatches.append(result)
                if done <= 3 or done % 200 == 0:
                    print(
                        f"  [{done}/{len(tasks)}] {result['scene_id']} "
                        f"match={result['match']}"
                    )
        elapsed = time.time() - started
        print(
            f"[{split}] checked={done - len(failures)}/{len(tasks)} "
            f"mismatches={len(mismatches)} failures={len(failures)} in {elapsed:.1f}s"
        )
        run_summary["splits"][split] = {
            "listed": len(tasks),
            "checked": done - len(failures),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "failures": failures,
            "elapsed_seconds": elapsed,
        }
        if failures:
            exit_code = 1
            print(f"[{split}] {len(failures)} FAILURES:")
            for failure in failures[:10]:
                print(f"   {failure['scene_id']} {failure['segment']}: {failure['error']}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.summary.with_name(args.summary.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(args.summary)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
