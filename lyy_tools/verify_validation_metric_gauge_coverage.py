#!/usr/bin/env python3
"""Verify formal validation clips use explicit, finite effective scene gauges."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.scene_gauge import (
    effective_scene_gauge,
    load_scene_gauge_lookup,
    scene_gauge_valid_channel_mean,
)
from pointcloud_validation.toolkits.waymo_name_index import val_name2index


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final_info_path",
        default="data/final_info_validation.json",
    )
    parser.add_argument(
        "--scene_gauge_path",
        default="data/scene_gauge/validation.json",
    )
    parser.add_argument("--output_json", default=None)
    return parser


def verify_validation_metric_gauge_coverage(
    final_info_path: str | Path,
    scene_gauge_path: str | Path,
) -> dict:
    rows = json.loads(Path(final_info_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("final_info_validation must be a non-empty JSON list")
    _, table_sha256, dggt_sha256, lookup = load_scene_gauge_lookup(
        scene_gauge_path,
        expected_split="validation",
    )
    valid_mean = scene_gauge_valid_channel_mean(lookup)
    missing_keys = []
    invalid_clips = []
    clip_records = []
    total_fallback_channels = 0
    for row_index, row in enumerate(rows):
        clip_name = str(row["clip_name"])
        segment, clip_index_text = clip_name.rsplit("_", 1)
        if segment not in val_name2index:
            raise KeyError(f"validation segment is not indexed: {segment}")
        table_key = f"{int(val_name2index[segment]):03d}/{int(clip_index_text)}"
        entry = lookup.get(table_key)
        if entry is None:
            missing_keys.append(table_key)
            continue
        raw, valid = entry
        effective, fallback_mask = effective_scene_gauge(raw, valid, valid_mean)
        if not all(math.isfinite(value) for value in effective):
            raise AssertionError(f"non-finite effective gauge for {clip_name}")
        if tuple(fallback_mask) != tuple(not flag for flag in valid):
            raise AssertionError(f"fallback mask mismatch for {clip_name}")
        fallback_count = sum(bool(flag) for flag in fallback_mask)
        total_fallback_channels += fallback_count
        if fallback_count:
            invalid_clips.append(clip_name)
        clip_records.append(
            {
                "row_index": row_index,
                "clip_name": clip_name,
                "table_key": table_key,
                "raw_valid": list(valid),
                "fallback_mask": list(fallback_mask),
                "effective_gauge": list(effective),
            }
        )
    if missing_keys:
        raise AssertionError(
            f"validation gauge table is missing {len(missing_keys)} clip keys: "
            f"{missing_keys[:10]}"
        )
    if len(clip_records) != len(rows):
        raise AssertionError("not every validation row produced an effective gauge")
    return {
        "schema": "validation_metric_gauge_coverage_v1",
        "status": "pass",
        "clip_count": len(rows),
        "unique_table_key_count": len({record["table_key"] for record in clip_records}),
        "invalid_raw_clip_count": len(invalid_clips),
        "fallback_channel_count": total_fallback_channels,
        "invalid_raw_clips": invalid_clips,
        "valid_channel_mean": list(valid_mean),
        "scene_gauge_table_sha256": table_sha256,
        "dggt_checkpoint_sha256": dggt_sha256,
        "clips": clip_records,
    }


def main() -> None:
    args = build_argparser().parse_args()
    result = verify_validation_metric_gauge_coverage(
        args.final_info_path,
        args.scene_gauge_path,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
