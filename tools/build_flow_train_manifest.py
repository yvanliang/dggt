"""Merge Mode A and Mode B feature caches into one diffusion training manifest.

Walks one or more cache roots (each populated by `tools/precompute_flow_features.py`),
peeks each `.pt` to extract `mode_kind` + clip metadata, and emits a JSONL
manifest the diffusion dataloader (`WaymoFlowCacheDataset(manifest_path=...)`)
consumes.

Each output line:
    {
      "index": <manifest/cache index>,
      "mode_kind": "mode_a" | "mode_b",
      "split": "training" | "validation",
      "scene_name": "<scene>",
      "clip_name": "<clip>",
      "clip_start": <int>,
      "num_frames": 29,
      "num_objects": <int>,         # mode_a: len(asset_pass); mode_b: imagined count
      "cache_path": "/abs/path/to/clip.pt"
    }

Usage:
    python tools/build_flow_train_manifest.py \
        --cache_root /data/flow_cache_mode_a:mode_a \
        --cache_root /data/flow_cache_mode_b:mode_b \
        --split training \
        --out_path /data/flow_cache/training_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.flow_cache_io import (
    is_chunked_flow_cache,
    load_chunked_flow_cache_probe,
    load_chunked_flow_cache_summary,
    load_flow_cache,
)


def parse_cache_root(arg: str) -> tuple[Path, str]:
    """Accept `path` or `path:mode_kind` (mode_kind ∈ {mode_a, mode_b, auto})."""
    if ":" in arg:
        path_str, mode = arg.rsplit(":", 1)
        if mode not in ("mode_a", "mode_b", "auto"):
            raise ValueError(f"--cache_root suffix must be mode_a|mode_b|auto, got '{mode}'")
        return Path(path_str), mode
    return Path(arg), "auto"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache_root",
        action="append",
        required=True,
        help="Repeatable. Format: '/path' or '/path:mode_a' or '/path:mode_b'. "
             "When suffix is 'auto' (default), the script reads each .pt's "
             "`mode_kind` field. When pinned, the suffix wins (faster — no peek).",
    )
    p.add_argument("--split", default="training", choices=["training", "validation"])
    p.add_argument("--out_path", required=True)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any unreadable .pt; default behavior is to log + skip.",
    )
    p.add_argument(
        "--peek_full",
        action="store_true",
        help="Open every .pt to read full metadata. Slow but produces num_objects.",
    )
    return p


def _to_int(value: Any, default: int = -1) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _peek_clip(path: Path) -> dict[str, Any] | None:
    if is_chunked_flow_cache(path):
        try:
            summary = load_chunked_flow_cache_summary(path)
            probe = load_chunked_flow_cache_probe(path)
        except Exception as e:
            print(f"[warn] failed to read chunked header {path}: {e}", file=sys.stderr)
            return None
        mode_kind = str(summary.get("mode_kind", "mode_a"))
        meta = probe.get("meta") or {}
        # The cheap header is enough for mode and counts; detailed scene names
        # still come from the filesystem unless --peek_full is used on legacy
        # monolithic files.
        num_objects = (
            len(summary.get("asset_object_keys", []))
            if mode_kind == "mode_a"
            else int(
                summary.get(
                    "mode_b_num_imagined_objects",
                    max(0, sum(1 for v in summary.get("mode_b_target_has_delete", []) if bool(v))),
                )
            )
        )
        return {
            "mode_kind": mode_kind,
            "index": _to_int(meta.get("manifest_index", -1)),
            "dataset_index": _to_int(meta.get("dataset_index", -1)),
            "scene_name": str(meta.get("scene_name", path.parent.name)),
            "clip_name": str(meta.get("clip_name", path.stem)),
            "clip_start": int(meta.get("frame_indices_scene", torch.tensor([0]))[0].item())
                if torch.is_tensor(meta.get("frame_indices_scene"))
                else (int(path.stem) if path.stem.isdigit() else 0),
            "num_frames": int(meta.get("num_frames", summary.get("num_frames", 29))),
            "num_objects": num_objects,
        }
    try:
        payload = load_flow_cache(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return None
    meta = payload.get("meta") or {}
    mode_kind = str(payload.get("mode_kind", "mode_a"))
    if mode_kind == "mode_b":
        block = payload.get("mode_b") or {}
        num_objects = int(block.get("num_imagined_objects", 0))
    else:
        num_objects = len(payload.get("asset_pass", {}) or {})
    return {
        "mode_kind": mode_kind,
        "index": _to_int(meta.get("manifest_index", -1)),
        "dataset_index": _to_int(meta.get("dataset_index", -1)),
        "scene_name": str(meta.get("scene_name", path.parent.name)),
        "clip_name": str(meta.get("clip_name", path.stem)),
        "clip_start": int(meta.get("frame_indices_scene", torch.tensor([0]))[0].item())
            if torch.is_tensor(meta.get("frame_indices_scene"))
            else int(path.stem),
        "num_frames": int(meta.get("num_frames", 29)),
        "num_objects": num_objects,
    }


def main() -> None:
    args = build_argparser().parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary = {
        "cache_roots": [],
        "split": args.split,
        "num_clips_total": 0,
        "num_mode_a": 0,
        "num_mode_b": 0,
    }

    for raw_root in args.cache_root:
        root_path, mode_pin = parse_cache_root(raw_root)
        split_dir = root_path / args.split
        if not split_dir.is_dir():
            print(f"[warn] {split_dir} not found, skipping", file=sys.stderr)
            continue
        cache_files = sorted(split_dir.rglob("*.pt"))
        print(f"[scan] root={root_path} mode_pin={mode_pin} files={len(cache_files)}")
        summary["cache_roots"].append({
            "path": str(root_path),
            "mode_pin": mode_pin,
            "num_clips": len(cache_files),
        })
        for clip_path in cache_files:
            row: dict[str, Any] = {"cache_path": str(clip_path), "split": args.split}
            if mode_pin in ("mode_a", "mode_b") and not args.peek_full:
                # Cheap path: pin mode + derive scene/clip from filesystem.
                rel = clip_path.relative_to(split_dir)
                parts = rel.parts
                try:
                    cache_index = int(clip_path.stem)
                except ValueError:
                    cache_index = -1
                row["mode_kind"] = mode_pin
                row["index"] = int(cache_index)
                row["scene_name"] = parts[0] if len(parts) >= 2 else ""
                row["clip_name"] = clip_path.stem
                try:
                    row["clip_start"] = int(clip_path.stem)
                except ValueError:
                    row["clip_start"] = 0
                row["num_frames"] = 29
                row["num_objects"] = -1
            else:
                peeked = _peek_clip(clip_path)
                if peeked is None:
                    if args.strict:
                        raise RuntimeError(f"Failed to read {clip_path}")
                    continue
                row.update(peeked)
                if mode_pin in ("mode_a", "mode_b") and row["mode_kind"] != mode_pin:
                    print(
                        f"[warn] mode_pin={mode_pin} mismatches payload mode_kind="
                        f"{row['mode_kind']} for {clip_path}; trusting payload."
                    )
            rows.append(row)
            if row["mode_kind"] == "mode_a":
                summary["num_mode_a"] += 1
            elif row["mode_kind"] == "mode_b":
                summary["num_mode_b"] += 1

    summary["num_clips_total"] = len(rows)
    rows.sort(key=lambda r: (r["mode_kind"], int(r.get("index", -1)), r["scene_name"], r["clip_name"]))

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(rows)} entries to {out_path}")
    print(f"[done] summary at {summary_path}")


if __name__ == "__main__":
    main()
