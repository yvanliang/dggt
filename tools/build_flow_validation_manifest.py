"""Build the validation flow-cache manifest.

Walks the FLAT ``{cache_root}/{split}/{index:06d}.pt`` layout produced by
``tools/precompute_flow_features_validation.py`` (``index = entry_index*5 +
variant_ord``), peeks each ``.pt``, and emits a JSONL manifest consumed by
``WaymoFlowCacheDataset(manifest_path=...)``.

Each output line (same shape as ``build_flow_train_manifest.py``):

    {"index": <unique>, "mode_kind": "mode_a", "split": "validation",
     "scene_name": "...", "clip_name": "...", "variant": "...",
     "clip_start": 0, "num_frames": 29, "num_objects": <int>,
     "cache_path": "/abs/path/{index:06d}.pt"}

``index = entry_index * 5 + variant_ord`` keeps every (entry, variant) unique.

Usage:
    python tools/build_flow_validation_manifest.py \
        --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
        --split validation \
        --out_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dggt.utils.flow_cache_io import load_flow_cache

VARIANT_ORDER = ["combined", "delete", "add", "replace", "move"]
VARIANT_ORD = {v: i for i, v in enumerate(VARIANT_ORDER)}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", required=True,
                   help="Root passed to precompute_flow_features_validation --out_root")
    p.add_argument("--split", default="validation")
    p.add_argument("--out_path", required=True)
    p.add_argument("--variants", default=",".join(VARIANT_ORDER))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    split_root = Path(args.cache_root) / args.split
    if not split_root.is_dir():
        raise FileNotFoundError(f"cache split root not found: {split_root}")

    rows: list[dict] = []
    n_bad = 0
    for cache_path in sorted(split_root.glob("*.pt")):
        try:
            file_index = int(cache_path.stem)
        except ValueError:
            continue
        try:
            payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to peek {cache_path}: {e}")
            n_bad += 1
            continue
        if int(payload.get("schema_version", 0)) != 6:
            print(f"[warn] {cache_path} schema_version != 6; skipping")
            n_bad += 1
            continue
        meta = payload.get("meta", {})
        mode_kind = str(payload.get("mode_kind", "mode_a"))
        variant = str(meta.get("variant", VARIANT_ORDER[file_index % len(VARIANT_ORDER)]))
        if variant not in variants:
            continue
        rows.append({
            "index": file_index,
            "mode_kind": mode_kind,
            "split": args.split,
            "scene_name": str(meta.get("scene_name", "")),
            "clip_name": str(meta.get("clip_name", "")),
            "variant": variant,
            "validation_entry_index": int(
                meta.get("validation_entry_index", file_index // len(VARIANT_ORDER))
            ),
            "clip_start": 0,
            "num_frames": int(meta.get("num_frames", 29)),
            "num_objects": int(len(payload.get("asset_pass", {}) or {})),
            "cache_path": str(cache_path.resolve()),
        })

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "num_entries": len(rows),
        "num_bad": n_bad,
        "by_variant": {
            v: sum(1 for r in rows if r["variant"] == v) for v in variants
        },
        "out_path": str(out_path),
    }
    with open(str(out_path) + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {len(rows)} rows -> {out_path}")
    print(f"[done] summary: {summary}")


if __name__ == "__main__":
    main()
