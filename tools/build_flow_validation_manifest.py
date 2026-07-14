"""Build the validation flow-cache manifest.

Walks the canonical
``{cache_root}/{split}/{entry_index:06d}_{edit_name}.pt`` layout produced
by ``tools/precompute_flow_features_validation.py``, peeks each ``.pt``, and
emits a JSONL manifest consumed by
``WaymoFlowCacheDataset(manifest_path=...)``. Legacy numeric filenames remain
readable; if both names exist for one pair, the canonical filename wins.

Only current schema-v9 chunked-zstd SQLite caches are accepted. Legacy caches
or monolithic files are skipped so a manifest cannot silently mix cache
semantics or physical formats.

Each output line (same shape as ``build_flow_train_manifest.py``):

    {"index": <unique>, "mode_kind": "mode_a", "split": "validation",
     "scene_name": "...", "clip_name": "...", "variant": "...",
     "clip_start": 0, "num_frames": 29, "num_objects": <int>,
     "cache_path": "/abs/path/000012_combined.pt"}

The numeric ``index = entry_index * 5 + variant_ord`` is retained only as a
backward-compatible logical manifest index.

Usage:
    python tools/build_flow_validation_manifest.py \
        --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
        --split validation \
        --out_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.flow_cache_io import (
    CURRENT_FLOW_CACHE_SCHEMA_VERSION,
    is_chunked_flow_cache,
    is_current_flow_cache_summary,
    load_chunked_flow_cache_probe,
)
from dggt.utils.validation_cache_naming import (
    VALIDATION_VARIANTS,
    normalize_validation_variant,
    parse_validation_cache_filename,
    validation_cache_filename,
    validation_cache_index,
)

VARIANT_ORDER = list(VALIDATION_VARIANTS)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", required=True,
                   help="Root passed to precompute_flow_features_validation --out_root")
    p.add_argument("--split", default="validation")
    p.add_argument("--out_path", required=True)
    p.add_argument("--variants", default=",".join(VARIANT_ORDER))
    return p


def build_manifest_rows(
    cache_root: str | Path,
    *,
    split: str = "validation",
    variants: list[str] | None = None,
) -> tuple[list[dict], int]:
    variants = [
        normalize_validation_variant(v)
        for v in (VARIANT_ORDER if variants is None else variants)
    ]
    split_root = Path(cache_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"cache split root not found: {split_root}")

    rows_by_key: dict[tuple[int, str], dict] = {}
    n_bad = 0
    for cache_path in sorted(split_root.glob("*.pt")):
        parsed_name = parse_validation_cache_filename(cache_path)
        if parsed_name is None:
            continue
        filename_entry_index, filename_variant = parsed_name
        if not is_chunked_flow_cache(cache_path):
            print(f"[warn] {cache_path} is not a chunked-zstd cache; skipping")
            n_bad += 1
            continue
        try:
            payload = load_chunked_flow_cache_probe(cache_path)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to probe {cache_path}: {e}")
            n_bad += 1
            continue
        if int(payload.get("schema_version", 0)) != CURRENT_FLOW_CACHE_SCHEMA_VERSION:
            print(
                f"[warn] {cache_path} schema_version != "
                f"{CURRENT_FLOW_CACHE_SCHEMA_VERSION}; skipping"
            )
            n_bad += 1
            continue
        meta = payload.get("meta", {})
        summary = payload.get("_chunked_summary", {})
        if not is_current_flow_cache_summary(summary):
            print(f"[warn] {cache_path} is not the current chunked cache format; skipping")
            n_bad += 1
            continue
        mode_kind = str(payload.get("mode_kind", "mode_a"))
        try:
            variant = normalize_validation_variant(
                str(meta.get("variant", filename_variant))
            )
        except ValueError:
            print(f"[warn] {cache_path} has unknown variant; skipping")
            n_bad += 1
            continue
        if variant not in variants:
            continue
        entry_index = int(meta.get("validation_entry_index", filename_entry_index))
        logical_index = validation_cache_index(entry_index, variant)
        row = {
            "index": logical_index,
            "mode_kind": mode_kind,
            "split": split,
            "scene_name": str(meta.get("scene_name", "")),
            "clip_name": str(meta.get("clip_name", "")),
            "variant": variant,
            "validation_entry_index": entry_index,
            "clip_start": 0,
            "num_frames": int(meta.get("num_frames", 29)),
            "num_objects": int(len(summary.get("asset_object_keys", []))),
            "cache_path": str(cache_path.resolve()),
        }
        key = (entry_index, variant)
        previous = rows_by_key.get(key)
        if previous is None:
            rows_by_key[key] = row
        else:
            canonical_name = validation_cache_filename(entry_index, variant)
            if cache_path.name == canonical_name:
                rows_by_key[key] = row
    rows = sorted(rows_by_key.values(), key=lambda row: int(row["index"]))
    return rows, n_bad


def main() -> None:
    args = build_argparser().parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows, n_bad = build_manifest_rows(
        args.cache_root,
        split=args.split,
        variants=variants,
    )

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
