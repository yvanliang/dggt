#!/usr/bin/env python3
"""Convert legacy FlowDGGT `.pt` caches in-place to chunked zstd containers.

The output keeps the same logical cache schema and `schema_version` as the
source payload.  Only the physical storage changes: each original `.pt` is
atomically replaced by a SQLite container with independently zstd-compressed
chunks, so SceneFlow and tokenizer Stage-B can read only the sampled frame
window.  With `--verify`, the temporary chunked file is compared against the
original before replacement.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
from dggt.utils.flow_cache_io import (
    is_chunked_flow_cache,
    load_chunked_flow_cache_summary,
    load_flow_cache,
    save_flow_cache_chunked,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest_path", help="JSONL manifest whose cache_path entries will be converted.")
    src.add_argument("--cache_root", action="append", help="Repeatable cache root. Scans {root}/{split}/**/*.pt.")
    p.add_argument("--split", default="training")
    p.add_argument("--workers", type=int, default=2, help="Thread workers. Keep modest to avoid RAM spikes.")
    p.add_argument("--zstd_level", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verify", action="store_true", help="Compare converted SceneFlow + Stage-B tensors.")
    p.add_argument("--verify_items", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode_a_source_dir",
        default=None,
        help="Temporary migration helper: read Mode-A files from this directory instead of manifest cache_path.",
    )
    p.add_argument(
        "--mode_a_output_dir",
        default=None,
        help="Temporary migration helper: write converted Mode-A files to this directory instead of overwriting the source.",
    )
    return p


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _entries_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if args.manifest_path:
        for row in _read_manifest(Path(args.manifest_path)):
            path = Path(row["cache_path"])
            mode = str(row.get("mode_kind", "unknown"))
            split = str(row.get("split", args.split))
            output_path = None
            if mode == "mode_a" and args.mode_a_source_dir:
                path = Path(args.mode_a_source_dir) / path.name
            if mode == "mode_a" and args.mode_a_output_dir:
                output_path = Path(args.mode_a_output_dir) / path.name
            entries.append({
                "cache_path": str(path),
                "output_path": None if output_path is None else str(output_path),
                "root": str(path.parent),
                "rel": str(Path(mode) / split / path.name),
                "mode_kind": row.get("mode_kind"),
            })
    else:
        for root_raw in args.cache_root or []:
            root = Path(str(root_raw).rsplit(":", 1)[0])
            split_dir = root / args.split
            scan_root = split_dir if split_dir.is_dir() else root
            for path in sorted(scan_root.rglob("*.pt")):
                entries.append({"cache_path": str(path), "root": str(root), "rel": str(path.relative_to(root))})
    if int(args.limit) > 0:
        entries = entries[: int(args.limit)]
    if not entries:
        raise RuntimeError("No cache files selected for conversion.")
    return entries


def _is_flow_cache_optimized(p: Path) -> bool:
    if not p.is_file() or not is_chunked_flow_cache(p):
        return False
    try:
        with open(p, "rb") as f:
            fast_header = f.read(65536)
        if b"pass1.semantic_logits" in fast_header:
            return True
        summary = load_chunked_flow_cache_summary(p)
        return "pass1.semantic_logits" in set(summary.get("omitted_fields", []))
    except Exception:
        return False


def _convert_one(
    entry: dict[str, Any],
    *,
    zstd_level: int,
    verify_before_replace: bool,
    verify_seed: int,
) -> dict[str, Any]:
    src = Path(entry["cache_path"])
    dst = Path(entry.get("output_path") or src)

    if dst.exists() and _is_flow_cache_optimized(dst):
        return {
            "src": str(src),
            "dst": str(dst),
            "skipped": True,
            "dst_bytes": dst.stat().st_size,
            "reason": "dst_already_chunked",
        }

    if _is_flow_cache_optimized(src):
        if dst != src:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp_copy = dst.with_name(f".{dst.name}.{os.getpid()}.{time.time_ns()}.tmp")
            shutil.copy2(src, tmp_copy)
            os.replace(tmp_copy, dst)
        return {
            "src": str(src),
            "dst": str(dst),
            "skipped": True,
            "dst_bytes": dst.stat().st_size,
            "reason": "src_already_chunked",
        }

    t0 = time.perf_counter()
    src_bytes = src.stat().st_size

    payload = load_flow_cache(src, map_location="cpu", weights_only=False, mmap=False)
    schema_version = int(payload.get("schema_version", 0))
    mode_kind = str(payload.get("mode_kind", ""))

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        save_flow_cache_chunked(payload, tmp, zstd_level=int(zstd_level))
        if verify_before_replace:
            _verify_pair(
                src,
                tmp,
                mode_kind=mode_kind,
                seed=int(verify_seed),
            )
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()

    return {
        "src": str(src),
        "dst": str(dst),
        "skipped": False,
        "schema_version": schema_version,
        "mode_kind": mode_kind,
        "src_bytes": src_bytes,
        "dst_bytes": dst.stat().st_size,
        "verified": bool(verify_before_replace),
        "sec": time.perf_counter() - t0,
    }


def _compare_values(a: Any, b: Any, path: str, errors: list[str]) -> None:
    if len(errors) >= 20:
        return
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            errors.append(f"{path}: tensor/non-tensor mismatch")
            return
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            errors.append(f"{path}: tensor metadata mismatch {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}")
            return
        if a.is_floating_point():
            if not torch.equal(a.cpu(), b.cpu()):
                diff = (a.cpu().float() - b.cpu().float()).abs().max().item() if a.numel() else 0.0
                errors.append(f"{path}: tensor values differ max_abs={diff}")
        elif not torch.equal(a.cpu(), b.cpu()):
            errors.append(f"{path}: tensor values differ")
        return
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            errors.append(f"{path}: dict/non-dict mismatch")
            return
        for key in sorted(set(a.keys()) | set(b.keys()), key=str):
            if key in {"cache_path"}:
                continue
            if key not in a or key not in b:
                errors.append(f"{path}.{key}: missing key")
                continue
            _compare_values(a[key], b[key], f"{path}.{key}", errors)
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            errors.append(f"{path}: sequence mismatch")
            return
        for i, (av, bv) in enumerate(zip(a, b)):
            _compare_values(av, bv, f"{path}[{i}]", errors)
        return
    if hasattr(a, "__dict__") or hasattr(b, "__dict__"):
        if not (hasattr(a, "__dict__") and hasattr(b, "__dict__")):
            errors.append(f"{path}: object mismatch")
            return
        _compare_values(vars(a), vars(b), path, errors)
        return
    if a != b:
        errors.append(f"{path}: {a!r} != {b!r}")


def _semantic_vehicle_from_logits_for_compare(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = value.detach().cpu().float()
    if logits.shape[-1] <= 4:
        shape = tuple(int(v) for v in logits.shape[:-1])
        return torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.bool)
    probs = torch.softmax(logits, dim=-1)
    return probs[..., 4].contiguous(), (probs.argmax(dim=-1) == 4).contiguous()


def _normalize_semantic_for_compare(item: dict[str, Any]) -> dict[str, Any]:
    """Compare the optimized chunked semantic payload, not omitted logits."""
    out = dict(item)
    predictions = out.get("predictions")
    if isinstance(predictions, dict):
        pred = dict(predictions)
        sem = pred.get("semantic_logits")
        if torch.is_tensor(sem):
            if pred.get("semantic_vehicle_prob") is None or pred.get("semantic_vehicle_mask") is None:
                prob, mask = _semantic_vehicle_from_logits_for_compare(sem)
                pred["semantic_vehicle_prob"] = prob
                pred["semantic_vehicle_mask"] = mask
            pred["semantic_logits"] = None
        out["predictions"] = pred
    return out


def _dataset_for_one(path: Path, mode_kind: str | None, seed: int) -> WaymoFlowCacheDataset:
    ds = WaymoFlowCacheDataset(
        cache_root=None,
        manifest_path=_write_temp_manifest(path, mode_kind),
        min_frames=8,
        max_frames=8,
        seed=seed,
        asset_lut_level_indices=(-1,),
    )
    return ds


def _tokenizer_stage_b_item(ds: WaymoFlowCacheDataset, path: Path) -> dict[str, Any]:
    entry = ds.entries[0]
    payload, subset_t, subset_payload = ds._load_payload_for_sample(
        path,
        entry,
        consumer="tokenizer_stage_b",
    )
    return {
        "sample": ds._build_sample(payload, subset_payload),
        "predictions": ds._build_predictions(payload, subset_payload),
        "tokenizer_teacher_levels": ds._subset_pass2_splatted_tok_low(
            payload["pass2_splatted_tok_low"],
            subset_payload,
            dtype=ds.lut_dtype,
        ),
        "mode_kind": str(payload["mode_kind"]),
        "subset_frames": subset_t,
        "cache_schema_version": int(payload["schema_version"]),
    }


def _write_temp_manifest(path: Path, mode_kind: str | None) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    with tmp:
        tmp.write(json.dumps({"cache_path": str(path), "mode_kind": mode_kind or "unknown"}) + "\n")
    return Path(tmp.name)


def _verify_pair(src: Path, dst: Path, *, mode_kind: str | None, seed: int) -> None:
    src_ds = _dataset_for_one(src, mode_kind, seed)
    dst_ds = _dataset_for_one(dst, mode_kind, seed)
    try:
        src_item = _normalize_semantic_for_compare(src_ds[0])
        dst_item = _normalize_semantic_for_compare(dst_ds[0])
        errors: list[str] = []
        _compare_values(src_item, dst_item, "scene_flow", errors)
        src_tok_ds = _dataset_for_one(src, mode_kind, seed)
        dst_tok_ds = _dataset_for_one(dst, mode_kind, seed)
        src_tok = _normalize_semantic_for_compare(_tokenizer_stage_b_item(src_tok_ds, src))
        dst_tok = _normalize_semantic_for_compare(_tokenizer_stage_b_item(dst_tok_ds, dst))
        _compare_values(src_tok, dst_tok, "tokenizer_stage_b", errors)
        if errors:
            raise AssertionError("\n".join(errors))
    finally:
        for ds in (src_ds, dst_ds, locals().get("src_tok_ds"), locals().get("dst_tok_ds")):
            if ds is None:
                continue
            manifest = getattr(ds, "manifest_path", None)
            if manifest is not None:
                Path(manifest).unlink(missing_ok=True)


def main() -> None:
    args = build_argparser().parse_args()
    entries = _entries_from_args(args)

    needs_conversion = []
    print(f"Scanning {len(entries)} files to determine which need conversion...")
    for entry in tqdm(entries, desc="scan"):
        src = Path(entry["cache_path"])
        dst = Path(entry.get("output_path") or src)
        if dst.exists() and _is_flow_cache_optimized(dst):
            continue
        needs_conversion.append(entry)
    
    skipped_count = len(entries) - len(needs_conversion)
    print(f"Scan complete. {skipped_count} files already converted, {len(needs_conversion)} need conversion.")
    entries = needs_conversion

    mode_a_entries = []
    mode_b_entries = []
    other_entries = []

    for e in entries:
        kind = e.get("mode_kind")
        if not kind:
            path_str = e.get("cache_path", "")
            if "mode_b" in path_str:
                kind = "mode_b"
            elif "mode_a" in path_str:
                kind = "mode_a"
        
        if kind == "mode_a":
            mode_a_entries.append(e)
        elif kind == "mode_b":
            mode_b_entries.append(e)
        else:
            other_entries.append(e)

    interleaved = []
    import itertools
    for a, b in itertools.zip_longest(mode_a_entries, mode_b_entries):
        if a is not None:
            interleaved.append(a)
        if b is not None:
            interleaved.append(b)

    entries = interleaved + other_entries

    results: list[dict[str, Any]] = []
    verify_limit = max(0, int(args.verify_items)) if bool(args.verify) else 0
    with futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futs = [
            pool.submit(
                _convert_one,
                entry,
                zstd_level=int(args.zstd_level),
                verify_before_replace=bool(args.verify) and i < verify_limit,
                verify_seed=int(args.seed) + i,
            )
            for i, entry in enumerate(entries)
        ]
        try:
            for fut in tqdm(futures.as_completed(futs), total=len(futs), desc="convert", unit="clip"):
                results.append(fut.result())
        except BaseException:
            for f in futs:
                f.cancel()
            raise

    converted = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    src_bytes = sum(int(r.get("src_bytes", 0)) for r in converted)
    dst_bytes = sum(int(r.get("dst_bytes", 0)) for r in results)
    print(
        f"[done] converted={len(converted)} skipped={len(skipped)} "
        f"source={src_bytes / 1e9:.3f}GB output={dst_bytes / 1e9:.3f}GB",
        flush=True,
    )

    if args.verify:
        verified = sum(1 for r in results if bool(r.get("verified", False)))
        print(
            f"[verify] exact tensor comparison passed before overwrite for {verified} converted caches",
            flush=True,
        )


if __name__ == "__main__":
    main()
