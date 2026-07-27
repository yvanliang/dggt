#!/usr/bin/env python3
"""Migrate legacy Mode-B flow caches to the canonical Gaussian time scale.

The migration is in-place and does not rerun DGGT or the Mode-B planner.  For
non-empty edits it rebuilds the timestamp-dependent pass2 splat and flow-input
chunks from the cached DGGT heads/LUT/delete masks, then updates ``global/meta``
and the summary in the same SQLite transaction.  Empty/no-op edits need only
the metadata update.

By default this command is a dry run.  Pass ``--write`` to commit changes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import gc
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.flow_cache_io import (  # noqa: E402
    CHUNKED_FLOW_CACHE_FORMAT,
    CHUNKED_FLOW_CACHE_FORMAT_VERSION,
    CURRENT_FLOW_CACHE_SCHEMA_VERSION,
    _get_chunk,
    _get_info,
    _put_chunk,
    _put_info,
    _require_zstd_module,
    is_chunked_flow_cache,
    is_current_flow_cache_summary,
    load_flow_cache,
)
from dggt.utils.gaussian_time import (  # noqa: E402
    GAUSSIAN_TIME_REPRESENTATION,
    gaussian_timestamps_from_frame_ids,
)


SUPPORTED_SOURCE_FORMAT_VERSIONS = (1, CHUNKED_FLOW_CACHE_FORMAT_VERSION)
MIGRATION_RECOMPUTED = "mode_b_gaussian_time_pass2_flow_inputs_v1"
MIGRATION_NOOP = "mode_b_gaussian_time_metadata_noop_v1"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cache_root",
        action="append",
        help="Mode-B cache root; repeatable. Scans {root}/{split}/**/*.pt.",
    )
    source.add_argument(
        "--cache_path",
        action="append",
        help="One Mode-B .pt file; repeatable. Useful for smoke tests and retries.",
    )
    source.add_argument(
        "--manifest_path",
        help="JSONL manifest. Only entries with mode_kind=mode_b are selected.",
    )
    parser.add_argument("--split", default="training")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used to rebuild non-empty pass2/flow-input chunks.",
    )
    parser.add_argument("--chunk_channels", type=int, default=64)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Commit the in-place migration. Without this flag, only inspect files.",
    )
    parser.add_argument(
        "--unsafe_no_journal",
        action="store_true",
        help="Disable the small SQLite rollback journal. Faster, but interruption can corrupt a cache.",
    )
    parser.add_argument(
        "--sqlite_quick_check",
        action="store_true",
        help="Run SQLite quick_check for every file (reads the full ~280MB DB and is much slower).",
    )
    parser.add_argument(
        "--out_jsonl",
        default=None,
        help="Optional per-file result log.",
    )
    parser.add_argument("--fail_fast", action="store_true")
    return parser


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mode_kind = str(row.get("mode_kind", "unknown"))
            if mode_kind == "mode_a":
                continue
            if mode_kind not in ("mode_b", "unknown", ""):
                raise ValueError(
                    f"Invalid mode_kind={mode_kind!r} at {path}:{line_number}"
                )
            cache_path = row.get("cache_path")
            if not cache_path:
                raise ValueError(f"Missing cache_path at {path}:{line_number}")
            rows.append({"cache_path": str(cache_path), "mode_kind": mode_kind})
    return rows


def _entries_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest_path:
        entries = _read_manifest(Path(args.manifest_path))
    elif args.cache_path:
        entries = [
            {"cache_path": str(Path(path).expanduser()), "mode_kind": "mode_b"}
            for path in args.cache_path
        ]
    else:
        entries = []
        for root_raw in args.cache_root or []:
            root = Path(root_raw).expanduser()
            split_root = root / str(args.split)
            scan_root = split_root if split_root.is_dir() else root
            entries.extend(
                {"cache_path": str(path), "mode_kind": "unknown"}
                for path in sorted(scan_root.rglob("*.pt"))
            )

    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        resolved = str(Path(entry["cache_path"]).expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduplicated.append({**entry, "cache_path": resolved})
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"Expected num_shards>0 and 0<=shard_index<num_shards, got "
            f"num_shards={num_shards}, shard_index={shard_index}"
        )
    deduplicated = deduplicated[shard_index::num_shards]
    if int(args.start) > 0:
        deduplicated = deduplicated[int(args.start) :]
    if int(args.limit) > 0:
        deduplicated = deduplicated[: int(args.limit)]
    if not deduplicated:
        raise RuntimeError("No Mode-B cache files were selected.")
    return deduplicated


def _canonical_timestamps(meta: dict[str, Any], summary: dict[str, Any]) -> torch.Tensor:
    num_frames = int(summary.get("num_frames", 0))
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames={num_frames}")
    if int(meta.get("num_frames", -1)) != num_frames:
        raise ValueError(
            f"global/meta.num_frames={meta.get('num_frames')!r} disagrees with "
            f"summary.num_frames={num_frames}"
        )

    timestamps = meta.get("timestamps")
    if not torch.is_tensor(timestamps) or timestamps.ndim != 1:
        raise ValueError("global/meta.timestamps must be a 1-D tensor")
    if int(timestamps.numel()) != num_frames:
        raise ValueError(
            f"global/meta.timestamps has {timestamps.numel()} values, expected {num_frames}"
        )

    cam_ids = meta.get("cam_ids")
    if not torch.is_tensor(cam_ids) or int(cam_ids.numel()) != 1:
        raise ValueError(
            "This migration only supports the existing single-camera Mode-B cache; "
            f"got cam_ids={cam_ids!r}"
        )

    frame_indices = meta.get("frame_indices_scene")
    if not torch.is_tensor(frame_indices) or frame_indices.ndim != 1:
        raise ValueError("global/meta.frame_indices_scene must be a 1-D tensor")
    if int(frame_indices.numel()) != num_frames:
        raise ValueError(
            f"global/meta.frame_indices_scene has {frame_indices.numel()} values, "
            f"expected {num_frames}"
        )
    if num_frames > 1 and not torch.equal(
        frame_indices[1:].to(torch.long) - frame_indices[:-1].to(torch.long),
        torch.ones(num_frames - 1, dtype=torch.long),
    ):
        raise ValueError(
            "frame_indices_scene is not a contiguous single-camera clip; refusing "
            "to infer clip-local Gaussian timestamps"
        )

    # Do not use frame_indices_scene directly: it is scene-global (e.g. 87..115).
    # Chunk positions are the authoritative clip-local frame ids (0..S-1).
    return gaussian_timestamps_from_frame_ids(torch.arange(num_frames, dtype=torch.long))


def _validate_container(summary: dict[str, Any], *, path: Path) -> None:
    if str(summary.get("format", "")) != CHUNKED_FLOW_CACHE_FORMAT:
        raise ValueError(f"Unsupported cache format in {path}: {summary.get('format')!r}")
    source_version = int(summary.get("format_version", 0))
    if source_version not in SUPPORTED_SOURCE_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported chunked format_version={source_version} in {path}; "
            f"expected one of {SUPPORTED_SOURCE_FORMAT_VERSIONS}"
        )
    schema_version = int(summary.get("schema_version", 0))
    if schema_version != CURRENT_FLOW_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={summary.get('schema_version')!r} in {path}; "
            f"expected {CURRENT_FLOW_CACHE_SCHEMA_VERSION}; regenerate old caches"
        )
    if str(summary.get("mode_kind", "")) != "mode_b":
        raise ValueError(
            f"Refusing to migrate non-Mode-B cache {path}: "
            f"mode_kind={summary.get('mode_kind')!r}"
        )
    if summary.get("asset_object_keys") not in (None, []):
        raise ValueError(
            f"Mode-B cache {path} unexpectedly contains asset_object_keys; "
            "format-v1 to v2 promotion is not safe"
        )


def inspect_mode_b_cache(
    path: str | os.PathLike[str],
    *,
    sqlite_quick_check: bool = False,
) -> dict[str, Any]:
    """Inspect one cache without modifying it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not is_chunked_flow_cache(path):
        raise ValueError(f"Not a chunked SQLite flow cache: {path}")

    zstd = _require_zstd_module()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        summary = _get_info(conn, "summary")
        _validate_container(summary, path=path)
        meta = _get_chunk(conn, zstd.ZstdDecompressor(), "global/meta")
        expected = _canonical_timestamps(meta, summary)
        actual = meta["timestamps"].detach().cpu().to(torch.float32)
        marker_ok = (
            str(meta.get("gaussian_time_representation", ""))
            == GAUSSIAN_TIME_REPRESENTATION
            and str(summary.get("gaussian_time_representation", ""))
            == GAUSSIAN_TIME_REPRESENTATION
        )
        timestamps_ok = torch.equal(actual, expected)
        schema_ok = int(summary.get("schema_version", 0)) == CURRENT_FLOW_CACHE_SCHEMA_VERSION
        format_ok = (
            str(summary.get("format", "")) == CHUNKED_FLOW_CACHE_FORMAT
            and int(summary.get("format_version", 0)) == CHUNKED_FLOW_CACHE_FORMAT_VERSION
        )
        current = schema_ok and format_ok and marker_ok and timestamps_ok
        target_has_delete = summary.get("mode_b_target_has_delete")
        if isinstance(target_has_delete, list):
            timestamp_dependent_pass2 = any(bool(value) for value in target_has_delete)
        else:
            timestamp_dependent_pass2 = int(summary.get("mode_b_num_imagined_objects", 0)) > 0
        requires_recompute = (not current) and bool(timestamp_dependent_pass2)
        if sqlite_quick_check:
            integrity = conn.execute("PRAGMA quick_check").fetchone()
            if not integrity or str(integrity[0]).lower() != "ok":
                raise RuntimeError(f"SQLite quick_check failed for {path}: {integrity!r}")
        return {
            "path": str(path),
            "status": "already_current" if current else "needs_migration",
            "source_format_version": int(summary["format_version"]),
            "target_format_version": CHUNKED_FLOW_CACHE_FORMAT_VERSION,
            "num_frames": int(summary["num_frames"]),
            "old_timestamp_first": float(actual[0].item()),
            "old_timestamp_last": float(actual[-1].item()),
            "new_timestamp_first": float(expected[0].item()),
            "new_timestamp_last": float(expected[-1].item()),
            "marker_ok": bool(marker_ok),
            "timestamps_ok": bool(timestamps_ok),
            "requires_pass2_recompute": bool(requires_recompute),
            "timestamp_dependent_pass2": bool(timestamp_dependent_pass2),
            "migration_kind": summary.get("gaussian_time_migration"),
            "bytes": int(path.stat().st_size),
        }


def _reader_stub():
    from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset

    reader = object.__new__(WaymoFlowCacheDataset)
    # gsplat's float32 Gaussian geometry requires float32 feature colors here.
    # The training reader may dequantize to fp16 after caching, but pass2
    # regeneration must match the original precompute rasterization path.
    reader.lut_dtype = torch.float32
    reader.include_aux_tokens = False
    reader.asset_lut_level_indices = None
    reader.mmap_plain_cache = False
    return reader


def _recompute_timestamp_dependent_payload(
    payload: dict[str, Any],
    *,
    device: torch.device,
    chunk_channels: int,
) -> dict[str, Any]:
    """Rebuild Mode-B pass2 and flow inputs from cached, timestamp-free inputs."""
    from dggt.utils.gaussian_edit import build_clean_scene_state
    from tools.precompute_flow_features import (
        _compute_and_pack_pass2_splatted_tok_low_mode_b,
    )

    meta = payload["meta"]
    num_frames = int(meta["num_frames"])
    subset = torch.arange(num_frames, dtype=torch.long)
    reader = _reader_stub()
    sample = reader._build_sample(payload, subset)
    predictions_cpu = reader._build_predictions(payload, subset)
    clean_state = build_clean_scene_state(sample, predictions_cpu)
    cameras_dggt = reader._build_cameras_dggt(payload, subset)
    mode_b_payload = reader._build_mode_b(payload, subset)

    # The helper needs only these prediction fields after clean_state exists.
    # Keeping semantic/GS heads on CPU materially reduces peak GPU memory.
    predictions_compute = {
        "image_tokens_levels": [
            level.to(device=device, non_blocking=False)
            for level in predictions_cpu["image_tokens_levels"]
        ],
        "dynamic_conf": predictions_cpu["dynamic_conf"].to(device),
        "depth": predictions_cpu["depth"].to(device),
        "patch_start_idx": int(predictions_cpu["patch_start_idx"]),
    }
    images = sample["images_clean"]
    result = _compute_and_pack_pass2_splatted_tok_low_mode_b(
        sample=sample,
        predictions=predictions_compute,
        cameras_dggt=cameras_dggt,
        clean_state=clean_state,
        mode_b_payload=mode_b_payload,
        patch_grid=tuple(int(value) for value in meta["patch_grid"]),
        H_img=int(images.shape[-2]),
        W_img=int(images.shape[-1]),
        chunk_channels=int(chunk_channels),
        device=device,
    )
    return result


def _validate_recomputed_pass2(pass2: dict[str, Any], *, num_frames: int) -> None:
    data = pass2.get("splatted_tok_low_int8")
    scale = pass2.get("splatted_tok_low_scale")
    if not torch.is_tensor(data) or data.dtype != torch.int8 or int(data.shape[0]) != num_frames:
        raise ValueError(
            "Recomputed splatted_tok_low_int8 must be int8 with first dim "
            f"{num_frames}, got {None if data is None else (data.dtype, tuple(data.shape))}"
        )
    if not torch.is_tensor(scale) or int(scale.shape[0]) != num_frames:
        raise ValueError("Recomputed splatted_tok_low_scale has an invalid frame dimension")
    flow_inputs = pass2.get("flow_inputs")
    if not isinstance(flow_inputs, dict):
        raise ValueError("Recomputed pass2 is missing flow_inputs")
    for key in ("M_preserve", "M_source", "M_dest", "scaffold_pooled"):
        value = flow_inputs.get(key)
        if not torch.is_tensor(value) or int(value.shape[0]) != num_frames:
            raise ValueError(f"Recomputed flow_inputs[{key!r}] has an invalid frame dimension")


def migrate_mode_b_cache_inplace(
    path: str | os.PathLike[str],
    *,
    unsafe_no_journal: bool = False,
    sqlite_quick_check: bool = False,
    device: str | torch.device = "cuda",
    chunk_channels: int = 64,
) -> dict[str, Any]:
    """Transactionally migrate one Mode-B cache in place."""
    started = time.perf_counter()
    before = inspect_mode_b_cache(path, sqlite_quick_check=sqlite_quick_check)
    if before["status"] == "already_current":
        return before

    path = Path(path)
    zstd = _require_zstd_module()
    num_frames = int(before["num_frames"])
    recomputed_pass2: dict[str, Any] | None = None
    if bool(before["requires_pass2_recompute"]):
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for non-empty Mode-B pass2 migration but is unavailable")
        payload = load_flow_cache(path, map_location="cpu", weights_only=False, mmap=False)
        payload["meta"] = dict(payload["meta"])
        payload["meta"]["timestamps"] = gaussian_timestamps_from_frame_ids(
            torch.arange(num_frames, dtype=torch.long)
        )
        payload["meta"]["gaussian_time_representation"] = GAUSSIAN_TIME_REPRESENTATION
        recomputed_pass2 = _recompute_timestamp_dependent_payload(
            payload,
            device=device_obj,
            chunk_channels=int(chunk_channels),
        )
        _validate_recomputed_pass2(recomputed_pass2, num_frames=num_frames)
        del payload

    conn = sqlite3.connect(str(path), timeout=60.0)
    try:
        if unsafe_no_journal:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
        else:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")

        summary = _get_info(conn, "summary")
        _validate_container(summary, path=path)
        meta = _get_chunk(conn, zstd.ZstdDecompressor(), "global/meta")
        expected = _canonical_timestamps(meta, summary)
        meta = dict(meta)
        meta["timestamps"] = expected
        meta["gaussian_time_representation"] = GAUSSIAN_TIME_REPRESENTATION
        _put_chunk(conn, zstd.ZstdCompressor(level=1), "global/meta", meta)

        if recomputed_pass2 is not None:
            cctx = zstd.ZstdCompressor(level=1)
            flow_inputs = recomputed_pass2["flow_inputs"]
            for frame in range(num_frames):
                _put_chunk(
                    conn,
                    cctx,
                    f"frame/{frame:02d}/pass2",
                    {
                        "splatted_tok_low_int8": recomputed_pass2[
                            "splatted_tok_low_int8"
                        ][frame],
                        "splatted_tok_low_scale": recomputed_pass2[
                            "splatted_tok_low_scale"
                        ][frame],
                    },
                )
                _put_chunk(
                    conn,
                    cctx,
                    f"frame/{frame:02d}/flow_inputs",
                    {
                        "M_preserve": flow_inputs["M_preserve"][frame],
                        "M_source": flow_inputs["M_source"][frame],
                        "M_dest": flow_inputs["M_dest"][frame],
                        "scaffold_pooled": flow_inputs["scaffold_pooled"][frame],
                    },
                )

        summary = dict(summary)
        summary["format_version"] = CHUNKED_FLOW_CACHE_FORMAT_VERSION
        summary["gaussian_time_representation"] = GAUSSIAN_TIME_REPRESENTATION
        summary["gaussian_time_migration"] = (
            MIGRATION_RECOMPUTED if recomputed_pass2 is not None else MIGRATION_NOOP
        )
        if recomputed_pass2 is not None:
            summary["has_flow_inputs"] = True
            coverage = recomputed_pass2["flow_inputs"].get("phase1_coverage")
            if torch.is_tensor(coverage):
                summary["flow_inputs_phase1_coverage_shape"] = list(coverage.shape)
        _put_info(conn, "summary", summary)

        row = conn.execute(
            "SELECT COUNT(*), SUM(zbytes), SUM(raw_bytes) FROM chunks"
        ).fetchone()
        _put_info(
            conn,
            "stats",
            {
                "chunk_count": int(row[0] or 0),
                "chunk_zbytes": int(row[1] or 0),
                "chunk_raw_torch_bytes": int(row[2] or 0),
            },
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    del recomputed_pass2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    after = inspect_mode_b_cache(path, sqlite_quick_check=sqlite_quick_check)
    if after["status"] != "already_current":
        raise RuntimeError(f"Post-migration verification failed for {path}: {after}")
    after.update(
        {
            "status": "migrated",
            "source_format_version": before["source_format_version"],
            "old_timestamp_first": before["old_timestamp_first"],
            "old_timestamp_last": before["old_timestamp_last"],
            "pass2_recomputed": bool(before["requires_pass2_recompute"]),
            "sec": time.perf_counter() - started,
        }
    )
    return after


def _run_one(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    path = entry["cache_path"]
    try:
        if args.write:
            return migrate_mode_b_cache_inplace(
                path,
                unsafe_no_journal=bool(args.unsafe_no_journal),
                sqlite_quick_check=bool(args.sqlite_quick_check),
                device=str(args.device),
                chunk_channels=int(args.chunk_channels),
            )
        return inspect_mode_b_cache(
            path,
            sqlite_quick_check=bool(args.sqlite_quick_check),
        )
    except Exception as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = build_argparser().parse_args()
    if (
        bool(args.write)
        and torch.device(str(args.device)).type == "cuda"
        and int(args.workers) != 1
    ):
        raise ValueError("CUDA write migration requires --workers 1 to avoid GPU OOM/races")
    entries = _entries_from_args(args)
    print(
        f"[mode-b-cache-migration] selected={len(entries)} "
        f"write={bool(args.write)} workers={max(1, int(args.workers))} "
        f"shard={int(args.shard_index)}/{int(args.num_shards)}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        pending = {pool.submit(_run_one, entry, args): entry for entry in entries}
        for future in tqdm(
            futures.as_completed(pending),
            total=len(pending),
            desc="migrate" if args.write else "inspect",
            unit="clip",
        ):
            result = future.result()
            results.append(result)
            if result["status"] == "error" and args.fail_fast:
                for other in pending:
                    other.cancel()
                break

    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for result in sorted(results, key=lambda item: item["path"]):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"[done] {json.dumps(counts, sort_keys=True)}", flush=True)
    errors = [result for result in results if result["status"] == "error"]
    for result in errors[:20]:
        print(f"[error] {result['path']}: {result['error']}", file=sys.stderr)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
