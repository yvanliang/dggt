"""Recompress FlowDGGT cache files without changing manifest paths.

This is intended for converting existing gzip-wrapped ``.pt`` caches to zstd.
The output file keeps the same path and extension by default; readers detect
the compression from magic bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.flow_cache_io import is_gzip_file, is_zstd_file, load_flow_cache, save_flow_cache


def _compression_label(path: Path) -> str:
    if is_zstd_file(path):
        return "zstd"
    if is_gzip_file(path):
        return "gzip"
    return "other"


def _read_manifest(path: Path) -> list[Path]:
    out: list[Path] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(Path(row["cache_path"]))
    return out


def _list_paths(args: argparse.Namespace) -> list[Path]:
    if args.manifest_path:
        paths = _read_manifest(Path(args.manifest_path))
    else:
        root = Path(args.cache_root)
        split_root = root / args.split
        search_root = split_root if split_root.is_dir() else root
        paths = sorted(search_root.rglob("*.pt"))
    if args.start is not None or args.end is not None:
        paths = paths[args.start:args.end]
    if args.max_files is not None:
        paths = paths[:args.max_files]
    return paths


def _convert_one(path_str: str, compression: str, gzip_level: int, force: bool) -> dict[str, Any]:
    path = Path(path_str)
    before = 0
    source = "unknown"
    try:
        before = path.stat().st_size
        source = _compression_label(path)
        if not force and compression in ("zstd", "zst") and source == "zstd":
            return {
                "path": str(path),
                "skipped": True,
                "error": False,
                "before": before,
                "after": before,
                "source": source,
            }
        payload = load_flow_cache(path, map_location="cpu", weights_only=False)
        save_flow_cache(payload, path, compression=compression, gzip_level=gzip_level)
        after = path.stat().st_size
        return {
            "path": str(path),
            "skipped": False,
            "error": False,
            "before": before,
            "after": after,
            "source": source,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "skipped": False,
            "error": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "before": before,
            "after": before,
            "source": source,
        }


def _format_result(index: int, total: int, result: dict[str, Any]) -> str:
    if result.get("error"):
        status = f"error:{result.get('error_type', 'Exception')}"
    else:
        status = "skip" if result["skipped"] else "ok"
    return (
        f"[{index}/{total}] {status} source={result['source']} "
        f"{result['path']} "
        f"{result['before'] / 1024**2:.1f} -> {result['after'] / 1024**2:.1f} MiB"
        + (f" message={result['error_message']}" if result.get("error") else "")
    )


def _append_error(error_log: str | None, result: dict[str, Any]) -> None:
    if not error_log:
        return
    with Path(error_log).open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest_path")
    src.add_argument("--cache_root")
    p.add_argument("--split", default="training")
    p.add_argument("--compression", choices=["zstd", "gzip"], default="zstd")
    p.add_argument("--gzip_level", type=int, default=1,
                   help="Compression level. For zstd this is the zstd level.")
    p.add_argument("--workers", type=int, default=1,
                   help="Keep low: each worker temporarily materializes one full cache payload.")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--error_log", default="recompress_errors.jsonl")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    paths = _list_paths(args)
    print(f"[recompress] files={len(paths)} compression={args.compression} workers={args.workers}", flush=True)
    if args.dry_run:
        for path in paths[:20]:
            print(path)
        if len(paths) > 20:
            print(f"... {len(paths) - 20} more")
        return

    total_before = 0
    total_after = 0
    converted = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    if args.error_log:
        Path(args.error_log).write_text("")
    if int(args.workers) <= 1:
        iterator = (
            _convert_one(str(path), args.compression, int(args.gzip_level), bool(args.force))
            for path in paths
        )
        progress = tqdm(iterator, total=len(paths), desc="recompress", unit="file")
        for i, result in enumerate(progress, start=1):
            total_before += int(result["before"])
            total_after += int(result["after"])
            errors.append(result) if result.get("error") else None
            _append_error(args.error_log, result) if result.get("error") else None
            skipped += 1 if result["skipped"] else 0
            converted += 1 if (not result["skipped"] and not result.get("error")) else 0
            progress.set_postfix(
                converted=converted,
                skipped=skipped,
                errors=len(errors),
                size=f"{total_before / 1024**3:.1f}->{total_after / 1024**3:.1f}GiB",
            )
            if result.get("error") or (
                args.log_every > 0 and (i == 1 or i % int(args.log_every) == 0 or i == len(paths))
            ):
                print(_format_result(i, len(paths), result), flush=True)
    else:
        ex = ProcessPoolExecutor(max_workers=int(args.workers))
        try:
            futures = [
                ex.submit(_convert_one, str(path), args.compression, int(args.gzip_level), bool(args.force))
                for path in paths
            ]
            progress = tqdm(as_completed(futures), total=len(futures), desc="recompress", unit="file")
            for i, fut in enumerate(progress, start=1):
                result = fut.result()
                total_before += int(result["before"])
                total_after += int(result["after"])
                errors.append(result) if result.get("error") else None
                _append_error(args.error_log, result) if result.get("error") else None
                skipped += 1 if result["skipped"] else 0
                converted += 1 if (not result["skipped"] and not result.get("error")) else 0
                progress.set_postfix(
                    converted=converted,
                    skipped=skipped,
                    errors=len(errors),
                    size=f"{total_before / 1024**3:.1f}->{total_after / 1024**3:.1f}GiB",
                )
                if result.get("error") or (
                    args.log_every > 0 and (i == 1 or i % int(args.log_every) == 0 or i == len(paths))
                ):
                    print(_format_result(i, len(paths), result), flush=True)
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            ex.shutdown(wait=True)

    if errors and args.error_log:
        print(f"[errors] wrote {len(errors)} failed files to {Path(args.error_log)}", flush=True)

    print(
        f"[done] converted={converted} skipped={skipped} errors={len(errors)} "
        f"total={total_before / 1024**3:.2f} -> "
        f"{total_after / 1024**3:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
