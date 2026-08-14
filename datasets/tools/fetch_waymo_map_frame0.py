#!/usr/bin/env python
"""Fetch and validate the first TFRecord record of every Waymo segment.

The source is the original Perception v1.4.x data on ``ssh 13``.  Exactly the
first 12 MiB of each segment is retained as a reusable local ``.head`` cache;
only the first record is parsed and no ``tf.data`` pipeline is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


HEAD_BYTES = 12 * 1024 * 1024
TFRECORD_HEADER_BYTES = 12
TFRECORD_FOOTER_BYTES = 4
DEFAULT_REMOTE_ROOT = "/data/liangyiyuan/waymo"
DEFAULT_CACHE_ROOT = "/data/disk2/lyy_dataset/waymo_tfrecord_frame0_cache"


class TruncatedRecordError(ValueError):
    """The cached prefix cannot hold the complete first TFRecord record."""


def read_first_record(path: str | Path) -> bytes:
    """Read one raw TFRecord payload, failing closed on a truncated cache."""
    path = Path(path)
    with open(path, "rb") as handle:
        header = handle.read(TFRECORD_HEADER_BYTES)
        if len(header) != TFRECORD_HEADER_BYTES:
            raise TruncatedRecordError(
                f"{path}: need {TFRECORD_HEADER_BYTES} header bytes, got {len(header)}"
            )
        record_length = struct.unpack("<Q", header[:8])[0]
        required = TFRECORD_HEADER_BYTES + record_length + TFRECORD_FOOTER_BYTES
        if required > HEAD_BYTES:
            raise TruncatedRecordError(
                f"{path}: first record requires {required} bytes, beyond {HEAD_BYTES}-byte cache"
            )
        payload = handle.read(record_length)
        footer = handle.read(TFRECORD_FOOTER_BYTES)
        if len(payload) != record_length or len(footer) != TFRECORD_FOOTER_BYTES:
            actual = TFRECORD_HEADER_BYTES + len(payload) + len(footer)
            raise TruncatedRecordError(
                f"{path}: first record requires {required} bytes, cache has only {actual}"
            )
    return payload


def _frame_from_payload(payload: bytes) -> Any:
    # dataset_pb2 depends only on google.protobuf.  Importing tensorflow or
    # constructing tf.data here is both unnecessary and known to hang.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        from waymo_open_dataset import dataset_pb2
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "waymo_open_dataset is required for local parsing; run in conda env chatsim"
        ) from exc
    frame = dataset_pb2.Frame()
    frame.ParseFromString(payload)
    return frame


def inspect_cache(path: str | Path) -> dict[str, Any]:
    frame = _frame_from_payload(read_first_record(path))
    return {
        "context_name": str(frame.context.name),
        "map_features": int(len(frame.map_features)),
        "record_bytes": int(frame.ByteSize()),
    }


def list_remote_segments(
    host: str,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    splits: Iterable[str] = ("training", "validation"),
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for split in splits:
        if split not in {"training", "validation"}:
            raise ValueError(f"unsupported split {split!r}")
        remote_dir = f"{remote_root.rstrip('/')}/{split}"
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "find",
            remote_dir,
            "-maxdepth",
            "1",
            "-type",
            "f",
            "-name",
            "segment-*.tfrecord",
            "-print",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        paths = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"remote listing for {split} contains duplicate paths")
        out[split] = paths
    return out


def _cache_path(cache_root: Path, split: str, remote_path: str) -> Path:
    return cache_root / split / f"{Path(remote_path).name}.head"


def _fetch_one(
    *,
    host: str,
    split: str,
    remote_path: str,
    cache_root: Path,
) -> dict[str, Any]:
    cache_path = _cache_path(cache_root, split, remote_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    reused = False
    if cache_path.is_file():
        try:
            if cache_path.stat().st_size != HEAD_BYTES:
                raise TruncatedRecordError(
                    f"{cache_path}: reusable cache must be exactly {HEAD_BYTES} bytes"
                )
            inspection = inspect_cache(cache_path)
            reused = True
        except Exception:
            inspection = None
    else:
        inspection = None

    if inspection is None:
        tmp = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp, "wb") as handle:
                subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        host,
                        "dd",
                        f"if={remote_path}",
                        "bs=1M",
                        "count=12",
                        "status=none",
                    ],
                    check=True,
                    stdout=handle,
                )
                handle.flush()
                os.fsync(handle.fileno())
            if tmp.stat().st_size != HEAD_BYTES:
                raise TruncatedRecordError(
                    f"{tmp}: ssh+dd returned {tmp.stat().st_size} bytes, expected {HEAD_BYTES}"
                )
            inspection = inspect_cache(tmp)
            tmp.replace(cache_path)
        finally:
            if tmp.exists():
                tmp.unlink()

    expected_context = Path(remote_path).name
    if expected_context.startswith("segment-"):
        expected_context = expected_context[len("segment-") :]
    if expected_context.endswith(".tfrecord"):
        expected_context = expected_context[: -len(".tfrecord")]
    camera_label_suffix = "_with_camera_labels"
    if expected_context.endswith(camera_label_suffix):
        expected_context = expected_context[: -len(camera_label_suffix)]
    if inspection["context_name"] != expected_context:
        raise ValueError(
            f"{remote_path}: frame context {inspection['context_name']!r} "
            f"!= filename segment {expected_context!r}"
        )
    return {
        "split": split,
        "segment": inspection["context_name"],
        "remote_path": remote_path,
        "cache_path": str(cache_path),
        "cache_bytes": int(cache_path.stat().st_size),
        "record_bytes": inspection["record_bytes"],
        "map_features": inspection["map_features"],
        "reused": reused,
    }


def fetch_all(
    *,
    host: str,
    remote_root: str,
    cache_root: str | Path,
    splits: tuple[str, ...] = ("training", "validation"),
    workers: int = 8,
    expected_total: int | None = 1008,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    listing = list_remote_segments(host, remote_root, splits)
    jobs = [
        (split, remote_path)
        for split in splits
        for remote_path in listing[split]
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_jobs = {
            executor.submit(
                _fetch_one,
                host=host,
                split=split,
                remote_path=remote_path,
                cache_root=cache_root,
            ): (split, remote_path)
            for split, remote_path in jobs
        }
        for future in as_completed(future_jobs):
            split, remote_path = future_jobs[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {"split": split, "remote_path": remote_path, "error": repr(exc)}
                )

    results.sort(key=lambda item: (item["split"], item["segment"]))
    empty = [item for item in results if item["map_features"] == 0]
    summary = {
        "host": host,
        "remote_root": remote_root,
        "cache_root": str(cache_root),
        "head_bytes": HEAD_BYTES,
        "expected_total": expected_total,
        "discovered": {split: len(listing[split]) for split in splits},
        "discovered_total": len(jobs),
        "expected_count_matches": expected_total is None or len(jobs) == int(expected_total),
        "completed": len(results),
        "downloaded": sum(not item["reused"] for item in results),
        "reused": sum(item["reused"] for item in results),
        "empty_map_features_count": len(empty),
        "empty_map_features_segments": [item["segment"] for item in empty],
        "failures": failures,
        "segments": results,
    }
    return summary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="13")
    parser.add_argument("--remote_root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--splits", nargs="+", choices=("training", "validation"), default=["training", "validation"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected_total", type=int, default=1008)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = args.summary or Path(args.cache_root) / "fetch_summary.json"
    summary = fetch_all(
        host=args.host,
        remote_root=args.remote_root,
        cache_root=args.cache_root,
        splits=tuple(args.splits),
        workers=args.workers,
        expected_total=args.expected_total,
    )
    _write_json_atomic(summary_path, summary)
    print(json.dumps({key: summary[key] for key in (
        "expected_total", "discovered", "discovered_total", "expected_count_matches",
        "completed", "downloaded", "reused", "empty_map_features_count", "failures",
    )}, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
