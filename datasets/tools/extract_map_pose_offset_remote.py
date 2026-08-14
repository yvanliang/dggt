#!/usr/bin/env python
"""Extract per-frame Waymo ``Frame.map_pose_offset`` without protobuf.

The same file is sent over stdin to ``ssh 13 python3 - --remote-worker`` and is
therefore intentionally compatible with Python 3.5.  The fast path reads the
TFRecord header and the last 64 payload bytes only.  If field 11 cannot be
decoded there, the worker falls back to a complete wire-format scan of that
frame and emits an explicit warning.
"""

from __future__ import print_function

import argparse
import glob
import json
import math
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


TFRECORD_HEADER_BYTES = 12
TFRECORD_FOOTER_BYTES = 4
TAIL_BYTES = 64
MAP_POSE_OFFSET_FIELD = 11
DEFAULT_REMOTE_ROOT = "/data/liangyiyuan/waymo"
DEFAULT_CACHE_ROOT = "/data/disk2/lyy_dataset/waymo_tfrecord_frame0_cache/map_pose_offset"
T14_WINDOW_FRAMES = 10
T14_WINDOW_XY_LIMIT_METRES = 0.034
T14_WINDOW_Z_LIMIT_METRES = 0.089


class WireDecodeError(ValueError):
    pass


def _read_exact(handle, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = handle.read(count - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _decode_varint(data, position, limit=None):
    if limit is None:
        limit = len(data)
    value = 0
    shift = 0
    while position < limit and shift < 70:
        byte = data[position]
        if not isinstance(byte, int):  # Python 2 defensive; harmless on 3.x.
            byte = ord(byte)
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7
    raise WireDecodeError("unterminated protobuf varint")


def _skip_wire_value(data, position, wire_type, limit):
    if wire_type == 0:
        _, position = _decode_varint(data, position, limit)
        return position
    if wire_type == 1:
        position += 8
    elif wire_type == 2:
        length, position = _decode_varint(data, position, limit)
        position += length
    elif wire_type == 5:
        position += 4
    else:
        raise WireDecodeError("unsupported protobuf wire type %d" % wire_type)
    if position > limit:
        raise WireDecodeError("protobuf field extends beyond message")
    return position


def _parse_point_message(data):
    """Parse Waymo's Vector3d/MapPoseOffset message into three doubles."""
    values = [0.0, 0.0, 0.0]
    position = 0
    limit = len(data)
    while position < limit:
        tag, position = _decode_varint(data, position, limit)
        field_number = tag >> 3
        wire_type = tag & 7
        if field_number in (1, 2, 3):
            if wire_type != 1 or position + 8 > limit:
                raise WireDecodeError("map_pose_offset coordinate is not fixed64")
            values[field_number - 1] = struct.unpack("<d", data[position : position + 8])[0]
            position += 8
        else:
            position = _skip_wire_value(data, position, wire_type, limit)
    if not all(math.isfinite(value) for value in values):
        raise WireDecodeError("map_pose_offset contains a non-finite value")
    return values


def parse_map_pose_offset_tail(tail):
    """Decode field 11 when it is the final field in a record payload."""
    candidates = []
    for start in range(len(tail)):
        try:
            tag, position = _decode_varint(tail, start)
            if tag >> 3 != MAP_POSE_OFFSET_FIELD or tag & 7 != 2:
                continue
            length, position = _decode_varint(tail, position)
            end = position + length
            if end != len(tail):
                continue
            candidates.append(_parse_point_message(tail[position:end]))
        except (WireDecodeError, IndexError, struct.error):
            continue
    if not candidates:
        raise WireDecodeError("Frame field 11 was not found at the record tail")
    # A byte pattern inside coordinate payload could resemble tag 11.  The
    # actual outer field is the earliest valid candidate ending at EOF.
    return candidates[0]


def parse_map_pose_offset_full(payload):
    """Fallback: scan all top-level Frame fields using protobuf wire rules."""
    position = 0
    limit = len(payload)
    found = None
    while position < limit:
        tag, position = _decode_varint(payload, position, limit)
        field_number = tag >> 3
        wire_type = tag & 7
        if field_number == MAP_POSE_OFFSET_FIELD:
            if wire_type != 2:
                raise WireDecodeError("Frame field 11 is not length-delimited")
            length, position = _decode_varint(payload, position, limit)
            end = position + length
            if end > limit:
                raise WireDecodeError("Frame field 11 extends beyond the record")
            found = _parse_point_message(payload[position:end])
            position = end
        else:
            position = _skip_wire_value(payload, position, wire_type, limit)
    if found is None:
        raise WireDecodeError("Frame payload has no field 11")
    return found


def scan_tfrecord_offsets(path, tail_bytes=TAIL_BYTES, warn_stream=None):
    """Return ``(offsets[S,3], fallback_frame_indices)`` for one segment."""
    if warn_stream is None:
        warn_stream = sys.stderr
    offsets = []
    fallbacks = []
    with open(path, "rb") as handle:
        frame_index = 0
        while True:
            header = _read_exact(handle, TFRECORD_HEADER_BYTES)
            if header is None:
                break
            record_length = struct.unpack("<Q", header[:8])[0]
            record_start = handle.tell()
            tail_length = min(int(tail_bytes), int(record_length))
            handle.seek(record_start + record_length - tail_length)
            tail = _read_exact(handle, tail_length)
            if tail is None:
                raise EOFError("%s frame %d has a truncated payload" % (path, frame_index))
            try:
                offset = parse_map_pose_offset_tail(tail)
            except WireDecodeError as tail_error:
                fallbacks.append(frame_index)
                print(
                    "WARNING: %s frame %d tail decode failed (%s); scanning full frame"
                    % (path, frame_index, tail_error),
                    file=warn_stream,
                )
                handle.seek(record_start)
                payload = _read_exact(handle, record_length)
                if payload is None:
                    raise EOFError("%s frame %d has a truncated payload" % (path, frame_index))
                offset = parse_map_pose_offset_full(payload)
            offsets.append(offset)
            handle.seek(record_start + record_length + TFRECORD_FOOTER_BYTES)
            frame_index += 1
    if not offsets:
        raise ValueError("%s contains no complete TFRecord records" % path)
    return offsets, fallbacks


def assert_t14_regression(processed_poses, original_poses, offsets):
    """Assert the already-measured T14 semantic contract without re-probing."""
    if processed_poses != original_poses:
        raise AssertionError("processed ego_pose differs from original Frame.pose")
    if len(offsets) < T14_WINDOW_FRAMES:
        raise AssertionError("T14 regression needs at least %d offsets" % T14_WINDOW_FRAMES)
    worst_xy = 0.0
    worst_z = 0.0
    for start in range(len(offsets) - T14_WINDOW_FRAMES + 1):
        window = offsets[start : start + T14_WINDOW_FRAMES]
        dx = max(row[0] for row in window) - min(row[0] for row in window)
        dy = max(row[1] for row in window) - min(row[1] for row in window)
        dz = max(row[2] for row in window) - min(row[2] for row in window)
        worst_xy = max(worst_xy, math.hypot(dx, dy))
        worst_z = max(worst_z, dz)
    if worst_xy > T14_WINDOW_XY_LIMIT_METRES + 1.0e-12:
        raise AssertionError("10-frame offset xy spread exceeds 34 mm")
    if worst_z > T14_WINDOW_Z_LIMIT_METRES + 1.0e-12:
        raise AssertionError("10-frame offset z spread exceeds 89 mm")
    return {"worst_xy_metres": worst_xy, "worst_z_metres": worst_z}


def _normalize_segment(path):
    name = os.path.basename(path)
    if name.endswith(".tfrecord"):
        name = name[: -len(".tfrecord")]
    if name.startswith("segment-"):
        name = name[len("segment-") :]
    suffix = "_with_camera_labels"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def _remote_scan_one(job):
    split, path = job
    try:
        offsets, fallbacks = scan_tfrecord_offsets(path)
        return {
            "split": split,
            "segment": _normalize_segment(path),
            "remote_path": path,
            "offsets": offsets,
            "fallback_frames": fallbacks,
        }
    except Exception as exc:
        return {
            "split": split,
            "segment": _normalize_segment(path),
            "remote_path": path,
            "error": repr(exc),
        }


def _run_remote_worker(args):
    jobs = []
    if args.paths:
        for path in args.paths:
            split = "validation" if "/validation/" in path else "training"
            jobs.append((split, path))
    else:
        for split in args.splits:
            pattern = os.path.join(args.remote_root, split, "segment-*.tfrecord")
            jobs.extend((split, path) for path in sorted(glob.glob(pattern)))
    pool = ThreadPoolExecutor(max_workers=max(1, int(args.workers)))
    try:
        for result in pool.map(_remote_scan_one, jobs):
            print(json.dumps(result, separators=(",", ":"), sort_keys=True))
            sys.stdout.flush()
    finally:
        pool.shutdown(wait=True)


def _write_json_atomic(path, payload):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.rename(tmp, path)


def _run_local(args):
    with open(os.path.abspath(__file__), "rb") as handle:
        source = handle.read()
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        args.host,
        "python3",
        "-",
        "--remote-worker",
        "--remote-root",
        args.remote_root,
        "--workers",
        str(args.workers),
        "--splits",
    ] + list(args.splits)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(source)
    stderr_text = stderr.decode("utf-8", "replace")
    results = []
    parse_errors = []
    for line_number, raw_line in enumerate(stdout.decode("utf-8", "replace").splitlines(), 1):
        try:
            results.append(json.loads(raw_line))
        except ValueError as exc:
            parse_errors.append({"line": line_number, "error": repr(exc), "text": raw_line})
    failures = [result for result in results if "error" in result]
    if process.returncode != 0 and not failures:
        failures.append({"error": "remote worker exited %d" % process.returncode})

    completed = []
    for result in results:
        if "error" in result:
            continue
        offsets = result.get("offsets")
        if (
            not isinstance(offsets, list)
            or not offsets
            or any(not isinstance(row, list) or len(row) != 3 for row in offsets)
            or any(not math.isfinite(float(value)) for row in offsets for value in row)
        ):
            failures.append(
                {"split": result.get("split"), "segment": result.get("segment"), "error": "invalid [S,3] offsets"}
            )
            continue
        output_path = os.path.join(
            args.cache_root,
            result["split"],
            result["segment"] + ".json",
        )
        _write_json_atomic(output_path, result)
        completed.append(result)

    discovered = {}
    for split in args.splits:
        discovered[split] = sum(1 for result in results if result.get("split") == split)
    summary = {
        "host": args.host,
        "remote_root": args.remote_root,
        "cache_root": args.cache_root,
        "expected_total": args.expected_total,
        "discovered": discovered,
        "discovered_total": len(results),
        "expected_count_matches": len(results) == int(args.expected_total),
        "completed": len(completed),
        "fallback_segment_count": sum(bool(result.get("fallback_frames")) for result in completed),
        "fallback_frame_count": sum(len(result.get("fallback_frames", [])) for result in completed),
        "failures": failures,
        "stdout_parse_errors": parse_errors,
        "remote_stderr": stderr_text,
    }
    _write_json_atomic(os.path.join(args.cache_root, "extract_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures or parse_errors or process.returncode != 0:
        return 1
    return 0


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="13")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--splits", nargs="+", choices=("training", "validation"), default=["training", "validation"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-total", type=int, default=1008)
    parser.add_argument("--paths", nargs="*", default=[])
    parser.add_argument("--remote-worker", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.remote_worker:
        _run_remote_worker(args)
        return 0
    return _run_local(args)


if __name__ == "__main__":
    sys.exit(main())
