"""Backfill Waymo camera GT into existing FlowDGGT cache files.

This is a lightweight repair for caches that already contain all expensive
FlowDGGT tensors but are missing ``object_meta.camera_to_world_corrected`` and
``object_meta.intrinsics``.  The values are rebuilt from the Waymo edit
annotation JSON plus per-cache ``meta.frame_indices_scene`` and ``meta.cam_ids``.
``meta.raw_image_size_hw`` is recovered from the annotation and written back
when it is missing, for both regular and SQLite chunk caches.
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.flow_cache_io import (
    is_chunked_flow_cache,
    is_gzip_file,
    is_zstd_file,
    load_flow_cache,
    save_flow_cache,
    _get_info,
    _put_info,
    _put_chunk,
    _require_zstd_module,
)


DEFAULT_CACHE_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b/training"
DEFAULT_PROCESSED_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt"
WAYMO_OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def _torch_from_bytes(data: bytes) -> Any:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def _get_chunk(conn: sqlite3.Connection, dctx: Any, key: str) -> Any:
    row = conn.execute("SELECT payload FROM chunks WHERE key=?", (str(key),)).fetchone()
    if row is None:
        raise KeyError(f"Missing chunk {key!r}")
    return _torch_from_bytes(dctx.decompress(bytes(row[0])))


def _has_chunk(conn: sqlite3.Connection, key: str) -> bool:
    return conn.execute("SELECT 1 FROM chunks WHERE key=? LIMIT 1", (str(key),)).fetchone() is not None


def _as_long_list(value: Any, name: str) -> list[int]:
    if not torch.is_tensor(value):
        raise TypeError(f"meta[{name!r}] must be a tensor, got {type(value).__name__}")
    return [int(v) for v in value.detach().cpu().view(-1).tolist()]


def _raw_hw_for_views(meta: dict[str, Any], cam_ids: list[int], annotation: dict[str, Any]) -> list[tuple[int, int]]:
    original = annotation.get("original_image_size") or {}
    out = []
    for cam_id in cam_ids:
        value = original.get(str(cam_id))
        if value is None:
            raise KeyError(f"Missing original_image_size for camera {cam_id}")
        if len(value) != 2:
            raise ValueError(f"Invalid original_image_size for camera {cam_id}: {value!r}")
        out.append((int(value[0]), int(value[1])))
    return out


def _build_intrinsic_matrix(normalized_intrinsics: Any, image_hw: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_hw
    fx_n, fy_n, cx_n, cy_n = [float(v) for v in normalized_intrinsics]
    return np.array(
        [
            [fx_n * image_w, 0.0, cx_n * image_w],
            [0.0, fy_n * image_h, cy_n * image_h],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _annotation_path(processed_root: Path, split: str, scene_name: str) -> Path:
    root = processed_root / "waymo_edit_cache" / "annotations" / split
    candidates = [root / f"{scene_name}.json"]
    if not scene_name.startswith("segment-"):
        candidates.append(root / f"segment-{scene_name}_with_camera_labels.json")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Annotation JSON not found. Tried: {[str(p) for p in candidates]}")


def build_camera_gt_from_meta(meta: dict[str, Any], processed_root: Path, split: str) -> dict[str, torch.Tensor]:
    scene_name = str(meta.get("scene_name", ""))
    if not scene_name:
        raise ValueError("cache meta is missing scene_name")
    with _annotation_path(processed_root, split, scene_name).open("r") as f:
        annotation = json.load(f)

    frame_indices = _as_long_list(meta.get("frame_indices_scene"), "frame_indices_scene")
    cam_ids = _as_long_list(meta.get("cam_ids"), "cam_ids")
    raw_hw_by_view = _raw_hw_for_views(meta, cam_ids, annotation)
    meta["raw_image_size_hw"] = torch.tensor(raw_hw_by_view, dtype=torch.long)

    ego_pose_all = np.asarray(annotation["ego_pose"], dtype=np.float32)
    camera_to_ego_by_cam = annotation["camera_to_ego"]
    norm_intr_by_cam = annotation["normalized_intrinsics"]

    c2w_by_view = []
    intrinsics = []
    for view_idx, cam_id in enumerate(cam_ids):
        cam_to_ego = np.asarray(camera_to_ego_by_cam[str(cam_id)], dtype=np.float32)
        corrected = []
        for frame_idx in frame_indices:
            if frame_idx < 0 or frame_idx >= int(ego_pose_all.shape[0]):
                raise IndexError(
                    f"frame index {frame_idx} out of range for scene {scene_name} "
                    f"with {ego_pose_all.shape[0]} poses"
                )
            corrected.append(ego_pose_all[frame_idx] @ cam_to_ego @ WAYMO_OPENCV2DATASET)
        c2w_by_view.append(np.stack(corrected, axis=0).astype(np.float32))
        intrinsics.append(_build_intrinsic_matrix(norm_intr_by_cam[str(cam_id)], raw_hw_by_view[view_idx]))

    camera_to_world = np.stack(c2w_by_view, axis=1).astype(np.float32)
    intrinsics_np = np.stack(intrinsics, axis=0).astype(np.float32)
    return {
        "camera_to_world_corrected": torch.tensor(camera_to_world.tolist(), dtype=torch.float32),
        "intrinsics": torch.tensor(intrinsics_np.tolist(), dtype=torch.float32),
    }


def _compare_existing(obj: dict[str, Any], expected: dict[str, torch.Tensor]) -> tuple[bool, str]:
    for key, exp in expected.items():
        cur = obj.get(key)
        if not torch.is_tensor(cur):
            return False, f"{key} is missing or not a tensor"
        if tuple(cur.shape) != tuple(exp.shape):
            return False, f"{key} shape mismatch: {tuple(cur.shape)} != {tuple(exp.shape)}"
        if not torch.allclose(cur.float(), exp.float(), atol=1e-4, rtol=1e-4):
            max_err = float((cur.float() - exp.float()).abs().max().item())
            return False, f"{key} value mismatch: max_abs_err={max_err:.6g}"
    return True, "ok"


def _compression_for_existing(path: Path) -> str:
    if is_gzip_file(path):
        return "gzip"
    if is_zstd_file(path):
        return "zstd"
    return "none"


def _raw_size_matches(value: Any, expected: torch.Tensor) -> bool:
    if not torch.is_tensor(value):
        return False
    current = value.detach().cpu().to(torch.long)
    if current.ndim == 1 and current.numel() == 2 and expected.ndim == 2:
        current = current.view(1, 2).expand_as(expected)
    return tuple(current.shape) == tuple(expected.shape) and torch.equal(current, expected)


def process_chunked(path: Path, processed_root: Path, split: str, args) -> dict[str, Any]:
    zstd = _require_zstd_module()
    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=max(1, min(19, int(args.zstd_level))))
    conn = sqlite3.connect(str(path))
    try:
        if bool(args.unsafe_no_journal):
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
        summary = _get_info(conn, "summary")
        meta = _get_chunk(conn, dctx, "global/meta")
        obj = _get_chunk(conn, dctx, "global/object_meta") if _has_chunk(conn, "global/object_meta") else {}
        raw_size_before = meta.get("raw_image_size_hw")
        expected = build_camera_gt_from_meta(meta, processed_root, split)
        raw_size_ok = _raw_size_matches(raw_size_before, meta["raw_image_size_hw"])
        has_existing = all(torch.is_tensor(obj.get(k)) for k in expected)
        if has_existing:
            ok, reason = _compare_existing(obj, expected)
            if not ok and not bool(args.force):
                return {"status": "mismatch", "reason": reason}
            if ok and raw_size_ok and not bool(args.force):
                return {"status": "present", "reason": reason}
            if ok and not raw_size_ok and not bool(args.force) and raw_size_before is not None:
                return {"status": "mismatch", "reason": "raw_image_size_hw differs from annotation"}

        if bool(args.verify_only):
            return {"status": "missing" if not has_existing else "mismatch", "reason": "verify_only"}
        if bool(args.dry_run):
            return {"status": "would_write", "reason": "dry_run"}

        obj.update({k: v.cpu().contiguous() for k, v in expected.items()})
        _put_chunk(conn, cctx, "global/meta", meta)
        _put_chunk(conn, cctx, "global/object_meta", obj)
        summary["has_camera_gt"] = True
        _put_info(conn, "summary", summary)
        row = conn.execute("SELECT COUNT(*), SUM(zbytes), SUM(raw_bytes) FROM chunks").fetchone()
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

        obj_after = _get_chunk(conn, dctx, "global/object_meta")
        ok, reason = _compare_existing(obj_after, expected)
        if not ok:
            raise RuntimeError(f"write verification failed: {reason}")
        return {"status": "written", "reason": "ok"}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def process_regular(path: Path, processed_root: Path, split: str, args) -> dict[str, Any]:
    payload = load_flow_cache(path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    obj = payload.setdefault("object_meta", {})
    raw_size_before = meta.get("raw_image_size_hw")
    expected = build_camera_gt_from_meta(meta, processed_root, split)
    raw_size_ok = _raw_size_matches(raw_size_before, meta["raw_image_size_hw"])
    has_existing = all(torch.is_tensor(obj.get(k)) for k in expected)
    if has_existing:
        ok, reason = _compare_existing(obj, expected)
        if not ok and not bool(args.force):
            return {"status": "mismatch", "reason": reason}
        if ok and raw_size_ok and not bool(args.force):
            return {"status": "present", "reason": reason}
        if ok and not raw_size_ok and not bool(args.force) and raw_size_before is not None:
            return {"status": "mismatch", "reason": "raw_image_size_hw differs from annotation"}
    if bool(args.verify_only):
        return {"status": "missing" if not has_existing else "mismatch", "reason": "verify_only"}
    if bool(args.dry_run):
        return {"status": "would_write", "reason": "dry_run"}

    obj.update({k: v.cpu().contiguous() for k, v in expected.items()})
    save_flow_cache(
        payload,
        path,
        compression=_compression_for_existing(path),
        gzip_level=int(args.gzip_level),
        zstd_level=int(args.zstd_level),
    )
    reloaded = load_flow_cache(path, map_location="cpu", weights_only=False)
    ok, reason = _compare_existing(reloaded.get("object_meta") or {}, expected)
    if not ok:
        raise RuntimeError(f"write verification failed: {reason}")
    return {"status": "written", "reason": "ok"}


def iter_cache_files(cache_root: Path, pattern: str) -> list[Path]:
    if cache_root.is_file():
        return [cache_root]
    return sorted(p for p in cache_root.rglob(pattern) if p.is_file())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--split", default="training")
    p.add_argument("--pattern", default="*.pt")
    p.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verify_only", action="store_true")
    p.add_argument("--force", action="store_true", help="Rewrite even when fields already exist.")
    p.add_argument("--gzip_level", type=int, default=1)
    p.add_argument("--zstd_level", type=int, default=1)
    p.add_argument("--unsafe_no_journal", action="store_true", default=True)
    p.add_argument("--progress_every", type=int, default=100)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cache_root = Path(args.cache_root)
    processed_root = Path(args.processed_root)
    files = iter_cache_files(cache_root, str(args.pattern))
    if int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"No cache files matched {cache_root} / {args.pattern}")

    counts: dict[str, int] = {}
    errors: list[tuple[str, str]] = []
    for i, path in enumerate(files, 1):
        try:
            if is_chunked_flow_cache(path):
                result = process_chunked(path, processed_root, str(args.split), args)
            else:
                result = process_regular(path, processed_root, str(args.split), args)
            status = str(result["status"])
            counts[status] = counts.get(status, 0) + 1
            if status not in ("present", "written"):
                print(f"[{status}] {path}: {result.get('reason', '')}", flush=True)
        except Exception as exc:
            counts["error"] = counts.get("error", 0) + 1
            errors.append((str(path), f"{type(exc).__name__}: {exc}"))
            print(f"[error] {path}: {type(exc).__name__}: {exc}", flush=True)
        if i % max(1, int(args.progress_every)) == 0:
            print(f"[progress] {i}/{len(files)} {counts}", flush=True)

    print(f"[done] processed={len(files)} counts={counts}", flush=True)
    if errors:
        print("[errors:first10]", flush=True)
        for path, message in errors[:10]:
            print(f"  {path}: {message}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
