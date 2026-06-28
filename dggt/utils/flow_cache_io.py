"""I/O helpers for FlowDGGT offline cache payloads.

The legacy cache schema is a nested PyTorch payload optionally wrapped in gzip
or zstd.  The current training cache keeps the same logical schema/version and
``.pt`` filenames, but stores the payload in a SQLite container with
independently zstd-compressed chunks.  This lets training read only the sampled
frame window instead of decompressing a whole 29-frame clip.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Any

import torch


GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
SQLITE_MAGIC = b"SQLite format 3\x00"
CHUNKED_FLOW_CACHE_FORMAT = "flow_cache_chunked_zstd_sqlite"
CHUNKED_FLOW_CACHE_FORMAT_VERSION = 1
CURRENT_FLOW_CACHE_SCHEMA_VERSION = 8


def is_current_flow_cache_summary(summary: dict[str, Any]) -> bool:
    return (
        str(summary.get("format", "")) == CHUNKED_FLOW_CACHE_FORMAT
        and int(summary.get("format_version", 0)) == CHUNKED_FLOW_CACHE_FORMAT_VERSION
        and int(summary.get("schema_version", 0)) == CURRENT_FLOW_CACHE_SCHEMA_VERSION
    )


def is_gzip_file(path: str | os.PathLike[str]) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def is_zstd_file(path: str | os.PathLike[str]) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == ZSTD_MAGIC


def is_chunked_flow_cache(path: str | os.PathLike[str]) -> bool:
    with open(path, "rb") as f:
        return f.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC


def _require_zstd_module():
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "chunked FlowDGGT cache requires the Python package `zstandard`. "
            "Install it in the dggt environment before generating or reading "
            "chunked cache files."
        ) from exc
    return zstd


def _find_zstd_binary() -> str:
    for candidate in (
        os.environ.get("ZSTD_BIN"),
        shutil.which("zstd"),
        "/home/dancer/anaconda3/bin/zstd",
        "/usr/bin/zstd",
    ):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError(
        "zstd compression requested but no zstd binary was found. "
        "Install zstd or set ZSTD_BIN=/path/to/zstd."
    )


def load_flow_cache(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
    mmap: bool | None = None,
) -> dict[str, Any]:
    """Load a gzip, zstd, or plain torch cache payload."""
    if is_chunked_flow_cache(path):
        return load_chunked_flow_cache(path, map_location=map_location, weights_only=weights_only)
    if is_gzip_file(path):
        with gzip.open(path, "rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=weights_only)
    if is_zstd_file(path):
        data = subprocess.check_output([_find_zstd_binary(), "-q", "-dc", str(path)])
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=weights_only)
    return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=mmap)


def save_flow_cache(
    payload: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    compression: str = "gzip",
    gzip_level: int = 1,
    zstd_level: int | None = None,
) -> None:
    """Save a FlowDGGT cache payload.

    `compression="gzip"` / `"zstd"` keeps the existing `.pt` payload layout
    while reducing the uncompressed tensor storages produced by `torch.save`.
    """
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    compression = str(compression).lower()
    try:
        if compression in ("none", "off", "false", "0"):
            torch.save(payload, tmp_path)
        elif compression == "gzip":
            level = max(0, min(9, int(gzip_level)))
            with gzip.open(tmp_path, "wb", compresslevel=level) as f:
                torch.save(payload, f)
        elif compression in ("zstd", "zst"):
            level = int(gzip_level if zstd_level is None else zstd_level)
            level = max(1, min(19, level))
            proc = subprocess.Popen(
                [_find_zstd_binary(), "-q", f"-{level}", "-T0", "-f", "-o", str(tmp_path), "-"],
                stdin=subprocess.PIPE,
            )
            assert proc.stdin is not None
            try:
                torch.save(payload, proc.stdin)
                proc.stdin.close()
                rc = proc.wait()
            except Exception:
                proc.kill()
                proc.wait()
                raise
            if rc != 0:
                raise RuntimeError(f"zstd failed while writing {tmp_path} with exit code {rc}")
        else:
            raise ValueError(f"Unsupported flow cache compression: {compression}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def _torch_to_bytes(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(_compact_for_save(obj), buffer)
    return buffer.getvalue()


def _torch_from_bytes(data: bytes, *, map_location: str | torch.device = "cpu", weights_only: bool = False) -> Any:
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=weights_only)


def _compact_for_save(obj: Any) -> Any:
    """Clone tensor views so torch.save writes only the selected chunk storage."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _compact_for_save(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_compact_for_save(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_compact_for_save(v) for v in obj)
    return obj


def _json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _open_chunked_ro(path: str | os.PathLike[str]) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)


def _put_info(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO info(key, value) VALUES (?, ?)",
        (str(key), json.dumps(value, default=_json_default, ensure_ascii=False)),
    )


def _get_info(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute("SELECT value FROM info WHERE key=?", (str(key),)).fetchone()
    if row is None:
        raise KeyError(f"Missing chunked FlowDGGT cache info key: {key}")
    return json.loads(row[0])


def _put_chunk(conn: sqlite3.Connection, cctx: Any, key: str, obj: Any) -> tuple[int, int]:
    raw = _torch_to_bytes(obj)
    blob = cctx.compress(raw)
    conn.execute(
        "INSERT OR REPLACE INTO chunks(key, zbytes, raw_bytes, payload) VALUES (?, ?, ?, ?)",
        (str(key), int(len(blob)), int(len(raw)), sqlite3.Binary(blob)),
    )
    return int(len(blob)), int(len(raw))


def _get_chunk_blob(conn: sqlite3.Connection, key: str) -> bytes:
    row = conn.execute("SELECT payload FROM chunks WHERE key=?", (str(key),)).fetchone()
    if row is None:
        raise KeyError(f"Missing chunked FlowDGGT cache chunk: {key}")
    return bytes(row[0])


def _get_chunk(conn: sqlite3.Connection, dctx: Any, key: str) -> Any:
    return _torch_from_bytes(dctx.decompress(_get_chunk_blob(conn, key)))


def _tensor_frame(value: Any, frame: int) -> Any:
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor or None for frame chunk, got {type(value).__name__}")
    return value[int(frame)]


def _slice_first_dim(value: Any, indices: torch.Tensor) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.index_select(0, indices)
    return value


def _slice_object_meta_for_frames(obj: dict[str, Any], subset: torch.Tensor) -> dict[str, Any]:
    """Slice object metadata to the requested original frames.

    The dataset reader will apply a local ``0..S-1`` subset afterwards, so every
    frame-dependent tensor/list here is rewritten to length ``S``.
    """
    out: dict[str, Any] = {}
    frame_dim1 = {
        "object_speed_mps_selected",
        "object_is_moving_frame_selected",
        "object_track_valid_mask_selected",
        "object_asset_image_valid_mask_selected",
        "object_bbox_present_mask_selected",
        "object_bbox_editable_mask_selected",
        "object_bbox_model_selected",
        "object_front_bbox_present_mask_selected",
        "object_front_bbox_editable_mask_selected",
        "object_front_bbox_model_selected",
        "object_obj_to_world_selected",
        "object_box_size_selected",
        "object_box_corners_world_selected",
    }
    subset_list = [int(v) for v in subset.tolist()]
    for key, value in obj.items():
        if key in frame_dim1 and torch.is_tensor(value) and value.dim() >= 2:
            out[key] = value.index_select(1, subset)
        elif key == "camera_to_world_corrected" and torch.is_tensor(value) and value.dim() >= 1:
            out[key] = value.index_select(0, subset)
        elif key == "object_asset_image_paths_selected" and isinstance(value, list):
            out[key] = [[paths[n] for n in subset_list] for paths in value]
        elif key == "protected_object_boxes_by_frame" and isinstance(value, list):
            out[key] = [value[n] for n in subset_list]
        else:
            out[key] = value
    return out


def _slice_meta_for_frames(meta: dict[str, Any], subset: torch.Tensor) -> dict[str, Any]:
    out = dict(meta)
    for key in ("frame_indices_scene", "timestamps"):
        value = out.get(key)
        if torch.is_tensor(value) and value.dim() >= 1:
            out[key] = value.index_select(0, subset)
    out["num_frames"] = int(subset.numel())
    return out


def _stack_optional(values: list[Any]) -> Any:
    if not values or values[0] is None:
        return None
    return torch.stack(values, dim=0).contiguous()


def _has_chunk(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute("SELECT 1 FROM chunks WHERE key=? LIMIT 1", (str(key),)).fetchone()
    return row is not None


def _pack_bool_tensor(mask: torch.Tensor) -> dict[str, Any]:
    flat = mask.to(torch.uint8).flatten().cpu()
    n = int(flat.numel())
    pad = (-n) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros((pad,))], dim=0)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    packed = (flat.view(-1, 8) * weights).sum(dim=1).to(torch.uint8)
    return {"packed": packed, "numel": n}


def _unpack_bool_tensor(obj: dict[str, Any]) -> torch.Tensor:
    packed = obj["packed"].to(torch.uint8).flatten()
    n = int(obj["numel"])
    bits = ((packed[:, None] >> torch.arange(8, dtype=torch.uint8)) & 1).to(torch.bool)
    return bits.flatten()[:n].contiguous()


def _frame_gauss_offsets_from_payload(payload: dict[str, Any]) -> torch.Tensor:
    depth = payload["pass1"]["depth"].float()
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    sky_mask = payload["raw"]["sky_mask"].float()
    sky_mask_hw = sky_mask.permute(0, 2, 3, 1)
    non_sky = (sky_mask_hw < 0.5).any(dim=-1)
    valid = non_sky & (depth > 1e-4)
    counts = valid.reshape(valid.shape[0], -1).sum(dim=1).to(torch.long)
    offsets = torch.zeros((int(counts.numel()) + 1,), dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def _semantic_vehicle_from_logits(value: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if value is None:
        return None, None
    if not torch.is_tensor(value):
        raise TypeError(f"Expected semantic_logits tensor or None, got {type(value).__name__}")
    logits = value.detach().cpu().float()
    if logits.shape[-1] <= 4:
        shape = tuple(int(v) for v in logits.shape[:-1])
        return torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.bool)
    probs = torch.softmax(logits, dim=-1)
    return probs[..., 4].contiguous(), (probs.argmax(dim=-1) == 4).contiguous()


def save_flow_cache_chunked(
    payload: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    zstd_level: int = 1,
) -> None:
    """Save the current FlowDGGT cache as a chunked zstd SQLite container.

    The logical payload keeps its existing ``schema_version`` and top-level
    structure.  Large per-frame tensors are split into independently compressed
    chunks so SceneFlow and tokenizer Stage-B can read only their sampled
    window.
    """
    zstd = _require_zstd_module()
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    cctx = zstd.ZstdCompressor(level=max(1, min(19, int(zstd_level))))
    total_zbytes = 0
    total_raw_bytes = 0
    chunk_count = 0
    payload = _compact_for_save(payload)
    meta = payload["meta"]
    raw = payload["raw"]
    pass1 = payload["pass1"]
    pass2 = payload["pass2_splatted_tok_low"]
    mode_kind = str(payload["mode_kind"])
    num_frames = int(meta["num_frames"])

    conn = sqlite3.connect(str(tmp_path))
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE TABLE info(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE chunks("
            "key TEXT PRIMARY KEY, "
            "zbytes INTEGER NOT NULL, "
            "raw_bytes INTEGER NOT NULL, "
            "payload BLOB NOT NULL)"
        )

        asset = payload.get("asset_pass") or {}
        asset_keys = sorted(int(k) for k in asset.keys())
        asset_num_levels = {
            str(k): 1
            for k in asset_keys
        }
        summary = {
            "format": CHUNKED_FLOW_CACHE_FORMAT,
            "format_version": CHUNKED_FLOW_CACHE_FORMAT_VERSION,
            "schema_version": int(payload.get("schema_version", 0)),
            "mode_kind": mode_kind,
            "num_frames": num_frames,
            "patch_grid": list(meta.get("patch_grid", [])),
            "patch_start_idx": int(meta.get("patch_start_idx", 0)),
            "asset_object_keys": asset_keys,
            "asset_num_levels": asset_num_levels,
            "consumers": ["scene_flow", "tokenizer_stage_b"],
            "omitted_fields": [
                "pass1.semantic_logits",
                "pass1.image_tokens_special",
                "pass1.aggregated_tokens_*",
                "pass1.dino_tokens_*",
            ],
        }
        flow_inputs = pass2.get("flow_inputs") if isinstance(pass2, dict) else None
        if isinstance(flow_inputs, dict):
            summary["has_flow_inputs"] = True
            if torch.is_tensor(flow_inputs.get("phase1_coverage")):
                summary["flow_inputs_phase1_coverage_shape"] = list(flow_inputs["phase1_coverage"].shape)
        if mode_kind == "mode_a" and payload.get("phase1_localized") is not None:
            phase1 = payload["phase1_localized"]
            summary["mode_a_edit_frames"] = (
                phase1["frame_idx"].detach().cpu().to(torch.long).unique().tolist()
                if torch.is_tensor(phase1.get("frame_idx"))
                else []
            )
        if mode_kind == "mode_b" and payload.get("mode_b") is not None:
            per_frame = payload["mode_b"]["delete_mask_per_frame"].to(torch.bool)
            summary["mode_b_target_has_delete"] = per_frame.any(dim=1).detach().cpu().tolist()
            summary["mode_b_num_imagined_objects"] = int(
                payload["mode_b"].get("num_imagined_objects", 0)
            )
        _put_info(conn, "summary", summary)

        for key, obj in (
            ("global/meta", meta),
            ("global/object_meta", payload.get("object_meta") or {}),
            ("global/phase1_alignment", payload.get("phase1_alignment") or {}),
        ):
            zbytes, raw_bytes = _put_chunk(conn, cctx, key, obj)
            total_zbytes += zbytes
            total_raw_bytes += raw_bytes
            chunk_count += 1

        cams = pass1["cameras_dggt"]
        for frame in range(num_frames):
            semantic_vehicle_prob, semantic_vehicle_mask = _semantic_vehicle_from_logits(
                _tensor_frame(pass1.get("semantic_logits"), frame)
            )
            chunks = {
                f"frame/{frame:02d}/raw": {
                    "images_u8": _tensor_frame(raw["images_u8"], frame),
                    "sky_mask": _tensor_frame(raw["sky_mask"], frame),
                    "dynamic_mask": _tensor_frame(raw.get("dynamic_mask"), frame),
                },
                f"frame/{frame:02d}/pass1_heads": {
                    "pose_enc": _tensor_frame(pass1["pose_enc"], frame),
                    "gs_map": _tensor_frame(pass1["gs_map"], frame),
                    "depth": _tensor_frame(pass1["depth"], frame),
                    "dynamic_conf": _tensor_frame(pass1["dynamic_conf"], frame),
                    "gs_conf": _tensor_frame(pass1["gs_conf"], frame),
                    "semantic_logits": None,
                    "semantic_vehicle_prob": semantic_vehicle_prob,
                    "semantic_vehicle_mask": semantic_vehicle_mask,
                    "cameras_dggt": {
                        "viewmats": _tensor_frame(cams["viewmats"], frame),
                        "Ks": _tensor_frame(cams["Ks"], frame),
                        "camera_to_world": _tensor_frame(cams["camera_to_world"], frame),
                    },
                },
                f"frame/{frame:02d}/scene_lut": {
                    "F_g_lut_scene_int8": _tensor_frame(pass1["F_g_lut_scene_int8"], frame),
                    "F_g_lut_scene_scale": _tensor_frame(pass1["F_g_lut_scene_scale"], frame),
                },
                f"frame/{frame:02d}/pass2": {
                    "splatted_tok_low_int8": _tensor_frame(pass2["splatted_tok_low_int8"], frame),
                    "splatted_tok_low_scale": _tensor_frame(pass2["splatted_tok_low_scale"], frame),
                },
            }
            if isinstance(flow_inputs, dict):
                chunks[f"frame/{frame:02d}/flow_inputs"] = {
                    "M_preserve": _tensor_frame(flow_inputs["M_preserve"], frame),
                    "M_source": _tensor_frame(flow_inputs["M_source"], frame),
                    "M_dest": _tensor_frame(flow_inputs["M_dest"], frame),
                    "scaffold_pooled": _tensor_frame(flow_inputs["scaffold_pooled"], frame),
                }
            for key, obj in chunks.items():
                zbytes, raw_bytes = _put_chunk(conn, cctx, key, obj)
                total_zbytes += zbytes
                total_raw_bytes += raw_bytes
                chunk_count += 1

        if mode_kind == "mode_a":
            phase1 = payload.get("phase1_localized") or {}
            offsets = phase1.get("frame_gauss_offsets")
            if not torch.is_tensor(offsets):
                raise RuntimeError("Mode-A chunked cache requires phase1_localized.frame_gauss_offsets")
            phase1_meta = {
                "schema_version": int(phase1.get("schema_version", 0)),
                "slot_idx": phase1["slot_idx"],
                "frame_idx": phase1["frame_idx"],
                "source_front_index": phase1["source_front_index"],
                "frame_gauss_offsets": offsets,
            }
            zbytes, raw_bytes = _put_chunk(conn, cctx, "mode_a/phase1_meta", phase1_meta)
            total_zbytes += zbytes
            total_raw_bytes += raw_bytes
            chunk_count += 1
            if isinstance(flow_inputs, dict):
                zbytes, raw_bytes = _put_chunk(
                    conn,
                    cctx,
                    "mode_a/flow_inputs_meta",
                    {
                        "phase1_coverage": flow_inputs.get("phase1_coverage"),
                        "phase4_slots": flow_inputs.get("phase4_slots", []),
                    },
                )
                total_zbytes += zbytes
                total_raw_bytes += raw_bytes
                chunk_count += 1
            for frame in range(num_frames):
                s = int(offsets[frame].item())
                e = int(offsets[frame + 1].item())
                zbytes, raw_bytes = _put_chunk(
                    conn,
                    cctx,
                    f"mode_a/phase1_masks/{frame:02d}",
                    {
                        "delete_mask": phase1["delete_mask"][s:e],
                        "shell_mask": phase1["shell_mask"][s:e],
                    },
                )
                total_zbytes += zbytes
                total_raw_bytes += raw_bytes
                chunk_count += 1

            for obj_key in asset_keys:
                entry = asset[obj_key]
                zbytes, raw_bytes = _put_chunk(
                    conn,
                    cctx,
                    f"mode_a/asset/{obj_key}/meta",
                    {
                        "object_key": obj_key,
                        "num_levels": 1,
                        "source_num_levels": int(entry["F_g_lut_asset_int8"].shape[2]),
                        "source_level_index": int(entry["F_g_lut_asset_int8"].shape[2]) - 1,
                        "has_fit_metrics": entry.get("fit_metrics") is not None,
                    },
                )
                total_zbytes += zbytes
                total_raw_bytes += raw_bytes
                chunk_count += 1
                for frame in range(num_frames):
                    frame_obj = {
                        "I_asset": _tensor_frame(entry["I_asset"], frame),
                        "A_asset": _tensor_frame(entry["A_asset"], frame),
                        "G_asset_dggt_per_frame": entry["G_asset_dggt_per_frame"][frame],
                        "ptr_patch_idx": entry["ptr_patch_idx"][frame],
                        "ptr_visible_mask": entry["ptr_visible_mask"][frame],
                        "ptr_view_n": entry["ptr_view_n"][frame],
                    }
                    if entry.get("fit_metrics") is not None:
                        frame_obj["fit_metrics"] = entry["fit_metrics"][frame]
                    zbytes, raw_bytes = _put_chunk(
                        conn,
                        cctx,
                        f"mode_a/asset/{obj_key}/frame/{frame:02d}",
                        frame_obj,
                    )
                    total_zbytes += zbytes
                    total_raw_bytes += raw_bytes
                    chunk_count += 1
                    source_level = int(entry["F_g_lut_asset_int8"].shape[2]) - 1
                    zbytes, raw_bytes = _put_chunk(
                        conn,
                        cctx,
                        f"mode_a/asset/{obj_key}/frame/{frame:02d}/lut/00",
                        {
                            "F_g_lut_asset_int8": entry["F_g_lut_asset_int8"][
                                frame, :, source_level : source_level + 1, :
                            ],
                            "F_g_lut_asset_scale": entry["F_g_lut_asset_scale"][
                                frame, source_level : source_level + 1
                            ],
                        },
                    )
                    total_zbytes += zbytes
                    total_raw_bytes += raw_bytes
                    chunk_count += 1
        elif mode_kind == "mode_b":
            block = payload.get("mode_b") or {}
            offsets = _frame_gauss_offsets_from_payload(payload)
            mode_b_meta = {
                k: v
                for k, v in block.items()
                if k not in ("delete_mask", "delete_mask_per_frame")
            }
            mode_b_meta["frame_gauss_offsets"] = offsets
            zbytes, raw_bytes = _put_chunk(conn, cctx, "mode_b/meta", mode_b_meta)
            total_zbytes += zbytes
            total_raw_bytes += raw_bytes
            chunk_count += 1
            per_frame = block["delete_mask_per_frame"].to(torch.bool)
            for target in range(num_frames):
                if per_frame.shape[0] == 0:
                    row = torch.zeros((int(offsets[-1].item()),), dtype=torch.bool)
                else:
                    row = per_frame[min(target, int(per_frame.shape[0]) - 1)]
                for source in range(num_frames):
                    s = int(offsets[source].item())
                    e = int(offsets[source + 1].item())
                    zbytes, raw_bytes = _put_chunk(
                        conn,
                        cctx,
                        f"mode_b/delete/{target:02d}/{source:02d}",
                        _pack_bool_tensor(row[s:e]),
                    )
                    total_zbytes += zbytes
                    total_raw_bytes += raw_bytes
                    chunk_count += 1
        else:
            raise RuntimeError(f"Unsupported flow cache mode_kind={mode_kind!r}")

        _put_info(
            conn,
            "stats",
            {
                "chunk_count": chunk_count,
                "chunk_zbytes": total_zbytes,
                "chunk_raw_torch_bytes": total_raw_bytes,
            },
        )
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        os.replace(tmp_path, path)
    except Exception:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise


def load_chunked_flow_cache_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    with _open_chunked_ro(path) as conn:
        return _get_info(conn, "summary")


def append_flow_inputs_to_chunked_flow_cache(
    path: str | os.PathLike[str],
    flow_inputs: dict[str, Any],
    *,
    force: bool = False,
    zstd_level: int = 1,
    unsafe_no_journal: bool = True,
) -> dict[str, Any]:
    """Append SceneFlow fast-path inputs to an existing chunked cache in place.

    This intentionally does not rewrite the SQLite container.  It only inserts
    ``frame/XX/flow_inputs`` chunks, optionally ``mode_a/flow_inputs_meta``,
    and refreshes the summary/stats rows.  It is meant for low-free-space
    backfills where making a full replacement ``.pt`` is not practical.
    """
    if not is_chunked_flow_cache(path):
        raise RuntimeError(f"Flow inputs can only be appended to chunked cache files: {path}")
    zstd = _require_zstd_module()
    path = Path(path)
    before_bytes = int(path.stat().st_size)
    cctx = zstd.ZstdCompressor(level=max(1, min(19, int(zstd_level))))
    conn = sqlite3.connect(str(path))
    written_keys: list[str] = []
    try:
        if bool(unsafe_no_journal):
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
        summary = _get_info(conn, "summary")
        if bool(summary.get("has_flow_inputs", False)) and not bool(force):
            return {
                "path": str(path),
                "skipped": True,
                "reason": "has_flow_inputs",
                "before_bytes": before_bytes,
                "after_bytes": before_bytes,
                "delta_bytes": 0,
                "written_chunks": 0,
            }
        num_frames = int(summary["num_frames"])
        for key in ("M_preserve", "M_source", "M_dest", "scaffold_pooled"):
            value = flow_inputs.get(key)
            if not torch.is_tensor(value) or int(value.shape[0]) != num_frames:
                raise ValueError(
                    f"flow_inputs[{key!r}] must be a tensor with first dim={num_frames}, "
                    f"got {None if value is None else tuple(value.shape)}"
                )
        for frame in range(num_frames):
            key = f"frame/{frame:02d}/flow_inputs"
            if _has_chunk(conn, key) and not bool(force):
                continue
            _put_chunk(
                conn,
                cctx,
                key,
                {
                    "M_preserve": _tensor_frame(flow_inputs["M_preserve"], frame),
                    "M_source": _tensor_frame(flow_inputs["M_source"], frame),
                    "M_dest": _tensor_frame(flow_inputs["M_dest"], frame),
                    "scaffold_pooled": _tensor_frame(flow_inputs["scaffold_pooled"], frame),
                },
            )
            written_keys.append(key)

        if str(summary.get("mode_kind")) == "mode_a":
            key = "mode_a/flow_inputs_meta"
            if bool(force) or not _has_chunk(conn, key):
                _put_chunk(
                    conn,
                    cctx,
                    key,
                    {
                        "phase1_coverage": flow_inputs.get("phase1_coverage"),
                        "phase4_slots": flow_inputs.get("phase4_slots", []),
                    },
                )
                written_keys.append(key)

        summary["has_flow_inputs"] = True
        if torch.is_tensor(flow_inputs.get("phase1_coverage")):
            summary["flow_inputs_phase1_coverage_shape"] = list(flow_inputs["phase1_coverage"].shape)
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
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    after_bytes = int(path.stat().st_size)
    return {
        "path": str(path),
        "skipped": False,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "delta_bytes": after_bytes - before_bytes,
        "written_chunks": len(written_keys),
        "written_keys": written_keys[:8],
    }


def load_chunked_flow_cache_probe(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load just enough metadata for validation and subset sampling."""
    zstd = _require_zstd_module()
    dctx = zstd.ZstdDecompressor()
    with _open_chunked_ro(path) as conn:
        summary = _get_info(conn, "summary")
        meta = _get_chunk(conn, dctx, "global/meta")
        probe: dict[str, Any] = {
            "schema_version": int(summary["schema_version"]),
            "mode_kind": str(summary["mode_kind"]),
            "meta": meta,
            "pass2_splatted_tok_low": {"chunked": True},
            "phase1_localized": None,
            "mode_b": None,
            "_chunked_summary": summary,
        }
        if probe["mode_kind"] == "mode_a":
            phase1 = _get_chunk(conn, dctx, "mode_a/phase1_meta")
            probe["phase1_localized"] = phase1
        elif probe["mode_kind"] == "mode_b":
            probe["mode_b"] = {
                "num_imagined_objects": int(
                    summary.get(
                        "mode_b_num_imagined_objects",
                        max(0, sum(1 for v in summary.get("mode_b_target_has_delete", []) if bool(v))),
                    )
                ),
                "target_has_delete": torch.tensor(
                    summary.get("mode_b_target_has_delete", []), dtype=torch.bool
                ),
            }
    return probe


def _normalize_level_indices(level_indices: list[int] | tuple[int, ...] | None, num_levels: int) -> list[int]:
    if level_indices is None:
        raw_levels = list(range(int(num_levels)))
    else:
        raw_levels = [int(v) for v in level_indices]
    out: list[int] = []
    for raw in raw_levels:
        idx = int(raw)
        if idx < 0:
            idx += int(num_levels)
        if idx < 0 or idx >= int(num_levels):
            raise IndexError(f"level index {raw} is out of range for {num_levels} levels")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("level_indices must contain at least one level")
    return out


def load_chunked_flow_cache_subset(
    path: str | os.PathLike[str],
    subset: torch.Tensor,
    *,
    consumer: str = "scene_flow",
    asset_lut_level_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Load a logical FlowDGGT payload containing only ``subset`` frames."""
    zstd = _require_zstd_module()
    dctx = zstd.ZstdDecompressor()
    subset = subset.detach().cpu().to(torch.long).contiguous()
    subset_list = [int(v) for v in subset.tolist()]
    local_s = int(subset.numel())
    local_subset = torch.arange(local_s, dtype=torch.long)

    with _open_chunked_ro(path) as conn:
        summary = _get_info(conn, "summary")
        meta_full = _get_chunk(conn, dctx, "global/meta")
        fast_scene_flow = consumer in ("scene_flow_fast", "scene_flow_fast_sky") and bool(
            summary.get("has_flow_inputs", False)
        )
        include_fast_raw = consumer == "scene_flow_fast_sky"
        object_meta_full = None if fast_scene_flow else _get_chunk(conn, dctx, "global/object_meta")
        phase1_alignment = {} if fast_scene_flow else _get_chunk(conn, dctx, "global/phase1_alignment")

        raw_chunks = (
            [_get_chunk(conn, dctx, f"frame/{f:02d}/raw") for f in subset_list]
            if (not fast_scene_flow or include_fast_raw)
            else []
        )
        pass1_head_chunks = [] if fast_scene_flow else [_get_chunk(conn, dctx, f"frame/{f:02d}/pass1_heads") for f in subset_list]
        scene_lut_chunks = [_get_chunk(conn, dctx, f"frame/{f:02d}/scene_lut") for f in subset_list]
        pass2_chunks = [_get_chunk(conn, dctx, f"frame/{f:02d}/pass2") for f in subset_list]

        cams = {}
        semantic_vehicle_prob = None
        semantic_vehicle_mask = None
        if not fast_scene_flow:
            cams = {
                key: torch.stack([chunk["cameras_dggt"][key] for chunk in pass1_head_chunks], dim=0).contiguous()
                for key in ("viewmats", "Ks", "camera_to_world")
            }
            semantic_vehicle_prob = _stack_optional(
                [chunk.get("semantic_vehicle_prob") for chunk in pass1_head_chunks]
            )
            semantic_vehicle_mask = _stack_optional(
                [chunk.get("semantic_vehicle_mask") for chunk in pass1_head_chunks]
            )
            if semantic_vehicle_prob is None or semantic_vehicle_mask is None:
                semantic_logits_legacy = _stack_optional(
                    [chunk.get("semantic_logits") for chunk in pass1_head_chunks]
                )
                if semantic_logits_legacy is not None:
                    semantic_vehicle_prob, semantic_vehicle_mask = _semantic_vehicle_from_logits(
                        semantic_logits_legacy
                    )
        pass1 = {
            "cameras_dggt": cams,
            "pose_enc": None if fast_scene_flow else torch.stack([chunk["pose_enc"] for chunk in pass1_head_chunks], dim=0).contiguous(),
            "gs_map": None if fast_scene_flow else torch.stack([chunk["gs_map"] for chunk in pass1_head_chunks], dim=0).contiguous(),
            "depth": None if fast_scene_flow else torch.stack([chunk["depth"] for chunk in pass1_head_chunks], dim=0).contiguous(),
            "dynamic_conf": None if fast_scene_flow else torch.stack([chunk["dynamic_conf"] for chunk in pass1_head_chunks], dim=0).contiguous(),
            "gs_conf": None if fast_scene_flow else torch.stack([chunk["gs_conf"] for chunk in pass1_head_chunks], dim=0).contiguous(),
            "semantic_logits": None,
            "semantic_vehicle_prob": semantic_vehicle_prob,
            "semantic_vehicle_mask": semantic_vehicle_mask,
            "F_g_lut_scene_int8": torch.stack(
                [chunk["F_g_lut_scene_int8"] for chunk in scene_lut_chunks], dim=0
            ).contiguous(),
            "F_g_lut_scene_scale": torch.stack(
                [chunk["F_g_lut_scene_scale"] for chunk in scene_lut_chunks], dim=0
            ).contiguous(),
            "image_tokens_special": None,
            "aggregated_tokens_patch_int8": None,
            "aggregated_tokens_patch_scale": None,
            "aggregated_tokens_special": None,
            "dino_tokens_patch_int8": None,
            "dino_tokens_patch_scale": None,
            "dino_tokens_special": None,
        }
        raw = {} if fast_scene_flow and not include_fast_raw else {
            "images_u8": torch.stack([chunk["images_u8"] for chunk in raw_chunks], dim=0).contiguous(),
            "sky_mask": torch.stack([chunk["sky_mask"] for chunk in raw_chunks], dim=0).contiguous(),
            "dynamic_mask": _stack_optional([chunk.get("dynamic_mask") for chunk in raw_chunks]),
        }
        pass2 = {
            "schema_version": 4,
            "splatted_tok_low_int8": torch.stack(
                [chunk["splatted_tok_low_int8"] for chunk in pass2_chunks], dim=0
            ).contiguous(),
            "splatted_tok_low_scale": torch.stack(
                [chunk["splatted_tok_low_scale"] for chunk in pass2_chunks], dim=0
            ).contiguous(),
            "patch_grid": tuple(int(v) for v in summary.get("patch_grid", [])),
            "num_levels": int(pass2_chunks[0]["splatted_tok_low_int8"].shape[1])
            if pass2_chunks
            else 0,
        }

        payload: dict[str, Any] = {
            "schema_version": int(summary["schema_version"]),
            "mode_kind": str(summary["mode_kind"]),
            "meta": _slice_meta_for_frames(meta_full, subset),
            "raw": raw,
            "object_meta": {} if object_meta_full is None else _slice_object_meta_for_frames(object_meta_full, subset),
            "pass1": pass1,
            "phase1_alignment": phase1_alignment,
            "asset_pass": {},
            "mode_b": None,
            "phase1_localized": None,
            "pass2_splatted_tok_low": pass2,
        }
        if fast_scene_flow:
            flow_chunks = [_get_chunk(conn, dctx, f"frame/{f:02d}/flow_inputs") for f in subset_list]
            payload["flow_inputs"] = {
                "M_preserve": torch.stack([chunk["M_preserve"] for chunk in flow_chunks], dim=0).contiguous(),
                "M_source": torch.stack([chunk["M_source"] for chunk in flow_chunks], dim=0).contiguous(),
                "M_dest": torch.stack([chunk["M_dest"] for chunk in flow_chunks], dim=0).contiguous(),
                "scaffold_pooled": torch.stack([chunk["scaffold_pooled"] for chunk in flow_chunks], dim=0).contiguous(),
            }
            if payload["mode_kind"] == "mode_a" and _has_chunk(conn, "mode_a/flow_inputs_meta"):
                payload["flow_inputs"].update(_get_chunk(conn, dctx, "mode_a/flow_inputs_meta"))
            payload["_fast_scene_flow"] = True

        if consumer == "tokenizer_stage_b":
            return payload

        if payload["mode_kind"] == "mode_a":
            phase1_meta = _get_chunk(conn, dctx, "mode_a/phase1_meta")
            orig_frame = phase1_meta["frame_idx"].to(torch.long).view(-1)
            slot = phase1_meta["slot_idx"]
            sf = phase1_meta["source_front_index"]
            keep = torch.zeros(orig_frame.numel(), dtype=torch.bool)
            local_frame = torch.full((orig_frame.numel(),), -1, dtype=torch.int32)
            local_sf = torch.full((orig_frame.numel(),), -1, dtype=torch.int32)
            remap = {int(f): i for i, f in enumerate(subset_list)}
            num_views = 1
            for i, f_t in enumerate(orig_frame.tolist()):
                f = int(f_t)
                if f in remap:
                    keep[i] = True
                    local_frame[i] = int(remap[f])
                    old_sf = int(sf[i].item())
                    view_off = old_sf - f * int(num_views)
                    local_sf[i] = int(remap[f]) * int(num_views) + view_off
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
            delete_chunks = []
            shell_chunks = []
            offsets = torch.zeros((local_s + 1,), dtype=torch.int64)
            running = 0
            for local_i, f in enumerate(subset_list):
                masks = _get_chunk(conn, dctx, f"mode_a/phase1_masks/{f:02d}")
                delete_chunks.append(masks["delete_mask"].to(torch.bool))
                shell_chunks.append(masks["shell_mask"].to(torch.bool))
                running += int(delete_chunks[-1].numel())
                offsets[local_i + 1] = running
            payload["phase1_localized"] = {
                "schema_version": int(phase1_meta.get("schema_version", 0)),
                "slot_idx": slot.index_select(0, keep_idx),
                "frame_idx": local_frame.index_select(0, keep_idx),
                "source_front_index": local_sf.index_select(0, keep_idx),
                "delete_mask": torch.cat(delete_chunks) if delete_chunks else torch.zeros(0, dtype=torch.bool),
                "shell_mask": torch.cat(shell_chunks) if shell_chunks else torch.zeros(0, dtype=torch.bool),
                "frame_gauss_offsets": offsets,
            }
            asset_pass: dict[int, dict[str, Any]] = {}
            for obj_key in [int(k) for k in summary.get("asset_object_keys", [])]:
                asset_meta = _get_chunk(conn, dctx, f"mode_a/asset/{obj_key}/meta")
                num_levels = int(asset_meta["num_levels"])
                levels = _normalize_level_indices(asset_lut_level_indices, num_levels)
                frame_entries = []
                if not fast_scene_flow:
                    frame_entries = [
                        _get_chunk(conn, dctx, f"mode_a/asset/{obj_key}/frame/{f:02d}")
                        for f in subset_list
                    ]
                    remap_view = {int(f): i for i, f in enumerate(subset_list)}
                    for local_i, frame_entry in enumerate(frame_entries):
                        view_n = frame_entry["ptr_view_n"].to(torch.int32)
                        frame_entry["ptr_view_n"] = torch.tensor(
                            [remap_view.get(int(v), local_i) for v in view_n.tolist()],
                            dtype=torch.int32,
                        )
                lut_frames = []
                scale_frames = []
                for f in subset_list:
                    lut_levels = []
                    scale_levels = []
                    for level in levels:
                        lut_chunk = _get_chunk(
                            conn,
                            dctx,
                            f"mode_a/asset/{obj_key}/frame/{f:02d}/lut/{level:02d}",
                        )
                        lut_levels.append(lut_chunk["F_g_lut_asset_int8"])
                        scale_levels.append(lut_chunk["F_g_lut_asset_scale"])
                    lut_frames.append(torch.cat(lut_levels, dim=1))
                    scale_frames.append(torch.cat(scale_levels, dim=0))
                entry = {
                    "F_g_lut_asset_int8": torch.stack(lut_frames, dim=0).contiguous(),
                    "F_g_lut_asset_scale": torch.stack(scale_frames, dim=0).contiguous(),
                }
                if not fast_scene_flow:
                    entry.update({
                        "I_asset": torch.stack([v["I_asset"] for v in frame_entries], dim=0).contiguous(),
                        "A_asset": torch.stack([v["A_asset"] for v in frame_entries], dim=0).contiguous(),
                        "G_asset_dggt_per_frame": [v["G_asset_dggt_per_frame"] for v in frame_entries],
                        "ptr_patch_idx": [v["ptr_patch_idx"] for v in frame_entries],
                        "ptr_visible_mask": [v["ptr_visible_mask"] for v in frame_entries],
                        "ptr_view_n": [v["ptr_view_n"] for v in frame_entries],
                    })
                    if bool(asset_meta.get("has_fit_metrics", False)):
                        entry["fit_metrics"] = [v.get("fit_metrics") for v in frame_entries]
                asset_pass[obj_key] = entry
            payload["asset_pass"] = asset_pass
        elif payload["mode_kind"] == "mode_b" and not fast_scene_flow:
            mode_b_meta = _get_chunk(conn, dctx, "mode_b/meta")
            rows = []
            for target in subset_list:
                pieces = []
                for source in subset_list:
                    packed = _get_chunk(conn, dctx, f"mode_b/delete/{target:02d}/{source:02d}")
                    pieces.append(_unpack_bool_tensor(packed))
                rows.append(torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.bool))
            delete_per_frame = (
                torch.stack(rows, dim=0).contiguous()
                if rows
                else torch.zeros((local_s, 0), dtype=torch.bool)
            )
            delete_mask = (
                delete_per_frame.any(dim=0)
                if delete_per_frame.numel() > 0
                else torch.zeros((0,), dtype=torch.bool)
            )
            mode_b_meta["delete_mask"] = delete_mask
            mode_b_meta["delete_mask_per_frame"] = delete_per_frame
            payload["mode_b"] = mode_b_meta
    return payload


def load_chunked_flow_cache(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    """Reconstruct a chunked cache into a regular payload.

    This is mainly for manifest peeking and diagnostics.  Training should use
    ``load_chunked_flow_cache_subset`` so it avoids full-clip decompression.
    """
    summary = load_chunked_flow_cache_summary(path)
    subset = torch.arange(int(summary["num_frames"]), dtype=torch.long)
    payload = load_chunked_flow_cache_subset(path, subset, consumer="scene_flow")
    if map_location != "cpu":
        # The legacy helper historically returned CPU cache payloads for
        # training.  Keep this conservative; callers that need CUDA can move
        # individual tensors explicitly.
        return torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: t.to(map_location), payload)
    _ = weights_only
    return payload
