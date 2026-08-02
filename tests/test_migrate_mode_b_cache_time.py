from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
import torch

from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
from dggt.utils.flow_cache_io import (
    CHUNKED_FLOW_CACHE_FORMAT,
    CHUNKED_FLOW_CACHE_FORMAT_VERSION,
    CURRENT_FLOW_CACHE_SCHEMA_VERSION,
    _get_chunk,
    _get_info,
    _put_chunk,
    _put_info,
    _require_zstd_module,
    is_current_flow_cache_summary,
    load_chunked_flow_cache_probe,
    load_chunked_flow_cache_subset,
    load_chunked_flow_cache_summary,
    save_flow_cache_chunked,
)
from dggt.utils.gaussian_time import (
    GAUSSIAN_TIME_REPRESENTATION,
    gaussian_timestamps_from_frame_ids,
)
from tools.migrate_mode_b_cache_time import (
    MIGRATION_RECOMPUTED,
    inspect_mode_b_cache,
    migrate_mode_b_cache_inplace,
)


def test_schema_v8_mode_b_is_not_current_or_training_compatible(tmp_path: Path):
    summary = {
        "format": CHUNKED_FLOW_CACHE_FORMAT,
        "format_version": CHUNKED_FLOW_CACHE_FORMAT_VERSION,
        "schema_version": 8,
        "mode_kind": "mode_b",
        "gaussian_time_representation": GAUSSIAN_TIME_REPRESENTATION,
    }
    assert not is_current_flow_cache_summary(summary)
    with pytest.raises(RuntimeError, match="schema_version=8"):
        WaymoFlowCacheDataset._validate_v6_payload(
            {
                "schema_version": 8,
                "mode_kind": "mode_b",
                "meta": {
                    "gaussian_time_representation": GAUSSIAN_TIME_REPRESENTATION,
                },
            },
            cache_path=tmp_path / "legacy_mode_b.pt",
            entry={"mode_kind": "mode_b"},
        )


def _mode_b_payload() -> dict:
    frames = 3
    height = width = 14
    gaussian_count = frames * height * width
    return {
        "schema_version": CURRENT_FLOW_CACHE_SCHEMA_VERSION,
        "mode_kind": "mode_b",
        "meta": {
            "gaussian_time_representation": GAUSSIAN_TIME_REPRESENTATION,
            "metric_box_mapping_mode": "generic_sim3",
            "scene_gauge_table_sha256": None,
            "dggt_checkpoint_sha256": "a" * 64,
            "scene_name": "synthetic",
            "clip_name": "clip_3",
            "frame_indices_scene": torch.tensor([87, 88, 89]),
            "cam_ids": torch.tensor([0]),
            "timestamps": gaussian_timestamps_from_frame_ids(torch.arange(frames)),
            "image_size_model_hw": (height, width),
            "patch_grid": (1, 1),
            "patch_start_idx": 5,
            "raw_image_size_hw": torch.tensor([[1280, 1920]]),
            "num_frames": frames,
            "asset_meta": {},
            "asset_pass_space": "none",
        },
        "raw": {
            "images_u8": torch.zeros(frames, 3, height, width, dtype=torch.uint8),
            "sky_mask": torch.zeros(frames, 1, height, width, dtype=torch.bool),
            "dynamic_mask": torch.zeros(frames, 1, height, width, dtype=torch.bool),
        },
        "object_meta": {},
        "pass1": {
            "cameras_dggt": {
                "viewmats": torch.eye(4).repeat(frames, 1, 1),
                "Ks": torch.eye(3).repeat(frames, 1, 1),
                "camera_to_world": torch.eye(4).repeat(frames, 1, 1),
            },
            "pose_enc": torch.zeros(frames, 9),
            "gs_map": torch.zeros(frames, height, width, 11),
            "depth": torch.ones(frames, height, width, 1),
            "dynamic_conf": torch.zeros(frames, height, width, 1),
            "gs_conf": torch.ones(frames, height, width),
            "semantic_logits": None,
            "F_g_lut_scene_int8": torch.zeros(frames, 1, 1, 2, dtype=torch.int8),
            "F_g_lut_scene_scale": torch.ones(frames, 1),
        },
        "phase1_alignment": {},
        "asset_pass": {},
        "phase1_localized": None,
        "mode_b": {
            "imagined_objects": [{"object_id": 0}],
            "num_imagined_objects": 1,
            "delete_mask": torch.zeros(gaussian_count, dtype=torch.bool),
            "delete_mask_per_frame": torch.zeros(
                frames, gaussian_count, dtype=torch.bool
            ),
        },
        "pass2_splatted_tok_low": {
            "splatted_tok_low_int8": torch.zeros(
                frames, 1, 1, 2, dtype=torch.int8
            ),
            "splatted_tok_low_scale": torch.ones(frames, 1),
        },
    }


def _downgrade_time_metadata(path: Path) -> None:
    zstd = _require_zstd_module()
    with sqlite3.connect(str(path)) as conn:
        summary = _get_info(conn, "summary")
        summary["format_version"] = 1
        summary.pop("gaussian_time_representation", None)
        _put_info(conn, "summary", summary)

        meta = _get_chunk(conn, zstd.ZstdDecompressor(), "global/meta")
        meta.pop("gaussian_time_representation", None)
        meta["timestamps"] = torch.linspace(0.0, 1.0, int(meta["num_frames"]))
        _put_chunk(conn, zstd.ZstdCompressor(level=1), "global/meta", meta)
        conn.commit()


def _immutable_chunk_hashes(path: Path) -> dict[str, str]:
    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute(
            "SELECT key, payload FROM chunks WHERE key != 'global/meta' ORDER BY key"
        ).fetchall()
    return {str(key): hashlib.sha256(bytes(payload)).hexdigest() for key, payload in rows}


def _nonderived_chunk_hashes(path: Path) -> dict[str, str]:
    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute("SELECT key, payload FROM chunks ORDER BY key").fetchall()
    return {
        str(key): hashlib.sha256(bytes(payload)).hexdigest()
        for key, payload in rows
        if key != "global/meta"
        and not str(key).endswith("/pass2")
        and not str(key).endswith("/flow_inputs")
    }


def test_mode_b_time_migration_is_inplace_exact_and_idempotent(tmp_path: Path):
    path = tmp_path / "mode_b.pt"
    save_flow_cache_chunked(_mode_b_payload(), path, zstd_level=1)
    _downgrade_time_metadata(path)

    before_hashes = _immutable_chunk_hashes(path)
    before_inode = path.stat().st_ino
    inspection = inspect_mode_b_cache(path)
    assert inspection["status"] == "needs_migration"
    assert inspection["source_format_version"] == 1
    assert inspection["old_timestamp_last"] == 1.0
    assert inspection["new_timestamp_last"] == 0.5

    result = migrate_mode_b_cache_inplace(path)

    assert result["status"] == "migrated"
    assert path.stat().st_ino == before_inode
    assert _immutable_chunk_hashes(path) == before_hashes
    summary = load_chunked_flow_cache_summary(path)
    probe = load_chunked_flow_cache_probe(path)
    assert summary["format_version"] == CHUNKED_FLOW_CACHE_FORMAT_VERSION
    assert summary["gaussian_time_representation"] == GAUSSIAN_TIME_REPRESENTATION
    assert is_current_flow_cache_summary(summary)
    assert probe["meta"]["gaussian_time_representation"] == GAUSSIAN_TIME_REPRESENTATION
    assert torch.equal(
        probe["meta"]["timestamps"],
        torch.tensor([0.0, 0.25, 0.5]),
    )
    WaymoFlowCacheDataset._validate_v6_payload(
        probe,
        cache_path=path,
        entry={"mode_kind": "mode_b"},
    )

    subset = load_chunked_flow_cache_subset(
        path, torch.tensor([1, 2]), consumer="scene_flow"
    )
    assert torch.equal(subset["meta"]["timestamps"], torch.tensor([0.25, 0.5]))
    rerun = migrate_mode_b_cache_inplace(path)
    assert rerun["status"] == "already_current"
    assert rerun["requires_pass2_recompute"] is False


def test_mode_b_time_migration_rejects_non_mode_b(tmp_path: Path):
    path = tmp_path / "mode_a_disguised.pt"
    save_flow_cache_chunked(_mode_b_payload(), path, zstd_level=1)
    with sqlite3.connect(str(path)) as conn:
        summary = _get_info(conn, "summary")
        summary["mode_kind"] = "mode_a"
        conn.execute(
            "UPDATE info SET value=? WHERE key='summary'",
            (json.dumps(summary),),
        )
        conn.commit()

    with pytest.raises(ValueError, match="non-Mode-B"):
        inspect_mode_b_cache(path)


def test_mode_b_time_migration_rejects_noncontiguous_frames(tmp_path: Path):
    path = tmp_path / "bad_frames.pt"
    payload = _mode_b_payload()
    payload["meta"]["frame_indices_scene"] = torch.tensor([87, 89, 90])
    save_flow_cache_chunked(payload, path, zstd_level=1)
    _downgrade_time_metadata(path)

    with pytest.raises(ValueError, match="not a contiguous"):
        inspect_mode_b_cache(path)


def test_nonempty_mode_b_recomputes_only_timestamp_dependent_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "nonempty.pt"
    payload = _mode_b_payload()
    payload["mode_b"]["delete_mask"][0] = True
    payload["mode_b"]["delete_mask_per_frame"][0, 0] = True
    save_flow_cache_chunked(payload, path, zstd_level=1)
    _downgrade_time_metadata(path)
    before_hashes = _nonderived_chunk_hashes(path)

    def fake_recompute(payload_arg, *, device, chunk_channels):
        assert str(device) == "cpu"
        assert chunk_channels == 17
        assert torch.equal(
            payload_arg["meta"]["timestamps"], torch.tensor([0.0, 0.25, 0.5])
        )
        frames = 3
        return {
            "splatted_tok_low_int8": torch.full(
                (frames, 1, 1, 2), 7, dtype=torch.int8
            ),
            "splatted_tok_low_scale": torch.full((frames, 1), 2.0),
            "flow_inputs": {
                "M_preserve": torch.full((frames, 1, 1), 0.75),
                "M_source": torch.full((frames, 1, 1), 0.25),
                "M_dest": torch.zeros(frames, 1, 1),
                "scaffold_pooled": torch.full((frames, 1, 7), 3.0),
                "phase1_coverage": torch.zeros(0, frames, dtype=torch.bool),
                "phase4_slots": [],
            },
        }

    monkeypatch.setattr(
        "tools.migrate_mode_b_cache_time._recompute_timestamp_dependent_payload",
        fake_recompute,
    )

    inspection = inspect_mode_b_cache(path)
    assert inspection["requires_pass2_recompute"] is True
    result = migrate_mode_b_cache_inplace(
        path,
        device="cpu",
        chunk_channels=17,
    )

    assert result["pass2_recomputed"] is True
    assert _nonderived_chunk_hashes(path) == before_hashes
    summary = load_chunked_flow_cache_summary(path)
    assert summary["gaussian_time_migration"] == MIGRATION_RECOMPUTED
    assert summary["has_flow_inputs"] is True
    migrated = load_chunked_flow_cache_subset(
        path,
        torch.tensor([0, 2]),
        consumer="scene_flow_fast",
    )
    assert torch.equal(
        migrated["pass2_splatted_tok_low"]["splatted_tok_low_int8"],
        torch.full((2, 1, 1, 2), 7, dtype=torch.int8),
    )
    assert torch.equal(
        migrated["flow_inputs"]["M_preserve"],
        torch.full((2, 1, 1), 0.75),
    )
