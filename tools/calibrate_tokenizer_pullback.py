#!/usr/bin/env python3
"""LiDAR selection gate for a tokenizer-v2 metric-depth pullback candidate.

The Gaussian/depth round-trip audit fixes the Gaussian correction to identity
and fits a metric-depth candidate on calibration scenes.  This tool evaluates
that frozen depth candidate against sparse Waymo LiDAR z-depth.

The calibration form and coefficients are read from a completed round-trip
audit whose fit used scenes 300--319.  The calibration decision may freeze an
identity, constant, or loglinear candidate.  This script then evaluates,
without refitting, identity versus that one frozen candidate on scenes
320--329:

    identity:  z = z0
    constant:  z = z0 * exp(a)
    loglinear: z = z0 * exp(a + b * log(clamp(z0, 0.5, 80) / 20))

where ``z0 = depth_recon / s_lidar`` is the uncorrected reconstructed metric
depth and ``s_lidar`` is the full-29-frame direct-teacher scale in DGGT units
    per metre.  Absolute relative error is computed at each original sparse
    LiDAR cell before taking medians in the fixed hierarchy pixel -> frame ->
    window -> trunk -> scene.  The five overlapping ten-frame windows are never
    treated as independent observations.  The final paired bootstrap resamples
    scenes only.

Every output is checkpoint- and window-bound.  This tool writes selection
evidence with ``artifact_role=candidate_v2`` and
``eligible_for_training=false``.  It never writes a production pullback; only
the final Phase-1b branch, after the independent Gaussian/depth and
reconstruction gates have also passed, may freeze an eligible artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.retest_scene_flow_gaussian_gauge import (
    D4_CALIBRATION_SCENES,
    D4_HOLDOUT_SCENES,
    SCHEMA_NAME as GAUSSIAN_SCHEMA_NAME,
    SCHEMA_VERSION as GAUSSIAN_SCHEMA_VERSION,
    PATCH_SIZE,
    TARGET_WIDTH,
    TOKENIZER_LEVELS,
    TRUNK_LENGTH,
    WINDOW_LENGTH,
    WINDOW_STARTS,
    _autocast_context,
    _clean_tensor_state,
    _depth_profile_variable_contract,
    _freeze_eval,
    _git_value,
    _joint_to_aggregated,
    _load_exclusion_masks,
    _load_rgb_trunk,
    _module_dtypes,
    _parse_integer_specs,
    _select_image_levels,
    _sparse_level_list,
    _state_mapping,
    _stat_manifest,
    _strict_load_prefixed,
    _strict_load_tokenizer,
    _torch_load_weights,
    _roundtrip_window,
)


SCHEMA_NAME = "tokenizer_metric_depth_lidar_gate"
SCHEMA_VERSION = "2.1.0"
CANDIDATE_ARTIFACT_ROLE = "candidate_v2"
PROFILE_FORMS = ("identity", "constant", "loglinear")
DEPTH_PROFILE_REFERENCE_M = 20.0
DEPTH_PROFILE_CLAMP_M = (0.5, 80.0)
LIDAR_VALID_RANGE_M = (1.0, 80.0)
BOOTSTRAP_SEED = 20260801
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_DATA_ROOT = Path("/data/disk2/lyy_dataset/waymo_processed_dggt/training")
DEFAULT_DGGT_CHECKPOINT = Path("/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt")
DEFAULT_TOKENIZER_CHECKPOINT = (
    REPO_ROOT / "logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt"
)
DEFAULT_REFERENCE_JSON = (
    REPO_ROOT
    / "runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json"
)
DEFAULT_D4_JSON = (
    REPO_ROOT
    / "runs/metric_gauge_retest/v2_gaussian_gauge_300_329_trunks012_d63b34f7.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs/metric_gauge_retest/"
    / "v2_tokenizer_lidar_metric_gate_320_329_<tokenizer_sha8>.json"
)
V1_HISTORICAL_OUTPUT = (
    REPO_ROOT / "runs/metric_gauge_retest/v1_tokenizer_lidar_metric_gate_320_329.json"
)


def _sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_result_self_hash(payload: Mapping[str, Any]) -> str:
    declared = _require_sha256(
        payload.get("result_sha256_excluding_self"),
        name="result_sha256_excluding_self",
    )
    without_self = dict(payload)
    without_self.pop("result_sha256_excluding_self", None)
    computed = _canonical_sha256(without_self)
    if declared != computed:
        raise ValueError("result_sha256_excluding_self mismatch")
    return declared


def _validate_resume_header(payload: Mapping[str, Any]) -> None:
    expected_schema = {
        "name": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "strict": True,
    }
    if payload.get("schema") != expected_schema:
        raise ValueError("existing output schema mismatch")
    if payload.get("status") not in {"running", "complete"}:
        raise ValueError("existing output status must be 'running' or 'complete'")


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _validate_window_contract(
    payload: Mapping[str, Any],
    *,
    expected_window_length: int,
    source: str,
) -> dict[str, Any]:
    """Require every explicit audit window declaration to match this runtime."""

    if (
        isinstance(expected_window_length, bool)
        or not isinstance(expected_window_length, int)
        or expected_window_length <= 0
    ):
        raise ValueError("expected_window_length must be a positive integer")
    lengths: dict[str, int] = {}
    starts: dict[str, list[int]] = {}

    def add_length(path: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{source} {path} must be a positive integer")
        lengths[path] = value

    def add_starts(path: str, value: Any) -> None:
        if value is None:
            return
        if (
            not isinstance(value, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f"{source} {path} must be an integer list")
        starts[path] = list(value)

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        add_length("metadata.window_length", metadata.get("window_length"))
        add_length(
            "metadata.roundtrip_window_length",
            metadata.get("roundtrip_window_length"),
        )
        add_starts(
            "metadata.roundtrip_window_starts",
            metadata.get("roundtrip_window_starts"),
        )
    method = payload.get("method")
    if isinstance(method, Mapping):
        overlap = method.get("window_overlap")
        if isinstance(overlap, Mapping):
            add_length(
                "method.window_overlap.window_length",
                overlap.get("window_length"),
            )
            add_starts("method.window_overlap.starts", overlap.get("starts"))
        schedule = method.get("window_schedule")
        if isinstance(schedule, Mapping):
            add_length("method.window_schedule.length", schedule.get("length"))
            add_starts("method.window_schedule.starts", schedule.get("starts"))
    if not lengths:
        raise ValueError(f"{source} is missing an explicit window length contract")
    if not starts:
        raise ValueError(f"{source} is missing an explicit window starts contract")
    wrong_lengths = {
        path: value for path, value in lengths.items() if value != expected_window_length
    }
    if wrong_lengths:
        raise ValueError(
            f"{source} window length mismatch: expected={expected_window_length}, "
            f"declarations={wrong_lengths}"
        )
    expected_starts = list(WINDOW_STARTS)
    wrong_starts = {path: value for path, value in starts.items() if value != expected_starts}
    if wrong_starts:
        raise ValueError(
            f"{source} window starts mismatch: expected={expected_starts}, "
            f"declarations={wrong_starts}"
        )
    return {
        "expected_window_length": expected_window_length,
        "expected_window_starts": expected_starts,
        "length_declarations": lengths,
        "start_declarations": starts,
    }


def _resolve_output_path(path: Path, *, tokenizer_sha256: str) -> Path:
    tokenizer_sha = _require_sha256(
        tokenizer_sha256, name="tokenizer checkpoint SHA-256"
    )
    resolved = path.expanduser().resolve()
    resolved = resolved.with_name(
        resolved.name.replace("<tokenizer_sha8>", tokenizer_sha[:8])
    )
    if tokenizer_sha[:8] not in resolved.name:
        raise ValueError("v2 output filename must include the tokenizer SHA-256 prefix")
    return resolved


def _validate_candidate_output_path(
    output_path: Path,
    *,
    input_paths: Sequence[Path] = (),
) -> None:
    resolved = output_path.resolve()
    if resolved in {path.resolve() for path in input_paths}:
        raise ValueError("--output must not overwrite an input")
    if resolved == V1_HISTORICAL_OUTPUT.resolve():
        raise ValueError("v2 output must not overwrite the immutable v1 historical result")
    production_root = (REPO_ROOT / "data/scene_gauge").resolve()
    if resolved == production_root or production_root in resolved.parents:
        raise ValueError(
            "candidate_v2 evidence cannot be written under data/scene_gauge; only an "
            "accepted Phase-1b decision may create a production pullback artifact"
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("median requires a non-empty finite sequence")
    return float(np.median(array))


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("mean requires a non-empty finite sequence")
    return float(np.mean(array))


def _depth_profile_correction(
    uncorrected_metric_depth: torch.Tensor,
    *,
    form: str,
    a: float,
    b: float,
) -> torch.Tensor:
    """Evaluate the frozen profile on *uncorrected* reconstructed depth."""

    if form not in PROFILE_FORMS:
        raise ValueError(f"unsupported depth profile form: {form!r}")
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        raise ValueError("depth profile coefficients must be finite")
    depth = uncorrected_metric_depth
    if form == "identity":
        if float(a) != 0.0 or float(b) != 0.0:
            raise ValueError("identity profile requires a=b=0")
        return torch.ones_like(depth)
    if form == "constant":
        if float(b) != 0.0:
            raise ValueError("constant profile requires b=0")
        return torch.full_like(depth, math.exp(float(a)))
    clamped = depth.clamp(min=DEPTH_PROFILE_CLAMP_M[0], max=DEPTH_PROFILE_CLAMP_M[1])
    return torch.exp(
        torch.as_tensor(a, dtype=depth.dtype, device=depth.device)
        + torch.as_tensor(b, dtype=depth.dtype, device=depth.device)
        * torch.log(clamped / DEPTH_PROFILE_REFERENCE_M)
    )


def _sample_hwc_at_lidar_cells(
    values: torch.Tensor,
    valid_cells: np.ndarray,
    *,
    mode: str,
) -> torch.Tensor:
    """Sample a model-canvas HWC map at original LiDAR cell centres."""

    if values.ndim != 3:
        raise ValueError(f"values must be HxWxC, got {tuple(values.shape)}")
    mask = np.asarray(valid_cells, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"valid_cells must be HxW, got {mask.shape}")
    if mode not in {"bilinear", "nearest"}:
        raise ValueError(f"unsupported sampling mode: {mode}")
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return values.new_empty((0, int(values.shape[-1])))
    lidar_height, lidar_width = mask.shape
    x = 2.0 * (
        torch.as_tensor(columns, dtype=torch.float32, device=values.device) + 0.5
    ) / float(lidar_width) - 1.0
    y = 2.0 * (
        torch.as_tensor(rows, dtype=torch.float32, device=values.device) + 0.5
    ) / float(lidar_height) - 1.0
    grid = torch.stack((x, y), dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        values.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0, :].transpose(0, 1).contiguous()


def _frame_metric_errors(
    reconstructed_depth: torch.Tensor,
    lidar_depth_m: np.ndarray,
    *,
    direct_units_per_metre: float,
    profile_form: str,
    profile_a: float,
    profile_b: float,
    support_canvas: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute paired identity/profile AbsRel on the same sparse cells."""

    depth = reconstructed_depth
    if depth.ndim == 3 and int(depth.shape[-1]) == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"reconstructed_depth must be HxW or HxWx1, got {tuple(depth.shape)}")
    lidar = np.asarray(lidar_depth_m)
    if lidar.ndim != 2:
        raise ValueError(f"lidar_depth_m must be HxW, got {lidar.shape}")
    scale = float(direct_units_per_metre)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"direct_units_per_metre must be positive, got {scale}")

    valid_lidar = (
        np.isfinite(lidar)
        & (lidar > LIDAR_VALID_RANGE_M[0])
        & (lidar < LIDAR_VALID_RANGE_M[1])
    )
    sampled = _sample_hwc_at_lidar_cells(
        depth.unsqueeze(-1), valid_lidar, mode="bilinear"
    )[:, 0]
    lidar_values = torch.as_tensor(
        np.asarray(lidar[valid_lidar], dtype=np.float32),
        dtype=torch.float32,
        device=sampled.device,
    )
    valid_prediction = torch.isfinite(sampled) & (sampled > 0.0)
    if support_canvas is not None:
        support = support_canvas
        if support.ndim != 2:
            raise ValueError(f"support_canvas must be HxW, got {tuple(support.shape)}")
        sampled_support = _sample_hwc_at_lidar_cells(
            support.to(dtype=torch.float32).unsqueeze(-1),
            valid_lidar,
            mode="nearest",
        )[:, 0]
        valid_prediction &= sampled_support > 0.5
    sampled = sampled[valid_prediction]
    lidar_values = lidar_values[valid_prediction]
    if sampled.numel() == 0:
        raise RuntimeError("No valid reconstructed depth at eligible LiDAR cells")

    uncorrected_metric = sampled.float() / scale
    correction = _depth_profile_correction(
        uncorrected_metric,
        form=profile_form,
        a=profile_a,
        b=profile_b,
    )
    corrected_metric = uncorrected_metric * correction
    identity_absrel = torch.abs(uncorrected_metric - lidar_values) / lidar_values
    profile_absrel = torch.abs(corrected_metric - lidar_values) / lidar_values
    identity_median = torch.quantile(identity_absrel.float(), 0.5)
    profile_median = torch.quantile(profile_absrel.float(), 0.5)
    return {
        "candidate_form": profile_form,
        "valid_cells": int(sampled.numel()),
        "identity_absrel_median": float(identity_median.item()),
        "candidate_absrel_median": float(profile_median.item()),
        "identity_minus_candidate": float((identity_median - profile_median).item()),
        "uncorrected_metric_depth_median_m": float(
            torch.quantile(uncorrected_metric.float(), 0.5).item()
        ),
        "lidar_depth_median_m": float(torch.quantile(lidar_values.float(), 0.5).item()),
        "correction_median": float(torch.quantile(correction.float(), 0.5).item()),
    }


def _collapse_error_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collapse empty error rows")
    candidate_forms = {str(row["candidate_form"]) for row in rows}
    if len(candidate_forms) != 1 or next(iter(candidate_forms)) not in PROFILE_FORMS:
        raise ValueError(f"error rows must share one supported candidate form: {candidate_forms}")
    candidate_form = next(iter(candidate_forms))
    identity = _median([float(row["identity_absrel_median"]) for row in rows])
    candidate = _median([float(row["candidate_absrel_median"]) for row in rows])
    return {
        "candidate_form": candidate_form,
        "source_count": len(rows),
        "valid_cells_sum_repeated_measurements": int(
            sum(
                int(
                    row.get(
                        "valid_cells",
                        row.get("valid_cells_sum_repeated_measurements", 0),
                    )
                )
                for row in rows
            )
        ),
        "identity_absrel_median": identity,
        "candidate_absrel_median": candidate,
        "identity_minus_candidate": identity - candidate,
    }


def _reference_gauge_validity(reference_case: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen Phase-1a validity thresholds to an existing D2 row."""

    depth = reference_case.get("depth")
    if not isinstance(depth, Mapping):
        raise ValueError("reference case is missing depth audit")
    frame_rows = depth.get("frame_scales")
    if not isinstance(frame_rows, Sequence):
        raise ValueError("reference depth audit is missing frame scales")
    frame_scales = np.asarray(
        [float(row["scale"]) for row in frame_rows if row.get("scale") is not None],
        dtype=np.float64,
    )
    frame_scales = frame_scales[np.isfinite(frame_scales) & (frame_scales > 0.0)]
    if frame_scales.size:
        logs = np.log(frame_scales)
        robust_sigma_log = 1.4826 * float(np.median(np.abs(logs - np.median(logs))))
        robust_cv = math.expm1(robust_sigma_log)
    else:
        robust_cv = float("inf")
    valid_pixels = int(depth.get("valid_sparse_points", 0))
    camera = reference_case.get("camera_scale", {})
    moving = bool(camera.get("scale_valid_2m_span", False)) if isinstance(camera, Mapping) else False
    ratio_value = reference_case.get("camera_over_depth_scale")
    if isinstance(ratio_value, Mapping):
        ratio_value = ratio_value.get("ratio")
    ruler_ratio = float(ratio_value) if ratio_value is not None else None
    camera_crosscheck_pass = (
        not moving
        or (
            ruler_ratio is not None
            and math.isfinite(ruler_ratio)
            and abs(ruler_ratio - 1.0) <= 0.10
        )
    )
    reasons: list[str] = []
    if valid_pixels < 5000:
        reasons.append("valid_lidar_pixels_lt_5000")
    if frame_scales.size < 15:
        reasons.append("valid_frames_lt_15")
    if not math.isfinite(robust_cv) or robust_cv > 0.03:
        reasons.append("frame_robust_cv_gt_3pct")
    if not camera_crosscheck_pass:
        reasons.append("moving_camera_over_lidar_ratio_outside_10pct")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "valid_lidar_pixels": valid_pixels,
        "valid_frame_scales": int(frame_scales.size),
        "frame_robust_cv": robust_cv if math.isfinite(robust_cv) else None,
        "moving_camera_crosscheck_required": moving,
        "camera_over_lidar_ruler_ratio": ruler_ratio,
        "camera_crosscheck_pass": camera_crosscheck_pass,
        "thresholds": {
            "minimum_valid_lidar_pixels": 5000,
            "minimum_valid_frames": 15,
            "maximum_frame_robust_cv": 0.03,
            "maximum_moving_ruler_ratio_deviation": 0.10,
        },
        "source": "Phase-0 D2 reference row; same thresholds preregistered for Phase 1a",
    }


def _load_lidar_depth_trunk(
    scene_root: Path,
    *,
    trunk: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames: list[np.ndarray] = []
    paths: list[str] = []
    grid_hw: tuple[int, int] | None = None
    for local_frame in range(TRUNK_LENGTH):
        global_frame = trunk * TRUNK_LENGTH + local_frame
        path = scene_root / "depth_flows_4" / f"{global_frame:03d}_0.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or int(array.shape[-1]) < 1:
            raise ValueError(f"Expected HxWxC depth-flow array at {path}, got {array.shape}")
        depth = np.asarray(array[..., 0], dtype=np.float32)
        if grid_hw is None:
            grid_hw = tuple(int(value) for value in depth.shape)
        elif tuple(depth.shape) != grid_hw:
            raise ValueError(f"Inconsistent LiDAR grid {depth.shape} vs {grid_hw}")
        frames.append(depth)
        paths.append(str(path.resolve()))
    assert grid_hw is not None
    return frames, {
        "paths": paths,
        "grid_hw": list(grid_hw),
        "channel": 0,
        "semantics": "camera z-depth in metres; original sparse cell centres; zeros are not resized",
    }


def _run_fp32_depth_head(
    selected_joint_levels: Sequence[torch.Tensor],
    *,
    depth_head: torch.nn.Module,
    patch_start_idx: int,
    image_hw: tuple[int, int],
    depth_chunk: int,
) -> torch.Tensor:
    fp32_joint = [tokens.float().contiguous() for tokens in selected_joint_levels]
    aggregated_levels = _sparse_level_list(
        [_joint_to_aggregated(tokens) for tokens in fp32_joint]
    )
    device_type = fp32_joint[0].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        depth, _confidence = depth_head(
            aggregated_levels,
            None,
            patch_start_idx,
            frames_chunk_size=depth_chunk,
            image_hw=image_hw,
        )
    if depth.dtype != torch.float32:
        raise RuntimeError(f"DepthHead must output fp32, got {depth.dtype}")
    if depth.ndim != 5 or int(depth.shape[-1]) != 1:
        raise ValueError(f"Expected depth [B,S,H,W,1], got {tuple(depth.shape)}")
    return depth


def _load_components(
    checkpoint_path: Path,
    tokenizer_checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    from dggt.heads.dpt_head import DPTHead
    from dggt.models.aggregator import Aggregator
    from dggt.models.joint_scene_tokenizer import JointSceneTokenizer

    aggregator = Aggregator(img_size=TARGET_WIDTH, patch_size=PATCH_SIZE, embed_dim=1024)
    depth_head = DPTHead(
        dim_in=2 * 1024,
        output_dim=2,
        activation="exp",
        conf_activation="sigmoid",
    )
    tokenizer = JointSceneTokenizer()

    payload = _torch_load_weights(checkpoint_path)
    cleaned = _clean_tensor_state(_state_mapping(payload, source=checkpoint_path))
    load_info = {
        "aggregator": _strict_load_prefixed(
            aggregator, cleaned, prefix="aggregator.", source=checkpoint_path
        ),
        "depth_head": _strict_load_prefixed(
            depth_head, cleaned, prefix="depth_head.", source=checkpoint_path
        ),
    }
    del payload, cleaned
    gc.collect()

    tokenizer_payload = _torch_load_weights(tokenizer_checkpoint_path)
    load_info["scene_tokenizer"] = _strict_load_tokenizer(
        tokenizer, tokenizer_payload, source=tokenizer_checkpoint_path
    )
    del tokenizer_payload
    gc.collect()

    components = {
        "aggregator": _freeze_eval(aggregator, device),
        "depth_head": _freeze_eval(depth_head, device),
        "scene_tokenizer": _freeze_eval(tokenizer, device),
    }
    load_info["parameter_dtypes_after_move"] = {
        name: _module_dtypes(module) for name, module in components.items()
    }
    if load_info["parameter_dtypes_after_move"]["depth_head"] != ["torch.float32"]:
        raise RuntimeError("DepthHead parameters must remain fp32")
    return components, load_info


def _reference_case_index(payload: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, Sequence):
        raise ValueError("reference JSON is missing cases")
    index: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["scene"]), int(row["trunk"]))
        if key in index:
            raise ValueError(f"duplicate reference case {key}")
        index[key] = row
    return index


def _validate_resumed_case_provenance(
    case: Mapping[str, Any],
    *,
    data_root: Path,
) -> dict[str, Any]:
    """Re-stat every recorded source before reusing a partial case."""

    try:
        scene = int(case["scene"])
        trunk = int(case["trunk"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resumed case is missing a valid scene/trunk key") from exc
    provenance = case.get("input_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"resumed case {(scene, trunk)} is missing input provenance")
    root = data_root.resolve()
    scene_root = (root / f"{scene:03d}").resolve()
    paths: list[str] = []
    for source in ("rgb", "lidar"):
        source_row = provenance.get(source)
        if not isinstance(source_row, Mapping) or not isinstance(source_row.get("paths"), list):
            raise ValueError(f"resumed case {(scene, trunk)} is missing {source} paths")
        paths.extend(str(path) for path in source_row["paths"])
    masks = provenance.get("masks")
    if not isinstance(masks, Mapping) or not isinstance(masks.get("paths"), Mapping):
        raise ValueError(f"resumed case {(scene, trunk)} is missing mask paths")
    for mask_paths in masks["paths"].values():
        if not isinstance(mask_paths, list):
            raise ValueError(f"resumed case {(scene, trunk)} has malformed mask paths")
        paths.extend(str(path) for path in mask_paths)
    if not paths:
        raise ValueError(f"resumed case {(scene, trunk)} records no input paths")
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(scene_root)
        except ValueError as exc:
            raise ValueError(
                f"resumed case {(scene, trunk)} path is outside the current data root: {path}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"resumed case {(scene, trunk)} input is missing: {path}")
    declared_manifest = provenance.get("source_stat_manifest")
    if not isinstance(declared_manifest, Mapping):
        raise ValueError(f"resumed case {(scene, trunk)} is missing its source stat manifest")
    computed_manifest = _stat_manifest(paths, root=root)
    if dict(declared_manifest) != computed_manifest:
        raise ValueError(f"resumed case {(scene, trunk)} source stat manifest changed")
    return computed_manifest


def _extract_frozen_profile(
    gaussian_payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    reference_json_sha256: str,
    expected_window_length: int = WINDOW_LENGTH,
) -> dict[str, Any]:
    if gaussian_payload.get("status") != "complete":
        raise ValueError(
            "Gaussian audit must be complete, "
            f"got {gaussian_payload.get('status')!r}"
        )
    expected_schema = {
        "name": GAUSSIAN_SCHEMA_NAME,
        "version": GAUSSIAN_SCHEMA_VERSION,
        "strict": True,
    }
    if gaussian_payload.get("schema") != expected_schema:
        raise ValueError(
            f"Gaussian audit schema mismatch: expected={expected_schema}, "
            f"got={gaussian_payload.get('schema')!r}"
        )
    requested_checkpoint_sha = _require_sha256(
        checkpoint_sha256, name="requested DGGT checkpoint SHA-256"
    )
    requested_tokenizer_sha = _require_sha256(
        tokenizer_sha256, name="requested tokenizer checkpoint SHA-256"
    )
    requested_reference_sha = _require_sha256(
        reference_json_sha256, name="requested reference JSON SHA-256"
    )
    metadata = gaussian_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Gaussian audit is missing metadata")
    audit_checkpoint_sha = _require_sha256(
        metadata.get("checkpoint_sha256"), name="Gaussian audit DGGT checkpoint SHA-256"
    )
    audit_tokenizer_sha = _require_sha256(
        metadata.get("tokenizer_checkpoint_sha256"),
        name="Gaussian audit tokenizer checkpoint SHA-256",
    )
    audit_reference_sha = _require_sha256(
        metadata.get("reference_result_json_sha256"),
        name="Gaussian audit reference JSON SHA-256",
    )
    if audit_checkpoint_sha != requested_checkpoint_sha:
        raise ValueError(
            "Gaussian audit DGGT checkpoint hash does not match the requested checkpoint"
        )
    if audit_tokenizer_sha != requested_tokenizer_sha:
        raise ValueError(
            "Gaussian audit tokenizer checkpoint hash does not match the requested checkpoint"
        )
    if audit_reference_sha != requested_reference_sha:
        raise ValueError(
            "Gaussian audit reference JSON hash does not match the requested reference JSON"
        )
    split = metadata.get("d4_fixed_split", {})
    if not isinstance(split, Mapping):
        raise ValueError("Gaussian audit is missing the fixed calibration/selection split")
    if split.get("calibration_scenes") != list(D4_CALIBRATION_SCENES):
        raise ValueError("Gaussian calibration split is not the frozen scenes 300-319")
    if split.get("selection_scenes") != list(D4_HOLDOUT_SCENES):
        raise ValueError("Gaussian selection split is not the frozen scenes 320-329")
    window_contract = _validate_window_contract(
        gaussian_payload,
        expected_window_length=expected_window_length,
        source="Gaussian audit",
    )
    audit = gaussian_payload.get("v2_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("Gaussian audit is missing v2_audit")
    if audit.get("formal_audit_complete") is not True:
        raise ValueError("Gaussian formal profile and paired audit is incomplete")
    if audit.get("status") != "complete_formal_v2_audit":
        raise ValueError("Gaussian audit status is not complete_formal_v2_audit")
    coverage = audit.get("formal_audit_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("Gaussian audit is missing formal audit coverage")
    expected_scenes = [*D4_CALIBRATION_SCENES, *D4_HOLDOUT_SCENES]
    if coverage.get("required_scenes") != expected_scenes:
        raise ValueError("Gaussian formal coverage does not require exactly scenes 300-329")
    if coverage.get("required_trunks") != [0, 1, 2]:
        raise ValueError("Gaussian formal coverage does not require exactly trunks 0,1,2")
    if coverage.get("paired_scene_order") != expected_scenes:
        raise ValueError("Gaussian paired audit does not cover exactly scenes 300-329")
    if coverage.get("complete_calibration_profile_fit") is not True:
        raise ValueError("Gaussian calibration profile fit is incomplete")
    try:
        depth_profile = audit["depth_profile"]
        decision = depth_profile["form_decision"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Gaussian audit is missing the frozen depth profile") from exc
    if not isinstance(depth_profile, Mapping) or not isinstance(decision, Mapping):
        raise ValueError("Gaussian depth-profile decision must be a JSON object")
    if depth_profile.get("calibration_scenes") != list(D4_CALIBRATION_SCENES):
        raise ValueError(
            "Gaussian depth-profile calibration scenes are not frozen 300-319"
        )
    if depth_profile.get("selection_scenes") != list(D4_HOLDOUT_SCENES):
        raise ValueError(
            "Gaussian depth-profile selection scenes are not frozen 320-329"
        )
    if depth_profile.get("candidate_forms") != list(PROFILE_FORMS):
        raise ValueError(
            "Gaussian depth-profile candidates must be identity, constant, loglinear"
        )
    if depth_profile.get("fit_data_boundary") != (
        "only calibration rows enter form/coefficient fitting"
    ):
        raise ValueError("Gaussian depth-profile calibration boundary is not frozen")
    if depth_profile.get("selection_role") != (
        "never refits form or coefficients; depth-bin rows are diagnostic-only in this tool"
    ):
        raise ValueError("Gaussian depth-profile selection role may refit the candidate")
    expected_variable = _depth_profile_variable_contract()
    if depth_profile.get("fit_variable") != expected_variable:
        raise ValueError(
            "Gaussian depth-profile fit variable is not uncorrected reconstructed metric depth"
        )
    if depth_profile.get("runtime_variable") != expected_variable:
        raise ValueError(
            "Gaussian depth-profile runtime variable is not uncorrected reconstructed metric depth"
        )
    calibration_rows = depth_profile.get("calibration_scene_bin_rows")
    if not isinstance(calibration_rows, list):
        raise ValueError("Gaussian audit is missing depth-profile calibration rows")
    observed_calibration_scenes: set[int] = set()
    for row in calibration_rows:
        if not isinstance(row, Mapping) or "scene" not in row:
            raise ValueError("Gaussian depth-profile calibration rows must declare a scene")
        try:
            observed_calibration_scenes.add(int(row["scene"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Gaussian depth-profile calibration row scene must be an integer"
            ) from exc
    if observed_calibration_scenes != set(D4_CALIBRATION_SCENES):
        raise ValueError(
            "Gaussian depth-profile fit does not cover all calibration scenes 300-319"
        )
    if decision.get("available") is not True:
        raise ValueError("Gaussian depth-profile calibration decision is unavailable")
    scene_count = decision.get("scene_count")
    if type(scene_count) is not int or scene_count != len(D4_CALIBRATION_SCENES):
        raise ValueError(
            "Gaussian depth-profile decision is not based on all 20 calibration scenes"
        )

    c_gs_recommendation = audit.get("c_gs_recommendation")
    if not isinstance(c_gs_recommendation, Mapping):
        raise ValueError("Gaussian audit is missing the fixed c_gs recommendation")
    if c_gs_recommendation.get("form") != "identity":
        raise ValueError("Gaussian audit c_gs form must be identity")
    c_gs_value = c_gs_recommendation.get("value")
    if isinstance(c_gs_value, bool):
        raise ValueError("Gaussian audit c_gs value must be exactly 1")
    try:
        c_gs_value = float(c_gs_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gaussian audit c_gs value must be exactly 1") from exc
    if not math.isfinite(c_gs_value) or c_gs_value != 1.0:
        raise ValueError("Gaussian audit c_gs value must be exactly 1")

    def finite_coefficient(value: Any, *, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    constant_fit = decision.get("constant_fit")
    loglinear_fit = decision.get("loglinear_fit")
    if not isinstance(constant_fit, Mapping) or not isinstance(loglinear_fit, Mapping):
        raise ValueError(
            "Gaussian calibration decision must contain independently fitted constant_fit "
            "and loglinear_fit candidates"
        )
    constant_a = finite_coefficient(constant_fit.get("a"), name="constant_fit.a")
    constant_b = finite_coefficient(constant_fit.get("b"), name="constant_fit.b")
    if constant_b != 0.0:
        raise ValueError("Gaussian constant_fit must have b=0")
    loglinear_a = finite_coefficient(
        loglinear_fit.get("a"), name="loglinear_fit.a"
    )
    loglinear_b = finite_coefficient(
        loglinear_fit.get("b"), name="loglinear_fit.b"
    )
    candidate_fits = {
        "identity": {"a": 0.0, "b": 0.0, "c_at_20m": 1.0},
        "constant": {
            "a": constant_a,
            "b": 0.0,
            "c_at_20m": math.exp(constant_a),
        },
        "loglinear": {
            "a": loglinear_a,
            "b": loglinear_b,
            "c_at_20m": math.exp(loglinear_a),
        },
    }
    selected_form = decision.get("selected_form")
    if selected_form not in PROFILE_FORMS:
        raise ValueError(f"unsupported Gaussian depth selected_form: {selected_form!r}")
    selected = decision.get("selected", {})
    if not isinstance(selected, Mapping):
        raise ValueError("Gaussian selected depth profile must be a JSON object")
    expected_fit = candidate_fits[str(selected_form)]
    selected_a = finite_coefficient(
        selected.get("a"), name="selected.a"
    )
    selected_b = finite_coefficient(
        selected.get("b"), name="selected.b"
    )
    if not math.isclose(selected_a, expected_fit["a"], rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "Gaussian selected.a does not match the frozen calibration candidate"
        )
    if not math.isclose(selected_b, expected_fit["b"], rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "Gaussian selected.b does not match the frozen calibration candidate"
        )
    if selected_form == "identity":
        equation = "c(z0) = 1"
    elif selected_form == "constant":
        equation = "log c(z0) = a"
    else:
        equation = "log c(z0) = a + b * log(clamp(z0,0.5m,80m)/20m)"
    return {
        "form": selected_form,
        "equation": equation,
        "evaluate_on": "uncorrected_reconstructed_metric_depth_m",
        "boundary_scope": "metric_only; render is forced identity",
        "a": selected_a,
        "b": selected_b,
        "c_at_20m": math.exp(selected_a),
        "reference_depth_m": DEPTH_PROFILE_REFERENCE_M,
        "runtime_depth_clamp_m": list(DEPTH_PROFILE_CLAMP_M),
        "fit_scenes": list(D4_CALIBRATION_SCENES),
        "selection_scenes": list(D4_HOLDOUT_SCENES),
        "calibration_candidate_forms": list(PROFILE_FORMS),
        "calibration_candidate_fits": candidate_fits,
        "calibration_decision_sha256": _canonical_sha256(decision),
        "reference_json_sha256": requested_reference_sha,
        "fit_variable": expected_variable,
        "runtime_variable": dict(expected_variable),
        "window_contract": window_contract,
    }


def _validate_reference_provenance(
    reference_payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    expected_window_length: int = WINDOW_LENGTH,
) -> dict[str, Any]:
    metadata = reference_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("reference artifact is missing metadata")
    reference_checkpoint_sha = _require_sha256(
        metadata.get("checkpoint_sha256"), name="reference DGGT checkpoint SHA-256"
    )
    reference_tokenizer_sha = _require_sha256(
        metadata.get("tokenizer_checkpoint_sha256"),
        name="reference tokenizer checkpoint SHA-256",
    )
    if reference_checkpoint_sha != _require_sha256(
        checkpoint_sha256, name="requested DGGT checkpoint SHA-256"
    ):
        raise ValueError("reference DGGT checkpoint hash mismatch")
    if reference_tokenizer_sha != _require_sha256(
        tokenizer_sha256, name="requested tokenizer checkpoint SHA-256"
    ):
        raise ValueError("reference tokenizer checkpoint hash mismatch")
    return _validate_window_contract(
        reference_payload,
        expected_window_length=expected_window_length,
        source="reference artifact",
    )


def _run_case(
    *,
    scene: int,
    trunk: int,
    data_root: Path,
    components: Mapping[str, torch.nn.Module],
    device: torch.device,
    precision: str,
    depth_chunk: int,
    reference_case: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.time()
    scene_root = data_root / f"{scene:03d}"
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)
    images_cpu, rgb_provenance = _load_rgb_trunk(scene_root, trunk=trunk)
    canvas_hw = (int(images_cpu.shape[-2]), int(images_cpu.shape[-1]))
    lidar_depths, lidar_provenance = _load_lidar_depth_trunk(scene_root, trunk=trunk)
    masks_cpu, mask_provenance = _load_exclusion_masks(
        scene_root,
        trunk=trunk,
        canvas_hw=canvas_hw,
        source_sizes_wh=rgb_provenance["source_sizes_wh"],
    )
    if masks_cpu is None:
        raise RuntimeError("static/non-sky sensitivity requires complete strict masks")
    static_nonsky_cpu = (~masks_cpu["sky"]) & (~masks_cpu["dynamic_fine"])
    input_paths = [*rgb_provenance["paths"], *lidar_provenance["paths"]]
    for values in mask_provenance["paths"].values():
        input_paths.extend(values)
    source_manifest = _stat_manifest(input_paths, root=data_root)

    images = images_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    del images_cpu
    with torch.inference_mode(), _autocast_context(device, precision):
        outputs = components["aggregator"](images)
    if not isinstance(outputs, tuple) or len(outputs) != 5:
        raise RuntimeError("Aggregator did not return the production five-output contract")
    _aggregated, image_tokens, _dino, _image_feature, patch_start_idx = outputs
    patch_start_idx = int(patch_start_idx)
    if patch_start_idx != 5:
        raise RuntimeError(f"Expected patch_start_idx=5, got {patch_start_idx}")
    patch_grid = (canvas_hw[0] // PATCH_SIZE, canvas_hw[1] // PATCH_SIZE)
    selected = _select_image_levels(
        image_tokens,
        patch_start_idx=patch_start_idx,
        patch_grid=patch_grid,
    )
    del outputs, _aggregated, image_tokens, _dino, _image_feature, images
    if device.type == "cuda":
        torch.cuda.empty_cache()

    s_lidar = float(reference_case["depth"]["scale_frame_balanced"])
    if not math.isfinite(s_lidar) or s_lidar <= 0.0:
        raise ValueError(f"invalid full-trunk s_lidar for scene={scene}, trunk={trunk}")
    windows: list[dict[str, Any]] = []
    for start in WINDOW_STARTS:
        end = start + WINDOW_LENGTH
        with torch.inference_mode():
            reconstructed = _roundtrip_window(
                selected,
                start=start,
                patch_start_idx=patch_start_idx,
                patch_grid=patch_grid,
                tokenizer=components["scene_tokenizer"],
                precision=precision,
                device=device,
            )
            recon_depth = _run_fp32_depth_head(
                reconstructed,
                depth_head=components["depth_head"],
                patch_start_idx=patch_start_idx,
                image_hw=canvas_hw,
                depth_chunk=depth_chunk,
            )
        support_rows: dict[str, list[dict[str, Any]]] = {
            "all_lidar": [],
            "static_nonsky": [],
        }
        for window_frame, local_frame in enumerate(range(start, end)):
            common = {
                "local_frame": local_frame,
                "global_frame": trunk * TRUNK_LENGTH + local_frame,
            }
            all_row = _frame_metric_errors(
                recon_depth[0, window_frame],
                lidar_depths[local_frame],
                direct_units_per_metre=s_lidar,
                profile_form=str(profile["form"]),
                profile_a=float(profile["a"]),
                profile_b=float(profile["b"]),
            )
            static_row = _frame_metric_errors(
                recon_depth[0, window_frame],
                lidar_depths[local_frame],
                direct_units_per_metre=s_lidar,
                profile_form=str(profile["form"]),
                profile_a=float(profile["a"]),
                profile_b=float(profile["b"]),
                support_canvas=static_nonsky_cpu[local_frame].to(
                    device=device, non_blocking=True
                ),
            )
            support_rows["all_lidar"].append({**common, **all_row})
            support_rows["static_nonsky"].append({**common, **static_row})
        windows.append(
            {
                "start": start,
                "end_exclusive": end,
                "local_frames": list(range(start, end)),
                "supports": {
                    name: {
                        "frames": rows,
                        "window_balanced": _collapse_error_rows(rows),
                    }
                    for name, rows in support_rows.items()
                },
            }
        )
        del reconstructed, recon_depth
        if device.type == "cuda":
            torch.cuda.empty_cache()

    trunk_balanced: dict[str, Any] = {}
    for support in ("all_lidar", "static_nonsky"):
        trunk_balanced[support] = _collapse_error_rows(
            [window["supports"][support]["window_balanced"] for window in windows]
        )
    validity = _reference_gauge_validity(reference_case)
    result = {
        "scene": f"{scene:03d}",
        "trunk": int(trunk),
        "global_frames": [trunk * TRUNK_LENGTH + index for index in range(TRUNK_LENGTH)],
        "full_29f_direct_units_per_metre": s_lidar,
        "log_metric_scale": math.log(1.0 / s_lidar),
        "phase1a_reference_validity": validity,
        "patch_grid_hw": list(patch_grid),
        "patch_start_idx": patch_start_idx,
        "windows": windows,
        "trunk_balanced": trunk_balanced,
        "input_provenance": {
            "rgb": rgb_provenance,
            "lidar": lidar_provenance,
            "masks": mask_provenance,
            "source_stat_manifest": source_manifest,
        },
        "elapsed_seconds": time.time() - started,
    }
    del selected
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _scene_balanced_rows(
    cases: Sequence[Mapping[str, Any]],
    *,
    support: str,
    require_gauge_valid: bool,
    candidate_form: str,
) -> list[dict[str, Any]]:
    if candidate_form not in PROFILE_FORMS:
        raise ValueError(f"unsupported candidate form: {candidate_form!r}")
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    seen: set[tuple[int, int]] = set()
    for case in cases:
        key = (int(case["scene"]), int(case["trunk"]))
        if key in seen:
            raise ValueError(f"duplicate metric-gate case {key}")
        seen.add(key)
        if require_gauge_valid and not bool(
            case["phase1a_reference_validity"]["valid"]
        ):
            continue
        grouped.setdefault(key[0], []).append(case)
    rows: list[dict[str, Any]] = []
    for scene, scene_cases in sorted(grouped.items()):
        observed_forms = {
            str(case["trunk_balanced"][support]["candidate_form"])
            for case in scene_cases
        }
        if observed_forms != {candidate_form}:
            raise ValueError(
                f"scene {scene:03d} candidate form mismatch: {sorted(observed_forms)}"
            )
        identity = _median(
            [float(case["trunk_balanced"][support]["identity_absrel_median"]) for case in scene_cases]
        )
        candidate = _median(
            [float(case["trunk_balanced"][support]["candidate_absrel_median"]) for case in scene_cases]
        )
        rows.append(
            {
                "scene": f"{scene:03d}",
                "trunk_count": len(scene_cases),
                "trunks": sorted(int(case["trunk"]) for case in scene_cases),
                "candidate_form": candidate_form,
                "identity_absrel": identity,
                "candidate_absrel": candidate,
                "identity_minus_candidate": identity - candidate,
            }
        )
    return rows


def _scene_bootstrap_mean_ci(
    deltas: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    # Sorting makes the deterministic result invariant to input/case order.
    values = np.sort(np.asarray(deltas, dtype=np.float64))
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("scene bootstrap requires at least two finite scene deltas")
    if int(samples) <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(samples), values.size))
    statistics = values[indices].mean(axis=1)
    return float(np.quantile(statistics, 0.025)), float(np.quantile(statistics, 0.975))


def _exact_sign_flip_pvalues(deltas: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0 or values.size > 20 or not np.isfinite(values).all():
        raise ValueError("exact sign flip requires 1--20 finite deltas")
    observed = float(values.mean())
    statistics = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=int(values.size))
        ],
        dtype=np.float64,
    )
    tolerance = 1.0e-15
    return {
        "scene_count": int(values.size),
        "permutation_count": int(statistics.size),
        "observed_mean_delta": observed,
        "one_sided_positive_p": float(np.mean(statistics >= observed - tolerance)),
        "two_sided_p": float(np.mean(np.abs(statistics) >= abs(observed) - tolerance)),
        "role": "sensitivity only; the preregistered gate is the scene-bootstrap CI",
    }


def _summarize_gate(
    cases: Sequence[Mapping[str, Any]],
    *,
    support: str,
    require_gauge_valid: bool,
    candidate_form: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    scene_rows = _scene_balanced_rows(
        cases,
        support=support,
        require_gauge_valid=require_gauge_valid,
        candidate_form=candidate_form,
    )
    expected_scenes = set(D4_HOLDOUT_SCENES)
    observed_scenes = {int(row["scene"]) for row in scene_rows}
    if observed_scenes != expected_scenes:
        raise ValueError(
            f"gate requires all frozen selection scenes; missing={sorted(expected_scenes-observed_scenes)}"
        )
    deltas = [float(row["identity_minus_candidate"]) for row in scene_rows]
    ci_low, ci_high = _scene_bootstrap_mean_ci(
        deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    identity_mean = _mean([float(row["identity_absrel"]) for row in scene_rows])
    candidate_mean = _mean([float(row["candidate_absrel"]) for row in scene_rows])
    mean_delta = _mean(deltas)
    introduced = mean_delta > 0.0 and ci_low > 0.0
    selected_form = candidate_form if introduced else "identity"
    return {
        "support": support,
        "candidate_form": candidate_form,
        "gauge_valid_cases_only": require_gauge_valid,
        "case_count": sum(int(row["trunk_count"]) for row in scene_rows),
        "scene_count": len(scene_rows),
        "scene_rows": scene_rows,
        "scene_balanced_mean_absrel": {
            "identity": identity_mean,
            "candidate": candidate_mean,
        },
        "scene_delta_identity_minus_candidate": {
            "mean": mean_delta,
            "median": _median(deltas),
            "bootstrap_95_ci_for_mean": [ci_low, ci_high],
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "improved_scene_count": int(sum(value > 0.0 for value in deltas)),
            "tied_scene_count": int(sum(value == 0.0 for value in deltas)),
            "relative_improvement_of_scene_mean": (
                (identity_mean - candidate_mean) / identity_mean
                if identity_mean > 0.0
                else None
            ),
        },
        "exact_sign_flip": _exact_sign_flip_pvalues(deltas),
        "gate": {
            "rule": (
                "select the calibration-frozen candidate only when both the point "
                "identity-candidate AbsRel delta and its scene-bootstrap 95% CI lower "
                "bound are strictly > 0; otherwise select identity"
            ),
            "pass": introduced,
            "selected_form": selected_form,
        },
    }


def _synthetic_assertions() -> dict[str, Any]:
    depth = torch.tensor([[0.02, 0.04], [0.06, 0.08]], dtype=torch.float32)
    lidar = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    row = _frame_metric_errors(
        depth,
        lidar,
        direct_units_per_metre=0.02,
        profile_form="constant",
        profile_a=math.log(0.5),
        profile_b=0.0,
    )
    # The 1m cell is excluded by the strict lower bound.  Identity is exact on
    # the remaining cells; applying 0.5 must therefore be worse.
    if row["valid_cells"] != 3 or row["identity_absrel_median"] > 1.0e-6:
        raise AssertionError(row)
    if row["candidate_absrel_median"] < 0.49:
        raise AssertionError(row)
    probes = torch.tensor([0.5, 20.0, 80.0], dtype=torch.float32)
    identity = _depth_profile_correction(
        probes, form="identity", a=0.0, b=0.0
    )
    constant = _depth_profile_correction(
        probes, form="constant", a=math.log(0.9), b=0.0
    )
    loglinear = _depth_profile_correction(
        probes, form="loglinear", a=0.0, b=0.1
    )
    if not torch.equal(identity, torch.ones_like(identity)):
        raise AssertionError(identity)
    if not torch.allclose(constant, torch.full_like(constant, 0.9)):
        raise AssertionError(constant)
    if not (loglinear[0] < loglinear[1] < loglinear[2]):
        raise AssertionError(loglinear)
    positive = [0.01 + 0.001 * index for index in range(10)]
    ci = _scene_bootstrap_mean_ci(positive, samples=2000, seed=BOOTSTRAP_SEED)
    if ci[0] <= 0.0:
        raise AssertionError(ci)
    return {
        "status": "passed",
        "same_cell_absrel": row,
        "candidate_forms": list(PROFILE_FORMS),
        "positive_scene_bootstrap_ci": list(ci),
        "sign_flip": _exact_sign_flip_pvalues(positive),
    }


def _candidate_artifact_contract(
    profile: Mapping[str, Any],
    *,
    selected_form: str,
) -> dict[str, Any]:
    """Build the non-production boundary contract recorded by this gate."""

    frozen_form = profile.get("form")
    if frozen_form not in PROFILE_FORMS:
        raise ValueError(f"unsupported frozen profile form: {frozen_form!r}")
    if selected_form not in {"identity", frozen_form}:
        raise ValueError(
            "selection may choose only identity or the calibration-frozen candidate"
        )
    metric_depth: dict[str, Any] = {"form": selected_form}
    if selected_form != "identity":
        metric_depth.update(
            {
                "a": float(profile["a"]),
                "b": float(profile["b"]),
                "evaluate_on": profile["evaluate_on"],
                "fit_variable": profile["fit_variable"],
                "runtime_variable": profile["runtime_variable"],
                "reference_depth_m": float(profile["reference_depth_m"]),
                "runtime_depth_clamp_m": list(profile["runtime_depth_clamp_m"]),
            }
        )
    return {
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "eligible_for_training": False,
        "boundaries": {
            "render": {
                "depth": {"form": "identity"},
                "gaussian_scale": {"form": "identity", "c_gs": 1.0},
            },
            "metric": {
                "depth": metric_depth,
                "gaussian_scale": {"form": "identity", "c_gs": 1.0},
            },
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=["320-329"])
    parser.add_argument("--trunks", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--depth-chunk", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--expected-window-length",
        type=int,
        default=WINDOW_LENGTH,
        help="Fail-closed tokenizer/audit window contract (production v2 uses 10)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DGGT_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-checkpoint", type=Path, default=DEFAULT_TOKENIZER_CHECKPOINT
    )
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument("--d4-json", type=Path, default=DEFAULT_D4_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--cpu-synthetic-only",
        action="store_true",
        help="Run deterministic pure CPU assertions without loading data or checkpoints",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cpu_synthetic_only:
        print(json.dumps(_synthetic_assertions(), indent=2, sort_keys=True))
        return 0
    if int(args.depth_chunk) <= 0 or int(args.bootstrap_samples) <= 0:
        raise ValueError("depth-chunk and bootstrap-samples must be positive")
    if int(args.expected_window_length) != WINDOW_LENGTH:
        raise ValueError(
            f"this audit implementation requires window length {WINDOW_LENGTH}; "
            f"got {args.expected_window_length}"
        )
    scenes = _parse_integer_specs(args.scenes, name="scenes")
    trunks = _parse_integer_specs(args.trunks, name="trunks")
    if scenes != list(D4_HOLDOUT_SCENES):
        raise ValueError("LiDAR metric gate requires exactly frozen selection scenes 320-329")
    if trunks != [0, 1, 2]:
        raise ValueError("LiDAR metric gate requires exactly trunks 0,1,2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"unsupported device: {device}")
    if device.type == "cpu" and args.precision != "fp32":
        raise ValueError("real CPU execution requires --precision fp32")

    script_path = Path(__file__).resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer_checkpoint.expanduser().resolve()
    reference_path = args.reference_json.expanduser().resolve()
    d4_path = args.d4_json.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    for required in (script_path, checkpoint_path, tokenizer_path, reference_path, d4_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    print("[provenance] hashing script/checkpoints/reference/D4", flush=True)
    hashes = {
        "script_sha256": _sha256(script_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "tokenizer_checkpoint_sha256": _sha256(tokenizer_path),
        "reference_json_sha256": _sha256(reference_path),
        "d4_json_sha256": _sha256(d4_path),
    }
    output_path = _resolve_output_path(
        args.output, tokenizer_sha256=hashes["tokenizer_checkpoint_sha256"]
    )
    _validate_candidate_output_path(
        output_path,
        input_paths=(script_path, checkpoint_path, tokenizer_path, reference_path, d4_path),
    )
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    d4_payload = json.loads(d4_path.read_text(encoding="utf-8"))
    reference_window_contract = _validate_reference_provenance(
        reference_payload,
        checkpoint_sha256=hashes["checkpoint_sha256"],
        tokenizer_sha256=hashes["tokenizer_checkpoint_sha256"],
        expected_window_length=int(args.expected_window_length),
    )
    profile = _extract_frozen_profile(
        d4_payload,
        checkpoint_sha256=hashes["checkpoint_sha256"],
        tokenizer_sha256=hashes["tokenizer_checkpoint_sha256"],
        reference_json_sha256=hashes["reference_json_sha256"],
        expected_window_length=int(args.expected_window_length),
    )
    frozen_profile_sha256 = _canonical_sha256(profile)
    reference_index = _reference_case_index(reference_payload)
    requested_keys = [(scene, trunk) for scene in scenes for trunk in trunks]
    missing = [key for key in requested_keys if key not in reference_index]
    if missing:
        raise ValueError(f"reference JSON is missing requested cases: {missing}")

    signature_payload = {
        "schema": [SCHEMA_NAME, SCHEMA_VERSION],
        "hashes": hashes,
        "scenes": scenes,
        "trunks": trunks,
        "profile": profile,
        "data_root": str(data_root),
        "expected_window_length": int(args.expected_window_length),
        "precision": args.precision,
        "depth_chunk": int(args.depth_chunk),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    signature = _canonical_sha256(signature_payload)
    existing_cases: list[dict[str, Any]] = []
    if args.resume and output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping):
            raise ValueError("existing output root must be an object")
        _validate_resume_header(existing)
        if existing.get("artifact_role") != CANDIDATE_ARTIFACT_ROLE:
            raise ValueError("existing output artifact_role mismatch")
        if existing.get("eligible_for_training") is not False:
            raise ValueError("candidate_v2 output must remain ineligible for training")
        if existing.get("metadata", {}).get("resume_signature_sha256") != signature:
            raise ValueError("existing output resume signature mismatch; choose a new output")
        if existing.get("metadata", {}).get("frozen_profile_sha256") != frozen_profile_sha256:
            raise ValueError("existing output frozen profile hash mismatch")
        if existing.get("frozen_profile") != profile:
            raise ValueError("existing output frozen profile mismatch")
        existing_cases = list(existing.get("cases", []))
        existing_keys = [
            (int(row["scene"]), int(row["trunk"])) for row in existing_cases
        ]
        if len(existing_keys) != len(set(existing_keys)):
            raise ValueError("existing output contains duplicate cases")
        unexpected = sorted(set(existing_keys) - set(requested_keys))
        if unexpected:
            raise ValueError(f"existing output contains unexpected cases: {unexpected}")
        for existing_case in existing_cases:
            _validate_resumed_case_provenance(existing_case, data_root=data_root)
        if existing.get("status") == "complete":
            try:
                existing_selected_form = existing["summary"][
                    "primary_phase1a_valid_all_lidar"
                ]["gate"]["selected_form"]
            except (KeyError, TypeError) as exc:
                raise ValueError("existing complete output is missing its gate decision") from exc
            expected_contract = _candidate_artifact_contract(
                profile, selected_form=str(existing_selected_form)
            )
            if existing.get("boundaries") != expected_contract["boundaries"]:
                raise ValueError("existing complete output boundary contract mismatch")
            _validate_result_self_hash(existing)
            print(f"[done] existing complete result {output_path}", flush=True)
            return 0

    print("[model] strict-loading Aggregator + DepthHead + JointSceneTokenizer", flush=True)
    components, load_info = _load_components(checkpoint_path, tokenizer_path, device)
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "CPU"
    )
    result: dict[str, Any] = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION, "strict": True},
        "status": "running",
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "eligible_for_training": False,
        "boundaries": None,
        "metadata": {
            "created_unix": time.time(),
            "script": str(script_path),
            **hashes,
            "resume_signature_sha256": signature,
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_status_for_script": _git_value(
                ["status", "--short", "--", str(script_path)]
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": device_name,
            "precision": args.precision,
            "production_depth_head_precision": "fp32 with autocast disabled",
            "data_root": str(data_root),
            "checkpoint": str(checkpoint_path),
            "tokenizer_checkpoint": str(tokenizer_path),
            "reference_json": str(reference_path),
            "d4_json": str(d4_path),
            "requested_scenes": scenes,
            "requested_trunks": trunks,
            "window_length": int(args.expected_window_length),
            "window_contract": {
                "reference": reference_window_contract,
                "calibration_audit": profile["window_contract"],
            },
            "frozen_profile_sha256": frozen_profile_sha256,
            "model_load": load_info,
        },
        "method": {
            "fit_boundary": (
                "calibration scenes 300-319 independently define identity, constant, and "
                "loglinear candidates and freeze one; no coefficient is fit or tuned here"
            ),
            "selection_boundary": (
                "fixed scenes 320-329 compare identity only with the one calibration-frozen "
                "candidate; validation/selection, not untouched test"
            ),
            "correction_variable": (
                "uncorrected reconstructed metric depth depth_recon/full_29f_s_lidar"
            ),
            "scope": "render is forced identity; the selected depth profile is metric-boundary only",
            "window_schedule": {"starts": list(WINDOW_STARTS), "length": WINDOW_LENGTH},
            "full_trunk_gauge": "one D2 full-29-frame s_lidar shared by all five windows",
            "lidar": "camera z-depth; finite 1m<z<80m at original sparse cell centres",
            "sampling": "bilinear model-depth sampling at cell centres, align_corners=False; sparse zeros never resized",
            "primary_support": "all valid LiDAR cells; no Gaussian opacity mask",
            "sensitivity_support": "nearest-sampled non-sky and fine-static mask",
            "aggregation": "median pixel->frame->window->trunk->scene; bootstrap mean paired delta by resampling scenes only",
            "gate": (
                "select the frozen candidate iff point identity-candidate delta >0 and the "
                "scene-bootstrap 95% CI lower bound is strictly >0; otherwise identity"
            ),
            "overlap_warning": "five windows are repeated measurements and never bootstrap units",
        },
        "frozen_profile": profile,
        "cases": existing_cases,
        "summary": None,
        "recommendation": None,
    }
    completed = {(int(row["scene"]), int(row["trunk"])) for row in existing_cases}
    total = len(requested_keys)
    for index, (scene, trunk) in enumerate(requested_keys, start=1):
        if (scene, trunk) in completed:
            print(f"[case {index}/{total}] resume-skip scene={scene:03d} trunk={trunk}", flush=True)
            continue
        print(f"[case {index}/{total}] scene={scene:03d} trunk={trunk}", flush=True)
        result["cases"].append(
            _run_case(
                scene=scene,
                trunk=trunk,
                data_root=data_root,
                components=components,
                device=device,
                precision=args.precision,
                depth_chunk=int(args.depth_chunk),
                reference_case=reference_index[(scene, trunk)],
                profile=profile,
            )
        )
        _atomic_write_json(output_path, result)

    cases = sorted(result["cases"], key=lambda row: (int(row["scene"]), int(row["trunk"])))
    result["cases"] = cases
    completed_keys = [(int(row["scene"]), int(row["trunk"])) for row in cases]
    if completed_keys != sorted(requested_keys):
        raise ValueError("completed output does not contain exactly the requested 30 cases")
    primary = _summarize_gate(
        cases,
        support="all_lidar",
        require_gauge_valid=True,
        candidate_form=str(profile["form"]),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    all_cases = _summarize_gate(
        cases,
        support="all_lidar",
        require_gauge_valid=False,
        candidate_form=str(profile["form"]),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    static = _summarize_gate(
        cases,
        support="static_nonsky",
        require_gauge_valid=True,
        candidate_form=str(profile["form"]),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    result["summary"] = {
        "primary_phase1a_valid_all_lidar": primary,
        "sensitivity_all_30_trunks": all_cases,
        "sensitivity_phase1a_valid_static_nonsky": static,
        "case_count": len(cases),
        "phase1a_reference_valid_case_count": sum(
            bool(case["phase1a_reference_validity"]["valid"]) for case in cases
        ),
    }
    selected_form = primary["gate"]["selected_form"]
    if _canonical_sha256(profile) != frozen_profile_sha256:
        raise RuntimeError("frozen calibration profile changed during selection")
    result.update(_candidate_artifact_contract(profile, selected_form=selected_form))
    result["recommendation"] = {
        "render_depth_correction": {
            "form": "identity",
            "reason": "render scope is preregistered identity and is not selected by this LiDAR gate",
        },
        "v2_metric_boundary_depth_correction": {
            "form": selected_form,
            "profile": profile if selected_form != "identity" else None,
            "reason": "fixed identity-vs-frozen-candidate LiDAR gate on scenes 320-329",
        },
        "gaussian_scale_correction": {
            "form": "identity",
            "c_gs": 1.0,
            "reason": "this metric-depth gate never fits or selects c_gs",
        },
        "eligible_for_training": False,
        "blocking_dependency": (
            "candidate evidence remains ineligible until the Phase-1b branch also accepts "
            "the Gaussian/depth practical-equivalence and reconstruction gate"
        ),
    }
    result["status"] = "complete"
    result["metadata"]["completed_unix"] = time.time()
    result["metadata"]["elapsed_seconds"] = (
        result["metadata"]["completed_unix"] - result["metadata"]["created_unix"]
    )
    result["result_sha256_excluding_self"] = _canonical_sha256(result)
    _atomic_write_json(output_path, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(f"[done] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
