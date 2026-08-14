#!/usr/bin/env python3
"""Freeze verified tokenizer-v2 evidence into a production pullback artifact.

The fail-closed projection requires the formal Gaussian practical-equivalence
audit and LiDAR selection gate plus explicit production authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.scene_gauge import (
    PULLBACK_ARTIFACT_ROLE,
    PULLBACK_DEPTH_EVALUATION,
    PULLBACK_DEPTH_FORMS,
    PULLBACK_DEPTH_VARIABLE_CONTRACT,
    PULLBACK_GS_SCALE_RULE,
    PULLBACK_LOG_METRIC_SCALE_UNITS,
    PULLBACK_RUNTIME_CONTRACT_VERSION,
    PULLBACK_SCHEMA_NAME,
    PULLBACK_SCHEMA_VERSION,
    PULLBACK_V2_SMOKE_THRESHOLDS,
    SCENE_GAUGE_REPRESENTATION,
    load_pullback_calibration,
)


V2_LIDAR_SCHEMA_NAME = "tokenizer_metric_depth_lidar_gate"
TOKENIZER_V2_GENERATION = "t0_v2"
FROZEN_FIT_SCENES = list(range(300, 320))
FROZEN_SELECTION_SCENES = list(range(320, 330))
DEFAULT_DIAGNOSTIC_JSON = (
    REPO_ROOT
    / "runs/metric_gauge_retest/v2_tokenizer_lidar_metric_gate_320_329_d63b34f7.json"
)
DEFAULT_GAUSSIAN_AUDIT_JSON = (
    REPO_ROOT
    / "runs/metric_gauge_retest/v2_gaussian_gauge_300_329_trunks012_d63b34f7.json"
)
DEFAULT_REFERENCE_JSON = (
    REPO_ROOT
    / "runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json"
)
DEFAULT_RECONSTRUCTION_SMOKE_JSON = (
    REPO_ROOT / "runs/tokenizer_v2_cuda0_render_smoke_300_d63b34f7/smoke.json"
)
DEFAULT_RECONSTRUCTION_SMOKE_SELECTION_MANIFEST = (
    REPO_ROOT / "runs/tokenizer_v2_fixed_selection_300.json"
)
DEFAULT_RECONSTRUCTION_SMOKE_VISUAL = (
    REPO_ROOT
    / "runs/tokenizer_v2_cuda0_render_smoke_300_d63b34f7/visuals/step_100000_frames_10/00_training_300.jpg"
)
DEFAULT_TOKENIZER_CHECKPOINT = (
    REPO_ROOT
    / "logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt"
)
DEFAULT_DGGT_CHECKPOINT = Path(
    "/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/scene_gauge/pullback_d63b34f7.json"
V2_GAUSSIAN_SCHEMA_NAME = "scene_flow_gaussian_gauge_retest"
V2_GAUSSIAN_SCHEMA_VERSION = "2.4.0"
V2_LIDAR_SCHEMA_VERSION = "2.1.0"
V2_WINDOW_STARTS = [0, 5, 10, 14, 19]
V2_REQUIRED_SCENES = list(range(300, 330))
V2_REQUIRED_TRUNKS = [0, 1, 2]
V2_EQUIVALENCE_MARGIN = [0.95, 1.05]
V2_DEPTH_VARIABLE_CONTRACT = PULLBACK_DEPTH_VARIABLE_CONTRACT
V2_GAUSSIAN_SUPPORT = "primary_static_nonsky_opacity_0p05"
V2_GAUSSIAN_BOOTSTRAP_SAMPLES = 10000
V2_GAUSSIAN_BOOTSTRAP_SEED = 20260805
V2_LIDAR_BOOTSTRAP_SAMPLES = 10000
V2_LIDAR_BOOTSTRAP_SEED = 20260801
V2_LIDAR_DEPTH_CHUNK_RECOVERY_MAX = 4096


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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _expect(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


def _resolve_v2_lidar_depth_chunk(
    metadata: Mapping[str, Any],
    *,
    signature_without_depth_chunk: Mapping[str, Any],
    resume_signature_sha256: str,
) -> int:
    raw_depth_chunk = metadata.get("depth_chunk")
    if raw_depth_chunk is not None:
        if (
            isinstance(raw_depth_chunk, bool)
            or not isinstance(raw_depth_chunk, int)
            or raw_depth_chunk <= 0
        ):
            raise ValueError("v2 LiDAR metadata.depth_chunk must be a positive integer")
        return raw_depth_chunk

    # Schema 2.1 results signed this CLI argument but did not persist it in metadata.
    # Recover it only when the signed payload has exactly one bounded positive match.
    matches = []
    for candidate in range(1, V2_LIDAR_DEPTH_CHUNK_RECOVERY_MAX + 1):
        signature = dict(signature_without_depth_chunk)
        signature["depth_chunk"] = candidate
        if _canonical_sha256(signature) == resume_signature_sha256:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "v2 LiDAR resume signature does not uniquely recover depth_chunk "
            f"in [1,{V2_LIDAR_DEPTH_CHUNK_RECOVERY_MAX}]"
        )
    return matches[0]


def _expect_close(actual: Any, expected: float, *, name: str, atol: float = 1.0e-12) -> float:
    value = _finite(actual, name=name)
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{name} mismatch: actual={value!r}, expected={expected!r}")
    return value


def _scene_bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> list[float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size < 2 or not np.isfinite(ordered).all():
        raise ValueError("scene bootstrap requires at least two finite scene values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, ordered.size, size=(int(samples), ordered.size))
    statistics = ordered[indices].mean(axis=1)
    return [float(np.quantile(statistics, 0.025)), float(np.quantile(statistics, 0.975))]


def _gaussian_scene_bootstrap_ratio_ci(
    scene_logs: Sequence[float], *, samples: int, seed: int
) -> list[float]:
    values = np.asarray(scene_logs, dtype=np.float64)
    if values.size != 30 or not np.isfinite(values).all():
        raise ValueError("Gaussian equivalence bootstrap requires 30 finite scene logs")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(samples), values.size), endpoint=False)
    medians = np.median(values[indices], axis=1)
    quantiles = np.quantile(medians, (0.025, 0.975))
    return [math.exp(float(quantiles[0])), math.exp(float(quantiles[1]))]


def _exact_sign_flip(values: Sequence[float]) -> dict[str, Any]:
    deltas = np.asarray(values, dtype=np.float64)
    if deltas.size != 10 or not np.isfinite(deltas).all():
        raise ValueError("formal exact sign flip requires ten finite scene deltas")
    observed = float(deltas.mean())
    statistics = np.asarray(
        [
            float(np.mean(deltas * np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=10)
        ],
        dtype=np.float64,
    )
    tolerance = 1.0e-15
    return {
        "scene_count": 10,
        "permutation_count": 1024,
        "observed_mean_delta": observed,
        "one_sided_positive_p": float(np.mean(statistics >= observed - tolerance)),
        "two_sided_p": float(
            np.mean(np.abs(statistics) >= abs(observed) - tolerance)
        ),
        "role": "sensitivity only; the preregistered gate is the scene-bootstrap CI",
    }


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _declared_script_sha256(payload: Mapping[str, Any], *, name: str) -> str:
    metadata = _mapping(payload.get("metadata"), name=f"{name} metadata")
    declared = _require_sha256(
        metadata.get("script_sha256"), name=f"{name} script SHA-256"
    )
    raw_path = metadata.get("script")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} metadata.script must be non-empty")
    script_path = Path(raw_path).expanduser()
    if not script_path.is_absolute():
        script_path = REPO_ROOT / script_path
    script_path = script_path.resolve()
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    _expect(_sha256(script_path), declared, name=f"{name} script SHA-256")
    return declared


def _validate_case_coverage(
    payload: Mapping[str, Any],
    *,
    scenes: Sequence[int],
    trunks: Sequence[int],
    name: str,
    require_patch_grid: bool = True,
) -> tuple[list[int], int]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{name} cases must be a JSON list")
    expected = [(int(scene), int(trunk)) for scene in scenes for trunk in trunks]
    observed: list[tuple[int, int]] = []
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, name=f"{name} case {index}")
        try:
            scene = int(case["scene"])
            trunk = int(case["trunk"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} case {index} has invalid scene/trunk") from error
        observed.append((scene, trunk))
    if observed != expected:
        raise ValueError(
            f"{name} must contain exact ordered scene/trunk coverage; "
            f"observed_count={len(observed)}, expected_count={len(expected)}"
        )
    patch_grid = _extract_patch_grid(cases) if require_patch_grid else []
    return patch_grid, len(observed)


def _validate_reference_signature(reference: Mapping[str, Any]) -> str:
    metadata = _mapping(reference.get("metadata"), name="reference metadata")
    fields = _mapping(
        metadata.get("resume_signature_fields"), name="reference signature fields"
    )
    declared = _require_sha256(
        metadata.get("resume_signature_sha256"),
        name="reference resume signature SHA-256",
    )
    _expect(
        _canonical_sha256(fields),
        declared,
        name="reference resume signature SHA-256",
    )
    _expect(
        fields.get("data_root"),
        metadata.get("data_root"),
        name="reference resume-signature data_root",
    )
    return declared


def _validate_reference_windows(reference: Mapping[str, Any]) -> None:
    cases = reference.get("cases")
    if not isinstance(cases, list):
        raise ValueError("reference cases must be a JSON list")
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, name=f"reference case {index}")
        roundtrip = _mapping(
            case.get("tokenizer_roundtrip"),
            name=f"reference case {index} tokenizer_roundtrip",
        )
        windows = roundtrip.get("windows")
        if not isinstance(windows, list):
            raise ValueError(f"reference case {index} windows must be a JSON list")
        starts = [int(_mapping(row, name="reference window").get("start")) for row in windows]
        ends = [
            int(_mapping(row, name="reference window").get("end_exclusive"))
            for row in windows
        ]
        _expect(starts, V2_WINDOW_STARTS, name=f"reference case {index} window starts")
        _expect(
            ends,
            [start + 10 for start in V2_WINDOW_STARTS],
            name=f"reference case {index} window ends",
        )


def _validate_source_self_hash(payload: Mapping[str, Any]) -> str:
    declared = payload.get("result_sha256_excluding_self")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("diagnostic source is missing result_sha256_excluding_self")
    without_self = dict(payload)
    without_self.pop("result_sha256_excluding_self", None)
    _expect(
        _canonical_sha256(without_self),
        declared,
        name="diagnostic result_sha256_excluding_self",
    )
    return declared


def _extract_patch_grid(cases: Sequence[Any]) -> list[int]:
    grids: set[tuple[int, int]] = set()
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, name=f"diagnostic case {index}")
        raw_grid = case.get("patch_grid_hw")
        if (
            not isinstance(raw_grid, list)
            or len(raw_grid) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_grid)
            or any(value <= 0 for value in raw_grid)
        ):
            raise ValueError(f"diagnostic case {index} has invalid patch_grid_hw")
        grids.add((raw_grid[0], raw_grid[1]))
    if len(grids) != 1:
        raise ValueError(f"diagnostic cases do not share one patch grid: {sorted(grids)}")
    return list(next(iter(grids)))


def _validate_reconstruction_render_smoke(
    smoke: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    *,
    smoke_path: Path,
    smoke_sha256: str,
    selection_manifest_path: Path,
    selection_manifest_sha256: str,
    visual_path: Path,
    visual_sha256: str,
    tokenizer_sha256: str,
    dggt_sha256: str,
) -> dict[str, Any]:
    for path, declared, name in (
        (smoke_path, smoke_sha256, "smoke JSON"),
        (selection_manifest_path, selection_manifest_sha256, "smoke selection manifest"),
        (visual_path, visual_sha256, "smoke visual"),
    ):
        _require_sha256(declared, name=f"{name} SHA-256")
        _expect(_sha256(path), declared, name=f"{name} SHA-256")

    _expect(smoke.get("schema"), "tokenizer_v2_cuda_render_smoke", name="smoke schema")
    _expect(smoke.get("schema_version"), "1.0.0", name="smoke schema version")
    _expect(smoke.get("artifact_role"), "diagnostic_smoke", name="smoke artifact role")
    _expect(smoke.get("eligible_for_training"), False, name="smoke eligibility")
    _expect(smoke.get("device"), "cuda:0", name="smoke device")
    _expect(smoke.get("precision"), "bf16", name="smoke precision")
    _expect(
        smoke.get("depth_gaussian_heads_precision"),
        "fp32",
        name="smoke DepthHead/GaussianHead precision",
    )
    checkpoint = _mapping(smoke.get("checkpoint_sha256"), name="smoke checkpoint hashes")
    _expect(checkpoint.get("tokenizer"), tokenizer_sha256, name="smoke tokenizer SHA-256")
    _expect(checkpoint.get("dggt"), dggt_sha256, name="smoke DGGT SHA-256")
    _expect(
        smoke.get("selection_manifest_sha256"),
        selection_manifest_sha256,
        name="smoke selection manifest SHA-256",
    )

    _expect(
        selection_manifest.get("schema"),
        "tokenizer_v2_fixed_scene_selection",
        name="smoke selection schema",
    )
    _expect(
        selection_manifest.get("schema_version"),
        "1.0.0",
        name="smoke selection schema version",
    )
    manifest_cases = selection_manifest.get("cases")
    if not isinstance(manifest_cases, list):
        raise ValueError("smoke selection manifest cases must be a list")

    case = _mapping(smoke.get("case"), name="smoke case")
    _expect(case.get("config"), "step_100000_frames_10", name="smoke config")
    _expect(case.get("step"), 100000, name="smoke step")
    _expect(case.get("frame_count"), 10, name="smoke frame count")
    dataset_index = case.get("dataset_index")
    if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
        raise ValueError("smoke dataset_index must be an integer")
    matching = [
        _mapping(row, name="smoke selection case")
        for row in manifest_cases
        if isinstance(row, Mapping) and row.get("dataset_index") == dataset_index
    ]
    if len(matching) != 1:
        raise ValueError("smoke selection manifest must contain exactly one matching dataset_index")
    manifest_case = matching[0]
    _expect(case.get("scene"), manifest_case.get("scene"), name="smoke selected scene")
    _expect(case.get("clip"), manifest_case.get("clip"), name="smoke selected clip")
    selections = _mapping(manifest_case.get("selections"), name="smoke selections")
    selected_ten = _mapping(selections.get("10"), name="smoke ten-frame selection")
    _expect(
        case.get("frame_indices"),
        selected_ten.get("global_frame_indices"),
        name="smoke selected global frames",
    )
    frame_indices = case.get("frame_indices")
    if (
        not isinstance(frame_indices, list)
        or len(frame_indices) != 10
        or any(isinstance(value, bool) or not isinstance(value, int) for value in frame_indices)
        or frame_indices != sorted(set(frame_indices))
    ):
        raise ValueError("smoke frame indices must be ten unique sorted integers")

    with Image.open(visual_path) as visual:
        visual_format = visual.format
        visual_mode = visual.mode
        visual_size = list(visual.size)
        visual.load()
    _expect(visual_format, "JPEG", name="smoke visual format")
    _expect(visual_mode, "RGB", name="smoke visual mode")

    metrics = _mapping(case.get("metrics"), name="smoke metrics")
    metric_names = (
        "render_gt_psnr_db",
        "render_gt_ssim",
        "render_gt_lpips",
        "direct_gt_psnr_db",
        "direct_gt_ssim",
        "direct_gt_lpips",
        "render_direct_psnr_db",
        "render_direct_ssim",
        "render_direct_lpips",
        "depth_recon_over_direct",
        "depth_recon_vs_direct_absrel",
        "point_xyz_relative_error",
        "gs_recon_over_direct",
        "paired_gs_over_depth",
        "gs_axis_anisotropy_log_rms",
        "support_pixels_median_per_frame",
    )
    observed = {name: _finite(metrics.get(name), name=f"smoke {name}") for name in metric_names}
    thresholds = PULLBACK_V2_SMOKE_THRESHOLDS

    def inside(metric_name: str, threshold_name: str) -> bool:
        lower, upper = thresholds[threshold_name]
        return float(lower) <= observed[metric_name] <= float(upper)

    passed = (
        observed["render_direct_psnr_db"] >= thresholds["render_direct_psnr_min_db"]
        and observed["render_direct_ssim"] >= thresholds["render_direct_ssim_min"]
        and observed["render_direct_lpips"] <= thresholds["render_direct_lpips_max"]
        and observed["render_gt_psnr_db"]
        >= observed["direct_gt_psnr_db"] - thresholds["render_gt_psnr_drop_max_db"]
        and observed["render_gt_ssim"]
        >= observed["direct_gt_ssim"] - thresholds["render_gt_ssim_drop_max"]
        and observed["render_gt_lpips"]
        <= observed["direct_gt_lpips"] + thresholds["render_gt_lpips_increase_max"]
        and inside("depth_recon_over_direct", "depth_recon_over_direct_range")
        and observed["depth_recon_vs_direct_absrel"]
        <= thresholds["depth_recon_vs_direct_absrel_max"]
        and observed["point_xyz_relative_error"] <= thresholds["point_xyz_relative_error_max"]
        and inside("gs_recon_over_direct", "gs_recon_over_direct_range")
        and inside("paired_gs_over_depth", "paired_gs_over_depth_range")
        and observed["gs_axis_anisotropy_log_rms"]
        <= thresholds["gs_axis_anisotropy_log_rms_max"]
        and observed["support_pixels_median_per_frame"]
        >= thresholds["support_pixels_median_per_frame_min"]
    )
    if not passed:
        raise ValueError("reconstruction/render smoke exceeds the frozen no-obvious-regression limits")
    return {
        "path": _display_path(smoke_path),
        "sha256": smoke_sha256,
        "selection_manifest": {
            "path": _display_path(selection_manifest_path),
            "sha256": selection_manifest_sha256,
        },
        "visual": {
            "path": _display_path(visual_path),
            "sha256": visual_sha256,
            "format": visual_format,
            "mode": visual_mode,
            "size_wh": visual_size,
        },
        "device": "cuda:0",
        "precision": "bf16",
        "depth_gaussian_heads_precision": "fp32",
        "case": {
            "config": case["config"],
            "step": int(case["step"]),
            "frame_count": int(case["frame_count"]),
            "dataset_index": int(dataset_index),
            "scene": str(case["scene"]),
            "clip": str(case["clip"]),
            "frame_indices": list(frame_indices),
        },
        "thresholds": json.loads(json.dumps(PULLBACK_V2_SMOKE_THRESHOLDS)),
        "observed": observed,
        "passed": True,
    }


def _validate_lidar_method_and_cases(diagnostic: Mapping[str, Any]) -> None:
    method = _mapping(diagnostic.get("method"), name="v2 LiDAR method")
    schedule = _mapping(method.get("window_schedule"), name="LiDAR window schedule")
    _expect(schedule.get("starts"), V2_WINDOW_STARTS, name="LiDAR window starts")
    _expect(schedule.get("length"), 10, name="LiDAR window length")
    expected_fields = {
        "full_trunk_gauge": "one D2 full-29-frame s_lidar shared by all five windows",
        "lidar": "camera z-depth; finite 1m<z<80m at original sparse cell centres",
        "sampling": "bilinear model-depth sampling at cell centres, align_corners=False; sparse zeros never resized",
        "primary_support": "all valid LiDAR cells; no Gaussian opacity mask",
        "sensitivity_support": "nearest-sampled non-sky and fine-static mask",
        "aggregation": "median pixel->frame->window->trunk->scene; bootstrap mean paired delta by resampling scenes only",
    }
    for key, expected in expected_fields.items():
        _expect(method.get(key), expected, name=f"LiDAR method.{key}")
    cases = diagnostic.get("cases")
    if not isinstance(cases, list):
        raise ValueError("v2 LiDAR cases must be a list")
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, name=f"v2 LiDAR case {index}")
        global_frames = case.get("global_frames")
        trunk = case.get("trunk")
        if isinstance(trunk, bool) or not isinstance(trunk, int):
            raise ValueError(f"v2 LiDAR case {index} trunk must be an integer")
        expected_frames = list(range(trunk * 29, trunk * 29 + 29))
        if not isinstance(global_frames, list) or global_frames != expected_frames:
            raise ValueError(f"v2 LiDAR case {index} must bind one complete 29-frame trunk")
        full_trunk_scale = _finite(
            case.get("full_29f_direct_units_per_metre"),
            name=f"v2 LiDAR case {index} full-trunk scale",
        )
        if full_trunk_scale <= 0.0:
            raise ValueError(f"v2 LiDAR case {index} full-trunk scale must be positive")
        windows = case.get("windows")
        if not isinstance(windows, list) or len(windows) != 5:
            raise ValueError(f"v2 LiDAR case {index} must contain five windows")
        starts = [int(_mapping(row, name="LiDAR window").get("start")) for row in windows]
        ends = [int(_mapping(row, name="LiDAR window").get("end_exclusive")) for row in windows]
        _expect(starts, V2_WINDOW_STARTS, name=f"v2 LiDAR case {index} window starts")
        _expect(
            ends,
            [start + 10 for start in V2_WINDOW_STARTS],
            name=f"v2 LiDAR case {index} window ends",
        )


def _validated_lidar_summary(
    raw: Any,
    *,
    name: str,
    expected_support: str,
    expected_gauge_filter: bool,
    candidate_form: str,
    require_all_trunks: bool,
) -> dict[str, Any]:
    summary = _mapping(raw, name=name)
    _expect(summary.get("support"), expected_support, name=f"{name} support")
    _expect(
        summary.get("gauge_valid_cases_only"),
        expected_gauge_filter,
        name=f"{name} gauge filter",
    )
    _expect(summary.get("candidate_form"), candidate_form, name=f"{name} candidate form")
    _expect(summary.get("scene_count"), 10, name=f"{name} scene count")
    rows = summary.get("scene_rows")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError(f"{name} must contain ten scene rows")
    sanitized_rows: list[dict[str, Any]] = []
    identities: list[float] = []
    candidates: list[float] = []
    deltas: list[float] = []
    case_count = 0
    for expected_scene, raw_row in zip(FROZEN_SELECTION_SCENES, rows):
        row = _mapping(raw_row, name=f"{name} scene row")
        _expect(int(row.get("scene")), expected_scene, name=f"{name} scene order")
        _expect(row.get("candidate_form"), candidate_form, name=f"{name} scene form")
        trunk_count = row.get("trunk_count")
        trunks = row.get("trunks")
        if (
            isinstance(trunk_count, bool)
            or not isinstance(trunk_count, int)
            or not 1 <= trunk_count <= 3
            or not isinstance(trunks, list)
            or trunks != sorted(set(trunks))
            or len(trunks) != trunk_count
            or any(trunk not in V2_REQUIRED_TRUNKS for trunk in trunks)
        ):
            raise ValueError(f"{name} has invalid per-scene trunk coverage")
        if require_all_trunks:
            _expect(trunks, V2_REQUIRED_TRUNKS, name=f"{name} all-trunk coverage")
        identity = _finite(row.get("identity_absrel"), name=f"{name} identity AbsRel")
        candidate = _finite(row.get("candidate_absrel"), name=f"{name} candidate AbsRel")
        delta = _finite(row.get("identity_minus_candidate"), name=f"{name} delta")
        _expect_close(identity - candidate, delta, name=f"{name} row delta")
        identities.append(identity)
        candidates.append(candidate)
        deltas.append(delta)
        case_count += trunk_count
        sanitized_rows.append(
            {
                "scene": expected_scene,
                "trunk_count": trunk_count,
                "trunks": list(trunks),
                "candidate_form": candidate_form,
                "identity_absrel": identity,
                "candidate_absrel": candidate,
                "identity_minus_candidate": delta,
            }
        )
    _expect(summary.get("case_count"), case_count, name=f"{name} case count")
    means = _mapping(summary.get("scene_balanced_mean_absrel"), name=f"{name} means")
    identity_mean = float(np.mean(np.asarray(identities, dtype=np.float64)))
    candidate_mean = float(np.mean(np.asarray(candidates, dtype=np.float64)))
    _expect_close(means.get("identity"), identity_mean, name=f"{name} identity mean")
    _expect_close(means.get("candidate"), candidate_mean, name=f"{name} candidate mean")
    delta_record = _mapping(
        summary.get("scene_delta_identity_minus_candidate"), name=f"{name} delta record"
    )
    _expect(
        delta_record.get("bootstrap_samples"),
        V2_LIDAR_BOOTSTRAP_SAMPLES,
        name=f"{name} bootstrap samples",
    )
    _expect(
        delta_record.get("bootstrap_seed"),
        V2_LIDAR_BOOTSTRAP_SEED,
        name=f"{name} bootstrap seed",
    )
    delta_mean = float(np.mean(np.asarray(deltas, dtype=np.float64)))
    _expect_close(delta_record.get("mean"), delta_mean, name=f"{name} delta mean")
    expected_ci = _scene_bootstrap_mean_ci(
        deltas,
        samples=V2_LIDAR_BOOTSTRAP_SAMPLES,
        seed=V2_LIDAR_BOOTSTRAP_SEED,
    )
    observed_ci = delta_record.get("bootstrap_95_ci_for_mean")
    if not isinstance(observed_ci, list) or len(observed_ci) != 2:
        raise ValueError(f"{name} bootstrap CI must contain two values")
    for index, expected in enumerate(expected_ci):
        _expect_close(observed_ci[index], expected, name=f"{name} bootstrap CI {index}")
    improved = sum(value > 0.0 for value in deltas)
    tied = sum(value == 0.0 for value in deltas)
    _expect(
        delta_record.get("improved_scene_count"), improved, name=f"{name} improved scenes"
    )
    _expect(delta_record.get("tied_scene_count"), tied, name=f"{name} tied scenes")
    expected_sign_flip = _exact_sign_flip(deltas)
    observed_sign_flip = _mapping(summary.get("exact_sign_flip"), name=f"{name} sign flip")
    for key, expected in expected_sign_flip.items():
        if isinstance(expected, float):
            _expect_close(observed_sign_flip.get(key), expected, name=f"{name} sign flip {key}")
        else:
            _expect(observed_sign_flip.get(key), expected, name=f"{name} sign flip {key}")
    gate = _mapping(summary.get("gate"), name=f"{name} gate")
    gate_pass = delta_mean > 0.0 and expected_ci[0] > 0.0
    selected_form = candidate_form if gate_pass else "identity"
    _expect(gate.get("pass"), gate_pass, name=f"{name} gate pass")
    _expect(gate.get("selected_form"), selected_form, name=f"{name} selected form")
    return {
        "support": expected_support,
        "gauge_valid_cases_only": expected_gauge_filter,
        "candidate_form": candidate_form,
        "selected_form": selected_form,
        "case_count": case_count,
        "scene_count": 10,
        "identity_absrel": identity_mean,
        "candidate_absrel": candidate_mean,
        "scene_delta_mean": delta_mean,
        "scene_delta_bootstrap_95_ci": expected_ci,
        "improved_scene_count": improved,
        "gate_pass": gate_pass,
        "bootstrap": {
            "unit": "scene",
            "samples": V2_LIDAR_BOOTSTRAP_SAMPLES,
            "seed": V2_LIDAR_BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "scene_rows": sanitized_rows,
        "exact_sign_flip": expected_sign_flip,
    }


def build_v2_production_artifact(
    diagnostic: Mapping[str, Any],
    gaussian: Mapping[str, Any],
    reference: Mapping[str, Any],
    reconstruction_smoke: Mapping[str, Any],
    smoke_selection_manifest: Mapping[str, Any],
    *,
    diagnostic_path: Path,
    diagnostic_sha256: str,
    gaussian_path: Path,
    gaussian_sha256: str,
    reference_path: Path,
    reference_sha256: str,
    reconstruction_smoke_path: Path,
    reconstruction_smoke_sha256: str,
    smoke_selection_manifest_path: Path,
    smoke_selection_manifest_sha256: str,
    reconstruction_smoke_visual_path: Path,
    reconstruction_smoke_visual_sha256: str,
    tokenizer_sha256: str,
    dggt_sha256: str,
) -> dict[str, Any]:
    """Project complete v2 evidence into the strict production schema."""

    for value, name in (
        (diagnostic_sha256, "LiDAR result SHA-256"),
        (gaussian_sha256, "Gaussian result SHA-256"),
        (reference_sha256, "reference result SHA-256"),
        (reconstruction_smoke_sha256, "reconstruction/render smoke SHA-256"),
        (smoke_selection_manifest_sha256, "smoke selection manifest SHA-256"),
        (reconstruction_smoke_visual_sha256, "reconstruction/render visual SHA-256"),
        (tokenizer_sha256, "tokenizer checkpoint SHA-256"),
        (dggt_sha256, "DGGT checkpoint SHA-256"),
    ):
        _require_sha256(value, name=name)

    smoke_evidence = _validate_reconstruction_render_smoke(
        reconstruction_smoke,
        smoke_selection_manifest,
        smoke_path=reconstruction_smoke_path,
        smoke_sha256=reconstruction_smoke_sha256,
        selection_manifest_path=smoke_selection_manifest_path,
        selection_manifest_sha256=smoke_selection_manifest_sha256,
        visual_path=reconstruction_smoke_visual_path,
        visual_sha256=reconstruction_smoke_visual_sha256,
        tokenizer_sha256=tokenizer_sha256,
        dggt_sha256=dggt_sha256,
    )

    source_schema = _mapping(diagnostic.get("schema"), name="v2 LiDAR schema")
    _expect(source_schema.get("name"), V2_LIDAR_SCHEMA_NAME, name="v2 LiDAR schema.name")
    _expect(source_schema.get("version"), V2_LIDAR_SCHEMA_VERSION, name="v2 LiDAR schema.version")
    _expect(source_schema.get("strict"), True, name="v2 LiDAR schema.strict")
    _expect(diagnostic.get("status"), "complete", name="v2 LiDAR status")
    _expect(diagnostic.get("artifact_role"), "candidate_v2", name="v2 LiDAR artifact_role")
    _expect(diagnostic.get("eligible_for_training"), False, name="v2 LiDAR eligibility")
    source_result_sha = _validate_source_self_hash(diagnostic)
    lidar_script_sha = _declared_script_sha256(diagnostic, name="v2 LiDAR gate")

    source_metadata = _mapping(diagnostic.get("metadata"), name="v2 LiDAR metadata")
    _expect(
        source_metadata.get("tokenizer_checkpoint_sha256"),
        tokenizer_sha256,
        name="v2 LiDAR tokenizer SHA-256",
    )
    _expect(
        source_metadata.get("checkpoint_sha256"),
        dggt_sha256,
        name="v2 LiDAR DGGT SHA-256",
    )
    _expect(
        source_metadata.get("d4_json_sha256"),
        gaussian_sha256,
        name="v2 LiDAR Gaussian result SHA-256",
    )
    _expect(
        source_metadata.get("reference_json_sha256"),
        reference_sha256,
        name="v2 LiDAR reference result SHA-256",
    )
    _expect(source_metadata.get("window_length"), 10, name="v2 LiDAR window length")
    _expect(
        source_metadata.get("requested_scenes"),
        FROZEN_SELECTION_SCENES,
        name="v2 LiDAR selection scenes",
    )
    _expect(
        source_metadata.get("requested_trunks"),
        V2_REQUIRED_TRUNKS,
        name="v2 LiDAR requested trunks",
    )
    source_data_root = source_metadata.get("data_root")
    if not isinstance(source_data_root, str) or not source_data_root:
        raise ValueError("v2 LiDAR metadata.data_root must be non-empty")
    source_resume_signature_sha = _require_sha256(
        source_metadata.get("resume_signature_sha256"),
        name="v2 LiDAR resume signature SHA-256",
    )
    lidar_grid, lidar_case_count = _validate_case_coverage(
        diagnostic,
        scenes=FROZEN_SELECTION_SCENES,
        trunks=V2_REQUIRED_TRUNKS,
        name="v2 LiDAR gate",
    )
    _expect(lidar_case_count, 30, name="v2 LiDAR complete case count")
    _validate_lidar_method_and_cases(diagnostic)

    profile = _mapping(diagnostic.get("frozen_profile"), name="v2 frozen profile")
    profile_sha = _require_sha256(
        source_metadata.get("frozen_profile_sha256"),
        name="v2 frozen profile SHA-256",
    )
    _expect(_canonical_sha256(profile), profile_sha, name="v2 frozen profile self SHA-256")
    profile_form = profile.get("form")
    if profile_form not in PULLBACK_DEPTH_FORMS:
        raise ValueError(f"unsupported v2 frozen profile form: {profile_form!r}")
    profile_a = _finite(profile.get("a"), name="v2 frozen profile a")
    profile_b = _finite(profile.get("b"), name="v2 frozen profile b")
    if profile_form == "identity" and (profile_a != 0.0 or profile_b != 0.0):
        raise ValueError("identity v2 frozen profile requires a=b=0")
    if profile_form == "constant" and profile_b != 0.0:
        raise ValueError("constant v2 frozen profile requires b=0")
    _expect(profile.get("fit_scenes"), FROZEN_FIT_SCENES, name="v2 profile fit scenes")
    _expect(
        profile.get("selection_scenes"),
        FROZEN_SELECTION_SCENES,
        name="v2 profile selection scenes",
    )
    _expect(
        profile.get("calibration_candidate_forms"),
        list(PULLBACK_DEPTH_FORMS),
        name="v2 profile candidate forms",
    )
    _expect(
        profile.get("boundary_scope"),
        "metric_only; render is forced identity",
        name="v2 profile boundary scope",
    )
    _expect(
        profile.get("fit_variable"),
        V2_DEPTH_VARIABLE_CONTRACT,
        name="v2 profile fit variable",
    )
    _expect(
        profile.get("runtime_variable"),
        V2_DEPTH_VARIABLE_CONTRACT,
        name="v2 profile runtime variable",
    )
    _expect(
        profile.get("reference_depth_m"), 20.0, name="v2 profile reference depth"
    )
    _expect(
        profile.get("runtime_depth_clamp_m"),
        [0.5, 80.0],
        name="v2 profile depth clamp",
    )
    profile_window = _mapping(profile.get("window_contract"), name="v2 profile window contract")
    _expect(
        profile_window.get("expected_window_length"),
        10,
        name="v2 profile window length",
    )
    _expect(
        profile_window.get("expected_window_starts"),
        V2_WINDOW_STARTS,
        name="v2 profile window starts",
    )
    calibration_decision_sha = _require_sha256(
        profile.get("calibration_decision_sha256"),
        name="v2 calibration decision SHA-256",
    )
    _expect(
        profile.get("reference_json_sha256"),
        reference_sha256,
        name="v2 profile reference result SHA-256",
    )

    gaussian_schema = _mapping(gaussian.get("schema"), name="v2 Gaussian schema")
    _expect(gaussian_schema.get("name"), V2_GAUSSIAN_SCHEMA_NAME, name="Gaussian schema.name")
    _expect(
        gaussian_schema.get("version"),
        V2_GAUSSIAN_SCHEMA_VERSION,
        name="Gaussian schema.version",
    )
    _expect(gaussian_schema.get("strict"), True, name="Gaussian schema.strict")
    _expect(gaussian.get("status"), "complete", name="v2 Gaussian status")
    gaussian_script_sha = _declared_script_sha256(gaussian, name="v2 Gaussian audit")
    gaussian_metadata = _mapping(gaussian.get("metadata"), name="v2 Gaussian metadata")
    _expect(
        gaussian_metadata.get("checkpoint_sha256"),
        dggt_sha256,
        name="v2 Gaussian DGGT SHA-256",
    )
    _expect(
        gaussian_metadata.get("tokenizer_checkpoint_sha256"),
        tokenizer_sha256,
        name="v2 Gaussian tokenizer SHA-256",
    )
    _expect(
        gaussian_metadata.get("reference_result_json_sha256"),
        reference_sha256,
        name="v2 Gaussian reference result SHA-256",
    )
    _expect(
        gaussian_metadata.get("requested_scenes"),
        V2_REQUIRED_SCENES,
        name="v2 Gaussian requested scenes",
    )
    _expect(
        gaussian_metadata.get("requested_trunks"),
        V2_REQUIRED_TRUNKS,
        name="v2 Gaussian requested trunks",
    )
    _expect(
        gaussian_metadata.get("data_root"),
        source_data_root,
        name="Gaussian/LiDAR data_root",
    )
    gaussian_bootstrap = _mapping(
        gaussian_metadata.get("paired_equivalence_bootstrap"),
        name="Gaussian bootstrap metadata",
    )
    _expect(gaussian_bootstrap.get("unit"), "scene", name="Gaussian bootstrap unit")
    _expect(
        gaussian_bootstrap.get("samples"),
        V2_GAUSSIAN_BOOTSTRAP_SAMPLES,
        name="Gaussian bootstrap samples",
    )
    _expect(
        gaussian_bootstrap.get("seed"),
        V2_GAUSSIAN_BOOTSTRAP_SEED,
        name="Gaussian bootstrap seed",
    )
    gaussian_grid, gaussian_case_count = _validate_case_coverage(
        gaussian,
        scenes=V2_REQUIRED_SCENES,
        trunks=V2_REQUIRED_TRUNKS,
        name="v2 Gaussian audit",
    )
    _expect(gaussian_case_count, 90, name="v2 Gaussian complete case count")
    _expect(gaussian_grid, lidar_grid, name="Gaussian/LiDAR patch grid")
    gaussian_method = _mapping(gaussian.get("method"), name="Gaussian method")
    overlap = _mapping(gaussian_method.get("window_overlap"), name="Gaussian window overlap")
    _expect(overlap.get("starts"), V2_WINDOW_STARTS, name="Gaussian window starts")
    _expect(overlap.get("window_length"), 10, name="Gaussian window length")
    primary_protocol = _mapping(
        gaussian_method.get("primary_paired_practical_equivalence_gate"),
        name="Gaussian primary equivalence protocol",
    )
    _expect(primary_protocol.get("margin"), V2_EQUIVALENCE_MARGIN, name="Gaussian margin")
    _expect(
        primary_protocol.get("support"), V2_GAUSSIAN_SUPPORT, name="Gaussian support"
    )
    _expect(
        primary_protocol.get("bootstrap_unit"), "scene", name="Gaussian method bootstrap unit"
    )
    _expect(
        primary_protocol.get("bootstrap_samples"),
        V2_GAUSSIAN_BOOTSTRAP_SAMPLES,
        name="Gaussian method bootstrap samples",
    )
    _expect(
        primary_protocol.get("bootstrap_seed"),
        V2_GAUSSIAN_BOOTSTRAP_SEED,
        name="Gaussian method bootstrap seed",
    )
    _expect(
        primary_protocol.get("c_gs_policy"),
        "identity fixed at 1; image-quality metrics are outside this audit",
        name="Gaussian c_gs policy",
    )

    gaussian_v2_audit = _mapping(gaussian.get("v2_audit"), name="Gaussian v2 audit")
    _expect(
        gaussian_v2_audit.get("status"),
        "complete_formal_v2_audit",
        name="Gaussian formal audit status",
    )
    _expect(
        gaussian_v2_audit.get("formal_audit_complete"),
        True,
        name="Gaussian formal audit completeness",
    )
    coverage = _mapping(
        gaussian_v2_audit.get("formal_audit_coverage"), name="Gaussian coverage"
    )
    _expect(coverage.get("required_scenes"), V2_REQUIRED_SCENES, name="Gaussian coverage scenes")
    _expect(coverage.get("required_trunks"), V2_REQUIRED_TRUNKS, name="Gaussian coverage trunks")
    _expect(coverage.get("paired_scene_order"), V2_REQUIRED_SCENES, name="Gaussian paired scenes")
    _expect(
        coverage.get("complete_calibration_profile_fit"),
        True,
        name="Gaussian calibration profile coverage",
    )
    depth_profile = _mapping(
        gaussian_v2_audit.get("depth_profile"), name="Gaussian depth profile"
    )
    _expect(
        depth_profile.get("calibration_scenes"),
        FROZEN_FIT_SCENES,
        name="Gaussian calibration split",
    )
    _expect(
        depth_profile.get("selection_scenes"),
        FROZEN_SELECTION_SCENES,
        name="Gaussian selection split",
    )
    _expect(
        depth_profile.get("candidate_forms"),
        list(PULLBACK_DEPTH_FORMS),
        name="Gaussian candidate forms",
    )
    _expect(
        depth_profile.get("fit_data_boundary"),
        "only calibration rows enter form/coefficient fitting",
        name="Gaussian fit data boundary",
    )
    _expect(
        depth_profile.get("selection_role"),
        "never refits form or coefficients; depth-bin rows are diagnostic-only in this tool",
        name="Gaussian selection role",
    )
    _expect(
        depth_profile.get("fit_variable"),
        V2_DEPTH_VARIABLE_CONTRACT,
        name="Gaussian c_depth fit variable",
    )
    _expect(
        depth_profile.get("runtime_variable"),
        V2_DEPTH_VARIABLE_CONTRACT,
        name="Gaussian c_depth runtime variable",
    )
    calibration_rows = depth_profile.get("calibration_scene_bin_rows")
    if not isinstance(calibration_rows, list):
        raise ValueError("Gaussian audit is missing c_depth calibration rows")
    observed_fit_scenes = {
        int(row["scene"])
        for row in calibration_rows
        if isinstance(row, Mapping) and "scene" in row
    }
    _expect(
        observed_fit_scenes,
        set(FROZEN_FIT_SCENES),
        name="Gaussian c_depth fit scene coverage",
    )
    selection_rows = depth_profile.get("selection_scene_bin_rows_diagnostic_only")
    if not isinstance(selection_rows, list):
        raise ValueError("Gaussian audit is missing diagnostic selection rows")
    observed_selection_scenes = {
        int(row["scene"])
        for row in selection_rows
        if isinstance(row, Mapping) and "scene" in row
    }
    _expect(
        observed_selection_scenes,
        set(FROZEN_SELECTION_SCENES),
        name="Gaussian diagnostic selection scene coverage",
    )
    decision = _mapping(
        depth_profile.get("form_decision"),
        name="Gaussian c_depth calibration decision",
    )
    _expect(decision.get("scene_count"), 20, name="Gaussian c_depth fit scene count")
    _expect(
        _canonical_sha256(decision),
        calibration_decision_sha,
        name="Gaussian calibration decision SHA-256",
    )
    _expect(decision.get("selected_form"), profile_form, name="Gaussian/profile selected form")
    selected_fit = _mapping(decision.get("selected"), name="Gaussian selected c_depth fit")
    _expect(_finite(selected_fit.get("a"), name="Gaussian selected a"), profile_a, name="profile a")
    _expect(_finite(selected_fit.get("b"), name="Gaussian selected b"), profile_b, name="profile b")

    gs_gate = _mapping(
        gaussian_v2_audit.get("primary_paired_practical_equivalence"),
        name="Gaussian practical-equivalence gate",
    )
    _expect(gs_gate.get("available"), True, name="Gaussian gate availability")
    _expect(gs_gate.get("margin"), V2_EQUIVALENCE_MARGIN, name="Gaussian gate margin")
    _expect(gs_gate.get("support"), V2_GAUSSIAN_SUPPORT, name="Gaussian gate support")
    _expect(
        gs_gate.get("analysis_unit"),
        "Waymo scene; bootstrap resamples scenes only",
        name="Gaussian gate analysis unit",
    )
    _expect(
        gs_gate.get("margin_frozen_before_v2_results"),
        True,
        name="Gaussian margin freeze",
    )
    _expect(gs_gate.get("scene_count"), 30, name="Gaussian gate scene count")
    _expect(gs_gate.get("scene_order"), V2_REQUIRED_SCENES, name="Gaussian gate scene order")
    gate_bootstrap = _mapping(gs_gate.get("bootstrap"), name="Gaussian gate bootstrap")
    _expect(gate_bootstrap.get("unit"), "scene", name="Gaussian gate bootstrap unit")
    _expect(
        gate_bootstrap.get("samples"),
        V2_GAUSSIAN_BOOTSTRAP_SAMPLES,
        name="Gaussian gate bootstrap samples",
    )
    _expect(
        gate_bootstrap.get("seed"),
        V2_GAUSSIAN_BOOTSTRAP_SEED,
        name="Gaussian gate bootstrap seed",
    )
    _expect_close(
        gate_bootstrap.get("confidence_level"),
        0.95,
        name="Gaussian gate confidence level",
    )
    per_scene = gs_gate.get("per_scene")
    if not isinstance(per_scene, list) or len(per_scene) != 30:
        raise ValueError("Gaussian gate must retain exactly 30 per-scene observations")
    scene_logs: list[float] = []
    gaussian_scene_rows: list[dict[str, Any]] = []
    for expected_scene, raw_row in zip(V2_REQUIRED_SCENES, per_scene):
        row = _mapping(raw_row, name=f"Gaussian gate scene {expected_scene}")
        _expect(row.get("scene"), expected_scene, name="Gaussian gate scene order")
        _expect(row.get("trunk_count"), 3, name="Gaussian gate per-scene trunk count")
        log_ratio = _finite(
            row.get("paired_log_gs_over_depth"), name="Gaussian per-scene paired log ratio"
        )
        ratio = _finite(
            row.get("paired_gs_over_depth_ratio"), name="Gaussian per-scene paired ratio"
        )
        _expect_close(ratio, math.exp(log_ratio), name="Gaussian per-scene exp(log ratio)")
        scene_logs.append(log_ratio)
        gaussian_scene_rows.append(
            {
                "scene": expected_scene,
                "trunk_count": 3,
                "paired_log_gs_over_depth": log_ratio,
                "paired_gs_over_depth_ratio": ratio,
            }
        )
    point_ratio = math.exp(float(np.median(np.asarray(scene_logs, dtype=np.float64))))
    _expect_close(gs_gate.get("point_estimate"), point_ratio, name="Gaussian point estimate")
    expected_gs_ci = _gaussian_scene_bootstrap_ratio_ci(
        scene_logs,
        samples=V2_GAUSSIAN_BOOTSTRAP_SAMPLES,
        seed=V2_GAUSSIAN_BOOTSTRAP_SEED,
    )
    gs_ci_raw = gs_gate.get("scene_bootstrap_95_ci")
    if not isinstance(gs_ci_raw, list) or len(gs_ci_raw) != 2:
        raise ValueError("Gaussian gate scene-bootstrap CI must contain two values")
    for index, expected in enumerate(expected_gs_ci):
        _expect_close(gs_ci_raw[index], expected, name=f"Gaussian bootstrap CI {index}")
    gs_ci = expected_gs_ci
    gaussian_pass = (
        V2_EQUIVALENCE_MARGIN[0] <= point_ratio <= V2_EQUIVALENCE_MARGIN[1]
        and V2_EQUIVALENCE_MARGIN[0] <= gs_ci[0] <= gs_ci[1] <= V2_EQUIVALENCE_MARGIN[1]
    )
    _expect(
        gs_gate.get("point_within_margin"),
        V2_EQUIVALENCE_MARGIN[0] <= point_ratio <= V2_EQUIVALENCE_MARGIN[1],
        name="Gaussian point gate",
    )
    _expect(
        gs_gate.get("ci_entirely_within_margin"),
        V2_EQUIVALENCE_MARGIN[0] <= gs_ci[0] <= gs_ci[1] <= V2_EQUIVALENCE_MARGIN[1],
        name="Gaussian CI gate",
    )
    _expect(gs_gate.get("passed"), gaussian_pass, name="Gaussian practical-equivalence gate")
    if not gaussian_pass:
        raise ValueError("Gaussian gate values do not satisfy practical equivalence")

    c_gs_recommendation = _mapping(
        gaussian_v2_audit.get("c_gs_recommendation"),
        name="Gaussian c_gs recommendation",
    )
    _expect(c_gs_recommendation.get("form"), "identity", name="Gaussian c_gs form")
    _expect_close(c_gs_recommendation.get("value"), 1.0, name="Gaussian c_gs value")
    _expect(
        c_gs_recommendation.get("paired_gate_passed"),
        True,
        name="Gaussian c_gs paired gate",
    )
    _expect(
        c_gs_recommendation.get("applicable_to_accepted_tokenizer"),
        True,
        name="Gaussian c_gs applicability",
    )

    reference_script_sha = _declared_script_sha256(reference, name="v2 reference")
    reference_signature_sha = _validate_reference_signature(reference)
    reference_metadata = _mapping(reference.get("metadata"), name="v2 reference metadata")
    _expect(
        reference_metadata.get("checkpoint_sha256"),
        dggt_sha256,
        name="v2 reference DGGT SHA-256",
    )
    _expect(
        reference_metadata.get("tokenizer_checkpoint_sha256"),
        tokenizer_sha256,
        name="v2 reference tokenizer SHA-256",
    )
    _expect(
        reference_metadata.get("requested_scenes"),
        V2_REQUIRED_SCENES,
        name="v2 reference scenes",
    )
    _expect(
        reference_metadata.get("requested_trunks"),
        V2_REQUIRED_TRUNKS,
        name="v2 reference trunks",
    )
    _expect(
        reference_metadata.get("data_root"),
        source_data_root,
        name="reference/LiDAR data_root",
    )
    _expect(
        reference_metadata.get("roundtrip_window_starts"),
        V2_WINDOW_STARTS,
        name="v2 reference window starts",
    )
    _expect(
        reference_metadata.get("roundtrip_window_length"),
        10,
        name="v2 reference window length",
    )
    _reference_grid, reference_case_count = _validate_case_coverage(
        reference,
        scenes=V2_REQUIRED_SCENES,
        trunks=V2_REQUIRED_TRUNKS,
        name="v2 reference",
        require_patch_grid=False,
    )
    _expect(reference_case_count, 90, name="v2 reference complete case count")
    _validate_reference_windows(reference)

    source_boundaries = _mapping(diagnostic.get("boundaries"), name="v2 source boundaries")
    _expect(
        source_boundaries.get("render"),
        {
            "depth": {"form": "identity"},
            "gaussian_scale": {"form": "identity", "c_gs": 1.0},
        },
        name="v2 source render boundary",
    )
    source_metric = _mapping(source_boundaries.get("metric"), name="v2 source metric boundary")
    _expect(
        source_metric.get("gaussian_scale"),
        {"form": "identity", "c_gs": 1.0},
        name="v2 source metric c_gs",
    )
    summary = _mapping(diagnostic.get("summary"), name="v2 LiDAR summary")
    _expect(summary.get("case_count"), 30, name="v2 LiDAR summary case count")
    primary_evidence = _validated_lidar_summary(
        summary.get("primary_phase1a_valid_all_lidar"),
        name="v2 primary LiDAR gate",
        expected_support="all_lidar",
        expected_gauge_filter=True,
        candidate_form=str(profile_form),
        require_all_trunks=False,
    )
    all_trunk_evidence = _validated_lidar_summary(
        summary.get("sensitivity_all_30_trunks"),
        name="v2 all-trunk LiDAR sensitivity",
        expected_support="all_lidar",
        expected_gauge_filter=False,
        candidate_form=str(profile_form),
        require_all_trunks=True,
    )
    static_evidence = _validated_lidar_summary(
        summary.get("sensitivity_phase1a_valid_static_nonsky"),
        name="v2 static/non-sky LiDAR sensitivity",
        expected_support="static_nonsky",
        expected_gauge_filter=True,
        candidate_form=str(profile_form),
        require_all_trunks=False,
    )
    selected_form = primary_evidence["selected_form"]
    if selected_form not in {"identity", profile_form}:
        raise ValueError("LiDAR gate selected an unfrozen depth profile")
    _expect(
        source_metric.get("depth", {}).get("form"),
        selected_form,
        name="LiDAR source metric form",
    )
    source_metric_depth = _mapping(source_metric.get("depth"), name="LiDAR source metric depth")
    if selected_form != "identity":
        _expect_close(source_metric_depth.get("a"), profile_a, name="LiDAR source metric a")
        _expect_close(source_metric_depth.get("b"), profile_b, name="LiDAR source metric b")
        _expect(
            source_metric_depth.get("evaluate_on"),
            profile.get("evaluate_on"),
            name="LiDAR source metric variable",
        )
        _expect(
            source_metric_depth.get("fit_variable"),
            profile.get("fit_variable"),
            name="LiDAR source fit variable",
        )
        _expect(
            source_metric_depth.get("runtime_variable"),
            profile.get("runtime_variable"),
            name="LiDAR source runtime variable",
        )
    candidate_selected = bool(primary_evidence["gate_pass"])
    bootstrap_samples = V2_LIDAR_BOOTSTRAP_SAMPLES
    bootstrap_seed = V2_LIDAR_BOOTSTRAP_SEED
    resume_signature_without_depth_chunk = {
        "schema": [V2_LIDAR_SCHEMA_NAME, V2_LIDAR_SCHEMA_VERSION],
        "hashes": {
            "script_sha256": lidar_script_sha,
            "checkpoint_sha256": dggt_sha256,
            "tokenizer_checkpoint_sha256": tokenizer_sha256,
            "reference_json_sha256": reference_sha256,
            "d4_json_sha256": gaussian_sha256,
        },
        "scenes": FROZEN_SELECTION_SCENES,
        "trunks": V2_REQUIRED_TRUNKS,
        "profile": dict(profile),
        "data_root": source_data_root,
        "expected_window_length": 10,
        "precision": source_metadata.get("precision"),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    depth_chunk = _resolve_v2_lidar_depth_chunk(
        source_metadata,
        signature_without_depth_chunk=resume_signature_without_depth_chunk,
        resume_signature_sha256=source_resume_signature_sha,
    )
    expected_resume_signature = dict(resume_signature_without_depth_chunk)
    expected_resume_signature["depth_chunk"] = depth_chunk
    _expect(
        _canonical_sha256(expected_resume_signature),
        source_resume_signature_sha,
        name="v2 LiDAR resume signature SHA-256",
    )

    metric_depth: dict[str, Any] = {"form": selected_form}
    if selected_form != "identity":
        metric_depth.update(
            {
                "a": profile_a,
                "b": profile_b,
                "reference_depth_m": 20.0,
                "runtime_depth_clamp_m": [0.5, 80.0],
                "evaluate_on": PULLBACK_DEPTH_EVALUATION,
            }
        )
    phase1b_scheme = "A" if selected_form == "identity" else "B"
    return {
        "schema": {
            "name": PULLBACK_SCHEMA_NAME,
            "version": PULLBACK_SCHEMA_VERSION,
            "strict": True,
        },
        "artifact_role": PULLBACK_ARTIFACT_ROLE,
        "eligible_for_training": True,
        "tokenizer_generation": TOKENIZER_V2_GENERATION,
        "tokenizer": {"sha256": tokenizer_sha256, "sha8": tokenizer_sha256[:8]},
        "dggt": {"sha256": dggt_sha256},
        "runtime_contract": {
            "version": PULLBACK_RUNTIME_CONTRACT_VERSION,
            "window_len": 10,
            "patch_grid_hw": gaussian_grid,
            "gauge_representation": SCENE_GAUGE_REPRESENTATION,
            "log_metric_scale_units": PULLBACK_LOG_METRIC_SCALE_UNITS,
        },
        "boundaries": {
            "render": {
                "depth": {"form": "identity"},
                "gaussian_scale": {"form": "identity", "c_gs": 1.0},
            },
            "metric": {
                "depth": metric_depth,
                "gaussian_scale": {
                    "rule": PULLBACK_GS_SCALE_RULE,
                    "channels_half_open": [4, 7],
                    "c_gs": 1.0,
                },
            },
        },
        "evidence": {
            "source_metric_gate": {
                "path": _display_path(diagnostic_path),
                "sha256": diagnostic_sha256,
                "script_sha256": lidar_script_sha,
                "result_sha256_excluding_self": source_result_sha,
            },
            "gaussian_roundtrip_audit": {
                "path": _display_path(gaussian_path),
                "sha256": gaussian_sha256,
                "script_sha256": gaussian_script_sha,
            },
            "metric_reference": {
                "path": _display_path(reference_path),
                "sha256": reference_sha256,
                "script_sha256": reference_script_sha,
                "resume_signature_sha256": reference_signature_sha,
            },
            "reconstruction_render_smoke": smoke_evidence,
            "fit_scenes": FROZEN_FIT_SCENES,
            "selection_scenes": FROZEN_SELECTION_SCENES,
            "frozen_profile": {"sha256": profile_sha, "payload": dict(profile)},
            "formal_coverage": {
                "scene_count": 30,
                "trunk_count": 90,
                "calibration_scene_count": 20,
                "selection_scene_count": 10,
                "required_scenes": V2_REQUIRED_SCENES,
                "required_trunks": V2_REQUIRED_TRUNKS,
                "window_starts": V2_WINDOW_STARTS,
                "window_length": 10,
                "data_root": source_data_root,
            },
            "gaussian_practical_equivalence_gate": {
                "passed": True,
                "margin": V2_EQUIVALENCE_MARGIN,
                "point_estimate": point_ratio,
                "scene_bootstrap_95_ci": gs_ci,
                "scene_count": 30,
                "c_gs_recommendation": {"form": "identity", "value": 1.0},
                "support": V2_GAUSSIAN_SUPPORT,
                "analysis_unit": "Waymo scene; bootstrap resamples scenes only",
                "bootstrap": {
                    "unit": "scene",
                    "samples": V2_GAUSSIAN_BOOTSTRAP_SAMPLES,
                    "seed": V2_GAUSSIAN_BOOTSTRAP_SEED,
                    "confidence_level": 0.95,
                },
                "per_scene": gaussian_scene_rows,
            },
            "primary_lidar_gate": {
                "candidate_form": profile_form,
                "selected_form": selected_form,
                "case_count": primary_evidence["case_count"],
                "scene_count": 10,
                "identity_absrel": primary_evidence["identity_absrel"],
                "candidate_absrel": primary_evidence["candidate_absrel"],
                "scene_delta_mean": primary_evidence["scene_delta_mean"],
                "scene_delta_bootstrap_95_ci": primary_evidence[
                    "scene_delta_bootstrap_95_ci"
                ],
                "improved_scene_count": primary_evidence["improved_scene_count"],
                "gate_pass": candidate_selected,
                "support": primary_evidence["support"],
                "gauge_valid_cases_only": True,
                "bootstrap": primary_evidence["bootstrap"],
                "scene_rows": primary_evidence["scene_rows"],
                "exact_sign_flip": primary_evidence["exact_sign_flip"],
                "sensitivities": {
                    "all_30_trunks": {
                        key: all_trunk_evidence[key]
                        for key in (
                            "support",
                            "gauge_valid_cases_only",
                            "candidate_form",
                            "selected_form",
                            "case_count",
                            "scene_count",
                            "scene_delta_mean",
                            "scene_delta_bootstrap_95_ci",
                            "improved_scene_count",
                            "gate_pass",
                            "exact_sign_flip",
                        )
                    },
                    "phase1a_valid_static_nonsky": {
                        key: static_evidence[key]
                        for key in (
                            "support",
                            "gauge_valid_cases_only",
                            "candidate_form",
                            "selected_form",
                            "case_count",
                            "scene_count",
                            "scene_delta_mean",
                            "scene_delta_bootstrap_95_ci",
                            "improved_scene_count",
                            "gate_pass",
                            "exact_sign_flip",
                        )
                    },
                },
            },
        },
        "limitations": {
            "phase1b_scheme": phase1b_scheme,
            "similarity_consistent": True,
            "c_gs": "identity",
            "render_scope": "identity_only",
            "metric_scope": (
                "identity" if selected_form == "identity" else "depth_profile_only"
            ),
            "residual_limits": [
                (
                    "Practical equivalence is established only for the preregistered "
                    "paired GS/depth estimator and support."
                ),
                (
                    "LiDAR selection scenes 320-329 are validation/selection evidence, "
                    "not an untouched test split."
                ),
                (
                    "The pullback does not establish absolute Gaussian covariance "
                    "calibration beyond the recorded gates."
                ),
            ],
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-json", type=Path, default=DEFAULT_DIAGNOSTIC_JSON)
    parser.add_argument(
        "--gaussian-audit-json",
        type=Path,
        default=DEFAULT_GAUSSIAN_AUDIT_JSON,
        help="Formal v2 30-scene/90-trunk Gaussian audit JSON.",
    )
    parser.add_argument(
        "--reconstruction-smoke-json",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_SMOKE_JSON,
        help="Accepted v2 CUDA reconstruction/render smoke JSON.",
    )
    parser.add_argument(
        "--reconstruction-smoke-selection-manifest",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_SMOKE_SELECTION_MANIFEST,
        help="Fixed-selection manifest bound by the v2 smoke JSON.",
    )
    parser.add_argument(
        "--reconstruction-smoke-visual",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_SMOKE_VISUAL,
        help="Decodable visual emitted by the accepted v2 smoke.",
    )
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument(
        "--tokenizer-checkpoint", type=Path, default=DEFAULT_TOKENIZER_CHECKPOINT
    )
    parser.add_argument("--dggt-checkpoint", type=Path, default=DEFAULT_DGGT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--authorize-v2-production",
        action="store_true",
        help=(
            "Required authorization to promote complete v2 Gaussian/LiDAR evidence "
            "into an eligible production artifact."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.authorize_v2_production:
        raise ValueError(
            "freezing tokenizer v2 requires explicit --authorize-v2-production"
        )
    diagnostic_path = args.diagnostic_json.expanduser().resolve()
    gaussian_path = args.gaussian_audit_json.expanduser().resolve()
    reference_path = args.reference_json.expanduser().resolve()
    tokenizer_path = args.tokenizer_checkpoint.expanduser().resolve()
    dggt_path = args.dggt_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    smoke_path = args.reconstruction_smoke_json.expanduser().resolve()
    smoke_selection_path = (
        args.reconstruction_smoke_selection_manifest.expanduser().resolve()
    )
    smoke_visual_path = args.reconstruction_smoke_visual.expanduser().resolve()
    inputs = (
        diagnostic_path,
        gaussian_path,
        reference_path,
        tokenizer_path,
        dggt_path,
        smoke_path,
        smoke_selection_path,
        smoke_visual_path,
    )
    for required in inputs:
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_path in inputs:
        raise ValueError("--output must not overwrite an evidence or checkpoint input")

    tokenizer_sha256 = _sha256(tokenizer_path)
    dggt_sha256 = _sha256(dggt_path)
    diagnostic_sha256 = _sha256(diagnostic_path)
    gaussian_sha256 = _sha256(gaussian_path)
    reference_sha256 = _sha256(reference_path)
    smoke_sha256 = _sha256(smoke_path)
    smoke_selection_sha256 = _sha256(smoke_selection_path)
    smoke_visual_sha256 = _sha256(smoke_visual_path)
    expected_name = f"pullback_{tokenizer_sha256[:8]}.json"
    _expect(output_path.name, expected_name, name="production artifact filename")

    diagnostic = _mapping(
        json.loads(diagnostic_path.read_text(encoding="utf-8")), name="diagnostic"
    )
    gaussian = _mapping(
        json.loads(gaussian_path.read_text(encoding="utf-8")), name="Gaussian audit"
    )
    reference = _mapping(
        json.loads(reference_path.read_text(encoding="utf-8")), name="reference"
    )
    reconstruction_smoke = _mapping(
        json.loads(smoke_path.read_text(encoding="utf-8")), name="smoke"
    )
    smoke_selection_manifest = _mapping(
        json.loads(smoke_selection_path.read_text(encoding="utf-8")),
        name="smoke selection manifest",
    )
    artifact = build_v2_production_artifact(
        diagnostic,
        gaussian,
        reference,
        reconstruction_smoke,
        smoke_selection_manifest,
        diagnostic_path=diagnostic_path,
        diagnostic_sha256=diagnostic_sha256,
        gaussian_path=gaussian_path,
        gaussian_sha256=gaussian_sha256,
        reference_path=reference_path,
        reference_sha256=reference_sha256,
        reconstruction_smoke_path=smoke_path,
        reconstruction_smoke_sha256=smoke_sha256,
        smoke_selection_manifest_path=smoke_selection_path,
        smoke_selection_manifest_sha256=smoke_selection_sha256,
        reconstruction_smoke_visual_path=smoke_visual_path,
        reconstruction_smoke_visual_sha256=smoke_visual_sha256,
        tokenizer_sha256=tokenizer_sha256,
        dggt_sha256=dggt_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pullback-validation-", dir=output_path.parent
    ) as validation_dir:
        validation_path = Path(validation_dir) / output_path.name
        _atomic_write_json(validation_path, artifact)
        loaded = load_pullback_calibration(
            validation_path,
            expected_window_len=10,
            expected_patch_grid=artifact["runtime_contract"]["patch_grid_hw"],
        )
    _atomic_write_json(output_path, artifact)
    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "eligible_for_training": True,
                "metric_depth_form": loaded.depth_form,
                "render_form": "identity",
                "c_gs": loaded.c_gs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
