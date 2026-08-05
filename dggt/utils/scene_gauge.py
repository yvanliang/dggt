"""Scene-gauge geometry and checkpoint-bound tokenizer pullback helpers.

The scene gauge lives in the frozen DGGT teacher coordinate system:

``[log(metres / DGGT unit), log(tan(FOVx / 2)), log(tan(FOVy / 2))]``.

Tokenizer pullback is deliberately scope-aware.  Rendering keeps the native
tokenizer reconstruction unchanged.  At a metric boundary, the calibrated
depth factor is evaluated on *uncorrected metric depth* and applied to both
depth and Gaussian scale channels.  This is the only production implementation
of that correction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from dggt.utils.rotation import mat_to_quat


SCENE_GAUGE_DIM = 3
SCENE_GAUGE_REPRESENTATION = "dggt_teacher_log_metric_scale_logfov_v1"
SCENE_GAUGE_STATS_VERSION = "scene_gauge_per_channel_v1"
SCENE_GAUGE_STATS_STD_FLOOR = 1.0e-6
GAUGE_MROPE_TEMPORAL_OFFSET = 15100

METRIC_BOX_MAPPING_MODE = "metric_gauge_v4"
GENERIC_BOX_MAPPING_MODE = "generic_sim3"

SCENE_GAUGE_TABLE_SCHEMA = "dggt_scene_gauge_table_v1"
SCENE_GAUGE_TABLE_SCHEMA_VERSION = "1.0.0"
SCENE_GAUGE_EXTRACTOR_IMPLEMENTATION_VERSION = "lean_full29_teacher_v2"

PULLBACK_SCHEMA_NAME = "dggt_tokenizer_pullback"
PULLBACK_SCHEMA_VERSION = "2.0.0"
PULLBACK_ARTIFACT_ROLE = "production_pullback"
PULLBACK_RUNTIME_CONTRACT_VERSION = (
    "metric_depth_profile_gs_same_factor_render_identity_v2"
)
PULLBACK_LOG_METRIC_SCALE_UNITS = "log_metres_per_dggt_unit"
PULLBACK_DEPTH_EVALUATION = (
    "depth_recon_times_exp_log_metric_scale_before_correction"
)
PULLBACK_GS_SCALE_RULE = "multiply_channels_4_7_by_same_depth_factor"
PULLBACK_RENDER_BOUNDARY = "render"
PULLBACK_METRIC_BOUNDARY = "metric"
PULLBACK_BOUNDARIES = (PULLBACK_RENDER_BOUNDARY, PULLBACK_METRIC_BOUNDARY)
PULLBACK_DEPTH_FORMS = ("identity", "constant", "loglinear")
PULLBACK_DEPTH_VARIABLE_CONTRACT = {
    "name": "uncorrected_reconstructed_metric_depth_m",
    "source_tensor": "reconstructed_depth",
    "metric_conversion": "divide_by_full_29f_direct_units_per_metre",
    "correction_state": "uncorrected",
    "runtime_clamp_m": [0.5, 80.0],
    "reference_depth_m": 20.0,
}
PULLBACK_V2_SMOKE_THRESHOLDS = {
    "render_direct_psnr_min_db": 30.0,
    "render_direct_ssim_min": 0.95,
    "render_direct_lpips_max": 0.05,
    "render_gt_psnr_drop_max_db": 0.25,
    "render_gt_ssim_drop_max": 0.02,
    "render_gt_lpips_increase_max": 0.02,
    "depth_recon_over_direct_range": [0.95, 1.05],
    "depth_recon_vs_direct_absrel_max": 0.05,
    "point_xyz_relative_error_max": 0.05,
    "gs_recon_over_direct_range": [0.95, 1.05],
    "paired_gs_over_depth_range": [0.95, 1.05],
    "gs_axis_anisotropy_log_rms_max": 0.1,
    "support_pixels_median_per_frame_min": 1000,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def resolve_scene_gauge_checkpoint_sha256(
    checkpoint_path: str | Path,
    explicit_expected_sha256: str | None = None,
) -> str:
    """Bind a scene-gauge consumer to the checkpoint file it actually loads.

    Formal edit entrypoints must never trust an optional caller-supplied digest
    as the sole binding.  Hash the runtime checkpoint, then treat an explicit
    digest only as an additional assertion against that measured value.
    """

    digest = hashlib.sha256()
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    with resolved_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if explicit_expected_sha256 is not None:
        explicit = str(explicit_expected_sha256).lower()
        if _SHA256_RE.fullmatch(explicit) is None:
            raise ValueError(
                "explicit scene-gauge DGGT SHA-256 must be 64 hexadecimal characters"
            )
        if explicit != actual:
            raise ValueError(
                "Explicit scene-gauge DGGT checkpoint SHA-256 does not match the "
                f"checkpoint file actually loaded: explicit={explicit}, actual={actual}, "
                f"path={resolved_path}"
            )
    return actual


def scene_gauge_production_protocol(checkpoint_sha256: str) -> dict[str, Any]:
    """Return the frozen Phase-1a extraction contract for production tables."""

    checkpoint_hash = str(checkpoint_sha256).lower()
    if _SHA256_RE.fullmatch(checkpoint_hash) is None:
        raise ValueError("checkpoint_sha256 must be a lowercase 64-character SHA-256")
    return {
        "representation": SCENE_GAUGE_REPRESENTATION,
        "trunk_frames": 29,
        "checkpoint_sha256": checkpoint_hash,
        "extractor_implementation_version": SCENE_GAUGE_EXTRACTOR_IMPLEMENTATION_VERSION,
        "production_contract": True,
        "precision": "aggregator bf16 autocast on CUDA; camera and depth heads fp32",
        "module_precision": {
            "aggregator": "bf16 autocast on CUDA (fp32 on CPU)",
            "camera_head": "fp32 with autocast disabled (D1/D3 route)",
            "depth_head": "fp32 with autocast disabled (D2 route)",
        },
        "scale_definition": "s_lidar = DGGT teacher depth units / LiDAR camera-z metres",
        "lidar_preprocessing": "validated D2 load_and_preprocess_flow protocol",
        "min_lidar_depth_m": 1.0,
        "max_lidar_depth_m": 80.0,
        "min_depth_pixels_per_frame": 64,
        "min_depth_pixels_per_trunk": 5_000,
        "min_depth_frames": 15,
        "max_frame_cv": 0.03,
        "min_camera_span_m": 2.0,
        "max_ruler_ratio_deviation": 0.10,
        "actor_enabled": True,
        "actor_classes": ["vehicle"],
        "actor_min_pixels": 32,
        "actor_min_frames": 3,
        "actor_edge_margin": 2,
        "actor_min_consensus_fraction": 0.6,
        "actor_max_log_interval_width": 0.25,
    }


def load_scene_gauge_lookup(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_split: str | None = None,
    expected_image_dir: str | Path | None = None,
) -> tuple[Path, str, str, dict[str, tuple[tuple[float, float, float], tuple[bool, bool, bool]]]]:
    """Load a complete production 29-frame gauge table for metric edit paths.

    This loader deliberately accepts only the wrapped production schema.  In
    particular, partial shards and the early bare-entry development format are
    rejected: using either for metric 3D editing could silently select a
    scalar/field-of-view from the wrong teacher or trunk.
    """

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Scene gauge table not found: {resolved_path}")
    payload = resolved_path.read_bytes()

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        table = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid scene gauge JSON at {resolved_path}: {error}") from error
    if not isinstance(table, dict):
        raise ValueError(f"Scene gauge table must be a JSON object: {resolved_path}")
    required = {"schema", "status", "metadata", "summary", "entries", "errors"}
    if set(table) != required:
        raise ValueError(
            "Scene gauge production wrapper must contain exactly "
            f"{sorted(required)}, got {sorted(table)}: {resolved_path}"
        )
    if table["schema"] != SCENE_GAUGE_TABLE_SCHEMA or table["status"] != "complete":
        raise ValueError(
            "Metric editing requires a complete dggt_scene_gauge_table_v1 artifact; "
            f"got schema={table['schema']!r}, status={table['status']!r}: {resolved_path}"
        )
    if not isinstance(table["metadata"], dict):
        raise ValueError(f"Scene gauge metadata must be an object: {resolved_path}")
    metadata = table["metadata"]
    expected_metadata = {
        "schema_version": SCENE_GAUGE_TABLE_SCHEMA_VERSION,
        "representation": SCENE_GAUGE_REPRESENTATION,
        "trunk_frames": 29,
    }
    for field_name, expected in expected_metadata.items():
        if metadata.get(field_name) != expected:
            raise ValueError(
                f"Scene gauge metadata {field_name!r} must be {expected!r}, "
                f"got {metadata.get(field_name)!r}: {resolved_path}"
            )
    split = metadata.get("split")
    image_dir = metadata.get("image_dir")
    if not isinstance(split, str) or not split.strip():
        raise ValueError(f"Scene gauge metadata split must be a non-empty string: {resolved_path}")
    if not isinstance(image_dir, str) or not image_dir.strip():
        raise ValueError(f"Scene gauge metadata image_dir must be a non-empty string: {resolved_path}")
    if expected_split is not None and split != str(expected_split):
        raise ValueError(
            f"Scene gauge split mismatch: table={split!r}, expected={str(expected_split)!r}"
        )
    # Retain ``expected_image_dir`` for API compatibility, but deliberately do
    # not compare it.  ``metadata.image_dir`` records where the table was built;
    # it is not a portable identity and normally differs after moving datasets
    # between machines.  The split, content hashes, protocol, and key coverage
    # below are the compatibility contract.
    checkpoint_sha256 = str(metadata.get("checkpoint_sha256", "")).lower()
    if _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ValueError(
            f"Scene gauge metadata checkpoint_sha256 must be 64 lowercase hex characters: {resolved_path}"
        )
    if expected_checkpoint_sha256 is not None:
        expected_hash = str(expected_checkpoint_sha256).lower()
        if _SHA256_RE.fullmatch(expected_hash) is None:
            raise ValueError("expected_checkpoint_sha256 must be 64 lowercase hex characters")
        if checkpoint_sha256 != expected_hash:
            raise ValueError(
                "Scene gauge teacher checkpoint SHA-256 mismatch: "
                f"table={checkpoint_sha256}, expected={expected_hash}, path={resolved_path}"
            )
    protocol = metadata.get("protocol")
    expected_protocol = scene_gauge_production_protocol(checkpoint_sha256)
    if protocol != expected_protocol:
        raise ValueError(
            f"Scene gauge table does not use the frozen production protocol: {resolved_path}"
        )
    protocol_sha256 = hashlib.sha256(
        json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if metadata.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Scene gauge protocol_sha256 mismatch: {resolved_path}")
    requested_keys = metadata.get("requested_keys")
    if (
        not isinstance(requested_keys, list)
        or not requested_keys
        or any(not isinstance(key, str) for key in requested_keys)
        or len(requested_keys) != len(set(requested_keys))
    ):
        raise ValueError(
            f"Scene gauge requested_keys must be a non-empty unique string list: {resolved_path}"
        )
    expected_entry_count = metadata.get("expected_entry_count")
    if (
        isinstance(expected_entry_count, bool)
        or not isinstance(expected_entry_count, int)
        or expected_entry_count != len(requested_keys)
    ):
        raise ValueError(f"Scene gauge expected_entry_count mismatch: {resolved_path}")
    if not isinstance(table["summary"], dict):
        raise ValueError(f"Scene gauge summary must be an object: {resolved_path}")
    if not isinstance(table["errors"], list) or table["errors"]:
        raise ValueError(
            f"Scene gauge production table must have an empty errors list: {resolved_path}"
        )
    if not isinstance(table["entries"], dict) or not table["entries"]:
        raise ValueError(f"Scene gauge production entries must be a non-empty object: {resolved_path}")
    if set(table["entries"]) != set(requested_keys):
        raise ValueError(
            f"Scene gauge entries do not exactly match requested_keys: {resolved_path}"
        )
    summary = table["summary"]
    if (
        int(summary.get("entry_count", -1)) != expected_entry_count
        or int(summary.get("expected_entry_count", -1)) != expected_entry_count
        or float(summary.get("coverage_fraction", -1.0)) != 1.0
    ):
        raise ValueError(
            f"Scene gauge summary does not prove complete coverage: {resolved_path}"
        )

    entries: dict[
        str, tuple[tuple[float, float, float], tuple[bool, bool, bool]]
    ] = {}
    for raw_key, entry in table["entries"].items():
        if not isinstance(raw_key, str) or raw_key.count("/") != 1:
            raise ValueError(f"Invalid scene gauge key {raw_key!r} in {resolved_path}")
        scene_name, trunk_text = raw_key.split("/", 1)
        if not scene_name or not trunk_text.isdigit():
            raise ValueError(f"Invalid scene gauge key {raw_key!r} in {resolved_path}")
        canonical_scene = scene_name.zfill(3) if scene_name.isdigit() else scene_name
        canonical_key = f"{canonical_scene}/{int(trunk_text)}"
        if canonical_key in entries:
            raise ValueError(f"Duplicate canonical gauge key {canonical_key!r}: {resolved_path}")
        if not isinstance(entry, dict):
            raise ValueError(f"Scene gauge entry {raw_key!r} must be an object")
        fov = entry.get("log_tan_half_fov")
        valid = entry.get("valid")
        if not isinstance(fov, list) or len(fov) != 2:
            raise ValueError(f"Scene gauge entry {raw_key!r} needs two log-FOV channels")
        if (
            not isinstance(valid, list)
            or len(valid) != SCENE_GAUGE_DIM
            or any(type(flag) is not bool for flag in valid)
        ):
            raise ValueError(f"Scene gauge entry {raw_key!r} needs three boolean validity flags")
        raw_values = (entry.get("log_metric_scale"), fov[0], fov[1])
        values = []
        for channel, (raw_value, channel_valid) in enumerate(zip(raw_values, valid)):
            if raw_value is None:
                if channel_valid:
                    raise ValueError(
                        f"Scene gauge entry {raw_key!r} channel {channel} is null but marked valid"
                    )
                values.append(0.0)
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"Scene gauge entry {raw_key!r} channel {channel} is not numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"Scene gauge entry {raw_key!r} channel {channel} is non-finite")
            values.append(value)
        entries[canonical_key] = (
            (values[0], values[1], values[2]),
            (valid[0], valid[1], valid[2]),
        )
    return (
        resolved_path,
        hashlib.sha256(payload).hexdigest(),
        checkpoint_sha256,
        entries,
    )


def scene_gauge_valid_channel_mean(
    entries: Mapping[str, tuple[Sequence[float], Sequence[bool]]],
) -> tuple[float, float, float]:
    """Compute the production-table mean over valid values per channel.

    Formal physical consumers use this deterministic value when a trunk has
    an invalid channel.  The raw estimate and validity mask remain available
    for diagnostics/supervision; no short-window Sim3 estimate is introduced.
    """

    sums = [0.0] * SCENE_GAUGE_DIM
    counts = [0] * SCENE_GAUGE_DIM
    for key, (raw_gauge, raw_valid) in entries.items():
        if len(raw_gauge) != SCENE_GAUGE_DIM or len(raw_valid) != SCENE_GAUGE_DIM:
            raise ValueError(f"Scene gauge entry {key!r} must contain three channels")
        for channel, (raw_value, channel_valid) in enumerate(
            zip(raw_gauge, raw_valid)
        ):
            value = float(raw_value)
            if bool(channel_valid):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Scene gauge entry {key!r} valid channel {channel} is non-finite"
                    )
                sums[channel] += value
                counts[channel] += 1
    if any(count <= 0 for count in counts):
        raise ValueError(
            "Scene gauge table needs at least one valid value per channel; "
            f"got counts={counts}"
        )
    return tuple(sums[i] / counts[i] for i in range(SCENE_GAUGE_DIM))


def effective_scene_gauge(
    raw_gauge: Sequence[float],
    raw_valid: Sequence[bool],
    valid_channel_mean: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[bool, bool, bool]]:
    """Replace invalid raw channels with the production-table valid mean."""

    if not (
        len(raw_gauge)
        == len(raw_valid)
        == len(valid_channel_mean)
        == SCENE_GAUGE_DIM
    ):
        raise ValueError("raw gauge, validity, and fallback mean must each have 3 channels")
    values = []
    fallback_mask = []
    for channel, (raw_value, channel_valid, fallback_value) in enumerate(
        zip(raw_gauge, raw_valid, valid_channel_mean)
    ):
        raw = float(raw_value)
        fallback = float(fallback_value)
        if not math.isfinite(fallback):
            raise ValueError(f"Scene gauge fallback channel {channel} is non-finite")
        use_fallback = not bool(channel_valid)
        value = fallback if use_fallback else raw
        if not math.isfinite(value):
            raise ValueError(f"Effective scene gauge channel {channel} is non-finite")
        values.append(value)
        fallback_mask.append(use_fallback)
    return (
        (values[0], values[1], values[2]),
        (fallback_mask[0], fallback_mask[1], fallback_mask[2]),
    )


@dataclass(frozen=True)
class PullbackCalibration:
    """Validated, immutable runtime view of a production pullback artifact."""

    path: Path
    artifact_sha256: str
    tokenizer_sha256: str
    dggt_sha256: str
    tokenizer_generation: str
    window_len: int
    patch_grid_hw: tuple[int, int]
    depth_a: float
    depth_b: float
    reference_depth_m: float
    runtime_depth_clamp_m: tuple[float, float]
    c_gs: float
    runtime_contract_version: str = PULLBACK_RUNTIME_CONTRACT_VERSION
    depth_form: Literal["identity", "constant", "loglinear"] = "identity"

    def __post_init__(self) -> None:
        if self.tokenizer_generation != "t0_v2":
            raise ValueError("PullbackCalibration only supports tokenizer generation t0_v2")
        if self.runtime_contract_version != PULLBACK_RUNTIME_CONTRACT_VERSION:
            raise ValueError("PullbackCalibration runtime contract is not the current v2 contract")
        for name, value in (
            ("artifact_sha256", self.artifact_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
            ("dggt_sha256", self.dggt_sha256),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"PullbackCalibration {name} must be a lowercase SHA-256")
        if isinstance(self.window_len, bool) or not isinstance(self.window_len, int):
            raise ValueError("PullbackCalibration window_len must be an integer")
        if self.window_len != 10:
            raise ValueError("PullbackCalibration window_len must be 10")
        if (
            not isinstance(self.patch_grid_hw, tuple)
            or len(self.patch_grid_hw) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.patch_grid_hw
            )
        ):
            raise ValueError("PullbackCalibration patch_grid_hw must contain two positive integers")
        if self.depth_form not in PULLBACK_DEPTH_FORMS:
            raise ValueError("PullbackCalibration depth_form is unsupported")
        if (
            not isinstance(self.runtime_depth_clamp_m, tuple)
            or len(self.runtime_depth_clamp_m) != 2
        ):
            raise ValueError("PullbackCalibration runtime_depth_clamp_m must contain two values")
        numeric_values = {
            "depth_a": self.depth_a,
            "depth_b": self.depth_b,
            "reference_depth_m": self.reference_depth_m,
            "c_gs": self.c_gs,
            "runtime_depth_clamp_min_m": self.runtime_depth_clamp_m[0],
            "runtime_depth_clamp_max_m": self.runtime_depth_clamp_m[1],
        }
        if any(not math.isfinite(float(value)) for value in numeric_values.values()):
            raise ValueError("PullbackCalibration numeric fields must be finite")
        if self.reference_depth_m != 20.0 or self.runtime_depth_clamp_m != (0.5, 80.0):
            raise ValueError("PullbackCalibration depth variable contract mismatch")
        if self.c_gs != 1.0:
            raise ValueError("PullbackCalibration v2 requires c_gs=1.0")
        if self.depth_form == "identity" and (self.depth_a != 0.0 or self.depth_b != 0.0):
            raise ValueError("PullbackCalibration identity form requires depth_a=depth_b=0")
        if self.depth_form == "constant" and self.depth_b != 0.0:
            raise ValueError("PullbackCalibration constant form requires depth_b=0")


@dataclass(frozen=True)
class PullbackResult:
    """Geometry after applying one explicit pullback boundary."""

    depth_dggt: torch.Tensor
    gs_map_dggt: torch.Tensor
    c_depth_factor: torch.Tensor
    boundary: Literal["render", "metric"]

    @property
    def depth(self) -> torch.Tensor:
        return self.depth_dggt

    @property
    def gs_map(self) -> torch.Tensor:
        return self.gs_map_dggt


def _sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{name} keys do not match strict schema: missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: artifact={actual!r}, expected={expected!r}")


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scene_bootstrap_median_exp_ci(
    log_values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(log_values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Gaussian scene bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size), endpoint=False)
    medians = np.median(values[indices], axis=1)
    ci_log = np.quantile(medians, (0.025, 0.975))
    return math.exp(float(ci_log[0])), math.exp(float(ci_log[1]))


def _scene_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size < 2 or not np.isfinite(ordered).all():
        raise ValueError("LiDAR scene bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, ordered.size, size=(samples, ordered.size))
    means = ordered[indices].mean(axis=1)
    ci = np.quantile(means, (0.025, 0.975))
    return float(ci[0]), float(ci[1])


def _validate_v2_smoke_evidence(root: Mapping[str, Any]) -> None:
    smoke = _require_mapping(
        root["evidence"]["reconstruction_render_smoke"],
        name="reconstruction_render_smoke",
    )
    _require_exact_keys(
        smoke,
        {
            "path",
            "sha256",
            "selection_manifest",
            "visual",
            "device",
            "precision",
            "depth_gaussian_heads_precision",
            "case",
            "thresholds",
            "observed",
            "passed",
        },
        name="evidence.reconstruction_render_smoke",
    )
    if not isinstance(smoke["path"], str) or not smoke["path"]:
        raise ValueError("reconstruction/render smoke path must be non-empty")
    _require_sha256(smoke["sha256"], name="reconstruction/render smoke SHA-256")
    for record_name in ("selection_manifest", "visual"):
        record = _require_mapping(smoke[record_name], name=f"smoke {record_name}")
        expected = {"path", "sha256"}
        if record_name == "visual":
            expected |= {"format", "mode", "size_wh"}
        _require_exact_keys(record, expected, name=f"smoke {record_name}")
        if not isinstance(record["path"], str) or not record["path"]:
            raise ValueError(f"smoke {record_name} path must be non-empty")
        _require_sha256(record["sha256"], name=f"smoke {record_name} SHA-256")
    visual = smoke["visual"]
    _require_equal(visual["format"], "JPEG", name="smoke visual format")
    _require_equal(visual["mode"], "RGB", name="smoke visual mode")
    size_wh = visual["size_wh"]
    if (
        not isinstance(size_wh, list)
        or len(size_wh) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size_wh)
    ):
        raise ValueError("smoke visual size_wh must contain two positive integers")
    _require_equal(smoke["device"], "cuda:0", name="smoke device")
    _require_equal(smoke["precision"], "bf16", name="smoke precision")
    _require_equal(
        smoke["depth_gaussian_heads_precision"],
        "fp32",
        name="smoke DepthHead/GaussianHead precision",
    )

    case = _require_mapping(smoke["case"], name="smoke case")
    _require_exact_keys(
        case,
        {
            "config",
            "step",
            "frame_count",
            "dataset_index",
            "scene",
            "clip",
            "frame_indices",
        },
        name="smoke case",
    )
    _require_equal(case["config"], "step_100000_frames_10", name="smoke config")
    _require_equal(case["step"], 100000, name="smoke checkpoint step")
    _require_equal(case["frame_count"], 10, name="smoke frame count")
    if isinstance(case["dataset_index"], bool) or not isinstance(case["dataset_index"], int):
        raise ValueError("smoke dataset_index must be an integer")
    for key in ("scene", "clip"):
        if not isinstance(case[key], str) or not case[key]:
            raise ValueError(f"smoke case {key} must be non-empty")
    frame_indices = case["frame_indices"]
    if (
        not isinstance(frame_indices, list)
        or len(frame_indices) != 10
        or any(isinstance(value, bool) or not isinstance(value, int) for value in frame_indices)
        or frame_indices != sorted(set(frame_indices))
    ):
        raise ValueError("smoke frame_indices must be ten unique sorted integers")

    _require_equal(
        smoke["thresholds"],
        PULLBACK_V2_SMOKE_THRESHOLDS,
        name="smoke thresholds",
    )
    observed = _require_mapping(smoke["observed"], name="smoke observed metrics")
    metric_names = {
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
    }
    _require_exact_keys(observed, metric_names, name="smoke observed metrics")
    values = {name: _finite_float(observed[name], name=f"smoke {name}") for name in metric_names}
    thresholds = PULLBACK_V2_SMOKE_THRESHOLDS

    def inside(name: str, range_name: str) -> bool:
        lower, upper = thresholds[range_name]
        return float(lower) <= values[name] <= float(upper)

    passed = (
        values["render_direct_psnr_db"] >= thresholds["render_direct_psnr_min_db"]
        and values["render_direct_ssim"] >= thresholds["render_direct_ssim_min"]
        and values["render_direct_lpips"] <= thresholds["render_direct_lpips_max"]
        and values["render_gt_psnr_db"]
        >= values["direct_gt_psnr_db"] - thresholds["render_gt_psnr_drop_max_db"]
        and values["render_gt_ssim"]
        >= values["direct_gt_ssim"] - thresholds["render_gt_ssim_drop_max"]
        and values["render_gt_lpips"]
        <= values["direct_gt_lpips"] + thresholds["render_gt_lpips_increase_max"]
        and inside("depth_recon_over_direct", "depth_recon_over_direct_range")
        and values["depth_recon_vs_direct_absrel"]
        <= thresholds["depth_recon_vs_direct_absrel_max"]
        and values["point_xyz_relative_error"] <= thresholds["point_xyz_relative_error_max"]
        and inside("gs_recon_over_direct", "gs_recon_over_direct_range")
        and inside("paired_gs_over_depth", "paired_gs_over_depth_range")
        and values["gs_axis_anisotropy_log_rms"]
        <= thresholds["gs_axis_anisotropy_log_rms_max"]
        and values["support_pixels_median_per_frame"]
        >= thresholds["support_pixels_median_per_frame_min"]
    )
    _require_equal(smoke["passed"], passed, name="reconstruction/render smoke passed")
    _require_equal(smoke["passed"], True, name="reconstruction/render smoke acceptance")


def _validate_v2_evidence_and_limits(
    root: Mapping[str, Any],
    *,
    metric_depth: Mapping[str, Any],
) -> None:
    evidence = _require_mapping(root["evidence"], name="evidence")
    _require_exact_keys(
        evidence,
        {
            "source_metric_gate",
            "gaussian_roundtrip_audit",
            "metric_reference",
            "reconstruction_render_smoke",
            "fit_scenes",
            "selection_scenes",
            "frozen_profile",
            "formal_coverage",
            "gaussian_practical_equivalence_gate",
            "primary_lidar_gate",
        },
        name="evidence",
    )
    _validate_v2_smoke_evidence(root)
    evidence_record_keys = {
        "source_metric_gate": {
            "path",
            "sha256",
            "script_sha256",
            "result_sha256_excluding_self",
        },
        "gaussian_roundtrip_audit": {"path", "sha256", "script_sha256"},
        "metric_reference": {
            "path",
            "sha256",
            "script_sha256",
            "resume_signature_sha256",
        },
    }
    for evidence_name, expected_keys in evidence_record_keys.items():
        record = _require_mapping(evidence[evidence_name], name=evidence_name)
        _require_exact_keys(record, expected_keys, name=f"evidence.{evidence_name}")
        if not isinstance(record["path"], str) or not record["path"]:
            raise ValueError(f"evidence.{evidence_name}.path must be non-empty")
        for key in expected_keys - {"path"}:
            _require_sha256(record[key], name=f"evidence.{evidence_name}.{key}")
    fit_scenes = list(range(300, 320))
    selection_scenes = list(range(320, 330))
    _require_equal(evidence["fit_scenes"], fit_scenes, name="evidence.fit_scenes")
    _require_equal(
        evidence["selection_scenes"],
        selection_scenes,
        name="evidence.selection_scenes",
    )

    frozen = _require_mapping(evidence["frozen_profile"], name="frozen_profile")
    _require_exact_keys(frozen, {"sha256", "payload"}, name="evidence.frozen_profile")
    profile_sha256 = _require_sha256(
        frozen["sha256"], name="evidence.frozen_profile.sha256"
    )
    profile = _require_mapping(frozen["payload"], name="frozen_profile.payload")
    _require_exact_keys(
        profile,
        {
            "form",
            "equation",
            "fit_variable",
            "runtime_variable",
            "evaluate_on",
            "boundary_scope",
            "a",
            "b",
            "c_at_20m",
            "reference_depth_m",
            "runtime_depth_clamp_m",
            "fit_scenes",
            "selection_scenes",
            "calibration_candidate_forms",
            "calibration_candidate_fits",
            "calibration_decision_sha256",
            "reference_json_sha256",
            "window_contract",
        },
        name="evidence.frozen_profile.payload",
    )
    _require_equal(
        _canonical_json_sha256(profile),
        profile_sha256,
        name="evidence.frozen_profile self SHA-256",
    )
    profile_form = profile["form"]
    if profile_form not in PULLBACK_DEPTH_FORMS:
        raise ValueError("frozen_profile.payload.form is unsupported")
    profile_a = _finite_float(profile["a"], name="frozen profile a")
    profile_b = _finite_float(profile["b"], name="frozen profile b")
    if profile_form == "identity" and (profile_a != 0.0 or profile_b != 0.0):
        raise ValueError("identity frozen profile requires a=b=0")
    if profile_form == "constant" and profile_b != 0.0:
        raise ValueError("constant frozen profile requires b=0")
    _require_equal(
        profile["evaluate_on"],
        "uncorrected_reconstructed_metric_depth_m",
        name="frozen profile evaluate_on",
    )
    _require_equal(
        _finite_float(profile["reference_depth_m"], name="frozen reference depth"),
        20.0,
        name="frozen profile reference_depth_m",
    )
    _require_equal(
        profile["runtime_depth_clamp_m"],
        [0.5, 80.0],
        name="frozen profile runtime_depth_clamp_m",
    )
    c_at_20m = _finite_float(profile["c_at_20m"], name="frozen c_at_20m")
    if not math.isclose(c_at_20m, math.exp(profile_a), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("frozen profile c_at_20m does not match exp(a)")
    _require_equal(profile["fit_scenes"], fit_scenes, name="frozen profile fit_scenes")
    _require_equal(
        profile["selection_scenes"],
        selection_scenes,
        name="frozen profile selection_scenes",
    )
    _require_equal(
        profile["calibration_candidate_forms"],
        list(PULLBACK_DEPTH_FORMS),
        name="frozen profile candidate forms",
    )
    candidate_fits = _require_mapping(
        profile["calibration_candidate_fits"], name="frozen candidate fits"
    )
    _require_exact_keys(
        candidate_fits,
        set(PULLBACK_DEPTH_FORMS),
        name="frozen calibration_candidate_fits",
    )
    for candidate_form in PULLBACK_DEPTH_FORMS:
        fit = _require_mapping(
            candidate_fits[candidate_form], name=f"frozen {candidate_form} fit"
        )
        _require_exact_keys(
            fit, {"a", "b", "c_at_20m"}, name=f"frozen {candidate_form} fit"
        )
        fit_a = _finite_float(fit["a"], name=f"frozen {candidate_form} a")
        fit_b = _finite_float(fit["b"], name=f"frozen {candidate_form} b")
        fit_c = _finite_float(
            fit["c_at_20m"], name=f"frozen {candidate_form} c_at_20m"
        )
        if candidate_form == "identity" and (fit_a != 0.0 or fit_b != 0.0):
            raise ValueError("frozen identity candidate requires a=b=0")
        if candidate_form == "constant" and fit_b != 0.0:
            raise ValueError("frozen constant candidate requires b=0")
        if not math.isclose(fit_c, math.exp(fit_a), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"frozen {candidate_form} c_at_20m does not match exp(a)")
    selected_fit = candidate_fits[profile_form]
    _require_equal(selected_fit["a"], profile_a, name="frozen selected profile a")
    _require_equal(selected_fit["b"], profile_b, name="frozen selected profile b")
    _require_sha256(
        profile["calibration_decision_sha256"],
        name="frozen profile calibration_decision_sha256",
    )
    profile_reference_sha = _require_sha256(
        profile["reference_json_sha256"],
        name="frozen profile reference_json_sha256",
    )
    _require_equal(
        profile_reference_sha,
        evidence["metric_reference"]["sha256"],
        name="frozen profile/reference evidence SHA-256",
    )
    _require_equal(
        profile["fit_variable"],
        PULLBACK_DEPTH_VARIABLE_CONTRACT,
        name="frozen profile fit_variable",
    )
    _require_equal(
        profile["runtime_variable"],
        PULLBACK_DEPTH_VARIABLE_CONTRACT,
        name="frozen profile runtime_variable",
    )
    _require_equal(
        profile["boundary_scope"],
        "metric_only; render is forced identity",
        name="frozen profile boundary_scope",
    )
    profile_window = _require_mapping(
        profile["window_contract"], name="frozen profile window_contract"
    )
    _require_equal(
        profile_window.get("expected_window_length"),
        10,
        name="frozen profile window length",
    )
    _require_equal(
        profile_window.get("expected_window_starts"),
        [0, 5, 10, 14, 19],
        name="frozen profile window starts",
    )

    coverage = _require_mapping(evidence["formal_coverage"], name="formal_coverage")
    _require_exact_keys(
        coverage,
        {
            "scene_count",
            "trunk_count",
            "calibration_scene_count",
            "selection_scene_count",
            "required_scenes",
            "required_trunks",
            "window_starts",
            "window_length",
            "data_root",
        },
        name="evidence.formal_coverage",
    )
    _require_equal(coverage["scene_count"], 30, name="formal coverage scene_count")
    _require_equal(coverage["trunk_count"], 90, name="formal coverage trunk_count")
    _require_equal(
        coverage["calibration_scene_count"],
        20,
        name="formal coverage calibration_scene_count",
    )
    _require_equal(
        coverage["selection_scene_count"],
        10,
        name="formal coverage selection_scene_count",
    )
    _require_equal(
        coverage["required_scenes"],
        list(range(300, 330)),
        name="formal coverage scenes",
    )
    _require_equal(coverage["required_trunks"], [0, 1, 2], name="formal coverage trunks")
    _require_equal(coverage["window_starts"], [0, 5, 10, 14, 19], name="window starts")
    _require_equal(coverage["window_length"], 10, name="formal coverage window length")
    if not isinstance(coverage["data_root"], str) or not coverage["data_root"]:
        raise ValueError("formal coverage data_root must be non-empty")

    gs_gate = _require_mapping(
        evidence["gaussian_practical_equivalence_gate"], name="Gaussian gate"
    )
    _require_exact_keys(
        gs_gate,
        {
            "passed",
            "margin",
            "point_estimate",
            "scene_bootstrap_95_ci",
            "scene_count",
            "c_gs_recommendation",
            "support",
            "analysis_unit",
            "bootstrap",
            "per_scene",
        },
        name="evidence.gaussian_practical_equivalence_gate",
    )
    _require_equal(gs_gate["passed"], True, name="Gaussian practical-equivalence passed")
    _require_equal(gs_gate["margin"], [0.95, 1.05], name="Gaussian equivalence margin")
    _require_equal(gs_gate["scene_count"], 30, name="Gaussian gate scene_count")
    c_gs_recommendation = _require_mapping(
        gs_gate["c_gs_recommendation"], name="Gaussian c_gs recommendation"
    )
    _require_exact_keys(
        c_gs_recommendation,
        {"form", "value"},
        name="Gaussian c_gs recommendation",
    )
    _require_equal(
        c_gs_recommendation["form"],
        "identity",
        name="Gaussian c_gs recommendation form",
    )
    _require_equal(
        _finite_float(
            c_gs_recommendation["value"], name="Gaussian c_gs recommendation value"
        ),
        1.0,
        name="Gaussian c_gs recommendation value",
    )
    _require_equal(
        gs_gate["support"],
        "primary_static_nonsky_opacity_0p05",
        name="Gaussian gate support",
    )
    _require_equal(
        gs_gate["analysis_unit"],
        "Waymo scene; bootstrap resamples scenes only",
        name="Gaussian gate analysis unit",
    )
    gs_bootstrap = _require_mapping(gs_gate["bootstrap"], name="Gaussian gate bootstrap")
    _require_exact_keys(
        gs_bootstrap,
        {"unit", "samples", "seed", "confidence_level"},
        name="Gaussian gate bootstrap",
    )
    _require_equal(gs_bootstrap["unit"], "scene", name="Gaussian bootstrap unit")
    _require_equal(gs_bootstrap["samples"], 10000, name="Gaussian bootstrap samples")
    _require_equal(gs_bootstrap["seed"], 20260805, name="Gaussian bootstrap seed")
    _require_equal(
        _finite_float(gs_bootstrap["confidence_level"], name="Gaussian confidence level"),
        0.95,
        name="Gaussian confidence level",
    )
    point = _finite_float(gs_gate["point_estimate"], name="Gaussian gate point")
    gs_ci = gs_gate["scene_bootstrap_95_ci"]
    if not isinstance(gs_ci, list) or len(gs_ci) != 2:
        raise ValueError("Gaussian gate CI must contain two values")
    gs_ci_values = (
        _finite_float(gs_ci[0], name="Gaussian gate CI lower"),
        _finite_float(gs_ci[1], name="Gaussian gate CI upper"),
    )
    if not (0.95 <= point <= 1.05 and 0.95 <= gs_ci_values[0] <= gs_ci_values[1] <= 1.05):
        raise ValueError("Gaussian practical-equivalence evidence is outside its frozen margin")
    gs_per_scene = gs_gate["per_scene"]
    if not isinstance(gs_per_scene, list) or len(gs_per_scene) != 30:
        raise ValueError("Gaussian gate per_scene must contain 30 rows")
    gs_logs: list[float] = []
    for expected_scene, raw_row in zip(range(300, 330), gs_per_scene):
        row = _require_mapping(raw_row, name=f"Gaussian scene {expected_scene}")
        _require_exact_keys(
            row,
            {"scene", "trunk_count", "paired_log_gs_over_depth", "paired_gs_over_depth_ratio"},
            name=f"Gaussian scene {expected_scene}",
        )
        _require_equal(row["scene"], expected_scene, name="Gaussian per-scene order")
        _require_equal(row["trunk_count"], 3, name="Gaussian per-scene trunk count")
        log_ratio = _finite_float(
            row["paired_log_gs_over_depth"], name="Gaussian per-scene paired log ratio"
        )
        ratio = _finite_float(
            row["paired_gs_over_depth_ratio"], name="Gaussian per-scene paired ratio"
        )
        if not math.isclose(ratio, math.exp(log_ratio), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Gaussian per-scene ratio does not match exp(log ratio)")
        gs_logs.append(log_ratio)
    ordered_logs = sorted(gs_logs)
    median_log = 0.5 * (ordered_logs[14] + ordered_logs[15])
    if not math.isclose(point, math.exp(median_log), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Gaussian gate point does not match the 30-scene median")
    recomputed_gs_ci = _scene_bootstrap_median_exp_ci(
        gs_logs,
        samples=10000,
        seed=20260805,
    )
    if any(
        not math.isclose(recorded, recomputed, rel_tol=0.0, abs_tol=1.0e-12)
        for recorded, recomputed in zip(gs_ci_values, recomputed_gs_ci)
    ):
        raise ValueError("Gaussian gate CI does not match the recorded per-scene rows")

    lidar = _require_mapping(evidence["primary_lidar_gate"], name="primary LiDAR gate")
    _require_exact_keys(
        lidar,
        {
            "candidate_form",
            "selected_form",
            "case_count",
            "scene_count",
            "identity_absrel",
            "candidate_absrel",
            "scene_delta_mean",
            "scene_delta_bootstrap_95_ci",
            "improved_scene_count",
            "gate_pass",
            "support",
            "gauge_valid_cases_only",
            "bootstrap",
            "scene_rows",
            "exact_sign_flip",
            "sensitivities",
        },
        name="evidence.primary_lidar_gate",
    )
    _require_equal(lidar["candidate_form"], profile_form, name="LiDAR candidate form")
    _require_equal(lidar["support"], "all_lidar", name="LiDAR primary support")
    _require_equal(
        lidar["gauge_valid_cases_only"], True, name="LiDAR primary gauge-valid filter"
    )
    selected_form = lidar["selected_form"]
    if selected_form not in {"identity", profile_form}:
        raise ValueError("LiDAR selected form must be identity or the frozen candidate")
    _require_equal(metric_depth["form"], selected_form, name="metric depth selected form")
    _require_equal(lidar["scene_count"], 10, name="LiDAR gate scene_count")
    case_count = lidar["case_count"]
    if isinstance(case_count, bool) or not isinstance(case_count, int) or not 1 <= case_count <= 30:
        raise ValueError("LiDAR gate case_count must be an integer in [1,30]")
    identity_absrel = _finite_float(lidar["identity_absrel"], name="identity AbsRel")
    candidate_absrel = _finite_float(lidar["candidate_absrel"], name="candidate AbsRel")
    delta = _finite_float(lidar["scene_delta_mean"], name="LiDAR scene delta mean")
    lidar_ci = lidar["scene_delta_bootstrap_95_ci"]
    if not isinstance(lidar_ci, list) or len(lidar_ci) != 2:
        raise ValueError("LiDAR gate CI must contain two values")
    lidar_ci_values = (
        _finite_float(lidar_ci[0], name="LiDAR gate CI lower"),
        _finite_float(lidar_ci[1], name="LiDAR gate CI upper"),
    )
    if lidar_ci_values[0] > lidar_ci_values[1]:
        raise ValueError("LiDAR gate CI must be ordered")
    introduced = delta > 0.0 and lidar_ci_values[0] > 0.0
    _require_equal(lidar["gate_pass"], introduced, name="LiDAR gate pass")
    _require_equal(
        selected_form,
        profile_form if introduced else "identity",
        name="LiDAR gate selected form",
    )
    if not math.isclose(
        identity_absrel - candidate_absrel, delta, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError("LiDAR AbsRel values do not match the recorded scene delta")
    improved = lidar["improved_scene_count"]
    if isinstance(improved, bool) or not isinstance(improved, int) or not 0 <= improved <= 10:
        raise ValueError("LiDAR improved_scene_count must be an integer in [0,10]")
    lidar_bootstrap = _require_mapping(lidar["bootstrap"], name="LiDAR bootstrap")
    _require_exact_keys(
        lidar_bootstrap,
        {"unit", "samples", "seed", "confidence_level"},
        name="LiDAR bootstrap",
    )
    _require_equal(lidar_bootstrap["unit"], "scene", name="LiDAR bootstrap unit")
    _require_equal(lidar_bootstrap["samples"], 10000, name="LiDAR bootstrap samples")
    _require_equal(lidar_bootstrap["seed"], 20260801, name="LiDAR bootstrap seed")
    _require_equal(
        _finite_float(lidar_bootstrap["confidence_level"], name="LiDAR confidence level"),
        0.95,
        name="LiDAR confidence level",
    )
    scene_rows = lidar["scene_rows"]
    if not isinstance(scene_rows, list) or len(scene_rows) != 10:
        raise ValueError("LiDAR primary scene_rows must contain scenes 320-329")
    scene_identity: list[float] = []
    scene_candidate: list[float] = []
    scene_deltas: list[float] = []
    for expected_scene, raw_row in zip(range(320, 330), scene_rows):
        row = _require_mapping(raw_row, name=f"LiDAR scene {expected_scene}")
        _require_exact_keys(
            row,
            {
                "scene",
                "trunk_count",
                "trunks",
                "candidate_form",
                "identity_absrel",
                "candidate_absrel",
                "identity_minus_candidate",
            },
            name=f"LiDAR scene {expected_scene}",
        )
        _require_equal(row["scene"], expected_scene, name="LiDAR per-scene order")
        _require_equal(row["candidate_form"], profile_form, name="LiDAR per-scene form")
        trunk_count = row["trunk_count"]
        trunks = row["trunks"]
        if (
            isinstance(trunk_count, bool)
            or not isinstance(trunk_count, int)
            or not 1 <= trunk_count <= 3
            or not isinstance(trunks, list)
            or trunks != sorted(set(trunks))
            or any(trunk not in {0, 1, 2} for trunk in trunks)
            or len(trunks) != trunk_count
        ):
            raise ValueError("LiDAR primary scene trunk coverage is invalid")
        row_identity = _finite_float(row["identity_absrel"], name="scene identity AbsRel")
        row_candidate = _finite_float(row["candidate_absrel"], name="scene candidate AbsRel")
        row_delta = _finite_float(row["identity_minus_candidate"], name="scene delta")
        if not math.isclose(
            row_identity - row_candidate, row_delta, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("LiDAR per-scene AbsRel values do not match the delta")
        scene_identity.append(row_identity)
        scene_candidate.append(row_candidate)
        scene_deltas.append(row_delta)
    for observed_value, recomputed_value, name in (
        (identity_absrel, sum(scene_identity) / 10.0, "identity AbsRel"),
        (candidate_absrel, sum(scene_candidate) / 10.0, "candidate AbsRel"),
        (delta, sum(scene_deltas) / 10.0, "scene delta mean"),
    ):
        if not math.isclose(observed_value, recomputed_value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"LiDAR {name} does not match scene_rows")
    _require_equal(
        improved,
        sum(value > 0.0 for value in scene_deltas),
        name="LiDAR improved scene count",
    )
    recomputed_lidar_ci = _scene_bootstrap_mean_ci(
        scene_deltas,
        samples=10000,
        seed=20260801,
    )
    if any(
        not math.isclose(recorded, recomputed, rel_tol=0.0, abs_tol=1.0e-12)
        for recorded, recomputed in zip(lidar_ci_values, recomputed_lidar_ci)
    ):
        raise ValueError("LiDAR gate CI does not match the recorded per-scene rows")
    sign_flip = _require_mapping(lidar["exact_sign_flip"], name="LiDAR exact sign flip")
    _require_exact_keys(
        sign_flip,
        {
            "scene_count",
            "permutation_count",
            "observed_mean_delta",
            "one_sided_positive_p",
            "two_sided_p",
            "role",
        },
        name="LiDAR exact sign flip",
    )
    _require_equal(sign_flip["scene_count"], 10, name="sign-flip scene count")
    _require_equal(sign_flip["permutation_count"], 1024, name="sign-flip permutation count")
    _require_equal(
        sign_flip["role"],
        "sensitivity only; the preregistered gate is the scene-bootstrap CI",
        name="sign-flip role",
    )
    if not math.isclose(
        _finite_float(sign_flip["observed_mean_delta"], name="sign-flip mean"),
        delta,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("sign-flip observed mean does not match the LiDAR gate")
    for key in ("one_sided_positive_p", "two_sided_p"):
        value = _finite_float(sign_flip[key], name=f"sign-flip {key}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"sign-flip {key} must lie in [0,1]")
    sensitivities = _require_mapping(lidar["sensitivities"], name="LiDAR sensitivities")
    _require_exact_keys(
        sensitivities,
        {"all_30_trunks", "phase1a_valid_static_nonsky"},
        name="LiDAR sensitivities",
    )
    for sensitivity_name, expected_support, expected_filter in (
        ("all_30_trunks", "all_lidar", False),
        ("phase1a_valid_static_nonsky", "static_nonsky", True),
    ):
        sensitivity = _require_mapping(
            sensitivities[sensitivity_name], name=f"LiDAR {sensitivity_name}"
        )
        _require_exact_keys(
            sensitivity,
            {
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
            },
            name=f"LiDAR {sensitivity_name}",
        )
        _require_equal(sensitivity["support"], expected_support, name="sensitivity support")
        _require_equal(
            sensitivity["gauge_valid_cases_only"],
            expected_filter,
            name="sensitivity gauge-valid filter",
        )
        _require_equal(sensitivity["candidate_form"], profile_form, name="sensitivity form")
        _require_equal(sensitivity["scene_count"], 10, name="sensitivity scene count")
        sensitivity_count = sensitivity["case_count"]
        if sensitivity_name == "all_30_trunks":
            _require_equal(sensitivity_count, 30, name="all-trunk sensitivity case count")
        elif (
            isinstance(sensitivity_count, bool)
            or not isinstance(sensitivity_count, int)
            or not 1 <= sensitivity_count <= 30
        ):
            raise ValueError("static/non-sky sensitivity case_count must be in [1,30]")
        sensitivity_delta = _finite_float(
            sensitivity["scene_delta_mean"], name="sensitivity scene delta"
        )
        sensitivity_ci = sensitivity["scene_delta_bootstrap_95_ci"]
        if not isinstance(sensitivity_ci, list) or len(sensitivity_ci) != 2:
            raise ValueError("sensitivity CI must contain two values")
        sensitivity_ci_values = [
            _finite_float(value, name="sensitivity CI") for value in sensitivity_ci
        ]
        if sensitivity_ci_values[0] > sensitivity_ci_values[1]:
            raise ValueError("sensitivity CI must be ordered")
        sensitivity_pass = sensitivity_delta > 0.0 and sensitivity_ci_values[0] > 0.0
        _require_equal(sensitivity["gate_pass"], sensitivity_pass, name="sensitivity gate")
        _require_equal(
            sensitivity["selected_form"],
            profile_form if sensitivity_pass else "identity",
            name="sensitivity selected form",
        )
        sensitivity_improved = sensitivity["improved_scene_count"]
        if (
            isinstance(sensitivity_improved, bool)
            or not isinstance(sensitivity_improved, int)
            or not 0 <= sensitivity_improved <= 10
        ):
            raise ValueError("sensitivity improved_scene_count must be in [0,10]")
        sensitivity_sign_flip = _require_mapping(
            sensitivity["exact_sign_flip"], name="sensitivity exact sign flip"
        )
        _require_equal(
            sensitivity_sign_flip.get("scene_count"), 10, name="sensitivity sign-flip scenes"
        )
        _require_equal(
            sensitivity_sign_flip.get("permutation_count"),
            1024,
            name="sensitivity sign-flip permutations",
        )
    if selected_form != "identity":
        _require_equal(metric_depth["a"], profile_a, name="metric depth profile a")
        _require_equal(metric_depth["b"], profile_b, name="metric depth profile b")

    limitations = _require_mapping(root["limitations"], name="limitations")
    _require_exact_keys(
        limitations,
        {
            "phase1b_scheme",
            "similarity_consistent",
            "c_gs",
            "render_scope",
            "metric_scope",
            "residual_limits",
        },
        name="limitations",
    )
    expected_scheme = "A" if selected_form == "identity" else "B"
    _require_equal(limitations["phase1b_scheme"], expected_scheme, name="Phase 1b scheme")
    _require_equal(limitations["similarity_consistent"], True, name="similarity_consistent")
    _require_equal(limitations["c_gs"], "identity", name="limitations.c_gs")
    _require_equal(limitations["render_scope"], "identity_only", name="render scope")
    expected_metric_scope = "identity" if selected_form == "identity" else "depth_profile_only"
    _require_equal(limitations["metric_scope"], expected_metric_scope, name="metric scope")
    residual_limits = limitations["residual_limits"]
    if not isinstance(residual_limits, list) or not residual_limits or any(
        not isinstance(value, str) or not value for value in residual_limits
    ):
        raise ValueError("limitations.residual_limits must contain non-empty strings")


def load_pullback_calibration(
    path: str | Path,
    *,
    tokenizer_checkpoint_path: str | Path,
    dggt_checkpoint_path: str | Path,
    expected_window_len: int,
    expected_patch_grid: Sequence[int],
    expected_artifact_sha256: str | None = None,
) -> PullbackCalibration:
    """Load a production pullback artifact, rejecting every contract mismatch.

    The checkpoint files are hashed by content.  ``expected_artifact_sha256``
    should be supplied when resuming or running inference from a SceneFlow
    checkpoint so a same-tokenizer calibration cannot be silently swapped.
    """

    artifact_path = Path(path).expanduser().resolve()
    tokenizer_path = Path(tokenizer_checkpoint_path).expanduser().resolve()
    dggt_path = Path(dggt_checkpoint_path).expanduser().resolve()
    for required in (artifact_path, tokenizer_path, dggt_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    artifact_sha256 = _sha256_file(artifact_path)
    if expected_artifact_sha256 is not None:
        expected_hash = _require_sha256(
            expected_artifact_sha256, name="expected_artifact_sha256"
        )
        _require_equal(
            artifact_sha256, expected_hash, name="pullback artifact SHA-256"
        )

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid pullback JSON: {artifact_path}") from error
    root = _require_mapping(payload, name="pullback artifact")
    _require_exact_keys(
        root,
        {
            "schema",
            "artifact_role",
            "eligible_for_training",
            "tokenizer_generation",
            "tokenizer",
            "dggt",
            "runtime_contract",
            "boundaries",
            "evidence",
            "limitations",
        },
        name="pullback artifact",
    )

    schema = _require_mapping(root["schema"], name="schema")
    _require_exact_keys(schema, {"name", "version", "strict"}, name="schema")
    _require_equal(schema["name"], PULLBACK_SCHEMA_NAME, name="schema.name")
    _require_equal(schema["strict"], True, name="schema.strict")
    _require_equal(
        root["artifact_role"], PULLBACK_ARTIFACT_ROLE, name="artifact_role"
    )
    _require_equal(
        root["eligible_for_training"], True, name="eligible_for_training"
    )
    tokenizer_generation = root["tokenizer_generation"]
    _require_equal(
        tokenizer_generation,
        "t0_v2",
        name="tokenizer_generation",
    )
    _require_equal(schema["version"], PULLBACK_SCHEMA_VERSION, name="schema.version")

    tokenizer = _require_mapping(root["tokenizer"], name="tokenizer")
    _require_exact_keys(tokenizer, {"sha256", "sha8"}, name="tokenizer")
    tokenizer_sha256 = _require_sha256(tokenizer["sha256"], name="tokenizer.sha256")
    _require_equal(
        tokenizer["sha8"], tokenizer_sha256[:8], name="tokenizer.sha8"
    )
    _require_equal(
        artifact_path.name,
        f"pullback_{tokenizer_sha256[:8]}.json",
        name="pullback artifact filename",
    )

    dggt = _require_mapping(root["dggt"], name="dggt")
    _require_exact_keys(dggt, {"sha256"}, name="dggt")
    dggt_sha256 = _require_sha256(dggt["sha256"], name="dggt.sha256")
    _require_equal(
        _sha256_file(tokenizer_path),
        tokenizer_sha256,
        name="tokenizer checkpoint SHA-256",
    )
    _require_equal(
        _sha256_file(dggt_path), dggt_sha256, name="DGGT checkpoint SHA-256"
    )

    runtime = _require_mapping(root["runtime_contract"], name="runtime_contract")
    _require_exact_keys(
        runtime,
        {
            "version",
            "window_len",
            "patch_grid_hw",
            "gauge_representation",
            "log_metric_scale_units",
        },
        name="runtime_contract",
    )
    _require_equal(
        runtime["version"],
        PULLBACK_RUNTIME_CONTRACT_VERSION,
        name="runtime_contract.version",
    )
    _require_equal(
        runtime["gauge_representation"],
        SCENE_GAUGE_REPRESENTATION,
        name="runtime_contract.gauge_representation",
    )
    _require_equal(
        runtime["log_metric_scale_units"],
        PULLBACK_LOG_METRIC_SCALE_UNITS,
        name="runtime_contract.log_metric_scale_units",
    )
    if (
        isinstance(expected_window_len, bool)
        or not isinstance(expected_window_len, int)
        or expected_window_len <= 0
    ):
        raise ValueError("expected_window_len must be a positive integer")
    window_len = runtime["window_len"]
    if isinstance(window_len, bool) or not isinstance(window_len, int):
        raise ValueError("runtime_contract.window_len must be an integer")
    _require_equal(
        window_len, expected_window_len, name="runtime_contract.window_len"
    )
    try:
        expected_grid = tuple(expected_patch_grid)
    except TypeError as error:
        raise ValueError("expected_patch_grid must contain two positive integers") from error
    if len(expected_grid) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in expected_grid
    ):
        raise ValueError("expected_patch_grid must contain two positive integers")
    artifact_grid_raw = runtime["patch_grid_hw"]
    if not isinstance(artifact_grid_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in artifact_grid_raw
    ):
        raise ValueError("runtime_contract.patch_grid_hw must be an integer JSON list")
    artifact_grid = tuple(artifact_grid_raw)
    _require_equal(
        artifact_grid, expected_grid, name="runtime_contract.patch_grid_hw"
    )

    boundaries = _require_mapping(root["boundaries"], name="boundaries")
    _require_exact_keys(
        boundaries,
        {PULLBACK_RENDER_BOUNDARY, PULLBACK_METRIC_BOUNDARY},
        name="boundaries",
    )
    render = _require_mapping(boundaries["render"], name="boundaries.render")
    _require_exact_keys(
        render, {"depth", "gaussian_scale"}, name="boundaries.render"
    )
    render_depth = _require_mapping(
        render["depth"], name="boundaries.render.depth"
    )
    _require_exact_keys(render_depth, {"form"}, name="boundaries.render.depth")
    _require_equal(
        render_depth["form"], "identity", name="boundaries.render.depth.form"
    )
    render_gs = _require_mapping(
        render["gaussian_scale"], name="boundaries.render.gaussian_scale"
    )
    _require_exact_keys(
        render_gs, {"form", "c_gs"}, name="boundaries.render.gaussian_scale"
    )
    _require_equal(
        render_gs["form"], "identity", name="boundaries.render.gaussian_scale.form"
    )
    _require_equal(
        _finite_float(render_gs["c_gs"], name="render c_gs"),
        1.0,
        name="boundaries.render.gaussian_scale.c_gs",
    )

    metric = _require_mapping(boundaries["metric"], name="boundaries.metric")
    _require_exact_keys(
        metric, {"depth", "gaussian_scale"}, name="boundaries.metric"
    )
    metric_depth = _require_mapping(
        metric["depth"], name="boundaries.metric.depth"
    )
    depth_form = metric_depth.get("form")
    if depth_form not in PULLBACK_DEPTH_FORMS:
        raise ValueError(
            "boundaries.metric.depth.form must be identity, constant, or loglinear"
        )
    if depth_form == "identity":
        _require_exact_keys(
            metric_depth, {"form"}, name="boundaries.metric.depth"
        )
        depth_a = 0.0
        depth_b = 0.0
        reference_depth_m = 20.0
        clamp_m = (0.5, 80.0)
    else:
        _require_exact_keys(
            metric_depth,
            {
                "form",
                "a",
                "b",
                "reference_depth_m",
                "runtime_depth_clamp_m",
                "evaluate_on",
            },
            name="boundaries.metric.depth",
        )
        _require_equal(
            metric_depth["evaluate_on"],
            PULLBACK_DEPTH_EVALUATION,
            name="boundaries.metric.depth.evaluate_on",
        )
        depth_a = _finite_float(metric_depth["a"], name="metric depth a")
        depth_b = _finite_float(metric_depth["b"], name="metric depth b")
        if depth_form == "constant":
            _require_equal(depth_b, 0.0, name="constant metric depth b")
        reference_depth_m = _finite_float(
            metric_depth["reference_depth_m"], name="metric reference_depth_m"
        )
        _require_equal(
            reference_depth_m,
            20.0,
            name="boundaries.metric.depth.reference_depth_m",
        )
        clamp_raw = metric_depth["runtime_depth_clamp_m"]
        if not isinstance(clamp_raw, list) or len(clamp_raw) != 2:
            raise ValueError(
                "metric runtime_depth_clamp_m must be a two-item JSON list"
            )
        clamp_m = (
            _finite_float(clamp_raw[0], name="metric depth clamp minimum"),
            _finite_float(clamp_raw[1], name="metric depth clamp maximum"),
        )
        if not 0.0 < clamp_m[0] < clamp_m[1]:
            raise ValueError(
                "metric runtime_depth_clamp_m must satisfy 0 < min < max"
            )
        _require_equal(
            clamp_m,
            (0.5, 80.0),
            name="boundaries.metric.depth.runtime_depth_clamp_m",
        )

    metric_gs = _require_mapping(
        metric["gaussian_scale"], name="boundaries.metric.gaussian_scale"
    )
    _require_exact_keys(
        metric_gs,
        {"rule", "channels_half_open", "c_gs"},
        name="boundaries.metric.gaussian_scale",
    )
    _require_equal(
        metric_gs["rule"],
        PULLBACK_GS_SCALE_RULE,
        name="boundaries.metric.gaussian_scale.rule",
    )
    _require_equal(
        metric_gs["channels_half_open"],
        [4, 7],
        name="boundaries.metric.gaussian_scale.channels_half_open",
    )
    c_gs = _finite_float(metric_gs["c_gs"], name="metric c_gs")
    _require_equal(c_gs, 1.0, name="boundaries.metric.gaussian_scale.c_gs")

    _validate_v2_evidence_and_limits(root, metric_depth=metric_depth)
    return PullbackCalibration(
        path=artifact_path,
        artifact_sha256=artifact_sha256,
        tokenizer_sha256=tokenizer_sha256,
        dggt_sha256=dggt_sha256,
        tokenizer_generation=tokenizer_generation,
        window_len=window_len,
        patch_grid_hw=artifact_grid,
        depth_a=depth_a,
        depth_b=depth_b,
        reference_depth_m=reference_depth_m,
        runtime_depth_clamp_m=clamp_m,
        c_gs=c_gs,
        runtime_contract_version=PULLBACK_RUNTIME_CONTRACT_VERSION,
        depth_form=depth_form,
    )


def _check_gauge_tensor(gauge: torch.Tensor, *, name: str) -> None:
    if gauge.ndim < 1 or int(gauge.shape[-1]) != SCENE_GAUGE_DIM:
        raise ValueError(
            f"{name} must end in {SCENE_GAUGE_DIM} channels, got {tuple(gauge.shape)}"
        )
    if not bool(torch.isfinite(gauge).all()):
        raise ValueError(f"{name} contains non-finite values")


def _scene_stats_like(
    value: torch.Tensor | Sequence[float], reference: torch.Tensor, *, name: str
) -> torch.Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tuple(result.shape) != (SCENE_GAUGE_DIM,):
        raise ValueError(f"{name} must have shape ({SCENE_GAUGE_DIM},), got {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def normalize_scene_gauge(
    gauge: torch.Tensor,
    mean: torch.Tensor | Sequence[float],
    std: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Normalize the three physical gauge channels independently."""

    _check_gauge_tensor(gauge, name="scene gauge")
    mean_tensor = _scene_stats_like(mean, gauge, name="scene gauge mean")
    std_tensor = _scene_stats_like(std, gauge, name="scene gauge std")
    if bool((std_tensor <= 0).any()):
        raise ValueError("scene gauge std must be positive")
    return (gauge - mean_tensor) / std_tensor.clamp_min(
        SCENE_GAUGE_STATS_STD_FLOOR
    )


def denormalize_scene_gauge(
    normalized: torch.Tensor,
    mean: torch.Tensor | Sequence[float],
    std: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Invert :func:`normalize_scene_gauge` channel by channel."""

    _check_gauge_tensor(normalized, name="normalized scene gauge")
    mean_tensor = _scene_stats_like(mean, normalized, name="scene gauge mean")
    std_tensor = _scene_stats_like(std, normalized, name="scene gauge std")
    if bool((std_tensor <= 0).any()):
        raise ValueError("scene gauge std must be positive")
    return normalized * std_tensor.clamp_min(SCENE_GAUGE_STATS_STD_FLOOR) + mean_tensor


def _scene_scale_for_prefix(
    log_metric_scale: torch.Tensor | float,
    *,
    prefix: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    name: str = "log_metric_scale",
) -> torch.Tensor:
    """Return metres-per-unit broadcastable over a scene tensor prefix."""

    log_scale = torch.as_tensor(log_metric_scale, device=device, dtype=dtype)
    if not bool(torch.isfinite(log_scale).all()):
        raise ValueError(f"{name} must be finite")
    if log_scale.ndim == 0 or log_scale.numel() == 1:
        shaped = log_scale.reshape((1,) * len(prefix))
    elif len(prefix) >= 1 and tuple(log_scale.shape) == (prefix[0],):
        shaped = log_scale.reshape((prefix[0],) + (1,) * (len(prefix) - 1))
    elif len(prefix) >= 1 and tuple(log_scale.shape) == (prefix[0], 1):
        shaped = log_scale.reshape((prefix[0],) + (1,) * (len(prefix) - 1))
    else:
        raise ValueError(
            f"{name} must be scalar, [B], or [B,1] for prefix {prefix}; "
            f"per-frame gauge values are forbidden, got {tuple(log_scale.shape)}"
        )
    return torch.exp(shaped)


def _scaled_camera_pose(
    c2w: torch.Tensor,
    log_metric_scale: torch.Tensor | float,
    *,
    metric_to_dggt: bool,
) -> torch.Tensor:
    if c2w.ndim < 2 or tuple(c2w.shape[-2:]) not in {(3, 4), (4, 4)}:
        raise ValueError(f"c2w must end in [3,4] or [4,4], got {tuple(c2w.shape)}")
    if not bool(torch.isfinite(c2w).all()):
        raise ValueError("c2w contains non-finite values")
    pose_prefix = tuple(c2w.shape[:-2])
    # A rank-3 pose tensor is an unbatched sequence, so a vector here would be
    # a forbidden per-frame gauge.  Rank-4 tensors are explicitly [B,S,...].
    scene_prefix = pose_prefix if c2w.ndim >= 4 else ()
    metres_per_unit = _scene_scale_for_prefix(
        log_metric_scale,
        prefix=scene_prefix,
        device=c2w.device,
        dtype=c2w.dtype,
    )
    result = c2w.clone()
    translation = c2w[..., :3, 3]
    scale = metres_per_unit.unsqueeze(-1)
    result[..., :3, 3] = (
        translation / scale if metric_to_dggt else translation * scale
    )
    return result


def metric_c2w_to_dggt(
    c2w_metric: torch.Tensor, log_metric_scale: torch.Tensor | float
) -> torch.Tensor:
    """Convert metric Waymo c2w to DGGT units by scaling translation only."""

    return _scaled_camera_pose(c2w_metric, log_metric_scale, metric_to_dggt=True)


def metric_c2w_to_teacher_anchor_dggt(
    c2w_metric: torch.Tensor,
    camera_anchor_to_world_metric: torch.Tensor,
    log_metric_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Express a metric trajectory in the teacher camera-anchor world.

    Waymo metric poses use the clip-start ego world (``+z`` up), while frozen
    DGGT geometry and the directional sky atlas use the first camera as their
    world basis (OpenCV/DGGT ``-y`` image-up convention).  Scaling translation
    alone cannot reconcile those bases.  Rebase the complete trajectory by
    the metric trunk-anchor camera first, then convert metres to DGGT units.

    The result has an identity pose at the trunk camera anchor and therefore
    shares the same world-axis convention as the teacher camera trajectory.
    """

    c2w = torch.as_tensor(c2w_metric)
    unbatched = c2w.ndim == 3
    if unbatched:
        c2w = c2w.unsqueeze(0)
    if c2w.ndim != 4 or tuple(c2w.shape[-2:]) != (4, 4):
        raise ValueError(
            "c2w_metric must be [S,4,4] or [B,S,4,4], got "
            f"{tuple(c2w.shape)}"
        )
    anchor = torch.as_tensor(
        camera_anchor_to_world_metric,
        device=c2w.device,
        dtype=c2w.dtype,
    )
    if anchor.ndim == 2:
        anchor = anchor.unsqueeze(0)
    if anchor.ndim == 4 and int(anchor.shape[1]) == 1:
        anchor = anchor[:, 0]
    if anchor.ndim != 3 or tuple(anchor.shape[-2:]) != (4, 4):
        raise ValueError(
            "camera_anchor_to_world_metric must be [4,4], [B,4,4], or "
            f"[B,1,4,4], got {tuple(anchor.shape)}"
        )
    if int(anchor.shape[0]) == 1 and int(c2w.shape[0]) > 1:
        anchor = anchor.expand(int(c2w.shape[0]), -1, -1)
    if int(anchor.shape[0]) != int(c2w.shape[0]):
        raise ValueError(
            "camera anchor batch does not match trajectory batch: "
            f"anchor={int(anchor.shape[0])} trajectory={int(c2w.shape[0])}"
        )
    if not bool(torch.isfinite(c2w).all()) or not bool(torch.isfinite(anchor).all()):
        raise ValueError("metric camera trajectory/anchor contains non-finite values")
    # PPU exposes itself through torch.cuda, but (unlike NVIDIA CUDA) its
    # linalg.inv kernel rejects BF16/FP16 inputs.  Validation reaches this path
    # under autocast, so do the small 4x4 rebase in FP32 on PPU only and retain
    # the caller-visible dtype.  DGGT_DEVICE_BACKEND is set by every PPU
    # launcher and is already the project's discriminator for PPU workarounds.
    is_ppu = os.environ.get("DGGT_DEVICE_BACKEND", "").strip().lower() == "ppu"
    if is_ppu and anchor.dtype in {torch.bfloat16, torch.float16}:
        camera_to_anchor = (
            torch.linalg.inv(anchor.float())[:, None] @ c2w.float()
        ).to(dtype=c2w.dtype)
    else:
        camera_to_anchor = torch.linalg.inv(anchor)[:, None] @ c2w
    result = metric_c2w_to_dggt(camera_to_anchor, log_metric_scale)
    return result[0] if unbatched else result


def dggt_c2w_to_metric(
    c2w_dggt: torch.Tensor, log_metric_scale: torch.Tensor | float
) -> torch.Tensor:
    """Convert DGGT c2w to metres by scaling translation only."""

    return _scaled_camera_pose(c2w_dggt, log_metric_scale, metric_to_dggt=False)


def gauge_to_pose_enc_fov(
    gauge: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """Decode a scene-global gauge into ``[B,S,FOVy,FOVx]`` pose channels.

    ``gauge`` may be ``[3]``, ``[B,3]`` or ``[B,1,3]``.  A time-varying gauge
    is rejected rather than silently averaged.
    """

    if isinstance(seq_len, bool) or int(seq_len) <= 0:
        raise ValueError("seq_len must be a positive integer")
    _check_gauge_tensor(gauge, name="scene gauge")
    if gauge.ndim == 1:
        scene = gauge.reshape(1, SCENE_GAUGE_DIM)
    elif gauge.ndim == 2:
        scene = gauge
    elif gauge.ndim == 3 and int(gauge.shape[1]) == 1:
        scene = gauge[:, 0]
    else:
        raise ValueError(
            "scene gauge must be [3], [B,3], or [B,1,3]; per-frame gauge is forbidden"
        )
    fov_x = 2.0 * torch.atan(torch.exp(scene[..., 1]))
    fov_y = 2.0 * torch.atan(torch.exp(scene[..., 2]))
    fov_yx = torch.stack((fov_y, fov_x), dim=-1)
    if not bool(torch.isfinite(fov_yx).all()) or bool(
        ((fov_yx <= 0.0) | (fov_yx >= math.pi)).any()
    ):
        raise ValueError("decoded gauge FOV must be finite and strictly inside (0, pi)")
    return fov_yx[:, None, :].expand(-1, int(seq_len), -1)


def assemble_dggt_pose_encoding(
    camera_to_world_dggt: torch.Tensor,
    gauge: torch.Tensor,
) -> torch.Tensor:
    """Assemble frozen-DGGT ``[t_w2c,q_xyzw,FOVy,FOVx]`` camera encoding.

    Args:
        camera_to_world_dggt: DGGT-unit c2w matrices shaped ``[S,4,4]`` or
            ``[B,S,4,4]``.
        gauge: Scene-global physical gauge shaped ``[3]``, ``[B,3]`` or
            ``[B,1,3]``.  Its log-scale channel is intentionally unused here;
            callers must first use :func:`metric_c2w_to_dggt` when starting
            from a metric trajectory.

    Returns:
        A batched ``[B,S,9]`` tensor in the exact convention consumed by
        ``pose_encoding_to_extri_intri`` and the frozen DGGT render path.
    """

    c2w = torch.as_tensor(camera_to_world_dggt)
    if c2w.ndim == 3:
        c2w = c2w.unsqueeze(0)
    if c2w.ndim != 4 or tuple(c2w.shape[-2:]) != (4, 4):
        raise ValueError(
            "camera_to_world_dggt must be [S,4,4] or [B,S,4,4], got "
            f"{tuple(c2w.shape)}"
        )
    if not bool(torch.isfinite(c2w).all()):
        raise ValueError("camera_to_world_dggt contains non-finite values")
    batch_size, seq_len = int(c2w.shape[0]), int(c2w.shape[1])
    fov_yx = gauge_to_pose_enc_fov(gauge, seq_len).to(
        device=c2w.device, dtype=c2w.dtype
    )
    if int(fov_yx.shape[0]) == 1 and batch_size > 1:
        fov_yx = fov_yx.expand(batch_size, -1, -1)
    if tuple(fov_yx.shape) != (batch_size, seq_len, 2):
        raise ValueError(
            f"gauge batch {fov_yx.shape[0]} does not match camera batch {batch_size}"
        )

    rotation_c2w = c2w[..., :3, :3]
    rotation_w2c = rotation_c2w.transpose(-1, -2)
    translation_w2c = -torch.matmul(
        rotation_w2c, c2w[..., :3, 3].unsqueeze(-1)
    ).squeeze(-1)
    quaternion_xyzw = mat_to_quat(rotation_w2c)
    return torch.cat((translation_w2c, quaternion_xyzw, fov_yx), dim=-1)


def _expand_xy_for_prefix(
    value: torch.Tensor | Sequence[float],
    *,
    prefix: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    xy = torch.as_tensor(value, device=device, dtype=dtype)
    if xy.ndim < 1 or int(xy.shape[-1]) != 2:
        raise ValueError(f"{name} must end in two [x,y] channels")
    if tuple(xy.shape) == (2,):
        shaped = xy.reshape((1,) * len(prefix) + (2,))
    elif tuple(xy.shape[:-1]) == prefix:
        shaped = xy
    elif len(prefix) >= 2 and tuple(xy.shape) == (prefix[0], 2):
        shaped = xy.reshape((prefix[0],) + (1,) * (len(prefix) - 1) + (2,))
    elif len(prefix) >= 2 and tuple(xy.shape) == (prefix[-1], 2):
        shaped = xy.reshape((1,) * (len(prefix) - 1) + (prefix[-1], 2))
    elif len(prefix) >= 2 and tuple(xy.shape) == (prefix[0], 1, 2):
        shaped = xy.reshape((prefix[0],) + (1,) * (len(prefix) - 1) + (2,))
    else:
        raise ValueError(f"{name} shape {tuple(xy.shape)} cannot map to prefix {prefix}")
    try:
        result = torch.broadcast_to(shaped, prefix + (2,))
    except RuntimeError as error:
        raise ValueError(f"{name} shape {tuple(xy.shape)} cannot map to prefix {prefix}") from error
    if not bool(torch.isfinite(result).all()) or bool(
        ((result <= 0.0) | (result >= math.pi)).any()
    ):
        raise ValueError(f"{name} must be finite and strictly inside (0, pi)")
    return result


def metric_box_to_dggt(
    box_points_metric: torch.Tensor,
    *,
    camera_to_anchor: torch.Tensor,
    log_metric_scale: torch.Tensor | float,
    dggt_fov_xy: torch.Tensor | Sequence[float],
    waymo_fov_xy: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Map metric box corners into the anisotropic DGGT anchor space.

    Args:
        box_points_metric: Points in the metric anchor frame.  Its leading
            dimensions must start with ``camera_to_anchor.shape[:-2]`` and it
            may contain any number of additional point/corner dimensions.
        camera_to_anchor: Per-frame metric camera-to-anchor transforms.  The
            anisotropic diagonal is defined in each camera frame.
        log_metric_scale: Scene-global log metres per DGGT unit.
        dggt_fov_xy: Gauge ``[FOVx,FOVy]`` in radians.
        waymo_fov_xy: Waymo ``[FOVx,FOVy]`` in radians.

    Returns:
        Points with the same shape as ``box_points_metric``, in DGGT anchor
        units.  When the two FOVs match this reduces exactly to scalar
        ``points / exp(log_metric_scale)``.
    """

    points = torch.as_tensor(box_points_metric)
    c2a = torch.as_tensor(
        camera_to_anchor, device=points.device, dtype=points.dtype
    )
    if c2a.ndim < 2 or tuple(c2a.shape[-2:]) != (4, 4):
        raise ValueError(
            f"camera_to_anchor must end in [4,4], got {tuple(c2a.shape)}"
        )
    prefix = tuple(c2a.shape[:-2])
    if points.ndim < len(prefix) + 1 or tuple(points.shape[: len(prefix)]) != prefix:
        raise ValueError(
            f"box_points_metric must begin with camera prefix {prefix}, got {tuple(points.shape)}"
        )
    if int(points.shape[-1]) != 3:
        raise ValueError("box_points_metric must end in xyz")
    if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(c2a).all()):
        raise ValueError("box points and camera_to_anchor must be finite")

    dggt_fov = _expand_xy_for_prefix(
        dggt_fov_xy,
        prefix=prefix,
        device=points.device,
        dtype=points.dtype,
        name="dggt_fov_xy",
    )
    waymo_fov = _expand_xy_for_prefix(
        waymo_fov_xy,
        prefix=prefix,
        device=points.device,
        dtype=points.dtype,
        name="waymo_fov_xy",
    )
    k_xy = torch.tan(dggt_fov / 2.0) / torch.tan(waymo_fov / 2.0)
    k_xyz = torch.cat((k_xy, torch.ones_like(k_xy[..., :1])), dim=-1)
    # A rank-3 camera tensor is an explicit batch of single-frame cameras for
    # this box helper (unbatched sequences should use [1,S,4,4]).  Therefore a
    # matching [B] log scale is valid and must not be mistaken for a forbidden
    # per-frame gauge vector.
    scene_prefix = prefix
    metres_per_unit = _scene_scale_for_prefix(
        log_metric_scale,
        prefix=scene_prefix,
        device=points.device,
        dtype=points.dtype,
    )

    point_shape = tuple(points.shape)
    flattened = points.reshape(prefix + (-1, 3))
    rotation = c2a[..., :3, :3]
    translation = c2a[..., :3, 3]
    points_camera_metric = torch.matmul(
        flattened - translation.unsqueeze(-2), rotation
    )
    points_camera_dggt = (
        points_camera_metric
        * k_xyz.unsqueeze(-2)
        / metres_per_unit.unsqueeze(-1).unsqueeze(-1)
    )
    points_anchor_dggt = torch.matmul(
        points_camera_dggt, rotation.transpose(-1, -2)
    ) + translation.unsqueeze(-2) / metres_per_unit.unsqueeze(-1).unsqueeze(-1)
    return points_anchor_dggt.reshape(point_shape)


def _validate_pullback_inputs(
    depth_recon: torch.Tensor, gs_map: torch.Tensor
) -> tuple[torch.Tensor, bool, tuple[int, ...]]:
    if not torch.is_tensor(depth_recon) or not torch.is_tensor(gs_map):
        raise TypeError("depth_recon and gs_map must be torch tensors")
    no_channel_gs_shape = tuple(depth_recon.shape) + (11,)
    channel_gs_shape = tuple(depth_recon.shape[:-1]) + (11,)
    if tuple(gs_map.shape) == no_channel_gs_shape:
        has_depth_channel = False
    elif (
        depth_recon.ndim in {4, 5}
        and int(depth_recon.shape[-1]) == 1
        and tuple(gs_map.shape) == channel_gs_shape
    ):
        has_depth_channel = True
    else:
        raise ValueError(
            f"gs_map shape {tuple(gs_map.shape)} is incompatible with depth_recon "
            f"shape {tuple(depth_recon.shape)}; expected {no_channel_gs_shape} or "
            f"a singleton-channel depth with GS shape {channel_gs_shape}"
        )
    depth = depth_recon[..., 0] if has_depth_channel else depth_recon
    if depth.ndim not in {3, 4}:
        raise ValueError(
            "depth_recon must be [S,H,W], [S,H,W,1], [B,S,H,W], or [B,S,H,W,1]"
        )
    prefix = (int(depth.shape[0]),) if depth.ndim == 4 else ()
    return depth, has_depth_channel, prefix


def _validate_depth_pullback_input(
    depth_recon: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(depth_recon):
        raise TypeError("depth_recon must be a torch tensor")
    # Four dimensions are canonical [B,S,H,W], even when W happens to be 1.
    # A channel-bearing batched tensor is unambiguous only at five dimensions.
    has_depth_channel = depth_recon.ndim == 5 and int(depth_recon.shape[-1]) == 1
    depth = depth_recon[..., 0] if has_depth_channel else depth_recon
    if depth.ndim not in {3, 4}:
        raise ValueError(
            "depth_recon must be canonical [S,H,W], [B,S,H,W], or [B,S,H,W,1]"
        )
    return depth, has_depth_channel


def metric_depth_correction_factor(
    depth_recon: torch.Tensor,
    *,
    log_metric_scale: torch.Tensor | float,
    calibration: PullbackCalibration,
) -> torch.Tensor:
    """Evaluate the checkpoint-bound metric depth factor exactly once.

    ``depth_recon`` stays in DGGT units. A log-linear profile is evaluated on
    its *uncorrected metric* value, matching the frozen LiDAR selection
    protocol. Identity is an exact no-op and constant profiles do not read the
    scene gauge. Rendering must use :func:`apply_pullback_calibration` with the
    explicit ``render`` boundary instead.
    """

    if not isinstance(calibration, PullbackCalibration):
        raise TypeError("calibration must come from load_pullback_calibration")
    depth, _ = _validate_depth_pullback_input(depth_recon)
    if calibration.depth_form == "identity":
        return torch.ones_like(depth, dtype=torch.float32)
    if calibration.depth_form == "constant":
        return torch.full_like(
            depth,
            math.exp(calibration.depth_a),
            dtype=torch.float32,
        )
    if calibration.depth_form != "loglinear":
        raise ValueError(f"unsupported metric depth form: {calibration.depth_form!r}")
    batch_prefix = (int(depth.shape[0]),) if depth.ndim == 4 else ()
    metres_per_unit = _scene_scale_for_prefix(
        log_metric_scale,
        prefix=batch_prefix,
        device=depth.device,
    )
    if depth.ndim == 4:
        metres_per_unit = metres_per_unit.reshape(int(depth.shape[0]), 1, 1, 1)
    else:
        metres_per_unit = metres_per_unit.reshape(1, 1, 1)
    z0_metric = depth.float() * metres_per_unit
    minimum, maximum = calibration.runtime_depth_clamp_m
    clamped = z0_metric.clamp(min=minimum, max=maximum)
    return torch.exp(
        torch.as_tensor(calibration.depth_a, device=depth.device, dtype=torch.float32)
        + torch.as_tensor(
            calibration.depth_b, device=depth.device, dtype=torch.float32
        )
        * torch.log(clamped / calibration.reference_depth_m)
    )


def apply_depth_pullback_calibration(
    depth_recon: torch.Tensor,
    *,
    log_metric_scale: torch.Tensor | float,
    calibration: PullbackCalibration,
    boundary: Literal["render", "metric"],
    depth_has_channel: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared pullback contract to depth without fabricating GS data.

    Returns ``(depth_dggt, c_depth_factor)``.  The render boundary returns the
    original tensor object and an all-one factor; the metric boundary preserves
    an optional singleton depth channel.  Diagnostics and exporters therefore
    name the same explicit boundary as the full depth+Gaussian path.
    """

    if not isinstance(calibration, PullbackCalibration):
        raise TypeError("calibration must come from load_pullback_calibration")
    if boundary not in PULLBACK_BOUNDARIES:
        raise ValueError(
            f"boundary must be one of {PULLBACK_BOUNDARIES}, got {boundary!r}"
        )
    if depth_has_channel is None:
        depth, has_depth_channel = _validate_depth_pullback_input(depth_recon)
    else:
        has_depth_channel = bool(depth_has_channel)
        if has_depth_channel:
            if depth_recon.ndim not in {4, 5} or int(depth_recon.shape[-1]) != 1:
                raise ValueError(
                    "depth_has_channel=True requires [S,H,W,1] or [B,S,H,W,1]"
                )
            depth = depth_recon[..., 0]
            if depth.ndim not in {3, 4}:
                raise ValueError("singleton-channel depth must reduce to [S,H,W] or [B,S,H,W]")
        else:
            depth, inferred_channel = _validate_depth_pullback_input(depth_recon)
            if inferred_channel:
                raise ValueError("depth_has_channel=False conflicts with rank-5 singleton depth")
    if boundary == PULLBACK_RENDER_BOUNDARY:
        return depth_recon, torch.ones_like(depth, dtype=torch.float32)
    if calibration.depth_form == "identity":
        return depth_recon, torch.ones_like(depth, dtype=torch.float32)
    factor = metric_depth_correction_factor(
        depth if has_depth_channel else depth_recon,
        log_metric_scale=log_metric_scale,
        calibration=calibration,
    )
    corrected_base = depth * factor.to(dtype=depth.dtype)
    corrected = corrected_base.unsqueeze(-1) if has_depth_channel else corrected_base
    return corrected, factor


def apply_pullback_calibration(
    depth_recon: torch.Tensor,
    gs_map: torch.Tensor,
    *,
    log_metric_scale: torch.Tensor | float,
    calibration: PullbackCalibration,
    boundary: Literal["render", "metric"],
) -> PullbackResult:
    """Apply the validated pullback at one explicitly named boundary.

    Outputs remain in DGGT units.  Metric callers must unproject the corrected
    depth first and then multiply means and Gaussian scales by
    ``exp(log_metric_scale)``.
    """

    if not isinstance(calibration, PullbackCalibration):
        raise TypeError("calibration must come from load_pullback_calibration")
    if boundary not in PULLBACK_BOUNDARIES:
        raise ValueError(
            f"boundary must be one of {PULLBACK_BOUNDARIES}, got {boundary!r}"
        )
    _, has_depth_channel, _ = _validate_pullback_inputs(depth_recon, gs_map)
    corrected_depth, factor = apply_depth_pullback_calibration(
        depth_recon,
        log_metric_scale=log_metric_scale,
        calibration=calibration,
        boundary=boundary,
        depth_has_channel=has_depth_channel,
    )
    if boundary == PULLBACK_RENDER_BOUNDARY or calibration.depth_form == "identity":
        return PullbackResult(
            depth_dggt=corrected_depth,
            gs_map_dggt=gs_map,
            c_depth_factor=factor,
            boundary=boundary,
        )
    scale_factor = factor.to(dtype=gs_map.dtype).unsqueeze(-1)
    corrected_gs = torch.cat(
        (
            gs_map[..., :4],
            gs_map[..., 4:7] * scale_factor,
            gs_map[..., 7:11],
        ),
        dim=-1,
    )
    return PullbackResult(
        depth_dggt=corrected_depth,
        gs_map_dggt=corrected_gs,
        c_depth_factor=factor,
        boundary=PULLBACK_METRIC_BOUNDARY,
    )


__all__ = [
    "GAUGE_MROPE_TEMPORAL_OFFSET",
    "PULLBACK_ARTIFACT_ROLE",
    "PULLBACK_BOUNDARIES",
    "PULLBACK_DEPTH_EVALUATION",
    "PULLBACK_DEPTH_FORMS",
    "PULLBACK_DEPTH_VARIABLE_CONTRACT",
    "PULLBACK_GS_SCALE_RULE",
    "PULLBACK_LOG_METRIC_SCALE_UNITS",
    "PULLBACK_METRIC_BOUNDARY",
    "PULLBACK_RENDER_BOUNDARY",
    "PULLBACK_RUNTIME_CONTRACT_VERSION",
    "PULLBACK_SCHEMA_NAME",
    "PULLBACK_SCHEMA_VERSION",
    "PULLBACK_V2_SMOKE_THRESHOLDS",
    "PullbackCalibration",
    "PullbackResult",
    "SCENE_GAUGE_DIM",
    "SCENE_GAUGE_REPRESENTATION",
    "SCENE_GAUGE_STATS_VERSION",
    "apply_depth_pullback_calibration",
    "apply_pullback_calibration",
    "assemble_dggt_pose_encoding",
    "denormalize_scene_gauge",
    "dggt_c2w_to_metric",
    "effective_scene_gauge",
    "gauge_to_pose_enc_fov",
    "load_pullback_calibration",
    "metric_box_to_dggt",
    "metric_depth_correction_factor",
    "metric_c2w_to_dggt",
    "metric_c2w_to_teacher_anchor_dggt",
    "normalize_scene_gauge",
    "resolve_scene_gauge_checkpoint_sha256",
    "scene_gauge_valid_channel_mean",
]
