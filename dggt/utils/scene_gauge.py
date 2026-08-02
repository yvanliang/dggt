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
PULLBACK_SCHEMA_VERSION = "1.0.0"
PULLBACK_ARTIFACT_ROLE = "production_pullback"
PULLBACK_RUNTIME_CONTRACT_VERSION = (
    "metric_depth_gs_same_factor_render_identity_v1"
)
PULLBACK_LOG_METRIC_SCALE_UNITS = "log_metres_per_dggt_unit"
PULLBACK_DEPTH_EVALUATION = (
    "depth_recon_times_exp_log_metric_scale_before_correction"
)
PULLBACK_GS_SCALE_RULE = "multiply_channels_4_7_by_same_depth_factor"
PULLBACK_RENDER_BOUNDARY = "render"
PULLBACK_METRIC_BOUNDARY = "metric"
PULLBACK_BOUNDARIES = (PULLBACK_RENDER_BOUNDARY, PULLBACK_METRIC_BOUNDARY)

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
    _require_equal(schema["version"], PULLBACK_SCHEMA_VERSION, name="schema.version")
    _require_equal(schema["strict"], True, name="schema.strict")
    _require_equal(
        root["artifact_role"], PULLBACK_ARTIFACT_ROLE, name="artifact_role"
    )
    _require_equal(
        root["eligible_for_training"], True, name="eligible_for_training"
    )
    if not isinstance(root["tokenizer_generation"], str) or not root[
        "tokenizer_generation"
    ]:
        raise ValueError("tokenizer_generation must be a non-empty string")

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
    if isinstance(expected_window_len, bool) or int(expected_window_len) <= 0:
        raise ValueError("expected_window_len must be a positive integer")
    window_len = runtime["window_len"]
    if isinstance(window_len, bool) or not isinstance(window_len, int):
        raise ValueError("runtime_contract.window_len must be an integer")
    _require_equal(
        window_len, int(expected_window_len), name="runtime_contract.window_len"
    )
    try:
        expected_grid = tuple(int(value) for value in expected_patch_grid)
    except (TypeError, ValueError) as error:
        raise ValueError("expected_patch_grid must contain two positive integers") from error
    if len(expected_grid) != 2 or any(value <= 0 for value in expected_grid):
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
        metric_depth["form"], "loglinear", name="boundaries.metric.depth.form"
    )
    _require_equal(
        metric_depth["evaluate_on"],
        PULLBACK_DEPTH_EVALUATION,
        name="boundaries.metric.depth.evaluate_on",
    )
    depth_a = _finite_float(metric_depth["a"], name="metric depth a")
    depth_b = _finite_float(metric_depth["b"], name="metric depth b")
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
        raise ValueError("metric runtime_depth_clamp_m must be a two-item JSON list")
    clamp_m = (
        _finite_float(clamp_raw[0], name="metric depth clamp minimum"),
        _finite_float(clamp_raw[1], name="metric depth clamp maximum"),
    )
    if not 0.0 < clamp_m[0] < clamp_m[1]:
        raise ValueError("metric runtime_depth_clamp_m must satisfy 0 < min < max")
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

    evidence = _require_mapping(root["evidence"], name="evidence")
    _require_exact_keys(
        evidence,
        {
            "source_metric_gate",
            "d4_renderer_gate",
            "phase1a_reference",
            "fit_scenes",
            "selection_scenes",
            "primary_lidar_gate",
        },
        name="evidence",
    )
    for evidence_name in (
        "source_metric_gate",
        "d4_renderer_gate",
        "phase1a_reference",
    ):
        record = _require_mapping(evidence[evidence_name], name=evidence_name)
        keys = {"path", "sha256"}
        if evidence_name == "source_metric_gate":
            keys.add("result_sha256_excluding_self")
        _require_exact_keys(record, keys, name=f"evidence.{evidence_name}")
        if not isinstance(record["path"], str) or not record["path"]:
            raise ValueError(f"evidence.{evidence_name}.path must be non-empty")
        _require_sha256(record["sha256"], name=f"evidence.{evidence_name}.sha256")
        if evidence_name == "source_metric_gate":
            _require_sha256(
                record["result_sha256_excluding_self"],
                name="evidence.source_metric_gate.result_sha256_excluding_self",
            )
    if evidence["fit_scenes"] != list(range(300, 320)):
        raise ValueError("evidence.fit_scenes must be the frozen scenes 300-319")
    if evidence["selection_scenes"] != list(range(320, 330)):
        raise ValueError("evidence.selection_scenes must be the frozen scenes 320-329")
    primary = _require_mapping(
        evidence["primary_lidar_gate"], name="evidence.primary_lidar_gate"
    )
    _require_exact_keys(
        primary,
        {
            "case_count",
            "scene_count",
            "identity_absrel",
            "loglinear_absrel",
            "relative_improvement",
            "scene_delta_mean",
            "scene_delta_bootstrap_95_ci",
            "improved_scene_count",
            "gate_pass",
        },
        name="evidence.primary_lidar_gate",
    )
    _require_equal(primary["gate_pass"], True, name="primary_lidar_gate.gate_pass")
    _require_equal(primary["scene_count"], 10, name="primary_lidar_gate.scene_count")
    _require_equal(primary["case_count"], 26, name="primary_lidar_gate.case_count")
    identity_absrel = _finite_float(
        primary["identity_absrel"], name="primary identity AbsRel"
    )
    loglinear_absrel = _finite_float(
        primary["loglinear_absrel"], name="primary loglinear AbsRel"
    )
    relative_improvement = _finite_float(
        primary["relative_improvement"], name="primary relative improvement"
    )
    scene_delta_mean = _finite_float(
        primary["scene_delta_mean"], name="primary scene delta mean"
    )
    improved_scene_count = primary["improved_scene_count"]
    if (
        isinstance(improved_scene_count, bool)
        or not isinstance(improved_scene_count, int)
        or not 0 <= improved_scene_count <= 10
    ):
        raise ValueError("primary improved_scene_count must be an integer in [0,10]")
    if not (
        0.0 <= loglinear_absrel < identity_absrel
        and relative_improvement > 0.0
        and scene_delta_mean > 0.0
        and improved_scene_count > 5
    ):
        raise ValueError("primary LiDAR gate summary is inconsistent with a passing loglinear gate")
    ci = primary["scene_delta_bootstrap_95_ci"]
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("primary LiDAR gate CI must contain two values")
    ci_values = (
        _finite_float(ci[0], name="primary LiDAR gate CI lower"),
        _finite_float(ci[1], name="primary LiDAR gate CI upper"),
    )
    if not 0.0 < ci_values[0] <= ci_values[1]:
        raise ValueError("primary LiDAR gate CI lower bound must be strictly positive")

    limitations = _require_mapping(root["limitations"], name="limitations")
    _require_exact_keys(
        limitations,
        {
            "paired_gaussian_scale_over_depth_ratio",
            "similarity_consistent",
            "c_gs",
            "scope",
        },
        name="limitations",
    )
    ratio = _finite_float(
        limitations["paired_gaussian_scale_over_depth_ratio"],
        name="paired Gaussian-scale/depth ratio",
    )
    if not 0.0 < ratio < 1.0:
        raise ValueError("v1 paired Gaussian-scale/depth ratio must remain in (0, 1)")
    _require_equal(
        limitations["similarity_consistent"],
        False,
        name="limitations.similarity_consistent",
    )
    _require_equal(limitations["c_gs"], "identity", name="limitations.c_gs")
    if not isinstance(limitations["scope"], str) or not limitations["scope"]:
        raise ValueError("limitations.scope must be non-empty")

    return PullbackCalibration(
        path=artifact_path,
        artifact_sha256=artifact_sha256,
        tokenizer_sha256=tokenizer_sha256,
        dggt_sha256=dggt_sha256,
        tokenizer_generation=str(root["tokenizer_generation"]),
        window_len=window_len,
        patch_grid_hw=artifact_grid,
        depth_a=depth_a,
        depth_b=depth_b,
        reference_depth_m=reference_depth_m,
        runtime_depth_clamp_m=clamp_m,
        c_gs=c_gs,
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

    ``depth_recon`` stays in DGGT units. The log-linear profile is evaluated
    on its *uncorrected metric* value, matching the frozen LiDAR selection
    protocol. This helper is shared by export and the end-to-end metric-depth
    diagnostic; rendering must use :func:`apply_pullback_calibration` with the
    explicit ``render`` boundary instead.
    """

    if not isinstance(calibration, PullbackCalibration):
        raise TypeError("calibration must come from load_pullback_calibration")
    depth, _ = _validate_depth_pullback_input(depth_recon)
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
    if boundary == PULLBACK_RENDER_BOUNDARY:
        return PullbackResult(
            depth_dggt=corrected_depth,
            gs_map_dggt=gs_map,
            c_depth_factor=factor,
            boundary=PULLBACK_RENDER_BOUNDARY,
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
    "PULLBACK_GS_SCALE_RULE",
    "PULLBACK_LOG_METRIC_SCALE_UNITS",
    "PULLBACK_METRIC_BOUNDARY",
    "PULLBACK_RENDER_BOUNDARY",
    "PULLBACK_RUNTIME_CONTRACT_VERSION",
    "PULLBACK_SCHEMA_NAME",
    "PULLBACK_SCHEMA_VERSION",
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
