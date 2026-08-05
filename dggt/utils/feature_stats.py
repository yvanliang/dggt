"""Latent feature statistics for SceneFlow training."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist

from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_STATS_STD_FLOOR,
    CAMERA_STATS_VERSION,
    CAMERA_TARGET_SOURCE,
    CAMERA_TARGET_SPACE,
)
from dggt.utils.factorized_asset_condition import (
    FACTORIZED_ASSET_CONDITION_VERSION,
    PLACEMENT_PASSTHROUGH_CHANNELS,
    PLACEMENT_STANDARDIZED_CHANNELS,
    PLACEMENT_STATE_DIM,
)
from dggt.utils.scene_gauge import (
    SCENE_GAUGE_DIM,
    SCENE_GAUGE_REPRESENTATION,
    SCENE_GAUGE_STATS_STD_FLOOR,
    SCENE_GAUGE_STATS_VERSION,
)


DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "scene_flow_pretrain_1024"
    / "feature_stats_pretrain_v5.pt"
)

FEATURE_STATS_SCHEMA = "scene_flow_metric_gauge_feature_stats"
FEATURE_STATS_SCHEMA_VERSION = "4.0.0"


def _canonical_sha256(value: object, *, name: str) -> str:
    digest = str(value).lower() if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest, got {value!r}")
    return digest


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint once on rank 0 and broadcast under DDP."""
    value: str | None = None
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
    if dist.is_available() and dist.is_initialized():
        values = [value]
        dist.broadcast_object_list(values, src=0)
        value = values[0]
    assert value is not None
    return value


def _extract_stats(payload: dict, token_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    for mean_key, std_key in (
        ("mu_z", "sigma_z"),
        ("mean_z", "std_z"),
        ("mean", "std"),
        ("mu", "sigma"),
    ):
        if mean_key in payload and std_key in payload:
            mu = torch.as_tensor(payload[mean_key]).float()
            sigma = torch.as_tensor(payload[std_key]).float()
            break
    else:
        raise KeyError(
            "Feature stats must contain one of (mu_z,sigma_z), "
            "(mean_z,std_z), (mean,std), or (mu,sigma)."
        )

    if mu.ndim > 1:
        mu = mu.reshape(-1)
    if sigma.ndim > 1:
        sigma = sigma.reshape(-1)
    if mu.numel() != int(token_dim) or sigma.numel() != int(token_dim):
        raise ValueError(
            f"Expected latent stats with {token_dim} channels, got "
            f"mu={tuple(mu.shape)} sigma={tuple(sigma.shape)}. "
            "SceneFlow requires tokenizer-latent statistics, not raw 3072-D "
            "aggregator feature stats."
        )
    if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()):
        raise ValueError("Feature statistics contain non-finite values")
    if bool((sigma <= 0.0).any()):
        raise ValueError("Latent feature standard deviation must be positive in every channel")
    return mu, sigma


def load_feature_stats(path: str | Path, token_dim: int = 768) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature stats at {path} must be a dict, got {type(payload).__name__}")
    return _extract_stats(payload, token_dim=token_dim)


def load_into_buffers(module, path: str | Path | None, token_dim: int = 768) -> None:
    """Load latent stats into any module exposing ``set_latent_stats`` or buffers."""
    if path is None:
        return
    mu, sigma = load_feature_stats(path, token_dim=token_dim)
    if hasattr(module, "set_latent_stats"):
        module.set_latent_stats(mu, sigma)
        return
    if not hasattr(module, "mu_z") or not hasattr(module, "sigma_z"):
        raise AttributeError("module must expose set_latent_stats() or mu_z/sigma_z buffers")
    module.mu_z.copy_(mu.to(device=module.mu_z.device, dtype=module.mu_z.dtype))
    module.sigma_z.copy_(sigma.to(device=module.sigma_z.device, dtype=module.sigma_z.dtype))


def validate_camera_stats_provenance(payload: dict, expected_dggt_sha256: str) -> None:
    if payload.get("camera_generation_representation") != CAMERA_GENERATION_REPRESENTATION:
        raise ValueError(
            "Camera stats representation mismatch: expected "
            f"{CAMERA_GENERATION_REPRESENTATION!r}, got {payload.get('camera_generation_representation')!r}"
        )
    if payload.get("camera_stats_version") != CAMERA_STATS_VERSION:
        raise ValueError(
            f"Camera stats version mismatch: expected {CAMERA_STATS_VERSION!r}, "
            f"got {payload.get('camera_stats_version')!r}"
        )
    if payload.get("camera_target_space") != CAMERA_TARGET_SPACE:
        raise ValueError(
            f"Camera stats target space mismatch: expected {CAMERA_TARGET_SPACE!r}, "
            f"got {payload.get('camera_target_space')!r}"
        )
    if payload.get("camera_target_source") != CAMERA_TARGET_SOURCE:
        raise ValueError(
            f"Camera stats target source mismatch: expected {CAMERA_TARGET_SOURCE!r}, "
            f"got {payload.get('camera_target_source')!r}"
        )
    actual_hash = payload.get("dggt_checkpoint_sha256")
    if not isinstance(actual_hash, str) or actual_hash != str(expected_dggt_sha256):
        raise ValueError(
            "Camera stats DGGT checkpoint mismatch: "
            f"expected {expected_dggt_sha256!r}, got {actual_hash!r}"
        )
    for role in ("anchor", "delta"):
        count = int(torch.as_tensor(payload.get(f"camera_{role}_count", 0)).item())
        if count <= 0:
            raise ValueError(f"Camera stats require a positive camera_{role}_count, got {count}")


def validate_scene_gauge_stats_provenance(
    payload: dict,
    expected_scene_gauge_sha256: str,
) -> None:
    """Validate that gauge statistics describe the exact offline GT table."""

    if payload.get("scene_gauge_representation") != SCENE_GAUGE_REPRESENTATION:
        raise ValueError(
            "Scene-gauge stats representation mismatch: expected "
            f"{SCENE_GAUGE_REPRESENTATION!r}, got {payload.get('scene_gauge_representation')!r}"
        )
    if payload.get("scene_gauge_stats_version") != SCENE_GAUGE_STATS_VERSION:
        raise ValueError(
            "Scene-gauge stats version mismatch: expected "
            f"{SCENE_GAUGE_STATS_VERSION!r}, got {payload.get('scene_gauge_stats_version')!r}"
        )
    if int(payload.get("scene_gauge_dim", -1)) != SCENE_GAUGE_DIM:
        raise ValueError(
            f"Scene-gauge stats dimension mismatch: expected {SCENE_GAUGE_DIM}, "
            f"got {payload.get('scene_gauge_dim')!r}"
        )
    expected_hash = _canonical_sha256(
        expected_scene_gauge_sha256,
        name="expected scene-gauge table SHA-256",
    )
    actual_raw = payload.get("gauge_table_sha256")
    actual_hash = _canonical_sha256(actual_raw, name="feature-stats gauge-table SHA-256")
    if actual_hash != expected_hash:
        raise ValueError(
            "Scene-gauge stats table mismatch: "
            f"expected {expected_hash!r}, got {actual_hash!r}"
        )


def validate_tokenizer_stats_provenance(
    payload: dict,
    expected_tokenizer_sha256: str,
) -> str:
    """Require latent statistics to be bound to the exact tokenizer weights."""

    expected_hash = _canonical_sha256(
        expected_tokenizer_sha256,
        name="expected tokenizer checkpoint SHA-256",
    )
    actual_hash = _canonical_sha256(
        payload.get("tokenizer_checkpoint_sha256"),
        name="feature-stats tokenizer checkpoint SHA-256",
    )
    if actual_hash != expected_hash:
        raise ValueError(
            "Feature-stats tokenizer checkpoint mismatch: "
            f"expected {expected_hash!r}, got {actual_hash!r}"
        )
    return actual_hash


def validate_production_stats_coverage(payload: dict) -> None:
    """Reject smoke/subset statistics at formal metric-gauge entrypoints."""

    if payload.get("stats_schema") != FEATURE_STATS_SCHEMA:
        raise ValueError(
            f"Feature stats schema mismatch: expected {FEATURE_STATS_SCHEMA!r}, "
            f"got {payload.get('stats_schema')!r}"
        )
    if payload.get("stats_schema_version") != FEATURE_STATS_SCHEMA_VERSION:
        raise ValueError(
            "Feature stats schema version mismatch: expected "
            f"{FEATURE_STATS_SCHEMA_VERSION!r}, got {payload.get('stats_schema_version')!r}"
        )
    if payload.get("stats_status") != "complete":
        raise ValueError(
            "Formal metric-gauge training/inference requires complete feature stats; "
            f"got status={payload.get('stats_status')!r}"
        )
    coverage = payload.get("stats_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Feature stats are missing stats_coverage")
    expected = int(coverage.get("expected_batches", -1))
    processed = int(coverage.get("processed_batches", -2))
    expected_latent_count = int(coverage.get("expected_latent_count", -1))
    latent_count = int(coverage.get("latent_count", -2))
    if (
        expected <= 0
        or processed != expected
        or expected_latent_count <= 0
        or latent_count != expected_latent_count
        or coverage.get("exact_latent_count") is not True
        or coverage.get("full_dataset_pass") is not True
        or coverage.get("exact_scene_gauge_scope") is not True
        or coverage.get("max_batches") is not None
    ):
        raise ValueError(
            "Feature stats do not prove a full exact-scope dataset pass: "
            f"{coverage!r}"
        )


def _scene_gauge_stats_from_payload(
    payload: dict,
    *,
    expected_scene_gauge_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_scene_gauge_stats_provenance(payload, expected_scene_gauge_sha256)
    values: list[torch.Tensor] = []
    for key in ("gauge_mean", "gauge_std", "gauge_count"):
        if key not in payload:
            raise KeyError(f"Feature stats are missing required scene-gauge field {key!r}")
        value = torch.as_tensor(payload[key]).reshape(-1)
        if value.numel() != SCENE_GAUGE_DIM:
            raise ValueError(
                f"{key} must have {SCENE_GAUGE_DIM} channels, got {value.numel()}"
            )
        values.append(value)
    mean, std = values[0].float(), values[1].float()
    count_raw = values[2]
    if count_raw.is_floating_point() and not torch.equal(count_raw, count_raw.round()):
        raise ValueError("gauge_count must contain integer counts")
    count = count_raw.long()
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("Scene-gauge statistics contain non-finite values")
    if bool((std <= 0.0).any()):
        raise ValueError("gauge_std must be positive in every channel")
    if bool((count <= 0).any()):
        raise ValueError("gauge_count must be positive in every channel")
    return mean, std.clamp_min(SCENE_GAUGE_STATS_STD_FLOOR), count


def _placement_stats_from_payload(payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    required_metadata = {
        "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
        "placement_dim": PLACEMENT_STATE_DIM,
        "placement_standardized_channels": list(PLACEMENT_STANDARDIZED_CHANNELS),
        "placement_passthrough_channels": list(PLACEMENT_PASSTHROUGH_CHANNELS),
    }
    for key, expected in required_metadata.items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(
                f"Placement stats provenance mismatch at {key}: expected {expected!r}, got {actual!r}"
            )
    if "placement_mean" not in payload or "placement_std" not in payload:
        raise KeyError("Feature stats require placement_mean and placement_std together")
    placement_mean = torch.as_tensor(payload["placement_mean"]).float().reshape(-1)
    placement_std = torch.as_tensor(payload["placement_std"]).float().reshape(-1)
    if placement_mean.numel() != PLACEMENT_STATE_DIM or placement_std.numel() != PLACEMENT_STATE_DIM:
        raise ValueError(
            f"placement_mean/std must each contain {PLACEMENT_STATE_DIM} values"
        )
    if not bool(torch.isfinite(placement_mean).all()) or not bool(
        torch.isfinite(placement_std).all()
    ):
        raise ValueError("placement statistics contain non-finite values")
    if bool((placement_std <= 0.0).any()):
        raise ValueError("placement_std must be positive")
    passthrough = torch.tensor(PLACEMENT_PASSTHROUGH_CHANNELS, dtype=torch.long)
    if not torch.equal(placement_mean.index_select(0, passthrough), torch.zeros(len(passthrough))):
        raise ValueError("placement passthrough channels must have mean=0")
    if not torch.equal(placement_std.index_select(0, passthrough), torch.ones(len(passthrough))):
        raise ValueError("placement passthrough channels must have std=1")
    return placement_mean, placement_std


def validate_stats_sequence_length(payload: dict, expected_sequence_length: int) -> None:
    source = payload.get("source")
    actual = source.get("sequence_length") if isinstance(source, dict) else None
    if actual is None:
        raise ValueError(
            "Feature stats are missing source.sequence_length; regenerate them with "
            "tools/compute_pretrain_feature_stats.py."
        )
    if int(actual) != int(expected_sequence_length):
        raise ValueError(
            "Feature stats sequence-length mismatch: expected "
            f"{int(expected_sequence_length)} frames, got {int(actual)}."
        )


def validate_stats_patch_grid(
    payload: dict,
    expected_patch_grid: tuple[int, int] | list[int],
) -> None:
    """Bind a feature-stats artifact to the tokenizer patch lattice.

    Camera/gauge statistics happen to be grid-independent, but the latent and
    placement statistics in the same production artifact are not.  Accepting a
    stats file produced on another lattice would therefore be a silent
    coordinate-system change even when all channel counts still match.
    """

    expected = tuple(int(value) for value in expected_patch_grid)
    if len(expected) != 2 or any(value <= 0 for value in expected):
        raise ValueError(
            f"expected_patch_grid must contain two positive integers, got {expected_patch_grid!r}"
        )
    source = payload.get("source")
    actual_raw = source.get("patch_grid") if isinstance(source, dict) else None
    if (
        not isinstance(actual_raw, (list, tuple))
        or len(actual_raw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in actual_raw)
    ):
        raise ValueError(
            "Feature stats are missing a valid source.patch_grid; regenerate them with "
            "tools/compute_pretrain_feature_stats.py."
        )
    actual = tuple(int(value) for value in actual_raw)
    if any(value <= 0 for value in actual):
        raise ValueError(f"Feature stats source.patch_grid must be positive, got {actual!r}")
    if actual != expected:
        raise ValueError(
            "Feature stats patch-grid mismatch: expected "
            f"{list(expected)}, got {list(actual)}."
        )


def _camera_stats_from_payload(
    payload: dict,
    *,
    expected_dggt_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_camera_stats_provenance(payload, expected_dggt_sha256)
    result = []
    for key in ("camera_anchor_mean", "camera_anchor_std", "camera_delta_mean", "camera_delta_std"):
        if key not in payload:
            raise KeyError(f"Feature stats are missing required camera field {key!r}")
        value = torch.as_tensor(payload[key]).float().reshape(-1)
        if value.numel() != CAMERA_GENERATION_DIM:
            raise ValueError(f"{key} must have {CAMERA_GENERATION_DIM} channels, got {value.numel()}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} contains non-finite values")
        if key.endswith("_std"):
            if bool((value <= 0.0).any()):
                raise ValueError(f"{key} must be positive in every channel")
        result.append(value)
    return tuple(result)  # type: ignore[return-value]


def load_camera_feature_stats(
    path: str | Path,
    *,
    expected_dggt_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature stats at {path} must be a dict, got {type(payload).__name__}")
    return _camera_stats_from_payload(payload, expected_dggt_sha256=expected_dggt_sha256)


def load_scene_gauge_feature_stats(
    path: str | Path,
    *,
    expected_scene_gauge_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature stats at {path} must be a dict, got {type(payload).__name__}")
    return _scene_gauge_stats_from_payload(
        payload,
        expected_scene_gauge_sha256=expected_scene_gauge_sha256,
    )


def load_all_stats_into_buffers(
    module,
    path: str | Path,
    token_dim: int = 768,
    *,
    dggt_ckpt_path: str | Path | None = None,
    expected_dggt_sha256: str | None = None,
    tokenizer_ckpt_path: str | Path | None = None,
    expected_tokenizer_sha256: str | None = None,
    scene_gauge_path: str | Path | None = None,
    expected_scene_gauge_sha256: str | None = None,
    expected_sequence_length: int | None = None,
    expected_patch_grid: tuple[int, int] | list[int] | None = None,
    require_existing_match: bool = False,
) -> str:
    """Load latent, camera, gauge, and placement statistics with provenance checks.

    ``require_existing_match`` is intended for checkpoint warm-start/resume.  A
    SceneFlow checkpoint was trained in the coordinate system defined by these
    buffers, so replacing them with different values would invalidate the
    checkpoint even when all tensor shapes still match.
    """
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature stats at {path} must be a dict, got {type(payload).__name__}")
    if expected_sequence_length is not None:
        validate_stats_sequence_length(payload, expected_sequence_length)
    if expected_patch_grid is not None:
        validate_stats_patch_grid(payload, expected_patch_grid)
    mu, sigma = _extract_stats(payload, token_dim=token_dim)
    if expected_tokenizer_sha256 is None and tokenizer_ckpt_path is not None:
        expected_tokenizer_sha256 = checkpoint_sha256(tokenizer_ckpt_path)
    elif expected_tokenizer_sha256 is not None and tokenizer_ckpt_path is not None:
        actual_tokenizer_sha256 = checkpoint_sha256(tokenizer_ckpt_path)
        if actual_tokenizer_sha256 != str(expected_tokenizer_sha256).lower():
            raise ValueError(
                "Explicit tokenizer hash does not match the provided checkpoint: "
                f"expected {expected_tokenizer_sha256!r}, got {actual_tokenizer_sha256!r}"
            )
    if expected_tokenizer_sha256 is not None:
        expected_tokenizer_sha256 = validate_tokenizer_stats_provenance(
            payload,
            str(expected_tokenizer_sha256),
        )
    if expected_dggt_sha256 is None:
        if dggt_ckpt_path is None:
            raise ValueError("DGGT checkpoint path/hash is required to validate camera statistics")
        expected_dggt_sha256 = checkpoint_sha256(dggt_ckpt_path)
    camera = _camera_stats_from_payload(payload, expected_dggt_sha256=expected_dggt_sha256)

    # Gauge is an explicit pretrain/inference contract. The formal edit path
    # instantiates the shared SceneFlow class but deliberately does not generate
    # camera/sky/gauge, so merely having dormant gauge buffers must not force it
    # to own a teacher-table artifact (Phase 6 clean separation).
    module_requires_gauge = (
        scene_gauge_path is not None or expected_scene_gauge_sha256 is not None
    )
    gauge = None
    if module_requires_gauge:
        validate_production_stats_coverage(payload)
        if not hasattr(module, "set_gauge_stats"):
            raise AttributeError(
                "Scene-gauge statistics were requested but the module does not expose set_gauge_stats()"
            )
        if expected_scene_gauge_sha256 is None:
            if scene_gauge_path is None:
                raise ValueError(
                    "Scene-gauge table path/hash is required to validate gauge statistics"
                )
            expected_scene_gauge_sha256 = checkpoint_sha256(scene_gauge_path)
        elif scene_gauge_path is not None:
            actual_table_hash = checkpoint_sha256(scene_gauge_path)
            if actual_table_hash != str(expected_scene_gauge_sha256):
                raise ValueError(
                    "Explicit scene-gauge hash does not match the provided table: "
                    f"expected {expected_scene_gauge_sha256!r}, got {actual_table_hash!r}"
                )
        gauge = _scene_gauge_stats_from_payload(
            payload,
            expected_scene_gauge_sha256=str(expected_scene_gauge_sha256),
        )

    placement = None
    has_placement_mean = "placement_mean" in payload
    has_placement_std = "placement_std" in payload
    if has_placement_mean != has_placement_std:
        raise KeyError("Feature stats require placement_mean and placement_std together")
    if has_placement_mean:
        placement = _placement_stats_from_payload(payload)
    elif str(getattr(getattr(module, "config", None), "asset_condition_protocol", "")) == "factorized_v1":
        raise KeyError(
            "Factorized SceneFlow pretraining requires placement_mean/placement_std "
            "in the feature-stats file; regenerate it with tools/compute_pretrain_feature_stats.py."
        )
    if not hasattr(module, "set_latent_stats") or not hasattr(module, "set_camera_stats"):
        raise AttributeError("SceneFlow module must expose set_latent_stats() and set_camera_stats()")
    values = {
        "mu_z": mu,
        "sigma_z": sigma,
        "camera_anchor_mean": camera[0],
        "camera_anchor_std": camera[1],
        "camera_delta_mean": camera[2],
        "camera_delta_std": camera[3],
    }
    if gauge is not None:
        values["gauge_mean"] = gauge[0]
        values["gauge_std"] = gauge[1]
    if placement is not None:
        values["placement_mean"] = placement[0]
        values["placement_std"] = placement[1]
    if require_existing_match:
        mismatches: list[str] = []
        for name, expected in values.items():
            if not hasattr(module, name):
                mismatches.append(f"{name}: checkpoint model has no such buffer")
                continue
            actual = torch.as_tensor(getattr(module, name)).detach().cpu().float().reshape(-1)
            expected = expected.detach().cpu().float().reshape(-1)
            if tuple(actual.shape) != tuple(expected.shape):
                mismatches.append(
                    f"{name}: checkpoint shape={tuple(actual.shape)} stats shape={tuple(expected.shape)}"
                )
                continue
            if not torch.equal(actual, expected):
                difference = (actual - expected).abs()
                mismatches.append(
                    f"{name}: max_abs_diff={float(difference.max().item()):.9g}, "
                    f"mean_abs_diff={float(difference.mean().item()):.9g}"
                )
        if hasattr(module, "camera_stats_valid") and not bool(
            torch.as_tensor(module.camera_stats_valid).item()
        ):
            mismatches.append("camera_stats_valid: checkpoint does not contain valid camera statistics")
        if gauge is not None and hasattr(module, "gauge_stats_valid") and not bool(
            torch.as_tensor(module.gauge_stats_valid).item()
        ):
            mismatches.append("gauge_stats_valid: checkpoint does not contain valid gauge statistics")
        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                f"Feature statistics at {path} do not match the statistics stored in the loaded "
                f"SceneFlow checkpoint: {details}. Refusing to replace the checkpoint's latent/camera/gauge "
                "coordinate system. Use the exact feature-stats file used to train that checkpoint."
            )
    module.set_latent_stats(mu, sigma)
    if expected_tokenizer_sha256 is not None:
        module._tokenizer_checkpoint_sha256 = str(expected_tokenizer_sha256)
    module.set_camera_stats(*camera)
    if gauge is not None:
        if not hasattr(module, "set_gauge_stats"):
            raise AttributeError("SceneFlow module must expose set_gauge_stats()")
        module.set_gauge_stats(gauge[0], gauge[1])
        module._scene_gauge_table_sha256 = str(expected_scene_gauge_sha256)
    if placement is not None:
        if not hasattr(module, "set_placement_stats"):
            raise AttributeError("SceneFlow module must expose set_placement_stats()")
        module.set_placement_stats(*placement)
    return expected_dggt_sha256


@torch.no_grad()
def compute_per_channel_stats(
    latents: Iterable[torch.Tensor],
    *,
    token_dim: int = 768,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute per-channel mean/std over an iterable of ``[... , D]`` tensors."""
    total = 0
    sum_x = torch.zeros(token_dim, dtype=torch.float64)
    sum_x2 = torch.zeros(token_dim, dtype=torch.float64)

    for z in latents:
        if z.shape[-1] != int(token_dim):
            raise ValueError(f"Expected last dim {token_dim}, got {z.shape[-1]}")
        flat = z.detach().to(device="cpu", dtype=torch.float64).reshape(-1, token_dim)
        total += int(flat.shape[0])
        sum_x += flat.sum(dim=0)
        sum_x2 += flat.square().sum(dim=0)

    if total == 0:
        raise ValueError("Cannot compute feature stats from an empty iterable.")

    mean = sum_x / float(total)
    var = (sum_x2 / float(total) - mean.square()).clamp_min(float(eps))
    std = var.sqrt()
    return {
        "mu_z": mean.float(),
        "sigma_z": std.float(),
        "count": torch.tensor(total, dtype=torch.long),
    }


@torch.no_grad()
def compute_scene_gauge_stats(
    gauges: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    scene_gauge_table_sha256: str,
    eps: float = SCENE_GAUGE_STATS_STD_FLOOR**2,
) -> dict[str, torch.Tensor | str | int | list[int]]:
    """Compute masked per-channel stats for one scene-global token per trunk."""

    table_sha256 = _canonical_sha256(
        scene_gauge_table_sha256,
        name="scene-gauge table SHA-256",
    )
    sums = torch.zeros(SCENE_GAUGE_DIM, dtype=torch.float64)
    sums2 = torch.zeros_like(sums)
    counts = torch.zeros(SCENE_GAUGE_DIM, dtype=torch.long)
    for gauge, valid in gauges:
        values = torch.as_tensor(gauge).detach().cpu().double()
        mask = torch.as_tensor(valid).detach().cpu().bool()
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if values.shape != mask.shape or int(values.shape[-1]) != SCENE_GAUGE_DIM:
            raise ValueError(
                "scene gauge/value masks must have matching shapes [...,3], got "
                f"{tuple(values.shape)} and {tuple(mask.shape)}"
            )
        if not bool(torch.isfinite(values[mask]).all()):
            raise ValueError("valid scene-gauge values contain non-finite values")
        flat_values = values.reshape(-1, SCENE_GAUGE_DIM)
        flat_mask = mask.reshape(-1, SCENE_GAUGE_DIM)
        for channel in range(SCENE_GAUGE_DIM):
            selected = flat_values[:, channel][flat_mask[:, channel]]
            counts[channel] += int(selected.numel())
            if selected.numel():
                sums[channel] += selected.sum()
                sums2[channel] += selected.square().sum()
    if bool((counts <= 0).any()):
        raise ValueError(
            "scene-gauge stats require at least one valid target in every channel; "
            f"got counts={counts.tolist()}"
        )
    mean = sums / counts.double()
    variance = (sums2 / counts.double() - mean.square()).clamp_min(float(eps))
    return {
        "scene_gauge_representation": SCENE_GAUGE_REPRESENTATION,
        "scene_gauge_stats_version": SCENE_GAUGE_STATS_VERSION,
        "scene_gauge_dim": SCENE_GAUGE_DIM,
        "gauge_table_sha256": table_sha256,
        "gauge_mean": mean.float(),
        "gauge_std": variance.sqrt().float().clamp_min(SCENE_GAUGE_STATS_STD_FLOOR),
        "gauge_count": counts,
    }


@torch.no_grad()
def compute_camera_role_stats(
    states: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    dggt_checkpoint_sha256: str,
    eps: float = CAMERA_STATS_STD_FLOOR**2,
) -> dict[str, torch.Tensor | str | int]:
    sums = {"anchor": torch.zeros(CAMERA_GENERATION_DIM, dtype=torch.float64),
            "delta": torch.zeros(CAMERA_GENERATION_DIM, dtype=torch.float64)}
    sums2 = {key: value.clone() for key, value in sums.items()}
    counts = {"anchor": 0, "delta": 0}
    for state, anchor_mask in states:
        if state.shape[-1] != CAMERA_GENERATION_DIM or anchor_mask.shape != state.shape[:-1]:
            raise ValueError(
                f"camera state/mask must have shapes [...,S,{CAMERA_GENERATION_DIM}] and [...,S]"
            )
        flat = state.detach().cpu().double().reshape(-1, CAMERA_GENERATION_DIM)
        mask = anchor_mask.detach().cpu().bool().reshape(-1)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("camera states contain non-finite values")
        for role, select in (("anchor", mask), ("delta", ~mask)):
            values = flat[select]
            counts[role] += int(values.shape[0])
            if values.numel():
                sums[role] += values.sum(0)
                sums2[role] += values.square().sum(0)
    if counts["anchor"] == 0 or counts["delta"] == 0:
        raise ValueError("camera stats require at least one anchor and one delta token")
    output: dict[str, torch.Tensor | str | int] = {
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_stats_version": CAMERA_STATS_VERSION,
        "camera_target_space": CAMERA_TARGET_SPACE,
        "camera_target_source": CAMERA_TARGET_SOURCE,
        "dggt_checkpoint_sha256": str(dggt_checkpoint_sha256),
        "camera_dim": CAMERA_GENERATION_DIM,
    }
    for role in ("anchor", "delta"):
        mean = sums[role] / counts[role]
        variance = (sums2[role] / counts[role] - mean.square()).clamp_min(float(eps))
        output[f"camera_{role}_mean"] = mean.float()
        output[f"camera_{role}_std"] = variance.sqrt().float().clamp_min(CAMERA_STATS_STD_FLOOR)
        output[f"camera_{role}_count"] = torch.tensor(counts[role], dtype=torch.long)
    return output


def save_feature_stats(stats: dict[str, torch.Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
