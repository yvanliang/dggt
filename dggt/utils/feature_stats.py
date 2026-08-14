"""Minimal latent and scene-gauge statistics for layout-v2 SceneFlow.

The clean-cut model has no generated-camera or appearance-placement state.
Consequently a feature-stat artifact contains only tokenizer-latent moments
and the three scene-gauge moments.  Loading performs numerical/shape checks;
it deliberately does not calculate or compare content digests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from dggt.utils.scene_gauge import SCENE_GAUGE_DIM


DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "scene_flow_pretrain_1024"
    / "feature_stats_pretrain_v5.pt"
)

FEATURE_STATS_SCHEMA = "scene_flow_layout_v2_feature_stats"
FEATURE_STATS_SCHEMA_VERSION = "1.0.0"


def _flat_finite(value: object, *, name: str, channels: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    if tensor.numel() != int(channels):
        raise ValueError(
            f"{name} must contain {int(channels)} channels, got {tensor.numel()}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def _extract_latent_stats(
    payload: dict,
    *,
    token_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for mean_key, std_key in (
        ("mu_z", "sigma_z"),
        ("mean_z", "std_z"),
        ("mean", "std"),
        ("mu", "sigma"),
    ):
        if mean_key in payload and std_key in payload:
            mean = _flat_finite(
                payload[mean_key], name=mean_key, channels=int(token_dim)
            )
            std = _flat_finite(
                payload[std_key], name=std_key, channels=int(token_dim)
            )
            break
    else:
        raise KeyError(
            "feature stats require latent mean/std fields (mu_z/sigma_z preferred)"
        )
    if bool((std <= 0.0).any()):
        raise ValueError("latent standard deviation must be positive in every channel")
    return mean, std


def _extract_gauge_stats(payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if "gauge_mean" not in payload or "gauge_std" not in payload:
        raise KeyError("feature stats require gauge_mean and gauge_std")
    mean = _flat_finite(
        payload["gauge_mean"], name="gauge_mean", channels=SCENE_GAUGE_DIM
    )
    std = _flat_finite(
        payload["gauge_std"], name="gauge_std", channels=SCENE_GAUGE_DIM
    )
    if bool((std <= 0.0).any()):
        raise ValueError("gauge standard deviation must be positive in every channel")
    return mean, std


def load_feature_stats(
    path: str | Path,
    token_dim: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"feature stats at {path} must be a dict, got {type(payload).__name__}"
        )
    return _extract_latent_stats(payload, token_dim=int(token_dim))


def load_into_buffers(
    module,
    path: str | Path | None,
    token_dim: int = 1024,
) -> None:
    """Load only latent moments into a generic normalizer module."""

    if path is None:
        return
    mean, std = load_feature_stats(path, token_dim=int(token_dim))
    if hasattr(module, "set_latent_stats"):
        module.set_latent_stats(mean, std)
        return
    if not hasattr(module, "mu_z") or not hasattr(module, "sigma_z"):
        raise AttributeError("module must expose set_latent_stats() or mu_z/sigma_z")
    module.mu_z.copy_(mean.to(device=module.mu_z.device, dtype=module.mu_z.dtype))
    module.sigma_z.copy_(std.to(device=module.sigma_z.device, dtype=module.sigma_z.dtype))


def load_all_stats_into_buffers(
    module,
    path: str | Path | None,
    token_dim: int = 1024,
) -> None:
    """Load the only four statistics used by layout-v2 SceneFlow."""

    if path is None:
        return
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"feature stats at {path} must be a dict, got {type(payload).__name__}"
        )
    latent_mean, latent_std = _extract_latent_stats(
        payload, token_dim=int(token_dim)
    )
    gauge_mean, gauge_std = _extract_gauge_stats(payload)
    if not hasattr(module, "set_latent_stats") or not hasattr(
        module, "set_gauge_stats"
    ):
        raise AttributeError(
            "layout-v2 SceneFlow must expose set_latent_stats() and set_gauge_stats()"
        )
    module.set_latent_stats(latent_mean, latent_std)
    module.set_gauge_stats(gauge_mean, gauge_std)


def validate_production_stats_coverage(payload: dict) -> None:
    """Reject partial statistics without relying on artifact identity values."""

    if payload.get("stats_status") != "complete":
        raise ValueError("production feature stats require stats_status='complete'")
    coverage = payload.get("stats_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("feature stats are missing stats_coverage")
    expected = int(coverage.get("expected_batches", -1))
    processed = int(coverage.get("processed_batches", -2))
    if expected <= 0 or processed != expected or coverage.get("full_dataset_pass") is not True:
        raise ValueError(
            "feature stats do not prove a complete dataset pass: "
            f"{coverage!r}"
        )


def compute_per_channel_stats(
    batches: Iterable[torch.Tensor],
    token_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute stable per-channel moments on CPU float64."""

    channels = int(token_dim)
    total = 0
    sum_value = torch.zeros(channels, dtype=torch.float64)
    sum_square = torch.zeros(channels, dtype=torch.float64)
    for value in batches:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0 or int(tensor.shape[-1]) != channels:
            raise ValueError(
                f"expected last dimension {channels}, got {tuple(tensor.shape)}"
            )
        flat = tensor.detach().cpu().double().reshape(-1, channels)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("feature batch contains NaN or Inf")
        total += int(flat.shape[0])
        sum_value += flat.sum(dim=0)
        sum_square += flat.square().sum(dim=0)
    if total <= 0:
        raise ValueError("cannot compute statistics from an empty iterator")
    mean = sum_value / float(total)
    variance = (sum_square / float(total) - mean.square()).clamp_min(1.0e-6)
    return mean.float(), variance.sqrt().float(), total


def compute_scene_gauge_stats(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor | str | int]:
    """Compute masked moments for the three physical gauge channels."""

    sums = torch.zeros(SCENE_GAUGE_DIM, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    counts = torch.zeros(SCENE_GAUGE_DIM, dtype=torch.long)
    for values, valid in batches:
        value = torch.as_tensor(values).detach().cpu().double()
        mask = torch.as_tensor(valid).detach().cpu().bool()
        if value.shape != mask.shape or int(value.shape[-1]) != SCENE_GAUGE_DIM:
            raise ValueError("scene gauge values/masks must match with last dimension 3")
        finite = torch.isfinite(value)
        if bool((mask & ~finite).any()):
            raise ValueError("valid scene-gauge entries contain NaN or Inf")
        flat_value = value.reshape(-1, SCENE_GAUGE_DIM)
        flat_mask = mask.reshape(-1, SCENE_GAUGE_DIM)
        safe = torch.where(flat_mask, flat_value, torch.zeros_like(flat_value))
        sums += safe.sum(dim=0)
        squares += safe.square().sum(dim=0)
        counts += flat_mask.sum(dim=0)
    if bool((counts <= 0).any()):
        raise ValueError("every scene-gauge channel needs at least one valid value")
    mean = sums / counts.double()
    variance = (squares / counts.double() - mean.square()).clamp_min(1.0e-6)
    return {
        "gauge_mean": mean.float(),
        "gauge_std": variance.sqrt().float(),
        "gauge_count": counts,
    }


def save_feature_stats(stats: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(stats, temporary)
    temporary.replace(output)


__all__ = [
    "DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH",
    "FEATURE_STATS_SCHEMA",
    "FEATURE_STATS_SCHEMA_VERSION",
    "compute_per_channel_stats",
    "compute_scene_gauge_stats",
    "load_all_stats_into_buffers",
    "load_feature_stats",
    "load_into_buffers",
    "save_feature_stats",
    "validate_production_stats_coverage",
]
