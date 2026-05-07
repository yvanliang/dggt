"""Latent feature statistics for SceneFlow training."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


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
    return mu, sigma.clamp_min(1e-6)


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


def save_feature_stats(stats: dict[str, torch.Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
