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


DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "scene_flow_pretrain_1024"
    / "feature_stats_pretrain_v2.pt"
)


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
    return mu, sigma.clamp_min(CAMERA_STATS_STD_FLOOR)


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
            value = value.clamp_min(CAMERA_STATS_STD_FLOOR)
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


def load_all_stats_into_buffers(
    module,
    path: str | Path,
    token_dim: int = 768,
    *,
    dggt_ckpt_path: str | Path | None = None,
    expected_dggt_sha256: str | None = None,
    expected_sequence_length: int | None = None,
    require_existing_match: bool = False,
) -> str:
    """Load latent plus mandatory DGGT-v3 camera stats and verify provenance.

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
    mu, sigma = _extract_stats(payload, token_dim=token_dim)
    if expected_dggt_sha256 is None:
        if dggt_ckpt_path is None:
            raise ValueError("DGGT checkpoint path/hash is required to validate camera statistics")
        expected_dggt_sha256 = checkpoint_sha256(dggt_ckpt_path)
    camera = _camera_stats_from_payload(payload, expected_dggt_sha256=expected_dggt_sha256)
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
        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                f"Feature statistics at {path} do not match the statistics stored in the loaded "
                f"SceneFlow checkpoint: {details}. Refusing to replace the checkpoint's latent/camera "
                "coordinate system. Use the exact feature-stats file used to train that checkpoint."
            )
    module.set_latent_stats(mu, sigma)
    module.set_camera_stats(*camera)
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
            raise ValueError("camera state/mask must have shapes [...,S,11] and [...,S]")
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
