"""Losses and rectified-flow target helpers for SceneFlow training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowTarget:
    sigmas: torch.Tensor
    sigmas4: torch.Tensor
    z_t: torch.Tensor
    v_gt: torch.Tensor
    eps: torch.Tensor
    weights: torch.Tensor


def masked_mean(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.sum().clamp_min(eps)


def flow_matching_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    M_preserve: torch.Tensor,
    sd3_weights: torch.Tensor | None = None,
    preserve_floor: float = 0.2,
) -> torch.Tensor:
    edit_weight = float(preserve_floor) + (1.0 - float(preserve_floor)) * (1.0 - M_preserve)
    if sd3_weights is not None:
        edit_weight = edit_weight * sd3_weights.to(device=edit_weight.device, dtype=edit_weight.dtype)
    diff = (v_pred - v_gt).square().mean(dim=-1, keepdim=True)
    return masked_mean(diff, edit_weight)


def preserve_loss(
    v_pred: torch.Tensor,
    eps: torch.Tensor,
    z_clean: torch.Tensor,
    M_preserve: torch.Tensor,
) -> torch.Tensor:
    z_hat = eps + v_pred
    diff = (z_hat - z_clean).square().mean(dim=-1, keepdim=True)
    return masked_mean(diff, M_preserve)


def repa_loss(mid_repa: torch.Tensor | None, z_clean: torch.Tensor | None) -> torch.Tensor:
    if mid_repa is None or z_clean is None:
        device = mid_repa.device if mid_repa is not None else "cpu"
        return torch.zeros((), device=device)
    mid = F.normalize(mid_repa.float(), dim=-1)
    target = F.normalize(z_clean.detach().float(), dim=-1)
    return (1.0 - (mid * target).sum(dim=-1)).mean()


def identity_loss(v_pred: torch.Tensor, v_gt: torch.Tensor) -> torch.Tensor:
    return (v_pred - v_gt).square().mean()


def zero_loss_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def compute_total_loss(
    *,
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    eps: torch.Tensor,
    bundle,
    sd3_weights: torch.Tensor | None = None,
    mid_repa: torch.Tensor | None = None,
    lambda_flow: float = 1.0,
    lambda_preserve: float = 1.0,
    lambda_repa: float = 0.0,
    lambda_identity: float = 0.0,
    identity_batch: bool = False,
    preserve_floor: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    z_clean = getattr(bundle, "z_clean_n", None)
    if z_clean is None:
        z_clean = bundle.z_clean

    loss_flow = flow_matching_loss(
        v_pred,
        v_gt,
        bundle.M_preserve,
        sd3_weights=sd3_weights,
        preserve_floor=preserve_floor,
    )
    loss_preserve = preserve_loss(v_pred, eps, z_clean, bundle.M_preserve)
    loss_repa = repa_loss(mid_repa, z_clean) if float(lambda_repa) != 0.0 else zero_loss_like(v_pred)
    loss_identity = identity_loss(v_pred, v_gt) if identity_batch and float(lambda_identity) != 0.0 else zero_loss_like(v_pred)

    total = (
        float(lambda_flow) * loss_flow
        + float(lambda_preserve) * loss_preserve
        + float(lambda_repa) * loss_repa
        + float(lambda_identity) * loss_identity
    )
    logs = {
        "loss": float(total.detach().item()),
        "loss_flow": float(loss_flow.detach().item()),
        "loss_preserve": float(loss_preserve.detach().item()),
        "loss_repa": float(loss_repa.detach().item()),
        "loss_identity": float(loss_identity.detach().item()),
    }
    return total, logs


def build_rectified_flow_target(
    scheduler,
    z_clean: torch.Tensor,
    *,
    weighting_scheme: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float | None = None,
    loss_weighting_scheme: str = "none",
    generator: torch.Generator | None = None,
) -> FlowTarget:
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )

    batch_size = int(z_clean.shape[0])
    device = z_clean.device
    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
        device=device,
        generator=generator,
    )
    num_train_timesteps = int(scheduler.config.num_train_timesteps)
    indices = (u * num_train_timesteps).long().clamp_(0, num_train_timesteps - 1)
    # diffusers' FlowMatchEulerDiscreteScheduler stores sigmas in the SD3/Wan
    # noise-progress convention (sigmas[0]=1 means "full noise", sigmas[-1]≈0
    # means "clean"). Our rectified-flow target below is written in the
    # clean-progress convention (σ=0 is noise, σ=1 is clean), so we flip the
    # lookup. Without this flip, `shift>1` — which Wan uses to bias sampling
    # toward the noise side — ends up biasing our training toward the clean
    # side instead, leaving the noise regime essentially untrained.
    sched_sigmas = scheduler.sigmas.to(device=device, dtype=z_clean.dtype)
    sigmas = 1.0 - sched_sigmas[indices]
    sigmas4 = sigmas.view(batch_size, 1, 1, 1)
    eps = torch.randn_like(z_clean)

    # Clean-progress rectified flow from docs/implement_scene_flow_plan.md:
    # σ=0 is noise, σ=1 is z_clean, v = z_clean - eps.
    z_t = (1.0 - sigmas4) * eps + sigmas4 * z_clean
    v_gt = z_clean - eps
    weights = compute_loss_weighting_for_sd3(
        weighting_scheme=loss_weighting_scheme,
        sigmas=sigmas4.float(),
    ).to(device=device, dtype=z_clean.dtype)
    return FlowTarget(sigmas=sigmas, sigmas4=sigmas4, z_t=z_t, v_gt=v_gt, eps=eps, weights=weights)
