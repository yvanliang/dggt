"""Losses and rectified-flow target helpers for SceneFlow training."""
from __future__ import annotations

import math
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
    z_cond: torch.Tensor | None = None
    z_clean_target: torch.Tensor | None = None
    M_edit: torch.Tensor | None = None
    t_eps: float = 0.05


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


def masked_flow_edit_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    M_edit: torch.Tensor,
    sd3_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = M_edit.to(device=v_pred.device, dtype=v_pred.dtype).clamp(0.0, 1.0)
    if sd3_weights is not None:
        weight = weight * sd3_weights.to(device=v_pred.device, dtype=v_pred.dtype)
    diff = (v_pred - v_gt).square().mean(dim=-1, keepdim=True)
    return masked_mean(diff, weight)


def preserve_loss(
    v_pred: torch.Tensor,
    eps: torch.Tensor,
    z_clean: torch.Tensor,
    M_preserve: torch.Tensor,
    z_pred: torch.Tensor | None = None,
    z_preserve_target: torch.Tensor | None = None,
) -> torch.Tensor:
    del v_pred, eps
    if z_pred is None:
        raise ValueError(
            "preserve_loss requires an explicit clean prediction. RAE velocity cannot be "
            "reconstructed from eps-v_pred when sigma is below t_eps or the flow path is masked."
        )
    z_hat = z_pred
    target = z_clean if z_preserve_target is None else z_preserve_target
    diff = (z_hat - target).square().mean(dim=-1, keepdim=True)
    return masked_mean(diff, M_preserve)


def boundary_loss(
    z_pred: torch.Tensor | None,
    z_clean: torch.Tensor,
    boundary_mask: torch.Tensor | None,
) -> torch.Tensor:
    if z_pred is None or boundary_mask is None:
        return zero_loss_like(z_clean)
    diff = (z_pred - z_clean).square().mean(dim=-1, keepdim=True)
    return masked_mean(diff, boundary_mask)


def repa_loss(mid_repa: torch.Tensor | None, z_clean: torch.Tensor | None) -> torch.Tensor:
    if mid_repa is None or z_clean is None:
        device = mid_repa.device if mid_repa is not None else "cpu"
        return torch.zeros((), device=device)
    return F.mse_loss(mid_repa.float(), z_clean.detach().float())


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
    repa_target: torch.Tensor | None = None,
    z_pred: torch.Tensor | None = None,
    z_preserve_target: torch.Tensor | None = None,
    M_edit: torch.Tensor | None = None,
    M_preserve_loss: torch.Tensor | None = None,
    boundary_mask: torch.Tensor | None = None,
    v_base_pred: torch.Tensor | None = None,
    base_model_coeff: float = 0.0,
    lambda_flow: float = 1.0,
    lambda_preserve: float = 1.0,
    lambda_boundary: float = 0.0,
    lambda_repa: float = 0.0,
    lambda_identity: float = 0.0,
    identity_batch: bool | torch.Tensor = False,
    preserve_floor: float = 0.2,
    defer_log_values: bool = False,
    collect_logs: bool = True,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    z_clean = getattr(bundle, "z_clean_n", None)
    if z_clean is None:
        z_clean = bundle.z_clean

    if M_edit is None:
        M_edit = getattr(bundle, "M_edit", None)
    if M_edit is None:
        loss_flow = flow_matching_loss(
            v_pred,
            v_gt,
            bundle.M_preserve,
            sd3_weights=sd3_weights,
            preserve_floor=preserve_floor,
        )
    else:
        loss_flow = masked_flow_edit_loss(v_pred, v_gt, M_edit, sd3_weights=sd3_weights)
    if M_preserve_loss is None:
        M_preserve_loss = bundle.M_preserve
    loss_preserve = (
        preserve_loss(
            v_pred,
            eps,
            z_clean,
            M_preserve_loss,
            z_pred=z_pred,
            z_preserve_target=z_preserve_target,
        )
        if float(lambda_preserve) != 0.0
        else zero_loss_like(v_pred)
    )
    loss_boundary = (
        boundary_loss(z_pred, z_clean, boundary_mask)
        if float(lambda_boundary) != 0.0
        else zero_loss_like(v_pred)
    )
    if repa_target is None:
        repa_target = z_clean
    loss_repa = repa_loss(mid_repa, repa_target) if float(lambda_repa) != 0.0 else zero_loss_like(v_pred)
    identity_mask = None
    if torch.is_tensor(identity_batch):
        identity_mask = identity_batch.to(device=v_pred.device, dtype=torch.bool).reshape(-1)
        if int(identity_mask.numel()) != int(v_pred.shape[0]):
            raise ValueError(
                f"identity mask must have B={v_pred.shape[0]} entries, got {identity_mask.numel()}"
            )
        has_identity = bool(identity_mask.any().item())
    else:
        has_identity = bool(identity_batch)

    if has_identity and float(lambda_identity) != 0.0:
        if z_pred is not None:
            identity_target = z_clean if z_preserve_target is None else z_preserve_target
            identity_error = (z_pred - identity_target).square().flatten(1).mean(dim=1)
        else:
            identity_error = (v_pred - v_gt).square().flatten(1).mean(dim=1)
        if identity_mask is None:
            loss_identity = identity_error.mean()
        else:
            weight = identity_mask.to(dtype=identity_error.dtype)
            loss_identity = (identity_error * weight).sum() / weight.sum().clamp_min(1.0)
    else:
        loss_identity = zero_loss_like(v_pred)
    if v_base_pred is not None and float(base_model_coeff) != 0.0:
        if M_edit is not None:
            loss_base = masked_flow_edit_loss(v_base_pred, v_gt, M_edit, sd3_weights=sd3_weights)
        else:
            loss_base = flow_matching_loss(
                v_base_pred,
                v_gt,
                bundle.M_preserve,
                sd3_weights=sd3_weights,
                preserve_floor=preserve_floor,
            )
    else:
        loss_base = zero_loss_like(v_pred)

    total = (
        float(lambda_flow) * loss_flow
        + float(lambda_preserve) * loss_preserve
        + float(lambda_boundary) * loss_boundary
        + float(lambda_repa) * loss_repa
        + float(lambda_identity) * loss_identity
        + float(base_model_coeff) * loss_base
    )
    def log_value(value: torch.Tensor) -> float | torch.Tensor:
        detached = value.detach().float().reshape(())
        return detached if defer_log_values else float(detached.item())

    logs = {
        "loss": log_value(total),
        "loss_flow": log_value(loss_flow),
        "loss_flow_edit": log_value(loss_flow),
        "loss_preserve": log_value(loss_preserve),
        "loss_boundary": log_value(loss_boundary),
        "loss_repa": log_value(loss_repa),
        "loss_identity": log_value(loss_identity),
        "loss_base": log_value(loss_base),
    } if collect_logs else {}
    return total, logs


def rae_time_shift_dim(z_clean: torch.Tensor) -> int:
    """RAEv2 uses the per-sample latent dimensionality for time shift."""
    return int(z_clean[0].numel())


def rae_time_shift(z_clean: torch.Tensor, base: int = 4096) -> float:
    return float((rae_time_shift_dim(z_clean) / float(base)) ** 0.5)


def sample_rae_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    time_shift: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    t = torch.randn(
        batch_size,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    t = (t * float(logit_std) + float(logit_mean)).sigmoid()
    shift = float(time_shift)
    t = shift * t / (1.0 + (shift - 1.0) * t)
    return t.to(dtype=dtype)


def sample_waver_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode_scale: float = 1.29,
    time_shift: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    u = torch.rand(
        batch_size,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    t = 1.0 - u - float(mode_scale) * (torch.cos(torch.pi * 0.5 * u).square() - 1.0 + u)
    t = t.clamp(0.0, 1.0)
    shift = float(time_shift)
    t = shift * t / (1.0 + (shift - 1.0) * t)
    return t.to(dtype=dtype)


def rae_t_grid(
    *,
    num_steps: int,
    time_shift: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    t_eps: float = 0.05,
) -> torch.Tensor:
    steps = int(num_steps)
    shift = float(time_shift)
    floor = float(t_eps)
    if steps <= 0:
        raise ValueError(f"RAE sampling requires num_steps > 0, got {steps}")
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"RAE sampling requires time_shift > 0, got {shift}")
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError(f"RAE sampling requires 0 < t_eps <= 1, got {floor}")

    t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    t = shift * t / (1.0 + (shift - 1.0) * t)
    last_sigma = float(t[-2].item())
    if last_sigma + 1e-7 < floor:
        max_steps = max(1, math.floor(shift / floor - shift + 1.0 + 1e-9))
        raise ValueError(
            "RAE sampling grid enters the clamped pseudo-velocity region: "
            f"last nonzero sigma={last_sigma:.8g} < t_eps={floor:.8g}. "
            f"Use num_steps <= {max_steps} for time_shift={shift:.8g}, or change the "
            "checkpointed flow schedule instead of silently producing an inexact clean endpoint."
        )
    return t.to(dtype=dtype)


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
    time_shift: float | None = None,
    t_eps: float = 0.05,
) -> FlowTarget:
    batch_size = int(z_clean.shape[0])
    device = z_clean.device
    scheme = str(weighting_scheme).lower().replace("-", "_")
    if scheme not in ("logit_normal", "waver"):
        raise ValueError(
            "RAEv2 flow target supports only logit-normal or waver timestep sampling; "
            f"got weighting_scheme={weighting_scheme!r}"
        )
    if str(loss_weighting_scheme) not in ("none", "", "None"):
        raise ValueError(
            "RAEv2 flow target does not apply SD3 loss weighting; "
            f"got loss_weighting_scheme={loss_weighting_scheme!r}"
        )

    if time_shift is None:
        time_shift = rae_time_shift(z_clean)
    if scheme == "waver":
        sigmas = sample_waver_timesteps(
            batch_size,
            device=device,
            dtype=z_clean.dtype,
            mode_scale=1.29 if mode_scale is None else float(mode_scale),
            time_shift=float(time_shift),
            generator=generator,
        )
    else:
        sigmas = sample_rae_timesteps(
            batch_size,
            device=device,
            dtype=z_clean.dtype,
            logit_mean=logit_mean,
            logit_std=logit_std,
            time_shift=float(time_shift),
            generator=generator,
        )
    sigmas4 = sigmas.view(batch_size, 1, 1, 1)
    eps = torch.randn(
        z_clean.shape,
        device=device,
        dtype=z_clean.dtype,
        generator=generator,
    )

    # Noise-progress RF convention: sigma=1 is Gaussian noise, sigma=0 is clean.
    z_t = (1.0 - sigmas4) * z_clean + sigmas4 * eps
    v_gt = (z_t - z_clean) / sigmas4.clamp_min(float(t_eps))
    weights = torch.ones((batch_size, 1, 1, 1), device=device, dtype=z_clean.dtype)
    return FlowTarget(
        sigmas=sigmas,
        sigmas4=sigmas4,
        z_t=z_t,
        v_gt=v_gt,
        eps=eps,
        weights=weights,
        t_eps=float(t_eps),
    )


def build_masked_rectified_flow_target(
    scheduler,
    z_clean: torch.Tensor,
    z_cond: torch.Tensor,
    M_edit: torch.Tensor,
    *,
    weighting_scheme: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float | None = None,
    loss_weighting_scheme: str = "none",
    generator: torch.Generator | None = None,
    time_shift: float | None = None,
    t_eps: float = 0.05,
) -> FlowTarget:
    if z_cond.shape != z_clean.shape:
        raise ValueError(f"z_cond shape {tuple(z_cond.shape)} != z_clean shape {tuple(z_clean.shape)}")
    if M_edit.shape != z_clean.shape[:-1] + (1,):
        raise ValueError(f"M_edit must be [B,S,P,1], got {tuple(M_edit.shape)}")
    base = build_rectified_flow_target(
        scheduler,
        z_clean,
        weighting_scheme=weighting_scheme,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
        loss_weighting_scheme=loss_weighting_scheme,
        generator=generator,
        time_shift=time_shift,
        t_eps=t_eps,
    )
    edit = M_edit.to(device=z_clean.device, dtype=z_clean.dtype).clamp(0.0, 1.0)
    keep = 1.0 - edit
    # Define one explicit conditional probability path.  Both endpoints keep
    # the condition latent outside the edit domain:
    #   clean_target = edit * clean + keep * condition
    #   noise_target = edit * noise + keep * condition
    # Hence z_t = edit * base.z_t + keep * condition and its RAE-style target
    # is exactly edit * base.v_gt.  Reusing the unscaled base velocity here is
    # only correct when edit is binary and the loss is restricted to edit=1.
    z_clean_target = edit * z_clean + keep * z_cond
    z_t = edit * base.z_t + keep * z_cond
    return FlowTarget(
        sigmas=base.sigmas,
        sigmas4=base.sigmas4,
        z_t=z_t,
        v_gt=edit * base.v_gt,
        eps=base.eps,
        weights=base.weights,
        z_cond=z_cond,
        z_clean_target=z_clean_target,
        M_edit=edit,
        t_eps=float(t_eps),
    )


def build_hard_edit_domain(
    M_source: torch.Tensor,
    M_dest: torch.Tensor,
    patch_grid: tuple[int, int] | list[int],
    *,
    threshold: float = 1e-4,
    dilation_radius: int = 1,
) -> torch.Tensor:
    """Build the binary state/noise domain from soft edit semantics.

    ``M_source`` and ``M_dest`` remain continuous model conditions.  This
    binary domain is reserved for the flow path, effective timestep, losses,
    and the idempotent sampler projection.
    """
    if M_source.shape != M_dest.shape or M_source.ndim != 4 or int(M_source.shape[-1]) != 1:
        raise ValueError(
            "M_source and M_dest must share shape [B,S,P,1], got "
            f"{tuple(M_source.shape)} and {tuple(M_dest.shape)}"
        )
    if float(threshold) < 0.0 or float(threshold) >= 1.0:
        raise ValueError(f"threshold must be in [0,1), got {threshold}")
    b, s, p, _ = M_source.shape
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    if gh * gw != int(p):
        raise ValueError(f"patch_grid={tuple(patch_grid)} incompatible with P={p}")
    soft_edit = (M_source.float() + M_dest.float()).clamp(0.0, 1.0)
    support = soft_edit.gt(float(threshold))
    radius = int(dilation_radius)
    if radius < 0:
        raise ValueError(f"dilation_radius must be non-negative, got {dilation_radius}")
    if radius > 0:
        grid = support.reshape(b * s, gh, gw, 1).permute(0, 3, 1, 2).float()
        kernel = 2 * radius + 1
        support = F.max_pool2d(grid, kernel_size=kernel, stride=1, padding=radius).gt(0.0)
        support = support.permute(0, 2, 3, 1).reshape(b, s, p, 1)
    return support.to(device=M_source.device, dtype=M_source.dtype)


def project_masked_flow_state(
    generated: torch.Tensor,
    condition: torch.Tensor,
    edit_domain: torch.Tensor,
) -> torch.Tensor:
    """Project onto a *binary* edit domain; the projection is idempotent."""
    if generated.shape != condition.shape:
        raise ValueError(f"generated shape {tuple(generated.shape)} != condition shape {tuple(condition.shape)}")
    if edit_domain.shape != generated.shape[:-1] + (1,):
        raise ValueError(f"edit_domain must be [B,S,P,1], got {tuple(edit_domain.shape)}")
    domain = edit_domain.to(device=generated.device)
    if domain.dtype != torch.bool:
        rounded = domain.round()
        if not bool(torch.allclose(domain.float(), rounded.float(), atol=1e-6, rtol=0.0)):
            raise ValueError("edit_domain must be binary; soft masks are not valid ODE projections")
        domain = rounded.to(dtype=torch.bool)
    return torch.where(domain, generated, condition.to(device=generated.device, dtype=generated.dtype))


def masked_flow_euler_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    step_size: torch.Tensor | float,
    condition: torch.Tensor,
    edit_domain: torch.Tensor,
) -> torch.Tensor:
    """Euler update in the editable subspace followed by a hard projection."""
    if velocity.shape != state.shape:
        raise ValueError(f"velocity shape {tuple(velocity.shape)} != state shape {tuple(state.shape)}")
    domain = edit_domain.to(device=state.device, dtype=state.dtype)
    updated = state - torch.as_tensor(step_size, device=state.device, dtype=state.dtype) * domain * velocity
    return project_masked_flow_state(updated, condition, edit_domain)


def boundary_mask_from_edit_mask(
    M_edit: torch.Tensor,
    patch_grid: tuple[int, int] | list[int],
    radius: int = 1,
) -> torch.Tensor:
    if M_edit.ndim != 4 or int(M_edit.shape[-1]) != 1:
        raise ValueError(f"M_edit must be [B,S,P,1], got {tuple(M_edit.shape)}")
    b, s, p, _ = M_edit.shape
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    if gh * gw != int(p):
        raise ValueError(f"patch_grid={tuple(patch_grid)} incompatible with P={p}")
    if int(radius) <= 0:
        return torch.zeros_like(M_edit)
    grid = M_edit.to(dtype=torch.float32).reshape(b * s, gh, gw, 1).permute(0, 3, 1, 2)
    k = 2 * int(radius) + 1
    dilated = F.max_pool2d(grid, kernel_size=k, stride=1, padding=int(radius))
    ring = (dilated - grid).clamp(0.0, 1.0)
    return ring.permute(0, 2, 3, 1).reshape(b, s, p, 1).to(device=M_edit.device, dtype=M_edit.dtype)
