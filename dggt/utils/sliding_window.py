"""Sliding-window inference utilities for SceneFlow."""
from __future__ import annotations

import torch


def cosine_window(window_size: int, *, device=None, dtype=None) -> torch.Tensor:
    if int(window_size) <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if int(window_size) == 1:
        return torch.ones(1, device=device, dtype=dtype or torch.float32)
    x = torch.linspace(-1.0, 1.0, int(window_size), device=device, dtype=dtype or torch.float32)
    w = torch.cos(0.5 * torch.pi * x).square()
    return w.clamp_min(1e-3)


def window_starts(seq_len: int, window_size: int, overlap: int) -> list[int]:
    seq_len = int(seq_len)
    window_size = int(window_size)
    overlap = int(overlap)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    if seq_len <= window_size:
        return [0]
    stride = window_size - overlap
    starts = list(range(0, seq_len - window_size + 1, stride))
    last = seq_len - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def sliding_window_inference(
    model,
    z_splat_full: torch.Tensor,
    scaffold_full: torch.Tensor,
    M_preserve_full: torch.Tensor,
    M_source_full: torch.Tensor,
    M_dest_full: torch.Tensor,
    F_asset_tokens: torch.Tensor,
    *,
    z_clean_full: torch.Tensor | None = None,
    scheduler=None,
    window_size: int = 8,
    overlap: int = 3,
    num_steps: int = 15,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    if scheduler is None:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=6.0,
            invert_sigmas=True,
        )
    scheduler.set_timesteps(num_inference_steps=num_steps, device=z_splat_full.device)

    z = torch.empty_like(z_splat_full)
    z.normal_(generator=generator)

    batch_size, seq_len = z_splat_full.shape[:2]
    starts = window_starts(seq_len, window_size, overlap)
    weights = cosine_window(window_size, device=z.device, dtype=z.dtype)

    if not bool(getattr(scheduler.config, "invert_sigmas", False)):
        raise ValueError(
            "sliding_window_inference requires a clean-progress scheduler "
            "(invert_sigmas=True). Pass FlowMatchEulerDiscreteScheduler(invert_sigmas=True)."
        )

    for timestep in scheduler.timesteps:
        sched_sigma = (timestep / scheduler.config.num_train_timesteps).to(device=z.device)
        model_sigma = sched_sigma.expand(batch_size)

        v_full = torch.zeros_like(z)
        weight_full = torch.zeros((1, seq_len, 1, 1), dtype=z.dtype, device=z.device)
        for start in starts:
            end = min(start + window_size, seq_len)
            actual = end - start
            w = weights[:actual].view(1, actual, 1, 1)
            v_window = model(
                z[:, start:end],
                model_sigma,
                z_splat_full[:, start:end],
                scaffold_full[:, start:end],
                M_preserve_full[:, start:end],
                M_source_full[:, start:end],
                M_dest_full[:, start:end],
                F_asset_tokens,
            )
            v_full[:, start:end] += v_window * w
            weight_full[:, start:end] += w
        v_blend = v_full / weight_full.clamp_min(1e-6)
        z = scheduler.step(
            model_output=v_blend,
            timestep=timestep,
            sample=z,
            return_dict=False,
        )[0]

    if z_clean_full is not None:
        z = M_preserve_full * z_clean_full + (1.0 - M_preserve_full) * z
    return z
