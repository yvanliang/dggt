from __future__ import annotations

import torch


def composite_gsplat_rgb(
    rendered_rgb: torch.Tensor,
    alpha: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """Composite gsplat's premultiplied RGB over a spatial background.

    With ``backgrounds=None``, gsplat returns the accumulated foreground color
    ``sum_i(T_i * alpha_i * color_i)``.  It is already premultiplied by the
    per-Gaussian alpha weights.  ``alpha`` is the accumulated opacity, so the
    remaining background contribution is only ``(1 - alpha) * background``.

    Multiplying ``rendered_rgb`` by ``alpha`` again would apply opacity twice
    and incorrectly darken partially covered pixels.
    """
    if rendered_rgb.ndim == 0 or rendered_rgb.shape[-1] != 3:
        raise ValueError(f"rendered_rgb must end in 3 channels, got {tuple(rendered_rgb.shape)}")
    if background.ndim == 0 or background.shape[-1] != 3:
        raise ValueError(f"background must end in 3 channels, got {tuple(background.shape)}")
    if alpha.ndim == 0 or alpha.shape[-1] != 1:
        raise ValueError(f"alpha must end in one channel, got {tuple(alpha.shape)}")
    return rendered_rgb + (1.0 - alpha) * background
