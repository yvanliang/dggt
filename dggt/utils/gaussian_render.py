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


def composite_original_sky(
    rendered_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    sky_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep input-sky RGB exactly while leaving the edited render elsewhere.

    Using the complete GT image as a rasterizer background would also leak
    original non-sky content through transparent or uncovered edited
    Gaussians.  Formal editing instead uses the explicit image-space blend
    ``sky_mask * gt_rgb + (1 - sky_mask) * rendered_rgb``.
    """
    if rendered_rgb.ndim == 0 or rendered_rgb.shape[-1] != 3:
        raise ValueError(f"rendered_rgb must end in 3 channels, got {tuple(rendered_rgb.shape)}")
    if gt_rgb.shape != rendered_rgb.shape:
        raise ValueError(
            f"gt_rgb shape {tuple(gt_rgb.shape)} must match rendered_rgb {tuple(rendered_rgb.shape)}"
        )
    if sky_mask.ndim == 0 or sky_mask.shape[-1] != 1:
        raise ValueError(f"sky_mask must end in one channel, got {tuple(sky_mask.shape)}")
    if sky_mask.shape[:-1] != rendered_rgb.shape[:-1]:
        raise ValueError(
            f"sky_mask spatial shape {tuple(sky_mask.shape[:-1])} must match "
            f"rendered_rgb {tuple(rendered_rgb.shape[:-1])}"
        )
    weight = sky_mask.to(device=rendered_rgb.device, dtype=rendered_rgb.dtype).clamp(0.0, 1.0)
    target = gt_rgb.to(device=rendered_rgb.device, dtype=rendered_rgb.dtype)
    return weight * target + (1.0 - weight) * rendered_rgb
