from __future__ import annotations

import torch

from dggt.utils.gaussian_render import composite_gsplat_rgb, composite_original_sky


def test_composite_gsplat_rgb_does_not_apply_alpha_twice() -> None:
    alpha = torch.tensor([[[[0.5]]]])
    # This is already premultiplied: straight red [1, 0, 0] at alpha 0.5.
    rendered = torch.tensor([[[[0.5, 0.0, 0.0]]]])
    background = torch.tensor([[[[0.0, 0.0, 1.0]]]])

    actual = composite_gsplat_rgb(rendered, alpha, background)

    assert torch.allclose(actual, torch.tensor([[[[0.5, 0.0, 0.5]]]]))
    assert not torch.allclose(
        actual,
        alpha * rendered + (1.0 - alpha) * background,
    )


def test_composite_gsplat_rgb_supports_spatial_backgrounds_and_gradients() -> None:
    rendered = torch.rand(1, 3, 4, 3, requires_grad=True)
    alpha = torch.rand(1, 3, 4, 1)
    background = torch.rand(1, 3, 4, 3)

    actual = composite_gsplat_rgb(rendered, alpha, background)
    expected = rendered + (1.0 - alpha) * background

    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert torch.equal(rendered.grad, torch.ones_like(rendered))


def test_composite_original_sky_preserves_gt_only_inside_mask() -> None:
    rendered = torch.tensor([[[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]]])
    gt_rgb = torch.tensor([[[[0.9, 0.8, 0.7], [0.8, 0.7, 0.6], [0.7, 0.6, 0.5]]]])
    sky_mask = torch.tensor([[[[1.0], [0.0], [0.25]]]])

    actual = composite_original_sky(rendered, gt_rgb, sky_mask)

    torch.testing.assert_close(actual[..., 0, :], gt_rgb[..., 0, :])
    torch.testing.assert_close(actual[..., 1, :], rendered[..., 1, :])
    torch.testing.assert_close(
        actual[..., 2, :],
        0.25 * gt_rgb[..., 2, :] + 0.75 * rendered[..., 2, :],
    )
