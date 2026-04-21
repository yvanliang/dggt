"""Smoke tests for `dggt.models.soft_mask.SoftMaskBuilder`.

Requires a CUDA device (gsplat rasterization).
"""
from __future__ import annotations

import pytest
import torch

from dggt.models.soft_mask import SoftMaskBuilder


CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="gsplat requires CUDA")


def _id_camera(H: int, W: int, device: torch.device) -> dict[str, torch.Tensor]:
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    fx = fy = float(W)
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]], device=device)
    return {
        "viewmats": viewmat.view(1, 1, 4, 4),
        "Ks": K.view(1, 1, 3, 3),
    }


def _gauss(x: float, y: float, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor([[x, y, 3.0]], device=device),
        "quats": torch.tensor([[1.0, 0, 0, 0]], device=device),
        "scales": torch.tensor([[0.3, 0.3, 0.3]], device=device),
        "opacities": torch.tensor([0.99], device=device),
    }


def _empty_gauss(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "means": torch.zeros((0, 3), device=device),
        "quats": torch.zeros((0, 4), device=device),
        "scales": torch.zeros((0, 3), device=device),
        "opacities": torch.zeros((0,), device=device),
    }


def test_only_kept_gaussians_yield_preserve_mask():
    device = torch.device("cuda")
    builder = SoftMaskBuilder()
    cameras = _id_camera(74, 74, device)

    K_map, D_map, I_map, _ = builder.render_coverage(
        G_kept=[_gauss(0.0, 0.0, device)],
        G_deleted=[_empty_gauss(device)],
        G_asset_dggt_dict=[{}],
        cameras_dggt=cameras,
        H=74, W=74,
    )
    assert K_map.shape == (1, 1, 74, 74, 1)
    assert K_map.sum() > 0, "K_map should have coverage"
    assert D_map.sum() == 0
    assert I_map.sum() == 0

    M_pre, M_src, M_dst = builder.pool_and_normalize(K_map, D_map, I_map, target_grid=37)
    assert M_pre.shape == (1, 1, 37 * 37, 1)
    covered = K_map.reshape(1, 1, 74 * 74).sum(dim=-1) > 0
    assert covered.all() or covered.any()

    # On patches where kept coverage exists, M_preserve should dominate.
    K_pool = torch.nn.functional.avg_pool2d(
        K_map.reshape(1, 74, 74).unsqueeze(0), kernel_size=2
    )
    covered_patches = K_pool.flatten() > 0.05
    if covered_patches.any():
        assert M_pre.flatten()[covered_patches.nonzero(as_tuple=True)[0]].mean() > 0.9


def test_only_assets_yield_dest_mask():
    device = torch.device("cuda")
    builder = SoftMaskBuilder()
    cameras = _id_camera(74, 74, device)

    K_map, D_map, I_map, per_obj = builder.render_coverage(
        G_kept=[_empty_gauss(device)],
        G_deleted=[_empty_gauss(device)],
        G_asset_dggt_dict=[{5: _gauss(0.0, 0.0, device)}],
        cameras_dggt=cameras,
        H=74, W=74,
    )
    assert K_map.sum() == 0 and D_map.sum() == 0
    assert I_map.sum() > 0
    assert 5 in per_obj[0]
    assert per_obj[0][5].shape == (1, 74, 74, 1)

    M_pre, M_src, M_dst = builder.pool_and_normalize(K_map, D_map, I_map, target_grid=37)
    covered = I_map.flatten(2).sum(dim=-1).squeeze() > 0
    assert covered.item()
    # Where I_map coverage is ~1, M_dest should be close to 1.
    dst_values = M_dst.flatten()
    hot = dst_values > 0.5
    assert hot.any(), "no patch was dominated by dest mask"


def test_empty_scene_returns_zero_masks():
    device = torch.device("cuda")
    builder = SoftMaskBuilder()
    cameras = _id_camera(74, 74, device)

    K_map, D_map, I_map, _ = builder.render_coverage(
        G_kept=[_empty_gauss(device)],
        G_deleted=[_empty_gauss(device)],
        G_asset_dggt_dict=[{}],
        cameras_dggt=cameras,
        H=74, W=74,
    )
    M_pre, M_src, M_dst = builder.pool_and_normalize(K_map, D_map, I_map, target_grid=37)
    assert (M_pre.abs() + M_src.abs() + M_dst.abs()).max() < 1e-3


def test_soft_masks_sum_to_unit_where_covered():
    device = torch.device("cuda")
    builder = SoftMaskBuilder()
    cameras = _id_camera(74, 74, device)

    K_map, D_map, I_map, _ = builder.render_coverage(
        G_kept=[_gauss(-0.3, 0.0, device)],
        G_deleted=[_gauss(0.0, 0.0, device)],
        G_asset_dggt_dict=[{0: _gauss(0.3, 0.0, device)}],
        cameras_dggt=cameras,
        H=74, W=74,
    )
    M_pre, M_src, M_dst = builder.pool_and_normalize(K_map, D_map, I_map, target_grid=37)
    total = (M_pre + M_src + M_dst).flatten()
    # Filter patches whose raw coverage is far above eps — those should sum to 1.
    raw_total = (
        torch.nn.functional.avg_pool2d(
            (K_map + D_map + I_map).reshape(1, 74, 74).unsqueeze(0), kernel_size=2
        )
        .flatten()
    )
    well_covered = raw_total > 0.5
    assert well_covered.any(), "no well-covered patches to check"
    assert (total[well_covered] - 1.0).abs().max() < 1e-3


def test_pool_normalize_shape_and_grad():
    device = torch.device("cuda")
    builder = SoftMaskBuilder()
    K = torch.rand((2, 3, 74, 74, 1), device=device, requires_grad=True)
    D = torch.rand((2, 3, 74, 74, 1), device=device, requires_grad=True)
    I = torch.rand((2, 3, 74, 74, 1), device=device, requires_grad=True)
    M_pre, M_src, M_dst = builder.pool_and_normalize(K, D, I, target_grid=37)
    assert M_pre.shape == (2, 3, 37 * 37, 1)
    (M_pre.sum() + M_src.sum() + M_dst.sum()).backward()
    assert K.grad is not None and K.grad.abs().sum() > 0
