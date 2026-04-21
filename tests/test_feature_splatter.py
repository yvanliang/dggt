"""Smoke tests for `dggt.models.feature_splatter.FeatureSplatter`.

Requires a CUDA device because gsplat rasterization is CUDA-only.
"""
from __future__ import annotations

import pytest
import torch

from dggt.models.feature_splatter import (
    FeatureSplatter,
    GaussianPointers,
    SRC_KIND_ASSET,
    SRC_KIND_SCENE,
    SCENE_OBJECT_ID,
)


CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="gsplat requires CUDA")


def _make_camera(H: int, W: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Single identity-pose camera at origin looking down +z.
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[2, 3] = 0.0  # camera at origin
    fx = fy = float(W)
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=torch.float32, device=device)
    return viewmat.unsqueeze(0), K.unsqueeze(0)  # [1,4,4], [1,3,3]


def _make_scene_gaussian(device: torch.device) -> dict[str, torch.Tensor]:
    # One Gaussian at (0, 0, 3), covering the image center.
    means = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.3, 0.3, 0.3]], device=device)
    opacities = torch.tensor([0.99], device=device)
    return {"means": means, "quats": quats, "scales": scales, "opacities": opacities}


def _trivial_pointers(n: int, device: torch.device) -> GaussianPointers:
    return GaussianPointers(
        src_kind=torch.full((n,), SRC_KIND_SCENE, dtype=torch.int32, device=device),
        object_id=torch.full((n,), SCENE_OBJECT_ID, dtype=torch.int32, device=device),
        view_n=torch.zeros((n,), dtype=torch.int32, device=device),
        patch_idx=torch.zeros((n,), dtype=torch.int32, device=device),
        visible_mask=torch.ones((n,), dtype=torch.bool, device=device),
    )


def test_output_shapes_and_pool():
    device = torch.device("cuda")
    splatter = FeatureSplatter(channels=16, chunk_channels=8, num_levels=2, patch_grid=(37, 37))

    gauss = _make_scene_gaussian(device)
    ptr = _trivial_pointers(n=1, device=device)

    # LUT: [B=1, N_scene=1, P=1369, C=16]; all ones on patch 0 row.
    lut_levels = [
        torch.ones((1, 1, 1369, 16), device=device, dtype=torch.float32) * (lvl + 1.0)
        for lvl in range(2)
    ]
    viewmat, K = _make_camera(74, 74, device)
    cameras = {"viewmats": viewmat.unsqueeze(0), "Ks": K.unsqueeze(0)}  # [1,1,4,4], [1,1,3,3]

    outs = splatter(
        gaussians_dggt=[gauss],
        pointers=[ptr],
        lut_scene=lut_levels,
        lut_asset_dict=None,
        cameras_dggt=cameras,
        H=74,
        W=74,
        pool_to=37,
    )
    assert len(outs) == 2
    for lvl, out in enumerate(outs):
        assert out.shape == (1, 1, 37 * 37, 16)
        # Covered patches should be roughly (lvl+1); empty patches should be 0.
        covered = out[0, 0].sum(dim=-1) > 0
        assert covered.any(), f"level {lvl} produced no coverage"


def test_gradient_flows_to_lut_only():
    device = torch.device("cuda")
    splatter = FeatureSplatter(channels=8, chunk_channels=8, num_levels=1, patch_grid=(37, 37))

    gauss = _make_scene_gaussian(device)
    # Require grad on geometry; the module must detach internally.
    gauss = {k: v.clone().requires_grad_(True) for k, v in gauss.items()}
    ptr = _trivial_pointers(n=1, device=device)

    lut = torch.randn((1, 1, 1369, 8), device=device, requires_grad=True)
    viewmat, K = _make_camera(74, 74, device)
    cameras = {"viewmats": viewmat.unsqueeze(0), "Ks": K.unsqueeze(0)}

    outs = splatter(
        gaussians_dggt=[gauss],
        pointers=[ptr],
        lut_scene=[lut],
        lut_asset_dict=None,
        cameras_dggt=cameras,
        H=74,
        W=74,
        pool_to=37,
    )
    loss = outs[0].pow(2).mean()
    loss.backward()

    assert lut.grad is not None and lut.grad.abs().sum() > 0
    # Geometry grads should be untouched by the splatter.
    for key in ("means", "quats", "scales", "opacities"):
        assert gauss[key].grad is None or gauss[key].grad.abs().sum() == 0, (
            f"{key} received unexpected gradient"
        )


def test_chunked_equals_unchunked():
    device = torch.device("cuda")
    torch.manual_seed(0)
    C = 24
    splatter_full = FeatureSplatter(channels=C, chunk_channels=C, num_levels=1, patch_grid=(37, 37))
    splatter_chunk = FeatureSplatter(channels=C, chunk_channels=8, num_levels=1, patch_grid=(37, 37))

    gauss = _make_scene_gaussian(device)
    ptr = _trivial_pointers(n=1, device=device)
    lut = torch.randn((1, 1, 1369, C), device=device)
    viewmat, K = _make_camera(74, 74, device)
    cameras = {"viewmats": viewmat.unsqueeze(0), "Ks": K.unsqueeze(0)}

    out_full = splatter_full(
        gaussians_dggt=[gauss], pointers=[ptr], lut_scene=[lut],
        lut_asset_dict=None, cameras_dggt=cameras, H=74, W=74, pool_to=37,
    )[0]
    out_chunk = splatter_chunk(
        gaussians_dggt=[gauss], pointers=[ptr], lut_scene=[lut],
        lut_asset_dict=None, cameras_dggt=cameras, H=74, W=74, pool_to=37,
    )[0]
    diff = (out_full - out_chunk).abs().max().item()
    assert diff < 1e-4, f"chunked vs full max diff {diff} > 1e-4"


def test_invisible_gaussian_contributes_zero():
    device = torch.device("cuda")
    splatter = FeatureSplatter(channels=4, chunk_channels=4, num_levels=1, patch_grid=(37, 37))

    gauss = _make_scene_gaussian(device)
    ptr = GaussianPointers(
        src_kind=torch.tensor([SRC_KIND_SCENE], dtype=torch.int32, device=device),
        object_id=torch.tensor([SCENE_OBJECT_ID], dtype=torch.int32, device=device),
        view_n=torch.tensor([0], dtype=torch.int32, device=device),
        patch_idx=torch.tensor([0], dtype=torch.int32, device=device),
        visible_mask=torch.tensor([False], dtype=torch.bool, device=device),
    )
    lut = torch.ones((1, 1, 1369, 4), device=device) * 5.0
    viewmat, K = _make_camera(74, 74, device)
    cameras = {"viewmats": viewmat.unsqueeze(0), "Ks": K.unsqueeze(0)}

    out = splatter(
        gaussians_dggt=[gauss], pointers=[ptr], lut_scene=[lut],
        lut_asset_dict=None, cameras_dggt=cameras, H=74, W=74, pool_to=37,
    )[0]
    assert out.abs().max().item() < 1e-5, "invisible gaussian leaked into splat"


def test_asset_pointer_gather():
    """Scene Gaussians at one patch, asset Gaussian at a different patch; verify
    global-index gather picks the right LUT entries."""
    device = torch.device("cuda")
    splatter = FeatureSplatter(channels=4, chunk_channels=4, num_levels=1, patch_grid=(37, 37))

    # Two Gaussians: one scene, one asset.
    means = torch.tensor([[-0.5, 0.0, 3.0], [0.5, 0.0, 3.0]], device=device)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, device=device)
    scales = torch.tensor([[0.2, 0.2, 0.2]] * 2, device=device)
    opacities = torch.tensor([0.99, 0.99], device=device)
    gauss = {"means": means, "quats": quats, "scales": scales, "opacities": opacities}

    ptr = GaussianPointers(
        src_kind=torch.tensor([SRC_KIND_SCENE, SRC_KIND_ASSET], dtype=torch.int32, device=device),
        object_id=torch.tensor([SCENE_OBJECT_ID, 7], dtype=torch.int32, device=device),
        view_n=torch.tensor([0, 0], dtype=torch.int32, device=device),
        patch_idx=torch.tensor([0, 0], dtype=torch.int32, device=device),
        visible_mask=torch.tensor([True, True], dtype=torch.bool, device=device),
    )

    scene_lut = torch.zeros((1, 1, 1369, 4), device=device)
    scene_lut[0, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    asset_lut = torch.zeros((1, 1, 1369, 4), device=device)
    asset_lut[0, 0, 0, :] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    lut_asset_dict = {7: [asset_lut]}

    viewmat, K = _make_camera(74, 74, device)
    cameras = {"viewmats": viewmat.unsqueeze(0), "Ks": K.unsqueeze(0)}
    out = splatter(
        gaussians_dggt=[gauss], pointers=[ptr], lut_scene=[scene_lut],
        lut_asset_dict=lut_asset_dict, cameras_dggt=cameras, H=74, W=74, pool_to=None,
    )[0]  # [1, 1, 74, 74, 4]
    # Left half should carry channel 0 (scene), right half channel 1 (asset).
    left = out[0, 0, :, : 74 // 2 - 4, :].sum(dim=(0, 1))
    right = out[0, 0, :, 74 // 2 + 4:, :].sum(dim=(0, 1))
    assert left[0] > left[1] + 1e-3, "scene channel did not dominate left half"
    assert right[1] > right[0] + 1e-3, "asset channel did not dominate right half"
