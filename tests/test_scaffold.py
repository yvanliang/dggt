"""Smoke tests for `dggt.models.scaffold.ScaffoldPacker`."""
from __future__ import annotations

import torch

from dggt.models.scaffold import ScaffoldPacker


def test_forward_shape():
    packer = ScaffoldPacker(in_channels=7, out_dim=768, hidden_dim=64)
    x = torch.randn(2, 3, 518, 518, 7)
    out = packer(x, target_grid=37)
    assert out.shape == (2, 3, 37 * 37, 768)


def test_gradient_flows():
    packer = ScaffoldPacker(in_channels=7, out_dim=64, hidden_dim=32)
    x = torch.randn(1, 2, 74, 74, 7, requires_grad=True)
    out = packer(x, target_grid=37)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_rejects_wrong_channels():
    packer = ScaffoldPacker(in_channels=7, out_dim=32)
    bad = torch.randn(1, 2, 74, 74, 5)
    try:
        packer(bad, target_grid=37)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong channel count")


def test_rejects_unpoolable_hw():
    packer = ScaffoldPacker(in_channels=3, out_dim=16)
    bad = torch.randn(1, 1, 75, 75, 3)  # 75 not divisible by 37
    try:
        packer(bad, target_grid=37)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible H/W")


def test_build_scaffold_hires_assembles_seven_channels():
    B, S, H, W = 1, 2, 74, 74
    D_edited = torch.rand(B, S, H, W, 1)
    A_edited = torch.rand(B, S, H, W, 1)
    K_map = torch.rand(B, S, H, W, 1)
    D_map = torch.rand(B, S, H, W, 1)
    I_map = torch.rand(B, S, H, W, 1)
    dyn = torch.rand(B, S, H, W, 1)

    scaffold = ScaffoldPacker.build_scaffold_hires(
        D_edited, A_edited, K_map, D_map, I_map, dyn
    )
    assert scaffold.shape == (B, S, H, W, 7)
    # Time channel (index 6) should increase monotonically across S.
    time_channel = scaffold[0, :, 0, 0, 6]
    assert torch.all(time_channel[1:] >= time_channel[:-1])


def test_build_scaffold_hires_explicit_time_index():
    B, S, H, W = 1, 3, 74, 74
    zeros = torch.zeros(B, S, H, W, 1)
    t = torch.tensor([[0.0, 0.5, 1.0]])
    scaffold = ScaffoldPacker.build_scaffold_hires(
        zeros, zeros, zeros, zeros, zeros, zeros, time_index=t
    )
    assert torch.allclose(scaffold[0, :, 0, 0, 6], t.squeeze(0), atol=1e-6)
