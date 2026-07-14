"""Smoke tests for `dggt.models.scaffold.ScaffoldPacker`."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dggt.models.scaffold import ScaffoldPacker


def _run_pooled_scaffold_ddp_step(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        packer = DistributedDataParallel(
            ScaffoldPacker(in_channels=7, out_dim=4, hidden_dim=8)
        )
        optimizer = torch.optim.SGD(packer.parameters(), lr=0.1)
        pooled = torch.full((1, 2, 3, 7), float(rank + 1))
        loss = packer(pooled, already_pooled=True).square().mean()
        loss.backward()
        optimizer.step()
        torch.save(packer.module.state_dict(), os.path.join(output_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


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


def test_already_pooled_forward_matches_mlp_and_keeps_gradients():
    packer = ScaffoldPacker(in_channels=7, out_dim=16, hidden_dim=8)
    pooled = torch.randn(2, 3, 5, 7, requires_grad=True)

    expected = packer.mlp(pooled)
    actual = packer(pooled, already_pooled=True)

    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert pooled.grad is not None and pooled.grad.abs().sum() > 0


def test_already_pooled_forward_rejects_hires_or_wrong_channels():
    packer = ScaffoldPacker(in_channels=7, out_dim=16, hidden_dim=8)

    for invalid in (torch.randn(1, 2, 4, 4, 7), torch.randn(1, 2, 4, 6)):
        try:
            packer(invalid, already_pooled=True)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for invalid already-pooled scaffold")


def test_already_pooled_forward_synchronizes_two_ddp_ranks(tmp_path):
    if not dist.is_available():
        return
    rendezvous = tmp_path / "ddp_init"
    mp.start_processes(
        _run_pooled_scaffold_ddp_step,
        args=(2, str(rendezvous), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    rank0 = torch.load(tmp_path / "rank0.pt", map_location="cpu")
    rank1 = torch.load(tmp_path / "rank1.pt", map_location="cpu")
    assert rank0.keys() == rank1.keys()
    for key in rank0:
        assert torch.equal(rank0[key], rank1[key]), key


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
