from __future__ import annotations

import torch

from dggt.losses.flow_losses import compute_total_loss
from dggt.utils.pretrain_pseudo_asset import (
    PretrainBundle,
    apply_uncond_drop,
    build_pretrain_bundle,
)


def _paint(mask: torch.Tensor, batch: int, frame: int, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        mask[batch, frame, :, r, c] = 1.0


def test_build_pretrain_bundle_k3_shapes_and_masks():
    B, S, gh, gw, z_dim, text_dim = 1, 2, 8, 8, 5, 7
    P = gh * gw
    z_clean_n = torch.randn(B, S, P, z_dim)
    image_tokens = torch.randn(B, S, P, text_dim)
    dynamic_mask = torch.zeros(B, S, 3, gh, gw)

    comp_a = [(1, 1), (1, 2), (2, 1), (2, 2)]
    comp_b = [(1, 5), (2, 5), (3, 5)]
    comp_c = [(5, 1), (5, 2)]
    comp_edge = [(0, 7), (1, 7)]
    comp_ref = [(4, 4), (4, 5), (5, 4)]
    for comp in (comp_a, comp_b, comp_c, comp_edge):
        _paint(dynamic_mask, 0, 0, comp)
    _paint(dynamic_mask, 0, 1, comp_ref)

    bundle = build_pretrain_bundle(
        z_clean_n=z_clean_n,
        image_tokens_last=image_tokens,
        dynamic_mask=dynamic_mask,
        patch_grid=(gh, gw),
        K_max=3,
        min_inst_patches=1,
        max_inst_patches=10,
    )

    assert isinstance(bundle, PretrainBundle)
    assert bundle.M_dest.shape == (B, S, P, 1)
    assert bundle.M_preserve.shape == (B, S, P, 1)
    assert bundle.M_source.shape == (B, S, P, 1)
    assert bundle.F_asset_tokens.shape == (B, 12, text_dim)
    assert bundle.encoder_attention_mask is not None
    assert bundle.encoder_attention_mask.tolist() == [[True] * 12]
    assert bundle.F_asset_lengths.tolist() == [12]

    assert int(bundle.M_dest[0, 0].sum().item()) == 9
    assert int(bundle.M_dest[0, 1].sum().item()) == 3
    assert torch.allclose(bundle.M_preserve + bundle.M_dest, torch.ones_like(bundle.M_dest))
    assert torch.count_nonzero(bundle.M_source) == 0


def test_build_pretrain_bundle_k0_all_batch_returns_empty_kv_and_no_mask():
    B, S, gh, gw, z_dim, text_dim = 2, 3, 4, 4, 5, 7
    P = gh * gw
    z_clean_n = torch.randn(B, S, P, z_dim)
    image_tokens = torch.randn(B, S, P, text_dim)
    dynamic_mask = torch.zeros(B, S, 3, gh, gw)

    bundle = build_pretrain_bundle(
        z_clean_n=z_clean_n,
        image_tokens_last=image_tokens,
        dynamic_mask=dynamic_mask,
        patch_grid=(gh, gw),
        K_max=3,
        min_inst_patches=1,
    )

    assert bundle.F_asset_tokens.shape == (B, 0, text_dim)
    assert bundle.encoder_attention_mask is None
    assert bundle.F_asset_lengths.tolist() == [0, 0]
    assert torch.allclose(bundle.M_preserve, torch.ones_like(bundle.M_preserve))
    assert torch.count_nonzero(bundle.M_source) == 0
    assert torch.count_nonzero(bundle.M_dest) == 0


def test_build_pretrain_bundle_new_signature_and_dtype():
    B, S, gh, gw, z_dim, text_dim = 1, 2, 4, 4, 3, 2
    P = gh * gw
    z_clean_n = torch.randn(B, S, P, z_dim)
    image_tokens = torch.randn(B, S, P, text_dim)
    dynamic_mask = torch.zeros(B, S, 3, gh, gw)
    _paint(dynamic_mask, 0, 0, [(1, 1), (1, 2)])
    _paint(dynamic_mask, 0, 1, [(2, 1), (2, 2)])

    bundle = build_pretrain_bundle(
        z_clean_n=z_clean_n,
        image_tokens_last=image_tokens,
        dynamic_mask=dynamic_mask,
        patch_grid=(gh, gw),
        K_max=1,
        min_inst_patches=1,
        dtype=torch.bfloat16,
    )

    assert bundle.F_asset_tokens.dtype == torch.bfloat16
    assert bundle.M_dest.dtype == torch.bfloat16
    assert bundle.encoder_attention_mask is not None


def test_build_pretrain_bundle_uses_8_connected_components():
    B, S, gh, gw, z_dim, text_dim = 1, 2, 4, 4, 3, 2
    P = gh * gw
    z_clean_n = torch.randn(B, S, P, z_dim)
    image_tokens = torch.randn(B, S, P, text_dim)
    dynamic_mask = torch.zeros(B, S, 3, gh, gw)
    diagonal_comp = [(1, 1), (2, 2)]
    _paint(dynamic_mask, 0, 0, diagonal_comp)
    _paint(dynamic_mask, 0, 1, diagonal_comp)

    bundle = build_pretrain_bundle(
        z_clean_n=z_clean_n,
        image_tokens_last=image_tokens,
        dynamic_mask=dynamic_mask,
        patch_grid=(gh, gw),
        K_max=1,
        min_inst_patches=2,
    )

    assert int(bundle.M_dest[0, 0].sum().item()) == 2
    assert int(bundle.M_dest[0, 1].sum().item()) == 2
    assert bundle.F_asset_lengths.tolist() == [4]


def test_build_pretrain_bundle_mixed_batch_empty_row_mask():
    B, S, gh, gw, z_dim, text_dim = 2, 2, 4, 4, 3, 2
    P = gh * gw
    z_clean_n = torch.randn(B, S, P, z_dim)
    image_tokens = torch.randn(B, S, P, text_dim)
    dynamic_mask = torch.zeros(B, S, 3, gh, gw)
    _paint(dynamic_mask, 0, 0, [(1, 1), (1, 2)])
    _paint(dynamic_mask, 0, 1, [(2, 1), (2, 2)])

    bundle = build_pretrain_bundle(
        z_clean_n=z_clean_n,
        image_tokens_last=image_tokens,
        dynamic_mask=dynamic_mask,
        patch_grid=(gh, gw),
        K_max=1,
        min_inst_patches=1,
    )

    assert bundle.encoder_attention_mask is not None
    assert bundle.F_asset_lengths.tolist() == [4, 0]
    assert bundle.encoder_attention_mask[0].tolist() == [True, True, True, True]
    assert bundle.encoder_attention_mask[1].tolist() == [False, False, False, False]


def test_build_pretrain_bundle_cross_frame_no_self_leak():
    B, S, gh, gw, z_dim, text_dim = 1, 2, 4, 4, 3, 2
    P = gh * gw
    z_clean_n = torch.randn(B, S, P, z_dim)
    image_tokens = torch.zeros(B, S, P, text_dim)
    image_tokens[:, 0, :, :] = 10.0
    image_tokens[:, 1, :, :] = 20.0
    dynamic_mask = torch.zeros(B, S, 3, gh, gw)
    comp = [(1, 1), (1, 2)]
    _paint(dynamic_mask, 0, 0, comp)
    _paint(dynamic_mask, 0, 1, comp)

    bundle = build_pretrain_bundle(
        z_clean_n=z_clean_n,
        image_tokens_last=image_tokens,
        dynamic_mask=dynamic_mask,
        patch_grid=(gh, gw),
        K_max=1,
        min_inst_patches=1,
    )

    assert bundle.F_asset_tokens.shape == (B, 4, text_dim)
    assert torch.equal(bundle.F_asset_tokens[0, :2], torch.full((2, text_dim), 20.0))
    assert torch.equal(bundle.F_asset_tokens[0, 2:], torch.full((2, text_dim), 10.0))


def test_apply_uncond_drop_zeroes_some_rows_when_prob_one():
    """uncond_drop_prob=1.0 forces every batch row's KV mask to all-False."""
    B, S, P, C, text_dim = 2, 2, 4, 3, 5
    bundle = PretrainBundle(
        z_clean_n=torch.randn(B, S, P, C),
        M_preserve=torch.zeros(B, S, P, 1),
        M_source=torch.zeros(B, S, P, 1),
        M_dest=torch.ones(B, S, P, 1),
        F_asset_tokens=torch.randn(B, 6, text_dim),
        encoder_attention_mask=torch.ones(B, 6, dtype=torch.bool),
        F_asset_lengths=torch.tensor([6, 6], dtype=torch.long),
    )
    dropped = apply_uncond_drop(bundle, prob=1.0)
    assert dropped.encoder_attention_mask is not None
    assert bool((~dropped.encoder_attention_mask).all().item())
    assert dropped.F_asset_lengths.tolist() == [0, 0]


def test_apply_uncond_drop_noop_when_prob_zero():
    B = 2
    mask = torch.ones(B, 4, dtype=torch.bool)
    bundle = PretrainBundle(
        z_clean_n=torch.zeros(B, 1, 4, 2),
        M_preserve=torch.zeros(B, 1, 4, 1),
        M_source=torch.zeros(B, 1, 4, 1),
        M_dest=torch.ones(B, 1, 4, 1),
        F_asset_tokens=torch.zeros(B, 4, 3),
        encoder_attention_mask=mask,
        F_asset_lengths=torch.tensor([4, 4], dtype=torch.long),
    )
    out = apply_uncond_drop(bundle, prob=0.0)
    assert out is bundle


def test_pretrain_bundle_compute_total_loss_compat():
    B, S, P, C = 2, 2, 4, 3
    z_clean_n = torch.randn(B, S, P, C)
    M_dest = torch.zeros(B, S, P, 1)
    M_dest[:, :, :2] = 1.0
    bundle = PretrainBundle(
        z_clean_n=z_clean_n,
        M_preserve=1.0 - M_dest,
        M_source=torch.zeros_like(M_dest),
        M_dest=M_dest,
        F_asset_tokens=torch.zeros(B, 0, 6),
        encoder_attention_mask=None,
        F_asset_lengths=torch.zeros(B, dtype=torch.long),
    )
    v_pred = torch.randn_like(z_clean_n)
    v_gt = torch.randn_like(z_clean_n)
    eps = torch.randn_like(z_clean_n)
    weights = torch.ones(B, 1, 1, 1)

    loss, logs = compute_total_loss(
        v_pred=v_pred,
        v_gt=v_gt,
        eps=eps,
        bundle=bundle,
        sd3_weights=weights,
        lambda_repa=0.0,
        lambda_identity=0.0,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert set(logs) == {"loss", "loss_flow", "loss_preserve", "loss_repa", "loss_identity"}
