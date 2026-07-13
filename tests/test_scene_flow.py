from __future__ import annotations

import inspect

import pytest
import torch

from dggt.losses.flow_losses import preserve_loss
from dggt.models.scene_flow import (
    AssetFrameCompressor,
    DDTEncoderBlock,
    DDTFinalLayer,
    ROPE_LAYOUT_VERSION,
    SKY_MROPE_TEMPORAL_OFFSET,
    WanSceneFlow,
)


def _tiny_model(**kwargs) -> WanSceneFlow:
    defaults = dict(
        patch_grid=(2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=27,
        out_channels=8,
        text_dim=12,
        ffn_dim=32,
        num_layers=1,
        base_model_depth=1,
        ddt_head_dim=16,
        ddt_head_heads=2,
        ddt_head_depth=1,
        rope_max_seq_len=8,
    )
    defaults.update(kwargs)
    return WanSceneFlow(**defaults)


def _inputs(batch=1, frames=2, patches=4, dim=8, text_dim=12, kv=0):
    z_t = torch.randn(batch, frames, patches, dim)
    z_splat = torch.randn_like(z_t)
    scaffold = torch.randn_like(z_t)
    M_preserve = torch.rand(batch, frames, patches, 1)
    M_source = torch.rand(batch, frames, patches, 1)
    M_dest = torch.rand(batch, frames, patches, 1)
    sigma = torch.rand(batch)
    assets = torch.randn(batch, kv, text_dim) if kv else torch.zeros(batch, 0, text_dim)
    return z_t, sigma, z_splat, scaffold, M_preserve, M_source, M_dest, assets


def test_forward_signature_does_not_accept_z_clean():
    assert "z_clean" not in inspect.signature(WanSceneFlow.forward).parameters


def test_scene_flow_shape_and_zero_start_k0():
    model = _tiny_model()
    out = model(*_inputs(kv=0))
    assert out.shape == (1, 2, 4, 8)
    assert float(out.abs().max()) < 1e-5


def test_scene_flow_shape_kv_masked():
    model = _tiny_model()
    args = _inputs(kv=3)
    mask = torch.tensor([[True, False, True]])
    out = model(*args, encoder_attention_mask=mask)
    assert out.shape == (1, 2, 4, 8)


def test_asset_frame_compressor_fixed_slots_and_mask():
    compressor = AssetFrameCompressor(in_dim=8, hidden_size=16, max_assets=5, max_frames=4)
    latents = torch.randn(2, 2, 3, 4, 8)
    valid = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    valid[:, 1, 2] = False
    tokens, mask = compressor(latents, patch_grid=(2, 2), valid_mask=valid)
    assert tokens.shape == (2, 5, 16)
    assert mask.shape == (2, 5)
    assert mask[:, :2].all()
    assert not mask[:, 2:].any()


def test_legacy_asset_compressor_has_no_max_frame_clamp():
    compressor = AssetFrameCompressor(in_dim=8, hidden_size=16, max_assets=1, max_frames=4)
    frame = torch.randn(1, 1, 1, 4, 8)
    short = frame.expand(-1, -1, 4, -1, -1).clone()
    long = frame.expand(-1, -1, 9, -1, -1).clone()

    short_tokens, short_mask = compressor(short, patch_grid=(2, 2))
    long_tokens, long_mask = compressor(long, patch_grid=(2, 2))

    assert short_mask.all() and long_mask.all()
    assert torch.allclose(short_tokens, long_tokens, atol=1e-6, rtol=1e-6)


def test_sparse_asset_tokens_are_identical_across_overlapping_global_windows():
    model = _tiny_model(
        max_frames=16,
        max_asset_patch_tokens_per_asset_frame=4,
        max_asset_tokens=256,
    )
    assets = torch.randn(1, 1, 24, 4, 8)
    valid = torch.ones(1, 1, 24, 4, dtype=torch.bool)

    tokens_a, mask_a, pos_a = model._build_sparse_asset_condition(
        assets[:, :, :16],
        valid[:, :, :16],
        seq_len=16,
        num_patches=4,
        patch_grid=(2, 2),
        frame_ids=torch.arange(16),
    )
    tokens_b, mask_b, pos_b = model._build_sparse_asset_condition(
        assets[:, :, 8:24],
        valid[:, :, 8:24],
        seq_len=16,
        num_patches=4,
        patch_grid=(2, 2),
        frame_ids=torch.arange(8, 24),
    )

    # Four patch tokens plus one summary token are retained per asset/frame.
    tokens_per_frame = 5
    overlap_a = slice(8 * tokens_per_frame, 16 * tokens_per_frame)
    overlap_b = slice(0, 8 * tokens_per_frame)
    assert mask_a is not None and mask_b is not None
    assert mask_a[:, overlap_a].all() and mask_b[:, overlap_b].all()
    assert torch.equal(tokens_a[:, overlap_a], tokens_b[:, overlap_b])
    assert torch.equal(pos_a[:, overlap_a], pos_b[:, overlap_b])

    # Temporal identity remains unbounded in mRoPE rather than clamping at 15.
    assert pos_b[0, 8 * tokens_per_frame, 0].item() == 16
    assert pos_b[0, 15 * tokens_per_frame, 0].item() == 23


def test_preserve_loss_velocity_fallback_recovers_clean_as_noise_minus_velocity():
    z_clean = torch.randn(2, 3, 4, 8)
    eps = torch.randn_like(z_clean)
    v_pred = eps - z_clean
    preserve = torch.ones(2, 3, 4, 1)

    loss = preserve_loss(v_pred, eps, z_clean, preserve)

    assert float(loss.item()) < 1e-12


def test_scene_flow_optional_qwen_text_tokens_and_padding_mask():
    model = _tiny_model(qwen_dim=10)
    args = _inputs(kv=0)
    text_tokens = torch.randn(1, 5, 10)
    text_mask = torch.tensor([[True, True, False, False, False]])
    out = model(*args, text_tokens=text_tokens, text_attention_mask=text_mask)
    assert out.shape == (1, 2, 4, 8)


def test_scene_flow_trunk_uses_full_self_attention_blocks():
    model = _tiny_model()
    assert isinstance(model.blocks[0], DDTEncoderBlock)


def test_scene_flow_defaults_to_clean_prediction():
    model = _tiny_model()
    assert model.config.prediction_type == "x"


def test_a1_rope_layout_is_versioned_in_config():
    model = _tiny_model()
    assert model.config.rope_layout_version == ROPE_LAYOUT_VERSION


def test_ddt_visual_embedders_consume_only_noisy_latent_tokens():
    model = _tiny_model(in_channels=27, out_channels=8)
    assert model.video_embed.in_features == 8
    assert model.decoder_video_embed.in_features == 8
    assert model.config.in_channels == 27


def test_legacy_packed_video_embed_weights_are_sliced_on_load():
    model = _tiny_model(in_channels=27, out_channels=8, ddt_head_dim=16)
    state = model.state_dict()
    old_video_weight = torch.randn(model.video_embed.out_features, 27)
    old_decoder_weight = torch.randn(model.decoder_video_embed.out_features, 27)
    state["video_embed.weight"] = old_video_weight
    state["decoder_video_embed.weight"] = old_decoder_weight

    loaded = _tiny_model(in_channels=27, out_channels=8, ddt_head_dim=16)
    loaded.load_state_dict(state, strict=True)

    assert torch.equal(loaded.video_embed.weight, old_video_weight[:, :8])
    assert torch.equal(loaded.decoder_video_embed.weight, old_decoder_weight[:, :8])


def test_base_model_depth_must_be_inside_encoder_depth():
    with pytest.raises(ValueError, match="base_model_depth"):
        _tiny_model(num_layers=1, base_model_depth=2)


def test_repa_uses_layer_depth_not_zero_based_block_index():
    model = _tiny_model(num_layers=3, base_model_depth=1, repa_block_frac=1.0 / 3.0)
    assert model.config.repa_layer_depth == 1
    assert model.repa_block_idx == 0


def test_ddt_head_ffn_dim_is_honored():
    model = _tiny_model(ddt_head_dim=18, ddt_head_heads=3, ddt_head_ffn_dim=30)
    assert model.ddt_head[0].mlp.w1.out_features == int(2.0 / 3.0 * 30)


def test_ddt_final_layer_broadcasts_batch_condition():
    layer = DDTFinalLayer(hidden_size=8, out_channels=3)
    x = torch.randn(2, 5, 8)
    c = torch.randn(2, 8)
    out = layer(x, c)
    assert out.shape == (2, 5, 3)


def test_return_base_controls_auxiliary_ddt_output():
    model = _tiny_model()
    args = _inputs(kv=0)
    out = model(*args, return_base=False)
    assert torch.is_tensor(out)

    out_with_base = model(*args, return_base=True)
    assert isinstance(out_with_base, tuple)
    assert out_with_base[0].shape == out_with_base[1].shape == (1, 2, 4, 8)


def test_qwen_text_condition_uses_zero_rope_positions():
    model = _tiny_model(qwen_dim=10)
    pos = model._text_position_ids(batch_size=2, num_tokens=5, device=torch.device("cpu"))
    assert pos.shape == (2, 5, 3)
    assert not pos.any()


def test_a1_rope_positions_use_video_camera_shared_grid_and_sky_offset():
    model = _tiny_model()
    video_pos = model._target_position_ids(
        batch_size=1,
        seq_len=2,
        num_patches=4,
        patch_grid=(2, 2),
        device=torch.device("cpu"),
    )
    camera_pos = model._camera_position_ids(batch_size=1, seq_len=2, device=torch.device("cpu"))
    sky_pos = model._sky_position_ids(batch_size=1, num_tokens=4, device=torch.device("cpu"))

    assert video_pos[0, [0, 4], 0].tolist() == [0, 1]
    assert camera_pos[0, :, 0].tolist() == [0, 1]
    assert camera_pos[0, :, 1:].tolist() == [[1, 1], [1, 1]]
    assert sky_pos[0, :, 0].tolist() == [SKY_MROPE_TEMPORAL_OFFSET] * 4


def test_global_mrope_temporal_margin_is_removed():
    with pytest.raises(TypeError, match="mrope_temporal_margin has been removed"):
        _tiny_model(mrope_temporal_margin=7)


def test_full_attention_padding_mask_masks_keys_and_zeroes_queries():
    model = _tiny_model()
    valid = torch.tensor([[True, False, True], [False, True, True]])

    attn_mask = model._key_padding_attention_mask(valid, torch.float32)
    assert attn_mask is not None
    assert attn_mask.shape == (2, 1, 1, 3)
    assert float(attn_mask[0, 0, 0, 0]) == 0.0
    assert float(attn_mask[0, 0, 0, 1]) < -1e20
    assert float(attn_mask[1, 0, 0, 0]) < -1e20

    x = torch.randn(2, 3, 4)
    masked = model._apply_token_valid_mask(x, valid)
    assert torch.equal(masked[0, 0], x[0, 0])
    assert torch.equal(masked[0, 2], x[0, 2])
    assert torch.equal(masked[0, 1], torch.zeros_like(masked[0, 1]))
    assert torch.equal(masked[1, 0], torch.zeros_like(masked[1, 0]))


def test_mode_b_empty_asset_condition_uses_learned_first_slot():
    """Mode-B empty is conditional and uses a visible learned empty token."""
    model = _tiny_model()
    assets = torch.zeros(2, 5, 2, 4, 8)
    mask = torch.zeros(2, 5, 2, 4, dtype=torch.bool)

    tokens, out_mask = model._build_asset_condition(
        assets,
        mask,
        seq_len=2,
        num_patches=4,
        patch_grid=(2, 2),
        asset_condition_kind=["mode_b_empty", "mode_a"],
    )

    assert tokens.shape == (2, 5, 16)
    assert out_mask is not None
    assert out_mask[0].tolist() == [True, False, False, False, False]
    assert out_mask[1].tolist() == [False, False, False, False, False]
    assert torch.allclose(tokens[0, 0], model.empty_asset_embed.detach().reshape(-1))


def test_sparse_mode_b_empty_asset_condition_keeps_learned_token_visible():
    model = _tiny_model()
    assets = torch.zeros(2, 5, 2, 4, 8)
    mask = torch.zeros(2, 5, 2, 4, dtype=torch.bool)

    tokens, out_mask, pos = model._build_sparse_asset_condition(
        assets,
        mask,
        seq_len=2,
        num_patches=4,
        patch_grid=(2, 2),
        asset_condition_kind=["mode_b_empty", "mode_a"],
    )

    assert tokens.shape == (2, 1, 16)
    assert out_mask is not None
    assert out_mask.tolist() == [[True], [False]]
    assert torch.allclose(tokens[0, 0], model.empty_asset_embed.detach().reshape(-1))
    assert pos[0, 0].tolist() == [0, 0, 0]


def test_sparse_asset_uncond_uses_single_learned_null_token():
    model = _tiny_model()
    assets = torch.randn(2, 5, 2, 4, 8)
    mask = torch.ones(2, 5, 2, 4, dtype=torch.bool)

    tokens, out_mask, pos = model._build_sparse_asset_condition(
        assets,
        mask,
        seq_len=2,
        num_patches=4,
        patch_grid=(2, 2),
        asset_condition_kind=["asset_uncond", "mode_a"],
    )

    assert out_mask is not None
    assert out_mask[0].tolist() == [True] + [False] * (tokens.shape[1] - 1)
    assert torch.allclose(tokens[0, 0], model.asset_null_condition_embed.detach().reshape(-1))
    assert pos[0, 0].tolist() == [0, 0, 0]
    assert int(out_mask[1].sum().item()) > 1


def test_camera_uncond_uses_per_frame_learned_null_tokens():
    model = _tiny_model()
    z = torch.randn(2, 3, 4, 8)
    camera = torch.randn(2, 3, int(model.config.camera_cond_dim))

    tokens, mask, pos = model._build_camera_condition(
        z,
        camera,
        None,
        camera_condition_kind=["camera_uncond", "camera"],
    )

    assert mask is not None
    assert mask[0].tolist() == [True, True, True]
    null_camera = model.camera_null_condition_embed.detach().reshape(-1).expand(3, -1)
    assert torch.allclose(tokens[0], null_camera)
    assert not torch.allclose(tokens[1], null_camera)
    assert pos[0, :, 1:].tolist() == [[1, 1], [1, 1], [1, 1]]


def test_forward_accepts_camera_uncond_without_camera_tokens():
    model = _tiny_model()
    z = torch.randn(1, 2, 4, 8)
    sigma = torch.tensor([0.5])
    masks = torch.zeros(1, 2, 4, 1)
    assets = torch.zeros(1, 0, 12)

    out = model(
        z,
        sigma,
        torch.zeros_like(z),
        torch.zeros_like(z),
        masks,
        masks,
        torch.ones_like(masks),
        assets,
        camera_pose_tokens=None,
        camera_attention_mask=None,
        camera_condition_kind=["camera_uncond"],
    )

    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_mode_a_with_empty_preserves_asset_and_adds_learned_slot():
    model = _tiny_model()
    assets = torch.randn(1, 5, 2, 4, 8)
    mask = torch.zeros(1, 5, 2, 4, dtype=torch.bool)
    mask[:, 0] = True

    tokens, out_mask = model._build_asset_condition(
        assets,
        mask,
        seq_len=2,
        num_patches=4,
        patch_grid=(2, 2),
        asset_condition_kind=["mode_a_with_empty"],
    )

    assert out_mask is not None
    assert out_mask.tolist() == [[True, True, False, False, False]]
    assert not torch.allclose(tokens[0, 0], model.empty_asset_embed.detach().reshape(-1))
    assert torch.allclose(tokens[0, 1], model.empty_asset_embed.detach().reshape(-1))


def test_timestep_embedder_uses_rae_gaussian_fourier_tokens():
    model = _tiny_model()
    state = model.state_dict()
    assert "t_embedder.W" in state
    assert "t_embedder.learnable_tokens" in state
    assert "t_embedder.mlp.0.weight" in state
    assert "t_embedder.mlp.2.weight" in state
    assert "t_embedder.linear_1.weight" not in state
    assert state["t_embedder.learnable_tokens"].shape == (4, 16)


def test_asset_kv_mask_is_none_when_no_padding():
    model = _tiny_model()
    assets = torch.randn(2, 3, 12)

    kv, mask = model._prepare_asset_kv(assets, None)
    assert kv is assets
    assert mask is None

    kv, mask = model._prepare_asset_kv(assets, torch.ones(2, 3, dtype=torch.bool))
    assert kv is assets
    assert mask is None


def test_asset_kv_mask_is_kept_only_for_padding():
    model = _tiny_model()
    assets = torch.randn(1, 3, 12)
    kv, mask = model._prepare_asset_kv(assets, torch.tensor([[True, False, True]]))
    assert kv is assets
    assert mask is not None
    assert mask.tolist() == [[True, False, True]]


def test_default_forward_does_not_anchor_legacy_null_kv():
    model = _tiny_model()
    torch.nn.init.normal_(model.proj_out.weight, std=0.02)
    out = model(*_inputs(kv=0))
    loss = out.square().mean()
    loss.backward()
    assert model.null_kv.grad is None


def test_legacy_null_kv_path_runs_and_can_receive_grad():
    model = _tiny_model()
    assets = torch.randn(1, 3, 12)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    tokens, out_mask, _ = model._build_sparse_asset_condition(
        assets,
        mask,
        seq_len=2,
        num_patches=4,
        patch_grid=(2, 2),
    )
    loss = tokens.square().mean()
    loss.backward()
    assert out_mask is not None
    assert out_mask.tolist() == [[True, False, False]]
    assert model.null_kv.grad is not None
    assert torch.isfinite(model.null_kv.grad).all()


def test_asset_kv_per_row_null_injection_no_length_change():
    """For mixed batches with some fully-empty rows, kv length must stay at
    num_tokens (no wasted slot appended for non-empty rows)."""
    model = _tiny_model()
    assets = torch.randn(2, 3, 12)
    mask = torch.tensor([[True, True, True], [False, False, False]])

    kv, kept_mask = model._prepare_asset_kv(assets, mask)

    # Length unchanged.
    assert kv.shape == (2, 3, 12)
    assert kept_mask is not None and kept_mask.shape == (2, 3)

    # Non-empty row 0 is untouched.
    assert torch.equal(kv[0], assets[0])
    assert kept_mask[0].tolist() == [True, True, True]

    # Legacy 3D-token path: empty row 1 replaces slot 0 with null_kv and
    # flips mask[1,0] to True. The 5D asset path uses all-False masks for
    # CFG uncond and empty_asset_embed for conditional Mode-B empty samples.
    null_value = model.null_kv.detach().reshape(-1)
    assert torch.allclose(kv[1, 0], null_value)
    # Slots 1..N-1 of the empty row keep the original (padding) values.
    assert torch.equal(kv[1, 1:], assets[1, 1:])
    assert kept_mask[1].tolist() == [True, False, False]


def test_asset_kv_per_row_null_kv_grad_flows_only_to_empty_rows():
    model = _tiny_model()
    assets = torch.randn(2, 3, 12)
    mask = torch.tensor([[True, True, True], [False, False, False]])
    kv, kept_mask = model._prepare_asset_kv(assets, mask)
    loss = kv.square().sum()
    loss.backward()
    # Legacy null_kv must accumulate grad from the empty row's slot 0.
    assert model.null_kv.grad is not None
    assert torch.isfinite(model.null_kv.grad).all()
    assert float(model.null_kv.grad.abs().sum()) > 0.0


def test_asset_kv_mixed_batch_forward_runs_and_gives_finite_outputs():
    """End-to-end: mixed batch with one empty and one full row produces
    finite, non-NaN outputs (the cross-attn softmax does not collapse on
    the empty row)."""
    model = _tiny_model()
    z_t, sigma, z_splat, scaffold, Mp, Ms, Md, _ = _inputs(batch=2, kv=3)
    mask = torch.tensor([[True, True, True], [False, False, False]])
    out = model(z_t, sigma, z_splat, scaffold, Mp, Ms, Md, _,
                encoder_attention_mask=mask)
    assert out.shape == (2, 2, 4, 8)
    assert torch.isfinite(out).all()


def test_normalize_denormalize_round_trip():
    model = _tiny_model()
    mu = torch.arange(8, dtype=torch.float32)
    sigma = torch.linspace(1.0, 2.0, 8)
    model.set_latent_stats(mu, sigma)
    z = torch.randn(1, 2, 4, 8)
    assert torch.allclose(model.denormalize(model.normalize(z)), z, atol=1e-6)


def test_sample_is_removed_legacy_interface():
    model = _tiny_model()
    z_splat = torch.zeros(1, 1, 4, 8)
    masks = torch.zeros(1, 1, 4, 1)
    assets = torch.zeros(1, 0, 12)

    with pytest.raises(RuntimeError, match=r"WanSceneFlow\.sample\(\) was removed"):
        model.sample(z_splat, z_splat, masks, masks, masks, assets, num_steps=1)


def test_sample_cfg_legacy_interface_is_removed():
    model = _tiny_model(qwen_dim=10)
    z_splat = torch.zeros(1, 1, 4, 8)
    masks = torch.zeros(1, 1, 4, 1)
    masks[:, :, 0] = 1.0
    assets = torch.randn(1, 1, 1, 4, 8)
    asset_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    text_tokens = torch.randn(1, 3, 10)
    text_mask = torch.tensor([[True, True, False]])

    with pytest.raises(RuntimeError, match="legacy CFG/scheduler path"):
        model.sample(
            z_splat,
            z_splat,
            1.0 - masks,
            torch.zeros_like(masks),
            masks,
            assets,
            num_steps=1,
            guidance_scale=3.0,
            asset_control_guidance_scale=2.0,
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            negative_text_tokens=torch.zeros_like(text_tokens),
            negative_text_attention_mask=text_mask,
            encoder_attention_mask=asset_mask,
            asset_condition_kind=["mode_a"],
        )


def test_state_dict_save_pretrained_round_trip(tmp_path):
    model = _tiny_model()
    args = _inputs(kv=1)
    out0 = model(*args)
    model.save_pretrained(tmp_path)
    loaded = WanSceneFlow.from_pretrained(tmp_path)
    out1 = loaded(*args)
    assert torch.allclose(out0, out1, atol=0.0, rtol=0.0)
