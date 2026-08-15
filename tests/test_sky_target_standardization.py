"""The sky flow target now lives in the same units as the scene latent.

The v6/s2.9.1 run generated a sky whose colour was right and whose texture was
absent: over the 3800 steps after the world-feedback ramp opened,
``loss_sky_view_charbonnier`` fell 4.88%/1k step while
``loss_sky_view_high_frequency`` -- the only term that measures texture -- moved
0.18%/1k.  The cause is measurable and is not the atlas resolution, which v4
already fixed: the scene latent is standardized (the run logs
``sample_latent_target_std`` at 1.0011) and the sky token was not.  Packed as
raw ``rgb * 2 - 1`` it carries a large mean and a std well under one, and per
channel blue is the worst -- nearly saturated with the smallest spread, which is
exactly where cloud contrast lives.  Flow matching adds ``eps ~ N(0, I)`` at
unit scale on top of that, so at the training sigma the cloud signal sat far
deeper in noise than anything the scene had to recover.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import train_scene_flow_pretrain as trainer
from dggt.losses.rgb_render_loss import (
    rgb_render_loss_ramp,
    should_apply_rgb_render_loss,
    should_apply_sky_view_loss,
    sky_view_loss_ramp,
)

SIGMA_MEAN = 0.8636  # train/sigma_mean, waver + shift 10
# Waymo front camera, from data/scene_gauge/training.json row 000/0.
LOG_TAN_HALF_FOV = (-0.7816628217697144, -1.1531145572662354)
HFOV_DEG = 2.0 * math.degrees(math.atan(math.exp(LOG_TAN_HALF_FOV[0])))
VFOV_DEG = 2.0 * math.degrees(math.atan(math.exp(LOG_TAN_HALF_FOV[1])))


def _snr(std: float, sigma: float = SIGMA_MEAN) -> float:
    """Signal-to-noise of ``z_clean`` inside ``z_t`` at this sigma."""
    return (1.0 - sigma) * std / sigma


# ------------------------------------------------------------------ #
# The constants and the packing they apply to                         #
# ------------------------------------------------------------------ #
def test_the_channel_constants_expand_over_the_patch_not_across_it() -> None:
    """``pixel_unshuffle`` lays the token out as ``c * patch**2 + sub``."""

    mean, std = trainer.sky_token_channel_stats()
    assert mean.shape == (trainer.SKY_TOKEN_DIM,)
    assert std.shape == (trainer.SKY_TOKEN_DIM,)
    block = trainer.SKY_PATCH_SIZE * trainer.SKY_PATCH_SIZE
    for channel in range(trainer.SKY_RGB_DIM):
        span = slice(channel * block, (channel + 1) * block)
        assert torch.allclose(
            mean[span], torch.full((block,), trainer.SKY_TOKEN_CHANNEL_MEAN[channel])
        )
        assert torch.allclose(
            std[span], torch.full((block,), trainer.SKY_TOKEN_CHANNEL_STD[channel])
        )


def test_the_constants_land_on_the_right_channel_of_a_solid_atlas() -> None:
    """A pure-red atlas must standardize using the red constants only."""

    atlas = torch.zeros((1, 3, *trainer.DEFAULT_SKY_ATLAS_HW))
    atlas[:, 0] = 1.0
    tokens = trainer.pack_sky_rgb_atlas(atlas)
    block = trainer.SKY_PATCH_SIZE * trainer.SKY_PATCH_SIZE
    expected_r = (1.0 - trainer.SKY_TOKEN_CHANNEL_MEAN[0]) / trainer.SKY_TOKEN_CHANNEL_STD[0]
    expected_b = (-1.0 - trainer.SKY_TOKEN_CHANNEL_MEAN[2]) / trainer.SKY_TOKEN_CHANNEL_STD[2]
    assert torch.allclose(tokens[..., :block], torch.tensor(expected_r), atol=1e-5)
    assert torch.allclose(tokens[..., 2 * block :], torch.tensor(expected_b), atol=1e-5)


def test_pack_and_decode_still_round_trip_exactly() -> None:
    torch.manual_seed(0)
    atlas = torch.rand((2, 3, *trainer.DEFAULT_SKY_ATLAS_HW))
    tokens = trainer.pack_sky_rgb_atlas(atlas)
    decoded = trainer.decode_sky_patch_tokens(tokens)
    recovered = ((decoded + 1.0) * 0.5).reshape(2, *trainer.DEFAULT_SKY_ATLAS_HW, 3)
    assert torch.allclose(recovered, atlas.permute(0, 2, 3, 1), atol=1e-5)


def test_standardization_can_be_switched_off_for_measuring_the_constants() -> None:
    torch.manual_seed(1)
    atlas = torch.rand((1, 3, *trainer.DEFAULT_SKY_ATLAS_HW))
    raw = trainer.pack_sky_rgb_atlas(atlas, standardize=False)
    std_tokens = trainer.pack_sky_rgb_atlas(atlas)
    mean, std = trainer.sky_token_channel_stats()
    assert torch.allclose(std_tokens, (raw - mean) / std, atol=1e-6)
    assert torch.allclose(
        trainer.decode_sky_patch_tokens(raw, standardized=False),
        trainer.decode_sky_patch_tokens(std_tokens),
        atol=1e-5,
    )


def test_standardization_is_a_per_dimension_affine_so_the_weight_still_aligns() -> None:
    """The loss weight is per atlas cell; a per-dim affine cannot disturb it."""

    atlas_h, atlas_w = trainer.DEFAULT_SKY_ATLAS_HW
    observed = torch.zeros((1, 1, atlas_h, atlas_w))
    observed[:, :, 40:60, 100:140] = 1.0
    weight = trainer.pack_sky_atlas_loss_weight(observed, unobserved_weight=0.005)
    unpacked = trainer.decode_sky_patch_tokens(weight * 2.0 - 1.0, standardized=False)
    unpacked = ((unpacked + 1.0) * 0.5).reshape(1, atlas_h, atlas_w, 3)[..., 0]
    expected = observed[:, 0] + (1.0 - observed[:, 0]) * 0.005
    assert torch.allclose(unpacked, expected, atol=1e-6)


# ------------------------------------------------------------------ #
# What standardization buys                                           #
# ------------------------------------------------------------------ #
def test_the_target_now_matches_the_scene_latent_scale() -> None:
    """Blue is the channel that decides whether a cloud is visible at all."""

    raw_std = trainer.SKY_TOKEN_CHANNEL_STD
    blue_before, blue_after = _snr(raw_std[2]), _snr(1.0)
    # Blue sat under 1:16 in the noise; the scene latent, at std 1.0011, sits
    # at 1:6, and that gap is what standardization closes.
    assert 1.0 / blue_before > 16.0, f"blue was 1:{1 / blue_before:.0f}"
    assert 1.0 / blue_after == pytest.approx(6.3, abs=0.2)
    assert blue_after / blue_before > 3.0, (
        f"blue only improved {blue_after / blue_before:.2f}x"
    )
    # The three channels start at three different scales and end at one.
    assert max(raw_std) / min(raw_std) > 1.4


def test_zero_init_of_the_sky_head_now_predicts_the_prior_mean() -> None:
    """``sky_gen_decoder[-1]`` is zero-initialized, so step 0 predicts zero.

    Standardized, that zero decodes to the dataset mean sky.  Unstandardized it
    decoded to ``rgb = 0.5`` grey while the real mean sky is blue, so the head
    started a fixed distance away from the answer in every channel.
    """
    zero = torch.zeros((1, 512, trainer.SKY_TOKEN_DIM))
    decoded = trainer.decode_sky_patch_tokens(zero)
    rgb = ((decoded + 1.0) * 0.5).reshape(-1, 3).mean(dim=0)
    expected = [(m + 1.0) * 0.5 for m in trainer.SKY_TOKEN_CHANNEL_MEAN]
    assert torch.allclose(rgb, torch.tensor(expected), atol=1e-5)
    assert float(rgb[2]) > float(rgb[0]) + 0.2, "the mean sky it starts from is blue"
    # Unstandardized, the same zero prediction decoded to flat grey.
    grey = trainer.decode_sky_patch_tokens(zero, standardized=False)
    assert torch.allclose(((grey + 1.0) * 0.5), torch.full_like(grey, 0.5), atol=1e-6)


# ------------------------------------------------------------------ #
# Separated normalization of the sky flow loss                        #
# ------------------------------------------------------------------ #
def _sky_target(v_gt: torch.Tensor, weight, observation) -> SimpleNamespace:
    return SimpleNamespace(v_gt=v_gt, loss_weight=weight, observation=observation)


def _observation(batch: int, tokens: int, dim: int, observed_cells: int) -> torch.Tensor:
    obs = torch.zeros((batch, tokens, dim))
    flat = obs.reshape(batch, -1)
    flat[:, :observed_cells] = 1.0
    return flat.reshape(batch, tokens, dim)


@pytest.mark.parametrize("observed_cells", [64, 512, 4096])
def test_the_observed_share_no_longer_moves_with_the_clip_sky_fraction(observed_cells) -> None:
    """One weighted mean made the split depend on how much sky a clip had."""

    torch.manual_seed(2)
    batch, tokens, dim = 1, 512, trainer.SKY_TOKEN_DIM
    pred = torch.zeros((batch, tokens, dim), requires_grad=True)
    v_gt = torch.ones((batch, tokens, dim))
    obs = _observation(batch, tokens, dim, observed_cells)
    weight = obs + (1.0 - obs) * trainer.DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT
    beta = trainer.DEFAULT_SKY_UNOBSERVED_LOSS_BETA

    loss = trainer.sky_flow_loss(pred, _sky_target(v_gt, weight, obs), None, unobserved_beta=beta)
    loss.backward()
    grad = pred.grad.abs()
    share = float((grad * obs).sum() / grad.sum())
    assert share == pytest.approx(1.0 / (1.0 + beta), abs=1e-4), (
        f"observed share {share:.4f} at {observed_cells} cells"
    )


def test_the_old_weighted_mean_did_move_with_the_sky_fraction() -> None:
    """Without an observation map the loss falls back, and the split drifts."""

    shares = []
    for observed_cells in (64, 4096):
        pred = torch.zeros((1, 512, trainer.SKY_TOKEN_DIM), requires_grad=True)
        v_gt = torch.ones_like(pred)
        obs = _observation(1, 512, trainer.SKY_TOKEN_DIM, observed_cells)
        weight = obs + (1.0 - obs) * trainer.DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT
        loss = trainer.sky_flow_loss(pred, _sky_target(v_gt, weight, None), None)
        loss.backward()
        shares.append(float((pred.grad.abs() * obs).sum() / pred.grad.abs().sum()))
    assert abs(shares[0] - shares[1]) > 0.2, (
        f"expected the old split to drift, got {shares}"
    )


def test_beta_zero_supervises_only_the_observed_atlas() -> None:
    pred = torch.zeros((1, 512, trainer.SKY_TOKEN_DIM), requires_grad=True)
    v_gt = torch.ones_like(pred)
    obs = _observation(1, 512, trainer.SKY_TOKEN_DIM, 256)
    weight = obs + (1.0 - obs) * 0.005
    trainer.sky_flow_loss(pred, _sky_target(v_gt, weight, obs), None, unobserved_beta=0.0).backward()
    assert float((pred.grad.abs() * (1.0 - obs)).sum()) == 0.0


def test_a_clip_with_no_sky_still_supervises_nothing() -> None:
    """Otherwise the completion prior teaches where there is no measurement."""

    pred = torch.zeros((1, 512, trainer.SKY_TOKEN_DIM), requires_grad=True)
    v_gt = torch.ones_like(pred)
    obs = torch.zeros_like(pred)
    weight = torch.zeros_like(pred)
    loss = trainer.sky_flow_loss(pred, _sky_target(v_gt, weight, obs), None)
    assert float(loss) == 0.0


def test_a_negative_beta_is_rejected() -> None:
    pred = torch.zeros((1, 4, trainer.SKY_TOKEN_DIM))
    target = _sky_target(torch.ones_like(pred), torch.ones_like(pred), torch.ones_like(pred))
    with pytest.raises(ValueError, match="beta"):
        trainer.sky_flow_loss(pred, target, None, unobserved_beta=-0.1)


# ------------------------------------------------------------------ #
# The sky-view gate is independent of the 3DGS render gate            #
# ------------------------------------------------------------------ #
def _args(**over):
    base = dict(
        rgb_render_start_step=5000,
        rgb_render_warmup_steps=5000,
        rgb_render_every=1,
        sky_view_start_step=trainer.SKY_VIEW_START_STEP_DEFAULT,
        sky_view_warmup_steps=trainer.SKY_VIEW_WARMUP_STEPS_DEFAULT,
        lambda_rgb_render=trainer.RGB_RENDER_LAMBDA_DEFAULT,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_the_sky_view_loss_starts_before_the_3dgs_render() -> None:
    args = _args()
    assert not should_apply_sky_view_loss(args, 1499, training=True)
    assert should_apply_sky_view_loss(args, 1500, training=True)
    # ...while the 3DGS render is still off, which is the whole point.
    assert not should_apply_rgb_render_loss(args, 1500, training=True)
    assert should_apply_rgb_render_loss(args, 5000, training=True)


def test_the_sky_view_loss_does_not_start_at_step_zero() -> None:
    """It reads the gauge FOV for ray directions and that is 5.7 deg off early.

    An atlas cell is 1.41 deg of azimuth, so a step-0 start would teach the
    atlas through rays four cells away from where they belong.
    """
    assert trainer.SKY_VIEW_START_STEP_DEFAULT >= 1000


def test_the_sky_view_ramp_is_zero_before_its_own_start() -> None:
    args = _args()
    assert sky_view_loss_ramp(args, 0) == 0.0
    assert sky_view_loss_ramp(args, 1499) == 0.0
    assert sky_view_loss_ramp(args, 1500) == 0.0
    assert sky_view_loss_ramp(args, 4000) == pytest.approx(0.5, abs=1e-6)
    assert sky_view_loss_ramp(args, 6500) == 1.0
    # It is a different schedule from the render ramp, not an alias of it.
    assert sky_view_loss_ramp(args, 4000) != rgb_render_loss_ramp(args, 4000)


def test_neither_gate_fires_outside_training() -> None:
    args = _args()
    assert not should_apply_sky_view_loss(args, 9000, training=False)


# ------------------------------------------------------------------ #
# Diagnostics and guards                                              #
# ------------------------------------------------------------------ #
def test_the_validation_mae_is_reported_in_rgb_units() -> None:
    """Otherwise the number moves when the representation is rescaled."""

    torch.manual_seed(3)
    atlas = torch.rand((1, 3, *trainer.DEFAULT_SKY_ATLAS_HW))
    noisy = (atlas + 0.02).clamp(0.0, 1.0)
    gt = trainer.pack_sky_rgb_atlas(atlas)
    pred = trainer.pack_sky_rgb_atlas(noisy)
    metrics = trainer.sky_token_validation_metrics(pred, gt, prefix="s")
    direct = float((noisy - atlas).abs().mean())
    assert metrics["s_rgb_mae"] == pytest.approx(direct, rel=0.02), (
        f"{metrics['s_rgb_mae']:.5f} vs {direct:.5f} in [0,1] RGB"
    )


def test_both_sky_renderers_reject_a_packed_patch_token() -> None:
    """[..., :3] of a packed token is three red subpixels, not an RGB triple."""

    from dggt.losses.rgb_render_loss import (
        sky_tokens_to_background as render_loss_sky_background,
    )

    packed = torch.zeros((1, 512, trainer.SKY_TOKEN_DIM))
    extrinsics = torch.eye(4).view(1, 1, 4, 4).expand(1, 2, 4, 4).clone()
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, 3, 3).clone()
    with pytest.raises(ValueError, match="decoded RGB atlas"):
        trainer.sky_tokens_to_background(
            packed, seq_len=2, height=8, width=8, grid_h=128, grid_w=256,
            extrinsics=extrinsics, intrinsics=intrinsics,
        )
    with pytest.raises(ValueError, match="decoded RGB atlas"):
        render_loss_sky_background(
            packed, seq_len=2, height=8, width=8, grid_h=128, grid_w=256,
            world_to_camera=extrinsics, intrinsics=intrinsics,
        )


def test_a_v4_checkpoint_cannot_resume_into_v5() -> None:
    """v5 is shape-identical to v4 and differs only in what the numbers mean."""

    assert "sky_representation_version" in trainer.PRETRAIN_RESUME_CRITICAL_ARGS
    assert trainer.SKY_REPRESENTATION_VERSION.endswith("v5")
    from dggt.models.scene_flow import (
        CURRENT_SKY_REPRESENTATION_VERSION,
        SKY_REPRESENTATION_CONTRACTS,
    )
    assert CURRENT_SKY_REPRESENTATION_VERSION == trainer.SKY_REPRESENTATION_VERSION
    # Same shapes as v4 -- which is exactly why the version string has to gate.
    assert (
        SKY_REPRESENTATION_CONTRACTS[CURRENT_SKY_REPRESENTATION_VERSION]
        == SKY_REPRESENTATION_CONTRACTS["rgb_patch_teacher_anchor_v4"]
    )


def test_the_loss_weight_defaults_land_where_they_were_sized() -> None:
    assert trainer.DEFAULT_LAMBDA_SKY_FLOW == 0.5
    assert trainer.SKY_VIEW_HIGH_FREQUENCY_WEIGHT_DEFAULT == 1.0
    assert trainer.DEFAULT_SKY_UNOBSERVED_LOSS_BETA == 0.05
    # Standardization multiplies the sky flow term by about 4.1x on its own, so
    # lambda 1.0 would have overshot the sky past its 5.2% token share.
    projected = 0.0229781 * 4.11 * 1.15 * trainer.DEFAULT_LAMBDA_SKY_FLOW
    total = 1.3493 - 0.0022978 + projected
    assert 0.03 < projected / total < 0.05, f"sky flow would be {projected / total:.3%}"


# ------------------------------------------------------------------ #
# The signed texture diagnostic                                       #
# ------------------------------------------------------------------ #
def _sky_view_logs(atlas: torch.Tensor, images: torch.Tensor, pose: torch.Tensor):
    _, logs = trainer.generated_sky_view_reconstruction_loss(
        vggt_model=None,
        sky_latent=trainer.pack_sky_rgb_atlas(atlas),
        images=images,
        sky_mask=torch.ones((*images.shape[:2], 1, *images.shape[-2:])),
        render_pose_enc_dggt=pose,
        lpips_model=None,
        lpips_weight=0.0,
        high_frequency_weight=trainer.SKY_VIEW_HIGH_FREQUENCY_WEIGHT_DEFAULT,
        defer_log_values=False,
        collect_logs=True,
    )
    return logs


def test_the_detail_ratio_separates_a_flat_sky_from_a_noisy_one() -> None:
    """``|grad(pred) - grad(target)|`` alone cannot tell those two apart."""

    torch.manual_seed(11)
    seq, height, width = 2, 48, 64
    pose = torch.zeros((1, seq, 9))
    pose[..., 6] = 1.0                      # identity rotation quaternion
    # pose_enc carries the field of view as an angle: vertical then horizontal.
    pose[..., 7] = math.radians(VFOV_DEG)
    pose[..., 8] = math.radians(HFOV_DEG)

    truth = torch.rand((1, 3, *trainer.DEFAULT_SKY_ATLAS_HW)) * 0.15 + 0.6
    # Render the true atlas through the real renderer so the target images and
    # the prediction live in the same space.
    extrinsics, intrinsics = trainer._predict_camera_mats(
        pose, (height, width), torch.device("cpu")
    )
    images = trainer.sky_tokens_to_background(
        trainer.decode_sky_patch_tokens(trainer.pack_sky_rgb_atlas(truth)),
        seq_len=seq, height=height, width=width,
        grid_h=trainer.DEFAULT_SKY_ATLAS_HW[0], grid_w=trainer.DEFAULT_SKY_ATLAS_HW[1],
        extrinsics=extrinsics.unsqueeze(0), intrinsics=intrinsics.unsqueeze(0),
    ).permute(0, 3, 1, 2).unsqueeze(0)

    flat = truth.mean(dim=(2, 3), keepdim=True).expand_as(truth).contiguous()
    noisy = (truth + torch.randn_like(truth) * 0.15).clamp(0.0, 1.0)

    exact_logs = _sky_view_logs(truth, images, pose)
    flat_logs = _sky_view_logs(flat, images, pose)
    noisy_logs = _sky_view_logs(noisy, images, pose)

    exact = float(exact_logs["sky_view_detail_ratio"])
    flat_ratio = float(flat_logs["sky_view_detail_ratio"])
    noisy_ratio = float(noisy_logs["sky_view_detail_ratio"])
    assert exact == pytest.approx(1.0, abs=0.05), exact
    assert flat_ratio < 0.2, f"a flat sky should carry almost no gradient, got {flat_ratio}"
    assert noisy_ratio > 1.2, f"a speckled sky should carry too much, got {noisy_ratio}"

    # And the reason it is needed: the unsigned term ranks the flat sky as
    # *better* than the correct one, because being smooth costs it less than
    # having texture in slightly the wrong place.
    assert float(flat_logs["loss_sky_view_high_frequency"]) < float(
        noisy_logs["loss_sky_view_high_frequency"]
    )


def test_the_render_background_does_not_depend_on_the_sky_view_gate() -> None:
    """The 3DGS render takes the decoded sky atlas as its background.

    Decoding it inside the sky-view loss branch would make the render fall back
    to a black background on any step the loss is not due -- a silent change to
    a different loss.  Guard the source so the two stay separate.
    """
    import inspect

    source = inspect.getsource(trainer.train_step)
    decode_at = source.index("selected_sky_tokens = decode_sky_patch_tokens")
    gate_at = source.index("if sky_view_active and selected_sky_latent is not None:")
    assert decode_at < gate_at, (
        "the sky atlas must be decoded before the sky-view loss gate"
    )
    prefix = source[:decode_at]
    guard = prefix[prefix.rindex("\n", 0, prefix.rindex("if ")) :]
    assert "sky_view_active" not in guard, (
        f"the decode is gated on the sky-view loss: {guard.strip()!r}"
    )
