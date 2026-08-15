"""Why the generated sky dome was a flat blue wash, and what fixes it.

The v6 run produced a smooth gradient with no cloud structure after 50k steps.
Two things caused it, and both are measured rather than assumed:

* **Resolution.**  ``_sky_direction_grid`` spans elevation 0..90 deg over the
  atlas rows and azimuth 0..360 deg over its columns, so a 32x64 cell was
  2.81 x 5.62 deg.  Against the Waymo front camera's 49.2 x 35.0 deg, the sky
  band of a typical frame landed on about 22 cells -- confirmed by the run,
  which logged ``sky_token_loss_weight_mean = 0.0601`` and therefore 1.06% of
  2048 cells = 21.8.  Twenty-two colours cannot draw a cloud.
* **Supervision.**  Unobserved directions carry a spherical-completion target,
  not a direct camera observation.  ``sky_flow_loss`` is a weighted mean, so a
  region's gradient share is exactly its weight share.  At
  ``unobserved_weight=0.05`` with 1% observed, 82% of the sky gradient followed
  that low-confidence completion prior instead of measured image colours.
"""
from __future__ import annotations

import math

import pytest
import torch

import inference_scene_flow_pretrain as offline_inference
import train_scene_flow_pretrain as trainer
from dggt.models.scene_flow import (
    CURRENT_SKY_ATLAS_HW,
    CURRENT_SKY_GRID,
    CURRENT_SKY_REPRESENTATION_VERSION,
    CURRENT_SKY_TOKEN_DIM,
    RAEVideoSceneFlow,
)
from inference_scene_flow_pretrain import (
    CURRENT_PRETRAIN_SKY_CHECKPOINT_CONTRACT,
    _require_current_pretrain_sky_checkpoint_config,
)

# Waymo front camera, from data/scene_gauge/training.json row 000/0.
LOG_TAN_HALF_FOV = (-0.7816628217697144, -1.1531145572662354)
HFOV_DEG = 2.0 * math.degrees(math.atan(math.exp(LOG_TAN_HALF_FOV[0])))
VFOV_DEG = 2.0 * math.degrees(math.atan(math.exp(LOG_TAN_HALF_FOV[1])))

SMALL_MODEL = dict(
    num_attention_heads=2,
    attention_head_dim=16,
    num_layers=2,
    ddt_head_depth=1,
    ddt_head_dim=64,
    ddt_head_heads=2,
    base_model_depth=1,
    repa_layer_depth=1,
)


# ------------------------------------------------------------------ #
# Resolution                                                          #
# ------------------------------------------------------------------ #
def test_the_atlas_resolves_the_camera_finely_enough_for_texture() -> None:
    atlas_h, atlas_w = trainer.DEFAULT_SKY_ATLAS_HW
    cell_elevation = 90.0 / atlas_h
    cell_azimuth = 360.0 / atlas_w
    rows_in_view = VFOV_DEG / cell_elevation
    cols_in_view = HFOV_DEG / cell_azimuth
    # A 20% sky band -- the training sky_mask_target_frac -- must land on
    # enough cells to carry structure rather than a handful of flat blocks.
    sky_rows = rows_in_view * 0.2
    assert sky_rows * cols_in_view > 100.0, (
        f"the sky band covers only {sky_rows * cols_in_view:.0f} atlas cells"
    )
    # The old 32x64 is what failed; keep the comparison in the test so the
    # threshold above is not mistaken for an arbitrary number.
    old = (VFOV_DEG / (90.0 / 32)) * 0.2 * (HFOV_DEG / (360.0 / 64))
    assert old == pytest.approx(21.8, abs=1.0)


def test_the_token_count_did_not_grow_with_the_atlas() -> None:
    """Resolution came from a wider patch, not from more sequence."""

    atlas_h, atlas_w = trainer.DEFAULT_SKY_ATLAS_HW
    grid_h, grid_w = trainer.DEFAULT_SKY_GRID
    assert (grid_h, grid_w) == (16, 32)
    assert atlas_h // trainer.SKY_PATCH_SIZE == grid_h
    assert atlas_w // trainer.SKY_PATCH_SIZE == grid_w
    assert trainer.SKY_TOKEN_DIM == 3 * trainer.SKY_PATCH_SIZE ** 2


def test_pack_and_decode_round_trip_exactly() -> None:
    torch.manual_seed(0)
    atlas = torch.rand((2, 3, *trainer.DEFAULT_SKY_ATLAS_HW))
    tokens = trainer.pack_sky_rgb_atlas(atlas)
    assert tokens.shape == (2, 512, trainer.SKY_TOKEN_DIM)
    decoded = trainer.decode_sky_patch_tokens(tokens)
    restored = ((decoded + 1.0) * 0.5).reshape(2, *trainer.DEFAULT_SKY_ATLAS_HW, 3)
    assert torch.allclose(restored.permute(0, 3, 1, 2), atlas, atol=1e-6)


def test_a_stale_token_width_is_rejected_by_name() -> None:
    """An old checkpoint must fail loudly, not unpack into a scrambled atlas."""

    with pytest.raises(ValueError, match=trainer.SKY_REPRESENTATION_VERSION):
        trainer.decode_sky_patch_tokens(torch.zeros((1, 512, 12)))
    with pytest.raises(ValueError, match=trainer.SKY_REPRESENTATION_VERSION):
        trainer.pack_sky_rgb_atlas(torch.zeros((1, 3, 32, 64)))


def test_the_loss_weight_uses_the_same_patch_layout_as_the_atlas() -> None:
    """One weight per RGB output channel, or partly visible patches leak."""

    atlas_h, atlas_w = trainer.DEFAULT_SKY_ATLAS_HW
    observed = torch.zeros((1, 1, atlas_h, atlas_w))
    observed[:, :, : atlas_h // 4, : atlas_w // 4] = 1.0
    weight = trainer.pack_sky_atlas_loss_weight(observed, unobserved_weight=0.005)
    assert weight.shape == (1, 512, trainer.SKY_TOKEN_DIM)
    # The weight is a per-cell map, not a standardized sky token, so it is
    # unpacked with the standardization left off.
    unpacked = trainer.decode_sky_patch_tokens(weight * 2.0 - 1.0, standardized=False)
    unpacked = ((unpacked + 1.0) * 0.5).reshape(1, atlas_h, atlas_w, 3)[..., 0]
    expected = observed[:, 0] + (1.0 - observed[:, 0]) * 0.005
    assert torch.allclose(unpacked, expected, atol=1e-6)


# ------------------------------------------------------------------ #
# Supervision balance                                                 #
# ------------------------------------------------------------------ #
def test_the_observed_sky_now_owns_most_of_the_gradient() -> None:
    """``sky_flow_loss`` is a weighted mean, so weight share == gradient share."""

    atlas_h, atlas_w = trainer.DEFAULT_SKY_ATLAS_HW
    observed = torch.zeros((1, 1, atlas_h, atlas_w))
    # The v6 run observed 1.06% of the atlas.
    visible_rows = max(1, round(0.0106 * atlas_h * atlas_w / atlas_w))
    observed[:, :, :visible_rows, :] = 1.0
    fraction = float(observed.mean())

    def observed_share(unobserved_weight: float) -> float:
        weight = trainer.pack_sky_atlas_loss_weight(
            observed, unobserved_weight=unobserved_weight
        )
        return fraction / float(weight.mean())

    assert observed_share(0.05) < 0.25          # what v6 ran
    assert observed_share(trainer.DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT) > 0.6
    assert trainer.DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT == pytest.approx(0.005)


def test_the_sky_view_term_carries_the_same_weight_as_its_siblings() -> None:
    """It is the dome's only pixel-space supervision; 0.1 left it invisible."""

    assert trainer.SKY_VIEW_LAMBDA_DEFAULT == pytest.approx(1.0)
    assert trainer.SKY_VIEW_LAMBDA_DEFAULT == pytest.approx(
        trainer.RGB_RENDER_LAMBDA_DEFAULT
    )


# ------------------------------------------------------------------ #
# The model side                                                      #
# ------------------------------------------------------------------ #
def test_the_model_accepts_the_current_sky_representation() -> None:
    model = RAEVideoSceneFlow(
        sky_token_dim=trainer.SKY_TOKEN_DIM,
        sky_atlas_hw=trainer.DEFAULT_SKY_ATLAS_HW,
        sky_grid=trainer.DEFAULT_SKY_GRID,
        sky_representation_version=trainer.SKY_REPRESENTATION_VERSION,
        **SMALL_MODEL,
    )
    assert model.sky_gen_proj.in_features == trainer.SKY_TOKEN_DIM
    assert model.sky_gen_decoder[-1].out_features == trainer.SKY_TOKEN_DIM


def test_the_model_defaults_to_the_current_sky_representation() -> None:
    model = RAEVideoSceneFlow(**SMALL_MODEL)

    assert model.config.sky_representation_version == trainer.SKY_REPRESENTATION_VERSION
    assert model.config.sky_token_dim == trainer.SKY_TOKEN_DIM
    assert tuple(model.config.sky_atlas_hw) == trainer.DEFAULT_SKY_ATLAS_HW
    assert tuple(model.config.sky_grid) == trainer.DEFAULT_SKY_GRID
    assert CURRENT_SKY_REPRESENTATION_VERSION == trainer.SKY_REPRESENTATION_VERSION
    assert CURRENT_SKY_TOKEN_DIM == trainer.SKY_TOKEN_DIM
    assert CURRENT_SKY_ATLAS_HW == trainer.DEFAULT_SKY_ATLAS_HW
    assert CURRENT_SKY_GRID == trainer.DEFAULT_SKY_GRID


def test_an_older_sky_representation_still_constructs() -> None:
    """Generic construction remains available for explicit inspection/conversion."""

    model = RAEVideoSceneFlow(
        sky_token_dim=12,
        sky_atlas_hw=(32, 64),
        sky_grid=(16, 32),
        sky_representation_version="rgb_patch_teacher_anchor_v3",
        **SMALL_MODEL,
    )
    assert model.sky_gen_proj.in_features == 12


def test_from_pretrained_rejects_v3_before_loading_weights(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RAEVideoSceneFlow(
        sky_token_dim=12,
        sky_atlas_hw=(32, 64),
        sky_grid=(16, 32),
        sky_representation_version="rgb_patch_teacher_anchor_v3",
        **SMALL_MODEL,
    )
    model.save_pretrained(tmp_path)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "old sky weights must be rejected before reading the state dict"
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported old sky checkpoint weights.*requires the complete.*v5",
    ):
        RAEVideoSceneFlow.from_pretrained(tmp_path)


def test_from_pretrained_loads_the_current_v4_contract(tmp_path) -> None:
    model = RAEVideoSceneFlow(**SMALL_MODEL)
    model.save_pretrained(tmp_path)

    loaded = RAEVideoSceneFlow.from_pretrained(tmp_path)

    assert loaded.config.sky_representation_version == CURRENT_SKY_REPRESENTATION_VERSION
    assert loaded.config.sky_token_dim == CURRENT_SKY_TOKEN_DIM
    assert tuple(loaded.config.sky_atlas_hw) == CURRENT_SKY_ATLAS_HW
    assert tuple(loaded.config.sky_grid) == CURRENT_SKY_GRID
    torch.testing.assert_close(
        loaded.sky_gen_decoder[-1].weight,
        model.sky_gen_decoder[-1].weight,
    )


def test_offline_inference_rejects_v3_sky_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "sky_representation_version": "rgb_patch_teacher_anchor_v3",
        "sky_atlas_hw": (32, 64),
        "sky_grid": (16, 32),
        "sky_token_dim": 12,
    }
    monkeypatch.setattr(
        offline_inference.torch,
        "load",
        lambda *_args, **_kwargs: {
            "scene_flow_config": config,
            "flow_schedule_config": {},
        },
    )
    monkeypatch.setattr(
        offline_inference,
        "WanSceneFlow",
        lambda **_kwargs: pytest.fail("v3 must be rejected before model construction"),
    )

    with pytest.raises(
        ValueError,
        match=r"old_v3\.pt.*requires the complete.*v5.*Old v3 sky checkpoints",
    ):
        offline_inference._require_current_checkpoint(
            "old_v3.pt",
            device=torch.device("cpu"),
            use_ema=True,
            args=object(),
        )


def test_offline_inference_requires_every_v4_sky_contract_field() -> None:
    current = dict(CURRENT_PRETRAIN_SKY_CHECKPOINT_CONTRACT)
    _require_current_pretrain_sky_checkpoint_config(current, "current_v4.pt")

    for missing in current:
        incomplete = dict(current)
        incomplete.pop(missing)
        with pytest.raises(ValueError, match=missing):
            _require_current_pretrain_sky_checkpoint_config(
                incomplete,
                f"missing_{missing}.pt",
            )


def test_a_mismatched_atlas_and_token_width_is_rejected() -> None:
    for kwargs in (
        dict(sky_token_dim=trainer.SKY_TOKEN_DIM, sky_atlas_hw=(32, 64)),
        dict(sky_token_dim=12, sky_atlas_hw=trainer.DEFAULT_SKY_ATLAS_HW),
    ):
        with pytest.raises(ValueError, match="requires sky_token_dim"):
            RAEVideoSceneFlow(
                sky_grid=trainer.DEFAULT_SKY_GRID,
                sky_representation_version=trainer.SKY_REPRESENTATION_VERSION,
                **kwargs,
                **SMALL_MODEL,
            )


def test_the_trainer_can_only_build_the_current_representation() -> None:
    parser = trainer.build_argparser()
    action = next(
        a for a in parser._actions if a.dest == "sky_representation_version"
    )
    assert tuple(action.choices) == (trainer.SKY_REPRESENTATION_VERSION,)
    assert parser.get_default("sky_atlas_h") == trainer.DEFAULT_SKY_ATLAS_HW[0]
    assert parser.get_default("sky_atlas_w") == trainer.DEFAULT_SKY_ATLAS_HW[1]


# ------------------------------------------------------------------ #
# End to end through the real target builder and renderer             #
# ------------------------------------------------------------------ #
def _camera(seq_len: int, height: int, width: int):
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, seq_len, 1, 1)
    fx = (width / 2) / math.exp(LOG_TAN_HALF_FOV[0])
    fy = (height / 2) / math.exp(LOG_TAN_HALF_FOV[1])
    intrinsics = torch.tensor(
        [[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3).repeat(1, seq_len, 1, 1)
    return extrinsics, intrinsics


def test_a_structured_sky_survives_the_round_trip_through_the_atlas() -> None:
    """The whole point: build the target, pack, decode, render, compare.

    The synthetic sky is a horizontal ramp with a fine vertical stripe -- the
    stripe is the cloud stand-in.  At 32x64 it is averaged away; at the current
    resolution it comes back.
    """

    seq_len, height, width = 2, 350, 518
    yy = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
    xx = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    stripe = (torch.sin(xx * 60.0) * 0.5 + 0.5) * 0.4
    frame = (yy * 0.3 + stripe).expand(1, 3, height, width).clone()
    images = frame.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).clamp(0, 1)
    masks = torch.zeros((1, seq_len, 1, height, width))
    masks[:, :, :, : height // 3] = 1.0
    extrinsics, intrinsics = _camera(seq_len, height, width)

    errors = {}
    for atlas_hw, patch in (((32, 64), 2), (trainer.DEFAULT_SKY_ATLAS_HW, trainer.SKY_PATCH_SIZE)):
        atlas, _ = trainer.build_sky_atlas_from_images(
            images, masks, atlas_hw=atlas_hw,
            extrinsics=extrinsics, intrinsics=intrinsics,
        )
        assert tuple(atlas.shape) == (1, 3, *atlas_hw)
        flat = (
            torch.nn.functional.pixel_unshuffle(atlas * 2.0 - 1.0, patch)
            .permute(0, 2, 3, 1)
            .reshape(1, -1, 3 * patch * patch)
        )
        flat = torch.nn.functional.pixel_shuffle(
            flat.reshape(1, atlas_hw[0] // patch, atlas_hw[1] // patch, -1)
            .permute(0, 3, 1, 2),
            patch,
        ).permute(0, 2, 3, 1).reshape(1, atlas_hw[0] * atlas_hw[1], 3)
        background = trainer.sky_tokens_to_background(
            flat, seq_len=seq_len, height=height, width=width,
            grid_h=atlas_hw[0], grid_w=atlas_hw[1],
            extrinsics=extrinsics, intrinsics=intrinsics,
        )
        grid = trainer._sky_background_image_grid(background, seq_len)
        assert tuple(grid.shape) == (seq_len, 3, height, width)
        sky = masks[0, 0]
        errors[atlas_hw] = float(
            ((grid[0] - images[0, 0]).abs() * sky).sum() / sky.sum().clamp_min(1) / 3
        )

    coarse = errors[(32, 64)]
    fine = errors[trainer.DEFAULT_SKY_ATLAS_HW]
    assert fine < 0.5 * coarse, f"32x64 {coarse:.4f} vs current {fine:.4f}"
