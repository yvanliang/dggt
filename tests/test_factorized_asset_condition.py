from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from dggt.models.canonical_asset_encoder import CanonicalAssetEncoder
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.factorized_asset_condition import (
    PLACEMENT_STATE_DIM,
    FactorizedAssetCondition,
    alpha_to_patch_mask,
    build_factorized_asset_condition,
    canonicalize_asset_reference,
    interpolate_box_keyframes,
    object_to_anchor_from_center_yaw,
    project_anchor_boxes_to_patch_bboxes,
    resize_crop_intrinsics_to_model_canvas,
    sample_canonical_tokens,
)
from train_scene_flow_pretrain import combine_pretrain_cfg_prediction
from train_scene_flow import FORMAL_SCENE_FPS


def _geometry(
    *,
    batch: int = 1,
    assets: int = 2,
    frames: int = 3,
    depth: float = 10.0,
):
    center = torch.zeros(batch, assets, frames, 3)
    center[..., 2] = depth
    center[:, 1:, :, 0] = 2.0
    size = torch.ones(batch, assets, frames, 3) * torch.tensor([4.0, 2.0, 2.0])
    yaw = torch.full((batch, assets, frames), torch.pi / 2)
    object_to_anchor = object_to_anchor_from_center_yaw(center, yaw)
    velocity = torch.zeros(batch, assets, frames, 3)
    track = torch.ones(batch, assets, frames, dtype=torch.bool)
    camera = torch.eye(4).view(1, 1, 4, 4).repeat(batch, frames, 1, 1)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).expand(batch, frames, -1, -1)
    return object_to_anchor, center, size, yaw, velocity, track, camera, intrinsics


def _condition(reference_seed: int = 0, **geometry_overrides) -> FactorizedAssetCondition:
    torch.manual_seed(reference_seed)
    values = _geometry(**geometry_overrides)
    object_to_anchor, center, size, yaw, velocity, track, camera, intrinsics = values
    batch, assets, frames = track.shape
    q, channels = 4, 8
    appearance = torch.randn(batch, assets, q, channels)
    appearance_mask = torch.ones(batch, assets, q, dtype=torch.bool)
    uv = torch.tensor(
        [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    ).view(1, 1, q, 2).expand(batch, assets, -1, -1).clone()
    return build_factorized_asset_condition(
        appearance_tokens=appearance,
        appearance_mask=appearance_mask,
        canonical_uv=uv,
        object_to_anchor=object_to_anchor,
        center_anchor=center,
        box_size_lwh=size,
        yaw=yaw,
        velocity_anchor=velocity,
        track_valid=track,
        camera_to_anchor=camera,
        intrinsics=intrinsics,
        image_size_hw=(100, 100),
        patch_grid=(4, 4),
    )


def _tiny_model(**kwargs) -> WanSceneFlow:
    config = dict(
        patch_grid=(4, 4),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=27,
        out_channels=8,
        text_dim=12,
        qwen_dim=12,
        num_layers=1,
        base_model_depth=1,
        ddt_head_dim=16,
        ddt_head_heads=2,
        ddt_head_depth=1,
        max_assets=2,
        max_asset_tokens=256,
        asset_condition_protocol="factorized_v1",
    )
    config.update(kwargs)
    return WanSceneFlow(**config)


def test_canonicalization_masks_background_and_uses_five_percent_coverage():
    alpha = torch.zeros(1, 20, 30)
    alpha[:, 5:15, 10:20] = 1.0
    rgb_a = torch.rand(3, 20, 30)
    rgb_b = rgb_a.clone()
    rgb_b[:, alpha[0] == 0] = torch.rand_like(rgb_b[:, alpha[0] == 0])

    canonical_a = canonicalize_asset_reference(rgb_a, alpha, (28, 42))
    canonical_b = canonicalize_asset_reference(rgb_b, alpha, (28, 42))

    assert torch.equal(canonical_a[0], canonical_b[0])
    assert torch.equal(canonical_a[1], canonical_b[1])
    assert bool(alpha_to_patch_mask(canonical_a[1], (2, 3)).any())


def test_canonicalization_does_not_multiply_soft_alpha_twice():
    rgb = torch.ones(3, 10, 10)
    alpha = torch.zeros(1, 10, 10)
    alpha[:, 2:8, 2:8] = 0.5

    canonical_rgb, canonical_alpha = canonicalize_asset_reference(
        rgb, alpha, (20, 20)
    )
    assert bool((canonical_alpha > 0.0).any())
    torch.testing.assert_close(
        canonical_rgb,
        canonical_alpha.expand_as(canonical_rgb),
        atol=1.0e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("source_hw", [(60, 200), (160, 200), (200, 60)])
def test_sampled_canonical_uv_is_relative_to_occupied_alpha_extent(source_hw):
    height, width = source_hw
    rgb = torch.ones(3, height, width)
    alpha = torch.ones(1, height, width)
    _, canonical_alpha = canonicalize_asset_reference(rgb, alpha, (350, 518))
    patch_grid = (25, 37)
    patch_mask = alpha_to_patch_mask(canonical_alpha, patch_grid).unsqueeze(0)
    tokens = torch.zeros(1, patch_grid[0] * patch_grid[1], 1)

    _, sampled_mask, sampled_uv = sample_canonical_tokens(
        tokens,
        patch_mask,
        patch_grid,
        max_tokens=32,
    )

    valid_uv = sampled_uv[0, sampled_mask[0]]
    torch.testing.assert_close(valid_uv.amin(dim=0), torch.zeros(2))
    torch.testing.assert_close(valid_uv.amax(dim=0), torch.ones(2))


def test_near_plane_crossing_box_returns_clipped_finite_canvas_bbox():
    center = torch.tensor([[[[0.0, 0.0, 0.3]]]])
    yaw = torch.zeros(1, 1, 1)
    object_to_anchor = object_to_anchor_from_center_yaw(center, yaw)
    size = torch.tensor([[[[5.0, 2.0, 1.6]]]])
    track = torch.ones(1, 1, 1, dtype=torch.bool)
    camera = torch.eye(4).view(1, 1, 4, 4)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)

    bbox, visible = project_anchor_boxes_to_patch_bboxes(
        object_to_anchor,
        size,
        track,
        camera,
        intrinsics,
        (100, 100),
        (25, 37),
    )

    assert bool(visible.item())
    assert bool(torch.isfinite(bbox).all())
    assert 0.0 <= float(bbox[..., 0].item()) <= float(bbox[..., 2].item()) <= 37.0
    assert 0.0 <= float(bbox[..., 1].item()) <= float(bbox[..., 3].item()) <= 25.0


def test_box_projection_preserves_front_box_and_rejects_box_behind_camera():
    centers = torch.tensor([[[[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]]])
    yaws = torch.zeros(1, 1, 2)
    object_to_anchor = object_to_anchor_from_center_yaw(centers, yaws)
    size = torch.tensor([4.0, 2.0, 2.0]).view(1, 1, 1, 3).expand(1, 1, 2, 3)
    track = torch.ones(1, 1, 2, dtype=torch.bool)
    camera = torch.eye(4).view(1, 1, 4, 4).expand(1, 2, 4, 4)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).expand(1, 2, 3, 3)

    bbox, visible = project_anchor_boxes_to_patch_bboxes(
        object_to_anchor,
        size,
        track,
        camera,
        intrinsics,
        (100, 100),
        (25, 37),
    )

    torch.testing.assert_close(
        bbox[0, 0, 0],
        torch.tensor([13.875, 9.375, 23.125, 15.625]),
    )
    assert visible[0, 0].tolist() == [True, False]
    torch.testing.assert_close(bbox[0, 0, 1], torch.full((4,), -1.0))


def test_resize_crop_intrinsics_matches_model_canvas_projection():
    center = torch.tensor([[[[0.0, 0.0, 10.0]]]])
    yaw = torch.zeros(1, 1, 1)
    object_to_anchor = object_to_anchor_from_center_yaw(center, yaw)
    size = torch.tensor([[[[4.0, 2.0, 2.0]]]])
    track = torch.ones(1, 1, 1, dtype=torch.bool)
    camera = torch.eye(4).view(1, 1, 4, 4)
    raw_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]]
    )

    model_intrinsics, model_hw = resize_crop_intrinsics_to_model_canvas(
        raw_intrinsics,
        (200, 100),
        target_width=518,
        patch_size=14,
    )
    bbox, visible = project_anchor_boxes_to_patch_bboxes(
        object_to_anchor,
        size,
        track,
        camera,
        model_intrinsics.view(1, 1, 3, 3),
        model_hw,
        (37, 37),
    )

    assert model_hw.tolist() == [518, 518]
    assert bool(visible.item())
    torch.testing.assert_close(
        bbox[0, 0, 0],
        torch.tensor([13.875, 13.875, 23.125, 23.125]),
    )


def test_resize_crop_intrinsics_broadcasts_batch_sizes_over_time():
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(2, 1, 3, 3)
    raw_hw = torch.tensor([[1280, 1920], [200, 100]])

    model_intrinsics, model_hw = resize_crop_intrinsics_to_model_canvas(
        intrinsics,
        raw_hw,
    )

    assert model_intrinsics.shape == (2, 1, 3, 3)
    assert model_hw.shape == (2, 1, 2)
    assert model_hw[:, 0].tolist() == [[350, 518], [518, 518]]


def test_formal_training_uses_pretraining_waymo_fps():
    assert FORMAL_SCENE_FPS == 10.0

    model = _tiny_model()
    frame_ids = torch.arange(4).view(1, 4)
    pretrain_positions = model._target_position_ids(
        batch_size=1,
        seq_len=4,
        num_patches=16,
        patch_grid=(4, 4),
        device=torch.device("cpu"),
        frame_ids=frame_ids,
        fps=10.0,
    )
    formal_positions = model._target_position_ids(
        batch_size=1,
        seq_len=4,
        num_patches=16,
        patch_grid=(4, 4),
        device=torch.device("cpu"),
        frame_ids=frame_ids,
        fps=FORMAL_SCENE_FPS,
    )
    torch.testing.assert_close(formal_positions, pretrain_positions)


def test_condition_api_has_no_target_latent_or_target_dynamic_mask_inputs():
    parameters = inspect.signature(build_factorized_asset_condition).parameters
    forbidden = {"z_clean", "z_clean_n", "dynamic_mask", "target_dynamic_mask"}
    assert forbidden.isdisjoint(parameters)
    condition_a = _condition(1)
    target_z_a = torch.randn(1, 3, 16, 8)
    target_z_b = torch.randn_like(target_z_a)
    del target_z_a, target_z_b
    condition_b = _condition(1)
    for field in (
        "appearance_tokens",
        "appearance_mask",
        "canonical_uv",
        "placement_state",
        "target_bbox_patch",
        "track_valid",
    ):
        assert torch.equal(getattr(condition_a, field), getattr(condition_b, field))


def test_training_dataset_and_external_inference_share_condition_implementation():
    import datasets.dataset as dataset_module
    import inference_scene_flow_pretrain as inference_module
    import train_scene_flow_pretrain as train_module

    assert train_module.build_factorized_asset_condition is build_factorized_asset_condition
    assert inference_module.build_factorized_asset_condition is build_factorized_asset_condition
    assert dataset_module.canonicalize_asset_reference is canonicalize_asset_reference
    assert inference_module.canonicalize_asset_reference is canonicalize_asset_reference
    assert train_module.CanonicalAssetEncoder is CanonicalAssetEncoder
    assert inference_module.CanonicalAssetEncoder is CanonicalAssetEncoder


def test_condition_rejects_non_waymo_height_axis() -> None:
    values = list(_geometry())
    values[0] = values[0].clone()
    values[0][..., :3, 2] = torch.tensor([-1.0, 0.0, 0.0])
    object_to_anchor, center, size, yaw, velocity, track, camera, intrinsics = values

    with pytest.raises(ValueError, match="local-z height axis"):
        build_factorized_asset_condition(
            appearance_tokens=torch.ones(1, 2, 4, 8),
            appearance_mask=torch.ones(1, 2, 4, dtype=torch.bool),
            canonical_uv=torch.zeros(1, 2, 4, 2),
            object_to_anchor=object_to_anchor,
            center_anchor=center,
            box_size_lwh=size,
            yaw=yaw,
            velocity_anchor=velocity,
            track_valid=track,
            camera_to_anchor=camera,
            intrinsics=intrinsics,
            image_size_hw=(100, 100),
            patch_grid=(4, 4),
        )


def test_independent_cfg_algebra_matches_closed_form():
    branch = lambda value: {"video": torch.tensor(float(value))}
    combined = combine_pretrain_cfg_prediction(
        "video",
        full=branch(10),
        no_text_full=branch(7),
        text_only=branch(2),
        text_asset=branch(5),
        text_scale=2.0,
        asset_scale=3.0,
        camera_scale=4.0,
    )
    assert combined is not None
    assert combined.item() == 34.0


def test_appearance_and_placement_are_independently_invariant():
    base = _condition(3)
    moved_geometry = list(_geometry())
    moved_geometry[0] = moved_geometry[0].clone()
    moved_geometry[0][..., 0, 3] += 4.0
    moved_geometry[1] = moved_geometry[0][..., :3, 3].clone()
    moved = build_factorized_asset_condition(
        appearance_tokens=base.appearance_tokens,
        appearance_mask=base.appearance_mask,
        canonical_uv=base.canonical_uv,
        object_to_anchor=moved_geometry[0],
        center_anchor=moved_geometry[1],
        box_size_lwh=moved_geometry[2],
        yaw=moved_geometry[3],
        velocity_anchor=moved_geometry[4],
        track_valid=moved_geometry[5],
        camera_to_anchor=moved_geometry[6],
        intrinsics=moved_geometry[7],
        image_size_hw=(100, 100),
        patch_grid=(4, 4),
    )
    other_reference = _condition(4)

    assert torch.equal(base.appearance_tokens, moved.appearance_tokens)
    assert not torch.equal(base.target_bbox_patch, moved.target_bbox_patch)
    assert not torch.equal(base.appearance_tokens, other_reference.appearance_tokens)
    assert torch.equal(base.placement_state, other_reference.placement_state)
    assert torch.equal(base.target_bbox_patch, other_reference.target_bbox_patch)


def test_interpolation_shortest_yaw_and_real_timestamp_velocity():
    center, size, yaw, velocity, valid = interpolate_box_keyframes(
        [
            {"frame_id": 0, "center": [0, 0, 10], "size": [4, 2, 2], "yaw": 3.0},
            {"frame_id": 2, "center": [2, 0, 10], "size": [6, 2, 2], "yaw": -3.0},
        ],
        num_frames=3,
        fps=10.0,
    )
    assert torch.allclose(center[:, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(size[:, 0], torch.tensor([4.0, 5.0, 6.0]))
    assert abs(float(yaw[1])) > 3.0
    assert torch.allclose(velocity[:, 0], torch.full((3,), 10.0))
    assert valid.all()


def test_partial_offscreen_missing_track_multiobject_empty_and_rope_are_legal():
    condition = _condition(5)
    placement = condition.placement_state.clone()
    bbox = condition.target_bbox_patch.clone()
    track = condition.track_valid.clone()
    # Object 0 frame 1 exists off-screen; object 1 frame 2 does not exist.
    placement[0, 0, 1, 11] = 0.0
    bbox[0, 0, 1] = torch.tensor([-20.0, 0.0, -10.0, 2.0])
    track[0, 1, 2] = False
    condition = FactorizedAssetCondition(
        condition.appearance_tokens,
        condition.appearance_mask,
        condition.canonical_uv,
        placement,
        bbox,
        track,
    ).validate()
    model = _tiny_model()
    tokens, mask, positions = model._build_factorized_asset_condition(
        condition,
        seq_len=3,
        num_patches=16,
        patch_grid=(4, 4),
        frame_ids=torch.tensor([0, 1, 2]),
        fps=10.0,
    )
    assert tokens.shape[:2] == mask.shape
    assert bool(torch.isfinite(positions).all())
    assert float(positions.min()) >= 0.0
    assert float(positions.max()) < float(model.config.rope_max_position)
    # Off-screen existing object keeps exactly its summary; missing track keeps none.
    offscreen_time = 1 * (24.0 / 10.0)
    assert int(torch.isclose(positions[0, :, 0], torch.tensor(offscreen_time)).sum()) >= 1

    empty = condition.drop_rows(torch.tensor([True]))
    empty_tokens, empty_mask, _ = model._build_factorized_asset_condition(
        empty,
        seq_len=3,
        num_patches=16,
        patch_grid=(4, 4),
        asset_condition_kind=["none"],
    )
    assert empty_tokens.shape[1] == 0
    assert empty_mask is None


def test_factorized_model_forward_smoke_and_legacy_gate():
    model = _tiny_model()
    condition = _condition(7)
    z = torch.randn(1, 3, 16, 8)
    zeros = torch.zeros(1, 3, 16, 1)
    output = model(
        z,
        torch.tensor([0.5]),
        torch.zeros_like(z),
        torch.zeros_like(z),
        zeros,
        zeros,
        torch.ones_like(zeros),
        torch.zeros(1, 0, 8),
        factorized_asset_condition=condition,
        frame_ids=torch.tensor([0, 1, 2]),
        fps=10.0,
    )
    assert output.shape == z.shape

    with pytest.raises(RuntimeError, match="requires FactorizedAssetCondition"):
        model(
            z,
            torch.tensor([0.5]),
            torch.zeros_like(z),
            torch.zeros_like(z),
            zeros,
            zeros,
            torch.ones_like(zeros),
            torch.zeros(1, 0, 8),
        )


def test_factorized_token_budget_rejects_truncating_late_frames_or_slots():
    condition = _condition(8, assets=2, frames=3)
    # Two assets * three visible frames * (four patches + one summary) = 30.
    model = _tiny_model(max_asset_tokens=29)
    with pytest.raises(RuntimeError, match="must not be silently truncated"):
        model._build_factorized_asset_condition(
            condition,
            seq_len=3,
            num_patches=16,
            patch_grid=(4, 4),
            frame_ids=torch.tensor([[0, 1, 2]]),
            fps=10.0,
        )


def test_factorized_sparse_packing_handles_mixed_row_controls():
    condition = _condition(9, batch=4, assets=2, frames=3)
    model = _tiny_model(max_asset_tokens=64)

    tokens, mask, positions = model._build_factorized_asset_condition(
        condition,
        seq_len=3,
        num_patches=16,
        patch_grid=(4, 4),
        asset_condition_kind=[
            "factorized_asset",
            "asset_uncond",
            "empty",
            "mode_a_with_empty",
        ],
        frame_ids=torch.tensor([[0, 1, 2]] * 4),
        fps=10.0,
    )

    assert mask.sum(dim=1).tolist() == [30, 1, 1, 31]
    assert tokens.shape == (4, 31, model.config.hidden_size)
    assert positions.shape == (4, 31, 3)
    assert torch.equal(
        tokens[1, 0],
        model.asset_null_condition_embed.reshape(-1),
    )
    assert torch.equal(tokens[2, 0], model.empty_asset_embed.reshape(-1))
    assert torch.equal(tokens[3, 30], model.empty_asset_embed.reshape(-1))


def test_legacy_checkpoint_state_loads_with_new_factorized_parameters_initialized():
    source = _tiny_model(asset_condition_protocol="legacy_compatible")
    state = source.state_dict()
    for key in list(state):
        if key.startswith(("asset_placement_mlp.", "asset_summary_appearance_proj.", "placement_")):
            del state[key]
    restored = _tiny_model(asset_condition_protocol="legacy_compatible")
    restored.load_state_dict(state, strict=True)
    assert restored.placement_mean.shape == (PLACEMENT_STATE_DIM,)
    assert restored.placement_std.shape == (PLACEMENT_STATE_DIM,)


class _FakeAggregator(nn.Module):
    patch_start_idx = 1

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, images):
        n = int(images.shape[0])
        self.batch_sizes.append(n)
        levels = []
        for level in range(24):
            value = torch.arange(n * 1 * 7 * 8, dtype=images.dtype, device=images.device).reshape(n, 1, 7, 8)
            levels.append(value + float(level))
        return levels, levels, levels, None, 1


class _FakeTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode(self, levels, patch_grid=None):
        assert patch_grid == (2, 3)
        return torch.stack(levels, dim=0).mean(dim=0)


class _FakeNormalizer(nn.Module):
    def normalize(self, value):
        return value / 2.0


def test_canonical_encoder_runs_s1_once_and_reuses_sampled_values():
    encoder = CanonicalAssetEncoder(
        _FakeAggregator(),
        _FakeTokenizer(),
        _FakeNormalizer(),
        patch_grid=(2, 3),
        max_tokens=4,
    )
    rgb = torch.rand(2, 1, 3, 28, 42)
    alpha = torch.ones(2, 1, 1, 28, 42)
    result = encoder(rgb, alpha, batch_size=1, num_assets=2)
    assert result.appearance_tokens.shape == (1, 2, 4, 8)
    assert result.appearance_mask.all()
    # No target-frame dimension exists in the encoder output.
    assert result.canonical_uv.shape == (1, 2, 4, 2)


def test_canonical_encoder_skips_empty_slots_and_reuses_cached_appearance():
    aggregator = _FakeAggregator()
    encoder = CanonicalAssetEncoder(
        aggregator,
        _FakeTokenizer(),
        _FakeNormalizer(),
        patch_grid=(2, 3),
        max_tokens=4,
        cache_size=8,
    )
    rgb = torch.rand(3, 1, 3, 28, 42)
    alpha = torch.zeros(3, 1, 1, 28, 42)
    alpha[0] = 1.0
    keys = [("scene", "object", 7), None, None]

    first = encoder(
        rgb,
        alpha,
        batch_size=1,
        num_assets=3,
        cache_keys=keys,
    )
    second = encoder(
        rgb,
        alpha,
        batch_size=1,
        num_assets=3,
        cache_keys=keys,
    )

    assert aggregator.batch_sizes == [1]
    assert first.appearance_mask[0, 0].all()
    assert not first.appearance_mask[0, 1:].any()
    assert torch.equal(first.appearance_tokens, second.appearance_tokens)
    assert torch.equal(first.canonical_uv, second.canonical_uv)
