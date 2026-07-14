from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from dggt.utils.camera_generation import (
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_GENERATION_DIM,
    camera_anchor_mask,
    camera_geometry_loss,
    camera_state_from_dggt_pose_enc,
    decode_camera_trajectory,
    invert_se3,
    normalize_camera_state,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)
from datasets.dataset import WaymoOpenDataset
from dggt.utils.camera_condition import camera_summary_from_waymo_gt, fov_from_intrinsics
from dggt.utils.gaussian_time import gaussian_timestamps_from_frame_ids
from dggt.utils.rotation import mat_to_quat
from dggt.models.scene_flow import ChannelScale
from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
from dggt.utils.flow_cache_io import _slice_object_meta_for_frames
from dggt.utils.feature_stats import (
    compute_camera_role_stats,
    load_all_stats_into_buffers,
    validate_camera_stats_provenance,
    validate_stats_sequence_length,
)
from train_scene_flow_pretrain import (
    build_camera_anchor_context_dropout,
    build_pretrain_bundle_from_batch,
    cfg_sample_pretrain_latents,
    decode_pose_from_camera_features,
    pretrain_validation_stride,
    pretrain_validation_window_offsets,
    select_rgb_render_rows,
)
from tools.compute_pretrain_feature_stats import compute_dggt_stats_single_pass


def _trajectory(batch: int = 2, frames: int = 5) -> torch.Tensor:
    angles = torch.linspace(0.0, 0.5, frames)
    c, s = angles.cos(), angles.sin()
    rotations = torch.zeros(frames, 3, 3)
    rotations[:, 0, 0] = c
    rotations[:, 0, 2] = s
    rotations[:, 1, 1] = 1.0
    rotations[:, 2, 0] = -s
    rotations[:, 2, 2] = c
    c2w = torch.eye(4).repeat(batch, frames, 1, 1)
    c2w[..., :3, :3] = rotations
    c2w[..., 0, 3] = torch.linspace(1.0, 5.0, frames)
    c2w[..., 1, 3] = torch.linspace(-0.5, 0.5, frames)
    return c2w


def _dggt_pose(c2w: torch.Tensor, fov_xy: torch.Tensor) -> torch.Tensor:
    w2c = invert_se3(c2w)
    return torch.cat(
        (w2c[..., :3, 3], mat_to_quat(w2c[..., :3, :3]), fov_xy[..., 1:2], fov_xy[..., 0:1]),
        dim=-1,
    )


def test_camera_v2_absolute_relative_round_trip_and_so3_projection() -> None:
    c2w = _trajectory()
    intrinsics = torch.eye(3).repeat(2, 5, 1, 1)
    intrinsics[..., 0, 0] = 600
    intrinsics[..., 1, 1] = 550
    intrinsics[..., 0, 2] = 300
    intrinsics[..., 1, 2] = 245
    fov_xy = fov_from_intrinsics(intrinsics, (500, 640))
    pose_enc_dggt = _dggt_pose(c2w, fov_xy)
    state, anchors = camera_state_from_dggt_pose_enc(pose_enc_dggt)
    decoded = decode_camera_trajectory(state, anchors)

    assert state.shape == (2, 5, CAMERA_GENERATION_DIM)
    assert torch.allclose(decoded.camera_to_world, c2w, atol=2e-6)
    rotation = decoded.camera_to_world[..., :3, :3]
    eye = torch.eye(3)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, eye, atol=1e-6)
    assert torch.allclose(torch.det(rotation), torch.ones(2, 5), atol=1e-6)
    assert (decoded.pose_encoding[..., 3:7].norm(dim=-1) - 1.0).abs().max() < 1e-6
    assert (decoded.pose_encoding[..., :3] - pose_enc_dggt[..., :3]).abs().max() < 1e-5
    assert (decoded.pose_encoding[..., 7:] - pose_enc_dggt[..., 7:]).abs().max() < 1e-5


def test_rot6d_round_trip_preserves_rotation_column_convention() -> None:
    rotation = _trajectory(batch=1, frames=4)[0, :, :3, :3]
    restored = rotation_6d_to_matrix(rotation_matrix_to_6d(rotation))
    assert torch.allclose(restored, rotation, atol=1e-6)


def test_principal_point_aware_fov_and_missing_raw_size_error() -> None:
    K = torch.tensor([[[500.0, 0.0, 100.0], [0.0, 400.0, 200.0], [0.0, 0.0, 1.0]]])
    fov = fov_from_intrinsics(K, (480, 640))
    expected_x = math.atan2(100, 500) + math.atan2(540, 500)
    expected_y = math.atan2(200, 400) + math.atan2(280, 400)
    assert torch.allclose(fov[0], torch.tensor([expected_x, expected_y]), atol=1e-6)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        fov_from_intrinsics(K, None)


def test_sliding_windows_only_slice_the_global_anchor_mask() -> None:
    mask = camera_anchor_mask(1, 12)
    assert mask[:, :8].tolist() == [[True] + [False] * 7]
    assert mask[:, 4:12].tolist() == [[False] * 8]
    state = torch.zeros(1, 8, CAMERA_GENERATION_DIM)
    with pytest.raises(ValueError, match="initial_camera_to_world"):
        decode_camera_trajectory(state, torch.zeros(1, 8, dtype=torch.bool))


def test_delta_only_window_keeps_global_roles_and_decodes_from_previous_frame() -> None:
    c2w = _trajectory(batch=1, frames=12)
    fov = torch.full((1, 12, 2), 1.0)
    full_state, full_anchors = camera_state_from_dggt_pose_enc(_dggt_pose(c2w, fov))
    start, end = 4, 10
    window_state = full_state[:, start:end]
    window_anchors = full_anchors[:, start:end]
    assert not bool(window_anchors.any())
    decoded = decode_camera_trajectory(
        window_state,
        window_anchors,
        initial_camera_to_world=c2w[:, start - 1],
    )
    assert torch.allclose(decoded.camera_to_world, c2w[:, start:end], atol=2e-6)
    prediction = window_state.clone().requires_grad_(True)
    loss, metrics = camera_geometry_loss(
        prediction,
        window_state,
        window_anchors,
        initial_camera_to_world=c2w[:, start - 1],
    )
    loss.backward()
    assert loss.item() < 1e-8
    assert all(value.item() < 1e-8 for value in metrics.values())
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_balanced_camera_window_sampling_separates_anchor_and_delta_starts() -> None:
    dataset = WaymoOpenDataset.__new__(WaymoOpenDataset)
    dataset.sequence_length = 10
    dataset.camera_anchor_window_probability = 0.5
    import random

    random.seed(1234)
    starts = [dataset._sample_balanced_camera_start(0, 29) for _ in range(4000)]
    anchor_fraction = sum(start == 0 for start in starts) / len(starts)
    assert anchor_fraction == pytest.approx(0.5, abs=0.03)
    assert all(start == 0 or 1 <= start <= 19 for start in starts)
    assert all(start != 0 for start in starts if start > 0)


def test_validation_window_offsets_cover_anchor_and_delta_only_rollout() -> None:
    assert pretrain_validation_window_offsets(10, 7) == (0, 7, 14, 19)
    assert pretrain_validation_stride(6, 7) == 3
    assert pretrain_validation_window_offsets(6, 3) == (0, 3, 6, 9, 12, 15, 18, 21, 23)

    dataset = WaymoOpenDataset.__new__(WaymoOpenDataset)
    dataset.sequence_length = 10
    dataset.trunk_frames = 29
    dataset.start_idx = 0
    assert dataset._fixed_start_in_trunk(58, 1, window_offset=0) == 29
    assert dataset._fixed_start_in_trunk(58, 1, window_offset=7) == 36
    assert dataset._fixed_start_in_trunk(58, 1, window_offset=19) == 48


def test_render_camera_decode_requires_global_roles_and_previous_pose() -> None:
    c2w = _trajectory(batch=1, frames=12)
    fov = torch.full((1, 12, 2), 1.0)
    state, anchors = camera_state_from_dggt_pose_enc(_dggt_pose(c2w, fov))

    with pytest.raises(ValueError, match="global camera_anchor_mask"):
        decode_pose_from_camera_features(SimpleNamespace(), state[:, 7:])

    decoded_pose = decode_pose_from_camera_features(
        SimpleNamespace(),
        state[:, 7:],
        camera_anchor_mask=anchors[:, 7:],
        initial_camera_to_world=c2w[:, 6],
    )
    expected_pose = _dggt_pose(c2w[:, 7:], fov[:, 7:])
    assert torch.allclose(decoded_pose, expected_pose, atol=2e-5)


def test_pretrain_bundle_slices_latents_and_camera_roles_after_full_context_dggt() -> None:
    batch_size, context_frames, window_frames, patches, channels = 2, 29, 10, 2, 4
    full_c2w = _trajectory(batch=batch_size, frames=context_frames)
    full_fov = torch.full((batch_size, context_frames, 2), 1.0)
    full_pose = _dggt_pose(full_c2w, full_fov)

    class _Tokenizer:
        @staticmethod
        def encode(levels, patch_grid):
            assert patch_grid == (1, 2)
            return levels[0]

    class _Vggt:
        def __init__(self) -> None:
            self.scene_tokenizer = _Tokenizer()
            self.camera_head = lambda _levels: [full_pose]

        def get_aggregator_token_outputs(self, context_images):
            assert tuple(context_images.shape[:2]) == (batch_size, context_frames)
            frame_value = torch.arange(context_frames, dtype=torch.float32).view(1, -1, 1, 1)
            tokens = frame_value.expand(batch_size, -1, patches, channels).clone()
            levels = [tokens.clone() for _ in range(24)]
            return {
                "aggregated_tokens_list": levels,
                "image_tokens_list": levels,
                "patch_start_idx": 0,
            }

    class _SceneFlow(torch.nn.Module):
        @staticmethod
        def normalize(value):
            return value

        @staticmethod
        def normalize_camera(value, anchor_mask):
            del anchor_mask
            return value

    window_indices = torch.stack((torch.arange(10), torch.arange(7, 17)))
    selected_c2w = torch.stack((full_c2w[0, :10], full_c2w[1, 7:17]))
    previous_c2w = torch.stack(
        (
            torch.cat((full_c2w[0, :1], full_c2w[0, :9]), dim=0),
            full_c2w[1, 6:16],
        )
    )
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(batch_size, window_frames, -1, -1).clone()
    intrinsics[..., 0, 0] = 500.0
    intrinsics[..., 1, 1] = 500.0
    intrinsics[..., 0, 2] = 1.0
    intrinsics[..., 1, 2] = 1.0
    batch = {
        "images": torch.zeros(batch_size, window_frames, 3, 2, 2),
        "dggt_context_images": torch.zeros(batch_size, context_frames, 3, 2, 2),
        "dggt_context_frame_ids": torch.arange(context_frames).view(1, -1).expand(batch_size, -1),
        "dggt_window_indices": window_indices,
        "frame_ids": window_indices,
        "masks": torch.zeros(batch_size, window_frames, 1, 2, 2),
        "camera_to_world_corrected": selected_c2w,
        "intrinsics": intrinsics,
        "raw_image_size_hw": torch.tensor([[2, 2], [2, 2]]),
        "camera_trajectory_anchor_to_world_corrected": full_c2w[:, :1],
        "camera_previous_to_world_corrected": previous_c2w,
        "pretrain_object_patch_mask": torch.ones(
            batch_size, 1, window_frames, patches, dtype=torch.bool
        ),
    }
    args = SimpleNamespace(
        patch_grid=(1, 2),
        precision="fp32",
        no_sky_generation=True,
        sky_mask_refine_scale=2,
        pretrain_asset_corruption_noise_std=0.01,
    )
    scene_flow = _SceneFlow()
    torch.manual_seed(123)
    bundle = build_pretrain_bundle_from_batch(
        batch,
        _Vggt(),
        scene_flow,
        torch.device("cpu"),
        args,
    )
    assert torch.equal(bundle.frame_ids, window_indices)
    assert bundle.camera_gen_anchor_mask.tolist() == [
        [True] + [False] * 9,
        [False] * 10,
    ]
    assert torch.equal(bundle.z_clean_n[:, :, 0, 0], window_indices.float())
    assert not torch.equal(bundle.F_asset_tokens[:, 0], bundle.z_clean_n)

    scene_flow.eval()
    eval_bundle = build_pretrain_bundle_from_batch(
        batch,
        _Vggt(),
        scene_flow,
        torch.device("cpu"),
        args,
    )
    assert torch.equal(eval_bundle.F_asset_tokens[:, 0], eval_bundle.z_clean_n)
    assert torch.allclose(bundle.camera_previous_c2w_dggt[1], full_c2w[1, 6])
    decoded = decode_camera_trajectory(
        bundle.camera_target_state_dggt[1:2],
        bundle.camera_gen_anchor_mask[1:2],
        initial_camera_to_world=bundle.camera_previous_c2w_dggt[1:2],
    )
    assert torch.allclose(decoded.camera_to_world, full_c2w[1:2, 7:17], atol=2e-6)


def test_global_waymo_camera_context_matches_full_trajectory_slice() -> None:
    frames = 14
    c2w = torch.eye(4).repeat(frames, 1, 1)
    c2w[:, 0, 3] = torch.arange(frames, dtype=torch.float32)
    K = torch.tensor([[1000.0, 0.0, 960.0], [0.0, 1000.0, 640.0], [0.0, 0.0, 1.0]]).repeat(frames, 1, 1)
    full, _ = camera_summary_from_waymo_gt(c2w, K, image_hw=(1280, 1920))
    start, end = 4, 14
    window, _ = camera_summary_from_waymo_gt(
        c2w[start:end],
        K[start:end],
        image_hw=(1280, 1920),
        trajectory_anchor_to_world=c2w[:1],
        previous_camera_to_world=c2w[start - 1 : end - 1],
    )
    assert torch.allclose(window, full[:, start:end], atol=1e-6)
    assert window[0, 0, 0].item() == pytest.approx(0.4)
    assert window[0, 0, 9].item() == pytest.approx(0.1)


@pytest.mark.parametrize(
    "frames, windows",
    [
        (10, [(0, 10)]),
        (29, [(0, 10), (7, 17), (14, 24), (19, 29)]),
    ],
)
def test_reconstructed_cache_camera_context_supports_formal_offline_windows(
    frames: int,
    windows: list[tuple[int, int]],
) -> None:
    """Formal offline inference reconstructs the clip before windowing it."""
    camera_full = torch.eye(4).view(1, 1, 4, 4).repeat(frames, 1, 1, 1)
    camera_full[:, 0, 0, 3] = torch.arange(frames, dtype=torch.float32)
    reconstructed = _slice_object_meta_for_frames(
        {
            "camera_to_world_corrected": camera_full,
            "intrinsics": torch.eye(3).unsqueeze(0),
        },
        torch.arange(frames, dtype=torch.long),
    )
    payload = {
        "meta": {"raw_image_size_hw": torch.tensor([[1280, 1920]])},
        "object_meta": reconstructed,
    }

    for start, end in windows:
        subset = torch.arange(start, end, dtype=torch.long)
        camera_gt = WaymoFlowCacheDataset._build_fast_camera_gt(
            payload,
            subset,
        )

        anchor = camera_gt["trajectory_anchor_to_world"]
        previous = camera_gt["previous_camera_to_world"]
        assert anchor[0, 0, 3].item() == 0.0
        expected_previous = (subset - 1).clamp_min(0).to(torch.float32)
        torch.testing.assert_close(previous[:, 0, 3], expected_previous)


def test_anchor_context_dropout_exposes_delta_only_training_context() -> None:
    anchors = camera_anchor_mask(2, 10)
    attention, supervision = build_camera_anchor_context_dropout(
        anchors, torch.tensor([True, False])
    )
    assert attention[0].tolist() == [False] + [True] * 9
    assert attention[1].tolist() == [True] * 10
    assert torch.equal(supervision.squeeze(-1), attention)


def test_rgb_render_rows_exclude_anchor_dropout_before_sample_cap() -> None:
    selected = select_rgb_render_rows(
        torch.tensor([True, False, False, True]),
        max_samples=1,
    )
    assert selected.tolist() == [1]


def test_rgb_render_rows_support_all_valid_and_all_dropped_batches() -> None:
    selected_all = select_rgb_render_rows(torch.tensor([False, False]), max_samples=0)
    selected_none = select_rgb_render_rows(torch.tensor([True, True]), max_samples=4)
    assert selected_all.tolist() == [0, 1]
    assert selected_none.numel() == 0


def test_rgb_render_row_selection_rejects_non_batch_mask() -> None:
    with pytest.raises(ValueError, match=r"must be \[B\]"):
        select_rgb_render_rows(torch.zeros(2, 1, dtype=torch.bool), max_samples=1)


def test_gaussian_time_is_window_length_independent() -> None:
    full = gaussian_timestamps_from_frame_ids(torch.arange(29))
    window = gaussian_timestamps_from_frame_ids(torch.arange(7, 17))
    assert torch.equal(window, full[7:17])
    assert full[-1].item() == pytest.approx(7.0)


def test_gt_camera_prediction_has_zero_geometry_loss_and_gradient() -> None:
    c2w = _trajectory(batch=1, frames=5)
    fov = torch.full((1, 5, 2), 1.0)
    target, anchors = camera_state_from_dggt_pose_enc(_dggt_pose(c2w, fov))
    prediction = target.clone().requires_grad_(True)
    loss, metrics = camera_geometry_loss(prediction, target, anchors)
    loss.backward()
    assert loss.item() < 1e-8
    assert all(value.item() < 1e-8 for value in metrics.values())
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_camera_geometry_loss_applies_component_weights_to_every_objective() -> None:
    c2w = _trajectory(batch=1, frames=5)
    fov = torch.full((1, 5, 2), 1.0)
    target, anchors = camera_state_from_dggt_pose_enc(_dggt_pose(c2w, fov))
    prediction = target.clone()
    prediction[..., :3] += torch.tensor([0.2, -0.1, 0.05])
    prediction[..., 3:9] += 0.03 * torch.arange(6, dtype=prediction.dtype)
    prediction[..., 9:11] += torch.tensor([0.04, -0.02])

    absolute_weight, relative_weight, smoothness_weight = 1.3, 0.7, 0.2
    translation_weight, rotation_weight, fov_weight = 2.0, 3.0, 5.0
    loss, metrics = camera_geometry_loss(
        prediction,
        target,
        anchors,
        absolute_weight=absolute_weight,
        relative_weight=relative_weight,
        smoothness_weight=smoothness_weight,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
        fov_weight=fov_weight,
    )
    expected = absolute_weight * (
        translation_weight * metrics["camera_absolute_translation"]
        + rotation_weight * metrics["camera_absolute_rotation_rad"]
        + fov_weight * metrics["camera_log_fov"]
    ) + relative_weight * (
        translation_weight * metrics["camera_relative_translation"]
        + rotation_weight * metrics["camera_relative_rotation_rad"]
    ) + smoothness_weight * (
        translation_weight * metrics["camera_acceleration_translation"]
        + rotation_weight * metrics["camera_acceleration_rotation_rad"]
        + fov_weight * metrics["camera_acceleration_fov"]
    )
    assert torch.allclose(loss, expected)


def test_role_normalization_matches_noise_scale() -> None:
    generator = torch.Generator().manual_seed(7)
    state = torch.randn(256, 8, CAMERA_GENERATION_DIM, generator=generator)
    anchors = camera_anchor_mask(256, 8)
    anchor_values = state[anchors]
    delta_values = state[~anchors]
    normalized = normalize_camera_state(
        state,
        anchors,
        anchor_values.mean(0),
        anchor_values.std(0, unbiased=False),
        delta_values.mean(0),
        delta_values.std(0, unbiased=False),
    )
    assert normalized[anchors].mean(0).abs().max() < 1e-5
    assert (normalized[anchors].std(0, unbiased=False) - 1).abs().max() < 1e-4
    assert abs(normalized.norm(dim=-1).mean().item() - math.sqrt(CAMERA_GENERATION_DIM)) < 0.3


@pytest.mark.parametrize(
    "code",
    [torch.zeros(6), torch.tensor([1.0, 0, 0, 2.0, 0, 0])],
)
def test_degenerate_rot6d_is_finite_right_handed_so3(code: torch.Tensor) -> None:
    rotation = rotation_6d_to_matrix(code)
    assert torch.isfinite(rotation).all()
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1e-6)


def test_channel_scale_preserves_camera_magnitude() -> None:
    layer = ChannelScale(CAMERA_GENERATION_DIM)
    x = torch.randn(2, 3, CAMERA_GENERATION_DIM)
    assert torch.allclose(layer(3.5 * x), 3.5 * layer(x))


def test_camera_stats_reject_old_source_and_wrong_dggt_hash() -> None:
    states = [(torch.randn(1, 3, CAMERA_GENERATION_DIM), camera_anchor_mask(1, 3))]
    payload = compute_camera_role_stats(states, dggt_checkpoint_sha256="abc")
    validate_camera_stats_provenance(payload, "abc")

    old = dict(payload, camera_stats_version="camera_anchor_delta_per_channel_v2")
    with pytest.raises(ValueError, match="version mismatch"):
        validate_camera_stats_provenance(old, "abc")

    wrong_source = dict(payload, camera_target_source="waymo_gt")
    with pytest.raises(ValueError, match="target source mismatch"):
        validate_camera_stats_provenance(wrong_source, "abc")

    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_camera_stats_provenance(payload, "def")


def test_stats_sequence_length_contract_rejects_old_or_mismatched_files() -> None:
    validate_stats_sequence_length({"source": {"sequence_length": 10}}, 10)
    with pytest.raises(ValueError, match="missing source.sequence_length"):
        validate_stats_sequence_length({}, 10)
    with pytest.raises(ValueError, match="expected 10 frames, got 8"):
        validate_stats_sequence_length({"source": {"sequence_length": 8}}, 10)


class _StatsBufferModule(torch.nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.register_buffer("mu_z", torch.zeros(latent_dim))
        self.register_buffer("sigma_z", torch.ones(latent_dim))
        self.register_buffer("camera_anchor_mean", torch.zeros(CAMERA_GENERATION_DIM))
        self.register_buffer("camera_anchor_std", torch.ones(CAMERA_GENERATION_DIM))
        self.register_buffer("camera_delta_mean", torch.zeros(CAMERA_GENERATION_DIM))
        self.register_buffer("camera_delta_std", torch.ones(CAMERA_GENERATION_DIM))
        self.register_buffer("camera_stats_valid", torch.tensor(False, dtype=torch.bool))

    def set_latent_stats(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu_z.copy_(mu)
        self.sigma_z.copy_(sigma)

    def set_camera_stats(
        self,
        anchor_mean: torch.Tensor,
        anchor_std: torch.Tensor,
        delta_mean: torch.Tensor,
        delta_std: torch.Tensor,
    ) -> None:
        self.camera_anchor_mean.copy_(anchor_mean)
        self.camera_anchor_std.copy_(anchor_std)
        self.camera_delta_mean.copy_(delta_mean)
        self.camera_delta_std.copy_(delta_std)
        self.camera_stats_valid.fill_(True)


def _stats_payload(latent_dim: int = 4) -> dict:
    states = [(torch.randn(2, 3, CAMERA_GENERATION_DIM), camera_anchor_mask(2, 3))]
    payload = compute_camera_role_stats(states, dggt_checkpoint_sha256="abc")
    payload.update(
        mean=torch.linspace(-1.0, 1.0, latent_dim),
        std=torch.linspace(0.5, 1.5, latent_dim),
    )
    return payload


def test_checkpoint_stats_contract_accepts_exact_match_and_loads_all_buffers(tmp_path) -> None:
    path = tmp_path / "stats.pt"
    torch.save(_stats_payload(), path)
    module = _StatsBufferModule(latent_dim=4)
    load_all_stats_into_buffers(module, path, token_dim=4, expected_dggt_sha256="abc")

    load_all_stats_into_buffers(
        module,
        path,
        token_dim=4,
        expected_dggt_sha256="abc",
        require_existing_match=True,
    )
    assert bool(module.camera_stats_valid.item())


@pytest.mark.parametrize("field", ["mean", "camera_delta_std"])
def test_checkpoint_stats_contract_rejects_latent_or_camera_mismatch(tmp_path, field: str) -> None:
    original_path = tmp_path / "original.pt"
    changed_path = tmp_path / "changed.pt"
    original = _stats_payload()
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in original.items()}
    changed[field].reshape(-1)[0] += 0.25
    torch.save(original, original_path)
    torch.save(changed, changed_path)
    module = _StatsBufferModule(latent_dim=4)
    load_all_stats_into_buffers(module, original_path, token_dim=4, expected_dggt_sha256="abc")

    expected_name = "mu_z" if field == "mean" else field
    with pytest.raises(ValueError, match=expected_name):
        load_all_stats_into_buffers(
            module,
            changed_path,
            token_dim=4,
            expected_dggt_sha256="abc",
            require_existing_match=True,
        )


def test_stats_single_dggt_forward_streams_latent_and_camera_without_waymo_fields() -> None:
    class _Model:
        def __init__(self) -> None:
            self.calls = 0
            self.scene_tokenizer = SimpleNamespace(
                encode=lambda levels, patch_grid: torch.ones(
                    int(levels[0].shape[0]), int(levels[0].shape[1]), 1, 4
                )
            )

            def camera_head(levels):
                frames = int(levels[0].shape[1])
                pose = torch.zeros(1, frames, 9)
                pose[..., 0] = torch.arange(frames, dtype=torch.float32) * 0.1
                pose[..., 6] = 1.0
                pose[..., 7] = 1.0
                pose[..., 8] = 1.1
                return [pose]

            self.camera_head = camera_head

        def get_aggregator_token_outputs(self, images):
            self.calls += 1
            levels = [torch.zeros(1, int(images.shape[1]), 1, 4) for _ in range(24)]
            return {"aggregated_tokens_list": levels, "image_tokens_list": levels, "patch_start_idx": 0}

    model = _Model()
    args = SimpleNamespace(
        latent_dim=4,
        max_batches=None,
        require_dynamic_mask=False,
        precision="fp32",
        log_every=100,
    )
    latent, camera = compute_dggt_stats_single_pass(
        model,
        [{
            "images": torch.zeros(1, 2, 3, 2, 2),
            "dggt_context_images": torch.zeros(1, 29, 3, 2, 2),
            "dggt_window_indices": torch.tensor([[27, 28]]),
        }],
        args,
        torch.device("cpu"),
        compute_latents=True,
        dggt_checkpoint_sha256="abc",
    )
    assert model.calls == 1
    assert latent is not None and int(latent["count"]) == 2
    assert int(camera["camera_anchor_count"]) == 1
    assert int(camera["camera_delta_count"]) == 28
    validate_camera_stats_provenance(camera, "abc")


class _CfgCameraFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            prediction_type="v",
            camera_gen_dim=CAMERA_GENERATION_DIM,
            camera_generation_representation=CAMERA_GENERATION_REPRESENTATION,
        )

    def denormalize_camera(self, value: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        del anchor_mask
        return value

    def forward(self, z: torch.Tensor, sigma: torch.Tensor, *args, **kwargs):
        del sigma, args
        text = kwargs.get("text_tokens")
        scalar = 0.0 if text is None else float(text.reshape(-1)[0].item())
        out = {"video": torch.full_like(z, scalar)}
        if kwargs.get("camera_gen_tokens") is not None:
            out["camera"] = torch.full_like(kwargs["camera_gen_tokens"], scalar)
        return out


class _RecordingSlidingCameraFlow(_CfgCameraFlow):
    def __init__(self) -> None:
        super().__init__()
        self.windows: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, z: torch.Tensor, sigma: torch.Tensor, *args, **kwargs):
        self.windows.append(
            (
                kwargs["frame_ids"].detach().cpu().clone(),
                kwargs["camera_gen_anchor_mask"].detach().cpu().clone(),
            )
        )
        return super().forward(z, sigma, *args, **kwargs)


class _CfgText(torch.nn.Module):
    def forward(self, captions):
        values = [0.0 if not caption else 1.0 for caption in captions]
        return {
            "tokens": torch.tensor(values).view(len(values), 1, 1),
            "attention_mask": torch.ones(len(values), 1, dtype=torch.bool),
        }


def test_global_text_cfg_does_not_change_camera_when_camera_text_scale_is_one() -> None:
    video = torch.zeros(1, 3, 2, 4)
    anchors = camera_anchor_mask(1, 3)
    bundle = SimpleNamespace(
        z_clean_n=video,
        z_splat_n=torch.zeros_like(video),
        M_preserve=torch.zeros(1, 3, 2, 1),
        M_source=torch.ones(1, 3, 2, 1),
        M_dest=torch.zeros(1, 3, 2, 1),
        F_asset_tokens=torch.zeros(1, 0, 4),
        encoder_attention_mask=None,
        asset_condition_kind=["asset_uncond"],
        camera_condition_tokens=None,
        camera_attention_mask=None,
        camera_condition_kind=["camera_uncond"],
        camera_target_clean_n=torch.zeros(1, 3, CAMERA_GENERATION_DIM),
        camera_gen_anchor_mask=anchors,
        camera_previous_c2w_dggt=torch.eye(4).view(1, 4, 4),
        frame_ids=torch.arange(3).view(1, 3),
        captions=["move forward"],
    )
    args = SimpleNamespace(
        guidance_scale=1.0,
        camera_text_guidance_scale=1.0,
        asset_control_guidance_scale=1.0,
        camera_guidance_scale=1.0,
        val_sample_steps=1,
        shift=10.0,
        seed=13,
        val_sliding_window=0,
    )
    flow = _CfgCameraFlow()
    sample_1 = cfg_sample_pretrain_latents(
        flow, bundle, args, 0, torch.device("cpu"), guidance_scale=1.0,
        text_encoder=_CfgText(), return_camera=True,
    )
    sample_4 = cfg_sample_pretrain_latents(
        flow, bundle, args, 0, torch.device("cpu"), guidance_scale=4.0,
        text_encoder=_CfgText(), return_camera=True,
    )
    assert torch.equal(sample_1.camera_state_dggt, sample_4.camera_state_dggt)
    assert torch.equal(sample_1.camera_anchor_mask, anchors)
    assert torch.equal(sample_1.camera_initial_c2w_dggt, bundle.camera_previous_c2w_dggt)
    assert not torch.equal(sample_1.video, sample_4.video)


def test_29_frame_sliding_sampler_slices_global_camera_roles() -> None:
    frames = 29
    video = torch.zeros(1, frames, 2, 4)
    anchors = camera_anchor_mask(1, frames)
    initial_c2w = torch.eye(4).view(1, 4, 4)
    bundle = SimpleNamespace(
        z_clean_n=video,
        z_splat_n=torch.zeros_like(video),
        M_preserve=torch.zeros(1, frames, 2, 1),
        M_source=torch.ones(1, frames, 2, 1),
        M_dest=torch.zeros(1, frames, 2, 1),
        F_asset_tokens=torch.zeros(1, 0, 4),
        encoder_attention_mask=None,
        asset_condition_kind=["asset_uncond"],
        camera_condition_tokens=None,
        camera_attention_mask=None,
        camera_condition_kind=["camera_uncond"],
        camera_target_clean_n=torch.zeros(1, frames, CAMERA_GENERATION_DIM),
        camera_gen_anchor_mask=anchors,
        camera_previous_c2w_dggt=initial_c2w,
        frame_ids=torch.arange(frames).view(1, frames),
        captions=[""],
    )
    args = SimpleNamespace(
        guidance_scale=1.0,
        camera_text_guidance_scale=1.0,
        asset_control_guidance_scale=1.0,
        camera_guidance_scale=1.0,
        val_sample_steps=1,
        shift=10.0,
        seed=13,
        val_sliding_window=10,
        val_sliding_stride=7,
    )
    flow = _RecordingSlidingCameraFlow()
    sample = cfg_sample_pretrain_latents(
        flow,
        bundle,
        args,
        0,
        torch.device("cpu"),
        return_camera=True,
    )

    expected_starts = (0, 7, 14, 19)
    assert len(flow.windows) == len(expected_starts)
    for (frame_ids, anchor_mask), start in zip(flow.windows, expected_starts):
        assert frame_ids.tolist() == [list(range(start, start + 10))]
        assert anchor_mask.tolist() == [([True] + [False] * 9 if start == 0 else [False] * 10)]
    assert torch.equal(sample.camera_anchor_mask, anchors)
    assert torch.equal(sample.camera_initial_c2w_dggt, initial_c2w)
