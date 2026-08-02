from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_STATS_VERSION,
    CAMERA_TARGET_SOURCE,
    CAMERA_TARGET_SPACE,
    camera_anchor_mask,
    camera_geometry_loss,
    camera_state_from_waymo_c2w,
    decode_camera_trajectory,
    denormalize_camera_state,
    invert_se3,
    normalize_camera_state,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)
from datasets.dataset import WaymoOpenDataset
from dggt.utils.camera_condition import (
    CAMERA_CONDITION_REPRESENTATION,
    camera_condition_from_waymo_metric_target,
    camera_summary_from_waymo_gt,
    fov_from_intrinsics,
)
from dggt.utils.gaussian_time import gaussian_timestamps_from_frame_ids
from dggt.utils.scene_gauge import assemble_dggt_pose_encoding
from dggt.models.scene_flow import ChannelScale
from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
from dggt.utils.flow_cache_io import _slice_object_meta_for_frames
from dggt.utils.feature_stats import (
    compute_camera_role_stats,
    load_all_stats_into_buffers,
    validate_camera_stats_provenance,
    validate_stats_sequence_length,
)
from dggt.utils.factorized_asset_condition import (
    object_to_anchor_from_center_yaw,
)
from train_scene_flow_pretrain import (
    build_camera_anchor_context_dropout,
    build_pretrain_bundle_from_batch,
    cfg_sample_pretrain_latents,
    decode_metric_camera_from_features,
    pretrain_validation_stride,
    pretrain_validation_window_offsets,
    select_rgb_render_rows,
)


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


def _world_trajectory(batch: int = 2, frames: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    relative = _trajectory(batch=batch, frames=frames)
    anchor = torch.eye(4).repeat(batch, 1, 1)
    anchor[..., :3, :3] = relative[:, -1, :3, :3]
    anchor[..., 0, 3] = torch.linspace(17.0, 23.0, batch)
    anchor[..., 1, 3] = torch.linspace(-4.0, 2.0, batch)
    return anchor[:, None] @ relative, anchor


def test_camera_v4_waymo_metric_round_trip_and_so3_projection() -> None:
    c2w, anchor = _world_trajectory()
    state, anchors = camera_state_from_waymo_c2w(c2w, anchor)
    decoded = decode_camera_trajectory(
        state,
        anchors,
        trajectory_anchor_to_world=anchor,
    )

    assert state.shape == (2, 5, CAMERA_GENERATION_DIM)
    assert CAMERA_GENERATION_DIM == 9
    assert CAMERA_GENERATION_REPRESENTATION == "waymo_metric_relative_se3_rot6d_v4"
    assert CAMERA_STATS_VERSION == "waymo_metric_camera_anchor_delta_per_channel_v5_global_context"
    assert CAMERA_TARGET_SPACE == "waymo_metric_camera_to_world"
    assert CAMERA_TARGET_SOURCE == "waymo_gt_extrinsics"
    assert torch.allclose(decoded.camera_to_world, c2w, atol=2e-6)
    assert torch.allclose(decoded.world_to_camera, invert_se3(c2w), atol=2e-6)
    rotation = decoded.camera_to_world[..., :3, :3]
    eye = torch.eye(3)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, eye, atol=1e-6)
    assert torch.allclose(torch.det(rotation), torch.ones(2, 5), atol=1e-6)
    expected_first = invert_se3(anchor) @ c2w[:, 0]
    assert torch.allclose(state[:, 0, :3], expected_first[:, :3, 3], atol=2e-6)
    assert state.shape[-1] == 9


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


def test_waymo_condition_delta_half_equals_metric_generation_target() -> None:
    assert CAMERA_CONDITION_REPRESENTATION == "waymo_metric_rel_delta_rot6d_fov20d_stats_v3"
    frames = 8
    c2w = _trajectory(batch=2, frames=frames)
    anchor = c2w[:, 0]
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(2, frames, 1, 1)
    intrinsics[..., 0, 0] = 700.0
    intrinsics[..., 1, 1] = 680.0
    intrinsics[..., 0, 2] = 320.0
    intrinsics[..., 1, 2] = 240.0
    target, anchors = camera_state_from_waymo_c2w(c2w, anchor)
    condition, valid = camera_summary_from_waymo_gt(
        c2w,
        intrinsics,
        image_hw=(480, 640),
        trajectory_anchor_to_world=anchor[:, None],
    )
    assert condition.shape == (2, frames, 20)
    assert valid.all()
    assert target.shape == (2, frames, 9)
    assert torch.allclose(condition[..., 9:18], target, atol=2e-6)
    assert torch.equal(anchors, camera_anchor_mask(2, frames))
    # FOV is condition-only and remains outside the 9D generation target.
    assert bool((condition[..., 18:20] > 0).all())

    start = 3
    window_anchors = anchors[:, start:]
    previous = torch.cat((c2w[:, start - 1 : start], c2w[:, start:-1]), dim=1)
    window_target, _ = camera_state_from_waymo_c2w(
        c2w[:, start:],
        anchor,
        previous_camera_to_world=previous[:, 0],
        anchor_mask=window_anchors,
    )
    window_condition, _ = camera_summary_from_waymo_gt(
        c2w[:, start:],
        intrinsics[:, start:],
        image_hw=(480, 640),
        trajectory_anchor_to_world=anchor[:, None],
        previous_camera_to_world=previous,
    )
    assert torch.allclose(window_condition[..., 9:18], window_target, atol=2e-6)


def test_shared_metric_camera_condition_uses_global_role_aware_stats() -> None:
    frames = 7
    c2w = _trajectory(batch=1, frames=frames)
    anchor = c2w[:, 0]
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, frames, 1, 1)
    intrinsics[..., 0, 0] = 700.0
    intrinsics[..., 1, 1] = 680.0
    intrinsics[..., 0, 2] = 320.0
    intrinsics[..., 1, 2] = 240.0
    anchor_mean = torch.linspace(-0.4, 0.4, CAMERA_GENERATION_DIM)
    anchor_std = torch.linspace(0.8, 1.6, CAMERA_GENERATION_DIM)
    delta_mean = torch.linspace(0.5, -0.5, CAMERA_GENERATION_DIM)
    delta_std = torch.linspace(1.7, 0.9, CAMERA_GENERATION_DIM)

    def normalize(state: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        return normalize_camera_state(
            state,
            roles,
            anchor_mean,
            anchor_std,
            delta_mean,
            delta_std,
        )

    roles = camera_anchor_mask(1, frames)
    condition, valid, target, returned_roles = (
        camera_condition_from_waymo_metric_target(
            c2w,
            intrinsics,
            image_hw=(480, 640),
            trajectory_anchor_to_world=anchor,
            previous_camera_to_world=None,
            anchor_mask=roles,
            normalize_camera=normalize,
        )
    )
    torch.testing.assert_close(condition[..., 9:18], normalize(target, roles))
    assert valid.all()
    assert torch.equal(returned_roles, roles)

    start = 3
    delta_only_roles = roles[:, start:]
    previous = c2w[:, start - 1 : start]
    window_condition, _, window_target, returned_window_roles = (
        camera_condition_from_waymo_metric_target(
            c2w[:, start:],
            intrinsics[:, start:],
            image_hw=(480, 640),
            trajectory_anchor_to_world=anchor,
            previous_camera_to_world=previous,
            anchor_mask=delta_only_roles,
            normalize_camera=normalize,
        )
    )
    torch.testing.assert_close(
        window_condition[..., 9:18],
        normalize(window_target, delta_only_roles),
    )
    assert torch.equal(returned_window_roles, delta_only_roles)
    assert not torch.allclose(
        window_condition[..., 9:18],
        window_target,
    )


def test_sliding_windows_only_slice_the_global_anchor_mask() -> None:
    mask = camera_anchor_mask(1, 12)
    assert mask[:, :8].tolist() == [[True] + [False] * 7]
    assert mask[:, 4:12].tolist() == [[False] * 8]
    state = torch.zeros(1, 8, CAMERA_GENERATION_DIM)
    with pytest.raises(ValueError, match="initial_camera_to_world"):
        decode_camera_trajectory(state, torch.zeros(1, 8, dtype=torch.bool))


def test_delta_only_window_keeps_global_roles_and_decodes_from_previous_frame() -> None:
    c2w, anchor = _world_trajectory(batch=1, frames=12)
    full_state, _ = camera_state_from_waymo_c2w(c2w, anchor)
    start, end = 4, 10
    window_anchors = torch.zeros(1, end - start, dtype=torch.bool)
    window_state, returned_anchors = camera_state_from_waymo_c2w(
        c2w[:, start:end],
        anchor,
        previous_camera_to_world=c2w[:, start - 1],
        anchor_mask=window_anchors,
    )
    assert torch.equal(returned_anchors, window_anchors)
    assert torch.allclose(window_state, full_state[:, start:end], atol=2e-6)
    assert not bool(window_anchors.any())
    decoded = decode_camera_trajectory(
        window_state,
        window_anchors,
        initial_camera_to_world=c2w[:, start - 1],
        trajectory_anchor_to_world=anchor,
    )
    assert torch.allclose(decoded.camera_to_world, c2w[:, start:end], atol=2e-6)
    prediction = window_state.clone().requires_grad_(True)
    loss, metrics = camera_geometry_loss(
        prediction,
        window_state,
        window_anchors,
        initial_camera_to_world=c2w[:, start - 1],
        trajectory_anchor_to_world=anchor,
    )
    loss.backward()
    assert loss.item() < 1e-8
    assert all(value.item() < 1e-8 for value in metrics.values())
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_unbatched_delta_only_window_accepts_per_frame_previous_context() -> None:
    c2w, anchor = _world_trajectory(batch=1, frames=9)
    start = 3
    window = c2w[0, start:]
    previous = c2w[0, start - 1 : -1]
    anchors = torch.zeros(window.shape[0], dtype=torch.bool)
    state, returned = camera_state_from_waymo_c2w(
        window,
        anchor[0],
        previous_camera_to_world=previous,
        anchor_mask=anchors,
    )
    decoded = decode_camera_trajectory(
        state,
        returned,
        initial_camera_to_world=previous[0],
        trajectory_anchor_to_world=anchor[0],
    )
    assert state.shape == (window.shape[0], 9)
    assert torch.allclose(decoded.camera_to_world, window, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_metric_camera_v4_cuda0_roundtrip_condition_and_loss_smoke() -> None:
    device = torch.device("cuda:0")
    c2w = _trajectory(batch=2, frames=6).to(device)
    anchor = c2w[:, 0]
    state, anchors = camera_state_from_waymo_c2w(c2w, anchor)
    decoded = decode_camera_trajectory(
        state,
        anchors,
        trajectory_anchor_to_world=anchor,
    )
    torch.testing.assert_close(decoded.camera_to_world, c2w, atol=2e-6, rtol=2e-6)

    intrinsics = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(2, 6, 1, 1)
    intrinsics[..., 0, 0] = 700.0
    intrinsics[..., 1, 1] = 680.0
    intrinsics[..., 0, 2] = 320.0
    intrinsics[..., 1, 2] = 240.0
    condition, valid = camera_summary_from_waymo_gt(
        c2w,
        intrinsics,
        image_hw=(480, 640),
        trajectory_anchor_to_world=anchor[:, None],
    )
    torch.testing.assert_close(condition[..., 9:18], state, atol=2e-6, rtol=2e-6)
    assert valid.all()

    prediction = state.detach().clone().requires_grad_(True)
    loss, metrics = camera_geometry_loss(
        prediction,
        state,
        anchors,
        trajectory_anchor_to_world=anchor,
    )
    loss.backward()
    assert prediction.grad is not None and bool(torch.isfinite(prediction.grad).all())
    assert len(metrics) == 6


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
    c2w, anchor = _world_trajectory(batch=1, frames=12)
    state, anchors = camera_state_from_waymo_c2w(c2w, anchor)

    with pytest.raises(ValueError, match="global camera_anchor_mask"):
        decode_metric_camera_from_features(
            state[:, 7:], camera_anchor_mask=None  # type: ignore[arg-type]
        )

    decoded = decode_metric_camera_from_features(
        state[:, 7:],
        camera_anchor_mask=anchors[:, 7:],
        initial_camera_to_world=c2w[:, 6],
        trajectory_anchor_to_world=anchor,
    )
    assert torch.allclose(decoded.camera_to_world, c2w[:, 7:], atol=2e-6)


def test_pretrain_bundle_uses_metric_waymo_target_and_matching_condition_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size, context_frames, window_frames, patches, channels = 2, 29, 10, 2, 4
    full_c2w = _trajectory(batch=batch_size, frames=context_frames)
    scene_gauge = torch.tensor(
        [[0.0, math.log(math.tan(0.5)), math.log(math.tan(0.4))]]
    ).repeat(batch_size, 1)
    full_pose = assemble_dggt_pose_encoding(full_c2w, scene_gauge)

    class _Tokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        @staticmethod
        def encode(levels, patch_grid):
            assert patch_grid == (1, 2)
            return levels[0]

    class _Aggregator(torch.nn.Module):
        patch_start_idx = 0

        @staticmethod
        def forward(reference_images):
            count = int(reference_images.shape[0])
            tokens = torch.ones(count, 1, patches, channels)
            levels = [tokens.clone() for _ in range(24)]
            return levels, levels, levels, None, 0

    class _Vggt:
        def __init__(self) -> None:
            self.scene_tokenizer = _Tokenizer()
            self.aggregator = _Aggregator()
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
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("gauge_mean", torch.zeros(3))

        @staticmethod
        def normalize(value):
            return value

        @staticmethod
        def normalize_camera(value, anchor_mask):
            del anchor_mask
            return value

        @staticmethod
        def normalize_gauge(value):
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
    object_center = torch.tensor([0.0, 0.0, 10.0]).view(1, 1, 1, 3).repeat(
        batch_size, 1, window_frames, 1
    )
    object_yaw = torch.full((batch_size, 1, window_frames), torch.pi / 2)
    object_to_anchor = object_to_anchor_from_center_yaw(object_center, object_yaw)
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
        "metric_lidar_depth_m": torch.full(
            (batch_size, window_frames, 2, 2), 10.0, dtype=torch.float32
        ),
        "scene_gauge": scene_gauge,
        "scene_gauge_valid": torch.ones(batch_size, 3, dtype=torch.bool),
        "pretrain_asset_condition_version": ["factorized_asset_v3"] * batch_size,
        "pretrain_asset_source_kind": ["instances_projected"] * batch_size,
        "pretrain_reference_rgb": torch.ones(batch_size, 1, 3, 2, 2),
        "pretrain_reference_alpha": torch.ones(batch_size, 1, 1, 2, 2),
        "pretrain_reference_frame_id": torch.tensor([[10], [17]]),
        "pretrain_object_obj_to_anchor": object_to_anchor,
        "pretrain_object_center_anchor": object_center,
        "pretrain_object_box_size": torch.tensor([1.0, 1.0, 1.0])
        .view(1, 1, 1, 3)
        .repeat(batch_size, 1, window_frames, 1),
        "pretrain_object_yaw": object_yaw,
        "pretrain_object_velocity_anchor": torch.zeros(batch_size, 1, window_frames, 3),
        "pretrain_object_track_valid": torch.ones(
            batch_size, 1, window_frames, dtype=torch.bool
        ),
        "pretrain_camera_to_anchor": torch.eye(4)
        .view(1, 1, 4, 4)
        .repeat(batch_size, window_frames, 1, 1),
        "pretrain_fps": torch.full((batch_size,), 10.0),
    }
    args = SimpleNamespace(
        patch_grid=(1, 2),
        precision="fp32",
        no_sky_generation=True,
        sky_mask_refine_scale=2,
    )
    scene_flow = _SceneFlow()
    import train_scene_flow_pretrain as pretrain_entry

    image_transfer_shapes: list[tuple[int, ...]] = []
    original_images_to_device = pretrain_entry._images_to_device

    def record_images_to_device(images, device):
        image_transfer_shapes.append(tuple(images.shape))
        return original_images_to_device(images, device)

    monkeypatch.setattr(pretrain_entry, "_images_to_device", record_images_to_device)
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
    assert (batch_size, context_frames, 3, 2, 2) in image_transfer_shapes
    assert (batch_size, window_frames, 3, 2, 2) not in image_transfer_shapes
    assert bundle.F_asset_tokens.shape == (batch_size, 0, channels)
    assert bundle.factorized_asset_condition.appearance_mask.any(dim=-1).all()

    scene_flow.eval()
    batch_without_lidar = dict(batch)
    batch_without_lidar.pop("metric_lidar_depth_m")
    eval_bundle = build_pretrain_bundle_from_batch(
        batch_without_lidar,
        _Vggt(),
        scene_flow,
        torch.device("cpu"),
        args,
    )
    assert eval_bundle.metric_lidar_depth_m is None
    assert eval_bundle.metric_lidar_depth_valid is None
    bundle_without_metric_transfer = build_pretrain_bundle_from_batch(
        batch,
        _Vggt(),
        scene_flow,
        torch.device("cpu"),
        args,
        include_metric_depth_diagnostic=False,
    )
    assert bundle_without_metric_transfer.metric_lidar_depth_m is None
    assert bundle_without_metric_transfer.metric_lidar_depth_valid is None
    assert torch.equal(
        eval_bundle.factorized_asset_condition.appearance_tokens,
        bundle.factorized_asset_condition.appearance_tokens,
    )
    assert torch.allclose(bundle.camera_previous_c2w_metric[1], full_c2w[1, 6])
    assert bundle.camera_target_state_metric.shape[-1] == 9
    assert torch.equal(
        bundle.camera_condition_tokens[..., 9:18], bundle.camera_target_clean_n
    )
    torch.testing.assert_close(
        bundle.sky_pose_enc_gauge,
        bundle.render_pose_enc_teacher_gauge,
        atol=0.0,
        rtol=0.0,
    )
    assert (
        bundle.sky_pose_enc_gauge.untyped_storage().data_ptr()
        == bundle.render_pose_enc_teacher_gauge.untyped_storage().data_ptr()
    )

    invalid_batch = dict(batch)
    invalid_batch["scene_gauge"] = batch["scene_gauge"].clone()
    invalid_batch["scene_gauge_valid"] = batch["scene_gauge_valid"].clone()
    invalid_batch["scene_gauge"][0, 1] = 0.0  # finite sentinel for table JSON null
    invalid_batch["scene_gauge_valid"][0, 1] = False
    scene_flow.gauge_mean.copy_(torch.tensor([0.3, -0.7, -0.9]))
    invalid_bundle = build_pretrain_bundle_from_batch(
        invalid_batch,
        _Vggt(),
        scene_flow,
        torch.device("cpu"),
        args,
    )
    assert invalid_bundle.scene_gauge_clean[0, 0, 1].item() == 0.0
    assert invalid_bundle.scene_gauge_effective[0, 0, 1].item() == pytest.approx(-0.7)
    assert invalid_bundle.scene_gauge_clean_n[0, 0, 1].item() == pytest.approx(-0.7)
    expected_sky_pose = assemble_dggt_pose_encoding(
        selected_c2w,
        invalid_bundle.scene_gauge_effective,
    )
    torch.testing.assert_close(invalid_bundle.sky_pose_enc_gauge, expected_sky_pose)
    decoded = decode_camera_trajectory(
        bundle.camera_target_state_metric[1:2],
        bundle.camera_gen_anchor_mask[1:2],
        initial_camera_to_world=bundle.camera_previous_c2w_metric[1:2],
        trajectory_anchor_to_world=bundle.camera_trajectory_anchor_to_world_metric[1:2],
    )
    assert torch.allclose(decoded.camera_to_world, full_c2w[1:2, 7:17], atol=2e-6)


def test_production_pretrain_launchers_enable_pinned_memory() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    launchers = (
        "pretrain_four_nodes.sh",
        "pretrain_half_node_p6000.sh",
        "pretrain_ppu.sh",
        "pretrain_ppu_two_nodes_dlc.sh",
        "pretrain_single_node.sh",
        "pretrain_single_node30.sh",
        "pretrain_three_nodes.sh",
        "pretrain_two_nodes26.sh",
        "pretrain_two_nodes31.sh",
    )
    for launcher in launchers:
        source = (repo_root / launcher).read_text(encoding="utf-8")
        assert "--pin_memory" in source, launcher


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
    assert window[0, 0, 0].item() == pytest.approx(4.0)
    assert window[0, 0, 9].item() == pytest.approx(1.0)


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
    c2w, anchor = _world_trajectory(batch=1, frames=5)
    target, anchors = camera_state_from_waymo_c2w(c2w, anchor)
    prediction = target.clone().requires_grad_(True)
    loss, metrics = camera_geometry_loss(
        prediction,
        target,
        anchors,
        trajectory_anchor_to_world=anchor,
    )
    loss.backward()
    assert loss.item() < 1e-8
    assert all(value.item() < 1e-8 for value in metrics.values())
    assert set(metrics) == {
        "camera_absolute_translation",
        "camera_absolute_rotation_rad",
        "camera_relative_translation",
        "camera_relative_rotation_rad",
        "camera_acceleration_translation",
        "camera_acceleration_rotation_rad",
    }
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_camera_geometry_loss_applies_component_weights_to_every_objective() -> None:
    c2w, anchor = _world_trajectory(batch=1, frames=5)
    target, anchors = camera_state_from_waymo_c2w(c2w, anchor)
    prediction = target.clone()
    prediction[..., :3] += torch.tensor([0.2, -0.1, 0.05])
    prediction[..., 3:9] += 0.03 * torch.arange(6, dtype=prediction.dtype)

    absolute_weight, relative_weight, smoothness_weight = 1.3, 0.7, 0.2
    translation_weight, rotation_weight = 2.0, 3.0
    loss, metrics = camera_geometry_loss(
        prediction,
        target,
        anchors,
        trajectory_anchor_to_world=anchor,
        absolute_weight=absolute_weight,
        relative_weight=relative_weight,
        smoothness_weight=smoothness_weight,
        translation_weight=translation_weight,
        rotation_weight=rotation_weight,
    )
    expected = absolute_weight * (
        translation_weight * metrics["camera_absolute_translation"]
        + rotation_weight * metrics["camera_absolute_rotation_rad"]
    ) + relative_weight * (
        translation_weight * metrics["camera_relative_translation"]
        + rotation_weight * metrics["camera_relative_rotation_rad"]
    ) + smoothness_weight * (
        translation_weight * metrics["camera_acceleration_translation"]
        + rotation_weight * metrics["camera_acceleration_rotation_rad"]
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
    restored = denormalize_camera_state(
        normalized,
        anchors,
        anchor_values.mean(0),
        anchor_values.std(0, unbiased=False),
        delta_values.mean(0),
        delta_values.std(0, unbiased=False),
    )
    assert torch.allclose(restored, state, atol=1e-6)
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
        camera_previous_c2w_metric=torch.eye(4).view(1, 4, 4),
        camera_trajectory_anchor_to_world_metric=torch.eye(4).view(1, 4, 4),
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
    assert torch.equal(sample_1.camera_state_metric, sample_4.camera_state_metric)
    assert torch.equal(sample_1.camera_anchor_mask, anchors)
    assert torch.equal(sample_1.camera_initial_c2w_metric, bundle.camera_previous_c2w_metric)
    assert torch.equal(
        sample_1.camera_trajectory_anchor_to_world_metric,
        bundle.camera_trajectory_anchor_to_world_metric,
    )
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
        camera_previous_c2w_metric=initial_c2w,
        camera_trajectory_anchor_to_world_metric=torch.eye(4).view(1, 4, 4),
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
    assert torch.equal(sample.camera_initial_c2w_metric, initial_c2w)
