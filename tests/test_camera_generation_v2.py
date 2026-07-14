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
from dggt.utils.camera_condition import camera_summary_from_waymo_gt, fov_from_intrinsics
from dggt.utils.gaussian_time import gaussian_timestamps_from_frame_ids
from dggt.utils.rotation import mat_to_quat
from dggt.models.scene_flow import ChannelScale
from dggt.utils.feature_stats import compute_camera_role_stats, validate_camera_stats_provenance
from train_scene_flow_pretrain import build_camera_anchor_context_dropout, cfg_sample_pretrain_latents
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
    with pytest.raises(ValueError, match="global anchor"):
        decode_camera_trajectory(state, torch.zeros(1, 8, dtype=torch.bool))


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


def test_anchor_context_dropout_exposes_delta_only_training_context() -> None:
    anchors = camera_anchor_mask(2, 10)
    attention, supervision = build_camera_anchor_context_dropout(
        anchors, torch.tensor([True, False])
    )
    assert attention[0].tolist() == [False] + [True] * 9
    assert attention[1].tolist() == [True] * 10
    assert torch.equal(supervision.squeeze(-1), attention)


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


def test_stats_single_dggt_forward_streams_latent_and_camera_without_waymo_fields() -> None:
    class _Model:
        def __init__(self) -> None:
            self.calls = 0
            self.camera_head = lambda levels: [
                torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.1],
                               [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.1]]])
            ]
            self.scene_tokenizer = SimpleNamespace(
                encode=lambda levels, patch_grid: torch.ones(1, 2, 1, 4)
            )

        def get_aggregator_token_outputs(self, images):
            self.calls += 1
            levels = [torch.zeros(1, 2, 1, 4) for _ in range(24)]
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
        [{"images": torch.zeros(1, 2, 3, 2, 2)}],
        args,
        torch.device("cpu"),
        compute_latents=True,
        dggt_checkpoint_sha256="abc",
    )
    assert model.calls == 1
    assert latent is not None and int(latent["count"]) == 2
    assert int(camera["camera_anchor_count"]) == 1
    assert int(camera["camera_delta_count"]) == 1
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
    assert not torch.equal(sample_1.video, sample_4.video)
