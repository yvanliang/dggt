from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import dggt.losses.reconstruction_feedback_loss as feedback_loss_module
import dggt.losses.rgb_render_loss as rgb_render_module
import train_scene_flow_pretrain as pretrain_train
from dggt.losses.reconstruction_feedback_loss import (
    compute_reconstruction_feedback_losses,
)
from dggt.losses.rgb_render_loss import (
    RGBRenderLossResult,
    _masked_lpips,
    _render_one_sample,
    compute_rgb_render_loss,
    decode_generated_dggt_geometry,
    rgb_render_loss_ramp,
    rgb_render_sigma_weight,
    scale_gradient,
    should_apply_rgb_render_loss,
    sky_tokens_to_background,
    torch_unproject_depth,
)
from dggt.utils.gaussian_render import composite_gsplat_rgb
from dggt.utils.scene_gauge import (
    PullbackCalibration,
    assemble_dggt_pose_encoding,
    metric_c2w_to_teacher_anchor_dggt,
)
from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
import train_scene_flow as formal_train
from train_scene_flow import (
    _cached_render_pose_from_payload,
    _formal_rgb_context,
    _prepare_visualization_batch,
    cached_render_pose_from_item,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_LAUNCH_SCRIPTS = (
    "pretrain_half_node_p6000.sh",
    "pretrain_ppu.sh",
    "pretrain_ppu_four_nodes_dlc.sh",
    "pretrain_ppu_two_nodes_dlc.sh",
    "pretrain_single_node.sh",
    "pretrain_two_nodes26.sh",
    "pretrain_two_nodes31.sh",
)


def _parse_pretrain_args(*extra: str):
    return pretrain_train.build_argparser().parse_args(
        [
            "--image_dir",
            "/tmp/training",
            "--hdmap_root",
            "/tmp/training_hdmap",
            "--dggt_ckpt_path",
            "/tmp/dggt.pt",
            "--scene_gauge_path",
            "/tmp/training_gauge.json",
            "--pullback_calibration_path",
            "/tmp/pullback.json",
            "--log_dir",
            "/tmp/run",
            *extra,
        ]
    )


def _parse_formal_args(*extra: str):
    return formal_train.build_argparser().parse_args(
        ["--ckpt_path", "/tmp/dggt.pt", "--log_dir", "/tmp/run", *extra]
    )


def test_scene_flow_training_gradient_checkpointing_cli() -> None:
    assert _parse_pretrain_args().gradient_checkpointing is True
    assert _parse_formal_args().gradient_checkpointing is True
    assert _parse_pretrain_args().half_gradient_checkpointing is False
    assert _parse_formal_args().half_gradient_checkpointing is False
    assert _parse_pretrain_args().three_quarter_gradient_checkpointing is False
    assert _parse_formal_args().three_quarter_gradient_checkpointing is False
    assert _parse_pretrain_args("--no_gradient_checkpointing").gradient_checkpointing is False
    assert _parse_formal_args("--no-gradient-checkpointing").gradient_checkpointing is False
    assert _parse_pretrain_args("--gradient_checkpointing").gradient_checkpointing is True
    assert _parse_formal_args("--gradient-checkpointing").gradient_checkpointing is True
    assert _parse_pretrain_args("--half_gradient_checkpointing").half_gradient_checkpointing is True
    assert _parse_formal_args("--half-gradient-checkpointing").half_gradient_checkpointing is True
    assert _parse_pretrain_args(
        "--three_quarter_gradient_checkpointing"
    ).three_quarter_gradient_checkpointing is True
    assert _parse_formal_args(
        "--three-quarter-gradient-checkpointing"
    ).three_quarter_gradient_checkpointing is True


def test_dlc_launchers_use_three_quarter_gradient_checkpointing_by_default() -> None:
    common = (REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh").read_text()
    four_node = (REPO_ROOT / "pretrain_ppu_four_nodes_dlc.sh").read_text()
    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-three_quarter}"' in common
    assert "TRAIN_ARGS+=(--no_gradient_checkpointing)" in common
    assert "TRAIN_ARGS+=(--half_gradient_checkpointing)" in common
    assert "TRAIN_ARGS+=(--three_quarter_gradient_checkpointing)" in common
    assert "TRAIN_ARGS+=(--gradient_checkpointing)" in common
    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-three_quarter}"' in four_node


class _Tokenizer(nn.Module):
    def decode(self, z: torch.Tensor, patch_grid):
        del patch_grid
        repeats = (3072 + int(z.shape[-1]) - 1) // int(z.shape[-1])
        base = z.repeat(*([1] * (z.ndim - 1)), repeats)[..., :3072]
        return [base * (1.0 + 0.05 * level) for level in range(4)]


class _DepthHead(nn.Module):
    intermediate_layer_idx = (4, 11, 17, 23)

    def forward(self, tokens, images, patch_start_idx, image_hw=None):
        del images
        h, w = image_hw
        source = tokens[4][:, :, patch_start_idx:, 0].mean(dim=-1, keepdim=True)
        depth = (2.0 + 0.05 * source).view(*source.shape[:2], 1, 1, 1).expand(-1, -1, h, w, 1)
        # DPTHead's scalar prediction keeps a channel dimension, but its
        # confidence output does not: depth=[B,S,H,W,1], conf=[B,S,H,W].
        return depth, torch.ones_like(depth[..., 0])


class _SpatialConfidenceDepthHead(_DepthHead):
    """Depth-head stub whose frozen-teacher confidence varies spatially."""

    def forward(self, tokens, images, patch_start_idx, image_hw=None):
        depth, confidence = super().forward(
            tokens,
            images,
            patch_start_idx,
            image_hw=image_hw,
        )
        height, _ = image_hw
        confidence = confidence.clone()
        confidence[..., height // 2 :, :] = 0.05
        return depth, confidence


class _GaussianHead(nn.Module):
    intermediate_layer_idx = (4, 11, 17, 23)

    def forward(self, tokens, images, patch_start_idx, image_hw=None):
        del images
        h, w = image_hw
        source = tokens[4][:, :, patch_start_idx:, 0].mean(dim=-1, keepdim=True)
        b, s = source.shape[:2]
        value = source.view(b, s, 1, 1, 1).expand(b, s, h, w, 1)
        rgb = torch.cat((0.45 + 0.02 * value, 0.35 + 0.01 * value, 0.25 + 0.01 * value), dim=-1)
        opacity = torch.full_like(value, 0.65)
        scales = torch.full((b, s, h, w, 3), 0.08, device=value.device, dtype=value.dtype)
        quats = torch.zeros((b, s, h, w, 4), device=value.device, dtype=value.dtype)
        quats[..., 0] = 1.0
        gs_map = torch.cat((rgb, opacity, scales, quats), dim=-1)
        return gs_map, torch.ones((b, s, h, w), device=value.device, dtype=value.dtype)


class _InstanceHead(nn.Module):
    intermediate_layer_idx = (4, 11, 17, 23)

    def forward(self, tokens, images, patch_start_idx, image_hw=None):
        del images
        h, w = image_hw
        source = tokens[4][:, :, patch_start_idx:, 0].mean(dim=-1, keepdim=True)
        b, s = source.shape[:2]
        logits = source.view(b, s, 1, 1, 1).expand(b, s, h, w, 1) * 0.0
        return logits, torch.ones_like(logits)


class _SkyModel(nn.Module):
    def forward(self, images, extrinsics, intrinsics):
        del extrinsics, intrinsics
        b, s, _, h, w = images.shape
        assert b == 1
        return torch.zeros((s, h, w, 3), device=images.device, dtype=images.dtype)


class _FailIfCalledSkyModel(nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("formal GT-sky rendering must not call sky_model")


class _VGGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_tokenizer = _Tokenizer()
        self.depth_head = _DepthHead()
        self.gs_head = _GaussianHead()
        self.instance_head = _InstanceHead()
        self.sky_model = _SkyModel()


class _SceneFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self._pullback_calibration = PullbackCalibration(
            path=Path("/tmp/pullback.json"),
            tokenizer_generation="t0_v2",
            window_len=10,
            patch_grid_hw=(25, 37),
            depth_a=0.0,
            depth_b=0.0,
            reference_depth_m=20.0,
            runtime_depth_clamp_m=(0.5, 80.0),
            c_gs=1.0,
            depth_form="identity",
        )

    def denormalize(self, z):
        return z


class _ValidationVGGT(nn.Module):
    def get_aggregator_token_outputs(self, images):
        del images
        return {"patch_start_idx": 5}

    def semantic_head(self, tokens, images, patch_start_idx, image_hw=None):
        del tokens, images, patch_start_idx
        h, w = image_hw
        return torch.zeros((1, 3, h, w, 2)), None


class _SpatialLPIPS(nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean(dim=1, keepdim=True)


def test_primary_rgb_api_has_no_teacher_depth_argument():
    parameters = inspect.signature(compute_rgb_render_loss).parameters
    feedback_parameters = inspect.signature(
        compute_reconstruction_feedback_losses
    ).parameters
    assert "depth" not in parameters
    assert "render_pose_enc_dggt" in parameters
    assert parameters["conf_weight_power"].default == pytest.approx(1.0)
    assert parameters["conf_weight_floor"].default == pytest.approx(0.05)
    assert feedback_parameters["conf_weight_power"].default == pytest.approx(1.0)
    assert feedback_parameters["conf_weight_floor"].default == pytest.approx(0.05)


def test_teacher_render_camera_gradient_scale_rejects_nonzero_value():
    with pytest.raises(ValueError, match="camera_grad_scale must be 0"):
        compute_rgb_render_loss(
            vggt_model=None,
            scene_flow_root=None,
            z_clean_pred_n=torch.empty(0),
            images=torch.empty(0),
            timestamps=torch.empty(0),
            render_pose_enc_dggt=torch.empty(0),
            render_sky_probability=None,
            loss_sky_mask_gt=None,
            patch_grid=(1, 1),
            patch_start_idx=0,
            max_samples=0,
            max_frames=0,
            render_stride=1,
            background_mode="black",
            camera_grad_scale=0.5,
        )


def test_gradient_scaling_preserves_value_and_scales_gradient():
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    y = scale_gradient(x, 0.125)
    torch.testing.assert_close(y, x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.125))


def test_torch_unprojection_keeps_depth_gradient():
    depth = torch.full((1, 3, 4), 2.0, requires_grad=True)
    w2c = torch.eye(4).view(1, 4, 4)
    k = torch.tensor([[[10.0, 0.0, 1.5], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]]])
    points = torch_unproject_depth(depth, w2c, k)
    points.square().mean().backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert float(depth.grad.abs().sum()) > 0.0


def test_masked_lpips_excludes_sky_and_keeps_foreground_signal():
    target = torch.zeros((1, 3, 8, 8))
    prediction = target.clone()
    prediction[..., :4, :] = 1.0
    foreground = torch.zeros((1, 1, 8, 8))
    loss_sky_only = _masked_lpips(_SpatialLPIPS(), prediction, target, foreground)
    torch.testing.assert_close(loss_sky_only, torch.zeros_like(loss_sky_only))

    foreground[..., :4, :] = 1.0
    loss_foreground = _masked_lpips(_SpatialLPIPS(), prediction, target, foreground)
    assert float(loss_foreground) > 0.5


def test_masked_lpips_applies_sigma_weight_per_sample():
    prediction = torch.ones((2, 3, 4, 4), requires_grad=True)
    target = torch.zeros_like(prediction)
    foreground = torch.ones((2, 1, 4, 4))
    loss = _masked_lpips(
        _SpatialLPIPS(),
        prediction,
        target,
        foreground,
        sample_weight=torch.tensor([1.0, 0.0]),
    )
    loss.backward()
    assert prediction.grad is not None
    assert float(prediction.grad[0].abs().sum()) > 0.0
    torch.testing.assert_close(prediction.grad[1], torch.zeros_like(prediction.grad[1]))


def test_gsplat_compositing_does_not_multiply_alpha_twice():
    premultiplied = torch.tensor([[[[0.5, 0.25, 0.125]]]])
    alpha = torch.tensor([[[[0.5]]]])
    background = torch.zeros_like(premultiplied)
    result = composite_gsplat_rgb(premultiplied, alpha, background)
    torch.testing.assert_close(result, premultiplied)


def test_formal_context_ignores_teacher_depth():
    item = {
        "cache_path": "dummy.pt",
        "rgb_training": {
            "images": torch.zeros((2, 3, 8, 8)),
            "masks": torch.zeros((2, 1, 8, 8)),
            "timestamps": torch.arange(2),
        },
        "predictions": {
            "pose_enc": torch.zeros((1, 2, 9)),
            "depth": torch.full((1, 2, 8, 8, 1), 12345.0),
            "patch_start_idx": 5,
        },
    }
    context = _formal_rgb_context(item, device=torch.device("cpu"), strict=True)
    assert context is not None
    assert "rgb_render_depth" not in context
    assert not any("depth" in key for key in context)


def test_validation_pose_uses_cached_full_context_then_slices_window():
    full_pose = torch.arange(29 * 9, dtype=torch.float32).reshape(29, 9)
    subset = torch.tensor([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    selected = _cached_render_pose_from_payload(
        {"pass1": {"pose_enc": full_pose}},
        subset,
    )
    torch.testing.assert_close(selected, full_pose[subset].unsqueeze(0))

    # Dataset items already contain the selected pose; subset_frames are
    # original clip indices and must not be applied a second time.
    item = {
        "cache_path": "validation.pt",
        "subset_frames": subset,
        "predictions": {"pose_enc": selected},
    }
    torch.testing.assert_close(cached_render_pose_from_item(item), selected)


def test_visualization_batch_requires_matching_cached_pose():
    sample = {
        "images": torch.zeros((3, 3, 4, 5)),
        "masks": torch.zeros((3, 3, 4, 5)),
        "timestamps": torch.arange(3),
    }
    pose = torch.zeros((1, 3, 9))
    batch = _prepare_visualization_batch(sample, render_pose_enc_dggt=pose)
    assert batch["render_pose_enc_dggt"] is pose

    with pytest.raises(ValueError, match="pose_enc must be"):
        _prepare_visualization_batch(
            sample,
            render_pose_enc_dggt=torch.zeros((1, 2, 9)),
        )


def test_generated_validation_renderer_uses_cached_pose_without_camera_head(monkeypatch):
    seq_len, height, width = 3, 4, 5
    cached_pose = torch.arange(seq_len * 9, dtype=torch.float32).reshape(1, seq_len, 9)
    batch = {
        "images": torch.zeros((1, seq_len, 3, height, width)),
        "masks": torch.zeros((1, seq_len, 3, height, width)),
        "timestamps": torch.arange(seq_len).unsqueeze(0),
        "render_pose_enc_dggt": cached_pose,
    }
    geometry = SimpleNamespace(
        dino_tokens=None,
        depth=torch.zeros((1, seq_len, height, width, 1)),
        dynamic_conf=torch.zeros((1, seq_len, height, width, 1)),
        gs_map=torch.zeros((1, seq_len, height, width, 11)),
        gs_conf=torch.ones((1, seq_len, height, width)),
    )
    captured: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(formal_train, "decode_generated_dggt_geometry", lambda **kwargs: geometry)
    monkeypatch.setattr(
        formal_train,
        "_semantic_logits_to_sky_mask",
        lambda logits: torch.zeros((1, seq_len, 1, height, width)),
    )

    def fake_render(*args, **kwargs):
        del kwargs
        captured["pose"] = args[4]
        return torch.zeros((seq_len, 3, height, width))

    monkeypatch.setattr(formal_train, "_render_gs_map_rgb", fake_render)
    formal_train.render_validation_generated_rgb_gt_sky(
        batch,
        _ValidationVGGT(),  # Intentionally has no CameraHead.
        _SceneFlow(),
        torch.zeros((1, seq_len, 1, 4)),
        SimpleNamespace(precision="fp32", val_log_images=seq_len, patch_grid=(1, 1)),
        torch.device("cpu"),
    )
    torch.testing.assert_close(captured["pose"], cached_pose)


def test_fast_flow_inputs_slice_asset_coverage_with_video_frames():
    payload = {
        "M_preserve": torch.zeros((6, 2, 1)),
        "M_source": torch.zeros((6, 2, 1)),
        "M_dest": torch.zeros((6, 2, 1)),
        "scaffold_pooled": torch.zeros((6, 2, 3)),
        "phase1_coverage": torch.tensor(
            [[False, True, False, True, False, True], [True, False, True, False, True, False]]
        ),
        "phase4_slots": [0, 1],
    }
    subset = torch.tensor([2, 4])
    out = WaymoFlowCacheDataset._subset_flow_inputs(payload, subset)
    torch.testing.assert_close(out["phase1_coverage"], payload["phase1_coverage"][:, subset])
    assert tuple(out["phase1_coverage"].shape) == (2, 2)

    # Chunked loaders return metadata already sliced; local frame indices must
    # not accidentally apply a second temporal selection.
    payload_local = dict(payload)
    payload_local["phase1_coverage"] = payload["phase1_coverage"][:, subset]
    out_local = WaymoFlowCacheDataset._subset_flow_inputs(payload_local, torch.arange(2))
    torch.testing.assert_close(out_local["phase1_coverage"], payload_local["phase1_coverage"])


def test_rgb_schedule_has_true_delay_and_period():
    args = SimpleNamespace(
        lambda_rgb_render=0.01,
        rgb_render_every=4,
        rgb_render_start_step=10,
        rgb_render_warmup_steps=20,
    )
    assert not should_apply_rgb_render_loss(args, 8, training=True)
    assert not should_apply_rgb_render_loss(args, 11, training=True)
    assert should_apply_rgb_render_loss(args, 12, training=True)
    assert rgb_render_loss_ramp(args, 10) == 0.0
    assert rgb_render_loss_ramp(args, 20) == 0.5
    assert rgb_render_loss_ramp(args, 30) == 1.0


def test_rgb_render_sigma_weight_smoothly_attenuates_high_noise():
    sigmas = torch.tensor([0.0, 0.25, 0.5, 0.9, 1.0])
    torch.testing.assert_close(
        rgb_render_sigma_weight(sigmas, power=1.0),
        torch.tensor([1.0, 0.75, 0.5, 0.1, 0.0]),
    )
    torch.testing.assert_close(
        rgb_render_sigma_weight(sigmas, power=2.0),
        torch.tensor([1.0, 0.75**2, 0.5**2, 0.1**2, 0.0]),
    )
    torch.testing.assert_close(
        rgb_render_sigma_weight(sigmas),
        torch.tensor([1.0, 0.75**2, 0.5**2, 0.1**2, 0.0]),
    )
    torch.testing.assert_close(
        rgb_render_sigma_weight(sigmas, power=0.0),
        torch.ones_like(sigmas),
    )


@pytest.mark.parametrize("power", [-1.0, float("nan"), float("inf")])
def test_rgb_render_sigma_weight_rejects_invalid_power(power):
    with pytest.raises(ValueError, match="finite and non-negative"):
        rgb_render_sigma_weight(torch.tensor([0.5]), power=power)


def test_pretrain_rgb_defaults_use_every_full_resolution_frame():
    from train_scene_flow_pretrain import build_argparser

    args = build_argparser().parse_args(
        [
            "--image_dir",
            "/tmp/images",
            "--hdmap_root",
            "/tmp/hdmap",
            "--dggt_ckpt_path",
            "/tmp/dggt.pt",
            "--feature_stats_path",
            "/tmp/stats.pt",
            "--scene_gauge_path",
            "/tmp/scene_gauge.json",
            "--pullback_calibration_path",
            "/tmp/pullback.json",
            "--log_dir",
            "/tmp/logs",
        ]
    )
    assert args.rgb_render_start_step == 5000
    assert args.rgb_render_every == 1
    assert args.rgb_render_max_frames == 0  # 0 means every frame in the clip.
    assert args.rgb_render_stride == 1
    assert args.rgb_render_sigma_power == 2.0
    # World-feedback balance.  These three moved together with the switch to
    # ``--head_dynamic_space probability``: at 0.1/0.1/0.1 the head term was
    # 97.8% unbounded dynamic logit, and the parts that describe the rendered
    # scene came to 0.035% of the training loss.
    # All three read the same decode of the same predicted latent, so they carry
    # the same weight.  0.05 and the 0.1 that briefly replaced it were both set
    # against the wrong reference (the unused 0.005 code default, then v5's
    # launcher pin), and left the only term that reads pixels 23x below the one
    # that reads the frozen heads.
    assert args.lambda_rgb_render == pytest.approx(1.0)
    assert args.lambda_level_consistency == pytest.approx(1.0)
    assert args.lambda_head_consistency == pytest.approx(1.0)
    assert args.lambda_rgb_render == pytest.approx(args.lambda_head_consistency)
    # L1 and L2 read the same decode under the same sigma weighting, so they
    # stay equal to each other the way v5 had them.
    assert args.lambda_level_consistency == pytest.approx(args.lambda_head_consistency)
    assert args.head_dynamic_space == "probability"
    assert args.feedback_conf_weight_power == pytest.approx(1.0)
    assert args.feedback_conf_weight_floor == pytest.approx(0.05)
    assert args.val_sample_steps == 50
    assert not hasattr(args, "render_use_" + "predicted_gauge")


def test_training_render_pose_uses_requested_c_and_predicted_gauge_rows() -> None:
    requested = torch.eye(4).reshape(1, 1, 4, 4).repeat(3, 2, 1, 1)
    requested[:, :, 0, 3] = torch.tensor(
        [[1.0, 2.0], [10.0, 11.0], [20.0, 22.0]]
    )
    anchor = torch.eye(4).reshape(1, 4, 4).repeat(3, 1, 1)
    gauge = torch.tensor(
        [[[2.0, -0.7, -0.8]], [[3.0, -0.6, -0.9]], [[4.0, -0.5, -1.0]]],
        requires_grad=True,
    )
    rows = torch.tensor([2, 0])
    bundle = SimpleNamespace(
        camera_to_world_requested_metric=requested,
        camera_trajectory_anchor_to_world_metric=anchor,
        unrelated_pose=torch.full((3, 2, 9), 777.0),
    )
    actual = pretrain_train.requested_render_pose_for_rows(bundle, gauge, rows)
    selected_c = requested.index_select(0, rows)
    selected_anchor = anchor.index_select(0, rows)
    selected_gauge = gauge.index_select(0, rows)
    expected = assemble_dggt_pose_encoding(
        metric_c2w_to_teacher_anchor_dggt(
            selected_c,
            selected_anchor,
            selected_gauge[..., 0],
        ),
        selected_gauge,
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert not bool((actual == 777.0).any())
    actual.sum().backward()
    assert gauge.grad is not None
    assert bool((gauge.grad.index_select(0, rows).abs() > 0).any())

    train_source = inspect.getsource(pretrain_train.train_step)
    assert "requested_render_pose_for_rows(" in train_source
    assert "render_pose_enc_dggt=render_pose_requested" in train_source
    assert "render_pose_enc_dggt=render_pose_requested" in train_source
    assert "render_pose_enc_" + "teacher_gauge" not in train_source
    assert "render_use_" + "predicted_gauge" not in train_source


def test_pretrain_bundle_camera_condition_uses_canvas_k_and_canvas_hw() -> None:
    raw = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(2, 3, 2, 1, 1)
    raw[:, :, :, 0, 0] = 1000.0
    raw[:, :, :, 1, 1] = 1000.0
    selected = pretrain_train._requested_front_intrinsics(
        raw,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
        name="camera_intrinsics_canvas",
    )
    torch.testing.assert_close(selected, raw[:, :, 0])

    # Raw Waymo K is a scene-level calibration and is intentionally stored
    # with a singleton time axis.  Collation must expand only that axis; this
    # is not permission to broadcast malformed batch/camera dimensions.
    scene_static = raw[:, :1, 0].clone()
    expanded = pretrain_train._requested_front_intrinsics(
        scene_static,
        batch_size=2,
        seq_len=3,
        device=torch.device("cpu"),
        name="intrinsics",
    )
    assert expanded.shape == (2, 3, 3, 3)
    torch.testing.assert_close(expanded, scene_static.expand(-1, 3, -1, -1))

    with pytest.raises(ValueError, match=r"shape .* != \(2, 3, 3, 3\)"):
        pretrain_train._requested_front_intrinsics(
            raw[:, :2, 0],
            batch_size=2,
            seq_len=3,
            device=torch.device("cpu"),
            name="intrinsics",
        )

    source = inspect.getsource(pretrain_train.build_pretrain_bundle_from_batch)
    assert 'batch.get("camera_intrinsics_canvas")' in source
    assert "requested_intrinsics_canvas," in source
    assert "image_hw=(int(images.shape[-2]), int(images.shape[-1]))" in source
    assert "camera_intrinsics_requested_canvas_metric" in source
    assert "camera_intrinsics_requested_raw_metric" in source


def test_formal_rgb_defaults_match_pretrain_feedback_schedule():
    from train_scene_flow import build_argparser

    args = build_argparser().parse_args(
        [
            "--ckpt_path",
            "/tmp/dggt.pt",
            "--log_dir",
            "/tmp/logs",
        ]
    )
    assert args.rgb_render_start_step == 5000
    assert args.rgb_render_every == 1
    assert args.rgb_render_max_samples == 1
    assert args.rgb_render_max_frames == 0
    assert args.rgb_render_stride == 1
    assert args.rgb_render_sigma_power == 2.0
    assert args.lambda_rgb_render == pytest.approx(0.1)
    assert args.lambda_level_consistency == pytest.approx(0.1)
    assert args.lambda_head_consistency == pytest.approx(0.1)
    assert args.feedback_conf_weight_power == pytest.approx(1.0)
    assert args.feedback_conf_weight_floor == pytest.approx(0.05)


def test_pretrain_launch_scripts_do_not_override_rgb_coverage_defaults():
    forbidden_flags = (
        "--rgb_render_every",
        "--rgb_render_max_frames",
        "--rgb_render_stride",
        # The three world-feedback weights are a balance argument that lives
        # with the loss code.  A launcher override silently pins one side of it
        # across a rebalance, which is how v6 ended up running the render loss
        # at 1/20 of the value it had just connected a new gradient path to.
        "--lambda_rgb_render",
        "--lambda_level_consistency",
        "--lambda_head_consistency",
    )
    for script_name in PRETRAIN_LAUNCH_SCRIPTS:
        script = (REPO_ROOT / script_name).read_text()
        for flag in forbidden_flags:
            assert flag not in script, f"{script_name} must inherit the pretrain default for {flag}"


def test_pretrain_launch_scripts_freeze_t59_validation_to_50_steps() -> None:
    for script_name in PRETRAIN_LAUNCH_SCRIPTS:
        script = (REPO_ROOT / script_name).read_text()
        assert "v6" in script, f"{script_name} must keep v6 log/wandb naming"
        if script_name != "pretrain_ppu_four_nodes_dlc.sh":
            assert "WANDB_API_KEY" in script, (
                f"{script_name} must preserve its existing WANDB_API_KEY handoff"
            )
        assert "VAL_SAMPLE_STEPS:-35" not in script
        assert "--val_sample_steps 35" not in script
        assert (
            "VAL_SAMPLE_STEPS:-50" in script
            or "--val_sample_steps 50" in script
        ), f"{script_name} must explicitly select the accepted T59 50-step regime"


def test_pretrain_resume_contract_fails_closed_on_tasks_rgb_and_args() -> None:
    args = _parse_pretrain_args()
    payload = {
        "pretrain_resume_contract_version": (
            pretrain_train.PRETRAIN_RESUME_CONTRACT_VERSION
        ),
        "pretrain_resume_reproducibility": (
            pretrain_train.PRETRAIN_RESUME_REPRODUCIBILITY
        ),
        "layout_task_probabilities": list(
            pretrain_train.LAYOUT_TASK_PROBABILITIES
        ),
        "rgb_render": pretrain_train.rgb_render_run_summary(args),
        "pretrain_resume_critical_args": (
            pretrain_train.pretrain_resume_critical_args(args)
        ),
        "args": vars(args).copy(),
    }
    pretrain_train.validate_pretrain_resume_contract(
        payload, args, "checkpoint.pt"
    )

    changed_tasks = dict(payload)
    changed_tasks["layout_task_probabilities"] = [0.2, 0.4, 0.4]
    with pytest.raises(ValueError, match="task probabilities"):
        pretrain_train.validate_pretrain_resume_contract(
            changed_tasks, args, "checkpoint.pt"
        )

    changed_rgb = dict(payload)
    changed_rgb["rgb_render"] = dict(payload["rgb_render"])
    changed_rgb["rgb_render"]["lambda_rgb_render"] = 0.00025
    with pytest.raises(ValueError, match="RGB/HDS"):
        pretrain_train.validate_pretrain_resume_contract(
            changed_rgb, args, "checkpoint.pt"
        )

    changed_args = dict(payload)
    changed_args["args"] = dict(payload["args"])
    changed_args["args"]["layout_max_actors"] = 7
    with pytest.raises(ValueError, match="argparse values"):
        pretrain_train.validate_pretrain_resume_contract(
            changed_args, args, "checkpoint.pt"
        )


def test_directional_sky_projection_matches_validation_path():
    from train_scene_flow_pretrain import render_sky_tokens_directional_background

    torch.manual_seed(7)
    seq_len, height, width = 2, 11, 17
    # Both renderers take the decoded RGB atlas, one row per atlas direction.
    sky_tokens = torch.rand((1, 32, 3)) * 2.0 - 1.0
    world_to_camera = torch.eye(4).view(1, 1, 4, 4).expand(1, seq_len, 4, 4).clone()
    world_to_camera[0, 1, 0, 3] = 0.2
    intrinsics = torch.tensor(
        [[[[20.0, 0.0, 8.5], [0.0, 18.0, 5.5], [0.0, 0.0, 1.0]]]]
    ).expand(1, seq_len, 3, 3).clone()
    training = sky_tokens_to_background(
        sky_tokens,
        seq_len=seq_len,
        height=height,
        width=width,
        grid_h=4,
        grid_w=8,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
    )
    validation = render_sky_tokens_directional_background(
        sky_tokens,
        seq_len=seq_len,
        height=height,
        width=width,
        grid_h=4,
        grid_w=8,
        extrinsics=world_to_camera,
        intrinsics=intrinsics,
    )
    torch.testing.assert_close(training, validation, rtol=0.0, atol=1.0e-6)


def test_generated_depth_head_is_differentiable_on_cpu():
    z = torch.randn((1, 2, 1, 4), requires_grad=True)
    geometry = decode_generated_dggt_geometry(
        vggt_model=_VGGT(),
        scene_flow_root=_SceneFlow(),
        z_clean_pred_n=z,
        patch_grid=(1, 1),
        patch_start_idx=1,
        image_hw=(8, 8),
    )
    geometry.depth.mean().backward()
    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0


def test_long_generated_geometry_decode_never_calls_tokenizer_above_calibrated_window():
    class _WindowSpyTokenizer(_Tokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.lengths: list[int] = []

        def decode(self, z: torch.Tensor, patch_grid):
            self.lengths.append(int(z.shape[1]))
            return super().decode(z, patch_grid)

    vggt = _VGGT()
    vggt.scene_tokenizer = _WindowSpyTokenizer()
    z = torch.randn((1, 29, 1, 4), requires_grad=True)
    geometry = decode_generated_dggt_geometry(
        vggt_model=vggt,
        scene_flow_root=_SceneFlow(),
        z_clean_pred_n=z,
        patch_grid=(1, 1),
        patch_start_idx=1,
        image_hw=(4, 4),
        tokenizer_window_len=10,
    )
    assert len(vggt.scene_tokenizer.lengths) > 1
    assert max(vggt.scene_tokenizer.lengths) <= 10
    assert geometry.depth.shape[:2] == (1, 29)
    geometry.depth.mean().backward()
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())


def _feedback_geometry(z: torch.Tensor, *, image_hw: tuple[int, int] = (8, 8)):
    return decode_generated_dggt_geometry(
        vggt_model=_VGGT(),
        scene_flow_root=_SceneFlow(),
        z_clean_pred_n=z,
        patch_grid=(1, 1),
        patch_start_idx=1,
        image_hw=image_hw,
    )


def _spatial_conf_feedback_geometry(
    z: torch.Tensor,
    *,
    image_hw: tuple[int, int] = (8, 8),
):
    vggt = _VGGT()
    vggt.depth_head = _SpatialConfidenceDepthHead()
    return decode_generated_dggt_geometry(
        vggt_model=vggt,
        scene_flow_root=_SceneFlow(),
        z_clean_pred_n=z,
        patch_grid=(1, 1),
        patch_start_idx=1,
        image_hw=image_hw,
    )


def _compute_feedback_with_conf_power(
    student,
    teacher,
    power: float | None,
):
    kwargs = {
        "student_geometry": student,
        "teacher_geometry": teacher,
        "patch_grid": (1, 1),
        "patch_weight_mask": torch.ones((1, 1, 1, 1)),
        "loss_sky_mask_gt": torch.zeros((1, 1, 1, 8, 8)),
        "sky_weight": 0.0,
        "max_frames": 0,
        "render_stride": 1,
        "sample_weight": torch.ones(1),
    }
    if power is not None:
        kwargs["conf_weight_power"] = power
    return compute_reconstruction_feedback_losses(**kwargs)


def test_reconstruction_feedback_is_zero_at_clean_target():
    z = torch.tensor([[[[0.2, -0.4, 0.6, 0.1]], [[-0.1, 0.3, 0.7, -0.5]]]])
    student = _feedback_geometry(z)
    with torch.no_grad():
        teacher = _feedback_geometry(z.detach())
    result = compute_reconstruction_feedback_losses(
        student_geometry=student,
        teacher_geometry=teacher,
        patch_grid=(1, 1),
        patch_weight_mask=torch.ones((1, 2, 1, 1)),
        loss_sky_mask_gt=torch.zeros((1, 2, 1, 8, 8)),
        sky_weight=0.0,
        max_frames=0,
        render_stride=1,
        sample_weight=torch.ones(1),
    )
    assert torch.isfinite(result.level_loss)
    assert torch.isfinite(result.head_loss)
    assert float(result.level_loss.abs()) < 1.0e-6
    assert float(result.head_loss.abs()) < 1.0e-6


def test_reconstruction_feedback_sigma_weight_attenuates_whole_sample():
    target_row = torch.tensor([[[[0.2, -0.4, 0.6, 0.1]], [[-0.1, 0.3, 0.7, -0.5]]]])
    target = target_row.expand(2, -1, -1, -1).clone()
    prediction = target.clone()
    prediction[..., 0] += 0.25
    student = _feedback_geometry(prediction)
    with torch.no_grad():
        teacher = _feedback_geometry(target)
    result = compute_reconstruction_feedback_losses(
        student_geometry=student,
        teacher_geometry=teacher,
        patch_grid=(1, 1),
        patch_weight_mask=torch.ones((2, 2, 1, 1)),
        loss_sky_mask_gt=torch.zeros((2, 2, 1, 8, 8)),
        sky_weight=0.0,
        max_frames=0,
        render_stride=1,
        sample_weight=torch.tensor([1.0, 0.0]),
    )
    assert result.logs["loss_level_consistency_unweighted"] > 0.0
    assert result.logs["loss_head_consistency_unweighted"] > 0.0
    assert result.logs["loss_level_consistency"] == pytest.approx(
        0.5 * result.logs["loss_level_consistency_unweighted"], rel=1.0e-5
    )
    assert result.logs["loss_head_consistency"] == pytest.approx(
        0.5 * result.logs["loss_head_consistency_unweighted"], rel=1.0e-5
    )


def test_feedback_power_zero_is_bit_identical_to_no_confidence_path(monkeypatch):
    z = torch.zeros((1, 1, 1, 4))
    student = _spatial_conf_feedback_geometry(z)
    with torch.no_grad():
        teacher = _spatial_conf_feedback_geometry(z)
    student.depth = teacher.depth.detach().clone()
    student.depth[..., 4:, :, 0] = student.depth[..., 4:, :, 0] + 1.0

    original_conf_weight = feedback_loss_module._teacher_conf_weight
    disabled = _compute_feedback_with_conf_power(student, teacher, 0.0)
    monkeypatch.setattr(
        feedback_loss_module,
        "_teacher_conf_weight",
        lambda *args, **kwargs: None,
    )
    legacy = _compute_feedback_with_conf_power(student, teacher, None)

    assert torch.equal(disabled.level_loss, legacy.level_loss)
    assert torch.equal(disabled.head_loss, legacy.head_loss)
    assert disabled.logs == legacy.logs

    monkeypatch.setattr(
        feedback_loss_module,
        "_slice_dense",
        lambda *args, **kwargs: pytest.fail("power=0 must short-circuit before slicing"),
    )
    assert original_conf_weight(
        teacher.depth_conf,
        stride=1,
        power=0.0,
        floor=float("nan"),
    ) is None


def test_teacher_depth_conf_downweights_head_error_in_low_confidence_region():
    z = torch.zeros((1, 1, 1, 4))
    student = _spatial_conf_feedback_geometry(z)
    with torch.no_grad():
        teacher = _spatial_conf_feedback_geometry(z)
    student.depth = teacher.depth.detach().clone()
    student.depth[..., 4:, :, 0] = student.depth[..., 4:, :, 0] + 1.0

    disabled = _compute_feedback_with_conf_power(student, teacher, 0.0)
    weighted = _compute_feedback_with_conf_power(student, teacher, 1.0)

    assert disabled.logs["loss_head_consistency"] > 0.0
    assert weighted.logs["loss_head_consistency"] < 0.2 * disabled.logs[
        "loss_head_consistency"
    ]
    assert weighted.logs["loss_head_consistency_no_conf"] == pytest.approx(
        disabled.logs["loss_head_consistency"], rel=0.0, abs=0.0
    )


def test_teacher_depth_conf_weight_is_detached():
    z = torch.zeros((1, 1, 1, 4))
    student = _spatial_conf_feedback_geometry(z)
    with torch.no_grad():
        teacher = _spatial_conf_feedback_geometry(z)
    depth_delta = torch.zeros_like(student.depth)
    depth_delta[..., 4:, :, 0] = 1.0
    student.depth = (teacher.depth.detach() + depth_delta).requires_grad_()
    teacher_conf = teacher.depth_conf.detach().clone().requires_grad_()
    teacher.depth_conf = teacher_conf

    weighted = _compute_feedback_with_conf_power(student, teacher, 1.0)
    weighted.head_loss.backward()

    assert student.depth.grad is not None
    assert float(student.depth.grad.abs().sum()) > 0.0
    assert teacher_conf.grad is None


def test_confidence_head_errors_are_not_reweighted_by_teacher_confidence():
    z = torch.zeros((1, 1, 1, 4))
    student = _spatial_conf_feedback_geometry(z)
    with torch.no_grad():
        teacher = _spatial_conf_feedback_geometry(z)
    student.depth_conf = teacher.depth_conf.detach().clone()
    student.depth_conf[..., 4:, :] = 0.5
    student.gs_conf = teacher.gs_conf.detach().clone()
    student.gs_conf[..., 4:, :] = 0.0

    disabled = _compute_feedback_with_conf_power(student, teacher, 0.0)
    weighted = _compute_feedback_with_conf_power(student, teacher, 1.0)

    assert weighted.logs["loss_head_depth_conf"] == disabled.logs[
        "loss_head_depth_conf"
    ]
    assert weighted.logs["loss_head_gs_conf"] == disabled.logs["loss_head_gs_conf"]
    assert weighted.logs["loss_head_consistency"] == disabled.logs[
        "loss_head_consistency"
    ]


def test_teacher_depth_conf_only_changes_rgb_photometric_weight(monkeypatch):
    def fake_render_one_sample(**kwargs):
        images = kwargs["images"]
        frames = int(images.shape[0])
        height, width = int(images.shape[-2]), int(images.shape[-1])
        rendered = images.new_zeros((frames, 3, height, width))
        alpha = images.new_zeros((frames, 1, height, width))
        return rendered, alpha

    monkeypatch.setattr(rgb_render_module, "_render_one_sample", fake_render_one_sample)
    vggt = _VGGT()
    vggt.depth_head = _SpatialConfidenceDepthHead()
    z = torch.zeros((1, 1, 1, 4))
    images = torch.zeros((1, 1, 3, 8, 8))
    images[..., 4:, :] = 1.0
    pose = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]
    )
    common = {
        "vggt_model": vggt,
        "scene_flow_root": _SceneFlow(),
        "z_clean_pred_n": z,
        "z_clean_target_n": z,
        "images": images,
        "timestamps": torch.zeros((1, 1)),
        "render_pose_enc_dggt": pose,
        "render_sky_probability": torch.zeros((1, 1, 1, 8, 8)),
        "loss_sky_mask_gt": torch.zeros((1, 1, 1, 8, 8)),
        "patch_grid": (1, 1),
        "patch_start_idx": 1,
        "max_samples": 1,
        "max_frames": 0,
        "render_stride": 1,
        "background_mode": "black",
        "patch_weight_mask": torch.ones((1, 1, 1, 1)),
    }

    disabled = compute_rgb_render_loss(**common, conf_weight_power=0.0)
    weighted = compute_rgb_render_loss(**common, conf_weight_power=1.0)

    assert weighted.logs["loss_rgb_render"] < 0.2 * disabled.logs["loss_rgb_render"]
    assert weighted.logs["rgb_render_weight_mean"] == disabled.logs[
        "rgb_render_weight_mean"
    ]
    assert weighted.logs["loss_rgb_render_no_conf"] == pytest.approx(
        disabled.logs["loss_rgb_render"], rel=0.0, abs=0.0
    )


def test_rgb_feedback_teacher_is_stop_grad_and_student_reuses_render_heads(monkeypatch):
    calls = {"decode": 0}
    original_decode = rgb_render_module.decode_generated_dggt_geometry

    def counted_decode(**kwargs):
        calls["decode"] += 1
        return original_decode(**kwargs)

    def fake_render_one_sample(**kwargs):
        images = kwargs["images"]
        frames = (
            int(images.shape[0])
            if int(kwargs["max_frames"]) <= 0
            else min(int(kwargs["max_frames"]), int(images.shape[0]))
        )
        stride = max(1, int(kwargs["stride"]))
        height = (int(images.shape[-2]) + stride - 1) // stride
        width = (int(images.shape[-1]) + stride - 1) // stride
        rendered = images.new_zeros((frames, 3, height, width))
        alpha = images.new_zeros((frames, 1, height, width))
        return rendered, alpha

    monkeypatch.setattr(rgb_render_module, "decode_generated_dggt_geometry", counted_decode)
    monkeypatch.setattr(rgb_render_module, "_render_one_sample", fake_render_one_sample)

    z_target = torch.tensor(
        [[[[0.2, -0.4, 0.6, 0.1]], [[-0.1, 0.3, 0.7, -0.5]]]],
        requires_grad=True,
    )
    z_prediction = (z_target.detach() + torch.tensor([0.25, 0.0, 0.0, 0.0])).requires_grad_()
    images = torch.zeros((1, 2, 3, 8, 8))
    pose = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]
    )
    result = compute_rgb_render_loss(
        vggt_model=_VGGT(),
        scene_flow_root=_SceneFlow(),
        z_clean_pred_n=z_prediction,
        z_clean_target_n=z_target,
        images=images,
        timestamps=torch.tensor([[0.0, 1.0]]),
        render_pose_enc_dggt=pose,
        render_sky_probability=torch.zeros((1, 2, 1, 8, 8)),
        loss_sky_mask_gt=torch.zeros((1, 2, 1, 8, 8)),
        patch_grid=(1, 1),
        patch_start_idx=1,
        max_samples=1,
        max_frames=0,
        render_stride=1,
        background_mode="black",
        patch_weight_mask=torch.ones((1, 2, 1, 1)),
        loss_sample_weight=torch.tensor([0.5]),
    )
    assert calls["decode"] == 2  # one shared student decode + one no-grad teacher decode
    assert float(result.level_loss.detach()) > 0.0
    assert float(result.head_loss.detach()) > 0.0
    (result.level_loss + result.head_loss).backward()
    assert z_prediction.grad is not None
    assert torch.isfinite(z_prediction.grad).all()
    assert float(z_prediction.grad.abs().sum()) > 0.0
    assert z_target.grad is None


def test_formal_training_adds_weighted_render_level_and_head_losses(monkeypatch):
    captured: dict[str, torch.Tensor] = {}

    def fake_rgb_loss(**kwargs):
        captured["target"] = kwargs["z_clean_target_n"]
        reference = kwargs["z_clean_pred_n"]
        return RGBRenderLossResult(
            loss=reference.sum() * 0.0 + 2.0,
            level_loss=reference.sum() * 0.0 + 3.0,
            head_loss=reference.sum() * 0.0 + 4.0,
            logs={
                "loss_rgb_render": 2.0,
                "loss_level_consistency": 3.0,
                "loss_head_consistency": 4.0,
            },
        )

    monkeypatch.setattr(formal_train, "compute_rgb_render_loss", fake_rgb_loss)
    z_pred = torch.zeros((1, 2, 1, 4), requires_grad=True)
    z_clean = torch.ones_like(z_pred)
    bundle = SimpleNamespace(
        z_clean_n=z_clean,
        rgb_render_images=torch.zeros((1, 2, 3, 8, 8)),
        rgb_render_masks=torch.zeros((1, 2, 1, 8, 8)),
        rgb_render_pose_enc_dggt=torch.zeros((1, 2, 9)),
        rgb_render_timestamps=torch.tensor([[0.0, 1.0]]),
        patch_grid=(1, 1),
        patch_start_idx=1,
    )
    target = SimpleNamespace(
        sigmas=torch.tensor([0.5]),
        M_edit=torch.ones((1, 2, 1, 1)),
    )
    args = SimpleNamespace(
        rgb_render_max_samples=1,
        rgb_render_sigma_power=1.0,
        rgb_render_max_frames=0,
        rgb_render_stride=1,
        sky_grid=(4, 8),
        rgb_render_lpips_weight=0.0,
        rgb_render_start_step=0,
        rgb_render_warmup_steps=0,
        lambda_rgb_render=0.01,
        lambda_level_consistency=0.02,
        lambda_head_consistency=0.03,
    )
    metrics: dict[str, float] = {}
    result = formal_train._add_formal_rgb_render_loss(
        z_pred.sum() * 0.0,
        metrics,
        args=args,
        global_step=1,
        active=True,
        render_vggt_model=_VGGT(),
        scene_flow_root=_SceneFlow(),
        z_pred=z_pred,
        bundle=bundle,
        target=target,
        lpips_model=None,
    )
    torch.testing.assert_close(captured["target"], z_clean)
    assert float(result.detach()) == pytest.approx(0.01 * 2.0 + 0.02 * 3.0 + 0.03 * 4.0)
    assert metrics["loss_level_consistency_weighted"] == pytest.approx(0.06)
    assert metrics["loss_head_consistency_weighted"] == pytest.approx(0.12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA gradient test")
def test_full_rgb_loss_backpropagates_through_generated_depth_and_gs():
    pytest.importorskip("gsplat")
    device = torch.device("cuda:0")
    z = torch.randn((1, 2, 1, 4), device=device, requires_grad=True)
    images = torch.zeros((1, 2, 3, 16, 16), device=device)
    ramp = torch.linspace(0.1, 0.9, 16, device=device).view(1, 1, 1, 1, 16)
    images = images + ramp
    pose = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]],
        device=device,
    )
    result = compute_rgb_render_loss(
        vggt_model=_VGGT().to(device),
        scene_flow_root=_SceneFlow().to(device),
        z_clean_pred_n=z,
        images=images,
        timestamps=torch.tensor([[0.0, 1.0]], device=device),
        render_pose_enc_dggt=pose,
        render_sky_probability=torch.zeros((1, 2, 1, 16, 16), device=device),
        loss_sky_mask_gt=torch.zeros((1, 2, 1, 16, 16), device=device),
        patch_grid=(1, 1),
        patch_start_idx=1,
        max_samples=1,
        max_frames=2,
        render_stride=2,
        background_mode="black",
        patch_weight_mask=torch.ones((1, 2, 1, 1), device=device),
        loss_sample_weight=torch.tensor([0.25], device=device),
        return_debug_tensors=True,
    )
    assert result.generated_depth is not None
    assert result.logs["loss_rgb_render"] == pytest.approx(
        0.25 * result.logs["loss_rgb_render_unweighted"],
        rel=1.0e-5,
        abs=1.0e-7,
    )
    result.generated_depth.retain_grad()
    result.loss.backward()
    assert result.generated_depth.grad is not None
    assert torch.isfinite(result.generated_depth.grad).all()
    assert float(result.generated_depth.grad.abs().sum()) > 0.0
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA GT-sky integration test")
def test_formal_rgb_renderer_preserves_original_gt_sky_without_sky_model() -> None:
    pytest.importorskip("gsplat")
    device = torch.device("cuda:0")
    seq_len, height, width = 2, 16, 16
    z = torch.zeros((1, seq_len, 1, 4), device=device)
    images = torch.full((1, seq_len, 3, height, width), 0.8, device=device)
    sky_mask = torch.zeros((1, seq_len, 1, height, width), device=device)
    sky_mask[..., : height // 2, :] = 1.0
    pose = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]],
        device=device,
    )
    vggt = _VGGT().to(device)
    vggt.sky_model = _FailIfCalledSkyModel().to(device)

    result = compute_rgb_render_loss(
        vggt_model=vggt,
        scene_flow_root=_SceneFlow().to(device),
        z_clean_pred_n=z,
        images=images,
        timestamps=torch.tensor([[0.0, 1.0]], device=device),
        render_pose_enc_dggt=pose,
        render_sky_probability=sky_mask,
        loss_sky_mask_gt=sky_mask,
        patch_grid=(1, 1),
        patch_start_idx=1,
        max_samples=1,
        max_frames=seq_len,
        render_stride=1,
        background_mode="gt_sky",
        patch_weight_mask=torch.ones((1, seq_len, 1, 1), device=device),
        return_debug_tensors=True,
    )

    assert result.rendered is not None
    sky = sky_mask.expand_as(images).bool()
    torch.testing.assert_close(result.rendered[sky], images[sky], rtol=0.0, atol=0.0)
    assert float((result.rendered[~sky] - images[~sky]).abs().mean()) > 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA parity test")
def test_differentiable_renderer_matches_deployment_renderer():
    pytest.importorskip("gsplat")
    from train_scene_flow_pretrain import _render_gs_map_rgb

    device = torch.device("cuda:0")
    seq_len, height, width = 2, 16, 16
    images = torch.zeros((1, seq_len, 3, height, width), device=device)
    pose = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
          [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]],
        device=device,
    )
    depth = torch.full((1, seq_len, height, width, 1), 2.5, device=device)
    y = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1, 1)
    x = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width, 1)
    rgb = torch.cat(
        (
            (0.2 + 0.5 * x).expand(1, seq_len, height, width, 1),
            (0.1 + 0.6 * y).expand(1, seq_len, height, width, 1),
            torch.full((1, seq_len, height, width, 1), 0.35, device=device),
        ),
        dim=-1,
    )
    opacity = torch.full((1, seq_len, height, width, 1), 0.65, device=device)
    scales = torch.full((1, seq_len, height, width, 3), 0.04, device=device)
    quats = torch.zeros((1, seq_len, height, width, 4), device=device)
    quats[..., 0] = 1.0
    gs_map = torch.cat((rgb, opacity, scales, quats), dim=-1)
    # DGGT's GaussianHead confidence contract is [B,S,H,W].
    gs_conf = torch.ones((1, seq_len, height, width), device=device)
    dynamic_conf = torch.full((1, seq_len, height, width, 1), -1.0, device=device)
    sky_probability = torch.zeros((1, seq_len, 1, height, width), device=device)
    timestamps = torch.tensor([0.0, 1.0], device=device)

    training, _ = _render_one_sample(
        vggt_model=_VGGT().to(device),
        images=images[0],
        timestamps=timestamps,
        pose_enc=pose[0],
        depth=depth[0, ..., 0],
        gs_map=gs_map[0],
        gs_conf=gs_conf[0],
        dynamic_conf=dynamic_conf[0],
        render_sky_probability=sky_probability[0, :, 0],
        max_frames=seq_len,
        stride=1,
        background_mode="black",
        sky_tokens=None,
        sky_grid=(4, 8),
    )
    deployment = _render_gs_map_rgb(
        _VGGT().to(device),
        None,
        sky_probability,
        timestamps,
        pose,
        depth,
        gs_map,
        gs_conf,
        dynamic_conf,
        device,
        seq_len,
        background_mode="black",
        use_sky_mask=True,
        image_hw=(height, width),
        soft_sky_mask=True,
    )
    torch.testing.assert_close(training.detach().cpu(), deployment, rtol=2.0e-4, atol=2.0e-4)


@pytest.mark.parametrize("grad_scale,expect_nonzero", [(0.0, False), (0.05, True)])
def test_rgb_to_mask_gradient_scale_is_finite(grad_scale: float, expect_nonzero: bool) -> None:
    logits = torch.tensor([0.25, -0.5], requires_grad=True)
    probability = scale_gradient(torch.sigmoid(logits), grad_scale)
    probability.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert bool((logits.grad.abs() > 0).any()) is expect_nonzero
