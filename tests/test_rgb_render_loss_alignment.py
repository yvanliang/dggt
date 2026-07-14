from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from dggt.losses.rgb_render_loss import (
    _masked_lpips,
    _render_one_sample,
    compute_rgb_render_loss,
    decode_generated_dggt_geometry,
    rgb_render_loss_ramp,
    scale_gradient,
    should_apply_rgb_render_loss,
    sky_tokens_to_background,
    torch_unproject_depth,
)
from dggt.utils.gaussian_render import composite_gsplat_rgb
from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
import train_scene_flow as formal_train
from train_scene_flow import (
    _cached_render_pose_from_payload,
    _formal_rgb_context,
    _prepare_visualization_batch,
    cached_render_pose_from_item,
)
from train_scene_flow_pretrain import (
    auxiliary_scene_flow_forward,
    should_apply_sky_mask_endpoint_supervision,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_LAUNCH_SCRIPTS = (
    "pretrain_half_node_p6000.sh",
    "pretrain_single_node.sh",
    "pretrain_single_node_test.sh",
    "pretrain_two_nodes.sh",
)


class _Tokenizer(nn.Module):
    def decode(self, z: torch.Tensor, patch_grid):
        del patch_grid
        base = z.mean(dim=-1, keepdim=True)
        return [base.expand(*base.shape[:-1], 3072) for _ in range(4)]


class _DepthHead(nn.Module):
    intermediate_layer_idx = (4, 11, 17, 23)

    def forward(self, tokens, images, patch_start_idx, image_hw=None):
        del images
        h, w = image_hw
        source = tokens[4][:, :, patch_start_idx:, 0].mean(dim=-1, keepdim=True)
        depth = (2.0 + 0.05 * source).view(*source.shape[:2], 1, 1, 1).expand(-1, -1, h, w, 1)
        return depth, torch.ones_like(depth)


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


class _AuxiliaryDDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(2, 2, bias=False)
        self.repa_proj = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor, *, return_mid: bool) -> torch.Tensor:
        outputs = self.trunk(inputs)
        if return_mid:
            outputs = outputs + self.repa_proj(inputs)
        return outputs


def test_primary_rgb_api_has_no_teacher_depth_argument():
    parameters = inspect.signature(compute_rgb_render_loss).parameters
    assert "depth" not in parameters
    assert "render_pose_enc_dggt" in parameters


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


def test_sky_mask_endpoint_supervision_has_independent_delay_and_period():
    args = SimpleNamespace(
        sky_mask_endpoint_start_step=5000,
        sky_mask_endpoint_every=4,
    )
    assert not should_apply_sky_mask_endpoint_supervision(args, None, training=True)
    assert not should_apply_sky_mask_endpoint_supervision(args, 4999, training=True)
    assert should_apply_sky_mask_endpoint_supervision(args, 5000, training=True)
    assert not should_apply_sky_mask_endpoint_supervision(args, 5002, training=True)
    assert should_apply_sky_mask_endpoint_supervision(args, 5004, training=True)
    assert not should_apply_sky_mask_endpoint_supervision(args, 5004, training=False)

    args.sky_mask_endpoint_every = 0
    assert not should_apply_sky_mask_endpoint_supervision(args, 5000, training=True)


def test_pretrain_rgb_defaults_use_every_full_resolution_frame():
    from train_scene_flow_pretrain import build_argparser

    args = build_argparser().parse_args(
        [
            "--image_dir",
            "/tmp/images",
            "--dggt_ckpt_path",
            "/tmp/dggt.pt",
            "--feature_stats_path",
            "/tmp/stats.pt",
            "--log_dir",
            "/tmp/logs",
        ]
    )
    assert args.rgb_render_start_step == 5000
    assert args.rgb_render_every == 2
    assert args.rgb_render_max_frames == 0  # 0 means every frame in the clip.
    assert args.rgb_render_stride == 1


def test_pretrain_launch_scripts_do_not_override_rgb_coverage_defaults():
    forbidden_flags = (
        "--rgb_render_every",
        "--rgb_render_max_frames",
        "--rgb_render_stride",
    )
    for script_name in PRETRAIN_LAUNCH_SCRIPTS:
        script = (REPO_ROOT / script_name).read_text()
        for flag in forbidden_flags:
            assert flag not in script, f"{script_name} must inherit the pretrain default for {flag}"


def test_auxiliary_scene_flow_forward_does_not_prepare_ddp_reducer_twice():
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if dist.is_initialized():
        pytest.skip("test requires ownership of a temporary process group")

    with tempfile.NamedTemporaryFile() as rendezvous:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{os.path.abspath(rendezvous.name)}",
            rank=0,
            world_size=1,
        )
        try:
            model = DistributedDataParallel(
                _AuxiliaryDDPModel(),
                find_unused_parameters=True,
            )
            inputs = torch.randn(3, 2)
            primary = model(inputs, return_mid=True)
            endpoint = auxiliary_scene_flow_forward(model, inputs, return_mid=False)
            (primary + endpoint).square().mean().backward()
            assert model.module.trunk.weight.grad is not None
            assert model.module.repa_proj.weight.grad is not None
        finally:
            dist.destroy_process_group()


def test_directional_sky_projection_matches_validation_path():
    from train_scene_flow_pretrain import render_sky_tokens_directional_background

    torch.manual_seed(7)
    seq_len, height, width = 2, 11, 17
    sky_tokens = torch.rand((1, 32, 8)) * 2.0 - 1.0
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
        return_debug_tensors=True,
    )
    assert result.generated_depth is not None
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
