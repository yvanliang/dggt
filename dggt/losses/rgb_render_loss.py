"""Deployment-aligned differentiable RGB supervision for SceneFlow.

The renderer deliberately decodes depth and Gaussian attributes from the
SceneFlow-predicted video tokens.  A cached/frozen DGGT depth map may be used
for diagnostics or a separate preserve-region distillation loss, but it must
never be passed into the primary RGB render path.

Camera policy is owned by the caller. Metric-camera pretraining deliberately
renders with the detached frozen-teacher DGGT trajectory, because the decoded
latents live in that teacher space. Formal editing likewise passes its frozen
DGGT camera.

Frozen DGGT/tokenizer parameters keep ``requires_grad=False`` but this module
must not run their decode/head calls under ``torch.no_grad``: gradients are
required with respect to the generated SceneFlow tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import torch
import torch.nn.functional as F

from dggt.losses.reconstruction_feedback_loss import (
    TOKENIZER_LEVELS,
    compute_reconstruction_feedback_losses,
)
from dggt.utils.gaussian_render import composite_gsplat_rgb, composite_original_sky
from dggt.utils.gs import get_split_gs
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.scene_gauge import (
    PULLBACK_RENDER_BOUNDARY,
    PullbackCalibration,
    apply_pullback_calibration,
)
from dggt.utils.sliding_window import OFFLINE_MAX_SINGLE_WINDOW
from dggt.utils.tokenizer_window import decode_tokenizer_windowed


@dataclass
class DecodedGeneratedGeometry:
    image_tokens: list[torch.Tensor | None]
    aggregated_tokens: list[torch.Tensor | None]
    dino_tokens: list[torch.Tensor | None]
    depth: torch.Tensor
    depth_conf: torch.Tensor
    gs_map: torch.Tensor
    gs_conf: torch.Tensor
    dynamic_conf: torch.Tensor


@dataclass
class RGBRenderLossResult:
    loss: torch.Tensor
    level_loss: torch.Tensor
    head_loss: torch.Tensor
    logs: dict[str, float]
    rendered: torch.Tensor | None = None
    generated_depth: torch.Tensor | None = None


def rgb_render_loss_enabled(args: Any) -> bool:
    return float(getattr(args, "lambda_rgb_render", 0.0)) > 0.0 and int(
        getattr(args, "rgb_render_every", 2)
    ) > 0


def should_apply_rgb_render_loss(args: Any, step: int | None, *, training: bool) -> bool:
    if not bool(training) or not rgb_render_loss_enabled(args):
        return False
    every = max(1, int(getattr(args, "rgb_render_every", 2)))
    if step is not None and int(step) < max(0, int(getattr(args, "rgb_render_start_step", 0))):
        return False
    return step is None or int(step) % every == 0


def rgb_render_loss_ramp(args: Any, step: int | None) -> float:
    """Delayed linear ramp; unlike the legacy ramp this can stay exactly zero."""
    if step is None:
        return 1.0
    start = max(0, int(getattr(args, "rgb_render_start_step", 0)))
    if int(step) < start:
        return 0.0
    warmup = max(0, int(getattr(args, "rgb_render_warmup_steps", 0)))
    if warmup == 0:
        return 1.0
    return max(0.0, min(1.0, float(int(step) - start) / float(warmup)))


def rgb_render_sigma_weight(sigmas: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    """Return per-sample clean-confidence weights ``(1 - sigma) ** power``.

    SceneFlow uses the noise-progress convention where sigma=0 is clean and
    sigma=1 is pure noise.  Keeping this weight outside the renderer's
    pixel-mask normalization makes it attenuate the whole reconstruction
    gradient instead of cancelling out in the normalized image loss.
    ``power=0`` explicitly disables sigma weighting.
    """
    exponent = float(power)
    if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError(f"RGB render sigma power must be finite and non-negative, got {power}")
    if not torch.is_tensor(sigmas):
        raise TypeError("sigmas must be a torch.Tensor")
    if sigmas.ndim != 1:
        raise ValueError(f"sigmas must be a per-sample vector [B], got {tuple(sigmas.shape)}")
    sigma = sigmas.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    if exponent == 0.0:
        return torch.ones_like(sigma)
    return (1.0 - sigma).pow(exponent)


def setup_lpips_for_rgb_loss(args: Any, device: torch.device):
    if not rgb_render_loss_enabled(args):
        return None
    if float(getattr(args, "rgb_render_lpips_weight", 0.0)) <= 0.0:
        return None
    try:
        import lpips  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "--rgb_render_lpips_weight > 0 requires lpips in the dggt environment."
        ) from exc
    # spatial=True is required so sky/edit masks are applied to the perceptual
    # loss itself rather than only to the pixel reconstruction term.
    model = lpips.LPIPS(
        net=str(getattr(args, "rgb_render_lpips_net", "alex")),
        spatial=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the forward value unchanged while scaling only its input gradient."""
    factor = float(scale)
    if factor < 0.0:
        raise ValueError(f"gradient scale must be non-negative, got {factor}")
    detached = value.detach()
    return detached + factor * (value - detached)


def _decode_sparse_image_tokens(
    *,
    vggt_model: torch.nn.Module,
    scene_flow_root: torch.nn.Module,
    z_clean_pred_n: torch.Tensor,
    patch_grid: tuple[int, int] | list[int],
    patch_start_idx: int,
    tokenizer_window_len: int,
) -> list[torch.Tensor | None]:
    z = scene_flow_root.denormalize(z_clean_pred_n.float())
    decoded = decode_tokenizer_windowed(
        vggt_model.scene_tokenizer,
        z,
        patch_grid=patch_grid,
        window_len=int(tokenizer_window_len),
    )
    if len(decoded) != len(TOKENIZER_LEVELS):
        raise RuntimeError(
            f"scene_tokenizer.decode returned {len(decoded)} levels; expected {len(TOKENIZER_LEVELS)}"
        )
    full: list[torch.Tensor | None] = [None] * (max(TOKENIZER_LEVELS) + 1)
    for level, patch_tokens in zip(TOKENIZER_LEVELS, decoded):
        special = patch_tokens.new_zeros(
            patch_tokens.shape[:2]
            + (int(patch_start_idx), int(patch_tokens.shape[-1]))
        )
        full[int(level)] = torch.cat((special, patch_tokens), dim=-2)
    return full


def _split_sparse_tokens_for_heads(
    image_tokens: list[torch.Tensor | None],
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    aggregated: list[torch.Tensor | None] = [None] * len(image_tokens)
    dino: list[torch.Tensor | None] = [None] * len(image_tokens)
    for level in TOKENIZER_LEVELS:
        tokens = image_tokens[int(level)]
        if tokens is None:
            raise RuntimeError(f"generated token level {level} is missing")
        if int(tokens.shape[-1]) != 3072:
            raise ValueError(
                f"generated DGGT level {level} must be 3072-wide, got {tokens.shape[-1]}"
            )
        dino_tokens, frame_tokens, global_tokens = tokens.split((1024, 1024, 1024), dim=-1)
        dino[int(level)] = dino_tokens
        aggregated[int(level)] = torch.cat((frame_tokens, global_tokens), dim=-1)
    return aggregated, dino


def decode_generated_dggt_geometry(
    *,
    vggt_model: torch.nn.Module,
    scene_flow_root: torch.nn.Module,
    z_clean_pred_n: torch.Tensor,
    patch_grid: tuple[int, int] | list[int],
    patch_start_idx: int,
    image_hw: tuple[int, int],
    pullback_calibration: PullbackCalibration | None = None,
    tokenizer_window_len: int | None = None,
) -> DecodedGeneratedGeometry:
    """Decode exactly the generated-token geometry used by validation/offline."""
    resolved_window_len = (
        int(pullback_calibration.window_len)
        if pullback_calibration is not None
        else int(tokenizer_window_len or OFFLINE_MAX_SINGLE_WINDOW)
    )
    image_tokens = _decode_sparse_image_tokens(
        vggt_model=vggt_model,
        scene_flow_root=scene_flow_root,
        z_clean_pred_n=z_clean_pred_n,
        patch_grid=patch_grid,
        patch_start_idx=int(patch_start_idx),
        tokenizer_window_len=resolved_window_len,
    )
    aggregated, dino = _split_sparse_tokens_for_heads(image_tokens)
    height, width = int(image_hw[0]), int(image_hw[1])
    device_type = z_clean_pred_n.device.type
    # DGGT heads are numerically defined in fp32.  Their parameters are frozen,
    # but autograd remains enabled for the generated input tokens.
    with torch.amp.autocast(device_type=device_type, enabled=False):
        depth, depth_conf = vggt_model.depth_head(
            aggregated,
            None,
            int(patch_start_idx),
            image_hw=(height, width),
        )
        gs_map, gs_conf = vggt_model.gs_head(
            image_tokens,
            None,
            int(patch_start_idx),
            image_hw=(height, width),
        )
        dynamic_conf, _ = vggt_model.instance_head(
            dino,
            None,
            int(patch_start_idx),
            image_hw=(height, width),
        )
    depth = depth.float()
    gs_map = gs_map.float()
    if pullback_calibration is not None:
        # Rendering is defined in the tokenizer's native reconstructed DGGT
        # geometry. Calling the shared helper here makes that scope explicit
        # and prevents the v1 metric-only correction from leaking into render.
        pullback = apply_pullback_calibration(
            depth,
            gs_map,
            log_metric_scale=0.0,
            calibration=pullback_calibration,
            boundary=PULLBACK_RENDER_BOUNDARY,
        )
        if pullback.depth_dggt is not depth or pullback.gs_map_dggt is not gs_map:
            raise AssertionError("render pullback must be an exact identity")
        depth = pullback.depth_dggt
        gs_map = pullback.gs_map_dggt
    return DecodedGeneratedGeometry(
        image_tokens=image_tokens,
        aggregated_tokens=aggregated,
        dino_tokens=dino,
        depth=depth,
        depth_conf=depth_conf.float(),
        gs_map=gs_map,
        gs_conf=gs_conf.float(),
        dynamic_conf=dynamic_conf.float(),
    )


def _depth_to_bshw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 5 and int(depth.shape[-1]) == 1:
        return depth[..., 0]
    if depth.ndim == 5 and int(depth.shape[2]) == 1:
        return depth[:, :, 0]
    if depth.ndim == 4:
        return depth
    raise ValueError(f"depth must be [B,S,H,W,1], [B,S,1,H,W], or [B,S,H,W], got {depth.shape}")


def _mask_to_bshw(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    b, s, _, h, w = reference.shape
    if mask is None:
        return torch.zeros((b, s, h, w), device=reference.device, dtype=torch.float32)
    value = mask.to(device=reference.device, dtype=torch.float32)
    if value.ndim == 5 and int(value.shape[2]) in (1, 3):
        value = value.mean(dim=2)
    elif value.ndim == 5 and int(value.shape[-1]) in (1, 3):
        value = value.mean(dim=-1)
    elif value.ndim != 4:
        raise ValueError(f"sky mask has unsupported shape {tuple(value.shape)}")
    if tuple(value.shape[:2]) != (b, s):
        raise ValueError(
            f"sky mask batch/sequence {tuple(value.shape[:2])} != reference {(b, s)}"
        )
    if tuple(value.shape[-2:]) != (h, w):
        value = F.interpolate(
            value.reshape(b * s, 1, *value.shape[-2:]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, s, h, w)
    return value.clamp(0.0, 1.0)


def _pose_to_mats(
    pose_enc: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsics3, intrinsics = pose_encoding_to_extri_intri(pose_enc.float(), image_hw)
    bottom = extrinsics3.new_tensor((0.0, 0.0, 0.0, 1.0)).view(1, 1, 1, 4)
    extrinsics4 = torch.cat(
        (extrinsics3, bottom.expand(extrinsics3.shape[0], extrinsics3.shape[1], -1, -1)),
        dim=-2,
    )
    return extrinsics4, intrinsics


def _scale_intrinsics(intrinsics: torch.Tensor, stride: int) -> torch.Tensor:
    if int(stride) <= 1:
        return intrinsics
    out = intrinsics.clone()
    out[..., 0, 0] /= float(stride)
    out[..., 1, 1] /= float(stride)
    out[..., 0, 2] /= float(stride)
    out[..., 1, 2] /= float(stride)
    return out


def torch_unproject_depth(
    depth: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Differentiable OpenCV-depth unprojection to DGGT world coordinates."""
    if depth.ndim != 3:
        raise ValueError(f"depth must be [S,H,W], got {tuple(depth.shape)}")
    s, h, w = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0].clamp_min(1.0e-6).view(s, 1, 1)
    fy = intrinsics[:, 1, 1].clamp_min(1.0e-6).view(s, 1, 1)
    cx = intrinsics[:, 0, 2].view(s, 1, 1)
    cy = intrinsics[:, 1, 2].view(s, 1, 1)
    z = depth.float()
    x = (xx.view(1, h, w) - cx) * z / fx
    y = (yy.view(1, h, w) - cy) * z / fy
    camera_h = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
    rotation = world_to_camera[:, :3, :3]
    translation = world_to_camera[:, :3, 3]
    # Rigid inverse avoids a generic matrix inverse and keeps gradients stable.
    world = torch.einsum(
        "sij,shwj->shwi",
        rotation.transpose(-1, -2),
        camera_h[..., :3] - translation[:, None, None, :],
    )
    return world


def _sky_grid_shape(num_tokens: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    if int(grid_h) * int(grid_w) == int(num_tokens):
        return int(grid_h), int(grid_w)
    h = max(1, int(math.sqrt(int(num_tokens))))
    while h > 1 and int(num_tokens) % h:
        h -= 1
    return h, int(num_tokens) // h


def sky_tokens_to_background(
    sky_tokens: torch.Tensor,
    *,
    seq_len: int,
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Differentiably project the generated directional sky atlas."""
    tokens = sky_tokens[:1].float()
    gh, gw = _sky_grid_shape(int(tokens.shape[1]), int(grid_h), int(grid_w))
    rgb = ((tokens[..., :3] + 1.0) * 0.5).clamp(0.0, 1.0)
    atlas = rgb.reshape(1, gh, gw, 3).permute(0, 3, 1, 2).contiguous()
    atlas = torch.cat((atlas[..., -1:], atlas, atlas[..., :1]), dim=-1)

    w2c = world_to_camera[0]
    k = intrinsics[0]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=tokens.device, dtype=torch.float32),
        torch.arange(width, device=tokens.device, dtype=torch.float32),
        indexing="ij",
    )
    fx = k[:, 0, 0].clamp_min(1.0e-6).view(seq_len, 1, 1)
    fy = k[:, 1, 1].clamp_min(1.0e-6).view(seq_len, 1, 1)
    cx = k[:, 0, 2].view(seq_len, 1, 1)
    cy = k[:, 1, 2].view(seq_len, 1, 1)
    dirs_camera = torch.stack(
        (
            (xx.view(1, height, width) - cx) / fx,
            (yy.view(1, height, width) - cy) / fy,
            torch.ones((seq_len, height, width), device=tokens.device),
        ),
        dim=-1,
    )
    dirs_camera = F.normalize(dirs_camera, dim=-1)
    dirs_world = torch.einsum(
        "sij,shwj->shwi", w2c[:, :3, :3].transpose(-1, -2), dirs_camera
    )
    dirs_world = F.normalize(dirs_world, dim=-1)
    pi = float(math.pi)
    azimuth = torch.atan2(dirs_world[..., 2], dirs_world[..., 0])
    u = torch.remainder((azimuth + pi) / (2.0 * pi), 1.0)
    elevation = torch.asin((-dirs_world[..., 1]).clamp(0.0, 1.0))
    v = (1.0 - elevation / (0.5 * pi)).clamp(0.0, 1.0)
    # Match validation/offline exactly: the atlas has one wrapped column on
    # either side and values live at pixel centres under align_corners=False.
    # Mixing this with align_corners=True shifts every azimuth sample slightly
    # and makes the RGB training target disagree with deployment at the seam.
    x_norm = (2.0 * (u * float(gw) + 1.0) / float(gw + 2)) - 1.0
    grid = torch.stack((x_norm, 2.0 * v - 1.0), dim=-1)
    background = F.grid_sample(
        atlas.expand(seq_len, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return background.permute(0, 2, 3, 1).contiguous()


def _render_one_sample(
    *,
    vggt_model: torch.nn.Module,
    images: torch.Tensor,
    timestamps: torch.Tensor,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
    render_sky_probability: torch.Tensor,
    max_frames: int,
    stride: int,
    background_mode: str,
    sky_tokens: torch.Tensor | None,
    sky_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from gsplat.rendering import rasterization

    seq_len, _, height, width = images.shape
    stride = max(1, int(stride))
    frames = seq_len if int(max_frames) <= 0 else min(int(max_frames), seq_len)
    render_h = int(math.ceil(float(height) / float(stride)))
    render_w = int(math.ceil(float(width) / float(stride)))
    world_to_camera_b, intrinsics_b = _pose_to_mats(pose_enc.unsqueeze(0), (height, width))
    world_to_camera = world_to_camera_b[0]
    intrinsics = _scale_intrinsics(intrinsics_b[0], stride)

    depth_low = depth[:, ::stride, ::stride].contiguous()
    point_map = torch_unproject_depth(depth_low, world_to_camera, intrinsics).unsqueeze(0)
    gs_low = gs_map[:, ::stride, ::stride, :].contiguous().unsqueeze(0)
    conf_low = gs_conf[:, ::stride, ::stride]
    if conf_low.ndim == 3:
        conf_low = conf_low.unsqueeze(0).unsqueeze(-1)
    elif conf_low.ndim == 4:
        conf_low = conf_low.unsqueeze(0)
    else:
        raise ValueError(f"gs_conf must be [S,H,W] or [S,H,W,1], got {gs_conf.shape}")
    dyn_low = dynamic_conf[:, ::stride, ::stride, :].contiguous().unsqueeze(0)
    sky_low = render_sky_probability[:, ::stride, ::stride].contiguous()
    if tuple(sky_low.shape[-2:]) != (render_h, render_w):
        sky_low = F.interpolate(
            sky_low.unsqueeze(1), size=(render_h, render_w), mode="bilinear", align_corners=False
        ).squeeze(1)
    non_sky_probability = (1.0 - sky_low.clamp(0.0, 1.0)).unsqueeze(0)

    gt_rgb_low = None
    gt_sky_low = None
    if background_mode == "gt_sky":
        # Formal editing keeps the original sky in image space.  The
        # rasterizer itself stays on a black background so original non-sky
        # pixels cannot leak through gaps in the edited Gaussians.
        gt_rgb_low = F.interpolate(
            images.float(), size=(render_h, render_w), mode="area"
        ).permute(0, 2, 3, 1).contiguous()
        gt_sky_low = F.interpolate(
            render_sky_probability.unsqueeze(1).float(),
            size=(render_h, render_w),
            mode="area",
        ).squeeze(1).clamp(0.0, 1.0)

    if background_mode == "sky_tokens" and sky_tokens is not None:
        background = sky_tokens_to_background(
            sky_tokens,
            seq_len=seq_len,
            height=render_h,
            width=render_w,
            grid_h=int(sky_grid[0]),
            grid_w=int(sky_grid[1]),
            world_to_camera=world_to_camera.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
        )
    elif background_mode == "sky_model":
        # Legacy/frozen sky-model background retained for non-formal callers.
        with torch.no_grad():
            full_nhwc = vggt_model.sky_model(
                images.unsqueeze(0), world_to_camera, intrinsics_b[0]
            ).float()
            minimum = full_nhwc.amin()
            full_nhwc = (full_nhwc - minimum) / (full_nhwc.amax() - minimum).clamp_min(1.0e-8)
            full = full_nhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2)
            background = F.interpolate(full, (render_h, render_w), mode="area").permute(0, 2, 3, 1)
    else:
        background = images.new_zeros((seq_len, render_h, render_w, 3), dtype=torch.float32)

    timestamps = timestamps[:seq_len].to(device=images.device, dtype=torch.float32)
    dynamic_probability = dyn_low.squeeze(-1).float().sigmoid()
    # Keep a tiny hard support threshold for memory, but multiply opacity by
    # the soft probability so useful gradients reach the sky-mask head.
    support = non_sky_probability > 1.0e-4
    # Match the deployed mode-2 assembly: static membership uses the raw
    # dynamic logit threshold; dynamic membership keeps all non-sky points.
    static_support = support & (dyn_low.squeeze(-1).float() < 0.5)
    static_points = point_map[static_support].reshape(-1, 3)
    static_rgb, static_opacity, static_scale, static_rotation = get_split_gs(gs_low, static_support)
    static_opacity = static_opacity * (1.0 - dynamic_probability[static_support])
    static_opacity = static_opacity * non_sky_probability[static_support]
    static_conf = conf_low[static_support].reshape(-1)
    static_frame = torch.nonzero(static_support, as_tuple=False)[:, 1]
    static_time = timestamps[static_frame] if static_frame.numel() else timestamps.new_zeros((0,))

    rendered_frames: list[torch.Tensor] = []
    alpha_frames: list[torch.Tensor] = []
    log_point_one_tenth = math.log(0.1)
    for frame_index in range(frames):
        dynamic_support = support[:, frame_index]
        dynamic_points = point_map[:, frame_index][dynamic_support].reshape(-1, 3)
        dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(
            gs_low[:, frame_index], dynamic_support
        )
        dynamic_opacity = dynamic_opacity * dynamic_probability[:, frame_index][dynamic_support]
        dynamic_opacity = dynamic_opacity * non_sky_probability[:, frame_index][dynamic_support]
        if static_opacity.numel():
            decay = torch.exp(
                log_point_one_tenth
                * (static_time - timestamps[frame_index]).square()
                / (static_conf.square() + 1.0e-6)
            )
            static_opacity_t = static_opacity * decay
        else:
            static_opacity_t = static_opacity
        means = torch.cat((static_points, dynamic_points), dim=0)
        colors = torch.cat((static_rgb, dynamic_rgb), dim=0)
        opacities = torch.cat((static_opacity_t, dynamic_opacity), dim=0)
        scales = torch.cat((static_scale, dynamic_scale), dim=0)
        rotations = torch.cat((static_rotation, dynamic_rotation), dim=0)
        if means.numel() == 0:
            premultiplied = background.new_zeros((1, render_h, render_w, 3))
            alpha = background.new_zeros((1, render_h, render_w, 1))
        else:
            premultiplied, alpha, _ = rasterization(
                means=means.float(),
                quats=rotations.float(),
                scales=scales.float().clamp_min(1.0e-5),
                opacities=opacities.float().view(-1),
                colors=colors.float().clamp(0.0, 1.0),
                viewmats=world_to_camera[frame_index : frame_index + 1],
                Ks=intrinsics[frame_index : frame_index + 1],
                width=render_w,
                height=render_h,
                render_mode="RGB",
            )
        # gsplat RGB is already premultiplied; multiplying it by alpha again is wrong.
        composed = composite_gsplat_rgb(
            premultiplied[..., :3], alpha, background[frame_index : frame_index + 1]
        )
        if gt_rgb_low is not None and gt_sky_low is not None:
            composed = composite_original_sky(
                composed,
                gt_rgb_low[frame_index : frame_index + 1],
                gt_sky_low[frame_index : frame_index + 1].unsqueeze(-1),
            )
        rendered_frames.append(composed[0].permute(2, 0, 1))
        alpha_frames.append(alpha[0].permute(2, 0, 1))
    return torch.stack(rendered_frames), torch.stack(alpha_frames)


def _masked_lpips(
    lpips_model: torch.nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # Give both images the same neutral value outside the supervised support.
    prediction_masked = weight * prediction + (1.0 - weight) * 0.5
    target_masked = weight * target + (1.0 - weight) * 0.5
    value = lpips_model(prediction_masked * 2.0 - 1.0, target_masked * 2.0 - 1.0)
    if sample_weight is None:
        sample_scale = torch.ones(
            (int(prediction.shape[0]),),
            device=prediction.device,
            dtype=torch.float32,
        )
    else:
        if sample_weight.ndim != 1 or int(sample_weight.shape[0]) != int(prediction.shape[0]):
            raise ValueError(
                "LPIPS sample_weight must be [N] aligned with prediction, got "
                f"{tuple(sample_weight.shape)} for N={int(prediction.shape[0])}"
            )
        sample_scale = sample_weight.to(device=prediction.device, dtype=torch.float32)
    if value.ndim >= 4 and tuple(value.shape[-2:]) != (1, 1):
        resized_weight = F.interpolate(weight, size=value.shape[-2:], mode="area")
        return (
            value * resized_weight * sample_scale.view(-1, 1, 1, 1)
        ).sum() / resized_weight.sum().clamp_min(1.0e-6)
    # Compatibility fallback for externally supplied non-spatial LPIPS models.
    valid = weight.mean(dim=(-3, -2, -1)).gt(0).to(value.dtype)
    per_image = value.reshape(value.shape[0], -1).mean(dim=-1)
    return (
        per_image * valid * sample_scale.to(device=value.device, dtype=value.dtype)
    ).sum() / valid.sum().clamp_min(1.0)


def compute_rgb_render_loss(
    *,
    vggt_model: torch.nn.Module,
    scene_flow_root: torch.nn.Module,
    z_clean_pred_n: torch.Tensor,
    z_clean_target_n: torch.Tensor | None = None,
    images: torch.Tensor,
    timestamps: torch.Tensor,
    render_pose_enc_dggt: torch.Tensor,
    render_sky_probability: torch.Tensor | None,
    loss_sky_mask_gt: torch.Tensor | None,
    patch_grid: tuple[int, int] | list[int],
    patch_start_idx: int,
    max_samples: int,
    max_frames: int,
    render_stride: int,
    background_mode: str,
    sky_tokens: torch.Tensor | None = None,
    sky_grid: tuple[int, int] = (16, 32),
    patch_weight_mask: torch.Tensor | None = None,
    sky_weight: float = 0.0,
    camera_grad_scale: float = 0.0,
    gauge_pose_grad_scale: float = 0.0,
    sky_mask_grad_scale: float = 0.0,
    lpips_model: torch.nn.Module | None = None,
    lpips_weight: float = 0.0,
    loss_sample_weight: torch.Tensor | None = None,
    conf_weight_power: float = 1.0,
    conf_weight_floor: float = 0.05,
    return_debug_tensors: bool = False,
    return_generated_depth: bool = False,
    pullback_calibration: PullbackCalibration | None = None,
) -> RGBRenderLossResult:
    conf_weight_power = float(conf_weight_power)
    if conf_weight_power != 0.0:
        conf_weight_floor = float(conf_weight_floor)
        if not math.isfinite(conf_weight_power) or conf_weight_power < 0.0:
            raise ValueError("conf_weight_power must be finite and non-negative")
        if not math.isfinite(conf_weight_floor) or not 0.0 < conf_weight_floor <= 1.0:
            raise ValueError("conf_weight_floor must be finite and in (0, 1]")
    if float(camera_grad_scale) != 0.0:
        raise ValueError(
            "camera_grad_scale must be 0: metric-gauge pretraining renders with "
            "detached teacher-space camera poses, so a non-zero value would be a "
            "silent no-op."
        )
    if float(gauge_pose_grad_scale) not in (0.0, 1.0):
        raise ValueError("gauge_pose_grad_scale must be exactly 0 or 1")
    if images.ndim != 5 or int(images.shape[2]) != 3:
        raise ValueError(f"images must be [B,S,3,H,W], got {tuple(images.shape)}")
    if str(background_mode) == "gt_sky" and render_sky_probability is None:
        raise ValueError("background_mode='gt_sky' requires a GT sky mask")
    if render_pose_enc_dggt.ndim != 3 or int(render_pose_enc_dggt.shape[-1]) != 9:
        raise ValueError(
            f"render_pose_enc_dggt must be DGGT [B,S,9], got {render_pose_enc_dggt.shape}"
        )
    available = min(int(images.shape[0]), int(z_clean_pred_n.shape[0]))
    batch_size = available if int(max_samples) <= 0 else min(int(max_samples), available)
    if batch_size <= 0:
        zero = z_clean_pred_n.sum() * 0.0
        return RGBRenderLossResult(
            loss=zero,
            level_loss=zero,
            head_loss=zero,
            logs={
                "loss_rgb_render": 0.0,
                "loss_level_consistency": 0.0,
                "loss_head_consistency": 0.0,
            },
        )
    if loss_sample_weight is None:
        sample_weight = torch.ones(
            (batch_size,),
            device=z_clean_pred_n.device,
            dtype=torch.float32,
        )
    else:
        if loss_sample_weight.ndim != 1 or int(loss_sample_weight.shape[0]) < batch_size:
            raise ValueError(
                "loss_sample_weight must be [B] with at least the rendered batch size, got "
                f"{tuple(loss_sample_weight.shape)} for B={batch_size}"
            )
        sample_weight = loss_sample_weight[:batch_size].to(
            device=z_clean_pred_n.device,
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0.0).any()):
            raise ValueError("loss_sample_weight must contain finite non-negative values")
    images = images[:batch_size].to(device=z_clean_pred_n.device, dtype=torch.float32)
    pose = scale_gradient(
        render_pose_enc_dggt[:batch_size].to(z_clean_pred_n.device, torch.float32),
        float(gauge_pose_grad_scale),
    )
    render_sky = _mask_to_bshw(render_sky_probability, images)
    render_sky = scale_gradient(render_sky, float(sky_mask_grad_scale))
    loss_sky = _mask_to_bshw(loss_sky_mask_gt, images)
    timestamps = timestamps.to(device=z_clean_pred_n.device, dtype=torch.float32)
    if timestamps.ndim == 1:
        timestamps = timestamps.view(1, -1).expand(batch_size, -1)
    else:
        timestamps = timestamps[:batch_size]
    height, width = int(images.shape[-2]), int(images.shape[-1])
    geometry = decode_generated_dggt_geometry(
        vggt_model=vggt_model,
        scene_flow_root=scene_flow_root,
        z_clean_pred_n=z_clean_pred_n[:batch_size],
        patch_grid=patch_grid,
        patch_start_idx=int(patch_start_idx),
        image_hw=(height, width),
        pullback_calibration=pullback_calibration,
    )
    depth = _depth_to_bshw(geometry.depth)
    if not bool(torch.isfinite(depth).all()):
        raise FloatingPointError("generated DGGT depth contains non-finite values")
    feedback = None
    teacher_depth_conf = None
    if z_clean_target_n is not None:
        if int(z_clean_target_n.shape[0]) < batch_size:
            raise ValueError(
                f"z_clean_target_n has B={z_clean_target_n.shape[0]}, expected at least {batch_size}"
            )
        # Teacher decoding is the self-consistent frozen target D(z_clean).
        # It must not build a graph; the student decode above remains fully
        # differentiable with respect to z_clean_pred_n.
        with torch.no_grad():
            teacher_geometry = decode_generated_dggt_geometry(
                vggt_model=vggt_model,
                scene_flow_root=scene_flow_root,
                z_clean_pred_n=z_clean_target_n[:batch_size].detach(),
                patch_grid=patch_grid,
                patch_start_idx=int(patch_start_idx),
                image_hw=(height, width),
                pullback_calibration=pullback_calibration,
            )
        teacher_depth_conf = _depth_to_bshw(teacher_geometry.depth_conf).detach().float()
        feedback = compute_reconstruction_feedback_losses(
            student_geometry=geometry,
            teacher_geometry=teacher_geometry,
            patch_grid=patch_grid,
            patch_weight_mask=patch_weight_mask,
            loss_sky_mask_gt=loss_sky_mask_gt,
            sky_weight=float(sky_weight),
            max_frames=int(max_frames),
            render_stride=int(render_stride),
            sample_weight=sample_weight,
            conf_weight_power=conf_weight_power,
            conf_weight_floor=conf_weight_floor,
        )
        del teacher_geometry

    renders: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    frames = int(images.shape[1]) if int(max_frames) <= 0 else min(int(max_frames), int(images.shape[1]))
    for row in range(batch_size):
        rendered, alpha = _render_one_sample(
            vggt_model=vggt_model,
            images=images[row],
            timestamps=timestamps[row],
            pose_enc=pose[row],
            depth=depth[row],
            gs_map=geometry.gs_map[row],
            gs_conf=geometry.gs_conf[row],
            dynamic_conf=geometry.dynamic_conf[row],
            render_sky_probability=render_sky[row],
            max_frames=frames,
            stride=int(render_stride),
            background_mode=str(background_mode),
            sky_tokens=None if sky_tokens is None else sky_tokens[row : row + 1],
            sky_grid=sky_grid,
        )
        renders.append(rendered)
        alphas.append(alpha)
    rendered_b = torch.stack(renders)
    alpha_b = torch.stack(alphas)
    target = F.interpolate(
        images[:, :frames].reshape(batch_size * frames, 3, height, width),
        size=rendered_b.shape[-2:],
        mode="area",
    ).reshape(batch_size, frames, 3, *rendered_b.shape[-2:])
    loss_sky_low = F.interpolate(
        loss_sky[:, :frames].reshape(batch_size * frames, 1, height, width),
        size=rendered_b.shape[-2:],
        mode="area",
    ).reshape(batch_size, frames, 1, *rendered_b.shape[-2:])
    weight = (1.0 - loss_sky_low) + float(sky_weight) * loss_sky_low
    if patch_weight_mask is not None:
        patch = patch_weight_mask[:batch_size, :frames].to(rendered_b.device, torch.float32)
        if patch.ndim != 4 or int(patch.shape[-1]) != 1:
            raise ValueError(f"patch_weight_mask must be [B,S,P,1], got {patch.shape}")
        gh, gw = int(patch_grid[0]), int(patch_grid[1])
        patch_image = F.interpolate(
            patch.reshape(batch_size * frames, gh, gw, 1).permute(0, 3, 1, 2),
            size=rendered_b.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, frames, 1, *rendered_b.shape[-2:]).clamp(0.0, 1.0)
        weight = weight * patch_image
    photometric_weight = weight
    conf_factor_mean = None
    if conf_weight_power != 0.0 and teacher_depth_conf is not None:
        conf_low = F.interpolate(
            teacher_depth_conf[:, :frames].reshape(
                batch_size * frames,
                1,
                *teacher_depth_conf.shape[-2:],
            ),
            size=rendered_b.shape[-2:],
            mode="area",
        ).reshape(batch_size, frames, 1, *rendered_b.shape[-2:])
        conf_factor = (
            conf_low.detach()
            .float()
            .clamp(conf_weight_floor, 1.0)
            .pow(conf_weight_power)
        )
        conf_factor_mean = conf_factor.mean()
        conf_factor = conf_factor / conf_factor.mean(
            dim=(1, 2, 3, 4), keepdim=True
        ).clamp_min(1.0e-6)
        photometric_weight = weight * conf_factor.detach()
    difference = rendered_b.float() - target.float()
    denominator = photometric_weight.sum().clamp_min(1.0e-6) * 3.0
    sample_scale = sample_weight.view(batch_size, 1, 1, 1, 1)
    charbonnier_map = torch.sqrt(difference.square() + 1.0e-6)
    charbonnier_unweighted = (
        charbonnier_map * photometric_weight
    ).sum() / denominator
    charbonnier = (
        charbonnier_map * photometric_weight * sample_scale
    ).sum() / denominator
    l1_unweighted = (difference.abs() * photometric_weight).sum() / denominator
    l1 = (
        difference.abs() * photometric_weight * sample_scale
    ).sum() / denominator
    loss = charbonnier
    logs = {
        "loss_rgb_render": float(charbonnier.detach().item()),
        "loss_rgb_render_unweighted": float(charbonnier_unweighted.detach().item()),
        "loss_rgb_render_l1": float(l1.detach().item()),
        "loss_rgb_render_l1_unweighted": float(l1_unweighted.detach().item()),
        "rgb_render_sample_weight_mean": float(sample_weight.detach().mean().item()),
        "rgb_render_weight_mean": float(weight.detach().mean().item()),
        "rgb_render_alpha_mean": float(alpha_b.detach().mean().item()),
        "rgb_render_depth_mean": float(depth.detach().mean().item()),
        "rgb_render_camera_grad_scale": float(camera_grad_scale),
        "rgb_render_gauge_pose_grad_scale": float(gauge_pose_grad_scale),
        "rgb_render_sky_mask_grad_scale": float(sky_mask_grad_scale),
        "rgb_render_frames": float(frames),
        "rgb_render_samples": float(batch_size),
    }
    if conf_factor_mean is not None:
        no_conf_denominator = weight.sum().clamp_min(1.0e-6) * 3.0
        charbonnier_no_conf = (
            charbonnier_map * weight * sample_scale
        ).sum() / no_conf_denominator
        if not bool(torch.isfinite(charbonnier_no_conf)):
            raise FloatingPointError("RGB render loss without confidence is non-finite")
        logs.update(
            {
                "loss_rgb_render_no_conf": float(
                    charbonnier_no_conf.detach().item()
                ),
                "rgb_render_conf_weight_mean": float(
                    conf_factor_mean.detach().item()
                ),
            }
        )
    zero = z_clean_pred_n.sum() * 0.0
    level_loss = zero if feedback is None else feedback.level_loss
    head_loss = zero if feedback is None else feedback.head_loss
    if feedback is None:
        logs["loss_level_consistency"] = 0.0
        logs["loss_head_consistency"] = 0.0
    else:
        logs.update(feedback.logs)
    if lpips_model is not None and float(lpips_weight) > 0.0:
        pred_flat = rendered_b.reshape(batch_size * frames, 3, *rendered_b.shape[-2:]).clamp(0.0, 1.0)
        target_flat = target.reshape(batch_size * frames, 3, *target.shape[-2:]).clamp(0.0, 1.0)
        weight_flat = weight.reshape(batch_size * frames, 1, *weight.shape[-2:]).clamp(0.0, 1.0)
        lpips_sample_weight = sample_weight.view(batch_size, 1).expand(-1, frames).reshape(-1)
        lpips_value = _masked_lpips(
            lpips_model,
            pred_flat,
            target_flat,
            weight_flat,
            sample_weight=lpips_sample_weight,
        )
        loss = loss + float(lpips_weight) * lpips_value
        logs["loss_rgb_render_lpips"] = float(lpips_value.detach().item())
    return RGBRenderLossResult(
        loss=loss,
        level_loss=level_loss,
        head_loss=head_loss,
        logs=logs,
        rendered=rendered_b if return_debug_tensors else None,
        generated_depth=(
            geometry.depth
            if return_debug_tensors or return_generated_depth
            else None
        ),
    )
