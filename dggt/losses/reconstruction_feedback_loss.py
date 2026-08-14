"""Decoded-feature and frozen-head feedback losses for SceneFlow.

The caller owns the decoder/head forwards so the generated geometry can be
shared with the differentiable RGB renderer.  Teacher tensors must come from a
detached/no-grad decode of the clean target latent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


TOKENIZER_LEVELS = (4, 11, 17, 23)


@dataclass
class ReconstructionFeedbackLossResult:
    level_loss: torch.Tensor
    head_loss: torch.Tensor
    logs: dict[str, float | torch.Tensor]


def _masked_sample_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    compute_unweighted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sigma-weighted and unweighted masked means.

    The denominator deliberately excludes ``sample_weight``.  This matches the
    RGB reconstruction loss: a high-noise sample attenuates the whole
    auxiliary gradient instead of being renormalized back to unit weight.
    """
    value = value.float()
    weight = weight.to(device=value.device, dtype=torch.float32)
    if value.shape != weight.shape:
        raise ValueError(
            f"feedback value/weight shapes must match, got {tuple(value.shape)} "
            f"and {tuple(weight.shape)}"
        )
    if sample_weight.ndim != 1 or int(sample_weight.shape[0]) != int(value.shape[0]):
        raise ValueError(
            f"sample_weight must be [B]={value.shape[0]}, got {tuple(sample_weight.shape)}"
        )
    sample_scale = sample_weight.to(device=value.device, dtype=torch.float32)
    sample_scale = sample_scale.view(value.shape[0], *([1] * (value.ndim - 1)))
    denominator = weight.sum().clamp_min(1.0e-6)
    weighted = (value * weight * sample_scale).sum() / denominator
    unweighted = (
        (value * weight).sum() / denominator
        if compute_unweighted
        else weighted.detach()
    )
    return weighted, unweighted


def _patch_weight(
    patch_weight_mask: torch.Tensor | None,
    *,
    batch_size: int,
    frames: int,
    patches: int,
    device: torch.device,
) -> torch.Tensor:
    if patch_weight_mask is None:
        return torch.ones((batch_size, frames, patches), device=device, dtype=torch.float32)
    patch = patch_weight_mask[:batch_size, :frames].to(device=device, dtype=torch.float32)
    if patch.ndim != 4 or int(patch.shape[-1]) != 1:
        raise ValueError(f"patch_weight_mask must be [B,S,P,1], got {tuple(patch.shape)}")
    if int(patch.shape[2]) != patches:
        raise ValueError(
            f"patch_weight_mask has P={patch.shape[2]}, decoded features have P={patches}"
        )
    return patch[..., 0].clamp(0.0, 1.0)


def _mask_to_bshw(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.zeros((batch_size, frames, height, width), device=device)
    value = mask[:batch_size, :frames].to(device=device, dtype=torch.float32)
    if value.ndim == 5 and int(value.shape[2]) in (1, 3):
        value = value.mean(dim=2)
    elif value.ndim == 5 and int(value.shape[-1]) in (1, 3):
        value = value.mean(dim=-1)
    elif value.ndim != 4:
        raise ValueError(f"feedback sky mask has unsupported shape {tuple(value.shape)}")
    if tuple(value.shape[-2:]) != (height, width):
        value = F.interpolate(
            value.reshape(batch_size * frames, 1, *value.shape[-2:]),
            size=(height, width),
            mode="area",
        ).reshape(batch_size, frames, height, width)
    return value.clamp(0.0, 1.0)


def _dense_weight(
    patch_weight_mask: torch.Tensor | None,
    loss_sky_mask_gt: torch.Tensor | None,
    *,
    batch_size: int,
    frames: int,
    patches: int,
    patch_grid: tuple[int, int] | list[int],
    height: int,
    width: int,
    stride: int,
    sky_weight: float,
    device: torch.device,
) -> torch.Tensor:
    sky = _mask_to_bshw(
        loss_sky_mask_gt,
        batch_size=batch_size,
        frames=frames,
        height=height,
        width=width,
        device=device,
    )
    weight = (1.0 - sky) + float(sky_weight) * sky
    patch = _patch_weight(
        patch_weight_mask,
        batch_size=batch_size,
        frames=frames,
        patches=patches,
        device=device,
    )
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    if gh * gw != patches:
        raise ValueError(f"patch_grid {(gh, gw)} does not match P={patches}")
    patch_dense = F.interpolate(
        patch.reshape(batch_size * frames, gh, gw, 1).permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch_size, frames, height, width)
    weight = weight * patch_dense.clamp(0.0, 1.0)
    stride = max(1, int(stride))
    return weight[:, :, ::stride, ::stride].contiguous()


def _teacher_conf_weight(
    conf: torch.Tensor,
    *,
    stride: int,
    power: float,
    floor: float,
) -> torch.Tensor | None:
    """Return detached, per-sample-normalised teacher depth confidence.

    ``power == 0`` is deliberately a true short circuit so the disabled path
    remains bit-identical to the pre-confidence-weighting implementation.
    """
    power = float(power)
    if power == 0.0:
        return None
    floor = float(floor)
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("conf_weight_power must be finite and non-negative")
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("conf_weight_floor must be finite and in (0, 1]")
    conf = _slice_dense(
        conf,
        batch_size=int(conf.shape[0]),
        frames=int(conf.shape[1]),
        stride=stride,
    )
    conf = _scalar_dense_error_map(conf, name="teacher depth confidence")
    weight = conf.detach().float().clamp(floor, 1.0).pow(power)
    return weight / weight.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)


def _selected_patch_tokens(
    geometry: Any,
    *,
    batch_size: int,
    frames: int,
    patches: int,
) -> list[torch.Tensor]:
    image_tokens = getattr(geometry, "image_tokens", None)
    if not isinstance(image_tokens, list):
        raise TypeError("decoded geometry must expose image_tokens as a sparse level list")
    selected: list[torch.Tensor] = []
    for level in TOKENIZER_LEVELS:
        tokens = image_tokens[int(level)]
        if not torch.is_tensor(tokens):
            raise RuntimeError(f"decoded feedback feature level {level} is missing")
        if int(tokens.shape[-2]) < patches:
            raise ValueError(
                f"decoded level {level} has {tokens.shape[-2]} tokens, expected at least {patches}"
            )
        selected.append(tokens[:batch_size, :frames, -patches:, :])
    return selected


def _level_consistency(
    student_geometry: Any,
    teacher_geometry: Any,
    *,
    batch_size: int,
    frames: int,
    patches: int,
    patch_weight: torch.Tensor,
    sample_weight: torch.Tensor,
    compute_unweighted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_levels = _selected_patch_tokens(
        student_geometry,
        batch_size=batch_size,
        frames=frames,
        patches=patches,
    )
    teacher_levels = _selected_patch_tokens(
        teacher_geometry,
        batch_size=batch_size,
        frames=frames,
        patches=patches,
    )
    weighted_levels: list[torch.Tensor] = []
    unweighted_levels: list[torch.Tensor] = []
    for student, teacher in zip(student_levels, teacher_levels):
        student_f = student.float()
        teacher_f = teacher.detach().float()
        student_norm = F.layer_norm(student_f, (int(student_f.shape[-1]),))
        teacher_norm = F.layer_norm(teacher_f, (int(teacher_f.shape[-1]),))
        normalized_l1 = (student_norm - teacher_norm).abs().mean(dim=-1)
        cosine = 1.0 - F.cosine_similarity(student_f, teacher_f, dim=-1, eps=1.0e-6)
        both_zero = (student_f.square().sum(dim=-1) <= 1.0e-12) & (
            teacher_f.square().sum(dim=-1) <= 1.0e-12
        )
        cosine = torch.where(both_zero, torch.zeros_like(cosine), cosine)
        level_map = normalized_l1 + cosine
        weighted, unweighted = _masked_sample_mean(
            level_map,
            patch_weight,
            sample_weight,
            compute_unweighted=compute_unweighted,
        )
        weighted_levels.append(weighted)
        unweighted_levels.append(unweighted)
    return torch.stack(weighted_levels).mean(), torch.stack(unweighted_levels).mean()


def _slice_dense(value: torch.Tensor, *, batch_size: int, frames: int, stride: int) -> torch.Tensor:
    value = value[:batch_size, :frames]
    stride = max(1, int(stride))
    if value.ndim == 5:
        return value[:, :, ::stride, ::stride, :]
    if value.ndim == 4:
        return value[:, :, ::stride, ::stride]
    raise ValueError(f"unsupported dense head tensor shape {tuple(value.shape)}")


def _scalar_dense_error_map(value: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize a scalar dense-head error to ``[B, S, H, W]``.

    DGGT predictions such as depth and dynamic confidence retain a singleton
    channel (``[B,S,H,W,1]``), while head confidence outputs such as
    ``depth_conf`` and ``gs_conf`` are already ``[B,S,H,W]``.  Treating the
    latter as channel-last tensors would accidentally reduce the image width.
    """
    if value.ndim == 4:
        return value
    if value.ndim == 5 and int(value.shape[-1]) == 1:
        return value[..., 0]
    raise ValueError(
        f"{name} feedback error must be [B,S,H,W] or [B,S,H,W,1], "
        f"got {tuple(value.shape)}"
    )


def _head_error_maps(
    student_geometry: Any,
    teacher_geometry: Any,
    *,
    batch_size: int,
    frames: int,
    stride: int,
) -> dict[str, torch.Tensor]:
    student_depth = _slice_dense(
        student_geometry.depth, batch_size=batch_size, frames=frames, stride=stride
    ).float()
    teacher_depth = _slice_dense(
        teacher_geometry.depth, batch_size=batch_size, frames=frames, stride=stride
    ).detach().float()
    depth = _scalar_dense_error_map(
        F.smooth_l1_loss(
            student_depth.clamp_min(1.0e-6).log(),
            teacher_depth.clamp_min(1.0e-6).log(),
            beta=0.1,
            reduction="none",
        ),
        name="depth",
    )

    student_depth_conf = _slice_dense(
        student_geometry.depth_conf, batch_size=batch_size, frames=frames, stride=stride
    ).float()
    teacher_depth_conf = _slice_dense(
        teacher_geometry.depth_conf, batch_size=batch_size, frames=frames, stride=stride
    ).detach().float()
    depth_conf = _scalar_dense_error_map(
        F.smooth_l1_loss(
            torch.log1p(student_depth_conf.clamp_min(0.0)),
            torch.log1p(teacher_depth_conf.clamp_min(0.0)),
            beta=0.1,
            reduction="none",
        ),
        name="depth_conf",
    )

    student_gs = _slice_dense(
        student_geometry.gs_map, batch_size=batch_size, frames=frames, stride=stride
    ).float()
    teacher_gs = _slice_dense(
        teacher_geometry.gs_map, batch_size=batch_size, frames=frames, stride=stride
    ).detach().float()
    if int(student_gs.shape[-1]) < 11 or int(teacher_gs.shape[-1]) < 11:
        raise ValueError("Gaussian head consistency requires RGB/opacity/scale/quaternion channels")
    gs_rgb = F.smooth_l1_loss(
        student_gs[..., :3], teacher_gs[..., :3], beta=0.1, reduction="none"
    ).mean(dim=-1)
    gs_opacity = F.smooth_l1_loss(
        student_gs[..., 3], teacher_gs[..., 3], beta=0.1, reduction="none"
    )
    gs_scale = F.smooth_l1_loss(
        student_gs[..., 4:7].clamp_min(1.0e-6).log(),
        teacher_gs[..., 4:7].clamp_min(1.0e-6).log(),
        beta=0.1,
        reduction="none",
    ).mean(dim=-1)
    student_quat = F.normalize(student_gs[..., 7:11], dim=-1, eps=1.0e-6)
    teacher_quat = F.normalize(teacher_gs[..., 7:11], dim=-1, eps=1.0e-6)
    gs_rotation = 1.0 - (student_quat * teacher_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
    both_quat_zero = (student_gs[..., 7:11].square().sum(dim=-1) <= 1.0e-12) & (
        teacher_gs[..., 7:11].square().sum(dim=-1) <= 1.0e-12
    )
    gs_rotation = torch.where(both_quat_zero, torch.zeros_like(gs_rotation), gs_rotation)
    gaussian = (gs_rgb + gs_opacity + gs_scale + gs_rotation) * 0.25

    student_gs_conf = _slice_dense(
        student_geometry.gs_conf, batch_size=batch_size, frames=frames, stride=stride
    ).float()
    teacher_gs_conf = _slice_dense(
        teacher_geometry.gs_conf, batch_size=batch_size, frames=frames, stride=stride
    ).detach().float()
    gs_conf = _scalar_dense_error_map(
        F.smooth_l1_loss(
            torch.log1p(student_gs_conf.clamp_min(0.0)),
            torch.log1p(teacher_gs_conf.clamp_min(0.0)),
            beta=0.1,
            reduction="none",
        ),
        name="gs_conf",
    )

    student_dynamic = _slice_dense(
        student_geometry.dynamic_conf, batch_size=batch_size, frames=frames, stride=stride
    ).float()
    teacher_dynamic = _slice_dense(
        teacher_geometry.dynamic_conf, batch_size=batch_size, frames=frames, stride=stride
    ).detach().float()
    dynamic = _scalar_dense_error_map(
        F.smooth_l1_loss(
            student_dynamic,
            teacher_dynamic,
            beta=0.5,
            reduction="none",
        ),
        name="dynamic_conf",
    )
    return {
        "depth": depth,
        "depth_conf": depth_conf,
        "gaussian": gaussian,
        "gs_conf": gs_conf,
        "dynamic": dynamic,
    }


def compute_reconstruction_feedback_losses(
    *,
    student_geometry: Any,
    teacher_geometry: Any,
    patch_grid: tuple[int, int] | list[int],
    patch_weight_mask: torch.Tensor | None,
    loss_sky_mask_gt: torch.Tensor | None,
    sky_weight: float,
    max_frames: int,
    render_stride: int,
    sample_weight: torch.Tensor,
    conf_weight_power: float = 1.0,
    conf_weight_floor: float = 0.05,
    defer_log_values: bool = False,
    collect_logs: bool = True,
) -> ReconstructionFeedbackLossResult:
    """Compute four-level and rendering-head consistency losses.

    Frame count, spatial stride, patch/edit mask, sky weighting, and
    per-sample sigma weights intentionally mirror the RGB render loss.
    """
    student_depth = student_geometry.depth
    teacher_depth = teacher_geometry.depth
    batch_size = min(int(student_depth.shape[0]), int(teacher_depth.shape[0]))
    available_frames = min(int(student_depth.shape[1]), int(teacher_depth.shape[1]))
    frames = (
        available_frames
        if int(max_frames) <= 0
        else min(int(max_frames), available_frames)
    )
    if batch_size <= 0 or frames <= 0:
        zero = student_depth.sum() * 0.0
        return ReconstructionFeedbackLossResult(
            level_loss=zero,
            head_loss=zero,
            logs={
                "loss_level_consistency": 0.0,
                "loss_head_consistency": 0.0,
            } if collect_logs else {},
        )
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    patches = gh * gw
    patch_weight = _patch_weight(
        patch_weight_mask,
        batch_size=batch_size,
        frames=frames,
        patches=patches,
        device=student_depth.device,
    )
    level_loss, level_unweighted = _level_consistency(
        student_geometry,
        teacher_geometry,
        batch_size=batch_size,
        frames=frames,
        patches=patches,
        patch_weight=patch_weight,
        sample_weight=sample_weight[:batch_size],
        compute_unweighted=collect_logs,
    )

    height, width = int(student_depth.shape[2]), int(student_depth.shape[3])
    dense_weight = _dense_weight(
        patch_weight_mask,
        loss_sky_mask_gt,
        batch_size=batch_size,
        frames=frames,
        patches=patches,
        patch_grid=patch_grid,
        height=height,
        width=width,
        stride=render_stride,
        sky_weight=sky_weight,
        device=student_depth.device,
    )
    head_maps = _head_error_maps(
        student_geometry,
        teacher_geometry,
        batch_size=batch_size,
        frames=frames,
        stride=render_stride,
    )
    teacher_depth_conf = teacher_geometry.depth_conf[:batch_size, :frames]
    conf_weight = _teacher_conf_weight(
        teacher_depth_conf,
        stride=render_stride,
        power=conf_weight_power,
        floor=conf_weight_floor,
    )
    geom_weight = dense_weight if conf_weight is None else dense_weight * conf_weight
    head_losses: dict[str, torch.Tensor] = {}
    head_unweighted: dict[str, torch.Tensor] = {}
    for name, value in head_maps.items():
        map_weight = (
            geom_weight
            if name in ("depth", "gaussian", "dynamic")
            else dense_weight
        )
        weighted, unweighted = _masked_sample_mean(
            value,
            map_weight,
            sample_weight[:batch_size],
            compute_unweighted=collect_logs,
        )
        head_losses[name] = weighted
        head_unweighted[name] = unweighted
    head_loss = (
        head_losses["depth"]
        + 0.1 * head_losses["depth_conf"]
        + head_losses["gaussian"]
        + 0.1 * head_losses["gs_conf"]
        + head_losses["dynamic"]
    )
    head_loss_unweighted = (
        head_unweighted["depth"]
        + 0.1 * head_unweighted["depth_conf"]
        + head_unweighted["gaussian"]
        + 0.1 * head_unweighted["gs_conf"]
        + head_unweighted["dynamic"]
    )
    head_loss_no_conf = None
    if collect_logs and conf_weight is not None:
        no_conf_losses = {
            name: _masked_sample_mean(
                value,
                dense_weight,
                sample_weight[:batch_size],
                compute_unweighted=False,
            )[0]
            for name, value in head_maps.items()
        }
        head_loss_no_conf = (
            no_conf_losses["depth"]
            + 0.1 * no_conf_losses["depth_conf"]
            + no_conf_losses["gaussian"]
            + 0.1 * no_conf_losses["gs_conf"]
            + no_conf_losses["dynamic"]
        )
    checked = [level_loss, head_loss]
    if collect_logs:
        checked.extend((level_unweighted, head_loss_unweighted))
    if not bool(torch.stack([value.detach() for value in checked]).isfinite().all()):
        raise FloatingPointError("reconstruction feedback loss is non-finite")
    if not collect_logs:
        return ReconstructionFeedbackLossResult(
            level_loss=level_loss,
            head_loss=head_loss,
            logs={},
        )

    def log_value(value: torch.Tensor) -> float | torch.Tensor:
        detached = value.detach().float().reshape(())
        return detached if defer_log_values else float(detached.item())

    logs = {
        "loss_level_consistency": log_value(level_loss),
        "loss_level_consistency_unweighted": log_value(level_unweighted),
        "loss_head_consistency": log_value(head_loss),
        "loss_head_consistency_unweighted": log_value(head_loss_unweighted),
        "loss_head_depth": log_value(head_losses["depth"]),
        "loss_head_depth_conf": log_value(head_losses["depth_conf"]),
        "loss_head_gaussian": log_value(head_losses["gaussian"]),
        "loss_head_gs_conf": log_value(head_losses["gs_conf"]),
        "loss_head_dynamic": log_value(head_losses["dynamic"]),
        "feedback_frames": float(frames),
        "feedback_stride": float(max(1, int(render_stride))),
        "feedback_sample_weight_mean": log_value(
            sample_weight[:batch_size].mean()
        ),
    }
    if conf_weight is not None:
        if head_loss_no_conf is None or not bool(torch.isfinite(head_loss_no_conf)):
            raise FloatingPointError("head consistency without confidence loss is non-finite")
        raw_teacher_conf = _slice_dense(
            teacher_depth_conf,
            batch_size=batch_size,
            frames=frames,
            stride=render_stride,
        )
        raw_teacher_conf = _scalar_dense_error_map(
            raw_teacher_conf,
            name="teacher depth confidence",
        )
        logs.update(
            {
                "loss_head_consistency_no_conf": log_value(head_loss_no_conf),
                "feedback_conf_weight_mean": log_value(raw_teacher_conf.mean()),
                "feedback_conf_weight_power": float(conf_weight_power),
            }
        )
    return ReconstructionFeedbackLossResult(
        level_loss=level_loss,
        head_loss=head_loss,
        logs=logs,
    )
