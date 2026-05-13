"""Pretraining entry point for SceneFlow on raw Waymo clips.

Frozen:
  - VGGT aggregator
  - JointSceneTokenizer

Trainable:
  - WanSceneFlow

Launch:
  torchrun --nproc_per_node=8 train_scene_flow_pretrain.py \
      --image_dir /data/waymo \
      --dggt_ckpt_path pretrained/dggt.pth \
      --tokenizer_ckpt_path logs/tokenizer_t0_waymo_views1/ckpt/scene_tokenizer_latest.pt \
      --feature_stats_path logs/tokenizer_t0_waymo_views1/feature_stats.pt \
      --log_dir logs/scene_flow_pretrain
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from datasets.dataset import WaymoOpenDataset
from dggt.losses.flow_losses import build_rectified_flow_target, compute_total_loss
from dggt.models.scene_flow import WanSceneFlow
from dggt.models.vggt import VGGT
from dggt.utils.feature_stats import load_into_buffers
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.pretrain_pseudo_asset import (
    PretrainBundle,
    apply_uncond_drop,
    build_pretrain_bundle,
)
from dggt.utils.tokens import (
    reattach_special_tokens,
    replace_selected_levels,
    select_patch_pyramid,
    split_special_and_patch,
)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel


TOKENIZER_LEVELS = (4, 11, 17, 23)
SKY_CLASS_INDEX = 9


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, local_rank, world_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(args: argparse.Namespace, device: torch.device):
    enabled = device.type == "cuda" and args.precision == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def format_key_examples(keys: list[str], limit: int = 5) -> str:
    if not keys:
        return "[]"
    examples = ", ".join(keys[:limit])
    suffix = "" if len(keys) <= limit else ", ..."
    return f"[{examples}{suffix}]"


def load_dggt_aggregator_and_tokenizer(
    dggt_ckpt_path: str,
    tokenizer_ckpt_path: str | None,
    device: torch.device,
) -> VGGT:
    model = VGGT().to(device)

    checkpoint = torch.load(dggt_ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported DGGT checkpoint format: {dggt_ckpt_path}")
    missing, unexpected = model.load_state_dict(strip_module_prefix(state), strict=False)
    if is_main_process():
        ignored_missing = [key for key in missing if key.startswith("scene_tokenizer.")]
        real_missing = [key for key in missing if not key.startswith("scene_tokenizer.")]
        print(
            "[ckpt:dggt] "
            f"missing={len(real_missing)} "
            f"ignored_missing_scene_tokenizer={len(ignored_missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )
        if real_missing or unexpected:
            print(
                "[ckpt:dggt] "
                f"missing_examples={format_key_examples(real_missing)} "
                f"unexpected_examples={format_key_examples(unexpected)}",
                flush=True,
            )

    if tokenizer_ckpt_path:
        tok_checkpoint = torch.load(tokenizer_ckpt_path, map_location="cpu")
        tok_state: Any = tok_checkpoint
        if isinstance(tok_checkpoint, dict):
            tok_state = tok_checkpoint.get("scene_tokenizer", tok_checkpoint.get("state_dict", tok_checkpoint))
        if not isinstance(tok_state, dict):
            raise ValueError(f"Unsupported tokenizer checkpoint format: {tokenizer_ckpt_path}")
        tok_state = strip_module_prefix(tok_state)
        if any(key.startswith("scene_tokenizer.") for key in tok_state):
            tok_state = {
                key[len("scene_tokenizer."):]: value
                for key, value in tok_state.items()
                if key.startswith("scene_tokenizer.")
            }
        missing, unexpected = model.scene_tokenizer.load_state_dict(tok_state, strict=False)
        if is_main_process():
            print(f"[ckpt:tokenizer] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            if missing or unexpected:
                print(
                    "[ckpt:tokenizer] "
                    f"missing_examples={format_key_examples(missing)} "
                    f"unexpected_examples={format_key_examples(unexpected)}",
                    flush=True,
                )
        if missing or unexpected:
            raise RuntimeError("Tokenizer checkpoint did not match VGGT.scene_tokenizer.")

    model.eval()
    freeze_module(model)
    return model


def discover_scene_names(image_dir: str, scene_start: int, scene_end: int) -> list[str]:
    root = Path(image_dir)
    scene_names = []
    for idx in range(int(scene_start), int(scene_end)):
        name = f"{idx:03d}"
        if (root / name / "images").is_dir():
            scene_names.append(name)
    if not scene_names:
        raise RuntimeError(
            f"No Waymo scene folders with images found in {image_dir} for "
            f"[{scene_start}, {scene_end})."
        )
    return scene_names


def validate_dynamic_masks(dataset: WaymoOpenDataset, min_frames: int) -> None:
    missing = []
    too_short = []
    for idx, paths in enumerate(dataset.dynamic_mask_path):
        if dataset.views == 1:
            count = len(paths)
        else:
            count = min((len(p) for p in paths), default=0)
        scene = dataset.scenes[idx] if idx < len(dataset.scenes) else f"idx={idx}"
        if count == 0:
            missing.append(scene)
        elif count < min_frames:
            too_short.append((scene, count))
    if missing or too_short:
        msg = []
        if missing:
            msg.append(f"missing fine_dynamic_masks/all: {missing[:8]}")
        if too_short:
            msg.append(f"too few dynamic masks for sequence_length={min_frames}: {too_short[:8]}")
        raise RuntimeError("; ".join(msg))


def split_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "scale_shift_table" in name
            or name.endswith("null_kv")
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_cosine_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int) -> LambdaLR:
    warmup_steps = max(1, int(warmup_steps))
    max_steps = max(warmup_steps + 1, int(max_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    scene_flow: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    step: int,
    log_dir: Path,
    args: argparse.Namespace,
) -> None:
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sf = unwrap_ddp(scene_flow)
    payload = {
        "step": int(step),
        "scene_flow": sf.state_dict(),
        "ema_scene_flow": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, ckpt_dir / f"pretrain_step{step:06d}.pt")
    torch.save({"scene_flow": sf.state_dict()}, ckpt_dir / f"pretrain_step{step:06d}_weights_only.pt")


def load_resume_checkpoint(
    scene_flow: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    resume_path: str | None,
    device: torch.device,
) -> int:
    if not resume_path:
        return 0
    payload = torch.load(resume_path, map_location=device)
    if not isinstance(payload, dict) or "scene_flow" not in payload:
        raise ValueError(f"Unsupported resume checkpoint format: {resume_path}")
    unwrap_ddp(scene_flow).load_state_dict(payload["scene_flow"], strict=True)
    if "ema_scene_flow" in payload:
        ema.load_state_dict(payload["ema_scene_flow"])
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "lr_scheduler" in payload:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    step = int(payload.get("step", 0))
    if is_main_process():
        print(f"[resume] loaded {resume_path} at step={step}", flush=True)
    return step


def init_wandb(args: argparse.Namespace, log_dir: Path):
    if not args.wandb or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or remove --wandb.") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        dir=str(log_dir),
        config=vars(args),
    )
    return run


def log_wandb(run, metrics: dict[str, float], step: int, prefix: str) -> None:
    if run is None:
        return
    run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)


def _latent_pca_grid(z: torch.Tensor, patch_grid: tuple[int, int], max_frames: int) -> torch.Tensor:
    """Project `[B,S,P,C]` latent tokens to an RGB patch grid for qualitative checks."""
    z = z[:1, :max_frames].detach().float().cpu()
    _, seq_len, num_patches, channels = z.shape
    gy, gx = patch_grid
    if num_patches != gy * gx:
        raise ValueError(f"latent patch count {num_patches} != patch_grid {patch_grid}")
    flat = z.reshape(-1, channels)
    flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.shape[0] < 3:
        rgb = flat[:, :3]
    else:
        _, _, vh = torch.pca_lowrank(flat, q=3, center=False)
        rgb = flat @ vh[:, :3]
    lo = rgb.quantile(0.01, dim=0, keepdim=True)
    hi = rgb.quantile(0.99, dim=0, keepdim=True)
    rgb = ((rgb - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    return rgb.reshape(1, seq_len, gy, gx, 3).reshape(seq_len, gy, gx, 3).permute(0, 3, 1, 2)


def _mask_grid(mask: torch.Tensor, patch_grid: tuple[int, int], max_frames: int) -> torch.Tensor:
    mask = mask[:1, :max_frames].detach().float().cpu()
    _, seq_len, num_patches, _ = mask.shape
    gy, gx = patch_grid
    if num_patches != gy * gx:
        raise ValueError(f"mask patch count {num_patches} != patch_grid {patch_grid}")
    return mask.reshape(seq_len, gy, gx, 1).permute(0, 3, 1, 2)


def _normalized_mask_grid(mask: torch.Tensor, patch_grid: tuple[int, int], max_frames: int) -> torch.Tensor:
    grid = _mask_grid(mask, patch_grid, max_frames)
    hi = grid.quantile(0.99).clamp_min(1e-6)
    return (grid / hi).clamp(0.0, 1.0)


def _image_grid(images: torch.Tensor, max_frames: int) -> torch.Tensor:
    return images[:1, :max_frames].detach().float().cpu().reshape(-1, *images.shape[2:]).clamp(0.0, 1.0)


def _semantic_logits_to_sky_mask(
    semantic_logits: torch.Tensor,
    *,
    sky_class_index: int = SKY_CLASS_INDEX,
) -> torch.Tensor:
    """Convert predicted semantic logits `[B,S,H,W,C]` to sky mask `[B,S,3,H,W]`."""
    if semantic_logits.ndim != 5:
        raise ValueError(f"Expected semantic_logits [B,S,H,W,C], got {tuple(semantic_logits.shape)}")
    if semantic_logits.shape[-1] <= int(sky_class_index):
        raise ValueError(
            f"semantic_logits has {semantic_logits.shape[-1]} classes, "
            f"cannot read sky_class_index={sky_class_index}"
        )
    sky = (semantic_logits.float().argmax(dim=-1) == int(sky_class_index)).to(dtype=semantic_logits.dtype)
    return sky[:, :, None].repeat(1, 1, 3, 1, 1)


def _sky_mask_image_grid(sky_mask: torch.Tensor, max_frames: int) -> torch.Tensor:
    mask = sky_mask[:1, :max_frames, :1].detach().float().cpu()
    return mask.reshape(-1, *mask.shape[2:]).clamp(0.0, 1.0)


def _predict_camera_mats(
    pose_enc: torch.Tensor,
    image_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_hw
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (height, width))
    extrinsic_3x4 = extrinsics[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=extrinsic_3x4.dtype).view(1, 1, 4)
    extrinsic = torch.cat([extrinsic_3x4, bottom.expand(extrinsic_3x4.shape[0], -1, -1)], dim=1)
    intrinsic = intrinsics[0]
    return extrinsic, intrinsic


def _render_background(
    model: VGGT,
    images: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    mode: str = "sky",
) -> torch.Tensor:
    _, seq_len, _, height, width = images.shape
    if mode == "sky" and hasattr(model, "sky_model") and model.sky_model is not None:
        bg_render = model.sky_model(images, extrinsic, intrinsic).float()
        denom = (bg_render.max() - bg_render.min()).clamp_min(1e-8)
        bg_render = ((bg_render - bg_render.min()) / denom).clamp(0.0, 1.0)
        return bg_render
    return torch.zeros((seq_len, height, width, 3), dtype=images.dtype, device=images.device)


def split_image_tokens_for_heads(image_tokens_list: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    aggregated_tokens_list = []
    dino_token_list = []
    for tokens in image_tokens_list:
        if tokens.shape[-1] != 3072:
            raise ValueError(f"Expected 3072-wide image tokens, got {tokens.shape[-1]}")
        dino, frame, global_tokens = tokens.split([1024, 1024, 1024], dim=-1)
        dino_token_list.append(dino)
        aggregated_tokens_list.append(torch.cat([frame, global_tokens], dim=-1))
    return aggregated_tokens_list, dino_token_list


def alpha_t(t: torch.Tensor, t0: torch.Tensor | float, alpha: torch.Tensor, gamma0: torch.Tensor, gamma1: float = 0.1):
    if not torch.is_tensor(t0):
        t0 = torch.tensor(float(t0), dtype=t.dtype, device=t.device)
    sigma = torch.log(torch.tensor(gamma1, dtype=alpha.dtype, device=alpha.device)) / ((gamma0) ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    return (alpha * conf).float()


def _rasterize_scene(
    means: torch.Tensor,
    rgbs: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotation: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from gsplat.rendering import rasterization

    if means.numel() == 0:
        empty_render = torch.zeros((1, height, width, 4), dtype=torch.float32, device=viewmat.device)
        empty_alpha = torch.zeros((1, height, width, 1), dtype=torch.float32, device=viewmat.device)
        return empty_render, empty_alpha

    renders_chunk, alphas_chunk, _ = rasterization(
        means=means,
        quats=rotation,
        scales=scales,
        opacities=opacity,
        colors=rgbs,
        viewmats=viewmat,
        Ks=intrinsic,
        width=width,
        height=height,
        render_mode="RGB+ED",
    )
    return renders_chunk, alphas_chunk


def _render_gs_map_rgb(
    model: VGGT,
    images: torch.Tensor,
    masks: torch.Tensor | None,
    timestamps: torch.Tensor,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
    device: torch.device,
    max_frames: int,
    background_mode: str = "sky",
    use_sky_mask: bool = True,
) -> torch.Tensor:
    """Render 3DGS exactly matching `inference.py` mode=2.

    Static branch: rasterized once with `bg_mask & (dy_map < 0.5)` (no extra
    valid_depth filter, no sigmoid on the threshold).
    Dynamic branch: per-frame, gated by `bg_mask` only.
    Background: either GT-image sky model (with min-max norm — matches official),
    or pure black when `background_mode != "sky"`.

    When `use_sky_mask=False` (used for the fully generated path the user
    requires), `bg_mask` is replaced with an all-True tensor — no GT sky
    information is consumed.
    """
    # The renderer only consumes sample 0 throughout; pin every batched tensor
    # to that slice up front so downstream indexing can't broadcast against B > 1.
    images = images[:1]
    pose_enc = pose_enc[:1]
    depth = depth[:1]
    gs_map = gs_map[:1]
    gs_conf = gs_conf[:1]
    dynamic_conf = dynamic_conf[:1]
    if masks is not None:
        masks = masks[:1]

    _, seq_len, _, height, width = images.shape
    # `frames` only controls how many output views are produced (displayed).
    # The static-GS accumulation MUST cover the full sequence to match
    # inference.py mode=2 — slicing the GS data to `[:frames]` was the source
    # of the dggt_clean_3dgs_rgb checkerboard artifacts when val_log_images
    # was much smaller than sequence_length.
    frames = min(int(max_frames), int(seq_len))
    depth = depth.float()
    pose_enc = pose_enc.float()
    extrinsic, intrinsic = _predict_camera_mats(pose_enc, (height, width), device)
    point_map = unproject_depth_map_to_point_map(
        depth[0].detach().cpu(),
        extrinsic[:, :3, :].detach().cpu(),
        intrinsic.detach().cpu(),
    )
    # [1, S, H, W, 3] to match inference.py indexing semantics.
    point_map = torch.from_numpy(point_map).to(device=device, dtype=torch.float32)[None, ...]

    if masks is not None and use_sky_mask:
        sky_mask = masks.to(device).permute(0, 1, 3, 4, 2)
        bg_mask = (sky_mask == 0).any(dim=-1)
    else:
        bg_mask = torch.ones((1, seq_len, height, width), dtype=torch.bool, device=device)

    bg_render = _render_background(
        model, images, extrinsic, intrinsic, mode=background_mode
    )
    timestamps = timestamps[:seq_len].to(device=device, dtype=torch.float32)
    # Raw logits — inference.py uses `dy_map < 0.5` (no sigmoid) for the threshold.
    dy_map = dynamic_conf.squeeze(-1).float()

    # === Static branch (matches inference.py mode=2 exactly, full S frames) ===
    static_mask = bg_mask & (dy_map < 0.5)
    static_points = point_map[static_mask].reshape(-1, 3)
    static_dynamic_prob = dy_map[static_mask].sigmoid()
    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
    static_opacity = static_opacity * (1.0 - static_dynamic_prob)
    static_gs_conf = gs_conf[static_mask]
    static_frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
    gs_timestamps = timestamps[static_frame_idx] if static_frame_idx.numel() > 0 else timestamps.new_zeros((0,))

    # === Dynamic branch (per-frame, bg_mask only — only for displayed frames) ===
    dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
    for frame_idx in range(frames):
        bg_mask_i = bg_mask[:, frame_idx]
        dynamic_point = point_map[:, frame_idx][bg_mask_i].reshape(-1, 3)
        dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, frame_idx], bg_mask_i)
        dynamic_prob = dy_map[:, frame_idx][bg_mask_i].sigmoid()
        dynamic_opacity = dynamic_opacity * dynamic_prob
        dynamic_points.append(dynamic_point)
        dynamic_rgbs.append(dynamic_rgb)
        dynamic_opacitys.append(dynamic_opacity)
        dynamic_scales.append(dynamic_scale)
        dynamic_rotations.append(dynamic_rotation)

    renders = []
    for frame_idx in range(frames):
        t0 = timestamps[frame_idx]
        static_opacity_t = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
        static_gs_list = [static_points, static_rgbs, static_opacity_t, static_scales, static_rotations]
        world_points, rgbs, opacity, scales, rotation = concat_list(
            static_gs_list,
            [
                dynamic_points[frame_idx],
                dynamic_rgbs[frame_idx],
                dynamic_opacitys[frame_idx],
                dynamic_scales[frame_idx],
                dynamic_rotations[frame_idx],
            ],
        )
        rendered_raw, alpha = _rasterize_scene(
            means=world_points.float(),
            rgbs=rgbs.float().clamp(0.0, 1.0),
            opacity=opacity.float().view(-1),
            scales=scales.float().clamp_min(1e-5),
            rotation=rotation.float(),
            viewmat=extrinsic[frame_idx : frame_idx + 1],
            intrinsic=intrinsic[frame_idx : frame_idx + 1],
            height=height,
            width=width,
        )
        foreground = rendered_raw[..., :3]
        composed = alpha * foreground + (1.0 - alpha) * bg_render[frame_idx : frame_idx + 1]
        renders.append(composed[0].permute(2, 0, 1).detach().cpu().float().clamp(0.0, 1.0))
    return torch.stack(renders, dim=0)


@torch.no_grad()
def render_validation_rgb(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render three validation RGB grids (clean / generated / recon).

    Memory optimization: each branch is processed sequentially — compute heads,
    render to CPU, then delete GPU tensors and empty the CUDA cache before the
    next branch.  This reduces peak GPU memory from ~3× a single branch to ~1×.
    """
    images = batch["images"].to(device, non_blocking=True)
    masks = batch.get("masks")
    if masks is not None:
        masks = masks.to(device, non_blocking=True)
    frames = min(int(args.val_log_images), int(images.shape[1]))
    sf = unwrap_ddp(scene_flow)
    timestamps = batch["timestamps"][0] if torch.is_tensor(batch["timestamps"]) else torch.as_tensor(batch["timestamps"][0])

    result: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Phase 0: Run the shared aggregator ONCE and build the three sets of
    # modified image tokens.  Keep image_tokens_list alive for heads;
    # free decode intermediates eagerly.
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        outputs = vggt_model.get_aggregator_token_outputs(images)
        aggregated_tokens_list = outputs["aggregated_tokens_list"]
        image_tokens_list = outputs["image_tokens_list"]
        dino_token_list = outputs["dino_token_list"]
        patch_start_idx = int(outputs["patch_start_idx"])
        del outputs

        # --- Generated branch tokens ---
        z_generated = sf.denormalize(z_generated_raw_n.float())
        decoded_patch_tokens = vggt_model.scene_tokenizer.decode(z_generated, patch_grid=args.patch_grid)
        del z_generated
        decoded_full_tokens = reattach_special_tokens(
            image_tokens_list, TOKENIZER_LEVELS, patch_start_idx, decoded_patch_tokens,
        )
        del decoded_patch_tokens
        generated_image_tokens = replace_selected_levels(
            image_tokens_list, TOKENIZER_LEVELS, decoded_full_tokens,
        )
        del decoded_full_tokens

        # --- Recon branch tokens ---
        tokens_4 = select_patch_pyramid(image_tokens_list, TOKENIZER_LEVELS, patch_start_idx)
        z_recon = vggt_model.scene_tokenizer.encode(tokens_4, patch_grid=args.patch_grid)
        del tokens_4
        recon_patch_tokens = vggt_model.scene_tokenizer.decode(z_recon, patch_grid=args.patch_grid)
        del z_recon
        recon_full_tokens = reattach_special_tokens(
            image_tokens_list, TOKENIZER_LEVELS, patch_start_idx, recon_patch_tokens,
        )
        del recon_patch_tokens
        recon_image_tokens = replace_selected_levels(
            image_tokens_list, TOKENIZER_LEVELS, recon_full_tokens,
        )
        del recon_full_tokens

    # ------------------------------------------------------------------
    # Phase 1: CLEAN branch (uses original aggregated/dino/image tokens)
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        # Heads MUST run with autocast disabled (matches VGGT.forward).
        with torch.cuda.amp.autocast(enabled=False):
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            depth, _ = vggt_model.depth_head(aggregated_tokens_list, images, patch_start_idx)
            dynamic_conf, _ = vggt_model.instance_head(dino_token_list, images, patch_start_idx)
            clean_gs_map, clean_gs_conf = vggt_model.gs_head(image_tokens_list, images, patch_start_idx)

    # Free original tokens — no longer needed after clean heads.
    del aggregated_tokens_list, dino_token_list, image_tokens_list

    result["dggt_clean_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model, images, masks, timestamps,
        pose_enc, depth, clean_gs_map, clean_gs_conf, dynamic_conf,
        device, frames, background_mode="sky", use_sky_mask=True,
    )
    del pose_enc, depth, dynamic_conf, clean_gs_map, clean_gs_conf
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: GENERATED branch
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        gen_agg, gen_dino = split_image_tokens_for_heads(generated_image_tokens)
        with torch.cuda.amp.autocast(enabled=False):
            raw_gs_map, raw_gs_conf = vggt_model.gs_head(generated_image_tokens, images, patch_start_idx)
            generated_pose_enc = vggt_model.camera_head(gen_agg)[-1]
            generated_depth, _ = vggt_model.depth_head(gen_agg, images, patch_start_idx)
            generated_dynamic_conf, _ = vggt_model.instance_head(gen_dino, images, patch_start_idx)
            generated_semantic_logits, _ = vggt_model.semantic_head(gen_dino, images, patch_start_idx)
            generated_sky_mask = _semantic_logits_to_sky_mask(generated_semantic_logits)

    del generated_image_tokens, gen_agg, gen_dino

    # Generated path consumes no GT sky mask and no sky_model background.
    # Its sky/non-sky split comes from the semantic_head output decoded from
    # generated tokens. Background remains black because sky_model reads GT RGB
    # source images and is therefore not allowed in this no-GT diagnostic.
    result["generated_pred_sky_mask"] = _sky_mask_image_grid(generated_sky_mask, frames)
    result["generated_raw_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model, images, generated_sky_mask, timestamps,
        generated_pose_enc, generated_depth, raw_gs_map, raw_gs_conf,
        generated_dynamic_conf,
        device, frames, background_mode="black", use_sky_mask=True,
    )
    del (
        generated_pose_enc,
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        generated_semantic_logits,
        generated_sky_mask,
    )
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 3: RECON branch (tokenizer round-trip)
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        recon_agg, recon_dino = split_image_tokens_for_heads(recon_image_tokens)
        with torch.cuda.amp.autocast(enabled=False):
            recon_pose_enc = vggt_model.camera_head(recon_agg)[-1]
            recon_depth, _ = vggt_model.depth_head(recon_agg, images, patch_start_idx)
            recon_dynamic_conf, _ = vggt_model.instance_head(recon_dino, images, patch_start_idx)
            recon_gs_map, recon_gs_conf = vggt_model.gs_head(recon_image_tokens, images, patch_start_idx)

    del recon_image_tokens, recon_agg, recon_dino

    result["tokenizer_recon_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model, images, masks, timestamps,
        recon_pose_enc, recon_depth, recon_gs_map, recon_gs_conf,
        recon_dynamic_conf,
        device, frames, background_mode="sky", use_sky_mask=True,
    )
    del recon_pose_enc, recon_depth, recon_dynamic_conf, recon_gs_map, recon_gs_conf
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # GT images grid (always cheap)
    # ------------------------------------------------------------------
    result["input_rgb_gt"] = _image_grid(images, frames)
    return result


@torch.no_grad()
def cfg_sample_pretrain_latents(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
    guidance_scale: float | None = None,
) -> torch.Tensor:
    """Classifier-free guidance sampling from pure noise.

    Two forward passes per step: a conditional pass with the real KV tokens, and
    an unconditional pass with length-0 KV (which `_prepare_asset_kv` replaces
    with `null_kv`). The velocity is `v = v_uncond + s * (v_cond - v_uncond)`.
    """
    scale = float(args.guidance_scale) if guidance_scale is None else float(guidance_scale)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=args.shift,
        invert_sigmas=True,
    )
    scheduler.set_timesteps(num_inference_steps=args.val_sample_steps, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))

    z_splat = torch.zeros_like(bundle.z_clean_n)
    scaffold_tok = torch.zeros_like(bundle.z_clean_n)
    z = torch.empty_like(bundle.z_clean_n)
    z.normal_(generator=generator)
    batch_size = z.shape[0]
    sf = unwrap_ddp(scene_flow)

    kv_dim = bundle.F_asset_tokens.shape[-1]
    F_uncond = bundle.F_asset_tokens.new_zeros((batch_size, 0, kv_dim))

    do_cfg = abs(scale - 1.0) > 1e-6 and bundle.F_asset_tokens.shape[1] > 0

    for timestep in scheduler.timesteps:
        sigma = (timestep / scheduler.config.num_train_timesteps).to(device=device)
        sigma = sigma.expand(batch_size)
        v_cond = sf(
            z,
            sigma,
            z_splat,
            scaffold_tok,
            bundle.M_preserve,
            bundle.M_source,
            bundle.M_dest,
            bundle.F_asset_tokens,
            encoder_attention_mask=bundle.encoder_attention_mask,
            return_mid=False,
        )
        if do_cfg:
            v_uncond = sf(
                z,
                sigma,
                z_splat,
                scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                F_uncond,
                encoder_attention_mask=None,
                return_mid=False,
            )
            v = v_uncond + scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = scheduler.step(model_output=v, timestep=timestep, sample=z, return_dict=False)[0]

    return z


def sample_pretrain_latents(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    """Backward-compatible alias kept for callers that want default-scale sampling."""
    return cfg_sample_pretrain_latents(scene_flow, bundle, args, step, device)


def build_full_scene_bundle(z_clean_n: torch.Tensor, kv_dim: int) -> PretrainBundle:
    B, S, P, _ = z_clean_n.shape
    mask = z_clean_n.new_zeros((B, S, P, 1))
    return PretrainBundle(
        z_clean_n=z_clean_n,
        M_preserve=mask,
        M_source=torch.zeros_like(mask),
        M_dest=torch.ones_like(mask),
        F_asset_tokens=z_clean_n.new_empty((B, 0, int(kv_dim))),
        encoder_attention_mask=None,
        F_asset_lengths=torch.zeros((B,), device=z_clean_n.device, dtype=torch.long),
    )


def save_validation_images(
    bundle,
    z_generated_raw: torch.Tensor,
    rgb_images: dict[str, torch.Tensor] | None,
    log_dir: Path,
    step: int,
    args: argparse.Namespace,
    scale_suffix: str | None = None,
    only_generated: bool = False,
) -> dict[str, Path]:
    """Dump validation artifacts.

    `scale_suffix` is appended to every filename so multi-CFG dumps don't clash.
    `only_generated=True` skips latent_pca/target/M_dest/input_rgb_gt and only
    writes the generated artifacts (used for the secondary CFG scales).
    """
    out_dir = log_dir / "validation" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = min(int(args.val_log_images), int(bundle.z_clean_n.shape[1]))
    suffix = f"__{scale_suffix}" if scale_suffix else ""

    images: dict[str, torch.Tensor] = {
        f"generated_raw_latent_pca{suffix}": _latent_pca_grid(z_generated_raw, args.patch_grid, frames),
        f"abs_error{suffix}": _normalized_mask_grid(
            (z_generated_raw - bundle.z_clean_n).abs().mean(dim=-1, keepdim=True),
            args.patch_grid,
            frames,
        ),
    }
    if not only_generated:
        images["target_latent_pca"] = _latent_pca_grid(bundle.z_clean_n, args.patch_grid, frames)
        # Skip M_dest/M_preserve in full_scene mode — they are constant 1/0 and carry
        # no info; in pseudo_edit they retain their per-frame structure.
        if args.pretrain_task != "full_scene":
            images["M_dest"] = _mask_grid(bundle.M_dest, args.patch_grid, frames)
            images["M_preserve"] = _mask_grid(bundle.M_preserve, args.patch_grid, frames)

    paths: dict[str, Path] = {}
    for name, tensor in images.items():
        path = out_dir / f"{name}.jpg"
        save_image_grid(tensor, path, nrow=frames)
        paths[name] = path

    if rgb_images:
        skip_for_extra = {"input_rgb_gt", "tokenizer_recon_3dgs_rgb", "dggt_clean_3dgs_rgb"}
        for name, tensor in rgb_images.items():
            if only_generated and name in skip_for_extra:
                continue
            fname = f"{name}{suffix}.jpg" if name.startswith("generated_") else f"{name}.jpg"
            path = out_dir / fname
            save_image_grid(tensor, path, nrow=frames)
            key = f"{name}{suffix}" if name.startswith("generated_") else name
            paths[key] = path
    return paths


def train_step(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    bundle = build_pretrain_bundle_from_batch(batch, vggt_model, scene_flow, device, args)

    # CFG training prerequisite: drop cross-attn KV per-sample with probability p.
    # Gated by .training so validation (eval mode) sees the full conditioning.
    if unwrap_ddp(scene_flow).training and args.uncond_drop_prob > 0.0:
        bundle = apply_uncond_drop(bundle, args.uncond_drop_prob)

    target = build_rectified_flow_target(
        scheduler,
        bundle.z_clean_n,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        loss_weighting_scheme=args.loss_weighting_scheme,
    )

    z_splat = torch.zeros_like(bundle.z_clean_n)
    scaffold_tok = torch.zeros_like(bundle.z_clean_n)
    with autocast_context(args, device):
        v_pred = scene_flow(
            target.z_t,
            target.sigmas,
            z_splat,
            scaffold_tok,
            bundle.M_preserve,
            bundle.M_source,
            bundle.M_dest,
            bundle.F_asset_tokens,
            encoder_attention_mask=bundle.encoder_attention_mask,
            return_mid=False,
        )
        loss, logs = compute_total_loss(
            v_pred=v_pred,
            v_gt=target.v_gt,
            eps=target.eps,
            bundle=bundle,
            sd3_weights=target.weights,
            lambda_flow=args.lambda_flow,
            lambda_preserve=args.lambda_preserve,
            lambda_repa=0.0,
            lambda_identity=0.0,
            identity_batch=False,
            preserve_floor=args.preserve_floor,
        )

    logs["kv_tokens_mean"] = float(bundle.F_asset_lengths.float().mean().item())
    logs["dest_frac"] = float(bundle.M_dest.float().mean().item())
    logs["sigma_mean"] = float(target.sigmas.float().mean().item())
    return loss, logs


def build_pretrain_bundle_from_batch(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
):
    images = batch["images"].to(device, non_blocking=True)
    if images.ndim != 5:
        raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
    _, seq_len = images.shape[:2]
    if seq_len < 2:
        raise ValueError("SceneFlow pretraining requires sequence_length >= 2 for cross-frame KV.")

    sf_root = unwrap_ddp(scene_flow)
    with torch.no_grad():
        with autocast_context(args, device):
            outputs = vggt_model.get_aggregator_token_outputs(images)
            image_tokens_list = outputs["image_tokens_list"]
            patch_start_idx = int(outputs["patch_start_idx"])
            tokens_4 = select_patch_pyramid(image_tokens_list, TOKENIZER_LEVELS, patch_start_idx)
            _, image_tokens_last = split_special_and_patch(image_tokens_list[-1], patch_start_idx)
            if image_tokens_last.shape[-2] != args.patch_grid[0] * args.patch_grid[1]:
                raise ValueError(
                    f"Expected {args.patch_grid[0] * args.patch_grid[1]} patch tokens, "
                    f"got {image_tokens_last.shape[-2]}."
                )
            z_clean = vggt_model.scene_tokenizer.encode(tokens_4, patch_grid=args.patch_grid)
        z_clean_n = sf_root.normalize(z_clean.float())
        image_tokens_last = image_tokens_last.detach()
        del outputs, image_tokens_list, tokens_4, z_clean

    has_dyn = "dynamic_mask" in batch
    if has_dyn:
        dynamic_mask = batch["dynamic_mask"].to(device, non_blocking=True)
        if dynamic_mask.ndim != 5:
            raise ValueError(f"Expected dynamic_mask [B,S,3,H,W], got {tuple(dynamic_mask.shape)}")
        bundle = build_pretrain_bundle(
            z_clean_n=z_clean_n,
            image_tokens_last=image_tokens_last.float(),
            dynamic_mask=dynamic_mask,
            patch_grid=args.patch_grid,
            K_max=args.K_max,
            min_inst_patches=args.min_inst_patches,
            max_inst_patches=args.max_inst_patches,
            ref_offset=max(1, seq_len // 2),
            device=device,
            dtype=z_clean_n.dtype,
            dyn_threshold=args.dyn_threshold,
        )
    else:
        if args.pretrain_task == "pseudo_edit":
            raise RuntimeError("dynamic_mask absent; pseudo_edit pretraining requires fine_dynamic_masks/all.")
        bundle = build_full_scene_bundle(z_clean_n, kv_dim=image_tokens_last.shape[-1])

    if args.pretrain_task == "full_scene":
        # Override masks to "full-scene generation" (M_dest=1 everywhere) while
        # retaining the pseudo-asset cross-attn KV from build_pretrain_bundle so
        # cross-attn is exercised during training (required for CFG inference).
        bundle = PretrainBundle(
            z_clean_n=bundle.z_clean_n,
            M_preserve=torch.zeros_like(bundle.M_dest),
            M_source=torch.zeros_like(bundle.M_dest),
            M_dest=torch.ones_like(bundle.M_dest),
            F_asset_tokens=bundle.F_asset_tokens,
            encoder_attention_mask=bundle.encoder_attention_mask,
            F_asset_lengths=bundle.F_asset_lengths,
        )
    return bundle


@torch.no_grad()
def run_validation(
    loader: DataLoader,
    vggt_model: VGGT,
    scene_flow: nn.Module,
    scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    log_dir: Path,
    wandb_run,
) -> dict[str, float]:
    scene_flow_was_training = scene_flow.training
    scene_flow.eval()
    sums: dict[str, float] = {}
    count = 0
    first_batch: dict[str, Any] | None = None

    iterator = loader
    if is_main_process() and not args.no_tqdm:
        iterator = tqdm(
            loader,
            total=args.val_batches,
            desc=f"val {step:06d}",
            dynamic_ncols=True,
            leave=False,
        )

    for batch in iterator:
        if count >= args.val_batches:
            break
        if first_batch is None and is_main_process():
            first_batch = batch
        loss, logs = train_step(batch, vggt_model, scene_flow, scheduler, device, args)
        logs = dict(logs)
        logs["loss"] = float(loss.detach().item())
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["batches"] = float(count)

    if is_main_process():
        if first_batch is not None and args.val_log_images > 0:
            # Free cached CUDA memory from the validation loss loop so the
            # memory-intensive CFG sampling + rendering has maximum headroom.
            torch.cuda.empty_cache()

            first_bundle = build_pretrain_bundle_from_batch(
                first_batch,
                vggt_model,
                scene_flow,
                device,
                args,
            )

            # Primary scale samples drive latent PCA / abs_error. Extra scales
            # only contribute additional RGB grids for side-by-side CFG comparison.
            primary_scale = float(args.guidance_scale)
            extra_scales = []
            if args.val_guidance_scales:
                for s in args.val_guidance_scales.split(","):
                    s = s.strip()
                    if not s:
                        continue
                    s_val = float(s)
                    if abs(s_val - primary_scale) > 1e-6:
                        extra_scales.append(s_val)

            z_generated_raw = cfg_sample_pretrain_latents(
                scene_flow,
                first_bundle,
                args,
                step,
                device,
                guidance_scale=primary_scale,
            )
            rgb_images = None
            if not args.no_val_render_rgb:
                # Free CFG sampling intermediates before the heavy rendering.
                torch.cuda.empty_cache()
                rgb_images = render_validation_rgb(
                    first_batch,
                    vggt_model,
                    scene_flow,
                    z_generated_raw,
                    args,
                    device,
                )
            image_paths = save_validation_images(
                first_bundle,
                z_generated_raw,
                rgb_images,
                log_dir,
                step,
                args,
                scale_suffix=f"cfg{primary_scale:g}",
            )

            extra_paths: dict[str, Path] = {}
            for s_val in extra_scales:
                z_extra = cfg_sample_pretrain_latents(
                    scene_flow,
                    first_bundle,
                    args,
                    step,
                    device,
                    guidance_scale=s_val,
                )
                rgb_extra = None
                if not args.no_val_render_rgb:
                    rgb_extra = render_validation_rgb(
                        first_batch,
                        vggt_model,
                        scene_flow,
                        z_extra,
                        args,
                        device,
                    )
                extra_paths.update(
                    save_validation_images(
                        first_bundle,
                        z_extra,
                        rgb_extra,
                        log_dir,
                        step,
                        args,
                        scale_suffix=f"cfg{s_val:g}",
                        only_generated=True,
                    )
                )

            if wandb_run is not None:
                import wandb

                image_log: dict[str, Any] = {}
                for name, path in image_paths.items():
                    image_log[f"validation/{name}"] = wandb.Image(str(path))
                for name, path in extra_paths.items():
                    image_log[f"validation/{name}"] = wandb.Image(str(path))
                wandb_run.log(image_log, step=step)
        metrics_text = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"[validation {step:06d}] {metrics_text}", flush=True)
        log_wandb(wandb_run, metrics, step, "validation")

    if scene_flow_was_training:
        scene_flow.train()
    # Rank 0 does CFG sampling + RGB rendering after the metric loop, which
    # can take seconds. A barrier here keeps all ranks aligned so the next
    # training iteration's allreduce won't stall on a rank-0 catch-up.
    if is_distributed():
        dist.barrier()
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneFlow pretraining on raw Waymo clips.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, default=None)
    parser.add_argument("--dggt_ckpt_path", type=str, required=True)
    parser.add_argument("--tokenizer_ckpt_path", type=str, default=None)
    parser.add_argument("--feature_stats_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--patch_grid_h", type=int, default=37)
    parser.add_argument("--patch_grid_w", type=int, default=37)

    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=600)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--pretrain_task", type=str, default="full_scene", choices=("full_scene", "pseudo_edit"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50,
                        help="Plain-text log cadence when tqdm is disabled (--no_tqdm).")
    parser.add_argument("--wandb_log_every", type=int, default=50,
                        help="Report averaged training metrics to wandb every N optimizer steps.")
    parser.add_argument("--val_scene_start", type=int, default=None)
    parser.add_argument("--val_scene_end", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--val_log_images", type=int, default=4)
    parser.add_argument("--val_sample_steps", type=int, default=30)
    parser.add_argument("--no_val_render_rgb", action="store_true")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="RAE official config uses wd=0.0 for from-scratch DiT on frozen-encoder latents.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9995,
                        help="RAE uses 0.9995 (half-life ~1.4k steps); smoother validation than 0.999.")
    parser.add_argument(
        "--shift",
        type=float,
        default=16.0,
        help=(
            "FlowMatch noise-schedule shift. Per RAE (arxiv 2510.11690) "
            "shift = sqrt(m / m_ref). m_ref=4096. Per-frame m = 25*37*768 "
            "= 710400 -> alpha ~= 13.2; per-clip (S=8) m = 5.68M -> alpha "
            "~= 37. We pick 16 as a compromise between per-frame and "
            "per-clip dimension counts. Use 6 only when matching the "
            "legacy low-D Wan recipe."
        ),
    )
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_preserve", type=float, default=1.0)
    parser.add_argument("--preserve_floor", type=float, default=0.2)

    parser.add_argument("--K_max", type=int, default=3)
    parser.add_argument("--min_inst_patches", type=int, default=4)
    parser.add_argument("--max_inst_patches", type=int, default=150)
    parser.add_argument("--dyn_threshold", type=float, default=0.05)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help=(
            "CFG scale for validation sampling. RAE's reported FID 1.51 uses "
            "scale=1.0 (no guidance). Higher scales amplify per-patch noise "
            "into grid artifacts early in training; bump only after the model "
            "converges enough that cond/uncond diverge meaningfully."
        ),
    )
    parser.add_argument("--uncond_drop_prob", type=float, default=0.1,
                        help="Per-sample probability of dropping cross-attn KV during training (CFG prerequisite).")
    parser.add_argument("--val_guidance_scales", type=str, default="",
                        help="Comma-separated extra CFG scales to dump in validation RGB (in addition to --guidance_scale).")

    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--loss_weighting_scheme", type=str, default="none")
    parser.add_argument("--precision", type=str, default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dggt-flow")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.patch_grid = (int(args.patch_grid_h), int(args.patch_grid_w))
    if args.patch_grid[0] <= 0 or args.patch_grid[1] <= 0:
        raise ValueError("--patch_grid_h and --patch_grid_w must be positive.")
    if args.sequence_length < 2:
        raise ValueError("--sequence_length must be >= 2.")
    if args.val_image_dir is None:
        args.val_image_dir = args.image_dir
    if args.val_scene_start is None:
        args.val_scene_start = 0 if args.val_image_dir != args.image_dir else args.scene_end
    if args.val_scene_end is None:
        args.val_scene_end = args.val_scene_start

    device, local_rank, world_size = setup_distributed()
    seed_everything(args.seed + get_rank())

    log_dir = Path(args.log_dir)
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        config = dict(vars(args))
        config["patch_grid"] = list(args.patch_grid)
        (log_dir / "config.json").write_text(json.dumps(config, indent=2))
    wandb_run = init_wandb(args, log_dir)

    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path,
        args.tokenizer_ckpt_path,
        device,
    )

    scene_flow = WanSceneFlow.from_scene_config(bring_up=False, patch_grid=args.patch_grid).to(device)
    scene_flow.enable_gradient_checkpointing()
    load_into_buffers(scene_flow, args.feature_stats_path, token_dim=768)

    scene_names = discover_scene_names(args.image_dir, args.scene_start, args.scene_end)
    dataset = WaymoOpenDataset(
        image_dir=args.image_dir,
        scene_names=scene_names,
        sequence_length=args.sequence_length,
        mode=1,
        views=1,
    )
    if args.pretrain_task == "pseudo_edit":
        validate_dynamic_masks(dataset, args.sequence_length)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = None
    if args.val_every > 0 and args.val_batches > 0 and args.val_scene_end > args.val_scene_start:
        val_scene_names = discover_scene_names(args.val_image_dir, args.val_scene_start, args.val_scene_end)
        val_dataset = WaymoOpenDataset(
            image_dir=args.val_image_dir,
            scene_names=val_scene_names,
            sequence_length=args.sequence_length,
            mode=1,
            views=1,
        )
        if args.pretrain_task == "pseudo_edit":
            validate_dynamic_masks(val_dataset, args.sequence_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        if is_main_process():
            print(
                f"[validation] scenes={len(val_scene_names)} batches_per_eval={args.val_batches}",
                flush=True,
            )

    decay_params, no_decay_params = split_param_groups(scene_flow)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    lr_scheduler = build_cosine_warmup(optimizer, args.warmup_steps, args.max_steps)
    flow_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.shift)
    ema = EMAModel(scene_flow.parameters(), decay=args.ema_decay)
    ema.to(device)
    global_step = load_resume_checkpoint(
        scene_flow,
        ema,
        optimizer,
        lr_scheduler,
        args.resume_path,
        device,
    )

    if world_size > 1:
        scene_flow = DistributedDataParallel(
            scene_flow,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )

    scene_flow.train()
    optimizer.zero_grad(set_to_none=True)
    accum_step = 0
    # Rolling sums for wandb so we report the mean over the last
    # `--wandb_log_every` optimizer steps instead of every individual step.
    wandb_sums: dict[str, float] = {}
    wandb_count = 0
    progress = None
    if is_main_process() and not args.no_tqdm:
        progress = tqdm(
            total=args.max_steps,
            initial=global_step,
            desc="pretrain",
            dynamic_ncols=True,
        )
    try:
        while global_step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(global_step)
            for batch in loader:
                if global_step >= args.max_steps:
                    break

                sync_grad = (accum_step + 1) % max(1, args.grad_accum_steps) == 0
                ddp_context = (
                    scene_flow.no_sync()
                    if isinstance(scene_flow, DistributedDataParallel) and not sync_grad
                    else nullcontext()
                )
                with ddp_context:
                    try:
                        loss, logs = train_step(batch, vggt_model, scene_flow, flow_scheduler, device, args)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        # In DDP, single-rank skip would desync allreduce. Re-raise so
                        # the entire job restarts with smaller batch / accum_steps.
                        if is_distributed():
                            raise
                        print(f"[step {global_step:06d}] CUDA OOM; skipping batch", flush=True)
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        accum_step = 0
                        continue
                    (loss / max(1, args.grad_accum_steps)).backward()
                accum_step += 1

                if not sync_grad:
                    continue

                params = unwrap_ddp(scene_flow).parameters()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.step(unwrap_ddp(scene_flow).parameters())
                accum_step = 0
                global_step += 1

                if is_main_process():
                    lr_now = optimizer.param_groups[0]["lr"]
                    train_metrics = dict(logs)
                    train_metrics["lr"] = float(lr_now)
                    if progress is not None:
                        postfix = {"lr": f"{lr_now:.2e}"}
                        for key, value in logs.items():
                            postfix[key] = f"{float(value):.4f}"
                        progress.set_postfix(postfix, refresh=False)
                    elif global_step % max(1, int(args.log_every)) == 0:
                        metrics_str = " | ".join(f"{key}={value:.4f}" for key, value in logs.items())
                        print(f"[step {global_step:06d}] lr={lr_now:.2e} | {metrics_str}", flush=True)

                    # Accumulate for averaged wandb reporting.
                    for key, value in train_metrics.items():
                        wandb_sums[key] = wandb_sums.get(key, 0.0) + float(value)
                    wandb_count += 1
                    if wandb_run is not None and wandb_count >= max(1, int(args.wandb_log_every)):
                        averaged = {key: value / wandb_count for key, value in wandb_sums.items()}
                        log_wandb(wandb_run, averaged, global_step, "train")
                        wandb_sums = {}
                        wandb_count = 0

                if (
                    val_loader is not None
                    and global_step > 0
                    and global_step % args.val_every == 0
                ):
                    run_validation(
                        val_loader,
                        vggt_model,
                        scene_flow,
                        flow_scheduler,
                        device,
                        args,
                        global_step,
                        log_dir,
                        wandb_run,
                    )

                if global_step > 0 and global_step % args.save_every == 0:
                    if is_distributed():
                        dist.barrier()
                    if is_main_process():
                        save_checkpoint(scene_flow, ema, optimizer, lr_scheduler, global_step, log_dir, args)
                    if is_distributed():
                        dist.barrier()

                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if is_distributed():
        dist.barrier()
    if is_main_process():
        save_checkpoint(scene_flow, ema, optimizer, lr_scheduler, global_step, log_dir, args)
        if wandb_run is not None:
            wandb_run.finish()
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
