from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any
import lpips
import wandb
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from gsplat.rendering import rasterization

from datasets import WaymoEditDataset
from datasets.samplers import VariableLengthDistributedSampler
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.tokens import (
    select_patch_pyramid,
    split_joint_channels,
    split_special_and_patch,
)

'''
NCCL_P2P_DISABLE=1 torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    train_tokenizer.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --log_dir logs/tokenizer_t0_waymo_views1 \
    --processed_root /data/disk2/lyy_dataset/waymo_processed_dggt \
    --transfer_root /data/disk2/lyy_dataset/waymo_transfer \
    --raw_root /data/disk2/lyy_dataset/waymo \
    --asset_root /data/disk2/lyy_dataset/test_transfer/objects_ply_transformed \
    --views 1 \
    --sample_window 20 \
    --min_frames 4 \
    --max_frames 8 \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --num_workers 16 \
    --max_steps 40000 \
    --save_every 1000 \
    --vis_every 500 \
    --log_every 50 \
    --stats_steps 512 \
    --lr 2e-4 \
    --weight_decay 0.05 \
    --warmup_steps 1000 \
    --head_start_step 2000 \
    --render_start_step 4000 \
    --noisy_start_step 8000 \
    --lambda_tok_rec 1.0 \
    --lambda_tok_cos 0.2 \
    --lambda_head_anchor 0.5 \
    --lambda_render_anchor 0.25 \
    --lambda_noisy 0.1 \
    --lambda_lat_stat 0.01 \
    --precision bf16 \
    --seed 0 \
    --wandb \
    --wandb_project dggt-tokenizer \
    --wandb_name tokenizer_rope_f48_bs1_acc8_lr2e-4_views1
'''


def alpha_t(
    t: torch.Tensor,
    t0: torch.Tensor | float,
    alpha: torch.Tensor,
    gamma0: torch.Tensor,
    gamma1: float = 0.1,
) -> torch.Tensor:
    if not torch.is_tensor(t0):
        t0 = torch.tensor(float(t0), dtype=t.dtype, device=t.device)
    sigma = torch.log(torch.tensor(gamma1, dtype=alpha.dtype, device=alpha.device)) / ((gamma0) ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    return (alpha * conf).float()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T0 JointSceneTokenizer pretraining.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Base DGGT checkpoint.")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging on rank 0.")
    parser.add_argument("--wandb_project", type=str, default="dggt-tokenizer")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])

    parser.add_argument("--processed_root", type=str, default=get_default_processed_root())
    parser.add_argument("--transfer_root", type=str, default=get_default_transfer_root())
    parser.add_argument("--raw_root", type=str, default=get_default_raw_root())
    parser.add_argument("--asset_root", type=str, default=get_default_asset_root())
    parser.add_argument("--clean_split_seed", type=int, default=0)
    parser.add_argument("--clean_train_ratio", type=float, default=0.9)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--candidate_path", type=str, default=None)
    parser.add_argument("--views", type=int, default=1, choices=[1, 3])
    parser.add_argument("--sample_window", type=int, default=20)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=60000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--vis_every", type=int, default=1000)
    parser.add_argument("--vis_test_index", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--stats_steps", type=int, default=512)
    parser.add_argument("--feature_stats_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--render_start_step", type=int, default=20000)
    parser.add_argument("--noisy_start_step", type=int, default=40000)
    parser.add_argument("--head_start_step", type=int, default=5000)
    parser.add_argument("--head_warmup_steps", type=int, default=5000)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--lambda_tok_rec", type=float, default=1.0)
    parser.add_argument("--lambda_tok_cos", type=float, default=0.2)
    parser.add_argument("--lambda_head_anchor", type=float, default=0.5)
    parser.add_argument("--lambda_render_anchor", type=float, default=0.25)
    parser.add_argument("--lambda_noisy", type=float, default=0.1)
    parser.add_argument("--lambda_lat_stat", type=float, default=0.01)
    parser.add_argument("--lambda_dynamic_bce", type=float, default=0.05)
    parser.add_argument("--lambda_gs_lifespan", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    return parser


def get_default_processed_root() -> str:
    return "/data/disk2/lyy_dataset/waymo_processed_dggt"


def get_default_transfer_root() -> str:
    return "/data/disk2/lyy_dataset/waymo_transfer"


def get_default_raw_root() -> str:
    return "/data/disk2/lyy_dataset/waymo"


def get_default_asset_root() -> str:
    return "/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed"


class TokenizerTrainWrapper(nn.Module):
    def __init__(self, tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(
        self,
        image_tokens: list[torch.Tensor] | None = None,
        latent: torch.Tensor | None = None,
        patch_grid: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | list[torch.Tensor]:
        if latent is not None:
            return self.tokenizer.decode(latent)
        if image_tokens is None:
            raise ValueError("Either image_tokens or latent must be provided")
        z = self.tokenizer.encode(image_tokens, patch_grid=patch_grid)
        decoded = self.tokenizer.decode(z)
        return z, decoded


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    if torch.cuda.is_available():
        if world_size == 1:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, local_rank, world_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(value)}")
    return value


def extract_levels(model: VGGT) -> tuple[int, ...]:
    levels = tuple(model.gs_head.intermediate_layer_idx)
    if levels != tuple(model.depth_head.intermediate_layer_idx):
        raise ValueError("gs_head and depth_head intermediate levels differ")
    if levels != tuple(model.point_head.intermediate_layer_idx):
        raise ValueError("gs_head and point_head intermediate levels differ")
    if levels != tuple(model.instance_head.intermediate_layer_idx):
        raise ValueError("gs_head and instance_head intermediate levels differ")
    return levels


def load_model_checkpoint(model: VGGT, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=False)


def freeze_model_for_t0(model: VGGT) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.scene_tokenizer.parameters():
        param.requires_grad = True
    model.eval()
    model.scene_tokenizer.train()


def autocast_context(args: argparse.Namespace, device: torch.device):
    enabled = device.type == "cuda" and args.precision == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def infer_patch_grid(images: torch.Tensor, num_patches: int, patch_size: int = 14) -> tuple[int, int]:
    image_h = int(images.shape[-2])
    image_w = int(images.shape[-1])
    patch_h = image_h // patch_size
    patch_w = image_w // patch_size
    if patch_h * patch_w != num_patches:
        raise ValueError(
            f"Image spatial size {(image_h, image_w)} with patch_size={patch_size} "
            f"does not match num_patches={num_patches}"
        )
    return patch_h, patch_w


def tokenizer_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Received an empty batch")

    frame_counts = []
    for sample in batch:
        num_frames = sample.get("num_frames")
        if num_frames is None:
            raise KeyError("Dataset sample is missing `num_frames`; cannot validate variable-length batching")
        if isinstance(num_frames, torch.Tensor):
            frame_counts.append(int(num_frames.item()))
        else:
            frame_counts.append(int(num_frames))

    unique_frame_counts = sorted(set(frame_counts))
    if len(unique_frame_counts) != 1:
        raise ValueError(
            f"All samples inside one batch must share the same num_frames, got {unique_frame_counts}. "
            "Keep DataLoader(batch_size) aligned with VariableLengthDistributedSampler(batch_size)."
        )

    return default_collate(batch)


def build_sparse_level_list(
    total_levels: int,
    levels: tuple[int, ...],
    values: list[torch.Tensor],
) -> list[torch.Tensor | None]:
    sparse_levels: list[torch.Tensor | None] = [None] * int(total_levels)
    for level_idx, value in zip(levels, values):
        sparse_levels[int(level_idx)] = value
    return sparse_levels


def reattach_special_tokens_from_selected(
    selected_template_tokens: list[torch.Tensor],
    patch_start_idx: int,
    patch_tokens: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(selected_template_tokens) != len(patch_tokens):
        raise ValueError(
            f"selected_template_tokens ({len(selected_template_tokens)}) and patch_tokens ({len(patch_tokens)}) must match"
        )

    outputs = []
    for template_tokens, new_patch_tokens in zip(selected_template_tokens, patch_tokens):
        special_tokens, _ = split_special_and_patch(template_tokens, patch_start_idx)
        outputs.append(torch.cat([special_tokens, new_patch_tokens], dim=-2))
    return outputs


def ensure_log_dirs(log_dir: Path) -> dict[str, Path]:
    ckpt_dir = log_dir / "ckpt"
    vis_dir = log_dir / "vis"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    return {"ckpt": ckpt_dir, "vis": vis_dir}


def tensor_to_pil_rgb(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image_u8 = image.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).contiguous()
    height, width = image_u8.shape[:2]
    return Image.frombytes("RGB", (width, height), bytes(image_u8.view(torch.uint8).untyped_storage()))


def make_grid(images: list[Image.Image], cols: int) -> Image.Image:
    width, height = images[0].size
    cols = max(1, min(cols, len(images)))
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = Image.new("RGB", (cols * width, rows * height))
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        canvas.paste(image, (col * width, row * height))
    return canvas


def build_triplet_grid(
    teacher: torch.Tensor,
    student: torch.Tensor,
    max_frames: int = 4,
) -> Image.Image:
    teacher = teacher.detach().cpu().float().clamp(0.0, 1.0)
    student = student.detach().cpu().float().clamp(0.0, 1.0)
    diff = (student - teacher).abs().clamp(0.0, 1.0)
    num_frames = min(max_frames, teacher.shape[0])
    pil_images: list[Image.Image] = []
    for frame_idx in range(num_frames):
        pil_images.extend(
            [
                tensor_to_pil_rgb(teacher[frame_idx]),
                tensor_to_pil_rgb(student[frame_idx]),
                tensor_to_pil_rgb(diff[frame_idx]),
            ]
        )
    return make_grid(pil_images, cols=3)


def save_triplet_grid(
    teacher: torch.Tensor,
    student: torch.Tensor,
    path: Path,
    max_frames: int = 4,
) -> Image.Image:
    grid = build_triplet_grid(teacher, student, max_frames=max_frames)
    grid.save(path)
    return grid


def reduce_per_sample(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        return values.unsqueeze(0)
    if values.ndim == 1:
        return values
    return values.reshape(values.shape[0], -1).mean(dim=1)


def normalized_token_reconstruction_loss(
    pred_tokens: list[torch.Tensor],
    target_tokens: list[torch.Tensor],
    std_stats: torch.Tensor,
) -> torch.Tensor:
    per_level_losses = []
    for level_idx, (pred, target) in enumerate(zip(pred_tokens, target_tokens)):
        pred_float = pred.float()
        target_float = target.float()
        std = std_stats[level_idx].to(device=pred.device, dtype=torch.float32).view(1, 1, 1, -1)
        per_element = ((pred_float - target_float) / (std + 1e-6)) ** 2
        per_level_losses.append(reduce_per_sample(per_element))
    return torch.stack(per_level_losses, dim=0).mean(dim=0).mean()


def token_cosine_loss(pred_tokens: list[torch.Tensor], target_tokens: list[torch.Tensor]) -> torch.Tensor:
    per_level_losses = []
    for pred, target in zip(pred_tokens, target_tokens):
        cosine = F.cosine_similarity(pred.float(), target.float(), dim=-1)
        per_level_losses.append(reduce_per_sample(1.0 - cosine))
    return torch.stack(per_level_losses, dim=0).mean(dim=0).mean()


def latent_stat_loss(z: torch.Tensor) -> torch.Tensor:
    z_float = z.float()
    z_mean = z_float.mean(dim=(1, 2)).abs().mean(dim=-1)
    z_std = z_float.std(dim=(1, 2), unbiased=False)
    per_sample = z_mean + (z_std - 1.0).abs().mean(dim=-1)
    return per_sample.mean()


def sample_noisy_latent(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(z)
    t = torch.rand((z.shape[0], 1, 1, 1), device=z.device, dtype=z.dtype)
    alpha = torch.cos(t * math.pi * 0.5)
    sigma = torch.sin(t * math.pi * 0.5)
    return alpha * z + sigma * noise, alpha, sigma


def scheduled_weight(step: int, start_step: int, warmup_steps: int, base_weight: float) -> float:
    if base_weight == 0.0 or step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return base_weight
    progress = min(float(step - start_step + 1) / float(warmup_steps), 1.0)
    return base_weight * progress


def normalized_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    flat_target = target.reshape(target.shape[0], -1)
    scale = flat_target.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    scale = scale.view(target.shape[0], *([1] * (target.ndim - 1)))
    per_element = F.smooth_l1_loss(pred / scale, target / scale, beta=delta, reduction="none")
    return reduce_per_sample(per_element).mean()


def dynamic_mask_bce_loss(dynamic_logits: torch.Tensor, dynamic_mask: torch.Tensor) -> torch.Tensor:
    logits = dynamic_logits.float().squeeze(-1)
    if dynamic_mask.ndim != 5:
        raise ValueError(f"Expected dynamic_mask to have shape [B,S,C,H,W], got {tuple(dynamic_mask.shape)}")
    target = dynamic_mask[:, :, 0].float()
    per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return reduce_per_sample(per_element).mean()


def compute_lifespan_loss(gs_conf: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gamma = gs_conf.float()
    per_element = torch.abs(1.0 / (gamma + eps))
    return reduce_per_sample(per_element).mean()


def split_gs_with_mask(gs_map: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rgbs = gs_map[..., :3][mask].reshape(-1, 3)
    opacity = gs_map[..., 3:4][mask].reshape(-1)
    scales = gs_map[..., 4:7][mask].reshape(-1, 3)
    rotation = gs_map[..., 7:11][mask].reshape(-1, 4)
    return rgbs, opacity, scales, rotation


def concat_tensor_lists(list_1: list[torch.Tensor], list_2: list[torch.Tensor]) -> list[torch.Tensor]:
    if len(list_1) != len(list_2):
        raise ValueError(f"List lengths must match, got {len(list_1)} and {len(list_2)}")
    if list_2[0].numel() == 0:
        return list_1
    return [torch.cat((item_1, item_2), dim=0) for item_1, item_2 in zip(list_1, list_2)]


def unproject_depth_map_to_point_map_torch(
    depth_map: torch.Tensor,
    extrinsics_cam: torch.Tensor,
    intrinsics_cam: torch.Tensor,
) -> torch.Tensor:
    if depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    bsz, seq_len, height, width = depth_map.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth_map.device, dtype=depth_map.dtype),
        torch.arange(width, device=depth_map.device, dtype=depth_map.dtype),
        indexing="ij",
    )
    xx = xx.view(1, 1, height, width)
    yy = yy.view(1, 1, height, width)

    fx = intrinsics_cam[..., 0, 0].view(bsz, seq_len, 1, 1)
    fy = intrinsics_cam[..., 1, 1].view(bsz, seq_len, 1, 1)
    cx = intrinsics_cam[..., 0, 2].view(bsz, seq_len, 1, 1)
    cy = intrinsics_cam[..., 1, 2].view(bsz, seq_len, 1, 1)

    x_cam = (xx - cx) * depth_map / fx
    y_cam = (yy - cy) * depth_map / fy
    cam_points = torch.stack([x_cam, y_cam, depth_map], dim=-1)

    rotation = extrinsics_cam[..., :3, :3]
    translation = extrinsics_cam[..., :3, 3]
    cam_points_centered = cam_points - translation.view(bsz, seq_len, 1, 1, 3)
    rotation_t = rotation.transpose(-1, -2).view(bsz, seq_len, 1, 1, 3, 3)
    world_points = torch.matmul(
        rotation_t,
        cam_points_centered.unsqueeze(-1),
    ).squeeze(-1)
    return world_points


def build_extrinsics_4x4(extrinsics_3x4: torch.Tensor) -> torch.Tensor:
    bsz, seq_len = extrinsics_3x4.shape[:2]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsics_3x4.device, dtype=extrinsics_3x4.dtype)
    bottom = bottom.view(1, 1, 1, 4).expand(bsz, seq_len, -1, -1)
    return torch.cat([extrinsics_3x4, bottom], dim=-2)


def render_background(
    sky_model: nn.Module,
    images: torch.Tensor,
    extrinsic_4x4: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    return sky_model(images, extrinsic_4x4[0], intrinsic[0])


def _render_scene_single_from_outputs(
    model: VGGT,
    images: torch.Tensor,
    sky_mask: torch.Tensor,
    timestamps: torch.Tensor,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
) -> torch.Tensor:
    if images.shape[0] != 1:
        raise ValueError(f"Single-scene renderer expects batch dimension 1, got {images.shape[0]}")

    _, _, _, height, width = images.shape
    extrinsics_3x4, intrinsics = pose_encoding_to_extri_intri(pose_enc, (height, width))
    extrinsic_4x4 = build_extrinsics_4x4(extrinsics_3x4)
    point_map = unproject_depth_map_to_point_map_torch(depth.float(), extrinsics_3x4.float(), intrinsics.float())

    sky_mask_hw = sky_mask.permute(0, 1, 3, 4, 2)
    non_sky_mask = (sky_mask_hw == 0).any(dim=-1)
    dynamic_prob = torch.sigmoid(dynamic_conf.float().squeeze(-1))

    static_mask = torch.ones_like(non_sky_mask, dtype=torch.bool)
    static_points = point_map[static_mask].reshape(-1, 3)
    static_rgbs, static_opacity, static_scales, static_rotations = split_gs_with_mask(gs_map.float(), static_mask)
    static_opacity = static_opacity * (1.0 - dynamic_prob[static_mask])
    static_gs_conf = gs_conf.float()[static_mask].reshape(-1)
    static_image_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
    gs_timestamps = timestamps[:, static_image_idx].reshape(-1) if static_image_idx.numel() > 0 else timestamps.new_zeros((0,))

    bg_render = render_background(model.sky_model, images.float(), extrinsic_4x4.float(), intrinsics.float())

    chunked_renders = []
    chunked_alphas = []
    seq_len = images.shape[1]

    for image_idx in range(seq_len):
        mask_i = non_sky_mask[:, image_idx]
        dynamic_points = point_map[:, image_idx][mask_i].reshape(-1, 3)
        dynamic_rgbs, dynamic_opacity, dynamic_scales, dynamic_rotations = split_gs_with_mask(gs_map[:, image_idx].float(), mask_i)
        dynamic_opacity = dynamic_opacity * dynamic_prob[:, image_idx][mask_i]

        opacity_t = alpha_t(gs_timestamps, timestamps[0, image_idx], static_opacity, gamma0=static_gs_conf)
        world_points, rgbs, opacity, scales, rotation = concat_tensor_lists(
            [static_points, static_rgbs, opacity_t, static_scales, static_rotations],
            [dynamic_points, dynamic_rgbs, dynamic_opacity, dynamic_scales, dynamic_rotations],
        )

        if world_points.numel() == 0:
            empty_render = torch.zeros((1, height, width, 3), device=images.device, dtype=images.dtype)
            empty_alpha = torch.zeros((1, height, width, 1), device=images.device, dtype=images.dtype)
            chunked_renders.append(empty_render)
            chunked_alphas.append(empty_alpha)
            continue

        render_chunk, alpha_chunk, _ = rasterization(
            means=world_points.float(),
            quats=rotation.float(),
            scales=scales.float(),
            opacities=opacity.float(),
            colors=rgbs.float(),
            viewmats=extrinsic_4x4[0, image_idx : image_idx + 1].float(),
            Ks=intrinsics[0, image_idx : image_idx + 1].float(),
            width=width,
            height=height,
        )
        chunked_renders.append(render_chunk)
        chunked_alphas.append(alpha_chunk)

    renders = torch.cat(chunked_renders, dim=0)
    alphas = torch.cat(chunked_alphas, dim=0)
    composed = alphas * renders + (1.0 - alphas) * bg_render
    return composed.permute(0, 3, 1, 2).contiguous()


def render_scene_from_outputs(
    model: VGGT,
    images: torch.Tensor,
    sky_mask: torch.Tensor,
    timestamps: torch.Tensor,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
) -> torch.Tensor:
    batch_renders = []
    for batch_idx in range(images.shape[0]):
        batch_renders.append(
            _render_scene_single_from_outputs(
                model,
                images[batch_idx : batch_idx + 1],
                sky_mask[batch_idx : batch_idx + 1],
                timestamps[batch_idx : batch_idx + 1],
                pose_enc[batch_idx : batch_idx + 1],
                depth[batch_idx : batch_idx + 1],
                gs_map[batch_idx : batch_idx + 1],
                gs_conf[batch_idx : batch_idx + 1],
                dynamic_conf[batch_idx : batch_idx + 1],
            )
        )
    return torch.stack(batch_renders, dim=0)


def compute_feature_stats(
    model: VGGT,
    dataset: WaymoEditDataset,
    args: argparse.Namespace,
    device: torch.device,
    levels: tuple[int, ...],
    stats_path: Path,
) -> dict[str, torch.Tensor]:
    sum_acc = torch.zeros((len(levels), 3072), dtype=torch.float64, device=device)
    sum_sq_acc = torch.zeros_like(sum_acc)
    count_acc = torch.zeros((len(levels),), dtype=torch.float64, device=device)

    model.eval()
    max_items = min(len(dataset), args.stats_steps)
    rank = get_rank()
    world_size = dist.get_world_size() if is_distributed() else 1
    local_indices = list(range(rank, max_items, world_size))
    stats_pbar = None
    if is_main_process():
        print(
            f"[feature_stats] start: total_items={max_items} world_size={world_size} "
            f"items_on_rank0={len(local_indices)}",
            flush=True,
        )
        stats_pbar = tqdm(
            total=len(local_indices),
            desc="feature_stats(rank0)",
            dynamic_ncols=True,
            leave=True,
        )

    for local_step, sample_idx in enumerate(local_indices, start=1):
        sample = dataset[(sample_idx, args.max_frames)]
        images = unwrap_tensor(sample["images_clean"]).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "extract_scene_tokens"):
                agg_all, image_all, dino_all, _, patch_start_idx = model.extract_scene_tokens(images)
            else:
                agg_all, image_all, dino_all, _, patch_start_idx = model.aggregator(images)
            del agg_all, dino_all
            image_patch = select_patch_pyramid(image_all, levels, patch_start_idx)
            for level_idx, tokens in enumerate(image_patch):
                flat = tokens.reshape(-1, tokens.shape[-1]).double()
                sum_acc[level_idx] += flat.sum(dim=0)
                sum_sq_acc[level_idx] += (flat * flat).sum(dim=0)
                count_acc[level_idx] += flat.shape[0]
        if stats_pbar is not None:
            stats_pbar.update(1)

    if is_distributed():
        dist.all_reduce(sum_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_sq_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_acc, op=dist.ReduceOp.SUM)
    if stats_pbar is not None:
        stats_pbar.close()

    mean = sum_acc / count_acc.view(-1, 1).clamp_min(1.0)
    var = sum_sq_acc / count_acc.view(-1, 1).clamp_min(1.0) - mean * mean
    std = var.clamp_min(1e-6).sqrt()
    payload = {
        "levels": torch.tensor(levels, dtype=torch.long),
        "mean": mean.cpu().float(),
        "std": std.cpu().float(),
    }
    if is_main_process():
        torch.save(payload, stats_path)
        print(f"[feature_stats] saved to {stats_path}", flush=True)
    return payload


def get_feature_stats(
    model: VGGT,
    dataset: WaymoEditDataset,
    args: argparse.Namespace,
    device: torch.device,
    levels: tuple[int, ...],
    stats_path: Path,
) -> dict[str, torch.Tensor]:
    if is_main_process() and not stats_path.is_file():
        compute_feature_stats(model, dataset, args, device, levels, stats_path)
    elif not stats_path.is_file():
        compute_feature_stats(model, dataset, args, device, levels, stats_path)
    if is_distributed():
        dist.barrier()
    return torch.load(stats_path, map_location="cpu")


def get_teacher_outputs(
    model: VGGT,
    images: torch.Tensor,
    levels: tuple[int, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad(), autocast_context(args, device):
        if hasattr(model, "extract_scene_tokens"):
            agg_all, image_all, dino_all, _, patch_start_idx = model.extract_scene_tokens(images)
        else:
            agg_all, image_all, dino_all, _, patch_start_idx = model.aggregator(images)
        num_levels = len(image_all)
        image_patch = select_patch_pyramid(image_all, levels, patch_start_idx)
        image_levels = [image_all[level_idx] for level_idx in levels]
        pose_enc = model.camera_head(agg_all)[-1]
        gs_map, gs_conf = model.gs_head(image_all, images, patch_start_idx)
        depth, depth_conf = model.depth_head(agg_all, images, patch_start_idx)
        dynamic_conf, _ = model.instance_head(dino_all, images, patch_start_idx)
        del agg_all, image_all, dino_all
    return {
        "num_levels": num_levels,
        "image_levels": image_levels,
        "patch_start_idx": patch_start_idx,
        "image_patch": image_patch,
        "pose_enc": pose_enc,
        "gs_map": gs_map,
        "gs_conf": gs_conf,
        "depth": depth,
        "depth_conf": depth_conf,
        "dynamic_conf": dynamic_conf,
    }


def build_student_outputs(
    model: VGGT,
    tokenizer_runner: nn.Module,
    images: torch.Tensor,
    levels: tuple[int, ...],
    teacher: dict[str, Any],
    autocast_enabled,
    global_step: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    patch_grid = infer_patch_grid(images, teacher["image_patch"][0].shape[2])
    with autocast_enabled:
        z, decoded_patch = tokenizer_runner(image_tokens=teacher["image_patch"], patch_grid=patch_grid)
        image_hat_4 = reattach_special_tokens_from_selected(
            teacher["image_levels"],
            teacher["patch_start_idx"],
            decoded_patch,
        )
        dino_hat_4 = []
        agg_hat_4 = []
        for joint_tokens in image_hat_4:
            dino_hat, frame_hat, global_hat = split_joint_channels(joint_tokens)
            dino_hat_4.append(dino_hat)
            agg_hat_4.append(torch.cat([frame_hat, global_hat], dim=-1))

        image_hat_all = build_sparse_level_list(teacher["num_levels"], levels, image_hat_4)
        dino_hat_all = build_sparse_level_list(teacher["num_levels"], levels, dino_hat_4)
        agg_hat_all = build_sparse_level_list(teacher["num_levels"], levels, agg_hat_4)

        gs_map_hat, gs_conf_hat = model.gs_head(image_hat_all, images, teacher["patch_start_idx"])
        depth_hat, depth_conf_hat = model.depth_head(agg_hat_all, images, teacher["patch_start_idx"])
        dynamic_hat, _ = model.instance_head(dino_hat_all, images, teacher["patch_start_idx"])
        del image_hat_all, dino_hat_all, agg_hat_all

        decoded_noisy = None
        if global_step >= args.noisy_start_step and args.lambda_noisy > 0.0:
            z_noisy, _, _ = sample_noisy_latent(z)
            decoded_noisy = tokenizer_runner(latent=z_noisy)

    return {
        "z": z,
        "decoded_patch": decoded_patch,
        "decoded_noisy": decoded_noisy,
        "gs_map": gs_map_hat,
        "gs_conf": gs_conf_hat,
        "depth": depth_hat,
        "depth_conf": depth_conf_hat,
        "dynamic_conf": dynamic_hat,
    }


def compute_losses(
    model: VGGT,
    sample: dict[str, Any],
    args: argparse.Namespace,
    feature_stats: dict[str, torch.Tensor],
    teacher: dict[str, Any],
    student: dict[str, Any],
    global_step: int,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    std_stats = feature_stats["std"]
    losses: dict[str, torch.Tensor] = {}
    losses["tok_rec"] = normalized_token_reconstruction_loss(student["decoded_patch"], teacher["image_patch"], std_stats)
    losses["tok_cos"] = token_cosine_loss(student["decoded_patch"], teacher["image_patch"])

    gs_anchor = normalized_huber_loss(student["gs_map"], teacher["gs_map"]) + 0.1 * normalized_huber_loss(
        student["gs_conf"], teacher["gs_conf"]
    )
    geom_anchor = normalized_huber_loss(student["depth"], teacher["depth"]) + 0.1 * normalized_huber_loss(
        student["depth_conf"], teacher["depth_conf"]
    )
    dyn_anchor = normalized_huber_loss(student["dynamic_conf"], teacher["dynamic_conf"])
    losses["gs_anchor"] = gs_anchor
    losses["geom_anchor"] = geom_anchor
    losses["dyn_anchor"] = dyn_anchor
    losses["head_anchor"] = gs_anchor + geom_anchor + dyn_anchor
    losses["dynamic_bce"] = student["z"].new_tensor(0.0)
    if "dynamic_mask" in sample:
        dynamic_mask = unwrap_tensor(sample["dynamic_mask"]).to(student["z"].device)
        losses["dynamic_bce"] = dynamic_mask_bce_loss(student["dynamic_conf"], dynamic_mask)
    losses["gs_lifespan"] = compute_lifespan_loss(student["gs_conf"])

    losses["noisy"] = student["z"].new_tensor(0.0)
    if student["decoded_noisy"] is not None:
        losses["noisy"] = normalized_token_reconstruction_loss(student["decoded_noisy"], teacher["image_patch"], std_stats)
    losses["lat_stat"] = latent_stat_loss(student["z"])

    render_ref = None
    render_hat = None
    losses["render_anchor"] = student["z"].new_tensor(0.0)
    losses["render_lpips"] = student["z"].new_tensor(0.0)

    if global_step >= args.render_start_step:
        if model.lpips_loss_fn is None:
            raise RuntimeError("Render anchor is active but LPIPS has not been initialized")
        sky_mask = unwrap_tensor(sample.get("sky_mask", sample["masks"])).to(student["z"].device)
        timestamps = unwrap_tensor(sample["timestamps"]).to(student["z"].device)
        render_ref = render_scene_from_outputs(
            model,
            unwrap_tensor(sample["images_clean"]).to(student["z"].device),
            sky_mask,
            timestamps,
            teacher["pose_enc"].float(),
            teacher["depth"].float(),
            teacher["gs_map"].float(),
            teacher["gs_conf"].float(),
            teacher["dynamic_conf"].float(),
        )
        render_hat = render_scene_from_outputs(
            model,
            unwrap_tensor(sample["images_clean"]).to(student["z"].device),
            sky_mask,
            timestamps,
            teacher["pose_enc"].float(),
            student["depth"].float(),
            student["gs_map"].float(),
            student["gs_conf"].float(),
            student["dynamic_conf"].float(),
        )
        render_mse_per_sample = reduce_per_sample((render_hat - render_ref) ** 2)
        render_mse = render_mse_per_sample.mean()
        render_hat_lpips = render_hat.flatten(0, 1)
        render_ref_lpips = render_ref.flatten(0, 1)
        render_lpips_per_frame = model.lpips_loss_fn(render_hat_lpips * 2.0 - 1.0, render_ref_lpips * 2.0 - 1.0)
        render_lpips_per_sample = reduce_per_sample(
            render_lpips_per_frame.reshape(render_hat.shape[0], render_hat.shape[1], -1)
        )
        render_lpips = render_lpips_per_sample.mean()
        losses["render_anchor"] = render_mse + 0.1 * render_lpips
        losses["render_lpips"] = render_lpips

    head_weight = scheduled_weight(
        global_step,
        args.head_start_step,
        args.head_warmup_steps,
        args.lambda_head_anchor,
    )
    dynamic_bce_weight = scheduled_weight(
        global_step,
        args.head_start_step,
        args.head_warmup_steps,
        args.lambda_dynamic_bce,
    )
    gs_lifespan_weight = scheduled_weight(
        global_step,
        args.head_start_step,
        args.head_warmup_steps,
        args.lambda_gs_lifespan,
    )
    noisy_weight = scheduled_weight(
        global_step,
        args.noisy_start_step,
        1,
        args.lambda_noisy,
    )
    render_weight = scheduled_weight(
        global_step,
        args.render_start_step,
        1,
        args.lambda_render_anchor,
    )

    total = (
        args.lambda_tok_rec * losses["tok_rec"]
        + args.lambda_tok_cos * losses["tok_cos"]
        + head_weight * losses["head_anchor"]
        + dynamic_bce_weight * losses["dynamic_bce"]
        + gs_lifespan_weight * losses["gs_lifespan"]
        + render_weight * losses["render_anchor"]
        + noisy_weight * losses["noisy"]
        + args.lambda_lat_stat * losses["lat_stat"]
    )

    scalar_logs = {
        "loss": float(total.detach().item()),
        "tok_rec": float(losses["tok_rec"].detach().item()),
        "tok_cos": float(losses["tok_cos"].detach().item()),
        "gs_anchor": float(losses["gs_anchor"].detach().item()),
        "geom_anchor": float(losses["geom_anchor"].detach().item()),
        "dyn_anchor": float(losses["dyn_anchor"].detach().item()),
        "dynamic_bce": float(losses["dynamic_bce"].detach().item()),
        "gs_lifespan": float(losses["gs_lifespan"].detach().item()),
        "render_anchor": float(losses["render_anchor"].detach().item()),
        "render_lpips": float(losses["render_lpips"].detach().item()),
        "noisy": float(losses["noisy"].detach().item()),
        "lat_stat": float(losses["lat_stat"].detach().item()),
        "latent_mean": float(student["z"].mean().detach().item()),
        "latent_std": float(student["z"].std(unbiased=False).detach().item()),
        "head_weight": float(head_weight),
        "dynamic_bce_weight": float(dynamic_bce_weight),
        "gs_lifespan_weight": float(gs_lifespan_weight),
        "render_weight": float(render_weight),
        "noisy_weight": float(noisy_weight),
    }

    aux = {}
    if render_ref is not None and render_hat is not None:
        aux["render_ref"] = render_ref.detach().cpu()
        aux["render_hat"] = render_hat.detach().cpu()
        mse_per_sample = reduce_per_sample((render_hat.detach() - render_ref.detach()) ** 2)
        psnr_per_sample = -10.0 * torch.log10(mse_per_sample.clamp_min(1e-8))
        scalar_logs["render_psnr"] = float(psnr_per_sample.mean().item())
    return total, scalar_logs, aux


def save_checkpoint(
    tokenizer_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    global_step: int,
    args: argparse.Namespace,
    feature_stats_path: Path,
    out_path: Path,
) -> None:
    state_dict = tokenizer_module.state_dict()
    payload = {
        "scene_tokenizer": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "args": vars(args),
        "feature_stats_path": str(feature_stats_path),
    }
    torch.save(payload, out_path)


def init_wandb_run(args: argparse.Namespace, log_dir: Path):
    if not args.wandb or not is_main_process():
        return None
    if wandb is None:
        raise ModuleNotFoundError(
            "train_tokenizer.py was launched with --wandb but the `wandb` package is not installed."
        )

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_name,
        "dir": str(log_dir),
        "config": vars(args),
        "mode": args.wandb_mode,
    }
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"
    return wandb.init(**init_kwargs)


def log_wandb_scalars(wandb_run, metrics: dict[str, float], step: int, prefix: str = "train") -> None:
    if wandb_run is None:
        return
    payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
    wandb_run.log(payload, step=step)


def log_wandb_visual(
    wandb_run,
    image: Image.Image,
    step: int,
    num_frames: int,
    *,
    prefix: str = "train",
    sample_index: int | None = None,
) -> None:
    if wandb_run is None:
        return
    caption = f"step={step}, num_frames={num_frames}"
    if sample_index is not None:
        caption += f", sample_index={sample_index}"
    wandb_run.log(
        {
            f"{prefix}/render_triplet": wandb.Image(
                image,
                caption=caption,
            )
        },
        step=step,
    )


def run_visualization_eval(
    model: VGGT,
    tokenizer_runner: nn.Module,
    sample: dict[str, Any],
    args: argparse.Namespace,
    feature_stats: dict[str, torch.Tensor],
    levels: tuple[int, ...],
    global_step: int,
    device: torch.device,
) -> tuple[Image.Image | None, dict[str, float]]:
    images = unwrap_tensor(sample["images_clean"]).to(device)
    was_training = tokenizer_runner.training
    tokenizer_runner.eval()
    teacher = get_teacher_outputs(model, images, levels, args, device)
    student = build_student_outputs(
        model,
        tokenizer_runner,
        images,
        levels,
        teacher,
        autocast_context(args, device),
        global_step,
        args,
    )
    if was_training:
        tokenizer_runner.train()
    _, scalar_logs, aux = compute_losses(model, sample, args, feature_stats, teacher, student, global_step)
    if not aux:
        return None, scalar_logs
    grid = build_triplet_grid(aux["render_ref"][0], aux["render_hat"][0])
    return grid, scalar_logs


def build_dataset(args: argparse.Namespace, split: str) -> WaymoEditDataset:
    return WaymoEditDataset(
        processed_root=args.processed_root,
        transfer_root=args.transfer_root,
        raw_root=args.raw_root,
        asset_root=args.asset_root,
        split=split,
        manifest_path=args.manifest_path,
        candidate_path=args.candidate_path,
        sequence_length=args.max_frames,
        mode=1,
        views=args.views,
        sample_window=args.sample_window,
        clean_only=True,
        clean_split_seed=args.clean_split_seed,
        clean_train_ratio=args.clean_train_ratio,
    )


def main() -> None:
    args = build_argparser().parse_args()

    device, local_rank, world_size = setup_distributed(args)
    seed_everything(args.seed + get_rank())

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    if args.min_frames < 1 or args.max_frames < args.min_frames:
        raise ValueError("Invalid frame range")

    log_dir = Path(args.log_dir)
    feature_stats_path = Path(args.feature_stats_path) if args.feature_stats_path else log_dir / "feature_stats.pt"
    dirs = ensure_log_dirs(log_dir)
    wandb_run = init_wandb_run(args, log_dir)
    if is_main_process():
        with (log_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)

    train_dataset = build_dataset(args, split="training")
    test_dataset = build_dataset(args, split="validation")
    if is_main_process() and hasattr(train_dataset, "clean_sample_stats"):
        stats = train_dataset.clean_sample_stats
        print(
            f"[dataset] clean={stats.get('clean_total', 0)} edited={stats.get('edited_total', 0)} "
            f"train={stats.get('train_total', 0)} test={stats.get('val_total', 0)} "
            f"train_active={len(train_dataset)} test_active={len(test_dataset)}",
            flush=True,
        )
    if world_size > 1:
        sampler = VariableLengthDistributedSampler(
            train_dataset,
            min_num_frames=args.min_frames,
            max_num_frames=args.max_frames,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
    else:
        sampler = VariableLengthDistributedSampler(
            train_dataset,
            min_num_frames=args.min_frames,
            max_num_frames=args.max_frames,
            batch_size=args.batch_size,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=args.seed,
        )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=tokenizer_collate_fn,
    )

    if lpips is None:
        raise ModuleNotFoundError(
            "train_tokenizer.py requires the `lpips` package at runtime. "
            "Install it in the current environment before launching training."
        )
    if rasterization is None:
        raise ModuleNotFoundError(
            "train_tokenizer.py requires the `gsplat` package at runtime. "
            "Install it in the current environment before launching training."
        )

    model = VGGT().to(device)
    if is_main_process():
        print(f"[init] loading checkpoint: {args.ckpt_path}", flush=True)
    load_model_checkpoint(model, args.ckpt_path)
    if model.sky_model is None:
        raise ModuleNotFoundError(
            "VGGT.sky_model is unavailable in the current environment. "
            "Install the dependencies required by dggt.models.sky (for example `open3d` and `gsplat`)."
        )
    model.lpips_loss_fn = None
    if args.lambda_render_anchor > 0.0 and args.render_start_step < args.max_steps:
        model.lpips_loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    freeze_model_for_t0(model)

    levels = extract_levels(model)
    if is_main_process():
        print(f"[init] tokenizer levels={levels}", flush=True)
    feature_stats = get_feature_stats(model, train_dataset, args, device, levels, feature_stats_path)
    if is_main_process():
        print("[init] feature stats ready", flush=True)

    if is_main_process() and len(test_dataset) > 0:
        vis_index = int(args.vis_test_index) % len(test_dataset)
        print(
            f"[vis] validation visualization starts at sample index={vis_index} and advances by 1 each vis step",
            flush=True,
        )

    tokenizer_runner: nn.Module = TokenizerTrainWrapper(model.scene_tokenizer).to(device)
    vis_tokenizer_runner = TokenizerTrainWrapper(model.scene_tokenizer).to(device)
    tokenizer_runner.train()
    if world_size > 1:
        tokenizer_runner = DDP(tokenizer_runner, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = AdamW(model.scene_tokenizer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        warmup = min((step + 1) / max(args.warmup_steps, 1), 1.0)
        progress = min(step / max(args.max_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return warmup * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    global_step = 0

    if args.resume_path:
        resume_payload = torch.load(args.resume_path, map_location="cpu")
        state_dict = resume_payload.get("scene_tokenizer", resume_payload)
        model.scene_tokenizer.load_state_dict(state_dict, strict=True)
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        if "scheduler" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler"])
        global_step = int(resume_payload.get("global_step", 0))

    try:
        start_time = time.time()
        epoch = 0
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        accum_log_sums: dict[str, float] = {}
        accum_num_frames = 0.0
        accum_local_batch = 0.0
        train_pbar = None
        if is_main_process():
            train_pbar = tqdm(
                total=args.max_steps,
                initial=global_step,
                desc="train",
                dynamic_ncols=True,
                leave=True,
            )
        while global_step < args.max_steps:
            sampler.set_epoch(epoch)
            num_batches = len(dataloader)
            for batch_idx, sample in enumerate(dataloader):
                if global_step >= args.max_steps:
                    break

                images = unwrap_tensor(sample["images_clean"]).to(device)
                local_batch_size = int(images.shape[0])
                num_frames = images.shape[1]

                accum_count += 1
                is_last_microbatch = batch_idx + 1 == num_batches
                should_step = accum_count >= args.grad_accum_steps or is_last_microbatch

                sync_context = nullcontext()
                if world_size > 1 and not should_step:
                    sync_context = tokenizer_runner.no_sync()

                with sync_context:
                    teacher = get_teacher_outputs(model, images, levels, args, device)
                    student = build_student_outputs(
                        model,
                        tokenizer_runner,
                        images,
                        levels,
                        teacher,
                        autocast_context(args, device),
                        global_step,
                        args,
                    )
                    total_loss, scalar_logs, aux = compute_losses(
                        model,
                        sample,
                        args,
                        feature_stats,
                        teacher,
                        student,
                        global_step,
                    )
                    del aux
                    (total_loss / float(args.grad_accum_steps)).backward()

                for key, value in scalar_logs.items():
                    accum_log_sums[key] = accum_log_sums.get(key, 0.0) + float(value)
                accum_num_frames += float(num_frames)
                accum_local_batch += float(local_batch_size)

                if not should_step:
                    continue

                if args.grad_clip_norm > 0.0:
                    clip_grad_norm_(model.scene_tokenizer.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step_logs = {key: value / float(accum_count) for key, value in accum_log_sums.items()}
                step_num_frames = accum_num_frames / float(accum_count)
                step_local_batch = accum_local_batch

                if train_pbar is not None:
                    train_pbar.set_postfix(
                        batch=f"{step_local_batch:.0f}",
                        frames=f"{step_num_frames:.1f}",
                        loss=f"{step_logs['loss']:.4g}",
                        tok=f"{step_logs['tok_rec']:.4g}",
                        head=f"{step_logs['gs_anchor'] + step_logs['geom_anchor'] + step_logs['dyn_anchor']:.4g}",
                        dynb=f"{step_logs['dynamic_bce']:.4g}",
                        life=f"{step_logs['gs_lifespan']:.4g}",
                        hwt=f"{step_logs['head_weight']:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

                if is_main_process() and global_step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    log_metrics = dict(step_logs)
                    log_metrics["lr"] = float(scheduler.get_last_lr()[0])
                    log_metrics["batch_size"] = float(step_local_batch)
                    log_metrics["micro_batch_size"] = float(local_batch_size)
                    log_metrics["grad_accum_steps"] = float(accum_count)
                    log_metrics["num_frames"] = float(step_num_frames)
                    log_metrics["elapsed_sec"] = float(elapsed)
                    log_wandb_scalars(wandb_run, log_metrics, global_step)

                should_run_vis = len(test_dataset) > 0 and global_step % args.vis_every == 0
                if should_run_vis and is_distributed():
                    dist.barrier()
                if is_main_process() and should_run_vis:
                    vis_index = (int(args.vis_test_index) + (global_step // args.vis_every)) % len(test_dataset)
                    vis_sample = tokenizer_collate_fn([test_dataset[(vis_index, args.max_frames)]])
                    with torch.no_grad():
                        grid, vis_logs = run_visualization_eval(
                            model,
                            vis_tokenizer_runner,
                            vis_sample,
                            args,
                            feature_stats,
                            levels,
                            global_step,
                            device,
                        )
                    if grid is not None:
                        grid.save(dirs["vis"] / f"validation_step_{global_step:06d}_sample_{vis_index:06d}.png")
                        log_wandb_visual(
                            wandb_run,
                            grid,
                            global_step,
                            int(vis_sample["num_frames"][0].item()),
                            prefix="validation",
                            sample_index=vis_index,
                        )
                    if vis_logs:
                        vis_logs["sample_index"] = float(vis_index)
                        vis_logs["num_frames"] = float(vis_sample["num_frames"][0].item())
                        log_wandb_scalars(
                            wandb_run,
                            vis_logs,
                            global_step,
                            prefix="validation",
                        )
                if should_run_vis and is_distributed():
                    dist.barrier()

                if is_main_process() and global_step > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        model.scene_tokenizer,
                        optimizer,
                        scheduler,
                        global_step,
                        args,
                        feature_stats_path,
                        dirs["ckpt"] / f"scene_tokenizer_step_{global_step:06d}.pt",
                    )

                global_step += 1
                if train_pbar is not None:
                    train_pbar.update(1)

                accum_count = 0
                accum_log_sums = {}
                accum_num_frames = 0.0
                accum_local_batch = 0.0

            epoch += 1

        if is_main_process():
            save_checkpoint(
                model.scene_tokenizer,
                optimizer,
                scheduler,
                global_step,
                args,
                feature_stats_path,
                dirs["ckpt"] / "scene_tokenizer_latest.pt",
            )
    finally:
        if "train_pbar" in locals() and train_pbar is not None:
            train_pbar.close()
        if wandb_run is not None:
            wandb_run.finish()
        if is_distributed():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
