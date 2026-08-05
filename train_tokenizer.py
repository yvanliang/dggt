from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import nullcontext
import json
import math
import os
import random
import sys
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
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from gsplat.rendering import rasterization

from datasets import WaymoEditDataset
from datasets.samplers import VariableLengthDistributedSampler
from dggt.models.vggt import VGGT
from dggt.utils.gaussian_render import composite_gsplat_rgb
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.tokens import (
    select_patch_pyramid,
    split_joint_channels,
    split_special_and_patch,
)

'''
2026-08-01 objective change (affects both stages; old checkpoints are NOT comparable)
-------------------------------------------------------------------------------------
Two defects were found by auditing the encode/decode round trip against the frozen
heads on 90 Waymo trunks (`tools/retest_scene_flow_gaussian_gauge.py`):

  depth_recon / depth_direct            = 1.0307   (CI [1.0208, 1.0421])
  gaussian scale_recon / scale_direct   = 0.8289
  paired GS/depth  (must be 1.0)        = 0.7964   (30/30 scenes below 1)

i.e. reconstruction inflates depth ~3% while shrinking every Gaussian radius ~17%,
so each splat ends up ~20% too small for the depth it sits at.  Downstream this shows
up as a 3% parallax error in the SceneFlow render loss (~11 px at 20 m) and as
undersized splats everywhere.

Root causes, both in this file:

  1. `gs_anchor` normalized all 11 gs_map channels by ONE per-sample std.  That std is
     set by rgb (~0.29) and quats (~0.50), while the three *linear* Gaussian scales are
     ~1e-4.  A scale-only error was therefore divided by a number ~3700x too large; a
     20% scale error contributed ~1.2e7x less loss than it should.  The scale channels
     were, in practice, unsupervised.  Fixed by `gs_channel_group_huber_loss`, which
     normalizes rgb / opacity / scale / quat each by its own std.
  2. `gs_anchor` and `geom_anchor` are independent terms, so nothing constrained their
     RATIO -- which is the only thing the rasterizer cares about geometrically.
     `render_anchor` does not pin it either: stage-B ran with render on from step 0 at
     weight 0.5 for 40k steps and the ratio was still 0.796, because a too-small splat
     with a raised opacity renders almost the same.  Fixed by the new
     `--lambda_gs_scale_sim` paired same-pixel term.

Watch `gs_scale_sim_ratio` in the logs: it is exp() of the audited quantity and must
converge to 1.0.  The v2 objective requires all three geometry losses to stay active.

Launch scripts: `train_tokenizer_two_nodes.sh` (GPU, SSH-orchestrated) and
`train_tokenizer_ppu_dlc.sh` (Aliyun PAI-DLC).  Both derive batch/accum from the GPU
count to hold the global batch at the reference value below.

Stage-A (online training, 2 x 80GB):

NCCL_P2P_DISABLE=1 torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    train_tokenizer.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --log_dir logs/tokenizer_t0_v2_stageA \
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
    --num_workers 12 \
    --max_steps 60000 \
    --save_every 2500 \
    --vis_every 1000 \
    --log_every 50 \
    --stats_steps 2048 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --warmup_steps 2000 \
    --grad_clip_norm 1.0 \
    --head_start_step 2000 \
    --head_warmup_steps 5000 \
    --render_start_step 8000 \
    --noisy_start_step 25000 \
    --decoder_noise_tau 0.8 \
    --decoder_noise_distribution uniform \
    --lambda_tok_rec 1.0 \
    --lambda_tok_cos 0.2 \
    --lambda_head_anchor 0.6 \
    --lambda_gs_scale_sim 0.3 \
    --gs_scale_sim_opacity 0.05 \
    --lambda_render_anchor 0.3 \
    --gt_render_ratio 1.0 \
    --render_dyn_alpha 6.0 \
    --lambda_noisy 0.15 \
    --lambda_lat_stat 0.05 \
    --lambda_dynamic_bce 0.2 \
    --dyn_patch_alpha 6.0 \
    --dyn_pixel_alpha 10.0 \
    --lambda_gs_lifespan 0.01 \
    --lambda_ghost_static 0.0 \
    --precision bf16 \
    --seed 0 \
    --wandb \
    --wandb_project dggt-tokenizer \
    --wandb_name t0_v2_stageA_dz1024_2x80g

Stage-B (flow-cache fine-tuning, 2 x 80GB):

NCCL_P2P_DISABLE=1 torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    train_tokenizer.py \
    --init_tokenizer_path logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --cache_manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --cache_split training \
    --log_dir logs/tokenizer_t0_v2_stageB \
    --stage_b_mix_raw \
    --min_frames 10 \
    --max_frames 10 \
    --batch_size 4 \
    --raw_batch_size 4 \
    --grad_accum_steps 2 \
    --num_workers 4 \
    --prefetch_factor 1 \
    --max_steps 40000 \
    --save_every 2500 \
    --vis_every 1000 \
    --log_every 50 \
    --lr 8e-5 \
    --weight_decay 0.05 \
    --warmup_steps 1000 \
    --grad_clip_norm 1.0 \
    --head_start_step 0 \
    --head_warmup_steps 1 \
    --render_start_step 0 \
    --noisy_start_step 0 \
    --decoder_noise_tau 0.8 \
    --decoder_noise_distribution uniform \
    --lambda_tok_rec 0.5 \
    --lambda_tok_cos 0.1 \
    --lambda_head_anchor 0.8 \
    --lambda_gs_scale_sim 0.5 \
    --gs_scale_sim_opacity 0.05 \
    --lambda_render_anchor 0.5 \
    --gt_render_ratio 1.5 \
    --render_dyn_alpha 8.0 \
    --lambda_noisy 0.2 \
    --lambda_lat_stat 0.05 \
    --lambda_dynamic_bce 0.3 \
    --dyn_patch_alpha 8.0 \
    --dyn_pixel_alpha 12.0 \
    --lambda_gs_lifespan 0.01 \
    --lambda_ghost_static 0.0 \
    --precision bf16 \
    --seed 0 \
    --wandb \
    --wandb_project dggt-tokenizer \
    --wandb_name t0_v2_stageB_cached_dz1024

For efficient cached training, recompress existing gzip flow caches to zstd
with tools/recompress_flow_cache.py and write new caches with
tools/precompute_flow_features.py --save_compression zstd --gzip_level 1.
'''


TOKENIZER_OBJECTIVE_VERSION = "t0_v2"
TOKENIZER_V2_REQUIRED_LOSS_WEIGHTS = (
    "lambda_head_anchor",
    "lambda_gs_scale_sim",
    "lambda_depth_log_bias",
)


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
    parser.add_argument(
        "--init_tokenizer_path",
        type=str,
        default=None,
        help="Load scene_tokenizer weights only, leaving optimizer/scheduler/global_step freshly initialized.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Resume a tokenizer training run, including optimizer, scheduler, and global_step.",
    )
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
    parser.add_argument(
        "--cache_dir",
        action="append",
        type=str,
        default=None,
        help=(
            "Use FlowDGGT .pt cache as tokenizer teacher data. Can be repeated. "
            "Each value may be a cache root, a split directory, or path:mode_a/mode_b/auto."
        ),
    )
    parser.add_argument(
        "--cache_manifest_path",
        type=str,
        default=None,
        help="Merged flow-cache manifest JSONL for cached tokenizer training.",
    )
    parser.add_argument("--cache_split", type=str, default="training")
    parser.add_argument(
        "--stage_b_mix_raw",
        action="store_true",
        help=(
            "Stage-B mixed training: cycle raw, mode_a cache, raw, mode_b cache "
            "microbatches for an approximate 50/25/25 mix."
        ),
    )
    parser.add_argument(
        "--raw_batch_size",
        type=int,
        default=1,
        help="Per-rank raw-data batch size in --stage_b_mix_raw mode.",
    )
    parser.add_argument(
        "--cache_mode_filter",
        type=str,
        default=None,
        help="Optional comma-separated cache mode filter, e.g. mode_a,mode_b.",
    )
    parser.add_argument("--views", type=int, default=1, choices=[1, 3])
    parser.add_argument("--sample_window", type=int, default=20)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="DataLoader batches prefetched per worker. Keep low for GB-scale cache items.",
    )
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory. Disabled by default for cached tokenizer training.",
    )
    parser.add_argument(
        "--mp_sharing_strategy",
        type=str,
        default="file_system",
        choices=("file_system", "file_descriptor"),
        help="Torch multiprocessing tensor sharing strategy for DataLoader workers.",
    )
    parser.add_argument(
        "--no_mmap_plain_cache",
        action="store_true",
        help="Disable mmap=True when reading uncompressed torch cache files.",
    )
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

    parser.add_argument(
        "--decoder_noise_tau",
        type=float,
        default=0.0,
        help=(
            "RAE-style decoder noise strength. During training, decode z+n where "
            "n~N(0,sigma^2 I) and sigma is sampled per sample. Default keeps existing "
            "training behavior; set to 0.8 for the RAE DINOv2-B recipe."
        ),
    )
    parser.add_argument(
        "--decoder_noise_distribution",
        type=str,
        default="half_normal",
        choices=["half_normal", "uniform"],
        help=(
            "How to sample decoder noise sigma. `half_normal` implements the paper "
            "sigma~|N(0,tau^2)|; `uniform` matches the released RAE code's U(0,tau)."
        ),
    )

    parser.add_argument("--lambda_tok_rec", type=float, default=1.0)
    parser.add_argument("--lambda_tok_cos", type=float, default=0.2)
    parser.add_argument("--lambda_head_anchor", type=float, default=0.5)
    parser.add_argument(
        "--lambda_gs_scale_sim",
        type=float,
        default=0.3,
        help=(
            "Paired same-pixel constraint that encode/decode be a similarity: "
            "log(scale_s/scale_t) - log(depth_s/depth_t) -> 0. This v2 loss "
            "must have a strictly positive weight."
        ),
    )
    parser.add_argument(
        "--lambda_depth_log_bias",
        type=float,
        default=0.2,
        help=(
            "Penalize the systematic multiplicative depth offset "
            "|mean(log(d_student/d_teacher))|. Audited at 1.0307 on the shipped "
            "checkpoint. This v2 loss must have a strictly positive weight."
        ),
    )
    parser.add_argument(
        "--gs_scale_sim_opacity",
        type=float,
        default=0.05,
        help="Both student and teacher opacity must exceed this to enter the similarity support.",
    )
    parser.add_argument("--lambda_render_anchor", type=float, default=0.25)
    parser.add_argument("--lambda_noisy", type=float, default=0.1)
    parser.add_argument("--lambda_lat_stat", type=float, default=0.01)
    parser.add_argument("--lambda_dynamic_bce", type=float, default=0.05)
    parser.add_argument("--lambda_gs_lifespan", type=float, default=0.01)
    parser.add_argument("--lambda_ghost_static", type=float, default=0.05)
    parser.add_argument(
        "--dyn_patch_alpha",
        type=float,
        default=1.0,
        help="Per-patch weight multiplier on tok_rec / tok_cos at fully-dynamic patches (>=1.0). 1.0 disables.",
    )
    parser.add_argument(
        "--dyn_pixel_alpha",
        type=float,
        default=1.0,
        help="Per-pixel weight multiplier on dyn_anchor at fully-dynamic pixels (>=1.0). 1.0 falls back to unweighted Huber.",
    )
    parser.add_argument(
        "--gt_render_ratio",
        type=float,
        default=0.0,
        help="Ratio of GT-image render loss relative to teacher-render anchor. 0.0 disables.",
    )
    parser.add_argument(
        "--render_dyn_alpha",
        type=float,
        default=1.0,
        help="Per-pixel weight multiplier on GT render L2 inside dynamic regions (>=1.0). 1.0 disables.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    return parser


def validate_tokenizer_v2_objective_args(args: argparse.Namespace) -> None:
    """Reject configurations that disable any required v2 geometry objective."""

    for field in TOKENIZER_V2_REQUIRED_LOSS_WEIGHTS:
        value = float(getattr(args, field))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{field} must be finite and strictly positive for "
                f"{TOKENIZER_OBJECTIVE_VERSION}, got {value!r}"
            )


def require_tokenizer_v2_checkpoint(payload: Any, *, path: str) -> str:
    """Require explicit v2 metadata or the audited pre-metadata v2 loss signature."""

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{path} has no tokenizer objective metadata; unversioned weight-only "
            "checkpoints are rejected"
        )
    version = payload.get("tokenizer_objective_version")
    if version is not None and version != TOKENIZER_OBJECTIVE_VERSION:
        raise ValueError(
            f"{path} tokenizer_objective_version={version!r}; only "
            f"{TOKENIZER_OBJECTIVE_VERSION!r} is supported"
        )
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise ValueError(
            f"{path} lacks saved v2 objective arguments; tokenizer v1 and ambiguous "
            "checkpoints are rejected"
        )
    for field in TOKENIZER_V2_REQUIRED_LOSS_WEIGHTS:
        try:
            value = float(saved_args[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} lacks a valid v2 objective field {field}") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{path} has {field}={value!r}; tokenizer v1 or a disabled-v2 "
                "objective is rejected"
            )
    return (
        TOKENIZER_OBJECTIVE_VERSION
        if version is not None
        else "t0_v2_inferred_from_positive_geometry_loss_weights"
    )


def get_default_processed_root() -> str:
    return "/data/disk2/lyy_dataset/waymo_processed_dggt"


def get_default_transfer_root() -> str:
    return "/data/disk2/lyy_dataset/waymo_transfer"


def get_default_raw_root() -> str:
    return "/data/disk2/lyy_dataset/waymo"


def get_default_asset_root() -> str:
    return "/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed"


def sample_decoder_noise_augmented_latent(
    z: torch.Tensor,
    tau: float,
    distribution: str = "half_normal",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RAE decoder noise augmentation: z+n, n~N(0,sigma^2 I)."""
    if tau <= 0.0:
        sigma = z.new_zeros((z.shape[0],) + (1,) * (z.ndim - 1))
        return z, sigma

    sigma_shape = (z.shape[0],) + (1,) * (z.ndim - 1)
    if distribution == "half_normal":
        sigma = torch.randn(sigma_shape, device=z.device, dtype=torch.float32).abs().mul_(float(tau))
    elif distribution == "uniform":
        sigma = torch.rand(sigma_shape, device=z.device, dtype=torch.float32).mul_(float(tau))
    else:
        raise ValueError(f"Unsupported decoder_noise_distribution={distribution!r}")

    noise = torch.randn_like(z, dtype=torch.float32) * sigma
    z_noisy = z.float() + noise
    return z_noisy.to(dtype=z.dtype), sigma.to(dtype=z.dtype)


class TokenizerTrainWrapper(nn.Module):
    def __init__(self, tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(
        self,
        image_tokens: list[torch.Tensor] | None = None,
        latent: torch.Tensor | None = None,
        patch_grid: tuple[int, int] | None = None,
        decoder_noise_tau: float = 0.0,
        decoder_noise_distribution: str = "half_normal",
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | list[torch.Tensor]:
        if latent is not None:
            return self.tokenizer.decode(latent, patch_grid=patch_grid)
        if image_tokens is None:
            raise ValueError("Either image_tokens or latent must be provided")
        z = self.tokenizer.encode(image_tokens, patch_grid=patch_grid)
        decode_z = z
        if self.training and decoder_noise_tau > 0.0:
            decode_z, _ = sample_decoder_noise_augmented_latent(
                z,
                decoder_noise_tau,
                decoder_noise_distribution,
            )
        decoded = self.tokenizer.decode(decode_z, patch_grid=patch_grid)
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
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        init_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", **init_kwargs)
    if torch.cuda.is_available():
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


def extract_scene_tokenizer_state_dict(payload: Any, ckpt_path: str) -> dict[str, torch.Tensor]:
    state_dict = payload.get("scene_tokenizer", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported scene_tokenizer checkpoint format: {ckpt_path}")
    return state_dict


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


def _collate_cache_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(predictions) == 0:
        raise ValueError("Received an empty cached predictions batch")

    out: dict[str, Any] = {}
    tensor_keys = (
        "pose_enc",
        "depth",
        "gs_map",
        "dynamic_conf",
        "gs_conf",
    )
    for key in tensor_keys:
        values = [pred.get(key) for pred in predictions]
        if any(value is None for value in values):
            raise KeyError(f"Cached predictions are missing required key {key!r}")
        out[key] = torch.cat([unwrap_tensor(value) for value in values], dim=0)

    sem_values = [pred.get("semantic_logits") for pred in predictions]
    out["semantic_logits"] = None if all(value is None for value in sem_values) else torch.cat(
        [unwrap_tensor(value) for value in sem_values],
        dim=0,
    )

    level_values = [pred.get("image_tokens_levels") for pred in predictions]
    if any(value is None for value in level_values):
        raise KeyError("Cached predictions are missing `image_tokens_levels`")
    num_levels = len(level_values[0])
    for levels in level_values:
        if len(levels) != num_levels:
            raise ValueError(f"Cached samples disagree on level count: {len(levels)} vs {num_levels}")
    out["image_tokens_levels"] = [
        torch.cat([unwrap_tensor(levels[level_idx]) for levels in level_values], dim=0)
        for level_idx in range(num_levels)
    ]

    patch_start_idx = int(predictions[0].get("patch_start_idx", 5))
    if any(int(pred.get("patch_start_idx", patch_start_idx)) != patch_start_idx for pred in predictions):
        raise ValueError("Cached samples disagree on patch_start_idx")
    out["patch_start_idx"] = patch_start_idx
    return out


def _collate_cache_level_list(items: list[list[torch.Tensor]], *, key: str) -> list[torch.Tensor]:
    if len(items) == 0:
        raise ValueError(f"Received an empty cached level list for {key!r}")
    num_levels = len(items[0])
    for levels in items:
        if len(levels) != num_levels:
            raise ValueError(f"Cached samples disagree on {key} level count: {len(levels)} vs {num_levels}")
    return [
        torch.cat([unwrap_tensor(levels[level_idx]) for levels in items], dim=0)
        for level_idx in range(num_levels)
    ]


def tokenizer_cache_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate the small subset of `WaymoFlowCacheDataset` needed by T0 training."""
    if len(batch) == 0:
        raise ValueError("Received an empty cached tokenizer batch")

    frame_counts = [int(item["sample"]["images_clean"].shape[0]) for item in batch]
    unique_frame_counts = sorted(set(frame_counts))
    if len(unique_frame_counts) != 1:
        raise ValueError(
            f"Cached tokenizer batches require a fixed frame count, got {unique_frame_counts}. "
            "Use --max_frames as the fixed cached training length, or set batch_size=1."
        )

    sample_keys = ("images_clean", "images", "sky_mask", "masks", "dynamic_mask", "timestamps")
    samples: dict[str, Any] = {}
    for key in sample_keys:
        values = [item["sample"].get(key) for item in batch]
        if any(value is None for value in values):
            if key in ("images", "masks"):
                continue
            raise KeyError(f"Cached sample is missing required key {key!r}")
        samples[key] = torch.stack([unwrap_tensor(value) for value in values], dim=0)
    samples["num_frames"] = torch.tensor(frame_counts, dtype=torch.long)

    predictions = _collate_cache_predictions([item["predictions"] for item in batch])
    teacher_values = [
        item.get("tokenizer_teacher_levels", item.get("splatted_tok_low_cached"))
        for item in batch
    ]
    if any(value is None for value in teacher_values):
        raise KeyError(
            "Cached tokenizer item is missing `tokenizer_teacher_levels`; "
            "Stage-B requires pass2_splatted_tok_low from the flow cache."
        )
    predictions["tokenizer_teacher_levels"] = _collate_cache_level_list(
        teacher_values,
        key="tokenizer_teacher_levels",
    )
    return {
        "sample": samples,
        "predictions": predictions,
        "mode_kind": [str(item.get("mode_kind", "")) for item in batch],
        "cache_path": [str(item.get("cache_path", "")) for item in batch],
        "subset_frames": [item.get("subset_frames") for item in batch],
    }


def dataloader_runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_workers) <= 0:
        return {}
    return {
        "prefetch_factor": max(1, int(args.prefetch_factor)),
        "persistent_workers": not bool(args.no_persistent_workers),
    }


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
    gt: torch.Tensor,
    pred: torch.Tensor,
    max_frames: int = 8,
) -> Image.Image:
    gt = gt.detach().cpu().float().clamp(0.0, 1.0)
    pred = pred.detach().cpu().float().clamp(0.0, 1.0)
    diff = (pred - gt).abs().clamp(0.0, 1.0)
    num_frames = min(int(max_frames), int(gt.shape[0]), int(pred.shape[0]))
    pil_images: list[Image.Image] = []
    for row in (pred, gt, diff):
        for frame_idx in range(num_frames):
            pil_images.append(tensor_to_pil_rgb(row[frame_idx]))
    return make_grid(pil_images, cols=num_frames)


def build_stage_a_validation_grid(
    raw_rgb: torch.Tensor,
    student_render: torch.Tensor,
    direct_teacher_render: torch.Tensor,
    max_frames: int = 8,
) -> Image.Image:
    """Build an unambiguous raw Stage-A validation diagnostic.

    The raw RGB observation is the only GT row.  The direct DGGT render is shown
    separately as the frozen teacher/renderer's ceiling; it must never be
    reconstructed from ``image_patch`` because those legacy joint tokens do not
    contain the layer-aligned DINO pyramid required by ``instance_head``.
    """
    tensors = [raw_rgb, student_render, direct_teacher_render]
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Stage-A validation rows must each be [S,C,H,W]")
    num_frames = min(int(max_frames), *(int(tensor.shape[0]) for tensor in tensors))
    if num_frames <= 0:
        raise ValueError("Cannot build a Stage-A validation grid with zero frames")

    raw = raw_rgb[:num_frames].detach().cpu().float().clamp(0.0, 1.0)
    student = student_render[:num_frames].detach().cpu().float().clamp(0.0, 1.0)
    direct = direct_teacher_render[:num_frames].detach().cpu().float().clamp(0.0, 1.0)
    rows = (
        ("student render", student),
        ("raw RGB GT", raw),
        ("direct DGGT render", direct),
        ("|student - raw|", (student - raw).abs().clamp(0.0, 1.0)),
    )

    first_image = tensor_to_pil_rgb(rows[0][1][0])
    cell_width, cell_height = first_image.size
    label_width = max(160, min(260, cell_width // 2))
    canvas = Image.new(
        "RGB",
        (label_width + num_frames * cell_width, len(rows) * cell_height),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(16, min(28, cell_height // 10)))
    for row_idx, (label, tensor_row) in enumerate(rows):
        y0 = row_idx * cell_height
        text_box = draw.textbbox((0, 0), label, font=font)
        text_height = text_box[3] - text_box[1]
        draw.text(
            (12, y0 + (cell_height - text_height) // 2),
            label,
            fill=(245, 245, 245),
            font=font,
        )
        for frame_idx in range(num_frames):
            image = tensor_to_pil_rgb(tensor_row[frame_idx])
            canvas.paste(image, (label_width + frame_idx * cell_width, y0))
    return canvas


def save_triplet_grid(
    gt: torch.Tensor,
    pred: torch.Tensor,
    path: Path,
    max_frames: int = 8,
) -> Image.Image:
    grid = build_triplet_grid(gt, pred, max_frames=max_frames)
    grid.save(path)
    return grid


def reduce_per_sample(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        return values.unsqueeze(0)
    if values.ndim == 1:
        return values
    return values.reshape(values.shape[0], -1).mean(dim=1)


def reduce_masked_per_sample(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must share shape, got {tuple(values.shape)} and {tuple(mask.shape)}")
    values = values.reshape(values.shape[0], -1)
    mask = mask.reshape(mask.shape[0], -1).float()
    denom = mask.sum(dim=1).clamp_min(eps)
    return (values * mask).sum(dim=1) / denom


def _masked_sample_diagnostics(
    values: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce supported values per sample and expose support coverage.

    Empty samples remain in the returned tensors so callers can preserve the
    computation graph, but ``valid_samples`` makes it explicit that they must
    not participate in a batch mean.
    """
    if values.shape != support.shape:
        raise ValueError(
            f"values and support must share shape, got {tuple(values.shape)} and {tuple(support.shape)}"
        )
    values_flat = values.reshape(values.shape[0], -1)
    support_flat = support.reshape(support.shape[0], -1)
    support_per_sample = support_flat.sum(dim=1)
    valid_samples = support_per_sample > 0
    per_sample = (values_flat * support_flat.to(values.dtype)).sum(dim=1)
    per_sample = per_sample / support_per_sample.clamp_min(1).to(values.dtype)
    diagnostics = {
        "support_count": support_per_sample.sum().detach(),
        "support_total_count": support_per_sample.new_tensor(support.numel()).detach(),
        "support_fraction": support_flat.float().mean().detach(),
        "valid_sample_count": valid_samples.sum().detach(),
        "sample_count": support_per_sample.new_tensor(support.shape[0]).detach(),
        "valid_sample_fraction": valid_samples.float().mean().detach(),
    }
    return per_sample, valid_samples, diagnostics


def _mean_over_valid_samples(
    per_sample: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    differentiable_zero: torch.Tensor,
) -> torch.Tensor:
    """Average valid samples without letting empty samples dilute the batch."""
    if bool(valid_samples.any()):
        return per_sample[valid_samples].mean()
    return differentiable_zero.sum() * 0.0


def dynamic_patch_weight(
    dynamic_mask: torch.Tensor,
    patch_grid: tuple[int, int],
    alpha: float,
    patch_size: int = 14,
) -> torch.Tensor:
    """Build a per-patch weight in [1, alpha] from a pixel-resolution dynamic mask.

    Args:
        dynamic_mask: [B, S, C, H, W] (channel 0 is the binary dynamic foreground).
        patch_grid: (patch_h, patch_w) for the patch token grid.
        alpha: weight scale at fully-dynamic patches; static patches stay at 1.
        patch_size: pixel size of each patch (Waymo default 14).

    Returns:
        weight: [B, S, P, 1] float32, where P = patch_h * patch_w.
    """
    if dynamic_mask.ndim != 5:
        raise ValueError(f"Expected dynamic_mask shape [B,S,C,H,W], got {tuple(dynamic_mask.shape)}")
    patch_h, patch_w = patch_grid
    m = dynamic_mask[:, :, 0:1].float()
    batch, seq_len = m.shape[:2]
    m_flat = m.flatten(0, 1)  # [B*S, 1, H, W]
    pooled = F.avg_pool2d(m_flat, kernel_size=patch_size, stride=patch_size)  # [B*S, 1, h, w]
    if pooled.shape[-2:] != (patch_h, patch_w):
        pooled = F.adaptive_avg_pool2d(pooled, output_size=(patch_h, patch_w))
    pooled = pooled.view(batch, seq_len, 1, patch_h * patch_w).permute(0, 1, 3, 2).contiguous()
    return 1.0 + (alpha - 1.0) * pooled  # [B, S, P, 1]


def normalized_token_reconstruction_loss(
    pred_tokens: list[torch.Tensor],
    target_tokens: list[torch.Tensor],
    std_stats: torch.Tensor,
    patch_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    per_level_losses = []
    for level_idx, (pred, target) in enumerate(zip(pred_tokens, target_tokens)):
        pred_float = pred.float()
        target_float = target.float()
        std = std_stats[level_idx].to(device=pred.device, dtype=torch.float32).view(1, 1, 1, -1)
        per_element = ((pred_float - target_float) / (std + 1e-6)) ** 2
        if patch_weight is not None:
            per_element = per_element * patch_weight.to(device=pred.device, dtype=torch.float32)
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


# ``gs_map`` channel layout, shared with dggt/utils/gs.py::get_split_gs.
GS_CHANNEL_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("rgb", 0, 3),
    ("opacity", 3, 4),
    ("scale", 4, 7),
    ("quat", 7, 11),
)


def gs_channel_group_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 1.0,
    eps: float = 1e-4,
    group_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Per-channel-group normalized Huber over ``gs_map``.

    ``normalized_huber_loss`` divides the whole 11-channel tensor by a single
    per-sample std.  That std is dominated by rgb (in [0,1]) and quats (in
    [-1,1]), while the three *linear* Gaussian scales are small and partly sit
    on their floor -- so the scale channels receive a vanishing share of the
    gradient even though they are exactly what the rasterizer consumes as
    geometry.  Normalizing each group by its own std removes that dilution.
    """
    if int(pred.shape[-1]) != GS_CHANNEL_GROUPS[-1][2]:
        raise ValueError(
            f"gs_map must have {GS_CHANNEL_GROUPS[-1][2]} channels, got {tuple(pred.shape)}"
        )
    weights = group_weights or {}
    total = None
    for name, start, end in GS_CHANNEL_GROUPS:
        group_loss = normalized_huber_loss(
            pred[..., start:end], target[..., start:end], delta=delta, eps=eps
        )
        scaled = float(weights.get(name, 1.0)) * group_loss
        total = scaled if total is None else total + scaled
    return total / float(len(GS_CHANNEL_GROUPS))


def gaussian_scale_depth_similarity_loss(
    student_gs_map: torch.Tensor,
    teacher_gs_map: torch.Tensor,
    student_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    *,
    dynamic_mask: torch.Tensor | None = None,
    sky_mask: torch.Tensor | None = None,
    opacity_threshold: float = 0.05,
    scale_floor: float = 1e-5,
    depth_floor: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paired same-pixel constraint that the encode/decode round trip be a similarity.

    If the round trip only rescaled the world by a common factor ``a``, then at
    every pixel ``scale_student/scale_teacher == depth_student/depth_teacher``.
    Measured on the shipped checkpoint the paired ratio is **0.796** (30/30
    scenes below 1): reconstruction inflates depth ~3% while shrinking the
    Gaussian radii ~17%, so every splat ends up ~20% too small for the depth it
    sits at.  Neither ``gs_anchor`` nor ``geom_anchor`` sees this because each
    is normalized independently, and the render loss does not pin it either --
    a too-small splat with a raised opacity renders almost the same.

    Returns ``(loss, diagnostics)``. ``diagnostics["mean_log_ratio"]`` should
    converge to 0 (it starts near ``log(0.796) = -0.228``); it is NaN when the
    whole batch has empty support. Support and valid-sample coverage are also
    returned so an empty or partially empty batch is visible in training logs.
    """
    scale_s = student_gs_map[..., 4:7].float().clamp_min(scale_floor)
    scale_t = teacher_gs_map[..., 4:7].float().clamp_min(scale_floor)
    log_scale_ratio = (scale_s.log() - scale_t.log()).mean(dim=-1)

    depth_s = student_depth.float()
    depth_t = teacher_depth.float()
    if depth_s.shape[-1] == 1:
        depth_s = depth_s[..., 0]
    if depth_t.shape[-1] == 1:
        depth_t = depth_t[..., 0]
    log_depth_ratio = depth_s.clamp_min(depth_floor).log() - depth_t.clamp_min(depth_floor).log()

    if log_scale_ratio.shape != log_depth_ratio.shape:
        raise ValueError(
            "gs_map and depth must agree on [B,S,H,W], got "
            f"{tuple(log_scale_ratio.shape)} and {tuple(log_depth_ratio.shape)}"
        )

    residual = log_scale_ratio - log_depth_ratio

    support = (
        torch.isfinite(residual)
        & (depth_s > depth_floor)
        & (depth_t > depth_floor)
        & (student_gs_map[..., 3].float() > opacity_threshold)
        & (teacher_gs_map[..., 3].float() > opacity_threshold)
    )
    if sky_mask is not None:
        support = support & (sky_mask[:, :, 0].float() < 0.5)
    if dynamic_mask is not None:
        # Dynamic pixels legitimately move between the two decodes; keep the
        # constraint on the static world where the similarity must hold.
        support = support & (dynamic_mask[:, :, 0].float() < 0.5)

    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    per_sample_loss, valid_samples, diagnostics = _masked_sample_diagnostics(
        residual.abs(), support
    )
    loss = _mean_over_valid_samples(
        per_sample_loss,
        valid_samples,
        differentiable_zero=residual,
    )
    with torch.no_grad():
        per_sample_ratio, _, _ = _masked_sample_diagnostics(residual.detach(), support)
        if bool(valid_samples.any()):
            mean_log_ratio = per_sample_ratio[valid_samples].mean()
        else:
            mean_log_ratio = residual.new_tensor(float("nan"))
    diagnostics["mean_log_ratio"] = mean_log_ratio
    return loss, diagnostics


def depth_log_bias_loss(
    student_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    *,
    sky_mask: torch.Tensor | None = None,
    depth_floor: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize only the *systematic multiplicative* depth offset.

    The audit measured ``depth_recon / depth_direct = 1.0307`` (CI [1.0208,
    1.0421]) -- ``geom_anchor`` converged to a solution carrying a 3% scale
    bias, because a Huber on raw depth is an absolute-error criterion and is
    happy to trade a uniform multiplicative offset against per-pixel accuracy.
    Downstream that bias is not benign: the SceneFlow render loss builds
    geometry from the reconstruction but takes its camera baseline from the
    teacher space, so a 3% inflation is a 3% parallax error (~11 px at 20 m).

    This term penalizes ``|mean_pixels(log(d_s / d_t))|`` -- the per-sample
    *bias*, not the per-pixel error.  Per-pixel accuracy remains ``geom_anchor``'s
    job; the two do not fight.

    Returns ``(loss, diagnostics)``. ``diagnostics["signed_log_bias"]`` is the
    audited log ratio and must converge to 0. It is NaN when the whole batch
    has empty support; support and valid-sample coverage are returned alongside
    it so empty samples cannot silently dilute either the loss or diagnostic.
    """
    depth_s = student_depth.float()
    depth_t = teacher_depth.float()
    if depth_s.shape[-1] == 1:
        depth_s = depth_s[..., 0]
    if depth_t.shape[-1] == 1:
        depth_t = depth_t[..., 0]

    log_ratio = depth_s.clamp_min(depth_floor).log() - depth_t.clamp_min(depth_floor).log()
    support = torch.isfinite(log_ratio) & (depth_s > depth_floor) & (depth_t > depth_floor)
    if sky_mask is not None:
        # Sky depth is unconstrained/degenerate; it would dominate the mean.
        support = support & (sky_mask[:, :, 0].float() < 0.5)

    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    per_sample_bias, valid_samples, diagnostics = _masked_sample_diagnostics(
        log_ratio, support
    )
    loss = _mean_over_valid_samples(
        per_sample_bias.abs(),
        valid_samples,
        differentiable_zero=log_ratio,
    )
    with torch.no_grad():
        if bool(valid_samples.any()):
            signed_log_bias = per_sample_bias.detach()[valid_samples].mean()
        else:
            signed_log_bias = log_ratio.new_tensor(float("nan"))
    diagnostics["signed_log_bias"] = signed_log_bias
    return loss, diagnostics


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    delta: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Per-element Huber loss with broadcastable weighting.

    Unlike `normalized_huber_loss`, this keeps the original scale (so weighting
    is meaningful) but normalizes by the target's global std for numerical stability.
    """
    pred = pred.float()
    target = target.float()
    flat_target = target.reshape(target.shape[0], -1)
    scale = flat_target.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    scale = scale.view(target.shape[0], *([1] * (target.ndim - 1)))
    per_element = F.smooth_l1_loss(pred / scale, target / scale, beta=delta, reduction="none")
    weight = weight.float()
    per_sample_weighted = (per_element * weight).reshape(per_element.shape[0], -1).mean(dim=1)
    return per_sample_weighted.mean()


def dynamic_mask_bce_loss(dynamic_logits: torch.Tensor, dynamic_mask: torch.Tensor) -> torch.Tensor:
    logits = dynamic_logits.float().squeeze(-1)
    if dynamic_mask.ndim != 5:
        raise ValueError(f"Expected dynamic_mask to have shape [B,S,C,H,W], got {tuple(dynamic_mask.shape)}")
    target = dynamic_mask[:, :, 0].float()
    flat_target = target.reshape(target.shape[0], -1)
    pos = flat_target.sum(dim=1)
    neg = flat_target.shape[1] - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp_min(1.0).clamp_max(48.0)
    weight = torch.where(
        target > 0.5,
        pos_weight.view(-1, 1, 1, 1),
        torch.ones_like(target),
    )
    per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight
    return reduce_per_sample(per_element).mean()


def compute_lifespan_loss(
    gs_conf: torch.Tensor,
    dynamic_mask: torch.Tensor | None = None,
    sky_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    gamma = gs_conf.float()
    gamma_inv = torch.abs(1.0 / gamma.clamp_min(eps))
    if dynamic_mask is None:
        return reduce_per_sample(gamma_inv).mean()

    static_mask = dynamic_mask[:, :, 0].float() < 0.5
    if sky_mask is not None:
        static_mask = static_mask & (sky_mask[:, :, 0].float() < 0.5)
    return reduce_masked_per_sample(gamma_inv, static_mask).mean()


def ghost_static_loss(
    gs_map: torch.Tensor,
    dynamic_conf: torch.Tensor,
    dynamic_mask: torch.Tensor,
    sky_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    opacity = gs_map[..., 3].float()
    dynamic_prob = torch.sigmoid(dynamic_conf.float().squeeze(-1))
    dynamic_target = dynamic_mask[:, :, 0].float()
    static_leak = opacity * (1.0 - dynamic_prob)
    mask = dynamic_target > 0.5
    if sky_mask is not None:
        mask = mask & (sky_mask[:, :, 0].float() < 0.5)
    return reduce_masked_per_sample(static_leak, mask).mean()


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
    dynamic_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if images.shape[0] != 1:
        raise ValueError(f"Single-scene renderer expects batch dimension 1, got {images.shape[0]}")

    _, _, _, height, width = images.shape
    extrinsics_3x4, intrinsics = pose_encoding_to_extri_intri(pose_enc, (height, width))
    extrinsic_4x4 = build_extrinsics_4x4(extrinsics_3x4)
    point_map = unproject_depth_map_to_point_map_torch(depth.float(), extrinsics_3x4.float(), intrinsics.float())

    sky_mask_hw = sky_mask.permute(0, 1, 3, 4, 2)
    non_sky_mask = (sky_mask_hw < 0.5).any(dim=-1)
    dynamic_logits = dynamic_conf.float().squeeze(-1)
    dynamic_prob = torch.sigmoid(dynamic_logits)
    # Match deployed DGGT inference: split static/dynamic with the raw dynamic
    # logit, then use sigmoid(dynamic_logit) as the opacity mixing weight.
    static_mask = non_sky_mask & (dynamic_logits < 0.5)
    if dynamic_mask is not None:
        if dynamic_mask.ndim != 5 or dynamic_mask.shape[:2] != images.shape[:2]:
            raise ValueError(
                "dynamic_mask must be [B,S,C,H,W] and align with images, got "
                f"{tuple(dynamic_mask.shape)} for images {tuple(images.shape)}"
            )
        dynamic_gt = dynamic_mask[:, :, 0:1].float().flatten(0, 1)
        if dynamic_gt.shape[-2:] != (height, width):
            dynamic_gt = F.interpolate(dynamic_gt, size=(height, width), mode="nearest")
        dynamic_gt = dynamic_gt.view(1, images.shape[1], height, width) >= 0.5
        # Keep calibrated probabilities wherever the head already recognizes
        # dynamics.  For GT false negatives only, move the complete Gaussian
        # into the current-frame branch: merely removing its static complement
        # would create a low-opacity hole, while leaving it static creates a
        # multi-frame ghost trail.
        false_negative = dynamic_gt & static_mask
        dynamic_prob = torch.where(false_negative, torch.ones_like(dynamic_prob), dynamic_prob)
        static_mask = static_mask & ~dynamic_gt
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
    # gsplat returns premultiplied RGB when no rasterizer background is given.
    # Applying alpha to `renders` again darkens partially covered pixels and
    # makes moving-object boundaries look like transparent ghost trails.
    composed = composite_gsplat_rgb(renders, alphas, bg_render)
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
    *,
    dynamic_mask: torch.Tensor | None = None,
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
                None if dynamic_mask is None else dynamic_mask[batch_idx : batch_idx + 1],
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


def compute_feature_stats_from_cache(
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
    levels: tuple[int, ...],
    stats_path: Path,
) -> dict[str, torch.Tensor]:
    sum_acc: torch.Tensor | None = None
    sum_sq_acc: torch.Tensor | None = None
    count_acc = torch.zeros((len(levels),), dtype=torch.float64, device=device)

    max_items = min(len(dataset), args.stats_steps)
    rank = get_rank()
    world_size = dist.get_world_size() if is_distributed() else 1
    local_indices = list(range(rank, max_items, world_size))
    stats_pbar = None
    if is_main_process():
        print(
            f"[feature_stats/cache] start: total_items={max_items} world_size={world_size} "
            f"items_on_rank0={len(local_indices)}",
            flush=True,
        )
        stats_pbar = tqdm(
            total=len(local_indices),
            desc="feature_stats_cache(rank0)",
            dynamic_ncols=True,
            leave=True,
        )

    for sample_idx in local_indices:
        item = dataset[sample_idx]
        image_patch = item.get("tokenizer_teacher_levels")
        if image_patch is None:
            image_patch = item["predictions"].get("tokenizer_teacher_levels")
        if image_patch is None:
            image_patch = item["predictions"]["image_tokens_levels"]
        if len(image_patch) != len(levels):
            raise ValueError(
                f"Cached item has {len(image_patch)} levels, but tokenizer expects {len(levels)}"
            )
        channels = int(image_patch[0].shape[-1])
        if sum_acc is None:
            sum_acc = torch.zeros((len(levels), channels), dtype=torch.float64, device=device)
            sum_sq_acc = torch.zeros_like(sum_acc)
        for level_idx, tokens in enumerate(image_patch):
            flat = tokens.reshape(-1, tokens.shape[-1]).to(device=device, dtype=torch.float64)
            sum_acc[level_idx] += flat.sum(dim=0)
            sum_sq_acc[level_idx] += (flat * flat).sum(dim=0)
            count_acc[level_idx] += flat.shape[0]
        if stats_pbar is not None:
            stats_pbar.update(1)

    if sum_acc is None or sum_sq_acc is None:
        raise RuntimeError("Could not compute feature stats from an empty cache dataset")

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
        print(f"[feature_stats/cache] saved to {stats_path}", flush=True)
    return payload


def get_feature_stats_from_cache(
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
    levels: tuple[int, ...],
    stats_path: Path,
) -> dict[str, torch.Tensor]:
    if is_main_process() and not stats_path.is_file():
        compute_feature_stats_from_cache(dataset, args, device, levels, stats_path)
    elif not stats_path.is_file():
        compute_feature_stats_from_cache(dataset, args, device, levels, stats_path)
    if is_distributed():
        dist.barrier()
    return torch.load(stats_path, map_location="cpu")


def _cache_teacher_levels_from_item(item: dict[str, Any]) -> list[torch.Tensor]:
    image_patch = item.get("tokenizer_teacher_levels")
    if image_patch is None:
        image_patch = item["predictions"].get("tokenizer_teacher_levels")
    if image_patch is None:
        image_patch = item["predictions"]["image_tokens_levels"]
    return image_patch


def _accumulate_feature_levels(
    image_patch: list[torch.Tensor],
    *,
    levels: tuple[int, ...],
    device: torch.device,
    sum_acc: torch.Tensor | None,
    sum_sq_acc: torch.Tensor | None,
    count_acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(image_patch) != len(levels):
        raise ValueError(
            f"Feature stats item has {len(image_patch)} levels, but tokenizer expects {len(levels)}"
        )
    channels = int(image_patch[0].shape[-1])
    if sum_acc is None:
        sum_acc = torch.zeros((len(levels), channels), dtype=torch.float64, device=device)
        sum_sq_acc = torch.zeros_like(sum_acc)
    if sum_sq_acc is None:
        raise RuntimeError("sum_sq_acc unexpectedly None after sum_acc initialization")
    for level_idx, tokens in enumerate(image_patch):
        flat = tokens.reshape(-1, tokens.shape[-1]).to(device=device, dtype=torch.float64)
        sum_acc[level_idx] += flat.sum(dim=0)
        sum_sq_acc[level_idx] += (flat * flat).sum(dim=0)
        count_acc[level_idx] += flat.shape[0]
    return sum_acc, sum_sq_acc


def compute_feature_stats_mixed(
    model: VGGT,
    datasets: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    levels: tuple[int, ...],
    stats_path: Path,
) -> dict[str, torch.Tensor]:
    schedule = ("raw", "cache_mode_a", "raw", "cache_mode_b")
    for name in schedule:
        if name not in datasets or len(datasets[name]) == 0:
            raise ValueError(f"Mixed feature stats require non-empty dataset {name!r}")

    sum_acc: torch.Tensor | None = None
    sum_sq_acc: torch.Tensor | None = None
    count_acc = torch.zeros((len(levels),), dtype=torch.float64, device=device)

    max_items = int(args.stats_steps)
    rank = get_rank()
    dist_world_size = dist.get_world_size() if is_distributed() else 1
    if max_items < 1:
        raise ValueError("--stats_steps must be positive for mixed feature stats")
    local_total = int(math.ceil(max_items / float(dist_world_size)))
    stats_pbar = None
    if is_main_process():
        print(
            f"[feature_stats/mixed] start: requested_total_items={max_items} "
            f"world_size={dist_world_size} items_per_rank={local_total} "
            "schedule=raw,mode_a,raw,mode_b ",
            flush=True,
        )
        stats_pbar = tqdm(
            total=local_total,
            desc="feature_stats_mixed(rank0)",
            dynamic_ncols=True,
            leave=True,
        )

    model.eval()
    error_message = None
    try:
        for local_idx in range(local_total):
            source_name = schedule[int(local_idx) % len(schedule)]
            source_dataset = datasets[source_name]
            item_idx = ((int(local_idx) // len(schedule)) * dist_world_size + rank) % len(source_dataset)
            if source_name == "raw":
                sample = source_dataset[(item_idx, args.max_frames)]
                images = unwrap_tensor(sample["images_clean"]).unsqueeze(0).to(device)
                with torch.no_grad(), autocast_context(args, device):
                    if hasattr(model, "extract_scene_tokens"):
                        agg_all, image_all, dino_all, _, patch_start_idx = model.extract_scene_tokens(images)
                    else:
                        agg_all, image_all, dino_all, _, patch_start_idx = model.aggregator(images)
                    del agg_all, dino_all
                    image_patch = select_patch_pyramid(image_all, levels, patch_start_idx)
            else:
                item = source_dataset[item_idx]
                image_patch = _cache_teacher_levels_from_item(item)
            sum_acc, sum_sq_acc = _accumulate_feature_levels(
                image_patch,
                levels=levels,
                device=device,
                sum_acc=sum_acc,
                sum_sq_acc=sum_sq_acc,
                count_acc=count_acc,
            )
            if stats_pbar is not None:
                stats_pbar.update(1)
    except Exception as exc:
        error_message = (
            f"[feature_stats/mixed] rank={rank}/{dist_world_size} failed "
            f"source={locals().get('source_name', 'unknown')} "
            f"item_idx={locals().get('item_idx', 'unknown')}: {type(exc).__name__}: {exc}"
        )
    finally:
        if stats_pbar is not None:
            stats_pbar.close()

    print(
        f"[feature_stats/mixed] rank={rank}/{dist_world_size} local loop finished; syncing error status...",
        flush=True,
    )
    status = torch.tensor(1 if error_message is not None else 0, dtype=torch.int32, device=device)
    if is_distributed():
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
    if int(status.item()) != 0:
        if error_message is not None:
            print(error_message, flush=True)
        raise RuntimeError("Mixed feature stats failed on at least one rank; see rank logs above.")

    if sum_acc is None or sum_sq_acc is None:
        raise RuntimeError("Could not compute feature stats from empty mixed datasets")

    print(
        f"[feature_stats/mixed] rank={rank}/{dist_world_size} local accumulation done "
        f"items={local_total} count={count_acc.detach().cpu().tolist()}",
        flush=True,
    )
    if is_distributed():
        if is_main_process():
            print("[feature_stats/mixed] reducing stats across ranks...", flush=True)
        dist.all_reduce(sum_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_sq_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_acc, op=dist.ReduceOp.SUM)
        if is_main_process():
            print(
                f"[feature_stats/mixed] reduce done global_count={count_acc.detach().cpu().tolist()}",
                flush=True,
            )

    mean = sum_acc / count_acc.view(-1, 1).clamp_min(1.0)
    var = sum_sq_acc / count_acc.view(-1, 1).clamp_min(1.0) - mean * mean
    std = var.clamp_min(1e-6).sqrt()
    payload = {
        "levels": torch.tensor(levels, dtype=torch.long),
        "mean": mean.cpu().float(),
        "std": std.cpu().float(),
        "source_mix": {
            "schedule": list(schedule),
            "raw": 0.5,
            "mode_a": 0.25,
            "mode_b": 0.25,
        },
    }
    if is_main_process():
        print(f"[feature_stats/mixed] saving to {stats_path}", flush=True)
        torch.save(payload, stats_path)
        print(f"[feature_stats/mixed] saved to {stats_path}", flush=True)
    return payload


def get_feature_stats_mixed(
    model: VGGT,
    datasets: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    levels: tuple[int, ...],
    stats_path: Path,
) -> dict[str, torch.Tensor]:
    if is_distributed():
        compute_flag = torch.tensor(
            1 if (is_main_process() and not stats_path.is_file()) else 0,
            dtype=torch.int32,
            device=device,
        )
        dist.broadcast(compute_flag, src=0)
        if int(compute_flag.item()) != 0:
            compute_feature_stats_mixed(model, datasets, args, device, levels, stats_path)
        dist.barrier()
    elif not stats_path.is_file():
        compute_feature_stats_mixed(model, datasets, args, device, levels, stats_path)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Mixed feature stats file was not created: {stats_path}")
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
        "head_targets_valid": True,
        "pose_enc": pose_enc,
        "gs_map": gs_map,
        "gs_conf": gs_conf,
        "depth": depth,
        "depth_conf": depth_conf,
        "dynamic_conf": dynamic_conf,
    }


def get_cached_teacher_outputs(
    predictions: dict[str, Any],
    levels: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    teacher_levels = predictions.get("tokenizer_teacher_levels")
    if teacher_levels is None:
        teacher_levels = predictions["image_tokens_levels"]
    image_patch = [unwrap_tensor(t).to(device, non_blocking=True) for t in teacher_levels]
    if len(image_patch) != len(levels):
        raise ValueError(
            f"Cached predictions contain {len(image_patch)} feature levels, "
            f"but tokenizer heads expect {len(levels)} levels"
        )
    teacher = {
        "num_levels": max(levels) + 1,
        "image_patch": image_patch,
        # Cached pass1 only stores patch tokens for the selected levels. Student
        # heads therefore receive patch-only sparse level lists with start index 0.
        "patch_start_idx": 0,
        "head_targets_valid": False,
        "pose_enc": unwrap_tensor(predictions["pose_enc"]).to(device, non_blocking=True),
        "gs_map": unwrap_tensor(predictions["gs_map"]).to(device, non_blocking=True),
        "gs_conf": unwrap_tensor(predictions["gs_conf"]).to(device, non_blocking=True),
        "depth": unwrap_tensor(predictions["depth"]).to(device, non_blocking=True),
        "depth_conf": None,
        "dynamic_conf": unwrap_tensor(predictions["dynamic_conf"]).to(device, non_blocking=True),
    }
    return teacher


def build_head_outputs_from_patch_features(
    model: VGGT,
    images: torch.Tensor,
    levels: tuple[int, ...],
    teacher: dict[str, Any],
    patch_features: list[torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    patch_start_idx = int(teacher.get("patch_start_idx", 0))
    if "image_levels" in teacher:
        image_features = reattach_special_tokens_from_selected(
            teacher["image_levels"],
            patch_start_idx,
            patch_features,
        )
    elif patch_start_idx == 0:
        image_features = patch_features
    else:
        raise RuntimeError("Cannot decode patch features with special tokens when teacher['image_levels'] is absent")

    dino_features = []
    agg_features = []
    for joint_tokens in image_features:
        dino_tokens, frame_tokens, global_tokens = split_joint_channels(joint_tokens)
        dino_features.append(dino_tokens)
        agg_features.append(torch.cat([frame_tokens, global_tokens], dim=-1))

    image_all = build_sparse_level_list(teacher["num_levels"], levels, image_features)
    dino_all = build_sparse_level_list(teacher["num_levels"], levels, dino_features)
    agg_all = build_sparse_level_list(teacher["num_levels"], levels, agg_features)

    gs_map, gs_conf = model.gs_head(image_all, images, patch_start_idx)
    depth, depth_conf = model.depth_head(agg_all, images, patch_start_idx)
    dynamic_conf, _ = model.instance_head(dino_all, images, patch_start_idx)
    return {
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
    args: argparse.Namespace,
) -> dict[str, Any]:
    patch_grid = infer_patch_grid(images, teacher["image_patch"][0].shape[2])
    with autocast_enabled:
        z, decoded_patch = tokenizer_runner(
            image_tokens=teacher["image_patch"],
            patch_grid=patch_grid,
            decoder_noise_tau=args.decoder_noise_tau,
            decoder_noise_distribution=args.decoder_noise_distribution,
        )
        head_outputs = build_head_outputs_from_patch_features(
            model,
            images,
            levels,
            teacher,
            decoded_patch,
        )

    return {
        "z": z,
        "decoded_patch": decoded_patch,
        **head_outputs,
    }


def build_student_outputs_from_patch_teacher(
    model: VGGT,
    tokenizer_runner: nn.Module,
    images: torch.Tensor,
    levels: tuple[int, ...],
    teacher: dict[str, Any],
    autocast_enabled,
    args: argparse.Namespace,
    *,
    force_decode_heads: bool = False,
) -> dict[str, Any]:
    patch_grid = infer_patch_grid(images, teacher["image_patch"][0].shape[2])
    with autocast_enabled:
        z, decoded_patch = tokenizer_runner(
            image_tokens=teacher["image_patch"],
            patch_grid=patch_grid,
            decoder_noise_tau=args.decoder_noise_tau,
            decoder_noise_distribution=args.decoder_noise_distribution,
        )
        if not force_decode_heads and not bool(teacher.get("head_targets_valid", True)):
            return {
                "z": z,
                "decoded_patch": decoded_patch,
            }
        head_outputs = build_head_outputs_from_patch_features(
            model,
            images,
            levels,
            teacher,
            decoded_patch,
        )

    return {
        "z": z,
        "decoded_patch": decoded_patch,
        **head_outputs,
    }


def compute_noisy_decoder_loss(
    tokenizer_runner: nn.Module,
    latent: torch.Tensor,
    target_tokens: list[torch.Tensor],
    std_stats: torch.Tensor,
    patch_grid: tuple[int, int],
    autocast_enabled,
    decoder_noise_tau: float = 0.0,
    decoder_noise_distribution: str = "half_normal",
) -> torch.Tensor:
    with autocast_enabled:
        if decoder_noise_tau > 0.0:
            z_noisy, _ = sample_decoder_noise_augmented_latent(
                latent,
                decoder_noise_tau,
                decoder_noise_distribution,
            )
        else:
            z_noisy, _, _ = sample_noisy_latent(latent)

        def _decode_noisy_tuple(noisy_latent: torch.Tensor):
            decoded = tokenizer_runner(latent=noisy_latent, patch_grid=patch_grid)
            return tuple(decoded)

        if torch.is_grad_enabled():
            decoded_noisy = activation_checkpoint(_decode_noisy_tuple, z_noisy, use_reentrant=False)
        else:
            decoded_noisy = _decode_noisy_tuple(z_noisy)
        noisy_loss = normalized_token_reconstruction_loss(list(decoded_noisy), target_tokens, std_stats)
        del decoded_noisy, z_noisy
    return noisy_loss


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
    dynamic_mask = unwrap_tensor(sample["dynamic_mask"]).to(student["z"].device) if "dynamic_mask" in sample else None
    sky_mask = unwrap_tensor(sample["sky_mask"] if "sky_mask" in sample else sample["masks"]).to(student["z"].device)

    patch_weight = None
    if dynamic_mask is not None and args.dyn_patch_alpha > 1.0:
        ref_patch = teacher["image_patch"][0]
        num_patches = int(ref_patch.shape[2])
        patch_h, patch_w = infer_patch_grid(
            unwrap_tensor(sample["images_clean"]).to(student["z"].device),
            num_patches,
        )
        patch_weight = dynamic_patch_weight(dynamic_mask, (patch_h, patch_w), args.dyn_patch_alpha)

    losses["tok_rec"] = normalized_token_reconstruction_loss(
        student["decoded_patch"], teacher["image_patch"], std_stats, patch_weight=patch_weight
    )
    losses["tok_cos"] = token_cosine_loss(student["decoded_patch"], teacher["image_patch"])

    head_targets_valid = bool(teacher.get("head_targets_valid", True))
    zero = student["z"].new_tensor(0.0)
    losses["gs_anchor"] = zero
    losses["geom_anchor"] = zero
    losses["dyn_anchor"] = zero
    losses["head_anchor"] = zero
    losses["gs_scale_sim"] = zero
    losses["depth_log_bias"] = zero
    losses["dynamic_bce"] = student["z"].new_tensor(0.0)
    losses["gs_lifespan"] = zero
    losses["ghost_static"] = student["z"].new_tensor(0.0)
    gs_scale_sim_log_ratio = float("nan")
    gs_scale_sim_support_count = 0.0
    gs_scale_sim_support_total_count = 0.0
    gs_scale_sim_support_fraction = 0.0
    gs_scale_sim_valid_sample_count = 0.0
    gs_scale_sim_sample_count = 0.0
    gs_scale_sim_valid_sample_fraction = 0.0
    depth_log_bias_value = float("nan")
    depth_log_bias_support_count = 0.0
    depth_log_bias_support_total_count = 0.0
    depth_log_bias_support_fraction = 0.0
    depth_log_bias_valid_sample_count = 0.0
    depth_log_bias_sample_count = 0.0
    depth_log_bias_valid_sample_fraction = 0.0
    if head_targets_valid:
        gs_anchor = gs_channel_group_huber_loss(
            student["gs_map"], teacher["gs_map"]
        ) + 0.1 * normalized_huber_loss(student["gs_conf"], teacher["gs_conf"])
        geom_anchor = normalized_huber_loss(student["depth"], teacher["depth"])
        if teacher.get("depth_conf") is not None:
            geom_anchor = geom_anchor + 0.1 * normalized_huber_loss(
                student["depth_conf"], teacher["depth_conf"]
            )
        if dynamic_mask is not None and args.dyn_pixel_alpha > 1.0:
            # student["dynamic_conf"] is [B, S, H, W, 1]; build matching pixel weight.
            dyn_logits = student["dynamic_conf"]
            target_hw = dyn_logits.shape[2:4]
            dyn_pix = dynamic_mask[:, :, 0:1].float().flatten(0, 1)  # [B*S, 1, H, W]
            if dyn_pix.shape[-2:] != target_hw:
                dyn_pix = F.interpolate(dyn_pix, size=target_hw, mode="bilinear", align_corners=False)
            dyn_pix = dyn_pix.view(*dyn_logits.shape[:2], target_hw[0], target_hw[1], 1)
            pix_weight = 1.0 + (args.dyn_pixel_alpha - 1.0) * dyn_pix
            dyn_anchor = masked_huber_loss(student["dynamic_conf"], teacher["dynamic_conf"], pix_weight)
        else:
            dyn_anchor = normalized_huber_loss(student["dynamic_conf"], teacher["dynamic_conf"])
        losses["gs_anchor"] = gs_anchor
        losses["geom_anchor"] = geom_anchor
        losses["dyn_anchor"] = dyn_anchor
        losses["head_anchor"] = gs_anchor + geom_anchor + dyn_anchor
        # Always measured, even when the weight is 0, so an ablation run still
        # reports the audited ratio instead of a misleading 1.0.
        gs_scale_sim, gs_scale_sim_diagnostics = gaussian_scale_depth_similarity_loss(
            student["gs_map"],
            teacher["gs_map"],
            student["depth"],
            teacher["depth"],
            dynamic_mask=dynamic_mask,
            sky_mask=sky_mask,
            opacity_threshold=float(args.gs_scale_sim_opacity),
        )
        losses["gs_scale_sim"] = gs_scale_sim
        gs_scale_sim_log_ratio = float(gs_scale_sim_diagnostics["mean_log_ratio"].item())
        gs_scale_sim_support_count = float(gs_scale_sim_diagnostics["support_count"].item())
        gs_scale_sim_support_total_count = float(
            gs_scale_sim_diagnostics["support_total_count"].item()
        )
        gs_scale_sim_support_fraction = float(
            gs_scale_sim_diagnostics["support_fraction"].item()
        )
        gs_scale_sim_valid_sample_count = float(
            gs_scale_sim_diagnostics["valid_sample_count"].item()
        )
        gs_scale_sim_sample_count = float(gs_scale_sim_diagnostics["sample_count"].item())
        gs_scale_sim_valid_sample_fraction = float(
            gs_scale_sim_diagnostics["valid_sample_fraction"].item()
        )

        depth_bias, depth_bias_diagnostics = depth_log_bias_loss(
            student["depth"], teacher["depth"], sky_mask=sky_mask
        )
        losses["depth_log_bias"] = depth_bias
        depth_log_bias_value = float(depth_bias_diagnostics["signed_log_bias"].item())
        depth_log_bias_support_count = float(depth_bias_diagnostics["support_count"].item())
        depth_log_bias_support_total_count = float(
            depth_bias_diagnostics["support_total_count"].item()
        )
        depth_log_bias_support_fraction = float(
            depth_bias_diagnostics["support_fraction"].item()
        )
        depth_log_bias_valid_sample_count = float(
            depth_bias_diagnostics["valid_sample_count"].item()
        )
        depth_log_bias_sample_count = float(depth_bias_diagnostics["sample_count"].item())
        depth_log_bias_valid_sample_fraction = float(
            depth_bias_diagnostics["valid_sample_fraction"].item()
        )
        if dynamic_mask is not None:
            losses["dynamic_bce"] = dynamic_mask_bce_loss(student["dynamic_conf"], dynamic_mask)
        losses["gs_lifespan"] = compute_lifespan_loss(
            student["gs_conf"],
            dynamic_mask=dynamic_mask,
            sky_mask=sky_mask,
        )
        if dynamic_mask is not None:
            losses["ghost_static"] = ghost_static_loss(
                student["gs_map"],
                student["dynamic_conf"],
                dynamic_mask,
                sky_mask=sky_mask,
            )

    losses["lat_stat"] = latent_stat_loss(student["z"])

    render_ref = None
    render_hat = None
    losses["render_anchor"] = student["z"].new_tensor(0.0)
    losses["render_lpips"] = student["z"].new_tensor(0.0)
    losses["render_gt"] = student["z"].new_tensor(0.0)

    if head_targets_valid and global_step >= args.render_start_step:
        if model.lpips_loss_fn is None:
            raise RuntimeError("Render anchor is active but LPIPS has not been initialized")
        timestamps = unwrap_tensor(sample["timestamps"]).to(student["z"].device)
        images_clean = unwrap_tensor(sample["images_clean"]).to(student["z"].device)
        render_ref = render_scene_from_outputs(
            model,
            images_clean,
            sky_mask,
            timestamps,
            teacher["pose_enc"].float(),
            teacher["depth"].float(),
            teacher["gs_map"].float(),
            teacher["gs_conf"].float(),
            teacher["dynamic_conf"].float(),
            dynamic_mask=dynamic_mask,
        )
        render_hat = render_scene_from_outputs(
            model,
            images_clean,
            sky_mask,
            timestamps,
            teacher["pose_enc"].float(),
            student["depth"].float(),
            student["gs_map"].float(),
            student["gs_conf"].float(),
            student["dynamic_conf"].float(),
            dynamic_mask=dynamic_mask,
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

        if args.gt_render_ratio > 0.0:
            gt_rgb = images_clean.float()  # [B, S, 3, H, W]
            if render_hat.shape != gt_rgb.shape:
                raise RuntimeError(
                    f"render_hat shape {tuple(render_hat.shape)} does not match images_clean "
                    f"{tuple(gt_rgb.shape)}"
                )
            if dynamic_mask is not None and args.render_dyn_alpha > 1.0:
                dyn_pix = dynamic_mask[:, :, 0:1].float().flatten(0, 1)  # [B*S, 1, H, W]
                if dyn_pix.shape[-2:] != gt_rgb.shape[-2:]:
                    dyn_pix = F.interpolate(
                        dyn_pix, size=gt_rgb.shape[-2:], mode="bilinear", align_corners=False
                    )
                dyn_pix = dyn_pix.view(*gt_rgb.shape[:2], 1, *gt_rgb.shape[-2:])
                w_pix = 1.0 + (args.render_dyn_alpha - 1.0) * dyn_pix
                gt_l2_per_element = (render_hat - gt_rgb) ** 2 * w_pix
                gt_l2 = reduce_per_sample(gt_l2_per_element).mean()
            else:
                gt_l2 = reduce_per_sample((render_hat - gt_rgb) ** 2).mean()
            gt_lpips_per_frame = model.lpips_loss_fn(
                render_hat.flatten(0, 1) * 2.0 - 1.0,
                gt_rgb.flatten(0, 1) * 2.0 - 1.0,
            )
            gt_lpips_per_sample = reduce_per_sample(
                gt_lpips_per_frame.reshape(render_hat.shape[0], render_hat.shape[1], -1)
            )
            losses["render_gt"] = gt_l2 + 0.1 * gt_lpips_per_sample.mean()

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
    ghost_static_weight = scheduled_weight(
        global_step,
        args.head_start_step,
        args.head_warmup_steps,
        args.lambda_ghost_static,
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

    # The similarity term consumes the same frozen-head tensors as head_anchor,
    # so it follows the identical start/warmup schedule.
    gs_scale_sim_weight = scheduled_weight(
        global_step,
        args.head_start_step,
        args.head_warmup_steps,
        args.lambda_gs_scale_sim,
    )

    depth_log_bias_weight = scheduled_weight(
        global_step,
        args.head_start_step,
        args.head_warmup_steps,
        args.lambda_depth_log_bias,
    )

    gt_render_weight = render_weight * float(args.gt_render_ratio)
    total = (
        args.lambda_tok_rec * losses["tok_rec"]
        + args.lambda_tok_cos * losses["tok_cos"]
        + head_weight * losses["head_anchor"]
        + gs_scale_sim_weight * losses["gs_scale_sim"]
        + depth_log_bias_weight * losses["depth_log_bias"]
        + dynamic_bce_weight * losses["dynamic_bce"]
        + gs_lifespan_weight * losses["gs_lifespan"]
        + ghost_static_weight * losses["ghost_static"]
        + render_weight * losses["render_anchor"]
        + gt_render_weight * losses["render_gt"]
        + args.lambda_lat_stat * losses["lat_stat"]
    )

    scalar_logs = {
        "loss": float(total.detach().item()),
        "tok_rec": float(losses["tok_rec"].detach().item()),
        "tok_cos": float(losses["tok_cos"].detach().item()),
        "gs_anchor": float(losses["gs_anchor"].detach().item()),
        "geom_anchor": float(losses["geom_anchor"].detach().item()),
        "dyn_anchor": float(losses["dyn_anchor"].detach().item()),
        "gs_scale_sim": float(losses["gs_scale_sim"].detach().item()),
        # Should converge to 0. Starts near log(0.796) = -0.228 on the old
        # checkpoint; exp() of it is the paired GS/depth ratio being audited.
        "gs_scale_sim_log_ratio": gs_scale_sim_log_ratio,
        "gs_scale_sim_ratio": float(math.exp(gs_scale_sim_log_ratio)),
        "gs_scale_sim_support_count": gs_scale_sim_support_count,
        "gs_scale_sim_support_total_count": gs_scale_sim_support_total_count,
        "gs_scale_sim_support_fraction": gs_scale_sim_support_fraction,
        "gs_scale_sim_valid_sample_count": gs_scale_sim_valid_sample_count,
        "gs_scale_sim_sample_count": gs_scale_sim_sample_count,
        "gs_scale_sim_valid_sample_fraction": gs_scale_sim_valid_sample_fraction,
        "gs_scale_sim_weight": float(gs_scale_sim_weight),
        "depth_log_bias": float(losses["depth_log_bias"].detach().item()),
        # exp() of this is the audited depth_recon/depth_direct ratio (was 1.0307).
        "depth_log_bias_signed": depth_log_bias_value,
        "depth_ratio": float(math.exp(depth_log_bias_value)),
        "depth_log_bias_support_count": depth_log_bias_support_count,
        "depth_log_bias_support_total_count": depth_log_bias_support_total_count,
        "depth_log_bias_support_fraction": depth_log_bias_support_fraction,
        "depth_log_bias_valid_sample_count": depth_log_bias_valid_sample_count,
        "depth_log_bias_sample_count": depth_log_bias_sample_count,
        "depth_log_bias_valid_sample_fraction": depth_log_bias_valid_sample_fraction,
        "dynamic_bce": float(losses["dynamic_bce"].detach().item()),
        "gs_lifespan": float(losses["gs_lifespan"].detach().item()),
        "ghost_static": float(losses["ghost_static"].detach().item()),
        "render_anchor": float(losses["render_anchor"].detach().item()),
        "render_lpips": float(losses["render_lpips"].detach().item()),
        "render_gt": float(losses["render_gt"].detach().item()),
        "noisy": 0.0,
        "lat_stat": float(losses["lat_stat"].detach().item()),
        "latent_mean": float(student["z"].mean().detach().item()),
        "latent_std": float(student["z"].std(unbiased=False).detach().item()),
        "head_weight": float(head_weight),
        "dynamic_bce_weight": float(dynamic_bce_weight),
        "gs_lifespan_weight": float(gs_lifespan_weight),
        "ghost_static_weight": float(ghost_static_weight),
        "render_weight": float(render_weight),
        "gt_render_weight": float(gt_render_weight),
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


_V2_LOSS_DIAGNOSTIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("gs_scale_sim", "gs_scale_sim_log_ratio", "gs_scale_sim_ratio"),
    ("depth_log_bias", "depth_log_bias_signed", "depth_ratio"),
)
_V2_LOSS_DIAGNOSTIC_LOG_KEYS = frozenset(
    key
    for prefix, signed_key, ratio_key in _V2_LOSS_DIAGNOSTIC_SPECS
    for key in (
        signed_key,
        ratio_key,
        f"{prefix}_support_count",
        f"{prefix}_support_total_count",
        f"{prefix}_support_fraction",
        f"{prefix}_valid_sample_count",
        f"{prefix}_sample_count",
        f"{prefix}_valid_sample_fraction",
    )
)


def _accumulate_v2_loss_diagnostics(
    totals: dict[str, float],
    scalar_logs: dict[str, float],
) -> None:
    """Accumulate supported diagnostics without cache-row NaN pollution."""
    for prefix, signed_key, _ratio_key in _V2_LOSS_DIAGNOSTIC_SPECS:
        for suffix in (
            "support_count",
            "support_total_count",
            "valid_sample_count",
            "sample_count",
        ):
            key = f"{prefix}_{suffix}"
            totals[key] = totals.get(key, 0.0) + float(scalar_logs[key])

        valid_sample_count = float(scalar_logs[f"{prefix}_valid_sample_count"])
        if valid_sample_count > 0.0:
            weighted_key = f"{signed_key}_weighted_sum"
            totals[weighted_key] = totals.get(weighted_key, 0.0) + (
                float(scalar_logs[signed_key]) * valid_sample_count
            )


def _finalize_v2_loss_diagnostics(totals: dict[str, float]) -> dict[str, float]:
    """Finalize exact-count fractions and valid-sample-weighted log ratios."""
    result: dict[str, float] = {}
    for prefix, signed_key, ratio_key in _V2_LOSS_DIAGNOSTIC_SPECS:
        support_count = totals.get(f"{prefix}_support_count", 0.0)
        support_total_count = totals.get(f"{prefix}_support_total_count", 0.0)
        valid_sample_count = totals.get(f"{prefix}_valid_sample_count", 0.0)
        sample_count = totals.get(f"{prefix}_sample_count", 0.0)
        if valid_sample_count > 0.0:
            signed_value = (
                totals.get(f"{signed_key}_weighted_sum", float("nan"))
                / valid_sample_count
            )
        else:
            signed_value = float("nan")

        result[signed_key] = signed_value
        result[ratio_key] = float(math.exp(signed_value))
        result[f"{prefix}_support_count"] = support_count
        result[f"{prefix}_support_total_count"] = support_total_count
        result[f"{prefix}_support_fraction"] = (
            support_count / support_total_count if support_total_count > 0.0 else 0.0
        )
        result[f"{prefix}_valid_sample_count"] = valid_sample_count
        result[f"{prefix}_sample_count"] = sample_count
        result[f"{prefix}_valid_sample_fraction"] = (
            valid_sample_count / sample_count if sample_count > 0.0 else 0.0
        )
    return result


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
        "tokenizer_objective_version": TOKENIZER_OBJECTIVE_VERSION,
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
    row_labels: str | None = None,
) -> None:
    if wandb_run is None:
        return
    caption = f"step={step}, num_frames={num_frames}"
    if sample_index is not None:
        caption += f", sample_index={sample_index}"
    if row_labels:
        caption += f", rows(top-to-bottom)={row_labels}"
    wandb_run.log(
        {
            f"{prefix}/render_triplet": wandb.Image(
                image,
                caption=caption,
            )
        },
        step=step,
    )


def render_head_outputs_for_visual(
    model: VGGT,
    images: torch.Tensor,
    sample: dict[str, Any],
    pose_enc: torch.Tensor,
    outputs: dict[str, torch.Tensor | None],
    device: torch.device,
) -> torch.Tensor:
    sky_mask = unwrap_tensor(sample["sky_mask"] if "sky_mask" in sample else sample["masks"]).to(device)
    timestamps = unwrap_tensor(sample["timestamps"]).to(device)
    return render_scene_from_outputs(
        model,
        images,
        sky_mask,
        timestamps,
        pose_enc.float(),
        outputs["depth"].float(),
        outputs["gs_map"].float(),
        outputs["gs_conf"].float(),
        outputs["dynamic_conf"].float(),
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
        args,
    )
    if was_training:
        tokenizer_runner.train()
    _, scalar_logs, _ = compute_losses(model, sample, args, feature_stats, teacher, student, global_step)
    noisy_weight = scalar_logs["noisy_weight"]
    if noisy_weight > 0.0:
        patch_grid = infer_patch_grid(images, teacher["image_patch"][0].shape[2])
        noisy_loss = compute_noisy_decoder_loss(
            tokenizer_runner,
            student["z"].detach(),
            teacher["image_patch"],
            feature_stats["std"],
            patch_grid,
            autocast_context(args, device),
            decoder_noise_tau=args.decoder_noise_tau,
            decoder_noise_distribution=args.decoder_noise_distribution,
        )
        scalar_logs["noisy"] = float(noisy_loss.detach().item())
        scalar_logs["loss"] += noisy_weight * scalar_logs["noisy"]
    if not all(key in student for key in ("depth", "gs_map", "gs_conf", "dynamic_conf")):
        return None, scalar_logs
    with torch.no_grad(), autocast_context(args, device):
        render_pred = render_head_outputs_for_visual(
            model,
            images,
            sample,
            teacher["pose_enc"],
            student,
            device,
        )
        render_direct_teacher = render_head_outputs_for_visual(
            model,
            images,
            sample,
            teacher["pose_enc"],
            teacher,
            device,
        )

    # Report the exact routing quantity that creates multi-frame dynamic
    # ghosts: GT dynamic pixels whose raw logit falls below DGGT's 0.5 static
    # threshold.  This makes an early/untrained student distinguishable from a
    # renderer or frozen-teacher failure without relying on visual judgment.
    dynamic_mask = unwrap_tensor(sample["dynamic_mask"]).to(device)[:, :, 0].float() >= 0.5
    for name, logits in (
        ("student", student["dynamic_conf"]),
        ("direct_teacher", teacher["dynamic_conf"]),
    ):
        predicted_dynamic = logits.float().squeeze(-1) >= 0.5
        true_positive = (predicted_dynamic & dynamic_mask).sum().float()
        false_negative = (~predicted_dynamic & dynamic_mask).sum().float()
        false_positive = (predicted_dynamic & ~dynamic_mask).sum().float()
        gt_positive = dynamic_mask.sum().float().clamp_min(1.0)
        union = (predicted_dynamic | dynamic_mask).sum().float().clamp_min(1.0)
        scalar_logs[f"dynamic_recall_{name}"] = float((true_positive / gt_positive).item())
        scalar_logs[f"dynamic_iou_{name}"] = float((true_positive / union).item())
        scalar_logs[f"dynamic_gt_static_route_{name}"] = float((false_negative / gt_positive).item())
        scalar_logs[f"dynamic_precision_{name}"] = float(
            (true_positive / (true_positive + false_positive).clamp_min(1.0)).item()
        )

    # The raw observation is the only GT.  The direct frozen DGGT render is a
    # separately labelled ceiling diagnostic, never a substitute for GT.
    grid = build_stage_a_validation_grid(
        images[0],
        render_pred[0],
        render_direct_teacher[0],
        max_frames=args.max_frames,
    )
    return grid, scalar_logs


def run_cached_visualization_eval(
    model: VGGT,
    tokenizer_runner: nn.Module,
    batch: dict[str, Any],
    args: argparse.Namespace,
    feature_stats: dict[str, torch.Tensor],
    levels: tuple[int, ...],
    global_step: int,
    device: torch.device,
) -> tuple[Image.Image | None, dict[str, float]]:
    sample = batch["sample"]
    predictions = batch["predictions"]
    images = unwrap_tensor(sample["images_clean"]).to(device)
    was_training = tokenizer_runner.training
    tokenizer_runner.eval()
    teacher = get_cached_teacher_outputs(predictions, levels, device)
    student = build_student_outputs_from_patch_teacher(
        model,
        tokenizer_runner,
        images,
        levels,
        teacher,
        autocast_context(args, device),
        args,
        force_decode_heads=True,
    )
    if was_training:
        tokenizer_runner.train()
    _, scalar_logs, _ = compute_losses(model, sample, args, feature_stats, teacher, student, global_step)
    noisy_weight = scalar_logs["noisy_weight"]
    if noisy_weight > 0.0:
        patch_grid = infer_patch_grid(images, teacher["image_patch"][0].shape[2])
        noisy_loss = compute_noisy_decoder_loss(
            tokenizer_runner,
            student["z"].detach(),
            teacher["image_patch"],
            feature_stats["std"],
            patch_grid,
            autocast_context(args, device),
            decoder_noise_tau=args.decoder_noise_tau,
            decoder_noise_distribution=args.decoder_noise_distribution,
        )
        scalar_logs["noisy"] = float(noisy_loss.detach().item())
        scalar_logs["loss"] += noisy_weight * scalar_logs["noisy"]
    if not all(key in student for key in ("depth", "gs_map", "gs_conf", "dynamic_conf")):
        return None, scalar_logs
    with torch.no_grad(), autocast_context(args, device):
        gt_outputs = build_head_outputs_from_patch_features(
            model,
            images,
            levels,
            teacher,
            teacher["image_patch"],
        )
        render_gt = render_head_outputs_for_visual(
            model,
            images,
            sample,
            teacher["pose_enc"],
            gt_outputs,
            device,
        )
        render_pred = render_head_outputs_for_visual(
            model,
            images,
            sample,
            teacher["pose_enc"],
            student,
            device,
        )
    grid = build_triplet_grid(render_gt[0], render_pred[0], max_frames=args.max_frames)
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


def parse_cache_mode_filter(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    modes = [item.strip() for item in str(value).split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in ("mode_a", "mode_b")]
    if invalid:
        raise ValueError(f"Invalid --cache_mode_filter values: {invalid}")
    return modes


def normalize_cache_dirs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(item) for item in value if item is not None]


class RoundRobinDatasetView:
    """Small index adapter used to balance multiple datasets during stats passes."""

    def __init__(self, datasets: list[Any]) -> None:
        if len(datasets) == 0:
            raise ValueError("RoundRobinDatasetView requires at least one dataset")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("RoundRobinDatasetView does not support empty datasets")
        self.datasets = list(datasets)

    def __len__(self) -> int:
        return max(len(dataset) for dataset in self.datasets) * len(self.datasets)

    def __getitem__(self, idx: int) -> Any:
        dataset_idx = int(idx) % len(self.datasets)
        item_idx = (int(idx) // len(self.datasets)) % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][item_idx]


def deterministic_vis_index(
    *,
    base_seed: int,
    vis_test_index: int,
    vis_count: int,
    source_name: str,
    dataset_len: int,
) -> int:
    if dataset_len <= 0:
        raise ValueError("dataset_len must be positive")
    source_offsets = {
        "raw": 101,
        "cache": 151,
        "cache_mode_a": 211,
        "cache_mode_b": 307,
    }
    generator = torch.Generator()
    generator.manual_seed(
        int(base_seed)
        + 1_000_003 * int(vis_count + 1)
        + 10_007 * int(vis_test_index)
        + source_offsets.get(str(source_name), 401)
    )
    return int(torch.randint(0, int(dataset_len), (1,), generator=generator).item())


def build_cached_dataset(
    args: argparse.Namespace,
    *,
    validation: bool = False,
    mode_filter: list[str] | None = None,
) -> Any:
    from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset

    class TokenizerFlowCacheDataset(WaymoFlowCacheDataset):
        """Lightweight flow-cache reader for tokenizer Stage-B."""

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self._getitem_with_cache_read_retry(idx, self._load_tokenizer_item_at_index)

        def _load_tokenizer_item_at_index(self, idx: int) -> dict[str, Any]:
            entry = self.entries[idx]
            cache_path = Path(entry["cache_path"])
            payload, subset_t, subset_payload = self._load_payload_for_sample(
                cache_path,
                entry,
                consumer="tokenizer_stage_b",
            )
            mode_kind = str(payload["mode_kind"])

            sample = self._build_sample(payload, subset_payload)
            sample["mode_kind"] = mode_kind
            sample["cache_index"] = int(entry.get("index", payload.get("meta", {}).get("manifest_index", -1)))
            predictions = self._build_predictions(payload, subset_payload)
            pass2_payload = payload.get("pass2_splatted_tok_low")
            if pass2_payload is None:
                raise RuntimeError(
                    f"Cache {cache_path} missing pass2_splatted_tok_low payload; "
                    "Stage-B tokenizer training requires edited splat/blend tokens."
                )
            tokenizer_teacher_levels = self._subset_pass2_splatted_tok_low(
                pass2_payload,
                subset_payload,
                dtype=self.lut_dtype,
            )
            return {
                "sample": sample,
                "predictions": predictions,
                "tokenizer_teacher_levels": tokenizer_teacher_levels,
                "mode_kind": mode_kind,
                "subset_frames": subset_t,
                "cache_path": str(cache_path),
                "cache_schema_version": int(payload["schema_version"]),
            }

    manifest_path = args.cache_manifest_path
    cache_root = normalize_cache_dirs(args.cache_dir)
    split = args.cache_split
    if len(cache_root) > 0:
        raise ValueError(
            "Cached tokenizer Stage-B requires explicit JSONL manifests; "
            "use --cache_manifest_path instead of --cache_dir."
        )
    if manifest_path is None:
        raise ValueError("Cached tokenizer Stage-B requires explicit --cache_manifest_path.")

    return TokenizerFlowCacheDataset(
        cache_root=None,
        manifest_path=manifest_path,
        split=split,
        # Tokenizer heads require all samples in a batch to have the same S.
        # Use the requested max length as a fixed Stage-B subsequence length.
        min_frames=args.max_frames,
        max_frames=args.max_frames,
        seed=args.seed + (100000 if validation else 0),
        lut_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32,
        mode_filter=mode_filter if mode_filter is not None else parse_cache_mode_filter(args.cache_mode_filter),
        mmap_plain_cache=not bool(args.no_mmap_plain_cache),
    )


def main() -> None:
    args = build_argparser().parse_args()
    validate_tokenizer_v2_objective_args(args)
    args.tokenizer_objective_version = TOKENIZER_OBJECTIVE_VERSION

    device, local_rank, world_size = setup_distributed(args)
    if int(args.num_workers) > 0:
        torch.multiprocessing.set_sharing_strategy(str(args.mp_sharing_strategy))
    seed_everything(args.seed + get_rank())

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.raw_batch_size < 1:
        raise ValueError("raw_batch_size must be positive")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    if args.min_frames < 1 or args.max_frames < args.min_frames:
        raise ValueError("Invalid frame range")
    if args.init_tokenizer_path and args.resume_path:
        raise ValueError("Use either --init_tokenizer_path or --resume_path, not both")
    cache_dirs = normalize_cache_dirs(args.cache_dir)
    use_mixed_teacher = bool(args.stage_b_mix_raw)
    if len(cache_dirs) > 0:
        raise ValueError(
            "Cached tokenizer Stage-B no longer supports cache directory discovery; "
            "pass --cache_manifest_path explicitly."
        )
    use_cached_teacher = (not use_mixed_teacher) and (args.cache_manifest_path is not None)
    if use_mixed_teacher or use_cached_teacher:
        if args.cache_manifest_path is None:
            raise ValueError("Cached tokenizer Stage-B requires --cache_manifest_path.")
    if use_mixed_teacher:
        requested_modes = parse_cache_mode_filter(args.cache_mode_filter)
        if requested_modes is not None and set(requested_modes) != {"mode_a", "mode_b"}:
            raise ValueError("--stage_b_mix_raw requires both cache modes; do not narrow --cache_mode_filter")

    log_dir = Path(args.log_dir)
    feature_stats_path = Path(args.feature_stats_path) if args.feature_stats_path else log_dir / "feature_stats.pt"
    dirs = ensure_log_dirs(log_dir)
    wandb_run = init_wandb_run(args, log_dir)
    if is_main_process():
        with (log_dir / "config.json").open("w") as f:
            json.dump(vars(args), f, indent=2)

    if use_mixed_teacher:
        train_datasets = {
            "raw": build_dataset(args, split="training"),
            "cache_mode_a": build_cached_dataset(args, validation=False, mode_filter=["mode_a"]),
            "cache_mode_b": build_cached_dataset(args, validation=False, mode_filter=["mode_b"]),
        }
        test_datasets = {
            "raw": build_dataset(args, split="validation"),
        }
        train_dataset = train_datasets["raw"]
        test_dataset = test_datasets["raw"]
    elif use_cached_teacher:
        train_dataset = build_cached_dataset(args, validation=False)
        test_dataset = build_cached_dataset(args, validation=True)
    else:
        train_dataset = build_dataset(args, split="training")
        test_dataset = build_dataset(args, split="validation")

    def try_build_cached_vis_dataset(mode_kind: str) -> Any | None:
        try:
            return build_cached_dataset(args, validation=False, mode_filter=[mode_kind])
        except (RuntimeError, ValueError) as exc:
            if is_main_process():
                print(f"[vis] skipping {mode_kind}: {exc}", flush=True)
            return None

    vis_datasets: dict[str, tuple[Any, bool]] = {}
    if use_mixed_teacher:
        mode_a_vis_dataset = try_build_cached_vis_dataset("mode_a")
        mode_b_vis_dataset = try_build_cached_vis_dataset("mode_b")
        vis_datasets = {
            "raw": (train_datasets["raw"], False),
        }
        if mode_a_vis_dataset is not None:
            vis_datasets["cache_mode_a"] = (mode_a_vis_dataset, True)
        if mode_b_vis_dataset is not None:
            vis_datasets["cache_mode_b"] = (mode_b_vis_dataset, True)
    elif use_cached_teacher:
        raw_vis_dataset = build_dataset(args, split="training")
        mode_a_vis_dataset = try_build_cached_vis_dataset("mode_a")
        mode_b_vis_dataset = try_build_cached_vis_dataset("mode_b")
        vis_datasets["raw"] = (raw_vis_dataset, False)
        if mode_a_vis_dataset is not None:
            vis_datasets["cache_mode_a"] = (mode_a_vis_dataset, True)
        if mode_b_vis_dataset is not None:
            vis_datasets["cache_mode_b"] = (mode_b_vis_dataset, True)
    else:
        vis_datasets = {"raw": (test_dataset, False)}
    vis_schedule = tuple(name for name in ("raw", "cache_mode_a", "cache_mode_b") if name in vis_datasets)

    if is_main_process() and use_mixed_teacher:
        raw_stats = train_datasets["raw"].clean_sample_stats
        print(
            f"[dataset/mixed] raw_train={len(train_datasets['raw'])} raw_val={len(test_datasets['raw'])} "
            f"cache_mode_a_train={len(train_datasets['cache_mode_a'])} "
            f"cache_mode_b_train={len(train_datasets['cache_mode_b'])} "
            "cache_vis_source=train_manifest "
            f"schedule=raw,mode_a,raw,mode_b raw_batch_size={args.raw_batch_size} "
            f"cache_batch_size={args.batch_size} raw_clean={raw_stats.get('clean_total', 0)}",
            flush=True,
        )
    elif is_main_process() and hasattr(train_dataset, "clean_sample_stats"):
        stats = train_dataset.clean_sample_stats
        print(
            f"[dataset] clean={stats.get('clean_total', 0)} edited={stats.get('edited_total', 0)} "
            f"train={stats.get('train_total', 0)} test={stats.get('val_total', 0)} "
            f"train_active={len(train_dataset)} test_active={len(test_dataset)}",
            flush=True,
        )
    elif is_main_process() and use_cached_teacher:
        print(
            f"[dataset/cache] train={len(train_dataset)} val={len(test_dataset)} "
            f"fixed_frames={args.max_frames} mode_filter={args.cache_mode_filter or 'all'}",
            flush=True,
        )

    def make_raw_sampler(dataset: Any, *, batch_size: int, seed_offset: int = 0):
        return VariableLengthDistributedSampler(
            dataset,
            min_num_frames=args.min_frames,
            max_num_frames=args.max_frames,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=get_rank(),
            shuffle=True,
            seed=args.seed + seed_offset,
        )

    def make_cache_sampler(dataset: Any, *, seed_offset: int = 0):
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=get_rank(),
            shuffle=True,
            seed=args.seed + seed_offset,
            drop_last=False,
        )

    def make_loader(dataset: Any, sampler_obj: Any, *, batch_size: int, collate_fn: Any) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler_obj,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            collate_fn=collate_fn,
            **dataloader_runtime_kwargs(args),
        )

    if use_mixed_teacher:
        samplers = {
            "raw": make_raw_sampler(train_datasets["raw"], batch_size=args.raw_batch_size, seed_offset=0),
            "cache_mode_a": make_cache_sampler(train_datasets["cache_mode_a"], seed_offset=11),
            "cache_mode_b": make_cache_sampler(train_datasets["cache_mode_b"], seed_offset=23),
        }
        dataloaders = {
            "raw": make_loader(
                train_datasets["raw"],
                samplers["raw"],
                batch_size=args.raw_batch_size,
                collate_fn=tokenizer_collate_fn,
            ),
            "cache_mode_a": make_loader(
                train_datasets["cache_mode_a"],
                samplers["cache_mode_a"],
                batch_size=args.batch_size,
                collate_fn=tokenizer_cache_collate_fn,
            ),
            "cache_mode_b": make_loader(
                train_datasets["cache_mode_b"],
                samplers["cache_mode_b"],
                batch_size=args.batch_size,
                collate_fn=tokenizer_cache_collate_fn,
            ),
        }
    elif use_cached_teacher:
        sampler = make_cache_sampler(train_dataset)
        collate_fn = tokenizer_cache_collate_fn
        dataloader = make_loader(train_dataset, sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    else:
        sampler = make_raw_sampler(train_dataset, batch_size=args.batch_size)
        collate_fn = tokenizer_collate_fn
        dataloader = make_loader(train_dataset, sampler, batch_size=args.batch_size, collate_fn=collate_fn)

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
    if use_mixed_teacher:
        feature_stats = get_feature_stats_mixed(model, train_datasets, args, device, levels, feature_stats_path)
    elif use_cached_teacher:
        feature_stats = get_feature_stats_from_cache(train_dataset, args, device, levels, feature_stats_path)
    else:
        feature_stats = get_feature_stats(model, train_dataset, args, device, levels, feature_stats_path)
    if is_main_process():
        print("[init] feature stats ready", flush=True)

    if is_main_process() and len(vis_schedule) > 0:
        vis_desc = ", ".join(
            f"{name}:n={len(vis_datasets[name][0])}" for name in vis_schedule
        )
        print(
            f"[vis] validation visualization schedule={vis_schedule}; "
            f"sample index is random per source/vis step (seed_offset={args.vis_test_index}); {vis_desc}",
            flush=True,
        )

    tokenizer_runner: nn.Module = TokenizerTrainWrapper(model.scene_tokenizer).to(device)
    vis_tokenizer_runner = TokenizerTrainWrapper(model.scene_tokenizer).to(device)
    tokenizer_runner.train()
    if world_size > 1:
        noisy_decoder_only = args.lambda_noisy > 0.0 and args.noisy_start_step < args.max_steps
        tokenizer_runner = DDP(
            tokenizer_runner,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=noisy_decoder_only,
            gradient_as_bucket_view=True,
        )

    optimizer = AdamW(model.scene_tokenizer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        warmup = min((step + 1) / max(args.warmup_steps, 1), 1.0)
        progress = min(step / max(args.max_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return warmup * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    global_step = 0

    if args.init_tokenizer_path:
        if is_main_process():
            print(f"[init] loading tokenizer weights: {args.init_tokenizer_path}", flush=True)
        init_payload = torch.load(args.init_tokenizer_path, map_location="cpu")
        objective_source = require_tokenizer_v2_checkpoint(
            init_payload, path=args.init_tokenizer_path
        )
        if is_main_process():
            print(f"[init] tokenizer objective: {objective_source}", flush=True)
        state_dict = extract_scene_tokenizer_state_dict(init_payload, args.init_tokenizer_path)
        model.scene_tokenizer.load_state_dict(state_dict, strict=True)

    elif args.resume_path:
        if is_main_process():
            print(f"[resume] loading training state: {args.resume_path}", flush=True)
        resume_payload = torch.load(args.resume_path, map_location="cpu")
        objective_source = require_tokenizer_v2_checkpoint(
            resume_payload, path=args.resume_path
        )
        if is_main_process():
            print(f"[resume] tokenizer objective: {objective_source}", flush=True)
        required_keys = {"scene_tokenizer", "optimizer", "scheduler", "global_step"}
        if not isinstance(resume_payload, dict) or not required_keys.issubset(resume_payload.keys()):
            missing = sorted(required_keys.difference(resume_payload.keys() if isinstance(resume_payload, dict) else ()))
            raise ValueError(
                f"--resume_path requires a full training checkpoint with {sorted(required_keys)}; "
                f"missing {missing}. Use --init_tokenizer_path to load tokenizer weights only."
            )
        state_dict = extract_scene_tokenizer_state_dict(resume_payload, args.resume_path)
        model.scene_tokenizer.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        global_step = int(resume_payload["global_step"])

    try:
        start_time = time.time()
        epoch = 0
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        accum_log_sums: dict[str, float] = {}
        accum_v2_diagnostic_totals: dict[str, float] = {}
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
        mixed_schedule = ("raw", "cache_mode_a", "raw", "cache_mode_b")
        mixed_step = 0
        mixed_epochs = {name: 0 for name in mixed_schedule}
        mixed_iters: dict[str, Any] = {}
        if use_mixed_teacher:
            for name, sampler_obj in samplers.items():
                sampler_obj.set_epoch(0)
                mixed_iters[name] = iter(dataloaders[name])
        else:
            epoch = 0
            sampler.set_epoch(epoch)
            dataloader_iter = iter(dataloader)
            num_batches = len(dataloader)
            batch_idx = 0

        def next_mixed_batch(source_name: str) -> Any:
            try:
                return next(mixed_iters[source_name])
            except StopIteration:
                mixed_epochs[source_name] += 1
                samplers[source_name].set_epoch(mixed_epochs[source_name])
                mixed_iters[source_name] = iter(dataloaders[source_name])
                return next(mixed_iters[source_name])

        accum_source_counts: dict[str, int] = {}
        while global_step < args.max_steps:
            if use_mixed_teacher:
                source_name = mixed_schedule[mixed_step % len(mixed_schedule)]
                mixed_step += 1
                sample = next_mixed_batch(source_name)
                is_last_microbatch = False
            else:
                try:
                    sample = next(dataloader_iter)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    dataloader_iter = iter(dataloader)
                    num_batches = len(dataloader)
                    batch_idx = 0
                    sample = next(dataloader_iter)
                source_name = "cache" if use_cached_teacher else "raw"
                batch_idx += 1
                is_last_microbatch = batch_idx == num_batches

            use_cache_batch = source_name.startswith("cache")
            if use_cache_batch:
                train_sample = sample["sample"]
                cached_predictions = sample["predictions"]
            else:
                train_sample = sample
                cached_predictions = None

            images = unwrap_tensor(train_sample["images_clean"]).to(device)
            local_batch_size = int(images.shape[0])
            num_frames = images.shape[1]

            accum_count += 1
            accum_source_counts[source_name] = accum_source_counts.get(source_name, 0) + 1
            should_step = accum_count >= args.grad_accum_steps or is_last_microbatch

            sync_context = nullcontext()
            if world_size > 1 and not should_step:
                sync_context = tokenizer_runner.no_sync()

            with sync_context:
                if use_cache_batch:
                    teacher = get_cached_teacher_outputs(cached_predictions, levels, device)
                    student = build_student_outputs_from_patch_teacher(
                        model,
                        tokenizer_runner,
                        images,
                        levels,
                        teacher,
                        autocast_context(args, device),
                        args,
                    )
                else:
                    teacher = get_teacher_outputs(model, images, levels, args, device)
                    student = build_student_outputs(
                        model,
                        tokenizer_runner,
                        images,
                        levels,
                        teacher,
                        autocast_context(args, device),
                        args,
                    )
                total_loss, scalar_logs, aux = compute_losses(
                    model,
                    train_sample,
                    args,
                    feature_stats,
                    teacher,
                    student,
                    global_step,
                )
                del aux
                (total_loss / float(args.grad_accum_steps)).backward()
                noisy_weight = scalar_logs["noisy_weight"]
                if noisy_weight > 0.0:
                    z_detached = student["z"].detach()
                    teacher_image_patch = teacher["image_patch"]
                    patch_grid = infer_patch_grid(images, teacher_image_patch[0].shape[2])
                    del teacher, student, total_loss
                    noisy_loss = compute_noisy_decoder_loss(
                        tokenizer_runner,
                        z_detached,
                        teacher_image_patch,
                        feature_stats["std"],
                        patch_grid,
                        autocast_context(args, device),
                        decoder_noise_tau=args.decoder_noise_tau,
                        decoder_noise_distribution=args.decoder_noise_distribution,
                    )
                    scalar_logs["noisy"] = float(noisy_loss.detach().item())
                    scalar_logs["loss"] += noisy_weight * scalar_logs["noisy"]
                    (noisy_weight * noisy_loss / float(args.grad_accum_steps)).backward()
                    del noisy_loss, z_detached, teacher_image_patch
                else:
                    del teacher, student, total_loss

            _accumulate_v2_loss_diagnostics(accum_v2_diagnostic_totals, scalar_logs)
            for key, value in scalar_logs.items():
                if key in _V2_LOSS_DIAGNOSTIC_LOG_KEYS:
                    continue
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
            step_logs.update(_finalize_v2_loss_diagnostics(accum_v2_diagnostic_totals))
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
                    ghost=f"{step_logs['ghost_static']:.4g}",
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
                for name, count in accum_source_counts.items():
                    log_metrics[f"source_{name}_microbatches"] = float(count)
                log_wandb_scalars(wandb_run, log_metrics, global_step)

            vis_count = global_step // max(args.vis_every, 1)
            if use_cached_teacher and not use_mixed_teacher:
                # Step 0 is skipped for cached training, so make the first
                # emitted visualization use the first schedule entry.
                vis_count = max(0, vis_count - 1)
            if len(vis_schedule) > 0:
                vis_source = vis_schedule[vis_count % len(vis_schedule)]
                vis_dataset, vis_is_cache = vis_datasets[vis_source]
            else:
                vis_source = "none"
                vis_dataset = []
                vis_is_cache = False

            should_run_vis = len(vis_dataset) > 0 and global_step % args.vis_every == 0
            if use_cached_teacher and not use_mixed_teacher and global_step == 0:
                should_run_vis = False
            if should_run_vis and is_distributed():
                dist.barrier()
            if is_main_process() and should_run_vis:
                vis_index = deterministic_vis_index(
                    base_seed=args.seed,
                    vis_test_index=args.vis_test_index,
                    vis_count=vis_count,
                    source_name=vis_source,
                    dataset_len=len(vis_dataset),
                )
                if vis_is_cache:
                    vis_sample = tokenizer_cache_collate_fn([vis_dataset[vis_index]])
                else:
                    vis_sample = tokenizer_collate_fn([vis_dataset[(vis_index, args.max_frames)]])
                with torch.no_grad():
                    if vis_is_cache:
                        grid, vis_logs = run_cached_visualization_eval(
                            model,
                            vis_tokenizer_runner,
                            vis_sample,
                            args,
                            feature_stats,
                            levels,
                            global_step,
                            device,
                        )
                    else:
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
                    grid.save(
                        dirs["vis"]
                        / f"validation_step_{global_step:06d}_{vis_source}_sample_{vis_index:06d}.png"
                    )
                    vis_num_frames = int(
                        vis_sample["sample"]["num_frames"][0].item()
                        if vis_is_cache
                        else vis_sample["num_frames"][0].item()
                    )
                    log_wandb_visual(
                        wandb_run,
                        grid,
                        global_step,
                        vis_num_frames,
                        prefix="validation",
                        sample_index=vis_index,
                        row_labels=(
                            "student render / cached teacher render / absolute difference"
                            if vis_is_cache
                            else "student render / raw RGB GT / direct DGGT render / absolute difference"
                        ),
                    )
                if vis_logs:
                    vis_logs["sample_index"] = float(vis_index)
                    vis_logs["num_frames"] = float(
                        vis_sample["sample"]["num_frames"][0].item()
                        if vis_is_cache
                        else vis_sample["num_frames"][0].item()
                    )
                    log_wandb_scalars(
                        wandb_run,
                        vis_logs,
                        global_step,
                        prefix="validation",
                    )
            if should_run_vis and is_distributed():
                dist.barrier()

            global_step += 1
            if train_pbar is not None:
                train_pbar.update(1)

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

            accum_count = 0
            accum_log_sums = {}
            accum_v2_diagnostic_totals = {}
            accum_num_frames = 0.0
            accum_local_batch = 0.0
            accum_source_counts = {}

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
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
