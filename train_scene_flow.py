"""T1 SceneFlow training entry point.

Reads the offline Phase-4.5 cache, drives `FlowFeatureAssembler` per step, and
computes a rectified-flow-style loss against a `SceneFlowMatching` module.

DDP scaffolding follows `train_tokenizer.py`. Visualization every `--vis_every`
steps dumps the same image set as `inference_scene_editor.py --dump_features`.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from datasets.waymo_flow_cache_dataset import SUPPORTED_CACHE_SCHEMA_VERSIONS, WaymoFlowCacheDataset
from dggt.losses.flow_losses import build_rectified_flow_target, compute_total_loss
from dggt.models.flow_feature_assembler import FlowFeatureAssembler
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.feature_stats import load_into_buffers
from dggt.utils.flow_cache_io import (
    is_chunked_flow_cache,
    load_chunked_flow_cache_subset,
    load_chunked_flow_cache_summary,
    load_flow_cache,
)
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.tokens import reattach_special_tokens, replace_selected_levels, select_patch_pyramid
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from train_scene_flow_pretrain import (
    TOKENIZER_LEVELS,
    _image_grid,
    _latent_pca_grid,
    _mask_grid,
    _normalized_mask_grid,
    _render_gs_map_rgb,
    _semantic_logits_to_sky_mask,
    _sky_mask_image_grid,
    load_dggt_aggregator_and_tokenizer,
    split_image_tokens_for_heads,
)


# ---------------------------------------------------------------------- #
# DDP + misc utilities (mirrored from train_tokenizer.py)                #
# ---------------------------------------------------------------------- #
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args) -> tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
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
    import random

    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def init_wandb(args: argparse.Namespace, log_dir: Path):
    if not args.wandb or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or remove --wandb.") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        dir=str(log_dir),
        config=vars(args),
    )


def log_wandb(run, metrics: dict[str, float], step: int, prefix: str) -> None:
    if run is None:
        return
    run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)


def autocast_context(args, device: torch.device):
    if args.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


@torch.no_grad()
def materialize_ema_state_dict(scene_flow: nn.Module, ema: EMAModel) -> dict[str, torch.Tensor]:
    sf = unwrap_ddp(scene_flow)
    params = list(sf.parameters())
    ema.store(params)
    ema.copy_to(params)
    try:
        return {key: value.detach().cpu().clone() for key, value in sf.state_dict().items()}
    finally:
        ema.restore(params)


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


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def _load_state_dict_checked(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    path: str,
    label: str,
) -> tuple[int, int]:
    missing, unexpected = module.load_state_dict(_strip_module_prefix(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} from {path} is incompatible: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    return len(missing), len(unexpected)


def load_scene_flow_warm_start(
    scene_flow: nn.Module,
    pretrain_path: str | None,
    *,
    use_ema: bool = True,
) -> dict[str, Any] | None:
    if not pretrain_path:
        return None

    sf = unwrap_ddp(scene_flow)
    payload = torch.load(pretrain_path, map_location="cpu")
    info: dict[str, Any] = {
        "path": pretrain_path,
        "step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "ema_used": False,
    }

    if use_ema:
        if isinstance(payload, dict) and "ema_scene_flow_state_dict" in payload:
            _load_state_dict_checked(
                sf,
                payload["ema_scene_flow_state_dict"],
                path=pretrain_path,
                label="EMA SceneFlow state_dict",
            )
            info["ema_used"] = True
            info["source"] = "ema_scene_flow_state_dict"
            return info

        if isinstance(payload, dict) and payload.get("is_ema_weights") and "scene_flow" in payload:
            _load_state_dict_checked(
                sf,
                payload["scene_flow"],
                path=pretrain_path,
                label="EMA SceneFlow weights-only state_dict",
            )
            info["ema_used"] = True
            info["source"] = "ema_weights_only"
            return info

        if isinstance(payload, dict) and "ema_scene_flow" in payload:
            if "scene_flow" not in payload:
                raise ValueError(f"{pretrain_path} has ema_scene_flow but no scene_flow buffers to initialize.")
            _load_state_dict_checked(
                sf,
                payload["scene_flow"],
                path=pretrain_path,
                label="raw SceneFlow state_dict",
            )
            ema = EMAModel(sf.parameters())
            ema.load_state_dict(payload["ema_scene_flow"])
            ema.copy_to(sf.parameters())
            info["ema_used"] = True
            info["source"] = "ema_scene_flow"
            return info

        raise ValueError(
            f"{pretrain_path} does not contain EMA SceneFlow weights. "
            "Use the full pretrain_step{N}.pt checkpoint, a new "
            "pretrain_step{N}_ema_weights_only.pt export, or pass "
            "--no_scene_flow_pretrain_ema to explicitly load raw weights."
        )

    if isinstance(payload, dict) and "scene_flow" in payload:
        state = payload["scene_flow"]
        info["source"] = "scene_flow"
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        info["source"] = "state_dict"
    else:
        state = payload
        info["source"] = "raw_state_dict"
    _load_state_dict_checked(sf, state, path=pretrain_path, label="SceneFlow state_dict")
    return info


def _infer_cache_patch_grid(dataset: WaymoFlowCacheDataset) -> tuple[int, int]:
    if len(dataset.entries) == 0:
        raise RuntimeError("Cannot infer patch grid from an empty cache dataset.")

    def _load_patch_grid(idx: int) -> tuple[int, int]:
        entry = dataset.entries[idx]
        cache_path = entry.get("cache_path")
        if cache_path is None:
            raise KeyError("Cache dataset entry is missing 'cache_path'.")
        if is_chunked_flow_cache(cache_path):
            summary = load_chunked_flow_cache_summary(cache_path)
            patch_grid = summary.get("patch_grid")
            if patch_grid is None or len(patch_grid) != 2:
                raise KeyError(f"Chunked cache payload {cache_path} is missing patch_grid=(H,W).")
            return (int(patch_grid[0]), int(patch_grid[1]))
        payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
        WaymoFlowCacheDataset._validate_v6_payload(
            payload,
            cache_path=Path(cache_path),
            entry=entry,
        )
        patch_grid = payload.get("meta", {}).get("patch_grid")
        if patch_grid is None or len(patch_grid) != 2:
            raise KeyError(f"Cache payload {cache_path} is missing meta.patch_grid=(H,W).")
        out = (int(patch_grid[0]), int(patch_grid[1]))
        if out[0] <= 0 or out[1] <= 0:
            raise ValueError(f"Invalid cache patch_grid {out} in {cache_path}.")
        return out

    return dataset._getitem_with_cache_read_retry(0, _load_patch_grid)


def split_train_val_entries(
    dataset: WaymoFlowCacheDataset,
    *,
    val_fraction: float,
    seed: int,
) -> WaymoFlowCacheDataset | None:
    entries = list(dataset.entries)
    if len(entries) < 2 or float(val_fraction) <= 0.0:
        return None
    val_count = int(round(len(entries) * float(val_fraction)))
    val_count = max(1, min(val_count, len(entries) - 1))
    indices = list(range(len(entries)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    val_indices = set(indices[:val_count])
    train_entries = [entry for idx, entry in enumerate(entries) if idx not in val_indices]
    val_entries = [entry for idx, entry in enumerate(entries) if idx in val_indices]

    dataset.entries = train_entries
    dataset._rng = random.Random(int(seed))
    val_dataset = copy.copy(dataset)
    val_dataset.entries = val_entries
    val_dataset._rng = random.Random(int(seed) + 1)
    return val_dataset


def _validate_item_patch_grid(
    asset_pass_result,
    assembler: FlowFeatureAssembler,
    cache_path: str | None = None,
) -> None:
    item_grid = tuple(int(v) for v in asset_pass_result.patch_grid)
    if item_grid != assembler.patch_grid:
        where = f" for {cache_path}" if cache_path else ""
        raise ValueError(
            f"Cache patch_grid{where} is {item_grid}, but assembler was initialized "
            f"with {assembler.patch_grid}. Use one training run per image geometry."
        )


# ---------------------------------------------------------------------- #
# CLI                                                                     #
# ---------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T1 SceneFlow training (Phase 9).")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Base DGGT checkpoint for tokenizer.")
    parser.add_argument("--tokenizer_ckpt_path", type=str, default=None,
                        help="Optional tokenizer checkpoint. Defaults to --ckpt_path.")
    parser.add_argument("--feature_stats_path", type=str, default=None,
                        help="Optional latent stats override. Warm-start checkpoints already carry these buffers.")
    parser.add_argument("--latent_dim", type=int, default=1024,
                        help="Tokenizer latent channels; must match pretrain warm-start and feature stats.")
    parser.add_argument("--scene_flow_pretrain_path", type=str, default=None,
                        help="Optional SceneFlow pretrain checkpoint for warm-start.")
    parser.add_argument("--scene_flow_pretrain_ema", dest="scene_flow_pretrain_ema",
                        action="store_true", default=True,
                        help="Load EMA weights from --scene_flow_pretrain_path. Enabled by default.")
    parser.add_argument("--no_scene_flow_pretrain_ema", dest="scene_flow_pretrain_ema",
                        action="store_false",
                        help="Load raw scene_flow weights from --scene_flow_pretrain_path.")
    parser.add_argument("--cache_root", type=str, default=None,
                        help="Offline feature cache root (Phase 4.5 output). Mutually exclusive with --manifest_path.")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Merged Mode A/B JSONL manifest from tools/build_flow_train_manifest.py.")
    parser.add_argument("--mode_filter", type=str, default=None,
                        help="When using --manifest_path, restrict to comma-sep modes (e.g. 'mode_a,mode_b').")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Hold out this fraction of the training cache entries for validation.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dggt-flow")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=50)

    parser.add_argument("--sequence_length", type=int, default=8,
                        help="Fixed number of frames sampled from each cache clip.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-process cache items per micro-batch.")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=1,
                        help="DataLoader batches prefetched per worker. Keep low because each cache item is large.")
    parser.add_argument("--no_persistent_workers", action="store_true",
                        help="Disable persistent DataLoader workers.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable DataLoader pin_memory. Disabled by default because cache items are GB-scale.")
    parser.add_argument("--mp_sharing_strategy", type=str, default="file_system",
                        choices=("file_system", "file_descriptor"),
                        help="Torch multiprocessing tensor sharing strategy for DataLoader workers.")
    parser.add_argument("--no_mmap_plain_cache", action="store_true",
                        help="Disable mmap=True when reading uncompressed torch cache files.")
    parser.add_argument("--no_batch_scene_flow", action="store_true",
                        help="Process cache items in a micro-batch serially instead of batching WanSceneFlow.")
    parser.add_argument(
        "--full_asset_lut_cache",
        action="store_true",
        help=(
            "Load all cached asset LUT levels. By default T1 loads only the last "
            "asset level because cached pass2_splatted_tok_low supplies z_splat "
            "and cross-attn uses only the last level."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--vis_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--val_log_images", type=int, default=4)
    parser.add_argument("--val_sample_steps", type=int, default=50)
    parser.add_argument("--no_val_render_rgb", action="store_true",
                        help="Skip validation 3DGS RGB renders and log latent/mask diagnostics only.")
    parser.add_argument("--no_val_ema", action="store_true",
                        help="Disable EMA weights for validation. Default matches pretrain: validate with EMA.")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--shift", type=float, default=3.0)

    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_preserve", type=float, default=1.0)
    parser.add_argument("--lambda_repa", type=float, default=0.5)
    parser.add_argument("--preserve_floor", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--uncond_drop_prob", type=float, default=0.1)
    parser.add_argument("--val_guidance_scales", type=str, default="")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--loss_weighting_scheme", type=str, default="none")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    return parser


# ---------------------------------------------------------------------- #
# Model setup                                                             #
# ---------------------------------------------------------------------- #
def _load_tokenizer(ckpt_path: str, device: torch.device) -> nn.Module:
    from dggt.models.vggt import VGGT

    model = VGGT().to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    # We only need scene_tokenizer; aggregator/heads stay offline.
    model.eval()
    tokenizer = model.scene_tokenizer.float()
    return tokenizer


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _move_v6_fast_path_inputs(
    item: dict[str, Any],
    mode_kind: str,
    device: torch.device,
) -> tuple[dict[str, Any] | None, list[torch.Tensor]]:
    schema_version = int(item.get("cache_schema_version", 0))
    if schema_version not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"Training item {item.get('cache_path', '<unknown>')} has "
            f"cache_schema_version={schema_version}; supported versions are "
            f"{SUPPORTED_CACHE_SCHEMA_VERSIONS}."
        )

    if mode_kind not in ("mode_a", "mode_b"):
        raise RuntimeError(
            f"Training item {item.get('cache_path', '<unknown>')} has "
            f"invalid mode_kind={mode_kind!r}."
        )

    phase1_localized_lite = item.get("phase1_localized")
    if mode_kind == "mode_a":
        if phase1_localized_lite is None:
            raise RuntimeError(
                f"Mode-A cache item {item.get('cache_path', '<unknown>')} missing phase1_localized."
            )
        phase1_localized_lite = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in phase1_localized_lite.items()
        }
    else:
        if phase1_localized_lite is not None:
            raise RuntimeError(
                f"Mode-B v6 item {item.get('cache_path', '<unknown>')} unexpectedly has phase1_localized."
            )
        phase1_localized_lite = None

    splatted_tok_low_cached = item.get("splatted_tok_low_cached")
    if splatted_tok_low_cached is None:
        raise RuntimeError(
            f"v6 item {item.get('cache_path', '<unknown>')} missing splatted_tok_low_cached."
        )
    return phase1_localized_lite, [t.to(device) for t in splatted_tok_low_cached]


def _flatten_fast_asset_kv(asset_pass_result, flow_inputs: dict[str, Any], device: torch.device) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    coverage = flow_inputs.get("phase1_coverage")
    if torch.is_tensor(coverage):
        coverage = coverage.to(device=device, dtype=torch.bool)
    phase4_slots = {int(v) for v in flow_inputs.get("phase4_slots", [])}
    for obj_key in sorted(int(k) for k in asset_pass_result.F_g_lut_asset.keys()):
        levels = asset_pass_result.F_g_lut_asset[obj_key]
        if not levels:
            continue
        lvl = levels[-1].to(device)
        if torch.is_tensor(coverage) and obj_key < int(coverage.shape[0]):
            cov = coverage[obj_key].to(device=device, dtype=lvl.dtype).view(1, -1, 1, 1)
            if cov.shape[1] == lvl.shape[1]:
                lvl = lvl * cov
        elif phase4_slots and obj_key not in phase4_slots:
            lvl = torch.zeros_like(lvl)
        B, S, P, C = lvl.shape
        chunks.append(lvl.reshape(B, S * P, C))
    if not chunks:
        return torch.zeros((1, 0, 3072), dtype=torch.float32, device=device)
    return torch.cat(chunks, dim=1)


def build_cached_flow_bundle(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    device: torch.device,
) -> Any:
    flow_inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in item["flow_inputs_cached"].items()
    }
    predictions = _move_predictions(item["predictions"], device)
    F_g_lut_scene = assembler._select_lut_scene(predictions)
    splatted_tok_low_cached = item.get("splatted_tok_low_cached")
    if splatted_tok_low_cached is None:
        raise RuntimeError(f"Fast cache item {item.get('cache_path', '<unknown>')} missing splatted_tok_low_cached.")
    splatted_tok_low = [t.to(device=device, dtype=F_g_lut_scene[0].dtype) for t in splatted_tok_low_cached]

    if assembler.scene_tokenizer is None:
        raise RuntimeError("FlowFeatureAssembler needs scene_tokenizer for cached SceneFlow inputs.")
    z_clean = assembler.scene_tokenizer.encode(F_g_lut_scene, patch_grid=assembler.patch_grid)
    z_splat = assembler.scene_tokenizer.encode(splatted_tok_low, patch_grid=assembler.patch_grid)

    M_preserve = flow_inputs["M_preserve"].to(device=device, dtype=torch.float32)
    M_source = flow_inputs["M_source"].to(device=device, dtype=torch.float32)
    M_dest = flow_inputs["M_dest"].to(device=device, dtype=torch.float32)
    scaffold_pooled = flow_inputs["scaffold_pooled"].to(device=device, dtype=torch.float32)
    scaffold_tok = unwrap_ddp(assembler.scaffold_packer).mlp(scaffold_pooled)

    B = int(z_clean.shape[0])
    base_t = assembler.noise_scheduler.sample_base_t(B, device=device)
    t_tok = assembler.noise_scheduler.build_t_tok(base_t, M_preserve, M_source, M_dest)
    z_init, eps_noise = assembler.noise_scheduler.compose_z_init(
        z_clean, z_splat, M_preserve, M_source, M_dest
    )

    asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
    F_asset_tokens = _flatten_fast_asset_kv(asset_pass_result, flow_inputs, device).to(dtype=z_clean.dtype)
    return SimpleNamespace(
        z_clean=z_clean,
        z_splat=z_splat,
        z_init=z_init,
        eps_noise=eps_noise,
        t_tok=t_tok,
        base_t=base_t,
        scaffold_tok=scaffold_tok,
        M_preserve=M_preserve,
        M_source=M_source,
        M_dest=M_dest,
        F_asset_tokens=F_asset_tokens,
        encoder_attention_mask=None,
        phase4_slots=list(flow_inputs.get("phase4_slots", [])),
        patch_grid=assembler.patch_grid,
        patch_start_idx=assembler.patch_start_idx,
        splatted_tok_low=splatted_tok_low,
        F_g_lut_scene=F_g_lut_scene,
    )


# ---------------------------------------------------------------------- #
# Train step                                                              #
# ---------------------------------------------------------------------- #
def build_flow_bundle(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    device: torch.device,
) -> Any:
    if item.get("flow_inputs_cached") is not None:
        return build_cached_flow_bundle(item, assembler, device)

    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["sample"].items()}
    predictions = _move_predictions(item["predictions"], device)
    asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
    _validate_item_patch_grid(asset_pass_result, assembler, item.get("cache_path"))
    cameras_dggt = {k: v.to(device) for k, v in item["cameras_dggt"].items()}
    mode_kind = str(item.get("mode_kind", sample.get("mode_kind", "mode_a")))
    mode_b_payload = item.get("mode_b")
    if mode_b_payload is not None:
        mode_b_payload = _move_mode_b(mode_b_payload, device)

    phase1_localized_lite, splatted_tok_low_cached = _move_v6_fast_path_inputs(
        item, mode_kind, device
    )

    return assembler(
        sample=sample,
        predictions=predictions,
        asset_pass_result=asset_pass_result,
        cameras_dggt=cameras_dggt,
        object_slots_spec="all",
        base_t=None,
        device=device,
        mode_kind=mode_kind,
        mode_b=mode_b_payload,
        phase1_localized_lite=phase1_localized_lite,
        splatted_tok_low_cached=splatted_tok_low_cached,
    )


def _maybe_drop_asset_kv(bundle, drop_prob: float) -> None:
    if float(drop_prob) <= 0.0 or not torch.is_tensor(bundle.F_asset_tokens):
        return
    if bundle.F_asset_tokens.shape[1] == 0:
        return
    if float(torch.rand((), device=bundle.F_asset_tokens.device).item()) >= float(drop_prob):
        return
    B, _, C = bundle.F_asset_tokens.shape
    bundle.F_asset_tokens = bundle.F_asset_tokens.new_zeros((B, 0, C))


def _pad_asset_tokens_for_batch(bundles: list[Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if len(bundles) == 0:
        raise ValueError("Cannot pad an empty bundle list.")
    token_lists = [bundle.F_asset_tokens for bundle in bundles]
    device = token_lists[0].device
    dtype = token_lists[0].dtype
    dim = int(token_lists[0].shape[-1])
    lengths = [int(tokens.shape[1]) for tokens in token_lists]
    max_len = max(lengths)
    batch = len(token_lists)
    if max_len == 0:
        return token_lists[0].new_zeros((batch, 0, dim)), None
    out = torch.zeros((batch, max_len, dim), device=device, dtype=dtype)
    mask = torch.zeros((batch, max_len), device=device, dtype=torch.bool)
    for row, tokens in enumerate(token_lists):
        n = int(tokens.shape[1])
        if n == 0:
            continue
        out[row, :n] = tokens.squeeze(0)
        mask[row, :n] = True
    if all(n == max_len for n in lengths):
        return out, None
    return out, mask


def _merge_bundles_for_scene_flow(bundles: list[Any]) -> tuple[Any, torch.Tensor | None, list[int]]:
    asset_tokens, asset_mask = _pad_asset_tokens_for_batch(bundles)
    lengths = [int(bundle.F_asset_tokens.shape[1]) for bundle in bundles]
    merged = SimpleNamespace(
        z_clean=torch.cat([bundle.z_clean for bundle in bundles], dim=0),
        z_splat=torch.cat([bundle.z_splat for bundle in bundles], dim=0),
        scaffold_tok=torch.cat([bundle.scaffold_tok for bundle in bundles], dim=0),
        M_preserve=torch.cat([bundle.M_preserve for bundle in bundles], dim=0),
        M_source=torch.cat([bundle.M_source for bundle in bundles], dim=0),
        M_dest=torch.cat([bundle.M_dest for bundle in bundles], dim=0),
        F_asset_tokens=asset_tokens,
        encoder_attention_mask=asset_mask,
        phase4_slots=[],
    )
    return merged, asset_mask, lengths


def train_step(
    item: dict[str, Any] | list[dict[str, Any]],
    assembler: FlowFeatureAssembler,
    scene_flow: nn.Module,
    scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(item, list):
        if len(item) == 0:
            raise ValueError("Received an empty training micro-batch.")
        if len(item) > 1 and not bool(getattr(args, "no_batch_scene_flow", False)):
            bundles = [build_flow_bundle(single, assembler, device) for single in item]
            if unwrap_ddp(scene_flow).training:
                for bundle_i in bundles:
                    _maybe_drop_asset_kv(bundle_i, args.uncond_drop_prob)
            bundle, asset_mask, asset_lengths = _merge_bundles_for_scene_flow(bundles)
            sf = unwrap_ddp(scene_flow)
            z_clean_n = sf.normalize(bundle.z_clean)
            z_splat_n = sf.normalize(bundle.z_splat)
            bundle.z_clean_n = z_clean_n

            target = build_rectified_flow_target(
                scheduler,
                z_clean_n,
                weighting_scheme=args.weighting_scheme,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                loss_weighting_scheme=args.loss_weighting_scheme,
            )

            use_repa = float(args.lambda_repa) != 0.0
            out = scene_flow(
                target.z_t,
                target.sigmas,
                z_splat_n,
                bundle.scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                bundle.F_asset_tokens,
                encoder_attention_mask=asset_mask,
                return_mid=use_repa,
            )
            if use_repa:
                v_pred, mid_repa = out
            else:
                v_pred, mid_repa = out, None
            loss, metrics = compute_total_loss(
                v_pred=v_pred,
                v_gt=target.v_gt,
                eps=target.eps,
                bundle=bundle,
                sd3_weights=target.weights,
                mid_repa=mid_repa,
                lambda_flow=args.lambda_flow,
                lambda_preserve=args.lambda_preserve,
                lambda_repa=args.lambda_repa,
                lambda_identity=0.0,
                identity_batch=False,
                preserve_floor=args.preserve_floor,
            )
            metrics.update({
                "edit_weight_mean": float((1.0 - bundle.M_preserve).mean().item()),
                "num_objects": sum(float(len(b.phase4_slots)) for b in bundles) / float(len(bundles)),
                "kv_tokens": sum(float(n) for n in asset_lengths) / float(len(asset_lengths)),
                "sigma_mean": float(target.sigmas.float().mean().item()),
                "micro_batch_size": float(len(item)),
            })
            return loss, metrics

        losses: list[torch.Tensor] = []
        metric_sums: dict[str, float] = {}
        for single in item:
            loss_i, metrics_i = train_step(single, assembler, scene_flow, scheduler, device, args)
            losses.append(loss_i)
            for key, value in metrics_i.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        scale = 1.0 / float(len(item))
        metrics = {key: value * scale for key, value in metric_sums.items()}
        metrics["micro_batch_size"] = float(len(item))
        return torch.stack(losses).mean(), metrics

    bundle = build_flow_bundle(item, assembler, device)
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean)
    z_splat_n = sf.normalize(bundle.z_splat)
    bundle.z_clean_n = z_clean_n
    if unwrap_ddp(scene_flow).training:
        _maybe_drop_asset_kv(bundle, args.uncond_drop_prob)

    target = build_rectified_flow_target(
        scheduler,
        z_clean_n,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        loss_weighting_scheme=args.loss_weighting_scheme,
    )

    use_repa = float(args.lambda_repa) != 0.0
    out = scene_flow(
        target.z_t,
        target.sigmas,
        z_splat_n,
        bundle.scaffold_tok,
        bundle.M_preserve,
        bundle.M_source,
        bundle.M_dest,
        bundle.F_asset_tokens,
        encoder_attention_mask=getattr(bundle, "encoder_attention_mask", None),
        return_mid=use_repa,
    )
    if use_repa:
        v_pred, mid_repa = out
    else:
        v_pred, mid_repa = out, None
    loss, metrics = compute_total_loss(
        v_pred=v_pred,
        v_gt=target.v_gt,
        eps=target.eps,
        bundle=bundle,
        sd3_weights=target.weights,
        mid_repa=mid_repa,
        lambda_flow=args.lambda_flow,
        lambda_preserve=args.lambda_preserve,
        lambda_repa=args.lambda_repa,
        lambda_identity=0.0,
        identity_batch=False,
        preserve_floor=args.preserve_floor,
    )
    metrics.update({
        "edit_weight_mean": float((1.0 - bundle.M_preserve).mean().item()),
        "num_objects": float(len(bundle.phase4_slots)),
        "kv_tokens": float(bundle.F_asset_tokens.shape[1]),
        "sigma_mean": float(target.sigmas.float().mean().item()),
    })
    return loss, metrics


def _move_predictions(predictions: dict, device: torch.device) -> dict:
    out: dict[str, Any] = {}
    for k, v in predictions.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if torch.is_tensor(x) else x for x in v] if v is not None else v
        elif v is None:
            out[k] = None
        else:
            out[k] = v
    return out


def _move_mode_b(mode_b: dict, device: torch.device) -> dict:
    out = dict(mode_b)
    for k in ("delete_mask", "delete_mask_per_frame_subset", "subset_frames",
              "delete_core_indices", "delete_shell_indices"):
        v = out.get(k)
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def _move_asset_pass(apr, device: torch.device):
    from dggt.models.asset_pass import AssetPassResult

    return AssetPassResult(
        patch_grid=apr.patch_grid,
        patch_start_idx=apr.patch_start_idx,
        object_keys=list(apr.object_keys),
        cameras_waymo={k: v.to(device) for k, v in apr.cameras_waymo.items()} if apr.cameras_waymo else {},
        F_g_lut_asset={k: [lv.to(device) for lv in v] for k, v in apr.F_g_lut_asset.items()},
        ptr_asset={k: [p.to(device) for p in v] for k, v in apr.ptr_asset.items()},
        G_asset_waymo={
            k: [{kk: vv.to(device) for kk, vv in g.items()} for g in v]
            for k, v in apr.G_asset_waymo.items()
        },
        G_asset_dggt=None
        if apr.G_asset_dggt is None
        else {
            k: [{kk: vv.to(device) for kk, vv in g.items()} for g in v]
            for k, v in apr.G_asset_dggt.items()
        },
        I_asset={k: v.to(device) for k, v in apr.I_asset.items()},
        A_asset={k: v.to(device) for k, v in apr.A_asset.items()},
        asset_pass_space=apr.asset_pass_space,
        fit_metrics=apr.fit_metrics,
    )


@torch.no_grad()
def cfg_sample_edit_latents(
    scene_flow: nn.Module,
    bundle,
    args,
    step: int,
    device: torch.device,
    guidance_scale: float,
) -> torch.Tensor:
    sf = unwrap_ddp(scene_flow)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=float(args.shift),
        invert_sigmas=True,
    )
    scheduler.set_timesteps(num_inference_steps=int(args.val_sample_steps), device=device)
    z_clean_n = sf.normalize(bundle.z_clean.float())
    z_splat_n = sf.normalize(bundle.z_splat.float())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))
    z = torch.empty_like(z_clean_n)
    z.normal_(generator=generator)
    batch_size = int(z.shape[0])
    F_asset = bundle.F_asset_tokens
    F_uncond = F_asset.new_zeros((batch_size, 0, F_asset.shape[-1]))
    encoder_attention_mask = getattr(bundle, "encoder_attention_mask", None)
    do_cfg = abs(float(guidance_scale) - 1.0) > 1e-6 and F_asset.shape[1] > 0

    for timestep in scheduler.timesteps:
        sigma = (timestep / scheduler.config.num_train_timesteps).to(device=device)
        sigma = sigma.expand(batch_size)
        v_cond = sf(
            z,
            sigma,
            z_splat_n,
            bundle.scaffold_tok,
            bundle.M_preserve,
            bundle.M_source,
            bundle.M_dest,
            F_asset,
            encoder_attention_mask=encoder_attention_mask,
            return_mid=False,
        )
        if do_cfg:
            v_uncond = sf(
                z,
                sigma,
                z_splat_n,
                bundle.scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                F_uncond,
                encoder_attention_mask=None,
                return_mid=False,
            )
            v = v_uncond + float(guidance_scale) * (v_cond - v_uncond)
        else:
            v = v_cond
        z = scheduler.step(model_output=v, timestep=timestep, sample=z, return_dict=False)[0]

    return z


def _validation_scales(args) -> list[float]:
    scales = [float(args.guidance_scale)]
    for raw in str(args.val_guidance_scales).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if all(abs(value - seen) > 1e-6 for seen in scales):
            scales.append(value)
    return scales


def _first_item(item: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(item, list):
        if len(item) == 0:
            raise ValueError("Received an empty collated batch.")
        return item[0]
    return item


def dataloader_runtime_kwargs(args) -> dict[str, Any]:
    if int(args.num_workers) <= 0:
        return {}
    return {
        "prefetch_factor": max(1, int(args.prefetch_factor)),
        "persistent_workers": not bool(args.no_persistent_workers),
    }


def _prepare_visualization_batch(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    images = sample.get("images", sample.get("images_clean"))
    masks = sample.get("masks", sample.get("sky_mask"))
    timestamps = sample.get("timestamps")
    if not torch.is_tensor(images):
        raise RuntimeError("Validation visualization sample is missing tensor images/images_clean.")
    if not torch.is_tensor(masks):
        raise RuntimeError("Validation visualization sample is missing tensor masks/sky_mask.")
    if not torch.is_tensor(timestamps):
        raise RuntimeError("Validation visualization sample is missing tensor timestamps.")

    if images.ndim == 4:
        images = images.unsqueeze(0)
    if masks.ndim == 4:
        masks = masks.unsqueeze(0)
    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"Expected visualization images [B,S,3,H,W], got {tuple(images.shape)}")
    if masks.ndim != 5:
        raise ValueError(f"Expected visualization masks [B,S,3,H,W], got {tuple(masks.shape)}")
    if timestamps.ndim != 2:
        raise ValueError(f"Expected visualization timestamps [B,S], got {tuple(timestamps.shape)}")
    return {
        "images": images.contiguous(),
        "masks": masks.contiguous(),
        "timestamps": timestamps.contiguous(),
    }


def load_validation_visualization_batch(
    item: dict[str, Any],
    dataset: WaymoFlowCacheDataset,
) -> dict[str, torch.Tensor]:
    """Return the raw batch fields needed by the 3DGS validation renderer."""
    if item.get("sample") is not None:
        return _prepare_visualization_batch(item["sample"])

    cache_path_raw = item.get("cache_path")
    if cache_path_raw is None:
        raise RuntimeError("Fast validation item is missing cache_path; cannot load RGB render inputs.")
    subset = item.get("subset_frames")
    if not torch.is_tensor(subset):
        subset = torch.as_tensor(subset, dtype=torch.long)
    subset = subset.detach().cpu().to(torch.long).contiguous()
    cache_path = Path(cache_path_raw)
    entry = {
        "cache_path": str(cache_path),
        "mode_kind": item.get("mode_kind", "unknown"),
    }

    if is_chunked_flow_cache(cache_path):
        payload = load_chunked_flow_cache_subset(
            cache_path,
            subset,
            consumer="tokenizer_stage_b",
        )
        subset_payload = torch.arange(int(subset.numel()), dtype=torch.long)
    else:
        payload = load_flow_cache(
            cache_path,
            map_location="cpu",
            weights_only=False,
            mmap=bool(getattr(dataset, "mmap_plain_cache", True)),
        )
        subset_payload = subset

    if not is_chunked_flow_cache(cache_path):
        WaymoFlowCacheDataset._validate_v6_payload(payload, cache_path=cache_path, entry=entry)
    sample = dataset._build_sample(payload, subset_payload)
    return _prepare_visualization_batch(sample)


def _save_rgb_validation_images(
    rgb_images: dict[str, torch.Tensor],
    out_dir: Path,
    paths: dict[str, Path],
    frames: int,
    suffix: str,
    *,
    only_generated: bool,
) -> None:
    skip_for_extra = {"input_rgb_gt", "tokenizer_recon_3dgs_rgb", "dggt_clean_3dgs_rgb"}
    for name, tensor in rgb_images.items():
        if only_generated and name in skip_for_extra:
            continue
        key = f"{name}{suffix}" if name.startswith("generated_") else name
        filename = f"{key}.jpg"
        path = out_dir / filename
        save_image_grid(tensor, path, nrow=frames)
        paths[key] = path


def _cuda_empty_cache_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def render_validation_rgb_gt_sky(
    batch: dict[str, torch.Tensor],
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render formal-training validation RGB with generated outputs composited over GT sky."""
    images = batch["images"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)
    frames = min(int(args.val_log_images), int(images.shape[1]))
    sf = unwrap_ddp(scene_flow)
    timestamps_raw = batch["timestamps"]
    timestamps = timestamps_raw[0] if torch.is_tensor(timestamps_raw) else torch.as_tensor(timestamps_raw[0])

    result: dict[str, torch.Tensor] = {}

    with autocast_context(args, device):
        outputs = vggt_model.get_aggregator_token_outputs(images)
        aggregated_tokens_list = outputs["aggregated_tokens_list"]
        image_tokens_list = outputs["image_tokens_list"]
        dino_token_list = outputs["dino_token_list"]
        patch_start_idx = int(outputs["patch_start_idx"])
        del outputs

        z_generated = sf.denormalize(z_generated_raw_n.float())
        decoded_patch_tokens = vggt_model.scene_tokenizer.decode(z_generated, patch_grid=args.patch_grid)
        del z_generated
        decoded_full_tokens = reattach_special_tokens(
            image_tokens_list,
            TOKENIZER_LEVELS,
            patch_start_idx,
            decoded_patch_tokens,
        )
        del decoded_patch_tokens
        generated_image_tokens = replace_selected_levels(
            image_tokens_list,
            TOKENIZER_LEVELS,
            decoded_full_tokens,
        )
        del decoded_full_tokens

        tokens_4 = select_patch_pyramid(image_tokens_list, TOKENIZER_LEVELS, patch_start_idx)
        z_recon = vggt_model.scene_tokenizer.encode(tokens_4, patch_grid=args.patch_grid)
        del tokens_4
        recon_patch_tokens = vggt_model.scene_tokenizer.decode(z_recon, patch_grid=args.patch_grid)
        del z_recon
        recon_full_tokens = reattach_special_tokens(
            image_tokens_list,
            TOKENIZER_LEVELS,
            patch_start_idx,
            recon_patch_tokens,
        )
        del recon_patch_tokens
        recon_image_tokens = replace_selected_levels(
            image_tokens_list,
            TOKENIZER_LEVELS,
            recon_full_tokens,
        )
        del recon_full_tokens

    with autocast_context(args, device):
        with torch.amp.autocast(device_type=device.type, enabled=False):
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            depth, _ = vggt_model.depth_head(aggregated_tokens_list, images, patch_start_idx)
            dynamic_conf, _ = vggt_model.instance_head(dino_token_list, images, patch_start_idx)
            clean_gs_map, clean_gs_conf = vggt_model.gs_head(image_tokens_list, images, patch_start_idx)

    del aggregated_tokens_list, dino_token_list, image_tokens_list

    result["dggt_clean_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model,
        images,
        masks,
        timestamps,
        pose_enc,
        depth,
        clean_gs_map,
        clean_gs_conf,
        dynamic_conf,
        device,
        frames,
        background_mode="sky",
        use_sky_mask=True,
    )
    del pose_enc, depth, dynamic_conf, clean_gs_map, clean_gs_conf
    _cuda_empty_cache_if_available()

    with autocast_context(args, device):
        gen_agg, gen_dino = split_image_tokens_for_heads(generated_image_tokens)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            raw_gs_map, raw_gs_conf = vggt_model.gs_head(generated_image_tokens, images, patch_start_idx)
            generated_pose_enc = vggt_model.camera_head(gen_agg)[-1]
            generated_depth, _ = vggt_model.depth_head(gen_agg, images, patch_start_idx)
            generated_dynamic_conf, _ = vggt_model.instance_head(gen_dino, images, patch_start_idx)
            generated_semantic_logits, _ = vggt_model.semantic_head(gen_dino, images, patch_start_idx)
            generated_sky_mask = _semantic_logits_to_sky_mask(generated_semantic_logits)

    del generated_image_tokens, gen_agg, gen_dino

    result["generated_pred_sky_mask"] = _sky_mask_image_grid(generated_sky_mask, frames)
    result["generated_raw_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model,
        images,
        masks,
        timestamps,
        generated_pose_enc,
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        device,
        frames,
        background_mode="sky",
        use_sky_mask=True,
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
    _cuda_empty_cache_if_available()

    with autocast_context(args, device):
        recon_agg, recon_dino = split_image_tokens_for_heads(recon_image_tokens)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            recon_pose_enc = vggt_model.camera_head(recon_agg)[-1]
            recon_depth, _ = vggt_model.depth_head(recon_agg, images, patch_start_idx)
            recon_dynamic_conf, _ = vggt_model.instance_head(recon_dino, images, patch_start_idx)
            recon_gs_map, recon_gs_conf = vggt_model.gs_head(recon_image_tokens, images, patch_start_idx)

    del recon_image_tokens, recon_agg, recon_dino

    result["tokenizer_recon_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model,
        images,
        masks,
        timestamps,
        recon_pose_enc,
        recon_depth,
        recon_gs_map,
        recon_gs_conf,
        recon_dynamic_conf,
        device,
        frames,
        background_mode="sky",
        use_sky_mask=True,
    )
    del recon_pose_enc, recon_depth, recon_dynamic_conf, recon_gs_map, recon_gs_conf
    _cuda_empty_cache_if_available()

    result["input_rgb_gt"] = _image_grid(images, frames)
    return result


@torch.no_grad()
def render_validation_generated_rgb_gt_sky(
    batch: dict[str, torch.Tensor],
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render the generated branch for secondary CFG scales over GT sky."""
    images = batch["images"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)
    frames = min(int(args.val_log_images), int(images.shape[1]))
    sf = unwrap_ddp(scene_flow)
    timestamps_raw = batch["timestamps"]
    timestamps = timestamps_raw[0] if torch.is_tensor(timestamps_raw) else torch.as_tensor(timestamps_raw[0])

    with autocast_context(args, device):
        outputs = vggt_model.get_aggregator_token_outputs(images)
        image_tokens_list = outputs["image_tokens_list"]
        patch_start_idx = int(outputs["patch_start_idx"])
        del outputs

        z_generated = sf.denormalize(z_generated_raw_n.float())
        decoded_patch_tokens = vggt_model.scene_tokenizer.decode(z_generated, patch_grid=args.patch_grid)
        del z_generated
        decoded_full_tokens = reattach_special_tokens(
            image_tokens_list,
            TOKENIZER_LEVELS,
            patch_start_idx,
            decoded_patch_tokens,
        )
        del decoded_patch_tokens
        generated_image_tokens = replace_selected_levels(
            image_tokens_list,
            TOKENIZER_LEVELS,
            decoded_full_tokens,
        )
        del decoded_full_tokens, image_tokens_list

    with autocast_context(args, device):
        gen_agg, gen_dino = split_image_tokens_for_heads(generated_image_tokens)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            raw_gs_map, raw_gs_conf = vggt_model.gs_head(generated_image_tokens, images, patch_start_idx)
            generated_pose_enc = vggt_model.camera_head(gen_agg)[-1]
            generated_depth, _ = vggt_model.depth_head(gen_agg, images, patch_start_idx)
            generated_dynamic_conf, _ = vggt_model.instance_head(gen_dino, images, patch_start_idx)
            generated_semantic_logits, _ = vggt_model.semantic_head(gen_dino, images, patch_start_idx)
            generated_sky_mask = _semantic_logits_to_sky_mask(generated_semantic_logits)

    del generated_image_tokens, gen_agg, gen_dino

    result = {
        "generated_pred_sky_mask": _sky_mask_image_grid(generated_sky_mask, frames),
        "generated_raw_3dgs_rgb": _render_gs_map_rgb(
            vggt_model,
            images,
            masks,
            timestamps,
            generated_pose_enc,
            generated_depth,
            raw_gs_map,
            raw_gs_conf,
            generated_dynamic_conf,
            device,
            frames,
            background_mode="sky",
            use_sky_mask=True,
        ),
    }
    del (
        generated_pose_enc,
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        generated_semantic_logits,
        generated_sky_mask,
    )
    _cuda_empty_cache_if_available()
    return result


@torch.no_grad()
def save_validation_images(
    bundle,
    scene_flow: nn.Module,
    log_dir: Path,
    step: int,
    args,
    device: torch.device,
    *,
    visualization_batch: dict[str, torch.Tensor] | None = None,
    vggt_model: nn.Module | None = None,
) -> dict[str, Path]:
    out_dir = log_dir / "validation" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean.float())
    frames = min(int(args.val_log_images), int(z_clean_n.shape[1]))
    paths: dict[str, Path] = {}

    base_images = {
        "target_latent_pca": _latent_pca_grid(z_clean_n, bundle.patch_grid, frames),
        "M_preserve": _mask_grid(bundle.M_preserve, bundle.patch_grid, frames),
        "M_source": _mask_grid(bundle.M_source, bundle.patch_grid, frames),
        "M_dest": _mask_grid(bundle.M_dest, bundle.patch_grid, frames),
    }
    for name, image in base_images.items():
        path = out_dir / f"{name}.jpg"
        save_image_grid(image, path, nrow=frames)
        paths[name] = path

    render_rgb = (
        vggt_model is not None
        and visualization_batch is not None
        and not bool(getattr(args, "no_val_render_rgb", False))
    )
    for scale_idx, scale in enumerate(_validation_scales(args)):
        z_generated_raw = cfg_sample_edit_latents(scene_flow, bundle, args, step, device, scale)
        z_generated_preserve_blend = bundle.M_preserve * z_clean_n + (1.0 - bundle.M_preserve) * z_generated_raw
        suffix = f"__cfg{scale:g}"
        images = {
            f"generated_raw_latent_pca{suffix}": _latent_pca_grid(z_generated_raw, bundle.patch_grid, frames),
            f"generated_preserve_blend_latent_pca{suffix}": _latent_pca_grid(
                z_generated_preserve_blend,
                bundle.patch_grid,
                frames,
            ),
            f"abs_error_raw{suffix}": _normalized_mask_grid(
                (z_generated_raw - z_clean_n).abs().mean(dim=-1, keepdim=True),
                bundle.patch_grid,
                frames,
            ),
            f"abs_error_preserve_blend{suffix}": _normalized_mask_grid(
                (z_generated_preserve_blend - z_clean_n).abs().mean(dim=-1, keepdim=True),
                bundle.patch_grid,
                frames,
            ),
        }
        for name, image in images.items():
            path = out_dir / f"{name}.jpg"
            save_image_grid(image, path, nrow=frames)
            paths[name] = path
        if render_rgb:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rgb_images = (
                    render_validation_rgb_gt_sky(
                        visualization_batch,
                        vggt_model,
                        scene_flow,
                        z_generated_raw,
                        args,
                        device,
                    )
                    if scale_idx == 0
                    else render_validation_generated_rgb_gt_sky(
                        visualization_batch,
                        vggt_model,
                        scene_flow,
                        z_generated_raw,
                        args,
                        device,
                    )
                )
                _save_rgb_validation_images(
                    rgb_images,
                    out_dir,
                    paths,
                    frames,
                    suffix,
                    only_generated=scale_idx != 0,
                )
            except Exception as exc:
                print(
                    f"[validation {step:06d}] warning: failed to render 3DGS RGB "
                    f"for cfg={scale:g}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                render_rgb = False
    return paths


@torch.no_grad()
def run_validation(
    loader: DataLoader,
    assembler: FlowFeatureAssembler,
    scene_flow: nn.Module,
    scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    args,
    step: int,
    log_dir: Path,
    wandb_run,
    ema: EMAModel | None = None,
    vggt_model: nn.Module | None = None,
) -> dict[str, float]:
    was_training = scene_flow.training
    scene_flow.eval()
    assembler.eval()
    use_val_ema = ema is not None and not args.no_val_ema
    ema_params = list(unwrap_ddp(scene_flow).parameters()) if use_val_ema else None
    if use_val_ema:
        ema.store(ema_params)
        ema.copy_to(ema_params)

    sums: dict[str, float] = {}
    count = 0
    first_item: dict[str, Any] | None = None
    iterator = loader
    if is_main_process() and not args.no_tqdm:
        iterator = tqdm(loader, total=args.val_batches, desc=f"val {step:06d}", dynamic_ncols=True, leave=False)

    for item in iterator:
        if count >= args.val_batches:
            break
        if first_item is None and is_main_process():
            first_item = _first_item(item)
        with autocast_context(args, device):
            loss, logs = train_step(item, assembler, scene_flow, scheduler, device, args)
        logs = dict(logs)
        logs["loss"] = float(loss.detach().item())
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["batches"] = float(count)

    if is_main_process():
        image_paths: dict[str, Path] = {}
        if first_item is not None and args.val_log_images > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            visualization_batch = None
            if vggt_model is not None and not bool(getattr(args, "no_val_render_rgb", False)):
                try:
                    visualization_batch = load_validation_visualization_batch(first_item, loader.dataset)
                except Exception as exc:
                    print(
                        f"[validation {step:06d}] warning: failed to load RGB render inputs: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            first_bundle = build_flow_bundle(first_item, assembler, device)
            image_paths = save_validation_images(
                first_bundle,
                scene_flow,
                log_dir,
                step,
                args,
                device,
                visualization_batch=visualization_batch,
                vggt_model=vggt_model,
            )
        metrics_text = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"[validation {step:06d}] {metrics_text}", flush=True)
        log_wandb(wandb_run, metrics, step, "validation")
        if wandb_run is not None and image_paths:
            import wandb

            wandb_run.log(
                {f"validation/{name}": wandb.Image(str(path)) for name, path in image_paths.items()},
                step=step,
            )

    if use_val_ema:
        ema.restore(ema_params)
    if was_training:
        scene_flow.train()
        assembler.scaffold_packer.train()
    if is_distributed():
        dist.barrier()
    return metrics


def load_resume_checkpoint(
    scene_flow: nn.Module,
    assembler: FlowFeatureAssembler,
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
    if "scaffold_packer" in payload:
        unwrap_ddp(assembler.scaffold_packer).load_state_dict(payload["scaffold_packer"], strict=True)
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


# ---------------------------------------------------------------------- #
# Main loop                                                               #
# ---------------------------------------------------------------------- #
def main() -> None:
    args = build_argparser().parse_args()
    device, local_rank, world_size = setup_distributed(args)
    if int(args.num_workers) > 0:
        torch.multiprocessing.set_sharing_strategy(str(args.mp_sharing_strategy))
    seed_everything(args.seed + get_rank())

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest_path is None and args.cache_root is None:
        raise ValueError("Provide either --cache_root or --manifest_path.")

    mode_filter = (
        [m.strip() for m in args.mode_filter.split(",") if m.strip()]
        if args.mode_filter else None
    )
    train_ds = WaymoFlowCacheDataset(
        cache_root=args.cache_root,
        manifest_path=args.manifest_path,
        mode_filter=mode_filter,
        split=args.split,
        min_frames=args.sequence_length,
        max_frames=args.sequence_length,
        seed=args.seed,
        mmap_plain_cache=not bool(args.no_mmap_plain_cache),
        asset_lut_level_indices=None if bool(args.full_asset_lut_cache) else (-1,),
    )
    val_ds = split_train_val_entries(
        train_ds,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if is_main_process():
        print(
            f"[data] train_entries={len(train_ds.entries)} "
            f"val_entries={0 if val_ds is None else len(val_ds.entries)} "
            f"val_fraction={float(args.val_fraction):.3f} "
            f"asset_lut_levels={'all' if bool(args.full_asset_lut_cache) else 'last'}",
            flush=True,
        )
    val_loader = None
    if val_ds is not None and args.val_every > 0 and args.val_batches > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: batch,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            **dataloader_runtime_kwargs(args),
        )
    patch_grid = _infer_cache_patch_grid(train_ds)
    h_splat = patch_grid[0] * 4
    w_splat = patch_grid[1] * 4
    args.patch_grid = list(patch_grid)
    args.H_splat = int(h_splat)
    args.W_splat = int(w_splat)
    if is_main_process():
        (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(
            f"[train] cache patch_grid={patch_grid}, H_splat={h_splat}, W_splat={w_splat}",
            flush=True,
        )
    wandb_run = init_wandb(args, log_dir)

    render_vggt = None
    enable_val_rgb_render = (
        not bool(args.no_val_render_rgb)
        and val_loader is not None
        and int(args.val_log_images) > 0
    )
    if is_main_process() and enable_val_rgb_render:
        render_vggt = load_dggt_aggregator_and_tokenizer(
            args.ckpt_path,
            args.tokenizer_ckpt_path,
            device,
        )
        render_vggt.scene_tokenizer.float()
        if is_main_process():
            print("[validation] 3DGS RGB rendering enabled on rank 0.", flush=True)

    tokenizer = (
        render_vggt.scene_tokenizer
        if render_vggt is not None
        else _load_tokenizer(args.tokenizer_ckpt_path or args.ckpt_path, device)
    )
    freeze_module(tokenizer)  # T1: encoder frozen; decoder layer_heads/local_refine can be unfrozen later.

    # Assembler: scaffold_packer + feature_splatter + soft_mask + noise_scheduler trainable.
    assembler = FlowFeatureAssembler(
        scene_tokenizer=tokenizer,
        patch_grid=patch_grid,
        H_splat=h_splat,
        W_splat=w_splat,
        scaffold_out_dim=int(args.latent_dim),
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    # Freeze inner editor / soft_mask (no params), scaffold packer trainable.
    freeze_module(assembler.editor)
    freeze_module(assembler.soft_mask)  # no params but safe.
    freeze_module(assembler.feature_splatter)

    sf_in_channels = 3 * int(args.latent_dim) + 3
    scene_flow = WanSceneFlow.from_scene_config(
        bring_up=False,
        patch_grid=patch_grid,
        in_channels=sf_in_channels,
        out_channels=int(args.latent_dim),
    ).to(device)
    scene_flow.enable_gradient_checkpointing()
    load_into_buffers(scene_flow, args.feature_stats_path, token_dim=int(args.latent_dim))
    warm_start_info = load_scene_flow_warm_start(
        scene_flow,
        args.scene_flow_pretrain_path,
        use_ema=bool(args.scene_flow_pretrain_ema),
    )
    if is_main_process() and warm_start_info is not None:
        print(f"[warm-start] {warm_start_info}", flush=True)

    ema = EMAModel(scene_flow.parameters(), decay=args.ema_decay)
    ema.to(device)

    if world_size > 1:
        scene_flow = DistributedDataParallel(
            scene_flow,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
        assembler.scaffold_packer = DistributedDataParallel(
            assembler.scaffold_packer,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )

    scene_decay, scene_no_decay = split_param_groups(unwrap_ddp(scene_flow))
    scaffold_decay, scaffold_no_decay = split_param_groups(unwrap_ddp(assembler.scaffold_packer))
    params = list(scene_flow.parameters()) + list(assembler.scaffold_packer.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": scene_decay + scaffold_decay, "weight_decay": args.weight_decay},
            {"params": scene_no_decay + scaffold_no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    lr_scheduler = build_cosine_warmup(optimizer, args.warmup_steps, args.max_steps)
    flow_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.shift)
    global_step = load_resume_checkpoint(
        scene_flow,
        assembler,
        ema,
        optimizer,
        lr_scheduler,
        args.resume_path,
        device,
    )

    sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        drop_last=True,
        **dataloader_runtime_kwargs(args),
    )

    accum_count = 0
    scene_flow.train()
    assembler.scaffold_packer.train()
    optimizer.zero_grad(set_to_none=True)
    wandb_sums: dict[str, float] = {}
    wandb_count = 0
    accum_data_wait_s = 0.0
    accum_train_wall_s = 0.0
    progress = None
    if is_main_process() and not args.no_tqdm:
        progress = tqdm(total=args.max_steps, initial=global_step, desc="train", dynamic_ncols=True)
    try:
        while global_step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(global_step)
            data_wait_t0 = time.perf_counter()
            for item in loader:
                data_wait_s = time.perf_counter() - data_wait_t0
                if global_step >= args.max_steps:
                    break
                micro_t0 = time.perf_counter()
                sync_grad = (accum_count + 1) % max(1, args.grad_accum_steps) == 0
                with ExitStack() as stack:
                    if isinstance(scene_flow, DistributedDataParallel) and not sync_grad:
                        stack.enter_context(scene_flow.no_sync())
                    if isinstance(assembler.scaffold_packer, DistributedDataParallel) and not sync_grad:
                        stack.enter_context(assembler.scaffold_packer.no_sync())
                    with autocast_context(args, device):
                        loss, metrics = train_step(item, assembler, scene_flow, flow_scheduler, device, args)
                        loss = loss / max(1, args.grad_accum_steps)
                    loss.backward()
                micro_wall_s = time.perf_counter() - micro_t0
                accum_data_wait_s += float(data_wait_s)
                accum_train_wall_s += float(micro_wall_s)
                accum_count += 1
                data_wait_t0 = time.perf_counter()
                if not sync_grad:
                    continue

                optim_t0 = time.perf_counter()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.step(unwrap_ddp(scene_flow).parameters())
                optim_s = time.perf_counter() - optim_t0
                data_wait_step_s = accum_data_wait_s
                train_wall_step_s = accum_train_wall_s
                step_wall_s = data_wait_step_s + train_wall_step_s + float(optim_s)
                accum_count = 0
                accum_data_wait_s = 0.0
                accum_train_wall_s = 0.0
                global_step += 1

                if is_main_process():
                    lr_now = float(optimizer.param_groups[0]["lr"])
                    train_metrics = dict(metrics)
                    train_metrics["lr"] = lr_now
                    train_metrics["data_wait_s"] = float(data_wait_step_s)
                    train_metrics["train_wall_s"] = float(train_wall_step_s)
                    train_metrics["optim_s"] = float(optim_s)
                    train_metrics["step_wall_s"] = float(step_wall_s)
                    train_metrics["data_wait_frac"] = (
                        float(data_wait_step_s / step_wall_s) if step_wall_s > 0.0 else 0.0
                    )
                    micro_bs = float(metrics.get("micro_batch_size", args.batch_size))
                    train_metrics["items_per_s_per_rank"] = (
                        micro_bs * max(1, int(args.grad_accum_steps)) / step_wall_s
                        if step_wall_s > 0.0
                        else 0.0
                    )
                    if progress is not None:
                        postfix = {"lr": f"{lr_now:.2e}"}
                        for key, value in train_metrics.items():
                            postfix[key] = f"{float(value):.4f}"
                        progress.set_postfix(postfix, refresh=False)
                    elif global_step % max(1, int(args.log_every)) == 0:
                        metrics_str = " | ".join(f"{key}={value:.4f}" for key, value in train_metrics.items())
                        print(f"[step {global_step:06d}] lr={lr_now:.2e} | {metrics_str}", flush=True)
                    for key, value in train_metrics.items():
                        wandb_sums[key] = wandb_sums.get(key, 0.0) + float(value)
                    wandb_count += 1
                    if wandb_run is not None and wandb_count >= max(1, int(args.wandb_log_every)):
                        averaged = {key: value / wandb_count for key, value in wandb_sums.items()}
                        log_wandb(wandb_run, averaged, global_step, "train")
                        wandb_sums = {}
                        wandb_count = 0

                if is_main_process() and args.vis_every > 0 and (global_step % args.vis_every == 0):
                    _dump_vis(_first_item(item), assembler, log_dir, global_step, device, args)

                if (
                    val_loader is not None
                    and args.val_every > 0
                    and args.val_batches > 0
                    and global_step % args.val_every == 0
                ):
                    run_validation(
                        val_loader,
                        assembler,
                        scene_flow,
                        flow_scheduler,
                        device,
                        args,
                        global_step,
                        log_dir,
                        wandb_run,
                        ema,
                        render_vggt,
                    )

                if global_step > 0 and global_step % args.save_every == 0:
                    if is_distributed():
                        dist.barrier()
                    if is_main_process():
                        _save_checkpoint(scene_flow, assembler, ema, optimizer, lr_scheduler, global_step, log_dir, args)
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
        _save_checkpoint(scene_flow, assembler, ema, optimizer, lr_scheduler, global_step, log_dir, args)
        if wandb_run is not None:
            wandb_run.finish()
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def _dump_vis(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    log_dir: Path,
    step: int,
    device: torch.device,
    args,
) -> None:
    from dggt.utils.flow_viz import dump_flow_features

    if item.get("flow_inputs_cached") is not None:
        if is_main_process():
            print(
                f"[vis] skipping full flow feature dump for fast cache item at step={step}; "
                "fast items do not load raw heads or asset Gaussians.",
                flush=True,
            )
        return

    vis_dir = log_dir / "vis" / f"step_{step:06d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["sample"].items()}
        predictions = _move_predictions(item["predictions"], device)
        apr = _move_asset_pass(item["asset_pass_result"], device)
        _validate_item_patch_grid(apr, assembler, item.get("cache_path"))
        cams = {k: v.to(device) for k, v in item["cameras_dggt"].items()}
        mode_kind = str(item.get("mode_kind", sample.get("mode_kind", "mode_a")))
        mode_b_payload = item.get("mode_b")
        if mode_b_payload is not None:
            mode_b_payload = _move_mode_b(mode_b_payload, device)
        phase1_localized_lite, splatted_tok_low_cached = _move_v6_fast_path_inputs(
            item, mode_kind, device
        )
        bundle = assembler(
            sample=sample,
            predictions=predictions,
            asset_pass_result=apr,
            cameras_dggt=cams,
            object_slots_spec="all",
            device=device,
            mode_kind=mode_kind,
            mode_b=mode_b_payload,
            phase1_localized_lite=phase1_localized_lite,
            splatted_tok_low_cached=splatted_tok_low_cached,
        )
    dump_flow_features(bundle, vis_dir, save_splat_pca=False)


def _save_checkpoint(
    scene_flow: nn.Module,
    assembler: FlowFeatureAssembler,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    step: int,
    log_dir: Path,
    args,
) -> None:
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sf = unwrap_ddp(scene_flow)
    scene_flow_state = sf.state_dict()
    ema_scene_flow_state = materialize_ema_state_dict(scene_flow, ema)
    state = {
        "step": int(step),
        "scene_flow": scene_flow_state,
        "ema_scene_flow": ema.state_dict(),
        "ema_scene_flow_state_dict": ema_scene_flow_state,
        "scaffold_packer": unwrap_ddp(assembler.scaffold_packer).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(state, ckpt_dir / f"flow_step{step:06d}.pt")
    torch.save({"scene_flow": scene_flow_state}, ckpt_dir / f"flow_step{step:06d}_weights_only.pt")
    torch.save(
        {
            "scene_flow": ema_scene_flow_state,
            "step": int(step),
            "is_ema_weights": True,
        },
        ckpt_dir / f"flow_step{step:06d}_ema_weights_only.pt",
    )


if __name__ == "__main__":
    main()
