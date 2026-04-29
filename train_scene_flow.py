"""T1 SceneFlow training entry point (Phase 9 skeleton).

Reads the offline Phase-4.5 cache, drives `FlowFeatureAssembler` per step, and
computes a rectified-flow-style loss against a `SceneFlowMatching` module.

Since Phase 6 (`dggt/models/scene_flow.py`) is not yet implemented, this file
defines a `StubSceneFlow` that returns `zeros_like(z_init)` so the pipeline is
runnable end-to-end and loss-shape assertions hold. When the real module lands,
replace the import + instantiation under the marker `# SceneFlow instantiation`.

DDP scaffolding follows `train_tokenizer.py`. Visualization every `--vis_every`
steps dumps the same image set as `inference_scene_editor.py --dump_features`.
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
from dggt.models.flow_feature_assembler import FlowFeatureAssembler
from dggt.utils.flow_cache_io import load_flow_cache


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


def autocast_context(args, device: torch.device):
    if args.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _infer_cache_patch_grid(dataset: WaymoFlowCacheDataset) -> tuple[int, int]:
    if len(dataset.entries) == 0:
        raise RuntimeError("Cannot infer patch grid from an empty cache dataset.")
    entry = dataset.entries[0]
    cache_path = entry.get("cache_path")
    if cache_path is None:
        raise KeyError("Cache dataset entry is missing 'cache_path'.")
    payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
    patch_grid = payload.get("meta", {}).get("patch_grid")
    if patch_grid is None or len(patch_grid) != 2:
        raise KeyError(f"Cache payload {cache_path} is missing meta.patch_grid=(H,W).")
    out = (int(patch_grid[0]), int(patch_grid[1]))
    if out[0] <= 0 or out[1] <= 0:
        raise ValueError(f"Invalid cache patch_grid {out} in {cache_path}.")
    return out


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
# Stub SceneFlow                                                          #
# ---------------------------------------------------------------------- #
class StubSceneFlow(nn.Module):
    """Placeholder for the Phase 6 `SceneFlowMatching` module.

    Same API (`forward(z_t, t_tok, z_clean, scaffold_tok, M_preserve, M_source,
    M_dest, F_asset_tokens) -> v_pred`) so training code doesn't change when
    the real module lands. Returns `zeros_like(z_t)` (velocity=0) — loss is
    finite and grads flow through a single trainable projection so the
    optimiser has something to optimise.
    """

    def __init__(self, token_dim: int = 768) -> None:
        super().__init__()
        self.identity_proj = nn.Linear(token_dim, token_dim)
        nn.init.zeros_(self.identity_proj.weight)
        nn.init.zeros_(self.identity_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t_tok: torch.Tensor,
        z_clean: torch.Tensor,
        scaffold_tok: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
        F_asset_tokens: torch.Tensor,
    ) -> torch.Tensor:
        del t_tok, z_clean, scaffold_tok, M_preserve, M_source, M_dest, F_asset_tokens
        return self.identity_proj(z_t)


# ---------------------------------------------------------------------- #
# Losses                                                                  #
# ---------------------------------------------------------------------- #
def masked_mse(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Preserve-weighted MSE (Phase 9 L_flow stand-in).

    `weight` is a `[B, S, P, 1]` tensor; we downweight preserve patches so the
    flow learns where edits live.
    """
    diff = (v_pred - v_gt).pow(2).mean(dim=-1, keepdim=True)
    num = (diff * weight).sum()
    den = weight.sum().clamp_min(eps)
    return num / den


# ---------------------------------------------------------------------- #
# CLI                                                                     #
# ---------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T1 SceneFlow training (Phase 9).")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Base DGGT checkpoint for tokenizer.")
    parser.add_argument("--cache_root", type=str, default=None,
                        help="Offline feature cache root (Phase 4.5 output). Mutually exclusive with --manifest_path.")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Merged Mode A/B JSONL manifest from tools/build_flow_train_manifest.py.")
    parser.add_argument("--mode_filter", type=str, default=None,
                        help="When using --manifest_path, restrict to comma-sep modes (e.g. 'mode_a,mode_b').")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--val_split", type=str, default="validation")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dggt-flow")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-process batch; keep at 1 for now.")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--vis_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--lambda_flow", type=float, default=1.0)
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
    tokenizer = model.scene_tokenizer
    return tokenizer


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


# ---------------------------------------------------------------------- #
# Train step                                                              #
# ---------------------------------------------------------------------- #
def train_step(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    scene_flow: nn.Module,
    device: torch.device,
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["sample"].items()}
    predictions = _move_predictions(item["predictions"], device)
    asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
    _validate_item_patch_grid(asset_pass_result, assembler, item.get("cache_path"))
    cameras_dggt = {k: v.to(device) for k, v in item["cameras_dggt"].items()}
    mode_kind = str(item.get("mode_kind", sample.get("mode_kind", "mode_a")))
    mode_b_payload = item.get("mode_b")
    if mode_b_payload is not None:
        mode_b_payload = _move_mode_b(mode_b_payload, device)

    bundle = assembler(
        sample=sample,
        predictions=predictions,
        asset_pass_result=asset_pass_result,
        cameras_dggt=cameras_dggt,
        object_slots_spec="all",
        base_t=None,
        device=device,
        mode_kind=mode_kind,
        mode_b=mode_b_payload,
    )
    v_pred = scene_flow(
        bundle.z_init,
        bundle.t_tok,
        bundle.z_clean,
        bundle.scaffold_tok,
        bundle.M_preserve,
        bundle.M_source,
        bundle.M_dest,
        bundle.F_asset_tokens,
    )
    v_gt = bundle.z_clean - bundle.z_init
    edit_weight = 1.0 - bundle.M_preserve
    loss_flow = masked_mse(v_pred, v_gt, edit_weight)
    loss = args.lambda_flow * loss_flow
    metrics = {
        "loss": float(loss.detach().item()),
        "loss_flow": float(loss_flow.detach().item()),
        "edit_weight_mean": float(edit_weight.mean().item()),
        "num_objects": float(len(bundle.phase4_slots)),
    }
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
    )


# ---------------------------------------------------------------------- #
# Main loop                                                               #
# ---------------------------------------------------------------------- #
def main() -> None:
    args = build_argparser().parse_args()
    device, local_rank, world_size = setup_distributed(args)
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
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        seed=args.seed,
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

    tokenizer = _load_tokenizer(args.ckpt_path, device)
    freeze_module(tokenizer)  # T1: encoder frozen; decoder layer_heads/local_refine can be unfrozen later.

    # Assembler: scaffold_packer + feature_splatter + soft_mask + noise_scheduler trainable.
    assembler = FlowFeatureAssembler(
        scene_tokenizer=tokenizer,
        patch_grid=patch_grid,
        H_splat=h_splat,
        W_splat=w_splat,
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    # Freeze inner editor / soft_mask (no params), scaffold packer trainable.
    freeze_module(assembler.editor)
    freeze_module(assembler.soft_mask)  # no params but safe.
    freeze_module(assembler.feature_splatter)

    scene_flow = StubSceneFlow(token_dim=768).to(device)

    params = list(scene_flow.parameters()) + list(assembler.scaffold_packer.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],  # single sample per step for now
        pin_memory=device.type == "cuda",
    )

    if world_size > 1:
        scene_flow = DistributedDataParallel(
            scene_flow, device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    step = 0
    accum_count = 0
    scene_flow.train()
    assembler.scaffold_packer.train()
    while step < args.max_steps:
        for item in loader:
            if step >= args.max_steps:
                break
            with autocast_context(args, device):
                loss, metrics = train_step(item, assembler, scene_flow, device, args)
                loss = loss / max(1, args.grad_accum_steps)
            loss.backward()
            accum_count += 1
            if accum_count >= args.grad_accum_steps:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

            if is_main_process() and (step % args.log_every == 0):
                print(f"[step {step:06d}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)

            if is_main_process() and args.vis_every > 0 and (step % args.vis_every == 0) and step > 0:
                _dump_vis(item, assembler, log_dir, step, device, args)

            if is_main_process() and (step > 0) and (step % args.save_every == 0):
                _save_checkpoint(scene_flow, assembler, optimizer, step, log_dir, args)

            step += 1

    if is_main_process():
        _save_checkpoint(scene_flow, assembler, optimizer, step, log_dir, args)
    if is_distributed():
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
        bundle = assembler(
            sample=sample,
            predictions=predictions,
            asset_pass_result=apr,
            cameras_dggt=cams,
            object_slots_spec="all",
            device=device,
            mode_kind=mode_kind,
            mode_b=mode_b_payload,
        )
    dump_flow_features(bundle, vis_dir, save_splat_pca=False)


def _save_checkpoint(
    scene_flow: nn.Module,
    assembler: FlowFeatureAssembler,
    optimizer: torch.optim.Optimizer,
    step: int,
    log_dir: Path,
    args,
) -> None:
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": int(step),
        "scene_flow": (
            scene_flow.module if isinstance(scene_flow, DistributedDataParallel) else scene_flow
        ).state_dict(),
        "scaffold_packer": assembler.scaffold_packer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state, ckpt_dir / f"flow_step{step:06d}.pt")


if __name__ == "__main__":
    main()
