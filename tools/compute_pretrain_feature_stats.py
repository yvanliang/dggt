"""Compute tokenizer-latent and DGGT CameraHead v3 stats in one frozen pass."""
from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset import WaymoOpenDataset
from dggt.models.vggt import VGGT
from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_STATS_STD_FLOOR,
    CAMERA_STATS_VERSION,
    CAMERA_TARGET_SOURCE,
    CAMERA_TARGET_SPACE,
    camera_state_from_dggt_pose_enc,
)
from dggt.utils.feature_stats import (
    checkpoint_sha256,
    load_feature_stats,
    save_feature_stats,
)
from dggt.utils.tokens import select_patch_pyramid


DEFAULT_LEVELS = (4, 11, 17, 23)
DEFAULT_PATCH_GRID = (37, 37)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def autocast_context(args: argparse.Namespace, device: torch.device):
    if args.precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def _clean_state_dict(state: dict[str, Any], prefix: str | None = None) -> dict[str, Any]:
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if prefix is not None:
            if not key.startswith(prefix):
                continue
            key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _format_key_examples(keys: list[str], limit: int = 5) -> str:
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
    payload = torch.load(dggt_ckpt_path, map_location="cpu")
    state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(_clean_state_dict(state), strict=False)
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
            f"missing_examples={_format_key_examples(real_missing)} "
            f"unexpected_examples={_format_key_examples(unexpected)}",
            flush=True,
        )

    if tokenizer_ckpt_path:
        tok_payload = torch.load(tokenizer_ckpt_path, map_location="cpu")
        if isinstance(tok_payload, dict) and "scene_tokenizer" in tok_payload:
            tok_state = tok_payload["scene_tokenizer"]
        elif isinstance(tok_payload, dict) and "state_dict" in tok_payload:
            tok_state = tok_payload["state_dict"]
        else:
            tok_state = tok_payload
        if not isinstance(tok_state, dict):
            raise TypeError(f"Tokenizer checkpoint must contain a state dict, got {type(tok_state).__name__}")
        if any(key.startswith("scene_tokenizer.") for key in tok_state):
            tok_state = _clean_state_dict(tok_state, prefix="scene_tokenizer.")
        else:
            tok_state = _clean_state_dict(tok_state)
        missing, unexpected = model.scene_tokenizer.load_state_dict(tok_state, strict=False)
        print(f"[ckpt:tokenizer] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing or unexpected:
            print(
                "[ckpt:tokenizer] "
                f"missing_examples={_format_key_examples(missing)} "
                f"unexpected_examples={_format_key_examples(unexpected)}",
                flush=True,
            )
            raise RuntimeError("Tokenizer checkpoint did not match VGGT.scene_tokenizer.")

    model.eval()
    freeze_module(model.aggregator)
    freeze_module(model.camera_head)
    freeze_module(model.scene_tokenizer)
    return model


def parse_scene_names(args: argparse.Namespace) -> list[str] | None:
    if args.scene_list:
        names = [line.strip() for line in Path(args.scene_list).read_text().splitlines() if line.strip()]
        missing = [name for name in names if not (Path(args.image_dir) / name / "images").is_dir()]
        if missing:
            raise RuntimeError(f"Selected scenes are missing image folders: {missing[:12]}")
        return names
    if args.scene_names:
        names = [part.strip() for part in args.scene_names.split(",") if part.strip()]
        missing = [name for name in names if not (Path(args.image_dir) / name / "images").is_dir()]
        if missing:
            raise RuntimeError(f"Selected scenes are missing image folders: {missing[:12]}")
        return names
    if args.scene_start is not None and args.scene_end is not None:
        names = [
            f"{idx:03d}"
            for idx in range(int(args.scene_start), int(args.scene_end))
            if (Path(args.image_dir) / f"{idx:03d}" / "images").is_dir()
        ]
        if not names:
            raise RuntimeError(
                f"No scene image folders found in {args.image_dir} for "
                f"[{args.scene_start}, {args.scene_end})."
            )
        return names
    return None


def validate_dynamic_masks(dataset: WaymoOpenDataset, *, max_missing_report: int = 12) -> None:
    if len(dataset) == 0:
        raise RuntimeError("WaymoOpenDataset is empty.")
    if len(dataset.image_paths) != len(dataset.scenes):
        raise RuntimeError(
            "WaymoOpenDataset did not find image folders for every selected scene: "
            f"image_paths={len(dataset.image_paths)} scenes={len(dataset.scenes)}. "
            "Check --image_dir/--scene_* selection."
        )
    if len(dataset.dynamic_mask_path) != len(dataset.scenes):
        raise RuntimeError(
            "WaymoOpenDataset dynamic_mask_path coverage does not match selected scenes: "
            f"dynamic_mask_path={len(dataset.dynamic_mask_path)} scenes={len(dataset.scenes)}."
        )
    missing = []
    empty = 0
    for idx, paths in enumerate(dataset.dynamic_mask_path):
        if isinstance(paths, list) and len(paths) > 0:
            continue
        empty += 1
        scene = dataset.scenes[idx] if idx < len(dataset.scenes) else f"index={idx}"
        missing.append(scene)
    if empty:
        shown = ", ".join(missing[:max_missing_report])
        suffix = "" if empty <= max_missing_report else f", ... (+{empty - max_missing_report})"
        raise RuntimeError(
            "Pretraining feature stats use the same raw Waymo pipeline and require "
            f"fine_dynamic_masks/all coverage. Missing dynamic_mask_path for {empty}/"
            f"{len(dataset.dynamic_mask_path)} scenes: {shown}{suffix}"
        )


def infer_patch_grid(num_patches: int) -> tuple[int, int]:
    root = int(round(num_patches ** 0.5))
    if root * root == int(num_patches):
        return root, root
    best_h, best_w, best_gap = 1, int(num_patches), int(num_patches)
    for h in range(1, int(num_patches**0.5) + 1):
        if num_patches % h == 0:
            w = num_patches // h
            gap = abs(w - h)
            if gap < best_gap:
                best_h, best_w, best_gap = h, w, gap
    return best_h, best_w


@torch.no_grad()
def compute_dggt_stats_single_pass(
    model: VGGT,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    compute_latents: bool,
    dggt_checkpoint_sha256: str,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    latent_total = 0
    latent_sum = torch.zeros(int(args.latent_dim), dtype=torch.float64)
    latent_sum2 = torch.zeros_like(latent_sum)
    camera_sums = {role: torch.zeros(CAMERA_GENERATION_DIM, dtype=torch.float64) for role in ("anchor", "delta")}
    camera_sums2 = {role: value.clone() for role, value in camera_sums.items()}
    camera_counts = {"anchor": 0, "delta": 0}
    yielded = 0
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if args.require_dynamic_mask and ("dynamic_mask" not in batch or batch["dynamic_mask"] is None):
            raise RuntimeError(
                "Batch is missing dynamic_mask. Check fine_dynamic_masks/all coverage for selected scenes."
            )
        images = batch["images"].to(device, non_blocking=True)
        with autocast_context(args, device):
            outputs = model.get_aggregator_token_outputs(images)
            pose_enc_dggt = model.camera_head(outputs["aggregated_tokens_list"])[-1]
            camera_state_dggt, anchor_mask = camera_state_from_dggt_pose_enc(pose_enc_dggt)
            flat_camera = camera_state_dggt.detach().cpu().double().reshape(-1, CAMERA_GENERATION_DIM)
            flat_anchor = anchor_mask.detach().cpu().bool().reshape(-1)
            for role, select in (("anchor", flat_anchor), ("delta", ~flat_anchor)):
                values = flat_camera[select]
                camera_counts[role] += int(values.shape[0])
                if values.numel():
                    camera_sums[role] += values.sum(0)
                    camera_sums2[role] += values.square().sum(0)
            if not compute_latents:
                z = None
            else:
                image_tokens_all = outputs["image_tokens_list"]
                patch_start_idx = int(outputs["patch_start_idx"])
                image_patch = select_patch_pyramid(image_tokens_all, DEFAULT_LEVELS, patch_start_idx)
                patch_count = int(image_patch[-1].shape[-2])
                patch_grid = (
                    DEFAULT_PATCH_GRID
                    if patch_count == DEFAULT_PATCH_GRID[0] * DEFAULT_PATCH_GRID[1]
                    else infer_patch_grid(patch_count)
                )
                z = model.scene_tokenizer.encode(image_patch, patch_grid=patch_grid)
        if z is not None:
            flat = z.detach().cpu().double().reshape(-1, int(args.latent_dim))
            latent_total += int(flat.shape[0])
            latent_sum += flat.sum(0)
            latent_sum2 += flat.square().sum(0)
        yielded += 1
        if yielded % max(1, args.log_every) == 0:
            print(f"[stats] processed_batches={yielded}", flush=True)
    if camera_counts["anchor"] == 0 or camera_counts["delta"] == 0:
        raise ValueError("DGGT camera stats require at least one anchor and one delta token")
    camera_stats: dict[str, Any] = {
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_stats_version": CAMERA_STATS_VERSION,
        "camera_target_space": CAMERA_TARGET_SPACE,
        "camera_target_source": CAMERA_TARGET_SOURCE,
        "dggt_checkpoint_sha256": str(dggt_checkpoint_sha256),
        "camera_dim": CAMERA_GENERATION_DIM,
    }
    for role in ("anchor", "delta"):
        count = camera_counts[role]
        mean = camera_sums[role] / float(count)
        variance = (camera_sums2[role] / float(count) - mean.square()).clamp_min(CAMERA_STATS_STD_FLOOR**2)
        camera_stats[f"camera_{role}_mean"] = mean.float()
        camera_stats[f"camera_{role}_std"] = variance.sqrt().float().clamp_min(CAMERA_STATS_STD_FLOOR)
        camera_stats[f"camera_{role}_count"] = torch.tensor(count, dtype=torch.long)
    if not compute_latents:
        return None, camera_stats
    if latent_total == 0:
        raise ValueError("Cannot compute latent stats from an empty loader")
    mean = latent_sum / float(latent_total)
    variance = (latent_sum2 / float(latent_total) - mean.square()).clamp_min(1.0e-6)
    return {
        "mu_z": mean.float(),
        "sigma_z": variance.sqrt().float(),
        "count": torch.tensor(latent_total, dtype=torch.long),
    }, camera_stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute SceneFlow latent plus camera anchor/delta statistics.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--dggt_ckpt_path", type=str, default=None)
    parser.add_argument("--tokenizer_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--latent_stats_path",
        type=str,
        default=None,
        help=(
            "Reuse validated mu_z/sigma_z while still running frozen DGGT CameraHead for v3 camera stats."
        ),
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--scene_list", type=str, default=None)
    parser.add_argument("--scene_names", type=str, default=None)
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=600)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="bf16", choices=("fp32", "bf16"))
    parser.add_argument("--require_dynamic_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=1024,
        help=(
            "Tokenizer latent channel count. Must match the tokenizer ckpt's "
            "actual output dim. SceneFlow's --latent_dim will read these stats "
            "(mu_z/sigma_z of this size) into its normalize buffers."
        ),
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.dggt_ckpt_path:
        raise ValueError("--dggt_ckpt_path is always required: camera stats come from frozen DGGT CameraHead")
    dggt_sha256 = checkpoint_sha256(args.dggt_ckpt_path)
    model = load_dggt_aggregator_and_tokenizer(args.dggt_ckpt_path, args.tokenizer_ckpt_path, device)
    dataset = WaymoOpenDataset(
        image_dir=args.image_dir,
        scene_names=parse_scene_names(args),
        sequence_length=args.sequence_length,
        mode=1,
        views=1,
    )
    if args.require_dynamic_mask:
        validate_dynamic_masks(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(
        f"[stats] scenes={len(dataset)} batch_size={args.batch_size} seq={args.sequence_length} "
        f"max_batches={args.max_batches}",
        flush=True,
    )

    computed_latent_stats, camera_stats = compute_dggt_stats_single_pass(
        model,
        loader,
        args,
        device,
        compute_latents=args.latent_stats_path is None,
        dggt_checkpoint_sha256=dggt_sha256,
    )
    if args.latent_stats_path is not None:
        mu_z, sigma_z = load_feature_stats(args.latent_stats_path, token_dim=int(args.latent_dim))
        latent_payload = torch.load(args.latent_stats_path, map_location="cpu", weights_only=False)
        stats = {
            "mu_z": mu_z,
            "sigma_z": sigma_z,
            "count": torch.as_tensor(latent_payload.get("count", 0), dtype=torch.long),
        }
        print(f"[stats] reused latent stats from {args.latent_stats_path}", flush=True)
    else:
        assert computed_latent_stats is not None
        stats = computed_latent_stats
    stats.update(camera_stats)
    stats["source"] = {
        "image_dir": args.image_dir,
        "scene_names": parse_scene_names(args),
        "sequence_length": int(args.sequence_length),
        "max_batches": args.max_batches,
        "levels": list(DEFAULT_LEVELS),
        "latent_stats_path": args.latent_stats_path,
    }
    save_feature_stats(stats, args.output_path)
    print(
        f"[stats] saved {args.output_path} count={int(stats['count'].item())} "
        f"mu_absmax={stats['mu_z'].abs().max().item():.6f} "
        f"sigma_mean={stats['sigma_z'].mean().item():.6f}",
        flush=True,
    )
    sidecar = Path(args.output_path).with_suffix(Path(args.output_path).suffix + ".json")
    sidecar.write_text(json.dumps({"args": vars(args), "count": int(stats["count"].item())}, indent=2))


if __name__ == "__main__":
    main()
