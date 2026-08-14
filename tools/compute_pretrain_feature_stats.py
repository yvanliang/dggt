"""Compute layout-v2 SceneFlow latent and scene-gauge statistics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from itertools import islice
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
from dggt.utils.feature_stats import (
    DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH,
    FEATURE_STATS_SCHEMA,
    FEATURE_STATS_SCHEMA_VERSION,
    compute_scene_gauge_stats,
    load_feature_stats,
    save_feature_stats,
    validate_production_stats_coverage,
)
from dggt.utils.scene_gauge import SCENE_GAUGE_DIM
from dggt.utils.sliding_window import OFFLINE_MAX_SINGLE_WINDOW
from dggt.utils.tokenizer_window import encode_tokenizer_windowed
from dggt.utils.tokens import batched_gather_frames, select_patch_pyramid


DEFAULT_LEVELS = (4, 11, 17, 23)
DEFAULT_PATCH_GRID = (25, 37)


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
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def load_reusable_latent_stats(
    path: str | Path,
    *,
    latent_dim: int,
    sequence_length: int,
    patch_grid: tuple[int, int],
    levels: tuple[int, ...] = DEFAULT_LEVELS,
) -> dict[str, torch.Tensor]:
    """Reuse complete latent moments after numerical and source-shape checks."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Reused latent stats at {path} must be a dict")
    validate_production_stats_coverage(payload)
    source = payload.get("source")
    expected_source = {
        "sequence_length": int(sequence_length),
        "dggt_context_length": 29,
        "patch_grid": list(map(int, patch_grid)),
        "levels": list(map(int, levels)),
    }
    if not isinstance(source, dict):
        raise ValueError("Reused latent stats are missing source metadata")
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(
                f"Reused latent stats source mismatch at {key}: "
                f"expected {expected!r}, got {source.get(key)!r}"
            )
    count = torch.as_tensor(payload.get("count", 0), dtype=torch.long)
    if count.numel() != 1 or int(count.item()) <= 0:
        raise ValueError("Reused latent stats require a positive scalar count")
    mean, std = load_feature_stats(path, token_dim=int(latent_dim))
    return {"mu_z": mean, "sigma_z": std, "count": count}


def _clean_state_dict(
    state: dict[str, Any], prefix: str | None = None
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for raw_key, value in state.items():
        key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
        if prefix is not None:
            if not key.startswith(prefix):
                continue
            key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _format_key_examples(keys: list[str], limit: int = 5) -> str:
    examples = ", ".join(keys[:limit])
    suffix = "" if len(keys) <= limit else ", ..."
    return f"[{examples}{suffix}]"


def load_dggt_aggregator_and_tokenizer(
    dggt_ckpt_path: str,
    tokenizer_ckpt_path: str,
    device: torch.device,
) -> VGGT:
    """Load only the frozen modules needed to reproduce tokenizer latents."""

    model = VGGT()
    payload = torch.load(dggt_ckpt_path, map_location="cpu", weights_only=False)
    state = (
        payload.get("state_dict", payload.get("model", payload))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(state, dict):
        raise TypeError("DGGT checkpoint must contain a state dict")
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
    missing_aggregator = [key for key in real_missing if key.startswith("aggregator.")]
    if missing_aggregator:
        raise RuntimeError(
            "DGGT checkpoint is missing aggregator weights: "
            f"{_format_key_examples(missing_aggregator)}"
        )

    tokenizer_payload = torch.load(
        tokenizer_ckpt_path, map_location="cpu", weights_only=False
    )
    if isinstance(tokenizer_payload, dict) and "scene_tokenizer" in tokenizer_payload:
        tokenizer_state = tokenizer_payload["scene_tokenizer"]
    elif isinstance(tokenizer_payload, dict) and "state_dict" in tokenizer_payload:
        tokenizer_state = tokenizer_payload["state_dict"]
    else:
        tokenizer_state = tokenizer_payload
    if not isinstance(tokenizer_state, dict):
        raise TypeError("Tokenizer checkpoint must contain a state dict")
    if any(key.startswith("scene_tokenizer.") for key in tokenizer_state):
        tokenizer_state = _clean_state_dict(
            tokenizer_state, prefix="scene_tokenizer."
        )
    else:
        tokenizer_state = _clean_state_dict(tokenizer_state)
    tokenizer_missing, tokenizer_unexpected = model.scene_tokenizer.load_state_dict(
        tokenizer_state, strict=False
    )
    if tokenizer_missing or tokenizer_unexpected:
        raise RuntimeError(
            "Tokenizer checkpoint did not match VGGT.scene_tokenizer: "
            f"missing={_format_key_examples(list(tokenizer_missing))}, "
            f"unexpected={_format_key_examples(list(tokenizer_unexpected))}"
        )

    for unused_name in (
        "camera_head",
        "point_head",
        "depth_head",
        "track_head",
        "gs_head",
        "instance_head",
        "semantic_head",
        "sky_model",
    ):
        setattr(model, unused_name, None)
    model.aggregator.to(device)
    model.scene_tokenizer.to(device)
    model.eval()
    freeze_module(model.aggregator)
    freeze_module(model.scene_tokenizer)
    return model


def parse_scene_names(args: argparse.Namespace) -> list[str] | None:
    if args.scene_list:
        names = [
            line.strip()
            for line in Path(args.scene_list).read_text().splitlines()
            if line.strip()
        ]
    elif args.scene_names:
        names = [part.strip() for part in args.scene_names.split(",") if part.strip()]
    elif args.scene_start is not None and args.scene_end is not None:
        names = [
            f"{index:03d}"
            for index in range(int(args.scene_start), int(args.scene_end))
            if (Path(args.image_dir) / f"{index:03d}" / "images").is_dir()
        ]
        if not names:
            raise RuntimeError(
                f"No scene image folders found in {args.image_dir} for "
                f"[{args.scene_start}, {args.scene_end})."
            )
    else:
        return None
    missing = [
        name for name in names if not (Path(args.image_dir) / name / "images").is_dir()
    ]
    if missing:
        raise RuntimeError(f"Selected scenes are missing image folders: {missing[:12]}")
    return names


def infer_patch_grid(num_patches: int) -> tuple[int, int]:
    root = int(round(num_patches**0.5))
    if root * root == int(num_patches):
        return root, root
    best_h, best_w, best_gap = 1, int(num_patches), int(num_patches)
    for height in range(1, int(num_patches**0.5) + 1):
        if num_patches % height == 0:
            width = num_patches // height
            gap = abs(width - height)
            if gap < best_gap:
                best_h, best_w, best_gap = height, width, gap
    return best_h, best_w


def _global_window_indices(
    batch: dict[str, Any], *, batch_size: int, sequence_length: int
) -> torch.Tensor:
    raw = batch.get("dggt_window_indices")
    if not torch.is_tensor(raw):
        raw = batch.get("frame_ids")
    if not torch.is_tensor(raw):
        raise RuntimeError(
            "Latent extraction requires dggt_window_indices or frame_ids [B,S]"
        )
    indices = torch.as_tensor(raw, dtype=torch.long)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
    expected = (int(batch_size), int(sequence_length))
    if tuple(indices.shape) != expected:
        raise ValueError(f"window indices shape {tuple(indices.shape)} != {expected}")
    return indices


@torch.no_grad()
def compute_dggt_stats_single_pass(
    model: VGGT | None,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    compute_latents: bool,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    """Stream the two representations consumed by layout-v2 SceneFlow."""

    if compute_latents and model is None:
        raise ValueError("A frozen DGGT/tokenizer model is required for latent stats")
    latent_total = 0
    latent_sum = torch.zeros(int(args.latent_dim), dtype=torch.float64)
    latent_sum2 = torch.zeros_like(latent_sum)
    gauge_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    processed_batches = 0
    batches = loader if args.max_batches is None else islice(loader, int(args.max_batches))

    for batch in batches:
        gauge_raw = batch.get("scene_gauge")
        gauge_valid_raw = batch.get("scene_gauge_valid")
        if not torch.is_tensor(gauge_raw) or not torch.is_tensor(gauge_valid_raw):
            raise RuntimeError(
                "Feature stats require scene_gauge and scene_gauge_valid from the offline table"
            )
        gauge = torch.as_tensor(gauge_raw).detach().cpu().float()
        gauge_valid = torch.as_tensor(gauge_valid_raw).detach().cpu().bool()
        if gauge.ndim == 1:
            gauge = gauge.unsqueeze(0)
        if gauge_valid.ndim == 1:
            gauge_valid = gauge_valid.unsqueeze(0)
        if gauge.shape != gauge_valid.shape or tuple(gauge.shape[1:]) != (
            SCENE_GAUGE_DIM,
        ):
            raise ValueError(
                "scene_gauge/valid must be [B,3], got "
                f"{tuple(gauge.shape)} and {tuple(gauge_valid.shape)}"
            )
        gauge_batches.append((gauge, gauge_valid))

        if compute_latents:
            context_raw = batch.get("dggt_context_images")
            if not torch.is_tensor(context_raw):
                raise RuntimeError(
                    "Latent stats require dggt_context_images from the complete 29-frame clip"
                )
            context_images = context_raw.to(device, non_blocking=True)
            if context_images.ndim != 5 or int(context_images.shape[1]) != 29:
                raise ValueError(
                    "dggt_context_images must be [B,29,3,H,W], got "
                    f"{tuple(context_images.shape)}"
                )
            batch_size = int(context_images.shape[0])
            window_indices = _global_window_indices(
                batch,
                batch_size=batch_size,
                sequence_length=int(args.sequence_length),
            ).to(device=device, non_blocking=True)
            assert model is not None
            with autocast_context(args, device):
                outputs = model.get_aggregator_token_outputs(context_images)
                image_tokens = [
                    batched_gather_frames(
                        tokens, window_indices, name=f"image_tokens_list[{level}]"
                    )
                    for level, tokens in enumerate(outputs["image_tokens_list"])
                ]
                image_patch = select_patch_pyramid(
                    image_tokens,
                    DEFAULT_LEVELS,
                    int(outputs["patch_start_idx"]),
                )
                patch_count = int(image_patch[-1].shape[-2])
                patch_grid = (
                    DEFAULT_PATCH_GRID
                    if patch_count == DEFAULT_PATCH_GRID[0] * DEFAULT_PATCH_GRID[1]
                    else infer_patch_grid(patch_count)
                )
                latent = encode_tokenizer_windowed(
                    model.scene_tokenizer,
                    image_patch,
                    patch_grid=patch_grid,
                    window_len=OFFLINE_MAX_SINGLE_WINDOW,
                )
            flat = latent.detach().cpu().double().reshape(-1, int(args.latent_dim))
            latent_total += int(flat.shape[0])
            latent_sum += flat.sum(0)
            latent_sum2 += flat.square().sum(0)

        processed_batches += 1
        if processed_batches % max(1, int(args.log_every)) == 0:
            print(f"[stats] processed_batches={processed_batches}", flush=True)

    representation_stats: dict[str, Any] = compute_scene_gauge_stats(gauge_batches)
    representation_stats["stats_processed_batches"] = int(processed_batches)
    if not compute_latents:
        return None, representation_stats
    if latent_total <= 0:
        raise ValueError("Cannot compute latent stats from an empty loader")
    mean = latent_sum / float(latent_total)
    variance = (latent_sum2 / float(latent_total) - mean.square()).clamp_min(1.0e-6)
    return (
        {
            "mu_z": mean.float(),
            "sigma_z": variance.sqrt().float(),
            "count": torch.tensor(latent_total, dtype=torch.long),
        },
        representation_stats,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute SceneFlow tokenizer-latent and 3D scene-gauge stats"
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--dggt_ckpt_path", type=str, default=None)
    parser.add_argument("--tokenizer_ckpt_path", type=str, default=None)
    parser.add_argument("--scene_gauge_path", type=str, required=True)
    parser.add_argument(
        "--latent_stats_path",
        type=str,
        default=None,
        help="Reuse complete mu_z/sigma_z while recomputing scene-gauge moments",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH),
    )
    parser.add_argument("--scene_list", type=str, default=None)
    parser.add_argument("--scene_names", type=str, default=None)
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=600)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="bf16", choices=("fp32", "bf16"))
    parser.add_argument("--latent_dim", type=int, default=1024)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    compute_latents = args.latent_stats_path is None
    if compute_latents and not args.dggt_ckpt_path:
        raise ValueError("--dggt_ckpt_path is required when computing latent stats")
    if compute_latents and not args.tokenizer_ckpt_path:
        raise ValueError("--tokenizer_ckpt_path is required when computing latent stats")
    model = (
        load_dggt_aggregator_and_tokenizer(
            args.dggt_ckpt_path,
            args.tokenizer_ckpt_path,
            device,
        )
        if compute_latents
        else None
    )
    scene_names = parse_scene_names(args)
    dataset = WaymoOpenDataset(
        image_dir=args.image_dir,
        scene_names=scene_names,
        sequence_length=int(args.sequence_length),
        mode=1,
        views=1,
        trunk_frames=29,
        trunk_major_samples=True,
        return_full_dggt_context=compute_latents,
        scene_gauge_path=args.scene_gauge_path,
        expected_scene_gauge_split=Path(args.image_dir).name,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(
        f"[stats] trunks={len(dataset)} batch_size={args.batch_size} "
        f"seq={args.sequence_length} max_batches={args.max_batches}",
        flush=True,
    )
    computed_latent, representation_stats = compute_dggt_stats_single_pass(
        model,
        loader,
        args,
        device,
        compute_latents=compute_latents,
    )
    patch_grid = tuple(map(int, dataset.pretrain_patch_grid))
    if args.latent_stats_path:
        stats = load_reusable_latent_stats(
            args.latent_stats_path,
            latent_dim=int(args.latent_dim),
            sequence_length=int(args.sequence_length),
            patch_grid=patch_grid,
        )
        print(f"[stats] reused latent stats from {args.latent_stats_path}", flush=True)
    else:
        assert computed_latent is not None
        stats = computed_latent
    stats.update(representation_stats)

    requested_keys = set(dataset.scene_gauge_requested_keys or ())
    required_keys = set(dataset.scene_gauge_required_keys or ())
    exact_gauge_scope = bool(requested_keys) and requested_keys == required_keys
    expected_batches = len(loader)
    processed_batches = int(stats.get("stats_processed_batches", -1))
    expected_latent_count = (
        int(len(dataset))
        * int(args.sequence_length)
        * int(patch_grid[0])
        * int(patch_grid[1])
    )
    actual_latent_count = int(torch.as_tensor(stats.get("count", -1)).item())
    exact_latent_count = actual_latent_count == expected_latent_count
    full_dataset_pass = (
        args.max_batches is None
        and processed_batches == expected_batches
        and exact_gauge_scope
        and exact_latent_count
    )
    stats.update(
        {
            "stats_schema": FEATURE_STATS_SCHEMA,
            "stats_schema_version": FEATURE_STATS_SCHEMA_VERSION,
            "stats_status": "complete" if full_dataset_pass else "smoke_only",
            "stats_coverage": {
                "full_dataset_pass": bool(full_dataset_pass),
                "exact_scene_gauge_scope": bool(exact_gauge_scope),
                "processed_batches": processed_batches,
                "expected_batches": int(expected_batches),
                "dataset_trunks": int(len(dataset)),
                "scene_gauge_requested_key_count": len(requested_keys),
                "scene_gauge_required_key_count": len(required_keys),
                "latent_count": actual_latent_count,
                "expected_latent_count": expected_latent_count,
                "exact_latent_count": bool(exact_latent_count),
                "max_batches": args.max_batches,
            },
            "source": {
                "image_dir": args.image_dir,
                "scene_names": scene_names,
                "sequence_length": int(args.sequence_length),
                "dggt_context_length": 29,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "precision": str(args.precision),
                "latent_dim": int(args.latent_dim),
                "max_batches": args.max_batches,
                "levels": list(DEFAULT_LEVELS),
                "patch_grid": list(patch_grid),
                "latent_stats_path": args.latent_stats_path,
                "scene_gauge_path": str(Path(args.scene_gauge_path).resolve()),
            },
        }
    )
    save_feature_stats(stats, args.output_path)
    print(
        f"[stats] saved {args.output_path} count={actual_latent_count} "
        f"mu_absmax={stats['mu_z'].abs().max().item():.6f} "
        f"sigma_mean={stats['sigma_z'].mean().item():.6f}",
        flush=True,
    )
    sidecar = Path(args.output_path).with_suffix(Path(args.output_path).suffix + ".json")
    sidecar.write_text(
        json.dumps({"args": vars(args), "count": actual_latent_count}, indent=2)
    )


if __name__ == "__main__":
    main()
