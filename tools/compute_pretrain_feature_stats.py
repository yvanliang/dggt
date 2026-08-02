"""Compute metric-gauge SceneFlow feature statistics in one dataset pass."""
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
from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_STATS_STD_FLOOR,
    CAMERA_STATS_VERSION,
    CAMERA_TARGET_SOURCE,
    CAMERA_TARGET_SPACE,
    camera_state_from_waymo_c2w,
)
from dggt.utils.feature_stats import (
    DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH,
    FEATURE_STATS_SCHEMA,
    FEATURE_STATS_SCHEMA_VERSION,
    checkpoint_sha256,
    compute_scene_gauge_stats,
    load_feature_stats,
    save_feature_stats,
    validate_production_stats_coverage,
    validate_tokenizer_stats_provenance,
)
from dggt.utils.tokens import batched_gather_frames, select_patch_pyramid
from dggt.utils.factorized_asset_condition import (
    FACTORIZED_ASSET_CONDITION_VERSION,
    PLACEMENT_PASSTHROUGH_CHANNELS,
    PLACEMENT_STANDARDIZED_CHANNELS,
    PLACEMENT_STATE_DIM,
    build_placement_state,
)
from dggt.utils.scene_gauge import SCENE_GAUGE_DIM
from dggt.utils.sliding_window import OFFLINE_MAX_SINGLE_WINDOW
from dggt.utils.tokenizer_window import encode_tokenizer_windowed


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


def load_reusable_latent_stats(
    path: str | Path,
    *,
    latent_dim: int,
    tokenizer_sha256: str,
    dggt_checkpoint_sha256: str,
    sequence_length: int,
    patch_grid: tuple[int, int],
    levels: tuple[int, ...] = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Load latent moments only when their tokenizer provenance is exact."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Reused latent stats at {path} must be a dict")
    # A smoke/subset latent pass must never be laundered into a production
    # artifact by combining it with a later full camera/gauge pass.
    validate_production_stats_coverage(payload)
    validate_tokenizer_stats_provenance(payload, tokenizer_sha256)
    if payload.get("dggt_checkpoint_sha256") != str(dggt_checkpoint_sha256):
        raise ValueError("Reused latent stats DGGT checkpoint mismatch")
    source = payload.get("source")
    expected_source = {
        "sequence_length": int(sequence_length),
        "dggt_context_length": 29,
        "patch_grid": list(map(int, patch_grid)),
        "levels": list(map(int, levels)),
    }
    if not isinstance(source, dict):
        raise ValueError("Reused latent stats are missing source provenance")
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(
                f"Reused latent stats source mismatch at {key}: "
                f"expected {expected!r}, got {source.get(key)!r}"
            )
    count = torch.as_tensor(payload.get("count", 0), dtype=torch.long)
    if count.numel() != 1 or int(count.item()) <= 0:
        raise ValueError("Reused latent stats require a positive scalar count")
    mu_z, sigma_z = load_feature_stats(path, token_dim=int(latent_dim))
    return {
        "mu_z": mu_z,
        "sigma_z": sigma_z,
        "count": count,
        "latent_stats_source_sha256": checkpoint_sha256(path),
    }


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
    # Load on CPU first. Feature-stat extraction only needs the aggregator and
    # tokenizer; moving every dense head to CUDA wastes tens of GiB and can make
    # the formal full pass impossible beside an in-flight tokenizer-v2 job.
    model = VGGT()
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
    missing_aggregator = [key for key in real_missing if key.startswith("aggregator.")]
    if missing_aggregator:
        raise RuntimeError(
            "DGGT checkpoint is missing aggregator weights required for feature "
            f"statistics: {_format_key_examples(missing_aggregator)}"
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


def _front_camera_sequence(value: torch.Tensor, *, name: str) -> torch.Tensor:
    sequence = torch.as_tensor(value).float()
    if sequence.ndim == 5 and int(sequence.shape[2]) == 1:
        sequence = sequence[:, :, 0]
    if sequence.ndim != 4 or tuple(sequence.shape[-2:]) != (4, 4):
        raise ValueError(
            f"{name} must be [B,S,4,4] or [B,S,1,4,4], got {tuple(sequence.shape)}"
        )
    return sequence


def _global_window_indices(batch: dict[str, Any], *, batch_size: int, seq_len: int) -> torch.Tensor:
    raw = batch.get("dggt_window_indices")
    if not torch.is_tensor(raw):
        raw = batch.get("frame_ids")
    if not torch.is_tensor(raw):
        raise RuntimeError("Metric camera stats require dggt_window_indices or frame_ids [B,S].")
    indices = torch.as_tensor(raw, dtype=torch.long)
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
    if tuple(indices.shape) != (batch_size, seq_len):
        raise ValueError(
            f"camera window indices shape {tuple(indices.shape)} != {(batch_size, seq_len)}"
        )
    return indices


@torch.no_grad()
def compute_dggt_stats_single_pass(
    model: VGGT | None,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    compute_latents: bool,
    dggt_checkpoint_sha256: str,
    scene_gauge_table_sha256: str,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    """Stream stats from the exact formal-training target representations."""

    if compute_latents and model is None:
        raise ValueError("A frozen DGGT/tokenizer model is required when computing latent stats")
    latent_total = 0
    latent_sum = torch.zeros(int(args.latent_dim), dtype=torch.float64)
    latent_sum2 = torch.zeros_like(latent_sum)
    camera_sums = {role: torch.zeros(CAMERA_GENERATION_DIM, dtype=torch.float64) for role in ("anchor", "delta")}
    camera_sums2 = {role: value.clone() for role, value in camera_sums.items()}
    camera_counts = {"anchor": 0, "delta": 0}
    gauge_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    placement_sum = torch.zeros(PLACEMENT_STATE_DIM, dtype=torch.float64)
    placement_sum2 = torch.zeros_like(placement_sum)
    placement_count = 0
    yielded = 0
    batches = loader if args.max_batches is None else islice(loader, int(args.max_batches))
    for batch_idx, batch in enumerate(batches):
        if args.require_dynamic_mask and ("dynamic_mask" not in batch or batch["dynamic_mask"] is None):
            raise RuntimeError(
                "Batch is missing dynamic_mask. Check fine_dynamic_masks/all coverage for selected scenes."
            )
        camera_raw = batch.get("camera_to_world_corrected")
        anchor_raw = batch.get("camera_trajectory_anchor_to_world_corrected")
        previous_raw = batch.get("camera_previous_to_world_corrected")
        if not all(torch.is_tensor(value) for value in (camera_raw, anchor_raw, previous_raw)):
            raise RuntimeError(
                "Metric camera stats require Waymo camera_to_world, trajectory anchor, and previous pose GT."
            )
        camera_to_world = _front_camera_sequence(
            camera_raw,
            name="camera_to_world_corrected",
        )
        batch_size, seq_len = int(camera_to_world.shape[0]), int(camera_to_world.shape[1])
        window_indices_cpu = _global_window_indices(
            batch,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        anchor_mask = window_indices_cpu.eq(0)
        camera_state_metric, anchor_mask = camera_state_from_waymo_c2w(
            camera_to_world,
            torch.as_tensor(anchor_raw).float(),
            previous_camera_to_world=torch.as_tensor(previous_raw).float(),
            anchor_mask=anchor_mask,
        )
        flat_camera = camera_state_metric.detach().cpu().double().reshape(
            -1, CAMERA_GENERATION_DIM
        )
        flat_anchor = anchor_mask.detach().cpu().bool().reshape(-1)
        for role, select in (("anchor", flat_anchor), ("delta", ~flat_anchor)):
            values = flat_camera[select]
            camera_counts[role] += int(values.shape[0])
            if values.numel():
                camera_sums[role] += values.sum(0)
                camera_sums2[role] += values.square().sum(0)

        gauge_raw = batch.get("scene_gauge")
        gauge_valid_raw = batch.get("scene_gauge_valid")
        if not torch.is_tensor(gauge_raw) or not torch.is_tensor(gauge_valid_raw):
            raise RuntimeError(
                "Feature stats require scene_gauge and scene_gauge_valid from the offline table."
            )
        gauge = torch.as_tensor(gauge_raw).detach().cpu().float()
        gauge_valid = torch.as_tensor(gauge_valid_raw).detach().cpu().bool()
        if gauge.ndim == 1:
            gauge = gauge.unsqueeze(0)
        if gauge_valid.ndim == 1:
            gauge_valid = gauge_valid.unsqueeze(0)
        if tuple(gauge.shape) != (batch_size, SCENE_GAUGE_DIM) or gauge_valid.shape != gauge.shape:
            raise ValueError(
                "scene_gauge/valid must be [B,3], got "
                f"{tuple(gauge.shape)} and {tuple(gauge_valid.shape)}"
            )
        gauge_batches.append((gauge, gauge_valid))

        center = batch.get("pretrain_object_center_anchor")
        size = batch.get("pretrain_object_box_size")
        yaw = batch.get("pretrain_object_yaw")
        velocity = batch.get("pretrain_object_velocity_anchor")
        valid_track = batch.get("pretrain_object_track_valid")
        in_frustum = batch.get("pretrain_object_in_frustum")
        if all(torch.is_tensor(value) for value in (center, size, yaw, velocity, valid_track, in_frustum)):
            placement_camera_raw = batch.get("pretrain_camera_to_anchor")
            if not torch.is_tensor(placement_camera_raw):
                raise RuntimeError(
                    "Factorized placement stats require pretrain_camera_to_anchor; "
                    "camera_to_world_corrected is in the ego world and must not be substituted."
                )
            placement_camera_to_anchor = _front_camera_sequence(
                placement_camera_raw,
                name="pretrain_camera_to_anchor",
            )
            if tuple(placement_camera_to_anchor.shape[:2]) != (batch_size, seq_len):
                raise ValueError(
                    "pretrain_camera_to_anchor batch/time shape does not match metric camera: "
                    f"{tuple(placement_camera_to_anchor.shape[:2])} != {(batch_size, seq_len)}"
                )
            placement = build_placement_state(
                torch.as_tensor(center).float(),
                torch.as_tensor(size).float(),
                torch.as_tensor(yaw).float(),
                torch.as_tensor(velocity).float(),
                torch.as_tensor(in_frustum).bool(),
                placement_camera_to_anchor,
            )
            selected = placement[torch.as_tensor(valid_track).bool()].double()
            if selected.numel():
                placement_count += int(selected.shape[0])
                placement_sum += selected.sum(dim=0)
                placement_sum2 += selected.square().sum(dim=0)
        z = None
        if compute_latents:
            context_raw = batch.get("dggt_context_images")
            if not torch.is_tensor(context_raw):
                raise RuntimeError(
                    "Latent stats require dggt_context_images from the complete 29-frame clip."
                )
            context_images = context_raw.to(device, non_blocking=True)
            if context_images.ndim != 5 or int(context_images.shape[1]) != 29:
                raise ValueError(
                    "Latent stats require dggt_context_images [B,29,3,H,W], got "
                    f"{tuple(context_images.shape)}"
                )
            window_indices = window_indices_cpu.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            assert model is not None
            with autocast_context(args, device):
                outputs = model.get_aggregator_token_outputs(context_images)
                image_tokens_all = [
                    batched_gather_frames(tokens, window_indices, name=f"image_tokens_list[{level}]")
                    for level, tokens in enumerate(outputs["image_tokens_list"])
                ]
                patch_start_idx = int(outputs["patch_start_idx"])
                del outputs
                image_patch = select_patch_pyramid(image_tokens_all, DEFAULT_LEVELS, patch_start_idx)
                patch_count = int(image_patch[-1].shape[-2])
                patch_grid = (
                    DEFAULT_PATCH_GRID
                    if patch_count == DEFAULT_PATCH_GRID[0] * DEFAULT_PATCH_GRID[1]
                    else infer_patch_grid(patch_count)
                )
                z = encode_tokenizer_windowed(
                    model.scene_tokenizer,
                    image_patch,
                    patch_grid=patch_grid,
                    window_len=OFFLINE_MAX_SINGLE_WINDOW,
                )
        if z is not None:
            flat = z.detach().cpu().double().reshape(-1, int(args.latent_dim))
            latent_total += int(flat.shape[0])
            latent_sum += flat.sum(0)
            latent_sum2 += flat.square().sum(0)
        yielded += 1
        if yielded % max(1, args.log_every) == 0:
            print(f"[stats] processed_batches={yielded}", flush=True)
    if camera_counts["anchor"] == 0 or camera_counts["delta"] == 0:
        raise ValueError("Metric camera stats require at least one anchor and one delta token")
    feature_stats: dict[str, Any] = {
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
        feature_stats[f"camera_{role}_mean"] = mean.float()
        feature_stats[f"camera_{role}_std"] = variance.sqrt().float().clamp_min(CAMERA_STATS_STD_FLOOR)
        feature_stats[f"camera_{role}_count"] = torch.tensor(count, dtype=torch.long)
    feature_stats.update(
        compute_scene_gauge_stats(
            gauge_batches,
            scene_gauge_table_sha256=scene_gauge_table_sha256,
        )
    )
    if placement_count <= 0:
        if bool(getattr(args, "require_factorized_placement", False)):
            raise ValueError(
                "factorized placement stats require at least one valid external-reference track"
            )
    else:
        placement_mean = placement_sum / float(placement_count)
        placement_var = (
            placement_sum2 / float(placement_count) - placement_mean.square()
        ).clamp_min(1.0e-6)
        passthrough = torch.tensor(PLACEMENT_PASSTHROUGH_CHANNELS, dtype=torch.long)
        placement_mean[passthrough] = 0.0
        placement_var[passthrough] = 1.0
        feature_stats.update(
            {
                "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
                "placement_dim": PLACEMENT_STATE_DIM,
                "placement_standardized_channels": list(PLACEMENT_STANDARDIZED_CHANNELS),
                "placement_passthrough_channels": list(PLACEMENT_PASSTHROUGH_CHANNELS),
                "placement_mean": placement_mean.float(),
                "placement_std": placement_var.sqrt().float(),
                "placement_count": torch.tensor(placement_count, dtype=torch.long),
            }
        )
    feature_stats["stats_processed_batches"] = int(yielded)
    if not compute_latents:
        return None, feature_stats
    if latent_total == 0:
        raise ValueError("Cannot compute latent stats from an empty loader")
    mean = latent_sum / float(latent_total)
    variance = (latent_sum2 / float(latent_total) - mean.square()).clamp_min(1.0e-6)
    return {
        "mu_z": mean.float(),
        "sigma_z": variance.sqrt().float(),
        "count": torch.tensor(latent_total, dtype=torch.long),
    }, feature_stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute SceneFlow latent, 9D metric camera, 3D gauge, and 16D placement stats."
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--dggt_ckpt_path", type=str, default=None)
    parser.add_argument("--tokenizer_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--scene_gauge_path",
        type=str,
        required=True,
        help="Immutable 29-frame offline scene-gauge JSON used by formal training.",
    )
    parser.add_argument(
        "--latent_stats_path",
        type=str,
        default=None,
        help=(
            "Reuse validated mu_z/sigma_z; camera/gauge/placement stats remain fully recomputed."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH),
        help="Output path shared by SceneFlow pretrain and formal training.",
    )
    parser.add_argument("--scene_list", type=str, default=None)
    parser.add_argument("--scene_names", type=str, default=None)
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=600)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument(
        "--camera_anchor_window_probability",
        type=float,
        default=0.5,
        help="Balance complete and delta-only metric camera target windows.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
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
    args.require_factorized_placement = True
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    if not args.dggt_ckpt_path:
        raise ValueError(
            "--dggt_ckpt_path is required for feature-stats provenance and latent extraction"
        )
    if not args.tokenizer_ckpt_path:
        raise ValueError(
            "--tokenizer_ckpt_path is required so latent statistics are bound to exact tokenizer weights"
        )
    dggt_sha256 = checkpoint_sha256(args.dggt_ckpt_path)
    tokenizer_sha256 = checkpoint_sha256(args.tokenizer_ckpt_path)
    gauge_table_sha256 = checkpoint_sha256(args.scene_gauge_path)
    compute_latents = args.latent_stats_path is None
    model = (
        load_dggt_aggregator_and_tokenizer(
            args.dggt_ckpt_path,
            args.tokenizer_ckpt_path,
            device,
        )
        if compute_latents
        else None
    )
    dataset = WaymoOpenDataset(
        image_dir=args.image_dir,
        scene_names=parse_scene_names(args),
        sequence_length=args.sequence_length,
        mode=1,
        views=1,
        trunk_frames=29,
        trunk_major_samples=True,
        camera_anchor_window_probability=float(args.camera_anchor_window_probability),
        return_full_dggt_context=compute_latents,
        scene_gauge_path=args.scene_gauge_path,
        expected_scene_gauge_dggt_sha256=dggt_sha256,
        expected_scene_gauge_split=Path(args.image_dir).name,
    )
    if dataset.scene_gauge_sha256 != gauge_table_sha256:
        raise RuntimeError(
            "Dataset scene-gauge SHA disagrees with the directly hashed table: "
            f"dataset={dataset.scene_gauge_sha256!r}, direct={gauge_table_sha256!r}"
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
        f"[stats] trunks={len(dataset)} batch_size={args.batch_size} seq={args.sequence_length} "
        f"max_batches={args.max_batches}",
        flush=True,
    )

    computed_latent_stats, representation_stats = compute_dggt_stats_single_pass(
        model,
        loader,
        args,
        device,
        compute_latents=compute_latents,
        dggt_checkpoint_sha256=dggt_sha256,
        scene_gauge_table_sha256=gauge_table_sha256,
    )
    if args.latent_stats_path is not None:
        stats = load_reusable_latent_stats(
            args.latent_stats_path,
            latent_dim=int(args.latent_dim),
            tokenizer_sha256=tokenizer_sha256,
            dggt_checkpoint_sha256=dggt_sha256,
            sequence_length=int(args.sequence_length),
            patch_grid=tuple(dataset.pretrain_patch_grid),
        )
        print(f"[stats] reused latent stats from {args.latent_stats_path}", flush=True)
    else:
        assert computed_latent_stats is not None
        stats = computed_latent_stats
    stats.update(representation_stats)
    stats["tokenizer_checkpoint_sha256"] = tokenizer_sha256
    requested_keys = set(dataset.scene_gauge_requested_keys or ())
    required_keys = set(dataset.scene_gauge_required_keys or ())
    exact_scene_gauge_scope = bool(requested_keys) and requested_keys == required_keys
    expected_batches = len(loader)
    processed_batches = int(stats.get("stats_processed_batches", -1))
    patch_grid = tuple(map(int, dataset.pretrain_patch_grid))
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
        and exact_scene_gauge_scope
        and exact_latent_count
    )
    stats.update(
        {
            "stats_schema": FEATURE_STATS_SCHEMA,
            "stats_schema_version": FEATURE_STATS_SCHEMA_VERSION,
            "stats_status": "complete" if full_dataset_pass else "smoke_only",
            "stats_coverage": {
                "full_dataset_pass": bool(full_dataset_pass),
                "exact_scene_gauge_scope": bool(exact_scene_gauge_scope),
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
        }
    )
    stats["source"] = {
        "image_dir": args.image_dir,
        "scene_names": parse_scene_names(args),
        "sequence_length": int(args.sequence_length),
        "dggt_context_length": 29,
        "tokenizer_checkpoint_sha256": tokenizer_sha256,
        "camera_anchor_window_probability": float(args.camera_anchor_window_probability),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "precision": str(args.precision),
        "require_dynamic_mask": bool(args.require_dynamic_mask),
        "latent_dim": int(args.latent_dim),
        "max_batches": args.max_batches,
        "levels": list(DEFAULT_LEVELS),
        "patch_grid": list(patch_grid),
        "latent_stats_path": args.latent_stats_path,
        "scene_gauge_path": str(Path(args.scene_gauge_path).resolve()),
        "scene_gauge_table_sha256": gauge_table_sha256,
        "camera_target_source": CAMERA_TARGET_SOURCE,
        "camera_target_space": CAMERA_TARGET_SPACE,
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
