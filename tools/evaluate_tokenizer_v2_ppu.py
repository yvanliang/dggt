#!/usr/bin/env python3
"""Distributed checkpoint sweep for JointSceneTokenizer v2.

The unit of parallelism is a (checkpoint, frame-count) configuration.  With
the default five checkpoints and 10/12/14-frame inputs there are 15 configs;
torchrun rank ``r`` evaluates configs whose index is congruent to ``r``.  This
keeps every metric for one config on one device and avoids counting distributed
padding or overlapping windows as independent observations.

Formal mode follows the Stage-A visualization path (raw RGB -> frozen DGGT
tokens -> tokenizer encode/decode -> frozen heads -> Gaussian renderer), while
running DepthHead/GaussianHead in fp32 as required by the metric-gauge audit.
It reports raw-video reconstruction quality and paired, same-pixel 3D gauge
metrics.  ``--mock`` exercises the same scheduling, aggregation, bootstrap and
artifact-writing code without data, checkpoints, CUDA, LPIPS, or gsplat.
Mock numbers are deliberately marked non-scientific.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "tokenizer_v2_ppu_checkpoint_sweep"
SCHEMA_VERSION = "1.0.0"
# Round 1: five checkpoints x three frame counts = 15 independent configs,
# which fit in one 16-PPU torchrun wave.  To run the planned second wave,
# replace this tuple (or pass --steps) with:
#     (80_000, 85_000, 90_000, 95_000, 100_000)
DEFAULT_STEPS = (55_000, 60_000, 65_000, 70_000, 75_000)
DEFAULT_FRAME_COUNTS = (10, 12, 14)


@dataclass(frozen=True)
class EvalConfig:
    index: int
    step: int
    checkpoint: Path | None
    frame_count: int

    @property
    def key(self) -> str:
        return f"step_{self.step:06d}_frames_{self.frame_count:02d}"


def _parse_int_csv(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates: {values}")
    return values


def build_configs(checkpoint_dir: Path, steps: Sequence[int], frame_counts: Sequence[int], *, mock: bool) -> list[EvalConfig]:
    configs: list[EvalConfig] = []
    for step in steps:
        checkpoint = None if mock else checkpoint_dir / f"scene_tokenizer_step_{step:06d}.pt"
        for frame_count in frame_counts:
            configs.append(EvalConfig(len(configs), int(step), checkpoint, int(frame_count)))
    return configs


def assigned_configs(configs: Sequence[EvalConfig], rank: int, world_size: int) -> list[EvalConfig]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid distributed identity rank={rank}, world_size={world_size}")
    return [config for config in configs if config.index % world_size == rank]


def _setup_worker_device(*, force_cpu: bool = False) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available() and not force_cpu:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _wait_for_file(path: Path, *, timeout_sec: float, poll_sec: float = 5.0) -> None:
    deadline = time.time() + timeout_sec
    while not path.is_file():
        if time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for file: {path}")
        time.sleep(poll_sec)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _seed_case(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value}")
    return result


def _frame_balanced(values: torch.Tensor, mask: torch.Tensor, *, reduction: str = "median") -> float:
    if values.shape != mask.shape or values.ndim < 2:
        raise ValueError(f"values/mask mismatch: {tuple(values.shape)} vs {tuple(mask.shape)}")
    rows: list[torch.Tensor] = []
    for frame in range(values.shape[0]):
        selected = values[frame][mask[frame]]
        if selected.numel() == 0:
            continue
        if reduction == "median":
            rows.append(selected.float().median())
        elif reduction == "mean":
            rows.append(selected.float().mean())
        elif reduction == "rmse":
            rows.append(selected.float().square().mean().sqrt())
        else:
            raise ValueError(reduction)
    if not rows:
        raise RuntimeError("metric support is empty for every frame")
    return _finite(torch.stack(rows).median().item())


def _ssim_per_frame(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """11x11-window RGB SSIM, averaged independently within each frame."""
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mu_x = F.avg_pool2d(pred, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(pred * pred, 11, stride=1, padding=5) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, 11, stride=1, padding=5) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return score.flatten(1).mean(dim=1)


def _quality_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_model: torch.nn.Module | None,
    lpips_chunk: int,
    prefix: str,
) -> dict[str, float]:
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"quality tensors must match [S,3,H,W], got {pred.shape}, {target.shape}")
    mse = (pred - target).square().flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    result = {
        f"{prefix}_psnr_db": _finite(psnr.mean().item()),
        f"{prefix}_ssim": _finite(_ssim_per_frame(pred, target).mean().item()),
    }
    if lpips_model is None:
        # Mock-only deterministic perceptual surrogate.  Formal mode requires LPIPS.
        result[f"{prefix}_lpips"] = _finite((pred - target).abs().mean().item())
    else:
        scores = []
        for start in range(0, pred.shape[0], lpips_chunk):
            scores.append(
                lpips_model(
                    pred[start : start + lpips_chunk] * 2 - 1,
                    target[start : start + lpips_chunk] * 2 - 1,
                ).reshape(-1)
            )
        result[f"{prefix}_lpips"] = _finite(torch.cat(scores).mean().item())
    if pred.shape[0] > 1:
        pred_delta = pred[1:] - pred[:-1]
        target_delta = target[1:] - target[:-1]
        result[f"{prefix}_temporal_delta_l1"] = _finite((pred_delta - target_delta).abs().mean().item())
    else:
        result[f"{prefix}_temporal_delta_l1"] = 0.0
    return result


def _resize_mask(mask: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    # input [1,S,C,H,W] -> [S,H,W]
    flat = mask[:, :, :1].float().flatten(0, 1)
    if flat.shape[-2:] != hw:
        flat = F.interpolate(flat, size=hw, mode="nearest")
    return flat[:, 0] >= 0.5


def _geometry_metrics(
    direct: Mapping[str, torch.Tensor],
    recon: Mapping[str, torch.Tensor],
    sky_mask: torch.Tensor,
    dynamic_mask: torch.Tensor,
    *,
    opacity_threshold: float,
    pose_enc: torch.Tensor | None = None,
    image_hw: tuple[int, int] | None = None,
) -> dict[str, float]:
    direct_depth = direct["depth"][0, ..., 0].float()
    recon_depth = recon["depth"][0, ..., 0].float()
    direct_gs = direct["gs_map"][0].float()
    recon_gs = recon["gs_map"][0].float()
    hw = tuple(int(value) for value in direct_depth.shape[-2:])
    exclude = _resize_mask(sky_mask, hw) | _resize_mask(dynamic_mask, hw)
    direct_opacity, recon_opacity = direct_gs[..., 3], recon_gs[..., 3]
    finite = (
        torch.isfinite(direct_depth)
        & torch.isfinite(recon_depth)
        & torch.isfinite(direct_gs).all(dim=-1)
        & torch.isfinite(recon_gs).all(dim=-1)
    )
    support = (
        finite
        & (direct_depth > 0)
        & (recon_depth > 0)
        & (direct_opacity > opacity_threshold)
        & (recon_opacity > opacity_threshold)
        & ~exclude
    )
    if min(int(row.sum()) for row in support) < 32:
        raise RuntimeError("fewer than 32 valid static/non-sky pixels in at least one frame")

    log_depth = torch.log(recon_depth.clamp_min(1e-8)) - torch.log(direct_depth.clamp_min(1e-8))
    depth_absrel = (recon_depth - direct_depth).abs() / direct_depth.clamp_min(1e-8)
    depth_abs = (recon_depth - direct_depth).abs()
    direct_scale = direct_gs[..., 4:7].clamp_min(1e-5)
    recon_scale = recon_gs[..., 4:7].clamp_min(1e-5)
    log_axes = torch.log(recon_scale) - torch.log(direct_scale)
    log_gs = log_axes.mean(dim=-1)
    anisotropy = ((log_axes - log_gs.unsqueeze(-1)).square().mean(dim=-1)).sqrt()
    paired = log_gs - log_depth
    if pose_enc is not None:
        if image_hw is None:
            raise ValueError("image_hw is required when pose_enc is supplied")
        from dggt.utils.pose_enc import pose_encoding_to_extri_intri

        _extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc.float(), image_hw)
        intrinsics = intrinsics[0].float()
        yy, xx = torch.meshgrid(
            torch.arange(hw[0], device=direct_depth.device, dtype=torch.float32),
            torch.arange(hw[1], device=direct_depth.device, dtype=torch.float32),
            indexing="ij",
        )
        fx = intrinsics[:, 0, 0].view(-1, 1, 1)
        fy = intrinsics[:, 1, 1].view(-1, 1, 1)
        cx = intrinsics[:, 0, 2].view(-1, 1, 1)
        cy = intrinsics[:, 1, 2].view(-1, 1, 1)
        ray_norm = torch.sqrt(((xx - cx) / fx).square() + ((yy - cy) / fy).square() + 1.0)
    else:
        # Synthetic mock rays point along +z.
        ray_norm = torch.ones_like(direct_depth)
    point_xyz_abs = depth_abs * ray_norm
    point_xyz_direct_norm = direct_depth * ray_norm
    return {
        "depth_recon_over_direct": math.exp(_frame_balanced(log_depth, support)),
        "depth_recon_vs_direct_absrel": _frame_balanced(depth_absrel, support),
        "depth_recon_vs_direct_log_rmse": _frame_balanced(log_depth, support, reduction="rmse"),
        "point_xyz_error_dggt": _frame_balanced(point_xyz_abs, support),
        "point_xyz_relative_error": _frame_balanced(
            point_xyz_abs / point_xyz_direct_norm.clamp_min(1e-8), support
        ),
        "gs_recon_over_direct": math.exp(_frame_balanced(log_gs, support)),
        "paired_gs_over_depth": math.exp(_frame_balanced(paired, support)),
        "gs_axis_anisotropy_log_rms": _frame_balanced(anisotropy, support),
        "support_pixels_median_per_frame": _finite(
            torch.stack([row.sum() for row in support]).float().median().item()
        ),
    }


def _sample_map_at_lidar(prediction: torch.Tensor, lidar_depths: Sequence[np.ndarray]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sample [S,H,W] dense values at original sparse LiDAR cell centers."""
    sampled: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frame, lidar in enumerate(lidar_depths):
        valid = np.isfinite(lidar) & (lidar > 0)
        rows, cols = np.nonzero(valid)
        if rows.size < 16:
            raise RuntimeError(f"only {rows.size} valid LiDAR cells in frame {frame}")
        lidar_h, lidar_w = lidar.shape
        x = 2 * (torch.as_tensor(cols, device=prediction.device, dtype=torch.float32) + 0.5) / lidar_w - 1
        y = 2 * (torch.as_tensor(rows, device=prediction.device, dtype=torch.float32) + 0.5) / lidar_h - 1
        grid = torch.stack((x, y), dim=-1).view(1, 1, -1, 2)
        values = F.grid_sample(
            prediction[frame].float().view(1, 1, *prediction.shape[-2:]),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(-1)
        gt = torch.as_tensor(lidar[rows, cols], device=prediction.device, dtype=torch.float32)
        sampled.append((values, gt))
    return sampled


def _lidar_metrics(
    direct_depth: torch.Tensor,
    recon_depth: torch.Tensor,
    lidar_depths: Sequence[np.ndarray],
) -> dict[str, float]:
    direct_rows = _sample_map_at_lidar(direct_depth, lidar_depths)
    recon_rows = _sample_map_at_lidar(recon_depth, lidar_depths)
    paired_direct: list[tuple[torch.Tensor, torch.Tensor]] = []
    paired_recon: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frame, ((direct, gt_direct), (recon, gt_recon)) in enumerate(zip(direct_rows, recon_rows)):
        if not torch.equal(gt_direct, gt_recon):
            raise RuntimeError(f"direct/recon LiDAR cells are not aligned in frame {frame}")
        keep = (
            torch.isfinite(direct)
            & (direct > 0)
            & torch.isfinite(recon)
            & (recon > 0)
            & torch.isfinite(gt_direct)
            & (gt_direct > 0)
        )
        if int(keep.sum()) < 16:
            raise RuntimeError(f"only {int(keep.sum())} shared valid LiDAR cells in frame {frame}")
        paired_direct.append((direct[keep], gt_direct[keep]))
        paired_recon.append((recon[keep], gt_direct[keep]))
    direct_rows, recon_rows = paired_direct, paired_recon
    frame_scales = [(pred / gt).median() for pred, gt in direct_rows]
    scale = torch.stack(frame_scales).median().clamp_min(1e-8)

    def summarize(rows: Sequence[tuple[torch.Tensor, torch.Tensor]], name: str) -> dict[str, float]:
        absrel, rmse, delta, ratios = [], [], [], []
        for pred, gt in rows:
            pred_m = pred / scale
            ratio = pred_m / gt
            absrel.append((pred_m - gt).abs().div(gt).median())
            rmse.append((pred_m - gt).square().mean().sqrt())
            delta.append((torch.maximum(ratio, 1 / ratio) < 1.25).float().mean())
            ratios.append(torch.log(ratio.clamp_min(1e-8)).median())
        return {
            f"lidar_{name}_absrel": _finite(torch.stack(absrel).median().item()),
            f"lidar_{name}_rmse_m": _finite(torch.stack(rmse).median().item()),
            f"lidar_{name}_delta1": _finite(torch.stack(delta).median().item()),
            f"lidar_{name}_over_gt": math.exp(_finite(torch.stack(ratios).median().item())),
        }

    result = {"lidar_direct_scale_dggt_per_m": _finite(scale.item())}
    result.update(summarize(direct_rows, "direct"))
    result.update(summarize(recon_rows, "recon"))
    return result


def _load_lidar_for_sample(processed_root: Path, record: Mapping[str, Any], frame_indices: Sequence[int]) -> list[np.ndarray]:
    scene_root = processed_root / str(record.get("source_split", "training")) / str(record["scene_dir"])
    result = []
    for frame in frame_indices:
        path = scene_root / "depth_flows_4" / f"{int(frame):03d}_0.npy"
        payload = np.load(path, mmap_mode="r", allow_pickle=False)
        if payload.ndim != 3 or payload.shape[-1] < 1:
            raise ValueError(f"bad depth-flow payload {path}: {payload.shape}")
        result.append(np.asarray(payload[..., 0]))
    return result


def _strict_load_tokenizer(module: torch.nn.Module, path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        payload = torch.load(path, map_location="cpu")
    state: Any = payload.get("scene_tokenizer", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError(f"unsupported tokenizer checkpoint: {path}")
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("scene_tokenizer."):
            key = key[len("scene_tokenizer.") :]
        cleaned[key] = value
    expected = module.state_dict()
    missing = sorted(set(expected) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(expected))
    bad_shapes = sorted(key for key in set(expected) & set(cleaned) if expected[key].shape != cleaned[key].shape)
    if missing or unexpected or bad_shapes:
        raise RuntimeError(
            f"strict tokenizer load failed: missing={missing[:6]} ({len(missing)}), "
            f"unexpected={unexpected[:6]} ({len(unexpected)}), shapes={bad_shapes[:6]} ({len(bad_shapes)})"
        )
    module.load_state_dict(cleaned, strict=True)
    del payload, state, cleaned


def _select_cases(dataset: Any, count: int) -> list[int]:
    """Select one clip per scene, evenly covering the deterministic record order."""
    unique, seen = [], set()
    for index, record in enumerate(dataset.samples):
        scene = str(record.get("scene_base", record.get("scene_name", record.get("scene_dir"))))
        if scene in seen:
            continue
        seen.add(scene)
        unique.append(index)
    if len(unique) < count:
        raise RuntimeError(f"requested {count} unique scenes, dataset provides only {len(unique)}")
    # Interval sampling prevents the first lexical/shuffled region from dominating.
    positions = np.linspace(0, len(unique) - 1, num=count, dtype=np.int64)
    selected = [unique[int(position)] for position in positions]
    if len(set(selected)) != count:
        raise RuntimeError("interval scene selection unexpectedly produced duplicate records")
    return selected


def _selection_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "tokenizer_v2_fixed_scene_selection",
        "schema_version": "1.0.0",
        "num_scenes": int(args.num_scenes),
        "dataset_split": str(args.dataset_split),
        "split_seed": int(args.split_seed),
        "clean_train_ratio": float(args.clean_train_ratio),
        "sample_window": int(args.sample_window),
        "sampling_seed": int(args.seed),
        "frame_counts": list(args.frame_counts_values),
        "selection": "one clip per unique scene, interval-sampled over deterministic dataset record order",
    }


def _create_selection_manifest(dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    contract = _selection_contract(args)
    cases = []
    for ordinal, dataset_index in enumerate(_select_cases(dataset, args.num_scenes)):
        record = dataset.samples[dataset_index]
        clip_frames = [int(value) for value in record["scene_frame_indices"]]
        present, editable = dataset._build_lightweight_sampling_flags(record)
        selections: dict[str, Any] = {}
        for frame_count in args.frame_counts_values:
            case_seed = int(args.seed + dataset_index * 1009 + frame_count * 100_003)
            _seed_case(case_seed)
            local_indices, _intervals = dataset._sample_local_indices(
                0,
                len(clip_frames),
                present,
                editable,
                sample_num_frames=frame_count,
            )
            selections[str(frame_count)] = {
                "sampling_seed": case_seed,
                "local_frame_indices": [int(value) for value in local_indices],
                "global_frame_indices": [clip_frames[int(value)] for value in local_indices],
                "start_local_frame": int(local_indices[0]),
                "start_global_frame": int(clip_frames[int(local_indices[0])]),
            }
        cases.append({
            "case_ordinal": ordinal,
            "dataset_index": int(dataset_index),
            "scene": str(record.get("scene_base", record.get("scene_name"))),
            "clip": str(record.get("clip_name")),
            "source_split": str(record.get("source_split")),
            "scene_dir": str(record.get("scene_dir")),
            "clip_index": int(record.get("clip_index", -1)),
            "selections": selections,
        })
    return {**contract, "cases": cases}


def _load_or_create_selection_manifest(
    dataset: Any,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, Any]:
    path = args.selection_manifest
    if rank == 0 and not path.is_file():
        payload = _create_selection_manifest(dataset, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        print(f"[selection] created immutable scene/frame manifest: {path}", flush=True)
    if rank != 0:
        _wait_for_file(path, timeout_sec=float(args.filesystem_sync_timeout_sec))
    if not path.is_file():
        raise FileNotFoundError(f"selection manifest was not created: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = _selection_contract(args)
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "selection manifest contract mismatch; refusing to change scenes/frames: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    if len(payload.get("cases", [])) != args.num_scenes:
        raise RuntimeError("selection manifest case count is inconsistent")
    # Bind each persisted record back to the current dataset and fail on reorder/drift.
    for case in payload["cases"]:
        index = int(case["dataset_index"])
        if not 0 <= index < len(dataset.samples):
            raise RuntimeError(f"persisted dataset_index is out of range: {index}")
        record = dataset.samples[index]
        actual = (str(record.get("scene_base", record.get("scene_name"))), str(record.get("clip_name")))
        expected_id = (str(case["scene"]), str(case["clip"]))
        if actual != expected_id:
            raise RuntimeError(f"dataset record drift at index {index}: expected {expected_id}, got {actual}")
    return payload


def _save_visual(path: Path, gt: torch.Tensor, recon: torch.Tensor, direct: torch.Tensor, max_frames: int = 6) -> None:
    from train_tokenizer import build_stage_a_validation_grid

    path.parent.mkdir(parents=True, exist_ok=True)
    build_stage_a_validation_grid(gt, recon, direct, max_frames=max_frames).save(path)


def _real_data_runtime(args: argparse.Namespace, rank: int) -> dict[str, Any]:
    # Lazy import keeps --mock independent of dataset dependencies.
    from datasets import WaymoEditDataset

    dataset = WaymoEditDataset(
        processed_root=str(args.processed_root),
        transfer_root=str(args.transfer_root),
        raw_root=str(args.raw_root),
        asset_root=str(args.asset_root),
        split=args.dataset_split,
        sequence_length=max(args.frame_counts_values),
        mode=1,
        views=1,
        sample_window=args.sample_window,
        clean_only=True,
        clean_split_seed=args.split_seed,
        clean_train_ratio=args.clean_train_ratio,
    )
    selection_manifest = _load_or_create_selection_manifest(dataset, args, rank)
    return {
        "dataset": dataset,
        "case_entries": selection_manifest["cases"],
        "selection_manifest": selection_manifest,
    }


def _real_runtime(args: argparse.Namespace, device: torch.device, data_runtime: Mapping[str, Any]) -> dict[str, Any]:
    # Lazy import keeps --mock independent of CUDA extensions and model weights.
    import lpips
    from dggt.models.vggt import VGGT
    from dggt.utils.tokens import select_patch_pyramid
    from train_tokenizer import (
        build_head_outputs_from_patch_features,
        extract_levels,
        infer_patch_grid,
        load_model_checkpoint,
    )

    model = VGGT().to(device)
    load_model_checkpoint(model, str(args.dggt_checkpoint))
    model.eval().requires_grad_(False)
    levels = extract_levels(model)
    perceptual = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    return {
        "model": model,
        "levels": levels,
        "dataset": data_runtime["dataset"],
        "case_entries": data_runtime["case_entries"],
        "selection_manifest": data_runtime["selection_manifest"],
        "lpips": perceptual,
        "select_patch_pyramid": select_patch_pyramid,
        "build_heads": build_head_outputs_from_patch_features,
        "infer_patch_grid": infer_patch_grid,
    }


@torch.inference_mode()
def _evaluate_real_config(
    config: EvalConfig,
    runtime: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
) -> list[dict[str, Any]]:
    model = runtime["model"]
    dataset = runtime["dataset"]
    levels = runtime["levels"]
    assert config.checkpoint is not None
    _strict_load_tokenizer(model.scene_tokenizer, config.checkpoint)
    model.scene_tokenizer.eval()
    rows: list[dict[str, Any]] = []

    for ordinal, case_entry in enumerate(runtime["case_entries"]):
        dataset_index = int(case_entry["dataset_index"])
        frozen_selection = case_entry["selections"][str(config.frame_count)]
        case_seed = int(frozen_selection["sampling_seed"])
        _seed_case(case_seed)
        sample = dataset[(dataset_index, config.frame_count)]
        actual_frames = [int(value) for value in sample["frame_indices"].tolist()]
        expected_frames = [int(value) for value in frozen_selection["global_frame_indices"]]
        actual_local = [int(value) for value in sample["local_frame_indices"].tolist()]
        expected_local = [int(value) for value in frozen_selection["local_frame_indices"]]
        if actual_frames != expected_frames or actual_local != expected_local:
            raise RuntimeError(
                f"frozen frame selection drift for {case_entry['scene']} {config.frame_count}f: "
                f"expected local/global={expected_local}/{expected_frames}, "
                f"got {actual_local}/{actual_frames}"
            )
        images = sample["images_clean"].unsqueeze(0).to(device)
        sky = sample["sky_mask"].unsqueeze(0).to(device)
        dynamic = sample["dynamic_mask"].unsqueeze(0).to(device)
        timestamps = sample["timestamps"].unsqueeze(0).to(device)

        with _autocast(device, args.precision):
            if hasattr(model, "extract_scene_tokens"):
                agg_all, image_all, _dino_all, _, patch_start_idx = model.extract_scene_tokens(images)
            else:
                agg_all, image_all, _dino_all, _, patch_start_idx = model.aggregator(images)
            pose_enc = model.camera_head(agg_all)[-1]
            selected_tokenizer = runtime["select_patch_pyramid"](image_all, levels, patch_start_idx)
            selected = [tensor.float() for tensor in selected_tokenizer]
            selected_full = [image_all[level].float() for level in levels]
        teacher = {
            "num_levels": len(image_all),
            "image_levels": selected_full,
            "image_patch": selected,
            "patch_start_idx": int(patch_start_idx),
            "pose_enc": pose_enc.float(),
        }
        del agg_all, image_all, _dino_all

        with torch.autocast(device_type=device.type, enabled=False):
            direct = runtime["build_heads"](model, images.float(), levels, teacher, selected)
        patch_grid = runtime["infer_patch_grid"](images, selected[0].shape[2])
        with _autocast(device, args.precision):
            latent = model.scene_tokenizer.encode(selected_tokenizer, patch_grid=patch_grid)
            decoded = model.scene_tokenizer.decode(latent, patch_grid=patch_grid)
        with torch.autocast(device_type=device.type, enabled=False):
            recon = runtime["build_heads"](
                model, images.float(), levels, teacher, [tensor.float() for tensor in decoded]
            )

        from train_tokenizer import render_head_outputs_for_visual

        render_sample = {"sky_mask": sky, "timestamps": timestamps, "dynamic_mask": dynamic}
        render_direct = render_head_outputs_for_visual(
            model, images, render_sample, teacher["pose_enc"], direct, device
        )[0]
        render_recon = render_head_outputs_for_visual(
            model, images, render_sample, teacher["pose_enc"], recon, device
        )[0]
        gt = images[0]
        metrics = {}
        metrics.update(_quality_metrics(render_recon, gt, lpips_model=runtime["lpips"], lpips_chunk=args.lpips_chunk, prefix="render_gt"))
        metrics.update(_quality_metrics(render_direct, gt, lpips_model=runtime["lpips"], lpips_chunk=args.lpips_chunk, prefix="direct_gt"))
        metrics.update(_quality_metrics(render_recon, render_direct, lpips_model=runtime["lpips"], lpips_chunk=args.lpips_chunk, prefix="render_direct"))
        metrics.update(
            _geometry_metrics(
                direct,
                recon,
                sky,
                dynamic,
                opacity_threshold=args.opacity_threshold,
                pose_enc=teacher["pose_enc"],
                image_hw=tuple(int(value) for value in images.shape[-2:]),
            )
        )
        record = dataset.samples[dataset_index]
        frame_indices = actual_frames
        lidar = _load_lidar_for_sample(args.processed_root, record, frame_indices)
        metrics.update(_lidar_metrics(direct["depth"][0, ..., 0], recon["depth"][0, ..., 0], lidar))
        metrics["latent_mean"] = _finite(latent.float().mean().item())
        metrics["latent_std"] = _finite(latent.float().std(unbiased=False).item())

        row = {
            "config": config.key,
            "step": config.step,
            "frame_count": config.frame_count,
            "dataset_index": int(dataset_index),
            "scene": str(record.get("scene_base", record.get("scene_name"))),
            "clip": str(record.get("clip_name")),
            "frame_indices": frame_indices,
            "metrics": metrics,
        }
        rows.append(row)
        if ordinal < args.save_visuals_per_config:
            _save_visual(
                args.output_dir / "visuals" / config.key / f"{ordinal:02d}_{row['scene']}.jpg",
                gt,
                render_recon,
                render_direct,
            )
        if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(runtime["case_entries"]):
            print(f"[rank {rank}] {config.key}: {ordinal + 1}/{len(runtime['case_entries'])}", flush=True)
        del sample, images, sky, dynamic, timestamps, teacher, direct, recon, latent, decoded
        del selected, selected_full, selected_tokenizer, render_sample
        del render_direct, render_recon, gt, lidar
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _mock_case(config: EvalConfig, case_index: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed + config.frame_count * 1009 + case_index)
    frames, height, width = config.frame_count, 24, 32
    gt = torch.rand((frames, 3, height, width), generator=generator)
    direct_render = (gt + 0.035 * torch.randn(gt.shape, generator=generator)).clamp(0, 1)
    # Synthetic curve improves through 90k then regresses slightly.  Thus the
    # first-round smoke should choose 75k and a later second-round smoke 90k.
    maturity = {
        55_000: 1.50,
        60_000: 1.25,
        65_000: 1.05,
        70_000: 0.88,
        75_000: 0.78,
        80_000: 0.74,
        85_000: 0.72,
        90_000: 0.70,
        95_000: 0.72,
        100_000: 0.80,
    }.get(config.step, 1.0)
    recon_render = (direct_render + 0.018 * maturity * torch.randn(gt.shape, generator=generator)).clamp(0, 1)
    direct_depth = torch.rand((1, frames, height, width, 1), generator=generator) * 2 + 1
    depth_bias = 1 + 0.03 * (maturity - 0.72) + 0.001 * torch.randn((), generator=generator)
    recon_depth = direct_depth * depth_bias
    direct_gs = torch.rand((1, frames, height, width, 11), generator=generator)
    direct_gs[..., 3] = 0.8
    direct_gs[..., 4:7] = torch.rand((1, frames, height, width, 3), generator=generator) * 0.01 + 0.01
    paired_bias = 1 - 0.11 * (maturity - 0.72)
    recon_gs = direct_gs.clone()
    recon_gs[..., 4:7] *= depth_bias * paired_bias
    sky = torch.zeros((1, frames, 3, height, width))
    dynamic = torch.zeros_like(sky)
    direct = {"depth": direct_depth, "gs_map": direct_gs}
    recon = {"depth": recon_depth, "gs_map": recon_gs}
    metrics = {}
    metrics.update(_quality_metrics(recon_render, gt, lpips_model=None, lpips_chunk=8, prefix="render_gt"))
    metrics.update(_quality_metrics(direct_render, gt, lpips_model=None, lpips_chunk=8, prefix="direct_gt"))
    metrics.update(_quality_metrics(recon_render, direct_render, lpips_model=None, lpips_chunk=8, prefix="render_direct"))
    metrics.update(_geometry_metrics(direct, recon, sky, dynamic, opacity_threshold=0.05))
    # Synthetic metre GT and direct DGGT gauge.
    lidar = []
    direct_dense = direct_depth[0, ..., 0]
    recon_dense = recon_depth[0, ..., 0]
    for frame in range(frames):
        gt_m = np.asarray((direct_dense[frame] / 0.03).cpu().tolist(), dtype=np.float32)
        lidar.append(gt_m)
    metrics.update(_lidar_metrics(direct_dense, recon_dense, lidar))
    metrics["latent_mean"] = 0.0
    metrics["latent_std"] = 1.0
    return {
        "config": config.key,
        "step": config.step,
        "frame_count": config.frame_count,
        "dataset_index": case_index,
        "scene": f"mock_scene_{case_index:03d}",
        "clip": f"mock_clip_{case_index:03d}",
        "frame_indices": list(range(frames)),
        "metrics": metrics,
    }


def _bootstrap_mean(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or samples <= 0:
        value = float(array.mean())
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(samples, array.size))
    means = array[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize_rows(rows: Sequence[Mapping[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[str(row["config"])].append(row)
    configs = []
    for config_key in sorted(by_config, key=lambda key: (by_config[key][0]["step"], by_config[key][0]["frame_count"])):
        config_rows = by_config[config_key]
        scenes = [str(row["scene"]) for row in config_rows]
        if len(set(scenes)) != len(scenes):
            raise RuntimeError(f"{config_key} contains repeated scenes; scene-balanced inference would be invalid")
        metric_names = sorted(set.intersection(*(set(row["metrics"]) for row in config_rows)))
        metric_summary = {}
        for metric_index, name in enumerate(metric_names):
            values = [_finite(row["metrics"][name]) for row in config_rows]
            low, high = _bootstrap_mean(values, bootstrap_samples, seed + metric_index * 7919 + int(config_rows[0]["step"]))
            metric_summary[name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "scene_bootstrap_mean_ci95": [low, high],
            }
        depth_ratio = metric_summary["depth_recon_over_direct"]["median"]
        paired_ratio = metric_summary["paired_gs_over_depth"]["median"]
        recovery_score = (
            abs(math.log(depth_ratio))
            + abs(math.log(paired_ratio))
            + metric_summary["depth_recon_vs_direct_absrel"]["median"]
            + metric_summary["gs_axis_anisotropy_log_rms"]["median"]
        )
        configs.append({
            "config": config_key,
            "step": int(config_rows[0]["step"]),
            "frame_count": int(config_rows[0]["frame_count"]),
            "scene_count": len(config_rows),
            "recovery_score_lower_is_better": recovery_score,
            "metrics": metric_summary,
        })
    configs.sort(key=lambda row: (row["recovery_score_lower_is_better"], -row["metrics"]["render_gt_psnr_db"]["mean"]))
    for rank, config in enumerate(configs, start=1):
        config["recovery_rank"] = rank

    checkpoint_rows = []
    for step in sorted({int(row["step"]) for row in configs}):
        members = [row for row in configs if int(row["step"]) == step]
        checkpoint_rows.append({
            "step": step,
            "frame_counts": [int(row["frame_count"]) for row in members],
            "mean_recovery_score": float(np.mean([row["recovery_score_lower_is_better"] for row in members])),
            "mean_render_gt_psnr_db": float(np.mean([row["metrics"]["render_gt_psnr_db"]["mean"] for row in members])),
            "mean_render_gt_ssim": float(np.mean([row["metrics"]["render_gt_ssim"]["mean"] for row in members])),
            "mean_render_gt_lpips": float(np.mean([row["metrics"]["render_gt_lpips"]["mean"] for row in members])),
            "mean_abs_log_depth_ratio": float(np.mean([abs(math.log(row["metrics"]["depth_recon_over_direct"]["median"])) for row in members])),
            "mean_abs_log_paired_gs_depth": float(np.mean([abs(math.log(row["metrics"]["paired_gs_over_depth"]["median"])) for row in members])),
        })
    checkpoint_rows.sort(key=lambda row: (row["mean_recovery_score"], -row["mean_render_gt_psnr_db"]))
    for rank, row in enumerate(checkpoint_rows, start=1):
        row["checkpoint_recovery_rank"] = rank
    return {"configs": configs, "checkpoint_ranking": checkpoint_rows, "best_step": checkpoint_rows[0]["step"]}


def _write_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (output_dir / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["cases"]:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (output_dir / "checkpoint_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = payload["summary"]["checkpoint_ranking"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = payload["summary"]["checkpoint_ranking"][0]
    mode_note = "MOCK（仅验证代码，不是科学结果）" if payload["metadata"]["mock"] else "正式 PPU 评测"
    lines = [
        f"# Tokenizer v2 checkpoint sweep：{mode_note}",
        "",
        f"按 3D recovery score 的最佳 checkpoint：**step {best['step']}**。",
        "",
        "| rank | step | recovery score ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["checkpoint_ranking"]:
        lines.append(
            f"| {row['checkpoint_recovery_rank']} | {row['step']} | {row['mean_recovery_score']:.6f} | "
            f"{row['mean_render_gt_psnr_db']:.4f} | {row['mean_render_gt_ssim']:.5f} | {row['mean_render_gt_lpips']:.5f} |"
        )
    lines.extend([
        "",
        "Recovery score = |log(depth recon/direct)| + |log(paired GS/depth)| + depth AbsRel + GS axis anisotropy；越低越好。",
        "正式结论请同时查看 summary.json 中逐帧长、逐 scene bootstrap CI，不能用 mock 输出选模型。",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rank_shard_dir(output_dir: Path, run_id: str) -> Path:
    return output_dir / "rank_shards" / run_id


def _write_rank_rows(output_dir: Path, run_id: str, rank: int, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Persist one rank's rows without using NCCL object collectives.

    PyTorch distributed Python-object collectives serialize objects into
    tensors and, with NCCL, materialize the coalesced buffer on device.  On PPU
    this can request hundreds of GiB even for moderate Python payloads.  JSONL
    shards keep synchronization on the filesystem and avoid NCCL collectives.
    """
    shard_dir = _rank_shard_dir(output_dir, run_id)
    shard_dir.mkdir(parents=True, exist_ok=True)
    final = shard_dir / f"rank_{rank:02d}.jsonl"
    temporary = shard_dir / f".rank_{rank:02d}.jsonl.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(final)
    return final


def _read_rank_rows(output_dir: Path, run_id: str, world_size: int, *, timeout_sec: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shard_dir = _rank_shard_dir(output_dir, run_id)
    deadline = time.time() + timeout_sec
    last_report = 0.0
    while True:
        missing = [rank for rank in range(world_size) if not (shard_dir / f"rank_{rank:02d}.jsonl").is_file()]
        if not missing:
            break
        now = time.time()
        if now >= deadline:
            missing_paths = [str(shard_dir / f"rank_{rank:02d}.jsonl") for rank in missing]
            raise TimeoutError("timed out waiting for rank result shards: " + ", ".join(missing_paths))
        if now - last_report >= 60:
            print(f"[rank 0] waiting for result shards from ranks: {missing}", flush=True)
            last_report = now
        time.sleep(5.0)
    for rank in range(world_size):
        path = shard_dir / f"rank_{rank:02d}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dggt-checkpoint", type=Path, default=Path("/mnt/workspace/dggt/pretrained/model_latest_waymo.pt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/mnt/workspace/logs/tokenizer_t0_v2_stageA/ckpt"))
    parser.add_argument("--steps", default=",".join(str(value) for value in DEFAULT_STEPS))
    parser.add_argument("--frame-counts", default=",".join(str(value) for value in DEFAULT_FRAME_COUNTS))
    parser.add_argument("--processed-root", type=Path, default=Path("/mnt/workspace/datasets/waymo_processed_dggt"))
    parser.add_argument("--transfer-root", type=Path, default=Path("/mnt/workspace/datasets/waymo_processed_dggt"))
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/workspace/datasets/waymo"))
    parser.add_argument("--asset-root", type=Path, default=Path("/mnt/workspace/datasets/waymo_processed_dggt/objects_ply_transformed"))
    parser.add_argument("--dataset-split", choices=("training", "validation"), default="validation")
    parser.add_argument("--num-scenes", type=int, default=300)
    parser.add_argument("--sample-window", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--clean-train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--opacity-threshold", type=float, default=0.05)
    parser.add_argument("--lpips-chunk", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--save-visuals-per-config", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/workspace/dggt/runs/tokenizer_v2_ppu_eval"))
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("/mnt/workspace/dggt/runs/tokenizer_v2_fixed_selection_300.json"),
        help="Persistent scene/start/all-frame selection shared unchanged by every checkpoint sweep round.",
    )
    parser.add_argument("--expected-world-size", type=int, default=16)
    parser.add_argument("--expected-v2-commit", default=None, help="Deprecated; recorded only for compatibility, not enforced.")
    parser.add_argument("--allow-revision-mismatch", action="store_true", help="Deprecated compatibility flag; revision checks are disabled.")
    parser.add_argument("--no-checkpoint-sha256", action="store_true")
    parser.add_argument("--filesystem-sync-timeout-sec", type=float, default=86_400.0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-cases", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.steps_values = _parse_int_csv(args.steps, name="steps")
    args.frame_counts_values = _parse_int_csv(args.frame_counts, name="frame-counts")
    if args.num_scenes <= 1 or args.mock_cases <= 1 or args.bootstrap_samples < 0:
        raise ValueError("num-scenes/mock-cases must exceed one and bootstrap-samples must be nonnegative")
    rank, world_size, local_rank, device = _setup_worker_device(force_cpu=args.mock)
    if world_size != args.expected_world_size:
        raise RuntimeError(f"expected WORLD_SIZE={args.expected_world_size}, got {world_size}")
    run_id = os.environ.get("DGGT_EVAL_RUN_ID")
    if not run_id:
        run_id = f"single_{os.getpid()}" if world_size == 1 else "torchrun_unscoped"
    configs = build_configs(args.checkpoint_dir, args.steps_values, args.frame_counts_values, mock=args.mock)
    mine = assigned_configs(configs, rank, world_size)
    if rank == 0:
        print(f"[plan] {len(configs)} configs, {world_size} ranks, frames={args.frame_counts_values}, steps={args.steps_values}", flush=True)
    print(f"[rank {rank}/{world_size} local={local_rank} device={device}] configs={[item.key for item in mine]}", flush=True)

    # Commit ancestry checks are intentionally disabled.  The evaluation script
    # is often run after rebases, where the implementation can be equivalent
    # while the old commit is no longer an ancestor of HEAD.
    revision_ok = None

    checkpoint_hashes: dict[str, str | None] = {}
    if not args.mock:
        required = [args.dggt_checkpoint, *(config.checkpoint for config in configs if config.checkpoint is not None)]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing formal-eval inputs: " + ", ".join(missing))
        if rank == 0:
            checkpoint_hashes[str(args.dggt_checkpoint)] = None if args.no_checkpoint_sha256 else _sha256(args.dggt_checkpoint)
            for config in configs:
                assert config.checkpoint is not None
                checkpoint_hashes.setdefault(str(config.checkpoint), None if args.no_checkpoint_sha256 else _sha256(config.checkpoint))

    started = time.time()
    local_rows: list[dict[str, Any]] = []
    data_runtime = None if args.mock else _real_data_runtime(args, rank)
    # All formal ranks validate the shared selection manifest through
    # _real_data_runtime.  Ranks without assigned configs do not load VGGT/LPIPS.
    runtime = None
    if not args.mock and mine:
        assert data_runtime is not None
        runtime = _real_runtime(args, device, data_runtime)
    for config in mine:
        print(f"[rank {rank}] starting {config.key}", flush=True)
        if args.mock:
            local_rows.extend(_mock_case(config, case, args.seed) for case in range(args.mock_cases))
        else:
            assert runtime is not None
            local_rows.extend(_evaluate_real_config(config, runtime, args, device, rank))

    if world_size > 1:
        shard_path = _write_rank_rows(args.output_dir, run_id, rank, local_rows)
        print(f"[rank {rank}] wrote result shard: {shard_path}", flush=True)
        del local_rows, runtime, data_runtime
        if device.type == "cuda":
            torch.cuda.empty_cache()
        all_rows = (
            _read_rank_rows(
                args.output_dir,
                run_id,
                world_size,
                timeout_sec=float(args.filesystem_sync_timeout_sec),
            )
            if rank == 0
            else []
        )
    else:
        all_rows = local_rows
    if rank == 0:
        expected_cases = len(configs) * (args.mock_cases if args.mock else args.num_scenes)
        if len(all_rows) != expected_cases:
            raise RuntimeError(f"expected {expected_cases} cases, gathered {len(all_rows)}")
        summary = _summarize_rows(all_rows, args.bootstrap_samples, args.seed)
        payload = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "mock": bool(args.mock),
                "scientific_result": not bool(args.mock),
                "warning": "mock values are synthetic and must never select a production checkpoint" if args.mock else None,
                "git_head": _git(["rev-parse", "HEAD"]),
                "expected_v2_commit": args.expected_v2_commit,
                "expected_v2_commit_is_ancestor": revision_ok,
                "world_size": world_size,
                "run_id": run_id,
                "parallelism": "configuration-sharded: (checkpoint,frame_count) index modulo WORLD_SIZE",
                "steps": list(args.steps_values),
                "frame_counts": list(args.frame_counts_values),
                "cases_per_config": args.mock_cases if args.mock else args.num_scenes,
                "dataset_split": None if args.mock else args.dataset_split,
                "sample_policy": "one deterministic clip per unique scene; no repeated-measure inflation",
                "selection_manifest": None if args.mock else str(args.selection_manifest.resolve()),
                "selection_manifest_sha256": None if args.mock else _sha256(args.selection_manifest),
                "lidar_policy": "sample dense depth at original nonzero depth_flows_4 cell centers; direct-window frame-balanced scale shared by direct/recon",
                "precision": "synthetic_fp32" if args.mock else args.precision,
                "heads_precision": "synthetic_fp32" if args.mock else "fp32",
                "bootstrap_samples": args.bootstrap_samples,
                "elapsed_seconds": time.time() - started,
                "checkpoint_sha256": checkpoint_hashes,
            },
            "metric_contract": {
                "quality": "student Gaussian render vs raw RGB GT; direct DGGT render is separately reported as a ceiling diagnostic",
                "primary_3d": "frame-balanced static/non-sky same-pixel ratios; ideal depth_recon/direct=1 and paired_GS/depth=1",
                "v1_reference": {"depth_recon_over_direct": 1.0307, "paired_gs_over_depth": 0.7964},
                "selection": "lowest checkpoint-average recovery score across 10/12/14 frames; PSNR is tie-breaker",
            },
            "summary": summary,
            "cases": sorted(all_rows, key=lambda row: (row["step"], row["frame_count"], row["scene"])),
        }
        _write_artifacts(args.output_dir, payload)
        print(f"[done] best step={summary['best_step']}; artifacts={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
