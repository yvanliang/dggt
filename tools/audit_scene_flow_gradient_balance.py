#!/usr/bin/env python3
"""Audit early-checkpoint gradient balance on one real Waymo training clip.

This tool is intentionally checkpoint-based: initialization gradients are not
representative of the shared representation after the heads have started to
specialize. Use a full (non-EMA-only) step-2500 checkpoint so the raw training
weights and saved launch arguments are available.

Example:

    CUDA_VISIBLE_DEVICES=0 conda run -n dggt --no-capture-output \
      python tools/audit_scene_flow_gradient_balance.py \
      --checkpoint /path/to/pretrain_step002500.pt \
      --dataset_index 0 \
      --output runs/audits/step2500_gradient_balance.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset import WaymoOpenDataset
from dggt.utils.scene_gauge import load_pullback_calibration
from inference_scene_flow_pretrain import build_scene_flow_from_checkpoint
from train_scene_flow_pretrain import (
    DEFAULT_SKY_ATLAS_HW,
    build_argparser as build_train_argparser,
    discover_scene_names,
    load_dggt_aggregator_and_tokenizer,
    seed_everything,
    setup_text_encoder,
    train_step,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Full raw step checkpoint (2500 recommended).")
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="runs/audits/scene_flow_gradient_balance.json")
    parser.add_argument(
        "--probe_blocks",
        default="0,mid,last",
        help="Comma-separated shared transformer block indices; supports mid and last.",
    )
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--caption_root", default=None)
    parser.add_argument("--scene_gauge_path", default=None)
    parser.add_argument("--dggt_ckpt_path", default=None)
    parser.add_argument("--tokenizer_ckpt_path", default=None)
    parser.add_argument("--pullback_calibration_path", default=None)
    parser.add_argument(
        "--no_text_condition",
        action="store_true",
        help="Skip the text encoder for a lighter audit; leave unset for exact saved conditions.",
    )
    return parser.parse_args()


def _training_args(payload: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    saved = payload.get("args")
    if not isinstance(saved, dict):
        raise ValueError(
            "Gradient audit requires a full checkpoint containing saved `args`; "
            "EMA/weights-only checkpoints are insufficient."
        )
    required_overrides = {
        "image_dir": cli.image_dir,
        "caption_root": cli.caption_root,
        "scene_gauge_path": cli.scene_gauge_path,
        "dggt_ckpt_path": cli.dggt_ckpt_path,
        "tokenizer_ckpt_path": cli.tokenizer_ckpt_path,
        "pullback_calibration_path": cli.pullback_calibration_path,
    }
    resolved = {
        name: override if override is not None else saved.get(name)
        for name, override in required_overrides.items()
    }
    missing = [name for name, value in resolved.items() if not value]
    if missing:
        raise ValueError(
            f"checkpoint args do not provide {missing}; pass the corresponding CLI overrides"
        )
    defaults = build_train_argparser().parse_args(
        [
            "--image_dir", str(resolved["image_dir"]),
            "--dggt_ckpt_path", str(resolved["dggt_ckpt_path"]),
            "--tokenizer_ckpt_path", str(resolved["tokenizer_ckpt_path"]),
            "--scene_gauge_path", str(resolved["scene_gauge_path"]),
            "--pullback_calibration_path", str(resolved["pullback_calibration_path"]),
            "--log_dir", "runs/audits/_unused",
        ]
    )
    for name, value in saved.items():
        setattr(defaults, name, value)
    for name, value in resolved.items():
        setattr(defaults, name, value)
    defaults.caption_root = str(resolved["caption_root"])
    defaults.no_text_condition = bool(cli.no_text_condition)
    defaults.camera_anchor_context_dropout = 0.0
    # Isolate the ordinary one-forward objectives. Rendering and endpoint
    # diagnostics add another forward/decoder and are not part of issue 3.
    defaults.lambda_rgb_render = 0.0
    defaults.rgb_render_every = 0
    defaults.metric_depth_diagnostic_every = 0
    defaults.lambda_level_consistency = 0.0
    defaults.lambda_head_consistency = 0.0
    defaults.patch_grid = (
        int(getattr(defaults, "patch_grid_h", 25)),
        int(getattr(defaults, "patch_grid_w", 37)),
    )
    defaults.sky_grid = (
        int(getattr(defaults, "sky_grid_h", 16)),
        int(getattr(defaults, "sky_grid_w", 32)),
    )
    defaults.sky_atlas_hw = DEFAULT_SKY_ATLAS_HW
    return defaults


def _resolve_probe_indices(spec: str, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("model has no shared transformer blocks")
    values: list[int] = []
    for token in str(spec).split(","):
        token = token.strip().lower()
        if not token:
            continue
        index = count // 2 if token == "mid" else count - 1 if token == "last" else int(token)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise ValueError(f"probe block {token!r} resolves outside [0,{count})")
        if index not in values:
            values.append(index)
    if not values:
        raise ValueError("--probe_blocks selected no blocks")
    return values


def _gradient_vector_stats(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[list[torch.Tensor | None], float]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared_norm = sum(
        float(gradient.detach().float().square().sum().item())
        for gradient in gradients
        if gradient is not None
    )
    return list(gradients), math.sqrt(squared_norm)


def _gradient_cosine(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    dot = 0.0
    for left_grad, right_grad in zip(left, right, strict=True):
        if left_grad is not None and right_grad is not None:
            dot += float(
                (left_grad.detach().float() * right_grad.detach().float()).sum().item()
            )
    return dot / (left_norm * right_norm)


def main() -> None:
    cli = _parse_args()
    checkpoint_path = Path(cli.checkpoint)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict) or "scene_flow" not in payload or bool(payload.get("is_ema_weights")):
        raise ValueError("--checkpoint must be a full raw training checkpoint")
    args = _training_args(payload, cli)
    checkpoint_step = int(payload.get("step", 0))
    # The strict checkpoint builder reloads the payload to construct the saved
    # architecture. Release this first copy (including optimizer state) before
    # that second load; full checkpoints are tens of GB.
    del payload
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    seed_everything(int(cli.seed))

    scene_flow, checkpoint_info = build_scene_flow_from_checkpoint(
        checkpoint_path,
        device=device,
        no_ema=True,
        fallback_args=args,
    )
    config_grid = tuple(int(value) for value in scene_flow.config.patch_grid)
    args.patch_grid = config_grid
    args.patch_grid_h, args.patch_grid_w = config_grid
    args.latent_dim = int(scene_flow.config.out_channels)
    args.prediction_type = str(scene_flow.config.prediction_type)
    args.sky_grid = tuple(int(value) for value in scene_flow.config.sky_grid)
    args.sky_grid_h, args.sky_grid_w = args.sky_grid
    args.sky_atlas_hw = tuple(int(value) for value in scene_flow.config.sky_atlas_hw)
    for parameter in scene_flow.parameters():
        parameter.requires_grad_(True)
    scene_flow.train()
    scene_flow.require_camera_stats()
    scene_flow.require_gauge_stats()
    pullback = load_pullback_calibration(
        args.pullback_calibration_path,
        tokenizer_checkpoint_path=args.tokenizer_ckpt_path,
        dggt_checkpoint_path=args.dggt_ckpt_path,
        expected_window_len=int(args.sequence_length),
        expected_patch_grid=args.patch_grid,
        expected_artifact_sha256=checkpoint_info["metric_gauge_provenance"][
            "pullback_artifact_sha256"
        ],
    )
    scene_flow._pullback_calibration = pullback

    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path,
        args.tokenizer_ckpt_path,
        device,
    )
    text_encoder = setup_text_encoder(args, device)
    scene_names = discover_scene_names(args.image_dir, args.scene_start, args.scene_end)
    dataset = WaymoOpenDataset(
        image_dir=args.image_dir,
        scene_names=scene_names,
        sequence_length=int(args.sequence_length),
        mode=1,
        views=1,
        caption_root=None if args.no_text_condition else args.caption_root,
        pretrain_patch_grid=args.patch_grid,
        pretrain_instance_cache_size=int(args.pretrain_instance_cache_size),
        trunk_frames=29,
        camera_anchor_window_probability=float(args.camera_anchor_window_probability),
        return_full_dggt_context=True,
        load_dynamic_masks=False,
        binary_mask_channels=1,
        image_output_dtype="uint8",
        scene_gauge_path=args.scene_gauge_path,
        expected_scene_gauge_dggt_sha256=pullback.dggt_sha256,
        expected_scene_gauge_split=Path(args.image_dir).name,
        load_metric_depth_diagnostic=False,
    )
    if not 0 <= int(cli.dataset_index) < len(dataset):
        raise IndexError(f"dataset_index={cli.dataset_index} outside [0,{len(dataset)})")
    batch = next(iter(DataLoader(Subset(dataset, [int(cli.dataset_index)]), batch_size=1)))

    loss_terms: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device=device).manual_seed(int(cli.seed))
    total_loss, logs = train_step(
        batch,
        vggt_model,
        scene_flow,
        None,
        device,
        args,
        text_encoder,
        global_step=checkpoint_step,
        generator=generator,
        loss_terms_out=loss_terms,
    )
    required_terms = {"video_core", "camera_flow", "camera_pose", "gauge_flow", "gauge_direct"}
    missing_terms = sorted(required_terms - set(loss_terms))
    if missing_terms:
        raise RuntimeError(
            f"audit batch did not produce {missing_terms}; choose an anchor-containing row and check loss weights"
        )
    combined_terms = {
        "video_core": loss_terms["video_core"],
        "camera_aux": loss_terms["camera_flow"] + loss_terms["camera_pose"],
        "gauge_aux": loss_terms["gauge_flow"] + loss_terms["gauge_direct"],
    }
    blocks = scene_flow.blocks
    probe_indices = _resolve_probe_indices(cli.probe_blocks, len(blocks))
    probe_parameters: list[torch.nn.Parameter] = []
    probe_parameter_names: list[str] = []
    prefixes = tuple(f"blocks.{index}." for index in probe_indices)
    for name, parameter in scene_flow.named_parameters():
        if name.startswith(prefixes) and parameter.requires_grad:
            probe_parameter_names.append(name)
            probe_parameters.append(parameter)
    if not probe_parameters:
        raise RuntimeError(f"no parameters matched shared block prefixes {prefixes}")

    gradients: dict[str, list[torch.Tensor | None]] = {}
    norms: dict[str, float] = {}
    names = list(combined_terms)
    for index, name in enumerate(names):
        gradients[name], norms[name] = _gradient_vector_stats(
            combined_terms[name],
            probe_parameters,
            retain_graph=index < len(names) - 1,
        )
    cosines = {
        "camera_aux_vs_video_core": _gradient_cosine(
            gradients["camera_aux"], gradients["video_core"],
            norms["camera_aux"], norms["video_core"],
        ),
        "gauge_aux_vs_video_core": _gradient_cosine(
            gradients["gauge_aux"], gradients["video_core"],
            norms["gauge_aux"], norms["video_core"],
        ),
        "camera_aux_vs_gauge_aux": _gradient_cosine(
            gradients["camera_aux"], gradients["gauge_aux"],
            norms["camera_aux"], norms["gauge_aux"],
        ),
    }
    video_norm = max(norms["video_core"], 1.0e-30)
    result = {
        "schema": "scene_flow_early_gradient_balance_v1",
        "checkpoint": checkpoint_info,
        "checkpoint_step": checkpoint_step,
        "dataset_index": int(cli.dataset_index),
        "seed": int(cli.seed),
        "probe_block_indices": probe_indices,
        "probe_parameter_count": len(probe_parameters),
        "probe_parameter_numel": sum(parameter.numel() for parameter in probe_parameters),
        "loss_values": {
            name: float(value.detach().item()) for name, value in combined_terms.items()
        },
        "gradient_norms": norms,
        "gradient_norm_ratios_to_video": {
            "camera_aux": norms["camera_aux"] / video_norm,
            "gauge_aux": norms["gauge_aux"] / video_norm,
        },
        "gradient_cosines": cosines,
        "camera_pose_ramp": logs.get("camera_pose_ramp"),
        "camera_pose_effective_weight": logs.get("camera_pose_effective_weight"),
        "total_loss": float(total_loss.detach().item()),
        "decision_hint": {
            "concerning_aux_ratio": ">0.5 on multiple clips",
            "concerning_conflict": "cosine<-0.1 on multiple clips",
            "note": "Use at least 4 dataset indices; a single clip is diagnostic, not a population estimate.",
        },
    }
    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
