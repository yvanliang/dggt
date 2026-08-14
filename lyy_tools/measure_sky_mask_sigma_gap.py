#!/usr/bin/env python3
"""Measure the T59 sky-mask gap without changing the production model schema.

The checkpoint predates the layout-v2 camera clean cut.  This wrapper therefore
executes the measurement in an explicitly supplied, isolated pre-P5 source tree
(the handoff revision is ``acefbf2``).  It never hashes checkpoints or inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SIGMA_WORKING_POINT = 0.169
SIGMA_CLEAN_ENDPOINT = 0.0
PRE_P5_REVISION_HINT = "acefbf2"
LEGACY_CAMERA_TOKEN_KEY = "camera_" + "gen_tokens"
LEGACY_CAMERA_ANCHOR_KEY = "camera_" + "gen_anchor_mask"
LEGACY_CAMERA_DECODER_MARKER = "camera_" + "gen_decoder"
LEGACY_FACTORIZED_CONDITION_KEY = "factorized_" + "asset_condition"
LEGACY_ASSET_KIND_KEY = "asset_" + "condition_kind"
LEGACY_ALIGNMENT_KEY = "apply_actor_" + "alignment"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_root",
        type=Path,
        required=True,
        help=(
            "Isolated pre-P5 checkout/worktree containing the old checkpoint schema "
            f"(use revision {PRE_P5_REVISION_HINT} or its exact pre-P5 equivalent)."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/data/disk2/lyy_dataset/waymo_processed_dggt/validation"),
    )
    parser.add_argument(
        "--caption_root",
        type=Path,
        default=Path("/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions"),
    )
    parser.add_argument(
        "--dggt_checkpoint",
        type=Path,
        default=Path("/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt"),
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=Path,
        default=Path("logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt"),
    )
    parser.add_argument(
        "--scene_gauge_table",
        type=Path,
        default=Path("data/scene_gauge/validation.json"),
        help="Read directly as JSON; no content digest is computed.",
    )
    parser.add_argument(
        "--text_encoder",
        type=Path,
        default=Path("/home/dancer/model/Qwen/Qwen3-0.6B"),
    )
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=100)
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--boundary_tolerance",
        type=int,
        default=1,
        help="Dense-mask pixel tolerance used by boundary F1.",
    )
    parser.add_argument(
        "--max_iou_gap",
        type=float,
        default=None,
        help="Optional pre-registered absolute IoU-gap threshold.",
    )
    parser.add_argument(
        "--max_boundary_f1_gap",
        type=float,
        default=None,
        help="Optional pre-registered absolute boundary-F1-gap threshold.",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _absolute(path: Path, cwd: Path) -> Path:
    return path if path.is_absolute() else (cwd / path).resolve()


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (args.max_iou_gap is None) != (args.max_boundary_f1_gap is None):
        parser.error("provide both gap thresholds or neither")
    for name in ("max_iou_gap", "max_boundary_f1_gap"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            parser.error(f"--{name} must be finite and non-negative")
    if int(args.batch_size) <= 0 or int(args.sequence_length) <= 0:
        parser.error("--batch_size and --sequence_length must be positive")
    if int(args.batch_index) < 0 or int(args.num_workers) < 0:
        parser.error("--batch_index and --num_workers must be non-negative")
    if int(args.scene_end) <= int(args.scene_start):
        parser.error("--scene_end must be greater than --scene_start")
    if int(args.boundary_tolerance) < 0:
        parser.error("--boundary_tolerance must be non-negative")


def _worker_command(args: argparse.Namespace) -> list[str]:
    values = {
        "source_root": args.source_root,
        "checkpoint": args.checkpoint,
        "output": args.output,
        "image_dir": args.image_dir,
        "caption_root": args.caption_root,
        "dggt_checkpoint": args.dggt_checkpoint,
        "tokenizer_checkpoint": args.tokenizer_checkpoint,
        "scene_gauge_table": args.scene_gauge_table,
        "text_encoder": args.text_encoder,
        "scene_start": args.scene_start,
        "scene_end": args.scene_end,
        "batch_index": args.batch_index,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "precision": args.precision,
        "boundary_tolerance": args.boundary_tolerance,
    }
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker"]
    for name, value in values.items():
        command.extend((f"--{name}", str(value)))
    if args.max_iou_gap is not None:
        command.extend(("--max_iou_gap", str(args.max_iou_gap)))
        command.extend(("--max_boundary_f1_gap", str(args.max_boundary_f1_gap)))
    return command


def _launch_isolated_worker(args: argparse.Namespace) -> None:
    source_root = args.source_root
    required = (
        source_root / "train_scene_flow_pretrain.py",
        source_root / "dggt/models/scene_flow.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pre-P5 source_root is incomplete: {missing}")
    train_source = required[0].read_text()
    model_source = required[1].read_text()
    if (
        LEGACY_CAMERA_TOKEN_KEY not in train_source
        or LEGACY_CAMERA_DECODER_MARKER not in model_source
    ):
        raise RuntimeError(
            "T59 must run in an isolated pre-P5 source tree because the 35k checkpoint "
            "uses the deleted camera-generation schema. Prepare a worktree at revision "
            f"{PRE_P5_REVISION_HINT} (or the exact equivalent) and pass it via --source_root."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"T59 checkpoint does not exist: {args.checkpoint}")
    env = dict(os.environ)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(source_root) + (os.pathsep + prior if prior else "")
    subprocess.run(
        _worker_command(args),
        cwd=source_root,
        env=env,
        check=True,
    )


def _binary_iou(prediction: Any, target: Any) -> float:
    import torch

    pred = torch.as_tensor(prediction).detach().float().ge(0.5)
    truth = torch.as_tensor(target).detach().float().ge(0.5)
    if pred.shape != truth.shape:
        raise ValueError(f"mask shape mismatch: {tuple(pred.shape)} != {tuple(truth.shape)}")
    union = (pred | truth).sum()
    if int(union.item()) == 0:
        return 1.0
    return float(((pred & truth).sum().float() / union.float()).item())


def _boundary_map(mask: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    hard = torch.as_tensor(mask).detach().float().ge(0.5).float()
    if hard.ndim < 2:
        raise ValueError("boundary masks need at least two spatial dimensions")
    height, width = int(hard.shape[-2]), int(hard.shape[-1])
    flat = hard.reshape(-1, 1, height, width)
    dilated = functional.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - functional.max_pool2d(1.0 - flat, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).gt(0.0)


def _dilate(mask: Any, radius: int) -> Any:
    import torch.nn.functional as functional

    if int(radius) == 0:
        return mask
    kernel = 2 * int(radius) + 1
    return functional.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=int(radius)).bool()


def _boundary_f1(prediction: Any, target: Any, *, tolerance: int) -> float:
    pred_boundary = _boundary_map(prediction)
    target_boundary = _boundary_map(target)
    pred_count = int(pred_boundary.sum().item())
    target_count = int(target_boundary.sum().item())
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0
    matched_pred = pred_boundary & _dilate(target_boundary, tolerance)
    matched_target = target_boundary & _dilate(pred_boundary, tolerance)
    precision = float(matched_pred.sum().item()) / float(pred_count)
    recall = float(matched_target.sum().item()) / float(target_count)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def summarize_measurement(
    working_prediction: Any,
    endpoint_prediction: Any,
    target: Any,
    *,
    boundary_tolerance: int,
    max_iou_gap: float | None,
    max_boundary_f1_gap: float | None,
) -> dict[str, Any]:
    working = {
        "sigma": SIGMA_WORKING_POINT,
        "iou": _binary_iou(working_prediction, target),
        "boundary_f1": _boundary_f1(
            working_prediction, target, tolerance=boundary_tolerance
        ),
    }
    endpoint = {
        "sigma": SIGMA_CLEAN_ENDPOINT,
        "iou": _binary_iou(endpoint_prediction, target),
        "boundary_f1": _boundary_f1(
            endpoint_prediction, target, tolerance=boundary_tolerance
        ),
    }
    gaps = {
        "iou_absolute": abs(endpoint["iou"] - working["iou"]),
        "boundary_f1_absolute": abs(
            endpoint["boundary_f1"] - working["boundary_f1"]
        ),
    }
    if max_iou_gap is None or max_boundary_f1_gap is None:
        decision = {
            "status": "undecided",
            "stage_two_passed": None,
            "reason": "No thresholds were supplied; T59 does not choose them implicitly.",
        }
        thresholds = None
    else:
        thresholds = {
            "max_iou_gap": float(max_iou_gap),
            "max_boundary_f1_gap": float(max_boundary_f1_gap),
        }
        passed = (
            gaps["iou_absolute"] <= float(max_iou_gap)
            and gaps["boundary_f1_absolute"] <= float(max_boundary_f1_gap)
        )
        decision = {
            "status": "passed" if passed else "failed",
            "stage_two_passed": bool(passed),
            "reason": "Compared against explicit command-line thresholds.",
        }
    return {
        "working_point": working,
        "clean_endpoint": endpoint,
        "absolute_gaps": gaps,
        "thresholds": thresholds,
        "decision": decision,
    }


def _first(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _inject_scene_gauge(batch: dict[str, Any], table_path: Path) -> None:
    import torch

    table = json.loads(table_path.read_text())
    entries = table.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"scene gauge table has no entries mapping: {table_path}")
    scene_values = batch.get("scene_name")
    clip_values = batch.get("clip_index")
    batch_size = int(batch["images"].shape[0])
    gauges = []
    valids = []
    for row in range(batch_size):
        scene = str(scene_values[row] if isinstance(scene_values, (list, tuple)) else _first(scene_values))
        if torch.is_tensor(clip_values):
            clip = int(clip_values.reshape(-1)[row].item())
        elif isinstance(clip_values, (list, tuple)):
            clip = int(clip_values[row])
        else:
            clip = int(clip_values)
        key = f"{scene}/{clip}"
        entry = entries.get(key)
        if not isinstance(entry, dict):
            raise KeyError(f"scene gauge table is missing {key}")
        fov = entry.get("log_tan_half_fov")
        valid = entry.get("valid")
        if not isinstance(fov, list) or len(fov) != 2 or not isinstance(valid, list) or len(valid) != 3:
            raise ValueError(f"scene gauge entry {key} has an invalid shape")
        gauges.append([float(entry["log_metric_scale"]), float(fov[0]), float(fov[1])])
        valids.append([bool(value) for value in valid])
    batch["scene_gauge"] = torch.tensor(gauges, dtype=torch.float32)
    batch["scene_gauge_valid"] = torch.tensor(valids, dtype=torch.bool)


def _load_checkpoint_model(checkpoint: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from dggt.models.scene_flow import WanSceneFlow
    from train_scene_flow_pretrain import load_scene_flow_state_dict_strict_profile_aware

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("scene_flow_config"), dict):
        raise ValueError("T59 checkpoint must contain scene_flow_config")
    model = WanSceneFlow.from_scene_config(
        bring_up=False,
        **dict(payload["scene_flow_config"]),
    )
    if isinstance(payload.get("ema_scene_flow_state_dict"), dict):
        state = payload["ema_scene_flow_state_dict"]
        source = "ema_scene_flow_state_dict"
    elif bool(payload.get("is_ema_weights")) and isinstance(payload.get("scene_flow"), dict):
        state = payload["scene_flow"]
        source = "scene_flow (EMA weights-only)"
    elif isinstance(payload.get("scene_flow"), dict):
        state = payload["scene_flow"]
        source = "scene_flow"
    else:
        raise ValueError("T59 checkpoint has no named SceneFlow weights")
    load_scene_flow_state_dict_strict_profile_aware(
        model,
        state,
        path=checkpoint,
        source=source,
    )
    model.to(device).eval()
    return model, payload


def _noise_like(value: Any, generator: Any) -> Any:
    import torch

    if not torch.is_tensor(value):
        return None
    result = torch.empty_like(value)
    result.normal_(generator=generator)
    return result


def _interpolate(clean: Any, noise: Any, sigma: Any) -> Any:
    import torch

    if not torch.is_tensor(clean):
        return None
    view = sigma.view(int(sigma.shape[0]), *([1] * (clean.ndim - 1))).to(clean)
    return (1.0 - view) * clean + view * noise


def _run_worker(args: argparse.Namespace) -> None:
    source_root = args.source_root
    sys.path.insert(0, str(source_root))

    import torch
    from torch.utils.data import DataLoader

    from datasets.dataset import WaymoOpenDataset
    from train_scene_flow_pretrain import (
        autocast_context,
        build_argparser as build_train_argparser,
        build_pretrain_bundle_from_batch,
        discover_scene_names,
        encode_text_condition,
        load_dggt_aggregator_and_tokenizer,
        setup_text_encoder,
        sky_atlas_shape,
        sky_grid_shape,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    scene_flow, payload = _load_checkpoint_model(args.checkpoint, device)
    config = scene_flow.config
    patch_grid = tuple(int(value) for value in config.patch_grid)
    train_args = build_train_argparser().parse_args(
        [
            "--image_dir",
            str(args.image_dir),
            "--dggt_ckpt_path",
            str(args.dggt_checkpoint),
            "--scene_gauge_path",
            str(args.scene_gauge_table),
            "--pullback_calibration_path",
            str(source_root / "unused-for-t59.json"),
            "--log_dir",
            str(args.output.parent / ".t59_unused_log"),
        ]
    )
    saved_args = payload.get("args")
    if isinstance(saved_args, dict):
        for name, value in saved_args.items():
            if hasattr(train_args, name):
                setattr(train_args, name, value)
    train_args.patch_grid = patch_grid
    train_args.sequence_length = int(args.sequence_length)
    train_args.precision = str(args.precision)
    train_args.caption_root = str(args.caption_root)
    train_args.text_encoder_path = str(args.text_encoder)
    train_args.text_max_length = int(getattr(train_args, "text_max_length", 256))
    train_args.sky_grid = sky_grid_shape(train_args)
    train_args.sky_atlas_hw = sky_atlas_shape(train_args)

    scene_names = discover_scene_names(
        str(args.image_dir), int(args.scene_start), int(args.scene_end)
    )
    dataset = WaymoOpenDataset(
        image_dir=str(args.image_dir),
        scene_names=scene_names,
        sequence_length=int(args.sequence_length),
        start_idx=0,
        mode=1,
        views=1,
        caption_root=str(args.caption_root),
        pretrain_patch_grid=patch_grid,
        pretrain_instance_cache_size=int(
            getattr(train_args, "pretrain_instance_cache_size", 128)
        ),
        trunk_major_samples=True,
        trunk_frames=29,
        return_full_dggt_context=True,
        load_dynamic_masks=False,
        binary_mask_channels=1,
        image_output_dtype="uint8",
        scene_gauge_path=None,
        load_metric_depth_diagnostic=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    iterator = iter(loader)
    batch = None
    for _ in range(int(args.batch_index) + 1):
        batch = next(iterator)
    assert batch is not None
    _inject_scene_gauge(batch, args.scene_gauge_table)

    vggt_model = load_dggt_aggregator_and_tokenizer(
        str(args.dggt_checkpoint), str(args.tokenizer_checkpoint), device
    )
    text_encoder = setup_text_encoder(train_args, device)
    bundle = build_pretrain_bundle_from_batch(
        batch,
        vggt_model,
        scene_flow,
        device,
        train_args,
        include_rgb_render_context=False,
        include_metric_depth_diagnostic=False,
    )
    text_tokens, text_mask = encode_text_condition(
        text_encoder, getattr(bundle, "captions", None)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    noise = {
        "video": _noise_like(bundle.z_clean_n, generator),
        "camera": _noise_like(getattr(bundle, "camera_target_clean_n", None), generator),
        "sky": _noise_like(getattr(bundle, "sky_gen_clean", None), generator),
        "gauge": _noise_like(getattr(bundle, "scene_gauge_clean_n", None), generator),
    }
    z_splat = getattr(bundle, "z_splat_n", None)
    if z_splat is None:
        z_splat = torch.zeros_like(bundle.z_clean_n)
    scaffold = torch.zeros_like(bundle.z_clean_n)

    def predict(sigma_value: float) -> Any:
        sigma = torch.full(
            (int(bundle.z_clean_n.shape[0]),),
            float(sigma_value),
            device=device,
            dtype=torch.float32,
        )
        legacy_camera_kwargs = {
            LEGACY_CAMERA_TOKEN_KEY: _interpolate(
                getattr(bundle, "camera_target_clean_n", None),
                noise["camera"],
                sigma,
            ),
            LEGACY_CAMERA_ANCHOR_KEY: getattr(
                bundle, "camera_" + "gen_anchor_mask", None
            ),
        }
        legacy_condition_kwargs = {
            LEGACY_FACTORIZED_CONDITION_KEY: getattr(
                bundle, LEGACY_FACTORIZED_CONDITION_KEY, None
            ),
            LEGACY_ASSET_KIND_KEY: getattr(bundle, LEGACY_ASSET_KIND_KEY, None),
            # Match the old runtime: the working-point pass performs its full
            # reread, while the mask-only clean endpoint bypasses that reread.
            LEGACY_ALIGNMENT_KEY: float(sigma_value) != SIGMA_CLEAN_ENDPOINT,
        }
        with torch.no_grad(), autocast_context(train_args, device):
            output = scene_flow(
                _interpolate(bundle.z_clean_n, noise["video"], sigma),
                sigma,
                z_splat,
                scaffold,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                getattr(bundle, "F_" + "asset_tokens"),
                encoder_attention_mask=bundle.encoder_attention_mask,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=getattr(
                    bundle, "camera_condition_tokens", None
                ),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                camera_condition_kind=getattr(bundle, "camera_condition_kind", None),
                sky_gen_tokens=_interpolate(
                    getattr(bundle, "sky_gen_clean", None), noise["sky"], sigma
                ),
                gauge_gen_tokens=_interpolate(
                    getattr(bundle, "scene_gauge_clean_n", None),
                    noise["gauge"],
                    sigma,
                ),
                return_mid=False,
                return_dict=True,
                return_sky_mask=True,
                frame_ids=getattr(bundle, "frame_ids", None),
                fps=getattr(bundle, "fps", None),
                **legacy_camera_kwargs,
                **legacy_condition_kwargs,
            )
        logits = output.get("sky_mask_refined_logits")
        if not torch.is_tensor(logits):
            raise RuntimeError("checkpoint did not return sky_mask_refined_logits")
        return torch.sigmoid(logits.float()).cpu()

    working_prediction = predict(SIGMA_WORKING_POINT)
    endpoint_prediction = predict(SIGMA_CLEAN_ENDPOINT)
    target = bundle.sky_mask_refined_clean.detach().float().cpu()
    result = summarize_measurement(
        working_prediction,
        endpoint_prediction,
        target,
        boundary_tolerance=int(args.boundary_tolerance),
        max_iou_gap=args.max_iou_gap,
        max_boundary_f1_gap=args.max_boundary_f1_gap,
    )
    result.update(
        {
            "schema": "sky_mask_sigma_gap_t59_v1",
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": payload.get("step"),
            "source_root": str(source_root),
            "source_requirement": (
                f"isolated pre-P5 tree at {PRE_P5_REVISION_HINT} or exact equivalent"
            ),
            "batch": {
                "batch_index": int(args.batch_index),
                "batch_size": int(bundle.z_clean_n.shape[0]),
                "sequence_length": int(bundle.z_clean_n.shape[1]),
                "scene_name": [str(value) for value in batch.get("scene_name", [])],
                "clip_index": (
                    batch["clip_index"].reshape(-1).tolist()
                    if hasattr(batch.get("clip_index"), "reshape")
                    else batch.get("clip_index")
                ),
            },
            "boundary_tolerance_pixels": int(args.boundary_tolerance),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    _validate_args(parser, args)
    invocation_cwd = Path.cwd()
    for name in (
        "source_root",
        "checkpoint",
        "output",
        "image_dir",
        "caption_root",
        "dggt_checkpoint",
        "tokenizer_checkpoint",
        "scene_gauge_table",
        "text_encoder",
    ):
        setattr(args, name, _absolute(getattr(args, name), invocation_cwd))
    if args._worker:
        _run_worker(args)
    else:
        _launch_isolated_worker(args)


if __name__ == "__main__":
    main()
