#!/usr/bin/env python3
"""Run T59 against the legacy v5 SceneFlow checkpoint on one visible GPU.

Copy this file into (or run it from) the legacy repository on the checkpoint
server.  The legacy repository supplies the model and dataset implementation;
this file supplies the complete measurement driver and metric implementation.
It does not initialize distributed training and does not calculate file
digests.

By default the script reads the validation/data dependency paths saved inside
the checkpoint and evaluates one deterministic trunk-0 sample from each of
100 distinct scenes. Explicit command-line paths always take precedence.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SIGMA_WORKING_POINT = 0.169
SIGMA_CLEAN_ENDPOINT = 0.0
DEFAULT_SOURCE_ROOT = Path("/mnt/workspace/dggt")
DEFAULT_CHECKPOINT = Path(
    "/mnt/workspace/logs/scene_flow_pretrain_v5/ckpt/pretrain_step035000.pt"
)
DEFAULT_OUTPUT = Path(
    "/mnt/workspace/logs/scene_flow_pretrain_v5/"
    "t59_step035000_100_scenes_single_gpu.json"
)

# Spell deleted legacy names in pieces so the layout-v2 repository's clean-cut
# grep remains meaningful while this copied script can address the old API.
LEGACY_CAMERA_TOKEN_KEY = "camera_" + "gen_tokens"
LEGACY_CAMERA_ANCHOR_KEY = "camera_" + "gen_anchor_mask"
LEGACY_CAMERA_DECODER_MARKER = "camera_" + "gen_decoder"
LEGACY_FACTORIZED_CONDITION_KEY = "factorized_" + "asset_condition"
LEGACY_ASSET_KIND_KEY = "asset_" + "condition_kind"
LEGACY_ALIGNMENT_KEY = "apply_actor_" + "alignment"


@dataclass(frozen=True)
class ResolvedInputs:
    image_dir: Path
    caption_root: Path | None
    dggt_checkpoint: Path
    tokenizer_checkpoint: Path
    scene_gauge_table: Path
    text_encoder: Path | None
    scene_start: int
    scene_end: int
    sequence_length: int
    precision: str


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSON. The default is to fail fast.",
    )

    parser.add_argument("--image_dir", type=Path, default=None)
    parser.add_argument("--caption_root", type=Path, default=None)
    parser.add_argument("--dggt_checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer_checkpoint", type=Path, default=None)
    parser.add_argument("--scene_gauge_table", type=Path, default=None)
    parser.add_argument("--text_encoder", type=Path, default=None)

    parser.add_argument("--scene_start", type=int, default=None)
    parser.add_argument("--scene_end", type=int, default=None)
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=100,
        help="Number of distinct validation scenes; one trunk-0 sample per scene.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the per-scene tqdm progress bar.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device_backend",
        choices=("ppu", "generic"),
        default="ppu",
        help=(
            "Tokenizer execution profile. The PPU profile chunks its large "
            "flattened MultiheadAttention batch; use generic only off PPU."
        ),
    )
    parser.add_argument(
        "--ppu_mha_batch_chunk_size",
        type=int,
        default=4096,
        help="Internal tokenizer MHA batch chunk used by the PPU profile.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help=(
            "Optional physical GPU list written to CUDA_VISIBLE_DEVICES before "
            "torch is imported. Omit when the scheduler already exposes one GPU."
        ),
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--expected_step", type=int, default=35000)
    parser.add_argument("--boundary_tolerance", type=int, default=1)
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
    return parser


def _validate_scalar_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if (args.max_iou_gap is None) != (args.max_boundary_f1_gap is None):
        parser.error("provide both gap thresholds or neither")
    for name in ("max_iou_gap", "max_boundary_f1_gap"):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(float(value)) or float(value) < 0.0
        ):
            parser.error(f"--{name} must be finite and non-negative")
    if int(args.num_scenes) <= 0:
        parser.error("--num_scenes must be positive")
    if int(args.batch_size) != 1:
        parser.error("the 100-scene T59 protocol requires --batch_size 1")
    if int(args.ppu_mha_batch_chunk_size) <= 0:
        parser.error("--ppu_mha_batch_chunk_size must be positive")
    if int(args.num_workers) < 0:
        parser.error("--num_workers must be non-negative")
    if args.sequence_length is not None and int(args.sequence_length) <= 0:
        parser.error("--sequence_length must be positive")
    if int(args.boundary_tolerance) < 0:
        parser.error("--boundary_tolerance must be non-negative")
    if int(args.expected_step) < -1:
        parser.error("--expected_step must be -1 or non-negative")


def _absolute(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _validate_legacy_source(source_root: Path) -> None:
    train_path = source_root / "train_scene_flow_pretrain.py"
    model_path = source_root / "dggt/models/scene_flow.py"
    missing = [str(path) for path in (train_path, model_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Legacy source_root is incomplete; missing " + ", ".join(missing)
        )
    train_source = train_path.read_text(encoding="utf-8", errors="replace")
    model_source = model_path.read_text(encoding="utf-8", errors="replace")
    if (
        LEGACY_CAMERA_TOKEN_KEY not in train_source
        or LEGACY_CAMERA_DECODER_MARKER not in model_source
    ):
        raise RuntimeError(
            "T59 must run from the old v5 source tree that produced the 35k "
            "checkpoint; the supplied source_root has the clean-cut model API."
        )


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch releases do not expose the weights_only keyword.
        return torch.load(path, map_location="cpu")


def _saved_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("args")
    if isinstance(raw, Mapping):
        return dict(raw)
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    return {}


def _saved_value(saved: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = saved.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _resolve_input_path(
    explicit: Path | None,
    saved: Mapping[str, Any],
    saved_names: tuple[str, ...],
    *,
    source_root: Path,
    option_name: str,
    kind: str,
    required: bool,
) -> Path | None:
    raw: Any = explicit
    if raw is None:
        raw = _saved_value(saved, saved_names)
    if raw is None:
        if required:
            raise ValueError(
                f"Cannot resolve {option_name} from the checkpoint; pass "
                f"--{option_name} explicitly."
            )
        return None
    path = _absolute(Path(str(raw)), source_root)
    valid = path.is_dir() if kind == "dir" else path.is_file()
    if not valid:
        expected = "directory" if kind == "dir" else "file"
        raise FileNotFoundError(
            f"Resolved --{option_name} is not a {expected}: {path}. "
            f"Pass the correct remote path explicitly."
        )
    return path


def resolve_inputs(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    source_root: Path,
) -> ResolvedInputs:
    saved = _saved_args(payload)
    no_text = bool(saved.get("no_text_condition", False))
    image_dir = _resolve_input_path(
        args.image_dir,
        saved,
        ("val_image_dir", "image_dir"),
        source_root=source_root,
        option_name="image_dir",
        kind="dir",
        required=True,
    )
    caption_root = _resolve_input_path(
        args.caption_root,
        saved,
        ("val_caption_root", "caption_root"),
        source_root=source_root,
        option_name="caption_root",
        kind="dir",
        required=not no_text,
    )
    dggt_checkpoint = _resolve_input_path(
        args.dggt_checkpoint,
        saved,
        ("dggt_ckpt_path",),
        source_root=source_root,
        option_name="dggt_checkpoint",
        kind="file",
        required=True,
    )
    tokenizer_checkpoint = _resolve_input_path(
        args.tokenizer_checkpoint,
        saved,
        ("tokenizer_ckpt_path",),
        source_root=source_root,
        option_name="tokenizer_checkpoint",
        kind="file",
        required=True,
    )
    scene_gauge_table = _resolve_input_path(
        args.scene_gauge_table,
        saved,
        ("val_scene_gauge_path", "scene_gauge_path"),
        source_root=source_root,
        option_name="scene_gauge_table",
        kind="file",
        required=True,
    )
    text_encoder = _resolve_input_path(
        args.text_encoder,
        saved,
        ("text_encoder_path",),
        source_root=source_root,
        option_name="text_encoder",
        kind="dir",
        required=not no_text,
    )
    assert image_dir is not None
    assert dggt_checkpoint is not None
    assert tokenizer_checkpoint is not None
    assert scene_gauge_table is not None

    scene_start_raw = (
        args.scene_start
        if args.scene_start is not None
        else saved.get("val_scene_start")
    )
    scene_start = 0 if scene_start_raw is None else int(scene_start_raw)
    scene_end_raw = (
        args.scene_end if args.scene_end is not None else saved.get("val_scene_end")
    )
    scene_end = int(scene_end_raw) if scene_end_raw is not None else scene_start + 100
    if scene_end <= scene_start:
        raise ValueError(
            "resolved validation scene range is empty; pass --scene_start and "
            "--scene_end explicitly"
        )
    saved_sequence = saved.get("sequence_length")
    if (
        args.sequence_length is not None
        and saved_sequence is not None
        and int(args.sequence_length) != int(saved_sequence)
    ):
        raise ValueError(
            "--sequence_length must match the checkpoint training value: "
            f"{args.sequence_length} != {saved_sequence}"
        )
    sequence_raw = (
        args.sequence_length
        if args.sequence_length is not None
        else saved.get("sequence_length", 10)
    )
    sequence_length = int(sequence_raw)
    if sequence_length <= 0:
        raise ValueError(f"resolved sequence_length must be positive, got {sequence_length}")
    precision = str(
        args.precision
        if args.precision is not None
        else saved.get("precision", "bf16")
    )
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"resolved precision must be fp32 or bf16, got {precision!r}")
    return ResolvedInputs(
        image_dir=image_dir,
        caption_root=caption_root,
        dggt_checkpoint=dggt_checkpoint,
        tokenizer_checkpoint=tokenizer_checkpoint,
        scene_gauge_table=scene_gauge_table,
        text_encoder=text_encoder,
        scene_start=scene_start,
        scene_end=scene_end,
        sequence_length=sequence_length,
        precision=precision,
    )


def _binary_iou(prediction: Any, target: Any) -> float:
    import torch

    pred = torch.as_tensor(prediction).detach().float().ge(0.5)
    truth = torch.as_tensor(target).detach().float().ge(0.5)
    if pred.shape != truth.shape:
        raise ValueError(
            f"mask shape mismatch: {tuple(pred.shape)} != {tuple(truth.shape)}"
        )
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
    eroded = 1.0 - functional.max_pool2d(
        1.0 - flat, kernel_size=3, stride=1, padding=1
    )
    return (dilated - eroded).gt(0.0)


def _dilate(mask: Any, radius: int) -> Any:
    import torch.nn.functional as functional

    if int(radius) == 0:
        return mask
    kernel = 2 * int(radius) + 1
    return functional.max_pool2d(
        mask.float(), kernel_size=kernel, stride=1, padding=int(radius)
    ).bool()


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


def _mask_metric_counts(
    prediction: Any,
    target: Any,
    *,
    tolerance: int,
) -> dict[str, int]:
    import torch

    pred = torch.as_tensor(prediction).detach().float().ge(0.5)
    truth = torch.as_tensor(target).detach().float().ge(0.5)
    if pred.shape != truth.shape:
        raise ValueError(
            f"mask shape mismatch: {tuple(pred.shape)} != {tuple(truth.shape)}"
        )
    pred_boundary = _boundary_map(pred)
    target_boundary = _boundary_map(truth)
    return {
        "intersection": int((pred & truth).sum().item()),
        "union": int((pred | truth).sum().item()),
        "pred_boundary_count": int(pred_boundary.sum().item()),
        "target_boundary_count": int(target_boundary.sum().item()),
        "matched_pred_count": int(
            (pred_boundary & _dilate(target_boundary, tolerance)).sum().item()
        ),
        "matched_target_count": int(
            (target_boundary & _dilate(pred_boundary, tolerance)).sum().item()
        ),
        "pred_positive_count": int(pred.sum().item()),
        "target_positive_count": int(truth.sum().item()),
        "pixel_count": int(pred.numel()),
    }


def _metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    union = int(counts["union"])
    intersection = int(counts["intersection"])
    iou = 1.0 if union == 0 else float(intersection) / float(union)
    pred_count = int(counts["pred_boundary_count"])
    target_count = int(counts["target_boundary_count"])
    if pred_count == 0 and target_count == 0:
        boundary_f1 = 1.0
    elif pred_count == 0 or target_count == 0:
        boundary_f1 = 0.0
    else:
        precision = float(counts["matched_pred_count"]) / float(pred_count)
        recall = float(counts["matched_target_count"]) / float(target_count)
        boundary_f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
    pixel_count = int(counts["pixel_count"])
    if pixel_count <= 0:
        raise ValueError("metric counts must contain at least one pixel")
    return {
        "iou": float(iou),
        "boundary_f1": float(boundary_f1),
        "pred_positive_fraction": (
            float(counts["pred_positive_count"]) / float(pixel_count)
        ),
        "target_positive_fraction": (
            float(counts["target_positive_count"]) / float(pixel_count)
        ),
    }


def _sum_metric_counts(rows: list[Mapping[str, int]]) -> dict[str, int]:
    if not rows:
        raise ValueError("cannot aggregate zero metric-count rows")
    keys = tuple(rows[0].keys())
    if any(tuple(row.keys()) != keys for row in rows):
        raise ValueError("metric-count rows do not share the same schema")
    return {key: sum(int(row[key]) for row in rows) for key in keys}


def _distribution(values: list[float]) -> dict[str, float | int]:
    import torch

    if not values:
        raise ValueError("cannot summarize an empty metric distribution")
    data = torch.tensor(values, dtype=torch.float64)
    count = int(data.numel())
    mean = float(data.mean().item())
    std_population = float(data.std(unbiased=False).item())
    std_sample = float(data.std(unbiased=True).item()) if count > 1 else 0.0
    standard_error = std_sample / math.sqrt(float(count))
    return {
        "count": count,
        "mean": mean,
        "std_population": std_population,
        "standard_error": standard_error,
        "normal_95pct_ci_low": mean - 1.96 * standard_error,
        "normal_95pct_ci_high": mean + 1.96 * standard_error,
        "min": float(data.min().item()),
        "q05": float(torch.quantile(data, 0.05).item()),
        "q25": float(torch.quantile(data, 0.25).item()),
        "q50": float(torch.quantile(data, 0.50).item()),
        "q75": float(torch.quantile(data, 0.75).item()),
        "q95": float(torch.quantile(data, 0.95).item()),
        "q99": float(torch.quantile(data, 0.99).item()),
        "max": float(data.max().item()),
    }


def aggregate_scene_measurements(
    scenes: list[Mapping[str, Any]],
    *,
    max_iou_gap: float | None,
    max_boundary_f1_gap: float | None,
) -> dict[str, Any]:
    if not scenes:
        raise ValueError("cannot aggregate zero scenes")
    working_counts = _sum_metric_counts(
        [scene["working_point"]["counts"] for scene in scenes]
    )
    endpoint_counts = _sum_metric_counts(
        [scene["clean_endpoint"]["counts"] for scene in scenes]
    )
    working_micro = _metrics_from_counts(working_counts)
    endpoint_micro = _metrics_from_counts(endpoint_counts)
    micro_gaps = {
        "iou_signed_endpoint_minus_working": (
            endpoint_micro["iou"] - working_micro["iou"]
        ),
        "iou_absolute": abs(endpoint_micro["iou"] - working_micro["iou"]),
        "boundary_f1_signed_endpoint_minus_working": (
            endpoint_micro["boundary_f1"] - working_micro["boundary_f1"]
        ),
        "boundary_f1_absolute": abs(
            endpoint_micro["boundary_f1"] - working_micro["boundary_f1"]
        ),
    }

    def values(*keys: str) -> list[float]:
        output: list[float] = []
        for scene in scenes:
            value: Any = scene
            for key in keys:
                value = value[key]
            output.append(float(value))
        return output

    macro = {
        "working_point": {
            "iou": _distribution(values("working_point", "iou")),
            "boundary_f1": _distribution(
                values("working_point", "boundary_f1")
            ),
        },
        "clean_endpoint": {
            "iou": _distribution(values("clean_endpoint", "iou")),
            "boundary_f1": _distribution(
                values("clean_endpoint", "boundary_f1")
            ),
        },
        "gaps": {
            key: _distribution(values("absolute_gaps", key))
            for key in (
                "iou_signed_endpoint_minus_working",
                "iou_absolute",
                "boundary_f1_signed_endpoint_minus_working",
                "boundary_f1_absolute",
            )
        },
    }
    if max_iou_gap is None or max_boundary_f1_gap is None:
        thresholds = None
        decision = {
            "status": "undecided",
            "stage_two_passed": None,
            "scope": "primary_micro",
            "reason": "No thresholds were supplied; T59 does not choose them implicitly.",
        }
    else:
        thresholds = {
            "max_iou_gap": float(max_iou_gap),
            "max_boundary_f1_gap": float(max_boundary_f1_gap),
        }
        passed = (
            micro_gaps["iou_absolute"] <= float(max_iou_gap)
            and micro_gaps["boundary_f1_absolute"]
            <= float(max_boundary_f1_gap)
        )
        decision = {
            "status": "passed" if passed else "failed",
            "stage_two_passed": bool(passed),
            "scope": "primary_micro",
            "reason": "Primary micro gaps were compared with explicit thresholds.",
        }
    return {
        "scene_count": len(scenes),
        "primary_micro": {
            "working_point": {**working_micro, "counts": working_counts},
            "clean_endpoint": {**endpoint_micro, "counts": endpoint_counts},
            "gaps": micro_gaps,
        },
        "macro_equal_scene_weight": macro,
        "empty_scene_diagnostics": {
            "working_empty_union": sum(
                int(scene["working_point"]["counts"]["union"] == 0)
                for scene in scenes
            ),
            "endpoint_empty_union": sum(
                int(scene["clean_endpoint"]["counts"]["union"] == 0)
                for scene in scenes
            ),
            "working_both_boundaries_empty": sum(
                int(
                    scene["working_point"]["counts"]["pred_boundary_count"]
                    == 0
                    and scene["working_point"]["counts"][
                        "target_boundary_count"
                    ]
                    == 0
                )
                for scene in scenes
            ),
            "endpoint_both_boundaries_empty": sum(
                int(
                    scene["clean_endpoint"]["counts"][
                        "pred_boundary_count"
                    ]
                    == 0
                    and scene["clean_endpoint"]["counts"][
                        "target_boundary_count"
                    ]
                    == 0
                )
                for scene in scenes
            ),
        },
        "thresholds": thresholds,
        "decision": decision,
    }


def summarize_measurement(
    working_prediction: Any,
    endpoint_prediction: Any,
    target: Any,
    *,
    boundary_tolerance: int,
    max_iou_gap: float | None,
    max_boundary_f1_gap: float | None,
) -> dict[str, Any]:
    import torch

    working_tensor = torch.as_tensor(working_prediction).detach().float()
    endpoint_tensor = torch.as_tensor(endpoint_prediction).detach().float()
    target_tensor = torch.as_tensor(target).detach().float()
    if (
        working_tensor.shape != endpoint_tensor.shape
        or working_tensor.shape != target_tensor.shape
    ):
        raise ValueError(
            "T59 tensors must have identical shapes: "
            f"working={tuple(working_tensor.shape)}, "
            f"endpoint={tuple(endpoint_tensor.shape)}, "
            f"target={tuple(target_tensor.shape)}"
        )
    for name, value in (
        ("working prediction", working_tensor),
        ("endpoint prediction", endpoint_tensor),
        ("target", target_tensor),
    ):
        if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be non-empty and finite")

    working_counts = _mask_metric_counts(
        working_tensor, target_tensor, tolerance=boundary_tolerance
    )
    endpoint_counts = _mask_metric_counts(
        endpoint_tensor, target_tensor, tolerance=boundary_tolerance
    )
    working_metrics = _metrics_from_counts(working_counts)
    endpoint_metrics = _metrics_from_counts(endpoint_counts)
    working = {
        "sigma": SIGMA_WORKING_POINT,
        "iou": working_metrics["iou"],
        "boundary_f1": working_metrics["boundary_f1"],
        "pred_positive_fraction": working_metrics["pred_positive_fraction"],
        "counts": working_counts,
    }
    endpoint = {
        "sigma": SIGMA_CLEAN_ENDPOINT,
        "iou": endpoint_metrics["iou"],
        "boundary_f1": endpoint_metrics["boundary_f1"],
        "pred_positive_fraction": endpoint_metrics["pred_positive_fraction"],
        "counts": endpoint_counts,
    }
    gaps = {
        "iou_signed_endpoint_minus_working": endpoint["iou"] - working["iou"],
        "iou_absolute": abs(endpoint["iou"] - working["iou"]),
        "boundary_f1_signed_endpoint_minus_working": (
            endpoint["boundary_f1"] - working["boundary_f1"]
        ),
        "boundary_f1_absolute": abs(
            endpoint["boundary_f1"] - working["boundary_f1"]
        ),
    }
    if max_iou_gap is None or max_boundary_f1_gap is None:
        thresholds = None
        decision = {
            "status": "undecided",
            "stage_two_passed": None,
            "reason": "No thresholds were supplied; T59 does not choose them implicitly.",
        }
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
        "tensor_shape": list(working_tensor.shape),
        "target_positive_fraction": working_metrics["target_positive_fraction"],
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


def _finite_gauge_value(raw: Any, valid: bool, *, key: str, channel: int) -> float:
    if raw is None:
        if valid:
            raise ValueError(
                f"scene gauge {key} channel {channel} is null but marked valid"
            )
        return 0.0
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"scene gauge {key} channel {channel} is non-finite")
    return value


def _load_scene_gauge_entries(table_path: Path) -> Mapping[str, Any]:
    table = json.loads(table_path.read_text(encoding="utf-8"))
    entries = table.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"scene gauge table has no entries mapping: {table_path}")
    return entries


def _inject_scene_gauge(
    batch: dict[str, Any],
    entries: Mapping[str, Any],
) -> None:
    import torch

    scene_values = batch.get("scene_name")
    clip_values = batch.get("clip_index")
    batch_size = int(batch["images"].shape[0])
    gauges: list[list[float]] = []
    valids: list[list[bool]] = []
    for row in range(batch_size):
        if isinstance(scene_values, (list, tuple)):
            scene = str(scene_values[row])
        else:
            scene = str(_first(scene_values))
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
        valid_raw = entry.get("valid")
        if (
            not isinstance(fov, list)
            or len(fov) != 2
            or not isinstance(valid_raw, list)
            or len(valid_raw) != 3
            or any(type(flag) is not bool for flag in valid_raw)
        ):
            raise ValueError(f"scene gauge entry {key} has an invalid shape")
        valid = [bool(flag) for flag in valid_raw]
        raw_values = (entry.get("log_metric_scale"), fov[0], fov[1])
        gauges.append(
            [
                _finite_gauge_value(value, valid[channel], key=key, channel=channel)
                for channel, value in enumerate(raw_values)
            ]
        )
        valids.append(valid)
    batch["scene_gauge"] = torch.tensor(gauges, dtype=torch.float32)
    batch["scene_gauge_valid"] = torch.tensor(valids, dtype=torch.bool)


def _load_checkpoint_model(
    payload: Mapping[str, Any], checkpoint: Path, device: Any
) -> tuple[Any, str]:
    from dggt.models.scene_flow import WanSceneFlow
    from train_scene_flow_pretrain import (
        load_scene_flow_state_dict_strict_profile_aware,
    )

    config = payload.get("scene_flow_config")
    if not isinstance(config, Mapping):
        raise ValueError("T59 checkpoint must contain scene_flow_config")
    model = WanSceneFlow.from_scene_config(bring_up=False, **dict(config))
    state = _require_ema_state(payload)
    source = "ema_scene_flow_state_dict"
    load_scene_flow_state_dict_strict_profile_aware(
        model,
        state,
        path=checkpoint,
        source=source,
    )
    model.to(device).eval()
    _validate_model_statistics(model)
    return model, source


def _require_ema_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("ema_scene_flow_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(
            "T59 requires the full 35k checkpoint with "
            "ema_scene_flow_state_dict; raw SceneFlow weights are not an "
            "equivalent measurement target."
        )
    return state


def _validate_model_statistics(model: Any) -> None:
    import torch

    def require_finite(name: str, *, positive: bool = False) -> None:
        value = getattr(model, name, None)
        if not torch.is_tensor(value) or value.numel() == 0:
            raise ValueError(f"EMA SceneFlow is missing statistics buffer {name}")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"EMA SceneFlow statistics buffer {name} is non-finite")
        if positive and not bool(value.gt(0).all().item()):
            raise ValueError(f"EMA SceneFlow statistics buffer {name} must be positive")

    require_finite("mu_z")
    require_finite("sigma_z", positive=True)
    require_finite("camera_anchor_mean")
    require_finite("camera_anchor_std", positive=True)
    require_finite("camera_delta_mean")
    require_finite("camera_delta_std", positive=True)
    require_finite("gauge_mean")
    require_finite("gauge_std", positive=True)
    for name in ("camera_stats_valid", "gauge_stats_valid"):
        valid = getattr(model, name, None)
        if not torch.is_tensor(valid) or valid.numel() != 1 or not bool(valid.item()):
            raise ValueError(f"EMA SceneFlow reports {name}=False")


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


def _make_train_args(
    payload: Mapping[str, Any],
    inputs: ResolvedInputs,
    source_root: Path,
    output: Path,
    patch_grid: tuple[int, int],
) -> Any:
    from train_scene_flow_pretrain import (
        build_argparser as build_train_argparser,
        sky_atlas_shape,
        sky_grid_shape,
    )

    parser_values = [
        "--image_dir",
        str(inputs.image_dir),
        "--dggt_ckpt_path",
        str(inputs.dggt_checkpoint),
        "--scene_gauge_path",
        str(inputs.scene_gauge_table),
        "--pullback_calibration_path",
        str(source_root / "unused-for-t59.json"),
        "--log_dir",
        str(output.parent / ".t59_unused_log"),
    ]
    train_args = build_train_argparser().parse_args(parser_values)
    for name, value in _saved_args(payload).items():
        if hasattr(train_args, name):
            setattr(train_args, name, value)
    train_args.image_dir = str(inputs.image_dir)
    train_args.val_image_dir = str(inputs.image_dir)
    train_args.dggt_ckpt_path = str(inputs.dggt_checkpoint)
    train_args.tokenizer_ckpt_path = str(inputs.tokenizer_checkpoint)
    train_args.scene_gauge_path = str(inputs.scene_gauge_table)
    train_args.val_scene_gauge_path = str(inputs.scene_gauge_table)
    train_args.caption_root = (
        None if inputs.caption_root is None else str(inputs.caption_root)
    )
    train_args.val_caption_root = train_args.caption_root
    train_args.text_encoder_path = (
        None if inputs.text_encoder is None else str(inputs.text_encoder)
    )
    train_args.patch_grid = patch_grid
    train_args.sequence_length = int(inputs.sequence_length)
    train_args.precision = str(inputs.precision)
    train_args.sky_grid = sky_grid_shape(train_args)
    train_args.sky_atlas_hw = sky_atlas_shape(train_args)
    return train_args


def select_distinct_trunk0_scenes(
    dataset: Any,
    *,
    num_scenes: int,
    candidate_scene_count: int,
) -> tuple[list[dict[str, Any]], int]:
    index = getattr(dataset, "trunk_major_index", None)
    scenes = getattr(dataset, "scenes", None)
    if not isinstance(index, list) or not isinstance(scenes, list):
        raise TypeError("legacy dataset does not expose trunk_major_index/scenes")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset_index, entry in enumerate(index):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(
                "T59 requires trunk_major_window_offsets=None and two-field "
                f"dataset indices, got {entry!r}"
            )
        scene_index, trunk_index = int(entry[0]), int(entry[1])
        if trunk_index != 0:
            continue
        scene_name = str(scenes[scene_index])
        if scene_name in seen:
            continue
        seen.add(scene_name)
        selected.append(
            {
                "ordinal": len(selected),
                "dataset_index": int(dataset_index),
                "scene_index": scene_index,
                "scene_name": scene_name,
                "trunk_index": 0,
            }
        )
    eligible_unique_scene_count = len(selected)
    if eligible_unique_scene_count < int(num_scenes):
        raise RuntimeError(
            "not enough distinct validation scenes with a complete trunk 0: "
            f"requested={num_scenes}, candidate_dirs={candidate_scene_count}, "
            f"eligible_unique={eligible_unique_scene_count}. Increase --scene_end or repair "
            "the validation scene pool; trunk 1 is never used as a substitute."
        )
    selected = selected[: int(num_scenes)]
    names = [row["scene_name"] for row in selected]
    if len(names) != len(set(names)):
        raise AssertionError("selected T59 scenes are not unique")
    return selected, eligible_unique_scene_count


def _batch_identity(batch: Mapping[str, Any]) -> dict[str, Any]:
    scene_name = str(_first(batch.get("scene_name")))
    clip_index = int(_first(batch.get("clip_index")))
    start_index = int(_first(batch.get("start_idx")))
    frame_ids_raw = batch.get("frame_ids")
    frame_ids = (
        frame_ids_raw.detach().cpu().reshape(-1).tolist()
        if hasattr(frame_ids_raw, "detach")
        else list(frame_ids_raw)
    )
    return {
        "scene_name": scene_name,
        "clip_index": clip_index,
        "start_index": start_index,
        "frame_ids": [int(value) for value in frame_ids],
    }


def run_measurement(
    args: argparse.Namespace,
    source_root: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    from datasets.dataset import WaymoOpenDataset
    from train_scene_flow_pretrain import (
        autocast_context,
        build_pretrain_bundle_from_batch,
        discover_scene_names,
        encode_text_condition,
        load_dggt_aggregator_and_tokenizer,
        setup_text_encoder,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("T59 single-GPU script must not run inside initialized DDP")
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise ValueError("T59 must run on one CUDA GPU; use --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current Python environment")
    torch.cuda.set_device(device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    run_started = time.perf_counter()

    payload = _torch_load(checkpoint)
    if not isinstance(payload, Mapping):
        raise ValueError("T59 checkpoint must be a mapping")
    checkpoint_step = payload.get("step")
    if int(args.expected_step) >= 0:
        if checkpoint_step is None or int(checkpoint_step) != int(args.expected_step):
            raise ValueError(
                f"checkpoint step {checkpoint_step!r} != expected {args.expected_step}"
            )
    inputs = resolve_inputs(args, payload, source_root)
    config = payload.get("scene_flow_config")
    if not isinstance(config, Mapping):
        raise ValueError("T59 checkpoint must contain scene_flow_config")
    patch_grid_raw = config.get("patch_grid")
    if not isinstance(patch_grid_raw, (list, tuple)) or len(patch_grid_raw) != 2:
        raise ValueError("scene_flow_config.patch_grid must contain two values")
    patch_grid = tuple(int(value) for value in patch_grid_raw)
    train_args = _make_train_args(
        payload, inputs, source_root, output, patch_grid
    )

    scene_names = discover_scene_names(
        str(inputs.image_dir), int(inputs.scene_start), int(inputs.scene_end)
    )
    dataset = WaymoOpenDataset(
        image_dir=str(inputs.image_dir),
        scene_names=scene_names,
        sequence_length=int(inputs.sequence_length),
        start_idx=0,
        mode=1,
        views=1,
        caption_root=(
            None if inputs.caption_root is None else str(inputs.caption_root)
        ),
        pretrain_patch_grid=patch_grid,
        pretrain_instance_cache_size=1,
        trunk_major_samples=True,
        trunk_frames=29,
        return_full_dggt_context=True,
        load_dynamic_masks=False,
        binary_mask_channels=1,
        image_output_dtype="uint8",
        # Inject the numeric table directly below. This avoids the old
        # dataset's checkpoint-binding/checksum path.
        scene_gauge_path=None,
        load_metric_depth_diagnostic=False,
    )
    selected, eligible_unique_scene_count = select_distinct_trunk0_scenes(
        dataset,
        num_scenes=int(args.num_scenes),
        candidate_scene_count=len(scene_names),
    )
    selected_dataset = Subset(
        dataset, [int(row["dataset_index"]) for row in selected]
    )
    loader = DataLoader(
        selected_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        drop_last=False,
    )
    # Start CPU workers before initializing the PPU context so forked workers
    # never inherit accelerator state. With num_workers=0 this is a no-op.
    loader_iterator = iter(loader)

    scene_flow, weight_source = _load_checkpoint_model(payload, checkpoint, device)
    model_patch_grid = tuple(int(value) for value in scene_flow.config.patch_grid)
    if model_patch_grid != patch_grid:
        raise ValueError(
            f"loaded model patch grid {model_patch_grid} != config {patch_grid}"
        )
    vggt_model = load_dggt_aggregator_and_tokenizer(
        str(inputs.dggt_checkpoint), str(inputs.tokenizer_checkpoint), device
    )
    text_encoder = setup_text_encoder(train_args, device)
    scene_gauge_entries = _load_scene_gauge_entries(inputs.scene_gauge_table)
    del payload
    gc.collect()
    scene_results: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(selected),
        desc="T59 sky-mask sigma gap",
        unit="scene",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=bool(args.no_progress),
    )
    try:
        for planned in selected:
            try:
                batch = next(loader_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "selected validation loader ended before all scenes"
                ) from error
            if not isinstance(batch, dict):
                raise TypeError(f"validation batch must be a dict, got {type(batch)}")
            identity = _batch_identity(batch)
            expected_scene = str(planned["scene_name"])
            if identity["scene_name"] != expected_scene:
                raise RuntimeError(
                    "selected scene order changed: "
                    f"expected={expected_scene}, actual={identity['scene_name']}"
                )
            expected_frames = list(range(int(inputs.sequence_length)))
            if (
                identity["clip_index"] != 0
                or identity["start_index"] != 0
                or identity["frame_ids"] != expected_frames
            ):
                raise RuntimeError(
                    "T59 requires the deterministic first window of trunk 0: "
                    f"identity={identity}, expected_frames={expected_frames}"
                )

            scene_started = time.perf_counter()
            _inject_scene_gauge(batch, scene_gauge_entries)
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
            scene_seed = int(args.seed) + int(planned["ordinal"])
            generator = torch.Generator(device=device)
            generator.manual_seed(scene_seed)
            noise = {
                "video": _noise_like(bundle.z_clean_n, generator),
                "camera": _noise_like(
                    getattr(bundle, "camera_target_clean_n", None), generator
                ),
                "sky": _noise_like(
                    getattr(bundle, "sky_gen_clean", None), generator
                ),
                "gauge": _noise_like(
                    getattr(bundle, "scene_gauge_clean_n", None), generator
                ),
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
                    LEGACY_ASSET_KIND_KEY: getattr(
                        bundle, LEGACY_ASSET_KIND_KEY, None
                    ),
                    LEGACY_ALIGNMENT_KEY: (
                        float(sigma_value) != SIGMA_CLEAN_ENDPOINT
                    ),
                }
                with torch.no_grad(), autocast_context(train_args, device):
                    model_output = scene_flow(
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
                        camera_attention_mask=getattr(
                            bundle, "camera_attention_mask", None
                        ),
                        camera_condition_kind=getattr(
                            bundle, "camera_condition_kind", None
                        ),
                        sky_gen_tokens=_interpolate(
                            getattr(bundle, "sky_gen_clean", None),
                            noise["sky"],
                            sigma,
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
                logits = model_output.get("sky_mask_refined_logits")
                if not torch.is_tensor(logits):
                    raise RuntimeError(
                        "checkpoint did not return sky_mask_refined_logits"
                    )
                return torch.sigmoid(logits.float()).cpu()

            working_prediction = predict(SIGMA_WORKING_POINT)
            endpoint_prediction = predict(SIGMA_CLEAN_ENDPOINT)
            target_raw = getattr(bundle, "sky_mask_refined_clean", None)
            if not torch.is_tensor(target_raw):
                raise RuntimeError("validation sample has no refined sky-mask GT")
            target = target_raw.detach().float().cpu()
            measurement = summarize_measurement(
                working_prediction,
                endpoint_prediction,
                target,
                boundary_tolerance=int(args.boundary_tolerance),
                max_iou_gap=None,
                max_boundary_f1_gap=None,
            )
            measurement.pop("thresholds")
            measurement.pop("decision")
            planned.update(identity)
            measurement.update(
                {
                    "ordinal": int(planned["ordinal"]),
                    "dataset_index": int(planned["dataset_index"]),
                    "scene_name": identity["scene_name"],
                    "trunk_index": 0,
                    "clip_index": identity["clip_index"],
                    "start_index": identity["start_index"],
                    "frame_ids": identity["frame_ids"],
                    "noise_seed": scene_seed,
                    "elapsed_seconds": time.perf_counter() - scene_started,
                }
            )
            scene_results.append(measurement)
            del (
                predict,
                batch,
                bundle,
                text_tokens,
                text_mask,
                generator,
                noise,
                z_splat,
                scaffold,
                working_prediction,
                endpoint_prediction,
                target,
                target_raw,
            )
            running = aggregate_scene_measurements(
                scene_results,
                max_iou_gap=None,
                max_boundary_f1_gap=None,
            )["primary_micro"]["gaps"]
            progress.set_postfix(
                scene=identity["scene_name"],
                iou_gap=f"{running['iou_absolute']:.5f}",
                boundary_gap=f"{running['boundary_f1_absolute']:.5f}",
                refresh=False,
            )
            progress.update(1)
    finally:
        progress.close()

    if len(scene_results) != int(args.num_scenes):
        raise RuntimeError(
            f"T59 completed {len(scene_results)} scenes, expected {args.num_scenes}"
        )
    observed_names = [str(scene["scene_name"]) for scene in scene_results]
    if len(observed_names) != len(set(observed_names)):
        raise RuntimeError("T59 produced duplicate scene measurements")
    aggregate = aggregate_scene_measurements(
        scene_results,
        max_iou_gap=args.max_iou_gap,
        max_boundary_f1_gap=args.max_boundary_f1_gap,
    )
    del (
        vggt_model,
        text_encoder,
        scene_gauge_entries,
        loader_iterator,
        loader,
        selected_dataset,
        dataset,
    )
    gc.collect()

    return {
        "schema": "sky_mask_sigma_gap_t59_100_scene_single_gpu_v2",
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "expected_checkpoint_step": int(args.expected_step),
        "weight_source": weight_source,
        "source_root": str(source_root),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "visible_cuda_device_count": int(torch.cuda.device_count()),
            "precision": inputs.precision,
            "base_seed": int(args.seed),
            "elapsed_seconds": time.perf_counter() - run_started,
            "distributed": False,
            "device_backend": os.environ.get("DGGT_DEVICE_BACKEND"),
            "ppu_mha_batch_chunk_size": os.environ.get(
                "DGGT_PPU_MHA_BATCH_CHUNK_SIZE"
            ),
        },
        "inputs": {
            "image_dir": str(inputs.image_dir),
            "caption_root": (
                None if inputs.caption_root is None else str(inputs.caption_root)
            ),
            "dggt_checkpoint": str(inputs.dggt_checkpoint),
            "tokenizer_checkpoint": str(inputs.tokenizer_checkpoint),
            "scene_gauge_table": str(inputs.scene_gauge_table),
            "text_encoder": (
                None if inputs.text_encoder is None else str(inputs.text_encoder)
            ),
        },
        "selection": {
            "policy": "first_complete_trunk0_per_unique_scene",
            "candidate_scene_start": int(inputs.scene_start),
            "candidate_scene_end": int(inputs.scene_end),
            "candidate_scene_count": len(scene_names),
            "eligible_unique_scene_count": eligible_unique_scene_count,
            "requested_scene_count": int(args.num_scenes),
            "completed_scene_count": len(scene_results),
            "unique_scene_count": len(set(observed_names)),
            "batch_size": 1,
            "trunk_frames": 29,
            "trunk_index": 0,
            "window_offset": 0,
            "sequence_length": int(inputs.sequence_length),
            "selected_samples": selected,
        },
        "boundary_tolerance_pixels": int(args.boundary_tolerance),
        "aggregate": aggregate,
        "thresholds": aggregate["thresholds"],
        "decision": aggregate["decision"],
        "scenes": scene_results,
    }


def _write_json_atomic(output: Path, result: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    _validate_scalar_args(parser, args)
    invocation_cwd = Path.cwd()
    source_root = _absolute(args.source_root, invocation_cwd)
    checkpoint = _absolute(args.checkpoint, invocation_cwd)
    output = _absolute(args.output, invocation_cwd)
    _validate_legacy_source(source_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"T59 checkpoint does not exist: {checkpoint}")
    if output.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"T59 output already exists: {output}; pass --overwrite to replace it"
        )
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["DGGT_DEVICE_BACKEND"] = str(args.device_backend)
    os.environ["DGGT_PPU_MHA_BATCH_CHUNK_SIZE"] = str(
        int(args.ppu_mha_batch_chunk_size)
    )
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)

    print(f"[T59] source_root={source_root}", flush=True)
    print(f"[T59] checkpoint={checkpoint}", flush=True)
    print(f"[T59] output={output}", flush=True)
    print(
        "[T59] tokenizer_backend="
        f"{os.environ['DGGT_DEVICE_BACKEND']} "
        "mha_batch_chunk_size="
        f"{os.environ['DGGT_PPU_MHA_BATCH_CHUNK_SIZE']}",
        flush=True,
    )
    result = run_measurement(args, source_root, checkpoint, output)
    _write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_scene_count": result["selection"][
                    "completed_scene_count"
                ],
                "primary_micro": result["aggregate"]["primary_micro"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"[T59] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
