#!/usr/bin/env python3
"""Independent SceneFlow tokenizer round-trip gauge test for Gaussian scales.

This file intentionally does not import or invoke any existing test/verification
tool.  It independently loads the production Aggregator, JointSceneTokenizer,
DepthHead, and GaussianHead; independently preprocesses Waymo RGB/masks/LiDAR;
and measures paired, same-pixel scale changes caused by a tokenizer round trip.

The primary quantity is evaluated per pixel before any aggregation:

    e = mean_axis(log(scale_recon / scale_direct))
        - log(depth_recon / depth_direct)

For each frame we take median(e), then take the median over frames.  exp(e) is
one under a uniform similarity change shared by depth and all three Gaussian
scale axes.  Five overlapping ten-frame windows are descriptive repeated
measurements, not independent samples; final inference is balanced by 29-frame
trunk/case.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_NAME = "scene_flow_gaussian_gauge_retest"
SCHEMA_VERSION = "2.4.0"
TOKENIZER_LEVELS = (4, 11, 17, 23)
WINDOW_STARTS = (0, 5, 10, 14, 19)
WINDOW_LENGTH = 10
TRUNK_LENGTH = 29
PATCH_SIZE = 14
TARGET_WIDTH = 518
SCALE_FLOOR = 1.0e-5
STRICT_SCALE_THRESHOLD = 1.01e-5
OPACITY_THRESHOLDS = (0.01, 0.05, 0.5)
DEPTH_BIN_EDGES_M = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0)
DEPTH_PROFILE_REFERENCE_M = 20.0
DEPTH_PROFILE_VARIABLE_NAME = "uncorrected_reconstructed_metric_depth_m"
# This split is deliberately literal rather than sampled at run time.  The
# first twenty scenes are the only observations allowed to fit the depth
# profile; the final ten are reserved for downstream metric-boundary selection.
D4_CALIBRATION_SCENES = tuple(range(300, 320))
D4_HOLDOUT_SCENES = tuple(range(320, 330))
D4_FORM_BOOTSTRAP_SEED = 20260731
D4_FORM_MIN_ABS_SPEARMAN = 0.8
D4_FORM_MIN_CV_IMPROVEMENT_FRACTION = 0.02
D4_IDENTITY_MIN_CV_IMPROVEMENT_FRACTION = 0.02
D4_PRIMARY_SUPPORT = "primary_static_nonsky_opacity_0p05"
PAIRED_EQUIVALENCE_MARGIN = (0.95, 1.05)
PAIRED_EQUIVALENCE_BOOTSTRAP_SEED = 20260805
PAIRED_EQUIVALENCE_BOOTSTRAP_SAMPLES = 10000
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data/disk2/lyy_dataset/waymo_processed_dggt/training")
DEFAULT_DGGT_CHECKPOINT = Path("/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt")
DEFAULT_TOKENIZER_CHECKPOINT = (
    REPO_ROOT / "logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt"
)
DEFAULT_REFERENCE_JSON = (
    REPO_ROOT
    / "runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs/metric_gauge_retest/"
    / "v2_gaussian_gauge_300_329_trunks012_<tokenizer_sha8>.json"
)
V1_HISTORICAL_OUTPUT = (
    REPO_ROOT / "runs/metric_gauge_retest/gaussian_gauge_300_329_trunks012.json"
)


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _safe_exp(value: float | None) -> float | None:
    if value is None:
        return None
    result = math.exp(float(value))
    if not math.isfinite(result):
        raise ValueError(f"exp({value}) is not finite")
    return result


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_output_path(path: Path, *, tokenizer_sha256: str) -> Path:
    if (
        len(tokenizer_sha256) != 64
        or tokenizer_sha256 != tokenizer_sha256.lower()
        or any(character not in "0123456789abcdef" for character in tokenizer_sha256)
    ):
        raise ValueError("tokenizer_sha256 must be a full lowercase SHA-256")
    resolved = Path(
        str(path).replace("<tokenizer_sha8>", tokenizer_sha256[:8])
    ).expanduser().resolve()
    if resolved == V1_HISTORICAL_OUTPUT.resolve():
        raise ValueError("refusing to overwrite immutable v1 Gaussian audit history")
    if tokenizer_sha256[:8] not in resolved.name:
        raise ValueError("--output filename must contain the tokenizer SHA-256 prefix")
    return resolved


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _depth_profile_variable_contract() -> dict[str, Any]:
    """Return the single independent-variable contract used by fit and runtime."""

    return {
        "name": DEPTH_PROFILE_VARIABLE_NAME,
        "source_tensor": "reconstructed_depth",
        "metric_conversion": "divide_by_full_29f_direct_units_per_metre",
        "correction_state": "uncorrected",
        "runtime_clamp_m": [0.5, 80.0],
        "reference_depth_m": DEPTH_PROFILE_REFERENCE_M,
    }


def _stat_manifest(paths: Sequence[str], *, root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for raw in sorted(set(paths)):
        path = Path(raw).resolve()
        stat = path.stat()
        try:
            name = str(path.relative_to(root.resolve()))
        except ValueError:
            name = str(path)
        row = {"path": name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {
        "rule": "sha256(canonical JSON lines of path,size,mtime_ns); not a content hash",
        "file_count": len(rows),
        "total_bytes": int(sum(row["size"] for row in rows)),
        "sha256": digest.hexdigest(),
    }


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_integer_specs(values: Sequence[str], *, name: str) -> list[int]:
    parsed: list[int] = []
    for raw in values:
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item[1:]:
                left, right = item.split("-", 1)
                start, end = int(left), int(right)
                if end < start:
                    raise ValueError(f"Descending {name} range is not allowed: {item}")
                parsed.extend(range(start, end + 1))
            else:
                parsed.append(int(item))
    unique = sorted(set(parsed))
    if not unique:
        raise ValueError(f"{name} must not be empty")
    return unique


def _torch_load_weights(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def _state_mapping(payload: Any, *, source: Path) -> Mapping[str, Any]:
    state = payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"Unsupported checkpoint structure: {source}")
    return state


def _clean_tensor_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            continue
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in cleaned:
            raise RuntimeError(f"Duplicate checkpoint key after module-prefix removal: {key}")
        cleaned[key] = value
    return cleaned


def _strict_load_prefixed(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    prefix: str,
    source: Path,
) -> dict[str, int]:
    subset = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    expected = module.state_dict()
    missing = sorted(set(expected) - set(subset))
    unexpected = sorted(set(subset) - set(expected))
    shape_mismatches = sorted(
        key
        for key in set(expected).intersection(subset)
        if tuple(expected[key].shape) != tuple(subset[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Strict load failed for {prefix[:-1]} from {source}: "
            f"missing={missing[:8]} ({len(missing)}), "
            f"unexpected={unexpected[:8]} ({len(unexpected)}), "
            f"shape_mismatches={shape_mismatches[:8]} ({len(shape_mismatches)})"
        )
    module.load_state_dict(subset, strict=True)
    return {
        "state_key_count": len(subset),
        "parameter_count": sum(int(parameter.numel()) for parameter in module.parameters()),
        "buffer_count": sum(int(buffer.numel()) for buffer in module.buffers()),
    }


def _strict_load_tokenizer(
    tokenizer: torch.nn.Module,
    payload: Any,
    *,
    source: Path,
) -> dict[str, int]:
    tokenizer_only = False
    state: Any = payload
    if isinstance(payload, Mapping):
        if isinstance(payload.get("scene_tokenizer"), Mapping):
            state = payload["scene_tokenizer"]
            tokenizer_only = True
        elif isinstance(payload.get("state_dict"), Mapping):
            state = payload["state_dict"]
        elif isinstance(payload.get("model"), Mapping):
            state = payload["model"]
    if not isinstance(state, Mapping):
        raise ValueError(f"Unsupported tokenizer checkpoint structure: {source}")
    cleaned = _clean_tensor_state(state)
    prefixed = {
        key[len("scene_tokenizer.") :]: value
        for key, value in cleaned.items()
        if key.startswith("scene_tokenizer.")
    }
    if prefixed:
        cleaned = prefixed
    elif not tokenizer_only:
        # A truly tokenizer-only raw state is accepted only if its keys exactly
        # match below.  A full checkpoint without a tokenizer cannot pass.
        cleaned = dict(cleaned)

    expected = tokenizer.state_dict()
    missing = sorted(set(expected) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(expected))
    shape_mismatches = sorted(
        key
        for key in set(expected).intersection(cleaned)
        if tuple(expected[key].shape) != tuple(cleaned[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Strict tokenizer load failed from {source}: "
            f"missing={missing[:8]} ({len(missing)}), "
            f"unexpected={unexpected[:8]} ({len(unexpected)}), "
            f"shape_mismatches={shape_mismatches[:8]} ({len(shape_mismatches)})"
        )
    tokenizer.load_state_dict(cleaned, strict=True)
    return {
        "state_key_count": len(cleaned),
        "parameter_count": sum(int(parameter.numel()) for parameter in tokenizer.parameters()),
        "buffer_count": sum(int(buffer.numel()) for buffer in tokenizer.buffers()),
    }


def _freeze_eval(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    module.eval()
    module.requires_grad_(False)
    module.to(device=device, dtype=torch.float32)
    return module


def _module_dtypes(module: torch.nn.Module) -> list[str]:
    return sorted({str(parameter.dtype) for parameter in module.parameters()})


def _load_components(
    checkpoint_path: Path,
    tokenizer_checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    # Production classes are imported lazily so --cpu-synthetic-only does not
    # construct or initialize any production model component.
    from dggt.heads.dpt_head import DPTHead, GaussianHead
    from dggt.models.aggregator import Aggregator
    from dggt.models.joint_scene_tokenizer import JointSceneTokenizer

    aggregator = Aggregator(img_size=TARGET_WIDTH, patch_size=PATCH_SIZE, embed_dim=1024)
    depth_head = DPTHead(
        dim_in=2 * 1024,
        output_dim=2,
        activation="exp",
        conf_activation="sigmoid",
    )
    gs_head = GaussianHead(
        dim_in=3 * 1024,
        output_dim=3 + 1 + 3 + 4 + 1,
        activation="sigmoid",
    )
    tokenizer = JointSceneTokenizer()

    payload = _torch_load_weights(checkpoint_path)
    cleaned = _clean_tensor_state(_state_mapping(payload, source=checkpoint_path))
    load_info = {
        "aggregator": _strict_load_prefixed(
            aggregator, cleaned, prefix="aggregator.", source=checkpoint_path
        ),
        "depth_head": _strict_load_prefixed(
            depth_head, cleaned, prefix="depth_head.", source=checkpoint_path
        ),
        "gs_head": _strict_load_prefixed(
            gs_head, cleaned, prefix="gs_head.", source=checkpoint_path
        ),
    }
    del cleaned, payload
    gc.collect()

    tokenizer_payload = _torch_load_weights(tokenizer_checkpoint_path)
    load_info["scene_tokenizer"] = _strict_load_tokenizer(
        tokenizer, tokenizer_payload, source=tokenizer_checkpoint_path
    )
    del tokenizer_payload
    gc.collect()

    components = {
        "aggregator": _freeze_eval(aggregator, device),
        "depth_head": _freeze_eval(depth_head, device),
        "gs_head": _freeze_eval(gs_head, device),
        "scene_tokenizer": _freeze_eval(tokenizer, device),
    }
    load_info["parameter_dtypes_after_move"] = {
        name: _module_dtypes(module) for name, module in components.items()
    }
    for name in ("depth_head", "gs_head"):
        if load_info["parameter_dtypes_after_move"][name] != ["torch.float32"]:
            raise RuntimeError(
                f"Production {name} must be entirely fp32, got "
                f"{load_info['parameter_dtypes_after_move'][name]}"
            )
    return components, load_info


def _target_height(original_width: int, original_height: int) -> int:
    scaled_height = original_height * TARGET_WIDTH / original_width
    result = int(round(scaled_height / PATCH_SIZE) * PATCH_SIZE)
    if result <= 0:
        raise ValueError(
            f"Invalid resized height for source {original_width}x{original_height}: {result}"
        )
    return result


def _load_rgb_trunk(
    scene_root: Path,
    *,
    trunk: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    frames: list[torch.Tensor] = []
    source_sizes: list[tuple[int, int]] = []
    paths: list[str] = []
    target_height: int | None = None
    for local_frame in range(TRUNK_LENGTH):
        global_frame = trunk * TRUNK_LENGTH + local_frame
        path = scene_root / "images" / f"{global_frame:03d}_0.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            this_target_height = _target_height(width, height)
            if target_height is None:
                target_height = this_target_height
            elif this_target_height != target_height:
                raise ValueError(
                    f"Inconsistent resized height in {scene_root}: "
                    f"{this_target_height} vs {target_height}"
                )
            resized = image.resize(
                (TARGET_WIDTH, this_target_height), resample=Image.Resampling.BICUBIC
            )
            array = np.array(resized, dtype=np.uint8, copy=True)
        frames.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0))
        source_sizes.append((width, height))
        paths.append(str(path.resolve()))
    assert target_height is not None
    images = torch.stack(frames, dim=0).unsqueeze(0)
    if tuple(images.shape) != (1, TRUNK_LENGTH, 3, target_height, TARGET_WIDTH):
        raise RuntimeError(f"Unexpected image tensor shape: {tuple(images.shape)}")
    if target_height % PATCH_SIZE or TARGET_WIDTH % PATCH_SIZE:
        raise RuntimeError("Canvas dimensions must be divisible by patch size")
    provenance = {
        "global_frames": [trunk * TRUNK_LENGTH + i for i in range(TRUNK_LENGTH)],
        "source_sizes_wh": [list(value) for value in source_sizes],
        "paths": paths,
        "canvas_hw": [target_height, TARGET_WIDTH],
    }
    return images, provenance


def _load_exclusion_masks(
    scene_root: Path,
    *,
    trunk: int,
    canvas_hw: tuple[int, int],
    source_sizes_wh: Sequence[Sequence[int]],
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    height, width = canvas_hw
    loaded: dict[str, list[torch.Tensor]] = {
        "sky": [],
        "dynamic_fine": [],
        "dynamic_rough": [],
    }
    paths: dict[str, list[str]] = {
        "sky": [],
        "dynamic_fine": [],
        "dynamic_rough": [],
    }
    folder_for = {
        "sky": "sky_masks",
        "dynamic_fine": "fine_dynamic_masks/all",
        "dynamic_rough": "dynamic_masks",
    }
    for kind, folder in folder_for.items():
        for local_frame in range(TRUNK_LENGTH):
            global_frame = trunk * TRUNK_LENGTH + local_frame
            path = scene_root / folder / f"{global_frame:03d}_0.png"
            if not path.is_file():
                return None, {
                    "available": False,
                    "reason": f"missing {kind} mask: {path}",
                    "semantics": "255 means excluded (sky or dynamic); 0 means retained",
                }
            with Image.open(path) as opened:
                mask = opened.convert("L")
                expected_size = tuple(int(value) for value in source_sizes_wh[local_frame])
                if mask.size != expected_size:
                    raise ValueError(
                        f"{kind} mask/image size mismatch for global frame {global_frame}: "
                        f"mask={mask.size}, image={expected_size}"
                    )
                raw = np.asarray(mask)
                unique = set(int(value) for value in np.unique(raw))
                if not unique.issubset({0, 255}):
                    raise ValueError(
                        f"Refusing ambiguous non-binary {kind} mask {path}; values={sorted(unique)[:20]}"
                    )
                resized = mask.resize((width, height), resample=Image.Resampling.NEAREST)
                array = np.array(resized, dtype=np.uint8, copy=True)
            loaded[kind].append(torch.from_numpy(array > 127))
            paths[kind].append(str(path.resolve()))
    result = {kind: torch.stack(values, dim=0) for kind, values in loaded.items()}
    return result, {
        "available": True,
        "semantics": "255 means excluded (sky or dynamic); 0 means retained",
        "dynamic_fine_role": "canonical semantic-pixel dynamic mask used by the primary support",
        "dynamic_rough_role": "legacy moving-box mask retained only as a conservative sensitivity",
        "resize": "PIL nearest-neighbour to the model canvas, without crop",
        "paths": paths,
    }


def _load_lidar_valid_cells(
    scene_root: Path,
    *,
    trunk: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    valid_cells: list[np.ndarray] = []
    paths: list[str] = []
    depth_hw: tuple[int, int] | None = None
    counts: list[int] = []
    for local_frame in range(TRUNK_LENGTH):
        global_frame = trunk * TRUNK_LENGTH + local_frame
        path = scene_root / "depth_flows_4" / f"{global_frame:03d}_0.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or int(array.shape[-1]) < 1:
            raise ValueError(f"Expected HxWxC LiDAR depth-flow array at {path}, got {array.shape}")
        this_hw = (int(array.shape[0]), int(array.shape[1]))
        if depth_hw is None:
            depth_hw = this_hw
        elif this_hw != depth_hw:
            raise ValueError(f"Inconsistent LiDAR grid shape: {this_hw} vs {depth_hw}")
        channel = np.asarray(array[..., 0])
        valid = np.isfinite(channel) & (channel > 0.0)
        valid_cells.append(valid)
        counts.append(int(valid.sum()))
        paths.append(str(path.resolve()))
    assert depth_hw is not None
    return valid_cells, {
        "paths": paths,
        "grid_hw": list(depth_hw),
        "valid_nonzero_cell_counts": counts,
        "channel": 0,
        "valid_rule": "finite and > 0 in original depth_flows_4 grid; zeros are never resized",
    }


def _autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return contextlib.nullcontext()


def _select_image_levels(
    all_levels: Sequence[torch.Tensor],
    *,
    patch_start_idx: int,
    patch_grid: tuple[int, int],
) -> list[torch.Tensor]:
    selected: list[torch.Tensor] = []
    expected_patches = patch_grid[0] * patch_grid[1]
    if len(all_levels) <= max(TOKENIZER_LEVELS):
        raise ValueError(f"Aggregator returned only {len(all_levels)} image-token levels")
    for level in TOKENIZER_LEVELS:
        tokens = all_levels[level]
        if tokens.ndim != 4 or int(tokens.shape[0]) != 1 or int(tokens.shape[1]) != TRUNK_LENGTH:
            raise ValueError(f"Unexpected token shape at level {level}: {tuple(tokens.shape)}")
        if int(tokens.shape[-1]) != 3072:
            raise ValueError(f"Level {level} must be 3072-wide, got {tokens.shape[-1]}")
        if int(tokens.shape[-2]) != patch_start_idx + expected_patches:
            raise ValueError(
                f"Level {level} has {tokens.shape[-2]} tokens; expected "
                f"{patch_start_idx}+{expected_patches}"
            )
        selected.append(tokens.detach())
    return selected


def _sparse_level_list(selected: Sequence[torch.Tensor]) -> list[torch.Tensor | None]:
    if len(selected) != len(TOKENIZER_LEVELS):
        raise ValueError(f"Expected {len(TOKENIZER_LEVELS)} selected levels, got {len(selected)}")
    result: list[torch.Tensor | None] = [None] * (max(TOKENIZER_LEVELS) + 1)
    for level, tokens in zip(TOKENIZER_LEVELS, selected):
        result[level] = tokens
    return result


def _joint_to_aggregated(tokens: torch.Tensor) -> torch.Tensor:
    if int(tokens.shape[-1]) != 3072:
        raise ValueError(f"Joint tokens must be 3072-wide, got {tokens.shape[-1]}")
    _dino, frame, global_tokens = tokens.split((1024, 1024, 1024), dim=-1)
    return torch.cat((frame, global_tokens), dim=-1)


def _run_fp32_heads(
    selected_joint_levels: Sequence[torch.Tensor],
    *,
    depth_head: torch.nn.Module,
    gs_head: torch.nn.Module,
    patch_start_idx: int,
    image_hw: tuple[int, int],
    depth_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp32_joint = [tokens.float().contiguous() for tokens in selected_joint_levels]
    image_levels = _sparse_level_list(fp32_joint)
    aggregated_levels = _sparse_level_list(
        [_joint_to_aggregated(tokens) for tokens in fp32_joint]
    )
    device_type = fp32_joint[0].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        gs_map, gs_conf = gs_head(
            image_levels,
            None,
            patch_start_idx,
            frames_chunk_size=depth_chunk,
            image_hw=image_hw,
        )
        depth, _depth_conf = depth_head(
            aggregated_levels,
            None,
            patch_start_idx,
            frames_chunk_size=depth_chunk,
            image_hw=image_hw,
        )
    if gs_map.dtype != torch.float32 or depth.dtype != torch.float32:
        raise RuntimeError(f"Heads must output fp32, got gs={gs_map.dtype}, depth={depth.dtype}")
    if gs_map.ndim != 5 or int(gs_map.shape[-1]) != 11:
        raise ValueError(f"Expected Gaussian map [B,S,H,W,11], got {tuple(gs_map.shape)}")
    if depth.ndim != 5 or int(depth.shape[-1]) != 1:
        raise ValueError(f"Expected depth [B,S,H,W,1], got {tuple(depth.shape)}")
    return gs_map, depth, gs_conf.float()


def _roundtrip_window(
    selected_trunk_levels: Sequence[torch.Tensor],
    *,
    start: int,
    patch_start_idx: int,
    patch_grid: tuple[int, int],
    tokenizer: torch.nn.Module,
    precision: str,
    device: torch.device,
) -> list[torch.Tensor]:
    end = start + WINDOW_LENGTH
    if start < 0 or end > TRUNK_LENGTH:
        raise ValueError(f"Invalid window [{start}, {end}) for trunk length {TRUNK_LENGTH}")
    originals = [tokens[:, start:end].contiguous() for tokens in selected_trunk_levels]
    patches = [tokens[..., patch_start_idx:, :].contiguous() for tokens in originals]
    with _autocast_context(device, precision):
        latent = tokenizer.encode(patches, patch_grid=patch_grid)
        decoded = tokenizer.decode(latent, patch_grid=patch_grid)
    if len(decoded) != len(TOKENIZER_LEVELS):
        raise RuntimeError(f"Tokenizer decoded {len(decoded)} levels, expected 4")
    reconstructed: list[torch.Tensor] = []
    expected_patches = patch_grid[0] * patch_grid[1]
    for level, original, decoded_patch in zip(TOKENIZER_LEVELS, originals, decoded):
        if tuple(decoded_patch.shape[:3]) != (1, WINDOW_LENGTH, expected_patches):
            raise ValueError(
                f"Decoded level {level} has invalid shape {tuple(decoded_patch.shape)}"
            )
        if int(decoded_patch.shape[-1]) != 3072:
            raise ValueError(f"Decoded level {level} width is {decoded_patch.shape[-1]}, expected 3072")
        special = original[..., :patch_start_idx, :].to(dtype=decoded_patch.dtype)
        reconstructed.append(torch.cat((special, decoded_patch), dim=-2))
    return reconstructed


def _paired_frame_fields(
    direct_gs: torch.Tensor,
    recon_gs: torch.Tensor,
    direct_depth: torch.Tensor,
    recon_depth: torch.Tensor,
    *,
    scale_floor: float = SCALE_FLOOR,
    strict_scale_threshold: float = STRICT_SCALE_THRESHOLD,
) -> dict[str, torch.Tensor]:
    if direct_gs.shape != recon_gs.shape or int(direct_gs.shape[-1]) != 11:
        raise ValueError(
            f"Direct/recon Gaussian maps must match with 11 channels, got "
            f"{tuple(direct_gs.shape)} and {tuple(recon_gs.shape)}"
        )
    if direct_depth.shape != recon_depth.shape:
        raise ValueError("Direct and reconstructed depth shapes do not match")
    if direct_depth.ndim == direct_gs.ndim and int(direct_depth.shape[-1]) == 1:
        direct_depth = direct_depth[..., 0]
        recon_depth = recon_depth[..., 0]
    if direct_depth.shape != direct_gs.shape[:-1]:
        raise ValueError(
            f"Depth spatial shape {tuple(direct_depth.shape)} does not match GS "
            f"{tuple(direct_gs.shape[:-1])}"
        )

    direct_opacity = direct_gs[..., 3]
    recon_opacity = recon_gs[..., 3]
    direct_scale = direct_gs[..., 4:7]
    recon_scale = recon_gs[..., 4:7]
    finite = (
        torch.isfinite(direct_scale).all(dim=-1)
        & torch.isfinite(recon_scale).all(dim=-1)
        & torch.isfinite(direct_depth)
        & torch.isfinite(recon_depth)
        & torch.isfinite(direct_opacity)
        & torch.isfinite(recon_opacity)
    )
    base = finite & (direct_depth > 0.0) & (recon_depth > 0.0)

    direct_scale_clamped = direct_scale.clamp_min(scale_floor)
    recon_scale_clamped = recon_scale.clamp_min(scale_floor)
    log_axis_ratio = torch.log(recon_scale_clamped) - torch.log(direct_scale_clamped)
    log_gs_ratio = log_axis_ratio.mean(dim=-1)
    log_depth_ratio = torch.log(recon_depth.clamp_min(torch.finfo(torch.float32).tiny)) - torch.log(
        direct_depth.clamp_min(torch.finfo(torch.float32).tiny)
    )
    paired_log_error = log_gs_ratio - log_depth_ratio
    anisotropy = torch.sqrt(
        torch.mean((log_axis_ratio - log_gs_ratio.unsqueeze(-1)) ** 2, dim=-1)
    )
    direct_rms_radius = torch.sqrt(torch.mean(direct_scale_clamped**2, dim=-1))
    recon_rms_radius = torch.sqrt(torch.mean(recon_scale_clamped**2, dim=-1))
    log_rms_radius_ratio = torch.log(recon_rms_radius) - torch.log(direct_rms_radius)
    paired_rms_log_error = log_rms_radius_ratio - log_depth_ratio
    return {
        "base": base,
        "direct_depth": direct_depth,
        "recon_depth": recon_depth,
        "direct_opacity": direct_opacity,
        "recon_opacity": recon_opacity,
        "log_axis_ratio": log_axis_ratio,
        "log_gs_ratio": log_gs_ratio,
        "log_depth_ratio": log_depth_ratio,
        "paired_log_error": paired_log_error,
        "anisotropy_log_rms": anisotropy,
        "log_rms_radius_ratio": log_rms_radius_ratio,
        "paired_rms_log_error": paired_rms_log_error,
        "direct_scale_at_floor_axes": direct_scale <= scale_floor,
        "recon_scale_at_floor_axes": recon_scale <= scale_floor,
        "strict_scale": (direct_scale > strict_scale_threshold).all(dim=-1)
        & (recon_scale > strict_scale_threshold).all(dim=-1),
    }


def _tensor_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        raise ValueError("Cannot take a quantile of an empty tensor")
    return float(torch.quantile(values.float(), q).item())


def _frame_row(
    fields: Mapping[str, torch.Tensor],
    support_mask: torch.Tensor,
    *,
    local_frame: int,
    global_frame: int,
) -> dict[str, Any] | None:
    mask = fields["base"] & support_mask.bool()
    count = int(mask.sum().item())
    if count == 0:
        return None

    log_axis = fields["log_axis_ratio"][mask]
    log_gs = fields["log_gs_ratio"][mask]
    log_depth = fields["log_depth_ratio"][mask]
    paired = fields["paired_log_error"][mask]
    anisotropy = fields["anisotropy_log_rms"][mask]
    log_rms = fields["log_rms_radius_ratio"][mask]
    paired_rms = fields["paired_rms_log_error"][mask]
    direct_floor = fields["direct_scale_at_floor_axes"][mask]
    recon_floor = fields["recon_scale_at_floor_axes"][mask]
    strict_count = int((fields["strict_scale"] & mask).sum().item())

    # One vectorized quantile call is materially faster than independently
    # sorting every diagnostic channel for every support/window/frame.
    median_columns = torch.quantile(
        torch.cat(
            (
                log_gs[:, None],
                log_depth[:, None],
                paired[:, None],
                log_axis,
                anisotropy[:, None],
                log_rms[:, None],
                paired_rms[:, None],
            ),
            dim=1,
        ).float(),
        0.5,
        dim=0,
    )
    log_gs_median = float(median_columns[0].item())
    log_depth_median = float(median_columns[1].item())
    paired_median = float(median_columns[2].item())
    axis_medians = [float(value) for value in median_columns[3:6].tolist()]
    anisotropy_median = float(median_columns[6].item())
    log_rms_median = float(median_columns[7].item())
    paired_rms_median = float(median_columns[8].item())
    depth_quartiles = torch.quantile(
        log_depth.float(),
        torch.tensor((0.25, 0.75), device=log_depth.device, dtype=torch.float32),
    )
    depth_log_iqr = float((depth_quartiles[1] - depth_quartiles[0]).item())
    depth_log_mad = _tensor_quantile(torch.abs(log_depth - log_depth_median), 0.5)
    return {
        "local_frame": int(local_frame),
        "global_frame": int(global_frame),
        "valid_pixel_count": count,
        "log_gs_ratio_pixel_median": log_gs_median,
        "gs_ratio": _safe_exp(log_gs_median),
        "log_depth_ratio_pixel_median": log_depth_median,
        "depth_ratio": _safe_exp(log_depth_median),
        "depth_log_ratio_pixel_mad": depth_log_mad,
        "depth_log_ratio_pixel_iqr": depth_log_iqr,
        "paired_log_gs_over_depth_pixel_median": paired_median,
        "paired_gs_over_depth_ratio": _safe_exp(paired_median),
        "separate_medians_gs_over_depth_ratio": _safe_exp(log_gs_median - log_depth_median),
        "axis_log_ratio_pixel_medians": axis_medians,
        "anisotropy_log_rms_pixel_median": anisotropy_median,
        "anisotropy_log_rms_pixel_mean": float(anisotropy.float().mean().item()),
        "anisotropy_log_rms_pixel_p95": _tensor_quantile(anisotropy, 0.95),
        "log_rms_radius_ratio_pixel_median": log_rms_median,
        "rms_radius_ratio": _safe_exp(log_rms_median),
        "paired_rms_radius_over_depth_log_pixel_median": paired_rms_median,
        "paired_rms_radius_over_depth_ratio": _safe_exp(paired_rms_median),
        "direct_scale_at_floor_axis_fraction": float(direct_floor.float().mean().item()),
        "recon_scale_at_floor_axis_fraction": float(recon_floor.float().mean().item()),
        "strict_all_six_axes_above_threshold_count": strict_count,
        "strict_all_six_axes_above_threshold_fraction": strict_count / count,
    }


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _collapse_frame_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "valid_frames": 0,
            "total_valid_pixels": 0,
            "estimator": "unavailable: no valid pixels",
            "log_gs_ratio": None,
            "gs_ratio": None,
            "log_depth_ratio": None,
            "depth_ratio": None,
            "paired_log_gs_over_depth": None,
            "paired_gs_over_depth_ratio": None,
        }

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    log_gs = _median(values("log_gs_ratio_pixel_median"))
    log_depth = _median(values("log_depth_ratio_pixel_median"))
    paired = _median(values("paired_log_gs_over_depth_pixel_median"))
    log_rms = _median(values("log_rms_radius_ratio_pixel_median"))
    paired_rms = _median(values("paired_rms_radius_over_depth_log_pixel_median"))
    axis_logs = [
        _median([float(row["axis_log_ratio_pixel_medians"][axis]) for row in rows])
        for axis in range(3)
    ]
    return {
        "valid_frames": len(rows),
        "total_valid_pixels": sum(int(row["valid_pixel_count"]) for row in rows),
        "estimator": "pixel paired log -> per-frame median -> median across frames -> exp",
        "log_gs_ratio": log_gs,
        "gs_ratio": _safe_exp(log_gs),
        "log_depth_ratio": log_depth,
        "depth_ratio": _safe_exp(log_depth),
        "depth_log_ratio_mad_median_of_frames": _median(
            values("depth_log_ratio_pixel_mad")
        ),
        "depth_log_ratio_iqr_median_of_frames": _median(
            values("depth_log_ratio_pixel_iqr")
        ),
        "depth_log_ratio_mad_mean_of_frames": _mean(
            values("depth_log_ratio_pixel_mad")
        ),
        "depth_log_ratio_iqr_mean_of_frames": _mean(
            values("depth_log_ratio_pixel_iqr")
        ),
        "paired_log_gs_over_depth": paired,
        "paired_gs_over_depth_ratio": _safe_exp(paired),
        "gs_ratio_div_depth_ratio_from_separate_aggregates": _safe_exp(log_gs - log_depth),
        "paired_minus_separate_aggregate_log": paired - (log_gs - log_depth),
        "axis_log_ratio_median_of_frame_medians": axis_logs,
        "axis_ratio_median_of_frame_medians": [_safe_exp(value) for value in axis_logs],
        "anisotropy_log_rms_median_of_frame_medians": _median(
            values("anisotropy_log_rms_pixel_median")
        ),
        "anisotropy_log_rms_mean_of_frame_means": _mean(
            values("anisotropy_log_rms_pixel_mean")
        ),
        "anisotropy_log_rms_median_of_frame_p95": _median(
            values("anisotropy_log_rms_pixel_p95")
        ),
        "log_rms_radius_ratio": log_rms,
        "rms_radius_ratio": _safe_exp(log_rms),
        "paired_rms_radius_over_depth_log": paired_rms,
        "paired_rms_radius_over_depth_ratio": _safe_exp(paired_rms),
        "direct_scale_at_floor_axis_fraction_frame_balanced": _mean(
            values("direct_scale_at_floor_axis_fraction")
        ),
        "recon_scale_at_floor_axis_fraction_frame_balanced": _mean(
            values("recon_scale_at_floor_axis_fraction")
        ),
        "strict_all_six_axes_above_threshold_fraction_frame_balanced": _mean(
            values("strict_all_six_axes_above_threshold_fraction")
        ),
    }


def _summarize_support(
    fields_by_frame: Sequence[Mapping[str, torch.Tensor]],
    masks_by_frame: Sequence[torch.Tensor],
    *,
    local_frames: Sequence[int],
    global_frames: Sequence[int],
    definition: str,
    include_strict_subset: bool = True,
) -> dict[str, Any]:
    if not (
        len(fields_by_frame)
        == len(masks_by_frame)
        == len(local_frames)
        == len(global_frames)
    ):
        raise ValueError("Support inputs do not have equal frame counts")
    rows: list[dict[str, Any]] = []
    for fields, mask, local_frame, global_frame in zip(
        fields_by_frame, masks_by_frame, local_frames, global_frames
    ):
        row = _frame_row(
            fields,
            mask,
            local_frame=int(local_frame),
            global_frame=int(global_frame),
        )
        if row is not None:
            rows.append(row)
    result: dict[str, Any] = {
        "available": True,
        "definition": definition,
        "frame_balanced": _collapse_frame_rows(rows),
        "frames": rows,
    }
    if include_strict_subset:
        strict_masks = [
            mask.bool() & fields["strict_scale"]
            for fields, mask in zip(fields_by_frame, masks_by_frame)
        ]
        result["strict_scale_sensitivity"] = _summarize_support(
            fields_by_frame,
            strict_masks,
            local_frames=local_frames,
            global_frames=global_frames,
            definition=(
                definition
                + f"; additionally all direct/recon scale axes > {STRICT_SCALE_THRESHOLD:g}"
            ),
            include_strict_subset=False,
        )
    return result


def _depth_bin_label(lower_m: float, upper_m: float) -> str:
    return f"{lower_m:g}-{upper_m:g}m"


def _summarize_depth_strata(
    fields_by_frame: Sequence[Mapping[str, torch.Tensor]],
    masks_by_frame: Sequence[torch.Tensor],
    *,
    local_frames: Sequence[int],
    global_frames: Sequence[int],
    direct_units_per_metre: float,
    definition: str,
) -> dict[str, Any]:
    """Summarize paired pullback ratios in fixed metric-depth strata.

    Reconstructed DepthHead values live in the teacher's arbitrary DGGT units.
    The independent full-29-frame LiDAR scale in the reference audit converts
    the *uncorrected reconstruction* to metres.  The fit and runtime therefore
    use the same independent variable; binning raw DGGT depth at
    5/10/20/40/80 would otherwise mix different physical ranges across trunks.
    """

    scale = _positive_float(direct_units_per_metre, name="direct_units_per_metre")
    if not (len(fields_by_frame) == len(masks_by_frame) == len(local_frames) == len(global_frames)):
        raise ValueError("Depth-stratification inputs do not have equal frame counts")
    bins: list[dict[str, Any]] = []
    for bin_index, (lower_m, upper_m) in enumerate(
        zip(DEPTH_BIN_EDGES_M[:-1], DEPTH_BIN_EDGES_M[1:])
    ):
        masks: list[torch.Tensor] = []
        for fields, support in zip(fields_by_frame, masks_by_frame):
            depth_m = fields["recon_depth"].float() / scale
            in_bin = depth_m >= float(lower_m)
            if bin_index == len(DEPTH_BIN_EDGES_M) - 2:
                in_bin &= depth_m <= float(upper_m)
            else:
                in_bin &= depth_m < float(upper_m)
            masks.append(support.bool() & in_bin)
        summary = _summarize_support(
            fields_by_frame,
            masks,
            local_frames=local_frames,
            global_frames=global_frames,
            definition=(
                f"{definition}; uncorrected_reconstructed_depth/{scale:.9g} in "
                f"[{lower_m:g},{upper_m:g}"
                f"{']' if bin_index == len(DEPTH_BIN_EDGES_M) - 2 else ')'} metres"
            ),
            include_strict_subset=False,
        )
        bins.append(
            {
                "index": bin_index,
                "label": _depth_bin_label(lower_m, upper_m),
                "lower_m": float(lower_m),
                "upper_m": float(upper_m),
                "upper_inclusive": bin_index == len(DEPTH_BIN_EDGES_M) - 2,
                **summary,
            }
        )
    return {
        "available": True,
        "depth_source": _depth_profile_variable_contract(),
        "direct_units_per_metre": scale,
        "edges_m": list(DEPTH_BIN_EDGES_M),
        "bins": bins,
    }


def _sample_bhwc_at_lidar_cells(
    values: torch.Tensor,
    valid_cells: np.ndarray,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected HxWxC values, got {tuple(values.shape)}")
    if valid_cells.ndim != 2:
        raise ValueError(f"Expected HxW LiDAR mask, got {valid_cells.shape}")
    rows, cols = np.nonzero(valid_cells)
    if len(rows) == 0:
        return values.new_empty((0, int(values.shape[-1])))
    lidar_h, lidar_w = valid_cells.shape
    x = 2.0 * (torch.as_tensor(cols, device=values.device, dtype=torch.float32) + 0.5) / lidar_w - 1.0
    y = 2.0 * (torch.as_tensor(rows, device=values.device, dtype=torch.float32) + 0.5) / lidar_h - 1.0
    grid = torch.stack((x, y), dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        values.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0, :].transpose(0, 1).contiguous()


def _window_supports(
    direct_gs: torch.Tensor,
    recon_gs: torch.Tensor,
    direct_depth: torch.Tensor,
    recon_depth: torch.Tensor,
    *,
    local_frames: Sequence[int],
    global_frames: Sequence[int],
    lidar_valid_cells: Sequence[np.ndarray],
    exclusion_masks: Mapping[str, torch.Tensor] | None,
    direct_units_per_metre: float,
) -> dict[str, Any]:
    if int(direct_gs.shape[0]) != 1 or int(direct_gs.shape[1]) != WINDOW_LENGTH:
        raise ValueError(f"Unexpected direct GS window shape: {tuple(direct_gs.shape)}")
    if direct_gs.shape != recon_gs.shape or direct_depth.shape != recon_depth.shape:
        raise ValueError("Direct/recon window outputs do not match")

    canvas_fields = [
        _paired_frame_fields(
            direct_gs[0, index],
            recon_gs[0, index],
            direct_depth[0, index],
            recon_depth[0, index],
        )
        for index in range(WINDOW_LENGTH)
    ]
    all_masks = [torch.ones_like(fields["base"]) for fields in canvas_fields]
    supports: dict[str, Any] = {
        "all_canvas": _summarize_support(
            canvas_fields,
            all_masks,
            local_frames=local_frames,
            global_frames=global_frames,
            definition=(
                "all model-canvas pixels with finite GS/depth values and positive direct/recon depth"
            ),
        )
    }
    for threshold in OPACITY_THRESHOLDS:
        key = f"opacity_intersection_{threshold:g}".replace(".", "p")
        opacity_masks = [
            (fields["direct_opacity"] > threshold)
            & (fields["recon_opacity"] > threshold)
            for fields in canvas_fields
        ]
        supports[key] = _summarize_support(
            canvas_fields,
            opacity_masks,
            local_frames=local_frames,
            global_frames=global_frames,
            definition=(
                f"same canvas pixels where direct opacity > {threshold:g} and "
                f"reconstructed opacity > {threshold:g}, plus finite/positive-depth base rule"
            ),
        )

    opacity_005_key = "opacity_intersection_0p05"
    if exclusion_masks is None:
        supports["primary_static_nonsky_opacity_0p05"] = {
            "available": False,
            "reason": "strict binary sky/dynamic masks were unavailable",
        }
    else:
        primary_masks = []
        rough_masks = []
        for index, fields in enumerate(canvas_fields):
            retained = (~exclusion_masks["sky"][index]) & (~exclusion_masks["dynamic_fine"][index])
            retained = retained.to(device=fields["base"].device)
            retained &= (fields["direct_opacity"] > 0.05) & (fields["recon_opacity"] > 0.05)
            primary_masks.append(retained)
            rough = (~exclusion_masks["sky"][index]) & (~exclusion_masks["dynamic_rough"][index])
            rough = rough.to(device=fields["base"].device)
            rough &= (fields["direct_opacity"] > 0.05) & (fields["recon_opacity"] > 0.05)
            rough_masks.append(rough)
        supports["primary_static_nonsky_opacity_0p05"] = _summarize_support(
            canvas_fields,
            primary_masks,
            local_frames=local_frames,
            global_frames=global_frames,
            definition=(
                "primary: same finite positive-depth canvas pixels, direct/recon opacity > 0.05, "
                "and original-data binary sky/canonical fine-dynamic masks both retain the pixel"
            ),
        )
        supports["primary_static_nonsky_opacity_0p05"]["depth_stratification"] = (
            _summarize_depth_strata(
                canvas_fields,
                primary_masks,
                local_frames=local_frames,
                global_frames=global_frames,
                direct_units_per_metre=direct_units_per_metre,
                definition=(
                    "primary static/non-sky/opacity support, stratified by LiDAR-calibrated "
                    "uncorrected reconstructed z-depth"
                ),
            )
        )
        supports["rough_bbox_static_nonsky_opacity_0p05_sensitivity"] = _summarize_support(
            canvas_fields,
            rough_masks,
            local_frames=local_frames,
            global_frames=global_frames,
            definition=(
                "sensitivity: same finite positive-depth canvas pixels, direct/recon opacity > 0.05, "
                "and sky/legacy rough moving-box dynamic masks both retain the pixel"
            ),
        )

    lidar_fields: list[dict[str, torch.Tensor]] = []
    lidar_masks: list[torch.Tensor] = []
    for index, valid in enumerate(lidar_valid_cells):
        joined = torch.cat(
            (
                direct_gs[0, index, ..., 3:7],
                recon_gs[0, index, ..., 3:7],
                direct_depth[0, index],
                recon_depth[0, index],
            ),
            dim=-1,
        )
        sampled = _sample_bhwc_at_lidar_cells(joined, valid)
        count = int(sampled.shape[0])
        direct_stub = sampled.new_zeros((count, 11))
        recon_stub = sampled.new_zeros((count, 11))
        direct_stub[:, 3:7] = sampled[:, 0:4]
        recon_stub[:, 3:7] = sampled[:, 4:8]
        frame_fields = _paired_frame_fields(
            direct_stub,
            recon_stub,
            sampled[:, 8:9],
            sampled[:, 9:10],
        )
        lidar_fields.append(frame_fields)
        lidar_masks.append(torch.ones_like(frame_fields["base"]))
    supports["lidar_depth_cell_centers"] = _summarize_support(
        lidar_fields,
        lidar_masks,
        local_frames=local_frames,
        global_frames=global_frames,
        definition=(
            "bilinear sample at centers of original nonzero finite depth_flows_4 channel-0 cells; "
            "align_corners=False; original zero cells are never resized or included"
        ),
    )
    supports["lidar_depth_cell_centers"]["depth_stratification"] = _summarize_depth_strata(
        lidar_fields,
        lidar_masks,
        local_frames=local_frames,
        global_frames=global_frames,
        direct_units_per_metre=direct_units_per_metre,
        definition=(
            "original LiDAR-cell-center support, stratified by uncorrected "
            "reconstructed z-depth"
        ),
    )

    if opacity_005_key not in supports:
        raise AssertionError("Required opacity-0.05 support was not produced")
    return supports


def _reference_case_index(payload: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Reference result JSON must contain a cases list")
    index: dict[tuple[int, int], Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping) or "scene" not in case or "trunk" not in case:
            raise ValueError("Every reference case must contain scene and trunk")
        key = (int(case["scene"]), int(case["trunk"]))
        if key in index:
            raise ValueError(f"Duplicate reference case {key}")
        index[key] = case
    return index


def _reference_window(
    case: Mapping[str, Any],
    *,
    scene: int,
    trunk: int,
    start: int,
) -> dict[str, float]:
    roundtrip = case.get("tokenizer_roundtrip")
    if not isinstance(roundtrip, Mapping):
        raise ValueError(
            f"Reference case scene={scene} trunk={trunk} lacks tokenizer_roundtrip; "
            "rerun the independent metric-gauge tool with round-trip depth enabled"
        )
    windows = roundtrip.get("windows")
    if not isinstance(windows, list):
        raise ValueError(f"Reference tokenizer_roundtrip.windows is absent for {(scene, trunk)}")
    matches = [row for row in windows if isinstance(row, Mapping) and int(row.get("start", -1)) == start]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one reference window start={start} for {(scene, trunk)}, got {len(matches)}"
        )
    row = matches[0]
    same_cells = row.get("roundtrip_over_direct_same_cells")
    metrics = row.get("metrics")
    full_depth = case.get("depth")
    if (
        not isinstance(same_cells, Mapping)
        or not isinstance(metrics, Mapping)
        or not isinstance(full_depth, Mapping)
    ):
        raise ValueError(f"Malformed reference round-trip window for {(scene, trunk, start)}")
    return {
        "full_trunk_direct_units_per_metre": _positive_float(
            full_depth.get("scale_frame_balanced"),
            name="reference depth.scale_frame_balanced (full 29-frame trunk)",
        ),
        "same_cell_depth_ratio": _positive_float(
            same_cells.get("scale_frame_balanced"),
            name="reference roundtrip_over_direct_same_cells.scale_frame_balanced",
        ),
        "ratio_of_separate_depth_scales": _positive_float(
            row.get("roundtrip_over_direct_scale"),
            name="reference roundtrip_over_direct_scale",
        ),
        "reconstructed_depth_lidar_scale": _positive_float(
            metrics.get("scale_frame_balanced"),
            name="reference metrics.scale_frame_balanced",
        ),
        "direct_depth_lidar_scale_same_frames": _positive_float(
            row.get("direct_depth_scale_same_frames"),
            name="reference direct_depth_scale_same_frames",
        ),
    }


def _crosscheck_reference(
    supports: Mapping[str, Any],
    reference: Mapping[str, float],
) -> dict[str, Any]:
    local = supports["lidar_depth_cell_centers"]["frame_balanced"]
    local_ratio = _positive_float(local["depth_ratio"], name="local LiDAR-cell depth ratio")
    reference_ratio = _positive_float(
        reference["same_cell_depth_ratio"], name="reference LiDAR-cell depth ratio"
    )
    ratio = local_ratio / reference_ratio
    if abs(ratio - 1.0) > 1.0e-4:
        raise RuntimeError(
            "Independent LiDAR-cell depth cross-check failed: "
            f"local/reference={ratio:.9f} exceeds rtol=1e-4"
        )
    local_gs_ratio = _positive_float(local["gs_ratio"], name="local LiDAR-cell GS ratio")
    return {
        "role": "cross-check only; primary paired GS/depth metric is computed directly in this tool",
        "reference_fields": dict(reference),
        "local_same_cell_depth_ratio": local_ratio,
        "local_over_reference_depth_ratio": ratio,
        "local_minus_reference_log_depth_ratio": math.log(local_ratio) - math.log(reference_ratio),
        "absolute_relative_difference": abs(ratio - 1.0),
        "within_rtol_1e-4": abs(ratio - 1.0) <= 1.0e-4,
        "lidar_gs_ratio_div_reference_depth_ratio_unpaired_crosscheck": (
            local_gs_ratio / reference_ratio
        ),
        "warning": (
            "The last value divides two separately aggregated statistics and is not the primary estimator; "
            "use supports.*.frame_balanced.paired_gs_over_depth_ratio for inference."
        ),
    }


STRICT_SUPPORT_SUFFIX = "__strict_scale_sensitivity"


def _support_names(window: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for name, row in window["supports"].items():
        names.add(str(name))
        if isinstance(row, Mapping) and isinstance(row.get("strict_scale_sensitivity"), Mapping):
            names.add(str(name) + STRICT_SUPPORT_SUFFIX)
    return names


def _resolve_support(window: Mapping[str, Any], support: str) -> Any:
    if support.endswith(STRICT_SUPPORT_SUFFIX):
        base = support[: -len(STRICT_SUPPORT_SUFFIX)]
        row = window["supports"].get(base)
        return row.get("strict_scale_sensitivity") if isinstance(row, Mapping) else None
    return window["supports"].get(support)


def _window_metric_rows(windows: Sequence[Mapping[str, Any]], support: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for window in windows:
        support_row = _resolve_support(window, support)
        if not isinstance(support_row, Mapping) or not support_row.get("available", False):
            continue
        balanced = support_row.get("frame_balanced")
        if not isinstance(balanced, Mapping) or balanced.get("paired_log_gs_over_depth") is None:
            continue
        rows.append(
            {
                "log_gs": float(balanced["log_gs_ratio"]),
                "log_depth": float(balanced["log_depth_ratio"]),
                "paired": float(balanced["paired_log_gs_over_depth"]),
                "anisotropy": float(balanced["anisotropy_log_rms_median_of_frame_medians"]),
                "paired_rms": float(balanced["paired_rms_radius_over_depth_log"]),
                "depth_log_mad": float(balanced["depth_log_ratio_mad_median_of_frames"]),
                "depth_log_iqr": float(balanced["depth_log_ratio_iqr_median_of_frames"]),
            }
        )
    return rows


def _aggregate_log_rows(rows: Sequence[Mapping[str, float]], *, unit: str) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "unit": unit, "available": False}
    result: dict[str, Any] = {"count": len(rows), "unit": unit, "available": True}
    for key in (
        "log_gs",
        "log_depth",
        "paired",
        "anisotropy",
        "paired_rms",
        "depth_log_mad",
        "depth_log_iqr",
    ):
        values = [float(row[key]) for row in rows]
        result[f"{key}_median"] = _median(values)
        result[f"{key}_mean"] = _mean(values)
        result[f"{key}_q25"] = float(np.quantile(values, 0.25))
        result[f"{key}_q75"] = float(np.quantile(values, 0.75))
    result["gs_ratio_at_median_log"] = _safe_exp(result["log_gs_median"])
    result["depth_ratio_at_median_log"] = _safe_exp(result["log_depth_median"])
    result["paired_gs_over_depth_ratio_at_median_log"] = _safe_exp(result["paired_median"])
    result["paired_gs_over_depth_geometric_mean"] = _safe_exp(result["paired_mean"])
    result["paired_rms_radius_over_depth_ratio_at_median_log"] = _safe_exp(
        result["paired_rms_median"]
    )
    return result


def _case_summary(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support_names = sorted(
        set.intersection(
            *(_support_names(window) for window in windows)
        )
    ) if windows else []
    return {
        "unit": "five overlapping tokenizer windows within one 29-frame trunk",
        "non_independent_windows": True,
        "support_summaries": {
            support: _aggregate_log_rows(
                _window_metric_rows(windows, support), unit="overlapping window"
            )
            for support in support_names
        },
    }


def _global_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"case_count": 0}
    support_names = sorted(
        set.intersection(
            *(
                set(case["trunk_balanced"]["support_summaries"].keys())
                for case in cases
            )
        )
    )
    summaries: dict[str, Any] = {}
    metric_keys = (
        "log_gs",
        "log_depth",
        "paired",
        "anisotropy",
        "paired_rms",
        "depth_log_mad",
        "depth_log_iqr",
    )
    for support in support_names:
        window_rows: list[dict[str, float]] = []
        case_rows: list[dict[str, float]] = []
        case_rows_by_scene: dict[str, list[dict[str, float]]] = {}
        for case in cases:
            window_rows.extend(_window_metric_rows(case["windows"], support))
            rows = _window_metric_rows(case["windows"], support)
            if rows:
                case_row = {key: _median([row[key] for row in rows]) for key in metric_keys}
                case_rows.append(case_row)
                case_rows_by_scene.setdefault(str(case["scene"]), []).append(case_row)
        scene_rows = [
            {
                key: _median([row[key] for row in rows])
                for key in metric_keys
            }
            for rows in case_rows_by_scene.values()
        ]
        summaries[support] = {
            "window_level_descriptive_only": _aggregate_log_rows(
                window_rows, unit="overlapping window (non-independent)"
            ),
            "trunk_case_balanced": _aggregate_log_rows(
                case_rows, unit="29-frame trunk (three trunks within a scene remain clustered)"
            ),
            "scene_balanced_primary": _aggregate_log_rows(
                scene_rows, unit="Waymo scene (median over its requested trunks)"
            ),
        }
    cross_errors = [
        float(window["reference_depth_crosscheck"]["local_minus_reference_log_depth_ratio"])
        for case in cases
        for window in case["windows"]
    ]
    return {
        "case_count": len(cases),
        "window_count": sum(len(case["windows"]) for case in cases),
        "analysis_unit": "Waymo scene; requested 29-frame trunks are clustered within scene",
        "window_dependence_warning": (
            "The five ten-frame windows overlap and are not independent. "
            "Use scene_balanced_primary results for across-dataset inference."
        ),
        "supports": summaries,
        "reference_depth_crosscheck": {
            "count": len(cross_errors),
            "median_log_difference": _median(cross_errors),
            "max_absolute_log_difference": max(abs(value) for value in cross_errors),
        },
    }


def _scene_balanced_paired_rows(
    cases: Sequence[Mapping[str, Any]],
    *,
    support: str = D4_PRIMARY_SUPPORT,
) -> list[dict[str, Any]]:
    """Collapse overlapping windows and trunks to one paired value per scene."""

    by_scene: dict[int, list[float]] = {}
    trunk_counts: dict[int, int] = {}
    for case in cases:
        rows = _window_metric_rows(case.get("windows", []), support)
        if not rows:
            continue
        scene = int(case["scene"])
        by_scene.setdefault(scene, []).append(_median([float(row["paired"]) for row in rows]))
        trunk_counts[scene] = trunk_counts.get(scene, 0) + 1
    return [
        {
            "scene": scene,
            "trunk_count": trunk_counts[scene],
            "paired_log_gs_over_depth": _median(by_scene[scene]),
            "paired_gs_over_depth_ratio": math.exp(_median(by_scene[scene])),
        }
        for scene in sorted(by_scene)
    ]


def _paired_practical_equivalence_gate(
    cases: Sequence[Mapping[str, Any]],
    *,
    support: str = D4_PRIMARY_SUPPORT,
    bootstrap_samples: int = PAIRED_EQUIVALENCE_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = PAIRED_EQUIVALENCE_BOOTSTRAP_SEED,
    margin: tuple[float, float] = PAIRED_EQUIVALENCE_MARGIN,
) -> dict[str, Any]:
    """Evaluate the frozen GS/depth equivalence margin with scene-only bootstrap."""

    samples = int(bootstrap_samples)
    if samples <= 0:
        raise ValueError("paired practical-equivalence bootstrap samples must be positive")
    lower_margin, upper_margin = (float(margin[0]), float(margin[1]))
    if not (0.0 < lower_margin < upper_margin):
        raise ValueError("paired practical-equivalence margin must be positive and ordered")
    scene_rows = _scene_balanced_paired_rows(cases, support=support)
    base = {
        "support": support,
        "analysis_unit": "Waymo scene; bootstrap resamples scenes only",
        "estimator": (
            "pixel paired log -> frame median -> window median -> trunk median -> "
            "scene median -> across-scene median -> exp"
        ),
        "margin": [lower_margin, upper_margin],
        "margin_frozen_before_v2_results": True,
        "bootstrap": {
            "unit": "scene",
            "clustered_lower_units": "trunks, overlapping windows, frames, and pixels",
            "samples": samples,
            "seed": int(bootstrap_seed),
            "confidence_level": 0.95,
        },
        "scene_count": len(scene_rows),
        "scene_order": [int(row["scene"]) for row in scene_rows],
        "per_scene": scene_rows,
    }
    if not scene_rows:
        return {
            **base,
            "available": False,
            "passed": False,
            "reason": "no scene-balanced paired observations for the primary support",
        }

    scene_logs = np.asarray(
        [float(row["paired_log_gs_over_depth"]) for row in scene_rows], dtype=np.float64
    )
    if not np.isfinite(scene_logs).all():
        raise ValueError("scene-balanced paired observations must be finite")
    rng = np.random.default_rng(int(bootstrap_seed))
    draw_indices = rng.integers(
        0,
        len(scene_logs),
        size=(samples, len(scene_logs)),
        endpoint=False,
    )
    bootstrap_log_medians = np.median(scene_logs[draw_indices], axis=1)
    point_ratio = math.exp(float(np.median(scene_logs)))
    ci_log = np.quantile(bootstrap_log_medians, (0.025, 0.975))
    ci = (math.exp(float(ci_log[0])), math.exp(float(ci_log[1])))
    point_pass = lower_margin <= point_ratio <= upper_margin
    ci_pass = lower_margin <= ci[0] and ci[1] <= upper_margin
    return {
        **base,
        "available": True,
        "point_estimate": point_ratio,
        "scene_bootstrap_95_ci": [ci[0], ci[1]],
        "point_within_margin": point_pass,
        "ci_entirely_within_margin": ci_pass,
        "passed": point_pass and ci_pass,
        "decision_rule": (
            "pass iff the scene-balanced point estimate and the entire scene-bootstrap "
            "95% CI lie within the inclusive frozen margin"
        ),
    }


def _depth_bin_representative_m(lower_m: float, upper_m: float) -> float:
    if lower_m <= 0.0:
        return 0.5 * upper_m
    return math.sqrt(lower_m * upper_m)


def _depth_profile_rows(
    cases: Sequence[Mapping[str, Any]],
    *,
    scenes: Sequence[int],
    support: str = D4_PRIMARY_SUPPORT,
) -> list[dict[str, float | int]]:
    """Collapse overlapping windows/trunks to one value per scene/depth bin."""

    allowed = {int(scene) for scene in scenes}
    accumulated: dict[tuple[int, int], list[float]] = {}
    bin_meta: dict[int, tuple[float, float]] = {}
    for case in cases:
        scene = int(case["scene"])
        if scene not in allowed:
            continue
        for window in case.get("windows", []):
            support_row = _resolve_support(window, support)
            if not isinstance(support_row, Mapping):
                continue
            strata = support_row.get("depth_stratification")
            if not isinstance(strata, Mapping):
                continue
            for bin_row in strata.get("bins", []):
                if not isinstance(bin_row, Mapping):
                    continue
                balanced = bin_row.get("frame_balanced")
                if not isinstance(balanced, Mapping):
                    continue
                if balanced.get("log_depth_ratio") is None:
                    continue
                bin_index = int(bin_row["index"])
                bin_meta[bin_index] = (float(bin_row["lower_m"]), float(bin_row["upper_m"]))
                accumulated.setdefault((scene, bin_index), []).append(
                    float(balanced["log_depth_ratio"])
                )

    output: list[dict[str, float | int]] = []
    for (scene, bin_index), log_depth_values in sorted(accumulated.items()):
        lower_m, upper_m = bin_meta[bin_index]
        depth_m = _depth_bin_representative_m(lower_m, upper_m)
        output.append(
            {
                "scene": scene,
                "bin_index": bin_index,
                "depth_m": depth_m,
                "lower_m": lower_m,
                "upper_m": upper_m,
                "source_observation_count": len(log_depth_values),
                # depth_direct = depth_recon * c_depth
                "log_correction": -_median(log_depth_values),
                "observed_log_ratio": _median(log_depth_values),
            }
        )
    return output


def _weighted_profile_fit(
    rows: Sequence[Mapping[str, Any]],
    *,
    linear: bool,
) -> tuple[float, float]:
    if not rows:
        raise ValueError("Cannot fit an empty depth profile")
    scene_counts: dict[Any, int] = {}
    for row in rows:
        scene_counts[row["scene"]] = scene_counts.get(row["scene"], 0) + 1
    x = np.asarray(
        [math.log(float(row["depth_m"]) / DEPTH_PROFILE_REFERENCE_M) for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([float(row["log_correction"]) for row in rows], dtype=np.float64)
    weights = np.asarray([1.0 / scene_counts[row["scene"]] for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all():
        raise ValueError("Depth-profile observations must be finite")
    if linear:
        design = np.stack((np.ones_like(x), x), axis=1)
        lhs = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * y)
        if np.linalg.matrix_rank(lhs) < 2:
            raise ValueError("Depth-profile design is rank deficient")
        a, b = np.linalg.solve(lhs, rhs)
        return float(a), float(b)
    return float(np.sum(weights * y) / np.sum(weights)), 0.0


def _profile_spearman(rows: Sequence[Mapping[str, Any]]) -> tuple[float, list[dict[str, float | int]]]:
    by_bin: dict[int, list[float]] = {}
    depth_by_bin: dict[int, float] = {}
    for row in rows:
        index = int(row["bin_index"])
        by_bin.setdefault(index, []).append(float(row["log_correction"]))
        depth_by_bin[index] = float(row["depth_m"])
    table = [
        {
            "bin_index": index,
            "depth_m": depth_by_bin[index],
            "scene_count": len(by_bin[index]),
            "median_log_correction": _median(by_bin[index]),
            "median_correction": math.exp(_median(by_bin[index])),
        }
        for index in sorted(by_bin)
    ]
    if len(table) < 2:
        return float("nan"), table
    values = np.asarray([row["median_log_correction"] for row in table], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    ordered = np.arange(len(table), dtype=np.float64)
    if float(np.std(ranks)) == 0.0:
        return 0.0, table
    return float(np.corrcoef(ordered, ranks)[0, 1]), table


def _profile_loso_rmse(rows: Sequence[Mapping[str, Any]], *, linear: bool) -> float:
    scenes = sorted({row["scene"] for row in rows})
    errors: list[float] = []
    for scene in scenes:
        train = [row for row in rows if row["scene"] != scene]
        test = [row for row in rows if row["scene"] == scene]
        if not train or not test:
            continue
        try:
            a, b = _weighted_profile_fit(train, linear=linear)
        except ValueError:
            return float("inf")
        squared = []
        for row in test:
            x = math.log(float(row["depth_m"]) / DEPTH_PROFILE_REFERENCE_M)
            squared.append((float(row["log_correction"]) - (a + b * x)) ** 2)
        errors.append(_mean(squared))
    return math.sqrt(_mean(errors)) if errors else float("inf")


def _profile_identity_scene_rmse(rows: Sequence[Mapping[str, Any]]) -> float:
    """Score the no-fit identity profile with the same scene-balanced loss as LOSO."""

    scenes = sorted({row["scene"] for row in rows})
    scene_mse = [
        _mean(
            [
                float(row["log_correction"]) ** 2
                for row in rows
                if row["scene"] == scene
            ]
        )
        for scene in scenes
    ]
    return math.sqrt(_mean(scene_mse)) if scene_mse else float("inf")


def _profile_slope_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int = D4_FORM_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    scenes = sorted({int(row["scene"]) for row in rows})
    if len(scenes) < 2 or int(samples) <= 0:
        return float("nan"), float("nan")
    by_scene = {scene: [row for row in rows if int(row["scene"]) == scene] for scene in scenes}
    rng = np.random.default_rng(int(seed))
    slopes: list[float] = []
    for _ in range(int(samples)):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        bootstrap_rows: list[dict[str, Any]] = []
        for draw_index, scene in enumerate(sampled.tolist()):
            for row in by_scene[int(scene)]:
                bootstrap_rows.append({**row, "scene": (draw_index, int(scene))})
        try:
            _a, slope = _weighted_profile_fit(bootstrap_rows, linear=True)
        except ValueError:
            continue
        slopes.append(slope)
    if not slopes:
        return float("nan"), float("nan")
    return (
        float(np.quantile(np.asarray(slopes), 0.025)),
        float(np.quantile(np.asarray(slopes), 0.975)),
    )


def _decide_profile_form(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Choose identity, constant, or loglinear using frozen scene-LOSO gates."""

    if not rows:
        return {"available": False, "reason": "no scene/depth-bin observations"}
    scenes = sorted({int(row["scene"]) for row in rows})
    bins = sorted({int(row["bin_index"]) for row in rows})
    constant_a, _ = _weighted_profile_fit(rows, linear=False)
    linear_a, linear_b = _weighted_profile_fit(rows, linear=True)
    spearman, bin_table = _profile_spearman(rows)
    slope_ci = _profile_slope_bootstrap_ci(rows, samples=int(bootstrap_samples))
    identity_rmse = _profile_identity_scene_rmse(rows)
    constant_rmse = _profile_loso_rmse(rows, linear=False)
    linear_rmse = _profile_loso_rmse(rows, linear=True)
    loglinear_vs_constant_improvement = (
        (constant_rmse - linear_rmse) / constant_rmse
        if math.isfinite(constant_rmse) and constant_rmse > 0.0
        else float("-inf")
    )
    ci_excludes_zero = (
        all(math.isfinite(value) for value in slope_ci)
        and (slope_ci[0] > 0.0 or slope_ci[1] < 0.0)
    )
    monotonic = math.isfinite(spearman) and abs(spearman) >= D4_FORM_MIN_ABS_SPEARMAN
    loglinear = (
        len(scenes) >= 10
        and len(bins) >= 4
        and monotonic
        and ci_excludes_zero
        and loglinear_vs_constant_improvement >= D4_FORM_MIN_CV_IMPROVEMENT_FRACTION
    )
    fitted_form = "loglinear" if loglinear else "constant"
    fitted_a, fitted_b = (linear_a, linear_b) if loglinear else (constant_a, 0.0)
    fitted_rmse = linear_rmse if loglinear else constant_rmse
    fitted_vs_identity_improvement = (
        (identity_rmse - fitted_rmse) / identity_rmse
        if math.isfinite(identity_rmse) and identity_rmse > 0.0
        else float("-inf")
    )
    fitted_beats_identity = (
        len(scenes) >= 10
        and math.isfinite(fitted_rmse)
        and fitted_vs_identity_improvement
        >= D4_IDENTITY_MIN_CV_IMPROVEMENT_FRACTION
    )
    selected_form = fitted_form if fitted_beats_identity else "identity"
    selected_a, selected_b = (fitted_a, fitted_b) if fitted_beats_identity else (0.0, 0.0)
    return {
        "available": True,
        "selected_form": selected_form,
        "selected": {
            "equation": "log c(d) = a + b * log(d / 20m)",
            "a": selected_a,
            "b": selected_b,
            "c_at_20m": math.exp(selected_a),
        },
        "identity": {"a": 0.0, "b": 0.0, "c_at_20m": 1.0},
        "fitted_candidate": {
            "form": fitted_form,
            "a": fitted_a,
            "b": fitted_b,
            "c_at_20m": math.exp(fitted_a),
        },
        "constant_fit": {"a": constant_a, "b": 0.0, "c": math.exp(constant_a)},
        "loglinear_fit": {
            "a": linear_a,
            "b": linear_b,
            "c_at_20m": math.exp(linear_a),
            "slope_scene_bootstrap_95_ci": [_finite_or_none(value) for value in slope_ci],
        },
        "scene_count": len(scenes),
        "bin_count": len(bins),
        "observation_count": len(rows),
        "bin_scene_balanced_medians": bin_table,
        "bin_median_spearman": _finite_or_none(spearman),
        "leave_one_scene_out_rmse": {
            "identity": _finite_or_none(identity_rmse),
            "constant": _finite_or_none(constant_rmse),
            "loglinear": _finite_or_none(linear_rmse),
            "fitted_candidate": _finite_or_none(fitted_rmse),
            "loglinear_vs_constant_relative_improvement": _finite_or_none(
                loglinear_vs_constant_improvement
            ),
            "fitted_vs_identity_relative_improvement": _finite_or_none(
                fitted_vs_identity_improvement
            ),
        },
        "gate": {
            "minimum_scenes": 10,
            "minimum_populated_bins": 4,
            "minimum_abs_bin_spearman": D4_FORM_MIN_ABS_SPEARMAN,
            "slope_ci_must_exclude_zero": True,
            "minimum_loso_rmse_improvement_fraction": D4_FORM_MIN_CV_IMPROVEMENT_FRACTION,
            "monotonic_pass": monotonic,
            "slope_ci_pass": ci_excludes_zero,
            "cv_pass": (
                loglinear_vs_constant_improvement
                >= D4_FORM_MIN_CV_IMPROVEMENT_FRACTION
            ),
            "identity_vs_fitted": {
                "minimum_scenes": 10,
                "minimum_loso_rmse_improvement_fraction": (
                    D4_IDENTITY_MIN_CV_IMPROVEMENT_FRACTION
                ),
                "pass": fitted_beats_identity,
                "fallback_when_not_passed": "identity",
            },
        },
    }


def _build_depth_profile_summary(
    cases: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    calib = _depth_profile_rows(cases, scenes=D4_CALIBRATION_SCENES)
    selection = _depth_profile_rows(cases, scenes=D4_HOLDOUT_SCENES)
    decision = _decide_profile_form(calib, bootstrap_samples=bootstrap_samples)
    variable_contract = _depth_profile_variable_contract()
    return {
        "metric_depth_edges_m": list(DEPTH_BIN_EDGES_M),
        "calibration_scenes": list(D4_CALIBRATION_SCENES),
        "selection_scenes": list(D4_HOLDOUT_SCENES),
        "fit_data_boundary": "only calibration rows enter form/coefficient fitting",
        "fit_variable": variable_contract,
        "runtime_variable": dict(variable_contract),
        "candidate_forms": ["identity", "constant", "loglinear"],
        "selection_role": (
            "never refits form or coefficients; depth-bin rows are diagnostic-only in this tool"
        ),
        "calibration_scene_bin_rows": calib,
        "selection_scene_bin_rows_diagnostic_only": selection,
        "form_decision": decision,
    }


def _depth_profile_correction(
    metric_depth_m: torch.Tensor,
    *,
    c_at_20m: float,
    slope: float,
) -> torch.Tensor:
    centre = _positive_float(c_at_20m, name="c_at_20m")
    exponent = _finite_float(slope, name="slope")
    depth = torch.as_tensor(metric_depth_m).float().clamp(0.5, 80.0)
    return depth.div(DEPTH_PROFILE_REFERENCE_M).log().mul(exponent).add(math.log(centre)).exp()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _overlap_provenance() -> dict[str, Any]:
    multiplicity = [0] * TRUNK_LENGTH
    for start in WINDOW_STARTS:
        for frame in range(start, start + WINDOW_LENGTH):
            multiplicity[frame] += 1
    return {
        "starts": list(WINDOW_STARTS),
        "window_length": WINDOW_LENGTH,
        "total_frame_observations": len(WINDOW_STARTS) * WINDOW_LENGTH,
        "unique_trunk_frames": sum(value > 0 for value in multiplicity),
        "per_local_frame_multiplicity": multiplicity,
        "non_independent": True,
    }


def _validate_reference_provenance(
    reference_payload: Mapping[str, Any],
    *,
    dggt_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    metadata = reference_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Reference result JSON lacks a metadata object")
    requested_dggt = _require_sha256(dggt_sha256, name="requested DGGT checkpoint SHA-256")
    requested_tokenizer = _require_sha256(
        tokenizer_sha256, name="requested tokenizer checkpoint SHA-256"
    )
    ref_dggt = _require_sha256(
        metadata.get("checkpoint_sha256"), name="reference DGGT checkpoint SHA-256"
    )
    ref_tokenizer = _require_sha256(
        metadata.get("tokenizer_checkpoint_sha256"),
        name="reference tokenizer checkpoint SHA-256",
    )
    if ref_dggt != requested_dggt:
        raise ValueError(
            f"Reference DGGT checkpoint hash differs: {ref_dggt} != {requested_dggt}"
        )
    if ref_tokenizer != requested_tokenizer:
        raise ValueError(
            "Reference tokenizer checkpoint hash differs: "
            f"{ref_tokenizer} != {requested_tokenizer}"
        )
    return {
        "reference_checkpoint_sha256": ref_dggt,
        "reference_tokenizer_checkpoint_sha256": ref_tokenizer,
        "dggt_hash_match": True,
        "tokenizer_hash_match": True,
        "reference_schema": reference_payload.get("schema"),
        "reference_script": metadata.get("script"),
        "reference_script_sha256": metadata.get("script_sha256"),
    }


def _run_case(
    *,
    scene: int,
    trunk: int,
    data_root: Path,
    components: Mapping[str, torch.nn.Module],
    device: torch.device,
    precision: str,
    depth_chunk: int,
    reference_case: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.time()
    scene_name = f"{scene:03d}"
    scene_root = data_root / scene_name
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)
    images_cpu, rgb_provenance = _load_rgb_trunk(scene_root, trunk=trunk)
    canvas_hw = (int(images_cpu.shape[-2]), int(images_cpu.shape[-1]))
    masks_cpu, mask_provenance = _load_exclusion_masks(
        scene_root,
        trunk=trunk,
        canvas_hw=canvas_hw,
        source_sizes_wh=rgb_provenance["source_sizes_wh"],
    )
    lidar_cells, lidar_provenance = _load_lidar_valid_cells(scene_root, trunk=trunk)
    input_paths = [*rgb_provenance["paths"], *lidar_provenance["paths"]]
    if mask_provenance.get("available"):
        for values in mask_provenance["paths"].values():
            input_paths.extend(values)
    source_manifest = _stat_manifest(input_paths, root=data_root)

    images = images_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    del images_cpu
    aggregator = components["aggregator"]
    with torch.inference_mode(), _autocast_context(device, precision):
        outputs = aggregator(images)
    if not isinstance(outputs, tuple) or len(outputs) != 5:
        raise RuntimeError("Production Aggregator did not return its five-output contract")
    _aggregated_all, image_tokens_all, _dino_all, _image_feature, patch_start_idx = outputs
    patch_start_idx = int(patch_start_idx)
    if patch_start_idx != 5:
        raise RuntimeError(f"Expected production patch_start_idx=5, got {patch_start_idx}")
    patch_grid = (canvas_hw[0] // PATCH_SIZE, canvas_hw[1] // PATCH_SIZE)
    selected = _select_image_levels(
        image_tokens_all,
        patch_start_idx=patch_start_idx,
        patch_grid=patch_grid,
    )
    del outputs, _aggregated_all, image_tokens_all, _dino_all, _image_feature, images
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.inference_mode():
        direct_gs, direct_depth, direct_gs_conf = _run_fp32_heads(
            selected,
            depth_head=components["depth_head"],
            gs_head=components["gs_head"],
            patch_start_idx=patch_start_idx,
            image_hw=canvas_hw,
            depth_chunk=depth_chunk,
        )
    if int(direct_gs.shape[1]) != TRUNK_LENGTH:
        raise ValueError(f"Direct GS head did not produce {TRUNK_LENGTH} frames")

    windows: list[dict[str, Any]] = []
    for start in WINDOW_STARTS:
        end = start + WINDOW_LENGTH
        local_frames = list(range(start, end))
        global_frames = [trunk * TRUNK_LENGTH + frame for frame in local_frames]
        with torch.inference_mode():
            reconstructed = _roundtrip_window(
                selected,
                start=start,
                patch_start_idx=patch_start_idx,
                patch_grid=patch_grid,
                tokenizer=components["scene_tokenizer"],
                precision=precision,
                device=device,
            )
            recon_gs, recon_depth, recon_gs_conf = _run_fp32_heads(
                reconstructed,
                depth_head=components["depth_head"],
                gs_head=components["gs_head"],
                patch_start_idx=patch_start_idx,
                image_hw=canvas_hw,
                depth_chunk=depth_chunk,
            )
        direct_gs_window = direct_gs[:, start:end]
        direct_depth_window = direct_depth[:, start:end]
        masks_window = None
        if masks_cpu is not None:
            masks_window = {
                key: value[start:end].to(device=device, non_blocking=True)
                for key, value in masks_cpu.items()
            }
        reference = _reference_window(
            reference_case,
            scene=scene,
            trunk=trunk,
            start=start,
        )
        supports = _window_supports(
            direct_gs_window,
            recon_gs,
            direct_depth_window,
            recon_depth,
            local_frames=local_frames,
            global_frames=global_frames,
            lidar_valid_cells=lidar_cells[start:end],
            exclusion_masks=masks_window,
            direct_units_per_metre=reference["full_trunk_direct_units_per_metre"],
        )
        windows.append(
            {
                "start": start,
                "end_exclusive": end,
                "local_frames": local_frames,
                "global_frames": global_frames,
                "supports": supports,
                "reference_depth_crosscheck": _crosscheck_reference(supports, reference),
            }
        )
        del reconstructed, recon_gs, recon_depth, recon_gs_conf, supports, masks_window
        if device.type == "cuda":
            torch.cuda.empty_cache()

    case: dict[str, Any] = {
        "scene": scene_name,
        "trunk": int(trunk),
        "global_start": int(trunk * TRUNK_LENGTH),
        "global_end_inclusive": int(trunk * TRUNK_LENGTH + TRUNK_LENGTH - 1),
        "patch_start_idx": patch_start_idx,
        "patch_grid_hw": list(patch_grid),
        "input_provenance": {
            "rgb": rgb_provenance,
            "exclusion_masks": mask_provenance,
            "lidar": lidar_provenance,
            "source_stat_manifest": source_manifest,
        },
        "windows": windows,
        "elapsed_seconds": time.time() - started,
    }
    case["trunk_balanced"] = _case_summary(windows)
    del selected, direct_gs, direct_depth, direct_gs_conf
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return case


def _synthetic_fields(
    *,
    gs_ratio_axes: tuple[float, float, float],
    depth_ratio: float,
    frames: int = 2,
    height: int = 3,
    width: int = 4,
) -> tuple[list[dict[str, torch.Tensor]], list[torch.Tensor]]:
    direct_gs = torch.zeros((frames, height, width, 11), dtype=torch.float32)
    recon_gs = torch.zeros_like(direct_gs)
    direct_gs[..., 3] = 0.9
    recon_gs[..., 3] = 0.9
    direct_gs[..., 4:7] = 0.02
    recon_gs[..., 4:7] = 0.02 * torch.tensor(gs_ratio_axes)
    direct_depth = torch.full((frames, height, width, 1), 3.0)
    recon_depth = direct_depth * depth_ratio
    fields = [
        _paired_frame_fields(direct_gs[index], recon_gs[index], direct_depth[index], recon_depth[index])
        for index in range(frames)
    ]
    masks = [torch.ones((height, width), dtype=torch.bool) for _ in range(frames)]
    return fields, masks


def _run_cpu_synthetic_assertions() -> dict[str, Any]:
    fields, masks = _synthetic_fields(gs_ratio_axes=(2.0, 2.0, 2.0), depth_ratio=2.0)
    summary = _summarize_support(
        fields,
        masks,
        local_frames=[0, 1],
        global_frames=[10, 11],
        definition="synthetic uniform similarity",
    )["frame_balanced"]
    assert abs(float(summary["gs_ratio"]) - 2.0) < 2.0e-6
    assert abs(float(summary["depth_ratio"]) - 2.0) < 2.0e-6
    assert abs(float(summary["paired_gs_over_depth_ratio"]) - 1.0) < 2.0e-6
    assert abs(float(summary["anisotropy_log_rms_median_of_frame_medians"])) < 2.0e-6
    assert abs(float(summary["depth_log_ratio_mad_median_of_frames"])) < 2.0e-6
    assert abs(float(summary["depth_log_ratio_iqr_median_of_frames"])) < 2.0e-6

    anis_fields, anis_masks = _synthetic_fields(
        gs_ratio_axes=(2.0, 1.0, 0.5), depth_ratio=1.0
    )
    anis = _summarize_support(
        anis_fields,
        anis_masks,
        local_frames=[0, 1],
        global_frames=[0, 1],
        definition="synthetic unit-geometric-mean anisotropy",
    )["frame_balanced"]
    expected_anisotropy = math.sqrt(2.0 * math.log(2.0) ** 2 / 3.0)
    assert abs(float(anis["gs_ratio"]) - 1.0) < 2.0e-6
    assert abs(float(anis["paired_gs_over_depth_ratio"]) - 1.0) < 2.0e-6
    assert abs(
        float(anis["anisotropy_log_rms_median_of_frame_medians"])
        - expected_anisotropy
    ) < 2.0e-6

    # Equal frame weighting: one valid pixel at ratio 2 and twelve valid
    # pixels at ratio 8 must aggregate to exp(median(log 2, log 8)) = 4,
    # rather than being dominated by the denser second frame.
    sparse_fields, _ = _synthetic_fields(
        gs_ratio_axes=(2.0, 2.0, 2.0), depth_ratio=1.0, frames=1
    )
    dense_fields, _ = _synthetic_fields(
        gs_ratio_axes=(8.0, 8.0, 8.0), depth_ratio=1.0, frames=1
    )
    sparse_mask = torch.zeros((3, 4), dtype=torch.bool)
    sparse_mask[0, 0] = True
    dense_mask = torch.ones((3, 4), dtype=torch.bool)
    balanced = _summarize_support(
        [sparse_fields[0], dense_fields[0]],
        [sparse_mask, dense_mask],
        local_frames=[0, 1],
        global_frames=[0, 1],
        definition="synthetic unequal per-frame support counts",
    )["frame_balanced"]
    assert abs(float(balanced["gs_ratio"]) - 4.0) < 5.0e-6

    floor_direct = torch.zeros((1, 2, 11), dtype=torch.float32)
    floor_recon = torch.zeros_like(floor_direct)
    floor_direct[..., 3] = 0.9
    floor_recon[..., 3] = 0.9
    floor_direct[..., 4:7] = 0.02
    floor_recon[..., 4:7] = 0.02
    floor_direct[0, 0, 4] = 1.0e-6
    floor_depth = torch.ones((1, 2, 1), dtype=torch.float32)
    floor_fields = _paired_frame_fields(
        floor_direct, floor_recon, floor_depth, floor_depth
    )
    floor_row = _frame_row(
        floor_fields,
        torch.ones((1, 2), dtype=torch.bool),
        local_frame=0,
        global_frame=0,
    )
    assert floor_row is not None
    assert float(floor_row["direct_scale_at_floor_axis_fraction"]) > 0.0
    assert float(floor_row["strict_all_six_axes_above_threshold_fraction"]) < 1.0

    # A constant canvas must remain constant at arbitrary original-grid cell
    # centers, verifying the align_corners=False sampling convention.
    constant = torch.full((7, 9, 2), 3.25, dtype=torch.float32)
    valid = np.zeros((4, 5), dtype=bool)
    valid[0, 0] = True
    valid[2, 3] = True
    valid[3, 4] = True
    sampled = _sample_bhwc_at_lidar_cells(constant, valid)
    assert tuple(sampled.shape) == (3, 2)
    assert torch.allclose(sampled, torch.full_like(sampled, 3.25), atol=1.0e-6)

    identity_profile = _depth_profile_correction(
        torch.tensor((5.0, 20.0, 80.0)), c_at_20m=1.0, slope=0.0
    )
    assert torch.equal(identity_profile, torch.ones_like(identity_profile))

    return {
        "status": "passed",
        "uniform_similarity": {
            "gs_ratio": summary["gs_ratio"],
            "depth_ratio": summary["depth_ratio"],
            "paired_gs_over_depth_ratio": summary["paired_gs_over_depth_ratio"],
            "depth_log_ratio_mad": summary["depth_log_ratio_mad_median_of_frames"],
            "depth_log_ratio_iqr": summary["depth_log_ratio_iqr_median_of_frames"],
        },
        "anisotropy": {
            "measured_log_rms": anis["anisotropy_log_rms_median_of_frame_medians"],
            "expected_log_rms": expected_anisotropy,
        },
        "frame_balancing": {
            "one_pixel_ratio_2_vs_twelve_pixels_ratio_8": balanced["gs_ratio"],
            "expected": 4.0,
        },
        "scale_floor_and_strict_subset": "passed",
        "lidar_cell_center_sampling": "passed",
        "v2_audit": {
            "fixed_scene_split": {
                "calibration_count": len(D4_CALIBRATION_SCENES),
                "selection_count": len(D4_HOLDOUT_SCENES),
                "disjoint": not bool(set(D4_CALIBRATION_SCENES) & set(D4_HOLDOUT_SCENES)),
            },
            "depth_edges_m": list(DEPTH_BIN_EDGES_M),
            "identity_profile": "passed",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent same-pixel Gaussian/depth tokenizer round-trip gauge retest. "
            "Run inside the conda dggt environment."
        )
    )
    parser.add_argument("--scenes", nargs="+", default=["300-329"], help="IDs/ranges, e.g. 300-329")
    parser.add_argument("--trunks", nargs="+", default=["0", "1", "2"], help="29-frame trunk IDs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--result-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument("--depth-chunk", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DGGT_CHECKPOINT)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=DEFAULT_TOKENIZER_CHECKPOINT)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--d4-form-bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--paired-equivalence-bootstrap-samples",
        type=int,
        default=PAIRED_EQUIVALENCE_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--paired-equivalence-bootstrap-seed",
        type=int,
        default=PAIRED_EQUIVALENCE_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--cpu-synthetic-only",
        action="store_true",
        help="Run small CPU assertions only; do not load checkpoints, data, or CUDA",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cpu_synthetic_only:
        print(json.dumps(_run_cpu_synthetic_assertions(), indent=2, sort_keys=True))
        return 0

    if args.depth_chunk <= 0:
        raise ValueError("--depth-chunk must be positive")
    if args.d4_form_bootstrap_samples <= 0:
        raise ValueError("--d4-form-bootstrap-samples must be positive")
    if args.paired_equivalence_bootstrap_samples <= 0:
        raise ValueError("--paired-equivalence-bootstrap-samples must be positive")
    scenes = _parse_integer_specs(args.scenes, name="scenes")
    trunks = _parse_integer_specs(args.trunks, name="trunks")
    if any(trunk < 0 for trunk in trunks):
        raise ValueError("trunks must be non-negative")
    device = torch.device(args.device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"Only CUDA and CPU devices are supported, got {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    precision = args.precision
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("Real CPU execution requires --precision fp32")

    script_path = Path(__file__).resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    tokenizer_checkpoint_path = args.tokenizer_checkpoint.expanduser().resolve()
    reference_path = args.result_json.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    for required in (script_path, checkpoint_path, tokenizer_checkpoint_path, reference_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    print("[provenance] hashing script/checkpoints/reference JSON", flush=True)
    hashes = {
        "script_sha256": _sha256(script_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "tokenizer_checkpoint_sha256": _sha256(tokenizer_checkpoint_path),
        "reference_result_json_sha256": _sha256(reference_path),
    }
    output_path = _resolve_output_path(
        args.output,
        tokenizer_sha256=hashes["tokenizer_checkpoint_sha256"],
    )
    protected_paths = {script_path, checkpoint_path, tokenizer_checkpoint_path, reference_path}
    if output_path in protected_paths:
        raise ValueError(f"--output must not overwrite an input or this script: {output_path}")
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(reference_payload, Mapping):
        raise ValueError("Reference JSON root must be an object")
    reference_index = _reference_case_index(reference_payload)
    reference_provenance = _validate_reference_provenance(
        reference_payload,
        dggt_sha256=hashes["checkpoint_sha256"],
        tokenizer_sha256=hashes["tokenizer_checkpoint_sha256"],
    )
    requested_keys = [(scene, trunk) for scene in scenes for trunk in trunks]
    missing_reference_cases = [key for key in requested_keys if key not in reference_index]
    if missing_reference_cases:
        raise ValueError(f"Reference JSON is missing requested cases: {missing_reference_cases[:20]}")
    # Fail before loading large models if the strict round-trip window schema is
    # absent or incomplete.
    for scene, trunk in requested_keys:
        for start in WINDOW_STARTS:
            _reference_window(
                reference_index[(scene, trunk)],
                scene=scene,
                trunk=trunk,
                start=start,
            )

    print(
        "[model] strict-loading Aggregator + DepthHead + GaussianHead + "
        "JointSceneTokenizer",
        flush=True,
    )
    components, load_info = _load_components(
        checkpoint_path,
        tokenizer_checkpoint_path,
        device,
    )
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or "CPU"

    result: dict[str, Any] = {
        "schema": {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "strict": True,
        },
        "status": "running",
        "metadata": {
            "created_unix": time.time(),
            "script": str(script_path),
            **hashes,
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_status_for_script": _git_value(["status", "--short", "--", str(script_path)]),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": device_name,
            "aggregator_tokenizer_precision": precision,
            "production_heads_precision": "fp32 (autocast disabled, fp32 inputs and parameters)",
            "data_root": str(data_root),
            "checkpoint": str(checkpoint_path),
            "tokenizer_checkpoint": str(tokenizer_checkpoint_path),
            "reference_result_json": str(reference_path),
            "reference_provenance": reference_provenance,
            "requested_scenes": scenes,
            "requested_trunks": trunks,
            "depth_chunk": args.depth_chunk,
            "d4_fixed_split": {
                "calibration_scenes": list(D4_CALIBRATION_SCENES),
                "selection_scenes": list(D4_HOLDOUT_SCENES),
                "role_warning": (
                    "Calibration scenes alone fit the profile. Scenes 320-329 contribute to "
                    "the paired equivalence audit and diagnostic depth strata but never refit "
                    "the form or coefficients."
                ),
            },
            "paired_equivalence_bootstrap": {
                "samples": int(args.paired_equivalence_bootstrap_samples),
                "seed": int(args.paired_equivalence_bootstrap_seed),
                "unit": "scene",
            },
            "model_load": load_info,
        },
        "method": {
            "preprocessing": (
                "PIL RGB; resize width to 518; height=round((H*518/W)/14)*14; bicubic; "
                "uint8/255 float32; no dataset/test loader and no crop"
            ),
            "aggregator_scope": "one production Aggregator forward over each full 29-frame trunk",
            "direct_heads_scope": "one fp32 GSHead and DepthHead forward over all 29 direct token frames",
            "roundtrip_scope": (
                "five independent tokenizer encode/decode calls on local ten-frame slices at starts 0/5/10/14/19; "
                "decoded patch tokens reattach that window's original special tokens"
            ),
            "primary_pixel_equation": (
                "e_p=mean_axis(log(clamp(scale_recon,1e-5)/clamp(scale_direct,1e-5)))"
                "-log(depth_recon/depth_direct)"
            ),
            "primary_aggregation": "per-frame median(e_p), then median over frames, then exp",
            "uniform_similarity_expectation": "paired_gs_over_depth_ratio approximately 1",
            "primary_paired_practical_equivalence_gate": {
                "support": D4_PRIMARY_SUPPORT,
                "margin": list(PAIRED_EQUIVALENCE_MARGIN),
                "point_and_entire_scene_bootstrap_95_ci_must_be_inside": True,
                "bootstrap_unit": "scene",
                "bootstrap_samples": int(args.paired_equivalence_bootstrap_samples),
                "bootstrap_seed": int(args.paired_equivalence_bootstrap_seed),
                "c_gs_policy": "identity fixed at 1; image-quality metrics are outside this audit",
            },
            "inference_boundary": (
                "This paired linear-scale test is a necessary falsification condition only. "
                "Passing does not prove direct Gaussian covariance is calibrated to Waymo meters "
                "or that opacity/quaternion/compositing preserve a complete render similarity."
            ),
            "anisotropy": (
                "per-pixel RMS across axes after subtracting mean-axis log scale ratio; "
                "zero for a uniform three-axis scale change"
            ),
            "scale_floor": SCALE_FLOOR,
            "strict_scale_threshold": STRICT_SCALE_THRESHOLD,
            "opacity_thresholds": list(OPACITY_THRESHOLDS),
            "rms_radius_sensitivity": (
                "sqrt(mean_axis(scale^2)) recon/direct, paired against same-pixel depth ratio"
            ),
            "reference_json_role": (
                "strict same-cell depth cross-check plus one full-29-frame case.depth.scale_frame_balanced "
                "LiDAR scale for uncorrected reconstructed metric-depth bins; ten-frame scales "
                "are never used for binning"
            ),
            "window_overlap": _overlap_provenance(),
            "d4_depth_stratification": {
                "edges_m": list(DEPTH_BIN_EDGES_M),
                "fit_variable": _depth_profile_variable_contract(),
                "runtime_variable": _depth_profile_variable_contract(),
                "equation": "log c(d) = a + b * log(d/20m)",
                "identity_vs_fitted_gate": {
                    "minimum_fitted_scene_loso_rmse_improvement_fraction": (
                        D4_IDENTITY_MIN_CV_IMPROVEMENT_FRACTION
                    ),
                    "fallback": "identity",
                },
                "constant_vs_loglinear_gate": {
                    "minimum_abs_bin_spearman": D4_FORM_MIN_ABS_SPEARMAN,
                    "scene_bootstrap_slope_ci_excludes_zero": True,
                    "minimum_loso_rmse_improvement_fraction": D4_FORM_MIN_CV_IMPROVEMENT_FRACTION,
                    "bootstrap_samples": int(args.d4_form_bootstrap_samples),
                    "bootstrap_seed": D4_FORM_BOOTSTRAP_SEED,
                },
            },
        },
        "cases": [],
        "summary": None,
        "v2_audit": {
            "status": "running",
            "depth_profile": None,
            "primary_paired_practical_equivalence": None,
            "formal_audit_complete": False,
            "formal_audit_coverage": None,
            "c_gs_recommendation": {
                "form": "identity",
                "value": 1.0,
                "selection_rule": (
                    "fixed identity; the paired GS/depth practical-equivalence gate only "
                    "determines whether tokenizer v2 is acceptable"
                ),
            },
        },
    }

    total = len(requested_keys)
    for case_index, (scene, trunk) in enumerate(requested_keys, start=1):
        print(f"[case {case_index}/{total}] scene={scene:03d} trunk={trunk}", flush=True)
        case = _run_case(
            scene=scene,
            trunk=trunk,
            data_root=data_root,
            components=components,
            device=device,
            precision=precision,
            depth_chunk=args.depth_chunk,
            reference_case=reference_index[(scene, trunk)],
        )
        result["cases"].append(case)
        result["summary"] = _global_summary(result["cases"])
        result["status"] = "running"
        _atomic_write_json(output_path, result)

    result["summary"] = _global_summary(result["cases"])
    depth_profile = _build_depth_profile_summary(
        result["cases"],
        bootstrap_samples=int(args.d4_form_bootstrap_samples),
    )
    profile_decision = depth_profile["form_decision"]
    result["v2_audit"]["depth_profile"] = depth_profile

    paired_equivalence = _paired_practical_equivalence_gate(
        result["cases"],
        bootstrap_samples=int(args.paired_equivalence_bootstrap_samples),
        bootstrap_seed=int(args.paired_equivalence_bootstrap_seed),
    )
    result["summary"]["primary_paired_practical_equivalence"] = paired_equivalence
    result["v2_audit"]["primary_paired_practical_equivalence"] = paired_equivalence

    required_scene_set = set(D4_CALIBRATION_SCENES) | set(D4_HOLDOUT_SCENES)
    paired_scene_set = {int(scene) for scene in paired_equivalence["scene_order"]}
    complete_profile_fit = (
        bool(profile_decision.get("available", False))
        and int(profile_decision.get("scene_count", 0)) == len(D4_CALIBRATION_SCENES)
    )
    formal_audit_complete = (
        set(scenes) == required_scene_set
        and set(trunks) == {0, 1, 2}
        and paired_scene_set == required_scene_set
        and complete_profile_fit
        and bool(paired_equivalence["available"])
    )
    result["v2_audit"]["formal_audit_complete"] = formal_audit_complete
    result["v2_audit"]["formal_audit_coverage"] = {
        "required_scenes": [*D4_CALIBRATION_SCENES, *D4_HOLDOUT_SCENES],
        "required_trunks": [0, 1, 2],
        "paired_scene_order": paired_equivalence["scene_order"],
        "complete_calibration_profile_fit": complete_profile_fit,
    }
    result["v2_audit"]["c_gs_recommendation"].update(
        {
            "paired_gate_passed": bool(paired_equivalence["passed"]),
            "applicable_to_accepted_tokenizer": (
                formal_audit_complete and bool(paired_equivalence["passed"])
            ),
        }
    )
    result["v2_audit"]["status"] = (
        "complete_formal_v2_audit"
        if formal_audit_complete
        else "complete_diagnostic_v2_audit"
    )
    result["status"] = "complete"
    result["metadata"]["completed_unix"] = time.time()
    result["metadata"]["elapsed_seconds"] = (
        result["metadata"]["completed_unix"] - result["metadata"]["created_unix"]
    )
    _atomic_write_json(output_path, result)
    print(f"[done] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
