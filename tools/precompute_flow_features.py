"""Offline precompute of FlowDGGT features for the diffusion training cache.

Two edit modes share the SAME on-disk schema; only a few fields differ:

* `--edit_mode mode_a`: runs Phase 1 (`GaussianSceneEditor`) + Phase 4
  (`AssetAggregatorPass`). Caches `asset_pass` (per-object asset Gaussians + LUTs
  + I/A asset renders).

* `--edit_mode mode_b`: runs Phase 1 + the Mode-B pseudo-deletion planner
  (`dggt.utils.mode_b_planner`). Caches `mode_b` (the planner output: imagined
  objects + per-Gaussian delete masks). NO asset pass — diffusion will hallucinate
  the new content.

One `.pt` per clip at `{out_root}/{split}/{manifest_index:06d}.pt`. Use
SEPARATE `--out_root` per mode (e.g. `/data/flow_cache_mode_a` and
`/data/flow_cache_mode_b`) so manifests stay clean. The downstream
`tools/build_flow_train_manifest.py` then merges them into a single training
manifest the diffusion dataloader consumes.

Usage:
    python tools/precompute_flow_features.py \
        --edit_mode mode_a \
        --ckpt_path /data/.../model_latest_waymo.pt \
        --processed_root ... --transfer_root ... --raw_root ... \
        --split training --out_root /data/flow_cache_mode_a --views 1

    python tools/precompute_flow_features.py \
        --edit_mode mode_b \
        --ckpt_path /data/.../model_latest_waymo.pt \
        --processed_root ... --transfer_root ... --raw_root ... \
        --split training --out_root /data/flow_cache_mode_b --views 1
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.waymo_edit_dataset import (
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_TRANSFER_ROOT,
    WaymoEditDataset,
)
from dggt.models.asset_pass import AssetAggregatorPass
from dggt.models.gaussian_scene_editor import GaussianSceneEditor
from dggt.utils.feature_quant import QuantizedTokens, dequantize_tokens, quantize_tokens
from dggt.utils.flow_cache_io import save_flow_cache
from dggt.utils.gaussian_edit import parse_object_slots
from dggt.utils.tokens import select_patch_pyramid


DEFAULT_LEVELS = (4, 11, 17, 23)
CLIP_LENGTH = 29


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--edit_mode", choices=["mode_a", "mode_b"], default="mode_a")
    p.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--transfer_root", default=DEFAULT_TRANSFER_ROOT)
    p.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--manifest_path", default=None,
                   help="Override; default resolves from edit_mode + processed_root.")
    p.add_argument("--candidate_path", default=None,
                   help="Override; default resolves from edit_mode + processed_root.")
    p.add_argument("--asset_root", default=None)
    p.add_argument("--split", default="training")
    p.add_argument("--out_root", required=True)
    p.add_argument("--views", type=int, default=1)
    p.add_argument("--start", type=int, default=None, help="Start dataset index, inclusive. Defaults to 0.")
    p.add_argument("--end", type=int, default=None, help="End dataset index, exclusive. Defaults to dataset length.")
    p.add_argument("--start_clip_idx", type=int, default=0)
    p.add_argument("--end_clip_idx", type=int, default=-1, help="-1 for all")
    p.add_argument("--force_overwrite", action="store_true")
    p.add_argument("--dataset_mode", type=int, default=2, help="2 = deterministic")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--save_compression", choices=["gzip", "none"], default="gzip",
                   help="Cache file compression. gzip keeps .pt paths but wraps torch serialization.")
    p.add_argument("--gzip_level", type=int, default=1,
                   help="gzip compression level for --save_compression gzip (0-9).")
    p.add_argument("--sync_save", action="store_true",
                   help="Save cache on the main thread instead of the default async background writer.")
    p.add_argument("--max_save_threads", type=int, default=0,
                   help="Max concurrent async save threads. 0 means one thread per clip with no limit.")
    p.add_argument("--skip_asset_pass", action="store_true",
                   help="mode_a only: skip Phase 4 (debug).")
    p.add_argument("--asset_batch_size", type=int, default=1,
                   help="mode_a only: number of editable object renders per asset aggregator forward.")
    p.add_argument("--save_fit_metrics", action="store_true",
                   help="mode_a only: store per-frame asset bbox fit diagnostics in the cache.")
    p.add_argument("--max_pose_refine_yaw_deg", type=float, default=15.0,
                   help="Clamp the shared per-track Mode-A yaw update around the Waymo heading.")
    p.add_argument("--asset_yaw_correction_deg", type=float, default=180.0,
                   help="Fixed local yaw mapping from canonical asset Gaussians into the Waymo 3D-box frame.")
    # Mode B planner knobs (mirroring inference_mode_b.py defaults)
    p.add_argument(
        "--planner_seed",
        type=int,
        default=0,
        help="Base Mode-B planner seed. Per-clip seed is base + 1009 * dataset_index.",
    )
    p.add_argument("--num_objects_target", type=int, default=None)
    p.add_argument("--max_trials_per_object", type=int, default=80)
    p.add_argument("--min_visible_frames", type=int, default=15)
    p.add_argument("--max_semantic_overlap_px", type=int, default=0)
    p.add_argument("--min_projected_transfer_size_px", type=float, default=128.0)
    p.add_argument("--max_projected_area_ratio", type=float, default=0.12)
    p.add_argument("--max_projected_width_ratio", type=float, default=0.45)
    p.add_argument("--max_projected_height_ratio", type=float, default=0.52)
    p.add_argument("--min_projected_top_y_ratio", type=float, default=0.20)
    p.add_argument("--min_projected_center_y_ratio", type=float, default=0.35)
    p.add_argument("--min_projected_bottom_y_ratio", type=float, default=0.50)
    p.add_argument("--max_projected_bottom_y_ratio", type=float, default=1.0)
    p.add_argument("--min_ground_support_ratio", type=float, default=0.18)
    p.add_argument("--require_first_frame_visible", action="store_true")
    p.add_argument("--fast_camera_step_ratio", type=float, default=0.018)
    p.add_argument("--slow_camera_step_ratio", type=float, default=0.006)
    p.add_argument("--allow_empty_plan", action="store_true",
                   help="mode_b: write cache even when planner accepts 0 imagined objects.")
    return p


def _resolve_default_manifest(processed_root: str, split: str, views: int, edit_mode: str) -> Path:
    base = Path(processed_root) / "waymo_edit_cache" / "manifests" / split
    return base / f"{split}_{edit_mode}_views{views}.jsonl"


def _resolve_default_candidate(processed_root: str, split: str, edit_mode: str) -> Path:
    base = Path(processed_root) / "waymo_edit_cache" / "metadata" / split
    return base / f"{edit_mode}_candidates.jsonl"


def _load_vggt(ckpt_path: str, device: torch.device):
    from dggt.models.vggt import VGGT

    model = VGGT().to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    ignore = ("track_head.", "sky_model.", "scene_tokenizer.")
    real_missing = [k for k in missing if not k.startswith(ignore)]
    if real_missing:
        raise RuntimeError(f"Missing checkpoint keys: {real_missing[:10]}")
    model.eval()
    return model


def _cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _assert_no_nan_tensor(name: str, tensor: torch.Tensor | None, *, require_finite: bool = False) -> None:
    if tensor is None or not tensor.is_floating_point():
        return
    if torch.isnan(tensor).any().item():
        raise RuntimeError(f"{name} contains NaN")
    if require_finite and torch.isinf(tensor).any().item():
        raise RuntimeError(f"{name} contains Inf")


def _assert_valid_mode_b_payload(payload: dict[str, Any]) -> None:
    mode_b = payload.get("mode_b")
    if mode_b is None:
        raise RuntimeError("Mode B payload is missing")
    if int(mode_b.get("num_imagined_objects", 0)) <= 0:
        raise RuntimeError(
            f"Mode B planner produced no imagined objects: "
            f"reason={mode_b.get('rejection_reason', '')!r} metrics={mode_b.get('metrics', {})}"
        )
    delete_mask = mode_b.get("delete_mask")
    delete_mask_per_frame = mode_b.get("delete_mask_per_frame")
    if not torch.is_tensor(delete_mask) or delete_mask.numel() == 0 or int(delete_mask.sum().item()) == 0:
        raise RuntimeError("Mode B delete_mask is empty")
    if (
        not torch.is_tensor(delete_mask_per_frame)
        or delete_mask_per_frame.numel() == 0
        or int(delete_mask_per_frame.sum().item()) == 0
    ):
        raise RuntimeError("Mode B delete_mask_per_frame is empty")


class AsyncFlowCacheWriter:
    """Async writer that starts one save thread per submitted payload."""

    def __init__(self, *, compression: str, gzip_level: int, max_threads: int = 0) -> None:
        self.compression = compression
        self.gzip_level = int(gzip_level)
        self.semaphore = threading.Semaphore(int(max_threads)) if int(max_threads) > 0 else None
        self.threads: list[threading.Thread] = []
        self.completed: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def submit(self, payload: dict[str, Any], path: Path, meta: dict[str, Any]) -> None:
        if self.semaphore is not None:
            self.semaphore.acquire()
        thread = threading.Thread(
            target=self._save_one,
            args=(payload, path, meta),
            name=f"flow-cache-writer-{int(meta.get('idx', len(self.threads))):05d}",
            daemon=True,
        )
        with self.lock:
            self.threads.append(thread)
        thread.start()

    def drain_completed(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.lock:
            completed = self.completed
            errors = self.errors
            self.completed = []
            self.errors = []
        return completed, errors

    def close(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.lock:
            threads = list(self.threads)
        for thread in threads:
            thread.join()
        return self.drain_completed()

    def _save_one(self, payload: dict[str, Any], path: Path, meta: dict[str, Any]) -> None:
        t0 = time.time()
        try:
            save_flow_cache(
                payload,
                path,
                compression=self.compression,
                gzip_level=self.gzip_level,
            )
            item = {
                **meta,
                "path": str(path),
                "save_sec": time.time() - t0,
                "size_mb": os.path.getsize(path) / 1e6,
            }
            with self.lock:
                self.completed.append(item)
        except Exception as e:
            with self.lock:
                self.errors.append({**meta, "path": str(path), "error": str(e)})
        finally:
            del payload
            gc.collect()
            if self.semaphore is not None:
                self.semaphore.release()


def _assemble_4level_lut(
    image_tokens_list: list[torch.Tensor],
    levels: tuple[int, ...],
    patch_start_idx: int,
) -> torch.Tensor:
    """Select 4 pyramid levels and stack to `[S, P, 4, C]`."""
    patches = select_patch_pyramid(image_tokens_list, levels, patch_start_idx)
    # Stack on CPU so pass1 packing does not add a transient multi-GB GPU copy
    # on top of the full VGGT token pyramid.
    arr = torch.stack([p.squeeze(0).detach().cpu() for p in patches], dim=2)   # [S, P, L, C]
    return arr.contiguous()


def _gather_special_tokens(
    image_tokens_list: list[torch.Tensor],
    levels: tuple[int, ...],
    patch_start_idx: int,
) -> torch.Tensor:
    """[S, L, patch_start_idx, C] — camera + register tokens needed by DPT heads."""
    specials = []
    for lvl in levels:
        tok = image_tokens_list[lvl]
        if tok.dim() != 4 or tok.shape[0] != 1:
            raise ValueError(f"expected [1, S, P, C] per level, got {tuple(tok.shape)}")
        specials.append(tok[0, :, :patch_start_idx, :].detach().cpu())
    return torch.stack(specials, dim=1).contiguous()


def _build_training_predictions_from_cache_payload(
    *,
    pass1_tokens: dict[str, Any],
    gs_map: torch.Tensor,
    depth: torch.Tensor,
    dynamic_conf: torch.Tensor,
    gs_conf: torch.Tensor,
    pose_enc: torch.Tensor,
    semantic_logits: torch.Tensor | None,
    patch_start_idx: int,
    lut_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Mirror ``WaymoFlowCacheDataset._build_predictions`` for the full clip.

    Phase1 masks and pass2 splatted features must be generated from exactly the
    same fp16 / int8 cache representation that training later reads.  Otherwise
    small depth quantization changes can alter the clean Gaussian count and make
    cached masks impossible to align with ``clean_state``.
    """
    lut_full = dequantize_tokens(
        QuantizedTokens(
            data=pass1_tokens["F_g_lut_scene_int8"],
            scale=pass1_tokens["F_g_lut_scene_scale"],
            layout="NPLC",
        ),
        dtype=lut_dtype,
    )
    image_tokens_levels = [
        lut_full[:, :, l, :].unsqueeze(0).contiguous()
        for l in range(int(lut_full.shape[2]))
    ]
    return {
        "pose_enc": pose_enc.unsqueeze(0),
        "depth": depth.unsqueeze(0),
        "gs_map": gs_map.unsqueeze(0),
        "dynamic_conf": dynamic_conf.unsqueeze(0),
        "gs_conf": gs_conf.unsqueeze(0),
        "semantic_logits": None if semantic_logits is None else semantic_logits.unsqueeze(0),
        "image_tokens_levels": image_tokens_levels,
        "patch_start_idx": int(patch_start_idx),
    }


def _replace_asset_luts_with_cached_dtype(asset_result, asset_payload: dict[int, dict[str, Any]], *, dtype: torch.dtype):
    """Make pass2 use the same quantized/dequantized asset LUTs as training."""
    F_g_lut_asset: dict[int, list[torch.Tensor]] = {}
    for k in asset_result.object_keys:
        k = int(k)
        entry = asset_payload[k]
        F_full = dequantize_tokens(
            QuantizedTokens(
                data=entry["F_g_lut_asset_int8"],
                scale=entry["F_g_lut_asset_scale"],
                layout="NPLC",
            ),
            dtype=dtype,
        )
        F_g_lut_asset[k] = [
            F_full[:, :, l, :].unsqueeze(0).contiguous()
            for l in range(int(F_full.shape[2]))
        ]
    asset_result.F_g_lut_asset = F_g_lut_asset
    return asset_result


def _as_homogeneous_viewmats(world_to_camera: torch.Tensor) -> torch.Tensor:
    """Normalize world-to-camera matrices to [S, 4, 4] for flow visualization."""
    if world_to_camera.dim() != 3:
        raise ValueError(f"Expected world_to_camera [S,3,4] or [S,4,4], got {tuple(world_to_camera.shape)}")
    if world_to_camera.shape[-2:] == (4, 4):
        return world_to_camera
    if world_to_camera.shape[-2:] != (3, 4):
        raise ValueError(f"Expected world_to_camera [S,3,4] or [S,4,4], got {tuple(world_to_camera.shape)}")
    bottom = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        dtype=world_to_camera.dtype,
        device=world_to_camera.device,
    ).view(1, 1, 4)
    return torch.cat([world_to_camera, bottom.expand(world_to_camera.shape[0], -1, -1)], dim=1)


def _sample_manifest_index(sample: dict[str, Any], fallback: int) -> int:
    value = sample.get("manifest_index", fallback)
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _planner_seed_for_index(base_seed: int, sample_index: int) -> int:
    return int(base_seed) + 1009 * int(sample_index)


def _pack_pass1_tokens(
    predictions: dict[str, torch.Tensor],
    patch_start_idx: int,
) -> dict[str, Any]:
    """Quantize the 4-level scene LUT + carry aggregated/dino LUTs as int8."""
    image_tokens_list = predictions["image_tokens_list"]
    lut_scene = _assemble_4level_lut(image_tokens_list, DEFAULT_LEVELS, patch_start_idx).cpu()
    special_image = _gather_special_tokens(image_tokens_list, DEFAULT_LEVELS, patch_start_idx).cpu().to(torch.float16)
    F_scene_q = quantize_tokens(lut_scene, layout="NPLC").save_dict()
    del lut_scene

    out: dict[str, Any] = {
        "F_g_lut_scene_int8": F_scene_q["data"],
        "F_g_lut_scene_scale": F_scene_q["scale"],
        "image_tokens_special": special_image,
    }

    aggregated_tokens_list = predictions.get("aggregated_tokens_list")
    if aggregated_tokens_list is not None:
        agg_lut = _assemble_4level_lut(aggregated_tokens_list, DEFAULT_LEVELS, patch_start_idx).cpu()
        agg_q = quantize_tokens(agg_lut, layout="NPLC").save_dict()
        del agg_lut
        out["aggregated_tokens_patch_int8"] = agg_q["data"]
        out["aggregated_tokens_patch_scale"] = agg_q["scale"]
        out["aggregated_tokens_special"] = _gather_special_tokens(
            aggregated_tokens_list, DEFAULT_LEVELS, patch_start_idx
        ).cpu().to(torch.float16)
    else:
        out["aggregated_tokens_patch_int8"] = None
        out["aggregated_tokens_patch_scale"] = None
        out["aggregated_tokens_special"] = None

    dino_tokens_list = predictions.get("dino_tokens_list") or predictions.get("dino_token_list")
    if dino_tokens_list is not None:
        dino_lut = _assemble_4level_lut(dino_tokens_list, DEFAULT_LEVELS, patch_start_idx).cpu()
        dino_q = quantize_tokens(dino_lut, layout="NPLC").save_dict()
        del dino_lut
        out["dino_tokens_patch_int8"] = dino_q["data"]
        out["dino_tokens_patch_scale"] = dino_q["scale"]
        out["dino_tokens_special"] = _gather_special_tokens(
            dino_tokens_list, DEFAULT_LEVELS, patch_start_idx
        ).cpu().to(torch.float16)
    else:
        out["dino_tokens_patch_int8"] = None
        out["dino_tokens_patch_scale"] = None
        out["dino_tokens_special"] = None
    return out


def _build_object_meta(sample: dict[str, Any]) -> dict[str, Any]:
    object_keys = [
        "object_asset_ids",
        "object_scene_raw_ids",
        "object_asset_paths",
        "object_valid_mask",
        "object_scene_match_scores",
        "object_max_speed_mps",
        "object_mean_speed_mps",
        "object_is_moving_track",
        "object_speed_mps_selected",
        "object_is_moving_frame_selected",
        "object_obj_to_world_selected",
        "object_box_size_selected",
        "object_box_corners_world_selected",
        "object_track_valid_mask_selected",
        "object_asset_image_valid_mask_selected",
        "object_asset_image_paths_selected",
        "object_bbox_present_mask_selected",
        "object_bbox_editable_mask_selected",
        "object_bbox_model_selected",
        "object_front_bbox_present_mask_selected",
        "object_front_bbox_editable_mask_selected",
        "object_front_bbox_model_selected",
        "editable_object_indices",
        "editable_object_count",
        "protected_object_indices",
        "protected_object_count",
        "protected_object_boxes_by_frame",
    ]
    out: dict[str, Any] = {}
    for k in object_keys:
        if k in sample:
            v = sample[k]
            out[k] = v.cpu() if torch.is_tensor(v) else v
    return out


def _run_mode_b_planner(
    sample: dict[str, Any],
    record: dict[str, Any] | None,
    clean_state,
    alignment,
    args,
    device: torch.device,
) -> dict[str, Any]:
    """Run ModeBPlanner + apply_mode_b. Returns the cache-ready `mode_b` block."""
    from dggt.utils.mode_b_planner import ModeBPlanner, apply_mode_b

    # Convert existing-object metadata into DGGT-coord centers/rotations the
    # planner uses for collision tests. Mirror inference_mode_b.py helpers.
    existing_objects = _collect_existing_objects_dggt(sample, alignment)
    if record is not None:
        from inference_mode_b import _convert_waymo_existing_objects_to_dggt
        existing_objects.extend(
            _convert_waymo_existing_objects_to_dggt(record.get("existing_objects", []), alignment)
        )
        existing_objects.extend(record.get("existing_objects_dggt", []))

    planner = ModeBPlanner(
        min_visible_frames=int(args.min_visible_frames),
        max_semantic_overlap_px=int(args.max_semantic_overlap_px),
        max_trials_per_object=int(args.max_trials_per_object),
        min_projected_transfer_size_px=float(args.min_projected_transfer_size_px),
        max_projected_area_ratio=float(args.max_projected_area_ratio),
        max_projected_width_ratio=float(args.max_projected_width_ratio),
        max_projected_height_ratio=float(args.max_projected_height_ratio),
        min_projected_top_y_ratio=float(args.min_projected_top_y_ratio),
        min_projected_center_y_ratio=float(args.min_projected_center_y_ratio),
        min_projected_bottom_y_ratio=float(args.min_projected_bottom_y_ratio),
        max_projected_bottom_y_ratio=float(args.max_projected_bottom_y_ratio),
        min_ground_support_ratio=float(args.min_ground_support_ratio),
        require_first_frame_visible=bool(args.require_first_frame_visible),
        fast_camera_step_ratio=float(args.fast_camera_step_ratio),
        slow_camera_step_ratio=float(args.slow_camera_step_ratio),
        rng_seed=_planner_seed_for_index(int(args.planner_seed), int(sample.get("sample_index", 0))),
    )
    plan = planner.plan(
        clean_state,
        existing_objects=existing_objects,
        num_objects_target=args.num_objects_target,
        views=int(args.views),
        scene_name=str(sample.get("scene_name", "")),
        clip_name=str(sample.get("clip_name", "")),
        clip_index=int(
            sample["clip_index"].item() if torch.is_tensor(sample.get("clip_index", 0)) else sample.get("clip_index", 0)
        ),
    )
    if plan.num_imagined_objects == 0 and not args.allow_empty_plan:
        raise RuntimeError(
            f"[mode_b] planner accepted 0 imagined objects for clip "
            f"{sample.get('clip_name', '?')}. Pass --allow_empty_plan to keep going "
            f"or tune planner thresholds (min_visible_frames, max_semantic_overlap_px, ...)."
        )
    deletion = apply_mode_b(clean_state, plan)
    plan_dict = plan.to_dict()
    return {
        "imagined_objects": plan_dict["imagined_objects"],
        "rejection_reason": plan_dict["rejection_reason"],
        "metrics": plan_dict["metrics"],
        "eligible": bool(plan_dict["eligible"]),
        "rng_seed": int(plan_dict["rng_seed"]),
        "num_imagined_objects": int(plan_dict["num_imagined_objects"]),
        "delete_mask": deletion.delete_mask.detach().cpu().bool(),
        "delete_mask_per_frame": deletion.delete_mask_per_frame.detach().cpu().bool(),
        "delete_core_indices": deletion.delete_core_indices.detach().cpu().to(torch.int32),
        "delete_shell_indices": deletion.delete_shell_indices.detach().cpu().to(torch.int32),
    }


def _collect_existing_objects_dggt(sample: dict[str, Any], alignment) -> list[dict[str, Any]]:
    """Slot-level existing-object DGGT centers/rotations (subset of inference_mode_b.py)."""
    if "object_valid_mask" not in sample or "object_track_valid_mask_selected" not in sample:
        return []
    object_valid = sample["object_valid_mask"].detach().cpu().bool()
    if object_valid.numel() == 0:
        return []
    track_valid = sample["object_track_valid_mask_selected"].detach().cpu().bool()
    obj_to_world = sample["object_obj_to_world_selected"].detach().cpu().float()
    box_size = sample["object_box_size_selected"].detach().cpu().float()
    out = []
    for slot_idx in range(int(object_valid.shape[0])):
        if not bool(object_valid[slot_idx].item()):
            continue
        present = track_valid[slot_idx]
        if not bool(present.any().item()):
            continue
        centers, rotations, sizes = [], [], []
        for frame_idx in range(int(present.shape[0])):
            if bool(present[frame_idx].item()):
                T = obj_to_world[slot_idx, frame_idx]
                centers.append(alignment.apply_points(T[:3, 3].view(1, 3))[0])
                rotations.append(alignment.rotation @ T[:3, :3])
                sizes.append(box_size[slot_idx, frame_idx] * float(alignment.scale))
            else:
                centers.append(torch.zeros(3, dtype=torch.float32))
                rotations.append(torch.eye(3, dtype=torch.float32))
                sizes.append(torch.zeros(3, dtype=torch.float32))
        first_valid = int(torch.nonzero(present, as_tuple=False).flatten()[0].item())
        out.append({
            "slot": int(slot_idx),
            "scene_raw_object_id": str(sample["object_scene_raw_ids"][slot_idx])
                if "object_scene_raw_ids" in sample else str(slot_idx),
            "center_dggt_per_frame": torch.stack(centers, dim=0).tolist(),
            "rotation_dggt_per_frame": torch.stack(rotations, dim=0).tolist(),
            "size_dggt": sizes[first_valid].tolist(),
            "present_mask": present.tolist(),
        })
    return out


def _pack_mode_a_asset_pass_result(
    asset_result,
) -> dict[int, dict[str, Any]]:
    """Quantize a Phase-4 AssetPassResult for the offline cache payload."""
    payload: dict[int, dict[str, Any]] = {}
    for k in asset_result.object_keys:
        k = int(k)
        I_k = asset_result.I_asset[k][0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
        A_k = asset_result.A_asset[k][0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
        F_k = torch.stack(asset_result.F_g_lut_asset[k], dim=2).squeeze(0)  # [S, L, P, C]
        F_k = F_k.permute(0, 2, 1, 3).contiguous()                          # [S, P, L, C]
        F_k_q = quantize_tokens(F_k, layout="NPLC").save_dict()
        del F_k
        if asset_result.G_asset_dggt is None or k not in asset_result.G_asset_dggt:
            raise RuntimeError(
                f"Mode-A DGGT-fitted asset result missing G_asset_dggt for object {k}"
            )
        entry = {
            "I_asset": I_k,
            "A_asset": A_k,
            "F_g_lut_asset_int8": F_k_q["data"].cpu(),
            "F_g_lut_asset_scale": F_k_q["scale"].cpu(),
            "G_asset_dggt_per_frame": [
                {kk: vv.cpu() for kk, vv in g.items()}
                for g in asset_result.G_asset_dggt[k]
            ],
            "ptr_patch_idx": [p.patch_idx.cpu() for p in asset_result.ptr_asset[k]],
            "ptr_visible_mask": [p.visible_mask.cpu() for p in asset_result.ptr_asset[k]],
            "ptr_view_n": [p.view_n.cpu() for p in asset_result.ptr_asset[k]],
        }
        if asset_result.fit_metrics is not None:
            entry["fit_metrics"] = asset_result.fit_metrics.get(k)
        payload[k] = entry
    return payload


def _run_mode_a_asset_pass(
    sample: dict[str, Any],
    asset_pass: AssetAggregatorPass,
    editor: GaussianSceneEditor,
    clean_state,
    alignment,
    cameras_dggt: dict[str, torch.Tensor],
    args,
) -> tuple[Any, list[Any]]:
    """Run Phase 4 on all editable objects × all S frames in fitted DGGT space.

    Returns (asset_result, localized_objects). The latter is the
    ``editor.localize`` output we just paid pose-refinement for, which the
    caller should persist into the cache so training time can skip it.
    """
    with torch.no_grad():
        object_slots = parse_object_slots(sample, "all")
        asset_cache: dict[str, dict[str, torch.Tensor]] = {}
        localized_objects = editor.localize(
            sample,
            clean_state,
            alignment,
            object_slots,
            asset_cache=asset_cache,
            load_asset=True,
        )
        asset_result = asset_pass(
            sample,
            selected_object_slots=None,  # None → derive from editable_object_indices
            alignment=alignment,
            asset_cache=asset_cache,
            occlusion_test=True,
            aggregator_batch_size=int(getattr(args, "asset_batch_size", 1)),
            localized_objects=localized_objects,
            cameras_dggt=cameras_dggt,
            render_space="dggt_fitted",
            collect_fit_metrics=bool(getattr(args, "save_fit_metrics", False)),
        )
        editable_count = int(sample.get("editable_object_count", torch.tensor(0)).item())
        if editable_count > 0 and len(asset_result.object_keys) == 0:
            raise RuntimeError(
                "Mode-A DGGT-fitted asset pass produced no objects. "
                "This usually means localization failed; inspect bbox/depth/semantic masks."
            )
    return asset_result, localized_objects


# ---------------------------------------------------------------------- #
# Schema v6 helpers — phase1_localized + pass2_splatted_tok_low           #
# ---------------------------------------------------------------------- #


def _pack_phase1_localized(
    localized_objects: list,
    clean_state: Any,
) -> dict[str, Any]:
    """Pack the minimal phase-1 outputs needed to skip ``editor.localize`` at train time.

    The on-disk format avoids per-entry index lists because those would be
    indices into the FULL 29-frame Gaussian set; subsetting frames at train
    time requires non-trivial remapping.  Instead we materialize the
    OR'd ``delete_mask`` and ``shell_mask`` directly (both [N_g_full] bool)
    and a per-frame CSR-style offset tensor so ``WaymoFlowCacheDataset`` can
    cheaply rebuild the subset masks via ``torch.cat([delete[s:e] for f])``.

    Layout:
        slot_idx, frame_idx, source_front_index : int32 [N_loc]
            Used by ``build_phase1_asset_coverage`` (only requires these
            three fields per LocalizedFrameObject).
        delete_mask, shell_mask : bool [N_g_full]
            ``delete_mask[i]`` is True iff Gaussian ``i`` (in
            ``clean_state.means``) is deleted (core ∪ shell).  ``shell_mask``
            tracks shell membership separately for downstream consumers.
        frame_gauss_offsets : int64 [N_frames + 1]
            CSR: Gaussians for original frame ``f`` are at indices
            ``[frame_gauss_offsets[f], frame_gauss_offsets[f+1])`` in
            ``delete_mask`` / ``shell_mask`` / ``clean_state.means``.
    """
    from dggt.utils.gaussian_edit import apply_mode_a

    n = len(localized_objects)
    slot = torch.zeros(n, dtype=torch.int32)
    frame = torch.zeros(n, dtype=torch.int32)
    sf = torch.zeros(n, dtype=torch.int32)
    for i, item in enumerate(localized_objects):
        slot[i] = int(item.slot_idx)
        frame[i] = int(item.frame_idx)
        sf[i] = int(item.source_front_index)

    # Materialize the apply_mode_a output so we don't need to redo it at
    # training time.  This is cheap (~0.15s); the heavy work was localize().
    edit_state = apply_mode_a(clean_state, localized_objects)
    delete_mask = edit_state.delete_mask.detach().cpu().to(torch.bool).contiguous()
    shell_mask = edit_state.shell_mask.detach().cpu().to(torch.bool).contiguous()

    # Per-frame Gaussian counts derived from clean_state.source_frame_ids
    # (one entry per Gaussian, value = original frame index).  Build CSR
    # offsets via bincount.
    sfi = clean_state.source_frame_ids.detach().cpu().to(torch.long).contiguous()
    n_frames_total = int(sfi.max().item()) + 1 if sfi.numel() > 0 else 0
    counts = torch.bincount(sfi, minlength=n_frames_total)
    frame_gauss_offsets = torch.zeros(int(counts.numel()) + 1, dtype=torch.int64)
    frame_gauss_offsets[1:] = torch.cumsum(counts, dim=0)
    if int(frame_gauss_offsets[-1].item()) != int(delete_mask.numel()):
        raise RuntimeError(
            f"frame_gauss_offsets[-1]={int(frame_gauss_offsets[-1].item())} != "
            f"N_g_full={int(delete_mask.numel())}; build_clean_scene_state layout "
            "is not contiguous-by-frame as assumed."
        )
    return {
        "schema_version": 2,
        "slot_idx": slot,
        "frame_idx": frame,
        "source_front_index": sf,
        "delete_mask": delete_mask,
        "shell_mask": shell_mask,
        "frame_gauss_offsets": frame_gauss_offsets,
    }


def _compute_and_pack_pass2_splatted_tok_low(
    *,
    sample: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    asset_pass_result: Any,
    cameras_dggt: dict[str, torch.Tensor],
    clean_state: Any,
    localized_objects: list,
    patch_grid: tuple[int, int],
    H_img: int,
    W_img: int,
    chunk_channels: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run splatter + soft-mask + blend and pack the post-blend ``splatted_tok_low`` as int8.

    Mirrors ``FlowFeatureAssembler._forward_mode_a`` from after ``editor.localize``
    through the second blend (``_blend_asset_tokens``).  ``tokenizer.encode`` is
    deliberately *not* called here: the tokenizer is still being trained, so we
    cache its **input** rather than its output.  Training runs ``tokenizer.encode``
    online with the latest weights, keeping ``z_clean`` and ``z_splat`` in the
    same latent space.

    The output cache enables training to skip the splatter + blend pass (~33s
    per step); ``tokenizer.encode`` remains live (38ms — negligible).
    """
    from dggt.models.flow_feature_assembler import FlowFeatureAssembler
    from dggt.utils.gaussian_edit import apply_mode_a

    # Instantiate the assembler purely to reuse its splatter + soft_mask + blend
    # methods (no scaffold_packer training dependency here). Since we already
    # ran editor.localize once for the asset pass, we hand-feed clean_state +
    # localized_objects to apply_mode_a and skip the assembler's editor stages.
    H_splat = patch_grid[0] * 4
    W_splat = patch_grid[1] * 4
    assembler = FlowFeatureAssembler(
        scene_tokenizer=None,
        patch_grid=patch_grid,
        H_splat=H_splat,
        W_splat=W_splat,
        chunk_channels=int(chunk_channels),
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    assembler.eval()
    for p in assembler.parameters():
        p.requires_grad_(False)

    edit_state = apply_mode_a(clean_state, localized_objects)

    # Build pointers (mirrors FlowFeatureAssembler.forward).
    from dggt.models.gaussian_pointers import GaussianPointers
    from dggt.models.scene_pointers import build_scene_pointers, concat_pointers
    from dggt.models.flow_feature_assembler import _concat_gauss_dicts

    ptr_scene = build_scene_pointers(
        clean_state.source_image_ids,
        clean_state.source_y,
        clean_state.source_x,
        patch_size=int(H_img // patch_grid[0]),
        patch_grid=patch_grid,
    ).to(device)
    asset_gauss_chunks: list[dict[str, torch.Tensor]] = []
    ptr_chunks: list[GaussianPointers] = [ptr_scene]
    for obj_key in asset_pass_result.object_keys:
        obj_ptrs = asset_pass_result.ptr_asset[int(obj_key)]
        obj_gauss_frames = asset_pass_result.G_asset_dggt[int(obj_key)]
        per_frame_gauss_cat = [
            {k: v.to(device) for k, v in g.items()} for g in obj_gauss_frames
        ]
        flat_gauss = _concat_gauss_dicts(per_frame_gauss_cat, device=device)
        asset_gauss_chunks.append(flat_gauss)
        flat_ptr = GaussianPointers(
            src_kind=torch.cat([p.src_kind for p in obj_ptrs]),
            object_id=torch.cat([p.object_id for p in obj_ptrs]),
            view_n=torch.cat([p.view_n for p in obj_ptrs]),
            patch_idx=torch.cat([p.patch_idx for p in obj_ptrs]),
            visible_mask=torch.cat([p.visible_mask for p in obj_ptrs]),
        )
        ptr_chunks.append(flat_ptr)

    gauss_scene = {k: v.to(device) for k, v in edit_state.clean.items()}
    gauss_scene_kept = {k: v[~edit_state.delete_mask] for k, v in gauss_scene.items()}
    keep_mask = (~edit_state.delete_mask).to(device)
    ptr_scene_kept = GaussianPointers(
        src_kind=ptr_scene.src_kind[keep_mask],
        object_id=ptr_scene.object_id[keep_mask],
        view_n=ptr_scene.view_n[keep_mask],
        patch_idx=ptr_scene.patch_idx[keep_mask],
        visible_mask=ptr_scene.visible_mask[keep_mask],
    )
    ptr_chunks[0] = ptr_scene_kept
    pointers_all = concat_pointers(ptr_chunks)
    gaussians_all = _concat_gauss_dicts(
        [gauss_scene_kept] + asset_gauss_chunks, device=device
    )

    # F_g_lut_scene from cached image_tokens_levels.
    F_g_lut_scene = assembler._select_lut_scene(predictions)
    F_g_lut_asset = {
        int(k): [
            level.to(device) for level in asset_pass_result.F_g_lut_asset[int(k)]
        ]
        for k in asset_pass_result.object_keys
    }

    # The precompute packs cameras_dggt with shape [S, ...] for on-disk
    # storage, but FeatureSplatter / SoftMaskBuilder need [B, S, ...]. The
    # dataset reader adds the batch dim when loading from cache; here we add
    # it locally so the assembler-level helpers see the same layout.
    cameras_dggt_b = _ensure_batched_cameras(cameras_dggt)
    cameras_splat = FlowFeatureAssembler.scale_cameras_for_render(
        cameras_dggt_b,
        source_hw=(H_img, W_img),
        target_hw=(H_splat, W_splat),
    )
    cameras_splat = {k: v.to(device) for k, v in cameras_splat.items()}
    cameras_dggt_dev = {k: v.to(device) for k, v in cameras_dggt_b.items()}

    with torch.no_grad():
        # The splatter rasterizes ALL frames in one gsplat call. For full clips
        # (S=29) with millions of Gaussians the intermediate index tensors can
        # OOM (>40 GiB), so chunk frames into groups of `frame_chunk` and
        # concat. Per-frame splat is independent — chunking is exact.
        S_total = int(F_g_lut_scene[0].shape[1])
        frame_chunk = 2
        splatted_levels: list[list[torch.Tensor]] = [[] for _ in range(assembler.num_levels)]
        for fs in range(0, S_total, frame_chunk):
            fe = min(fs + frame_chunk, S_total)
            # Only target cameras are frame-chunked. The LUT S dimension is
            # indexed by GaussianPointers.view_n (source frame), so slicing it
            # would change the token assigned to gaussians outside this target
            # camera chunk.
            lut_asset_chunk = F_g_lut_asset if len(F_g_lut_asset) > 0 else None
            cameras_splat_chunk = {k: v[:, fs:fe].contiguous() for k, v in cameras_splat.items()}
            chunk_out = assembler.feature_splatter(
                gaussians_dggt=[gaussians_all],
                pointers=[pointers_all],
                lut_scene=F_g_lut_scene,
                lut_asset_dict=lut_asset_chunk,
                cameras_dggt=cameras_splat_chunk,
                H=H_splat,
                W=W_splat,
                pool_to=patch_grid,
            )
            for lvl_idx, lvl_tensor in enumerate(chunk_out):
                splatted_levels[lvl_idx].append(lvl_tensor)
        splatted_tok_low = [torch.cat(parts, dim=1) for parts in splatted_levels]

        # Soft mask coverage (needed for blend), per-object alpha for asset blend.
        G_kept_list = [gauss_scene_kept]
        G_deleted_list = [
            {k: v[edit_state.delete_mask].to(device) for k, v in gauss_scene.items()}
        ]
        G_asset_dict_list = [
            {
                int(k): _concat_gauss_dicts(
                    asset_pass_result.G_asset_dggt[int(k)], device=device
                )
                for k in asset_pass_result.object_keys
            }
        ]
        K_map, D_map, I_map, I_per_obj = assembler.soft_mask.render_coverage(
            G_kept_list,
            G_deleted_list,
            G_asset_dict_list,
            cameras_dggt=cameras_dggt_dev,
            H=H_img,
            W=W_img,
        )
        M_preserve, M_source, M_dest = assembler.soft_mask.pool_and_normalize(
            K_map, D_map, I_map, target_grid=patch_grid
        )
        splatted_tok_low = assembler._blend_preserve_tokens(
            clean_levels=F_g_lut_scene,
            splatted_levels=splatted_tok_low,
            M_preserve=M_preserve,
        )
        splatted_tok_low = assembler._blend_asset_tokens(
            splatted_levels=splatted_tok_low,
            F_g_lut_asset=F_g_lut_asset,
            I_map_per_obj=I_per_obj,
        )

    # splatted_tok_low: list[L × [1, S, P, C]]. Stack to [S, P, L, C] (NPLC) and
    # quantize per (frame, level) with the existing int8 utility.
    if len(splatted_tok_low) == 0:
        raise RuntimeError("splatted_tok_low is empty after blending")
    stacked = torch.stack(
        [lvl.detach().squeeze(0).cpu() for lvl in splatted_tok_low], dim=2
    ).contiguous()
    S_full, P_full, L_full, C_full = stacked.shape
    q = quantize_tokens(stacked.float(), layout="NPLC")
    return {
        "schema_version": 2,
        "splatted_tok_low_int8": q.data,   # [S, P, L, C]
        "splatted_tok_low_scale": q.scale, # [S, L] fp16
        "patch_grid": (int(patch_grid[0]), int(patch_grid[1])),
        "channels": int(C_full),
        "num_levels": int(L_full),
    }


def _compute_and_pack_pass2_splatted_tok_low_mode_b(
    *,
    sample: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    cameras_dggt: dict[str, torch.Tensor],
    clean_state: Any,
    mode_b_payload: dict[str, Any],
    patch_grid: tuple[int, int],
    H_img: int,
    W_img: int,
    chunk_channels: int,
    device: torch.device,
) -> dict[str, Any]:
    """Mode B variant of the splatter+blend caching helper (pre-tokenizer).

    Mirrors ``FlowFeatureAssembler._forward_mode_b`` from after ``delete_mask``
    is determined through ``_blend_preserve_tokens``.  ``tokenizer.encode`` runs
    online during training (see Mode A docstring for the rationale).  Returns
    the same packed dict layout as the Mode A helper so the dataset reader
    doesn't need a per-mode branch.

    Empty plans are still packed: v6 caches make ``pass2_splatted_tok_low``
    mandatory, and an all-kept Mode-B clip simply produces the no-deletion
    splatted/blended features.
    """
    from dggt.models.flow_feature_assembler import FlowFeatureAssembler
    from dggt.models.gaussian_pointers import GaussianPointers
    from dggt.models.scene_pointers import build_scene_pointers

    delete_mask_full = mode_b_payload.get("delete_mask")
    if delete_mask_full is None:
        raise RuntimeError("Mode B pass2 cache requires mode_b.delete_mask")
    delete_mask_full = delete_mask_full.to(torch.bool)

    H_splat = patch_grid[0] * 4
    W_splat = patch_grid[1] * 4
    assembler = FlowFeatureAssembler(
        scene_tokenizer=None,
        patch_grid=patch_grid,
        H_splat=H_splat,
        W_splat=W_splat,
        chunk_channels=int(chunk_channels),
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    assembler.eval()
    for p in assembler.parameters():
        p.requires_grad_(False)

    # Build kept / imagined Gaussian dicts from clean_state + delete_mask.
    n_g = int(clean_state.means.shape[0])
    if int(delete_mask_full.numel()) != n_g:
        raise RuntimeError(
            f"Mode B delete_mask length {int(delete_mask_full.numel())} does not "
            f"match clean_state Gaussians {n_g}; cannot write mandatory v6 pass2 cache."
        )
    clean_dict = {
        "means": clean_state.means.to(device),
        "colors": clean_state.colors.to(device),
        "opacities": clean_state.opacities.to(device).view(-1, 1)
            if clean_state.opacities.dim() == 1 else clean_state.opacities.to(device),
        "scales": clean_state.scales.to(device),
        "quats": clean_state.quats.to(device),
    }
    keep_mask_dev = (~delete_mask_full).to(device)
    del_mask_dev = delete_mask_full.to(device)
    gauss_kept = {k: v[keep_mask_dev] for k, v in clean_dict.items()}
    gauss_imagined = {k: v[del_mask_dev] for k, v in clean_dict.items()}

    ptr_scene = build_scene_pointers(
        clean_state.source_image_ids,
        clean_state.source_y,
        clean_state.source_x,
        patch_size=int(H_img // patch_grid[0]),
        patch_grid=patch_grid,
    ).to(device)
    ptr_scene_kept = GaussianPointers(
        src_kind=ptr_scene.src_kind[keep_mask_dev],
        object_id=ptr_scene.object_id[keep_mask_dev],
        view_n=ptr_scene.view_n[keep_mask_dev],
        patch_idx=ptr_scene.patch_idx[keep_mask_dev],
        visible_mask=ptr_scene.visible_mask[keep_mask_dev],
    )

    F_g_lut_scene = assembler._select_lut_scene(predictions)
    cameras_dggt_b = _ensure_batched_cameras(cameras_dggt)
    cameras_splat = FlowFeatureAssembler.scale_cameras_for_render(
        cameras_dggt_b, source_hw=(H_img, W_img), target_hw=(H_splat, W_splat),
    )
    cameras_splat = {k: v.to(device) for k, v in cameras_splat.items()}
    cameras_dggt_dev = {k: v.to(device) for k, v in cameras_dggt_b.items()}

    with torch.no_grad():
        # Frame-chunk the splatter to keep peak memory bounded on full clips.
        S_total = int(F_g_lut_scene[0].shape[1])
        frame_chunk = 2
        splatted_levels: list[list[torch.Tensor]] = [[] for _ in range(assembler.num_levels)]
        for fs in range(0, S_total, frame_chunk):
            fe = min(fs + frame_chunk, S_total)
            cameras_splat_chunk = {k: v[:, fs:fe].contiguous() for k, v in cameras_splat.items()}
            chunk_out = assembler.feature_splatter(
                gaussians_dggt=[gauss_kept],
                pointers=[ptr_scene_kept],
                lut_scene=F_g_lut_scene,
                lut_asset_dict=None,
                cameras_dggt=cameras_splat_chunk,
                H=H_splat,
                W=W_splat,
                pool_to=patch_grid,
            )
            for lvl_idx, lvl_tensor in enumerate(chunk_out):
                splatted_levels[lvl_idx].append(lvl_tensor)
        splatted_tok_low = [torch.cat(parts, dim=1) for parts in splatted_levels]

        # Soft mask coverage with imagined Gaussians as the I_map source.
        K_map, _D_map, I_map_proxy, _ = assembler.soft_mask.render_coverage(
            G_kept=[gauss_kept],
            G_deleted=[{}],
            G_asset_dggt_dict=[{0: gauss_imagined}] if gauss_imagined["means"].numel() > 0 else [{}],
            cameras_dggt=cameras_dggt_dev,
            H=H_img,
            W=W_img,
        )
        D_map = torch.zeros_like(I_map_proxy)
        I_map = I_map_proxy
        M_preserve, _M_source, _M_dest = assembler.soft_mask.pool_and_normalize(
            K_map, D_map, I_map, target_grid=patch_grid
        )
        splatted_tok_low = assembler._blend_preserve_tokens(
            clean_levels=F_g_lut_scene,
            splatted_levels=splatted_tok_low,
            M_preserve=M_preserve,
        )

    if len(splatted_tok_low) == 0:
        raise RuntimeError("splatted_tok_low is empty after blending (mode_b)")
    stacked = torch.stack(
        [lvl.detach().squeeze(0).cpu() for lvl in splatted_tok_low], dim=2
    ).contiguous()
    S_full, P_full, L_full, C_full = stacked.shape
    q = quantize_tokens(stacked.float(), layout="NPLC")
    return {
        "schema_version": 2,
        "splatted_tok_low_int8": q.data,
        "splatted_tok_low_scale": q.scale,
        "patch_grid": (int(patch_grid[0]), int(patch_grid[1])),
        "channels": int(C_full),
        "num_levels": int(L_full),
    }


def _ensure_batched_cameras(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """[S,4,4] → [1,S,4,4]; [S,3,3] → [1,S,3,3]; [B,S,...] left alone."""
    out: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.dim() == 3 and v.shape[-2:] in [(4, 4), (3, 3)]:
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def precompute_one_clip(
    sample: dict[str, Any],
    record: dict[str, Any] | None,
    model,
    asset_pass: AssetAggregatorPass | None,
    editor: GaussianSceneEditor,
    args,
    device: torch.device,
) -> dict[str, Any]:
    """Return the cache-ready payload for one 29-frame clip (mode A or B)."""
    images_clean = sample["images_clean"].unsqueeze(0).to(device)
    S = int(images_clean.shape[1])
    H_img, W_img = int(images_clean.shape[-2]), int(images_clean.shape[-1])

    with torch.no_grad():
        predictions = model(images_clean, return_tokens=True)

    patch_start_idx = int(predictions.get("patch_start_idx", 5))
    pass1_tokens = _pack_pass1_tokens(predictions, patch_start_idx)

    # Pass-1 dense outputs
    gs_map = predictions["gs_map"][0].to(torch.float16).cpu()
    depth = predictions["depth"][0].to(torch.float16).cpu()
    dynamic_conf = predictions["dynamic_conf"][0].to(torch.float16).cpu()
    gs_conf = predictions["gs_conf"][0].to(torch.float16).cpu()
    pose_enc = predictions["pose_enc"][0].to(torch.float16).cpu()
    semantic_logits = predictions.get("semantic_logits")
    if semantic_logits is not None:
        semantic_logits = semantic_logits[0].to(torch.float16).cpu()
    _assert_no_nan_tensor("pass1.gs_map", gs_map)
    _assert_no_nan_tensor("pass1.depth", depth)
    _assert_no_nan_tensor("pass1.dynamic_conf", dynamic_conf)
    _assert_no_nan_tensor("pass1.gs_conf", gs_conf)
    _assert_no_nan_tensor("pass1.pose_enc", pose_enc, require_finite=True)
    _assert_no_nan_tensor("pass1.semantic_logits", semantic_logits)

    images_u8 = images_clean[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
    sky_mask = sample["sky_mask"].to(torch.bool).cpu()
    dynamic_mask_src = sample.get("dynamic_mask", sample.get("masks"))
    dynamic_mask = (dynamic_mask_src.to(torch.bool).cpu() if dynamic_mask_src is not None else None)
    sample_cached = dict(sample)
    sample_cached["images_clean"] = images_u8.to(torch.float32).div(255.0)
    sample_cached["images"] = sample_cached["images_clean"]
    sample_cached["sky_mask"] = sky_mask
    sample_cached["masks"] = sky_mask
    sample_cached["dynamic_mask"] = dynamic_mask

    # From here on, run Phase1 and pass2 from the exact representation that the
    # training dataset will read back from disk.  This keeps cached masks,
    # Gaussian counts, and splatted features aligned with training-time
    # ``clean_state`` construction.
    del predictions
    predictions = _build_training_predictions_from_cache_payload(
        pass1_tokens=pass1_tokens,
        gs_map=gs_map,
        depth=depth,
        dynamic_conf=dynamic_conf,
        gs_conf=gs_conf,
        pose_enc=pose_enc,
        semantic_logits=semantic_logits,
        patch_start_idx=patch_start_idx,
        lut_dtype=torch.float32,
    )
    _cleanup_cuda(device)

    # Phase 1
    clean_state = editor.build_clean_bundle(sample_cached, predictions)
    alignment = editor.align(sample_cached, clean_state)
    _assert_no_nan_tensor("clean_state.world_to_camera", clean_state.world_to_camera, require_finite=True)
    _assert_no_nan_tensor("clean_state.intrinsics", clean_state.intrinsics, require_finite=True)
    _assert_no_nan_tensor("clean_state.camera_to_world", clean_state.camera_to_world, require_finite=True)
    cameras_dggt = {
        "viewmats": _as_homogeneous_viewmats(clean_state.world_to_camera).cpu().to(torch.float32),
        "Ks": clean_state.intrinsics.cpu().to(torch.float32),
        "camera_to_world": clean_state.camera_to_world.cpu().to(torch.float32),
    }
    asset_pass_payload: dict[int, dict[str, Any]] = {}
    asset_pass_result = None
    mode_b_payload: dict[str, Any] | None = None
    phase1_localized_payload: dict[str, Any] | None = None
    pass2_splatted_tok_low_payload: dict[str, Any] | None = None
    if args.edit_mode == "mode_a":
        if asset_pass is not None and not args.skip_asset_pass:
            asset_pass_result, localized_objects = _run_mode_a_asset_pass(
                sample_cached,
                asset_pass,
                editor,
                clean_state,
                alignment,
                cameras_dggt,
                args,
            )
            asset_pass_payload = _pack_mode_a_asset_pass_result(asset_pass_result)
            asset_pass_result = _replace_asset_luts_with_cached_dtype(
                asset_pass_result,
                asset_pass_payload,
                dtype=torch.float32,
            )
            _cleanup_cuda(device)

            # Schema v6: cache the editor's localized_objects + the
            # splatter→blend output so training can skip the 67s pose-refine
            # + 33s gsplat pass per step.  ``tokenizer.encode`` is NOT cached
            # because the tokenizer is still being trained — see
            # ``_compute_and_pack_pass2_splatted_tok_low`` docstring.
            phase1_localized_payload = _pack_phase1_localized(localized_objects, clean_state)
            # The asset_pass leaves significant transient allocations
            # (gsplat colors, intermediate features). Drop Python references
            # and synchronize before the splatter helper without calling
            # torch.cuda.empty_cache().
            _cleanup_cuda(device)
            pass2_splatted_tok_low_payload = _compute_and_pack_pass2_splatted_tok_low(
                sample=sample_cached,
                predictions=predictions,
                asset_pass_result=asset_pass_result,
                cameras_dggt=cameras_dggt,
                clean_state=clean_state,
                localized_objects=localized_objects,
                patch_grid=(H_img // 14, W_img // 14),
                H_img=H_img,
                W_img=W_img,
                chunk_channels=64,
                device=device,
            )
            _cleanup_cuda(device)
    elif args.edit_mode == "mode_b":
        mode_b_payload = _run_mode_b_planner(sample_cached, record, clean_state, alignment, args, device)
        # Schema v6: cache the splatter+blend output for Mode B too. Mode B
        # does NOT cache phase1_localized because it doesn't run editor.localize
        # (delete_mask comes from the planner, already saved).
        _cleanup_cuda(device)
        pass2_splatted_tok_low_payload = _compute_and_pack_pass2_splatted_tok_low_mode_b(
            sample=sample_cached,
            predictions=predictions,
            cameras_dggt=cameras_dggt,
            clean_state=clean_state,
            mode_b_payload=mode_b_payload,
            patch_grid=(H_img // 14, W_img // 14),
            H_img=H_img,
            W_img=W_img,
            chunk_channels=64,
            device=device,
        )
        _cleanup_cuda(device)
    else:
        raise ValueError(f"Unknown edit_mode: {args.edit_mode}")

    object_meta = _build_object_meta(sample)

    payload: dict[str, Any] = {
        "schema_version": 6,
        "mode_kind": args.edit_mode,
        "meta": {
            "manifest_index": _sample_manifest_index(sample, int(sample.get("sample_index", 0))),
            "dataset_index": int(sample.get("sample_index", 0)),
            "scene_name": sample.get("scene_name", ""),
            "clip_name": sample.get("clip_name", ""),
            "mode_b_source": sample.get("mode_b_source", ""),
            "mode_b_source_views1": sample.get("mode_b_source_views1", ""),
            "mode_b_source_views3": sample.get("mode_b_source_views3", ""),
            "mode_b_in_mode_a_views1": bool(sample.get("mode_b_in_mode_a_views1", False)),
            "mode_b_in_mode_a_views3": bool(sample.get("mode_b_in_mode_a_views3", False)),
            "frame_indices_scene": sample["frame_indices"].cpu().to(torch.long),
            "cam_ids": sample["cam_ids"].cpu().to(torch.long),
            "timestamps": sample["timestamps"].cpu().to(torch.float32),
            "image_size_model_hw": (H_img, W_img),
            "patch_grid": (H_img // 14, W_img // 14),
            "patch_start_idx": patch_start_idx,
            "raw_image_size_hw": sample["raw_image_size_hw"].cpu().to(torch.long),
            "num_frames": int(S),
            "asset_meta": sample.get("asset_meta", {}),
            "asset_pass_space": "none"
            if args.edit_mode != "mode_a"
            else (
                "skipped"
                if asset_pass_result is None
                else str(asset_pass_result.asset_pass_space)
            ),
            "editor_config": {
                "use_pose_refine": True,
                "max_pose_refine_yaw_deg": float(args.max_pose_refine_yaw_deg),
                "asset_yaw_correction_deg": float(args.asset_yaw_correction_deg),
                "pose_policy": "waymo_dggt_corner_projection_refine_v1",
            },
        },
        "raw": {
            "images_u8": images_u8,
            "sky_mask": sky_mask,
            "dynamic_mask": dynamic_mask,
        },
        "object_meta": object_meta,
        "pass1": {
            "cameras_dggt": cameras_dggt,
            "pose_enc": pose_enc,
            "gs_map": gs_map,
            "depth": depth,
            "dynamic_conf": dynamic_conf,
            "gs_conf": gs_conf,
            "semantic_logits": semantic_logits,
            **pass1_tokens,
        },
        "phase1_alignment": alignment.as_dict(),
        "asset_pass": asset_pass_payload,
        "mode_b": mode_b_payload,
        "phase1_localized": phase1_localized_payload,
        "pass2_splatted_tok_low": pass2_splatted_tok_low_payload,
    }
    if args.edit_mode == "mode_b" and not args.allow_empty_plan:
        _assert_valid_mode_b_payload(payload)
    del images_clean, clean_state, predictions, asset_pass_result
    _cleanup_cuda(device)
    return payload


def main() -> None:
    args = build_argparser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.manifest_path is None:
        args.manifest_path = str(_resolve_default_manifest(args.processed_root, args.split, args.views, args.edit_mode))
    if args.candidate_path is None:
        args.candidate_path = str(_resolve_default_candidate(args.processed_root, args.split, args.edit_mode))
    print(f"[init] edit_mode={args.edit_mode} manifest={args.manifest_path}")
    print(f"[init] out_root={out_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")
    model = _load_vggt(args.ckpt_path, device)
    asset_pass = (
        AssetAggregatorPass(model.aggregator).to(device) if args.edit_mode == "mode_a" else None
    )
    editor = GaussianSceneEditor(
        use_pose_refine=True,
        max_pose_refine_yaw_deg=args.max_pose_refine_yaw_deg,
        asset_yaw_correction_deg=args.asset_yaw_correction_deg,
    )

    dataset_kwargs = dict(
        processed_root=args.processed_root,
        transfer_root=args.transfer_root,
        raw_root=args.raw_root,
        split=args.split,
        manifest_path=args.manifest_path,
        candidate_path=args.candidate_path,
        views=args.views,
        mode=args.dataset_mode,
        sequence_length=CLIP_LENGTH,
        sample_window=CLIP_LENGTH,
    )
    if args.asset_root is not None:
        dataset_kwargs["asset_root"] = args.asset_root
    dataset = WaymoEditDataset(**dataset_kwargs)
    total = len(dataset)
    start_idx = int(args.start) if args.start is not None else int(args.start_clip_idx)
    end_idx = int(args.end) if args.end is not None else (total if args.end_clip_idx < 0 else int(args.end_clip_idx))
    start_idx = max(0, start_idx)
    end_idx = min(total, end_idx)
    if end_idx < start_idx:
        raise ValueError(f"Invalid range: start={start_idx} end={end_idx} for dataset length {total}")
    print(f"[init] dataset size={total}, range=[{start_idx},{end_idx})")

    done_count = 0
    saved_count = 0
    skip_count = 0
    err_count = 0
    scheduled_paths: set[Path] = set()
    save_writer = None if args.sync_save else AsyncFlowCacheWriter(
        compression=args.save_compression,
        gzip_level=int(args.gzip_level),
        max_threads=int(args.max_save_threads),
    )
    progress = tqdm(
        range(start_idx, end_idx),
        desc=f"precompute {args.edit_mode}/{args.split}",
        unit="clip",
        dynamic_ncols=True,
    )
    progress.set_postfix(done=done_count, saved=saved_count, skip=skip_count, err=err_count)

    def _drain_save_status() -> str | None:
        nonlocal saved_count, err_count
        if save_writer is None:
            return None
        last = None
        completed, errors = save_writer.drain_completed()
        for item in completed:
            saved_count += 1
            extra = item.get("extra", "")
            last = f"save {item['save_sec']:.1f}s/{item['size_mb']:.0f}MB{extra}"
        for item in errors:
            err_count += 1
            progress.write(
                f"[err ] save idx={int(item['idx']):05d} clip={int(item['clip_start']):04d} "
                f"scene={item['scene_name']}: {item['error']}"
            )
        return last

    try:
        for idx in progress:
            last_save = _drain_save_status()
            try:
                sample = dataset[idx]
            except Exception as e:
                err_count += 1
                progress.write(f"[err ] idx={idx:05d} dataset.__getitem__ failed: {e}")
                progress.set_postfix(done=done_count, saved=saved_count, skip=skip_count, err=err_count)
                continue
            record = dataset.samples[idx] if hasattr(dataset, "samples") and idx < len(dataset.samples) else None
            scene_name = str(sample.get("scene_name", f"scene{idx:06d}"))
            clip_start = int(sample["frame_indices"][0].item())
            manifest_index = _sample_manifest_index(sample, idx)
            out_dir = out_root / args.split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{manifest_index:06d}.pt"
            if out_path in scheduled_paths:
                skip_count += 1
                progress.write(
                    f"[skip] duplicate output path idx={idx:05d} manifest={manifest_index:06d} clip={clip_start:04d} "
                    f"scene={scene_name}: {out_path}"
                )
                del sample, record
                progress.set_postfix(done=done_count, saved=saved_count, skip=skip_count, err=err_count)
                continue
            if out_path.is_file() and not args.force_overwrite:
                skip_count += 1
                del sample, record
                progress.set_postfix(done=done_count, saved=saved_count, skip=skip_count, err=err_count)
                continue

            t0 = time.time()
            try:
                payload = precompute_one_clip(
                    sample,
                    record,
                    model,
                    asset_pass,
                    editor,
                    args,
                    device,
                )
            except Exception as e:
                err_count += 1
                progress.write(f"[err ] idx={idx:05d} clip={clip_start:04d} scene={scene_name}: {e}")
                del sample, record
                _cleanup_cuda(device)
                progress.set_postfix(done=done_count, saved=saved_count, skip=skip_count, err=err_count)
                continue
            extra = ""
            if args.edit_mode == "mode_b" and payload.get("mode_b") is not None:
                extra = f" imagined={payload['mode_b']['num_imagined_objects']}"
            elif args.edit_mode == "mode_a":
                extra = f" assets={len(payload['asset_pass'])}"
            compute_dt = time.time() - t0
            scheduled_paths.add(out_path)
            if save_writer is None:
                save_flow_cache(
                    payload,
                    out_path,
                    compression=args.save_compression,
                    gzip_level=int(args.gzip_level),
                )
                sz_mb = os.path.getsize(out_path) / 1e6
                saved_count += 1
                last = f"{compute_dt:.1f}s/{sz_mb:.0f}MB{extra}"
                del payload
            else:
                save_writer.submit(
                    payload,
                    out_path,
                    {
                        "idx": int(idx),
                        "manifest_index": int(manifest_index),
                        "clip_start": int(clip_start),
                        "scene_name": scene_name,
                        "extra": extra,
                        "compute_sec": compute_dt,
                    },
                )
                last = f"compute {compute_dt:.1f}s save-thread{extra}"
            del sample, record
            _cleanup_cuda(device)
            done_count += 1
            if last_save is not None:
                last = last_save
            progress.set_postfix(
                done=done_count,
                saved=saved_count,
                skip=skip_count,
                err=err_count,
                pending=max(0, done_count - saved_count),
                last=last,
            )
    finally:
        if save_writer is not None:
            completed, errors = save_writer.close()
            for item in completed:
                saved_count += 1
                extra = item.get("extra", "")
                progress.set_postfix(
                    done=done_count,
                    saved=saved_count,
                    skip=skip_count,
                    err=err_count,
                    pending=max(0, done_count - saved_count),
                    last=f"save {item['save_sec']:.1f}s/{item['size_mb']:.0f}MB{extra}",
                )
            for item in errors:
                err_count += 1
                progress.write(
                    f"[err ] save idx={int(item['idx']):05d} clip={int(item['clip_start']):04d} "
                    f"scene={item['scene_name']}: {item['error']}"
                )
            progress.set_postfix(
                done=done_count,
                saved=saved_count,
                skip=skip_count,
                err=err_count,
                pending=max(0, done_count - saved_count),
            )


if __name__ == "__main__":
    main()
