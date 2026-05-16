"""Validation flow-cache precompute (Mode-A schema, decoupled edits).

For each ``data/final_info_validation.json`` entry, run VGGT Pass-1 +
decoupled localization ONCE, then derive 5 variant caches:

* ``combined`` -- delete {deletion, replacement-src, reposition-src} +
  insert {insertion, replacement, reposition} (all 4 edits in one scene).
* ``delete`` / ``add`` / ``replace`` / ``move`` -- single-edit-type caches.

Output (schema-identical to Mode A v6, ``mode_kind="mode_a"``; FLAT layout so
the unmodified Mode-A toolchain works -- ``index = entry_index*5 + variant_ord``,
``variant_ord`` = combined0/delete1/add2/replace3/move4):

    {out_root}/validation/{index:06d}.pt
    {out_root}/validation/_errors.jsonl   # skipped entries (e.g. missing asset)

The heavy work (VGGT forward, clean_state, Sim3 alignment, pose-refine inside
``localize_validation_objects``) runs once per entry; only the genuinely
variant-dependent steps (asset pass, phase1 pack, splat+blend) run per variant.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/precompute_flow_features_validation.py \
        --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
        --asset_root /data/disk2/lyy_dataset/test_transfer/objects_ply_transformed \
        --out_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from datasets.waymo_validation_edit_dataset import (
    DEFAULT_ALL_OBJECT_INFO_INSERTION_ROOT,
    DEFAULT_ALL_OBJECT_INFO_ROOT,
    DEFAULT_ASSET_ROOT,
    NUM_SLOTS,
    MissingAssetError,
    WaymoValidationEditDataset,
)
from dggt.models.asset_pass import AssetAggregatorPass
from dggt.models.gaussian_scene_editor import GaussianSceneEditor
from dggt.utils.flow_cache_io import save_flow_cache
from dggt.utils.validation_edit_localize import (
    VARIANT_SLOTS,
    localize_validation_objects,
    subset_localized_for_variant,
)
from tools.precompute_flow_features import (
    DEFAULT_PROCESSED_ROOT,
    AsyncFlowCacheWriter,
    _as_homogeneous_viewmats,
    _assert_no_nan_tensor,
    _build_object_meta,
    _build_training_predictions_from_cache_payload,
    _cleanup_cuda,
    _compute_and_pack_pass2_splatted_tok_low,
    _load_vggt,
    _pack_mode_a_asset_pass_result,
    _pack_pass1_tokens,
    _pack_phase1_localized,
    _replace_asset_luts_with_cached_dtype,
)

ALL_VARIANTS = ["combined", "delete", "add", "replace", "move"]
VARIANT_ORD = {v: i for i, v in enumerate(ALL_VARIANTS)}


def variant_cache_index(entry_index: int, variant: str) -> int:
    """Flat, unique cache index: entry_index*5 + variant_ord.

    Matches ``build_flow_validation_manifest`` and keeps the on-disk layout
    flat (``{out_root}/{split}/{index:06d}.pt``) so the unmodified Mode-A
    toolchain (verify_flow_cache_wysiwyg, WaymoFlowCacheDataset cache_root
    mode) works on validation caches without special-casing.
    """
    return int(entry_index) * len(ALL_VARIANTS) + VARIANT_ORD[variant]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--asset_root", default=DEFAULT_ASSET_ROOT)
    p.add_argument("--processed_root", default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--final_info_path", default="data/final_info_validation.json")
    p.add_argument("--all_object_info_root", default=DEFAULT_ALL_OBJECT_INFO_ROOT)
    p.add_argument(
        "--all_object_info_insertion_root",
        default=DEFAULT_ALL_OBJECT_INFO_INSERTION_ROOT,
    )
    p.add_argument("--split", default="validation")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS),
                   help="comma list subset of: " + ",".join(ALL_VARIANTS))
    p.add_argument("--start", type=int, default=None, help="entry index, inclusive")
    p.add_argument("--end", type=int, default=None, help="entry index, exclusive")
    p.add_argument("--force_overwrite", action="store_true")
    p.add_argument("--asset_batch_size", type=int, default=1)
    p.add_argument("--save_fit_metrics", action="store_true")
    p.add_argument("--max_pose_refine_yaw_deg", type=float, default=15.0)
    p.add_argument("--asset_yaw_correction_deg", type=float, default=180.0)
    p.add_argument("--save_compression", choices=["gzip", "none"], default="gzip")
    p.add_argument("--gzip_level", type=int, default=1)
    p.add_argument("--sync_save", action="store_true")
    p.add_argument("--max_save_threads", type=int, default=0)
    p.add_argument("--chunk_channels", type=int, default=64)
    return p


def _variant_sample(sample_cached: dict[str, Any], loc_v: list) -> dict[str, Any]:
    """Shallow-copy sample with editable_object_* set to the asset slots that
    actually have a localized object in ``loc_v``."""
    asset_slots = sorted(
        {int(o.slot_idx) for o in loc_v if int(o.asset_means_world.shape[0]) > 0}
    )
    sv = dict(sample_cached)
    eidx = torch.full((NUM_SLOTS,), -1, dtype=torch.long)
    for i, s in enumerate(asset_slots):
        eidx[i] = s
    sv["editable_object_indices"] = eidx
    sv["editable_object_count"] = torch.tensor(len(asset_slots), dtype=torch.long)
    return sv


def _assemble_payload(
    *,
    args,
    variant: str,
    sample_cached: dict[str, Any],
    sample_v: dict[str, Any],
    S: int,
    H_img: int,
    W_img: int,
    patch_start_idx: int,
    pass1_tokens: dict[str, Any],
    gs_map: torch.Tensor,
    depth: torch.Tensor,
    dynamic_conf: torch.Tensor,
    gs_conf: torch.Tensor,
    pose_enc: torch.Tensor,
    semantic_logits: torch.Tensor | None,
    images_u8: torch.Tensor,
    sky_mask: torch.Tensor,
    dynamic_mask: torch.Tensor | None,
    cameras_dggt: dict[str, torch.Tensor],
    alignment,
    asset_pass_payload: dict[int, dict[str, Any]],
    phase1_localized_payload: dict[str, Any],
    pass2_payload: dict[str, Any],
) -> dict[str, Any]:
    ve = sample_cached["validation_edit"]
    return {
        "schema_version": 6,
        "mode_kind": "mode_a",
        "meta": {
            "manifest_index": int(ve["entry_index"]),
            "dataset_index": int(sample_cached.get("sample_index", 0)),
            "scene_name": sample_cached.get("scene_name", ""),
            "clip_name": sample_cached.get("clip_name", ""),
            "variant": variant,
            "validation_entry_index": int(ve["entry_index"]),
            "frame_indices_scene": sample_cached["frame_indices"].cpu().to(torch.long),
            "cam_ids": sample_cached["cam_ids"].cpu().to(torch.long),
            "timestamps": sample_cached["timestamps"].cpu().to(torch.float32),
            "image_size_model_hw": (H_img, W_img),
            "patch_grid": (H_img // 14, W_img // 14),
            "patch_start_idx": patch_start_idx,
            "raw_image_size_hw": sample_cached["raw_image_size_hw"].cpu().to(torch.long),
            "num_frames": int(S),
            "asset_meta": sample_cached.get("asset_meta", {}),
            "asset_pass_space": "dggt_fitted",
            "editor_config": {
                "use_pose_refine": True,
                "max_pose_refine_yaw_deg": float(args.max_pose_refine_yaw_deg),
                "asset_yaw_correction_deg": float(args.asset_yaw_correction_deg),
                "pose_policy": "waymo_dggt_corner_projection_refine_v1",
            },
            "validation_edit": {
                "clip_name": ve["clip_name"],
                "segment": ve["segment"],
                "scene_dir": ve["scene_dir"],
                "clip_index": ve["clip_index"],
                "slot_role": ve["slot_role"],
                "slot_raw_id": ve["slot_raw_id"],
                "action_for_reposition": ve["action_for_reposition"],
                "variant_slots": {
                    k: list(v) for k, v in VARIANT_SLOTS[variant].items()
                },
            },
        },
        "raw": {
            "images_u8": images_u8,
            "sky_mask": sky_mask,
            "dynamic_mask": dynamic_mask,
        },
        "object_meta": _build_object_meta(sample_v),
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
        "mode_b": None,
        "phase1_localized": phase1_localized_payload,
        "pass2_splatted_tok_low": pass2_payload,
    }


def precompute_one_entry(
    sample: dict[str, Any],
    model,
    asset_pass: AssetAggregatorPass,
    editor: GaussianSceneEditor,
    args,
    device: torch.device,
    variants: list[str],
) -> dict[str, dict[str, Any]]:
    """Return {variant: payload} for one validation entry (one VGGT pass)."""
    images_clean = sample["images_clean"].unsqueeze(0).to(device)
    S = int(images_clean.shape[1])
    H_img, W_img = int(images_clean.shape[-2]), int(images_clean.shape[-1])

    with torch.no_grad():
        predictions = model(images_clean, return_tokens=True)

    patch_start_idx = int(predictions.get("patch_start_idx", 5))
    pass1_tokens = _pack_pass1_tokens(predictions, patch_start_idx)

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
    dynamic_mask = (
        dynamic_mask_src.to(torch.bool).cpu() if dynamic_mask_src is not None else None
    )
    sample_cached = dict(sample)
    sample_cached["images_clean"] = images_u8.to(torch.float32).div(255.0)
    sample_cached["images"] = sample_cached["images_clean"]
    sample_cached["sky_mask"] = sky_mask
    sample_cached["masks"] = sky_mask
    sample_cached["dynamic_mask"] = dynamic_mask

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

    # One decoupled localization (delete pose-refine + asset placement).
    localized = localize_validation_objects(
        sample_cached,
        clean_state,
        alignment,
        editor=editor,
        asset_yaw_correction_deg=float(args.asset_yaw_correction_deg),
    )
    _cleanup_cuda(device)

    out: dict[str, dict[str, Any]] = {}
    for variant in variants:
        loc_v = subset_localized_for_variant(localized, variant)
        sample_v = _variant_sample(sample_cached, loc_v)

        with torch.no_grad():
            asset_result = asset_pass(
                sample_v,
                selected_object_slots=None,
                alignment=alignment,
                asset_cache={},
                occlusion_test=True,
                aggregator_batch_size=int(args.asset_batch_size),
                localized_objects=loc_v,
                cameras_dggt=cameras_dggt,
                render_space="dggt_fitted",
                collect_fit_metrics=bool(args.save_fit_metrics),
            )
        asset_pass_payload = _pack_mode_a_asset_pass_result(asset_result)
        asset_result = _replace_asset_luts_with_cached_dtype(
            asset_result, asset_pass_payload, dtype=torch.float32
        )
        _cleanup_cuda(device)

        phase1_localized_payload = _pack_phase1_localized(loc_v, clean_state)
        _cleanup_cuda(device)
        pass2_payload = _compute_and_pack_pass2_splatted_tok_low(
            sample=sample_v,
            predictions=predictions,
            asset_pass_result=asset_result,
            cameras_dggt=cameras_dggt,
            clean_state=clean_state,
            localized_objects=loc_v,
            patch_grid=(H_img // 14, W_img // 14),
            H_img=H_img,
            W_img=W_img,
            chunk_channels=int(args.chunk_channels),
            device=device,
        )
        _cleanup_cuda(device)

        out[variant] = _assemble_payload(
            args=args,
            variant=variant,
            sample_cached=sample_cached,
            sample_v=sample_v,
            S=S,
            H_img=H_img,
            W_img=W_img,
            patch_start_idx=patch_start_idx,
            pass1_tokens=pass1_tokens,
            gs_map=gs_map,
            depth=depth,
            dynamic_conf=dynamic_conf,
            gs_conf=gs_conf,
            pose_enc=pose_enc,
            semantic_logits=semantic_logits,
            images_u8=images_u8,
            sky_mask=sky_mask,
            dynamic_mask=dynamic_mask,
            cameras_dggt=cameras_dggt,
            alignment=alignment,
            asset_pass_payload=asset_pass_payload,
            phase1_localized_payload=phase1_localized_payload,
            pass2_payload=pass2_payload,
        )
        del asset_result
        _cleanup_cuda(device)

    del images_clean, clean_state, predictions
    _cleanup_cuda(device)
    return out


def main() -> None:
    args = build_argparser().parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ALL_VARIANTS:
            raise ValueError(f"unknown variant {v!r}; choose from {ALL_VARIANTS}")

    out_root = Path(args.out_root)
    split_root = out_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    errors_path = split_root / "_errors.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} variants={variants}")
    model = _load_vggt(args.ckpt_path, device)
    asset_pass = AssetAggregatorPass(model.aggregator).to(device)
    editor = GaussianSceneEditor(
        use_pose_refine=True,
        max_pose_refine_yaw_deg=args.max_pose_refine_yaw_deg,
        asset_yaw_correction_deg=args.asset_yaw_correction_deg,
    )

    dataset = WaymoValidationEditDataset(
        final_info_path=args.final_info_path,
        processed_root=args.processed_root,
        all_object_info_root=args.all_object_info_root,
        all_object_info_insertion_root=args.all_object_info_insertion_root,
        asset_root=args.asset_root,
        split=args.split,
    )
    total = len(dataset)
    start = int(args.start) if args.start is not None else 0
    end = int(args.end) if args.end is not None else total
    start = max(0, start)
    end = min(total, end)
    print(f"[init] entries={total} range=[{start},{end}) out_root={split_root}")

    save_writer = None if args.sync_save else AsyncFlowCacheWriter(
        compression=args.save_compression,
        gzip_level=int(args.gzip_level),
        max_threads=int(args.max_save_threads),
    )
    done = saved = skipped = err = 0
    progress = tqdm(range(start, end), desc="precompute validation", unit="entry",
                    dynamic_ncols=True)

    def _log_error(idx: int, clip_name: str, reason: str, **extra: Any) -> None:
        nonlocal skipped
        skipped += 1
        rec = {"index": int(idx), "clip_name": clip_name, "reason": reason}
        rec.update(extra)
        with open(errors_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        progress.write(f"[skip] entry={idx:03d} {clip_name}: {reason} {extra}")

    try:
        for idx in progress:
            entry = dataset.entries[idx]
            clip_name = str(entry.get("clip_name", ""))
            entry_index = int(entry.get("index", idx))
            todo = [
                v for v in variants
                if args.force_overwrite
                or not (split_root / f"{variant_cache_index(entry_index, v):06d}.pt").is_file()
            ]
            if not todo:
                continue

            try:
                sample = dataset[idx]
            except MissingAssetError as e:
                _log_error(idx, clip_name, "missing_asset",
                           missing_asset_ids=[aid for aid, _ in e.missing])
                continue
            except Exception as e:  # noqa: BLE001
                _log_error(idx, clip_name, f"dataset_error:{type(e).__name__}:{e}")
                continue

            t0 = time.time()
            try:
                payloads = precompute_one_entry(
                    sample, model, asset_pass, editor, args, device, todo
                )
            except MissingAssetError as e:
                _log_error(idx, clip_name, "missing_asset",
                           missing_asset_ids=[aid for aid, _ in e.missing])
                _cleanup_cuda(device)
                continue
            except Exception as e:  # noqa: BLE001
                err += 1
                progress.write(f"[err ] entry={idx:03d} {clip_name}: {type(e).__name__}: {e}")
                _cleanup_cuda(device)
                continue

            for variant, payload in payloads.items():
                out_path = split_root / f"{variant_cache_index(entry_index, variant):06d}.pt"
                if save_writer is None:
                    save_flow_cache(
                        payload, out_path,
                        compression=args.save_compression,
                        gzip_level=int(args.gzip_level),
                    )
                    saved += 1
                else:
                    save_writer.submit(payload, out_path, {
                        "idx": int(idx), "manifest_index": int(entry_index),
                        "clip_start": 0, "scene_name": str(sample.get("scene_name", "")),
                        "extra": f" {variant}", "compute_sec": time.time() - t0,
                    })
            done += 1
            if save_writer is not None:
                completed, errs = save_writer.drain_completed()
                saved += len(completed)
                err += len(errs)
            progress.set_postfix(done=done, saved=saved, skip=skipped, err=err,
                                 dt=f"{time.time()-t0:.0f}s")
            del sample, payloads
            _cleanup_cuda(device)
    finally:
        if save_writer is not None:
            completed, errs = save_writer.close()
            saved += len(completed)
            err += len(errs)

    print(f"[done] entries_done={done} pts_saved={saved} skipped={skipped} err={err}")
    if errors_path.is_file():
        print(f"[done] skipped entries logged to {errors_path}")


if __name__ == "__main__":
    main()
