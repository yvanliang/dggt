"""Verify cache schema v6 produces correct output.

Three checks:

1. **Backward compatibility** — every v4-era field present in the OLD cache must
   appear in the NEW cache with bit-identical (tensors / values) content,
   modulo the bumped ``schema_version``.  The removed v5 tokenizer-output
   field (``pass2_z_splat``) and new v6 pre-tokenizer field
   (``pass2_splatted_tok_low``) are skipped.

2. **New v6 fields are consistent** — ``phase1_localized`` (Mode A) and
   ``pass2_splatted_tok_low`` (both modes) are present with the expected
   shape: ``splatted_tok_low_int8 [N,P,L,C]`` with ``L=4``, ``C=3072``.

3. **v6 cache semantics** — ``pass2_splatted_tok_low`` is a full-source-Gaussian
   cache: each target frame is splatted from the whole clip's Gaussian set.
   Therefore ``cache.index_select(subset)`` is not expected to match a live
   re-splat that first drops the non-subset source Gaussians.  We instead check:

   * full-clip quantized cache equals a full-clip live recompute;
   * subset cache slices dequantize to the expected shape and finite values;
   * cached-vs-live subset assembler outputs still match for fields that do not
     depend on the full-source-vs-subset-source splat choice.

Usage:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 conda run -n dggt --no-capture-output \\
        python -u tools/verify_v5_cache.py \\
            --old_cache /data/.../flow_cache_mode_a/training/000000.pt \\
            --new_cache /data/.../flow_cache_mode_a_v6/training/000000.pt \\
            --ckpt_path /data/.../model_latest_waymo.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.flow_cache_io import load_flow_cache


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def _diff_tensor(name: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> str | None:
    if a.shape != b.shape:
        return f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"dtype mismatch {a.dtype} vs {b.dtype}"
    af = a.detach().float().cpu()
    bf = b.detach().float().cpu()
    if torch.equal(a, b):
        return None
    if torch.allclose(af, bf, atol=atol, rtol=rtol):
        return None
    diff = (af - bf).abs()
    return (
        f"value mismatch (max_abs={float(diff.max()):.6g}, "
        f"mean_abs={float(diff.mean()):.6g})"
    )


def _diff_recursive(prefix: str, a, b, atol: float, rtol: float, errors: list[str]) -> None:
    if type(a) is not type(b):
        # Allow int↔int, float↔float subtypes
        if not (
            (isinstance(a, (int, bool)) and isinstance(b, (int, bool)))
            or (isinstance(a, (int, float)) and isinstance(b, (int, float)))
        ):
            errors.append(f"{prefix}: type mismatch {type(a).__name__} vs {type(b).__name__}")
            return
    if _is_tensor(a):
        msg = _diff_tensor(prefix, a, b, atol, rtol)
        if msg is not None:
            errors.append(f"{prefix}: {msg}")
        return
    if isinstance(a, dict):
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb:
            missing = ka - kb
            extra = kb - ka
            if missing:
                errors.append(f"{prefix}: missing keys in NEW: {sorted(missing)}")
            if extra:
                errors.append(f"{prefix}: extra keys in NEW: {sorted(extra)}")
        for k in sorted(ka & kb):
            _diff_recursive(f"{prefix}.{k}", a[k], b[k], atol, rtol, errors)
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            errors.append(f"{prefix}: length mismatch {len(a)} vs {len(b)}")
            return
        for i, (ai, bi) in enumerate(zip(a, b)):
            _diff_recursive(f"{prefix}[{i}]", ai, bi, atol, rtol, errors)
        return
    if a != b:
        errors.append(f"{prefix}: value mismatch {a!r} vs {b!r}")


def check_backward_compat(old, new, atol: float, rtol: float) -> list[str]:
    """For every key in OLD, the same key in NEW must match (tolerated)."""
    errors: list[str] = []
    if int(old.get("schema_version", 0)) >= int(new.get("schema_version", 0)):
        errors.append(
            f"schema_version did not advance: old={old.get('schema_version')} "
            f"new={new.get('schema_version')}"
        )
    skip_keys = {"schema_version", "pass2_z_splat", "pass2_splatted_tok_low"}
    # Compare every preserved key.
    v4_keys = set(old.keys())
    for key in sorted(v4_keys):
        if key in skip_keys:
            continue
        if key not in new:
            errors.append(f"NEW cache is missing v4 key: {key!r}")
            continue
        _diff_recursive(key, old[key], new[key], atol, rtol, errors)
    return errors


def check_new_fields_present(new) -> list[str]:
    errors: list[str] = []
    if int(new.get("schema_version", 0)) < 6:
        errors.append(f"schema_version must be >= 6, got {new.get('schema_version')}")
    mode_kind = str(new.get("mode_kind", ""))
    is_mode_a = mode_kind == "mode_a"
    is_mode_b = mode_kind == "mode_b"

    if is_mode_a:
        if "phase1_localized" not in new:
            errors.append("phase1_localized missing from new Mode A cache")
        elif not isinstance(new["phase1_localized"], dict):
            errors.append("phase1_localized is not a dict")
        else:
            p = new["phase1_localized"]
            for req in (
                "slot_idx", "frame_idx", "source_front_index",
                "delete_mask", "shell_mask", "frame_gauss_offsets",
            ):
                if req not in p:
                    errors.append(f"phase1_localized missing field: {req!r}")
            if "slot_idx" in p and "frame_idx" in p:
                n = int(p["slot_idx"].numel())
                if int(p["frame_idx"].numel()) != n:
                    errors.append("phase1_localized: slot_idx / frame_idx length mismatch")
                if int(p["source_front_index"].numel()) != n:
                    errors.append("phase1_localized: source_front_index length mismatch")
            if "delete_mask" in p and "shell_mask" in p:
                if p["delete_mask"].dtype != torch.bool:
                    errors.append(f"phase1_localized.delete_mask must be bool, got {p['delete_mask'].dtype}")
                if p["delete_mask"].numel() != p["shell_mask"].numel():
                    errors.append("phase1_localized: delete_mask / shell_mask length mismatch")
            if "frame_gauss_offsets" in p and "delete_mask" in p:
                if p["frame_gauss_offsets"][-1].item() != p["delete_mask"].numel():
                    errors.append(
                        f"phase1_localized: frame_gauss_offsets[-1]={int(p['frame_gauss_offsets'][-1].item())} "
                        f"!= delete_mask length {int(p['delete_mask'].numel())}"
                    )
    elif is_mode_b:
        # phase1_localized is N/A for Mode B (no editor.localize call).
        if new.get("phase1_localized") is not None:
            errors.append("Mode B cache should not carry phase1_localized")

    if "pass2_z_splat" in new and new["pass2_z_splat"] is not None:
        errors.append("v6 cache should not carry pass2_z_splat tokenizer outputs")

    if "pass2_splatted_tok_low" not in new:
        errors.append("pass2_splatted_tok_low missing from new cache")
    elif not isinstance(new["pass2_splatted_tok_low"], dict):
        errors.append("pass2_splatted_tok_low is not a dict")
    else:
        p = new["pass2_splatted_tok_low"]
        for req in ("splatted_tok_low_int8", "splatted_tok_low_scale", "channels", "num_levels", "patch_grid"):
            if req not in p:
                errors.append(f"pass2_splatted_tok_low missing field: {req!r}")
        tok = p.get("splatted_tok_low_int8")
        scale = p.get("splatted_tok_low_scale")
        if tok is not None:
            if tok.dtype != torch.int8:
                errors.append(f"pass2_splatted_tok_low.splatted_tok_low_int8 dtype must be int8, got {tok.dtype}")
            if tok.dim() != 4:
                errors.append(
                    f"pass2_splatted_tok_low_int8 must be 4-D [N,P,L,C], got shape {tuple(tok.shape)}"
                )
            else:
                if int(tok.shape[2]) != int(p.get("num_levels", -1)):
                    errors.append(
                        f"pass2_splatted_tok_low num_levels mismatch: tensor L={tok.shape[2]} vs field={p.get('num_levels')}"
                    )
                if int(tok.shape[3]) != int(p.get("channels", -1)):
                    errors.append(
                        f"pass2_splatted_tok_low channels mismatch: tensor C={tok.shape[3]} vs field={p.get('channels')}"
                    )
                if int(tok.shape[2]) != 4:
                    errors.append(f"pass2_splatted_tok_low L should be 4, got {tok.shape[2]}")
                if int(tok.shape[3]) != 3072:
                    errors.append(f"pass2_splatted_tok_low C should be 3072, got {tok.shape[3]}")
        if scale is not None and tok is not None:
            if scale.shape != (tok.shape[0], tok.shape[2]):
                errors.append(
                    f"pass2_splatted_tok_low_scale shape should be [N,L]={tuple((tok.shape[0], tok.shape[2]))}, got {tuple(scale.shape)}"
                )
    return errors


def _cache_root_and_split(cache_path: Path) -> tuple[Path, str]:
    """Infer ``WaymoFlowCacheDataset(cache_root, split)`` from a cache file path."""
    for parent in cache_path.resolve().parents:
        if parent.name in {"training", "validation", "testing"}:
            return parent.parent, parent.name
    # Preserve the historical flat layout fallback: <cache_root>/<split>/<file>.pt.
    return cache_path.parent.parent, cache_path.parent.name


def _load_single_cache_item(cache_path: Path, num_frames: int) -> dict:
    from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset

    cache_root, split = _cache_root_and_split(cache_path)
    # lut_dtype=fp32 because the verifier doesn't run autocast (training does).
    # Without autocast the tokenizer's fp32 LayerNorm cannot accept fp16 input.
    ds = WaymoFlowCacheDataset(
        cache_root=str(cache_root),
        split=split,
        min_frames=int(num_frames),
        max_frames=int(num_frames),
        seed=0,
        lut_dtype=torch.float32,
    )
    target = cache_path.resolve()
    for i, entry in enumerate(ds.entries):
        if Path(entry["cache_path"]).resolve() == target:
            return ds[i]
    raise FileNotFoundError(f"could not find {cache_path} in dataset entries")


def _move_item_inputs(item: dict, device: torch.device) -> tuple:
    from train_scene_flow import _move_predictions, _move_asset_pass

    sample = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in item["sample"].items()
    }
    predictions = _move_predictions(item["predictions"], device)
    asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
    cameras_dggt = {k: v.to(device) for k, v in item["cameras_dggt"].items()}
    mode_kind = str(item.get("mode_kind", "mode_a"))
    mode_b_payload = item.get("mode_b")
    if mode_b_payload is not None:
        mode_b_payload = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in mode_b_payload.items()
        }
    phase1_lite = item.get("phase1_localized")
    if phase1_lite is not None:
        phase1_lite = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in phase1_lite.items()
        }
    splatted_tok_low_cached = item.get("splatted_tok_low_cached")
    if splatted_tok_low_cached is not None:
        splatted_tok_low_cached = [t.to(device) for t in splatted_tok_low_cached]
    return (
        sample,
        predictions,
        asset_pass_result,
        cameras_dggt,
        mode_kind,
        mode_b_payload,
        phase1_lite,
        splatted_tok_low_cached,
    )


def _build_assembler(item: dict, ckpt_path: Path | None, device: torch.device):
    from dggt.models.flow_feature_assembler import FlowFeatureAssembler
    from train_scene_flow import _load_tokenizer, freeze_module

    tokenizer = _load_tokenizer(str(ckpt_path), device) if ckpt_path is not None else None
    patch_grid = tuple(int(v) for v in item["asset_pass_result"].patch_grid)
    assembler = FlowFeatureAssembler(
        scene_tokenizer=tokenizer,
        patch_grid=patch_grid,
        H_splat=patch_grid[0] * 4,
        W_splat=patch_grid[1] * 4,
        chunk_channels=128,
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    if ckpt_path is not None:
        freeze_module(assembler)
    else:
        assembler.eval()
        for p in assembler.parameters():
            p.requires_grad_(False)
    return assembler


def _run_assembler(
    item: dict,
    ckpt_path: Path,
    device: torch.device,
    *,
    use_cached_splatted_tok_low: bool,
    assembler=None,
):
    if assembler is None:
        assembler = _build_assembler(item, ckpt_path, device)
    (
        sample,
        predictions,
        asset_pass_result,
        cameras_dggt,
        mode_kind,
        mode_b_payload,
        phase1_lite,
        splatted_tok_low_cached,
    ) = _move_item_inputs(item, device)

    if use_cached_splatted_tok_low and splatted_tok_low_cached is None:
        raise RuntimeError("v6 fast-path splatted_tok_low_cached missing")
    if mode_kind == "mode_a" and phase1_lite is None:
        raise RuntimeError("v6 fast-path phase1_localized missing for Mode A")

    base_t = torch.zeros(1, device=device)
    with torch.no_grad():
        return assembler(
            sample=sample,
            predictions=predictions,
            asset_pass_result=asset_pass_result,
            cameras_dggt=cameras_dggt,
            object_slots_spec="all",
            base_t=base_t,
            device=device,
            mode_kind=mode_kind,
            mode_b=mode_b_payload,
            # Keep Mode-A localized objects fixed so checks isolate cache
            # semantics rather than re-running pose refinement.
            phase1_localized_lite=phase1_lite if mode_kind == "mode_a" else None,
            splatted_tok_low_cached=(
                splatted_tok_low_cached if use_cached_splatted_tok_low else None
            ),
        )


def _chunked_splat(
    assembler,
    *,
    gaussians,
    pointers,
    lut_scene,
    lut_asset_dict,
    cameras_splat,
    H: int,
    W: int,
    patch_grid: tuple[int, int],
    frame_chunk: int,
) -> list[torch.Tensor]:
    S_total = int(lut_scene[0].shape[1])
    chunks: list[list[torch.Tensor]] = [[] for _ in range(assembler.num_levels)]
    for fs in range(0, S_total, int(frame_chunk)):
        fe = min(fs + int(frame_chunk), S_total)
        cameras_chunk = {k: v[:, fs:fe].contiguous() for k, v in cameras_splat.items()}
        chunk_out = assembler.feature_splatter(
            gaussians_dggt=[gaussians],
            pointers=[pointers],
            lut_scene=lut_scene,
            lut_asset_dict=lut_asset_dict,
            cameras_dggt=cameras_chunk,
            H=H,
            W=W,
            pool_to=patch_grid,
        )
        for level_idx, level_tensor in enumerate(chunk_out):
            chunks[level_idx].append(level_tensor)
    return [torch.cat(parts, dim=1) for parts in chunks]


def _quantize_full_live_pass2(
    item: dict,
    device: torch.device,
    frame_chunk: int = 2,
):
    """Recompute full-source-Gaussian pass2 cache with bounded peak memory."""
    from dggt.models.flow_feature_assembler import (
        _concat_gauss_dicts,
        _hydrate_lite_localized,
    )
    from dggt.models.gaussian_pointers import GaussianPointers
    from dggt.models.gaussian_scene_editor import EditedSceneState
    from dggt.models.scene_pointers import build_scene_pointers, concat_pointers
    from dggt.utils.edit_coverage import build_phase1_asset_coverage
    from dggt.utils.feature_quant import quantize_tokens

    assembler = _build_assembler(item, None, device)
    (
        sample,
        predictions,
        asset_pass_result,
        cameras_dggt,
        mode_kind,
        mode_b_payload,
        phase1_lite,
        _splatted_tok_low_cached,
    ) = _move_item_inputs(item, device)

    patch_grid = tuple(int(v) for v in asset_pass_result.patch_grid)
    H_splat, W_splat = patch_grid[0] * 4, patch_grid[1] * 4

    clean_state = assembler.editor.build_clean_bundle(sample, predictions)
    H_img, W_img = int(clean_state.images.shape[-2]), int(clean_state.images.shape[-1])
    F_g_lut_scene = assembler._select_lut_scene(predictions)
    cameras_splat = assembler.scale_cameras_for_render(
        cameras_dggt,
        source_hw=(H_img, W_img),
        target_hw=(H_splat, W_splat),
    )

    if mode_kind == "mode_a":
        if phase1_lite is None:
            raise RuntimeError("Mode-A full pass2 check requires phase1_localized")
        localized_objects = _hydrate_lite_localized(phase1_lite)
        cached_delete = phase1_lite["delete_mask"].to(device).bool()
        cached_shell = phase1_lite["shell_mask"].to(device).bool()
        clean_dict = {
            "means": clean_state.means.to(device),
            "colors": clean_state.colors.to(device),
            "opacities": clean_state.opacities.to(device),
            "scales": clean_state.scales.to(device),
            "quats": clean_state.quats.to(device),
        }
        if int(cached_delete.numel()) != int(clean_dict["means"].shape[0]):
            raise RuntimeError(
                "phase1_localized.delete_mask length does not match full clean_state"
            )
        kept_dict = {k: v[~cached_delete] for k, v in clean_dict.items()}
        edit_state = EditedSceneState(
            clean=clean_dict,
            deleted=kept_dict,
            asset_only={k: v[:0] for k, v in clean_dict.items()},
            edited=kept_dict,
            localized_objects=localized_objects,
            delete_mask=cached_delete,
            shell_mask=cached_shell,
        )
        phase1_coverage, phase4_slots = build_phase1_asset_coverage(
            sample["object_asset_image_valid_mask_selected"].detach().cpu(),
            localized_objects,
        )
        asset_pass_result = assembler._mask_asset_pass_by_coverage(
            asset_pass_result=asset_pass_result,
            phase1_coverage=phase1_coverage,
            phase4_slots=phase4_slots,
            device=device,
        )

        ptr_scene = build_scene_pointers(
            clean_state.source_image_ids,
            clean_state.source_y,
            clean_state.source_x,
            patch_size=int(H_img // patch_grid[0]),
            patch_grid=patch_grid,
        ).to(device)
        ptr_chunks: list[GaussianPointers] = [ptr_scene]
        asset_gauss_chunks: list[dict[str, torch.Tensor]] = []
        for obj_key in asset_pass_result.object_keys:
            obj_key = int(obj_key)
            obj_ptrs = asset_pass_result.ptr_asset[obj_key]
            obj_gauss_frames = asset_pass_result.G_asset_dggt[obj_key]
            flat_gauss = _concat_gauss_dicts(
                [{k: v.to(device) for k, v in g.items()} for g in obj_gauss_frames],
                device=device,
            )
            asset_gauss_chunks.append(flat_gauss)
            ptr_chunks.append(
                GaussianPointers(
                    src_kind=torch.cat([p.src_kind for p in obj_ptrs]),
                    object_id=torch.cat([p.object_id for p in obj_ptrs]),
                    view_n=torch.cat([p.view_n for p in obj_ptrs]),
                    patch_idx=torch.cat([p.patch_idx for p in obj_ptrs]),
                    visible_mask=torch.cat([p.visible_mask for p in obj_ptrs]),
                )
            )
        keep_mask = (~edit_state.delete_mask).to(device)
        ptr_chunks[0] = GaussianPointers(
            src_kind=ptr_scene.src_kind[keep_mask],
            object_id=ptr_scene.object_id[keep_mask],
            view_n=ptr_scene.view_n[keep_mask],
            patch_idx=ptr_scene.patch_idx[keep_mask],
            visible_mask=ptr_scene.visible_mask[keep_mask],
        )
        pointers_all = concat_pointers(ptr_chunks)
        gaussians_all = _concat_gauss_dicts(
            [kept_dict] + asset_gauss_chunks,
            device=device,
        )
        F_g_lut_asset = {
            int(k): [level.to(device) for level in asset_pass_result.F_g_lut_asset[int(k)]]
            for k in asset_pass_result.object_keys
        }

        with torch.no_grad():
            splatted_tok_low = _chunked_splat(
                assembler,
                gaussians=gaussians_all,
                pointers=pointers_all,
                lut_scene=F_g_lut_scene,
                lut_asset_dict=F_g_lut_asset if F_g_lut_asset else None,
                cameras_splat=cameras_splat,
                H=H_splat,
                W=W_splat,
                patch_grid=patch_grid,
                frame_chunk=frame_chunk,
            )
            K_map, D_map, I_map, I_per_obj = assembler._render_mode_a_depth_aware_coverage(
                [kept_dict],
                [{k: v[cached_delete] for k, v in clean_dict.items()}],
                [
                    {
                        int(k): _concat_gauss_dicts(
                            asset_pass_result.G_asset_dggt[int(k)],
                            device=device,
                        )
                        for k in asset_pass_result.object_keys
                    }
                ],
                cameras_dggt=cameras_dggt,
                H=H_img,
                W=W_img,
            )
            M_preserve, _M_source, _M_dest = assembler.soft_mask.pool_and_normalize(
                K_map,
                D_map,
                I_map,
                target_grid=patch_grid,
            )
            M_preserve, _M_source, _M_dest = assembler._force_preserve_unedited_tokens(
                K_map=K_map,
                D_map=D_map,
                I_map=I_map,
                M_preserve=M_preserve,
                M_source=_M_source,
                M_dest=_M_dest,
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
    elif mode_kind == "mode_b":
        if mode_b_payload is None or "delete_mask" not in mode_b_payload:
            raise RuntimeError("Mode-B full pass2 check requires mode_b.delete_mask")
        delete_mask = mode_b_payload["delete_mask"].to(device).bool()
        clean_dict = {
            "means": clean_state.means.to(device),
            "colors": clean_state.colors.to(device),
            "opacities": clean_state.opacities.to(device).view(-1, 1)
                if clean_state.opacities.dim() == 1 else clean_state.opacities.to(device),
            "scales": clean_state.scales.to(device),
            "quats": clean_state.quats.to(device),
        }
        if int(delete_mask.numel()) != int(clean_dict["means"].shape[0]):
            raise RuntimeError("mode_b.delete_mask length does not match full clean_state")
        ptr_scene = build_scene_pointers(
            clean_state.source_image_ids,
            clean_state.source_y,
            clean_state.source_x,
            patch_size=int(H_img // patch_grid[0]),
            patch_grid=patch_grid,
        ).to(device)
        delete_masks_by_target = assembler._mode_b_delete_masks_by_target(
            mode_b=mode_b_payload,
            delete_mask=delete_mask,
            S=int(F_g_lut_scene[0].shape[1]),
            n_g=int(clean_dict["means"].shape[0]),
            device=device,
        )
        with torch.no_grad():
            K_map, D_map, I_map, _ = assembler._render_mode_b_per_target_coverage(
                sample=sample,
                clean_state=clean_state,
                clean_dict=clean_dict,
                delete_masks_by_target=delete_masks_by_target,
                cameras_dggt=cameras_dggt,
                H=H_img,
                W=W_img,
            )
            M_preserve, _M_source, _M_dest = assembler.soft_mask.pool_and_normalize(
                K_map,
                D_map,
                I_map,
                target_grid=patch_grid,
            )
            M_preserve, _M_source, _M_dest = assembler._force_preserve_unedited_tokens(
                K_map=K_map,
                D_map=D_map,
                I_map=I_map,
                M_preserve=M_preserve,
                M_source=_M_source,
                M_dest=_M_dest,
            )
            tile_masks = assembler._splat_weight_to_tile_masks(
                (1.0 - M_preserve).clamp(0.0, 1.0),
                threshold=1e-3,
                H_splat=H_splat,
                W_splat=W_splat,
            )
            splatted_tok_low = assembler._splat_mode_b_per_target(
                sample=sample,
                clean_state=clean_state,
                clean_dict=clean_dict,
                ptr_scene=ptr_scene,
                delete_masks_by_target=delete_masks_by_target,
                lut_scene=F_g_lut_scene,
                cameras_splat=cameras_splat,
                tile_masks=tile_masks,
            )
            splatted_tok_low = assembler._blend_preserve_tokens(
                clean_levels=F_g_lut_scene,
                splatted_levels=splatted_tok_low,
                M_preserve=M_preserve,
            )
    else:
        raise RuntimeError(f"unknown mode_kind {mode_kind!r}")

    stacked = torch.stack(
        [level.detach().squeeze(0).cpu() for level in splatted_tok_low],
        dim=2,
    ).contiguous()
    return quantize_tokens(stacked.float(), layout="NPLC")


def _check_subset_cache_slice(item: dict, pass2_payload: dict, errors: list[str]) -> None:
    from dggt.utils.feature_quant import QuantizedTokens, dequantize_tokens

    cached_levels = item.get("splatted_tok_low_cached")
    if cached_levels is None:
        errors.append("subset cache: splatted_tok_low_cached missing")
        return
    saved_data = pass2_payload.get("splatted_tok_low_int8")
    saved_scale = pass2_payload.get("splatted_tok_low_scale")
    if not torch.is_tensor(saved_data) or not torch.is_tensor(saved_scale):
        errors.append("subset cache: missing saved int8/scale payload")
        return

    subset = item["subset_frames"].detach().cpu().to(torch.long)
    deq_full = dequantize_tokens(
        QuantizedTokens(data=saved_data, scale=saved_scale, layout="NPLC"),
        dtype=torch.float32,
    )
    expected = deq_full.index_select(0, subset)
    actual = torch.stack(
        [level.detach().squeeze(0).cpu().float() for level in cached_levels],
        dim=2,
    ).contiguous()
    msg = _diff_tensor("subset_cache.dequantized", actual, expected, 0.0, 0.0)
    if msg is not None:
        errors.append(f"subset_cache.dequantized: {msg}")
    if not torch.isfinite(actual).all():
        errors.append("subset_cache.dequantized: contains non-finite values")
    if float(actual.float().std().item()) <= 1e-8:
        errors.append("subset_cache.dequantized: appears constant")
    print(
        "  [subset cache] "
        f"frames={subset.tolist()} shape={tuple(actual.shape)} "
        f"mean={float(actual.mean().item()):.6g} std={float(actual.std().item()):.6g}",
        flush=True,
    )


def check_assembler_equivalence(
    new_cache_dir: Path,
    ckpt_path: Path,
    device: torch.device,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    splat_atol: float = 3e-2,
    splat_rtol: float = 5e-2,
    num_frames: int = 8,
) -> list[str]:
    """Verify v6 semantics without requiring subset cache == subset live splat."""
    errors: list[str] = []
    cache_payload = load_flow_cache(new_cache_dir, map_location="cpu", weights_only=False)
    pass2_payload = cache_payload.get("pass2_splatted_tok_low") or {}
    num_frames_all = int(cache_payload["meta"]["num_frames"])
    subset_frames_count = min(int(num_frames), num_frames_all)

    saved_data = pass2_payload.get("splatted_tok_low_int8")
    saved_scale = pass2_payload.get("splatted_tok_low_scale")
    if not torch.is_tensor(saved_data) or not torch.is_tensor(saved_scale):
        errors.append("full pass2: missing saved int8/scale payload")
        return errors

    print(
        f"[verify] full-source pass2: recomputing full clip ({num_frames_all} frames) ...",
        flush=True,
    )
    try:
        full_item = _load_single_cache_item(new_cache_dir, num_frames_all)
        q_full_live = _quantize_full_live_pass2(full_item, device=device, frame_chunk=2)
    except Exception as exc:
        errors.append(f"full pass2 recompute failed: {type(exc).__name__}: {exc}")
        return errors

    if not torch.equal(q_full_live.data, saved_data):
        diff = (q_full_live.data.to(torch.int16) - saved_data.to(torch.int16)).abs()
        errors.append(
            "full pass2.int8: live full-source recompute differs from cache "
            f"(max_abs={int(diff.max().item())}, changed={int((diff != 0).sum().item())})"
        )
    msg = _diff_tensor("full pass2.scale", q_full_live.scale, saved_scale, 0.0, 0.0)
    if msg is not None:
        errors.append(f"full pass2.scale: {msg}")

    print(
        f"[verify] subset semantics: checking {subset_frames_count}-frame cache slice "
        "and non-pass2 assembler fields ...",
        flush=True,
    )
    try:
        subset_item = _load_single_cache_item(new_cache_dir, subset_frames_count)
    except Exception as exc:
        errors.append(f"subset dataset load failed: {type(exc).__name__}: {exc}")
        return errors
    _check_subset_cache_slice(subset_item, pass2_payload, errors)

    try:
        print("[verify] running subset assembler with cached fast path ...", flush=True)
        subset_assembler = _build_assembler(subset_item, ckpt_path, device)
        bundle_fast = _run_assembler(
            subset_item,
            ckpt_path,
            device,
            use_cached_splatted_tok_low=True,
            assembler=subset_assembler,
        )
        print("[verify] running subset assembler with live subset splatter path ...", flush=True)
        bundle_live_subset = _run_assembler(
            subset_item,
            ckpt_path,
            device,
            use_cached_splatted_tok_low=False,
            assembler=subset_assembler,
        )
    except Exception as exc:
        errors.append(f"subset assembler check failed: {type(exc).__name__}: {exc}")
        return errors

    def _cmp_field(name: str, a, b, fatol=atol, frtol=rtol):
        if not _is_tensor(a) or not _is_tensor(b):
            return
        msg = _diff_tensor(name, a, b, fatol, frtol)
        if msg is not None:
            errors.append(f"bundle.{name}: {msg}")

    mode_kind = str(subset_item.get("mode_kind", "mode_a"))
    if mode_kind == "mode_a":
        edit_fast = bundle_fast.edit_bundle.edited_state
        edit_live = bundle_live_subset.edit_bundle.edited_state
        _cmp_field("delete_mask", edit_fast.delete_mask, edit_live.delete_mask, fatol=0.0, frtol=0.0)
        _cmp_field("shell_mask", edit_fast.shell_mask, edit_live.shell_mask, fatol=0.0, frtol=0.0)

    # These fields do not depend on requiring full-source cache slices to equal
    # live subset-source splats.  They should remain identical between the two
    # subset assembler paths.
    _cmp_field("M_preserve", bundle_fast.M_preserve, bundle_live_subset.M_preserve)
    _cmp_field("M_source", bundle_fast.M_source, bundle_live_subset.M_source)
    _cmp_field("M_dest", bundle_fast.M_dest, bundle_live_subset.M_dest)
    _cmp_field("z_clean", bundle_fast.z_clean, bundle_live_subset.z_clean, fatol=0.0, frtol=0.0)
    _cmp_field("F_asset_tokens", bundle_fast.F_asset_tokens, bundle_live_subset.F_asset_tokens)
    _cmp_field("scaffold_tok", bundle_fast.scaffold_tok, bundle_live_subset.scaffold_tok)
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old_cache", required=True, type=str)
    p.add_argument("--new_cache", required=True, type=str)
    p.add_argument("--ckpt_path", required=True, type=str)
    p.add_argument("--atol", type=float, default=1e-4,
                   help="Tolerance for v4 backward-compat tensor diff.")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--splat_atol", type=float, default=3e-2,
                   help="Tolerance for cached int8 splatted_tok_low / z_splat diff.")
    p.add_argument("--splat_rtol", type=float, default=5e-2)
    p.add_argument("--num_frames", type=int, default=8,
                   help="Subset size for assembler fast-vs-live check.")
    p.add_argument("--skip_backward", action="store_true",
                   help="Skip OLD-vs-NEW preserved-field comparison.")
    p.add_argument("--skip_assembler", action="store_true",
                   help="Only check schema. Skip the cached vs live assembler diff (slow).")
    args = p.parse_args()

    old_path = Path(args.old_cache)
    new_path = Path(args.new_cache)
    ckpt_path = Path(args.ckpt_path)

    print(f"[load] OLD: {old_path}", flush=True)
    old = load_flow_cache(old_path, map_location="cpu", weights_only=False)
    print(f"  schema_version = {old.get('schema_version')}, mode_kind = {old.get('mode_kind')}", flush=True)
    print(f"[load] NEW: {new_path}", flush=True)
    new = load_flow_cache(new_path, map_location="cpu", weights_only=False)
    print(f"  schema_version = {new.get('schema_version')}, mode_kind = {new.get('mode_kind')}", flush=True)

    print("\n[check 1/3] v4 fields preserved in NEW cache (backward compat) ...", flush=True)
    if args.skip_backward:
        e1 = []
        print("  SKIPPED (--skip_backward).", flush=True)
    else:
        e1 = check_backward_compat(old, new, atol=float(args.atol), rtol=float(args.rtol))
        if not e1:
            print("  PASS — every v4 field in NEW matches OLD.", flush=True)
        else:
            print(f"  FAIL — {len(e1)} differences:", flush=True)
            for line in e1[:30]:
                print(f"    - {line}", flush=True)
            if len(e1) > 30:
                print(f"    ... and {len(e1) - 30} more", flush=True)

    print("\n[check 2/3] new v6 fields present and well-formed ...", flush=True)
    e2 = check_new_fields_present(new)
    if not e2:
        print("  PASS — phase1_localized + pass2_splatted_tok_low present and consistent.", flush=True)
    else:
        print(f"  FAIL — {len(e2)} schema issues:", flush=True)
        for line in e2:
            print(f"    - {line}", flush=True)

    e3: list[str] = []
    if not args.skip_assembler:
        print("\n[check 3/3] v6 full-source cache semantics ...", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        e3 = check_assembler_equivalence(
            new_path,
            ckpt_path,
            device,
            atol=float(args.atol),
            rtol=float(args.rtol),
            splat_atol=float(args.splat_atol),
            splat_rtol=float(args.splat_rtol),
            num_frames=int(args.num_frames),
        )
        if not e3:
            print("  PASS — full pass2 cache and subset non-pass2 fields are consistent.", flush=True)
        else:
            print(f"  FAIL — {len(e3)} differences:", flush=True)
            for line in e3:
                print(f"    - {line}", flush=True)
    else:
        print("\n[check 3/3] SKIPPED (--skip_assembler).", flush=True)

    total = len(e1) + len(e2) + len(e3)
    print(f"\n[summary] {len(e1)} backward-compat + {len(e2)} schema + {len(e3)} assembler = {total} issue(s).", flush=True)
    if total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
