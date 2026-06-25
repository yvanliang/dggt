"""WYSIWYG visualization of a flow-cache ``.pt`` (Mode A / validation).

Produces the SAME visualization set that ``docs/flow_cache_cmd.md`` lists for
Mode A's ``inference_scene_editor.py --dump_features`` -- but read ENTIRELY
from the ``.pt`` via the exact code path training/inference consumes
(``WaymoFlowCacheDataset`` -> ``build_clean_scene_state`` ->
``FlowFeatureAssembler``), so it is what-you-see-is-what-you-get:

    {out}/{name}/
        input_grid.jpg              # cache raw input frames
        clean_render_grid.jpg       # all scene Gaussians (before edit)
        deleted_render_grid.jpg     # kept Gaussians (scene after deletion)
        asset_image_grid.jpg        # cached I_asset of all asset slots
        asset_slot{XX}_grid.jpg     # per-slot cached I_asset
        asset_alpha_slot{XX}_grid.jpg
        edited_grid.jpg             # deleted render + composited assets (final)
        flow_features/{flow_features.pt, masks/, coverage/, scaffold/, depth/}
        visualize_summary.json

The RGB renders use the cache's authoritative ``cameras_dggt`` and the
cache-stored Gaussians (kept = clean minus ``phase1_localized.delete_mask``;
assets = ``asset_pass.G_asset_dggt`` / ``I_asset``). The ``flow_features/``
block is produced by the SAME ``FlowFeatureAssembler`` + ``dump_flow_features``
that ``verify_flow_cache_wysiwyg`` and training use, so the masks/coverage/
scaffold/depth grids are byte-aligned with the training-consumed bundle.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u tools/visualize_flow_cache.py \
        --cache_path /path/to/{index:06d}.pt \
        --output_dir runs/flow_cache_vis
    # add --ckpt_path <dggt.pt> for the real scene_tokenizer (saves true
    #     flow_features.pt latents; otherwise a zero-tokenizer stub is used,
    #     which only affects flow_features.pt, not the RGB / mask grids).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.models.flow_feature_assembler import DGGT_DYNAMIC_STATIC_PROB_THRESHOLD
from dggt.utils.flow_cache_io import (
    is_chunked_flow_cache,
    load_chunked_flow_cache_subset,
    load_chunked_flow_cache_summary,
    load_flow_cache,
)
from dggt.utils.flow_viz import dump_flow_features
from dggt.utils.gaussian_edit import build_clean_scene_state
from dggt.utils.gs import concat_list
from inference_scene_editor import (
    _composite_asset_over_clean,
    _rasterize_scene,
    _save_grid,
    alpha_t,
)
from tools.verify_flow_cache_wysiwyg import _build_verifier_assembler
from tools.verify_v5_cache import _cache_root_and_split, _run_assembler


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_path", required=True, help="A flow-cache {index:06d}.pt")
    p.add_argument("--output_dir", default="runs/flow_cache_vis")
    p.add_argument("--ckpt_path", default=None,
                   help="Optional dggt ckpt -> real scene_tokenizer for flow_features.pt.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--nrow", type=int, default=6, help="Frames per grid row.")
    p.add_argument("--frame_start", type=int, default=0, help="First cache frame to visualize.")
    p.add_argument("--frame_count", type=int, default=0, help="Number of cache frames to visualize; 0 means all remaining frames.")
    p.add_argument("--render_background", choices=["input_deleted_black", "black", "input"], default="input_deleted_black",
                   help="Background for clean/deleted Gaussian renders.")
    p.add_argument("--chunk_channels", type=int, default=64)
    p.add_argument("--splat_pca", action="store_true",
                   help="Also dump PCA-RGB of the splatted tokens per level.")
    p.add_argument("--skip_flow_features", action="store_true",
                   help="Only RGB grids (skip the FlowFeatureAssembler bundle).")
    p.add_argument("--fast_flow_inputs_only", action="store_true",
                   help="Only dump the required new fast-path flow_inputs chunks; skip full RGB render.")
    return p


def _squeeze_cameras(cameras: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """[1,S,4,4]->[S,4,4], [1,S,3,3]->[S,3,3] (drop the batch dim)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in cameras.items():
        if torch.is_tensor(v) and v.dim() == 4 and v.shape[0] == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


def _safe_name(cache_path: Path, payload_meta: dict[str, Any]) -> str:
    clip = str(payload_meta.get("clip_name", "")).replace("/", "_")
    variant = str(payload_meta.get("variant", ""))
    stem = cache_path.stem
    parts = [p for p in (stem, variant, clip) if p]
    return "_".join(parts) or stem


def _select_frame_subset(num_frames: int, frame_start: int, frame_count: int) -> torch.Tensor:
    start = max(0, int(frame_start))
    if start >= int(num_frames):
        raise ValueError(f"--frame_start {start} is outside cache length {num_frames}")
    count = int(frame_count)
    end = int(num_frames) if count <= 0 else min(int(num_frames), start + count)
    if end <= start:
        raise ValueError(f"Invalid frame slice start={start} end={end}")
    return torch.arange(start, end, dtype=torch.long)


def _load_single_cache_item_full(cache_path: Path, frame_subset: torch.Tensor) -> dict:
    """Load a full cache item for visualization, bypassing the training fast path."""
    from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset

    cache_root, split = _cache_root_and_split(cache_path)
    ds = WaymoFlowCacheDataset(
        cache_root=str(cache_root),
        split=split,
        min_frames=int(frame_subset.numel()),
        max_frames=int(frame_subset.numel()),
        seed=0,
        lut_dtype=torch.float32,
    )
    target = cache_path.resolve()
    for entry in ds.entries:
        if Path(entry["cache_path"]).resolve() == target:
            subset_t = frame_subset.clone()
            if is_chunked_flow_cache(cache_path):
                payload = load_chunked_flow_cache_subset(
                    cache_path,
                    subset_t,
                    consumer="scene_flow",
                    asset_lut_level_indices=ds.asset_lut_level_indices,
                )
                subset_payload = torch.arange(int(subset_t.numel()), dtype=torch.long)
            else:
                payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
                subset_payload = subset_t
            return ds._build_item_from_payload(
                payload=payload,
                entry=entry,
                cache_path=cache_path,
                subset_t=subset_t,
                subset_payload=subset_payload,
            )
    raise FileNotFoundError(f"could not find {cache_path} in dataset entries")


def _patch_hw(meta: dict[str, Any]) -> tuple[int, int]:
    grid = meta.get("patch_grid", (0, 0))
    if len(grid) != 2:
        raise RuntimeError(f"Invalid patch_grid={grid!r}")
    return int(grid[0]), int(grid[1])


def _token_grid(x: torch.Tensor, patch_grid: tuple[int, int]) -> torch.Tensor:
    """[S,P,1] or [S,P] -> [S,3,gh,gw]."""
    gh, gw = patch_grid
    if x.dim() == 3:
        x = x[..., 0]
    x = x.float().reshape(int(x.shape[0]), gh, gw).unsqueeze(1)
    return x.expand(-1, 3, -1, -1).contiguous().clamp(0.0, 1.0)


def _channel_grid(x: torch.Tensor, channel: int, patch_grid: tuple[int, int]) -> torch.Tensor:
    """[S,P,C] channel -> normalized [S,3,gh,gw]."""
    gh, gw = patch_grid
    y = x[..., int(channel)].float().reshape(int(x.shape[0]), gh, gw)
    lo = torch.quantile(y.flatten(), 0.01)
    hi = torch.quantile(y.flatten(), 0.99)
    if float((hi - lo).abs()) < 1e-6:
        y = torch.zeros_like(y)
    else:
        y = ((y - lo) / (hi - lo)).clamp(0.0, 1.0)
    y = y.unsqueeze(1)
    return y.expand(-1, 3, -1, -1).contiguous()


def _dump_fast_flow_inputs(
    cache_path: Path,
    out_dir: Path,
    nrow: int,
    frame_subset: torch.Tensor,
) -> dict[str, Any] | None:
    if not is_chunked_flow_cache(cache_path):
        raise RuntimeError(f"{cache_path} is not a chunked flow cache; new flow_inputs chunks are required.")
    summary = load_chunked_flow_cache_summary(cache_path)
    if not bool(summary.get("has_flow_inputs", False)):
        raise RuntimeError(
            f"{cache_path} is missing required generated flow_inputs chunks. "
            "Run tools/backfill_flow_cache_flow_inputs.py --write first."
        )

    payload = load_chunked_flow_cache_subset(
        cache_path,
        frame_subset,
        consumer="scene_flow_fast",
        asset_lut_level_indices=(-1,),
    )
    flow_inputs = payload.get("flow_inputs")
    if not isinstance(flow_inputs, dict):
        raise RuntimeError(f"{cache_path} did not load required flow_inputs payload.")
    required = ("M_preserve", "M_source", "M_dest", "scaffold_pooled")
    missing = [key for key in required if not torch.is_tensor(flow_inputs.get(key))]
    if missing:
        raise RuntimeError(f"{cache_path} flow_inputs missing required tensor(s): {missing}")

    fast_dir = out_dir / "fast_flow_inputs"
    masks_dir = fast_dir / "masks"
    scaffold_dir = fast_dir / "scaffold"
    masks_dir.mkdir(parents=True, exist_ok=True)
    scaffold_dir.mkdir(parents=True, exist_ok=True)

    patch_grid = _patch_hw(payload["meta"])
    mask_files: list[str] = []
    for key in ("M_preserve", "M_source", "M_dest"):
        tensor = flow_inputs[key].detach().cpu()
        rel = Path("fast_flow_inputs") / "masks" / f"{key}_grid.jpg"
        _save_grid(_token_grid(tensor, patch_grid), out_dir / rel, nrow=nrow)
        mask_files.append(str(rel))

    scaffold = flow_inputs["scaffold_pooled"].detach().cpu()
    scaffold_files: list[str] = []
    names = ["depth", "alpha", "keep", "delete", "insert", "dynamic", "time"]
    for ch in range(int(scaffold.shape[-1])):
        label = names[ch] if ch < len(names) else f"ch{ch}"
        rel = Path("fast_flow_inputs") / "scaffold" / f"scaffold_{ch:02d}_{label}_grid.jpg"
        _save_grid(_channel_grid(scaffold, ch, patch_grid), out_dir / rel, nrow=nrow)
        scaffold_files.append(str(rel))

    fast_summary = {
        "has_flow_inputs": True,
        "mode_kind": str(payload.get("mode_kind", "")),
        "patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
        "num_frames": int(frame_subset.numel()),
        "frame_indices": [int(v) for v in frame_subset.tolist()],
        "shapes": {
            key: list(value.shape)
            for key, value in flow_inputs.items()
            if torch.is_tensor(value)
        },
        "phase4_slots": [int(v) for v in flow_inputs.get("phase4_slots", [])],
        "mask_grids": mask_files,
        "scaffold_grids": scaffold_files,
    }
    with open(fast_dir / "flow_inputs_summary.json", "w") as f:
        json.dump(fast_summary, f, indent=2)
    return fast_summary


def _render_kept_over_input(
    clean_state,
    delete_mask: torch.Tensor | None,
    cameras_dggt: dict[str, torch.Tensor],
    input_images: torch.Tensor,
    timestamps: torch.Tensor,
    device: torch.device,
    *,
    render_background: str,
) -> torch.Tensor:
    """Per-frame RGB of kept Gaussians composited over the input frame.

    Mirrors ``inference_scene_editor._render_edited_sequence_with_dggt``
    (static/dynamic split + ``alpha_t`` temporal fade) but uses the cache's
    ``cameras_dggt`` and composites over the real input frame instead of a
    learned sky/background (no model needed -> pure cache WYSIWYG).
    """
    viewmats = cameras_dggt["viewmats"].to(device).float()
    Ks = cameras_dggt["Ks"].to(device).float()
    if viewmats.dim() == 4:
        viewmats = viewmats[0]
    if Ks.dim() == 4:
        Ks = Ks[0]

    means = clean_state.means.to(device).float()
    colors = clean_state.colors.to(device).float()
    opacities = clean_state.opacities.to(device).float().view(-1)
    scales = clean_state.scales.to(device).float()
    quats = clean_state.quats.to(device).float()
    gs_conf = clean_state.gs_conf.to(device).float()
    dynamic_prob = clean_state.dynamic_prob.to(device).float()
    source_image_ids = clean_state.source_image_ids.to(device)

    keep = (
        torch.ones((means.shape[0],), dtype=torch.bool, device=device)
        if delete_mask is None
        else ~delete_mask.to(device).bool()
    )
    static_mask = keep & (dynamic_prob < DGGT_DYNAMIC_STATIC_PROB_THRESHOLD)
    s_pts = means[static_mask]
    s_rgb = colors[static_mask]
    s_op = opacities[static_mask] * (1.0 - dynamic_prob[static_mask])
    s_sc = scales[static_mask]
    s_rot = quats[static_mask]
    s_conf = gs_conf[static_mask]
    ts = timestamps.to(device).float()
    s_ts = ts[source_image_ids[static_mask]] if static_mask.any() else torch.zeros((0,), device=device)

    num_images = int(clean_state.images.shape[0])
    H, W = int(clean_state.images.shape[-2]), int(clean_state.images.shape[-1])
    if render_background in {"input", "input_deleted_black"}:
        base = input_images.to(device).float().clamp(0.0, 1.0)
    else:
        base = torch.zeros_like(input_images, device=device, dtype=torch.float32)
    out_frames: list[torch.Tensor] = []
    with torch.no_grad():
        deleted_alpha: list[torch.Tensor] | None = None
        if render_background == "input_deleted_black" and delete_mask is not None:
            removed = delete_mask.to(device).bool()
            removed_static = removed & (dynamic_prob < DGGT_DYNAMIC_STATIC_PROB_THRESHOLD)
            rs_pts = means[removed_static]
            rs_rgb = colors[removed_static]
            rs_op = opacities[removed_static] * (1.0 - dynamic_prob[removed_static])
            rs_sc = scales[removed_static]
            rs_rot = quats[removed_static]
            rs_conf = gs_conf[removed_static]
            rs_ts = ts[source_image_ids[removed_static]] if removed_static.any() else torch.zeros((0,), device=device)
            deleted_alpha = []
            for i in range(num_images):
                rd_mask = removed & (source_image_ids == i)
                rd_pts, rd_rgb = means[rd_mask], colors[rd_mask]
                rd_op = opacities[rd_mask] * dynamic_prob[rd_mask]
                rd_sc, rd_rot = scales[rd_mask], quats[rd_mask]
                if rs_pts.numel() > 0:
                    rs_op_t = alpha_t(rs_ts, ts[i], rs_op, gamma0=rs_conf)
                    pts_del, rgb_del, op_del, sc_del, rot_del = concat_list(
                        [rs_pts, rs_rgb, rs_op_t, rs_sc, rs_rot],
                        [rd_pts, rd_rgb, rd_op, rd_sc, rd_rot],
                    )
                else:
                    pts_del, rgb_del, op_del, sc_del, rot_del = rd_pts, rd_rgb, rd_op, rd_sc, rd_rot
                if pts_del.numel() == 0:
                    deleted_alpha.append(torch.zeros((1, H, W), dtype=torch.float32, device=device))
                    continue
                _render_del, alpha_del = _rasterize_scene(
                    means=pts_del, rgbs=rgb_del, opacity=op_del, scales=sc_del, rotation=rot_del,
                    viewmat=viewmats[i : i + 1], intrinsic=Ks[i : i + 1],
                    height=H, width=W,
                )
                deleted_alpha.append(alpha_del[0, ..., 0].clamp(0.0, 1.0).unsqueeze(0))

        for i in range(num_images):
            d_mask = keep & (source_image_ids == i)
            d_pts, d_rgb = means[d_mask], colors[d_mask]
            d_op = opacities[d_mask] * dynamic_prob[d_mask]
            d_sc, d_rot = scales[d_mask], quats[d_mask]
            if s_pts.numel() > 0:
                s_op_t = alpha_t(s_ts, ts[i], s_op, gamma0=s_conf)
                pts, rgb, op, sc, rot = concat_list(
                    [s_pts, s_rgb, s_op_t, s_sc, s_rot],
                    [d_pts, d_rgb, d_op, d_sc, d_rot],
                )
            else:
                pts, rgb, op, sc, rot = d_pts, d_rgb, d_op, d_sc, d_rot
            render, alpha = _rasterize_scene(
                means=pts, rgbs=rgb, opacity=op, scales=sc, rotation=rot,
                viewmat=viewmats[i : i + 1], intrinsic=Ks[i : i + 1],
                height=H, width=W,
            )
            fg = render[0, ..., :-1].permute(2, 0, 1).clamp(0.0, 1.0)  # [3,H,W]
            a = alpha[0, ..., 0].clamp(0.0, 1.0).unsqueeze(0)          # [1,H,W]
            base_i = base[i]
            if deleted_alpha is not None:
                da = deleted_alpha[i]
                base_i = base_i * (1.0 - da)
            comp = a * fg + (1.0 - a) * base_i
            out_frames.append(comp.detach().cpu().clamp(0.0, 1.0))
    return torch.stack(out_frames, dim=0)


def main() -> None:
    args = build_argparser().parse_args()
    cache_path = Path(args.cache_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    num_frames = int(meta.get("num_frames", 29))
    frame_subset = _select_frame_subset(num_frames, args.frame_start, args.frame_count)
    name = _safe_name(cache_path, meta)
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = int(args.nrow)

    summary: dict[str, Any] = {
        "cache_path": str(cache_path),
        "name": name,
        "mode_kind": str(payload.get("mode_kind", "")),
        "variant": str(meta.get("variant", "")),
        "clip_name": str(meta.get("clip_name", "")),
        "num_frames": num_frames,
        "visualized_frame_indices": [int(v) for v in frame_subset.tolist()],
    }
    fast_summary = _dump_fast_flow_inputs(cache_path, out_dir, nrow, frame_subset)
    summary["fast_flow_inputs_dir"] = str(out_dir / "fast_flow_inputs")
    summary["fast_flow_inputs"] = fast_summary

    if args.fast_flow_inputs_only:
        with open(out_dir / "visualize_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] {cache_path.name} -> {out_dir}")
        print(f"[done] fast_flow_inputs -> {summary['fast_flow_inputs_dir']}")
        return

    # Canonical training read of the cache (full clip) -> WYSIWYG inputs.
    item = _load_single_cache_item_full(cache_path, frame_subset)
    sample = item["sample"]
    predictions = item["predictions"]
    cameras_dggt = item["cameras_dggt"]
    asset_pass_result = item["asset_pass_result"]
    phase1 = item.get("phase1_localized") or {}
    delete_mask = phase1.get("delete_mask")

    clean_state = build_clean_scene_state(sample, predictions)
    n_g = int(clean_state.means.shape[0])
    if delete_mask is not None:
        delete_mask = delete_mask.cpu().bool()
        if int(delete_mask.numel()) != n_g:
            raise RuntimeError(
                f"delete_mask len {int(delete_mask.numel())} != clean Gaussians {n_g}; "
                "cache layout not contiguous-by-frame (re-run precompute)."
            )

    input_images = sample["images_clean"].detach().cpu().float()  # [S,3,H,W]
    timestamps = sample["timestamps"].detach().cpu().float()
    if timestamps.numel() != input_images.shape[0]:
        timestamps = torch.linspace(0.0, 1.0, input_images.shape[0])

    _save_grid(input_images, out_dir / "input_grid.jpg", nrow=nrow)

    clean_render = _render_kept_over_input(
        clean_state, None, cameras_dggt, input_images, timestamps, device,
        render_background=str(args.render_background),
    )
    _save_grid(clean_render, out_dir / "clean_render_grid.jpg", nrow=nrow)

    deleted_render = _render_kept_over_input(
        clean_state, delete_mask, cameras_dggt, input_images, timestamps, device,
        render_background=str(args.render_background),
    )
    _save_grid(deleted_render, out_dir / "deleted_render_grid.jpg", nrow=nrow)

    # Asset RGB (cached I_asset -> pure WYSIWYG) + per-slot alpha.
    asset_keys = list(getattr(asset_pass_result, "object_keys", []) or [])
    asset_panels: list[torch.Tensor] = []
    for slot in asset_keys:
        i_asset = asset_pass_result.I_asset[int(slot)]
        a_asset = asset_pass_result.A_asset[int(slot)]
        i_asset = i_asset[0] if i_asset.dim() == 5 else i_asset
        a_asset = a_asset[0] if a_asset.dim() == 5 else a_asset
        _save_grid(i_asset.float(), out_dir / f"asset_slot{int(slot):02d}_grid.jpg", nrow=nrow)
        _save_grid(
            a_asset.float().expand(-1, 3, -1, -1).contiguous(),
            out_dir / f"asset_alpha_slot{int(slot):02d}_grid.jpg",
            nrow=nrow,
        )
        asset_panels.append(i_asset.float())
    if asset_panels:
        # Stack slots vertically per frame into one asset_image_grid.
        combined = torch.cat(asset_panels, dim=-2)  # [S,3,H*nslots,W]
        _save_grid(combined, out_dir / "asset_image_grid.jpg", nrow=nrow)

    # Final edited scene = assets composited over the deleted render.
    edited = _composite_asset_over_clean(
        deleted_render, asset_pass_result, _squeeze_cameras(cameras_dggt), device
    )
    _save_grid(edited, out_dir / "edited_grid.jpg", nrow=nrow)

    summary.update({
        "clean_gaussian_count": n_g,
        "deleted_gaussian_count": int(delete_mask.sum().item()) if delete_mask is not None else 0,
        "asset_slots": [int(k) for k in asset_keys],
        "rgb_grids": [
            "input_grid.jpg", "clean_render_grid.jpg", "deleted_render_grid.jpg",
            "asset_image_grid.jpg", "edited_grid.jpg",
        ],
    })

    if not args.skip_flow_features:
        ckpt = Path(args.ckpt_path) if args.ckpt_path else None
        assembler = _build_verifier_assembler(
            item, ckpt, device,
            with_tokenizer=ckpt is not None,
            chunk_channels=int(args.chunk_channels),
        )
        bundle = _run_assembler(
            item, ckpt or Path(""), device,
            use_cached_splatted_tok_low=True, assembler=assembler,
        )
        ff_summary = dump_flow_features(
            bundle,
            out_dir,
            save_tensors=ckpt is not None,
            save_masks=True,
            save_coverage=True,
            save_scaffold=True,
            save_splat_pca=bool(args.splat_pca),
            nrow=nrow,
        )
        summary["flow_features_dir"] = str(out_dir / "flow_features")
        summary["flow_features_pt_saved"] = ckpt is not None
        summary["flow_features_shapes"] = ff_summary.get("shapes", {})
        summary["tokenizer_mode"] = "real" if ckpt is not None else "zero_stub_pretokenizer_only"

    with open(out_dir / "visualize_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {cache_path.name} -> {out_dir}")
    print(f"[done] RGB: {summary['rgb_grids']}")
    if not args.skip_flow_features:
        print(f"[done] flow_features -> {summary['flow_features_dir']}")


if __name__ == "__main__":
    main()
