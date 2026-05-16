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
from pathlib import Path
from typing import Any

import torch

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
from tools.verify_v5_cache import _load_single_cache_item, _run_assembler


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_path", required=True, help="A flow-cache {index:06d}.pt")
    p.add_argument("--output_dir", default="runs/flow_cache_vis")
    p.add_argument("--ckpt_path", default=None,
                   help="Optional dggt ckpt -> real scene_tokenizer for flow_features.pt.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--nrow", type=int, default=6, help="Frames per grid row.")
    p.add_argument("--chunk_channels", type=int, default=64)
    p.add_argument("--splat_pca", action="store_true",
                   help="Also dump PCA-RGB of the splatted tokens per level.")
    p.add_argument("--skip_flow_features", action="store_true",
                   help="Only RGB grids (skip the FlowFeatureAssembler bundle).")
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


def _render_kept_over_input(
    clean_state,
    delete_mask: torch.Tensor | None,
    cameras_dggt: dict[str, torch.Tensor],
    input_images: torch.Tensor,
    timestamps: torch.Tensor,
    device: torch.device,
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
    static_mask = keep & (dynamic_prob < 0.5)
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
    base = input_images.to(device).float().clamp(0.0, 1.0)
    out_frames: list[torch.Tensor] = []
    with torch.no_grad():
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
            comp = a * fg + (1.0 - a) * base[i]
            out_frames.append(comp.detach().cpu().clamp(0.0, 1.0))
    return torch.stack(out_frames, dim=0)


def main() -> None:
    args = build_argparser().parse_args()
    cache_path = Path(args.cache_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from dggt.utils.flow_cache_io import load_flow_cache

    payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    num_frames = int(meta.get("num_frames", 29))
    name = _safe_name(cache_path, meta)
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = int(args.nrow)

    # Canonical training read of the cache (full clip) -> WYSIWYG inputs.
    item = _load_single_cache_item(cache_path, num_frames)
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
        clean_state, None, cameras_dggt, input_images, timestamps, device
    )
    _save_grid(clean_render, out_dir / "clean_render_grid.jpg", nrow=nrow)

    deleted_render = _render_kept_over_input(
        clean_state, delete_mask, cameras_dggt, input_images, timestamps, device
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

    summary: dict[str, Any] = {
        "cache_path": str(cache_path),
        "name": name,
        "mode_kind": str(payload.get("mode_kind", "")),
        "variant": str(meta.get("variant", "")),
        "clip_name": str(meta.get("clip_name", "")),
        "num_frames": num_frames,
        "clean_gaussian_count": n_g,
        "deleted_gaussian_count": int(delete_mask.sum().item()) if delete_mask is not None else 0,
        "asset_slots": [int(k) for k in asset_keys],
        "rgb_grids": [
            "input_grid.jpg", "clean_render_grid.jpg", "deleted_render_grid.jpg",
            "asset_image_grid.jpg", "edited_grid.jpg",
        ],
    }

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
