"""Visualization + feature-pack dumping helpers for FlowDGGT.

Shared by `inference_scene_editor.py` (single-sample debug) and
`train_scene_flow.py` (training-time eval dumps).

Everything writes under `{out_dir}/flow_features/` with a stable subdirectory
layout:

    flow_features.pt                 # FlowFeatureBundle tensors, CPU float16
    masks/M_{preserve,source,dest}_grid.jpg   # [B*S, H, W]
    coverage/{K,D,I}_map_grid.jpg
    coverage/I_per_obj_slot{XX}_grid.jpg
    scaffold/chan{0..6}_grid.jpg
    depth/{D_edited_hires,A_edited_hires}_grid.jpg
    splat_pca/level{0..3}_grid.jpg   # only when save_splat_pca=True
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


# ---------------------------------------------------------------------- #
# Low-level grid savers                                                   #
# ---------------------------------------------------------------------- #
def _tensor_to_pil_rgb(image: torch.Tensor):
    if Image is None:
        raise ModuleNotFoundError("Pillow is required for flow_viz JPG dumps")
    if image.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(image.shape)}")
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}")
    u8 = image.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).contiguous()
    h, w = u8.shape[:2]
    return Image.frombytes("RGB", (w, h), bytes(u8.reshape(-1).tolist()))


def _make_pil_grid(images: list, nrow: int):
    if len(images) == 0:
        raise ValueError("Cannot build a grid from zero images")
    w, h = images[0].size
    nrow = max(1, min(nrow, len(images)))
    ncol = int(math.ceil(len(images) / float(nrow)))
    canvas = Image.new("RGB", (w * nrow, h * ncol))
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        canvas.paste(img, (col * w, row * h))
    return canvas


def save_image_grid(
    images: torch.Tensor,
    path: Path,
    nrow: int | None = None,
) -> None:
    """Save a `[N, C, H, W]` tensor as a JPG grid (C ∈ {1, 3})."""
    if images.dim() != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(images.shape)}")
    N = images.shape[0]
    nrow = min(4, N) if nrow is None else max(1, min(nrow, N))
    pil_images = [_tensor_to_pil_rgb(img) for img in images]
    _make_pil_grid(pil_images, nrow=nrow).save(path)


def _depth_to_viz_images(depth: torch.Tensor) -> torch.Tensor:
    """Normalize raw depth per image to `[N,1,H,W]` for inspection."""
    if depth.dim() == 5:
        B, S, H, W, C = depth.shape
        if C != 1:
            raise ValueError(f"Expected depth trailing dim=1, got {C}")
        x = depth.reshape(B * S, H, W, 1).permute(0, 3, 1, 2)
    elif depth.dim() == 4 and depth.shape[-1] == 1:
        x = depth.permute(0, 3, 1, 2)
    elif depth.dim() == 4 and depth.shape[1] == 1:
        x = depth
    else:
        raise ValueError(f"Unsupported depth shape {tuple(depth.shape)}")

    x = x.detach().cpu().float()
    out = torch.zeros_like(x)
    for idx in range(int(x.shape[0])):
        d = x[idx, 0]
        valid = torch.isfinite(d) & (d > 0.0)
        if not bool(valid.any().item()):
            continue
        vals = d[valid]
        if int(vals.numel()) >= 16:
            lo = torch.quantile(vals, 0.02)
            hi = torch.quantile(vals, 0.98)
        else:
            lo = vals.min()
            hi = vals.max()
        if float((hi - lo).abs().item()) < 1e-6:
            out[idx, 0][valid] = 1.0
        else:
            out[idx, 0] = ((d - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
            out[idx, 0][~valid] = 0.0
    return out


def _patch_mask_to_image(
    mask: torch.Tensor,
    target_hw: tuple[int, int],
    patch_grid: tuple[int, int],
) -> torch.Tensor:
    """Upsample a `[B, S, P, 1]` mask to `[B*S, 1, H, W]` for JPG viz."""
    B, S, P, _ = mask.shape
    gy, gx = patch_grid
    assert P == gy * gx, f"P {P} != patch_grid product {gy}*{gx}"
    m = mask.reshape(B * S, gy, gx, 1).permute(0, 3, 1, 2).float()
    H, W = target_hw
    m = F.interpolate(m, size=(H, W), mode="nearest")
    return m


# ---------------------------------------------------------------------- #
# Feature-pack dumper                                                     #
# ---------------------------------------------------------------------- #
def dump_flow_features(
    bundle,                             # FlowFeatureBundle (duck-typed)
    out_dir: Path,
    *,
    save_tensors: bool = True,
    save_masks: bool = True,
    save_coverage: bool = True,
    save_scaffold: bool = True,
    save_splat_pca: bool = False,
    nrow: int | None = None,
) -> dict[str, Any]:
    """Write `flow_features.pt` + JPG grids. Returns metadata summary dict."""
    out_dir = Path(out_dir)
    feat_dir = out_dir / "flow_features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    if save_tensors:
        pack = _build_feature_pack(bundle)
        torch.save(pack, feat_dir / "flow_features.pt")
    else:
        (feat_dir / "flow_features.pt").unlink(missing_ok=True)

    H, W = int(bundle.K_map.shape[2]), int(bundle.K_map.shape[3])
    patch_grid = bundle.patch_grid

    if save_masks:
        mask_dir = feat_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for name, m in (
            ("M_preserve", bundle.M_preserve),
            ("M_source", bundle.M_source),
            ("M_dest", bundle.M_dest),
        ):
            grid = _patch_mask_to_image(m, (H, W), patch_grid)  # [B*S, 1, H, W]
            save_image_grid(grid, mask_dir / f"{name}_grid.jpg", nrow=nrow)

    if save_coverage:
        cov_dir = feat_dir / "coverage"
        cov_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in (
            ("K_map", bundle.K_map),
            ("D_map", bundle.D_map),
            ("I_map", bundle.I_map),
        ):
            arr = tensor[0].permute(0, 3, 1, 2).float()  # [S, 1, H, W]
            save_image_grid(arr, cov_dir / f"{name}_grid.jpg", nrow=nrow)
        # per-object I_map
        if bundle.I_map_per_obj and len(bundle.I_map_per_obj) > 0:
            per_obj = bundle.I_map_per_obj[0]
            for slot_idx, tensor in per_obj.items():
                arr = tensor.permute(0, 3, 1, 2).float()  # [S, 1, H, W]
                save_image_grid(
                    arr, cov_dir / f"I_per_obj_slot{int(slot_idx):02d}_grid.jpg", nrow=nrow
                )

    if save_scaffold:
        sc_dir = feat_dir / "scaffold"
        sc_dir.mkdir(parents=True, exist_ok=True)
        sh = bundle.scaffold_hires[0]  # [S, H, W, 7]
        for c in range(sh.shape[-1]):
            arr = sh[..., c : c + 1].permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
            save_image_grid(arr, sc_dir / f"chan{c}_grid.jpg", nrow=nrow)

        depth_dir = feat_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        raw_depth = getattr(bundle, "D_edited_hires", None)
        if torch.is_tensor(raw_depth):
            depth_grid = _depth_to_viz_images(raw_depth)
        else:
            depth_grid = sh[..., 0:1].permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
        save_image_grid(depth_grid, depth_dir / "D_edited_hires_grid.jpg", nrow=nrow)

        edited_alpha = getattr(bundle, "A_edited_hires", None)
        if torch.is_tensor(edited_alpha):
            alpha_grid = edited_alpha[0].permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
            save_image_grid(alpha_grid, depth_dir / "A_edited_hires_grid.jpg", nrow=nrow)

    if save_splat_pca:
        pca_dir = feat_dir / "splat_pca"
        pca_dir.mkdir(parents=True, exist_ok=True)
        for lvl_idx, tok in enumerate(bundle.splatted_tok_low):
            rgb = _pca3_rgb_of_patch_tokens(tok, patch_grid)       # [B*S, 3, H, W]
            save_image_grid(rgb, pca_dir / f"level{lvl_idx}_grid.jpg", nrow=nrow)

    mode_kind = str(bundle.extras.get("mode_kind", "mode_a"))
    if mode_kind == "mode_b":
        _dump_mode_b_extras(bundle, feat_dir, nrow=nrow)

    summary = {
        "mode_kind": mode_kind,
        "num_objects": (
            len(bundle.phase4_slots)
            if mode_kind == "mode_a"
            else int(bundle.extras.get("num_imagined_objects", 0))
        ),
        "phase4_slots": list(bundle.phase4_slots),
        "patch_grid": list(bundle.patch_grid),
        "shapes": {
            "M_preserve": list(bundle.M_preserve.shape),
            "D_edited_hires": list(getattr(bundle, "D_edited_hires", bundle.scaffold_hires[..., 0:1]).shape),
            "A_edited_hires": list(getattr(bundle, "A_edited_hires", bundle.scaffold_hires[..., 1:2]).shape),
            "scaffold_tok": list(bundle.scaffold_tok.shape),
            "z_clean": list(bundle.z_clean.shape),
            "z_splat": list(bundle.z_splat.shape),
            "F_asset_tokens": list(bundle.F_asset_tokens.shape),
        },
    }
    raw_depth = getattr(bundle, "D_edited_hires", None)
    if torch.is_tensor(raw_depth):
        depth_float = raw_depth.detach().float()
        valid_depth = torch.isfinite(depth_float) & (depth_float > 0.0)
        summary["depth"] = {
            "D_edited_valid_px": int(valid_depth.sum().item()),
            "D_edited_max": float(depth_float[valid_depth].max().item()) if bool(valid_depth.any().item()) else 0.0,
            "D_edited_mean": float(depth_float[valid_depth].mean().item()) if bool(valid_depth.any().item()) else 0.0,
        }
    if mode_kind == "mode_b":
        summary["imagined_objects"] = list(bundle.extras.get("imagined_objects", []))
        summary["rejection_reason"] = str(bundle.extras.get("rejection_reason", ""))
    return summary


def _dump_mode_b_extras(bundle, feat_dir: Path, nrow: int | None = None) -> None:
    """Mode B viz: imagined-box overlay + clean / pseudo-deleted alpha grids.

    Produces (under `{feat_dir}/mode_b/`):
      imagined_boxes_overlay.jpg   — 2D bboxes per imagined object on clean images
      clean_grid.jpg               — input images
      I_map_grid.jpg               — alpha of pseudo-deleted (imagined) Gaussians
    """
    if Image is None:
        return
    extras = bundle.extras
    imagined_objects = list(extras.get("imagined_objects", []))
    edit_bundle = bundle.edit_bundle
    clean_images = edit_bundle.clean_state.images.detach().cpu().float().clamp(0.0, 1.0)
    if clean_images.dim() != 4:
        return
    mode_b_dir = feat_dir / "mode_b"
    mode_b_dir.mkdir(parents=True, exist_ok=True)
    pil_images = [_tensor_to_pil_rgb(img) for img in clean_images]
    nrow_eff = min(4, len(pil_images)) if nrow is None else nrow
    _make_pil_grid(list(pil_images), nrow=nrow_eff).save(mode_b_dir / "clean_grid.jpg")

    save_image_grid(
        bundle.I_map[0].permute(0, 3, 1, 2).float(),
        mode_b_dir / "I_map_grid.jpg",
        nrow=nrow,
    )

    overlay = _draw_imagined_boxes(pil_images, imagined_objects, num_views=1)
    _make_pil_grid(overlay, nrow=nrow_eff).save(mode_b_dir / "imagined_boxes_overlay.jpg")


def _draw_imagined_boxes(pil_images: list, imagined_objects: list, num_views: int = 1) -> list:
    """Per-frame, per-view 2D bboxes from `ImaginedObject.to_dict()` over images."""
    if Image is None:
        return list(pil_images)
    from PIL import ImageDraw

    out = [img.copy() for img in pil_images]
    palette = [(255, 64, 64), (255, 200, 0), (0, 200, 255), (255, 0, 255), (80, 255, 120)]
    for obj in imagined_objects:
        slot = int(obj.get("slot", 0))
        color = palette[slot % len(palette)]
        bboxes = obj.get("bbox_2d_per_view")
        visibility = obj.get("visible_in_frame_per_view")
        if bboxes is None or visibility is None:
            continue
        for frame_idx, frame_bboxes in enumerate(bboxes):
            for view_idx, box in enumerate(frame_bboxes):
                try:
                    visible = bool(visibility[view_idx][frame_idx])
                except (IndexError, TypeError):
                    visible = False
                if not visible:
                    continue
                image_idx = frame_idx * num_views + view_idx
                if image_idx >= len(out):
                    continue
                ImageDraw.Draw(out[image_idx]).rectangle(
                    [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    outline=color,
                    width=2,
                )
    return out


def _build_feature_pack(bundle) -> dict[str, Any]:
    """Serialize the bundle to CPU float16 tensors + small Python objects."""
    def _f16(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu().to(torch.float16).contiguous()

    def _pack_ptr(p) -> dict[str, torch.Tensor]:
        return {
            "src_kind": p.src_kind.detach().cpu(),
            "object_id": p.object_id.detach().cpu(),
            "view_n": p.view_n.detach().cpu(),
            "patch_idx": p.patch_idx.detach().cpu(),
            "visible_mask": p.visible_mask.detach().cpu(),
        }

    pack = {
        "patch_grid": list(bundle.patch_grid),
        "patch_start_idx": int(bundle.patch_start_idx),
        "phase4_slots": list(bundle.phase4_slots),
        "phase1_coverage": bundle.phase1_coverage.cpu(),
        "cameras_dggt": {k: v.detach().cpu() for k, v in bundle.cameras_dggt.items()},
        "F_g_lut_scene": [_f16(x) for x in bundle.F_g_lut_scene],
        "F_g_lut_asset": {
            int(k): [_f16(lvl) for lvl in v] for k, v in bundle.F_g_lut_asset.items()
        },
        "splatted_tok_low": [_f16(x) for x in bundle.splatted_tok_low],
        "K_map": _f16(bundle.K_map),
        "D_map": _f16(bundle.D_map),
        "I_map": _f16(bundle.I_map),
        "M_preserve": _f16(bundle.M_preserve),
        "M_source": _f16(bundle.M_source),
        "M_dest": _f16(bundle.M_dest),
        "D_edited_hires": _f16(bundle.D_edited_hires),
        "A_edited_hires": _f16(bundle.A_edited_hires),
        "scaffold_hires": _f16(bundle.scaffold_hires),
        "scaffold_tok": _f16(bundle.scaffold_tok),
        "z_clean": _f16(bundle.z_clean),
        "z_splat": _f16(bundle.z_splat),
        "F_asset_tokens": _f16(bundle.F_asset_tokens),
        "pointers_scene": _pack_ptr(bundle.pointers_scene),
        "pointers_asset_by_obj": {
            int(k): _pack_ptr(v) for k, v in bundle.pointers_asset_by_obj.items()
        },
        "gaussians_all_dggt": {
            k: v.detach().cpu() for k, v in bundle.gaussians_all_dggt.items()
        },
    }
    return pack


def _pca3_rgb_of_patch_tokens(
    tok: torch.Tensor,
    patch_grid: tuple[int, int],
) -> torch.Tensor:
    """3-component PCA of `[B, S, P, C]` tokens → `[B*S, 3, gy, gx]` RGB image."""
    B, S, P, C = tok.shape
    flat = tok.reshape(B * S * P, C).float()
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    # Use torch.pca_lowrank for stability
    try:
        u, s, v = torch.pca_lowrank(centered, q=3, center=False)
        proj = centered @ v[:, :3]
    except Exception:
        proj = centered[:, :3]
    proj = proj.reshape(B * S, P, 3)
    # Normalize each image to [0, 1]
    lo = proj.amin(dim=1, keepdim=True)
    hi = proj.amax(dim=1, keepdim=True)
    proj = (proj - lo) / (hi - lo).clamp_min(1e-6)
    gy, gx = patch_grid
    proj = proj.reshape(B * S, gy, gx, 3).permute(0, 3, 1, 2).contiguous()
    return proj
