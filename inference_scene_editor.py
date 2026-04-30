from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.waymo_edit_dataset import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_TRANSFER_ROOT,
    WaymoEditDataset,
    draw_projected_3d_box,
    project_world_points_to_model_image,
)
from dggt.models.asset_pass import AssetAggregatorPass, AssetPassResult
from dggt.models.gaussian_scene_editor import GaussianSceneEditor
from dggt.utils.asset_bank import AssetBank
from dggt.utils.gaussian_edit import parse_object_slots
from dggt.utils.gaussian_ply import write_gaussian_ply, write_point_ply
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.pose_enc import pose_encoding_to_extri_intri

PRED_VEHICLE_SEMANTIC_CLASS_ID = 4
CUSTOM_MASK_VEHICLE_VALUE = 40


def alpha_t(t: torch.Tensor, t0: torch.Tensor | float, alpha: torch.Tensor, gamma0: torch.Tensor, gamma1: float = 0.1):
    if not torch.is_tensor(t0):
        t0 = torch.tensor(float(t0), dtype=t.dtype, device=t.device)
    sigma = torch.log(torch.tensor(gamma1, dtype=alpha.dtype, device=alpha.device)) / ((gamma0) ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    return (alpha * conf).float()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-sample Mode A inference: runs GaussianSceneEditor (Phase 1 "
            "deletion) and AssetAggregatorPass (Phase 4 DGGT-fitted per-object "
            "render) on the same sample, with Phase 4 gated by Phase 1's "
            "(slot, frame) deletion coverage so per-frame asset renders line "
            "up one-to-one with deletions."
        )
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="WaymoEditDataset sample index. If omitted, process every sample.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write renders and PLY files.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt",
        help="DGGT checkpoint path.",
    )
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--dataset_mode", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--processed_root", type=str, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--transfer_root", type=str, default=DEFAULT_TRANSFER_ROOT)
    parser.add_argument("--raw_root", type=str, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--asset_root", type=str, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--candidate_path", type=str, default=None)
    parser.add_argument("--object_slots", type=str, default="all", help="Comma-separated slot ids or 'all'.")
    parser.add_argument("--min_match_score", type=float, default=0.1, help="Skip low-confidence slot-to-scene matches.")
    parser.add_argument("--dynamic_thresh", type=float, default=0.5)
    parser.add_argument("--motion_speed_thresh", type=float, default=1.0)
    parser.add_argument("--dynamic_prob_thresh", type=float, default=0.55)
    parser.add_argument("--dynamic_ratio_thresh", type=float, default=0.35)
    parser.add_argument("--core_scale", type=float, default=0.85)
    parser.add_argument("--shell_scale", type=float, default=1.05)
    parser.add_argument("--proposal_scale", type=float, default=1.25)
    parser.add_argument("--render_max_points", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean_only", action="store_true", help="Only run clean-pass render and GT 3D box overlay.")
    parser.add_argument(
        "--skip_ply",
        action="store_true",
        default=True,
        help="Skip writing PLY outputs during debug runs. PLY writing is disabled by default.",
    )
    parser.add_argument(
        "--pose_refine",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable (on) or bypass (off) the 3D-box pose refinement before semantic-mask deletion.",
    )
    parser.add_argument(
        "--max_pose_refine_yaw_deg",
        type=float,
        default=15.0,
        help="Clamp the shared per-track yaw update around the Waymo 3D-box heading.",
    )
    parser.add_argument(
        "--asset_yaw_correction_deg",
        type=float,
        default=180.0,
        help="Fixed local yaw mapping from canonical asset Gaussians into the Waymo 3D-box frame.",
    )
    parser.add_argument(
        "--dump_features",
        action="store_true",
        help=(
            "Run the FlowFeatureAssembler (Phase 2/3/5/6 input composition) and dump "
            "mask / coverage / scaffold grids under flow_features/."
        ),
    )
    parser.add_argument(
        "--splat_pca",
        action="store_true",
        help="When --dump_features is set, also dump PCA-RGB of splatted_tok per level.",
    )
    return parser


def _load_model(ckpt_path: str, device: torch.device):
    from dggt.models.vggt import VGGT

    model = VGGT().to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    ignored_prefixes = ("track_head.", "sky_model.", "scene_tokenizer.")
    real_missing = [key for key in missing if not key.startswith(ignored_prefixes)]
    real_unexpected = [key for key in unexpected if not key.startswith(ignored_prefixes)]
    if real_missing:
        raise RuntimeError(f"Missing checkpoint keys: {real_missing[:10]}")
    if real_unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {real_unexpected[:10]}")
    model.eval()
    return model


def _tensor_to_pil_rgb(image: torch.Tensor) -> Image.Image:
    if image.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(image.shape)}")
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}")
    image_u8 = image.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).contiguous()
    height, width = image_u8.shape[:2]
    return Image.frombytes("RGB", (width, height), bytes(image_u8.view(torch.uint8).untyped_storage()))


def _make_pil_grid(images: list[Image.Image], nrow: int) -> Image.Image:
    if len(images) == 0:
        raise ValueError("Cannot build a grid from zero images")
    width, height = images[0].size
    nrow = max(1, min(nrow, len(images)))
    ncol = int(math.ceil(len(images) / float(nrow)))
    canvas = Image.new("RGB", (width * nrow, height * ncol))
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        canvas.paste(image, (col * width, row * height))
    return canvas


def _save_grid(images: torch.Tensor, path: Path, nrow: int | None = None) -> None:
    if images.dim() != 4:
        raise ValueError(f"Expected [N,3,H,W], got {tuple(images.shape)}")
    if nrow is None:
        nrow = min(4, images.shape[0])
    pil_images = [_tensor_to_pil_rgb(image) for image in images]
    _make_pil_grid(pil_images, nrow=nrow).save(path)


def _overlay_binary_masks_on_images(
    images: torch.Tensor,
    masks: torch.Tensor,
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
    alpha: float = 0.55,
) -> torch.Tensor:
    if images.dim() != 4 or masks.dim() != 3:
        raise ValueError(f"Expected images [N,3,H,W] and masks [N,H,W], got {tuple(images.shape)} and {tuple(masks.shape)}")
    if images.shape[0] != masks.shape[0] or tuple(images.shape[-2:]) != tuple(masks.shape[-2:]):
        raise ValueError(
            f"Image/mask shape mismatch: images {tuple(images.shape)}, masks {tuple(masks.shape)}"
        )
    overlays = images.detach().cpu().float().clamp(0.0, 1.0).clone()
    color_tensor = torch.tensor(color, dtype=torch.float32).view(1, 3, 1, 1)
    mask_bool = masks.detach().cpu().bool().unsqueeze(1)
    blended = overlays * (1.0 - alpha) + color_tensor * alpha
    return torch.where(mask_bool, blended, overlays).clamp(0.0, 1.0)


def _resize_label_mask_to_model(
    label_mask: Image.Image,
    raw_hw: tuple[int, int],
    target_hw: tuple[int, int],
    target_width: int = 518,
) -> torch.Tensor:
    raw_h, raw_w = [int(v) for v in raw_hw]
    target_h, target_w = [int(v) for v in target_hw]
    new_width = target_width
    new_height = round(raw_h * (new_width / raw_w) / 14) * 14
    resized = label_mask.resize((new_width, new_height), Image.Resampling.NEAREST)
    if new_height > target_width:
        crop_top = (new_height - target_width) // 2
        resized = resized.crop((0, crop_top, new_width, crop_top + target_width))
    resized_np = np.array(resized, dtype=np.uint8)
    out_h, out_w = resized_np.shape[:2]
    if out_h != target_h or out_w != target_w:
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        pad_top = max(0, (target_h - out_h) // 2)
        pad_left = max(0, (target_w - out_w) // 2)
        y_end = min(target_h, pad_top + out_h)
        x_end = min(target_w, pad_left + out_w)
        canvas[pad_top:y_end, pad_left:x_end] = resized_np[: y_end - pad_top, : x_end - pad_left]
        resized_np = canvas
    return torch.from_numpy(resized_np)


def _find_custom_mask_path(custom_mask_root: Path, frame_idx: int, cam_id: int) -> Path | None:
    for ext in (".png", ".jpg"):
        path = custom_mask_root / f"{frame_idx:03d}_{cam_id}{ext}"
        if path.is_file():
            return path
    return None


def _load_input_vehicle_semantic_masks(sample: dict, processed_root: str, split: str) -> torch.Tensor | None:
    custom_mask_root = Path(processed_root) / split / str(sample["scene_dir"]) / "custom_masks"
    if not custom_mask_root.is_dir():
        return None

    masks: list[torch.Tensor] = []
    num_views = int(sample["cam_ids"].numel())
    num_frames = int(sample["frame_indices"].numel())
    image_hw = tuple(int(v) for v in sample["images_clean"].shape[-2:])
    raw_hw_all = sample["raw_image_size_hw"]

    for frame_idx in range(num_frames):
        scene_frame = int(sample["frame_indices"][frame_idx].item())
        for view_offset, cam_id_tensor in enumerate(sample["cam_ids"]):
            cam_id = int(cam_id_tensor.item())
            mask_path = _find_custom_mask_path(custom_mask_root, scene_frame, cam_id)
            if mask_path is None:
                return None
            raw_hw = tuple(int(v) for v in raw_hw_all[view_offset].tolist())
            with Image.open(mask_path) as label_mask:
                label_mask = label_mask.convert("L")
                resized = _resize_label_mask_to_model(label_mask, raw_hw=raw_hw, target_hw=image_hw)
            masks.append((resized == CUSTOM_MASK_VEHICLE_VALUE).to(torch.bool))

    if len(masks) != num_frames * num_views:
        return None
    return torch.stack(masks, dim=0)


def _save_vehicle_semantic_outputs(
    sample: dict,
    predictions: dict[str, torch.Tensor],
    processed_root: str,
    split: str,
    output_dir: Path,
) -> dict[str, object]:
    semantic_logits = predictions.get("semantic_logits")
    if semantic_logits is None:
        return {
            "semantic_available": False,
            "vehicle_class_index": None,
            "input_vehicle_semantic_available": False,
        }

    pred_classes = semantic_logits[0].detach().cpu().argmax(dim=-1)
    pred_vehicle_mask = pred_classes == PRED_VEHICLE_SEMANTIC_CLASS_ID
    pred_vehicle_overlay = _overlay_binary_masks_on_images(sample["images_clean"], pred_vehicle_mask)
    _save_grid(pred_vehicle_overlay, output_dir / "pred_vehicle_semantic_overlay_grid.jpg")

    summary: dict[str, object] = {
        "semantic_available": True,
        "vehicle_class_index": int(PRED_VEHICLE_SEMANTIC_CLASS_ID),
        "pred_vehicle_pixel_counts": [int(v) for v in pred_vehicle_mask.view(pred_vehicle_mask.shape[0], -1).sum(dim=1).tolist()],
        "input_vehicle_semantic_available": False,
    }

    input_vehicle_mask = _load_input_vehicle_semantic_masks(sample, processed_root=processed_root, split=split)
    if input_vehicle_mask is not None:
        input_vehicle_overlay = _overlay_binary_masks_on_images(sample["images_clean"], input_vehicle_mask)
        _save_grid(input_vehicle_overlay, output_dir / "input_vehicle_semantic_overlay_grid.jpg")

        input_counts = [int(v) for v in input_vehicle_mask.view(input_vehicle_mask.shape[0], -1).sum(dim=1).tolist()]
        ious = []
        for idx in range(pred_vehicle_mask.shape[0]):
            pred_mask_i = pred_vehicle_mask[idx]
            input_mask_i = input_vehicle_mask[idx]
            inter = int((pred_mask_i & input_mask_i).sum().item())
            union = int((pred_mask_i | input_mask_i).sum().item())
            ious.append(float(inter / union) if union > 0 else 0.0)
        summary["input_vehicle_semantic_available"] = True
        summary["input_vehicle_pixel_counts"] = input_counts
        summary["pred_vehicle_iou_vs_input"] = ious

    with open(output_dir / "semantic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _save_dataset_box_overlay_grid(sample: dict, output_path: Path) -> None:
    overlay_images = [np.array(_tensor_to_pil_rgb(img), copy=True) for img in sample["images_clean"]]
    editable_count = int(sample["editable_object_count"].item())
    editable_slots = sample["editable_object_indices"][:editable_count].tolist()
    num_views = int(sample["cam_ids"].numel())
    colors = [
        (0, 255, 0),
        (255, 200, 0),
        (255, 0, 0),
        (0, 200, 255),
        (255, 0, 255),
        (160, 255, 0),
    ]

    for color_idx, object_slot in enumerate(editable_slots):
        track_valid = sample["object_track_valid_mask_selected"][object_slot]
        corners_world = sample["object_box_corners_world_selected"][object_slot]
        scene_raw_id = sample["object_scene_raw_ids"][object_slot]
        asset_object_id = sample["object_asset_ids"][object_slot]
        color = colors[color_idx % len(colors)]

        for frame_idx in range(min(sample["frame_indices"].numel(), corners_world.shape[0])):
            if not bool(track_valid[frame_idx].item()):
                continue
            for cam_offset, cam_id in enumerate(sample["cam_ids"].tolist()):
                image_idx = frame_idx * num_views + cam_offset
                if image_idx >= len(overlay_images):
                    continue
                projected_corners, projected_valid = project_world_points_to_model_image(
                    corners_world[frame_idx].tolist(),
                    sample["camera_to_world_corrected"][frame_idx, cam_offset].tolist(),
                    sample["intrinsics"][cam_offset].tolist(),
                    sample["raw_image_size_hw"][cam_offset].tolist(),
                )
                if not projected_valid.any():
                    continue
                object_tag = scene_raw_id if scene_raw_id else asset_object_id
                label = f"{object_tag[:8]} f{int(sample['frame_indices'][frame_idx].item())} c{cam_id}"
                overlay_images[image_idx] = draw_projected_3d_box(
                    overlay_images[image_idx],
                    projected_corners,
                    projected_valid,
                    color=color,
                    label=label,
                )

    pil_images = [Image.fromarray(image) for image in overlay_images]
    _make_pil_grid(pil_images, nrow=max(1, num_views)).save(output_path)


def _save_target_boxes(clean_images: torch.Tensor, localized_objects, output_path: Path) -> None:
    pil_images = [_tensor_to_pil_rgb(img) for img in clean_images]
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (255, 200, 0),
        (0, 200, 255),
        (255, 0, 255),
        (160, 255, 0),
    ]
    for item in localized_objects:
        if item.frame_idx < 0 or item.frame_idx >= len(pil_images):
            continue
        draw = ImageDraw.Draw(pil_images[item.frame_idx])
        color = colors[item.slot_idx % len(colors)]
        if item.target_bbox_model is not None:
            draw.rectangle([float(v) for v in item.target_bbox_model.tolist()], outline=color, width=2)
    _make_pil_grid(pil_images, nrow=min(4, len(pil_images))).save(output_path)


def _save_corner_projection_overlay_grid(clean_images: torch.Tensor, localized_objects, output_path: Path) -> None:
    overlay_images = [np.array(_tensor_to_pil_rgb(img), copy=True) for img in clean_images]
    layers = [
        ("waymo_box_corners_model", "waymo_box_corners_valid", (0, 255, 0), "waymo"),
        ("initial_box_corners_model", "initial_box_corners_valid", (255, 200, 0), "dggt init"),
        ("refined_box_corners_model", "refined_box_corners_valid", (255, 0, 80), "dggt refined"),
    ]
    for item in localized_objects:
        image_idx = int(getattr(item, "source_front_index", getattr(item, "frame_idx", -1)))
        if image_idx < 0 or image_idx >= len(overlay_images):
            continue
        for corners_attr, valid_attr, color, label_prefix in layers:
            corners = getattr(item, corners_attr, None)
            valid = getattr(item, valid_attr, None)
            if corners is None or valid is None:
                continue
            corners_np = corners.detach().cpu().float().numpy()
            valid_np = valid.detach().cpu().bool().numpy()
            if corners_np.shape != (8, 2) or not valid_np.any():
                continue
            label = f"{label_prefix} s{int(item.slot_idx)} f{int(item.frame_idx)}"
            overlay_images[image_idx] = draw_projected_3d_box(
                overlay_images[image_idx],
                corners_np,
                valid_np,
                color=color,
                label=label,
                thickness=2,
            )

    pil_images = [Image.fromarray(image) for image in overlay_images]
    _make_pil_grid(pil_images, nrow=min(4, len(pil_images))).save(output_path)


def _save_bbox_overlay_on_asset_clean_grid(
    asset_clean_images: torch.Tensor,
    localized_objects,
    output_path: Path,
    num_views: int,
) -> None:
    overlay_images = [np.array(_tensor_to_pil_rgb(img), copy=True) for img in asset_clean_images]
    layers = [
        ("waymo_box_corners_model", "waymo_box_corners_valid", (0, 255, 0), "waymo"),
        ("initial_box_corners_model", "initial_box_corners_valid", (255, 200, 0), "dggt init"),
        ("refined_box_corners_model", "refined_box_corners_valid", (255, 0, 80), "dggt"),
    ]
    for item in localized_objects:
        image_idx = int(getattr(item, "source_front_index", getattr(item, "frame_idx", -1)))
        if image_idx < 0 or image_idx >= len(overlay_images):
            continue

        for corners_attr, valid_attr, color, label_prefix in layers:
            corners = getattr(item, corners_attr, None)
            valid = getattr(item, valid_attr, None)
            if corners is None or valid is None:
                continue
            corners_np = corners.detach().cpu().float().numpy()
            valid_np = valid.detach().cpu().bool().numpy()
            if corners_np.shape == (8, 2) and valid_np.any():
                overlay_images[image_idx] = draw_projected_3d_box(
                    overlay_images[image_idx],
                    corners_np,
                    valid_np,
                    color=color,
                    label=f"{label_prefix} s{int(item.slot_idx)}",
                    thickness=2,
                )

    pil_images = [Image.fromarray(image) for image in overlay_images]
    _make_pil_grid(pil_images, nrow=max(1, int(num_views))).save(output_path)


def _save_mask_overlay_grid(clean_images: torch.Tensor, localized_objects, output_path: Path, mask_attr: str) -> None:
    overlay_images = [np.array(_tensor_to_pil_rgb(img), copy=True) for img in clean_images]
    colors = [
        np.array((0, 255, 0), dtype=np.float32),
        np.array((255, 0, 0), dtype=np.float32),
        np.array((255, 200, 0), dtype=np.float32),
        np.array((0, 200, 255), dtype=np.float32),
        np.array((255, 0, 255), dtype=np.float32),
        np.array((160, 255, 0), dtype=np.float32),
    ]
    for item in localized_objects:
        if item.frame_idx < 0 or item.frame_idx >= len(overlay_images):
            continue
        mask = getattr(item, mask_attr, None)
        if mask is None:
            continue
        mask_np = mask.detach().cpu().numpy().astype(bool)
        if mask_np.ndim != 2 or not mask_np.any():
            continue
        color = colors[item.slot_idx % len(colors)]
        base = overlay_images[item.frame_idx].astype(np.float32)
        base[mask_np] = 0.35 * base[mask_np] + 0.65 * color
        overlay_images[item.frame_idx] = base.clip(0.0, 255.0).astype(np.uint8)

    pil_images = [Image.fromarray(image) for image in overlay_images]
    _make_pil_grid(pil_images, nrow=min(4, len(pil_images))).save(output_path)


def _erode_binary_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    mask_f = mask.to(torch.float32).unsqueeze(1)
    kernel = radius * 2 + 1
    eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - mask_f, kernel_size=kernel, stride=1, padding=radius)
    return eroded[:, 0] > 0.5


def _rasterize_scene(
    means: torch.Tensor,
    rgbs: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotation: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from gsplat.rendering import rasterization

    if means.numel() == 0:
        empty_render = torch.zeros((1, height, width, 4), dtype=torch.float32, device=viewmat.device)
        empty_alpha = torch.zeros((1, height, width, 1), dtype=torch.float32, device=viewmat.device)
        return empty_render, empty_alpha

    renders_chunk, alphas_chunk, _ = rasterization(
        means=means,
        quats=rotation,
        scales=scales,
        opacities=opacity,
        colors=rgbs,
        viewmats=viewmat,
        Ks=intrinsic,
        width=width,
        height=height,
        render_mode="RGB+ED",
    )
    return renders_chunk, alphas_chunk


def _repeat_timestamps_for_views(sample: dict, num_images: int) -> torch.Tensor:
    timestamps = sample["timestamps"].detach().cpu().float()
    num_frames = int(sample["frame_indices"].numel())
    num_views = max(1, int(sample["cam_ids"].numel()))
    if timestamps.numel() == num_images:
        return timestamps
    if timestamps.numel() == num_frames and num_frames * num_views == num_images:
        return timestamps.repeat_interleave(num_views)
    raise ValueError(
        f"Unexpected timestamp shape: got {timestamps.numel()} values for {num_images} images "
        f"(frames={num_frames}, views={num_views})"
    )


def _predict_camera_mats(
    predictions: dict[str, torch.Tensor],
    image_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_hw
    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (height, width))
    extrinsic_3x4 = extrinsics[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=extrinsic_3x4.dtype).view(1, 1, 4)
    extrinsic = torch.cat([extrinsic_3x4, bottom.expand(extrinsic_3x4.shape[0], -1, -1)], dim=1)
    intrinsic = intrinsics[0]
    return extrinsic, intrinsic


def _render_background(model, images: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "sky_model") and model.sky_model is not None:
        bg_render = model.sky_model(images, extrinsic, intrinsic)
        return (bg_render - bg_render.min()) / (bg_render.max() - bg_render.min() + 1e-8)
    seq_len, _, _, height, width = images.shape
    return torch.zeros((seq_len, height, width, 3), dtype=images.dtype, device=images.device)


def _render_clean_with_dggt(
    model,
    sample: dict,
    predictions: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    images = sample["images_clean"].unsqueeze(0).to(device)
    sky_mask = sample["sky_mask"].unsqueeze(0).to(device).permute(0, 1, 3, 4, 2)
    bg_mask = (sky_mask == 0).any(dim=-1)
    timestamps = _repeat_timestamps_for_views(sample, images.shape[1]).to(device)

    _, _, _, height, width = images.shape
    extrinsic, intrinsic = _predict_camera_mats(predictions, (height, width), device)
    extrinsic_3x4 = extrinsic[:, :3, :]

    depth_map = predictions["depth"][0]
    point_map = unproject_depth_map_to_point_map(depth_map, extrinsic_3x4, intrinsic)
    point_map = torch.from_numpy(point_map).to(device).float()[None, ...]

    gs_map = predictions["gs_map"]
    gs_conf = predictions["gs_conf"]
    dy_map = predictions["dynamic_conf"].squeeze(-1)

    static_mask = bg_mask & (dy_map < 0.5)
    static_points = point_map[static_mask].reshape(-1, 3)
    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
    static_dynamic_prob = dy_map[static_mask].sigmoid()
    static_opacity = static_opacity * (1.0 - static_dynamic_prob)
    static_gs_conf = gs_conf[static_mask]
    static_image_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
    gs_timestamps = timestamps[static_image_idx]

    dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
    for image_idx in range(dy_map.shape[1]):
        point_map_i = point_map[:, image_idx]
        bg_mask_i = bg_mask[:, image_idx]
        dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
        dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, image_idx], bg_mask_i)
        dynamic_prob = dy_map[:, image_idx][bg_mask_i].sigmoid()
        dynamic_opacity = dynamic_opacity * dynamic_prob

        dynamic_points.append(dynamic_point)
        dynamic_rgbs.append(dynamic_rgb)
        dynamic_opacitys.append(dynamic_opacity)
        dynamic_scales.append(dynamic_scale)
        dynamic_rotations.append(dynamic_rotation)

    chunked_renders, chunked_alphas = [], []
    for image_idx in range(dy_map.shape[1]):
        t0 = timestamps[image_idx]
        static_opacity_t = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
        static_gs_list = [static_points, static_rgbs, static_opacity_t, static_scales, static_rotations]
        if len(dynamic_points) > 0:
            world_points, rgbs, opacity, scales, rotation = concat_list(
                static_gs_list,
                [
                    dynamic_points[image_idx],
                    dynamic_rgbs[image_idx],
                    dynamic_opacitys[image_idx],
                    dynamic_scales[image_idx],
                    dynamic_rotations[image_idx],
                ],
            )
        else:
            world_points, rgbs, opacity, scales, rotation = static_gs_list
        renders_chunk, alphas_chunk = _rasterize_scene(
            means=world_points,
            rgbs=rgbs,
            opacity=opacity,
            scales=scales,
            rotation=rotation,
            viewmat=extrinsic[image_idx : image_idx + 1],
            intrinsic=intrinsic[image_idx : image_idx + 1],
            height=height,
            width=width,
        )
        chunked_renders.append(renders_chunk)
        chunked_alphas.append(alphas_chunk)

    renders = torch.cat(chunked_renders, dim=0)[..., :-1]
    alphas = torch.cat(chunked_alphas, dim=0)
    bg_render = _render_background(model, images, extrinsic, intrinsic)
    renders = alphas * renders + (1.0 - alphas) * bg_render
    return renders.permute(0, 3, 1, 2).detach().cpu().float().clamp(0.0, 1.0)


def _render_edited_sequence_with_dggt(
    model,
    sample: dict,
    predictions: dict[str, torch.Tensor],
    clean_state,
    delete_mask: torch.Tensor | None,
    device: torch.device,
    *,
    include_static: bool = True,
    include_dynamic: bool = True,
    return_aux: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    images = sample["images_clean"].unsqueeze(0).to(device)
    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    extrinsic, intrinsic = _predict_camera_mats(predictions, image_hw, device)
    bg_render = _render_background(model, images, extrinsic, intrinsic)
    timestamps = _repeat_timestamps_for_views(sample, clean_state.images.shape[0]).to(device)

    means = clean_state.means.to(device).float()
    colors = clean_state.colors.to(device).float()
    opacities = clean_state.opacities.to(device).float().view(-1)
    scales = clean_state.scales.to(device).float()
    quats = clean_state.quats.to(device).float()
    gs_conf = clean_state.gs_conf.to(device).float()
    dynamic_prob = clean_state.dynamic_prob.to(device).float()
    source_image_ids = clean_state.source_image_ids.to(device)

    if delete_mask is None:
        keep_mask = torch.ones((means.shape[0],), dtype=torch.bool, device=device)
    else:
        keep_mask = ~delete_mask.to(device)

    if not include_static:
        static_mask = torch.zeros_like(keep_mask)
    else:
        static_mask = keep_mask & (dynamic_prob < 0.5)

    static_points = means[static_mask]
    static_rgbs = colors[static_mask]
    static_opacity = opacities[static_mask] * (1.0 - dynamic_prob[static_mask])
    static_scales = scales[static_mask]
    static_rotations = quats[static_mask]
    static_gs_conf = gs_conf[static_mask]
    gs_timestamps = timestamps[source_image_ids[static_mask]] if static_mask.any() else torch.zeros((0,), device=device)

    chunked_renders, chunked_alphas = [], []
    num_images = clean_state.images.shape[0]
    for image_idx in range(num_images):
        if not include_dynamic:
            dynamic_mask = torch.zeros_like(keep_mask)
        else:
            dynamic_mask = keep_mask & (source_image_ids == image_idx)

        dynamic_points = means[dynamic_mask]
        dynamic_rgbs = colors[dynamic_mask]
        dynamic_opacity = opacities[dynamic_mask] * dynamic_prob[dynamic_mask]
        dynamic_scales = scales[dynamic_mask]
        dynamic_rotations = quats[dynamic_mask]

        if static_points.numel() > 0:
            t0 = timestamps[image_idx]
            static_opacity_t = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
            world_points, rgbs, opacity, scales_t, rotation = concat_list(
                [static_points, static_rgbs, static_opacity_t, static_scales, static_rotations],
                [dynamic_points, dynamic_rgbs, dynamic_opacity, dynamic_scales, dynamic_rotations],
            )
        else:
            world_points, rgbs, opacity, scales_t, rotation = (
                dynamic_points,
                dynamic_rgbs,
                dynamic_opacity,
                dynamic_scales,
                dynamic_rotations,
            )

        renders_chunk, alphas_chunk = _rasterize_scene(
            means=world_points,
            rgbs=rgbs,
            opacity=opacity,
            scales=scales_t,
            rotation=rotation,
            viewmat=extrinsic[image_idx : image_idx + 1],
            intrinsic=intrinsic[image_idx : image_idx + 1],
            height=image_hw[0],
            width=image_hw[1],
        )
        chunked_renders.append(renders_chunk)
        chunked_alphas.append(alphas_chunk)

    renders_raw = torch.cat(chunked_renders, dim=0)
    foreground = renders_raw[..., :-1]
    depths = renders_raw[..., -1]
    alphas = torch.cat(chunked_alphas, dim=0)
    renders = alphas * foreground + (1.0 - alphas) * bg_render
    renders_chw = renders.permute(0, 3, 1, 2).detach().cpu().float().clamp(0.0, 1.0)
    if not return_aux:
        return renders_chw
    return {
        "composed": renders_chw,
        "foreground": foreground.permute(0, 3, 1, 2).detach().cpu().float().clamp(0.0, 1.0),
        "alpha": alphas[..., 0].detach().cpu().float().clamp(0.0, 1.0),
        "depth": depths.detach().cpu().float(),
    }


def _gaussian_to_point_payload(scene: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return scene["means"], scene["colors"]


def _write_scene_outputs(prefix: str, scene: dict[str, torch.Tensor], output_dir: Path) -> None:
    write_gaussian_ply(
        {
            "means": scene["means"],
            "features_dc_rgb": scene["colors"],
            "opacities": scene["opacities"],
            "scales": scene["scales"],
            "quats": scene["quats"],
        },
        output_dir / f"{prefix}_gaussians.ply",
    )
    points, colors = _gaussian_to_point_payload(scene)
    write_point_ply(points, colors, output_dir / f"{prefix}_points.ply")


def _build_phase1_asset_coverage(
    sample: dict,
    localized_objects,
) -> tuple[torch.Tensor, list[int]]:
    """For each (slot, image) pair that Phase 1 actually deleted, mark True.

    Phase 4 re-uses this mask to guarantee that per-frame asset renders match
    Phase 1 one-to-one in both count and pose.
    """
    image_valid = sample["object_asset_image_valid_mask_selected"]
    num_slots, num_images = int(image_valid.shape[0]), int(image_valid.shape[1])
    coverage = torch.zeros((num_slots, num_images), dtype=torch.bool)
    slot_set: set[int] = set()
    for item in localized_objects:
        slot_idx = int(item.slot_idx)
        image_idx = int(item.source_front_index)
        if 0 <= slot_idx < num_slots and 0 <= image_idx < num_images:
            coverage[slot_idx, image_idx] = True
            slot_set.add(slot_idx)
    return coverage, sorted(slot_set)


def _composite_asset_over_clean(
    clean_images: torch.Tensor,
    i_asset_per_slot: dict[int, torch.Tensor],
    a_asset_per_slot: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Over-compose every object's I_asset / A_asset onto the clean views.

    Mirrors how Phase 4's asset renders are consumed downstream — this grid
    answers "on each frame, which pixels does the asset pass claim to own".
    """
    base = clean_images.detach().cpu().float().clamp(0.0, 1.0).clone()
    num_images = base.shape[0]
    for slot_idx, i_asset in i_asset_per_slot.items():
        alpha = a_asset_per_slot[slot_idx]
        rgb = i_asset[0].detach().cpu().float().clamp(0.0, 1.0)
        a = alpha[0].detach().cpu().float().clamp(0.0, 1.0)
        if rgb.shape[0] != num_images or a.shape[0] != num_images:
            continue
        base = a * rgb + (1.0 - a) * base
    return base.clamp(0.0, 1.0)


def _save_asset_pass_outputs(
    result: AssetPassResult,
    clean_images: torch.Tensor,
    output_dir: Path,
    num_views: int,
    skip_ply: bool,
    localized_objects=None,
) -> dict:
    asset_pass_dir = output_dir / "asset_pass"
    asset_pass_dir.mkdir(parents=True, exist_ok=True)

    per_object_info: list[dict] = []
    for slot_idx in result.object_keys:
        if result.G_asset_dggt is None or int(slot_idx) not in result.G_asset_dggt:
            raise RuntimeError(
                "inference_scene_editor requires DGGT-fitted asset pass outputs; "
                f"missing G_asset_dggt for slot {int(slot_idx)}"
            )
        i_asset = result.I_asset[slot_idx][0]
        a_asset = result.A_asset[slot_idx][0]
        nrow = max(1, num_views)
        _save_grid(i_asset, asset_pass_dir / f"I_asset_slot{slot_idx:02d}_grid.jpg", nrow=nrow)
        _save_grid(
            a_asset.expand(-1, 3, -1, -1).contiguous(),
            asset_pass_dir / f"A_asset_slot{slot_idx:02d}_grid.jpg",
            nrow=nrow,
        )
        per_object_info.append(
            {
                "slot_idx": int(slot_idx),
                "num_gauss_per_frame": [
                    int(g["means"].shape[0]) for g in result.G_asset_dggt[slot_idx]
                ],
                "num_visible_pointers_per_frame": [
                    int(p.visible_mask.sum().item()) for p in result.ptr_asset[slot_idx]
                ],
                "F_g_lut_asset_shape": list(result.F_g_lut_asset[slot_idx][0].shape),
            }
        )

    composite = _composite_asset_over_clean(
        clean_images=clean_images,
        i_asset_per_slot=result.I_asset,
        a_asset_per_slot=result.A_asset,
    )
    _save_grid(composite, asset_pass_dir / "asset_pass_over_clean_grid.jpg", nrow=max(1, num_views))
    if localized_objects is not None:
        _save_bbox_overlay_on_asset_clean_grid(
            composite,
            localized_objects,
            asset_pass_dir / "bbox_over_asset_clean_grid.jpg",
            num_views=num_views,
        )

    if not skip_ply:
        for slot_idx in result.object_keys:
            if result.G_asset_dggt is None or int(slot_idx) not in result.G_asset_dggt:
                continue
            for frame_i, gauss in enumerate(result.G_asset_dggt[slot_idx]):
                if gauss["means"].numel() == 0:
                    continue
                write_gaussian_ply(
                    {
                        "means": gauss["means"],
                        "features_dc_rgb": gauss["colors"],
                        "opacities": gauss["opacities"],
                        "scales": gauss["scales"],
                        "quats": gauss["quats"],
                    },
                    asset_pass_dir / f"slot{slot_idx:02d}_frame{frame_i:02d}_dggt_gaussians.ply",
                )

    return {
        "num_objects": len(result.object_keys),
        "object_keys": [int(k) for k in result.object_keys],
        "asset_pass_space": str(result.asset_pass_space),
        "patch_grid": list(result.patch_grid),
        "patch_start_idx": int(result.patch_start_idx),
        "bbox_overlay_path": "asset_pass/bbox_over_asset_clean_grid.jpg"
        if localized_objects is not None
        else None,
        "per_object": per_object_info,
    }


def _build_summary(args, sample, alignment, edited_state) -> dict:
    localized = []
    for item in edited_state.localized_objects:
        localized.append(
            {
                "slot_idx": int(item.slot_idx),
                "frame_idx_local": int(item.frame_idx),
                "frame_idx_scene": int(sample["frame_indices"][item.frame_idx].item()),
                "asset_object_id": item.asset_object_id,
                "scene_raw_object_id": item.scene_raw_object_id,
                "asset_path": item.asset_path,
                "match_score": float(item.match_score),
                "delete_motion_mode": item.delete_motion_mode,
                "waymo_frame_speed_mps": float(item.waymo_frame_speed_mps),
                "waymo_max_speed_mps": float(item.waymo_max_speed_mps),
                "waymo_mean_speed_mps": float(item.waymo_mean_speed_mps),
                "render_dynamic_ratio": float(item.render_dynamic_ratio),
                "candidate_count": int(item.candidate_count),
                "seed_point_count": int(item.seed_point_count),
                "candidate_pool_count": int(item.candidate_pool_count),
                "cluster_kept_count": int(item.cluster_kept_count),
                "delete_core_count": int(item.delete_core_indices.numel()),
                "delete_shell_count": int(item.delete_shell_indices.numel()),
                "target_delete_coverage": float(item.target_delete_coverage),
                "outside_box_leak_ratio": float(item.outside_box_leak_ratio),
                "gt_center": [float(v) for v in item.gt_center.tolist()],
                "gt_size": [float(v) for v in item.gt_size.tolist()],
                "proposal_center": [float(v) for v in item.proposal_center.tolist()],
                "proposal_size": [float(v) for v in item.proposal_size.tolist()],
                "refined_center": [float(v) for v in item.refined_center.tolist()],
                "refined_size": [float(v) for v in item.refined_size.tolist()],
                "target_bbox_model": None
                if item.target_bbox_model is None
                else [float(v) for v in item.target_bbox_model.tolist()],
                "pose_refine": getattr(item, "pose_refine_diagnostics", None),
            }
        )

    return {
        "sample_index": int(args.index),
        "scene_name": sample["scene_name"],
        "clip_name": sample["clip_name"],
        "selected_scene_frame_indices": [int(v) for v in sample["frame_indices"].tolist()],
        "selected_slots": parse_object_slots(sample, args.object_slots),
        "editable_asset_object_ids": list(sample["asset_meta"]["editable_asset_object_ids"]),
        "editable_scene_raw_object_ids": list(sample["asset_meta"]["editable_scene_raw_object_ids"]),
        "sim3_waymo_to_dggt": alignment.as_dict(),
        "clean_gaussian_count": int(edited_state.clean["means"].shape[0]),
        "deleted_gaussian_count": int(edited_state.deleted["means"].shape[0]),
        "localized_objects": localized,
    }


def _as_homogeneous_viewmats(world_to_camera: torch.Tensor) -> torch.Tensor:
    """Normalize predicted world-to-camera matrices to [S, 4, 4]."""
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


def _run_one_sample(
    *,
    args: argparse.Namespace,
    dataset: WaymoEditDataset,
    model,
    device: torch.device,
    sample_index: int,
    output_dir: Path,
) -> dict:
    args.index = int(sample_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset[sample_index]

    images = sample["images_clean"].unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(images, return_tokens=args.dump_features)

    editor = GaussianSceneEditor(
        min_match_score=args.min_match_score,
        dynamic_thresh=args.dynamic_thresh,
        core_scale=args.core_scale,
        shell_scale=args.shell_scale,
        proposal_scale=args.proposal_scale,
        motion_speed_thresh=args.motion_speed_thresh,
        dynamic_prob_thresh=args.dynamic_prob_thresh,
        dynamic_ratio_thresh=args.dynamic_ratio_thresh,
        use_pose_refine=(args.pose_refine == "on"),
        max_pose_refine_yaw_deg=args.max_pose_refine_yaw_deg,
        asset_yaw_correction_deg=args.asset_yaw_correction_deg,
    )

    semantic_summary = _save_vehicle_semantic_outputs(
        sample,
        predictions,
        processed_root=args.processed_root,
        split=args.split,
        output_dir=output_dir,
    )

    clean_state = editor.build_clean_bundle(sample, predictions)
    clean_render = _render_clean_with_dggt(model, sample, predictions, device)
    clean_dynamic_render = _render_edited_sequence_with_dggt(
        model,
        sample,
        predictions,
        clean_state,
        delete_mask=None,
        device=device,
        include_static=False,
        include_dynamic=True,
    )
    clean_static_render = _render_edited_sequence_with_dggt(
        model,
        sample,
        predictions,
        clean_state,
        delete_mask=None,
        device=device,
        include_static=True,
        include_dynamic=False,
    )

    _save_grid(clean_state.images, output_dir / "input_images_grid.jpg")
    _save_grid(clean_render, output_dir / "clean_render_grid.jpg")
    _save_grid(clean_dynamic_render, output_dir / "clean_dynamic_render_grid.jpg")
    _save_grid(clean_static_render, output_dir / "clean_static_render_grid.jpg")
    _save_dataset_box_overlay_grid(sample, output_dir / "projected_boxes_on_inputs.jpg")

    if args.clean_only:
        summary = {
            "sample_index": int(args.index),
            "scene_name": sample["scene_name"],
            "clip_name": sample["clip_name"],
            "selected_scene_frame_indices": [int(v) for v in sample["frame_indices"].tolist()],
            "editable_asset_object_ids": list(sample["asset_meta"]["editable_asset_object_ids"]),
            "editable_scene_raw_object_ids": list(sample["asset_meta"]["editable_scene_raw_object_ids"]),
            "semantic_summary": semantic_summary,
            "clean_only": True,
        }
        with open(output_dir / "edit_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return summary

    alignment = editor.align(sample, clean_state)

    asset_bank = AssetBank()
    asset_cache = asset_bank.as_raw_cache()
    selected_slots = parse_object_slots(sample, args.object_slots)
    localized_objects = editor.localize(
        sample,
        clean_state,
        alignment,
        selected_slots,
        asset_cache=asset_cache,
        load_asset=True,
    )
    _save_target_boxes(
        clean_state.images,
        localized_objects,
        output_dir / "target_boxes.jpg",
    )
    _save_corner_projection_overlay_grid(
        clean_state.images,
        localized_objects,
        output_dir / "corner_projection_refine_overlay.jpg",
    )
    _save_mask_overlay_grid(
        clean_state.images,
        localized_objects,
        output_dir / "delete_component_overlay.jpg",
        "delete_component_pixel_mask",
    )
    edited_state = editor.apply_mode_a(clean_state, localized_objects)

    deleted_bundle = _render_edited_sequence_with_dggt(
        model,
        sample,
        predictions,
        clean_state,
        edited_state.delete_mask,
        device,
        return_aux=True,
    )
    deleted_render = deleted_bundle["composed"]

    _save_grid(deleted_render, output_dir / "deleted_render_grid.jpg")

    if not args.skip_ply:
        _write_scene_outputs("clean", edited_state.clean, output_dir)
        _write_scene_outputs("deleted", edited_state.deleted, output_dir)

    phase1_coverage, phase4_slots = _build_phase1_asset_coverage(sample, localized_objects)
    phase4_sample = dict(sample)
    phase4_sample["object_asset_image_valid_mask_selected"] = (
        sample["object_asset_image_valid_mask_selected"].bool() & phase1_coverage
    )

    asset_pass = AssetAggregatorPass(model.aggregator).to(device)
    cameras_dggt_for_asset = {
        "viewmats": _as_homogeneous_viewmats(clean_state.world_to_camera).to(device),
        "Ks": clean_state.intrinsics.to(device),
    }
    with torch.no_grad():
        asset_pass_result = asset_pass(
            phase4_sample,
            selected_object_slots=phase4_slots,
            alignment=alignment,
            asset_cache=asset_cache,
            occlusion_test=True,
            localized_objects=localized_objects,
            cameras_dggt=cameras_dggt_for_asset,
            render_space="dggt_fitted",
        )

    num_views = int(sample["cam_ids"].numel())
    asset_pass_summary = _save_asset_pass_outputs(
        asset_pass_result,
        clean_images=clean_state.images,
        output_dir=output_dir,
        num_views=num_views,
        skip_ply=args.skip_ply,
        localized_objects=localized_objects,
    )
    asset_pass_summary["phase1_coverage_per_slot"] = {
        int(slot_idx): [int(i) for i in torch.nonzero(phase1_coverage[slot_idx], as_tuple=False).flatten().tolist()]
        for slot_idx in phase4_slots
    }

    summary = _build_summary(args, sample, alignment, edited_state)
    summary["semantic_summary"] = semantic_summary
    summary["asset_pass_summary"] = asset_pass_summary

    if args.dump_features:
        flow_summary = _run_flow_feature_dump(
            sample=phase4_sample,
            predictions=predictions,
            asset_pass_result=asset_pass_result,
            clean_state=clean_state,
            model=model,
            args=args,
            output_dir=output_dir,
            device=device,
        )
        summary["flow_features_summary"] = flow_summary

    with open(output_dir / "edit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = build_argparser().parse_args()
    if args.views != 1:
        raise NotImplementedError(
            f"inference_scene_editor.py currently supports --views 1 only; got --views {args.views}"
        )
    torch.manual_seed(args.seed)

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WaymoEditDataset(
        processed_root=args.processed_root,
        transfer_root=args.transfer_root,
        raw_root=args.raw_root,
        asset_root=args.asset_root,
        split=args.split,
        manifest_path=args.manifest_path,
        candidate_path=args.candidate_path,
        views=args.views,
        mode=args.dataset_mode,
        sequence_length=args.sequence_length,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.ckpt_path, device)

    if args.index is None:
        indices = list(range(len(dataset)))
        multi_sample = True
    else:
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f"--index {args.index} is out of range for dataset length {len(dataset)}")
        indices = [int(args.index)]
        multi_sample = False

    all_summaries = []
    for position, sample_index in enumerate(indices, start=1):
        sample_output_dir = (
            root_output_dir / f"sample_{sample_index:06d}" if multi_sample else root_output_dir
        )
        print(f"[{position}/{len(indices)}] Processing sample index {sample_index} -> {sample_output_dir}")
        summary = _run_one_sample(
            args=args,
            dataset=dataset,
            model=model,
            device=device,
            sample_index=sample_index,
            output_dir=sample_output_dir,
        )
        all_summaries.append(
            {
                "sample_index": int(sample_index),
                "output_dir": str(sample_output_dir),
                "scene_name": summary.get("scene_name"),
                "clip_name": summary.get("clip_name"),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if multi_sample:
        run_summary = {
            "num_samples": len(all_summaries),
            "sample_outputs": all_summaries,
        }
        with open(root_output_dir / "all_samples_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)
        print(json.dumps(run_summary, indent=2))


def _run_flow_feature_dump(
    sample: dict,
    predictions: dict,
    asset_pass_result,
    clean_state,
    model,
    args,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Assemble Phase 2/3/5/6 inputs and write flow_features/ artifacts."""
    from dggt.models.flow_feature_assembler import FlowFeatureAssembler
    from dggt.utils.flow_viz import dump_flow_features

    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    patch_grid = (image_hw[0] // 14, image_hw[1] // 14)
    assembler = FlowFeatureAssembler(
        scene_tokenizer=getattr(model, "scene_tokenizer", None),
        patch_grid=patch_grid,
        H_splat=patch_grid[0] * 4,
        W_splat=patch_grid[1] * 4,
        editor_kwargs={
            "min_match_score": args.min_match_score,
            "dynamic_thresh": args.dynamic_thresh,
            "core_scale": args.core_scale,
            "shell_scale": args.shell_scale,
            "proposal_scale": args.proposal_scale,
            "motion_speed_thresh": args.motion_speed_thresh,
            "dynamic_prob_thresh": args.dynamic_prob_thresh,
            "dynamic_ratio_thresh": args.dynamic_ratio_thresh,
            "use_pose_refine": (args.pose_refine == "on"),
            "max_pose_refine_yaw_deg": args.max_pose_refine_yaw_deg,
            "asset_yaw_correction_deg": args.asset_yaw_correction_deg,
        },
    ).to(device)

    viewmats = _as_homogeneous_viewmats(clean_state.world_to_camera).unsqueeze(0).to(device)
    Ks = clean_state.intrinsics.unsqueeze(0).to(device)
    cameras_dggt = {"viewmats": viewmats, "Ks": Ks}

    with torch.no_grad():
        bundle = assembler(
            sample=sample,
            predictions=predictions,
            asset_pass_result=asset_pass_result,
            cameras_dggt=cameras_dggt,
            object_slots_spec=args.object_slots,
            base_t=None,
            device=device,
        )

    return dump_flow_features(
        bundle,
        output_dir,
        save_tensors=False,
        save_masks=True,
        save_coverage=True,
        save_scaffold=True,
        save_splat_pca=args.splat_pca,
    )


if __name__ == "__main__":
    main()
