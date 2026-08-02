from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.waymo_edit_dataset import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_TRANSFER_ROOT,
    WaymoEditDataset,
)
from dggt.models.gaussian_scene_editor import GaussianSceneEditor
from dggt.utils.gaussian_edit import _transform_sample_track_box
from dggt.utils.gs import concat_list
from dggt.utils.mode_b_planner import ModeBPlanner, apply_mode_b
from dggt.utils.scene_gauge import resolve_scene_gauge_checkpoint_sha256
from inference_scene_editor import (
    _as_homogeneous_viewmats,
    _load_model,
    _make_pil_grid,
    _predict_camera_mats,
    _render_background,
    _render_clean_with_dggt,
    _repeat_timestamps_for_views,
    _rasterize_scene,
    _save_grid,
    _tensor_to_pil_rgb,
    alpha_t,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mode B pseudo deletion debug runner.")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="WaymoEditDataset sample index. If omitted, process every sample.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start dataset index, inclusive, used when --index is omitted.")
    parser.add_argument("--end", type=int, default=None, help="End dataset index, exclusive, used when --index is omitted.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write debug outputs.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt",
        help="DGGT checkpoint path.",
    )
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--views", type=int, default=1, choices=[1, 3])
    parser.add_argument("--dataset_mode", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=29)
    parser.add_argument("--processed_root", type=str, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--transfer_root", type=str, default=DEFAULT_TRANSFER_ROOT)
    parser.add_argument("--raw_root", type=str, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--asset_root", type=str, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--mode_b_manifest", type=str, default=None)
    parser.add_argument("--mode_b_candidate_path", type=str, default=None)
    parser.add_argument(
        "--metric_box_mapping_mode",
        choices=["metric_gauge_v4", "generic_sim3"],
        default="metric_gauge_v4",
    )
    parser.add_argument("--scene_gauge_path", type=str, default=None)
    parser.add_argument("--expected_scene_gauge_dggt_sha256", type=str, default=None)
    parser.add_argument("--render_max_points", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planner_seed",
        type=int,
        default=0,
        help="Base planner seed. Per-sample seed is base + 1009 * dataset_index, matching precompute_flow_features.py.",
    )
    parser.add_argument("--num_objects_target", type=int, default=None)
    parser.add_argument("--max_trials_per_object", type=int, default=80)
    parser.add_argument("--min_visible_frames", type=int, default=15)
    parser.add_argument("--max_semantic_overlap_px", type=int, default=0)
    parser.add_argument("--min_projected_transfer_size_px", type=float, default=128.0)
    parser.add_argument("--max_projected_area_ratio", type=float, default=0.12)
    parser.add_argument("--max_projected_width_ratio", type=float, default=0.45)
    parser.add_argument("--max_projected_height_ratio", type=float, default=0.52)
    parser.add_argument("--min_projected_top_y_ratio", type=float, default=0.20)
    parser.add_argument("--min_projected_center_y_ratio", type=float, default=0.35)
    parser.add_argument("--min_projected_bottom_y_ratio", type=float, default=0.50)
    parser.add_argument("--max_projected_bottom_y_ratio", type=float, default=1.0)
    parser.add_argument("--min_ground_support_ratio", type=float, default=0.18)
    parser.add_argument("--require_first_frame_visible", action="store_true")
    parser.add_argument("--fast_camera_step_ratio", type=float, default=0.018)
    parser.add_argument("--slow_camera_step_ratio", type=float, default=0.006)
    parser.add_argument("--allow_empty_plan", action="store_true")
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument(
        "--dump_features",
        action="store_true",
        help=(
            "Run the training FlowFeatureAssembler for Mode B and dump the actual "
            "bundle masks/coverage/scaffold under flow_features/."
        ),
    )
    parser.add_argument(
        "--splat_pca",
        action="store_true",
        help="When --dump_features is set, also dump PCA-RGB of splatted_tok per level.",
    )
    return parser


def _default_mode_b_manifest(processed_root: str, split: str, views: int) -> Path:
    return Path(processed_root) / "waymo_edit_cache" / "manifests" / split / f"{split}_mode_b_views{views}.jsonl"


def _default_mode_b_candidates(processed_root: str, split: str) -> Path:
    return Path(processed_root) / "waymo_edit_cache" / "metadata" / split / "mode_b_candidates.jsonl"


def _collect_existing_objects_dggt(
    sample: dict[str, Any], clean_state, alignment
) -> list[dict[str, Any]]:
    if "object_valid_mask" not in sample or "object_track_valid_mask_selected" not in sample:
        return []
    object_valid = sample["object_valid_mask"].detach().cpu().bool()
    if object_valid.numel() == 0:
        return []

    track_valid = sample["object_track_valid_mask_selected"].detach().cpu().bool()
    obj_to_world = sample["object_obj_to_world_selected"].detach().cpu().float()
    box_size = sample["object_box_size_selected"].detach().cpu().float()
    objects = []
    for slot_idx in range(int(object_valid.shape[0])):
        if not bool(object_valid[slot_idx].item()):
            continue
        present = track_valid[slot_idx]
        if not bool(present.any().item()):
            continue
        centers = []
        rotations = []
        sizes = []
        for frame_idx in range(int(present.shape[0])):
            if bool(present[frame_idx].item()):
                transform = obj_to_world[slot_idx, frame_idx]
                center, size, rotation = _transform_sample_track_box(
                    sample,
                    clean_state,
                    alignment,
                    transform,
                    box_size[slot_idx, frame_idx],
                    frame_idx=frame_idx,
                    view_offset=0,
                )
                centers.append(center)
                rotations.append(rotation)
                sizes.append(size)
            else:
                centers.append(torch.zeros(3, dtype=torch.float32))
                rotations.append(torch.eye(3, dtype=torch.float32))
                sizes.append(torch.zeros(3, dtype=torch.float32))
        first_valid = int(torch.nonzero(present, as_tuple=False).flatten()[0].item())
        objects.append(
            {
                "slot": int(slot_idx),
                "scene_raw_object_id": str(sample["object_scene_raw_ids"][slot_idx])
                if "object_scene_raw_ids" in sample
                else str(slot_idx),
                "center_dggt_per_frame": torch.stack(centers, dim=0).tolist(),
                "rotation_dggt_per_frame": torch.stack(rotations, dim=0).tolist(),
                "size_dggt": sizes[first_valid].tolist(),
                "present_mask": present.tolist(),
            }
        )
    return objects


def _convert_waymo_existing_objects_to_dggt(
    raw_objects: list[dict[str, Any]],
    sample: dict[str, Any],
    clean_state,
    alignment,
) -> list[dict[str, Any]]:
    objects = []
    for obj in raw_objects:
        obj_to_world_seq = list(obj.get("obj_to_world_waymo", []))
        box_size_seq = list(obj.get("box_size_waymo", []))
        present_mask = [bool(v) for v in obj.get("present_mask", [])]
        if not obj_to_world_seq or not box_size_seq:
            continue
        num_frames = min(len(obj_to_world_seq), len(box_size_seq))
        if not present_mask:
            present_mask = [True] * num_frames
        centers = []
        rotations = []
        sizes = []
        for frame_idx in range(num_frames):
            present = bool(present_mask[frame_idx]) if frame_idx < len(present_mask) else False
            if present:
                transform = torch.tensor(obj_to_world_seq[frame_idx], dtype=torch.float32)
                center, size, rotation = _transform_sample_track_box(
                    sample,
                    clean_state,
                    alignment,
                    transform,
                    torch.tensor(box_size_seq[frame_idx], dtype=torch.float32),
                    frame_idx=frame_idx,
                    view_offset=0,
                )
                centers.append(center)
                rotations.append(rotation)
                sizes.append(size)
            else:
                centers.append(torch.zeros(3, dtype=torch.float32))
                rotations.append(torch.eye(3, dtype=torch.float32))
                sizes.append(torch.zeros(3, dtype=torch.float32))
        objects.append(
            {
                "scene_raw_object_id": str(obj.get("scene_raw_object_id", "")),
                "center_dggt_per_frame": torch.stack(centers, dim=0).tolist(),
                "rotation_dggt_per_frame": torch.stack(rotations, dim=0).tolist(),
                "size_dggt_per_frame": torch.stack(sizes, dim=0).tolist(),
                "present_mask": present_mask[:num_frames],
            }
        )
    return objects


def _delete_mask_for_image(
    delete_mask_per_frame: torch.Tensor,
    image_idx: int,
    num_views: int,
    device: torch.device,
) -> torch.Tensor:
    if delete_mask_per_frame.numel() == 0:
        return torch.zeros((delete_mask_per_frame.shape[-1],), dtype=torch.bool, device=device)
    frame_idx = min(int(image_idx) // max(int(num_views), 1), delete_mask_per_frame.shape[0] - 1)
    return delete_mask_per_frame[frame_idx].to(device).bool()


def _render_edited_sequence_with_frame_masks(
    model,
    sample: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    clean_state,
    delete_mask_per_frame: torch.Tensor,
    device: torch.device,
    *,
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
    num_views = int(sample["cam_ids"].numel())

    render_chunks = []
    alpha_chunks = []
    depth_chunks = []
    for image_idx in range(clean_state.images.shape[0]):
        delete_mask = _delete_mask_for_image(delete_mask_per_frame, image_idx, num_views, device)
        keep_mask = ~delete_mask

        static_mask = keep_mask & (dynamic_prob < 0.5)
        static_points = means[static_mask]
        static_rgbs = colors[static_mask]
        static_opacity = opacities[static_mask] * (1.0 - dynamic_prob[static_mask])
        static_scales = scales[static_mask]
        static_rotations = quats[static_mask]
        static_gs_conf = gs_conf[static_mask]
        gs_timestamps = (
            timestamps[source_image_ids[static_mask]]
            if static_mask.any()
            else torch.zeros((0,), dtype=torch.float32, device=device)
        )

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
        renders_chunk, alphas = _rasterize_scene(
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
        render_chunks.append(renders_chunk[..., :-1])
        alpha_chunks.append(alphas[..., 0])
        depth_chunks.append(renders_chunk[..., -1])

    foreground = torch.cat(render_chunks, dim=0)
    alphas = torch.cat(alpha_chunks, dim=0).unsqueeze(-1)
    renders = alphas * foreground + (1.0 - alphas) * bg_render
    renders_chw = renders.permute(0, 3, 1, 2).detach().cpu().float().clamp(0.0, 1.0)
    if not return_aux:
        return renders_chw
    return {
        "composed": renders_chw,
        "foreground": foreground.permute(0, 3, 1, 2).detach().cpu().float().clamp(0.0, 1.0),
        "alpha": alphas[..., 0].detach().cpu().float().clamp(0.0, 1.0),
        "depth": torch.cat(depth_chunks, dim=0).detach().cpu().float(),
    }


def _render_deleted_alpha_sequence(
    sample: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    clean_state,
    delete_mask_per_frame: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    extrinsic, intrinsic = _predict_camera_mats(predictions, image_hw, device)
    timestamps = _repeat_timestamps_for_views(sample, clean_state.images.shape[0]).to(device)

    means = clean_state.means.to(device).float()
    colors = clean_state.colors.to(device).float()
    opacities = clean_state.opacities.to(device).float().view(-1)
    scales = clean_state.scales.to(device).float()
    quats = clean_state.quats.to(device).float()
    gs_conf = clean_state.gs_conf.to(device).float()
    dynamic_prob = clean_state.dynamic_prob.to(device).float()
    source_image_ids = clean_state.source_image_ids.to(device)
    num_views = int(sample["cam_ids"].numel())

    alpha_chunks = []
    for image_idx in range(clean_state.images.shape[0]):
        selected_mask = _delete_mask_for_image(delete_mask_per_frame, image_idx, num_views, device)

        static_mask = selected_mask & (dynamic_prob < 0.5)
        static_points = means[static_mask]
        static_rgbs = colors[static_mask]
        static_opacity = opacities[static_mask] * (1.0 - dynamic_prob[static_mask])
        static_scales = scales[static_mask]
        static_rotations = quats[static_mask]
        static_gs_conf = gs_conf[static_mask]
        gs_timestamps = (
            timestamps[source_image_ids[static_mask]]
            if static_mask.any()
            else torch.zeros((0,), dtype=torch.float32, device=device)
        )

        dynamic_mask = selected_mask & (source_image_ids == image_idx)
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
        _, alphas = _rasterize_scene(
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
        alpha_chunks.append(alphas[..., 0])
    return torch.cat(alpha_chunks, dim=0).detach().cpu().float().clamp(0.0, 1.0)


def _alpha_heatmap(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.detach().cpu().float().clamp(0.0, 1.0)
    return torch.stack([alpha, alpha * 0.55, torch.zeros_like(alpha)], dim=1).clamp(0.0, 1.0)


def _save_imagined_boxes_overlay(
    clean_images: torch.Tensor,
    plan,
    output_path: Path,
    *,
    semantic_mask=None,
    nrow: int | None = None,
) -> None:
    pil_images = [_tensor_to_pil_rgb(image) for image in clean_images]
    num_views = int(plan.views)
    colors = [(255, 64, 64), (255, 200, 0), (0, 200, 255), (255, 0, 255), (80, 255, 120)]

    if semantic_mask is not None:
        semantic = semantic_mask.detach().cpu().bool()
        for idx, image in enumerate(pil_images):
            if idx >= semantic.shape[0]:
                continue
            mask = semantic[idx]
            if not bool(mask.any().item()):
                continue
            mask_img = Image.fromarray(mask.numpy().astype(np.uint8) * 255, mode="L")
            color_img = image.copy()
            ImageDraw.Draw(color_img).rectangle((0, 0, image.size[0], image.size[1]), fill=(0, 190, 80))
            blended = Image.blend(image, color_img, 0.35)
            pil_images[idx] = Image.composite(blended, image, mask_img)

    for obj in plan.imagined_objects:
        color = colors[int(obj.slot) % len(colors)]
        for frame_idx in range(obj.bbox_2d_per_view.shape[0]):
            for view_idx in range(obj.bbox_2d_per_view.shape[1]):
                if not bool(obj.visible_in_frame_per_view[view_idx, frame_idx].item()):
                    continue
                image_idx = frame_idx * num_views + view_idx
                if image_idx >= len(pil_images):
                    continue
                box = obj.bbox_2d_per_view[frame_idx, view_idx].tolist()
                draw = ImageDraw.Draw(pil_images[image_idx])
                draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline=color, width=2)
    if nrow is None:
        nrow = num_views if num_views > 1 else min(4, len(pil_images))
    _make_pil_grid(pil_images, nrow=nrow).save(output_path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _planner_seed_for_index(base_seed: int, sample_index: int) -> int:
    return int(base_seed) + 1009 * int(sample_index)


def _mode_b_record_meta(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode_b_source": str(record.get("source", "")),
        "mode_b_source_views1": str(record.get("source_views1", "")),
        "mode_b_source_views3": str(record.get("source_views3", "")),
        "mode_b_in_mode_a_views1": bool(record.get("in_mode_a_views1", False)),
        "mode_b_in_mode_a_views3": bool(record.get("in_mode_a_views3", False)),
        "front_editable_count_mean": float(
            sum(record.get("front_editable_count_per_frame", [])) / max(len(record.get("front_editable_count_per_frame", [])), 1)
        ),
        "front3_editable_count_mean": float(
            sum(record.get("front3_editable_count_per_frame", [])) / max(len(record.get("front3_editable_count_per_frame", [])), 1)
        ),
    }


def _run_one_sample(
    *,
    args: argparse.Namespace,
    dataset: WaymoEditDataset,
    model,
    device: torch.device,
    sample_index: int,
    output_dir: Path,
    manifest_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[sample_index]
    record = dataset.samples[sample_index]
    images = sample["images_clean"].unsqueeze(0).to(device)
    print(f"[mode_b] sample {sample_index}: running VGGT forward", flush=True)
    with torch.no_grad():
        predictions = model(images, return_tokens=bool(args.dump_features and args.splat_pca))

    print(f"[mode_b] sample {sample_index}: building clean scene", flush=True)
    editor = GaussianSceneEditor()
    clean_state = editor.build_clean_bundle(sample, predictions)
    alignment = editor.align(sample, clean_state)
    existing_objects = _collect_existing_objects_dggt(sample, clean_state, alignment)
    existing_objects.extend(
        _convert_waymo_existing_objects_to_dggt(
            record.get("existing_objects", []), sample, clean_state, alignment
        )
    )
    existing_objects.extend(record.get("existing_objects_dggt", []))

    planner_seed = _planner_seed_for_index(args.planner_seed, sample_index)
    planner = ModeBPlanner(
        min_visible_frames=args.min_visible_frames,
        max_semantic_overlap_px=args.max_semantic_overlap_px,
        max_trials_per_object=args.max_trials_per_object,
        min_projected_transfer_size_px=args.min_projected_transfer_size_px,
        max_projected_area_ratio=args.max_projected_area_ratio,
        max_projected_width_ratio=args.max_projected_width_ratio,
        max_projected_height_ratio=args.max_projected_height_ratio,
        min_projected_top_y_ratio=args.min_projected_top_y_ratio,
        min_projected_center_y_ratio=args.min_projected_center_y_ratio,
        min_projected_bottom_y_ratio=args.min_projected_bottom_y_ratio,
        max_projected_bottom_y_ratio=args.max_projected_bottom_y_ratio,
        min_ground_support_ratio=args.min_ground_support_ratio,
        require_first_frame_visible=args.require_first_frame_visible,
        fast_camera_step_ratio=args.fast_camera_step_ratio,
        slow_camera_step_ratio=args.slow_camera_step_ratio,
        rng_seed=planner_seed,
    )
    print(f"[mode_b] sample {sample_index}: planning imagined objects with seed {planner_seed}", flush=True)
    plan = planner.plan(
        clean_state,
        existing_objects=existing_objects,
        num_objects_target=args.num_objects_target,
        views=args.views,
        scene_name=str(sample["scene_name"]),
        clip_name=str(sample["clip_name"]),
        clip_index=int(sample["clip_index"].item() if torch.is_tensor(sample["clip_index"]) else sample["clip_index"]),
    )
    if plan.num_imagined_objects == 0 and not args.allow_empty_plan:
        summary = plan.to_dict()
        summary.update(
            {
                "sample_index": int(sample_index),
                "status": "skipped_empty_plan",
                "skip_reason": (
                    "Mode B planner accepted zero imagined objects. "
                    "Try a different --index/--planner_seed or relax "
                    "--min_visible_frames/--max_semantic_overlap_px."
                ),
                "selected_scene_frame_indices": [int(v) for v in sample["frame_indices"].tolist()],
                "cam_ids": [int(v) for v in sample["cam_ids"].tolist()],
                "clean_gaussian_count": int(clean_state.means.shape[0]),
                "deleted_gaussian_count": 0,
                "delete_core_count": 0,
                "delete_shell_count": 0,
                "mode_b_manifest": str(manifest_path),
                "mode_b_candidate_path": str(candidate_path),
                **_mode_b_record_meta(record),
            }
        )
        grid_nrow = args.views if args.views > 1 else min(4, int(clean_state.images.shape[0]))
        _save_grid(clean_state.images, output_dir / "input_images_grid.jpg", nrow=grid_nrow)
        _save_imagined_boxes_overlay(clean_state.images, plan, output_dir / "imagined_boxes_overlay.jpg", nrow=grid_nrow)
        if args.dump_features:
            zero_delete = torch.zeros(
                (int(clean_state.means.shape[0]),),
                dtype=torch.bool,
                device=device,
            )
            summary["flow_features_summary"] = _run_flow_feature_dump(
                sample=sample,
                predictions=predictions,
                clean_state=clean_state,
                model=model,
                plan=plan,
                delete_mask=zero_delete,
                delete_mask_per_frame=None,
                output_dir=output_dir,
                device=device,
                args=args,
            )
        _write_json(output_dir / "mode_b_summary.json", summary)
        print(
            f"[mode_b] sample {sample_index}: accepted zero imagined objects; "
            f"wrote summary to {output_dir / 'mode_b_summary.json'} and continuing",
            flush=True,
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    print(f"[mode_b] sample {sample_index}: accepted {plan.num_imagined_objects} imagined objects", flush=True)
    if args.plan_only:
        summary = plan.to_dict()
        summary.update(
            {
                "sample_index": int(sample_index),
                "selected_scene_frame_indices": [int(v) for v in sample["frame_indices"].tolist()],
                "cam_ids": [int(v) for v in sample["cam_ids"].tolist()],
                "clean_gaussian_count": int(clean_state.means.shape[0]),
                "deleted_gaussian_count": 0,
                "delete_core_count": 0,
                "delete_shell_count": 0,
                "mode_b_manifest": str(manifest_path),
                "mode_b_candidate_path": str(candidate_path),
                "status": "plan_only",
                "plan_only": True,
                **_mode_b_record_meta(record),
            }
        )
        grid_nrow = args.views if args.views > 1 else min(4, int(clean_state.images.shape[0]))
        _save_grid(clean_state.images, output_dir / "input_images_grid.jpg", nrow=grid_nrow)
        _save_imagined_boxes_overlay(clean_state.images, plan, output_dir / "imagined_boxes_overlay.jpg", nrow=grid_nrow)
        _write_json(output_dir / "mode_b_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    deletion = apply_mode_b(clean_state, plan)

    print(f"[mode_b] sample {sample_index}: rendering clean/deleted/D_map outputs", flush=True)
    clean_render = _render_clean_with_dggt(model, sample, predictions, device)
    deleted_bundle = _render_edited_sequence_with_frame_masks(
        model,
        sample,
        predictions,
        clean_state,
        deletion.delete_mask_per_frame,
        device,
        return_aux=True,
    )
    deleted_render = deleted_bundle["composed"]
    d_map = _render_deleted_alpha_sequence(sample, predictions, clean_state, deletion.delete_mask_per_frame, device)

    grid_nrow = args.views if args.views > 1 else min(4, int(clean_state.images.shape[0]))
    _save_grid(clean_state.images, output_dir / "input_images_grid.jpg", nrow=grid_nrow)
    _save_grid(clean_render, output_dir / "clean_render_grid.jpg", nrow=grid_nrow)
    _save_grid(deleted_render, output_dir / "deleted_render_grid.jpg", nrow=grid_nrow)
    _save_grid(_alpha_heatmap(d_map), output_dir / "d_map_grid.jpg", nrow=grid_nrow)
    _save_imagined_boxes_overlay(clean_state.images, plan, output_dir / "imagined_boxes_overlay.jpg", nrow=grid_nrow)
    _save_imagined_boxes_overlay(
        clean_state.images,
        plan,
        output_dir / "semantic_vehicle_mask_overlay.jpg",
        semantic_mask=clean_state.semantic_vehicle_mask,
        nrow=grid_nrow,
    )

    summary = plan.to_dict()
    summary.update(
        {
            "sample_index": int(sample_index),
            "selected_scene_frame_indices": [int(v) for v in sample["frame_indices"].tolist()],
            "cam_ids": [int(v) for v in sample["cam_ids"].tolist()],
            "clean_gaussian_count": int(clean_state.means.shape[0]),
            "deleted_gaussian_count": int(deletion.delete_mask.sum().item()),
            "delete_core_count": int(deletion.delete_core_indices.numel()),
            "delete_shell_count": int(deletion.delete_shell_indices.numel()),
            "mode_b_manifest": str(manifest_path),
            "mode_b_candidate_path": str(candidate_path),
            "status": "completed",
            **_mode_b_record_meta(record),
        }
    )
    if args.dump_features:
        summary["flow_features_summary"] = _run_flow_feature_dump(
            sample=sample,
            predictions=predictions,
            clean_state=clean_state,
            model=model,
            plan=plan,
            delete_mask=deletion.delete_mask,
            delete_mask_per_frame=deletion.delete_mask_per_frame,
            output_dir=output_dir,
            device=device,
            args=args,
        )
    _write_json(output_dir / "mode_b_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def _run_flow_feature_dump(
    *,
    sample: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    clean_state,
    model,
    plan,
    delete_mask: torch.Tensor,
    delete_mask_per_frame: torch.Tensor | None,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Dump Mode-B masks by invoking the same assembler path used by training."""
    from dggt.models.asset_pass import AssetPassResult
    from dggt.models.flow_feature_assembler import FlowFeatureAssembler
    from dggt.utils.flow_viz import dump_flow_features

    if not args.splat_pca:
        return _run_flow_feature_mask_dump(
            sample=sample,
            predictions=predictions,
            clean_state=clean_state,
            plan=plan,
            delete_mask=delete_mask,
            delete_mask_per_frame=delete_mask_per_frame,
            output_dir=output_dir,
            device=device,
            args=args,
        )

    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    patch_grid = (image_hw[0] // 14, image_hw[1] // 14)
    patch_start_idx = int(predictions.get("patch_start_idx", 5))
    asset_pass_empty = AssetPassResult(
        patch_grid=patch_grid,
        patch_start_idx=patch_start_idx,
        object_keys=[],
        cameras_waymo={},
        F_g_lut_asset={},
        ptr_asset={},
        G_asset_waymo={},
        G_asset_dggt={},
        I_asset={},
        A_asset={},
        asset_pass_space="mode_b_empty",
        fit_metrics={},
    )
    cameras_dggt = {
        "viewmats": _as_homogeneous_viewmats(clean_state.world_to_camera).unsqueeze(0).to(device),
        "Ks": clean_state.intrinsics.unsqueeze(0).to(device),
        "camera_to_world": clean_state.camera_to_world.unsqueeze(0).to(device),
    }
    assembler = FlowFeatureAssembler(
        scene_tokenizer=getattr(model, "scene_tokenizer", None),
        patch_grid=patch_grid,
        H_splat=patch_grid[0] * 4,
        W_splat=patch_grid[1] * 4,
        chunk_channels=64,
        editor_kwargs={"use_pose_refine": True},
    ).to(device)

    mode_b_payload = dict(plan.to_dict())
    mode_b_payload["delete_mask"] = delete_mask.to(device).bool()
    if delete_mask_per_frame is not None:
        mode_b_payload["delete_mask_per_frame"] = delete_mask_per_frame.to(device).bool()
    mode_b_payload["num_imagined_objects"] = int(plan.num_imagined_objects)

    with torch.no_grad():
        bundle = assembler(
            sample=sample,
            predictions=predictions,
            asset_pass_result=asset_pass_empty,
            cameras_dggt=cameras_dggt,
            object_slots_spec="all",
            base_t=None,
            device=device,
            mode_kind="mode_b",
            mode_b=mode_b_payload,
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


def _run_flow_feature_mask_dump(
    *,
    sample: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    clean_state,
    plan,
    delete_mask: torch.Tensor,
    delete_mask_per_frame: torch.Tensor | None,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Dump Mode-B masks/scaffold without materializing 3072-D splat features."""
    from dggt.models.flow_feature_assembler import FlowFeatureAssembler
    from dggt.models.scaffold import ScaffoldPacker
    from dggt.utils.flow_viz import dump_flow_features

    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])
    patch_grid = (image_hw[0] // 14, image_hw[1] // 14)
    patch_start_idx = int(predictions.get("patch_start_idx", 5))
    B = 1
    S = int(clean_state.images.shape[0])
    H_img, W_img = image_hw
    P = int(patch_grid[0] * patch_grid[1])

    assembler = FlowFeatureAssembler(
        scene_tokenizer=None,
        patch_grid=patch_grid,
        H_splat=patch_grid[0] * 4,
        W_splat=patch_grid[1] * 4,
        chunk_channels=64,
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    assembler.eval()

    delete_mask = delete_mask.to(device=device, dtype=torch.bool)
    clean_dict = {
        "means": clean_state.means.to(device),
        "colors": clean_state.colors.to(device),
        "opacities": clean_state.opacities.to(device).view(-1, 1)
            if clean_state.opacities.dim() == 1 else clean_state.opacities.to(device),
        "scales": clean_state.scales.to(device),
        "quats": clean_state.quats.to(device),
    }
    cameras_dggt = {
        "viewmats": _as_homogeneous_viewmats(clean_state.world_to_camera).unsqueeze(0).to(device),
        "Ks": clean_state.intrinsics.unsqueeze(0).to(device),
        "camera_to_world": clean_state.camera_to_world.unsqueeze(0).to(device),
    }
    mode_b_payload = {"delete_mask": delete_mask}
    if delete_mask_per_frame is not None:
        mode_b_payload["delete_mask_per_frame"] = delete_mask_per_frame.to(device).bool()
    delete_masks_by_target = assembler._mode_b_delete_masks_by_target(
        mode_b=mode_b_payload,
        delete_mask=delete_mask,
        S=S,
        n_g=int(clean_state.means.shape[0]),
        device=device,
    )
    mode_b_noop = not bool(delete_masks_by_target.any().item())

    with torch.no_grad():
        if mode_b_noop:
            K_map = torch.ones((B, S, H_img, W_img, 1), dtype=torch.float32, device=device)
            D_map = torch.zeros_like(K_map)
            I_map = torch.zeros_like(K_map)
            D_edited_hires = assembler._pass1_depth_hires(
                predictions,
                B=B,
                S=S,
                H=H_img,
                W=W_img,
                device=device,
                dtype=K_map.dtype,
            )
            I_per_obj: list[dict[int, torch.Tensor]] = [{}]
            M_preserve = torch.ones((B, S, P, 1), dtype=torch.float32, device=device)
            M_source = torch.zeros_like(M_preserve)
            M_dest = torch.zeros_like(M_preserve)
        else:
            K_map, D_map, I_map, I_per_obj, D_edited_hires = assembler._render_mode_b_per_target_coverage(
                sample=sample,
                clean_state=clean_state,
                clean_dict=clean_dict,
                delete_masks_by_target=delete_masks_by_target,
                cameras_dggt=cameras_dggt,
                H=H_img,
                W=W_img,
                return_effective_depth=True,
            )
            M_preserve, M_source, M_dest = assembler.soft_mask.pool_and_normalize(
                K_map, D_map, I_map, target_grid=patch_grid
            )
            M_preserve, M_source, M_dest = assembler._force_preserve_unedited_tokens(
                K_map=K_map,
                D_map=D_map,
                I_map=I_map,
                M_preserve=M_preserve,
                M_source=M_source,
                M_dest=M_dest,
            )

        A_edited_hires = assembler.soft_mask.compose_deleted_hole_alpha(
            K_alpha=K_map,
            hole_alpha=I_map,
        )
        dyn_prior = torch.sigmoid(
            predictions["dynamic_conf"].reshape(B, S, H_img, W_img, 1).to(device)
        ).float()
        time_index = torch.arange(S, dtype=torch.float32, device=device).view(1, S)
        time_index = time_index / max(S - 1, 1)
        scaffold_hires = ScaffoldPacker.build_scaffold_hires(
            D_edited=D_edited_hires,
            A_edited=A_edited_hires,
            K_map=K_map,
            D_map=D_map,
            I_map=I_map,
            dynamic_prior=dyn_prior,
            time_index=time_index,
        )

    empty_shape = SimpleNamespace(shape=(0,))
    bundle = SimpleNamespace(
        edit_bundle=SimpleNamespace(clean_state=clean_state),
        phase4_slots=[],
        splatted_tok_low=[],
        K_map=K_map,
        D_map=D_map,
        I_map=I_map,
        I_map_per_obj=I_per_obj,
        M_preserve=M_preserve,
        M_source=M_source,
        M_dest=M_dest,
        D_edited_hires=D_edited_hires,
        A_edited_hires=A_edited_hires,
        scaffold_hires=scaffold_hires,
        scaffold_tok=SimpleNamespace(shape=(B, S, P, 768)),
        z_clean=empty_shape,
        z_splat=empty_shape,
        z_init=empty_shape,
        t_tok=empty_shape,
        F_asset_tokens=SimpleNamespace(shape=(B, 0, 3072)),
        patch_grid=patch_grid,
        patch_start_idx=patch_start_idx,
        extras={
            "mode_kind": "mode_b",
            "imagined_objects": list(plan.to_dict().get("imagined_objects", [])),
            "num_imagined_objects": int(plan.num_imagined_objects),
            "rejection_reason": str(plan.to_dict().get("rejection_reason", "")),
            "dump_kind": "mask_only",
        },
    )
    return dump_flow_features(
        bundle,
        output_dir,
        save_tensors=False,
        save_masks=True,
        save_coverage=True,
        save_scaffold=True,
        save_splat_pca=False,
    )


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    scene_gauge_dggt_sha256 = resolve_scene_gauge_checkpoint_sha256(
        args.ckpt_path,
        args.expected_scene_gauge_dggt_sha256,
    )

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_b_manifest) if args.mode_b_manifest else _default_mode_b_manifest(
        args.processed_root,
        args.split,
        args.views,
    )
    candidate_path = Path(args.mode_b_candidate_path) if args.mode_b_candidate_path else _default_mode_b_candidates(
        args.processed_root,
        args.split,
    )

    print("[mode_b] loading dataset", flush=True)
    dataset = WaymoEditDataset(
        processed_root=args.processed_root,
        transfer_root=args.transfer_root,
        raw_root=args.raw_root,
        asset_root=args.asset_root,
        split=args.split,
        manifest_path=manifest_path,
        candidate_path=candidate_path,
        views=args.views,
        mode=args.dataset_mode,
        sequence_length=args.sequence_length,
        sample_window=args.sequence_length,
        metric_box_mapping_mode=args.metric_box_mapping_mode,
        scene_gauge_path=args.scene_gauge_path,
        expected_scene_gauge_dggt_sha256=scene_gauge_dggt_sha256,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mode_b] loading model on {device}", flush=True)
    model = _load_model(args.ckpt_path, device)

    dataset_len = len(dataset)
    if args.index is None:
        start_idx = max(0, int(args.start))
        end_idx = dataset_len if args.end is None else min(int(args.end), dataset_len)
        if end_idx < start_idx:
            raise ValueError(f"Invalid range: start={start_idx} end={end_idx} for dataset length {dataset_len}")
        indices = list(range(start_idx, end_idx))
        multi_sample = True
    else:
        if args.index < 0 or args.index >= dataset_len:
            raise IndexError(f"--index {args.index} is out of range for dataset length {dataset_len}")
        indices = [int(args.index)]
        multi_sample = False

    all_summaries = []
    for position, sample_index in enumerate(indices, start=1):
        sample_output_dir = root_output_dir / f"sample_{sample_index:06d}" if multi_sample else root_output_dir
        print(f"[mode_b] [{position}/{len(indices)}] sample {sample_index} -> {sample_output_dir}", flush=True)
        summary = _run_one_sample(
            args=args,
            dataset=dataset,
            model=model,
            device=device,
            sample_index=sample_index,
            output_dir=sample_output_dir,
            manifest_path=manifest_path,
            candidate_path=candidate_path,
        )
        all_summaries.append(
            {
                "sample_index": int(sample_index),
                "output_dir": str(sample_output_dir),
                "scene_name": summary.get("scene_name"),
                "clip_name": summary.get("clip_name"),
                "rng_seed": summary.get("rng_seed"),
                "num_imagined_objects": summary.get("num_imagined_objects"),
                "status": summary.get("status", "completed"),
                "skip_reason": summary.get("skip_reason"),
                "mode_b_source": summary.get("mode_b_source"),
                "mode_b_in_mode_a_views1": summary.get("mode_b_in_mode_a_views1"),
                "mode_b_in_mode_a_views3": summary.get("mode_b_in_mode_a_views3"),
            }
        )
    if multi_sample:
        run_summary = {
            "num_samples": len(all_summaries),
            "range_start": int(indices[0]) if indices else None,
            "range_end": int(indices[-1] + 1) if indices else None,
            "sample_outputs": all_summaries,
            "planner_seed_rule": "base_planner_seed + 1009 * dataset_index",
            "base_planner_seed": int(args.planner_seed),
            "num_completed": sum(1 for item in all_summaries if item.get("status") in {"completed", "plan_only"}),
            "num_skipped": sum(1 for item in all_summaries if str(item.get("status", "")).startswith("skipped_")),
        }
        _write_json(root_output_dir / "all_samples_summary.json", run_summary)
        print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
