#!/usr/bin/env python3
"""Render a reproducible visual audit of the layout-v2 conditioning contract.

Each MP4 contains four synchronized panels:

1. ground-truth front-camera video;
2. GT plus the production HD-map raster (M), excluding lane centers/driveways
   and clipped at 120 m optical depth;
3. GT plus production actor coverage, 2-D boxes and cached cuboid projection (G);
4. GT plus canonical appearance references and their explicit A -> G bindings.

The script also independently re-reads the emitted tensors and fails closed on
projection, bbox, metric-map, or appearance-binding inconsistencies.  It does
not load a checkpoint and does not run P12/T59.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset import WaymoOpenDataset, load_and_preprocess_images
from datasets.tools.hdmap_schema import (
    MAP_METRIC_RESERVED_ZERO_GROUPS,
    RASTER_RESERVED_ZERO_CHANNELS,
)
from dggt.utils.appearance_binding_condition import appearance_alpha_to_patch_mask
from dggt.utils.layout_raster import STATIC_FAR_PLANE_M, dequantize_layout_raster


BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
MAP_LAYERS: tuple[tuple[int, str, tuple[int, int, int]], ...] = (
    (1, "road-line", (255, 220, 30)),
    (2, "road-edge", (255, 70, 220)),
    (3, "crosswalk", (20, 220, 255)),
    (4, "speed-bump", (255, 120, 20)),
    (5, "stop-sign", (255, 40, 40)),
)
ACTOR_COLORS: tuple[tuple[int, int, int], ...] = (
    (50, 170, 255),
    (255, 135, 40),
    (205, 80, 255),
)
ACTOR_NAMES: tuple[str, ...] = ("vehicle", "pedestrian", "cyclist")
BINDING_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 255, 20),
    (20, 255, 255),
    (255, 80, 80),
    (80, 255, 120),
    (255, 110, 255),
)


def _tensor_scalar(value: Any) -> int:
    return int(value.item()) if torch.is_tensor(value) else int(value)


def _rgb_image(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 3 or int(value.shape[0]) != 3:
        raise ValueError(f"expected [3,H,W] image, got {tuple(value.shape)}")
    image = value.detach().cpu()
    if image.dtype == torch.uint8:
        array = image.permute(1, 2, 0).numpy()
    else:
        array = (
            image.float().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
        ).round().astype(np.uint8)
    return np.ascontiguousarray(array)


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    opacity: float,
) -> np.ndarray:
    alpha = np.clip(mask.astype(np.float32) * float(opacity), 0.0, 1.0)[..., None]
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(
        image.astype(np.float32) * (1.0 - alpha) + color_array * alpha,
        0.0,
        255.0,
    ).astype(np.uint8)


def _put_title(image: np.ndarray, title: str, detail: str = "") -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, dst=image)
    cv2.putText(
        image,
        title,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if detail:
        width = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(
            image,
            detail,
            (max(8, image.shape[1] - width - 8), 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )


def _map_panel(gt: np.ndarray, raster: np.ndarray) -> np.ndarray:
    height, width = gt.shape[:2]
    result = gt.copy()
    for channel, _, color in MAP_LAYERS:
        mask = cv2.resize(
            raster[channel].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        result = _blend_mask(result, mask, color, opacity=0.72)
    legend_x, legend_y = 8, height - 9
    for _, name, color in MAP_LAYERS:
        cv2.circle(result, (legend_x, legend_y - 4), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            result,
            name,
            (legend_x + 6, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        legend_x += cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1
        )[0][0] + 15
    return result


def _binding_by_geometry(payload: dict[str, Any]) -> dict[int, int]:
    output: dict[int, int] = {}
    indices = payload["appearance_geometry_idx"]
    valid = payload["appearance_binding_valid"]
    for appearance_slot in valid.nonzero(as_tuple=False).flatten().tolist():
        output[int(indices[appearance_slot])] = int(appearance_slot)
    return output


def _draw_actor_geometry(
    image: np.ndarray,
    payload: dict[str, Any],
    frame_index: int,
    *,
    binding_only: bool,
) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    raster = payload["_dequantized_layout"][frame_index].cpu().numpy()
    for class_id, color in enumerate(ACTOR_COLORS):
        mask = cv2.resize(
            raster[22 + class_id].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        result = _blend_mask(
            result,
            mask,
            color,
            opacity=0.16 if binding_only else 0.38,
        )

    bound = _binding_by_geometry(payload)
    in_frustum = payload["projected_actor_geometry_in_frustum"][:, frame_index]
    track_valid = payload["actor_geometry_track_valid"][:, frame_index]
    bboxes = payload["projected_actor_geometry_bbox_patch"][:, frame_index]
    uv = payload["projected_actor_geometry_uv_corners"][:, frame_index]
    camera_corners = payload["projected_actor_geometry_corners_camera"][:, frame_index]
    classes = payload["actor_geometry_class_id"]
    log_depth = payload["projected_actor_geometry_log_z_w"][:, frame_index]
    raw_keys = payload["actor_geometry_raw_track_key"]
    bbox_scale = np.asarray((width / 37.0, height / 25.0) * 2, dtype=np.float32)
    uv_scale = np.asarray((width, height), dtype=np.float32)

    for slot in (in_frustum & track_valid).nonzero(as_tuple=False).flatten().tolist():
        appearance_slot = bound.get(int(slot))
        if binding_only and appearance_slot is None:
            continue
        class_id = int(classes[slot])
        base_color = ACTOR_COLORS[class_id] if 0 <= class_id < 3 else (220, 220, 220)
        color = (
            BINDING_COLORS[appearance_slot]
            if appearance_slot is not None
            else base_color
        )
        thickness = 3 if appearance_slot is not None else 1
        xyxy = (bboxes[slot].cpu().numpy() * bbox_scale).round().astype(np.int32)
        x0, y0, x1, y1 = xyxy.tolist()
        cv2.rectangle(result, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

        points_uv = uv[slot].cpu().numpy()
        depths = camera_corners[slot, :, 2].cpu().numpy()
        for first, second in BOX_EDGES:
            if (
                depths[first] < 0.5
                or depths[second] < 0.5
                or not np.isfinite(points_uv[[first, second]]).all()
                or np.abs(points_uv[[first, second]]).max() > 8.0
            ):
                continue
            endpoints = np.rint(points_uv[[first, second]] * uv_scale).astype(np.int32)
            visible, clipped_first, clipped_second = cv2.clipLine(
                (0, 0, width, height), tuple(endpoints[0]), tuple(endpoints[1])
            )
            if visible:
                cv2.line(
                    result,
                    clipped_first,
                    clipped_second,
                    color,
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )

        class_name = ACTOR_NAMES[class_id] if 0 <= class_id < 3 else f"class-{class_id}"
        depth = math.exp(float(log_depth[slot]))
        binding_label = f" A{appearance_slot}->" if appearance_slot is not None else " "
        label = f"{binding_label}G{slot} {class_name} {depth:.1f}m"
        raw_key = str(raw_keys[slot])
        if raw_key:
            label += f" [{raw_key[-10:]}]"
        text_y = max(40, min(height - 5, y0 - 4))
        cv2.putText(
            result,
            label,
            (max(2, x0), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def _appearance_thumbnail(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    *,
    output_hw: tuple[int, int] = (62, 92),
) -> np.ndarray:
    rgb_np = np.moveaxis(rgb.detach().cpu().float().numpy(), 0, -1)
    alpha_np = alpha.detach().cpu().float().numpy()[0]
    foreground = alpha_np > 0.01
    out_h, out_w = output_hw
    output = np.full((out_h, out_w, 3), 18, dtype=np.uint8)
    if not bool(foreground.any()):
        return output
    ys, xs = np.where(foreground)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(1, int(round((y1 - y0) * 0.08)))
    pad_x = max(1, int(round((x1 - x0) * 0.08)))
    y0, y1 = max(0, y0 - pad_y), min(rgb_np.shape[0], y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(rgb_np.shape[1], x1 + pad_x)
    crop_rgb = rgb_np[y0:y1, x0:x1]
    crop_alpha = alpha_np[y0:y1, x0:x1]
    scale = min((out_h - 5) / crop_rgb.shape[0], (out_w - 5) / crop_rgb.shape[1])
    new_h = max(1, int(round(crop_rgb.shape[0] * scale)))
    new_w = max(1, int(round(crop_rgb.shape[1] * scale)))
    resized_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(crop_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = (out_h - new_h) // 2, (out_w - new_w) // 2
    patch = output[top : top + new_h, left : left + new_w].astype(np.float32)
    blend = resized_alpha[..., None].clip(0.0, 1.0)
    patch[:] = np.clip(
        patch * (1.0 - blend) + resized_rgb * 255.0 * blend,
        0.0,
        255.0,
    )
    return output


def _appearance_panel(
    gt: np.ndarray,
    payload: dict[str, Any],
    frame_index: int,
) -> np.ndarray:
    result = _draw_actor_geometry(gt, payload, frame_index, binding_only=True)
    valid = payload["appearance_binding_valid"]
    indices = payload["appearance_geometry_idx"]
    rgb = payload["appearance_reference_rgb"]
    alpha = payload["appearance_reference_alpha"]
    for slot in valid.nonzero(as_tuple=False).flatten().tolist():
        thumb = _appearance_thumbnail(rgb[slot], alpha[slot])
        x0 = 5 + int(slot) * 101
        y0 = 34
        if x0 + thumb.shape[1] > result.shape[1]:
            break
        result[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb
        color = BINDING_COLORS[int(slot)]
        cv2.rectangle(
            result,
            (x0, y0),
            (x0 + thumb.shape[1], y0 + thumb.shape[0]),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            f"A{slot} -> G{int(indices[slot])}",
            (x0 + 3, y0 + thumb.shape[0] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    if not bool(valid.any()):
        cv2.putText(
            result,
            "A=NULL (geometry G is intentionally retained)",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 40),
            1,
            cv2.LINE_AA,
        )
    return result


def _audit_payload(
    payload: dict[str, Any],
    camera: Any,
    *,
    canvas_hw: tuple[int, int],
) -> dict[str, Any]:
    """Strict tensor-level audit independent of the visualization drawing."""

    height, width = canvas_hw
    gh, gw = 25, 37
    if float(payload["static_far_plane_m"]) != STATIC_FAR_PLANE_M:
        raise AssertionError(
            f"static_far_plane_m must be {STATIC_FAR_PLANE_M:g}"
        )
    raster = payload["layout_raster"]
    if raster.dtype != torch.uint8 or tuple(raster.shape[1:]) != (33, 100, 148):
        raise AssertionError(f"invalid layout raster contract: {raster.dtype} {tuple(raster.shape)}")
    restored = dequantize_layout_raster(raster)
    if not bool(torch.isfinite(restored).all()):
        raise AssertionError("dequantized raster contains NaN/Inf")
    excluded_raster_max_abs = float(
        restored[:, RASTER_RESERVED_ZERO_CHANNELS].abs().max()
    )
    if excluded_raster_max_abs != 0.0:
        raise AssertionError(
            "excluded static-map raster slots are not exactly zero: "
            f"{excluded_raster_max_abs}"
        )

    corners = payload["projected_actor_geometry_corners_camera"].double()
    cached_uv = payload["projected_actor_geometry_uv_corners"].double()
    projected_h = torch.einsum(
        "sij,ksqj->ksqi", camera.normalized_canvas_intrinsics[0], corners
    )
    safe_z = torch.where(
        projected_h[..., 2:3].abs().gt(torch.finfo(torch.float64).eps),
        projected_h[..., 2:3],
        torch.ones_like(projected_h[..., 2:3]),
    )
    expected_uv = projected_h[..., :2] / safe_z
    valid_corners = (
        payload["projected_actor_geometry_valid"][..., None]
        & corners[..., 2].ge(0.5)
        & expected_uv.isfinite().all(dim=-1)
        & expected_uv.abs().lt(8.0).all(dim=-1)
    )
    if bool(valid_corners.any()):
        pixel_scale = cached_uv.new_tensor((float(width), float(height)))
        uv_error_px = (
            (cached_uv - expected_uv).abs() * pixel_scale
        )[valid_corners].max().item()
    else:
        uv_error_px = 0.0
    if uv_error_px > 0.005:
        raise AssertionError(f"cached corner projection error {uv_error_px:.6f}px > 0.005px")

    in_frustum = payload["projected_actor_geometry_in_frustum"]
    fully_supported = in_frustum & corners[..., 2].ge(0.5).all(dim=-1)
    bbox_error_patch = 0.0
    if bool(fully_supported.any()):
        clamped_uv = expected_uv.clamp(0.0, 1.0)
        expected_bbox = torch.stack(
            (
                clamped_uv[..., 0].amin(dim=-1) * gw,
                clamped_uv[..., 1].amin(dim=-1) * gh,
                clamped_uv[..., 0].amax(dim=-1) * gw,
                clamped_uv[..., 1].amax(dim=-1) * gh,
            ),
            dim=-1,
        )
        actual_bbox = payload["projected_actor_geometry_bbox_patch"].double()
        bbox_error_patch = (actual_bbox - expected_bbox).abs()[fully_supported].max().item()
    if bbox_error_patch > 2.0e-5:
        raise AssertionError(
            f"bbox/cache mismatch {bbox_error_patch:.8f} patch > 2e-5"
        )

    center_z = corners.mean(dim=-2)[..., 2]
    center_valid = payload["projected_actor_geometry_valid"] & center_z.ge(0.5)
    center_depth_error_m = 0.0
    if bool(center_valid.any()):
        cached_depth = torch.exp(
            payload["projected_actor_geometry_log_z_w"].double()
        )
        center_depth_error_m = (cached_depth - center_z).abs()[center_valid].max().item()
    if center_depth_error_m > 2.0e-4:
        raise AssertionError(
            f"center log-depth mismatch {center_depth_error_m:.8f}m > 2e-4m"
        )
    supported = in_frustum
    patch_weight = payload["projected_actor_geometry_patch_weight"]
    if bool((supported & patch_weight.sum(dim=-1).le(0.0)).any()):
        raise AssertionError("an in-frustum actor has empty patch support")

    metric = payload["map_metric"]
    if metric.dtype != torch.float32 or tuple(metric.shape[-2:]) != (5, 4):
        raise AssertionError(f"invalid map_metric contract {metric.dtype} {tuple(metric.shape)}")
    metric_valid = metric[..., 3].eq(1.0)
    if not bool(((metric[..., 3] == 0.0) | metric_valid).all()):
        raise AssertionError("map_metric valid flag is not binary")
    patch_index = torch.arange(gh * gw, dtype=torch.float32)
    patch_x = torch.remainder(patch_index, gw)[None, :, None]
    patch_y = torch.div(patch_index, gw, rounding_mode="floor")[None, :, None]
    u_patch = metric[..., 0] * gw
    v_patch = metric[..., 1] * gh
    metric_inside = (
        u_patch.ge(patch_x - 2.0e-6)
        & u_patch.le(patch_x + 1.0 + 2.0e-6)
        & v_patch.ge(patch_y - 2.0e-6)
        & v_patch.le(patch_y + 1.0 + 2.0e-6)
    )
    if bool((metric_valid & ~metric_inside).any()):
        raise AssertionError("a valid map_metric representative lies outside its patch")
    if not bool(torch.isfinite(metric).all()):
        raise AssertionError("map_metric contains NaN/Inf")
    lane_metric_max_abs = float(
        metric[..., MAP_METRIC_RESERVED_ZERO_GROUPS, :].abs().max()
    )
    if lane_metric_max_abs != 0.0:
        raise AssertionError(
            "excluded lane-centerline map_metric group is not exactly zero: "
            f"{lane_metric_max_abs}"
        )

    binding_valid = payload["appearance_binding_valid"]
    geometry_idx = payload["appearance_geometry_idx"]
    active_idx = geometry_idx[binding_valid]
    slot_valid = payload["actor_geometry_slot_valid"]
    if active_idx.numel():
        if bool((active_idx < 0).any() or (active_idx >= slot_valid.numel()).any()):
            raise AssertionError("A.geometry_idx is out of G range")
        if active_idx.unique().numel() != active_idx.numel():
            raise AssertionError("two A rows bind the same G slot")
        if not bool(slot_valid[active_idx].all()):
            raise AssertionError("A binds a padded G slot")
        if not torch.equal(
            payload["appearance_class_id"][binding_valid],
            payload["actor_geometry_class_id"][active_idx],
        ):
            raise AssertionError("A class does not match bound G class")
        alpha_support = appearance_alpha_to_patch_mask(
            payload["appearance_reference_alpha"][binding_valid],
            (gh, gw),
        ).sum(dim=-1)
        if bool(alpha_support.lt(4).any()):
            raise AssertionError("a valid A reference has fewer than four alpha patches")
    if bool((~binding_valid & geometry_idx.ne(-1)).any()):
        raise AssertionError("an invalid A row has a non-sentinel geometry_idx")

    return {
        "actor_slots": int(slot_valid.sum()),
        "actor_visible_pairs": int(in_frustum.sum()),
        "appearance_bindings": int(binding_valid.sum()),
        "appearance_candidates": _tensor_scalar(payload["appearance_candidate_count"]),
        "map_metric_valid_rows": int(metric_valid.sum()),
        "static_far_plane_m": float(payload["static_far_plane_m"]),
        "excluded_static_raster_max_abs": excluded_raster_max_abs,
        "excluded_lane_metric_max_abs": lane_metric_max_abs,
        "cached_uv_max_error_px": uv_error_px,
        "bbox_max_error_patch": bbox_error_patch,
        "center_depth_max_error_m": center_depth_error_m,
        "layout_mode": _tensor_scalar(payload["actor_geometry_layout_mode"]),
        "actor_overflow": _tensor_scalar(payload["actor_geometry_overflow"]),
    }


def _sample_indices(dataset: WaymoOpenDataset) -> list[int]:
    first_per_scene: dict[int, int] = {}
    for sample_index, entry in enumerate(dataset.trunk_major_index):
        scene_index = int(entry[0])
        first_per_scene.setdefault(scene_index, sample_index)
    ordered_scene_indices = sorted(first_per_scene)
    if not ordered_scene_indices:
        return []
    # Interleave the validation set spatially in index space instead of taking
    # twenty neighboring segments, then retain the rest as deterministic
    # fallbacks when an A-present clip is requested.
    stride = max(1, len(ordered_scene_indices) // 20)
    primary = ordered_scene_indices[::stride]
    remainder = [index for index in ordered_scene_indices if index not in set(primary)]
    return [first_per_scene[index] for index in primary + remainder]


def _render_clip(
    frames: torch.Tensor,
    payload: dict[str, Any],
    *,
    scene_name: str,
    start_frame: int,
    output_path: Path,
    fps: float,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width * 4, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open MP4 writer for {output_path}")
    payload["_dequantized_layout"] = dequantize_layout_raster(
        payload["layout_raster"]
    )
    actor_count = int(payload["actor_geometry_slot_valid"].sum())
    appearance_count = int(payload["appearance_binding_valid"].sum())
    layout_mode = {0: "NULL", 1: "EMPTY", 2: "PARTIAL", 3: "FULL"}.get(
        _tensor_scalar(payload["actor_geometry_layout_mode"]), "?"
    )
    middle_frame: np.ndarray | None = None
    try:
        for frame_index in range(int(frames.shape[0])):
            gt = _rgb_image(frames[frame_index])
            raster = payload["_dequantized_layout"][frame_index].cpu().numpy()
            gt_panel = gt.copy()
            map_panel = _map_panel(gt, raster)
            actor_panel = _draw_actor_geometry(
                gt, payload, frame_index, binding_only=False
            )
            appearance_panel = _appearance_panel(gt, payload, frame_index)
            detail = f"scene {scene_name} frame {start_frame + frame_index:03d}"
            _put_title(gt_panel, "GT front-camera video", detail)
            _put_title(
                map_panel,
                "M: no center/driveway; far=120m",
                detail,
            )
            _put_title(
                actor_panel,
                "G: actor coverage + cached cuboids",
                f"{layout_mode} | actors={actor_count}",
            )
            _put_title(
                appearance_panel,
                "A -> G: canonical identity binding",
                f"bindings={appearance_count}",
            )
            composite = np.concatenate(
                (gt_panel, map_panel, actor_panel, appearance_panel), axis=1
            )
            writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            if frame_index == int(frames.shape[0]) // 2:
                middle_frame = composite.copy()
    finally:
        writer.release()
        payload.pop("_dequantized_layout", None)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"empty MP4 output: {output_path}")
    if middle_frame is None:
        raise RuntimeError("clip contained no frames")
    return middle_frame


def _write_html(output_dir: Path, manifest: dict[str, Any]) -> None:
    cards = []
    for item in manifest["clips"]:
        video = html.escape(item["video"])
        scene = html.escape(item["scene"])
        metrics = html.escape(json.dumps(item["audit"], ensure_ascii=False))
        cards.append(
            f'<section><h2>{scene} · frames {item["start_frame"]:03d}–'
            f'{item["end_frame"]:03d}</h2><video controls preload="metadata" '
            f'src="{video}"></video><pre>{metrics}</pre></section>'
        )
    document = """<!doctype html><meta charset="utf-8">
<title>layout-v2 visual audit</title>
<style>
body{background:#111;color:#eee;font:14px system-ui;margin:20px}h1{margin-bottom:4px}
section{margin:28px 0;padding:14px;background:#1c1c1c;border-radius:8px}
video{width:100%;max-width:1600px;background:#000}pre{white-space:pre-wrap;color:#bde}
</style><h1>layout-v2 GT / M / G / A visual audit</h1>
<p>Every clip is rendered from the production online projection tensors. Waymo lane
centerlines and driveway polygons are excluded from both the model condition and this
overlay; static line and polygon geometry is clipped at 120 m optical depth. No checkpoint
is used.</p>
""" + "\n".join(cards)
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/data/disk2/lyy_dataset/waymo_processed_dggt/validation"),
    )
    parser.add_argument(
        "--hdmap-root",
        type=Path,
        default=Path("/data/disk2/lyy_dataset/waymo_processed_dggt/validation_hdmap"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/layout_condition_visual_audit_2026-08-14"),
    )
    parser.add_argument("--num-clips", type=int, default=20)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--layout-max-actors", type=int, default=96)
    parser.add_argument(
        "--static-far-plane-m",
        type=float,
        default=STATIC_FAR_PLANE_M,
        help="Frozen camera optical-depth far plane for static HD-map geometry.",
    )
    parser.add_argument(
        "--require-appearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="select A-present clips so every visualized A->G binding is inspectable",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.num_clips <= 0 or args.frames <= 0 or args.fps <= 0.0:
        raise ValueError("num-clips, frames, and fps must be positive")
    if float(args.static_far_plane_m) != STATIC_FAR_PLANE_M:
        raise ValueError(
            f"--static-far-plane-m is frozen to {STATIC_FAR_PLANE_M:g}"
        )
    image_root = args.image_root.resolve()
    hdmap_root = args.hdmap_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_names = sorted(
        path.name
        for path in image_root.iterdir()
        if path.is_dir() and (hdmap_root / path.name / "hdmap.npz").is_file()
    )
    dataset = WaymoOpenDataset(
        image_dir=str(image_root),
        scene_names=scene_names,
        sequence_length=int(args.frames),
        start_idx=0,
        mode=1,
        views=1,
        caption_root=None,
        hdmap_root=str(hdmap_root),
        layout_max_actors=int(args.layout_max_actors),
        static_far_plane_m=float(args.static_far_plane_m),
        trunk_major_samples=True,
        trunk_major_window_offsets=(0,),
        return_full_dggt_context=False,
        load_dynamic_masks=False,
        image_output_dtype="uint8",
    )
    manifest: dict[str, Any] = {
        "schema": "layout_condition_visual_audit_v3_no_lane_no_driveway_far120m",
        "image_root": str(image_root),
        "hdmap_root": str(hdmap_root),
        "frames_per_clip": int(args.frames),
        "fps": float(args.fps),
        "require_appearance": bool(args.require_appearance),
        "static_far_plane_m": float(args.static_far_plane_m),
        "excluded_map_source_classes": ["lane", "driveway"],
        "clips": [],
    }
    overview_frames: list[np.ndarray] = []
    attempted = 0
    started = time.perf_counter()
    for sample_index in _sample_indices(dataset):
        if len(manifest["clips"]) >= int(args.num_clips):
            break
        attempted += 1
        entry = dataset.trunk_major_index[sample_index]
        scene_index, trunk_index = int(entry[0]), int(entry[1])
        scene_name = str(dataset.scenes[scene_index])
        total_frames = len(dataset.image_paths[scene_index])
        offset = int(entry[2]) if len(entry) == 3 else 0
        start_frame = dataset._fixed_start_in_trunk(
            total_frames, trunk_index, window_offset=offset
        )
        frame_indices = list(range(start_frame, start_frame + int(args.frames)))
        (
            camera_to_world,
            intrinsics,
            raw_image_size_hw,
            trajectory_anchor_to_world,
            _,
        ) = dataset._load_front_waymo_camera_gt(scene_index, frame_indices)
        scene = dataset._load_hdmap_scene(scene_index)
        camera = dataset._camera_spec_from_absolute(
            scene,
            frame_indices,
            camera_to_world,
            trajectory_anchor_to_world,
            intrinsics,
            raw_image_size_hw,
        )
        payload_started = time.perf_counter()
        payload = dataset._build_layout_payload_from_camera_gt(
            scene_index,
            frame_indices,
            camera_to_world,
            intrinsics,
            raw_image_size_hw,
            trajectory_anchor_to_world,
            deterministic_reference=True,
        )
        if bool(args.require_appearance) and not bool(
            payload["appearance_binding_valid"].any()
        ):
            continue
        frames = load_and_preprocess_images(
            [dataset.image_paths[scene_index][index] for index in frame_indices],
            output_dtype="uint8",
        )
        audit = _audit_payload(
            payload,
            camera,
            canvas_hw=(int(frames.shape[-2]), int(frames.shape[-1])),
        )
        ordinal = len(manifest["clips"]) + 1
        filename = (
            f"{ordinal:02d}_scene{scene_name}_trunk{trunk_index:02d}_"
            f"frames{start_frame:03d}-{frame_indices[-1]:03d}.mp4"
        )
        overview = _render_clip(
            frames,
            payload,
            scene_name=scene_name,
            start_frame=start_frame,
            output_path=output_dir / filename,
            fps=float(args.fps),
        )
        overview_frames.append(
            cv2.resize(overview, (1036, 175), interpolation=cv2.INTER_AREA)
        )
        audit["projection_and_binding_seconds"] = time.perf_counter() - payload_started
        manifest["clips"].append(
            {
                "scene": scene_name,
                "trunk": trunk_index,
                "start_frame": start_frame,
                "end_frame": frame_indices[-1],
                "video": filename,
                "audit": audit,
            }
        )
        print(
            f"[{ordinal:02d}/{args.num_clips:02d}] {filename} "
            f"G={audit['actor_slots']} A={audit['appearance_bindings']} "
            f"uv_err={audit['cached_uv_max_error_px']:.6f}px",
            flush=True,
        )

    if len(manifest["clips"]) != int(args.num_clips):
        raise RuntimeError(
            f"rendered {len(manifest['clips'])}/{args.num_clips} clips after "
            f"checking {attempted} distinct scenes"
        )
    manifest["attempted_samples"] = attempted
    manifest["wall_seconds"] = time.perf_counter() - started
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_html(output_dir, manifest)
    if overview_frames:
        overview = np.concatenate(overview_frames, axis=0)
        cv2.imwrite(
            str(output_dir / "overview.jpg"),
            cv2.cvtColor(overview, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    print(
        f"Wrote {len(manifest['clips'])} clips and strict audit manifest to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
