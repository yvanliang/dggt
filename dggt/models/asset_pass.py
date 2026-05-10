"""Asset Aggregator Pass for isolated asset renders.

This module bridges the current `WaymoEditDataset` sample structure with the
FlowDGGT asset-pass design:

* place each editable asset at its per-frame fitted DGGT pose
* render isolated RGB/alpha sequences on the model image grid
* run the shared aggregator per object
* extract patch-token LUTs
* annotate per-image Gaussian pointers `(object_id, view_n, patch_idx, visible)`

The current repository flattens images as `frame-major, view-minor`, so all
sequence-facing outputs follow that same order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from dggt.models.gaussian_pointers import GaussianPointers, SRC_KIND_ASSET
from dggt.utils.gaussian_edit import (
    Sim3Transform,
    empty_gaussian_dict,
)
from dggt.utils.tokens import select_patch_pyramid


DEFAULT_LEVELS = (4, 11, 17, 23)


@dataclass
class AssetPassResult:
    patch_grid: tuple[int, int]
    patch_start_idx: int
    object_keys: list[int]
    cameras_waymo: dict[str, torch.Tensor]
    F_g_lut_asset: dict[int, list[torch.Tensor]]
    ptr_asset: dict[int, list[GaussianPointers]]
    G_asset_dggt: dict[int, list[dict[str, torch.Tensor]]] | None
    I_asset: dict[int, torch.Tensor]
    A_asset: dict[int, torch.Tensor]
    G_asset_waymo: dict[int, list[dict[str, torch.Tensor]]] | None = None
    asset_pass_space: str = ""
    fit_metrics: dict[int, list[dict[str, Any]]] | None = None

    def __post_init__(self) -> None:
        if self.G_asset_waymo is None:
            self.G_asset_waymo = {}


def compute_runtime_patch_grid(model_hw: tuple[int, int], patch_size: int = 14) -> tuple[int, int]:
    """Infer the runtime patch grid from the actual model image resolution."""
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    if model_h <= 0 or model_w <= 0:
        raise ValueError(f"model_hw must be positive, got {model_hw}")
    if model_h % patch_size != 0 or model_w % patch_size != 0:
        raise ValueError(
            f"model_hw={model_hw} is incompatible with patch_size={patch_size}"
        )
    return model_h // patch_size, model_w // patch_size


def compute_model_resize_geometry(
    raw_hw: tuple[int, int],
    model_hw: tuple[int, int],
    target_width: int = 518,
) -> dict[str, int | float]:
    """Mirror `WaymoEditDataset` resize/crop/pad logic for camera intrinsics."""
    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    if raw_h <= 0 or raw_w <= 0:
        raise ValueError(f"raw_hw must be positive, got {raw_hw}")

    new_w = int(target_width)
    new_h = round(raw_h * (new_w / float(raw_w)) / 14.0) * 14
    crop_top = max((new_h - target_width) // 2, 0) if new_h > target_width else 0
    out_h = target_width if new_h > target_width else new_h
    out_w = new_w
    if model_h < out_h or model_w < out_w:
        raise ValueError(
            f"model_hw {model_hw} is smaller than resized image {(out_h, out_w)}"
        )
    pad_top = max((model_h - out_h) // 2, 0)
    pad_left = max((model_w - out_w) // 2, 0)
    return {
        "scale_x": new_w / float(raw_w),
        "scale_y": new_h / float(raw_h),
        "crop_top": int(crop_top),
        "out_h": int(out_h),
        "out_w": int(out_w),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
    }


def compute_model_intrinsics(
    K_raw: torch.Tensor,
    raw_hw: tuple[int, int],
    model_hw: tuple[int, int],
    target_width: int = 518,
) -> torch.Tensor:
    """Project raw-image intrinsics onto the model's resized/cropped image grid."""
    K_raw = K_raw.detach().cpu().float()
    geom = compute_model_resize_geometry(raw_hw, model_hw, target_width=target_width)
    K = K_raw.clone()
    K[0, 0] = K_raw[0, 0] * float(geom["scale_x"])
    K[1, 1] = K_raw[1, 1] * float(geom["scale_y"])
    K[0, 2] = K_raw[0, 2] * float(geom["scale_x"]) + float(geom["pad_left"])
    K[1, 2] = (
        K_raw[1, 2] * float(geom["scale_y"])
        - float(geom["crop_top"])
        + float(geom["pad_top"])
    )
    return K


def apply_pointer_fallback(
    patch_idx_per_image: Sequence[torch.Tensor],
    visible_mask_per_image: Sequence[torch.Tensor],
    image_to_frame: torch.Tensor,
    image_to_view: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Fallback invisible pointers to the nearest visible source image.

    We keep `visible_mask=False` for the current image, but redirect `(view_n,
    patch_idx)` to the nearest visible image so later feature lookup still has a
    stable semantic source.
    """
    num_images = len(patch_idx_per_image)
    if len(visible_mask_per_image) != num_images:
        raise ValueError("patch_idx_per_image and visible_mask_per_image must match in length")

    image_to_frame = image_to_frame.detach().cpu().long()
    image_to_view = image_to_view.detach().cpu().long()

    adjusted_view_n: list[torch.Tensor] = []
    adjusted_patch_idx: list[torch.Tensor] = []

    for image_idx in range(num_images):
        patch_idx = patch_idx_per_image[image_idx].detach().cpu().long().clone()
        visible = visible_mask_per_image[image_idx].detach().cpu().bool()
        view_n = torch.full_like(patch_idx, int(image_idx))
        if patch_idx.numel() == 0:
            adjusted_view_n.append(view_n)
            adjusted_patch_idx.append(patch_idx)
            continue

        frame_i = int(image_to_frame[image_idx].item())
        view_i = int(image_to_view[image_idx].item())
        best_rank = torch.full_like(patch_idx, torch.iinfo(torch.long).max)
        best_image_idx = torch.full_like(patch_idx, -1)
        for cand_idx in range(num_images):
            cand_visible = visible_mask_per_image[cand_idx].detach().cpu().bool()
            cand_patch = patch_idx_per_image[cand_idx].detach().cpu().long()
            if cand_visible.shape[0] != patch_idx.shape[0] or cand_patch.shape[0] != patch_idx.shape[0]:
                continue
            frame_c = int(image_to_frame[cand_idx].item())
            view_c = int(image_to_view[cand_idx].item())
            rank = (
                abs(frame_i - frame_c) * 1_000_000
                + (0 if view_i == view_c else 1) * 10_000
                + abs(image_idx - cand_idx) * 100
                + cand_idx
            )
            update = cand_visible & (rank < best_rank)
            best_rank[update] = int(rank)
            best_image_idx[update] = int(cand_idx)

        fallback_mask = (~visible) & (best_image_idx >= 0)
        if bool(fallback_mask.any().item()):
            fallback_ids = torch.nonzero(fallback_mask, as_tuple=False).flatten()
            fallback_images = best_image_idx[fallback_ids]
            patch_bank = torch.stack(
                [
                    patch_idx_per_image[cand_idx].detach().cpu().long()
                    if patch_idx_per_image[cand_idx].shape[0] == patch_idx.shape[0]
                    else torch.zeros_like(patch_idx)
                    for cand_idx in range(num_images)
                ],
                dim=0,
            )
            view_n[fallback_ids] = fallback_images
            patch_idx[fallback_ids] = patch_bank[fallback_images, fallback_ids]
        adjusted_view_n.append(view_n)
        adjusted_patch_idx.append(patch_idx)

    return adjusted_view_n, adjusted_patch_idx


class AssetAggregatorPass(nn.Module):
    def __init__(
        self,
        aggregator: nn.Module,
        levels: Sequence[int] = DEFAULT_LEVELS,
        patch_size: int = 14,
        depth_tol: float = 0.05,
        alpha_thresh: float = 1e-3,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.levels = tuple(int(level) for level in levels)
        self.patch_size = int(patch_size)
        self.depth_tol = float(depth_tol)
        self.alpha_thresh = float(alpha_thresh)

    def forward(
        self,
        sample: dict[str, Any],
        selected_object_slots: Sequence[int] | None = None,
        alignment: Sim3Transform | None = None, # Deprecated
        asset_cache: dict[str, dict[str, torch.Tensor]] | None = None, # Deprecated
        occlusion_test: bool = True,
        aggregator_batch_size: int = 0,
        localized_objects: Sequence[Any] | None = None,
        cameras_dggt: dict[str, torch.Tensor] | None = None,
        render_space: str = "dggt_fitted", # Deprecated
        collect_fit_metrics: bool = False,
    ) -> AssetPassResult:
        images_clean = sample["images_clean"]
        if images_clean.dim() != 4:
            raise ValueError(
                f"sample['images_clean'] must be [S,3,H,W], got {tuple(images_clean.shape)}"
            )

        device = self._infer_device(images_clean)
        model_hw = (int(images_clean.shape[-2]), int(images_clean.shape[-1]))
        patch_grid = compute_runtime_patch_grid(model_hw, patch_size=self.patch_size)

        if localized_objects is None:
            raise ValueError("AssetAggregatorPass requires localized_objects")
        if cameras_dggt is None:
            raise ValueError("AssetAggregatorPass requires cameras_dggt")
            
        render_cameras = self._build_dggt_cameras(sample, cameras_dggt, device=device)
        candidate_slots = self._resolve_object_slots(sample, selected_object_slots)

        object_keys: list[int] = []
        patch_start_idx = int(getattr(self.aggregator, "patch_start_idx", 0))

        F_g_lut_asset: dict[int, list[torch.Tensor]] = {}
        I_asset: dict[int, torch.Tensor] = {}
        A_asset: dict[int, torch.Tensor] = {}
        ptr_asset: dict[int, list[GaussianPointers]] = {}
        G_asset_dggt: dict[int, list[dict[str, torch.Tensor]]] = {}
        fit_metrics: dict[int, list[dict[str, Any]]] | None = (
            {} if bool(collect_fit_metrics) else None
        )

        batch_size = int(aggregator_batch_size)
        if batch_size <= 0:
            batch_size = max(1, len(candidate_slots))
        patch_start_idx_out = int(patch_start_idx)

        pending_keys: list[int] = []
        pending_renders: list[torch.Tensor] = []
        pending_alpha_renders: list[torch.Tensor] = []
        pending_depth_renders: list[torch.Tensor] = []
        pending_gaussians: list[list[dict[str, torch.Tensor]]] = []

        def _to_cpu_gaussian_sequence(
            gauss_seq: Sequence[dict[str, torch.Tensor]],
        ) -> list[dict[str, torch.Tensor]]:
            return [
                {name: value.detach().cpu().contiguous() for name, value in gauss.items()}
                for gauss in gauss_seq
            ]

        def _flush_pending() -> None:
            nonlocal patch_start_idx_out
            if len(pending_keys) == 0:
                return

            render_batch = torch.stack(pending_renders, dim=0)
            _, image_tokens_all, _, _, patch_start_idx = self.aggregator(render_batch)
            patch_tokens = select_patch_pyramid(image_tokens_all, self.levels, patch_start_idx)
            patch_start_idx_out = int(patch_start_idx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            for object_batch_idx, slot_idx in enumerate(pending_keys):
                F_g_lut_asset[int(slot_idx)] = [
                    level_tokens[object_batch_idx : object_batch_idx + 1].detach().cpu().contiguous()
                    for level_tokens in patch_tokens
                ]

            for object_batch_idx, slot_idx in enumerate(pending_keys):
                alpha_seq = pending_alpha_renders[object_batch_idx]
                depth_seq = pending_depth_renders[object_batch_idx]
                gauss_seq = pending_gaussians[object_batch_idx]
                ptr_asset[int(slot_idx)] = self._annotate_object_pointers(
                    int(slot_idx),
                    gauss_seq,
                    render_cameras,
                    patch_grid,
                    alpha_seq,
                    depth_seq,
                    occlusion_test,
                )
                gauss_seq_cpu = _to_cpu_gaussian_sequence(gauss_seq)
                G_asset_dggt[int(slot_idx)] = gauss_seq_cpu

                if fit_metrics is not None:
                    fit_metrics[int(slot_idx)] = self._build_fit_metrics_for_object(
                        slot_idx=int(slot_idx),
                        localized_by_key=localized_by_key,
                        alpha_seq=alpha_seq.detach().cpu(),
                    )
                I_asset[int(slot_idx)] = (
                    pending_renders[object_batch_idx].detach().cpu().unsqueeze(0).contiguous()
                )
                A_asset[int(slot_idx)] = (
                    alpha_seq.detach().cpu().unsqueeze(0).contiguous()
                )

            del image_tokens_all, patch_tokens, render_batch
            pending_keys.clear()
            pending_renders.clear()
            pending_alpha_renders.clear()
            pending_depth_renders.clear()
            pending_gaussians.clear()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        localized_by_key = self._index_localized_objects(localized_objects or [])
        for slot_idx in candidate_slots:
            if not self._is_valid_asset_slot(sample, slot_idx):
                continue

            gauss_seq, rgb_seq, alpha_seq, depth_seq = self._render_dggt_fitted_object_sequence(
                sample=sample,
                slot_idx=slot_idx,
                cameras=render_cameras,
                model_hw=model_hw,
                device=device,
                localized_by_key=localized_by_key,
            )
            
            if len(gauss_seq) == 0:
                continue
            if not any(gauss["means"].numel() > 0 for gauss in gauss_seq):
                continue

            object_keys.append(int(slot_idx))
            pending_keys.append(int(slot_idx))
            pending_gaussians.append(gauss_seq)
            pending_renders.append(torch.stack(rgb_seq, dim=0))
            pending_alpha_renders.append(torch.stack(alpha_seq, dim=0))
            pending_depth_renders.append(torch.stack(depth_seq, dim=0))
            if len(pending_keys) >= batch_size:
                _flush_pending()

        _flush_pending()

        return AssetPassResult(
            patch_grid=patch_grid,
            patch_start_idx=patch_start_idx_out,
            object_keys=object_keys,
            cameras_waymo={k: v.detach().cpu() for k, v in render_cameras.items()},
            F_g_lut_asset=F_g_lut_asset,
            ptr_asset=ptr_asset,
            G_asset_dggt=G_asset_dggt,
            I_asset=I_asset,
            A_asset=A_asset,
            G_asset_waymo={},
            asset_pass_space=str(render_space),
            fit_metrics=fit_metrics,
        )

    def _infer_device(self, images_clean: torch.Tensor) -> torch.device:
        for param in self.aggregator.parameters():
            return param.device
        for buffer in self.aggregator.buffers():
            return buffer.device
        return images_clean.device

    def _resolve_object_slots(
        self,
        sample: dict[str, Any],
        selected_object_slots: Sequence[int] | None,
    ) -> list[int]:
        if selected_object_slots is not None:
            return [int(slot) for slot in selected_object_slots]
        editable_count = int(sample["editable_object_count"].item())
        editable_indices = sample["editable_object_indices"][:editable_count].tolist()
        return [int(slot) for slot in editable_indices if int(slot) >= 0]

    @staticmethod
    def _is_valid_asset_slot(sample: dict[str, Any], slot_idx: int) -> bool:
        image_valid = sample.get("object_asset_image_valid_mask_selected")
        if isinstance(image_valid, torch.Tensor):
            if slot_idx < 0 or slot_idx >= image_valid.shape[0]:
                return False
            return bool(image_valid[slot_idx].any().item())
        asset_valid = sample.get("object_asset_valid_mask")
        if isinstance(asset_valid, torch.Tensor):
            if slot_idx < 0 or slot_idx >= asset_valid.shape[0]:
                return False
            if not bool(asset_valid[slot_idx].item()):
                return False
        asset_paths = sample.get("object_asset_paths")
        if asset_paths is None or slot_idx < 0 or slot_idx >= len(asset_paths):
            return False
        return Path(str(asset_paths[slot_idx])).is_file()

    def _build_dggt_cameras(
        self,
        sample: dict[str, Any],
        cameras_dggt: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if "viewmats" not in cameras_dggt or "Ks" not in cameras_dggt:
            raise ValueError("cameras_dggt must contain 'viewmats' and 'Ks'")
        num_frames = int(sample["frame_indices"].numel())
        num_views = int(sample["cam_ids"].numel())
        num_images = num_frames * num_views

        viewmats = cameras_dggt["viewmats"].detach().to(device).float()
        Ks = cameras_dggt["Ks"].detach().to(device).float()
        if viewmats.dim() == 4:
            if viewmats.shape[0] != 1:
                raise ValueError(f"Expected batch=1 cameras_dggt['viewmats'], got {tuple(viewmats.shape)}")
            viewmats = viewmats[0]
        if Ks.dim() == 4:
            if Ks.shape[0] != 1:
                raise ValueError(f"Expected batch=1 cameras_dggt['Ks'], got {tuple(Ks.shape)}")
            Ks = Ks[0]
        view_shape = tuple(viewmats.shape[-2:])
        if viewmats.dim() != 3 or view_shape not in ((3, 4), (4, 4)):
            raise ValueError(f"cameras_dggt['viewmats'] should be [S,3|4,4], got {tuple(viewmats.shape)}")
        if Ks.dim() != 3 or tuple(Ks.shape[-2:]) != (3, 3):
            raise ValueError(f"cameras_dggt['Ks'] should be [S,3,3], got {tuple(Ks.shape)}")
        if viewmats.shape[0] != num_images or Ks.shape[0] != num_images:
            raise ValueError(
                f"DGGT camera count must match images: got viewmats={viewmats.shape[0]} "
                f"Ks={Ks.shape[0]} images={num_images}"
            )
        if view_shape == (3, 4):
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=viewmats.dtype, device=device)
            bottom = bottom.view(1, 1, 4).expand(viewmats.shape[0], -1, -1)
            viewmats = torch.cat([viewmats, bottom], dim=1)

        image_to_frame = torch.arange(num_frames, dtype=torch.long, device=device).repeat_interleave(num_views)
        image_to_view = torch.arange(num_views, dtype=torch.long, device=device).repeat(num_frames)
        cam_ids = sample["cam_ids"].detach().to(device).long()[image_to_view]
        return {
            "camera_to_world": torch.linalg.inv(viewmats),
            "world_to_camera": viewmats,
            "K_model": Ks,
            "image_to_frame": image_to_frame,
            "image_to_view": image_to_view,
            "cam_ids": cam_ids,
        }

    @staticmethod
    def _index_localized_objects(
        localized_objects: Sequence[Any],
    ) -> dict[tuple[int, int, int], Any]:
        indexed: dict[tuple[int, int, int], Any] = {}
        for item in localized_objects:
            slot_idx = int(getattr(item, "slot_idx"))
            frame_idx = int(getattr(item, "frame_idx"))
            source_index = int(getattr(item, "source_front_index", frame_idx))
            indexed[(slot_idx, frame_idx, source_index)] = item
        return indexed

    @staticmethod
    def _localized_gaussian_dict(item: Any, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "means": getattr(item, "asset_means_world").to(device).float(),
            "colors": getattr(item, "asset_colors").to(device).float(),
            "opacities": getattr(item, "asset_opacities").to(device).float(),
            "scales": getattr(item, "asset_scales").to(device).float(),
            "quats": getattr(item, "asset_quats").to(device).float(),
        }

    def _render_dggt_fitted_object_sequence(
        self,
        sample: dict[str, Any],
        slot_idx: int,
        cameras: dict[str, torch.Tensor],
        model_hw: tuple[int, int],
        device: torch.device,
        localized_by_key: dict[tuple[int, int, int], Any],
    ) -> tuple[
        list[dict[str, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        num_images = int(cameras["world_to_camera"].shape[0])
        H, W = int(model_hw[0]), int(model_hw[1])
        zero_rgb = torch.zeros((3, H, W), dtype=torch.float32, device=device)
        zero_alpha = torch.zeros((1, H, W), dtype=torch.float32, device=device)
        zero_depth = torch.zeros((H, W), dtype=torch.float32, device=device)

        gauss_seq: list[dict[str, torch.Tensor]] = []
        rgb_seq: list[torch.Tensor] = []
        alpha_seq: list[torch.Tensor] = []
        depth_seq: list[torch.Tensor] = []

        for image_idx in range(num_images):
            frame_idx = int(cameras["image_to_frame"][image_idx].item())
            source_index = image_idx
            item = localized_by_key.get((int(slot_idx), frame_idx, source_index))
            if item is None:
                gauss_seq.append(empty_gaussian_dict())
                rgb_seq.append(zero_rgb.clone())
                alpha_seq.append(zero_alpha.clone())
                depth_seq.append(zero_depth.clone())
                continue

            gauss_dggt = self._localized_gaussian_dict(item, device)
            if gauss_dggt["means"].numel() == 0:
                gauss_seq.append(empty_gaussian_dict())
                rgb_seq.append(zero_rgb.clone())
                alpha_seq.append(zero_alpha.clone())
                depth_seq.append(zero_depth.clone())
                continue
            rgb, alpha, depth = self._render_gaussians_for_camera(
                gauss_dggt,
                cameras["world_to_camera"][image_idx],
                cameras["K_model"][image_idx],
                (H, W),
            )
            gauss_seq.append(gauss_dggt)
            rgb_seq.append(rgb)
            alpha_seq.append(alpha)
            depth_seq.append(depth)

        return gauss_seq, rgb_seq, alpha_seq, depth_seq

    @staticmethod
    def _box_iou_xyxy(box_a: torch.Tensor | None, box_b: torch.Tensor | None) -> float:
        if box_a is None or box_b is None:
            return float("nan")
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a.reshape(-1).tolist()]
        bx1, by1, bx2, by2 = [float(v) for v in box_b.reshape(-1).tolist()]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    @staticmethod
    def _box_center_error(box_a: torch.Tensor | None, box_b: torch.Tensor | None) -> float:
        if box_a is None or box_b is None:
            return float("nan")
        box_a = box_a.reshape(-1).float()
        box_b = box_b.reshape(-1).float()
        center_a = 0.5 * (box_a[:2] + box_a[2:])
        center_b = 0.5 * (box_b[:2] + box_b[2:])
        size_b = (box_b[2:] - box_b[:2]).clamp_min(1.0)
        return float(((center_a - center_b).abs() / size_b).mean().item())

    @staticmethod
    def _alpha_bbox(alpha: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor | None:
        alpha = alpha.detach().cpu().float()
        if alpha.dim() == 3:
            alpha = alpha[0]
        ys, xs = torch.nonzero(alpha > threshold, as_tuple=True)
        if ys.numel() == 0:
            return None
        return torch.tensor(
            [
                float(xs.min().item()),
                float(ys.min().item()),
                float(xs.max().item() + 1),
                float(ys.max().item() + 1),
            ],
            dtype=torch.float32,
        )

    def _build_fit_metrics_for_object(
        self,
        slot_idx: int,
        localized_by_key: dict[tuple[int, int, int], Any],
        alpha_seq: torch.Tensor,
    ) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        for image_idx in range(int(alpha_seq.shape[0])):
            matches = [
                item
                for (obj_key, _frame_idx, source_idx), item in localized_by_key.items()
                if obj_key == int(slot_idx) and source_idx == image_idx
            ]
            if len(matches) == 0:
                metrics.append(
                    {
                        "image_idx": int(image_idx),
                        "localized": False,
                    }
                )
                continue
            item = matches[0]
            target_bbox = getattr(item, "target_bbox_model", None)
            projected_bbox = getattr(item, "projected_asset_bbox", None)
            alpha_bbox = self._alpha_bbox(alpha_seq[image_idx], threshold=self.alpha_thresh)
            row = {
                "image_idx": int(image_idx),
                "frame_idx": int(getattr(item, "frame_idx")),
                "localized": True,
                "target_bbox_model": None
                if target_bbox is None
                else [float(v) for v in target_bbox.detach().cpu().reshape(-1).tolist()],
                "projected_asset_bbox": None
                if projected_bbox is None
                else [float(v) for v in projected_bbox.detach().cpu().reshape(-1).tolist()],
                "alpha_bbox": None
                if alpha_bbox is None
                else [float(v) for v in alpha_bbox.reshape(-1).tolist()],
                "projected_iou": self._box_iou_xyxy(projected_bbox, target_bbox),
                "projected_center_error": self._box_center_error(projected_bbox, target_bbox),
                "alpha_iou": self._box_iou_xyxy(alpha_bbox, target_bbox),
                "alpha_center_error": self._box_center_error(alpha_bbox, target_bbox),
            }
            pose_diag = getattr(item, "pose_refine_diagnostics", None)
            if isinstance(pose_diag, dict):
                row["corner_refine"] = dict(pose_diag)
                for key in (
                    "corner_refine_status",
                    "corner_refine_accepted",
                    "corner_refine_reason",
                    "corner_rms_before",
                    "corner_rms_after",
                    "bbox_iou_before",
                    "bbox_iou_after",
                    "yaw_delta_deg",
                    "center_shift",
                    "depth_ratio",
                ):
                    row[key] = pose_diag.get(key)
            metrics.append(row)
        return metrics

    def _render_gaussians_for_camera(
        self,
        gaussians: dict[str, torch.Tensor],
        world_to_camera: torch.Tensor,
        K_model: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = int(image_hw[0]), int(image_hw[1])
        device = world_to_camera.device
        if gaussians["means"].numel() == 0:
            return (
                torch.zeros((3, H, W), dtype=torch.float32, device=device),
                torch.zeros((1, H, W), dtype=torch.float32, device=device),
                torch.zeros((H, W), dtype=torch.float32, device=device),
            )

        try:
            from gsplat.rendering import rasterization
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "AssetAggregatorPass requires the `gsplat` package for isolated asset rendering."
            ) from exc

        rendered, alpha, _ = rasterization(
            means=gaussians["means"].to(device).float(),
            quats=gaussians["quats"].to(device).float(),
            scales=gaussians["scales"].to(device).float(),
            opacities=gaussians["opacities"].to(device).float().view(-1),
            colors=gaussians["colors"].to(device).float(),
            viewmats=world_to_camera.view(1, 4, 4).float(),
            Ks=K_model.view(1, 3, 3).float(),
            width=W,
            height=H,
            render_mode="RGB+ED",
        )
        rgb = rendered[0, ..., :3].permute(2, 0, 1).contiguous()
        depth = rendered[0, ..., -1].contiguous()
        alpha_chw = alpha[0, ..., 0].unsqueeze(0).contiguous()
        return rgb, alpha_chw, depth

    def _annotate_object_pointers(
        self,
        object_id: int,
        gaussians_seq: Sequence[dict[str, torch.Tensor]],
        cameras: dict[str, torch.Tensor],
        patch_grid: tuple[int, int],
        alpha_seq: torch.Tensor,
        depth_seq: torch.Tensor,
        occlusion_test: bool,
    ) -> list[GaussianPointers]:
        patch_h, patch_w = int(patch_grid[0]), int(patch_grid[1])
        num_images = len(gaussians_seq)
        image_hw = (int(alpha_seq.shape[-2]), int(alpha_seq.shape[-1]))

        provisional_patch_idx: list[torch.Tensor] = []
        visible_mask_seq: list[torch.Tensor] = []
        for image_idx in range(num_images):
            gaussians = gaussians_seq[image_idx]
            patch_idx, visible = self._project_gaussians_to_patches(
                gaussians=gaussians,
                world_to_camera=cameras["world_to_camera"][image_idx],
                K_model=cameras["K_model"][image_idx],
                image_hw=image_hw,
                patch_grid=patch_grid,
                alpha_map=alpha_seq[image_idx],
                depth_map=depth_seq[image_idx],
                occlusion_test=occlusion_test,
            )
            provisional_patch_idx.append(patch_idx)
            visible_mask_seq.append(visible)

        view_n_seq, patch_idx_seq = apply_pointer_fallback(
            provisional_patch_idx,
            visible_mask_seq,
            cameras["image_to_frame"],
            cameras["image_to_view"],
        )

        out: list[GaussianPointers] = []
        for image_idx in range(num_images):
            patch_idx = patch_idx_seq[image_idx]
            visible = visible_mask_seq[image_idx]
            num_gauss = int(patch_idx.shape[0])
            out.append(
                GaussianPointers(
                    src_kind=torch.full((num_gauss,), SRC_KIND_ASSET, dtype=torch.int32),
                    object_id=torch.full((num_gauss,), int(object_id), dtype=torch.int32),
                    view_n=view_n_seq[image_idx].to(torch.int32),
                    patch_idx=patch_idx.to(torch.int32).clamp_(min=0, max=patch_h * patch_w - 1),
                    visible_mask=visible,
                )
            )
        return out

    def _project_gaussians_to_patches(
        self,
        gaussians: dict[str, torch.Tensor],
        world_to_camera: torch.Tensor,
        K_model: torch.Tensor,
        image_hw: tuple[int, int],
        patch_grid: tuple[int, int],
        alpha_map: torch.Tensor,
        depth_map: torch.Tensor,
        occlusion_test: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_gauss = int(gaussians["means"].shape[0])
        if num_gauss == 0:
            return (
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.bool),
            )

        means = gaussians["means"].detach().cpu().float()
        world_to_camera = world_to_camera.detach().cpu().float()
        K_model = K_model.detach().cpu().float()
        alpha_map = alpha_map.detach().cpu().float()
        depth_map = depth_map.detach().cpu().float()

        H, W = int(image_hw[0]), int(image_hw[1])
        patch_h, patch_w = int(patch_grid[0]), int(patch_grid[1])

        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        points_cam = means @ rotation.T + translation
        depth = points_cam[:, 2]

        fx = K_model[0, 0]
        fy = K_model[1, 1]
        cx = K_model[0, 2]
        cy = K_model[1, 2]
        u = fx * points_cam[:, 0] / depth.clamp_min(1e-6) + cx
        v = fy * points_cam[:, 1] / depth.clamp_min(1e-6) + cy
        valid = (
            torch.isfinite(u)
            & torch.isfinite(v)
            & torch.isfinite(depth)
            & (depth > 1e-6)
            & (u >= 0.0)
            & (u < float(W))
            & (v >= 0.0)
            & (v < float(H))
        )

        if occlusion_test and bool(valid.any().item()):
            x_pix = torch.round(u[valid]).long().clamp_(0, W - 1)
            y_pix = torch.round(v[valid]).long().clamp_(0, H - 1)
            alpha_hit = alpha_map[0, y_pix, x_pix] > self.alpha_thresh
            depth_hit = depth_map[y_pix, x_pix]
            depth_ok = depth_hit > 0
            depth_ok &= (depth_hit - depth[valid]).abs() <= self.depth_tol
            visible_valid = alpha_hit & depth_ok
            visible = torch.zeros_like(valid)
            visible[valid] = visible_valid
        else:
            visible = valid

        patch_x = torch.floor(u / float(self.patch_size)).long()
        patch_y = torch.floor(v / float(self.patch_size)).long()
        patch_x = patch_x.clamp_(0, patch_w - 1)
        patch_y = patch_y.clamp_(0, patch_h - 1)
        patch_idx = patch_y * patch_w + patch_x
        patch_idx = patch_idx.clamp_(min=0, max=patch_h * patch_w - 1)
        return patch_idx, visible
