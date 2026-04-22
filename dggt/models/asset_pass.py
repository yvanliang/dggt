"""Asset Aggregator Pass for isolated Waymo-coordinate asset renders.

This module bridges the current `WaymoEditDataset` sample structure with the
FlowDGGT asset-pass design:

* place each editable asset at its per-frame Waymo box pose
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
    apply_sim3_to_gaussian_dict,
    empty_gaussian_dict,
    load_asset_gaussians,
    transform_asset_gaussians,
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
    G_asset_waymo: dict[int, list[dict[str, torch.Tensor]]]
    G_asset_dggt: dict[int, list[dict[str, torch.Tensor]]] | None
    I_asset: dict[int, torch.Tensor]
    A_asset: dict[int, torch.Tensor]


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
        invisible_ids = torch.nonzero(~visible, as_tuple=False).flatten().tolist()
        for gauss_idx in invisible_ids:
            best_key: tuple[int, int, int, int] | None = None
            best_image_idx: int | None = None
            for cand_idx in range(num_images):
                cand_visible = visible_mask_per_image[cand_idx].detach().cpu().bool()
                cand_patch = patch_idx_per_image[cand_idx].detach().cpu().long()
                if cand_visible.shape[0] != patch_idx.shape[0] or cand_patch.shape[0] != patch_idx.shape[0]:
                    continue
                if not bool(cand_visible[gauss_idx].item()):
                    continue
                frame_c = int(image_to_frame[cand_idx].item())
                view_c = int(image_to_view[cand_idx].item())
                cand_key = (
                    abs(frame_i - frame_c),
                    0 if view_i == view_c else 1,
                    abs(image_idx - cand_idx),
                    cand_idx,
                )
                if best_key is None or cand_key < best_key:
                    best_key = cand_key
                    best_image_idx = cand_idx
            if best_image_idx is not None:
                view_n[gauss_idx] = int(best_image_idx)
                patch_idx[gauss_idx] = int(
                    patch_idx_per_image[best_image_idx].detach().cpu().long()[gauss_idx].item()
                )
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
        alignment: Sim3Transform | None = None,
        asset_cache: dict[str, dict[str, torch.Tensor]] | None = None,
        occlusion_test: bool = True,
    ) -> AssetPassResult:
        images_clean = sample["images_clean"]
        if images_clean.dim() != 4:
            raise ValueError(
                f"sample['images_clean'] must be [S,3,H,W], got {tuple(images_clean.shape)}"
            )

        device = self._infer_device(images_clean)
        model_hw = (int(images_clean.shape[-2]), int(images_clean.shape[-1]))
        patch_grid = compute_runtime_patch_grid(model_hw, patch_size=self.patch_size)
        cameras_waymo = self._build_waymo_cameras(sample, model_hw, device=device)
        candidate_slots = self._resolve_object_slots(sample, selected_object_slots)
        cache = {} if asset_cache is None else asset_cache

        object_keys: list[int] = []
        object_renders: list[torch.Tensor] = []
        object_alpha_renders: list[torch.Tensor] = []
        object_depth_renders: list[torch.Tensor] = []
        G_asset_waymo: dict[int, list[dict[str, torch.Tensor]]] = {}

        for slot_idx in candidate_slots:
            if not self._is_valid_asset_slot(sample, slot_idx):
                continue
            asset_path = str(sample["object_asset_paths"][slot_idx])
            asset_local = load_asset_gaussians(asset_path, cache)
            gauss_seq, rgb_seq, alpha_seq, depth_seq = self._render_object_sequence(
                sample,
                slot_idx,
                asset_local,
                cameras_waymo,
                model_hw,
                device,
            )
            if len(gauss_seq) == 0:
                continue
            if not any(gauss["means"].numel() > 0 for gauss in gauss_seq):
                continue

            object_keys.append(int(slot_idx))
            G_asset_waymo[int(slot_idx)] = gauss_seq
            object_renders.append(torch.stack(rgb_seq, dim=0))
            object_alpha_renders.append(torch.stack(alpha_seq, dim=0))
            object_depth_renders.append(torch.stack(depth_seq, dim=0))

        patch_start_idx = int(getattr(self.aggregator, "patch_start_idx", 0))
        if len(object_keys) == 0:
            return AssetPassResult(
                patch_grid=patch_grid,
                patch_start_idx=patch_start_idx,
                object_keys=[],
                cameras_waymo=cameras_waymo,
                F_g_lut_asset={},
                ptr_asset={},
                G_asset_waymo={},
                G_asset_dggt={} if alignment is not None else None,
                I_asset={},
                A_asset={},
            )

        render_batch = torch.stack(object_renders, dim=0)
        _, image_tokens_all, _, _, patch_start_idx = self.aggregator(render_batch)
        patch_tokens = select_patch_pyramid(image_tokens_all, self.levels, patch_start_idx)

        F_g_lut_asset: dict[int, list[torch.Tensor]] = {}
        I_asset: dict[int, torch.Tensor] = {}
        A_asset: dict[int, torch.Tensor] = {}
        ptr_asset: dict[int, list[GaussianPointers]] = {}
        G_asset_dggt: dict[int, list[dict[str, torch.Tensor]]] | None = {} if alignment is not None else None

        for object_batch_idx, slot_idx in enumerate(object_keys):
            F_g_lut_asset[int(slot_idx)] = [
                level_tokens[object_batch_idx : object_batch_idx + 1].contiguous()
                for level_tokens in patch_tokens
            ]
            I_asset[int(slot_idx)] = object_renders[object_batch_idx].unsqueeze(0).contiguous()
            A_asset[int(slot_idx)] = object_alpha_renders[object_batch_idx].unsqueeze(0).contiguous()
            ptr_asset[int(slot_idx)] = self._annotate_object_pointers(
                object_id=int(slot_idx),
                gaussians_seq=G_asset_waymo[int(slot_idx)],
                cameras_waymo=cameras_waymo,
                patch_grid=patch_grid,
                alpha_seq=object_alpha_renders[object_batch_idx],
                depth_seq=object_depth_renders[object_batch_idx],
                occlusion_test=occlusion_test,
            )
            if G_asset_dggt is not None:
                G_asset_dggt[int(slot_idx)] = [
                    apply_sim3_to_gaussian_dict(gauss, alignment)
                    for gauss in G_asset_waymo[int(slot_idx)]
                ]

        return AssetPassResult(
            patch_grid=patch_grid,
            patch_start_idx=int(patch_start_idx),
            object_keys=object_keys,
            cameras_waymo=cameras_waymo,
            F_g_lut_asset=F_g_lut_asset,
            ptr_asset=ptr_asset,
            G_asset_waymo=G_asset_waymo,
            G_asset_dggt=G_asset_dggt,
            I_asset=I_asset,
            A_asset=A_asset,
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

    def _build_waymo_cameras(
        self,
        sample: dict[str, Any],
        model_hw: tuple[int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        num_frames = int(sample["frame_indices"].numel())
        num_views = int(sample["cam_ids"].numel())

        camera_to_world = sample["camera_to_world_corrected"].detach().cpu().float().view(-1, 4, 4)
        world_to_camera = torch.linalg.inv(camera_to_world)
        intrinsics_raw = sample["intrinsics"].detach().cpu().float()
        raw_hw = sample["raw_image_size_hw"].detach().cpu().long()

        K_model_per_view = []
        for view_idx in range(num_views):
            view_hw = (int(raw_hw[view_idx, 0].item()), int(raw_hw[view_idx, 1].item()))
            K_model_per_view.append(
                compute_model_intrinsics(
                    intrinsics_raw[view_idx],
                    view_hw,
                    model_hw,
                )
            )
        K_model = torch.stack(
            [
                K_model_per_view[view_idx]
                for _frame_idx in range(num_frames)
                for view_idx in range(num_views)
            ],
            dim=0,
        )

        image_to_frame = torch.arange(num_frames, dtype=torch.long).repeat_interleave(num_views)
        image_to_view = torch.arange(num_views, dtype=torch.long).repeat(num_frames)
        cam_ids = sample["cam_ids"].detach().cpu().long()[image_to_view]

        return {
            "camera_to_world": camera_to_world.to(device),
            "world_to_camera": world_to_camera.to(device),
            "K_model": K_model.to(device),
            "image_to_frame": image_to_frame.to(device),
            "image_to_view": image_to_view.to(device),
            "cam_ids": cam_ids.to(device),
        }

    def _render_object_sequence(
        self,
        sample: dict[str, Any],
        slot_idx: int,
        asset_local: dict[str, torch.Tensor],
        cameras_waymo: dict[str, torch.Tensor],
        model_hw: tuple[int, int],
        device: torch.device,
    ) -> tuple[
        list[dict[str, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        num_images = int(cameras_waymo["world_to_camera"].shape[0])
        H, W = int(model_hw[0]), int(model_hw[1])
        zero_rgb = torch.zeros((3, H, W), dtype=torch.float32, device=device)
        zero_alpha = torch.zeros((1, H, W), dtype=torch.float32, device=device)
        zero_depth = torch.zeros((H, W), dtype=torch.float32, device=device)

        track_valid = sample["object_track_valid_mask_selected"][slot_idx].detach().cpu().bool()
        obj_to_world = sample["object_obj_to_world_selected"][slot_idx].detach().cpu().float()
        box_size = sample["object_box_size_selected"][slot_idx].detach().cpu().float()

        gauss_seq: list[dict[str, torch.Tensor]] = []
        rgb_seq: list[torch.Tensor] = []
        alpha_seq: list[torch.Tensor] = []
        depth_seq: list[torch.Tensor] = []

        for image_idx in range(num_images):
            frame_idx = int(cameras_waymo["image_to_frame"][image_idx].item())
            if frame_idx < 0 or frame_idx >= track_valid.shape[0] or not bool(track_valid[frame_idx].item()):
                gauss_seq.append(empty_gaussian_dict())
                rgb_seq.append(zero_rgb.clone())
                alpha_seq.append(zero_alpha.clone())
                depth_seq.append(zero_depth.clone())
                continue

            object_rotation = obj_to_world[frame_idx, :3, :3].to(device)
            object_center = obj_to_world[frame_idx, :3, 3].to(device)
            target_lwh = box_size[frame_idx].to(device)
            gauss_waymo = transform_asset_gaussians(
                asset_local,
                target_lwh,
                object_rotation,
                object_center,
            )
            rgb, alpha, depth = self._render_gaussians_for_camera(
                gauss_waymo,
                cameras_waymo["world_to_camera"][image_idx],
                cameras_waymo["K_model"][image_idx],
                (H, W),
            )
            gauss_seq.append(gauss_waymo)
            rgb_seq.append(rgb)
            alpha_seq.append(alpha)
            depth_seq.append(depth)

        return gauss_seq, rgb_seq, alpha_seq, depth_seq

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
        cameras_waymo: dict[str, torch.Tensor],
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
                world_to_camera=cameras_waymo["world_to_camera"][image_idx],
                K_model=cameras_waymo["K_model"][image_idx],
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
            cameras_waymo["image_to_frame"],
            cameras_waymo["image_to_view"],
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
