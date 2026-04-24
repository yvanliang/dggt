"""Scene-side per-Gaussian pointer builder for FeatureSplatter.

Scene Gaussians come from `build_clean_scene_state` which records each
Gaussian's source `(image_id, y, x)`. Translating those to `(view_n, patch_idx)`
is a pure indexing operation — no network needed.
"""
from __future__ import annotations

import torch

from dggt.models.gaussian_pointers import (
    GaussianPointers,
    SCENE_OBJECT_ID,
    SRC_KIND_SCENE,
)


def build_scene_pointers(
    source_image_ids: torch.Tensor,
    source_y: torch.Tensor,
    source_x: torch.Tensor,
    patch_size: int = 14,
    patch_grid: tuple[int, int] = (37, 37),
) -> GaussianPointers:
    """Construct scene-side GaussianPointers from CleanSceneState source coords.

    Parameters
    ----------
    source_image_ids : [N_g] int
        Flat image index (frame * num_views + view_offset) per Gaussian.
    source_y, source_x : [N_g] int
        Source pixel coordinates per Gaussian (0-based, inside 518x518 image).
    patch_size : int
        Patch side in pixels (14 for DINOv2 ViT-L/14).
    patch_grid : (int, int)
        Target patch grid (patch_h, patch_w). 518 / 14 = 37.
    """
    if source_image_ids.shape != source_y.shape or source_y.shape != source_x.shape:
        raise ValueError(
            "source_image_ids, source_y, source_x must share shape; got "
            f"{tuple(source_image_ids.shape)}, {tuple(source_y.shape)}, {tuple(source_x.shape)}"
        )
    patch_h, patch_w = int(patch_grid[0]), int(patch_grid[1])
    py = (source_y.to(torch.long) // int(patch_size)).clamp_(0, patch_h - 1)
    px = (source_x.to(torch.long) // int(patch_size)).clamp_(0, patch_w - 1)
    patch_idx = py * patch_w + px
    n = int(patch_idx.numel())
    device = patch_idx.device
    return GaussianPointers(
        src_kind=torch.full((n,), SRC_KIND_SCENE, dtype=torch.int32, device=device),
        object_id=torch.full((n,), SCENE_OBJECT_ID, dtype=torch.int32, device=device),
        view_n=source_image_ids.to(torch.int32),
        patch_idx=patch_idx.to(torch.int32),
        visible_mask=torch.ones((n,), dtype=torch.bool, device=device),
    )


def concat_pointers(pointer_list: list[GaussianPointers]) -> GaussianPointers:
    """Concatenate per-source GaussianPointers (scene + per-object asset) into one."""
    if len(pointer_list) == 0:
        return GaussianPointers(
            src_kind=torch.zeros((0,), dtype=torch.int32),
            object_id=torch.zeros((0,), dtype=torch.int32),
            view_n=torch.zeros((0,), dtype=torch.int32),
            patch_idx=torch.zeros((0,), dtype=torch.int32),
            visible_mask=torch.zeros((0,), dtype=torch.bool),
        )
    return GaussianPointers(
        src_kind=torch.cat([p.src_kind for p in pointer_list], dim=0),
        object_id=torch.cat([p.object_id for p in pointer_list], dim=0),
        view_n=torch.cat([p.view_n for p in pointer_list], dim=0),
        patch_idx=torch.cat([p.patch_idx for p in pointer_list], dim=0),
        visible_mask=torch.cat([p.visible_mask for p in pointer_list], dim=0),
    )
