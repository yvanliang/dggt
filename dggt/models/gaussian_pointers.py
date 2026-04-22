"""Shared pointer payloads for feature splatting and asset passes.

This module deliberately avoids importing `gsplat` so CPU-side tests can import
pointer types without pulling in CUDA rasterization dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


SRC_KIND_SCENE = 0
SRC_KIND_ASSET = 1
SCENE_OBJECT_ID = -1


@dataclass
class GaussianPointers:
    """Per-Gaussian pointer arrays. All tensors share length `N_g`."""

    src_kind: torch.Tensor
    object_id: torch.Tensor
    view_n: torch.Tensor
    patch_idx: torch.Tensor
    visible_mask: torch.Tensor

    def to(self, device: torch.device) -> "GaussianPointers":
        return GaussianPointers(
            src_kind=self.src_kind.to(device),
            object_id=self.object_id.to(device),
            view_n=self.view_n.to(device),
            patch_idx=self.patch_idx.to(device),
            visible_mask=self.visible_mask.to(device),
        )
