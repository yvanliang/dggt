"""FeatureSplatter: pointer-based feature splatting for FlowDGGT.

Each 3D Gaussian carries a pointer `(src_kind, object_id, view_n, patch_idx,
visible_mask)` into a LUT of 3072-d aggregator patch tokens (one LUT per DPT
pyramid level). Per-Gaussian colors are gathered from the LUT and rasterized
via gsplat. Rendered feature maps are area-pooled down to a 37x37 grid and
stacked across the 4 levels.

Gradients flow only into the LUT; Gaussian geometry (means, quats, scales,
opacities) is detached inside this module.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dggt.models.gaussian_pointers import (
    GaussianPointers,
    SCENE_OBJECT_ID,
    SRC_KIND_ASSET,
    SRC_KIND_SCENE,
)
from gsplat.rendering import rasterization


class FeatureSplatter(nn.Module):
    """Pointer-based chunked feature splatter.

    Parameters
    ----------
    channels
        Feature width of the LUT (3072 for `image_tokens_list`).
    chunk_channels
        Number of channels per rasterize pass (memory knob, ~1 GB peak at 512).
    num_levels
        Number of DPT pyramid levels (4 for [4, 11, 17, 23]).
    patch_grid
        Target spatial grid to pool to (default 37x37 = 1369 patches).
    """

    def __init__(
        self,
        channels: int = 3072,
        chunk_channels: int = 512,
        num_levels: int = 4,
        patch_grid: tuple[int, int] = (37, 37),
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if chunk_channels <= 0:
            raise ValueError(f"chunk_channels must be positive, got {chunk_channels}")
        self.channels = int(channels)
        self.chunk_channels = int(chunk_channels)
        self.num_levels = int(num_levels)
        self.patch_h, self.patch_w = int(patch_grid[0]), int(patch_grid[1])
        self.num_patches = self.patch_h * self.patch_w

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        gaussians_dggt: Sequence[Mapping[str, torch.Tensor]],
        pointers: Sequence[GaussianPointers],
        lut_scene: Sequence[torch.Tensor],
        lut_asset_dict: Mapping[int, Sequence[torch.Tensor]] | None,
        cameras_dggt: Mapping[str, torch.Tensor],
        H: int,
        W: int,
        pool_to: int | None = 37,
    ) -> list[torch.Tensor]:
        """Splat features per level.

        Parameters
        ----------
        gaussians_dggt
            Length-B list. Each entry is a dict with keys `means [N_g, 3]`,
            `quats [N_g, 4]`, `scales [N_g, 3]`, `opacities [N_g]`, all in
            DGGT-coord. Geometry is detached internally.
        pointers
            Length-B list of `GaussianPointers` (lengths must match per-batch
            Gaussian count).
        lut_scene
            Length-`num_levels` sequence. Each tensor is
            `[B, N_scene, P, C]` and carries the scene LUT for that level.
            `P == num_patches`; `C == channels`.
        lut_asset_dict
            Optional dict mapping asset object key (int) to a length-`num_levels`
            sequence of `[B, N_asset_k, P, C]` tensors.
        cameras_dggt
            Dict with `viewmats [B, S, 4, 4]` (world-to-camera in DGGT-coord)
            and `Ks [B, S, 3, 3]`.
        H, W
            Full rasterization resolution (e.g. 148 or 296). Must satisfy
            `H % pool_to == 0`, `W % pool_to == 0` when `pool_to` is set.
        pool_to
            Target side length of the patch grid (37 for low-res splat).
            `None` skips pooling and returns full resolution.

        Returns
        -------
        list[Tensor]
            Length `num_levels`. Each tensor has shape
            `[B, S, pool_to * pool_to, C]` when pool_to is set, otherwise
            `[B, S, H, W, C]`.
        """
        self._validate_inputs(gaussians_dggt, pointers, lut_scene, lut_asset_dict, cameras_dggt)

        B = len(gaussians_dggt)
        viewmats = cameras_dggt["viewmats"]  # [B, S, 4, 4]
        Ks = cameras_dggt["Ks"]              # [B, S, 3, 3]
        S = viewmats.shape[1]

        if pool_to is not None:
            if H % pool_to != 0 or W % pool_to != 0:
                raise ValueError(
                    f"H ({H}) and W ({W}) must be divisible by pool_to ({pool_to})"
                )

        outputs_per_level: list[list[torch.Tensor]] = [[] for _ in range(self.num_levels)]

        for b in range(B):
            g = gaussians_dggt[b]
            ptr = pointers[b]
            viewmats_b = viewmats[b]     # [S, 4, 4]
            Ks_b = Ks[b]                 # [S, 3, 3]

            rendered_per_level = self._splat_one_batch(
                gaussians=g,
                pointers=ptr,
                lut_scene_per_level=[lut_scene[l][b] for l in range(self.num_levels)],
                lut_asset_per_level_dict=None
                if lut_asset_dict is None
                else {
                    k: [lut_asset_dict[k][l][b] for l in range(self.num_levels)]
                    for k in lut_asset_dict.keys()
                },
                viewmats=viewmats_b,
                Ks=Ks_b,
                H=H,
                W=W,
            )

            for l in range(self.num_levels):
                rendered = rendered_per_level[l]  # [S, H, W, C]
                if pool_to is not None:
                    rendered = self._area_pool(rendered, pool_to)  # [S, pool_to*pool_to, C]
                outputs_per_level[l].append(rendered)

        stacked = [torch.stack(outs, dim=0) for outs in outputs_per_level]  # each [B, S, ...]
        return stacked

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _splat_one_batch(
        self,
        gaussians: Mapping[str, torch.Tensor],
        pointers: GaussianPointers,
        lut_scene_per_level: list[torch.Tensor],       # each [N_scene, P, C]
        lut_asset_per_level_dict: dict[int, list[torch.Tensor]] | None,
        viewmats: torch.Tensor,                        # [S, 4, 4]
        Ks: torch.Tensor,                              # [S, 3, 3]
        H: int,
        W: int,
    ) -> list[torch.Tensor]:
        device = viewmats.device

        means = gaussians["means"].to(device).detach().float()                    # [N_g, 3]
        quats = gaussians["quats"].to(device).detach().float()                    # [N_g, 4]
        scales = gaussians["scales"].to(device).detach().float()                  # [N_g, 3]
        opacities = gaussians["opacities"].to(device).detach().float().view(-1)   # [N_g]

        ptr = pointers.to(device)
        visible = ptr.visible_mask.to(opacities.dtype)
        opacities = opacities * visible  # dead Gaussians contribute nothing

        S = viewmats.shape[0]
        rendered_levels: list[torch.Tensor] = []

        for level_idx in range(self.num_levels):
            flat_lut, global_idx = self._build_flat_lut_and_index(
                pointers=ptr,
                lut_scene=lut_scene_per_level[level_idx],
                lut_asset_dict=None
                if lut_asset_per_level_dict is None
                else {k: lut_asset_per_level_dict[k][level_idx] for k in lut_asset_per_level_dict},
            )
            # Gather colors keeps gradient on flat_lut → original LUT tensors.
            colors_full = flat_lut.index_select(0, global_idx)  # [N_g, C]

            if means.numel() == 0:
                rendered_levels.append(
                    torch.zeros((S, H, W, self.channels), dtype=colors_full.dtype, device=device)
                )
                continue

            rendered = self._rasterize_chunked(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors_full,
                viewmats=viewmats,
                Ks=Ks,
                H=H,
                W=W,
            )
            rendered_levels.append(rendered)

        return rendered_levels

    def _build_flat_lut_and_index(
        self,
        pointers: GaussianPointers,
        lut_scene: torch.Tensor,                         # [N_scene, P, C]
        lut_asset_dict: dict[int, torch.Tensor] | None,  # each [N_asset_k, P, C]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N_scene, P, C = lut_scene.shape
        if P != self.num_patches:
            raise ValueError(f"lut patch count {P} != num_patches {self.num_patches}")
        if C != self.channels:
            raise ValueError(f"lut channels {C} != channels {self.channels}")

        flat_chunks = [lut_scene.reshape(N_scene * P, C)]
        # (src_kind, object_id) -> offset into flat LUT
        offsets: dict[tuple[int, int], int] = {(SRC_KIND_SCENE, SCENE_OBJECT_ID): 0}
        running = N_scene * P

        if lut_asset_dict is not None:
            for obj_key in sorted(lut_asset_dict.keys()):
                asset_lut = lut_asset_dict[obj_key]   # [N_asset, P, C]
                N_asset, P_a, C_a = asset_lut.shape
                if P_a != P or C_a != C:
                    raise ValueError(
                        f"asset LUT[{obj_key}] shape {(N_asset, P_a, C_a)} incompatible with scene "
                        f"({N_scene}, {P}, {C})"
                    )
                offsets[(SRC_KIND_ASSET, int(obj_key))] = running
                running += N_asset * P
                flat_chunks.append(asset_lut.reshape(N_asset * P, C))

        flat_lut = torch.cat(flat_chunks, dim=0)  # [N_total, C]

        src_kind = pointers.src_kind.to(torch.long)
        obj_id = pointers.object_id.to(torch.long)
        view_n = pointers.view_n.to(torch.long)
        patch_idx = pointers.patch_idx.to(torch.long)

        offset_per_gauss = torch.zeros_like(view_n)
        for (sk, oid), off in offsets.items():
            sel = (src_kind == sk) & (obj_id == oid)
            if sel.any():
                offset_per_gauss = torch.where(
                    sel, torch.full_like(offset_per_gauss, off), offset_per_gauss
                )
        global_idx = offset_per_gauss + view_n * P + patch_idx  # [N_g]
        # Clamp to keep indexing safe for invisible / fallback pointers.
        global_idx = global_idx.clamp_(min=0, max=flat_lut.shape[0] - 1)
        return flat_lut, global_idx

    def _rasterize_chunked(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,      # [N_g, C]
        viewmats: torch.Tensor,    # [S, 4, 4]
        Ks: torch.Tensor,          # [S, 3, 3]
        H: int,
        W: int,
    ) -> torch.Tensor:
        C = colors.shape[-1]
        S = viewmats.shape[0]
        chunks: list[torch.Tensor] = []
        for start in range(0, C, self.chunk_channels):
            end = min(start + self.chunk_channels, C)
            colors_chunk = colors[:, start:end].contiguous()
            rendered_chunk, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors_chunk,
                viewmats=viewmats.float(),
                Ks=Ks.float(),
                width=int(W),
                height=int(H),
                render_mode="RGB",
            )  # [S, H, W, end-start]
            chunks.append(rendered_chunk)
        return torch.cat(chunks, dim=-1)  # [S, H, W, C]

    @staticmethod
    def _area_pool(rendered: torch.Tensor, pool_to: int) -> torch.Tensor:
        """Area-pool `[S, H, W, C]` down to `[S, pool_to*pool_to, C]`."""
        S, H, W, C = rendered.shape
        k_h = H // pool_to
        k_w = W // pool_to
        x = rendered.permute(0, 3, 1, 2).contiguous()        # [S, C, H, W]
        x = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w))
        # x: [S, C, pool_to, pool_to]
        x = x.permute(0, 2, 3, 1).contiguous()               # [S, pool_to, pool_to, C]
        return x.reshape(S, pool_to * pool_to, C)

    def _validate_inputs(
        self,
        gaussians_dggt: Sequence[Mapping[str, torch.Tensor]],
        pointers: Sequence[GaussianPointers],
        lut_scene: Sequence[torch.Tensor],
        lut_asset_dict: Mapping[int, Sequence[torch.Tensor]] | None,
        cameras_dggt: Mapping[str, torch.Tensor],
    ) -> None:
        if len(lut_scene) != self.num_levels:
            raise ValueError(f"lut_scene must have {self.num_levels} levels, got {len(lut_scene)}")
        if lut_asset_dict is not None:
            for k, lvls in lut_asset_dict.items():
                if len(lvls) != self.num_levels:
                    raise ValueError(
                        f"lut_asset_dict[{k}] must have {self.num_levels} levels, got {len(lvls)}"
                    )

        B = len(gaussians_dggt)
        if len(pointers) != B:
            raise ValueError(f"pointers length ({len(pointers)}) != gaussians_dggt length ({B})")
        if lut_scene[0].shape[0] != B:
            raise ValueError(
                f"lut_scene batch dim ({lut_scene[0].shape[0]}) != gaussians_dggt length ({B})"
            )

        if "viewmats" not in cameras_dggt or "Ks" not in cameras_dggt:
            raise ValueError("cameras_dggt must contain 'viewmats' and 'Ks'")
        viewmats = cameras_dggt["viewmats"]
        Ks = cameras_dggt["Ks"]
        if viewmats.dim() != 4 or viewmats.shape[-2:] != (4, 4):
            raise ValueError(f"viewmats must be [B,S,4,4], got {tuple(viewmats.shape)}")
        if Ks.dim() != 4 or Ks.shape[-2:] != (3, 3):
            raise ValueError(f"Ks must be [B,S,3,3], got {tuple(Ks.shape)}")
        if viewmats.shape[0] != B or Ks.shape[0] != B:
            raise ValueError(
                f"camera batch dim must match B={B}, got viewmats={viewmats.shape[0]}, "
                f"Ks={Ks.shape[0]}"
            )

        for b in range(B):
            g = gaussians_dggt[b]
            for key in ("means", "quats", "scales", "opacities"):
                if key not in g:
                    raise ValueError(f"gaussians_dggt[{b}] missing key {key!r}")
            N_g = g["means"].shape[0]
            for attr in ("src_kind", "object_id", "view_n", "patch_idx", "visible_mask"):
                arr = getattr(pointers[b], attr)
                if arr.shape[0] != N_g:
                    raise ValueError(
                        f"pointers[{b}].{attr} length ({arr.shape[0]}) != means length ({N_g})"
                    )
