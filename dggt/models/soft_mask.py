"""SoftMaskBuilder: render coverage maps and produce soft edit masks.

Given the three Gaussian groups produced by `GaussianSceneEditor`:

* `G_kept`     — primitives preserved from the clean scene
* `G_deleted`  — primitives about to be removed (rendered at the *target* view
  to localize the deletion footprint in each frame)
* `G_asset`    — inserted / replacing primitives, organized per object

we rasterize alpha-only maps `K_map`, `D_map`, `I_map` at full resolution, then
area-pool them to a 37x37 patch grid and normalize so the three soft masks sum
to (almost) one per patch.

Note: The occlusion semantics here is intentional footprint normalization, not
full scene depth compositing. This explicitly expresses the "edit footprint"
rather than strict visibility against the kept scene.  For Mode A insertions the
scaffold depth follows the same edited visibility rule: inserted objects are
composited over kept DGGT Gaussians, while inserted objects still resolve depth
among themselves.  Mode B deletion holes use a different helper that clears
depth inside the imagined hole footprint.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gsplat.rendering import rasterization as _gsplat_rasterization
except ModuleNotFoundError:  # pragma: no cover - exercised only on CPU-only dev envs
    _gsplat_rasterization = None


def _rasterization(*args, **kwargs):
    if _gsplat_rasterization is None:
        raise ModuleNotFoundError("SoftMaskBuilder requires the `gsplat` package for rasterization.")
    return _gsplat_rasterization(*args, **kwargs)


class SoftMaskBuilder(nn.Module):
    """Pure-function coverage renderer + pooler. No parameters."""

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def render_coverage(
        self,
        G_kept: Sequence[Mapping[str, torch.Tensor]],
        G_deleted: Sequence[Mapping[str, torch.Tensor]],
        G_asset_dggt_dict: Sequence[Mapping[int, Mapping[str, torch.Tensor]]],
        cameras_dggt: Mapping[str, torch.Tensor],
        H: int = 518,
        W: int = 518,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[int, torch.Tensor]]]:
        """Rasterize K/D/I alpha maps at the target camera views.

        Parameters
        ----------
        G_kept, G_deleted
            Length-B sequences. Each entry is a gaussian dict
            (`means`, `quats`, `scales`, `opacities`). Empty dicts are allowed
            (return zero alpha).
        G_asset_dggt_dict
            Length-B sequence of per-object gaussian dicts, keyed by asset
            object_id. Assets are already in DGGT-coord (post `T_w2d`).
        cameras_dggt
            `viewmats [B, S, 4, 4]`, `Ks [B, S, 3, 3]`.
        H, W
            Full rendering resolution (default 518).

        Returns
        -------
        K_map, D_map, I_map : Tensor [B, S, H, W, 1]
        I_map_per_obj       : length-B list of `dict[int, Tensor[S, H, W, 1]]`
        """
        self._validate_cameras(cameras_dggt, expected_B=len(G_kept))
        viewmats_all = cameras_dggt["viewmats"]
        Ks_all = cameras_dggt["Ks"]
        device = viewmats_all.device
        dtype = torch.float32

        B, S = viewmats_all.shape[:2]
        K_list: list[torch.Tensor] = []
        D_list: list[torch.Tensor] = []
        I_list: list[torch.Tensor] = []
        I_per_obj_list: list[dict[int, torch.Tensor]] = []

        for b in range(B):
            viewmats = viewmats_all[b]  # [S, 4, 4]
            Ks = Ks_all[b]              # [S, 3, 3]

            K_map_b = self._render_alpha(G_kept[b], viewmats, Ks, H, W, device, dtype)
            D_map_b = self._render_alpha(G_deleted[b], viewmats, Ks, H, W, device, dtype)

            I_map_b, per_obj = self._render_asset_owner_alpha(
                G_asset_dggt_dict[b],
                viewmats,
                Ks,
                H,
                W,
                device,
                dtype,
            )

            K_list.append(K_map_b)
            D_list.append(D_map_b)
            I_list.append(I_map_b)
            I_per_obj_list.append(per_obj)

        K_map = torch.stack(K_list, dim=0)  # [B, S, H, W, 1]
        D_map = torch.stack(D_list, dim=0)
        I_map = torch.stack(I_list, dim=0)
        return K_map, D_map, I_map, I_per_obj_list

    def render_coverage_and_effective_depth(
        self,
        G_kept: Sequence[Mapping[str, torch.Tensor]],
        G_deleted: Sequence[Mapping[str, torch.Tensor]],
        G_asset_dggt_dict: Sequence[Mapping[int, Mapping[str, torch.Tensor]]],
        cameras_dggt: Mapping[str, torch.Tensor],
        H: int = 518,
        W: int = 518,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[int, torch.Tensor]],
        torch.Tensor,
    ]:
        """Render K/D/I coverage and the scaffold's effective edited depth.

        `D_edited` intentionally follows the final edit visibility semantics, not
        a physical z-buffer over `G_kept + G_asset`.  Inserted / imagined objects
        are rendered together to resolve their mutual depth ordering, then
        composited over kept-scene depth by their visible alpha.  This keeps the
        depth channel aligned with the visual/mask contract where DGGT foreground
        Gaussians do not hide an inserted target merely because their estimated
        depth is closer.
        """
        self._validate_cameras(cameras_dggt, expected_B=len(G_kept))
        viewmats_all = cameras_dggt["viewmats"]
        Ks_all = cameras_dggt["Ks"]
        device = viewmats_all.device
        dtype = torch.float32

        B, S = viewmats_all.shape[:2]
        K_list: list[torch.Tensor] = []
        D_list: list[torch.Tensor] = []
        I_list: list[torch.Tensor] = []
        depth_list: list[torch.Tensor] = []
        I_per_obj_list: list[dict[int, torch.Tensor]] = []

        for b in range(B):
            viewmats = viewmats_all[b]
            Ks = Ks_all[b]

            K_alpha_b, K_depth_b = self._render_alpha_depth(
                G_kept[b], viewmats, Ks, H, W, device, dtype
            )
            D_map_b = self._render_alpha(G_deleted[b], viewmats, Ks, H, W, device, dtype)
            I_map_b, per_obj = self._render_asset_owner_alpha(
                G_asset_dggt_dict[b],
                viewmats,
                Ks,
                H,
                W,
                device,
                dtype,
            )
            _I_alpha_depth_b, I_depth_b = self._render_asset_alpha_depth(
                G_asset_dggt_dict[b],
                viewmats,
                Ks,
                H,
                W,
                device,
                dtype,
            )
            D_edited_b = self.compose_effective_edited_depth(
                K_alpha=K_alpha_b,
                K_depth=K_depth_b,
                I_alpha=I_map_b,
                I_depth=I_depth_b,
            )

            K_list.append(K_alpha_b)
            D_list.append(D_map_b)
            I_list.append(I_map_b)
            I_per_obj_list.append(per_obj)
            depth_list.append(D_edited_b)

        K_map = torch.stack(K_list, dim=0)
        D_map = torch.stack(D_list, dim=0)
        I_map = torch.stack(I_list, dim=0)
        D_edited = torch.stack(depth_list, dim=0)
        return K_map, D_map, I_map, I_per_obj_list, D_edited

    def pool_and_normalize(
        self,
        K_map: torch.Tensor,
        D_map: torch.Tensor,
        I_map: torch.Tensor,
        target_grid: int | tuple[int, int] = 37,
        eps: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Area-pool coverage maps to `target_grid` and normalize to soft masks.

        Inputs shape `[B, S, H, W, 1]`; outputs shape `[B, S, grid_h*grid_w, 1]`.
        `M_preserve + M_source + M_dest` sums to `1 - eps/(K+D+I+eps)`
        (≈ 1 where coverage exists, 0 where nothing was rendered).
        """
        if K_map.shape != D_map.shape or K_map.shape != I_map.shape:
            raise ValueError(
                f"K/D/I shape mismatch: {tuple(K_map.shape)} vs "
                f"{tuple(D_map.shape)} vs {tuple(I_map.shape)}"
            )
        B, S, H, W, C = K_map.shape
        if C != 1:
            raise ValueError(f"Expected trailing dim=1 (alpha), got {C}")
        grid_h, grid_w = self._normalize_grid(target_grid)
        if H % grid_h != 0 or W % grid_w != 0:
            raise ValueError(
                f"H ({H}) and W ({W}) must be divisible by target_grid ({(grid_h, grid_w)})"
            )

        K_pool = self._area_pool_to_grid(K_map, (grid_h, grid_w))  # [B, S, P, 1]
        D_pool = self._area_pool_to_grid(D_map, (grid_h, grid_w))
        I_pool = self._area_pool_to_grid(I_map, (grid_h, grid_w))

        total = K_pool + D_pool + I_pool + eps
        M_preserve = K_pool / total
        M_source = D_pool / total
        M_dest = I_pool / total
        return M_preserve, M_source, M_dest

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_alpha(
        gauss: Mapping[str, torch.Tensor] | None,
        viewmats: torch.Tensor,   # [S, 4, 4]
        Ks: torch.Tensor,         # [S, 3, 3]
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return `[S, H, W, 1]` alpha map; zero when gaussians are empty."""
        S = viewmats.shape[0]
        zero = torch.zeros((S, H, W, 1), dtype=dtype, device=device)
        if gauss is None:
            return zero
        means = gauss.get("means")
        if means is None or means.numel() == 0:
            return zero
        means = means.to(device).detach().float()
        quats = gauss["quats"].to(device).detach().float()
        scales = gauss["scales"].to(device).detach().float()
        opacities = gauss["opacities"].to(device).detach().float().view(-1)

        probe_colors = torch.ones((means.shape[0], 1), dtype=dtype, device=device)
        _, alphas, _ = _rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=probe_colors,
            viewmats=viewmats.float(),
            Ks=Ks.float(),
            width=int(W),
            height=int(H),
            render_mode="RGB",
        )
        # alphas: [S, H, W, 1], already in [0, 1]
        return alphas.clamp(0.0, 1.0)

    @staticmethod
    def _render_alpha_depth(
        gauss: Mapping[str, torch.Tensor] | None,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return alpha and expected depth, both `[S, H, W, 1]`.

        Depth is zero where the rendered alpha is effectively empty.
        """
        S = viewmats.shape[0]
        zero = torch.zeros((S, H, W, 1), dtype=dtype, device=device)
        if gauss is None:
            return zero, zero.clone()
        means = gauss.get("means")
        if means is None or means.numel() == 0:
            return zero, zero.clone()
        means = means.to(device).detach().float()
        quats = gauss["quats"].to(device).detach().float()
        scales = gauss["scales"].to(device).detach().float()
        opacities = gauss["opacities"].to(device).detach().float().view(-1)

        probe_colors = torch.zeros((means.shape[0], 3), dtype=dtype, device=device)
        rendered, alphas, _ = _rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=probe_colors,
            viewmats=viewmats.float(),
            Ks=Ks.float(),
            width=int(W),
            height=int(H),
            render_mode="RGB+ED",
        )
        alpha = alphas.clamp(0.0, 1.0)
        depth = rendered[..., -1:].to(dtype=dtype)
        valid = torch.isfinite(depth) & (alpha > 1e-6) & (depth > 0.0)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        return alpha, depth

    @staticmethod
    def _render_asset_owner_alpha(
        asset_dict: Mapping[int, Mapping[str, torch.Tensor]] | None,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Render inserted assets together so only asset-asset visibility is resolved."""
        S = int(viewmats.shape[0])
        zero = torch.zeros((S, H, W, 1), dtype=dtype, device=device)
        if asset_dict is None:
            return zero, {}

        assets = {int(k): v for k, v in asset_dict.items()}
        all_keys = sorted(assets.keys())
        nonempty_keys = [
            k
            for k in all_keys
            if assets[k].get("means") is not None and assets[k]["means"].numel() > 0
        ]
        if len(nonempty_keys) == 0:
            return zero, {k: zero.clone() for k in all_keys}

        means_chunks: list[torch.Tensor] = []
        quats_chunks: list[torch.Tensor] = []
        scales_chunks: list[torch.Tensor] = []
        opacities_chunks: list[torch.Tensor] = []
        color_chunks: list[torch.Tensor] = []
        channel_by_key = {k: i for i, k in enumerate(nonempty_keys)}

        for obj_key in nonempty_keys:
            gauss = assets[obj_key]
            means = gauss["means"].to(device).detach().float()
            n = int(means.shape[0])
            means_chunks.append(means)
            quats_chunks.append(gauss["quats"].to(device).detach().float())
            scales_chunks.append(gauss["scales"].to(device).detach().float())
            opacities_chunks.append(gauss["opacities"].to(device).detach().float().view(-1))
            colors = torch.zeros((n, len(nonempty_keys)), dtype=dtype, device=device)
            colors[:, channel_by_key[obj_key]] = 1.0
            color_chunks.append(colors)

        rendered, _, _ = _rasterization(
            means=torch.cat(means_chunks, dim=0),
            quats=torch.cat(quats_chunks, dim=0),
            scales=torch.cat(scales_chunks, dim=0),
            opacities=torch.cat(opacities_chunks, dim=0),
            colors=torch.cat(color_chunks, dim=0),
            viewmats=viewmats.float(),
            Ks=Ks.float(),
            width=int(W),
            height=int(H),
            render_mode="RGB",
            channel_chunk=min(len(nonempty_keys), 32),
        )
        owner = rendered.clamp(0.0, 1.0)
        per_obj: dict[int, torch.Tensor] = {}
        for obj_key in all_keys:
            channel = channel_by_key.get(obj_key)
            per_obj[obj_key] = zero.clone() if channel is None else owner[..., channel : channel + 1]
        I_map = owner.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        return I_map, per_obj

    @staticmethod
    def _render_asset_alpha_depth(
        asset_dict: Mapping[int, Mapping[str, torch.Tensor]] | None,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render all inserted / imagined assets together for alpha + depth."""
        S = int(viewmats.shape[0])
        zero = torch.zeros((S, H, W, 1), dtype=dtype, device=device)
        if asset_dict is None:
            return zero, zero.clone()

        means_chunks: list[torch.Tensor] = []
        quats_chunks: list[torch.Tensor] = []
        scales_chunks: list[torch.Tensor] = []
        opacities_chunks: list[torch.Tensor] = []
        for gauss in asset_dict.values():
            means = gauss.get("means")
            if means is None or means.numel() == 0:
                continue
            means = means.to(device).detach().float()
            means_chunks.append(means)
            quats_chunks.append(gauss["quats"].to(device).detach().float())
            scales_chunks.append(gauss["scales"].to(device).detach().float())
            opacities_chunks.append(gauss["opacities"].to(device).detach().float().view(-1))

        if len(means_chunks) == 0:
            return zero, zero.clone()

        means_all = torch.cat(means_chunks, dim=0)
        probe_colors = torch.zeros((means_all.shape[0], 3), dtype=dtype, device=device)
        rendered, alphas, _ = _rasterization(
            means=means_all,
            quats=torch.cat(quats_chunks, dim=0),
            scales=torch.cat(scales_chunks, dim=0),
            opacities=torch.cat(opacities_chunks, dim=0),
            colors=probe_colors,
            viewmats=viewmats.float(),
            Ks=Ks.float(),
            width=int(W),
            height=int(H),
            render_mode="RGB+ED",
        )
        alpha = alphas.clamp(0.0, 1.0)
        depth = rendered[..., -1:].to(dtype=dtype)
        valid = torch.isfinite(depth) & (alpha > 1e-6) & (depth > 0.0)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        return alpha, depth

    @staticmethod
    def compose_effective_edited_depth(
        *,
        K_alpha: torch.Tensor,
        K_depth: torch.Tensor,
        I_alpha: torch.Tensor,
        I_depth: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Depth aligned to edit visibility: inserted/imagined over kept.

        This deliberately does not compare `I_depth` against `K_depth`.  If an
        inserted object is visible according to the edit footprint, its depth is
        used even when the DGGT kept scene has a closer estimated depth.
        """
        if K_alpha.shape != K_depth.shape or I_alpha.shape != I_depth.shape:
            raise ValueError(
                f"alpha/depth shape mismatch: K {tuple(K_alpha.shape)} / {tuple(K_depth.shape)}, "
                f"I {tuple(I_alpha.shape)} / {tuple(I_depth.shape)}"
            )
        if K_alpha.shape != I_alpha.shape:
            raise ValueError(
                f"K/I shape mismatch: {tuple(K_alpha.shape)} vs {tuple(I_alpha.shape)}"
            )

        k_alpha = K_alpha.clamp(0.0, 1.0)
        i_alpha = I_alpha.clamp(0.0, 1.0)
        k_depth = torch.where(torch.isfinite(K_depth), K_depth, torch.zeros_like(K_depth))
        i_depth = torch.where(torch.isfinite(I_depth), I_depth, torch.zeros_like(I_depth))

        has_k = (k_alpha > float(eps)) & (k_depth > 0.0)
        has_i = (i_alpha > float(eps)) & (i_depth > 0.0)
        over_kept = i_alpha * i_depth + (1.0 - i_alpha) * k_depth
        inserted_visible = torch.where(has_k, over_kept, i_depth)
        depth = torch.where(has_i, inserted_visible, torch.where(has_k, k_depth, torch.zeros_like(k_depth)))
        return torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))

    @staticmethod
    def compose_deleted_hole_depth(
        *,
        K_alpha: torch.Tensor,
        K_depth: torch.Tensor,
        hole_alpha: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """Depth for a deletion-hole edit.

        `hole_alpha` is an edit footprint, not a visible inserted object.  Pixels
        covered by the footprint are intentionally marked unknown/empty so the
        downstream diffusion model learns to complete the hole instead of seeing
        either the deleted Gaussian depth or a background depth leak.
        """
        if K_alpha.shape != K_depth.shape or K_alpha.shape != hole_alpha.shape:
            raise ValueError(
                f"K_alpha/K_depth/hole_alpha shape mismatch: "
                f"{tuple(K_alpha.shape)} / {tuple(K_depth.shape)} / {tuple(hole_alpha.shape)}"
            )
        k_alpha = K_alpha.clamp(0.0, 1.0)
        k_depth = torch.where(torch.isfinite(K_depth), K_depth, torch.zeros_like(K_depth))
        keep_valid = (k_alpha > float(eps)) & (k_depth > 0.0)
        outside_hole = hole_alpha.clamp(0.0, 1.0) <= float(eps)
        depth = torch.where(keep_valid & outside_hole, k_depth, torch.zeros_like(k_depth))
        return torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))

    @staticmethod
    def compose_deleted_hole_alpha(
        *,
        K_alpha: torch.Tensor,
        hole_alpha: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """Edited-scene alpha for a deletion-hole edit."""
        if K_alpha.shape != hole_alpha.shape:
            raise ValueError(
                f"K_alpha/hole_alpha shape mismatch: {tuple(K_alpha.shape)} / {tuple(hole_alpha.shape)}"
            )
        outside_hole = hole_alpha.clamp(0.0, 1.0) <= float(eps)
        return torch.where(outside_hole, K_alpha.clamp(0.0, 1.0), torch.zeros_like(K_alpha))

    @staticmethod
    def _normalize_grid(grid: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(grid, int):
            return int(grid), int(grid)
        return int(grid[0]), int(grid[1])

    @staticmethod
    def _area_pool_to_grid(x: torch.Tensor, target_grid: int | tuple[int, int]) -> torch.Tensor:
        """Area-pool `[B, S, H, W, 1]` to `[B, S, grid_h*grid_w, 1]`."""
        B, S, H, W, C = x.shape
        grid_h, grid_w = SoftMaskBuilder._normalize_grid(target_grid)
        k_h = H // grid_h
        k_w = W // grid_w
        y = x.reshape(B * S, H, W, C).permute(0, 3, 1, 2)  # [B*S, C, H, W]
        y = F.avg_pool2d(y, kernel_size=(k_h, k_w), stride=(k_h, k_w))
        # y: [B*S, C, grid_h, grid_w]
        y = y.permute(0, 2, 3, 1).reshape(B, S, grid_h * grid_w, C)
        return y

    @staticmethod
    def _validate_cameras(cameras_dggt: Mapping[str, torch.Tensor], expected_B: int) -> None:
        if "viewmats" not in cameras_dggt or "Ks" not in cameras_dggt:
            raise ValueError("cameras_dggt must contain 'viewmats' and 'Ks'")
        viewmats = cameras_dggt["viewmats"]
        Ks = cameras_dggt["Ks"]
        if viewmats.dim() != 4 or viewmats.shape[-2:] != (4, 4):
            raise ValueError(f"viewmats must be [B,S,4,4], got {tuple(viewmats.shape)}")
        if Ks.dim() != 4 or Ks.shape[-2:] != (3, 3):
            raise ValueError(f"Ks must be [B,S,3,3], got {tuple(Ks.shape)}")
        if viewmats.shape[0] != expected_B:
            raise ValueError(
                f"viewmats batch dim ({viewmats.shape[0]}) != expected B ({expected_B})"
            )
