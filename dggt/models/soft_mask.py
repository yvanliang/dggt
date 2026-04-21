"""SoftMaskBuilder: render coverage maps and produce soft edit masks.

Given the three Gaussian groups produced by `GaussianSceneEditor`:

* `G_kept`     — primitives preserved from the clean scene
* `G_deleted`  — primitives about to be removed (rendered at the *target* view
  to localize the deletion footprint in each frame)
* `G_asset`    — inserted / replacing primitives, organized per object

we rasterize alpha-only maps `K_map`, `D_map`, `I_map` at full resolution, then
area-pool them to a 37x37 patch grid and normalize so the three soft masks sum
to (almost) one per patch.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.rendering import rasterization


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

            per_obj: dict[int, torch.Tensor] = {}
            if G_asset_dggt_dict[b] is not None:
                for obj_key, gauss in G_asset_dggt_dict[b].items():
                    per_obj[int(obj_key)] = self._render_alpha(
                        gauss, viewmats, Ks, H, W, device, dtype
                    )
            if len(per_obj) > 0:
                I_map_b = torch.clamp(
                    torch.stack(list(per_obj.values()), dim=0).sum(dim=0),
                    min=0.0,
                    max=1.0,
                )
            else:
                I_map_b = torch.zeros((S, H, W, 1), dtype=dtype, device=device)

            K_list.append(K_map_b)
            D_list.append(D_map_b)
            I_list.append(I_map_b)
            I_per_obj_list.append(per_obj)

        K_map = torch.stack(K_list, dim=0)  # [B, S, H, W, 1]
        D_map = torch.stack(D_list, dim=0)
        I_map = torch.stack(I_list, dim=0)
        return K_map, D_map, I_map, I_per_obj_list

    def pool_and_normalize(
        self,
        K_map: torch.Tensor,
        D_map: torch.Tensor,
        I_map: torch.Tensor,
        target_grid: int = 37,
        eps: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Area-pool coverage maps to `target_grid` and normalize to soft masks.

        Inputs shape `[B, S, H, W, 1]`; outputs shape `[B, S, target_grid**2, 1]`.
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
        if H % target_grid != 0 or W % target_grid != 0:
            raise ValueError(
                f"H ({H}) and W ({W}) must be divisible by target_grid ({target_grid})"
            )

        K_pool = self._area_pool_to_grid(K_map, target_grid)  # [B, S, P, 1]
        D_pool = self._area_pool_to_grid(D_map, target_grid)
        I_pool = self._area_pool_to_grid(I_map, target_grid)

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
        _, alphas, _ = rasterization(
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
    def _area_pool_to_grid(x: torch.Tensor, target_grid: int) -> torch.Tensor:
        """Area-pool `[B, S, H, W, 1]` → `[B, S, target_grid**2, 1]`."""
        B, S, H, W, C = x.shape
        k_h = H // target_grid
        k_w = W // target_grid
        y = x.reshape(B * S, H, W, C).permute(0, 3, 1, 2)  # [B*S, C, H, W]
        y = F.avg_pool2d(y, kernel_size=(k_h, k_w), stride=(k_h, k_w))
        # y: [B*S, C, target_grid, target_grid]
        y = y.permute(0, 2, 3, 1).reshape(B, S, target_grid * target_grid, C)
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
