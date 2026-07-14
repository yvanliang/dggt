"""ScaffoldPacker: pack the 7-channel hires scaffold into 768-d per-patch tokens.

The scaffold carries geometric / mask / temporal priors that condition the
scene flow. Channels (per the implementation plan):

    0 : D_edited        — effective edited-scene depth at target view
    1 : A_edited        — alpha of the edited scene
    2 : K_soft          — preserve coverage (upsampled hires; aligns with gs_head)
    3 : D_soft          — source (deletion) coverage
    4 : I_soft          — dest (insertion) coverage
    5 : dynamic_prior   — Pass-1 dynamic_conf sigmoided, per pixel
    6 : time_index      — normalized frame index in [0, 1]

`ScaffoldPacker.forward` area-pools to the patch grid and runs a shared
MLP to 768-d. `build_scaffold_hires` is a convenience constructor that fills
out all 7 channels from their natural sources.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaffoldPacker(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        out_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_dim <= 0:
            raise ValueError("in_channels and out_dim must be positive")
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        scaffold_hires: torch.Tensor,
        target_grid: int | tuple[int, int] = 37,
        *,
        already_pooled: bool = False,
    ) -> torch.Tensor:
        """Pack a hires scaffold or an already-pooled cache tensor.

        The live path passes ``[B,S,H,W,C]`` and pools it to the requested
        patch grid.  Schema-v9 fast caches persist that pooled representation
        as ``[B,S,P,C]``; accepting it through ``forward`` is important because
        formal multi-GPU training wraps this module in DDP and must not bypass
        the wrapper by calling ``module.mlp`` directly.
        """
        if already_pooled:
            if scaffold_hires.dim() != 4:
                raise ValueError(
                    "already-pooled scaffold must be 4D [B,S,P,C], "
                    f"got {tuple(scaffold_hires.shape)}"
                )
            if int(scaffold_hires.shape[-1]) != self.in_channels:
                raise ValueError(
                    f"already-pooled scaffold last dim {scaffold_hires.shape[-1]} "
                    f"!= in_channels {self.in_channels}"
                )
            return self.mlp(scaffold_hires)

        if scaffold_hires.dim() != 5:
            raise ValueError(
                f"scaffold_hires must be 5D [B,S,H,W,C], got {tuple(scaffold_hires.shape)}"
            )
        B, S, H, W, C = scaffold_hires.shape
        if C != self.in_channels:
            raise ValueError(
                f"scaffold_hires last dim {C} != in_channels {self.in_channels}"
            )
        grid_h, grid_w = self._normalize_grid(target_grid)
        if H % grid_h != 0 or W % grid_w != 0:
            raise ValueError(
                f"H ({H}) and W ({W}) must be divisible by target_grid ({(grid_h, grid_w)})"
            )
        k_h = H // grid_h
        k_w = W // grid_w
        x = scaffold_hires.reshape(B * S, H, W, C).permute(0, 3, 1, 2)   # [B*S, C, H, W]
        x = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w))    # [B*S, C, gh, gw]
        x = x.permute(0, 2, 3, 1).reshape(B, S, grid_h * grid_w, C)
        return self.mlp(x)                                                # [B, S, P, out_dim]

    @staticmethod
    def _normalize_grid(grid: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(grid, int):
            return int(grid), int(grid)
        return int(grid[0]), int(grid[1])

    @staticmethod
    def build_scaffold_hires(
        D_edited: torch.Tensor,          # [B, S, H, W, 1]
        A_edited: torch.Tensor,          # [B, S, H, W, 1]
        K_map: torch.Tensor,             # [B, S, H, W, 1]
        D_map: torch.Tensor,             # [B, S, H, W, 1]
        I_map: torch.Tensor,             # [B, S, H, W, 1]
        dynamic_prior: torch.Tensor,     # [B, S, H, W, 1] (e.g. sigmoided dynamic_conf)
        time_index: torch.Tensor | None = None,  # [B, S] or None
        depth_scale: float | None = None,
    ) -> torch.Tensor:
        """Assemble the 7-channel hires scaffold from its components.

        All spatial tensors must share the same `[B, S, H, W, 1]` shape. The
        `time_index` channel is broadcast from a per-frame scalar in [0, 1]; if
        omitted it is derived as `s / max(S-1, 1)`.

        `depth_scale` optionally normalizes depth by a fixed scale; when None
        we use per-clip max to keep channels roughly in [0, 1].
        """
        comps = {
            "D_edited": D_edited,
            "A_edited": A_edited,
            "K_map": K_map,
            "D_map": D_map,
            "I_map": I_map,
            "dynamic_prior": dynamic_prior,
        }
        shape_ref = D_edited.shape
        for name, tensor in comps.items():
            if tensor.shape != shape_ref:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} != reference {tuple(shape_ref)}"
                )
        B, S, H, W, _ = shape_ref

        if depth_scale is None:
            with torch.no_grad():
                scale = D_edited.reshape(B, -1).amax(dim=-1).clamp_min(1e-3)
            D_norm = D_edited / scale.view(B, 1, 1, 1, 1)
        else:
            D_norm = D_edited / float(depth_scale)

        if time_index is None:
            denom = float(max(S - 1, 1))
            time_row = torch.arange(S, dtype=D_edited.dtype, device=D_edited.device) / denom
            time_channel = time_row.view(1, S, 1, 1, 1).expand(B, S, H, W, 1)
        else:
            if time_index.shape != (B, S):
                raise ValueError(
                    f"time_index shape {tuple(time_index.shape)} != ({B}, {S})"
                )
            time_channel = time_index.to(D_edited.dtype).view(B, S, 1, 1, 1).expand(B, S, H, W, 1)

        return torch.cat(
            [
                D_norm.clamp(0.0, 1.0),
                A_edited.clamp(0.0, 1.0),
                K_map.clamp(0.0, 1.0),
                D_map.clamp(0.0, 1.0),
                I_map.clamp(0.0, 1.0),
                dynamic_prior.clamp(0.0, 1.0),
                time_channel,
            ],
            dim=-1,
        )
