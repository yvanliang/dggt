"""Frozen single-reference appearance encoder for factorized asset control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from dggt.utils.factorized_asset_condition import (
    MAX_CANONICAL_APPEARANCE_TOKENS,
    alpha_to_patch_mask,
    sample_canonical_tokens,
)
from dggt.utils.tokens import select_patch_pyramid


CANONICAL_ASSET_ENCODER_LEVELS = (4, 11, 17, 23)


@dataclass(frozen=True)
class CanonicalAppearanceEncoding:
    appearance_tokens: torch.Tensor
    appearance_mask: torch.Tensor
    canonical_uv: torch.Tensor


class CanonicalAssetEncoder(nn.Module):
    """Encode isolated canonical references with DGGT at sequence length one.

    The API intentionally cannot accept a target clip, trajectory, target
    latent, target bbox, or target mask.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        scene_tokenizer: nn.Module,
        scene_flow_normalizer: nn.Module,
        *,
        patch_grid: tuple[int, int],
        levels: Sequence[int] = CANONICAL_ASSET_ENCODER_LEVELS,
        max_tokens: int = MAX_CANONICAL_APPEARANCE_TOKENS,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.scene_tokenizer = scene_tokenizer
        self._normalize_fn: Callable[[torch.Tensor], torch.Tensor] = scene_flow_normalizer.normalize
        self.patch_grid = (int(patch_grid[0]), int(patch_grid[1]))
        self.levels = tuple(int(value) for value in levels)
        self.max_tokens = int(max_tokens)
        if len(self.levels) != 4:
            raise ValueError(f"CanonicalAssetEncoder requires four tokenizer levels, got {self.levels}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        for module in (self.aggregator, self.scene_tokenizer):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "CanonicalAssetEncoder":
        # This adapter may live beside a training SceneFlow module, but the
        # shared DGGT/tokenizer always remain frozen and deterministic.
        super().train(mode)
        self.aggregator.eval()
        self.scene_tokenizer.eval()
        return self

    def forward(
        self,
        canonical_rgb: torch.Tensor,
        canonical_alpha: torch.Tensor,
        *,
        batch_size: int,
        num_assets: int,
    ) -> CanonicalAppearanceEncoding:
        if canonical_rgb.ndim != 5 or int(canonical_rgb.shape[1]) != 1 or int(canonical_rgb.shape[2]) != 3:
            raise ValueError(
                "canonical_rgb must be [B*K,1,3,H,W], got "
                f"{tuple(canonical_rgb.shape)}"
            )
        if (
            canonical_alpha.ndim != 5
            or int(canonical_alpha.shape[1]) != 1
            or int(canonical_alpha.shape[2]) != 1
            or canonical_alpha.shape[-2:] != canonical_rgb.shape[-2:]
        ):
            raise ValueError(
                "canonical_alpha must be [B*K,1,1,H,W] matching canonical_rgb, got "
                f"{tuple(canonical_alpha.shape)}"
            )
        n = int(canonical_rgb.shape[0])
        if n != int(batch_size) * int(num_assets):
            raise ValueError(
                f"canonical reference count {n} != batch_size*num_assets "
                f"{int(batch_size) * int(num_assets)}"
            )
        with torch.no_grad():
            _, image_tokens, _, _, patch_start_idx = self.aggregator(canonical_rgb)
            levels = select_patch_pyramid(image_tokens, self.levels, int(patch_start_idx))
            encoded = self.scene_tokenizer.encode(levels, patch_grid=self.patch_grid)
            if encoded.ndim != 4 or tuple(encoded.shape[:2]) != (n, 1):
                raise ValueError(
                    "single-reference tokenizer output must be [B*K,1,P,Ca], got "
                    f"{tuple(encoded.shape)}"
                )
            normalized = self._normalize_fn(encoded.float())[:, 0]

        patch_mask = alpha_to_patch_mask(
            canonical_alpha[:, 0],
            self.patch_grid,
        ).to(device=normalized.device)
        tokens, mask, uv = sample_canonical_tokens(
            normalized,
            patch_mask,
            self.patch_grid,
            max_tokens=self.max_tokens,
        )
        q, channels = int(tokens.shape[1]), int(tokens.shape[2])
        return CanonicalAppearanceEncoding(
            appearance_tokens=tokens.reshape(int(batch_size), int(num_assets), q, channels),
            appearance_mask=mask.reshape(int(batch_size), int(num_assets), q),
            canonical_uv=uv.reshape(int(batch_size), int(num_assets), q, 2),
        )
