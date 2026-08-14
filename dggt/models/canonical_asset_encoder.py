"""Frozen single-reference encoder for non-positional appearance bindings."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import torch
import torch.nn as nn

from dggt.utils.appearance_binding_condition import (
    MAX_APPEARANCE_TOKENS,
    appearance_alpha_to_patch_mask,
    sample_appearance_tokens,
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
        max_tokens: int = MAX_APPEARANCE_TOKENS,
        cache_size: int = 1024,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.scene_tokenizer = scene_tokenizer
        self._normalize_fn: Callable[[torch.Tensor], torch.Tensor] = scene_flow_normalizer.normalize
        self.patch_grid = (int(patch_grid[0]), int(patch_grid[1]))
        self.levels = tuple(int(value) for value in levels)
        self.max_tokens = int(max_tokens)
        self.cache_size = int(cache_size)
        self._appearance_cache: OrderedDict[
            Hashable,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        normalizer_config = getattr(scene_flow_normalizer, "config", None)
        configured_asset_dim = getattr(normalizer_config, "asset_dim", None)
        self._asset_dim = (
            None
            if configured_asset_dim is None
            else int(configured_asset_dim)
        )
        if len(self.levels) != 4:
            raise ValueError(f"CanonicalAssetEncoder requires four tokenizer levels, got {self.levels}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {cache_size}")
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
        cache_keys: Sequence[Hashable | None] | None = None,
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
        if cache_keys is not None and len(cache_keys) != n:
            raise ValueError(
                f"cache_keys length {len(cache_keys)} != canonical references {n}"
            )
        patch_mask = appearance_alpha_to_patch_mask(
            canonical_alpha[:, 0],
            self.patch_grid,
        )
        active_indices = torch.nonzero(
            patch_mask.any(dim=-1),
            as_tuple=False,
        ).flatten().detach().cpu().tolist()
        cached_rows = {}
        missing_indices = []
        for row in active_indices:
            key = None if cache_keys is None else cache_keys[row]
            if (
                key is not None
                and self.cache_size > 0
                and key in self._appearance_cache
            ):
                value = self._appearance_cache.pop(key)
                self._appearance_cache[key] = value
                cached_rows[row] = value
            else:
                missing_indices.append(row)

        encoded_rows = {}
        if missing_indices:
            selected = torch.tensor(
                missing_indices,
                device=canonical_rgb.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                _, image_tokens, _, _, patch_start_idx = self.aggregator(
                    canonical_rgb.index_select(0, selected)
                )
                levels = select_patch_pyramid(
                    image_tokens,
                    self.levels,
                    int(patch_start_idx),
                )
                encoded = self.scene_tokenizer.encode(
                    levels,
                    patch_grid=self.patch_grid,
                )
                expected_prefix = (len(missing_indices), 1)
                if (
                    encoded.ndim != 4
                    or tuple(encoded.shape[:2]) != expected_prefix
                ):
                    raise ValueError(
                        "single-reference tokenizer output must be "
                        "[N_active,1,P,Ca], got "
                        f"{tuple(encoded.shape)}"
                    )
                normalized = self._normalize_fn(encoded.float())[:, 0]
            selected_patch_mask = patch_mask.index_select(
                0,
                selected.to(device=patch_mask.device),
            ).to(device=normalized.device)
            tokens, mask, uv = sample_appearance_tokens(
                normalized,
                selected_patch_mask,
                self.patch_grid,
                max_tokens=self.max_tokens,
            )
            self._asset_dim = int(tokens.shape[-1])
            for local_row, original_row in enumerate(missing_indices):
                value = (
                    tokens[local_row],
                    mask[local_row],
                    uv[local_row],
                )
                encoded_rows[original_row] = value
                key = (
                    None
                    if cache_keys is None
                    else cache_keys[original_row]
                )
                if key is not None and self.cache_size > 0:
                    self._appearance_cache[key] = tuple(
                        tensor.detach().cpu()
                        for tensor in value
                    )
                    while len(self._appearance_cache) > self.cache_size:
                        self._appearance_cache.popitem(last=False)

        asset_dim = self._asset_dim
        if asset_dim is None:
            raise RuntimeError(
                "Cannot infer canonical asset width from an all-empty batch; "
                "the SceneFlow normalizer must expose config.asset_dim."
            )
        output_tokens = torch.zeros(
            (n, self.max_tokens, asset_dim),
            device=canonical_rgb.device,
            dtype=torch.float32,
        )
        output_mask = torch.zeros(
            (n, self.max_tokens),
            device=canonical_rgb.device,
            dtype=torch.bool,
        )
        output_uv = torch.zeros(
            (n, self.max_tokens, 2),
            device=canonical_rgb.device,
            dtype=torch.float32,
        )
        for row, value in {**cached_rows, **encoded_rows}.items():
            row_tokens, row_mask, row_uv = value
            output_tokens[row] = row_tokens.to(
                device=canonical_rgb.device,
                dtype=torch.float32,
            )
            output_mask[row] = row_mask.to(
                device=canonical_rgb.device,
                dtype=torch.bool,
            )
            output_uv[row] = row_uv.to(
                device=canonical_rgb.device,
                dtype=torch.float32,
            )
        return CanonicalAppearanceEncoding(
            appearance_tokens=output_tokens.reshape(
                int(batch_size),
                int(num_assets),
                self.max_tokens,
                asset_dim,
            ),
            appearance_mask=output_mask.reshape(
                int(batch_size),
                int(num_assets),
                self.max_tokens,
            ),
            canonical_uv=output_uv.reshape(
                int(batch_size),
                int(num_assets),
                self.max_tokens,
                2,
            ),
        )
