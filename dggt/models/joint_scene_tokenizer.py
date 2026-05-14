from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from dggt.layers.block import Block
from dggt.layers.rope import (
    PositionGetter,
    RotaryPositionEmbedding1D,
    RotaryPositionEmbedding2D,
)


def _reset_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _resolve_joint_refine_heads(joint_dim: int, preferred_heads: int) -> int:
    if joint_dim % preferred_heads == 0:
        return preferred_heads

    fallback_candidates = [24, 16, 12, 8, 6, 4, 3, 2, 1]
    for candidate in fallback_candidates:
        if joint_dim % candidate == 0:
            return candidate

    raise ValueError(f"Could not find a valid attention head count for joint_dim={joint_dim}")


def _validate_rope_head_dim(dim: int, num_heads: int, module_name: str) -> None:
    if dim % num_heads != 0:
        raise ValueError(f"{module_name}: dim={dim} must be divisible by num_heads={num_heads}")
    head_dim = dim // num_heads
    if head_dim % 4 != 0:
        raise ValueError(
            f"{module_name}: head_dim={head_dim} must be divisible by 4 for 2D RoPE"
        )


def _normalize_patch_grid(
    num_patches: int,
    patch_grid: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if patch_grid is not None:
        patch_h, patch_w = int(patch_grid[0]), int(patch_grid[1])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"Patch grid must be positive, got {patch_grid}")
        if patch_h * patch_w != num_patches:
            raise ValueError(
                f"Patch grid {patch_grid} does not match num_patches={num_patches}"
            )
        return patch_h, patch_w

    best_h = 1
    best_w = num_patches
    best_gap = best_w - best_h
    limit = int(num_patches**0.5)
    for patch_h in range(1, limit + 1):
        if num_patches % patch_h != 0:
            continue
        patch_w = num_patches // patch_h
        gap = abs(patch_w - patch_h)
        if gap < best_gap:
            best_h = patch_h
            best_w = patch_w
            best_gap = gap
    return best_h, best_w


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    omega = torch.arange(embed_dim // 2, device=positions.device, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = positions.reshape(-1).to(dtype=torch.float32)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _get_2d_sincos_pos_embed(
    embed_dim: int,
    patch_h: int,
    patch_w: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    grid_h = torch.arange(patch_h, device=device, dtype=torch.float32)
    grid_w = torch.arange(patch_w, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    return torch.cat([emb_h, emb_w], dim=1)


def _get_cached_2d_sincos_pos_embed(
    cache: dict[tuple[int, int, int, str], torch.Tensor],
    embed_dim: int,
    patch_h: int,
    patch_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cache_key = (embed_dim, patch_h, patch_w, str(device))
    if cache_key not in cache:
        cache[cache_key] = _get_2d_sincos_pos_embed(embed_dim, patch_h, patch_w, device=device)
    return cache[cache_key].to(dtype=dtype).view(1, 1, patch_h * patch_w, embed_dim)


class FrameGlobalBlockPair(nn.Module):
    """Alternates per-frame attention with global cross-frame attention.

    Per-frame attention uses 2D RoPE over patch positions; cross-frame attention
    uses 1D RoPE over frame positions so the latent is sensitive to temporal
    order and generalizes from S=8 (train) to longer clips at inference.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 1e-6,
        rope: RotaryPositionEmbedding2D | None = None,
        temporal_rope: RotaryPositionEmbedding1D | None = None,
    ) -> None:
        super().__init__()
        if rope is not None:
            _validate_rope_head_dim(dim, num_heads, "FrameGlobalBlockPair")
        if temporal_rope is not None and dim % num_heads != 0:
            raise ValueError(
                f"FrameGlobalBlockPair: dim={dim} must be divisible by num_heads={num_heads}"
            )
        if temporal_rope is not None and (dim // num_heads) % 2 != 0:
            raise ValueError(
                f"FrameGlobalBlockPair: head_dim={dim // num_heads} must be even for 1D RoPE"
            )
        self.frame_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            rope=rope,
        )
        self.global_block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            rope=temporal_rope,
        )
        self._uses_temporal_rope = temporal_rope is not None

    def forward(
        self,
        x: torch.Tensor,
        patch_positions: torch.Tensor | None = None,
        frame_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, S, P, D], got shape={tuple(x.shape)}")

        batch_size, seq_len, num_patches, dim = x.shape
        frame_patch_positions = None
        if patch_positions is not None:
            expected_shape = (batch_size, seq_len, num_patches, 2)
            if tuple(patch_positions.shape) != expected_shape:
                raise ValueError(
                    f"Expected patch_positions shape {expected_shape}, got {tuple(patch_positions.shape)}"
                )
            frame_patch_positions = patch_positions.reshape(batch_size * seq_len, num_patches, 2)
        x = x.reshape(batch_size * seq_len, num_patches, dim)
        x = self.frame_block(x, pos=frame_patch_positions)
        x = x.reshape(batch_size, seq_len, num_patches, dim)

        # Cross-frame attention is applied per patch location instead of over the
        # full S*P token product. This keeps memory bounded for longer clips.
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_patches, seq_len, dim)
        temporal_pos = None
        if self._uses_temporal_rope:
            if frame_positions is None:
                temporal_pos = torch.arange(seq_len, device=x.device).view(1, seq_len)
                temporal_pos = temporal_pos.expand(batch_size * num_patches, seq_len)
            else:
                expected_fp = (batch_size, seq_len)
                if tuple(frame_positions.shape) != expected_fp:
                    raise ValueError(
                        f"Expected frame_positions shape {expected_fp}, got {tuple(frame_positions.shape)}"
                    )
                temporal_pos = (
                    frame_positions.view(batch_size, 1, seq_len)
                    .expand(batch_size, num_patches, seq_len)
                    .reshape(batch_size * num_patches, seq_len)
                    .contiguous()
                )
        x = self.global_block(x, pos=temporal_pos)
        x = x.reshape(batch_size, num_patches, seq_len, dim).permute(0, 2, 1, 3)
        return x


class LayerAttnStack(nn.Module):
    """Runs self-attention over the selected pyramid levels."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 1e-6,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, S, P, L, D], got shape={tuple(x.shape)}")

        batch_size, seq_len, num_patches, num_levels, dim = x.shape
        x = x.reshape(batch_size * seq_len * num_patches, num_levels, dim)
        for block in self.blocks:
            x = block(x)
        return x.reshape(batch_size, seq_len, num_patches, num_levels, dim)


class LearnedQueryPool(nn.Module):
    """Pools or unpools tokens using learned queries over the level axis."""

    def __init__(self, dim: int, n_query: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(n_query, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.query, std=0.02)
        _reset_linear(self.attn.out_proj)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.trunc_normal_(self.attn.in_proj_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            batch_size, seq_len, num_patches, num_levels, dim = x.shape
            kv = x.reshape(batch_size * seq_len * num_patches, num_levels, dim)
        elif x.ndim == 4:
            batch_size, seq_len, num_patches, dim = x.shape
            kv = x.reshape(batch_size * seq_len * num_patches, 1, dim)
        else:
            raise ValueError(f"Expected [B, S, P, D] or [B, S, P, L, D], got shape={tuple(x.shape)}")

        q = self.query.unsqueeze(0).expand(kv.shape[0], -1, -1)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.norm(out)
        out = out.reshape(batch_size, seq_len, num_patches, -1, dim)
        if out.shape[-2] == 1:
            return out.squeeze(-2)
        return out


class DetailConvBranch(nn.Module):
    """Adds a lightweight 2D detail branch on top of patch tokens."""

    def __init__(self, in_dim: int = 768, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, S, P, D], got shape={tuple(x.shape)}")

        batch_size, seq_len, num_patches, channels = x.shape
        patch_h, patch_w = _normalize_patch_grid(num_patches, patch_grid)
        x = x.reshape(batch_size * seq_len, patch_h, patch_w, channels).permute(0, 3, 1, 2)
        x = self.conv(x)
        return x.permute(0, 2, 3, 1).reshape(batch_size, seq_len, num_patches, -1)


class PerLayerDecoderHead(nn.Module):
    """Expands a hidden per-layer token stream back to the 3072-d joint token."""

    def __init__(
        self,
        hidden_dim: int,
        stream_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 1e-6,
        rope: RotaryPositionEmbedding2D | None = None,
    ) -> None:
        super().__init__()
        self.joint_dim = stream_dim * 3
        refine_heads = _resolve_joint_refine_heads(self.joint_dim, num_heads)
        if rope is not None:
            _validate_rope_head_dim(self.joint_dim, refine_heads, "PerLayerDecoderHead")
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.pre_proj = nn.Linear(hidden_dim, self.joint_dim)
        self.refine = Block(
            dim=self.joint_dim,
            num_heads=refine_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            rope=rope,
        )
        self.out_norm = nn.LayerNorm(self.joint_dim)
        self.apply(_reset_linear)

    def forward(self, x: torch.Tensor, patch_positions: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, S, P, D], got shape={tuple(x.shape)}")

        batch_size, seq_len, num_patches, _ = x.shape
        frame_positions = None
        if patch_positions is not None:
            expected_shape = (batch_size, seq_len, num_patches, 2)
            if tuple(patch_positions.shape) != expected_shape:
                raise ValueError(
                    f"Expected patch_positions shape {expected_shape}, got {tuple(patch_positions.shape)}"
                )
            frame_positions = patch_positions.reshape(batch_size * seq_len, num_patches, 2)
        x = self.pre_proj(self.pre_norm(x))
        x = self.refine(
            x.reshape(batch_size * seq_len, num_patches, self.joint_dim),
            pos=frame_positions,
        ).reshape(batch_size, seq_len, num_patches, self.joint_dim)
        return self.out_norm(x)


class JointSceneTokenizerEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        hidden_dim: int = 1152,
        num_layers: int = 4,
        num_block_pairs: int = 3,
        num_heads: int = 16,
        layer_attn_depth: int = 2,
        layer_attn_heads: int = 8,
        stream_dim: int = 1024,
        detail_dim: int = 128,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 1e-6,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if hidden_dim != latent_dim + detail_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal latent_dim + detail_dim ({latent_dim + detail_dim})"
            )

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.stream_dim = stream_dim
        # Split latent_dim into three non-uniform sub-streams (sum == latent_dim).
        sub_b = latent_dim // 3
        sub_a = latent_dim - 2 * sub_b
        self._sub_dims = (sub_a, sub_b, sub_b)
        self.position_getter = PositionGetter()
        self.patch_rope = RotaryPositionEmbedding2D()
        self.temporal_rope = RotaryPositionEmbedding1D()
        self._patch_pos_embed_cache: dict[tuple[int, int, int, str], torch.Tensor] = {}
        self.stream_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "dino": nn.LayerNorm(stream_dim),
                        "frame": nn.LayerNorm(stream_dim),
                        "global": nn.LayerNorm(stream_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.stream_proj = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "dino": nn.Linear(stream_dim, sub_a),
                        "frame": nn.Linear(stream_dim, sub_b),
                        "global": nn.Linear(stream_dim, sub_b),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.detail_branch = DetailConvBranch(in_dim=latent_dim, out_dim=detail_dim)
        self.layer_embed = nn.Parameter(torch.zeros(num_layers, hidden_dim))
        self.layer_attn = LayerAttnStack(
            hidden_dim,
            depth=layer_attn_depth,
            num_heads=layer_attn_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
        )
        self.layer_pool = LearnedQueryPool(hidden_dim, n_query=1, num_heads=layer_attn_heads)
        self.blocks = nn.ModuleList(
            [
                FrameGlobalBlockPair(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    rope=self.patch_rope,
                    temporal_rope=self.temporal_rope,
                )
                for _ in range(num_block_pairs)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.layer_embed, std=0.02)
        self.apply(_reset_linear)

    def forward(
        self,
        image_tokens_list_4: Sequence[torch.Tensor],
        patch_grid: tuple[int, int] | None = None,
        frame_positions_1d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(image_tokens_list_4) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} levels, got {len(image_tokens_list_4)}")

        reference_tokens = image_tokens_list_4[0]
        if reference_tokens.ndim != 4:
            raise ValueError(f"Expected [B, S, P, C] input, got shape={tuple(reference_tokens.shape)}")
        batch_size, seq_len, num_patches, _ = reference_tokens.shape
        patch_h, patch_w = _normalize_patch_grid(num_patches, patch_grid)
        patch_pos_embed = _get_cached_2d_sincos_pos_embed(
            self._patch_pos_embed_cache,
            self.hidden_dim,
            patch_h,
            patch_w,
            device=reference_tokens.device,
            dtype=reference_tokens.dtype,
        )
        patch_positions = self.position_getter(batch_size * seq_len, patch_h, patch_w, reference_tokens.device)
        patch_positions = patch_positions.view(batch_size, seq_len, num_patches, 2)

        if frame_positions_1d is not None:
            if frame_positions_1d.ndim != 2 or tuple(frame_positions_1d.shape) != (batch_size, seq_len):
                raise ValueError(
                    f"Expected frame_positions_1d shape ({batch_size}, {seq_len}), got "
                    f"{tuple(frame_positions_1d.shape)}"
                )
            frame_positions_1d = frame_positions_1d.to(device=reference_tokens.device, dtype=torch.long)

        per_layer_tokens = []
        for layer_idx, x_layer in enumerate(image_tokens_list_4):
            if x_layer.ndim != 4:
                raise ValueError(f"Expected [B, S, P, C] input, got shape={tuple(x_layer.shape)}")
            if x_layer.shape[:3] != (batch_size, seq_len, num_patches):
                raise ValueError(
                    "All levels must share the same [B, S, P] dimensions, got "
                    f"{tuple(x_layer.shape[:3])} vs {(batch_size, seq_len, num_patches)}"
                )
            if x_layer.shape[-1] != self.stream_dim * 3:
                raise ValueError(
                    f"Expected joint channel dim {self.stream_dim * 3}, got {x_layer.shape[-1]}"
                )

            dino, frame, global_tokens = x_layer.split([self.stream_dim, self.stream_dim, self.stream_dim], dim=-1)
            projected = torch.cat(
                [
                    self.stream_proj[layer_idx]["dino"](self.stream_norms[layer_idx]["dino"](dino)),
                    self.stream_proj[layer_idx]["frame"](self.stream_norms[layer_idx]["frame"](frame)),
                    self.stream_proj[layer_idx]["global"](self.stream_norms[layer_idx]["global"](global_tokens)),
                ],
                dim=-1,
            )
            detail = self.detail_branch(projected, patch_grid=patch_grid)
            per_layer_tokens.append(
                torch.cat([projected, detail], dim=-1) + self.layer_embed[layer_idx] + patch_pos_embed
            )

        x = torch.stack(per_layer_tokens, dim=-2)
        x = self.layer_attn(x)
        x = self.layer_pool(x)
        for block in self.blocks:
            x = block(x, patch_positions=patch_positions, frame_positions=frame_positions_1d)
        z = self.out_proj(self.out_norm(x))
        return z


class JointSceneTokenizerDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        hidden_dim: int = 1152,
        num_layers: int = 4,
        num_block_pairs: int = 3,
        num_heads: int = 16,
        layer_attn_depth: int = 2,
        layer_attn_heads: int = 8,
        stream_dim: int = 1024,
        detail_dim: int | None = None,  # accepted for API parity with the encoder
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 1e-6,
    ) -> None:
        super().__init__()
        del detail_dim  # decoder does not have a detail branch
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.position_getter = PositionGetter()
        self.patch_rope = RotaryPositionEmbedding2D()
        self.temporal_rope = RotaryPositionEmbedding1D()
        self._patch_pos_embed_cache: dict[tuple[int, int, int, str], torch.Tensor] = {}
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                FrameGlobalBlockPair(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    rope=self.patch_rope,
                    temporal_rope=self.temporal_rope,
                )
                for _ in range(num_block_pairs)
            ]
        )
        self.layer_unpool = LearnedQueryPool(hidden_dim, n_query=num_layers, num_heads=layer_attn_heads)
        self.layer_attn = LayerAttnStack(
            hidden_dim,
            depth=layer_attn_depth,
            num_heads=layer_attn_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            qk_norm=qk_norm,
            init_values=init_values,
        )
        self.layer_embed = nn.Parameter(torch.zeros(num_layers, hidden_dim))
        self.layer_heads = nn.ModuleList(
            [
                PerLayerDecoderHead(
                    hidden_dim=hidden_dim,
                    stream_dim=stream_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    rope=self.patch_rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.layer_embed, std=0.02)
        self.apply(_reset_linear)

    def forward(
        self,
        z: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
        frame_positions_1d: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if z.ndim != 4:
            raise ValueError(f"Expected latent shape [B, S, P, C], got {tuple(z.shape)}")

        batch_size, seq_len, num_patches, _ = z.shape
        patch_h, patch_w = _normalize_patch_grid(num_patches, patch_grid)
        patch_pos_embed = _get_cached_2d_sincos_pos_embed(
            self._patch_pos_embed_cache,
            self.hidden_dim,
            patch_h,
            patch_w,
            device=z.device,
            dtype=z.dtype,
        )
        patch_positions = self.position_getter(batch_size * seq_len, patch_h, patch_w, z.device)
        patch_positions = patch_positions.view(batch_size, seq_len, num_patches, 2)

        if frame_positions_1d is not None:
            if frame_positions_1d.ndim != 2 or tuple(frame_positions_1d.shape) != (batch_size, seq_len):
                raise ValueError(
                    f"Expected frame_positions_1d shape ({batch_size}, {seq_len}), got "
                    f"{tuple(frame_positions_1d.shape)}"
                )
            frame_positions_1d = frame_positions_1d.to(device=z.device, dtype=torch.long)

        x = self.in_norm(self.in_proj(z)) + patch_pos_embed
        for block in self.blocks:
            x = block(x, patch_positions=patch_positions, frame_positions=frame_positions_1d)

        x = self.layer_unpool(x)
        x = x + self.layer_embed.view(1, 1, 1, self.num_layers, -1)
        x = self.layer_attn(x)

        outputs = []
        for layer_idx in range(self.num_layers):
            outputs.append(self.layer_heads[layer_idx](x[..., layer_idx, :], patch_positions=patch_positions))
        return outputs


class JointSceneTokenizer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = JointSceneTokenizerEncoder(**kwargs)
        self.decoder = JointSceneTokenizerDecoder(**kwargs)

    def encode(
        self,
        image_tokens_4: Sequence[torch.Tensor],
        patch_grid: tuple[int, int] | None = None,
        frame_positions_1d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(image_tokens_4, patch_grid=patch_grid, frame_positions_1d=frame_positions_1d)

    def decode(
        self,
        z: torch.Tensor,
        patch_grid: tuple[int, int] | None = None,
        frame_positions_1d: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return self.decoder(z, patch_grid=patch_grid, frame_positions_1d=frame_positions_1d)

    def forward(
        self,
        image_tokens_4: Sequence[torch.Tensor],
        patch_grid: tuple[int, int] | None = None,
        frame_positions_1d: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return self.decode(
            self.encode(image_tokens_4, patch_grid=patch_grid, frame_positions_1d=frame_positions_1d),
            patch_grid=patch_grid,
            frame_positions_1d=frame_positions_1d,
        )
