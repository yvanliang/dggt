"""SceneFlow model for high-dimensional latent video generation.

The public class name remains ``WanSceneFlow`` for compatibility with the
existing training and inference scripts. The trunk follows the RAEv2 T2I
conditioning pattern: noisy video tokens and condition tokens are concatenated
and processed with full self-attention, then only the video span is decoded by
the RAEv2 DDT head for high-dimensional latent prediction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from dggt.utils.camera_condition import CAMERA_POSE_SUMMARY_DIM


def _normalize_patch_grid(
    num_patches: int,
    patch_grid: tuple[int, int] | list[int] | None,
) -> tuple[int, int]:
    if patch_grid is not None:
        h, w = int(patch_grid[0]), int(patch_grid[1])
        if h <= 0 or w <= 0 or h * w != int(num_patches):
            raise ValueError(f"patch_grid={patch_grid} is incompatible with P={num_patches}")
        return h, w

    side = int(round(num_patches**0.5))
    if side * side == int(num_patches):
        return side, side

    best_h, best_w = 1, int(num_patches)
    best_gap = best_w - best_h
    for h in range(1, int(num_patches**0.5) + 1):
        if num_patches % h == 0:
            w = num_patches // h
            gap = abs(w - h)
            if gap < best_gap:
                best_h, best_w, best_gap = h, w, gap
    return best_h, best_w


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


COSMOS_MROPE_SECTION = (24, 20, 20)
VIDEO_STATE_DIM = 6
ROPE_LAYOUT_VERSION = "a1_camera_center_sky128"
SKY_MROPE_TEMPORAL_OFFSET = 128
CAMERA_ROPE_SPATIAL_MODE = "center"
ASSET_POSITION_MODES = ("localized", "canonical")


def _scaled_mrope_section(
    head_dim: int,
    base: tuple[int, int, int] = COSMOS_MROPE_SECTION,
) -> tuple[int, int, int]:
    """Scale Cosmos mRoPE sections to the available rotary frequency slots."""
    head_dim = int(head_dim)
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be positive and even, got {head_dim}")
    half = head_dim // 2
    if half < 3:
        raise ValueError(f"RoPE head_dim={head_dim} leaves too few frequency slots for 3D mRoPE")
    base_sum = float(sum(int(v) for v in base))
    raw = [half * int(v) / base_sum for v in base]
    section = [max(1, int(math.floor(v))) for v in raw]
    while sum(section) > half:
        idx = max(range(3), key=lambda i: section[i])
        if section[idx] <= 1:
            break
        section[idx] -= 1
    remainder = half - sum(section)
    order = sorted(range(3), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in range(remainder):
        section[order[i % 3]] += 1
    return tuple(int(v) for v in section)


def _normalize_mrope_section(
    section: tuple[int, int, int] | list[int] | None,
    *,
    head_dim: int,
    name: str,
) -> tuple[int, int, int]:
    if section is None:
        section_t = _scaled_mrope_section(head_dim)
    else:
        if len(section) != 3:
            raise ValueError(f"{name} must have three entries, got {section}")
        section_t = tuple(int(v) for v in section)
    if any(v <= 0 for v in section_t):
        raise ValueError(f"{name} entries must be positive, got {section_t}")
    half = int(head_dim) // 2
    if sum(section_t) != half:
        raise ValueError(
            f"{name}={section_t} must sum to head_dim/2={half}; "
            "use None to scale the Cosmos (24,20,20) proportions automatically."
        )
    for axis, offset in (("H", 1), ("W", 2)):
        count = section_t[1 if axis == "H" else 2]
        max_index = offset + 3 * (count - 1)
        if max_index >= half:
            raise ValueError(
                f"{name}={section_t} cannot allocate {count} {axis} slots in head_dim={head_dim}; "
                f"last interleaved index would be {max_index}, but only 0..{half - 1} exist."
            )
    return section_t


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float = 0,
    use_float_positions: bool = False,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int | float]:
    dtype = torch.float32 if bool(use_float_positions) else torch.long
    ids = torch.arange(int(num_tokens), device=device, dtype=dtype)
    ids = ids + (float(temporal_offset) if bool(use_float_positions) else int(temporal_offset))
    return ids.unsqueeze(0).expand(3, -1).contiguous(), temporal_offset + int(num_tokens)


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 1,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
    temporal_positions: torch.Tensor | None = None,
    actual_temporal_compression_factor: int | None = None,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int | float]:
    grid_t = int(grid_t)
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    if grid_t < 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"Invalid mRoPE grid {(grid_t, grid_h, grid_w)}")
    if temporal_positions is not None:
        device = temporal_positions.device
    fps_modulation_enabled = fps is not None
    explicit_temporal_positions = temporal_positions is not None
    effective_base_tcf = (
        int(base_temporal_compression_factor)
        if base_temporal_compression_factor is not None
        else int(temporal_compression_factor)
    )
    effective_actual_tcf = (
        int(actual_temporal_compression_factor)
        if actual_temporal_compression_factor is not None
        else int(temporal_compression_factor)
    )

    if explicit_temporal_positions:
        assert temporal_positions is not None
        if temporal_positions.ndim != 1 or int(temporal_positions.shape[0]) != grid_t:
            raise ValueError(
                f"temporal_positions must be shape ({grid_t},), got {tuple(temporal_positions.shape)}"
            )
        frame_indices = temporal_positions.to(device=device, dtype=torch.float32)
        if int(start_frame_offset) != 0:
            frame_indices = frame_indices + float(start_frame_offset) / float(effective_actual_tcf)
        if fps_modulation_enabled:
            scaled_t = (
                frame_indices
                * float(effective_actual_tcf)
                * (float(base_fps) / float(effective_base_tcf))
                / float(fps)
                + float(temporal_offset)
            )
        else:
            scaled_t = frame_indices + float(temporal_offset)
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    elif fps_modulation_enabled:
        tps = float(fps) / float(temporal_compression_factor)
        base_tps = float(base_fps) / float(effective_base_tcf)
        frame_indices = torch.arange(grid_t, device=device, dtype=torch.float32)
        scaled_t = (frame_indices + float(start_frame_offset)) / tps * base_tps + float(temporal_offset)
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, device=device, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + int(start_frame_offset)
        )

    h_index = (
        torch.arange(grid_h, device=t_index.device, dtype=torch.long)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, device=t_index.device, dtype=torch.long)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .flatten()
    )
    if not bool(reset_spatial_indices):
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset
    if fps_modulation_enabled or explicit_temporal_positions:
        mrope_ids = torch.stack([t_index, h_index.float(), w_index.float()], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    next_temporal_offset = math.ceil(float(mrope_ids.max().item())) + 1 if mrope_ids.numel() else temporal_offset
    return mrope_ids, next_temporal_offset


class Config(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "asset_position_mode"
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight.to(device=x.device, dtype=x.dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GaussianFourierEmbedding(nn.Module):
    """RAEv2 timestep embedder.

    Returns both a base timestep embedding for the DDT head and learnable
    timestep tokens that are concatenated into the full-attention sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        n_tokens: int = 4,
        embedding_size: int = 256,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, float(scale), (int(embedding_size),)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(int(embedding_size) * 2, int(hidden_size), bias=True),
            nn.SiLU(),
            nn.Linear(int(hidden_size), int(hidden_size), bias=True),
        )
        self.learnable_tokens = nn.Parameter(
            torch.normal(0, 1 / int(hidden_size) ** 0.5, (int(n_tokens), int(hidden_size)))
        )

    def forward(self, t: torch.Tensor, return_base_embed: bool = False):
        t = t[:, None] * self.W[None, :].to(device=t.device, dtype=t.dtype) * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed.to(dtype=self.mlp[0].weight.dtype))
        if return_base_embed:
            t_embed = t_embed.unsqueeze(1)
            return t_embed, self.learnable_tokens.to(device=t.device, dtype=t_embed.dtype) + t_embed
        return self.learnable_tokens.to(device=t.device, dtype=t_embed.dtype) + t_embed.unsqueeze(1)


class VideoRoPE3D:
    """3D RoPE for video/text/condition tokens.

    Older callers pass ``video_tokens`` and get frame/y/x positions for the
    leading video segment with zero-angle condition tokens. Cosmos-lite callers
    pass explicit ``position_ids`` shaped ``[B,N,3]``, ``[N,3]``,
    ``[3,N]`` or ``[3,B,N]``.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        patch_grid: tuple[int, int],
        video_tokens: int,
        total_tokens: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        theta: float = 10000.0,
        position_ids: torch.Tensor | None = None,
        mrope_section: tuple[int, int, int] | list[int] | None = None,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        mrope_section_t = _normalize_mrope_section(
            mrope_section,
            head_dim=head_dim,
            name="mrope_section",
        )
        h, w = patch_grid
        if position_ids is None:
            angles = torch.zeros(total_tokens, head_dim, device=device, dtype=torch.float32)
            if video_tokens > 0:
                coords = torch.arange(video_tokens, device=device)
                f = torch.div(coords, h * w, rounding_mode="floor")
                rem = coords % (h * w)
                y = torch.div(rem, w, rounding_mode="floor")
                x = rem % w
                self._fill_angles(
                    angles[:video_tokens],
                    torch.stack([f, y, x], dim=-1),
                    head_dim,
                    theta,
                    mrope_section=mrope_section_t,
                )
            self.cos = angles.cos().to(dtype=dtype).view(1, 1, total_tokens, head_dim)
            self.sin = angles.sin().to(dtype=dtype).view(1, 1, total_tokens, head_dim)
        else:
            pos = self._normalize_position_ids(position_ids, device=device)
            if int(pos.shape[1]) != int(total_tokens):
                raise ValueError(f"position_ids length {pos.shape[1]} != total_tokens {total_tokens}")
            angles = torch.zeros(pos.shape[0], total_tokens, head_dim, device=device, dtype=torch.float32)
            self._fill_angles(angles, pos, head_dim, theta, mrope_section=mrope_section_t)
            self.cos = angles.cos().to(dtype=dtype).unsqueeze(1)
            self.sin = angles.sin().to(dtype=dtype).unsqueeze(1)
        self.seq_len = int(seq_len)

    @staticmethod
    def _normalize_position_ids(position_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        pos = position_ids.to(device=device)
        if pos.ndim == 2:
            if int(pos.shape[-1]) == 3:
                pos = pos.unsqueeze(0)
            elif int(pos.shape[0]) == 3:
                pos = pos.transpose(0, 1).unsqueeze(0)
            else:
                raise ValueError(f"position_ids must be [N,3] or [3,N], got {tuple(position_ids.shape)}")
        elif pos.ndim == 3:
            if int(pos.shape[-1]) == 3:
                pass
            elif int(pos.shape[0]) == 3:
                pos = pos.permute(1, 2, 0).contiguous()
            else:
                raise ValueError(f"position_ids must be [B,N,3] or [3,B,N], got {tuple(position_ids.shape)}")
        else:
            raise ValueError(f"position_ids must be rank 2 or 3, got {tuple(position_ids.shape)}")
        return pos.to(dtype=torch.float32)

    @staticmethod
    def _fill_angles(
        angles: torch.Tensor,
        pos: torch.Tensor,
        head_dim: int,
        theta: float,
        *,
        mrope_section: tuple[int, int, int] | list[int],
    ) -> None:
        half = head_dim // 2
        inv_freq = 1.0 / (
            float(theta) ** (torch.arange(0, head_dim, 2, device=pos.device).float() / max(head_dim, 1))
        )
        freqs = torch.stack(
            [pos[..., axis].float().unsqueeze(-1) * inv_freq for axis in range(3)],
            dim=0,
        )
        mixed = VideoRoPE3D._apply_interleaved_mrope(freqs, mrope_section)
        angles.copy_(torch.cat([mixed, mixed], dim=-1))

    @staticmethod
    def _apply_interleaved_mrope(
        freqs: torch.Tensor,
        mrope_section: tuple[int, int, int] | list[int],
    ) -> torch.Tensor:
        mixed = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = int(mrope_section[dim]) * 3
            mixed[..., offset:length:3] = freqs[dim, ..., offset:length:3]
        return mixed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cos.to(device=x.device, dtype=x.dtype) + _rotate_half(x) * self.sin.to(device=x.device, dtype=x.dtype)


class NormAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.dim = int(dim)
        self.head_dim = int(dim) // int(num_heads)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope: Any, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        return self.proj(out)


class DDTEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        ffn_dim = int(ffn_dim if ffn_dim is not None else hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2.0 / 3.0 * ffn_dim))

    def forward(self, x: torch.Tensor, rope: Any, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        ffn_dim: int | None = None,
    ) -> None:
        super().__init__(hidden_size, num_heads, mlp_ratio, ffn_dim=ffn_dim)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope: Any,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa), rope=rope, attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim < x.ndim:
            c = c.unsqueeze(1)
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


def _build_2d_sincos(num_patches: int, hidden_size: int, patch_grid: tuple[int, int], device, dtype) -> torch.Tensor:
    h, w = patch_grid
    if h * w != num_patches:
        raise ValueError(f"patch_grid={patch_grid} incompatible with num_patches={num_patches}")
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    coords = torch.stack([y.reshape(-1), x.reshape(-1)], dim=1).float()
    half = hidden_size // 2
    half -= half % 2
    if half <= 0:
        return torch.zeros(num_patches, hidden_size, device=device, dtype=dtype)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, 2, device=device).float() / half))
    y_emb = coords[:, 0:1] * freqs.unsqueeze(0)
    x_emb = coords[:, 1:2] * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(y_emb), torch.cos(y_emb), torch.sin(x_emb), torch.cos(x_emb)], dim=-1)
    if emb.shape[-1] < hidden_size:
        emb = F.pad(emb, (0, hidden_size - emb.shape[-1]))
    return emb[:, :hidden_size].to(dtype=dtype)


class AssetFrameCompressor(nn.Module):
    """Compress each asset video latent `[S, P, C]` into one hidden token."""

    def __init__(self, in_dim: int, hidden_size: int, max_assets: int = 5, max_frames: int = 16) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_size = int(hidden_size)
        self.max_assets = int(max_assets)
        self.max_frames = int(max_frames)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden_size)
        self.pool_gate = nn.Linear(hidden_size, 1)
        self.asset_type_embed = nn.Parameter(torch.randn(1, hidden_size) / math.sqrt(float(hidden_size)))
        self.asset_slot_embed = nn.Embedding(max_assets, hidden_size)
        self.frame_embed = nn.Embedding(max_frames, hidden_size)

    def forward(
        self,
        asset_latents: torch.Tensor,
        *,
        patch_grid: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if asset_latents.ndim != 5:
            raise ValueError(f"asset_latents must be [B,K,S,P,D], got {tuple(asset_latents.shape)}")
        b, k_raw, s, p, d = asset_latents.shape
        if d != self.in_dim:
            raise ValueError(f"asset latent dim {d} != compressor dim {self.in_dim}")
        k = min(k_raw, self.max_assets)
        x = asset_latents[:, :k].contiguous()
        if valid_mask is not None:
            if valid_mask.shape != (b, k_raw, s, p):
                raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} != {(b, k_raw, s, p)}")
            valid_patch = valid_mask[:, :k].to(device=x.device, dtype=torch.bool)
            valid_asset = valid_patch.any(dim=(2, 3))
        else:
            valid_patch = torch.ones((b, k, s, p), device=x.device, dtype=torch.bool)
            valid_asset = torch.ones((b, k), device=x.device, dtype=torch.bool)

        h = self.proj(self.norm(x.float()).to(dtype=x.dtype))
        h = h + _build_2d_sincos(p, self.hidden_size, patch_grid, h.device, h.dtype).view(1, 1, 1, p, -1)

        slot_ids = torch.arange(k, device=x.device)
        frame_ids = torch.arange(s, device=x.device).clamp_max(self.max_frames - 1)
        h = (
            h
            + self.asset_type_embed.to(device=x.device, dtype=h.dtype).view(1, 1, 1, 1, -1)
            + self.asset_slot_embed(slot_ids).to(dtype=h.dtype).view(1, k, 1, 1, -1)
            + self.frame_embed(frame_ids).to(dtype=h.dtype).view(1, 1, s, 1, -1)
        )
        gate = torch.sigmoid(self.pool_gate(h)).squeeze(-1) * valid_patch.to(dtype=h.dtype)
        pooled = (h * gate.unsqueeze(-1)).sum(dim=(2, 3)) / gate.sum(dim=(2, 3)).unsqueeze(-1).clamp_min(1e-6)
        pooled = torch.where(valid_asset.unsqueeze(-1), pooled, torch.zeros_like(pooled))

        if k < self.max_assets:
            pad = pooled.new_zeros((b, self.max_assets - k, self.hidden_size))
            mask_pad = torch.zeros((b, self.max_assets - k), device=x.device, dtype=torch.bool)
            pooled = torch.cat([pooled, pad], dim=1)
            valid_asset = torch.cat([valid_asset, mask_pad], dim=1)
        return pooled, valid_asset


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, int(num_channels))
    while int(num_channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(num_channels))


class DepthwiseSeparableResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.norm1 = _group_norm(in_channels)
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.norm2 = _group_norm(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.dw(F.silu(self.norm1(x)))
        x = self.pw(F.silu(self.norm2(x)))
        return x + residual


class SkyMaskRefineDecoder(nn.Module):
    """Lightweight dense decoder for refined image-plane sky masks."""

    def __init__(
        self,
        hidden_size: int,
        *,
        channels: int = 256,
        refine_scale: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        channels = int(channels)
        refine_scale = int(refine_scale)
        if channels <= 0:
            raise ValueError(f"sky_mask_refine_channels must be positive, got {channels}")
        if refine_scale <= 0 or refine_scale & (refine_scale - 1):
            raise ValueError(f"sky_mask_refine_scale must be a positive power of two, got {refine_scale}")
        self.refine_scale = refine_scale
        self.token_norm = RMSNorm(hidden_size, eps=float(eps))
        self.token_proj = nn.Linear(hidden_size, channels)
        self.skip_norm = RMSNorm(hidden_size, eps=float(eps))
        self.skip_proj = nn.Linear(hidden_size, channels)
        self.coord_proj = nn.Conv2d(channels + 2, channels, kernel_size=1)
        stages: list[nn.Module] = [DepthwiseSeparableResBlock(channels, channels)]
        current = channels
        min_channels = min(64, channels)
        upsample_steps = int(math.log2(refine_scale)) if refine_scale > 1 else 0
        for step in range(upsample_steps):
            next_channels = max(min_channels, current // 2) if step > 0 else current
            stages.append(DepthwiseSeparableResBlock(current, next_channels))
            current = next_channels
        self.stages = nn.ModuleList(stages)
        self.final_norm = _group_norm(current)
        self.final = nn.Conv2d(current, 1, kernel_size=1)

    @staticmethod
    def _coord_grid(
        *,
        height: int,
        width: int,
        batch_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([yy, xx], dim=0).unsqueeze(0)
        return coords.expand(int(batch_frames), -1, -1, -1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        patch_grid: tuple[int, int],
        seq_len: int,
        skip_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = tokens.shape
        gh, gw = int(patch_grid[0]), int(patch_grid[1])
        if n != int(seq_len) * gh * gw:
            raise ValueError(f"sky-mask tokens length {n} != S*H*W={int(seq_len) * gh * gw}")
        x = self.token_proj(self.token_norm(tokens))
        if skip_tokens is not None:
            if skip_tokens.shape != tokens.shape:
                raise ValueError(f"skip_tokens shape {tuple(skip_tokens.shape)} != tokens {tuple(tokens.shape)}")
            x = x + self.skip_proj(self.skip_norm(skip_tokens)).to(dtype=x.dtype)
        x = x.reshape(b * int(seq_len), gh, gw, -1).permute(0, 3, 1, 2).contiguous()
        coords = self._coord_grid(
            height=gh,
            width=gw,
            batch_frames=b * int(seq_len),
            device=x.device,
            dtype=x.dtype,
        )
        x = self.coord_proj(torch.cat([x, coords], dim=1))
        x = self.stages[0](x)
        for block in self.stages[1:]:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        x = self.final(F.silu(self.final_norm(x)))
        return x.reshape(b, int(seq_len), 1, gh * self.refine_scale, gw * self.refine_scale)


class RAEVideoSceneFlow(nn.Module):
    """RAEv2-style full-attention trunk with RAE timestep tokens and 3D mRoPE."""

    _no_split_modules = ["DDTEncoderBlock", "DDTDecoderBlock"]
    _repeated_blocks = ["DDTEncoderBlock", "DDTDecoderBlock"]

    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 1, 1),
        patch_grid: tuple[int, int] = (25, 37),
        num_attention_heads: int = 20,
        attention_head_dim: int = 72,
        in_channels: int = 3075,
        out_channels: int = 1024,
        text_dim: int = 1024,
        qwen_dim: int = 1024,
        freq_dim: int = 256,
        ffn_dim: int | None = None,
        num_layers: int = 28,
        eps: float = 1e-6,
        rope_max_seq_len: int = 128,
        repa_block_frac: float = 1.0 / 3.0,
        repa_layer_depth: int | None = None,
        null_kv_std: float = 0.02,
        ddt_head_depth: int = 2,
        ddt_head_dim: int = 2048,
        ddt_head_heads: int = 16,
        ddt_head_ffn_dim: int | None = None,
        max_assets: int = 5,
        max_frames: int = 16,
        num_timestep_tokens: int = 4,
        base_model_depth: int = 8,
        prediction_type: str = "x",
        architecture: str = "cosmos_lite",
        max_asset_patch_tokens_per_asset_frame: int = 32,
        max_asset_tokens: int = 4096,
        max_control_tokens_per_frame: int = 128,
        max_control_tokens: int = 1024,
        camera_cond_dim: int = CAMERA_POSE_SUMMARY_DIM,
        camera_gen_dim: int = 2048,
        sky_token_dim: int = 3,
        sky_grid: tuple[int, int] | list[int] | None = (16, 32),
        max_sky_tokens: int = 512,
        video_state_dim: int = VIDEO_STATE_DIM,
        sky_mask_head_version: str = "patch_mlp_refine_v1",
        sky_mask_refine_scale: int = 4,
        sky_mask_refine_channels: int = 256,
        rope_theta: float = 5000000.0,
        encoder_mrope_section: tuple[int, int, int] | list[int] | None = None,
        ddt_mrope_section: tuple[int, int, int] | list[int] | None = None,
        asset_position_mode: str = "localized",
        timestep_scale: float = 1.0,
        t_eps: float = 0.05,
        **unused: Any,
    ) -> None:
        super().__init__()
        if tuple(patch_size) != (1, 1, 1):
            raise ValueError("RAEVideoSceneFlow expects patch_size=(1, 1, 1) for DGGT token grids.")
        num_layers = int(num_layers)
        out_channels = int(out_channels)
        base_model_depth = int(base_model_depth)
        hidden_size = int(num_attention_heads) * int(attention_head_dim)
        if hidden_size % int(num_attention_heads) != 0:
            raise ValueError("hidden size must be divisible by encoder heads")
        if int(ddt_head_dim) % int(ddt_head_heads) != 0:
            raise ValueError("ddt_head_dim must be divisible by ddt_head_heads")
        encoder_mrope_section_i = _normalize_mrope_section(
            encoder_mrope_section,
            head_dim=int(attention_head_dim),
            name="encoder_mrope_section",
        )
        ddt_mrope_section_i = _normalize_mrope_section(
            ddt_mrope_section,
            head_dim=int(ddt_head_dim) // int(ddt_head_heads),
            name="ddt_mrope_section",
        )
        if prediction_type not in ("x", "v"):
            raise ValueError("prediction_type must be 'x' or 'v'")
        asset_position_mode_i = str(asset_position_mode).lower()
        if asset_position_mode_i not in ASSET_POSITION_MODES:
            raise ValueError(
                f"asset_position_mode must be one of {ASSET_POSITION_MODES}, got {asset_position_mode!r}"
            )
        sky_grid_t = None
        if sky_grid is not None:
            if len(sky_grid) != 2:
                raise ValueError(f"sky_grid must be (H,W) or None, got {sky_grid}")
            sky_grid_t = (int(sky_grid[0]), int(sky_grid[1]))
            if sky_grid_t[0] <= 0 or sky_grid_t[1] <= 0:
                raise ValueError(f"sky_grid entries must be positive, got {sky_grid_t}")
        video_state_dim = int(video_state_dim)
        if video_state_dim != VIDEO_STATE_DIM:
            raise ValueError(f"video_state_dim must be {VIDEO_STATE_DIM}, got {video_state_dim}")
        sky_mask_refine_scale = int(sky_mask_refine_scale)
        sky_mask_refine_channels = int(sky_mask_refine_channels)
        if sky_mask_refine_scale <= 0 or sky_mask_refine_scale & (sky_mask_refine_scale - 1):
            raise ValueError(
                f"sky_mask_refine_scale must be a positive power of two, got {sky_mask_refine_scale}"
            )
        if sky_mask_refine_channels <= 0:
            raise ValueError(f"sky_mask_refine_channels must be positive, got {sky_mask_refine_channels}")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 1 <= base_model_depth <= num_layers:
            raise ValueError(f"base_model_depth={base_model_depth} must be in [1, num_layers={num_layers}]")
        if repa_layer_depth is None:
            repa_layer_depth_i = int(num_layers * float(repa_block_frac))
            repa_layer_depth_i = min(max(1, repa_layer_depth_i), num_layers)
        else:
            repa_layer_depth_i = int(repa_layer_depth)
            if not 1 <= repa_layer_depth_i <= num_layers:
                raise ValueError(f"repa_layer_depth={repa_layer_depth_i} must be in [1, num_layers={num_layers}]")
        ffn_dim_i = int(ffn_dim if ffn_dim is not None else hidden_size * 4)
        ddt_head_ffn_dim_i = int(ddt_head_ffn_dim if ddt_head_ffn_dim is not None else int(ddt_head_dim) * 4)
        # Retained as an accepted legacy Cosmos-style config field. RAEv2 uses
        # the raw continuous flow time sigma in [0, 1] without timestep scaling.
        del timestep_scale
        if "mrope_temporal_margin" in unused:
            raise TypeError(
                "mrope_temporal_margin has been removed. SceneFlow now uses fixed A1 RoPE positions: "
                "video/asset/control temporal offset 0, camera on the video frame center, and sky temporal offset 128."
            )

        self.config = Config(
            patch_size=tuple(patch_size),
            patch_grid=tuple(int(v) for v in patch_grid),
            num_attention_heads=int(num_attention_heads),
            attention_head_dim=int(attention_head_dim),
            hidden_size=hidden_size,
            in_channels=int(in_channels),
            out_channels=out_channels,
            text_dim=int(text_dim),
            qwen_dim=int(qwen_dim),
            freq_dim=int(freq_dim),
            ffn_dim=ffn_dim_i,
            num_layers=num_layers,
            eps=float(eps),
            rope_max_seq_len=int(rope_max_seq_len),
            repa_block_frac=float(repa_block_frac),
            repa_layer_depth=repa_layer_depth_i,
            null_kv_std=float(null_kv_std),
            ddt_head_depth=int(ddt_head_depth),
            ddt_head_dim=int(ddt_head_dim),
            ddt_head_heads=int(ddt_head_heads),
            ddt_head_ffn_dim=ddt_head_ffn_dim_i,
            max_assets=int(max_assets),
            max_frames=int(max_frames),
            num_timestep_tokens=int(num_timestep_tokens),
            asset_dim=out_channels,
            base_model_depth=base_model_depth,
            prediction_type=str(prediction_type),
            architecture=str(architecture),
            max_asset_patch_tokens_per_asset_frame=int(max_asset_patch_tokens_per_asset_frame),
            max_asset_tokens=int(max_asset_tokens),
            max_control_tokens_per_frame=int(max_control_tokens_per_frame),
            max_control_tokens=int(max_control_tokens),
            camera_cond_dim=int(camera_cond_dim),
            camera_gen_dim=int(camera_gen_dim),
            sky_token_dim=int(sky_token_dim),
            sky_grid=sky_grid_t,
            max_sky_tokens=int(max_sky_tokens),
            video_state_dim=video_state_dim,
            sky_mask_head_version=str(sky_mask_head_version),
            sky_mask_refine_scale=sky_mask_refine_scale,
            sky_mask_refine_channels=sky_mask_refine_channels,
            rope_layout_version=ROPE_LAYOUT_VERSION,
            sky_rope_temporal_offset=SKY_MROPE_TEMPORAL_OFFSET,
            camera_rope_spatial_mode=CAMERA_ROPE_SPATIAL_MODE,
            rope_theta=float(rope_theta),
            encoder_mrope_section=encoder_mrope_section_i,
            ddt_mrope_section=ddt_mrope_section_i,
            asset_position_mode=asset_position_mode_i,
            t_eps=float(t_eps),
        )
        self.gradient_checkpointing = False

        self.video_embed = nn.Linear(out_channels, hidden_size)
        self.decoder_video_embed = nn.Linear(out_channels, int(ddt_head_dim))
        self.video_state_proj = nn.Sequential(
            nn.Linear(video_state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.decoder_state_proj = nn.Sequential(
            nn.Linear(video_state_dim, int(ddt_head_dim)),
            nn.SiLU(),
            nn.Linear(int(ddt_head_dim), int(ddt_head_dim)),
        )
        self.t_embedder = GaussianFourierEmbedding(
            hidden_size,
            n_tokens=int(num_timestep_tokens),
            embedding_size=int(freq_dim),
        )
        self.text_norm = RMSNorm(int(qwen_dim), eps=float(eps))
        self.text_proj = nn.Linear(int(qwen_dim), hidden_size)
        self.asset_latent_norm = RMSNorm(int(out_channels), eps=float(eps))
        self.asset_latent_proj = nn.Linear(int(out_channels), hidden_size)
        self.asset_slot_embed = nn.Embedding(max(1, int(max_assets)), hidden_size)
        self.asset_frame_embed = nn.Embedding(max(1, int(max_frames)), hidden_size)
        self.control_norm = RMSNorm(int(out_channels) * 2 + 3, eps=float(eps))
        self.control_proj = nn.Linear(int(out_channels) * 2 + 3, hidden_size)
        self.camera_norm = RMSNorm(int(camera_cond_dim), eps=float(eps))
        self.camera_proj = nn.Sequential(
            nn.Linear(int(camera_cond_dim), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.camera_gen_norm = RMSNorm(int(camera_gen_dim), eps=float(eps))
        self.camera_gen_proj = nn.Linear(int(camera_gen_dim), hidden_size)
        self.camera_gen_decoder = nn.Sequential(
            RMSNorm(hidden_size, eps=float(eps)),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, int(camera_gen_dim)),
        )
        self.sky_gen_norm = RMSNorm(int(sky_token_dim), eps=float(eps))
        self.sky_gen_proj = nn.Linear(int(sky_token_dim), hidden_size)
        self.sky_gen_decoder = nn.Sequential(
            RMSNorm(hidden_size, eps=float(eps)),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, int(sky_token_dim)),
        )
        self.sky_mask_decoder = nn.Sequential(
            RMSNorm(hidden_size, eps=float(eps)),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.sky_mask_refine_decoder = SkyMaskRefineDecoder(
            hidden_size,
            channels=sky_mask_refine_channels,
            refine_scale=sky_mask_refine_scale,
            eps=float(eps),
        )
        self.text_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.camera_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.camera_gen_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.sky_gen_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.asset_patch_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.asset_summary_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.edit_control_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.video_target_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.asset_direct_proj = nn.Linear(int(text_dim), hidden_size)
        self.asset_encoded_proj = nn.Linear(int(out_channels), hidden_size)
        self.asset_compressor: AssetFrameCompressor | None = None
        self.empty_asset_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.asset_null_condition_embed = nn.Parameter(
            torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size))
        )
        self.camera_null_condition_embed = nn.Parameter(
            torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size))
        )

        self.blocks = nn.ModuleList([
            DDTEncoderBlock(hidden_size, int(num_attention_heads), mlp_ratio=4.0, ffn_dim=ffn_dim_i)
            for _ in range(num_layers)
        ])
        self.s_projector = (
            nn.Linear(hidden_size, int(ddt_head_dim)) if hidden_size != int(ddt_head_dim) else nn.Identity()
        )
        self.ddt_head = nn.ModuleList([
            DDTDecoderBlock(
                int(ddt_head_dim),
                int(ddt_head_heads),
                mlp_ratio=4.0,
                ffn_dim=ddt_head_ffn_dim_i,
            )
            for _ in range(int(ddt_head_depth))
        ])
        self.final_layer = DDTFinalLayer(int(ddt_head_dim), int(out_channels))
        self.base_final_layer = DDTFinalLayer(hidden_size, int(out_channels))
        self.proj_out = self.final_layer.linear
        self.null_kv = nn.Parameter(torch.randn(1, 1, int(text_dim)) * float(null_kv_std))

        self.register_buffer("mu_z", torch.zeros(int(out_channels)))
        self.register_buffer("sigma_z", torch.ones(int(out_channels)))
        self.repa_layer_depth = repa_layer_depth_i
        self.repa_block_idx = repa_layer_depth_i - 1
        self.repa_proj = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size, int(out_channels))),
            nn.SiLU(),
            nn.Linear(max(hidden_size, int(out_channels)), int(out_channels)),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.video_embed.weight)
        nn.init.zeros_(self.video_embed.bias)
        nn.init.xavier_uniform_(self.decoder_video_embed.weight)
        nn.init.zeros_(self.decoder_video_embed.bias)
        for proj in (self.video_state_proj, self.decoder_state_proj):
            for module in proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            final = proj[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        for module in (
            self.text_proj,
            self.asset_latent_proj,
            self.control_proj,
            self.asset_direct_proj,
            self.asset_encoded_proj,
            self.camera_gen_proj,
            self.sky_gen_proj,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in self.camera_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.camera_gen_decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.sky_gen_decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.sky_mask_decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.sky_mask_refine_decoder.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        camera_final = self.camera_gen_decoder[-1]
        if isinstance(camera_final, nn.Linear):
            nn.init.zeros_(camera_final.weight)
            nn.init.zeros_(camera_final.bias)
        sky_final = self.sky_gen_decoder[-1]
        if isinstance(sky_final, nn.Linear):
            nn.init.zeros_(sky_final.weight)
            nn.init.zeros_(sky_final.bias)
        sky_mask_final = self.sky_mask_decoder[-1]
        if isinstance(sky_mask_final, nn.Linear):
            nn.init.zeros_(sky_mask_final.weight)
            nn.init.zeros_(sky_mask_final.bias)
        nn.init.zeros_(self.sky_mask_refine_decoder.final.weight)
        nn.init.zeros_(self.sky_mask_refine_decoder.final.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.ddt_head:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.base_final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.base_final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.base_final_layer.linear.weight)
        nn.init.zeros_(self.base_final_layer.linear.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Older worktree checkpoints used Cosmos deterministic timestep
        # embeddings. The current model uses RAEv2 Gaussian Fourier timestep
        # tokens, so those stale keys are intentionally ignored.
        for stale_key in [
            prefix + "timestep_modality_embed",
        ]:
            state_dict.pop(stale_key, None)
        for stale_key in [key for key in state_dict.keys() if key.startswith(prefix + "t_embedder.linear_")]:
            state_dict.pop(stale_key)
        stale_prefix = prefix + "asset_compressor."
        for stale_key in [key for key in state_dict.keys() if key.startswith(stale_prefix)]:
            state_dict.pop(stale_key)
        # Newly introduced optional-condition sentinels should not make older
        # SceneFlow checkpoints unloadable. Missing keys are initialized from
        # the current module parameters and will continue training normally.
        for param_name in ("asset_null_condition_embed", "camera_null_condition_embed"):
            key = prefix + param_name
            if key not in state_dict:
                state_dict[key] = getattr(self, param_name).detach().clone()
        for name, value in self.state_dict().items():
            if name.startswith(("sky_mask_decoder.", "sky_mask_refine_decoder.")):
                key = prefix + name
                if key not in state_dict:
                    state_dict[key] = value.detach().clone()
        # Earlier checkpoints projected the packed vector
        # [z_t, z_splat, scaffold, masks]. The RAEv2-style DDT path embeds only
        # z_t, which is the first out_channels slice in that packed layout.
        for module_name in ("video_embed", "decoder_video_embed"):
            weight_key = prefix + module_name + ".weight"
            weight = state_dict.get(weight_key)
            expected = getattr(self, module_name).weight
            if (
                torch.is_tensor(weight)
                and weight.ndim == 2
                and tuple(weight.shape) != tuple(expected.shape)
                and int(weight.shape[0]) == int(expected.shape[0])
                and int(weight.shape[1]) >= int(expected.shape[1])
            ):
                state_dict[weight_key] = weight[:, : expected.shape[1]].contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_scene_config(
        cls,
        *,
        bring_up: bool = False,
        patch_grid: tuple[int, int] = (25, 37),
        **kwargs: Any,
    ) -> "RAEVideoSceneFlow":
        if "mrope_temporal_margin" in kwargs:
            raise TypeError(
                "mrope_temporal_margin has been removed. Use the fixed A1 RoPE layout "
                "(video/asset/camera shared grid, sky offset 128)."
            )
        if bring_up:
            defaults = {
                "num_attention_heads": 8,
                "attention_head_dim": 96,
                "encoder_mrope_section": (18, 15, 15),
                "ddt_mrope_section": COSMOS_MROPE_SECTION,
                "num_layers": 8,
                "ddt_head_dim": 1024,
                "ddt_head_heads": 8,
                "ddt_head_depth": 1,
                "patch_grid": patch_grid,
            }
        else:
            defaults = {
                "num_attention_heads": 20,
                "attention_head_dim": 72,
                "encoder_mrope_section": (12, 12, 12),
                "ddt_mrope_section": COSMOS_MROPE_SECTION,
                "num_layers": 28,
                "ddt_head_dim": 2048,
                "ddt_head_heads": 16,
                "ddt_head_depth": 2,
                "patch_grid": patch_grid,
            }
        defaults.update(kwargs)
        return cls(**defaults)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def set_latent_stats(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        if tuple(mu.shape) != tuple(self.mu_z.shape):
            raise ValueError(f"mu shape {tuple(mu.shape)} != {tuple(self.mu_z.shape)}")
        if tuple(sigma.shape) != tuple(self.sigma_z.shape):
            raise ValueError(f"sigma shape {tuple(sigma.shape)} != {tuple(self.sigma_z.shape)}")
        self.mu_z.copy_(mu.to(device=self.mu_z.device, dtype=self.mu_z.dtype))
        self.sigma_z.copy_(sigma.to(device=self.sigma_z.device, dtype=self.sigma_z.dtype).clamp_min(1e-6))

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu_z.to(device=z.device, dtype=z.dtype)) / self.sigma_z.to(
            device=z.device, dtype=z.dtype
        ).clamp_min(1e-6)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.sigma_z.to(device=z.device, dtype=z.dtype) + self.mu_z.to(device=z.device, dtype=z.dtype)

    @staticmethod
    def _key_padding_attention_mask(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
        if valid.ndim != 2:
            raise ValueError(f"valid mask must be [B,N], got {tuple(valid.shape)}")
        valid = valid.to(dtype=torch.bool)
        if valid.shape[1] == 0 or bool(valid.all().item()):
            return None
        return (~valid[:, None, None, :]).to(dtype=dtype) * torch.finfo(dtype).min

    @staticmethod
    def _apply_token_valid_mask(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if valid.ndim != 2 or valid.shape != x.shape[:2]:
            raise ValueError(f"valid mask shape {tuple(valid.shape)} != token shape {tuple(x.shape[:2])}")
        return x * valid.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

    def _validate_video_control_inputs(
        self,
        z_t: torch.Tensor,
        z_splat: torch.Tensor,
        scaffold_tok: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
    ) -> None:
        if z_t.ndim != 4:
            raise ValueError(f"z_t must be [B,S,P,D], got {tuple(z_t.shape)}")
        if int(z_t.shape[-1]) != int(self.config.out_channels):
            raise ValueError(f"z_t dim {z_t.shape[-1]} != out_channels {self.config.out_channels}")
        if not (z_splat.shape == scaffold_tok.shape == z_t.shape):
            raise ValueError("z_t, z_splat and scaffold_tok must share shape [B,S,P,D]")
        for name, mask in (("M_preserve", M_preserve), ("M_source", M_source), ("M_dest", M_dest)):
            if mask.shape != z_t.shape[:-1] + (1,):
                raise ValueError(f"{name} must be [B,S,P,1], got {tuple(mask.shape)}")

    def _build_video_state(
        self,
        sigma: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token inpaint state: preserve/source/dest/edit/keep/effective sigma."""
        b, s, p, c = M_preserve.shape
        if c != 1:
            raise ValueError(f"M_preserve must be [B,S,P,1], got {tuple(M_preserve.shape)}")
        if sigma.shape != (b,):
            raise ValueError(f"sigma must be shape [B], got {tuple(sigma.shape)}")
        preserve = M_preserve.to(dtype=torch.float32).clamp(0.0, 1.0)
        source = M_source.to(device=M_preserve.device, dtype=torch.float32).clamp(0.0, 1.0)
        dest = M_dest.to(device=M_preserve.device, dtype=torch.float32).clamp(0.0, 1.0)
        if source.shape != (b, s, p, 1) or dest.shape != (b, s, p, 1):
            raise ValueError(
                f"M_source/M_dest must match M_preserve [B,S,P,1], got {tuple(source.shape)} and {tuple(dest.shape)}"
            )
        edit = (source + dest).clamp(0.0, 1.0)
        keep = 1.0 - edit
        sigma_eff = sigma.to(device=M_preserve.device, dtype=torch.float32).view(b, 1, 1, 1) * edit
        state = torch.cat([preserve, source, dest, edit, keep, sigma_eff], dim=-1)
        if int(state.shape[-1]) != int(self.config.video_state_dim):
            raise RuntimeError(
                f"video state dim {state.shape[-1]} != configured {self.config.video_state_dim}"
            )
        return state

    def _pack_video(
        self,
        z_t: torch.Tensor,
        z_splat: torch.Tensor,
        scaffold_tok: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_video_control_inputs(z_t, z_splat, scaffold_tok, M_preserve, M_source, M_dest)
        packed = torch.cat([z_t, z_splat, scaffold_tok, M_preserve, M_source, M_dest], dim=-1)
        if packed.shape[-1] != int(self.config.in_channels):
            raise ValueError(f"Packed input channels {packed.shape[-1]} != model in_channels {self.config.in_channels}")
        return packed

    def _prepare_asset_kv(
        self,
        F_asset_tokens: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Legacy 3D asset-token compatibility path.

        The current RAE-style training path passes 5D tokenizer latents through
        ``AssetFrameCompressor``. For that path, unconditional asset dropout is
        represented by an all-False asset mask, not by injecting ``null_kv``.
        """
        if F_asset_tokens.ndim != 3:
            raise ValueError(f"F_asset_tokens must be [B,N,C], got {tuple(F_asset_tokens.shape)}")
        batch_size, num_tokens, text_dim = F_asset_tokens.shape
        if text_dim != int(self.config.text_dim):
            raise ValueError(f"F_asset_tokens dim {text_dim} != text_dim {self.config.text_dim}")
        null = self.null_kv.to(device=F_asset_tokens.device, dtype=F_asset_tokens.dtype)
        if num_tokens == 0:
            return null.expand(batch_size, 1, -1), None
        if encoder_attention_mask is None:
            return F_asset_tokens, None
        mask = encoder_attention_mask.to(device=F_asset_tokens.device, dtype=torch.bool)
        if mask.shape != (batch_size, num_tokens):
            raise ValueError(f"encoder_attention_mask shape {tuple(mask.shape)} != {(batch_size, num_tokens)}")
        if bool(mask.all().item()):
            return F_asset_tokens, None

        empty_rows = ~mask.any(dim=1)
        if not bool(empty_rows.any().item()):
            return F_asset_tokens, mask

        first_slot = F_asset_tokens[:, :1, :]
        null_first = null.expand(batch_size, 1, -1)
        keep_real = (~empty_rows).view(batch_size, 1, 1)
        new_first = torch.where(keep_real, first_slot, null_first)
        kv = torch.cat([new_first, F_asset_tokens[:, 1:, :]], dim=1)
        new_first_mask = mask[:, :1] | empty_rows.view(batch_size, 1)
        new_mask = torch.cat([new_first_mask, mask[:, 1:]], dim=1)
        return kv, new_mask

    def _empty_asset_rows(
        self,
        asset_condition_kind: Any,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if asset_condition_kind is None:
            return None, None
        if isinstance(asset_condition_kind, str):
            replace = asset_condition_kind in ("mode_b", "mode_b_empty", "empty")
            append = asset_condition_kind in (
                "mode_a_with_empty",
                "mode_a_plus_empty",
                "with_empty",
                "plus_empty",
            )
            return (
                torch.full((batch_size,), replace, device=device, dtype=torch.bool),
                torch.full((batch_size,), append, device=device, dtype=torch.bool),
            )
        if torch.is_tensor(asset_condition_kind):
            rows = asset_condition_kind.to(device=device)
            if rows.dtype == torch.bool:
                return rows.reshape(batch_size), None
            return rows.reshape(batch_size).ne(0), None
        values = list(asset_condition_kind)
        if len(values) != batch_size:
            raise ValueError(f"asset_condition_kind length {len(values)} != batch size {batch_size}")
        replace = torch.tensor(
            [str(v) in ("mode_b", "mode_b_empty", "empty") for v in values],
            device=device,
            dtype=torch.bool,
        )
        append = torch.tensor(
            [
                str(v)
                in (
                    "mode_a_with_empty",
                    "mode_a_plus_empty",
                    "with_empty",
                    "plus_empty",
                )
                for v in values
            ],
            device=device,
            dtype=torch.bool,
        )
        return replace, append

    @staticmethod
    def _condition_kind_rows(
        condition_kind: Any,
        batch_size: int,
        device: torch.device,
        true_values: set[str],
    ) -> torch.Tensor | None:
        if condition_kind is None:
            return None
        if isinstance(condition_kind, str):
            value = condition_kind.lower()
            return torch.full((batch_size,), value in true_values, device=device, dtype=torch.bool)
        if torch.is_tensor(condition_kind):
            return None
        values = list(condition_kind)
        if len(values) != batch_size:
            raise ValueError(f"condition kind length {len(values)} != batch size {batch_size}")
        return torch.tensor(
            [str(v).lower() in true_values for v in values],
            device=device,
            dtype=torch.bool,
        )

    def _asset_null_rows(
        self,
        asset_condition_kind: Any,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        return self._condition_kind_rows(
            asset_condition_kind,
            batch_size,
            device,
            {"asset_uncond", "asset_null", "asset_missing", "missing_asset"},
        )

    def _camera_null_rows(
        self,
        camera_condition_kind: Any,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        return self._condition_kind_rows(
            camera_condition_kind,
            batch_size,
            device,
            {"camera_uncond", "camera_null", "camera_missing", "missing_camera"},
        )

    @staticmethod
    def _uniform_sample_indices(valid_idx: torch.Tensor, max_count: int) -> torch.Tensor:
        if int(valid_idx.numel()) <= int(max_count):
            return valid_idx
        sample_pos = torch.linspace(
            0,
            int(valid_idx.numel()) - 1,
            int(max_count),
            device=valid_idx.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
        return valid_idx[sample_pos].unique(sorted=True)

    def _get_asset_compressor(self) -> AssetFrameCompressor:
        if self.asset_compressor is None:
            self.asset_compressor = AssetFrameCompressor(
                int(self.config.asset_dim),
                int(self.config.hidden_size),
                max_assets=int(self.config.max_assets),
                max_frames=int(self.config.max_frames),
            ).to(device=self.empty_asset_embed.device, dtype=self.empty_asset_embed.dtype)
        return self.asset_compressor

    @staticmethod
    def _uniform_sample_mask_indices(valid_mask: torch.Tensor, max_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must be [N,P], got {tuple(valid_mask.shape)}")
        n, p = valid_mask.shape
        max_count = int(max_count)
        if max_count <= 0:
            idx = torch.zeros((n, 0), device=valid_mask.device, dtype=torch.long)
            keep = torch.zeros((n, 0), device=valid_mask.device, dtype=torch.bool)
            return idx, keep
        patch_idx = torch.arange(p, device=valid_mask.device, dtype=torch.long).view(1, p)
        sorted_idx = torch.where(valid_mask, patch_idx, torch.full_like(patch_idx, p)).sort(dim=1).values
        counts = valid_mask.sum(dim=1)
        slots = torch.arange(max_count, device=valid_mask.device, dtype=torch.long).view(1, max_count)
        sample_count = counts.clamp_max(max_count)
        large_ranks = (
            torch.linspace(0.0, 1.0, max_count, device=valid_mask.device).view(1, max_count)
            * counts.sub(1).clamp_min(0).to(dtype=torch.float32).view(n, 1)
        ).round().to(dtype=torch.long)
        ranks = torch.where(counts.view(n, 1).le(max_count), slots.expand(n, -1), large_ranks)
        ranks = ranks.clamp_min(0).clamp_max(max(p - 1, 0))
        sampled = sorted_idx.gather(1, ranks).clamp_max(max(p - 1, 0))
        return sampled, slots.lt(sample_count.view(n, 1))

    def _pad_sparse_condition(
        self,
        token_rows: list[torch.Tensor],
        pos_rows: list[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch_size = len(token_rows)
        hidden_size = int(self.config.hidden_size)
        if max_tokens is None:
            max_len = max((int(row.shape[0]) for row in token_rows), default=0)
        else:
            max_len = min(max((int(row.shape[0]) for row in token_rows), default=0), int(max_tokens))
        if max_len <= 0:
            empty_tokens = torch.zeros((batch_size, 0, hidden_size), device=device, dtype=dtype)
            empty_pos = torch.zeros((batch_size, 0, 3), device=device, dtype=torch.long)
            return empty_tokens, None, empty_pos
        tokens = torch.zeros((batch_size, max_len, hidden_size), device=device, dtype=dtype)
        mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.bool)
        pos_dtype = (
            torch.float32
            if any(torch.is_tensor(row) and row.dtype.is_floating_point for row in pos_rows)
            else torch.long
        )
        positions = torch.zeros((batch_size, max_len, 3), device=device, dtype=pos_dtype)
        for b, row in enumerate(token_rows):
            n = min(int(row.shape[0]), max_len)
            if n <= 0:
                continue
            tokens[b, :n] = row[:n].to(device=device, dtype=dtype)
            positions[b, :n] = pos_rows[b][:n].to(device=device, dtype=pos_dtype)
            mask[b, :n] = True
        return tokens, mask, positions

    def _build_sparse_asset_condition(
        self,
        F_asset_tokens: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        *,
        seq_len: int,
        num_patches: int,
        patch_grid: tuple[int, int],
        asset_condition_kind: Any = None,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b = int(F_asset_tokens.shape[0])
        device = F_asset_tokens.device
        dtype = F_asset_tokens.dtype
        hidden_size = int(self.config.hidden_size)
        max_assets = int(self.config.max_assets)
        max_asset_tokens = int(self.config.max_asset_tokens)
        max_patch = int(self.config.max_asset_patch_tokens_per_asset_frame)
        gh, gw = patch_grid
        if frame_ids is None:
            frame_ids_t = torch.arange(int(seq_len), device=device, dtype=torch.long).view(1, int(seq_len)).expand(b, -1)
        else:
            frame_ids_t = frame_ids.to(device=device, dtype=torch.long)
            if frame_ids_t.ndim == 1:
                frame_ids_t = frame_ids_t.view(1, -1).expand(b, -1)
            if frame_ids_t.shape != (b, int(seq_len)):
                raise ValueError(f"frame_ids must be [S] or [B,S], got {tuple(frame_ids.shape)}")

        replace_rows, append_rows = self._empty_asset_rows(asset_condition_kind, b, device)
        null_rows = self._asset_null_rows(asset_condition_kind, b, device)
        token_rows: list[torch.Tensor] = []
        pos_rows: list[torch.Tensor] = []

        if F_asset_tokens.ndim == 5:
            if int(F_asset_tokens.shape[2]) != int(seq_len) or int(F_asset_tokens.shape[3]) != int(num_patches):
                raise ValueError(
                    f"asset latents shape {tuple(F_asset_tokens.shape)} incompatible with S={seq_len}, P={num_patches}"
                )
            if int(F_asset_tokens.shape[-1]) != int(self.config.asset_dim):
                raise ValueError(
                    f"5D asset latent dim {F_asset_tokens.shape[-1]} != asset_dim {self.config.asset_dim}"
                )
            k = min(int(F_asset_tokens.shape[1]), max_assets)
            if encoder_attention_mask is None:
                valid = torch.ones((b, int(F_asset_tokens.shape[1]), seq_len, num_patches), device=device, dtype=torch.bool)
            else:
                if encoder_attention_mask.shape != F_asset_tokens.shape[:-1]:
                    raise ValueError(
                        f"asset valid mask shape {tuple(encoder_attention_mask.shape)} "
                        f"!= {tuple(F_asset_tokens.shape[:-1])}"
                    )
                valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
            projected = self.asset_latent_proj(self.asset_latent_norm(F_asset_tokens[:, :k]))
            visual_pos = self._target_position_ids(
                batch_size=b,
                seq_len=seq_len,
                num_patches=num_patches,
                patch_grid=patch_grid,
                device=device,
                frame_ids=frame_ids_t,
                fps=fps,
            ).reshape(b, int(seq_len), num_patches, 3)
            flat_valid = valid[:, :k].reshape(b * k * int(seq_len), num_patches)
            sampled, sample_mask = self._uniform_sample_mask_indices(flat_valid, max_patch)
            projected_flat = projected.reshape(b * k * int(seq_len), num_patches, hidden_size)
            patch_h = projected_flat.gather(
                1,
                sampled.unsqueeze(-1).expand(-1, -1, hidden_size),
            )
            valid_f = flat_valid.to(dtype=projected.dtype)
            summary_h = (projected_flat * valid_f.unsqueeze(-1)).sum(dim=1) / valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)

            asset_ids = torch.arange(k, device=device, dtype=torch.long).view(1, k, 1).expand(b, -1, int(seq_len))
            frame_idx_t = torch.arange(int(seq_len), device=device, dtype=torch.long).view(1, 1, int(seq_len)).expand(b, k, -1)
            row_ids = torch.arange(b, device=device, dtype=torch.long).view(b, 1, 1).expand(-1, k, int(seq_len))
            asset_ids_flat = asset_ids.reshape(-1).clamp_max(self.asset_slot_embed.num_embeddings - 1)
            frame_idx_flat = frame_idx_t.reshape(-1).clamp_max(self.asset_frame_embed.num_embeddings - 1)
            row_ids_flat = row_ids.reshape(-1)
            slot_emb = self.asset_slot_embed(asset_ids_flat).to(dtype=projected.dtype)
            frame_emb = self.asset_frame_embed(frame_idx_flat).to(dtype=projected.dtype)
            patch_h = (
                patch_h
                + slot_emb[:, None, :]
                + frame_emb[:, None, :]
                + self.asset_patch_modality_embed.to(device=device, dtype=projected.dtype)
            )
            summary_h = (
                summary_h
                + slot_emb
                + frame_emb
                + self.asset_summary_modality_embed.to(device=device, dtype=projected.dtype).view(1, -1)
            )
            visual_pos_flat = (
                visual_pos[:, None]
                .expand(-1, k, -1, -1, -1)
                .reshape(b * k * int(seq_len), num_patches, 3)
            )
            patch_y = torch.div(torch.arange(num_patches, device=device, dtype=torch.long), gw, rounding_mode="floor")
            patch_x = torch.arange(num_patches, device=device, dtype=torch.long) % gw
            frame_has_tokens = flat_valid.any(dim=1)
            asset_position_mode = str(getattr(self.config, "asset_position_mode", "localized"))
            if asset_position_mode not in ASSET_POSITION_MODES:
                raise ValueError(
                    f"asset_position_mode must be one of {ASSET_POSITION_MODES}, got {asset_position_mode!r}"
                )
            if asset_position_mode == "canonical":
                valid_f32 = flat_valid.to(dtype=torch.float32)
                y_full = patch_y.to(dtype=torch.float32).view(1, -1).expand(flat_valid.shape[0], -1)
                x_full = patch_x.to(dtype=torch.float32).view(1, -1).expand(flat_valid.shape[0], -1)
                min_y = torch.where(
                    flat_valid,
                    y_full,
                    torch.full_like(y_full, float(gh)),
                ).min(dim=1).values
                min_x = torch.where(
                    flat_valid,
                    x_full,
                    torch.full_like(x_full, float(gw)),
                ).min(dim=1).values
                min_y = torch.where(frame_has_tokens, min_y, torch.zeros_like(min_y))
                min_x = torch.where(frame_has_tokens, min_x, torch.zeros_like(min_x))
                local_y = y_full - min_y.view(-1, 1)
                local_x = x_full - min_x.view(-1, 1)
                canonical_pos_flat = torch.zeros(
                    visual_pos_flat.shape,
                    device=device,
                    dtype=torch.float32,
                )
                canonical_pos_flat[..., 0] = visual_pos_flat[..., 0].to(dtype=torch.float32)
                canonical_pos_flat[..., 1] = float(gh) + local_y
                canonical_pos_flat[..., 2] = float(gw) + local_x
                patch_pos = canonical_pos_flat.gather(1, sampled.unsqueeze(-1).expand(-1, -1, 3))
                patch_pos = torch.where(sample_mask.unsqueeze(-1), patch_pos, torch.zeros_like(patch_pos))
                counts_f = valid_f32.sum(dim=1).clamp_min(1.0)
                mean_y = (valid_f32 * local_y).sum(dim=1) / counts_f
                mean_x = (valid_f32 * local_x).sum(dim=1) / counts_f
                summary_t = visual_pos[row_ids_flat, frame_idx_t.reshape(-1), 0, 0].to(dtype=torch.float32)
                summary_pos = torch.stack(
                    [
                        summary_t,
                        torch.full_like(mean_y, float(gh)) + mean_y,
                        torch.full_like(mean_x, float(gw)) + mean_x,
                    ],
                    dim=-1,
                )
            else:
                patch_pos = visual_pos_flat.gather(1, sampled.unsqueeze(-1).expand(-1, -1, 3))
                patch_pos = torch.where(sample_mask.unsqueeze(-1), patch_pos, torch.zeros_like(patch_pos))
                counts_f = valid_f.sum(dim=1).clamp_min(1.0)
                mean_y = (
                    flat_valid.to(dtype=torch.float32) * patch_y.to(dtype=torch.float32).view(1, -1)
                ).sum(dim=1) / counts_f
                mean_x = (
                    flat_valid.to(dtype=torch.float32) * patch_x.to(dtype=torch.float32).view(1, -1)
                ).sum(dim=1) / counts_f
                summary_t = visual_pos[row_ids_flat, frame_idx_t.reshape(-1), 0, 0]
                summary_y = mean_y.round().clamp(0, gh - 1)
                summary_x = mean_x.round().clamp(0, gw - 1)
                if visual_pos.dtype.is_floating_point:
                    summary_pos = torch.stack([summary_t, summary_y, summary_x], dim=-1).to(dtype=visual_pos.dtype)
                else:
                    summary_pos = torch.stack(
                        [
                            summary_t.to(dtype=torch.long),
                            summary_y.to(dtype=torch.long),
                            summary_x.to(dtype=torch.long),
                        ],
                        dim=-1,
                    )
            block_tokens = torch.cat([patch_h, summary_h[:, None, :]], dim=1).reshape(
                b, k * int(seq_len) * (max_patch + 1), hidden_size
            )
            block_pos = torch.cat([patch_pos, summary_pos[:, None, :]], dim=1).reshape(
                b, k * int(seq_len) * (max_patch + 1), 3
            )
            block_mask = torch.cat([sample_mask, frame_has_tokens[:, None]], dim=1).reshape(
                b, k * int(seq_len) * (max_patch + 1)
            )
            null_token = self.asset_null_condition_embed.to(device=device, dtype=projected.dtype).reshape(1, hidden_size)
            for row in range(b):
                pieces = [block_tokens[row, block_mask[row]]]
                positions = [block_pos[row, block_mask[row]]]
                if null_rows is not None and bool(null_rows[row].item()):
                    pieces = [null_token]
                    positions = [torch.tensor([[0, 0, 0]], device=device, dtype=block_pos.dtype)]
                elif replace_rows is not None and bool(replace_rows[row].item()):
                    pieces = [self.empty_asset_embed.to(device=device, dtype=projected.dtype).reshape(1, hidden_size)]
                    positions = [torch.tensor([[0, 0, 0]], device=device, dtype=block_pos.dtype)]
                elif append_rows is not None and bool(append_rows[row].item()):
                    pieces.append(self.empty_asset_embed.to(device=device, dtype=projected.dtype).reshape(1, hidden_size))
                    positions.append(torch.tensor([[0, 0, 0]], device=device, dtype=block_pos.dtype))
                if pieces and sum(int(piece.shape[0]) for piece in pieces) > 0:
                    token_rows.append(torch.cat(pieces, dim=0)[:max_asset_tokens])
                    pos_rows.append(torch.cat(positions, dim=0)[:max_asset_tokens])
                else:
                    token_rows.append(torch.zeros((0, hidden_size), device=device, dtype=projected.dtype))
                    pos_rows.append(torch.zeros((0, 3), device=device, dtype=block_pos.dtype))
            return self._pad_sparse_condition(token_rows, pos_rows, device=device, dtype=projected.dtype, max_tokens=max_asset_tokens)

        if F_asset_tokens.ndim == 4:
            return self._build_sparse_asset_condition(
                F_asset_tokens.unsqueeze(1),
                None if encoder_attention_mask is None else encoder_attention_mask.unsqueeze(1),
                seq_len=seq_len,
                num_patches=num_patches,
                patch_grid=patch_grid,
                asset_condition_kind=asset_condition_kind,
                frame_ids=frame_ids_t,
                fps=fps,
            )

        if F_asset_tokens.ndim != 3:
            raise ValueError(f"F_asset_tokens must be [B,N,C] or [B,K,S,P,C], got {tuple(F_asset_tokens.shape)}")
        n = int(F_asset_tokens.shape[1])
        if n == 0:
            null_token = self.asset_null_condition_embed.to(device=device, dtype=dtype).reshape(1, hidden_size)
            for row in range(b):
                needs_null = null_rows is not None and bool(null_rows[row].item())
                needs_empty = (
                    (replace_rows is not None and bool(replace_rows[row].item()))
                    or (append_rows is not None and bool(append_rows[row].item()))
                )
                if needs_null:
                    token_rows.append(null_token)
                    pos_rows.append(torch.tensor([[0, 0, 0]], device=device, dtype=torch.long))
                elif needs_empty:
                    token_rows.append(self.empty_asset_embed.to(device=device, dtype=dtype).reshape(1, hidden_size))
                    pos_rows.append(torch.tensor([[0, 0, 0]], device=device, dtype=torch.long))
                else:
                    token_rows.append(torch.zeros((0, hidden_size), device=device, dtype=dtype))
                    pos_rows.append(torch.zeros((0, 3), device=device, dtype=torch.long))
            return self._pad_sparse_condition(token_rows, pos_rows, device=device, dtype=dtype, max_tokens=max_asset_tokens)

        c = int(F_asset_tokens.shape[-1])
        if c == int(self.config.asset_dim):
            tokens = self.asset_encoded_proj(F_asset_tokens)
        elif c == int(self.config.text_dim):
            kv, kv_mask = self._prepare_asset_kv(F_asset_tokens, encoder_attention_mask)
            tokens = self.asset_direct_proj(kv)
            encoder_attention_mask = kv_mask
        else:
            raise ValueError(
                f"F_asset_tokens dim {c} must match asset_dim={self.config.asset_dim} "
                f"or legacy text_dim={self.config.text_dim}"
            )
        if encoder_attention_mask is None:
            mask = torch.ones(tokens.shape[:2], device=device, dtype=torch.bool)
        else:
            mask = encoder_attention_mask.to(device=device, dtype=torch.bool)
            if mask.shape != tokens.shape[:2]:
                raise ValueError(f"encoder_attention_mask shape {tuple(mask.shape)} != {tuple(tokens.shape[:2])}")
        tokens = tokens + self.asset_patch_modality_embed.to(device=device, dtype=tokens.dtype)
        positions = torch.zeros((b, tokens.shape[1], 3), device=device, dtype=torch.long)
        if null_rows is not None and bool(null_rows.any().item()):
            tokens = tokens.clone()
            mask = mask.clone()
            null = self.asset_null_condition_embed.to(device=device, dtype=tokens.dtype).expand(b, 1, -1)
            tokens[:, :1] = torch.where(null_rows.view(-1, 1, 1), null, tokens[:, :1])
            mask[null_rows] = False
            mask[null_rows, 0] = True
        if replace_rows is not None and bool(replace_rows.any().item()):
            tokens = tokens.clone()
            mask = mask.clone()
            empty = self.empty_asset_embed.to(device=device, dtype=tokens.dtype).expand(b, 1, -1)
            tokens[:, :1] = torch.where(replace_rows.view(-1, 1, 1), empty, tokens[:, :1])
            mask[replace_rows] = False
            mask[replace_rows, 0] = True
        if append_rows is not None and bool(append_rows.any().item()):
            tokens = tokens.clone()
            mask = mask.clone()
            empty = self.empty_asset_embed.to(device=device, dtype=tokens.dtype).expand(b, 1, -1)
            for row in append_rows.nonzero(as_tuple=False).flatten().tolist():
                free = (~mask[row]).nonzero(as_tuple=False).flatten()
                slot = int(free[0].item()) if int(free.numel()) > 0 else 0
                tokens[row, slot] = empty[row, 0]
                mask[row, slot] = True
        return tokens[:, :max_asset_tokens], mask[:, :max_asset_tokens], positions[:, :max_asset_tokens]

    def _build_asset_condition(
        self,
        F_asset_tokens: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        *,
        seq_len: int,
        num_patches: int,
        patch_grid: tuple[int, int],
        asset_condition_kind: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        def null_condition(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
            # Legacy 3D-token path only. The 5D asset path uses
            # ``empty_asset_embed`` for conditional Mode-B empty scenes.
            null = self.asset_direct_proj(
                self.null_kv.to(device=F_asset_tokens.device, dtype=F_asset_tokens.dtype)
            )
            null = null.expand(batch_size, 1, -1)
            mask = torch.ones((batch_size, 1), device=F_asset_tokens.device, dtype=torch.bool)
            return null, mask

        def inject_null_rows(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            empty_rows = ~mask.any(dim=1)
            if not bool(empty_rows.any().item()):
                return tokens, mask
            null, _ = null_condition(tokens.shape[0])
            tokens = tokens.clone()
            mask = mask.clone()
            tokens[:, :1, :] = torch.where(empty_rows.view(-1, 1, 1), null.to(dtype=tokens.dtype), tokens[:, :1, :])
            mask[:, :1] = mask[:, :1] | empty_rows.view(-1, 1)
            return tokens, mask

        def apply_asset_null_condition(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            null_rows = self._asset_null_rows(asset_condition_kind, tokens.shape[0], tokens.device)
            if null_rows is None or not bool(null_rows.any().item()):
                return tokens, mask
            tokens = tokens.clone()
            mask = mask.clone()
            null = self.asset_null_condition_embed.to(device=tokens.device, dtype=tokens.dtype).expand(
                tokens.shape[0], 1, -1
            )
            tokens[:, :1, :] = torch.where(null_rows.view(-1, 1, 1), null, tokens[:, :1, :])
            mask[null_rows] = False
            mask[null_rows, 0] = True
            return tokens, mask

        def empty_condition_rows(batch_size: int, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if asset_condition_kind is None:
                return None, None
            if isinstance(asset_condition_kind, str):
                replace = asset_condition_kind in ("mode_b", "mode_b_empty", "empty")
                append = asset_condition_kind in (
                    "mode_a_with_empty",
                    "mode_a_plus_empty",
                    "with_empty",
                    "plus_empty",
                )
                return (
                    torch.full((batch_size,), replace, device=device, dtype=torch.bool),
                    torch.full((batch_size,), append, device=device, dtype=torch.bool),
                )
            if torch.is_tensor(asset_condition_kind):
                rows = asset_condition_kind.to(device=device)
                if rows.dtype == torch.bool:
                    return rows.reshape(batch_size), None
                return rows.reshape(batch_size).ne(0), None
            values = list(asset_condition_kind)
            if len(values) != batch_size:
                raise ValueError(f"asset_condition_kind length {len(values)} != batch size {batch_size}")
            replace = torch.tensor(
                [str(v) in ("mode_b", "mode_b_empty", "empty") for v in values],
                device=device,
                dtype=torch.bool,
            )
            append = torch.tensor(
                [
                    str(v)
                    in (
                        "mode_a_with_empty",
                        "mode_a_plus_empty",
                        "with_empty",
                        "plus_empty",
                    )
                    for v in values
                ],
                device=device,
                dtype=torch.bool,
            )
            return replace, append

        def apply_empty_condition(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            replace_rows, append_rows = empty_condition_rows(tokens.shape[0], tokens.device)
            if (
                (replace_rows is None or not bool(replace_rows.any().item()))
                and (append_rows is None or not bool(append_rows.any().item()))
            ):
                return tokens, mask
            tokens = tokens.clone()
            mask = mask.clone()
            empty = self.empty_asset_embed.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.shape[0], 1, -1)
            if replace_rows is not None and bool(replace_rows.any().item()):
                # Mode B is still a conditional edit: there is an explicit hole
                # but no target asset to insert. Keep exactly one learned asset
                # token visible so the trunk can distinguish this from
                # CFG/unconditional asset dropout.
                tokens[:, :1, :] = torch.where(replace_rows.view(-1, 1, 1), empty, tokens[:, :1, :])
                mask[replace_rows] = False
                mask[replace_rows, 0] = True
            if append_rows is not None and bool(append_rows.any().item()):
                # Validation combined/replacement/repositioning edits may contain
                # both a deletion target and real inserted/replaced/moved assets.
                # Preserve existing real asset slots and put the deletion signal
                # in the first free slot; if all slots are occupied, overwrite
                # slot 0 rather than growing beyond the fixed five-token budget.
                for row in append_rows.nonzero(as_tuple=False).flatten().tolist():
                    free = (~mask[row]).nonzero(as_tuple=False).flatten()
                    slot = int(free[0].item()) if int(free.numel()) > 0 else 0
                    tokens[row, slot, :] = empty[row, 0, :]
                    mask[row, slot] = True
            return tokens, mask

        if F_asset_tokens.ndim == 5:
            valid = None
            if encoder_attention_mask is not None:
                if encoder_attention_mask.shape != F_asset_tokens.shape[:-1]:
                    raise ValueError(
                        f"asset valid mask shape {tuple(encoder_attention_mask.shape)} "
                        f"!= {tuple(F_asset_tokens.shape[:-1])}"
                    )
                valid = encoder_attention_mask.to(device=F_asset_tokens.device, dtype=torch.bool)
            tokens, mask = self._get_asset_compressor()(F_asset_tokens, patch_grid=patch_grid, valid_mask=valid)
            tokens, mask = apply_asset_null_condition(tokens, mask)
            return apply_empty_condition(tokens, mask)
        if F_asset_tokens.ndim == 4:
            tokens, mask = self._get_asset_compressor()(F_asset_tokens.unsqueeze(1), patch_grid=patch_grid)
            tokens, mask = apply_asset_null_condition(tokens, mask)
            return apply_empty_condition(tokens, mask)
        if F_asset_tokens.ndim != 3:
            raise ValueError(f"F_asset_tokens must be [B,N,C] or [B,K,S,P,C], got {tuple(F_asset_tokens.shape)}")

        b, n, c = F_asset_tokens.shape
        if n == 0:
            replace_rows, append_rows = empty_condition_rows(b, F_asset_tokens.device)
            null_rows = self._asset_null_rows(asset_condition_kind, b, F_asset_tokens.device)
            if (
                (null_rows is not None and bool(null_rows.any().item()))
                or
                (replace_rows is not None and bool(replace_rows.any().item()))
                or (append_rows is not None and bool(append_rows.any().item()))
            ):
                tokens = F_asset_tokens.new_zeros((b, int(self.config.max_assets), int(self.config.hidden_size)))
                mask = torch.zeros((b, int(self.config.max_assets)), device=F_asset_tokens.device, dtype=torch.bool)
                tokens, mask = apply_asset_null_condition(tokens, mask)
                return apply_empty_condition(tokens, mask)
            return F_asset_tokens.new_zeros((b, 0, int(self.config.hidden_size))), None

        tokens_per_asset = int(seq_len) * int(num_patches)
        if c == int(self.config.asset_dim):
            mask = None
            if encoder_attention_mask is not None:
                mask = encoder_attention_mask.to(device=F_asset_tokens.device, dtype=torch.bool)
                if mask.shape != (b, n):
                    raise ValueError(f"encoder_attention_mask shape {tuple(mask.shape)} != {(b, n)}")
            if tokens_per_asset > 0 and n >= tokens_per_asset and n % tokens_per_asset == 0:
                k = min(n // tokens_per_asset, int(self.config.max_assets))
                usable = k * tokens_per_asset
                latents = F_asset_tokens[:, :usable].reshape(b, k, seq_len, num_patches, c)
                valid = None
                if mask is not None:
                    valid = mask[:, :usable].reshape(b, k, seq_len, num_patches)
                tokens, out_mask = self._get_asset_compressor()(latents, patch_grid=patch_grid, valid_mask=valid)
                tokens, out_mask = apply_asset_null_condition(tokens, out_mask)
                return apply_empty_condition(tokens, out_mask)
            projected = self.asset_encoded_proj(F_asset_tokens)
            if mask is None:
                mask = torch.ones((b, n), device=F_asset_tokens.device, dtype=torch.bool)
            projected, mask = apply_asset_null_condition(projected, mask)
            return apply_empty_condition(projected, mask)

        if c != int(self.config.text_dim):
            raise ValueError(
                f"F_asset_tokens dim {c} must match asset_dim={self.config.asset_dim} "
                f"or legacy text_dim={self.config.text_dim}"
            )
        kv, kv_mask = self._prepare_asset_kv(F_asset_tokens, encoder_attention_mask)
        projected = self.asset_direct_proj(kv)
        if kv_mask is None:
            kv_mask = torch.ones((b, projected.shape[1]), device=projected.device, dtype=torch.bool)
        projected, kv_mask = apply_asset_null_condition(projected, kv_mask)
        return apply_empty_condition(projected, kv_mask)

    def _build_text_condition(
        self,
        z_t: torch.Tensor,
        text_tokens: torch.Tensor | None,
        text_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b = int(z_t.shape[0])
        if text_tokens is None:
            return z_t.new_zeros((b, 0, int(self.config.hidden_size))), None
        if text_tokens.ndim != 3 or int(text_tokens.shape[0]) != b:
            raise ValueError(f"text_tokens must be [B,T,C], got {tuple(text_tokens.shape)}")
        if int(text_tokens.shape[-1]) != int(self.config.qwen_dim):
            raise ValueError(f"text_tokens dim {text_tokens.shape[-1]} != qwen_dim {self.config.qwen_dim}")
        text_dtype = self.text_proj.weight.dtype
        tokens = self.text_proj(self.text_norm(text_tokens).to(dtype=text_dtype))
        tokens = tokens + self.text_modality_embed.to(device=tokens.device, dtype=tokens.dtype)
        if text_attention_mask is None:
            return tokens, None
        mask = text_attention_mask.to(device=z_t.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"text_attention_mask shape {tuple(mask.shape)} != {tuple(tokens.shape[:2])}")
        return tokens, mask

    def _target_position_ids(
        self,
        *,
        batch_size: int,
        seq_len: int,
        num_patches: int,
        patch_grid: tuple[int, int],
        device: torch.device,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        gh, gw = patch_grid
        if frame_ids is None:
            frames = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
        else:
            frames = frame_ids.to(device=device, dtype=torch.long)
            if frames.ndim == 1:
                frames = frames.view(1, -1).expand(batch_size, -1)
            if frames.shape != (batch_size, seq_len):
                raise ValueError(f"frame_ids must be [S] or [B,S], got {tuple(frame_ids.shape)}")
        patch_idx = torch.arange(num_patches, device=device, dtype=torch.long)
        y = torch.div(patch_idx, gw, rounding_mode="floor")
        x = patch_idx % gw
        if fps is None:
            temporal = frames.to(dtype=torch.long)
            pos_dtype = torch.long
        else:
            fps_t = torch.as_tensor(fps, device=device, dtype=torch.float32)
            if fps_t.ndim == 0:
                fps_t = fps_t.view(1).expand(batch_size)
            elif fps_t.ndim == 1 and int(fps_t.shape[0]) == 1:
                fps_t = fps_t.expand(batch_size)
            if fps_t.shape != (batch_size,):
                raise ValueError(f"fps must be scalar or [B], got {tuple(fps_t.shape)}")
            temporal = frames.to(dtype=torch.float32) * (24.0 / fps_t.clamp_min(1e-6)).view(batch_size, 1)
            pos_dtype = torch.float32
        pos = torch.zeros((batch_size, seq_len, num_patches, 3), device=device, dtype=pos_dtype)
        pos[..., 0] = temporal[:, :, None].to(dtype=pos_dtype)
        pos[..., 1] = y.view(1, 1, num_patches)
        pos[..., 2] = x.view(1, 1, num_patches)
        return pos.reshape(batch_size, seq_len * num_patches, 3)

    def _text_position_ids(self, batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        # RAEv2 does not apply RoPE to text condition tokens. The Qwen text
        # hidden states already contain token order, so use zero-angle RoPE here.
        return torch.zeros((batch_size, num_tokens, 3), device=device, dtype=torch.long)

    def _camera_position_ids(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        patch_grid: tuple[int, int] | None = None,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_ids is None:
            frames = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
        else:
            frames = frame_ids.to(device=device, dtype=torch.long)
            if frames.ndim == 1:
                frames = frames.view(1, -1).expand(batch_size, -1)
            if frames.shape != (batch_size, seq_len):
                raise ValueError(f"frame_ids must be [S] or [B,S], got {tuple(frame_ids.shape)}")
        if fps is None:
            temporal = frames.to(dtype=torch.long)
            pos_dtype = torch.long
        else:
            fps_t = torch.as_tensor(fps, device=device, dtype=torch.float32)
            if fps_t.ndim == 0:
                fps_t = fps_t.view(1).expand(batch_size)
            elif fps_t.ndim == 1 and int(fps_t.shape[0]) == 1:
                fps_t = fps_t.expand(batch_size)
            if fps_t.shape != (batch_size,):
                raise ValueError(f"fps must be scalar or [B], got {tuple(fps_t.shape)}")
            temporal = frames.to(dtype=torch.float32) * (24.0 / fps_t.clamp_min(1e-6)).view(batch_size, 1)
            pos_dtype = torch.float32
        pos = torch.zeros((batch_size, seq_len, 3), device=device, dtype=pos_dtype)
        pos[..., 0] = temporal.to(dtype=pos_dtype)
        grid = tuple(int(v) for v in (patch_grid if patch_grid is not None else self.config.patch_grid))
        pos[..., 1] = grid[0] // 2
        pos[..., 2] = grid[1] // 2
        return pos

    def _sky_position_ids(
        self,
        *,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Cosmos-style 3D RoPE ids for scene-level sky atlas tokens."""
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            return torch.zeros((batch_size, 0, 3), device=device, dtype=torch.long)
        grid = getattr(self.config, "sky_grid", None)
        if grid is not None and int(grid[0]) * int(grid[1]) >= num_tokens:
            gh, gw = int(grid[0]), int(grid[1])
        else:
            gh = max(1, int(math.floor(num_tokens**0.5)))
            gw = max(1, int(math.ceil(num_tokens / float(gh))))
        idx = torch.arange(num_tokens, device=device, dtype=torch.long)
        y = torch.div(idx, gw, rounding_mode="floor").clamp_max(max(gh - 1, 0))
        x = (idx % gw).clamp_max(max(gw - 1, 0))
        pos = torch.zeros((batch_size, num_tokens, 3), device=device, dtype=torch.long)
        pos[..., 0] = int(self.config.sky_rope_temporal_offset)
        pos[..., 1] = y.view(1, num_tokens)
        pos[..., 2] = x.view(1, num_tokens)
        return pos

    def _build_camera_condition(
        self,
        z_t: torch.Tensor,
        camera_pose_tokens: torch.Tensor | None,
        camera_attention_mask: torch.Tensor | None,
        *,
        camera_condition_kind: Any = None,
        patch_grid: tuple[int, int] | None = None,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b, s = int(z_t.shape[0]), int(z_t.shape[1])
        hidden_size = int(self.config.hidden_size)
        null_rows = self._camera_null_rows(camera_condition_kind, b, z_t.device)
        if camera_pose_tokens is None:
            if null_rows is not None and bool(null_rows.any().item()):
                if not bool(null_rows.all().item()):
                    raise ValueError("camera_pose_tokens=None is only valid when all rows use camera_uncond.")
                tokens = self.camera_null_condition_embed.to(device=z_t.device, dtype=z_t.dtype).expand(b, s, -1)
                mask = torch.ones((b, s), device=z_t.device, dtype=torch.bool)
                pos = self._camera_position_ids(
                    batch_size=b,
                    seq_len=s,
                    device=z_t.device,
                    patch_grid=patch_grid,
                    frame_ids=frame_ids,
                    fps=fps,
                )
                return tokens, mask, pos
            empty_tokens = z_t.new_zeros((b, 0, hidden_size))
            empty_pos = torch.zeros((b, 0, 3), device=z_t.device, dtype=torch.long)
            return empty_tokens, None, empty_pos
        if camera_pose_tokens.ndim != 3 or int(camera_pose_tokens.shape[0]) != b:
            raise ValueError(f"camera_pose_tokens must be [B,S,C], got {tuple(camera_pose_tokens.shape)}")
        if int(camera_pose_tokens.shape[1]) != s:
            raise ValueError(
                f"camera_pose_tokens must have one token per video frame: "
                f"got {camera_pose_tokens.shape[1]} for S={s}"
            )
        if int(camera_pose_tokens.shape[-1]) != int(self.config.camera_cond_dim):
            raise ValueError(
                f"camera_pose_tokens dim {camera_pose_tokens.shape[-1]} "
                f"!= camera_cond_dim {self.config.camera_cond_dim}"
            )
        camera_dtype = self.camera_proj[0].weight.dtype
        tokens = self.camera_proj(self.camera_norm(camera_pose_tokens).to(dtype=camera_dtype))
        tokens = tokens + self.camera_modality_embed.to(device=tokens.device, dtype=tokens.dtype)
        if camera_attention_mask is None:
            mask = torch.ones((b, s), device=z_t.device, dtype=torch.bool)
        else:
            mask = camera_attention_mask.to(device=z_t.device, dtype=torch.bool)
            if mask.shape != (b, s):
                raise ValueError(f"camera_attention_mask shape {tuple(mask.shape)} != {(b, s)}")
        if null_rows is not None and bool(null_rows.any().item()):
            tokens = tokens.clone()
            mask = mask.clone()
            null = self.camera_null_condition_embed.to(device=tokens.device, dtype=tokens.dtype).expand(b, s, -1)
            tokens = torch.where(null_rows.view(b, 1, 1), null, tokens)
            mask[null_rows] = True
        tokens = tokens * mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        pos = self._camera_position_ids(
            batch_size=b,
            seq_len=s,
            device=z_t.device,
            patch_grid=patch_grid,
            frame_ids=frame_ids,
            fps=fps,
        )
        return tokens, mask, pos

    def _build_camera_generation(
        self,
        z_t: torch.Tensor,
        camera_gen_tokens: torch.Tensor | None,
        camera_gen_attention_mask: torch.Tensor | None,
        *,
        patch_grid: tuple[int, int] | None = None,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b, s = int(z_t.shape[0]), int(z_t.shape[1])
        hidden_size = int(self.config.hidden_size)
        if camera_gen_tokens is None:
            empty_tokens = z_t.new_zeros((b, 0, hidden_size))
            empty_pos = torch.zeros((b, 0, 3), device=z_t.device, dtype=torch.long)
            return empty_tokens, None, empty_pos
        if camera_gen_tokens.ndim != 3 or int(camera_gen_tokens.shape[0]) != b:
            raise ValueError(f"camera_gen_tokens must be [B,S,C], got {tuple(camera_gen_tokens.shape)}")
        if int(camera_gen_tokens.shape[1]) != s:
            raise ValueError(
                f"camera_gen_tokens must have one token per video frame: "
                f"got {camera_gen_tokens.shape[1]} for S={s}"
            )
        if int(camera_gen_tokens.shape[-1]) != int(self.config.camera_gen_dim):
            raise ValueError(
                f"camera_gen_tokens dim {camera_gen_tokens.shape[-1]} "
                f"!= camera_gen_dim {self.config.camera_gen_dim}"
            )
        camera_dtype = self.camera_gen_proj.weight.dtype
        tokens = self.camera_gen_proj(self.camera_gen_norm(camera_gen_tokens).to(dtype=camera_dtype))
        tokens = tokens + self.camera_gen_modality_embed.to(device=tokens.device, dtype=tokens.dtype)
        if camera_gen_attention_mask is None:
            mask = torch.ones((b, s), device=z_t.device, dtype=torch.bool)
        else:
            mask = camera_gen_attention_mask.to(device=z_t.device, dtype=torch.bool)
            if mask.shape != (b, s):
                raise ValueError(f"camera_gen_attention_mask shape {tuple(mask.shape)} != {(b, s)}")
        tokens = tokens * mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        pos = self._camera_position_ids(
            batch_size=b,
            seq_len=s,
            device=z_t.device,
            patch_grid=patch_grid,
            frame_ids=frame_ids,
            fps=fps,
        )
        return tokens, mask, pos

    def _build_sky_generation(
        self,
        z_t: torch.Tensor,
        sky_gen_tokens: torch.Tensor | None,
        sky_gen_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b = int(z_t.shape[0])
        hidden_size = int(self.config.hidden_size)
        if sky_gen_tokens is None:
            empty_tokens = z_t.new_zeros((b, 0, hidden_size))
            empty_pos = torch.zeros((b, 0, 3), device=z_t.device, dtype=torch.long)
            return empty_tokens, None, empty_pos
        if sky_gen_tokens.ndim != 3 or int(sky_gen_tokens.shape[0]) != b:
            raise ValueError(f"sky_gen_tokens must be [B,K,C], got {tuple(sky_gen_tokens.shape)}")
        k = int(sky_gen_tokens.shape[1])
        if k > int(self.config.max_sky_tokens):
            raise ValueError(f"sky_gen_tokens K={k} exceeds max_sky_tokens={self.config.max_sky_tokens}")
        if int(sky_gen_tokens.shape[-1]) != int(self.config.sky_token_dim):
            raise ValueError(
                f"sky_gen_tokens dim {sky_gen_tokens.shape[-1]} != sky_token_dim {self.config.sky_token_dim}"
            )
        sky_dtype = self.sky_gen_proj.weight.dtype
        tokens = self.sky_gen_proj(self.sky_gen_norm(sky_gen_tokens).to(dtype=sky_dtype))
        tokens = tokens + self.sky_gen_modality_embed.to(device=tokens.device, dtype=tokens.dtype)
        if sky_gen_attention_mask is None:
            mask = torch.ones((b, k), device=z_t.device, dtype=torch.bool)
        else:
            mask = sky_gen_attention_mask.to(device=z_t.device, dtype=torch.bool)
            if mask.shape != (b, k):
                raise ValueError(f"sky_gen_attention_mask shape {tuple(mask.shape)} != {(b, k)}")
        tokens = tokens * mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        pos = self._sky_position_ids(batch_size=b, num_tokens=k, device=z_t.device)
        return tokens, mask, pos

    def _dilate_edit_mask(self, M_edit: torch.Tensor, patch_grid: tuple[int, int], radius: int) -> torch.Tensor:
        if radius <= 0:
            return M_edit.to(dtype=torch.bool)
        b, s, p, _ = M_edit.shape
        gh, gw = patch_grid
        grid = M_edit.reshape(b * s, gh, gw, 1).permute(0, 3, 1, 2).float()
        k = 2 * int(radius) + 1
        dilated = F.max_pool2d(grid, kernel_size=k, stride=1, padding=int(radius))
        return dilated.permute(0, 2, 3, 1).reshape(b, s, p, 1).gt(0.0)

    def _build_edit_control_condition(
        self,
        z_splat: torch.Tensor,
        scaffold_tok: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
        *,
        patch_grid: tuple[int, int],
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
        use_masked_edit: bool = True,
        control_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b, s, p, _ = z_splat.shape
        device = z_splat.device
        dtype = z_splat.dtype
        hidden_size = int(self.config.hidden_size)
        if not use_masked_edit:
            empty_tokens = torch.zeros((b, 0, hidden_size), device=device, dtype=dtype)
            empty_pos = torch.zeros((b, 0, 3), device=device, dtype=torch.long)
            return empty_tokens, None, empty_pos
        M_edit = (M_source.float() + M_dest.float()).clamp(0.0, 1.0)
        # Full-scene pretraining has no useful local edit-control prefix.
        if bool((M_edit > 0.999).all().item()) and bool((M_preserve <= 1e-6).all().item()):
            empty_tokens = torch.zeros((b, 0, hidden_size), device=device, dtype=dtype)
            empty_pos = torch.zeros((b, 0, 3), device=device, dtype=torch.long)
            return empty_tokens, None, empty_pos
        support = self._dilate_edit_mask(M_edit, patch_grid, radius=2)
        control_in = torch.cat([z_splat, scaffold_tok, M_preserve, M_source, M_dest], dim=-1)
        control_h = self.control_proj(self.control_norm(control_in))
        control_h = control_h + self.edit_control_modality_embed.to(device=device, dtype=control_h.dtype)
        max_per_frame = int(self.config.max_control_tokens_per_frame)
        max_total = int(self.config.max_control_tokens)
        target_pos = self._target_position_ids(
            batch_size=b,
            seq_len=s,
            num_patches=p,
            patch_grid=patch_grid,
            device=device,
            frame_ids=frame_ids,
            fps=fps,
        ).reshape(b, s, p, 3)
        sampled, sample_mask = self._uniform_sample_mask_indices(support[..., 0].reshape(b * s, p), max_per_frame)
        control_flat = control_h.reshape(b * s, p, hidden_size)
        pos_flat = target_pos.reshape(b * s, p, 3)
        sampled_tokens = control_flat.gather(1, sampled.unsqueeze(-1).expand(-1, -1, hidden_size))
        sampled_pos = pos_flat.gather(1, sampled.unsqueeze(-1).expand(-1, -1, 3))
        sampled_tokens = sampled_tokens.reshape(b, s * max_per_frame, hidden_size)
        sampled_pos = sampled_pos.reshape(b, s * max_per_frame, 3)
        sample_mask = sample_mask.reshape(b, s * max_per_frame)
        token_rows: list[torch.Tensor] = []
        pos_rows: list[torch.Tensor] = []
        for row in range(b):
            if control_drop_mask is not None and bool(
                control_drop_mask.to(device=device, dtype=torch.bool).view(b)[row].item()
            ):
                token_rows.append(torch.zeros((0, hidden_size), device=device, dtype=control_h.dtype))
                pos_rows.append(torch.zeros((0, 3), device=device, dtype=sampled_pos.dtype))
                continue
            row_mask = sample_mask[row]
            if bool(row_mask.any().item()):
                token_rows.append(sampled_tokens[row, row_mask][:max_total])
                pos_rows.append(sampled_pos[row, row_mask][:max_total])
            else:
                token_rows.append(torch.zeros((0, hidden_size), device=device, dtype=control_h.dtype))
                pos_rows.append(torch.zeros((0, 3), device=device, dtype=sampled_pos.dtype))
        return self._pad_sparse_condition(token_rows, pos_rows, device=device, dtype=control_h.dtype, max_tokens=max_total)

    def forward(
        self,
        z_t: torch.Tensor,
        sigma: torch.Tensor,
        z_splat: torch.Tensor,
        scaffold_tok: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
        F_asset_tokens: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        return_mid: bool = False,
        return_dict: bool = False,
        text_tokens: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        camera_pose_tokens: torch.Tensor | None = None,
        camera_attention_mask: torch.Tensor | None = None,
        camera_gen_tokens: torch.Tensor | None = None,
        camera_gen_attention_mask: torch.Tensor | None = None,
        sky_gen_tokens: torch.Tensor | None = None,
        sky_gen_attention_mask: torch.Tensor | None = None,
        return_base: bool = False,
        asset_condition_kind: Any = None,
        camera_condition_kind: Any = None,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
        use_masked_edit: bool = True,
        control_drop_mask: torch.Tensor | None = None,
        return_sky_mask: bool = False,
    ):
        self._validate_video_control_inputs(z_t, z_splat, scaffold_tok, M_preserve, M_source, M_dest)
        b, s, p, _ = z_t.shape
        patch_grid = _normalize_patch_grid(p, self.config.patch_grid)
        sigma = sigma.to(device=z_t.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.expand(b)
        if sigma.shape != (b,):
            raise ValueError(f"sigma must be shape [B], got {tuple(sigma.shape)}")

        video_flat = z_t.reshape(b, s * p, int(self.config.out_channels))
        video_state = self._build_video_state(sigma, M_preserve, M_source, M_dest)
        state_flat = video_state.reshape(b, s * p, int(self.config.video_state_dim))

        video_seq = self.video_embed(video_flat)
        video_seq = video_seq + self.video_state_proj(state_flat).to(dtype=video_seq.dtype)
        video_seq = video_seq + self.video_target_modality_embed.to(device=video_seq.device, dtype=video_seq.dtype)

        dec_x = self.decoder_video_embed(video_flat)
        dec_x = dec_x + self.decoder_state_proj(state_flat).to(dtype=dec_x.dtype)
        t_base, t_seq = self.t_embedder(sigma, return_base_embed=True)
        t_base = t_base.to(device=video_seq.device, dtype=video_seq.dtype)
        t_seq = t_seq.to(device=video_seq.device, dtype=video_seq.dtype)

        text_seq, text_mask = self._build_text_condition(z_t, text_tokens, text_attention_mask)
        camera_seq, camera_mask, camera_pos = self._build_camera_condition(
            z_t,
            camera_pose_tokens,
            camera_attention_mask,
            camera_condition_kind=camera_condition_kind,
            patch_grid=patch_grid,
            frame_ids=frame_ids,
            fps=fps,
        )
        camera_gen_seq, camera_gen_mask, camera_gen_pos = self._build_camera_generation(
            z_t,
            camera_gen_tokens,
            camera_gen_attention_mask,
            patch_grid=patch_grid,
            frame_ids=frame_ids,
            fps=fps,
        )
        sky_gen_seq, sky_gen_mask, sky_gen_pos = self._build_sky_generation(
            z_t,
            sky_gen_tokens,
            sky_gen_attention_mask,
        )
        asset_seq, asset_mask, asset_pos = self._build_sparse_asset_condition(
            F_asset_tokens,
            encoder_attention_mask,
            seq_len=s,
            num_patches=p,
            patch_grid=patch_grid,
            asset_condition_kind=asset_condition_kind,
            frame_ids=frame_ids,
            fps=fps,
        )
        control_seq, control_mask, control_pos = self._build_edit_control_condition(
            z_splat,
            scaffold_tok,
            M_preserve,
            M_source,
            M_dest,
            patch_grid=patch_grid,
            frame_ids=frame_ids,
            fps=fps,
            use_masked_edit=use_masked_edit,
            control_drop_mask=control_drop_mask,
        )
        if text_mask is None and text_seq.shape[1] > 0:
            text_mask = torch.ones((b, text_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if camera_mask is None and camera_seq.shape[1] > 0:
            camera_mask = torch.ones((b, camera_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if camera_gen_mask is None and camera_gen_seq.shape[1] > 0:
            camera_gen_mask = torch.ones((b, camera_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if sky_gen_mask is None and sky_gen_seq.shape[1] > 0:
            sky_gen_mask = torch.ones((b, sky_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if asset_mask is None and asset_seq.shape[1] > 0:
            asset_mask = torch.ones((b, asset_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if control_mask is None and control_seq.shape[1] > 0:
            control_mask = torch.ones((b, control_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if text_mask is None:
            text_mask = torch.ones((b, text_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if camera_mask is None:
            camera_mask = torch.ones((b, camera_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if camera_gen_mask is None:
            camera_gen_mask = torch.ones((b, camera_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if sky_gen_mask is None:
            sky_gen_mask = torch.ones((b, sky_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if asset_mask is None:
            asset_mask = torch.ones((b, asset_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if control_mask is None:
            control_mask = torch.ones((b, control_seq.shape[1]), device=z_t.device, dtype=torch.bool)

        target_pos = self._target_position_ids(
            batch_size=b,
            seq_len=s,
            num_patches=p,
            patch_grid=patch_grid,
            device=z_t.device,
            frame_ids=frame_ids,
            fps=fps,
        )
        text_pos = self._text_position_ids(b, text_seq.shape[1], z_t.device)
        timestep_pos = self._text_position_ids(b, t_seq.shape[1], z_t.device)
        timestep_mask = torch.ones((b, t_seq.shape[1]), device=z_t.device, dtype=torch.bool)

        cond_seq = torch.cat(
            [
                t_seq,
                text_seq.to(dtype=video_seq.dtype),
                camera_seq.to(dtype=video_seq.dtype),
                asset_seq.to(dtype=video_seq.dtype),
                control_seq.to(dtype=video_seq.dtype),
            ],
            dim=1,
        )
        cond_pos = torch.cat([timestep_pos, text_pos, camera_pos, asset_pos, control_pos], dim=1)
        cond_mask = torch.cat([timestep_mask, text_mask, camera_mask, asset_mask, control_mask], dim=1)
        gen_seq = torch.cat(
            [
                video_seq,
                camera_gen_seq.to(dtype=video_seq.dtype),
                sky_gen_seq.to(dtype=video_seq.dtype),
            ],
            dim=1,
        )
        gen_mask = torch.ones((b, s * p), device=z_t.device, dtype=torch.bool)
        if camera_gen_mask.shape[1] > 0:
            gen_mask = torch.cat([gen_mask, camera_gen_mask], dim=1)
        if sky_gen_mask.shape[1] > 0:
            gen_mask = torch.cat([gen_mask, sky_gen_mask], dim=1)
        gen_pos = torch.cat([target_pos, camera_gen_pos, sky_gen_pos], dim=1)
        full_seq = torch.cat([gen_seq, cond_seq], dim=1)
        full_pos = torch.cat([gen_pos, cond_pos], dim=1)
        full_mask = torch.cat([gen_mask, cond_mask], dim=1)
        full_attn_mask = self._key_padding_attention_mask(full_mask, full_seq.dtype)
        if full_attn_mask is not None:
            full_seq = self._apply_token_valid_mask(full_seq, full_mask)
        full_rope = VideoRoPE3D(
            seq_len=s,
            patch_grid=patch_grid,
            video_tokens=0,
            total_tokens=full_seq.shape[1],
            head_dim=int(self.config.attention_head_dim),
            device=full_seq.device,
            dtype=full_seq.dtype,
            theta=float(self.config.rope_theta),
            position_ids=full_pos,
            mrope_section=self.config.encoder_mrope_section,
        )

        mid_feat = None
        base_feat = None
        video_len = s * p
        camera_gen_len = int(camera_gen_seq.shape[1])
        sky_gen_len = int(sky_gen_seq.shape[1])
        camera_hidden = None
        sky_hidden = None
        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                full_seq = torch.utils.checkpoint.checkpoint(
                    block,
                    full_seq,
                    full_rope,
                    full_attn_mask,
                    use_reentrant=False,
                )
            else:
                full_seq = block(full_seq, full_rope, full_attn_mask)
            if full_attn_mask is not None:
                full_seq = self._apply_token_valid_mask(full_seq, full_mask)
            gen_seq = full_seq[:, :video_len]
            if return_mid and (idx + 1) == int(self.config.repa_layer_depth):
                mid_feat = gen_seq
            if (idx + 1) == int(self.config.base_model_depth):
                base_feat = gen_seq

        enc_video = gen_seq
        if camera_gen_len > 0:
            camera_hidden = full_seq[:, video_len : video_len + camera_gen_len]
        if sky_gen_len > 0:
            sky_start = video_len + camera_gen_len
            sky_hidden = full_seq[:, sky_start : sky_start + sky_gen_len]
        cond = self.s_projector(F.silu(enc_video + t_base.to(dtype=enc_video.dtype)))
        dec_rope = VideoRoPE3D(
            seq_len=s,
            patch_grid=patch_grid,
            video_tokens=0,
            total_tokens=s * p,
            head_dim=int(self.config.ddt_head_dim) // int(self.config.ddt_head_heads),
            device=dec_x.device,
            dtype=dec_x.dtype,
            theta=float(self.config.rope_theta),
            position_ids=target_pos,
            mrope_section=self.config.ddt_mrope_section,
        )
        for block in self.ddt_head:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                dec_x = torch.utils.checkpoint.checkpoint(block, dec_x, cond, dec_rope, None, use_reentrant=False)
            else:
                dec_x = block(dec_x, cond, dec_rope, None)

        out = self.final_layer(dec_x, cond).reshape(b, s, p, int(self.config.out_channels))
        model_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = out
        camera_out = None
        if camera_hidden is not None:
            camera_cond = F.silu(camera_hidden + t_base.to(dtype=camera_hidden.dtype))
            camera_out = self.camera_gen_decoder(camera_cond).reshape(b, s, int(self.config.camera_gen_dim))
        sky_out = None
        if sky_hidden is not None:
            sky_cond = F.silu(sky_hidden + t_base.to(dtype=sky_hidden.dtype))
            sky_out = self.sky_gen_decoder(sky_cond).reshape(b, sky_gen_len, int(self.config.sky_token_dim))
        sky_mask_logits = None
        sky_mask_refined_logits = None
        if return_sky_mask:
            sky_context = enc_video.new_zeros((b, 1, int(self.config.hidden_size)))
            if sky_hidden is not None and sky_gen_len > 0:
                sky_valid = sky_gen_mask[:, :sky_gen_len].to(device=sky_hidden.device, dtype=sky_hidden.dtype)
                denom = sky_valid.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
                sky_context = (sky_hidden * sky_valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
            sky_mask_cond = F.silu(enc_video + t_base.to(dtype=enc_video.dtype) + sky_context.to(dtype=enc_video.dtype))
            sky_mask_logits = self.sky_mask_decoder(sky_mask_cond).reshape(b, s, p, 1)
            sky_mask_refined_logits = self.sky_mask_refine_decoder(
                sky_mask_cond,
                patch_grid=patch_grid,
                seq_len=s,
                skip_tokens=base_feat,
            )
        if return_base:
            if base_feat is None:
                raise RuntimeError("base_feat was not captured; check base_model_depth and num_layers")
            base_cond = F.silu(base_feat + t_base.to(dtype=base_feat.dtype))
            base_out = self.base_final_layer(base_cond, base_cond).reshape(b, s, p, int(self.config.out_channels))
            model_out = (out, base_out)
        if return_dict:
            result: dict[str, torch.Tensor | None] = {"video": out}
            if return_base:
                result["video_base"] = model_out[1] if isinstance(model_out, tuple) else None
            if camera_out is not None:
                result["camera"] = camera_out
            if sky_out is not None:
                result["sky"] = sky_out
            result["sky_mask_logits"] = sky_mask_logits
            result["sky_mask_refined_logits"] = sky_mask_refined_logits
            if return_mid:
                if mid_feat is None:
                    mid_feat = enc_video
                result["mid_repa"] = self.repa_proj(mid_feat).reshape(b, s, p, int(self.config.out_channels))
            return result
        if return_mid:
            if mid_feat is None:
                mid_feat = enc_video
            return model_out, self.repa_proj(mid_feat).reshape(b, s, p, int(self.config.out_channels))
        if camera_out is not None:
            if sky_out is not None:
                return model_out, camera_out, sky_out
            return model_out, camera_out
        if sky_out is not None:
            return model_out, sky_out
        return model_out

    @torch.no_grad()
    def sample(
        self,
        z_splat: torch.Tensor,
        scaffold_tok: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
        F_asset_tokens: torch.Tensor,
        z_clean_for_blend: torch.Tensor | None = None,
        scheduler: Any | None = None,
        shift: float = 10.0,
        num_steps: int = 50,
        generator: torch.Generator | None = None,
        guidance_scale: float = 1.0,
        asset_control_guidance_scale: float = 1.0,
        text_tokens: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        camera_pose_tokens: torch.Tensor | None = None,
        camera_attention_mask: torch.Tensor | None = None,
        negative_text_tokens: torch.Tensor | None = None,
        negative_text_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        asset_condition_kind: Any = None,
        camera_condition_kind: Any = None,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            z_splat,
            scaffold_tok,
            M_preserve,
            M_source,
            M_dest,
            F_asset_tokens,
            z_clean_for_blend,
            scheduler,
            shift,
            num_steps,
            generator,
            guidance_scale,
            asset_control_guidance_scale,
            text_tokens,
            text_attention_mask,
            camera_pose_tokens,
            camera_attention_mask,
            negative_text_tokens,
            negative_text_attention_mask,
            encoder_attention_mask,
            asset_condition_kind,
            camera_condition_kind,
            frame_ids,
            fps,
            edit_mask,
        )
        raise RuntimeError(
            "WanSceneFlow.sample() was removed because it used a legacy CFG/scheduler path. "
            "Use train_scene_flow_pretrain.cfg_sample_pretrain_latents for pretrain sampling, "
            "train_scene_flow.cfg_sample_edit_latents for training-time T1 validation, or "
            "inference_scene_flow_validation.cfg_sample_edit_latents for offline T1 inference."
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path, map_location: str | torch.device = "cpu") -> "RAEVideoSceneFlow":
        load_path = Path(load_directory)
        config = json.loads((load_path / "config.json").read_text())
        model = cls(**config)
        state = torch.load(load_path / "pytorch_model.bin", map_location=map_location)
        model.load_state_dict(state, strict=True)
        return model


WanSceneFlow = RAEVideoSceneFlow
