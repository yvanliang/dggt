"""SceneFlow model for high-dimensional latent video generation.

The public training entry point is ``WanSceneFlow``. The trunk follows the RAEv2 T2I
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

from datasets.tools.hdmap_schema import (
    MAP_METRIC_RESERVED_ZERO_GROUPS,
    RASTER_ACTOR_CHANNELS,
    RASTER_CHANNEL_COUNT,
    RASTER_MAP_CHANNELS,
    RASTER_RESERVED_ZERO_CHANNELS,
)
from dggt.utils.camera_condition import CAMERA_CONDITION_REPRESENTATION, CAMERA_POSE_SUMMARY_DIM
from dggt.utils.actor_geometry_condition import (
    ActorGeometryCondition,
    LayoutMode,
    ProjectedActorGeometry,
)
from dggt.utils.appearance_binding_condition import (
    APPEARANCE_TOKEN_DIM,
    AppearanceBindingCondition,
    gather_appearance_geometry,
)
from dggt.utils.layout_condition import (
    LAYOUT_CONDITION_VERSION,
    MapMode,
    assert_neutral_raster_rows,
)
from dggt.utils.layout_raster import (
    RASTER_SCHEMA_HASH,
    STATIC_FAR_PLANE_M,
    build_map_metric_features,
    dequantize_layout_raster,
    reread_metric_geometry,
    scale_gradient,
)
from dggt.utils.scene_gauge import (
    GAUGE_MROPE_TEMPORAL_OFFSET,
    SCENE_GAUGE_DIM,
    SCENE_GAUGE_REPRESENTATION,
    SCENE_GAUGE_STATS_STD_FLOOR,
    SCENE_GAUGE_STATS_VERSION,
    SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN,
    SCENE_UNITS_PROFILE_GENERATED,
    SCENE_UNITS_PROFILES,
    denormalize_scene_gauge,
    normalize_scene_gauge,
)


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
DEFAULT_ENCODER_MROPE_SECTION = (14, 11, 11)
DEFAULT_ENCODER_ROPE_THETA = 50000.0
DEFAULT_DDT_ROPE_THETA = 10000.0
VIDEO_STATE_DIM = 6
ROPE_LAYOUT_VERSION = "a3_camera_center_spherical_sky15000"
SKY_MROPE_TEMPORAL_OFFSET = 15000
SKY_MROPE_SPHERE_RADIUS = 8.0
VIDEO_MROPE_TEMPORAL_LIMIT = SKY_MROPE_TEMPORAL_OFFSET
DEFAULT_ROPE_MAX_POSITION = 16384
CAMERA_ROPE_SPATIAL_MODE = "center"
ACTOR_METRIC_VELOCITY_REF_MPS = 1.0
ACTOR_METRIC_SPEED_MAX_MPS = 100.0
CURRENT_SKY_REPRESENTATION_VERSION = "rgb_patch_teacher_anchor_v5"
CURRENT_SKY_TOKEN_DIM = 192
CURRENT_SKY_ATLAS_HW = (128, 256)
CURRENT_SKY_GRID = (16, 32)
# v5 keeps every shape v4 had and changes only the units the sky token is
# expressed in: it is standardized per RGB channel so the flow target matches
# the scene latent's scale.  That makes a v4 checkpoint load-compatible by
# shape and wrong by meaning, so the version string -- not the tensor width --
# is what has to separate them.  ``sky_representation_version`` is a critical
# resume argument for exactly this reason.
SKY_REPRESENTATION_CONTRACTS = {
    CURRENT_SKY_REPRESENTATION_VERSION: (
        CURRENT_SKY_TOKEN_DIM,
        CURRENT_SKY_ATLAS_HW,
        CURRENT_SKY_GRID,
    ),
    "rgb_patch_teacher_anchor_v4": (192, (128, 256), (16, 32)),
    "rgb_patch_teacher_anchor_v3": (12, (32, 64), (16, 32)),
}


def _current_sky_contract_mismatches(config: dict[str, Any]) -> list[str]:
    expected = {
        "sky_representation_version": CURRENT_SKY_REPRESENTATION_VERSION,
        "sky_token_dim": CURRENT_SKY_TOKEN_DIM,
        "sky_atlas_hw": CURRENT_SKY_ATLAS_HW,
        "sky_grid": CURRENT_SKY_GRID,
    }
    actual = {name: config.get(name) for name in expected}
    mismatches: list[str] = []
    for name, required in expected.items():
        value = actual[name]
        if name in ("sky_atlas_hw", "sky_grid"):
            try:
                value = tuple(int(item) for item in value)
            except (TypeError, ValueError):
                pass
        elif name == "sky_token_dim":
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        if value != required:
            mismatches.append(
                f"{name}: checkpoint={actual[name]!r}, required={required!r}"
            )
    return mismatches


def _validate_video_frame_ids(frame_ids: torch.Tensor) -> None:
    if frame_ids.numel() == 0:
        return
    minimum = int(frame_ids.min().item())
    maximum = int(frame_ids.max().item())
    if minimum < 0 or maximum >= VIDEO_MROPE_TEMPORAL_LIMIT:
        raise ValueError(
            f"frame_ids must satisfy 0 <= id < {VIDEO_MROPE_TEMPORAL_LIMIT} "
            f"to stay below the sky RoPE modality margin; got range [{minimum}, {maximum}]"
        )


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


class Config(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight.to(device=x.device, dtype=x.dtype)


class ChannelScale(nn.Module):
    """Learned per-channel scale without sample-dependent normalization.

    This intentionally keeps the same ``weight`` state-dict surface as
    ``RMSNorm``.  It is used for low-dimensional physical states whose vector
    magnitude carries information and therefore must not be normalized away.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.to(device=x.device, dtype=x.dtype)


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


class LayoutRasterStem(nn.Module):
    """Independent early image-space reader for one layout channel group."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        *,
        num_modes: int,
        stem_dim: int = 96,
    ) -> None:
        super().__init__()
        stem_dim = int(stem_dim)
        self.conv_in = nn.Conv2d(int(in_channels), 64, kernel_size=3, padding=1)
        self.norm_in = _group_norm(64)
        self.down = nn.Conv2d(64, stem_dim, kernel_size=4, stride=4)
        self.norm_down = _group_norm(stem_dim)
        self.blocks = nn.Sequential(
            DepthwiseSeparableResBlock(stem_dim, stem_dim),
            DepthwiseSeparableResBlock(stem_dim, stem_dim),
        )
        self.mode_embedding = nn.Embedding(int(num_modes), stem_dim)
        self.zero_out = nn.Conv2d(stem_dim, int(hidden_size), kernel_size=1)
        nn.init.zeros_(self.zero_out.weight)
        nn.init.zeros_(self.zero_out.bias)

    def forward(self, raster: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        if raster.ndim != 5:
            raise ValueError(f"layout stem input must be [B,S,C,H,W], got {tuple(raster.shape)}")
        b, s, channels, height, width = (int(v) for v in raster.shape)
        if channels != int(self.conv_in.in_channels):
            raise ValueError(
                f"layout stem input has {channels} channels, expected {self.conv_in.in_channels}"
            )
        if tuple(mode.shape) != (b,):
            raise ValueError(f"layout mode must be [B], got {tuple(mode.shape)}")
        x = raster.reshape(b * s, channels, height, width)
        x = F.silu(self.norm_in(self.conv_in(x)))
        x = F.silu(self.norm_down(self.down(x)))
        x = self.blocks(x)
        mode_embed = self.mode_embedding(mode.to(dtype=torch.long))
        mode_embed = mode_embed[:, None].expand(b, s, -1).reshape(b * s, -1, 1, 1)
        x = self.zero_out(x + mode_embed.to(dtype=x.dtype))
        out_h, out_w = int(x.shape[-2]), int(x.shape[-1])
        return x.reshape(b, s, -1, out_h, out_w).permute(0, 1, 3, 4, 2).contiguous()


class MapMetricAdapter(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.SiLU(),
            nn.Linear(128, int(hidden_size)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if int(features.shape[-1]) != 20:
            raise ValueError(f"map metric feature dim must be 20, got {features.shape[-1]}")
        return self.net(features.to(dtype=self.net[0].weight.dtype))


class FullActorGaugeAdapter(nn.Module):
    """Per-instance metric reader and depth-aware soft z-buffer."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 128),
            nn.SiLU(),
            nn.Linear(128, int(hidden_size)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        log_z_d_patch: torch.Tensor,
        patch_weight: torch.Tensor,
        *,
        depth_tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or int(features.shape[-1]) != 27:
            raise ValueError(f"actor metric features must be [B,Kg,S,27], got {tuple(features.shape)}")
        b, kg, s = (int(v) for v in valid.shape)
        if tuple(features.shape[:3]) != (b, kg, s):
            raise ValueError("actor feature and validity axes differ")
        if patch_weight.ndim != 4 or tuple(patch_weight.shape[:3]) != (b, kg, s):
            raise ValueError("actor patch weights must be [B,Kg,S,P]")
        if tuple(log_z_d_patch.shape) != tuple(patch_weight.shape):
            raise ValueError("actor patch log-depth must match patch weights")
        depth_tau = float(depth_tau)
        if not math.isfinite(depth_tau) or depth_tau <= 0.0:
            raise ValueError(f"layout_depth_tau must be finite and positive, got {depth_tau}")
        embedding = self.net(features.to(dtype=self.net[0].weight.dtype))
        embedding = embedding * valid[..., None].to(dtype=embedding.dtype)
        support = (
            valid[..., None]
            & patch_weight.gt(0.0)
            & torch.isfinite(log_z_d_patch)
        )
        logits = -log_z_d_patch / depth_tau
        logits = torch.where(support, logits, torch.full_like(logits, -1.0e4))
        weights = torch.softmax(logits, dim=1)
        weights = weights * support.to(dtype=weights.dtype) * patch_weight
        context = torch.einsum("bksp,bksh->bsph", weights.to(dtype=embedding.dtype), embedding)
        return context, weights


class AppearanceContextAdapter(nn.Module):
    """Scatter pooled A values to G, then reuse G's actor z-buffer weights."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_size), 128),
            nn.SiLU(),
            nn.Linear(128, int(hidden_size)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        pooled_appearance: torch.Tensor,
        geometry_idx: torch.Tensor,
        binding_valid: torch.Tensor,
        actor_weights: torch.Tensor,
    ) -> torch.Tensor:
        if pooled_appearance.ndim != 3:
            raise ValueError("pooled appearance must be [B,Ka,H]")
        b, ka, hidden = (int(v) for v in pooled_appearance.shape)
        if tuple(geometry_idx.shape) != (b, ka) or tuple(binding_valid.shape) != (b, ka):
            raise ValueError("appearance binding axes differ from pooled values")
        if actor_weights.ndim != 4 or int(actor_weights.shape[0]) != b:
            raise ValueError("actor weights must be [B,Kg,S,P]")
        kg = int(actor_weights.shape[1])
        safe_idx = geometry_idx.clamp(min=0, max=kg - 1)
        values = self.net(pooled_appearance.to(dtype=self.net[0].weight.dtype))
        values = values * binding_valid[..., None].to(dtype=values.dtype)
        scattered = values.new_zeros((b, kg, hidden))
        scattered.scatter_add_(
            1,
            safe_idx[..., None].expand(-1, -1, hidden),
            values,
        )
        return torch.einsum(
            "bksp,bkh->bsph",
            actor_weights.to(dtype=scattered.dtype),
            scattered,
        )


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


def decode_scene_gauge(
    decoder: nn.Module,
    gauge_hidden: torch.Tensor,
    t_base: torch.Tensor,
    *,
    batch: int,
    gauge_dim: int,
) -> torch.Tensor:
    """Decode the scene-global gauge in float32, whatever the outer autocast is.

    The gauge is three numbers that every metric quantity in this model is
    derived through -- ``_clean_predicted_gauge``, the late metric reader, and
    the offline diagnostics -- so its readout resolution is the resolution of
    every metre downstream.  Left under bf16 autocast the final Linear returns
    eight significant bits, so the normalized gauge lands on a grid of spacing
    ``2**-7`` in the binade the validation scene sits in (~1.3 sigma).  At the
    measured ``gauge_std`` of 0.317 that is 0.0025 in log-metres, i.e. 0.25% of
    the scene scale -- about a sixth of the model's own error at step 8k, and
    coarse enough that a single-sample validation diagnostic sat bit-identical
    across 4000 training steps and read as "converged" when it was really
    "unresolvable".

    The decoder is one token wide against an ``S*P`` video stream, so float32
    here costs nothing measurable.
    """

    with torch.amp.autocast(device_type=gauge_hidden.device.type, enabled=False):
        gauge_cond = F.silu(gauge_hidden.float() + t_base.float())
        return decoder(gauge_cond).reshape(int(batch), 1, int(gauge_dim))


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
        qwen_dim: int = 1024,
        freq_dim: int = 256,
        ffn_dim: int | None = None,
        num_layers: int = 28,
        eps: float = 1e-6,
        rope_max_position: int = DEFAULT_ROPE_MAX_POSITION,
        repa_block_frac: float = 1.0 / 3.0,
        repa_layer_depth: int | None = None,
        ddt_head_depth: int = 2,
        ddt_head_dim: int = 2048,
        ddt_head_heads: int = 16,
        ddt_head_ffn_dim: int | None = None,
        num_timestep_tokens: int = 4,
        base_model_depth: int = 8,
        prediction_type: str = "x",
        architecture: str = "cosmos_lite",
        max_control_tokens_per_frame: int = 128,
        max_control_tokens: int = 1024,
        camera_cond_dim: int = CAMERA_POSE_SUMMARY_DIM,
        gauge_gen_dim: int = SCENE_GAUGE_DIM,
        scene_gauge_representation: str = SCENE_GAUGE_REPRESENTATION,
        scene_gauge_stats_version: str = SCENE_GAUGE_STATS_VERSION,
        scene_units_profile: str = SCENE_UNITS_PROFILE_GENERATED,
        camera_condition_representation: str = CAMERA_CONDITION_REPRESENTATION,
        mask_compositing_version: str = "soft_opacity_premultiplied_v2",
        layout_condition_version: str = LAYOUT_CONDITION_VERSION,
        layout_raster_channels: int = RASTER_CHANNEL_COUNT,
        layout_raster_hw: tuple[int, int] | list[int] = (100, 148),
        layout_map_channels: tuple[int, int] | list[int] = RASTER_MAP_CHANNELS,
        layout_actor_channels: tuple[int, int] | list[int] = RASTER_ACTOR_CHANNELS,
        layout_map_groups: int = 5,
        layout_stem_dim: int = 96,
        layout_max_actors: int = 96,
        layout_depth_tau: float = 0.5,
        layout_map_injection: bool = True,
        layout_actor_injection: bool = True,
        layout_map_metric_injection: bool = True,
        layout_actor_metric_injection: bool = True,
        appearance_context_injection: bool = True,
        layout_to_gauge_grad_scale: float = 1.0,
        raster_schema_hash: str = RASTER_SCHEMA_HASH,
        static_far_plane_m: float = STATIC_FAR_PLANE_M,
        sky_token_dim: int = CURRENT_SKY_TOKEN_DIM,
        sky_grid: tuple[int, int] | list[int] | None = CURRENT_SKY_GRID,
        max_sky_tokens: int = 512,
        video_state_dim: int = VIDEO_STATE_DIM,
        sky_mask_refine_scale: int = 4,
        sky_mask_refine_channels: int = 256,
        encoder_rope_theta: float | None = None,
        ddt_rope_theta: float | None = None,
        encoder_mrope_section: tuple[int, int, int] | list[int] | None = None,
        ddt_mrope_section: tuple[int, int, int] | list[int] | None = None,
        t_eps: float = 0.05,
        sky_representation_version: str | None = None,
        sky_atlas_hw: tuple[int, int] | list[int] = CURRENT_SKY_ATLAS_HW,
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
        if encoder_rope_theta is None:
            encoder_rope_theta_i = DEFAULT_ENCODER_ROPE_THETA
        else:
            encoder_rope_theta_i = float(encoder_rope_theta)
        if ddt_rope_theta is None:
            ddt_rope_theta_i = DEFAULT_DDT_ROPE_THETA
        else:
            ddt_rope_theta_i = float(ddt_rope_theta)
        for name, value in (
            ("encoder_rope_theta", encoder_rope_theta_i),
            ("ddt_rope_theta", ddt_rope_theta_i),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if prediction_type not in ("x", "v"):
            raise ValueError("prediction_type must be 'x' or 'v'")
        if str(camera_condition_representation) != CAMERA_CONDITION_REPRESENTATION:
            raise ValueError(
                f"camera_condition_representation must be {CAMERA_CONDITION_REPRESENTATION!r}, "
                f"got {camera_condition_representation!r}"
            )
        if int(gauge_gen_dim) != SCENE_GAUGE_DIM:
            raise ValueError(
                f"scene gauge generation requires gauge_gen_dim={SCENE_GAUGE_DIM}, got {gauge_gen_dim}"
            )
        if str(scene_gauge_representation) != SCENE_GAUGE_REPRESENTATION:
            raise ValueError(
                "unsupported scene gauge representation "
                f"{scene_gauge_representation!r}; expected {SCENE_GAUGE_REPRESENTATION!r}"
            )
        if str(scene_gauge_stats_version) != SCENE_GAUGE_STATS_VERSION:
            raise ValueError(
                "unsupported scene gauge stats version "
                f"{scene_gauge_stats_version!r}; expected {SCENE_GAUGE_STATS_VERSION!r}"
            )
        scene_units_profile = str(scene_units_profile)
        if scene_units_profile not in SCENE_UNITS_PROFILES:
            raise ValueError(
                f"scene_units_profile must be one of {SCENE_UNITS_PROFILES}, "
                f"got {scene_units_profile!r}"
            )
        layout_condition_version = str(layout_condition_version)
        if layout_condition_version not in (LAYOUT_CONDITION_VERSION, "none"):
            raise ValueError(
                f"layout_condition_version must be {LAYOUT_CONDITION_VERSION!r} or 'none', "
                f"got {layout_condition_version!r}"
            )
        layout_raster_hw_t = tuple(int(v) for v in layout_raster_hw)
        layout_map_channels_t = tuple(int(v) for v in layout_map_channels)
        layout_actor_channels_t = tuple(int(v) for v in layout_actor_channels)
        if int(layout_raster_channels) != RASTER_CHANNEL_COUNT:
            raise ValueError(
                f"layout_v2 requires exactly {RASTER_CHANNEL_COUNT} raster channels"
            )
        if layout_raster_hw_t != (100, 148):
            raise ValueError("layout_v2 requires layout_raster_hw=(100,148)")
        if (
            layout_map_channels_t != RASTER_MAP_CHANNELS
            or layout_actor_channels_t != RASTER_ACTOR_CHANNELS
        ):
            raise ValueError(
                "layout_v2 raster slices are frozen to the schema's map "
                f"{RASTER_MAP_CHANNELS[0]}:{RASTER_MAP_CHANNELS[1]} and actor "
                f"{RASTER_ACTOR_CHANNELS[0]}:{RASTER_ACTOR_CHANNELS[1]}"
            )
        if int(layout_map_groups) != 5:
            raise ValueError("layout_v2 requires five map metric groups")
        if int(layout_stem_dim) != 96:
            raise ValueError("layout_v2 requires layout_stem_dim=96")
        if int(layout_max_actors) <= 0:
            raise ValueError("layout_max_actors must be positive")
        if not math.isfinite(float(layout_depth_tau)) or float(layout_depth_tau) <= 0.0:
            raise ValueError("layout_depth_tau must be finite and positive")
        injection_flags = {
            "layout_map_injection": layout_map_injection,
            "layout_actor_injection": layout_actor_injection,
            "layout_map_metric_injection": layout_map_metric_injection,
            "layout_actor_metric_injection": layout_actor_metric_injection,
            "appearance_context_injection": appearance_context_injection,
        }
        if any(not isinstance(value, bool) for value in injection_flags.values()):
            raise TypeError("all five layout injection flags must be bool")
        if not math.isfinite(float(layout_to_gauge_grad_scale)) or not (
            0.0 <= float(layout_to_gauge_grad_scale) <= 1.0
        ):
            raise ValueError("layout_to_gauge_grad_scale must be in [0,1]")
        if raster_schema_hash != RASTER_SCHEMA_HASH:
            raise ValueError(
                f"raster_schema_hash must be the frozen identifier {RASTER_SCHEMA_HASH!r}, "
                f"got {raster_schema_hash!r}"
            )
        if float(static_far_plane_m) != STATIC_FAR_PLANE_M:
            raise ValueError(
                f"static_far_plane_m must be frozen to {STATIC_FAR_PLANE_M:g} metres"
            )
        sky_grid_t = None
        if sky_grid is not None:
            if len(sky_grid) != 2:
                raise ValueError(f"sky_grid must be (H,W) or None, got {sky_grid}")
            sky_grid_t = (int(sky_grid[0]), int(sky_grid[1]))
            if sky_grid_t[0] <= 0 or sky_grid_t[1] <= 0:
                raise ValueError(f"sky_grid entries must be positive, got {sky_grid_t}")
        # v4 raised the atlas from 32x64 to 128x256 while keeping the same 512
        # sky tokens: the patch grew from 2x2 to 8x8, so the token carries 192
        # channels instead of 12 and the attended sequence is unchanged.  v5
        # keeps those shapes and standardizes the token.  The generic
        # constructor still accepts the older contracts by name for checkpoint
        # inspection/conversion; production pretrain inference is deliberately
        # current-version-only and rejects them before constructing the model.
        # Width alone cannot separate v4 from v5, so an unnamed contract
        # resolves to the current version first.
        if sky_representation_version is None:
            sky_representation_version = next(
                (
                    name
                    for name, (dim, _, _) in SKY_REPRESENTATION_CONTRACTS.items()
                    if int(sky_token_dim) == dim
                ),
                "rgb_token_v1",
            )
        expected = SKY_REPRESENTATION_CONTRACTS.get(str(sky_representation_version))
        if expected is not None:
            want_dim, want_atlas, want_grid = expected
            if (
                int(sky_token_dim) != want_dim
                or tuple(int(v) for v in sky_atlas_hw) != want_atlas
                or sky_grid_t != want_grid
            ):
                raise ValueError(
                    f"{sky_representation_version} requires "
                    f"sky_token_dim={want_dim}, sky_atlas_hw={want_atlas}, "
                    f"and sky_grid={want_grid}"
                )
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
        if int(camera_cond_dim) != CAMERA_POSE_SUMMARY_DIM:
            raise ValueError(
                f"camera_cond_dim is frozen at {CAMERA_POSE_SUMMARY_DIM}, got {camera_cond_dim}"
            )
        required_rope_position = max(
            SKY_MROPE_TEMPORAL_OFFSET + SKY_MROPE_SPHERE_RADIUS,
            float(GAUGE_MROPE_TEMPORAL_OFFSET),
        )
        if float(rope_max_position) <= required_rope_position:
            raise ValueError(
                f"rope_max_position={rope_max_position} must exceed all scene-global RoPE positions; "
                f"required > {required_rope_position}"
            )
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
        self.config = Config(
            patch_size=tuple(patch_size),
            patch_grid=tuple(int(v) for v in patch_grid),
            num_attention_heads=int(num_attention_heads),
            attention_head_dim=int(attention_head_dim),
            hidden_size=hidden_size,
            in_channels=int(in_channels),
            out_channels=out_channels,
            qwen_dim=int(qwen_dim),
            freq_dim=int(freq_dim),
            ffn_dim=ffn_dim_i,
            num_layers=num_layers,
            eps=float(eps),
            rope_max_position=int(rope_max_position),
            repa_block_frac=float(repa_block_frac),
            repa_layer_depth=repa_layer_depth_i,
            ddt_head_depth=int(ddt_head_depth),
            ddt_head_dim=int(ddt_head_dim),
            ddt_head_heads=int(ddt_head_heads),
            ddt_head_ffn_dim=ddt_head_ffn_dim_i,
            num_timestep_tokens=int(num_timestep_tokens),
            base_model_depth=base_model_depth,
            prediction_type=str(prediction_type),
            architecture=str(architecture),
            max_control_tokens_per_frame=int(max_control_tokens_per_frame),
            max_control_tokens=int(max_control_tokens),
            camera_cond_dim=int(camera_cond_dim),
            gauge_gen_dim=int(gauge_gen_dim),
            scene_gauge_representation=str(scene_gauge_representation),
            scene_gauge_stats_version=str(scene_gauge_stats_version),
            scene_units_profile=scene_units_profile,
            camera_condition_representation=str(camera_condition_representation),
            mask_compositing_version=str(mask_compositing_version),
            layout_condition_version=layout_condition_version,
            layout_raster_channels=int(layout_raster_channels),
            layout_raster_hw=layout_raster_hw_t,
            layout_map_channels=layout_map_channels_t,
            layout_actor_channels=layout_actor_channels_t,
            layout_map_groups=int(layout_map_groups),
            layout_stem_dim=int(layout_stem_dim),
            layout_max_actors=int(layout_max_actors),
            layout_depth_tau=float(layout_depth_tau),
            **injection_flags,
            layout_to_gauge_grad_scale=float(layout_to_gauge_grad_scale),
            raster_schema_hash=str(raster_schema_hash),
            static_far_plane_m=float(static_far_plane_m),
            sky_token_dim=int(sky_token_dim),
            sky_grid=sky_grid_t,
            max_sky_tokens=int(max_sky_tokens),
            video_state_dim=video_state_dim,
            sky_mask_refine_scale=sky_mask_refine_scale,
            sky_mask_refine_channels=sky_mask_refine_channels,
            sky_representation_version=str(sky_representation_version),
            sky_atlas_hw=tuple(int(v) for v in sky_atlas_hw),
            rope_layout_version=ROPE_LAYOUT_VERSION,
            sky_rope_temporal_offset=SKY_MROPE_TEMPORAL_OFFSET,
            camera_rope_spatial_mode=CAMERA_ROPE_SPATIAL_MODE,
            encoder_rope_theta=encoder_rope_theta_i,
            ddt_rope_theta=ddt_rope_theta_i,
            encoder_mrope_section=encoder_mrope_section_i,
            ddt_mrope_section=ddt_mrope_section_i,
            t_eps=float(t_eps),
        )
        self.gradient_checkpointing = False
        self.gradient_checkpointing_mode = "off"

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
        self.appearance_norm = RMSNorm(APPEARANCE_TOKEN_DIM, eps=float(eps))
        self.appearance_proj = nn.Linear(APPEARANCE_TOKEN_DIM, hidden_size)
        self.appearance_summary_proj = nn.Linear(hidden_size, hidden_size)
        self.control_norm = RMSNorm(int(out_channels) * 2 + 3, eps=float(eps))
        self.control_proj = nn.Linear(int(out_channels) * 2 + 3, hidden_size)
        self.camera_norm = ChannelScale(int(camera_cond_dim))
        self.camera_proj = nn.Linear(int(camera_cond_dim), hidden_size)
        self.gauge_gen_norm = ChannelScale(int(gauge_gen_dim))
        self.gauge_gen_proj = nn.Linear(int(gauge_gen_dim), hidden_size)
        self.gauge_gen_decoder = nn.Sequential(
            RMSNorm(hidden_size, eps=float(eps)),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, int(gauge_gen_dim)),
        )
        if layout_condition_version == LAYOUT_CONDITION_VERSION:
            self.layout_map_stem: LayoutRasterStem | None = LayoutRasterStem(
                22,
                hidden_size,
                num_modes=len(MapMode),
                stem_dim=int(layout_stem_dim),
            )
            self.layout_actor_stem: LayoutRasterStem | None = LayoutRasterStem(
                11,
                hidden_size,
                num_modes=len(LayoutMode),
                stem_dim=int(layout_stem_dim),
            )
            self.map_metric_adapter: MapMetricAdapter | None = MapMetricAdapter(hidden_size)
            self.full_actor_gauge_adapter: FullActorGaugeAdapter | None = FullActorGaugeAdapter(hidden_size)
            self.appearance_context_adapter: AppearanceContextAdapter | None = AppearanceContextAdapter(hidden_size)
        else:
            self.layout_map_stem = None
            self.layout_actor_stem = None
            self.map_metric_adapter = None
            self.full_actor_gauge_adapter = None
            self.appearance_context_adapter = None
        # Sky tokens are deterministically packed RGB patch states. Their norm
        # encodes absolute brightness, so RMSNorm across the packed channels
        # would make proportional colors indistinguishable. Retain a
        # learned channel calibration, but never normalize each RGB token.
        self.sky_gen_norm = ChannelScale(int(sky_token_dim))
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
        self.appearance_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.appearance_summary_modality_embed = nn.Parameter(
            torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size))
        )
        self.sky_gen_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.gauge_gen_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.edit_control_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
        self.video_target_modality_embed = nn.Parameter(torch.randn(1, 1, hidden_size) / math.sqrt(float(hidden_size)))
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

        self.register_buffer("mu_z", torch.zeros(int(out_channels)))
        self.register_buffer("sigma_z", torch.ones(int(out_channels)))
        self.register_buffer("gauge_mean", torch.zeros(int(gauge_gen_dim)))
        self.register_buffer("gauge_std", torch.ones(int(gauge_gen_dim)))
        self.register_buffer("gauge_stats_valid", torch.tensor(False, dtype=torch.bool))
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
            self.appearance_proj,
            self.appearance_summary_proj,
            self.control_proj,
            self.camera_proj,
            self.sky_gen_proj,
            self.gauge_gen_proj,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        for module in self.sky_gen_decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.gauge_gen_decoder:
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
        sky_final = self.sky_gen_decoder[-1]
        if isinstance(sky_final, nn.Linear):
            nn.init.zeros_(sky_final.weight)
            nn.init.zeros_(sky_final.bias)
        gauge_final = self.gauge_gen_decoder[-1]
        if isinstance(gauge_final, nn.Linear):
            nn.init.zeros_(gauge_final.weight)
            nn.init.zeros_(gauge_final.bias)
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

    @classmethod
    def from_scene_config(
        cls,
        *,
        bring_up: bool = False,
        patch_grid: tuple[int, int] = (25, 37),
        **kwargs: Any,
    ) -> "RAEVideoSceneFlow":
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
                "encoder_mrope_section": DEFAULT_ENCODER_MROPE_SECTION,
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
        self.gradient_checkpointing_mode = "full"

    def enable_half_gradient_checkpointing(self) -> None:
        """Checkpoint alternating blocks in both the encoder and DDT head."""
        self.gradient_checkpointing = True
        self.gradient_checkpointing_mode = "half"

    def enable_three_quarter_gradient_checkpointing(self) -> None:
        """Checkpoint three of every four encoder blocks and every DDT block."""
        self.gradient_checkpointing = True
        self.gradient_checkpointing_mode = "three_quarter"

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.gradient_checkpointing_mode = "off"

    def _should_checkpoint_block(self, block_index: int, *, block_group: str) -> bool:
        if not self.gradient_checkpointing:
            return False
        mode = str(getattr(self, "gradient_checkpointing_mode", "full"))
        if block_group not in {"encoder", "ddt"}:
            raise ValueError(f"block_group must be 'encoder' or 'ddt', got {block_group!r}")
        if mode == "full":
            return True
        if mode == "half":
            return int(block_index) % 2 == 0
        if mode == "three_quarter":
            return block_group == "ddt" or int(block_index) % 4 != 3
        if mode == "off":
            return False
        raise RuntimeError(f"Unsupported gradient checkpointing mode: {mode!r}")

    def checkpointed_block_indices(
        self,
        block_count: int,
        *,
        block_group: str = "encoder",
    ) -> tuple[int, ...]:
        """Return the deterministic block selection used by the active mode."""
        count = int(block_count)
        if count < 0:
            raise ValueError(f"block_count must be non-negative, got {block_count}")
        return tuple(
            index
            for index in range(count)
            if self._should_checkpoint_block(index, block_group=block_group)
        )

    def set_latent_stats(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        if tuple(mu.shape) != tuple(self.mu_z.shape):
            raise ValueError(f"mu shape {tuple(mu.shape)} != {tuple(self.mu_z.shape)}")
        if tuple(sigma.shape) != tuple(self.sigma_z.shape):
            raise ValueError(f"sigma shape {tuple(sigma.shape)} != {tuple(self.sigma_z.shape)}")
        self.mu_z.copy_(mu.to(device=self.mu_z.device, dtype=self.mu_z.dtype))
        self.sigma_z.copy_(sigma.to(device=self.sigma_z.device, dtype=self.sigma_z.dtype).clamp_min(1.0e-4))

    def set_gauge_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = (SCENE_GAUGE_DIM,)
        mean_t = torch.as_tensor(mean)
        std_t = torch.as_tensor(std)
        if tuple(mean_t.shape) != expected or not bool(torch.isfinite(mean_t).all()):
            raise ValueError(f"gauge_mean must be finite with shape {expected}, got {tuple(mean_t.shape)}")
        if tuple(std_t.shape) != expected or not bool(torch.isfinite(std_t).all()):
            raise ValueError(f"gauge_std must be finite with shape {expected}, got {tuple(std_t.shape)}")
        if bool((std_t <= 0).any()):
            raise ValueError("gauge_std must be strictly positive")
        self.gauge_mean.copy_(mean_t.to(device=self.gauge_mean.device, dtype=self.gauge_mean.dtype))
        self.gauge_std.copy_(
            std_t.to(device=self.gauge_std.device, dtype=self.gauge_std.dtype).clamp_min(
                SCENE_GAUGE_STATS_STD_FLOOR
            )
        )
        self.gauge_stats_valid.fill_(True)

    def require_gauge_stats(self) -> None:
        if not bool(self.gauge_stats_valid.item()):
            raise RuntimeError(
                "Scene gauge generation requires per-channel statistics. Recompute them with "
                "`python tools/compute_pretrain_feature_stats.py ... --scene_gauge_path <table> "
                "--output <stats.pt>` and pass --feature_stats_path."
            )

    def normalize_gauge(self, gauge: torch.Tensor) -> torch.Tensor:
        self.require_gauge_stats()
        return normalize_scene_gauge(gauge, self.gauge_mean, self.gauge_std)

    def denormalize_gauge(self, normalized: torch.Tensor) -> torch.Tensor:
        self.require_gauge_stats()
        return denormalize_scene_gauge(normalized, self.gauge_mean, self.gauge_std)

    def uses_generated_scene_units(self) -> bool:
        """Whether this model integrates a per-scene gauge generation stream."""

        return str(self.config.scene_units_profile) == SCENE_UNITS_PROFILE_GENERATED

    def fixed_scene_gauge(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the physical training-mean gauge as ``[B,1,3]``."""

        batch_size = int(batch_size)
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        self.require_gauge_stats()
        return self.gauge_mean.to(device=device, dtype=dtype).view(
            1, 1, SCENE_GAUGE_DIM
        ).expand(batch_size, 1, SCENE_GAUGE_DIM)

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
        flow_edit_mask: torch.Tensor | None = None,
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
        if flow_edit_mask is None:
            noise_domain = edit
        else:
            if flow_edit_mask.shape != (b, s, p, 1):
                raise ValueError(
                    f"flow_edit_mask must match [B,S,P,1], got {tuple(flow_edit_mask.shape)}"
                )
            noise_domain = flow_edit_mask.to(device=M_preserve.device, dtype=torch.float32)
            rounded = noise_domain.round()
            if not bool(torch.allclose(noise_domain, rounded, atol=1e-6, rtol=0.0)):
                raise ValueError("flow_edit_mask must be binary")
            noise_domain = rounded
        sigma_eff = sigma.to(device=M_preserve.device, dtype=torch.float32).view(b, 1, 1, 1) * noise_domain
        state = torch.cat([preserve, source, dest, edit, keep, sigma_eff], dim=-1)
        if int(state.shape[-1]) != int(self.config.video_state_dim):
            raise RuntimeError(
                f"video state dim {state.shape[-1]} != configured {self.config.video_state_dim}"
            )
        return state

    @staticmethod
    def _balanced_frame_token_counts(
        valid_counts: torch.Tensor,
        *,
        max_per_frame: int,
        max_total: int,
    ) -> torch.Tensor:
        """Allocate a clip-level sparse-token budget without temporal prefix bias.

        The allocation is integer max-min fair across active frames. Frames whose
        support contains fewer tokens keep only their available tokens; the freed
        budget is redistributed to frames with remaining capacity. Any final
        sub-round remainder is spread uniformly over the eligible frame indices
        instead of being assigned to the earliest frames.
        """
        if valid_counts.ndim != 2:
            raise ValueError(f"valid_counts must be [B,S], got {tuple(valid_counts.shape)}")
        max_per_frame = int(max_per_frame)
        max_total = int(max_total)
        counts = valid_counts.to(dtype=torch.long).clamp_min(0)
        if max_per_frame <= 0 or max_total <= 0 or counts.shape[1] == 0:
            return torch.zeros_like(counts)

        capacity = counts.clamp_max(max_per_frame)
        target_total = capacity.sum(dim=1).clamp_max(max_total)

        # Find the largest common integer water level whose capped allocation
        # fits in the clip-level budget. The loop count depends only on the
        # static per-frame ceiling and does not synchronize device tensors.
        low = torch.zeros_like(target_total)
        high = torch.full_like(target_total, max_per_frame + 1)
        for _ in range((max_per_frame + 1).bit_length()):
            mid = torch.div(low + high, 2, rounding_mode="floor")
            used = torch.minimum(capacity, mid.unsqueeze(1)).sum(dim=1)
            fits = used <= target_total
            low = torch.where(fits, mid, low)
            high = torch.where(fits, high, mid)

        allocation = torch.minimum(capacity, low.unsqueeze(1))
        remaining = target_total - allocation.sum(dim=1)
        eligible = capacity > allocation
        eligible_count = eligible.sum(dim=1)

        # At the maximal water level, `remaining` is strictly smaller than the
        # number of eligible frames. Select exactly that many frames at evenly
        # spaced eligible ranks so the one-token remainder has no prefix bias.
        seq_len = int(counts.shape[1])
        remainder_slots = torch.arange(seq_len, device=counts.device, dtype=torch.long).view(1, seq_len)
        valid_remainder_slot = remainder_slots < remaining.unsqueeze(1)
        remaining_safe = remaining.clamp_min(1).unsqueeze(1)
        eligible_count_safe = eligible_count.clamp_min(1).unsqueeze(1)
        target_ranks = torch.div(
            (2 * remainder_slots + 1) * eligible_count_safe,
            2 * remaining_safe,
            rounding_mode="floor",
        ).clamp_max(eligible_count_safe - 1)
        eligible_ranks = eligible.to(dtype=torch.long).cumsum(dim=1) - 1
        receive_extra = (
            eligible_ranks.unsqueeze(2).eq(target_ranks.unsqueeze(1))
            & valid_remainder_slot.unsqueeze(1)
        ).any(dim=2)
        allocation = allocation + (receive_extra & eligible).to(dtype=allocation.dtype)
        return allocation

    @staticmethod
    def _uniform_sample_mask_indices_with_counts(
        valid_mask: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        max_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample a different deterministic token count for each row."""
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must be [N,P], got {tuple(valid_mask.shape)}")
        if sample_counts.shape != (valid_mask.shape[0],):
            raise ValueError(
                f"sample_counts must be [N]={valid_mask.shape[0]}, got {tuple(sample_counts.shape)}"
            )
        n, p = valid_mask.shape
        max_count = int(max_count)
        if max_count <= 0:
            idx = torch.zeros((n, 0), device=valid_mask.device, dtype=torch.long)
            keep = torch.zeros((n, 0), device=valid_mask.device, dtype=torch.bool)
            return idx, keep

        counts = valid_mask.sum(dim=1).to(dtype=torch.long)
        requested = sample_counts.to(device=valid_mask.device, dtype=torch.long).clamp_min(0)
        requested = torch.minimum(requested, counts).clamp_max(max_count)
        patch_idx = torch.arange(p, device=valid_mask.device, dtype=torch.long).view(1, p)
        sorted_idx = torch.where(valid_mask, patch_idx, torch.full_like(patch_idx, p)).sort(dim=1).values
        slots = torch.arange(max_count, device=valid_mask.device, dtype=torch.long).view(1, max_count)

        # For q>1, ranks span the complete valid support from first to last. A
        # single requested token uses the support midpoint rather than its first
        # patch. Masked slots may contain arbitrary safe indices and are ignored.
        denom = requested.sub(1).clamp_min(1).unsqueeze(1)
        numer = slots * counts.sub(1).clamp_min(0).unsqueeze(1)
        ranks = torch.div(numer + torch.div(denom, 2, rounding_mode="floor"), denom, rounding_mode="floor")
        midpoint = torch.div(counts.sub(1).clamp_min(0), 2, rounding_mode="floor").unsqueeze(1)
        ranks = torch.where(requested.unsqueeze(1).eq(1), midpoint, ranks)
        ranks = ranks.clamp_min(0).clamp_max(max(p - 1, 0))
        sampled = sorted_idx.gather(1, ranks).clamp_max(max(p - 1, 0))
        return sampled, slots < requested.unsqueeze(1)

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

    def _validate_layout_inputs(
        self,
        *,
        z_t: torch.Tensor,
        layout_raster: torch.Tensor | None,
        map_metric: torch.Tensor | None,
        actor_geometry: ActorGeometryCondition | None,
        projected_actor_geometry: ProjectedActorGeometry | None,
        appearance: AppearanceBindingCondition | None,
        map_mode: torch.Tensor | None,
        raster_schema_hash: str | None,
        appearance_class_id: torch.Tensor | None,
    ) -> None:
        values = (
            layout_raster,
            map_metric,
            actor_geometry,
            projected_actor_geometry,
            appearance,
            map_mode,
            raster_schema_hash,
        )
        if str(self.config.layout_condition_version) == "none":
            if any(value is not None for value in values) or appearance_class_id is not None:
                raise ValueError("layout_condition_version='none' forbids layout inputs")
            return
        names = (
            "layout_raster",
            "map_metric",
            "actor_geometry",
            "projected_actor_geometry",
            "appearance",
            "map_mode",
            "raster_schema_hash",
        )
        missing = [name for name, value in zip(names, values) if value is None]
        if missing:
            raise ValueError(f"layout_v2 requires all layout inputs; missing {', '.join(missing)}")
        if not isinstance(actor_geometry, ActorGeometryCondition):
            raise TypeError("actor_geometry must be ActorGeometryCondition")
        if not isinstance(projected_actor_geometry, ProjectedActorGeometry):
            raise TypeError("projected_actor_geometry must be ProjectedActorGeometry")
        if not isinstance(appearance, AppearanceBindingCondition):
            raise TypeError("appearance must be AppearanceBindingCondition")
        assert layout_raster is not None
        assert map_metric is not None
        assert map_mode is not None
        assert raster_schema_hash is not None
        b, s, p = (int(v) for v in z_t.shape[:3])
        expected_hw = tuple(int(v) for v in self.config.layout_raster_hw)
        expected_raster = (b, s, int(self.config.layout_raster_channels), *expected_hw)
        if tuple(layout_raster.shape) != expected_raster or layout_raster.dtype != torch.uint8:
            raise ValueError(
                f"layout_raster must be uint8 {expected_raster}, got "
                f"{layout_raster.dtype} {tuple(layout_raster.shape)}"
            )
        expected_metric = (b, s, p, int(self.config.layout_map_groups), 4)
        if tuple(map_metric.shape) != expected_metric or map_metric.dtype != torch.float32:
            raise ValueError(
                f"map_metric must be float32 {expected_metric}, got "
                f"{map_metric.dtype} {tuple(map_metric.shape)}"
            )
        if not bool(torch.isfinite(map_metric).all()):
            raise ValueError("map_metric contains NaN or Inf")
        map_valid = map_metric[..., 3]
        if not bool(((map_valid == 0.0) | (map_valid == 1.0)).all()):
            raise ValueError("map_metric valid field must be binary")
        if tuple(map_mode.shape) != (b,) or map_mode.dtype != torch.int8:
            raise ValueError("map_mode must be int8 [B]")
        if bool(((map_mode < int(MapMode.NULL)) | (map_mode > int(MapMode.PRESENT))).any()):
            raise ValueError("map_mode contains an unknown value")
        if raster_schema_hash != str(self.config.raster_schema_hash):
            raise ValueError(
                f"layout raster schema hash {raster_schema_hash!r} != "
                f"model hash {self.config.raster_schema_hash!r}"
            )
        if bool(layout_raster[:, :, RASTER_RESERVED_ZERO_CHANNELS].any()):
            raise ValueError("layout raster contains excluded static-map features")
        if bool(map_metric[..., MAP_METRIC_RESERVED_ZERO_GROUPS, :].any()):
            raise ValueError("map_metric contains excluded lane-centerline features")
        actor_geometry.validate(layout_max_actors=int(self.config.layout_max_actors))
        projected_actor_geometry.validate()
        appearance.validate_against_geometry(
            actor_geometry,
            appearance_class_id=appearance_class_id,
        )
        if actor_geometry.batch_size != b or actor_geometry.num_frames != s:
            raise ValueError("actor geometry does not match video batch/time axes")
        if (
            projected_actor_geometry.batch_size != b
            or projected_actor_geometry.num_frames != s
            or projected_actor_geometry.num_slots != actor_geometry.num_slots
        ):
            raise ValueError("projected actor geometry does not match G axes")
        if int(projected_actor_geometry.patch_weight.shape[-1]) != p:
            raise ValueError("projected actor patch weights do not match video patch count")
        if appearance.batch_size != b:
            raise ValueError("appearance does not match video batch axis")
        tensor_devices = {
            layout_raster.device,
            map_metric.device,
            map_mode.device,
            actor_geometry.slot_valid.device,
            projected_actor_geometry.valid.device,
            appearance.binding_valid.device,
            z_t.device,
        }
        if len(tensor_devices) != 1:
            raise ValueError("all layout inputs and z_t must share one device")
        g_valid = actor_geometry.slot_valid[..., None] & actor_geometry.track_valid
        if not torch.equal(projected_actor_geometry.valid, g_valid):
            raise ValueError(
                "projected validity must exactly match slot_valid & track_valid; "
                "partial/stale G caches are forbidden"
            )
        absent_map = (map_mode == int(MapMode.NULL)) | (map_mode == int(MapMode.EMPTY))
        if bool(map_metric[absent_map].any()):
            raise ValueError("M NULL/EMPTY rows must not carry map_metric values")
        if bool(absent_map.any()):
            assert_neutral_raster_rows(
                layout_raster,
                absent_map,
                channel_start=int(self.config.layout_map_channels[0]),
                channel_end=int(self.config.layout_map_channels[1]),
                label="M=NULL/EMPTY",
            )
        absent_actor = (actor_geometry.layout_mode == int(LayoutMode.NULL)) | (
            actor_geometry.layout_mode == int(LayoutMode.EMPTY)
        )
        if bool(absent_actor.any()):
            assert_neutral_raster_rows(
                layout_raster,
                absent_actor,
                channel_start=int(self.config.layout_actor_channels[0]),
                channel_end=int(self.config.layout_actor_channels[1]),
                label="G=NULL/EMPTY",
            )

    @staticmethod
    def _dequantize_layout_raster(raster: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_layout_raster(raster).to(dtype=dtype)

    @staticmethod
    def _map_metric_valid_fraction(
        map_metric: torch.Tensor | None,
        map_mode: torch.Tensor | None,
        *,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """Mean per-patch map-group validity over the rows that carry a map.

        The reserved zero group never holds a feature, so it stays out of the
        denominator (§7.5).  Rows whose ``map_mode`` is not ``PRESENT`` were
        neutralized to zero and are excluded as well; averaging them in would
        report task sampling as a collapsing curve.
        """

        if map_metric is None:
            return fallback.new_zeros(())
        valid_group_start = max(MAP_METRIC_RESERVED_ZERO_GROUPS) + 1
        valid = map_metric[..., valid_group_start:, 3].float()
        if map_mode is None:
            return valid.mean()
        present = (map_mode == int(MapMode.PRESENT)).to(device=valid.device)
        if not bool(present.any()):
            return valid.new_zeros(())
        return valid[present].mean()

    @staticmethod
    def _pack_masked_condition(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b, _n, hidden = (int(v) for v in tokens.shape)
        counts = mask.sum(dim=1, dtype=torch.long)
        max_count = int(counts.max().detach().cpu()) if b else 0
        if max_count == 0:
            return (
                tokens.new_zeros((b, 0, hidden)),
                None,
                positions.new_zeros((b, 0, 3)),
            )
        packed = tokens.new_zeros((b, max_count, hidden))
        packed_mask = torch.zeros((b, max_count), device=tokens.device, dtype=torch.bool)
        packed_pos = positions.new_zeros((b, max_count, 3))
        ranks = mask.long().cumsum(dim=1).sub(1)
        row = torch.arange(b, device=tokens.device)[:, None].expand_as(mask)
        packed[row[mask], ranks[mask]] = tokens[mask]
        packed_pos[row[mask], ranks[mask]] = positions[mask]
        packed_mask[row[mask], ranks[mask]] = True
        return packed, packed_mask, packed_pos

    def _build_appearance_condition(
        self,
        appearance: AppearanceBindingCondition,
        projected: ProjectedActorGeometry,
        *,
        seq_len: int,
        num_patches: int,
        patch_grid: tuple[int, int],
        frame_ids: torch.Tensor | None,
        fps: float | torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        appearance = appearance.canonicalized()
        gathered = gather_appearance_geometry(appearance, projected)
        b, ka, q, _ = appearance.appearance_tokens.shape
        if int(projected.num_frames) != int(seq_len):
            raise ValueError("projected actor window length differs from target video")
        token_dtype = self.appearance_proj.weight.dtype
        projected_values = self.appearance_proj(
            self.appearance_norm(appearance.appearance_tokens).to(dtype=token_dtype)
        )
        projected_values = projected_values + self.appearance_modality_embed.to(
            device=projected_values.device,
            dtype=projected_values.dtype,
        )
        appearance_mask = appearance.appearance_mask.to(dtype=projected_values.dtype)
        pooled = (
            projected_values * appearance_mask[..., None]
        ).sum(dim=2) / appearance_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        pooled = pooled * gathered.effective_binding_valid[..., None].to(dtype=pooled.dtype)
        summary = self.appearance_summary_proj(pooled) + self.appearance_summary_modality_embed.to(
            device=pooled.device,
            dtype=pooled.dtype,
        )
        token_values = projected_values[:, :, None].expand(-1, -1, seq_len, -1, -1)
        summary_values = summary[:, :, None, None].expand(-1, -1, seq_len, 1, -1)
        block_tokens = torch.cat((token_values, summary_values), dim=3)
        token_mask = gathered.token_attention_mask
        summary_mask = gathered.addr_ok[..., None]
        block_mask = torch.cat((token_mask, summary_mask), dim=3)
        temporal = self._target_position_ids(
            batch_size=b,
            seq_len=seq_len,
            num_patches=num_patches,
            patch_grid=patch_grid,
            device=projected_values.device,
            frame_ids=frame_ids,
            fps=fps,
        ).reshape(b, seq_len, num_patches, 3)[:, :, 0, 0].float()
        token_xy = gathered.token_patch_xy
        pooled_xy = gathered.pooled_patch_xy[..., None, :]
        block_xy = torch.cat((token_xy, pooled_xy), dim=3)
        block_pos = torch.stack(
            (
                temporal[:, None, :, None].expand(-1, ka, -1, q + 1),
                block_xy[..., 1],
                block_xy[..., 0],
            ),
            dim=-1,
        )
        block_tokens = block_tokens.reshape(b, ka * seq_len * (q + 1), -1)
        block_mask = block_mask.reshape(b, ka * seq_len * (q + 1))
        block_pos = block_pos.reshape(b, ka * seq_len * (q + 1), 3)
        block_tokens = block_tokens * block_mask[..., None].to(dtype=block_tokens.dtype)
        packed = self._pack_masked_condition(block_tokens, block_mask, block_pos)
        return (
            packed[0],
            packed[1],
            packed[2],
            pooled,
            gathered.effective_binding_valid,
            gathered.invalid_all_window_count,
        )

    def _clean_predicted_gauge(
        self,
        gauge_prediction: torch.Tensor,
        gauge_noisy_tokens: torch.Tensor,
        gauge_attention_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = int(sigma.shape[0])
        expected = (b, 1, int(self.config.gauge_gen_dim))
        if tuple(gauge_prediction.shape) != expected or tuple(gauge_noisy_tokens.shape) != expected:
            raise ValueError(f"gauge prediction/noisy tokens must both be {expected}")
        if tuple(gauge_attention_mask.shape) != (b, 1):
            raise ValueError(f"gauge attention mask must be {(b, 1)}")
        with torch.amp.autocast(device_type=gauge_prediction.device.type, enabled=False):
            prediction = gauge_prediction[:, 0].float()
            if str(self.config.prediction_type) == "x":
                clean_normalized = prediction
            elif str(self.config.prediction_type) == "v":
                clean_normalized = gauge_noisy_tokens[:, 0].float() - (
                    sigma.float().clamp_min(float(self.config.t_eps))[:, None] * prediction
                )
            else:
                raise RuntimeError(f"unsupported prediction_type={self.config.prediction_type!r}")
            physical = self.denormalize_gauge(clean_normalized).float()
            valid = (
                gauge_attention_mask[:, 0]
                & clean_normalized.isfinite().all(dim=-1)
                & physical.isfinite().all(dim=-1)
                & physical.abs().le(20.0).all(dim=-1)
            )
            physical = torch.where(valid[:, None], physical, torch.zeros_like(physical))
        return physical, valid

    def _build_map_metric_context(
        self,
        map_metric: torch.Tensor,
        gauge_physical: torch.Tensor,
        gauge_valid: torch.Tensor,
        *,
        grad_scale: float,
    ) -> torch.Tensor:
        if self.map_metric_adapter is None:
            raise RuntimeError("layout map metric adapter is not instantiated")
        features = build_map_metric_features(
            map_metric,
            gauge_physical,
            gauge_valid=gauge_valid,
            grad_scale=grad_scale,
        )
        return self.map_metric_adapter(features)

    def _build_actor_metric_context(
        self,
        actor_geometry: ActorGeometryCondition,
        projected: ProjectedActorGeometry,
        gauge_physical: torch.Tensor,
        gauge_valid: torch.Tensor,
        *,
        grad_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.full_actor_gauge_adapter is None:
            raise RuntimeError("full actor gauge adapter is not instantiated")
        with torch.amp.autocast(device_type=projected.valid.device.type, enabled=False):
            corners = projected.corners_camera.float()
            uv = projected.uv_corners.float()
            velocity = projected.velocity_camera.float()
            gauge_for_layout = scale_gradient(
                gauge_physical.float(),
                grad_scale,
            )
            base_valid = (
                actor_geometry.slot_valid[..., None]
                & actor_geometry.track_valid
                & projected.valid
                & projected.metric_support
                & gauge_valid[:, None, None]
            )
            finite_metric = corners.isfinite().all(dim=(-2, -1)) & velocity.isfinite().all(dim=-1)
            velocity_speed = torch.linalg.vector_norm(velocity, dim=-1)
            velocity_supported = velocity_speed.isfinite() & velocity_speed.le(
                ACTOR_METRIC_SPEED_MAX_MPS
            )
            # The 27-D actor embedding describes the full cuboid.  A cuboid
            # crossing the clip slab can still contribute early silhouette
            # and token support; metric_support explicitly records whether all
            # eight original corners are representable by this log-depth reader.
            valid = base_valid & finite_metric & velocity_supported
            placeholder = torch.zeros_like(corners)
            placeholder[..., 2] = 1.0
            safe_corners = torch.where(valid[..., None, None], corners, placeholder)
            projection_ok = uv.isfinite().all(dim=(-2, -1))
            safe_uv = torch.where(valid[..., None, None], uv, torch.zeros_like(uv))
            log_z_w = torch.log(safe_corners[..., 2])
            reread_corners = reread_metric_geometry(
                safe_uv,
                log_z_w,
                gauge_for_layout,
                valid[..., None].expand_as(log_z_w),
                gauge_valid=gauge_valid,
            )
            velocity_feature = torch.asinh(
                torch.where(valid[..., None], velocity, torch.zeros_like(velocity))
                / ACTOR_METRIC_VELOCITY_REF_MPS
            )
            valid = (
                valid
                & projection_ok
                & reread_corners.valid.all(dim=-1)
                & velocity_feature.isfinite().all(dim=-1)
            )
            feature = torch.cat(
                (reread_corners.features.reshape(*reread_corners.features.shape[:3], 24), velocity_feature),
                dim=-1,
            )
            feature = torch.where(valid[..., None], feature, torch.zeros_like(feature))
            patch_count = int(projected.patch_weight.shape[-1])
            grid_h, grid_w = _normalize_patch_grid(
                patch_count,
                tuple(int(value) for value in self.config.patch_grid),
            )
            patch_index = torch.arange(
                patch_count,
                device=projected.patch_weight.device,
                dtype=torch.float32,
            )
            patch_uv = torch.stack(
                (
                    (torch.remainder(patch_index, grid_w) + 0.5) / float(grid_w),
                    (torch.div(patch_index, grid_w, rounding_mode="floor") + 0.5)
                    / float(grid_h),
                ),
                dim=-1,
            ).view(1, 1, 1, patch_count, 2)
            patch_uv = patch_uv.expand(
                projected.batch_size,
                projected.num_slots,
                projected.num_frames,
                -1,
                -1,
            )
            patch_support = (
                projected.metric_support[..., None]
                & projected.patch_weight.gt(0.0)
                & valid[..., None]
            )
            reread_patch_depth = reread_metric_geometry(
                patch_uv,
                projected.log_z_patch.float(),
                gauge_for_layout,
                patch_support,
                gauge_valid=gauge_valid,
            )
            log_z_d_patch = torch.where(
                reread_patch_depth.valid,
                reread_patch_depth.log_z_d,
                torch.zeros_like(reread_patch_depth.log_z_d),
            )
            patch_weight = torch.where(
                reread_patch_depth.valid,
                projected.patch_weight.float(),
                torch.zeros_like(projected.patch_weight, dtype=torch.float32),
            )
        context, weights = self.full_actor_gauge_adapter(
            feature,
            valid,
            log_z_d_patch,
            patch_weight,
            depth_tau=float(self.config.layout_depth_tau),
        )
        return context, weights, valid

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
        _validate_video_frame_ids(frames)
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
            if not bool(torch.isfinite(fps_t).all()) or bool(fps_t.le(0.0).any()):
                raise ValueError("fps must be finite and strictly positive")
            temporal = frames.to(dtype=torch.float32) * (24.0 / fps_t).view(batch_size, 1)
            pos_dtype = torch.float32
        pos = torch.zeros((batch_size, seq_len, num_patches, 3), device=device, dtype=pos_dtype)
        pos[..., 0] = temporal[:, :, None].to(dtype=pos_dtype)
        pos[..., 1] = y.view(1, 1, num_patches)
        pos[..., 2] = x.view(1, 1, num_patches)
        return pos.reshape(batch_size, seq_len * num_patches, 3)

    def _text_position_ids(self, batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        # RAEv2 does not apply RoPE to text condition tokens. The Qwen text
        # hidden states already contain token order, so use zero-angle RoPE here.
        # Timestep tokens reuse this: both are global conditions with no spatial
        # coordinate. When frame_ids[0] == 0 the video frame's top-left patch
        # shares (0,0,0) with them. That is an accepted convention, not a
        # collision (design §6.5.1): mRoPE coordinates set an attention rotation
        # phase, not token identity, and the three groups still differ in
        # sequence position, content, source projection and modality embedding.
        # RoPE coordinates are not required to be globally unique.
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
        _validate_video_frame_ids(frames)
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
            if not bool(torch.isfinite(fps_t).all()) or bool(fps_t.le(0.0).any()):
                raise ValueError("fps must be finite and strictly positive")
            temporal = frames.to(dtype=torch.float32) * (24.0 / fps_t).view(batch_size, 1)
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
        """3D RoPE coordinates on the upper unit hemisphere.

        Encoding Cartesian directions makes the first and last longitude
        columns adjacent in RoPE space without a separate seam objective.
        """
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            return torch.zeros((batch_size, 0, 3), device=device, dtype=torch.float32)
        grid = getattr(self.config, "sky_grid", None)
        if grid is not None and int(grid[0]) * int(grid[1]) >= num_tokens:
            gh, gw = int(grid[0]), int(grid[1])
        else:
            gh = max(1, int(math.floor(num_tokens**0.5)))
            gw = max(1, int(math.ceil(num_tokens / float(gh))))
        idx = torch.arange(num_tokens, device=device, dtype=torch.long)
        y = torch.div(idx, gw, rounding_mode="floor").clamp_max(max(gh - 1, 0))
        x = (idx % gw).clamp_max(max(gw - 1, 0))
        elevation = (1.0 - (y.float() + 0.5) / float(gh)) * (math.pi * 0.5)
        azimuth = ((x.float() + 0.5) / float(gw)) * (math.pi * 2.0) - math.pi
        horizontal = torch.cos(elevation)
        direction = torch.stack(
            [
                horizontal * torch.cos(azimuth),
                -torch.sin(elevation),
                horizontal * torch.sin(azimuth),
            ],
            dim=-1,
        )
        center = float(self.config.sky_rope_temporal_offset)
        pos = center + SKY_MROPE_SPHERE_RADIUS * direction
        return pos.view(1, num_tokens, 3).expand(batch_size, -1, -1).contiguous()

    def _gauge_position_ids(self, *, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the single scene-global gauge coordinate outside video/sky ranges."""

        return torch.full(
            (int(batch_size), 1, 3),
            int(GAUGE_MROPE_TEMPORAL_OFFSET),
            device=device,
            dtype=torch.long,
        )

    def _build_camera_condition(
        self,
        z_t: torch.Tensor,
        camera_condition_tokens: torch.Tensor | None,
        camera_attention_mask: torch.Tensor | None,
        *,
        patch_grid: tuple[int, int] | None = None,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b, s = int(z_t.shape[0]), int(z_t.shape[1])
        if camera_condition_tokens is None:
            raise ValueError("camera_condition_tokens is mandatory for every layout task")
        if camera_condition_tokens.ndim != 3 or int(camera_condition_tokens.shape[0]) != b:
            raise ValueError(f"camera condition tokens must be [B,S,C], got {tuple(camera_condition_tokens.shape)}")
        if int(camera_condition_tokens.shape[1]) != s:
            raise ValueError(
                f"camera_condition_tokens must have one token per video frame: "
                f"got {camera_condition_tokens.shape[1]} for S={s}"
            )
        if int(camera_condition_tokens.shape[-1]) != int(self.config.camera_cond_dim):
            raise ValueError(
                f"camera_condition_tokens dim {camera_condition_tokens.shape[-1]} "
                f"!= camera_cond_dim {self.config.camera_cond_dim}"
            )
        if not bool(torch.isfinite(camera_condition_tokens).all()):
            raise ValueError("Waymo camera condition tokens contain non-finite values")
        camera_dtype = self.camera_proj.weight.dtype
        tokens = self.camera_proj(self.camera_norm(camera_condition_tokens).to(dtype=camera_dtype))
        tokens = tokens + self.camera_modality_embed.to(device=tokens.device, dtype=tokens.dtype)
        if camera_attention_mask is None:
            mask = torch.ones((b, s), device=z_t.device, dtype=torch.bool)
        else:
            mask = camera_attention_mask.to(device=z_t.device, dtype=torch.bool)
            if mask.shape != (b, s):
                raise ValueError(f"camera_attention_mask shape {tuple(mask.shape)} != {(b, s)}")
        if not bool(mask.all()):
            raise ValueError("camera_attention_mask must keep every requested camera frame")
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

    def _build_gauge_generation(
        self,
        z_t: torch.Tensor,
        gauge_gen_tokens: torch.Tensor | None,
        gauge_gen_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Embed one scene-global gauge token, independent of video length."""

        b = int(z_t.shape[0])
        hidden_size = int(self.config.hidden_size)
        if not self.uses_generated_scene_units():
            if gauge_gen_tokens is not None or gauge_gen_attention_mask is not None:
                raise ValueError(
                    "fixed_train_mean forbids caller-provided gauge generation "
                    "tokens and attention masks"
                )
            # Retain the generation-token shape without exposing a readable or
            # writable scene-specific channel.  The public forward masks this
            # placeholder after every encoder block.
            tokens = z_t.new_zeros((b, 1, hidden_size))
            mask = torch.zeros((b, 1), device=z_t.device, dtype=torch.bool)
            return tokens, mask, self._gauge_position_ids(
                batch_size=b, device=z_t.device
            )
        if gauge_gen_tokens is None:
            empty_tokens = z_t.new_zeros((b, 0, hidden_size))
            empty_pos = torch.zeros((b, 0, 3), device=z_t.device, dtype=torch.long)
            return empty_tokens, None, empty_pos
        if tuple(gauge_gen_tokens.shape) != (b, 1, int(self.config.gauge_gen_dim)):
            raise ValueError(
                "gauge_gen_tokens must be scene-global [B,1,3], got "
                f"{tuple(gauge_gen_tokens.shape)}"
            )
        if not bool(torch.isfinite(gauge_gen_tokens).all()):
            raise ValueError("scene gauge generation tokens contain non-finite values")
        gauge_dtype = self.gauge_gen_proj.weight.dtype
        tokens = self.gauge_gen_proj(
            self.gauge_gen_norm(gauge_gen_tokens).to(dtype=gauge_dtype)
        )
        tokens = tokens + self.gauge_gen_modality_embed.to(device=tokens.device, dtype=tokens.dtype)
        if gauge_gen_attention_mask is None:
            mask = torch.ones((b, 1), device=z_t.device, dtype=torch.bool)
        else:
            mask = gauge_gen_attention_mask.to(device=z_t.device, dtype=torch.bool)
            if tuple(mask.shape) != (b, 1):
                raise ValueError(f"gauge_gen_attention_mask must be [B,1], got {tuple(mask.shape)}")
        tokens = tokens * mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        return tokens, mask, self._gauge_position_ids(batch_size=b, device=z_t.device)

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
        support_flat = support[..., 0].reshape(b * s, p)
        valid_counts = support[..., 0].sum(dim=-1)
        frame_token_counts = self._balanced_frame_token_counts(
            valid_counts,
            max_per_frame=max_per_frame,
            max_total=max_total,
        )
        sampled, sample_mask = self._uniform_sample_mask_indices_with_counts(
            support_flat,
            frame_token_counts.reshape(b * s),
            max_count=max_per_frame,
        )
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
                # The balanced allocator already guarantees the clip-level
                # budget, so concatenation cannot discard later frames.
                token_rows.append(sampled_tokens[row, row_mask])
                pos_rows.append(sampled_pos[row, row_mask])
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
        layout_raster: torch.Tensor | None = None,
        map_metric: torch.Tensor | None = None,
        actor_geometry: ActorGeometryCondition | None = None,
        projected_actor_geometry: ProjectedActorGeometry | None = None,
        appearance: AppearanceBindingCondition | None = None,
        map_mode: torch.Tensor | None = None,
        raster_schema_hash: str | None = None,
        appearance_class_id: torch.Tensor | None = None,
        return_mid: bool = False,
        return_dict: bool = True,
        text_tokens: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        camera_condition_tokens: torch.Tensor | None = None,
        camera_attention_mask: torch.Tensor | None = None,
        sky_gen_tokens: torch.Tensor | None = None,
        sky_gen_attention_mask: torch.Tensor | None = None,
        gauge_gen_tokens: torch.Tensor | None = None,
        gauge_gen_attention_mask: torch.Tensor | None = None,
        return_base: bool = False,
        frame_ids: torch.Tensor | None = None,
        fps: float | torch.Tensor | None = None,
        use_masked_edit: bool = True,
        control_drop_mask: torch.Tensor | None = None,
        return_sky_mask: bool = False,
        return_layout_diagnostics: bool = False,
        layout_to_gauge_grad_scale: float | None = None,
        flow_edit_mask: torch.Tensor | None = None,
    ):
        if not isinstance(return_layout_diagnostics, bool):
            raise TypeError("return_layout_diagnostics must be a bool")
        if layout_to_gauge_grad_scale is None:
            layout_to_gauge_grad_scale = float(self.config.layout_to_gauge_grad_scale)
        layout_to_gauge_grad_scale = float(layout_to_gauge_grad_scale)
        if not math.isfinite(layout_to_gauge_grad_scale) or not (
            0.0 <= layout_to_gauge_grad_scale <= float(self.config.layout_to_gauge_grad_scale)
        ):
            raise ValueError(
                "runtime layout_to_gauge_grad_scale must lie in [0, configured maximum]"
            )
        if not self.uses_generated_scene_units():
            layout_to_gauge_grad_scale = 0.0
        self._validate_video_control_inputs(z_t, z_splat, scaffold_tok, M_preserve, M_source, M_dest)
        b, s, p, _ = z_t.shape
        if frame_ids is not None:
            frame_ids_check = torch.as_tensor(frame_ids, device=z_t.device)
            _validate_video_frame_ids(frame_ids_check)
        patch_grid = _normalize_patch_grid(p, self.config.patch_grid)
        self._validate_layout_inputs(
            z_t=z_t,
            layout_raster=layout_raster,
            map_metric=map_metric,
            actor_geometry=actor_geometry,
            projected_actor_geometry=projected_actor_geometry,
            appearance=appearance,
            map_mode=map_mode,
            raster_schema_hash=raster_schema_hash,
            appearance_class_id=appearance_class_id,
        )
        sigma = sigma.to(device=z_t.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.expand(b)
        if sigma.shape != (b,):
            raise ValueError(f"sigma must be shape [B], got {tuple(sigma.shape)}")

        video_flat = z_t.reshape(b, s * p, int(self.config.out_channels))
        video_state = self._build_video_state(
            sigma,
            M_preserve,
            M_source,
            M_dest,
            flow_edit_mask=flow_edit_mask,
        )
        state_flat = video_state.reshape(b, s * p, int(self.config.video_state_dim))

        video_seq = self.video_embed(video_flat)
        video_seq = video_seq + self.video_state_proj(state_flat).to(dtype=video_seq.dtype)
        video_seq = video_seq + self.video_target_modality_embed.to(device=video_seq.device, dtype=video_seq.dtype)
        layout_enabled = str(self.config.layout_condition_version) == LAYOUT_CONDITION_VERSION
        if layout_enabled:
            assert layout_raster is not None
            assert map_mode is not None
            assert actor_geometry is not None
            if self.layout_map_stem is None or self.layout_actor_stem is None:
                raise RuntimeError("layout_v2 early stems are not instantiated")
            stem_dtype = self.layout_map_stem.conv_in.weight.dtype
            raster_value = self._dequantize_layout_raster(layout_raster, dtype=stem_dtype)
            map_lo, map_hi = (int(v) for v in self.config.layout_map_channels)
            actor_lo, actor_hi = (int(v) for v in self.config.layout_actor_channels)
            if bool(self.config.layout_map_injection):
                map_early = self.layout_map_stem(
                    raster_value[:, :, map_lo:map_hi], map_mode
                )
                if tuple(map_early.shape[:4]) != (b, s, patch_grid[0], patch_grid[1]):
                    raise ValueError("map stem output grid does not match video patch grid")
                video_seq = video_seq + map_early.reshape(b, s * p, -1).to(dtype=video_seq.dtype)
            if bool(self.config.layout_actor_injection):
                actor_early = self.layout_actor_stem(
                    raster_value[:, :, actor_lo:actor_hi],
                    actor_geometry.layout_mode,
                )
                if tuple(actor_early.shape[:4]) != (b, s, patch_grid[0], patch_grid[1]):
                    raise ValueError("actor stem output grid does not match video patch grid")
                video_seq = video_seq + actor_early.reshape(b, s * p, -1).to(dtype=video_seq.dtype)

        dec_x = self.decoder_video_embed(video_flat)
        dec_x = dec_x + self.decoder_state_proj(state_flat).to(dtype=dec_x.dtype)
        t_base, t_seq = self.t_embedder(sigma, return_base_embed=True)
        t_base = t_base.to(device=video_seq.device, dtype=video_seq.dtype)
        t_seq = t_seq.to(device=video_seq.device, dtype=video_seq.dtype)

        text_seq, text_mask = self._build_text_condition(z_t, text_tokens, text_attention_mask)
        camera_seq, camera_mask, camera_pos = self._build_camera_condition(
            z_t,
            camera_condition_tokens,
            camera_attention_mask,
            patch_grid=patch_grid,
            frame_ids=frame_ids,
            fps=fps,
        )
        sky_gen_seq, sky_gen_mask, sky_gen_pos = self._build_sky_generation(
            z_t,
            sky_gen_tokens,
            sky_gen_attention_mask,
        )
        gauge_gen_seq, gauge_gen_mask, gauge_gen_pos = self._build_gauge_generation(
            z_t,
            gauge_gen_tokens,
            gauge_gen_attention_mask,
        )
        appearance_seq = z_t.new_zeros((b, 0, int(self.config.hidden_size)))
        appearance_mask = None
        appearance_pos = torch.zeros((b, 0, 3), device=z_t.device, dtype=torch.float32)
        pooled_appearance = z_t.new_zeros((b, 0, int(self.config.hidden_size)))
        effective_binding_valid = torch.zeros((b, 0), device=z_t.device, dtype=torch.bool)
        invalid_all_window_count = torch.zeros((b,), device=z_t.device, dtype=torch.int64)
        if layout_enabled:
            assert appearance is not None
            assert projected_actor_geometry is not None
            # Keep one canonical A ordering for both token routing and the
            # later scatter into G.  Canonicalizing only inside the helper
            # would pair sorted pooled values with stale geometry indices.
            appearance = appearance.canonicalized()
            (
                appearance_seq,
                appearance_mask,
                appearance_pos,
                pooled_appearance,
                effective_binding_valid,
                invalid_all_window_count,
            ) = self._build_appearance_condition(
                appearance,
                projected_actor_geometry,
                seq_len=s,
                num_patches=p,
                patch_grid=patch_grid,
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
        if sky_gen_mask is None and sky_gen_seq.shape[1] > 0:
            sky_gen_mask = torch.ones((b, sky_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if gauge_gen_mask is None and gauge_gen_seq.shape[1] > 0:
            gauge_gen_mask = torch.ones((b, gauge_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if appearance_mask is None and appearance_seq.shape[1] > 0:
            appearance_mask = torch.ones((b, appearance_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if control_mask is None and control_seq.shape[1] > 0:
            control_mask = torch.ones((b, control_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if text_mask is None:
            text_mask = torch.ones((b, text_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if camera_mask is None:
            camera_mask = torch.ones((b, camera_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if sky_gen_mask is None:
            sky_gen_mask = torch.ones((b, sky_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if gauge_gen_mask is None:
            gauge_gen_mask = torch.ones((b, gauge_gen_seq.shape[1]), device=z_t.device, dtype=torch.bool)
        if appearance_mask is None:
            appearance_mask = torch.ones((b, appearance_seq.shape[1]), device=z_t.device, dtype=torch.bool)
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
                appearance_seq.to(dtype=video_seq.dtype),
                control_seq.to(dtype=video_seq.dtype),
            ],
            dim=1,
        )
        cond_pos = torch.cat([timestep_pos, text_pos, camera_pos, appearance_pos, control_pos], dim=1)
        cond_mask = torch.cat([timestep_mask, text_mask, camera_mask, appearance_mask, control_mask], dim=1)
        gen_seq = torch.cat(
            [
                video_seq,
                sky_gen_seq.to(dtype=video_seq.dtype),
                gauge_gen_seq.to(dtype=video_seq.dtype),
            ],
            dim=1,
        )
        gen_mask = torch.ones((b, s * p), device=z_t.device, dtype=torch.bool)
        if sky_gen_mask.shape[1] > 0:
            gen_mask = torch.cat([gen_mask, sky_gen_mask], dim=1)
        if gauge_gen_mask.shape[1] > 0:
            gen_mask = torch.cat([gen_mask, gauge_gen_mask], dim=1)
        gen_pos = torch.cat([target_pos, sky_gen_pos, gauge_gen_pos], dim=1)
        full_seq = torch.cat([gen_seq, cond_seq], dim=1)
        full_pos = torch.cat([gen_pos, cond_pos], dim=1)
        if full_pos.numel():
            if not bool(torch.isfinite(full_pos).all()):
                raise ValueError("RoPE position ids contain NaN or Inf")
            max_position = float(full_pos.max().item())
            min_position = float(full_pos.min().item())
            if min_position < 0.0 or max_position >= float(self.config.rope_max_position):
                raise ValueError(
                    f"RoPE position ids must be in [0,{self.config.rope_max_position}); "
                    f"got range [{min_position},{max_position}]"
                )
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
            theta=float(self.config.encoder_rope_theta),
            position_ids=full_pos,
            mrope_section=self.config.encoder_mrope_section,
        )

        mid_feat = None
        base_feat = None
        video_len = s * p
        sky_gen_len = int(sky_gen_seq.shape[1])
        gauge_gen_len = int(gauge_gen_seq.shape[1])
        sky_hidden = None
        gauge_hidden = None
        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self._should_checkpoint_block(
                idx, block_group="encoder"
            ):
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
        if sky_gen_len > 0:
            sky_hidden = full_seq[:, video_len : video_len + sky_gen_len]
        if gauge_gen_len > 0:
            gauge_start = video_len + sky_gen_len
            gauge_hidden = full_seq[:, gauge_start : gauge_start + gauge_gen_len]
        gauge_out = None
        if gauge_hidden is not None:
            gauge_out = decode_scene_gauge(
                self.gauge_gen_decoder,
                gauge_hidden,
                t_base,
                batch=b,
                gauge_dim=int(self.config.gauge_gen_dim),
            )
            if not self.uses_generated_scene_units():
                gauge_out = gauge_out * 0.0
        gauge_context = enc_video.new_zeros((b, 1, int(self.config.hidden_size)))
        if gauge_hidden is not None and self.uses_generated_scene_units():
            gauge_context = gauge_hidden
        map_context = enc_video.new_zeros((b, s * p, int(self.config.hidden_size)))
        actor_context = enc_video.new_zeros((b, s * p, int(self.config.hidden_size)))
        appearance_context = enc_video.new_zeros((b, s * p, int(self.config.hidden_size)))
        actor_metric_valid = torch.zeros((b,), device=z_t.device, dtype=torch.int64)
        if layout_enabled:
            assert map_metric is not None
            assert actor_geometry is not None
            assert projected_actor_geometry is not None
            assert appearance is not None
            late_enabled = any(
                bool(value)
                for value in (
                    self.config.layout_map_metric_injection,
                    self.config.layout_actor_metric_injection,
                    self.config.appearance_context_injection,
                )
            )
            if late_enabled:
                if self.uses_generated_scene_units():
                    if gauge_out is None or gauge_gen_tokens is None:
                        raise ValueError("layout_v2 late readers require the gauge generation stream")
                    if gauge_gen_mask is None:
                        raise RuntimeError("gauge mask was not materialized")
                    gauge_physical, gauge_valid = self._clean_predicted_gauge(
                        gauge_out,
                        gauge_gen_tokens,
                        gauge_gen_mask,
                        sigma,
                    )
                else:
                    gauge_physical = self.fixed_scene_gauge(
                        b,
                        device=z_t.device,
                        dtype=torch.float32,
                    )[:, 0]
                    gauge_valid = torch.ones(
                        (b,), device=z_t.device, dtype=torch.bool
                    )
                if bool(self.config.layout_map_metric_injection):
                    map_context = self._build_map_metric_context(
                        map_metric,
                        gauge_physical,
                        gauge_valid,
                        grad_scale=layout_to_gauge_grad_scale,
                    ).reshape(b, s * p, -1).to(dtype=enc_video.dtype)
                actor_weights = None
                if bool(self.config.layout_actor_metric_injection) or bool(
                    self.config.appearance_context_injection
                ):
                    actor_metric_context, actor_weights, actor_valid = self._build_actor_metric_context(
                        actor_geometry,
                        projected_actor_geometry,
                        gauge_physical,
                        gauge_valid,
                        grad_scale=layout_to_gauge_grad_scale,
                    )
                    actor_metric_valid = actor_valid.sum(dim=(1, 2), dtype=torch.int64)
                    if bool(self.config.layout_actor_metric_injection):
                        actor_context = actor_metric_context.reshape(b, s * p, -1).to(
                            dtype=enc_video.dtype
                        )
                if bool(self.config.appearance_context_injection):
                    if actor_weights is None or self.appearance_context_adapter is None:
                        raise RuntimeError("appearance context requires shared actor z-buffer weights")
                    appearance_context = self.appearance_context_adapter(
                        pooled_appearance,
                        appearance.geometry_idx,
                        effective_binding_valid,
                        actor_weights,
                    ).reshape(b, s * p, -1).to(dtype=enc_video.dtype)
        cond = self.s_projector(
            F.silu(
                enc_video
                + t_base.to(dtype=enc_video.dtype)
                + gauge_context.to(dtype=enc_video.dtype)
                + map_context.to(dtype=enc_video.dtype)
                + actor_context.to(dtype=enc_video.dtype)
                + appearance_context.to(dtype=enc_video.dtype)
            )
        )
        dec_rope = VideoRoPE3D(
            seq_len=s,
            patch_grid=patch_grid,
            video_tokens=0,
            total_tokens=s * p,
            head_dim=int(self.config.ddt_head_dim) // int(self.config.ddt_head_heads),
            device=dec_x.device,
            dtype=dec_x.dtype,
            theta=float(self.config.ddt_rope_theta),
            position_ids=target_pos,
            mrope_section=self.config.ddt_mrope_section,
        )
        for idx, block in enumerate(self.ddt_head):
            if torch.is_grad_enabled() and self._should_checkpoint_block(
                idx, block_group="ddt"
            ):
                dec_x = torch.utils.checkpoint.checkpoint(block, dec_x, cond, dec_rope, None, use_reentrant=False)
            else:
                dec_x = block(dec_x, cond, dec_rope, None)

        out = self.final_layer(dec_x, cond).reshape(b, s, p, int(self.config.out_channels))
        model_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = out
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
            result: dict[str, Any] = {
                "video": out,
                "sky": sky_out,
                "gauge": gauge_out,
            }
            if return_base:
                result["video_base"] = model_out[1] if isinstance(model_out, tuple) else None
            if return_sky_mask:
                result["sky_mask_logits"] = sky_mask_logits
                result["sky_mask_refined_logits"] = sky_mask_refined_logits
            if return_layout_diagnostics:
                # §7.5 asks for a stable curve, so restrict it to the rows that
                # actually carry a map.  A TC row is neutralized to all zeros;
                # letting those into the denominator makes the curve swing
                # between zero and the true value at batch size one, which reads
                # as a collapse rather than as task sampling.
                map_metric_valid_fraction = self._map_metric_valid_fraction(
                    map_metric, map_mode, fallback=out
                )
                result["actor_alignment_diagnostics"] = {
                    "appearance_invalid_all_window_count": invalid_all_window_count,
                    "actor_metric_valid_count": actor_metric_valid,
                    "map_residual_rms": map_context.detach().float().square().mean().sqrt(),
                    "actor_residual_rms": actor_context.detach().float().square().mean().sqrt(),
                    "map_metric_valid_fraction": map_metric_valid_fraction.detach(),
                }
            if return_mid:
                if mid_feat is None:
                    mid_feat = enc_video
                result["mid_repa"] = self.repa_proj(mid_feat).reshape(b, s, p, int(self.config.out_channels))
            return result
        if return_mid:
            if mid_feat is None:
                mid_feat = enc_video
            return model_out, self.repa_proj(mid_feat).reshape(b, s, p, int(self.config.out_channels))
        if sky_out is not None:
            if gauge_out is not None:
                return model_out, sky_out, gauge_out
            return model_out, sky_out
        if gauge_out is not None:
            return model_out, gauge_out
        return model_out

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path, map_location: str | torch.device = "cpu") -> "RAEVideoSceneFlow":
        load_path = Path(load_directory)
        config = json.loads((load_path / "config.json").read_text())
        sky_mismatches = _current_sky_contract_mismatches(config)
        if sky_mismatches:
            raise ValueError(
                f"{load_path} contains unsupported old sky checkpoint weights; "
                f"RAEVideoSceneFlow.from_pretrained requires the complete "
                f"{CURRENT_SKY_REPRESENTATION_VERSION} contract ("
                + "; ".join(sky_mismatches)
                + "). Use the raw RAEVideoSceneFlow constructor explicitly for "
                "checkpoint inspection or conversion; standard weight loading does "
                "not accept v3."
            )
        # Checkpoints predating the explicit profile are Full/generated only.
        config.setdefault("scene_units_profile", SCENE_UNITS_PROFILE_GENERATED)
        for derived_name in (
            "hidden_size",
            "rope_layout_version",
            "sky_rope_temporal_offset",
            "camera_rope_spatial_mode",
        ):
            config.pop(derived_name, None)
        model = cls(**config)
        state = torch.load(load_path / "pytorch_model.bin", map_location=map_location)
        model.load_state_dict(state, strict=True)
        return model


WanSceneFlow = RAEVideoSceneFlow
