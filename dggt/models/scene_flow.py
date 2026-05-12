"""Scene-flow rectified flow model built on diffusers WAN.

The model follows the T1 contract in ``docs/implement_scene_flow_plan.md``:
``z_clean`` is never a forward input, edit conditions are channel-concatenated
with the current latent, and asset tokens are consumed through WAN
cross-attention.
"""
from __future__ import annotations

import types
import typing
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_torch24_custom_op_annotation_patch() -> None:
    """Let torch 2.4 infer schemas for diffusers functions with postponed annotations.

    diffusers 0.37.1 defines some custom-op wrappers under
    ``from __future__ import annotations``. torch 2.4's schema inference sees
    those annotations as strings, while torch 2.5+ resolves them. This patch is
    local and idempotent; it resolves annotations before delegating to torch's
    original implementation.
    """

    try:
        import torch._custom_op.impl as custom_op_impl
    except Exception:
        return

    original = custom_op_impl.infer_schema
    if getattr(original, "_dggt_resolves_annotations", False):
        return

    none_type = type(None)

    def _normalize_annotation(annotation):
        origin = typing.get_origin(annotation)
        if origin in (typing.Union, types.UnionType):
            args = tuple(typing.get_args(annotation))
            if len(args) == 2 and none_type in args:
                inner = args[0] if args[1] is none_type else args[1]
                inner = _normalize_annotation(inner)
                if inner is torch.Tensor:
                    return typing.Optional[torch.Tensor]
                if inner is int:
                    return typing.Optional[int]
                if inner is float:
                    return typing.Optional[float]
                if inner is bool:
                    return typing.Optional[bool]
                if inner is str:
                    return typing.Optional[str]
        return annotation

    def _patched_infer_schema(prototype_function, mutates_args=()):
        annotations = getattr(prototype_function, "__annotations__", None)
        if annotations and any(isinstance(value, str) for value in annotations.values()):
            old_annotations = dict(annotations)
            try:
                hints = typing.get_type_hints(
                    prototype_function,
                    globalns=getattr(prototype_function, "__globals__", None),
                    localns=None,
                )
                prototype_function.__annotations__ = {
                    key: _normalize_annotation(value) for key, value in hints.items()
                }
                return original(prototype_function, mutates_args)
            finally:
                prototype_function.__annotations__ = old_annotations
        return original(prototype_function, mutates_args)

    _patched_infer_schema._dggt_resolves_annotations = True
    custom_op_impl.infer_schema = _patched_infer_schema


def _install_torch24_sdpa_enable_gqa_patch() -> None:
    """Drop ``enable_gqa=False`` for torch versions whose SDPA lacks the kwarg."""

    original = F.scaled_dot_product_attention
    if getattr(original, "_dggt_accepts_enable_gqa", False):
        return

    q = torch.empty(1, 1, 1, 1)
    try:
        original(query=q, key=q, value=q, enable_gqa=False)
        return
    except TypeError as exc:
        if "enable_gqa" not in str(exc):
            return

    def _patched_scaled_dot_product_attention(*args, enable_gqa: bool = False, **kwargs):
        if enable_gqa:
            raise NotImplementedError(
                "This PyTorch build does not support scaled_dot_product_attention(enable_gqa=True)."
            )
        return original(*args, **kwargs)

    _patched_scaled_dot_product_attention._dggt_accepts_enable_gqa = True
    F.scaled_dot_product_attention = _patched_scaled_dot_product_attention


_install_torch24_custom_op_annotation_patch()
_install_torch24_sdpa_enable_gqa_patch()

try:
    from diffusers.configuration_utils import register_to_config
    from diffusers.models.attention import FeedForward
    from diffusers.models.normalization import FP32LayerNorm
    from diffusers.models.transformers.transformer_wan import (
        WanAttention,
        WanAttnProcessor,
        WanTransformer3DModel,
        WanTransformerBlock,
    )
except Exception as exc:  # pragma: no cover - exercised only without deps.
    raise ImportError(
        "WanSceneFlow requires a compatible diffusers>=0.37.1 and PyTorch "
        "installation. Install the project requirements before importing "
        "dggt.models.scene_flow."
    ) from exc


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


class SceneFlowWanBlock(WanTransformerBlock):
    """WAN block with padding-mask support for asset-token cross-attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
            hidden_states
        )
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class DDTHeadBlock(nn.Module):
    """Wide-shallow denoising block (Wan-idiomatic DDT head).

    Sits after the main DiT trunk. Widens to ``dim_hid`` for richer
    channel mixing and cross-patch self-attention, then projects back
    to ``dim_in``. ``out_proj`` is zero-init and ``scale_shift_table``
    is zero-init so the block is exactly identity at step 0.

    Reuses Wan's building blocks (``FP32LayerNorm``, ``WanAttention``,
    ``FeedForward``) and AdaLN idiom (``scale_shift_table[1, 6, dim] +
    head_temb[B, 6, dim]``) so the head is structurally consistent with
    ``SceneFlowWanBlock`` — only width differs and cross-attention is
    omitted (asset KV was already consumed by the trunk). ``head_temb``
    is produced once by the parent model's shared ``head_time_proj`` and
    broadcast across all DDT blocks, mirroring trunk's ``timestep_proj``.

    Reference: Zheng et al. 2024 "Diffusion Transformers with
    Representation Autoencoders" (arxiv 2510.11690).
    """

    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim_hid % num_heads != 0:
            raise ValueError(f"dim_hid={dim_hid} not divisible by num_heads={num_heads}")
        head_dim = dim_hid // num_heads
        self.in_proj = nn.Linear(dim_in, dim_hid)
        # Wan-style norms (FP32 LN with affine=False; AdaLN-zero supplies
        # the time-driven shift/scale via scale_shift_table + head_temb).
        self.norm1 = FP32LayerNorm(dim_hid, eps=eps, elementwise_affine=False)
        self.attn = WanAttention(
            dim_hid,
            heads=num_heads,
            dim_head=head_dim,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )
        self.norm3 = FP32LayerNorm(dim_hid, eps=eps, elementwise_affine=False)
        # Diffusers FeedForward with gelu-approximate matches the trunk's FFN.
        self.ffn = FeedForward(dim_hid, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        # Per-block AdaLN bias (Wan idiom). Zero-init so the head is identity at start;
        # the time-driven additive comes from the parent's shared head_time_proj.
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, dim_hid))
        self.out_proj = nn.Linear(dim_hid, dim_in)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # hidden_states: [B, N, dim_in]; head_temb: [B, 6, dim_hid]; rotary_emb: trunk's
        residual = hidden_states
        h = self.in_proj(hidden_states)                                  # [B, N, dim_hid]

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + head_temb.float()
        ).chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1).unsqueeze(1)
        scale_msa = scale_msa.squeeze(1).unsqueeze(1)
        gate_msa = gate_msa.squeeze(1).unsqueeze(1)
        c_shift_msa = c_shift_msa.squeeze(1).unsqueeze(1)
        c_scale_msa = c_scale_msa.squeeze(1).unsqueeze(1)
        c_gate_msa = c_gate_msa.squeeze(1).unsqueeze(1)

        norm_h = (self.norm1(h.float()) * (1 + scale_msa) + shift_msa).type_as(h)
        h = (h.float() + self.attn(norm_h, None, None, rotary_emb).float() * gate_msa).type_as(h)

        norm_h = (self.norm3(h.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(h)
        h = (h.float() + self.ffn(norm_h).float() * c_gate_msa).type_as(h)
        return residual + self.out_proj(h)


class WanSceneFlow(WanTransformer3DModel):
    """Scene-edit rectified flow model in tokenizer latent space.

    Forward inputs are all in normalized latent space. ``z_clean`` is
    intentionally absent from the signature to avoid target leakage.
    """

    _no_split_modules = ["WanTransformerBlock", "SceneFlowWanBlock", "DDTHeadBlock"]
    _repeated_blocks = ["WanTransformerBlock", "SceneFlowWanBlock", "DDTHeadBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 1, 1),
        patch_grid: tuple[int, int] = (37, 37),
        num_attention_heads: int = 12,
        attention_head_dim: int = 128,
        in_channels: int = 2307,
        out_channels: int = 768,
        text_dim: int = 3072,
        freq_dim: int = 256,
        ffn_dim: int = 6144,
        num_layers: int = 14,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 128,
        pos_embed_seq_len: int | None = None,
        repa_block_frac: float = 1.0 / 3.0,
        null_kv_std: float = 0.02,
        ddt_head_depth: int = 2,
        ddt_head_dim: int = 2048,
        ddt_head_heads: int = 16,
        ddt_head_ffn_dim: int = 8192,
    ) -> None:
        if tuple(patch_size) != (1, 1, 1):
            raise ValueError("WanSceneFlow expects patch_size=(1, 1, 1) for DGGT token grids.")

        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        inner_dim = int(num_attention_heads) * int(attention_head_dim)
        self.blocks = nn.ModuleList(
            [
                SceneFlowWanBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self._init_patch_embedding_instruct_pix2pix(token_dim=out_channels, inner_dim=inner_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        if int(ddt_head_depth) > 0:
            self.ddt_head = nn.ModuleList(
                [
                    DDTHeadBlock(
                        dim_in=inner_dim,
                        dim_hid=int(ddt_head_dim),
                        num_heads=int(ddt_head_heads),
                        ffn_dim=int(ddt_head_ffn_dim),
                        eps=eps,
                    )
                    for _ in range(int(ddt_head_depth))
                ]
            )
            # Shared time projection for the DDT head (mirrors Wan trunk's
            # condition_embedder.time_proj). Zero-init so AdaLN modulation
            # starts at exactly 0 -> head is identity at step 0 alongside
            # the zero-init out_proj. One projection is broadcast to all
            # DDT blocks (avoids duplicating Linear(temb_dim, 6*dim_hid)
            # per block).
            self.head_time_proj = nn.Linear(inner_dim, 6 * int(ddt_head_dim))
            nn.init.zeros_(self.head_time_proj.weight)
            nn.init.zeros_(self.head_time_proj.bias)
        else:
            self.ddt_head = nn.ModuleList()
            self.head_time_proj = None

        self.register_buffer("mu_z", torch.zeros(out_channels))
        self.register_buffer("sigma_z", torch.ones(out_channels))
        self.null_kv = nn.Parameter(torch.randn(1, 1, text_dim) * float(null_kv_std))

        self.repa_block_idx = min(
            max(0, int(float(num_layers) * float(repa_block_frac))),
            max(0, int(num_layers) - 1),
        )
        self.repa_proj = nn.Sequential(
            nn.Linear(inner_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, out_channels),
        )

    @classmethod
    def from_scene_config(
        cls,
        *,
        bring_up: bool = False,
        patch_grid: tuple[int, int] = (37, 37),
        **kwargs: Any,
    ) -> "WanSceneFlow":
        """Build the documented bring-up or T1 model configuration."""
        if bring_up:
            defaults = {
                "num_attention_heads": 8,
                "attention_head_dim": 128,
                "num_layers": 8,
                "ffn_dim": 4096,
                "patch_grid": patch_grid,
            }
        else:
            # T1 config: 20-layer trunk (was 14) to close the gap with RAE
            # DiTDH-XL's 28-layer encoder. Same hidden=1536, ffn=6144.
            # Params delta ~ +90M (~14% of 620M base).
            defaults = {
                "num_attention_heads": 12,
                "attention_head_dim": 128,
                "num_layers": 20,
                "ffn_dim": 6144,
                "patch_grid": patch_grid,
            }
        defaults.update(kwargs)
        return cls(**defaults)

    def _init_patch_embedding_instruct_pix2pix(self, token_dim: int, inner_dim: int) -> None:
        nn.init.zeros_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
        for c in range(min(int(token_dim), int(inner_dim))):
            self.patch_embedding.weight.data[c, c, 0, 0, 0] = 1.0

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

    def _to_5d(
        self,
        z_t: torch.Tensor,
        z_splat: torch.Tensor,
        scaffold_tok: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.ndim != 4:
            raise ValueError(f"z_t must be [B,S,P,D], got {tuple(z_t.shape)}")
        if not (z_splat.shape == scaffold_tok.shape == z_t.shape):
            raise ValueError("z_t, z_splat and scaffold_tok must share shape [B,S,P,D]")
        for name, mask in (("M_preserve", M_preserve), ("M_source", M_source), ("M_dest", M_dest)):
            if mask.shape != z_t.shape[:-1] + (1,):
                raise ValueError(f"{name} must be [B,S,P,1], got {tuple(mask.shape)}")

        batch_size, seq_len, num_patches, _ = z_t.shape
        patch_h, patch_w = _normalize_patch_grid(num_patches, self.config.patch_grid)
        hidden_states = torch.cat([z_t, z_splat, scaffold_tok, M_preserve, M_source, M_dest], dim=-1)
        expected_channels = int(self.config.in_channels)
        if hidden_states.shape[-1] != expected_channels:
            raise ValueError(
                f"Packed input channels {hidden_states.shape[-1]} != model in_channels {expected_channels}"
            )
        return (
            hidden_states.view(batch_size, seq_len, patch_h, patch_w, expected_channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    @staticmethod
    def _from_5d(v_5d: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len, patch_h, patch_w = v_5d.shape
        return (
            v_5d.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(batch_size, seq_len, patch_h * patch_w, channels)
        )

    def _prepare_asset_kv(
        self,
        F_asset_tokens: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if F_asset_tokens.ndim != 3:
            raise ValueError(f"F_asset_tokens must be [B,N,C], got {tuple(F_asset_tokens.shape)}")
        batch_size, num_tokens, text_dim = F_asset_tokens.shape
        if text_dim != int(self.config.text_dim):
            raise ValueError(f"F_asset_tokens dim {text_dim} != text_dim {self.config.text_dim}")

        null = self.null_kv.to(device=F_asset_tokens.device, dtype=F_asset_tokens.dtype)

        # Edge case: no asset slots at all in the batch -> single null per row.
        if num_tokens == 0:
            return null.expand(batch_size, 1, -1), None

        # Static-graph anchor for DDP: route null_kv through the autograd graph
        # on every forward so `find_unused_parameters=False` doesn't trip when
        # the fast paths below don't otherwise reference it.
        null_anchor = (null.sum() * 0.0).to(F_asset_tokens.dtype)
        F_asset_tokens = F_asset_tokens + null_anchor

        # No mask provided -> assume all slots valid.
        if encoder_attention_mask is None:
            return F_asset_tokens, None

        mask = encoder_attention_mask.to(device=F_asset_tokens.device, dtype=torch.bool)
        if mask.shape != (batch_size, num_tokens):
            raise ValueError(
                f"encoder_attention_mask shape {tuple(mask.shape)} != {(batch_size, num_tokens)}"
            )

        # Fast path: every slot is valid -> no mask, no null injection.
        if bool(mask.all().item()):
            return F_asset_tokens, None

        empty_rows = ~mask.any(dim=1)  # [B] True for rows with zero valid tokens

        # Padding exists but no row is fully empty -> just pass mask through.
        if not bool(empty_rows.any().item()):
            return F_asset_tokens, mask

        # Inject null_kv into slot 0 of fully-empty rows ONLY.
        # Non-empty rows are untouched, so the kv length stays at num_tokens
        # (unlike the legacy implementation that appended a wasted null slot
        # to every row in the batch).
        first_slot = F_asset_tokens[:, :1, :]                        # [B, 1, C]
        null_first = null.expand(batch_size, 1, -1)                  # [B, 1, C]
        keep_real = (~empty_rows).view(batch_size, 1, 1)             # [B, 1, 1]
        new_first = torch.where(keep_real, first_slot, null_first)   # [B, 1, C]
        kv = torch.cat([new_first, F_asset_tokens[:, 1:, :]], dim=1)

        new_first_mask = mask[:, :1] | empty_rows.view(batch_size, 1)  # [B, 1]
        new_mask = torch.cat([new_first_mask, mask[:, 1:]], dim=1)
        return kv, new_mask

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
    ):
        del return_dict  # Kept for diffusers-style call compatibility.
        batch_size, seq_len, num_patches, _ = z_t.shape
        patch_h, patch_w = _normalize_patch_grid(num_patches, self.config.patch_grid)

        hidden_5d = self._to_5d(z_t, z_splat, scaffold_tok, M_preserve, M_source, M_dest)
        kv_raw, kv_mask = self._prepare_asset_kv(F_asset_tokens, encoder_attention_mask)

        sigma = sigma.to(device=z_t.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)
        if sigma.shape != (batch_size,):
            raise ValueError(f"sigma must be shape [B], got {tuple(sigma.shape)}")
        timestep = sigma * 1000.0

        rotary_emb = self.rope(hidden_5d)
        hidden_states = self.patch_embedding(hidden_5d)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep,
            kv_raw,
            None,
            timestep_seq_len=None,
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if kv_mask is None:
            cross_attn_mask = None
        else:
            invalid = (~kv_mask).to(dtype=hidden_states.dtype) * torch.finfo(hidden_states.dtype).min
            cross_attn_mask = invalid[:, None, None, :]

        mid_feat = None
        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    cross_attn_mask,
                    timestep_proj,
                    rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    cross_attn_mask,
                    timestep_proj,
                    rotary_emb,
                )
            if return_mid and idx == self.repa_block_idx:
                mid_feat = hidden_states

        # DDT head (Wan-idiomatic, RAE-inspired): wide-shallow refinement
        # before proj_out. ``head_temb`` is the shared time projection at
        # head width, mirroring trunk's ``timestep_proj``. Identity at init
        # (zero-init head_time_proj + zero-init scale_shift_table + zero-init
        # out_proj) preserves DiT-Zero behavior.
        if len(self.ddt_head) > 0 and self.head_time_proj is not None:
            head_temb = self.head_time_proj(temb).unflatten(1, (6, -1))
        else:
            head_temb = None
        for head_block in self.ddt_head:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    head_block,
                    hidden_states,
                    head_temb,
                    rotary_emb,
                )
            else:
                hidden_states = head_block(hidden_states, head_temb, rotary_emb)

        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        v_flat = self.proj_out(hidden_states)
        v = v_flat.view(batch_size, seq_len, patch_h, patch_w, -1).reshape(
            batch_size,
            seq_len,
            num_patches,
            -1,
        )

        if return_mid:
            if mid_feat is None:
                mid_feat = hidden_states
            repa_tokens = self.repa_proj(mid_feat).view(batch_size, seq_len, num_patches, -1)
            return v, repa_tokens
        return v

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
        num_steps: int = 15,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        if scheduler is None:
            # shift=13 matches RAE's dimension-dependent prescription for our
            # 25*37*768 per-frame latent (sqrt(710400/4096) ~= 13). Callers
            # that need a different schedule should pass their own scheduler.
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=13.0,
                invert_sigmas=True,
            )
        scheduler.set_timesteps(num_inference_steps=num_steps, device=z_splat.device)

        z = torch.empty_like(z_splat)
        z.normal_(generator=generator)
        batch_size = z_splat.shape[0]

        if not bool(getattr(scheduler.config, "invert_sigmas", False)):
            raise ValueError(
                "WanSceneFlow.sample requires a clean-progress scheduler "
                "(invert_sigmas=True). Pass FlowMatchEulerDiscreteScheduler(invert_sigmas=True)."
            )

        for timestep in scheduler.timesteps:
            sched_sigma = (timestep / scheduler.config.num_train_timesteps).to(device=z_splat.device)
            model_sigma = sched_sigma.expand(batch_size)
            v = self.forward(
                z,
                model_sigma,
                z_splat,
                scaffold_tok,
                M_preserve,
                M_source,
                M_dest,
                F_asset_tokens,
            )
            z = scheduler.step(
                model_output=v,
                timestep=timestep,
                sample=z,
                return_dict=False,
            )[0]

        if z_clean_for_blend is not None:
            z = M_preserve * z_clean_for_blend + (1.0 - M_preserve) * z
        return z
