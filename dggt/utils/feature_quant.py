"""Int8 symmetric quantization for 4-level patch-token LUTs.

The token LUTs produced by the VGGT aggregator (`image_tokens_list`, shape
`[N, P, 4, C]` after level selection) dominate offline cache size. We store
them as `int8` with one `fp16` scale per `(level, frame)` pair — a 2× reduction
vs `fp16` that preserves cosine similarity ≥ 0.999 in practice.

The quantizer is symmetric (scale = |x|.amax() / 127) and clamps before cast,
which is numerically robust against outliers.

Shapes accepted:
* `[N_frames, P_patches, L_levels, C_channels]` (preferred)
* `[N_frames, L_levels, P_patches, C_channels]` (also supported)

The per-(level, frame) scale tensor is always `[N_frames, L_levels]`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizedTokens:
    """Container for quantized 4-level token LUT."""

    data: torch.Tensor          # int8, shape matches input
    scale: torch.Tensor         # fp16 or fp32, [N_frames, L_levels]
    layout: str                 # "NPLC" or "NLPC"

    def to(self, device: torch.device | str) -> "QuantizedTokens":
        return QuantizedTokens(
            data=self.data.to(device),
            scale=self.scale.to(device),
            layout=self.layout,
        )

    def save_dict(self) -> dict[str, torch.Tensor | str]:
        return {"data": self.data, "scale": self.scale, "layout": self.layout}

    @classmethod
    def load_dict(cls, d: dict[str, torch.Tensor | str]) -> "QuantizedTokens":
        return cls(data=d["data"], scale=d["scale"], layout=str(d["layout"]))


def quantize_tokens(
    tokens: torch.Tensor,
    layout: str = "NPLC",
    scale_dtype: torch.dtype = torch.float16,
) -> QuantizedTokens:
    """Symmetric int8 quantize per (level, frame) pair."""
    if layout not in ("NPLC", "NLPC"):
        raise ValueError(f"Unsupported layout '{layout}'; expected 'NPLC' or 'NLPC'")
    if tokens.dim() != 4:
        raise ValueError(f"Expected 4-D tensor, got shape {tuple(tokens.shape)}")
    x = tokens.detach()
    if not bool(torch.isfinite(x).all().item()):
        nan_count = int(torch.isnan(x).sum().item()) if x.is_floating_point() else 0
        inf_count = int(torch.isinf(x).sum().item()) if x.is_floating_point() else 0
        raise ValueError(
            f"Cannot quantize non-finite tokens: shape={tuple(x.shape)} "
            f"dtype={x.dtype} nan={nan_count} inf={inf_count}"
        )
    if layout == "NPLC":
        # [N, P, L, C] -> reduce over (P, C) to get per (N, L) scale
        absmax = x.abs().amax(dim=(1, 3))           # [N, L]
    else:
        # [N, L, P, C]
        absmax = x.abs().amax(dim=(2, 3))           # [N, L]
    scale = (absmax / 127.0).clamp_min(1e-8).to(scale_dtype)
    scale_expand = scale.to(x.dtype)
    if layout == "NPLC":
        inv = 1.0 / scale_expand.view(scale_expand.shape[0], 1, scale_expand.shape[1], 1)
    else:
        inv = 1.0 / scale_expand.view(scale_expand.shape[0], scale_expand.shape[1], 1, 1)
    q = (x * inv).round_().clamp_(-127.0, 127.0).to(torch.int8)
    return QuantizedTokens(data=q, scale=scale, layout=layout)


def dequantize_tokens(q: QuantizedTokens, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return fp tensor recovered from int8 + per-(level, frame) scale."""
    x = q.data.to(dtype)
    s = q.scale.to(dtype)
    if q.layout == "NPLC":
        s = s.view(s.shape[0], 1, s.shape[1], 1)
    else:
        s = s.view(s.shape[0], s.shape[1], 1, 1)
    return x * s


def roundtrip_cosine(
    tokens: torch.Tensor,
    layout: str = "NPLC",
) -> float:
    """Mean cosine similarity after a quant + dequant round trip (debug util)."""
    q = quantize_tokens(tokens, layout=layout)
    deq = dequantize_tokens(q, dtype=tokens.dtype)
    a = tokens.reshape(-1, tokens.shape[-1]).float()
    b = deq.reshape(-1, deq.shape[-1]).float()
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1).clamp_min(1e-8) * b.norm(dim=-1).clamp_min(1e-8)
    return float((num / den).mean().item())
