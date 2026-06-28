"""Sliding-window helper utilities for SceneFlow validation/inference."""
from __future__ import annotations

import torch


def cosine_window(window_size: int, *, device=None, dtype=None) -> torch.Tensor:
    if int(window_size) <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if int(window_size) == 1:
        return torch.ones(1, device=device, dtype=dtype or torch.float32)
    x = torch.linspace(-1.0, 1.0, int(window_size), device=device, dtype=dtype or torch.float32)
    w = torch.cos(0.5 * torch.pi * x).square()
    return w.clamp_min(1e-3)


def window_starts(seq_len: int, window_size: int, overlap: int) -> list[int]:
    seq_len = int(seq_len)
    window_size = int(window_size)
    overlap = int(overlap)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    if seq_len <= window_size:
        return [0]
    stride = window_size - overlap
    starts = list(range(0, seq_len - window_size + 1, stride))
    last = seq_len - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def window_slices(seq_len: int, window_size: int, stride: int | None = None) -> list[tuple[int, int]]:
    """Return clamped [start, end) temporal windows covering the full sequence."""
    seq_len = int(seq_len)
    window_size = int(window_size)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    window_size = min(window_size, seq_len)
    if seq_len <= window_size:
        return [(0, seq_len)]
    stride_i = int(window_size if stride is None or int(stride) <= 0 else stride)
    stride_i = max(1, stride_i)
    starts = list(range(0, seq_len - window_size + 1, stride_i))
    last = seq_len - window_size
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + window_size) for start in starts]
