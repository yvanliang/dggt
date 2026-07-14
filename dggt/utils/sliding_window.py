"""Sliding-window helper utilities for SceneFlow validation/inference."""
from __future__ import annotations

import torch


OFFLINE_MAX_SINGLE_WINDOW = 10
OFFLINE_DEFAULT_OVERLAP = 3


def default_window_stride(window_size: int) -> int:
    """Use a three-frame overlap while keeping tiny windows valid."""
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    return max(1, window_size - min(OFFLINE_DEFAULT_OVERLAP, window_size - 1))


def resolve_offline_window(
    seq_len: int,
    requested_window: int | None,
    requested_stride: int | None,
    *,
    max_single_window: int = OFFLINE_MAX_SINGLE_WINDOW,
) -> tuple[int, int, bool]:
    """Resolve the bounded window policy shared by offline inference.

    A non-positive window means automatic selection, not disabling sliding.
    No offline SceneFlow forward receives more than ``max_single_window``
    frames; longer requests are covered by overlapping windows.
    """
    seq_len = int(seq_len)
    max_single_window = int(max_single_window)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if max_single_window <= 0:
        raise ValueError(
            f"max_single_window must be positive, got {max_single_window}"
        )

    raw_window = 0 if requested_window is None else int(requested_window)
    window = max_single_window if raw_window <= 0 else min(raw_window, max_single_window)
    window = max(1, window)
    sliding = seq_len > window

    raw_stride = 0 if requested_stride is None else int(requested_stride)
    if raw_stride < 0:
        raise ValueError(f"stride must be non-negative, got {raw_stride}")
    stride = default_window_stride(window) if raw_stride == 0 else raw_stride
    if sliding and stride >= window:
        raise ValueError(
            "offline sliding inference requires overlap: "
            f"frames={seq_len}, effective_window={window}, stride={stride}"
        )
    return window, stride, sliding


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
    if seq_len <= window_size:
        return [(0, seq_len)]
    stride_i = max(1, window_size // 2) if stride is None or int(stride) == 0 else int(stride)
    if stride_i < 0:
        raise ValueError(f"stride must be non-negative, got {stride_i}")
    if stride_i >= window_size:
        raise ValueError(
            f"sliding windows require overlap when seq_len > window_size: "
            f"expected 1 <= stride < {window_size}, got {stride_i}"
        )
    starts = list(range(0, seq_len - window_size + 1, stride_i))
    last = seq_len - window_size
    if starts[-1] != last:
        starts.append(last)
    windows = [(start, start + window_size) for start in starts]
    coverage = torch.zeros(seq_len, dtype=torch.long)
    for start, end in windows:
        coverage[start:end] += 1
    if bool((coverage <= 0).any()):
        missing = (coverage <= 0).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(f"sliding-window schedule left frames uncovered: {missing}")
    return windows


def cosine_coverage(
    seq_len: int,
    windows: list[tuple[int, int]],
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Precompute per-frame cosine coverage for a validated window schedule."""
    coverage = torch.zeros(int(seq_len), device=device, dtype=dtype or torch.float32)
    for start, end in windows:
        if start < 0 or end > int(seq_len) or end <= start:
            raise ValueError(f"invalid window [{start}, {end}) for sequence length {seq_len}")
        coverage[start:end] += cosine_window(end - start, device=device, dtype=coverage.dtype)
    if bool((coverage <= 0).any()):
        raise RuntimeError("cosine coverage must be positive for every global frame")
    return coverage


def scene_global_window_weight(
    start: int,
    end: int,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Weight a scene-global prediction so every global frame contributes equally."""
    local = cosine_window(end - start, device=coverage.device, dtype=coverage.dtype)
    return (local / coverage[start:end]).sum()
