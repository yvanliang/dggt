"""Window-bounded JointSceneTokenizer encode/decode for long clips.

The tokenizer contains temporal attention, so a calibrated 10-frame
pullback cannot be applied to a single 29-frame tokenizer call. Long clips are
therefore encoded/decoded in overlapping calibrated windows and blended in
token space with the same cosine coverage policy used by SceneFlow sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from dggt.utils.sliding_window import (
    cosine_coverage,
    cosine_window,
    default_window_stride,
    window_slices,
)


def _validate_window_len(window_len: int) -> int:
    value = int(window_len)
    if value <= 0:
        raise ValueError(f"tokenizer window_len must be positive, got {window_len}")
    return value


def _windows(seq_len: int, window_len: int, stride: int | None) -> list[tuple[int, int]]:
    window_len = _validate_window_len(window_len)
    stride_i = default_window_stride(window_len) if stride is None else int(stride)
    return window_slices(int(seq_len), window_len, stride_i)


def _blend_window_tensors(
    values: Sequence[tuple[int, int, torch.Tensor]],
    *,
    seq_len: int,
) -> torch.Tensor:
    if not values:
        raise ValueError("cannot blend an empty tokenizer window sequence")
    windows = [(start, end) for start, end, _ in values]
    reference = values[0][2]
    coverage = cosine_coverage(
        seq_len,
        windows,
        device=reference.device,
        dtype=torch.float32,
    )
    output_shape = (int(reference.shape[0]), int(seq_len), *reference.shape[2:])
    accumulated = torch.zeros(output_shape, device=reference.device, dtype=reference.dtype)
    for start, end, value in values:
        if int(value.shape[1]) != end - start:
            raise ValueError(
                f"tokenizer window [{start},{end}) returned S={value.shape[1]}"
            )
        weight = cosine_window(
            end - start, device=value.device, dtype=torch.float32
        ).to(dtype=value.dtype)
        weight = weight.view(1, end - start, *([1] * (value.ndim - 2)))
        indices = torch.arange(start, end, device=value.device, dtype=torch.long)
        accumulated = accumulated.index_add(1, indices, value * weight)
    denominator = coverage.to(dtype=accumulated.dtype).view(
        1, int(seq_len), *([1] * (accumulated.ndim - 2))
    )
    return accumulated / denominator


def encode_tokenizer_windowed(
    tokenizer: Any,
    patch_tokens_by_level: Sequence[torch.Tensor],
    *,
    patch_grid: tuple[int, int] | list[int],
    window_len: int,
    stride: int | None = None,
) -> torch.Tensor:
    if not patch_tokens_by_level:
        raise ValueError("patch_tokens_by_level cannot be empty")
    seq_len = int(patch_tokens_by_level[0].shape[1])
    if any(int(tokens.shape[1]) != seq_len for tokens in patch_tokens_by_level):
        raise ValueError("all tokenizer input levels must share the same sequence length")
    windows = _windows(seq_len, window_len, stride)
    encoded: list[tuple[int, int, torch.Tensor]] = []
    for start, end in windows:
        if end - start > int(window_len):
            raise AssertionError("tokenizer encode window exceeded calibrated window_len")
        value = tokenizer.encode(
            [tokens[:, start:end] for tokens in patch_tokens_by_level],
            patch_grid=patch_grid,
        )
        encoded.append((start, end, value))
    if len(encoded) == 1:
        return encoded[0][2]
    return _blend_window_tensors(encoded, seq_len=seq_len)


def decode_tokenizer_windowed(
    tokenizer: Any,
    latents: torch.Tensor,
    *,
    patch_grid: tuple[int, int] | list[int],
    window_len: int,
    stride: int | None = None,
) -> list[torch.Tensor]:
    if latents.ndim < 3:
        raise ValueError(f"tokenizer latents must include [B,S,...], got {latents.shape}")
    seq_len = int(latents.shape[1])
    windows = _windows(seq_len, window_len, stride)
    decoded_by_level: list[list[tuple[int, int, torch.Tensor]]] | None = None
    for start, end in windows:
        if end - start > int(window_len):
            raise AssertionError("tokenizer decode window exceeded calibrated window_len")
        decoded = list(tokenizer.decode(latents[:, start:end], patch_grid=patch_grid))
        if decoded_by_level is None:
            decoded_by_level = [[] for _ in decoded]
        if len(decoded) != len(decoded_by_level):
            raise RuntimeError("tokenizer.decode returned a different level count across windows")
        for level_values, value in zip(decoded_by_level, decoded, strict=True):
            level_values.append((start, end, value))
    assert decoded_by_level is not None
    if len(windows) == 1:
        return [level[0][2] for level in decoded_by_level]
    return [
        _blend_window_tensors(level_values, seq_len=seq_len)
        for level_values in decoded_by_level
    ]
