from __future__ import annotations

from typing import Iterable, Sequence

import torch


def split_special_and_patch(tokens: torch.Tensor, patch_start_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a token tensor into special tokens and patch tokens."""
    if tokens.ndim < 2:
        raise ValueError(f"Expected token tensor with at least 2 dims, got shape={tuple(tokens.shape)}")
    if patch_start_idx < 0 or patch_start_idx > tokens.shape[-2]:
        raise ValueError(
            f"patch_start_idx must be in [0, {tokens.shape[-2]}], got {patch_start_idx}"
        )
    return tokens[..., :patch_start_idx, :], tokens[..., patch_start_idx:, :]


def select_patch_pyramid(
    image_tokens_all: Sequence[torch.Tensor],
    levels: Sequence[int],
    patch_start_idx: int,
) -> list[torch.Tensor]:
    """Select patch tokens from the requested feature pyramid levels."""
    selected = []
    for level in levels:
        special_tokens, patch_tokens = split_special_and_patch(image_tokens_all[level], patch_start_idx)
        del special_tokens
        selected.append(patch_tokens)
    return selected


def reattach_special_tokens(
    template_tokens: Sequence[torch.Tensor],
    levels: Sequence[int],
    patch_start_idx: int,
    patch_tokens: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Re-attach special tokens from template levels to new patch tokens."""
    if len(levels) != len(patch_tokens):
        raise ValueError(f"levels ({len(levels)}) and patch_tokens ({len(patch_tokens)}) must match")

    outputs = []
    for level, new_patch_tokens in zip(levels, patch_tokens):
        special_tokens, _ = split_special_and_patch(template_tokens[level], patch_start_idx)
        if special_tokens.shape[:-2] != new_patch_tokens.shape[:-2]:
            raise ValueError(
                "Template special tokens and replacement patch tokens must match on leading dimensions: "
                f"{tuple(special_tokens.shape[:-2])} vs {tuple(new_patch_tokens.shape[:-2])}"
            )
        outputs.append(torch.cat([special_tokens, new_patch_tokens], dim=-2))
    return outputs


def replace_selected_levels(
    all_levels: Sequence[torch.Tensor],
    levels: Sequence[int],
    new_values: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Return a full token pyramid with the requested levels replaced."""
    if len(levels) != len(new_values):
        raise ValueError(f"levels ({len(levels)}) and new_values ({len(new_values)}) must match")

    updated_levels = list(all_levels)
    for level, value in zip(levels, new_values):
        updated_levels[level] = value
    return updated_levels


def split_joint_channels(
    joint_tokens: torch.Tensor | Sequence[torch.Tensor],
    dims: Iterable[int] = (1024, 1024, 1024),
) -> tuple[torch.Tensor | list[torch.Tensor], ...]:
    """Split joint tokens into channel groups for either one tensor or a tensor list."""
    split_dims = tuple(int(dim) for dim in dims)
    if len(split_dims) == 0:
        raise ValueError("dims must contain at least one split")

    if isinstance(joint_tokens, torch.Tensor):
        if joint_tokens.shape[-1] != sum(split_dims):
            raise ValueError(
                f"Last dimension ({joint_tokens.shape[-1]}) does not match requested split {split_dims}"
            )
        return joint_tokens.split(split_dims, dim=-1)

    split_per_level = [split_joint_channels(tokens, split_dims) for tokens in joint_tokens]
    outputs = []
    for split_idx in range(len(split_dims)):
        outputs.append([parts[split_idx] for parts in split_per_level])
    return tuple(outputs)
