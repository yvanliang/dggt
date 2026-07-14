from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


def _format_keys(keys: list[str], limit: int = 5) -> str:
    if not keys:
        return "[]"
    suffix = "" if len(keys) <= limit else ", ..."
    return f"[{', '.join(keys[:limit])}{suffix}]"


def extract_scene_tokenizer_state_dict(
    payload: Any,
    *,
    source: str | Path,
) -> dict[str, torch.Tensor]:
    """Extract tokenizer-only weights from a tokenizer or full-DGGT checkpoint.

    Supported checkpoint layouts are the tokenizer trainer's
    ``{"scene_tokenizer": ...}`` payload and full model payloads under
    ``state_dict``/``model`` (including DDP's leading ``module.`` prefix).
    A full model checkpoint that does not contain ``scene_tokenizer.*`` is
    deliberately returned as a non-matching state and rejected by the strict
    validator below; it must never be mistaken for valid tokenizer weights.
    """

    state: Any = payload
    tokenizer_only_payload = False
    if isinstance(payload, Mapping):
        if isinstance(payload.get("scene_tokenizer"), Mapping):
            state = payload["scene_tokenizer"]
            tokenizer_only_payload = True
        elif isinstance(payload.get("state_dict"), Mapping):
            state = payload["state_dict"]
        elif isinstance(payload.get("model"), Mapping):
            state = payload["model"]

    if not isinstance(state, Mapping):
        raise ValueError(f"Unsupported scene tokenizer checkpoint format: {source}")

    cleaned: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            continue
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        cleaned[key] = value

    prefixed = {
        key[len("scene_tokenizer.") :]: value
        for key, value in cleaned.items()
        if key.startswith("scene_tokenizer.")
    }
    if prefixed:
        return prefixed
    if tokenizer_only_payload:
        return cleaned
    return cleaned


def load_scene_tokenizer_state_dict_strict(
    tokenizer: nn.Module,
    payload: Any,
    *,
    source: str | Path,
) -> None:
    """Load a complete tokenizer state, rejecting absent/partial/random fallback.

    Key and tensor-shape validation happens before ``load_state_dict`` so a bad
    checkpoint cannot partially mutate the tokenizer before raising.
    """

    state = extract_scene_tokenizer_state_dict(payload, source=source)
    expected = tokenizer.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = sorted(
        key
        for key in set(expected).intersection(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"{source} does not contain a complete compatible scene_tokenizer state. "
            f"missing={len(missing)} {_format_keys(missing)}, "
            f"unexpected={len(unexpected)} {_format_keys(unexpected)}, "
            f"shape_mismatches={len(shape_mismatches)} {_format_keys(shape_mismatches)}. "
            "Pass the tokenizer checkpoint used to create the SceneFlow latents; "
            "falling back to a randomly initialized tokenizer is forbidden."
        )
    tokenizer.load_state_dict(state, strict=True)


def load_scene_tokenizer_checkpoint_strict(
    tokenizer: nn.Module,
    checkpoint_path: str | Path,
) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scene tokenizer checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    load_scene_tokenizer_state_dict_strict(tokenizer, payload, source=path)
