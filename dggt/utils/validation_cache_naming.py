"""Readable, backward-compatible naming for validation flow caches."""
from __future__ import annotations

import re
from pathlib import Path

VALIDATION_VARIANTS = (
    "combined",
    "deletion",
    "insertion",
    "replacement",
    "repositioning",
)
VALIDATION_VARIANT_ORD = {
    variant: ordinal for ordinal, variant in enumerate(VALIDATION_VARIANTS)
}
LEGACY_VARIANT_ALIAS = {
    "delete": "deletion",
    "add": "insertion",
    "replace": "replacement",
    "move": "repositioning",
}
_CANONICAL_NAME_RE = re.compile(
    r"^(?P<entry>\d{6})_(?P<variant>combined|deletion|insertion|replacement|repositioning)\.pt$"
)
_OLD_READABLE_NAME_RE = re.compile(
    r"^entry_(?P<entry>\d+)__(?P<variant>combined|delete|add|replace|move)\.pt$"
)


def validation_cache_index(entry_index: int, variant: str) -> int:
    variant = normalize_validation_variant(variant)
    return int(entry_index) * len(VALIDATION_VARIANTS) + VALIDATION_VARIANT_ORD[variant]


def validation_cache_filename(entry_index: int, variant: str) -> str:
    variant = normalize_validation_variant(variant)
    return f"{int(entry_index):06d}_{variant}.pt"


def validation_cache_path(split_root: str | Path, entry_index: int, variant: str) -> Path:
    return Path(split_root) / validation_cache_filename(entry_index, variant)


def legacy_validation_cache_path(
    split_root: str | Path,
    entry_index: int,
    variant: str,
) -> Path:
    return Path(split_root) / f"{validation_cache_index(entry_index, variant):06d}.pt"


def old_readable_validation_cache_path(
    split_root: str | Path,
    entry_index: int,
    variant: str,
) -> Path:
    canonical = normalize_validation_variant(variant)
    legacy = {
        "combined": "combined",
        "deletion": "delete",
        "insertion": "add",
        "replacement": "replace",
        "repositioning": "move",
    }[canonical]
    return Path(split_root) / f"entry_{int(entry_index):03d}__{legacy}.pt"


def is_canonical_validation_cache_filename(path: str | Path) -> bool:
    return _CANONICAL_NAME_RE.match(Path(path).name) is not None


def parse_validation_cache_filename(path: str | Path) -> tuple[int, str] | None:
    path = Path(path)
    match = _CANONICAL_NAME_RE.match(path.name)
    if match is not None:
        return int(match.group("entry")), str(match.group("variant"))
    match = _OLD_READABLE_NAME_RE.match(path.name)
    if match is not None:
        return int(match.group("entry")), normalize_validation_variant(
            str(match.group("variant"))
        )
    try:
        flat_index = int(path.stem)
    except ValueError:
        return None
    variant = VALIDATION_VARIANTS[flat_index % len(VALIDATION_VARIANTS)]
    return flat_index // len(VALIDATION_VARIANTS), variant


def normalize_validation_variant(variant: str) -> str:
    variant = str(variant)
    variant = LEGACY_VARIANT_ALIAS.get(variant, variant)
    if variant not in VALIDATION_VARIANT_ORD:
        raise ValueError(f"unknown validation variant: {variant!r}")
    return variant


def validation_asset_condition_kind(variant: str) -> str:
    """Map a validation edit to its formal asset-condition semantics.

    A pure deletion uses only the learned conditional empty-asset token.
    Combined validation edits retain their real asset tokens and append the
    empty token for their independent deletion component. Replacement and
    repositioning are represented by their real target asset tokens.
    """
    canonical = normalize_validation_variant(variant)
    if canonical == "deletion":
        return "mode_b_empty"
    if canonical == "combined":
        return "mode_a_with_empty"
    return "mode_a"
