"""LRU cache for 3D Gaussian assets loaded from disk.

Wraps the ``_load_asset_gaussians`` loader in :mod:`dggt.utils.gaussian_edit`
with a bounded LRU cache so callers can share a single asset store across
many ``localize_objects`` calls (inference sweeps, training loops, etc.) while
retaining backwards compatibility with code that expects a plain
``dict[str, dict[str, torch.Tensor]]`` cache.

The loader itself still lives in :mod:`dggt.utils.gaussian_edit` (it is tightly
coupled to the Waymo PLY / SPZ schema); this module only controls cache size
and ordering.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterator

import torch

from dggt.utils.gaussian_edit import load_asset_gaussians


class AssetBank:
    """Bounded LRU cache of loaded 3DGS assets keyed by file path.

    Entries are returned as the same dict-of-tensors structure produced by
    :func:`dggt.utils.gaussian_edit.load_asset_gaussians` (``means_raw``,
    ``colors``, ``opacities``, ``scales``, ``quats``, ``vertex_count``).
    """

    def __init__(self, max_size: int = 128):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = int(max_size)
        self._cache: "OrderedDict[str, dict[str, torch.Tensor]]" = OrderedDict()

    def get(self, path: str) -> dict[str, torch.Tensor]:
        entry = self._cache.get(path)
        if entry is None:
            entry = load_asset_gaussians(path, self._cache)
            self._evict()
        else:
            self._cache.move_to_end(path)
        return entry

    def as_raw_cache(self) -> dict[str, dict[str, torch.Tensor]]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()

    def __contains__(self, path: str) -> bool:
        return path in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)

    def _evict(self) -> None:
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
