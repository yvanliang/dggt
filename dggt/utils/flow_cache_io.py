"""I/O helpers for FlowDGGT offline cache payloads.

The cache schema is a nested PyTorch payload.  PyTorch's default serializer is
an uncompressed ZIP64 container, so wrap it in gzip when disk space matters.
"""
from __future__ import annotations

import gzip
import io
import os
import threading
from pathlib import Path
from typing import Any

import torch


GZIP_MAGIC = b"\x1f\x8b"


def is_gzip_file(path: str | os.PathLike[str]) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def load_flow_cache(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    """Load either a gzip-wrapped or plain torch cache payload."""
    if is_gzip_file(path):
        with gzip.open(path, "rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=weights_only)
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def save_flow_cache(
    payload: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    compression: str = "gzip",
    gzip_level: int = 1,
) -> None:
    """Save a FlowDGGT cache payload.

    `compression="gzip"` keeps the existing `.pt` layout while reducing the
    uncompressed tensor storages produced by `torch.save`.
    """
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    compression = str(compression).lower()
    try:
        if compression in ("none", "off", "false", "0"):
            torch.save(payload, tmp_path)
        elif compression == "gzip":
            level = max(0, min(9, int(gzip_level)))
            with gzip.open(tmp_path, "wb", compresslevel=level) as f:
                torch.save(payload, f)
        else:
            raise ValueError(f"Unsupported flow cache compression: {compression}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise
