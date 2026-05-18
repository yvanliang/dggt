"""I/O helpers for FlowDGGT offline cache payloads.

The cache schema is a nested PyTorch payload.  PyTorch's default serializer is
an uncompressed ZIP64 container, so wrap it in gzip when disk space matters.
"""
from __future__ import annotations

import gzip
import io
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import torch


GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def is_gzip_file(path: str | os.PathLike[str]) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def is_zstd_file(path: str | os.PathLike[str]) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == ZSTD_MAGIC


def _find_zstd_binary() -> str:
    for candidate in (
        os.environ.get("ZSTD_BIN"),
        shutil.which("zstd"),
        "/home/dancer/anaconda3/bin/zstd",
        "/usr/bin/zstd",
    ):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError(
        "zstd compression requested but no zstd binary was found. "
        "Install zstd or set ZSTD_BIN=/path/to/zstd."
    )


def load_flow_cache(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
    mmap: bool | None = None,
) -> dict[str, Any]:
    """Load a gzip, zstd, or plain torch cache payload."""
    if is_gzip_file(path):
        with gzip.open(path, "rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=weights_only)
    if is_zstd_file(path):
        data = subprocess.check_output([_find_zstd_binary(), "-q", "-dc", str(path)])
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=weights_only)
    return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=mmap)


def save_flow_cache(
    payload: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    compression: str = "gzip",
    gzip_level: int = 1,
    zstd_level: int | None = None,
) -> None:
    """Save a FlowDGGT cache payload.

    `compression="gzip"` / `"zstd"` keeps the existing `.pt` payload layout
    while reducing the uncompressed tensor storages produced by `torch.save`.
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
        elif compression in ("zstd", "zst"):
            level = int(gzip_level if zstd_level is None else zstd_level)
            level = max(1, min(19, level))
            proc = subprocess.Popen(
                [_find_zstd_binary(), "-q", f"-{level}", "-T0", "-f", "-o", str(tmp_path), "-"],
                stdin=subprocess.PIPE,
            )
            assert proc.stdin is not None
            try:
                torch.save(payload, proc.stdin)
                proc.stdin.close()
                rc = proc.wait()
            except Exception:
                proc.kill()
                proc.wait()
                raise
            if rc != 0:
                raise RuntimeError(f"zstd failed while writing {tmp_path} with exit code {rc}")
        else:
            raise ValueError(f"Unsupported flow cache compression: {compression}")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise
