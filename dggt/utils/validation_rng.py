"""RNG isolation helpers for deterministic validation."""
from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


VALIDATION_FLOW_SEED_OFFSET = 10_000_019


def make_validation_generator(device: torch.device, base_seed: int) -> torch.Generator:
    """Create a fixed validation-only generator without advancing global RNG."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(base_seed) + VALIDATION_FLOW_SEED_OFFSET)
    return generator


@contextmanager
def preserve_validation_rng_state(device: torch.device) -> Iterator[None]:
    """Restore process-global RNG streams after validation, including on errors."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
