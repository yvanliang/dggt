"""Canonical time units for DGGT Gaussian temporal opacity.

One unit represents four Waymo frames.  Timestamps are derived from clip-local
frame ids instead of being normalized by the sampled window length, so the same
frame has the same timestamp in pretrain, formal training, and every sliding
window used at inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


GAUSSIAN_FRAMES_PER_TIME_UNIT = 4.0
GAUSSIAN_TIME_REPRESENTATION = "clip_local_frame_id_div4_v1"


def gaussian_timestamps_from_frame_ids(frame_ids: Any):
    """Convert clip-local frame ids to the canonical Gaussian time scale."""
    if torch.is_tensor(frame_ids):
        return frame_ids.to(dtype=torch.float32) / GAUSSIAN_FRAMES_PER_TIME_UNIT
    values = np.asarray(frame_ids, dtype=np.float32)
    return values / np.float32(GAUSSIAN_FRAMES_PER_TIME_UNIT)
