from __future__ import annotations

import pytest
import torch

from dggt.utils.sliding_window import (
    cosine_coverage,
    cosine_window,
    default_window_stride,
    resolve_offline_window,
    scene_global_window_weight,
    window_slices,
)


def test_sliding_window_requires_overlap_and_covers_tail() -> None:
    assert window_slices(5, 8, 8) == [(0, 5)]
    assert window_slices(17, 8, 0) == [(0, 8), (4, 12), (8, 16), (9, 17)]
    assert window_slices(10, 8, 3)[-1] == (2, 10)
    with pytest.raises(ValueError, match="require overlap"):
        window_slices(16, 8, 8)
    with pytest.raises(ValueError, match="require overlap"):
        window_slices(16, 8, 9)


def test_scene_global_sky_weights_give_equal_per_frame_contribution() -> None:
    windows = window_slices(17, 8, 4)
    coverage = cosine_coverage(17, windows)
    per_frame = torch.zeros(17)
    total_global_weight = 0.0
    for start, end in windows:
        local = cosine_window(end - start)
        per_frame[start:end] += local / coverage[start:end]
        total_global_weight += float(scene_global_window_weight(start, end, coverage))
    assert torch.allclose(per_frame, torch.ones_like(per_frame), atol=1e-6)
    assert total_global_weight == pytest.approx(17.0, abs=1e-5)


def test_offline_window_policy_automatically_bounds_long_requests() -> None:
    assert default_window_stride(10) == 7
    assert resolve_offline_window(10, 0, 0) == (10, 7, False)
    assert resolve_offline_window(11, 0, 0) == (10, 7, True)
    assert resolve_offline_window(29, 29, 7) == (10, 7, True)
    assert resolve_offline_window(7, 4, 2) == (4, 2, True)
    with pytest.raises(ValueError, match="requires overlap"):
        resolve_offline_window(11, 10, 10)
