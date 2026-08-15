"""The two rebalances asked for after reading the v6 run at 8k steps.

Both are stated as measured shares rather than as bare numbers, so a future
change to one of the neighbouring weights shows up here as a failing ratio
instead of as a silently different objective.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import train_scene_flow_pretrain as trainer
from dggt.losses.reconstruction_feedback_loss import (
    DYNAMIC_HEAD_LOSS_WEIGHT,
    compute_reconstruction_feedback_losses,
)

# Head sub-losses measured on the v6 run, steps 7000-7950 (already carrying the
# sigma sample weight, which is why they are small).
V6_HEAD = {
    "depth": 2.5248e-4,
    "depth_conf": 2.9132e-5,
    "gaussian": 1.2117e-3,
    "gs_conf": 1.0410e-2,
    "dynamic": 5.7413e-5,
}
# Refined sky-mask sub-losses over the same window.
V6_SKY_REFINE = {"bce": 0.054401, "dice": 0.13781, "boundary_bce": 0.77449}
V6_TOTAL_LOSS = 1.4333


def _head_split(dynamic_weight: float) -> dict[str, float]:
    terms = {
        "depth": V6_HEAD["depth"],
        "depth_conf": 0.1 * V6_HEAD["depth_conf"],
        "gaussian": V6_HEAD["gaussian"],
        "gs_conf": 0.1 * V6_HEAD["gs_conf"],
        "dynamic": dynamic_weight * V6_HEAD["dynamic"],
    }
    total = sum(terms.values())
    return {name: value / total for name, value in terms.items()}


def test_dynamic_reaches_parity_with_depth_in_the_head_loss() -> None:
    """It was 2.2% -- a fifth of depth -- after the probability-space fix.

    Separating a moving actor from static background is what the dynamic head
    is for, and it is the capability the layout conditioning is built around,
    so it should not be the smallest term but one.
    """

    before = _head_split(1.0)
    after = _head_split(DYNAMIC_HEAD_LOSS_WEIGHT)
    assert before["dynamic"] == pytest.approx(0.0224, abs=5e-4)
    assert after["dynamic"] == pytest.approx(0.084, abs=5e-3)
    # Parity with depth, without displacing the two GS terms.
    assert after["dynamic"] == pytest.approx(after["depth"], rel=0.15)
    assert after["gaussian"] > after["gs_conf"] > after["depth"]


def _only_dynamic_geometry(dynamic_logit: float) -> SimpleNamespace:
    """Minimal decoded geometry whose only student/teacher difference is dynamic."""

    image_tokens: list[torch.Tensor | None] = [None] * 24
    for level in (4, 11, 17, 23):
        image_tokens[level] = torch.zeros((1, 1, 1, 2))
    gs_map = torch.zeros((1, 1, 1, 1, 11))
    gs_map[..., 4:7] = 1.0
    gs_map[..., 7] = 1.0
    return SimpleNamespace(
        image_tokens=image_tokens,
        depth=torch.ones((1, 1, 1, 1, 1)),
        depth_conf=torch.ones((1, 1, 1, 1)),
        gs_map=gs_map,
        gs_conf=torch.ones((1, 1, 1, 1)),
        dynamic_conf=torch.full((1, 1, 1, 1, 1), dynamic_logit),
    )


def test_the_dynamic_weight_is_actually_applied_to_the_head_loss() -> None:
    """Exercise the production aggregation, not a duplicate of its arithmetic."""

    result = compute_reconstruction_feedback_losses(
        student_geometry=_only_dynamic_geometry(1.0),
        teacher_geometry=_only_dynamic_geometry(0.0),
        patch_grid=(1, 1),
        patch_weight_mask=torch.ones((1, 1, 1, 1)),
        loss_sky_mask_gt=torch.zeros((1, 1, 1, 1, 1)),
        sky_weight=0.0,
        max_frames=0,
        render_stride=1,
        sample_weight=torch.ones(1),
        conf_weight_power=0.0,
    )
    dynamic = result.logs["loss_head_dynamic"]
    assert DYNAMIC_HEAD_LOSS_WEIGHT == pytest.approx(4.0)
    assert dynamic > 0.0
    assert result.logs["loss_head_consistency"] == pytest.approx(
        DYNAMIC_HEAD_LOSS_WEIGHT * dynamic
    )


def _sky_refine_total(boundary_loss_weight: float, boundary_weight: float = 4.0) -> float:
    return (
        V6_SKY_REFINE["bce"]
        + 0.5 * V6_SKY_REFINE["dice"]
        + boundary_loss_weight * boundary_weight * V6_SKY_REFINE["boundary_bce"]
    )


def test_the_sky_mask_boundary_term_is_halved() -> None:
    """It was 86% of loss_sky_mask_refine and 5.4% of the whole objective.

    That is seventeen times the entire world-feedback stack, spent on the
    slowest-moving term in the run: -2.4%/1k against the flow loss's -6.7%/1k.
    """

    parser = trainer.build_argparser()
    weight = parser.get_default("sky_mask_refine_boundary_loss_weight")
    band = parser.get_default("sky_mask_refine_boundary_weight")
    assert weight == pytest.approx(0.125)
    assert weight * band == pytest.approx(0.5)  # was 0.25 * 4 = 1.0

    lambda_refine = parser.get_default("lambda_sky_mask_refine")
    before = 0.25 * band * V6_SKY_REFINE["boundary_bce"]
    after = weight * band * V6_SKY_REFINE["boundary_bce"]
    assert after == pytest.approx(0.5 * before)
    # v6 logged loss_sky_mask_refine = 0.8978 at the old weight.
    assert _sky_refine_total(0.25) == pytest.approx(0.8978, rel=1e-3)

    share_before = lambda_refine * before / V6_TOTAL_LOSS
    share_after = lambda_refine * after / V6_TOTAL_LOSS
    assert share_before == pytest.approx(0.054, abs=1e-3)
    assert share_after == pytest.approx(0.027, abs=1e-3)


def test_the_boundary_band_multiplier_is_untouched() -> None:
    """Halve how much the boundary term counts, not which pixels are boundary."""

    parser = trainer.build_argparser()
    assert parser.get_default("sky_mask_refine_boundary_weight") == pytest.approx(4.0)


def test_the_sky_dome_gets_one_mosaic_row_per_cfg_scale() -> None:
    """The dome is its own generated stream; one row hid how it moves with CFG."""

    frames, gy, gx, channels = 2, 5, 7, 8
    bundle = trainer.SimpleNamespace(
        z_clean_n=torch.randn((1, frames, gy * gx, channels)),
        sky_mask_refined_clean=None,
    )
    args = trainer.argparse.Namespace(
        val_log_images=frames, patch_grid=(gy, gx), val_mosaic_cell_width=16
    )
    dome = torch.rand((frames, 3, 14 * gy, 14 * gx))
    groups = {}
    for scale_index, scale in enumerate((1.0, 2.0, 4.0)):
        rows = trainer.collect_validation_mosaic_rows(
            bundle,
            torch.randn_like(bundle.z_clean_n),
            {"generated_sky_rgb": dome},
            args,
            scene_slot=0,
            scene_label="000",
            guidance_scale=scale,
            scale_index=scale_index,
            is_primary=scale_index == 0,
        )
        for row in rows:
            groups.setdefault(row["group"], []).append(row["order"])
    assert sorted(groups["sky_rgb"]) == [0, 1, 2]
