from __future__ import annotations

import pytest
import torch

from inference_scene_flow_pretrain import (
    CAMERA_GAUGE_ATTRIBUTION_ARMS,
    camera_gauge_attribution_arm_inputs,
)
from train_scene_flow_pretrain import sampled_gauge_validation_metrics
from tools.audit_scene_flow_gradient_balance import (
    _gradient_cosine,
    _gradient_vector_stats,
    _resolve_probe_indices,
)


def test_final_sampled_gauge_diagnostics_are_separately_prefixed() -> None:
    target = torch.tensor([[[3.0, -0.7, -0.9]]])
    prediction = target + torch.tensor([[[0.25, 0.1, -0.1]]])
    metrics = sampled_gauge_validation_metrics(
        prediction,
        target,
        torch.ones(1, 3, dtype=torch.bool),
        prior_log_scale=3.5,
        prefix="sample_gauge_cfg2",
    )
    assert set(metrics) == {
        "sample_gauge_cfg2_valid_frac",
        "sample_gauge_cfg2_log_scale_error",
        "sample_gauge_cfg2_fov_error_deg",
        "sample_gauge_cfg2_vs_prior_gain",
    }
    assert metrics["sample_gauge_cfg2_log_scale_error"] == pytest.approx(0.25)


def test_camera_gauge_four_arm_design_changes_only_declared_factor() -> None:
    teacher_camera = torch.full((1, 3, 9), 1.0)
    generated_camera = torch.full((1, 3, 9), 2.0)
    teacher_gauge = torch.full((1, 1, 3), 3.0)
    generated_gauge = torch.full((1, 1, 3), 4.0)
    arms = camera_gauge_attribution_arm_inputs(
        teacher_camera=teacher_camera,
        generated_camera=generated_camera,
        teacher_gauge=teacher_gauge,
        generated_gauge=generated_gauge,
    )
    assert tuple(arms) == CAMERA_GAUGE_ATTRIBUTION_ARMS
    expected = (
        (teacher_camera, teacher_gauge),
        (generated_camera, teacher_gauge),
        (teacher_camera, generated_gauge),
        (generated_camera, generated_gauge),
    )
    for actual, expected_pair in zip(arms.values(), expected, strict=True):
        assert actual[0] is expected_pair[0]
        assert actual[1] is expected_pair[1]


def test_gradient_audit_reports_norm_and_conflict_on_shared_parameters() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    video = parameter.square().sum()
    camera = -parameter.square().sum()
    video_grads, video_norm = _gradient_vector_stats(
        video, [parameter], retain_graph=True
    )
    camera_grads, camera_norm = _gradient_vector_stats(
        camera, [parameter], retain_graph=False
    )
    assert video_norm == pytest.approx(camera_norm)
    assert _gradient_cosine(
        video_grads, camera_grads, video_norm, camera_norm
    ) == pytest.approx(-1.0)
    assert _resolve_probe_indices("0,mid,last", 6) == [0, 3, 5]
