#!/usr/bin/env python3
"""Small executable contract check for Phase-6 metric inference/export."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.camera_generation import camera_state_from_waymo_c2w
from dggt.utils.camera_geometry_flow_consistency import (
    camera_geometry_flow_consistency,
)
from dggt.utils.scene_gauge import (
    PullbackCalibration,
    assemble_dggt_pose_encoding,
    metric_c2w_to_dggt,
)
from inference_scene_flow_pretrain import (
    decode_metric_camera_from_features,
    prepare_generated_geometry_boundaries,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    c2w_metric = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
    c2w_metric[0, :, 0, 3] = torch.tensor([0.0, 1.0, 3.0], device=device)
    anchor_to_world = torch.eye(4, device=device).view(1, 4, 4)
    camera_state, anchor_mask = camera_state_from_waymo_c2w(
        c2w_metric, anchor_to_world
    )
    decoded = decode_metric_camera_from_features(
        camera_state,
        camera_anchor_mask=anchor_mask,
        trajectory_anchor_to_world=anchor_to_world,
    )
    torch.testing.assert_close(decoded.camera_to_world, c2w_metric)

    gauge = torch.tensor(
        [[[math.log(2.0), math.log(0.5), math.log(0.4)]]], device=device
    )
    pose_dggt = assemble_dggt_pose_encoding(
        metric_c2w_to_dggt(decoded.camera_to_world, gauge[..., 0]), gauge
    )
    if tuple(pose_dggt.shape) != (1, 3, 9):
        raise AssertionError(f"unexpected DGGT pose shape {tuple(pose_dggt.shape)}")

    depth = torch.tensor(
        [[[[[1.0], [2.0]], [[4.0], [8.0]]]]], device=device
    )
    gs_map = torch.arange(1 * 1 * 2 * 2 * 11, device=device, dtype=torch.float32).reshape(
        1, 1, 2, 2, 11
    )
    calibration = PullbackCalibration(
        path=Path("synthetic_pullback.json"),
        artifact_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        dggt_sha256="c" * 64,
        tokenizer_generation="t0_v2",
        window_len=10,
        patch_grid_hw=(25, 37),
        depth_a=math.log(1.1),
        depth_b=0.2,
        reference_depth_m=20.0,
        runtime_depth_clamp_m=(0.5, 80.0),
        c_gs=1.0,
        depth_form="loglinear",
    )
    render, metric = prepare_generated_geometry_boundaries(
        depth=depth,
        gs_map=gs_map,
        gauge=gauge,
        calibration=calibration,
        export_units="metric",
    )
    if render.depth_dggt is not depth or render.gs_map_dggt is not gs_map:
        raise AssertionError("render boundary is not identity")
    torch.testing.assert_close(metric.depth_dggt[..., 0], depth[..., 0] * metric.c_depth_factor)
    torch.testing.assert_close(
        metric.gs_map_dggt[..., 4:7],
        gs_map[..., 4:7] * metric.c_depth_factor[..., None],
    )
    torch.testing.assert_close(metric.gs_map_dggt[..., :4], gs_map[..., :4])
    torch.testing.assert_close(metric.gs_map_dggt[..., 7:], gs_map[..., 7:])

    plane_depth = torch.full((1, 3, 16, 16, 1), 5.0, device=device)
    flow_consistency = camera_geometry_flow_consistency(
        plane_depth,
        pose_dggt,
        sample_stride=2,
        min_pair_support_pixels=8,
    )
    if flow_consistency["status"] != "ok":
        raise AssertionError(flow_consistency)
    if flow_consistency["informative_metrics"]["flow_cycle_epe_px"]["mean"] > 1.0e-5:
        raise AssertionError(flow_consistency)

    print(
        json.dumps(
            {
                "status": "pass",
                "device": str(device),
                "camera_metric_roundtrip_max_abs": float(
                    (decoded.camera_to_world - c2w_metric).abs().max().item()
                ),
                "pose_shape": list(pose_dggt.shape),
                "render_identity": True,
                "c_depth_factor_range": [
                    float(metric.c_depth_factor.min().item()),
                    float(metric.c_depth_factor.max().item()),
                ],
                "metric_means_and_scales_global_factor": math.exp(
                    float(gauge[0, 0, 0].item())
                ),
                "camera_geometry_flow_consistency": {
                    "status": flow_consistency["status"],
                    "informative_pair_count": flow_consistency[
                        "informative_pair_count"
                    ],
                    "metric_support_fraction_of_sampled": flow_consistency[
                        "support"
                    ]["metric_support_fraction_of_sampled"],
                    "flow_cycle_epe_px_mean": flow_consistency[
                        "informative_metrics"
                    ]["flow_cycle_epe_px"]["mean"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
