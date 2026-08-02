"""Small executable checks for the 2026-08-02 D1-D3 review fixes.

Run with::

    CUDA_VISIBLE_DEVICES=0 conda run -n dggt python \
        lyy_tools/verify_metric_gauge_postreview.py

The script intentionally checks the production feature-stats artifact in
addition to pure helper math, so a source-only fix cannot be mistaken for a
complete D2 repair.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggt.utils.factorized_asset_condition import build_placement_state
from dggt.utils.scene_gauge import metric_c2w_to_teacher_anchor_dggt


def verify_teacher_atlas_world(device: torch.device) -> dict[str, float]:
    anchor = torch.eye(4, dtype=torch.float64, device=device).view(1, 4, 4)
    # Camera image-up (-camera y) points along +z in the Waymo ego world.
    anchor[:, :3, :3] = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]],
        dtype=torch.float64,
        device=device,
    )
    anchor[:, :3, 3] = torch.tensor([[5.0, -2.0, 1.5]], dtype=torch.float64, device=device)
    metric = anchor[:, None].repeat(1, 4, 1, 1)
    metric[:, :, 0, 3] += torch.arange(4, dtype=torch.float64, device=device)
    teacher = metric_c2w_to_teacher_anchor_dggt(
        metric,
        anchor,
        torch.tensor([[math.log(2.0)]], dtype=torch.float64, device=device),
    )
    identity_error = float(
        (teacher[:, 0] - torch.eye(4, dtype=torch.float64, device=device)).abs().max().item()
    )
    image_up = teacher[:, 0, :3, :3] @ torch.tensor(
        [0.0, -1.0, 0.0], dtype=torch.float64, device=device
    )
    up_error = float(
        (image_up - torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64, device=device))
        .abs()
        .max()
        .item()
    )
    displacement_error = abs(float(teacher[:, 2, :3, 3].norm(dim=-1).item()) - 1.0)
    if max(identity_error, up_error, displacement_error) > 1.0e-10:
        raise AssertionError("D1 metric-camera rebasing did not produce the teacher atlas basis")
    return {
        "anchor_identity_max_abs": identity_error,
        "image_up_minus_y_max_abs": up_error,
        "two_metres_to_one_unit_abs_err": displacement_error,
    }


def verify_production_placement_stats(stats_path: Path) -> dict[str, float | int]:
    payload = torch.load(stats_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"feature stats must be a dict, got {type(payload).__name__}")
    mean = torch.as_tensor(payload["placement_mean"]).float()
    std = torch.as_tensor(payload["placement_std"]).float()
    count = int(torch.as_tensor(payload["placement_count"]).item())
    log_z_mean = float(mean[3].item())
    log_z_std = float(std[3].item())
    if not (0.0 < log_z_mean < 6.0 and 0.2 < log_z_std < 3.0):
        raise AssertionError(
            "D2 production stats still look like ego-world/near-plane polluted log_z_depth: "
            f"mean={log_z_mean:.6f} std={log_z_std:.6f}"
        )
    if float(mean[14].item()) != 0.0 or float(std[14].item()) != 1.0:
        raise AssertionError("placement passthrough channel 14 must remain identity-normalized")
    return {
        "placement_count": count,
        "log_z_depth_mean": log_z_mean,
        "log_z_depth_std": log_z_std,
    }


def verify_bounded_speed_depth(device: torch.device) -> dict[str, float]:
    z_values = torch.tensor([-20.0, 0.001, 1.0, 10.0, 100.0], device=device)
    center = torch.zeros(1, 1, len(z_values), 3, device=device)
    center[..., 2] = z_values
    size = torch.ones_like(center)
    yaw = torch.zeros(1, 1, len(z_values), device=device)
    velocity = torch.zeros_like(center)
    velocity[..., 0] = 20.0
    visible = z_values.gt(0.5).view(1, 1, -1)
    camera = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, len(z_values), 1, 1)
    state = build_placement_state(center, size, yaw, velocity, visible, camera)
    channel = state[..., 14]
    minimum = float(channel.min().item())
    maximum = float(channel.max().item())
    if not bool(torch.isfinite(channel).all()) or minimum < 0.0 or maximum > 1.0:
        raise AssertionError(f"D3 bounded speed/depth channel escaped [0,1]: [{minimum}, {maximum}]")
    return {"channel14_min": minimum, "channel14_max": maximum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=Path("logs/scene_flow_pretrain_1024/feature_stats_pretrain_v4.pt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    result = {
        "schema": "metric_gauge_postreview_d1_d3_v1",
        "stats_path": str(args.stats_path.resolve()),
        "d1_teacher_atlas_world": verify_teacher_atlas_world(device),
        "d2_placement_stats": verify_production_placement_stats(args.stats_path),
        "d3_bounded_speed_depth": verify_bounded_speed_depth(device),
        "status": "pass",
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
