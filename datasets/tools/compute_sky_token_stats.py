"""Compute the frozen per-channel standardization constants for sky tokens.

The scene latent the generator lives in is standardized to unit variance
(``validation/sample_latent_target_std`` logs 1.0011).  The sky flow target was
not: packed as raw ``rgb * 2 - 1`` it has a large mean and a std well under one,
and per channel blue is the worst -- nearly saturated with the smallest spread,
which is exactly where cloud contrast lives.  At the training sigma that buries
the cloud signal far deeper than the scene's.

This tool measures the mean and std of the packed sky token over *observed*
atlas cells -- the ones the loss weights at 1.0 -- so
``SKY_TOKEN_CHANNEL_MEAN`` / ``SKY_TOKEN_CHANNEL_STD`` can be frozen in
``train_scene_flow_pretrain.py``.

    python datasets/tools/compute_sky_token_stats.py \
        --image_dir /data/lyy_dataset/waymo_processed_dggt/training \
        --scenes 200 --frames 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_scene_flow_pretrain as trainer  # noqa: E402


def _load(root: Path, sub: str, ext: str, mode: str, frames, hw) -> torch.Tensor:
    out = []
    for frame in frames:
        with Image.open(root / sub / f"{frame:03d}_0.{ext}") as handle:
            array = np.asarray(handle.convert(mode), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)
        out.append(tensor.permute(2, 0, 1) if tensor.ndim == 3 else tensor[None])
    return F.interpolate(torch.stack(out), size=hw, mode="area").unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=350)
    parser.add_argument("--width", type=int, default=518)
    # The gauge row this camera was calibrated at; only the ray directions
    # matter, so any consistent physical intrinsic gives the same atlas.
    parser.add_argument("--log_tan_half_fov_x", type=float, default=-0.7816628217697144)
    parser.add_argument("--log_tan_half_fov_y", type=float, default=-1.1531145572662354)
    args = parser.parse_args()

    root = Path(args.image_dir)
    scenes = sorted(p.name for p in root.iterdir() if p.is_dir())[: int(args.scenes)]
    hw = (int(args.height), int(args.width))
    fx = (args.width / 2) / float(np.exp(args.log_tan_half_fov_x))
    fy = (args.height / 2) / float(np.exp(args.log_tan_half_fov_y))

    total = torch.zeros(trainer.SKY_RGB_DIM, dtype=torch.float64)
    total_sq = torch.zeros(trainer.SKY_RGB_DIM, dtype=torch.float64)
    count = torch.zeros(trainer.SKY_RGB_DIM, dtype=torch.float64)
    used = 0
    for name in scenes:
        scene = root / name
        available = sorted(
            int(p.stem.split("_")[0]) for p in (scene / "images").glob("*_0.jpg")
        )
        if len(available) < 2:
            continue
        picked = [available[i] for i in np.linspace(0, len(available) - 1, int(args.frames)).astype(int)]
        try:
            images = _load(scene, "images", "jpg", "RGB", picked, hw)
            masks = _load(scene, "sky_masks", "png", "L", picked, hw)
        except (OSError, ValueError):
            continue
        seq = images.shape[1]
        extrinsics = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, seq, 1, 1)
        intrinsics = torch.tensor(
            [[fx, 0.0, args.width / 2], [0.0, fy, args.height / 2], [0.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3).repeat(1, seq, 1, 1)
        atlas, observation = trainer.build_sky_atlas_from_images(
            images, masks, extrinsics=extrinsics, intrinsics=intrinsics
        )
        tokens = trainer.pack_sky_rgb_atlas(atlas, standardize=False)
        weight = trainer.pack_sky_atlas_loss_weight(observation, unobserved_weight=0.0)
        p = trainer.SKY_PATCH_SIZE
        values = tokens.reshape(-1, trainer.SKY_RGB_DIM, p * p).double()
        keep = weight.reshape(-1, trainer.SKY_RGB_DIM, p * p).double().gt(0.5)
        total += (values * keep).sum(dim=(0, 2))
        total_sq += (values.square() * keep).sum(dim=(0, 2))
        count += keep.sum(dim=(0, 2))
        used += 1

    if not bool(count.gt(1).all()):
        raise SystemExit("no observed sky cells found; check --image_dir")
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min(0.0).sqrt()
    print(f"scenes used {used}, frames each {args.frames}, "
          f"observed cells per channel {int(count[0])}")
    print(f"SKY_TOKEN_CHANNEL_MEAN = ({', '.join(f'{v:.6f}' for v in mean.tolist())})")
    print(f"SKY_TOKEN_CHANNEL_STD = ({', '.join(f'{v:.6f}' for v in std.tolist())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
