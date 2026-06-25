"""
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u tools/visualize_tokenizer_result.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --tokenizer_ckpt_path /data/disk2/lyy_dataset/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt \
    --processed_root /data/disk2/lyy_dataset/waymo_processed_dggt \
    --split validation \
    --start 0 \
    --end 100 \
    --num_frames 8 \
    --vis_frames 8 \
    --views 1 \
    --sample_window 8 \
    --output_dir runs/tokenizer_visualize_val_000_099 \
    --precision bf16 \
    --device cuda:0
"""


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import WaymoEditDataset
from dggt.models.vggt import VGGT
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.gaussian_ply import write_point_ply
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.tokens import (
    replace_selected_levels,
    select_patch_pyramid,
    split_joint_channels,
    split_special_and_patch,
)
from train_tokenizer import render_scene_from_outputs


TOKENIZER_CKPT_PATH = "/data/disk2/lyy_dataset/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt"
DGGT_CKPT_PATH = "/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt"
PROCESSED_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt"
TRANSFER_ROOT = "/data/disk2/lyy_dataset/waymo_transfer"
RAW_ROOT = "/data/disk2/lyy_dataset/waymo"
ASSET_ROOT = "/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed"
OUTPUT_DIR = "runs/tokenizer_pointcloud_compare"


def unwrap_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(value)}")
    return value


def tokenizer_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Received an empty batch")
    frame_counts = [int(unwrap_tensor(sample["num_frames"]).item()) for sample in batch]
    if len(set(frame_counts)) != 1:
        raise ValueError(f"All samples in the batch must have the same frame count, got {frame_counts}")
    out: dict[str, Any] = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def infer_patch_grid(images: torch.Tensor, num_patches: int, patch_size: int = 14) -> tuple[int, int]:
    patch_h = int(images.shape[-2]) // patch_size
    patch_w = int(images.shape[-1]) // patch_size
    if patch_h * patch_w != int(num_patches):
        raise ValueError(
            f"Image size {tuple(images.shape[-2:])} with patch_size={patch_size} "
            f"does not match num_patches={num_patches}"
        )
    return patch_h, patch_w


def extract_levels(model: VGGT) -> tuple[int, ...]:
    levels = tuple(model.gs_head.intermediate_layer_idx)
    for name in ("depth_head", "point_head", "instance_head"):
        head_levels = tuple(getattr(model, name).intermediate_layer_idx)
        if head_levels != levels:
            raise ValueError(f"Head level mismatch: gs_head={levels}, {name}={head_levels}")
    return levels


def load_model_checkpoint(model: VGGT, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(cleaned, strict=False)


def _extract_scene_tokenizer_state_dict(payload: Any, ckpt_path: str) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("scene_tokenizer", "state_dict", "model"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                break
        else:
            state_dict = payload
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported scene_tokenizer checkpoint format: {ckpt_path}")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        for prefix in ("module.", "scene_tokenizer.", "tokenizer."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_tokenizer_weights(model: VGGT, ckpt_path: str) -> None:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    state_dict = _extract_scene_tokenizer_state_dict(payload, ckpt_path)
    missing, unexpected = model.scene_tokenizer.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[load_tokenizer] non-strict load completed "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )


def build_dataset(args: argparse.Namespace) -> WaymoEditDataset:
    return WaymoEditDataset(
        processed_root=args.processed_root,
        transfer_root=args.transfer_root,
        raw_root=args.raw_root,
        asset_root=args.asset_root,
        split=args.split,
        sequence_length=args.num_frames,
        mode=1,
        views=args.views,
        sample_window=args.sample_window,
        clean_only=True,
        clean_split_seed=args.clean_split_seed,
        clean_train_ratio=args.clean_train_ratio,
    )


def reattach_special_tokens_from_selected(
    selected_template_tokens: list[torch.Tensor],
    patch_start_idx: int,
    patch_tokens: list[torch.Tensor],
) -> list[torch.Tensor]:
    outputs = []
    for template_tokens, new_patch_tokens in zip(selected_template_tokens, patch_tokens):
        special_tokens, _ = split_special_and_patch(template_tokens, patch_start_idx)
        outputs.append(torch.cat([special_tokens, new_patch_tokens], dim=-2))
    return outputs


def dense_heads_from_tokenizer_roundtrip(
    model: VGGT,
    images: torch.Tensor,
    levels: tuple[int, ...],
) -> dict[str, Any]:
    agg_all, image_all, _dino_all, _image_feature, patch_start_idx = model.extract_scene_tokens(images)
    image_patch = select_patch_pyramid(image_all, levels, patch_start_idx)
    patch_grid = infer_patch_grid(images, image_patch[0].shape[2])

    z = model.scene_tokenizer.encode(image_patch, patch_grid=patch_grid)
    decoded_patch = model.scene_tokenizer.decode(z, patch_grid=patch_grid)

    selected_image_levels = [image_all[level_idx] for level_idx in levels]
    decoded_image_levels = reattach_special_tokens_from_selected(
        selected_image_levels,
        patch_start_idx,
        decoded_patch,
    )
    decoded_agg_levels = []
    decoded_dino_levels = []
    for joint_tokens in decoded_image_levels:
        dino_tokens, frame_tokens, global_tokens = split_joint_channels(joint_tokens)
        decoded_dino_levels.append(dino_tokens)
        decoded_agg_levels.append(torch.cat([frame_tokens, global_tokens], dim=-1))

    agg_hat_all = replace_selected_levels(agg_all, levels, decoded_agg_levels)
    image_hat_all = replace_selected_levels(image_all, levels, decoded_image_levels)
    dino_hat_all = replace_selected_levels(_dino_all, levels, decoded_dino_levels)

    pose_enc = model.camera_head(agg_hat_all)[-1]
    depth, depth_conf = model.depth_head(agg_hat_all, images=images, patch_start_idx=patch_start_idx)
    gs_map, gs_conf = model.gs_head(image_hat_all, images, patch_start_idx)
    dynamic_conf, _ = model.instance_head(dino_hat_all, images, patch_start_idx)
    return {
        "pose_enc": pose_enc,
        "depth": depth,
        "depth_conf": depth_conf,
        "gs_map": gs_map,
        "gs_conf": gs_conf,
        "dynamic_conf": dynamic_conf,
        "z": z,
        "decoded_patch": decoded_patch,
        "patch_grid": patch_grid,
        "patch_start_idx": int(patch_start_idx),
    }


def point_map_from_depth_and_pose(
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    image_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(depth.shape[0])
    point_maps = []
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc.float(), image_hw)
    for batch_idx in range(batch_size):
        point_map_np = unproject_depth_map_to_point_map(
            depth[batch_idx].detach().float().cpu(),
            extrinsics[batch_idx, :, :3, :].detach().float().cpu(),
            intrinsics[batch_idx].detach().float().cpu(),
        )
        point_maps.append(torch.from_numpy(point_map_np))
    return torch.stack(point_maps, dim=0).to(device=device, dtype=torch.float32)


def _valid_points_mask(points: torch.Tensor, conf: torch.Tensor | None) -> torch.Tensor:
    mask = torch.isfinite(points).all(dim=-1)
    if conf is not None:
        conf_s = conf.squeeze(-1) if conf.ndim == points.ndim else conf
        mask = mask & torch.isfinite(conf_s)
    return mask


def _flatten_frame_points(
    points_frame: torch.Tensor,
    colors_frame: torch.Tensor,
    conf_frame: torch.Tensor | None,
    valid_mask_frame: torch.Tensor | None,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if stride > 1:
        points_frame = points_frame[::stride, ::stride]
        colors_frame = colors_frame[::stride, ::stride]
        if conf_frame is not None:
            conf_frame = conf_frame[::stride, ::stride]
        if valid_mask_frame is not None:
            valid_mask_frame = valid_mask_frame[::stride, ::stride]

    mask = _valid_points_mask(points_frame, conf_frame)
    if valid_mask_frame is not None:
        mask = mask & valid_mask_frame.bool()
    points = points_frame[mask].detach().cpu().float()
    colors = colors_frame[mask].detach().cpu().float().clamp(0.0, 1.0)
    conf = None
    if conf_frame is not None:
        conf_s = conf_frame.squeeze(-1) if conf_frame.ndim == 3 else conf_frame
        conf = conf_s[mask].detach().cpu().float()
    return points, colors, conf


def save_pointcloud_set(
    root: Path,
    prefix: str,
    points_b: torch.Tensor,
    colors_b: torch.Tensor,
    conf_b: torch.Tensor | None,
    valid_mask_b: torch.Tensor | None,
    stride: int,
) -> dict[str, Any]:
    out_dir = root / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size, seq_len = points_b.shape[:2]
    summary: dict[str, Any] = {"frames": [], "stride": int(stride)}

    merged_points = []
    merged_colors = []
    merged_conf = []
    for b in range(batch_size):
        for s in range(seq_len):
            conf_frame = None if conf_b is None else conf_b[b, s]
            valid_frame = None if valid_mask_b is None else valid_mask_b[b, s]
            points, colors, conf = _flatten_frame_points(
                points_b[b, s],
                colors_b[b, s],
                conf_frame,
                valid_frame,
                stride=stride,
            )
            frame_stem = f"sample{b:02d}_frame{s:02d}"
            ply_path = out_dir / f"{frame_stem}.ply"
            npz_path = out_dir / f"{frame_stem}.npz"
            write_point_ply(points, colors, ply_path)
            np.savez_compressed(
                npz_path,
                points=points.numpy(),
                colors=colors.numpy(),
                conf=None if conf is None else conf.numpy(),
            )
            summary["frames"].append(
                {
                    "sample": b,
                    "frame": s,
                    "num_points": int(points.shape[0]),
                    "ply": str(ply_path),
                    "npz": str(npz_path),
                }
            )
            merged_points.append(points)
            merged_colors.append(colors)
            if conf is not None:
                merged_conf.append(conf)

    if merged_points:
        points_all = torch.cat(merged_points, dim=0)
        colors_all = torch.cat(merged_colors, dim=0)
        write_point_ply(points_all, colors_all, out_dir / "merged.ply")
        np.savez_compressed(
            out_dir / "merged.npz",
            points=points_all.numpy(),
            colors=colors_all.numpy(),
            conf=None if not merged_conf else torch.cat(merged_conf, dim=0).numpy(),
        )
        summary["merged"] = {
            "num_points": int(points_all.shape[0]),
            "ply": str(out_dir / "merged.ply"),
            "npz": str(out_dir / "merged.npz"),
        }
    return summary


def save_tokenizer_render_comparison_grids(
    root: Path,
    images: torch.Tensor,
    direct_render: torch.Tensor,
    tokenizer_render: torch.Tensor,
    *,
    max_frames: int,
) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    images = images.detach().float().cpu().clamp(0.0, 1.0)
    direct_render = direct_render.detach().float().cpu().clamp(0.0, 1.0)
    tokenizer_render = tokenizer_render.detach().float().cpu().clamp(0.0, 1.0)
    batch_size = int(images.shape[0])
    summaries: list[dict[str, Any]] = []

    for batch_idx in range(batch_size):
        frames = min(
            int(max_frames),
            int(images.shape[1]),
            int(direct_render.shape[1]),
            int(tokenizer_render.shape[1]),
        )
        if frames <= 0:
            raise ValueError("Cannot save tokenizer render grid with zero frames")
        input_rgb = images[batch_idx, :frames]
        direct_rgb = direct_render[batch_idx, :frames]
        recon_rgb = tokenizer_render[batch_idx, :frames]
        mse = (direct_rgb - recon_rgb).pow(2).mean(dim=1, keepdim=True)
        mse_viz = (mse / mse.max().clamp_min(1e-8)).clamp(0.0, 1.0).repeat(1, 3, 1, 1)

        grid = torch.cat([input_rgb, direct_rgb, recon_rgb, mse_viz], dim=0)
        filename = (
            "tokenizer_roundtrip_3dgs_comparison_grid.jpg"
            if batch_size == 1
            else f"sample{batch_idx:02d}_tokenizer_roundtrip_3dgs_comparison_grid.jpg"
        )
        path = root / filename
        save_image_grid(grid, path, nrow=frames)

        save_image_grid(input_rgb, root / f"sample{batch_idx:02d}_input_rgb_gt.jpg", nrow=frames)
        save_image_grid(direct_rgb, root / f"sample{batch_idx:02d}_dggt_direct_3dgs_rgb.jpg", nrow=frames)
        save_image_grid(recon_rgb, root / f"sample{batch_idx:02d}_tokenizer_roundtrip_3dgs_rgb.jpg", nrow=frames)
        summaries.append(
            {
                "sample": batch_idx,
                "frames": frames,
                "comparison_grid": str(path),
                "mse_mean": float(mse.mean().item()),
                "mse_max": float(mse.max().item()),
            }
        )
    return summaries


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export DGGT vs tokenizer-roundtrip point clouds.")
    parser.add_argument("--tokenizer_ckpt_path", default=TOKENIZER_CKPT_PATH)
    parser.add_argument("--ckpt_path", default=DGGT_CKPT_PATH)
    parser.add_argument("--processed_root", default=PROCESSED_ROOT)
    parser.add_argument("--transfer_root", default=TRANSFER_ROOT)
    parser.add_argument("--raw_root", default=RAW_ROOT)
    parser.add_argument("--asset_root", default=ASSET_ROOT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--split", default="validation", choices=["training", "validation", "train", "val"])
    parser.add_argument("--index", type=int, default=0, help="Single-sample index, or start index when --start is omitted.")
    parser.add_argument("--start", type=int, default=None, help="First dataset index to process, inclusive.")
    parser.add_argument("--end", type=int, default=None, help="Last dataset index to process, exclusive.")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--vis_frames", type=int, default=8, help="Frames/columns in the 4-row render grid.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Legacy compatibility: if --end is omitted, process index:index+batch_size. Samples are still looped one by one.",
    )
    parser.add_argument("--views", type=int, default=1, choices=[1, 3])
    parser.add_argument("--sample_window", type=int, default=20)
    parser.add_argument("--clean_split_seed", type=int, default=0)
    parser.add_argument("--clean_train_ratio", type=float, default=0.9)
    parser.add_argument("--stride", type=int, default=1, help="Pixel stride for exported point clouds.")
    parser.add_argument("--include_sky", action="store_true", help="Also export sky pixels. Default filters sky.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if not Path(args.ckpt_path).is_file():
        raise FileNotFoundError(f"DGGT checkpoint not found: {args.ckpt_path}")
    if not Path(args.tokenizer_ckpt_path).is_file():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {args.tokenizer_ckpt_path}")

    print(f"[init] loading dataset split={args.split}", flush=True)
    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError(f"No Stage-A clean-only samples found for split={args.split}")

    print("[init] loading DGGT", flush=True)
    model = VGGT().to(device)
    load_model_checkpoint(model, args.ckpt_path)
    print("[init] loading tokenizer", flush=True)
    load_tokenizer_weights(model, args.tokenizer_ckpt_path)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    levels = extract_levels(model)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"
    start = int(args.index if args.start is None else args.start)
    if args.end is None:
        end = start + max(1, int(args.batch_size))
    else:
        end = int(args.end)
    start = max(0, min(start, len(dataset)))
    end = max(start, min(end, len(dataset)))
    indices = list(range(start, end))
    if not indices:
        raise RuntimeError(f"No dataset indices selected: start={start} end={end} len={len(dataset)}")

    print(f"[run] loop indices=[{start}, {end}) count={len(indices)} levels={levels}", flush=True)
    all_summaries: list[dict[str, Any]] = []

    for pos, idx in enumerate(indices, start=1):
        sample_out = out_root / f"sample_{idx:06d}"
        sample_out.mkdir(parents=True, exist_ok=True)
        print(f"[{pos}/{len(indices)}] index={idx} -> {sample_out}", flush=True)

        sample = dataset[(idx, args.num_frames)]
        batch = tokenizer_collate_fn([sample])
        images = unwrap_tensor(batch["images_clean"]).to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            direct = model(images)
            tok = dense_heads_from_tokenizer_roundtrip(model, images, levels)
            sky_mask = unwrap_tensor(batch["sky_mask"] if "sky_mask" in batch else batch["masks"]).to(device)
            timestamps = unwrap_tensor(batch["timestamps"]).to(device)
            direct_render = render_scene_from_outputs(
                model,
                images,
                sky_mask,
                timestamps,
                direct["pose_enc"].float(),
                direct["depth"].float(),
                direct["gs_map"].float(),
                direct["gs_conf"].float(),
                direct["dynamic_conf"].float(),
            )
            tok_render = render_scene_from_outputs(
                model,
                images,
                sky_mask,
                timestamps,
                tok["pose_enc"].float(),
                tok["depth"].float(),
                tok["gs_map"].float(),
                tok["gs_conf"].float(),
                tok["dynamic_conf"].float(),
            )

        render_summary = save_tokenizer_render_comparison_grids(
            sample_out,
            images,
            direct_render,
            tok_render,
            max_frames=int(args.vis_frames),
        )

        image_hw = (int(images.shape[-2]), int(images.shape[-1]))
        direct_points = point_map_from_depth_and_pose(
            direct["pose_enc"],
            direct["depth"],
            image_hw,
            device,
        )
        tok_points = point_map_from_depth_and_pose(
            tok["pose_enc"],
            tok["depth"],
            image_hw,
            device,
        )
        valid_mask = None
        if not args.include_sky and "sky_mask" in batch:
            sky_mask = unwrap_tensor(batch["sky_mask"]).to(device)
            valid_mask = (sky_mask.permute(0, 1, 3, 4, 2) == 0).any(dim=-1)

        direct_summary = save_pointcloud_set(
            sample_out,
            "direct_dggt",
            direct_points.detach().cpu().float(),
            direct["gs_map"][..., :3].detach().cpu().float(),
            direct.get("gs_conf").detach().cpu().float()
            if direct.get("gs_conf") is not None
            else None,
            valid_mask.detach().cpu() if valid_mask is not None else None,
            stride=max(1, int(args.stride)),
        )
        tok_summary = save_pointcloud_set(
            sample_out,
            "tokenizer_roundtrip",
            tok_points.detach().cpu().float(),
            tok["gs_map"][..., :3].detach().cpu().float(),
            tok.get("gs_conf").detach().cpu().float()
            if tok.get("gs_conf") is not None
            else None,
            valid_mask.detach().cpu() if valid_mask is not None else None,
            stride=max(1, int(args.stride)),
        )

        diff = (direct_points.float() - tok_points.float()).detach()
        finite = torch.isfinite(diff).all(dim=-1)
        diff_norm = torch.linalg.norm(diff[finite], dim=-1) if finite.any() else diff.new_zeros((0,))
        summary = {
            "tokenizer_ckpt_path": args.tokenizer_ckpt_path,
            "dggt_ckpt_path": args.ckpt_path,
            "dataset": {
                "split": args.split,
                "index": idx,
                "num_frames": int(args.num_frames),
                "views": int(args.views),
                "sample_window": int(args.sample_window),
                "include_sky": bool(args.include_sky),
                "sample_keys": sorted(batch.keys()),
            },
            "levels": list(levels),
            "patch_grid": list(tok["patch_grid"]),
            "patch_start_idx": int(tok["patch_start_idx"]),
            "z_shape": list(tok["z"].shape),
            "point_map_shape": list(direct_points.shape),
            "diff_l2": {
                "count": int(diff_norm.numel()),
                "mean": float(diff_norm.mean().item()) if diff_norm.numel() else None,
                "median": float(diff_norm.median().item()) if diff_norm.numel() else None,
                "p95": float(torch.quantile(diff_norm, 0.95).item()) if diff_norm.numel() else None,
                "max": float(diff_norm.max().item()) if diff_norm.numel() else None,
            },
            "render_comparison": render_summary,
            "direct_dggt": direct_summary,
            "tokenizer_roundtrip": tok_summary,
        }
        with (sample_out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        all_summaries.append({"index": idx, "output_dir": str(sample_out)})
        del batch, images, direct, tok, direct_render, tok_render, direct_points, tok_points
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (out_root / "all_summary.json").open("w") as f:
        json.dump(
            {
                "split": args.split,
                "start": start,
                "end": end,
                "count": len(all_summaries),
                "rows": all_summaries,
            },
            f,
            indent=2,
        )
    print(f"[done] wrote {len(all_summaries)} samples -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
