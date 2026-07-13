#!/usr/bin/env python3
"""Offline inference for the SceneFlow full-scene pretrain model.

The data path intentionally matches ``train_scene_flow_pretrain.py`` validation:

* raw ``WaymoOpenDataset(mode=1, views=1)`` validation clips;
* trunk-major sample ordering (all scenes' trunk 0, then trunk 1, ...);
* pure-noise RAE/FlowMatch sampling through ``cfg_sample_pretrain_latents``;
* generated camera/sky/sky-mask states and the no-image DGGT head/render path.

Condition modes rotate by global trunk-major validation row:

    none -> asset -> cam -> asset_cam -> ...

``none`` means text-only; text remains the required base condition.  Missing
asset/camera modalities use the learned null-condition tokens used by pretrain
condition dropout.  Every CFG value is applied to all modalities actually
present in the selected mode, so equal factored scales telescope to ordinary
CFG between the full condition and the all-null condition.

Example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow_pretrain.py \
        --cfg 1.0,2.0,4.0 \
        --num_frames 8 \
        --start 0 --end 8 \
        --output_dir runs/scene_flow_pretrain_inference

Each sample gets one directory whose name includes its condition mode.  In
addition to training-validation image grids (except ``abs_error``), each CFG
result contains:

* ``generated_raw_gaussians__cfg*.ply``: DGGT/3DGS Gaussian PLY schema;
* ``generated_raw_points__cfg*.ply``: xyz + uchar RGB, directly viewable in
  MeshLab.

The PLYs are built by DGGT's canonical ``build_clean_scene_state`` and written
by its shared PLY writers.  Consequently they use the same non-sky + valid-depth
selection, world-coordinate unprojection, Gaussian activations and serialization
as ``inference_scene_editor.py``.  As in DGGT, overlapping input frames can
contain duplicate surface samples; no unrequested fusion heuristic is applied.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ``datasets.dataset`` imports Open3D at module scope, but the raw pretrain
# dataset path used here never calls it.  Match the repository's lightweight
# dataloader inspection utility so ``--help`` and inference remain usable in
# the training environment where Open3D is intentionally not installed.
try:
    import open3d as _open3d  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open3d"] = types.ModuleType("open3d")

try:
    import gsplat as _gsplat  # noqa: F401
except ModuleNotFoundError:
    # Keep argument parsing/import-time unit tests available.  Actual RGB
    # rendering still fails with an explicit dependency error if invoked.
    gsplat_stub = types.ModuleType("gsplat")
    gsplat_rendering_stub = types.ModuleType("gsplat.rendering")

    def _missing_gsplat(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "gsplat is required for SceneFlow 3DGS rendering; run this script in the dggt training environment."
        )

    gsplat_stub.rasterization = _missing_gsplat
    gsplat_rendering_stub.rasterization = _missing_gsplat
    sys.modules["gsplat"] = gsplat_stub
    sys.modules["gsplat.rendering"] = gsplat_rendering_stub

from datasets.dataset import WaymoOpenDataset
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.feature_stats import load_into_buffers
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.gaussian_edit import CleanSceneState, build_clean_scene_state
from dggt.utils.gaussian_ply import write_gaussian_ply, write_point_ply
from train_scene_flow_pretrain import (
    DEFAULT_SKY_GRID,
    SKY_TOKEN_DIM,
    CyclicSequentialSampler,
    _decode_generated_tokens_without_template,
    _fixed_render_hw,
    _image_grid,
    _latent_pca_grid,
    _mask_grid,
    _predict_camera_mats,
    _render_gs_map_rgb,
    _sky_background_image_grid,
    _sky_mask_image_grid,
    _sky_mask_patch_to_image,
    _split_sparse_generated_tokens_for_heads,
    _timestamps_for_generated_render,
    autocast_context,
    build_pretrain_bundle_from_batch,
    cfg_sample_pretrain_latents,
    decode_pose_from_camera_features,
    discover_scene_names,
    load_dggt_aggregator_and_tokenizer,
    seed_everything,
    setup_text_encoder,
    sky_generation_enabled,
    sky_grid_shape,
    sky_tokens_to_background,
    validate_scene_flow_checkpoint_config,
)


DEFAULT_WEIGHTS = "logs/scene_flow_pretrain_1024/ckpt/pretrain_step048000_ema_weights_only.pt"
DEFAULT_VAL_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt/validation"
DEFAULT_CAPTION_ROOT = "/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions"
DEFAULT_DGGT_CKPT = "/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt"
DEFAULT_TOKENIZER_CKPT = "logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt"
DEFAULT_TEXT_ENCODER = "/home/dancer/model/Qwen/Qwen3-0.6B/"
CONDITION_MODES = ("none", "asset", "cam", "asset_cam")


def parse_cfg_scales(value: str | Sequence[str]) -> list[float]:
    """Parse comma/space separated CFG values while preserving user order."""
    raw_values = [value] if isinstance(value, str) else list(value)
    scales: list[float] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            scale = float(part)
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(f"CFG scales must be finite and non-negative, got {part!r}.")
            if scale not in scales:
                scales.append(scale)
    if not scales:
        raise ValueError("--cfg must contain at least one scale.")
    return scales


def cfg_tag(scale: float) -> str:
    return f"{float(scale):g}".replace("-", "m").replace(".", "p")


def condition_mode_for_position(position: int) -> str:
    return CONDITION_MODES[int(position) % len(CONDITION_MODES)]


def _batch_size_from_bundle(bundle: Any) -> int:
    return int(bundle.z_clean_n.shape[0])


def apply_condition_mode(bundle: Any, mode: str) -> Any:
    """Hide optional conditions with the same learned-null semantics as training."""
    if mode not in CONDITION_MODES:
        raise ValueError(f"Unknown condition mode {mode!r}; expected one of {CONDITION_MODES}.")
    batch_size = _batch_size_from_bundle(bundle)
    use_asset = mode in {"asset", "asset_cam"}
    use_camera = mode in {"cam", "asset_cam"}

    if not use_asset:
        if torch.is_tensor(getattr(bundle, "encoder_attention_mask", None)):
            bundle.encoder_attention_mask = torch.zeros_like(bundle.encoder_attention_mask, dtype=torch.bool)
        if torch.is_tensor(getattr(bundle, "F_asset_lengths", None)):
            bundle.F_asset_lengths = torch.zeros_like(bundle.F_asset_lengths)
        bundle.asset_condition_kind = ["asset_uncond"] * batch_size

    if not use_camera:
        # Do not pass GT-derived pose summaries at all.  The sampler inserts one
        # learned camera-null token per generated frame via camera_condition_kind.
        bundle.camera_pose_tokens = None
        bundle.camera_attention_mask = None
        bundle.camera_condition_kind = ["camera_uncond"] * batch_size
    elif getattr(bundle, "camera_condition_kind", None) is None:
        bundle.camera_condition_kind = ["camera"] * batch_size
    return bundle


def _row_has_any(mask: torch.Tensor | None, batch_size: int) -> list[bool]:
    if not torch.is_tensor(mask):
        return [False] * int(batch_size)
    return mask.detach().to(torch.bool).reshape(int(batch_size), -1).any(dim=1).cpu().tolist()


def actual_condition_rows(bundle: Any) -> dict[str, list[bool]]:
    batch_size = _batch_size_from_bundle(bundle)
    asset_rows = _row_has_any(getattr(bundle, "encoder_attention_mask", None), batch_size)
    asset_kinds = getattr(bundle, "asset_condition_kind", None)
    if isinstance(asset_kinds, str):
        asset_kinds = [asset_kinds] * batch_size
    if asset_kinds is not None:
        asset_rows = [
            bool(row) and str(asset_kinds[idx]).lower() not in {"none", "asset_uncond", "asset_null"}
            for idx, row in enumerate(asset_rows)
        ]
    camera_tokens = getattr(bundle, "camera_pose_tokens", None)
    camera_rows = (
        _row_has_any(getattr(bundle, "camera_attention_mask", None), batch_size)
        if torch.is_tensor(camera_tokens)
        else [False] * batch_size
    )
    camera_kinds = getattr(bundle, "camera_condition_kind", None)
    if isinstance(camera_kinds, str):
        camera_kinds = [camera_kinds] * batch_size
    if camera_kinds is not None:
        camera_rows = [
            bool(row) and str(camera_kinds[idx]).lower() not in {"camera_uncond", "camera_null"}
            for idx, row in enumerate(camera_rows)
        ]
    return {"asset": asset_rows, "camera": camera_rows}


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def _checkpoint_config(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    config = payload.get("scene_flow_config")
    return dict(config) if isinstance(config, dict) else None


def build_scene_flow_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    no_ema: bool,
    fallback_args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the exact saved architecture and load EMA weights strictly."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SceneFlow checkpoint not found: {path}. The requested default is intentionally allowed "
            "to be absent until training writes it; pass --weights when it becomes available."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(payload)
    if config is not None:
        scene_flow = WanSceneFlow(**config)
    else:
        scene_flow = WanSceneFlow.from_scene_config(
            bring_up=False,
            patch_grid=fallback_args.patch_grid,
            in_channels=3 * int(fallback_args.latent_dim) + 3,
            out_channels=int(fallback_args.latent_dim),
            camera_gen_dim=int(fallback_args.camera_gen_dim),
            sky_token_dim=SKY_TOKEN_DIM,
            sky_grid=fallback_args.sky_grid,
            max_sky_tokens=int(fallback_args.sky_grid[0] * fallback_args.sky_grid[1]),
            sky_mask_refine_scale=int(fallback_args.sky_mask_refine_scale),
            sky_mask_refine_channels=int(fallback_args.sky_mask_refine_channels),
            prediction_type=str(fallback_args.prediction_type),
        )
    validate_scene_flow_checkpoint_config(scene_flow, payload, path)

    source = ""
    if no_ema:
        if not isinstance(payload, dict) or "scene_flow" not in payload:
            raise ValueError(f"{path} has no raw scene_flow state for --no_ema.")
        if bool(payload.get("is_ema_weights")):
            raise ValueError(f"{path} is EMA-only; remove --no_ema or use a full/raw checkpoint.")
        state = payload["scene_flow"]
        source = "raw scene_flow"
    elif isinstance(payload, dict) and "ema_scene_flow_state_dict" in payload:
        state = payload["ema_scene_flow_state_dict"]
        source = "ema_scene_flow_state_dict"
    elif isinstance(payload, dict) and bool(payload.get("is_ema_weights")) and "scene_flow" in payload:
        state = payload["scene_flow"]
        source = "EMA-only scene_flow"
    elif isinstance(payload, dict) and "ema_scene_flow" in payload and "scene_flow" in payload:
        # Legacy full checkpoint: materialize EMA shadow tensors on the model.
        from diffusers.training_utils import EMAModel

        scene_flow.load_state_dict(_strip_module_prefix(payload["scene_flow"]), strict=True)
        ema = EMAModel(scene_flow.parameters())
        ema.load_state_dict(payload["ema_scene_flow"])
        ema.copy_to(scene_flow.parameters())
        state = None
        source = "legacy ema_scene_flow"
    elif isinstance(payload, dict) and "scene_flow" in payload:
        state = payload["scene_flow"]
        source = "raw scene_flow (EMA unavailable)"
        print(f"[warn] {path} does not contain EMA weights; falling back to raw weights.", flush=True)
    elif isinstance(payload, dict):
        state = payload
        source = "bare state_dict"
        print(f"[warn] {path} is a bare state_dict; EMA provenance cannot be verified.", flush=True)
    else:
        raise ValueError(f"Unsupported SceneFlow checkpoint format: {path}")

    if state is not None:
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint state {source} is not a state_dict.")
        scene_flow.load_state_dict(_strip_module_prefix(state), strict=True)
    scene_flow.to(device).eval()
    for parameter in scene_flow.parameters():
        parameter.requires_grad_(False)
    info = {
        "path": str(path),
        "weight_source": source,
        "step": int(payload.get("step", 0)) if isinstance(payload, dict) else 0,
        "scene_flow_config": config,
    }
    del payload
    return scene_flow, info


def sync_args_from_model(args: argparse.Namespace, scene_flow: nn.Module) -> None:
    config = scene_flow.config
    args.patch_grid = tuple(int(v) for v in config.patch_grid)
    args.latent_dim = int(config.out_channels)
    args.prediction_type = str(config.prediction_type)
    args.sky_grid = tuple(int(v) for v in config.sky_grid)
    args.sky_grid_h, args.sky_grid_w = args.sky_grid
    args.sky_mask_refine_scale = int(config.sky_mask_refine_scale)
    args.sky_mask_refine_channels = int(config.sky_mask_refine_channels)


def build_generated_dggt_scene_state(
    *,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
    sky_mask: torch.Tensor,
) -> CleanSceneState:
    """Build the generated scene through DGGT's canonical scene-state path.

    Only ``images_clean`` metadata is synthetic because generated inference has
    no source RGB image.  DGGT uses that tensor solely for ``[S,H,W]`` shape and
    stores it in the returned state; geometry and PLY fields come exclusively
    from pose/depth/GS/dynamic predictions plus the generated sky mask.
    """
    if int(pose_enc.shape[0]) != 1:
        raise ValueError(f"PLY export currently expects batch size 1, got pose_enc={tuple(pose_enc.shape)}")
    if (
        sky_mask.ndim != 5
        or int(sky_mask.shape[0]) != 1
        or int(sky_mask.shape[2]) not in (1, 3)
    ):
        raise ValueError(f"sky_mask must be [1,S,1/3,H,W], got {tuple(sky_mask.shape)}")
    seq_len, height, width = int(sky_mask.shape[1]), int(sky_mask.shape[-2]), int(sky_mask.shape[-1])
    images_clean = torch.zeros((seq_len, 3, height, width), dtype=torch.float32)
    sky_mask_cpu = sky_mask[0].detach().cpu().float().contiguous()
    sample = {
        "images_clean": images_clean,
        # build_clean_scene_state's fallback expression accesses `masks`
        # eagerly, so provide both aliases exactly as normal DGGT samples do.
        "sky_mask": sky_mask_cpu,
        "masks": sky_mask_cpu,
        "cam_ids": torch.tensor([0], dtype=torch.long),
    }
    predictions = {
        "pose_enc": pose_enc,
        "depth": depth,
        "gs_map": gs_map,
        "gs_conf": gs_conf,
        "dynamic_conf": dynamic_conf,
    }
    return build_clean_scene_state(sample, predictions)


def export_generated_pointclouds(
    *,
    scene_state: CleanSceneState,
    output_dir: Path,
    suffix: str,
    stride: int,
) -> dict[str, Any]:
    """Write DGGT canonical Gaussian PLY plus MeshLab RGB point PLY."""
    stride = max(1, int(stride))
    keep = torch.ones_like(scene_state.source_image_ids, dtype=torch.bool)
    if stride > 1:
        keep &= torch.remainder(scene_state.source_y, stride) == 0
        keep &= torch.remainder(scene_state.source_x, stride) == 0

    points = scene_state.means[keep]
    colors = scene_state.colors[keep]
    opacities = scene_state.opacities[keep]
    scales = scene_state.scales[keep]
    quats = scene_state.quats[keep]
    source_ids = scene_state.source_image_ids[keep]
    seq_len = int(scene_state.images.shape[0])
    frame_counts = torch.bincount(source_ids, minlength=seq_len).tolist()

    gaussian_path = output_dir / f"generated_raw_gaussians__{suffix}.ply"
    point_path = output_dir / f"generated_raw_points__{suffix}.ply"
    write_gaussian_ply(
        {
            "means": points,
            "features_dc_rgb": colors,
            "opacities": opacities,
            "scales": scales,
            "quats": quats,
        },
        gaussian_path,
    )
    write_point_ply(points, colors, point_path)
    return {
        "gaussian_ply": str(gaussian_path),
        "point_ply": str(point_path),
        "num_points": int(points.shape[0]),
        "frame_counts": frame_counts,
        "stride": stride,
        "schema": "DGGT Gaussian PLY + MeshLab xyz/uchar-RGB PLY",
        "scene_builder": "dggt.utils.gaussian_edit.build_clean_scene_state",
        "validity_rule": "generated non-sky mask AND generated depth > 1e-4",
        "coordinates": "DGGT generated-camera world coordinates",
    }


@torch.no_grad()
def render_and_export_generated(
    *,
    batch: dict[str, Any],
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    generated_sample: Any,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    suffix: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    z_generated = generated_sample.video
    camera_generated = generated_sample.camera
    sky_generated = generated_sample.sky
    sky_mask_patch = generated_sample.sky_mask_patch
    sky_mask_refined = generated_sample.sky_mask_refined
    if camera_generated is None:
        raise RuntimeError("Pretrain sampling did not generate camera tokens.")
    if sky_mask_patch is None:
        raise RuntimeError("Pretrain sampling did not generate a sky mask.")

    seq_len = int(z_generated.shape[1])
    height, width = _fixed_render_hw(args)
    frames = min(int(args.val_log_images), seq_len)
    timestamps = _timestamps_for_generated_render(batch, seq_len=seq_len, device=device)

    with autocast_context(args, device):
        generated_tokens, patch_start_idx = _decode_generated_tokens_without_template(
            vggt_model, scene_flow, z_generated, args, device=device
        )
        gen_agg, gen_dino = _split_sparse_generated_tokens_for_heads(generated_tokens)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            gs_map, gs_conf = vggt_model.gs_head(
                generated_tokens, None, patch_start_idx, image_hw=(height, width)
            )
            generated_pose = decode_pose_from_camera_features(vggt_model, camera_generated.to(device))
            generated_depth, _ = vggt_model.depth_head(
                gen_agg, None, patch_start_idx, image_hw=(height, width)
            )
            generated_dynamic, _ = vggt_model.instance_head(
                gen_dino, None, patch_start_idx, image_hw=(height, width)
            )
            generated_sky_mask = _sky_mask_patch_to_image(
                sky_mask_refined if sky_mask_refined is not None else sky_mask_patch,
                patch_grid=args.patch_grid,
                height=height,
                width=width,
                device=device,
            )
    del generated_tokens, gen_agg, gen_dino

    sky_background = None
    sky_grid_image = None
    if sky_generated is not None:
        sky_h, sky_w = sky_grid_shape(args)
        extrinsic, intrinsic = _predict_camera_mats(generated_pose, (height, width), device)
        sky_background = sky_tokens_to_background(
            sky_generated.to(device),
            seq_len=seq_len,
            height=height,
            width=width,
            grid_h=sky_h,
            grid_w=sky_w,
            extrinsics=extrinsic,
            intrinsics=intrinsic,
        )
        sky_grid_image = _sky_background_image_grid(sky_background, frames)

    images = {
        "generated_pred_sky_mask": _sky_mask_image_grid(generated_sky_mask, frames),
        "generated_raw_3dgs_rgb": _render_gs_map_rgb(
            vggt_model,
            None,
            generated_sky_mask,
            timestamps,
            generated_pose,
            generated_depth,
            gs_map,
            gs_conf,
            generated_dynamic,
            device,
            frames,
            background_mode="black",
            use_sky_mask=True,
            background_override=sky_background,
            image_hw=(height, width),
        ),
    }
    if sky_grid_image is not None:
        images["generated_sky_rgb"] = sky_grid_image

    scene_state = build_generated_dggt_scene_state(
        pose_enc=generated_pose,
        depth=generated_depth,
        gs_map=gs_map,
        gs_conf=gs_conf,
        dynamic_conf=generated_dynamic,
        sky_mask=generated_sky_mask,
    )
    ply_summary = export_generated_pointclouds(
        scene_state=scene_state,
        output_dir=output_dir,
        suffix=suffix,
        stride=args.ply_stride,
    )
    del generated_pose, generated_depth, generated_dynamic, generated_sky_mask, gs_map, gs_conf, scene_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images, ply_summary


def _first(value: Any, default: Any = None) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        result = value.reshape(-1)[0].item()
        return result
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def save_common_validation_images(
    bundle: Any,
    batch: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    frames = min(int(args.val_log_images), int(bundle.z_clean_n.shape[1]))
    images: dict[str, torch.Tensor] = {
        "target_latent_pca": _latent_pca_grid(bundle.z_clean_n, args.patch_grid, frames),
        "input_rgb_gt": _image_grid(batch["images"], frames),
    }
    sky_patch = getattr(bundle, "sky_mask_clean", None)
    if torch.is_tensor(sky_patch):
        images["target_sky_mask_patch"] = _mask_grid(sky_patch, args.patch_grid, frames)
    sky_refined = getattr(bundle, "sky_mask_refined_clean", None)
    if torch.is_tensor(sky_refined):
        images["target_sky_mask_refined"] = _sky_mask_image_grid(sky_refined, frames)

    paths: dict[str, str] = {}
    for name, tensor in images.items():
        path = output_dir / f"{name}.jpg"
        save_image_grid(tensor, path, nrow=frames)
        paths[name] = str(path)
    caption = str(_first(batch.get("caption"), ""))
    (output_dir / "caption.txt").write_text(caption + "\n")
    return paths


def save_cfg_images(
    z_generated: torch.Tensor,
    rgb_images: dict[str, torch.Tensor],
    output_dir: Path,
    args: argparse.Namespace,
    suffix: str,
) -> dict[str, str]:
    frames = min(int(args.val_log_images), int(z_generated.shape[1]))
    images = {
        f"generated_raw_latent_pca__{suffix}": _latent_pca_grid(
            z_generated, args.patch_grid, frames
        )
    }
    images.update({f"{name}__{suffix}": value for name, value in rgb_images.items()})
    paths: dict[str, str] = {}
    for name, tensor in images.items():
        path = output_dir / f"{name}.jpg"
        save_image_grid(tensor, path, nrow=frames)
        paths[name] = str(path)
    return paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full-scene SceneFlow pretrain inference on raw Waymo validation clips."
    )
    parser.add_argument(
        "--weights",
        "--scene_flow_ckpt_path",
        dest="weights",
        default=DEFAULT_WEIGHTS,
        help="Pretrain full/EMA-only checkpoint. Default is the requested step-048000 EMA path.",
    )
    parser.add_argument("--dggt_ckpt_path", default=DEFAULT_DGGT_CKPT)
    parser.add_argument("--tokenizer_ckpt_path", default=DEFAULT_TOKENIZER_CKPT)
    parser.add_argument(
        "--feature_stats_path",
        default=None,
        help="Optional override; by default mu_z/sigma_z are loaded from the SceneFlow checkpoint.",
    )
    parser.add_argument("--val_image_dir", "--image_dir", dest="val_image_dir", default=DEFAULT_VAL_ROOT)
    parser.add_argument("--val_caption_root", "--caption_root", dest="val_caption_root", default=DEFAULT_CAPTION_ROOT)
    parser.add_argument("--text_encoder_path", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--no_text_condition", action="store_true")
    parser.add_argument("--output_dir", default="runs/scene_flow_pretrain_inference")

    parser.add_argument("--val_scene_start", type=int, default=0)
    parser.add_argument("--val_scene_end", type=int, default=100)
    parser.add_argument("--start", type=int, default=0, help="First trunk-major validation dataset row.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive row; default processes the selected dataset.")
    parser.add_argument("--index", type=int, default=None, help="Process one trunk-major dataset row.")
    parser.add_argument(
        "--num_frames",
        "--frames",
        dest="num_frames",
        type=int,
        default=8,
        help="Contiguous frames per validation sample (default: 8).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument(
        "--cfg",
        "--cfg_scales",
        "--guidance_scales",
        dest="cfg_values",
        nargs="+",
        default=["1.0"],
        help="One or more comma/space-separated CFG scales (default: 1.0).",
    )
    parser.add_argument("--sample_steps", "--val_sample_steps", dest="val_sample_steps", type=int, default=35)
    parser.add_argument("--shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_ema", action="store_true", help="Use raw weights from a full checkpoint.")
    parser.add_argument("--no_sky_generation", action="store_true")
    parser.add_argument(
        "--ply_stride",
        type=int,
        default=1,
        help=(
            "Pixel-grid stride applied after DGGT canonical scene construction. "
            "Default 1 preserves the exact DGGT scene; >1 is explicit export-only downsampling."
        ),
    )
    parser.add_argument("--val_sliding_window", type=int, default=0)
    parser.add_argument("--val_sliding_stride", type=int, default=0)

    # Fallback architecture only; saved scene_flow_config takes precedence.
    parser.add_argument("--patch_grid_h", type=int, default=25)
    parser.add_argument("--patch_grid_w", type=int, default=37)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--prediction_type", choices=("x", "v"), default="x")
    parser.add_argument("--sky_grid_h", type=int, default=DEFAULT_SKY_GRID[0])
    parser.add_argument("--sky_grid_w", type=int, default=DEFAULT_SKY_GRID[1])
    parser.add_argument("--sky_mask_refine_scale", type=int, default=4)
    parser.add_argument("--sky_mask_refine_channels", type=int, default=256)
    parser.add_argument("--sky_unobserved_loss_weight", type=float, default=0.05)
    return parser


def validate_args(args: argparse.Namespace) -> list[float]:
    if int(args.num_frames) < 2:
        raise ValueError("--num_frames must be >= 2 (the raw dataset timestamp path and pretrain require it).")
    if int(args.num_frames) > 29:
        raise ValueError("--num_frames cannot exceed the 29-frame validation caption trunk.")
    if int(args.val_sample_steps) <= 0:
        raise ValueError("--sample_steps must be positive.")
    if int(args.ply_stride) <= 0:
        raise ValueError("--ply_stride must be positive.")
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be non-negative.")
    if int(args.val_scene_end) <= int(args.val_scene_start):
        raise ValueError("--val_scene_end must be greater than --val_scene_start.")
    return parse_cfg_scales(args.cfg_values)


def main() -> None:
    args = build_argparser().parse_args()
    cfg_scales = validate_args(args)
    args.patch_grid = (int(args.patch_grid_h), int(args.patch_grid_w))
    args.sky_grid = (int(args.sky_grid_h), int(args.sky_grid_w))
    args.camera_gen_dim = 2048
    args.caption_root = args.val_caption_root
    args.val_log_images = int(args.num_frames)
    # These are overwritten per CFG.  Equal scales implement ordinary CFG for
    # precisely the optional modalities present in the current mode.
    args.guidance_scale = 1.0
    args.asset_control_guidance_scale = 1.0
    args.camera_guidance_scale = 1.0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    seed_everything(int(args.seed))

    scene_flow, checkpoint_info = build_scene_flow_from_checkpoint(
        args.weights,
        device=device,
        no_ema=bool(args.no_ema),
        fallback_args=args,
    )
    sync_args_from_model(args, scene_flow)
    if args.feature_stats_path:
        load_into_buffers(scene_flow, args.feature_stats_path, token_dim=int(args.latent_dim))

    print(
        f"[checkpoint] {checkpoint_info['weight_source']} step={checkpoint_info['step']} "
        f"grid={args.patch_grid} latent_dim={args.latent_dim} prediction={args.prediction_type}",
        flush=True,
    )
    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path, args.tokenizer_ckpt_path, device
    )
    camera_dim = int(vggt_model.camera_head.token_norm.normalized_shape[0])
    if int(scene_flow.config.camera_gen_dim) != camera_dim:
        raise ValueError(
            f"SceneFlow camera_gen_dim={scene_flow.config.camera_gen_dim} does not match "
            f"DGGT CameraHead dim={camera_dim}."
        )
    text_encoder = setup_text_encoder(args, device)

    scene_names = discover_scene_names(args.val_image_dir, args.val_scene_start, args.val_scene_end)
    dataset = WaymoOpenDataset(
        image_dir=args.val_image_dir,
        scene_names=scene_names,
        sequence_length=int(args.num_frames),
        start_idx=0,
        mode=1,
        views=1,
        caption_root=None if bool(args.no_text_condition) else args.val_caption_root,
        pretrain_patch_grid=args.patch_grid,
        trunk_major_samples=True,
        trunk_frames=29,
    )
    if len(dataset) == 0:
        raise RuntimeError("The selected validation dataset has no full caption-trunk samples.")

    if args.index is not None:
        start = int(args.index)
        end = start + 1
    else:
        start = int(args.start)
        end = len(dataset) if args.end is None else int(args.end)
    start = max(0, min(start, len(dataset)))
    end = max(start, min(end, len(dataset)))
    if start == end:
        raise RuntimeError(f"No validation rows selected: [{start}, {end}) of {len(dataset)}.")

    sampler = CyclicSequentialSampler(dataset)
    sampler.set_offset(start)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        drop_last=False,
    )
    selected_count = end - start
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[data] scenes={len(scene_names)} rows={len(dataset)} selected=[{start},{end}) "
        f"order=trunk-major modes={','.join(CONDITION_MODES)} frames={args.num_frames}",
        flush=True,
    )
    print(f"[sampling] cfg={cfg_scales} steps={args.val_sample_steps} shift={args.shift}", flush=True)

    all_summaries: list[dict[str, Any]] = []
    iterator = tqdm(loader, total=selected_count, desc="pretrain inference", dynamic_ncols=True)
    for position, batch in enumerate(iterator):
        if position >= selected_count:
            break
        dataset_index = start + position
        # Bind the condition cycle to the global trunk-major row, not to the
        # current --start offset.  Sub-runs therefore keep the same mode for a
        # given validation sample.
        mode = condition_mode_for_position(dataset_index)
        scene_name = str(_first(batch.get("scene_name"), "unknown"))
        start_idx = int(_first(batch.get("start_idx"), 0))
        clip_index = int(_first(batch.get("clip_index"), start_idx // 29))
        sample_dir = output_root / (
            f"{dataset_index:06d}_scene{scene_name}_clip{clip_index:02d}_start{start_idx:03d}_{mode}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        bundle = build_pretrain_bundle_from_batch(batch, vggt_model, scene_flow, device, args)
        apply_condition_mode(bundle, mode)
        condition_rows = actual_condition_rows(bundle)
        common_paths = save_common_validation_images(bundle, batch, sample_dir, args)

        per_cfg: list[dict[str, Any]] = []
        for scale in cfg_scales:
            # Same sample step across CFG values => identical initial video,
            # camera and sky noise, matching training validation comparisons.
            args.guidance_scale = float(scale)
            args.asset_control_guidance_scale = float(scale)
            args.camera_guidance_scale = float(scale)
            generated = cfg_sample_pretrain_latents(
                scene_flow,
                bundle,
                args,
                step=dataset_index,
                device=device,
                guidance_scale=float(scale),
                text_encoder=text_encoder,
                return_camera=True,
                return_sky=sky_generation_enabled(args),
                return_sky_mask=True,
            )
            suffix = f"cfg{cfg_tag(scale)}"
            rgb_images, ply_summary = render_and_export_generated(
                batch=batch,
                vggt_model=vggt_model,
                scene_flow=scene_flow,
                generated_sample=generated,
                args=args,
                device=device,
                output_dir=sample_dir,
                suffix=suffix,
            )
            image_paths = save_cfg_images(
                generated.video, rgb_images, sample_dir, args, suffix
            )
            per_cfg.append(
                {
                    "cfg": float(scale),
                    "noise_seed": int(args.seed) + dataset_index,
                    "factored_scales": {
                        "text": float(scale),
                        "asset": float(scale),
                        "camera": float(scale),
                    },
                    "images": image_paths,
                    "pointcloud": ply_summary,
                }
            )
            del generated, rgb_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = {
            "dataset_index": dataset_index,
            "dataset_order": "trunk-major: every scene trunk 0, then every scene trunk 1, ...",
            "scene_name": scene_name,
            "clip_index": clip_index,
            "start_idx": start_idx,
            "frame_ids": batch["frame_ids"].detach().cpu().tolist()
            if torch.is_tensor(batch.get("frame_ids"))
            else batch.get("frame_ids"),
            "image_paths": batch.get("image_paths"),
            "caption": str(_first(batch.get("caption"), "")),
            "caption_path": str(_first(batch.get("caption_path"), "")),
            "condition_mode": mode,
            "condition_policy": "text always; optional asset/camera rotate none,asset,cam,asset_cam",
            "actual_condition_rows": condition_rows,
            "num_frames": int(args.num_frames),
            "sample_steps": int(args.val_sample_steps),
            "shift": float(args.shift),
            "checkpoint": checkpoint_info,
            "common_images": common_paths,
            "cfg_results": per_cfg,
            "abs_error_saved": False,
            "rgb_compositing": "gsplat_premultiplied_over_background",
        }
        (sample_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        all_summaries.append(summary)
        iterator.set_postfix(scene=scene_name, mode=mode)
        del bundle, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_summary = {
        "weights": str(args.weights),
        "checkpoint": checkpoint_info,
        "validation_root": str(args.val_image_dir),
        "caption_root": None if args.no_text_condition else str(args.val_caption_root),
        "rgb_compositing": "gsplat_premultiplied_over_background",
        "selected_range": [start, end],
        "cfg_scales": cfg_scales,
        "condition_cycle": list(CONDITION_MODES),
        "num_frames": int(args.num_frames),
        "sample_steps": int(args.val_sample_steps),
        "shift": float(args.shift),
        "ply_stride": int(args.ply_stride),
        "samples": all_summaries,
    }
    (output_root / "all_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))
    print(f"[done] wrote {len(all_summaries)} sample folders under {output_root}", flush=True)


if __name__ == "__main__":
    main()
