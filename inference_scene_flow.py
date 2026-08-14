"""Run the layout-free formal SceneFlow editor on an offline validation cache.

The editor is a separate from-scratch ``layout_condition_version='none'``
model. Requested Waymo camera parameters are conditioning only. Text CFG keeps
camera and edit controls fixed in both branches. The DGGT semantic head is used
only for a sky-mask diagnostic; production RGB rendering composites GT sky.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from datasets.waymo_flow_cache_dataset import (
    WaymoFlowCacheDataset,
)
from dggt.losses.flow_losses import (
    masked_flow_euler_step,
    project_masked_flow_state,
    rae_t_grid,
)
from dggt.models.flow_feature_assembler import FlowFeatureAssembler
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.feature_stats import (
    DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH,
)
from dggt.utils.flow_cache_io import (
    is_chunked_flow_cache,
    load_chunked_flow_cache_probe,
    load_chunked_flow_cache_subset,
    load_flow_cache,
)
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.flow_schedule import resolve_inference_flow_schedule
from dggt.utils.sliding_window import (
    OFFLINE_MAX_SINGLE_WINDOW,
    cosine_window,
    resolve_offline_window,
    window_slices,
)

# Reuse the formal-training (train_scene_flow.py) cache->bundle helpers verbatim
# so the bundle is byte-for-byte what the trainer feeds the model.
from train_scene_flow import (
    FORMAL_FLOW_DOMAIN_VERSION,
    FORMAL_LAYOUT_DISABLED_CONFIG,
    FORMAL_LAYOUT_CONDITION_VERSION,
    FORMAL_TOKENIZER_WINDOW_LEN,
    FORMAL_SCENE_FPS,
    _bundle_frame_ids,
    _infer_cache_patch_grid,
    _slice_time,
    build_formal_edit_domains,
    build_flow_bundle as build_train_flow_bundle,
    cached_render_pose_from_item,
    encode_text_condition,
    freeze_module,
    load_formal_latent_stats,
    render_validation_rgb_gt_sky,
    sampler_prediction_to_velocity,
    scene_flow_t_eps,
    setup_text_encoder,
    validate_formal_flow_domain_config,
)

# Reuse pretrain latent-grid helpers; RGB rendering uses formal T1 GT-sky helper.
from train_scene_flow_pretrain import (
    _image_grid,
    _latent_pca_grid,
    _mask_grid,
    _normalized_mask_grid,
    load_dggt_aggregator_and_tokenizer,
    unwrap_ddp,
)


# ---------------------------------------------------------------------- #
# CLI                                                                     #
# ---------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run trained WanSceneFlow on the offline validation flow-cache and "
            "dump training-style visualizations."
        )
    )
    # Models
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="DGGT checkpoint (aggregator + dense heads + scene tokenizer).")
    p.add_argument(
        "--tokenizer_ckpt_path",
        type=str,
        default=None,
        help=(
            "JointSceneTokenizer checkpoint matching SceneFlow training. It may be omitted "
            "only if --ckpt_path embeds a complete tokenizer state; no random fallback is allowed."
        ),
    )
    p.add_argument("--scene_flow_ckpt_path", type=str, required=True,
                   help="Formal T1 WanSceneFlow checkpoint with trained scaffold_packer.")
    p.add_argument("--feature_stats_path", type=str, default=str(DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH),
                   help=(
                       "Tokenizer latent statistics. They must exactly match the buffers "
                       "stored inside --scene_flow_ckpt_path; the checkpoint remains authoritative."
                   ))
    p.add_argument("--no_ema", action="store_true",
                   help="Use raw weights. By DEFAULT the EMA shadow weights are "
                        "used when the checkpoint carries them (mandatory for "
                        "meaningful diffusion samples; see docs §1.5 / 6e2c039f). "
                        "weights_only checkpoints have no EMA -> raw is used "
                        "with a warning.")
    p.add_argument("--text_encoder_path", type=str, default="/home/dancer/model/Qwen/Qwen3-0.6B",
                   help="Qwen text encoder path used by RAE-style SceneFlow training.")
    p.add_argument("--text_max_length", type=int, default=256)
    p.add_argument("--caption_root", type=str,
                   default="/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions",
                   help="Caption root containing pinhole_front/{clip_id}_{trunk_id}.json.")
    p.add_argument("--no_text_condition", action="store_true",
                   help="Disable text conditioning and caption lookup.")

    # Data (mirrors train_scene_flow.py / WaymoFlowCacheDataset)
    p.add_argument("--cache_root", type=str, default=None,
                   help="Validation cache root; flat {cache_root}/{split}/*.pt.")
    p.add_argument("--manifest_path", type=str, default=None,
                   help="validation_manifest.jsonl from build_flow_validation_manifest.py.")
    p.add_argument("--mode_filter", type=str, default=None,
                   help="Restrict manifest to comma-sep modes (validation is mode_a).")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Root directory for per-entry validation inference outputs.")

    # Sliding window over the (29-frame) clip; each denoising step blends
    # window velocities into one full-clip latent state.
    p.add_argument(
        "--window",
        type=int,
        default=10,
        help=(
            "Frames per SceneFlow window, capped at 10. Values <=0 select automatic mode; "
            "clips longer than 10 frames always use overlapping sliding windows."
        ),
    )
    p.add_argument("--window_stride", type=int, default=7,
                   help="Window step in frames; overlap is mandatory for clips longer than --window.")

    # Selection
    p.add_argument("--index", type=int, default=None,
                   help="Single dataset row index. If omitted, process [start,end).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)

    # Sampling
    p.add_argument("--sample_steps", type=int, default=50,
                   help="FlowMatch inference steps (15 smoke / 35 fast / 50 formal, matching formal validation).")
    p.add_argument("--shift", type=float, default=None,
                   help="Optional assertion for the checkpoint's FlowMatch / RAE shift.")
    p.add_argument("--edit_domain_threshold", type=float, default=1e-4,
                   help="Threshold soft source+destination coverage into the binary flow domain.")
    p.add_argument("--edit_domain_dilation", type=int, default=1,
                   help="Patch-grid dilation radius for the binary flow domain.")
    p.add_argument("--guidance_scales", type=str, default="1.0,2.0",
                   help="Comma-sep text CFG scales; one edited render per scale.")

    # Visualization (consumed by reused pretrain render helpers)
    p.add_argument(
        "--val_log_images",
        type=int,
        default=10,
        help=(
            "Number of frames rendered/tiled per grid; pass the full clip length "
            "(for example 29) to export all frames."
        ),
    )
    p.add_argument("--no_render_rgb", action="store_true",
                   help="Skip the (heavy) 3DGS RGB renders; latent/mask viz only.")
    p.add_argument("--render_per_window", action="store_true",
                   help="Render each window separately (per-window VGGT pass, "
                        "low memory) instead of one stitched full-clip render.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", type=str, default="bf16", choices=("fp32", "bf16"))
    return p


# ---------------------------------------------------------------------- #
# Bundle construction (identical to train_scene_flow.py:train_step)        #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def build_bundle(
    item: dict[str, Any],
    assembler: FlowFeatureAssembler,
    device: torch.device,
):
    bundle = build_train_flow_bundle(item, assembler, device)
    return bundle, item["sample"]


# ---------------------------------------------------------------------- #
# CFG editing sampler                                                     #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def _cfg_sample_edit_latents_sliding(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    device: torch.device,
    guidance_scale: float,
    seed: int,
    text_encoder: nn.Module | None,
    *,
    window: int,
    stride: int,
) -> torch.Tensor:
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean.float())
    t_steps = rae_t_grid(
        num_steps=int(args.sample_steps),
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
        t_eps=scene_flow_t_eps(scene_flow),
    )
    z_splat_n = sf.normalize(bundle.z_splat.float())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    M_preserve = bundle.M_preserve.to(device=device, dtype=z_clean_n.dtype)
    M_source = bundle.M_source.to(device=device, dtype=z_clean_n.dtype)
    M_dest = bundle.M_dest.to(device=device, dtype=z_clean_n.dtype)
    _, M_edit, _, _ = build_formal_edit_domains(
        bundle,
        args,
        device=device,
        dtype=z_clean_n.dtype,
    )
    batch_size = int(z_clean_n.shape[0])
    seq_len = int(z_clean_n.shape[1])
    frame_ids = _bundle_frame_ids(bundle, batch_size=batch_size, seq_len=seq_len, device=device)
    windows = window_slices(seq_len, window, stride)

    z = torch.empty_like(z_clean_n)
    z.normal_(generator=generator)
    z = project_masked_flow_state(z, z_splat_n, M_edit)

    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(
        text_encoder,
        [""] * batch_size if text_tokens is not None else None,
    )
    do_cfg = abs(float(guidance_scale) - 1.0) > 1e-6
    camera_condition_tokens = getattr(bundle, "camera_condition_tokens", None)
    camera_attention_mask = getattr(bundle, "camera_attention_mask", None)

    for i in range(int(args.sample_steps)):
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)
        v_acc = torch.zeros_like(z)
        v_weight = torch.zeros((1, seq_len, 1, 1), device=device, dtype=z.dtype)

        for start, end in windows:
            actual = int(end - start)
            w = cosine_window(actual, device=device, dtype=z.dtype).view(1, actual, 1, 1)
            z_w = z[:, start:end]
            z_splat_w = z_splat_n[:, start:end]
            M_preserve_w = M_preserve[:, start:end]
            M_source_w = M_source[:, start:end]
            M_dest_w = M_dest[:, start:end]
            M_edit_w = M_edit[:, start:end]
            frame_ids_w = frame_ids[:, start:end]
            camera_tokens_w = _slice_time(camera_condition_tokens, start, end, seq_len)
            camera_mask_w = _slice_time(camera_attention_mask, start, end, seq_len)
            scaffold_w = bundle.scaffold_tok[:, start:end]

            out_full = sf(
                z_w, sigma, z_splat_w, scaffold_w,
                M_preserve_w, M_source_w, M_dest_w,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=camera_tokens_w,
                camera_attention_mask=camera_mask_w,
                return_mid=False,
                return_dict=True,
                frame_ids=frame_ids_w,
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit_w,
            )
            pred = out_full["video"]
            if do_cfg:
                out_null = sf(
                    z_w, sigma, z_splat_w, scaffold_w,
                    M_preserve_w, M_source_w, M_dest_w,
                    text_tokens=text_null,
                    text_attention_mask=text_null_mask,
                    camera_condition_tokens=camera_tokens_w,
                    camera_attention_mask=camera_mask_w,
                    return_mid=False,
                    return_dict=True,
                    frame_ids=frame_ids_w,
                    fps=FORMAL_SCENE_FPS,
                    flow_edit_mask=M_edit_w,
                )
                pred_null = out_null["video"]
                pred = pred_null + float(guidance_scale) * (pred - pred_null)
            v = sampler_prediction_to_velocity(sf, pred, z_w, sigma)
            v_acc[:, start:end] += v * w
            v_weight[:, start:end] += w

        v = v_acc / v_weight.clamp_min(1e-6)
        z = masked_flow_euler_step(z, v, step_h, z_splat_n, M_edit)

    return project_masked_flow_state(z, z_splat_n, M_edit)


@torch.no_grad()
def cfg_sample_edit_latents(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    device: torch.device,
    guidance_scale: float,
    seed: int,
    text_encoder: nn.Module | None = None,
    *,
    sliding_window: int | None = None,
    sliding_stride: int | None = None,
) -> torch.Tensor:
    """Conditional rectified-flow sampling from noise -> edited latent.

    Mirrors ``train_scene_flow.cfg_sample_edit_latents`` but keeps the offline
    inference seed explicit. Returns the latent in the model's normalized space.
    """
    sf = unwrap_ddp(scene_flow)
    z_clean_n = sf.normalize(bundle.z_clean.float())
    if sliding_window is not None and int(sliding_window) > 0 and int(z_clean_n.shape[1]) > int(sliding_window):
        return _cfg_sample_edit_latents_sliding(
            scene_flow,
            bundle,
            args,
            device,
            guidance_scale,
            seed,
            text_encoder,
            window=int(sliding_window),
            stride=int(sliding_stride or max(1, int(sliding_window) // 2)),
        )
    t_steps = rae_t_grid(
        num_steps=int(args.sample_steps),
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
        t_eps=scene_flow_t_eps(scene_flow),
    )
    z_splat_n = sf.normalize(bundle.z_splat.float())
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    M_preserve = bundle.M_preserve.to(device=device, dtype=z_clean_n.dtype)
    M_source = bundle.M_source.to(device=device, dtype=z_clean_n.dtype)
    M_dest = bundle.M_dest.to(device=device, dtype=z_clean_n.dtype)
    _, M_edit, _, _ = build_formal_edit_domains(
        bundle,
        args,
        device=device,
        dtype=z_clean_n.dtype,
    )
    batch_size = z_clean_n.shape[0]

    z = torch.empty_like(z_clean_n)
    z.normal_(generator=generator)
    z = project_masked_flow_state(z, z_splat_n, M_edit)
    frame_ids = _bundle_frame_ids(bundle, batch_size=int(batch_size), seq_len=int(z.shape[1]), device=device)

    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(
        text_encoder,
        [""] * batch_size if text_tokens is not None else None,
    )
    do_cfg = abs(float(guidance_scale) - 1.0) > 1e-6

    for i in range(int(args.sample_steps)):
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)
        out_full = sf(
            z, sigma, z_splat_n, bundle.scaffold_tok,
            M_preserve, M_source, M_dest,
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
            camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
            return_mid=False,
            return_dict=True,
            frame_ids=frame_ids,
            fps=FORMAL_SCENE_FPS,
            flow_edit_mask=M_edit,
        )
        pred = out_full["video"]
        if do_cfg:
            out_null = sf(
                z, sigma, z_splat_n, bundle.scaffold_tok,
                M_preserve, M_source, M_dest,
                text_tokens=text_null,
                text_attention_mask=text_null_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                return_mid=False,
                return_dict=True,
                frame_ids=frame_ids,
                fps=FORMAL_SCENE_FPS,
                flow_edit_mask=M_edit,
            )
            pred_null = out_null["video"]
            pred = pred_null + float(guidance_scale) * (pred - pred_null)
        v = sampler_prediction_to_velocity(sf, pred, z, sigma)
        z = masked_flow_euler_step(z, v, step_h, z_splat_n, M_edit)

    return project_masked_flow_state(z, z_splat_n, M_edit)


# ---------------------------------------------------------------------- #
# Latent / mask diagnostics (mirrors save_validation_images)               #
# ---------------------------------------------------------------------- #
def save_edit_grids(
    z_edited_n: torch.Tensor,
    z_clean_n: torch.Tensor,
    out_dir: Path,
    args: argparse.Namespace,
    suffix: str,
    frames: int,
) -> None:
    """Scale-dependent diagnostics on the (stitched) edited latent."""
    grids = {
        f"generated_raw_latent_pca{suffix}": _latent_pca_grid(z_edited_n, args.patch_grid, frames),
        f"abs_error{suffix}": _normalized_mask_grid(
            (z_edited_n - z_clean_n).abs().mean(dim=-1, keepdim=True),
            args.patch_grid, frames,
        ),
    }
    for name, tensor in grids.items():
        save_image_grid(tensor, out_dir / f"{name}.jpg", nrow=frames)


def save_clean_mask_grids(
    z_clean_n: torch.Tensor,
    M_preserve: torch.Tensor,
    M_source: torch.Tensor,
    M_dest: torch.Tensor,
    out_dir: Path,
    args: argparse.Namespace,
    frames: int,
) -> None:
    """Scale-independent constants (stitched clean latent + dual masks)."""
    save_image_grid(
        _latent_pca_grid(z_clean_n, args.patch_grid, frames),
        out_dir / "target_latent_pca.jpg", nrow=frames,
    )
    for name, mask in (
        ("M_preserve", M_preserve),
        ("M_source", M_source),
        ("M_dest", M_dest),
    ):
        save_image_grid(
            _mask_grid(mask, args.patch_grid, frames),
            out_dir / f"{name}.jpg", nrow=frames,
        )


def save_gt_rgb_grid_from_sample(
    sample: dict[str, Any],
    out_dir: Path,
    frames: int,
) -> None:
    """Save cached input/GT RGB frames without requiring 3DGS rendering."""
    if int(frames) <= 0:
        return
    images = sample.get("images", sample.get("images_clean"))
    if not torch.is_tensor(images):
        return
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5:
        raise ValueError(f"Expected GT images [S,3,H,W] or [B,S,3,H,W], got {tuple(images.shape)}")
    save_image_grid(_image_grid(images, frames), out_dir / "input_rgb_gt.jpg", nrow=frames)


# ---------------------------------------------------------------------- #
# Checkpoint loading                                                      #
# ---------------------------------------------------------------------- #
def _scene_flow_prediction_type(scene_flow: nn.Module) -> str:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return str(getattr(cfg, "prediction_type", "x"))


def build_scene_flow_from_checkpoint_config(
    ckpt_path: str | Path,
    *,
    patch_grid: tuple[int, int],
    device: torch.device,
) -> WanSceneFlow:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("scene_flow_config"), dict):
        raise ValueError(
            f"{ckpt_path} has no scene_flow_config; only current formal-editor checkpoints are accepted."
        )
    config = dict(payload["scene_flow_config"])
    if config.get("layout_condition_version") != FORMAL_LAYOUT_CONDITION_VERSION:
        raise ValueError(
            f"{ckpt_path} layout_condition_version={config.get('layout_condition_version')!r}; "
            "the formal editor is a separate layout-free model and will not ignore layout_v2."
        )
    for field, expected in FORMAL_LAYOUT_DISABLED_CONFIG.items():
        if config.get(field) != expected:
            raise ValueError(
                f"{ckpt_path} {field}={config.get(field)!r}; "
                f"layout-free formal checkpoints require {expected!r}"
            )
    if tuple(config.get("patch_grid", ())) != tuple(patch_grid):
        raise ValueError(f"checkpoint patch_grid={config.get('patch_grid')} != cache patch_grid={patch_grid}")
    for derived in (
        "hidden_size",
        "rope_layout_version",
        "sky_rope_temporal_offset",
        "camera_rope_spatial_mode",
    ):
        config.pop(derived, None)
    return WanSceneFlow(**config).to(device)


def _validate_scene_flow_checkpoint_config(scene_flow: nn.Module, payload: Any, path: str | Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a current formal-editor checkpoint")
    saved_cfg = payload.get("scene_flow_config")
    if not isinstance(saved_cfg, dict):
        raise ValueError(f"{path} is missing scene_flow_config")
    if saved_cfg.get("layout_condition_version") != FORMAL_LAYOUT_CONDITION_VERSION:
        raise ValueError(f"{path} is not a layout-free formal-editor checkpoint")
    current_cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    for field in (
        "layout_condition_version",
        *FORMAL_LAYOUT_DISABLED_CONFIG,
        "patch_grid",
        "out_channels",
        "prediction_type",
    ):
        current_value = getattr(current_cfg, field)
        saved_value = saved_cfg[field]
        same = tuple(current_value) == tuple(saved_value) if field == "patch_grid" else current_value == saved_value
        if not same:
            raise ValueError(f"{path} config {field}={saved_value!r} != model {current_value!r}")


def load_scene_flow_ckpt(
    scene_flow: WanSceneFlow,
    assembler: FlowFeatureAssembler,
    ckpt_path: str,
    disable_ema: bool,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{ckpt_path} must be a current formal-editor checkpoint dictionary")
    flow_schedule = resolve_inference_flow_schedule(payload, args, ckpt_path)
    saved_flow_domain = payload.get("formal_flow_domain_version") if isinstance(payload, dict) else None
    if saved_flow_domain != FORMAL_FLOW_DOMAIN_VERSION:
        raise ValueError(
            f"{ckpt_path} formal_flow_domain_version={saved_flow_domain!r}, expected "
            f"{FORMAL_FLOW_DOMAIN_VERSION!r}. Refusing to combine an earlier soft-mask-flow "
            "checkpoint with the corrected binary-domain sampler."
        )
    _validate_scene_flow_checkpoint_config(scene_flow, payload, ckpt_path)
    info: dict[str, Any] = {
        "ckpt_path": ckpt_path,
        "ema_used": False,
        "prediction_type": _scene_flow_prediction_type(scene_flow),
        "checkpoint_prediction_type": payload["scene_flow_config"]["prediction_type"],
        "flow_schedule_config": flow_schedule,
    }

    if not disable_ema and "ema_scene_flow_state_dict" in payload:
        sf_state = payload["ema_scene_flow_state_dict"]
        info["step"] = int(payload.get("step", -1))
        info["ema_used"] = True
        info["source"] = "ema_scene_flow_state_dict"
    elif payload.get("is_ema_weights") and "scene_flow" in payload:
        if disable_ema:
            raise ValueError("--no_ema cannot be used with an EMA-only checkpoint")
        sf_state = payload["scene_flow"]
        info["step"] = int(payload.get("step", -1))
        info["ema_used"] = True
        info["source"] = "ema_weights_only"
    elif "scene_flow" in payload:
        sf_state = payload["scene_flow"]
        info["step"] = int(payload.get("step", -1))
        info["source"] = "scene_flow"
    else:
        raise ValueError(f"{ckpt_path} is missing scene_flow weights")

    scene_flow.load_state_dict(sf_state, strict=True)

    if disable_ema:
        info["ema_note"] = "--no_ema set; using raw scene_flow weights"
        print("[ckpt:scene_flow] --no_ema set: using raw weights.", flush=True)
    elif info["ema_used"]:
        print(f"[ckpt:scene_flow] using EMA weights from {info['source']}.", flush=True)
    else:
        info["ema_note"] = "checkpoint carries only raw scene_flow weights"
        print("[warn] checkpoint carries only raw SceneFlow weights.", flush=True)

    # The formal trainer also trains assembler.scaffold_packer.  Formal
    # inference is invalid without it: using the constructor initialization
    # would change edit-control conditioning while appearing to load cleanly.
    if "scaffold_packer" in payload:
        assembler.scaffold_packer.load_state_dict(payload["scaffold_packer"], strict=True)
    else:
        raise ValueError(
            f"{ckpt_path} does not contain the trained scaffold_packer. "
            "A current formal checkpoint must carry this trainable module."
        )

    scene_flow.to(device).eval()
    return info


def _entry_tag(entry: dict[str, Any], sample: dict[str, Any], cache_path: str) -> str:
    stem = Path(cache_path).stem
    idx = entry.get("index")
    scene = entry.get("scene_name") or sample.get("scene_name")
    variant = entry.get("variant")
    parts: list[str] = []
    if idx is not None:
        parts.append(f"{int(idx):06d}")
    else:
        parts.append(stem)
    if scene:
        parts.append(str(scene))
    if variant:
        parts.append(str(variant))
    return "_".join(parts)


def _item_for_subset(
    dataset: WaymoFlowCacheDataset,
    payload: dict[str, Any],
    entry: dict[str, Any],
    cache_path: Path,
    idx: int,
    subset_t: torch.Tensor,
) -> dict[str, Any]:
    """Build the same item shape as ``WaymoFlowCacheDataset.__getitem__`` for
    an explicit validation window, while preserving the sample needed for RGB
    rendering even when the cache uses the fast SceneFlow path.
    """
    item = dataset._build_item_from_payload(
        payload=payload,
        entry=entry,
        cache_path=cache_path,
        subset_t=subset_t,
        subset_payload=subset_t,
    )
    if "sample" not in item:
        mode_kind = str(payload["mode_kind"])
        sample = dataset._build_sample(payload, subset_t)
        sample["mode_kind"] = mode_kind
        sample["cache_index"] = int(
            entry.get("index", payload.get("meta", {}).get("manifest_index", idx))
        )
        item["sample"] = sample
    variant = entry.get("variant") or payload.get("meta", {}).get("variant")
    if variant is not None:
        item["validation_variant"] = str(variant)
    return item


def _load_formal_offline_payload(
    dataset: WaymoFlowCacheDataset,
    entry: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    """Load a full clip in the same cached representation formal training uses.

    Formal training consumes the precomputed ``flow_inputs`` path.
    Offline inference additionally needs RGB and DGGT pose data for rendering,
    so use the corresponding fast-RGB consumer without recomputing flow
    conditions online.
    """
    if is_chunked_flow_cache(cache_path):
        probe = load_chunked_flow_cache_probe(cache_path)
        dataset._validate_loaded_payload(
            probe,
            cache_path=cache_path,
            entry=entry,
        )
        if not bool(probe.get("_chunked_summary", {}).get("has_flow_inputs", False)):
            raise RuntimeError(
                f"{cache_path} has no precomputed flow_inputs; rebuild it with the "
                "current offline feature pipeline"
            )
        all_frames = torch.arange(int(probe["meta"]["num_frames"]), dtype=torch.long)
        payload = load_chunked_flow_cache_subset(
            cache_path,
            all_frames,
            consumer="scene_flow_fast_rgb",
        )
    else:
        payload = load_flow_cache(
            cache_path,
            map_location="cpu",
            weights_only=False,
            mmap=bool(getattr(dataset, "mmap_plain_cache", True)),
        )
    dataset._validate_loaded_payload(
        payload,
        cache_path=cache_path,
        entry=entry,
    )
    return payload


def _make_batch(
    sample: dict[str, Any],
    device: torch.device,
    *,
    render_pose_enc_dggt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build a renderer batch with the cached full-context DGGT pose slice."""
    return {
        "images": sample["images"].unsqueeze(0).to(device),
        "masks": sample["masks"].unsqueeze(0).to(device),
        "timestamps": sample["timestamps"].unsqueeze(0),
        "render_pose_enc_dggt": render_pose_enc_dggt,
    }


def _save_rgb_grids(
    rgb: dict[str, torch.Tensor],
    out_dir: Path,
    suffix: str,
    frames: int,
    write_refs: bool,
) -> None:
    """Edited grids carry the cfg/window suffix; clean/recon/input refs are
    scale-independent and written once (``write_refs``)."""
    refs = {"input_rgb_gt", "dggt_clean_3dgs_rgb", "tokenizer_recon_3dgs_rgb"}
    for name, tensor in rgb.items():
        if name in refs:
            if write_refs:
                save_image_grid(tensor, out_dir / f"{name}.jpg", nrow=frames)
        else:
            save_image_grid(tensor, out_dir / f"{name}{suffix}.jpg", nrow=frames)

# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #
def main() -> None:
    args = build_argparser().parse_args()
    if args.manifest_path is None and args.cache_root is None:
        raise ValueError("Provide either --cache_root or --manifest_path.")
    if not args.tokenizer_ckpt_path:
        raise ValueError("Formal editor inference requires an explicit --tokenizer_ckpt_path")
    if not args.feature_stats_path:
        raise ValueError("Formal editor inference requires --feature_stats_path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available; 3DGS rendering / VGGT-L will be very slow.",
              flush=True)

    import random

    import numpy as np

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    guidance_scales = [
        float(s.strip()) for s in args.guidance_scales.split(",") if s.strip()
    ]
    if not guidance_scales:
        raise ValueError("--guidance_scales must list at least one scale.")

    mode_filter = (
        [m.strip() for m in args.mode_filter.split(",") if m.strip()]
        if args.mode_filter else None
    )
    # We never call dataset[idx] (the random-subset path) — frames are chosen
    # by the sliding window via _item_for_subset, so min/max_frames here are
    # placeholders that only need to satisfy the constructor's validation.
    dataset = WaymoFlowCacheDataset(
        cache_root=args.cache_root,
        manifest_path=args.manifest_path,
        mode_filter=mode_filter,
        split=args.split,
        min_frames=1,
        max_frames=1,
        seed=args.seed,
        caption_root=None if bool(args.no_text_condition) else args.caption_root,
    )
    patch_grid = _infer_cache_patch_grid(dataset)
    args.patch_grid = (int(patch_grid[0]), int(patch_grid[1]))
    h_splat, w_splat = patch_grid[0] * 4, patch_grid[1] * 4
    print(f"[setup] cache patch_grid={patch_grid} H_splat={h_splat} W_splat={w_splat} "
          f"rows={len(dataset)}", flush=True)

    # The checkpoint is authoritative for all SceneFlow architecture fields.
    # Construct it before dependent
    # modules such as the scaffold packer.
    scene_flow = build_scene_flow_from_checkpoint_config(
        args.scene_flow_ckpt_path,
        patch_grid=patch_grid,
        device=device,
    )
    args.latent_dim = int(scene_flow.config.out_channels)
    args.prediction_type = str(scene_flow.config.prediction_type)
    checkpoint_payload = torch.load(args.scene_flow_ckpt_path, map_location="cpu")

    # Full VGGT (aggregator + dense heads + scene_tokenizer). Formal rendering
    # preserves GT sky RGB directly and does not call the frozen sky_model.
    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.ckpt_path, args.tokenizer_ckpt_path, device
    )

    # Assembler — same construction/freezing as train_scene_flow.py.
    assembler = FlowFeatureAssembler(
        scene_tokenizer=vggt_model.scene_tokenizer,
        patch_grid=patch_grid,
        H_splat=h_splat,
        W_splat=w_splat,
        scaffold_out_dim=int(args.latent_dim),
        tokenizer_window_len=FORMAL_TOKENIZER_WINDOW_LEN,
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    freeze_module(assembler.editor)
    freeze_module(assembler.soft_mask)
    freeze_module(assembler.feature_splatter)
    assembler.eval()

    ckpt_info = load_scene_flow_ckpt(
        scene_flow, assembler, args.scene_flow_ckpt_path, args.no_ema, device, args
    )
    print(f"[ckpt:scene_flow] {ckpt_info}", flush=True)
    validate_formal_flow_domain_config(checkpoint_payload, args, args.scene_flow_ckpt_path)
    load_formal_latent_stats(
        scene_flow,
        args.feature_stats_path,
        token_dim=int(args.latent_dim),
        require_existing_match=True,
    )
    print(
        f"[stats] verified tokenizer latent statistics against the checkpoint: "
        f"{args.feature_stats_path}",
        flush=True,
    )
    scene_flow.eval()
    text_encoder = setup_text_encoder(args, device)

    # Index selection.
    n = len(dataset)
    if args.index is not None:
        indices = [int(args.index)]
    else:
        end = n if args.end is None else min(int(args.end), n)
        indices = list(range(int(args.start), end))
    if not indices:
        raise RuntimeError("No dataset rows selected.")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []

    sf = unwrap_ddp(scene_flow)
    cuda = device.type == "cuda"

    for pos, idx in enumerate(indices, start=1):
        entry = dataset.entries[idx]
        cache_path = Path(entry["cache_path"])
        payload = _load_formal_offline_payload(dataset, entry, cache_path)
        num_frames = int(payload["meta"]["num_frames"])
        effective_window, effective_stride, offline_sliding = resolve_offline_window(
            num_frames,
            int(args.window),
            int(args.window_stride),
            max_single_window=OFFLINE_MAX_SINGLE_WINDOW,
        )
        windows = [
            list(range(start, end))
            for start, end in window_slices(num_frames, effective_window, effective_stride)
        ]

        # Full-clip bundle drives stepwise sliding sampling; model forward still
        # receives only window slices, so conditioning token counts stay bounded.
        all_frames_t = torch.arange(num_frames, dtype=torch.long)
        item_full = _item_for_subset(
            dataset, payload, entry, cache_path, idx, all_frames_t
        )
        render_pose_full = cached_render_pose_from_item(item_full)
        bundle_full, sample_full = build_bundle(item_full, assembler, device)
        ldim_full = int(bundle_full.z_clean.shape[-1])
        if ldim_full != int(args.latent_dim):
            raise ValueError(
                f"bundle latent_dim={ldim_full} != --latent_dim={args.latent_dim}; "
                f"set --latent_dim {ldim_full} to match this cache/tokenizer."
            )
        tag = _entry_tag(entry, sample_full, str(cache_path))
        out_dir = root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        win_root = out_dir / "windows"
        print(f"[{pos}/{len(indices)}] row={idx} frames={num_frames} "
              f"window={effective_window} stride={effective_stride} sliding={offline_sliding} "
              f"windows={[ (w[0], w[-1]) for w in windows ]} -> {out_dir}",
              flush=True)

        z_clean_full = sf.normalize(bundle_full.z_clean.float())
        Mp_full = bundle_full.M_preserve.float()
        Ms_full = bundle_full.M_source.float()
        Md_full = bundle_full.M_dest.float()
        z_edit_full: dict[float, torch.Tensor] = {}
        window_summaries: list[dict[str, Any]] = []
        window_records: list[dict[str, Any]] = []

        for wi, fsel in enumerate(windows):
            a, b = fsel[0], fsel[-1] + 1
            subset_t = torch.tensor(fsel, dtype=torch.long)
            item_w = _item_for_subset(
                dataset, payload, entry, cache_path, idx, subset_t
            )
            bundle_w, sample_w = build_bundle(item_w, assembler, device)

            ldim = int(bundle_w.z_clean.shape[-1])
            if ldim != int(args.latent_dim):
                raise ValueError(
                    f"bundle latent_dim={ldim} != --latent_dim={args.latent_dim}; "
                    f"set --latent_dim {ldim} to match this cache/tokenizer."
                )

            win_dir = win_root / f"win{fsel[0]:03d}_{fsel[-1]:03d}"
            win_dir.mkdir(parents=True, exist_ok=True)
            z_clean_w = sf.normalize(bundle_w.z_clean.float())
            save_clean_mask_grids(
                z_clean_w,
                bundle_w.M_preserve.float(),
                bundle_w.M_source.float(),
                bundle_w.M_dest.float(),
                win_dir,
                args,
                min(int(args.val_log_images), len(fsel)),
            )
            save_gt_rgb_grid_from_sample(
                sample_w,
                win_dir,
                min(int(args.val_log_images), len(fsel)),
            )

            window_summaries.append({
                "window": [fsel[0], fsel[-1]],
                "shift": float(args.shift),
                "latent_shape": list(bundle_w.z_clean.shape),
                "per_scale": [],
            })
            window_records.append({
                "start": a,
                "end": b,
                "frames": list(fsel),
                "sample": sample_w,
                "win_dir": win_dir,
                "summary_index": len(window_summaries) - 1,
            })
            del bundle_w, item_w
            if cuda:
                torch.cuda.empty_cache()

        frames = min(int(args.val_log_images), num_frames)
        save_clean_mask_grids(
            z_clean_full, Mp_full, Ms_full, Md_full, out_dir, args, frames
        )
        save_gt_rgb_grid_from_sample(sample_full, out_dir, frames)

        scale_summaries: list[dict[str, Any]] = []
        for scale in guidance_scales:
            suffix = f"__cfg{scale:g}"
            z_edit_full[scale] = cfg_sample_edit_latents(
                scene_flow,
                bundle_full,
                args,
                device,
                guidance_scale=scale,
                seed=int(args.seed) + idx * 1000,
                text_encoder=text_encoder,
                sliding_window=effective_window,
                sliding_stride=effective_stride,
            )
            save_edit_grids(
                z_edit_full[scale], z_clean_full, out_dir, args, suffix, frames
            )
            for record in window_records:
                a = int(record["start"])
                b = int(record["end"])
                z_edit_w = z_edit_full[scale][:, a:b]
                z_clean_w = z_clean_full[:, a:b]
                window_summaries[int(record["summary_index"])]["per_scale"].append({
                    "guidance_scale": scale,
                    "abs_error_mean": float((z_edit_w - z_clean_w).abs().mean().item()),
                })
                if args.render_per_window and not args.no_render_rgb:
                    if cuda:
                        torch.cuda.empty_cache()
                    rgb = render_validation_rgb_gt_sky(
                        _make_batch(
                            record["sample"],
                            device,
                            render_pose_enc_dggt=render_pose_full[:, a:b],
                        ),
                        vggt_model,
                        scene_flow,
                        z_edit_w,
                        args,
                        device,
                    )
                    _save_rgb_grids(
                        rgb,
                        record["win_dir"],
                        suffix,
                        min(int(args.val_log_images), len(record["frames"])),
                        write_refs=(scale == guidance_scales[0]),
                    )
                    del rgb
                    if cuda:
                        torch.cuda.empty_cache()
            # Default: ONE stitched full-clip render (per-window already done above).
            if not args.no_render_rgb and not args.render_per_window:
                if cuda:
                    torch.cuda.empty_cache()
                rgb = render_validation_rgb_gt_sky(
                    _make_batch(
                        sample_full,
                        device,
                        render_pose_enc_dggt=render_pose_full,
                    ),
                    vggt_model, scene_flow, z_edit_full[scale], args, device,
                )
                _save_rgb_grids(
                    rgb, out_dir, suffix, frames,
                    write_refs=(scale == guidance_scales[0]),
                )
                del rgb
                if cuda:
                    torch.cuda.empty_cache()
            scale_summaries.append({
                "guidance_scale": scale,
                "abs_error_mean": float(
                    (z_edit_full[scale] - z_clean_full).abs().mean().item()
                ),
            })

        summary = {
            "row_index": idx,
            "cache_path": str(cache_path),
            "tag": tag,
            "scene_name": sample_full.get("scene_name"),
            "clip_name": sample_full.get("clip_name"),
            "caption_path": item_full.get("caption_path"),
            "variant": entry.get("variant"),
            "mode_kind": str(payload["mode_kind"]),
            "num_frames": num_frames,
            "frames_rendered": frames,
            "window": effective_window,
            "window_stride": effective_stride,
            "sliding_window_active": offline_sliding,
            "num_windows": len(windows),
            "render_per_window": bool(args.render_per_window),
            "patch_grid": list(args.patch_grid),
            "guidance_scales": guidance_scales,
            "sample_steps": int(args.sample_steps),
            "shift": float(args.shift),
            "prediction_type": str(args.prediction_type),
            "non_edit_clamp": "z_splat_each_step",
            "sliding_sampling": "stepwise_velocity_blend",
            "rgb_compositing": "gsplat_premultiplied_over_background",
            "ckpt": ckpt_info,
            "per_scale_stitched": scale_summaries,
            "windows": window_summaries,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        all_summaries.append({"row_index": idx, "tag": tag, "output_dir": str(out_dir)})
        del z_clean_full, z_edit_full, Mp_full, Ms_full, Md_full, payload, bundle_full, item_full, render_pose_full, window_records
        if cuda:
            torch.cuda.empty_cache()

    (root / "all_summary.json").write_text(
        json.dumps(
            {
                "num_rows": len(all_summaries),
                "scene_flow_ckpt": args.scene_flow_ckpt_path,
                "manifest_path": args.manifest_path,
                "cache_root": args.cache_root,
                "rows": all_summaries,
            },
            indent=2,
            default=str,
        )
    )
    print(f"[done] {len(all_summaries)} rows -> {root}", flush=True)


if __name__ == "__main__":
    main()
