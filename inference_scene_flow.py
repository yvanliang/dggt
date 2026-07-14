"""SceneFlow validation inference — run the trained editor on the offline
validation flow-cache and dump the same visualization set the training code
produces.

This script combines two documented pipelines:

* The **formal-training data path** of ``train_scene_flow.py`` (see
  ``docs/scene_flow_cmd.md`` §3): read the offline v6 flow cache through
  ``WaymoFlowCacheDataset`` and drive ``FlowFeatureAssembler`` per clip to
  build the exact ``FlowFeatureBundle`` the trainer consumes. Its training-time
  visualization op (``train_scene_flow.py:_dump_vis`` ->
  ``dggt.utils.flow_viz.dump_flow_features``) is reproduced verbatim.

* The **validation flow cache** of ``tools/precompute_flow_features_validation.py``
  (see ``docs/flow_cache_validation_cmd.md``): Mode-A-schema v8 chunked-zstd
  SQLite ``.pt`` files,
  canonical layout
  ``{cache_root}/validation/{entry_index:06d}_{edit_name}.pt``,
  consumed unchanged via the validation manifest.

On top of the bundle it runs the trained ``WanSceneFlow`` with classifier-free
guidance (mirroring ``train_scene_flow.py:cfg_sample_edit_latents``: factored
text CFG plus asset/control CFG, with non-edit tokens pinned to ``z_splat`` at
every ODE step), then decodes the edited latent and renders 3DGS with
``train_scene_flow.py:render_validation_rgb_gt_sky`` so formal offline
validation matches training-time T1 validation: generated outputs are
composited with GT sky mask / sky background.

================================ COMMANDS ================================

Environment (see docs/scene_flow_cmd.md §0 and docs/flow_cache_validation_cmd.md):

    conda activate dggt
    export DGGT_CKPT=/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt
    export TOKENIZER_CKPT=/home/dancer/code/dm/dggt/logs/tokenizer_t0_waymo_views1/ckpt/scene_tokenizer_step_014000.pt
    export FEATURE_STATS=logs/scene_flow_pretrain/feature_stats_pretrain.pt
    export VAL_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl
    export SCENE_CAPTION_VAL_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions

Sliding window: the validation clips are 29 frames but WanSceneFlow is trained
on 10-frame windows in both stages. Requests longer than 10 frames are
automatically tiled into at most 8-frame windows with ``--window_stride``;
``--window <= 0`` selects this automatic policy and values above 8 are capped.
Every full window has exactly the resolved effective-window number of frames
(the last start is clamped so the clip tail is
covered). Sampling keeps one full-clip latent state; at each denoising step the
model is run on window slices, window velocities are blended in overlap regions,
and the full latent is updated once. Per-window bundles are still dumped for
diagnostics, while the final 3DGS render uses the full-clip latent.

A) Validation manifest, formal-training (T1) checkpoint, all entries:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow.py \
        --ckpt_path $DGGT_CKPT \
        --tokenizer_ckpt_path $TOKENIZER_CKPT \
        --scene_flow_ckpt_path logs/scene_flow_t1/ckpt/flow_step040000.pt \
        --feature_stats_path $FEATURE_STATS \
        --manifest_path $VAL_MANIFEST \
        --split validation \
        --output_dir runs/scene_flow_val_t1 \
        --window 0 --window_stride 7 \
        --sample_steps 30 --shift 10.0 \
        --guidance_scales 1.0,2.0,4.0 \
        --asset_control_guidance_scale 1.0 \
        --val_log_images 10 \
        --seed 0 --precision bf16

B) Pretrain checkpoint, cache_root scan (no manifest), EMA weights (default):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow.py \
        --ckpt_path $DGGT_CKPT \
        --tokenizer_ckpt_path $TOKENIZER_CKPT \
        --scene_flow_ckpt_path logs/scene_flow_pretrain/ckpt/pretrain_step100000.pt \
        --feature_stats_path $FEATURE_STATS \
        --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
        --split validation \
        --output_dir runs/scene_flow_val_pretrain \
        --start 0 --end 5 \
        --sample_steps 30 --shift 10.0 --guidance_scales 2.0 \
        --window 0 --render_per_window

C) Single entry smoke (entry 0 -> manifest index 0, combined variant):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow.py \
        --ckpt_path $DGGT_CKPT --tokenizer_ckpt_path $TOKENIZER_CKPT \
        --scene_flow_ckpt_path logs/scene_flow_pretrain/ckpt/pretrain_step100000.pt \
        --feature_stats_path $FEATURE_STATS \
        --manifest_path $VAL_MANIFEST --split validation \
        --output_dir /tmp/scene_flow_val_smoke \
        --index 0 --sample_steps 15 --guidance_scales 2.0 \
        --window 0 --no_render_rgb --splat_pca

Notes:
  * ``--scene_flow_ckpt_path`` accepts the formal-training checkpoint
    (``{step, scene_flow, scaffold_packer, ...}`` from ``train_scene_flow.py``)
    or the pretrain full / EMA-only / weights-only checkpoint
    (``{scene_flow[, ema_scene_flow], ...}``,
    ``{scene_flow, ema_scene_flow_state_dict, ...}``, or
    ``{scene_flow, is_ema_weights=True}`` from ``train_scene_flow_pretrain.py``).
  * EMA weights are used BY DEFAULT (pass ``--no_ema`` to use raw weights).
    Per ``docs/scene_flow_cmd.md`` §1.5 / commit 6e2c039f, raw mid-training
    weights produce drastically worse diffusion samples; DiT/SD3/Wan/RAE all
    sample from EMA. IMPORTANT: ``pretrain_step{N}_weights_only.pt`` stores
    ONLY the raw ``scene_flow`` (no EMA) — point at the FULL
    ``pretrain_step{N}.pt`` or ``pretrain_step{N}_ema_weights_only.pt`` to get
    EMA. The script warns if EMA is requested but absent.
  * ``--shift`` defaults to 10.0 to match ``docs/scene_flow_cmd.md`` and the
    current pretrain / T1 training defaults. Flow-matching inference should use
    the same schedule the model was trained on.
  * ``--prediction_type`` defaults to ``x`` to match RAEv2-style SceneFlow
    training. Checkpoints that record a different prediction type are rejected
    before weights are loaded; pass ``--prediction_type v`` only for explicit
    velocity-prediction checkpoints.
  * ``--feature_stats_path`` is optional: the checkpoint already carries
    ``mu_z``/``sigma_z`` buffers. Pass it only to override (same dim as
    ``--latent_dim``; e.g. ``feature_stats_pretrain.pt``).
  * Memory: the default single 29-frame render does one VGGT-L pass on the
    full clip (~25GB free, same as the validation-cache precompute). Use
    ``--render_per_window`` (per-window 10-frame renders) or ``--no_render_rgb``
    if tight.

Output (per entry, under ``{output_dir}/{tag}/``):

    windows/win{a:03d}_{b:03d}/flow_features/...   # per-window, == _dump_vis
    target_latent_pca.jpg                   # stitched 29-frame clean latent PCA
    M_preserve.jpg / M_source.jpg / M_dest.jpg     # stitched
    generated_raw_latent_pca__cfg{S}.jpg    # stitched edited latent PCA
    abs_error__cfg{S}.jpg                   # stitched |z_edited - z_clean|
    dggt_clean_3dgs_rgb.jpg                 # DGGT recon of input (sky bg)
    tokenizer_recon_3dgs_rgb.jpg            # tokenizer round-trip of input
    input_rgb_gt.jpg                        # cached input frames
    generated_raw_3dgs_rgb__cfg{S}.jpg      # EDITED render per guidance scale
    generated_pred_sky_mask__cfg{S}.jpg     # DGGT semantic-head diagnostic only; render uses GT sky
    summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

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
from dggt.utils.feature_stats import checkpoint_sha256, load_into_buffers, validate_camera_stats_provenance
from dggt.utils.flow_cache_io import load_flow_cache
from dggt.utils.flow_viz import dump_flow_features, save_image_grid
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
    _asset_condition_kind_for_model,
    _bundle_frame_ids,
    _infer_cache_patch_grid,
    _slice_asset_time,
    _slice_time,
    build_formal_edit_domains,
    build_flow_bundle as build_train_flow_bundle,
    encode_text_condition,
    freeze_module,
    normalize_asset_latents,
    render_validation_rgb_gt_sky,
    sampler_prediction_to_velocity,
    setup_text_encoder,
    validate_formal_flow_domain_config,
)

# Reuse pretrain latent-grid helpers; RGB rendering uses formal T1 GT-sky helper.
from train_scene_flow_pretrain import (
    DEFAULT_SKY_GRID,
    SKY_TOKEN_DIM,
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
                   help="DGGT checkpoint (aggregator + dense heads + sky_model).")
    p.add_argument("--tokenizer_ckpt_path", type=str, default=None,
                   help="JointSceneTokenizer checkpoint matching scene_flow training "
                        "(omit to use tokenizer weights inside --ckpt_path).")
    p.add_argument("--scene_flow_ckpt_path", type=str, required=True,
                   help="Trained WanSceneFlow checkpoint (T1 or pretrain).")
    p.add_argument("--feature_stats_path", type=str, default=None,
                   help="Optional latent stats override (mu_z/sigma_z). Default: "
                        "use the buffers stored inside --scene_flow_ckpt_path.")
    p.add_argument("--no_ema", action="store_true",
                   help="Use raw weights. By DEFAULT the EMA shadow weights are "
                        "used when the checkpoint carries them (mandatory for "
                        "meaningful diffusion samples; see docs §1.5 / 6e2c039f). "
                        "weights_only checkpoints have no EMA -> raw is used "
                        "with a warning.")
    p.add_argument("--bring_up", action="store_true",
                   help="Build the small bring-up WanSceneFlow config instead of T1.")
    p.add_argument("--latent_dim", type=int, default=1024,
                   help="Tokenizer latent channels (WanSceneFlow out_channels; "
                        "in_channels = 3*latent_dim + 3). Must match training.")
    p.add_argument("--sky_grid_h", type=int, default=DEFAULT_SKY_GRID[0])
    p.add_argument("--sky_grid_w", type=int, default=DEFAULT_SKY_GRID[1])
    p.add_argument(
        "--prediction_type",
        type=str,
        choices=("v", "x"),
        default="x",
        help=(
            "SceneFlow output parameterization. Default 'x' matches RAEv2-style training; "
            "use 'v' only for explicit velocity-prediction checkpoints."
        ),
    )
    p.add_argument("--asset_position_mode", choices=("localized", "canonical"), default="localized")
    p.add_argument("--text_encoder_path", type=str, default="/home/dancer/model/Qwen/Qwen3-0.6B/",
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
    p.add_argument("--sample_steps", type=int, default=30,
                   help="FlowMatch inference steps (15 smoke / 30 normal / 50 stable).")
    p.add_argument("--shift", type=float, default=10.0,
                   help="FlowMatch / RAE time distribution shift used for sampling.")
    p.add_argument("--edit_domain_threshold", type=float, default=1e-4,
                   help="Threshold soft source+destination coverage into the binary flow domain.")
    p.add_argument("--edit_domain_dilation", type=int, default=1,
                   help="Patch-grid dilation radius for the binary flow domain.")
    p.add_argument("--guidance_scales", type=str, default="1.0,2.0",
                   help="Comma-sep text CFG scales; one edited render per scale.")
    p.add_argument("--asset_control_guidance_scale", type=float, default=1.0,
                   help="Factored CFG scale for asset + edit-control conditions.")
    p.add_argument("--cond_norm", type=str, default="zsplat",
                   choices=("zsplat", "all", "none"),
                   help="Deprecated no-op; inference now matches training: z_splat is normalized and scaffold_tok is raw.")
    p.add_argument("--no_preserve_blend", action="store_true",
                   help="Deprecated compatibility flag; non-edit tokens are pinned to z_splat each ODE step.")

    # Visualization (consumed by reused pretrain render helpers)
    p.add_argument("--val_log_images", type=int, default=10,
                   help="Number of frames rendered/tiled per grid.")
    p.add_argument("--splat_pca", action="store_true",
                   help="Also dump splat_pca grids in flow_features/.")
    p.add_argument("--no_flow_tensors", action="store_true",
                   help="Skip flow_features.pt (keep only the JPG grids).")
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
def build_bundle(item: dict[str, Any], assembler: FlowFeatureAssembler, device: torch.device):
    bundle = build_train_flow_bundle(item, assembler, device)
    variant = str(item.get("validation_variant", ""))
    if variant:
        bundle.asset_condition_kind = validation_asset_condition_kind(variant)
    return bundle, item["sample"]


def validation_asset_condition_kind(variant: str) -> str:
    variant = str(variant)
    has_asset = variant in {"combined", "insertion", "replacement", "repositioning"}
    has_delete = variant in {"combined", "deletion", "replacement", "repositioning"}
    if has_asset and has_delete:
        return "mode_a_with_empty"
    if has_delete:
        return "mode_b_empty"
    return "mode_a"


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

    F_asset = normalize_asset_latents(sf, bundle.F_asset_tokens)
    if F_asset.ndim in (4, 5):
        F_uncond = torch.zeros_like(F_asset)
        uncond_asset_mask = torch.zeros(F_asset.shape[:-1], device=F_asset.device, dtype=torch.bool)
    else:
        F_uncond = F_asset.new_zeros((batch_size, 0, F_asset.shape[-1]))
        uncond_asset_mask = None
    encoder_attention_mask = getattr(bundle, "encoder_attention_mask", None)
    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(
        text_encoder,
        [""] * batch_size if text_tokens is not None else None,
    )
    asset_kinds = _asset_condition_kind_for_model(bundle, batch_size)
    asset_control_scale = float(getattr(args, "asset_control_guidance_scale", 1.0))
    do_cfg = (
        abs(float(guidance_scale) - 1.0) > 1e-6
        or abs(asset_control_scale - 1.0) > 1e-6
    )
    drop_all_control = torch.ones((batch_size,), device=device, dtype=torch.bool)
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
            F_asset_w = _slice_asset_time(F_asset, start, end, seq_len)
            asset_mask_w = _slice_asset_time(encoder_attention_mask, start, end, seq_len)
            F_uncond_w = _slice_asset_time(F_uncond, start, end, seq_len)
            uncond_mask_w = _slice_asset_time(uncond_asset_mask, start, end, seq_len)
            scaffold_w = bundle.scaffold_tok[:, start:end]

            v_full = sf(
                z_w, sigma, z_splat_w, scaffold_w,
                M_preserve_w, M_source_w, M_dest_w, F_asset_w,
                encoder_attention_mask=asset_mask_w,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=camera_tokens_w,
                camera_attention_mask=camera_mask_w,
                asset_condition_kind=asset_kinds,
                return_mid=False,
                frame_ids=frame_ids_w,
                fps=None,
                flow_edit_mask=M_edit_w,
            )
            if do_cfg:
                v_text = sf(
                    z_w, sigma, z_splat_w, scaffold_w,
                    M_preserve_w, M_source_w, M_dest_w, F_uncond_w,
                    encoder_attention_mask=uncond_mask_w,
                    text_tokens=text_tokens,
                    text_attention_mask=text_mask,
                    camera_condition_tokens=camera_tokens_w,
                    camera_attention_mask=camera_mask_w,
                    asset_condition_kind=["asset_uncond"] * batch_size,
                    return_mid=False,
                    control_drop_mask=drop_all_control,
                    frame_ids=frame_ids_w,
                    fps=None,
                    flow_edit_mask=M_edit_w,
                )
                v_uncond = sf(
                    z_w, sigma, z_splat_w, scaffold_w,
                    M_preserve_w, M_source_w, M_dest_w, F_uncond_w,
                    encoder_attention_mask=uncond_mask_w,
                    text_tokens=text_null,
                    text_attention_mask=text_null_mask,
                    camera_condition_tokens=camera_tokens_w,
                    camera_attention_mask=camera_mask_w,
                    asset_condition_kind=["asset_uncond"] * batch_size,
                    return_mid=False,
                    control_drop_mask=drop_all_control,
                    frame_ids=frame_ids_w,
                    fps=None,
                    flow_edit_mask=M_edit_w,
                )
                v_pred = (
                    v_uncond
                    + float(guidance_scale) * (v_text - v_uncond)
                    + asset_control_scale * (v_full - v_text)
                )
            else:
                v_pred = v_full
            v = sampler_prediction_to_velocity(sf, v_pred, z_w, sigma)
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

    F_asset = normalize_asset_latents(sf, bundle.F_asset_tokens)
    if F_asset.ndim in (4, 5):
        F_uncond = torch.zeros_like(F_asset)
        uncond_asset_mask = torch.zeros(F_asset.shape[:-1], device=F_asset.device, dtype=torch.bool)
    else:
        F_uncond = F_asset.new_zeros((batch_size, 0, F_asset.shape[-1]))
        uncond_asset_mask = None
    encoder_attention_mask = getattr(bundle, "encoder_attention_mask", None)
    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(
        text_encoder,
        [""] * batch_size if text_tokens is not None else None,
    )
    asset_kinds = _asset_condition_kind_for_model(bundle, batch_size)
    asset_control_scale = float(getattr(args, "asset_control_guidance_scale", 1.0))
    do_cfg = (
        abs(float(guidance_scale) - 1.0) > 1e-6
        or abs(asset_control_scale - 1.0) > 1e-6
    )
    drop_all_control = torch.ones((batch_size,), device=device, dtype=torch.bool)

    for i in range(int(args.sample_steps)):
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)
        v_full = sf(
            z, sigma, z_splat_n, bundle.scaffold_tok,
            M_preserve, M_source, M_dest, F_asset,
            encoder_attention_mask=encoder_attention_mask,
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
            camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
            asset_condition_kind=asset_kinds,
            return_mid=False,
            frame_ids=frame_ids,
            fps=None,
            flow_edit_mask=M_edit,
        )
        if do_cfg:
            v_text = sf(
                z, sigma, z_splat_n, bundle.scaffold_tok,
                M_preserve, M_source, M_dest, F_uncond,
                encoder_attention_mask=uncond_asset_mask,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                asset_condition_kind=["asset_uncond"] * batch_size,
                return_mid=False,
                control_drop_mask=drop_all_control,
                frame_ids=frame_ids,
                fps=None,
                flow_edit_mask=M_edit,
            )
            v_uncond = sf(
                z, sigma, z_splat_n, bundle.scaffold_tok,
                M_preserve, M_source, M_dest, F_uncond,
                encoder_attention_mask=uncond_asset_mask,
                text_tokens=text_null,
                text_attention_mask=text_null_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                asset_condition_kind=["asset_uncond"] * batch_size,
                return_mid=False,
                control_drop_mask=drop_all_control,
                frame_ids=frame_ids,
                fps=None,
                flow_edit_mask=M_edit,
            )
            v = (
                v_uncond
                + float(guidance_scale) * (v_text - v_uncond)
                + asset_control_scale * (v_full - v_text)
            )
        else:
            v = v_full
        v = sampler_prediction_to_velocity(sf, v, z, sigma)
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
def _strip_module(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def _format_key_list(keys: Sequence[str], limit: int = 32) -> str:
    shown = list(keys[:limit])
    suffix = "" if len(keys) <= limit else f", ... (+{len(keys) - limit} more)"
    return "[" + ", ".join(repr(k) for k in shown) + suffix + "]"


def _raise_on_state_dict_mismatch(
    *,
    ckpt_path: str | Path,
    module_name: str,
    source: str,
    missing: Sequence[str],
    unexpected: Sequence[str],
) -> None:
    if not missing and not unexpected:
        return
    parts = [
        f"{ckpt_path} is not compatible with {module_name} when loading {source}.",
    ]
    if missing:
        parts.append(f"missing keys ({len(missing)}): {_format_key_list(missing)}")
    if unexpected:
        parts.append(f"unexpected keys ({len(unexpected)}): {_format_key_list(unexpected)}")
    parts.append("Refusing to run offline inference with a partially loaded checkpoint.")
    raise RuntimeError(" ".join(parts))


def _scene_flow_prediction_type(scene_flow: nn.Module) -> str:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return str(getattr(cfg, "prediction_type", "x"))


def _checkpoint_prediction_type(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    cfg = payload.get("scene_flow_config")
    if isinstance(cfg, dict) and "prediction_type" in cfg:
        return str(cfg["prediction_type"])
    args = payload.get("args")
    if isinstance(args, dict) and "prediction_type" in args:
        return str(args["prediction_type"])
    return None


SCENE_FLOW_CONFIG_COMPAT_FIELDS = (
    "rope_layout_version",
    "rope_theta",
    "encoder_mrope_section",
    "ddt_mrope_section",
    "patch_grid",
    "out_channels",
    "sky_grid",
    "camera_gen_dim",
    "camera_generation_representation",
    "camera_stats_version",
    "camera_condition_representation",
    "mask_compositing_version",
    "asset_position_mode",
    "sky_rope_temporal_offset",
    "camera_rope_spatial_mode",
)


def build_scene_flow_from_checkpoint_config(
    ckpt_path: str | Path,
    *,
    patch_grid: tuple[int, int],
    device: torch.device,
) -> WanSceneFlow:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("scene_flow_config"), dict):
        raise ValueError(
            f"{ckpt_path} has no scene_flow_config; formal offline inference requires a versioned checkpoint."
        )
    config = dict(payload["scene_flow_config"])
    camera_dim = int(config.get("camera_gen_dim", 2048))
    config.setdefault("camera_generation_representation", "dggt_hidden_v1" if camera_dim == 2048 else "relative_se3_rot6d_logfov_v2")
    config.setdefault("asset_position_mode", "localized")
    config.setdefault("mask_compositing_version", "legacy_hard_mask_v1")
    if tuple(config.get("patch_grid", ())) != tuple(patch_grid):
        raise ValueError(f"checkpoint patch_grid={config.get('patch_grid')} != cache patch_grid={patch_grid}")
    return WanSceneFlow(**config).to(device)


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_config_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_normalize_config_value(v) for v in value)
    return value


def _config_values_match(current: Any, saved: Any) -> bool:
    current = _normalize_config_value(current)
    saved = _normalize_config_value(saved)
    if isinstance(current, tuple) and isinstance(saved, tuple):
        return len(current) == len(saved) and all(_config_values_match(c, s) for c, s in zip(current, saved))
    if isinstance(current, float) or isinstance(saved, float):
        try:
            return abs(float(current) - float(saved)) <= 1e-6
        except (TypeError, ValueError):
            return False
    return current == saved


def _validate_scene_flow_checkpoint_config(scene_flow: nn.Module, payload: Any, path: str | Path) -> None:
    _validate_scene_flow_prediction_type(scene_flow, payload, path)
    if not isinstance(payload, dict):
        return
    saved_cfg = payload.get("scene_flow_config")
    if not isinstance(saved_cfg, dict):
        return
    if "rope_layout_version" not in saved_cfg and "mrope_temporal_margin" in saved_cfg:
        raise ValueError(
            f"{path} was saved with the legacy global mrope_temporal_margin RoPE layout. "
            "The current SceneFlow model uses the fixed A1 layout "
            "(video/asset/camera shared video time, camera center, sky temporal offset 128); "
            "do not run inference across these incompatible position semantics."
        )
    current_cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    mismatches: list[str] = []
    for field in SCENE_FLOW_CONFIG_COMPAT_FIELDS:
        if field not in saved_cfg or not hasattr(current_cfg, field):
            continue
        current_value = getattr(current_cfg, field)
        saved_value = saved_cfg[field]
        if not _config_values_match(current_value, saved_value):
            mismatches.append(f"{field}: checkpoint={saved_value!r}, current={current_value!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            f"{path} SceneFlow config does not match the current model: {joined}. "
            "Do not run inference across incompatible RoPE/model geometry settings."
        )


def _validate_scene_flow_prediction_type(scene_flow: nn.Module, payload: Any, path: str | Path) -> None:
    current = _scene_flow_prediction_type(scene_flow)
    saved = _checkpoint_prediction_type(payload)
    if saved is None:
        if current == "v":
            raise ValueError(
                f"{path} does not record SceneFlow prediction_type. Refusing to load it into "
                "a velocity-prediction model because legacy checkpoints were x-prediction by default. "
                "Use --prediction_type x for that checkpoint or use a checkpoint saved with scene_flow_config."
            )
        return
    if saved != current:
        raise ValueError(
            f"{path} prediction_type={saved!r} does not match current model prediction_type={current!r}. "
            "Do not run inference across x-prediction and velocity-prediction checkpoints."
        )


def load_scene_flow_ckpt(
    scene_flow: WanSceneFlow,
    assembler: FlowFeatureAssembler,
    ckpt_path: str,
    disable_ema: bool,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu")
    saved_flow_domain = payload.get("formal_flow_domain_version") if isinstance(payload, dict) else None
    if saved_flow_domain != FORMAL_FLOW_DOMAIN_VERSION:
        raise ValueError(
            f"{ckpt_path} formal_flow_domain_version={saved_flow_domain!r}, expected "
            f"{FORMAL_FLOW_DOMAIN_VERSION!r}. Refusing to combine a legacy soft-mask-flow "
            "checkpoint with the corrected binary-domain sampler."
        )
    _validate_scene_flow_checkpoint_config(scene_flow, payload, ckpt_path)
    info: dict[str, Any] = {
        "ckpt_path": ckpt_path,
        "ema_used": False,
        "prediction_type": _scene_flow_prediction_type(scene_flow),
        "checkpoint_prediction_type": _checkpoint_prediction_type(payload),
    }

    if not disable_ema and isinstance(payload, dict) and "ema_scene_flow_state_dict" in payload:
        sf_state = payload["ema_scene_flow_state_dict"]
        info["step"] = int(payload.get("step", -1))
        info["ema_used"] = True
        info["source"] = "ema_scene_flow_state_dict"
    elif not disable_ema and isinstance(payload, dict) and payload.get("is_ema_weights") and "scene_flow" in payload:
        sf_state = payload["scene_flow"]
        info["step"] = int(payload.get("step", -1))
        info["ema_used"] = True
        info["source"] = "ema_weights_only"
    elif isinstance(payload, dict) and "scene_flow" in payload:
        sf_state = payload["scene_flow"]
        info["step"] = int(payload.get("step", -1))
        info["source"] = "scene_flow"
    elif isinstance(payload, dict) and "state_dict" in payload and "scene_flow" not in payload:
        sf_state = payload["state_dict"]
        info["source"] = "state_dict"
    else:
        sf_state = payload  # raw module state_dict
        info["source"] = "raw_state_dict"

    missing, unexpected = scene_flow.load_state_dict(_strip_module(sf_state), strict=False)
    _raise_on_state_dict_mismatch(
        ckpt_path=ckpt_path,
        module_name="WanSceneFlow",
        source=str(info["source"]),
        missing=missing,
        unexpected=unexpected,
    )
    info["missing"] = len(missing)
    info["unexpected"] = len(unexpected)

    # EMA shadow weights are used BY DEFAULT (pretrain full checkpoints store an
    # EMAModel dict under "ema_scene_flow"). Per docs/scene_flow_cmd.md §1.5 /
    # commit 6e2c039f, raw mid-training weights give drastically worse diffusion
    # samples; validation/inference must run under EMA.
    has_ema = isinstance(payload, dict) and "ema_scene_flow" in payload
    info["ema_in_ckpt"] = bool(has_ema)
    if disable_ema:
        info["ema_note"] = "--no_ema set; using raw scene_flow weights"
        print("[ckpt:scene_flow] --no_ema set: using RAW weights "
              "(diffusion samples will be worse than the model actually is).",
              flush=True)
    elif info["ema_used"]:
        print(f"[ckpt:scene_flow] using EMA weights from {info['source']}.", flush=True)
    elif has_ema:
        try:
            from diffusers.training_utils import EMAModel

            ema = EMAModel(scene_flow.parameters())
            ema.load_state_dict(payload["ema_scene_flow"])
            ema.copy_to(scene_flow.parameters())
            info["ema_used"] = True
            info["source"] = "ema_scene_flow"
            print("[ckpt:scene_flow] using EMA weights.", flush=True)
        except Exception as exc:  # pragma: no cover - best effort
            info["ema_error"] = repr(exc)
            print(f"[warn] EMA load failed ({exc!r}); falling back to raw weights.",
                  flush=True)
    else:
        info["ema_note"] = "no ema_scene_flow in checkpoint"
        is_weights_only = str(ckpt_path).endswith("_weights_only.pt")
        print(
            "[warn] checkpoint has NO ema_scene_flow"
            + (" (this is a *_weights_only.pt which stores ONLY raw weights)"
               if is_weights_only else "")
            + ". Per docs/scene_flow_cmd.md §1.5 / 6e2c039f, raw mid-training "
            "weights produce drastically worse samples. Point "
            "--scene_flow_ckpt_path at the FULL pretrain_step{N}.pt (it carries "
            "ema_scene_flow) for meaningful results.",
            flush=True,
        )

    # The formal trainer also trains assembler.scaffold_packer; load it if present.
    if isinstance(payload, dict) and "scaffold_packer" in payload:
        sp_missing, sp_unexpected = assembler.scaffold_packer.load_state_dict(
            _strip_module(payload["scaffold_packer"]), strict=False
        )
        _raise_on_state_dict_mismatch(
            ckpt_path=ckpt_path,
            module_name="FlowFeatureAssembler.scaffold_packer",
            source="scaffold_packer",
            missing=sp_missing,
            unexpected=sp_unexpected,
        )
        info["scaffold_packer_missing"] = len(sp_missing)
        info["scaffold_packer_unexpected"] = len(sp_unexpected)
    else:
        info["scaffold_packer_missing"] = "not_in_ckpt"

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


def _make_batch(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """render_validation_rgb_gt_sky expects a raw-batch dict (B,S,...) of cached frames."""
    return {
        "images": sample["images"].unsqueeze(0).to(device),
        "masks": sample["masks"].unsqueeze(0).to(device),
        "timestamps": sample["timestamps"].unsqueeze(0),
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
    args.sky_grid = (int(args.sky_grid_h), int(args.sky_grid_w))
    h_splat, w_splat = patch_grid[0] * 4, patch_grid[1] * 4
    print(f"[setup] cache patch_grid={patch_grid} H_splat={h_splat} W_splat={w_splat} "
          f"rows={len(dataset)}", flush=True)

    # The checkpoint is authoritative for all SceneFlow architecture fields,
    # including latent and camera dimensions. Construct it before dependent
    # modules such as the scaffold packer.
    scene_flow = build_scene_flow_from_checkpoint_config(
        args.scene_flow_ckpt_path,
        patch_grid=patch_grid,
        device=device,
    )
    args.latent_dim = int(scene_flow.config.out_channels)
    args.prediction_type = str(scene_flow.config.prediction_type)
    if str(scene_flow.config.asset_position_mode) != str(args.asset_position_mode):
        raise ValueError(
            f"checkpoint asset_position_mode={scene_flow.config.asset_position_mode!r} "
            f"!= --asset_position_mode={args.asset_position_mode!r}"
        )

    # Full VGGT (aggregator + dense heads + sky_model + scene_tokenizer).
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
        editor_kwargs={"use_pose_refine": True},
    ).to(device)
    freeze_module(assembler.editor)
    freeze_module(assembler.soft_mask)
    freeze_module(assembler.feature_splatter)
    assembler.eval()

    ckpt_info = load_scene_flow_ckpt(
        scene_flow, assembler, args.scene_flow_ckpt_path, args.no_ema, device
    )
    print(f"[ckpt:scene_flow] {ckpt_info}", flush=True)
    dggt_sha256 = checkpoint_sha256(args.ckpt_path)
    checkpoint_payload = torch.load(args.scene_flow_ckpt_path, map_location="cpu")
    validate_formal_flow_domain_config(checkpoint_payload, args, args.scene_flow_ckpt_path)
    provenance = checkpoint_payload.get("camera_dggt_provenance") if isinstance(checkpoint_payload, dict) else None
    recorded_hash = provenance.get("dggt_checkpoint_sha256") if isinstance(provenance, dict) else None
    if recorded_hash != dggt_sha256:
        raise ValueError(
            "Formal offline inference DGGT checkpoint does not match SceneFlow provenance: "
            f"checkpoint={recorded_hash!r}, current={dggt_sha256!r}."
        )
    if args.feature_stats_path:
        load_into_buffers(scene_flow, args.feature_stats_path, token_dim=int(args.latent_dim))
        stats_payload = torch.load(args.feature_stats_path, map_location="cpu")
        if not isinstance(stats_payload, dict):
            raise TypeError("feature stats payload must be a dict")
        validate_camera_stats_provenance(stats_payload, dggt_sha256)
        print(f"[stats] overrode mu_z/sigma_z from {args.feature_stats_path}", flush=True)
    elif float(scene_flow.sigma_z.std().item()) < 1e-8 and float(scene_flow.mu_z.abs().max().item()) < 1e-8:
        print("[warn] checkpoint latent stats look like defaults (mu=0, sigma=1); "
              "pass --feature_stats_path if normalization was used in training.",
              flush=True)
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
        payload = load_flow_cache(cache_path, map_location="cpu", weights_only=False)
        WaymoFlowCacheDataset._validate_v6_payload(
            payload, cache_path=cache_path, entry=entry
        )
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
        # receives only window slices, so asset/camera token counts stay bounded.
        all_frames_t = torch.arange(num_frames, dtype=torch.long)
        item_full = _item_for_subset(
            dataset, payload, entry, cache_path, idx, all_frames_t
        )
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
            # Per-window formal-training viz op (train_scene_flow.py:_dump_vis).
            win_flow_summary = dump_flow_features(
                bundle_w, win_dir,
                save_tensors=not args.no_flow_tensors,
                save_masks=True, save_coverage=True, save_scaffold=True,
                save_splat_pca=args.splat_pca,
            )
            save_gt_rgb_grid_from_sample(
                sample_w,
                win_dir,
                min(int(args.val_log_images), len(fsel)),
            )

            window_summaries.append({
                "window": [fsel[0], fsel[-1]],
                "shift": float(args.shift),
                "asset_condition_kind": getattr(bundle_w, "asset_condition_kind", None),
                "flow_features": win_flow_summary,
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
                        _make_batch(record["sample"], device),
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
                    _make_batch(sample_full, device),
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
            "asset_control_guidance_scale": float(args.asset_control_guidance_scale),
            "sample_steps": int(args.sample_steps),
            "shift": float(args.shift),
            "prediction_type": str(args.prediction_type),
            "non_edit_clamp": "z_splat_each_step",
            "legacy_clean_preserve_blend": False,
            "sliding_sampling": "stepwise_velocity_blend",
            "rgb_compositing": "gsplat_premultiplied_over_background",
            "ckpt": ckpt_info,
            "per_scale_stitched": scale_summaries,
            "windows": window_summaries,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        all_summaries.append({"row_index": idx, "tag": tag, "output_dir": str(out_dir)})
        del z_clean_full, z_edit_full, Mp_full, Ms_full, Md_full, payload, bundle_full, item_full, window_records
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
