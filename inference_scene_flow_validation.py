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
  (see ``docs/flow_cache_validation_cmd.md``): Mode-A-schema v6 ``.pt`` files,
  flat layout ``{cache_root}/validation/{index:06d}.pt`` with
  ``index = entry_index*5 + variant_ord`` (combined/delete/add/replace/move),
  consumed unchanged via the validation manifest.

On top of the bundle it runs the trained ``WanSceneFlow`` with classifier-free
guidance (mirroring ``train_scene_flow_pretrain.py:cfg_sample_pretrain_latents``,
but conditioned on the assembler's real ``z_splat`` / ``scaffold_tok`` / dual
masks / asset KV instead of the pretrain full-scene zeros), then decodes the
edited latent and renders 3DGS exactly like
``train_scene_flow_pretrain.py:render_validation_rgb`` (clean / edited /
tokenizer-recon / input-GT grids, latent PCA, abs-error, mask grids).

================================ COMMANDS ================================

Environment (see docs/scene_flow_cmd.md §0 and docs/flow_cache_validation_cmd.md):

    conda activate dggt
    export DGGT_CKPT=/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt
    export TOKENIZER_CKPT=/home/dancer/code/dm/dggt/logs/tokenizer_t0_waymo_views1/ckpt/scene_tokenizer_step_014000.pt
    export FEATURE_STATS=logs/scene_flow_pretrain/feature_stats_pretrain.pt
    export VAL_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl

Sliding window: the validation clips are 29 frames but WanSceneFlow is trained
on short windows (pretrain S=8 / T1 4-8). The clip is tiled into ``--window``
(default 8) frame windows with stride ``--window_stride``; every window has
exactly ``--window`` frames (the last start is clamped so the clip tail is
covered, overlap is averaged). Each window is sampled independently (its own
``FlowFeatureAssembler`` bundle, exactly how the trainer builds bundles from
random windows of the same cache); the per-window edited latents are stitched
back into a full 29-frame latent for the diagnostics and the final 3DGS render.

A) Validation manifest, formal-training (T1) checkpoint, all entries:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow_validation.py \
        --ckpt_path $DGGT_CKPT \
        --tokenizer_ckpt_path $TOKENIZER_CKPT \
        --scene_flow_ckpt_path logs/scene_flow_t1/ckpt/flow_step040000.pt \
        --feature_stats_path $FEATURE_STATS \
        --manifest_path $VAL_MANIFEST \
        --split validation \
        --output_dir runs/scene_flow_val_t1 \
        --window 8 --window_stride 8 \
        --sample_steps 30 --shift 3.0 \
        --guidance_scales 1.0,2.0,4.0 \
        --val_log_images 8 \
        --seed 0 --precision bf16

B) Pretrain checkpoint, cache_root scan (no manifest), EMA weights (default):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow_validation.py \
        --ckpt_path $DGGT_CKPT \
        --tokenizer_ckpt_path $TOKENIZER_CKPT \
        --scene_flow_ckpt_path logs/scene_flow_pretrain/ckpt/pretrain_step100000.pt \
        --feature_stats_path $FEATURE_STATS \
        --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
        --split validation \
        --output_dir runs/scene_flow_val_pretrain \
        --start 60 --end 65 \
        --sample_steps 30 --shift 3.0 --guidance_scales 2.0 \
        --window 8 --render_per_window

C) Single entry smoke (entry 12 -> manifest index 60, combined variant):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u inference_scene_flow_validation.py \
        --ckpt_path $DGGT_CKPT --tokenizer_ckpt_path $TOKENIZER_CKPT \
        --scene_flow_ckpt_path logs/scene_flow_pretrain/ckpt/pretrain_step100000.pt \
        --feature_stats_path $FEATURE_STATS \
        --manifest_path $VAL_MANIFEST --split validation \
        --output_dir /tmp/scene_flow_val_smoke \
        --index 60 --sample_steps 15 --guidance_scales 2.0 \
        --window 8 --no_render_rgb --splat_pca

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
  * ``--shift`` defaults to 3.0 to match the current training schedule
    (6e2c039f lowered it from ~11; flow-matching inference should use the
    same schedule the model was trained on).
  * ``--feature_stats_path`` is optional: the checkpoint already carries
    ``mu_z``/``sigma_z`` buffers. Pass it only to override (same dim as
    ``--latent_dim``; e.g. ``feature_stats_pretrain.pt``).
  * Memory: the default single 29-frame render does one VGGT-L pass on the
    full clip (~25GB free, same as the validation-cache precompute). Use
    ``--render_per_window`` (per-window 8-frame renders) or ``--no_render_rgb``
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
    generated_pred_sky_mask__cfg{S}.jpg     # predicted sky mask (no GT leak)
    summary.json
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
    _empty_asset_pass,
)
from dggt.models.flow_feature_assembler import FlowFeatureAssembler
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.feature_stats import load_into_buffers
from dggt.utils.flow_cache_io import load_flow_cache
from dggt.utils.flow_viz import dump_flow_features, save_image_grid

# Reuse the formal-training (train_scene_flow.py) cache->bundle helpers verbatim
# so the bundle is byte-for-byte what the trainer feeds the model.
from train_scene_flow import (
    _infer_cache_patch_grid,
    _move_asset_pass,
    _move_mode_b,
    _move_predictions,
    _move_v6_fast_path_inputs,
    _validate_item_patch_grid,
    freeze_module,
)

# Reuse the pretrain (train_scene_flow_pretrain.py) model + render helpers so the
# RGB / latent diagnostics match the documented training-time visualization.
from train_scene_flow_pretrain import (
    _latent_pca_grid,
    _mask_grid,
    _normalized_mask_grid,
    load_dggt_aggregator_and_tokenizer,
    render_validation_rgb,
    unwrap_ddp,
)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


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

    # Data (mirrors train_scene_flow.py / WaymoFlowCacheDataset)
    p.add_argument("--cache_root", type=str, default=None,
                   help="Validation cache root; flat {cache_root}/{split}/*.pt.")
    p.add_argument("--manifest_path", type=str, default=None,
                   help="validation_manifest.jsonl from build_flow_validation_manifest.py.")
    p.add_argument("--mode_filter", type=str, default=None,
                   help="Restrict manifest to comma-sep modes (validation is mode_a).")
    p.add_argument("--split", type=str, default="validation")

    # Sliding window over the (29-frame) clip; WanSceneFlow is trained on
    # short windows so each window is sampled independently then stitched.
    p.add_argument("--window", type=int, default=8,
                   help="Frames per scene_flow window (match training S; "
                        "pretrain S=8 / T1 4-8).")
    p.add_argument("--window_stride", type=int, default=8,
                   help="Window step in frames (==window: non-overlapping tiles; "
                        "<window: overlap, averaged on stitch).")

    # Selection
    p.add_argument("--index", type=int, default=None,
                   help="Single dataset row index. If omitted, process [start,end).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)

    # Sampling
    p.add_argument("--sample_steps", type=int, default=30,
                   help="FlowMatch inference steps (15 smoke / 30 normal / 50 stable).")
    p.add_argument("--shift", type=float, default=3.0,
                   help="FlowMatch schedule shift. Default 3.0 matches the current "
                        "training schedule (commit 6e2c039f lowered it from ~11); "
                        "inference should use the shift the model was trained on.")
    p.add_argument("--guidance_scales", type=str, default="1.0,2.0",
                   help="Comma-sep CFG scales; one edited render per scale. For "
                        "conditional T1 editing s~=1 is the converged optimum; "
                        "s>1 mostly helps undertrained / unconditional models.")
    p.add_argument("--cond_norm", type=str, default="zsplat",
                   choices=("zsplat", "all", "none"),
                   help="Which conditioning latents to normalize with the model's "
                        "mu_z/sigma_z. 'zsplat' (default): z_splat only (it is a "
                        "tokenizer latent like z_clean). 'all': also scaffold_tok. "
                        "'none': pass both raw.")
    p.add_argument("--no_preserve_blend", action="store_true",
                   help="Do not pin M_preserve tokens back to the clean latent.")

    # Visualization (consumed by reused pretrain render helpers)
    p.add_argument("--val_log_images", type=int, default=8,
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
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item["sample"].items()}
    predictions = _move_predictions(item["predictions"], device)
    asset_pass_result = _move_asset_pass(item["asset_pass_result"], device)
    _validate_item_patch_grid(asset_pass_result, assembler, item.get("cache_path"))
    cameras_dggt = {k: v.to(device) for k, v in item["cameras_dggt"].items()}
    mode_kind = str(item.get("mode_kind", sample.get("mode_kind", "mode_a")))
    mode_b_payload = item.get("mode_b")
    if mode_b_payload is not None:
        mode_b_payload = _move_mode_b(mode_b_payload, device)
    phase1_localized_lite, splatted_tok_low_cached = _move_v6_fast_path_inputs(
        item, mode_kind, device
    )
    bundle = assembler(
        sample=sample,
        predictions=predictions,
        asset_pass_result=asset_pass_result,
        cameras_dggt=cameras_dggt,
        object_slots_spec="all",
        base_t=None,
        device=device,
        mode_kind=mode_kind,
        mode_b=mode_b_payload,
        phase1_localized_lite=phase1_localized_lite,
        splatted_tok_low_cached=splatted_tok_low_cached,
    )
    return bundle, sample


# ---------------------------------------------------------------------- #
# CFG editing sampler                                                     #
# ---------------------------------------------------------------------- #
@torch.no_grad()
def cfg_sample_edit_latents(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    device: torch.device,
    guidance_scale: float,
    seed: int,
) -> torch.Tensor:
    """Conditional rectified-flow sampling from pure noise -> edited latent.

    Mirrors ``train_scene_flow_pretrain.cfg_sample_pretrain_latents`` but uses
    the assembler's real edit conditioning (``z_splat`` / ``scaffold_tok`` /
    ``M_preserve`` / ``M_source`` / ``M_dest`` / asset KV). Returns the latent
    in the model's NORMALIZED space (``render_validation_rgb`` denormalizes it).
    """
    sf = unwrap_ddp(scene_flow)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=float(args.shift),
        invert_sigmas=True,
    )
    scheduler.set_timesteps(num_inference_steps=int(args.sample_steps), device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    z_clean_n = sf.normalize(bundle.z_clean.float())
    z_splat_in = bundle.z_splat.float()
    scaffold_in = bundle.scaffold_tok.float()
    if args.cond_norm in ("zsplat", "all"):
        z_splat_in = sf.normalize(z_splat_in)
    if args.cond_norm == "all":
        scaffold_in = sf.normalize(scaffold_in)

    M_preserve = bundle.M_preserve
    M_source = bundle.M_source
    M_dest = bundle.M_dest
    F_asset = bundle.F_asset_tokens
    batch_size = z_clean_n.shape[0]

    z = torch.empty_like(z_clean_n)
    z.normal_(generator=generator)

    F_uncond = F_asset.new_zeros((batch_size, 0, F_asset.shape[-1]))
    do_cfg = abs(float(guidance_scale) - 1.0) > 1e-6 and F_asset.shape[1] > 0

    for timestep in scheduler.timesteps:
        sigma = (timestep / scheduler.config.num_train_timesteps).to(device=device)
        sigma = sigma.expand(batch_size)
        v_cond = sf(
            z, sigma, z_splat_in, scaffold_in,
            M_preserve, M_source, M_dest, F_asset,
            encoder_attention_mask=None, return_mid=False,
        )
        if do_cfg:
            v_uncond = sf(
                z, sigma, z_splat_in, scaffold_in,
                M_preserve, M_source, M_dest, F_uncond,
                encoder_attention_mask=None, return_mid=False,
            )
            v = v_uncond + float(guidance_scale) * (v_cond - v_uncond)
        else:
            v = v_cond
        z = scheduler.step(model_output=v, timestep=timestep, sample=z, return_dict=False)[0]

    if not args.no_preserve_blend:
        z = M_preserve * z_clean_n + (1.0 - M_preserve) * z
    return z


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


# ---------------------------------------------------------------------- #
# Checkpoint loading                                                      #
# ---------------------------------------------------------------------- #
def _strip_module(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def load_scene_flow_ckpt(
    scene_flow: WanSceneFlow,
    assembler: FlowFeatureAssembler,
    ckpt_path: str,
    disable_ema: bool,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu")
    info: dict[str, Any] = {"ckpt_path": ckpt_path, "ema_used": False}

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


# ---------------------------------------------------------------------- #
# Sliding window over the clip                                            #
# ---------------------------------------------------------------------- #
def _window_starts(n: int, window: int, stride: int) -> list[list[int]]:
    """Tile ``n`` frames into windows of exactly ``window`` frames.

    The last start is clamped to ``n - window`` so the clip tail is covered
    (the overlap is averaged during stitching). If ``n <= window`` a single
    full-clip window is returned.
    """
    w = min(int(window), int(n))
    if w <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if n <= w:
        return [list(range(n))]
    step = max(1, int(stride))
    starts = list(range(0, n - w + 1, step))
    if starts[-1] != n - w:
        starts.append(n - w)
    return [list(range(s, s + w)) for s in starts]


def _item_for_subset(
    dataset: WaymoFlowCacheDataset,
    payload: dict[str, Any],
    entry: dict[str, Any],
    cache_path: Path,
    idx: int,
    subset_t: torch.Tensor,
) -> dict[str, Any]:
    """Reproduce ``WaymoFlowCacheDataset.__getitem__`` for an EXPLICIT frame
    subset (no random sampling, payload loaded once by the caller).

    Uses the dataset's own ``_build_*`` / ``_subset_*`` methods so every
    per-frame tensor — including the flattened asset KV, cameras,
    phase1_localized and pass2 splat cache — is sliced consistently, exactly
    as the trainer does for its random windows.
    """
    mode_kind = str(payload["mode_kind"])
    meta = payload["meta"]
    sample = dataset._build_sample(payload, subset_t)
    sample["mode_kind"] = mode_kind
    sample["cache_index"] = int(
        entry.get("index", payload.get("meta", {}).get("manifest_index", idx))
    )
    predictions = dataset._build_predictions(payload, subset_t)
    if mode_kind == "mode_a":
        asset_pass_result = dataset._build_asset_pass(payload, subset_t)
        mode_b_block = None
    else:
        patch_grid = tuple(int(v) for v in meta["patch_grid"])
        asset_pass_result = _empty_asset_pass(patch_grid, int(meta["patch_start_idx"]))
        mode_b_block = dataset._build_mode_b(payload, subset_t)
    cameras_dggt = dataset._build_cameras_dggt(payload, subset_t)
    alignment = dataset._build_alignment(payload)

    phase1_localized_subset = None
    if mode_kind == "mode_a":
        phase1_payload = payload.get("phase1_localized")
        if phase1_payload is None:
            raise RuntimeError(f"Mode-A cache {cache_path} missing phase1_localized.")
        phase1_localized_subset = dataset._subset_phase1_localized(phase1_payload, subset_t)
    pass2_payload = payload.get("pass2_splatted_tok_low")
    if pass2_payload is None:
        raise RuntimeError(f"Cache {cache_path} missing pass2_splatted_tok_low.")
    splatted_tok_low_cached = dataset._subset_pass2_splatted_tok_low(
        pass2_payload, subset_t, dtype=dataset.lut_dtype
    )
    return {
        "sample": sample,
        "predictions": predictions,
        "asset_pass_result": asset_pass_result,
        "cameras_dggt": cameras_dggt,
        "alignment": alignment,
        "mode_kind": mode_kind,
        "mode_b": mode_b_block,
        "subset_frames": subset_t,
        "cache_path": str(cache_path),
        "phase1_localized": phase1_localized_subset,
        "splatted_tok_low_cached": splatted_tok_low_cached,
        "cache_schema_version": int(payload["schema_version"]),
    }


def _make_batch(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """render_validation_rgb expects a raw-batch dict (B,S,...) of cached frames."""
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
    )
    patch_grid = _infer_cache_patch_grid(dataset)
    args.patch_grid = (int(patch_grid[0]), int(patch_grid[1]))
    h_splat, w_splat = patch_grid[0] * 4, patch_grid[1] * 4
    print(f"[setup] cache patch_grid={patch_grid} H_splat={h_splat} W_splat={w_splat} "
          f"rows={len(dataset)}", flush=True)

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

    # WanSceneFlow — same config as train_scene_flow_pretrain.py.
    sf_in_channels = 3 * int(args.latent_dim) + 3
    scene_flow = WanSceneFlow.from_scene_config(
        bring_up=args.bring_up,
        patch_grid=patch_grid,
        in_channels=sf_in_channels,
        out_channels=int(args.latent_dim),
    ).to(device)
    ckpt_info = load_scene_flow_ckpt(
        scene_flow, assembler, args.scene_flow_ckpt_path, args.no_ema, device
    )
    print(f"[ckpt:scene_flow] {ckpt_info}", flush=True)
    if args.feature_stats_path:
        load_into_buffers(scene_flow, args.feature_stats_path, token_dim=int(args.latent_dim))
        print(f"[stats] overrode mu_z/sigma_z from {args.feature_stats_path}", flush=True)
    elif float(scene_flow.sigma_z.std().item()) < 1e-8 and float(scene_flow.mu_z.abs().max().item()) < 1e-8:
        print("[warn] checkpoint latent stats look like defaults (mu=0, sigma=1); "
              "pass --feature_stats_path if normalization was used in training.",
              flush=True)
    scene_flow.eval()

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
        windows = _window_starts(num_frames, args.window, args.window_stride)

        # Full-clip sample (no model) for naming + the stitched final render.
        all_frames_t = torch.arange(num_frames, dtype=torch.long)
        sample_full = dataset._build_sample(payload, all_frames_t)
        tag = _entry_tag(entry, sample_full, str(cache_path))
        out_dir = root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        win_root = out_dir / "windows"
        print(f"[{pos}/{len(indices)}] row={idx} frames={num_frames} "
              f"windows={[ (w[0], w[-1]) for w in windows ]} -> {out_dir}",
              flush=True)

        # Lazily-allocated full-clip stitch buffers (NORMALIZED latent space).
        z_clean_full: torch.Tensor | None = None
        z_edit_full: dict[float, torch.Tensor] = {}
        Mp_full = Ms_full = Md_full = None
        cnt = torch.zeros(num_frames, device=device)
        window_summaries: list[dict[str, Any]] = []

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

            B = int(bundle_w.z_clean.shape[0])
            P = int(bundle_w.z_clean.shape[2])
            if z_clean_full is None:
                z_clean_full = torch.zeros(B, num_frames, P, ldim, device=device)
                Mp_full = torch.zeros(B, num_frames, P, 1, device=device)
                Ms_full = torch.zeros(B, num_frames, P, 1, device=device)
                Md_full = torch.zeros(B, num_frames, P, 1, device=device)

            z_clean_full[:, a:b] += sf.normalize(bundle_w.z_clean.float())
            Mp_full[:, a:b] += bundle_w.M_preserve.float()
            Ms_full[:, a:b] += bundle_w.M_source.float()
            Md_full[:, a:b] += bundle_w.M_dest.float()
            cnt[a:b] += 1.0

            win_scales: list[dict[str, Any]] = []
            for scale in guidance_scales:
                z_edit_w = cfg_sample_edit_latents(
                    scene_flow, bundle_w, args, device,
                    guidance_scale=scale,
                    seed=int(args.seed) + idx * 1000 + wi,
                )
                if scale not in z_edit_full:
                    z_edit_full[scale] = torch.zeros(
                        B, num_frames, P, ldim, device=device
                    )
                z_edit_full[scale][:, a:b] += z_edit_w

                if args.render_per_window and not args.no_render_rgb:
                    if cuda:
                        torch.cuda.empty_cache()
                    rgb = render_validation_rgb(
                        _make_batch(sample_w, device),
                        vggt_model, scene_flow, z_edit_w, args, device,
                    )
                    _save_rgb_grids(
                        rgb, win_dir, f"__cfg{scale:g}",
                        min(int(args.val_log_images), len(fsel)),
                        write_refs=(scale == guidance_scales[0]),
                    )
                    del rgb
                    if cuda:
                        torch.cuda.empty_cache()
                win_scales.append({
                    "guidance_scale": scale,
                    "abs_error_mean": float(
                        (z_edit_w - sf.normalize(bundle_w.z_clean.float()))
                        .abs().mean().item()
                    ),
                })

            window_summaries.append({
                "window": [fsel[0], fsel[-1]],
                "flow_features": win_flow_summary,
                "per_scale": win_scales,
            })
            del bundle_w, sample_w, item_w
            if cuda:
                torch.cuda.empty_cache()

        # Average the window overlap so every frame is in [stitched] exactly once.
        denom = cnt.clamp_min(1.0).view(1, num_frames, 1, 1)
        z_clean_full = z_clean_full / denom
        Mp_full = Mp_full / denom
        Ms_full = Ms_full / denom
        Md_full = Md_full / denom
        for scale in z_edit_full:
            z_edit_full[scale] = z_edit_full[scale] / denom

        frames = min(int(args.val_log_images), num_frames)
        save_clean_mask_grids(
            z_clean_full, Mp_full, Ms_full, Md_full, out_dir, args, frames
        )

        scale_summaries: list[dict[str, Any]] = []
        for scale in guidance_scales:
            suffix = f"__cfg{scale:g}"
            save_edit_grids(
                z_edit_full[scale], z_clean_full, out_dir, args, suffix, frames
            )
            # Default: ONE stitched full-clip render (per-window already done above).
            if not args.no_render_rgb and not args.render_per_window:
                if cuda:
                    torch.cuda.empty_cache()
                rgb = render_validation_rgb(
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
            "variant": entry.get("variant"),
            "mode_kind": str(payload["mode_kind"]),
            "num_frames": num_frames,
            "frames_rendered": frames,
            "window": int(args.window),
            "window_stride": int(args.window_stride),
            "num_windows": len(windows),
            "render_per_window": bool(args.render_per_window),
            "patch_grid": list(args.patch_grid),
            "guidance_scales": guidance_scales,
            "sample_steps": int(args.sample_steps),
            "shift": float(args.shift),
            "cond_norm": args.cond_norm,
            "preserve_blend": not args.no_preserve_blend,
            "ckpt": ckpt_info,
            "per_scale_stitched": scale_summaries,
            "windows": window_summaries,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        all_summaries.append({"row_index": idx, "tag": tag, "output_dir": str(out_dir)})
        del z_clean_full, z_edit_full, Mp_full, Ms_full, Md_full, payload
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
