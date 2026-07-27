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
        --num_frames 10 \
        --start 0 --end 8 \
        --output_dir runs/scene_flow_pretrain_inference

Each sample gets one directory whose name includes its condition mode.  In
addition to training-validation image grids (except ``abs_error``), each CFG
result contains:

* ``generated_raw_gaussians__cfg*_frame*.ply``: per-frame DGGT/3DGS Gaussian
  PLY schema;
* ``generated_raw_points__cfg*_frame*.ply``: per-frame xyz + uchar RGB,
  directly viewable in MeshLab.

The PLYs are built by DGGT's canonical ``build_clean_scene_state`` and written
by its shared PLY writers.  Consequently they use the same non-sky + valid-depth
selection, world-coordinate unprojection, Gaussian activations and serialization
as ``inference_scene_editor.py``.  As in DGGT, frames are exported separately
so overlapping input frames do not visually over-densify a single point cloud;
no unrequested fusion heuristic is applied.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
from dggt.models.canonical_asset_encoder import CanonicalAssetEncoder
from dggt.losses.rgb_render_loss import decode_generated_dggt_geometry
from dggt.utils.feature_stats import checkpoint_sha256, load_all_stats_into_buffers, load_into_buffers
from dggt.utils.camera_generation import CAMERA_GENERATION_DIM, CAMERA_GENERATION_REPRESENTATION
from dggt.utils.camera_condition import camera_summary_from_waymo_gt
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.flow_schedule import resolve_inference_flow_schedule
from dggt.utils.gaussian_edit import CleanSceneState, build_clean_scene_state
from dggt.utils.gaussian_ply import write_gaussian_ply, write_point_ply
from dggt.utils.sliding_window import resolve_offline_window, window_slices
from dggt.utils.factorized_asset_condition import (
    FACTORIZED_ASSET_CONDITION_VERSION,
    FactorizedAssetCondition,
    build_factorized_asset_condition,
    canonicalize_asset_reference,
    interpolate_box_keyframes,
)
from train_scene_flow_pretrain import (
    DEFAULT_SKY_GRID,
    SKY_TOKEN_DIM,
    CyclicSequentialSampler,
    _fixed_render_hw,
    _image_grid,
    _latent_pca_grid,
    _mask_grid,
    _predict_camera_mats,
    _render_gs_map_rgb,
    _sky_background_image_grid,
    _sky_mask_image_grid,
    _sky_mask_patch_to_image,
    _timestamps_for_generated_render,
    autocast_context,
    build_factorized_asset_condition_from_batch,
    build_pretrain_bundle_from_batch,
    build_full_scene_bundle,
    cfg_sample_pretrain_latents,
    decode_pose_from_camera_features,
    decode_sky_patch_tokens,
    discover_scene_names,
    load_dggt_aggregator_and_tokenizer,
    seed_everything,
    setup_text_encoder,
    sky_generation_enabled,
    sky_grid_shape,
    sky_tokens_to_background,
    validate_scene_flow_checkpoint_config,
    unwrap_ddp,
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
        factorized = getattr(bundle, "factorized_asset_condition", None)
        if isinstance(factorized, FactorizedAssetCondition):
            bundle.factorized_asset_condition = factorized.drop_rows(
                torch.ones((batch_size,), device=factorized.appearance_mask.device, dtype=torch.bool)
            )
        by_window = getattr(bundle, "factorized_asset_conditions_by_window", None)
        if by_window is not None:
            bundle.factorized_asset_conditions_by_window = {
                key: condition.drop_rows(
                    torch.ones(
                        (batch_size,),
                        device=condition.appearance_mask.device,
                        dtype=torch.bool,
                    )
                )
                for key, condition in by_window.items()
            }

    if not use_camera:
        # Do not pass GT-derived pose summaries at all.  The sampler inserts one
        # learned camera-null token per generated frame via camera_condition_kind.
        bundle.camera_condition_tokens = None
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
    factorized = getattr(bundle, "factorized_asset_condition", None)
    if isinstance(factorized, FactorizedAssetCondition):
        asset_rows = (
            factorized.appearance_mask.any(dim=(1, 2))
            & factorized.track_valid.any(dim=(1, 2))
        ).detach().cpu().tolist()
    else:
        asset_rows = _row_has_any(getattr(bundle, "encoder_attention_mask", None), batch_size)
    by_window = getattr(bundle, "factorized_asset_conditions_by_window", None)
    if by_window:
        window_rows = [False] * batch_size
        for condition in by_window.values():
            rows = (
                condition.appearance_mask.any(dim=(1, 2))
                & condition.track_valid.any(dim=(1, 2))
            ).detach().cpu().tolist()
            window_rows = [
                bool(window_rows[row]) or bool(rows[row])
                for row in range(batch_size)
            ]
        asset_rows = [
            bool(asset_rows[row]) or bool(window_rows[row])
            for row in range(batch_size)
        ]
    asset_kinds = getattr(bundle, "asset_condition_kind", None)
    if isinstance(asset_kinds, str):
        asset_kinds = [asset_kinds] * batch_size
    if asset_kinds is not None:
        asset_rows = [
            bool(row) and str(asset_kinds[idx]).lower() not in {"none", "asset_uncond", "asset_null"}
            for idx, row in enumerate(asset_rows)
        ]
    camera_tokens = getattr(bundle, "camera_condition_tokens", None)
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


def _load_external_rgba(
    image_path: Path,
    mask_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).contiguous()
    embedded_alpha = torch.from_numpy(rgba[..., 3]).unsqueeze(0).contiguous()
    if mask_path is not None:
        mask_array = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        alpha = torch.from_numpy(mask_array).unsqueeze(0).contiguous()
    elif image.mode in ("RGBA", "LA") or "transparency" in image.info:
        alpha = embedded_alpha
    else:
        raise ValueError(
            f"RGB asset {image_path} requires an explicit `mask`; "
            "automatic foreground extraction is outside the model protocol."
        )
    if tuple(alpha.shape[-2:]) != tuple(rgb.shape[-2:]):
        raise ValueError(f"asset mask {mask_path} does not match image {image_path}")
    return rgb, alpha


def _object_to_anchor_from_center_yaw(
    centers: torch.Tensor,
    yaws: torch.Tensor,
) -> torch.Tensor:
    if centers.ndim != 3 or yaws.shape != centers.shape[:-1]:
        raise ValueError("centers/yaws must be [K,S,3] and [K,S]")
    k, s = int(centers.shape[0]), int(centers.shape[1])
    result = torch.eye(4, dtype=centers.dtype).view(1, 1, 4, 4).repeat(k, s, 1, 1)
    cosine, sine = torch.cos(yaws), torch.sin(yaws)
    # Match training metadata: yaw is atan2(local-x heading_x,
    # local-x heading_z), so yaw=0 points the object's length axis along
    # anchor +z and positive yaw turns it toward anchor +x.
    result[..., 0, 0] = sine
    result[..., 2, 0] = cosine
    result[..., 0, 2] = -cosine
    result[..., 2, 2] = sine
    result[..., :3, 3] = centers
    return result


def build_external_factorized_pretrain_bundle(
    manifest_path: str | Path,
    *,
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    device: torch.device,
    patch_grid: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    """Build an open-inference bundle without a target video or target mask."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    if payload.get("coordinate_frame") != "camera_anchor":
        raise ValueError("external asset manifest coordinate_frame must be 'camera_anchor'")
    fps = float(payload.get("fps", 10.0))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("manifest fps must be finite and positive")
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("manifest requires a camera object")
    camera_to_anchor = torch.as_tensor(camera.get("camera_to_anchor"), dtype=torch.float32)
    if camera_to_anchor.ndim != 3 or camera_to_anchor.shape[-2:] != (4, 4):
        raise ValueError("camera.camera_to_anchor must be [S,4,4]")
    seq_len = int(camera_to_anchor.shape[0])
    intrinsics = torch.as_tensor(camera.get("intrinsics"), dtype=torch.float32)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("camera.intrinsics must be [1/S,3,3]")
    if int(intrinsics.shape[0]) == 1:
        intrinsics = intrinsics.expand(seq_len, -1, -1).contiguous()
    if int(intrinsics.shape[0]) != seq_len:
        raise ValueError("camera intrinsics length must be one or match camera_to_anchor")
    image_size_hw = torch.as_tensor(camera.get("image_size_hw"), dtype=torch.long)
    if tuple(image_size_hw.shape) != (2,) or bool((image_size_hw <= 0).any()):
        raise ValueError("camera.image_size_hw must contain positive [height,width]")

    sf = unwrap_ddp(scene_flow)
    if str(getattr(sf.config, "asset_condition_protocol", "")) != "factorized_v1":
        raise RuntimeError(
            "External factorized asset inference requires a checkpoint trained with "
            "asset_condition_protocol='factorized_v1'; legacy checkpoints do not "
            "contain trained placement-adapter weights."
        )
    max_assets = int(sf.config.max_assets)
    canvas_hw = (int(patch_grid[0]) * 14, int(patch_grid[1]) * 14)
    reference_rgb = torch.zeros((max_assets, 3, *canvas_hw), dtype=torch.float32)
    reference_alpha = torch.zeros((max_assets, 1, *canvas_hw), dtype=torch.float32)
    centers = torch.zeros((max_assets, seq_len, 3), dtype=torch.float32)
    sizes = torch.ones((max_assets, seq_len, 3), dtype=torch.float32)
    yaws = torch.zeros((max_assets, seq_len), dtype=torch.float32)
    velocities = torch.zeros((max_assets, seq_len, 3), dtype=torch.float32)
    track_valid = torch.zeros((max_assets, seq_len), dtype=torch.bool)
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("manifest objects must be a list")
    if len(objects) > max_assets:
        raise ValueError(f"manifest has {len(objects)} objects, model supports {max_assets}")
    object_ids = []
    for slot, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError("each manifest object must be a JSON object")
        image_value = item.get("image")
        if not image_value:
            raise ValueError("each manifest object requires `image`")
        image_path = (path.parent / str(image_value)).resolve()
        mask_value = item.get("mask")
        mask_path = None if not mask_value else (path.parent / str(mask_value)).resolve()
        rgb, alpha = _load_external_rgba(image_path, mask_path)
        reference_rgb[slot], reference_alpha[slot] = canonicalize_asset_reference(
            rgb, alpha, canvas_hw
        )
        center, size, yaw, velocity, valid = interpolate_box_keyframes(
            item.get("keyframes", []),
            seq_len,
            fps,
        )
        centers[slot] = center
        sizes[slot] = size
        yaws[slot] = yaw
        velocities[slot] = velocity
        track_valid[slot] = valid
        object_ids.append(str(item.get("id", f"asset_{slot}")))

    encoder = CanonicalAssetEncoder(
        vggt_model.aggregator,
        vggt_model.scene_tokenizer,
        sf,
        patch_grid=patch_grid,
        max_tokens=32,
    ).to(device)
    appearance = encoder(
        reference_rgb.to(device).reshape(max_assets, 1, 3, *canvas_hw),
        reference_alpha.to(device).reshape(max_assets, 1, 1, *canvas_hw),
        batch_size=1,
        num_assets=max_assets,
    )
    object_to_anchor = _object_to_anchor_from_center_yaw(centers, yaws).unsqueeze(0).to(device)
    reference_frame_id = torch.full((1, max_assets), -1, device=device, dtype=torch.long)
    condition = build_factorized_asset_condition(
        appearance_tokens=appearance.appearance_tokens,
        appearance_mask=appearance.appearance_mask,
        canonical_uv=appearance.canonical_uv,
        object_to_anchor=object_to_anchor,
        center_anchor=centers.unsqueeze(0).to(device),
        box_size_lwh=sizes.unsqueeze(0).to(device),
        yaw=yaws.unsqueeze(0).to(device),
        velocity_anchor=velocities.unsqueeze(0).to(device),
        track_valid=track_valid.unsqueeze(0).to(device),
        camera_to_anchor=camera_to_anchor.unsqueeze(0).to(device),
        intrinsics=intrinsics.unsqueeze(0).to(device),
        image_size_hw=image_size_hw.to(device),
        patch_grid=patch_grid,
        reference_frame_id=reference_frame_id,
    )
    # Anchor coordinates are a valid "world" for the shared camera summary.
    camera_tokens, camera_mask = camera_summary_from_waymo_gt(
        camera_to_anchor.unsqueeze(0).to(device),
        intrinsics.unsqueeze(0).to(device),
        image_hw=tuple(int(value) for value in image_size_hw.tolist()),
        trajectory_anchor_to_world=torch.eye(4, device=device).view(1, 1, 4, 4),
    )
    latent_dim = int(sf.config.out_channels)
    dummy_endpoint = torch.zeros(
        (1, seq_len, int(patch_grid[0]) * int(patch_grid[1]), latent_dim),
        device=device,
    )
    frame_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1)
    bundle = build_full_scene_bundle(
        dummy_endpoint,
        kv_dim=latent_dim,
        camera_condition_tokens=camera_tokens,
        camera_attention_mask=camera_mask,
        camera_condition_kind=["camera"],
        frame_ids=frame_ids,
    )
    bundle.F_asset_tokens = dummy_endpoint.new_zeros((1, 0, latent_dim))
    bundle.encoder_attention_mask = None
    bundle.factorized_asset_condition = condition
    bundle.F_asset_lengths = condition.appearance_mask.any(dim=-1).sum(dim=-1).long()
    bundle.asset_condition_kind = [
        "factorized_asset" if bool(condition.appearance_mask.any()) else "none"
    ]
    bundle.asset_condition_source_kind = ["external_manifest"]
    bundle.captions = [str(payload.get("text", ""))]
    bundle.fps = torch.tensor([fps], device=device)
    bundle.camera_gen_anchor_mask = frame_ids.eq(0)
    return bundle, {
        "manifest": str(path),
        "condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
        "object_ids": object_ids,
        "num_frames": seq_len,
        "fps": fps,
        "image_size_hw": image_size_hw.tolist(),
    }


def _align_sliding_asset_payload_slots(
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Give the same object the same slot in every inference window."""
    if not payloads:
        return [], []
    max_assets = len(payloads[0].get("pretrain_object_ids", []))
    if max_assets <= 0:
        return payloads, []
    scores: dict[str, float] = {}
    for payload in payloads:
        ids = [str(value) for value in payload["pretrain_object_ids"]]
        values = torch.as_tensor(payload["pretrain_object_scores"]).reshape(-1)
        if len(ids) != max_assets or int(values.numel()) != max_assets:
            raise ValueError("sliding asset payload slot counts disagree")
        for slot, object_id in enumerate(ids):
            if object_id:
                scores[object_id] = scores.get(object_id, 0.0) + float(
                    values[slot].item()
                )
    global_ids = [
        object_id
        for object_id, _ in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:max_assets]
    ]
    global_ids.extend([""] * (max_assets - len(global_ids)))

    aligned = []
    for payload in payloads:
        source_ids = [str(value) for value in payload["pretrain_object_ids"]]
        source_lookup = {
            object_id: slot
            for slot, object_id in enumerate(source_ids)
            if object_id
        }
        result = dict(payload)
        result["pretrain_object_ids"] = [""] * max_assets
        result["pretrain_object_class_names"] = [""] * max_assets
        slot_tensor_keys = [
            key
            for key, value in payload.items()
            if torch.is_tensor(value)
            and value.ndim >= 1
            and int(value.shape[0]) == max_assets
            and (
                key.startswith("pretrain_object_")
                or key.startswith("pretrain_reference_")
            )
        ]
        for key in slot_tensor_keys:
            value = payload[key]
            if key == "pretrain_object_obj_to_anchor":
                reset = torch.eye(4, dtype=value.dtype, device=value.device)
                reset = reset.view(1, 1, 4, 4).repeat(
                    max_assets, int(value.shape[1]), 1, 1
                )
            elif key in (
                "pretrain_object_bbox_patch",
                "pretrain_reference_frame_id",
            ):
                reset = torch.full_like(value, -1)
            else:
                reset = torch.zeros_like(value)
            result[key] = reset
        for destination, object_id in enumerate(global_ids):
            source = source_lookup.get(object_id)
            if source is None:
                continue
            result["pretrain_object_ids"][destination] = object_id
            result["pretrain_object_class_names"][destination] = payload[
                "pretrain_object_class_names"
            ][source]
            for key in slot_tensor_keys:
                result[key][destination] = payload[key][source]
        aligned.append(result)
    return aligned, global_ids


def attach_training_equivalent_sliding_asset_conditions(
    bundle: Any,
    *,
    dataset: WaymoOpenDataset,
    dataset_index: int,
    batch: dict[str, Any],
    vggt_model: nn.Module,
    scene_flow: nn.Module,
    device: torch.device,
    patch_grid: tuple[int, int],
    window: int,
    stride: int,
) -> dict[str, Any]:
    """Attach per-window references for raw target-video inference.

    A complete 29-frame rollout has no frame outside the *whole* rollout, but
    each model forward sees only one sliding window.  For target-informed
    evaluation we therefore rebuild the asset condition for every actual model
    window with the dataset's training projector.  Its reference is guaranteed
    to come from the same trunk but outside that window.
    """
    seq_len = int(bundle.z_clean_n.shape[1])
    windows = window_slices(seq_len, int(window), int(stride))
    if len(windows) <= 1:
        return {"active": False, "windows": []}
    if int(bundle.z_clean_n.shape[0]) != 1:
        raise ValueError(
            "window-specific raw inference asset extraction currently requires batch size 1"
        )
    intrinsics = batch.get("intrinsics")
    if not torch.is_tensor(intrinsics):
        raise RuntimeError("raw sliding inference requires batch intrinsics")
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    raw_hw = torch.as_tensor(batch.get("raw_image_size_hw"), device=device)
    if raw_hw.ndim >= 3 and int(raw_hw.shape[-1]) == 2:
        raw_hw = raw_hw[:, 0]
    frame_ids = torch.as_tensor(bundle.frame_ids, device=device, dtype=torch.long)
    if frame_ids.ndim == 1:
        frame_ids = frame_ids.unsqueeze(0)

    raw_payloads = [
        dataset.build_pretrain_asset_payload_for_sample_window(
            int(dataset_index), int(start), int(end)
        )
        for start, end in windows
    ]
    aligned_payloads, global_object_ids = _align_sliding_asset_payload_slots(
        raw_payloads
    )
    conditions: dict[tuple[int, int], FactorizedAssetCondition] = {}
    diagnostics = []
    max_lengths = torch.zeros_like(bundle.F_asset_lengths)
    any_asset = False
    for (start, end), payload in zip(windows, aligned_payloads):
        payload_batch: dict[str, Any] = {}
        for key, value in payload.items():
            payload_batch[key] = value.unsqueeze(0) if torch.is_tensor(value) else value
        built = build_factorized_asset_condition_from_batch(
            payload_batch,
            vggt_model,
            scene_flow,
            device,
            patch_grid=patch_grid,
            frame_ids=frame_ids[:, start:end],
            intrinsics=intrinsics[:, start:end],
            image_size_hw=raw_hw,
        )
        condition = built.condition.validate()
        if condition.seq_len != int(end - start):
            raise RuntimeError(
                f"window [{start},{end}) condition has S={condition.seq_len}"
            )
        conditions[(int(start), int(end))] = condition
        max_lengths = torch.maximum(max_lengths, built.lengths)
        has_asset = bool(
            (
                condition.appearance_mask.any(dim=(1, 2))
                & condition.track_valid.any(dim=(1, 2))
            ).any()
        )
        any_asset |= has_asset
        refs = condition.reference_frame_id.detach().cpu().tolist()
        target_ids = frame_ids[:, start:end].detach().cpu().tolist()
        for row_refs, row_targets in zip(refs, target_ids):
            target_set = set(int(value) for value in row_targets)
            for reference_id in row_refs:
                if int(reference_id) >= 0 and int(reference_id) in target_set:
                    raise RuntimeError(
                        f"sliding inference reference {reference_id} lies inside "
                        f"window target frames {sorted(target_set)}"
                    )
        diagnostics.append(
            {
                "window": [int(start), int(end)],
                "target_frame_ids": target_ids,
                "reference_frame_ids": refs,
                "source_kinds": built.source_kinds,
                "has_asset": has_asset,
            }
        )
    for left_index, (left_start, left_end) in enumerate(windows):
        left = conditions[(left_start, left_end)]
        for right_start, right_end in windows[left_index + 1 :]:
            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)
            if overlap_start >= overlap_end:
                continue
            right = conditions[(right_start, right_end)]
            left_slice = left.slice_time(
                overlap_start - left_start, overlap_end - left_start
            )
            right_slice = right.slice_time(
                overlap_start - right_start, overlap_end - right_start
            )
            shared = left_slice.track_valid & right_slice.track_valid
            if bool(shared.any()):
                placement_error = (
                    left_slice.placement_state - right_slice.placement_state
                ).abs().amax(dim=-1)
                bbox_error = (
                    left_slice.target_bbox_patch - right_slice.target_bbox_patch
                ).abs().amax(dim=-1)
                if bool((placement_error[shared] > 1.0e-4).any()) or bool(
                    (bbox_error[shared] > 1.0e-4).any()
                ):
                    raise RuntimeError(
                        "overlapping inference windows disagree on shared-object "
                        f"placement/bbox for frames [{overlap_start},{overlap_end})"
                    )
    bundle.factorized_asset_conditions_by_window = conditions
    bundle.F_asset_lengths = max_lengths
    bundle.asset_condition_kind = [
        "factorized_asset" if any_asset else "none"
    ]
    bundle.asset_condition_source_kind = ["sliding_outside_window_reference"]
    return {
        "active": True,
        "global_object_ids": global_object_ids,
        "overlap_geometry_verified": True,
        "windows": diagnostics,
    }


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
    flow_schedule = resolve_inference_flow_schedule(payload, fallback_args, path)
    config = _checkpoint_config(payload)
    if config is not None:
        camera_dim = int(config.get("camera_gen_dim", 2048))
        config.setdefault(
            "camera_generation_representation",
            "dggt_hidden_v1" if camera_dim == 2048 else CAMERA_GENERATION_REPRESENTATION,
        )
        config.setdefault("asset_position_mode", "localized")
        config.setdefault("mask_compositing_version", "legacy_hard_mask_v1")
        scene_flow = WanSceneFlow(**config)
    else:
        scene_flow = WanSceneFlow.from_scene_config(
            bring_up=False,
            patch_grid=fallback_args.patch_grid,
            in_channels=3 * int(fallback_args.latent_dim) + 3,
            out_channels=int(fallback_args.latent_dim),
            camera_gen_dim=int(fallback_args.camera_gen_dim),
            camera_generation_representation="dggt_hidden_v1",
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
        "camera_dggt_provenance": payload.get("camera_dggt_provenance") if isinstance(payload, dict) else None,
        "flow_schedule_config": flow_schedule,
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
    args.sky_atlas_hw = tuple(int(v) for v in getattr(config, "sky_atlas_hw", (32, 64)))
    args.sky_mask_refine_scale = int(config.sky_mask_refine_scale)
    args.sky_mask_refine_channels = int(config.sky_mask_refine_channels)
    args.asset_position_mode = str(getattr(config, "asset_position_mode", "localized"))


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
    sky_probability = sky_mask[0].detach().cpu().float().mean(dim=1).clamp(0.0, 1.0)
    non_sky_probability = 1.0 - sky_probability
    # Keep depth-valid points in the canonical state, then apply one explicit
    # hard sky threshold to both PLY formats at export time.
    sky_mask_cpu = torch.zeros_like(non_sky_probability).unsqueeze(1)
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
    state = build_clean_scene_state(sample, predictions)
    point_non_sky = non_sky_probability[
        state.source_image_ids,
        state.source_y,
        state.source_x,
    ].reshape(-1, 1)
    state.sky_probabilities = 1.0 - point_non_sky
    state.opacities = state.opacities * point_non_sky
    return state


def export_generated_pointclouds(
    *,
    scene_state: CleanSceneState,
    output_dir: Path,
    suffix: str,
    stride: int,
    sky_probability_threshold: float = 0.5,
    min_effective_opacity: float = 0.01,
) -> dict[str, Any]:
    """Write one DGGT canonical Gaussian/point PLY pair per generated frame.

    DGGT's inspection/export utilities write frame-local point clouds such as
    ``sample00_frame00.ply``.  Keeping that convention matters for MeshLab:
    opening an all-frame merged cloud makes the per-pixel samples from adjacent
    camera poses overlap densely, which visually reads as oversized points even
    though the PLY schema has no point-size field.
    """
    stride = max(1, int(stride))
    seq_len = int(scene_state.images.shape[0])
    frame_summaries: list[dict[str, Any]] = []
    frame_counts: list[int] = []
    total_points = 0
    total_before = 0
    total_sky_removed = 0
    total_opacity_removed = 0
    sky_probabilities = getattr(scene_state, "sky_probabilities", None)
    if not torch.is_tensor(sky_probabilities):
        raise RuntimeError("generated scene state is missing per-point sky probabilities")
    for frame_idx in range(seq_len):
        frame_keep = scene_state.source_image_ids == int(frame_idx)
        if stride > 1:
            frame_keep &= torch.remainder(scene_state.source_y, stride) == 0
            frame_keep &= torch.remainder(scene_state.source_x, stride) == 0
        before = int(frame_keep.sum().item())
        sky_keep = sky_probabilities.reshape(-1) < float(sky_probability_threshold)
        opacity_keep = scene_state.opacities.reshape(-1) >= float(min_effective_opacity)
        sky_removed = int((frame_keep & ~sky_keep).sum().item())
        opacity_removed = int((frame_keep & sky_keep & ~opacity_keep).sum().item())
        keep = frame_keep & sky_keep & opacity_keep
        total_before += before
        total_sky_removed += sky_removed
        total_opacity_removed += opacity_removed

        points = scene_state.means[keep]
        colors = scene_state.colors[keep]
        opacities = scene_state.opacities[keep]
        scales = scene_state.scales[keep]
        quats = scene_state.quats[keep]
        count = int(points.shape[0])
        frame_counts.append(count)
        total_points += count

        frame_suffix = f"{suffix}_frame{frame_idx:02d}"
        gaussian_path = output_dir / f"generated_raw_gaussians__{frame_suffix}.ply"
        point_path = output_dir / f"generated_raw_points__{frame_suffix}.ply"
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
        write_point_ply(points, colors, point_path, opacities=opacities)
        frame_summaries.append(
            {
                "frame": int(frame_idx),
                "gaussian_ply": str(gaussian_path),
                "point_ply": str(point_path),
                "num_points": count,
                "num_points_before_filter": before,
                "sky_removed": sky_removed,
                "low_opacity_removed": opacity_removed,
            }
        )

    return {
        "frames": frame_summaries,
        "num_points": int(total_points),
        "num_points_before_filter": int(total_before),
        "sky_removed": int(total_sky_removed),
        "low_opacity_removed": int(total_opacity_removed),
        "sky_probability_threshold": float(sky_probability_threshold),
        "min_effective_opacity": float(min_effective_opacity),
        "frame_counts": frame_counts,
        "stride": stride,
        "schema": "Per-frame DGGT Gaussian PLY + MeshLab xyz/uchar-RGB PLY",
        "scene_builder": "dggt.utils.gaussian_edit.build_clean_scene_state",
        "validity_rule": "generated depth > 1e-4 AND p_sky < threshold AND effective opacity >= threshold",
        "coordinates": "DGGT generated-camera world coordinates",
        "merged_ply_saved": False,
        "meshlab_note": "Open one frame PLY at a time; merged multi-frame clouds visually over-densify points.",
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
    camera_generated = generated_sample.camera_state_dggt
    camera_anchor_mask = generated_sample.camera_anchor_mask
    camera_initial_c2w = generated_sample.camera_initial_c2w_dggt
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

    patch_start_idx = int(getattr(vggt_model.aggregator, "patch_start_idx", 5))
    with autocast_context(args, device):
        geometry = decode_generated_dggt_geometry(
            vggt_model=vggt_model,
            scene_flow_root=unwrap_ddp(scene_flow),
            z_clean_pred_n=z_generated,
            patch_grid=args.patch_grid,
            patch_start_idx=patch_start_idx,
            image_hw=(height, width),
        )
        generated_pose = decode_pose_from_camera_features(
            vggt_model,
            camera_generated.to(device),
            camera_anchor_mask=camera_anchor_mask,
            initial_camera_to_world=camera_initial_c2w,
        )
        generated_sky_mask = _sky_mask_patch_to_image(
            sky_mask_refined if sky_mask_refined is not None else sky_mask_patch,
            patch_grid=args.patch_grid,
            height=height,
            width=width,
            device=device,
        )
    gs_map, gs_conf = geometry.gs_map, geometry.gs_conf
    generated_depth, generated_dynamic = geometry.depth, geometry.dynamic_conf

    sky_background = None
    sky_grid_image = None
    if sky_generated is not None:
        sky_h, sky_w = args.sky_atlas_hw
        extrinsic, intrinsic = _predict_camera_mats(generated_pose, (height, width), device)
        sky_background = sky_tokens_to_background(
            decode_sky_patch_tokens(sky_generated.to(device)),
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
            soft_sky_mask=True,
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
        sky_probability_threshold=float(args.ply_sky_probability_threshold),
        min_effective_opacity=float(args.ply_min_effective_opacity),
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
    parser.add_argument(
        "--asset_manifest",
        default=None,
        help=(
            "External factorized asset manifest (RGBA or RGB+mask, box keyframes, "
            "camera_to_anchor, intrinsics and FPS). This mode does not read a target video."
        ),
    )

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
        default=10,
        help="Contiguous frames per validation sample (default: 10).",
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
    parser.add_argument(
        "--asset_control_guidance_scale",
        type=float,
        default=1.0,
        help=(
            "Independent asset-condition guidance scale. Default 1.0 keeps the supplied visual "
            "condition fixed while --cfg sweeps text guidance, matching Cosmos conditional generation."
        ),
    )
    parser.add_argument(
        "--camera_guidance_scale",
        type=float,
        default=1.0,
        help=(
            "Independent camera-condition guidance scale. Default 1.0 keeps the requested camera "
            "trajectory fixed while --cfg sweeps text guidance."
        ),
    )
    parser.add_argument(
        "--camera_text_guidance_scale",
        type=float,
        default=1.0,
        help="Independent text-CFG scale for generated camera state.",
    )
    parser.add_argument("--sample_steps", "--val_sample_steps", dest="val_sample_steps", type=int, default=35)
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Optional assertion for the flow shift stored in the checkpoint.",
    )
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
    parser.add_argument("--ply_sky_probability_threshold", type=float, default=0.5)
    parser.add_argument("--ply_min_effective_opacity", type=float, default=0.01)
    parser.add_argument(
        "--val_sliding_window",
        type=int,
        default=10,
        help=(
            "Offline SceneFlow window size, capped at 10. Values <=0 select automatic mode; "
            "requests longer than 10 frames always use overlapping sliding windows."
        ),
    )
    parser.add_argument(
        "--val_sliding_stride",
        type=int,
        default=7,
        help="Sliding stride; it must be smaller than the effective window for long clips.",
    )

    # Fallback architecture only; saved scene_flow_config takes precedence.
    parser.add_argument("--patch_grid_h", type=int, default=25)
    parser.add_argument("--patch_grid_w", type=int, default=37)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--prediction_type", choices=("x", "v"), default="x")
    parser.add_argument("--sky_grid_h", type=int, default=DEFAULT_SKY_GRID[0])
    parser.add_argument("--sky_grid_w", type=int, default=DEFAULT_SKY_GRID[1])
    parser.add_argument("--sky_mask_refine_scale", type=int, default=4)
    parser.add_argument("--sky_mask_refine_channels", type=int, default=256)
    parser.add_argument("--asset_position_mode", choices=("localized", "canonical"), default="localized")
    parser.add_argument("--sky_unobserved_loss_weight", type=float, default=0.0)
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
    if not 0.0 <= float(args.ply_sky_probability_threshold) <= 1.0:
        raise ValueError("--ply_sky_probability_threshold must be in [0,1].")
    if not 0.0 <= float(args.ply_min_effective_opacity) <= 1.0:
        raise ValueError("--ply_min_effective_opacity must be in [0,1].")
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be non-negative.")
    if int(args.val_scene_end) <= int(args.val_scene_start):
        raise ValueError("--val_scene_end must be greater than --val_scene_start.")
    return parse_cfg_scales(args.cfg_values)


def main() -> None:
    args = build_argparser().parse_args()
    cfg_scales = validate_args(args)
    args.val_sliding_window, args.val_sliding_stride, offline_sliding = resolve_offline_window(
        int(args.num_frames),
        int(args.val_sliding_window),
        int(args.val_sliding_stride),
    )
    args.patch_grid = (int(args.patch_grid_h), int(args.patch_grid_w))
    args.sky_grid = (int(args.sky_grid_h), int(args.sky_grid_w))
    # Bare legacy checkpoints have no config and are interpreted as CameraHead hidden v1.
    args.camera_gen_dim = 2048
    args.caption_root = args.val_caption_root
    args.val_log_images = int(args.num_frames)
    # `--cfg` is text guidance.  As in Cosmos conditional video generation,
    # clean structural conditions remain present in both CFG branches instead
    # of being amplified together with text. Asset/camera can still be swept
    # explicitly through their independent flags.
    args.guidance_scale = 1.0

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
    dggt_sha256 = checkpoint_sha256(args.dggt_ckpt_path)
    provenance = checkpoint_info.get("camera_dggt_provenance")
    recorded_hash = provenance.get("dggt_checkpoint_sha256") if isinstance(provenance, dict) else None
    if recorded_hash != dggt_sha256:
        raise ValueError(
            "Pretrain inference DGGT checkpoint does not match camera provenance: "
            f"checkpoint={recorded_hash!r}, current={dggt_sha256!r}."
        )
    if args.feature_stats_path:
        if str(scene_flow.config.camera_generation_representation) == CAMERA_GENERATION_REPRESENTATION:
            load_all_stats_into_buffers(
                scene_flow,
                args.feature_stats_path,
                token_dim=int(args.latent_dim),
                dggt_ckpt_path=args.dggt_ckpt_path,
                require_existing_match=True,
            )
        else:
            load_into_buffers(scene_flow, args.feature_stats_path, token_dim=int(args.latent_dim))
    if str(scene_flow.config.camera_generation_representation) == CAMERA_GENERATION_REPRESENTATION:
        scene_flow.require_camera_stats()

    print(
        f"[checkpoint] {checkpoint_info['weight_source']} step={checkpoint_info['step']} "
        f"grid={args.patch_grid} latent_dim={args.latent_dim} prediction={args.prediction_type}",
        flush=True,
    )
    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path, args.tokenizer_ckpt_path, device
    )
    if (
        str(scene_flow.config.camera_generation_representation) == CAMERA_GENERATION_REPRESENTATION
        and int(scene_flow.config.camera_gen_dim) != CAMERA_GENERATION_DIM
    ):
        raise ValueError(
            f"DGGT v3 SceneFlow camera_gen_dim={scene_flow.config.camera_gen_dim} must be {CAMERA_GENERATION_DIM}."
        )
    text_encoder = setup_text_encoder(args, device)

    if args.asset_manifest:
        bundle, manifest_info = build_external_factorized_pretrain_bundle(
            args.asset_manifest,
            vggt_model=vggt_model,
            scene_flow=scene_flow,
            device=device,
            patch_grid=args.patch_grid,
        )
        args.num_frames = int(bundle.z_clean_n.shape[1])
        args.val_log_images = int(args.num_frames)
        args.val_sliding_window, args.val_sliding_stride, offline_sliding = resolve_offline_window(
            int(args.num_frames),
            int(args.val_sliding_window),
            int(args.val_sliding_stride),
        )
        apply_condition_mode(bundle, "asset_cam")
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        results = []
        for scale in cfg_scales:
            args.guidance_scale = float(scale)
            generated = cfg_sample_pretrain_latents(
                scene_flow,
                bundle,
                args,
                step=0,
                device=device,
                guidance_scale=float(scale),
                text_encoder=text_encoder,
                return_camera=False,
                return_sky=sky_generation_enabled(args),
                return_sky_mask=True,
            )
            output_path = output_root / f"external_factorized__cfg{cfg_tag(scale)}.pt"
            torch.save(
                {
                    "video_latent_normalized": generated.video.detach().cpu(),
                    "sky_tokens": None if generated.sky is None else generated.sky.detach().cpu(),
                    "sky_mask_patch": None
                    if generated.sky_mask_patch is None
                    else generated.sky_mask_patch.detach().cpu(),
                    "frame_ids": bundle.frame_ids.detach().cpu(),
                    "fps": bundle.fps.detach().cpu(),
                    "factorized_asset_condition_version": FACTORIZED_ASSET_CONDITION_VERSION,
                },
                output_path,
            )
            results.append({"cfg": float(scale), "output": str(output_path)})
        summary = {
            "mode": "external_factorized_asset",
            "target_video_read": False,
            "target_dynamic_mask_read": False,
            "manifest": manifest_info,
            "checkpoint": checkpoint_info,
            "window": int(args.val_sliding_window),
            "window_stride": int(args.val_sliding_stride),
            "sliding_window_active": bool(offline_sliding),
            "results": results,
        }
        (output_root / "external_factorized_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print(f"[done] wrote external factorized outputs under {output_root}", flush=True)
        return

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
        return_full_dggt_context=True,
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
    print(
        f"[sampling] cfg={cfg_scales} steps={args.val_sample_steps} shift={args.shift} "
        f"window={args.val_sliding_window} stride={args.val_sliding_stride} "
        f"sliding={offline_sliding}",
        flush=True,
    )

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
        sliding_asset_info = {"active": False, "windows": []}
        if offline_sliding:
            sliding_asset_info = attach_training_equivalent_sliding_asset_conditions(
                bundle,
                dataset=dataset,
                dataset_index=dataset_index,
                batch=batch,
                vggt_model=vggt_model,
                scene_flow=scene_flow,
                device=device,
                patch_grid=args.patch_grid,
                window=int(args.val_sliding_window),
                stride=int(args.val_sliding_stride),
            )
        apply_condition_mode(bundle, mode)
        condition_rows = actual_condition_rows(bundle)
        common_paths = save_common_validation_images(bundle, batch, sample_dir, args)

        per_cfg: list[dict[str, Any]] = []
        for scale in cfg_scales:
            # Same sample step across CFG values => identical initial video,
            # camera and sky noise, matching training validation comparisons.
            args.guidance_scale = float(scale)
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
                        "asset": float(args.asset_control_guidance_scale),
                        "camera": float(args.camera_guidance_scale),
                        "camera_text": float(args.camera_text_guidance_scale),
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
            "window": int(args.val_sliding_window),
            "window_stride": int(args.val_sliding_stride),
            "sliding_window_active": bool(offline_sliding),
            "sliding_asset_reference_policy": sliding_asset_info,
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
        "asset_control_guidance_scale": float(args.asset_control_guidance_scale),
        "camera_guidance_scale": float(args.camera_guidance_scale),
        "camera_text_guidance_scale": float(args.camera_text_guidance_scale),
        "condition_cycle": list(CONDITION_MODES),
        "num_frames": int(args.num_frames),
        "window": int(args.val_sliding_window),
        "window_stride": int(args.val_sliding_stride),
        "sliding_window_active": bool(offline_sliding),
        "sample_steps": int(args.val_sample_steps),
        "shift": float(args.shift),
        "ply_stride": int(args.ply_stride),
        "samples": all_summaries,
    }
    (output_root / "all_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))
    print(f"[done] wrote {len(all_summaries)} sample folders under {output_root}", flush=True)


if __name__ == "__main__":
    main()
