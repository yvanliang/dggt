"""Pretraining entry point for SceneFlow on raw Waymo clips.

Frozen:
  - VGGT aggregator
  - JointSceneTokenizer

Trainable:
  - WanSceneFlow

Launch:
  torchrun --nproc_per_node=8 train_scene_flow_pretrain.py \
      --image_dir /data/waymo \
      --dggt_ckpt_path pretrained/dggt.pth \
      --tokenizer_ckpt_path logs/tokenizer_t0_waymo_views1/ckpt/scene_tokenizer_latest.pt \
      --feature_stats_path logs/tokenizer_t0_waymo_views1/feature_stats.pt \
      --log_dir logs/scene_flow_pretrain
"""
from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm.auto import tqdm

from datasets.dataset import WaymoOpenDataset
from dggt.losses.flow_losses import (
    boundary_mask_from_edit_mask,
    build_masked_rectified_flow_target,
    compute_total_loss,
    rae_t_grid,
)
from dggt.losses.rgb_render_loss import (
    compute_rgb_render_loss,
    decode_generated_dggt_geometry,
    rgb_render_loss_enabled,
    rgb_render_loss_ramp,
    setup_lpips_for_rgb_loss,
    should_apply_rgb_render_loss,
)
from dggt.models.scene_flow import WanSceneFlow
from dggt.models.embedders.text_encoder import TextEncoder
from dggt.models.vggt import VGGT
from dggt.utils.feature_stats import load_all_stats_into_buffers
from dggt.utils.gaussian_render import composite_gsplat_rgb
from dggt.utils.camera_condition import (
    camera_summary_from_waymo_gt,
)
from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    CAMERA_TARGET_SOURCE,
    CAMERA_TARGET_SPACE,
    camera_anchor_mask,
    camera_state_from_dggt_pose_enc,
    camera_geometry_loss,
    decode_camera_trajectory,
    rotation_6d_to_matrix,
    so3_geodesic_angle,
)
from dggt.utils.flow_viz import save_image_grid
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.pretrain_asset_slots import (
    build_pretrain_asset_slots_from_dynamic_mask,
    build_pretrain_asset_slots_from_object_patch_mask,
)
from dggt.utils.rae_optim import build_rae_optimizer, build_rae_scheduler
from dggt.utils.sliding_window import (
    cosine_coverage,
    cosine_window,
    default_window_stride,
    scene_global_window_weight,
    window_slices,
)
from dggt.utils.gaussian_time import gaussian_timestamps_from_frame_ids
from dggt.utils.tokens import (
    reattach_special_tokens,
    replace_selected_levels,
    select_patch_pyramid,
    split_special_and_patch,
)

from diffusers.training_utils import EMAModel


TOKENIZER_LEVELS = (4, 11, 17, 23)
SKY_CLASS_INDEX = 9
SKY_RGB_DIM = 3
SKY_REPRESENTATION_VERSION = "rgb_patch_v2"
DEFAULT_SKY_ATLAS_HW = (32, 64)
DEFAULT_SKY_GRID = (16, 32)
SKY_PATCH_SIZE = 2
SKY_TOKEN_DIM = SKY_RGB_DIM * SKY_PATCH_SIZE * SKY_PATCH_SIZE
DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT = 0.0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args: argparse.Namespace | None = None) -> tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            timeout_minutes = int(
                getattr(
                    args,
                    "ddp_timeout_minutes",
                    os.environ.get("DDP_TIMEOUT_MINUTES", 60),
                )
            )
            init_kwargs = {}
            if timeout_minutes > 0:
                init_kwargs["timeout"] = timedelta(minutes=timeout_minutes)
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                **init_kwargs,
            )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, local_rank, world_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(args: argparse.Namespace, device: torch.device):
    enabled = device.type == "cuda" and args.precision == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


@torch.no_grad()
def materialize_ema_state_dict(scene_flow: nn.Module, ema: EMAModel) -> dict[str, torch.Tensor]:
    """Return a named SceneFlow state_dict with EMA parameters and live buffers."""
    sf = unwrap_ddp(scene_flow)
    params = list(sf.parameters())
    ema.store(params)
    ema.copy_to(params)
    try:
        return {key: value.detach().cpu().clone() for key, value in sf.state_dict().items()}
    finally:
        ema.restore(params)


@torch.no_grad()
def sync_ema_shadow_from_model(scene_flow: nn.Module, ema: EMAModel) -> None:
    """Initialize EMAModel shadow params from the currently loaded model."""
    params = list(unwrap_ddp(scene_flow).parameters())
    if len(params) != len(ema.shadow_params):
        raise ValueError(
            f"EMA shadow param count {len(ema.shadow_params)} != model param count {len(params)}"
        )
    ema.shadow_params = [p.detach().clone() for p in params]


@torch.no_grad()
def load_warm_start_ema_or_sync(scene_flow: nn.Module, ema: EMAModel, ema_state: Any) -> bool:
    """Load compatible EMA state; fall back to current model params for architecture warm-starts."""
    try:
        ema.load_state_dict(ema_state)
        return True
    except Exception as exc:  # noqa: BLE001 - diffusers may raise ValueError or RuntimeError here.
        sync_ema_shadow_from_model(scene_flow, ema)
        if is_main_process():
            print(f"[warm-start] EMA state is incompatible with the current SceneFlow; synced EMA from model ({exc}).")
        return False


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def format_key_examples(keys: list[str], limit: int = 5) -> str:
    if not keys:
        return "[]"
    examples = ", ".join(keys[:limit])
    suffix = "" if len(keys) <= limit else ", ..."
    return f"[{examples}{suffix}]"


def load_dggt_aggregator_and_tokenizer(
    dggt_ckpt_path: str,
    tokenizer_ckpt_path: str | None,
    device: torch.device,
) -> VGGT:
    model = VGGT().to(device)

    checkpoint = torch.load(dggt_ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported DGGT checkpoint format: {dggt_ckpt_path}")
    missing, unexpected = model.load_state_dict(strip_module_prefix(state), strict=False)
    if is_main_process():
        ignored_missing = [key for key in missing if key.startswith("scene_tokenizer.")]
        real_missing = [key for key in missing if not key.startswith("scene_tokenizer.")]
        print(
            "[ckpt:dggt] "
            f"missing={len(real_missing)} "
            f"ignored_missing_scene_tokenizer={len(ignored_missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )
        if real_missing or unexpected:
            print(
                "[ckpt:dggt] "
                f"missing_examples={format_key_examples(real_missing)} "
                f"unexpected_examples={format_key_examples(unexpected)}",
                flush=True,
            )

    if tokenizer_ckpt_path:
        tok_checkpoint = torch.load(tokenizer_ckpt_path, map_location="cpu")
        tok_state: Any = tok_checkpoint
        if isinstance(tok_checkpoint, dict):
            tok_state = tok_checkpoint.get("scene_tokenizer", tok_checkpoint.get("state_dict", tok_checkpoint))
        if not isinstance(tok_state, dict):
            raise ValueError(f"Unsupported tokenizer checkpoint format: {tokenizer_ckpt_path}")
        tok_state = strip_module_prefix(tok_state)
        if any(key.startswith("scene_tokenizer.") for key in tok_state):
            tok_state = {
                key[len("scene_tokenizer."):]: value
                for key, value in tok_state.items()
                if key.startswith("scene_tokenizer.")
            }
        missing, unexpected = model.scene_tokenizer.load_state_dict(tok_state, strict=False)
        if is_main_process():
            print(f"[ckpt:tokenizer] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            if missing or unexpected:
                print(
                    "[ckpt:tokenizer] "
                    f"missing_examples={format_key_examples(missing)} "
                    f"unexpected_examples={format_key_examples(unexpected)}",
                    flush=True,
                )
        if missing or unexpected:
            raise RuntimeError("Tokenizer checkpoint did not match VGGT.scene_tokenizer.")

    model.eval()
    freeze_module(model)
    return model


def discover_scene_names(image_dir: str, scene_start: int, scene_end: int) -> list[str]:
    root = Path(image_dir)
    scene_names = []
    for idx in range(int(scene_start), int(scene_end)):
        name = f"{idx:03d}"
        if (root / name / "images").is_dir():
            scene_names.append(name)
    if not scene_names:
        raise RuntimeError(
            f"No Waymo scene folders with images found in {image_dir} for "
            f"[{scene_start}, {scene_end})."
        )
    return scene_names


class CyclicSequentialSampler(Sampler[int]):
    """Sequential sampler whose starting point advances between validations."""

    def __init__(self, data_source) -> None:
        self.data_source = data_source
        self.offset = 0

    def set_offset(self, offset: int) -> None:
        length = len(self.data_source)
        self.offset = int(offset) % length if length else 0

    def __iter__(self):
        length = len(self.data_source)
        return iter((self.offset + i) % length for i in range(length))

    def __len__(self) -> int:
        return len(self.data_source)


def split_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "scale_shift_table" in name
            or name.endswith("null_kv")
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def scene_flow_prediction_type(scene_flow: nn.Module) -> str:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return str(getattr(cfg, "prediction_type", "x"))


def checkpoint_prediction_type(payload: Any) -> str | None:
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
    "sky_representation_version",
    "sky_atlas_hw",
    "sky_token_dim",
    "rope_max_position",
    "camera_gen_dim",
    "camera_generation_representation",
    "camera_stats_version",
    "camera_condition_representation",
    "mask_compositing_version",
    "asset_position_mode",
    "sky_rope_temporal_offset",
    "camera_rope_spatial_mode",
    "sky_mask_head_version",
    "sky_mask_refine_scale",
    "sky_mask_refine_channels",
)


def _checkpoint_camera_representation(payload: Any) -> tuple[str, int]:
    cfg = payload.get("scene_flow_config") if isinstance(payload, dict) else None
    if not isinstance(cfg, dict):
        return "dggt_hidden_v1", 2048
    return str(cfg.get("camera_generation_representation", "dggt_hidden_v1")), int(cfg.get("camera_gen_dim", 2048))


def migrate_legacy_camera_checkpoint(scene_flow: nn.Module, payload: Any) -> tuple[int, int, str]:
    """Load only shape-compatible shared EMA trunk weights from camera v1."""
    representation, camera_dim = _checkpoint_camera_representation(payload)
    if representation != "dggt_hidden_v1" or camera_dim != 2048:
        raise ValueError(
            f"legacy camera migration expects dggt_hidden_v1/2048D, got {representation}/{camera_dim}D"
        )
    if isinstance(payload, dict) and isinstance(payload.get("ema_scene_flow_state_dict"), dict):
        source = payload["ema_scene_flow_state_dict"]
        source_name = "ema_scene_flow_state_dict"
    elif isinstance(payload, dict) and isinstance(payload.get("scene_flow"), dict):
        source = payload["scene_flow"]
        source_name = "scene_flow"
    elif isinstance(payload, dict):
        source = payload
        source_name = "state_dict"
    else:
        raise TypeError("legacy warm-start checkpoint must contain a state dict")
    source = strip_module_prefix(source)
    target = unwrap_ddp(scene_flow)
    current = target.state_dict()
    camera_prefixes = (
        "camera_norm.",
        "camera_proj.",
        "camera_modality_embed",
        "camera_null_condition_embed",
        "camera_gen_norm.",
        "camera_gen_proj.",
        "camera_gen_decoder.",
        "camera_gen_modality_embed",
        "camera_gen_role_embed.",
        "camera_anchor_",
        "camera_delta_",
        "camera_stats_valid",
    )
    transferred = {
        key: value
        for key, value in source.items()
        if torch.is_tensor(value)
        and key in current
        and tuple(value.shape) == tuple(current[key].shape)
        and not key.startswith(camera_prefixes)
    }
    current.update(transferred)
    target.load_state_dict(current, strict=True)
    return len(transferred), len(source) - len(transferred), source_name


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


def validate_scene_flow_checkpoint_config(
    scene_flow: nn.Module,
    payload: Any,
    path: str | Path,
    expected_dggt_sha256: str | None = None,
) -> None:
    validate_prediction_type_checkpoint(scene_flow, payload, path)
    if not isinstance(payload, dict):
        return
    saved_cfg = payload.get("scene_flow_config")
    if not isinstance(saved_cfg, dict):
        return
    if saved_cfg.get("sky_representation_version") != SKY_REPRESENTATION_VERSION:
        raise ValueError(
            f"{path} is not a {SKY_REPRESENTATION_VERSION} sky checkpoint. Old RGB-atlas checkpoints "
            "cannot be resumed directly; use an explicit non-sky partial warm-start."
        )
    if saved_cfg.get("camera_generation_representation") == CAMERA_GENERATION_REPRESENTATION:
        provenance = payload.get("camera_dggt_provenance")
        recorded_hash = provenance.get("dggt_checkpoint_sha256") if isinstance(provenance, dict) else None
        if expected_dggt_sha256 is not None and recorded_hash != expected_dggt_sha256:
            raise ValueError(
                f"{path} DGGT camera provenance mismatch: "
                f"checkpoint={recorded_hash!r}, current={expected_dggt_sha256!r}"
            )
        state = payload.get("scene_flow")
        if isinstance(state, dict):
            state = strip_module_prefix(state)
            required_camera_stats = {
                "camera_anchor_mean",
                "camera_anchor_std",
                "camera_delta_mean",
                "camera_delta_std",
                "camera_stats_valid",
                "camera_gen_role_embed.weight",
            }
            missing_camera_stats = sorted(required_camera_stats.difference(state))
            if missing_camera_stats:
                raise ValueError(f"{path} is a DGGT v3 camera checkpoint missing {missing_camera_stats}")
            if not bool(torch.as_tensor(state["camera_stats_valid"]).item()):
                raise ValueError(f"{path} records DGGT v3 camera generation but camera statistics are not valid")
    if "rope_layout_version" not in saved_cfg and "mrope_temporal_margin" in saved_cfg:
        raise ValueError(
            f"{path} was saved with the legacy global mrope_temporal_margin RoPE layout. "
            "The current SceneFlow model uses the fixed A2 layout "
            "(video/asset/camera shared video time, camera center, sky temporal offset 15000); "
            "do not resume/warm-start across these incompatible position semantics."
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
            "Do not resume/warm-start across incompatible RoPE/model geometry settings."
        )


def validate_prediction_type_checkpoint(scene_flow: nn.Module, payload: Any, path: str | Path) -> None:
    current = scene_flow_prediction_type(scene_flow)
    saved = checkpoint_prediction_type(payload)
    if saved is None:
        if current == "v":
            raise ValueError(
                f"{path} does not record SceneFlow prediction_type. Refusing to load it into "
                "a velocity-prediction model because legacy checkpoints were x-prediction by default. "
                "Use --prediction_type x for that checkpoint or warm-start from a checkpoint saved with scene_flow_config."
            )
        return
    if saved != current:
        raise ValueError(
            f"{path} prediction_type={saved!r} does not match current model prediction_type={current!r}. "
            "Do not warm-start/resume across x-prediction and velocity-prediction checkpoints."
        )


def model_prediction_to_velocity(scene_flow: nn.Module, prediction: torch.Tensor, target) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        # Match RAEv2 Transport.convert_model_pred: x-pred is converted to
        # velocity with the same t_eps clamp used to build the RF target.
        return (target.z_t - prediction) / target.sigmas4.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).clamp_min(float(getattr(target, "t_eps", 0.05)))
    return prediction


def scene_flow_t_eps(scene_flow: nn.Module) -> float:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return float(getattr(cfg, "t_eps", 0.05))


# Unlike the legacy RAE-compatible video target, sky uses the exact derivative
# of its linear clean-to-noise path.  The floor is only a numerical guard for
# x-prediction conversion; sampled sigmoid times and ODE evaluation times are
# strictly positive whenever the model is called.
SKY_FLOW_T_EPS = 1.0e-6


def model_prediction_to_clean(scene_flow: nn.Module, prediction: torch.Tensor, target) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        return prediction
    return target.z_t - target.sigmas4.to(device=prediction.device, dtype=prediction.dtype) * prediction


def sampler_prediction_to_velocity(
    scene_flow: nn.Module,
    prediction: torch.Tensor,
    z: torch.Tensor,
    sigma: torch.Tensor,
    *,
    t_eps: float | None = None,
) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        while sigma.ndim < z.ndim:
            sigma = sigma.view(*sigma.shape, 1)
        denom_floor = scene_flow_t_eps(scene_flow) if t_eps is None else float(t_eps)
        return (z - prediction) / sigma.to(device=z.device, dtype=z.dtype).clamp_min(denom_floor)
    return prediction


def _init_pretrain_camera_noise(
    scene_flow: nn.Module,
    bundle,
    generator: torch.Generator,
    *,
    return_camera: bool,
) -> torch.Tensor | None:
    camera_clean = getattr(bundle, "camera_target_clean_n", None)
    if torch.is_tensor(camera_clean):
        camera_z = torch.empty_like(camera_clean)
        camera_z.normal_(generator=generator)
        return camera_z
    if not return_camera:
        return None
    z_template = getattr(bundle, "z_clean_n", None)
    if not torch.is_tensor(z_template) or z_template.ndim < 2:
        raise RuntimeError("Pretrain camera sampling requires bundle.z_clean_n with batch and sequence dimensions.")
    sf = unwrap_ddp(scene_flow)
    config = getattr(sf, "config", None)
    camera_dim = getattr(config, "camera_gen_dim", None)
    if camera_dim is None:
        raise RuntimeError(
            "Pretrain camera sampling was requested without bundle.camera_target_clean_n, "
            "but scene_flow.config.camera_gen_dim is unavailable."
        )
    camera_z = z_template.new_empty((int(z_template.shape[0]), int(z_template.shape[1]), int(camera_dim)))
    camera_z.normal_(generator=generator)
    return camera_z


def build_camera_rectified_flow_target(
    camera_clean: torch.Tensor | None,
    video_target,
) -> SimpleNamespace | None:
    """Noise normalized DGGT CameraHead pose tokens with the same RF time as video."""
    if camera_clean is None:
        return None
    if camera_clean.ndim != 3:
        raise ValueError(f"camera_clean must be [B,S,C], got {tuple(camera_clean.shape)}")
    b = int(camera_clean.shape[0])
    if video_target.sigmas.shape != (b,):
        raise ValueError(f"video_target sigmas shape {tuple(video_target.sigmas.shape)} != {(b,)}")
    sigmas = video_target.sigmas.to(device=camera_clean.device, dtype=camera_clean.dtype)
    sigmas3 = sigmas.view(b, 1, 1)
    eps = torch.randn_like(camera_clean)
    z_t = (1.0 - sigmas3) * camera_clean + sigmas3 * eps
    v_gt = (z_t - camera_clean) / sigmas3.clamp_min(float(getattr(video_target, "t_eps", 0.05)))
    return SimpleNamespace(
        sigmas=video_target.sigmas,
        sigmas4=sigmas3,
        z_t=z_t,
        v_gt=v_gt,
        eps=eps,
        weights=torch.ones((b, 1, 1), device=camera_clean.device, dtype=camera_clean.dtype),
        t_eps=float(getattr(video_target, "t_eps", 0.05)),
    )


def build_camera_anchor_context_dropout(
    anchor_mask: torch.Tensor,
    drop_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide the global camera anchor while retaining supervision on deltas.

    This exposes training to the exact context available in later sliding
    windows: all non-anchor camera-state tokens remain visible, while the one
    global anchor token is outside the window.  The hidden anchor output is not
    included in the camera flow loss for dropped rows.
    """
    anchors = anchor_mask.to(dtype=torch.bool)
    rows = drop_rows.to(device=anchors.device, dtype=torch.bool).view(-1)
    if anchors.ndim != 2 or int(anchors.shape[0]) != int(rows.numel()):
        raise ValueError(
            f"anchor_mask must be [B,S] and drop_rows [B], got {tuple(anchors.shape)} and {tuple(rows.shape)}"
        )
    hidden_anchor = anchors & rows[:, None]
    attention_mask = ~hidden_anchor
    supervision_mask = (~hidden_anchor).unsqueeze(-1)
    return attention_mask, supervision_mask


def camera_generation_tokens_from_aggregated(
    vggt_model: VGGT,
    aggregated_tokens_list: list[torch.Tensor],
) -> torch.Tensor:
    """Return the normalized CameraHead pose tokens used as generation targets."""
    if not aggregated_tokens_list:
        raise ValueError("aggregated_tokens_list must be non-empty")
    tokens = aggregated_tokens_list[-1]
    if tokens.ndim != 4:
        raise ValueError(f"aggregated camera tokens must be [B,S,N,C], got {tuple(tokens.shape)}")
    pose_tokens = tokens[:, :, 0].float()
    return vggt_model.camera_head.token_norm(pose_tokens)


def decode_pose_from_camera_features(
    vggt_model: VGGT,
    camera_features: torch.Tensor,
) -> torch.Tensor:
    """Decode legacy CameraHead tokens or a denormalized v2 11D trajectory."""
    if camera_features.ndim != 3:
        raise ValueError(f"camera_features must be camera state [B,S,C], got {tuple(camera_features.shape)}")
    if int(camera_features.shape[-1]) == CAMERA_GENERATION_DIM:
        anchors = camera_anchor_mask(
            int(camera_features.shape[0]), int(camera_features.shape[1]), device=camera_features.device
        )
        return decode_camera_trajectory(camera_features, anchors).pose_encoding
    return vggt_model.camera_head.trunk_fn(camera_features.float(), num_iterations=4)[-1]


def camera_pose_loss(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred_pose.shape != gt_pose.shape or pred_pose.shape[-1] != 9:
        raise ValueError(f"Camera pose shapes must both be [B,S,9], got {tuple(pred_pose.shape)} and {tuple(gt_pose.shape)}")
    pred = pred_pose.float()
    gt = gt_pose.to(device=pred.device, dtype=torch.float32)
    loss_t = torch.nn.functional.smooth_l1_loss(pred[..., :3], gt[..., :3])
    q_pred = torch.nn.functional.normalize(pred[..., 3:7], dim=-1, eps=1e-6)
    q_gt = torch.nn.functional.normalize(gt[..., 3:7], dim=-1, eps=1e-6)
    loss_r = (1.0 - (q_pred * q_gt).sum(dim=-1).abs().clamp(0.0, 1.0)).mean()
    loss_fov = torch.nn.functional.smooth_l1_loss(pred[..., 7:], gt[..., 7:])
    loss = (
        float(args.camera_translation_weight) * loss_t
        + float(args.camera_rotation_weight) * loss_r
        + float(args.camera_fov_weight) * loss_fov
    )
    logs = {
        "loss_camera_pose_t": float(loss_t.detach().item()),
        "loss_camera_pose_r": float(loss_r.detach().item()),
        "loss_camera_pose_fov": float(loss_fov.detach().item()),
    }
    return loss, logs


def camera_pose_validation_metrics(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    *,
    prefix: str = "sample_camera",
) -> dict[str, float]:
    """Scalar diagnostics for generated DGGT-space camera pose."""
    if pred_pose.shape != gt_pose.shape or pred_pose.shape[-1] != 9:
        raise ValueError(f"Camera pose shapes must both be [B,S,9], got {tuple(pred_pose.shape)} and {tuple(gt_pose.shape)}")
    pred = pred_pose.detach().float()
    gt = gt_pose.detach().to(device=pred.device, dtype=torch.float32)

    trans_err = pred[..., :3] - gt[..., :3]
    trans_l2 = torch.linalg.norm(trans_err, dim=-1)
    q_pred = torch.nn.functional.normalize(pred[..., 3:7], dim=-1, eps=1e-6)
    q_gt = torch.nn.functional.normalize(gt[..., 3:7], dim=-1, eps=1e-6)
    quat_dot = (q_pred * q_gt).sum(dim=-1).abs().clamp(0.0, 1.0)
    rot_deg = torch.rad2deg(2.0 * torch.acos(quat_dot))
    fov_err_deg = torch.rad2deg((pred[..., 7:] - gt[..., 7:]).abs())
    fov_pred_deg = torch.rad2deg(pred[..., 7:])
    fov_gt_deg = torch.rad2deg(gt[..., 7:])

    finite = torch.isfinite(pred).all(dim=-1).float()
    return {
        f"{prefix}_t_mae": float(trans_err.abs().mean().item()),
        f"{prefix}_t_rmse": float(trans_err.square().mean().sqrt().item()),
        f"{prefix}_t_l2_mean": float(trans_l2.mean().item()),
        f"{prefix}_rot_deg_mean": float(rot_deg.mean().item()),
        f"{prefix}_rot_deg_max": float(rot_deg.max().item()),
        f"{prefix}_fov_mae_deg": float(fov_err_deg.mean().item()),
        f"{prefix}_fov_rmse_deg": float(fov_err_deg.square().mean().sqrt().item()),
        f"{prefix}_pred_t_norm": float(torch.linalg.norm(pred[..., :3], dim=-1).mean().item()),
        f"{prefix}_gt_t_norm": float(torch.linalg.norm(gt[..., :3], dim=-1).mean().item()),
        f"{prefix}_pred_fov_h_deg": float(fov_pred_deg[..., 0].mean().item()),
        f"{prefix}_gt_fov_h_deg": float(fov_gt_deg[..., 0].mean().item()),
        f"{prefix}_pred_fov_w_deg": float(fov_pred_deg[..., 1].mean().item()),
        f"{prefix}_gt_fov_w_deg": float(fov_gt_deg[..., 1].mean().item()),
        f"{prefix}_finite_frac": float(finite.mean().item()),
    }


def camera_feature_validation_metrics(
    pred_features: torch.Tensor,
    gt_features: torch.Tensor,
    *,
    prefix: str = "sample_camera_feature",
) -> dict[str, float]:
    if pred_features.shape != gt_features.shape:
        raise ValueError(
            f"Camera feature shapes must match, got {tuple(pred_features.shape)} and {tuple(gt_features.shape)}"
        )
    pred = pred_features.detach().float()
    gt = gt_features.detach().to(device=pred.device, dtype=torch.float32)
    err = pred - gt
    return {
        f"{prefix}_mae": float(err.abs().mean().item()),
        f"{prefix}_rmse": float(err.square().mean().sqrt().item()),
        f"{prefix}_pred_norm": float(torch.linalg.norm(pred, dim=-1).mean().item()),
        f"{prefix}_gt_norm": float(torch.linalg.norm(gt, dim=-1).mean().item()),
    }


def _cfg_metric_prefix(base: str, guidance_scale: float) -> str:
    scale = f"{float(guidance_scale):g}".replace("-", "m").replace(".", "p")
    return f"{base}_cfg{scale}"


def sample_uncond_drop_mask(
    batch_size: int,
    prob: float,
    *,
    device: torch.device,
    training: bool = True,
) -> torch.Tensor | None:
    if not bool(training) or float(prob) <= 0.0:
        return None
    return torch.rand(int(batch_size), device=device) < float(prob)


def _kind_list(
    kinds: Any,
    batch_size: int,
    *,
    default: str,
) -> list[str]:
    if kinds is None:
        return [default] * int(batch_size)
    if isinstance(kinds, str):
        return [kinds] * int(batch_size)
    values = list(kinds)
    if len(values) != int(batch_size):
        raise ValueError(f"condition kind length {len(values)} != batch size {batch_size}")
    return [str(v) for v in values]


def _mask_frac(mask: torch.Tensor | None) -> float:
    if mask is None:
        return 0.0
    return float(mask.detach().to(dtype=torch.float32).mean().item())


_ASSET_NULL_OR_MISSING_KINDS = {"none", "asset_uncond", "asset_null", "asset_missing", "missing_asset"}
_ASSET_EXPLICIT_EMPTY_KINDS = {
    "mode_b",
    "mode_b_empty",
    "empty",
    "mode_a_with_empty",
    "mode_a_plus_empty",
    "with_empty",
    "plus_empty",
}
_CAMERA_NULL_OR_MISSING_KINDS = {"camera_uncond", "camera_null", "camera_missing", "missing_camera"}


def _condition_kind_values(kinds: Any, batch_size: int) -> list[str] | None:
    if kinds is None:
        return None
    if isinstance(kinds, str):
        return [kinds.lower()] * int(batch_size)
    if torch.is_tensor(kinds):
        return None
    values = list(kinds)
    if len(values) != int(batch_size):
        raise ValueError(f"condition kind length {len(values)} != batch size {batch_size}")
    return [str(v).lower() for v in values]


def _row_has_any(mask: torch.Tensor | None, batch_size: int) -> list[bool] | None:
    if mask is None or not torch.is_tensor(mask):
        return None
    if int(mask.shape[0]) != int(batch_size):
        raise ValueError(f"condition mask batch size {int(mask.shape[0])} != {int(batch_size)}")
    return mask.detach().to(dtype=torch.bool).reshape(int(batch_size), -1).any(dim=1).cpu().tolist()


def _asset_condition_rows(
    F_asset_tokens: torch.Tensor | None,
    encoder_attention_mask: torch.Tensor | None,
    asset_condition_kind: Any,
    batch_size: int,
) -> list[bool]:
    mask_rows = _row_has_any(encoder_attention_mask, batch_size)
    if mask_rows is not None:
        rows = list(mask_rows)
    elif torch.is_tensor(F_asset_tokens):
        if int(F_asset_tokens.shape[0]) != int(batch_size):
            raise ValueError(f"asset token batch size {int(F_asset_tokens.shape[0])} != {int(batch_size)}")
        has_slots = int(F_asset_tokens.numel()) > 0 and any(int(dim) > 0 for dim in F_asset_tokens.shape[1:-1])
        rows = [bool(has_slots)] * int(batch_size)
    else:
        rows = [False] * int(batch_size)

    kinds = _condition_kind_values(asset_condition_kind, batch_size)
    if kinds is not None:
        for idx, kind in enumerate(kinds):
            if kind in _ASSET_NULL_OR_MISSING_KINDS:
                rows[idx] = False
            elif kind in _ASSET_EXPLICIT_EMPTY_KINDS:
                rows[idx] = True
    return rows


def _camera_condition_rows(
    camera_condition_tokens: torch.Tensor | None,
    camera_attention_mask: torch.Tensor | None,
    camera_condition_kind: Any,
    batch_size: int,
) -> list[bool]:
    if torch.is_tensor(camera_condition_tokens):
        if int(camera_condition_tokens.shape[0]) != int(batch_size):
            raise ValueError(f"camera token batch size {int(camera_condition_tokens.shape[0])} != {int(batch_size)}")
        if camera_condition_tokens.ndim >= 2 and int(camera_condition_tokens.shape[1]) > 0:
            mask_rows = _row_has_any(camera_attention_mask, batch_size)
            rows = list(mask_rows) if mask_rows is not None else [True] * int(batch_size)
        else:
            rows = [False] * int(batch_size)
    else:
        rows = [False] * int(batch_size)

    kinds = _condition_kind_values(camera_condition_kind, batch_size)
    if kinds is not None:
        for idx, kind in enumerate(kinds):
            if kind in _CAMERA_NULL_OR_MISSING_KINDS:
                rows[idx] = False
    return rows


def _condition_kind_with_null_rows(
    kinds: Any,
    rows: list[bool],
    *,
    batch_size: int,
    default: str,
    null_kind: str,
) -> list[str]:
    full = _kind_list(kinds, batch_size, default=default)
    return [kind if bool(rows[idx]) else null_kind for idx, kind in enumerate(full)]


def resolve_pretrain_optional_cfg_conditions(
    bundle: SimpleNamespace,
    batch_size: int,
    *,
    asset_control_scale: float,
    camera_scale: float,
) -> SimpleNamespace:
    asset_rows = _asset_condition_rows(
        getattr(bundle, "F_asset_tokens", None),
        getattr(bundle, "encoder_attention_mask", None),
        getattr(bundle, "asset_condition_kind", None),
        int(batch_size),
    )
    camera_rows = _camera_condition_rows(
        getattr(bundle, "camera_condition_tokens", None),
        getattr(bundle, "camera_attention_mask", None),
        getattr(bundle, "camera_condition_kind", None),
        int(batch_size),
    )
    has_asset = any(asset_rows)
    has_camera = any(camera_rows)
    asset_null_kind = ["asset_uncond"] * int(batch_size)
    camera_null_kind = ["camera_uncond"] * int(batch_size)
    full_asset_kind = _condition_kind_with_null_rows(
        getattr(bundle, "asset_condition_kind", None),
        asset_rows,
        batch_size=int(batch_size),
        default="mode_a",
        null_kind="asset_uncond",
    )
    full_camera_kind = _condition_kind_with_null_rows(
        getattr(bundle, "camera_condition_kind", None),
        camera_rows,
        batch_size=int(batch_size),
        default="camera",
        null_kind="camera_uncond",
    )
    full_camera_tokens = getattr(bundle, "camera_condition_tokens", None)
    full_camera_mask = getattr(bundle, "camera_attention_mask", None)
    if not has_camera:
        full_camera_tokens = None
        full_camera_mask = None
    return SimpleNamespace(
        has_asset_condition=has_asset,
        has_camera_condition=has_camera,
        asset_condition_rows=asset_rows,
        camera_condition_rows=camera_rows,
        asset_control_scale=float(asset_control_scale) if has_asset else 1.0,
        camera_scale=float(camera_scale) if has_camera else 1.0,
        asset_null_kind=asset_null_kind,
        camera_null_kind=camera_null_kind,
        full_asset_kind=full_asset_kind,
        full_camera_kind=full_camera_kind,
        full_camera_tokens=full_camera_tokens,
        full_camera_mask=full_camera_mask,
    )


def apply_asset_uncond_drop(
    bundle: SimpleNamespace,
    drop_mask: torch.Tensor | None,
) -> SimpleNamespace:
    if drop_mask is None or not torch.is_tensor(bundle.F_asset_tokens):
        return bundle
    drop = drop_mask.to(device=bundle.F_asset_tokens.device, dtype=torch.bool).view(-1)
    if not bool(drop.any().item()):
        return bundle
    if int(bundle.F_asset_tokens.shape[1]) > 0:
        if bundle.encoder_attention_mask is None:
            mask = torch.ones(bundle.F_asset_tokens.shape[:-1], device=bundle.F_asset_tokens.device, dtype=torch.bool)
        else:
            mask = bundle.encoder_attention_mask.clone().to(device=bundle.F_asset_tokens.device, dtype=torch.bool)
        mask[drop] = False
        bundle.encoder_attention_mask = mask
    lengths = getattr(bundle, "F_asset_lengths", None)
    if torch.is_tensor(lengths):
        lengths = lengths.clone()
        lengths[drop.to(device=lengths.device)] = 0
        bundle.F_asset_lengths = lengths
    kinds = _kind_list(
        getattr(bundle, "asset_condition_kind", None),
        int(bundle.F_asset_tokens.shape[0]),
        default="mode_a",
    )
    for idx, should_drop in enumerate(drop.detach().cpu().tolist()):
        if should_drop:
            kinds[idx] = "asset_uncond"
    bundle.asset_condition_kind = kinds
    return bundle


def apply_camera_uncond_drop(
    bundle: SimpleNamespace,
    drop_mask: torch.Tensor | None,
) -> SimpleNamespace:
    if drop_mask is None:
        return bundle
    ref = getattr(bundle, "camera_condition_tokens", None)
    if torch.is_tensor(ref):
        device = ref.device
        batch_size = int(ref.shape[0])
    else:
        ref = getattr(bundle, "z_clean_n", None)
        if not torch.is_tensor(ref):
            return bundle
        device = ref.device
        batch_size = int(ref.shape[0])
    drop = drop_mask.to(device=device, dtype=torch.bool).view(-1)
    if not bool(drop.any().item()):
        return bundle
    kinds = _kind_list(
        getattr(bundle, "camera_condition_kind", None),
        batch_size,
        default="camera",
    )
    for idx, should_drop in enumerate(drop.detach().cpu().tolist()):
        if should_drop:
            kinds[idx] = "camera_uncond"
    bundle.camera_condition_kind = kinds
    return bundle


def estimate_sparse_asset_token_count(
    scene_flow_root: nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor | None,
) -> float:
    cfg = getattr(scene_flow_root, "config", None)
    max_patch = int(getattr(cfg, "max_asset_patch_tokens_per_asset_frame", 32))
    max_total = int(getattr(cfg, "max_asset_tokens", 4096))
    if tokens.ndim == 5:
        valid = (
            torch.ones(tokens.shape[:-1], device=tokens.device, dtype=torch.bool)
            if mask is None
            else mask.to(device=tokens.device, dtype=torch.bool)
        )
        counts = valid.sum(dim=-1)
        per_asset_frame = torch.where(
            counts > 0,
            counts.clamp_max(max_patch) + 1,
            torch.zeros_like(counts),
        )
        return float(per_asset_frame.sum(dim=(1, 2)).clamp_max(max_total).float().mean().item())
    if tokens.ndim == 3:
        if mask is None:
            return float(tokens.shape[1])
        return float(mask.to(device=tokens.device, dtype=torch.bool).sum(dim=1).float().mean().item())
    return 0.0


def estimate_control_token_count(
    scene_flow_root: nn.Module,
    M_source: torch.Tensor,
    M_dest: torch.Tensor,
    patch_grid: tuple[int, int],
) -> float:
    cfg = getattr(scene_flow_root, "config", None)
    max_per_frame = int(getattr(cfg, "max_control_tokens_per_frame", 128))
    max_total = int(getattr(cfg, "max_control_tokens", 1024))
    M_edit = (M_source.float() + M_dest.float()).clamp(0.0, 1.0)
    if bool((M_edit > 0.999).all().item()):
        return 0.0
    b, s, p, _ = M_edit.shape
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    grid = M_edit.reshape(b * s, gh, gw, 1).permute(0, 3, 1, 2)
    support = torch.nn.functional.max_pool2d(grid, kernel_size=5, stride=1, padding=2).gt(0.0)
    counts = support.reshape(b, s, p).sum(dim=-1)
    return float(counts.clamp_max(max_per_frame).sum(dim=1).clamp_max(max_total).float().mean().item())


def build_training_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> LambdaLR:
    decay_end_steps = int(args.decay_end_steps) if int(args.decay_end_steps) > 0 else int(args.max_steps)
    return build_rae_scheduler(
        optimizer,
        scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
        decay_end_steps=decay_end_steps,
        base_lr=args.lr,
        final_lr=args.final_lr,
        warmup_from_zero=args.warmup_from_zero,
    )


def save_checkpoint(
    scene_flow: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    step: int,
    log_dir: Path,
    args: argparse.Namespace,
) -> None:
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sf = unwrap_ddp(scene_flow)
    scene_flow_state = sf.state_dict()
    ema_scene_flow_state = materialize_ema_state_dict(scene_flow, ema)
    scene_flow_config = sf.config.to_dict() if hasattr(sf, "config") and hasattr(sf.config, "to_dict") else {}
    provenance = {
        "dggt_checkpoint_sha256": str(args.dggt_checkpoint_sha256),
        "camera_generation_representation": CAMERA_GENERATION_REPRESENTATION,
        "camera_target_source": "frozen_dggt_camera_head",
    }
    payload = {
        "step": int(step),
        "scene_flow": scene_flow_state,
        "scene_flow_config": scene_flow_config,
        "ema_scene_flow": ema.state_dict(),
        "ema_scene_flow_state_dict": ema_scene_flow_state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": vars(args),
        "camera_dggt_provenance": provenance,
    }
    torch.save(payload, ckpt_dir / f"pretrain_step{step:06d}.pt")
    torch.save(
        {
            "scene_flow": scene_flow_state,
            "scene_flow_config": scene_flow_config,
            "camera_dggt_provenance": provenance,
        },
        ckpt_dir / f"pretrain_step{step:06d}_weights_only.pt",
    )
    torch.save(
        {
            "scene_flow": ema_scene_flow_state,
            "scene_flow_config": scene_flow_config,
            "step": int(step),
            "is_ema_weights": True,
            "camera_dggt_provenance": provenance,
        },
        ckpt_dir / f"pretrain_step{step:06d}_ema_weights_only.pt",
    )


def load_resume_checkpoint(
    scene_flow: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    resume_path: str | None,
    device: torch.device,
    expected_dggt_sha256: str | None = None,
) -> int:
    if not resume_path:
        return 0
    payload = torch.load(resume_path, map_location=device)
    if not isinstance(payload, dict) or "scene_flow" not in payload:
        raise ValueError(f"Unsupported resume checkpoint format: {resume_path}")
    validate_scene_flow_checkpoint_config(
        scene_flow, payload, resume_path, expected_dggt_sha256=expected_dggt_sha256
    )
    required_keys = {"step", "scene_flow", "ema_scene_flow", "optimizer", "lr_scheduler"}
    missing_keys = sorted(required_keys.difference(payload.keys()))
    if missing_keys:
        raise ValueError(
            f"`--resume_path` requires a full training checkpoint, but {resume_path} "
            f"is missing keys: {missing_keys}. Do not pass *_weights_only.pt or "
            f"*_ema_weights_only.pt to --resume_path; those files are for warm-start "
            f"or inference, not exact training resume."
        )
    unwrap_ddp(scene_flow).load_state_dict(payload["scene_flow"], strict=True)
    ema.load_state_dict(payload["ema_scene_flow"])
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    step = int(payload.get("step", 0))
    if is_main_process():
        print(f"[resume] loaded {resume_path} at step={step}", flush=True)
    return step


def init_wandb(args: argparse.Namespace, log_dir: Path):
    if not args.wandb or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or remove --wandb.") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        dir=str(log_dir),
        config=vars(args),
        id=args.wandb_run_id,
        resume=args.wandb_resume,
    )
    return run


def log_wandb(run, metrics: dict[str, float], step: int, prefix: str) -> None:
    if run is None:
        return
    run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)


def setup_text_encoder(args: argparse.Namespace, device: torch.device) -> nn.Module | None:
    if bool(getattr(args, "no_text_condition", False)):
        return None
    if not getattr(args, "caption_root", None):
        return None
    encoder = TextEncoder(
        model_name=str(args.text_encoder_path),
        max_length=int(args.text_max_length),
    ).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_text_condition(
    text_encoder: nn.Module | None,
    captions: list[str] | tuple[str, ...] | None,
    drop_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if text_encoder is None:
        return None, None
    if captions is None:
        raise RuntimeError("Text encoder is enabled but the batch has no captions.")
    clean = [str(c) if c is not None else "" for c in captions]
    if drop_mask is not None and int(drop_mask.numel()) != len(clean):
        raise ValueError(f"drop_mask has {int(drop_mask.numel())} rows, captions has {len(clean)}")
    out = text_encoder(clean)
    tokens = out["tokens"]
    attention_mask = out["attention_mask"]
    if drop_mask is None or not bool(drop_mask.to(dtype=torch.bool).any().item()):
        return tokens, attention_mask
    null_out = text_encoder([""] * len(clean))
    drop = drop_mask.to(device=tokens.device, dtype=torch.bool).view(len(clean), 1, 1)
    tokens = torch.where(drop, null_out["tokens"].to(device=tokens.device, dtype=tokens.dtype), tokens)
    mask_drop = drop_mask.to(device=attention_mask.device, dtype=torch.bool).view(len(clean), 1)
    attention_mask = torch.where(
        mask_drop,
        null_out["attention_mask"].to(device=attention_mask.device, dtype=attention_mask.dtype),
        attention_mask,
    )
    return tokens, attention_mask


def captions_from_pretrain_batch(batch: dict[str, Any], batch_size: int) -> list[str]:
    captions = batch.get("caption")
    if captions is None:
        return [""] * int(batch_size)
    if isinstance(captions, str):
        return [captions]
    return [str(c) if c is not None else "" for c in list(captions)]


def _latent_pca_grid(z: torch.Tensor, patch_grid: tuple[int, int], max_frames: int) -> torch.Tensor:
    """Project `[B,S,P,C]` latent tokens to an RGB patch grid for qualitative checks."""
    z = z[:1, :max_frames].detach().float().cpu()
    _, seq_len, num_patches, channels = z.shape
    gy, gx = patch_grid
    if num_patches != gy * gx:
        raise ValueError(f"latent patch count {num_patches} != patch_grid {patch_grid}")
    flat = z.reshape(-1, channels)
    flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.shape[0] < 3:
        rgb = flat[:, :3]
    else:
        _, _, vh = torch.pca_lowrank(flat, q=3, center=False)
        rgb = flat @ vh[:, :3]
    lo = rgb.quantile(0.01, dim=0, keepdim=True)
    hi = rgb.quantile(0.99, dim=0, keepdim=True)
    rgb = ((rgb - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    return rgb.reshape(1, seq_len, gy, gx, 3).reshape(seq_len, gy, gx, 3).permute(0, 3, 1, 2)


def _mask_grid(mask: torch.Tensor, patch_grid: tuple[int, int], max_frames: int) -> torch.Tensor:
    mask = mask[:1, :max_frames].detach().float().cpu()
    _, seq_len, num_patches, _ = mask.shape
    gy, gx = patch_grid
    if num_patches != gy * gx:
        raise ValueError(f"mask patch count {num_patches} != patch_grid {patch_grid}")
    return mask.reshape(seq_len, gy, gx, 1).permute(0, 3, 1, 2)


def _normalized_mask_grid(mask: torch.Tensor, patch_grid: tuple[int, int], max_frames: int) -> torch.Tensor:
    grid = _mask_grid(mask, patch_grid, max_frames)
    hi = grid.quantile(0.99).clamp_min(1e-6)
    return (grid / hi).clamp(0.0, 1.0)


def _image_grid(images: torch.Tensor, max_frames: int) -> torch.Tensor:
    return images[:1, :max_frames].detach().float().cpu().reshape(-1, *images.shape[2:]).clamp(0.0, 1.0)


def _slice_batch_for_visualization(batch: dict[str, Any], max_samples: int = 1) -> dict[str, Any]:
    """Keep qualitative validation bounded even when the val loader batch is large."""
    images = batch.get("images")
    if not torch.is_tensor(images) or images.ndim == 0:
        return batch
    batch_size = int(images.shape[0])
    keep = max(1, min(int(max_samples), batch_size))

    def maybe_slice(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                return value[:keep]
            return value
        if isinstance(value, dict):
            return {key: maybe_slice(item) for key, item in value.items()}
        if isinstance(value, list) and len(value) == batch_size:
            return value[:keep]
        if isinstance(value, tuple) and len(value) == batch_size:
            return value[:keep]
        return value

    return {key: maybe_slice(value) for key, value in batch.items()}


def _semantic_logits_to_sky_mask(
    semantic_logits: torch.Tensor,
    *,
    sky_class_index: int = SKY_CLASS_INDEX,
) -> torch.Tensor:
    """Convert predicted semantic logits `[B,S,H,W,C]` to sky mask `[B,S,3,H,W]`."""
    if semantic_logits.ndim != 5:
        raise ValueError(f"Expected semantic_logits [B,S,H,W,C], got {tuple(semantic_logits.shape)}")
    if semantic_logits.shape[-1] <= int(sky_class_index):
        raise ValueError(
            f"semantic_logits has {semantic_logits.shape[-1]} classes, "
            f"cannot read sky_class_index={sky_class_index}"
        )
    sky = (semantic_logits.float().argmax(dim=-1) == int(sky_class_index)).to(dtype=semantic_logits.dtype)
    return sky[:, :, None].repeat(1, 1, 3, 1, 1)


def _sky_mask_image_grid(sky_mask: torch.Tensor, max_frames: int) -> torch.Tensor:
    mask = sky_mask[:1, :max_frames, :1].detach().float().cpu()
    return mask.reshape(-1, *mask.shape[2:]).clamp(0.0, 1.0)


def sky_generation_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "no_sky_generation", False))


def sky_grid_shape(args: argparse.Namespace) -> tuple[int, int]:
    h = int(getattr(args, "sky_grid_h", DEFAULT_SKY_GRID[0]))
    w = int(getattr(args, "sky_grid_w", DEFAULT_SKY_GRID[1]))
    if h <= 0 or w <= 0:
        raise ValueError(f"sky grid must be positive, got {(h, w)}")
    return h, w


def sky_atlas_shape(args: argparse.Namespace) -> tuple[int, int]:
    value = getattr(args, "sky_atlas_hw", DEFAULT_SKY_ATLAS_HW)
    if value is None:
        value = DEFAULT_SKY_ATLAS_HW
    h, w = (int(v) for v in value)
    if h <= 0 or w <= 0:
        raise ValueError(f"sky atlas must be positive, got {(h, w)}")
    return h, w


def pack_sky_rgb_atlas(atlas_rgb: torch.Tensor) -> torch.Tensor:
    """Deterministically pack a 32x64 RGB atlas into 512 12D tokens."""
    if atlas_rgb.ndim != 4 or int(atlas_rgb.shape[1]) != SKY_RGB_DIM:
        raise ValueError(f"sky atlas must be [B,3,H,W], got {tuple(atlas_rgb.shape)}")
    ah, aw = int(atlas_rgb.shape[-2]), int(atlas_rgb.shape[-1])
    if (ah, aw) != DEFAULT_SKY_ATLAS_HW:
        raise ValueError(f"sky atlas must be {DEFAULT_SKY_ATLAS_HW}, got {(ah, aw)}")
    packed = torch.nn.functional.pixel_unshuffle(atlas_rgb.float() * 2.0 - 1.0, SKY_PATCH_SIZE)
    return packed.permute(0, 2, 3, 1).reshape(atlas_rgb.shape[0], -1, SKY_TOKEN_DIM).contiguous()


def decode_sky_patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Decode [B,512,12] tokens to flattened [-1,1] 32x64 RGB atlas."""
    if tokens.ndim != 3 or int(tokens.shape[1]) != DEFAULT_SKY_GRID[0] * DEFAULT_SKY_GRID[1] or int(tokens.shape[2]) != SKY_TOKEN_DIM:
        raise ValueError(
            f"sky tokens must be [B,{DEFAULT_SKY_GRID[0] * DEFAULT_SKY_GRID[1]},{SKY_TOKEN_DIM}], "
            f"got {tuple(tokens.shape)}"
        )
    packed = tokens.reshape(tokens.shape[0], *DEFAULT_SKY_GRID, SKY_TOKEN_DIM)
    packed = packed.permute(0, 3, 1, 2).contiguous()
    atlas = torch.nn.functional.pixel_shuffle(packed, SKY_PATCH_SIZE)
    return atlas.permute(0, 2, 3, 1).reshape(tokens.shape[0], -1, SKY_RGB_DIM).contiguous()


def _sky_mask_1ch(masks: torch.Tensor | None, images: torch.Tensor) -> torch.Tensor:
    if masks is None:
        return torch.zeros(
            images.shape[:2] + (1,) + images.shape[-2:],
            device=images.device,
            dtype=images.dtype,
        )
    mask = masks.to(device=images.device, dtype=images.dtype)
    if mask.ndim == 4:
        mask = mask.unsqueeze(0)
    if mask.ndim != 5:
        raise ValueError(f"Expected sky mask [B,S,C,H,W], got {tuple(mask.shape)}")
    if mask.shape[:2] != images.shape[:2] or mask.shape[-2:] != images.shape[-2:]:
        raise ValueError(f"sky mask shape {tuple(mask.shape)} incompatible with images {tuple(images.shape)}")
    if int(mask.shape[2]) == 1:
        return mask.clamp(0.0, 1.0)
    return mask[:, :, :1].clamp(0.0, 1.0)


def build_sky_mask_patch_target(
    images: torch.Tensor,
    masks: torch.Tensor | None,
    *,
    patch_grid: tuple[int, int] | list[int],
) -> torch.Tensor | None:
    """Pool GT sky masks to the SceneFlow patch grid as `[B,S,P,1]` soft targets."""
    if masks is None:
        return None
    if images.ndim != 5:
        raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    if gh <= 0 or gw <= 0:
        raise ValueError(f"patch_grid must be positive, got {tuple(patch_grid)}")
    sky = _sky_mask_1ch(masks, images)
    b, s, _, height, width = sky.shape
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        sky.float().reshape(b * s, 1, height, width),
        (gh, gw),
    )
    return pooled.reshape(b, s, gh * gw, 1).to(device=images.device, dtype=images.dtype).clamp(0.0, 1.0)


def sky_mask_refine_shape(args: argparse.Namespace) -> tuple[int, int]:
    scale = int(getattr(args, "sky_mask_refine_scale", 4))
    if scale <= 0 or scale & (scale - 1):
        raise ValueError(f"--sky_mask_refine_scale must be a positive power of two, got {scale}")
    gh, gw = int(args.patch_grid[0]), int(args.patch_grid[1])
    return gh * scale, gw * scale


def build_sky_mask_refined_target(
    images: torch.Tensor,
    masks: torch.Tensor | None,
    *,
    refined_hw: tuple[int, int],
) -> torch.Tensor | None:
    """Pool GT sky masks to the refined dense decoder target `[B,S,1,Hr,Wr]`."""
    if masks is None:
        return None
    if images.ndim != 5:
        raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
    rh, rw = int(refined_hw[0]), int(refined_hw[1])
    if rh <= 0 or rw <= 0:
        raise ValueError(f"refined sky mask size must be positive, got {tuple(refined_hw)}")
    sky = _sky_mask_1ch(masks, images)
    b, s, _, height, width = sky.shape
    target = torch.nn.functional.adaptive_avg_pool2d(
        sky.float().reshape(b * s, 1, height, width),
        (rh, rw),
    )
    return target.reshape(b, s, 1, rh, rw).to(device=images.device, dtype=images.dtype).clamp(0.0, 1.0)


def _sky_mask_pos_weight(
    target: torch.Tensor,
    *,
    pos_weight_max: float,
) -> torch.Tensor:
    pos = target.sum()
    neg = target.numel() - pos
    if bool(pos.gt(0.0).item()) and bool(neg.gt(0.0).item()):
        return (neg / pos.clamp_min(1.0)).clamp(1.0, float(pos_weight_max))
    return target.new_tensor(1.0)


def sky_mask_patch_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float,
    pos_weight_max: float,
    log_prefix: str = "sky_mask",
) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.shape != target.shape:
        raise ValueError(f"sky mask logits shape {tuple(logits.shape)} != target {tuple(target.shape)}")
    logits_f = logits.float()
    target_f = target.to(device=logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    pos_weight = _sky_mask_pos_weight(target_f, pos_weight_max=pos_weight_max)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_f,
        target_f,
        pos_weight=pos_weight,
    )
    prob = torch.sigmoid(logits_f)
    intersection = (prob * target_f).sum()
    denom = prob.sum() + target_f.sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    loss = bce + float(dice_weight) * dice
    pred_hard = prob.ge(0.5)
    target_hard = target_f.ge(0.5)
    union = (pred_hard | target_hard).float().sum()
    if bool(union.eq(0.0).item()):
        iou = logits_f.new_tensor(1.0)
    else:
        iou = ((pred_hard & target_hard).float().sum() / union).detach()
    return loss, {
        f"loss_{log_prefix}_bce": float(bce.detach().item()),
        f"loss_{log_prefix}_dice": float(dice.detach().item()),
        f"{log_prefix}_pos_weight": float(pos_weight.detach().item()),
        f"{log_prefix}_target_frac": float(target_f.mean().detach().item()),
        f"{log_prefix}_pred_frac": float(prob.mean().detach().item()),
        f"{log_prefix}_iou": float(iou.item()),
    }


def _sky_mask_boundary_band(target: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if target.ndim != 5 or int(target.shape[2]) != 1:
        raise ValueError(f"refined sky mask target must be [B,S,1,H,W], got {tuple(target.shape)}")
    b, s, _, h, w = target.shape
    hard = target.float().ge(0.5).to(dtype=torch.float32).reshape(b * s, 1, h, w)
    k = 2 * int(radius) + 1
    dilated = torch.nn.functional.max_pool2d(hard, kernel_size=k, stride=1, padding=int(radius))
    eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - hard, kernel_size=k, stride=1, padding=int(radius))
    return (dilated - eroded).clamp(0.0, 1.0).reshape(b, s, 1, h, w)


def sky_mask_refined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float,
    pos_weight_max: float,
    boundary_weight: float,
    boundary_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    region_loss, logs = sky_mask_patch_loss(
        logits,
        target,
        dice_weight=dice_weight,
        pos_weight_max=pos_weight_max,
        log_prefix="sky_mask_refine",
    )
    logits_f = logits.float()
    target_f = target.to(device=logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    pos_weight = _sky_mask_pos_weight(target_f, pos_weight_max=pos_weight_max)
    bce_px = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_f,
        target_f,
        pos_weight=pos_weight,
        reduction="none",
    )
    band = _sky_mask_boundary_band(target_f, radius=1).to(device=logits.device, dtype=torch.float32)
    if bool(band.sum().gt(0.0).item()):
        boundary_bce = (bce_px * band).sum() / band.sum().clamp_min(1.0)
    else:
        boundary_bce = logits_f.sum() * 0.0
    loss = region_loss + float(boundary_loss_weight) * float(boundary_weight) * boundary_bce
    logs["loss_sky_mask_refine_boundary_bce"] = float(boundary_bce.detach().item())
    logs["sky_mask_refine_boundary_frac"] = float(band.mean().detach().item())
    return loss, logs


def sky_mask_validation_metrics(
    pred_patch: torch.Tensor | None,
    target_patch: torch.Tensor | None,
    *,
    prefix: str,
) -> dict[str, float]:
    if pred_patch is None or target_patch is None:
        return {}
    if pred_patch.shape != target_patch.shape:
        raise ValueError(f"sky mask pred shape {tuple(pred_patch.shape)} != target {tuple(target_patch.shape)}")
    pred = pred_patch.detach().float().ge(0.5)
    target = target_patch.detach().float().ge(0.5)
    union = (pred | target).float().sum()
    inter = (pred & target).float().sum()
    iou = torch.ones((), device=union.device) if bool(union.eq(0.0).item()) else inter / union
    return {
        f"{prefix}_iou": float(iou.item()),
        f"{prefix}_pred_frac": float(pred.float().mean().item()),
        f"{prefix}_target_frac": float(target.float().mean().item()),
    }


def _sky_mask_patch_to_image(
    sky_mask_patch: torch.Tensor,
    *,
    patch_grid: tuple[int, int] | list[int],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if sky_mask_patch.ndim == 5:
        mask = sky_mask_patch.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        b, s, c, h, w = mask.shape
        if int(c) not in (1, 3):
            raise ValueError(f"sky_mask_patch image tensor must have 1 or 3 channels, got {tuple(mask.shape)}")
        if (int(h), int(w)) != (int(height), int(width)):
            mask = torch.nn.functional.interpolate(
                mask.reshape(b * s, c, h, w),
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, s, c, int(height), int(width)).clamp(0.0, 1.0)
        if int(c) == 1:
            return mask.repeat(1, 1, 3, 1, 1)
        if int(c) == 3:
            return mask
    if sky_mask_patch.ndim != 4 or int(sky_mask_patch.shape[-1]) != 1:
        raise ValueError(f"sky_mask_patch must be [B,S,P,1] or [B,S,C,H,W], got {tuple(sky_mask_patch.shape)}")
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    b, s, p, _ = sky_mask_patch.shape
    if p != gh * gw:
        raise ValueError(f"sky_mask_patch P={p} does not match patch_grid={tuple(patch_grid)}")
    patch = sky_mask_patch.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    image = torch.nn.functional.interpolate(
        patch.reshape(b * s, gh, gw, 1).permute(0, 3, 1, 2),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).reshape(b, s, 1, int(height), int(width))
    return image.repeat(1, 1, 3, 1, 1).clamp(0.0, 1.0)


def _as_batched_camera_mats(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    if extrinsics.ndim == 3:
        extrinsics = extrinsics.unsqueeze(0)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics.unsqueeze(0)
    if extrinsics.ndim != 4:
        raise ValueError(f"Expected extrinsics [B,S,3/4,4] or [S,3/4,4], got {tuple(extrinsics.shape)}")
    if intrinsics.ndim != 4:
        raise ValueError(f"Expected intrinsics [B,S,3/4,3/4] or [S,3/4,3/4], got {tuple(intrinsics.shape)}")

    if extrinsics.shape[-2:] == (3, 4):
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 4)
        bottom = bottom.expand(extrinsics.shape[0], extrinsics.shape[1], -1, -1)
        extrinsics = torch.cat([extrinsics, bottom], dim=-2)
    elif extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsics trailing shape [3,4] or [4,4], got {tuple(extrinsics.shape)}")

    if intrinsics.shape[-2:] == (4, 4):
        intrinsics = intrinsics[..., :3, :3]
    elif intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics trailing shape [3,3] or [4,4], got {tuple(intrinsics.shape)}")

    if int(extrinsics.shape[1]) != int(seq_len) or int(intrinsics.shape[1]) != int(seq_len):
        raise ValueError(
            f"Camera sequence length must be {seq_len}, got extrinsics={tuple(extrinsics.shape)} "
            f"intrinsics={tuple(intrinsics.shape)}"
        )
    if int(extrinsics.shape[0]) == 1 and batch_size > 1:
        extrinsics = extrinsics.expand(batch_size, -1, -1, -1)
    if int(intrinsics.shape[0]) == 1 and batch_size > 1:
        intrinsics = intrinsics.expand(batch_size, -1, -1, -1)
    if int(extrinsics.shape[0]) != int(batch_size) or int(intrinsics.shape[0]) != int(batch_size):
        raise ValueError(
            f"Camera batch size must be {batch_size}, got extrinsics={tuple(extrinsics.shape)} "
            f"intrinsics={tuple(intrinsics.shape)}"
        )
    return extrinsics.contiguous(), intrinsics.contiguous()


def _sky_direction_grid(
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return DGGT-style upper-hemisphere world directions `[K,3]`.

    Rows run from zenith (`y=-1`) to horizon (`y=0`) and columns cover full
    azimuth. The sign convention follows DGGT/OpenCV cameras where image-up
    rays have negative camera/world y under identity pose.
    """
    row = (torch.arange(int(grid_h), device=device, dtype=dtype) + 0.5) / float(grid_h)
    col = (torch.arange(int(grid_w), device=device, dtype=dtype) + 0.5) / float(grid_w)
    elevation = (1.0 - row) * (float(np.pi) * 0.5)
    azimuth = col * (float(np.pi) * 2.0) - float(np.pi)
    horiz = torch.cos(elevation)
    y = -torch.sin(elevation)
    x = horiz[:, None] * torch.cos(azimuth)[None, :]
    z = horiz[:, None] * torch.sin(azimuth)[None, :]
    dirs = torch.stack([x, y[:, None].expand_as(x), z], dim=-1)
    return torch.nn.functional.normalize(dirs.reshape(int(grid_h) * int(grid_w), 3), dim=-1)


def _build_directional_sky_tokens_from_images(
    images: torch.Tensor,
    masks: torch.Tensor | None,
    *,
    grid_h: int,
    grid_w: int,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_threshold: float,
    unobserved_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s, _, height, width = images.shape
    sky = _sky_mask_1ch(masks, images)
    work_dtype = torch.float32
    extrinsics4, intrinsics3 = _as_batched_camera_mats(
        extrinsics,
        intrinsics,
        batch_size=b,
        seq_len=s,
        device=images.device,
        dtype=work_dtype,
    )
    dirs = _sky_direction_grid(grid_h, grid_w, device=images.device, dtype=work_dtype)
    k = int(dirs.shape[0])
    # The sky atlas is an environment map at infinity.  Its renderer below
    # transforms camera rays by rotation only, so target construction must do
    # the exact inverse operation: world directions -> camera directions using
    # R_world_to_camera.  Treating the atlas as a finite sphere and applying
    # camera translation makes the learned target change when the camera moves
    # even though the renderer is translation invariant.
    cam_dirs = torch.einsum("bsij,kj->bski", extrinsics4[..., :3, :3], dirs)
    z = cam_dirs[..., 2]
    z_safe = z.clamp_min(1e-6)
    px = intrinsics3[..., 0, 0].unsqueeze(-1) * (cam_dirs[..., 0] / z_safe) + intrinsics3[..., 0, 2].unsqueeze(-1)
    py = intrinsics3[..., 1, 1].unsqueeze(-1) * (cam_dirs[..., 1] / z_safe) + intrinsics3[..., 1, 2].unsqueeze(-1)
    visible = z.gt(1e-6) & px.ge(0.0) & px.le(float(width - 1)) & py.ge(0.0) & py.le(float(height - 1))

    x_norm = (2.0 * px / float(max(width - 1, 1))) - 1.0
    y_norm = (2.0 * py / float(max(height - 1, 1))) - 1.0
    sample_grid = torch.stack([x_norm, y_norm], dim=-1).reshape(b * s, 1, k, 2)
    flat_images = images.float().reshape(b * s, 3, height, width)
    flat_sky = sky.float().reshape(b * s, 1, height, width)
    sampled_rgb = torch.nn.functional.grid_sample(
        flat_images,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(b, s, 3, k).permute(0, 1, 3, 2)
    sampled_sky = torch.nn.functional.grid_sample(
        flat_sky,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(b, s, k)
    # Keep one sharp observation per direction. Averaging across frames turns
    # small CameraHead rotation errors into visibly blurred cloud texture.
    confidence = sampled_sky.clamp(0.0, 1.0) * visible.to(dtype=sampled_sky.dtype)
    best_confidence, best_frame = confidence.max(dim=1)
    gather_index = best_frame[:, None, :, None].expand(b, 1, k, 3)
    rgb = sampled_rgb.gather(1, gather_index).squeeze(1)
    coverage_valid = best_confidence.gt(float(valid_threshold))
    # Unknown atlas cells carry no supervision. Their RGB payload is a stable
    # zero placeholder and is disambiguated by the tokenizer observation mask.
    rgb = torch.where(coverage_valid.unsqueeze(-1), rgb, torch.zeros_like(rgb))
    rgb = rgb.clamp(0.0, 1.0)
    loss_weight = torch.where(
        coverage_valid,
        torch.ones_like(best_confidence, dtype=torch.float32),
        torch.full_like(best_confidence, float(unobserved_weight), dtype=torch.float32),
    )
    tokens = rgb * 2.0 - 1.0
    return tokens.to(dtype=images.dtype), loss_weight.to(device=images.device, dtype=torch.float32)


def build_sky_tokens_from_images(
    images: torch.Tensor,
    masks: torch.Tensor | None,
    *,
    grid_h: int,
    grid_w: int,
    valid_threshold: float = 0.05,
    unobserved_weight: float = DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT,
    extrinsics: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build scene-level sky atlas targets `[B,K,3]`.

    GT sky masks are used only to estimate RGB targets and per-token loss
    confidence. They are never used as model attention masks, so training uses
    the same full-sky-token surface as open inference. Channels are RGB in
    `[-1,1]`.

    When DGGT-space camera matrices are provided, each token is an upper-
    hemisphere direction bin. The target color comes from the visible frame
    with the highest sky confidence; frames are never averaged.
    Without camera matrices the legacy image-grid averaging path is used.
    """
    if images.ndim != 5:
        raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
    if int(images.shape[2]) != 3:
        raise ValueError(f"Expected RGB images, got C={images.shape[2]}")
    b, s, _, height, width = images.shape
    grid_h, grid_w = int(grid_h), int(grid_w)
    if (extrinsics is None) != (intrinsics is None):
        raise ValueError("extrinsics and intrinsics must be provided together for directional sky tokens.")
    if extrinsics is not None and intrinsics is not None:
        return _build_directional_sky_tokens_from_images(
            images,
            masks,
            grid_h=grid_h,
            grid_w=grid_w,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            valid_threshold=valid_threshold,
            unobserved_weight=unobserved_weight,
        )
    sky = _sky_mask_1ch(masks, images)
    flat_images = images.float().reshape(b * s, 3, height, width)
    flat_sky = sky.float().reshape(b * s, 1, height, width)
    weighted_rgb = torch.nn.functional.interpolate(
        flat_images,
        size=(grid_h, grid_w),
        mode="area",
    ).reshape(b, s, 3, grid_h, grid_w)
    coverage = torch.nn.functional.interpolate(
        flat_sky,
        size=(grid_h, grid_w),
        mode="area",
    ).reshape(b, s, 1, grid_h, grid_w)
    best_confidence, best_frame = coverage.squeeze(2).max(dim=1)
    gather_index = best_frame[:, None, None].expand(b, 1, 3, grid_h, grid_w)
    rgb = weighted_rgb.gather(1, gather_index).squeeze(1)
    coverage_valid = best_confidence.gt(float(valid_threshold))
    rgb = torch.where(coverage_valid[:, None], rgb, torch.zeros_like(rgb))
    rgb = rgb.clamp(0.0, 1.0)
    loss_weight = torch.where(
        coverage_valid,
        torch.ones_like(best_confidence, dtype=torch.float32),
        torch.full_like(best_confidence, float(unobserved_weight), dtype=torch.float32),
    ).to(device=images.device)
    tokens = (rgb * 2.0 - 1.0).permute(0, 2, 3, 1).reshape(b, grid_h * grid_w, SKY_RGB_DIM)
    return tokens.to(dtype=images.dtype), loss_weight.reshape(b, grid_h * grid_w)


def build_sky_atlas_from_images(
    images: torch.Tensor,
    masks: torch.Tensor | None,
    *,
    atlas_hw: tuple[int, int] = DEFAULT_SKY_ATLAS_HW,
    valid_threshold: float = 0.05,
    extrinsics: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a sharp RGB atlas and its binary observation mask.

    RGB is in [0,1] and shaped [B,3,H,W]; unknown cells are zeros and must only
    be consumed together with the returned [B,1,H,W] mask.
    """
    ah, aw = (int(v) for v in atlas_hw)
    tokens, observed = build_sky_tokens_from_images(
        images,
        masks,
        grid_h=ah,
        grid_w=aw,
        valid_threshold=float(valid_threshold),
        unobserved_weight=0.0,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    atlas = ((tokens.float() + 1.0) * 0.5).reshape(images.shape[0], ah, aw, 3)
    atlas = atlas.permute(0, 3, 1, 2).contiguous()
    observation = observed.reshape(images.shape[0], 1, ah, aw).to(dtype=atlas.dtype)
    atlas = atlas * observation
    return atlas.to(dtype=images.dtype), observation


def build_sky_rectified_flow_target(
    sky_clean: torch.Tensor | None,
    video_target,
    loss_weight: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> SimpleNamespace | None:
    if sky_clean is None:
        return None
    if sky_clean.ndim != 3 or int(sky_clean.shape[-1]) not in (SKY_RGB_DIM, SKY_TOKEN_DIM):
        raise ValueError(
            f"sky_clean must be legacy RGB [B,K,{SKY_RGB_DIM}] or latent [B,K,{SKY_TOKEN_DIM}], "
            f"got {tuple(sky_clean.shape)}"
        )
    b = int(sky_clean.shape[0])
    if video_target.sigmas.shape != (b,):
        raise ValueError(f"video_target sigmas shape {tuple(video_target.sigmas.shape)} != {(b,)}")
    if loss_weight is None and attention_mask is not None:
        # Backward compatibility for older callers/tests. This is now a loss
        # confidence, not a model attention mask.
        loss_weight = attention_mask
    if loss_weight is None:
        token_weight = torch.ones(sky_clean.shape[:2], device=sky_clean.device, dtype=sky_clean.dtype)
    else:
        token_weight = loss_weight.to(device=sky_clean.device, dtype=sky_clean.dtype).clamp_min(0.0)
        if token_weight.shape != sky_clean.shape[:2]:
            raise ValueError(f"sky loss weight shape {tuple(token_weight.shape)} != {tuple(sky_clean.shape[:2])}")
    if not bool(token_weight.gt(0.0).any().item()):
        return None
    sigmas = video_target.sigmas.to(device=sky_clean.device, dtype=sky_clean.dtype)
    sigmas3 = sigmas.view(b, 1, 1)
    eps = torch.randn_like(sky_clean)
    z_t = (1.0 - sigmas3) * sky_clean + sigmas3 * eps
    # The path is affine in sigma, hence its translation/velocity is exactly
    # eps - clean for every sigma.  Recovering it by division through a
    # t_eps-clamped sigma incorrectly shrinks the target when sigma < t_eps.
    v_gt = eps - sky_clean
    return SimpleNamespace(
        sigmas=video_target.sigmas,
        sigmas4=sigmas3,
        z_t=z_t,
        v_gt=v_gt,
        eps=eps,
        weights=torch.ones((b, 1, 1), device=sky_clean.device, dtype=sky_clean.dtype),
        t_eps=SKY_FLOW_T_EPS,
        loss_weight=token_weight,
    )


def sky_flow_loss(
    v_sky_pred: torch.Tensor,
    sky_target,
    sky_clean: torch.Tensor,
) -> torch.Tensor:
    del sky_clean
    token_weight = getattr(sky_target, "loss_weight", None)
    if token_weight is None:
        return torch.nn.functional.mse_loss(
            v_sky_pred.float(),
            sky_target.v_gt.to(device=v_sky_pred.device, dtype=torch.float32),
        )
    weight = token_weight.to(device=v_sky_pred.device, dtype=torch.float32).clamp_min(0.0).unsqueeze(-1)
    if not bool(weight.gt(0.0).any().item()):
        return v_sky_pred.sum() * 0.0
    diff = (
        v_sky_pred.float()
        - sky_target.v_gt.to(device=v_sky_pred.device, dtype=torch.float32)
    ).square()
    denom = weight.sum().clamp_min(1e-6) * float(v_sky_pred.shape[-1])
    return (diff * weight).sum() / denom


def _sky_grid_for_token_count(num_tokens: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    if int(grid_h) * int(grid_w) == int(num_tokens):
        return int(grid_h), int(grid_w)
    h = max(1, int(num_tokens**0.5))
    while h > 1 and int(num_tokens) % h != 0:
        h -= 1
    return h, int(num_tokens) // h


def render_sky_tokens_directional_background(
    sky_tokens: torch.Tensor,
    *,
    seq_len: int,
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    if sky_tokens.ndim != 3 or int(sky_tokens.shape[-1]) < 3:
        raise ValueError(f"sky_tokens must be [B,K,C>=3], got {tuple(sky_tokens.shape)}")
    tokens = sky_tokens[:1].float()
    gh, gw = _sky_grid_for_token_count(int(tokens.shape[1]), int(grid_h), int(grid_w))
    rgb = ((tokens[..., :3] + 1.0) * 0.5).clamp(0.0, 1.0)
    rgb_grid = rgb.reshape(1, gh, gw, 3).permute(0, 3, 1, 2).contiguous()
    rgb_grid = torch.cat([rgb_grid[..., -1:], rgb_grid, rgb_grid[..., :1]], dim=-1)

    extrinsics4, intrinsics3 = _as_batched_camera_mats(
        extrinsics,
        intrinsics,
        batch_size=1,
        seq_len=int(seq_len),
        device=tokens.device,
        dtype=torch.float32,
    )
    extrinsic = extrinsics4[0]
    intrinsic = intrinsics3[0]
    yy, xx = torch.meshgrid(
        torch.arange(int(height), device=tokens.device, dtype=torch.float32),
        torch.arange(int(width), device=tokens.device, dtype=torch.float32),
        indexing="ij",
    )
    fx = intrinsic[:, 0, 0].clamp_min(1e-6).view(int(seq_len), 1, 1)
    fy = intrinsic[:, 1, 1].clamp_min(1e-6).view(int(seq_len), 1, 1)
    cx = intrinsic[:, 0, 2].view(int(seq_len), 1, 1)
    cy = intrinsic[:, 1, 2].view(int(seq_len), 1, 1)
    x_cam = (xx.view(1, int(height), int(width)) - cx) / fx
    y_cam = (yy.view(1, int(height), int(width)) - cy) / fy
    z_cam = torch.ones_like(x_cam)
    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1)
    rotation_world_from_cam = extrinsic[:, :3, :3].transpose(-1, -2)
    dirs_world = torch.einsum("sij,shwj->shwi", rotation_world_from_cam, dirs_cam)
    dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)

    azimuth = torch.atan2(dirs_world[..., 2], dirs_world[..., 0])
    u = torch.remainder((azimuth + float(np.pi)) / (2.0 * float(np.pi)), 1.0)
    upper = (-dirs_world[..., 1]).clamp(0.0, 1.0)
    elevation = torch.asin(upper)
    v = (1.0 - elevation / (0.5 * float(np.pi))).clamp(0.0, 1.0)

    # Atlas values live at bin centres. With align_corners=False, 2*u-1 maps
    # (j+0.5)/W exactly to pixel j. The azimuth axis has one wrapped padding
    # column on either side, hence the corresponding padded-grid coordinate.
    x_norm = (2.0 * (u * float(gw) + 1.0) / float(gw + 2)) - 1.0
    y_norm = (2.0 * v) - 1.0
    sample_grid = torch.stack([x_norm, y_norm], dim=-1)
    bg = torch.nn.functional.grid_sample(
        rgb_grid.expand(int(seq_len), -1, -1, -1),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return bg.permute(0, 2, 3, 1).contiguous()


def sky_tokens_to_background(
    sky_tokens: torch.Tensor | None,
    *,
    seq_len: int,
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    extrinsics: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if sky_tokens is None:
        return None
    if sky_tokens.ndim != 3 or int(sky_tokens.shape[-1]) < 3:
        raise ValueError(f"sky_tokens must be [B,K,C>=3], got {tuple(sky_tokens.shape)}")
    if (extrinsics is None) != (intrinsics is None):
        raise ValueError("extrinsics and intrinsics must be provided together for directional sky rendering.")
    if extrinsics is not None and intrinsics is not None:
        return render_sky_tokens_directional_background(
            sky_tokens,
            seq_len=seq_len,
            height=height,
            width=width,
            grid_h=grid_h,
            grid_w=grid_w,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
    tokens = sky_tokens[:1].float()
    gh, gw = _sky_grid_for_token_count(int(tokens.shape[1]), int(grid_h), int(grid_w))
    rgb = ((tokens[..., :3] + 1.0) * 0.5).clamp(0.0, 1.0)
    rgb = rgb.reshape(1, gh, gw, 3).permute(0, 3, 1, 2)
    rgb = torch.nn.functional.interpolate(
        rgb,
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return rgb.unsqueeze(0).expand(int(seq_len), -1, -1, -1).contiguous()


def _sky_background_image_grid(background: torch.Tensor | None, max_frames: int) -> torch.Tensor | None:
    if background is None:
        return None
    bg = background[:max_frames].detach().float().cpu().permute(0, 3, 1, 2).clamp(0.0, 1.0)
    return bg


def sky_token_validation_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    prefix: str = "sample_sky",
    loss_weight: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"sky token shapes must match, got {tuple(pred.shape)} and {tuple(gt.shape)}")
    if loss_weight is None and attention_mask is not None:
        loss_weight = attention_mask
    if loss_weight is None:
        weight = torch.ones(pred.shape[:2], device=pred.device, dtype=torch.float32)
    else:
        weight = loss_weight.to(device=pred.device, dtype=torch.float32).clamp_min(0.0)
        if weight.shape != pred.shape[:2]:
            raise ValueError(f"sky loss weight shape {tuple(weight.shape)} != {tuple(pred.shape[:2])}")
    if not bool(weight.gt(0.0).any().item()):
        return {
            f"{prefix}_rgb_mae": 0.0,
            f"{prefix}_weight_mean": 0.0,
        }
    diff = (pred[..., :3].float() - gt.to(device=pred.device, dtype=torch.float32)[..., :3]).abs()
    diff = diff * weight.unsqueeze(-1).to(dtype=diff.dtype)
    return {
        f"{prefix}_rgb_mae": float((diff.sum() / (weight.sum().clamp_min(1e-6) * 3.0)).detach().item()),
        f"{prefix}_weight_mean": float(weight.mean().detach().item()),
    }


def generated_sky_view_reconstruction_loss(
    *,
    vggt_model: VGGT,
    sky_latent: torch.Tensor,
    images: torch.Tensor,
    sky_mask: torch.Tensor,
    gt_pose_enc_dggt: torch.Tensor,
    lpips_model: nn.Module | None = None,
    lpips_weight: float = 0.01,
    high_frequency_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reproject generated sky with the frozen GT DGGT camera."""
    del vggt_model
    rgb_tokens = decode_sky_patch_tokens(sky_latent)
    atlas_hw = DEFAULT_SKY_ATLAS_HW
    mask = _sky_mask_1ch(sky_mask, images).float()
    total = sky_latent.sum() * 0.0
    charbonnier_total = 0.0
    high_total = 0.0
    lpips_total = 0.0
    for row in range(int(images.shape[0])):
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            gt_pose_enc_dggt[row : row + 1].float(),
            (int(images.shape[-2]), int(images.shape[-1])),
        )
        background = sky_tokens_to_background(
            rgb_tokens[row : row + 1],
            seq_len=int(images.shape[1]),
            height=int(images.shape[-2]),
            width=int(images.shape[-1]),
            grid_h=int(atlas_hw[0]),
            grid_w=int(atlas_hw[1]),
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        ).permute(0, 3, 1, 2)
        target = images[row].float()
        weight = mask[row]
        denom = (weight.sum() * 3.0).clamp_min(1.0)
        charb = (torch.sqrt((background - target).square() + 1.0e-6) * weight).sum() / denom
        dx = (background[..., :, 1:] - background[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
        dy = (background[..., 1:, :] - background[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
        wx = weight[..., :, 1:] * weight[..., :, :-1]
        wy = weight[..., 1:, :] * weight[..., :-1, :]
        high = (dx.abs() * wx).sum() / (wx.sum() * 3.0).clamp_min(1.0)
        high = high + (dy.abs() * wy).sum() / (wy.sum() * 3.0).clamp_min(1.0)
        row_loss = charb + float(high_frequency_weight) * high
        lpips_value = background.new_zeros(())
        if lpips_model is not None and float(lpips_weight) > 0.0:
            pred_masked = weight * background + (1.0 - weight) * 0.5
            target_masked = weight * target + (1.0 - weight) * 0.5
            lpips_value = lpips_model(pred_masked * 2.0 - 1.0, target_masked * 2.0 - 1.0).mean()
            row_loss = row_loss + float(lpips_weight) * lpips_value
        total = total + row_loss / float(images.shape[0])
        charbonnier_total += float(charb.detach().item()) / float(images.shape[0])
        high_total += float(high.detach().item()) / float(images.shape[0])
        lpips_total += float(lpips_value.detach().item()) / float(images.shape[0])
    return total, {
        "loss_sky_view_charbonnier": charbonnier_total,
        "loss_sky_view_high_frequency": high_total,
        "loss_sky_view_lpips": lpips_total,
    }


def _predict_camera_mats(
    pose_enc: torch.Tensor,
    image_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_hw
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (height, width))
    extrinsic_3x4 = extrinsics[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=extrinsic_3x4.dtype).view(1, 1, 4)
    extrinsic = torch.cat([extrinsic_3x4, bottom.expand(extrinsic_3x4.shape[0], -1, -1)], dim=1)
    intrinsic = intrinsics[0]
    return extrinsic, intrinsic


def _render_background(
    model: VGGT,
    images: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    mode: str = "sky",
) -> torch.Tensor:
    _, seq_len, _, height, width = images.shape
    if mode == "sky" and hasattr(model, "sky_model") and model.sky_model is not None:
        bg_render = model.sky_model(images, extrinsic, intrinsic).float()
        denom = (bg_render.max() - bg_render.min()).clamp_min(1e-8)
        bg_render = ((bg_render - bg_render.min()) / denom).clamp(0.0, 1.0)
        return bg_render
    return torch.zeros((seq_len, height, width, 3), dtype=images.dtype, device=images.device)


def split_image_tokens_for_heads(image_tokens_list: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    aggregated_tokens_list = []
    dino_token_list = []
    for tokens in image_tokens_list:
        if tokens.shape[-1] != 3072:
            raise ValueError(f"Expected 3072-wide image tokens, got {tokens.shape[-1]}")
        dino, frame, global_tokens = tokens.split([1024, 1024, 1024], dim=-1)
        dino_token_list.append(dino)
        aggregated_tokens_list.append(torch.cat([frame, global_tokens], dim=-1))
    return aggregated_tokens_list, dino_token_list


def alpha_t(t: torch.Tensor, t0: torch.Tensor | float, alpha: torch.Tensor, gamma0: torch.Tensor, gamma1: float = 0.1):
    if not torch.is_tensor(t0):
        t0 = torch.tensor(float(t0), dtype=t.dtype, device=t.device)
    sigma = torch.log(torch.tensor(gamma1, dtype=alpha.dtype, device=alpha.device)) / ((gamma0) ** 2 + 1e-6)
    conf = torch.exp(sigma * (t0 - t) ** 2)
    return (alpha * conf).float()


def _rasterize_scene(
    means: torch.Tensor,
    rgbs: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotation: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from gsplat.rendering import rasterization

    if means.numel() == 0:
        empty_render = torch.zeros((1, height, width, 4), dtype=torch.float32, device=viewmat.device)
        empty_alpha = torch.zeros((1, height, width, 1), dtype=torch.float32, device=viewmat.device)
        return empty_render, empty_alpha

    renders_chunk, alphas_chunk, _ = rasterization(
        means=means,
        quats=rotation,
        scales=scales,
        opacities=opacity,
        colors=rgbs,
        viewmats=viewmat,
        Ks=intrinsic,
        width=width,
        height=height,
        render_mode="RGB+ED",
    )
    return renders_chunk, alphas_chunk


def _render_gs_map_rgb(
    model: VGGT,
    images: torch.Tensor | None,
    masks: torch.Tensor | None,
    timestamps: torch.Tensor,
    pose_enc: torch.Tensor,
    depth: torch.Tensor,
    gs_map: torch.Tensor,
    gs_conf: torch.Tensor,
    dynamic_conf: torch.Tensor,
    device: torch.device,
    max_frames: int,
    background_mode: str = "sky",
    use_sky_mask: bool = True,
    background_override: torch.Tensor | None = None,
    image_hw: tuple[int, int] | None = None,
    soft_sky_mask: bool = False,
) -> torch.Tensor:
    """Render 3DGS with DGGT mode-2 scene assembly and correct compositing.

    Static branch: rasterized once with `bg_mask & (dy_map < 0.5)` (no extra
    valid_depth filter, no sigmoid on the threshold).
    Dynamic branch: per-frame, gated by `bg_mask` only.
    Background: either GT-image sky model (with min-max norm), or pure black
    when ``background_mode != "sky"``. Unlike the legacy ``inference.py``
    renderer, gsplat's premultiplied RGB is not multiplied by alpha twice.

    When `use_sky_mask=False` (used for the fully generated path the user
    requires), `bg_mask` is replaced with an all-True tensor — no GT sky
    information is consumed.
    """
    # The renderer only consumes sample 0 throughout; pin every batched tensor
    # to that slice up front so downstream indexing can't broadcast against B > 1.
    if images is not None:
        images = images[:1]
    pose_enc = pose_enc[:1]
    depth = depth[:1]
    gs_map = gs_map[:1]
    gs_conf = gs_conf[:1]
    dynamic_conf = dynamic_conf[:1]
    if masks is not None:
        masks = masks[:1]

    if images is not None:
        _, seq_len, _, height, width = images.shape
        image_dtype = images.dtype
    else:
        if image_hw is None:
            raise ValueError("_render_gs_map_rgb requires image_hw when images is None.")
        seq_len = int(pose_enc.shape[1])
        height, width = int(image_hw[0]), int(image_hw[1])
        image_dtype = depth.dtype
    # `frames` only controls how many output views are produced (displayed).
    # The static-GS accumulation MUST cover the full sequence to match
    # inference.py mode=2 — slicing the GS data to `[:frames]` was the source
    # of the dggt_clean_3dgs_rgb checkerboard artifacts when val_log_images
    # was much smaller than sequence_length.
    frames = min(int(max_frames), int(seq_len))
    depth = depth.float()
    pose_enc = pose_enc.float()
    extrinsic, intrinsic = _predict_camera_mats(pose_enc, (height, width), device)
    point_map = unproject_depth_map_to_point_map(
        depth[0].detach().cpu(),
        extrinsic[:, :3, :].detach().cpu(),
        intrinsic.detach().cpu(),
    )
    # [1, S, H, W, 3] to match inference.py indexing semantics.
    point_map = torch.from_numpy(point_map).to(device=device, dtype=torch.float32)[None, ...]

    if masks is not None and use_sky_mask:
        sky_mask = masks.to(device).permute(0, 1, 3, 4, 2)
        sky_probability = sky_mask.float().mean(dim=-1).clamp(0.0, 1.0)
        non_sky_probability = 1.0 - sky_probability
        threshold = 1.0e-4 if soft_sky_mask else 0.5
        bg_mask = non_sky_probability > threshold
    else:
        bg_mask = torch.ones((1, seq_len, height, width), dtype=torch.bool, device=device)
        non_sky_probability = torch.ones_like(bg_mask, dtype=torch.float32)

    if background_override is not None:
        bg_render = background_override.to(device=device, dtype=image_dtype)
        if bg_render.ndim == 5:
            bg_render = bg_render[:1, :seq_len, ...]
            if int(bg_render.shape[0]) != 1:
                raise ValueError(f"background_override batch must be 1, got {tuple(bg_render.shape)}")
            bg_render = bg_render[0]
        elif bg_render.ndim == 4 and int(bg_render.shape[-1]) == 3:
            bg_render = bg_render[:seq_len]
        elif bg_render.ndim == 4 and int(bg_render.shape[1]) == 3:
            bg_render = bg_render[:seq_len].permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(
                "background_override must be [S,H,W,3], [S,3,H,W], or [1,S,H,W,3], "
                f"got {tuple(bg_render.shape)}"
            )
        if bg_render.shape != (seq_len, height, width, 3):
            raise ValueError(
                f"background_override shape {tuple(bg_render.shape)} != {(seq_len, height, width, 3)}"
            )
    else:
        if images is None and background_mode == "sky":
            raise ValueError("Sky-model background rendering requires real images or background_override.")
        if images is None:
            bg_render = torch.zeros((seq_len, height, width, 3), dtype=image_dtype, device=device)
        else:
            bg_render = _render_background(
                model, images, extrinsic, intrinsic, mode=background_mode
            )
    timestamps = timestamps[:seq_len].to(device=device, dtype=torch.float32)
    # Raw logits — inference.py uses `dy_map < 0.5` (no sigmoid) for the threshold.
    dy_map = dynamic_conf.squeeze(-1).float()

    # === Static branch (matches inference.py mode=2 exactly, full S frames) ===
    static_mask = bg_mask & (dy_map < 0.5)
    static_points = point_map[static_mask].reshape(-1, 3)
    static_dynamic_prob = dy_map[static_mask].sigmoid()
    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
    static_opacity = static_opacity * (1.0 - static_dynamic_prob)
    if soft_sky_mask:
        static_opacity = static_opacity * non_sky_probability[static_mask]
    static_gs_conf = gs_conf[static_mask]
    static_frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
    gs_timestamps = timestamps[static_frame_idx] if static_frame_idx.numel() > 0 else timestamps.new_zeros((0,))

    # === Dynamic branch (per-frame, bg_mask only — only for displayed frames) ===
    dynamic_points, dynamic_rgbs, dynamic_opacitys, dynamic_scales, dynamic_rotations = [], [], [], [], []
    for frame_idx in range(frames):
        bg_mask_i = bg_mask[:, frame_idx]
        dynamic_point = point_map[:, frame_idx][bg_mask_i].reshape(-1, 3)
        dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, frame_idx], bg_mask_i)
        dynamic_prob = dy_map[:, frame_idx][bg_mask_i].sigmoid()
        dynamic_opacity = dynamic_opacity * dynamic_prob
        if soft_sky_mask:
            dynamic_opacity = dynamic_opacity * non_sky_probability[:, frame_idx][bg_mask_i]
        dynamic_points.append(dynamic_point)
        dynamic_rgbs.append(dynamic_rgb)
        dynamic_opacitys.append(dynamic_opacity)
        dynamic_scales.append(dynamic_scale)
        dynamic_rotations.append(dynamic_rotation)

    renders = []
    for frame_idx in range(frames):
        t0 = timestamps[frame_idx]
        static_opacity_t = alpha_t(gs_timestamps, t0, static_opacity, gamma0=static_gs_conf)
        static_gs_list = [static_points, static_rgbs, static_opacity_t, static_scales, static_rotations]
        world_points, rgbs, opacity, scales, rotation = concat_list(
            static_gs_list,
            [
                dynamic_points[frame_idx],
                dynamic_rgbs[frame_idx],
                dynamic_opacitys[frame_idx],
                dynamic_scales[frame_idx],
                dynamic_rotations[frame_idx],
            ],
        )
        rendered_raw, alpha = _rasterize_scene(
            means=world_points.float(),
            rgbs=rgbs.float().clamp(0.0, 1.0),
            opacity=opacity.float().view(-1),
            scales=scales.float().clamp_min(1e-5),
            rotation=rotation.float(),
            viewmat=extrinsic[frame_idx : frame_idx + 1],
            intrinsic=intrinsic[frame_idx : frame_idx + 1],
            height=height,
            width=width,
        )
        # gsplat returns premultiplied RGB when rasterized without a
        # background: sum_i(T_i * alpha_i * color_i).  Only the residual
        # transmittance belongs to the spatial sky/background image.
        premultiplied_rgb = rendered_raw[..., :3]
        composed = composite_gsplat_rgb(
            premultiplied_rgb,
            alpha,
            bg_render[frame_idx : frame_idx + 1],
        )
        renders.append(composed[0].permute(2, 0, 1).detach().cpu().float().clamp(0.0, 1.0))
    return torch.stack(renders, dim=0)


def _fixed_render_hw(args: argparse.Namespace) -> tuple[int, int]:
    """Return the fixed model-space render size implied by the token grid."""
    patch_grid = tuple(int(v) for v in args.patch_grid)
    patch_size = 14
    return patch_grid[0] * patch_size, patch_grid[1] * patch_size


def _timestamps_for_generated_render(
    batch: dict[str, Any] | None,
    *,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if batch is not None and "timestamps" in batch:
        raw = batch["timestamps"]
        timestamps = raw[0] if torch.is_tensor(raw) else torch.as_tensor(raw[0])
        return timestamps[:seq_len].to(device=device, dtype=torch.float32)
    if seq_len <= 1:
        return torch.zeros((seq_len,), device=device, dtype=torch.float32)
    return gaussian_timestamps_from_frame_ids(
        torch.arange(seq_len, device=device, dtype=torch.long)
    )


def _required_head_levels(vggt_model: VGGT) -> tuple[int, ...]:
    levels: set[int] = set(int(v) for v in TOKENIZER_LEVELS)
    for name in ("gs_head", "depth_head", "instance_head"):
        head = getattr(vggt_model, name, None)
        if head is not None and hasattr(head, "intermediate_layer_idx"):
            levels.update(int(v) for v in getattr(head, "intermediate_layer_idx"))
    missing = sorted(level for level in levels if level not in set(TOKENIZER_LEVELS))
    if missing:
        raise RuntimeError(
            "No-image SceneFlow pretrain rendering can only feed DGGT heads from "
            f"tokenizer-decoded levels {TOKENIZER_LEVELS}; required extra levels: {missing}."
        )
    return tuple(sorted(levels))


def _decode_generated_tokens_without_template(
    vggt_model: VGGT,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor | None], int]:
    """Decode SceneFlow latents into sparse DGGT token levels without GT image tokens."""
    sf = unwrap_ddp(scene_flow)
    z_generated = sf.denormalize(z_generated_raw_n.float())
    decoded_patch_tokens = vggt_model.scene_tokenizer.decode(z_generated, patch_grid=args.patch_grid)
    del z_generated
    if len(decoded_patch_tokens) != len(TOKENIZER_LEVELS):
        raise RuntimeError(
            f"scene_tokenizer.decode returned {len(decoded_patch_tokens)} levels, "
            f"expected {len(TOKENIZER_LEVELS)} for {TOKENIZER_LEVELS}."
        )

    patch_start_idx = int(getattr(getattr(vggt_model, "aggregator", None), "patch_start_idx", 5))
    required_levels = _required_head_levels(vggt_model)
    num_levels = max(required_levels) + 1
    full_levels: list[torch.Tensor | None] = [None] * num_levels
    expected_patches = int(args.patch_grid[0]) * int(args.patch_grid[1])
    for level, patch_tokens in zip(TOKENIZER_LEVELS, decoded_patch_tokens):
        patch_tokens = patch_tokens.to(device=device)
        if patch_tokens.ndim != 4:
            raise ValueError(f"decoded patch tokens must be [B,S,P,C], got {tuple(patch_tokens.shape)}")
        if int(patch_tokens.shape[2]) != expected_patches:
            raise ValueError(
                f"decoded level {level} patch count {patch_tokens.shape[2]} "
                f"!= patch_grid product {expected_patches}"
            )
        special = torch.zeros(
            patch_tokens.shape[:2] + (patch_start_idx, int(patch_tokens.shape[-1])),
            device=device,
            dtype=patch_tokens.dtype,
        )
        full_levels[int(level)] = torch.cat([special, patch_tokens], dim=-2)
    return full_levels, patch_start_idx


def _split_sparse_generated_tokens_for_heads(
    image_tokens_list: list[torch.Tensor | None],
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    aggregated_tokens_list: list[torch.Tensor | None] = [None] * len(image_tokens_list)
    dino_token_list: list[torch.Tensor | None] = [None] * len(image_tokens_list)
    for level in TOKENIZER_LEVELS:
        tokens = image_tokens_list[int(level)]
        if tokens is None:
            raise RuntimeError(f"generated token level {level} is missing")
        if int(tokens.shape[-1]) != 3072:
            raise ValueError(f"Expected 3072-wide generated tokens at level {level}, got {tokens.shape[-1]}")
        dino, frame, global_tokens = tokens.split([1024, 1024, 1024], dim=-1)
        dino_token_list[int(level)] = dino
        aggregated_tokens_list[int(level)] = torch.cat([frame, global_tokens], dim=-1)
    return aggregated_tokens_list, dino_token_list


@torch.no_grad()
def render_validation_rgb(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    generated_camera_features: torch.Tensor | None = None,
    generated_sky_tokens: torch.Tensor | None = None,
    generated_sky_mask_patch: torch.Tensor | None = None,
    generated_sky_mask_refined: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Render three validation RGB grids (clean / generated / recon).

    Memory optimization: each branch is processed sequentially — compute heads,
    render to CPU, then delete GPU tensors and empty the CUDA cache before the
    next branch.  This reduces peak GPU memory from ~3× a single branch to ~1×.
    """
    if generated_camera_features is None:
        raise RuntimeError("Pretrain generated RGB validation requires generated_camera_features from SceneFlow.")
    if generated_sky_mask_patch is None:
        raise RuntimeError("Pretrain generated RGB validation requires generated_sky_mask_patch from SceneFlow.")
    images = batch["images"].to(device, non_blocking=True)
    masks = batch.get("masks")
    if masks is not None:
        masks = masks.to(device, non_blocking=True)
    frames = min(int(args.val_log_images), int(images.shape[1]))
    sf = unwrap_ddp(scene_flow)
    timestamps = batch["timestamps"][0] if torch.is_tensor(batch["timestamps"]) else torch.as_tensor(batch["timestamps"][0])

    result: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Phase 0: Run the shared aggregator ONCE and build the three sets of
    # modified image tokens.  Keep image_tokens_list alive for heads;
    # free decode intermediates eagerly.
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        outputs = vggt_model.get_aggregator_token_outputs(images)
        aggregated_tokens_list = outputs["aggregated_tokens_list"]
        image_tokens_list = outputs["image_tokens_list"]
        dino_token_list = outputs["dino_token_list"]
        patch_start_idx = int(outputs["patch_start_idx"])
        del outputs

        # --- Generated branch tokens ---
        z_generated = sf.denormalize(z_generated_raw_n.float())
        decoded_patch_tokens = vggt_model.scene_tokenizer.decode(z_generated, patch_grid=args.patch_grid)
        del z_generated
        decoded_full_tokens = reattach_special_tokens(
            image_tokens_list, TOKENIZER_LEVELS, patch_start_idx, decoded_patch_tokens,
        )
        del decoded_patch_tokens
        generated_image_tokens = replace_selected_levels(
            image_tokens_list, TOKENIZER_LEVELS, decoded_full_tokens,
        )
        del decoded_full_tokens

        # --- Recon branch tokens ---
        tokens_4 = select_patch_pyramid(image_tokens_list, TOKENIZER_LEVELS, patch_start_idx)
        z_recon = vggt_model.scene_tokenizer.encode(tokens_4, patch_grid=args.patch_grid)
        del tokens_4
        recon_patch_tokens = vggt_model.scene_tokenizer.decode(z_recon, patch_grid=args.patch_grid)
        del z_recon
        recon_full_tokens = reattach_special_tokens(
            image_tokens_list, TOKENIZER_LEVELS, patch_start_idx, recon_patch_tokens,
        )
        del recon_patch_tokens
        recon_image_tokens = replace_selected_levels(
            image_tokens_list, TOKENIZER_LEVELS, recon_full_tokens,
        )
        del recon_full_tokens

    # ------------------------------------------------------------------
    # Phase 1: CLEAN branch (uses original aggregated/dino/image tokens)
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        # Heads MUST run with autocast disabled (matches VGGT.forward).
        with torch.amp.autocast(device_type=device.type, enabled=False):
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            depth, _ = vggt_model.depth_head(aggregated_tokens_list, images, patch_start_idx)
            dynamic_conf, _ = vggt_model.instance_head(dino_token_list, images, patch_start_idx)
            clean_gs_map, clean_gs_conf = vggt_model.gs_head(image_tokens_list, images, patch_start_idx)

    # Free original tokens — no longer needed after clean heads.
    del aggregated_tokens_list, dino_token_list, image_tokens_list

    result["dggt_clean_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model, images, masks, timestamps,
        pose_enc, depth, clean_gs_map, clean_gs_conf, dynamic_conf,
        device, frames, background_mode="sky", use_sky_mask=True,
    )
    del pose_enc, depth, dynamic_conf, clean_gs_map, clean_gs_conf
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: GENERATED branch
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        gen_agg, gen_dino = split_image_tokens_for_heads(generated_image_tokens)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            raw_gs_map, raw_gs_conf = vggt_model.gs_head(generated_image_tokens, images, patch_start_idx)
            generated_pose_enc = decode_pose_from_camera_features(vggt_model, generated_camera_features.to(device))
            generated_depth, _ = vggt_model.depth_head(gen_agg, images, patch_start_idx)
            generated_dynamic_conf, _ = vggt_model.instance_head(gen_dino, images, patch_start_idx)
            generated_sky_mask = _sky_mask_patch_to_image(
                generated_sky_mask_refined if generated_sky_mask_refined is not None else generated_sky_mask_patch,
                patch_grid=args.patch_grid,
                height=int(images.shape[-2]),
                width=int(images.shape[-1]),
                device=device,
            )

    del generated_image_tokens, gen_agg, gen_dino

    # Generated path consumes no GT sky mask and no sky_model background.
    # Its sky/non-sky split comes from the SceneFlow sky-mask head. If
    # generated sky tokens are available, render them as a camera-aware
    # directional atlas; otherwise keep a black fallback.
    result["generated_pred_sky_mask"] = _sky_mask_image_grid(generated_sky_mask, frames)
    generated_sky_background = None
    if generated_sky_tokens is not None:
        sky_h, sky_w = sky_atlas_shape(args)
        generated_sky_extrinsic, generated_sky_intrinsic = _predict_camera_mats(
            generated_pose_enc,
            (int(images.shape[-2]), int(images.shape[-1])),
            device,
        )
        generated_sky_background = sky_tokens_to_background(
            decode_sky_patch_tokens(generated_sky_tokens.to(device)),
            seq_len=int(images.shape[1]),
            height=int(images.shape[-2]),
            width=int(images.shape[-1]),
            grid_h=sky_h,
            grid_w=sky_w,
            extrinsics=generated_sky_extrinsic,
            intrinsics=generated_sky_intrinsic,
        )
        sky_grid_image = _sky_background_image_grid(generated_sky_background, frames)
        if sky_grid_image is not None:
            result["generated_sky_rgb"] = sky_grid_image
    result["generated_raw_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model, images, generated_sky_mask, timestamps,
        generated_pose_enc, generated_depth, raw_gs_map, raw_gs_conf,
        generated_dynamic_conf,
        device, frames, background_mode="black", use_sky_mask=True,
        background_override=generated_sky_background,
        soft_sky_mask=True,
    )
    del (
        generated_pose_enc,
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        generated_sky_mask,
    )
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 3: RECON branch (tokenizer round-trip)
    # ------------------------------------------------------------------
    with autocast_context(args, device):
        recon_agg, recon_dino = split_image_tokens_for_heads(recon_image_tokens)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            recon_pose_enc = vggt_model.camera_head(recon_agg)[-1]
            recon_depth, _ = vggt_model.depth_head(recon_agg, images, patch_start_idx)
            recon_dynamic_conf, _ = vggt_model.instance_head(recon_dino, images, patch_start_idx)
            recon_gs_map, recon_gs_conf = vggt_model.gs_head(recon_image_tokens, images, patch_start_idx)

    del recon_image_tokens, recon_agg, recon_dino

    result["tokenizer_recon_3dgs_rgb"] = _render_gs_map_rgb(
        vggt_model, images, masks, timestamps,
        recon_pose_enc, recon_depth, recon_gs_map, recon_gs_conf,
        recon_dynamic_conf,
        device, frames, background_mode="sky", use_sky_mask=True,
    )
    del recon_pose_enc, recon_depth, recon_dynamic_conf, recon_gs_map, recon_gs_conf
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # GT images grid (always cheap)
    # ------------------------------------------------------------------
    result["input_rgb_gt"] = _image_grid(images, frames)
    return result


@torch.no_grad()
def render_validation_generated_rgb(
    batch: dict[str, Any] | None,
    vggt_model: VGGT,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    generated_camera_features: torch.Tensor | None = None,
    generated_sky_tokens: torch.Tensor | None = None,
    generated_sky_mask_patch: torch.Tensor | None = None,
    generated_sky_mask_refined: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Render generated SceneFlow latents using the generated DGGT-space camera tokens."""
    if generated_camera_features is None:
        raise RuntimeError("Pretrain generated RGB validation requires generated_camera_features from SceneFlow.")
    if generated_sky_mask_patch is None:
        raise RuntimeError("Pretrain generated RGB validation requires generated_sky_mask_patch from SceneFlow.")
    batch_size, seq_len = int(z_generated_raw_n.shape[0]), int(z_generated_raw_n.shape[1])
    height, width = _fixed_render_hw(args)
    frames = min(int(args.val_log_images), seq_len)
    timestamps = _timestamps_for_generated_render(batch, seq_len=seq_len, device=device)

    patch_start_idx = int(getattr(vggt_model.aggregator, "patch_start_idx", 5))
    with autocast_context(args, device):
        geometry = decode_generated_dggt_geometry(
            vggt_model=vggt_model,
            scene_flow_root=unwrap_ddp(scene_flow),
            z_clean_pred_n=z_generated_raw_n,
            patch_grid=args.patch_grid,
            patch_start_idx=patch_start_idx,
            image_hw=(height, width),
        )
        generated_pose_enc = decode_pose_from_camera_features(
            vggt_model, generated_camera_features.to(device)
        )
        generated_sky_mask = _sky_mask_patch_to_image(
            generated_sky_mask_refined if generated_sky_mask_refined is not None else generated_sky_mask_patch,
            patch_grid=args.patch_grid,
            height=height,
            width=width,
            device=device,
        )
    raw_gs_map, raw_gs_conf = geometry.gs_map, geometry.gs_conf
    generated_depth = geometry.depth
    generated_dynamic_conf = geometry.dynamic_conf

    generated_sky_background = None
    sky_grid_image = None
    if generated_sky_tokens is not None:
        sky_h, sky_w = sky_atlas_shape(args)
        generated_sky_extrinsic, generated_sky_intrinsic = _predict_camera_mats(
            generated_pose_enc,
            (height, width),
            device,
        )
        generated_sky_background = sky_tokens_to_background(
            decode_sky_patch_tokens(generated_sky_tokens.to(device)),
            seq_len=seq_len,
            height=height,
            width=width,
            grid_h=sky_h,
            grid_w=sky_w,
            extrinsics=generated_sky_extrinsic,
            intrinsics=generated_sky_intrinsic,
        )
        sky_grid_image = _sky_background_image_grid(generated_sky_background, frames)
    result = {
        "generated_pred_sky_mask": _sky_mask_image_grid(generated_sky_mask, frames),
        "generated_raw_3dgs_rgb": _render_gs_map_rgb(
            vggt_model, None, generated_sky_mask, timestamps,
            generated_pose_enc, generated_depth, raw_gs_map, raw_gs_conf,
            generated_dynamic_conf,
            device, frames, background_mode="black", use_sky_mask=True,
            background_override=generated_sky_background,
            image_hw=(height, width),
            soft_sky_mask=True,
        ),
    }
    if sky_grid_image is not None:
        result["generated_sky_rgb"] = sky_grid_image
    del (
        generated_pose_enc,
        generated_depth,
        raw_gs_map,
        raw_gs_conf,
        generated_dynamic_conf,
        generated_sky_mask,
    )
    torch.cuda.empty_cache()
    return result


def _bundle_frame_ids(
    bundle: Any,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    frame_ids = getattr(bundle, "frame_ids", None)
    if frame_ids is None:
        return torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
    if torch.is_tensor(frame_ids):
        frame_ids_t = frame_ids.to(device=device, dtype=torch.long)
    else:
        frame_ids_t = torch.as_tensor(frame_ids, device=device, dtype=torch.long)
    if frame_ids_t.ndim == 1:
        frame_ids_t = frame_ids_t.view(1, -1).expand(batch_size, -1)
    if frame_ids_t.shape != (batch_size, seq_len):
        raise ValueError(f"bundle.frame_ids must be [S] or [B,S], got {tuple(frame_ids_t.shape)}")
    return frame_ids_t.contiguous()


def _slice_time(tensor: torch.Tensor | None, start: int, end: int, seq_len: int) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if tensor.ndim >= 2 and int(tensor.shape[1]) == int(seq_len):
        return tensor[:, start:end]
    return tensor


def _slice_asset_time(tensor: torch.Tensor | None, start: int, end: int, seq_len: int) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if tensor.ndim == 5 and int(tensor.shape[2]) == int(seq_len):
        return tensor[:, :, start:end]
    if tensor.ndim == 4 and int(tensor.shape[2]) == int(seq_len):
        return tensor[:, :, start:end]
    if tensor.ndim == 4 and int(tensor.shape[1]) == int(seq_len):
        return tensor[:, start:end]
    return tensor


def _validation_sliding_params(args: argparse.Namespace, seq_len: int) -> tuple[int, int] | None:
    window = int(getattr(args, "val_sliding_window", 0) or 0)
    if window <= 0 or int(seq_len) <= window:
        return None
    stride = int(getattr(args, "val_sliding_stride", 0) or 0)
    if stride <= 0:
        stride = default_window_stride(window)
    return min(window, int(seq_len)), max(1, stride)


@torch.no_grad()
def _cfg_sample_pretrain_latents_sliding(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
    *,
    window: int,
    stride: int,
    guidance_scale: float | None = None,
    text_encoder: nn.Module | None = None,
    return_camera: bool = False,
    return_sky: bool = False,
    return_sky_mask: bool = False,
) -> torch.Tensor | SimpleNamespace:
    scale = float(args.guidance_scale) if guidance_scale is None else float(guidance_scale)
    t_steps = rae_t_grid(
        num_steps=args.val_sample_steps,
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))

    z_splat = getattr(bundle, "z_splat_n", None)
    if z_splat is None:
        z_splat = torch.zeros_like(bundle.z_clean_n)
    scaffold_tok = torch.zeros_like(bundle.z_clean_n)
    M_edit = (bundle.M_source.float() + bundle.M_dest.float()).clamp(0.0, 1.0).to(
        device=bundle.z_clean_n.device,
        dtype=bundle.z_clean_n.dtype,
    )
    M_keep = 1.0 - M_edit
    z = torch.empty_like(bundle.z_clean_n)
    z.normal_(generator=generator)
    z = M_edit * z + M_keep * z_splat
    camera_z = _init_pretrain_camera_noise(
        scene_flow,
        bundle,
        generator,
        return_camera=return_camera,
    )
    sky_z = None
    if return_sky:
        sky_h, sky_w = sky_grid_shape(args)
        sky_z = bundle.z_clean_n.new_empty((int(bundle.z_clean_n.shape[0]), int(sky_h * sky_w), SKY_TOKEN_DIM))
        sky_z.normal_(generator=generator)

    batch_size = int(z.shape[0])
    seq_len = int(z.shape[1])
    sf = unwrap_ddp(scene_flow)
    frame_ids_full = _bundle_frame_ids(bundle, batch_size=batch_size, seq_len=seq_len, device=device)
    windows = window_slices(seq_len, window, stride)
    coverage = cosine_coverage(seq_len, windows, device=device, dtype=z.dtype)

    kv_dim = bundle.F_asset_tokens.shape[-1]
    if bundle.F_asset_tokens.ndim in (4, 5):
        F_uncond = torch.zeros_like(bundle.F_asset_tokens)
        uncond_asset_mask = torch.zeros(
            bundle.F_asset_tokens.shape[:-1],
            device=bundle.F_asset_tokens.device,
            dtype=torch.bool,
        )
    else:
        F_uncond = bundle.F_asset_tokens.new_zeros((batch_size, 0, kv_dim))
        uncond_asset_mask = None
    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(text_encoder, [""] * batch_size if text_tokens is not None else None)
    asset_control_scale = float(getattr(args, "asset_control_guidance_scale", 1.0))
    camera_scale = float(getattr(args, "camera_guidance_scale", 1.0))
    camera_text_scale = float(getattr(args, "camera_text_guidance_scale", 1.0))
    optional_cfg = resolve_pretrain_optional_cfg_conditions(
        bundle,
        batch_size,
        asset_control_scale=asset_control_scale,
        camera_scale=camera_scale,
    )
    asset_control_scale = optional_cfg.asset_control_scale
    camera_scale = optional_cfg.camera_scale
    do_cfg = (
        abs(scale - 1.0) > 1e-6
        or abs(camera_text_scale - 1.0) > 1e-6
        or abs(asset_control_scale - 1.0) > 1e-6
        or abs(camera_scale - 1.0) > 1e-6
    )
    drop_all_control = torch.ones((batch_size,), device=device, dtype=torch.bool)
    asset_null_kind = optional_cfg.asset_null_kind
    camera_null_kind = optional_cfg.camera_null_kind
    full_asset_kind = optional_cfg.full_asset_kind
    full_camera_kind = optional_cfg.full_camera_kind
    full_camera_tokens = optional_cfg.full_camera_tokens
    full_camera_mask = optional_cfg.full_camera_mask
    final_sky_mask_logits = None
    final_sky_mask_refined_logits = None

    for i in range(int(args.val_sample_steps)):
        # The clean-state mask is evaluated once at sigma=0 after ODE
        # integration. Reading logits here would condition them on z_t>0.
        read_final_sky_mask = False
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)
        v_acc = torch.zeros_like(z)
        v_weight = torch.zeros((1, seq_len, 1, 1), device=device, dtype=z.dtype)
        camera_acc = torch.zeros_like(camera_z) if camera_z is not None else None
        camera_weight = (
            torch.zeros((1, seq_len, 1), device=device, dtype=camera_z.dtype)
            if camera_z is not None
            else None
        )
        sky_acc = torch.zeros_like(sky_z) if sky_z is not None else None
        sky_weight = 0.0
        mask_logit_acc = torch.zeros(z.shape[:3] + (1,), device=device, dtype=z.dtype) if read_final_sky_mask else None
        mask_logit_weight = (
            torch.zeros((1, seq_len, 1, 1), device=device, dtype=z.dtype)
            if read_final_sky_mask
            else None
        )
        mask_refined_acc = None
        mask_refined_weight = None

        for start, end in windows:
            actual = int(end - start)
            w_video = cosine_window(actual, device=device, dtype=z.dtype).view(1, actual, 1, 1)
            w_camera = (
                cosine_window(actual, device=device, dtype=camera_z.dtype).view(1, actual, 1)
                if camera_z is not None
                else None
            )
            z_w = z[:, start:end]
            z_splat_w = z_splat[:, start:end]
            scaffold_w = scaffold_tok[:, start:end]
            M_preserve_w = bundle.M_preserve[:, start:end]
            M_source_w = bundle.M_source[:, start:end]
            M_dest_w = bundle.M_dest[:, start:end]
            frame_ids_w = frame_ids_full[:, start:end]
            camera_tokens_w = _slice_time(full_camera_tokens, start, end, seq_len)
            camera_mask_w = _slice_time(full_camera_mask, start, end, seq_len)
            camera_z_w = camera_z[:, start:end] if camera_z is not None else None
            camera_anchor_w = _slice_time(
                getattr(bundle, "camera_gen_anchor_mask", None), start, end, seq_len
            )

            def _run_branch(
                *,
                F_asset_tokens: torch.Tensor,
                asset_mask: torch.Tensor | None,
                asset_kind: Any,
                branch_text_tokens: torch.Tensor | None,
                branch_text_mask: torch.Tensor | None,
                camera_kind: Any,
                control_drop_mask: torch.Tensor | None = None,
            ) -> dict[str, torch.Tensor]:
                out = sf(
                    z_w,
                    sigma,
                    z_splat_w,
                    scaffold_w,
                    M_preserve_w,
                    M_source_w,
                    M_dest_w,
                    _slice_asset_time(F_asset_tokens, start, end, seq_len),
                    encoder_attention_mask=_slice_asset_time(asset_mask, start, end, seq_len),
                    text_tokens=branch_text_tokens,
                    text_attention_mask=branch_text_mask,
                    camera_condition_tokens=camera_tokens_w,
                    camera_attention_mask=camera_mask_w,
                    camera_condition_kind=camera_kind,
                    camera_gen_tokens=camera_z_w,
                    camera_gen_anchor_mask=camera_anchor_w,
                    sky_gen_tokens=sky_z,
                    sky_gen_attention_mask=None,
                    return_mid=False,
                    return_dict=True,
                    return_sky_mask=read_final_sky_mask,
                    asset_condition_kind=asset_kind,
                    control_drop_mask=control_drop_mask,
                    frame_ids=frame_ids_w,
                    fps=None,
                )
                if not isinstance(out, dict):
                    raise RuntimeError("SceneFlow return_dict=True must return dicts in pretrain sliding sampling.")
                return out

            out_full = _run_branch(
                F_asset_tokens=bundle.F_asset_tokens,
                asset_mask=bundle.encoder_attention_mask,
                asset_kind=full_asset_kind,
                branch_text_tokens=text_tokens,
                branch_text_mask=text_mask,
                camera_kind=full_camera_kind,
            )
            v_full = out_full["video"]
            v_camera_full = out_full.get("camera")
            v_sky_full = out_full.get("sky")
            if do_cfg:
                out_no_text_full = None
                out_text = None
                out_text_asset = None
                if abs(scale - 1.0) > 1e-6 or abs(camera_text_scale - 1.0) > 1e-6:
                    # Cosmos-style text CFG: keep all clean structural
                    # conditions identical and remove text only.
                    out_no_text_full = _run_branch(
                        F_asset_tokens=bundle.F_asset_tokens,
                        asset_mask=bundle.encoder_attention_mask,
                        asset_kind=full_asset_kind,
                        branch_text_tokens=text_null,
                        branch_text_mask=text_null_mask,
                        camera_kind=full_camera_kind,
                    )
                if abs(asset_control_scale - 1.0) > 1e-6:
                    out_text = _run_branch(
                        F_asset_tokens=F_uncond,
                        asset_mask=uncond_asset_mask,
                        asset_kind=asset_null_kind,
                        branch_text_tokens=text_tokens,
                        branch_text_mask=text_mask,
                        camera_kind=camera_null_kind,
                        control_drop_mask=drop_all_control,
                    )
                    out_text_asset = _run_branch(
                        F_asset_tokens=bundle.F_asset_tokens,
                        asset_mask=bundle.encoder_attention_mask,
                        asset_kind=full_asset_kind,
                        branch_text_tokens=text_tokens,
                        branch_text_mask=text_mask,
                        camera_kind=camera_null_kind,
                    )
                elif abs(camera_scale - 1.0) > 1e-6:
                    out_text_asset = _run_branch(
                        F_asset_tokens=bundle.F_asset_tokens,
                        asset_mask=bundle.encoder_attention_mask,
                        asset_kind=full_asset_kind,
                        branch_text_tokens=text_tokens,
                        branch_text_mask=text_mask,
                        camera_kind=camera_null_kind,
                    )

                def _combine_cfg(key: str) -> torch.Tensor | None:
                    full = out_full.get(key)
                    if full is None:
                        return None
                    guided = full
                    text_guidance = camera_text_scale if key == "camera" else scale
                    if out_no_text_full is not None and abs(text_guidance - 1.0) > 1e-6:
                        no_text_full = out_no_text_full.get(key)
                        if no_text_full is None:
                            raise RuntimeError(f"Text-CFG branch is missing `{key}` predictions.")
                        guided = guided + (text_guidance - 1.0) * (full - no_text_full)
                    if abs(asset_control_scale - 1.0) > 1e-6:
                        assert out_text is not None and out_text_asset is not None
                        text = out_text.get(key)
                        text_asset = out_text_asset.get(key)
                        if text is None or text_asset is None:
                            raise RuntimeError(f"Asset-CFG branch is missing `{key}` predictions.")
                        guided = guided + (asset_control_scale - 1.0) * (text_asset - text)
                    if abs(camera_scale - 1.0) > 1e-6:
                        assert out_text_asset is not None
                        text_asset = out_text_asset.get(key)
                        if text_asset is None:
                            raise RuntimeError(f"Camera-CFG branch is missing `{key}` predictions.")
                        guided = guided + (camera_scale - 1.0) * (full - text_asset)
                    return guided

                v = _combine_cfg("video")
                if v is None:
                    raise RuntimeError("CFG branch is missing `video` predictions.")
                v_camera = _combine_cfg("camera") if camera_z is not None and v_camera_full is not None else None
                v_sky = _combine_cfg("sky") if sky_z is not None and v_sky_full is not None else None
                mask_logits_w = _combine_cfg("sky_mask_logits") if read_final_sky_mask else None
                mask_refined_w = _combine_cfg("sky_mask_refined_logits") if read_final_sky_mask else None
            else:
                v = v_full
                v_camera = v_camera_full
                v_sky = v_sky_full
                mask_logits_w = out_full.get("sky_mask_logits") if read_final_sky_mask else None
                mask_refined_w = out_full.get("sky_mask_refined_logits") if read_final_sky_mask else None

            v = sampler_prediction_to_velocity(sf, v, z_w, sigma)
            v_acc[:, start:end] += v * w_video
            v_weight[:, start:end] += w_video
            if camera_acc is not None and v_camera is not None and w_camera is not None and camera_z_w is not None:
                v_camera = sampler_prediction_to_velocity(sf, v_camera, camera_z_w, sigma)
                camera_acc[:, start:end] += v_camera * w_camera
                camera_weight[:, start:end] += w_camera
            if sky_acc is not None and v_sky is not None and sky_z is not None:
                v_sky = sampler_prediction_to_velocity(
                    sf,
                    v_sky,
                    sky_z,
                    sigma,
                    t_eps=SKY_FLOW_T_EPS,
                )
                global_weight = scene_global_window_weight(start, end, coverage).to(dtype=v_sky.dtype)
                sky_acc += v_sky * global_weight
                sky_weight += float(global_weight.item())
            if read_final_sky_mask:
                if mask_logits_w is None or mask_refined_w is None:
                    raise RuntimeError("Final denoising branch did not return sky-mask logits.")
                assert mask_logit_acc is not None and mask_logit_weight is not None
                mask_logit_acc[:, start:end] += mask_logits_w.to(dtype=mask_logit_acc.dtype) * w_video
                mask_logit_weight[:, start:end] += w_video
                if mask_refined_acc is None:
                    mask_refined_acc = torch.zeros(
                        (batch_size, seq_len) + tuple(mask_refined_w.shape[2:]),
                        device=device,
                        dtype=mask_refined_w.dtype,
                    )
                    mask_refined_weight = torch.zeros(
                        (1, seq_len, 1, 1, 1),
                        device=device,
                        dtype=mask_refined_w.dtype,
                    )
                refined_weight_w = w_video.view(1, actual, 1, 1, 1).to(dtype=mask_refined_w.dtype)
                mask_refined_acc[:, start:end] += mask_refined_w * refined_weight_w
                assert mask_refined_weight is not None
                mask_refined_weight[:, start:end] += refined_weight_w

        v = v_acc / v_weight.clamp_min(1e-6)
        z = z - step_h.to(dtype=z.dtype) * v
        z = M_keep * z_splat + M_edit * z
        if camera_z is not None and camera_acc is not None and camera_weight is not None:
            v_camera = camera_acc / camera_weight.clamp_min(1e-6)
            camera_z = camera_z - step_h.to(dtype=camera_z.dtype) * v_camera
        if sky_z is not None and sky_acc is not None and sky_weight > 0.0:
            v_sky = sky_acc / float(sky_weight)
            sky_z = sky_z - step_h.to(dtype=sky_z.dtype) * v_sky
        if read_final_sky_mask:
            if (
                mask_logit_acc is None
                or mask_logit_weight is None
                or mask_refined_acc is None
                or mask_refined_weight is None
            ):
                raise RuntimeError("Sliding sampling did not accumulate final sky-mask logits.")
            final_sky_mask_logits = mask_logit_acc / mask_logit_weight.clamp_min(1e-6)
            final_sky_mask_refined_logits = mask_refined_acc / mask_refined_weight.clamp_min(1e-6)

    z = M_keep * z_splat + M_edit * z
    if return_sky_mask:
        sigma = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        patch_acc = torch.zeros(z.shape[:3] + (1,), device=device, dtype=z.dtype)
        patch_weight = torch.zeros((1, seq_len, 1, 1), device=device, dtype=z.dtype)
        refined_acc = None
        refined_weight = None
        for start, end in windows:
            actual = end - start
            weight_w = cosine_window(actual, device=device, dtype=z.dtype).view(1, actual, 1, 1)
            camera_tokens_w = _slice_time(full_camera_tokens, start, end, seq_len)
            camera_mask_w = _slice_time(full_camera_mask, start, end, seq_len)

            def endpoint_branch(
                asset_tokens: torch.Tensor,
                asset_mask: torch.Tensor | None,
                asset_kind: Any,
                branch_text: torch.Tensor | None,
                branch_text_mask: torch.Tensor | None,
                camera_kind: Any,
                control_drop: torch.Tensor | None = None,
            ) -> dict[str, torch.Tensor]:
                result = sf(
                    z[:, start:end], sigma, z_splat[:, start:end], scaffold_tok[:, start:end],
                    bundle.M_preserve[:, start:end], bundle.M_source[:, start:end], bundle.M_dest[:, start:end],
                    _slice_asset_time(asset_tokens, start, end, seq_len),
                    encoder_attention_mask=_slice_asset_time(asset_mask, start, end, seq_len),
                    text_tokens=branch_text, text_attention_mask=branch_text_mask,
                    camera_condition_tokens=camera_tokens_w, camera_attention_mask=camera_mask_w,
                    camera_condition_kind=camera_kind,
                    camera_gen_tokens=None if camera_z is None else camera_z[:, start:end],
                    camera_gen_anchor_mask=_slice_time(getattr(bundle, "camera_gen_anchor_mask", None), start, end, seq_len),
                    sky_gen_tokens=sky_z, return_dict=True, return_sky_mask=True,
                    asset_condition_kind=asset_kind, control_drop_mask=control_drop,
                    frame_ids=frame_ids_full[:, start:end], fps=None,
                )
                if not isinstance(result, dict):
                    raise RuntimeError("SceneFlow sliding endpoint mask forward must return a dict")
                return result

            full = endpoint_branch(bundle.F_asset_tokens, bundle.encoder_attention_mask, full_asset_kind,
                                   text_tokens, text_mask, full_camera_kind)
            no_text = text_only = text_asset = None
            if do_cfg and (abs(scale - 1.0) > 1e-6 or abs(camera_text_scale - 1.0) > 1e-6):
                no_text = endpoint_branch(bundle.F_asset_tokens, bundle.encoder_attention_mask,
                                          full_asset_kind, text_null, text_null_mask, full_camera_kind)
            if do_cfg and abs(asset_control_scale - 1.0) > 1e-6:
                text_only = endpoint_branch(F_uncond, uncond_asset_mask, asset_null_kind,
                                            text_tokens, text_mask, camera_null_kind, drop_all_control)
                text_asset = endpoint_branch(bundle.F_asset_tokens, bundle.encoder_attention_mask,
                                             full_asset_kind, text_tokens, text_mask, camera_null_kind)
            elif do_cfg and abs(camera_scale - 1.0) > 1e-6:
                text_asset = endpoint_branch(bundle.F_asset_tokens, bundle.encoder_attention_mask,
                                             full_asset_kind, text_tokens, text_mask, camera_null_kind)

            def combine(key: str) -> torch.Tensor:
                value = full[key]
                if no_text is not None and abs(scale - 1.0) > 1e-6:
                    value = value + (scale - 1.0) * (full[key] - no_text[key])
                if abs(asset_control_scale - 1.0) > 1e-6:
                    assert text_only is not None and text_asset is not None
                    value = value + (asset_control_scale - 1.0) * (text_asset[key] - text_only[key])
                if abs(camera_scale - 1.0) > 1e-6:
                    assert text_asset is not None
                    value = value + (camera_scale - 1.0) * (full[key] - text_asset[key])
                return value

            patch_w = combine("sky_mask_logits")
            refined_w = combine("sky_mask_refined_logits")
            patch_acc[:, start:end] += patch_w.to(patch_acc.dtype) * weight_w
            patch_weight[:, start:end] += weight_w
            if refined_acc is None:
                refined_acc = torch.zeros((batch_size, seq_len) + tuple(refined_w.shape[2:]), device=device, dtype=refined_w.dtype)
                refined_weight = torch.zeros((1, seq_len, 1, 1, 1), device=device, dtype=refined_w.dtype)
            refined_weight_w = weight_w.view(1, actual, 1, 1, 1).to(refined_w.dtype)
            refined_acc[:, start:end] += refined_w * refined_weight_w
            refined_weight[:, start:end] += refined_weight_w
        final_sky_mask_logits = patch_acc / patch_weight.clamp_min(1.0e-6)
        assert refined_acc is not None and refined_weight is not None
        final_sky_mask_refined_logits = refined_acc / refined_weight.clamp_min(1.0e-6)
    sky_mask_logits = final_sky_mask_logits
    sky_mask_refined_logits = final_sky_mask_refined_logits
    sky_mask_patch = None if sky_mask_logits is None else torch.sigmoid(sky_mask_logits.float()).to(sky_mask_logits.dtype)
    sky_mask_refined = (
        None
        if sky_mask_refined_logits is None
        else torch.sigmoid(sky_mask_refined_logits.float()).to(sky_mask_refined_logits.dtype)
    )
    if return_sky_mask and (sky_mask_patch is None or sky_mask_refined is None):
        raise RuntimeError("Sampling requested a sky mask but the final denoising step did not produce one.")
    if return_camera or return_sky or return_sky_mask:
        camera_output = camera_z
        if camera_output is not None and str(getattr(sf.config, "camera_generation_representation", "dggt_hidden_v1")) == CAMERA_GENERATION_REPRESENTATION:
            anchor_mask = getattr(bundle, "camera_gen_anchor_mask", None)
            if anchor_mask is None:
                raise RuntimeError("DGGT v3 camera sampling is missing the global camera anchor mask")
            camera_output = sf.denormalize_camera(camera_output, anchor_mask)
        return SimpleNamespace(
            video=z,
            camera_state_dggt=camera_output,
            camera_anchor_mask=getattr(bundle, "camera_gen_anchor_mask", None),
            sky=sky_z,
            sky_mask_logits=sky_mask_logits,
            sky_mask_patch=sky_mask_patch,
            sky_mask_refined_logits=sky_mask_refined_logits,
            sky_mask_refined=sky_mask_refined,
        )
    return z


@torch.no_grad()
def cfg_sample_pretrain_latents(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
    guidance_scale: float | None = None,
    text_encoder: nn.Module | None = None,
    return_camera: bool = False,
    return_sky: bool = False,
    return_sky_mask: bool = False,
) -> torch.Tensor | SimpleNamespace:
    """Classifier-free guidance sampling from pure noise."""
    sliding = _validation_sliding_params(args, int(bundle.z_clean_n.shape[1]))
    if sliding is not None:
        return _cfg_sample_pretrain_latents_sliding(
            scene_flow,
            bundle,
            args,
            step,
            device,
            window=sliding[0],
            stride=sliding[1],
            guidance_scale=guidance_scale,
            text_encoder=text_encoder,
            return_camera=return_camera,
            return_sky=return_sky,
            return_sky_mask=return_sky_mask,
        )
    scale = float(args.guidance_scale) if guidance_scale is None else float(guidance_scale)
    t_steps = rae_t_grid(
        num_steps=args.val_sample_steps,
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(step))

    z_splat = getattr(bundle, "z_splat_n", None)
    if z_splat is None:
        z_splat = torch.zeros_like(bundle.z_clean_n)
    scaffold_tok = torch.zeros_like(bundle.z_clean_n)
    M_edit = (bundle.M_source.float() + bundle.M_dest.float()).clamp(0.0, 1.0).to(
        device=bundle.z_clean_n.device,
        dtype=bundle.z_clean_n.dtype,
    )
    M_keep = 1.0 - M_edit
    z = torch.empty_like(bundle.z_clean_n)
    z.normal_(generator=generator)
    z = M_edit * z + M_keep * z_splat
    camera_z = _init_pretrain_camera_noise(
        scene_flow,
        bundle,
        generator,
        return_camera=return_camera,
    )
    sky_z = None
    if return_sky:
        sky_h, sky_w = sky_grid_shape(args)
        sky_z = bundle.z_clean_n.new_empty((int(bundle.z_clean_n.shape[0]), int(sky_h * sky_w), SKY_TOKEN_DIM))
        sky_z.normal_(generator=generator)
    batch_size = z.shape[0]
    frame_ids = _bundle_frame_ids(bundle, batch_size=int(batch_size), seq_len=int(z.shape[1]), device=device)
    sf = unwrap_ddp(scene_flow)

    kv_dim = bundle.F_asset_tokens.shape[-1]
    if bundle.F_asset_tokens.ndim in (4, 5):
        F_uncond = torch.zeros_like(bundle.F_asset_tokens)
        uncond_asset_mask = torch.zeros(
            bundle.F_asset_tokens.shape[:-1],
            device=bundle.F_asset_tokens.device,
            dtype=torch.bool,
        )
    else:
        F_uncond = bundle.F_asset_tokens.new_zeros((batch_size, 0, kv_dim))
        uncond_asset_mask = None
    text_tokens, text_mask = encode_text_condition(text_encoder, getattr(bundle, "captions", None))
    text_null, text_null_mask = encode_text_condition(text_encoder, [""] * batch_size if text_tokens is not None else None)
    asset_control_scale = float(getattr(args, "asset_control_guidance_scale", 1.0))
    camera_scale = float(getattr(args, "camera_guidance_scale", 1.0))
    camera_text_scale = float(getattr(args, "camera_text_guidance_scale", 1.0))
    optional_cfg = resolve_pretrain_optional_cfg_conditions(
        bundle,
        int(batch_size),
        asset_control_scale=asset_control_scale,
        camera_scale=camera_scale,
    )
    asset_control_scale = optional_cfg.asset_control_scale
    camera_scale = optional_cfg.camera_scale
    do_cfg = (
        abs(scale - 1.0) > 1e-6
        or abs(camera_text_scale - 1.0) > 1e-6
        or abs(asset_control_scale - 1.0) > 1e-6
        or abs(camera_scale - 1.0) > 1e-6
    )
    drop_all_control = torch.ones((batch_size,), device=device, dtype=torch.bool)
    asset_null_kind = optional_cfg.asset_null_kind
    camera_null_kind = optional_cfg.camera_null_kind
    full_asset_kind = optional_cfg.full_asset_kind
    full_camera_kind = optional_cfg.full_camera_kind
    full_camera_tokens = optional_cfg.full_camera_tokens
    full_camera_mask = optional_cfg.full_camera_mask
    final_sky_mask_logits = None
    final_sky_mask_refined_logits = None

    for i in range(int(args.val_sample_steps)):
        read_final_sky_mask = False
        step_h = t_steps[i] - t_steps[i + 1]
        sigma = torch.full((batch_size,), float(t_steps[i].item()), device=device)

        def _run_branch(
            *,
            F_asset_tokens: torch.Tensor,
            asset_mask: torch.Tensor | None,
            asset_kind: Any,
            branch_text_tokens: torch.Tensor | None,
            branch_text_mask: torch.Tensor | None,
            camera_kind: Any,
            control_drop_mask: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            out = sf(
                z,
                sigma,
                z_splat,
                scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                F_asset_tokens,
                encoder_attention_mask=asset_mask,
                text_tokens=branch_text_tokens,
                text_attention_mask=branch_text_mask,
                camera_condition_tokens=full_camera_tokens,
                camera_attention_mask=full_camera_mask,
                camera_condition_kind=camera_kind,
                camera_gen_tokens=camera_z,
                camera_gen_anchor_mask=getattr(bundle, "camera_gen_anchor_mask", None),
                sky_gen_tokens=sky_z,
                sky_gen_attention_mask=None,
                return_mid=False,
                return_dict=True,
                return_sky_mask=read_final_sky_mask,
                asset_condition_kind=asset_kind,
                control_drop_mask=control_drop_mask,
                frame_ids=frame_ids,
                fps=None,
            )
            if not isinstance(out, dict):
                raise RuntimeError("SceneFlow return_dict=True must return dicts in pretrain sampling.")
            return out

        out_full = _run_branch(
            F_asset_tokens=bundle.F_asset_tokens,
            asset_mask=bundle.encoder_attention_mask,
            asset_kind=full_asset_kind,
            branch_text_tokens=text_tokens,
            branch_text_mask=text_mask,
            camera_kind=full_camera_kind,
        )
        if not isinstance(out_full, dict):
            raise RuntimeError("SceneFlow return_dict=True must return a dict in pretrain sampling.")
        v_full = out_full["video"]
        v_camera_full = out_full.get("camera")
        v_sky_full = out_full.get("sky")
        if do_cfg:
            out_no_text_full = None
            out_text = None
            out_text_asset = None
            if abs(scale - 1.0) > 1e-6 or abs(camera_text_scale - 1.0) > 1e-6:
                out_no_text_full = _run_branch(
                    F_asset_tokens=bundle.F_asset_tokens,
                    asset_mask=bundle.encoder_attention_mask,
                    asset_kind=full_asset_kind,
                    branch_text_tokens=text_null,
                    branch_text_mask=text_null_mask,
                    camera_kind=full_camera_kind,
                )
            if abs(asset_control_scale - 1.0) > 1e-6:
                out_text = _run_branch(
                    F_asset_tokens=F_uncond,
                    asset_mask=uncond_asset_mask,
                    asset_kind=asset_null_kind,
                    branch_text_tokens=text_tokens,
                    branch_text_mask=text_mask,
                    camera_kind=camera_null_kind,
                    control_drop_mask=drop_all_control,
                )
                out_text_asset = _run_branch(
                    F_asset_tokens=bundle.F_asset_tokens,
                    asset_mask=bundle.encoder_attention_mask,
                    asset_kind=full_asset_kind,
                    branch_text_tokens=text_tokens,
                    branch_text_mask=text_mask,
                    camera_kind=camera_null_kind,
                )
            elif abs(camera_scale - 1.0) > 1e-6:
                out_text_asset = _run_branch(
                    F_asset_tokens=bundle.F_asset_tokens,
                    asset_mask=bundle.encoder_attention_mask,
                    asset_kind=full_asset_kind,
                    branch_text_tokens=text_tokens,
                    branch_text_mask=text_mask,
                    camera_kind=camera_null_kind,
                )

            def _combine_cfg(key: str) -> torch.Tensor | None:
                full = out_full.get(key)
                if full is None:
                    return None
                guided = full
                text_guidance = camera_text_scale if key == "camera" else scale
                if out_no_text_full is not None and abs(text_guidance - 1.0) > 1e-6:
                    no_text_full = out_no_text_full.get(key)
                    if no_text_full is None:
                        raise RuntimeError(f"Text-CFG branch is missing `{key}` predictions.")
                    guided = guided + (text_guidance - 1.0) * (full - no_text_full)
                if abs(asset_control_scale - 1.0) > 1e-6:
                    assert out_text is not None and out_text_asset is not None
                    text = out_text.get(key)
                    text_asset = out_text_asset.get(key)
                    if text is None or text_asset is None:
                        raise RuntimeError(f"Asset-CFG branch is missing `{key}` predictions.")
                    guided = guided + (asset_control_scale - 1.0) * (text_asset - text)
                if abs(camera_scale - 1.0) > 1e-6:
                    assert out_text_asset is not None
                    text_asset = out_text_asset.get(key)
                    if text_asset is None:
                        raise RuntimeError(f"Camera-CFG branch is missing `{key}` predictions.")
                    guided = guided + (camera_scale - 1.0) * (full - text_asset)
                return guided

            v = _combine_cfg("video")
            if v is None:
                raise RuntimeError("CFG branch is missing `video` predictions.")
            v_camera = _combine_cfg("camera") if camera_z is not None and v_camera_full is not None else None
            v_sky = _combine_cfg("sky") if sky_z is not None and v_sky_full is not None else None
            final_sky_mask_logits = _combine_cfg("sky_mask_logits") if read_final_sky_mask else None
            final_sky_mask_refined_logits = (
                _combine_cfg("sky_mask_refined_logits") if read_final_sky_mask else None
            )
        else:
            v = v_full
            v_camera = v_camera_full
            v_sky = v_sky_full
            if read_final_sky_mask:
                final_sky_mask_logits = out_full.get("sky_mask_logits")
                final_sky_mask_refined_logits = out_full.get("sky_mask_refined_logits")
        v = sampler_prediction_to_velocity(sf, v, z, sigma)
        z = z - step_h.to(dtype=z.dtype) * v
        z = M_keep * z_splat + M_edit * z
        if camera_z is not None and v_camera is not None:
            v_camera = sampler_prediction_to_velocity(sf, v_camera, camera_z, sigma)
            camera_z = camera_z - step_h.to(dtype=camera_z.dtype) * v_camera
        if sky_z is not None and v_sky is not None:
            v_sky = sampler_prediction_to_velocity(
                sf,
                v_sky,
                sky_z,
                sigma,
                t_eps=SKY_FLOW_T_EPS,
            )
            sky_z = sky_z - step_h.to(dtype=sky_z.dtype) * v_sky

    z = M_keep * z_splat + M_edit * z
    if return_sky_mask:
        # Reuse the exact CFG branches at the final clean state. This forward
        # predicts only the masks and does not mutate video/camera/sky states.
        sigma = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        read_final_sky_mask = True
        endpoint_full = _run_branch(
            F_asset_tokens=bundle.F_asset_tokens,
            asset_mask=bundle.encoder_attention_mask,
            asset_kind=full_asset_kind,
            branch_text_tokens=text_tokens,
            branch_text_mask=text_mask,
            camera_kind=full_camera_kind,
        )
        endpoint_no_text = endpoint_text = endpoint_text_asset = None
        if do_cfg and (abs(scale - 1.0) > 1e-6 or abs(camera_text_scale - 1.0) > 1e-6):
            endpoint_no_text = _run_branch(
                F_asset_tokens=bundle.F_asset_tokens,
                asset_mask=bundle.encoder_attention_mask,
                asset_kind=full_asset_kind,
                branch_text_tokens=text_null,
                branch_text_mask=text_null_mask,
                camera_kind=full_camera_kind,
            )
        if do_cfg and abs(asset_control_scale - 1.0) > 1e-6:
            endpoint_text = _run_branch(
                F_asset_tokens=F_uncond,
                asset_mask=uncond_asset_mask,
                asset_kind=asset_null_kind,
                branch_text_tokens=text_tokens,
                branch_text_mask=text_mask,
                camera_kind=camera_null_kind,
                control_drop_mask=drop_all_control,
            )
            endpoint_text_asset = _run_branch(
                F_asset_tokens=bundle.F_asset_tokens,
                asset_mask=bundle.encoder_attention_mask,
                asset_kind=full_asset_kind,
                branch_text_tokens=text_tokens,
                branch_text_mask=text_mask,
                camera_kind=camera_null_kind,
            )
        elif do_cfg and abs(camera_scale - 1.0) > 1e-6:
            endpoint_text_asset = _run_branch(
                F_asset_tokens=bundle.F_asset_tokens,
                asset_mask=bundle.encoder_attention_mask,
                asset_kind=full_asset_kind,
                branch_text_tokens=text_tokens,
                branch_text_mask=text_mask,
                camera_kind=camera_null_kind,
            )

        def combine_endpoint(key: str) -> torch.Tensor:
            full = endpoint_full[key]
            value = full
            if endpoint_no_text is not None and abs(scale - 1.0) > 1e-6:
                value = value + (scale - 1.0) * (full - endpoint_no_text[key])
            if abs(asset_control_scale - 1.0) > 1e-6:
                assert endpoint_text is not None and endpoint_text_asset is not None
                value = value + (asset_control_scale - 1.0) * (
                    endpoint_text_asset[key] - endpoint_text[key]
                )
            if abs(camera_scale - 1.0) > 1e-6:
                assert endpoint_text_asset is not None
                value = value + (camera_scale - 1.0) * (full - endpoint_text_asset[key])
            return value

        final_sky_mask_logits = combine_endpoint("sky_mask_logits")
        final_sky_mask_refined_logits = combine_endpoint("sky_mask_refined_logits")
    sky_mask_logits = final_sky_mask_logits
    sky_mask_refined_logits = final_sky_mask_refined_logits
    sky_mask_patch = None if sky_mask_logits is None else torch.sigmoid(sky_mask_logits.float()).to(sky_mask_logits.dtype)
    sky_mask_refined = (
        None
        if sky_mask_refined_logits is None
        else torch.sigmoid(sky_mask_refined_logits.float()).to(sky_mask_refined_logits.dtype)
    )
    if return_sky_mask and (sky_mask_patch is None or sky_mask_refined is None):
        raise RuntimeError("Sampling requested a sky mask but the final denoising step did not produce one.")
    if return_camera or return_sky or return_sky_mask:
        camera_output = camera_z
        if camera_output is not None and str(getattr(sf.config, "camera_generation_representation", "dggt_hidden_v1")) == CAMERA_GENERATION_REPRESENTATION:
            anchor_mask = getattr(bundle, "camera_gen_anchor_mask", None)
            if anchor_mask is None:
                raise RuntimeError("DGGT v3 camera sampling is missing the global camera anchor mask")
            camera_output = sf.denormalize_camera(camera_output, anchor_mask)
        return SimpleNamespace(
            video=z,
            camera_state_dggt=camera_output,
            camera_anchor_mask=getattr(bundle, "camera_gen_anchor_mask", None),
            sky=sky_z,
            sky_mask_logits=sky_mask_logits,
            sky_mask_patch=sky_mask_patch,
            sky_mask_refined_logits=sky_mask_refined_logits,
            sky_mask_refined=sky_mask_refined,
        )
    return z


def sample_pretrain_latents(
    scene_flow: nn.Module,
    bundle,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
    text_encoder: nn.Module | None = None,
) -> torch.Tensor:
    """Backward-compatible alias kept for callers that want default-scale sampling."""
    return cfg_sample_pretrain_latents(scene_flow, bundle, args, step, device, text_encoder=text_encoder)


def build_full_scene_bundle(
    z_clean_n: torch.Tensor,
    kv_dim: int,
    camera_condition_tokens: torch.Tensor | None = None,
    camera_attention_mask: torch.Tensor | None = None,
    camera_condition_kind: Any = None,
    camera_target_clean_n: torch.Tensor | None = None,
    camera_target_state_dggt: torch.Tensor | None = None,
    camera_gen_anchor_mask: torch.Tensor | None = None,
    camera_pose_gt_dggt: torch.Tensor | None = None,
    sky_gen_clean: torch.Tensor | None = None,
    sky_gen_loss_weight: torch.Tensor | None = None,
    sky_gen_attention_mask: torch.Tensor | None = None,
    sky_mask_clean: torch.Tensor | None = None,
    sky_mask_refined_clean: torch.Tensor | None = None,
    frame_ids: torch.Tensor | None = None,
) -> SimpleNamespace:
    B, S, P, _ = z_clean_n.shape
    mask = z_clean_n.new_zeros((B, S, P, 1))
    asset_tokens = z_clean_n.new_zeros((B, 5, S, P, int(kv_dim)))
    asset_mask = torch.zeros((B, 5, S, P), device=z_clean_n.device, dtype=torch.bool)
    return SimpleNamespace(
        z_clean_n=z_clean_n,
        M_preserve=mask,
        M_source=torch.zeros_like(mask),
        M_dest=torch.ones_like(mask),
        F_asset_tokens=asset_tokens,
        encoder_attention_mask=asset_mask,
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        camera_condition_kind=camera_condition_kind,
        camera_target_clean_n=camera_target_clean_n,
        camera_target_state_dggt=camera_target_state_dggt,
        camera_gen_anchor_mask=camera_gen_anchor_mask,
        camera_pose_gt_dggt=camera_pose_gt_dggt,
        camera_target_space=CAMERA_TARGET_SPACE,
        camera_target_source=CAMERA_TARGET_SOURCE,
        camera_loss_gt_space=CAMERA_TARGET_SPACE,
        camera_validation_gt_space=CAMERA_TARGET_SPACE,
        camera_render_pose_space=CAMERA_TARGET_SPACE,
        sky_gen_clean=sky_gen_clean,
        sky_gen_loss_weight=sky_gen_loss_weight,
        # Kept for old checkpoints/callers. Pretrain no longer passes this to
        # SceneFlow because open inference has no GT-derived sky mask.
        sky_gen_attention_mask=sky_gen_attention_mask,
        sky_mask_clean=sky_mask_clean,
        sky_mask_refined_clean=sky_mask_refined_clean,
        frame_ids=frame_ids,
        F_asset_lengths=torch.zeros((B,), device=z_clean_n.device, dtype=torch.long),
        z_splat_n=torch.zeros_like(z_clean_n),
    )


def save_validation_images(
    bundle,
    z_generated_raw: torch.Tensor,
    rgb_images: dict[str, torch.Tensor] | None,
    log_dir: Path,
    step: int,
    args: argparse.Namespace,
    scale_suffix: str | None = None,
    only_generated: bool = False,
    visualization_batch: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Dump validation artifacts.

    `scale_suffix` is appended to every filename so multi-CFG dumps don't clash.
    `only_generated=True` skips latent_pca/target/input_rgb_gt and only
    writes the generated artifacts (used for the secondary CFG scales).
    """
    out_dir = log_dir / "validation" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = min(int(args.val_log_images), int(bundle.z_clean_n.shape[1]))
    suffix = f"__{scale_suffix}" if scale_suffix else ""

    images: dict[str, torch.Tensor] = {
        f"generated_raw_latent_pca{suffix}": _latent_pca_grid(z_generated_raw, args.patch_grid, frames),
        f"abs_error{suffix}": _normalized_mask_grid(
            (z_generated_raw - bundle.z_clean_n).abs().mean(dim=-1, keepdim=True),
            args.patch_grid,
            frames,
        ),
    }
    if not only_generated:
        images["target_latent_pca"] = _latent_pca_grid(bundle.z_clean_n, args.patch_grid, frames)
        sky_mask_clean = getattr(bundle, "sky_mask_clean", None)
        if torch.is_tensor(sky_mask_clean):
            images["target_sky_mask_patch"] = _mask_grid(sky_mask_clean, args.patch_grid, frames)
        sky_mask_refined_clean = getattr(bundle, "sky_mask_refined_clean", None)
        if torch.is_tensor(sky_mask_refined_clean):
            images["target_sky_mask_refined"] = _sky_mask_image_grid(sky_mask_refined_clean, frames)
        if not (rgb_images and "input_rgb_gt" in rgb_images) and visualization_batch is not None:
            gt_images = visualization_batch.get("images")
            if torch.is_tensor(gt_images) and gt_images.ndim == 5:
                images["input_rgb_gt"] = _image_grid(gt_images, frames)

    paths: dict[str, Path] = {}
    for name, tensor in images.items():
        path = out_dir / f"{name}.jpg"
        save_image_grid(tensor, path, nrow=frames)
        paths[name] = path

    if rgb_images:
        skip_for_extra = {"input_rgb_gt", "tokenizer_recon_3dgs_rgb", "dggt_clean_3dgs_rgb"}
        for name, tensor in rgb_images.items():
            if only_generated and name in skip_for_extra:
                continue
            fname = f"{name}{suffix}.jpg" if name.startswith("generated_") else f"{name}.jpg"
            path = out_dir / fname
            save_image_grid(tensor, path, nrow=frames)
            key = f"{name}{suffix}" if name.startswith("generated_") else name
            paths[key] = path
    return paths


def train_step(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    scheduler: Any,
    device: torch.device,
    args: argparse.Namespace,
    text_encoder: nn.Module | None = None,
    *,
    global_step: int | None = None,
    lpips_model: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    is_training = unwrap_ddp(scene_flow).training
    rgb_render_active = should_apply_rgb_render_loss(
        args, global_step, training=is_training
    )
    bundle = build_pretrain_bundle_from_batch(
        batch,
        vggt_model,
        scene_flow,
        device,
        args,
        include_rgb_render_context=rgb_render_active,
    )

    legacy_drop_prob = float(getattr(args, "uncond_drop_prob", 0.0))
    text_drop_prob = float(
        legacy_drop_prob if getattr(args, "text_uncond_drop_prob", None) is None else args.text_uncond_drop_prob
    )
    asset_drop_prob = float(
        legacy_drop_prob if getattr(args, "asset_uncond_drop_prob", None) is None else args.asset_uncond_drop_prob
    )
    camera_drop_prob = float(
        legacy_drop_prob if getattr(args, "camera_uncond_drop_prob", None) is None else args.camera_uncond_drop_prob
    )
    all_drop_prob = float(getattr(args, "all_cond_drop_prob", 0.0))
    # CFG training prerequisite: independently hide text, asset/control, and
    # camera condition tokens per sample. Camera generation targets/losses stay
    # present; only the input camera condition is replaced by a learned null.
    text_drop_mask = sample_uncond_drop_mask(
        int(bundle.z_clean_n.shape[0]),
        text_drop_prob,
        device=bundle.z_clean_n.device,
        training=is_training,
    )
    asset_control_drop_mask = sample_uncond_drop_mask(
        int(bundle.z_clean_n.shape[0]),
        asset_drop_prob,
        device=bundle.z_clean_n.device,
        training=is_training,
    )
    camera_drop_mask = sample_uncond_drop_mask(
        int(bundle.z_clean_n.shape[0]),
        camera_drop_prob,
        device=bundle.z_clean_n.device,
        training=is_training,
    )
    all_drop_mask = sample_uncond_drop_mask(
        int(bundle.z_clean_n.shape[0]),
        all_drop_prob,
        device=bundle.z_clean_n.device,
        training=is_training,
    )
    if all_drop_mask is not None:
        text_drop_mask = all_drop_mask if text_drop_mask is None else (text_drop_mask | all_drop_mask)
        asset_control_drop_mask = (
            all_drop_mask if asset_control_drop_mask is None else (asset_control_drop_mask | all_drop_mask)
        )
        camera_drop_mask = all_drop_mask if camera_drop_mask is None else (camera_drop_mask | all_drop_mask)
    if asset_control_drop_mask is not None:
        bundle = apply_asset_uncond_drop(bundle, asset_control_drop_mask)
    if camera_drop_mask is not None:
        bundle = apply_camera_uncond_drop(bundle, camera_drop_mask)

    M_edit = (bundle.M_source.float() + bundle.M_dest.float()).clamp(0.0, 1.0)
    bundle.M_edit = M_edit
    z_splat = getattr(bundle, "z_splat_n", None)
    if z_splat is None:
        z_splat = torch.zeros_like(bundle.z_clean_n)
    target = build_masked_rectified_flow_target(
        scheduler,
        bundle.z_clean_n,
        z_splat,
        M_edit,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
        loss_weighting_scheme=args.loss_weighting_scheme,
        time_shift=float(args.shift),
    )
    camera_target = build_camera_rectified_flow_target(getattr(bundle, "camera_target_clean_n", None), target)
    camera_gen_attention_mask = None
    camera_flow_supervision_mask = None
    camera_anchor_context_drop_rows = None
    if camera_target is not None:
        anchor_mask = getattr(bundle, "camera_gen_anchor_mask", None)
        if not torch.is_tensor(anchor_mask):
            raise RuntimeError("camera target requires bundle.camera_gen_anchor_mask")
        drop_prob = float(getattr(args, "camera_anchor_context_dropout", 0.0))
        camera_anchor_context_drop_rows = torch.rand(
            int(camera_target.z_t.shape[0]), device=camera_target.z_t.device
        ) < drop_prob
        camera_gen_attention_mask, camera_flow_supervision_mask = build_camera_anchor_context_dropout(
            anchor_mask,
            camera_anchor_context_drop_rows,
        )
    sky_target = build_sky_rectified_flow_target(
        getattr(bundle, "sky_gen_clean", None),
        target,
        loss_weight=getattr(bundle, "sky_gen_loss_weight", None),
    )
    boundary = boundary_mask_from_edit_mask(M_edit, args.patch_grid, radius=1)

    scaffold_tok = torch.zeros_like(bundle.z_clean_n)
    use_repa = float(args.lambda_repa) != 0.0
    text_tokens, text_mask = encode_text_condition(
        text_encoder,
        getattr(bundle, "captions", None),
        drop_mask=text_drop_mask,
    )
    with autocast_context(args, device):
        out = scene_flow(
            target.z_t,
            target.sigmas,
            z_splat,
            scaffold_tok,
            bundle.M_preserve,
            bundle.M_source,
            bundle.M_dest,
            bundle.F_asset_tokens,
            encoder_attention_mask=bundle.encoder_attention_mask,
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
            camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
            camera_condition_kind=getattr(bundle, "camera_condition_kind", None),
            camera_gen_tokens=None if camera_target is None else camera_target.z_t,
            camera_gen_attention_mask=camera_gen_attention_mask,
            camera_gen_anchor_mask=getattr(bundle, "camera_gen_anchor_mask", None),
            sky_gen_tokens=None if sky_target is None else sky_target.z_t,
            sky_gen_attention_mask=None,
            return_mid=use_repa,
            return_dict=True,
            return_sky_mask=True,
            return_base=float(args.base_model_coeff) != 0.0,
            asset_condition_kind=getattr(bundle, "asset_condition_kind", None),
            control_drop_mask=asset_control_drop_mask,
            frame_ids=getattr(bundle, "frame_ids", None),
            fps=None,
        )
        if not isinstance(out, dict):
            raise RuntimeError("SceneFlow return_dict=True must return a dict in pretrain train_step.")
        pred_clean = out["video"]
        pred_base = out.get("video_base")
        mid_repa = out.get("mid_repa") if use_repa else None
        pred_camera = out.get("camera")
        pred_sky = out.get("sky")
        pred_sky_mask_logits = out.get("sky_mask_logits")
        pred_sky_mask_refined_logits = out.get("sky_mask_refined_logits")
        v_pred = model_prediction_to_velocity(scene_flow, pred_clean, target)
        z_pred = model_prediction_to_clean(scene_flow, pred_clean, target)
        v_base_pred = (
            model_prediction_to_velocity(scene_flow, pred_base, target)
            if pred_base is not None
            else None
        )
        loss, logs = compute_total_loss(
            v_pred=v_pred,
            v_gt=target.v_gt,
            eps=target.eps,
            bundle=bundle,
            sd3_weights=target.weights,
            mid_repa=mid_repa,
            z_pred=z_pred,
            z_preserve_target=target.z_cond,
            M_edit=target.M_edit,
            boundary_mask=boundary,
            v_base_pred=v_base_pred,
            base_model_coeff=args.base_model_coeff,
            lambda_flow=args.lambda_flow,
            lambda_preserve=args.lambda_preserve,
            lambda_boundary=args.lambda_boundary,
            lambda_repa=args.lambda_repa,
            lambda_identity=0.0,
            identity_batch=False,
            preserve_floor=args.preserve_floor,
        )
        z_camera_pred = None
        pred_camera_state = None
        if camera_target is not None and pred_camera is not None:
            v_camera_pred = model_prediction_to_velocity(scene_flow, pred_camera, camera_target)
            z_camera_pred = model_prediction_to_clean(scene_flow, pred_camera, camera_target)
            camera_sq_error = (
                v_camera_pred.float()
                - camera_target.v_gt.to(device=v_camera_pred.device, dtype=torch.float32)
            ).square()
            camera_loss_mask = camera_flow_supervision_mask.to(
                device=v_camera_pred.device, dtype=torch.float32
            )
            loss_camera_flow = (camera_sq_error * camera_loss_mask).sum() / (
                camera_loss_mask.sum() * int(camera_sq_error.shape[-1])
            ).clamp_min(1.0)
            loss = loss + float(args.lambda_camera_flow) * loss_camera_flow
            logs["loss_camera_flow"] = float(loss_camera_flow.detach().item())
            logs["camera_anchor_context_dropout_frac"] = float(
                camera_anchor_context_drop_rows.float().mean().detach().item()
            )
            camera_anchor_mask = bundle.camera_gen_anchor_mask
            pred_camera_state = unwrap_ddp(scene_flow).denormalize_camera(
                z_camera_pred.float(), camera_anchor_mask
            )
            camera_pose_rows = ~camera_anchor_context_drop_rows
            if float(args.lambda_camera_pose) != 0.0 and bool(camera_pose_rows.any().item()):
                if getattr(bundle, "camera_loss_gt_space", None) != CAMERA_TARGET_SPACE:
                    raise RuntimeError("camera geometry loss GT must be explicitly marked as DGGT CameraHead space")
                loss_cam_pose, geometry_logs = camera_geometry_loss(
                    pred_camera_state[camera_pose_rows],
                    bundle.camera_target_state_dggt.to(
                        device=pred_camera_state.device, dtype=pred_camera_state.dtype
                    )[camera_pose_rows],
                    camera_anchor_mask[camera_pose_rows],
                    absolute_weight=float(getattr(args, "camera_absolute_weight", 1.0)),
                    relative_weight=float(getattr(args, "camera_relative_weight", 1.0)),
                    smoothness_weight=float(getattr(args, "camera_smoothness_weight", 0.25)),
                )
                loss = loss + float(args.lambda_camera_pose) * loss_cam_pose
                logs["loss_camera_pose"] = float(loss_cam_pose.detach().item())
                for name, value in geometry_logs.items():
                    logs[name] = float(value.detach().item())
                decoded_pred = decode_camera_trajectory(
                    pred_camera_state[camera_pose_rows], camera_anchor_mask[camera_pose_rows]
                )
                decoded_gt = decode_camera_trajectory(
                    bundle.camera_target_state_dggt[camera_pose_rows],
                    camera_anchor_mask[camera_pose_rows],
                )
                logs["camera_translation_error"] = float(
                    (decoded_pred.camera_to_world[..., :3, 3] - decoded_gt.camera_to_world[..., :3, 3])
                    .norm(dim=-1).mean().detach().item()
                )
                logs["camera_rotation_error_deg"] = float(
                    torch.rad2deg(so3_geodesic_angle(
                        decoded_pred.camera_to_world[..., :3, :3],
                        decoded_gt.camera_to_world[..., :3, :3],
                    )).mean().detach().item()
                )
                logs["camera_fov_error_deg"] = float(
                    torch.rad2deg((decoded_pred.fov_xy - decoded_gt.fov_xy).abs()).mean().detach().item()
                )
                pred_rot = rotation_6d_to_matrix(pred_camera_state[camera_pose_rows][..., 3:9])
                identity = torch.eye(3, device=pred_rot.device, dtype=pred_rot.dtype)
                logs["camera_so3_determinant_error"] = float(
                    (torch.det(pred_rot) - 1.0).abs().mean().detach().item()
                )
                logs["camera_so3_orthogonality_error"] = float(
                    (pred_rot.transpose(-1, -2) @ pred_rot - identity).abs().amax(dim=(-2, -1)).mean().detach().item()
                )
            else:
                loss = loss + 0.0 * pred_camera.sum()
                logs["loss_camera_pose"] = 0.0
        elif pred_camera is not None:
            loss = loss + 0.0 * pred_camera.sum()
            logs["loss_camera_flow"] = 0.0
            logs["loss_camera_pose"] = 0.0
        z_sky_pred = None
        if sky_target is not None and pred_sky is not None:
            v_sky_pred = model_prediction_to_velocity(scene_flow, pred_sky, sky_target)
            z_sky_pred = model_prediction_to_clean(scene_flow, pred_sky, sky_target)
            loss_sky_flow = sky_flow_loss(v_sky_pred, sky_target, bundle.sky_gen_clean)
            loss = loss + float(args.lambda_sky_flow) * loss_sky_flow
            logs["loss_sky_flow"] = float(loss_sky_flow.detach().item())
        elif pred_sky is not None:
            loss = loss + 0.0 * pred_sky.sum()
            logs["loss_sky_flow"] = 0.0
        elif getattr(bundle, "sky_gen_clean", None) is not None:
            logs["loss_sky_flow"] = 0.0

        endpoint_every = int(getattr(args, "sky_mask_endpoint_every", 4))
        endpoint_start = int(getattr(args, "sky_mask_endpoint_start_step", 5000))
        endpoint_due = bool(
            is_training
            and global_step is not None
            and int(global_step) >= endpoint_start
            and endpoint_every > 0
            and int(global_step) % endpoint_every == 0
        )
        endpoint_due = endpoint_due or bool(rgb_render_active)
        endpoint_out = None
        if endpoint_due:
            endpoint_video = (
                (1.0 - target.M_edit).to(dtype=z_pred.dtype) * z_splat
                + target.M_edit.to(dtype=z_pred.dtype) * z_pred.detach()
            )
            endpoint_sigma = torch.zeros_like(target.sigmas, dtype=torch.float32)
            endpoint_out = scene_flow(
                endpoint_video,
                endpoint_sigma,
                z_splat,
                scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                bundle.F_asset_tokens,
                encoder_attention_mask=bundle.encoder_attention_mask,
                text_tokens=text_tokens,
                text_attention_mask=text_mask,
                camera_condition_tokens=getattr(bundle, "camera_condition_tokens", None),
                camera_attention_mask=getattr(bundle, "camera_attention_mask", None),
                camera_condition_kind=getattr(bundle, "camera_condition_kind", None),
                camera_gen_tokens=None if z_camera_pred is None else z_camera_pred.detach(),
                camera_gen_attention_mask=camera_gen_attention_mask,
                camera_gen_anchor_mask=getattr(bundle, "camera_gen_anchor_mask", None),
                sky_gen_tokens=None if z_sky_pred is None else z_sky_pred.detach(),
                sky_gen_attention_mask=None,
                return_mid=False,
                return_dict=True,
                return_sky_mask=True,
                asset_condition_kind=getattr(bundle, "asset_condition_kind", None),
                control_drop_mask=asset_control_drop_mask,
                frame_ids=getattr(bundle, "frame_ids", None),
                fps=None,
            )
            if not isinstance(endpoint_out, dict):
                raise RuntimeError("sigma=0 sky-mask endpoint forward must return a dict")
            logs["sky_mask_endpoint_active"] = 1.0
        else:
            logs["sky_mask_endpoint_active"] = 0.0
        sky_mask_target = getattr(bundle, "sky_mask_clean", None)
        if pred_sky_mask_logits is not None and torch.is_tensor(sky_mask_target):
            loss_sky_mask, sky_mask_logs = sky_mask_patch_loss(
                pred_sky_mask_logits,
                sky_mask_target,
                dice_weight=float(getattr(args, "sky_mask_dice_weight", 0.5)),
                pos_weight_max=float(getattr(args, "sky_mask_pos_weight_max", 10.0)),
            )
            loss = loss + float(getattr(args, "lambda_sky_mask", 0.05)) * loss_sky_mask
            logs["loss_sky_mask"] = float(loss_sky_mask.detach().item())
            logs.update(sky_mask_logs)
        elif pred_sky_mask_logits is not None:
            loss = loss + 0.0 * pred_sky_mask_logits.sum()
            logs["loss_sky_mask"] = 0.0
            logs["sky_mask_pred_frac"] = float(torch.sigmoid(pred_sky_mask_logits.float()).mean().detach().item())
            logs["sky_mask_target_frac"] = 0.0
        sky_mask_refined_target = getattr(bundle, "sky_mask_refined_clean", None)
        if pred_sky_mask_refined_logits is not None and torch.is_tensor(sky_mask_refined_target):
            loss_sky_mask_refine, refine_logs = sky_mask_refined_loss(
                pred_sky_mask_refined_logits,
                sky_mask_refined_target,
                dice_weight=float(getattr(args, "sky_mask_dice_weight", 0.5)),
                pos_weight_max=float(getattr(args, "sky_mask_pos_weight_max", 10.0)),
                boundary_weight=float(getattr(args, "sky_mask_refine_boundary_weight", 4.0)),
                boundary_loss_weight=float(getattr(args, "sky_mask_refine_boundary_loss_weight", 0.25)),
            )
            loss = loss + float(getattr(args, "lambda_sky_mask_refine", 0.1)) * loss_sky_mask_refine
            logs["loss_sky_mask_refine"] = float(loss_sky_mask_refine.detach().item())
            logs.update(refine_logs)
        elif pred_sky_mask_refined_logits is not None:
            loss = loss + 0.0 * pred_sky_mask_refined_logits.sum()
            logs["loss_sky_mask_refine"] = 0.0
            logs["sky_mask_refine_pred_frac"] = float(
                torch.sigmoid(pred_sky_mask_refined_logits.float()).mean().detach().item()
            )

        if endpoint_out is not None and torch.is_tensor(sky_mask_target) and torch.is_tensor(sky_mask_refined_target):
            endpoint_patch_logits = endpoint_out["sky_mask_logits"]
            endpoint_refined_logits = endpoint_out["sky_mask_refined_logits"]
            endpoint_patch_loss, endpoint_patch_logs = sky_mask_patch_loss(
                endpoint_patch_logits,
                sky_mask_target,
                dice_weight=float(getattr(args, "sky_mask_dice_weight", 0.5)),
                pos_weight_max=float(getattr(args, "sky_mask_pos_weight_max", 10.0)),
            )
            endpoint_refined_loss, endpoint_refined_logs = sky_mask_refined_loss(
                endpoint_refined_logits,
                sky_mask_refined_target,
                dice_weight=float(getattr(args, "sky_mask_dice_weight", 0.5)),
                pos_weight_max=float(getattr(args, "sky_mask_pos_weight_max", 10.0)),
                boundary_weight=float(getattr(args, "sky_mask_refine_boundary_weight", 4.0)),
                boundary_loss_weight=float(getattr(args, "sky_mask_refine_boundary_loss_weight", 0.25)),
            )
            loss = loss + float(getattr(args, "lambda_sky_mask", 0.05)) * endpoint_patch_loss
            loss = loss + float(getattr(args, "lambda_sky_mask_refine", 0.1)) * endpoint_refined_loss
            logs["loss_sky_mask_endpoint"] = float(endpoint_patch_loss.detach().item())
            logs["loss_sky_mask_refine_endpoint"] = float(endpoint_refined_loss.detach().item())
            logs.update({f"endpoint_{key}": value for key, value in endpoint_patch_logs.items()})
            logs.update({f"endpoint_{key}": value for key, value in endpoint_refined_logs.items()})

        if rgb_render_active:
            if z_camera_pred is None or pred_camera_state is None:
                raise RuntimeError(
                    "Pretrain RGB loss requires SceneFlow-generated camera tokens; "
                    "teacher-camera fallback is forbidden."
                )
            if not torch.is_tensor(getattr(bundle, "rgb_render_images", None)):
                raise RuntimeError("Pretrain RGB render context is missing target images.")
            if not torch.is_tensor(getattr(bundle, "rgb_render_masks", None)):
                raise RuntimeError(
                    "Pretrain RGB loss requires the GT sky mask for loss weighting; "
                    "the renderer itself still uses the predicted sky mask."
                )
            generated_pose = decode_camera_trajectory(
                pred_camera_state,
                bundle.camera_gen_anchor_mask,
            ).pose_encoding
            sky_tokens_for_rgb = None
            if pred_sky is not None and sky_target is not None:
                sky_latent_for_rgb = model_prediction_to_clean(scene_flow, pred_sky, sky_target)
                sky_tokens_for_rgb = decode_sky_patch_tokens(sky_latent_for_rgb)
                sky_view_samples = int(args.rgb_render_max_samples)
                sky_view_frames = int(args.rgb_render_max_frames)
                sky_view_samples = (
                    int(sky_latent_for_rgb.shape[0])
                    if sky_view_samples <= 0
                    else min(sky_view_samples, int(sky_latent_for_rgb.shape[0]))
                )
                sky_view_frames = (
                    int(bundle.rgb_render_images.shape[1])
                    if sky_view_frames <= 0
                    else min(sky_view_frames, int(bundle.rgb_render_images.shape[1]))
                )
                sky_view_loss, sky_view_logs = generated_sky_view_reconstruction_loss(
                    vggt_model=vggt_model,
                    sky_latent=sky_latent_for_rgb[:sky_view_samples],
                    images=bundle.rgb_render_images[:sky_view_samples, :sky_view_frames],
                    sky_mask=bundle.rgb_render_masks[:sky_view_samples, :sky_view_frames],
                    gt_pose_enc_dggt=bundle.camera_pose_gt_dggt[:sky_view_samples, :sky_view_frames],
                    lpips_model=lpips_model,
                    lpips_weight=float(getattr(args, "sky_view_lpips_weight", 0.01)),
                    high_frequency_weight=float(getattr(args, "sky_view_high_frequency_weight", 0.25)),
                )
                loss = loss + float(getattr(args, "lambda_sky_view_reconstruction", 0.1)) * sky_view_loss
                logs.update(sky_view_logs)
                logs["loss_sky_view_weighted"] = float(
                    (float(getattr(args, "lambda_sky_view_reconstruction", 0.1)) * sky_view_loss).detach().item()
                )
            endpoint_refined_logits = None if endpoint_out is None else endpoint_out.get("sky_mask_refined_logits")
            endpoint_patch_logits = None if endpoint_out is None else endpoint_out.get("sky_mask_logits")
            if endpoint_refined_logits is not None:
                render_sky_probability = torch.sigmoid(endpoint_refined_logits.float())
            elif endpoint_patch_logits is not None:
                render_sky_probability = torch.sigmoid(endpoint_patch_logits.float())
            else:
                raise RuntimeError(
                    "Pretrain RGB loss requires a generated sky-mask prediction."
                )
            rgb_result = compute_rgb_render_loss(
                vggt_model=vggt_model,
                scene_flow_root=unwrap_ddp(scene_flow),
                z_clean_pred_n=z_pred,
                images=bundle.rgb_render_images,
                timestamps=bundle.rgb_render_timestamps,
                render_pose_enc_dggt=generated_pose,
                render_sky_probability=render_sky_probability,
                loss_sky_mask_gt=bundle.rgb_render_masks,
                patch_grid=args.patch_grid,
                patch_start_idx=int(bundle.rgb_render_patch_start_idx),
                max_samples=int(args.rgb_render_max_samples),
                max_frames=int(args.rgb_render_max_frames),
                render_stride=int(args.rgb_render_stride),
                background_mode="sky_tokens" if sky_tokens_for_rgb is not None else "black",
                sky_tokens=sky_tokens_for_rgb,
                sky_grid=sky_atlas_shape(args),
                patch_weight_mask=target.M_edit,
                sky_weight=float(args.rgb_render_sky_weight),
                camera_grad_scale=float(args.rgb_render_camera_grad_scale),
                sky_mask_grad_scale=(
                    float(args.rgb_render_sky_mask_grad_scale)
                    * float(rgb_render_loss_ramp(args, global_step))
                ),
                lpips_model=lpips_model,
                lpips_weight=float(args.rgb_render_lpips_weight),
            )
            ramp = rgb_render_loss_ramp(args, global_step)
            weighted = float(args.lambda_rgb_render) * float(ramp) * rgb_result.loss
            loss = loss + weighted
            logs.update(rgb_result.logs)
            logs["loss_rgb_render_weighted"] = float(weighted.detach().item())
            logs["rgb_render_ramp"] = float(ramp)
            logs["rgb_render_active"] = 1.0
        else:
            logs["rgb_render_active"] = 0.0

    logs["kv_tokens_mean"] = float(bundle.F_asset_lengths.float().mean().item())
    asset_source_kinds = getattr(bundle, "asset_condition_source_kind", None)
    if asset_source_kinds is not None:
        source_values = [str(v) for v in (asset_source_kinds if not isinstance(asset_source_kinds, str) else [asset_source_kinds])]
        denom = max(1, len(source_values))
        logs["asset_source_instances_projected_frac"] = sum(v == "instances_projected" for v in source_values) / float(denom)
        logs["asset_source_legacy_dynamic_mask_frac"] = sum(v == "legacy_dynamic_mask" for v in source_values) / float(denom)
        logs["asset_source_empty_frac"] = sum(
            v not in ("instances_projected", "legacy_dynamic_mask") for v in source_values
        ) / float(denom)
    sky_weight = getattr(bundle, "sky_gen_loss_weight", None)
    logs["sky_token_loss_weight_mean"] = (
        float(sky_weight.float().mean().item()) if torch.is_tensor(sky_weight) and sky_weight.numel() > 0 else 0.0
    )
    sf_root = unwrap_ddp(scene_flow)
    logs["asset_token_count"] = estimate_sparse_asset_token_count(
        sf_root,
        bundle.F_asset_tokens,
        bundle.encoder_attention_mask,
    )
    logs["control_token_count"] = estimate_control_token_count(
        sf_root,
        bundle.M_source,
        bundle.M_dest,
        args.patch_grid,
    )
    logs["dest_frac"] = float(bundle.M_dest.float().mean().item())
    logs["edit_frac"] = float(M_edit.float().mean().item())
    logs["sigma_mean"] = float(target.sigmas.float().mean().item())
    logs["text_uncond_drop_frac"] = _mask_frac(text_drop_mask)
    logs["asset_uncond_drop_frac"] = _mask_frac(asset_control_drop_mask)
    logs["camera_uncond_drop_frac"] = _mask_frac(camera_drop_mask)
    logs["all_cond_drop_frac"] = _mask_frac(all_drop_mask)
    logs["loss"] = float(loss.detach().item())
    camera_clean = getattr(bundle, "camera_target_clean_n", None)
    if torch.is_tensor(camera_clean):
        logs["camera_clean_token_norm"] = float(camera_clean.float().norm(dim=-1).mean().item())
    if camera_target is not None:
        logs["camera_noise_token_norm"] = float(camera_target.eps.float().norm(dim=-1).mean().item())
    logs["cfg_text_scale"] = float(getattr(args, "guidance_scale", 1.0))
    logs["cfg_camera_text_scale"] = float(getattr(args, "camera_text_guidance_scale", 1.0))
    logs["cfg_camera_condition_scale"] = float(getattr(args, "camera_guidance_scale", 1.0))
    return loss, logs


def build_pretrain_bundle_from_batch(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    *,
    include_rgb_render_context: bool = False,
):
    images = batch["images"].to(device, non_blocking=True)
    if images.ndim != 5:
        raise ValueError(f"Expected images [B,S,3,H,W], got {tuple(images.shape)}")
    _, seq_len = images.shape[:2]
    if seq_len < 2:
        raise ValueError("SceneFlow pretraining requires sequence_length >= 2 for cross-frame asset conditions.")

    sf_root = unwrap_ddp(scene_flow)
    with torch.no_grad():
        with autocast_context(args, device):
            outputs = vggt_model.get_aggregator_token_outputs(images)
            aggregated_tokens_list = outputs["aggregated_tokens_list"]
            image_tokens_list = outputs["image_tokens_list"]
            patch_start_idx = int(outputs["patch_start_idx"])
            tokens_4 = select_patch_pyramid(image_tokens_list, TOKENIZER_LEVELS, patch_start_idx)
            _, image_tokens_last = split_special_and_patch(image_tokens_list[-1], patch_start_idx)
            if image_tokens_last.shape[-2] != args.patch_grid[0] * args.patch_grid[1]:
                raise ValueError(
                    f"Expected {args.patch_grid[0] * args.patch_grid[1]} patch tokens, "
                    f"got {image_tokens_last.shape[-2]}."
                )
            z_clean = vggt_model.scene_tokenizer.encode(tokens_4, patch_grid=args.patch_grid)
            # One frozen DGGT aggregator pass feeds both the tokenizer and the
            # CameraHead.  This pose is the sole camera generation/loss/render
            # target; Waymo cameras below remain 20D conditioning only.
            camera_pose_gt_dggt = vggt_model.camera_head(aggregated_tokens_list)[-1].float()
        z_clean_n = sf_root.normalize(z_clean.float())
        camera_to_world_gt = batch.get("camera_to_world_corrected")
        intrinsics_gt = batch.get("intrinsics")
        if not torch.is_tensor(camera_to_world_gt) or not torch.is_tensor(intrinsics_gt):
            raise RuntimeError(
                "Raw Waymo pretrain batch is missing camera_to_world_corrected/intrinsics; "
                "SceneFlow camera conditioning must use Waymo camera parameters, while camera generation targets stay in DGGT space."
            )
        raw_hw = batch.get("raw_image_size_hw")
        if raw_hw is None:
            raise RuntimeError(
                "Raw Waymo pretrain batch is missing raw_image_size_hw. Repair the data/cache metadata before training."
            )
        c2w_all = camera_to_world_gt.to(device=device, dtype=torch.float32, non_blocking=True)
        intrinsics_all = intrinsics_gt.to(device=device, dtype=torch.float32, non_blocking=True)
        raw_hw_front = torch.as_tensor(raw_hw, device=device)
        if raw_hw_front.ndim >= 3 and raw_hw_front.shape[-1] == 2:
            raw_hw_front = raw_hw_front[:, 0]
        camera_target_state_dggt, camera_gen_anchor_mask = camera_state_from_dggt_pose_enc(
            camera_pose_gt_dggt
        )
        camera_target_clean_n = sf_root.normalize_camera(camera_target_state_dggt, camera_gen_anchor_mask)
        camera_condition_tokens, camera_attention_mask = camera_summary_from_waymo_gt(
            c2w_all,
            intrinsics_all,
            image_hw=raw_hw_front,
            trajectory_anchor_to_world=(
                batch["camera_trajectory_anchor_to_world_corrected"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                if torch.is_tensor(batch.get("camera_trajectory_anchor_to_world_corrected"))
                else None
            ),
            previous_camera_to_world=(
                batch["camera_previous_to_world_corrected"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                if torch.is_tensor(batch.get("camera_previous_to_world_corrected"))
                else None
            ),
        )
        del outputs, aggregated_tokens_list, image_tokens_list, tokens_4, z_clean

    masks = batch.get("masks", batch.get("sky_mask"))
    masks_device = None if masks is None else masks.to(device, non_blocking=True)
    sky_mask_clean = build_sky_mask_patch_target(
        images,
        masks_device,
        patch_grid=args.patch_grid,
    )
    sky_mask_refined_clean = build_sky_mask_refined_target(
        images,
        masks_device,
        refined_hw=sky_mask_refine_shape(args),
    )
    sky_gen_clean = None
    sky_gen_loss_weight = None
    sky_atlas_clean = None
    sky_atlas_observation_mask = None
    if sky_generation_enabled(args):
        sky_h, sky_w = sky_grid_shape(args)
        atlas_h, atlas_w = sky_atlas_shape(args)
        sky_extrinsics, sky_intrinsics = pose_encoding_to_extri_intri(
            camera_pose_gt_dggt,
            (int(images.shape[-2]), int(images.shape[-1])),
        )
        sky_atlas_clean, sky_atlas_observation_mask = build_sky_atlas_from_images(
            images,
            masks_device,
            atlas_hw=(atlas_h, atlas_w),
            extrinsics=sky_extrinsics,
            intrinsics=sky_intrinsics,
        )
        sky_gen_clean = pack_sky_rgb_atlas(sky_atlas_clean).to(dtype=images.dtype)
        latent_coverage = torch.nn.functional.adaptive_avg_pool2d(
            sky_atlas_observation_mask.float(), (sky_h, sky_w)
        )
        sky_gen_loss_weight = latent_coverage.reshape(images.shape[0], sky_h * sky_w)

    frame_ids_raw = batch.get("frame_ids")
    if frame_ids_raw is not None:
        if torch.is_tensor(frame_ids_raw):
            frame_ids = frame_ids_raw.to(device=device, dtype=torch.long, non_blocking=True)
        else:
            frame_ids = torch.as_tensor(frame_ids_raw, device=device, dtype=torch.long)
        if frame_ids.ndim == 1:
            frame_ids = frame_ids.view(1, -1).expand(int(z_clean_n.shape[0]), -1)
    else:
        start_idx = batch.get("start_idx")
        clip_index = batch.get("clip_index")
        if torch.is_tensor(start_idx):
            start_idx_t = start_idx.to(device=device, dtype=torch.long).view(-1)
        else:
            start_idx_t = torch.as_tensor(start_idx, device=device, dtype=torch.long).view(-1)
        if clip_index is None:
            clip_index_t = torch.div(start_idx_t, 29, rounding_mode="floor")
        elif torch.is_tensor(clip_index):
            clip_index_t = clip_index.to(device=device, dtype=torch.long).view(-1)
        else:
            clip_index_t = torch.as_tensor(clip_index, device=device, dtype=torch.long).view(-1)
        local_start = start_idx_t - clip_index_t * 29
        frame_ids = local_start[:, None] + torch.arange(int(z_clean_n.shape[1]), device=device, dtype=torch.long).view(1, -1)
    if frame_ids.shape != (int(z_clean_n.shape[0]), int(z_clean_n.shape[1])):
        raise ValueError(
            f"pretrain frame_ids must be [B,S]={tuple(z_clean_n.shape[:2])}, got {tuple(frame_ids.shape)}"
        )

    bundle = build_full_scene_bundle(
        z_clean_n,
        kv_dim=z_clean_n.shape[-1],
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        camera_target_clean_n=camera_target_clean_n,
        camera_target_state_dggt=camera_target_state_dggt,
        camera_gen_anchor_mask=camera_gen_anchor_mask,
        camera_pose_gt_dggt=camera_pose_gt_dggt,
        sky_gen_clean=sky_gen_clean,
        sky_gen_loss_weight=sky_gen_loss_weight,
        sky_gen_attention_mask=None,
        sky_mask_clean=sky_mask_clean,
        sky_mask_refined_clean=sky_mask_refined_clean,
        frame_ids=frame_ids.contiguous(),
    )
    bundle.sky_atlas_clean = sky_atlas_clean
    bundle.sky_atlas_observation_mask = sky_atlas_observation_mask
    object_patch_mask = batch.get("pretrain_object_patch_mask")
    source_kind_raw = batch.get("pretrain_asset_source_kind")
    if isinstance(source_kind_raw, str):
        asset_source_kinds = [source_kind_raw] * int(z_clean_n.shape[0])
    elif source_kind_raw is None or torch.is_tensor(source_kind_raw):
        asset_source_kinds = ["unknown"] * int(z_clean_n.shape[0])
    else:
        asset_source_kinds = [str(v) for v in list(source_kind_raw)]
        if len(asset_source_kinds) != int(z_clean_n.shape[0]):
            asset_source_kinds = ["unknown"] * int(z_clean_n.shape[0])

    if torch.is_tensor(object_patch_mask):
        asset_tokens, asset_mask, asset_lengths, asset_kinds = build_pretrain_asset_slots_from_object_patch_mask(
            z_clean_n,
            object_patch_mask,
            max_assets=5,
            # z_clean_n is normalized to roughly unit scale. A small 0.15 std
            # perturbation keeps coarse appearance conditioning while weakening
            # exact per-patch latent copying from the target scene.
            corruption_noise_std=0.0,  # 0.15
        )
        for row, length in enumerate(asset_lengths.detach().cpu().tolist()):
            if int(length) > 0:
                asset_source_kinds[row] = "instances_projected"
    else:
        asset_tokens, asset_mask, asset_lengths, asset_kinds = build_pretrain_asset_slots_from_object_patch_mask(
            z_clean_n,
            None,
            max_assets=5,
            corruption_noise_std=0.0,
        )

    dynamic_mask = batch.get("dynamic_mask")
    needs_legacy = asset_lengths.detach().to(device=z_clean_n.device).eq(0)
    if torch.is_tensor(dynamic_mask) and bool(needs_legacy.any().item()):
        legacy_tokens, legacy_mask, legacy_lengths, legacy_kinds = build_pretrain_asset_slots_from_dynamic_mask(
            z_clean_n,
            dynamic_mask,
            args.patch_grid,
            max_assets=5,
            corruption_noise_std=0.0,  # 0.15
        )
        for row, needs_row in enumerate(needs_legacy.detach().cpu().tolist()):
            if not needs_row:
                continue
            asset_tokens[row] = legacy_tokens[row]
            asset_mask[row] = legacy_mask[row]
            asset_lengths[row] = legacy_lengths[row]
            asset_kinds[row] = legacy_kinds[row]
            if int(legacy_lengths[row].detach().cpu().item()) > 0:
                asset_source_kinds[row] = "legacy_dynamic_mask"

    bundle.F_asset_tokens = asset_tokens
    bundle.encoder_attention_mask = asset_mask
    bundle.F_asset_lengths = asset_lengths
    bundle.asset_condition_kind = asset_kinds
    bundle.asset_condition_source_kind = asset_source_kinds
    bundle.captions = captions_from_pretrain_batch(batch, int(z_clean_n.shape[0]))
    if bool(include_rgb_render_context):
        timestamps = batch.get("timestamps")
        if not torch.is_tensor(timestamps):
            timestamps = torch.as_tensor(timestamps, device=device, dtype=torch.float32)
        else:
            timestamps = timestamps.to(device=device, dtype=torch.float32, non_blocking=True)
        bundle.rgb_render_images = images.detach()
        bundle.rgb_render_masks = None if masks_device is None else masks_device.detach()
        bundle.rgb_render_timestamps = timestamps.detach()
        bundle.rgb_render_patch_start_idx = int(patch_start_idx)
    return bundle


@torch.no_grad()
def run_validation(
    loader: DataLoader,
    vggt_model: VGGT,
    scene_flow: nn.Module,
    scheduler: Any,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    log_dir: Path,
    wandb_run,
    ema: EMAModel | None = None,
    text_encoder: nn.Module | None = None,
) -> dict[str, float]:
    scene_flow_was_training = scene_flow.training
    scene_flow.eval()

    # Validate under EMA weights (DiT/SD3/Wan/RAE all sample from EMA).
    # Raw mid-training weights give drastically worse diffusion samples.
    # Every rank performs the identical param swap so DDP stays in sync.
    use_val_ema = ema is not None and not args.no_val_ema
    ema_params = list(unwrap_ddp(scene_flow).parameters()) if use_val_ema else None
    if use_val_ema:
        ema.store(ema_params)
        ema.copy_to(ema_params)

    sums: dict[str, float] = {}
    count = 0
    first_batch: dict[str, Any] | None = None

    iterator = loader
    if is_main_process() and not args.no_tqdm:
        iterator = tqdm(
            loader,
            total=args.val_batches,
            desc=f"val {step:06d}",
            dynamic_ncols=True,
            leave=False,
        )

    for batch in iterator:
        if count >= args.val_batches:
            break
        if first_batch is None and is_main_process():
            first_batch = _slice_batch_for_visualization(batch, max_samples=1)
        loss, logs = train_step(batch, vggt_model, scene_flow, scheduler, device, args, text_encoder)
        logs = dict(logs)
        logs["loss"] = float(loss.detach().item())
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1

    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["batches"] = float(count)

    if is_main_process():
        if first_batch is not None and args.val_log_images > 0:
            # Free cached CUDA memory from the validation loss loop so the
            # memory-intensive CFG sampling + rendering has maximum headroom.
            torch.cuda.empty_cache()

            first_bundle = build_pretrain_bundle_from_batch(
                first_batch,
                vggt_model,
                scene_flow,
                device,
                args,
            )

            # Primary scale samples drive latent PCA / abs_error. Extra scales
            # only contribute additional RGB grids for side-by-side CFG comparison.
            primary_scale = float(args.guidance_scale)
            extra_scales = []
            if args.val_guidance_scales:
                for s in args.val_guidance_scales.split(","):
                    s = s.strip()
                    if not s:
                        continue
                    s_val = float(s)
                    if abs(s_val - primary_scale) > 1e-6:
                        extra_scales.append(s_val)

            generated_sample = cfg_sample_pretrain_latents(
                scene_flow,
                first_bundle,
                args,
                step,
                device,
                guidance_scale=primary_scale,
                text_encoder=text_encoder,
                return_camera=True,
                return_sky=sky_generation_enabled(args),
                return_sky_mask=True,
            )
            z_generated_raw = generated_sample.video
            camera_generated_raw = generated_sample.camera_state_dggt
            sky_generated_raw = generated_sample.sky
            sky_mask_generated_patch = generated_sample.sky_mask_patch
            sky_mask_generated_refined = generated_sample.sky_mask_refined
            if camera_generated_raw is None:
                raise RuntimeError("Pretrain validation sampling did not return generated camera state.")
            if sky_mask_generated_patch is None or sky_mask_generated_refined is None:
                raise RuntimeError("Pretrain validation sampling did not return generated sky mask.")
            camera_pose_gt_dggt = getattr(first_bundle, "camera_pose_gt_dggt", None)
            camera_target_state_dggt = getattr(first_bundle, "camera_target_state_dggt", None)
            if camera_pose_gt_dggt is None or camera_target_state_dggt is None:
                raise RuntimeError("Pretrain validation bundle is missing DGGT-space camera GT.")
            with torch.amp.autocast(device_type=device.type, enabled=False):
                camera_generated_pose = decode_pose_from_camera_features(
                    vggt_model,
                    camera_generated_raw.to(device),
                )
            metrics.update(
                camera_pose_validation_metrics(
                    camera_generated_pose,
                    camera_pose_gt_dggt,
                    prefix="sample_camera",
                )
            )
            metrics.update(
                camera_feature_validation_metrics(
                    camera_generated_raw,
                    camera_target_state_dggt,
                    prefix="sample_camera_feature",
                )
            )
            sky_feature_gt = getattr(first_bundle, "sky_gen_clean", None)
            if sky_generated_raw is not None and sky_feature_gt is not None:
                metrics.update(
                    sky_token_validation_metrics(
                        sky_generated_raw,
                        sky_feature_gt,
                        prefix="sample_sky",
                        loss_weight=getattr(first_bundle, "sky_gen_loss_weight", None),
                    )
                )
            metrics.update(
                sky_mask_validation_metrics(
                    sky_mask_generated_patch,
                    getattr(first_bundle, "sky_mask_clean", None),
                    prefix="sample_sky_mask",
                )
            )
            metrics.update(
                sky_mask_validation_metrics(
                    sky_mask_generated_refined,
                    getattr(first_bundle, "sky_mask_refined_clean", None),
                    prefix="sample_sky_mask_refine",
                )
            )
            del camera_generated_pose
            rgb_images = None
            if not args.no_val_render_rgb:
                # Free CFG sampling intermediates before the heavy rendering.
                torch.cuda.empty_cache()
                rgb_images = render_validation_generated_rgb(
                    first_batch,
                    vggt_model,
                    scene_flow,
                    z_generated_raw,
                    args,
                    device,
                    generated_camera_features=camera_generated_raw,
                    generated_sky_tokens=sky_generated_raw,
                    generated_sky_mask_patch=sky_mask_generated_patch,
                    generated_sky_mask_refined=sky_mask_generated_refined,
                )
            image_paths = save_validation_images(
                first_bundle,
                z_generated_raw,
                rgb_images,
                log_dir,
                step,
                args,
                scale_suffix=f"cfg{primary_scale:g}",
                visualization_batch=first_batch,
            )

            extra_paths: dict[str, Path] = {}
            for s_val in extra_scales:
                extra_sample = cfg_sample_pretrain_latents(
                    scene_flow,
                    first_bundle,
                    args,
                    step,
                    device,
                    guidance_scale=s_val,
                    text_encoder=text_encoder,
                    return_camera=True,
                    return_sky=sky_generation_enabled(args),
                    return_sky_mask=True,
                )
                z_extra = extra_sample.video
                camera_extra = extra_sample.camera_state_dggt
                sky_extra = extra_sample.sky
                sky_mask_extra = extra_sample.sky_mask_patch
                sky_mask_refined_extra = extra_sample.sky_mask_refined
                if camera_extra is None:
                    raise RuntimeError("Pretrain validation sampling did not return generated camera state.")
                if sky_mask_extra is None or sky_mask_refined_extra is None:
                    raise RuntimeError("Pretrain validation sampling did not return generated sky mask.")
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    camera_extra_pose = decode_pose_from_camera_features(
                        vggt_model,
                        camera_extra.to(device),
                    )
                metrics.update(
                    camera_pose_validation_metrics(
                        camera_extra_pose,
                        camera_pose_gt_dggt,
                        prefix=_cfg_metric_prefix("sample_camera", s_val),
                    )
                )
                metrics.update(
                    camera_feature_validation_metrics(
                        camera_extra,
                        camera_target_state_dggt,
                        prefix=_cfg_metric_prefix("sample_camera_feature", s_val),
                    )
                )
                if sky_extra is not None and sky_feature_gt is not None:
                    metrics.update(
                        sky_token_validation_metrics(
                            sky_extra,
                            sky_feature_gt,
                            prefix=_cfg_metric_prefix("sample_sky", s_val),
                            loss_weight=getattr(first_bundle, "sky_gen_loss_weight", None),
                        )
                    )
                metrics.update(
                    sky_mask_validation_metrics(
                        sky_mask_extra,
                        getattr(first_bundle, "sky_mask_clean", None),
                        prefix=_cfg_metric_prefix("sample_sky_mask", s_val),
                    )
                )
                metrics.update(
                    sky_mask_validation_metrics(
                        sky_mask_refined_extra,
                        getattr(first_bundle, "sky_mask_refined_clean", None),
                        prefix=_cfg_metric_prefix("sample_sky_mask_refine", s_val),
                    )
                )
                del camera_extra_pose
                rgb_extra = None
                if not args.no_val_render_rgb:
                    rgb_extra = render_validation_generated_rgb(
                        first_batch,
                        vggt_model,
                        scene_flow,
                        z_extra,
                        args,
                        device,
                        generated_camera_features=camera_extra,
                        generated_sky_tokens=sky_extra,
                        generated_sky_mask_patch=sky_mask_extra,
                        generated_sky_mask_refined=sky_mask_refined_extra,
                    )
                extra_paths.update(
                    save_validation_images(
                        first_bundle,
                        z_extra,
                        rgb_extra,
                        log_dir,
                        step,
                        args,
                        scale_suffix=f"cfg{s_val:g}",
                        only_generated=True,
                    )
                )

            if wandb_run is not None:
                import wandb

                image_log: dict[str, Any] = {}
                for name, path in image_paths.items():
                    image_log[f"validation/{name}"] = wandb.Image(str(path))
                for name, path in extra_paths.items():
                    image_log[f"validation/{name}"] = wandb.Image(str(path))
                wandb_run.log(image_log, step=step)
        metrics_text = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"[validation {step:06d}] {metrics_text}", flush=True)
        log_wandb(wandb_run, metrics, step, "validation")

    if use_val_ema:
        ema.restore(ema_params)
    if scene_flow_was_training:
        scene_flow.train()
    # Rank 0 does CFG sampling + RGB rendering after the metric loop, which
    # can take seconds. A barrier here keeps all ranks aligned so the next
    # training iteration's allreduce won't stall on a rank-0 catch-up.
    if is_distributed():
        dist.barrier()
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneFlow pretraining on raw Waymo clips.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, default=None)
    parser.add_argument("--dggt_ckpt_path", type=str, required=True)
    parser.add_argument("--tokenizer_ckpt_path", type=str, default=None)
    parser.add_argument("--feature_stats_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument(
        "--caption_root",
        type=str,
        default="/data/disk2/lyy_dataset/waymo_processed_dggt/training_captions",
    )
    parser.add_argument(
        "--val_caption_root",
        type=str,
        default="/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions",
        help="Optional caption root for validation; defaults to a sibling validation_captions directory when available.",
    )
    parser.add_argument("--text_encoder_path", type=str, default="/home/dancer/model/Qwen/Qwen3-0.6B/")
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--no_text_condition", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--warm_start_path", type=str, default=None,
                        help="加载模型权重以从头开始训练，不恢复优化器和step")
    parser.add_argument(
        "--partial_warm_start_non_sky",
        action="store_true",
        help="Explicitly transfer shape-compatible non-sky weights from an old checkpoint; optimizer/EMA are reset.",
    )
    parser.add_argument("--patch_grid_h", type=int, default=25)
    parser.add_argument("--patch_grid_w", type=int, default=37)
    parser.add_argument("--rope_max_position", type=int, default=16384)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=1024,
        help=(
            "Tokenizer latent channel count. Must match the tokenizer "
            "ckpt's output dim and the channel count in feature_stats. "
            "Sets WanSceneFlow out_channels; the RAEv2 DDT visual embedders "
            "consume z_t directly while in_channels keeps the legacy packed "
            "control dimensionality 3 * latent_dim + 3 for compatibility. "
            "RAE tests 384/768/1024; "
            "the 4-layer DGGT pyramid (12288-D) compresses naturally to 1024 "
            "(12:1 ratio) — better than 768 (16:1) for our pretrained-feature "
            "input. Theorem 1 requires trunk hidden_dim >= latent_dim "
            "(we have hidden=1440, OK up to latent_dim=1440)."
        ),
    )

    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end", type=int, default=600)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--pretrain_instance_cache_size",
        type=int,
        default=8,
        help=(
            "Maximum parsed Waymo instance-metadata scenes retained per DataLoader worker. "
            "The bounded LRU prevents persistent workers from caching the full dataset; "
            "8 preserves a small hot set without changing worker count or prefetching."
        ),
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="DataLoader batches prefetched per worker. Keep low because pretrain batches are large.",
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_true",
        help="Disable persistent DataLoader workers.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable DataLoader pin_memory. Disabled by default because pretrain batches are large.",
    )
    parser.add_argument(
        "--mp_sharing_strategy",
        type=str,
        default="file_system",
        choices=("file_system", "file_descriptor"),
        help="Torch multiprocessing tensor sharing strategy for DataLoader workers.",
    )
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50,
                        help="Plain-text log cadence when tqdm is disabled (--no_tqdm).")
    parser.add_argument("--wandb_log_every", type=int, default=50,
                        help="Report averaged training metrics to wandb every N optimizer steps.")
    parser.add_argument("--val_scene_start", type=int, default=None)
    parser.add_argument("--val_scene_end", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--val_log_images", type=int, default=10)
    parser.add_argument("--val_sample_steps", type=int, default=30)
    parser.add_argument(
        "--val_sliding_window",
        type=int,
        default=10,
        help="Validation CFG sampling window. 0 disables sliding; use sequence_length/training S for long clips.",
    )
    parser.add_argument(
        "--val_sliding_stride",
        type=int,
        default=7,
        help="Validation CFG sampling stride. 0 defaults to a three-frame overlap; overlap is mandatory for long clips.",
    )
    parser.add_argument("--no_val_render_rgb", action="store_true")
    parser.add_argument(
        "--no_val_ema",
        action="store_true",
        help=(
            "Disable EMA weights for validation. By default validation "
            "(loss + CFG samples + RGB render) runs under EMA weights, "
            "which is mandatory for meaningful diffusion samples — raw "
            "mid-training weights produce far worse samples than the model "
            "actually is. All ranks swap identically so DDP stays in sync."
        ),
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--final_lr", type=float, default=2e-5)
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=("linear", "cosine"))
    parser.add_argument("--decay_end_steps", type=int, default=0,
                        help="LR decay end step. 0 means --max_steps, matching step-based RAE training.")
    parser.add_argument("--warmup_from_zero", action="store_true",
                        help="RAEv2 t2i keeps warmup_from_zero=false by default.")
    parser.add_argument("--optimizer_type", type=str, default="gmuon", choices=("gmuon", "adamw"))
    parser.add_argument("--gmuon_momentum", type=float, default=0.95)
    parser.add_argument("--gmuon_nesterov", action="store_true", default=True)
    parser.add_argument("--no_gmuon_nesterov", dest="gmuon_nesterov", action="store_false")
    parser.add_argument("--gmuon_ns_coefficients_preset", type=str, default="POLAR_EXPRESS_COEFFICIENTS")
    parser.add_argument("--gmuon_ns_use_kernels", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="RAE official config uses wd=0.0 for from-scratch DiT on frozen-encoder latents.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9995,
                        help="RAE uses 0.9995 (half-life ~1.4k steps); smoother validation than 0.999.")
    parser.add_argument(
        "--shift",
        type=float,
        default=10.0,
        help=(
            "Manually specified FlowMatch / RAE time-distribution shift for "
            "both training timestep sampling and validation ODE sampling. Do "
            "not multiply by video frame count; sweep this value explicitly."
        ),
    )
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_preserve", type=float, default=1.0)
    parser.add_argument(
        "--lambda_repa",
        type=float,
        default=0.0,
        help=(
            "REPA (Yu et al. 2024) representation-alignment weight. >0 enables "
            "return_mid and aligns the trunk's mid-block features "
            "(repa_layer_depth) to the clean latent z_clean_n via RAEv2 MSE. "
            "RAEv2 t2i defaults to use_repa=false, so this defaults to 0.0. "
            "SceneFlow anchors repa_proj with zero weight for DDP stability when disabled."
        ),
    )
    parser.add_argument("--base_model_coeff", type=float, default=0.25)
    parser.add_argument("--lambda_boundary", type=float, default=0.25)
    parser.add_argument("--preserve_floor", type=float, default=0.2)
    parser.add_argument(
        "--lambda_camera_flow",
        type=float,
        default=0.1,
        help="Flow-matching loss weight for generated normalized DGGT CameraHead pose tokens.",
    )
    parser.add_argument(
        "--lambda_camera_pose",
        type=float,
        default=1.0,
        help="Geometry loss weight after denormalizing and integrating the generated 11D camera trajectory.",
    )
    parser.add_argument("--camera_translation_weight", type=float, default=1.0)
    parser.add_argument("--camera_rotation_weight", type=float, default=1.0)
    parser.add_argument("--camera_fov_weight", type=float, default=1.0)
    parser.add_argument("--camera_absolute_weight", type=float, default=1.0)
    parser.add_argument("--camera_relative_weight", type=float, default=1.0)
    parser.add_argument("--camera_smoothness_weight", type=float, default=0.25)
    parser.add_argument(
        "--no_sky_generation",
        action="store_true",
        help=(
            "Disable sky generation tokens/loss. Pretrain generated RGB validation "
            "then keeps sky background pure black."
        ),
    )
    parser.add_argument("--sky_grid_h", type=int, default=DEFAULT_SKY_GRID[0])
    parser.add_argument("--sky_grid_w", type=int, default=DEFAULT_SKY_GRID[1])
    parser.add_argument("--sky_atlas_h", type=int, default=DEFAULT_SKY_ATLAS_HW[0])
    parser.add_argument("--sky_atlas_w", type=int, default=DEFAULT_SKY_ATLAS_HW[1])
    parser.add_argument(
        "--sky_representation_version", choices=(SKY_REPRESENTATION_VERSION,),
        default=SKY_REPRESENTATION_VERSION,
    )
    parser.add_argument(
        "--sky_unobserved_loss_weight",
        type=float,
        default=DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT,
        help=(
            "Soft loss weight for sky atlas cells not observed by the clip sky mask. "
            "This is used only for training supervision, never as a model input mask."
        ),
    )
    parser.add_argument(
        "--lambda_sky_flow",
        type=float,
        default=0.1,
        help="Flow-matching loss weight for generated scene-level sky RGB tokens.",
    )
    parser.add_argument(
        "--lambda_sky_mask",
        type=float,
        default=0.05,
        help="Auxiliary BCE+Dice loss weight for SceneFlow-generated patch-grid sky mask logits.",
    )
    parser.add_argument(
        "--lambda_sky_mask_refine",
        type=float,
        default=0.1,
        help="Auxiliary BCE+Dice+boundary loss weight for refined dense sky mask logits.",
    )
    parser.add_argument(
        "--sky_mask_refine_scale",
        type=int,
        default=4,
        help="Power-of-two spatial upsample factor for refined sky mask logits relative to the patch grid.",
    )
    parser.add_argument(
        "--sky_mask_refine_channels",
        type=int,
        default=256,
        help="Hidden channels for the lightweight refined sky mask decoder.",
    )
    parser.add_argument(
        "--sky_mask_dice_weight",
        type=float,
        default=0.5,
        help="Dice component weight inside the sky mask auxiliary loss.",
    )
    parser.add_argument(
        "--sky_mask_pos_weight_max",
        type=float,
        default=10.0,
        help="Maximum positive-class weight for imbalanced sky mask BCE.",
    )
    parser.add_argument(
        "--sky_mask_refine_boundary_weight",
        type=float,
        default=4.0,
        help="Boundary-band multiplier inside the refined sky mask auxiliary loss.",
    )
    parser.add_argument(
        "--sky_mask_refine_boundary_loss_weight",
        type=float,
        default=0.25,
        help="Weight of the boundary BCE term inside the refined sky mask auxiliary loss.",
    )
    parser.add_argument("--sky_mask_endpoint_start_step", type=int, default=5000)
    parser.add_argument("--sky_mask_endpoint_every", type=int, default=4)
    parser.add_argument(
        "--lambda_rgb_render",
        type=float,
        default=0.01,
        help="Deployment-aligned differentiable RGB loss using generated camera/depth/GS.",
    )
    parser.add_argument("--rgb_render_every", type=int, default=4)
    parser.add_argument(
        "--rgb_render_start_step",
        type=int,
        default=5000,
        help="Keep RGB supervision disabled until video/camera flow has a stable x0 estimate.",
    )
    parser.add_argument("--rgb_render_warmup_steps", type=int, default=5000)
    parser.add_argument("--rgb_render_max_samples", type=int, default=1)
    parser.add_argument("--rgb_render_max_frames", type=int, default=4)
    parser.add_argument("--rgb_render_stride", type=int, default=2)
    parser.add_argument("--rgb_render_sky_weight", type=float, default=1.0)
    parser.add_argument(
        "--rgb_render_camera_grad_scale",
        type=float,
        default=0.0,
        help="RGB-to-generated-camera gradient scale; forward always uses generated camera.",
    )
    parser.add_argument(
        "--rgb_render_sky_mask_grad_scale",
        type=float,
        default=0.05,
        help="RGB-to-sky-mask gradient scale; render always uses the predicted mask.",
    )
    parser.add_argument("--rgb_render_lpips_weight", type=float, default=0.01)
    parser.add_argument("--rgb_render_lpips_net", type=str, default="alex")
    parser.add_argument("--lambda_sky_view_reconstruction", type=float, default=0.1)
    parser.add_argument("--sky_view_lpips_weight", type=float, default=0.01)
    parser.add_argument("--sky_view_high_frequency_weight", type=float, default=0.25)

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help=(
            "CFG scale for validation sampling. RAE's reported FID 1.51 uses "
            "scale=1.0 (no guidance). Higher scales amplify per-patch noise "
            "into grid artifacts early in training; bump only after the model "
            "converges enough that cond/uncond diverge meaningfully."
        ),
    )
    parser.add_argument("--asset_control_guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--camera_guidance_scale",
        type=float,
        default=1.0,
        help=(
            "CFG scale for camera condition in pretrain validation sampling. "
            "1.0 is no-op; values >1 compare text+asset+camera against text+asset+camera-null."
        ),
    )
    parser.add_argument(
        "--camera_text_guidance_scale",
        type=float,
        default=1.0,
        help="Independent text-CFG scale for generated camera state; global text CFG does not affect camera.",
    )
    parser.add_argument(
        "--camera_anchor_context_dropout",
        type=float,
        default=0.25,
        help=(
            "Training probability of hiding the global generated-camera anchor token while "
            "supervising all delta tokens; covers anchorless later sliding windows."
        ),
    )
    parser.add_argument("--asset_position_mode", choices=("localized", "canonical"), default="localized")
    parser.add_argument("--uncond_drop_prob", type=float, default=0.1,
                        help="Legacy fallback per-sample condition dropout probability for text/asset/camera.")
    parser.add_argument(
        "--text_uncond_drop_prob",
        type=float,
        default=None,
        help="Per-sample probability of replacing text with the null/empty prompt. Defaults to --uncond_drop_prob.",
    )
    parser.add_argument(
        "--asset_uncond_drop_prob",
        type=float,
        default=None,
        help="Per-sample probability of replacing asset condition with learned asset_null_condition_embed.",
    )
    parser.add_argument(
        "--camera_uncond_drop_prob",
        type=float,
        default=None,
        help="Per-sample probability of replacing camera condition with per-frame learned camera_null_condition_embed.",
    )
    parser.add_argument(
        "--all_cond_drop_prob",
        type=float,
        default=0.0,
        help="Extra per-sample probability of dropping text, asset, and camera conditions together.",
    )
    parser.add_argument("--val_guidance_scales", type=str, default="",
                        help="Comma-separated extra CFG scales to dump in validation RGB (in addition to --guidance_scale).")

    parser.add_argument("--weighting_scheme", type=str, default="waver")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--loss_weighting_scheme", type=str, default="none")
    parser.add_argument(
        "--prediction_type",
        type=str,
        choices=("v", "x"),
        default="x",
        help=(
            "SceneFlow model output parameterization. Default 'x' follows RAEv2 T2I "
            "by predicting the clean latent and converting it to RF velocity for loss/sampling."
        ),
    )
    parser.add_argument("--precision", type=str, default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ddp_timeout_minutes",
        type=int,
        default=60,
        help=(
            "Distributed process-group timeout. Validation image rendering is "
            "rank-0-only, so keep this above the worst-case qualitative dump time. "
            "Use <=0 to leave the PyTorch default."
        ),
    )
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dggt-flow")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Optional existing wandb run id for resume. Use with --wandb_resume.",
    )
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default="allow",
        choices=("allow", "must", "never", "auto"),
        help="wandb resume mode passed to wandb.init(..., resume=...).",
    )
    return parser


def dataloader_runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.num_workers) <= 0:
        return {}
    return {
        "prefetch_factor": max(1, int(args.prefetch_factor)),
        "persistent_workers": not bool(args.no_persistent_workers),
    }


def main() -> None:
    args = build_argparser().parse_args()
    args.patch_grid = (int(args.patch_grid_h), int(args.patch_grid_w))
    args.sky_grid = sky_grid_shape(args)
    args.sky_atlas_hw = (int(args.sky_atlas_h), int(args.sky_atlas_w))
    if args.patch_grid[0] <= 0 or args.patch_grid[1] <= 0:
        raise ValueError("--patch_grid_h and --patch_grid_w must be positive.")
    if int(args.sky_mask_refine_scale) <= 0 or int(args.sky_mask_refine_scale) & (int(args.sky_mask_refine_scale) - 1):
        raise ValueError("--sky_mask_refine_scale must be a positive power of two.")
    if int(args.sky_mask_refine_channels) <= 0:
        raise ValueError("--sky_mask_refine_channels must be positive.")
    if int(args.sky_mask_endpoint_start_step) < 0 or int(args.sky_mask_endpoint_every) < 0:
        raise ValueError("sky mask endpoint start/every must be non-negative.")
    if args.sky_atlas_hw != DEFAULT_SKY_ATLAS_HW or args.sky_grid != DEFAULT_SKY_GRID:
        raise ValueError(
            f"{SKY_REPRESENTATION_VERSION} requires sky_atlas_hw={DEFAULT_SKY_ATLAS_HW} "
            f"and sky_grid={DEFAULT_SKY_GRID}"
        )
    if args.sequence_length < 2:
        raise ValueError("--sequence_length must be >= 2.")
    if not 0.0 <= float(args.camera_anchor_context_dropout) <= 1.0:
        raise ValueError("--camera_anchor_context_dropout must be in [0, 1].")
    if int(args.pretrain_instance_cache_size) < 0:
        raise ValueError("--pretrain_instance_cache_size must be non-negative.")
    if float(args.lambda_rgb_render) < 0.0:
        raise ValueError("--lambda_rgb_render must be non-negative.")
    if int(args.rgb_render_every) < 0:
        raise ValueError("--rgb_render_every must be >= 0.")
    if int(args.rgb_render_max_samples) < 0 or int(args.rgb_render_max_frames) < 0:
        raise ValueError("--rgb_render_max_samples and --rgb_render_max_frames must be non-negative; 0 means all.")
    if int(args.rgb_render_stride) <= 0:
        raise ValueError("--rgb_render_stride must be positive.")
    if int(args.rgb_render_start_step) < 0 or int(args.rgb_render_warmup_steps) < 0:
        raise ValueError("RGB render start/warmup steps must be non-negative.")
    if not 0.0 <= float(args.rgb_render_sky_weight) <= 1.0:
        raise ValueError("--rgb_render_sky_weight must be in [0,1].")
    if float(args.rgb_render_camera_grad_scale) < 0.0:
        raise ValueError("--rgb_render_camera_grad_scale must be non-negative.")
    if float(args.rgb_render_sky_mask_grad_scale) < 0.0:
        raise ValueError("--rgb_render_sky_mask_grad_scale must be non-negative.")
    if args.val_image_dir is None:
        args.val_image_dir = args.image_dir
    if args.val_scene_start is None:
        args.val_scene_start = 0 if args.val_image_dir != args.image_dir else args.scene_end
    if args.val_scene_end is None:
        args.val_scene_end = args.val_scene_start
    if args.val_caption_root is None:
        caption_root_path = Path(args.caption_root)
        if caption_root_path.name == "training_captions":
            candidate = caption_root_path.with_name("validation_captions")
            if candidate.exists():
                args.val_caption_root = str(candidate)
    if args.val_caption_root is None:
        args.val_caption_root = args.caption_root
    legacy_drop_prob = float(args.uncond_drop_prob)
    if args.text_uncond_drop_prob is None:
        args.text_uncond_drop_prob = legacy_drop_prob
    if args.asset_uncond_drop_prob is None:
        args.asset_uncond_drop_prob = legacy_drop_prob
    if args.camera_uncond_drop_prob is None:
        args.camera_uncond_drop_prob = legacy_drop_prob
    for name in (
        "uncond_drop_prob",
        "text_uncond_drop_prob",
        "asset_uncond_drop_prob",
        "camera_uncond_drop_prob",
        "all_cond_drop_prob",
    ):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}.")
    if args.resume_path and args.warm_start_path:
        raise ValueError(
            "--resume_path and --warm_start_path are mutually exclusive. "
            "Use --resume_path for exact training resume, or --warm_start_path "
            "to initialize weights and start a fresh optimizer/scheduler run."
        )

    device, local_rank, world_size = setup_distributed(args)
    if int(args.num_workers) > 0:
        torch.multiprocessing.set_sharing_strategy(str(args.mp_sharing_strategy))
    seed_everything(args.seed + get_rank())

    log_dir = Path(args.log_dir)
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        config = dict(vars(args))
        config["patch_grid"] = list(args.patch_grid)
        (log_dir / "config.json").write_text(json.dumps(config, indent=2))
    wandb_run = init_wandb(args, log_dir)

    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path,
        args.tokenizer_ckpt_path,
        device,
    )
    lpips_model = setup_lpips_for_rgb_loss(args, device)

    # Legacy config compatibility: in_channels records the old packed vector
    # [z_t, z_splat, scaffold_tok, M_preserve, M_source, M_dest]. The RAEv2 DDT
    # visual embedders consume z_t only; edit context enters as condition tokens.
    sf_in_channels = 3 * int(args.latent_dim) + 3
    camera_gen_dim = CAMERA_GENERATION_DIM
    scene_flow = WanSceneFlow.from_scene_config(
        bring_up=False,
        patch_grid=args.patch_grid,
        in_channels=sf_in_channels,
        out_channels=int(args.latent_dim),
        camera_gen_dim=camera_gen_dim,
        camera_generation_representation=CAMERA_GENERATION_REPRESENTATION,
        asset_position_mode=str(args.asset_position_mode),
        sky_token_dim=SKY_TOKEN_DIM,
        sky_grid=args.sky_grid,
        max_sky_tokens=int(args.sky_grid[0] * args.sky_grid[1]),
        sky_mask_refine_scale=int(args.sky_mask_refine_scale),
        sky_mask_refine_channels=int(args.sky_mask_refine_channels),
        rope_max_position=int(args.rope_max_position),
        sky_representation_version=str(args.sky_representation_version),
        sky_atlas_hw=args.sky_atlas_hw,
        prediction_type=args.prediction_type,
    ).to(device)
    scene_flow.enable_gradient_checkpointing()
    args.dggt_checkpoint_sha256 = load_all_stats_into_buffers(
        scene_flow,
        args.feature_stats_path,
        token_dim=int(args.latent_dim),
        dggt_ckpt_path=args.dggt_ckpt_path,
    )
    text_encoder = setup_text_encoder(args, device)

    scene_names = discover_scene_names(args.image_dir, args.scene_start, args.scene_end)
    dataset = WaymoOpenDataset(
        image_dir=args.image_dir,
        scene_names=scene_names,
        sequence_length=args.sequence_length,
        mode=1,
        views=1,
        caption_root=args.caption_root,
        pretrain_patch_grid=args.patch_grid,
        pretrain_instance_cache_size=args.pretrain_instance_cache_size,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        drop_last=True,
        **dataloader_runtime_kwargs(args),
    )
    val_loader = None
    if args.val_every > 0 and args.val_batches > 0 and args.val_scene_end > args.val_scene_start:
        val_scene_names = discover_scene_names(args.val_image_dir, args.val_scene_start, args.val_scene_end)
        val_dataset = WaymoOpenDataset(
            image_dir=args.val_image_dir,
            scene_names=val_scene_names,
            sequence_length=args.sequence_length,
            start_idx=0,
            mode=1,
            views=1,
            caption_root=args.val_caption_root,
            pretrain_patch_grid=args.patch_grid,
            pretrain_instance_cache_size=args.pretrain_instance_cache_size,
            trunk_major_samples=True,
            trunk_frames=29,
        )
        val_sampler = CyclicSequentialSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            drop_last=False,
            **dataloader_runtime_kwargs(args),
        )
        if is_main_process():
            print(
                f"[validation] scenes={len(val_scene_names)} batches_per_eval={args.val_batches}",
                flush=True,
            )

    decay_params, no_decay_params = split_param_groups(scene_flow)
    optimizer, optimizer_msg = build_rae_optimizer(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.gmuon_momentum,
        nesterov=args.gmuon_nesterov,
        ns_coefficients_preset=args.gmuon_ns_coefficients_preset,
        ns_use_kernels=args.gmuon_ns_use_kernels,
    )
    lr_scheduler = build_training_scheduler(optimizer, args)
    if is_main_process():
        decay_end_steps = int(args.decay_end_steps) if int(args.decay_end_steps) > 0 else int(args.max_steps)
        print(
            f"[optim] {optimizer_msg}; scheduler={args.scheduler_type} "
            f"warmup={args.warmup_steps} decay_end={decay_end_steps} "
            f"lr={args.lr}->{args.final_lr}",
            flush=True,
        )
    flow_scheduler = None
    ema = EMAModel(scene_flow.parameters(), decay=args.ema_decay)
    ema.to(device)
    global_step = load_resume_checkpoint(
        scene_flow,
        ema,
        optimizer,
        lr_scheduler,
        args.resume_path,
        device,
        expected_dggt_sha256=args.dggt_checkpoint_sha256,
    )
    # 如果指定了 warm_start_path，则覆盖模型和 EMA 的权重，但保留 global_step = 0
    if args.warm_start_path:
        payload = torch.load(args.warm_start_path, map_location=device)
        if args.partial_warm_start_non_sky:
            if isinstance(payload, dict) and isinstance(payload.get("ema_scene_flow_state_dict"), dict):
                source_state = payload["ema_scene_flow_state_dict"]
                warm_source = "ema_scene_flow_state_dict:partial_non_sky"
            elif isinstance(payload, dict) and isinstance(payload.get("scene_flow"), dict):
                source_state = payload["scene_flow"]
                warm_source = "scene_flow:partial_non_sky"
            elif isinstance(payload, dict):
                source_state = payload
                warm_source = "state_dict:partial_non_sky"
            else:
                raise ValueError("partial warm-start requires a checkpoint state dict")
            source_state = strip_module_prefix(source_state)
            target_model = unwrap_ddp(scene_flow)
            current_state = target_model.state_dict()
            transferred = {
                key: value
                for key, value in source_state.items()
                if key in current_state
                and torch.is_tensor(value)
                and tuple(value.shape) == tuple(current_state[key].shape)
                and not key.startswith(("sky_gen_",))
            }
            current_state.update(transferred)
            target_model.load_state_dict(current_state, strict=True)
            sync_ema_shadow_from_model(scene_flow, ema)
            warm_source += f":loaded={len(transferred)}:skipped={len(source_state) - len(transferred)}"
            ema.optimization_step = 0
            if is_main_process():
                print(f"[warm start] loaded {warm_source}; optimizer and EMA history were not restored", flush=True)
            payload = None
        if payload is None:
            pass
        else:
            saved_camera_rep, saved_camera_dim = _checkpoint_camera_representation(payload)
        legacy_camera_migration = False if payload is None else saved_camera_rep == "dggt_hidden_v1" and saved_camera_dim == 2048
        if payload is None:
            pass
        elif legacy_camera_migration:
            validate_prediction_type_checkpoint(scene_flow, payload, args.warm_start_path)
            loaded_count, skipped_count, warm_source = migrate_legacy_camera_checkpoint(scene_flow, payload)
            sync_ema_shadow_from_model(scene_flow, ema)
            warm_source = f"{warm_source}:shared={loaded_count}:skipped={skipped_count}"
        elif payload is not None:
            validate_scene_flow_checkpoint_config(
                scene_flow,
                payload,
                args.warm_start_path,
                expected_dggt_sha256=args.dggt_checkpoint_sha256,
            )

        if payload is None:
            pass
        elif legacy_camera_migration:
            pass
        elif isinstance(payload, dict) and "ema_scene_flow_state_dict" in payload:
            unwrap_ddp(scene_flow).load_state_dict(payload["ema_scene_flow_state_dict"], strict=True)
            if "ema_scene_flow" in payload:
                load_warm_start_ema_or_sync(scene_flow, ema, payload["ema_scene_flow"])
            else:
                sync_ema_shadow_from_model(scene_flow, ema)
            warm_source = "ema_scene_flow_state_dict"
        elif isinstance(payload, dict) and payload.get("is_ema_weights") and "scene_flow" in payload:
            unwrap_ddp(scene_flow).load_state_dict(payload["scene_flow"], strict=True)
            sync_ema_shadow_from_model(scene_flow, ema)
            warm_source = "ema_weights_only"
        elif isinstance(payload, dict) and "ema_scene_flow" in payload:
            if "scene_flow" not in payload:
                raise ValueError(f"{args.warm_start_path} has ema_scene_flow but no scene_flow weights.")
            unwrap_ddp(scene_flow).load_state_dict(payload["scene_flow"], strict=True)
            if load_warm_start_ema_or_sync(scene_flow, ema, payload["ema_scene_flow"]):
                ema.copy_to(unwrap_ddp(scene_flow).parameters())
                warm_source = "ema_scene_flow"
            else:
                warm_source = "scene_flow"
        else:
            state_dict = payload.get("scene_flow", payload) if isinstance(payload, dict) else payload
            unwrap_ddp(scene_flow).load_state_dict(state_dict, strict=True)
            sync_ema_shadow_from_model(scene_flow, ema)
            warm_source = "scene_flow"
        ema.optimization_step = 0

        if is_main_process() and payload is not None:
            print(
                f"[warm start] 成功从 {args.warm_start_path} 加载 {warm_source}，将从 step 0 开始全新训练",
                flush=True,
            )

    if world_size > 1:
        scene_flow = DistributedDataParallel(
            scene_flow,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )

    scene_flow.train()
    optimizer.zero_grad(set_to_none=True)
    accum_step = 0
    # Rolling sums for wandb so we report the mean over the last
    # `--wandb_log_every` optimizer steps instead of every individual step.
    wandb_sums: dict[str, float] = {}
    wandb_count = 0
    progress = None
    if is_main_process() and not args.no_tqdm:
        progress = tqdm(
            total=args.max_steps,
            initial=global_step,
            desc="pretrain",
            dynamic_ncols=True,
        )
    try:
        while global_step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(global_step)
            for batch in loader:
                if global_step >= args.max_steps:
                    break

                sync_grad = (accum_step + 1) % max(1, args.grad_accum_steps) == 0
                ddp_context = (
                    scene_flow.no_sync()
                    if isinstance(scene_flow, DistributedDataParallel) and not sync_grad
                    else nullcontext()
                )
                with ddp_context:
                    try:
                        loss, logs = train_step(
                            batch,
                            vggt_model,
                            scene_flow,
                            flow_scheduler,
                            device,
                            args,
                            text_encoder,
                            global_step=global_step,
                            lpips_model=lpips_model,
                        )
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        # In DDP, single-rank skip would desync allreduce. Re-raise so
                        # the entire job restarts with smaller batch / accum_steps.
                        if is_distributed():
                            raise
                        print(f"[step {global_step:06d}] CUDA OOM; skipping batch", flush=True)
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        accum_step = 0
                        continue
                    (loss / max(1, args.grad_accum_steps)).backward()
                accum_step += 1

                if not sync_grad:
                    continue

                params = unwrap_ddp(scene_flow).parameters()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.step(unwrap_ddp(scene_flow).parameters())
                accum_step = 0
                global_step += 1

                if is_main_process():
                    lr_now = optimizer.param_groups[0]["lr"]
                    train_metrics = dict(logs)
                    train_metrics["lr"] = float(lr_now)
                    if progress is not None:
                        postfix = {"lr": f"{lr_now:.2e}"}
                        for key, value in logs.items():
                            postfix[key] = f"{float(value):.4f}"
                        progress.set_postfix(postfix, refresh=False)
                    elif global_step % max(1, int(args.log_every)) == 0:
                        metrics_str = " | ".join(f"{key}={value:.4f}" for key, value in logs.items())
                        print(f"[step {global_step:06d}] lr={lr_now:.2e} | {metrics_str}", flush=True)

                    # Accumulate for averaged wandb reporting.
                    for key, value in train_metrics.items():
                        wandb_sums[key] = wandb_sums.get(key, 0.0) + float(value)
                    wandb_count += 1
                    if wandb_run is not None and wandb_count >= max(1, int(args.wandb_log_every)):
                        averaged = {key: value / wandb_count for key, value in wandb_sums.items()}
                        log_wandb(wandb_run, averaged, global_step, "train")
                        wandb_sums = {}
                        wandb_count = 0

                if (
                    val_loader is not None
                    and global_step > 0
                    and global_step % args.val_every == 0
                ):
                    # Continue through trunk-major validation samples instead
                    # of restarting from scene 000 at every validation call.
                    validation_index = global_step // args.val_every - 1
                    val_loader.sampler.set_offset(
                        validation_index * args.val_batches * args.batch_size
                    )
                    run_validation(
                        val_loader,
                        vggt_model,
                        scene_flow,
                        flow_scheduler,
                        device,
                        args,
                        global_step,
                        log_dir,
                        wandb_run,
                        ema,
                        text_encoder,
                    )

                if global_step > 0 and global_step % args.save_every == 0:
                    if is_distributed():
                        dist.barrier()
                    if is_main_process():
                        save_checkpoint(scene_flow, ema, optimizer, lr_scheduler, global_step, log_dir, args)
                    if is_distributed():
                        dist.barrier()

                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if is_distributed():
        dist.barrier()
    if is_main_process():
        save_checkpoint(scene_flow, ema, optimizer, lr_scheduler, global_step, log_dir, args)
        if wandb_run is not None:
            wandb_run.finish()
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
