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
      --tokenizer_ckpt_path logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
      --feature_stats_path logs/scene_flow_pretrain_1024/feature_stats_pretrain_v5.pt \
      --scene_gauge_path data/scene_gauge/training.json \
      --val_scene_gauge_path data/scene_gauge/validation.json \
      --pullback_calibration_path data/scene_gauge/pullback_d63b34f7.json \
      --log_dir logs/scene_flow_pretrain_tokenizer_v2
"""
from __future__ import annotations

import argparse
from functools import partial
from itertools import islice
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm.auto import tqdm

from datasets.dataset import (
    WaymoOpenDataset,
    load_metric_depth_diagnostic_paths,
)
from dggt.losses.flow_losses import (
    boundary_mask_from_edit_mask,
    build_masked_rectified_flow_target,
    compute_total_loss,
    rae_t_grid,
)
from dggt.losses.reconstruction_feedback_loss import DYNAMIC_HEAD_SPACES
from dggt.losses.rgb_render_loss import (
    compute_rgb_render_loss,
    decode_generated_dggt_geometry,
    rgb_render_loss_enabled,
    rgb_render_loss_ramp,
    rgb_render_sigma_weight,
    setup_lpips_for_rgb_loss,
    should_apply_rgb_render_loss,
    should_apply_sky_view_loss,
    sky_view_loss_ramp,
)
from dggt.models.scene_flow import WanSceneFlow
from dggt.models.canonical_asset_encoder import CanonicalAssetEncoder
from dggt.models.embedders.text_encoder import TextEncoder
from dggt.models.vggt import VGGT
from dggt.utils.feature_stats import (
    DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH,
    load_all_stats_into_buffers,
)
from dggt.utils.gaussian_render import composite_gsplat_rgb, composite_original_sky
from dggt.utils.camera_condition import (
    CAMERA_CONDITION_REPRESENTATION,
    camera_condition_from_waymo_request,
)
from dggt.utils.actor_geometry_condition import (
    ActorGeometryCondition,
    ProjectedActorGeometry,
)
from dggt.utils.appearance_binding_condition import (
    APPEARANCE_TOKEN_DIM,
    AppearanceBindingCondition,
    AppearanceMode,
)
from dggt.utils.layout_condition import (
    LAYOUT_CONDITION_VERSION,
    LAYOUT_TASK_PROBABILITIES,
    LayoutConditionBatch,
    LayoutTask,
    combine_chained_cfg,
    layout_to_gauge_scale,
    required_cfg_branches,
    sample_layout_tasks,
)
from datasets.tools.hdmap_schema import RASTER_SCHEMA_HASH
from dggt.utils.layout_raster import STATIC_FAR_PLANE_M
from dggt.utils.scene_gauge import (
    SCENE_GAUGE_DIM,
    SCENE_GAUGE_REPRESENTATION,
    SCENE_GAUGE_STATS_VERSION,
    PullbackCalibration,
    apply_depth_pullback_calibration,
    assemble_dggt_pose_encoding,
    dggt_pose_encoding_to_camera_to_world,
    metric_c2w_to_teacher_anchor_dggt,
    load_pullback_calibration,
)
from dggt.utils.validation_mosaic import (
    GT_ROW_ORDER,
    MOSAIC_CELL_WIDTH_DEFAULT,
    build_validation_mosaics,
    encode_mosaic_row,
    mosaic_group,
)
from dggt.utils.flow_schedule import (
    build_flow_schedule_config,
    validate_checkpoint_flow_schedule,
)
from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import concat_list, get_split_gs
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from dggt.utils.rae_optim import build_rae_optimizer, build_rae_scheduler
from dggt.utils.sliding_window import (
    OFFLINE_MAX_SINGLE_WINDOW,
    cosine_coverage,
    cosine_window,
    default_window_stride,
    scene_global_window_weight,
    window_slices,
)
from dggt.utils.tokenizer_window import (
    decode_tokenizer_windowed,
    encode_tokenizer_windowed,
)
from dggt.utils.gaussian_time import gaussian_timestamps_from_frame_ids
from dggt.utils.tokens import (
    batched_gather_frames,
    select_patch_pyramid,
    split_special_and_patch,
)
from dggt.utils.tokenizer_checkpoint import load_scene_tokenizer_state_dict_strict
from dggt.utils.validation_rng import make_validation_generator, preserve_validation_rng_state

from diffusers.training_utils import EMAModel


TOKENIZER_LEVELS = (4, 11, 17, 23)


def _tokenizer_window_len(scene_flow: nn.Module, args: argparse.Namespace) -> int:
    calibration = getattr(unwrap_ddp(scene_flow), "_pullback_calibration", None)
    value = (
        int(calibration.window_len)
        if isinstance(calibration, PullbackCalibration)
        else min(
            int(getattr(args, "sequence_length", OFFLINE_MAX_SINGLE_WINDOW)),
            OFFLINE_MAX_SINGLE_WINDOW,
        )
    )
    if value <= 0 or value > OFFLINE_MAX_SINGLE_WINDOW:
        raise ValueError(
            "Tokenizer window must be positive and no larger than the offline "
            f"calibration ceiling {OFFLINE_MAX_SINGLE_WINDOW}, got {value}."
        )
    return value
SKY_CLASS_INDEX = 9
SKY_RGB_DIM = 3
SKY_REPRESENTATION_VERSION = "rgb_patch_teacher_anchor_v5"
# The atlas is an upper-hemisphere environment map: ``_sky_direction_grid``
# spans elevation 0..90 deg over its rows and azimuth 0..360 deg over its
# columns, so a cell is (90/H) x (360/W) degrees.
#
# At the previous 32x64 a cell was 2.81 x 5.62 deg.  The Waymo front camera is
# 49.2 x 35.0 deg, so the whole frame landed on 12 x 9 cells and the *sky part*
# of a typical frame -- 20% of the height, 7 deg -- landed on about 22.  That
# number is confirmed by the run itself: v6 logged
# ``sky_token_loss_weight_mean = 0.0601`` which, against
# ``sky_unobserved_loss_weight``, inverts to 1.06% of 2048 cells = 21.8.
# Twenty-two colours cannot draw a cloud; handed the ground-truth sky, the best
# a 32x64 atlas can reconstruct is L1 0.066 on a clear-sky frame, and the flat
# blue wash that comes out of it is what the run produced at 50k steps.
#
# 128x256 puts a cell at 0.70 x 1.41 deg and the frame on 50 x 35 cells, taking
# best-case L1 to 0.028.  It stays at 512 tokens by packing 8x8 atlas patches
# instead of 2x2, so the sequence the trunk attends over is unchanged and only
# the two sky Linear layers get wider (12 -> 192 channels).
DEFAULT_SKY_ATLAS_HW = (128, 256)
DEFAULT_SKY_GRID = (16, 32)
SKY_PATCH_SIZE = 8
SKY_TOKEN_DIM = SKY_RGB_DIM * SKY_PATCH_SIZE * SKY_PATCH_SIZE
DEFAULT_SKY_VALID_THRESHOLD = 0.5
# The scene latent the generator lives in is standardized -- the run logs
# ``sample_latent_target_std`` at 1.0011 -- but the sky token was packed as raw
# ``rgb * 2 - 1`` and was not.  Measured over observed atlas cells it carries
# mean +0.55 and std 0.42, and per channel blue is the worst: nearly saturated
# with the *smallest* spread, which is exactly where cloud contrast lives.
#
# Flow matching adds ``eps ~ N(0, I)`` at unit scale, so at the training sigma
# (waver, shift 10, mean 0.864) the signal-to-noise of what the model has to
# recover was
#
#     scene latent  std 1.00  ->  1 : 6
#     sky token     std 0.42  ->  1 : 15
#     blue channel  std 0.26  ->  1 : 24
#     within-token  std 0.15  ->  1 : 42
#
# A flat blue sky is the correct answer to that problem: the mean is nearly
# free and the cloud is the lowest-variance direction in the whole target.
# Standardizing per channel puts the sky on the same footing as the scene, and
# it is a per-dimension affine so ``pack_sky_atlas_loss_weight`` -- which is
# per atlas cell -- keeps applying unchanged.
#
# Constants are frozen from ``datasets/tools/compute_sky_token_stats.py`` over
# observed cells only; the spherical completion covers the rest of the sphere
# with a wider spread that must not set the scale for the sky that renders.
SKY_TOKEN_CHANNEL_MEAN = (0.205424, 0.454743, 0.745609)
SKY_TOKEN_CHANNEL_STD = (0.466658, 0.389231, 0.321221)
# 0.005, not 0.05. Unobserved atlas directions carry a deterministic spherical
# completion prior, not a measurement, so they must remain low-confidence.
# ``sky_flow_loss`` is a weighted mean and a region's gradient share is exactly
# its weight share. At 0.05 with ~1% of the atlas observed, the completion prior
# took 82.4% of the sky-flow gradient, leaving 17.6% for the sky that actually
# renders. At 0.005 that flips to 68% observed.
DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT = 0.005
# ``sky_flow_loss`` normalizes the observed and unobserved atlas separately
# instead of sharing one ``sum(w*e)/sum(w)`` denominator.  A single weighted
# mean makes the observed sky's share of the gradient depend on how much sky
# the clip happens to contain -- measured across Waymo clips it swings from
# 42% on an 11%-sky frame to 78% on a 32%-sky frame -- so the sky term changed
# meaning from clip to clip.  Separate means pin it at ``1 / (1 + beta)`` for
# every clip.  beta=0.05 leaves the spherical completion prior 4.8% of the sky
# gradient: enough to keep the unseen hemisphere plausible for a requested
# trajectory that turns, far less than the 36% it used to take.
DEFAULT_SKY_UNOBSERVED_LOSS_BETA = 0.05
T59_VALIDATION_SAMPLE_STEPS = 50


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


@torch.no_grad()
def synchronize_validation_model(scene_flow: nn.Module) -> None:
    """Make rank 0 the source of truth before rank-local validation sampling.

    DDP keeps the trainable weights synchronized, and every rank maintains its
    own EMA shadow.  Parallel CFG sampling must not merely rely on those two
    facts continuing to hold, though: explicitly broadcasting both parameters
    and buffers after the EMA swap guarantees that every sampling rank starts
    from the exact same SceneFlow state.
    """
    if not is_distributed():
        return

    model = unwrap_ddp(scene_flow)
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        if tensor.is_contiguous():
            dist.broadcast(tensor.detach(), src=0)
            continue
        contiguous = tensor.detach().contiguous()
        dist.broadcast(contiguous, src=0)
        tensor.copy_(contiguous)


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
        missing_aggregator = [
            key for key in real_missing if key.startswith("aggregator.")
        ]
        if missing_aggregator:
            raise RuntimeError(
                "DGGT checkpoint is missing aggregator weights required by "
                "metric-gauge training: "
                f"{format_key_examples(missing_aggregator)}"
            )

    if tokenizer_ckpt_path:
        tok_checkpoint = torch.load(tokenizer_ckpt_path, map_location="cpu")
        load_scene_tokenizer_state_dict_strict(
            model.scene_tokenizer,
            tok_checkpoint,
            source=tokenizer_ckpt_path,
        )
        if is_main_process():
            print(f"[ckpt:tokenizer] loaded complete state from {tokenizer_ckpt_path}", flush=True)
    else:
        # ``VGGT()`` constructs a random tokenizer.  The base DGGT checkpoint
        # may legitimately predate JointSceneTokenizer, so a non-strict full
        # model load above is not proof that tokenizer weights were restored.
        # Strictly validate the embedded state before allowing this fallback.
        load_scene_tokenizer_state_dict_strict(
            model.scene_tokenizer,
            checkpoint,
            source=dggt_ckpt_path,
        )
        if is_main_process():
            print(f"[ckpt:tokenizer] loaded embedded complete state from {dggt_ckpt_path}", flush=True)

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


class SpreadSequentialSampler(Sampler[int]):
    """A fixed cover spread across a dataset, identical at every validation.

    The scalar validation block exists to be *compared across steps*, so its
    samples must not move.  ``CyclicSequentialSampler`` moved them: the trunk-
    major index is ordered trunk -> scene -> window offset, so advancing by the
    number of samples consumed (``val_batches * batch_size``) walks the
    innermost axis first.  At ``--val_batches 1`` that is one window offset per
    validation, and with the four pretraining offsets ``(0, 7, 14, 19)`` the
    whole ``validation/*`` scalar block spent its first four validations inside
    scene 000 trunk 0 and would have needed 800k steps -- four times the
    planned run -- to see the 100 configured validation scenes.  Two symptoms
    followed: the per-scene gauge diagnostics were bit-identical because
    ``log_metric_scale`` is a per-scene/trunk constant, and ``validation/loss``
    was about to step discontinuously every fourth validation as the cover
    crossed into the next scene.

    Spreading a fixed set across the split fixes both: the same windows are
    scored at every step, and they come from different scenes and trunks
    instead of from overlapping frames of one clip.

    Positions come from the golden-ratio low-discrepancy sequence rather than
    from even spacing, because this index aliases badly: its length factorizes
    as ``trunks * scenes * offsets`` with the offset innermost, so any stride
    that shares a factor with the scene axis revisits the same scenes.  Evenly
    spacing eight samples over 100 scenes and four offsets covers four scenes,
    each twice; nudging the stride to be coprime with the length fixes eight
    but still collapses four samples onto two scenes.  The golden ratio is
    irrational, so it aliases with no period: measured on the real shape it
    yields ``count`` distinct scenes at every count from 2 to 24.
    """

    _GOLDEN_RATIO_CONJUGATE = 0.6180339887498949

    def __init__(self, data_source, count: int) -> None:
        length = len(data_source)
        if int(count) <= 0:
            raise ValueError(f"spread sampler count must be positive, got {count}")
        self.data_source = data_source
        self.count = int(count)
        if length <= 0:
            self._indices: tuple[int, ...] = ()
        elif self.count >= length:
            self._indices = tuple(range(length))
        else:
            picked: list[int] = []
            seen: set[int] = set()
            step = 0
            # The sequence only repeats an index when two turns land in the
            # same bucket, which is rare; the bound just keeps this finite.
            while len(picked) < self.count and step < 100 * self.count:
                index = int(((step * self._GOLDEN_RATIO_CONJUGATE) % 1.0) * length)
                if index not in seen:
                    seen.add(index)
                    picked.append(index)
                step += 1
            self._indices = tuple(picked)

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class ContinuousDistributedBatchSampler(Sampler[list[int]]):
    """Concatenate distributed shuffle epochs without draining worker queues.

    ``DistributedSampler`` is intentionally retained as the source of truth for
    shuffle, padding and per-rank slicing.  The only behavioral change is that
    its finite epoch batches are yielded through one continuous iterator, so a
    DataLoader can prefetch the first batch of the next logical epoch while the
    accelerator is still consuming the current one.

    The epoch argument also follows the historical training loop: it was the
    optimizer step at the start of each finite DataLoader pass, rather than a
    separate ``0, 1, 2, ...`` counter.  Gradient-accumulation phase is therefore
    tracked here when advancing to the next logical epoch.
    """

    def __init__(
        self,
        data_source: Any,
        *,
        batch_size: int,
        grad_accum_steps: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.grad_accum_steps = int(grad_accum_steps)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {grad_accum_steps}"
            )
        self.distributed_sampler = DistributedSampler(
            data_source,
            num_replicas=int(num_replicas),
            rank=int(rank),
            shuffle=True,
            seed=int(seed),
            drop_last=False,
        )
        self.usable_samples_per_epoch = (
            int(self.distributed_sampler.num_samples) // self.batch_size
        ) * self.batch_size
        if self.usable_samples_per_epoch <= 0:
            raise ValueError(
                "distributed rank has fewer samples than one local batch: "
                f"samples={self.distributed_sampler.num_samples}, "
                f"batch_size={self.batch_size}"
            )
        self.batches_per_logical_epoch = (
            self.usable_samples_per_epoch // self.batch_size
        )
        self._start_optimizer_step = 0
        self._iteration_started = False

    def set_start_optimizer_step(self, step: int) -> None:
        """Set the first shuffle epoch before the DataLoader iterator starts."""

        if self._iteration_started:
            raise RuntimeError(
                "continuous sampler start step cannot change after iteration begins"
            )
        if int(step) < 0:
            raise ValueError(f"optimizer step must be non-negative, got {step}")
        self._start_optimizer_step = int(step)

    def __iter__(self):
        if self._iteration_started:
            raise RuntimeError(
                "continuous training DataLoader iterator must be created only once"
            )
        self._iteration_started = True
        epoch_optimizer_step = int(self._start_optimizer_step)
        accumulation_phase = 0
        while True:
            self.distributed_sampler.set_epoch(epoch_optimizer_step)
            indices = list(iter(self.distributed_sampler))
            indices = indices[: self.usable_samples_per_epoch]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

            accumulated_micro_batches = (
                accumulation_phase + self.batches_per_logical_epoch
            )
            epoch_optimizer_step += (
                accumulated_micro_batches // self.grad_accum_steps
            )
            accumulation_phase = (
                accumulated_micro_batches % self.grad_accum_steps
            )

    def __len__(self) -> int:
        """Return one logical epoch length for diagnostics only."""

        return self.batches_per_logical_epoch


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


def validate_scene_flow_state_dict_exact(
    scene_flow: nn.Module,
    state_dict: Any,
    *,
    path: str | Path,
    source: str,
) -> dict[str, torch.Tensor]:
    """Require an exact current-architecture state before strict loading."""

    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} {source} is not a state_dict")
    stripped = strip_module_prefix(state_dict)
    expected = unwrap_ddp(scene_flow).state_dict()
    actual_keys = set(stripped)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"{path} {source} state-key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    invalid_values = sorted(
        key for key, value in stripped.items() if not torch.is_tensor(value)
    )
    if invalid_values:
        raise ValueError(f"{path} {source} contains non-tensor values at {invalid_values}")
    shape_mismatches = sorted(
        key
        for key, value in stripped.items()
        if tuple(value.shape) != tuple(expected[key].shape)
    )
    if shape_mismatches:
        raise ValueError(f"{path} {source} state-shape mismatch at {shape_mismatches}")
    return stripped


def load_scene_flow_state_dict_strict(
    scene_flow: nn.Module,
    state_dict: Any,
    *,
    path: str | Path,
    source: str,
) -> None:
    prepared = validate_scene_flow_state_dict_exact(
        scene_flow,
        state_dict,
        path=path,
        source=source,
    )
    unwrap_ddp(scene_flow).load_state_dict(dict(prepared), strict=True)


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_config_value(item) for item in value)
    return value


def validate_scene_flow_checkpoint_config(
    scene_flow: nn.Module,
    payload: Any,
    path: str | Path,
) -> None:
    """Validate an exact layout-v2 training resume; no migration is supported."""

    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a versioned SceneFlow checkpoint")
    saved_cfg = payload.get("scene_flow_config")
    current_cfg_obj = getattr(unwrap_ddp(scene_flow), "config", None)
    current_cfg = (
        current_cfg_obj.to_dict()
        if current_cfg_obj is not None and hasattr(current_cfg_obj, "to_dict")
        else None
    )
    if not isinstance(saved_cfg, dict) or not isinstance(current_cfg, dict):
        raise ValueError(f"{path} is missing the exact SceneFlow config")
    if _normalize_config_value(saved_cfg) != _normalize_config_value(current_cfg):
        raise ValueError(f"{path} SceneFlow config is not an exact match for this run")
    validate_scene_flow_state_dict_exact(
        scene_flow,
        payload.get("scene_flow"),
        path=path,
        source="scene_flow",
    )


def model_prediction_to_velocity(scene_flow: nn.Module, prediction: torch.Tensor, target) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        # Match RAEv2 Transport.convert_model_pred: x-pred is converted to
        # velocity with the same t_eps clamp used to build the RF target.
        return (target.z_t - prediction) / target.sigmas4.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).clamp_min(float(getattr(target, "t_eps", scene_flow_t_eps(scene_flow))))
    return prediction


def scene_flow_t_eps(scene_flow: nn.Module) -> float:
    cfg = getattr(unwrap_ddp(scene_flow), "config", None)
    return float(getattr(cfg, "t_eps", 0.05))


# Sky shares the RAE-compatible x-prediction conversion floor with video. A
# smaller sky-only floor makes a clean-prediction error contribute
# as 1 / sigma**2 near the clean endpoint and can catastrophically amplify the
# loss.  Keep this fallback aligned with RAEVideoSceneFlowConfig.t_eps; actual
# targets and samplers use the value carried by the target/model config.
SKY_FLOW_T_EPS = 0.05


def model_prediction_to_clean(scene_flow: nn.Module, prediction: torch.Tensor, target) -> torch.Tensor:
    if scene_flow_prediction_type(scene_flow) == "x":
        return prediction
    # RAEv2 trains velocity against (z_t - z_clean) / max(sigma, t_eps), so
    # recovering the clean endpoint must invert that same clamped denominator.
    sigma_safe = target.sigmas4.to(device=prediction.device, dtype=prediction.dtype).clamp_min(
        float(getattr(target, "t_eps", scene_flow_t_eps(scene_flow)))
    )
    return target.z_t - sigma_safe * prediction


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


def _init_pretrain_gauge_noise(
    scene_flow: nn.Module,
    bundle,
    generator: torch.Generator,
    *,
    return_gauge: bool,
) -> torch.Tensor | None:
    gauge_clean = getattr(bundle, "scene_gauge_clean_n", None)
    if torch.is_tensor(gauge_clean):
        gauge_z = torch.empty_like(gauge_clean)
        gauge_z.normal_(generator=generator)
        return gauge_z
    if not return_gauge:
        return None
    z_template = getattr(bundle, "z_clean_n", None)
    if not torch.is_tensor(z_template) or z_template.ndim < 2:
        raise RuntimeError("Pretrain gauge sampling requires bundle.z_clean_n")
    gauge_dim = int(getattr(getattr(unwrap_ddp(scene_flow), "config", None), "gauge_gen_dim", 0))
    if gauge_dim != SCENE_GAUGE_DIM:
        raise RuntimeError(f"SceneFlow gauge_gen_dim must be {SCENE_GAUGE_DIM}, got {gauge_dim}")
    gauge_z = z_template.new_empty((int(z_template.shape[0]), 1, gauge_dim))
    gauge_z.normal_(generator=generator)
    return gauge_z


def build_gauge_rectified_flow_target(
    gauge_clean: torch.Tensor | None,
    video_target,
    *,
    generator: torch.Generator | None = None,
) -> SimpleNamespace | None:
    """Noise one scene-global gauge token with the video's rectified-flow time."""

    if gauge_clean is None:
        return None
    if gauge_clean.ndim != 3 or tuple(gauge_clean.shape[1:]) != (1, SCENE_GAUGE_DIM):
        raise ValueError(
            f"gauge_clean must be [B,1,{SCENE_GAUGE_DIM}], got {tuple(gauge_clean.shape)}"
        )
    b = int(gauge_clean.shape[0])
    if video_target.sigmas.shape != (b,):
        raise ValueError(f"video_target sigmas shape {tuple(video_target.sigmas.shape)} != {(b,)}")
    sigmas = video_target.sigmas.to(device=gauge_clean.device, dtype=gauge_clean.dtype)
    sigmas3 = sigmas.view(b, 1, 1)
    eps = torch.empty_like(gauge_clean)
    eps.normal_(generator=generator)
    z_t = (1.0 - sigmas3) * gauge_clean + sigmas3 * eps
    v_gt = (z_t - gauge_clean) / sigmas3.clamp_min(float(getattr(video_target, "t_eps", 0.05)))
    return SimpleNamespace(
        sigmas=video_target.sigmas,
        sigmas4=sigmas3,
        z_t=z_t,
        v_gt=v_gt,
        eps=eps,
        weights=torch.ones((b, 1, 1), device=gauge_clean.device, dtype=gauge_clean.dtype),
        t_eps=float(getattr(video_target, "t_eps", 0.05)),
    )


def masked_gauge_direct_loss(
    predicted_gauge: torch.Tensor,
    target_gauge: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 in physical log units with an independent channel mask."""

    if predicted_gauge.shape != target_gauge.shape or predicted_gauge.shape[-2:] != (1, SCENE_GAUGE_DIM):
        raise ValueError("predicted and target gauge must match [B,1,3]")
    mask = torch.as_tensor(valid, device=predicted_gauge.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    if mask.shape != predicted_gauge.shape:
        raise ValueError(f"gauge valid mask shape {tuple(mask.shape)} != {tuple(predicted_gauge.shape)}")
    error = torch.nn.functional.smooth_l1_loss(
        predicted_gauge.float(),
        target_gauge.to(device=predicted_gauge.device, dtype=torch.float32),
        reduction="none",
    )
    weights = mask.to(dtype=error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def requested_render_pose_encoding(
    requested_c2w: torch.Tensor,
    trajectory_anchor_to_world: torch.Tensor,
    predicted_gauge: torch.Tensor,
) -> torch.Tensor:
    """Encode the requested metric camera with the generated scene gauge.

    This is the sole RGB/HDS camera construction used by training, validation,
    and offline inference.  The requested trajectory is expressed in the
    frozen teacher anchor, while both metric scale and render intrinsics come
    from ``predicted_gauge``.  No CameraHead trajectory or teacher-gauge
    fallback participates in this path.
    """

    requested = torch.as_tensor(requested_c2w)
    anchor = torch.as_tensor(trajectory_anchor_to_world)
    gauge = torch.as_tensor(predicted_gauge)
    if requested.ndim != 4 or tuple(requested.shape[-2:]) != (4, 4):
        raise ValueError("requested_c2w must be [B,S,4,4]")
    if anchor.ndim != 3 or tuple(anchor.shape[-2:]) != (4, 4):
        raise ValueError("trajectory_anchor_to_world must be [B,4,4]")
    if gauge.ndim != 3 or tuple(gauge.shape[-2:]) != (1, SCENE_GAUGE_DIM):
        raise ValueError(f"predicted_gauge must be [B,1,{SCENE_GAUGE_DIM}]")
    if int(requested.shape[0]) != int(anchor.shape[0]) or int(
        requested.shape[0]
    ) != int(gauge.shape[0]):
        raise ValueError("requested C, anchor, and gauge batch axes must match")
    if not bool(torch.isfinite(requested).all()):
        raise ValueError("requested_c2w contains non-finite values")
    if not bool(torch.isfinite(anchor).all()):
        raise ValueError("trajectory_anchor_to_world contains non-finite values")
    if not bool(torch.isfinite(gauge).all()):
        raise ValueError("predicted_gauge contains non-finite values")
    with torch.amp.autocast(device_type=requested.device.type, enabled=False):
        camera_to_world_dggt = metric_c2w_to_teacher_anchor_dggt(
            requested.float(),
            anchor.float(),
            gauge[..., 0].float(),
        )
        return assemble_dggt_pose_encoding(camera_to_world_dggt, gauge.float())


def requested_render_pose_for_rows(
    bundle: Any,
    predicted_gauge: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Select RGB-budget rows and apply the shared requested-camera contract."""

    rows = torch.as_tensor(
        row_indices, device=predicted_gauge.device, dtype=torch.long
    )
    if rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")

    def select(value: Any, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise RuntimeError(f"pretrain bundle is missing tensor {name}")
        if int(value.shape[0]) != int(predicted_gauge.shape[0]):
            raise ValueError(
                f"{name} batch {int(value.shape[0])} != predicted gauge batch "
                f"{int(predicted_gauge.shape[0])}"
            )
        return value.index_select(0, rows.to(device=value.device))

    return requested_render_pose_encoding(
        select(
            bundle.camera_to_world_requested_metric,
            "camera_to_world_requested_metric",
        ).detach(),
        select(
            bundle.camera_trajectory_anchor_to_world_metric,
            "camera_trajectory_anchor_to_world_metric",
        ).detach(),
        select(predicted_gauge, "predicted_gauge"),
    )


METRIC_UNAVAILABLE = -1.0
"""Sentinel written when a step produced no measurement for a gated metric.

Every series that can legitimately have no data on a step -- LiDAR AbsRel, and
the per-channel gauge diagnostics -- writes this value together with an
``*_available`` flag.  The flag, not the value, decides whether the observation
enters a W&B window mean, so a missing measurement never masquerades as a
perfect one.
"""


@torch.no_grad()
def gauge_diagnostic_metrics(
    predicted_gauge: torch.Tensor,
    target_gauge: torch.Tensor,
    valid: torch.Tensor,
    *,
    prior_log_scale: torch.Tensor | float,
    defer_log_values: bool = False,
) -> dict[str, TrainLogValue]:
    """Return physical gauge diagnostics, including the explicit constant prior.

    Keeping these formulas outside the trainer makes their channel masking and
    marginal-prior baseline directly testable without a full diffusion step.
    """

    predicted = predicted_gauge.detach().float()
    target = target_gauge.detach().to(device=predicted.device, dtype=torch.float32)
    if predicted.ndim != 3 or tuple(predicted.shape[1:]) != (1, SCENE_GAUGE_DIM):
        raise ValueError(
            f"predicted_gauge must be [B,1,{SCENE_GAUGE_DIM}], got {tuple(predicted.shape)}"
        )
    if target.shape != predicted.shape:
        raise ValueError(
            f"target_gauge shape {tuple(target.shape)} != prediction {tuple(predicted.shape)}"
        )
    mask = torch.as_tensor(valid, device=predicted.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    if mask.shape != predicted.shape:
        raise ValueError(
            f"valid must be [B,3] or [B,1,3], got {tuple(mask.shape)}"
        )
    weights = mask.float()
    absolute_error = (predicted - target).abs()
    scale_weights = weights[..., 0]
    scale_denominator = scale_weights.sum().clamp_min(1.0)
    fov_weights = weights[..., 1:3]
    fov_pred = 2.0 * torch.atan(torch.exp(predicted[..., 1:3]))
    fov_gt = 2.0 * torch.atan(torch.exp(target[..., 1:3]))
    prior = torch.as_tensor(
        prior_log_scale,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    if prior.numel() != 1 or not bool(torch.isfinite(prior).all()):
        raise ValueError("prior_log_scale must be one finite scalar")
    prior_error = (prior.reshape(()) - target[..., 0]).abs()
    pred_error = absolute_error[..., 0]
    prior_log_scale_error = (
        (prior_error * scale_weights).sum().div(scale_denominator)
    )
    predicted_log_scale_error = (
        (pred_error * scale_weights).sum().div(scale_denominator)
    )
    fov_denominator = fov_weights.sum()
    fov_error_deg = (
        (torch.rad2deg((fov_pred - fov_gt).abs()) * fov_weights)
        .sum()
        .div(fov_denominator.clamp_min(1.0))
    )
    # A sample whose offline gauge row has no valid ``log_metric_scale`` makes
    # both the numerator and ``clamp_min`` denominator above degenerate, and the
    # ratio lands on exactly 0.0 -- which in an absolute-error series reads as a
    # perfect prediction.  Emit the shared unavailable sentinel instead and let
    # the availability gate drop the observation, the same contract the LiDAR
    # AbsRel diagnostic already uses.  Scale and FOV are masked per channel, so
    # they need independent gates.
    scale_available = scale_weights.sum().gt(0.0).float()
    fov_available = fov_denominator.gt(0.0).float()
    unavailable = torch.full_like(predicted_log_scale_error, METRIC_UNAVAILABLE)

    def _gated(value: torch.Tensor, available: torch.Tensor) -> TrainLogValue:
        return deferred_log_value(
            torch.where(available.gt(0.0), value, unavailable),
            defer=defer_log_values,
        )

    return {
        "gauge_valid_frac": deferred_log_value(
            weights.mean(), defer=defer_log_values
        ),
        "gauge_scale_available": deferred_log_value(
            scale_available, defer=defer_log_values
        ),
        "gauge_fov_available": deferred_log_value(
            fov_available, defer=defer_log_values
        ),
        "gauge_log_scale_error": _gated(predicted_log_scale_error, scale_available),
        "gauge_prior_log_scale_error": _gated(prior_log_scale_error, scale_available),
        "gauge_fov_error_deg": _gated(fov_error_deg, fov_available),
        "gauge_vs_prior_gain": _gated(
            prior_log_scale_error - predicted_log_scale_error, scale_available
        ),
    }


@torch.no_grad()
def sampled_gauge_validation_metrics(
    predicted_gauge: torch.Tensor,
    target_gauge: torch.Tensor,
    valid: torch.Tensor,
    *,
    prior_log_scale: torch.Tensor | float,
    prefix: str = "sample_gauge",
) -> dict[str, float]:
    """Prefix the physical diagnostics for the final ODE-sampled gauge.

    The training-time x-prediction diagnostics are not a substitute for this:
    validation renders with the final sampled gauge, so the same sample must be
    measured directly against the offline 29-frame target and marginal prior.
    """

    if not prefix:
        raise ValueError("sampled gauge metric prefix must be non-empty")
    base = gauge_diagnostic_metrics(
        predicted_gauge,
        target_gauge,
        valid,
        prior_log_scale=prior_log_scale,
    )
    # Keep the sampled-gauge API stable and intentionally compact. The
    # training-only prior error is already encoded by ``vs_prior_gain`` plus
    # ``log_scale_error`` and should not silently create a fifth W&B series.
    # The two ``*_available`` gates travel with the errors they guard: without
    # them a masked channel is logged as an exact 0.0, which is the best
    # possible score rather than the absence of one.
    sampled_names = (
        "gauge_valid_frac",
        "gauge_scale_available",
        "gauge_fov_available",
        "gauge_log_scale_error",
        "gauge_fov_error_deg",
        "gauge_vs_prior_gain",
    )
    return {
        f"{prefix}_{name.removeprefix('gauge_')}": base[name]
        for name in sampled_names
    }


@torch.no_grad()
def sampled_latent_validation_metrics(
    generated_latent_n: torch.Tensor,
    target_latent_n: torch.Tensor,
    *,
    prefix: str = "sample_latent",
) -> dict[str, float]:
    """Score the sampled scene latent, which is what this run actually produces.

    Gauge, sky and sky-mask are side channels; the latent is what the frozen
    tokenizer decoder and DGGT heads turn into the scene, and it used to leave
    validation only as a PCA thumbnail and an absolute-error image.

    ``z`` is per-channel standardized by ``feature_stats``, so ``mse`` reads
    directly as the fraction of latent variance the sample fails to explain:
    1.0 is "no better than the marginal mean", 2.0 is "an independent draw".
    ``pred_std`` separates the two ways to land on 1.0 -- a blurred mean-like
    prediction collapses it toward 0, a correctly scaled but wrong sample keeps
    it near 1 -- which is the cheapest mode-collapse alarm available here.
    """

    if not prefix:
        raise ValueError("sampled latent metric prefix must be non-empty")
    predicted = generated_latent_n.detach().float()
    target = target_latent_n.detach().to(device=predicted.device, dtype=torch.float32)
    if predicted.shape != target.shape:
        raise ValueError(
            f"sampled latent shape {tuple(predicted.shape)} != target "
            f"{tuple(target.shape)}"
        )
    difference = predicted - target
    cosine = F.cosine_similarity(predicted, target, dim=-1, eps=1.0e-6)
    values = torch.stack(
        [
            difference.square().mean(),
            difference.abs().mean(),
            cosine.mean(),
            predicted.std(),
            target.std(),
        ]
    ).cpu()
    return {
        f"{prefix}_mse": float(values[0]),
        f"{prefix}_mae": float(values[1]),
        f"{prefix}_cosine": float(values[2]),
        f"{prefix}_pred_std": float(values[3]),
        f"{prefix}_target_std": float(values[4]),
    }


@torch.no_grad()
def sampled_render_validation_metrics(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    prefix: str = "sample_render",
) -> dict[str, float]:
    """PSNR/MAE of the sampled scene's 3DGS render against the real frames.

    ``loss_rgb_render_*`` during training measures the one-step x-prediction at
    a random sigma; it says nothing about what 50 Euler steps from noise
    actually look like.  Both tensors are ``[frames, 3, H, W]`` in [0,1] -- the
    same pair already written to disk as ``generated_raw_3dgs_rgb`` and
    ``input_rgb_gt`` -- so this is a subtraction over images that exist anyway.
    """

    if not prefix:
        raise ValueError("sampled render metric prefix must be non-empty")
    predicted = rendered_rgb.detach().float().clamp(0.0, 1.0)
    target = target_rgb.detach().to(device=predicted.device, dtype=torch.float32)
    if target.dtype == torch.uint8:
        target = target.float().div(255.0)
    target = target.clamp(0.0, 1.0)
    if predicted.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "sampled render metrics need [frames,3,H,W] tensors, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"rendered RGB shape {tuple(predicted.shape)} != target "
            f"{tuple(target.shape)}"
        )
    if predicted.shape[-2:] != target.shape[-2:]:
        # ``_fixed_render_hw`` renders at ``patch_grid * 14``; a dataset served
        # at a different resolution should downscale rather than abort a
        # validation pass over a diagnostic.
        target = F.interpolate(target, size=predicted.shape[-2:], mode="area")
    difference = predicted - target
    mse = difference.square().mean()
    values = torch.stack(
        [mse, difference.abs().mean(), -10.0 * torch.log10(mse.clamp_min(1.0e-10))]
    ).cpu()
    return {
        f"{prefix}_mse": float(values[0]),
        f"{prefix}_mae": float(values[1]),
        f"{prefix}_psnr": float(values[2]),
    }


METRIC_DEPTH_UNAVAILABLE = METRIC_UNAVAILABLE
"""Sentinel written when a step produced no LiDAR-anchored measurement."""


def metric_depth_diagnostic_log_values(
    relative_error: torch.Tensor | float | None = None,
    *,
    prefix: str = "metric_depth_rel_err",
) -> dict[str, float]:
    """Encode metric-depth availability without treating a sentinel as data."""

    if not prefix:
        raise ValueError("metric depth diagnostic prefix must be non-empty")
    available_key = f"{prefix}_available"
    if relative_error is None:
        return {prefix: METRIC_DEPTH_UNAVAILABLE, available_key: 0.0}
    value = torch.as_tensor(relative_error).detach().float()
    if value.numel() != 1:
        raise ValueError("metric depth relative error must be one scalar")
    if not bool(torch.isfinite(value).item()):
        return {prefix: METRIC_DEPTH_UNAVAILABLE, available_key: 0.0}
    return {
        prefix: float(value.item()),
        available_key: 1.0,
    }


@torch.no_grad()
def metric_depth_relative_error(
    depth_recon_dggt: torch.Tensor,
    lidar_depth_m: torch.Tensor,
    log_metric_scale: torch.Tensor,
    *,
    calibration,
    scale_valid: torch.Tensor | None = None,
    lidar_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Median LiDAR AbsRel for the generated gauge + decoded geometry chain."""

    depth = depth_recon_dggt.detach().float()
    if depth.ndim == 5 and int(depth.shape[-1]) == 1:
        depth = depth[..., 0]
    if depth.ndim != 4:
        raise ValueError(f"decoded depth must be [B,S,H,W](,1), got {tuple(depth.shape)}")
    lidar = lidar_depth_m.detach().to(device=depth.device, dtype=torch.float32)
    if lidar.ndim == 5 and int(lidar.shape[-1]) == 1:
        lidar = lidar[..., 0]
    if lidar.shape != depth.shape:
        raise ValueError(
            f"LiDAR depth shape {tuple(lidar.shape)} != decoded depth {tuple(depth.shape)}"
        )
    log_scale = torch.as_tensor(
        log_metric_scale, device=depth.device, dtype=torch.float32
    )
    if log_scale.ndim == 2 and int(log_scale.shape[-1]) == 1:
        log_scale = log_scale[:, 0]
    if log_scale.shape != (int(depth.shape[0]),):
        raise ValueError(
            f"log_metric_scale must be [B] or [B,1], got {tuple(log_scale.shape)}"
        )
    corrected_depth, _ = apply_depth_pullback_calibration(
        depth,
        log_metric_scale=log_scale,
        calibration=calibration,
        boundary="metric",
    )
    metric_depth = corrected_depth * torch.exp(log_scale).view(-1, 1, 1, 1)
    support = (
        torch.isfinite(metric_depth)
        & torch.isfinite(lidar)
        & (metric_depth > 0.0)
        & (lidar > 1.0)
        & (lidar < 80.0)
    )
    if lidar_valid is not None:
        lidar_support = torch.as_tensor(
            lidar_valid, device=depth.device, dtype=torch.bool
        )
        if lidar_support.ndim == 5 and int(lidar_support.shape[-1]) == 1:
            lidar_support = lidar_support[..., 0]
        if lidar_support.shape != depth.shape:
            raise ValueError(
                f"LiDAR validity shape {tuple(lidar_support.shape)} != decoded "
                f"depth {tuple(depth.shape)}"
            )
        support &= lidar_support
    if scale_valid is not None:
        valid_rows = torch.as_tensor(scale_valid, device=depth.device, dtype=torch.bool)
        if valid_rows.ndim > 1:
            valid_rows = valid_rows.reshape(int(depth.shape[0]), -1).all(dim=1)
        if valid_rows.shape != (int(depth.shape[0]),):
            raise ValueError(f"scale_valid must reduce to [B], got {tuple(valid_rows.shape)}")
        support &= valid_rows.view(-1, 1, 1, 1)
    if not bool(support.any()):
        return depth.new_tensor(float("nan"))
    return ((metric_depth - lidar).abs() / lidar.clamp_min(1.0e-6))[support].median()


# World-feedback weights.
#
# The v5 run measured these three at 0.1/0.1/0.1 and, after weighting, they came
# to 0.76% of the training loss -- of which 97.8% sat in the dynamic-conf term,
# an unbounded logit that never improved.  The parts that describe the rendered
# scene (depth, GS attributes, the decoded feature levels, the render itself)
# together carried 0.035%.  ``--head_dynamic_space probability`` removes the
# parasitic channel; these weights restore the remainder to a level where it can
# actually shape the trunk.
#
# Why raising them cannot drown the flow objective: the flow target carries
# ``1/sigma**2`` and every term below carries ``(1-sigma)**2``, so their relative
# scale is ``(1-sigma)**2 * sigma**2 <= 1/16``, maximal at sigma=0.5.  Whatever
# lambda is chosen, at no noise level can these outweigh generation.  The terms
# that genuinely compete are the ones with no sigma attenuation -- repa (0.5),
# the early-head coefficient (0.25) and the refined sky mask (0.1) -- and those
# are deliberately left alone.
# 1.0, matching its two siblings below.  The history here is a chain of wrong
# reference points: 0.005 was the unused code default, 0.05 was set against that
# rather than against the 0.1 the v5 launcher actually pinned, and 0.1 only
# restores v5 -- where the weight demonstrably did nothing.  In v5 the render
# term's effective weight quadrupled between step 5k and 10k as the ramp filled,
# and the render loss did not accelerate at all (-0.80%/1k against the flow
# loss's -0.58%/1k); it was riding the latent improving, not driving it.
#
# Measured on v6 at step 7.5k, at full ramp:
#
#     lambda    share of the training loss
#     0.05      0.0039%
#     0.1       0.0078%
#     1.0       0.0781%     <- level_consistency sits at 0.112%, head at 0.179%
#     13.0      1.01%
#
# The 41.6x attenuation between the raw pixel L1 (0.0466) and the weighted term
# (0.00112) is the sigma window, not the lambda, so lambda has to carry the
# whole correction.  At 0.1 this term is 23x below head_consistency, which was
# just deliberately raised to 1.0 -- and all four read the same decode of the
# same predicted latent, so that gap is not a considered ratio, it is an
# accident.  1.0 puts it in the same band as its siblings.
#
# Two independent reasons this is safe: the worst-case weight ratio against the
# flow target is lambda/16 = 0.062, so it cannot outrun generation at any noise
# level; and empirically the entire world-feedback stack switching on at step
# 5000 and ramping to 49% moved grad_norm by 0.45 of its own pre-existing
# standard deviation, in the negative direction.  ``grad_clip_active`` is the
# thing to watch -- it sat at 0.15-0.18 through v6, and a persistent move toward
# 1.0 is the signal to come back down.
RGB_RENDER_LAMBDA_DEFAULT = 1.0
# The sky dome's only pixel-space supervision, and the only term that can teach
# it detail finer than the atlas cells the flow loss sees.  It sat at 0.1 --
# 0.012% of the training loss, the second smallest term in the objective -- next
# to a dome that was still a flat blue wash at 50k steps.  It reads the same
# decode under the same sigma window as the three weights above, so it joins
# them at 1.0.
SKY_VIEW_LAMBDA_DEFAULT = 1.0
# The only term that measures sky *texture* rather than sky colour.  At 0.25 it
# was 0.014% of the total loss and, measured over the 3800 steps after the
# world-feedback ramp opened, the only sky term that did not move: charbonnier
# fell 4.88%/1k step and the flow loss 7.23%/1k while this sat at 0.18%/1k.
SKY_VIEW_HIGH_FREQUENCY_WEIGHT_DEFAULT = 1.0
# The sky atlas is an environment map at infinity and its renderer transforms
# camera rays by rotation only, so this loss touches neither the scene latent
# nor the gauge scale -- it shares the 5000-step gate with the 3DGS render only
# because it sits in the same block.  It does read the gauge FOV for ray
# directions, and ``train/gauge/fov_error`` is 5.7 deg at step 50 against a
# 1.41 deg atlas cell, so starting at 0 would teach the atlas through rays that
# are four cells off.  By step 1150 that is 1.13 deg, under one cell.
SKY_VIEW_START_STEP_DEFAULT = 1500
SKY_VIEW_WARMUP_STEPS_DEFAULT = 5000
# Sky tokens are 512 of the 9763 the trunk attends over -- 5.2% -- but the sky
# flow term was 0.170% of the loss while the sky *mask* family took 4.12%, a
# 15:1 split against the thing that decides what the sky looks like.
# Standardizing the target already multiplies this term by about 4.1x on its
# own, so 0.5 lands the sky at 4.0% of the loss; 1.0 would overshoot to 7.6%
# and cost the scene twice as much share.
DEFAULT_LAMBDA_SKY_FLOW = 0.5
SKY_MASK_REFINE_BOUNDARY_LOSS_WEIGHT_DEFAULT = 0.125
# Feature-level (L1) and frozen-head (L2) consistency.  Kept equal to each other,
# as in v5: both are read off the same decode of the same predicted latent under
# the same sigma weighting and the same masks, so there is no reason for one to
# outrank the other.  At 1.0 the head term reproduces v5's dynamic-conf gradient
# contribution exactly while giving depth/GS ten times the pull they had.
LEVEL_CONSISTENCY_LAMBDA_DEFAULT = 1.0
HEAD_CONSISTENCY_LAMBDA_DEFAULT = 1.0
RGB_RENDER_ROW_CAP = "per_rank_input_order_v1"
RGB_RENDER_CONTINUOUS_SIGMA_WEIGHT = "(1-sigma)^rgb_render_sigma_power"
PRETRAIN_RESUME_CONTRACT_VERSION = "layout_v2_pretrain_resume_v2"
PRETRAIN_RESUME_REPRODUCIBILITY = (
    "model_optimizer_scheduler_ema_only_rng_and_data_stream_not_restored"
)


def sky_mask_refine_boundary_loss_weight(args: argparse.Namespace) -> float:
    """Resolve the refined-mask boundary coefficient from one shared default."""

    return float(
        getattr(
            args,
            "sky_mask_refine_boundary_loss_weight",
            SKY_MASK_REFINE_BOUNDARY_LOSS_WEIGHT_DEFAULT,
        )
    )


RGB_RENDER_RESUME_ARGS = (
    "lambda_rgb_render",
    "lambda_level_consistency",
    "lambda_head_consistency",
    "head_dynamic_space",
    "rgb_render_every",
    "rgb_render_start_step",
    "rgb_render_warmup_steps",
    "rgb_render_sigma_power",
    "feedback_conf_weight_power",
    "feedback_conf_weight_floor",
    "rgb_render_max_samples",
    "rgb_render_max_frames",
    "rgb_render_stride",
    "rgb_render_sky_weight",
    "rgb_render_sky_mask_grad_scale",
    "rgb_render_lpips_weight",
    "rgb_render_lpips_net",
    "lambda_sky_view_reconstruction",
    "sky_view_lpips_weight",
    "sky_view_high_frequency_weight",
    "sky_view_start_step",
    "sky_view_warmup_steps",
)
RGB_RENDER_RESUME_DEFAULTS = {
    "lambda_rgb_render": RGB_RENDER_LAMBDA_DEFAULT,
    "lambda_level_consistency": LEVEL_CONSISTENCY_LAMBDA_DEFAULT,
    "lambda_head_consistency": HEAD_CONSISTENCY_LAMBDA_DEFAULT,
    "head_dynamic_space": "probability",
    "rgb_render_every": 1,
    "rgb_render_start_step": 5000,
    "rgb_render_warmup_steps": 5000,
    "rgb_render_sigma_power": 2.0,
    "feedback_conf_weight_power": 1.0,
    "feedback_conf_weight_floor": 0.05,
    "rgb_render_max_samples": 1,
    "rgb_render_max_frames": 0,
    "rgb_render_stride": 1,
    "rgb_render_sky_weight": 1.0,
    "rgb_render_sky_mask_grad_scale": 0.05,
    "rgb_render_lpips_weight": 0.01,
    "rgb_render_lpips_net": "alex",
    "lambda_sky_view_reconstruction": SKY_VIEW_LAMBDA_DEFAULT,
    "sky_view_lpips_weight": 0.01,
    "sky_view_high_frequency_weight": SKY_VIEW_HIGH_FREQUENCY_WEIGHT_DEFAULT,
    "sky_view_start_step": SKY_VIEW_START_STEP_DEFAULT,
    "sky_view_warmup_steps": SKY_VIEW_WARMUP_STEPS_DEFAULT,
}

PRETRAIN_RESUME_CRITICAL_ARGS = (
    "image_dir",
    "hdmap_root",
    "caption_root",
    "scene_gauge_path",
    "pullback_calibration_path",
    "dggt_ckpt_path",
    "tokenizer_ckpt_path",
    "feature_stats_path",
    "text_encoder_path",
    "text_max_length",
    "no_text_condition",
    "scene_start",
    "scene_end",
    "sequence_length",
    "batch_size",
    "grad_accum_steps",
    "seed",
    "precision",
    "max_steps",
    "optimizer_type",
    "lr",
    "final_lr",
    "scheduler_type",
    "warmup_steps",
    "decay_end_steps",
    "warmup_from_zero",
    "gmuon_momentum",
    "gmuon_nesterov",
    "gmuon_ns_coefficients_preset",
    "gmuon_ns_use_kernels",
    "weight_decay",
    "grad_clip_norm",
    "ema_decay",
    "lambda_flow",
    "lambda_preserve",
    "lambda_repa",
    "base_model_coeff",
    "lambda_boundary",
    "preserve_floor",
    "lambda_gauge_flow",
    "lambda_gauge_direct",
    "no_sky_generation",
    "sky_unobserved_loss_weight",
    # v5 standardizes the sky flow target.  Its tensors are shape-identical to
    # v4, so nothing else would stop a v4 checkpoint from loading into code
    # that means something different by the same numbers.
    "sky_representation_version",
    "sky_unobserved_loss_beta",
    "lambda_sky_flow",
    "lambda_sky_mask",
    "lambda_sky_mask_refine",
    "sky_mask_dice_weight",
    "sky_mask_pos_weight_max",
    "sky_mask_refine_boundary_weight",
    "sky_mask_refine_boundary_loss_weight",
    "text_uncond_drop_prob",
    "layout_max_actors",
    "static_far_plane_m",
    "layout_depth_tau",
    "layout_to_gauge_grad_scale",
    "layout_map_injection",
    "layout_actor_injection",
    "layout_map_metric_injection",
    "layout_actor_metric_injection",
    "appearance_context_injection",
    "val_sample_steps",
) + RGB_RENDER_RESUME_ARGS


def capped_render_row_indices(
    batch_size: int,
    max_samples: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Apply the per-rank render budget without reordering batch rows."""

    batch_size = int(batch_size)
    max_samples = int(max_samples)
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative; 0 means all rows")
    row_count = batch_size if max_samples == 0 else min(batch_size, max_samples)
    return torch.arange(row_count, device=device, dtype=torch.long)


def rgb_render_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Run-summary fields for capped rows with continuous sigma weighting."""

    effective = float(getattr(args, "lambda_rgb_render", RGB_RENDER_LAMBDA_DEFAULT))
    return {
        "hard_sigma_selection": False,
        "rgb_render_row_cap": RGB_RENDER_ROW_CAP,
        "rgb_render_max_samples_scope": "per_rank_input_order",
        "rgb_render_continuous_sigma_weighting": (
            RGB_RENDER_CONTINUOUS_SIGMA_WEIGHT
        ),
        "rgb_render_sigma_power": float(
            getattr(args, "rgb_render_sigma_power", 2.0)
        ),
        "lambda_rgb_render_effective": effective,
        "camera_source": "requested_C",
        "gauge_source": "predicted_gauge",
        **{
            name: _normalize_config_value(
                getattr(args, name, RGB_RENDER_RESUME_DEFAULTS[name])
            )
            for name in RGB_RENDER_RESUME_ARGS
        },
    }


def pretrain_resume_critical_args(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize training choices that a strict state resume may not change."""

    missing = [name for name in PRETRAIN_RESUME_CRITICAL_ARGS if not hasattr(args, name)]
    if missing:
        raise ValueError(f"runtime is missing critical resume arguments: {missing}")
    return {
        name: _normalize_config_value(getattr(args, name))
        for name in PRETRAIN_RESUME_CRITICAL_ARGS
    }


def validate_pretrain_resume_contract(
    payload: Any,
    args: argparse.Namespace,
    path: str | Path,
) -> None:
    """Reject any resume that changes task mixing, RGB/HDS, or critical args."""

    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a versioned pretraining checkpoint")
    version = payload.get("pretrain_resume_contract_version")
    if version != PRETRAIN_RESUME_CONTRACT_VERSION:
        raise ValueError(
            f"{path} pretrain resume contract={version!r}, expected "
            f"{PRETRAIN_RESUME_CONTRACT_VERSION!r}"
        )
    reproducibility = payload.get("pretrain_resume_reproducibility")
    if reproducibility != PRETRAIN_RESUME_REPRODUCIBILITY:
        raise ValueError(
            f"{path} resume reproducibility={reproducibility!r}, expected "
            f"{PRETRAIN_RESUME_REPRODUCIBILITY!r}"
        )
    saved_probabilities = payload.get("layout_task_probabilities")
    expected_probabilities = tuple(float(value) for value in LAYOUT_TASK_PROBABILITIES)
    if not isinstance(saved_probabilities, (list, tuple)) or tuple(
        float(value) for value in saved_probabilities
    ) != expected_probabilities:
        raise ValueError(
            f"{path} layout task probabilities do not match TC/TCMG/TCMGA "
            f"{expected_probabilities}"
        )
    expected_rgb = rgb_render_run_summary(args)
    if _normalize_config_value(payload.get("rgb_render")) != _normalize_config_value(
        expected_rgb
    ):
        raise ValueError(f"{path} RGB/HDS render contract does not match runtime")
    expected_critical = pretrain_resume_critical_args(args)
    saved_critical = payload.get("pretrain_resume_critical_args")
    if _normalize_config_value(saved_critical) != _normalize_config_value(
        expected_critical
    ):
        raise ValueError(f"{path} critical pretraining arguments do not match runtime")
    saved_args = payload.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError(f"{path} strict state resume requires the saved argparse mapping")
    missing_saved_args = sorted(set(PRETRAIN_RESUME_CRITICAL_ARGS) - set(saved_args))
    if missing_saved_args:
        raise ValueError(
            f"{path} saved argparse mapping is missing critical keys "
            f"{missing_saved_args}"
        )
    saved_arg_contract = {
        name: _normalize_config_value(saved_args[name])
        for name in PRETRAIN_RESUME_CRITICAL_ARGS
    }
    if saved_arg_contract != expected_critical:
        raise ValueError(f"{path} saved argparse values do not match runtime")


def render_sky_probability_from_primary_output(
    output: dict[str, Any],
) -> torch.Tensor:
    """Read the differentiable refined sky probability from the main forward."""

    logits = output.get("sky_mask_refined_logits")
    if not torch.is_tensor(logits):
        raise RuntimeError(
            "Pretrain RGB loss requires `sky_mask_refined_logits` from the primary "
            "SceneFlow forward."
        )
    return torch.sigmoid(logits.float())


def _cfg_metric_prefix(base: str, guidance_scale: float) -> str:
    scale = f"{float(guidance_scale):g}".replace("-", "m").replace(".", "p")
    return f"{base}_cfg{scale}"


def validation_guidance_scales(args: argparse.Namespace) -> list[float]:
    """Return primary + unique extra CFG scales in their requested order."""
    scales = [float(getattr(args, "guidance_scale", 1.0))]
    raw_extra = str(getattr(args, "val_guidance_scales", "") or "")
    for raw_scale in raw_extra.split(","):
        raw_scale = raw_scale.strip()
        if not raw_scale:
            continue
        scale = float(raw_scale)
        if not any(abs(scale - existing) <= 1e-6 for existing in scales):
            scales.append(scale)
    return scales


def validation_scale_indices_for_rank(
    num_scales: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, ...]:
    """Shard validation scales across ranks while retaining small-world fallback.

    With three scales and at least three ranks this maps scale 0/1/2 to rank
    0/1/2.  With fewer ranks the remaining scales are handled round-robin; with
    more ranks, surplus ranks wait at the common result-gather collective.
    """
    if num_scales < 0:
        raise ValueError(f"num_scales must be non-negative, got {num_scales}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world_size={world_size}")
    return tuple(range(rank, num_scales, world_size))


def validation_sampling_tasks_for_rank(
    num_scales: int,
    requested_scenes: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[tuple[int, int], ...]:
    """Assign ``(scene_offset, scale_index)`` validation jobs to one rank.

    A full validation uses ten scenes and three CFG scales.  When all 30 jobs
    can run concurrently, scene-major assignment maps them one-to-one to ranks
    0..29.  A smaller job falls back to :data:`VALIDATION_SCENE_FALLBACK`
    scenes rather than serializing ten expensive rollouts on too few
    accelerators; the CFG scales are still sharded round-robin and share the
    same scene and noise.

    The fallback is two, not one, because the scene set is split into a pinned
    half and a rotating half (see :func:`validation_pinned_scene_count`).  One
    scene would leave nothing rotating, so the smallest useful set is one of
    each -- six jobs, which fits the eight-accelerator floor.
    """

    if num_scales <= 0:
        raise ValueError(f"num_scales must be positive, got {num_scales}")
    if requested_scenes <= 0:
        raise ValueError(
            f"requested_scenes must be positive, got {requested_scenes}"
        )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world_size={world_size}")

    scene_count = effective_validation_scene_count(
        num_scales, requested_scenes, world_size=world_size
    )
    tasks = tuple(
        (scene_offset, scale_index)
        for scene_offset in range(scene_count)
        for scale_index in range(int(num_scales))
    )
    return tasks[rank::world_size]


VALIDATION_SCENE_FALLBACK = 2
"""Scenes kept when the job is too small to run the full set concurrently.

One pinned scene for the statistics and one rotating scene for coverage.  At
three CFG scales that is six jobs, so it still fits inside a single eight-GPU
node.
"""


def effective_validation_scene_count(
    num_scales: int,
    requested_scenes: int,
    *,
    world_size: int,
) -> int:
    """Return the scene count selected by the validation fallback policy."""

    if num_scales <= 0:
        raise ValueError(f"num_scales must be positive, got {num_scales}")
    if requested_scenes <= 0:
        raise ValueError(
            f"requested_scenes must be positive, got {requested_scenes}"
        )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size >= int(num_scales) * int(requested_scenes):
        return int(requested_scenes)
    return min(int(requested_scenes), VALIDATION_SCENE_FALLBACK)


def validation_pinned_scene_count(scene_count: int) -> int:
    """How many of the sampled scenes are pinned, i.e. never rotate.

    The pinned half is the measurement set: only those scenes feed the
    ``validation/sample_*`` series, so every point is scored on the same
    scenes and a change between two steps is the model changing.  The rotating
    half exists purely to put fresh pictures in the mosaic, and deliberately
    contributes no numbers -- averaging a rotating scene into the mean is what
    made the v6 sample series unreadable in the first place.
    """

    if int(scene_count) <= 0:
        raise ValueError(f"scene_count must be positive, got {scene_count}")
    return max(1, (int(scene_count) + 1) // 2)


def validation_scene_is_pinned(scene_offset: int, *, scene_count: int) -> bool:
    """Return whether one validation slot belongs to the fixed metric set."""

    pinned = validation_pinned_scene_count(scene_count)
    if int(scene_offset) < 0 or int(scene_offset) >= int(scene_count):
        raise ValueError(
            f"scene_offset {scene_offset} is outside scene_count={scene_count}"
        )
    return int(scene_offset) < pinned


def validation_scene_metrics_for_merge(
    scene_metrics: dict[str, float],
    *,
    scene_offset: int,
    scene_count: int,
) -> dict[str, float]:
    """Keep sampled metrics only for fixed slots; rotating slots are visual-only."""

    if validation_scene_is_pinned(scene_offset, scene_count=scene_count):
        return scene_metrics
    return {}


def _validation_scene_entry_groups(
    trunk_major_index: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Group flattened long-form dataset rows by their real scene identity."""

    grouped: dict[int, list[int]] = {}
    for dataset_index, entry in enumerate(trunk_major_index):
        if len(entry) < 2:
            raise ValueError(
                "trunk_major_index entries must contain at least "
                f"(scene_idx, trunk_idx), got {entry!r}"
            )
        grouped.setdefault(int(entry[0]), []).append(int(dataset_index))
    return tuple(tuple(indices) for indices in grouped.values())


def validation_available_scene_count(
    trunk_major_index: Sequence[Sequence[int]],
) -> int:
    """Count distinct usable scenes represented by complete long-form trunks."""

    return len(_validation_scene_entry_groups(trunk_major_index))


def validation_scene_dataset_index(
    scene_offset: int,
    *,
    scene_count: int,
    validation_index: int,
    trunk_major_index: Sequence[Sequence[int]],
) -> int:
    """Map a slot to a long-form dataset index for this validation.

    Pinned slots select the first usable trunk of their scene forever. Rotating
    slots sweep the remaining *scene identities* in blocks, never selecting a
    trunk from a pinned scene. Once the rotating scene pool wraps, each scene
    advances to its next available trunk so long-form coverage still grows.
    """

    pinned = validation_pinned_scene_count(scene_count)
    is_pinned = validation_scene_is_pinned(
        scene_offset, scene_count=scene_count
    )
    if int(validation_index) < 0:
        raise ValueError(
            f"validation_index must be non-negative, got {validation_index}"
        )
    scene_entries = _validation_scene_entry_groups(trunk_major_index)
    if len(scene_entries) < int(scene_count):
        raise ValueError(
            "long-form validation contains fewer distinct usable scenes than "
            f"scene_count: {len(scene_entries)} < {int(scene_count)}"
        )
    if is_pinned:
        return int(scene_entries[int(scene_offset)][0])

    rotating_total = int(scene_count) - pinned
    rotating_slot = int(scene_offset) - pinned
    rotating_scenes = scene_entries[pinned:]
    ordinal = int(validation_index) * rotating_total + rotating_slot
    scene_position = ordinal % len(rotating_scenes)
    visit = ordinal // len(rotating_scenes)
    entries = rotating_scenes[scene_position]
    return int(entries[visit % len(entries)])


def validation_scene_label(batch: dict[str, Any], scene_offset: int) -> str:
    """Return a filesystem/W&B-safe scene label from a collated batch."""

    raw = batch.get("scene_name")
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    if torch.is_tensor(raw) and raw.numel() == 1:
        raw = raw.item()
    label = str(raw) if raw is not None else f"slot{int(scene_offset):02d}"
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
    return safe or f"slot{int(scene_offset):02d}"


def all_gather_validation_results(
    payload: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Gather small JSON payloads with explicit NCCL-safe tensor collectives.

    PyTorch object collectives stage an internal pickle tensor on CUDA under
    NCCL and are not reliable on every supported PyTorch/NCCL combination.
    Explicit UTF-8 byte tensors keep device placement and collective ordering
    visible and deterministic.
    """
    if not is_distributed():
        return [payload]

    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    local_size = torch.tensor([len(encoded)], dtype=torch.int64, device=device)
    gathered_sizes = [torch.empty_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)

    local_bytes = torch.zeros((max_size,), dtype=torch.uint8, device=device)
    if encoded:
        local_bytes[: len(encoded)] = torch.tensor(
            list(encoded), dtype=torch.uint8, device=device
        )
    gathered_bytes = [torch.empty_like(local_bytes) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_bytes, local_bytes)

    results: list[dict[str, Any]] = []
    for byte_tensor, size in zip(gathered_bytes, sizes):
        raw = bytes(byte_tensor[:size].cpu().tolist())
        result = json.loads(raw.decode("utf-8"))
        if not isinstance(result, dict):
            raise TypeError(f"Validation rank payload must decode to a dict, got {type(result)!r}")
        results.append(result)
    return results


# Sampled-metric suffix -> the suffix of the gate that guards it.  Validation
# prefixes are built at runtime (``sample_gauge``, ``sample_gauge_cfg2``, ...),
# so the pairing is resolved by suffix instead of by a fixed key list.  Without
# this, one validation scene whose offline gauge row has no ``log_metric_scale``
# would drag the ``METRIC_UNAVAILABLE`` sentinel into the mean over the five
# sampled scenes.
_SAMPLED_METRIC_AVAILABILITY_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_prior_log_scale_error", "_scale_available"),
    ("_log_scale_error", "_scale_available"),
    ("_vs_prior_gain", "_scale_available"),
    ("_fov_error_deg", "_fov_available"),
)


def sampled_metric_availability_key(key: str) -> str | None:
    """Return the gate guarding ``key``, or None when it is always measured."""

    for suffix, gate_suffix in _SAMPLED_METRIC_AVAILABILITY_SUFFIXES:
        if key.endswith(suffix):
            return f"{key[: -len(suffix)]}{gate_suffix}"
    return None


def merge_validation_rank_results(
    rank_results: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Average repeated sampled metrics and collect every unique mosaic row.

    One scene's CFG scales are sampled on different ranks, so the rows of a
    single mosaic arrive from several payloads.  ``(slot, group, order)`` is the
    row's identity, and a collision means two ranks rendered the same cell —
    which would silently drop one of them from the mosaic.
    """

    sampled_metric_values: dict[str, list[float]] = {}
    mosaic_rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[int, str, int]] = set()
    for rank_result in rank_results:
        metrics = rank_result["metrics"]
        for key, value in metrics.items():
            key = str(key)
            gate = sampled_metric_availability_key(key)
            if gate is not None and float(metrics.get(gate, 1.0)) <= 0.0:
                # Keep the key so the series never vanishes from the dashboard,
                # but contribute no sample to its mean.
                sampled_metric_values.setdefault(key, [])
                continue
            sampled_metric_values.setdefault(key, []).append(float(value))
        for row in rank_result.get("mosaic_rows", ()):
            identity = (int(row["slot"]), str(row["group"]), int(row["order"]))
            if identity in seen_rows:
                raise RuntimeError(
                    f"duplicate validation artifact key across ranks: {identity}"
                )
            seen_rows.add(identity)
            mosaic_rows.append(row)
    averaged_metrics = {
        key: (sum(values) / len(values) if values else METRIC_UNAVAILABLE)
        for key, values in sampled_metric_values.items()
    }
    return averaged_metrics, mosaic_rows


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


def _batch_where(
    rows: torch.Tensor,
    when_true: torch.Tensor,
    when_false: torch.Tensor,
) -> torch.Tensor:
    """Select complete batch rows without broadcasting across semantic axes."""

    if tuple(when_true.shape) != tuple(when_false.shape):
        raise ValueError(
            f"row-select shape mismatch: {tuple(when_true.shape)} != "
            f"{tuple(when_false.shape)}"
        )
    if when_true.ndim == 0 or int(when_true.shape[0]) != int(rows.numel()):
        raise ValueError("row-select tensors must share their leading batch axis")
    mask = rows.to(device=when_true.device, dtype=torch.bool).view(
        int(rows.numel()), *([1] * (when_true.ndim - 1))
    )
    return torch.where(mask, when_true, when_false)


def _collated_string_rows(value: Any, *, batch_size: int, width: int) -> list[list[str]]:
    """Undo default-collate's transpose of a fixed-width string slot list."""

    if not isinstance(value, (list, tuple)):
        raise TypeError("actor_geometry_raw_track_key must be a collated string list")
    outer = list(value)
    if len(outer) == int(width) and all(
        isinstance(column, (list, tuple)) and len(column) == int(batch_size)
        for column in outer
    ):
        return [
            [str(outer[slot][row]) for slot in range(int(width))]
            for row in range(int(batch_size))
        ]
    if len(outer) == int(batch_size) and all(
        isinstance(row, (list, tuple)) and len(row) == int(width)
        for row in outer
    ):
        return [[str(item) for item in row] for row in outer]
    raise ValueError(
        "actor_geometry_raw_track_key must collate to [Kg][B] or [B][Kg], "
        f"got outer length {len(outer)} for B={batch_size}, Kg={width}"
    )


def _single_raster_schema_hash(value: Any, *, batch_size: int) -> str:
    values = [str(value)] if isinstance(value, str) else [str(item) for item in value]
    if len(values) == 1 and int(batch_size) > 1:
        values *= int(batch_size)
    if len(values) != int(batch_size) or len(set(values)) != 1:
        raise ValueError("raster_schema_hash must be identical for every batch row")
    schema = values[0]
    if schema != RASTER_SCHEMA_HASH:
        raise ValueError(
            f"raster schema mismatch: batch={schema!r}, runtime={RASTER_SCHEMA_HASH!r}"
        )
    return schema


def build_layout_condition_from_batch(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    device: torch.device,
    *,
    layout_max_actors: int,
    patch_grid: tuple[int, int],
) -> SimpleNamespace:
    """Rebuild the frozen G/projected-G/A/layout objects from collated fields."""

    required_tensors = (
        "layout_raster",
        "map_metric",
        "map_mode",
        "static_far_plane_m",
        "actor_geometry_slot_track_id",
        "actor_geometry_class_id",
        "actor_geometry_corners_world",
        "actor_geometry_velocity_world",
        "actor_geometry_box_size",
        "actor_geometry_yaw",
        "actor_geometry_is_moving",
        "actor_geometry_track_valid",
        "actor_geometry_slot_valid",
        "actor_geometry_layout_mode",
        "projected_actor_geometry_bbox_patch",
        "projected_actor_geometry_patch_weight",
        "projected_actor_geometry_log_z_patch",
        "projected_actor_geometry_silhouette_uv",
        "projected_actor_geometry_silhouette_vertex_valid",
        "projected_actor_geometry_corners_camera",
        "projected_actor_geometry_uv_corners",
        "projected_actor_geometry_velocity_camera",
        "projected_actor_geometry_uv_center",
        "projected_actor_geometry_log_z_w",
        "projected_actor_geometry_center_depth_valid",
        "projected_actor_geometry_frame_support",
        "projected_actor_geometry_metric_support",
        "projected_actor_geometry_in_frustum",
        "projected_actor_geometry_valid",
        "appearance_reference_rgb",
        "appearance_reference_alpha",
        "appearance_geometry_idx",
        "appearance_binding_valid",
        "appearance_class_id",
        "appearance_mode",
    )
    missing = [name for name in required_tensors if not torch.is_tensor(batch.get(name))]
    if missing:
        raise RuntimeError(
            "layout-v2 pretrain batch is missing tensor fields: " + ", ".join(missing)
        )
    if "actor_geometry_raw_track_key" not in batch:
        raise RuntimeError("layout-v2 pretrain batch is missing actor_geometry_raw_track_key")
    if "raster_schema_hash" not in batch:
        raise RuntimeError("layout-v2 pretrain batch is missing raster_schema_hash")

    raster = batch["layout_raster"].to(
        device=device, dtype=torch.uint8, non_blocking=True
    )
    batch_size = int(raster.shape[0])
    static_far_plane = batch["static_far_plane_m"].to(dtype=torch.float32).reshape(-1)
    if static_far_plane.numel() != batch_size or not bool(
        static_far_plane.eq(STATIC_FAR_PLANE_M).all()
    ):
        raise ValueError(
            f"static_far_plane_m must be {STATIC_FAR_PLANE_M:g} for every batch row"
        )
    geometry = ActorGeometryCondition(
        slot_track_id=batch["actor_geometry_slot_track_id"].to(
            device=device, dtype=torch.int64, non_blocking=True
        ),
        class_id=batch["actor_geometry_class_id"].to(
            device=device, dtype=torch.int8, non_blocking=True
        ),
        corners_world=batch["actor_geometry_corners_world"].to(
            device=device, dtype=torch.float64, non_blocking=True
        ),
        velocity_world=batch["actor_geometry_velocity_world"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        box_size=batch["actor_geometry_box_size"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        yaw=batch["actor_geometry_yaw"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        is_moving=batch["actor_geometry_is_moving"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        track_valid=batch["actor_geometry_track_valid"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        slot_valid=batch["actor_geometry_slot_valid"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        layout_mode=batch["actor_geometry_layout_mode"].to(
            device=device, dtype=torch.int8, non_blocking=True
        ).reshape(batch_size),
        raw_track_key=_collated_string_rows(
            batch["actor_geometry_raw_track_key"],
            batch_size=batch_size,
            width=int(batch["actor_geometry_slot_track_id"].shape[1]),
        ),
    )
    # ``ActorGeometryCondition.__post_init__`` already validates the complete
    # payload.  The only extra contract carried by ``layout_max_actors`` is a
    # cheap shape equality; do not rescan every CUDA field a second time here.
    if geometry.num_slots != int(layout_max_actors):
        raise ValueError(
            f"Kg={geometry.num_slots} must equal layout_max_actors="
            f"{int(layout_max_actors)}"
        )
    projected = ProjectedActorGeometry(
        bbox_patch=batch["projected_actor_geometry_bbox_patch"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        patch_weight=batch["projected_actor_geometry_patch_weight"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        log_z_patch=batch["projected_actor_geometry_log_z_patch"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        silhouette_uv=batch["projected_actor_geometry_silhouette_uv"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        silhouette_vertex_valid=batch[
            "projected_actor_geometry_silhouette_vertex_valid"
        ].to(device=device, dtype=torch.bool, non_blocking=True),
        corners_camera=batch["projected_actor_geometry_corners_camera"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        uv_corners=batch["projected_actor_geometry_uv_corners"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        velocity_camera=batch["projected_actor_geometry_velocity_camera"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        uv_center=batch["projected_actor_geometry_uv_center"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        log_z_w=batch["projected_actor_geometry_log_z_w"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        center_depth_valid=batch[
            "projected_actor_geometry_center_depth_valid"
        ].to(device=device, dtype=torch.bool, non_blocking=True),
        frame_support=batch["projected_actor_geometry_frame_support"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        metric_support=batch["projected_actor_geometry_metric_support"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        in_frustum=batch["projected_actor_geometry_in_frustum"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        valid=batch["projected_actor_geometry_valid"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
    )

    reference_rgb = batch["appearance_reference_rgb"].to(
        device=device, non_blocking=True
    )
    reference_alpha = batch["appearance_reference_alpha"].to(
        device=device, non_blocking=True
    )
    if reference_rgb.ndim != 5 or reference_alpha.ndim != 5:
        raise ValueError("appearance references must be [B,Ka,3/1,H,W]")
    num_bindings = int(reference_rgb.shape[1])
    encoder = _canonical_asset_encoder_for_model(
        vggt_model,
        scene_flow,
        device,
        patch_grid,
    )
    encoded = encoder(
        reference_rgb.reshape(
            batch_size * num_bindings, 1, 3, *reference_rgb.shape[-2:]
        ),
        reference_alpha.reshape(
            batch_size * num_bindings, 1, 1, *reference_alpha.shape[-2:]
        ),
        batch_size=batch_size,
        num_assets=num_bindings,
    )
    appearance = AppearanceBindingCondition(
        appearance_tokens=encoded.appearance_tokens,
        appearance_mask=encoded.appearance_mask,
        canonical_uv=encoded.canonical_uv.to(dtype=torch.float32),
        geometry_idx=batch["appearance_geometry_idx"].to(
            device=device, dtype=torch.int64, non_blocking=True
        ),
        binding_valid=batch["appearance_binding_valid"].to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        appearance_mode=batch["appearance_mode"].to(
            device=device, dtype=torch.int8, non_blocking=True
        ).reshape(batch_size),
    )
    appearance_class_id = batch["appearance_class_id"].to(
        device=device, dtype=torch.int8, non_blocking=True
    )
    schema = _single_raster_schema_hash(
        batch["raster_schema_hash"], batch_size=batch_size
    )
    layout = LayoutConditionBatch(
        raster=raster,
        map_metric=batch["map_metric"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        actor_geometry=geometry,
        projected_actor_geometry=projected,
        appearance=appearance,
        map_mode=batch["map_mode"].to(
            device=device, dtype=torch.int8, non_blocking=True
        ).reshape(batch_size),
        raster_schema_hash=schema,
    )
    return SimpleNamespace(
        layout=layout,
        appearance_class_id=appearance_class_id,
    )


def apply_layout_training_tasks(
    layout: LayoutConditionBatch,
    appearance_class_id: torch.Tensor,
    tasks: torch.Tensor,
) -> tuple[LayoutConditionBatch, torch.Tensor]:
    """Apply TC/TCMG/TCMGA row-wise while preserving all NULL semantics."""

    tasks = torch.as_tensor(tasks, device=layout.raster.device, dtype=torch.int8)
    if tuple(tasks.shape) != (layout.batch_size,):
        raise ValueError("layout tasks must be [B]")
    legal = torch.tensor(
        [int(task) for task in LayoutTask], device=tasks.device, dtype=tasks.dtype
    )
    if not bool((tasks[:, None] == legal[None]).any(dim=1).all()):
        raise ValueError("layout tasks contain an unknown state")
    drop_layout = tasks == int(LayoutTask.TC)
    drop_appearance = tasks != int(LayoutTask.TCMGA)

    null_layout = layout.without_layout()
    geometry = layout.actor_geometry
    null_geometry = null_layout.actor_geometry
    geometry_updates = {
        name: _batch_where(drop_layout, getattr(null_geometry, name), getattr(geometry, name))
        for name in (
            "slot_track_id",
            "class_id",
            "corners_world",
            "velocity_world",
            "box_size",
            "yaw",
            "is_moving",
            "track_valid",
            "slot_valid",
            "layout_mode",
        )
    }
    geometry_updates["raw_track_key"] = [
        (["" for _ in row] if bool(drop_layout[index].item()) else list(row))
        for index, row in enumerate(geometry.raw_track_key)
    ]
    tasked_geometry = replace(geometry, **geometry_updates)

    projected = layout.projected_actor_geometry
    null_projected = null_layout.projected_actor_geometry
    tasked_projected = replace(
        projected,
        **{
            name: _batch_where(
                drop_layout,
                getattr(null_projected, name),
                getattr(projected, name),
            )
            for name in projected.__dataclass_fields__
        },
    )
    appearance = layout.appearance
    null_appearance = appearance.null_like()
    tasked_appearance = replace(
        appearance,
        **{
            name: _batch_where(
                drop_appearance,
                getattr(null_appearance, name),
                getattr(appearance, name),
            )
            for name in appearance.__dataclass_fields__
        },
    )
    tasked_class_id = _batch_where(
        drop_appearance,
        torch.full_like(appearance_class_id, -1),
        appearance_class_id,
    )
    tasked = replace(
        layout,
        raster=_batch_where(drop_layout, null_layout.raster, layout.raster),
        map_metric=_batch_where(
            drop_layout, null_layout.map_metric, layout.map_metric
        ),
        actor_geometry=tasked_geometry,
        projected_actor_geometry=tasked_projected,
        appearance=tasked_appearance,
        map_mode=_batch_where(drop_layout, null_layout.map_mode, layout.map_mode),
    )
    return tasked, tasked_class_id


def layout_model_kwargs(
    layout: LayoutConditionBatch,
    appearance_class_id: torch.Tensor,
    *,
    gauge_grad_scale: float,
) -> dict[str, Any]:
    """Return the single frozen layout-v2 model-call payload."""

    # All dataclass replacements above validate their own wire invariants, and
    # WanSceneFlow performs the cross-object A/G/map checks at the public model
    # boundary.  Repeating the full raster/projected-G scans here forces many
    # device-to-host synchronizations immediately before that same validation.
    return {
        "layout_raster": layout.raster,
        "map_metric": layout.map_metric,
        "actor_geometry": layout.actor_geometry,
        "projected_actor_geometry": layout.projected_actor_geometry,
        "appearance": layout.appearance,
        "map_mode": layout.map_mode,
        "raster_schema_hash": layout.raster_schema_hash,
        "appearance_class_id": appearance_class_id,
        "layout_to_gauge_grad_scale": float(gauge_grad_scale),
    }


def layout_training_monitor_log_values(
    full_layout: LayoutConditionBatch,
    tasked_layout: LayoutConditionBatch,
    layout_tasks: torch.Tensor,
    *,
    gauge_grad_scale: float,
    defer_log_values: bool = False,
) -> dict[str, TrainLogValue]:
    """Compute the batch/task-dependent layout monitors emitted by train_step."""

    return {
        "layout/actor_count_mean": deferred_log_value(
            full_layout.actor_geometry.slot_valid.sum(dim=-1).float().mean(),
            defer=defer_log_values,
        ),
        # §7.5 expects this to read as `P(TCMGA) x mean K_a`, i.e. a per-sample
        # count of bound anchors.  Averaging over the fixed K_a slot axis instead
        # would report a fraction 1/K_a as large and make the documented
        # threshold look wrong.
        "layout/appearance_binding_rate": deferred_log_value(
            tasked_layout.appearance.binding_valid.sum(dim=-1).float().mean(),
            defer=defer_log_values,
        ),
        "layout_to_gauge_grad_scale": float(gauge_grad_scale),
        "task/TC_frac": deferred_log_value(
            (layout_tasks == int(LayoutTask.TC)).float().mean(),
            defer=defer_log_values,
        ),
        "task/TCMG_frac": deferred_log_value(
            (layout_tasks == int(LayoutTask.TCMG)).float().mean(),
            defer=defer_log_values,
        ),
        "task/TCMGA_frac": deferred_log_value(
            (layout_tasks == int(LayoutTask.TCMGA)).float().mean(),
            defer=defer_log_values,
        ),
    }


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
    """Save current layout-v2 training state under a fail-closed contract.

    RNG state and the DataLoader cursor are deliberately not serialized.  A
    resume restores model/EMA/optimizer/scheduler state, but does not claim a
    bit-exact continuation of the stochastic sample stream.
    """

    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sf = unwrap_ddp(scene_flow)
    scene_flow_state = sf.state_dict()
    ema_scene_flow_state = materialize_ema_state_dict(scene_flow, ema)
    scene_flow_config = sf.config.to_dict()
    flow_schedule_config = build_flow_schedule_config(
        args,
        prediction_type=scene_flow_prediction_type(sf),
        t_eps=scene_flow_t_eps(sf),
    )
    shared = {
        "pretrain_resume_contract_version": PRETRAIN_RESUME_CONTRACT_VERSION,
        "pretrain_resume_reproducibility": PRETRAIN_RESUME_REPRODUCIBILITY,
        "scene_flow_config": scene_flow_config,
        "flow_schedule_config": flow_schedule_config,
        "layout_task_probabilities": list(LAYOUT_TASK_PROBABILITIES),
        "layout_condition_version": LAYOUT_CONDITION_VERSION,
        "raster_schema_hash": RASTER_SCHEMA_HASH,
        "static_far_plane_m": float(args.static_far_plane_m),
        "rgb_render": rgb_render_run_summary(args),
        "pretrain_resume_critical_args": pretrain_resume_critical_args(args),
    }
    torch.save(
        {
            **shared,
            "step": int(step),
            "scene_flow": scene_flow_state,
            "ema_scene_flow": ema.state_dict(),
            "ema_scene_flow_state_dict": ema_scene_flow_state,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(args),
        },
        ckpt_dir / f"pretrain_step{step:06d}.pt",
    )
    torch.save(
        {**shared, "scene_flow": scene_flow_state, "step": int(step)},
        ckpt_dir / f"pretrain_step{step:06d}_weights_only.pt",
    )
    torch.save(
        {
            **shared,
            "scene_flow": ema_scene_flow_state,
            "step": int(step),
            "is_ema_weights": True,
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
    expected_step: int | None = None,
    args: argparse.Namespace | None = None,
) -> int:
    """Strictly restore current-architecture training state (not RNG/cursor)."""

    if not resume_path:
        return 0
    payload = torch.load(resume_path, map_location=device)
    if not isinstance(payload, dict) or "scene_flow" not in payload:
        raise ValueError(f"Unsupported resume checkpoint format: {resume_path}")
    validate_scene_flow_checkpoint_config(scene_flow, payload, resume_path)
    if args is None:
        raise ValueError("strict state resume requires runtime flow-schedule arguments")
    validate_checkpoint_flow_schedule(
        payload,
        args,
        resume_path,
        prediction_type=scene_flow_prediction_type(scene_flow),
        t_eps=scene_flow_t_eps(scene_flow),
    )
    validate_pretrain_resume_contract(payload, args, resume_path)
    required = {"step", "scene_flow", "ema_scene_flow", "optimizer", "lr_scheduler"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"--resume_path requires a full training checkpoint; missing {missing}"
        )
    step = int(payload["step"])
    if expected_step is not None and step != int(expected_step):
        raise ValueError(
            f"resume step mismatch: expected {int(expected_step)}, found {step}"
        )
    load_scene_flow_state_dict_strict(
        scene_flow,
        payload["scene_flow"],
        path=resume_path,
        source="scene_flow",
    )
    ema.load_state_dict(payload["ema_scene_flow"])
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    if is_main_process():
        print(f"[resume] loaded {resume_path} at step={step}", flush=True)
    return step




def init_wandb(args: argparse.Namespace, log_dir: Path):
    if not args.wandb or not is_main_process():
        return None
    # W&B distinguishes an absent run id from an exported empty string and
    # rejects the latter before ``wandb.init`` can use the explicit ``id``.
    # Sanitize both launcher residue and a manually supplied empty CLI value.
    if not os.environ.get("WANDB_RUN_ID", "").strip():
        os.environ.pop("WANDB_RUN_ID", None)
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is not None and not str(wandb_run_id).strip():
        wandb_run_id = None
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
        id=wandb_run_id,
        resume=args.wandb_resume,
    )
    return run


TrainLogValue = float | torch.Tensor
TrainLogSeries = list[TrainLogValue]


def deferred_log_value(
    value: torch.Tensor | float,
    *,
    defer: bool,
) -> TrainLogValue:
    """Keep scalar metrics on-device until a real logging consumer needs them."""

    tensor = torch.as_tensor(value).detach().float().reshape(())
    return tensor if defer else float(tensor.item())


def materialize_log_values(
    metrics: dict[str, TrainLogValue],
) -> dict[str, float]:
    """Transfer scalar Tensor metrics in one batch per device.

    Calling ``float(cuda_scalar)`` once per key creates one host synchronization
    per metric.  Grouped stacks preserve every value while reducing that to one
    synchronization per device and logging event.
    """

    result: dict[str, float] = {}
    grouped: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    for key, raw_value in metrics.items():
        if torch.is_tensor(raw_value):
            if raw_value.numel() != 1:
                raise ValueError(
                    f"log metric {key!r} must be scalar, got {tuple(raw_value.shape)}"
                )
            value = raw_value.detach().float().reshape(())
            grouped.setdefault(value.device, []).append((key, value))
        else:
            result[key] = float(raw_value)
    for entries in grouped.values():
        values = torch.stack([value for _, value in entries]).cpu().tolist()
        for (key, _), value in zip(entries, values):
            result[key] = float(value)
    return result


def log_wandb(run, metrics: dict[str, float], step: int, prefix: str) -> None:
    if run is None:
        return
    run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)


# Measurement -> availability gate.  A metric listed here is only averaged into
# a W&B window on the steps where its gate is positive; on every other step it
# carries ``METRIC_UNAVAILABLE`` and is skipped outright.  Register any series
# that a masked channel or a missing sensor can legitimately leave undefined,
# otherwise the degenerate value silently enters the mean.
_WANDB_METRIC_AVAILABILITY = {
    "metric_depth_rel_err": "metric_depth_rel_err_available",
    # The same generated depth measured against the offline LiDAR-calibrated
    # gauge instead of the generated one.  Its gap to the line above is the only
    # read on how much of the residual metric error is scale rather than
    # geometry, so it is a second measurement and not a prefix alias.
    "metric_depth_rel_err_teacher_gauge": (
        "metric_depth_rel_err_teacher_gauge_available"
    ),
    # ``scene_gauge_valid`` masks scale and FOV independently, and roughly one
    # sample in eight has no valid ``log_metric_scale`` at all.
    "gauge/log_scale_error": "gauge_scale_available",
    "gauge_prior_log_scale_error": "gauge_scale_available",
    "gauge_vs_prior_gain": "gauge_scale_available",
    "gauge/fov_error": "gauge_fov_available",
}
_METRIC_DEPTH_WANDB_AVAILABILITY = _WANDB_METRIC_AVAILABILITY
_METRIC_DEPTH_WANDB_HIDDEN_KEYS = frozenset(
    {
        # These two are pure control flow: ``train_step`` also branches on them,
        # and they only say whether the every-N-steps LiDAR diagnostic ran.  The
        # measurements themselves are reported -- LiDAR AbsRel is the only
        # absolute scale check in the run, and the ``_pred_gauge`` alias that
        # used to shadow it was never filled with an independent value.
        "metric_depth_rel_err_available",
        "metric_depth_rel_err_teacher_gauge_available",
    }
    # ``gauge_scale_available`` / ``gauge_fov_available`` are deliberately NOT
    # hidden.  Averaged over a logging window each one is the coverage rate of
    # its gauge channel -- how much of the training data has a metric scale to
    # learn from at all -- which is a measurement in its own right and the only
    # place that number is visible.
)


# Log keys this trainer pins to a constant, or that repeat a value another key
# already carries.  They are dropped once, where ``train_step`` finishes its log
# dict, so the same key stays out of the progress bar, the plain-text log and
# W&B without three separate filters drifting apart.
UNINFORMATIVE_TRAIN_LOG_KEYS = frozenset(
    {
        # Pretraining has no edit region: ``build_full_scene_bundle`` fixes
        # ``M_source`` to zeros and ``M_dest`` to ones, so the edit fraction is
        # 1.0 and the control-token estimate is 0.0 on every step.
        "dest_frac",
        "edit_frac",
        # ``compute_total_loss`` is called with ``lambda_identity=0.0`` and
        # ``identity_batch=False``, which makes this term exactly zero forever.
        "loss_identity",
        # Emitted twice from the same tensor by ``compute_total_loss``.
        "loss_flow_edit",
        # Pinned at the call site: 0.0 and 1.0 respectively.
        "rgb_render_camera_grad_scale",
        "rgb_render_gauge_pose_grad_scale",
        # Fixed by ``--rgb_render_max_frames`` / ``--rgb_render_max_samples``
        # and the window length; W&B already stores both in the run config.
        "rgb_render_frames",
        "rgb_render_samples",
        "feedback_frames",
        # ``--rgb_render_stride`` and ``--feedback_conf_weight_power`` verbatim.
        "feedback_stride",
        "feedback_conf_weight_power",
        # ``--rgb_render_sky_mask_grad_scale`` times ``rgb_render_ramp``, which
        # is logged on its own.
        "rgb_render_sky_mask_grad_scale",
    }
)


def drop_uninformative_log_values(
    logs: dict[str, TrainLogValue],
) -> dict[str, TrainLogValue]:
    """Strip the constant and duplicated series from one ``train_step`` log."""

    return {
        key: value
        for key, value in logs.items()
        if key not in UNINFORMATIVE_TRAIN_LOG_KEYS
    }


# Every rank packs into this fixed order.  A training step is allowed to omit a
# data-dependent metric (for example sky-view reconstruction or LiDAR AbsRel),
# but it may never invent an unregistered name.  Fixed slots make a sparse
# union safe without exchanging Python objects, hashing schemas, or assuming
# that every data shard took the same optional diagnostic branch.
ALL_RANK_TRAIN_LOG_KEYS: tuple[str, ...] = (
    "dataloader/wait_seconds",
    "feedback_conf_weight_mean",
    "feedback_sample_weight_mean",
    "gauge/fov_error",
    "gauge/log_scale_error",
    "gauge_fov_available",
    "gauge_prior_log_scale_error",
    "gauge_scale_available",
    "gauge_valid_frac",
    "gauge_vs_prior_gain",
    "layout/actor_count_mean",
    "layout/actor_metric_support_fraction",
    "layout/actor_residual_rms",
    "layout/appearance_binding_rate",
    "layout/appearance_invalid_all_window_count",
    "layout/appearance_invalid_all_window_rate",
    "layout/map_metric_valid_fraction",
    "layout/map_residual_rms",
    "layout_to_gauge_grad_scale",
    "loss",
    "loss_base",
    "loss_boundary",
    "loss_flow",
    "loss_gauge_direct",
    "loss_gauge_flow",
    "loss_head_consistency",
    "loss_head_consistency_no_conf",
    "loss_head_consistency_unweighted",
    "loss_head_consistency_weighted",
    "loss_head_depth",
    "loss_head_depth_conf",
    "loss_head_dynamic",
    "loss_head_gaussian",
    "loss_head_gs_conf",
    "loss_level_consistency",
    "loss_level_consistency_unweighted",
    "loss_level_consistency_weighted",
    "loss_preserve",
    "loss_repa",
    "loss_rgb_render",
    "loss_rgb_render_l1",
    "loss_rgb_render_l1_unweighted",
    "loss_rgb_render_lpips",
    "loss_rgb_render_no_conf",
    "loss_rgb_render_unweighted",
    "loss_rgb_render_weighted",
    "loss_sky_flow",
    "loss_sky_mask",
    "loss_sky_mask_bce",
    "loss_sky_mask_dice",
    "loss_sky_mask_refine",
    "loss_sky_mask_refine_bce",
    "loss_sky_mask_refine_boundary_bce",
    "loss_sky_mask_refine_dice",
    "loss_sky_view_charbonnier",
    "loss_sky_view_high_frequency",
    "loss_sky_view_lpips",
    "sky_view_detail_ratio",
    "loss_sky_view_weighted",
    "metric_depth_rel_err",
    "metric_depth_rel_err_available",
    "metric_depth_rel_err_teacher_gauge",
    "metric_depth_rel_err_teacher_gauge_available",
    "rgb_render_active",
    "rgb_render_alpha_mean",
    "rgb_render_conf_weight_mean",
    "rgb_render_depth_mean",
    "rgb_render_ramp",
    "rgb_render_sample_weight_mean",
    "rgb_render_sigma_mean",
    "rgb_render_sigma_weight_mean",
    "rgb_render_weight_mean",
    "sigma_mean",
    "sky_mask_iou",
    "sky_mask_pos_weight",
    "sky_mask_pred_frac",
    "sky_mask_refine_boundary_frac",
    "sky_mask_refine_iou",
    "sky_mask_refine_pos_weight",
    "sky_mask_refine_pred_frac",
    "sky_mask_refine_target_frac",
    "sky_mask_target_frac",
    "sky_token_loss_weight_mean",
    "sky_view_ramp",
    "sky_view_sample_weight_mean",
    "task/TCMG_frac",
    "task/TCMGA_frac",
    "task/TC_frac",
    "text_uncond_drop_frac",
)
if len(ALL_RANK_TRAIN_LOG_KEYS) != len(set(ALL_RANK_TRAIN_LOG_KEYS)):
    raise RuntimeError("ALL_RANK_TRAIN_LOG_KEYS contains duplicate names")
_ALL_RANK_TRAIN_LOG_INDEX = {
    key: index for index, key in enumerate(ALL_RANK_TRAIN_LOG_KEYS)
}


def all_rank_log_mean(
    logs: dict[str, TrainLogValue],
    *,
    device: torch.device,
) -> dict[str, TrainLogValue]:
    """Replace one step's rank-local metrics with the mean over every rank.

    Rank 0's own losses describe only one shard of the global batch. Detached
    scalar tensors therefore stay on the accelerator and are packed into one
    buffer before the collective. This avoids both rank-0 bias and the former
    per-key ``.item()`` synchronizations.

    The buffer is ``float32`` on purpose.  This is the only collective the
    training loop issues outside DDP, and it runs on whatever accelerator the
    job was launched on; vendor collective libraries (ACCL-P on the PPU nodes,
    for instance) commonly implement fp32/bf16 and not fp64, where an unsupported
    dtype shows up as a hung all-reduce rather than an error -- every GPU busy,
    the step counter frozen.  fp32 costs nothing here: these are display values
    summed over at most a few thousand ranks.

    Every registered key has a total and an observation-count slot. Optional
    keys therefore form a sparse union across ranks: a missing key contributes
    neither a fabricated zero nor a sentinel. Availability-gated metrics use
    their explicit gate as the observation count. One final protocol-error slot
    makes unknown/non-scalar values fail on every rank *after* the collective,
    instead of letting one rank raise early while its peers hang in all-reduce.
    """

    if not is_distributed():
        return logs
    world_size = dist.get_world_size()
    if world_size <= 1:
        return logs

    def scalar_tensor(raw_value: TrainLogValue) -> torch.Tensor:
        if torch.is_tensor(raw_value):
            if raw_value.numel() != 1:
                raise ValueError(
                    "all-rank log values must be scalar, got "
                    f"{tuple(raw_value.shape)}"
                )
            return raw_value.detach().to(
                device=device, dtype=torch.float32, non_blocking=True
            ).reshape(())
        return torch.tensor(float(raw_value), device=device, dtype=torch.float32)

    width = len(ALL_RANK_TRAIN_LOG_KEYS)
    packed = torch.zeros(
        (2 * width + 1,), device=device, dtype=torch.float32
    )
    one = torch.ones((), device=device, dtype=torch.float32)
    zero = torch.zeros((), device=device, dtype=torch.float32)
    protocol_errors = 0
    for key, raw_value in logs.items():
        index = _ALL_RANK_TRAIN_LOG_INDEX.get(key)
        if index is None or (torch.is_tensor(raw_value) and raw_value.numel() != 1):
            protocol_errors += 1
            continue
        availability_key = _WANDB_METRIC_AVAILABILITY.get(key)
        if availability_key is not None and availability_key not in logs:
            # Route this through the same all-reduced error slot as an
            # unregistered key: raising here would abort one rank inside a
            # collective and hang every other rank in ``all_reduce``.
            protocol_errors += 1
            continue
        mask = (
            one
            if availability_key is None
            else torch.where(
                scalar_tensor(logs[availability_key]).gt(0.0),
                one,
                zero,
            )
        )
        packed[index] = scalar_tensor(raw_value) * mask
        packed[width + index] = mask
    packed[-1] = float(protocol_errors)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = packed.cpu().tolist()
    if reduced[-1] > 0.0:
        raise RuntimeError(
            "at least one rank produced an unregistered or non-scalar training "
            "log value; update ALL_RANK_TRAIN_LOG_KEYS before training"
        )

    result: dict[str, TrainLogValue] = {}
    for index, key in enumerate(ALL_RANK_TRAIN_LOG_KEYS):
        count = reduced[width + index]
        if count > 0.0:
            result[key] = reduced[index] / count
        elif key in _METRIC_DEPTH_WANDB_AVAILABILITY and reduced[
            width
            + _ALL_RANK_TRAIN_LOG_INDEX[
                _METRIC_DEPTH_WANDB_AVAILABILITY[key]
            ]
        ] > 0.0:
            # Keep the existing unavailable marker so W&B's metric-depth gate
            # can suppress the point when no rank had a valid LiDAR sample.
            result[key] = METRIC_DEPTH_UNAVAILABLE
    return result


# The head of the plain-text ``--no_tqdm`` line: the total plus one field per
# loss that is separately weighted into it, so a rising total can be attributed
# without opening W&B.  The dataloader wait leads -- it is the one number that
# says whether the GPUs are starved rather than slow.  The rest of the ~80 keys
# still follow on the same line; this only fixes what comes first.
#
# The tqdm bar and newline log use the same compact leading fields. W&B keeps
# the complete metric dictionary.
PROGRESS_POSTFIX_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("wait", "dataloader/wait_seconds", ".2f"),
    ("loss", "loss", ".3f"),
    ("flow", "loss_flow", ".3f"),
    ("sky", "loss_sky_flow", ".3f"),
    ("repa", "loss_repa", ".3f"),
    ("mask", "loss_sky_mask_refine", ".3f"),
    ("rgb", "loss_rgb_render", ".3f"),
    ("fov", "gauge/fov_error", ".1f"),
    # Leads the loss weights: above --grad_clip_norm every lambda is competing
    # for a fixed budget rather than adding to it.
    ("gnorm", "grad_norm", ".2f"),
    ("lr", "lr", ".1e"),
)


_PROGRESS_POSTFIX_KEYS = frozenset(key for _, key, _ in PROGRESS_POSTFIX_FIELDS)


def progress_postfix(metrics: dict[str, TrainLogValue]) -> dict[str, str]:
    """Select and order the compact progress/plain-text loss summary."""

    selected: dict[str, TrainLogValue] = {}
    for label, key, spec in PROGRESS_POSTFIX_FIELDS:
        value = metrics.get(key)
        if value is None:
            # The RGB render loss only exists on the steps it actually ran.
            continue
        selected[label] = value
    materialized = materialize_log_values(selected)
    return {
        label: format(materialized[label], spec)
        for label, _key, spec in PROGRESS_POSTFIX_FIELDS
        if label in materialized
    }


_ANSI_CONTROL_SEQUENCE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class WebConsoleTqdmStream:
    """Turn carriage-return tqdm refreshes into flushed Web-log records.

    A terminal interprets ``\r`` as "rewrite the current line". PAI-DLC's Web
    collector is line-oriented and does not publish that record until it sees
    ``\n``. This adapter preserves tqdm's formatting/ETA calculation while
    making every refresh an ordinary newline-terminated log entry. Cursor
    movement emitted by nested validation bars is removed as well.
    """

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def write(self, value: str) -> int:
        raw = str(value)
        cleaned = _ANSI_CONTROL_SEQUENCE.sub("", raw)
        fragments = re.split(r"[\r\n]+", cleaned)
        wrote_record = False
        for fragment in fragments:
            line = fragment.rstrip()
            if not line:
                continue
            self.stream.write(line + "\n")
            wrote_record = True
        if wrote_record:
            self.stream.flush()
        return len(raw)

    def flush(self) -> None:
        self.stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


def tqdm_output_stream(*, force_tqdm: bool) -> Any | None:
    """Return a newline adapter only for the non-TTY PPU Web-console mode."""

    return WebConsoleTqdmStream(sys.stderr) if bool(force_tqdm) else None


def use_interactive_tqdm(
    no_tqdm: bool,
    *,
    force_tqdm: bool = False,
    stream: Any | None = None,
) -> bool:
    """Use tqdm on a terminal, or when a Web-console launcher forces it.

    PAI-DLC and similar Web consoles collect ``stderr`` through a pipe (the PPU
    launchers additionally use ``tee``), so ``isatty()`` is false even though
    their log viewer can display tqdm output and its ETA.  ``force_tqdm`` is an
    explicit launcher policy for that case; ordinary redirected jobs retain the
    newline/flush fallback.
    """

    if bool(no_tqdm):
        return False
    if bool(force_tqdm):
        return True
    target = sys.stderr if stream is None else stream
    isatty = getattr(target, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except OSError:
        return False


def accumulate_wandb_metrics(
    sums: dict[str, TrainLogSeries],
    observation_counts: dict[str, int],
    metrics: dict[str, TrainLogValue],
) -> None:
    """Accumulate one W&B observation without averaging missing diagnostics."""

    for key, raw_value in metrics.items():
        availability_key = _WANDB_METRIC_AVAILABILITY.get(key)
        if availability_key is not None:
            if availability_key not in metrics:
                # Silently treating a missing gate as "unavailable" would drop
                # the series from W&B with no diagnosis; treating it as
                # "available" would let a sentinel into the mean.  Both are
                # worse than saying which emitter forgot the gate.
                raise KeyError(
                    f"log metric {key!r} is availability-gated but its gate "
                    f"{availability_key!r} was not emitted on the same step"
                )
            if float(metrics[availability_key]) <= 0.0:
                continue
        value: TrainLogValue
        if torch.is_tensor(raw_value):
            value = raw_value.detach().float().reshape(())
        else:
            value = float(raw_value)
        # Do not add CUDA scalars here: one addition per metric and step would
        # replace host synchronizations with dozens of tiny GPU kernels.  A
        # logging window contains only a few thousand scalar bytes, so retain
        # the detached leaves and reduce them once when the window is flushed.
        sums.setdefault(key, []).append(value)
        observation_counts[key] = observation_counts.get(key, 0) + 1


def finalize_wandb_metrics(
    sums: dict[str, TrainLogSeries],
    observation_counts: dict[str, int],
) -> dict[str, float]:
    """Build the compact W&B view from a completed aggregation window."""

    deferred: dict[str, TrainLogValue] = {}
    for key, values in sums.items():
        if key in _METRIC_DEPTH_WANDB_HIDDEN_KEYS:
            continue
        count = int(observation_counts.get(key, 0))
        if count <= 0:
            continue
        if len(values) != count:
            raise RuntimeError(
                f"wandb metric {key!r} has {len(values)} values but count={count}"
            )
        tensor_values = [value for value in values if torch.is_tensor(value)]
        if tensor_values:
            device = tensor_values[0].device
            if any(value.device != device for value in tensor_values):
                raise RuntimeError(f"wandb metric {key!r} changed device in one window")
            stacked = torch.stack(
                [
                    value
                    if torch.is_tensor(value)
                    else torch.tensor(float(value), device=device)
                    for value in values
                ]
            )
            deferred[key] = stacked.mean()
        else:
            deferred[key] = sum(float(value) for value in values) / float(count)
    return materialize_log_values(deferred)


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


LATENT_PCA_BASIS_SEED = 0
"""Fixed seed for the randomized PCA behind the latent thumbnails.

Visualization only.  It never touches the sampling or noise RNG -- the fit runs
inside ``torch.random.fork_rng``.
"""


@dataclass(frozen=True)
class LatentPcaBasis:
    """One projection shared by every latent row of a mosaic.

    Fitting ``pca_lowrank`` separately per row -- which is what the mosaic did
    at first -- gives each row its own axes and its own contrast stretch, so
    the same colour means something different in the GT row and in each CFG
    row.  The group caption invites exactly the comparison that setup cannot
    support.  Fitting once on the ground-truth latent and projecting every row
    through it makes colour mean one thing per mosaic, and lets a generated
    latent that drifts out of the GT range say so by clipping.
    """

    mean: torch.Tensor
    components: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor


def _latent_pca_basis(z: torch.Tensor, max_frames: int) -> LatentPcaBasis:
    """Fit the shared basis on one reference latent, with a stable sign."""

    reference = z[:1, :max_frames].detach().float().cpu()
    flat = reference.reshape(-1, int(reference.shape[-1]))
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    if centered.shape[0] < 3 or centered.shape[1] < 3:
        components = torch.zeros((centered.shape[1], 3), dtype=centered.dtype)
        width = min(3, centered.shape[1])
        components[:width, :width] = torch.eye(width, dtype=centered.dtype)
    else:
        # ``pca_lowrank`` projects onto a *random* subspace, so back-to-back
        # calls on the same tensor return different axes -- the per-row fit this
        # replaces was not even reproducible for one latent across two steps.
        # Every rank has to land on the same basis without talking to the
        # others, so pin the projection with a local seed instead of leaking one
        # into the sampling RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(LATENT_PCA_BASIS_SEED)
            _, _, vh = torch.pca_lowrank(centered, q=3, center=False)
        components = vh[:, :3]
        # Pin the sign as well: it is free, and it keeps the colours stable when
        # the reference latent shifts slightly between steps.  A flipped
        # component would repaint a whole row in the complementary colour.
        dominant = components.abs().argmax(dim=0)
        signs = torch.sign(components[dominant, torch.arange(components.shape[1])])
        components = components * torch.where(
            signs == 0, torch.ones_like(signs), signs
        )
    projected = centered @ components
    return LatentPcaBasis(
        mean=mean,
        components=components,
        lo=projected.quantile(0.01, dim=0, keepdim=True),
        hi=projected.quantile(0.99, dim=0, keepdim=True),
    )


def _latent_pca_grid(
    z: torch.Tensor,
    patch_grid: tuple[int, int],
    max_frames: int,
    basis: LatentPcaBasis | None = None,
) -> torch.Tensor:
    """Project `[B,S,P,C]` latent tokens to an RGB patch grid for qualitative checks."""
    z = z[:1, :max_frames].detach().float().cpu()
    _, seq_len, num_patches, channels = z.shape
    gy, gx = patch_grid
    if num_patches != gy * gx:
        raise ValueError(f"latent patch count {num_patches} != patch_grid {patch_grid}")
    if basis is None:
        basis = _latent_pca_basis(z, max_frames)
    if int(basis.components.shape[0]) != channels:
        raise ValueError(
            f"latent PCA basis has {int(basis.components.shape[0])} channels, "
            f"latent has {channels}"
        )
    rgb = (z.reshape(-1, channels) - basis.mean) @ basis.components
    rgb = ((rgb - basis.lo) / (basis.hi - basis.lo).clamp_min(1e-6)).clamp(0.0, 1.0)
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


# The mosaic stacks one latent-error row per CFG scale, so a per-image
# percentile normalisation would silently give every row its own scale and make
# the comparison meaningless.  ``z_clean_n`` is standardised, so a fixed
# absolute scale is both comparable across scales and stable across steps: an
# uncorrelated prediction sits near E|N(0, sqrt(2))| = 1.13, i.e. saturated
# white, and the row darkens as training converges.
LATENT_ERROR_VIZ_SCALE = 1.0


def _absolute_mask_grid(
    mask: torch.Tensor,
    patch_grid: tuple[int, int],
    max_frames: int,
    scale: float = LATENT_ERROR_VIZ_SCALE,
) -> torch.Tensor:
    if float(scale) <= 0.0:
        raise ValueError(f"absolute mask scale must be positive, got {scale}")
    grid = _mask_grid(mask, patch_grid, max_frames)
    return (grid / float(scale)).clamp(0.0, 1.0)


def _image_grid(images: torch.Tensor, max_frames: int) -> torch.Tensor:
    selected = images[:1, :max_frames].detach().cpu()
    if selected.dtype == torch.uint8:
        selected = selected.float().div_(255.0)
    else:
        selected = selected.float()
    return selected.reshape(
        -1,
        *images.shape[2:],
    ).clamp(0.0, 1.0)


def _images_to_device(
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        # Match torchvision ToTensor exactly so uint8 IPC is representation
        # only and cannot perturb the frozen DGGT input by one ULP.
        return images.float().div_(255.0)
    return images


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


def _sky_mask_overlay_grid(
    target_sky_mask: torch.Tensor | None,
    predicted_sky_mask_grid: torch.Tensor | None,
    max_frames: int,
) -> tuple[torch.Tensor, bool] | None:
    """Overlay GT (red) and predicted (green) sky so agreement reads yellow.

    Two stacked binary rows are hard to diff by eye; one overlay makes every
    disagreement a saturated hue: red is sky the model missed, green is sky it
    invented, yellow is agreement.
    """

    if not torch.is_tensor(predicted_sky_mask_grid):
        return None
    predicted = (
        predicted_sky_mask_grid[:max_frames, :1]
        .detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
    )
    if not torch.is_tensor(target_sky_mask):
        return predicted.repeat(1, 3, 1, 1), False
    target = _sky_mask_image_grid(target_sky_mask, max_frames)
    if target.shape[0] != predicted.shape[0]:
        frames = min(int(target.shape[0]), int(predicted.shape[0]))
        target, predicted = target[:frames], predicted[:frames]
    if target.shape[-2:] != predicted.shape[-2:]:
        target = F.interpolate(target, size=predicted.shape[-2:], mode="area")
    return torch.cat([target, predicted, torch.zeros_like(predicted)], dim=1), True


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


def sky_token_channel_stats(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token-channel ``(mean, std)`` of shape ``[SKY_TOKEN_DIM]``.

    ``pixel_unshuffle`` lays the packed channel out as ``c * patch**2 + sub``,
    so the 192 token channels are 64 red, then 64 green, then 64 blue.  The
    standardization is therefore one constant per RGB channel repeated over its
    64 sub-positions, which keeps it a per-dimension affine.
    """
    repeat = SKY_PATCH_SIZE * SKY_PATCH_SIZE
    mean = torch.tensor(SKY_TOKEN_CHANNEL_MEAN, device=device, dtype=dtype)
    std = torch.tensor(SKY_TOKEN_CHANNEL_STD, device=device, dtype=dtype)
    if not bool(std.gt(0.0).all()):
        raise ValueError(f"SKY_TOKEN_CHANNEL_STD must be positive, got {SKY_TOKEN_CHANNEL_STD}")
    return mean.repeat_interleave(repeat), std.repeat_interleave(repeat)


def pack_sky_rgb_atlas(atlas_rgb: torch.Tensor, *, standardize: bool = True) -> torch.Tensor:
    """Pack a ``DEFAULT_SKY_ATLAS_HW`` RGB atlas into the sky token grid.

    Under ``SKY_REPRESENTATION_VERSION`` that is a 128x256 atlas packed as 8x8
    patches into 512 tokens of 192 channels.  The token count is what the
    trunk attends over and is deliberately held fixed across atlas sizes; the
    patch absorbs the resolution instead.

    The packed token is then standardized per RGB channel so the flow target
    has zero mean and unit variance like the scene latent.  Pass
    ``standardize=False`` only to measure the constants themselves.
    """
    if atlas_rgb.ndim != 4 or int(atlas_rgb.shape[1]) != SKY_RGB_DIM:
        raise ValueError(f"sky atlas must be [B,3,H,W], got {tuple(atlas_rgb.shape)}")
    ah, aw = int(atlas_rgb.shape[-2]), int(atlas_rgb.shape[-1])
    if (ah, aw) != DEFAULT_SKY_ATLAS_HW:
        raise ValueError(
            f"sky atlas must be {DEFAULT_SKY_ATLAS_HW} for "
            f"{SKY_REPRESENTATION_VERSION}, got {(ah, aw)}"
        )
    packed = torch.nn.functional.pixel_unshuffle(atlas_rgb.float() * 2.0 - 1.0, SKY_PATCH_SIZE)
    tokens = packed.permute(0, 2, 3, 1).reshape(atlas_rgb.shape[0], -1, SKY_TOKEN_DIM).contiguous()
    if not standardize:
        return tokens
    mean, std = sky_token_channel_stats(device=tokens.device, dtype=tokens.dtype)
    return (tokens - mean) / std


def pack_sky_atlas_loss_weight(
    observation_mask: torch.Tensor,
    *,
    unobserved_weight: float = 0.0,
) -> torch.Tensor:
    """Pack per-atlas-pixel visibility into weights aligned with 192D RGB tokens.

    A sky token contains the RGB values of an 8x8 atlas patch. Packing one
    scalar weight per token would incorrectly supervise unobserved subpixels
    whenever only part of that patch is visible.  Apply the exact same
    pixel-unshuffle layout as :func:`pack_sky_rgb_atlas` so every RGB output
    channel has its own supervision weight.
    """
    if observation_mask.ndim != 4 or int(observation_mask.shape[1]) != 1:
        raise ValueError(
            f"sky observation mask must be [B,1,H,W], got {tuple(observation_mask.shape)}"
        )
    ah, aw = int(observation_mask.shape[-2]), int(observation_mask.shape[-1])
    if (ah, aw) != DEFAULT_SKY_ATLAS_HW:
        raise ValueError(f"sky observation mask must be {DEFAULT_SKY_ATLAS_HW}, got {(ah, aw)}")
    unobserved_weight = float(unobserved_weight)
    if not 0.0 <= unobserved_weight <= 1.0:
        raise ValueError(f"unobserved sky loss weight must be in [0,1], got {unobserved_weight}")

    observed = observation_mask.float().clamp(0.0, 1.0).expand(-1, SKY_RGB_DIM, -1, -1)
    # A clip with no valid sky observation has no meaningful completion target.
    # Keep the entire atlas unsupervised instead of teaching the black placeholder.
    has_observation = observed.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    weight = (observed + (1.0 - observed) * unobserved_weight) * has_observation
    packed = torch.nn.functional.pixel_unshuffle(weight, SKY_PATCH_SIZE)
    return packed.permute(0, 2, 3, 1).reshape(observation_mask.shape[0], -1, SKY_TOKEN_DIM).contiguous()


def decode_sky_patch_tokens(tokens: torch.Tensor, *, standardized: bool = True) -> torch.Tensor:
    """Decode the sky token grid to a flattened [-1,1] RGB atlas.

    The inverse of :func:`pack_sky_rgb_atlas`, including its per-channel
    standardization: everything downstream of the generator -- the sky-view
    reconstruction loss, the 3DGS background, the mosaic -- reads RGB through
    here, so the un-standardization lives in exactly one place.  A checkpoint
    from an older sky representation carries a different token width and fails
    here rather than unpacking into a silently scrambled atlas, which is why
    the version is named in the error.
    """
    if tokens.ndim != 3 or int(tokens.shape[1]) != DEFAULT_SKY_GRID[0] * DEFAULT_SKY_GRID[1] or int(tokens.shape[2]) != SKY_TOKEN_DIM:
        raise ValueError(
            f"sky tokens must be [B,{DEFAULT_SKY_GRID[0] * DEFAULT_SKY_GRID[1]},"
            f"{SKY_TOKEN_DIM}] for {SKY_REPRESENTATION_VERSION}, "
            f"got {tuple(tokens.shape)}"
        )
    if standardized:
        mean, std = sky_token_channel_stats(device=tokens.device, dtype=tokens.dtype)
        tokens = tokens * std + mean
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
    candidate = (neg / pos.clamp_min(1.0)).clamp(1.0, float(pos_weight_max))
    return torch.where(
        pos.gt(0.0) & neg.gt(0.0),
        candidate,
        target.new_tensor(1.0),
    )


def sky_mask_patch_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float,
    pos_weight_max: float,
    log_prefix: str = "sky_mask",
    defer_log_values: bool = False,
    collect_logs: bool = True,
) -> tuple[torch.Tensor, dict[str, TrainLogValue]]:
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
    if not collect_logs:
        return loss, {}
    pred_hard = prob.ge(0.5)
    target_hard = target_f.ge(0.5)
    union = (pred_hard | target_hard).float().sum()
    iou = torch.where(
        union.eq(0.0),
        logits_f.new_tensor(1.0),
        (pred_hard & target_hard).float().sum() / union.clamp_min(1.0),
    ).detach()
    return loss, {
        f"loss_{log_prefix}_bce": deferred_log_value(
            bce, defer=defer_log_values
        ),
        f"loss_{log_prefix}_dice": deferred_log_value(
            dice, defer=defer_log_values
        ),
        f"{log_prefix}_pos_weight": deferred_log_value(
            pos_weight, defer=defer_log_values
        ),
        f"{log_prefix}_target_frac": deferred_log_value(
            target_f.mean(), defer=defer_log_values
        ),
        f"{log_prefix}_pred_frac": deferred_log_value(
            prob.mean(), defer=defer_log_values
        ),
        f"{log_prefix}_iou": deferred_log_value(
            iou, defer=defer_log_values
        ),
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
    defer_log_values: bool = False,
    collect_logs: bool = True,
) -> tuple[torch.Tensor, dict[str, TrainLogValue]]:
    region_loss, logs = sky_mask_patch_loss(
        logits,
        target,
        dice_weight=dice_weight,
        pos_weight_max=pos_weight_max,
        log_prefix="sky_mask_refine",
        defer_log_values=defer_log_values,
        collect_logs=collect_logs,
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
    band_sum = band.sum()
    boundary_bce = torch.where(
        band_sum.gt(0.0),
        (bce_px * band).sum() / band_sum.clamp_min(1.0),
        logits_f.sum() * 0.0,
    )
    loss = region_loss + float(boundary_loss_weight) * float(boundary_weight) * boundary_bce
    if collect_logs:
        logs["loss_sky_mask_refine_boundary_bce"] = deferred_log_value(
            boundary_bce, defer=defer_log_values
        )
        logs["sky_mask_refine_boundary_frac"] = deferred_log_value(
            band.mean(), defer=defer_log_values
        )
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
    valid_threshold: float = DEFAULT_SKY_VALID_THRESHOLD,
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
    valid_threshold: float = DEFAULT_SKY_VALID_THRESHOLD,
    extrinsics: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a sharp, directionally completed RGB atlas and observation mask.

    RGB is in [0,1] and shaped [B,3,H,W]. Unobserved cells are filled from
    spherical neighbors; the returned [B,1,H,W] mask distinguishes real RGB
    observations from low-confidence completion targets.
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
    atlas = complete_sky_atlas_spherical(atlas * observation, observation)
    return atlas.to(dtype=images.dtype), observation


def complete_sky_atlas_spherical(
    atlas: torch.Tensor,
    observation_mask: torch.Tensor,
) -> torch.Tensor:
    """Fill unobserved atlas cells from nearby observed sky directions.

    Longitude wraps circularly; elevation is bounded at zenith and horizon.
    The completion is only a low-confidence target. The original observation
    mask remains authoritative for assigning full supervision weight.
    """
    if atlas.ndim != 4 or int(atlas.shape[1]) != SKY_RGB_DIM:
        raise ValueError(f"sky atlas must be [B,3,H,W], got {tuple(atlas.shape)}")
    if observation_mask.shape != (atlas.shape[0], 1, atlas.shape[2], atlas.shape[3]):
        raise ValueError(
            f"sky observation mask shape {tuple(observation_mask.shape)} is incompatible "
            f"with atlas {tuple(atlas.shape)}"
        )

    values = atlas.float()
    valid = observation_mask.float().clamp(0.0, 1.0)
    height, width = int(atlas.shape[-2]), int(atlas.shape[-1])
    radius = 1
    while radius < max(height, width):
        value_sum = values * valid
        weight_sum = valid.clone()
        row_ids = torch.arange(height, device=atlas.device)
        for dy in (-radius, 0, radius):
            source_rows = (row_ids + dy).clamp(0, height - 1)
            shifted_values = values.index_select(-2, source_rows)
            shifted_valid = valid.index_select(-2, source_rows)
            for dx in (-radius, 0, radius):
                if dy == 0 and dx == 0:
                    continue
                value_sum = value_sum + torch.roll(shifted_values * shifted_valid, shifts=dx, dims=-1)
                weight_sum = weight_sum + torch.roll(shifted_valid, shifts=dx, dims=-1)
        fillable = valid.lt(0.5) & weight_sum.gt(0.0)
        proposal = value_sum / weight_sum.clamp_min(1.0)
        values = torch.where(fillable.expand_as(values), proposal, values)
        valid = torch.where(fillable, torch.ones_like(valid), valid)
        radius *= 2
    return values.to(dtype=atlas.dtype)


def build_sky_rectified_flow_target(
    sky_clean: torch.Tensor | None,
    video_target,
    loss_weight: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    observation: torch.Tensor | None = None,
) -> SimpleNamespace | None:
    if sky_clean is None:
        return None
    if sky_clean.ndim != 3 or int(sky_clean.shape[-1]) != SKY_TOKEN_DIM:
        raise ValueError(
            f"sky_clean must be latent [B,K,{SKY_TOKEN_DIM}], "
            f"got {tuple(sky_clean.shape)}"
        )
    b = int(sky_clean.shape[0])
    if video_target.sigmas.shape != (b,):
        raise ValueError(f"video_target sigmas shape {tuple(video_target.sigmas.shape)} != {(b,)}")
    if loss_weight is None:
        token_weight = torch.ones(sky_clean.shape[:2], device=sky_clean.device, dtype=sky_clean.dtype)
    else:
        token_weight = loss_weight.to(device=sky_clean.device, dtype=sky_clean.dtype).clamp_min(0.0)
        valid_shapes = (sky_clean.shape[:2], sky_clean.shape)
        if token_weight.shape not in valid_shapes:
            raise ValueError(
                f"sky loss weight shape {tuple(token_weight.shape)} must be token-level "
                f"{tuple(sky_clean.shape[:2])} or channel-level {tuple(sky_clean.shape)}"
            )
    if not bool(token_weight.gt(0.0).any().item()):
        return None
    sigmas = video_target.sigmas.to(device=sky_clean.device, dtype=sky_clean.dtype)
    sigmas3 = sigmas.view(b, 1, 1)
    eps = torch.empty_like(sky_clean)
    eps.normal_(generator=generator)
    z_t = (1.0 - sigmas3) * sky_clean + sigmas3 * eps
    target_t_eps = float(getattr(video_target, "t_eps", SKY_FLOW_T_EPS))
    # Match the x-prediction -> velocity conversion exactly.  In particular,
    # both sides use the same denominator floor below t_eps, which caps the
    # effective clean-prediction error weight at 1 / t_eps**2.
    v_gt = (z_t - sky_clean) / sigmas3.clamp_min(target_t_eps)
    return SimpleNamespace(
        sigmas=video_target.sigmas,
        sigmas4=sigmas3,
        z_t=z_t,
        v_gt=v_gt,
        eps=eps,
        weights=torch.ones((b, 1, 1), device=sky_clean.device, dtype=sky_clean.dtype),
        t_eps=target_t_eps,
        loss_weight=token_weight,
        observation=(
            None
            if observation is None
            else observation.to(device=sky_clean.device, dtype=sky_clean.dtype).clamp(0.0, 1.0)
        ),
    )


def sky_flow_loss(
    v_sky_pred: torch.Tensor,
    sky_target,
    sky_clean: torch.Tensor,
    *,
    unobserved_beta: float = DEFAULT_SKY_UNOBSERVED_LOSS_BETA,
) -> torch.Tensor:
    """Flow loss on the sky atlas, normalized separately on each region.

    When ``sky_target`` carries an ``observation`` map the observed atlas and
    the spherical-completion prior get their own denominators, so the observed
    sky's share of the gradient is ``1 / (1 + beta)`` no matter how much sky
    the clip contains.  Without it this falls back to the single weighted mean,
    whose split moves with the clip's sky fraction.
    """
    del sky_clean
    beta = float(unobserved_beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError(f"unobserved sky loss beta must be finite and non-negative, got {unobserved_beta}")
    token_weight = getattr(sky_target, "loss_weight", None)
    if token_weight is None:
        return torch.nn.functional.mse_loss(
            v_sky_pred.float(),
            sky_target.v_gt.to(device=v_sky_pred.device, dtype=torch.float32),
        )

    def _as_channel_map(value: torch.Tensor, name: str) -> torch.Tensor:
        tensor = value.to(device=v_sky_pred.device, dtype=torch.float32).clamp_min(0.0)
        if tensor.shape == v_sky_pred.shape[:2]:
            return tensor.unsqueeze(-1).expand_as(v_sky_pred)
        if tensor.shape != v_sky_pred.shape:
            raise ValueError(
                f"sky {name} shape {tuple(tensor.shape)} must be token-level "
                f"{tuple(v_sky_pred.shape[:2])} or channel-level {tuple(v_sky_pred.shape)}"
            )
        return tensor

    weight = _as_channel_map(token_weight, "loss weight")
    if not bool(weight.gt(0.0).any().item()):
        return v_sky_pred.sum() * 0.0
    diff = (
        v_sky_pred.float()
        - sky_target.v_gt.to(device=v_sky_pred.device, dtype=torch.float32)
    ).square()

    observation = getattr(sky_target, "observation", None)
    if observation is None:
        denom = weight.sum().clamp_min(1e-6)
        return (diff * weight).sum() / denom

    observed = _as_channel_map(observation, "observation").clamp(0.0, 1.0)
    # A clip with no valid sky observation supervises nothing at all, exactly
    # as the blended weight does; without this the completion prior would be
    # the only teacher precisely where there is no measurement to complete.
    has_observation = observed.flatten(1).amax(dim=1).gt(0.0).float().view(-1, *([1] * (observed.ndim - 1)))
    observed = observed * has_observation
    unobserved = (1.0 - observed) * has_observation

    observed_loss = (diff * observed).sum() / observed.sum().clamp_min(1e-6)
    if beta == 0.0:
        return observed_loss
    unobserved_loss = (diff * unobserved).sum() / unobserved.sum().clamp_min(1e-6)
    return observed_loss + beta * unobserved_loss


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
    # Exactly 3: this takes the decoded per-cell RGB atlas.  A packed patch
    # token is laid out as ``c * patch**2 + sub``, so slicing [..., :3] off one
    # would read three *red* subpixels and render a wrong sky that still looks
    # plausible.  The twin in dggt/losses/rgb_render_loss.py enforces the same.
    if sky_tokens.ndim != 3 or int(sky_tokens.shape[-1]) != SKY_RGB_DIM:
        raise ValueError(
            "sky_tokens must be the decoded RGB atlas [B,K,3]; decode patch "
            f"tokens before rendering, got {tuple(sky_tokens.shape)}"
        )
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
    # Exactly 3: this takes the decoded per-cell RGB atlas.  A packed patch
    # token is laid out as ``c * patch**2 + sub``, so slicing [..., :3] off one
    # would read three *red* subpixels and render a wrong sky that still looks
    # plausible.  The twin in dggt/losses/rgb_render_loss.py enforces the same.
    if sky_tokens.ndim != 3 or int(sky_tokens.shape[-1]) != SKY_RGB_DIM:
        raise ValueError(
            "sky_tokens must be the decoded RGB atlas [B,K,3]; decode patch "
            f"tokens before rendering, got {tuple(sky_tokens.shape)}"
        )
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
        weight = torch.ones_like(pred, dtype=torch.float32)
    else:
        weight = loss_weight.to(device=pred.device, dtype=torch.float32).clamp_min(0.0)
        if weight.shape == pred.shape[:2]:
            weight = weight.unsqueeze(-1).expand_as(pred)
        elif weight.shape != pred.shape:
            raise ValueError(
                f"sky loss weight shape {tuple(weight.shape)} must be token-level "
                f"{tuple(pred.shape[:2])} or channel-level {tuple(pred.shape)}"
            )
    if not bool(weight.gt(0.0).any().item()):
        return {
            f"{prefix}_rgb_mae": 0.0,
            f"{prefix}_weight_mean": 0.0,
        }
    # Report in [0,1] RGB, not in whatever units the token currently uses.
    # The packed token is standardized per channel and ``* 0.5`` undoes the
    # ``rgb * 2 - 1`` packing, so this number stays comparable across sky
    # representation versions instead of moving when the scale changes.
    _, channel_std = sky_token_channel_stats(device=pred.device, dtype=torch.float32)
    scale = channel_std * 0.5
    diff = (pred.float() - gt.to(device=pred.device, dtype=torch.float32)).abs() * scale
    diff = diff * weight.to(dtype=diff.dtype)
    return {
        f"{prefix}_rgb_mae": float((diff.sum() / weight.sum().clamp_min(1e-6)).detach().item()),
        f"{prefix}_weight_mean": float(weight.mean().detach().item()),
    }


def generated_sky_view_reconstruction_loss(
    *,
    vggt_model: VGGT,
    sky_latent: torch.Tensor,
    images: torch.Tensor,
    sky_mask: torch.Tensor,
    render_pose_enc_dggt: torch.Tensor,
    lpips_model: nn.Module | None = None,
    lpips_weight: float = 0.01,
    high_frequency_weight: float = 0.25,
    loss_sample_weight: torch.Tensor | None = None,
    defer_log_values: bool = False,
    collect_logs: bool = True,
) -> tuple[torch.Tensor, dict[str, TrainLogValue]]:
    """Reproject generated sky through requested C and the predicted gauge."""
    del vggt_model
    rgb_tokens = decode_sky_patch_tokens(sky_latent)
    atlas_hw = DEFAULT_SKY_ATLAS_HW
    mask = _sky_mask_1ch(sky_mask, images).float()
    total = sky_latent.sum() * 0.0
    charbonnier_total = sky_latent.new_zeros((), dtype=torch.float32)
    high_total = sky_latent.new_zeros((), dtype=torch.float32)
    detail_ratio_total = sky_latent.new_zeros((), dtype=torch.float32)
    lpips_total = sky_latent.new_zeros((), dtype=torch.float32)
    if loss_sample_weight is None:
        sample_weight = torch.ones(
            (int(images.shape[0]),),
            device=sky_latent.device,
            dtype=torch.float32,
        )
    else:
        if (
            loss_sample_weight.ndim != 1
            or int(loss_sample_weight.shape[0]) != int(images.shape[0])
        ):
            raise ValueError(
                "sky-view loss_sample_weight must be [B], got "
                f"{tuple(loss_sample_weight.shape)} for B={int(images.shape[0])}"
            )
        sample_weight = loss_sample_weight.to(device=sky_latent.device, dtype=torch.float32)
        if not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0.0).any()):
            raise ValueError("sky-view loss_sample_weight must contain finite non-negative values")
    for row in range(int(images.shape[0])):
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            render_pose_enc_dggt[row : row + 1].float(),
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
        # ``high`` is |grad(pred) - grad(target)| and cannot tell a sky that is
        # too smooth from one that is too noisy -- both raise it.  This ratio
        # can: below 1 the generated sky carries less gradient than the real
        # one (the flat-tile failure), above 1 it carries more (speckle inside
        # a tile, the failure mode on the other side of a heavier sky weight).
        with torch.no_grad():
            pred_grad = (
                ((background[..., :, 1:] - background[..., :, :-1]).abs() * wx).sum()
                + ((background[..., 1:, :] - background[..., :-1, :]).abs() * wy).sum()
            )
            target_grad = (
                ((target[..., :, 1:] - target[..., :, :-1]).abs() * wx).sum()
                + ((target[..., 1:, :] - target[..., :-1, :]).abs() * wy).sum()
            )
            detail_ratio = pred_grad / target_grad.clamp_min(1e-6)
        row_loss = charb + float(high_frequency_weight) * high
        lpips_value = background.new_zeros(())
        if lpips_model is not None and float(lpips_weight) > 0.0:
            pred_masked = weight * background + (1.0 - weight) * 0.5
            target_masked = weight * target + (1.0 - weight) * 0.5
            lpips_value = lpips_model(pred_masked * 2.0 - 1.0, target_masked * 2.0 - 1.0).mean()
            row_loss = row_loss + float(lpips_weight) * lpips_value
        total = total + sample_weight[row] * row_loss / float(images.shape[0])
        if collect_logs:
            charbonnier_total = charbonnier_total + charb.detach().float() / float(images.shape[0])
            high_total = high_total + high.detach().float() / float(images.shape[0])
            detail_ratio_total = (
                detail_ratio_total + detail_ratio.detach().float() / float(images.shape[0])
            )
            lpips_total = lpips_total + lpips_value.detach().float() / float(images.shape[0])
    if not collect_logs:
        return total, {}
    return total, {
        "loss_sky_view_charbonnier": deferred_log_value(
            charbonnier_total, defer=defer_log_values
        ),
        "loss_sky_view_high_frequency": deferred_log_value(
            high_total, defer=defer_log_values
        ),
        "sky_view_detail_ratio": deferred_log_value(
            detail_ratio_total, defer=defer_log_values
        ),
        "loss_sky_view_lpips": deferred_log_value(
            lpips_total, defer=defer_log_values
        ),
        "sky_view_sample_weight_mean": deferred_log_value(
            sample_weight.mean(), defer=defer_log_values
        ),
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
    Background: generated/pretrained paths may use the sky model; formal
    editing uses ``background_mode="gt_sky"`` and explicitly composites the
    original input RGB only inside the GT sky mask. Unlike the legacy
    ``inference.py`` renderer, gsplat's premultiplied RGB is not multiplied by
    alpha twice.

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

    preserve_original_sky = background_mode == "gt_sky"
    if preserve_original_sky and (images is None or masks is None or not use_sky_mask):
        raise ValueError(
            "background_mode='gt_sky' requires input images, GT sky masks, and use_sky_mask=True."
        )
    if preserve_original_sky and background_override is not None:
        raise ValueError("background_mode='gt_sky' cannot be combined with background_override.")

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
    elif preserve_original_sky:
        # Do not use the full GT image as a rasterizer background: that would
        # leak original non-sky content through transparent edited Gaussians.
        # The exact GT-sky blend is applied after rasterization below.
        bg_render = torch.zeros(
            (seq_len, height, width, 3), dtype=torch.float32, device=device
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
        if preserve_original_sky:
            gt_rgb = images[0, frame_idx : frame_idx + 1].float().permute(0, 2, 3, 1)
            gt_sky = sky_probability[0, frame_idx : frame_idx + 1].unsqueeze(-1)
            composed = composite_original_sky(composed, gt_rgb, gt_sky)
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
    decoded_patch_tokens = decode_tokenizer_windowed(
        vggt_model.scene_tokenizer,
        z_generated,
        patch_grid=args.patch_grid,
        window_len=_tokenizer_window_len(scene_flow, args),
    )
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
def render_validation_generated_rgb(
    batch: dict[str, Any] | None,
    vggt_model: VGGT,
    scene_flow: nn.Module,
    z_generated_raw_n: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    requested_camera_to_world: torch.Tensor,
    trajectory_anchor_to_world: torch.Tensor,
    generated_gauge: torch.Tensor,
    generated_sky_tokens: torch.Tensor | None = None,
    generated_sky_mask_patch: torch.Tensor | None = None,
    generated_sky_mask_refined: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Render generated latents with the requested camera C and sampled gauge."""

    if not torch.is_tensor(requested_camera_to_world):
        raise RuntimeError("generated RGB validation requires requested camera C")
    if not torch.is_tensor(trajectory_anchor_to_world):
        raise RuntimeError("generated RGB validation requires the trajectory anchor")
    if not torch.is_tensor(generated_gauge):
        raise RuntimeError("generated RGB validation requires generated gauge tokens")
    if generated_sky_mask_patch is None:
        raise RuntimeError("generated RGB validation requires a generated sky mask")
    batch_size, seq_len = (
        int(z_generated_raw_n.shape[0]),
        int(z_generated_raw_n.shape[1]),
    )
    if tuple(requested_camera_to_world.shape[:2]) != (batch_size, seq_len):
        raise ValueError(
            "requested camera C must match generated [B,S], got "
            f"{tuple(requested_camera_to_world.shape)}"
        )
    height, width = _fixed_render_hw(args)
    frames = min(int(args.val_log_images), seq_len)
    timestamps = _timestamps_for_generated_render(
        batch, seq_len=seq_len, device=device
    )
    patch_start_idx = int(getattr(vggt_model.aggregator, "patch_start_idx", 5))
    with autocast_context(args, device):
        geometry = decode_generated_dggt_geometry(
            vggt_model=vggt_model,
            scene_flow_root=unwrap_ddp(scene_flow),
            z_clean_pred_n=z_generated_raw_n,
            patch_grid=args.patch_grid,
            patch_start_idx=patch_start_idx,
            image_hw=(height, width),
            pullback_calibration=getattr(
                unwrap_ddp(scene_flow), "_pullback_calibration", None
            ),
        )
        generated_sky_mask = _sky_mask_patch_to_image(
            (
                generated_sky_mask_refined
                if generated_sky_mask_refined is not None
                else generated_sky_mask_patch
            ),
            patch_grid=args.patch_grid,
            height=height,
            width=width,
            device=device,
        )
    generated_pose_enc = requested_render_pose_encoding(
        requested_camera_to_world.to(device=device, dtype=torch.float32),
        trajectory_anchor_to_world.to(device=device, dtype=torch.float32),
        generated_gauge.to(device=device, dtype=torch.float32),
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
        sky_grid_image = _sky_background_image_grid(
            generated_sky_background, frames
        )
    result = {
        "generated_pred_sky_mask": _sky_mask_image_grid(
            generated_sky_mask, frames
        ),
        "generated_raw_3dgs_rgb": _render_gs_map_rgb(
            vggt_model,
            None,
            generated_sky_mask,
            timestamps,
            generated_pose_enc,
            generated_depth,
            raw_gs_map,
            raw_gs_conf,
            generated_dynamic_conf,
            device,
            frames,
            background_mode="black",
            use_sky_mask=True,
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


def _validation_sliding_params(args: argparse.Namespace, seq_len: int) -> tuple[int, int] | None:
    requested_window = int(getattr(args, "val_sliding_window", 0) or 0)
    if requested_window <= 0:
        return None
    window = min(requested_window, OFFLINE_MAX_SINGLE_WINDOW, int(seq_len))
    if int(seq_len) <= window:
        return None
    stride = int(getattr(args, "val_sliding_stride", 0) or 0)
    if stride <= 0:
        stride = default_window_stride(window)
    if stride >= window:
        raise ValueError(
            f"validation sliding requires stride < bounded window, got {stride} >= {window}"
        )
    return min(window, int(seq_len)), max(1, stride)


def pretrain_validation_window_offsets(
    sequence_length: int,
    stride: int,
    *,
    trunk_frames: int = 29,
) -> tuple[int, ...]:
    """Return the clip-global starts used by local pretrain validation.

    This intentionally shares the same window schedule as long-form sampling,
    so validation loss sees both the frame-0 anchor window and delta-only
    windows instead of repeatedly evaluating only the first crop.
    """
    return tuple(
        start
        for start, _ in window_slices(
            int(trunk_frames),
            int(sequence_length),
            int(stride),
        )
    )


def pretrain_validation_stride(sequence_length: int, requested_stride: int) -> int:
    """Resolve an overlapping validation stride for the training window size."""
    window = int(sequence_length)
    stride = int(requested_stride)
    if stride <= 0 or stride >= window:
        return default_window_stride(window)
    return stride


@torch.no_grad()
def _appearance_present(layout: LayoutConditionBatch) -> bool:
    return bool(
        (
            layout.appearance.binding_valid
            & (layout.appearance.appearance_mode[:, None] == int(AppearanceMode.REAL))
        ).any().item()
    )


def _layout_cfg_branch(
    layout: LayoutConditionBatch,
    appearance_class_id: torch.Tensor,
    branch: str,
) -> tuple[LayoutConditionBatch, torch.Tensor]:
    if branch in {"full", "no_text_full"}:
        return layout, appearance_class_id
    if branch == "appearance_dropped":
        return (
            layout.without_appearance(),
            torch.full_like(appearance_class_id, -1),
        )
    if branch == "layout_dropped":
        return (
            layout.without_layout(),
            torch.full_like(appearance_class_id, -1),
        )
    raise ValueError(f"unknown chained-CFG branch {branch!r}")


def _layout_cfg_scales(
    args: argparse.Namespace,
    guidance_scale: float | None,
) -> tuple[float, float, float]:
    text_scale = (
        float(getattr(args, "guidance_scale", 1.0))
        if guidance_scale is None
        else float(guidance_scale)
    )
    return (
        text_scale,
        float(getattr(args, "layout_guidance_scale", 1.0)),
        float(getattr(args, "asset_control_guidance_scale", 1.0)),
    )


def validation_layout_to_gauge_grad_scale(args: argparse.Namespace) -> float:
    """Use the configured model cap for validation's no-grad layout read."""

    scale = float(getattr(args, "layout_to_gauge_grad_scale", 1.0))
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("layout_to_gauge_grad_scale must be finite and in [0,1]")
    return scale


def _combine_layout_cfg_outputs(
    outputs: dict[str, dict[str, torch.Tensor]],
    *,
    text_scale: float,
    layout_scale: float,
    appearance_scale: float,
    appearance_present: bool,
) -> dict[str, torch.Tensor]:
    return combine_chained_cfg(
        outputs,
        text_scale=text_scale,
        layout_scale=layout_scale,
        appearance_scale=appearance_scale,
        appearance_present=appearance_present,
    )


VALIDATION_SCENE_SAMPLING_SEED_OFFSET = 20_000_033


def validation_scene_sampling_seed(base_seed: int, scene_offset: int) -> int:
    """Stable validation noise seed shared by every CFG scale of one slot."""

    if int(scene_offset) < 0:
        raise ValueError(f"scene_offset must be non-negative, got {scene_offset}")
    return (
        int(base_seed)
        + VALIDATION_SCENE_SAMPLING_SEED_OFFSET
        + int(scene_offset)
    )


def make_pretrain_sampling_generator(
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    *,
    sampling_seed: int | None = None,
) -> torch.Generator:
    """Build the ODE noise generator, optionally overriding the legacy step seed."""

    generator = torch.Generator(device=device)
    seed = (
        int(args.seed) + int(step)
        if sampling_seed is None
        else int(sampling_seed)
    )
    generator.manual_seed(seed)
    return generator


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
    return_sky: bool = False,
    return_gauge: bool = False,
    return_sky_mask: bool = False,
    sampling_seed: int | None = None,
) -> torch.Tensor | SimpleNamespace:
    """Chained-CFG sliding sampler with the full layout bundle sliced per window."""

    t_steps = rae_t_grid(
        num_steps=args.val_sample_steps,
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
        t_eps=scene_flow_t_eps(scene_flow),
    )
    generator = make_pretrain_sampling_generator(
        device, args, step, sampling_seed=sampling_seed
    )
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
    gauge_z = _init_pretrain_gauge_noise(
        scene_flow, bundle, generator, return_gauge=True
    )
    if gauge_z is None:
        raise RuntimeError("layout-v2 sampling requires a generated gauge state")
    sky_z = None
    if return_sky or torch.is_tensor(getattr(bundle, "sky_gen_clean", None)):
        sky_h, sky_w = sky_grid_shape(args)
        sky_z = bundle.z_clean_n.new_empty(
            (int(bundle.z_clean_n.shape[0]), int(sky_h * sky_w), SKY_TOKEN_DIM)
        )
        sky_z.normal_(generator=generator)

    layout = getattr(bundle, "layout_condition", None)
    appearance_class_id = getattr(bundle, "appearance_class_id", None)
    if not isinstance(layout, LayoutConditionBatch) or not torch.is_tensor(appearance_class_id):
        raise RuntimeError("layout-v2 sampling requires LayoutConditionBatch and A class ids")
    appearance_present = _appearance_present(layout)
    text_scale, layout_scale, appearance_scale = _layout_cfg_scales(
        args, guidance_scale
    )
    gauge_grad_scale = validation_layout_to_gauge_grad_scale(args)
    branches = required_cfg_branches(
        text_scale=text_scale,
        layout_scale=layout_scale,
        appearance_scale=appearance_scale,
        appearance_present=appearance_present,
    )
    batch_size, seq_len = int(z.shape[0]), int(z.shape[1])
    text_tokens, text_mask = encode_text_condition(
        text_encoder, getattr(bundle, "captions", None)
    )
    text_null, text_null_mask = encode_text_condition(
        text_encoder, [""] * batch_size if text_tokens is not None else None
    )
    frame_ids_full = _bundle_frame_ids(
        bundle, batch_size=batch_size, seq_len=seq_len, device=device
    )
    windows = window_slices(seq_len, window, stride)
    coverage = cosine_coverage(seq_len, windows, device=device, dtype=z.dtype)
    sf = unwrap_ddp(scene_flow)
    # Window slicing is a pure function of the outer layout, so hoist it out of
    # the ODE loop: it clones the raster and can rebuild every projected field,
    # and the loop would otherwise repeat identical work once per step.
    window_layouts = {
        (start, end): layout.slice_frames(slice(start, end))
        for start, end in windows
    }

    def run_window_branches(
        *,
        start: int,
        end: int,
        sigma: torch.Tensor,
        request_sky_mask: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        layout_window = window_layouts[(start, end)]
        camera_tokens = _slice_time(
            bundle.camera_condition_tokens, start, end, seq_len
        )
        camera_mask = _slice_time(bundle.camera_attention_mask, start, end, seq_len)
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for branch in branches:
            branch_layout, branch_class_id = _layout_cfg_branch(
                layout_window, appearance_class_id, branch
            )
            branch_text = text_null if branch == "no_text_full" else text_tokens
            branch_text_mask = (
                text_null_mask if branch == "no_text_full" else text_mask
            )
            result = sf(
                z[:, start:end],
                sigma,
                z_splat[:, start:end],
                scaffold_tok[:, start:end],
                bundle.M_preserve[:, start:end],
                bundle.M_source[:, start:end],
                bundle.M_dest[:, start:end],
                **layout_model_kwargs(
                    branch_layout,
                    branch_class_id,
                    gauge_grad_scale=gauge_grad_scale,
                ),
                text_tokens=branch_text,
                text_attention_mask=branch_text_mask,
                camera_condition_tokens=camera_tokens,
                camera_attention_mask=camera_mask,
                sky_gen_tokens=sky_z,
                sky_gen_attention_mask=None,
                gauge_gen_tokens=gauge_z,
                gauge_gen_attention_mask=None,
                return_mid=False,
                return_dict=True,
                return_sky_mask=request_sky_mask,
                frame_ids=frame_ids_full[:, start:end],
                fps=getattr(bundle, "fps", None),
            )
            if not isinstance(result, dict):
                raise RuntimeError("SceneFlow chained-CFG branch must return a dict")
            outputs[branch] = result
        return outputs

    mask_logits = None
    refined_logits = None
    sample_steps = int(args.val_sample_steps)
    for index in range(sample_steps):
        step_h = t_steps[index] - t_steps[index + 1]
        sigma = torch.full(
            (batch_size,), float(t_steps[index].item()), device=device
        )
        capture_sky_mask = bool(return_sky_mask and index == sample_steps - 1)
        video_acc = torch.zeros_like(z)
        video_weight = torch.zeros(
            (1, seq_len, 1, 1), device=device, dtype=z.dtype
        )
        sky_acc = torch.zeros_like(sky_z) if sky_z is not None else None
        gauge_acc = torch.zeros_like(gauge_z)
        sky_weight = 0.0
        gauge_weight = 0.0
        patch_acc = (
            torch.zeros(z.shape[:3] + (1,), device=device, dtype=z.dtype)
            if capture_sky_mask
            else None
        )
        patch_weight = (
            torch.zeros((1, seq_len, 1, 1), device=device, dtype=z.dtype)
            if capture_sky_mask
            else None
        )
        refined_acc = None
        refined_weight = None
        for start, end in windows:
            actual = int(end - start)
            weights = cosine_window(actual, device=device, dtype=z.dtype).view(
                1, actual, 1, 1
            )
            branch_outputs = run_window_branches(
                start=start,
                end=end,
                sigma=sigma,
                request_sky_mask=capture_sky_mask,
            )
            combined = _combine_layout_cfg_outputs(
                branch_outputs,
                text_scale=text_scale,
                layout_scale=layout_scale,
                appearance_scale=appearance_scale,
                appearance_present=appearance_present,
            )
            if capture_sky_mask:
                patch = combined.get("sky_mask_logits")
                refined = combined.get("sky_mask_refined_logits")
                if not torch.is_tensor(patch) or not torch.is_tensor(refined):
                    raise RuntimeError(
                        "final ODE step did not return both sky-mask logits"
                    )
                if patch_acc is None or patch_weight is None:
                    raise RuntimeError("sliding sky-mask accumulators were not initialized")
                patch_acc[:, start:end] += patch.to(patch_acc.dtype) * weights
                patch_weight[:, start:end] += weights
                if refined_acc is None:
                    refined_acc = torch.zeros(
                        (batch_size, seq_len) + tuple(refined.shape[2:]),
                        device=device,
                        dtype=refined.dtype,
                    )
                    refined_weight = torch.zeros(
                        (1, seq_len, 1, 1, 1),
                        device=device,
                        dtype=refined.dtype,
                    )
                refined_window_weight = weights.view(1, actual, 1, 1, 1).to(
                    refined.dtype
                )
                refined_acc[:, start:end] += refined * refined_window_weight
                refined_weight[:, start:end] += refined_window_weight
            video_state = z[:, start:end]
            video_velocity = sampler_prediction_to_velocity(
                sf, combined["video"], video_state, sigma
            )
            video_acc[:, start:end] += video_velocity * weights
            video_weight[:, start:end] += weights
            global_weight = scene_global_window_weight(start, end, coverage)
            if sky_acc is not None and torch.is_tensor(combined.get("sky")):
                sky_velocity = sampler_prediction_to_velocity(
                    sf, combined["sky"], sky_z, sigma
                )
                sky_acc += sky_velocity * global_weight.to(dtype=sky_velocity.dtype)
                sky_weight += float(global_weight.item())
            gauge_velocity = sampler_prediction_to_velocity(
                sf, combined["gauge"], gauge_z, sigma
            )
            gauge_acc += gauge_velocity * global_weight.to(dtype=gauge_velocity.dtype)
            gauge_weight += float(global_weight.item())
        if capture_sky_mask:
            if (
                patch_acc is None
                or patch_weight is None
                or refined_acc is None
                or refined_weight is None
            ):
                raise RuntimeError(
                    "final ODE step produced no sliding-window sky-mask output"
                )
            mask_logits = patch_acc / patch_weight.clamp_min(1.0e-6)
            refined_logits = refined_acc / refined_weight.clamp_min(1.0e-6)
        z = z - step_h.to(dtype=z.dtype) * (
            video_acc / video_weight.clamp_min(1.0e-6)
        )
        z = M_keep * z_splat + M_edit * z
        if sky_z is not None and sky_acc is not None and sky_weight > 0.0:
            sky_z = sky_z - step_h.to(dtype=sky_z.dtype) * (sky_acc / sky_weight)
        if gauge_weight <= 0.0:
            raise RuntimeError("sliding gauge aggregation received zero window weight")
        gauge_z = gauge_z - step_h.to(dtype=gauge_z.dtype) * (
            gauge_acc / gauge_weight
        )

    z = M_keep * z_splat + M_edit * z
    mask_patch = (
        None
        if mask_logits is None
        else torch.sigmoid(mask_logits.float()).to(mask_logits.dtype)
    )
    mask_refined = (
        None
        if refined_logits is None
        else torch.sigmoid(refined_logits.float()).to(refined_logits.dtype)
    )
    if return_sky_mask and (mask_patch is None or mask_refined is None):
        raise RuntimeError("requested sky mask was not produced")
    if return_sky or return_gauge or return_sky_mask:
        return SimpleNamespace(
            video=z,
            sky=sky_z,
            gauge=sf.denormalize_gauge(gauge_z),
            sky_mask_logits=mask_logits,
            sky_mask_patch=mask_patch,
            sky_mask_refined_logits=refined_logits,
            sky_mask_refined=mask_refined,
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
    return_sky: bool = False,
    return_gauge: bool = False,
    return_sky_mask: bool = False,
    sampling_seed: int | None = None,
) -> torch.Tensor | SimpleNamespace:
    """Sample video/sky/gauge with lazy four-branch chained CFG."""

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
            return_sky=return_sky,
            return_gauge=return_gauge,
            return_sky_mask=return_sky_mask,
            sampling_seed=sampling_seed,
        )

    t_steps = rae_t_grid(
        num_steps=args.val_sample_steps,
        time_shift=float(args.shift),
        device=device,
        dtype=torch.float32,
        t_eps=scene_flow_t_eps(scene_flow),
    )
    generator = make_pretrain_sampling_generator(
        device, args, step, sampling_seed=sampling_seed
    )
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
    gauge_z = _init_pretrain_gauge_noise(
        scene_flow, bundle, generator, return_gauge=True
    )
    if gauge_z is None:
        raise RuntimeError("layout-v2 sampling requires a generated gauge state")
    sky_z = None
    if return_sky or torch.is_tensor(getattr(bundle, "sky_gen_clean", None)):
        sky_h, sky_w = sky_grid_shape(args)
        sky_z = bundle.z_clean_n.new_empty(
            (int(bundle.z_clean_n.shape[0]), int(sky_h * sky_w), SKY_TOKEN_DIM)
        )
        sky_z.normal_(generator=generator)

    layout = getattr(bundle, "layout_condition", None)
    appearance_class_id = getattr(bundle, "appearance_class_id", None)
    if not isinstance(layout, LayoutConditionBatch) or not torch.is_tensor(appearance_class_id):
        raise RuntimeError("layout-v2 sampling requires LayoutConditionBatch and A class ids")
    appearance_present = _appearance_present(layout)
    text_scale, layout_scale, appearance_scale = _layout_cfg_scales(
        args, guidance_scale
    )
    gauge_grad_scale = validation_layout_to_gauge_grad_scale(args)
    branches = required_cfg_branches(
        text_scale=text_scale,
        layout_scale=layout_scale,
        appearance_scale=appearance_scale,
        appearance_present=appearance_present,
    )
    batch_size = int(z.shape[0])
    text_tokens, text_mask = encode_text_condition(
        text_encoder, getattr(bundle, "captions", None)
    )
    text_null, text_null_mask = encode_text_condition(
        text_encoder, [""] * batch_size if text_tokens is not None else None
    )
    frame_ids = _bundle_frame_ids(
        bundle,
        batch_size=batch_size,
        seq_len=int(z.shape[1]),
        device=device,
    )
    sf = unwrap_ddp(scene_flow)

    def run_branches(
        sigma: torch.Tensor,
        *,
        request_sky_mask: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for branch in branches:
            branch_layout, branch_class_id = _layout_cfg_branch(
                layout, appearance_class_id, branch
            )
            branch_text = text_null if branch == "no_text_full" else text_tokens
            branch_text_mask = (
                text_null_mask if branch == "no_text_full" else text_mask
            )
            result = sf(
                z,
                sigma,
                z_splat,
                scaffold_tok,
                bundle.M_preserve,
                bundle.M_source,
                bundle.M_dest,
                **layout_model_kwargs(
                    branch_layout,
                    branch_class_id,
                    gauge_grad_scale=gauge_grad_scale,
                ),
                text_tokens=branch_text,
                text_attention_mask=branch_text_mask,
                camera_condition_tokens=bundle.camera_condition_tokens,
                camera_attention_mask=bundle.camera_attention_mask,
                sky_gen_tokens=sky_z,
                sky_gen_attention_mask=None,
                gauge_gen_tokens=gauge_z,
                gauge_gen_attention_mask=None,
                return_mid=False,
                return_dict=True,
                return_sky_mask=request_sky_mask,
                frame_ids=frame_ids,
                fps=getattr(bundle, "fps", None),
            )
            if not isinstance(result, dict):
                raise RuntimeError("SceneFlow chained-CFG branch must return a dict")
            outputs[branch] = result
        return outputs

    mask_logits = None
    refined_logits = None
    sample_steps = int(args.val_sample_steps)
    for index in range(sample_steps):
        step_h = t_steps[index] - t_steps[index + 1]
        sigma = torch.full(
            (batch_size,), float(t_steps[index].item()), device=device
        )
        capture_sky_mask = bool(return_sky_mask and index == sample_steps - 1)
        combined = _combine_layout_cfg_outputs(
            run_branches(sigma, request_sky_mask=capture_sky_mask),
            text_scale=text_scale,
            layout_scale=layout_scale,
            appearance_scale=appearance_scale,
            appearance_present=appearance_present,
        )
        if capture_sky_mask:
            mask_logits = combined.get("sky_mask_logits")
            refined_logits = combined.get("sky_mask_refined_logits")
            if not torch.is_tensor(mask_logits) or not torch.is_tensor(
                refined_logits
            ):
                raise RuntimeError(
                    "final ODE step did not return both sky-mask logits"
                )
        video_velocity = sampler_prediction_to_velocity(
            sf, combined["video"], z, sigma
        )
        z = z - step_h.to(dtype=z.dtype) * video_velocity
        z = M_keep * z_splat + M_edit * z
        if sky_z is not None and torch.is_tensor(combined.get("sky")):
            sky_velocity = sampler_prediction_to_velocity(
                sf, combined["sky"], sky_z, sigma
            )
            sky_z = sky_z - step_h.to(dtype=sky_z.dtype) * sky_velocity
        gauge_velocity = sampler_prediction_to_velocity(
            sf, combined["gauge"], gauge_z, sigma
        )
        gauge_z = gauge_z - step_h.to(dtype=gauge_z.dtype) * gauge_velocity

    z = M_keep * z_splat + M_edit * z
    mask_patch = (
        None
        if mask_logits is None
        else torch.sigmoid(mask_logits.float()).to(mask_logits.dtype)
    )
    mask_refined = (
        None
        if refined_logits is None
        else torch.sigmoid(refined_logits.float()).to(refined_logits.dtype)
    )
    if return_sky_mask and (mask_patch is None or mask_refined is None):
        raise RuntimeError("requested sky mask was not produced")
    if return_sky or return_gauge or return_sky_mask:
        return SimpleNamespace(
            video=z,
            sky=sky_z,
            gauge=sf.denormalize_gauge(gauge_z),
            sky_mask_logits=mask_logits,
            sky_mask_patch=mask_patch,
            sky_mask_refined_logits=refined_logits,
            sky_mask_refined=mask_refined,
        )
    return z


def build_full_scene_bundle(
    z_clean_n: torch.Tensor,
    camera_condition_tokens: torch.Tensor | None = None,
    camera_attention_mask: torch.Tensor | None = None,
    sky_gen_clean: torch.Tensor | None = None,
    sky_gen_loss_weight: torch.Tensor | None = None,
    sky_gen_observation: torch.Tensor | None = None,
    sky_gen_attention_mask: torch.Tensor | None = None,
    scene_gauge_clean_n: torch.Tensor | None = None,
    scene_gauge_clean: torch.Tensor | None = None,
    scene_gauge_valid: torch.Tensor | None = None,
    sky_mask_clean: torch.Tensor | None = None,
    sky_mask_refined_clean: torch.Tensor | None = None,
    frame_ids: torch.Tensor | None = None,
) -> SimpleNamespace:
    B, S, P, _ = z_clean_n.shape
    mask = z_clean_n.new_zeros((B, S, P, 1))
    return SimpleNamespace(
        z_clean_n=z_clean_n,
        M_preserve=mask,
        M_source=torch.zeros_like(mask),
        M_dest=torch.ones_like(mask),
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        sky_gen_clean=sky_gen_clean,
        sky_gen_loss_weight=sky_gen_loss_weight,
        sky_gen_observation=sky_gen_observation,
        sky_gen_attention_mask=sky_gen_attention_mask,
        scene_gauge_clean_n=scene_gauge_clean_n,
        scene_gauge_clean=scene_gauge_clean,
        scene_gauge_valid=scene_gauge_valid,
        sky_mask_clean=sky_mask_clean,
        sky_mask_refined_clean=sky_mask_refined_clean,
        frame_ids=frame_ids,
        z_splat_n=torch.zeros_like(z_clean_n),
    )


def validation_mosaic_cell_width(args: argparse.Namespace) -> int:
    width = int(getattr(args, "val_mosaic_cell_width", MOSAIC_CELL_WIDTH_DEFAULT))
    if width <= 0:
        raise ValueError(f"--val_mosaic_cell_width must be positive, got {width}")
    return width


def collect_validation_mosaic_rows(
    bundle,
    z_generated_raw: torch.Tensor,
    rgb_images: dict[str, torch.Tensor] | None,
    args: argparse.Namespace,
    *,
    scene_slot: int,
    scene_label: str,
    guidance_scale: float,
    scale_index: int,
    is_primary: bool,
    visualization_batch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Encode one CFG scale's validation artifacts as mosaic rows.

    Every row is a JSON-safe dict so it can ride the existing validation gather
    to the composing rank.  Rows that describe ground truth are emitted only by
    the primary scale, which is the only task guaranteed to hold the
    visualization batch; the remaining scales contribute generated rows that
    land directly under it inside the same quantity group.
    """

    frames = min(int(args.val_log_images), int(bundle.z_clean_n.shape[1]))
    cell_width = validation_mosaic_cell_width(args)
    rgb_images = rgb_images or {}
    rows: list[dict[str, Any]] = []

    def add(group: str, order: int, caption: str, tensor: torch.Tensor) -> None:
        rows.append(
            {
                "slot": int(scene_slot),
                "scene": str(scene_label),
                "group": str(group),
                "order": int(order),
                "caption": str(caption),
                "frames": int(frames),
                "png": encode_mosaic_row(
                    tensor,
                    cell_width=cell_width,
                    photographic=mosaic_group(group).photographic,
                ),
            }
        )

    cfg_caption = f"cfg {float(guidance_scale):g}" + (" · primary" if is_primary else "")
    # Every rank fits this on the same ground-truth latent of the same scene,
    # so the CFG rows that arrive from other ranks land in the same colour
    # space as the GT row without any cross-rank transport.
    latent_basis = _latent_pca_basis(bundle.z_clean_n, frames)

    if is_primary:
        gt_rgb = rgb_images.get("input_rgb_gt")
        if not torch.is_tensor(gt_rgb) and visualization_batch is not None:
            gt_images = visualization_batch.get("images")
            if torch.is_tensor(gt_images) and gt_images.ndim == 5:
                gt_rgb = _image_grid(gt_images, frames)
        if torch.is_tensor(gt_rgb):
            add("rgb", GT_ROW_ORDER, "GT · recorded frames", gt_rgb)
        add(
            "latent",
            GT_ROW_ORDER,
            "GT",
            _latent_pca_grid(
                bundle.z_clean_n, args.patch_grid, frames, basis=latent_basis
            ),
        )

    generated_rgb = rgb_images.get("generated_raw_3dgs_rgb")
    if torch.is_tensor(generated_rgb):
        add("rgb", scale_index, cfg_caption, generated_rgb)

    overlay = _sky_mask_overlay_grid(
        getattr(bundle, "sky_mask_refined_clean", None),
        rgb_images.get("generated_pred_sky_mask"),
        frames,
    )
    if overlay is not None:
        sky_grid, has_target = overlay
        add(
            "sky",
            scale_index,
            cfg_caption if has_target else f"{cfg_caption} · prediction only, no GT mask",
            sky_grid,
        )

    generated_sky_rgb = rgb_images.get("generated_sky_rgb")
    # One row per CFG scale, like every other group.  The dome is a separate
    # generated stream from the foreground, so how it responds to guidance is
    # its own question -- and with the dome underfitting, seeing all three
    # side by side is how you tell "guidance washes it out" from "it was flat
    # to begin with".
    if torch.is_tensor(generated_sky_rgb):
        add("sky_rgb", scale_index, cfg_caption, generated_sky_rgb)

    add(
        "latent",
        scale_index,
        cfg_caption,
        _latent_pca_grid(
            z_generated_raw, args.patch_grid, frames, basis=latent_basis
        ),
    )
    add(
        "latent_err",
        scale_index,
        cfg_caption,
        _absolute_mask_grid(
            (z_generated_raw - bundle.z_clean_n).abs().mean(dim=-1, keepdim=True),
            args.patch_grid,
            frames,
        ),
    )
    return rows


def write_validation_mosaics(
    rows: Sequence[dict[str, Any]],
    log_dir: Path,
    step: int,
    args: argparse.Namespace,
) -> dict[str, Path]:
    """Compose and write one mosaic per scene slot; return W&B keys to paths."""

    if not rows:
        return {}
    mosaics = build_validation_mosaics(
        rows, step=int(step), cell_width=validation_mosaic_cell_width(args)
    )
    out_dir = log_dir / "validation" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for slot, mosaic in mosaics.items():
        path = out_dir / f"mosaic_slot{int(slot):02d}.jpg"
        # 4:4:4 is not optional here: the sky row encodes disagreement as pure
        # red/green fringes a few pixels wide, and chroma subsampling smears
        # exactly those away.
        mosaic.save(path, format="JPEG", quality=88, subsampling=0)
        paths[f"mosaic/slot{int(slot):02d}"] = path
    return paths


def should_apply_metric_depth_diagnostic(
    args: argparse.Namespace,
    global_step: int | None,
    *,
    training: bool,
) -> bool:
    """Schedule the physical LiDAR diagnostic independently of RGB loss."""

    if not training or global_step is None:
        return False
    every = int(getattr(args, "metric_depth_diagnostic_every", 500))
    start = int(getattr(args, "metric_depth_diagnostic_start_step", 0))
    step = int(global_step)
    return every > 0 and step >= start and step % every == 0


def hydrate_metric_depth_diagnostic_batch(
    batch: dict[str, Any],
    *,
    max_samples: int,
) -> dict[str, Any]:
    """Read lazy LiDAR paths only for the scheduled diagnostic rows.

    ``default_collate`` transposes ``list[str]`` from ``[B][S]`` to
    ``[S][B]``.  Rebuild rows here, leaving unscheduled rows as invalid zeros
    so the existing masked metric calculation remains unchanged.
    """

    images = batch.get("images")
    collated_paths = batch.get("metric_lidar_depth_paths")
    if not torch.is_tensor(images) or images.ndim != 5:
        raise ValueError("metric-depth hydration requires images [B,S,C,H,W]")
    if not isinstance(collated_paths, (list, tuple)):
        raise RuntimeError(
            "metric-depth diagnostic is due but the dataset did not return lazy paths"
        )
    batch_size, seq_len = (int(images.shape[0]), int(images.shape[1]))
    if len(collated_paths) != seq_len:
        raise ValueError(
            f"metric depth path axis {len(collated_paths)} != sequence length {seq_len}"
        )
    rows: list[list[str]] = [[] for _ in range(batch_size)]
    for frame_paths in collated_paths:
        if not isinstance(frame_paths, (list, tuple)) or len(frame_paths) != batch_size:
            raise ValueError("collated metric depth paths must be [S][B]")
        for row, path in enumerate(frame_paths):
            rows[row].append(os.fspath(path))

    height, width = int(images.shape[-2]), int(images.shape[-1])
    depth = torch.zeros((batch_size, seq_len, height, width), dtype=torch.float32)
    valid = torch.zeros_like(depth, dtype=torch.bool)
    row_count = batch_size if int(max_samples) <= 0 else min(
        batch_size, int(max_samples)
    )
    for row in range(row_count):
        row_depth, row_valid = load_metric_depth_diagnostic_paths(
            rows[row], height=height, width=width
        )
        depth[row] = row_depth
        valid[row] = row_valid
    hydrated = dict(batch)
    hydrated["metric_lidar_depth_m"] = depth
    hydrated["metric_lidar_depth_valid"] = valid
    return hydrated


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
    generator: torch.Generator | None = None,
    loss_terms_out: dict[str, torch.Tensor] | None = None,
    collect_expensive_diagnostics: bool = True,
    collect_logs: bool = True,
) -> tuple[torch.Tensor, dict[str, TrainLogValue]]:
    """One primary SceneFlow call with frozen TC/TCMG/TCMGA conditioning."""

    is_training = unwrap_ddp(scene_flow).training
    rgb_render_active = should_apply_rgb_render_loss(
        args, global_step, training=is_training
    )
    sky_view_active = should_apply_sky_view_loss(
        args, global_step, training=is_training
    )
    # Both losses read the same render context off the bundle, so the context
    # has to be built when either one is due.
    render_context_active = bool(rgb_render_active) or bool(sky_view_active)
    metric_depth_diagnostic_due = should_apply_metric_depth_diagnostic(
        args, global_step, training=is_training
    ) and torch.is_tensor(batch.get("metric_lidar_depth_m"))
    bundle = build_pretrain_bundle_from_batch(
        batch,
        vggt_model,
        scene_flow,
        device,
        args,
        include_rgb_render_context=render_context_active,
        include_metric_depth_diagnostic=metric_depth_diagnostic_due,
    )
    full_layout = getattr(bundle, "layout_condition", None)
    appearance_class_id = getattr(bundle, "appearance_class_id", None)
    if not isinstance(full_layout, LayoutConditionBatch):
        raise RuntimeError("layout-v2 training requires bundle.layout_condition")
    if not torch.is_tensor(appearance_class_id):
        raise RuntimeError("layout-v2 training requires appearance_class_id")

    batch_size = int(bundle.z_clean_n.shape[0])
    if is_training:
        layout_tasks = sample_layout_tasks(
            batch_size,
            device=bundle.z_clean_n.device,
            generator=generator,
        )
    else:
        layout_tasks = torch.full(
            (batch_size,),
            int(LayoutTask.TCMGA),
            device=bundle.z_clean_n.device,
            dtype=torch.int8,
        )
    tasked_layout, tasked_appearance_class_id = apply_layout_training_tasks(
        full_layout,
        appearance_class_id,
        layout_tasks,
    )
    text_drop_mask = sample_uncond_drop_mask(
        batch_size,
        float(getattr(args, "text_uncond_drop_prob", 0.1)),
        device=bundle.z_clean_n.device,
        training=is_training,
    )

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
        t_eps=scene_flow_t_eps(scene_flow),
        generator=generator,
    )
    sky_target = build_sky_rectified_flow_target(
        getattr(bundle, "sky_gen_clean", None),
        target,
        loss_weight=getattr(bundle, "sky_gen_loss_weight", None),
        generator=generator,
        observation=getattr(bundle, "sky_gen_observation", None),
    )
    gauge_target = build_gauge_rectified_flow_target(
        getattr(bundle, "scene_gauge_clean_n", None),
        target,
        generator=generator,
    )
    if gauge_target is None:
        raise RuntimeError("layout-v2 training requires the scene-gauge target")
    boundary = boundary_mask_from_edit_mask(M_edit, args.patch_grid, radius=1)
    scaffold_tok = torch.zeros_like(bundle.z_clean_n)
    use_repa = float(args.lambda_repa) != 0.0
    text_tokens, text_mask = encode_text_condition(
        text_encoder,
        getattr(bundle, "captions", None),
        drop_mask=text_drop_mask,
    )
    gauge_grad_scale = layout_to_gauge_scale(
        0 if global_step is None else int(global_step),
        upper=float(getattr(args, "layout_to_gauge_grad_scale", 1.0)),
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
            **layout_model_kwargs(
                tasked_layout,
                tasked_appearance_class_id,
                gauge_grad_scale=gauge_grad_scale,
            ),
            text_tokens=text_tokens,
            text_attention_mask=text_mask,
            camera_condition_tokens=bundle.camera_condition_tokens,
            camera_attention_mask=bundle.camera_attention_mask,
            sky_gen_tokens=None if sky_target is None else sky_target.z_t,
            sky_gen_attention_mask=None,
            gauge_gen_tokens=gauge_target.z_t,
            gauge_gen_attention_mask=None,
            return_mid=use_repa,
            return_dict=True,
            return_sky_mask=True,
            return_base=float(args.base_model_coeff) != 0.0,
            return_layout_diagnostics=bool(
                collect_logs and collect_expensive_diagnostics
            ),
            control_drop_mask=None,
            frame_ids=getattr(bundle, "frame_ids", None),
            fps=getattr(bundle, "fps", None),
        )
        if not isinstance(out, dict):
            raise RuntimeError("SceneFlow return_dict=True must return a dict")
        pred_clean = out["video"]
        pred_base = out.get("video_base")
        pred_sky = out.get("sky")
        pred_gauge = out.get("gauge")
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
            mid_repa=out.get("mid_repa") if use_repa else None,
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
            defer_log_values=True,
            collect_logs=collect_logs,
        )
        if loss_terms_out is not None:
            loss_terms_out["video_core"] = loss

        if collect_logs and collect_expensive_diagnostics:
            layout_diagnostics = out.get("actor_alignment_diagnostics")
            required_layout_diagnostics = (
                "map_residual_rms",
                "actor_residual_rms",
                "map_metric_valid_fraction",
            )
            if not isinstance(layout_diagnostics, dict):
                raise RuntimeError("SceneFlow did not return layout diagnostics")
            for name in required_layout_diagnostics:
                value = layout_diagnostics.get(name)
                if not torch.is_tensor(value) or int(value.numel()) != 1:
                    raise RuntimeError(f"layout diagnostic {name!r} must be scalar")
                logs[f"layout/{name}"] = value.detach().float().reshape(())
            invalid_all_window = layout_diagnostics.get(
                "appearance_invalid_all_window_count"
            )
            if not torch.is_tensor(invalid_all_window) or invalid_all_window.ndim != 1:
                raise RuntimeError(
                    "layout diagnostic 'appearance_invalid_all_window_count' must be [B]"
                )
            if int(invalid_all_window.shape[0]) != int(bundle.z_clean_n.shape[0]):
                raise RuntimeError(
                    "appearance_invalid_all_window_count batch does not match the task batch"
                )
            # Actor-side metric coverage.  ``metric_support`` is ``frame_support``
            # restricted to the actor-frames whose eight cuboid corners all
            # survived the metric projection, so this ratio answers "of the
            # actors the camera can see, how many can be placed in metres at
            # all".  A structural hole here caps every position-control claim
            # regardless of how well the model trains; the v5 run sat at 0.85 and
            # the number was not logged anywhere in v6.  Two reductions over a
            # bool tensor already in memory.
            projected = full_layout.projected_actor_geometry
            frame_support_total = projected.frame_support.detach().float().sum()
            logs["layout/actor_metric_support_fraction"] = (
                projected.metric_support.detach().float().sum()
                / frame_support_total.clamp_min(1.0)
            )
            invalid_all_window_total = invalid_all_window.detach().float().sum()
            tasked_binding_total = (
                tasked_layout.appearance.binding_valid.detach().float().sum()
            )
            logs["layout/appearance_invalid_all_window_count"] = invalid_all_window_total
            logs["layout/appearance_invalid_all_window_rate"] = (
                invalid_all_window_total / tasked_binding_total.clamp_min(1.0)
            )

        z_sky_pred = None
        if sky_target is not None and torch.is_tensor(pred_sky):
            v_sky_pred = model_prediction_to_velocity(
                scene_flow, pred_sky, sky_target
            )
            z_sky_pred = model_prediction_to_clean(
                scene_flow, pred_sky, sky_target
            )
            loss_sky_flow = sky_flow_loss(
                v_sky_pred,
                sky_target,
                bundle.sky_gen_clean,
                unobserved_beta=float(
                    getattr(
                        args,
                        "sky_unobserved_loss_beta",
                        DEFAULT_SKY_UNOBSERVED_LOSS_BETA,
                    )
                ),
            )
            loss = loss + float(args.lambda_sky_flow) * loss_sky_flow
            if collect_logs:
                logs["loss_sky_flow"] = loss_sky_flow.detach().float().reshape(())
                sky_loss_weight = getattr(bundle, "sky_gen_loss_weight", None)
                if torch.is_tensor(sky_loss_weight):
                    # Unobserved atlas directions carry
                    # ``--sky_unobserved_loss_weight`` instead of 1.0, so this
                    # mean is the denominator every sky number has to be read
                    # against, and the observed fraction inverts out of it as
                    # ``(mean - w) / (1 - w)``.  ``loss_sky_flow`` is a weighted
                    # mean, so the observed sky's share of the gradient is
                    # exactly ``observed_fraction / mean``: at the old w=0.05
                    # that was 17.6%, at w=0.005 it is 68%.
                    logs["sky_token_loss_weight_mean"] = (
                        sky_loss_weight.detach().float().mean()
                    )
        elif torch.is_tensor(pred_sky):
            loss = loss + 0.0 * pred_sky.sum()
            if collect_logs:
                logs["loss_sky_flow"] = 0.0

        z_gauge_pred = None
        pred_gauge_physical = None
        if not torch.is_tensor(pred_gauge):
            raise RuntimeError("layout-v2 SceneFlow must return gauge prediction")
        v_gauge_pred = model_prediction_to_velocity(
            scene_flow, pred_gauge, gauge_target
        )
        z_gauge_pred = model_prediction_to_clean(
            scene_flow, pred_gauge, gauge_target
        )
        gauge_valid = bundle.scene_gauge_valid.to(
            device=v_gauge_pred.device, dtype=torch.bool
        )
        if gauge_valid.ndim == 2:
            gauge_valid = gauge_valid.unsqueeze(1)
        if gauge_valid.shape != v_gauge_pred.shape:
            raise ValueError(
                "scene_gauge_valid must be [B,1,3], got "
                f"{tuple(gauge_valid.shape)}"
            )
        gauge_sq_error = (
            v_gauge_pred.float()
            - gauge_target.v_gt.to(
                device=v_gauge_pred.device, dtype=torch.float32
            )
        ).square()
        gauge_mask = gauge_valid.to(dtype=gauge_sq_error.dtype)
        loss_gauge_flow = (gauge_sq_error * gauge_mask).sum() / gauge_mask.sum().clamp_min(1.0)
        pred_gauge_physical = unwrap_ddp(scene_flow).denormalize_gauge(
            z_gauge_pred.float()
        )
        loss_gauge_direct = masked_gauge_direct_loss(
            pred_gauge_physical,
            bundle.scene_gauge_clean,
            gauge_valid,
        )
        loss = (
            loss
            + float(args.lambda_gauge_flow) * loss_gauge_flow
            + float(args.lambda_gauge_direct) * loss_gauge_direct
        )
        if loss_terms_out is not None:
            loss_terms_out["gauge_flow"] = (
                float(args.lambda_gauge_flow) * loss_gauge_flow
            )
            loss_terms_out["gauge_direct"] = (
                float(args.lambda_gauge_direct) * loss_gauge_direct
            )
        if collect_logs:
            logs["loss_gauge_flow"] = loss_gauge_flow.detach().float().reshape(())
            logs["loss_gauge_direct"] = loss_gauge_direct.detach().float().reshape(())
            gauge_logs = gauge_diagnostic_metrics(
                pred_gauge_physical,
                bundle.scene_gauge_clean,
                gauge_valid,
                prior_log_scale=unwrap_ddp(scene_flow).gauge_mean[0],
                defer_log_values=True,
            )
            logs["gauge/log_scale_error"] = gauge_logs.pop(
                "gauge_log_scale_error"
            )
            logs["gauge/fov_error"] = gauge_logs.pop("gauge_fov_error_deg")
            logs.update(gauge_logs)

        sky_mask_target = getattr(bundle, "sky_mask_clean", None)
        if torch.is_tensor(pred_sky_mask_logits) and torch.is_tensor(sky_mask_target):
            loss_sky_mask, sky_mask_logs = sky_mask_patch_loss(
                pred_sky_mask_logits,
                sky_mask_target,
                dice_weight=float(getattr(args, "sky_mask_dice_weight", 0.5)),
                pos_weight_max=float(getattr(args, "sky_mask_pos_weight_max", 10.0)),
                defer_log_values=True,
                collect_logs=collect_logs,
            )
            loss = loss + float(getattr(args, "lambda_sky_mask", 0.05)) * loss_sky_mask
            if collect_logs:
                logs["loss_sky_mask"] = loss_sky_mask.detach().float().reshape(())
                logs.update(sky_mask_logs)
        elif torch.is_tensor(pred_sky_mask_logits):
            loss = loss + 0.0 * pred_sky_mask_logits.sum()
            if collect_logs:
                logs["loss_sky_mask"] = 0.0

        sky_mask_refined_target = getattr(bundle, "sky_mask_refined_clean", None)
        if torch.is_tensor(pred_sky_mask_refined_logits) and torch.is_tensor(
            sky_mask_refined_target
        ):
            loss_sky_mask_refine, refine_logs = sky_mask_refined_loss(
                pred_sky_mask_refined_logits,
                sky_mask_refined_target,
                dice_weight=float(getattr(args, "sky_mask_dice_weight", 0.5)),
                pos_weight_max=float(getattr(args, "sky_mask_pos_weight_max", 10.0)),
                boundary_weight=float(getattr(args, "sky_mask_refine_boundary_weight", 4.0)),
                boundary_loss_weight=sky_mask_refine_boundary_loss_weight(args),
                defer_log_values=True,
                collect_logs=collect_logs,
            )
            weighted_sky_mask_refine = float(
                getattr(args, "lambda_sky_mask_refine", 0.1)
            ) * loss_sky_mask_refine
            loss = loss + weighted_sky_mask_refine
            if loss_terms_out is not None:
                # The largest weighted auxiliary after repa/base, and unlike the
                # world-feedback levels it carries no sigma attenuation, so it
                # belongs in the same audit.
                loss_terms_out["sky_mask_refine"] = weighted_sky_mask_refine
            if collect_logs:
                logs["loss_sky_mask_refine"] = (
                    loss_sky_mask_refine.detach().float().reshape(())
                )
                logs.update(refine_logs)
        elif torch.is_tensor(pred_sky_mask_refined_logits):
            loss = loss + 0.0 * pred_sky_mask_refined_logits.sum()
            if collect_logs:
                logs["loss_sky_mask_refine"] = 0.0

        # Seed the sentinels so the availability flags below can gate both the
        # W&B mean and the fallback diagnostic path.  Two metric-depth series
        # share the LiDAR sample: the generated gauge and the offline calibrated
        # one.  (The old ``_pred_gauge`` prefix was a pure alias of the first and
        # stays deleted.)
        if collect_logs:
            logs.update(metric_depth_diagnostic_log_values())
            logs.update(
                metric_depth_diagnostic_log_values(
                    prefix="metric_depth_rel_err_teacher_gauge"
                )
            )

        if render_context_active:
            if not torch.is_tensor(getattr(bundle, "rgb_render_images", None)):
                raise RuntimeError("RGB render context is missing target images")
            if not torch.is_tensor(getattr(bundle, "rgb_render_masks", None)):
                raise RuntimeError("RGB render context is missing GT sky masks")
            rgb_row_indices = capped_render_row_indices(
                batch_size,
                int(args.rgb_render_max_samples),
                device=target.sigmas.device,
            )
            ramp = rgb_render_loss_ramp(args, global_step)
            if collect_logs:
                logs["rgb_render_ramp"] = float(ramp)
            if rgb_row_indices.numel() == 0:
                if collect_logs:
                    logs.update(
                        {
                            "loss_rgb_render": 0.0,
                            "loss_rgb_render_weighted": 0.0,
                            "loss_level_consistency": 0.0,
                            "loss_head_consistency": 0.0,
                            "loss_level_consistency_weighted": 0.0,
                            "loss_head_consistency_weighted": 0.0,
                            "rgb_render_active": 0.0,
                        }
                    )
            else:
                def select_rows(value: torch.Tensor, name: str) -> torch.Tensor:
                    if (
                        not torch.is_tensor(value)
                        or value.ndim == 0
                        or int(value.shape[0]) != batch_size
                    ):
                        shape = None if not torch.is_tensor(value) else tuple(value.shape)
                        raise ValueError(
                            f"{name} must have batch dimension {batch_size}, got {shape}"
                        )
                    return value.index_select(
                        0, rgb_row_indices.to(device=value.device)
                    )

                render_pose_requested = requested_render_pose_for_rows(
                    bundle,
                    pred_gauge_physical,
                    rgb_row_indices,
                )
                selected_sky_tokens = None
                selected_sky_latent = None
                if z_sky_pred is not None:
                    selected_sky_latent = select_rows(
                        z_sky_pred, "z_sky_pred"
                    )
                    # The 3DGS render below uses this as its background, so it
                    # is decoded whenever a sky exists -- not only on the steps
                    # the sky-view loss happens to be due.  Tying the two would
                    # silently swap the render's background to black.
                    selected_sky_tokens = decode_sky_patch_tokens(
                        selected_sky_latent
                    )
                if sky_view_active and selected_sky_latent is not None:
                    sky_view_frames = int(args.rgb_render_max_frames)
                    sky_view_frames = (
                        int(bundle.rgb_render_images.shape[1])
                        if sky_view_frames <= 0
                        else min(
                            sky_view_frames,
                            int(bundle.rgb_render_images.shape[1]),
                        )
                    )
                    selected_sigma_for_sky = select_rows(
                        target.sigmas, "sigmas"
                    )
                    sky_view_weights = rgb_render_sigma_weight(
                        selected_sigma_for_sky,
                        float(getattr(args, "rgb_render_sigma_power", 2.0)),
                    )
                    sky_view_loss, sky_view_logs = generated_sky_view_reconstruction_loss(
                        vggt_model=vggt_model,
                        sky_latent=selected_sky_latent,
                        images=select_rows(
                            bundle.rgb_render_images, "rgb_render_images"
                        )[:, :sky_view_frames],
                        sky_mask=select_rows(
                            bundle.rgb_render_masks, "rgb_render_masks"
                        )[:, :sky_view_frames],
                        render_pose_enc_dggt=render_pose_requested[
                            :, :sky_view_frames
                        ],
                        lpips_model=lpips_model,
                        lpips_weight=float(getattr(args, "sky_view_lpips_weight", 0.01)),
                        high_frequency_weight=float(
                            getattr(
                                args,
                                "sky_view_high_frequency_weight",
                                SKY_VIEW_HIGH_FREQUENCY_WEIGHT_DEFAULT,
                            )
                        ),
                        loss_sample_weight=sky_view_weights,
                        defer_log_values=True,
                        collect_logs=collect_logs,
                    )
                    sky_view_ramp = sky_view_loss_ramp(args, global_step)
                    weighted_sky_view_loss = (
                        float(
                            getattr(
                                args,
                                "lambda_sky_view_reconstruction",
                                SKY_VIEW_LAMBDA_DEFAULT,
                            )
                        )
                        * sky_view_ramp
                        * sky_view_loss
                    )
                    loss = loss + weighted_sky_view_loss
                    if collect_logs:
                        logs.update(sky_view_logs)
                        logs["sky_view_ramp"] = float(sky_view_ramp)
                        logs["loss_sky_view_weighted"] = (
                            weighted_sky_view_loss.detach().float().reshape(())
                        )

                # The 3DGS render path is the expensive half and keeps its own
                # 5000-step gate; the sky atlas above renders by rotation alone.
                if rgb_render_active:
                    rgb_timestamps = bundle.rgb_render_timestamps
                    if rgb_timestamps.ndim != 1:
                        rgb_timestamps = select_rows(
                            rgb_timestamps, "rgb_render_timestamps"
                        )
                    selected_sigma = select_rows(target.sigmas, "sigmas")
                    sigma_weights = rgb_render_sigma_weight(
                        selected_sigma,
                        float(getattr(args, "rgb_render_sigma_power", 2.0)),
                    )
                    rgb_result = compute_rgb_render_loss(
                        vggt_model=vggt_model,
                        scene_flow_root=unwrap_ddp(scene_flow),
                        z_clean_pred_n=select_rows(z_pred, "z_pred"),
                        z_clean_target_n=select_rows(bundle.z_clean_n, "z_clean_n"),
                        images=select_rows(bundle.rgb_render_images, "rgb_render_images"),
                        timestamps=rgb_timestamps,
                        render_pose_enc_dggt=render_pose_requested,
                        render_sky_probability=select_rows(
                            render_sky_probability_from_primary_output(out),
                            "render_sky_probability",
                        ),
                        loss_sky_mask_gt=select_rows(
                            bundle.rgb_render_masks, "rgb_render_masks"
                        ),
                        patch_grid=args.patch_grid,
                        patch_start_idx=int(bundle.rgb_render_patch_start_idx),
                        max_samples=0,
                        max_frames=int(args.rgb_render_max_frames),
                        render_stride=int(args.rgb_render_stride),
                        background_mode=(
                            "sky_tokens" if selected_sky_tokens is not None else "black"
                        ),
                        sky_tokens=selected_sky_tokens,
                        sky_grid=sky_atlas_shape(args),
                        patch_weight_mask=select_rows(target.M_edit, "M_edit"),
                        sky_weight=float(args.rgb_render_sky_weight),
                        camera_grad_scale=0.0,
                        gauge_pose_grad_scale=1.0,
                        sky_mask_grad_scale=(
                            float(args.rgb_render_sky_mask_grad_scale) * float(ramp)
                        ),
                        lpips_model=lpips_model,
                        lpips_weight=float(args.rgb_render_lpips_weight),
                        loss_sample_weight=sigma_weights,
                        conf_weight_power=float(
                            getattr(args, "feedback_conf_weight_power", 1.0)
                        ),
                        conf_weight_floor=float(
                            getattr(args, "feedback_conf_weight_floor", 0.05)
                        ),
                        dynamic_space=str(
                            getattr(args, "head_dynamic_space", "probability")
                        ),
                        return_generated_depth=bool(metric_depth_diagnostic_due),
                        pullback_calibration=getattr(
                            unwrap_ddp(scene_flow), "_pullback_calibration", None
                        ),
                        defer_log_values=True,
                        collect_logs=collect_logs,
                    )
                    weighted_rgb = (
                        float(args.lambda_rgb_render)
                        * float(ramp)
                        * rgb_result.loss
                    )
                    weighted_level = float(
                        getattr(args, "lambda_level_consistency", 0.0)
                    ) * float(ramp) * getattr(rgb_result, "level_loss", rgb_result.loss * 0.0)
                    weighted_head = float(
                        getattr(args, "lambda_head_consistency", 0.0)
                    ) * float(ramp) * getattr(rgb_result, "head_loss", rgb_result.loss * 0.0)
                    loss = loss + weighted_rgb + weighted_level + weighted_head
                    if loss_terms_out is not None:
                        # Kept separable so the gradient-balance audit can probe each
                        # world-feedback level against the flow objective on the
                        # shared trunk.  Loss share is a poor proxy for influence
                        # here: these three are attenuated by ``(1-sigma)**2`` while
                        # the flow target carries a ``1/sigma**2`` factor, so their
                        # relative pull has to be measured, not inferred.
                        loss_terms_out["rgb_render"] = weighted_rgb
                        loss_terms_out["level_consistency"] = weighted_level
                        loss_terms_out["head_consistency"] = weighted_head
                    if collect_logs:
                        logs.update(rgb_result.logs)
                        logs["rgb_render_sigma_mean"] = selected_sigma.detach().float().mean()
                        logs["rgb_render_sigma_weight_mean"] = sigma_weights.detach().float().mean()
                        logs["loss_rgb_render_weighted"] = weighted_rgb.detach().float().reshape(())
                        logs["loss_level_consistency_weighted"] = weighted_level.detach().float().reshape(())
                        logs["loss_head_consistency_weighted"] = weighted_head.detach().float().reshape(())
                        logs["rgb_render_active"] = 1.0
                    metric_lidar = getattr(bundle, "metric_lidar_depth_m", None)
                    if (
                        rgb_result.generated_depth is not None
                        and torch.is_tensor(metric_lidar)
                    ):
                        selected_metric_lidar = select_rows(
                            metric_lidar, "metric_lidar_depth_m"
                        )
                        selected_scale_valid = select_rows(
                            bundle.scene_gauge_valid[..., 0],
                            "scene_gauge_scale_valid",
                        )
                        selected_lidar_valid = (
                            select_rows(
                                bundle.metric_lidar_depth_valid,
                                "metric_lidar_depth_valid",
                            )
                            if torch.is_tensor(
                                getattr(bundle, "metric_lidar_depth_valid", None)
                            )
                            else None
                        )
                        metric_rel_err = metric_depth_relative_error(
                            rgb_result.generated_depth,
                            selected_metric_lidar,
                            select_rows(
                                pred_gauge_physical, "pred_gauge_physical"
                            )[..., 0],
                            calibration=getattr(
                                unwrap_ddp(scene_flow), "_pullback_calibration", None
                            ),
                            scale_valid=selected_scale_valid,
                            lidar_valid=selected_lidar_valid,
                        )
                        if collect_logs:
                            logs.update(metric_depth_diagnostic_log_values(metric_rel_err))
                            # Same generated depth, same LiDAR, only the scale swapped
                            # for the offline calibrated one.  The gap between the two
                            # series is the only read on how much of the residual
                            # metric error is the generated gauge rather than the
                            # generated geometry; when they meet, scale has stopped
                            # being the bottleneck.  Reuses the depth already
                            # rendered above, so it costs one masked median.
                            logs.update(
                                metric_depth_diagnostic_log_values(
                                    metric_depth_relative_error(
                                        rgb_result.generated_depth,
                                        selected_metric_lidar,
                                        select_rows(
                                            bundle.scene_gauge_clean,
                                            "scene_gauge_clean",
                                        )[..., 0],
                                        calibration=getattr(
                                            unwrap_ddp(scene_flow),
                                            "_pullback_calibration",
                                            None,
                                        ),
                                        scale_valid=selected_scale_valid,
                                        lidar_valid=selected_lidar_valid,
                                    ),
                                    prefix="metric_depth_rel_err_teacher_gauge",
                                )
                            )
                elif collect_logs:
                    logs.update(
                        {
                            "loss_rgb_render": 0.0,
                            "loss_rgb_render_weighted": 0.0,
                            "loss_level_consistency": 0.0,
                            "loss_head_consistency": 0.0,
                            "loss_level_consistency_weighted": 0.0,
                            "loss_head_consistency_weighted": 0.0,
                            "rgb_render_active": 0.0,
                        }
                    )
        elif collect_logs:
            logs.update(
                {
                    "rgb_render_active": 0.0,
                    "loss_level_consistency": 0.0,
                    "loss_head_consistency": 0.0,
                    "loss_level_consistency_weighted": 0.0,
                    "loss_head_consistency_weighted": 0.0,
                }
            )

        if (
            collect_logs
            and metric_depth_diagnostic_due
            and logs["metric_depth_rel_err_available"] == 0.0
            and torch.is_tensor(getattr(bundle, "metric_lidar_depth_m", None))
        ):
            metric_lidar = bundle.metric_lidar_depth_m
            max_rows = int(
                getattr(args, "metric_depth_diagnostic_max_samples", 1)
            )
            row_count = (
                batch_size if max_rows <= 0 else min(max_rows, batch_size)
            )
            if row_count > 0:
                with torch.no_grad():
                    metric_geometry = decode_generated_dggt_geometry(
                        vggt_model=vggt_model,
                        scene_flow_root=unwrap_ddp(scene_flow),
                        z_clean_pred_n=z_pred[:row_count].detach(),
                        patch_grid=args.patch_grid,
                        patch_start_idx=int(
                            getattr(vggt_model.aggregator, "patch_start_idx", 5)
                        ),
                        image_hw=(
                            int(metric_lidar.shape[-2]),
                            int(metric_lidar.shape[-1]),
                        ),
                        pullback_calibration=getattr(
                            unwrap_ddp(scene_flow), "_pullback_calibration", None
                        ),
                    )
                    metric_rel_err = metric_depth_relative_error(
                        metric_geometry.depth,
                        metric_lidar[:row_count],
                        pred_gauge_physical[:row_count, ..., 0],
                        calibration=getattr(
                            unwrap_ddp(scene_flow), "_pullback_calibration", None
                        ),
                        scale_valid=bundle.scene_gauge_valid[:row_count, ..., 0],
                        lidar_valid=(
                            bundle.metric_lidar_depth_valid[:row_count]
                            if torch.is_tensor(
                                getattr(bundle, "metric_lidar_depth_valid", None)
                            )
                            else None
                        ),
                    )
                logs.update(metric_depth_diagnostic_log_values(metric_rel_err))

    if collect_logs and collect_expensive_diagnostics:
        logs.update(
            layout_training_monitor_log_values(
                full_layout,
                tasked_layout,
                layout_tasks,
                gauge_grad_scale=gauge_grad_scale,
                defer_log_values=True,
            )
        )
    if collect_logs:
        logs["text_uncond_drop_frac"] = (
            0.0
            if text_drop_mask is None
            else text_drop_mask.detach().float().mean()
        )
        logs["sigma_mean"] = target.sigmas.detach().float().mean()
        logs["loss"] = loss.detach().float().reshape(())
    return loss, drop_uninformative_log_values(logs)



def _canonical_asset_encoder_for_model(
    vggt_model: VGGT,
    scene_flow: nn.Module,
    device: torch.device,
    patch_grid: tuple[int, int],
) -> CanonicalAssetEncoder:
    """Reuse the frozen adapter and its bounded appearance-token LRU."""
    cache = vggt_model.__dict__.setdefault(
        "_scene_flow_canonical_asset_encoders",
        {},
    )
    key = (
        id(unwrap_ddp(scene_flow)),
        int(patch_grid[0]),
        int(patch_grid[1]),
        str(device),
    )
    encoder = cache.get(key)
    if encoder is None:
        encoder = CanonicalAssetEncoder(
            vggt_model.aggregator,
            vggt_model.scene_tokenizer,
            unwrap_ddp(scene_flow),
            patch_grid=patch_grid,
            max_tokens=32,
            cache_size=1024,
        ).to(device)
        # Layout-v2 fixes A's token width at 1024.  CanonicalAssetEncoder can
        # otherwise infer it only after seeing a non-empty reference, which
        # makes a legitimate all-NULL TCMG/TC batch fail before task sampling.
        if encoder._asset_dim is None:
            encoder._asset_dim = APPEARANCE_TOKEN_DIM
        cache[key] = encoder
    return encoder


def _canonical_raw_image_size_hw(
    image_size_hw: torch.Tensor | tuple[int, int],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Accept only frozen raw-size layouts; never discard per-frame values."""
    value = torch.as_tensor(image_size_hw, device=device)
    if value.ndim == 1 and tuple(value.shape) == (2,):
        result = value
    elif value.ndim == 2 and tuple(value.shape) == (int(batch_size), 2):
        result = value
    elif value.ndim == 3 and tuple(value.shape) == (int(batch_size), 1, 2):
        result = value[:, 0]
    else:
        raise ValueError(
            "raw_image_size_hw must be [2], [B,2], or singleton [B,1,2]; "
            f"time-varying [B,S,2] is unsupported, got {tuple(value.shape)} for B={batch_size}"
        )
    if bool((result <= 0).any()):
        raise ValueError("raw_image_size_hw must contain positive height/width")
    return result


def _requested_front_intrinsics(
    value: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Select front-camera K and expand an explicitly scene-static calibration.

    Raw Waymo calibration is stored once per scene, so the dataloader emits
    ``[B,1,3,3]`` after collation.  Layout projection additionally carries a
    per-frame canvas K as ``[B,S,3,3]``.  Accept exactly those two temporal
    layouts: a singleton calibration may be expanded across the requested
    frames, while every other partial/mismatched time axis remains an error.
    """

    intrinsics = torch.as_tensor(
        value, device=device, dtype=torch.float32
    )
    if intrinsics.ndim == 5 and tuple(intrinsics.shape[-2:]) == (3, 3):
        if int(intrinsics.shape[2]) < 1:
            raise ValueError(f"{name} has an empty camera-view axis")
        intrinsics = intrinsics[:, :, 0]
    if intrinsics.ndim != 4 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError(
            f"{name} must be [B,S,3,3] or [B,S,V,3,3], got "
            f"{tuple(intrinsics.shape)}"
        )
    if int(intrinsics.shape[0]) != int(batch_size):
        raise ValueError(
            f"{name} batch dimension {int(intrinsics.shape[0])} != {int(batch_size)}"
        )
    if int(intrinsics.shape[1]) == 1 and int(seq_len) != 1:
        intrinsics = intrinsics.expand(-1, int(seq_len), -1, -1)
    expected = (int(batch_size), int(seq_len), 3, 3)
    if tuple(intrinsics.shape) != expected:
        raise ValueError(f"{name} shape {tuple(intrinsics.shape)} != {expected}")
    if not bool(torch.isfinite(intrinsics).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool((intrinsics[..., 0, 0] <= 0).any()) or bool(
        (intrinsics[..., 1, 1] <= 0).any()
    ):
        raise ValueError(f"{name} must contain positive focal lengths")
    return intrinsics.contiguous()


def build_pretrain_bundle_from_batch(
    batch: dict[str, Any],
    vggt_model: VGGT,
    scene_flow: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    *,
    include_rgb_render_context: bool = False,
    include_metric_depth_diagnostic: bool = True,
):
    images_raw = batch["images"]
    if not torch.is_tensor(images_raw) or images_raw.ndim != 5:
        shape = tuple(images_raw.shape) if torch.is_tensor(images_raw) else type(images_raw)
        raise ValueError(f"Expected images [B,S,3,H,W], got {shape}")
    batch_size_raw, seq_len = images_raw.shape[:2]
    if seq_len < 2:
        raise ValueError("SceneFlow pretraining requires sequence_length >= 2 for cross-frame asset conditions.")

    sf_root = unwrap_ddp(scene_flow)
    with torch.no_grad():
        with autocast_context(args, device):
            dggt_context_raw = batch.get("dggt_context_images")
            if not torch.is_tensor(dggt_context_raw):
                raise RuntimeError(
                    "Raw SceneFlow pretraining requires dggt_context_images so DGGT camera/latent "
                    "targets are computed in the complete 29-frame clip context."
                )
            dggt_context_images = _images_to_device(
                dggt_context_raw,
                device,
            )
            if dggt_context_images.ndim != 5 or int(dggt_context_images.shape[0]) != int(batch_size_raw):
                raise ValueError(
                    "dggt_context_images must be [B,T,3,H,W], got "
                    f"{tuple(dggt_context_images.shape)}"
                )
            if int(dggt_context_images.shape[1]) != 29:
                raise ValueError(
                    "Raw SceneFlow DGGT context must contain the complete 29-frame caption clip, "
                    f"got {int(dggt_context_images.shape[1])} frames."
                )
            context_frame_ids_raw = batch.get("dggt_context_frame_ids")
            if not torch.is_tensor(context_frame_ids_raw):
                raise RuntimeError("Raw SceneFlow pretraining requires dggt_context_frame_ids [B,29].")
            context_frame_ids = context_frame_ids_raw.to(device=device, dtype=torch.long, non_blocking=True)
            if context_frame_ids.ndim == 1:
                context_frame_ids = context_frame_ids.view(1, -1).expand(int(batch_size_raw), -1)
            expected_context_ids = torch.arange(29, device=device, dtype=torch.long).view(1, -1).expand(
                int(batch_size_raw), -1
            )
            if not torch.equal(context_frame_ids, expected_context_ids):
                raise ValueError("dggt_context_frame_ids must be clip-global [0, ..., 28] for every row.")
            window_indices_raw = batch.get("dggt_window_indices")
            if not torch.is_tensor(window_indices_raw):
                raise RuntimeError("Raw SceneFlow pretraining requires dggt_window_indices [B,S].")
            window_indices = window_indices_raw.to(device=device, dtype=torch.long, non_blocking=True)
            if window_indices.ndim == 1:
                window_indices = window_indices.view(1, -1).expand(int(batch_size_raw), -1)
            if tuple(window_indices.shape) != (int(batch_size_raw), int(seq_len)):
                raise ValueError(
                    "dggt_window_indices shape "
                    f"{tuple(window_indices.shape)} != {(int(batch_size_raw), int(seq_len))}"
                )
            # `images` is an exact frame subset of the mandatory 29-frame DGGT
            # context. Gather it after the single host-to-device transfer instead
            # of copying the same pixels to CUDA a second time.
            images = batched_gather_frames(
                dggt_context_images,
                window_indices,
                name="pretrain_images_from_dggt_context",
            )
            outputs = vggt_model.get_aggregator_token_outputs(dggt_context_images)
            aggregated_tokens_list = outputs["aggregated_tokens_list"]
            image_tokens_context = outputs["image_tokens_list"]
            image_tokens_list = [
                batched_gather_frames(tokens, window_indices, name=f"image_tokens_list[{level}]")
                for level, tokens in enumerate(image_tokens_context)
            ]
            patch_start_idx = int(outputs["patch_start_idx"])
            tokens_4 = select_patch_pyramid(image_tokens_list, TOKENIZER_LEVELS, patch_start_idx)
            _, image_tokens_last = split_special_and_patch(image_tokens_list[-1], patch_start_idx)
            if image_tokens_last.shape[-2] != args.patch_grid[0] * args.patch_grid[1]:
                raise ValueError(
                    f"Expected {args.patch_grid[0] * args.patch_grid[1]} patch tokens, "
                    f"got {image_tokens_last.shape[-2]}."
                )
            z_clean = encode_tokenizer_windowed(
                vggt_model.scene_tokenizer,
                tokens_4,
                patch_grid=args.patch_grid,
                window_len=_tokenizer_window_len(scene_flow, args),
            )
            # CameraHead is retained only to define the frozen teacher-space
            # sky-atlas target. RGB/HDS rendering uses requested C below.
            camera_pose_context_dggt = vggt_model.camera_head(aggregated_tokens_list)[-1].float()
            camera_pose_gt_dggt = batched_gather_frames(
                camera_pose_context_dggt, window_indices, name="camera_pose_context_dggt"
            )
        z_clean_n = sf_root.normalize(z_clean.float())
        camera_to_world_gt = batch.get("camera_to_world_corrected")
        intrinsics_gt = batch.get("intrinsics")
        intrinsics_canvas_gt = batch.get("camera_intrinsics_canvas")
        if not torch.is_tensor(camera_to_world_gt) or not torch.is_tensor(intrinsics_gt):
            raise RuntimeError(
                "Raw Waymo pretrain batch is missing camera_to_world_corrected/intrinsics; "
                "the requested camera condition requires Waymo metric poses."
            )
        if not torch.is_tensor(intrinsics_canvas_gt):
            raise RuntimeError(
                "Raw Waymo pretrain batch is missing camera_intrinsics_canvas. "
                "The camera condition must use the same post-crop canvas K as M/G."
            )
        raw_hw = batch.get("raw_image_size_hw")
        if raw_hw is None:
            raise RuntimeError(
                "Raw Waymo pretrain batch is missing raw_image_size_hw. Repair the data/cache metadata before training."
            )
        c2w_all = camera_to_world_gt.to(device=device, dtype=torch.float32, non_blocking=True)
        if c2w_all.ndim not in (4, 5):
            raise ValueError(f"camera_to_world_corrected must be [B,S,V,4,4] or [B,S,4,4], got {tuple(c2w_all.shape)}")
        intrinsics_all = intrinsics_gt.to(device=device, dtype=torch.float32, non_blocking=True)
        intrinsics_canvas_all = intrinsics_canvas_gt.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        requested_intrinsics_raw = _requested_front_intrinsics(
            intrinsics_all,
            batch_size=int(batch_size_raw),
            seq_len=int(seq_len),
            device=device,
            name="intrinsics",
        )
        requested_intrinsics_canvas = _requested_front_intrinsics(
            intrinsics_canvas_all,
            batch_size=int(batch_size_raw),
            seq_len=int(seq_len),
            device=device,
            name="camera_intrinsics_canvas",
        )
        raw_hw_front = _canonical_raw_image_size_hw(
            raw_hw,
            batch_size=int(batch_size_raw),
            device=device,
        )
        trajectory_anchor_raw = batch.get("camera_trajectory_anchor_to_world_corrected")
        previous_camera_raw = batch.get("camera_previous_to_world_corrected")
        if not torch.is_tensor(trajectory_anchor_raw) or not torch.is_tensor(previous_camera_raw):
            raise RuntimeError(
                "Requested camera conditioning requires camera_trajectory_anchor_to_world_corrected and "
                "camera_previous_to_world_corrected."
            )
        trajectory_anchor_metric = trajectory_anchor_raw.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if trajectory_anchor_metric.ndim == 4 and int(trajectory_anchor_metric.shape[1]) == 1:
            trajectory_anchor_metric = trajectory_anchor_metric[:, 0]
        previous_camera_metric_all = previous_camera_raw.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if previous_camera_metric_all.ndim == 5:
            previous_camera_metric_all = previous_camera_metric_all[:, :, 0]
        camera_condition_tokens, camera_attention_mask = camera_condition_from_waymo_request(
            c2w_all,
            requested_intrinsics_canvas,
            image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            trajectory_anchor_to_world=trajectory_anchor_metric,
            previous_camera_to_world=previous_camera_metric_all,
        )
        scene_gauge_raw = batch.get("scene_gauge")
        scene_gauge_valid_raw = batch.get("scene_gauge_valid")
        if not torch.is_tensor(scene_gauge_raw) or not torch.is_tensor(scene_gauge_valid_raw):
            raise RuntimeError("Raw SceneFlow pretraining requires scene_gauge and scene_gauge_valid from the offline table")
        scene_gauge_clean = scene_gauge_raw.to(
            device=device, dtype=torch.float32, non_blocking=True
        ).view(int(images.shape[0]), 1, SCENE_GAUGE_DIM)
        scene_gauge_valid = scene_gauge_valid_raw.to(
            device=device, dtype=torch.bool, non_blocking=True
        ).view(int(images.shape[0]), 1, SCENE_GAUGE_DIM)
        gauge_fill = sf_root.gauge_mean.to(
            device=device, dtype=scene_gauge_clean.dtype
        ).view(1, 1, SCENE_GAUGE_DIM)
        # The dataset uses a finite zero sentinel for a JSON-null invalid
        # channel. Keep that raw value only for masked direct supervision;
        # every physical consumer must receive the training-mean fallback.
        scene_gauge_effective = torch.where(
            scene_gauge_valid, scene_gauge_clean, gauge_fill
        )
        scene_gauge_clean_n = sf_root.normalize_gauge(scene_gauge_effective)
        del outputs, aggregated_tokens_list, image_tokens_context, image_tokens_list, tokens_4, z_clean

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
    sky_gen_observation = None
    sky_atlas_clean = None
    sky_atlas_observation_mask = None
    if sky_generation_enabled(args):
        camera_pose_sky_gauge = assemble_dggt_pose_encoding(
            dggt_pose_encoding_to_camera_to_world(camera_pose_gt_dggt),
            scene_gauge_effective,
        )
        sky_h, sky_w = sky_grid_shape(args)
        atlas_h, atlas_w = sky_atlas_shape(args)
        # Sky directions live in the frozen teacher camera-anchor world
        # (identity camera => image-up is world -y).  Only replace the noisy
        # per-frame CameraHead FOV with the trunk-constant gauge FOV; using the
        # Waymo ego world here would rotate the atlas into a +z-up basis.
        sky_extrinsics, sky_intrinsics = pose_encoding_to_extri_intri(
            camera_pose_sky_gauge,
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
        sky_gen_loss_weight = pack_sky_atlas_loss_weight(
            sky_atlas_observation_mask,
            unobserved_weight=float(
                getattr(args, "sky_unobserved_loss_weight", DEFAULT_SKY_UNOBSERVED_LOSS_WEIGHT)
            ),
        )
        # The same packing at zero unobserved weight is exactly the observation
        # map, which ``sky_flow_loss`` needs to normalize the two regions apart.
        # The blended weight above stays for logging and validation metrics.
        sky_gen_observation = pack_sky_atlas_loss_weight(
            sky_atlas_observation_mask, unobserved_weight=0.0
        )

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
    if not torch.equal(frame_ids, window_indices):
        raise ValueError(
            "frame_ids and dggt_window_indices must describe the same clip-global frames: "
            f"frame_ids={frame_ids.tolist()} dggt_window_indices={window_indices.tolist()}"
        )
    bundle = build_full_scene_bundle(
        z_clean_n,
        camera_condition_tokens=camera_condition_tokens,
        camera_attention_mask=camera_attention_mask,
        sky_gen_clean=sky_gen_clean,
        sky_gen_loss_weight=sky_gen_loss_weight,
        sky_gen_observation=sky_gen_observation,
        sky_gen_attention_mask=None,
        scene_gauge_clean_n=scene_gauge_clean_n,
        scene_gauge_clean=scene_gauge_clean,
        scene_gauge_valid=scene_gauge_valid,
        sky_mask_clean=sky_mask_clean,
        sky_mask_refined_clean=sky_mask_refined_clean,
        frame_ids=frame_ids.contiguous(),
    )
    bundle.sky_atlas_clean = sky_atlas_clean
    bundle.sky_atlas_observation_mask = sky_atlas_observation_mask
    bundle.scene_gauge_effective = scene_gauge_effective
    bundle.camera_trajectory_anchor_to_world_metric = trajectory_anchor_metric
    bundle.camera_to_world_requested_metric = (
        c2w_all[:, :, 0] if c2w_all.ndim == 5 else c2w_all
    ).detach()
    bundle.camera_intrinsics_requested_raw_metric = (
        requested_intrinsics_raw.detach()
    )
    bundle.camera_intrinsics_requested_canvas_metric = (
        requested_intrinsics_canvas.detach()
    )
    bundle.camera_requested_raw_image_size_hw = raw_hw_front.detach()
    bundle.camera_requested_canvas_image_size_hw = (
        int(images.shape[-2]),
        int(images.shape[-1]),
    )
    metric_lidar_depth = (
        batch.get("metric_lidar_depth_m")
        if include_metric_depth_diagnostic
        else None
    )
    bundle.metric_lidar_depth_m = (
        metric_lidar_depth.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        if torch.is_tensor(metric_lidar_depth)
        else None
    )
    metric_lidar_valid = (
        batch.get("metric_lidar_depth_valid")
        if include_metric_depth_diagnostic
        else None
    )
    bundle.metric_lidar_depth_valid = (
        metric_lidar_valid.to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        if torch.is_tensor(metric_lidar_valid)
        else None
    )
    layout_built = build_layout_condition_from_batch(
        batch,
        vggt_model,
        scene_flow,
        device,
        layout_max_actors=int(args.layout_max_actors),
        patch_grid=args.patch_grid,
    )
    batch_size = int(z_clean_n.shape[0])
    bundle.layout_condition = layout_built.layout
    bundle.appearance_class_id = layout_built.appearance_class_id
    fps_raw = batch.get("pretrain_fps", 10.0)
    bundle.fps = torch.as_tensor(
        fps_raw, device=device, dtype=torch.float32
    ).reshape(-1)
    if int(bundle.fps.numel()) == 1 and batch_size > 1:
        bundle.fps = bundle.fps.expand(batch_size)
    if tuple(bundle.fps.shape) != (batch_size,):
        raise ValueError(f"pretrain_fps must be scalar or [B], got {tuple(bundle.fps.shape)}")
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
    *,
    long_loader: DataLoader | None = None,
    validation_index: int = 0,
) -> dict[str, float]:
    with preserve_validation_rng_state(device):
        return _run_validation_impl(
            loader,
            vggt_model,
            scene_flow,
            scheduler,
            device,
            args,
            step,
            log_dir,
            wandb_run,
            ema,
            text_encoder,
            long_loader=long_loader,
            validation_index=validation_index,
        )


@torch.no_grad()
def _run_validation_impl(
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
    *,
    long_loader: DataLoader | None = None,
    validation_index: int = 0,
) -> dict[str, float]:
    scene_flow_was_training = scene_flow.training
    scene_flow.eval()
    use_val_ema = ema is not None and not args.no_val_ema
    ema_params = list(unwrap_ddp(scene_flow).parameters()) if use_val_ema else None
    if use_val_ema:
        ema.store(ema_params)
        ema.copy_to(ema_params)

    sums: dict[str, TrainLogSeries] = {}
    observation_counts: dict[str, int] = {}
    count = 0
    first_batch: dict[str, Any] | None = None
    validation_generator = make_validation_generator(device, int(args.seed))
    guidance_scales = validation_guidance_scales(args)
    sampling_enabled = int(getattr(args, "val_log_images", 0)) > 0
    world_size = dist.get_world_size() if is_distributed() else 1
    requested_scene_count = int(getattr(args, "val_inference_scenes", 1))
    effective_scene_count = effective_validation_scene_count(
        len(guidance_scales),
        requested_scene_count,
        world_size=world_size,
    )
    assigned_sampling_tasks = (
        validation_sampling_tasks_for_rank(
            len(guidance_scales),
            requested_scene_count,
            rank=get_rank(),
            world_size=world_size,
        )
        if sampling_enabled
        else ()
    )
    # Only rank 0 contributes the scalar validation loss, and only ranks with
    # an assigned CFG scale need a visualization batch.  Iterating the loader
    # on every surplus rank used to start its workers and prefetch many costly
    # online-layout samples that were immediately discarded.  Restricting I/O
    # here does not change any reported value: non-main scalar metrics were
    # never gathered, while assigned scale ranks still consume the same
    # deterministic long-form batch used by the original sampling path.
    needs_scalar_batches = is_main_process()
    # A configured long-form loader replaces the short batch before sampling,
    # so loading the short batch solely for visualization would be wasted I/O.
    needs_visualization_batch = bool(assigned_sampling_tasks) and long_loader is None
    local_batch_limit = (
        int(args.val_batches)
        if needs_scalar_batches
        else (1 if needs_visualization_batch else 0)
    )
    if local_batch_limit > 0:
        iterator = loader
        if is_main_process() and use_interactive_tqdm(
            args.no_tqdm,
            force_tqdm=bool(getattr(args, "force_tqdm", False)),
        ):
            force_web_tqdm = bool(getattr(args, "force_tqdm", False))
            iterator = tqdm(
                loader,
                total=args.val_batches,
                desc=f"val {step:06d}",
                dynamic_ncols=not force_web_tqdm,
                leave=False,
                file=tqdm_output_stream(force_tqdm=force_web_tqdm),
            )
        for batch in islice(iterator, local_batch_limit):
            if first_batch is None and needs_visualization_batch:
                first_batch = _slice_batch_for_visualization(batch, max_samples=1)
            if needs_scalar_batches:
                loss, logs = train_step(
                    batch,
                    vggt_model,
                    scene_flow,
                    scheduler,
                    device,
                    args,
                    text_encoder,
                    global_step=step,
                    generator=validation_generator,
                )
                logs = dict(logs)
                logs["loss"] = float(loss.detach().item())
                accumulate_wandb_metrics(sums, observation_counts, logs)
                count += 1
    metrics = finalize_wandb_metrics(sums, observation_counts)
    metrics["batches"] = float(count)

    if sampling_enabled:
        synchronize_validation_model(scene_flow)
    sampling_args = args
    visualization_batches: dict[int, dict[str, Any]] = {}
    if assigned_sampling_tasks and long_loader is not None:
        long_sampler = getattr(long_loader, "sampler", None)
        if not isinstance(long_sampler, CyclicSequentialSampler):
            raise TypeError("long-form validation requires CyclicSequentialSampler")
        scene_offsets = sorted({task[0] for task in assigned_sampling_tasks})
        for scene_offset in scene_offsets:
            long_sampler.set_offset(
                validation_scene_dataset_index(
                    scene_offset,
                    scene_count=effective_scene_count,
                    validation_index=validation_index,
                    trunk_major_index=long_sampler.data_source.trunk_major_index,
                )
            )
            try:
                long_batch = next(iter(long_loader))
            except StopIteration:
                continue
            visualization_batches[scene_offset] = _slice_batch_for_visualization(
                long_batch, max_samples=1
            )
        sampling_args = argparse.Namespace(**vars(args))
        sampling_args.val_sliding_window = int(args.sequence_length)
        sampling_args.val_sliding_stride = pretrain_validation_stride(
            int(sampling_args.val_sliding_window),
            int(getattr(args, "val_sliding_stride", 0) or 0),
        )
        if visualization_batches:
            representative_batch = next(iter(visualization_batches.values()))
            metrics["long_sliding_rollout_frames"] = float(
                representative_batch["images"].shape[1]
            )
            metrics["long_sliding_window"] = float(
                sampling_args.val_sliding_window
            )
            metrics["long_sliding_stride"] = float(
                sampling_args.val_sliding_stride
            )
    elif assigned_sampling_tasks and first_batch is not None:
        if effective_scene_count != 1:
            raise RuntimeError(
                "multi-scene validation inference requires the long-form loader"
            )
        # Single-scene fallback: that one scene is pinned by definition.
        visualization_batches[0] = first_batch

    metric_keys_before_sampling = set(metrics)
    local_mosaic_rows: list[dict[str, Any]] = []
    if assigned_sampling_tasks and visualization_batches:
        torch.cuda.empty_cache()
        bundle_cache: dict[int, Any] = {}
        for local_index, (scene_offset, scale_index) in enumerate(
            assigned_sampling_tasks
        ):
            first_batch = visualization_batches.get(scene_offset)
            if first_batch is None:
                continue
            first_bundle = bundle_cache.get(scene_offset)
            if first_bundle is None:
                first_bundle = build_pretrain_bundle_from_batch(
                    first_batch,
                    vggt_model,
                    scene_flow,
                    device,
                    args,
                )
                bundle_cache[scene_offset] = first_bundle
            scale = guidance_scales[scale_index]
            is_scene_primary = scale_index == 0
            # Rotating scenes are here for the pictures.  Their numbers stay in
            # this local dict and are dropped, so every validation/sample_*
            # point is a mean over the same pinned scenes and a change between
            # two steps is the model, not the draw.
            scene_metrics: dict[str, float] = {}
            scene_label = validation_scene_label(first_batch, scene_offset)
            generated = cfg_sample_pretrain_latents(
                scene_flow,
                first_bundle,
                sampling_args,
                step,
                device,
                guidance_scale=scale,
                text_encoder=text_encoder,
                return_sky=sky_generation_enabled(args),
                return_gauge=True,
                return_sky_mask=True,
                # Pinned sample_* series must use the same draw at every
                # training step. A slot-specific seed also keeps all CFG
                # scales for one scene on identical initial noise.
                sampling_seed=validation_scene_sampling_seed(
                    int(args.seed), int(scene_offset)
                ),
            )
            if not isinstance(generated, SimpleNamespace):
                raise RuntimeError("validation sampling must return structured outputs")
            if generated.gauge is None:
                raise RuntimeError("validation sampling did not return gauge")
            if generated.sky_mask_patch is None or generated.sky_mask_refined is None:
                raise RuntimeError("validation sampling did not return sky masks")
            prefix_suffix = "" if is_scene_primary else f"_cfg{scale:g}"
            scene_metrics.update(
                sampled_gauge_validation_metrics(
                    generated.gauge,
                    first_bundle.scene_gauge_clean,
                    first_bundle.scene_gauge_valid,
                    prior_log_scale=unwrap_ddp(scene_flow).gauge_mean[0],
                    prefix=f"sample_gauge{prefix_suffix}",
                )
            )
            scene_metrics.update(
                sampled_latent_validation_metrics(
                    generated.video,
                    first_bundle.z_clean_n,
                    prefix=f"sample_latent{prefix_suffix}",
                )
            )
            if generated.sky is not None and first_bundle.sky_gen_clean is not None:
                scene_metrics.update(
                    sky_token_validation_metrics(
                        generated.sky,
                        first_bundle.sky_gen_clean,
                        prefix=f"sample_sky{prefix_suffix}",
                        loss_weight=first_bundle.sky_gen_loss_weight,
                    )
                )
            scene_metrics.update(
                sky_mask_validation_metrics(
                    generated.sky_mask_patch,
                    first_bundle.sky_mask_clean,
                    prefix=f"sample_sky_mask{prefix_suffix}",
                )
            )
            scene_metrics.update(
                sky_mask_validation_metrics(
                    generated.sky_mask_refined,
                    first_bundle.sky_mask_refined_clean,
                    prefix=f"sample_sky_mask_refine{prefix_suffix}",
                )
            )
            rgb_images = None
            if not args.no_val_render_rgb:
                torch.cuda.empty_cache()
                rgb_images = render_validation_generated_rgb(
                    first_batch,
                    vggt_model,
                    scene_flow,
                    generated.video,
                    args,
                    device,
                    requested_camera_to_world=(
                        first_bundle.camera_to_world_requested_metric
                    ),
                    trajectory_anchor_to_world=(
                        first_bundle.camera_trajectory_anchor_to_world_metric
                    ),
                    generated_gauge=generated.gauge,
                    generated_sky_tokens=generated.sky,
                    generated_sky_mask_patch=generated.sky_mask_patch,
                    generated_sky_mask_refined=generated.sky_mask_refined,
                )
                generated_rgb = rgb_images.get("generated_raw_3dgs_rgb")
                target_rgb = first_batch.get("images") if first_batch else None
                if torch.is_tensor(generated_rgb) and torch.is_tensor(target_rgb):
                    scene_metrics.update(
                        sampled_render_validation_metrics(
                            generated_rgb,
                            _image_grid(target_rgb, int(generated_rgb.shape[0])),
                            prefix=f"sample_render{prefix_suffix}",
                        )
                    )
            metrics.update(
                validation_scene_metrics_for_merge(
                    scene_metrics,
                    scene_offset=scene_offset,
                    scene_count=effective_scene_count,
                )
            )
            local_mosaic_rows.extend(
                collect_validation_mosaic_rows(
                    first_bundle,
                    generated.video,
                    rgb_images,
                    args,
                    scene_slot=scene_offset,
                    scene_label=scene_label,
                    guidance_scale=scale,
                    scale_index=scale_index,
                    is_primary=is_scene_primary,
                    visualization_batch=(
                        first_batch if is_scene_primary else None
                    ),
                )
            )
            if local_index + 1 < len(assigned_sampling_tasks):
                torch.cuda.empty_cache()

    local_result = {
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in metric_keys_before_sampling
        },
        "mosaic_rows": local_mosaic_rows,
    }
    gathered_results = all_gather_validation_results(local_result, device)
    if is_main_process():
        sampled_metrics, gathered_mosaic_rows = merge_validation_rank_results(
            gathered_results
        )
        metrics.update(sampled_metrics)
        if sampling_enabled:
            metrics["inference_scene_count"] = float(effective_scene_count)
            metrics["inference_cfg_count"] = float(len(guidance_scales))
        mosaic_paths = write_validation_mosaics(
            gathered_mosaic_rows, log_dir, step, args
        )
        if wandb_run is not None and mosaic_paths:
            import wandb

            wandb_run.log(
                {
                    f"validation/{name}": wandb.Image(str(path))
                    for name, path in mosaic_paths.items()
                },
                step=step,
            )
        metrics_text = " | ".join(
            f"{key}={value:.4f}" for key, value in metrics.items()
        )
        print(f"[validation {step:06d}] {metrics_text}", flush=True)
        log_wandb(wandb_run, metrics, step, "validation")

    if use_val_ema:
        ema.restore(ema_params)
    if scene_flow_was_training:
        scene_flow.train()
    if is_distributed():
        dist.barrier()
    return metrics




def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneFlow pretraining on raw Waymo clips.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, default=None)
    parser.add_argument(
        "--hdmap_root",
        type=str,
        required=True,
        help="Root containing the training split's per-scene layout-v2 sidecars.",
    )
    parser.add_argument(
        "--val_hdmap_root",
        type=str,
        default=None,
        help=(
            "Root containing validation layout-v2 sidecars. Required when "
            "--val_image_dir differs from --image_dir."
        ),
    )
    parser.add_argument("--dggt_ckpt_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_ckpt_path",
        type=str,
        default=None,
        help=(
            "JointSceneTokenizer checkpoint. It may be omitted only if --dggt_ckpt_path "
            "embeds a complete tokenizer state; otherwise startup fails."
        ),
    )
    parser.add_argument(
        "--feature_stats_path",
        type=str,
        default=str(DEFAULT_SCENE_FLOW_FEATURE_STATS_PATH),
    )
    parser.add_argument(
        "--scene_gauge_path",
        type=str,
        required=True,
        help="Offline full-29-frame teacher gauge table for the training split.",
    )
    parser.add_argument(
        "--val_scene_gauge_path",
        type=str,
        default=None,
        help="Validation split gauge table; required when validation is enabled.",
    )
    parser.add_argument(
        "--pullback_calibration_path",
        type=str,
        required=True,
        help="Numeric tokenizer pullback calibration used at metric boundaries.",
    )
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
    parser.add_argument("--text_encoder_path", type=str, default="/home/dancer/model/Qwen/Qwen3-0.6B")
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--no_text_condition", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument(
        "--resume_expected_step",
        type=int,
        default=-1,
        help=(
            "Optional strict step assertion for --resume_path. A non-negative value "
            "fails before loading optimizer/model state when the checkpoint step differs."
        ),
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
            "consume z_t directly; in_channels records the packed control "
            "dimensionality 3 * latent_dim + 3. "
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
    gradient_checkpointing_group = parser.add_mutually_exclusive_group()
    gradient_checkpointing_group.add_argument(
        "--gradient_checkpointing",
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable SceneFlow activation checkpointing to reduce training memory.",
    )
    gradient_checkpointing_group.add_argument(
        "--no_gradient_checkpointing",
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable SceneFlow activation checkpointing to avoid backward recomputation.",
    )
    gradient_checkpointing_group.add_argument(
        "--half_gradient_checkpointing",
        "--half-gradient-checkpointing",
        dest="half_gradient_checkpointing",
        action="store_true",
        help="Checkpoint alternating SceneFlow encoder and DDT blocks.",
    )
    gradient_checkpointing_group.add_argument(
        "--three_quarter_gradient_checkpointing",
        "--three-quarter-gradient-checkpointing",
        dest="three_quarter_gradient_checkpointing",
        action="store_true",
        help="Checkpoint three of every four SceneFlow encoder blocks and no DDT blocks.",
    )
    parser.set_defaults(
        gradient_checkpointing=True,
        half_gradient_checkpointing=False,
        three_quarter_gradient_checkpointing=False,
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=0,
        help=(
            "Validation DataLoader workers per rank. Validation consumes only a "
            "few batches, so this is intentionally independent of training workers "
            "to avoid prefetching large unused layout samples."
        ),
    )
    parser.add_argument(
        "--dataloader_worker_threads",
        type=int,
        default=1,
        help=(
            "Torch intra-op threads inside each DataLoader worker. Keep at one when "
            "many ranks perform online HD-map projection on the same node."
        ),
    )
    parser.add_argument(
        "--dataloader_out_of_order",
        action="store_true",
        help=(
            "Yield ready training samples without waiting for a slower earlier "
            "worker. This changes only the order of an already shuffled epoch; "
            "validation remains ordered."
        ),
    )
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
        help=(
            "Enable DataLoader pin_memory. Production launchers enable it after "
            "measured H2D gains; the CLI default stays off for low-host-memory smoke tests."
        ),
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
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help=(
            "Newline-terminated plain-text log cadence when tqdm is disabled "
            "explicitly or stderr is not an interactive terminal."
        ),
    )
    parser.add_argument("--wandb_log_every", type=int, default=50,
                        help="Report averaged training metrics to wandb every N optimizer steps.")
    parser.add_argument(
        "--no_rank_metric_mean",
        action="store_true",
        help=(
            "Report rank 0's own metrics instead of the all-rank mean. This "
            "drops the one small all-reduce the training loop issues per step, "
            "which is the switch to try first if a run stops advancing with "
            "every accelerator still busy."
        ),
    )
    parser.add_argument("--val_scene_start", type=int, default=None)
    parser.add_argument("--val_scene_end", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=2000)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--val_log_images", type=int, default=10)
    parser.add_argument(
        "--val_mosaic_cell_width",
        type=int,
        default=MOSAIC_CELL_WIDTH_DEFAULT,
        help=(
            "Pixel width of one frame in the validation mosaic. The mosaic is "
            "val_log_images cells wide, so 10 frames at 256 give a 2560px image."
        ),
    )
    parser.add_argument(
        "--val_inference_scenes",
        type=int,
        default=10,
        help=(
            "Number of scenes sampled at each validation. The first half is "
            "pinned and is the only source of the validation/sample_* numbers; "
            "the second half rotates through the split for fresh mosaics and "
            "contributes no numbers. If world_size is smaller than scenes "
            "times the number of CFG scales, validation falls back to "
            f"{VALIDATION_SCENE_FALLBACK} scenes (one of each) rather than "
            "serializing all scene jobs."
        ),
    )
    parser.add_argument(
        "--val_sample_steps",
        type=int,
        default=T59_VALIDATION_SAMPLE_STEPS,
        help=(
            "Frozen validation ODE step count used by the accepted T59 sky-mask "
            "working-sigma decision."
        ),
    )
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
    parser.add_argument("--lambda_gauge_flow", type=float, default=0.1)
    parser.add_argument("--lambda_gauge_direct", type=float, default=1.0)
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
        default=DEFAULT_LAMBDA_SKY_FLOW,
        help="Flow-matching loss weight for generated scene-level sky RGB tokens.",
    )
    parser.add_argument(
        "--sky_unobserved_loss_beta",
        type=float,
        default=DEFAULT_SKY_UNOBSERVED_LOSS_BETA,
        help=(
            "Weight of the spherical-completion prior relative to the observed "
            "atlas in the sky flow loss, each normalized on its own region."
        ),
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
        default=SKY_MASK_REFINE_BOUNDARY_LOSS_WEIGHT_DEFAULT,
        help=(
            "Weight of the boundary BCE term inside the refined sky mask "
            "auxiliary loss. It enters as boundary_loss_weight * "
            "boundary_weight, so 0.125 * 4 = 0.5. At the previous 0.25 the "
            "product was 1.0 and the boundary BCE alone was 86% of "
            "loss_sky_mask_refine and 5.4% of the whole training loss -- "
            "seventeen times the entire world-feedback stack, on the slowest "
            "moving term in the objective (-2.4%/1k against the flow loss's "
            "-6.7%/1k)."
        ),
    )
    parser.add_argument(
        "--lambda_rgb_render",
        type=float,
        default=RGB_RENDER_LAMBDA_DEFAULT,
        help=(
            "Differentiable RGB loss using generated depth/GS, requested C, and "
            "the generated scene gauge. "
            "Rows retain per-rank input order under the resource cap and keep "
            "continuous sigma weighting."
        ),
    )
    parser.add_argument(
        "--lambda_level_consistency",
        type=float,
        default=LEVEL_CONSISTENCY_LAMBDA_DEFAULT,
        help="Four-level tokenizer-decoder consistency weight, evaluated on RGB render steps.",
    )
    parser.add_argument(
        "--lambda_head_consistency",
        type=float,
        default=HEAD_CONSISTENCY_LAMBDA_DEFAULT,
        help="Frozen depth/GS/dynamic-head consistency weight, evaluated on RGB render steps.",
    )
    parser.add_argument(
        "--head_dynamic_space",
        type=str,
        default="probability",
        choices=list(DYNAMIC_HEAD_SPACES),
        help=(
            "Space the dynamic-head consistency term compares in. 'probability' "
            "applies the sigmoid the renderer already uses for static opacity; "
            "'logit' restores the unbounded pre-v6 comparison, where this single "
            "term carried ~98% of the head loss and stopped improving."
        ),
    )
    parser.add_argument("--rgb_render_every", type=int, default=1)
    parser.add_argument(
        "--metric_depth_diagnostic_every",
        type=int,
        default=500,
        help=(
            "Independent no-grad LiDAR metric-depth diagnostic cadence. It is "
            "not gated by RGB-render warm-up; 0 disables it."
        ),
    )
    parser.add_argument(
        "--metric_depth_diagnostic_start_step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--metric_depth_diagnostic_max_samples",
        type=int,
        default=1,
        help="Maximum rows decoded per physical diagnostic step; 0 means all rows.",
    )
    parser.add_argument(
        "--rgb_render_start_step",
        type=int,
        default=5000,
        help="First optimizer step eligible for RGB supervision.",
    )
    parser.add_argument("--rgb_render_warmup_steps", type=int, default=5000)
    parser.add_argument(
        "--rgb_render_sigma_power",
        type=float,
        default=2.0,
        help=(
            "Continuously attenuate render/view reconstruction at noisy timesteps "
            "with w(sigma)=(1-sigma)^power; 0 disables sigma weighting."
        ),
    )
    parser.add_argument(
        "--feedback_conf_weight_power",
        type=float,
        default=1.0,
        help=(
            "Teacher depth-confidence exponent for reconstruction feedback; "
            "0 disables confidence weighting."
        ),
    )
    parser.add_argument(
        "--feedback_conf_weight_floor",
        type=float,
        default=0.05,
        help="Lower clamp for teacher depth confidence before weighting.",
    )
    parser.add_argument("--rgb_render_max_samples", type=int, default=1)
    parser.add_argument(
        "--rgb_render_max_frames",
        type=int,
        default=0,
        help="Maximum RGB-supervised frames per clip; 0 uses the full training sequence.",
    )
    parser.add_argument(
        "--rgb_render_stride",
        type=int,
        default=1,
        help="Spatial rendering stride; 1 keeps the full image resolution.",
    )
    parser.add_argument("--rgb_render_sky_weight", type=float, default=1.0)
    parser.add_argument(
        "--rgb_render_sky_mask_grad_scale",
        type=float,
        default=0.05,
        help="RGB-to-sky-mask gradient scale; render always uses the predicted mask.",
    )
    parser.add_argument("--rgb_render_lpips_weight", type=float, default=0.01)
    parser.add_argument("--rgb_render_lpips_net", type=str, default="alex")
    parser.add_argument(
        "--lambda_sky_view_reconstruction",
        type=float,
        default=SKY_VIEW_LAMBDA_DEFAULT,
    )
    parser.add_argument("--sky_view_lpips_weight", type=float, default=0.01)
    parser.add_argument(
        "--sky_view_high_frequency_weight",
        type=float,
        default=SKY_VIEW_HIGH_FREQUENCY_WEIGHT_DEFAULT,
    )
    parser.add_argument(
        "--sky_view_start_step",
        type=int,
        default=SKY_VIEW_START_STEP_DEFAULT,
        help=(
            "First step at which the sky-view reconstruction loss applies. It "
            "is independent of --rgb_render_start_step because the sky atlas "
            "renders by rotation only, so it needs the gauge FOV but neither "
            "the scene latent nor the gauge scale."
        ),
    )
    parser.add_argument(
        "--sky_view_warmup_steps", type=int, default=SKY_VIEW_WARMUP_STEPS_DEFAULT
    )

    parser.add_argument(
        "--cfg",
        dest="guidance_scale",
        type=float,
        default=1.0,
        help="Text guidance scale used by chained CFG.",
    )
    parser.add_argument("--layout_guidance_scale", type=float, default=1.0)
    parser.add_argument("--asset_control_guidance_scale", type=float, default=1.0)
    parser.add_argument("--layout_max_actors", type=int, default=96)
    parser.add_argument(
        "--static_far_plane_m",
        type=float,
        default=STATIC_FAR_PLANE_M,
        help="Frozen camera optical-depth far plane for static HD-map geometry.",
    )
    parser.add_argument("--layout_depth_tau", type=float, default=0.5)
    parser.add_argument(
        "--layout_to_gauge_grad_scale",
        type=float,
        default=1.0,
        help="Upper bound of the mandatory 5k-to-15k layout/gauge gradient ramp.",
    )
    for option in (
        "layout_map_injection",
        "layout_actor_injection",
        "layout_map_metric_injection",
        "layout_actor_metric_injection",
        "appearance_context_injection",
    ):
        parser.add_argument(
            f"--{option}",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
    parser.add_argument(
        "--text_uncond_drop_prob",
        dest="text_uncond_drop_prob",
        type=float,
        default=0.1,
        help="Per-sample probability of replacing text with the null/empty prompt.",
    )
    parser.add_argument("--val_guidance_scales", type=str, default="",
                        help="Comma-separated extra text-CFG scales for validation RGB.")

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
    tqdm_group = parser.add_mutually_exclusive_group()
    tqdm_group.add_argument(
        "--no_tqdm",
        action="store_true",
        help="Disable tqdm and emit newline-delimited progress logs.",
    )
    tqdm_group.add_argument(
        "--force_tqdm",
        action="store_true",
        help=(
            "Enable tqdm even when stderr is not a TTY. Intended for Web log "
            "consoles that support carriage-return progress and need ETA/rate."
        ),
    )
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
    kwargs = dataloader_runtime_kwargs_for_workers(args, int(args.num_workers))
    if kwargs:
        kwargs["in_order"] = not bool(args.dataloader_out_of_order)
    return kwargs


def _configure_dataloader_worker(
    _worker_id: int,
    *,
    torch_num_threads: int,
) -> None:
    """Keep one loader process from oversubscribing all host CPU cores.

    PyTorch seeds Python, NumPy and torch before invoking ``worker_init_fn``;
    this hook deliberately leaves those RNG states untouched.  It only caps
    intra-op torch kernels used by online layout projection, so the dataset's
    window/reference sampling contract is unchanged.
    """

    torch.set_num_threads(max(1, int(torch_num_threads)))


def dataloader_runtime_kwargs_for_workers(
    args: argparse.Namespace,
    num_workers: int,
) -> dict[str, Any]:
    if int(num_workers) <= 0:
        return {}
    return {
        "prefetch_factor": max(1, int(args.prefetch_factor)),
        "persistent_workers": not bool(args.no_persistent_workers),
        "worker_init_fn": partial(
            _configure_dataloader_worker,
            torch_num_threads=max(1, int(args.dataloader_worker_threads)),
        ),
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
    if args.sky_atlas_hw != DEFAULT_SKY_ATLAS_HW or args.sky_grid != DEFAULT_SKY_GRID:
        raise ValueError(
            f"{SKY_REPRESENTATION_VERSION} requires sky_atlas_hw={DEFAULT_SKY_ATLAS_HW} "
            f"and sky_grid={DEFAULT_SKY_GRID}"
        )
    if not 0.0 <= float(args.sky_unobserved_loss_weight) <= 1.0:
        raise ValueError("--sky_unobserved_loss_weight must be in [0, 1].")
    if args.sequence_length < 2:
        raise ValueError("--sequence_length must be >= 2.")
    if args.sequence_length > 29:
        raise ValueError("--sequence_length cannot exceed the 29-frame caption/DGGT context.")
    if int(args.val_sample_steps) != T59_VALIDATION_SAMPLE_STEPS:
        raise ValueError(
            "--val_sample_steps is frozen to "
            f"{T59_VALIDATION_SAMPLE_STEPS} by the accepted T59 decision."
        )
    generation_loss_weights = {
        "--lambda_gauge_flow": args.lambda_gauge_flow,
        "--lambda_gauge_direct": args.lambda_gauge_direct,
        "--lambda_sky_flow": args.lambda_sky_flow,
    }
    for name, value in generation_loss_weights.items():
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if int(args.pretrain_instance_cache_size) < 0:
        raise ValueError("--pretrain_instance_cache_size must be non-negative.")
    if int(args.num_workers) < 0 or int(args.val_num_workers) < 0:
        raise ValueError("--num_workers and --val_num_workers must be non-negative.")
    if int(args.dataloader_worker_threads) <= 0:
        raise ValueError("--dataloader_worker_threads must be positive.")
    if float(args.lambda_rgb_render) < 0.0:
        raise ValueError("--lambda_rgb_render must be non-negative.")
    if float(args.lambda_level_consistency) < 0.0:
        raise ValueError("--lambda_level_consistency must be non-negative.")
    if float(args.lambda_head_consistency) < 0.0:
        raise ValueError("--lambda_head_consistency must be non-negative.")
    if (
        float(args.lambda_level_consistency) > 0.0
        or float(args.lambda_head_consistency) > 0.0
    ) and not rgb_render_loss_enabled(args):
        raise ValueError(
            "Reconstruction feedback shares the RGB render schedule and requires "
            "--lambda_rgb_render > 0 with --rgb_render_every > 0. Set both feedback "
            "weights to zero when disabling the render path."
        )
    if int(args.rgb_render_every) < 0:
        raise ValueError("--rgb_render_every must be >= 0.")
    if int(args.metric_depth_diagnostic_every) < 0:
        raise ValueError("--metric_depth_diagnostic_every must be >= 0.")
    if int(args.metric_depth_diagnostic_start_step) < 0:
        raise ValueError("--metric_depth_diagnostic_start_step must be >= 0.")
    if int(args.metric_depth_diagnostic_max_samples) < 0:
        raise ValueError("--metric_depth_diagnostic_max_samples must be >= 0.")
    if int(args.rgb_render_max_samples) < 0 or int(args.rgb_render_max_frames) < 0:
        raise ValueError("--rgb_render_max_samples and --rgb_render_max_frames must be non-negative; 0 means all.")
    if int(args.rgb_render_stride) <= 0:
        raise ValueError("--rgb_render_stride must be positive.")
    if int(args.rgb_render_start_step) < 0 or int(args.rgb_render_warmup_steps) < 0:
        raise ValueError("RGB render start/warmup steps must be non-negative.")
    if not math.isfinite(float(args.rgb_render_sigma_power)) or float(args.rgb_render_sigma_power) < 0.0:
        raise ValueError("--rgb_render_sigma_power must be finite and non-negative.")
    if (
        not math.isfinite(float(args.feedback_conf_weight_power))
        or float(args.feedback_conf_weight_power) < 0.0
    ):
        raise ValueError("--feedback_conf_weight_power must be finite and non-negative.")
    if (
        not math.isfinite(float(args.feedback_conf_weight_floor))
        or not 0.0 < float(args.feedback_conf_weight_floor) <= 1.0
    ):
        raise ValueError("--feedback_conf_weight_floor must be finite and in (0, 1].")
    if not 0.0 <= float(args.rgb_render_sky_weight) <= 1.0:
        raise ValueError("--rgb_render_sky_weight must be in [0,1].")
    if float(args.rgb_render_sky_mask_grad_scale) < 0.0:
        raise ValueError("--rgb_render_sky_mask_grad_scale must be non-negative.")
    if not 1 <= int(args.layout_max_actors) <= 96:
        raise ValueError("--layout_max_actors must be in [1,96].")
    if float(args.static_far_plane_m) != STATIC_FAR_PLANE_M:
        raise ValueError(
            f"--static_far_plane_m is frozen to {STATIC_FAR_PLANE_M:g}."
        )
    if not math.isfinite(float(args.layout_depth_tau)) or float(args.layout_depth_tau) <= 0.0:
        raise ValueError("--layout_depth_tau must be finite and positive.")
    if not 0.0 <= float(args.layout_to_gauge_grad_scale) <= 1.0:
        raise ValueError("--layout_to_gauge_grad_scale must be in [0,1].")
    for name in (
        "guidance_scale",
        "layout_guidance_scale",
        "asset_control_guidance_scale",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} must be finite and non-negative.")
    if args.val_image_dir is None:
        args.val_image_dir = args.image_dir
    if args.val_hdmap_root is None:
        if Path(args.val_image_dir).resolve() != Path(args.image_dir).resolve():
            raise ValueError(
                "--val_hdmap_root is required when validation uses a different image root"
            )
        args.val_hdmap_root = args.hdmap_root
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
    if args.val_every > 0 and args.val_batches > 0 and args.val_scene_end > args.val_scene_start:
        if args.val_scene_gauge_path is None:
            if Path(args.val_image_dir).resolve() == Path(args.image_dir).resolve():
                args.val_scene_gauge_path = args.scene_gauge_path
            else:
                raise ValueError(
                    "--val_scene_gauge_path is required when validation uses a different image root"
                )
    if int(args.val_inference_scenes) <= 0:
        raise ValueError("--val_inference_scenes must be positive")
    for name in ("text_uncond_drop_prob",):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}.")
    if tuple(float(value) for value in LAYOUT_TASK_PROBABILITIES) != (0.1, 0.5, 0.4):
        raise RuntimeError("layout task probabilities must remain TC/TCMG/TCMGA=.10/.50/.40")
    if int(args.resume_expected_step) >= 0 and not args.resume_path:
        raise ValueError("--resume_expected_step requires --resume_path.")

    device, local_rank, world_size = setup_distributed(args)
    if int(args.num_workers) > 0:
        torch.multiprocessing.set_sharing_strategy(str(args.mp_sharing_strategy))
    seed_everything(args.seed + get_rank())
    rgb_render_summary = rgb_render_run_summary(args)
    for key, value in rgb_render_summary.items():
        setattr(args, key, value)

    log_dir = Path(args.log_dir)
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
        config = dict(vars(args))
        config["patch_grid"] = list(args.patch_grid)
        (log_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(
            "[rgb-render] "
            f"hard_sigma_selection={rgb_render_summary['hard_sigma_selection']} "
            f"row_cap={rgb_render_summary['rgb_render_row_cap']} "
            "continuous_sigma_weighting="
            f"{rgb_render_summary['rgb_render_continuous_sigma_weighting']} "
            "lambda="
            f"{rgb_render_summary['lambda_rgb_render_effective']:g}",
            flush=True,
        )
    wandb_run = init_wandb(args, log_dir)
    if wandb_run is not None:
        wandb_run.summary.update(rgb_render_summary)

    vggt_model = load_dggt_aggregator_and_tokenizer(
        args.dggt_ckpt_path,
        args.tokenizer_ckpt_path,
        device,
    )
    lpips_model = setup_lpips_for_rgb_loss(args, device)

    # The frozen SceneFlow input contract keeps the packed channel declaration;
    # the DDT visual embedders consume z_t and the edit masks enter as controls.
    sf_in_channels = 3 * int(args.latent_dim) + 3
    scene_flow = WanSceneFlow.from_scene_config(
        bring_up=False,
        patch_grid=args.patch_grid,
        in_channels=sf_in_channels,
        out_channels=int(args.latent_dim),
        camera_condition_representation=CAMERA_CONDITION_REPRESENTATION,
        gauge_gen_dim=SCENE_GAUGE_DIM,
        scene_gauge_representation=SCENE_GAUGE_REPRESENTATION,
        scene_gauge_stats_version=SCENE_GAUGE_STATS_VERSION,
        layout_condition_version=LAYOUT_CONDITION_VERSION,
        layout_raster_channels=33,
        layout_raster_hw=(100, 148),
        layout_map_channels=(0, 22),
        layout_actor_channels=(22, 33),
        layout_map_groups=5,
        layout_stem_dim=96,
        layout_max_actors=int(args.layout_max_actors),
        layout_depth_tau=float(args.layout_depth_tau),
        layout_map_injection=bool(args.layout_map_injection),
        layout_actor_injection=bool(args.layout_actor_injection),
        layout_map_metric_injection=bool(args.layout_map_metric_injection),
        layout_actor_metric_injection=bool(args.layout_actor_metric_injection),
        appearance_context_injection=bool(args.appearance_context_injection),
        layout_to_gauge_grad_scale=float(args.layout_to_gauge_grad_scale),
        raster_schema_hash=RASTER_SCHEMA_HASH,
        static_far_plane_m=float(args.static_far_plane_m),
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
    if bool(args.three_quarter_gradient_checkpointing):
        scene_flow.enable_three_quarter_gradient_checkpointing()
    elif bool(args.half_gradient_checkpointing):
        scene_flow.enable_half_gradient_checkpointing()
    elif bool(args.gradient_checkpointing):
        scene_flow.enable_gradient_checkpointing()
    else:
        scene_flow.disable_gradient_checkpointing()
    if is_main_process():
        print(
            "[memory] SceneFlow gradient checkpointing "
            f"mode={scene_flow.gradient_checkpointing_mode} "
            f"encoder_blocks={len(scene_flow.checkpointed_block_indices(len(scene_flow.blocks)))}/{len(scene_flow.blocks)} "
            f"ddt_blocks={len(scene_flow.checkpointed_block_indices(len(scene_flow.ddt_head), block_group='ddt'))}/{len(scene_flow.ddt_head)}",
            flush=True,
        )
    load_all_stats_into_buffers(
        scene_flow,
        args.feature_stats_path,
        token_dim=int(args.latent_dim),
    )
    pullback_calibration = load_pullback_calibration(
        args.pullback_calibration_path,
        expected_window_len=int(args.sequence_length),
        expected_patch_grid=args.patch_grid,
    )
    scene_flow._pullback_calibration = pullback_calibration
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
        hdmap_root=args.hdmap_root,
        layout_max_actors=int(args.layout_max_actors),
        static_far_plane_m=float(args.static_far_plane_m),
        pretrain_instance_cache_size=args.pretrain_instance_cache_size,
        trunk_frames=29,
        return_full_dggt_context=True,
        load_dynamic_masks=False,
        binary_mask_channels=1,
        image_output_dtype="uint8",
        scene_gauge_path=args.scene_gauge_path,
        expected_scene_gauge_split=Path(args.image_dir).name,
        load_metric_depth_diagnostic=False,
        return_metric_depth_diagnostic_paths=(
            int(args.metric_depth_diagnostic_every) > 0
        ),
    )
    sampler: ContinuousDistributedBatchSampler | None = None
    if world_size > 1:
        sampler = ContinuousDistributedBatchSampler(
            dataset,
            batch_size=int(args.batch_size),
            grad_accum_steps=int(args.grad_accum_steps),
            num_replicas=world_size,
            rank=get_rank(),
            seed=0,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            **dataloader_runtime_kwargs(args),
        )
        if is_main_process():
            print(
                "[dataloader] continuous distributed prefetch enabled: "
                f"dataset_samples={len(dataset)} "
                f"rank_samples={sampler.distributed_sampler.num_samples} "
                f"batches_per_logical_epoch={sampler.batches_per_logical_epoch}; "
                "worker queues will not drain at logical epoch boundaries",
                flush=True,
            )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            drop_last=True,
            **dataloader_runtime_kwargs(args),
        )
    val_loader = None
    val_long_loader = None
    if args.val_every > 0 and args.val_batches > 0 and args.val_scene_end > args.val_scene_start:
        val_scene_names = discover_scene_names(args.val_image_dir, args.val_scene_start, args.val_scene_end)
        validation_stride = pretrain_validation_stride(
            int(args.sequence_length),
            int(getattr(args, "val_sliding_stride", 0) or 0),
        )
        validation_offsets = pretrain_validation_window_offsets(
            int(args.sequence_length),
            validation_stride,
        )
        val_dataset = WaymoOpenDataset(
            image_dir=args.val_image_dir,
            scene_names=val_scene_names,
            sequence_length=args.sequence_length,
            start_idx=0,
            mode=1,
            views=1,
            caption_root=args.val_caption_root,
            pretrain_patch_grid=args.patch_grid,
            hdmap_root=args.val_hdmap_root,
            layout_max_actors=int(args.layout_max_actors),
            static_far_plane_m=float(args.static_far_plane_m),
            pretrain_instance_cache_size=args.pretrain_instance_cache_size,
            trunk_major_samples=True,
            deterministic_layout_reference=True,
            trunk_frames=29,
            return_full_dggt_context=True,
            trunk_major_window_offsets=validation_offsets,
            load_dynamic_masks=False,
            binary_mask_channels=1,
            image_output_dtype="uint8",
            scene_gauge_path=args.val_scene_gauge_path,
            expected_scene_gauge_split=Path(args.val_image_dir).name,
            # Validation never schedules the training-only LiDAR diagnostic;
            # do not read and collate depth arrays that no validation consumer uses.
            load_metric_depth_diagnostic=False,
        )
        val_sampler = SpreadSequentialSampler(
            val_dataset, int(args.val_batches) * int(args.batch_size)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.val_num_workers,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            drop_last=False,
            **dataloader_runtime_kwargs_for_workers(
                args,
                int(args.val_num_workers),
            ),
        )
        val_long_dataset = WaymoOpenDataset(
            image_dir=args.val_image_dir,
            scene_names=val_scene_names,
            sequence_length=29,
            start_idx=0,
            mode=1,
            views=1,
            caption_root=args.val_caption_root,
            pretrain_patch_grid=args.patch_grid,
            hdmap_root=args.val_hdmap_root,
            layout_max_actors=int(args.layout_max_actors),
            static_far_plane_m=float(args.static_far_plane_m),
            pretrain_instance_cache_size=args.pretrain_instance_cache_size,
            trunk_major_samples=True,
            deterministic_layout_reference=True,
            trunk_frames=29,
            return_full_dggt_context=True,
            load_dynamic_masks=False,
            binary_mask_channels=1,
            image_output_dtype="uint8",
            scene_gauge_path=args.val_scene_gauge_path,
            expected_scene_gauge_split=Path(args.val_image_dir).name,
            load_metric_depth_diagnostic=False,
        )
        val_long_sampler = CyclicSequentialSampler(val_long_dataset)
        val_long_loader = DataLoader(
            val_long_dataset,
            batch_size=1,
            sampler=val_long_sampler,
            num_workers=0,
            pin_memory=bool(args.pin_memory) and device.type == "cuda",
            drop_last=False,
        )
        validation_scales = validation_guidance_scales(args)
        validation_scene_count = effective_validation_scene_count(
            len(validation_scales),
            int(args.val_inference_scenes),
            world_size=world_size,
        )
        available_long_scenes = validation_available_scene_count(
            val_long_dataset.trunk_major_index
        )
        if available_long_scenes < validation_scene_count:
            raise ValueError(
                "long-form validation contains fewer distinct usable complete-trunk "
                "scenes than the requested concurrent inference count: "
                f"{available_long_scenes} "
                f"< {validation_scene_count}"
            )
        if is_main_process():
            print(
                f"[validation] scenes={len(val_scene_names)} batches_per_eval={args.val_batches} "
                f"usable_long_scenes={available_long_scenes} "
                f"local_offsets={validation_offsets} long_rollout_frames=29 "
                f"workers_per_active_rank={args.val_num_workers} "
                f"inference_scenes={validation_scene_count}/{args.val_inference_scenes} "
                f"pinned={validation_pinned_scene_count(validation_scene_count)}"
                f"+rotating={validation_scene_count - validation_pinned_scene_count(validation_scene_count)} "
                f"scalar_cover={val_sampler.indices} "
                f"cfg_scales={validation_scales} "
                f"active_ranks={validation_scene_count * len(validation_scales)}",
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
        expected_step=(
            int(args.resume_expected_step)
            if int(args.resume_expected_step) >= 0
            else None
        ),
        args=args,
    )
    # DDP broadcasts rank-0 parameters in its constructor; rebuild a fresh EMA
    # after that broadcast, while strict state resume preserves the loaded EMA.
    sync_ema_after_ddp_initial_broadcast = not bool(args.resume_path)
    if world_size > 1:
        scene_flow = DistributedDataParallel(
            scene_flow,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
    if sync_ema_after_ddp_initial_broadcast:
        sync_ema_shadow_from_model(scene_flow, ema)

    scene_flow.train()
    optimizer.zero_grad(set_to_none=True)
    accum_step = 0
    step_wait_seconds = 0.0
    # Rolling sums for wandb so we report the mean over the last
    # `--wandb_log_every` optimizer steps instead of every individual step.
    wandb_sums: dict[str, TrainLogSeries] = {}
    wandb_observation_counts: dict[str, int] = {}
    wandb_count = 0
    interactive_tqdm = use_interactive_tqdm(
        args.no_tqdm,
        force_tqdm=bool(args.force_tqdm),
    )
    plain_text_logging = not interactive_tqdm
    progress = None
    if is_main_process() and interactive_tqdm:
        force_web_tqdm = bool(args.force_tqdm)
        if force_web_tqdm:
            print(
                "[logging] tqdm Web-console newline mode enabled; "
                "ETA/rate will be emitted once per optimizer step",
                flush=True,
            )
        progress = tqdm(
            total=args.max_steps,
            initial=global_step,
            desc="pretrain",
            dynamic_ncols=not force_web_tqdm,
            mininterval=0.0,
            miniters=1,
            file=tqdm_output_stream(force_tqdm=force_web_tqdm),
        )
    elif is_main_process():
        reason = "--no_tqdm" if args.no_tqdm else "non-interactive stderr"
        print(
            f"[logging] {reason}; newline training metrics every "
            f"{max(1, int(args.log_every))} optimizer steps",
            flush=True,
        )
    try:
        if sampler is not None:
            sampler.set_start_optimizer_step(global_step)
        while global_step < args.max_steps:
            batch_wait_started = time.perf_counter()
            for batch in loader:
                # One optimizer step pays the input wait of every micro-batch it
                # accumulates, so sum them: reporting only the last one hides a
                # ``--grad_accum_steps`` fold of the real starvation.
                step_wait_seconds += time.perf_counter() - batch_wait_started
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
                        if (
                            sync_grad
                            and is_main_process()
                            and should_apply_metric_depth_diagnostic(
                                args,
                                global_step,
                                training=True,
                            )
                        ):
                            batch = hydrate_metric_depth_diagnostic_batch(
                                batch,
                                max_samples=int(
                                    args.metric_depth_diagnostic_max_samples
                                ),
                            )
                        collect_step_logs = bool(
                            sync_grad
                            and (
                                not bool(args.no_rank_metric_mean)
                                or is_main_process()
                            )
                        )
                        next_optimizer_step = global_step + (1 if sync_grad else 0)
                        collect_expensive_diagnostics = bool(
                            collect_step_logs
                            and (
                                (
                                    bool(args.wandb)
                                    and wandb_count + 1
                                    >= max(1, int(args.wandb_log_every))
                                )
                                or (
                                    plain_text_logging
                                    and next_optimizer_step
                                    % max(1, int(args.log_every))
                                    == 0
                                )
                            )
                        )
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
                            collect_expensive_diagnostics=(
                                collect_expensive_diagnostics
                            ),
                            collect_logs=collect_step_logs,
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
                        step_wait_seconds = 0.0
                        batch_wait_started = time.perf_counter()
                        continue
                    logs = dict(logs)
                    (loss / max(1, args.grad_accum_steps)).backward()
                accum_step += 1

                if not sync_grad:
                    batch_wait_started = time.perf_counter()
                    continue

                logs["dataloader/wait_seconds"] = float(step_wait_seconds)
                step_wait_seconds = 0.0

                params = unwrap_ddp(scene_flow).parameters()
                grad_norm: torch.Tensor | None = None
                if args.grad_clip_norm > 0:
                    # ``clip_grad_norm_`` already computes the pre-clip norm, so
                    # reporting it is free.  It decides how to read every loss
                    # weight: once the total norm sits above --grad_clip_norm the
                    # whole gradient is rescaled by clip/||g||, so raising any
                    # auxiliary lambda no longer adds signal, it takes it away
                    # from the flow objective.  Without this series the tradeoff
                    # is invisible.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, args.grad_clip_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.step(unwrap_ddp(scene_flow).parameters())
                accum_step = 0
                global_step += 1

                # Every rank must reach this collective, so it stays outside the
                # rank-0 reporting branch, and it runs once the optimizer and
                # EMA work is already queued so the read-back does not stall
                # them.  The flag is a CLI value, so ranks never disagree about
                # whether the collective happens.
                if not args.no_rank_metric_mean:
                    logs = all_rank_log_mean(logs, device=device)

                if is_main_process():
                    lr_now = optimizer.param_groups[0]["lr"]
                    train_metrics = dict(logs)
                    train_metrics["lr"] = float(lr_now)
                    if grad_norm is not None:
                        # Added after the all-rank reduce for the same reason as
                        # ``lr``: it is a rank-0 optimizer property, not a shard
                        # measurement, and DDP has already averaged the grads.
                        train_metrics["grad_norm"] = float(grad_norm)
                        train_metrics["grad_clip_active"] = float(
                            float(grad_norm) > float(args.grad_clip_norm)
                        )
                    if progress is not None:
                        progress.set_postfix(
                            progress_postfix(train_metrics), refresh=False
                        )
                    elif global_step % max(1, int(args.log_every)) == 0:
                        # Newline-terminated, so this is the form that survives
                        # a log collector.  It keeps every key but leads with
                        # the wait and the per-loss summary.
                        head = progress_postfix(train_metrics)
                        rest = materialize_log_values({
                            key: value
                            for key, value in train_metrics.items()
                            if key not in _PROGRESS_POSTFIX_KEYS
                        })
                        metrics_str = " | ".join(
                            [f"{label}={text}" for label, text in head.items()]
                            + [f"{key}={value:.4f}" for key, value in rest.items()]
                        )
                        print(f"[step {global_step:06d}] {metrics_str}", flush=True)

                    if wandb_run is not None:
                        accumulate_wandb_metrics(
                            wandb_sums,
                            wandb_observation_counts,
                            train_metrics,
                        )

                if bool(args.wandb):
                    wandb_count += 1
                    if wandb_count >= max(1, int(args.wandb_log_every)):
                        if is_main_process() and wandb_run is not None:
                            averaged = finalize_wandb_metrics(
                                wandb_sums,
                                wandb_observation_counts,
                            )
                            log_wandb(wandb_run, averaged, global_step, "train")
                            wandb_sums = {}
                            wandb_observation_counts = {}
                        wandb_count = 0

                if (
                    val_loader is not None
                    and global_step > 0
                    and global_step % args.val_every == 0
                ):
                    # The scalar block reads a fixed spread cover (see
                    # SpreadSequentialSampler) so its series stay comparable
                    # across steps.  ``validation_index`` reaches only the
                    # long-form sampling loader, where it advances the rotating
                    # half of the scene slots and leaves the pinned half alone.
                    validation_index = global_step // args.val_every - 1
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
                        long_loader=val_long_loader,
                        validation_index=validation_index,
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
                batch_wait_started = time.perf_counter()
            if sampler is not None and global_step < args.max_steps:
                raise RuntimeError(
                    "continuous distributed training DataLoader ended unexpectedly"
                )
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
