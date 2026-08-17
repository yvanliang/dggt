from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset

import train_scene_flow_pretrain as pretrain
from datasets.dataset import WAYMO_OPENCV2DATASET, WaymoOpenDataset
from datasets.tools.hdmap_schema import (
    ATTRIBUTE_SOURCE_NONE,
    HDMapFeature,
    HDMapScene,
    write_scene,
)
from dggt.models.scene_flow import WanSceneFlow


PATCH_GRID = (25, 37)
PATCHES = PATCH_GRID[0] * PATCH_GRID[1]
ACTORS = 4
T58_DEFAULT_IMAGE_ROOT = Path(
    "/data/disk2/lyy_dataset/waymo_processed_dggt/validation"
)
T58_DEFAULT_HDMAP_ROOT = Path(
    "/data/disk2/lyy_dataset/waymo_processed_dggt/validation_hdmap"
)
T58_WORKERS = 8
T58_WARMUP_SAMPLES = 8
T58_MEASURE_SAMPLES = 24


def _write_matrix(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(value, dtype=np.float64).reshape(4, 4))


def _write_synthetic_waymo(tmp_path: Path) -> tuple[Path, Path]:
    processed_root = tmp_path / "processed"
    hdmap_root = tmp_path / "hdmap"
    scene_root = processed_root / "000"
    for name in (
        "images",
        "sky_masks",
        "custom_masks",
        "ego_pose",
        "extrinsics",
        "intrinsics",
        "instances",
    ):
        (scene_root / name).mkdir(parents=True, exist_ok=True)

    height, width = 100, 148
    for frame in range(3):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = 50 + frame * 20
        rgb[..., 1] = np.arange(width, dtype=np.uint8)[None]
        Image.fromarray(rgb, mode="RGB").save(
            scene_root / "images" / f"{frame:03d}_0.png"
        )
        Image.fromarray(
            np.zeros((height, width), dtype=np.uint8), mode="L"
        ).save(scene_root / "sky_masks" / f"{frame:03d}_0.png")
        semantic = np.zeros((height, width), dtype=np.uint8)
        if frame == 2:
            semantic[:] = 40
        Image.fromarray(semantic, mode="L").save(
            scene_root / "custom_masks" / f"{frame:03d}_0.png"
        )
        _write_matrix(
            scene_root / "ego_pose" / f"{frame:03d}.txt",
            np.eye(4, dtype=np.float64),
        )

    _write_matrix(
        scene_root / "extrinsics" / "0.txt",
        np.linalg.inv(WAYMO_OPENCV2DATASET),
    )
    np.savetxt(
        scene_root / "intrinsics" / "0.txt",
        np.asarray((100.0, 100.0, 74.0, 50.0), dtype=np.float64),
    )

    instances: dict[str, object] = {}
    frame_instances = {str(frame): [] for frame in range(3)}
    for actor, x_position in enumerate((-2.0, 0.0, 2.0)):
        poses = []
        for frame in range(3):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = (x_position, 0.0, 12.0 + 0.2 * frame)
            poses.append(pose.tolist())
            frame_instances[str(frame)].append(actor)
        instances[str(actor)] = {
            "id": str(actor),
            "raw_object_id": f"vehicle-{actor}",
            "class_name": "Vehicle",
            "is_moving_track": True,
            "frame_annotations": {
                "frame_idx": [0, 1, 2],
                "obj_to_world": poses,
                "box_size": [[4.0, 2.0, 2.0]] * 3,
            },
        }
    (scene_root / "instances" / "instances_info.json").write_text(
        json.dumps(instances), encoding="utf-8"
    )
    (scene_root / "instances" / "frame_instances.json").write_text(
        json.dumps(frame_instances), encoding="utf-8"
    )
    (scene_root / "instances" / "object_id_map.json").write_text(
        "{}", encoding="utf-8"
    )

    write_scene(
        hdmap_root,
        HDMapScene(
            segment="synthetic-segment",
            scene_id="000",
            split="training",
            features=[
                HDMapFeature(
                    cls="road_line",
                    vertices=np.asarray(
                        (
                            (-4.0, -1.5, 8.0),
                            (0.0, -1.5, 16.0),
                            (4.0, -1.5, 24.0),
                        ),
                        dtype=np.float64,
                    ),
                )
            ],
            attribute_source=ATTRIBUTE_SOURCE_NONE,
            map_pose_offset=np.zeros((3, 3), dtype=np.float64),
        ),
    )
    return processed_root, hdmap_root


class _CanonicalAggregator(nn.Module):
    def forward(self, images: torch.Tensor):
        rows, frames = int(images.shape[0]), int(images.shape[1])
        value = images.float().mean(dim=(2, 3, 4), keepdim=True)
        tokens = value.reshape(rows, frames, 1, 1).expand(
            rows, frames, PATCHES + 5, 1
        )
        pyramid = [tokens] * 24
        return None, pyramid, None, None, 5


class _CanonicalTokenizer(nn.Module):
    def encode(
        self,
        levels: list[torch.Tensor],
        *,
        patch_grid: tuple[int, int],
    ) -> torch.Tensor:
        assert patch_grid == PATCH_GRID
        scalar = torch.stack(levels, dim=0).mean(dim=0)
        return scalar.expand(*scalar.shape[:-1], 1024).contiguous()


class _CanonicalVGGT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = _CanonicalAggregator()
        self.scene_tokenizer = _CanonicalTokenizer()


def _tiny_scene_flow() -> WanSceneFlow:
    model = WanSceneFlow(
        patch_grid=PATCH_GRID,
        num_attention_heads=1,
        attention_head_dim=24,
        out_channels=1024,
        qwen_dim=8,
        freq_dim=8,
        ffn_dim=48,
        num_layers=1,
        base_model_depth=1,
        repa_layer_depth=1,
        ddt_head_depth=1,
        ddt_head_dim=24,
        ddt_head_heads=1,
        ddt_head_ffn_dim=48,
        num_timestep_tokens=1,
        sky_mask_refine_channels=8,
        layout_condition_version="layout_v2",
        layout_max_actors=ACTORS,
    )
    model.set_latent_stats(torch.zeros(1024), torch.ones(1024))
    model.set_gauge_stats(torch.zeros(3), torch.ones(3))
    return model


def _layout_terminal_modules(model: WanSceneFlow) -> dict[str, nn.Module]:
    assert model.layout_map_stem is not None
    assert model.layout_actor_stem is not None
    assert model.map_metric_adapter is not None
    assert model.full_actor_gauge_adapter is not None
    assert model.appearance_context_adapter is not None
    return {
        "map_early": model.layout_map_stem.zero_out,
        "actor_early": model.layout_actor_stem.zero_out,
        "map_metric": model.map_metric_adapter.net[-1],
        "actor_metric": model.full_actor_gauge_adapter.net[-1],
        "appearance": model.appearance_context_adapter.net[-1],
    }


def _open_downstream_gradient_paths(model: WanSceneFlow) -> dict[str, nn.Module]:
    terminals = _layout_terminal_modules(model)
    terminal_ids = {id(module) for module in terminals.values()}
    with torch.no_grad():
        for module in model.modules():
            if id(module) in terminal_ids or not isinstance(
                module, (nn.Linear, nn.Conv2d)
            ):
                continue
            if module.weight.numel() and not bool(module.weight.count_nonzero()):
                module.weight.normal_(mean=0.0, std=0.02)
    for module in terminals.values():
        assert not bool(module.weight.count_nonzero())
        if module.bias is not None:
            assert not bool(module.bias.count_nonzero())
    return terminals


def _dataset_batch(tmp_path: Path) -> dict[str, object]:
    processed_root, hdmap_root = _write_synthetic_waymo(tmp_path)
    dataset = WaymoOpenDataset(
        image_dir=str(processed_root),
        scene_names=["000"],
        sequence_length=1,
        start_idx=0,
        mode=1,
        views=1,
        caption_root=None,
        pretrain_patch_grid=PATCH_GRID,
        hdmap_root=str(hdmap_root),
        layout_max_actors=ACTORS,
        load_dynamic_masks=False,
        image_output_dtype="uint8",
    )
    # T54 must exercise the appearance branch.  Production legitimately
    # samples Ka=0, so pin the local Python RNG instead of making this test
    # depend on whatever random state earlier tests happened to leave behind.
    state = random.getstate()
    try:
        random.seed(0)
        return next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    finally:
        random.setstate(state)


def test_t54_sidecar_dataset_assembler_forward_backward(tmp_path: Path) -> None:
    batch = _dataset_batch(tmp_path)
    assert bool(batch["map_metric"][..., 3].any())
    assert bool(batch["projected_actor_geometry_valid"].any())
    assert bool(batch["appearance_binding_valid"].any())

    model = _tiny_scene_flow().train()
    terminals = _open_downstream_gradient_paths(model)
    built = pretrain.build_layout_condition_from_batch(
        batch,
        _CanonicalVGGT(),
        model,
        torch.device("cpu"),
        layout_max_actors=ACTORS,
        patch_grid=PATCH_GRID,
    )
    z_t = torch.randn((1, 1, PATCHES, 1024))
    zeros = torch.zeros_like(z_t)
    masks = torch.zeros((1, 1, PATCHES, 1))
    output = model(
        z_t,
        torch.tensor([0.5]),
        zeros,
        zeros,
        masks,
        masks,
        masks,
        **pretrain.layout_model_kwargs(
            built.layout,
            built.appearance_class_id,
            gauge_grad_scale=1.0,
        ),
        camera_condition_tokens=torch.zeros((1, 1, 20)),
        camera_attention_mask=torch.ones((1, 1), dtype=torch.bool),
        gauge_gen_tokens=torch.zeros((1, 1, 3)),
        gauge_gen_attention_mask=torch.ones((1, 1), dtype=torch.bool),
        return_dict=True,
    )
    output["video"].float().sum().backward()

    for name, module in terminals.items():
        assert module.weight.grad is not None, name
        assert bool(torch.isfinite(module.weight.grad).all()), name
        assert bool(module.weight.grad.count_nonzero()), name
        assert not bool(module.weight.count_nonzero()), name


def test_t57_model_output_and_training_loss_are_camera_clean() -> None:
    model = _tiny_scene_flow().eval()
    forbidden_state_fragments = (
        "camera_" + "gen",
        "asset_" + "placement",
        "asset_" + "slot_embed",
    )
    assert all(
        all(fragment not in key for fragment in forbidden_state_fragments)
        for key in model.state_dict()
    )

    source = inspect.getsource(pretrain.train_step)
    forbidden_loss_fragments = (
        "camera_" + "flow",
        "camera_" + "pose_loss",
        "loss_" + "camera",
        "lambda_" + "camera_pose",
    )
    assert all(fragment not in source for fragment in forbidden_loss_fragments)

    condition, appearance_class = _minimal_null_layout()
    z_t = torch.randn((1, 1, PATCHES, 1024))
    zeros = torch.zeros_like(z_t)
    masks = torch.zeros((1, 1, PATCHES, 1))
    output = model(
        z_t,
        torch.tensor([0.5]),
        zeros,
        zeros,
        masks,
        masks,
        masks,
        layout_raster=condition.raster,
        map_metric=condition.map_metric,
        actor_geometry=condition.actor_geometry,
        projected_actor_geometry=condition.projected_actor_geometry,
        appearance=condition.appearance,
        map_mode=condition.map_mode,
        raster_schema_hash=condition.raster_schema_hash,
        appearance_class_id=appearance_class,
        camera_condition_tokens=torch.zeros((1, 1, 20)),
        camera_attention_mask=torch.ones((1, 1), dtype=torch.bool),
        gauge_gen_tokens=torch.zeros((1, 1, 3)),
        gauge_gen_attention_mask=torch.ones((1, 1), dtype=torch.bool),
        return_dict=True,
    )
    assert set(output) == {"video", "sky", "gauge"}
    assert "camera" not in output


def _t58_dataset(
    *,
    image_root: Path,
    hdmap_root: Path | None,
    scene_names: list[str],
) -> WaymoOpenDataset:
    return WaymoOpenDataset(
        image_dir=str(image_root),
        scene_names=scene_names,
        sequence_length=10,
        start_idx=0,
        mode=1,
        views=1,
        caption_root=None,
        pretrain_patch_grid=PATCH_GRID,
        hdmap_root=None if hdmap_root is None else str(hdmap_root),
        layout_max_actors=96,
        trunk_major_samples=True,
        trunk_frames=29,
        return_full_dggt_context=True,
        load_dynamic_masks=False,
        binary_mask_channels=1,
        image_output_dtype="uint8",
        scene_gauge_path=None,
    )


def _t58_measure_rows(
    dataset: WaymoOpenDataset,
    *,
    layout_enabled: bool,
) -> tuple[float, float]:
    total = T58_WARMUP_SAMPLES + T58_MEASURE_SAMPLES
    if len(dataset) < total:
        pytest.skip(
            f"T58 needs {total} real trunk rows, but the selected split has {len(dataset)}"
        )
    loader = DataLoader(
        Subset(dataset, range(total)),
        batch_size=1,
        shuffle=False,
        num_workers=T58_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    iterator = iter(loader)
    for _ in range(T58_WARMUP_SAMPLES):
        batch = next(iterator)
        assert ("layout_raster" in batch) is layout_enabled
    started = time.perf_counter()
    for _ in range(T58_MEASURE_SAMPLES):
        batch = next(iterator)
        assert ("layout_raster" in batch) is layout_enabled
    wall_seconds = time.perf_counter() - started
    del iterator, loader
    return wall_seconds, wall_seconds / float(T58_MEASURE_SAMPLES)


def test_t58_real_hdmap_eight_worker_extra_throughput() -> None:
    """Measure the frozen 8-worker T58 protocol against a no-layout baseline.

    v2.1 defines the protocol but no numeric limit, so this test always records
    the real result and only enforces a performance budget when the runner sets
    ``DGGT_T58_MAX_EXTRA_SEC_PER_SAMPLE`` explicitly.
    """

    image_root = Path(
        os.environ.get("DGGT_T58_IMAGE_ROOT", str(T58_DEFAULT_IMAGE_ROOT))
    )
    hdmap_root = Path(
        os.environ.get("DGGT_T58_HDMAP_ROOT", str(T58_DEFAULT_HDMAP_ROOT))
    )
    if not image_root.is_dir() or not hdmap_root.is_dir():
        pytest.skip(
            "T58 real Waymo/HD-map roots are unavailable: "
            f"image_root={image_root}, hdmap_root={hdmap_root}"
        )

    scene_names = [f"{index:03d}" for index in range(16)]
    missing = [
        scene
        for scene in scene_names
        if not (image_root / scene / "images").is_dir()
        or not (hdmap_root / scene / "hdmap.npz").is_file()
    ]
    if missing:
        pytest.skip(f"T58 real scene range 000..015 is incomplete: {missing}")

    baseline_wall, baseline_sec = _t58_measure_rows(
        _t58_dataset(
            image_root=image_root,
            hdmap_root=None,
            scene_names=scene_names,
        ),
        layout_enabled=False,
    )
    layout_wall, layout_sec = _t58_measure_rows(
        _t58_dataset(
            image_root=image_root,
            hdmap_root=hdmap_root,
            scene_names=scene_names,
        ),
        layout_enabled=True,
    )
    extra_sec = layout_sec - baseline_sec
    assert all(
        math.isfinite(value)
        for value in (baseline_wall, baseline_sec, layout_wall, layout_sec, extra_sec)
    )
    assert baseline_sec > 0.0 and layout_sec > 0.0
    assert extra_sec >= 0.0

    budget_raw = os.environ.get("DGGT_T58_MAX_EXTRA_SEC_PER_SAMPLE")
    budget = None if budget_raw is None else float(budget_raw)
    if budget is not None:
        assert budget > 0.0
        assert extra_sec <= budget, (
            f"T58 extra {extra_sec:.6f} sec/sample exceeds explicit budget "
            f"{budget:.6f}"
        )
    print(
        "T58_RESULT "
        + json.dumps(
            {
                "baseline_sec_per_sample": baseline_sec,
                "baseline_wall_seconds": baseline_wall,
                "budget_sec_per_sample": budget,
                "extra_sec_per_sample": extra_sec,
                "hdmap_root": str(hdmap_root),
                "image_root": str(image_root),
                "layout_sec_per_sample": layout_sec,
                "layout_wall_seconds": layout_wall,
                "measured_samples": T58_MEASURE_SAMPLES,
                "scene_range": "000..015",
                "warmup_samples": T58_WARMUP_SAMPLES,
                "workers": T58_WORKERS,
            },
            sort_keys=True,
        )
    )


def _minimal_null_layout():
    # Reuse the production NULL constructors through one synthetic assembled
    # batch, while keeping T57 independent of filesystem I/O.
    from datasets.tools.hdmap_schema import RASTER_SCHEMA_HASH
    from dggt.utils.actor_geometry_condition import (
        MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
        ActorGeometryCondition,
        LayoutMode,
        ProjectedActorGeometry,
    )
    from dggt.utils.appearance_binding_condition import (
        AppearanceBindingCondition,
        AppearanceMode,
    )
    from dggt.utils.layout_condition import LayoutConditionBatch, MapMode

    geometry = ActorGeometryCondition(
        slot_track_id=torch.full((1, ACTORS), -1, dtype=torch.int64),
        class_id=torch.full((1, ACTORS), -1, dtype=torch.int8),
        corners_world=torch.zeros((1, ACTORS, 1, 8, 3), dtype=torch.float64),
        velocity_world=torch.zeros((1, ACTORS, 1, 3)),
        box_size=torch.zeros((1, ACTORS, 1, 3)),
        yaw=torch.zeros((1, ACTORS, 1)),
        is_moving=torch.zeros((1, ACTORS, 1), dtype=torch.bool),
        track_valid=torch.zeros((1, ACTORS, 1), dtype=torch.bool),
        slot_valid=torch.zeros((1, ACTORS), dtype=torch.bool),
        layout_mode=torch.full((1,), int(LayoutMode.NULL), dtype=torch.int8),
        raw_track_key=[[""] * ACTORS],
    )
    projected = ProjectedActorGeometry(
        bbox_patch=torch.zeros((1, ACTORS, 1, 4)),
        patch_weight=torch.zeros((1, ACTORS, 1, PATCHES)),
        log_z_patch=torch.zeros((1, ACTORS, 1, PATCHES)),
        silhouette_uv=torch.zeros(
            (1, ACTORS, 1, MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES, 2)
        ),
        silhouette_vertex_valid=torch.zeros(
            (1, ACTORS, 1, MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES),
            dtype=torch.bool,
        ),
        corners_camera=torch.zeros((1, ACTORS, 1, 8, 3)),
        uv_corners=torch.zeros((1, ACTORS, 1, 8, 2)),
        velocity_camera=torch.zeros((1, ACTORS, 1, 3)),
        uv_center=torch.zeros((1, ACTORS, 1, 2)),
        log_z_w=torch.zeros((1, ACTORS, 1)),
        center_depth_valid=torch.zeros((1, ACTORS, 1), dtype=torch.bool),
        frame_support=torch.zeros((1, ACTORS, 1), dtype=torch.bool),
        metric_support=torch.zeros((1, ACTORS, 1), dtype=torch.bool),
        in_frustum=torch.zeros((1, ACTORS, 1), dtype=torch.bool),
        valid=torch.zeros((1, ACTORS, 1), dtype=torch.bool),
    )
    appearance = AppearanceBindingCondition(
        appearance_tokens=torch.zeros((1, 5, 32, 1024)),
        appearance_mask=torch.zeros((1, 5, 32), dtype=torch.bool),
        canonical_uv=torch.zeros((1, 5, 32, 2)),
        geometry_idx=torch.full((1, 5), -1, dtype=torch.int64),
        binding_valid=torch.zeros((1, 5), dtype=torch.bool),
        appearance_mode=torch.full(
            (1,), int(AppearanceMode.NULL), dtype=torch.int8
        ),
    )
    raster = torch.zeros((1, 1, 33, 100, 148), dtype=torch.uint8)
    raster[:, :, (10, 11, 28, 29)] = 127
    condition = LayoutConditionBatch(
        raster=raster,
        map_metric=torch.zeros((1, 1, PATCHES, 5, 4)),
        actor_geometry=geometry,
        projected_actor_geometry=projected,
        appearance=appearance,
        map_mode=torch.full((1,), int(MapMode.NULL), dtype=torch.int8),
        raster_schema_hash=RASTER_SCHEMA_HASH,
    )
    return condition, torch.full((1, 5), -1, dtype=torch.int8)
