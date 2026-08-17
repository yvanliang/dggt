from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import inference_scene_flow_pretrain as inference
import train_scene_flow_pretrain as pretrain
from datasets.tools.hdmap_schema import RASTER_SCHEMA_HASH
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.actor_geometry_condition import (
    ActorGeometryCondition,
    LayoutMode,
    MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
    ProjectedActorGeometry,
)
from dggt.utils.appearance_binding_condition import (
    AppearanceBindingCondition,
    AppearanceMode,
)
from dggt.utils.layout_condition import LayoutConditionBatch, MapMode
from dggt.utils.flow_schedule import build_flow_schedule_config
from dggt.utils.scene_gauge import (
    assemble_dggt_pose_encoding,
    metric_c2w_to_teacher_anchor_dggt,
)


def _layout(*, frames: int = 4, appearance: bool = True) -> LayoutConditionBatch:
    batch, slots, bindings, patches = 1, 2, 1, 25 * 37
    track_valid = torch.tensor(
        [[[True] * frames, [False] * frames]], dtype=torch.bool
    )
    corners = torch.zeros(
        batch, slots, frames, 8, 3, dtype=torch.float64
    )
    corners[:, 0, :, :, 2] = 10.0
    box_size = torch.zeros(batch, slots, frames, 3, dtype=torch.float32)
    box_size[:, 0] = torch.tensor([4.0, 2.0, 1.5])
    geometry = ActorGeometryCondition(
        slot_track_id=torch.tensor([[17, -1]], dtype=torch.int64),
        class_id=torch.tensor([[0, -1]], dtype=torch.int8),
        corners_world=corners,
        velocity_world=torch.zeros(
            batch, slots, frames, 3, dtype=torch.float32
        ),
        box_size=box_size,
        yaw=torch.zeros(batch, slots, frames, dtype=torch.float32),
        is_moving=torch.zeros(batch, slots, frames, dtype=torch.bool),
        track_valid=track_valid,
        slot_valid=torch.tensor([[True, False]], dtype=torch.bool),
        layout_mode=torch.tensor([int(LayoutMode.FULL)], dtype=torch.int8),
        raw_track_key=[["vehicle-17", ""]],
    )
    patch_weight = torch.zeros(
        batch, slots, frames, patches, dtype=torch.float32
    )
    patch_weight[:, 0, :, patches // 2] = 1.0
    log_z_patch = torch.zeros_like(patch_weight)
    log_z_patch[:, 0, :, patches // 2] = math.log(10.0)
    silhouette_uv = torch.zeros(
        batch,
        slots,
        frames,
        MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
        2,
        dtype=torch.float32,
    )
    silhouette_uv[:, 0, :, :4] = torch.tensor(
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
    )
    silhouette_vertex_valid = torch.zeros(
        batch,
        slots,
        frames,
        MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
        dtype=torch.bool,
    )
    silhouette_vertex_valid[:, 0, :, :4] = True
    projected = ProjectedActorGeometry(
        bbox_patch=torch.zeros(batch, slots, frames, 4, dtype=torch.float32),
        patch_weight=patch_weight,
        log_z_patch=log_z_patch,
        silhouette_uv=silhouette_uv,
        silhouette_vertex_valid=silhouette_vertex_valid,
        corners_camera=corners.float(),
        uv_corners=torch.zeros(
            batch, slots, frames, 8, 2, dtype=torch.float32
        ),
        velocity_camera=torch.zeros(
            batch, slots, frames, 3, dtype=torch.float32
        ),
        uv_center=torch.zeros(
            batch, slots, frames, 2, dtype=torch.float32
        ),
        log_z_w=torch.where(
            track_valid,
            torch.full((batch, slots, frames), math.log(10.0)),
            torch.zeros((batch, slots, frames)),
        ).float(),
        center_depth_valid=track_valid.clone(),
        frame_support=track_valid.clone(),
        metric_support=track_valid.clone(),
        in_frustum=track_valid.clone(),
        valid=track_valid.clone(),
    )
    binding_valid = torch.tensor([[appearance]], dtype=torch.bool)
    appearance_condition = AppearanceBindingCondition(
        appearance_tokens=torch.ones(
            batch, bindings, 1, 1024, dtype=torch.float32
        )
        if appearance
        else torch.zeros(batch, bindings, 1, 1024, dtype=torch.float32),
        appearance_mask=binding_valid[..., None].clone(),
        canonical_uv=torch.zeros(
            batch, bindings, 1, 2, dtype=torch.float32
        ),
        geometry_idx=torch.tensor(
            [[0 if appearance else -1]], dtype=torch.int64
        ),
        binding_valid=binding_valid,
        appearance_mode=torch.tensor(
            [
                int(
                    AppearanceMode.REAL
                    if appearance
                    else AppearanceMode.NULL
                )
            ],
            dtype=torch.int8,
        ),
    )
    raster = torch.zeros(batch, frames, 33, 100, 148, dtype=torch.uint8)
    raster[:, :, 21, 0, 0] = 255
    return LayoutConditionBatch(
        raster=raster,
        map_metric=torch.zeros(
            batch, frames, patches, 5, 4, dtype=torch.float32
        ),
        actor_geometry=geometry,
        projected_actor_geometry=projected,
        appearance=appearance_condition,
        map_mode=torch.tensor([int(MapMode.PRESENT)], dtype=torch.int8),
        raster_schema_hash=RASTER_SCHEMA_HASH,
    )


def _bundle(*, appearance: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        layout_condition=_layout(appearance=appearance),
        appearance_class_id=torch.tensor(
            [[0 if appearance else -1]], dtype=torch.int8
        ),
    )


def test_p9_cli_and_condition_modes_are_clean_cut() -> None:
    parser = inference.build_argparser()
    args = parser.parse_args(["--weights", "model.pt"])
    assert inference.CONDITION_MODES == (
        "tc",
        "tcmg",
        "tcmga",
    )
    assert args.condition_mode == "tcmga"
    assert args.layout_mode == "full"
    assert args.layout_max_actors is None
    assert args.static_far_plane_m == 120.0
    assert args.val_sample_steps == 50
    destinations = {action.dest for action in parser._actions}
    assert {
        "hdmap_root",
        "layout_mode",
        "condition_mode",
        "layout_guidance_scale",
        "layout_max_actors",
        "static_far_plane_m",
    } <= destinations
    assert "camera_" + "guidance_scale" not in destinations
    assert "camera_" + "text_guidance_scale" not in destinations


def test_t55_requested_camera_is_the_render_pose_source() -> None:
    requested = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    requested[:, :, 0, 3] = torch.tensor([0.0, 2.0, 5.0])
    anchor = torch.eye(4).reshape(1, 4, 4)
    gauge = torch.tensor([[[3.0, -0.7, -0.8]]], dtype=torch.float32)
    actual = inference.requested_render_pose_encoding(requested, anchor, gauge)
    expected = assemble_dggt_pose_encoding(
        metric_c2w_to_teacher_anchor_dggt(
            requested,
            anchor,
            gauge[..., 0],
        ),
        gauge,
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    edited = requested.clone()
    edited[:, :, 1, 3] += 4.0
    changed = inference.requested_render_pose_encoding(edited, anchor, gauge)
    assert not torch.equal(changed, actual)


def test_condition_modes_match_tc_tcmg_tcmga_and_a_absence_skips_delta() -> None:
    cam = _bundle()
    rows = inference.apply_condition_mode(cam, "tc", layout_mode="full")
    assert rows == {"camera": True, "layout": False, "appearance": False}
    assert inference.inference_cfg_branches(
        cam,
        text_scale=1.0,
        layout_scale=8.0,
        appearance_scale=9.0,
    ) == ("full",)

    layout_cam = _bundle()
    rows = inference.apply_condition_mode(
        layout_cam, "tcmg", layout_mode="full"
    )
    assert rows == {"camera": True, "layout": True, "appearance": False}
    assert inference.inference_cfg_branches(
        layout_cam,
        text_scale=1.0,
        layout_scale=2.0,
        appearance_scale=9.0,
    ) == ("full", "appearance_dropped", "layout_dropped")

    full = _bundle()
    rows = inference.apply_condition_mode(
        full, "tcmga", layout_mode="full"
    )
    assert rows == {"camera": True, "layout": True, "appearance": True}
    assert inference.inference_cfg_branches(
        full,
        text_scale=2.0,
        layout_scale=2.0,
        appearance_scale=2.0,
    ) == (
        "full",
        "no_text_full",
        "appearance_dropped",
        "layout_dropped",
    )

    absent = _bundle(appearance=False)
    inference.apply_condition_mode(
        absent, "tcmga", layout_mode="full"
    )
    assert inference.inference_cfg_branches(
        absent,
        text_scale=1.0,
        layout_scale=1.0,
        appearance_scale=7.0,
    ) == ("full",)


def test_layout_mode_full_preserves_real_partial_and_cli_rejects_fake_partial() -> None:
    partial = _bundle()
    partial.layout_condition = replace(
        partial.layout_condition,
        actor_geometry=replace(
            partial.layout_condition.actor_geometry,
            layout_mode=torch.tensor([int(LayoutMode.PARTIAL)], dtype=torch.int8),
        ),
    )
    inference.apply_condition_mode(
        partial, "tcmga", layout_mode="full"
    )
    assert int(
        partial.layout_condition.actor_geometry.layout_mode.item()
    ) == int(LayoutMode.PARTIAL)
    assert bool(partial.layout_condition.appearance.binding_valid.any())

    with pytest.raises(SystemExit):
        inference.build_argparser().parse_args(
            ["--weights", "model.pt", "--layout_mode", "partial"]
        )

    empty = _bundle()
    inference.apply_condition_mode(
        empty, "tcmga", layout_mode="empty"
    )
    assert int(empty.layout_condition.actor_geometry.layout_mode.item()) == int(
        LayoutMode.EMPTY
    )
    assert int(empty.layout_condition.map_mode.item()) == int(MapMode.PRESENT)
    assert not bool(empty.layout_condition.appearance.binding_valid.any())

    null = _bundle()
    inference.apply_condition_mode(
        null, "tcmga", layout_mode="null"
    )
    assert int(null.layout_condition.actor_geometry.layout_mode.item()) == int(
        LayoutMode.NULL
    )
    assert int(null.layout_condition.map_mode.item()) == int(MapMode.NULL)


def test_t56_two_window_slice_keeps_a_global_and_checks_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _layout(frames=4)
    windows = ((0, 3), (1, 4))
    sliced = inference.strict_layout_slices(full, windows)
    assert sliced[0].appearance is full.appearance
    assert sliced[1].appearance is full.appearance
    assert sliced[0].num_frames == sliced[1].num_frames == 3
    result = inference.audit_strict_layout_windows(full, windows)
    assert result == [
        {"start": 0, "end": 3, "actor_slots": 1},
        {"start": 1, "end": 4, "actor_slots": 1},
    ]

    bad_projected = replace(
        sliced[1].projected_actor_geometry,
        bbox_patch=sliced[1].projected_actor_geometry.bbox_patch.clone(),
    )
    bad_projected.bbox_patch[0, 0, 0, 0] = 1.0
    bad = replace(sliced[1], projected_actor_geometry=bad_projected)
    monkeypatch.setattr(
        inference,
        "strict_layout_slices",
        lambda *_args, **_kwargs: (sliced[0], bad),
    )
    with pytest.raises(ValueError, match="projected overlap bbox_patch"):
        inference.audit_strict_layout_windows(full, windows)


def test_requested_camera_refresh_calls_atomic_dataset_builder() -> None:
    class FakeDataset:
        trunk_major_index = [(4, 0)]
        trunk_frames = 29

        def __init__(self) -> None:
            self.call = None

        def build_layout_payload_for_camera(self, *args, **kwargs):
            self.call = (args, kwargs)
            return {"layout_max_actors": torch.tensor(96)}

    camera = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 3, 1, 1, 1)
    batch = {
        "camera_to_world_corrected": camera,
        "start_idx": torch.tensor([29]),
    }
    dataset = FakeDataset()
    inference._refresh_layout_for_requested_camera(
        dataset=dataset,
        dataset_index=0,
        batch=batch,
    )
    args, kwargs = dataset.call
    assert args == (4, [29, 30, 31])
    assert kwargs["deterministic_reference"] is True
    assert torch.equal(kwargs["requested_camera_to_world"], camera[0])
    assert kwargs["outer_frame_indices"] == list(range(29, 58))
    assert int(batch["layout_max_actors"].item()) == 96


def test_train_layout_assembler_restores_all_projected_actor_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _layout(frames=2)
    geometry = source.actor_geometry
    projected = source.projected_actor_geometry
    appearance = source.appearance
    batch = {
        "layout_raster": source.raster,
        "map_metric": source.map_metric,
        "map_mode": source.map_mode,
        "static_far_plane_m": torch.tensor([120.0]),
        "actor_geometry_slot_track_id": geometry.slot_track_id,
        "actor_geometry_class_id": geometry.class_id,
        "actor_geometry_corners_world": geometry.corners_world,
        "actor_geometry_velocity_world": geometry.velocity_world,
        "actor_geometry_box_size": geometry.box_size,
        "actor_geometry_yaw": geometry.yaw,
        "actor_geometry_is_moving": geometry.is_moving,
        "actor_geometry_track_valid": geometry.track_valid,
        "actor_geometry_slot_valid": geometry.slot_valid,
        "actor_geometry_layout_mode": geometry.layout_mode,
        "actor_geometry_raw_track_key": geometry.raw_track_key,
        **{
            f"projected_actor_geometry_{name}": getattr(projected, name)
            for name in projected.__dataclass_fields__
        },
        "appearance_reference_rgb": torch.zeros((1, 1, 3, 4, 4)),
        "appearance_reference_alpha": torch.ones((1, 1, 1, 4, 4)),
        "appearance_geometry_idx": appearance.geometry_idx,
        "appearance_binding_valid": appearance.binding_valid,
        "appearance_class_id": torch.tensor([[0]], dtype=torch.int8),
        "appearance_mode": appearance.appearance_mode,
        "raster_schema_hash": RASTER_SCHEMA_HASH,
    }
    encoded = SimpleNamespace(
        appearance_tokens=appearance.appearance_tokens,
        appearance_mask=appearance.appearance_mask,
        canonical_uv=appearance.canonical_uv,
    )
    monkeypatch.setattr(
        pretrain,
        "_canonical_asset_encoder_for_model",
        lambda *_args, **_kwargs: lambda *_args, **_kwargs: encoded,
    )
    built = pretrain.build_layout_condition_from_batch(
        batch,
        None,
        None,
        torch.device("cpu"),
        layout_max_actors=2,
        patch_grid=(25, 37),
    )
    restored = built.layout.projected_actor_geometry
    for name in (
        "log_z_patch",
        "silhouette_uv",
        "silhouette_vertex_valid",
        "center_depth_valid",
        "frame_support",
        "metric_support",
        "in_frustum",
    ):
        torch.testing.assert_close(getattr(restored, name), getattr(projected, name))


def test_t56_summary_contains_layout_provenance_without_digest_fields() -> None:
    bundle = _bundle()
    batch = {
        "hdmap_schema_version": ["hdmap-v2"],
        "raster_schema_hash": [RASTER_SCHEMA_HASH],
        "attribute_source": ["none"],
        "actor_geometry_annotated_count": torch.tensor([12]),
        "actor_geometry_supported_count": torch.tensor([11]),
        "actor_geometry_eligible_count": torch.tensor([8]),
        "actor_geometry_ignored_unsupported_count": torch.tensor([1]),
        "actor_geometry_ignored_invalid_geometry_count": torch.tensor([2]),
        "actor_geometry_ignored_outside_requested_view_count": torch.tensor([3]),
        "actor_geometry_pre_cap_count": torch.tensor([105]),
        "actor_geometry_post_cap_count": torch.tensor([96]),
        "actor_geometry_overflow": torch.tensor([9]),
        "appearance_class_bucket_counts": torch.tensor([[2, 1, 1]]),
        "appearance_distance_bucket_counts": torch.tensor([[1, 2, 1]]),
        "appearance_area_bucket_counts": torch.tensor([[3, 1]]),
        "appearance_motion_bucket_counts": torch.tensor([[1, 3]]),
    }
    args = SimpleNamespace(
        hdmap_root="/maps",
        layout_mode="full",
        layout_max_actors=96,
        layout_guidance_scale=1.5,
        asset_control_guidance_scale=2.0,
    )
    summary = inference.layout_run_provenance(
        batch,
        bundle=bundle,
        args=args,
        condition_mode="tcmga",
        windows=[{"start": 0, "end": 3, "actor_slots": 1}],
    )
    assert summary["hdmap_schema_version"] == "hdmap-v2"
    assert summary["pre_cap_count"] == 105
    assert summary["post_cap_count"] == 96
    assert summary["overflow"] == 9
    assert summary["layout_max_actors"] == 96
    assert summary["static_far_plane_m"] == 120.0
    assert summary["render_camera_source"] == "requested_C"
    assert summary["render_gauge_source"] == "predicted_gauge"
    assert summary["layout_guidance_scale_requested"] == pytest.approx(1.5)
    assert summary["layout_guidance_scale_effective"] == pytest.approx(1.5)
    serialized = str(summary).lower()
    assert "sha" + "256" not in serialized


def test_p11_inference_source_has_no_removed_runtime_terms() -> None:
    source = Path(inference.__file__).read_text().lower()
    forbidden = (
        "camera_" + "gen",
        "camera_" + "guidance_scale",
        "camera_" + "text_guidance_scale",
        "decode_metric_camera_" + "from_features",
        "factor" + "ized",
        "asset_" + "manifest",
        "camera_geometry_" + "flow",
        "attri" + "bution",
        "checkpoint_" + "sha",
        "require_" + "sha",
        "sha" + "256",
        "place" + "ment_state",
        "target_bbox_" + "patch",
        "asset_" + "slot_embed",
    )
    assert all(term not in source for term in forbidden)


def _write_current_checkpoint(
    path: Path,
    *,
    include_ema: bool,
    ema_only: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model = WanSceneFlow(
        patch_grid=(25, 37),
        num_attention_heads=1,
        attention_head_dim=24,
        out_channels=4,
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
        layout_max_actors=2,
        layout_map_injection=False,
        layout_actor_injection=False,
        layout_map_metric_injection=False,
        layout_actor_metric_injection=False,
        appearance_context_injection=False,
    )
    model.set_gauge_stats(torch.zeros(3), torch.ones(3))
    raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ema = {key: value.detach().clone() for key, value in raw.items()}
    first_float = next(key for key, value in ema.items() if value.is_floating_point())
    ema[first_float] = ema[first_float] + 0.25
    schedule_args = SimpleNamespace(
        shift=10.0,
        weighting_scheme="waver",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        loss_weighting_scheme="none",
    )
    payload = {
        "step": 17,
        "scene_flow_config": model.config.to_dict(),
        "flow_schedule_config": build_flow_schedule_config(
            schedule_args,
            prediction_type=str(model.config.prediction_type),
            t_eps=float(model.config.t_eps),
        ),
        "scene_flow": ema if ema_only else raw,
    }
    if include_ema:
        payload["ema_scene_flow_state_dict"] = ema
    if ema_only:
        payload["is_ema_weights"] = True
    torch.save(payload, path)
    return raw, ema


def test_current_checkpoint_loader_handles_derived_config_and_ema_semantics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "full.pt"
    raw, ema = _write_current_checkpoint(path, include_ema=True)
    args = inference.build_argparser().parse_args(["--weights", str(path)])
    loaded, info = inference._require_current_checkpoint(
        path,
        device=torch.device("cpu"),
        use_ema=True,
        args=args,
    )
    assert info["weight_source"] == "ema_scene_flow_state_dict"
    assert info["ema_used"] is True
    assert info["step"] == 17
    first_float = next(key for key, value in ema.items() if value.is_floating_point())
    torch.testing.assert_close(loaded.state_dict()[first_float], ema[first_float])
    assert args.shift == pytest.approx(10.0)

    raw_args = inference.build_argparser().parse_args(["--weights", str(path)])
    loaded_raw, raw_info = inference._require_current_checkpoint(
        path,
        device=torch.device("cpu"),
        use_ema=False,
        args=raw_args,
    )
    assert raw_info["weight_source"] == "scene_flow"
    assert raw_info["ema_used"] is False
    torch.testing.assert_close(loaded_raw.state_dict()[first_float], raw[first_float])

    assert raw_args.layout_max_actors is None
    inference._sync_args_from_model(raw_args, loaded_raw, raw_info)
    assert raw_args.layout_max_actors == int(loaded_raw.config.layout_max_actors)

    requested_c = torch.eye(4).reshape(1, 1, 4, 4)
    anchor = torch.eye(4).reshape(1, 4, 4)
    patches = 25 * 37
    video = torch.zeros((1, 1, patches, 4))
    edit_mask = torch.zeros((1, 1, patches, 1))
    bundle = SimpleNamespace(
        z_clean_n=video,
        z_splat_n=torch.zeros_like(video),
        M_preserve=edit_mask,
        M_source=torch.zeros_like(edit_mask),
        M_dest=torch.ones_like(edit_mask),
        scene_gauge_clean_n=torch.zeros((1, 1, 3)),
        sky_gen_clean=None,
        layout_condition=_layout(frames=1),
        appearance_class_id=torch.tensor([[0]], dtype=torch.int8),
        camera_condition_tokens=torch.zeros((1, 1, 20)),
        camera_attention_mask=torch.ones((1, 1), dtype=torch.bool),
        frame_ids=torch.tensor([[0]]),
        fps=None,
        captions=None,
        camera_to_world_requested_metric=requested_c,
        camera_trajectory_anchor_to_world_metric=anchor,
        camera_intrinsics_requested_canvas_metric=torch.eye(3).reshape(
            1, 1, 3, 3
        ),
        camera_intrinsics_requested_raw_metric=2.0
        * torch.eye(3).reshape(1, 1, 3, 3),
        camera_requested_canvas_image_size_hw=(350, 518),
        camera_requested_raw_image_size_hw=torch.tensor([[1280, 1920]]),
    )
    sample_args = SimpleNamespace(
        val_sample_steps=2,
        shift=10.0,
        seed=5,
        guidance_scale=1.0,
        layout_guidance_scale=1.0,
        asset_control_guidance_scale=1.0,
        layout_to_gauge_grad_scale=1.0,
        val_sliding_window=0,
        val_sliding_stride=1,
    )
    sampled = pretrain.cfg_sample_pretrain_latents(
        loaded_raw,
        bundle,
        sample_args,
        step=17,
        device=torch.device("cpu"),
        return_gauge=True,
        return_sky_mask=True,
    )
    generated_gauge = sampled.gauge
    assert generated_gauge is not None
    assert sampled.video.shape == video.shape
    assert bool(torch.isfinite(sampled.video).all())
    assert bool(torch.isfinite(generated_gauge).all())
    requested_pose = inference.requested_render_pose_encoding(
        requested_c, anchor, generated_gauge
    )
    artifact = inference.offline_sample_tensor_payload(
        bundle, sampled, requested_pose
    )
    artifact_path = tmp_path / "sample.pt"
    torch.save(artifact, artifact_path)
    restored = torch.load(artifact_path, map_location="cpu")
    assert restored["render_camera_source"] == "requested_C"
    assert restored["render_gauge_source"] == "predicted_gauge"
    assert restored["requested_camera_canvas_image_size_hw"] == [350, 518]
    torch.testing.assert_close(
        restored["requested_camera_intrinsics_canvas_metric"],
        bundle.camera_intrinsics_requested_canvas_metric,
    )
    torch.testing.assert_close(
        restored["requested_render_pose_encoding"], requested_pose
    )

    mismatch_args = inference.build_argparser().parse_args(
        ["--weights", str(path), "--layout_max_actors", "7"]
    )
    with pytest.raises(ValueError, match="exactly match"):
        inference._sync_args_from_model(mismatch_args, loaded_raw, raw_info)

    raw_only_path = tmp_path / "raw_only.pt"
    raw_only, _ = _write_current_checkpoint(raw_only_path, include_ema=False)
    raw_only_args = inference.build_argparser().parse_args(
        ["--weights", str(raw_only_path)]
    )
    loaded_fallback, fallback_info = inference._require_current_checkpoint(
        raw_only_path,
        device=torch.device("cpu"),
        use_ema=True,
        args=raw_only_args,
    )
    assert fallback_info["weight_source"] == "scene_flow"
    assert fallback_info["ema_used"] is False
    assert fallback_info["ema_note"] == "checkpoint carries only raw scene_flow weights"
    fallback_key = next(
        key for key, value in raw_only.items() if value.is_floating_point()
    )
    torch.testing.assert_close(
        loaded_fallback.state_dict()[fallback_key], raw_only[fallback_key]
    )


def test_ema_only_checkpoint_rejects_no_ema(tmp_path: Path) -> None:
    path = tmp_path / "ema_only.pt"
    _write_current_checkpoint(path, include_ema=False, ema_only=True)
    args = inference.build_argparser().parse_args(["--weights", str(path)])
    with pytest.raises(ValueError, match="EMA-only"):
        inference._require_current_checkpoint(
            path,
            device=torch.device("cpu"),
            use_ema=False,
            args=args,
        )


def test_condition_mode_is_explicit_and_never_dataset_index_cycled() -> None:
    assert not hasattr(inference, "condition_mode_for_position")
    args = inference.build_argparser().parse_args(
        ["--weights", "model.pt", "--condition_mode", "tc"]
    )
    assert args.condition_mode == "tc"

    wrong_steps = inference.build_argparser().parse_args(
        ["--weights", "model.pt", "--sample_steps", "35"]
    )
    with pytest.raises(ValueError, match="frozen to 50"):
        inference.validate_args(wrong_steps)
