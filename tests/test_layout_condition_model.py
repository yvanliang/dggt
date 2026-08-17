from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from datasets.tools.hdmap_schema import RASTER_SCHEMA_HASH
from dggt.models.scene_flow import (
    AppearanceContextAdapter,
    FullActorGaugeAdapter,
    WanSceneFlow,
)
from dggt.utils.actor_geometry_condition import (
    MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
    ActorGeometryCondition,
    LayoutMode,
    ProjectedActorGeometry,
)
from dggt.utils.appearance_binding_condition import (
    AppearanceBindingCondition,
    AppearanceMode,
    gather_appearance_geometry,
)
from dggt.utils.layout_condition import LayoutConditionBatch, MapMode


PATCHES = 25 * 37
ACTORS = 96


def _tiny_model(*, layout_version: str = "layout_v2") -> WanSceneFlow:
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
        layout_condition_version=layout_version,
        layout_max_actors=ACTORS,
    )
    model.set_gauge_stats(torch.zeros(3), torch.ones(3))
    return model


def _appearance(batch: int, *, real: bool = False) -> AppearanceBindingCondition:
    ka, q = 2, 1
    binding = torch.zeros((batch, ka), dtype=torch.bool)
    indices = torch.full((batch, ka), -1, dtype=torch.int64)
    token_mask = torch.zeros((batch, ka, q), dtype=torch.bool)
    tokens = torch.zeros((batch, ka, q, 1024), dtype=torch.float32)
    if real:
        binding[:, 0] = True
        indices[:, 0] = 0
        token_mask[:, 0] = True
        tokens[:, 0] = 1.0
    return AppearanceBindingCondition(
        appearance_tokens=tokens,
        appearance_mask=token_mask,
        canonical_uv=torch.full((batch, ka, q, 2), 0.5, dtype=torch.float32),
        geometry_idx=indices,
        binding_valid=binding,
        appearance_mode=torch.full(
            (batch,),
            int(AppearanceMode.REAL if real else AppearanceMode.NULL),
            dtype=torch.int8,
        ),
    )


def _projected_fixture(
    valid: torch.Tensor,
    corners_camera: torch.Tensor,
) -> ProjectedActorGeometry:
    """Build the smallest contract-valid projected actor batch for model tests."""

    batch, slots, frames = (int(value) for value in valid.shape)
    bbox = torch.zeros((batch, slots, frames, 4), dtype=torch.float32)
    patch_weight = torch.zeros(
        (batch, slots, frames, PATCHES), dtype=torch.float32
    )
    log_z_patch = torch.zeros_like(patch_weight)
    silhouette_uv = torch.zeros(
        (
            batch,
            slots,
            frames,
            MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
            2,
        ),
        dtype=torch.float32,
    )
    silhouette_vertex_valid = torch.zeros(
        (
            batch,
            slots,
            frames,
            MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
        ),
        dtype=torch.bool,
    )
    uv_corners = torch.zeros((batch, slots, frames, 8, 2), dtype=torch.float32)
    uv_center = torch.zeros((batch, slots, frames, 2), dtype=torch.float32)
    log_z_w = torch.zeros((batch, slots, frames), dtype=torch.float32)
    base_triangle = torch.tensor(
        [[-0.04, -0.04], [0.04, -0.04], [0.0, 0.04]], dtype=torch.float32
    )
    for batch_index, slot, frame in valid.nonzero(as_tuple=False).tolist():
        depth_m = float(corners_camera[batch_index, slot, frame, :, 2].mean())
        if not depth_m >= 0.5:
            raise ValueError("active projected fixture needs an in-range centre")
        patch = PATCHES // 2 + slot
        center = torch.tensor(
            [0.50 + 0.01 * slot, 0.50 + 0.01 * slot], dtype=torch.float32
        )
        bbox[batch_index, slot, frame] = torch.tensor(
            [18.0 + 3.0 * slot, 12.0 + 2.0 * slot,
             19.0 + 3.0 * slot, 13.0 + 2.0 * slot],
            dtype=torch.float32,
        )
        patch_weight[batch_index, slot, frame, patch] = 1.0
        log_z_patch[batch_index, slot, frame, patch] = math.log(depth_m)
        silhouette_uv[batch_index, slot, frame, :3] = base_triangle + center
        silhouette_vertex_valid[batch_index, slot, frame, :3] = True
        uv_corners[batch_index, slot, frame] = center
        uv_center[batch_index, slot, frame] = center
        log_z_w[batch_index, slot, frame] = math.log(depth_m)
    return ProjectedActorGeometry(
        bbox_patch=bbox,
        patch_weight=patch_weight,
        log_z_patch=log_z_patch,
        silhouette_uv=silhouette_uv,
        silhouette_vertex_valid=silhouette_vertex_valid,
        corners_camera=corners_camera,
        uv_corners=uv_corners,
        velocity_camera=torch.zeros(
            (batch, slots, frames, 3), dtype=torch.float32
        ),
        uv_center=uv_center,
        log_z_w=log_z_w,
        center_depth_valid=valid.clone(),
        frame_support=valid.clone(),
        metric_support=(
            valid
            & corners_camera[..., 2].ge(0.5).all(dim=-1)
            & corners_camera[..., 2].le(120.0).all(dim=-1)
        ),
        in_frustum=valid.clone(),
        valid=valid.clone(),
    )


def _activate_second_actor(condition: LayoutConditionBatch) -> LayoutConditionBatch:
    """Activate G1 consistently in all geometry and projected-cache fields."""

    geometry = condition.actor_geometry
    geometry.slot_valid[:, 1] = True
    geometry.track_valid[:, 1] = True
    geometry.corners_world[:, 1, :, :, 2] = 14.0
    geometry.box_size[:, 1] = torch.tensor([4.0, 2.0, 1.5])
    geometry.slot_track_id[:, 1] = 2
    geometry.class_id[:, 1] = 0
    geometry.raw_track_key[0][1] = "vehicle-2"

    projected = condition.projected_actor_geometry
    projected.valid[:, 1] = True
    projected.center_depth_valid[:, 1] = True
    projected.frame_support[:, 1] = True
    projected.metric_support[:, 1] = True
    projected.in_frustum[:, 1] = True
    projected.corners_camera[:, 1, :, :, 2] = 14.0
    projected.uv_corners[:, 1] = 0.6
    projected.uv_center[:, 1] = 0.6
    projected.log_z_w[:, 1] = math.log(14.0)
    projected.bbox_patch[:, 1] = torch.tensor([21.0, 14.0, 23.0, 16.0])
    projected.patch_weight[:, 1].zero_()
    projected.log_z_patch[:, 1].zero_()
    projected.patch_weight[:, 1, :, PATCHES // 2 + 1] = 1.0
    projected.log_z_patch[:, 1, :, PATCHES // 2 + 1] = math.log(14.0)
    projected.silhouette_uv[:, 1].zero_()
    projected.silhouette_vertex_valid[:, 1].zero_()
    projected.silhouette_uv[:, 1, :, :3] = torch.tensor(
        [[0.56, 0.56], [0.64, 0.56], [0.60, 0.64]], dtype=torch.float32
    )
    projected.silhouette_vertex_valid[:, 1, :, :3] = True
    condition.validate()
    return condition


def _layout_batch(modes: list[LayoutMode]) -> tuple[LayoutConditionBatch, torch.Tensor]:
    batch, frames = len(modes), 1
    slot_valid = torch.zeros((batch, ACTORS), dtype=torch.bool)
    track_valid = torch.zeros((batch, ACTORS, frames), dtype=torch.bool)
    corners = torch.zeros((batch, ACTORS, frames, 8, 3), dtype=torch.float64)
    sizes = torch.zeros((batch, ACTORS, frames, 3), dtype=torch.float32)
    ids = torch.full((batch, ACTORS), -1, dtype=torch.int64)
    classes = torch.full((batch, ACTORS), -1, dtype=torch.int8)
    raw_keys = [[""] * ACTORS for _ in range(batch)]
    for row, mode in enumerate(modes):
        if mode in (LayoutMode.PARTIAL, LayoutMode.FULL):
            slot_valid[row, 0] = True
            track_valid[row, 0, 0] = True
            corners[row, 0, 0, :, 2] = 10.0
            sizes[row, 0, 0] = torch.tensor([4.0, 2.0, 1.5])
            ids[row, 0] = row + 1
            classes[row, 0] = 0
            raw_keys[row][0] = f"vehicle-{row + 1}"
    geometry = ActorGeometryCondition(
        slot_track_id=ids,
        class_id=classes,
        corners_world=corners,
        velocity_world=torch.zeros((batch, ACTORS, frames, 3), dtype=torch.float32),
        box_size=sizes,
        yaw=torch.zeros((batch, ACTORS, frames), dtype=torch.float32),
        is_moving=torch.zeros((batch, ACTORS, frames), dtype=torch.bool),
        track_valid=track_valid,
        slot_valid=slot_valid,
        layout_mode=torch.tensor([int(mode) for mode in modes], dtype=torch.int8),
        raw_track_key=raw_keys,
    )

    projected = _projected_fixture(track_valid, corners.float())

    raster = torch.zeros((batch, frames, 33, 100, 148), dtype=torch.uint8)
    raster[:, :, (10, 11, 28, 29)] = 127
    metric = torch.zeros((batch, frames, PATCHES, 5, 4), dtype=torch.float32)
    map_mode = torch.empty((batch,), dtype=torch.int8)
    for row, mode in enumerate(modes):
        if mode == LayoutMode.NULL:
            map_mode[row] = int(MapMode.NULL)
        elif mode == LayoutMode.EMPTY:
            map_mode[row] = int(MapMode.EMPTY)
        else:
            map_mode[row] = int(MapMode.PRESENT)
            raster[row, :, 1, 48:52, 72:76] = 255
            raster[row, :, 21, 48:52, 72:76] = 255
            metric[row, :, PATCHES // 2, 1] = torch.tensor(
                [0.5, 0.5, math.log(10.0), 1.0]
            )
            raster[row, :, 22, 48:52, 72:76] = 255
            raster[row, :, 32, 48:52, 72:76] = 255
    appearance = _appearance(batch)
    condition = LayoutConditionBatch(
        raster=raster,
        map_metric=metric,
        actor_geometry=geometry,
        projected_actor_geometry=projected,
        appearance=appearance,
        map_mode=map_mode,
        raster_schema_hash=RASTER_SCHEMA_HASH,
    )
    appearance_class = torch.full((batch, 2), -1, dtype=torch.int8)
    return condition, appearance_class


def _model_inputs(batch: int) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    shape = (batch, 1, PATCHES, 4)
    z_t = torch.randn(shape)
    zeros = torch.zeros(shape)
    masks = torch.zeros((batch, 1, PATCHES, 1))
    args = (z_t, torch.full((batch,), 0.5), zeros, zeros, masks, masks, masks)
    kwargs = {
        "camera_condition_tokens": torch.zeros((batch, 1, 20)),
        "gauge_gen_tokens": torch.zeros((batch, 1, 3)),
        "gauge_gen_attention_mask": torch.ones((batch, 1), dtype=torch.bool),
    }
    return args, kwargs


def _layout_kwargs(
    condition: LayoutConditionBatch,
    appearance_class_id: torch.Tensor,
) -> dict[str, object]:
    return {
        "layout_raster": condition.raster,
        "map_metric": condition.map_metric,
        "actor_geometry": condition.actor_geometry,
        "projected_actor_geometry": condition.projected_actor_geometry,
        "appearance": condition.appearance,
        "map_mode": condition.map_mode,
        "raster_schema_hash": condition.raster_schema_hash,
        "appearance_class_id": appearance_class_id,
    }


def _copy_common_weights(source: WanSceneFlow, target: WanSceneFlow) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in tuple(target_state.items()):
        if key in source_state and source_state[key].shape == value.shape:
            target_state[key] = source_state[key].clone()
    target.load_state_dict(target_state, strict=True)


def test_excluded_static_map_slots_fail_closed_at_batch_boundary():
    condition, _ = _layout_batch([LayoutMode.FULL])

    for channel in (0, 6):
        raster = condition.raster.clone()
        raster[:, :, channel, 0, 0] = 255
        with pytest.raises(ValueError, match="excluded static-map"):
            replace(condition, raster=raster)

    metric = condition.map_metric.clone()
    metric[:, :, 0, 0] = torch.tensor([0.5, 0.5, math.log(10.0), 1.0])
    with pytest.raises(ValueError, match="excluded lane-centerline"):
        replace(condition, map_metric=metric)


def test_excluded_static_map_slots_fail_closed_at_model_entry():
    condition, appearance_class = _layout_batch([LayoutMode.FULL])
    model = _tiny_model().eval()
    args, common = _model_inputs(1)
    kwargs = _layout_kwargs(condition, appearance_class)

    for channel in (6, 7):
        raster = condition.raster.clone()
        raster[:, :, channel, 0, 0] = 255
        with pytest.raises(ValueError, match="excluded static-map"):
            model(*args, **common, **{**kwargs, "layout_raster": raster})

    metric = condition.map_metric.clone()
    metric[:, :, 0, 0] = torch.tensor([0.5, 0.5, math.log(10.0), 1.0])
    with pytest.raises(ValueError, match="excluded lane-centerline"):
        model(*args, **common, **{**kwargs, "map_metric": metric})


def test_t37_t38_zero_init_matches_layout_none_and_does_not_add_tokens():
    torch.manual_seed(7)
    layout_model = _tiny_model().eval()
    none_model = _tiny_model(layout_version="none").eval()
    _copy_common_weights(layout_model, none_model)
    condition, appearance_class = _layout_batch([LayoutMode.FULL])
    args, common = _model_inputs(1)
    lengths: dict[str, int] = {}

    def capture(name: str):
        def hook(_module, values):
            lengths[name] = int(values[0].shape[1])
        return hook

    handle = layout_model.blocks[0].register_forward_pre_hook(capture("layout"))
    with torch.no_grad():
        out_layout = layout_model(
            *args,
            **common,
            **_layout_kwargs(condition, appearance_class),
        )
    handle.remove()
    handle = none_model.blocks[0].register_forward_pre_hook(capture("none"))
    with torch.no_grad():
        out_none = none_model(*args, **common)
    handle.remove()

    assert lengths["layout"] == lengths["none"]
    assert set(out_layout) == {"video", "sky", "gauge"}
    for key in ("video", "gauge"):
        assert torch.equal(out_layout[key], out_none[key])
    forbidden_fragment = "camera" + "_gen"
    assert all(forbidden_fragment not in key for key in layout_model.state_dict())


def test_t39_t40_modes_are_distinct_and_tc_is_leak_free():
    model = _tiny_model()
    assert torch.unique(model.layout_actor_stem.mode_embedding.weight, dim=0).shape[0] == 4
    assert torch.unique(model.layout_map_stem.mode_embedding.weight, dim=0).shape[0] == 3

    full, _ = _layout_batch([LayoutMode.FULL])
    tc = full.without_layout()
    assert int(tc.actor_geometry.layout_mode.item()) == int(LayoutMode.NULL)
    assert int(tc.map_mode.item()) == int(MapMode.NULL)
    assert not bool(tc.map_metric.any())
    assert not bool(tc.actor_geometry.slot_valid.any())
    assert not bool(tc.projected_actor_geometry.valid.any())
    assert not bool(tc.appearance.binding_valid.any())
    # Signed directions encode physical zero at their quantized zero point.
    assert torch.all(tc.raster[:, :, (10, 11, 28, 29)] == 127)


def test_t41_t42c_a_permutation_and_geometry_move_use_only_g_addresses():
    base_condition, _ = _layout_batch([LayoutMode.FULL])
    appearance = _appearance(1, real=True)
    first = gather_appearance_geometry(appearance, base_condition.projected_actor_geometry)

    moved_bbox = base_condition.projected_actor_geometry.bbox_patch.clone()
    moved_bbox[:, 0, :, (0, 2)] += 3.0
    moved_projected = replace(
        base_condition.projected_actor_geometry,
        bbox_patch=moved_bbox,
    )
    moved = gather_appearance_geometry(appearance, moved_projected)
    assert torch.equal(appearance.appearance_tokens, appearance.appearance_tokens.clone())
    torch.testing.assert_close(
        moved.token_patch_xy[0, 0, ..., 0],
        first.token_patch_xy[0, 0, ..., 0] + 3.0,
    )
    assert not any("slot_embed" in key for key in _tiny_model().state_dict())

    permuted = appearance.permute_bindings([1, 0])
    for name in appearance.__dataclass_fields__:
        assert torch.equal(
            getattr(appearance.canonicalized(), name),
            getattr(permuted.canonicalized(), name),
        )


def test_t41_permuted_a_uses_the_same_canonical_order_in_late_scatter():
    """The pooled A rows and geometry_idx reaching late scatter must agree."""

    condition, appearance_class = _layout_batch([LayoutMode.FULL])
    # Make a second real G slot so both A bindings are contract-valid.
    condition = _activate_second_actor(condition)

    appearance = _appearance(1, real=True)
    appearance.binding_valid[:, 1] = True
    appearance.geometry_idx[:, 1] = 1
    appearance.appearance_mask[:, 1] = True
    appearance.appearance_tokens[:, 1] = 2.0
    permuted = appearance.permute_bindings([1, 0])
    condition = LayoutConditionBatch(
        raster=condition.raster,
        map_metric=condition.map_metric,
        actor_geometry=condition.actor_geometry,
        projected_actor_geometry=condition.projected_actor_geometry,
        appearance=permuted,
        map_mode=condition.map_mode,
        raster_schema_hash=condition.raster_schema_hash,
    )
    appearance_class[:] = 0
    model = _tiny_model().eval()
    seen: dict[str, torch.Tensor] = {}

    def capture_late_scatter(_module, values):
        seen["pooled"] = values[0].detach().clone()
        seen["geometry_idx"] = values[1].detach().clone()
        seen["binding_valid"] = values[2].detach().clone()

    handle = model.appearance_context_adapter.register_forward_pre_hook(
        capture_late_scatter
    )
    args, common = _model_inputs(1)
    with torch.no_grad():
        model(*args, **common, **_layout_kwargs(condition, appearance_class))
    handle.remove()

    assert seen["geometry_idx"].tolist() == [[0, 1]]
    assert seen["binding_valid"].tolist() == [[True, True]]
    assert not torch.equal(seen["pooled"][:, 0], seen["pooled"][:, 1])


def test_t44_depth_aware_soft_z_buffer_prefers_near_actor():
    adapter = FullActorGaugeAdapter(hidden_size=1)
    with torch.no_grad():
        adapter.net[-1].weight.fill_(1.0)
        adapter.net[-1].bias.zero_()
    features = torch.ones((1, 2, 1, 27))
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    log_z_patch = torch.log(torch.tensor([[[[2.0]], [[8.0]]]]))
    support = torch.ones((1, 2, 1, 1))
    _context, weights = adapter(
        features, valid, log_z_patch, support, depth_tau=0.5
    )
    assert weights[0, 0, 0, 0] > weights[0, 1, 0, 0]
    assert not torch.isclose(weights[0, 0, 0, 0], torch.tensor(0.5))


def test_t44_patch_depth_can_change_actor_order_across_the_image():
    adapter = FullActorGaugeAdapter(hidden_size=1)
    with torch.no_grad():
        adapter.net[-1].weight.fill_(1.0)
        adapter.net[-1].bias.zero_()
    features = torch.ones((1, 2, 1, 27))
    valid = torch.ones((1, 2, 1), dtype=torch.bool)
    # Actor 0 is nearer in patch 0 while actor 1 is nearer in patch 1.
    log_z_patch = torch.log(
        torch.tensor([[[[2.0, 9.0]], [[8.0, 3.0]]]])
    )
    support = torch.ones((1, 2, 1, 2))
    _context, weights = adapter(
        features, valid, log_z_patch, support, depth_tau=0.5
    )
    assert weights[0, 0, 0, 0] > weights[0, 1, 0, 0]
    assert weights[0, 1, 0, 1] > weights[0, 0, 0, 1]


def test_t44_model_reader_consumes_per_patch_surface_depth():
    condition, _ = _layout_batch([LayoutMode.FULL])
    condition = _activate_second_actor(condition)
    projected = condition.projected_actor_geometry
    left_patch = PATCHES // 2
    right_patch = left_patch + 1
    projected.patch_weight[:, :2].zero_()
    projected.log_z_patch[:, :2].zero_()
    projected.patch_weight[0, :2, 0, left_patch] = 1.0
    projected.patch_weight[0, :2, 0, right_patch] = 1.0
    projected.log_z_patch[0, 0, 0, left_patch] = math.log(2.0)
    projected.log_z_patch[0, 1, 0, left_patch] = math.log(8.0)
    projected.log_z_patch[0, 0, 0, right_patch] = math.log(9.0)
    projected.log_z_patch[0, 1, 0, right_patch] = math.log(3.0)
    projected.validate()

    model = _tiny_model().eval()
    _context, weights, valid = model._build_actor_metric_context(
        condition.actor_geometry,
        projected,
        torch.zeros((1, 3), dtype=torch.float32),
        torch.ones((1,), dtype=torch.bool),
        grad_scale=1.0,
    )

    assert valid[0, :2, 0].tolist() == [True, True]
    assert weights[0, 0, 0, left_patch] > weights[0, 1, 0, left_patch]
    assert weights[0, 1, 0, right_patch] > weights[0, 0, 0, right_patch]


def test_clipped_actor_token_support_is_explicitly_separate_from_late_metric_support():
    condition, _ = _layout_batch([LayoutMode.FULL])
    corners = condition.projected_actor_geometry.corners_camera.clone()
    corners[0, 0, 0, :, 2] = torch.tensor(
        [0.3, 1.7, 0.3, 1.7, 0.3, 1.7, 0.3, 1.7],
        dtype=torch.float32,
    )
    projected = _projected_fixture(
        condition.actor_geometry.track_valid,
        corners,
    )
    condition = replace(condition, projected_actor_geometry=projected)

    assert bool(projected.frame_support[0, 0, 0])
    assert not bool(projected.metric_support[0, 0, 0])
    gathered = gather_appearance_geometry(_appearance(1, real=True), projected)
    assert bool(gathered.addr_ok[0, 0, 0])

    model = _tiny_model().eval()
    _context, weights, metric_valid = model._build_actor_metric_context(
        condition.actor_geometry,
        projected,
        torch.zeros((1, 3), dtype=torch.float32),
        torch.ones((1,), dtype=torch.bool),
        grad_scale=1.0,
    )
    assert not bool(metric_valid[0, 0, 0])
    assert not bool(weights[0, 0, 0].any())


def test_t42b_late_appearance_scatter_binds_values_to_exact_g_support():
    adapter = AppearanceContextAdapter(hidden_size=1)
    adapter.net = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        adapter.net[0].weight.fill_(1.0)
    pooled = torch.tensor([[[2.0], [7.0], [999.0]]])
    geometry_idx = torch.tensor([[1, 0, -1]], dtype=torch.int64)
    binding_valid = torch.tensor([[True, True, False]])
    actor_weights = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])

    context = adapter(pooled, geometry_idx, binding_valid, actor_weights)

    assert tuple(context.shape) == (1, 1, 2, 1)
    # A1=7 is bound to G0/p0, A0=2 is bound to G1/p1.  The invalid 999 row
    # cannot leak through geometry_idx=-1 into the final G slot.
    torch.testing.assert_close(context.flatten(), torch.tensor([7.0, 2.0]))


def test_t45_t46_kg96_all_four_modes_forward_backward():
    modes = [LayoutMode.NULL, LayoutMode.EMPTY, LayoutMode.PARTIAL, LayoutMode.FULL]
    condition, appearance_class = _layout_batch(modes)
    model = _tiny_model().train()
    args, common = _model_inputs(len(modes))
    out = model(
        *args,
        **common,
        **_layout_kwargs(condition, appearance_class),
        return_layout_diagnostics=True,
        layout_to_gauge_grad_scale=0.5,
    )
    assert condition.actor_geometry.num_slots == ACTORS
    assert set(out) == {
        "video",
        "sky",
        "gauge",
        "actor_alignment_diagnostics",
    }
    loss = out["video"].square().mean() + out["gauge"].square().mean()
    loss.backward()
    assert model.final_layer.linear.weight.grad is not None


def test_map_metric_valid_fraction_excludes_reserved_group_and_absent_rows():
    # [B=3, S=1, P=2, Gm=5, 4].  Row 0 carries a map, rows 1 and 2 were
    # neutralized by the TC task and by a factual EMPTY window.
    metric = torch.zeros((3, 1, 2, 5, 4))
    metric[0, 0, 0, 1:, 3] = 1.0
    fallback = torch.zeros(())
    map_mode = torch.tensor(
        [int(MapMode.PRESENT), int(MapMode.NULL), int(MapMode.EMPTY)],
        dtype=torch.int8,
    )

    fraction = WanSceneFlow._map_metric_valid_fraction(
        metric, map_mode, fallback=fallback
    )
    # Only row 0 counts: 4 valid groups on 1 of its 2 patches.
    assert float(fraction) == pytest.approx(0.5)

    # The reserved zero group must never dilute the denominator: adding it back
    # would give 4/10 instead of 4/8.
    assert float(fraction) != pytest.approx(0.4)

    # Averaging over all three rows is what made this curve look collapsing.
    assert float(metric[..., 1:, 3].mean()) == pytest.approx(0.5 / 3.0)

    # No PRESENT row at all is a defined zero, not a NaN.
    absent = torch.full((3,), int(MapMode.NULL), dtype=torch.int8)
    assert float(
        WanSceneFlow._map_metric_valid_fraction(metric, absent, fallback=fallback)
    ) == 0.0


def test_t43_model_entry_rechecks_a_g_class_contract():
    condition, appearance_class = _layout_batch([LayoutMode.FULL])
    real = _appearance(1, real=True)
    condition = LayoutConditionBatch(
        raster=condition.raster,
        map_metric=condition.map_metric,
        actor_geometry=condition.actor_geometry,
        projected_actor_geometry=condition.projected_actor_geometry,
        appearance=real,
        map_mode=condition.map_mode,
        raster_schema_hash=condition.raster_schema_hash,
    )
    appearance_class[:] = 1
    model = _tiny_model()
    args, common = _model_inputs(1)
    with pytest.raises(ValueError, match="A class must match"):
        model(
            *args,
            **common,
            **_layout_kwargs(condition, appearance_class),
        )
