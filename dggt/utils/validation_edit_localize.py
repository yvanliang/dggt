"""Decoupled localization for the validation flow-cache pipeline.

Mode A's :func:`localize_objects` couples *delete* + *re-render the same
object's asset* per Waymo slot. Validation needs them decoupled and with
external assets / shifted boxes:

* **Delete slots (0,1,2)** -> stock :func:`localize_objects` with
  ``load_asset=False`` (``asset_local`` becomes ``None`` so the produced
  ``LocalizedFrameObject`` has *empty* asset fields and contributes only
  deletion via :func:`apply_mode_a`). 100% the battle-tested delete path,
  including Sim3 alignment + corner-projection pose refine + semantic-mask
  component extraction.
* **Asset slots (3,4,5)** -> custom placement (no semantic gating): the
  target box is the *refined* box of the corresponding delete slot
  (replacement = slot-1 box, reposition = slot-2 box + local-axis shift) or
  the Sim3-transformed insertion tar box (insertion = slot-3, no source).
  The chosen asset's Gaussians are fitted with the same
  :func:`_transform_asset_gaussians_simple` used by Mode A, so the emitted
  ``LocalizedFrameObject`` is schema-identical and the downstream
  ``AssetAggregatorPass`` / cache packers run unchanged.

The reposition shift replicates ``pointcloud_validation`` 's
``EditProcessor.shift_object_tfm_by_action`` (REPOSITION_DISTANCE_M = 3.0):
``up -> +localX``, ``down -> -localX``, ``left -> +localY``, ``right -> -localY``
(local axes = columns of the DGGT box rotation; distance scaled by the Sim3
scale so it stays a physical 3 m).
"""
from __future__ import annotations

import math
from typing import Any

import torch

from dggt.utils.gaussian_edit import (
    LocalizedFrameObject,
    Sim3Transform,
    _asset_object_to_world_matrix,
    _base_corner_projection_result,
    _compute_asset_scale_factors,
    _edge_midpoints,
    _load_asset_gaussians,
    _orthonormalize_rotation,
    _project_asset_bbox_simple,
    _solve_proposal_center_with_fixed_rotation,
    _transform_asset_gaussians_simple,
    _transform_track_box,
    build_box_corners,
    compute_bbox_from_projected_points,
    project_world_points,
)

REPOSITION_DISTANCE_M = 3.0


def _empty_long() -> torch.Tensor:
    return torch.zeros((0,), dtype=torch.long)


def _apply_reposition_shift(
    center: torch.Tensor,
    rotation: torch.Tensor,
    action: str,
    distance: float,
) -> torch.Tensor:
    """Replicate ``shift_object_tfm_by_action`` in the DGGT box frame.

    ``rotation`` columns are the box local axes (col0 = local X / forward,
    col1 = local Y / left), matching the 4x4 ``tfm[:3,0]`` / ``tfm[:3,1]``
    convention of the reference implementation.
    """
    action = str(action).lower()
    if action == "up":
        dir_vec = rotation[:, 0]
    elif action == "down":
        dir_vec = -rotation[:, 0]
    elif action == "left":
        dir_vec = rotation[:, 1]
    elif action == "right":
        dir_vec = -rotation[:, 1]
    else:  # unknown -> forward (matches reference 'else' branch)
        dir_vec = rotation[:, 0]
    dir_unit = dir_vec / (dir_vec.norm() + 1e-8)
    return center + dir_unit * float(distance)


def _build_asset_lfo(
    *,
    sample: dict[str, Any],
    slot_idx: int,
    frame_idx: int,
    asset_local: dict[str, torch.Tensor],
    asset_path: str,
    target_center: torch.Tensor,
    target_size: torch.Tensor,
    target_rotation: torch.Tensor,
    clean_state: Any,
    image_hw: tuple[int, int],
    asset_yaw_correction_deg: float,
) -> LocalizedFrameObject:
    """Place ``asset_local`` on the DGGT target box; emit an asset-only LFO.

    Mirrors the asset branch of :func:`localize_objects` (gaussian_edit.py
    ~4765-4886) but with empty delete indices and no semantic mask.
    """
    c = target_center.detach().cpu().float()
    s = target_size.detach().cpu().float()
    R = _orthonormalize_rotation(target_rotation.detach().cpu().float())

    asset_object_to_world = _asset_object_to_world_matrix(R, c)
    asset_scale_factors = _compute_asset_scale_factors(asset_local, s)
    asset_world = _transform_asset_gaussians_simple(
        asset_local,
        s,
        R,
        c,
        asset_yaw_correction_deg=asset_yaw_correction_deg,
    )
    source_front_index = int(frame_idx)  # views=1
    projected_asset_bbox = _project_asset_bbox_simple(
        asset_local=asset_local,
        target_lwh=s,
        object_rotation=R,
        object_center=c,
        camera_to_world=clean_state.camera_to_world[source_front_index],
        intrinsics=clean_state.intrinsics[source_front_index],
        image_hw=image_hw,
        asset_yaw_correction_deg=asset_yaw_correction_deg,
        scale_factors=asset_scale_factors,
    )
    means_raw = asset_local["means_raw"]
    asset_scale = float(
        torch.mean(
            s
            / (means_raw.max(dim=0).values - means_raw.min(dim=0).values).clamp_min(1e-6)
        ).item()
    )

    # Corner-projection diagnostics from the (DGGT) target box itself.
    corners = build_box_corners(c, s, R)
    c2w = clean_state.camera_to_world[source_front_index]
    K = clean_state.intrinsics[source_front_index]
    corner_uv, _, corner_valid = project_world_points(corners, c2w, K, image_hw)
    edge_uv, _, edge_valid = project_world_points(_edge_midpoints(corners), c2w, K, image_hw)
    target_bbox_model = compute_bbox_from_projected_points(corner_uv, corner_valid)
    if target_bbox_model is None:
        target_bbox_model = projected_asset_bbox
    if target_bbox_model is None:
        target_bbox_model = torch.zeros((4,), dtype=torch.float32)
    pose_result = _base_corner_projection_result(
        base_center=c,
        object_size=s,
        base_rotation=R,
        waymo_corner_uv=corner_uv.detach().cpu().float(),
        waymo_corner_valid=corner_valid.detach().cpu(),
        waymo_edge_uv=edge_uv.detach().cpu().float(),
        waymo_edge_valid=edge_valid.detach().cpu(),
        target_bbox_model=target_bbox_model,
        camera_to_world=c2w,
        intrinsics=K,
        status="asset_only",
        reason="validation_asset_placement",
    )

    return LocalizedFrameObject(
        slot_idx=int(slot_idx),
        frame_idx=int(frame_idx),
        source_front_index=source_front_index,
        asset_object_id=str(sample["object_asset_ids"][slot_idx]),
        scene_raw_object_id=str(sample["object_scene_raw_ids"][slot_idx]),
        asset_path=str(asset_path),
        match_score=1.0,
        delete_motion_mode="asset_only",
        waymo_frame_dynamic=bool(
            sample["object_is_moving_frame_selected"][slot_idx, frame_idx].item()
        ),
        waymo_frame_speed_mps=0.0,
        waymo_max_speed_mps=0.0,
        waymo_mean_speed_mps=0.0,
        render_dynamic_ratio=0.0,
        gt_center=c.clone(),
        gt_size=s.clone(),
        gt_rotation=R.clone(),
        proposal_center=c.clone(),
        proposal_size=s.clone(),
        proposal_rotation=R.clone(),
        refined_center=c.clone(),
        refined_size=s.clone(),
        refined_rotation=R.clone(),
        asset_rotation=R.clone(),
        asset_scale=asset_scale,
        asset_bottom_center=c.clone(),
        delete_core_indices=_empty_long(),
        delete_shell_indices=_empty_long(),
        candidate_count=0,
        seed_point_count=0,
        candidate_pool_count=0,
        cluster_kept_count=0,
        target_delete_coverage=0.0,
        outside_box_leak_ratio=0.0,
        target_bbox_model=target_bbox_model,
        projected_asset_bbox=projected_asset_bbox,
        seed_pixel_mask=None,
        delete_component_pixel_mask=None,
        asset_means_world=asset_world["means"],
        asset_colors=asset_world["colors"],
        asset_opacities=asset_world["opacities"],
        asset_scales=asset_world["scales"],
        asset_quats=asset_world["quats"],
        asset_means_local=asset_local["means_raw"],
        asset_scales_local=asset_local["scales"],
        asset_quats_local=asset_local["quats"],
        asset_scale_factors=asset_scale_factors,
        asset_object_to_world=asset_object_to_world,
        asset_local_yaw_deg=float(asset_yaw_correction_deg),
        pose_refine_diagnostics=pose_result["diagnostics"],
        waymo_box_corners_model=pose_result["waymo_uv"],
        waymo_box_corners_valid=pose_result["waymo_valid"],
        initial_box_corners_model=pose_result["initial_uv"],
        initial_box_corners_valid=pose_result["initial_valid"],
        refined_box_corners_model=pose_result["refined_uv"],
        refined_box_corners_valid=pose_result["refined_valid"],
    )


def localize_validation_objects(
    sample: dict[str, Any],
    clean_state: Any,
    alignment: Sim3Transform,
    *,
    editor: Any,
    asset_yaw_correction_deg: float = 180.0,
) -> dict[str, Any]:
    """Return ``{"delete_lfos", "asset_lfos", "refined_box"}``.

    ``delete_lfos`` are produced by stock ``editor.localize`` over the delete
    slots with ``load_asset=False`` (empty assets). ``asset_lfos`` are the
    custom placements for the present asset slots.
    """
    ve = sample["validation_edit"]
    delete_slots = [int(s) for s in ve["delete_slots"]]
    asset_slots = [int(s) for s in ve["asset_slots"]]
    asset_source_box_slot = {int(k): int(v) for k, v in ve["asset_source_box_slot"].items()}
    action = str(ve["action_for_reposition"])

    image_hw = tuple(int(v) for v in clean_state.images.shape[-2:])

    # ---- Phase A: stock delete localization (no asset) --------------------
    delete_lfos: list[LocalizedFrameObject] = editor.localize(
        sample,
        clean_state,
        alignment,
        delete_slots,
        load_asset=False,
    )

    # refined_box[slot][frame] = (center, size, rotation) -- the final
    # pose-refined box the asset side reuses (== Mode A's asset placement box).
    refined_box: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for lfo in delete_lfos:
        refined_box.setdefault(int(lfo.slot_idx), {})[int(lfo.frame_idx)] = (
            lfo.proposal_center.detach().cpu().float(),
            lfo.proposal_size.detach().cpu().float(),
            lfo.proposal_rotation.detach().cpu().float(),
        )

    # ---- Phase B: asset placement ----------------------------------------
    asset_cache: dict[str, dict[str, torch.Tensor]] = {}
    asset_lfos: list[LocalizedFrameObject] = []
    obj_to_world = sample["object_obj_to_world_selected"]
    box_size = sample["object_box_size_selected"]
    track_valid = sample["object_track_valid_mask_selected"]
    bbox_present = sample["object_bbox_present_mask_selected"]
    bbox_model = sample["object_bbox_model_selected"]

    for slot in asset_slots:
        asset_path = str(sample["object_asset_paths"][slot])
        asset_local = _load_asset_gaussians(asset_path, asset_cache)
        for f in range(int(track_valid.shape[1])):
            if not bool(track_valid[slot, f].item()):
                continue

            if slot == 3:
                # Insertion: no source object -> Sim3-transformed tar box,
                # optional depth-snap against the projected box bbox.
                c, s, R = _transform_track_box(
                    obj_to_world[slot, f], box_size[slot, f], alignment
                )
                R = _orthonormalize_rotation(R)
                if bool(bbox_present[slot, f, 0].item()):
                    try:
                        c, _ = _solve_proposal_center_with_fixed_rotation(
                            object_center=c,
                            object_size=s,
                            object_rotation=R,
                            target_bbox_model=bbox_model[slot, f, 0],
                            camera_to_world=clean_state.camera_to_world[f],
                            intrinsics=clean_state.intrinsics[f],
                            image_hw=image_hw,
                            point_map_world=clean_state.point_map_world[f],
                            depth_map=clean_state.depth[f],
                            valid_mask=clean_state.valid_mask[f],
                        )
                    except Exception:
                        pass
            else:
                src = asset_source_box_slot[slot]
                rb = refined_box.get(src, {}).get(f)
                if rb is None:
                    # delete localization missed this frame -> Sim3 fallback.
                    c, s, R = _transform_track_box(
                        obj_to_world[src, f], box_size[src, f], alignment
                    )
                    R = _orthonormalize_rotation(R)
                else:
                    c, s, R = rb[0].clone(), rb[1].clone(), rb[2].clone()
                if slot == 5:
                    c = _apply_reposition_shift(
                        c, R, action, REPOSITION_DISTANCE_M * float(alignment.scale)
                    )

            asset_lfos.append(
                _build_asset_lfo(
                    sample=sample,
                    slot_idx=slot,
                    frame_idx=f,
                    asset_local=asset_local,
                    asset_path=asset_path,
                    target_center=c,
                    target_size=s,
                    target_rotation=R,
                    clean_state=clean_state,
                    image_hw=image_hw,
                    asset_yaw_correction_deg=asset_yaw_correction_deg,
                )
            )

    return {
        "delete_lfos": delete_lfos,
        "asset_lfos": asset_lfos,
        "refined_box": refined_box,
    }


# Per-variant slot membership (delete slots / asset slots).
VARIANT_SLOTS: dict[str, dict[str, tuple[int, ...]]] = {
    "combined": {"delete": (0, 1, 2), "asset": (3, 4, 5)},
    "delete": {"delete": (0, 1, 2), "asset": ()},
    "add": {"delete": (), "asset": (3,)},
    "replace": {"delete": (1,), "asset": (4,)},
    "move": {"delete": (2,), "asset": (5,)},
}


def subset_localized_for_variant(
    localized: dict[str, Any],
    variant: str,
) -> list[LocalizedFrameObject]:
    """Filter the full localize output to one edit variant's LFO list."""
    spec = VARIANT_SLOTS[variant]
    del_slots = set(spec["delete"])
    asset_slots = set(spec["asset"])
    out: list[LocalizedFrameObject] = []
    for lfo in localized["delete_lfos"]:
        if int(lfo.slot_idx) in del_slots:
            out.append(lfo)
    for lfo in localized["asset_lfos"]:
        if int(lfo.slot_idx) in asset_slots:
            out.append(lfo)
    return out
