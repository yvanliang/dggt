"""Deterministic Mode-A scene editor, repackaged as an :class:`nn.Module`.

Phase 1 of the FlowDGGT implementation plan. Wraps the existing
``gaussian_edit.*`` utilities (``build_clean_scene_state`` →
``estimate_scene_alignment`` → ``localize_objects`` → ``apply_mode_a``) so that
the inference script, future training loops, and smoke tests can invoke one
shared pipeline instead of copying the same procedural chain.

The module is currently ``views=1`` only to match the baseline; ``views>=2``
support is planned for a later Phase 1 iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from dggt.models.gaussian_pointers import GaussianPointers
from dggt.utils.gaussian_edit import (
    CleanSceneState,
    EditedSceneState,
    LocalizedFrameObject,
    Sim3Transform,
    apply_mode_a,
    build_clean_scene_state,
    estimate_scene_alignment,
    localize_objects,
    parse_object_slots,
)
from dggt.utils.pose_enc import pose_encoding_to_extri_intri


@dataclass
class EditedSceneBundle:
    """Unified container for the Phase 1 editor output.

    Fields match the contract in the implementation plan so Phase 2+ can drop
    in per-Gaussian pointers / asset aggregation without changing this shape.
    """

    clean_state: CleanSceneState
    alignment: Sim3Transform
    edited_state: EditedSceneState
    cameras_dggt: dict[str, torch.Tensor]
    cameras_waymo: dict[str, torch.Tensor]
    T_w2d: Sim3Transform
    G_kept: dict[str, torch.Tensor]
    G_deleted: dict[str, torch.Tensor]
    G_asset_per_object: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    per_gauss_pointers: GaussianPointers | None = None
    edit_meta: dict[str, Any] = field(default_factory=dict)


def _subset_gaussians_by_mask(
    clean: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "means": clean["means"][mask],
        "colors": clean["colors"][mask],
        "opacities": clean["opacities"][mask],
        "scales": clean["scales"][mask],
        "quats": clean["quats"][mask],
    }


def _predict_cameras_from_pose_enc(
    predictions: dict[str, torch.Tensor],
    image_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    height, width = image_hw
    pose_enc = predictions["pose_enc"]
    extrinsic_3x4, intrinsic = pose_encoding_to_extri_intri(pose_enc, (height, width))
    extrinsic_3x4 = extrinsic_3x4[0]
    intrinsic = intrinsic[0]
    bottom = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        dtype=extrinsic_3x4.dtype,
        device=extrinsic_3x4.device,
    ).view(1, 1, 4)
    world_to_camera = torch.cat(
        [extrinsic_3x4, bottom.expand(extrinsic_3x4.shape[0], -1, -1)], dim=1
    )
    return {"world_to_camera": world_to_camera, "intrinsics": intrinsic}


def _extract_waymo_cameras(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Flatten GT Waymo cameras to the same per-image layout as DGGT predictions."""
    camera_to_world = sample["camera_to_world_corrected"].detach().cpu().float()
    intrinsics = sample["intrinsics"].detach().cpu().float()
    num_frames = camera_to_world.shape[0]
    num_views = camera_to_world.shape[1] if camera_to_world.dim() == 4 else 1
    if camera_to_world.dim() == 4:
        camera_to_world = camera_to_world.reshape(num_frames * num_views, 4, 4)
        if intrinsics.dim() == 3 and intrinsics.shape[0] == num_views:
            intrinsics = intrinsics.unsqueeze(0).expand(num_frames, -1, -1, -1).reshape(
                num_frames * num_views, 3, 3
            )
        elif intrinsics.dim() == 4:
            intrinsics = intrinsics.reshape(num_frames * num_views, 3, 3)
    return {"camera_to_world": camera_to_world, "intrinsics": intrinsics}


def _extract_asset_gaussians(
    localized: list[LocalizedFrameObject],
) -> dict[int, dict[str, torch.Tensor]]:
    """Gather per-object asset gaussians (Waymo-world coords) from localization output."""
    per_object: dict[int, dict[str, torch.Tensor]] = {}
    for item in localized:
        if item.asset_means_world.numel() == 0:
            continue
        per_object[int(item.slot_idx)] = {
            "means": item.asset_means_world,
            "colors": item.asset_colors,
            "opacities": item.asset_opacities,
            "scales": item.asset_scales,
            "quats": item.asset_quats,
        }
    return per_object


class GaussianSceneEditor(nn.Module):
    """Reusable deterministic Mode-A editor.

    Parameters mirror the CLI defaults of ``inference_mode_a.py`` / the
    baseline ``localize_objects`` call so swapping the script over produces
    bit-identical output.
    """

    def __init__(
        self,
        min_match_score: float = 0.1,
        dynamic_thresh: float = 0.5,
        core_scale: float = 0.85,
        shell_scale: float = 1.05,
        proposal_scale: float = 1.25,
        motion_speed_thresh: float = 1.0,
        dynamic_prob_thresh: float = 0.55,
        dynamic_ratio_thresh: float = 0.35,
        use_pose_refine: bool = True,
        max_pose_refine_yaw_deg: float = 15.0,
        asset_yaw_correction_deg: float = 180.0,
    ) -> None:
        super().__init__()
        self.min_match_score = float(min_match_score)
        self.dynamic_thresh = float(dynamic_thresh)
        self.core_scale = float(core_scale)
        self.shell_scale = float(shell_scale)
        self.proposal_scale = float(proposal_scale)
        self.motion_speed_thresh = float(motion_speed_thresh)
        self.dynamic_prob_thresh = float(dynamic_prob_thresh)
        self.dynamic_ratio_thresh = float(dynamic_ratio_thresh)
        self.use_pose_refine = bool(use_pose_refine)
        self.max_pose_refine_yaw_deg = float(max_pose_refine_yaw_deg)
        self.asset_yaw_correction_deg = float(asset_yaw_correction_deg)

    def build_clean_bundle(
        self,
        sample: dict[str, Any],
        predictions: dict[str, torch.Tensor],
    ) -> CleanSceneState:
        return build_clean_scene_state(sample, predictions)

    def align(
        self,
        sample: dict[str, Any],
        clean_state: CleanSceneState,
    ) -> Sim3Transform:
        return estimate_scene_alignment(sample, clean_state)

    def localize(
        self,
        sample: dict[str, Any],
        clean_state: CleanSceneState,
        alignment: Sim3Transform,
        object_slots: list[int],
        asset_cache: dict[str, dict[str, torch.Tensor]] | None = None,
        load_asset: bool = True,
    ) -> list[LocalizedFrameObject]:
        return localize_objects(
            sample,
            clean_state,
            alignment,
            object_slots,
            min_match_score=self.min_match_score,
            dynamic_thresh=self.dynamic_thresh,
            core_scale=self.core_scale,
            shell_scale=self.shell_scale,
            proposal_scale=self.proposal_scale,
            motion_speed_thresh=self.motion_speed_thresh,
            dynamic_prob_thresh=self.dynamic_prob_thresh,
            dynamic_ratio_thresh=self.dynamic_ratio_thresh,
            use_pose_refine=self.use_pose_refine,
            max_pose_refine_yaw_deg=self.max_pose_refine_yaw_deg,
            asset_yaw_correction_deg=self.asset_yaw_correction_deg,
            asset_cache=asset_cache,
            load_asset=load_asset,
        )

    def apply_mode_a(
        self,
        clean_state: CleanSceneState,
        localized: list[LocalizedFrameObject],
    ) -> EditedSceneState:
        return apply_mode_a(clean_state, localized)

    def forward(
        self,
        sample: dict[str, Any],
        predictions: dict[str, torch.Tensor],
        asset_bank: Any | None = None,
        edit_instruction: dict[str, Any] | None = None,
    ) -> EditedSceneBundle:
        num_views = int(sample["cam_ids"].numel())
        if num_views != 1:
            raise NotImplementedError(
                f"GaussianSceneEditor currently supports views=1 only; got views={num_views}"
            )

        clean_state = self.build_clean_bundle(sample, predictions)
        alignment = self.align(sample, clean_state)

        if edit_instruction is not None and "object_slots" in edit_instruction:
            slots_spec = edit_instruction["object_slots"]
            if isinstance(slots_spec, str):
                object_slots = parse_object_slots(sample, slots_spec)
            else:
                object_slots = [int(v) for v in slots_spec]
        else:
            object_slots = parse_object_slots(sample, "all")

        asset_cache = None
        if asset_bank is not None and hasattr(asset_bank, "as_raw_cache"):
            asset_cache = asset_bank.as_raw_cache()
        elif isinstance(asset_bank, dict):
            asset_cache = asset_bank

        localized = self.localize(
            sample, clean_state, alignment, object_slots, asset_cache=asset_cache
        )
        edited_state = self.apply_mode_a(clean_state, localized)

        image_hw = (int(clean_state.images.shape[-2]), int(clean_state.images.shape[-1]))
        cameras_dggt = _predict_cameras_from_pose_enc(predictions, image_hw)
        cameras_waymo = _extract_waymo_cameras(sample)

        clean_gauss = edited_state.clean
        kept_mask = ~edited_state.delete_mask
        g_kept = _subset_gaussians_by_mask(clean_gauss, kept_mask)
        g_deleted = _subset_gaussians_by_mask(clean_gauss, edited_state.delete_mask)

        edit_meta = {
            "object_slots": list(object_slots),
            "delete_mask": edited_state.delete_mask,
            "shell_mask": edited_state.shell_mask,
            "localized_objects": edited_state.localized_objects,
            "num_views": num_views,
        }

        return EditedSceneBundle(
            clean_state=clean_state,
            alignment=alignment,
            edited_state=edited_state,
            cameras_dggt=cameras_dggt,
            cameras_waymo=cameras_waymo,
            T_w2d=alignment,
            G_kept=g_kept,
            G_deleted=g_deleted,
            G_asset_per_object=_extract_asset_gaussians(edited_state.localized_objects),
            per_gauss_pointers=None,  # populated in Phase 2 once FeatureSplatter lands
            edit_meta=edit_meta,
        )
