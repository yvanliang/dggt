"""Edit-coverage helpers shared by `inference_scene_editor.py`, the offline
feature precomputer, and the online `FlowFeatureAssembler`.

Two responsibilities:

1. `build_phase1_asset_coverage` — given Phase 1's `localized_objects`, build a
   `[num_slots, num_images]` boolean matrix mapping which (slot, image) pairs
   were actually deleted. Used to gate `AssetAggregatorPass` renders to match
   Phase 1 deletions exactly.

2. `resolve_editable_subset` — given per-frame `(bbox_present_mask, bbox_editable_mask)`
   for all frames of a clip and a selected subset of frames, produce the
   `editable_object_indices` and per-frame execution mask using the simplified
   rule `per_frame_edit_mask = present ∧ editable` (Phase 4.5 — anchor/follower
   deprecated).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch


def build_phase1_asset_coverage(
    image_valid_mask_selected: torch.Tensor,
    localized_objects: Sequence[object],
) -> tuple[torch.Tensor, list[int]]:
    """Boolean coverage of (slot, image) pairs actually deleted by Phase 1.

    Parameters
    ----------
    image_valid_mask_selected : [M, num_images] bool
        `sample["object_asset_image_valid_mask_selected"]`.
    localized_objects : iterable of LocalizedFrameObject
        Output of `GaussianSceneEditor.localize` / `apply_mode_a`.

    Returns
    -------
    coverage : [M, num_images] bool
    slot_ids : sorted list of slot indices that had >= 1 coverage cell.
    """
    num_slots, num_images = int(image_valid_mask_selected.shape[0]), int(image_valid_mask_selected.shape[1])
    coverage = torch.zeros((num_slots, num_images), dtype=torch.bool)
    slot_set: set[int] = set()
    for item in localized_objects:
        slot_idx = int(getattr(item, "slot_idx"))
        image_idx = int(getattr(item, "source_front_index"))
        if 0 <= slot_idx < num_slots and 0 <= image_idx < num_images:
            coverage[slot_idx, image_idx] = True
            slot_set.add(slot_idx)
    return coverage, sorted(slot_set)


def resolve_editable_subset(
    bbox_present_mask: torch.Tensor,
    bbox_editable_mask: torch.Tensor,
    subset_frames: Iterable[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the simplified per-frame rule on a selected frame subset.

    Parameters
    ----------
    bbox_present_mask, bbox_editable_mask : [M, N_all] bool
        Per-object per-frame flags over all cached frames (typically 29).
    subset_frames : iterable of int, length |subset|
        Frame indices (into the full N_all axis) the dataloader picked for this
        training step.

    Returns
    -------
    editable_object_indices : [|E|] long
        Sorted indices of editable objects (those with >= 1 True in the subset).
    per_frame_edit_mask : [M, |subset|] bool
        True iff object m is edited on frame f inside the subset, i.e.
        `bbox_present_mask[m, subset[f]] ∧ bbox_editable_mask[m, subset[f]]`.
    """
    if bbox_present_mask.shape != bbox_editable_mask.shape:
        raise ValueError(
            "bbox_present_mask / bbox_editable_mask shape mismatch: "
            f"{tuple(bbox_present_mask.shape)} vs {tuple(bbox_editable_mask.shape)}"
        )
    if not isinstance(subset_frames, torch.Tensor):
        subset_frames = torch.tensor(list(subset_frames), dtype=torch.long)
    subset_frames = subset_frames.to(torch.long)
    pres_sub = bbox_present_mask.index_select(1, subset_frames).bool()
    edit_sub = bbox_editable_mask.index_select(1, subset_frames).bool()
    per_frame_edit_mask = pres_sub & edit_sub
    any_edit = per_frame_edit_mask.any(dim=1)
    editable_object_indices = torch.nonzero(any_edit, as_tuple=False).flatten().to(torch.long)
    return editable_object_indices, per_frame_edit_mask
