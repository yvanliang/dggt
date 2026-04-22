from __future__ import annotations

import torch

from datasets.tools.build_edit_metadata import (
    MIN_EDIT_BOX_SIZE_PX,
    box_overlap_ratios_xyxy,
    is_transfer_box_large_enough,
    resolve_occluded_slots,
)
from datasets.waymo_edit_dataset import WaymoEditDataset


def test_is_transfer_box_large_enough_uses_transfer_long_edge_threshold():
    assert MIN_EDIT_BOX_SIZE_PX == 128.0
    assert not is_transfer_box_large_enough([0.0, 0.0, 127.0, 200.0])
    assert not is_transfer_box_large_enough([0.0, 0.0, 220.0, 127.0])
    assert is_transfer_box_large_enough([0.0, 0.0, 128.0, 128.0])


def test_resolve_occluded_slots_keeps_only_frontmost_in_high_iou_cluster():
    frame_boxes = {
        0: [10.0, 10.0, 210.0, 210.0],
        1: [12.0, 12.0, 208.0, 208.0],
        2: [14.0, 14.0, 206.0, 206.0],
    }
    frame_depths = {
        0: 18.0,
        1: 14.0,
        2: 10.0,
    }

    occluded_slots, high_iou_pairs = resolve_occluded_slots(frame_boxes, frame_depths)

    assert high_iou_pairs == 3
    assert occluded_slots == {0, 1}


def test_resolve_occluded_slots_skips_missing_depth_and_low_iou_pairs():
    frame_boxes = {
        0: [10.0, 10.0, 210.0, 210.0],
        1: [12.0, 12.0, 208.0, 208.0],
        2: [260.0, 260.0, 320.0, 320.0],
    }
    frame_depths = {
        0: 12.0,
    }

    occluded_slots, high_iou_pairs = resolve_occluded_slots(frame_boxes, frame_depths)

    assert high_iou_pairs == 1
    assert occluded_slots == set()


def test_box_overlap_ratios_detects_full_containment_of_smaller_box():
    overlap_a, overlap_b = box_overlap_ratios_xyxy(
        [860.0, 220.0, 1200.0, 590.0],
        [925.0, 365.0, 1058.0, 490.0],
    )

    assert overlap_a < 0.2
    assert overlap_b == 1.0


def test_build_editable_object_indices_by_frame_pads_with_negative_one():
    dataset = object.__new__(WaymoEditDataset)
    dataset.max_objects = 4

    frame_editable_object_mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [True, True, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    padded = dataset._build_editable_object_indices_by_frame(frame_editable_object_mask)

    assert padded.tolist() == [
        [0, 2, -1, -1],
        [1, 2, -1, -1],
        [-1, -1, -1, -1],
    ]


def test_resolve_object_bbox_editable_view_flags_reads_bbox_editable_metadata():
    dataset = object.__new__(WaymoEditDataset)

    flags = dataset._resolve_object_bbox_editable_view_flags(
        {
            "bbox_editable_by_view": {"pinhole_front": [True, False, True]},
            "bbox_present_by_view": {"pinhole_front": [True, True, True]},
        },
        "pinhole_front",
        expected_length=3,
    )

    assert flags.tolist() == [True, False, True]
