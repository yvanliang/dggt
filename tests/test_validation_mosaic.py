"""The validation mosaic: one legible image per scene instead of dozens.

These tests pin the properties that make the mosaic readable — quantity-major
ordering so CFG scales sit adjacent, a fixed error scale so those adjacent rows
are comparable, ASCII-only labels because Pillow's built-in font renders
anything else as a tofu box — and the invariant that made the rewrite necessary
at all: the panel key must not carry the scene name.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from dggt.utils.validation_mosaic import (
    GT_ROW_ORDER,
    LABEL_CHARSET_EXTRA,
    MOSAIC_GROUPS,
    _chrome_metrics,
    build_validation_mosaics,
    compose_scene_mosaic,
    encode_mosaic_row,
    mosaic_group,
    sort_mosaic_rows,
)

FRAMES, GY, GX = 4, 5, 7


def _row(group: str, order: int, *, slot: int = 0, frames: int = FRAMES) -> dict:
    tensor = torch.rand((frames, 3, GY, GX))
    return {
        "slot": slot,
        "scene": "0137",
        "group": group,
        "order": order,
        "caption": "GT" if order == GT_ROW_ORDER else f"cfg {order + 1}",
        "frames": frames,
        "png": encode_mosaic_row(
            tensor, cell_width=16, photographic=mosaic_group(group).photographic
        ),
    }


def _decode(payload: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(payload)))


# ------------------------------------------------------------------ #
# Layout                                                              #
# ------------------------------------------------------------------ #
def test_rows_are_grouped_by_quantity_with_cfg_scales_adjacent() -> None:
    """Comparing CFG scales must be a vertical glance, not a hunt."""

    scrambled = [
        _row("latent_err", 2),
        _row("rgb", 1),
        _row("latent", GT_ROW_ORDER),
        _row("rgb", GT_ROW_ORDER),
        _row("latent_err", 0),
        _row("rgb", 0),
    ]
    ordered = [(row["group"], row["order"]) for row in sort_mosaic_rows(scrambled)]
    assert ordered == [
        ("rgb", GT_ROW_ORDER),
        ("rgb", 0),
        ("rgb", 1),
        ("latent", GT_ROW_ORDER),
        ("latent_err", 0),
        ("latent_err", 2),
    ]


def test_ground_truth_sorts_above_every_cfg_scale_in_its_group() -> None:
    assert GT_ROW_ORDER < 0
    ordered = sort_mosaic_rows([_row("rgb", 0), _row("rgb", GT_ROW_ORDER)])
    assert ordered[0]["order"] == GT_ROW_ORDER


def test_mosaic_never_grows_sideways_with_the_cfg_count() -> None:
    """Ten frames already span the width; scales must stack, never concatenate."""

    one_scale = compose_scene_mosaic(
        [_row("rgb", 0)], header="one", cell_width=16
    )
    three_scales = compose_scene_mosaic(
        [_row("rgb", 0), _row("rgb", 1), _row("rgb", 2)], header="three", cell_width=16
    )
    assert three_scales.width == one_scale.width == 16 * FRAMES
    assert three_scales.height > one_scale.height


def test_a_short_row_keeps_its_frame_pitch_instead_of_stretching() -> None:
    """A row with fewer frames stays frame-aligned with the rows above it.

    Stretching it to the canvas width would silently misalign frame k of the
    latent row with frame k of the render above, which is the whole point of
    stacking them.
    """

    short = _row("latent", 0, frames=FRAMES - 1)
    mosaic = compose_scene_mosaic([_row("rgb", 0), short], header="mixed", cell_width=16)
    assert mosaic.width == 16 * FRAMES
    # The last cell of the short row is left unpainted rather than filled.
    last = np.asarray(mosaic.crop((16 * (FRAMES - 1), 0, 16 * FRAMES, mosaic.height)))
    first = np.asarray(mosaic.crop((0, 0, 16, mosaic.height)))
    assert len(np.unique(last.reshape(-1, 3), axis=0)) < len(
        np.unique(first.reshape(-1, 3), axis=0)
    )


def test_unknown_group_is_rejected_rather_than_silently_dropped() -> None:
    with pytest.raises(KeyError, match="unknown validation mosaic group"):
        mosaic_group("not_a_group")


# ------------------------------------------------------------------ #
# Encoding                                                            #
# ------------------------------------------------------------------ #
def test_low_resolution_rows_are_not_upscaled_before_they_cross_ranks() -> None:
    """A 7x5 latent grid must ride the gather at 7x5, not at mosaic scale."""

    payload = encode_mosaic_row(
        torch.rand((FRAMES, 3, GY, GX)), cell_width=256, photographic=False
    )
    assert _decode(payload).size == (GX * FRAMES, GY)


def test_photographic_rows_are_downscaled_to_the_mosaic_cell_width() -> None:
    payload = encode_mosaic_row(
        torch.rand((FRAMES, 3, 350, 518)), cell_width=64, photographic=True
    )
    assert _decode(payload).size[0] == 64 * FRAMES


def test_masks_stay_lossless_so_a_patch_edge_survives_the_gather() -> None:
    mask = torch.zeros((1, 1, 4, 4))
    mask[0, 0, :, 2:] = 1.0
    decoded = _decode(
        encode_mosaic_row(mask, cell_width=256, photographic=False)
    ).convert("L")
    assert sorted(np.unique(np.asarray(decoded)).tolist()) == [0, 255]


def test_single_channel_rows_are_promoted_to_rgb() -> None:
    payload = encode_mosaic_row(
        torch.rand((FRAMES, 1, GY, GX)), cell_width=16, photographic=False
    )
    assert _decode(payload).convert("RGB").size == (GX * FRAMES, GY)


def test_rejects_tensors_that_are_not_frame_stacks() -> None:
    with pytest.raises(ValueError, match=r"\[N,C,H,W\]"):
        encode_mosaic_row(torch.rand((3, GY, GX)), cell_width=16, photographic=False)
    with pytest.raises(ValueError, match="1 or 3 channels"):
        encode_mosaic_row(
            torch.rand((FRAMES, 2, GY, GX)), cell_width=16, photographic=False
        )


# ------------------------------------------------------------------ #
# Labels                                                              #
# ------------------------------------------------------------------ #
def test_group_titles_stay_inside_the_default_font_charset() -> None:
    """Pillow's built-in font draws an em dash or an arrow as a tofu box."""

    allowed = set(LABEL_CHARSET_EXTRA)
    for group in MOSAIC_GROUPS:
        for char in group.title:
            assert char in allowed or ord(char) < 128, (
                f"group {group.key!r} title contains {char!r}, which the "
                "built-in font cannot draw"
            )


def test_labels_scale_with_the_configured_cell_width() -> None:
    """Captions must stay readable at any --val_mosaic_cell_width."""

    small = _chrome_metrics(64)
    large = _chrome_metrics(512)
    assert large["caption_font"] > small["caption_font"]
    assert large["header_font"] > large["group_font"] > large["caption_font"]
    for chrome in (small, large):
        # Every band must clear its own glyphs.
        assert chrome["caption_height"] > chrome["caption_font"]
        assert chrome["group_height"] > chrome["group_font"]
        assert chrome["header_height"] > chrome["header_font"]
    # A tiny cell width still yields legible text rather than a hairline.
    assert _chrome_metrics(8)["caption_font"] >= 12


# ------------------------------------------------------------------ #
# Slot keying                                                         #
# ------------------------------------------------------------------ #
def test_one_mosaic_per_slot_regardless_of_how_many_scenes_rotate_through() -> None:
    """Keying by slot is what bounds the W&B panel count.

    The cyclic sampler puts a different scene in each slot every validation, so
    a scene-name-keyed panel set would grow without bound over a long run.
    """

    rows = [
        _row("rgb", 0, slot=0),
        _row("rgb", 1, slot=0),
        _row("rgb", 0, slot=1),
    ]
    mosaics = build_validation_mosaics(rows, step=4000, cell_width=16)
    assert sorted(mosaics) == [0, 1]


def test_the_scene_name_lands_in_the_header_not_in_the_key() -> None:
    rows = [_row("rgb", 0, slot=3)]
    rows[0]["scene"] = "0421"
    mosaics = build_validation_mosaics(rows, step=4000, cell_width=16)
    assert list(mosaics) == [3]
    # The header band is the top strip; it is the only place the name appears.
    assert mosaics[3].width == 16 * FRAMES


def test_composing_without_rows_is_an_error_not_a_blank_image() -> None:
    with pytest.raises(ValueError, match="without rows"):
        compose_scene_mosaic([], header="empty", cell_width=16)
    assert build_validation_mosaics([], step=0, cell_width=16) == {}
