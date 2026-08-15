"""Fold validation dumps into one legible mosaic per scene slot.

Validation used to emit one JPEG per ``(scene, CFG scale, quantity)``.  A
ten-scene / three-scale sweep produced 190 files and 190 W&B panels, and the
panel keys carried the *scene name*, so the panel set kept growing every time
the cyclic sampler rotated onto new scenes.

This module folds the same content into a single image per scene slot.  Rows
are grouped by **quantity** (RGB, sky mask, latent, latent error) and the CFG
scales sit adjacent *inside* a group, so comparing scales is a vertical glance
instead of a hunt across panels.  Nothing is concatenated horizontally: a row
already spans the clip's frames left to right.

Rows travel between ranks as base64 image bytes.  The CFG scales of one scene
are sampled on different ranks, so the rank that composes a mosaic is usually
not the rank that rendered any given row of it.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is a hard dependency in practice
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


# Width in pixels each *frame* occupies in the composed mosaic.  Ten frames at
# 256px give a 2560px-wide image: wide enough to judge a render, small enough
# that ten scenes per validation stay a few MB in W&B.
MOSAIC_CELL_WIDTH_DEFAULT = 256

# Rows that describe ground truth rather than one CFG scale sort above every
# scale inside their group.
GT_ROW_ORDER = -1

# Chrome.  Deliberately dark and low-chroma so the labels never compete with
# the imagery they annotate.
_BACKGROUND = (11, 13, 17)
_HEADER_BG = (46, 54, 66)
_HEADER_FG = (241, 245, 249)
_GROUP_BG = (28, 34, 43)
_GROUP_FG = (203, 213, 225)
_CAPTION_BG = (17, 21, 27)
_CAPTION_FG = (148, 163, 184)

_TEXT_PAD_X = 10
_GROUP_GAP = 8
# Frames are butted together inside a row; a thin gap in the background colour
# keeps the clip readable as ten frames rather than one smeared panorama.
_FRAME_GAP = 2

# Pillow's built-in font covers Latin-1 plus the middle dot and renders every
# other codepoint as a tofu box, so every label here stays inside that set.
# `_assert_label_charset` in the tests is what keeps it that way.
LABEL_CHARSET_EXTRA = "·"


@dataclass(frozen=True)
class MosaicGroup:
    """One quantity, shown once per CFG scale."""

    key: str
    title: str
    # Photographic rows are resampled smoothly and stored as JPEG; masks and
    # latent grids keep their hard patch edges and stay lossless.
    photographic: bool


MOSAIC_GROUPS: tuple[MosaicGroup, ...] = (
    MosaicGroup("rgb", "3DGS render · GT first, then one row per CFG scale", True),
    MosaicGroup(
        "sky",
        "sky mask · GT red · predicted green · agreement yellow",
        False,
    ),
    MosaicGroup(
        "latent",
        "scene latent · 3-component PCA · one basis fitted on GT, so colour is comparable",
        False,
    ),
    MosaicGroup(
        "latent_err",
        "latent abs(generated - GT) · fixed 0..1 scale, comparable across steps",
        False,
    ),
    MosaicGroup("sky_rgb", "generated sky dome", True),
)
_MOSAIC_GROUP_BY_KEY = {group.key: group for group in MOSAIC_GROUPS}
_MOSAIC_GROUP_ORDER = {group.key: index for index, group in enumerate(MOSAIC_GROUPS)}


def mosaic_group(key: str) -> MosaicGroup:
    """Return the group spec for ``key``, rejecting unknown groups loudly."""

    try:
        return _MOSAIC_GROUP_BY_KEY[str(key)]
    except KeyError:
        raise KeyError(
            f"unknown validation mosaic group {key!r}; "
            f"known groups: {sorted(_MOSAIC_GROUP_BY_KEY)}"
        ) from None


def _require_pillow() -> None:
    if Image is None:
        raise ModuleNotFoundError("Pillow is required to compose validation mosaics")


def _frames_to_strip(frames: torch.Tensor) -> "Image.Image":
    """Lay `[N,C,H,W]` in `[0,1]` out left to right as one RGB strip."""

    _require_pillow()
    if frames.ndim != 4:
        raise ValueError(f"Expected frames [N,C,H,W], got {tuple(frames.shape)}")
    strip = frames.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    count, channels, height, width = strip.shape
    if count <= 0:
        raise ValueError("Cannot build a mosaic row from zero frames")
    if channels == 1:
        strip = strip.expand(count, 3, height, width)
    elif channels != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {channels}")
    # [N,3,H,W] -> [3,H,N,W] -> [3,H,N*W] -> [H,N*W,3], frame-major along width.
    packed = (
        strip.permute(1, 2, 0, 3)
        .reshape(3, height, count * width)
        .permute(1, 2, 0)
        .contiguous()
    )
    array = packed.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8).numpy()
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _resize_strip(image: "Image.Image", width: int, *, smooth: bool) -> "Image.Image":
    """Rescale a strip to ``width``, preserving aspect ratio.

    ``smooth`` picks the resampling pair: photographic rows get area/Lanczos,
    while masks and latent grids stay blocky so a patch boundary in the mosaic
    is a patch boundary in the tensor.
    """

    width = int(width)
    if width <= 0:
        raise ValueError(f"mosaic width must be positive, got {width}")
    if image.width == width:
        return image
    height = max(1, round(image.height * width / image.width))
    if width > image.width:
        resample = Image.BICUBIC if smooth else Image.NEAREST
    else:
        resample = Image.LANCZOS if smooth else Image.BOX
    return image.resize((width, height), resample)


def encode_mosaic_row(
    frames: torch.Tensor,
    *,
    cell_width: int = MOSAIC_CELL_WIDTH_DEFAULT,
    photographic: bool,
) -> str:
    """Encode `[N,C,H,W]` as a base64 strip sized for the mosaic.

    The strip is downscaled to the mosaic's cell width here — before it crosses
    the rank boundary — but never upscaled, so a 37x25 latent grid stays 37x25
    on the wire and is enlarged only when it is pasted.
    """

    _require_pillow()
    strip = _frames_to_strip(frames)
    count = int(frames.shape[0])
    target = int(cell_width) * count
    if target < strip.width:
        strip = _resize_strip(strip, target, smooth=photographic)
    buffer = io.BytesIO()
    if photographic:
        strip.save(buffer, format="JPEG", quality=92, subsampling=0)
    else:
        strip.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_mosaic_row(payload: str) -> "Image.Image":
    _require_pillow()
    image = Image.open(io.BytesIO(base64.b64decode(payload)))
    return image.convert("RGB")


def _load_font(size: int):
    _require_pillow()
    try:
        return ImageFont.load_default(size=int(size))
    except TypeError:  # pragma: no cover - Pillow < 10.1 bitmap fallback
        return ImageFont.load_default()


def _draw_label(
    draw: "ImageDraw.ImageDraw",
    text: str,
    *,
    top: int,
    height: int,
    font,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    baseline = top + max(0, (height - (box[3] - box[1]))) // 2 - box[1]
    draw.text((_TEXT_PAD_X, baseline), text, fill=fill, font=font)


def sort_mosaic_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order rows by quantity, then by CFG scale within the quantity."""

    return sorted(
        rows,
        key=lambda row: (
            _MOSAIC_GROUP_ORDER[str(mosaic_group(row["group"]).key)],
            int(row["order"]),
        ),
    )


def _chrome_metrics(cell_width: int) -> dict[str, int]:
    """Size labels off one frame's width so any --val_mosaic_cell_width reads."""

    caption = max(12, round(int(cell_width) * 0.062))
    group = max(14, round(int(cell_width) * 0.076))
    header = max(16, round(int(cell_width) * 0.088))
    return {
        "caption_font": caption,
        "group_font": group,
        "header_font": header,
        "caption_height": caption + 10,
        "group_height": group + 14,
        "header_height": header + 16,
    }


def compose_scene_mosaic(
    rows: Sequence[dict[str, Any]],
    *,
    header: str,
    cell_width: int = MOSAIC_CELL_WIDTH_DEFAULT,
) -> "Image.Image":
    """Stack labelled rows into one image, grouped by quantity.

    Every row keeps its own frame count: a row is scaled so that one frame
    occupies ``cell_width`` pixels, then left-aligned on the canvas.
    """

    _require_pillow()
    if not rows:
        raise ValueError("Cannot compose a mosaic without rows")
    ordered = sort_mosaic_rows(rows)
    chrome = _chrome_metrics(cell_width)

    strips: list["Image.Image"] = []
    for row in ordered:
        group = mosaic_group(row["group"])
        strip = _decode_mosaic_row(row["png"])
        width = int(cell_width) * int(row["frames"])
        strips.append(_resize_strip(strip, width, smooth=group.photographic))

    canvas_width = max(strip.width for strip in strips)
    layout: list[tuple[str, Any]] = []
    total_height = chrome["header_height"]
    current_group: str | None = None
    for row, strip in zip(ordered, strips):
        if row["group"] != current_group:
            current_group = str(row["group"])
            if layout:
                total_height += _GROUP_GAP
                layout.append(("gap", None))
            layout.append(("group", mosaic_group(current_group).title))
            total_height += chrome["group_height"]
        layout.append(("caption", str(row["caption"])))
        total_height += chrome["caption_height"]
        layout.append(("strip", strip))
        total_height += strip.height

    canvas = Image.new("RGB", (canvas_width, total_height), color=_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(chrome["header_font"])
    group_font = _load_font(chrome["group_font"])
    caption_font = _load_font(chrome["caption_font"])

    draw.rectangle([0, 0, canvas_width, chrome["header_height"]], fill=_HEADER_BG)
    _draw_label(
        draw,
        header,
        top=0,
        height=chrome["header_height"],
        font=header_font,
        fill=_HEADER_FG,
    )

    cursor = chrome["header_height"]
    for kind, payload in layout:
        if kind == "gap":
            cursor += _GROUP_GAP
        elif kind == "group":
            height = chrome["group_height"]
            draw.rectangle([0, cursor, canvas_width, cursor + height], fill=_GROUP_BG)
            _draw_label(
                draw,
                str(payload),
                top=cursor,
                height=height,
                font=group_font,
                fill=_GROUP_FG,
            )
            cursor += height
        elif kind == "caption":
            height = chrome["caption_height"]
            draw.rectangle([0, cursor, canvas_width, cursor + height], fill=_CAPTION_BG)
            _draw_label(
                draw,
                str(payload),
                top=cursor,
                height=height,
                font=caption_font,
                fill=_CAPTION_FG,
            )
            cursor += height
        else:
            canvas.paste(payload, (0, cursor))
            _draw_frame_gaps(
                draw,
                top=cursor,
                height=payload.height,
                width=payload.width,
                cell_width=int(cell_width),
            )
            cursor += payload.height
    return canvas


def _draw_frame_gaps(
    draw: "ImageDraw.ImageDraw",
    *,
    top: int,
    height: int,
    width: int,
    cell_width: int,
) -> None:
    for x in range(cell_width, width, cell_width):
        draw.rectangle(
            [x - _FRAME_GAP // 2, top, x + (_FRAME_GAP - _FRAME_GAP // 2) - 1, top + height - 1],
            fill=_BACKGROUND,
        )


def build_validation_mosaics(
    rows: Sequence[dict[str, Any]],
    *,
    step: int,
    cell_width: int = MOSAIC_CELL_WIDTH_DEFAULT,
) -> dict[int, "Image.Image"]:
    """Compose one mosaic per scene *slot*.

    Keying by slot rather than by scene name is what keeps the W&B panel count
    fixed: the cyclic validation sampler rotates a different scene into each
    slot every round, and a name-keyed panel set would grow without bound.  The
    scene that filled the slot is named in the mosaic header instead.
    """

    by_slot: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_slot.setdefault(int(row["slot"]), []).append(row)

    mosaics: dict[int, "Image.Image"] = {}
    for slot, slot_rows in sorted(by_slot.items()):
        scenes = sorted({str(row["scene"]) for row in slot_rows})
        frames = max(int(row["frames"]) for row in slot_rows)
        header = (
            f"slot {slot} · scene {'/'.join(scenes)} · step {int(step):06d} · "
            f"{frames} frames, time runs left to right"
        )
        mosaics[slot] = compose_scene_mosaic(
            slot_rows, header=header, cell_width=cell_width
        )
    return mosaics
