#!/usr/bin/env python
"""Frozen on-disk contract for the ChoraGen HD-map (static layout) condition.

This module is the single source of truth shared by every producer and consumer
of ``<split>_hdmap``:

* ``datasets/tools/build_hdmap_from_tfrecord.py`` -- writes geometry and all
  attributes together from frame 0 of an original Waymo v1.4.x tfrecord.
* ``datasets/tools/build_hdmap_dataset.py`` -- reads RDS only as the T16
  geometry cross-validation reference.
* ``datasets/dataset.py`` -- reads the ``.npz`` at training / inference time.
* ``tests/test_hdmap_schema.py`` -- pins every constant below.

Why two files per scene
-----------------------
``hdmap.json`` is the human-readable source of truth.  It deliberately keeps the
RDS-HQ nesting (``labels[i].labelData.shape3d.polyline3d.vertices``) so that a
reader written against the Cosmos-Drive-Dreams dump still works, and *adds*
sibling keys (``id``, ``class``, ``attributes``) rather than restructuring.

``hdmap.npz`` is the same content in CSR-packed arrays.  The dataloader must
never parse JSON on the hot path.  ``build_hdmap_dataset.py`` writes both from
one in-memory object, and ``tests/test_hdmap_schema.py`` asserts that the JSON
and the NPZ round-trip to an identical object, so the two cannot drift.

Coordinate frame
----------------
Vertices are stored **exactly** as Waymo publishes them: the per-segment world
frame used by ``map_features``, which is the same frame as the processed
``ego_pose/*.txt``.  No offset, no resampling, no unit change.  This was
verified against the raw proto: for segment ``10017090168044687777_...`` all
73 lanes / 7 road lines / 31 road edges / 2 crosswalks matched the tfrecord
polylines to ``atol=1e-6`` *and* in the same index order.

Numerics
--------
Coordinates are stored as float64 and must stay float64 until the anchor origin
has been subtracted.  Waymo world coordinates reach ~1e4 m; casting to float32
first costs ~1 mm and, worse, silently varies with map origin.  The contract is:
``(vertices - anchor_origin).astype(np.float32)``, never the reverse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = "waymo_hdmap_v1"

# ---------------------------------------------------------------------------
# Class enum.  APPEND ONLY.  The integer values are baked into every cached
# raster and every trained checkpoint's channel order; reordering them silently
# relabels the condition.
# ---------------------------------------------------------------------------
CLASS_NAMES: tuple[str, ...] = (
    "lane",  # 0  lane centerline        (polyline)
    "road_line",  # 1  painted lane divider   (polyline)
    "road_edge",  # 2  road boundary / median (polyline)
    "crosswalk",  # 3                         (polygon)
    "speed_bump",  # 4                         (polygon)
    "stop_sign",  # 5                         (point)
    "driveway",  # 6                         (polygon)
)
CLASS_IDS: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Geometry kind per feature.  Stored explicitly instead of being derived from
# the class so that a future class can change kind without breaking readers.
GEOMETRY_NAMES: tuple[str, ...] = ("polyline", "polygon", "point")
GEOMETRY_IDS: dict[str, int] = {name: idx for idx, name in enumerate(GEOMETRY_NAMES)}

CLASS_GEOMETRY: dict[str, str] = {
    "lane": "polyline",
    "road_line": "polyline",
    "road_edge": "polyline",
    "crosswalk": "polygon",
    "speed_bump": "polygon",
    "stop_sign": "point",
    "driveway": "polygon",
}

# The RDS-HQ dump only carries these four, under these directory names.
RDS_CLASS_DIRS: dict[str, str] = {
    "lane": "3d_lanes",
    "road_line": "3d_lanelines",
    "road_edge": "3d_road_boundaries",
    "crosswalk": "3d_crosswalks",
}
RDS_FILE_SUFFIX: dict[str, str] = {
    "lane": "lanes.json",
    "road_line": "lanelines.json",
    "road_edge": "road_boundaries.json",
    "crosswalk": "crosswalks.json",
}
# Classes the RDS converter dropped entirely; only the authoritative tfrecord
# source can supply them.
RDS_MISSING_CLASSES: tuple[str, ...] = ("speed_bump", "stop_sign", "driveway")

# ---------------------------------------------------------------------------
# Sub-type enums, mirrored verbatim from waymo_open_dataset/protos/map.proto so
# that training does not need the ``waymo_open_dataset`` package installed.
# ``tests/test_hdmap_schema.py::test_subtype_tables_match_waymo_proto`` compares
# these against the real proto whenever it is importable.
# ---------------------------------------------------------------------------
SUBTYPE_TABLES: dict[str, tuple[str, ...]] = {
    "lane": (
        "TYPE_UNDEFINED",
        "TYPE_FREEWAY",
        "TYPE_SURFACE_STREET",
        "TYPE_BIKE_LANE",
    ),
    "road_line": (
        "TYPE_UNKNOWN",
        "TYPE_BROKEN_SINGLE_WHITE",
        "TYPE_SOLID_SINGLE_WHITE",
        "TYPE_SOLID_DOUBLE_WHITE",
        "TYPE_BROKEN_SINGLE_YELLOW",
        "TYPE_BROKEN_DOUBLE_YELLOW",
        "TYPE_SOLID_SINGLE_YELLOW",
        "TYPE_SOLID_DOUBLE_YELLOW",
        "TYPE_PASSING_DOUBLE_YELLOW",
    ),
    "road_edge": (
        "TYPE_UNKNOWN",
        "TYPE_ROAD_EDGE_BOUNDARY",
        "TYPE_ROAD_EDGE_MEDIAN",
    ),
}
# Classes with no sub-type in the proto.
SUBTYPE_UNKNOWN = -1

ATTRIBUTE_SOURCE_NONE = "none"
ATTRIBUTE_SOURCE_TFRECORD = "waymo_tfrecord_frame0"
GEOMETRY_SOURCE_RDS = "waymo_rds"
GEOMETRY_SOURCE_TFRECORD = "waymo_tfrecord_frame0"

# A feature id is only meaningful when it came from the proto.  RDS carries no
# ids, so geometry-only features get this sentinel.
FEATURE_ID_UNKNOWN = -1


# ---------------------------------------------------------------------------
# Frozen layout-raster contract.  Consumers must compare RASTER_SCHEMA_HASH
# before dequantizing a cached/batched raster.  The tuple index is the channel
# index and every quantization row is (scale, zero_point, clip_lo, clip_hi).
# ---------------------------------------------------------------------------
RASTER_CHANNEL_NAMES: tuple[str, ...] = (
    "reserved_zero_lane_coverage",
    "road_line_coverage",
    "road_edge_coverage",
    "crosswalk_coverage",
    "speed_bump_coverage",
    "stop_sign_coverage",
    "reserved_zero_driveway_coverage",
    "reserved_zero_lane_distance",
    "road_line_distance",
    "road_edge_distance",
    "static_tangent_x",
    "static_tangent_y",
    "static_subpatch_du",
    "static_subpatch_dv",
    "attributes_known",
    "reserved_zero_lane_type_freeway",
    "reserved_zero_lane_type_surface_street",
    "reserved_zero_lane_type_bike_lane",
    "road_line_is_solid",
    "road_line_is_yellow",
    "road_edge_is_median",
    "static_valid",
    "vehicle_coverage",
    "pedestrian_coverage",
    "cyclist_coverage",
    "actor_edge_distance",
    "actor_subpatch_du",
    "actor_subpatch_dv",
    "actor_velocity_direction_x",
    "actor_velocity_direction_y",
    "actor_is_moving",
    "actor_count_normalized",
    "actor_valid",
)
RASTER_CHANNEL_COUNT = len(RASTER_CHANNEL_NAMES)
if RASTER_CHANNEL_COUNT != 33:  # pragma: no cover - import-time contract guard
    raise RuntimeError(f"layout raster must have 33 channels, got {RASTER_CHANNEL_COUNT}")

# Waymo ``LaneCenter`` polylines are lane reference centerlines, not visible
# road markings or boundaries.  ``driveway`` polygons are amodal access areas
# whose occlusion noise outweighs their control value.  Both remain in the
# camera-independent vector sidecar for source fidelity, but every online
# model/render condition excludes them.  Their former ABI slots stay reserved
# and must be exactly zero so that the remaining channel indices do not shift.
RASTER_SOURCE_EXCLUDED_CLASSES: tuple[str, ...] = ("lane", "driveway")
RASTER_RESERVED_ZERO_CHANNELS: tuple[int, ...] = (0, 6, 7, 15, 16, 17)
MAP_METRIC_RESERVED_ZERO_GROUPS: tuple[int, ...] = (0,)

# The static-map / actor split of the channel axis.  The model's two early
# stems, the negative-CFG branch and every window neutralization slice on this
# boundary, so it is derived from the channel table once here rather than
# copied as a literal into each of those call sites.
RASTER_MAP_CHANNELS: tuple[int, int] = (0, RASTER_CHANNEL_NAMES.index("vehicle_coverage"))
RASTER_ACTOR_CHANNELS: tuple[int, int] = (RASTER_MAP_CHANNELS[1], RASTER_CHANNEL_COUNT)
# The channel a window's factual static-map support is read from.
RASTER_STATIC_VALID_CHANNEL: int = RASTER_CHANNEL_NAMES.index("static_valid")
if (  # pragma: no cover - import-time contract guard
    RASTER_MAP_CHANNELS != (0, 22) or RASTER_ACTOR_CHANNELS != (22, 33)
):
    raise RuntimeError(
        "layout raster map/actor channel split drifted from the frozen ABI: "
        f"{RASTER_MAP_CHANNELS} / {RASTER_ACTOR_CHANNELS}"
    )

RASTER_QUANTIZATION_FIELDS: tuple[str, ...] = (
    "scale",
    "zero_point",
    "clip_lo",
    "clip_hi",
)
_UNSIGNED_QUANTIZATION = (1.0 / 255.0, 0, 0.0, 1.0)
_SIGNED_QUANTIZATION = (2.0 / 254.0, 127, -1.0, 1.0)
RASTER_SIGNED_CHANNELS: tuple[int, ...] = (10, 11, 28, 29)
RASTER_FLAG_CHANNELS: tuple[int, ...] = (14, 15, 16, 17, 18, 19, 20, 21, 30, 32)
RASTER_QUANTIZATION: tuple[tuple[float, int, float, float], ...] = tuple(
    _SIGNED_QUANTIZATION if channel in RASTER_SIGNED_CHANNELS else _UNSIGNED_QUANTIZATION
    for channel in range(RASTER_CHANNEL_COUNT)
)

# This is a frozen schema identifier, not a content digest.  Bump it explicitly
# only when the channel semantics or quantization contract changes.
RASTER_SCHEMA_HASH = "layout_raster_v3_33ch_no_lane_centerline_no_driveway_far120m"
# Batch/config metadata uses the lower-case spelling from the design contract.
raster_schema_hash = RASTER_SCHEMA_HASH


@dataclass
class HDMapFeature:
    """One map primitive in the segment's Waymo world frame."""

    cls: str
    vertices: np.ndarray  # [N,3] float64
    feature_id: int = FEATURE_ID_UNKNOWN
    subtype: int = SUBTYPE_UNKNOWN
    speed_limit_mph: float = float("nan")
    interpolating: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cls not in CLASS_IDS:
            raise ValueError(f"unknown hdmap class {self.cls!r}; expected one of {CLASS_NAMES}")
        vertices = np.asarray(self.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"{self.cls} vertices must be [N,3], got {tuple(vertices.shape)}")
        if vertices.shape[0] < 1:
            raise ValueError(f"{self.cls} feature has no vertices")
        if not np.isfinite(vertices).all():
            raise ValueError(f"{self.cls} feature has non-finite vertices")
        geometry = CLASS_GEOMETRY[self.cls]
        if geometry == "point" and vertices.shape[0] != 1:
            raise ValueError(f"{self.cls} is a point feature but has {vertices.shape[0]} vertices")
        # Waymo really does publish single-vertex lane stubs (segment
        # 10017090168044687777_... has two), so a >=2 minimum would silently
        # delete real map features during migration.  Fidelity is the schema's
        # job; the rasterizer decides what a degenerate primitive draws.
        self.vertices = vertices
        table = SUBTYPE_TABLES.get(self.cls)
        if self.subtype != SUBTYPE_UNKNOWN:
            if table is None:
                raise ValueError(f"class {self.cls} has no sub-type table but subtype={self.subtype}")
            if not 0 <= int(self.subtype) < len(table):
                raise ValueError(f"{self.cls} subtype {self.subtype} out of range for {table}")
        self.subtype = int(self.subtype)
        self.feature_id = int(self.feature_id)

    @property
    def geometry(self) -> str:
        return CLASS_GEOMETRY[self.cls]

    @property
    def is_degenerate(self) -> bool:
        """A polyline/polygon with too few vertices to have extent."""
        if self.geometry == "point":
            return False
        if self.geometry == "polygon":
            return self.vertices.shape[0] < 3
        return self.vertices.shape[0] < 2

    @property
    def has_attributes(self) -> bool:
        """True when this feature came directly from a proto MapFeature."""
        return self.feature_id != FEATURE_ID_UNKNOWN

    def subtype_name(self) -> str | None:
        table = SUBTYPE_TABLES.get(self.cls)
        if table is None or self.subtype == SUBTYPE_UNKNOWN:
            return None
        return table[self.subtype]


@dataclass
class HDMapScene:
    """Every static map primitive of one Waymo segment."""

    segment: str
    scene_id: str
    split: str
    features: list[HDMapFeature]
    geometry_source: str = GEOMETRY_SOURCE_RDS
    attribute_source: str = ATTRIBUTE_SOURCE_NONE
    map_pose_offset: np.ndarray | Sequence[Sequence[float]] | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> "HDMapScene":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")
        if not self.segment:
            raise ValueError("segment must be non-empty")
        if len(self.scene_id) != 3 or not self.scene_id.isdigit():
            raise ValueError(f"scene_id must be a 3-digit string, got {self.scene_id!r}")
        if self.split not in {"training", "validation"}:
            raise ValueError(f"unexpected split {self.split!r}")
        if self.attribute_source not in {ATTRIBUTE_SOURCE_NONE, ATTRIBUTE_SOURCE_TFRECORD}:
            raise ValueError(f"unexpected attribute_source {self.attribute_source!r}")
        if self.map_pose_offset is not None:
            map_pose_offset = np.asarray(self.map_pose_offset, dtype=np.float64)
            if (
                map_pose_offset.ndim != 2
                or map_pose_offset.shape[1] != 3
                or map_pose_offset.shape[0] < 1
            ):
                raise ValueError(
                    "map_pose_offset must be [S,3] with S >= 1, got "
                    f"{tuple(map_pose_offset.shape)}"
                )
            if not np.isfinite(map_pose_offset).all():
                raise ValueError("map_pose_offset contains non-finite values")
            self.map_pose_offset = map_pose_offset
        if self.attribute_source == ATTRIBUTE_SOURCE_NONE:
            # Fail closed: a geometry-only scene must not claim ids or sub-types,
            # otherwise a downstream raster would silently use garbage channels.
            for feature in self.features:
                if feature.has_attributes or feature.subtype != SUBTYPE_UNKNOWN:
                    raise ValueError(
                        "attribute_source=none but a feature carries id/subtype; "
                        "the tfrecord producer must set attribute_source"
                    )
                if feature.cls in RDS_MISSING_CLASSES:
                    raise ValueError(
                        f"class {feature.cls} cannot exist without tfrecord attributes "
                        "(the RDS converter never emitted it)"
                    )
        else:
            ids = [f.feature_id for f in self.features]
            if any(i == FEATURE_ID_UNKNOWN for i in ids):
                raise ValueError("tfrecord scene has a feature without a proto id")
            if len(ids) != len(set(ids)):
                raise ValueError("tfrecord scene has duplicate proto feature ids")
        for feature in self.features:
            feature.__post_init__()
        return self

    def counts(self) -> dict[str, int]:
        out = {name: 0 for name in CLASS_NAMES}
        for feature in self.features:
            out[feature.cls] += 1
        return out

    def by_class(self, cls: str) -> list[HDMapFeature]:
        return [f for f in self.features if f.cls == cls]


# ---------------------------------------------------------------------------
# JSON  (source of truth, RDS-compatible nesting)
# ---------------------------------------------------------------------------

def _shape3d(feature: HDMapFeature) -> dict[str, Any]:
    vertices = [[float(x) for x in row] for row in feature.vertices]
    if feature.geometry == "polygon":
        return {"surface": {"vertices": vertices}}
    if feature.geometry == "point":
        return {"point3d": {"vertices": vertices}}
    return {"polyline3d": {"vertices": vertices}}


def scene_to_json_obj(scene: HDMapScene) -> dict[str, Any]:
    scene.validate()
    classes: dict[str, Any] = {}
    for name in CLASS_NAMES:
        labels = []
        for feature in scene.by_class(name):
            label: dict[str, Any] = {
                # RDS-compatible payload first, so an RDS reader still works.
                "labelData": {"shape3d": _shape3d(feature)},
                # Supplemented fields.
                "id": feature.feature_id,
                "class": feature.cls,
                "geometry": feature.geometry,
                "subtype": feature.subtype,
                "subtype_name": feature.subtype_name(),
                "attributes": feature.attributes,
            }
            if feature.cls == "lane":
                label["speed_limit_mph"] = (
                    None if np.isnan(feature.speed_limit_mph) else float(feature.speed_limit_mph)
                )
                label["interpolating"] = bool(feature.interpolating)
            labels.append(label)
        classes[name] = {"labels": labels}
    return {
        "schema_version": scene.schema_version,
        "segment": scene.segment,
        "scene_id": scene.scene_id,
        "split": scene.split,
        "coordinate_frame": "waymo_segment_world",
        "geometry_source": scene.geometry_source,
        "attribute_source": scene.attribute_source,
        "map_pose_offset": (
            None
            if scene.map_pose_offset is None
            else np.asarray(scene.map_pose_offset, dtype=np.float64).tolist()
        ),
        "counts": scene.counts(),
        "classes": classes,
    }


def json_obj_to_scene(obj: dict[str, Any]) -> HDMapScene:
    features: list[HDMapFeature] = []
    for name in CLASS_NAMES:
        block = obj.get("classes", {}).get(name)
        if not block:
            continue
        for label in block.get("labels", []):
            shape = label["labelData"]["shape3d"]
            payload = shape.get("polyline3d") or shape.get("surface") or shape.get("point3d")
            if payload is None:
                raise ValueError(f"{name} label has no recognised shape3d payload")
            speed = label.get("speed_limit_mph")
            features.append(
                HDMapFeature(
                    cls=name,
                    vertices=np.asarray(payload["vertices"], dtype=np.float64),
                    feature_id=int(label.get("id", FEATURE_ID_UNKNOWN)),
                    subtype=int(label.get("subtype", SUBTYPE_UNKNOWN)),
                    speed_limit_mph=float("nan") if speed is None else float(speed),
                    interpolating=bool(label.get("interpolating", False)),
                    attributes=dict(label.get("attributes", {})),
                )
            )
    offset = obj.get("map_pose_offset")
    return HDMapScene(
        segment=str(obj["segment"]),
        scene_id=str(obj["scene_id"]),
        split=str(obj["split"]),
        features=features,
        geometry_source=str(obj.get("geometry_source", GEOMETRY_SOURCE_RDS)),
        attribute_source=str(obj.get("attribute_source", ATTRIBUTE_SOURCE_NONE)),
        map_pose_offset=None if offset is None else np.asarray(offset, dtype=np.float64),
        schema_version=str(obj.get("schema_version", SCHEMA_VERSION)),
    ).validate()


# ---------------------------------------------------------------------------
# NPZ  (hot path)
# ---------------------------------------------------------------------------

def scene_to_npz_arrays(scene: HDMapScene) -> dict[str, np.ndarray]:
    scene.validate()
    if scene.features:
        vertices = np.concatenate([f.vertices for f in scene.features], axis=0)
        lengths = np.asarray([len(f.vertices) for f in scene.features], dtype=np.int64)
    else:
        vertices = np.zeros((0, 3), dtype=np.float64)
        lengths = np.zeros((0,), dtype=np.int64)
    offsets = np.zeros((len(scene.features) + 1,), dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return {
        "vertices": vertices.astype(np.float64),
        "feature_offsets": offsets,
        "feature_class": np.asarray([CLASS_IDS[f.cls] for f in scene.features], dtype=np.int8),
        "feature_geometry": np.asarray(
            [GEOMETRY_IDS[f.geometry] for f in scene.features], dtype=np.int8
        ),
        "feature_id": np.asarray([f.feature_id for f in scene.features], dtype=np.int64),
        "feature_subtype": np.asarray([f.subtype for f in scene.features], dtype=np.int8),
        "feature_speed_limit_mph": np.asarray(
            [f.speed_limit_mph for f in scene.features], dtype=np.float32
        ),
        "feature_interpolating": np.asarray(
            [bool(f.interpolating) for f in scene.features], dtype=np.bool_
        ),
        # A scalar unicode JSON blob remains compatible with allow_pickle=False
        # while preserving every topology/boundary/neighbour attribute field.
        "feature_attributes_json": np.asarray(
            json.dumps(
                [f.attributes for f in scene.features],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
        "map_pose_offset": (
            np.zeros((0, 3), dtype=np.float64)
            if scene.map_pose_offset is None
            else np.asarray(scene.map_pose_offset, dtype=np.float64)
        ),
        "meta": np.asarray(
            json.dumps(
                {
                    "schema_version": scene.schema_version,
                    "segment": scene.segment,
                    "scene_id": scene.scene_id,
                    "split": scene.split,
                    "geometry_source": scene.geometry_source,
                    "attribute_source": scene.attribute_source,
                    "map_pose_offset": (
                        None
                        if scene.map_pose_offset is None
                        else np.asarray(scene.map_pose_offset, dtype=np.float64).tolist()
                    ),
                    "raster_schema_hash": RASTER_SCHEMA_HASH,
                    "class_names": list(CLASS_NAMES),
                    "geometry_names": list(GEOMETRY_NAMES),
                    "subtype_tables": {k: list(v) for k, v in SUBTYPE_TABLES.items()},
                },
                sort_keys=True,
            )
        ),
    }


def npz_arrays_to_scene(arrays: Any) -> HDMapScene:
    meta = json.loads(str(arrays["meta"]))
    if meta["class_names"] != list(CLASS_NAMES):
        raise ValueError(
            "hdmap npz was written with a different class enum: "
            f"{meta['class_names']} != {list(CLASS_NAMES)}"
        )
    vertices = np.asarray(arrays["vertices"], dtype=np.float64)
    offsets = np.asarray(arrays["feature_offsets"], dtype=np.int64)
    classes = np.asarray(arrays["feature_class"], dtype=np.int8)
    geometry = np.asarray(arrays["feature_geometry"], dtype=np.int8)
    ids = np.asarray(arrays["feature_id"], dtype=np.int64)
    subtypes = np.asarray(arrays["feature_subtype"], dtype=np.int8)
    speeds = np.asarray(arrays["feature_speed_limit_mph"], dtype=np.float32)
    interpolating = np.asarray(arrays["feature_interpolating"], dtype=np.bool_)
    attributes = json.loads(str(arrays["feature_attributes_json"]))
    count = int(classes.shape[0])
    if offsets.shape[0] != count + 1:
        raise ValueError("feature_offsets must have one more entry than feature_class")
    if count and int(offsets[-1]) != int(vertices.shape[0]):
        raise ValueError("feature_offsets do not cover the vertex buffer exactly")
    if not isinstance(attributes, list) or len(attributes) != count:
        raise ValueError("feature_attributes_json must contain one object per feature")
    # ``hdmap.npz`` stores camera-independent vector geometry.  Its historical
    # raster hash is producer provenance, not the runtime raster contract: the
    # current online projector advertises and validates RASTER_SCHEMA_HASH on
    # every emitted batch.  This decoupling lets a projection-only contract
    # change reuse the exact same source geometry without rewriting sidecars.
    features = []
    for i in range(count):
        cls = CLASS_NAMES[int(classes[i])]
        if GEOMETRY_NAMES[int(geometry[i])] != CLASS_GEOMETRY[cls]:
            raise ValueError(f"feature {i} geometry disagrees with class {cls}")
        features.append(
            HDMapFeature(
                cls=cls,
                vertices=vertices[int(offsets[i]) : int(offsets[i + 1])],
                feature_id=int(ids[i]),
                subtype=int(subtypes[i]),
                speed_limit_mph=float(speeds[i]),
                interpolating=bool(interpolating[i]),
                attributes=dict(attributes[i]),
            )
        )
    offset = meta.get("map_pose_offset")
    map_pose_offset = np.asarray(arrays["map_pose_offset"], dtype=np.float64)
    if offset is None:
        if map_pose_offset.shape != (0, 3):
            raise ValueError("map_pose_offset array is populated but metadata says it is absent")
        map_pose_offset_or_none = None
    else:
        meta_offset = np.asarray(offset, dtype=np.float64)
        if not np.array_equal(map_pose_offset, meta_offset):
            raise ValueError("map_pose_offset array disagrees with metadata")
        map_pose_offset_or_none = map_pose_offset
    return HDMapScene(
        segment=meta["segment"],
        scene_id=meta["scene_id"],
        split=meta["split"],
        features=features,
        geometry_source=meta["geometry_source"],
        attribute_source=meta["attribute_source"],
        map_pose_offset=map_pose_offset_or_none,
        schema_version=meta["schema_version"],
    ).validate()


# ---------------------------------------------------------------------------
# Disk IO
# ---------------------------------------------------------------------------

def scene_dir(root: str | Path, scene_id: str) -> Path:
    return Path(root) / str(scene_id)


def write_scene(root: str | Path, scene: HDMapScene, *, write_json: bool = True) -> Path:
    """Write ``hdmap.json`` + ``hdmap.npz`` atomically into ``root/<scene_id>``."""
    scene.validate()
    directory = scene_dir(root, scene.scene_id)
    directory.mkdir(parents=True, exist_ok=True)
    tmp_json = directory / "hdmap.json.tmp"
    tmp_npz = directory / "hdmap.npz.tmp"
    try:
        # Prepare both complete files before replacing either old artifact.  A
        # serialization failure therefore leaves the previous readable pair
        # untouched and the finally block removes every temporary.
        if write_json:
            with open(tmp_json, "w", encoding="utf-8") as handle:
                json.dump(scene_to_json_obj(scene), handle, ensure_ascii=False)
        # Pass a file handle, not a path: ``np.savez_compressed`` appends
        # ``.npz`` to a path whose suffix is not already .npz.
        with open(tmp_npz, "wb") as handle:
            np.savez_compressed(handle, **scene_to_npz_arrays(scene))
        if write_json:
            tmp_json.replace(directory / "hdmap.json")
        tmp_npz.replace(directory / "hdmap.npz")
    finally:
        for tmp in (tmp_json, tmp_npz):
            if tmp.exists():
                tmp.unlink()
    return directory


def read_scene_npz(root: str | Path, scene_id: str) -> HDMapScene:
    path = scene_dir(root, scene_id) / "hdmap.npz"
    with np.load(path, allow_pickle=False) as handle:
        return npz_arrays_to_scene(handle)


def read_scene_json(root: str | Path, scene_id: str) -> HDMapScene:
    path = scene_dir(root, scene_id) / "hdmap.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json_obj_to_scene(json.load(handle))


# ---------------------------------------------------------------------------
# Segment name / scene id helpers (must match datasets/dataset.py exactly)
# ---------------------------------------------------------------------------

def normalize_segment_name(name: str) -> str:
    """Mirror ``datasets/dataset.py::_normalize_waymo_caption_base``."""
    base = str(name).lstrip("﻿").rstrip("/").rsplit("/", 1)[-1]
    if base.endswith(".tfrecord"):
        base = base[: -len(".tfrecord")]
    if base.endswith(".tar"):
        base = base[: -len(".tar")]
    if base.startswith("segment-"):
        base = base[len("segment-") :]
    suffix = "_with_camera_labels"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def scene_id_map(list_path: str | Path) -> dict[str, str]:
    """``segment -> "000"`` using the same sort that ``WaymoOpenDataset`` uses.

    ``datasets/dataset.py::_load_scene_name_to_base_from_lists`` sorts the
    normalized names and enumerates them, so a scene folder ``000`` is the
    lexicographically first segment of that split -- *not* the first line of the
    list file.  Reproducing the sort here is mandatory; using file order would
    attach every hdmap to the wrong scene for validation.
    """
    with open(list_path, "r", encoding="utf-8-sig") as handle:
        lines = [line.strip().lstrip("﻿") for line in handle if line.strip()]
    bases = sorted(normalize_segment_name(item) for item in lines)
    if len(bases) != len(set(bases)):
        raise ValueError(f"segment list contains duplicates: {list_path}")
    return {base: f"{idx:03d}" for idx, base in enumerate(bases)}


def summarize(scene: HDMapScene) -> str:
    counts = scene.counts()
    total_vertices = sum(len(f.vertices) for f in scene.features)
    parts = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    return (
        f"{scene.scene_id} {scene.segment[:24]}... "
        f"[{parts or 'empty'}] verts={total_vertices} attrs={scene.attribute_source}"
    )


__all__ = [
    "SCHEMA_VERSION",
    "CLASS_NAMES",
    "CLASS_IDS",
    "CLASS_GEOMETRY",
    "GEOMETRY_NAMES",
    "GEOMETRY_IDS",
    "RDS_CLASS_DIRS",
    "RDS_FILE_SUFFIX",
    "RDS_MISSING_CLASSES",
    "SUBTYPE_TABLES",
    "SUBTYPE_UNKNOWN",
    "FEATURE_ID_UNKNOWN",
    "RASTER_CHANNEL_NAMES",
    "RASTER_CHANNEL_COUNT",
    "RASTER_SOURCE_EXCLUDED_CLASSES",
    "RASTER_RESERVED_ZERO_CHANNELS",
    "MAP_METRIC_RESERVED_ZERO_GROUPS",
    "RASTER_QUANTIZATION_FIELDS",
    "RASTER_QUANTIZATION",
    "RASTER_SIGNED_CHANNELS",
    "RASTER_FLAG_CHANNELS",
    "RASTER_SCHEMA_HASH",
    "raster_schema_hash",
    "ATTRIBUTE_SOURCE_NONE",
    "ATTRIBUTE_SOURCE_TFRECORD",
    "GEOMETRY_SOURCE_RDS",
    "GEOMETRY_SOURCE_TFRECORD",
    "HDMapFeature",
    "HDMapScene",
    "scene_to_json_obj",
    "json_obj_to_scene",
    "scene_to_npz_arrays",
    "npz_arrays_to_scene",
    "write_scene",
    "read_scene_npz",
    "read_scene_json",
    "scene_dir",
    "normalize_segment_name",
    "scene_id_map",
    "summarize",
]
