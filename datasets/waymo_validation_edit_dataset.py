"""Validation edit dataset for the FlowDGGT validation flow-cache pipeline.

Each ``data/final_info_validation.json`` entry describes four edits on one
29-frame clip: ``deletion`` / ``insertion`` / ``replacement`` / ``repositioning``.
This dataset turns one entry into a ``sample`` dict consumable by the shared
``gaussian_edit`` / ``asset_pass`` primitives, using a fixed **6-slot** layout:

====  ============================  ===========================  ====================
slot  role                          3D box source                     asset
====  ============================  ===========================  ====================
0     delete: deletion source       all_object_info tar          none (deleted)
1     delete: replacement source    all_object_info tar          none (deleted)
2     delete: repositioning source  all_object_info tar          none (deleted)
3     asset: insertion              all_object_info_insertion    insertion_candidates
4     asset: replacement            all_object_info_replacement  replacement_candidates
5     asset: repositioning          all_object_info_reposition   {repositioning_id}.ply
====  ============================  ===========================  ====================

Delete slots (0..2) are localized by the stock :func:`localize_objects` with
``load_asset=False`` (asset stays empty -> contributes only deletion). Asset
slots (3..5) are *not* in the localize loop and carry ``object_valid_mask=False``
so :func:`_collect_protected_boxes` skips destination boxes; otherwise a
destination overlapping its source could block source deletion. Asset
placement happens in :mod:`dggt.utils.validation_edit_localize`.

3D boxes come from ``data/validation_info/*`` tars (NOT tfrecord / instances).
The tar ``object_to_world`` is in the same absolute Waymo world frame as the
processed dataset's ``ego_pose`` (verified), so NO normalization is applied.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.waymo_edit_dataset import (
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_TRANSFER_HW,
    build_box_corners_world,
    build_intrinsic_matrix,
    compose_waymo_camera_to_world,
    load_and_preprocess_binary_masks,
    load_and_preprocess_images,
    numpy_like_to_torch,
    read_image_size,
    resolve_image_path,
)

# Segment-name -> processed validation scene-dir index (authoritative mapping).
from pointcloud_validation.toolkits.waymo_name_index import val_name2index

CLIP_LENGTH = 29
NUM_SLOTS = 6
DEFAULT_ASSET_ROOT = "/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed"
DEFAULT_ALL_OBJECT_INFO_ROOT = "data/validation_info/all_object_info"
DEFAULT_ALL_OBJECT_INFO_INSERTION_ROOT = "data/validation_info/all_object_info_insertion"
DEFAULT_ALL_OBJECT_INFO_REPLACEMENT_ROOT = "data/validation_info/all_object_info_replacement"
DEFAULT_ALL_OBJECT_INFO_REPOSITION_ROOT = "data/validation_info/all_object_info_reposition"

# slot index -> (role, edit-variant it participates in)
DELETE_SLOTS = (0, 1, 2)
ASSET_SLOTS = (3, 4, 5)
SLOT_ROLE = {
    0: "delete_deletion",
    1: "delete_replacement",
    2: "delete_repositioning",
    3: "asset_insertion",
    4: "asset_replacement",
    5: "asset_repositioning",
}
class MissingAssetError(RuntimeError):
    """Raised when a required asset ``.ply`` cannot be resolved."""

    def __init__(self, missing: list[tuple[str, str]]):
        self.missing = missing
        msg = "; ".join(f"{aid} -> {path}" for aid, path in missing)
        super().__init__(f"missing asset ply: {msg}")


def _resolve_mask_root(scene_root: Path) -> Path | None:
    for cand in (
        scene_root / "fine_dynamic_masks" / "all",
        scene_root / "dynamic_masks",
        scene_root / "fine_dynamic_masks" / "vehicle",
        scene_root / "dynamic_masks" / "vehicle",
    ):
        if cand.is_dir():
            return cand
    return None


def _list_cam0_frame_indices(scene_root: Path) -> list[int]:
    image_root = scene_root / "images"
    out: list[int] = []
    for path in sorted(image_root.glob("*_0.jpg")) + sorted(image_root.glob("*_0.png")):
        try:
            out.append(int(path.stem.split("_")[0]))
        except Exception:
            continue
    return sorted(out)


class _TarFrames:
    """Lazy reader over a ``<segment>.tar`` of per-frame all_object_info JSONs.

    Member names contain the source timestamp index, e.g.
    ``<segment>.000174.all_object_info.json`` for scene frame ``174 / 3 = 58``.
    Some validation target tars start in the middle of a scene, so indexing by
    sorted-member position would silently shift every target frame.
    """

    def __init__(self, tar_path: Path):
        self.tar_path = tar_path
        with tarfile.open(tar_path) as tf:
            names = sorted(n for n in tf.getnames() if n.endswith(".json"))
        self._name_by_scene_frame: dict[int, str] = {}
        for name in names:
            parts = Path(name).name.split(".")
            if len(parts) < 4:
                continue
            try:
                source_frame_number = int(parts[-3])
            except ValueError:
                continue
            if source_frame_number % 3 != 0:
                raise ValueError(
                    f"unexpected all_object_info member frame number: {name}"
                )
            scene_frame_idx = source_frame_number // 3
            if scene_frame_idx in self._name_by_scene_frame:
                raise ValueError(
                    f"duplicate scene frame {scene_frame_idx} in {tar_path}"
                )
            self._name_by_scene_frame[scene_frame_idx] = name
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._name_by_scene_frame)

    def frame(self, scene_frame_idx: int) -> dict[str, Any]:
        member_name = self._name_by_scene_frame.get(int(scene_frame_idx))
        if member_name is None:
            return {}
        if scene_frame_idx in self._cache:
            return self._cache[scene_frame_idx]
        with tarfile.open(self.tar_path) as tf:
            member = tf.extractfile(member_name)
            data = json.load(member) if member is not None else {}
        self._cache[scene_frame_idx] = data
        return data


class WaymoValidationEditDataset:
    """``final_info_validation`` entry -> 29-frame 6-slot ``sample`` dict."""

    def __init__(
        self,
        final_info_path: str = "data/final_info_validation.json",
        processed_root: str = DEFAULT_PROCESSED_ROOT,
        all_object_info_root: str = DEFAULT_ALL_OBJECT_INFO_ROOT,
        all_object_info_insertion_root: str = DEFAULT_ALL_OBJECT_INFO_INSERTION_ROOT,
        all_object_info_replacement_root: str = DEFAULT_ALL_OBJECT_INFO_REPLACEMENT_ROOT,
        all_object_info_reposition_root: str = DEFAULT_ALL_OBJECT_INFO_REPOSITION_ROOT,
        asset_root: str = DEFAULT_ASSET_ROOT,
        split: str = "validation",
        clip_length: int = CLIP_LENGTH,
    ) -> None:
        self.final_info_path = Path(final_info_path)
        self.processed_root = Path(processed_root)
        self.split = str(split)
        self.processed_split_root = self.processed_root / self.split
        self.annotations_root = (
            self.processed_root / "waymo_edit_cache" / "annotations" / self.split
        )
        self.all_object_info_root = Path(all_object_info_root)
        self.all_object_info_insertion_root = Path(all_object_info_insertion_root)
        self.all_object_info_replacement_root = Path(all_object_info_replacement_root)
        self.all_object_info_reposition_root = Path(all_object_info_reposition_root)
        self.asset_root = Path(asset_root)
        self.clip_length = int(clip_length)
        self.camera_ids = [0]  # views=1 only (editor constraint)

        with open(self.final_info_path) as f:
            self.entries: list[dict[str, Any]] = json.load(f)

        self._ann_cache: dict[str, dict[str, Any]] = {}
        self._tar_cache: dict[str, _TarFrames] = {}
        self._ins_tar_cache: dict[str, _TarFrames] = {}
        self._replacement_tar_cache: dict[str, _TarFrames] = {}
        self._reposition_tar_cache: dict[str, _TarFrames] = {}

    def __len__(self) -> int:
        return len(self.entries)

    # ---- helpers -----------------------------------------------------------

    def _annotation(self, segment: str) -> dict[str, Any]:
        if segment not in self._ann_cache:
            ann_path = (
                self.annotations_root
                / f"segment-{segment}_with_camera_labels.json"
            )
            if not ann_path.is_file():
                raise FileNotFoundError(f"annotation json not found: {ann_path}")
            with open(ann_path) as f:
                self._ann_cache[segment] = json.load(f)
        return self._ann_cache[segment]

    def _all_object_info(self, segment: str) -> _TarFrames:
        if segment not in self._tar_cache:
            tar_path = self.all_object_info_root / f"{segment}.tar"
            if not tar_path.is_file():
                raise FileNotFoundError(f"all_object_info tar not found: {tar_path}")
            self._tar_cache[segment] = _TarFrames(tar_path)
        return self._tar_cache[segment]

    def _insertion_info(self, segment: str) -> _TarFrames:
        if segment not in self._ins_tar_cache:
            tar_path = self.all_object_info_insertion_root / f"{segment}.tar"
            if not tar_path.is_file():
                raise FileNotFoundError(
                    f"all_object_info_insertion tar not found: {tar_path}"
                )
            self._ins_tar_cache[segment] = _TarFrames(tar_path)
        return self._ins_tar_cache[segment]

    def _replacement_info(self, segment: str) -> _TarFrames:
        if segment not in self._replacement_tar_cache:
            tar_path = self.all_object_info_replacement_root / f"{segment}.tar"
            if not tar_path.is_file():
                raise FileNotFoundError(
                    f"all_object_info_replacement tar not found: {tar_path}"
                )
            self._replacement_tar_cache[segment] = _TarFrames(tar_path)
        return self._replacement_tar_cache[segment]

    def _reposition_info(self, segment: str) -> _TarFrames:
        if segment not in self._reposition_tar_cache:
            tar_path = self.all_object_info_reposition_root / f"{segment}.tar"
            if not tar_path.is_file():
                raise FileNotFoundError(
                    f"all_object_info_reposition tar not found: {tar_path}"
                )
            self._reposition_tar_cache[segment] = _TarFrames(tar_path)
        return self._reposition_tar_cache[segment]

    def _resolve_asset_path(self, asset_id: str) -> str:
        return str(self.asset_root / f"{asset_id}.ply")

    # ---- main --------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        clip_name = str(entry["clip_name"])
        segment, clip_index_str = clip_name.rsplit("_", 1)
        clip_index = int(clip_index_str)

        if segment not in val_name2index:
            raise KeyError(f"segment not in val_name2index: {segment}")
        scene_idx = int(val_name2index[segment])
        scene_dir = f"{scene_idx:03d}"
        scene_root = self.processed_split_root / scene_dir
        if not scene_root.is_dir():
            raise FileNotFoundError(f"processed scene dir not found: {scene_root}")

        frames = _list_cam0_frame_indices(scene_root)
        start = clip_index * self.clip_length
        scene_frame_indices = frames[start : start + self.clip_length]
        if len(scene_frame_indices) != self.clip_length:
            raise ValueError(
                f"clip {clip_name}: need {self.clip_length} frames, got "
                f"{len(scene_frame_indices)} (scene has {len(frames)} frames)"
            )

        annotation = self._annotation(segment)
        all_obj = self._all_object_info(segment)
        ins_obj = self._insertion_info(segment)
        replacement_obj = self._replacement_info(segment)
        reposition_obj = self._reposition_info(segment)

        # ---- images / sky / dynamic ---------------------------------------
        dynamic_root = _resolve_mask_root(scene_root)
        image_paths, sky_paths, dyn_paths = [], [], []
        for f in scene_frame_indices:
            for cam_id in self.camera_ids:
                image_paths.append(resolve_image_path(scene_root / "images", f, cam_id))
                sky_paths.append(resolve_image_path(scene_root / "sky_masks", f, cam_id))
                dyn_paths.append(
                    resolve_image_path(dynamic_root, f, cam_id)
                    if dynamic_root is not None
                    else ""
                )
        images = load_and_preprocess_images(image_paths)
        sky_masks = load_and_preprocess_binary_masks(sky_paths)
        if all(p and Path(p).is_file() for p in dyn_paths):
            dynamic_masks = load_and_preprocess_binary_masks(dyn_paths)
        else:
            dynamic_masks = torch.zeros_like(images)
        model_hw = (int(images.shape[-2]), int(images.shape[-1]))

        # ---- cameras (views=1: cam 0) -------------------------------------
        ego_pose_all = np.asarray(annotation["ego_pose"], dtype=np.float32)
        ego_pose_selected = ego_pose_all[scene_frame_indices]
        cam_id = self.camera_ids[0]
        cam_to_world_full = np.asarray(
            annotation["camera_to_world"][str(cam_id)], dtype=np.float32
        )
        camera_to_world = cam_to_world_full[scene_frame_indices]
        cam_to_ego = np.asarray(
            annotation["camera_to_ego"][str(cam_id)], dtype=np.float32
        )
        camera_to_world_corrected = np.stack(
            [compose_waymo_camera_to_world(ep, cam_to_ego) for ep in ego_pose_selected],
            axis=0,
        )
        cam_img_path = resolve_image_path(scene_root / "images", scene_frame_indices[0], cam_id)
        cam_hw = read_image_size(cam_img_path)  # (H, W)
        intrinsic = build_intrinsic_matrix(
            annotation["normalized_intrinsics"][str(cam_id)], cam_hw
        )

        camera_to_world = numpy_like_to_torch(
            camera_to_world[:, None, ...], dtype=torch.float32
        )  # [S,1,4,4]
        camera_to_world_corrected = numpy_like_to_torch(
            camera_to_world_corrected[:, None, ...], dtype=torch.float32
        )  # [S,1,4,4]
        intrinsics = numpy_like_to_torch(
            np.asarray(intrinsic, dtype=np.float32)[None, ...], dtype=torch.float32
        )  # [1,3,3]
        camera_to_ego_t = numpy_like_to_torch(
            cam_to_ego[None, ...], dtype=torch.float32
        )  # [1,4,4]
        ego_pose_t = numpy_like_to_torch(ego_pose_selected, dtype=torch.float32)
        raw_hw = np.asarray([[cam_hw[0], cam_hw[1]]], dtype=np.int64)  # [1,2]

        # ---- per-slot Waymo boxes from tars -------------------------------
        od = entry["origin_object_dict"]
        slot_raw_id = {
            0: str(od.get("deletion", "")),
            1: str(od.get("replacement", "")),
            2: str(od.get("repositioning", "")),
            3: str(entry.get("insertion_candidates", "")),
            4: str(entry.get("replacement_candidates", "")),
            # Move uses the authoritative external PLY keyed by the source raw id.
            5: str(od.get("repositioning", "")),
        }

        S = self.clip_length
        obj_to_world = torch.zeros((NUM_SLOTS, S, 4, 4), dtype=torch.float32)
        box_size = torch.zeros((NUM_SLOTS, S, 3), dtype=torch.float32)
        box_corners = torch.zeros((NUM_SLOTS, S, 8, 3), dtype=torch.float32)
        track_valid = torch.zeros((NUM_SLOTS, S), dtype=torch.bool)
        is_moving_frame = torch.zeros((NUM_SLOTS, S), dtype=torch.bool)
        bbox_model = torch.zeros((NUM_SLOTS, S, 1, 4), dtype=torch.float32)
        bbox_present = torch.zeros((NUM_SLOTS, S, 1), dtype=torch.bool)

        def _fill_box(slot: int, j: int, o2w: np.ndarray, lwh: np.ndarray, moving: bool) -> None:
            corners = build_box_corners_world(o2w, lwh)
            obj_to_world[slot, j] = numpy_like_to_torch(o2w, dtype=torch.float32)
            box_size[slot, j] = numpy_like_to_torch(lwh, dtype=torch.float32)
            box_corners[slot, j] = numpy_like_to_torch(corners, dtype=torch.float32)
            track_valid[slot, j] = True
            is_moving_frame[slot, j] = bool(moving)
            # 2D model-space bbox = projected Waymo corners (same convention as
            # localize_objects' internal waymo_bbox_model -> consistent target).
            from dggt.utils.gaussian_edit import (
                compute_bbox_from_projected_points,
                project_waymo_box_corners_model,
            )

            uv, _, valid = project_waymo_box_corners_model(
                torch.as_tensor(corners, dtype=torch.float32),
                camera_to_world_corrected[j, 0],
                intrinsics[0],
                torch.as_tensor(raw_hw[0]),
                model_hw,
            )
            box = compute_bbox_from_projected_points(uv, valid)
            if box is not None:
                bbox_model[slot, j, 0] = box
                bbox_present[slot, j, 0] = True

        for j, sf in enumerate(scene_frame_indices):
            frame_objs = all_obj.frame(sf)
            # Delete slots always use the original scene-object tracks.
            for slot in (0, 1, 2):
                rid = slot_raw_id[slot]
                info = frame_objs.get(rid)
                if info is None:
                    continue
                _fill_box(
                    slot,
                    j,
                    np.asarray(info["object_to_world"], dtype=np.float32),
                    np.asarray(info["object_lwh"], dtype=np.float32),
                    bool(info.get("object_is_moving", False)),
                )
            # slot 3 (insertion) box from insertion tar
            ins_frame = ins_obj.frame(sf)
            ins = ins_frame.get("insertion_0")
            if ins is not None:
                _fill_box(
                    3,
                    j,
                    np.asarray(ins["object_to_world"], dtype=np.float32),
                    np.asarray(ins["object_lwh"], dtype=np.float32),
                    bool(ins.get("object_is_moving", False)),
                )
            # Slots 4 and 5 use their authoritative per-frame destination
            # tracks. The reposition tar already contains the requested
            # 3-m local-axis displacement; do not shift it again here.
            replacement_target = replacement_obj.frame(sf).get(str(od.get("replacement", "")))
            if replacement_target is not None:
                _fill_box(
                    4,
                    j,
                    np.asarray(replacement_target["object_to_world"], dtype=np.float32),
                    np.asarray(replacement_target["object_lwh"], dtype=np.float32),
                    bool(
                        replacement_target.get(
                            "object_is_moving",
                            is_moving_frame[1, j].item() if track_valid[1, j] else False,
                        )
                    ),
                )
            reposition_target = reposition_obj.frame(sf).get(str(od.get("repositioning", "")))
            if reposition_target is not None:
                _fill_box(
                    5,
                    j,
                    np.asarray(reposition_target["object_to_world"], dtype=np.float32),
                    np.asarray(reposition_target["object_lwh"], dtype=np.float32),
                    bool(
                        reposition_target.get(
                            "object_is_moving",
                            is_moving_frame[2, j].item() if track_valid[2, j] else False,
                        )
                    ),
                )

        # ---- asset path resolution + missing report -----------------------
        missing: list[tuple[str, str]] = []
        object_asset_paths: list[str] = [""] * NUM_SLOTS
        for slot in DELETE_SLOTS:
            # truthy string so localize_objects (load_asset=False) does not skip;
            # file need not exist (asset never loaded for delete slots).
            object_asset_paths[slot] = self._resolve_asset_path(slot_raw_id[slot])
        for slot in ASSET_SLOTS:
            aid = slot_raw_id[slot]
            path = self._resolve_asset_path(aid)
            object_asset_paths[slot] = path
            if bool(track_valid[slot].any().item()) and not Path(path).is_file():
                missing.append((aid, path))
        if missing:
            raise MissingAssetError(missing)

        # ---- scalar / mask object tensors ---------------------------------
        object_valid_mask = torch.zeros((NUM_SLOTS,), dtype=torch.bool)
        object_asset_valid_mask = torch.zeros((NUM_SLOTS,), dtype=torch.bool)
        asset_image_valid = torch.zeros((NUM_SLOTS, S, 1), dtype=torch.bool)
        for slot in DELETE_SLOTS:
            object_valid_mask[slot] = bool(track_valid[slot].any().item())
            object_asset_valid_mask[slot] = True  # not skipped by localize_objects
        for slot in ASSET_SLOTS:
            # valid_mask False -> _collect_protected_boxes skips destination
            # slots so they cannot block deletion of their source objects.
            object_valid_mask[slot] = False
            object_asset_valid_mask[slot] = bool(track_valid[slot].any().item())
            asset_image_valid[slot, :, 0] = track_valid[slot]

        editable_object_indices = torch.full((NUM_SLOTS,), -1, dtype=torch.long)
        present_asset_slots = [s for s in ASSET_SLOTS if bool(track_valid[s].any().item())]
        for i, s in enumerate(present_asset_slots):
            editable_object_indices[i] = s
        editable_object_count = torch.tensor(len(present_asset_slots), dtype=torch.long)

        zeros_o = torch.zeros((NUM_SLOTS,), dtype=torch.float32)
        sample: dict[str, Any] = {
            "sample_index": int(idx),
            "manifest_index": int(entry.get("index", idx)),
            "num_frames": torch.tensor(S, dtype=torch.long),
            "images": images,
            "images_clean": images,
            "image_paths": image_paths,
            "timestamps": torch.linspace(0.0, 1.0, S, dtype=torch.float32),
            "scene_id": scene_idx,
            "scene_name": f"{self.split}_{scene_dir}",
            "scene_dir": scene_dir,
            "scene_base": f"{self.split}_{scene_dir}",
            "clip_name": clip_name,
            "clip_index": torch.tensor(clip_index, dtype=torch.long),
            "cam_ids": torch.tensor(self.camera_ids, dtype=torch.long),
            "frame_indices": torch.tensor(scene_frame_indices, dtype=torch.long),
            "local_frame_indices": torch.arange(S, dtype=torch.long),
            "clip_frame_indices": torch.tensor(scene_frame_indices, dtype=torch.long),
            "edit_mode": "validation_edit",
            "masks": sky_masks,
            "sky_mask": sky_masks,
            "dynamic_mask": dynamic_masks,
            "camera_to_world": camera_to_world,
            "camera_to_world_corrected": camera_to_world_corrected,
            "camera_to_ego": camera_to_ego_t,
            "ego_pose": ego_pose_t,
            "intrinsics": intrinsics,
            "raw_image_size_hw": torch.tensor(raw_hw, dtype=torch.long),
            "transfer_image_size_hw": torch.tensor(DEFAULT_TRANSFER_HW, dtype=torch.long),
            # object tensors (the editor / asset_pass contract)
            "object_track_valid_mask_selected": track_valid,
            "object_obj_to_world_selected": obj_to_world,
            "object_box_size_selected": box_size,
            "object_box_corners_world_selected": box_corners,
            "object_is_moving_frame_selected": is_moving_frame,
            "object_speed_mps_selected": torch.zeros((NUM_SLOTS, S), dtype=torch.float32),
            "object_is_moving_track": is_moving_frame.any(dim=1),
            "object_max_speed_mps": zeros_o.clone(),
            "object_mean_speed_mps": zeros_o.clone(),
            "object_asset_valid_mask": object_asset_valid_mask,
            "object_scene_match_scores": torch.ones((NUM_SLOTS,), dtype=torch.float32),
            "object_scene_raw_ids": [slot_raw_id[s] for s in range(NUM_SLOTS)],
            "object_asset_ids": [slot_raw_id[s] for s in range(NUM_SLOTS)],
            "object_asset_paths": object_asset_paths,
            "object_valid_mask": object_valid_mask,
            "object_bbox_present_mask_selected": bbox_present,
            "object_bbox_model_selected": bbox_model,
            "object_bbox_editable_mask_selected": bbox_present.clone(),
            "object_front_bbox_present_mask_selected": bbox_present[:, :, 0].clone(),
            "object_front_bbox_model_selected": bbox_model[:, :, 0].clone(),
            "object_front_bbox_editable_mask_selected": bbox_present[:, :, 0].clone(),
            "object_asset_image_valid_mask_selected": asset_image_valid,
            "editable_object_indices": editable_object_indices,
            "editable_object_count": editable_object_count,
            "protected_object_indices": torch.full((NUM_SLOTS,), -1, dtype=torch.long),
            "protected_object_count": torch.tensor(0, dtype=torch.long),
            "asset_meta": {"asset_root": str(self.asset_root)},
            # validation routing payload (consumed by validation_edit_localize)
            "validation_edit": {
                "entry_index": int(entry.get("index", idx)),
                "clip_name": clip_name,
                "segment": segment,
                "scene_dir": scene_dir,
                "clip_index": clip_index,
                "slot_role": dict(SLOT_ROLE),
                "slot_raw_id": dict(slot_raw_id),
                "delete_slots": list(DELETE_SLOTS),
                "asset_slots": list(present_asset_slots),
                "action_for_reposition": str(entry.get("action_for_reposition", "up")),
                "trajectory": entry.get("trajectory", {}),
            },
        }
        return sample
