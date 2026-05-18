"""`WaymoFlowCacheDataset` — cache-backed dataloader for FlowDGGT training.

Each item is one `.pt` clip produced by `tools/precompute_flow_features.py`.
Both edit modes (`mode_a` and `mode_b`) share the same payload schema; the
`mode_kind` field distinguishes them. The dataset:

1. Loads the clip cache (metadata + int8 LUTs + per-object asset renders OR
   Mode-B planner output).
2. Randomly picks a 4–8 frame subset from the 29-frame clip.
3. Subsets every per-frame tensor to the chosen subset.
4. Dequantizes the int8 LUTs to fp16.
5. Returns a dict structured as `(sample, predictions, asset_pass_result, mode_b)`
   — exactly the inputs `FlowFeatureAssembler.forward` consumes for either
   mode.

Two ways to construct:

* `cache_root=...` — walk a single directory (one mode) and load all `.pt`s.
* `manifest_path=...` — read a merged JSONL produced by
  `tools/build_flow_train_manifest.py`. Each entry is one cache file path with
  its `mode_kind`; both modes can coexist in one manifest.

No VGGT is invoked here; the aggregator stays strictly offline.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from dggt.models.asset_pass import AssetPassResult
from dggt.models.gaussian_pointers import GaussianPointers, SRC_KIND_ASSET
from dggt.utils.feature_quant import QuantizedTokens, dequantize_tokens
from dggt.utils.flow_cache_io import load_flow_cache
from dggt.utils.gaussian_edit import Sim3Transform


def _parse_cache_root_spec(cache_root: str | Path) -> tuple[Path, str]:
    """Accept `path`, `path:mode_a`, `path:mode_b`, or `path:auto`."""
    raw = str(cache_root)
    if ":" in raw:
        path_str, mode_pin = raw.rsplit(":", 1)
        if mode_pin in ("mode_a", "mode_b", "auto"):
            return Path(path_str), mode_pin
    path = Path(raw)
    parts = set(path.parts)
    if "flow_cache_mode_a" in parts:
        return path, "mode_a"
    if "flow_cache_mode_b" in parts:
        return path, "mode_b"
    return path, "auto"


def _list_cache_files(cache_root: Path, split: str) -> list[Path]:
    split_root = cache_root / split
    if split_root.is_dir():
        root = split_root
    elif cache_root.is_dir():
        root = cache_root
    else:
        raise FileNotFoundError(f"Cache root not found: {split_root} or {cache_root}")
    return sorted(root.rglob("*.pt"))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _empty_asset_pass(patch_grid: tuple[int, int], patch_start_idx: int) -> AssetPassResult:
    return AssetPassResult(
        patch_grid=patch_grid,
        patch_start_idx=patch_start_idx,
        object_keys=[],
        cameras_waymo={},
        F_g_lut_asset={},
        ptr_asset={},
        G_asset_waymo={},
        G_asset_dggt={},
        I_asset={},
        A_asset={},
        asset_pass_space="empty",
        fit_metrics={},
    )


def _dequantize_nplc_subset(
    *,
    data: torch.Tensor,
    scale: torch.Tensor,
    subset: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Select target frames before dequantizing a `[N, P, L, C]` int8 LUT."""
    q = QuantizedTokens(
        data=data.index_select(0, subset),
        scale=scale.index_select(0, subset),
        layout="NPLC",
    )
    return dequantize_tokens(q, dtype=dtype)


def _split_nplc_levels(x: torch.Tensor) -> list[torch.Tensor]:
    """Convert `[S, P, L, C]` into level tensors `[1, S, P, C]`."""
    return [
        x[:, :, level, :].unsqueeze(0).contiguous()
        for level in range(int(x.shape[2]))
    ]


class WaymoFlowCacheDataset(Dataset):
    """Index-based dataset reading pre-computed FlowDGGT clip caches."""

    def __init__(
        self,
        cache_root: str | Path | None = None,
        split: str = "training",
        min_frames: int = 4,
        max_frames: int = 8,
        seed: int = 0,
        lut_dtype: torch.dtype = torch.float16,
        manifest_path: str | Path | None = None,
        mode_filter: list[str] | None = None,
        include_aux_tokens: bool = False,
        mmap_plain_cache: bool = True,
    ) -> None:
        super().__init__()
        if min_frames <= 0 or max_frames < min_frames:
            raise ValueError(f"Invalid frame range [{min_frames}, {max_frames}]")
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self._rng = random.Random(int(seed))
        self.lut_dtype = lut_dtype
        self.split = str(split)
        self.include_aux_tokens = bool(include_aux_tokens)
        self.mmap_plain_cache = bool(mmap_plain_cache)

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
            rows = _read_manifest(self.manifest_path)
            if mode_filter is not None:
                allowed = {str(m) for m in mode_filter}
                rows = [r for r in rows if str(r.get("mode_kind", "")) in allowed]
            if len(rows) == 0:
                raise RuntimeError(
                    f"Empty manifest after filtering: {self.manifest_path} "
                    f"(mode_filter={mode_filter})"
                )
            self.entries: list[dict[str, Any]] = rows
        else:
            if cache_root is None:
                raise ValueError("Either `cache_root` or `manifest_path` must be provided.")
            cache_roots: list[str | Path]
            if isinstance(cache_root, (list, tuple)):
                cache_roots = list(cache_root)
            else:
                cache_roots = [cache_root]
            entries: list[dict[str, Any]] = []
            allowed = {str(m) for m in mode_filter} if mode_filter is not None else None
            for raw_root in cache_roots:
                root_path, mode_pin = _parse_cache_root_spec(raw_root)
                if allowed is not None and mode_pin in ("mode_a", "mode_b") and mode_pin not in allowed:
                    continue
                files = _list_cache_files(root_path, self.split)
                entry_mode = mode_pin if mode_pin in ("mode_a", "mode_b") else "unknown"
                entries.extend({"cache_path": str(f), "mode_kind": entry_mode} for f in files)
            files = [Path(entry["cache_path"]) for entry in entries]
            if len(files) == 0:
                raise RuntimeError(
                    f"No cache files under {cache_roots} split={self.split}. "
                    "Run tools/precompute_flow_features.py first."
                )
            self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        cache_path = Path(entry["cache_path"])
        payload = load_flow_cache(
            cache_path,
            map_location="cpu",
            weights_only=False,
            mmap=self.mmap_plain_cache,
        )
        self._validate_v6_payload(payload, cache_path=cache_path, entry=entry)
        mode_kind = str(payload["mode_kind"])
        meta = payload["meta"]
        num_frames_all = int(meta["num_frames"])
        n_select = self._rng.randint(self.min_frames, self.max_frames)
        n_select = min(n_select, num_frames_all)
        subset = sorted(self._rng.sample(range(num_frames_all), n_select))
        subset_t = torch.tensor(subset, dtype=torch.long)

        sample = self._build_sample(payload, subset_t)
        sample["mode_kind"] = mode_kind
        sample["cache_index"] = int(entry.get("index", payload.get("meta", {}).get("manifest_index", idx)))
        predictions = self._build_predictions(payload, subset_t)
        if mode_kind == "mode_a":
            asset_pass_result = self._build_asset_pass(payload, subset_t)
            mode_b_block = None
        else:
            patch_grid = tuple(int(v) for v in meta["patch_grid"])
            asset_pass_result = _empty_asset_pass(patch_grid, int(meta["patch_start_idx"]))
            mode_b_block = self._build_mode_b(payload, subset_t)
        cameras_dggt = self._build_cameras_dggt(payload, subset_t)
        alignment = self._build_alignment(payload)

        # Schema v6 fast-path inputs.
        # * phase1_localized          — Mode A only (Mode B doesn't run editor.localize).
        # * pass2_splatted_tok_low    — both modes (precomputed splat→blend output,
        #                               i.e. the *input* to tokenizer.encode).
        #   This cache is generated with all clip Gaussians as splat sources and
        #   then sliced on the target-frame axis here. It is intentionally not
        #   equivalent to re-running live splat after dropping non-subset source
        #   Gaussians.
        schema_version = int(payload["schema_version"])
        phase1_localized_subset = None
        splatted_tok_low_cached = None
        if mode_kind == "mode_a":
            phase1_payload = payload.get("phase1_localized")
            if phase1_payload is None:
                raise RuntimeError(
                    f"Mode-A cache {cache_path} missing phase1_localized payload (schema v6)."
                )
            phase1_localized_subset = self._subset_phase1_localized(
                phase1_payload, subset_t
            )
        pass2_payload = payload.get("pass2_splatted_tok_low")
        if pass2_payload is None:
            raise RuntimeError(
                f"Cache {cache_path} missing pass2_splatted_tok_low payload "
                "(schema v6). Re-run tools/precompute_flow_features.py."
            )
        splatted_tok_low_cached = self._subset_pass2_splatted_tok_low(
            pass2_payload, subset_t, dtype=self.lut_dtype,
        )

        return {
            "sample": sample,
            "predictions": predictions,
            "asset_pass_result": asset_pass_result,
            "cameras_dggt": cameras_dggt,
            "alignment": alignment,
            "mode_kind": mode_kind,
            "mode_b": mode_b_block,
            "subset_frames": subset_t,
            "cache_path": str(cache_path),
            "phase1_localized": phase1_localized_subset,
            "splatted_tok_low_cached": splatted_tok_low_cached,
            "cache_schema_version": schema_version,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_v6_payload(
        payload: dict[str, Any],
        *,
        cache_path: Path,
        entry: dict[str, Any],
    ) -> None:
        """Fail early unless the payload is the latest v6 training cache."""
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != 6:
            raise RuntimeError(
                f"Cache {cache_path} has schema_version={schema_version}; "
                "training now supports only schema_version == 6. "
                "Re-run tools/precompute_flow_features.py."
            )
        mode_kind = payload.get("mode_kind")
        if mode_kind not in ("mode_a", "mode_b"):
            raise RuntimeError(
                f"Cache {cache_path} has invalid mode_kind={mode_kind!r}; "
                "v6 training caches must set 'mode_a' or 'mode_b'."
            )
        entry_mode = entry.get("mode_kind")
        if entry_mode not in (None, "", "unknown") and str(entry_mode) != str(mode_kind):
            raise RuntimeError(
                f"Manifest mode_kind={entry_mode!r} disagrees with cache "
                f"mode_kind={mode_kind!r} for {cache_path}."
            )
        if payload.get("pass2_splatted_tok_low") is None:
            raise RuntimeError(
                f"Cache {cache_path} missing pass2_splatted_tok_low; "
                "v6 training requires pre-tokenizer pass2 cache."
            )
        if payload.get("pass2_z_splat") is not None:
            raise RuntimeError(
                f"Cache {cache_path} still carries legacy pass2_z_splat; "
                "regenerate as schema v6 without tokenizer-output cache."
            )
        if mode_kind == "mode_a" and payload.get("phase1_localized") is None:
            raise RuntimeError(
                f"Mode-A cache {cache_path} missing phase1_localized; "
                "v6 training requires cached Phase-1 localization."
            )
        if mode_kind == "mode_b" and payload.get("phase1_localized") is not None:
            raise RuntimeError(
                f"Mode-B cache {cache_path} should not carry phase1_localized."
            )

    # ------------------------------------------------------------------ #
    def _build_sample(self, payload: dict[str, Any], subset: torch.Tensor) -> dict[str, Any]:
        meta = payload["meta"]
        raw = payload["raw"]
        images_u8 = raw["images_u8"].index_select(0, subset)
        images_clean = images_u8.to(torch.float32) / 255.0
        sky_mask = raw["sky_mask"].index_select(0, subset).to(torch.float32)
        dynamic_mask = raw["dynamic_mask"]
        if dynamic_mask is not None:
            dynamic_mask = dynamic_mask.index_select(0, subset).to(torch.float32)
        else:
            dynamic_mask = torch.zeros_like(sky_mask)

        obj = payload["object_meta"]
        out: dict[str, Any] = {}
        for key in (
            "object_asset_ids",
            "object_scene_raw_ids",
            "object_asset_paths",
            "object_valid_mask",
            "object_scene_match_scores",
            "object_max_speed_mps",
            "object_mean_speed_mps",
            "object_is_moving_track",
            "editable_object_indices",
            "editable_object_count",
            "protected_object_indices",
            "protected_object_count",
        ):
            if key in obj:
                out[key] = obj[key]
        for key in (
            "object_speed_mps_selected",
            "object_is_moving_frame_selected",
            "object_track_valid_mask_selected",
            "object_asset_image_valid_mask_selected",
            "object_bbox_present_mask_selected",
            "object_bbox_editable_mask_selected",
            "object_bbox_model_selected",
            "object_front_bbox_present_mask_selected",
            "object_front_bbox_editable_mask_selected",
            "object_front_bbox_model_selected",
        ):
            if key in obj:
                out[key] = obj[key].index_select(1, subset)
        for key in (
            "object_obj_to_world_selected",
            "object_box_size_selected",
            "object_box_corners_world_selected",
        ):
            if key in obj:
                out[key] = obj[key].index_select(1, subset)
        if "object_asset_image_paths_selected" in obj:
            out["object_asset_image_paths_selected"] = [
                [paths[n] for n in subset.tolist()] for paths in obj["object_asset_image_paths_selected"]
            ]
        if "protected_object_boxes_by_frame" in obj:
            boxes = obj["protected_object_boxes_by_frame"]
            out["protected_object_boxes_by_frame"] = [boxes[n] for n in subset.tolist()]

        cam_ids = meta["cam_ids"]
        V = int(cam_ids.numel())
        out["cam_ids"] = cam_ids
        out["frame_indices"] = meta["frame_indices_scene"].index_select(0, subset)
        out["timestamps"] = meta["timestamps"].index_select(0, subset)
        out["raw_image_size_hw"] = meta["raw_image_size_hw"]
        out["scene_name"] = meta["scene_name"]
        out["clip_name"] = meta["clip_name"]
        out["manifest_index"] = int(meta.get("manifest_index", -1))
        out["dataset_index"] = int(meta.get("dataset_index", -1))
        out["asset_meta"] = meta["asset_meta"]

        if "camera_to_world_corrected" in obj:
            out["camera_to_world_corrected"] = obj["camera_to_world_corrected"].index_select(0, subset)
        else:
            out["camera_to_world_corrected"] = torch.eye(4).view(1, 1, 4, 4).expand(subset.numel(), V, 4, 4).contiguous()
        if "intrinsics" in obj:
            out["intrinsics"] = obj["intrinsics"]
        else:
            out["intrinsics"] = torch.eye(3).view(1, 3, 3).expand(V, 3, 3).contiguous()

        out["images"] = images_clean
        out["images_clean"] = images_clean
        out["masks"] = sky_mask
        out["sky_mask"] = sky_mask
        out["dynamic_mask"] = dynamic_mask
        return out

    # ------------------------------------------------------------------ #
    def _build_predictions(self, payload: dict[str, Any], subset: torch.Tensor) -> dict[str, torch.Tensor]:
        pass1 = payload["pass1"]

        def _sub(t: torch.Tensor) -> torch.Tensor:
            return t.index_select(0, subset)

        gs_map = _sub(pass1["gs_map"]).unsqueeze(0)
        depth = _sub(pass1["depth"]).unsqueeze(0)
        dyn = _sub(pass1["dynamic_conf"]).unsqueeze(0)
        gs_conf = _sub(pass1["gs_conf"]).unsqueeze(0)
        pose_enc = _sub(pass1["pose_enc"]).unsqueeze(0)
        sem = pass1.get("semantic_logits")
        if sem is not None:
            sem = _sub(sem).unsqueeze(0)

        lut_sub = _dequantize_nplc_subset(
            data=pass1["F_g_lut_scene_int8"],
            scale=pass1["F_g_lut_scene_scale"],
            subset=subset,
            dtype=self.lut_dtype,
        )
        image_tokens_levels = _split_nplc_levels(lut_sub)

        agg_levels = None
        if self.include_aux_tokens and pass1.get("aggregated_tokens_patch_int8") is not None:
            agg_sub = _dequantize_nplc_subset(
                data=pass1["aggregated_tokens_patch_int8"],
                scale=pass1["aggregated_tokens_patch_scale"],
                subset=subset,
                dtype=self.lut_dtype,
            )
            agg_levels = _split_nplc_levels(agg_sub)
        dino_levels = None
        if self.include_aux_tokens and pass1.get("dino_tokens_patch_int8") is not None:
            dino_sub = _dequantize_nplc_subset(
                data=pass1["dino_tokens_patch_int8"],
                scale=pass1["dino_tokens_patch_scale"],
                subset=subset,
                dtype=self.lut_dtype,
            )
            dino_levels = _split_nplc_levels(dino_sub)

        return {
            "pose_enc": pose_enc,
            "depth": depth,
            "gs_map": gs_map,
            "dynamic_conf": dyn,
            "gs_conf": gs_conf,
            "semantic_logits": sem,
            "image_tokens_levels": image_tokens_levels,
            "aggregated_tokens_levels": agg_levels,
            "dino_tokens_levels": dino_levels,
            "patch_start_idx": int(payload["meta"]["patch_start_idx"]),
        }

    # ------------------------------------------------------------------ #
    def _build_asset_pass(self, payload: dict[str, Any], subset: torch.Tensor) -> AssetPassResult:
        meta = payload["meta"]
        asset = payload["asset_pass"]
        patch_grid = tuple(int(v) for v in meta["patch_grid"])
        patch_start_idx = int(meta["patch_start_idx"])
        object_keys = sorted(int(k) for k in asset.keys())
        if len(object_keys) == 0:
            return _empty_asset_pass(patch_grid, patch_start_idx)
        asset_pass_space = str(meta.get("asset_pass_space", ""))
        if asset_pass_space != "dggt_fitted":
            raise RuntimeError(
                "Mode-A cache asset_pass must be regenerated with fitted DGGT asset geometry. "
                f"Expected meta.asset_pass_space='dggt_fitted', got {asset_pass_space!r}."
            )

        cameras_waymo: dict[str, torch.Tensor] = {}

        F_g_lut_asset: dict[int, list[torch.Tensor]] = {}
        ptr_asset: dict[int, list[GaussianPointers]] = {}
        G_asset_dggt: dict[int, list[dict[str, torch.Tensor]]] = {}
        I_asset: dict[int, torch.Tensor] = {}
        A_asset: dict[int, torch.Tensor] = {}
        fit_metrics: dict[int, list[dict[str, Any]]] = {}

        for k in object_keys:
            entry = asset[k]
            F_sub = _dequantize_nplc_subset(
                data=entry["F_g_lut_asset_int8"],
                scale=entry["F_g_lut_asset_scale"],
                subset=subset,
                dtype=self.lut_dtype,
            )
            F_g_lut_asset[k] = _split_nplc_levels(F_sub)

            subset_list = subset.tolist()
            remap = {n: i for i, n in enumerate(subset_list)}
            ptr_list: list[GaussianPointers] = []
            for s_idx, orig_frame in enumerate(subset_list):
                patch_idx = entry["ptr_patch_idx"][orig_frame]
                visible = entry["ptr_visible_mask"][orig_frame]
                view_n_orig = entry["ptr_view_n"][orig_frame]
                view_n_mapped = torch.tensor(
                    [remap.get(int(v), s_idx) for v in view_n_orig.tolist()],
                    dtype=torch.int32,
                )
                n = int(patch_idx.numel())
                ptr_list.append(
                    GaussianPointers(
                        src_kind=torch.full((n,), SRC_KIND_ASSET, dtype=torch.int32),
                        object_id=torch.full((n,), int(k), dtype=torch.int32),
                        view_n=view_n_mapped,
                        patch_idx=patch_idx.to(torch.int32),
                        visible_mask=visible.to(torch.bool),
                    )
                )
            ptr_asset[k] = ptr_list

            dggt_frames = entry.get("G_asset_dggt_per_frame")
            if dggt_frames is None:
                raise RuntimeError(
                    f"Mode-A v6 cache object {k} lacks G_asset_dggt_per_frame. "
                    "Re-run tools/precompute_flow_features.py."
                )
            G_asset_dggt[k] = [dggt_frames[n] for n in subset_list]
            fit_full = entry.get("fit_metrics")
            if fit_full is not None:
                fit_metrics[k] = [fit_full[n] for n in subset_list]

            I_asset[k] = entry["I_asset"].index_select(0, subset).to(torch.float32).div(255.0).unsqueeze(0)
            A_asset[k] = entry["A_asset"].index_select(0, subset).to(torch.float32).div(255.0).unsqueeze(0)

        return AssetPassResult(
            patch_grid=patch_grid,
            patch_start_idx=patch_start_idx,
            object_keys=object_keys,
            cameras_waymo=cameras_waymo,
            F_g_lut_asset=F_g_lut_asset,
            ptr_asset=ptr_asset,
            G_asset_waymo={},
            G_asset_dggt=G_asset_dggt,
            I_asset=I_asset,
            A_asset=A_asset,
            asset_pass_space=asset_pass_space,
            fit_metrics=fit_metrics,
        )

    # ------------------------------------------------------------------ #
    def _build_mode_b(self, payload: dict[str, Any], subset: torch.Tensor) -> dict[str, Any]:
        """Return the Mode-B payload subset to the chosen frames.

        Keys:
          imagined_objects: list[dict] (UNCHANGED — geometry refers to absolute
                            frame indices in the 29-frame clip; the assembler
                            uses subset_frames to pick visibility per chosen
                            frame).
          delete_mask:      [N_gauss_subset] bool, the union of the per-target
                            frame masks below.
          delete_mask_per_frame_subset: [|subset|, N_gauss_subset] bool,
                            aligned with the subset clean_state.
          subset_frames:    [|subset|] long  (mirror of the dataset-wide subset)
          rejection_reason, eligible, num_imagined_objects, metrics, rng_seed.
        """
        block = payload["mode_b"]
        if block is None:
            raise RuntimeError(
                f"Mode B payload missing for cache: {payload.get('meta', {}).get('clip_name', '?')}. "
                "Re-run tools/precompute_flow_features.py --edit_mode mode_b on this clip."
            )
        delete_mask_full = block["delete_mask"].to(torch.bool)
        delete_mask_per_frame = block["delete_mask_per_frame"].to(torch.bool)
        if delete_mask_per_frame.dim() != 2:
            raise ValueError(
                f"delete_mask_per_frame should be [N_clip, N_gauss], got {tuple(delete_mask_per_frame.shape)}"
            )
        if int(delete_mask_per_frame.shape[1]) != int(delete_mask_full.numel()):
            raise ValueError(
                "delete_mask_per_frame N_gauss does not match delete_mask: "
                f"{tuple(delete_mask_per_frame.shape)} vs {int(delete_mask_full.numel())}"
            )
        offsets = self._frame_gauss_offsets_from_payload(payload)
        subset_clip = subset.clone()
        # Some Mode B planner flows use num_frames = max frame_idx + 1, which can
        # be < num_clip_frames if the imagined objects span fewer frames. Clamp.
        n_clip = int(delete_mask_per_frame.shape[0])
        subset_list = subset.tolist()
        local_rows: list[torch.Tensor] = []
        if n_clip > 0:
            for target_f in subset_list:
                row_full = delete_mask_per_frame[min(int(target_f), n_clip - 1)]
                row_chunks: list[torch.Tensor] = []
                for source_f in subset_list:
                    s = int(offsets[int(source_f)].item())
                    e = int(offsets[int(source_f) + 1].item())
                    row_chunks.append(row_full[s:e])
                local_rows.append(
                    torch.cat(row_chunks) if row_chunks else torch.zeros(0, dtype=torch.bool)
                )
        if local_rows:
            delete_mask_subset = torch.stack(local_rows, dim=0)
        else:
            subset_count = 0
            for source_f in subset_list:
                subset_count += int(offsets[int(source_f) + 1].item() - offsets[int(source_f)].item())
            delete_mask_subset = torch.zeros((int(subset_clip.numel()), subset_count), dtype=torch.bool)
        delete_mask = (
            delete_mask_subset.any(dim=0)
            if delete_mask_subset.numel() > 0
            else torch.zeros((0,), dtype=torch.bool)
        )
        return {
            "imagined_objects": list(block.get("imagined_objects", [])),
            "rejection_reason": str(block.get("rejection_reason", "")),
            "eligible": bool(block.get("eligible", True)),
            "num_imagined_objects": int(block.get("num_imagined_objects", 0)),
            "metrics": dict(block.get("metrics", {})),
            "rng_seed": int(block.get("rng_seed", 0)),
            "delete_mask": delete_mask,
            "delete_mask_per_frame": delete_mask_subset,
            "delete_mask_per_frame_subset": delete_mask_subset,
            "subset_frames": subset_clip,
            "delete_core_indices": block.get("delete_core_indices"),
            "delete_shell_indices": block.get("delete_shell_indices"),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame_gauss_offsets_from_payload(payload: dict[str, Any]) -> torch.Tensor:
        """Rebuild per-frame Gaussian offsets from the cached dense pass1 layout."""
        depth = payload["pass1"]["depth"].float()
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        sky_mask = payload["raw"]["sky_mask"].float()
        sky_mask_hw = sky_mask.permute(0, 2, 3, 1)
        non_sky = (sky_mask_hw < 0.5).any(dim=-1)
        valid = non_sky & (depth > 1e-4)
        counts = valid.reshape(valid.shape[0], -1).sum(dim=1).to(torch.long)
        offsets = torch.zeros((int(counts.numel()) + 1,), dtype=torch.long)
        offsets[1:] = torch.cumsum(counts, dim=0)
        return offsets

    # ------------------------------------------------------------------ #
    def _build_cameras_dggt(self, payload: dict[str, Any], subset: torch.Tensor) -> dict[str, torch.Tensor]:
        cams = payload["pass1"]["cameras_dggt"]
        return {
            "viewmats": cams["viewmats"].index_select(0, subset).unsqueeze(0),
            "Ks": cams["Ks"].index_select(0, subset).unsqueeze(0),
            "camera_to_world": cams["camera_to_world"].index_select(0, subset).unsqueeze(0),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _subset_phase1_localized(
        payload: dict[str, Any],
        subset: torch.Tensor,
        num_views: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Subset cached phase1_localized to the chosen frame subset.

        Two outputs:
          (a) Filtered ``slot_idx`` / ``frame_idx`` / ``source_front_index``
              for ``build_phase1_asset_coverage``. ``frame_idx`` is remapped
              to the subset-local index; with ``num_views=1``,
              ``source_front_index`` is the same.
          (b) ``delete_mask`` / ``shell_mask`` rebuilt for the subsetted
              clean_state Gaussian set.  The cache stored the FULL-clip
              flat masks plus a per-frame CSR offset array; we cat the
              per-frame slices in the subset's order to match the subset
              ``clean_state.means`` layout.

        This function does not re-run ``resolve_editable_subset``.  That
        helper is subset-dependent, while v6 caches are generated once for the
        full 29-frame clip before the random 4-8 frame training subsequence is
        known.  The cache therefore preserves the full-clip edit decision and
        this reader only remaps/slices it to the sampled target frames.
        """
        slot = payload["slot_idx"]
        frame = payload["frame_idx"]
        sf = payload["source_front_index"]
        delete_mask_full = payload["delete_mask"]
        shell_mask_full = payload["shell_mask"]
        offsets = payload["frame_gauss_offsets"].to(torch.int64)

        subset_list = subset.tolist()
        remap = {int(f): i for i, f in enumerate(subset_list)}
        # Filter (slot, frame, source_front) entries to subset frames; remap.
        keep_mask = torch.zeros(slot.numel(), dtype=torch.bool)
        new_frame = torch.full((slot.numel(),), -1, dtype=torch.int32)
        new_sf = torch.full((slot.numel(),), -1, dtype=torch.int32)
        for i in range(slot.numel()):
            f = int(frame[i].item())
            if f in remap:
                keep_mask[i] = True
                new_frame[i] = int(remap[f])
                # views=1: source_front_index == frame_idx after remap.
                # views>1: source_front_index = frame*num_views + view_off.
                old_sf = int(sf[i].item())
                view_off = old_sf - f * int(num_views)
                new_sf[i] = int(remap[f]) * int(num_views) + view_off
        kept_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        slot_kept = slot.index_select(0, kept_idx)
        frame_kept = new_frame.index_select(0, kept_idx)
        sf_kept = new_sf.index_select(0, kept_idx)

        # Rebuild masks in subset frame order.
        delete_chunks: list[torch.Tensor] = []
        shell_chunks: list[torch.Tensor] = []
        for f in subset_list:
            s, e = int(offsets[int(f)].item()), int(offsets[int(f) + 1].item())
            delete_chunks.append(delete_mask_full[s:e])
            shell_chunks.append(shell_mask_full[s:e])
        delete_subset = (
            torch.cat(delete_chunks) if delete_chunks else torch.zeros(0, dtype=torch.bool)
        )
        shell_subset = (
            torch.cat(shell_chunks) if shell_chunks else torch.zeros(0, dtype=torch.bool)
        )

        return {
            "slot_idx": slot_kept,
            "frame_idx": frame_kept,           # subset-local
            "source_front_index": sf_kept,     # subset-local
            "delete_mask": delete_subset,      # bool [N_g_subset], aligned with clean_state.means
            "shell_mask": shell_subset,
        }

    @staticmethod
    def _subset_pass2_splatted_tok_low(
        payload: dict[str, Any],
        subset: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        """Dequantize cached post-blend ``splatted_tok_low`` and select frames.

        Returns a list of ``num_levels`` tensors, each shape
        ``[1, |subset|, P, C]`` — the format expected by
        ``FlowFeatureAssembler.forward(splatted_tok_low_cached=...)``.

        The cached tensor was splatted from the full clip's Gaussian set.  The
        ``index_select`` below slices target frames only; it deliberately does
        not reproduce live splatting on a subsetted source-Gaussian set.
        """
        sub = _dequantize_nplc_subset(
            data=payload["splatted_tok_low_int8"],     # [N_frames, P, L, C]
            scale=payload["splatted_tok_low_scale"],   # [N_frames, L]
            subset=subset,
            dtype=dtype,
        )
        return _split_nplc_levels(sub)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_alignment(payload: dict[str, Any]) -> Sim3Transform:
        d = payload.get("phase1_alignment") or {}
        if not d:
            return Sim3Transform(
                scale=1.0,
                rotation=torch.eye(3),
                translation=torch.zeros(3),
                mean_alignment_error=0.0,
            )
        return Sim3Transform(
            scale=float(d.get("scale", 1.0)),
            rotation=torch.tensor(d.get("rotation", torch.eye(3).tolist()), dtype=torch.float32),
            translation=torch.tensor(d.get("translation", [0.0, 0.0, 0.0]), dtype=torch.float32),
            mean_alignment_error=float(d.get("mean_alignment_error", 0.0)),
        )
