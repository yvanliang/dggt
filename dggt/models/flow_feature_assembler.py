"""FlowFeatureAssembler: compose the diffusion-model input tensor set from one
cached clip sample.

Given a sample produced by `WaymoFlowCacheDataset` (sample-level subset already
applied, pre-computed VGGT outputs included), the assembler runs the online
portion of the FlowDGGT pipeline:

    Phase 1: build CleanSceneState from cached heads, localize + apply_mode_a
    Phase 4 (cache reconstruction): pick editable subset, build asset LUT/ptr dicts
    Phase 2: FeatureSplatter → splatted_tok_low
    Phase 3: SoftMaskBuilder + ScaffoldPacker → soft masks, scaffold
    Phase 5 precursor: tokenizer.encode → z_clean, z_splat
    Phase 6 input prep: PerTokenNoiseScheduler → t_tok, z_init

The result is a `FlowFeatureBundle` dataclass consumed by training scripts and
inference dumpers. Aggregator is NEVER invoked here — this module is strictly
the "online part" that follows offline caching.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dggt.models.asset_pass import AssetPassResult
from dggt.models.feature_splatter import FeatureSplatter
from dggt.models.gaussian_pointers import (
    GaussianPointers,
    SCENE_OBJECT_ID,
    SRC_KIND_ASSET,
    SRC_KIND_SCENE,
)
from dggt.models.gaussian_scene_editor import (
    EditedSceneBundle,
    GaussianSceneEditor,
)
from dggt.models.per_token_noise import PerTokenNoiseScheduler
from dggt.models.scaffold import ScaffoldPacker
from dggt.models.scene_pointers import build_scene_pointers, concat_pointers
from dggt.models.soft_mask import SoftMaskBuilder
from dggt.utils.edit_coverage import build_phase1_asset_coverage
from dggt.utils.gaussian_edit import (
    CleanSceneState,
    Sim3Transform,
    estimate_scene_alignment,
    apply_sim3_to_gaussian_dict,
    parse_object_slots,
)


DEFAULT_LEVELS = (4, 11, 17, 23)


@dataclass
class FlowFeatureBundle:
    """Every tensor SceneFlowMatching (and viz) consumes."""

    # Scene editing outputs (Phase 1)
    edit_bundle: EditedSceneBundle
    alignment: Sim3Transform

    # Asset pass outputs (from cache, masked by Phase-1 coverage)
    asset_pass_result: AssetPassResult
    phase1_coverage: torch.Tensor                              # [M, S] bool
    phase4_slots: list[int]

    # Pointers (all Gaussians together)
    pointers_scene: GaussianPointers
    pointers_asset_by_obj: dict[int, GaussianPointers]
    pointers_all: GaussianPointers
    gaussians_all_dggt: dict[str, torch.Tensor]
    N_scene: int

    # Camera frames
    cameras_dggt: dict[str, torch.Tensor]                      # viewmats [B,S,4,4], Ks [B,S,3,3]

    # Scene / asset LUTs (4 levels each) for FeatureSplatter + tokenizer.encode
    F_g_lut_scene: list[torch.Tensor]                          # len=4, each [B, S, P, 3072]
    F_g_lut_asset: dict[int, list[torch.Tensor]]               # obj_key -> len=4

    # Phase 2 output
    splatted_tok_low: list[torch.Tensor]                       # len=4, each [B, S, P, 3072]

    # Phase 3 outputs
    K_map: torch.Tensor                                        # [B, S, H, W, 1]
    D_map: torch.Tensor
    I_map: torch.Tensor
    I_map_per_obj: list[dict[int, torch.Tensor]]
    M_preserve: torch.Tensor                                   # [B, S, P, 1]
    M_source: torch.Tensor
    M_dest: torch.Tensor
    scaffold_hires: torch.Tensor                               # [B, S, H, W, 7]
    scaffold_tok: torch.Tensor                                 # [B, S, P, 768]

    # Phase 5/6 inputs
    z_clean: torch.Tensor                                      # [B, S, P, 768]
    z_splat: torch.Tensor
    z_init: torch.Tensor
    eps_noise: torch.Tensor
    t_tok: torch.Tensor                                        # [B, S, P, 1]
    base_t: torch.Tensor                                       # [B]

    # Flow cross-attn K/V (asset tokens flattened)
    F_asset_tokens: torch.Tensor                               # [B, sum_k(S * P), 3072]

    # Metadata mirrors
    patch_grid: tuple[int, int]
    patch_start_idx: int
    extras: dict[str, Any] = field(default_factory=dict)


def _dataclass_to_dict(gauss: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in gauss.items()}


def _empty_gauss_dict(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "means": torch.zeros((0, 3), dtype=torch.float32, device=device),
        "colors": torch.zeros((0, 3), dtype=torch.float32, device=device),
        "opacities": torch.zeros((0, 1), dtype=torch.float32, device=device),
        "scales": torch.zeros((0, 3), dtype=torch.float32, device=device),
        "quats": torch.zeros((0, 4), dtype=torch.float32, device=device),
    }


def _concat_gauss_dicts(
    chunks: Sequence[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(chunks) == 0:
        return _empty_gauss_dict(device)
    out: dict[str, torch.Tensor] = {}
    for key in ("means", "colors", "opacities", "scales", "quats"):
        tensors = [c[key].to(device) for c in chunks if c[key].numel() > 0]
        if tensors:
            out[key] = torch.cat(tensors, dim=0)
        else:
            out[key] = torch.zeros_like(chunks[0][key]).to(device)
    return out


class _LiteLocalizedObject:
    """Minimal duck-typed stand-in for ``LocalizedFrameObject``.

    Cache schema v6 only persists the three fields downstream consumers
    actually read (``slot_idx``, ``frame_idx``, ``source_front_index`` for
    ``build_phase1_asset_coverage``).  The OR'd ``delete_mask`` /
    ``shell_mask`` arrive as separate top-level tensors — apply_mode_a is
    bypassed entirely on the fast path, so we no longer hydrate per-entry
    delete/shell index lists.
    """

    __slots__ = ("slot_idx", "frame_idx", "source_front_index")

    def __init__(self, slot_idx: int, frame_idx: int, source_front_index: int) -> None:
        self.slot_idx = int(slot_idx)
        self.frame_idx = int(frame_idx)
        self.source_front_index = int(source_front_index)


def _hydrate_lite_localized(
    payload: dict[str, torch.Tensor],
) -> list[_LiteLocalizedObject]:
    """Reconstruct meta-only ``localized_objects`` list from v6 cache payload."""
    slot = payload["slot_idx"]
    frame = payload["frame_idx"]
    sf = payload["source_front_index"]
    return [
        _LiteLocalizedObject(
            slot_idx=int(slot[i].item()),
            frame_idx=int(frame[i].item()),
            source_front_index=int(sf[i].item()),
        )
        for i in range(int(slot.numel()))
    ]


def _flatten_asset_kv(
    F_g_lut_asset_by_obj: dict[int, list[torch.Tensor]],
    last_level: int = -1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Concatenate per-object asset tokens along patch axis for cross-attn K/V.

    Each `F_g_lut_asset[k][l]` has shape `[B, S, P, C]`. We pick one level
    (by default the last), flatten `(S, P)`, and concat all objects along the
    token axis → `[B, sum_k(S * P), C]`.
    """
    flat_chunks: list[torch.Tensor] = []
    for obj_key in sorted(F_g_lut_asset_by_obj.keys()):
        lvl = F_g_lut_asset_by_obj[obj_key][last_level]  # [B, S, P, C]
        if device is not None:
            lvl = lvl.to(device)
        if lvl.dim() != 4:
            continue
        B, S, P, C = lvl.shape
        flat_chunks.append(lvl.reshape(B, S * P, C))
    if len(flat_chunks) == 0:
        # Fallback to a zero-width tensor to keep downstream shape checks happy.
        return torch.zeros((1, 0, 3072), dtype=torch.float32, device=device)
    return torch.cat(flat_chunks, dim=1)


class FlowFeatureAssembler(nn.Module):
    """Orchestrator for the online part of FlowDGGT.

    Parameters mirror `SoftMaskBuilder`, `ScaffoldPacker`, `FeatureSplatter` and
    `PerTokenNoiseScheduler` defaults from research_plan.md §3.3–3.6.
    """

    def __init__(
        self,
        scene_tokenizer: nn.Module | None = None,
        channels: int = 3072,
        chunk_channels: int = 512,
        num_levels: int = 4,
        patch_grid: tuple[int, int] = (37, 37),
        patch_start_idx: int = 5,
        H_splat: int = 148,
        W_splat: int = 148,
        scaffold_out_dim: int = 768,
        gamma_dest: float = 0.4,
        eps_floor: float = 0.05,
        sigma_partial: float = 0.3,
        preserve_token_bypass: bool = True,
        asset_token_direct_blend: bool = True,
        asset_token_full_alpha: float = 0.5,
        unedited_preserve_threshold: float = 1e-4,
        editor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.scene_tokenizer = scene_tokenizer
        self.channels = int(channels)
        self.num_levels = int(num_levels)
        self.patch_grid = (int(patch_grid[0]), int(patch_grid[1]))
        self.patch_start_idx = int(patch_start_idx)
        self.H_splat = int(H_splat)
        self.W_splat = int(W_splat)
        self.preserve_token_bypass = bool(preserve_token_bypass)
        self.asset_token_direct_blend = bool(asset_token_direct_blend)
        self.asset_token_full_alpha = float(asset_token_full_alpha)
        self.unedited_preserve_threshold = float(unedited_preserve_threshold)

        self.editor = GaussianSceneEditor(**(editor_kwargs or {}))
        self.feature_splatter = FeatureSplatter(
            channels=self.channels,
            chunk_channels=int(chunk_channels),
            num_levels=self.num_levels,
            patch_grid=self.patch_grid,
        )
        self.soft_mask = SoftMaskBuilder()
        self.scaffold_packer = ScaffoldPacker(
            in_channels=7, out_dim=int(scaffold_out_dim)
        )
        self.noise_scheduler = PerTokenNoiseScheduler(
            gamma_dest=gamma_dest,
            eps_floor=eps_floor,
            sigma_partial=sigma_partial,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        sample: dict[str, Any],
        predictions: dict[str, torch.Tensor],
        asset_pass_result: AssetPassResult,
        cameras_dggt: dict[str, torch.Tensor],
        object_slots_spec: str | list[int] = "all",
        base_t: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        mode_kind: str | None = None,
        mode_b: dict[str, Any] | None = None,
        phase1_localized_lite: dict[str, torch.Tensor] | None = None,
        splatted_tok_low_cached: list[torch.Tensor] | None = None,
    ) -> FlowFeatureBundle:
        """Assemble the full diffusion-input bundle.

        Parameters
        ----------
        sample
            Frame-subset sample from `WaymoFlowCacheDataset.__getitem__`;
            shape conventions identical to `WaymoEditDataset.__getitem__`.
        predictions
            Dict with cached VGGT outputs: `pose_enc, depth, gs_map, dynamic_conf,
            gs_conf, semantic_logits` and the 24-layer `image_tokens_list` (or
            at least the 4 levels we need as `image_tokens_levels`, keyed in
            order `DEFAULT_LEVELS`).
        asset_pass_result
            `AssetPassResult` reconstructed from cache. Mode A requires fitted
            DGGT-coordinate asset Gaussians produced during offline precompute.
        cameras_dggt
            `{"viewmats": [B,S,4,4], "Ks": [B,S,3,3]}` in DGGT coords. Taken
            from the cache's `pass1_cameras_dggt` unchanged.
        object_slots_spec
            "all" / comma-separated slot ids / explicit list. Passed to
            `parse_object_slots`.
        base_t
            Optional `[B]` tensor; if None, sampled ~ U(0,1) from `generator`.
        mode_kind
            "mode_a" or "mode_b". If None, taken from `sample["mode_kind"]`
            (default "mode_a").
        mode_b
            Mode B payload (the `mode_b` block emitted by the cache dataset).
            Required when `mode_kind == "mode_b"`.
        """
        if device is None:
            device = sample["images_clean"].device
        device = torch.device(device)
        if mode_kind is None:
            mode_kind = str(sample.get("mode_kind", "mode_a"))

        if mode_kind == "mode_b":
            return self._forward_mode_b(
                sample=sample,
                predictions=predictions,
                cameras_dggt=cameras_dggt,
                mode_b=mode_b or {},
                base_t=base_t,
                generator=generator,
                device=device,
                splatted_tok_low_cached=splatted_tok_low_cached,
            )

        # ----- Mode A path ----- #
        self._validate_mode_a_asset_pass(asset_pass_result)
        # Phase 1: build clean state + align + localize (load_asset=False) + apply
        clean_state = self.editor.build_clean_bundle(sample, predictions)
        alignment = self.editor.align(sample, clean_state)
        object_slots = (
            parse_object_slots(sample, object_slots_spec)
            if isinstance(object_slots_spec, str)
            else [int(s) for s in object_slots_spec]
        )
        if phase1_localized_lite is not None:
            # Schema v6 fast path: skip the 67s editor.localize pose-refine,
            # skip apply_mode_a, build edit_state directly from cached masks.
            from dggt.utils.gaussian_edit import EditedSceneState
            localized_objects = _hydrate_lite_localized(phase1_localized_lite)
            cached_delete = phase1_localized_lite["delete_mask"].to(device).bool()
            cached_shell = phase1_localized_lite["shell_mask"].to(device).bool()
            n_g_clean = int(clean_state.means.shape[0])
            if int(cached_delete.numel()) != n_g_clean:
                raise RuntimeError(
                    f"phase1_localized_lite.delete_mask length ({int(cached_delete.numel())}) "
                    f"does not match clean_state.means count ({n_g_clean}). The cache and the "
                    "current frame subset disagree on the per-Gaussian Gaussian count."
                )
            clean_dict = {
                "means": clean_state.means,
                "colors": clean_state.colors,
                "opacities": clean_state.opacities,
                "scales": clean_state.scales,
                "quats": clean_state.quats,
            }
            clean_dict_dev = {k: v.to(device) for k, v in clean_dict.items()}
            keep_mask_cpu = (~cached_delete).cpu()
            deleted_dict = {k: v[keep_mask_cpu] for k, v in clean_dict.items()}
            edit_state = EditedSceneState(
                clean=clean_dict,
                deleted=deleted_dict,
                asset_only={k: v[:0] for k, v in clean_dict.items()},  # empty placeholder
                edited=deleted_dict,
                localized_objects=localized_objects,
                delete_mask=cached_delete.cpu(),
                shell_mask=cached_shell.cpu(),
            )
        else:
            localized_objects = self.editor.localize(
                sample,
                clean_state,
                alignment,
                object_slots,
                load_asset=False,
            )
            edit_state = self.editor.apply_mode_a(clean_state, localized_objects)

        # Phase 1 asset coverage gating for the asset pass.
        phase1_coverage, phase4_slots = build_phase1_asset_coverage(
            sample["object_asset_image_valid_mask_selected"],
            localized_objects,
        )

        # Mask cached asset pass to the Phase-1 coverage subset.
        asset_pass_result = self._mask_asset_pass_by_coverage(
            asset_pass_result=asset_pass_result,
            phase1_coverage=phase1_coverage,
            phase4_slots=phase4_slots,
            device=device,
        )

        # Assemble EditedSceneBundle for downstream consumers (viz, loss masks).
        clean_images = clean_state.images
        H_img, W_img = int(clean_images.shape[-2]), int(clean_images.shape[-1])
        g_kept = _dataclass_to_dict(edit_state.clean)
        g_kept_masked = {
            k: v[~edit_state.delete_mask] for k, v in edit_state.clean.items()
        }
        g_deleted_masked = {
            k: v[edit_state.delete_mask] for k, v in edit_state.clean.items()
        }

        edit_bundle = EditedSceneBundle(
            clean_state=clean_state,
            alignment=alignment,
            edited_state=edit_state,
            cameras_dggt=cameras_dggt,
            cameras_waymo=asset_pass_result.cameras_waymo,
            T_w2d=alignment,
            G_kept=g_kept_masked,
            G_deleted=g_deleted_masked,
            G_asset_per_object={
                int(k): _concat_gauss_dicts(
                    asset_pass_result.G_asset_dggt[int(k)],
                    device=device,
                )
                for k in asset_pass_result.object_keys
                if asset_pass_result.G_asset_dggt is not None
            },
            per_gauss_pointers=None,
            edit_meta={
                "delete_mask": edit_state.delete_mask,
                "shell_mask": edit_state.shell_mask,
                "phase4_slots": phase4_slots,
            },
        )

        # Build pointers
        ptr_scene = build_scene_pointers(
            clean_state.source_image_ids,
            clean_state.source_y,
            clean_state.source_x,
            patch_size=int(H_img // self.patch_grid[0]),
            patch_grid=self.patch_grid,
        ).to(device)
        pointers_asset_by_obj: dict[int, GaussianPointers] = {}
        asset_gauss_chunks: list[dict[str, torch.Tensor]] = []
        ptr_chunks: list[GaussianPointers] = [ptr_scene]
        for obj_key in asset_pass_result.object_keys:
            obj_ptrs = asset_pass_result.ptr_asset[int(obj_key)]
            obj_gauss_frames = asset_pass_result.G_asset_dggt[int(obj_key)]
            # Concat per-frame Gaussians → per-object Gaussian cloud (for splatter).
            # Pointers are per-frame (each frame's Gaussians have view_n=image_idx already)
            # so just concat along N dim.
            per_frame_gauss_cat = [
                {k: v.to(device) for k, v in g.items()} for g in obj_gauss_frames
            ]
            flat_gauss = _concat_gauss_dicts(per_frame_gauss_cat, device=device)
            asset_gauss_chunks.append(flat_gauss)
            flat_ptr = GaussianPointers(
                src_kind=torch.cat([p.src_kind for p in obj_ptrs]),
                object_id=torch.cat([p.object_id for p in obj_ptrs]),
                view_n=torch.cat([p.view_n for p in obj_ptrs]),
                patch_idx=torch.cat([p.patch_idx for p in obj_ptrs]),
                visible_mask=torch.cat([p.visible_mask for p in obj_ptrs]),
            )
            pointers_asset_by_obj[int(obj_key)] = flat_ptr
            ptr_chunks.append(flat_ptr)

        # Scene Gaussians (post-edit: kept only) + all asset Gaussians, concat.
        gauss_scene = {k: v.to(device) for k, v in edit_state.clean.items()}
        gauss_scene_kept = {k: v[~edit_state.delete_mask] for k, v in gauss_scene.items()}
        # Filter scene pointers by the kept mask
        keep_mask = (~edit_state.delete_mask).to(device)
        ptr_scene_kept = GaussianPointers(
            src_kind=ptr_scene.src_kind[keep_mask],
            object_id=ptr_scene.object_id[keep_mask],
            view_n=ptr_scene.view_n[keep_mask],
            patch_idx=ptr_scene.patch_idx[keep_mask],
            visible_mask=ptr_scene.visible_mask[keep_mask],
        )
        ptr_chunks[0] = ptr_scene_kept
        pointers_all = concat_pointers(ptr_chunks)
        gaussians_all = _concat_gauss_dicts([gauss_scene_kept] + asset_gauss_chunks, device=device)

        # F_g_lut_scene from cached 4-level image tokens (already subsetted to S frames)
        F_g_lut_scene = self._select_lut_scene(predictions)
        F_g_lut_asset = asset_pass_result.F_g_lut_asset

        # ------------------- Phase 2: FeatureSplatter -------------------- #
        B = int(F_g_lut_scene[0].shape[0])
        S = int(F_g_lut_scene[0].shape[1])
        cameras_splat = self.scale_cameras_for_render(
            cameras_dggt,
            source_hw=(H_img, W_img),
            target_hw=(self.H_splat, self.W_splat),
        )

        # ------------------- Phase 3: Soft masks + Scaffold -------------- #
        # NOTE: soft_mask still runs in the cached path because M_preserve /
        # M_source / M_dest are first-class WAN inputs (not just used for
        # blending). They are cheap (~36ms total) so caching them isn't worth
        # the extra disk.
        K_map, D_map, I_map, I_per_obj = self._render_mode_a_per_target_coverage(
            sample=sample,
            clean_state=clean_state,
            clean_dict=gauss_scene,
            keep_mask=keep_mask,
            delete_mask=edit_state.delete_mask.to(device),
            asset_pass_result=asset_pass_result,
            cameras_dggt=cameras_dggt,
            H=H_img,
            W=W_img,
        )
        M_preserve, M_source, M_dest = self.soft_mask.pool_and_normalize(
            K_map, D_map, I_map, target_grid=self.patch_grid
        )
        M_preserve, M_source, M_dest = self._force_preserve_unedited_tokens(
            K_map=K_map,
            D_map=D_map,
            I_map=I_map,
            M_preserve=M_preserve,
            M_source=M_source,
            M_dest=M_dest,
        )
        if splatted_tok_low_cached is None:
            splatted_tok_low = self._splat_mode_a_per_target(
                sample=sample,
                clean_state=clean_state,
                clean_dict=gauss_scene,
                keep_mask=keep_mask,
                ptr_scene=ptr_scene,
                asset_pass_result=asset_pass_result,
                lut_scene=F_g_lut_scene,
                lut_asset_dict=F_g_lut_asset,
                cameras_splat=cameras_splat,
                tile_masks=None,
            )
            splatted_tok_low = self._blend_preserve_tokens(
                clean_levels=F_g_lut_scene,
                splatted_levels=splatted_tok_low,
                M_preserve=M_preserve,
            )
            splatted_tok_low = self._blend_asset_tokens(
                splatted_levels=splatted_tok_low,
                F_g_lut_asset=F_g_lut_asset,
                I_map_per_obj=I_per_obj,
            )
        else:
            # Schema v6 fast path: post-blend splatted_tok_low is cached.
            # tokenizer.encode below still runs live so z_clean / z_splat use
            # the latest tokenizer weights.
            splatted_tok_low = [
                t.to(device=F_g_lut_scene[0].device, dtype=F_g_lut_scene[0].dtype)
                for t in splatted_tok_low_cached
            ]
        # Cached path already stores the post-blend splatted_tok_low.

        # Scaffold precursors
        D_edited_hires = D_map.new_zeros((B, S, H_img, W_img, 1))
        A_edited_hires = (K_map + I_map).clamp(0.0, 1.0)
        dyn_prior = torch.sigmoid(
            predictions["dynamic_conf"].reshape(B, S, H_img, W_img, 1).to(device)
        ).float()
        time_index = torch.arange(S, dtype=torch.float32, device=device).view(1, S).expand(B, S)
        time_index = time_index / max(S - 1, 1)
        scaffold_hires = ScaffoldPacker.build_scaffold_hires(
            D_edited=D_edited_hires,
            A_edited=A_edited_hires,
            K_map=K_map,
            D_map=D_map,
            I_map=I_map,
            dynamic_prior=dyn_prior,
            time_index=time_index,
        )
        scaffold_tok = self.scaffold_packer(scaffold_hires, target_grid=self.patch_grid)

        # ------------------- Phase 5/6 inputs: z_clean, z_splat --------- #
        if self.scene_tokenizer is None:
            raise RuntimeError(
                "FlowFeatureAssembler needs `scene_tokenizer` to produce z_clean/z_splat. "
                "Pass it via the constructor (or VGGT.scene_tokenizer)."
            )
        # tokenizer.encode runs online for BOTH branches so z_clean and z_splat
        # are produced with the same (current) tokenizer weights.  The cache
        # only stores the *input* to tokenizer.encode (splatted_tok_low); the
        # encode itself (~38ms) is cheap and must reflect live weights.
        z_clean = self.scene_tokenizer.encode(F_g_lut_scene, patch_grid=self.patch_grid)
        z_splat = self.scene_tokenizer.encode(splatted_tok_low, patch_grid=self.patch_grid)
        if z_splat.shape != z_clean.shape:
            raise ValueError(
                f"z_splat shape {tuple(z_splat.shape)} does not match "
                f"z_clean shape {tuple(z_clean.shape)}"
            )

        if base_t is None:
            base_t = self.noise_scheduler.sample_base_t(B, device=device, generator=generator)
        else:
            base_t = base_t.to(device)
        t_tok = self.noise_scheduler.build_t_tok(base_t, M_preserve, M_source, M_dest)
        z_init, eps_noise = self.noise_scheduler.compose_z_init(
            z_clean, z_splat, M_preserve, M_source, M_dest, generator=generator
        )

        F_asset_tokens = _flatten_asset_kv(F_g_lut_asset, device=device)

        return FlowFeatureBundle(
            edit_bundle=edit_bundle,
            alignment=alignment,
            asset_pass_result=asset_pass_result,
            phase1_coverage=phase1_coverage,
            phase4_slots=phase4_slots,
            pointers_scene=ptr_scene_kept,
            pointers_asset_by_obj=pointers_asset_by_obj,
            pointers_all=pointers_all,
            gaussians_all_dggt=gaussians_all,
            N_scene=int(gauss_scene_kept["means"].shape[0]),
            cameras_dggt=cameras_dggt,
            F_g_lut_scene=F_g_lut_scene,
            F_g_lut_asset=F_g_lut_asset,
            splatted_tok_low=splatted_tok_low,
            K_map=K_map,
            D_map=D_map,
            I_map=I_map,
            I_map_per_obj=I_per_obj,
            M_preserve=M_preserve,
            M_source=M_source,
            M_dest=M_dest,
            scaffold_hires=scaffold_hires,
            scaffold_tok=scaffold_tok,
            z_clean=z_clean,
            z_splat=z_splat,
            z_init=z_init,
            eps_noise=eps_noise,
            t_tok=t_tok,
            base_t=base_t,
            F_asset_tokens=F_asset_tokens,
            patch_grid=self.patch_grid,
            patch_start_idx=self.patch_start_idx,
            extras={
                "mode_kind": "mode_a",
                "object_slots_requested": object_slots,
                "localized_objects": localized_objects,
            },
        )

    # ------------------------------------------------------------------ #
    # Mode B path                                                         #
    # ------------------------------------------------------------------ #
    def _forward_mode_b(
        self,
        sample: dict[str, Any],
        predictions: dict[str, torch.Tensor],
        cameras_dggt: dict[str, torch.Tensor],
        mode_b: dict[str, Any],
        base_t: torch.Tensor | None,
        generator: torch.Generator | None,
        device: torch.device,
        splatted_tok_low_cached: list[torch.Tensor] | None = None,
    ) -> FlowFeatureBundle:
        """Mode B forward: pseudo-deletion only, no asset features.

        Mode B treats the planner-imagined region as M_dest (where the
        diffusion model must hallucinate new content) with NO conditioning
        asset (`F_asset_tokens` is shape `[B, 0, C]`). M_source = 0 because
        we are not asking for background completion of a deleted Waymo object.
        """
        # Phase 1 (clean only — no localize/apply_mode_a).
        clean_state = self.editor.build_clean_bundle(sample, predictions)
        alignment = self.editor.align(sample, clean_state)

        clean_images = clean_state.images
        H_img, W_img = int(clean_images.shape[-2]), int(clean_images.shape[-1])

        # Pseudo-delete masks from cache. Prefer the per-target-frame masks so
        # the diffusion edit footprint matches the rendered Mode-B hole.
        delete_mask = mode_b.get("delete_mask")
        if delete_mask is None:
            raise RuntimeError("Mode B forward requires `mode_b.delete_mask`.")
        delete_mask = delete_mask.to(torch.bool)
        n_g = int(clean_state.means.shape[0])
        if int(delete_mask.numel()) != n_g:
            # Defensive: if Gaussian count drifted (it shouldn't — Pass1 is
            # deterministic), fall back to "delete nothing" to keep training going.
            delete_mask = torch.zeros((n_g,), dtype=torch.bool)
        clean_dict = {
            "means": clean_state.means.to(device),
            "colors": clean_state.colors.to(device),
            "opacities": clean_state.opacities.to(device).view(-1, 1)
                if clean_state.opacities.dim() == 1 else clean_state.opacities.to(device),
            "scales": clean_state.scales.to(device),
            "quats": clean_state.quats.to(device),
        }

        # Pointers (scene only). Per-target-frame deletion is applied later
        # because each target frame can remove a different subset of Gaussians.
        ptr_scene = build_scene_pointers(
            clean_state.source_image_ids,
            clean_state.source_y,
            clean_state.source_x,
            patch_size=int(H_img // self.patch_grid[0]),
            patch_grid=self.patch_grid,
        ).to(device)

        F_g_lut_scene = self._select_lut_scene(predictions)
        B = int(F_g_lut_scene[0].shape[0])
        S = int(F_g_lut_scene[0].shape[1])
        delete_masks_by_target = self._mode_b_delete_masks_by_target(
            mode_b=mode_b,
            delete_mask=delete_mask,
            S=S,
            n_g=n_g,
            device=device,
        )
        delete_mask_union = delete_masks_by_target.any(dim=0)
        mode_b_noop = not bool(delete_mask_union.any().item())
        keep_mask_union = ~delete_mask_union
        gauss_kept = {k: v[keep_mask_union] for k, v in clean_dict.items()}
        gauss_imagined = {k: v[delete_mask_union] for k, v in clean_dict.items()}
        ptr_scene_kept = GaussianPointers(
            src_kind=ptr_scene.src_kind[keep_mask_union],
            object_id=ptr_scene.object_id[keep_mask_union],
            view_n=ptr_scene.view_n[keep_mask_union],
            patch_idx=ptr_scene.patch_idx[keep_mask_union],
            visible_mask=ptr_scene.visible_mask[keep_mask_union],
        )
        pointers_all = ptr_scene_kept

        # Phase 3: Soft masks. K = render(per-frame kept),
        # I = render(per-frame pseudo-deleted Gaussians), D = 0.
        if mode_b_noop:
            K_map = torch.ones((B, S, H_img, W_img, 1), dtype=torch.float32, device=device)
            D_map = torch.zeros_like(K_map)
            I_map = torch.zeros_like(K_map)
            I_per_obj: list[dict[int, torch.Tensor]] = [{} for _ in range(B)]
            M_preserve = torch.ones(
                (B, S, self.patch_grid[0] * self.patch_grid[1], 1),
                dtype=torch.float32,
                device=device,
            )
            M_source = torch.zeros_like(M_preserve)
            M_dest = torch.zeros_like(M_preserve)
        else:
            K_map, D_map, I_map, I_per_obj = self._render_mode_b_per_target_coverage(
                sample=sample,
                clean_state=clean_state,
                clean_dict=clean_dict,
                delete_masks_by_target=delete_masks_by_target,
                cameras_dggt=cameras_dggt,
                H=H_img,
                W=W_img,
            )
            M_preserve, M_source, M_dest = self.soft_mask.pool_and_normalize(
                K_map, D_map, I_map, target_grid=self.patch_grid
            )
            M_preserve, M_source, M_dest = self._force_preserve_unedited_tokens(
                K_map=K_map,
                D_map=D_map,
                I_map=I_map,
                M_preserve=M_preserve,
                M_source=M_source,
                M_dest=M_dest,
            )

        # Phase 2: Splat scene tokens onto per-frame kept Gaussians only.
        if mode_b_noop:
            # No imagined/deleted region means Mode B is a true no-op for this
            # frame subset. Re-splatting the clean scene would still alter
            # low-coverage/sky tokens through K/(K+eps), so bypass it exactly.
            splatted_tok_low = [t.to(device=device) for t in F_g_lut_scene]
        elif splatted_tok_low_cached is None:
            cameras_splat = self.scale_cameras_for_render(
                cameras_dggt,
                source_hw=(H_img, W_img),
                target_hw=(self.H_splat, self.W_splat),
            )
            active_tile_masks = self._splat_weight_to_tile_masks(
                (1.0 - M_preserve).clamp(0.0, 1.0),
                threshold=1e-3,
                H_splat=self.H_splat,
                W_splat=self.W_splat,
            )
            splatted_tok_low = self._splat_mode_b_per_target(
                sample=sample,
                clean_state=clean_state,
                clean_dict=clean_dict,
                ptr_scene=ptr_scene,
                delete_masks_by_target=delete_masks_by_target,
                lut_scene=F_g_lut_scene,
                cameras_splat=cameras_splat,
                tile_masks=active_tile_masks,
            )
            splatted_tok_low = self._blend_preserve_tokens(
                clean_levels=F_g_lut_scene,
                splatted_levels=splatted_tok_low,
                M_preserve=M_preserve,
            )
        else:
            # Schema v6 fast path: post-blend splatted_tok_low is cached.
            # tokenizer.encode below still runs live so z_clean / z_splat use
            # the latest tokenizer weights.
            splatted_tok_low = [
                t.to(device=F_g_lut_scene[0].device, dtype=F_g_lut_scene[0].dtype)
                for t in splatted_tok_low_cached
            ]
        # Cached path already stores the post-blend splatted_tok_low.

        # Scaffold (D_edited not meaningful for mode B — pass zeros).
        D_edited_hires = D_map.new_zeros((B, S, H_img, W_img, 1))
        A_edited_hires = (K_map + I_map).clamp(0.0, 1.0)
        dyn_prior = torch.sigmoid(
            predictions["dynamic_conf"].reshape(B, S, H_img, W_img, 1).to(device)
        ).float()
        time_index = torch.arange(S, dtype=torch.float32, device=device).view(1, S).expand(B, S)
        time_index = time_index / max(S - 1, 1)
        scaffold_hires = ScaffoldPacker.build_scaffold_hires(
            D_edited=D_edited_hires,
            A_edited=A_edited_hires,
            K_map=K_map,
            D_map=D_map,
            I_map=I_map,
            dynamic_prior=dyn_prior,
            time_index=time_index,
        )
        scaffold_tok = self.scaffold_packer(scaffold_hires, target_grid=self.patch_grid)

        if self.scene_tokenizer is None:
            raise RuntimeError(
                "FlowFeatureAssembler needs `scene_tokenizer` to produce z_clean/z_splat."
            )
        # tokenizer.encode runs online for both branches so z_clean and z_splat
        # share the latest tokenizer weights.  The cache only stores the input
        # to tokenizer.encode (post-blend splatted_tok_low).
        z_clean = self.scene_tokenizer.encode(F_g_lut_scene, patch_grid=self.patch_grid)
        z_splat = z_clean if mode_b_noop else self.scene_tokenizer.encode(
            splatted_tok_low, patch_grid=self.patch_grid
        )
        if z_splat.shape != z_clean.shape:
            raise ValueError(
                f"z_splat shape {tuple(z_splat.shape)} does not match "
                f"z_clean shape {tuple(z_clean.shape)} (Mode B)"
            )

        if base_t is None:
            base_t = self.noise_scheduler.sample_base_t(B, device=device, generator=generator)
        else:
            base_t = base_t.to(device)
        t_tok = self.noise_scheduler.build_t_tok(base_t, M_preserve, M_source, M_dest)
        z_init, eps_noise = self.noise_scheduler.compose_z_init(
            z_clean, z_splat, M_preserve, M_source, M_dest, generator=generator
        )

        F_asset_tokens = torch.zeros(
            (B, 0, self.channels), dtype=z_clean.dtype, device=z_clean.device
        )

        # Build a slim AssetPassResult / EditedSceneBundle so downstream consumers
        # (viz, loss) work without a Mode A path.
        asset_pass_empty = AssetPassResult(
            patch_grid=self.patch_grid,
            patch_start_idx=self.patch_start_idx,
            object_keys=[],
            cameras_waymo={},
            F_g_lut_asset={},
            ptr_asset={},
            G_asset_waymo={},
            G_asset_dggt={},
            I_asset={},
            A_asset={},
            asset_pass_space="mode_b_empty",
            fit_metrics={},
        )
        from dggt.utils.gaussian_edit import EditedSceneState

        edit_state_proxy = EditedSceneState(
            clean=clean_dict,
            deleted=gauss_imagined,
            asset_only={},
            edited=gauss_kept,
            localized_objects=[],
            delete_mask=delete_mask_union.to(device),
            shell_mask=delete_mask_union.to(device),
        )
        edit_bundle = EditedSceneBundle(
            clean_state=clean_state,
            alignment=alignment,
            edited_state=edit_state_proxy,
            cameras_dggt=cameras_dggt,
            cameras_waymo={},
            T_w2d=alignment,
            G_kept=gauss_kept,
            G_deleted=gauss_imagined,
            G_asset_per_object={},
            per_gauss_pointers=None,
            edit_meta={
                "delete_mask": delete_mask_union.detach().cpu(),
                "delete_mask_per_frame": delete_masks_by_target.detach().cpu(),
                "shell_mask": delete_mask_union.detach().cpu(),
                "phase4_slots": [],
                "mode_kind": "mode_b",
                "imagined_objects": list(mode_b.get("imagined_objects", [])),
            },
        )

        # phase1_coverage placeholder for downstream code that expects [M, S].
        phase1_coverage = torch.zeros((0, S), dtype=torch.bool)

        return FlowFeatureBundle(
            edit_bundle=edit_bundle,
            alignment=alignment,
            asset_pass_result=asset_pass_empty,
            phase1_coverage=phase1_coverage,
            phase4_slots=[],
            pointers_scene=ptr_scene_kept,
            pointers_asset_by_obj={},
            pointers_all=pointers_all,
            gaussians_all_dggt=gauss_kept,
            N_scene=int(gauss_kept["means"].shape[0]),
            cameras_dggt=cameras_dggt,
            F_g_lut_scene=F_g_lut_scene,
            F_g_lut_asset={},
            splatted_tok_low=splatted_tok_low,
            K_map=K_map,
            D_map=D_map,
            I_map=I_map,
            I_map_per_obj=I_per_obj,
            M_preserve=M_preserve,
            M_source=M_source,
            M_dest=M_dest,
            scaffold_hires=scaffold_hires,
            scaffold_tok=scaffold_tok,
            z_clean=z_clean,
            z_splat=z_splat,
            z_init=z_init,
            eps_noise=eps_noise,
            t_tok=t_tok,
            base_t=base_t,
            F_asset_tokens=F_asset_tokens,
            patch_grid=self.patch_grid,
            patch_start_idx=self.patch_start_idx,
            extras={
                "mode_kind": "mode_b",
                "imagined_objects": list(mode_b.get("imagined_objects", [])),
                "num_imagined_objects": int(mode_b.get("num_imagined_objects", 0)),
                "rejection_reason": str(mode_b.get("rejection_reason", "")),
                "delete_mask": delete_mask_union.detach().cpu(),
                "delete_mask_per_frame": delete_masks_by_target.detach().cpu(),
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mode_b_delete_masks_by_target(
        *,
        mode_b: dict[str, Any],
        delete_mask: torch.Tensor,
        S: int,
        n_g: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return target-frame delete masks as `[S, N_g]`.

        `mode_b.delete_mask` is a union mask kept for compatibility. New v6
        caches also carry per-target-frame masks; those are the authoritative
        edit footprint for Mode B.
        """
        for key in ("delete_mask_per_frame_subset", "delete_mask_per_frame"):
            per_frame = mode_b.get(key)
            if not torch.is_tensor(per_frame) or per_frame.numel() == 0:
                continue
            per_frame = per_frame.to(device=device, dtype=torch.bool)
            if per_frame.dim() != 2 or int(per_frame.shape[1]) != int(n_g):
                continue
            if int(per_frame.shape[0]) == int(S):
                return per_frame.contiguous()
            row_idx = torch.arange(int(S), device=device).clamp_max(int(per_frame.shape[0]) - 1)
            return per_frame.index_select(0, row_idx).contiguous()

        delete_mask = delete_mask.to(device=device, dtype=torch.bool)
        if int(delete_mask.numel()) != int(n_g):
            delete_mask = torch.zeros((int(n_g),), dtype=torch.bool, device=device)
        return delete_mask.view(1, int(n_g)).expand(int(S), int(n_g)).contiguous()

    def _render_mode_b_per_target_coverage(
        self,
        *,
        sample: dict[str, Any],
        clean_state: CleanSceneState,
        clean_dict: dict[str, torch.Tensor],
        delete_masks_by_target: torch.Tensor,
        cameras_dggt: dict[str, torch.Tensor],
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[int, torch.Tensor]]]:
        """Render Mode-B K/I coverage with the target frame's own delete mask."""
        viewmats = cameras_dggt["viewmats"]
        if int(viewmats.shape[0]) != 1:
            raise ValueError("Mode-B per-target coverage currently expects batch size 1.")
        S = int(viewmats.shape[1])
        if tuple(delete_masks_by_target.shape) != (S, int(clean_dict["means"].shape[0])):
            raise ValueError(
                "delete_masks_by_target must be [S,N_g], got "
                f"{tuple(delete_masks_by_target.shape)} for S={S}, N_g={int(clean_dict['means'].shape[0])}"
            )

        timestamps = self._mode_b_timestamps(sample, num_images=S, device=viewmats.device)
        K_chunks: list[torch.Tensor] = []
        I_chunks: list[torch.Tensor] = []
        for target_idx in range(S):
            del_mask = delete_masks_by_target[target_idx].to(
                device=clean_dict["means"].device, dtype=torch.bool
            )
            keep_mask = ~del_mask
            gauss_kept = self._mode_b_time_aware_gaussians_for_target(
                clean_state=clean_state,
                clean_dict=clean_dict,
                base_mask=keep_mask,
                target_idx=target_idx,
                timestamps=timestamps,
            )
            gauss_deleted = self._mode_b_time_aware_gaussians_for_target(
                clean_state=clean_state,
                clean_dict=clean_dict,
                base_mask=del_mask,
                target_idx=target_idx,
                timestamps=timestamps,
            )
            cameras_one = {
                "viewmats": cameras_dggt["viewmats"][:, target_idx : target_idx + 1].contiguous(),
                "Ks": cameras_dggt["Ks"][:, target_idx : target_idx + 1].contiguous(),
            }
            K_one, _D_dummy, I_one, _ = self.soft_mask.render_coverage(
                G_kept=[gauss_kept],
                G_deleted=[{}],
                G_asset_dggt_dict=[{0: gauss_deleted}],
                cameras_dggt=cameras_one,
                H=H,
                W=W,
            )
            K_chunks.append(K_one)
            I_chunks.append(I_one)

        K_map = torch.cat(K_chunks, dim=1)
        I_map = torch.cat(I_chunks, dim=1)
        D_map = torch.zeros_like(I_map)
        I_per_obj = [{0: I_map[0]}]
        return K_map, D_map, I_map, I_per_obj

    @staticmethod
    def _mode_b_timestamps(
        sample: dict[str, Any],
        *,
        num_images: int,
        device: torch.device,
    ) -> torch.Tensor:
        timestamps = sample["timestamps"].detach().float().to(device)
        if int(timestamps.numel()) == int(num_images):
            return timestamps
        num_frames = int(sample["frame_indices"].numel()) if "frame_indices" in sample else int(timestamps.numel())
        num_views = max(1, int(sample["cam_ids"].numel())) if "cam_ids" in sample else 1
        if int(timestamps.numel()) == num_frames and num_frames * num_views == int(num_images):
            return timestamps.repeat_interleave(num_views)
        raise ValueError(
            f"Unexpected timestamp shape: got {int(timestamps.numel())} values for {int(num_images)} images "
            f"(frames={num_frames}, views={num_views})"
        )

    @staticmethod
    def _mode_b_alpha_t(
        t: torch.Tensor,
        t0: torch.Tensor,
        alpha: torch.Tensor,
        gamma0: torch.Tensor,
        gamma1: float = 0.1,
    ) -> torch.Tensor:
        sigma = torch.log(torch.tensor(gamma1, dtype=alpha.dtype, device=alpha.device)) / (
            gamma0.to(device=alpha.device, dtype=alpha.dtype) ** 2 + 1e-6
        )
        conf = torch.exp(sigma * (t0.to(device=alpha.device, dtype=alpha.dtype) - t) ** 2)
        return (alpha * conf).float()

    @staticmethod
    def _subset_gauss_with_opacity(
        clean_dict: dict[str, torch.Tensor],
        mask: torch.Tensor,
        opacity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {k: v[mask] for k, v in clean_dict.items()}
        out["opacities"] = opacity.reshape(-1, 1)
        return out

    @staticmethod
    def _empty_pointers(device: torch.device) -> GaussianPointers:
        return GaussianPointers(
            src_kind=torch.zeros((0,), dtype=torch.int32, device=device),
            object_id=torch.zeros((0,), dtype=torch.int32, device=device),
            view_n=torch.zeros((0,), dtype=torch.int32, device=device),
            patch_idx=torch.zeros((0,), dtype=torch.int32, device=device),
            visible_mask=torch.zeros((0,), dtype=torch.bool, device=device),
        )

    @staticmethod
    def _subset_pointers(ptr: GaussianPointers, mask: torch.Tensor) -> GaussianPointers:
        mask = mask.to(device=ptr.src_kind.device, dtype=torch.bool)
        return GaussianPointers(
            src_kind=ptr.src_kind[mask],
            object_id=ptr.object_id[mask],
            view_n=ptr.view_n[mask],
            patch_idx=ptr.patch_idx[mask],
            visible_mask=ptr.visible_mask[mask],
        )

    def _time_aware_gaussians_and_pointers_for_target(
        self,
        *,
        clean_state: CleanSceneState,
        clean_dict: dict[str, torch.Tensor],
        ptr_scene: GaussianPointers,
        base_mask: torch.Tensor,
        target_idx: int,
        timestamps: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], GaussianPointers]:
        """Match DGGT's target-frame static/dynamic split and keep pointers aligned."""
        device = clean_dict["means"].device
        base_mask = base_mask.to(device=device, dtype=torch.bool)
        if not bool(base_mask.any().item()):
            return _empty_gauss_dict(device), self._empty_pointers(device)

        dynamic_prob = clean_state.dynamic_prob.to(device=device, dtype=torch.float32)
        source_image_ids = clean_state.source_image_ids.to(device=device, dtype=torch.long)
        gs_conf = clean_state.gs_conf.to(device=device, dtype=torch.float32)
        opacities = clean_dict["opacities"].to(device=device, dtype=torch.float32).view(-1)
        ptr_scene = ptr_scene.to(device)

        gauss_chunks: list[dict[str, torch.Tensor]] = []
        ptr_chunks: list[GaussianPointers] = []

        static_mask = base_mask & (dynamic_prob < 0.5)
        if bool(static_mask.any().item()):
            static_opacity = opacities[static_mask] * (1.0 - dynamic_prob[static_mask])
            static_opacity = self._mode_b_alpha_t(
                timestamps[source_image_ids[static_mask]].to(device=device, dtype=torch.float32),
                timestamps[int(target_idx)],
                static_opacity,
                gs_conf[static_mask],
            )
            gauss_chunks.append(self._subset_gauss_with_opacity(clean_dict, static_mask, static_opacity))
            ptr_chunks.append(self._subset_pointers(ptr_scene, static_mask))

        dynamic_mask = base_mask & (source_image_ids == int(target_idx))
        if bool(dynamic_mask.any().item()):
            dynamic_opacity = opacities[dynamic_mask] * dynamic_prob[dynamic_mask]
            gauss_chunks.append(self._subset_gauss_with_opacity(clean_dict, dynamic_mask, dynamic_opacity))
            ptr_chunks.append(self._subset_pointers(ptr_scene, dynamic_mask))

        if len(gauss_chunks) == 0:
            return _empty_gauss_dict(device), self._empty_pointers(device)
        return _concat_gauss_dicts(gauss_chunks, device=device), concat_pointers(ptr_chunks)

    def _mode_b_time_aware_gaussians_for_target(
        self,
        *,
        clean_state: CleanSceneState,
        clean_dict: dict[str, torch.Tensor],
        base_mask: torch.Tensor,
        target_idx: int,
        timestamps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Match DGGT's static/dynamic Gaussian alpha split for one target image."""
        device = clean_dict["means"].device
        base_mask = base_mask.to(device=device, dtype=torch.bool)
        if not bool(base_mask.any().item()):
            return _empty_gauss_dict(device)

        dynamic_prob = clean_state.dynamic_prob.to(device=device, dtype=torch.float32)
        source_image_ids = clean_state.source_image_ids.to(device=device, dtype=torch.long)
        gs_conf = clean_state.gs_conf.to(device=device, dtype=torch.float32)
        opacities = clean_dict["opacities"].to(device=device, dtype=torch.float32).view(-1)

        chunks: list[dict[str, torch.Tensor]] = []
        static_mask = base_mask & (dynamic_prob < 0.5)
        if bool(static_mask.any().item()):
            static_opacity = opacities[static_mask] * (1.0 - dynamic_prob[static_mask])
            static_opacity = self._mode_b_alpha_t(
                timestamps[source_image_ids[static_mask]].to(device=device, dtype=torch.float32),
                timestamps[int(target_idx)],
                static_opacity,
                gs_conf[static_mask],
            )
            chunks.append(self._subset_gauss_with_opacity(clean_dict, static_mask, static_opacity))

        dynamic_mask = base_mask & (source_image_ids == int(target_idx))
        if bool(dynamic_mask.any().item()):
            dynamic_opacity = opacities[dynamic_mask] * dynamic_prob[dynamic_mask]
            chunks.append(self._subset_gauss_with_opacity(clean_dict, dynamic_mask, dynamic_opacity))

        if len(chunks) == 0:
            return _empty_gauss_dict(device)
        return _concat_gauss_dicts(chunks, device=device)

    @staticmethod
    def _mode_a_asset_dict_for_target(
        asset_pass_result: AssetPassResult,
        target_idx: int,
        *,
        device: torch.device,
    ) -> dict[int, dict[str, torch.Tensor]]:
        if asset_pass_result.G_asset_dggt is None:
            return {}
        out: dict[int, dict[str, torch.Tensor]] = {}
        for obj_key in asset_pass_result.object_keys:
            obj_key = int(obj_key)
            frames = asset_pass_result.G_asset_dggt.get(obj_key)
            if frames is None or int(target_idx) >= len(frames):
                out[obj_key] = _empty_gauss_dict(device)
            else:
                out[obj_key] = {k: v.to(device) for k, v in frames[int(target_idx)].items()}
        return out

    def _mode_a_asset_gaussians_and_pointers_for_target(
        self,
        asset_pass_result: AssetPassResult,
        target_idx: int,
        *,
        device: torch.device,
    ) -> tuple[list[dict[str, torch.Tensor]], list[GaussianPointers]]:
        if asset_pass_result.G_asset_dggt is None:
            return [], []
        gauss_chunks: list[dict[str, torch.Tensor]] = []
        ptr_chunks: list[GaussianPointers] = []
        for obj_key in asset_pass_result.object_keys:
            obj_key = int(obj_key)
            frames = asset_pass_result.G_asset_dggt.get(obj_key)
            ptr_frames = asset_pass_result.ptr_asset.get(obj_key)
            if frames is None or ptr_frames is None or int(target_idx) >= len(frames) or int(target_idx) >= len(ptr_frames):
                continue
            gauss = {k: v.to(device) for k, v in frames[int(target_idx)].items()}
            n = int(gauss["means"].shape[0])
            if n == 0:
                continue
            ptr = ptr_frames[int(target_idx)].to(device)
            if int(ptr.patch_idx.numel()) != n:
                raise ValueError(
                    f"Mode-A asset pointer/Gaussian count mismatch for object {obj_key}, "
                    f"target {target_idx}: ptr={int(ptr.patch_idx.numel())}, gauss={n}"
                )
            gauss_chunks.append(gauss)
            ptr_chunks.append(ptr)
        return gauss_chunks, ptr_chunks

    def _render_mode_a_per_target_coverage(
        self,
        *,
        sample: dict[str, Any],
        clean_state: CleanSceneState,
        clean_dict: dict[str, torch.Tensor],
        keep_mask: torch.Tensor,
        delete_mask: torch.Tensor,
        asset_pass_result: AssetPassResult,
        cameras_dggt: dict[str, torch.Tensor],
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[int, torch.Tensor]]]:
        viewmats = cameras_dggt["viewmats"]
        if int(viewmats.shape[0]) != 1:
            raise ValueError("Mode-A per-target coverage currently expects batch size 1.")
        S = int(viewmats.shape[1])
        device = clean_dict["means"].device
        timestamps = self._mode_b_timestamps(sample, num_images=S, device=viewmats.device)

        K_chunks: list[torch.Tensor] = []
        D_chunks: list[torch.Tensor] = []
        I_chunks: list[torch.Tensor] = []
        per_obj_chunks: dict[int, list[torch.Tensor]] = {
            int(k): [] for k in asset_pass_result.object_keys
        }
        keep_mask = keep_mask.to(device=device, dtype=torch.bool)
        delete_mask = delete_mask.to(device=device, dtype=torch.bool)

        for target_idx in range(S):
            gauss_kept = self._mode_b_time_aware_gaussians_for_target(
                clean_state=clean_state,
                clean_dict=clean_dict,
                base_mask=keep_mask,
                target_idx=target_idx,
                timestamps=timestamps,
            )
            gauss_deleted = self._mode_b_time_aware_gaussians_for_target(
                clean_state=clean_state,
                clean_dict=clean_dict,
                base_mask=delete_mask,
                target_idx=target_idx,
                timestamps=timestamps,
            )
            asset_dict = self._mode_a_asset_dict_for_target(
                asset_pass_result,
                target_idx,
                device=device,
            )
            cameras_one = {
                "viewmats": cameras_dggt["viewmats"][:, target_idx : target_idx + 1].contiguous(),
                "Ks": cameras_dggt["Ks"][:, target_idx : target_idx + 1].contiguous(),
            }
            K_one, D_one, I_one, per_one = self.soft_mask.render_coverage(
                G_kept=[gauss_kept],
                G_deleted=[gauss_deleted],
                G_asset_dggt_dict=[asset_dict],
                cameras_dggt=cameras_one,
                H=H,
                W=W,
            )
            K_chunks.append(K_one)
            D_chunks.append(D_one)
            I_chunks.append(I_one)
            for obj_key in per_obj_chunks:
                per_obj_chunks[obj_key].append(
                    per_one[0].get(obj_key, I_one.new_zeros((1, H, W, 1)))
                )

        K_map = torch.cat(K_chunks, dim=1)
        D_map = torch.cat(D_chunks, dim=1)
        I_map = torch.cat(I_chunks, dim=1)
        I_per_obj = [
            {obj_key: torch.cat(parts, dim=0) for obj_key, parts in per_obj_chunks.items()}
        ]
        return K_map, D_map, I_map, I_per_obj

    def _render_mode_a_depth_aware_coverage(
        self,
        G_kept: Sequence[Mapping[str, torch.Tensor]],
        G_deleted: Sequence[Mapping[str, torch.Tensor]],
        G_asset_dggt_dict: Sequence[Mapping[int, Mapping[str, torch.Tensor]]],
        cameras_dggt: Mapping[str, torch.Tensor],
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[int, torch.Tensor]]]:
        """Compatibility wrapper for older debug tools.

        The current Mode-A rule is: K/D stay independent, while inserted
        assets are rendered together so only asset-asset visibility is resolved.
        """
        return self.soft_mask.render_coverage(
            G_kept,
            G_deleted,
            G_asset_dggt_dict,
            cameras_dggt=cameras_dggt,
            H=H,
            W=W,
        )

    def _splat_mode_a_per_target(
        self,
        *,
        sample: dict[str, Any],
        clean_state: CleanSceneState,
        clean_dict: dict[str, torch.Tensor],
        keep_mask: torch.Tensor,
        ptr_scene: GaussianPointers,
        asset_pass_result: AssetPassResult,
        lut_scene: Sequence[torch.Tensor],
        lut_asset_dict: dict[int, list[torch.Tensor]],
        cameras_splat: dict[str, torch.Tensor],
        tile_masks: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        viewmats = cameras_splat["viewmats"]
        if int(viewmats.shape[0]) != 1:
            raise ValueError("Mode-A per-target splat currently expects batch size 1.")
        S = int(viewmats.shape[1])
        device = clean_dict["means"].device
        timestamps = self._mode_b_timestamps(sample, num_images=S, device=viewmats.device)
        keep_mask = keep_mask.to(device=device, dtype=torch.bool)

        splatted_levels: list[list[torch.Tensor]] = [[] for _ in range(self.num_levels)]
        for target_idx in range(S):
            gauss_scene_t, ptr_scene_t = self._time_aware_gaussians_and_pointers_for_target(
                clean_state=clean_state,
                clean_dict=clean_dict,
                ptr_scene=ptr_scene,
                base_mask=keep_mask,
                target_idx=target_idx,
                timestamps=timestamps,
            )
            asset_gauss_chunks, asset_ptr_chunks = self._mode_a_asset_gaussians_and_pointers_for_target(
                asset_pass_result,
                target_idx,
                device=device,
            )
            gauss_all = _concat_gauss_dicts([gauss_scene_t] + asset_gauss_chunks, device=device)
            ptr_all = concat_pointers([ptr_scene_t] + asset_ptr_chunks)
            cameras_one = {
                "viewmats": cameras_splat["viewmats"][:, target_idx : target_idx + 1].contiguous(),
                "Ks": cameras_splat["Ks"][:, target_idx : target_idx + 1].contiguous(),
            }
            tile_one = None if tile_masks is None else tile_masks[:, target_idx : target_idx + 1].contiguous()
            chunk_out = self.feature_splatter(
                gaussians_dggt=[gauss_all],
                pointers=[ptr_all],
                lut_scene=lut_scene,
                lut_asset_dict=lut_asset_dict if len(asset_ptr_chunks) > 0 else None,
                cameras_dggt=cameras_one,
                H=self.H_splat,
                W=self.W_splat,
                pool_to=self.patch_grid,
                tile_masks=tile_one,
            )
            for level_idx, level_tensor in enumerate(chunk_out):
                splatted_levels[level_idx].append(level_tensor)
        return [torch.cat(parts, dim=1) for parts in splatted_levels]

    def _splat_mode_b_per_target(
        self,
        *,
        sample: dict[str, Any],
        clean_state: CleanSceneState,
        clean_dict: dict[str, torch.Tensor],
        ptr_scene: GaussianPointers,
        delete_masks_by_target: torch.Tensor,
        lut_scene: Sequence[torch.Tensor],
        cameras_splat: dict[str, torch.Tensor],
        tile_masks: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Feature-splat Mode B one target frame at a time.

        FeatureSplatter accepts one Gaussian set for all target cameras. Mode B
        needs a different kept set per target frame, so exact semantics require
        frame-wise splats.
        """
        viewmats = cameras_splat["viewmats"]
        if int(viewmats.shape[0]) != 1:
            raise ValueError("Mode-B per-target splat currently expects batch size 1.")
        S = int(viewmats.shape[1])
        if tuple(delete_masks_by_target.shape) != (S, int(clean_dict["means"].shape[0])):
            raise ValueError(
                "delete_masks_by_target must be [S,N_g], got "
                f"{tuple(delete_masks_by_target.shape)} for S={S}, N_g={int(clean_dict['means'].shape[0])}"
            )
        timestamps = self._mode_b_timestamps(sample, num_images=S, device=viewmats.device)
        device = clean_dict["means"].device

        splatted_levels: list[list[torch.Tensor]] = [[] for _ in range(self.num_levels)]
        for target_idx in range(S):
            del_mask = delete_masks_by_target[target_idx].to(
                device=device, dtype=torch.bool
            )
            keep_mask = ~del_mask
            gauss_kept, ptr_kept = self._time_aware_gaussians_and_pointers_for_target(
                clean_state=clean_state,
                clean_dict=clean_dict,
                ptr_scene=ptr_scene,
                base_mask=keep_mask,
                target_idx=target_idx,
                timestamps=timestamps,
            )
            cameras_one = {
                "viewmats": cameras_splat["viewmats"][:, target_idx : target_idx + 1].contiguous(),
                "Ks": cameras_splat["Ks"][:, target_idx : target_idx + 1].contiguous(),
            }
            tile_one = None if tile_masks is None else tile_masks[:, target_idx : target_idx + 1].contiguous()
            chunk_out = self.feature_splatter(
                gaussians_dggt=[gauss_kept],
                pointers=[ptr_kept],
                lut_scene=lut_scene,
                lut_asset_dict=None,
                cameras_dggt=cameras_one,
                H=self.H_splat,
                W=self.W_splat,
                pool_to=self.patch_grid,
                tile_masks=tile_one,
            )
            for level_idx, level_tensor in enumerate(chunk_out):
                splatted_levels[level_idx].append(level_tensor)
        return [torch.cat(parts, dim=1) for parts in splatted_levels]

    def _splat_weight_to_tile_masks(
        self,
        splat_weight: torch.Tensor,
        *,
        threshold: float,
        H_splat: int,
        W_splat: int,
        tile_size: int = 16,
    ) -> torch.Tensor:
        """Convert per-token splat contribution `[B,S,P,1]` into gsplat tiles."""
        if splat_weight.dim() != 4 or splat_weight.shape[-1] != 1:
            raise ValueError(f"splat_weight must be [B,S,P,1], got {tuple(splat_weight.shape)}")
        B, S, P, _ = splat_weight.shape
        patch_h, patch_w = self.patch_grid
        if int(P) != patch_h * patch_w:
            raise ValueError(f"splat_weight P={int(P)} does not match patch_grid={self.patch_grid}")
        if int(H_splat) % patch_h != 0 or int(W_splat) % patch_w != 0:
            raise ValueError(
                f"splat size {(H_splat, W_splat)} must be divisible by patch_grid={self.patch_grid}"
            )

        active = splat_weight[..., 0].detach() > float(threshold)
        active_grid = active.reshape(B, S, patch_h, patch_w)
        tile_h = math.ceil(int(H_splat) / float(tile_size))
        tile_w = math.ceil(int(W_splat) / float(tile_size))
        pix_h = int(H_splat) // patch_h
        pix_w = int(W_splat) // patch_w
        tile_masks = torch.zeros((B, S, tile_h, tile_w), dtype=torch.bool, device=active.device)
        for ty in range(tile_h):
            y0 = int(ty * tile_size)
            y1 = min(int((ty + 1) * tile_size), int(H_splat))
            py0 = max(0, min(patch_h, y0 // pix_h))
            py1 = max(py0, min(patch_h, math.ceil(y1 / float(pix_h))))
            for tx in range(tile_w):
                x0 = int(tx * tile_size)
                x1 = min(int((tx + 1) * tile_size), int(W_splat))
                px0 = max(0, min(patch_w, x0 // pix_w))
                px1 = max(px0, min(patch_w, math.ceil(x1 / float(pix_w))))
                if py1 > py0 and px1 > px0:
                    tile_masks[:, :, ty, tx] = active_grid[:, :, py0:py1, px0:px1].any(dim=(-1, -2))
        return tile_masks

    def _unedited_preserve_token_mask(
        self,
        *,
        D_map: torch.Tensor,
        I_map: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        edit_map = (D_map + I_map).clamp(0.0, 1.0)
        edit_tok = SoftMaskBuilder._area_pool_to_grid(
            edit_map,
            self.patch_grid,
        ).to(device=reference.device, dtype=reference.dtype)
        return edit_tok <= float(self.unedited_preserve_threshold)

    def _force_preserve_unedited_tokens(
        self,
        *,
        K_map: torch.Tensor,
        D_map: torch.Tensor,
        I_map: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make edit-free tokens true preserve tokens.

        DGGT renders sky through ``SkyGaussian`` and excludes sky pixels from
        foreground scene Gaussians.  Therefore low/zero kept-Gaussian coverage
        is not, by itself, an edit request.  The diffusion masks should mark a
        token editable only when deleted or inserted/imagined coverage is
        actually present.  This keeps sky and other untouched alpha holes on
        the clean-token path while preserving the original soft D/I ratios on
        real edit footprints.
        """
        del K_map  # kept coverage does not define whether a token is edited
        force_preserve = self._unedited_preserve_token_mask(
            D_map=D_map,
            I_map=I_map,
            reference=M_preserve,
        )

        M_preserve = torch.where(force_preserve, torch.ones_like(M_preserve), M_preserve)
        M_source = torch.where(force_preserve, torch.zeros_like(M_source), M_source)
        M_dest = torch.where(force_preserve, torch.zeros_like(M_dest), M_dest)
        return M_preserve, M_source, M_dest

    def _blend_preserve_tokens(
        self,
        clean_levels: Sequence[torch.Tensor],
        splatted_levels: Sequence[torch.Tensor],
        M_preserve: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Use clean tokens in confidently preserved regions.

        Feature splatting is useful where geometry changes, but unchanged
        background should not be forced through a lossy GS reprojection.  The
        soft preserve mask gives a conservative per-token bypass that keeps
        clean features exact where no source/destination edit footprint exists.
        """
        if not self.preserve_token_bypass:
            return list(splatted_levels)
        if len(clean_levels) != len(splatted_levels):
            raise ValueError(
                f"clean/splatted level count mismatch: {len(clean_levels)} vs {len(splatted_levels)}"
            )
        if M_preserve.dim() != 4 or M_preserve.shape[-1] != 1:
            raise ValueError(f"M_preserve must be [B,S,P,1], got {tuple(M_preserve.shape)}")

        out: list[torch.Tensor] = []
        for level_idx, (clean, splatted) in enumerate(zip(clean_levels, splatted_levels)):
            if clean.shape != splatted.shape:
                raise ValueError(
                    f"Level {level_idx} clean/splatted shape mismatch: "
                    f"{tuple(clean.shape)} vs {tuple(splatted.shape)}"
                )
            if clean.shape[:-1] != M_preserve.shape[:-1]:
                raise ValueError(
                    f"Level {level_idx} token axes {tuple(clean.shape[:-1])} do not match "
                    f"M_preserve {tuple(M_preserve.shape[:-1])}"
                )
            preserve = M_preserve.to(device=splatted.device, dtype=splatted.dtype).clamp(0.0, 1.0)
            clean = clean.to(device=splatted.device, dtype=splatted.dtype)
            out.append(preserve * clean + (1.0 - preserve) * splatted)
        return out

    def _blend_asset_tokens(
        self,
        splatted_levels: Sequence[torch.Tensor],
        F_g_lut_asset: dict[int, list[torch.Tensor]],
        I_map_per_obj: list[dict[int, torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Use the aligned asset-pass LUT in confidently covered asset patches.

        Asset RGB/alpha placement is already rendered in the DGGT camera space
        during the asset pass.  Re-splatting 3072-D asset tokens through one
        patch pointer per Gaussian is lossy, especially for small objects where
        a patch mixes several projected Gaussian footprints.  This blend keeps
        FeatureSplatter for geometry-changing low-coverage boundaries, while
        directly using the asset LUT where the asset owns most of the token.
        """
        if not self.asset_token_direct_blend or len(F_g_lut_asset) == 0:
            return list(splatted_levels)
        if len(I_map_per_obj) == 0:
            return list(splatted_levels)
        if self.asset_token_full_alpha <= 0.0:
            raise ValueError("asset_token_full_alpha must be positive")

        out: list[torch.Tensor] = []
        for level_idx, splatted in enumerate(splatted_levels):
            device = splatted.device
            dtype = splatted.dtype
            B, S, P, C = splatted.shape
            token_sum = torch.zeros_like(splatted)
            weight_sum = torch.zeros((B, S, P, 1), dtype=dtype, device=device)

            for b, per_obj in enumerate(I_map_per_obj):
                if b >= B:
                    break
                for obj_key, alpha_hires in per_obj.items():
                    obj_key = int(obj_key)
                    if obj_key not in F_g_lut_asset:
                        continue
                    asset_levels = F_g_lut_asset[obj_key]
                    if level_idx >= len(asset_levels):
                        continue
                    asset_lut = asset_levels[level_idx].to(device=device, dtype=dtype)
                    if asset_lut.shape != splatted.shape:
                        raise ValueError(
                            f"asset LUT[{obj_key}][{level_idx}] shape {tuple(asset_lut.shape)} "
                            f"does not match splatted level {tuple(splatted.shape)}"
                        )
                    alpha = alpha_hires.to(device=device, dtype=dtype)
                    if alpha.dim() != 4 or alpha.shape[-1] != 1:
                        raise ValueError(
                            f"I_map_per_obj[{b}][{obj_key}] must be [S,H,W,1], got {tuple(alpha.shape)}"
                        )
                    alpha_tok = SoftMaskBuilder._area_pool_to_grid(
                        alpha.unsqueeze(0), self.patch_grid
                    ).clamp(0.0, 1.0)
                    if alpha_tok.shape[1:] != splatted.shape[1:-1] + (1,):
                        raise ValueError(
                            f"pooled asset alpha shape {tuple(alpha_tok.shape)} is incompatible "
                            f"with splatted level {tuple(splatted.shape)}"
                        )
                    w = alpha_tok[0:1]
                    token_sum[b : b + 1] = token_sum[b : b + 1] + w * asset_lut[b : b + 1]
                    weight_sum[b : b + 1] = weight_sum[b : b + 1] + w

            asset_ref = token_sum / weight_sum.clamp_min(1e-6)
            blend = (weight_sum / float(self.asset_token_full_alpha)).clamp(0.0, 1.0)
            out.append(blend * asset_ref + (1.0 - blend) * splatted)
        return out

    @staticmethod
    def _validate_mode_a_asset_pass(asset_pass_result: AssetPassResult) -> None:
        if len(asset_pass_result.object_keys) == 0:
            return
        if str(asset_pass_result.asset_pass_space) != "dggt_fitted":
            raise RuntimeError(
                "Mode-A FlowFeatureAssembler requires fitted DGGT asset-pass cache. "
                f"Expected asset_pass_space='dggt_fitted', got {asset_pass_result.asset_pass_space!r}. "
                "Regenerate the .pt files with tools/precompute_flow_features.py."
            )
        if asset_pass_result.G_asset_dggt is None:
            raise RuntimeError(
                "Mode-A fitted asset pass is missing G_asset_dggt. "
                "Regenerate the .pt files with tools/precompute_flow_features.py."
            )
        missing = [
            int(k)
            for k in asset_pass_result.object_keys
            if int(k) not in asset_pass_result.G_asset_dggt
        ]
        if missing:
            raise RuntimeError(f"Mode-A fitted asset pass missing DGGT gaussians for object keys: {missing}")

    def _select_lut_scene(self, predictions: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Return 4-level patch-token LUT for scene Gaussians.

        Accepts either `image_tokens_levels: list[4 × [B, S, P_patch, 3072]]`
        or a full `image_tokens_list: list[24 × [B, S, P_patch+5, 3072]]` with
        `patch_start_idx`.
        """
        if "image_tokens_levels" in predictions:
            levels = predictions["image_tokens_levels"]
            if len(levels) != self.num_levels:
                raise ValueError(
                    f"`image_tokens_levels` must have {self.num_levels} entries, "
                    f"got {len(levels)}"
                )
            return list(levels)
        tokens_list = predictions.get("image_tokens_list")
        if tokens_list is None:
            raise KeyError(
                "predictions must contain either 'image_tokens_levels' or "
                "'image_tokens_list' (full 24-layer pyramid)."
            )
        from dggt.utils.tokens import select_patch_pyramid

        return list(
            select_patch_pyramid(tokens_list, DEFAULT_LEVELS, self.patch_start_idx)
        )

    @staticmethod
    def scale_cameras_for_render(
        cameras_dggt: dict[str, torch.Tensor],
        source_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Return camera intrinsics scaled from source image size to render size.

        `cameras_dggt["Ks"]` is predicted for the model image grid (for example
        350x518), while FeatureSplatter rasterizes on a smaller intermediate
        canvas (for example 100x148). gsplat expects intrinsics in the same pixel
        coordinate system as `height`/`width`, so focal lengths and principal
        points must be scaled before splatting. View matrices are unchanged.
        """
        if "viewmats" not in cameras_dggt or "Ks" not in cameras_dggt:
            raise ValueError("cameras_dggt must contain 'viewmats' and 'Ks'")
        src_h, src_w = int(source_hw[0]), int(source_hw[1])
        dst_h, dst_w = int(target_hw[0]), int(target_hw[1])
        if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
            raise ValueError(
                f"source_hw and target_hw must be positive, got {source_hw} -> {target_hw}"
            )

        out = dict(cameras_dggt)
        Ks = cameras_dggt["Ks"].clone()
        scale_x = float(dst_w) / float(src_w)
        scale_y = float(dst_h) / float(src_h)
        Ks[..., 0, :] = Ks[..., 0, :] * scale_x
        Ks[..., 1, :] = Ks[..., 1, :] * scale_y
        out["Ks"] = Ks
        return out

    @staticmethod
    def _mask_asset_pass_by_coverage(
        asset_pass_result: AssetPassResult,
        phase1_coverage: torch.Tensor,
        phase4_slots: list[int],
        device: torch.device,
    ) -> AssetPassResult:
        """Zero out asset-pass frames that Phase 1 didn't actually delete.

        Aligns Phase 4 outputs exactly with Phase 1 per-frame coverage.
        """
        if len(asset_pass_result.object_keys) == 0:
            return asset_pass_result

        kept_keys = [int(k) for k in asset_pass_result.object_keys if int(k) in phase4_slots]
        if len(kept_keys) == 0:
            return AssetPassResult(
                patch_grid=asset_pass_result.patch_grid,
                patch_start_idx=asset_pass_result.patch_start_idx,
                object_keys=[],
                cameras_waymo=asset_pass_result.cameras_waymo,
                F_g_lut_asset={},
                ptr_asset={},
                G_asset_waymo={},
                G_asset_dggt={} if asset_pass_result.G_asset_dggt is not None else None,
                I_asset={},
                A_asset={},
                asset_pass_space=asset_pass_result.asset_pass_space,
                fit_metrics={},
            )

        F_g_lut_asset: dict[int, list[torch.Tensor]] = {}
        ptr_asset: dict[int, list[GaussianPointers]] = {}
        G_asset_dggt_out: dict[int, list[dict[str, torch.Tensor]]] | None = (
            {} if asset_pass_result.G_asset_dggt is not None else None
        )
        I_asset: dict[int, torch.Tensor] = {}
        A_asset: dict[int, torch.Tensor] = {}

        for k in kept_keys:
            cov_k = phase1_coverage[k]  # [S] bool for this slot
            # LUT shape: [1, S, P, 3072] per level; zero-out non-covered frames.
            F_g_lut_asset[k] = [
                lvl * cov_k.view(1, -1, 1, 1).to(lvl.dtype).to(lvl.device)
                for lvl in asset_pass_result.F_g_lut_asset[k]
            ]
            # Per-frame pointers: keep only covered frames, others become empty ptrs.
            new_ptrs: list[GaussianPointers] = []
            for s, ptr in enumerate(asset_pass_result.ptr_asset[k]):
                if bool(cov_k[s].item()):
                    new_ptrs.append(ptr)
                else:
                    n = int(ptr.patch_idx.numel())
                    ptr_device = ptr.patch_idx.device
                    new_ptrs.append(
                        GaussianPointers(
                            src_kind=torch.full((n,), SRC_KIND_ASSET, dtype=torch.int32, device=ptr_device),
                            object_id=torch.full((n,), int(k), dtype=torch.int32, device=ptr_device),
                            view_n=ptr.view_n,
                            patch_idx=ptr.patch_idx,
                            visible_mask=torch.zeros((n,), dtype=torch.bool, device=ptr_device),
                        )
                    )
            ptr_asset[k] = new_ptrs
            # Per-frame DGGT Gaussian dicts: replace non-covered frames with empties.
            dggt_frames: list[dict[str, torch.Tensor]] = []
            if asset_pass_result.G_asset_dggt is None:
                raise RuntimeError("Mode-A asset pass lost DGGT gaussians during coverage masking")
            for s in range(len(asset_pass_result.G_asset_dggt[k])):
                if bool(cov_k[s].item()):
                    if G_asset_dggt_out is not None:
                        dggt_frames.append(asset_pass_result.G_asset_dggt[k][s])
                else:
                    if G_asset_dggt_out is not None:
                        dggt_frames.append(_empty_gauss_dict(device))
            if G_asset_dggt_out is not None:
                G_asset_dggt_out[k] = dggt_frames
            # Renders likewise — stored unmasked; SoftMaskBuilder ignores Gaussians anyway.
            I_asset[k] = asset_pass_result.I_asset[k]
            A_asset[k] = asset_pass_result.A_asset[k] * cov_k.view(1, -1, 1, 1, 1).to(
                asset_pass_result.A_asset[k].dtype
            ).to(asset_pass_result.A_asset[k].device)

        return AssetPassResult(
            patch_grid=asset_pass_result.patch_grid,
            patch_start_idx=asset_pass_result.patch_start_idx,
            object_keys=kept_keys,
            cameras_waymo=asset_pass_result.cameras_waymo,
            F_g_lut_asset=F_g_lut_asset,
            ptr_asset=ptr_asset,
            G_asset_waymo={},
            G_asset_dggt=G_asset_dggt_out,
            I_asset=I_asset,
            A_asset=A_asset,
            asset_pass_space=asset_pass_result.asset_pass_space,
            fit_metrics=None
            if asset_pass_result.fit_metrics is None
            else {k: asset_pass_result.fit_metrics.get(k, []) for k in kept_keys},
        )
