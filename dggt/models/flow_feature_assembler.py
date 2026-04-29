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

from dataclasses import dataclass, field
from typing import Any, Sequence

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
            `AssetPassResult` reconstructed from cache. Uses DGGT-coord asset
            Gaussians via `alignment` applied during offline precompute.
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
            )

        # ----- Mode A path ----- #
        # Phase 1: build clean state + align + localize (load_asset=False) + apply
        clean_state = self.editor.build_clean_bundle(sample, predictions)
        alignment = self.editor.align(sample, clean_state)
        object_slots = (
            parse_object_slots(sample, object_slots_spec)
            if isinstance(object_slots_spec, str)
            else [int(s) for s in object_slots_spec]
        )
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
        )
        pointers_asset_by_obj: dict[int, GaussianPointers] = {}
        asset_gauss_chunks: list[dict[str, torch.Tensor]] = []
        ptr_chunks: list[GaussianPointers] = [ptr_scene]
        for obj_key in asset_pass_result.object_keys:
            obj_ptrs = asset_pass_result.ptr_asset[int(obj_key)]
            obj_gauss_frames = (
                asset_pass_result.G_asset_dggt[int(obj_key)]
                if asset_pass_result.G_asset_dggt is not None
                else asset_pass_result.G_asset_waymo[int(obj_key)]
            )
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
            src_kind=ptr_scene.src_kind[keep_mask.cpu()],
            object_id=ptr_scene.object_id[keep_mask.cpu()],
            view_n=ptr_scene.view_n[keep_mask.cpu()],
            patch_idx=ptr_scene.patch_idx[keep_mask.cpu()],
            visible_mask=ptr_scene.visible_mask[keep_mask.cpu()],
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
        splatted_tok_low = self.feature_splatter(
            gaussians_dggt=[gaussians_all],
            pointers=[pointers_all],
            lut_scene=F_g_lut_scene,
            lut_asset_dict=F_g_lut_asset if len(F_g_lut_asset) > 0 else None,
            cameras_dggt=cameras_splat,
            H=self.H_splat,
            W=self.W_splat,
            pool_to=self.patch_grid,
        )

        # ------------------- Phase 3: Soft masks + Scaffold -------------- #
        G_kept_list = [gauss_scene_kept]
        G_deleted_list = [{k: v[edit_state.delete_mask].to(device) for k, v in gauss_scene.items()}]
        G_asset_dict_list = [
            {int(k): gauss for k, gauss in edit_bundle.G_asset_per_object.items()}
        ]
        K_map, D_map, I_map, I_per_obj = self.soft_mask.render_coverage(
            G_kept_list,
            G_deleted_list,
            G_asset_dict_list,
            cameras_dggt=cameras_dggt,
            H=H_img,
            W=W_img,
        )
        M_preserve, M_source, M_dest = self.soft_mask.pool_and_normalize(
            K_map, D_map, I_map, target_grid=self.patch_grid
        )
        splatted_tok_low = self._blend_preserve_tokens(
            clean_levels=F_g_lut_scene,
            splatted_levels=splatted_tok_low,
            M_preserve=M_preserve,
        )

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
        z_clean = self.scene_tokenizer.encode(F_g_lut_scene, patch_grid=self.patch_grid)
        z_splat = self.scene_tokenizer.encode(splatted_tok_low, patch_grid=self.patch_grid)

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

        # Pseudo-delete mask from cache. The cache stored the per-Gaussian
        # delete mask computed against the same deterministic clean_state.
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
        keep_mask_dev = (~delete_mask).to(device)
        del_mask_dev = delete_mask.to(device)
        gauss_kept = {k: v[keep_mask_dev] for k, v in clean_dict.items()}
        gauss_imagined = {k: v[del_mask_dev] for k, v in clean_dict.items()}

        # Pointers (scene only). The kept/imagined slices keep the per-Gaussian
        # ordering so we can splat scene tokens onto kept gaussians.
        ptr_scene = build_scene_pointers(
            clean_state.source_image_ids,
            clean_state.source_y,
            clean_state.source_x,
            patch_size=int(H_img // self.patch_grid[0]),
            patch_grid=self.patch_grid,
        )
        keep_mask_cpu = keep_mask_dev.cpu()
        ptr_scene_kept = GaussianPointers(
            src_kind=ptr_scene.src_kind[keep_mask_cpu],
            object_id=ptr_scene.object_id[keep_mask_cpu],
            view_n=ptr_scene.view_n[keep_mask_cpu],
            patch_idx=ptr_scene.patch_idx[keep_mask_cpu],
            visible_mask=ptr_scene.visible_mask[keep_mask_cpu],
        )
        pointers_all = ptr_scene_kept

        F_g_lut_scene = self._select_lut_scene(predictions)

        # Phase 2: Splat scene tokens onto kept Gaussians only.
        cameras_splat = self.scale_cameras_for_render(
            cameras_dggt,
            source_hw=(H_img, W_img),
            target_hw=(self.H_splat, self.W_splat),
        )
        splatted_tok_low = self.feature_splatter(
            gaussians_dggt=[gauss_kept],
            pointers=[pointers_all],
            lut_scene=F_g_lut_scene,
            lut_asset_dict=None,
            cameras_dggt=cameras_splat,
            H=self.H_splat,
            W=self.W_splat,
            pool_to=self.patch_grid,
        )

        # Phase 3: Soft masks. K = render(kept), I = render(imagined Gaussians),
        # D = 0. Soft-normalize → M_preserve, M_source(=0), M_dest.
        K_map, _D_map_dummy, I_map_proxy, _ = self.soft_mask.render_coverage(
            G_kept=[gauss_kept],
            G_deleted=[{}],
            G_asset_dggt_dict=[{0: gauss_imagined}] if gauss_imagined["means"].numel() > 0 else [{}],
            cameras_dggt=cameras_dggt,
            H=H_img,
            W=W_img,
        )
        D_map = torch.zeros_like(I_map_proxy)
        I_map = I_map_proxy
        I_per_obj: list[dict[int, torch.Tensor]] = [{}]

        M_preserve, M_source, M_dest = self.soft_mask.pool_and_normalize(
            K_map, D_map, I_map, target_grid=self.patch_grid
        )
        splatted_tok_low = self._blend_preserve_tokens(
            clean_levels=F_g_lut_scene,
            splatted_levels=splatted_tok_low,
            M_preserve=M_preserve,
        )

        # Scaffold (D_edited not meaningful for mode B — pass zeros).
        B = int(F_g_lut_scene[0].shape[0])
        S = int(F_g_lut_scene[0].shape[1])
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
        z_clean = self.scene_tokenizer.encode(F_g_lut_scene, patch_grid=self.patch_grid)
        z_splat = self.scene_tokenizer.encode(splatted_tok_low, patch_grid=self.patch_grid)

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
        )
        from dggt.utils.gaussian_edit import EditedSceneState

        edit_state_proxy = EditedSceneState(
            clean=clean_dict,
            deleted=gauss_imagined,
            asset_only={},
            edited=gauss_kept,
            localized_objects=[],
            delete_mask=delete_mask.to(device),
            shell_mask=delete_mask.to(device),
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
                "delete_mask": delete_mask,
                "shell_mask": delete_mask,
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
                "delete_mask": delete_mask.cpu(),
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
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
            )

        F_g_lut_asset: dict[int, list[torch.Tensor]] = {}
        ptr_asset: dict[int, list[GaussianPointers]] = {}
        G_asset_waymo: dict[int, list[dict[str, torch.Tensor]]] = {}
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
                    new_ptrs.append(
                        GaussianPointers(
                            src_kind=torch.full((n,), SRC_KIND_ASSET, dtype=torch.int32),
                            object_id=torch.full((n,), int(k), dtype=torch.int32),
                            view_n=ptr.view_n,
                            patch_idx=ptr.patch_idx,
                            visible_mask=torch.zeros((n,), dtype=torch.bool),
                        )
                    )
            ptr_asset[k] = new_ptrs
            # Per-frame Gaussian dicts: replace non-covered frames with empties.
            G_asset_waymo[k] = []
            dggt_frames: list[dict[str, torch.Tensor]] = []
            for s in range(len(asset_pass_result.G_asset_waymo[k])):
                if bool(cov_k[s].item()):
                    G_asset_waymo[k].append(asset_pass_result.G_asset_waymo[k][s])
                    if G_asset_dggt_out is not None:
                        dggt_frames.append(asset_pass_result.G_asset_dggt[k][s])
                else:
                    G_asset_waymo[k].append(_empty_gauss_dict(device))
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
            G_asset_waymo=G_asset_waymo,
            G_asset_dggt=G_asset_dggt_out,
            I_asset=I_asset,
            A_asset=A_asset,
        )
