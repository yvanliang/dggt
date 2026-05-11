"""Pseudo-asset bundle helpers for SceneFlow pretraining.

The pretraining task uses raw Waymo dynamic masks to synthesize an
edit-like conditioning bundle for ``WanSceneFlow``.  Dynamic connected
components become destination masks in the current frame, while cross-attention
KV tokens are taken from a different frame to avoid giving the model a direct
copy of the target tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import torch
import torch.nn.functional as F


@dataclass
class PretrainBundle:
    z_clean_n: torch.Tensor
    M_preserve: torch.Tensor
    M_source: torch.Tensor
    M_dest: torch.Tensor
    F_asset_tokens: torch.Tensor
    encoder_attention_mask: torch.Tensor | None
    F_asset_lengths: torch.Tensor | None = None


def build_pretrain_bundle(
    z_clean_n: torch.Tensor,
    image_tokens_last: torch.Tensor,
    dynamic_mask: torch.Tensor,
    *,
    patch_grid: tuple[int, int] = (37, 37),
    K_max: int = 3,
    min_inst_patches: int = 4,
    max_inst_patches: int = 150,
    ref_offset: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dyn_threshold: float = 0.05,
    pool_threshold: float | None = None,
) -> PretrainBundle:
    """Build pretraining masks and cross-frame pseudo-asset KV tokens."""
    _validate_inputs(z_clean_n, dynamic_mask, image_tokens_last, patch_grid)

    B, S, P, _ = z_clean_n.shape
    _, _, _, C = image_tokens_last.shape
    device = z_clean_n.device if device is None else device
    dtype = z_clean_n.dtype if dtype is None else dtype
    mask_dtype = dtype
    if pool_threshold is not None:
        dyn_threshold = float(pool_threshold)
    if ref_offset is None:
        ref_offset = max(1, S // 2)
    ref_offset = int(ref_offset)

    dyn_patch = _downsample_dynamic_mask(dynamic_mask, patch_grid, dyn_threshold)
    picked = _pick_components(
        dyn_patch,
        K_max=K_max,
        min_inst_patches=min_inst_patches,
        max_inst_patches=max_inst_patches,
    )

    M_dest = torch.zeros((B, S, P, 1), device=device, dtype=mask_dtype)
    for b in range(B):
        for s in range(S):
            for comp in picked[b][s]:
                if comp:
                    idx = torch.tensor(comp, device=device, dtype=torch.long)
                    M_dest[b, s, idx, 0] = 1.0

    M_preserve = 1.0 - M_dest
    M_source = torch.zeros_like(M_dest)

    kv_chunks: list[torch.Tensor] = []
    lengths: list[int] = []
    for b in range(B):
        chunks_b = []
        for s in range(S):
            s_ref = (s + ref_offset) % S
            for comp in picked[b][s_ref]:
                if comp:
                    idx = torch.tensor(comp, device=image_tokens_last.device, dtype=torch.long)
                    chunks_b.append(image_tokens_last[b, s_ref, idx, :])
        if chunks_b:
            kv_b = torch.cat(chunks_b, dim=0)
        else:
            kv_b = image_tokens_last.new_empty((0, C))
        kv_chunks.append(kv_b)
        lengths.append(int(kv_b.shape[0]))

    max_len = max(lengths) if lengths else 0
    lengths_t = torch.tensor(lengths, device=device, dtype=torch.long)
    if max_len == 0:
        F_asset_tokens = torch.empty((B, 0, C), device=device, dtype=dtype)
        encoder_attention_mask = None
    else:
        F_asset_tokens = torch.zeros((B, max_len, C), device=device, dtype=dtype)
        encoder_attention_mask = torch.zeros((B, max_len), device=device, dtype=torch.bool)
        for b, kv_b in enumerate(kv_chunks):
            n = int(kv_b.shape[0])
            if n == 0:
                continue
            F_asset_tokens[b, :n, :] = kv_b.to(device=device, dtype=dtype)
            encoder_attention_mask[b, :n] = True

    return PretrainBundle(
        z_clean_n=z_clean_n,
        M_preserve=M_preserve,
        M_source=M_source,
        M_dest=M_dest,
        F_asset_tokens=F_asset_tokens,
        encoder_attention_mask=encoder_attention_mask,
        F_asset_lengths=lengths_t,
    )


def apply_uncond_drop(bundle: PretrainBundle, prob: float) -> PretrainBundle:
    """Per-sample Bernoulli drop of cross-attn KV (CFG training prerequisite).

    For dropped rows we mark every KV slot invalid so ``_prepare_asset_kv``
    injects ``null_kv`` (the uncond conditioning the model will see at inference).
    No-op when prob <= 0 or there are no KV tokens.
    """
    if prob <= 0.0 or bundle.F_asset_tokens.shape[1] == 0:
        return bundle
    batch_size = bundle.z_clean_n.shape[0]
    device = bundle.F_asset_tokens.device
    drop = torch.rand(batch_size, device=device) < float(prob)
    if not bool(drop.any().item()):
        return bundle
    num_tokens = bundle.F_asset_tokens.shape[1]
    if bundle.encoder_attention_mask is None:
        mask = torch.ones((batch_size, num_tokens), device=device, dtype=torch.bool)
    else:
        mask = bundle.encoder_attention_mask.clone()
    mask[drop] = False
    lengths = bundle.F_asset_lengths
    if lengths is not None:
        lengths = lengths.clone()
        lengths[drop.to(lengths.device)] = 0
    return PretrainBundle(
        z_clean_n=bundle.z_clean_n,
        M_preserve=bundle.M_preserve,
        M_source=bundle.M_source,
        M_dest=bundle.M_dest,
        F_asset_tokens=bundle.F_asset_tokens,
        encoder_attention_mask=mask,
        F_asset_lengths=lengths,
    )


def _validate_inputs(
    z_clean_n: torch.Tensor,
    dynamic_mask: torch.Tensor,
    image_tokens_last: torch.Tensor,
    patch_grid: tuple[int, int],
) -> None:
    if z_clean_n.ndim != 4:
        raise ValueError(f"z_clean_n must be [B,S,P,C], got {tuple(z_clean_n.shape)}")
    if dynamic_mask.ndim != 5 or dynamic_mask.shape[2] < 1:
        raise ValueError(f"dynamic_mask must be [B,S,3,H,W], got {tuple(dynamic_mask.shape)}")
    if image_tokens_last.ndim != 4:
        raise ValueError(f"image_tokens_last must be [B,S,P,C], got {tuple(image_tokens_last.shape)}")
    if z_clean_n.shape[:3] != image_tokens_last.shape[:3]:
        raise ValueError(
            "z_clean_n and image_tokens_last must share [B,S,P], "
            f"got {tuple(z_clean_n.shape[:3])} vs {tuple(image_tokens_last.shape[:3])}"
        )
    if dynamic_mask.shape[:2] != z_clean_n.shape[:2]:
        raise ValueError(
            "dynamic_mask and z_clean_n must share [B,S], "
            f"got {tuple(dynamic_mask.shape[:2])} vs {tuple(z_clean_n.shape[:2])}"
        )
    gh, gw = patch_grid
    if gh <= 0 or gw <= 0:
        raise ValueError(f"patch_grid must be positive, got {patch_grid}")
    if z_clean_n.shape[2] != gh * gw:
        raise ValueError(f"z_clean_n patch count {z_clean_n.shape[2]} != patch_grid product {gh * gw}")


def _downsample_dynamic_mask(
    dynamic_mask: torch.Tensor,
    patch_grid: tuple[int, int],
    pool_threshold: float,
) -> torch.Tensor:
    B, S = dynamic_mask.shape[:2]
    dyn = (dynamic_mask[:, :, 0] > 0.5).to(dtype=torch.float32)
    pooled = F.adaptive_avg_pool2d(dyn.reshape(B * S, 1, *dyn.shape[-2:]), patch_grid)
    return pooled.reshape(B, S, *patch_grid).gt(float(pool_threshold)).cpu()


def _pick_components(
    dyn_patch: torch.Tensor,
    *,
    K_max: int,
    min_inst_patches: int,
    max_inst_patches: int,
) -> list[list[list[list[int]]]]:
    B, S, gh, gw = dyn_patch.shape
    picked: list[list[list[list[int]]]] = []
    for b in range(B):
        picked_b: list[list[list[int]]] = []
        for s in range(S):
            comps = _connected_components(dyn_patch[b, s])
            filtered = []
            for comp in comps:
                area = len(comp)
                if area < min_inst_patches or area > max_inst_patches:
                    continue
                rows = [idx // gw for idx in comp]
                cols = [idx % gw for idx in comp]
                if min(rows) == 0 or max(rows) == gh - 1:
                    continue
                if min(cols) == 0 or max(cols) == gw - 1:
                    continue
                filtered.append((area, comp))
            filtered.sort(key=lambda item: item[0], reverse=True)
            picked_b.append([comp for _, comp in filtered[: max(0, K_max)]])
        picked.append(picked_b)
    return picked


def _connected_components(mask: torch.Tensor) -> list[list[int]]:
    gh, gw = mask.shape
    seen = [[False for _ in range(gw)] for _ in range(gh)]
    comps: list[list[int]] = []
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for r in range(gh):
        for c in range(gw):
            if seen[r][c] or not bool(mask[r, c].item()):
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r][c] = True
            comp: list[int] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append(rr * gw + cc)
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= gh or nc < 0 or nc >= gw:
                        continue
                    if seen[nr][nc] or not bool(mask[nr, nc].item()):
                        continue
                    seen[nr][nc] = True
                    queue.append((nr, nc))
            comps.append(comp)
    return comps
