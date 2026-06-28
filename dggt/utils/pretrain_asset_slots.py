from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_dynamic_mask_shape(
    dynamic_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if dynamic_mask is None:
        return None
    mask = dynamic_mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(2)
    elif mask.ndim == 4:
        if int(mask.shape[0]) == batch_size and int(mask.shape[1]) == seq_len:
            mask = mask.unsqueeze(2)
        elif int(mask.shape[0]) == seq_len:
            mask = mask.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported dynamic_mask shape {tuple(dynamic_mask.shape)}")
    elif mask.ndim != 5:
        raise ValueError(f"Unsupported dynamic_mask shape {tuple(dynamic_mask.shape)}")
    if int(mask.shape[0]) == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1, -1)
    if int(mask.shape[0]) != batch_size or int(mask.shape[1]) != seq_len:
        raise ValueError(
            f"dynamic_mask batch/sequence shape {tuple(mask.shape[:2])} != {(batch_size, seq_len)}"
        )
    return mask.max(dim=2).values


def _dynamic_mask_to_patch_grid(
    dynamic_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    patch_grid: tuple[int, int] | list[int],
    device: torch.device,
    threshold: float = 0.5,
) -> torch.Tensor:
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    mask = _normalize_dynamic_mask_shape(
        dynamic_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    if mask is None:
        return torch.zeros((batch_size, seq_len, gh * gw), device=device, dtype=torch.bool)
    pooled = F.adaptive_max_pool2d(mask.reshape(batch_size * seq_len, 1, *mask.shape[-2:]), (gh, gw))
    return pooled.reshape(batch_size, seq_len, gh * gw).gt(float(threshold))


def _connected_components_4n(union_mask: torch.Tensor, patch_grid: tuple[int, int] | list[int]) -> list[list[int]]:
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    grid = union_mask.detach().to(device="cpu", dtype=torch.bool).reshape(gh, gw)
    visited = torch.zeros((gh, gw), dtype=torch.bool)
    components: list[list[int]] = []
    for y in range(gh):
        for x in range(gw):
            if not bool(grid[y, x].item()) or bool(visited[y, x].item()):
                continue
            stack = [(y, x)]
            visited[y, x] = True
            comp: list[int] = []
            while stack:
                cy, cx = stack.pop()
                comp.append(cy * gw + cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= gh or nx < 0 or nx >= gw:
                        continue
                    if bool(grid[ny, nx].item()) and not bool(visited[ny, nx].item()):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(comp)
    components.sort(key=len, reverse=True)
    return components


def build_pretrain_asset_slots_from_dynamic_mask(
    z_clean_n: torch.Tensor,
    dynamic_mask: torch.Tensor | None,
    patch_grid: tuple[int, int] | list[int],
    *,
    max_assets: int = 5,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    B, S, P, C = z_clean_n.shape
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    if gh * gw != int(P):
        raise ValueError(f"patch_grid={patch_grid} is incompatible with P={P}")
    max_assets = int(max_assets)
    asset_tokens = z_clean_n.new_zeros((B, max_assets, S, P, C))
    asset_mask = torch.zeros((B, max_assets, S, P), device=z_clean_n.device, dtype=torch.bool)
    lengths = torch.zeros((B,), device=z_clean_n.device, dtype=torch.long)
    kinds: list[str] = []
    patch_dynamic = _dynamic_mask_to_patch_grid(
        dynamic_mask,
        batch_size=B,
        seq_len=S,
        patch_grid=(gh, gw),
        device=z_clean_n.device,
        threshold=threshold,
    )
    for row in range(B):
        components = _connected_components_4n(patch_dynamic[row].any(dim=0), (gh, gw))[:max_assets]
        slot = 0
        for comp in components:
            comp_idx = torch.tensor(comp, device=z_clean_n.device, dtype=torch.long)
            comp_mask = torch.zeros((S, P), device=z_clean_n.device, dtype=torch.bool)
            comp_mask[:, comp_idx] = patch_dynamic[row, :, comp_idx]
            if not bool(comp_mask.any().item()):
                continue
            asset_mask[row, slot] = comp_mask
            asset_tokens[row, slot] = torch.where(comp_mask.unsqueeze(-1), z_clean_n[row], asset_tokens[row, slot])
            slot += 1
            if slot >= max_assets:
                break
        lengths[row] = slot
        kinds.append("mode_a" if slot > 0 else "none")
    return asset_tokens, asset_mask, lengths, kinds
