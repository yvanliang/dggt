from __future__ import annotations

import torch

from dggt.utils.gaussian_edit import CleanSceneState


def estimate_ground_plane_per_frame(
    clean_state: CleanSceneState,
    *,
    quantile: float = 0.95,
    dynamic_prob_threshold: float = 0.5,
) -> torch.Tensor:
    """Estimate a per-frame DGGT ground y coordinate from static non-sky points.

    `CleanSceneState.means` already contains only non-sky valid pixels. DGGT's
    camera convention has image-down roughly aligned with +Y, so ground support
    is near the high-Y tail rather than the low-Y tail. We use source-frame ids
    to keep the estimate local to each 29-frame clip frame and fall back to the
    global static quantile for sparse frames.
    """
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    means = clean_state.means.detach().cpu().float()
    if means.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    source_frame_ids = clean_state.source_frame_ids.detach().cpu().long()
    if source_frame_ids.numel() != means.shape[0]:
        raise ValueError(
            "CleanSceneState source_frame_ids must have one entry per Gaussian: "
            f"{source_frame_ids.numel()} vs {means.shape[0]}"
        )

    dynamic_prob = clean_state.dynamic_prob.detach().cpu().float()
    if dynamic_prob.numel() != means.shape[0]:
        static_mask = torch.ones((means.shape[0],), dtype=torch.bool)
    else:
        static_mask = dynamic_prob < float(dynamic_prob_threshold)

    static_y = means[static_mask, 1]
    if static_y.numel() == 0:
        static_y = means[:, 1]
    global_ground_y = torch.quantile(static_y, float(quantile)).to(torch.float32)

    num_frames = int(source_frame_ids.max().item()) + 1 if source_frame_ids.numel() > 0 else 0
    ground_y = torch.full((num_frames,), float(global_ground_y.item()), dtype=torch.float32)
    for frame_idx in range(num_frames):
        frame_mask = static_mask & (source_frame_ids == frame_idx)
        frame_y = means[frame_mask, 1]
        if frame_y.numel() == 0:
            continue
        ground_y[frame_idx] = torch.quantile(frame_y, float(quantile)).to(torch.float32)
    return ground_y
