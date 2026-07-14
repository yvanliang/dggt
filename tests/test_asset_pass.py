from __future__ import annotations

from pathlib import Path

import torch

from datasets.waymo_edit_dataset import transform_box_xyxy
from dggt.models.asset_pass import (
    AssetAggregatorPass,
    apply_pointer_fallback,
    build_asset_patch_valid_mask,
    compute_model_intrinsics,
    compute_runtime_patch_grid,
)
from dggt.models.gaussian_pointers import GaussianPointers, SRC_KIND_ASSET
from dggt.utils.gaussian_edit import Sim3Transform, apply_sim3_to_gaussian_dict


def test_asset_patch_valid_mask_uses_exact_alpha_support_without_dilation():
    alpha = torch.zeros(1, 2, 1, 28, 42)
    alpha[0, 0, 0, 2, 3] = 0.049
    alpha[0, 0, 0, 5, 16] = 0.05
    alpha[0, 1, 0, 20, 41] = 1.0

    mask = build_asset_patch_valid_mask(alpha, (2, 3))

    assert mask.shape == (1, 2, 6)
    assert torch.equal(mask[0, 0], torch.tensor([False, True, False, False, False, False]))
    assert torch.equal(mask[0, 1], torch.tensor([False, False, False, False, False, True]))


def _map_raw_box_with_intrinsics(
    box_xyxy: torch.Tensor,
    K_raw: torch.Tensor,
    K_model: torch.Tensor,
) -> torch.Tensor:
    corners = torch.tensor(
        [
            [box_xyxy[0], box_xyxy[1], 1.0],
            [box_xyxy[2], box_xyxy[1], 1.0],
            [box_xyxy[0], box_xyxy[3], 1.0],
            [box_xyxy[2], box_xyxy[3], 1.0],
        ],
        dtype=torch.float32,
    )
    K_raw_inv = torch.linalg.inv(K_raw.float())
    rays = corners @ K_raw_inv.T
    projected = rays @ K_model.float().T
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-6)
    return torch.tensor(
        [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
        dtype=torch.float32,
    )


def _make_stub_sample(tmp_path: Path) -> dict[str, object]:
    asset_a = tmp_path / "asset_a.spz"
    asset_b = tmp_path / "asset_b.spz"
    asset_a.write_text("stub\n", encoding="ascii")
    asset_b.write_text("stub\n", encoding="ascii")

    eye = torch.eye(4, dtype=torch.float32)
    pose_f0 = eye.clone()
    pose_f0[:3, 3] = torch.tensor([0.0, 0.0, 8.0])
    pose_f1 = eye.clone()
    pose_f1[:3, 3] = torch.tensor([1.0, 0.0, 8.0])

    obj_to_world = torch.stack(
        [
            torch.stack([pose_f0.clone(), pose_f1.clone()], dim=0),
            torch.stack([pose_f0.clone(), pose_f1.clone()], dim=0),
        ],
        dim=0,
    )

    return {
        "images_clean": torch.zeros((2, 3, 350, 518), dtype=torch.float32),
        "frame_indices": torch.tensor([10, 11], dtype=torch.long),
        "cam_ids": torch.tensor([0], dtype=torch.long),
        "camera_to_world_corrected": torch.stack(
            [eye.clone(), eye.clone()],
            dim=0,
        ).view(2, 1, 4, 4),
        "intrinsics": torch.tensor(
            [[[1600.0, 0.0, 960.0], [0.0, 1600.0, 640.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        "raw_image_size_hw": torch.tensor([[1280, 1920]], dtype=torch.long),
        "editable_object_indices": torch.tensor([0, 1], dtype=torch.long),
        "editable_object_count": torch.tensor(2, dtype=torch.long),
        "object_asset_valid_mask": torch.tensor([True, True], dtype=torch.bool),
        "object_asset_paths": [str(asset_a), str(asset_b)],
        "object_asset_image_valid_mask_selected": torch.tensor(
            [
                [True, True],
                [True, True],
            ],
            dtype=torch.bool,
        ),
        "object_asset_image_paths_selected": [
            [str(asset_a), str(asset_a)],
            [str(asset_b), str(asset_b)],
        ],
        "object_track_valid_mask_selected": torch.tensor(
            [[True, True], [True, True]],
            dtype=torch.bool,
        ),
        "object_obj_to_world_selected": obj_to_world,
        "object_box_size_selected": torch.tensor(
            [
                [[4.0, 2.0, 1.5], [4.0, 2.0, 1.5]],
                [[3.5, 1.8, 1.4], [3.5, 1.8, 1.4]],
            ],
            dtype=torch.float32,
        ),
    }


def _toy_gaussians(device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
    return {
        "means": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32, device=device),
        "colors": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=device),
        "opacities": torch.tensor([[0.5], [0.7]], dtype=torch.float32, device=device),
        "scales": torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32, device=device),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
    }


def test_compute_model_intrinsics_matches_dataset_box_transform():
    raw_hw = (1280, 1920)
    model_hw = (350, 518)
    K_raw = torch.tensor(
        [[1400.0, 0.0, 960.0], [0.0, 1350.0, 640.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    raw_box = torch.tensor([700.0, 420.0, 1230.0, 980.0], dtype=torch.float32)
    K_model = compute_model_intrinsics(K_raw, raw_hw, model_hw)
    box_from_intrinsics = _map_raw_box_with_intrinsics(raw_box, K_raw, K_model)
    box_from_dataset, _ = transform_box_xyxy(raw_box.tolist(), raw_hw, target_width=518)
    assert torch.allclose(
        box_from_intrinsics,
        torch.tensor(box_from_dataset.tolist(), dtype=torch.float32),
        atol=1e-4,
    )


def test_runtime_patch_grid_is_not_hardcoded_to_37():
    assert compute_runtime_patch_grid((350, 518), patch_size=14) == (25, 37)
    assert compute_runtime_patch_grid((518, 518), patch_size=14) == (37, 37)


def test_pointer_fallback_prefers_nearest_visible_frame_same_view():
    patch_idx_per_image = [
        torch.tensor([10, 20, 30], dtype=torch.long),
        torch.tensor([11, 21, 31], dtype=torch.long),
        torch.tensor([12, 22, 32], dtype=torch.long),
        torch.tensor([13, 23, 33], dtype=torch.long),
    ]
    visible_mask_per_image = [
        torch.tensor([True, False, False]),
        torch.tensor([False, False, True]),
        torch.tensor([False, True, False]),
        torch.tensor([False, False, False]),
    ]
    image_to_frame = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    image_to_view = torch.tensor([0, 0, 1, 0], dtype=torch.long)

    view_n, patch_idx = apply_pointer_fallback(
        patch_idx_per_image,
        visible_mask_per_image,
        image_to_frame,
        image_to_view,
    )

    # image 3 (frame=2, view=0) is invisible for all gaussians; fallback should
    # prefer the closest visible image, with same-view priority when distances tie.
    assert view_n[3].tolist() == [0, 2, 1]
    assert patch_idx[3].tolist() == [10, 22, 31]


def test_apply_sim3_to_gaussian_dict_rotates_translates_and_scales():
    gaussians = _toy_gaussians()
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    transform = Sim3Transform(
        scale=2.0,
        rotation=rotation,
        translation=torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32),
        mean_alignment_error=0.0,
    )

    out = apply_sim3_to_gaussian_dict(gaussians, transform)
    expected_means = 2.0 * (gaussians["means"] @ rotation.T) + transform.translation

    assert torch.allclose(out["means"], expected_means, atol=1e-5)
    assert torch.allclose(out["scales"], gaussians["scales"] * 2.0, atol=1e-5)
    assert torch.allclose(out["colors"], gaussians["colors"])
    assert torch.allclose(out["opacities"], gaussians["opacities"])


def test_asset_pass_forward_batches_objects_independently(tmp_path, monkeypatch):
    class StubAggregator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_start_idx = 2
            self.seen_shape: tuple[int, ...] | None = None

        def forward(self, images: torch.Tensor):
            self.seen_shape = tuple(images.shape)
            B, S, _C, H, W = images.shape
            patch_h, patch_w = compute_runtime_patch_grid((H, W), patch_size=14)
            patch_count = patch_h * patch_w
            total_tokens = self.patch_start_idx + patch_count
            levels = []
            for level_idx in range(24):
                levels.append(
                    torch.full(
                        (B, S, total_tokens, 3072),
                        float(level_idx + 1),
                        dtype=torch.float32,
                        device=images.device,
                    )
                )
            return levels, levels, levels, torch.zeros((B, S, patch_h, patch_w, 1024)), self.patch_start_idx

    def fake_load_asset_gaussians(_path: str, _cache):
        return {
            "means_raw": torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32),
            "colors": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
            "opacities": torch.tensor([[0.8], [0.9]], dtype=torch.float32),
            "scales": torch.tensor([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=torch.float32),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "vertex_count": torch.tensor([2], dtype=torch.long),
        }

    def fake_render_object_sequence(self, sample, slot_idx, cameras_waymo, model_hw, device, asset_cache):
        del sample, cameras_waymo, asset_cache
        H, W = model_hw
        gauss = {
            "means": torch.tensor([[0.0, 0.0, 4.0], [0.2, 0.0, 4.0]], dtype=torch.float32, device=device),
            "colors": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=device),
            "opacities": torch.tensor([[0.9], [0.8]], dtype=torch.float32, device=device),
            "scales": torch.tensor([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=torch.float32, device=device),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
        }
        rgb_seq = [
            torch.full((3, H, W), float(slot_idx + 1), dtype=torch.float32, device=device),
            torch.full((3, H, W), float(slot_idx + 1), dtype=torch.float32, device=device),
        ]
        alpha_seq = [
            torch.ones((1, H, W), dtype=torch.float32, device=device),
            torch.ones((1, H, W), dtype=torch.float32, device=device),
        ]
        depth_seq = [
            torch.full((H, W), 4.0, dtype=torch.float32, device=device),
            torch.full((H, W), 4.0, dtype=torch.float32, device=device),
        ]
        return [gauss, gauss], rgb_seq, alpha_seq, depth_seq

    def fake_annotate_object_pointers(
        self,
        object_id,
        gaussians_seq,
        cameras_waymo,
        patch_grid,
        alpha_seq,
        depth_seq,
        occlusion_test,
    ):
        del self, cameras_waymo, patch_grid, alpha_seq, depth_seq, occlusion_test
        out = []
        for image_idx, gauss in enumerate(gaussians_seq):
            n = gauss["means"].shape[0]
            out.append(
                GaussianPointers(
                    src_kind=torch.full((n,), SRC_KIND_ASSET, dtype=torch.int32),
                    object_id=torch.full((n,), int(object_id), dtype=torch.int32),
                    view_n=torch.full((n,), int(image_idx), dtype=torch.int32),
                    patch_idx=torch.arange(n, dtype=torch.int32),
                    visible_mask=torch.ones((n,), dtype=torch.bool),
                )
            )
        return out

    monkeypatch.setattr("dggt.models.asset_pass.load_asset_gaussians", fake_load_asset_gaussians)
    monkeypatch.setattr(AssetAggregatorPass, "_render_object_sequence", fake_render_object_sequence)
    monkeypatch.setattr(AssetAggregatorPass, "_annotate_object_pointers", fake_annotate_object_pointers)

    sample = _make_stub_sample(tmp_path)
    aggregator = StubAggregator()
    module = AssetAggregatorPass(aggregator)
    result = module(sample)

    assert aggregator.seen_shape == (2, 2, 3, 350, 518)
    assert result.object_keys == [0, 1]
    assert result.patch_grid == (25, 37)
    assert result.patch_start_idx == 2

    for slot_idx in result.object_keys:
        level0 = result.F_g_lut_asset[slot_idx][0]
        assert level0.shape == (1, 2, 25 * 37, 3072)
        assert result.I_asset[slot_idx].shape == (1, 2, 3, 350, 518)
        assert result.A_asset[slot_idx].shape == (1, 2, 1, 350, 518)
        assert len(result.ptr_asset[slot_idx]) == 2
        assert result.ptr_asset[slot_idx][0].patch_idx.shape[0] == 2

    assert result.G_asset_dggt is None


def test_asset_pass_resolves_per_image_asset_paths(tmp_path):
    sample = _make_stub_sample(tmp_path)
    module = AssetAggregatorPass(torch.nn.Identity())

    assert module._resolve_asset_path_for_image(sample, 0, 0).endswith("asset_a.spz")
    assert module._resolve_asset_path_for_image(sample, 1, 1).endswith("asset_b.spz")

    sample["object_asset_image_valid_mask_selected"] = torch.tensor(
        [[True, False], [False, False]],
        dtype=torch.bool,
    )
    assert module._resolve_asset_path_for_image(sample, 0, 1) == ""
    assert not module._is_valid_asset_slot(sample, 1)
