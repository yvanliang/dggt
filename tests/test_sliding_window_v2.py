from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dggt.utils.camera_generation import (
    CAMERA_GENERATION_DIM,
    CAMERA_GENERATION_REPRESENTATION,
    camera_anchor_mask,
)
from dggt.utils.sliding_window import (
    cosine_coverage,
    cosine_window,
    default_window_stride,
    resolve_offline_window,
    scene_global_window_weight,
    window_slices,
)
from train_scene_flow_pretrain import (
    _validation_sliding_params,
    cfg_sample_pretrain_latents,
)


def test_sliding_window_requires_overlap_and_covers_tail() -> None:
    assert window_slices(5, 8, 8) == [(0, 5)]
    assert window_slices(17, 8, 0) == [(0, 8), (4, 12), (8, 16), (9, 17)]
    assert window_slices(10, 8, 3)[-1] == (2, 10)
    with pytest.raises(ValueError, match="require overlap"):
        window_slices(16, 8, 8)
    with pytest.raises(ValueError, match="require overlap"):
        window_slices(16, 8, 9)


def test_scene_global_sky_weights_give_equal_per_frame_contribution() -> None:
    windows = window_slices(17, 8, 4)
    coverage = cosine_coverage(17, windows)
    per_frame = torch.zeros(17)
    total_global_weight = 0.0
    for start, end in windows:
        local = cosine_window(end - start)
        per_frame[start:end] += local / coverage[start:end]
        total_global_weight += float(scene_global_window_weight(start, end, coverage))
    assert torch.allclose(per_frame, torch.ones_like(per_frame), atol=1e-6)
    assert total_global_weight == pytest.approx(17.0, abs=1e-5)


def test_offline_window_policy_automatically_bounds_long_requests() -> None:
    assert default_window_stride(10) == 7
    assert resolve_offline_window(10, 0, 0) == (10, 7, False)
    assert resolve_offline_window(11, 0, 0) == (10, 7, True)
    assert resolve_offline_window(29, 29, 7) == (10, 7, True)
    assert resolve_offline_window(7, 4, 2) == (4, 2, True)
    with pytest.raises(ValueError, match="requires overlap"):
        resolve_offline_window(11, 10, 10)


def test_training_validation_defensively_caps_tokenizer_windows_to_ten() -> None:
    args = SimpleNamespace(val_sliding_window=29, val_sliding_stride=0)
    assert _validation_sliding_params(args, 29) == (10, 7)
    with pytest.raises(ValueError, match="stride < bounded window"):
        _validation_sliding_params(
            SimpleNamespace(val_sliding_window=29, val_sliding_stride=10), 29
        )


class _GaugeSamplerFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            prediction_type="v",
            t_eps=0.05,
            gauge_gen_dim=3,
            camera_generation_representation="waymo_metric_relative_se3_rot6d_v4",
        )
        self.gauge_shapes: list[tuple[int, ...]] = []

    def denormalize_gauge(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def forward(self, z_t: torch.Tensor, sigma: torch.Tensor, *args, **kwargs):
        gauge = kwargs.get("gauge_gen_tokens")
        assert gauge is not None
        self.gauge_shapes.append(tuple(gauge.shape))
        return {
            # Pointwise, window-independent video velocity lets both samplers
            # exercise their real Euler state while remaining schedule invariant.
            "video": 0.1 * z_t + 0.02,
            # A window-independent scene-global velocity makes the exact
            # integration result schedule invariant. The test still fails if
            # either sampler slices the token, skips CFG combination, or does
            # per-window rather than one global Euler update.
            "gauge": 0.25 * gauge + 0.125,
        }


def _gauge_bundle(seq_len: int = 9) -> SimpleNamespace:
    clean = torch.zeros(1, seq_len, 2, 4)
    edit = torch.ones(1, seq_len, 2, 1)
    return SimpleNamespace(
        z_clean_n=clean,
        z_splat_n=torch.zeros_like(clean),
        M_preserve=torch.zeros_like(edit),
        M_source=torch.zeros_like(edit),
        M_dest=edit,
        F_asset_tokens=torch.zeros(1, 0, 4),
        encoder_attention_mask=None,
        factorized_asset_condition=None,
        scene_gauge_clean_n=torch.zeros(1, 1, 3),
        frame_ids=torch.arange(seq_len).view(1, seq_len),
    )


def _sampler_args(*, window: int, stride: int, guidance: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        guidance_scale=float(guidance),
        asset_control_guidance_scale=1.0,
        camera_guidance_scale=1.0,
        camera_text_guidance_scale=1.0,
        val_sample_steps=4,
        shift=1.0,
        seed=73,
        val_sliding_window=window,
        val_sliding_stride=stride,
    )


@pytest.mark.parametrize("window,stride", [(4, 2), (5, 3)])
@pytest.mark.parametrize("guidance", [1.0, 2.5])
def test_scene_global_gauge_sampler_matches_non_sliding_path(
    window: int, stride: int, guidance: float
) -> None:
    bundle = _gauge_bundle()
    direct_flow = _GaugeSamplerFlow()
    sliding_flow = _GaugeSamplerFlow()
    direct = cfg_sample_pretrain_latents(
        direct_flow,
        bundle,
        _sampler_args(window=0, stride=0, guidance=guidance),
        step=11,
        device=torch.device("cpu"),
        return_gauge=True,
    )
    sliding = cfg_sample_pretrain_latents(
        sliding_flow,
        bundle,
        _sampler_args(window=window, stride=stride, guidance=guidance),
        step=11,
        device=torch.device("cpu"),
        return_gauge=True,
    )
    torch.testing.assert_close(sliding.video, direct.video, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(sliding.gauge, direct.gauge, atol=1.0e-6, rtol=1.0e-6)
    assert sliding_flow.gauge_shapes
    assert set(sliding_flow.gauge_shapes) == {(1, 1, 3)}


class _ParityTextEncoder(torch.nn.Module):
    def forward(self, captions):
        values = torch.tensor(
            [0.0 if str(caption) == "" else 1.0 for caption in captions],
            dtype=torch.float32,
        ).view(len(captions), 1, 1)
        return {
            "tokens": values,
            "attention_mask": torch.ones(values.shape[:2], dtype=torch.bool),
        }


class _AllModalityParityFlow(torch.nn.Module):
    """Window-equivariant fake with genuinely distinct CFG branches."""

    def __init__(self, prediction_type: str) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            prediction_type=str(prediction_type),
            t_eps=0.05,
            camera_gen_dim=CAMERA_GENERATION_DIM,
            camera_generation_representation=CAMERA_GENERATION_REPRESENTATION,
            gauge_gen_dim=3,
        )
        self.gauge_shapes: list[tuple[int, ...]] = []
        self.camera_shapes: list[tuple[int, ...]] = []
        self.branch_signatures: set[tuple[float, tuple[str, ...], tuple[str, ...]]] = set()

    @staticmethod
    def _kind_values(kinds, batch_size: int, default: str) -> tuple[str, ...]:
        if kinds is None:
            return (default,) * int(batch_size)
        if isinstance(kinds, str):
            return (str(kinds),) * int(batch_size)
        return tuple(str(kind) for kind in kinds)

    def denormalize_camera(
        self, value: torch.Tensor, anchor_mask: torch.Tensor
    ) -> torch.Tensor:
        # Make the role-aware final conversion observable in the parity check.
        return value + 0.01 * anchor_mask.to(value).unsqueeze(-1)

    def denormalize_gauge(self, value: torch.Tensor) -> torch.Tensor:
        return value + value.new_tensor([[[0.03, -0.02, 0.01]]])

    def forward(self, z_t: torch.Tensor, sigma: torch.Tensor, *args, **kwargs):
        del sigma, args
        batch_size, seq_len = int(z_t.shape[0]), int(z_t.shape[1])
        text = kwargs.get("text_tokens")
        text_value = (
            z_t.new_zeros((batch_size,))
            if text is None
            else text.to(z_t).mean(dim=(1, 2))
        )
        asset_kinds = self._kind_values(
            kwargs.get("asset_condition_kind"), batch_size, "asset_uncond"
        )
        camera_kinds = self._kind_values(
            kwargs.get("camera_condition_kind"), batch_size, "camera_uncond"
        )
        asset_present = z_t.new_tensor(
            [0.0 if kind.endswith("uncond") else 1.0 for kind in asset_kinds]
        )
        camera_present = z_t.new_tensor(
            [0.0 if kind.endswith("uncond") else 1.0 for kind in camera_kinds]
        )
        self.branch_signatures.add(
            (float(text_value[0].item()), asset_kinds, camera_kinds)
        )

        asset_tokens = kwargs.get("F_asset_tokens")
        if torch.is_tensor(asset_tokens) and asset_tokens.ndim == 5:
            asset_per_frame = asset_tokens.to(z_t).mean(dim=(1, 3, 4))
        else:
            asset_per_frame = z_t.new_zeros((batch_size, seq_len))
        camera_condition = kwargs.get("camera_condition_tokens")
        if torch.is_tensor(camera_condition):
            camera_per_frame = camera_condition.to(z_t).mean(dim=-1)
        else:
            camera_per_frame = z_t.new_zeros((batch_size, seq_len))
        per_frame_bias = (
            text_value[:, None]
            + asset_present[:, None] * (2.0 + 0.01 * asset_per_frame)
            + camera_present[:, None] * (3.0 + 0.01 * camera_per_frame)
        ) * 0.01
        global_bias = (
            text_value + 2.0 * asset_present + 3.0 * camera_present
        ).view(batch_size, 1, 1) * 0.01

        result = {
            "video": 0.10 * z_t + per_frame_bias[:, :, None, None],
        }
        camera = kwargs.get("camera_gen_tokens")
        if torch.is_tensor(camera):
            self.camera_shapes.append(tuple(camera.shape))
            result["camera"] = 0.20 * camera + per_frame_bias[:, :, None]
        sky = kwargs.get("sky_gen_tokens")
        if torch.is_tensor(sky):
            result["sky"] = 0.30 * sky + global_bias
        gauge = kwargs.get("gauge_gen_tokens")
        if torch.is_tensor(gauge):
            self.gauge_shapes.append(tuple(gauge.shape))
            result["gauge"] = 0.40 * gauge + global_bias
        if kwargs.get("return_sky_mask", False):
            patch_logits = (
                0.05 * z_t.sum(dim=-1, keepdim=True)
                + per_frame_bias[:, :, None, None]
            )
            result["sky_mask_logits"] = patch_logits
            result["sky_mask_refined_logits"] = (
                patch_logits.mean(dim=2)
                .view(batch_size, seq_len, 1, 1, 1)
                .expand(-1, -1, -1, 2, 3)
            )
        return result


def _all_modality_bundle(seq_len: int = 29) -> SimpleNamespace:
    video = torch.zeros(1, seq_len, 2, 4)
    edit = torch.ones(1, seq_len, 2, 1)
    frame_signal = torch.arange(seq_len, dtype=torch.float32)
    asset_tokens = frame_signal.view(1, 1, seq_len, 1, 1).expand(
        1, 2, seq_len, 2, 4
    )
    camera_tokens = frame_signal.view(1, seq_len, 1).expand(1, seq_len, 3)
    identity = torch.eye(4).view(1, 4, 4)
    return SimpleNamespace(
        z_clean_n=video,
        z_splat_n=torch.zeros_like(video),
        M_preserve=torch.zeros_like(edit),
        M_source=torch.zeros_like(edit),
        M_dest=edit,
        F_asset_tokens=asset_tokens,
        encoder_attention_mask=torch.ones(
            asset_tokens.shape[:-1], dtype=torch.bool
        ),
        factorized_asset_condition=None,
        asset_condition_kind=["mode_a"],
        camera_condition_tokens=camera_tokens,
        camera_attention_mask=torch.ones(1, seq_len, dtype=torch.bool),
        camera_condition_kind=["camera"],
        camera_target_clean_n=torch.zeros(
            1, seq_len, CAMERA_GENERATION_DIM
        ),
        camera_gen_anchor_mask=camera_anchor_mask(1, seq_len),
        camera_previous_c2w_metric=identity,
        camera_trajectory_anchor_to_world_metric=identity.clone(),
        scene_gauge_clean_n=torch.zeros(1, 1, 3),
        frame_ids=torch.arange(seq_len).view(1, seq_len),
        captions=["conditioned road"],
    )


def _all_modality_args(
    *, prediction_type: str, guidance: float, sliding: bool
) -> SimpleNamespace:
    del prediction_type  # carried by the fake model, not sampler args
    return SimpleNamespace(
        guidance_scale=float(guidance),
        asset_control_guidance_scale=2.0,
        camera_guidance_scale=3.0,
        camera_text_guidance_scale=1.7,
        val_sample_steps=4,
        shift=1.0,
        seed=271,
        val_sliding_window=10 if sliding else 0,
        val_sliding_stride=7 if sliding else 0,
        sky_grid_h=1,
        sky_grid_w=2,
    )


@pytest.mark.parametrize("prediction_type", ["v", "x"])
@pytest.mark.parametrize("guidance", [1.0, 2.5])
def test_29_frame_all_modality_sampler_matches_non_sliding_path(
    prediction_type: str, guidance: float
) -> None:
    bundle = _all_modality_bundle()
    direct_flow = _AllModalityParityFlow(prediction_type)
    sliding_flow = _AllModalityParityFlow(prediction_type)
    common = dict(
        step=17,
        device=torch.device("cpu"),
        guidance_scale=guidance,
        text_encoder=_ParityTextEncoder(),
        return_camera=True,
        return_sky=True,
        return_gauge=True,
        return_sky_mask=True,
    )
    direct = cfg_sample_pretrain_latents(
        direct_flow,
        bundle,
        _all_modality_args(
            prediction_type=prediction_type, guidance=guidance, sliding=False
        ),
        **common,
    )
    sliding = cfg_sample_pretrain_latents(
        sliding_flow,
        bundle,
        _all_modality_args(
            prediction_type=prediction_type, guidance=guidance, sliding=True
        ),
        **common,
    )

    expected_fields = {
        "video",
        "camera_state_metric",
        "camera_anchor_mask",
        "camera_initial_c2w_metric",
        "camera_trajectory_anchor_to_world_metric",
        "sky",
        "gauge",
        "sky_mask_logits",
        "sky_mask_patch",
        "sky_mask_refined_logits",
        "sky_mask_refined",
    }
    assert set(vars(direct)) == expected_fields
    assert set(vars(sliding)) == expected_fields
    for field in sorted(expected_fields):
        direct_value = getattr(direct, field)
        sliding_value = getattr(sliding, field)
        if torch.is_tensor(direct_value):
            torch.testing.assert_close(
                sliding_value, direct_value, atol=2.0e-6, rtol=2.0e-6
            )
        else:
            assert sliding_value == direct_value

    assert sliding.camera_state_metric.shape == (1, 29, CAMERA_GENERATION_DIM)
    assert sliding.gauge.shape == (1, 1, 3)
    assert sliding.sky.shape == (1, 2, 12)
    assert sliding.sky_mask_patch.shape == (1, 29, 2, 1)
    assert sliding.sky_mask_refined.shape == (1, 29, 1, 2, 3)
    assert set(sliding_flow.gauge_shapes) == {(1, 1, 3)}
    assert set(sliding_flow.camera_shapes) == {(1, 10, CAMERA_GENERATION_DIM)}
    expected_branches = {
        (1.0, ("mode_a",), ("camera",)),
        (0.0, ("mode_a",), ("camera",)),
        (1.0, ("asset_uncond",), ("camera_uncond",)),
        (1.0, ("mode_a",), ("camera_uncond",)),
    }
    assert expected_branches.issubset(direct_flow.branch_signatures)
    assert expected_branches.issubset(sliding_flow.branch_signatures)
