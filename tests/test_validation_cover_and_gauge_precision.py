"""What made ``validation/gauge/log_scale_error`` freeze for 4000 steps.

The v6 run logged that series as ``0.0161888599`` at steps 2000, 4000 and 6000
-- bit-identical to ten significant figures -- while ``gauge/fov_error``, read
off the same tensor by the same function, moved by 4x.  Two independent defects
combined:

* the scalar validation loader advanced by the number of samples consumed, and
  the trunk-major index puts the window offset innermost, so four validations
  stayed inside scene 000 trunk 0.  ``log_metric_scale`` is a per-scene/trunk
  constant, so the target could not move;
* the gauge was decoded under bf16 autocast, quantizing that scale to ~0.25% of
  the scene, so the prediction could not move either at single-sample
  granularity.

These tests pin both halves, plus the two visualization-level consequences: the
sampled scenes rotating out from under every ``sample_*`` series, and the mosaic
latent rows each getting their own PCA basis.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import textwrap

import pytest
import torch
from torch import nn

import train_scene_flow_pretrain as pretrain_train
from datasets.dataset import WaymoOpenDataset
from dggt.models.scene_flow import decode_scene_gauge

SPREAD = pretrain_train.SpreadSequentialSampler


class _Dataset:
    """Length is all the samplers read."""

    def __init__(self, length: int) -> None:
        self._length = int(length)

    def __len__(self) -> int:
        return self._length


# ------------------------------------------------------------------ #
# The scalar validation cover                                         #
# ------------------------------------------------------------------ #
def test_the_scalar_cover_is_identical_at_every_validation() -> None:
    """A validation loss is only readable if it scores the same windows."""

    sampler = SPREAD(_Dataset(2400), 8)
    assert list(sampler) == list(sampler)
    assert list(SPREAD(_Dataset(2400), 8)) == list(sampler)


def test_the_cover_visits_distinct_samples() -> None:
    sampler = SPREAD(_Dataset(2400), 8)
    assert len(set(sampler)) == 8
    assert len(sampler) == 8


def test_the_cover_does_not_alias_onto_a_few_scenes() -> None:
    """The regression even spacing would have reintroduced.

    The trunk-major length factorizes as ``trunks * scenes * offsets`` with the
    offset innermost, so any stride sharing a factor with the scene axis
    revisits the same scenes.  With 100 scenes and the four pretraining
    offsets, ``i * len // 8`` gives 0, 300, 600, ... -- four scenes, each
    sampled twice.
    """

    trunks, scenes, offsets = 6, 100, 4
    length = trunks * scenes * offsets

    def scenes_covered(indices) -> int:
        return len({(index // offsets) % scenes for index in indices})

    for count in (2, 3, 4, 5, 6, 8, 12, 16):
        assert scenes_covered(SPREAD(_Dataset(length), count)) == count

    evenly_spaced = [(i * length) // 8 for i in range(8)]
    assert scenes_covered(evenly_spaced) < 8


def test_a_cover_larger_than_the_split_falls_back_to_the_whole_split() -> None:
    assert list(SPREAD(_Dataset(3), 8)) == [0, 1, 2]
    assert list(SPREAD(_Dataset(0), 8)) == []


def test_a_non_positive_cover_is_rejected() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        SPREAD(_Dataset(100), 0)


# ------------------------------------------------------------------ #
# The sampled scene set                                               #
# ------------------------------------------------------------------ #
def _args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**overrides)


def _trunk_major_index(
    *, num_scenes: int = 100, trunks: int = 6
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (scene_index, trunk_index)
        for trunk_index in range(trunks)
        for scene_index in range(num_scenes)
    )


def _index(
    slot: int,
    validation_index: int,
    *,
    scene_count: int = 10,
    num_scenes: int = 100,
    trunks: int = 6,
) -> int:
    return pretrain_train.validation_scene_dataset_index(
        slot,
        scene_count=scene_count,
        validation_index=validation_index,
        trunk_major_index=_trunk_major_index(
            num_scenes=num_scenes, trunks=trunks
        ),
    )


def _entry(
    slot: int,
    validation_index: int,
    **kwargs,
) -> tuple[int, int]:
    index = _index(slot, validation_index, **kwargs)
    return _trunk_major_index(
        num_scenes=int(kwargs.get("num_scenes", 100)),
        trunks=int(kwargs.get("trunks", 6)),
    )[index]


def test_the_pinned_half_never_moves() -> None:
    """Slot k of the pinned half must be the same scene at every step."""

    assert pretrain_train.validation_pinned_scene_count(10) == 5
    for validation_index in range(6):
        assert [_entry(slot, validation_index) for slot in range(5)] == [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
        ]


def test_the_rotating_half_advances_every_validation() -> None:
    assert [_index(slot, 0) for slot in range(5, 10)] == [5, 6, 7, 8, 9]
    assert [_index(slot, 1) for slot in range(5, 10)] == [10, 11, 12, 13, 14]
    assert [_index(slot, 2) for slot in range(5, 10)] == [15, 16, 17, 18, 19]


def test_a_rotating_slot_never_lands_on_a_pinned_scene_identity() -> None:
    """A collision would silently duplicate a mosaic panel."""

    pinned = set(range(pretrain_train.validation_pinned_scene_count(10)))
    for validation_index in range(400):  # past the wrap of a 600-long split
        for slot in range(5, 10):
            scene_index, _ = _entry(slot, validation_index)
            assert scene_index not in pinned


def test_rotating_scenes_advance_to_their_next_trunk_after_scene_wrap() -> None:
    """The old flattened-index fix revisited pinned scene 0 at this point."""

    assert [_entry(slot, 18) for slot in range(5, 10)] == [
        (95, 0),
        (96, 0),
        (97, 0),
        (98, 0),
        (99, 0),
    ]
    assert [_entry(slot, 19) for slot in range(5, 10)] == [
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
    ]


def test_the_two_scene_fallback_keeps_one_of_each() -> None:
    assert pretrain_train.validation_pinned_scene_count(2) == 1
    assert [_entry(0, i, scene_count=2) for i in range(4)] == [(0, 0)] * 4
    assert [_entry(1, i, scene_count=2) for i in range(4)] == [
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
    ]


def test_too_few_distinct_usable_scenes_is_rejected_even_with_many_trunks() -> None:
    index = _trunk_major_index(num_scenes=4, trunks=6)
    assert len(index) == 24
    assert pretrain_train.validation_available_scene_count(index) == 4
    with pytest.raises(ValueError, match="fewer distinct usable scenes"):
        pretrain_train.validation_scene_dataset_index(
            0,
            scene_count=10,
            validation_index=0,
            trunk_major_index=index,
        )


def test_a_slot_outside_the_scene_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside scene_count"):
        _index(10, 0)


def test_a_negative_validation_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="validation_index must be non-negative"):
        _index(0, -1)


def test_the_pretrain_defaults_match_the_thirty_rank_layout() -> None:
    args = pretrain_train.build_argparser().parse_args(
        [
            "--image_dir", "/tmp/training",
            "--hdmap_root", "/tmp/training_hdmap",
            "--dggt_ckpt_path", "/tmp/dggt.pt",
            "--scene_gauge_path", "/tmp/gauge.json",
            "--tokenizer_ckpt_path", "/tmp/tok.pt",
            "--feature_stats_path", "/tmp/stats.pt",
            "--pullback_calibration_path", "/tmp/pullback.json",
            "--log_dir", "/tmp/logs",
        ]
    )
    assert args.val_inference_scenes == 10
    # One batch made every validation/* scalar a single window of a single
    # scene; the fixed cover only pays off when it spans more than one.
    assert args.val_batches == 8


def test_only_validation_datasets_enable_deterministic_layout_references() -> None:
    """Both fixed validation covers opt in; stochastic training stays unchanged."""

    signature = inspect.signature(WaymoOpenDataset.__init__)
    assert signature.parameters["deterministic_layout_reference"].default is False

    tree = ast.parse(textwrap.dedent(inspect.getsource(pretrain_train.main)))
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "WaymoOpenDataset"
        ),
        key=lambda node: node.lineno,
    )
    assert len(calls) == 3  # train, scalar validation, long-form validation

    def deterministic_keyword(call: ast.Call):
        for keyword in call.keywords:
            if keyword.arg == "deterministic_layout_reference":
                assert isinstance(keyword.value, ast.Constant)
                return keyword.value.value
        return None

    assert [deterministic_keyword(call) for call in calls] == [None, True, True]


def test_pinned_sampling_noise_is_stable_across_global_steps() -> None:
    """Changing the model step must not change the fixed scene's ODE draw."""

    args = _args(seed=123)
    sampling_seed = pretrain_train.validation_scene_sampling_seed(
        args.seed, scene_offset=3
    )
    at_step_2k = pretrain_train.make_pretrain_sampling_generator(
        torch.device("cpu"), args, 2000, sampling_seed=sampling_seed
    )
    at_step_6k = pretrain_train.make_pretrain_sampling_generator(
        torch.device("cpu"), args, 6000, sampling_seed=sampling_seed
    )
    assert torch.equal(
        torch.randn(64, generator=at_step_2k),
        torch.randn(64, generator=at_step_6k),
    )

    # Without the validation override, retain the legacy dataset-index/step
    # behavior used by the standalone inference caller.
    legacy_2k = pretrain_train.make_pretrain_sampling_generator(
        torch.device("cpu"), args, 2000
    )
    legacy_6k = pretrain_train.make_pretrain_sampling_generator(
        torch.device("cpu"), args, 6000
    )
    assert not torch.equal(
        torch.randn(64, generator=legacy_2k),
        torch.randn(64, generator=legacy_6k),
    )


def test_cfg_scales_share_noise_but_different_scene_slots_do_not() -> None:
    args = _args(seed=123)
    slot_0_seed = pretrain_train.validation_scene_sampling_seed(args.seed, 0)
    slot_1_seed = pretrain_train.validation_scene_sampling_seed(args.seed, 1)
    assert slot_0_seed != slot_1_seed

    first_scale = pretrain_train.make_pretrain_sampling_generator(
        torch.device("cpu"), args, 2000, sampling_seed=slot_0_seed
    )
    other_scale = pretrain_train.make_pretrain_sampling_generator(
        torch.device("cpu"), args, 2000, sampling_seed=slot_0_seed
    )
    other_scene = pretrain_train.make_pretrain_sampling_generator(
        torch.device("cpu"), args, 2000, sampling_seed=slot_1_seed
    )
    first_draw = torch.randn(64, generator=first_scale)
    assert torch.equal(first_draw, torch.randn(64, generator=other_scale))
    assert not torch.equal(first_draw, torch.randn(64, generator=other_scene))


def test_rotating_scene_metrics_are_filtered_before_rank_merge() -> None:
    """Mosaics rotate, but only the five pinned slots contribute numbers."""

    pinned_payload = {
        "metrics": pretrain_train.validation_scene_metrics_for_merge(
            {"sample_latent_mse": 1.5}, scene_offset=0, scene_count=10
        ),
        "mosaic_rows": [],
    }
    rotating_payload = {
        "metrics": pretrain_train.validation_scene_metrics_for_merge(
            {"sample_latent_mse": 999.0}, scene_offset=5, scene_count=10
        ),
        "mosaic_rows": [],
    }
    assert rotating_payload["metrics"] == {}
    merged, _ = pretrain_train.merge_validation_rank_results(
        [pinned_payload, rotating_payload]
    )
    assert merged == {"sample_latent_mse": pytest.approx(1.5)}


def test_boundary_loss_fallback_and_argparse_share_one_default() -> None:
    assert pretrain_train.sky_mask_refine_boundary_loss_weight(_args()) == pytest.approx(
        pretrain_train.SKY_MASK_REFINE_BOUNDARY_LOSS_WEIGHT_DEFAULT
    )
    parser = pretrain_train.build_argparser()
    assert parser.get_default("sky_mask_refine_boundary_loss_weight") == pytest.approx(
        pretrain_train.SKY_MASK_REFINE_BOUNDARY_LOSS_WEIGHT_DEFAULT
    )
    assert pretrain_train.sky_mask_refine_boundary_loss_weight(
        _args(sky_mask_refine_boundary_loss_weight=0.75)
    ) == pytest.approx(0.75)


# ------------------------------------------------------------------ #
# Gauge readout precision                                             #
# ------------------------------------------------------------------ #
def _toy_gauge_decoder(hidden: int = 16, gauge_dim: int = 3) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, gauge_dim),
    )


def test_the_gauge_is_decoded_in_float32_under_bf16_autocast() -> None:
    decoder = _toy_gauge_decoder()
    hidden = torch.randn(2, 1, 16)
    t_base = torch.randn(2, 1, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        gauge = decode_scene_gauge(
            decoder, hidden, t_base, batch=2, gauge_dim=3
        )
        # The guard is local: the surrounding autocast is still in force.
        assert decoder(hidden).dtype is torch.bfloat16
    assert gauge.dtype is torch.float32
    assert gauge.shape == (2, 1, 3)


def test_float32_decoding_resolves_a_change_bf16_would_swallow() -> None:
    """The mechanism behind the frozen series, on our own decode helper."""

    decoder = _toy_gauge_decoder()
    hidden = torch.randn(1, 1, 16)
    t_base = torch.randn(1, 1, 16)

    def readout() -> torch.Tensor:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            return decode_scene_gauge(decoder, hidden, t_base, batch=1, gauge_dim=3)

    before = readout()
    # A shift well inside one bf16 step of the readout: bf16 keeps eight
    # mantissa bits, so anything below 2**-8 of the value is below the grid.
    shift = float(before.abs().max()) * 2.0 ** -11
    assert shift < _bf16_step(float(before.abs().max()))
    with torch.no_grad():
        decoder[-1].bias.add_(shift)
    after = readout()

    assert float((after - before).abs().max()) == pytest.approx(shift, rel=1e-3)


def _bf16_step(value: float) -> float:
    """Spacing of the bf16 grid at ``value`` (8 significant mantissa bits)."""

    import math

    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def test_a_sub_grid_gauge_change_is_invisible_once_the_readout_is_bf16() -> None:
    """Why the reported error could not move: it never left one bf16 bin.

    The v6 validation scene sits ~1.3 sigma from the training gauge mean, in
    the binade where the bf16 grid has spacing 2**-7.  At the measured
    ``gauge_std`` of 0.317 that is 0.0025 in log-metres -- a sixth of the 0.0162
    the run reported -- so two model states a third of a step apart produce the
    same number.
    """

    normalized = torch.tensor(1.2998, dtype=torch.bfloat16).float()
    step = _bf16_step(float(normalized))
    gauge_std, gauge_mean = 0.317467, 3.527166
    assert step == pytest.approx(2.0**-7)
    assert step * gauge_std == pytest.approx(0.00248, abs=1.0e-5)
    # The whole reported error is only a handful of grid steps wide.
    assert 0.0161888599 / (step * gauge_std) == pytest.approx(6.5, abs=0.2)

    target = torch.tensor([[[3.9398, -0.78, -1.15]]])
    valid = torch.ones_like(target, dtype=torch.bool)

    def scale_error(sub_step: float) -> float:
        predicted = target.clone()
        predicted[..., 0] = gauge_mean + gauge_std * float(
            (normalized + sub_step).to(torch.bfloat16)
        )
        logs = pretrain_train.gauge_diagnostic_metrics(
            predicted, target, valid, prior_log_scale=gauge_mean
        )
        return float(logs["gauge_log_scale_error"])

    # A quarter and a third of a step both round back to the same bf16 value.
    assert scale_error(step / 4.0) == scale_error(step / 3.0)
    # float32 keeps them apart, which is what the fix restores.
    assert (
        float((normalized + step / 4.0)) != float((normalized + step / 3.0))
    )


def test_a_frozen_scale_channel_still_reports_a_moving_fov() -> None:
    """Reproduce the v6 signature: one channel pinned, the others alive.

    ``gauge_diagnostic_metrics`` reads the scale and the two FOV channels off
    one tensor, so the run's "scale bit-identical, FOV changed 4x" shape is only
    possible if the scale channel itself could not move.
    """

    target = torch.tensor([[[3.9398, -0.78, -1.15]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    scale_errors, fov_errors = [], []
    for fov_shift in (0.0, 0.02, 0.05):
        predicted = target.clone()
        predicted[..., 0] += 0.0161888599  # pinned, as the run reported
        predicted[..., 1:] += fov_shift
        logs = pretrain_train.gauge_diagnostic_metrics(
            predicted, target, valid, prior_log_scale=3.5272
        )
        scale_errors.append(float(logs["gauge_log_scale_error"]))
        fov_errors.append(float(logs["gauge_fov_error_deg"]))

    assert len(set(scale_errors)) == 1
    assert scale_errors[0] == pytest.approx(0.0161888599, abs=1e-6)
    assert len(set(fov_errors)) == 3


# ------------------------------------------------------------------ #
# One PCA basis per mosaic                                            #
# ------------------------------------------------------------------ #
def test_every_latent_row_shares_the_ground_truth_basis() -> None:
    torch.manual_seed(0)
    reference = torch.randn(1, 2, 12, 8)
    basis = pretrain_train._latent_pca_basis(reference, 2)
    grid = pretrain_train._latent_pca_grid(reference, (3, 4), 2, basis=basis)
    assert grid.shape == (2, 3, 3, 4)
    assert torch.equal(
        grid, pretrain_train._latent_pca_grid(reference, (3, 4), 2, basis=basis)
    )


def test_the_shared_basis_keeps_a_washed_out_latent_visibly_washed_out() -> None:
    """Per-row normalization is what hid this.

    A generated latent with half the ground truth's spread is a real, visible
    difference.  Re-fitting the 1%/99% stretch per row rescales it back to full
    contrast and reports the two as identical images.
    """

    torch.manual_seed(0)
    reference = torch.randn(1, 2, 12, 8)
    washed_out = reference * 0.5
    basis = pretrain_train._latent_pca_basis(reference, 2)

    shared_gt = pretrain_train._latent_pca_grid(reference, (3, 4), 2, basis=basis)
    shared_pred = pretrain_train._latent_pca_grid(washed_out, (3, 4), 2, basis=basis)
    assert shared_pred.std() < 0.75 * shared_gt.std()

    # The old behaviour, still what the standalone inference scripts get.
    per_row_gt = pretrain_train._latent_pca_grid(reference, (3, 4), 2)
    per_row_pred = pretrain_train._latent_pca_grid(washed_out, (3, 4), 2)
    assert per_row_pred.std() == pytest.approx(per_row_gt.std(), rel=1e-3)


def test_the_basis_sign_does_not_depend_on_the_decomposition() -> None:
    """Each rank fits this independently; a flipped axis would recolour a row."""

    torch.manual_seed(0)
    reference = torch.randn(1, 2, 12, 8)
    first = pretrain_train._latent_pca_basis(reference, 2)
    second = pretrain_train._latent_pca_basis(reference.clone(), 2)
    assert torch.allclose(first.components, second.components, atol=1e-5)
    dominant = first.components.abs().argmax(dim=0)
    peaks = first.components[dominant, torch.arange(first.components.shape[1])]
    assert bool((peaks > 0).all())


def test_a_basis_from_the_wrong_latent_width_is_rejected() -> None:
    basis = pretrain_train._latent_pca_basis(torch.randn(1, 2, 12, 8), 2)
    with pytest.raises(ValueError, match="basis has 8 channels"):
        pretrain_train._latent_pca_grid(torch.randn(1, 2, 12, 5), (3, 4), 2, basis=basis)
