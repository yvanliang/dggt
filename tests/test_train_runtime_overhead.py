from __future__ import annotations

import io
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

import train_scene_flow_pretrain as trainer
from datasets.dataset import load_metric_depth_diagnostic_paths
from dggt.losses.flow_losses import compute_total_loss


def _flow_loss_inputs() -> dict[str, object]:
    v_pred = torch.tensor([[[[1.0, -2.0]]]], requires_grad=True)
    v_gt = torch.tensor([[[[0.25, -0.5]]]])
    clean = torch.tensor([[[[0.5, 0.75]]]])
    return {
        "v_pred": v_pred,
        "v_gt": v_gt,
        "eps": torch.zeros_like(v_pred),
        "bundle": SimpleNamespace(
            z_clean_n=clean,
            M_preserve=torch.ones((1, 1, 1, 1)),
        ),
        "lambda_preserve": 0.0,
    }


def test_deferred_loss_logs_preserve_values_and_training_graph() -> None:
    eager_loss, eager_logs = compute_total_loss(**_flow_loss_inputs())
    deferred_inputs = _flow_loss_inputs()
    deferred_loss, deferred_logs = compute_total_loss(
        **deferred_inputs,
        defer_log_values=True,
    )

    assert torch.equal(eager_loss, deferred_loss)
    assert all(isinstance(value, float) for value in eager_logs.values())
    assert all(
        torch.is_tensor(value) and value.ndim == 0 and not value.requires_grad
        for value in deferred_logs.values()
    )
    assert trainer.materialize_log_values(deferred_logs) == eager_logs

    deferred_loss.backward()
    assert deferred_inputs["v_pred"].grad is not None
    assert torch.isfinite(deferred_inputs["v_pred"].grad).all()


def test_non_logging_rank_skips_metrics_without_changing_loss() -> None:
    eager_loss, _ = compute_total_loss(**_flow_loss_inputs())
    quiet_loss, quiet_logs = compute_total_loss(
        **_flow_loss_inputs(),
        defer_log_values=True,
        collect_logs=False,
    )
    assert torch.equal(quiet_loss, eager_loss)
    assert quiet_logs == {}

    logits = torch.tensor([[[[[0.2, -0.4], [1.5, -2.0]]]]])
    target = torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]]]]])
    logged_mask_loss, _ = trainer.sky_mask_patch_loss(
        logits,
        target,
        dice_weight=0.5,
        pos_weight_max=10.0,
    )
    quiet_mask_loss, quiet_mask_logs = trainer.sky_mask_patch_loss(
        logits,
        target,
        dice_weight=0.5,
        pos_weight_max=10.0,
        collect_logs=False,
    )
    assert torch.equal(quiet_mask_loss, logged_mask_loss)
    assert quiet_mask_logs == {}


def test_wandb_window_averages_deferred_and_float_metrics() -> None:
    sums: dict[str, trainer.TrainLogSeries] = {}
    counts: dict[str, int] = {}
    trainer.accumulate_wandb_metrics(
        sums,
        counts,
        {"loss": torch.tensor(2.0), "lr": 1.0e-4},
    )
    trainer.accumulate_wandb_metrics(
        sums,
        counts,
        {"loss": torch.tensor(4.0), "lr": 3.0e-4},
    )

    result = trainer.finalize_wandb_metrics(sums, counts)
    assert result["loss"] == 3.0
    assert result["lr"] == pytest.approx(2.0e-4)


class _TTYProbe:
    def __init__(self, interactive: bool):
        self.interactive = bool(interactive)

    def isatty(self) -> bool:
        return self.interactive


def test_tqdm_can_be_forced_for_a_noninteractive_web_console() -> None:
    assert trainer.use_interactive_tqdm(False, stream=_TTYProbe(True))
    assert not trainer.use_interactive_tqdm(False, stream=_TTYProbe(False))
    assert trainer.use_interactive_tqdm(
        False,
        force_tqdm=True,
        stream=_TTYProbe(False),
    )
    assert not trainer.use_interactive_tqdm(True, stream=_TTYProbe(True))
    assert not trainer.use_interactive_tqdm(
        True,
        force_tqdm=True,
        stream=_TTYProbe(True),
    )
    assert not trainer.use_interactive_tqdm(False, stream=object())


def test_web_console_tqdm_stream_publishes_every_refresh_as_a_line() -> None:
    sink = io.StringIO()
    stream = trainer.WebConsoleTqdmStream(sink)
    bar = trainer.tqdm(
        total=2,
        desc="pretrain",
        mininterval=0.0,
        miniters=1,
        file=stream,
    )
    bar.set_postfix({"wait": "0.00", "loss": "2.200"}, refresh=False)
    bar.update(1)
    first_update = sink.getvalue()
    assert first_update.endswith("\n")
    assert "1/2" in first_update
    bar.update(1)
    bar.close()

    output = sink.getvalue()
    assert "\r" not in output
    assert "\x1b[" not in output
    assert "2/2" in output
    assert all(line.startswith("pretrain:") for line in output.splitlines())


def test_ppu_web_launchers_force_tqdm_for_eta_and_rate() -> None:
    root = Path(trainer.__file__).resolve().parent
    for name in ("pretrain_ppu.sh", "pretrain_ppu_two_nodes_dlc.sh"):
        source = (root / name).read_text(encoding="utf-8")
        assert 'LOG_EVERY="${LOG_EVERY:-1}"' in source
        assert "--force_tqdm" in source
        assert "\n        --no_tqdm\n" not in source
        assert '--log_every "${LOG_EVERY}"' in source
    four_node = (root / "pretrain_ppu_four_nodes_dlc.sh").read_text(
        encoding="utf-8"
    )
    assert 'exec bash "${BASE_DLC_LAUNCHER}"' in four_node
    assert trainer.build_argparser().get_default("log_every") == 1
    trainer_source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "mininterval=0.0" in trainer_source
    assert "miniters=1" in trainer_source


@pytest.mark.parametrize(
    ("world_size", "grad_accum_steps", "expected_epoch_steps"),
    (
        (64, 1, (0, 13, 26)),
        (32, 2, (0, 12, 25)),
    ),
)
def test_continuous_sampler_matches_historical_distributed_epoch_chunks(
    world_size: int,
    grad_accum_steps: int,
    expected_epoch_steps: tuple[int, ...],
) -> None:
    dataset = list(range(800))
    rank = 7
    sampler = trainer.ContinuousDistributedBatchSampler(
        dataset,
        batch_size=1,
        grad_accum_steps=grad_accum_steps,
        num_replicas=world_size,
        rank=rank,
        seed=0,
    )
    batches_per_epoch = sampler.batches_per_logical_epoch
    actual = list(
        islice(iter(sampler), batches_per_epoch * len(expected_epoch_steps))
    )

    expected: list[list[int]] = []
    reference = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=False,
    )
    for epoch_step in expected_epoch_steps:
        reference.set_epoch(epoch_step)
        expected.extend([[index] for index in reference])
    assert actual == expected


def test_continuous_sampler_keeps_dataloader_alive_across_epoch_boundary() -> None:
    dataset = list(range(8))
    sampler = trainer.ContinuousDistributedBatchSampler(
        dataset,
        batch_size=1,
        grad_accum_steps=1,
        num_replicas=2,
        rank=0,
        seed=0,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    batches = list(islice(iter(loader), 9))
    assert sampler.batches_per_logical_epoch == 4
    assert len(batches) == 9


def test_continuous_sampler_preserves_drop_last_at_each_logical_epoch() -> None:
    dataset = list(range(800))
    sampler = trainer.ContinuousDistributedBatchSampler(
        dataset,
        batch_size=2,
        grad_accum_steps=2,
        num_replicas=64,
        rank=3,
        seed=0,
    )
    # DistributedSampler gives this rank 13 rows. Historical DataLoader
    # drop_last=True used 12 as six batches and discarded one every pass.
    assert sampler.distributed_sampler.num_samples == 13
    assert sampler.usable_samples_per_epoch == 12
    assert sampler.batches_per_logical_epoch == 6

    actual = list(islice(iter(sampler), 12))
    expected: list[list[int]] = []
    reference = DistributedSampler(
        dataset,
        num_replicas=64,
        rank=3,
        shuffle=True,
        seed=0,
        drop_last=False,
    )
    # Six micro-batches / accumulation 2 advances optimizer step by three.
    for epoch_step in (0, 3):
        reference.set_epoch(epoch_step)
        rows = list(reference)[:12]
        expected.extend([rows[start : start + 2] for start in range(0, 12, 2)])
    assert actual == expected


def test_validation_defaults_and_ppu_launchers_use_requested_cadence() -> None:
    parser = trainer.build_argparser()
    assert parser.get_default("val_every") == 2000
    # Ten scenes: five pinned carry the numbers, five rotate for the pictures.
    assert parser.get_default("val_inference_scenes") == 10

    root = Path(trainer.__file__).resolve().parent
    for name in ("pretrain_ppu.sh", "pretrain_ppu_two_nodes_dlc.sh"):
        source = (root / name).read_text(encoding="utf-8")
        assert 'VAL_EVERY="${VAL_EVERY:-2000}"' in source
        assert 'VAL_INFERENCE_SCENES="${VAL_INFERENCE_SCENES:-10}"' in source
        assert '--val_inference_scenes "${VAL_INFERENCE_SCENES}"' in source

    for name in (
        "pretrain_half_node_p6000.sh",
        "pretrain_single_node.sh",
        "pretrain_two_nodes26.sh",
        "pretrain_two_nodes31.sh",
    ):
        source = (root / name).read_text(encoding="utf-8")
        assert "val_every 1000" not in source
        assert "VAL_EVERY:-1000" not in source


def test_five_scene_three_cfg_validation_maps_one_job_to_fifteen_ranks() -> None:
    assignments = [
        trainer.validation_sampling_tasks_for_rank(
            3, 5, rank=rank, world_size=64
        )
        for rank in range(64)
    ]
    assert assignments[:15] == [
        ((scene, scale),)
        for scene in range(5)
        for scale in range(3)
    ]
    assert all(tasks == () for tasks in assignments[15:])
    assert trainer.effective_validation_scene_count(3, 5, world_size=64) == 5


def test_validation_falls_back_to_two_scenes_when_ranks_are_short() -> None:
    """One pinned and one rotating, so neither half disappears.

    Falling back to a single scene would drop the rotating half entirely, and
    falling back on the pinned half would leave the sample_* series with no
    fixed reference at all.
    """

    assert trainer.effective_validation_scene_count(3, 10, world_size=29) == 2
    assert trainer.effective_validation_scene_count(3, 10, world_size=30) == 10
    # Six jobs fit inside one eight-accelerator node.
    assignments = [
        trainer.validation_sampling_tasks_for_rank(3, 10, rank=rank, world_size=8)
        for rank in range(8)
    ]
    assert assignments[:6] == [
        ((scene, scale),) for scene in range(2) for scale in range(3)
    ]
    assert assignments[6:] == [(), ()]
    assert trainer.validation_pinned_scene_count(2) == 1


def _mosaic_row(slot: int, group: str, order: int) -> dict[str, object]:
    return {
        "slot": slot,
        "scene": f"{slot:03d}",
        "group": group,
        "order": order,
        "caption": "cfg 2",
        "frames": 1,
        "png": "",
    }


def test_validation_rank_merge_averages_cfg_metrics_and_keeps_scene_images() -> None:
    metrics, rows = trainer.merge_validation_rank_results(
        [
            {
                "metrics": {"sample_gauge_cfg2/error": 1.0},
                "mosaic_rows": [_mosaic_row(0, "rgb", 1)],
            },
            {
                "metrics": {"sample_gauge_cfg2/error": 3.0},
                "mosaic_rows": [_mosaic_row(1, "rgb", 1)],
            },
        ]
    )
    assert metrics == {"sample_gauge_cfg2/error": 2.0}
    # Both scenes survive: they are different slots of the same quantity.
    assert [(row["slot"], row["group"], row["order"]) for row in rows] == [
        (0, "rgb", 1),
        (1, "rgb", 1),
    ]

    with pytest.raises(RuntimeError, match="duplicate validation artifact"):
        trainer.merge_validation_rank_results(
            [
                {"metrics": {}, "mosaic_rows": [_mosaic_row(0, "rgb", 1)]},
                {"metrics": {}, "mosaic_rows": [_mosaic_row(0, "rgb", 1)]},
            ]
        )


def test_validation_emits_one_mosaic_per_scene_slot(tmp_path: Path) -> None:
    """Every scene collapses to a single artifact, whatever the CFG count."""

    frames, gy, gx = 2, 3, 4
    args = SimpleNamespace(
        val_log_images=frames,
        patch_grid=(gy, gx),
        val_mosaic_cell_width=32,
    )
    z_clean = torch.rand((1, frames, gy * gx, 5))
    bundle = SimpleNamespace(
        z_clean_n=z_clean,
        sky_mask_refined_clean=torch.rand((1, frames, 3, gy * 14, gx * 14)),
    )
    rgb = torch.rand((frames, 3, gy * 14, gx * 14))
    rows: list[dict[str, object]] = []
    for slot in range(2):
        for scale_index, scale in enumerate((1.0, 2.0, 4.0)):
            rows.extend(
                trainer.collect_validation_mosaic_rows(
                    bundle,
                    torch.rand_like(z_clean),
                    {
                        "generated_raw_3dgs_rgb": rgb,
                        "generated_pred_sky_mask": rgb[:, :1],
                    },
                    args,
                    scene_slot=slot,
                    scene_label=f"{slot:03d}",
                    guidance_scale=scale,
                    scale_index=scale_index,
                    is_primary=scale_index == 0,
                    visualization_batch={"images": rgb.unsqueeze(0)},
                )
            )

    # Two GT rows plus four generated rows per scale, for each of two scenes.
    assert len(rows) == 2 * (2 + 3 * 4)
    paths = trainer.write_validation_mosaics(rows, tmp_path, 2000, args)
    assert set(paths) == {"mosaic/slot00", "mosaic/slot01"}
    assert all(path.exists() for path in paths.values())
    # Ground truth is emitted once per scene, not once per CFG scale.
    gt_rows = [row for row in rows if row["order"] == trainer.GT_ROW_ORDER]
    assert len(gt_rows) == 4


def test_sky_overlay_puts_gt_in_red_and_the_prediction_in_green() -> None:
    """One overlay beats two stacked binary rows: every disagreement is a hue."""

    target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]).unsqueeze(0)  # [1,1,1,1,4]
    target = target.repeat(1, 1, 3, 1, 1)
    predicted = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])  # [1,1,1,4]
    overlay, has_target = trainer._sky_mask_overlay_grid(target, predicted, 1)
    assert has_target
    red, green, blue = overlay[0, 0, 0], overlay[0, 1, 0], overlay[0, 2, 0]
    assert red.tolist() == [1.0, 1.0, 0.0, 0.0]  # GT sky
    assert green.tolist() == [1.0, 0.0, 1.0, 0.0]  # predicted sky
    assert blue.tolist() == [0.0, 0.0, 0.0, 0.0]
    # agreement -> yellow, missed sky -> red, invented sky -> green, else black.


def test_sky_overlay_without_ground_truth_is_grey_not_all_green() -> None:
    """A missing GT mask must not read as 'the model invented all the sky'."""

    predicted = torch.tensor([[[[1.0, 0.0]]]])
    overlay, has_target = trainer._sky_mask_overlay_grid(None, predicted, 1)
    assert not has_target
    assert overlay.shape[1] == 3
    assert torch.equal(overlay[:, 0], overlay[:, 1])
    assert torch.equal(overlay[:, 1], overlay[:, 2])


def test_latent_error_rows_share_one_absolute_scale_across_cfg_scales() -> None:
    """Per-image percentile normalisation would make every scale look alike.

    The mosaic stacks one error row per CFG scale; if each row rescaled to its
    own 99th percentile, a scale that is twice as wrong would render identically
    to one that is twice as right.
    """

    small = torch.full((1, 1, 4, 1), 0.2)
    large = small * 2.0
    small_grid = trainer._absolute_mask_grid(small, (2, 2), 1)
    large_grid = trainer._absolute_mask_grid(large, (2, 2), 1)
    assert pytest.approx(float(large_grid.mean()), rel=1e-6) == 2.0 * float(
        small_grid.mean()
    )
    # The legacy normalisation collapsed exactly this difference.
    assert float(trainer._normalized_mask_grid(small, (2, 2), 1).mean()) == pytest.approx(
        float(trainer._normalized_mask_grid(large, (2, 2), 1).mean()), rel=1e-6
    )
    # Saturating errors clamp instead of rescaling the whole row.
    assert float(trainer._absolute_mask_grid(small * 100.0, (2, 2), 1).max()) == 1.0


def test_all_rank_mean_reduces_deferred_scalars_in_one_buffer(monkeypatch) -> None:
    reduce_calls = 0

    def fake_reduce(packed, op=None):
        nonlocal reduce_calls
        reduce_calls += 1
        width = len(trainer.ALL_RANK_TRAIN_LOG_KEYS)
        wait = trainer._ALL_RANK_TRAIN_LOG_INDEX["dataloader/wait_seconds"]
        loss = trainer._ALL_RANK_TRAIN_LOG_INDEX["loss"]
        # The peer observed wait=3 and loss=4.
        packed[wait] += 3.0
        packed[loss] += 4.0
        packed[width + wait] += 1.0
        packed[width + loss] += 1.0

    monkeypatch.setattr(trainer, "is_distributed", lambda: True)
    monkeypatch.setattr(trainer.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(trainer.dist, "all_reduce", fake_reduce)
    result = trainer.all_rank_log_mean(
        {
            "loss": torch.tensor(2.0),
            "dataloader/wait_seconds": 1.0,
        },
        device=torch.device("cpu"),
    )
    assert result == pytest.approx(
        {"loss": 3.0, "dataloader/wait_seconds": 2.0}
    )
    assert reduce_calls == 1

    # Each step uses exactly one fixed-size value collective.
    trainer.all_rank_log_mean(
        {"loss": torch.tensor(2.0), "dataloader/wait_seconds": 1.0},
        device=torch.device("cpu"),
    )
    assert reduce_calls == 2


def test_all_rank_mean_sparse_union_does_not_mix_optional_metrics(monkeypatch) -> None:
    def fake_reduce(packed, op=None):
        width = len(trainer.ALL_RANK_TRAIN_LOG_KEYS)
        sky = trainer._ALL_RANK_TRAIN_LOG_INDEX["loss_sky_flow"]
        loss = trainer._ALL_RANK_TRAIN_LOG_INDEX["loss"]
        # The peer observed loss and an optional sky loss, but no RGB loss.
        packed[loss] += 4.0
        packed[width + loss] += 1.0
        packed[sky] += 0.25
        packed[width + sky] += 1.0

    monkeypatch.setattr(trainer, "is_distributed", lambda: True)
    monkeypatch.setattr(trainer.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(trainer.dist, "all_reduce", fake_reduce)
    result = trainer.all_rank_log_mean(
        {"loss": torch.tensor(2.0), "loss_rgb_render": 0.5},
        device=torch.device("cpu"),
    )
    assert result == pytest.approx(
        {"loss": 3.0, "loss_rgb_render": 0.5, "loss_sky_flow": 0.25}
    )


def test_all_rank_mean_unknown_key_fails_after_collective(monkeypatch) -> None:
    reduce_calls = 0

    def fake_reduce(packed, op=None):
        nonlocal reduce_calls
        reduce_calls += 1

    monkeypatch.setattr(trainer, "is_distributed", lambda: True)
    monkeypatch.setattr(trainer.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(trainer.dist, "all_reduce", fake_reduce)
    with pytest.raises(RuntimeError, match="unregistered or non-scalar"):
        trainer.all_rank_log_mean(
            {"future_metric": torch.tensor(2.0)},
            device=torch.device("cpu"),
        )
    assert reduce_calls == 1


def test_metric_depth_path_loader_preserves_missing_frame_slots(tmp_path) -> None:
    first = np.zeros((2, 3, 3), dtype=np.float32)
    first[..., 0] = np.array([[1.0, np.nan, -2.0], [4.0, 5.0, 6.0]])
    third = np.zeros((2, 3, 3), dtype=np.float32)
    third[..., 0] = 7.0
    first_path = tmp_path / "000_0.npy"
    missing_path = tmp_path / "001_0.npy"
    third_path = tmp_path / "002_0.npy"
    np.save(first_path, first)
    np.save(third_path, third)

    depth, valid = load_metric_depth_diagnostic_paths(
        [first_path, missing_path, third_path],
        height=2,
        width=3,
    )

    assert depth.shape == (3, 2, 3)
    assert valid.shape == depth.shape
    assert torch.equal(depth[1], torch.zeros_like(depth[1]))
    assert not valid[1].any()
    assert torch.equal(valid[0], torch.tensor([[True, False, False], [True, True, True]]))
    assert torch.equal(depth[0], torch.tensor([[1.0, 0.0, 0.0], [4.0, 5.0, 6.0]]))
    assert torch.equal(depth[2], torch.full((2, 3), 7.0))


def test_metric_depth_hydration_reads_only_scheduled_rows(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_load(paths, *, height, width):
        row_paths = [str(path) for path in paths]
        calls.append(row_paths)
        row_id = float(row_paths[0].split("_")[0])
        depth = torch.full((len(row_paths), height, width), row_id)
        return depth, torch.ones_like(depth, dtype=torch.bool)

    monkeypatch.setattr(trainer, "load_metric_depth_diagnostic_paths", fake_load)
    batch = {
        "images": torch.zeros((3, 2, 3, 4, 5)),
        # torch default_collate transposes each sample's list[str] to [S][B].
        "metric_lidar_depth_paths": [
            ("0_f0", "1_f0", "2_f0"),
            ("0_f1", "1_f1", "2_f1"),
        ],
        "untouched": torch.tensor([9]),
    }

    hydrated = trainer.hydrate_metric_depth_diagnostic_batch(batch, max_samples=1)

    assert calls == [["0_f0", "0_f1"]]
    assert hydrated is not batch
    assert hydrated["untouched"] is batch["untouched"]
    assert hydrated["metric_lidar_depth_m"].shape == (3, 2, 4, 5)
    assert hydrated["metric_lidar_depth_valid"][0].all()
    assert not hydrated["metric_lidar_depth_valid"][1:].any()


def test_expensive_layout_diagnostics_follow_logging_cadence() -> None:
    source = open(trainer.__file__, encoding="utf-8").read()
    assert "collect_logs and collect_expensive_diagnostics" in source
    assert "defer_log_values=True" in source
    assert "collect_logs=collect_step_logs" in source
    assert "return_generated_depth=bool(metric_depth_diagnostic_due)" in source
    assert "load_metric_depth_diagnostic=False" in source
    assert "return_metric_depth_diagnostic_paths=" in source
