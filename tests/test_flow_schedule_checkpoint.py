from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dggt.losses.flow_losses import rae_t_grid
from dggt.utils.flow_schedule import (
    FLOW_PREDICTION_TO_CLEAN_VERSION,
    FLOW_SCHEDULE_VERSION,
    LEGACY_FLOW_SCHEDULE_VERSION,
    build_flow_schedule_config,
    checkpoint_flow_schedule_config,
    resolve_inference_flow_schedule,
    validate_checkpoint_flow_schedule,
)


def _args(**overrides):
    values = {
        "shift": 10.0,
        "weighting_scheme": "waver",
        "logit_mean": 0.0,
        "logit_std": 1.0,
        "mode_scale": 1.29,
        "loss_weighting_scheme": "none",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_flow_schedule_round_trip_and_inactive_fields_are_canonical() -> None:
    config = build_flow_schedule_config(_args(logit_mean=7.0), prediction_type="x", t_eps=0.05)
    assert config["version"] == FLOW_SCHEDULE_VERSION
    assert config["training_timestep_sampling"] == "waver"
    assert config["logit_mean"] is None
    assert config["logit_std"] is None
    assert config["mode_scale"] == pytest.approx(1.29)
    assert config["prediction_to_clean"] == FLOW_PREDICTION_TO_CLEAN_VERSION
    assert checkpoint_flow_schedule_config({"flow_schedule_config": config}, "new.pt") == config


def test_rae_sampling_grid_rejects_steps_that_enter_t_eps_floor() -> None:
    safe = rae_t_grid(
        num_steps=191,
        time_shift=10.0,
        t_eps=0.05,
        device=torch.device("cpu"),
    )
    assert float(safe[-2].item()) == pytest.approx(0.05)

    with pytest.raises(ValueError, match="last nonzero sigma=.*< t_eps=.*num_steps <= 191"):
        rae_t_grid(
            num_steps=192,
            time_shift=10.0,
            t_eps=0.05,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("time_shift", "t_eps", "message"),
    [(float("nan"), 0.05, "time_shift"), (10.0, float("nan"), "t_eps"), (10.0, 1.1, "t_eps")],
)
def test_rae_sampling_grid_rejects_invalid_schedule_values(time_shift, t_eps, message) -> None:
    with pytest.raises(ValueError, match=message):
        rae_t_grid(
            num_steps=50,
            time_shift=time_shift,
            t_eps=t_eps,
            device=torch.device("cpu"),
        )


def test_training_resume_rejects_any_effective_schedule_change() -> None:
    saved = build_flow_schedule_config(_args(), prediction_type="x", t_eps=0.05)
    payload = {"flow_schedule_config": saved}
    validate_checkpoint_flow_schedule(payload, _args(), "checkpoint.pt", prediction_type="x", t_eps=0.05)

    with pytest.raises(ValueError, match="shift: checkpoint=10.0, runtime=6.0"):
        validate_checkpoint_flow_schedule(
            payload,
            _args(shift=6.0),
            "checkpoint.pt",
            prediction_type="x",
            t_eps=0.05,
        )
    with pytest.raises(ValueError, match="mode_scale"):
        validate_checkpoint_flow_schedule(
            payload,
            _args(mode_scale=1.0),
            "checkpoint.pt",
            prediction_type="x",
            t_eps=0.05,
        )


def test_inference_uses_checkpoint_shift_and_rejects_override() -> None:
    saved = build_flow_schedule_config(_args(shift=8.0), prediction_type="x", t_eps=0.05)
    payload = {
        "flow_schedule_config": saved,
        "scene_flow_config": {"prediction_type": "x", "t_eps": 0.05},
    }
    runtime = SimpleNamespace(shift=None)
    resolved = resolve_inference_flow_schedule(payload, runtime, "checkpoint.pt")
    assert resolved == saved
    assert runtime.shift == pytest.approx(8.0)

    with pytest.raises(ValueError, match="runtime requested --shift=10.0"):
        resolve_inference_flow_schedule(payload, SimpleNamespace(shift=10.0), "checkpoint.pt")


def test_schedule_cannot_disagree_with_model_parameterization() -> None:
    saved = build_flow_schedule_config(_args(), prediction_type="v", t_eps=0.05)
    payload = {
        "flow_schedule_config": saved,
        "scene_flow_config": {"prediction_type": "x", "t_eps": 0.05},
    }
    with pytest.raises(ValueError, match="does not match scene_flow_config prediction_type"):
        checkpoint_flow_schedule_config(payload, "checkpoint.pt")


def test_legacy_full_checkpoint_is_recoverable_but_weights_only_is_not() -> None:
    legacy_full = {
        "args": vars(_args()),
        "scene_flow_config": {"prediction_type": "x", "t_eps": 0.05},
    }
    recovered = checkpoint_flow_schedule_config(legacy_full, "legacy_full.pt")
    assert recovered["shift"] == pytest.approx(10.0)
    assert recovered["training_timestep_sampling"] == "waver"

    with pytest.raises(ValueError, match="cannot be inferred from weights alone"):
        checkpoint_flow_schedule_config(
            {"scene_flow": {}},
            "legacy_weights_only.pt",
        )


def test_v1_x_schedule_upgrades_but_v1_velocity_schedule_is_rejected() -> None:
    legacy_x = build_flow_schedule_config(_args(), prediction_type="x", t_eps=0.05)
    legacy_x["version"] = LEGACY_FLOW_SCHEDULE_VERSION
    legacy_x.pop("prediction_to_clean")
    recovered = checkpoint_flow_schedule_config(
        {
            "flow_schedule_config": legacy_x,
            "scene_flow_config": {"prediction_type": "x", "t_eps": 0.05},
        },
        "legacy_x.pt",
    )
    assert recovered["version"] == FLOW_SCHEDULE_VERSION
    assert recovered["prediction_to_clean"] == FLOW_PREDICTION_TO_CLEAN_VERSION

    legacy_v = dict(legacy_x, prediction_type="v")
    with pytest.raises(ValueError, match="legacy flow schedule.*prediction_type='v'"):
        checkpoint_flow_schedule_config(
            {
                "flow_schedule_config": legacy_v,
                "scene_flow_config": {"prediction_type": "v", "t_eps": 0.05},
            },
            "legacy_v.pt",
        )


def test_legacy_full_velocity_checkpoint_is_rejected() -> None:
    legacy_full = {
        "args": {**vars(_args()), "prediction_type": "v"},
        "scene_flow_config": {"prediction_type": "v", "t_eps": 0.05},
    }
    with pytest.raises(ValueError, match="Legacy velocity-prediction checkpoints"):
        checkpoint_flow_schedule_config(legacy_full, "legacy_full_v.pt")
