"""Versioned checkpoint contract for SceneFlow rectified-flow schedules.

The model weights do not encode the distribution of training times or the
time warp used by the ODE sampler.  Keeping those values only in argparse
defaults makes a checkpoint ambiguous, so every new SceneFlow export carries
this small, solver-independent contract.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


FLOW_SCHEDULE_VERSION = "rae_rectified_flow_schedule_v2"
LEGACY_FLOW_SCHEDULE_VERSION = "rae_rectified_flow_schedule_v1"
FLOW_PATH_VERSION = "linear_clean_to_noise_v1"
FLOW_INFERENCE_GRID_VERSION = "uniform_noise_to_clean_shifted_v1"
FLOW_AUXILIARY_TIME_VERSION = "shared_video_sigma_v1"
FLOW_SOLVER_VERSION = "explicit_euler_v1"
FLOW_PREDICTION_TO_CLEAN_VERSION = "rae_clamped_target_inverse_v1"


def _prediction_type_from_payload(payload: dict[str, Any]) -> str:
    config = payload.get("scene_flow_config")
    if isinstance(config, dict) and "prediction_type" in config:
        return str(config["prediction_type"])
    args = payload.get("args")
    if isinstance(args, dict) and "prediction_type" in args:
        return str(args["prediction_type"])
    return "x"


def _t_eps_from_payload(payload: dict[str, Any]) -> float:
    config = payload.get("scene_flow_config")
    if isinstance(config, dict) and "t_eps" in config:
        return float(config["t_eps"])
    return 0.05


def build_flow_schedule_config(
    args: Any,
    *,
    prediction_type: str = "x",
    t_eps: float = 0.05,
) -> dict[str, Any]:
    """Return the canonical schedule represented by a training CLI namespace."""
    weighting_scheme = str(getattr(args, "weighting_scheme", "waver")).lower().replace("-", "_")
    if weighting_scheme not in ("waver", "logit_normal"):
        raise ValueError(f"Unsupported flow weighting_scheme={weighting_scheme!r}")
    loss_weighting = str(getattr(args, "loss_weighting_scheme", "none")).lower()
    if loss_weighting in ("", "none"):
        loss_weighting = "none"
    if loss_weighting != "none":
        raise ValueError(
            "RAE SceneFlow checkpoints only support loss_weighting_scheme='none', "
            f"got {loss_weighting!r}"
        )
    shift = float(getattr(args, "shift", 10.0))
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"Flow schedule shift must be positive, got {shift}")
    prediction_type = str(prediction_type)
    if prediction_type not in ("x", "v"):
        raise ValueError(f"Unsupported prediction_type={prediction_type!r}")

    # Inactive sampler parameters are canonicalized to None.  Changing an
    # unused CLI default must not make otherwise identical checkpoints appear
    # schedule-incompatible.
    logit_mean = float(getattr(args, "logit_mean", 0.0))
    logit_std = float(getattr(args, "logit_std", 1.0))
    mode_scale = float(getattr(args, "mode_scale", 1.29))
    t_eps = float(t_eps)
    active_values = (mode_scale,) if weighting_scheme == "waver" else (logit_mean, logit_std)
    if not all(math.isfinite(value) for value in active_values):
        raise ValueError("Active flow timestep-sampling parameters must be finite")
    if weighting_scheme == "logit_normal" and logit_std <= 0.0:
        raise ValueError(f"logit_std must be positive, got {logit_std}")
    if not math.isfinite(t_eps) or not 0.0 < t_eps <= 1.0:
        raise ValueError(f"Flow t_eps must be in (0, 1], got {t_eps}")

    return {
        "version": FLOW_SCHEDULE_VERSION,
        "flow_path": FLOW_PATH_VERSION,
        "inference_time_grid": FLOW_INFERENCE_GRID_VERSION,
        "ode_solver": FLOW_SOLVER_VERSION,
        "auxiliary_time_coupling": FLOW_AUXILIARY_TIME_VERSION,
        "prediction_to_clean": FLOW_PREDICTION_TO_CLEAN_VERSION,
        "shift": shift,
        "training_timestep_sampling": weighting_scheme,
        "logit_mean": logit_mean if weighting_scheme == "logit_normal" else None,
        "logit_std": logit_std if weighting_scheme == "logit_normal" else None,
        "mode_scale": mode_scale if weighting_scheme == "waver" else None,
        "loss_weighting_scheme": loss_weighting,
        "prediction_type": prediction_type,
        "t_eps": t_eps,
    }


def _legacy_flow_schedule_config(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recover the contract from an old *full* checkpoint when unambiguous."""
    saved_args = payload.get("args")
    required = {
        "shift",
        "weighting_scheme",
        "logit_mean",
        "logit_std",
        "mode_scale",
        "loss_weighting_scheme",
    }
    if not isinstance(saved_args, dict) or not required.issubset(saved_args):
        return None
    prediction_type = _prediction_type_from_payload(payload)
    if prediction_type == "v":
        raise ValueError(
            "Legacy velocity-prediction checkpoints do not record the clean-reconstruction semantic. "
            "They may have used z_t-sigma*v with an RAE clamped target and cannot be resumed or "
            "used for inference as mathematically equivalent checkpoints."
        )
    return build_flow_schedule_config(
        SimpleNamespace(**saved_args),
        prediction_type=prediction_type,
        t_eps=_t_eps_from_payload(payload),
    )


def checkpoint_flow_schedule_config(
    payload: Any,
    path: str | Path,
    *,
    allow_legacy_full_checkpoint: bool = True,
) -> dict[str, Any]:
    """Read and schema-check the checkpoint's authoritative flow schedule."""
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a versioned SceneFlow checkpoint")
    saved = payload.get("flow_schedule_config")
    if saved is None and allow_legacy_full_checkpoint:
        saved = _legacy_flow_schedule_config(payload)
    if not isinstance(saved, dict):
        raise ValueError(
            f"{path} has no flow_schedule_config. The schedule cannot be inferred from weights alone; "
            "use the corresponding legacy full checkpoint (with args) once and re-export it, or use a "
            "new checkpoint that records the schedule contract."
        )
    saved = dict(saved)
    saved_version = saved.get("version")
    if saved_version == LEGACY_FLOW_SCHEDULE_VERSION:
        if str(saved.get("prediction_type", _prediction_type_from_payload(payload))) == "v":
            raise ValueError(
                f"{path} uses legacy flow schedule {LEGACY_FLOW_SCHEDULE_VERSION!r} with "
                "prediction_type='v'. Its clean-reconstruction semantic is incompatible with the "
                "RAE clamped target inverse; restart or explicitly warm-start a new training run."
            )
        # The corrected inverse is not exercised by x-prediction checkpoints,
        # so their v1 schedule can be upgraded without changing model behavior.
        saved["version"] = FLOW_SCHEDULE_VERSION
        saved["prediction_to_clean"] = FLOW_PREDICTION_TO_CLEAN_VERSION
    elif saved_version != FLOW_SCHEDULE_VERSION:
        raise ValueError(
            f"{path} flow schedule version={saved_version!r}, expected {FLOW_SCHEDULE_VERSION!r}"
        )
    expected_constants = {
        "flow_path": FLOW_PATH_VERSION,
        "inference_time_grid": FLOW_INFERENCE_GRID_VERSION,
        "ode_solver": FLOW_SOLVER_VERSION,
        "auxiliary_time_coupling": FLOW_AUXILIARY_TIME_VERSION,
        "prediction_to_clean": FLOW_PREDICTION_TO_CLEAN_VERSION,
    }
    mismatches = [
        f"{key}={saved.get(key)!r} (expected {expected!r})"
        for key, expected in expected_constants.items()
        if saved.get(key) != expected
    ]
    required_fields = {
        "shift",
        "training_timestep_sampling",
        "logit_mean",
        "logit_std",
        "mode_scale",
        "loss_weighting_scheme",
        "prediction_type",
        "t_eps",
        "prediction_to_clean",
    }
    missing = sorted(required_fields.difference(saved))
    if missing or mismatches:
        details = []
        if missing:
            details.append(f"missing fields={missing}")
        if mismatches:
            details.append("; ".join(mismatches))
        raise ValueError(f"{path} has an invalid flow_schedule_config: {'; '.join(details)}")
    # Re-canonicalize to validate ranges/enums and to avoid accepting arbitrary
    # inactive values in hand-edited checkpoints.
    proxy = SimpleNamespace(
        shift=saved["shift"],
        weighting_scheme=saved["training_timestep_sampling"],
        logit_mean=0.0 if saved["logit_mean"] is None else saved["logit_mean"],
        logit_std=1.0 if saved["logit_std"] is None else saved["logit_std"],
        mode_scale=1.29 if saved["mode_scale"] is None else saved["mode_scale"],
        loss_weighting_scheme=saved["loss_weighting_scheme"],
    )
    canonical = build_flow_schedule_config(
        proxy,
        prediction_type=str(saved["prediction_type"]),
        t_eps=float(saved["t_eps"]),
    )
    if canonical != saved:
        raise ValueError(f"{path} flow_schedule_config is not canonical: saved={saved!r}, canonical={canonical!r}")
    model_prediction_type = _prediction_type_from_payload(payload)
    if canonical["prediction_type"] != model_prediction_type:
        raise ValueError(
            f"{path} flow schedule prediction_type={canonical['prediction_type']!r} does not match "
            f"scene_flow_config prediction_type={model_prediction_type!r}"
        )
    model_t_eps = _t_eps_from_payload(payload)
    if abs(float(canonical["t_eps"]) - model_t_eps) > 1e-12:
        raise ValueError(
            f"{path} flow schedule t_eps={canonical['t_eps']} does not match "
            f"scene_flow_config t_eps={model_t_eps}"
        )
    return canonical


def validate_checkpoint_flow_schedule(
    payload: Any,
    args: Any,
    path: str | Path,
    *,
    prediction_type: str = "x",
    t_eps: float = 0.05,
) -> dict[str, Any]:
    """Require exact train/resume/warm-start schedule equivalence."""
    saved = checkpoint_flow_schedule_config(payload, path)
    runtime = build_flow_schedule_config(args, prediction_type=prediction_type, t_eps=t_eps)
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, runtime={runtime.get(key)!r}"
        for key in runtime
        if saved.get(key) != runtime.get(key)
    ]
    if mismatches:
        raise ValueError(
            f"{path} flow schedule does not match runtime: {'; '.join(mismatches)}. "
            "Training resume/warm-start must preserve the checkpoint's flow schedule."
        )
    return saved


def resolve_inference_flow_schedule(
    payload: Any,
    args: Any,
    path: str | Path,
) -> dict[str, Any]:
    """Make a checkpoint authoritative for inference, rejecting an override."""
    saved = checkpoint_flow_schedule_config(payload, path)
    requested_shift = getattr(args, "shift", None)
    if requested_shift is not None and abs(float(requested_shift) - float(saved["shift"])) > 1e-12:
        raise ValueError(
            f"{path} flow schedule shift={saved['shift']}, but runtime requested --shift={requested_shift}. "
            "Inference must use the schedule stored in the checkpoint."
        )
    setattr(args, "shift", float(saved["shift"]))
    return saved
