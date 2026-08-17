from __future__ import annotations

import inspect
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import train_scene_flow_pretrain as trainer
from dggt.losses.rgb_render_loss import (
    rgb_render_loss_enabled,
    should_apply_rgb_render_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _args(profile: str = "full", *extra: str):
    return trainer.build_argparser().parse_args(
        [
            "--image_dir",
            "/tmp/training",
            "--hdmap_root",
            "/tmp/training_hdmap",
            "--dggt_ckpt_path",
            "/tmp/dggt.pt",
            "--scene_gauge_path",
            "/tmp/training_gauge.json",
            "--pullback_calibration_path",
            "/tmp/pullback.json",
            "--log_dir",
            "/tmp/run",
            "--world_feedback_profile",
            profile,
            *extra,
        ]
    )


def _resume_payload(args):
    trainer.resolve_world_feedback_profile(args)
    return {
        "pretrain_resume_contract_version": trainer.PRETRAIN_RESUME_CONTRACT_VERSION,
        "pretrain_resume_reproducibility": trainer.PRETRAIN_RESUME_REPRODUCIBILITY,
        "layout_task_probabilities": list(trainer.LAYOUT_TASK_PROBABILITIES),
        "world_feedback_contract": deepcopy(args.world_feedback_contract),
        "rgb_render": trainer.rgb_render_run_summary(args),
        "pretrain_resume_critical_args": trainer.pretrain_resume_critical_args(args),
        "args": deepcopy(vars(args)),
    }


def test_parser_defaults_to_full_and_accepts_latent_only() -> None:
    assert _args().world_feedback_profile == "full"
    assert _args("latent_only").world_feedback_profile == "latent_only"


def test_full_profile_preserves_original_values_and_gradients() -> None:
    args = _args()
    contract = trainer.resolve_world_feedback_profile(args)
    assert contract["raw_baseline"] == {
        "lambda_rgb_render": pytest.approx(1.0),
        "lambda_level_consistency": pytest.approx(1.0),
        "lambda_head_consistency": pytest.approx(1.0),
    }
    assert contract["effective"] == contract["raw_baseline"]

    rgb = torch.tensor(2.0, requires_grad=True)
    level = torch.tensor(3.0, requires_grad=True)
    head = torch.tensor(4.0, requires_grad=True)
    weighted = trainer.weighted_world_feedback_terms(
        args,
        ramp=0.25,
        rgb_loss=rgb,
        level_loss=level,
        head_loss=head,
    )
    expected = (
        args.lambda_rgb_render * 0.25 * rgb,
        args.lambda_level_consistency * 0.25 * level,
        args.lambda_head_consistency * 0.25 * head,
    )
    for actual, reference in zip(weighted, expected):
        torch.testing.assert_close(actual, reference)
    sum(weighted).backward()
    assert rgb.grad.item() == pytest.approx(1.0 * 0.25)
    assert level.grad.item() == pytest.approx(0.25)
    assert head.grad.item() == pytest.approx(0.25)


def test_latent_only_keeps_raw_weights_and_zeroes_effective_graph_terms() -> None:
    args = _args("latent_only")
    contract = trainer.resolve_world_feedback_profile(args)
    assert (args.lambda_rgb_render, args.lambda_level_consistency, args.lambda_head_consistency) == pytest.approx(
        (1.0, 1.0, 1.0)
    )
    assert tuple(contract["effective"].values()) == (0.0, 0.0, 0.0)
    assert contract["compute_matched"] is True
    assert contract["schema"] == trainer.WORLD_FEEDBACK_CONTRACT_SCHEMA

    losses = [torch.tensor(value, requires_grad=True) for value in (2.0, 3.0, 4.0)]
    weighted = trainer.weighted_world_feedback_terms(
        args,
        ramp=0.75,
        rgb_loss=losses[0],
        level_loss=losses[1],
        head_loss=losses[2],
    )
    assert all(value.grad_fn is not None for value in weighted)
    assert tuple(value.item() for value in weighted) == (0.0, 0.0, 0.0)
    sum(weighted).backward()
    assert tuple(value.grad.item() for value in losses) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "extra",
    (
        ("--lambda_rgb_render", "0"),
        ("--lambda_rgb_render", "nan"),
        ("--lambda_level_consistency", "0"),
        ("--lambda_head_consistency", "0"),
        ("--rgb_render_every", "0"),
    ),
)
def test_latent_only_rejects_compute_mismatch_configuration(extra) -> None:
    args = _args("latent_only", *extra)
    with pytest.raises(ValueError, match="latent_only requires"):
        trainer.resolve_world_feedback_profile(args)


def test_latent_only_keeps_render_schedule_and_non_hds_settings_active() -> None:
    full = _args("full")
    latent_only = _args("latent_only")
    preserved = (
        "lambda_sky_view_reconstruction",
        "lambda_sky_mask",
        "lambda_sky_mask_refine",
        "lambda_gauge_flow",
        "lambda_gauge_direct",
        "lambda_sky_flow",
        "layout_map_injection",
        "layout_actor_injection",
        "layout_map_metric_injection",
        "layout_actor_metric_injection",
        "appearance_context_injection",
    )
    before = {name: getattr(latent_only, name) for name in preserved}
    trainer.resolve_world_feedback_profile(latent_only)
    assert {name: getattr(latent_only, name) for name in preserved} == before
    assert before == {name: getattr(full, name) for name in preserved}
    assert rgb_render_loss_enabled(latent_only)
    assert should_apply_rgb_render_loss(
        latent_only, latent_only.rgb_render_start_step, training=True
    )

    summary = trainer.rgb_render_run_summary(latent_only)
    assert summary["world_feedback_profile"] == "latent_only"
    assert summary["lambda_rgb_render_effective"] == 0.0
    assert summary["lambda_level_consistency_effective"] == 0.0
    assert summary["lambda_head_consistency_effective"] == 0.0
    assert vars(latent_only)["world_feedback_contract"] == latent_only.world_feedback_contract


def test_train_step_uses_only_the_unified_world_feedback_weighting_function() -> None:
    source = inspect.getsource(trainer.train_step)
    assert "weighted_world_feedback_terms(" in source
    assert "float(args.lambda_rgb_render)" not in source
    assert 'getattr(args, "lambda_level_consistency"' not in source
    assert 'getattr(args, "lambda_head_consistency"' not in source


@pytest.mark.parametrize("profile", trainer.WORLD_FEEDBACK_PROFILES)
def test_same_profile_resume_contract_passes(profile: str) -> None:
    args = _args(profile)
    trainer.validate_pretrain_resume_contract(
        _resume_payload(args), args, "checkpoint.pt"
    )


@pytest.mark.parametrize(
    ("saved_profile", "runtime_profile"),
    (("full", "latent_only"), ("latent_only", "full")),
)
def test_cross_profile_resume_contract_is_rejected(
    saved_profile: str,
    runtime_profile: str,
) -> None:
    payload = _resume_payload(_args(saved_profile))
    with pytest.raises(ValueError, match="world-feedback contract"):
        trainer.validate_pretrain_resume_contract(
            payload, _args(runtime_profile), "checkpoint.pt"
        )


def test_old_resume_contract_is_rejected() -> None:
    args = _args("latent_only")
    payload = _resume_payload(args)
    payload["pretrain_resume_contract_version"] = "layout_v2_pretrain_resume_v2"
    payload.pop("world_feedback_contract")
    with pytest.raises(ValueError, match="pretrain resume contract"):
        trainer.validate_pretrain_resume_contract(payload, args, "old.pt")


def test_every_checkpoint_variant_carries_world_feedback_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Config:
        prediction_type = "x"
        t_eps = 0.05
        scene_units_profile = trainer.SCENE_UNITS_PROFILE_GENERATED

        def to_dict(self):
            return {
                "prediction_type": self.prediction_type,
                "t_eps": self.t_eps,
                "scene_units_profile": self.scene_units_profile,
            }

    class _SceneFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = _Config()
            self._scene_units_contract = {
                "schema": trainer.SCENE_UNITS_CONTRACT_SCHEMA,
                "profile": trainer.SCENE_UNITS_PROFILE_GENERATED,
            }

    saved: dict[str, dict] = {}

    def capture(payload, path):
        saved[Path(path).name] = payload

    scene_flow = _SceneFlow()
    monkeypatch.setattr(
        trainer,
        "materialize_ema_state_dict",
        lambda _model, _ema: scene_flow.state_dict(),
    )
    monkeypatch.setattr(torch, "save", capture)
    args = _args("latent_only")
    trainer.resolve_world_feedback_profile(args)
    trainer.save_checkpoint(
        scene_flow,
        SimpleNamespace(state_dict=lambda: {"decay": 0.9995}),
        SimpleNamespace(state_dict=lambda: {"optimizer": True}),
        SimpleNamespace(state_dict=lambda: {"scheduler": True}),
        5000,
        tmp_path,
        args,
    )

    assert set(saved) == {
        "pretrain_step005000.pt",
        "pretrain_step005000_weights_only.pt",
        "pretrain_step005000_ema_weights_only.pt",
    }
    for payload in saved.values():
        assert payload["world_feedback_contract"] == args.world_feedback_contract
        assert payload["world_feedback_contract"]["profile"] == "latent_only"


def test_los_launcher_is_fixed_tagged_and_reuses_common_launcher() -> None:
    wrapper_path = REPO_ROOT / "pretrain_ppu_two_nodes_los_dlc.sh"
    common_path = REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh"
    wrapper = wrapper_path.read_text(encoding="utf-8")
    common = common_path.read_text(encoding="utf-8")

    assert "SCENE_UNITS_PROFILE=generated" in wrapper
    assert "WORLD_FEEDBACK_PROFILE=latent_only" in wrapper
    assert "scene_flow_pretrain_los_v6" in wrapper
    assert "ppu_dlc_two_nodes_los_v6_launch" in wrapper
    assert "scene_flow_pretrain_waymo_gb64_los_v6" in wrapper
    assert 'exec bash "${BASE_DLC_LAUNCHER}"' in wrapper
    assert 'export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_' in wrapper
    assert "--lambda_" not in wrapper
    assert '--world_feedback_profile "${WORLD_FEEDBACK_PROFILE}"' in common
    assert '--resume_path "${RESUME_PATH}"' in common
    assert '--resume_expected_step "${RESUME_EXPECTED_STEP}"' in common
    assert '--wandb_run_id "${WANDB_RUN_ID}"' in common

    help_result = subprocess.run(
        ["bash", str(wrapper_path), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Latent-Only Supervision" in help_result.stdout
    assert "step 0" in help_result.stdout
    assert "Full checkpoint" in help_result.stdout


def test_one_node_ppu_los_launcher_only_changes_topology_and_accumulation(
    tmp_path: Path,
) -> None:
    two_node_path = REPO_ROOT / "pretrain_ppu_two_nodes_los_dlc.sh"
    one_node_path = REPO_ROOT / "pretrain_ppu_one_node_los_dlc.sh"
    two_node = two_node_path.read_text(encoding="utf-8")
    one_node = one_node_path.read_text(encoding="utf-8")

    def assignment(text: str, name: str) -> str:
        prefix = f"export {name}="
        return next(line for line in text.splitlines() if line.startswith(prefix))

    assert one_node_path.stat().st_mode & 0o111
    assert assignment(one_node, "WANDB_API_KEY") == assignment(
        two_node, "WANDB_API_KEY"
    )
    assert assignment(one_node, "EXPECTED_NNODES") == "export EXPECTED_NNODES=1"
    assert assignment(one_node, "GRAD_ACCUM_STEPS") == (
        "export GRAD_ACCUM_STEPS=4"
    )
    assert next(
        line for line in one_node.splitlines() if line.startswith("PROJECT_ROOT=")
    ) == next(
        line for line in two_node.splitlines() if line.startswith("PROJECT_ROOT=")
    )
    for name in (
        "SCENE_UNITS_PROFILE",
        "WORLD_FEEDBACK_PROFILE",
        "LOG_DIR",
        "WANDB_NAME",
        "WANDB_RESUME",
    ):
        assert assignment(one_node, name) == assignment(two_node, name)
    assert "ppu_dlc_one_node_los_v6_launch" in one_node
    assert 'exec bash "${BASE_DLC_LAUNCHER}"' in one_node
    assert "--lambda_" not in one_node

    fake_common = tmp_path / "pretrain_ppu_two_nodes_dlc.sh"
    fake_common.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'TOPOLOGY=%s ACCUM=%s PROFILE=%s/%s\\n' "
        '"$EXPECTED_NNODES" "$GRAD_ACCUM_STEPS" '
        '"$SCENE_UNITS_PROFILE" "$WORLD_FEEDBACK_PROFILE"\n'
    )
    env = os.environ.copy()
    for name in (
        "EXPECTED_NNODES",
        "GRAD_ACCUM_STEPS",
        "SCENE_UNITS_PROFILE",
        "WORLD_FEEDBACK_PROFILE",
        "LOG_DIR",
        "LAUNCH_LOG_DIR",
        "WANDB_NAME",
        "WANDB_RESUME",
    ):
        env.pop(name, None)
    env["PROJECT_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        ["bash", str(one_node_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout.strip() == "TOPOLOGY=1 ACCUM=4 PROFILE=generated/latent_only"

    help_result = subprocess.run(
        ["bash", str(one_node_path), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "1 个 PPU 节点" in help_result.stdout
    assert "gradient accumulation 4" in help_result.stdout
    assert "global batch 64" in help_result.stdout


def test_single_node_los_launcher_matches_a2_two_node_contract() -> None:
    single = REPO_ROOT / "pretrain_single_node_los_ablation.sh"
    common = REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh"
    single_text = single.read_text(encoding="utf-8")
    common_text = common.read_text(encoding="utf-8")

    assert single.stat().st_mode & 0o111
    assert "NPROC_PER_NODE=8" in single_text
    assert "BATCH_SIZE_PER_GPU=1" in single_text
    assert "GRAD_ACCUM_STEPS=8" in single_text
    assert "EXPECTED_GLOBAL_BATCH_SIZE=64" in single_text
    assert "NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS" in single_text
    assert 'SCENE_UNITS_PROFILE="generated"' in single_text
    assert 'WORLD_FEEDBACK_PROFILE="latent_only"' in single_text
    assert "scene_flow_pretrain_los_v6" in single_text
    assert "single_node_los_v6_launch" in single_text
    assert "scene_flow_pretrain_waymo_gb64_los_v6" in single_text
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-dggt-flow}"' in single_text
    assert 'WANDB_RESUME="never"' in single_text
    assert 'WANDB_RESUME="must"' in single_text
    assert "world_feedback_contract" in single_text
    assert 'export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_' in single_text
    for raw_flag in (
        "--lambda_rgb_render",
        "--lambda_level_consistency",
        "--lambda_head_consistency",
    ):
        assert raw_flag not in single_text

    shared_defaults = (
        'MAX_STEPS="${MAX_STEPS:-200000}"',
        'DECAY_END_STEPS="${DECAY_END_STEPS:-0}"',
        'SAVE_EVERY="${SAVE_EVERY:-2500}"',
        'VAL_EVERY="${VAL_EVERY:-2000}"',
        'VAL_BATCHES="${VAL_BATCHES:-8}"',
        'VAL_LOG_IMAGES="${VAL_LOG_IMAGES:-10}"',
        'VAL_INFERENCE_SCENES="${VAL_INFERENCE_SCENES:-10}"',
        'VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-50}"',
    )
    for fragment in shared_defaults:
        assert fragment in single_text
        assert fragment in common_text

    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"' in single_text
    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-three_quarter}"' in common_text

    shared_train_args = (
        "--lr 1e-4",
        "--final_lr 1e-5",
        "--weight_decay 0.0",
        "--optimizer_type gmuon",
        "--ema_decay 0.9995",
        "--warmup_steps 4000",
        "--shift 10.0",
        "--weighting_scheme waver",
        "--mode_scale 1.29",
        "--prediction_type x",
        "--lambda_repa 0.5",
        "--base_model_coeff 0.25",
        "--lambda_boundary 0.25",
        "--lambda_sky_flow 0.5",
        '--world_feedback_profile "${WORLD_FEEDBACK_PROFILE}"',
    )
    for fragment in shared_train_args:
        assert fragment in single_text
        assert fragment in common_text

    completed = subprocess.run(
        ["bash", str(single), "--help"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={},
    )
    assert "gradient accumulation 8" in completed.stdout
    assert "global batch 64" in completed.stdout
    assert "step 0" in completed.stdout


def test_los_launcher_rejects_untagged_overrides_before_dlc_setup() -> None:
    wrapper_path = REPO_ROOT / "pretrain_ppu_two_nodes_los_dlc.sh"
    result = subprocess.run(
        ["bash", str(wrapper_path)],
        env={
            "PATH": "/usr/bin:/bin",
            "PROJECT_ROOT": str(REPO_ROOT),
            "LOG_DIR": "/tmp/scene_flow_pretrain_v6",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "los 标签" in result.stderr
