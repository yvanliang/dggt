from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from diffusers.training_utils import EMAModel
import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

import inference_scene_flow_pretrain as inference
import train_scene_flow_pretrain as pretrain
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.actor_geometry_condition import LayoutMode
from tests.test_layout_condition_model import (
    _layout_batch,
    _layout_kwargs,
    _model_inputs,
)
from tests.test_image_space_layout_only import _fake_common_env
from tests.test_rgb_render_loss_alignment import _parse_pretrain_args


REPO_ROOT = Path(__file__).resolve().parents[1]
MLO_INJECTIONS = {
    "layout_map_injection": False,
    "layout_actor_injection": False,
    "layout_map_metric_injection": True,
    "layout_actor_metric_injection": True,
    "appearance_context_injection": True,
}
FULL_INJECTIONS = {name: True for name in pretrain.LAYOUT_INJECTION_FLAGS}


def _resolved_args(profile: str, *extra: str) -> argparse.Namespace:
    args = _parse_pretrain_args("--layout_path_profile", profile, *extra)
    pretrain.resolve_layout_path_profile(args)
    args.patch_grid = (int(args.patch_grid_h), int(args.patch_grid_w))
    return args


def _tiny_profile_model(profile: str) -> WanSceneFlow:
    injections = (
        MLO_INJECTIONS
        if profile == pretrain.LAYOUT_PATH_PROFILE_METRIC_LAYOUT_ONLY
        else FULL_INJECTIONS
    )
    model = WanSceneFlow(
        patch_grid=(25, 37),
        num_attention_heads=1,
        attention_head_dim=24,
        out_channels=4,
        qwen_dim=8,
        freq_dim=8,
        ffn_dim=48,
        num_layers=1,
        base_model_depth=1,
        repa_layer_depth=1,
        ddt_head_depth=1,
        ddt_head_dim=24,
        ddt_head_heads=1,
        ddt_head_ffn_dim=48,
        num_timestep_tokens=1,
        sky_mask_refine_channels=8,
        layout_condition_version="layout_v2",
        layout_max_actors=96,
        **injections,
    )
    model.set_gauge_stats(torch.zeros(3), torch.ones(3))
    model._scene_units_contract = {
        "schema": pretrain.SCENE_UNITS_CONTRACT_SCHEMA,
        "profile": "generated",
    }
    return model


def _training_state(profile: str):
    model = _tiny_profile_model(profile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    ema = EMAModel(model.parameters(), decay=0.9995)
    args = _resolved_args(profile)
    return model, ema, optimizer, scheduler, args


def _save_profile_checkpoint(tmp_path: Path, profile: str, *, step: int = 7):
    model, ema, optimizer, scheduler, args = _training_state(profile)
    pretrain.save_checkpoint(
        model,
        ema,
        optimizer,
        scheduler,
        step,
        tmp_path,
        args,
    )
    return model, ema, optimizer, scheduler, args


def test_full_and_metric_layout_only_resolve_exactly_and_idempotently() -> None:
    full = _parse_pretrain_args("--layout_path_profile", "full")
    full_contract = pretrain.resolve_layout_path_profile(full)
    assert full_contract == {
        "schema": "layout_path_profile_v1",
        "requested_profile": "full",
        "resolved_profile": "full",
        "resolved_injections": FULL_INJECTIONS,
    }
    assert pretrain.resolve_layout_path_profile(full) == full_contract

    mlo = _parse_pretrain_args(
        "--layout_path_profile",
        "metric_layout_only",
        # An explicit profile must replace the entire low-level combination.
        "--layout_map_injection",
        "--layout_actor_injection",
        "--no-layout_map_metric_injection",
        "--no-layout_actor_metric_injection",
        "--no-appearance_context_injection",
    )
    mlo_contract = pretrain.resolve_layout_path_profile(mlo)
    assert mlo_contract == {
        "schema": "layout_path_profile_v1",
        "requested_profile": "metric_layout_only",
        "resolved_profile": "metric_layout_only",
        "resolved_injections": MLO_INJECTIONS,
    }
    assert pretrain.resolve_layout_path_profile(mlo) == mlo_contract


def test_auto_preserves_legacy_low_level_flags_and_records_custom() -> None:
    args = _parse_pretrain_args(
        "--layout_path_profile",
        "auto",
        "--no-layout_map_injection",
        "--layout_actor_injection",
        "--no-layout_map_metric_injection",
        "--layout_actor_metric_injection",
        "--no-appearance_context_injection",
    )
    before = {
        name: getattr(args, name) for name in pretrain.LAYOUT_INJECTION_FLAGS
    }
    contract = pretrain.resolve_layout_path_profile(args)
    assert contract["resolved_profile"] == "custom"
    assert contract["resolved_injections"] == before
    assert {
        name: getattr(args, name) for name in pretrain.LAYOUT_INJECTION_FLAGS
    } == before


def test_mlo_skips_early_stems_but_calls_all_late_adapters() -> None:
    model = _tiny_profile_model("metric_layout_only").eval()
    calls = {
        "early_map": 0,
        "early_actor": 0,
        "late_map": 0,
        "late_actor": 0,
        "late_appearance": 0,
    }

    def count(name: str):
        def hook(_module, _inputs, _output):
            calls[name] += 1

        return hook

    handles = [
        model.layout_map_stem.register_forward_hook(count("early_map")),
        model.layout_actor_stem.register_forward_hook(count("early_actor")),
        model.map_metric_adapter.register_forward_hook(count("late_map")),
        model.full_actor_gauge_adapter.register_forward_hook(count("late_actor")),
        model.appearance_context_adapter.register_forward_hook(
            count("late_appearance")
        ),
    ]
    condition, appearance_class = _layout_batch([LayoutMode.FULL])
    model_args, common = _model_inputs(1)
    with torch.no_grad():
        output = model(
            *model_args,
            **common,
            **_layout_kwargs(condition, appearance_class),
        )
    for handle in handles:
        handle.remove()

    assert set(output) == {"video", "sky", "gauge"}
    assert calls == {
        "early_map": 0,
        "early_actor": 0,
        "late_map": 1,
        "late_actor": 1,
        "late_appearance": 1,
    }


def test_full_and_mlo_have_identical_state_schema_shapes_and_parameter_count() -> None:
    full = _tiny_profile_model("full")
    mlo = _tiny_profile_model("metric_layout_only")
    full_state = full.state_dict()
    mlo_state = mlo.state_dict()
    assert tuple(full_state) == tuple(mlo_state)
    assert {
        name: tuple(value.shape) for name, value in full_state.items()
    } == {name: tuple(value.shape) for name, value in mlo_state.items()}
    assert sum(parameter.numel() for parameter in full.parameters()) == sum(
        parameter.numel() for parameter in mlo.parameters()
    )


def test_all_three_checkpoint_variants_record_mlo_contract(tmp_path: Path) -> None:
    _save_profile_checkpoint(tmp_path, "metric_layout_only", step=7)
    ckpt_dir = tmp_path / "ckpt"
    expected = {
        "schema": "layout_path_profile_v1",
        "requested_profile": "metric_layout_only",
        "resolved_profile": "metric_layout_only",
        "resolved_injections": MLO_INJECTIONS,
    }
    for suffix in ("", "_weights_only", "_ema_weights_only"):
        payload = torch.load(
            ckpt_dir / f"pretrain_step000007{suffix}.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert payload["layout_path_profile_contract"] == expected


def test_mlo_resume_succeeds_and_cross_profile_resume_is_rejected(
    tmp_path: Path,
) -> None:
    _save_profile_checkpoint(tmp_path, "metric_layout_only", step=11)
    checkpoint = tmp_path / "ckpt" / "pretrain_step000011.pt"

    model, ema, optimizer, scheduler, args = _training_state(
        "metric_layout_only"
    )
    assert pretrain.load_resume_checkpoint(
        model,
        ema,
        optimizer,
        scheduler,
        str(checkpoint),
        torch.device("cpu"),
        expected_step=11,
        args=args,
    ) == 11

    full_model, full_ema, full_optimizer, full_scheduler, full_args = (
        _training_state("full")
    )
    with pytest.raises(ValueError, match="config|layout-path"):
        pretrain.load_resume_checkpoint(
            full_model,
            full_ema,
            full_optimizer,
            full_scheduler,
            str(checkpoint),
            torch.device("cpu"),
            expected_step=11,
            args=full_args,
        )

    full_contract = pretrain.resolve_layout_path_profile(full_args)
    mlo_contract = pretrain.resolve_layout_path_profile(args)
    with pytest.raises(ValueError, match="layout-path"):
        pretrain.validate_layout_path_profile_resume_contract(
            {"layout_path_profile_contract": full_contract},
            args,
            "full.pt",
        )
    with pytest.raises(ValueError, match="layout-path"):
        pretrain.validate_layout_path_profile_resume_contract(
            {"layout_path_profile_contract": mlo_contract},
            full_args,
            "mlo.pt",
        )


def test_legacy_full_checkpoint_is_inferred_from_saved_flags() -> None:
    full_args = _resolved_args("full")
    payload = {
        "args": {
            name: True for name in pretrain.LAYOUT_INJECTION_FLAGS
        },
        "scene_flow_config": {
            name: True for name in pretrain.LAYOUT_INJECTION_FLAGS
        },
    }
    pretrain.validate_layout_path_profile_resume_contract(
        payload, full_args, "legacy_full.pt"
    )
    with pytest.raises(ValueError, match="active paths"):
        pretrain.validate_layout_path_profile_resume_contract(
            payload,
            _resolved_args("metric_layout_only"),
            "legacy_full.pt",
        )


def test_inference_loader_restores_mlo_routing_from_scene_flow_config(
    tmp_path: Path,
) -> None:
    _save_profile_checkpoint(tmp_path, "metric_layout_only", step=5)
    path = tmp_path / "ckpt" / "pretrain_step000005_weights_only.pt"
    args = inference.build_argparser().parse_args(["--weights", str(path)])
    loaded, _info = inference._require_current_checkpoint(
        path,
        device=torch.device("cpu"),
        use_ema=False,
        args=args,
    )
    assert {
        name: getattr(loaded.config, name)
        for name in pretrain.LAYOUT_INJECTION_FLAGS
    } == MLO_INJECTIONS


def test_mlo_wrapper_help_labels_and_common_launcher_reuse(tmp_path: Path) -> None:
    wrapper_path = REPO_ROOT / "pretrain_ppu_two_nodes_mlo_dlc.sh"
    common_path = REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh"
    wrapper = wrapper_path.read_text()
    common = common_path.read_text()

    help_result = subprocess.run(
        ["bash", str(wrapper_path), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "step 0" in help_result.stdout
    assert "同一 MLO setting" in help_result.stdout
    assert "全部 late M/G/A context" in help_result.stdout
    assert "RESUME_EXPECTED_STEP" in help_result.stdout

    assert "export LAYOUT_PATH_PROFILE=metric_layout_only" in wrapper
    assert "export SCENE_UNITS_PROFILE=generated" in wrapper
    assert "export WORLD_FEEDBACK_PROFILE=full" in wrapper
    assert "exec bash \"${PROJECT_ROOT}/pretrain_ppu_two_nodes_dlc.sh\"" in wrapper
    assert 'export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_' in wrapper
    assert "--layout_map_injection" not in wrapper
    assert "--lambda_rgb_render" not in wrapper
    assert "VAL_SAMPLE_STEPS:-50" in common
    assert '--layout_path_profile "${LAYOUT_PATH_PROFILE}"' in common
    assert "RESUME_EXPECTED_STEP" in common
    assert "WANDB_RUN_ID" in common

    env = dict(os.environ)
    env.update(
        {
            "PROJECT_ROOT": str(REPO_ROOT),
            "LOG_DIR": str(tmp_path / "scene_flow_pretrain_full_v6"),
        }
    )
    rejected = subprocess.run(
        ["bash", str(wrapper_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "独立的 mlo 标签" in rejected.stderr


def test_a4_single_node_launcher_matches_mlo_two_node_contract(
    tmp_path: Path,
) -> None:
    launcher = REPO_ROOT / "pretrain_single_node_mlo_ablation.sh"
    common = REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh"
    single_text = launcher.read_text(encoding="utf-8")
    common_text = common.read_text(encoding="utf-8")

    assert launcher.stat().st_mode & 0o111
    assert "NPROC_PER_NODE=8" in single_text
    assert "BATCH_SIZE_PER_GPU=1" in single_text
    assert "GRAD_ACCUM_STEPS=8" in single_text
    assert "EXPECTED_GLOBAL_BATCH_SIZE=64" in single_text
    assert (
        "NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS"
        in single_text
    )
    assert 'SCENE_UNITS_PROFILE="generated"' in single_text
    assert 'WORLD_FEEDBACK_PROFILE="full"' in single_text
    assert 'LAYOUT_PATH_PROFILE="metric_layout_only"' in single_text
    assert "scene_flow_pretrain_mlo_v6" in single_text
    assert "single_node_mlo_v6_launch" in single_text
    assert "scene_flow_pretrain_waymo_gb64_mlo_v6" in single_text
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-dggt-flow}"' in single_text
    assert 'WANDB_RESUME="never"' in single_text
    assert 'WANDB_RESUME="must"' in single_text
    assert 'export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_' in single_text

    for fragment in (
        'MAX_STEPS="${MAX_STEPS:-200000}"',
        'DECAY_END_STEPS="${DECAY_END_STEPS:-0}"',
        'SAVE_EVERY="${SAVE_EVERY:-2500}"',
        'VAL_EVERY="${VAL_EVERY:-2000}"',
        'VAL_BATCHES="${VAL_BATCHES:-8}"',
        'VAL_LOG_IMAGES="${VAL_LOG_IMAGES:-10}"',
        'VAL_INFERENCE_SCENES="${VAL_INFERENCE_SCENES:-10}"',
        'VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-50}"',
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
        '--layout_path_profile "${LAYOUT_PATH_PROFILE}"',
    ):
        assert fragment in single_text
        assert fragment in common_text

    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"' in single_text
    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-three_quarter}"' in common_text

    help_result = subprocess.run(
        ["bash", str(launcher), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "global batch 64" in help_result.stdout
    assert "gradient accumulation 8" in help_result.stdout
    assert "late M/G/A context 固定 on/on/on" in help_result.stdout

    env = _fake_common_env(tmp_path)
    for name in (
        "LAYOUT_PATH_PROFILE",
        "SCENE_UNITS_PROFILE",
        "WORLD_FEEDBACK_PROFILE",
    ):
        env.pop(name, None)
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { return 0; }\n")
    env.update(
        {
            "CONDA_SH": str(conda_sh),
            "LOG_DIR": str(tmp_path / "scene_flow_pretrain_mlo_v6"),
            "LAUNCH_LOG_DIR": str(tmp_path / "single_node_mlo_v6_launch"),
            "WANDB_NAME": "scene_flow_pretrain_waymo_gb64_mlo_v6",
        }
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert (
        "global batch size: 64 = 1 node × 8 gpu/node × 1 batch/gpu × 8 accum"
        in lines
    )
    assert lines[lines.index("--grad_accum_steps") + 1] == "8"
    assert lines[lines.index("--scene_units_profile") + 1] == "generated"
    assert lines[lines.index("--world_feedback_profile") + 1] == "full"
    assert (
        lines[lines.index("--layout_path_profile") + 1]
        == "metric_layout_only"
    )
    assert lines[lines.index("--wandb_resume") + 1] == "never"
    assert "--resume_path" not in lines
    assert "--wandb_run_id" not in lines
