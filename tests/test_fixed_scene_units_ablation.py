from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import inference_scene_flow_pretrain as offline_inference
import train_scene_flow_pretrain as pretrain
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.scene_gauge import (
    SCENE_UNITS_CONTRACT_SCHEMA,
    SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN,
    SCENE_UNITS_PROFILE_GENERATED,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _model(profile: str) -> WanSceneFlow:
    model = WanSceneFlow(
        patch_grid=(1, 1),
        num_attention_heads=1,
        attention_head_dim=12,
        in_channels=15,
        out_channels=4,
        qwen_dim=12,
        freq_dim=8,
        ffn_dim=24,
        num_layers=1,
        repa_layer_depth=1,
        ddt_head_depth=1,
        ddt_head_dim=12,
        ddt_head_heads=1,
        ddt_head_ffn_dim=24,
        base_model_depth=1,
        layout_condition_version="none",
        scene_units_profile=profile,
        encoder_mrope_section=(2, 2, 2),
        ddt_mrope_section=(2, 2, 2),
        # Keep the production v4 sky contract even though these scene-unit
        # tests do not feed sky tokens.  This makes the fixture exercise the
        # same checkpoint surface as the rebuilt s2.9.1 baseline.
        sky_grid=(16, 32),
        max_sky_tokens=512,
        sky_mask_refine_scale=1,
        sky_mask_refine_channels=4,
    )
    model.set_gauge_stats(
        torch.tensor([0.25, -0.5, -0.75]),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    return model


def _forward(model: WanSceneFlow) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    batch, frames, patches = 2, 2, 1
    z = torch.randn(batch, frames, patches, 4)
    mask = torch.zeros(batch, frames, patches, 1)
    result = model(
        z,
        torch.full((batch,), 0.5),
        torch.zeros_like(z),
        torch.zeros_like(z),
        mask,
        mask,
        torch.ones_like(mask),
        camera_condition_tokens=torch.zeros(batch, frames, 20),
        gauge_gen_tokens=None,
        gauge_gen_attention_mask=None,
        return_dict=True,
    )
    assert isinstance(result, dict)
    return result


def _required_cli(*extra: str) -> argparse.Namespace:
    return pretrain.build_argparser().parse_args(
        [
            "--image_dir",
            "/tmp/training",
            "--hdmap_root",
            "/tmp/training_hdmap",
            "--dggt_ckpt_path",
            "/tmp/dggt.pt",
            "--scene_gauge_path",
            "/tmp/training.json",
            "--pullback_calibration_path",
            "/tmp/pullback.json",
            "--log_dir",
            "/tmp/fsu",
            *extra,
        ]
    )


def test_cli_profile_is_single_enum_with_generated_default() -> None:
    assert _required_cli().scene_units_profile == SCENE_UNITS_PROFILE_GENERATED
    assert (
        _required_cli("--scene_units_profile", "fixed_train_mean").scene_units_profile
        == SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN
    )
    with pytest.raises(SystemExit):
        _required_cli("--scene_units_profile", "partly_fixed")


def test_profiles_have_identical_state_surface_and_parameter_count() -> None:
    generated = _model(SCENE_UNITS_PROFILE_GENERATED)
    fixed = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN)
    generated_state = generated.state_dict()
    fixed_state = fixed.state_dict()
    assert generated_state.keys() == fixed_state.keys()
    assert {
        name: tuple(value.shape) for name, value in generated_state.items()
    } == {name: tuple(value.shape) for name, value in fixed_state.items()}
    assert sum(parameter.numel() for parameter in generated.parameters()) == sum(
        parameter.numel() for parameter in fixed.parameters()
    )


def test_fixed_placeholder_is_zero_and_fully_masked() -> None:
    model = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN)
    z = torch.randn(3, 2, 1, 4)
    tokens, mask, position = model._build_gauge_generation(z, None, None)
    assert tokens.shape == (3, 1, model.config.hidden_size)
    assert torch.equal(tokens, torch.zeros_like(tokens))
    assert mask is not None and not bool(mask.any())
    assert position.shape == (3, 1, 3)


def test_fixed_rejects_any_caller_gauge_stream() -> None:
    model = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN)
    z = torch.randn(2, 2, 1, 4)
    with pytest.raises(ValueError, match="forbids caller-provided"):
        model._build_gauge_generation(z, torch.randn(2, 1, 3), None)
    with pytest.raises(ValueError, match="forbids caller-provided"):
        model._build_gauge_generation(
            z, None, torch.zeros(2, 1, dtype=torch.bool)
        )


def test_decoder_changes_cannot_affect_fixed_video_or_normalized_gauge() -> None:
    model = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN).eval()
    with torch.no_grad():
        model.final_layer.linear.weight.normal_()
        model.final_layer.linear.bias.normal_()
        before = _forward(model)
        for parameter in model.gauge_gen_decoder.parameters():
            parameter.fill_(37.0)
        after = _forward(model)
    torch.testing.assert_close(after["video"], before["video"], atol=0.0, rtol=0.0)
    assert torch.equal(before["gauge"], torch.zeros_like(before["gauge"]))
    assert torch.equal(after["gauge"], torch.zeros_like(after["gauge"]))
    expected = model.gauge_mean.view(1, 1, 3).expand(2, 1, 3)
    torch.testing.assert_close(
        model.fixed_scene_gauge(2, device="cpu", dtype=torch.float32),
        expected,
        atol=0.0,
        rtol=0.0,
    )


def _write_fixed_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    stats_path = tmp_path / "feature_stats.pt"
    torch.save(
        {
            "stats_status": "complete",
            "stats_coverage": {
                "expected_batches": 4,
                "processed_batches": 4,
                "full_dataset_pass": True,
            },
            "source": {
                "image_dir": "/relocated/dataset/training",
                "scene_gauge_path": "/old/root/training.json",
            },
            "gauge_mean": torch.tensor([0.25, -0.5, -0.75]),
            "gauge_std": torch.tensor([1.0, 2.0, 3.0]),
            "gauge_count": torch.tensor([10, 11, 12]),
        },
        stats_path,
    )
    gauge_path = tmp_path / "training.json"
    gauge_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "metadata": {"split": "training"},
            }
        )
    )
    return stats_path, gauge_path


def test_fixed_contract_validates_stats_and_ignores_absolute_source_paths(
    tmp_path: Path,
) -> None:
    model = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN)
    stats_path, gauge_path = _write_fixed_artifacts(tmp_path)
    contract = pretrain.build_scene_units_contract(
        model,
        feature_stats_path=stats_path,
        scene_gauge_path=gauge_path,
    )
    assert contract["schema"] == SCENE_UNITS_CONTRACT_SCHEMA
    assert contract["profile"] == SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN
    assert contract["raw_mean"] == pytest.approx([0.25, -0.5, -0.75])
    assert contract["normalized_zero"] == [0.0, 0.0, 0.0]
    assert contract["channel_counts"] == [10, 11, 12]
    relocated = dict(contract)
    relocated["source"] = {
        "feature_stats_path": "/another/machine/stats.pt",
        "scene_gauge_path": "/another/machine/gauge.json",
    }
    assert pretrain.scene_units_contract_identity(
        contract, path="a"
    ) == pretrain.scene_units_contract_identity(relocated, path="b")


def test_fixed_stats_contract_rejects_incomplete_coverage(tmp_path: Path) -> None:
    model = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN)
    stats_path, gauge_path = _write_fixed_artifacts(tmp_path)
    payload = torch.load(stats_path, weights_only=False)
    payload["stats_status"] = "smoke_only"
    torch.save(payload, stats_path)
    with pytest.raises(ValueError, match="stats_status='complete'"):
        pretrain.build_scene_units_contract(
            model,
            feature_stats_path=stats_path,
            scene_gauge_path=gauge_path,
        )


def test_old_profileless_config_is_generated_only() -> None:
    generated = _model(SCENE_UNITS_PROFILE_GENERATED)
    old_config = generated.config.to_dict()
    old_config.pop("scene_units_profile")
    payload = {
        "scene_flow_config": old_config,
        "scene_flow": generated.state_dict(),
    }
    pretrain.validate_scene_flow_checkpoint_config(generated, payload, "old.pt")
    fixed = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN)
    with pytest.raises(ValueError, match="config is not an exact match"):
        pretrain.validate_scene_flow_checkpoint_config(fixed, payload, "old.pt")


def test_fixed_checkpoint_contract_is_same_arm_only(tmp_path: Path) -> None:
    fixed = _model(SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN)
    stats_path, gauge_path = _write_fixed_artifacts(tmp_path)
    contract = pretrain.build_scene_units_contract(
        fixed,
        feature_stats_path=stats_path,
        scene_gauge_path=gauge_path,
    )
    fixed._scene_units_contract = contract
    payload = {
        "scene_flow_config": fixed.config.to_dict(),
        "scene_units_contract": contract,
        "scene_flow": fixed.state_dict(),
    }
    pretrain.validate_scene_flow_checkpoint_config(fixed, payload, "fixed.pt")

    generated = _model(SCENE_UNITS_PROFILE_GENERATED)
    with pytest.raises(ValueError, match="config is not an exact match"):
        pretrain.validate_scene_flow_checkpoint_config(
            generated, payload, "fixed.pt"
        )

    wrong_contract = dict(contract)
    wrong_contract["raw_mean"] = [0.0, 0.0, 0.0]
    wrong_payload = dict(payload, scene_units_contract=wrong_contract)
    with pytest.raises(ValueError, match="contract does not match runtime"):
        pretrain.validate_scene_flow_checkpoint_config(
            fixed, wrong_payload, "wrong-contract.pt"
        )

    wrong_state = dict(fixed.state_dict())
    wrong_state["gauge_mean"] = torch.zeros(3)
    wrong_payload = dict(payload, scene_flow=wrong_state)
    with pytest.raises(ValueError, match="gauge_mean buffer"):
        pretrain.validate_scene_flow_checkpoint_config(
            fixed, wrong_payload, "wrong-mean.pt"
        )


def test_resume_critical_identity_does_not_depend_on_mount_root() -> None:
    first = _required_cli(
        "--scene_units_profile",
        "fixed_train_mean",
        "--feature_stats_path",
        "/mnt/a/feature_stats.pt",
    )
    second = _required_cli(
        "--scene_units_profile",
        "fixed_train_mean",
        "--feature_stats_path",
        "/relocated/b/feature_stats.pt",
    )
    first.image_dir = "/mnt/a/training"
    second.image_dir = "/relocated/b/training"
    first.hdmap_root = "/mnt/a/training_hdmap"
    second.hdmap_root = "/relocated/b/training_hdmap"
    assert pretrain.pretrain_resume_critical_args(
        first
    ) == pretrain.pretrain_resume_critical_args(second)


def test_run_summary_and_offline_tensor_payload_use_fixed_source() -> None:
    args = _required_cli("--scene_units_profile", "fixed_train_mean")
    summary = pretrain.rgb_render_run_summary(args)
    assert summary["gauge_source"] == "fixed_train_mean"
    assert summary["lambda_gauge_flow_raw"] == pytest.approx(args.lambda_gauge_flow)
    assert summary["lambda_gauge_flow_effective"] == 0.0
    assert summary["rgb_render_gauge_pose_grad_scale_effective"] == 0.0

    contract = {
        "schema": SCENE_UNITS_CONTRACT_SCHEMA,
        "profile": SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN,
    }
    bundle = SimpleNamespace(
        scene_units_profile=SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN,
        scene_units_contract=contract,
        camera_to_world_requested_metric=torch.eye(4).view(1, 1, 4, 4),
        camera_intrinsics_requested_canvas_metric=torch.eye(3).view(1, 1, 3, 3),
        camera_intrinsics_requested_raw_metric=torch.eye(3).view(1, 1, 3, 3),
        camera_requested_raw_image_size_hw=torch.tensor([[100, 200]]),
        camera_requested_canvas_image_size_hw=(64, 96),
        frame_ids=torch.tensor([[0]]),
    )
    sampled = SimpleNamespace(
        video=torch.zeros(1, 1, 1, 4),
        sky=None,
        sky_mask_patch=torch.zeros(1, 1, 1, 1),
        sky_mask_refined=torch.zeros(1, 1, 1, 1, 1),
        gauge=torch.tensor([[[0.25, -0.5, -0.75]]]),
    )
    payload = offline_inference.offline_sample_tensor_payload(
        bundle, sampled, torch.zeros(1, 1, 9)
    )
    assert payload["scene_units_profile"] == SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN
    assert payload["render_gauge_source"] == "fixed_train_mean"
    assert payload["scene_units_contract"] == contract


def test_fixed_gauge_weight_logs_are_registered_for_all_rank_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _required_cli("--scene_units_profile", "fixed_train_mean")
    summary = pretrain.rgb_render_run_summary(args)
    keys = (
        "lambda_gauge_flow_raw",
        "lambda_gauge_direct_raw",
        "lambda_gauge_flow_effective",
        "lambda_gauge_direct_effective",
    )
    logs = {key: summary[key] for key in keys}

    monkeypatch.setattr(pretrain, "is_distributed", lambda: True)
    monkeypatch.setattr(pretrain.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(pretrain.dist, "all_reduce", lambda packed, op=None: None)

    reduced = pretrain.all_rank_log_mean(logs, device=torch.device("cpu"))
    assert reduced == pytest.approx(logs)
    assert set(keys).issubset(pretrain.ALL_RANK_TRAIN_LOG_KEYS)


class _FixedSamplerSceneFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            t_eps=0.05,
            prediction_type="v",
            gauge_gen_dim=3,
            scene_units_profile=SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN,
        )
        self.register_buffer("gauge_mean", torch.tensor([0.25, -0.5, -0.75]))
        self.calls: list[tuple[torch.Tensor | None, float]] = []

    def uses_generated_scene_units(self) -> bool:
        return False

    def fixed_scene_gauge(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.gauge_mean.to(device=device, dtype=dtype).view(1, 1, 3).expand(
            int(batch_size), 1, 3
        )

    def forward(
        self,
        z_t: torch.Tensor,
        sigma: torch.Tensor,
        *unused: torch.Tensor,
        gauge_gen_tokens: torch.Tensor | None,
        layout_to_gauge_grad_scale: float,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del sigma, unused, kwargs
        self.calls.append((gauge_gen_tokens, float(layout_to_gauge_grad_scale)))
        return {
            "video": torch.zeros_like(z_t),
            # Deliberately nonzero: the Fixed sampler must ignore this decoder
            # output instead of integrating it as an ODE state.
            "gauge": z_t.new_full((int(z_t.shape[0]), 1, 3), 999.0),
        }


def _fixed_sampler_bundle(num_frames: int) -> SimpleNamespace:
    z = torch.zeros((1, num_frames, 2, 4), dtype=torch.float32)
    mask = torch.zeros((1, num_frames, 2, 1), dtype=torch.float32)
    layout = object.__new__(pretrain.LayoutConditionBatch)
    object.__setattr__(
        layout,
        "appearance",
        SimpleNamespace(
            binding_valid=torch.zeros((1, 1), dtype=torch.bool),
            appearance_mode=torch.zeros((1,), dtype=torch.int8),
        ),
    )
    return SimpleNamespace(
        z_clean_n=z,
        z_splat_n=torch.zeros_like(z),
        M_preserve=mask,
        M_source=torch.zeros_like(mask),
        M_dest=torch.ones_like(mask),
        scene_gauge_clean_n=torch.zeros((1, 1, 3)),
        sky_gen_clean=None,
        layout_condition=layout,
        appearance_class_id=torch.full((1, 1), -1, dtype=torch.int8),
        camera_condition_tokens=torch.zeros((1, num_frames, 20)),
        camera_attention_mask=torch.ones((1, num_frames), dtype=torch.bool),
        frame_ids=torch.arange(num_frames).view(1, num_frames),
        fps=None,
        captions=None,
    )


@pytest.mark.parametrize("sliding", [False, True])
def test_fixed_samplers_have_no_gauge_ode_and_return_exact_mean(
    monkeypatch: pytest.MonkeyPatch,
    sliding: bool,
) -> None:
    flow = _FixedSamplerSceneFlow()
    num_frames = 4 if sliding else 2
    monkeypatch.setattr(
        pretrain.LayoutConditionBatch,
        "slice_frames",
        lambda self, frame_index: self,
    )
    monkeypatch.setattr(
        pretrain,
        "layout_model_kwargs",
        lambda layout, appearance_class_id, *, gauge_grad_scale: {
            "layout_to_gauge_grad_scale": float(gauge_grad_scale)
        },
    )
    velocity_state_shapes: list[tuple[int, ...]] = []
    original_velocity = pretrain.sampler_prediction_to_velocity

    def record_velocity(
        scene_flow: torch.nn.Module,
        prediction: torch.Tensor,
        state: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        velocity_state_shapes.append(tuple(state.shape))
        return original_velocity(
            scene_flow, prediction, state, sigma, **kwargs
        )

    monkeypatch.setattr(pretrain, "sampler_prediction_to_velocity", record_velocity)
    args = SimpleNamespace(
        val_sample_steps=2,
        shift=1.0,
        seed=7,
        guidance_scale=1.0,
        layout_guidance_scale=1.0,
        asset_control_guidance_scale=1.0,
        layout_to_gauge_grad_scale=1.0,
        scene_units_profile=SCENE_UNITS_PROFILE_FIXED_TRAIN_MEAN,
        val_sliding_window=3 if sliding else 0,
        val_sliding_stride=1,
    )
    sampled = pretrain.cfg_sample_pretrain_latents(
        flow,
        _fixed_sampler_bundle(num_frames),
        args,
        step=11,
        device=torch.device("cpu"),
        return_gauge=True,
    )

    expected = flow.gauge_mean.view(1, 1, 3)
    assert torch.equal(sampled.gauge, expected)
    assert flow.calls
    assert all(tokens is None and scale == 0.0 for tokens, scale in flow.calls)
    assert velocity_state_shapes
    assert all(shape[-1] == 4 for shape in velocity_state_shapes)


def test_fsu_launcher_is_thin_tagged_and_help_is_environment_free() -> None:
    wrapper = REPO_ROOT / "pretrain_ppu_two_nodes_fsu_dlc.sh"
    common = REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh"
    wrapper_text = wrapper.read_text()
    common_text = common.read_text()
    assert 'SCENE_UNITS_PROFILE="fixed_train_mean"' in wrapper_text
    assert "scene_flow_pretrain_fsu_v6" in wrapper_text
    assert "ppu_dlc_two_nodes_fsu_v6_launch" in wrapper_text
    assert "scene_flow_pretrain_waymo_gb64_fsu_v6" in wrapper_text
    assert 'exec bash "${COMMON_LAUNCHER}"' in wrapper_text
    assert 'export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_' in wrapper_text
    assert '--scene_units_profile "${SCENE_UNITS_PROFILE}"' in common_text
    assert '--wandb_run_id "${WANDB_RUN_ID}"' in common_text
    completed = subprocess.run(
        ["bash", str(wrapper), "--help"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={},
    )
    assert "Fixed Scene Units" in completed.stdout


def test_one_node_ppu_fsu_launcher_only_changes_topology_and_accumulation(
    tmp_path: Path,
) -> None:
    two_node_path = REPO_ROOT / "pretrain_ppu_two_nodes_fsu_dlc.sh"
    one_node_path = REPO_ROOT / "pretrain_ppu_one_node_fsu_dlc.sh"
    two_node = two_node_path.read_text(encoding="utf-8")
    one_node = one_node_path.read_text(encoding="utf-8")

    def assignment(text: str, name: str) -> str:
        prefixes = (f"export {name}=", f"{name}=")
        return next(
            line
            for line in text.splitlines()
            if line.startswith(prefixes)
        )

    assert one_node_path.stat().st_mode & 0o111
    assert hashlib.sha256(
        assignment(one_node, "WANDB_API_KEY").encode()
    ).digest() == hashlib.sha256(
        assignment(two_node, "WANDB_API_KEY").encode()
    ).digest()
    assert assignment(one_node, "EXPECTED_NNODES") == "export EXPECTED_NNODES=1"
    assert assignment(one_node, "GRAD_ACCUM_STEPS") == (
        "export GRAD_ACCUM_STEPS=4"
    )
    for name in (
        "PROJECT_ROOT",
        "SCENE_UNITS_PROFILE",
        "LOG_DIR",
        "WANDB_NAME",
    ):
        assert assignment(one_node, name) == assignment(two_node, name)
    assert "ppu_dlc_one_node_fsu_v6_launch" in one_node
    assert 'exec bash "${COMMON_LAUNCHER}"' in one_node

    fake_common = tmp_path / "pretrain_ppu_two_nodes_dlc.sh"
    fake_common.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'TOPOLOGY=%s ACCUM=%s PROFILE=%s RESUME=%s\\n' "
        '"$EXPECTED_NNODES" "$GRAD_ACCUM_STEPS" '
        '"$SCENE_UNITS_PROFILE" "$WANDB_RESUME"\n'
    )
    completed = subprocess.run(
        ["bash", str(one_node_path)],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PROJECT_ROOT": str(tmp_path)},
    )
    assert completed.stdout.strip() == (
        "TOPOLOGY=1 ACCUM=4 PROFILE=fixed_train_mean RESUME=never"
    )

    help_result = subprocess.run(
        ["bash", str(one_node_path), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "1 节点 × 16 PPU" in help_result.stdout
    assert "gradient accumulation 4" in help_result.stdout
    assert "global batch 64" in help_result.stdout


def test_single_node_fsu_launcher_matches_a1_two_node_contract() -> None:
    single = REPO_ROOT / "pretrain_single_node_fsu_ablation.sh"
    common = REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh"
    single_text = single.read_text()
    common_text = common.read_text()

    assert single.stat().st_mode & 0o111
    assert "NPROC_PER_NODE=8" in single_text
    assert "BATCH_SIZE_PER_GPU=1" in single_text
    assert "GRAD_ACCUM_STEPS=8" in single_text
    assert "EXPECTED_GLOBAL_BATCH_SIZE=64" in single_text
    assert "NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS" in single_text
    assert 'SCENE_UNITS_PROFILE="fixed_train_mean"' in single_text
    assert "scene_flow_pretrain_fsu_v6" in single_text
    assert "single_node_fsu_v6_launch" in single_text
    assert "scene_flow_pretrain_waymo_gb64_fsu_v6" in single_text
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-dggt-flow}"' in single_text
    assert 'WANDB_RESUME="never"' in single_text
    assert 'WANDB_RESUME="must"' in single_text
    assert 'export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_' in single_text

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
