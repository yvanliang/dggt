from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

import train_scene_flow_pretrain as pretrain
from datasets.tools.hdmap_schema import RASTER_SCHEMA_HASH, RASTER_SIGNED_CHANNELS
from dggt.models.scene_flow import WanSceneFlow
from dggt.utils.actor_geometry_condition import (
    MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
    ActorGeometryCondition,
    LayoutMode,
    ProjectedActorGeometry,
)
from dggt.utils.appearance_binding_condition import (
    AppearanceBindingCondition,
    AppearanceMode,
)
from dggt.utils.layout_condition import LayoutConditionBatch, MapMode


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES = 25 * 37
ACTORS = 96
INJECTION_ENV = (
    "LAYOUT_MAP_INJECTION",
    "LAYOUT_ACTOR_INJECTION",
    "LAYOUT_MAP_METRIC_INJECTION",
    "LAYOUT_ACTOR_METRIC_INJECTION",
    "APPEARANCE_CONTEXT_INJECTION",
)


def _tiny_model(*, islo: bool) -> WanSceneFlow:
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
        layout_max_actors=ACTORS,
        layout_map_injection=True,
        layout_actor_injection=True,
        layout_map_metric_injection=not islo,
        layout_actor_metric_injection=not islo,
        appearance_context_injection=not islo,
    )
    model.set_gauge_stats(torch.zeros(3), torch.ones(3))
    return model


def _null_layout() -> tuple[LayoutConditionBatch, torch.Tensor]:
    batch = frames = 1
    raster = torch.zeros((batch, frames, 33, 100, 148), dtype=torch.uint8)
    raster[:, :, RASTER_SIGNED_CHANNELS] = 127

    actor_geometry = ActorGeometryCondition(
        slot_track_id=torch.full((batch, ACTORS), -1, dtype=torch.int64),
        class_id=torch.full((batch, ACTORS), -1, dtype=torch.int8),
        corners_world=torch.zeros(
            (batch, ACTORS, frames, 8, 3), dtype=torch.float64
        ),
        velocity_world=torch.zeros(
            (batch, ACTORS, frames, 3), dtype=torch.float32
        ),
        box_size=torch.zeros((batch, ACTORS, frames, 3), dtype=torch.float32),
        yaw=torch.zeros((batch, ACTORS, frames), dtype=torch.float32),
        is_moving=torch.zeros((batch, ACTORS, frames), dtype=torch.bool),
        track_valid=torch.zeros((batch, ACTORS, frames), dtype=torch.bool),
        slot_valid=torch.zeros((batch, ACTORS), dtype=torch.bool),
        layout_mode=torch.full(
            (batch,), int(LayoutMode.NULL), dtype=torch.int8
        ),
        raw_track_key=[[""] * ACTORS for _ in range(batch)],
    )
    projected = ProjectedActorGeometry(
        bbox_patch=torch.zeros((batch, ACTORS, frames, 4)),
        patch_weight=torch.zeros((batch, ACTORS, frames, PATCHES)),
        log_z_patch=torch.zeros((batch, ACTORS, frames, PATCHES)),
        silhouette_uv=torch.zeros(
            (
                batch,
                ACTORS,
                frames,
                MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
                2,
            )
        ),
        silhouette_vertex_valid=torch.zeros(
            (
                batch,
                ACTORS,
                frames,
                MAX_PROJECTED_ACTOR_SILHOUETTE_VERTICES,
            ),
            dtype=torch.bool,
        ),
        corners_camera=torch.zeros((batch, ACTORS, frames, 8, 3)),
        uv_corners=torch.zeros((batch, ACTORS, frames, 8, 2)),
        velocity_camera=torch.zeros((batch, ACTORS, frames, 3)),
        uv_center=torch.zeros((batch, ACTORS, frames, 2)),
        log_z_w=torch.zeros((batch, ACTORS, frames)),
        center_depth_valid=torch.zeros(
            (batch, ACTORS, frames), dtype=torch.bool
        ),
        frame_support=torch.zeros((batch, ACTORS, frames), dtype=torch.bool),
        metric_support=torch.zeros((batch, ACTORS, frames), dtype=torch.bool),
        in_frustum=torch.zeros((batch, ACTORS, frames), dtype=torch.bool),
        valid=torch.zeros((batch, ACTORS, frames), dtype=torch.bool),
    )
    appearance = AppearanceBindingCondition(
        appearance_tokens=torch.zeros((batch, 1, 1, 1024)),
        appearance_mask=torch.zeros((batch, 1, 1), dtype=torch.bool),
        canonical_uv=torch.zeros((batch, 1, 1, 2)),
        geometry_idx=torch.full((batch, 1), -1, dtype=torch.int64),
        binding_valid=torch.zeros((batch, 1), dtype=torch.bool),
        appearance_mode=torch.full(
            (batch,), int(AppearanceMode.NULL), dtype=torch.int8
        ),
    )
    layout = LayoutConditionBatch(
        raster=raster,
        map_metric=torch.zeros((batch, frames, PATCHES, 5, 4)),
        actor_geometry=actor_geometry,
        projected_actor_geometry=projected,
        appearance=appearance,
        map_mode=torch.full((batch,), int(MapMode.NULL), dtype=torch.int8),
        raster_schema_hash=RASTER_SCHEMA_HASH,
    )
    return layout, torch.full((batch, 1), -1, dtype=torch.int8)


def _forward_islo(model: WanSceneFlow) -> dict[str, torch.Tensor | None]:
    layout, appearance_class_id = _null_layout()
    z_t = torch.randn((1, 1, PATCHES, 4))
    zeros = torch.zeros_like(z_t)
    masks = torch.zeros((1, 1, PATCHES, 1))
    return model(
        z_t,
        torch.tensor([0.5]),
        zeros,
        zeros,
        masks,
        masks,
        masks,
        layout_raster=layout.raster,
        map_metric=layout.map_metric,
        actor_geometry=layout.actor_geometry,
        projected_actor_geometry=layout.projected_actor_geometry,
        appearance=layout.appearance,
        map_mode=layout.map_mode,
        raster_schema_hash=layout.raster_schema_hash,
        appearance_class_id=appearance_class_id,
        camera_condition_tokens=torch.zeros((1, 1, 20)),
        gauge_gen_tokens=torch.zeros((1, 1, 3)),
        gauge_gen_attention_mask=torch.ones((1, 1), dtype=torch.bool),
    )


def _parse_pretrain_args(*extra: str):
    return pretrain.build_argparser().parse_args(
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
            *extra,
        ]
    )


def test_islo_preserves_state_shape_and_only_skips_late_adapters() -> None:
    torch.manual_seed(11)
    full = _tiny_model(islo=False)
    islo = _tiny_model(islo=True).train()
    full_shapes = {key: tuple(value.shape) for key, value in full.state_dict().items()}
    islo_shapes = {key: tuple(value.shape) for key, value in islo.state_dict().items()}
    assert islo_shapes == full_shapes
    for prefix in (
        "map_metric_adapter.",
        "full_actor_gauge_adapter.",
        "appearance_context_adapter.",
    ):
        assert any(key.startswith(prefix) for key in islo_shapes)

    calls = {
        "map_stem": 0,
        "actor_stem": 0,
        "map_metric": 0,
        "actor_metric": 0,
        "appearance_context": 0,
        "appearance_tokens": 0,
    }

    def count(name: str):
        def hook(_module, _inputs, _output):
            calls[name] += 1

        return hook

    watched = {
        "map_stem": islo.layout_map_stem,
        "actor_stem": islo.layout_actor_stem,
        "map_metric": islo.map_metric_adapter,
        "actor_metric": islo.full_actor_gauge_adapter,
        "appearance_context": islo.appearance_context_adapter,
        "appearance_tokens": islo.appearance_proj,
    }
    assert all(module is not None for module in watched.values())
    decay, no_decay = pretrain.split_param_groups(islo)
    optimizer_parameter_ids = {id(parameter) for parameter in (*decay, *no_decay)}
    for name in ("map_metric", "actor_metric", "appearance_context"):
        assert {
            id(parameter) for parameter in watched[name].parameters()
        } <= optimizer_parameter_ids
    ema = pretrain.EMAModel(islo.parameters(), decay=0.9995)
    assert len(ema.shadow_params) == len(list(islo.parameters()))
    del ema

    handles = [module.register_forward_hook(count(name)) for name, module in watched.items()]
    try:
        output = _forward_islo(islo)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {
        "map_stem": 1,
        "actor_stem": 1,
        "map_metric": 0,
        "actor_metric": 0,
        "appearance_context": 0,
        "appearance_tokens": 1,
    }
    assert torch.is_tensor(output["gauge"])
    assert tuple(output["gauge"].shape) == (1, 1, 3)
    loss = output["video"].square().mean() + output["gauge"].square().mean()
    loss.backward()
    for name in ("map_metric", "actor_metric", "appearance_context"):
        assert all(parameter.grad is None for parameter in watched[name].parameters())


def test_islo_cli_and_strict_resume_contract() -> None:
    full_args = _parse_pretrain_args()
    islo_args = _parse_pretrain_args(
        "--no-layout_map_metric_injection",
        "--no-layout_actor_metric_injection",
        "--no-appearance_context_injection",
    )
    assert (
        islo_args.layout_map_injection,
        islo_args.layout_actor_injection,
        islo_args.layout_map_metric_injection,
        islo_args.layout_actor_metric_injection,
        islo_args.appearance_context_injection,
    ) == (True, True, False, False, False)

    payload = {
        "pretrain_resume_contract_version": pretrain.PRETRAIN_RESUME_CONTRACT_VERSION,
        "pretrain_resume_reproducibility": pretrain.PRETRAIN_RESUME_REPRODUCIBILITY,
        "layout_task_probabilities": list(pretrain.LAYOUT_TASK_PROBABILITIES),
        "world_feedback_contract": pretrain.resolve_world_feedback_profile(
            full_args
        ),
        "rgb_render": pretrain.rgb_render_run_summary(full_args),
        "pretrain_resume_critical_args": pretrain.pretrain_resume_critical_args(
            full_args
        ),
        "args": vars(full_args).copy(),
    }
    with pytest.raises(ValueError, match="critical pretraining arguments"):
        pretrain.validate_pretrain_resume_contract(
            payload, islo_args, "full_checkpoint.pt"
        )


def _fake_common_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        *INJECTION_ENV,
        "RESUME_PATH",
        "RESUME_EXPECTED_STEP",
        "WANDB_RUN_ID",
        "SCENE_UNITS_PROFILE",
        "WORLD_FEEDBACK_PROFILE",
    ):
        env.pop(name, None)

    dirs = {
        "WAYMO_DGGT_ROOT": tmp_path / "training",
        "WAYMO_DGGT_VAL_ROOT": tmp_path / "validation",
        "HDMAP_ROOT": tmp_path / "training_hdmap",
        "VAL_HDMAP_ROOT": tmp_path / "validation_hdmap",
        "SCENE_CAPTION_ROOT": tmp_path / "training_captions",
        "SCENE_CAPTION_VAL_ROOT": tmp_path / "validation_captions",
        "QWEN_TEXT_ENCODER": tmp_path / "qwen",
    }
    files = {
        "DGGT_CKPT": tmp_path / "dggt.pt",
        "TOKENIZER_CKPT": tmp_path / "tokenizer.pt",
        "FEATURE_STATS": tmp_path / "feature_stats.pt",
        "SCENE_GAUGE_PATH": tmp_path / "training_gauge.json",
        "VAL_SCENE_GAUGE_PATH": tmp_path / "validation_gauge.json",
        "PULLBACK_CALIBRATION_PATH": tmp_path / "pullback.json",
    }
    for path in dirs.values():
        path.mkdir(parents=True)
    for path in files.values():
        path.touch()

    torch_home = tmp_path / "torch"
    alexnet = torch_home / "hub/checkpoints/alexnet-owt-7be5be79.pth"
    alexnet.parent.mkdir(parents=True)
    alexnet.touch()
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    fake_python.chmod(0o755)

    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29501",
            "WORLD_SIZE": "2",
            "RANK": "0",
            "NPROC_PER_NODE": "16",
            "PROJECT_ROOT": str(REPO_ROOT),
            "PYTHON_BIN": str(fake_python),
            "TORCH_HOME": str(torch_home),
            "LOG_DIR": str(tmp_path / "logs"),
            "LAUNCH_LOG_DIR": str(tmp_path / "launch"),
            **{name: str(path) for name, path in dirs.items()},
            **{name: str(path) for name, path in files.items()},
        }
    )
    return env


def test_full_common_launcher_defaults_to_all_five_injections(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh")],
        env=_fake_common_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "layout injection: early M/G=1/1, late M/G/A=1/1/1" in lines
    for option in (
        "--layout_map_injection",
        "--layout_actor_injection",
        "--layout_map_metric_injection",
        "--layout_actor_metric_injection",
        "--appearance_context_injection",
    ):
        assert lines.count(option) == 1
        assert f"--no-{option[2:]}" not in lines


def test_common_launcher_maps_islo_injections_to_negative_flags(
    tmp_path: Path,
) -> None:
    env = _fake_common_env(tmp_path)
    env.update(
        {
            "LAYOUT_MAP_INJECTION": "1",
            "LAYOUT_ACTOR_INJECTION": "1",
            "LAYOUT_MAP_METRIC_INJECTION": "0",
            "LAYOUT_ACTOR_METRIC_INJECTION": "0",
            "APPEARANCE_CONTEXT_INJECTION": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "layout injection: early M/G=1/1, late M/G/A=0/0/0" in lines
    assert lines.count("--layout_map_injection") == 1
    assert lines.count("--layout_actor_injection") == 1
    assert lines.count("--no-layout_map_metric_injection") == 1
    assert lines.count("--no-layout_actor_metric_injection") == 1
    assert lines.count("--no-appearance_context_injection") == 1


def test_common_launcher_rejects_non_binary_injection(tmp_path: Path) -> None:
    env = _fake_common_env(tmp_path)
    env["LAYOUT_MAP_INJECTION"] = "true"
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "LAYOUT_MAP_INJECTION 必须是 0 或 1" in result.stderr


def _fake_wrapper_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    common = root / "pretrain_ppu_two_nodes_dlc.sh"
    common.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'INJECTION=%s/%s/%s/%s/%s\\n' \"$LAYOUT_MAP_INJECTION\" "
        "\"$LAYOUT_ACTOR_INJECTION\" \"$LAYOUT_MAP_METRIC_INJECTION\" "
        "\"$LAYOUT_ACTOR_METRIC_INJECTION\" \"$APPEARANCE_CONTEXT_INJECTION\"\n"
        "printf 'NAMES=%s|%s|%s|%s\\n' \"$LOG_DIR\" \"$LAUNCH_LOG_DIR\" "
        "\"$WANDB_NAME\" \"$WANDB_RESUME\"\n"
    )
    return root


def _wrapper_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        *INJECTION_ENV,
        "RESUME_PATH",
        "RESUME_EXPECTED_STEP",
        "LOG_DIR",
        "LAUNCH_LOG_DIR",
        "WANDB_NAME",
        "WANDB_RESUME",
    ):
        env.pop(name, None)
    env["PROJECT_ROOT"] = str(root)
    return env


def test_islo_wrapper_locks_environment_and_names(tmp_path: Path) -> None:
    root = _fake_wrapper_root(tmp_path)
    env = _wrapper_env(root)
    env.update(
        {
            "LAYOUT_MAP_INJECTION": "0",
            "LAYOUT_ACTOR_INJECTION": "0",
            "LAYOUT_MAP_METRIC_INJECTION": "1",
            "LAYOUT_ACTOR_METRIC_INJECTION": "1",
            "APPEARANCE_CONTEXT_INJECTION": "1",
            "WANDB_RESUME": "allow",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "pretrain_ppu_two_nodes_islo_dlc.sh"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fresh step-0 architecture ablation" in result.stdout
    assert "INJECTION=1/1/0/0/0" in result.stdout
    names_line = next(
        line for line in result.stdout.splitlines() if line.startswith("NAMES=")
    )
    assert names_line.endswith("|never")
    assert all("islo" in value for value in names_line[6:].split("|")[:3])


@pytest.mark.parametrize("variable", ("LOG_DIR", "LAUNCH_LOG_DIR", "WANDB_NAME"))
def test_islo_wrapper_rejects_ambiguous_name(
    tmp_path: Path, variable: str
) -> None:
    root = _fake_wrapper_root(tmp_path)
    env = _wrapper_env(root)
    env[variable] = "full_v6"
    result = subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "pretrain_ppu_two_nodes_islo_dlc.sh"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"{variable} 必须包含小写 islo" in result.stderr


def test_islo_wrapper_rejects_resume_path(tmp_path: Path) -> None:
    root = _fake_wrapper_root(tmp_path)
    env = _wrapper_env(root)
    env["RESUME_PATH"] = "/tmp/full_checkpoint.pt"
    result = subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "pretrain_ppu_two_nodes_islo_dlc.sh"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "必须从 step 0 训练" in result.stderr


def test_islo_wrapper_help_and_positional_contract() -> None:
    wrapper = REPO_ROOT / "pretrain_ppu_two_nodes_islo_dlc.sh"
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert 'export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_' in wrapper_text
    help_result = subprocess.run(
        ["bash", str(wrapper), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "fresh step-0 architecture ablation" in help_result.stdout

    positional_result = subprocess.run(
        ["bash", str(wrapper), "unexpected"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert positional_result.returncode == 2
    assert "不接受位置参数" in positional_result.stderr


def test_a3_single_node_launcher_matches_islo_contract(tmp_path: Path) -> None:
    launcher = REPO_ROOT / "pretrain_single_node_islo_ablation.sh"
    common = REPO_ROOT / "pretrain_ppu_two_nodes_dlc.sh"
    single_text = launcher.read_text(encoding="utf-8")
    common_text = common.read_text(encoding="utf-8")
    assert launcher.stat().st_mode & 0o111
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
    ):
        assert fragment in single_text
        assert fragment in common_text

    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"' in single_text
    assert 'GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-three_quarter}"' in common_text

    env = _fake_common_env(tmp_path)
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { return 0; }\n")
    env.update(
        {
            "CONDA_SH": str(conda_sh),
            "LOG_DIR": str(tmp_path / "scene_flow_pretrain_islo_v6"),
            "LAUNCH_LOG_DIR": str(tmp_path / "single_node_islo_v6_launch"),
            "WANDB_NAME": "scene_flow_pretrain_waymo_gb64_lr1e4_islo_v6",
        }
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert (
        "global batch size: 64 = 1 node × 8 gpu/node × 1 batch/gpu × 8 accum"
        in lines
    )
    assert "layout injection: early M/G=1/1, late M/G/A=0/0/0" in lines
    assert "training start: fresh step-0 architecture ablation" in lines
    for option in (
        "--layout_map_injection",
        "--layout_actor_injection",
        "--no-layout_map_metric_injection",
        "--no-layout_actor_metric_injection",
        "--no-appearance_context_injection",
    ):
        assert lines.count(option) == 1
    assert lines[lines.index("--grad_accum_steps") + 1] == "8"
    assert lines[lines.index("--world_feedback_profile") + 1] == "full"
    assert lines[lines.index("--scene_units_profile") + 1] == "generated"
    assert lines[lines.index("--wandb_resume") + 1] == "never"
    assert "--resume_path" not in lines
    assert "--wandb_run_id" not in lines


def test_a3_half_node_p6000_launcher_matches_islo_contract() -> None:
    reference = (
        REPO_ROOT / "pretrain_single_node_islo_ablation.sh"
    ).read_text(encoding="utf-8")
    launcher_path = REPO_ROOT / "pretrain_half_node_p6000_islo_ablation.sh"
    launcher = launcher_path.read_text(encoding="utf-8")
    p6000_base = (REPO_ROOT / "pretrain_half_node_p6000.sh").read_text(
        encoding="utf-8"
    )

    def assignment(text: str, name: str) -> str:
        prefix = f"{name}="
        return next(line for line in text.splitlines() if line.startswith(prefix))

    def function(text: str, name: str) -> str:
        start = text.index(f"{name}() {{\n")
        end = text.index("\n}\n", start) + len("\n}\n")
        return text[start:end]

    def wandb_export(text: str) -> str:
        return next(
            line
            for line in text.splitlines()
            if line.startswith("export WANDB_API_KEY=")
        )

    assert launcher_path.stat().st_mode & 0o111
    assert wandb_export(launcher) == wandb_export(p6000_base)
    assert assignment(launcher, "NPROC_PER_NODE") == "NPROC_PER_NODE=4"
    assert assignment(launcher, "GRAD_ACCUM_STEPS") == "GRAD_ACCUM_STEPS=16"
    assert assignment(launcher, "EXPECTED_GLOBAL_BATCH_SIZE") == (
        "EXPECTED_GLOBAL_BATCH_SIZE=64"
    )
    assert function(launcher, "build_train_args") == function(
        reference, "build_train_args"
    )

    for name in (
        "BATCH_SIZE_PER_GPU",
        "EXPECTED_GLOBAL_BATCH_SIZE",
        "NUM_WORKERS",
        "PREFETCH_FACTOR",
        "VAL_NUM_WORKERS",
        "DATALOADER_WORKER_THREADS",
        "DATALOADER_OUT_OF_ORDER",
        "GRADIENT_CHECKPOINTING",
        "MAX_STEPS",
        "DECAY_END_STEPS",
        "SAVE_EVERY",
        "VAL_EVERY",
        "VAL_BATCHES",
        "VAL_LOG_IMAGES",
        "VAL_INFERENCE_SCENES",
        "VAL_SAMPLE_STEPS",
        "LOG_EVERY",
        "CFG_SCALE",
        "LAYOUT_GUIDANCE_SCALE",
        "ASSET_CONTROL_GUIDANCE_SCALE",
        "VAL_GUIDANCE_SCALES",
        "LAYOUT_MAX_ACTORS",
        "STATIC_FAR_PLANE_M",
        "SCENE_UNITS_PROFILE",
        "WORLD_FEEDBACK_PROFILE",
        "LAYOUT_MAP_INJECTION",
        "LAYOUT_ACTOR_INJECTION",
        "LAYOUT_MAP_METRIC_INJECTION",
        "LAYOUT_ACTOR_METRIC_INJECTION",
        "APPEARANCE_CONTEXT_INJECTION",
        "WANDB_PROJECT",
        "WANDB_NAME",
    ):
        assert assignment(launcher, name) == assignment(reference, name)


def test_a3_single_node_launcher_rejects_resume_and_positionals() -> None:
    launcher = REPO_ROOT / "pretrain_single_node_islo_ablation.sh"
    positional = subprocess.run(
        ["bash", str(launcher), "unexpected"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert positional.returncode == 2
    assert "不接受位置参数" in positional.stderr

    env = os.environ.copy()
    env["RESUME_PATH"] = "/tmp/full_checkpoint.pt"
    resume = subprocess.run(
        ["bash", str(launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert resume.returncode != 0
    assert "必须从 step 0 训练" in resume.stderr


@pytest.mark.parametrize("variable", ("LOG_DIR", "LAUNCH_LOG_DIR", "WANDB_NAME"))
def test_a3_single_node_launcher_rejects_ambiguous_name(variable: str) -> None:
    launcher = REPO_ROOT / "pretrain_single_node_islo_ablation.sh"
    env = os.environ.copy()
    for name in ("LOG_DIR", "LAUNCH_LOG_DIR", "WANDB_NAME", "RESUME_PATH"):
        env.pop(name, None)
    env[variable] = "full_v6"
    result = subprocess.run(
        ["bash", str(launcher)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"{variable} 必须包含小写 islo" in result.stderr


def test_islo_inactive_adapters_are_supported_by_ddp_unused_parameter_mode() -> None:
    source = Path(pretrain.__file__).read_text(encoding="utf-8")
    assert "find_unused_parameters=True" in source
