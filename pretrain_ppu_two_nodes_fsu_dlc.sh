#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

usage() {
    cat <<'EOF'
用法：
  bash pretrain_ppu_two_nodes_fsu_dlc.sh
  bash pretrain_ppu_two_nodes_fsu_dlc.sh --resume CHECKPOINT EXPECTED_STEP WANDB_RUN_ID

Fixed Scene Units（fsu）消融：
  - 无参数时从 step 0 开始全新训练，禁止从 Full/generated 分叉；
  - resume 只接受显式记录 fixed_train_mean 的完整 FSU checkpoint；
  - resume 必须同时给出 checkpoint、期望 step 和原 W&B run id，W&B resume 固定为 must；
  - 保留公共 launcher 的 2 节点 × 16 PPU、global batch 64 和 50-step validation。
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 && ! ( $# -eq 4 && "$1" == "--resume" ) ]]; then
    usage >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
COMMON_LAUNCHER="${PROJECT_ROOT}/pretrain_ppu_two_nodes_dlc.sh"
if [[ ! -r "${COMMON_LAUNCHER}" ]]; then
    echo "[错误] 找不到公共两机 PPU launcher：${COMMON_LAUNCHER}" >&2
    exit 1
fi

export PROJECT_ROOT
export SCENE_UNITS_PROFILE="fixed_train_mean"
export LOG_DIR="${LOG_DIR:-/mnt/workspace/dggt/logs/scene_flow_pretrain_fsu_v6}"
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-/mnt/workspace/dggt/logs/ppu_dlc_two_nodes_fsu_v6_launch}"
export WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_fsu_v6}"

for tagged_name in LOG_DIR LAUNCH_LOG_DIR WANDB_NAME; do
    tagged_value="${!tagged_name}"
    if [[ "${tagged_value,,}" != *fsu* ]]; then
        echo "[错误] ${tagged_name} 必须包含固定简称 fsu，当前值：${tagged_value}" >&2
        exit 1
    fi
done

if [[ $# -eq 0 ]]; then
    unset RESUME_PATH RESUME_EXPECTED_STEP WANDB_RUN_ID || true
    export RESUME_PATH=""
    export RESUME_EXPECTED_STEP="-1"
    export WANDB_RUN_ID=""
    export WANDB_RESUME="never"
else
    export RESUME_PATH="$2"
    export RESUME_EXPECTED_STEP="$3"
    export WANDB_RUN_ID="$4"
    export WANDB_RESUME="must"
    if [[ ! "${RESUME_EXPECTED_STEP}" =~ ^[0-9]+$ ]]; then
        echo "[错误] EXPECTED_STEP 必须是非负整数，当前值：${RESUME_EXPECTED_STEP}" >&2
        exit 1
    fi
    if [[ -z "${WANDB_RUN_ID}" ]]; then
        echo "[错误] FSU resume 必须提供原 W&B run id。" >&2
        exit 1
    fi
    if [[ ! -f "${RESUME_PATH}" ]]; then
        echo "[错误] FSU resume checkpoint 不存在：${RESUME_PATH}" >&2
        exit 1
    fi
    "${PYTHON_BIN:-python}" - "${RESUME_PATH}" "${RESUME_EXPECTED_STEP}" <<'PY'
import sys
import torch

path, expected_step = sys.argv[1], int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
    raise SystemExit(f"[错误] {path} 不是版本化 checkpoint")
required = {"step", "scene_flow", "ema_scene_flow", "optimizer", "lr_scheduler"}
missing = sorted(required - set(payload))
if missing:
    raise SystemExit(
        f"[错误] FSU resume 只接受完整训练 checkpoint；缺少 {missing}"
    )
config = payload.get("scene_flow_config")
profile = config.get("scene_units_profile", "generated") if isinstance(config, dict) else "generated"
if profile != "fixed_train_mean":
    raise SystemExit(
        f"[错误] 明确拒绝 Full/generated checkpoint；scene_units_profile={profile!r}"
    )
step = int(payload["step"])
if step != expected_step:
    raise SystemExit(
        f"[错误] checkpoint step={step} 与 EXPECTED_STEP={expected_step} 不一致"
    )
PY
fi

exec bash "${COMMON_LAUNCHER}"
