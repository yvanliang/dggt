#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

usage() {
    cat <<'EOF'
用法：
  bash /mnt/workspace/dggt/pretrain_ppu_two_nodes_islo_dlc.sh

Image-Space Layout Only（A3/ISLO）是 fresh step-0 architecture ablation：
  - early map/actor raster stem: on/on
  - late map/actor metric reader: off/off
  - late appearance context/scatter: off

该实验不允许从 Full 或其他 setting 的 checkpoint 分叉，并继承公共双机
launcher 的 2 节点 × 16 PPU、global batch 64 和 50-step validation 合同。

可覆盖环境变量：
  LOG_DIR         训练输出目录；值必须包含小写 islo
  LAUNCH_LOG_DIR  双机启动日志目录；值必须包含小写 islo
  WANDB_NAME      W&B run 名；值必须包含小写 islo
EOF
}

if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    echo "[错误] ISLO launcher 不接受位置参数。" >&2
    usage >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
COMMON_LAUNCHER="${PROJECT_ROOT}/pretrain_ppu_two_nodes_dlc.sh"
if [[ ! -r "${COMMON_LAUNCHER}" ]]; then
    echo "[错误] 找不到可读的公共双机 PPU 启动脚本：${COMMON_LAUNCHER}" >&2
    exit 1
fi

if [[ -n "${RESUME_PATH:-}" ]]; then
    echo "[错误] ISLO 必须从 step 0 训练，拒绝 RESUME_PATH=${RESUME_PATH}" >&2
    exit 1
fi

export PROJECT_ROOT
export SCENE_UNITS_PROFILE=generated
export WORLD_FEEDBACK_PROFILE=full
export LAYOUT_MAP_INJECTION=1
export LAYOUT_ACTOR_INJECTION=1
export LAYOUT_MAP_METRIC_INJECTION=0
export LAYOUT_ACTOR_METRIC_INJECTION=0
export APPEARANCE_CONTEXT_INJECTION=0
export WANDB_RESUME=never

export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/scene_flow_pretrain_islo_v6}"
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/ppu_dlc_islo_v6_launch}"
export WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_lr1e4_islo_v6}"

require_islo_name() {
    local name="$1"
    local value="${2%/}"
    local basename="${value##*/}"

    if [[ "${basename}" != *islo* ]]; then
        echo "[错误] ${name} 必须包含小写 islo，当前值：${value}" >&2
        exit 1
    fi
}

require_islo_name LOG_DIR "${LOG_DIR}"
require_islo_name LAUNCH_LOG_DIR "${LAUNCH_LOG_DIR}"
require_islo_name WANDB_NAME "${WANDB_NAME}"

unset RESUME_PATH RESUME_EXPECTED_STEP WANDB_RUN_ID

echo "[A3/ISLO] fresh step-0 architecture ablation"
echo "[ISLO] layout injection: early M/G=1/1, late M/G/A=0/0/0"
echo "[ISLO] training log dir: ${LOG_DIR}"
echo "[ISLO] launch log dir: ${LAUNCH_LOG_DIR}"
echo "[ISLO] wandb name: ${WANDB_NAME} (resume=${WANDB_RESUME})"

exec bash "${PROJECT_ROOT}/pretrain_ppu_two_nodes_dlc.sh"
