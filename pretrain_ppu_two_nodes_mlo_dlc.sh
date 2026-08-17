#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] MLO 启动失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

show_help() {
    cat <<'EOF'
Metric Layout Only（A4/MLO）双机 PPU 训练启动器

用法：
  bash /mnt/workspace/dggt/pretrain_ppu_two_nodes_mlo_dlc.sh

固定实验合同：
  - 首次运行从 step 0 开始，不得从 Full checkpoint 分叉。
  - 恢复时只能使用同一 MLO setting 的完整训练 checkpoint。
  - MLO 关闭 early map/actor raster stem，保留全部 late M/G/A context。
  - 设置 RESUME_PATH 时必须同时提供非负 RESUME_EXPECTED_STEP。
  - 可选 WANDB_RUN_ID 用于继续同一个 W&B run。

LOG_DIR、LAUNCH_LOG_DIR 和 WANDB_NAME 可以覆盖，但 basename 必须包含
独立的 mlo 标签（由下划线或连字符分隔），以免覆盖 Full 或其他消融。
EOF
}

if (( $# > 0 )); then
    if (( $# == 1 )) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    echo "[错误] MLO wrapper 不接受 setting 或位置参数；请使用 --help 查看固定实验合同。" >&2
    exit 2
fi

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
export SCENE_UNITS_PROFILE=generated
export WORLD_FEEDBACK_PROFILE=full
export LAYOUT_PATH_PROFILE=metric_layout_only
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/scene_flow_pretrain_mlo_v6}"
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/ppu_dlc_two_nodes_mlo_v6_launch}"
export WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_mlo_v6}"
export WANDB_RESUME="${WANDB_RESUME:-never}"

require_mlo_label() {
    local label="$1"
    local value="$2"
    local base
    base="$(basename "${value%/}")"
    if [[ ! "${base}" =~ (^|[_-])mlo([_-]|$) ]]; then
        echo "[错误] ${label} 的 basename 必须包含独立的 mlo 标签：${value}" >&2
        exit 1
    fi
}

require_mlo_label LOG_DIR "${LOG_DIR}"
require_mlo_label LAUNCH_LOG_DIR "${LAUNCH_LOG_DIR}"
require_mlo_label WANDB_NAME "${WANDB_NAME}"

if [[ -n "${RESUME_PATH:-}" && ! "${RESUME_EXPECTED_STEP:--1}" =~ ^[0-9]+$ ]]; then
    echo "[错误] MLO 恢复必须提供非负 RESUME_EXPECTED_STEP。" >&2
    exit 1
fi

exec bash "${PROJECT_ROOT}/pretrain_ppu_two_nodes_dlc.sh"
