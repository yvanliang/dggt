#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

usage() {
    cat <<'EOF'
用法：
  bash /mnt/workspace/dggt/pretrain_ppu_two_nodes_los_dlc.sh

LOS 是 Latent-Only Supervision 的固定简称。此脚本不接受 setting 参数，
并固定使用 latent_only world-feedback profile：
  - 只关闭 RGB、decoded-feature、frozen-head 三层 scene world feedback；
  - sky-view reconstruction、patch/refined sky-mask、gauge 与 layout 均保留；
  - decoder、DGGT heads、Gaussian renderer 与 LPIPS 仍完整执行并参与 backward；
  - 初次运行必须从 step 0 开始，不得从 Full checkpoint 分叉；
  - 后续只能从同一 LOS 配置的完整 checkpoint 恢复，且必须显式给出
    RESUME_EXPECTED_STEP；恢复时建议同时设置 WANDB_RESUME=must 和 WANDB_RUN_ID。

可覆盖环境变量：
  LOG_DIR             训练输出目录；basename 必须含独立的 los 标签
  LAUNCH_LOG_DIR      两节点启动日志目录；basename 必须含独立的 los 标签
  WANDB_NAME          W&B run 名；必须含独立的 los 标签
  RESUME_PATH         LOS 自身的完整训练 checkpoint；默认空（step 0）
  RESUME_EXPECTED_STEP
                       恢复 step 的严格断言；恢复时必须为非负整数
  WANDB_RUN_ID        恢复已有 W&B run 时使用
  WANDB_RESUME        初次运行默认 never；恢复建议 must
EOF
}

if [[ $# -gt 0 ]]; then
    if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
        usage
        exit 0
    fi
    echo "[错误] LOS launcher 不接受 setting 参数。" >&2
    usage >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
BASE_DLC_LAUNCHER="${PROJECT_ROOT}/pretrain_ppu_two_nodes_dlc.sh"
if [[ ! -r "${BASE_DLC_LAUNCHER}" ]]; then
    echo "[错误] 找不到公共双机 PPU 启动脚本：${BASE_DLC_LAUNCHER}" >&2
    exit 1
fi

export PROJECT_ROOT
export SCENE_UNITS_PROFILE=generated
export WORLD_FEEDBACK_PROFILE=latent_only
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/scene_flow_pretrain_los_v6}"
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/ppu_dlc_two_nodes_los_v6_launch}"
export WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_los_v6}"
export WANDB_RESUME="${WANDB_RESUME:-never}"

require_los_tag() {
    local label="$1"
    local value="${2%/}"
    local basename="${value##*/}"
    local normalized="${basename,,}"
    if [[ ! "${normalized}" =~ (^|[^[:alnum:]])los([^[:alnum:]]|$) ]]; then
        echo "[错误] ${label} 的 basename 必须含独立的 los 标签：${value}" >&2
        exit 1
    fi
}

require_los_tag LOG_DIR "${LOG_DIR}"
require_los_tag LAUNCH_LOG_DIR "${LAUNCH_LOG_DIR}"
require_los_tag WANDB_NAME "${WANDB_NAME}"

exec bash "${BASE_DLC_LAUNCHER}"
