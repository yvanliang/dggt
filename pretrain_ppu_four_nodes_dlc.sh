#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

# ============================================================
# 阿里云 PAI-DLC：4 节点 × 每节点 16 张真武 810E PPU
#
# DLC 控制台配置：
#   - 节点数量：4
#   - 每节点规格：ml.gp7vf.16.40xlarge（16 PPU）
#   - 启动命令：
#       bash /mnt/workspace/dggt/pretrain_ppu_four_nodes_dlc.sh
#
# 训练 batch：
#   4 nodes × 16 PPU/node × 1 batch/PPU × 1 accum = global batch 64
#
# DLC 会在四个节点分别执行同一条启动命令，并自动注入：
#   MASTER_ADDR、MASTER_PORT、WORLD_SIZE、RANK、NPROC_PER_NODE
#
# 公共训练参数、PPU 环境检查、ACCL-P 网络配置、wandb 和 torchrun
# 启动逻辑复用 pretrain_ppu_two_nodes_dlc.sh；以下变量将其切换为四节点。
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
BASE_DLC_LAUNCHER="${PROJECT_ROOT}/pretrain_ppu_two_nodes_dlc.sh"

if [[ ! -r "${BASE_DLC_LAUNCHER}" ]]; then
    echo "[错误] 找不到公共 PPU DLC 启动脚本：" >&2
    echo "       ${BASE_DLC_LAUNCHER}" >&2
    exit 1
fi

export PROJECT_ROOT
export EXPECTED_NNODES=4
export EXPECTED_NPROC_PER_NODE=16
export BATCH_SIZE_PER_PPU=1
export GRAD_ACCUM_STEPS=1
export EXPECTED_GLOBAL_BATCH_SIZE=64
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

# 与双机任务分开保存每个 DLC 节点的 torchrun 启动日志。
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/ppu_dlc_four_nodes_launch}"

# metric-gauge v5 固定从 step 0 开始，公共启动器不会 resume 旧 run。
export WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_lr1e4_v5}"

exec bash "${BASE_DLC_LAUNCHER}"
