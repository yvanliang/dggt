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
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-three_quarter}"
export STATIC_FAR_PLANE_M="${STATIC_FAR_PLANE_M:-120}"
# Checkpoint serialization is a deliberate CPFS write barrier.  On four nodes
# it need not run every 2500 steps; 5000 halves those unavoidable global stalls
# while preserving the training computation and can still be overridden.
export SAVE_EVERY="${SAVE_EVERY:-5000}"

# Each node runs 16 training ranks. Eight single-threaded workers/rank plus the
# 16 rank processes use 144 of 176 host CPU execution slots, leaving 32 for
# communication and the CPFS client. A single prefetched batch halves the old
# in-flight sample count and prevents synchronized 2x read bursts. Validation
# data is consumed only by rank 0 and the few ranks assigned a CFG scale, so
# synchronous loading avoids unused worker prefetch entirely.
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
export DATALOADER_WORKER_THREADS="${DATALOADER_WORKER_THREADS:-1}"
export DATALOADER_OUT_OF_ORDER="${DATALOADER_OUT_OF_ORDER:-1}"

# The old 4-thread defaults multiplied across 16 ranks × 8 workers and heavily
# oversubscribed the node.  Layout projection consists mostly of many small
# CPU kernels, for which one native thread per process is the stable setting.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# 与双机任务分开保存每个 DLC 节点的 torchrun 启动日志。
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/ppu_dlc_four_nodes_v6_launch}"
export VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-50}"

# layout-v2 v6 固定从 step 0 开始，公共启动器不会 resume 旧 run。
export WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_lr1e4_v6}"

exec bash "${BASE_DLC_LAUNCHER}"
