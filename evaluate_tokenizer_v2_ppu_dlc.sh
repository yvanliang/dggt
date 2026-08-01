#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] line ${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2' ERR
umask 000

# ============================================================
# PAI-DLC：单机 16 张真武 810E PPU，Tokenizer v2 checkpoint sweep
#
# DLC 控制台：
#   节点数量：1
#   每节点规格：ml.gp7vf.16.40xlarge（16 PPU）
#   启动命令：
#     bash /mnt/workspace/dggt/evaluate_tokenizer_v2_ppu_dlc.sh
#
# 第一轮（默认）：
#   55000,60000,65000,70000,75000 × 10,12,14 帧 = 15 个配置
#   15 张 PPU 各跑一个配置；第 16 张只参与启动与最终汇总。
#
# 第二轮（后续测试时，把下面 STEPS 的默认值替换为这一行）：
#   STEPS=80000,85000,90000,95000,100000
#
# 每个配置固定选择 300 个互不重复 scene。最终统计以 scene 为单位，
# 不把同一 scene 的帧或窗口当成独立样本。
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
DATASETS_DIR="${DATASETS_DIR:-/mnt/workspace/datasets}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${DATASETS_DIR}/waymo_processed_dggt}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${PROCESSED_ROOT}}"
RAW_ROOT="${RAW_ROOT:-${DATASETS_DIR}/waymo}"
ASSET_ROOT="${ASSET_ROOT:-${PROCESSED_ROOT}/objects_ply_transformed}"
DGGT_CKPT="${DGGT_CKPT:-${PROJECT_ROOT}/pretrained/model_latest_waymo.pt}"
TOKENIZER_CKPT_DIR="${TOKENIZER_CKPT_DIR:-/mnt/workspace/logs/tokenizer_t0_v2_stageA/ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/tokenizer_v2_ppu_eval_round1}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-${PROJECT_ROOT}/runs/tokenizer_v2_fixed_selection_300.json}"

# 第一轮。第二轮只需改为：80000,85000,90000,95000,100000
STEPS="${STEPS:-55000,60000,65000,70000,75000}"
FRAME_COUNTS="${FRAME_COUNTS:-10,12,14}"
NUM_SCENES="${NUM_SCENES:-300}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
DRY_RUN="${DRY_RUN:-0}"
DGGT_EVAL_RUN_ID="${DGGT_EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}}"

IFS=',' read -r -a STEP_ARRAY <<< "${STEPS}"
IFS=',' read -r -a FRAME_ARRAY <<< "${FRAME_COUNTS}"
CONFIG_COUNT=$(( ${#STEP_ARRAY[@]} * ${#FRAME_ARRAY[@]} ))
if (( CONFIG_COUNT > 16 )); then
    echo "[错误] ${CONFIG_COUNT} 个配置无法在 16 PPU 内一轮跑完；请拆成两轮。" >&2
    exit 1
fi

require_dlc_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "[错误] 缺少 PAI-DLC 注入变量 ${name}" >&2
        exit 1
    fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
    for name in MASTER_ADDR MASTER_PORT WORLD_SIZE RANK NPROC_PER_NODE; do
        require_dlc_env "${name}"
    done
    if [[ "${WORLD_SIZE}" != "1" || "${RANK}" != "0" || "${NPROC_PER_NODE}" != "16" ]]; then
        echo "[错误] 必须是单节点 16 PPU DLC 任务；收到 WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}, NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
        exit 1
    fi
    DLC_MASTER_ADDR="${MASTER_ADDR}"
    DLC_MASTER_PORT="${MASTER_PORT}"
else
    DLC_MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    DLC_MASTER_PORT="${MASTER_PORT:-29621}"
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-/mnt/workspace/model/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${DGGT_NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export DGGT_DEVICE_BACKEND="${DGGT_DEVICE_BACKEND:-ppu}"
export DGGT_PPU_MHA_BATCH_CHUNK_SIZE="${DGGT_PPU_MHA_BATCH_CHUNK_SIZE:-4096}"
export DGGT_EVAL_RUN_ID

EVAL_ARGS=(
    "${PROJECT_ROOT}/tools/evaluate_tokenizer_v2_ppu.py"
    --dggt-checkpoint "${DGGT_CKPT}"
    --checkpoint-dir "${TOKENIZER_CKPT_DIR}"
    --steps "${STEPS}"
    --frame-counts "${FRAME_COUNTS}"
    --processed-root "${PROCESSED_ROOT}"
    --transfer-root "${TRANSFER_ROOT}"
    --raw-root "${RAW_ROOT}"
    --asset-root "${ASSET_ROOT}"
    --dataset-split validation
    --num-scenes "${NUM_SCENES}"
    --sample-window 20
    --precision bf16
    --bootstrap-samples 2000
    --save-visuals-per-config 2
    --expected-world-size 16
    --selection-manifest "${SELECTION_MANIFEST}"
    --output-dir "${OUTPUT_DIR}"
)

echo "============================================================"
echo "Tokenizer v2 PPU evaluation"
echo "steps: ${STEPS}"
echo "frame counts: ${FRAME_COUNTS}"
echo "scenes/config: ${NUM_SCENES}"
echo "parallel configs: ${CONFIG_COUNT}/16"
echo "checkpoint dir: ${TOKENIZER_CKPT_DIR}"
echo "output: ${OUTPUT_DIR}"
echo "fixed selection manifest: ${SELECTION_MANIFEST}"
echo "python: ${PYTHON_BIN}"
echo "run id: ${DGGT_EVAL_RUN_ID}"
echo "nccl debug: ${NCCL_DEBUG}"
echo "master: ${DLC_MASTER_ADDR}:${DLC_MASTER_PORT}"
echo "============================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN command: %q -m torch.distributed.run ' "${PYTHON_BIN}"
    printf '%q ' \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=16 \
        --master_addr="${DLC_MASTER_ADDR}" \
        --master_port="${DLC_MASTER_PORT}" \
        "${EVAL_ARGS[@]}"
    printf '\n'
    exit 0
fi

for required_path in \
    "${PROJECT_ROOT}/tools/evaluate_tokenizer_v2_ppu.py" \
    "${DGGT_CKPT}" \
    "${PROCESSED_ROOT}" \
    "${TOKENIZER_CKPT_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[错误] 路径不存在：${required_path}" >&2
        exit 1
    fi
done

for step in "${STEP_ARRAY[@]}"; do
    checkpoint="${TOKENIZER_CKPT_DIR}/scene_tokenizer_step_$(printf '%06d' "${step}").pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "[错误] 缺少 tokenizer checkpoint：${checkpoint}" >&2
        exit 1
    fi
done

ALEXNET_CKPT="${TORCH_HOME}/hub/checkpoints/alexnet-owt-7be5be79.pth"
if [[ ! -f "${ALEXNET_CKPT}" ]]; then
    echo "[错误] LPIPS 离线评测缺少 AlexNet 权重：${ALEXNET_CKPT}" >&2
    echo "       请先运行 ${PROJECT_ROOT}/download_ppu_model_weights.sh" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[错误] 找不到 Python 可执行文件：${PYTHON_BIN}" >&2
    exit 1
fi

EXPECTED_DEVICE_COUNT=16 "${PYTHON_BIN}" - <<'PY'
import os
import torch

expected = int(os.environ["EXPECTED_DEVICE_COUNT"])
actual = torch.cuda.device_count()
print(f"PPU/CUDA available={torch.cuda.is_available()}, visible devices={actual}")
if not torch.cuda.is_available() or actual != expected:
    raise RuntimeError(f"expected {expected} PPU devices, got {actual}")
PY

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=16 \
    --master_addr="${DLC_MASTER_ADDR}" \
    --master_port="${DLC_MASTER_PORT}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/launch.log"
