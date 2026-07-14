#!/usr/bin/env bash
set -euo pipefail

umask 000

# ============================================================
# 单机测试配置：1 台机器 x N GPU
# 可在命令行覆盖，例如：
#   N=4 ./pretrain_single_node_test.sh
# ============================================================
N="${N:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-22229}"

NNODES=1
NPROC_PER_NODE="${N}"

# ============================================================
# 项目与环境
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/vol1/liangyy_workspace/dggt}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/vol1/liangyy_workspace/waymo_processed_dggt}"
CONDA_SH="${CONDA_SH:-/mnt/vol1/liangyy_workspace/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dggt}"

# ============================================================
# 数据与模型路径
# ============================================================
WAYMO_DGGT_ROOT="${DATASET_ROOT}/training"
WAYMO_DGGT_VAL_ROOT="${DATASET_ROOT}/validation"
DGGT_CKPT="${PROJECT_ROOT}/pretrained/model_latest_waymo.pt"
TOKENIZER_CKPT="${PROJECT_ROOT}/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt"
FEATURE_STATS="${PROJECT_ROOT}/logs/scene_flow_pretrain_1024/feature_stats_pretrain_v3_798.pt"
SCENE_CAPTION_ROOT="${DATASET_ROOT}/training_captions"
SCENE_CAPTION_VAL_ROOT="${DATASET_ROOT}/validation_captions"
QWEN_TEXT_ENCODER="${QWEN_TEXT_ENCODER:-/mnt/vol1/liangyy_workspace/model/Qwen/Qwen3-0.6B}"
LOG_DIR="${PROJECT_ROOT}/logs/scene_flow_pretrain_single_node_test"
LAUNCH_LOG_DIR="${PROJECT_ROOT}/logs/single_node_test_launch"

# ============================================================
# 测试训练配置
# ============================================================
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
VAL_EVERY="${VAL_EVERY:-100}"
SCENE_END="${SCENE_END:-798}"
VAL_SCENE_END="${VAL_SCENE_END:-202}"

UNCOND_DROP_PROB=0.1
TEXT_UNCOND_DROP_PROB=0.1
ASSET_UNCOND_DROP_PROB=0.2
CAMERA_UNCOND_DROP_PROB=0.2
ALL_COND_DROP_PROB=0.05
GUIDANCE_SCALE=1.0
ASSET_CONTROL_GUIDANCE_SCALE=1.0
CAMERA_GUIDANCE_SCALE=1.0
VAL_GUIDANCE_SCALES="1.0,2.0,4.0"
GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))

cuda_visible_devices() {
    local count="$1"
    local devices=""
    local i
    for ((i = 0; i < count; i++)); do
        if [[ -n "${devices}" ]]; then
            devices+=","
        fi
        devices+="${i}"
    done
    printf '%s\n' "${devices}"
}

setup_common_env() {
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(cuda_visible_devices "${NPROC_PER_NODE}")"
    export PYTHONUNBUFFERED=1

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export WANDB_DISABLED=true
    export WANDB_MODE=disabled

    # 单机测试默认不走 IB，避免继承双节点脚本的网卡/HCA 约束。
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
}

check_dir() {
    local name="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "缺少目录 ${name}: ${path}" >&2
        exit 1
    fi
    echo "OK 目录 ${name}: ${path}"
}

check_file() {
    local name="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "缺少文件 ${name}: ${path}" >&2
        exit 1
    fi
    echo "OK 文件 ${name}: ${path}"
}

if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N must be a positive integer, got: ${N}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
setup_common_env

mkdir -p "${LOG_DIR}" "${LAUNCH_LOG_DIR}"

echo "检查本机文件……"
check_dir "PROJECT_ROOT" "${PROJECT_ROOT}"
check_file "CONDA_SH" "${CONDA_SH}"
check_dir "WAYMO_DGGT_ROOT" "${WAYMO_DGGT_ROOT}"
check_dir "WAYMO_DGGT_VAL_ROOT" "${WAYMO_DGGT_VAL_ROOT}"
check_file "DGGT_CKPT" "${DGGT_CKPT}"
check_file "TOKENIZER_CKPT" "${TOKENIZER_CKPT}"
check_file "FEATURE_STATS" "${FEATURE_STATS}"
check_dir "QWEN_TEXT_ENCODER" "${QWEN_TEXT_ENCODER}"

cd "${PROJECT_ROOT}"

echo "=== 启动单机 ${NPROC_PER_NODE} GPU 测试预训练 ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "全局 batch size: ${GLOBAL_BATCH_SIZE} = ${NNODES} node x ${NPROC_PER_NODE} gpu/node x ${BATCH_SIZE_PER_GPU} batch/gpu x ${GRAD_ACCUM_STEPS} accum"
echo "日志: ${LAUNCH_LOG_DIR}/single_node_test.log"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_scene_flow_pretrain.py \
    --image_dir "${WAYMO_DGGT_ROOT}" \
    --val_image_dir "${WAYMO_DGGT_VAL_ROOT}" \
    --dggt_ckpt_path "${DGGT_CKPT}" \
    --tokenizer_ckpt_path "${TOKENIZER_CKPT}" \
    --feature_stats_path "${FEATURE_STATS}" \
    --log_dir "${LOG_DIR}" \
    --caption_root "${SCENE_CAPTION_ROOT}" \
    --val_caption_root "${SCENE_CAPTION_VAL_ROOT}" \
    --text_encoder_path "${QWEN_TEXT_ENCODER}" \
    --scene_start 0 \
    --scene_end "${SCENE_END}" \
    --sequence_length 10 \
    --val_sliding_window 10 \
    --val_sliding_stride 7 \
    --camera_anchor_context_dropout 0.25 \
    --patch_grid_h 25 \
    --patch_grid_w 37 \
    --latent_dim 1024 \
    --batch_size "${BATCH_SIZE_PER_GPU}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --lr 5e-5 \
    --final_lr 5e-6 \
    --weight_decay 0.0 \
    --optimizer_type gmuon \
    --ema_decay 0.9997 \
    --warmup_steps 3000 \
    --max_steps "${MAX_STEPS}" \
    --save_every "${SAVE_EVERY}" \
    --shift 10.0 \
    --weighting_scheme waver \
    --mode_scale 1.29 \
    --loss_weighting_scheme none \
    --prediction_type v \
    --lambda_repa 0.5 \
    --base_model_coeff 0.25 \
    --lambda_boundary 0.25 \
    --lambda_camera_flow 0.1 \
    --lambda_camera_pose 0.5 \
    --lambda_sky_flow 0.1 \
    --uncond_drop_prob "${UNCOND_DROP_PROB}" \
    --text_uncond_drop_prob "${TEXT_UNCOND_DROP_PROB}" \
    --asset_uncond_drop_prob "${ASSET_UNCOND_DROP_PROB}" \
    --camera_uncond_drop_prob "${CAMERA_UNCOND_DROP_PROB}" \
    --all_cond_drop_prob "${ALL_COND_DROP_PROB}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --asset_control_guidance_scale "${ASSET_CONTROL_GUIDANCE_SCALE}" \
    --camera_guidance_scale "${CAMERA_GUIDANCE_SCALE}" \
    --val_guidance_scales "${VAL_GUIDANCE_SCALES}" \
    --val_scene_start 0 \
    --val_scene_end "${VAL_SCENE_END}" \
    --val_every "${VAL_EVERY}" \
    --val_batches 1 \
    --val_log_images 10 \
    --val_sample_steps 35 \
    --grad_clip_norm 1.0 \
    --seed 0 \
    --precision bf16 \
    --ddp_timeout_minutes 60 \
    2>&1 | tee "${LAUNCH_LOG_DIR}/single_node_test.log"
