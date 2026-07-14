#!/usr/bin/env bash
set -euo pipefail

umask 000

# ============================================================
# Single-node training config: 1 machine x 8 GPUs
# ============================================================
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-22229}"

NNODES=1
NPROC_PER_NODE=8

# ============================================================
# Project and environment
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/vol1/liangyy_workspace/dggt}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/vol1/liangyy_workspace/waymo_processed_dggt}"
CONDA_SH="${CONDA_SH:-/mnt/vol1/liangyy_workspace/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dggt}"

# ============================================================
# Data and model paths. These match pretrain_two_nodes.sh.
# ============================================================
WAYMO_DGGT_ROOT="${DATASET_ROOT}/training"
WAYMO_DGGT_VAL_ROOT="${DATASET_ROOT}/validation"
DGGT_CKPT="${PROJECT_ROOT}/pretrained/model_latest_waymo.pt"
TOKENIZER_CKPT="${PROJECT_ROOT}/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt"
FEATURE_STATS="${PROJECT_ROOT}/logs/scene_flow_pretrain_1024/feature_stats_pretrain_v3_798.pt"
SCENE_CAPTION_ROOT="${DATASET_ROOT}/training_captions"
SCENE_CAPTION_VAL_ROOT="${DATASET_ROOT}/validation_captions"
QWEN_TEXT_ENCODER="${QWEN_TEXT_ENCODER:-/mnt/vol1/liangyy_workspace/model/Qwen/Qwen3-0.6B}"

LOG_DIR="${PROJECT_ROOT}/logs/scene_flow_pretrain_1024"
LAUNCH_LOG_DIR="${PROJECT_ROOT}/logs/single_node_launch"

# ============================================================
# Training config. Aligned with pretrain_two_nodes.sh except:
#   - NNODES=1
#   - NPROC_PER_NODE=8
#   - BATCH_SIZE_PER_GPU=8
# This keeps global batch size at 64.
# ============================================================
BATCH_SIZE_PER_GPU=8
GRAD_ACCUM_STEPS=1
NUM_WORKERS=4
PREFETCH_FACTOR=2

UNCOND_DROP_PROB=0.1
TEXT_UNCOND_DROP_PROB=0.1
ASSET_UNCOND_DROP_PROB=0.2
CAMERA_UNCOND_DROP_PROB=0.2
ALL_COND_DROP_PROB=0.05

GUIDANCE_SCALE=1.0
ASSET_CONTROL_GUIDANCE_SCALE=1.0
CAMERA_GUIDANCE_SCALE=1.0
VAL_GUIDANCE_SCALES="1.0,2.0,4.0"

WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_1node_8gpu_b8_gb64_lr1e4_optcond}"
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

    # Single-node training does not need InfiniBand bootstrap constraints.
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
}

check_dir() {
    local name="$1"
    local path="$2"

    if [[ ! -d "${path}" ]]; then
        echo "Missing directory ${name}: ${path}" >&2
        exit 1
    fi

    echo "OK directory ${name}: ${path}"
}

check_file() {
    local name="$1"
    local path="$2"

    if [[ ! -f "${path}" ]]; then
        echo "Missing file ${name}: ${path}" >&2
        exit 1
    fi

    echo "OK file ${name}: ${path}"
}

check_python_and_gpu() {
    EXPECTED_GPU_COUNT="${NPROC_PER_NODE}" python - <<'PY'
import os
import sys
import torch

expected = int(os.environ["EXPECTED_GPU_COUNT"])
count = torch.cuda.device_count()

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version seen by PyTorch: {torch.version.cuda}")
print(f"visible GPU count: {count}")

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot use CUDA")
if count != expected:
    raise RuntimeError(f"Expected {expected} GPUs, got {count}")
PY

    python -m torch.distributed.run --help >/dev/null
}

build_train_args() {
    TRAIN_ARGS=(
        train_scene_flow_pretrain.py
        --image_dir "${WAYMO_DGGT_ROOT}"
        --val_image_dir "${WAYMO_DGGT_VAL_ROOT}"
        --dggt_ckpt_path "${DGGT_CKPT}"
        --tokenizer_ckpt_path "${TOKENIZER_CKPT}"
        --feature_stats_path "${FEATURE_STATS}"
        --log_dir "${LOG_DIR}"
        --caption_root "${SCENE_CAPTION_ROOT}"
        --val_caption_root "${SCENE_CAPTION_VAL_ROOT}"
        --text_encoder_path "${QWEN_TEXT_ENCODER}"
        --scene_start 0
        --scene_end 800
        --sequence_length 10
        --val_sliding_window 10
        --val_sliding_stride 7
        --camera_anchor_context_dropout 0.25
        --patch_grid_h 25
        --patch_grid_w 37
        --latent_dim 1024
        --batch_size "${BATCH_SIZE_PER_GPU}"
        --grad_accum_steps "${GRAD_ACCUM_STEPS}"
        --num_workers "${NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
        --lr 1e-4
        --final_lr 1e-5
        --weight_decay 0.0
        --optimizer_type gmuon
        --ema_decay 0.9995
        --warmup_steps 4000
        --max_steps 200000
        --save_every 2000
        --shift 10.0
        --weighting_scheme waver
        --mode_scale 1.29
        --loss_weighting_scheme none
        --prediction_type x
        --lambda_repa 0.5
        --base_model_coeff 0.25
        --lambda_boundary 0.25
        --lambda_camera_flow 0.1
        --lambda_camera_pose 0.5
        --lambda_sky_flow 0.05
        --uncond_drop_prob "${UNCOND_DROP_PROB}"
        --text_uncond_drop_prob "${TEXT_UNCOND_DROP_PROB}"
        --asset_uncond_drop_prob "${ASSET_UNCOND_DROP_PROB}"
        --camera_uncond_drop_prob "${CAMERA_UNCOND_DROP_PROB}"
        --all_cond_drop_prob "${ALL_COND_DROP_PROB}"
        --guidance_scale "${GUIDANCE_SCALE}"
        --asset_control_guidance_scale "${ASSET_CONTROL_GUIDANCE_SCALE}"
        --camera_guidance_scale "${CAMERA_GUIDANCE_SCALE}"
        --val_guidance_scales "${VAL_GUIDANCE_SCALES}"
        --val_scene_start 0
        --val_scene_end 100
        --val_every 1000
        --val_batches 1
        --val_log_images 10
        --val_sample_steps 35
        --grad_clip_norm 1.0
        --seed 0
        --precision bf16
        --ddp_timeout_minutes 60
        --wandb
        --wandb_project dggt-flow
        --wandb_name "${WANDB_NAME}"
    )
}

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
setup_common_env

mkdir -p "${LOG_DIR}" "${LAUNCH_LOG_DIR}"

echo "Checking local files..."
check_dir "PROJECT_ROOT" "${PROJECT_ROOT}"
check_file "train_scene_flow_pretrain.py" "${PROJECT_ROOT}/train_scene_flow_pretrain.py"
check_file "CONDA_SH" "${CONDA_SH}"
check_dir "WAYMO_DGGT_ROOT" "${WAYMO_DGGT_ROOT}"
check_dir "WAYMO_DGGT_VAL_ROOT" "${WAYMO_DGGT_VAL_ROOT}"
check_file "DGGT_CKPT" "${DGGT_CKPT}"
check_file "TOKENIZER_CKPT" "${TOKENIZER_CKPT}"
check_file "FEATURE_STATS" "${FEATURE_STATS}"
check_dir "SCENE_CAPTION_ROOT" "${SCENE_CAPTION_ROOT}"
check_dir "SCENE_CAPTION_VAL_ROOT" "${SCENE_CAPTION_VAL_ROOT}"
check_dir "QWEN_TEXT_ENCODER" "${QWEN_TEXT_ENCODER}"
check_python_and_gpu

cd "${PROJECT_ROOT}"
build_train_args

echo "=== Starting single-node 8-GPU pretraining ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "global batch size: ${GLOBAL_BATCH_SIZE} = ${NNODES} node x ${NPROC_PER_NODE} gpu/node x ${BATCH_SIZE_PER_GPU} batch/gpu x ${GRAD_ACCUM_STEPS} accum"
echo "training log dir: ${LOG_DIR}"
echo "launch log: ${LAUNCH_LOG_DIR}/single_node.log"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LAUNCH_LOG_DIR}/single_node.log"
