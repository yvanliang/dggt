#!/usr/bin/env bash
export WANDB_API_KEY="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"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

umask 000

usage() {
    cat <<'EOF'
用法：
  bash pretrain_half_node_p6000_islo_ablation.sh

Image-Space Layout Only（A3/ISLO）单机半节点训练：
  - 单机 4 张 RTX PRO 6000，每 GPU batch 1，gradient accumulation 16，global batch 64；
  - 从 step 0 开始全新训练，禁止从 Full、FSU 或 LOS checkpoint 分叉；
  - early map/actor raster stem 固定 on/on；
  - late map/actor metric reader 与 appearance context 固定 off/off/off；
  - 除四卡拓扑和 accumulation=16 外，训练、验证和 W&B 参数对齐
    pretrain_single_node_islo_ablation.sh。
EOF
}

if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    echo "[错误] A3/ISLO 四卡 launcher 不接受位置参数。" >&2
    usage >&2
    exit 2
fi

# ============================================================
# Single-node half-node topology: 1 machine x 4 RTX PRO 6000 GPUs.
# ============================================================
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-22229}"

NNODES=1
NPROC_PER_NODE=4

# ============================================================
# Project and environment
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/home/dancer/code/dm/dggt}"
DATASET_ROOT="${DATASET_ROOT:-/data/disk2/lyy_dataset/waymo_processed_dggt/}"
CONDA_SH="${CONDA_SH:-/home/dancer/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dggt}"

# ============================================================
# Data and model paths. Keep the local P6000 machine paths.
# ============================================================
WAYMO_DGGT_ROOT="${WAYMO_DGGT_ROOT:-${DATASET_ROOT}/training}"
WAYMO_DGGT_VAL_ROOT="${WAYMO_DGGT_VAL_ROOT:-${DATASET_ROOT}/validation}"
HDMAP_ROOT="${HDMAP_ROOT:-${DATASET_ROOT}/training_hdmap}"
VAL_HDMAP_ROOT="${VAL_HDMAP_ROOT:-${DATASET_ROOT}/validation_hdmap}"
DGGT_CKPT="${DGGT_CKPT:-/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${PROJECT_ROOT}/logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT_ROOT}/logs/scene_flow_pretrain_1024/feature_stats_pretrain_v5.pt}"
SCENE_GAUGE_PATH="${SCENE_GAUGE_PATH:-${PROJECT_ROOT}/data/scene_gauge/training.json}"
VAL_SCENE_GAUGE_PATH="${VAL_SCENE_GAUGE_PATH:-${PROJECT_ROOT}/data/scene_gauge/validation.json}"
PULLBACK_CALIBRATION_PATH="${PULLBACK_CALIBRATION_PATH:-${PROJECT_ROOT}/data/scene_gauge/pullback_d63b34f7.json}"
SCENE_CAPTION_ROOT="${SCENE_CAPTION_ROOT:-${DATASET_ROOT}/training_captions}"
SCENE_CAPTION_VAL_ROOT="${SCENE_CAPTION_VAL_ROOT:-${DATASET_ROOT}/validation_captions}"
QWEN_TEXT_ENCODER="${QWEN_TEXT_ENCODER:-/home/dancer/model/Qwen/Qwen3-0.6B}"

# A3 changes only the early/late layout injection architecture switches.
SCENE_UNITS_PROFILE="generated"
WORLD_FEEDBACK_PROFILE="full"
WORLD_FEEDBACK_EFFECTIVE="1/1/1"
LAYOUT_MAP_INJECTION=1
LAYOUT_ACTOR_INJECTION=1
LAYOUT_MAP_METRIC_INJECTION=0
LAYOUT_ACTOR_METRIC_INJECTION=0
APPEARANCE_CONTEXT_INJECTION=0
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/scene_flow_pretrain_islo_v6}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/half_node_p6000_islo_v6_launch}"

# ============================================================
# Training contract: 1 node x 4 GPUs x 1 sample/GPU x 16 accumulation = 64.
# Apart from the four-GPU topology and accumulation, values match
# pretrain_single_node_islo_ablation.sh.
# ============================================================
BATCH_SIZE_PER_GPU=1
GRAD_ACCUM_STEPS=16
EXPECTED_GLOBAL_BATCH_SIZE=64
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
DATALOADER_WORKER_THREADS="${DATALOADER_WORKER_THREADS:-1}"
DATALOADER_OUT_OF_ORDER="${DATALOADER_OUT_OF_ORDER:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

MAX_STEPS="${MAX_STEPS:-200000}"
DECAY_END_STEPS="${DECAY_END_STEPS:-0}"
SAVE_EVERY="${SAVE_EVERY:-2500}"
VAL_EVERY="${VAL_EVERY:-2000}"
VAL_BATCHES="${VAL_BATCHES:-8}"
VAL_LOG_IMAGES="${VAL_LOG_IMAGES:-10}"
VAL_INFERENCE_SCENES="${VAL_INFERENCE_SCENES:-10}"
VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-50}"
LOG_EVERY="${LOG_EVERY:-1}"

CFG_SCALE="${CFG_SCALE:-1.0}"
LAYOUT_GUIDANCE_SCALE="${LAYOUT_GUIDANCE_SCALE:-1.0}"
ASSET_CONTROL_GUIDANCE_SCALE="${ASSET_CONTROL_GUIDANCE_SCALE:-1.0}"
VAL_GUIDANCE_SCALES="${VAL_GUIDANCE_SCALES:-1.0,2.0,4.0}"
LAYOUT_MAX_ACTORS="${LAYOUT_MAX_ACTORS:-96}"
STATIC_FAR_PLANE_M="${STATIC_FAR_PLANE_M:-120}"

WANDB_PROJECT="${WANDB_PROJECT:-dggt-flow}"
WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_lr1e4_islo_v6}"

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

if [[ -n "${RESUME_PATH:-}" ]]; then
    echo "[错误] A3/ISLO 必须从 step 0 训练，拒绝 RESUME_PATH=${RESUME_PATH}" >&2
    exit 1
fi
unset RESUME_PATH RESUME_EXPECTED_STEP WANDB_RUN_ID
WANDB_RESUME="never"

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

for integer_value in "${MASTER_PORT}" "${NNODES}" "${NPROC_PER_NODE}" \
    "${BATCH_SIZE_PER_GPU}" "${GRAD_ACCUM_STEPS}" \
    "${EXPECTED_GLOBAL_BATCH_SIZE}"; do
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
        echo "[错误] 四卡拓扑与 batch 配置必须是非负整数，当前值：${integer_value}" >&2
        exit 1
    fi
done
for value_name in NUM_WORKERS PREFETCH_FACTOR VAL_NUM_WORKERS \
    DATALOADER_WORKER_THREADS MAX_STEPS DECAY_END_STEPS SAVE_EVERY VAL_EVERY \
    VAL_BATCHES VAL_LOG_IMAGES VAL_INFERENCE_SCENES VAL_SAMPLE_STEPS LOG_EVERY; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[错误] ${value_name} 必须是非负整数，当前值：${value}" >&2
        exit 1
    fi
done
if [[ "${GRAD_ACCUM_STEPS}" != "16" ]]; then
    echo "[错误] A3 四卡实验固定 GRAD_ACCUM_STEPS=16。" >&2
    exit 1
fi
if [[ "${DATALOADER_OUT_OF_ORDER}" != "0" && "${DATALOADER_OUT_OF_ORDER}" != "1" ]]; then
    echo "[错误] DATALOADER_OUT_OF_ORDER 必须是 0 或 1。" >&2
    exit 1
fi
case "${GRADIENT_CHECKPOINTING}" in
    0|1|half|three_quarter) ;;
    *)
        echo "[错误] GRADIENT_CHECKPOINTING 必须是 0、half、three_quarter 或 1。" >&2
        exit 1
        ;;
esac
if (( MAX_STEPS <= 0 || SAVE_EVERY <= 0 || VAL_INFERENCE_SCENES <= 0 || LOG_EVERY <= 0 )); then
    echo "[错误] MAX_STEPS、SAVE_EVERY、VAL_INFERENCE_SCENES 和 LOG_EVERY 必须大于 0。" >&2
    exit 1
fi
if (( PREFETCH_FACTOR <= 0 || DATALOADER_WORKER_THREADS <= 0 )); then
    echo "[错误] PREFETCH_FACTOR 和 DATALOADER_WORKER_THREADS 必须大于 0。" >&2
    exit 1
fi

GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
if (( GLOBAL_BATCH_SIZE != EXPECTED_GLOBAL_BATCH_SIZE )); then
    echo "[错误] global batch size 配置不符合 A3 四卡合同：" >&2
    echo "       ${GLOBAL_BATCH_SIZE} = ${NNODES} node × ${NPROC_PER_NODE} gpu/node × ${BATCH_SIZE_PER_GPU} batch/gpu × ${GRAD_ACCUM_STEPS} accum" >&2
    echo "       期望 ${EXPECTED_GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi

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
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export NCCL_ALGO="${NCCL_ALGO:-Ring}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    unset WANDB_DISABLED || true
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
        --hdmap_root "${HDMAP_ROOT}"
        --val_hdmap_root "${VAL_HDMAP_ROOT}"
        --dggt_ckpt_path "${DGGT_CKPT}"
        --tokenizer_ckpt_path "${TOKENIZER_CKPT}"
        --feature_stats_path "${FEATURE_STATS}"
        --scene_gauge_path "${SCENE_GAUGE_PATH}"
        --scene_units_profile "${SCENE_UNITS_PROFILE}"
        --val_scene_gauge_path "${VAL_SCENE_GAUGE_PATH}"
        --pullback_calibration_path "${PULLBACK_CALIBRATION_PATH}"
        --log_dir "${LOG_DIR}"
        --caption_root "${SCENE_CAPTION_ROOT}"
        --val_caption_root "${SCENE_CAPTION_VAL_ROOT}"
        --text_encoder_path "${QWEN_TEXT_ENCODER}"
        --scene_start 0
        --scene_end 800
        --sequence_length 10
        --val_sliding_window 10
        --val_sliding_stride 7
        --patch_grid_h 25
        --patch_grid_w 37
        --latent_dim 1024
        --batch_size "${BATCH_SIZE_PER_GPU}"
        --grad_accum_steps "${GRAD_ACCUM_STEPS}"
        --num_workers "${NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
        --val_num_workers "${VAL_NUM_WORKERS}"
        --dataloader_worker_threads "${DATALOADER_WORKER_THREADS}"
        --pin_memory
        --lr 1e-4
        --final_lr 1e-5
        --weight_decay 0.0
        --optimizer_type gmuon
        --ema_decay 0.9995
        --warmup_steps 4000
        --max_steps "${MAX_STEPS}"
        --decay_end_steps "${DECAY_END_STEPS}"
        --save_every "${SAVE_EVERY}"
        --shift 10.0
        --weighting_scheme waver
        --mode_scale 1.29
        --loss_weighting_scheme none
        --prediction_type x
        --lambda_repa 0.5
        --base_model_coeff 0.25
        --lambda_boundary 0.25
        --lambda_sky_flow 0.5
        --world_feedback_profile "${WORLD_FEEDBACK_PROFILE}"
        --cfg "${CFG_SCALE}"
        --layout_guidance_scale "${LAYOUT_GUIDANCE_SCALE}"
        --asset_control_guidance_scale "${ASSET_CONTROL_GUIDANCE_SCALE}"
        --layout_max_actors "${LAYOUT_MAX_ACTORS}"
        --static_far_plane_m "${STATIC_FAR_PLANE_M}"
        --val_guidance_scales "${VAL_GUIDANCE_SCALES}"
        --val_scene_start 0
        --val_scene_end 100
        --val_every "${VAL_EVERY}"
        --val_batches "${VAL_BATCHES}"
        --val_log_images "${VAL_LOG_IMAGES}"
        --val_inference_scenes "${VAL_INFERENCE_SCENES}"
        --val_sample_steps "${VAL_SAMPLE_STEPS}"
        --grad_clip_norm 1.0
        --seed 0
        --precision bf16
        --ddp_timeout_minutes 60
        --force_tqdm
        --log_every "${LOG_EVERY}"
        --wandb
        --wandb_project "${WANDB_PROJECT}"
        --wandb_name "${WANDB_NAME}"
        --wandb_resume "${WANDB_RESUME}"
    )
    TRAIN_ARGS+=(
        --layout_map_injection
        --layout_actor_injection
        --no-layout_map_metric_injection
        --no-layout_actor_metric_injection
        --no-appearance_context_injection
    )
    if [[ "${DATALOADER_OUT_OF_ORDER}" == "1" ]]; then
        TRAIN_ARGS+=(--dataloader_out_of_order)
    fi
    case "${GRADIENT_CHECKPOINTING}" in
        1) TRAIN_ARGS+=(--gradient_checkpointing) ;;
        three_quarter) TRAIN_ARGS+=(--three_quarter_gradient_checkpointing) ;;
        half) TRAIN_ARGS+=(--half_gradient_checkpointing) ;;
        0) TRAIN_ARGS+=(--no_gradient_checkpointing) ;;
    esac
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
check_dir "HDMAP_ROOT" "${HDMAP_ROOT}"
check_dir "VAL_HDMAP_ROOT" "${VAL_HDMAP_ROOT}"
check_file "DGGT_CKPT" "${DGGT_CKPT}"
check_file "TOKENIZER_CKPT" "${TOKENIZER_CKPT}"
check_file "FEATURE_STATS" "${FEATURE_STATS}"
check_file "SCENE_GAUGE_PATH" "${SCENE_GAUGE_PATH}"
check_file "VAL_SCENE_GAUGE_PATH" "${VAL_SCENE_GAUGE_PATH}"
check_file "PULLBACK_CALIBRATION_PATH" "${PULLBACK_CALIBRATION_PATH}"
check_dir "SCENE_CAPTION_ROOT" "${SCENE_CAPTION_ROOT}"
check_dir "SCENE_CAPTION_VAL_ROOT" "${SCENE_CAPTION_VAL_ROOT}"
check_dir "QWEN_TEXT_ENCODER" "${QWEN_TEXT_ENCODER}"
check_python_and_gpu

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[警告] 未检测到 WANDB_API_KEY；仅当环境中已有 W&B 登录凭证时才能上传。" >&2
fi

cd "${PROJECT_ROOT}"
build_train_args
LAUNCH_LOG="${LAUNCH_LOG_DIR}/half_node_p6000_islo.log"

echo "============================================================"
echo "A3 Image-Space Layout Only 单机 4-GPU RTX PRO 6000 预训练"
echo "master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NCCL: P2P_DISABLE=${NCCL_P2P_DISABLE}, SHM_DISABLE=${NCCL_SHM_DISABLE}, SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, ALGO=${NCCL_ALGO}"
echo "global batch size: ${GLOBAL_BATCH_SIZE} = ${NNODES} node × ${NPROC_PER_NODE} gpu/node × ${BATCH_SIZE_PER_GPU} batch/gpu × ${GRAD_ACCUM_STEPS} accum"
echo "dataloader: workers/rank=${NUM_WORKERS}, prefetch/worker=${PREFETCH_FACTOR}, validation_workers=${VAL_NUM_WORKERS}, worker_threads=${DATALOADER_WORKER_THREADS}, out_of_order=${DATALOADER_OUT_OF_ORDER}"
echo "gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
echo "scene units profile: ${SCENE_UNITS_PROFILE}"
echo "world feedback: profile=${WORLD_FEEDBACK_PROFILE}, raw=1/1/1, effective=${WORLD_FEEDBACK_EFFECTIVE}, compute_matched=true"
echo "layout injection: early M/G=${LAYOUT_MAP_INJECTION}/${LAYOUT_ACTOR_INJECTION}, late M/G/A=${LAYOUT_MAP_METRIC_INJECTION}/${LAYOUT_ACTOR_METRIC_INJECTION}/${APPEARANCE_CONTEXT_INJECTION}"
echo "training start: fresh step-0 architecture ablation"
echo "training steps: max=${MAX_STEPS}, lr_decay_end=${DECAY_END_STEPS}, save_every=${SAVE_EVERY}"
echo "validation: every=${VAL_EVERY}, batches=${VAL_BATCHES}, inference_scenes=${VAL_INFERENCE_SCENES}, log_images=${VAL_LOG_IMAGES}, sample_steps=${VAL_SAMPLE_STEPS}"
echo "training log dir: ${LOG_DIR}"
echo "launch log: ${LAUNCH_LOG}"
echo "wandb: ${WANDB_PROJECT}/${WANDB_NAME} (resume=${WANDB_RESUME})"
echo "============================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LAUNCH_LOG}"
