#!/usr/bin/env bash
export WANDB_API_KEY="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"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

umask 000

# ============================================================
# 阿里 PPU 单节点双卡预训练脚本
#
# 使用方式：
#   bash pretrain_ppu.sh
#
# 说明：
#   - 不执行 conda activate，直接使用当前 shell/container 的 python；
#   - 默认单节点 2 张 PPU；
#   - PPU PyTorch 仍通过 torch.cuda 接口暴露设备，按阿里 PPU 文档使用
#     CUDA_VISIBLE_DEVICES 控制可见设备；
#   - 不调用 nvidia-smi / nvcc / IB HCA 等 NVIDIA 设备检查命令；
#   - 暂不启用 wandb 上传。
# ============================================================

# ============================================================
# 分布式配置：1 台机器 x 2 PPU
# ============================================================
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-22229}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# PPU/MIG 设备选择也使用 CUDA_VISIBLE_DEVICES。需要指定 MIG UUID 时：
#   CUDA_VISIBLE_DEVICES=MIG-xxxx bash pretrain_ppu.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ============================================================
# 项目与 Python 环境
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/workspace/datasets/waymo_processed_dggt}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/model}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# torchvision/lpips 通过 TORCH_HOME 查找 AlexNet backbone。将缓存放到
# 共享 model 目录后，所有 DLC 节点都会直接读取同一份本地权重。
TORCH_HOME="${TORCH_HOME:-${MODEL_ROOT}/torch}"
ALEXNET_CKPT="${TORCH_HOME}/hub/checkpoints/alexnet-owt-7be5be79.pth"

# ============================================================
# 数据与模型路径
# ============================================================
WAYMO_DGGT_ROOT="${WAYMO_DGGT_ROOT:-${DATASET_ROOT}/training}"
WAYMO_DGGT_VAL_ROOT="${WAYMO_DGGT_VAL_ROOT:-${DATASET_ROOT}/validation}"
DEFAULT_DGGT_CKPT="${PROJECT_ROOT}/pretrained/model_latest_waymo.pt"
if [[ ! -f "${DEFAULT_DGGT_CKPT}" && -f /data/lyy_dataset/model/dggt/model_latest_waymo.pt ]]; then
  DEFAULT_DGGT_CKPT=/data/lyy_dataset/model/dggt/model_latest_waymo.pt
elif [[ ! -f "${DEFAULT_DGGT_CKPT}" && -f /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt ]]; then
  DEFAULT_DGGT_CKPT=/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt
fi
DGGT_CKPT="${DGGT_CKPT:-${DEFAULT_DGGT_CKPT}}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${PROJECT_ROOT}/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT_ROOT}/logs/scene_flow_pretrain_1024/feature_stats_pretrain_v4.pt}"
SCENE_GAUGE_PATH="${SCENE_GAUGE_PATH:-${PROJECT_ROOT}/data/scene_gauge/training.json}"
VAL_SCENE_GAUGE_PATH="${VAL_SCENE_GAUGE_PATH:-${PROJECT_ROOT}/data/scene_gauge/validation.json}"
PULLBACK_CALIBRATION_PATH="${PULLBACK_CALIBRATION_PATH:-${PROJECT_ROOT}/data/scene_gauge/pullback_75e566ef.json}"
SCENE_CAPTION_ROOT="${SCENE_CAPTION_ROOT:-${DATASET_ROOT}/training_captions}"
SCENE_CAPTION_VAL_ROOT="${SCENE_CAPTION_VAL_ROOT:-${DATASET_ROOT}/validation_captions}"
QWEN_TEXT_ENCODER="${QWEN_TEXT_ENCODER:-${MODEL_ROOT}/Qwen/Qwen3-0.6B}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/scene_flow_pretrain_1024_v4}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/ppu_launch}"

# ============================================================
# 训练配置
# 当前全局 batch = NNODES x NPROC_PER_NODE x BATCH_SIZE_PER_PPU x GRAD_ACCUM_STEPS。
# 默认全局 batch = 1 x 2 x 1 x 4 = 8。
# ============================================================
BATCH_SIZE_PER_PPU="${BATCH_SIZE_PER_PPU:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

UNCOND_DROP_PROB="${UNCOND_DROP_PROB:-0.1}"
TEXT_UNCOND_DROP_PROB="${TEXT_UNCOND_DROP_PROB:-0.1}"
ASSET_UNCOND_DROP_PROB="${ASSET_UNCOND_DROP_PROB:-0.2}"
CAMERA_UNCOND_DROP_PROB="${CAMERA_UNCOND_DROP_PROB:-0.2}"
ALL_COND_DROP_PROB="${ALL_COND_DROP_PROB:-0.05}"

GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
ASSET_CONTROL_GUIDANCE_SCALE="${ASSET_CONTROL_GUIDANCE_SCALE:-1.0}"
CAMERA_GUIDANCE_SCALE="${CAMERA_GUIDANCE_SCALE:-1.0}"
VAL_GUIDANCE_SCALES="${VAL_GUIDANCE_SCALES:-1.0,2.0,4.0}"

# Keep production defaults unchanged while allowing the real-training PPU
# smoke launcher to shorten the run without maintaining a second copy of the
# training argument list.
MAX_STEPS="${MAX_STEPS:-200000}"
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
VAL_EVERY="${VAL_EVERY:-1000}"
VAL_BATCHES="${VAL_BATCHES:-1}"
VAL_LOG_IMAGES="${VAL_LOG_IMAGES:-10}"
VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-35}"
RESUME_PATH="${RESUME_PATH:-}"
WARM_START_PATH="${WARM_START_PATH:-}"

GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_PPU * GRAD_ACCUM_STEPS))

setup_common_env() {
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES
    export TORCH_HOME

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

    # PPU 官方训练镜像建议 1/2/4/8 卡任务使用 eth0；整机 16 卡可由调用方覆盖为 hpn0。
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    # 不继承 PAI 镜像可能预置的 NCCL_DEBUG=INFO，避免打印逐连接日志。
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

    # 保留 PyTorch 默认 SDPA 调度。仅将 LearnedQueryPool 按 batch 分块，
    # 避开 ZW810E 对大 flattened batch 的 fused MHA kernel 限制。
    export DGGT_DEVICE_BACKEND="${DGGT_DEVICE_BACKEND:-ppu}"
    export DGGT_PPU_MHA_BATCH_CHUNK_SIZE="${DGGT_PPU_MHA_BATCH_CHUNK_SIZE:-4096}"

    # 保险起见，即使误传了 --wandb 之外的初始化路径，也不上传。
    export WANDB_DISABLED="${WANDB_DISABLED:-true}"
}

check_dir() {
    local label="$1"
    local path="$2"

    if [[ ! -d "${path}" ]]; then
        echo "[错误] 目录不存在：${label}" >&2
        echo "       ${path}" >&2
        exit 1
    fi
    echo "[OK] ${label}: ${path}"
}

check_file() {
    local label="$1"
    local path="$2"

    if [[ ! -f "${path}" ]]; then
        echo "[错误] 文件不存在：${label}" >&2
        echo "       ${path}" >&2
        exit 1
    fi
    echo "[OK] ${label}: ${path}"
}

check_python_and_ppu() {
    EXPECTED_DEVICE_COUNT="${NPROC_PER_NODE}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

expected = int(os.environ["EXPECTED_DEVICE_COUNT"])
count = torch.cuda.device_count()

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available through PPU torch: {torch.cuda.is_available()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
print(f"visible device count: {count}")
for idx in range(count):
    try:
        print(f"device {idx}: {torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"device {idx}: <name unavailable: {exc}>")

if not torch.cuda.is_available():
    raise RuntimeError("当前 PyTorch 无法使用 PPU/CUDA 兼容设备")
if count < expected:
    raise RuntimeError(f"期望至少 {expected} 个可见设备，实际检测到 {count} 个")
PY

    "${PYTHON_BIN}" -m torch.distributed.run --help >/dev/null
}

check_required_paths() {
    echo "检查本机文件和目录..."

    check_dir "PROJECT_ROOT" "${PROJECT_ROOT}"
    check_file "train_scene_flow_pretrain.py" "${PROJECT_ROOT}/train_scene_flow_pretrain.py"
    check_dir "WAYMO_DGGT_ROOT" "${WAYMO_DGGT_ROOT}"
    check_dir "WAYMO_DGGT_VAL_ROOT" "${WAYMO_DGGT_VAL_ROOT}"
    check_file "DGGT_CKPT" "${DGGT_CKPT}"
    check_file "TOKENIZER_CKPT" "${TOKENIZER_CKPT}"
    check_file "FEATURE_STATS" "${FEATURE_STATS}"
    check_file "SCENE_GAUGE_PATH" "${SCENE_GAUGE_PATH}"
    check_file "VAL_SCENE_GAUGE_PATH" "${VAL_SCENE_GAUGE_PATH}"
    check_file "PULLBACK_CALIBRATION_PATH" "${PULLBACK_CALIBRATION_PATH}"
    if [[ -n "${RESUME_PATH}" ]]; then
        check_file "RESUME_PATH" "${RESUME_PATH}"
    fi
    if [[ -n "${WARM_START_PATH}" ]]; then
        check_file "WARM_START_PATH" "${WARM_START_PATH}"
    fi
    if [[ -n "${RESUME_PATH}" && -n "${WARM_START_PATH}" ]]; then
        echo "[错误] RESUME_PATH 和 WARM_START_PATH 不能同时设置。" >&2
        exit 1
    fi
    check_dir "SCENE_CAPTION_ROOT" "${SCENE_CAPTION_ROOT}"
    check_dir "SCENE_CAPTION_VAL_ROOT" "${SCENE_CAPTION_VAL_ROOT}"
    check_dir "QWEN_TEXT_ENCODER" "${QWEN_TEXT_ENCODER}"
    if [[ ! -f "${ALEXNET_CKPT}" ]]; then
        echo "[错误] 缺少 LPIPS AlexNet 本地权重：" >&2
        echo "       ${ALEXNET_CKPT}" >&2
        echo "       请先执行：bash ${PROJECT_ROOT}/download_ppu_model_weights.sh" >&2
        exit 1
    fi
    echo "[OK] ALEXNET_CKPT: ${ALEXNET_CKPT}"

    mkdir -p "${LOG_DIR}" "${LAUNCH_LOG_DIR}"
    echo "[OK] LOG_DIR: ${LOG_DIR}"
    echo "[OK] LAUNCH_LOG_DIR: ${LAUNCH_LOG_DIR}"
}

build_train_args() {
    TRAIN_ARGS=(
        train_scene_flow_pretrain.py
        --image_dir "${WAYMO_DGGT_ROOT}"
        --val_image_dir "${WAYMO_DGGT_VAL_ROOT}"
        --dggt_ckpt_path "${DGGT_CKPT}"
        --tokenizer_ckpt_path "${TOKENIZER_CKPT}"
        --feature_stats_path "${FEATURE_STATS}"
        --scene_gauge_path "${SCENE_GAUGE_PATH}"
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
        --camera_anchor_context_dropout 0.25
        --patch_grid_h 25
        --patch_grid_w 37
        --latent_dim 1024
        --batch_size "${BATCH_SIZE_PER_PPU}"
        --grad_accum_steps "${GRAD_ACCUM_STEPS}"
        --num_workers "${NUM_WORKERS}"
        --prefetch_factor "${PREFETCH_FACTOR}"
        --pin_memory
        --lr 1e-4
        --final_lr 1e-5
        --weight_decay 0.0
        --optimizer_type gmuon
        --ema_decay 0.9995
        --warmup_steps "${WARMUP_STEPS}"
        --max_steps "${MAX_STEPS}"
        --save_every "${SAVE_EVERY}"
        --shift 10.0
        --weighting_scheme waver
        --mode_scale 1.29
        --loss_weighting_scheme none
        --prediction_type x
        --lambda_repa 0.5
        --base_model_coeff 0.25
        --lambda_boundary 0.25
        --lambda_camera_flow 0.1
        --lambda_camera_pose 1.0
        --lambda_sky_flow 0.1
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
        --val_every "${VAL_EVERY}"
        --val_batches "${VAL_BATCHES}"
        --val_log_images "${VAL_LOG_IMAGES}"
        --val_sample_steps "${VAL_SAMPLE_STEPS}"
        --grad_clip_norm 1.0
        --seed 0
        --precision bf16
        --ddp_timeout_minutes 60
    )
    if [[ -n "${RESUME_PATH}" ]]; then
        TRAIN_ARGS+=(--resume_path "${RESUME_PATH}")
    elif [[ -n "${WARM_START_PATH}" ]]; then
        TRAIN_ARGS+=(--warm_start_path "${WARM_START_PATH}")
    fi
}

setup_common_env
check_required_paths
check_python_and_ppu

cd "${PROJECT_ROOT}"
build_train_args

echo "=== Starting PPU pretraining ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "DGGT_DEVICE_BACKEND=${DGGT_DEVICE_BACKEND}"
echo "DGGT_PPU_MHA_BATCH_CHUNK_SIZE=${DGGT_PPU_MHA_BATCH_CHUNK_SIZE}"
echo "global batch size: ${GLOBAL_BATCH_SIZE} = ${NNODES} node x ${NPROC_PER_NODE} ppu/node x ${BATCH_SIZE_PER_PPU} batch/ppu x ${GRAD_ACCUM_STEPS} accum"
echo "training log dir: ${LOG_DIR}"
echo "launch log: ${LAUNCH_LOG_DIR}/ppu_2card.log"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LAUNCH_LOG_DIR}/ppu_2card.log"
