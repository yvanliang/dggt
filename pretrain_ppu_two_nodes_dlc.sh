#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

umask 000

# ============================================================
# 阿里云 PAI-DLC：2 节点 × 每节点 16 张真武 810E PPU
#
# DLC 控制台配置：
#   - 节点数量：2
#   - 每节点规格：ml.gp7vf.16.40xlarge（16 PPU）
#   - 启动命令：
#       bash /mnt/workspace/dggt/pretrain_ppu_two_nodes_dlc.sh
#
# DLC 会在每个节点分别执行同一条启动命令，并自动注入：
#   MASTER_ADDR、MASTER_PORT、WORLD_SIZE、RANK、NPROC_PER_NODE
#
# 注意：
#   - WORLD_SIZE/RANK 在脚本入口表示 DLC 的节点数和节点 rank；
#   - torchrun 启动后，会为训练子进程重新设置进程级 WORLD_SIZE/RANK；
#   - 不需要 SSH，也不要手工分别启动 master/worker；
#   - PPU PyTorch 通过 torch.cuda 兼容接口暴露设备；
#   - WANDB_API_KEY 请通过 DLC 自定义环境变量注入，不要写入脚本。
# ============================================================

require_dlc_env() {
    local name="$1"

    if [[ -z "${!name:-}" ]]; then
        echo "[错误] 缺少 PAI-DLC 自动注入的环境变量：${name}" >&2
        echo "       请用 PAI-DLC PyTorch 分布式任务启动此脚本。" >&2
        exit 1
    fi
}

for required_name in MASTER_ADDR MASTER_PORT WORLD_SIZE RANK NPROC_PER_NODE; do
    require_dlc_env "${required_name}"
done

# 先保存 DLC 的节点级变量。torchrun 会在训练进程中重新定义同名的
# WORLD_SIZE/RANK，届时它们分别表示总训练进程数和进程全局 rank。
DLC_MASTER_ADDR="${MASTER_ADDR}"
DLC_MASTER_PORT="${MASTER_PORT}"
DLC_NNODES="${WORLD_SIZE}"
DLC_NODE_RANK="${RANK}"
DLC_NPROC_PER_NODE="${NPROC_PER_NODE}"

EXPECTED_NNODES="${EXPECTED_NNODES:-2}"
EXPECTED_NPROC_PER_NODE="${EXPECTED_NPROC_PER_NODE:-16}"

for integer_value in \
    "${DLC_MASTER_PORT}" \
    "${DLC_NNODES}" \
    "${DLC_NODE_RANK}" \
    "${DLC_NPROC_PER_NODE}" \
    "${EXPECTED_NNODES}" \
    "${EXPECTED_NPROC_PER_NODE}"; do
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
        echo "[错误] DLC 分布式配置应为非负整数，实际收到：${integer_value}" >&2
        exit 1
    fi
done

if (( DLC_NNODES != EXPECTED_NNODES )); then
    echo "[错误] 该脚本要求 ${EXPECTED_NNODES} 个 DLC 节点，实际 WORLD_SIZE=${DLC_NNODES}。" >&2
    exit 1
fi
if (( DLC_NPROC_PER_NODE != EXPECTED_NPROC_PER_NODE )); then
    echo "[错误] 该脚本要求每节点 ${EXPECTED_NPROC_PER_NODE} 张 PPU，实际 NPROC_PER_NODE=${DLC_NPROC_PER_NODE}。" >&2
    exit 1
fi
if (( DLC_NODE_RANK >= DLC_NNODES )); then
    echo "[错误] DLC 节点 RANK=${DLC_NODE_RANK} 超出 [0, $((DLC_NNODES - 1))]。" >&2
    exit 1
fi

# PPU/MIG 设备选择仍使用 CUDA_VISIBLE_DEVICES。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

# ============================================================
# 项目与 Python 环境
# 沿用已验证可运行的 pretrain_ppu.sh 默认路径。
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
# 数据、模型与日志路径
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
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/ppu_dlc_launch}"

# ============================================================
# 训练配置
# global batch = 节点数 × 每节点 PPU 数 × 每 PPU batch × 梯度累积。
# 默认：2 × 16 × 1 × 2 = 64。
# ============================================================
BATCH_SIZE_PER_PPU="${BATCH_SIZE_PER_PPU:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EXPECTED_GLOBAL_BATCH_SIZE="${EXPECTED_GLOBAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

TEXT_UNCOND_DROP_PROB="${TEXT_UNCOND_DROP_PROB:-0.1}"
JOINT_GENERATION_PROB="${JOINT_GENERATION_PROB:-0.2}"
CAMERA_CONTROLLED_PROB="${CAMERA_CONTROLLED_PROB:-0.2}"
ASSET_CAMERA_CONTROLLED_PROB="${ASSET_CAMERA_CONTROLLED_PROB:-0.6}"

GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
ASSET_CONTROL_GUIDANCE_SCALE="${ASSET_CONTROL_GUIDANCE_SCALE:-1.0}"
CAMERA_GUIDANCE_SCALE="${CAMERA_GUIDANCE_SCALE:-1.0}"
VAL_GUIDANCE_SCALES="${VAL_GUIDANCE_SCALES:-1.0,2.0,4.0}"

# metric-gauge v4 使用独立的新 wandb run，不续接旧 v3 run。
WANDB_PROJECT="${WANDB_PROJECT:-dggt-flow}"
WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_lr1e4_v4}"
WANDB_RESUME="never"

GLOBAL_BATCH_SIZE=$((DLC_NNODES * DLC_NPROC_PER_NODE * BATCH_SIZE_PER_PPU * GRAD_ACCUM_STEPS))

if (( GLOBAL_BATCH_SIZE != EXPECTED_GLOBAL_BATCH_SIZE )); then
    echo "[错误] global batch size 配置不符合预期：" >&2
    echo "       ${GLOBAL_BATCH_SIZE} = ${DLC_NNODES} nodes × ${DLC_NPROC_PER_NODE} ppu/node × ${BATCH_SIZE_PER_PPU} batch/ppu × ${GRAD_ACCUM_STEPS} accum" >&2
    echo "       期望 EXPECTED_GLOBAL_BATCH_SIZE=${EXPECTED_GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi

setup_common_env() {
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES
    export TORCH_HOME

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

    # PAI-PPU ACCL-P 官方最佳实践为 eth0、NCCL_IB_HCA=""、
    # NCCL_IB_DISABLE=1。PAI-PPU 官方镜像已内置这些值，因此优先保留
    # 镜像/平台注入值，仅在变量缺失时补齐官方默认值。
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    # PAI 官方镜像可能预置 NCCL_DEBUG=INFO；训练时固定为 WARN，
    # 只保留警告和错误，不打印逐连接 INFO 日志。
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

    # 保留 PAI-DLC/PPU 官方镜像预置的其余高性能网络配置，不在这里
    # 覆盖其他 NCCL_*、PCCL_* 或 ACCL-P 相关变量。

    # 保留 PyTorch 默认 SDPA 调度。仅将 LearnedQueryPool 按 batch 分块，
    # 避开真武 810E 对大 flattened batch 的 fused MHA kernel 限制。
    export DGGT_DEVICE_BACKEND="${DGGT_DEVICE_BACKEND:-ppu}"
    export DGGT_PPU_MHA_BATCH_CHUNK_SIZE="${DGGT_PPU_MHA_BATCH_CHUNK_SIZE:-4096}"

    # 本任务明确启用 wandb；凭证由 DLC 自定义环境变量或镜像凭证提供。
    unset WANDB_DISABLED || true
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

check_required_paths() {
    echo "[DLC rank ${DLC_NODE_RANK}] 检查本节点文件和目录……"

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

check_python_and_ppu() {
    EXPECTED_DEVICE_COUNT="${DLC_NPROC_PER_NODE}" DLC_NODE_RANK="${DLC_NODE_RANK}" \
        "${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

node_rank = int(os.environ["DLC_NODE_RANK"])
expected = int(os.environ["EXPECTED_DEVICE_COUNT"])
count = torch.cuda.device_count()

print(f"[DLC rank {node_rank}] Python: {sys.executable}")
print(f"[DLC rank {node_rank}] PyTorch: {torch.__version__}")
print(f"[DLC rank {node_rank}] CUDA available through PPU torch: {torch.cuda.is_available()}")
print(f"[DLC rank {node_rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
print(f"[DLC rank {node_rank}] visible device count: {count}")
for idx in range(count):
    try:
        print(f"[DLC rank {node_rank}] device {idx}: {torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"[DLC rank {node_rank}] device {idx}: <name unavailable: {exc}>")

if not torch.cuda.is_available():
    raise RuntimeError(f"[DLC rank {node_rank}] 当前 PyTorch 无法使用 PPU/CUDA 兼容设备")
if count != expected:
    raise RuntimeError(
        f"[DLC rank {node_rank}] 期望 {expected} 个可见 PPU，实际检测到 {count} 个"
    )
PY

    "${PYTHON_BIN}" -m torch.distributed.run --help >/dev/null
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
        --lambda_camera_pose 0.25
        --camera_pose_start_step 0
        --camera_pose_warmup_steps 10000
        --camera_absolute_translation_scale_m 10.0
        --camera_relative_translation_scale_m 1.0
        --camera_acceleration_translation_scale_m 1.0
        --lambda_sky_flow 0.1
        --text_uncond_drop_prob "${TEXT_UNCOND_DROP_PROB}"
        --joint_generation_prob "${JOINT_GENERATION_PROB}"
        --camera_controlled_prob "${CAMERA_CONTROLLED_PROB}"
        --asset_camera_controlled_prob "${ASSET_CAMERA_CONTROLLED_PROB}"
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
        --wandb_project "${WANDB_PROJECT}"
        --wandb_name "${WANDB_NAME}"
        --wandb_resume "${WANDB_RESUME}"
    )
}

setup_common_env
check_required_paths
check_python_and_ppu

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[警告] 未检测到 WANDB_API_KEY；仅当镜像中已有 wandb 登录凭证时才能正常上传。" >&2
    echo "       建议在 DLC 任务的自定义环境变量中配置 WANDB_API_KEY。" >&2
fi

cd "${PROJECT_ROOT}"
build_train_args

LAUNCH_LOG="${LAUNCH_LOG_DIR}/ppu_dlc_rank${DLC_NODE_RANK}.log"

echo "============================================================"
echo "PAI-DLC ${DLC_NNODES} 节点 PPU 预训练"
echo "DLC node rank: ${DLC_NODE_RANK}/${DLC_NNODES}"
echo "master: ${DLC_MASTER_ADDR}:${DLC_MASTER_PORT}"
echo "PPU per node: ${DLC_NPROC_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "NCCL_IB_HCA: ${NCCL_IB_HCA}"
echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
echo "DGGT_DEVICE_BACKEND: ${DGGT_DEVICE_BACKEND}"
echo "global batch size: ${GLOBAL_BATCH_SIZE} = ${DLC_NNODES} nodes × ${DLC_NPROC_PER_NODE} ppu/node × ${BATCH_SIZE_PER_PPU} batch/ppu × ${GRAD_ACCUM_STEPS} accum"
echo "training log dir: ${LOG_DIR}"
echo "launch log: ${LAUNCH_LOG}"
echo "wandb: ${WANDB_PROJECT}/${WANDB_NAME}, new run (resume=${WANDB_RESUME})"
echo "============================================================"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${DLC_NNODES}" \
    --node_rank="${DLC_NODE_RANK}" \
    --nproc_per_node="${DLC_NPROC_PER_NODE}" \
    --master_addr="${DLC_MASTER_ADDR}" \
    --master_port="${DLC_MASTER_PORT}" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LAUNCH_LOG}"
