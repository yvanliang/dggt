#!/usr/bin/env bash
export WANDB_API_KEY="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"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

# ============================================================
# 使用方式
#   在主节点 A 上执行：
#     bash pretrain_two_nodes.sh
#
# 脚本会：
#   1. 检查主节点 A 的文件、Python、PyTorch、GPU、网卡和 IB；
#   2. 通过 SSH 检查副节点 B；
#   3. 先在 B 上启动 node_rank=1；
#   4. 再在 A 上启动 node_rank=0。
#
# 注意：
#   - 两台机器上的项目、数据、环境和模型路径统一通过 /home/wuzn/liangyy 访问；
#   - 不执行 conda activate；
#   - 始终使用 CONDA_ENV/bin/python，避免激活到其他同名环境。
# ============================================================

MODE="${1:---launch}"

# ============================================================
# 分布式配置：2 台机器 × 每台 8 GPU
# ============================================================
MASTER_ADDR="10.199.7.31"
MASTER_PORT="22229"

NNODES=2
NPROC_PER_NODE=8

MASTER_RANK=0
WORKER_RANK=1

WORKER_HOST="10.199.7.30"
WORKER_USER="wuzn"
SSH_PORT=2288

# ============================================================
# 项目与 Python 环境
# ============================================================
LIANGYY_ROOT="${LIANGYY_ROOT:-/home/wuzn/liangyy}"
PROJECT_ROOT="${PROJECT_ROOT:-${LIANGYY_ROOT}/dggt}"
DATASET_ROOT="${DATASET_ROOT:-${LIANGYY_ROOT}/waymo_processed_dggt}"

CONDA_ROOT="${CONDA_ROOT:-/home/wuzn/miniconda3}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ROOT}/envs/dggt}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"

# 远端 SSH 始终执行 canonical 项目路径下的脚本。这样即使主节点从
# /mnt/vol1/liangyy_workspace/... 启动，也不会把 /mnt/vol1 前缀传给副节点。
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/pretrain_two_nodes.sh}"

# ============================================================
# 数据与模型路径
# 要求两台服务器上这些路径均存在
# ============================================================
WAYMO_DGGT_ROOT="${DATASET_ROOT}/training"
WAYMO_DGGT_VAL_ROOT="${DATASET_ROOT}/validation"
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
SCENE_CAPTION_ROOT="${DATASET_ROOT}/training_captions"
SCENE_CAPTION_VAL_ROOT="${DATASET_ROOT}/validation_captions"
QWEN_TEXT_ENCODER="${QWEN_TEXT_ENCODER:-${LIANGYY_ROOT}/model/Qwen/Qwen3-0.6B}"

# 双机训练使用独立目录，避免覆盖单机训练的 checkpoint、验证结果和状态文件。
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/scene_flow_pretrain_1024_v4}"
LAUNCH_LOG_DIR="${PROJECT_ROOT}/logs/distributed_launch"

# 以下变量在原脚本中定义，但当前训练命令没有使用。
# 如果 train_scene_flow_pretrain.py 确实需要它们，请按实际参数名加入 TRAIN_ARGS。
SCENE_FLOW_PRETRAIN_CKPT="${PROJECT_ROOT}/logs/scene_flow_pretrain_1024/ckpt/pretrain_step100000.pt"
SCENE_FLOW_TRAIN_MANIFEST="${DATASET_ROOT}/waymo_edit_cache/manifests/training/training_manifest.jsonl"
SCENE_FLOW_VAL_MANIFEST="${DATASET_ROOT}/waymo_edit_cache/manifests/validation/validation_manifest.jsonl"

# ============================================================
# 训练配置
# 当前全局 batch = NNODES x NPROC_PER_NODE x BATCH_SIZE_PER_GPU x GRAD_ACCUM_STEPS。
# 修改 BATCH_SIZE_PER_GPU 或 GRAD_ACCUM_STEPS 后，以启动时打印的 GLOBAL_BATCH_SIZE 为准。
# ============================================================
BATCH_SIZE_PER_GPU=1
GRAD_ACCUM_STEPS=4
NUM_WORKERS=8
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

WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_lr1e4_v4}"
GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))

# ============================================================
# NCCL 网络配置
# ============================================================

# 用于 torchrun rendezvous、Gloo 和 NCCL bootstrap。
# 默认优先 bond4；如果当前节点没有该网卡，启动时会按到对端 IP 的路由自动选择。
SOCKET_IFNAME="${SOCKET_IFNAME:-bond4}"

# 大数据通信使用两条 200G HDR InfiniBand。
# 两台机器上的 HCA 名称都必须为 mlx5_4、mlx5_5。
NCCL_IB_HCA_VALUE="=mlx5_4:1,mlx5_5:1"

# ============================================================
# SSH 公共参数
# ============================================================
SSH_OPTS=(
    -p "${SSH_PORT}"
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=no
)

# ============================================================
# 清理旧 Conda 状态并构造确定的运行环境
# ============================================================
detect_socket_ifname() {
    local requested="$1"
    local target
    local dev
    local candidate

    if [[ -n "${requested}" && -d "/sys/class/net/${requested}" ]]; then
        echo "${requested}"
        return 0
    fi

    if [[ -n "${requested}" ]]; then
        echo "[警告] 当前节点不存在网卡 ${requested}，改为自动检测 socket 网卡。" >&2
    fi

    # 优先使用系统路由表选择到对端 IP 的网卡。主节点到 WORKER_HOST，
    # 副节点到 MASTER_ADDR；如果某个目标是本机导致 dev=lo，则跳过。
    for target in "${MASTER_ADDR}" "${WORKER_HOST}"; do
        dev="$(ip -o -4 route get "${target}" 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="dev") {print $(i+1); exit}}' || true)"
        if [[ -n "${dev}" && "${dev}" != "lo" && -d "/sys/class/net/${dev}" ]]; then
            echo "${dev}"
            return 0
        fi
    done

    # 路由检测失败时，按常见物理/聚合网卡名兜底；不选 lo/docker/veth。
    for candidate in bond4 bond0 bond1 ib0 eth0 eno1 eno2 ens1 ens2 ens3 ens4 enp1s0 enp2s0 enp3s0 enp4s0; do
        if [[ -d "/sys/class/net/${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "[错误] 无法自动检测可用于 NCCL/Gloo bootstrap 的 socket 网卡。" >&2
    echo "       当前网卡列表：" >&2
    ip -br link >&2 || true
    return 1
}

setup_common_env() {
    # 清除调用脚本的 Shell 中残留的另一套 Conda 状态。
    unset CONDARC || true
    unset CONDA_PREFIX || true
    unset CONDA_PREFIX_1 || true
    unset CONDA_DEFAULT_ENV || true
    unset CONDA_PROMPT_MODIFIER || true
    unset CONDA_SHLVL || true
    unset CONDA_EXE || true
    unset CONDA_PYTHON_EXE || true
    unset _CE_CONDA || true
    unset _CE_M || true
    unset PYTHONHOME || true

    # 让当前环境中的命令优先，但训练启动仍使用 PYTHON_BIN 的绝对路径。
    export CONDA_PREFIX="${CONDA_ENV}"
    export CONDA_DEFAULT_ENV="${CONDA_ENV}"

    # 使用干净 PATH，避免旧的 ai_lab/anaconda3 环境残留在前面。
    export PATH="${CONDA_ENV}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    # 避免不同服务器各自的 ~/.local Python 包混入共享环境。
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="${PROJECT_ROOT}"
    export PYTHONUNBUFFERED=1

    # 优先使用当前环境和系统 CUDA 库。
    export LD_LIBRARY_PATH="${CONDA_ENV}/lib:/usr/local/cuda/lib64"

    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

    # 避免每个训练进程占用过多 CPU 线程。
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    # Hugging Face 只读取本地模型。
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    # torchrun/Gloo/NCCL bootstrap 使用当前节点实际存在的 socket 网卡。
    RUNTIME_SOCKET_IFNAME="$(detect_socket_ifname "${SOCKET_IFNAME}")"
    export RUNTIME_SOCKET_IFNAME
    export GLOO_SOCKET_IFNAME="${RUNTIME_SOCKET_IFNAME}"
    export NCCL_SOCKET_IFNAME="=${RUNTIME_SOCKET_IFNAME}"
    export NCCL_SOCKET_FAMILY="AF_INET"

    # 跨节点张量通信使用双 HDR IB。
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA="${NCCL_IB_HCA_VALUE}"

    # 首次启动用于确认 NCCL 是否走 IB；稳定后可改为 WARN。
    export NCCL_DEBUG=WARN  # export NCCL_DEBUG=INFO
    unset NCCL_DEBUG_SUBSYS || true  # export NCCL_DEBUG_SUBSYS="INIT,NET"

    # 异步 NCCL 错误尽快终止，而不是无限挂住。
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
}

# ============================================================
# 文件与设备检查
# ============================================================
check_dir() {
    local label="$1"
    local path="$2"

    if [[ ! -d "${path}" ]]; then
        echo "[错误] 目录不存在：${label}" >&2
        echo "       ${path}" >&2
        return 1
    fi
    echo "[OK] ${label}: ${path}"
}

check_file() {
    local label="$1"
    local path="$2"

    if [[ ! -f "${path}" ]]; then
        echo "[错误] 文件不存在：${label}" >&2
        echo "       ${path}" >&2
        return 1
    fi
    echo "[OK] ${label}: ${path}"
}

check_executable() {
    local label="$1"
    local path="$2"

    if [[ ! -x "${path}" ]]; then
        echo "[错误] 文件不存在或不可执行：${label}" >&2
        echo "       ${path}" >&2
        return 1
    fi
    echo "[OK] ${label}: ${path}"
}

check_required_paths() {
    local node_name="$1"

    echo "[${node_name}] 检查共享文件和目录……"

    check_dir "PROJECT_ROOT" "${PROJECT_ROOT}"
    check_file "train_scene_flow_pretrain.py" "${PROJECT_ROOT}/train_scene_flow_pretrain.py"
    check_executable "PYTHON_BIN" "${PYTHON_BIN}"

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

    echo "[OK] RUNTIME_SOCKET_IFNAME: ${RUNTIME_SOCKET_IFNAME:-未设置}"
    check_dir "IB HCA mlx5_4" /sys/class/infiniband/mlx5_4
    check_dir "IB HCA mlx5_5" /sys/class/infiniband/mlx5_5

    if ! mkdir -p "${LOG_DIR}" "${LAUNCH_LOG_DIR}"; then
        echo "[错误] 创建日志目录失败：" >&2
        echo "       LOG_DIR=${LOG_DIR}" >&2
        echo "       LAUNCH_LOG_DIR=${LAUNCH_LOG_DIR}" >&2
        return 1
    fi
    echo "[OK] LOG_DIR: ${LOG_DIR}"
    echo "[OK] LAUNCH_LOG_DIR: ${LAUNCH_LOG_DIR}"
}

check_python_and_gpu() {
    local node_name="$1"

    echo "[${node_name}] 检查 Python、PyTorch 和 GPU……"

    EXPECTED_GPU_COUNT="${NPROC_PER_NODE}" NODE_NAME="${node_name}" \
        "${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

node = os.environ["NODE_NAME"]
expected = int(os.environ["EXPECTED_GPU_COUNT"])
count = torch.cuda.device_count()

print(f"[{node}] Python: {sys.executable}")
print(f"[{node}] PyTorch: {torch.__version__}")
print(f"[{node}] CUDA available: {torch.cuda.is_available()}")
print(f"[{node}] CUDA version seen by PyTorch: {torch.version.cuda}")
print(f"[{node}] visible GPU count: {count}")

if not torch.cuda.is_available():
    raise RuntimeError(f"[{node}] PyTorch 无法使用 CUDA")
if count != expected:
    raise RuntimeError(f"[{node}] 期望 {expected} 张 GPU，实际检测到 {count} 张")
PY

    # 确认该 Python 中存在 torch.distributed.run。
    "${PYTHON_BIN}" -m torch.distributed.run --help >/dev/null
}

preflight_check() {
    local node_name="$1"

    setup_common_env
    check_required_paths "${node_name}"
    check_python_and_gpu "${node_name}"

    echo "[${node_name}] 检查通过。"
}

# ============================================================
# 训练参数：主副节点共用同一份，避免两边参数不一致
# ============================================================
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
        --batch_size "${BATCH_SIZE_PER_GPU}"
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

# ============================================================
# 在当前节点启动一个 torchrun 节点
# ============================================================
launch_node() {
    local node_rank="$1"
    local node_name="$2"
    local log_file="$3"
    local pid_file="$4"

    preflight_check "${node_name}"
    build_train_args

    cd "${PROJECT_ROOT}"

    # 防止重复启动同一节点。
    if [[ -f "${pid_file}" ]]; then
        local old_pid
        old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
            echo "[${node_name}] 已有训练进程运行，PID=${old_pid}" >&2
            echo "[${node_name}] PID 文件：${pid_file}" >&2
            exit 1
        fi
    fi

    echo "[${node_name}] 启动 torchrun，node_rank=${node_rank}……"
    echo "[${node_name}] Python: ${PYTHON_BIN}"
    echo "[${node_name}] 日志: ${log_file}"

    nohup "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes="${NNODES}" \
        --node_rank="${node_rank}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${TRAIN_ARGS[@]}" \
        > "${log_file}" 2>&1 < /dev/null &

    local launch_pid=$!
    echo "${launch_pid}" > "${pid_file}"

    # 检查是否因参数、导入或环境错误而立即退出。
    sleep 3
    if ! kill -0 "${launch_pid}" 2>/dev/null; then
        echo "[${node_name}] torchrun 启动后立即退出。最近日志如下：" >&2
        tail -n 100 "${log_file}" >&2 || true
        exit 1
    fi

    echo "[${node_name}] 已启动，PID=${launch_pid}"
}

stop_node() {
    local node_name="$1"
    local pid_file="$2"

    if [[ ! -f "${pid_file}" ]]; then
        echo "[${node_name}] 不存在 PID 文件：${pid_file}"
        return 0
    fi

    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"

    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
        echo "[${node_name}] PID 文件内容无效：${pid_file}" >&2
        return 1
    fi

    if kill -0 "${pid}" 2>/dev/null; then
        echo "[${node_name}] 终止 PID=${pid}"
        kill "${pid}"
    else
        echo "[${node_name}] PID=${pid} 已不在运行。"
    fi

    rm -f "${pid_file}"
}

# ============================================================
# 日志与 PID 文件
# ============================================================
MASTER_LOG="${LAUNCH_LOG_DIR}/master_rank0.log"
WORKER_LOG="${LAUNCH_LOG_DIR}/worker_rank1.log"
MASTER_PID_FILE="${LAUNCH_LOG_DIR}/master_rank0.pid"
WORKER_PID_FILE="${LAUNCH_LOG_DIR}/worker_rank1.pid"

# ============================================================
# 运行模式
# ============================================================
case "${MODE}" in
    --check-worker)
        preflight_check "副节点 ${WORKER_HOST}"
        ;;

    --worker)
        launch_node \
            "${WORKER_RANK}" \
            "副节点 ${WORKER_HOST}" \
            "${WORKER_LOG}" \
            "${WORKER_PID_FILE}"
        ;;

    --master)
        launch_node \
            "${MASTER_RANK}" \
            "主节点 ${MASTER_ADDR}" \
            "${MASTER_LOG}" \
            "${MASTER_PID_FILE}"
        ;;

    --stop-worker)
        stop_node "副节点 ${WORKER_HOST}" "${WORKER_PID_FILE}"
        ;;

    --stop-master)
        stop_node "主节点 ${MASTER_ADDR}" "${MASTER_PID_FILE}"
        ;;

    --launch)
        echo "============================================================"
        echo "2 节点 × 8 GPU 分布式训练启动器"
        echo "脚本路径：${SCRIPT_PATH}"
        echo "主节点：${MASTER_ADDR}, rank=${MASTER_RANK}"
        echo "副节点：${WORKER_USER}@${WORKER_HOST}:${SSH_PORT}, rank=${WORKER_RANK}"
        echo "Python 环境：${CONDA_ENV}"
        echo "全局 batch size：${GLOBAL_BATCH_SIZE}"
        echo "============================================================"

        echo
        echo "=== 1/4 检查主节点 ==="
        preflight_check "主节点 ${MASTER_ADDR}"

        echo
        echo "=== 2/4 检查副节点 SSH 和运行环境 ==="
        ssh "${SSH_OPTS[@]}" \
            "${WORKER_USER}@${WORKER_HOST}" \
            "test -r '${SCRIPT_PATH}' && bash '${SCRIPT_PATH}' --check-worker"

        echo
        echo "=== 3/4 启动副节点 rank ${WORKER_RANK} ==="
        ssh "${SSH_OPTS[@]}" \
            "${WORKER_USER}@${WORKER_HOST}" \
            "bash '${SCRIPT_PATH}' --worker"

        echo
        echo "=== 4/4 启动主节点 rank ${MASTER_RANK} ==="
        if ! launch_node \
            "${MASTER_RANK}" \
            "主节点 ${MASTER_ADDR}" \
            "${MASTER_LOG}" \
            "${MASTER_PID_FILE}"; then

            echo "主节点启动失败，尝试终止副节点进程……" >&2
            ssh "${SSH_OPTS[@]}" \
                "${WORKER_USER}@${WORKER_HOST}" \
                "bash '${SCRIPT_PATH}' --stop-worker" || true
            exit 1
        fi

        echo
        echo "============================================================"
        echo "2 节点 16 GPU 训练已启动"
        echo "主节点日志：${MASTER_LOG}"
        echo "副节点日志：${WORKER_LOG}"
        echo "主节点 PID 文件：${MASTER_PID_FILE}"
        echo "副节点 PID 文件：${WORKER_PID_FILE}"
        echo "全局 batch size：${GLOBAL_BATCH_SIZE} = ${NNODES} nodes × ${NPROC_PER_NODE} gpu/node × ${BATCH_SIZE_PER_GPU} batch/gpu × ${GRAD_ACCUM_STEPS} accum"
        echo
        echo "查看日志："
        echo "  tail -f '${MASTER_LOG}'"
        echo "  tail -f '${WORKER_LOG}'"
        echo "============================================================"
        ;;

    *)
        echo "未知参数：${MODE}" >&2
        echo "可用参数：" >&2
        echo "  --launch        从主节点启动两机训练（默认）" >&2
        echo "  --check-worker  只检查当前节点" >&2
        echo "  --worker        在当前节点启动 worker rank" >&2
        echo "  --master        在当前节点启动 master rank" >&2
        echo "  --stop-worker   停止当前节点的 worker" >&2
        echo "  --stop-master   停止当前节点的 master" >&2
        exit 2
        ;;
esac
