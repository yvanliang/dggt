#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

# ============================================================
# 使用方式
#   在主节点 A 上执行：
#     bash pretrain_two_nodes26.sh
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
MASTER_ADDR="10.199.7.26"
MASTER_PORT="22229"

NNODES=2
NPROC_PER_NODE=8

MASTER_RANK=0
WORKER_RANK=1

WORKER_HOST="10.199.7.25"
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
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/pretrain_two_nodes26.sh}"

# ============================================================
# 数据与模型路径
# 要求两台服务器上这些路径均存在
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
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${PROJECT_ROOT}/logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT_ROOT}/logs/scene_flow_pretrain_1024/feature_stats_pretrain_v5.pt}"
SCENE_GAUGE_PATH="${SCENE_GAUGE_PATH:-${PROJECT_ROOT}/data/scene_gauge/training.json}"
VAL_SCENE_GAUGE_PATH="${VAL_SCENE_GAUGE_PATH:-${PROJECT_ROOT}/data/scene_gauge/validation.json}"
PULLBACK_CALIBRATION_PATH="${PULLBACK_CALIBRATION_PATH:-${PROJECT_ROOT}/data/scene_gauge/pullback_d63b34f7.json}"
SCENE_CAPTION_ROOT="${SCENE_CAPTION_ROOT:-${DATASET_ROOT}/training_captions}"
SCENE_CAPTION_VAL_ROOT="${SCENE_CAPTION_VAL_ROOT:-${DATASET_ROOT}/validation_captions}"
QWEN_TEXT_ENCODER="${QWEN_TEXT_ENCODER:-${LIANGYY_ROOT}/model/Qwen/Qwen3-0.6B}"

# 双机训练使用独立目录，避免覆盖单机训练的 checkpoint、验证结果和状态文件。
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/scene_flow_pretrain_waymo_gb64_lr1e4_v5}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/distributed_launch}"
RESUME_PATH="${RESUME_PATH:-}"
RESUME_EXPECTED_STEP="${RESUME_EXPECTED_STEP:--1}"

# ============================================================
# 训练配置
# 当前全局 batch = NNODES x NPROC_PER_NODE x BATCH_SIZE_PER_GPU x GRAD_ACCUM_STEPS。
# 修改 BATCH_SIZE_PER_GPU 或 GRAD_ACCUM_STEPS 后，以启动时打印的 GLOBAL_BATCH_SIZE 为准。
# ============================================================
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
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

MAX_STEPS="${MAX_STEPS:-200000}"
DECAY_END_STEPS="${DECAY_END_STEPS:-0}"
SAVE_EVERY="${SAVE_EVERY:-2500}"
VAL_EVERY="${VAL_EVERY:-1000}"
VAL_BATCHES="${VAL_BATCHES:-1}"
VAL_LOG_IMAGES="${VAL_LOG_IMAGES:-10}"
VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-35}"
for value_name in \
    BATCH_SIZE_PER_GPU GRAD_ACCUM_STEPS NUM_WORKERS PREFETCH_FACTOR \
    MAX_STEPS DECAY_END_STEPS SAVE_EVERY VAL_EVERY VAL_BATCHES \
    VAL_LOG_IMAGES VAL_SAMPLE_STEPS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[错误] ${value_name} 必须是非负整数，当前值：${value}" >&2
        exit 1
    fi
done
if [[ ! "${RESUME_EXPECTED_STEP}" =~ ^-?[0-9]+$ ]]; then
    echo "[错误] RESUME_EXPECTED_STEP 必须是整数，当前值：${RESUME_EXPECTED_STEP}" >&2
    exit 1
fi
if (( BATCH_SIZE_PER_GPU <= 0 || GRAD_ACCUM_STEPS <= 0 || MAX_STEPS <= 0 || SAVE_EVERY <= 0 )); then
    echo "[错误] batch、梯度累积、MAX_STEPS 和 SAVE_EVERY 必须大于 0。" >&2
    exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-dggt-flow}"
WANDB_NAME="${WANDB_NAME:-scene_flow_pretrain_waymo_gb64_lr1e4_v5}"
WANDB_RESUME="${WANDB_RESUME:-never}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
if [[ -z "${WANDB_RUN_ID}" ]]; then
    # Keep the shell-side empty sentinel for CLI construction, but do not let
    # W&B see an exported empty WANDB_RUN_ID.
    export -n WANDB_RUN_ID 2>/dev/null || true
fi
GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))

# ============================================================
# NCCL 网络配置
# ============================================================

# 用于 torchrun rendezvous、Gloo 和 NCCL bootstrap。
# 26/25 的管理/启动网络走 bond0；如果当前节点没有该网卡，启动时会按到对端 IP 的路由自动选择。
SOCKET_IFNAME="${SOCKET_IFNAME:-bond0}"

# 大数据通信优先使用两条 200G HDR InfiniBand。
# 如果两台机器上的 HDR 都不可用，则尝试 mlx5_bond_0 的 25G RDMA/RoCE。
NCCL_HDR_IB_HCA_VALUE="=mlx5_4:1,mlx5_5:1"
NCCL_BOND_RDMA_HCA_VALUE="=mlx5_bond_0:1"
BOND_RDMA_MIN_RATE_GBPS=25

# auto:      两端均可用时依次选择 mlx5_4/5 HDR、mlx5_bond_0 RDMA，最后退到 bond0 socket。
# ib:        强制要求本节点 mlx5_4/5 可用，否则检查失败。
# bond_rdma: 强制要求本节点 mlx5_bond_0 达到 ACTIVE/LinkUp/至少 25G，否则检查失败。
# socket:    强制禁用 RDMA，只走 SOCKET_IFNAME。
NETWORK_MODE="${NETWORK_MODE:-auto}"

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

# SSH 不会自动继承主节点的 shell 环境。所有会改变训练语义、输出位置或
# W&B run 的变量都必须显式传给 worker，否则两个 rank 可能启动成不同实验。
remote_train_env() {
    local name
    local quoted
    local names=(
        LIANGYY_ROOT PROJECT_ROOT DATASET_ROOT CONDA_ROOT CONDA_ENV PYTHON_BIN SCRIPT_PATH
        WAYMO_DGGT_ROOT WAYMO_DGGT_VAL_ROOT DGGT_CKPT TOKENIZER_CKPT FEATURE_STATS
        SCENE_GAUGE_PATH VAL_SCENE_GAUGE_PATH PULLBACK_CALIBRATION_PATH
        SCENE_CAPTION_ROOT SCENE_CAPTION_VAL_ROOT QWEN_TEXT_ENCODER
        LOG_DIR LAUNCH_LOG_DIR RESUME_PATH RESUME_EXPECTED_STEP
        BATCH_SIZE_PER_GPU GRAD_ACCUM_STEPS NUM_WORKERS PREFETCH_FACTOR
        TEXT_UNCOND_DROP_PROB JOINT_GENERATION_PROB CAMERA_CONTROLLED_PROB
        ASSET_CAMERA_CONTROLLED_PROB GUIDANCE_SCALE ASSET_CONTROL_GUIDANCE_SCALE
        CAMERA_GUIDANCE_SCALE VAL_GUIDANCE_SCALES
        MAX_STEPS DECAY_END_STEPS SAVE_EVERY
        VAL_EVERY VAL_BATCHES VAL_LOG_IMAGES VAL_SAMPLE_STEPS
        WANDB_PROJECT WANDB_NAME WANDB_RESUME SOCKET_IFNAME NETWORK_MODE
    )
    for name in "${names[@]}"; do
        printf -v quoted '%q' "${!name}"
        printf '%s=%s ' "${name}" "${quoted}"
    done
    if [[ -n "${WANDB_RUN_ID:-}" ]]; then
        printf -v quoted '%q' "${WANDB_RUN_ID}"
        printf 'WANDB_RUN_ID=%s ' "${quoted}"
    fi
}

run_remote_script() {
    local action="$1"
    local remote_env
    local script_quoted
    local action_quoted
    local remote_command

    remote_env="$(remote_train_env)"
    printf -v script_quoted '%q' "${SCRIPT_PATH}"
    printf -v action_quoted '%q' "${action}"
    remote_command="test -r ${script_quoted} && env ${remote_env} bash ${script_quoted} ${action_quoted}"

    # 若主节点通过环境变量提供 W&B key，只在 stdin 中传给真正启动训练的
    # worker，避免把凭证写进仓库、远端命令行或日志。也可让两个节点事先
    # `wandb login`，此时无需设置 WANDB_API_KEY。
    if [[ "${action}" == "--worker" && -n "${WANDB_API_KEY:-}" ]]; then
        printf '%s\n' "${WANDB_API_KEY}" | \
            ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
                "IFS= read -r WANDB_API_KEY; export WANDB_API_KEY; ${remote_command}"
    else
        ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" "${remote_command}"
    fi
}

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

rdma_hca_port_ready() {
    local hca="$1"
    local port="${2:-1}"
    local min_rate_gbps="${3:-1}"
    local base="/sys/class/infiniband/${hca}/ports/${port}"
    local state
    local phys
    local rate
    local rate_gbps

    [[ -d "${base}" ]] || return 1

    state="$(cat "${base}/state" 2>/dev/null || true)"
    phys="$(cat "${base}/phys_state" 2>/dev/null || true)"
    rate="$(cat "${base}/rate" 2>/dev/null || true)"

    [[ "${state}" == *ACTIVE* ]] || return 1
    [[ "${phys}" == *LinkUp* || "${phys}" == *LINK_UP* ]] || return 1
    if [[ ! "${rate}" =~ ([0-9]+)[[:space:]]*Gb/sec ]]; then
        return 1
    fi
    rate_gbps="${BASH_REMATCH[1]}"
    (( rate_gbps >= min_rate_gbps ))
}

hdr_ib_ready() {
    rdma_hca_port_ready mlx5_4 1 200 && rdma_hca_port_ready mlx5_5 1 200
}

bond_rdma_ready() {
    rdma_hca_port_ready mlx5_bond_0 1 "${BOND_RDMA_MIN_RATE_GBPS}"
}

print_ib_status() {
    local hca
    local port
    local base

    for hca in mlx5_4 mlx5_5 mlx5_bond_0; do
        if [[ ! -d "/sys/class/infiniband/${hca}" ]]; then
            echo "[WARN] IB HCA ${hca}: missing"
            continue
        fi

        for base in "/sys/class/infiniband/${hca}/ports/"*; do
            [[ -d "${base}" ]] || continue
            port="${base##*/}"
            echo "[INFO] ${hca}:${port} state=$(cat "${base}/state" 2>/dev/null || echo NA) phys=$(cat "${base}/phys_state" 2>/dev/null || echo NA) rate=$(cat "${base}/rate" 2>/dev/null || echo NA) layer=$(cat "${base}/link_layer" 2>/dev/null || echo NA)"
        done
    done
}

resolve_network_mode() {
    case "${NETWORK_MODE}" in
        ib)
            if ! hdr_ib_ready; then
                echo "[错误] NETWORK_MODE=ib，但本节点 mlx5_4:1 和 mlx5_5:1 未同时达到 ACTIVE/LinkUp/200G。" >&2
                print_ib_status >&2
                return 1
            fi
            echo "ib"
            ;;
        bond_rdma)
            if ! bond_rdma_ready; then
                echo "[错误] NETWORK_MODE=bond_rdma，但本节点 mlx5_bond_0:1 未达到 ACTIVE/LinkUp/至少 ${BOND_RDMA_MIN_RATE_GBPS}G。" >&2
                print_ib_status >&2
                return 1
            fi
            echo "bond_rdma"
            ;;
        socket)
            echo "socket"
            ;;
        auto)
            if hdr_ib_ready; then
                echo "ib"
            elif bond_rdma_ready; then
                echo "bond_rdma"
            else
                echo "socket"
            fi
            ;;
        *)
            echo "[错误] NETWORK_MODE 只能是 auto、ib、bond_rdma 或 socket，当前为：${NETWORK_MODE}" >&2
            return 1
            ;;
    esac
}

select_launch_network_mode() {
    case "${NETWORK_MODE}" in
        ib|bond_rdma|socket)
            echo "${NETWORK_MODE}"
            return 0
            ;;
        auto)
            if hdr_ib_ready && \
                ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
                    "test -r '${SCRIPT_PATH}' && bash '${SCRIPT_PATH}' --check-ib-ready" >/dev/null 2>&1; then
                echo "ib"
            elif bond_rdma_ready && \
                ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
                    "test -r '${SCRIPT_PATH}' && bash '${SCRIPT_PATH}' --check-bond-rdma-ready" >/dev/null 2>&1; then
                echo "bond_rdma"
            else
                echo "socket"
            fi
            ;;
        *)
            echo "[错误] NETWORK_MODE 只能是 auto、ib、bond_rdma 或 socket，当前为：${NETWORK_MODE}" >&2
            return 1
            ;;
    esac
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
    unset WANDB_DISABLED || true

    # torchrun/Gloo/NCCL bootstrap 使用当前节点实际存在的 socket 网卡。
    RUNTIME_SOCKET_IFNAME="$(detect_socket_ifname "${SOCKET_IFNAME}")"
    export RUNTIME_SOCKET_IFNAME
    export GLOO_SOCKET_IFNAME="${RUNTIME_SOCKET_IFNAME}"
    export NCCL_SOCKET_IFNAME="=${RUNTIME_SOCKET_IFNAME}"
    export NCCL_SOCKET_FAMILY="AF_INET"

    # 跨节点张量通信依次使用双 HDR IB、mlx5_bond_0 RDMA，最后退回 bond0 socket。
    RUNTIME_NETWORK_MODE="$(resolve_network_mode)"
    export RUNTIME_NETWORK_MODE
    case "${RUNTIME_NETWORK_MODE}" in
        ib)
            export NCCL_IB_DISABLE=0
            export NCCL_IB_HCA="${NCCL_HDR_IB_HCA_VALUE}"
            ;;
        bond_rdma)
            export NCCL_IB_DISABLE=0
            export NCCL_IB_HCA="${NCCL_BOND_RDMA_HCA_VALUE}"
            ;;
        socket)
            export NCCL_IB_DISABLE=1
            unset NCCL_IB_HCA || true
            ;;
    esac

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
    if [[ -n "${RESUME_PATH}" ]]; then
        check_file "RESUME_PATH" "${RESUME_PATH}"
    fi

    echo "[OK] RUNTIME_SOCKET_IFNAME: ${RUNTIME_SOCKET_IFNAME:-未设置}"
    echo "[OK] RUNTIME_NETWORK_MODE: ${RUNTIME_NETWORK_MODE:-未设置}"
    echo "[OK] NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-未设置}"
    if [[ "${RUNTIME_NETWORK_MODE:-}" == "ib" || "${RUNTIME_NETWORK_MODE:-}" == "bond_rdma" ]]; then
        echo "[OK] NCCL_IB_HCA: ${NCCL_IB_HCA:-未设置}"
    fi
    print_ib_status

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
        --lambda_camera_flow 0.1
        --lambda_camera_pose 0.25
        --camera_pose_start_step 0
        --camera_pose_warmup_steps 10000
        --camera_absolute_translation_scale_m 10.0
        --camera_relative_translation_scale_m 1.0
        --camera_acceleration_translation_scale_m 1.0
        --lambda_sky_flow 0.1
        --actor_geometry_alignment_version camera_pullback_8corner_v1
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
        --val_every "${VAL_EVERY}"
        --val_batches "${VAL_BATCHES}"
        --val_log_images "${VAL_LOG_IMAGES}"
        --val_sample_steps "${VAL_SAMPLE_STEPS}"
        --grad_clip_norm 1.0
        --seed 0
        --precision bf16
        --ddp_timeout_minutes 60
        --wandb
        --wandb_project "${WANDB_PROJECT}"
        --wandb_name "${WANDB_NAME}"
        --wandb_resume "${WANDB_RESUME}"
    )
    if [[ -n "${WANDB_RUN_ID}" ]]; then
        TRAIN_ARGS+=(--wandb_run_id "${WANDB_RUN_ID}")
    fi
    if [[ -n "${RESUME_PATH}" ]]; then
        TRAIN_ARGS+=(--resume_path "${RESUME_PATH}")
        if (( RESUME_EXPECTED_STEP >= 0 )); then
            TRAIN_ARGS+=(--resume_expected_step "${RESUME_EXPECTED_STEP}")
        fi
    elif (( RESUME_EXPECTED_STEP >= 0 )); then
        echo "[错误] RESUME_EXPECTED_STEP=${RESUME_EXPECTED_STEP} 但未设置 RESUME_PATH。" >&2
        exit 1
    fi
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
    echo "[${node_name}] steps: max=${MAX_STEPS}, decay_end=${DECAY_END_STEPS}, save_every=${SAVE_EVERY}"
    echo "[${node_name}] validation: every=${VAL_EVERY}, batches=${VAL_BATCHES}, log_images=${VAL_LOG_IMAGES}, sample_steps=${VAL_SAMPLE_STEPS}, cfg=${VAL_GUIDANCE_SCALES}"
    echo "[${node_name}] output: ${LOG_DIR}"
    echo "[${node_name}] wandb: ${WANDB_PROJECT}/${WANDB_NAME} (resume=${WANDB_RESUME}, id=${WANDB_RUN_ID:-<none>})"

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
    --print-train-args)
        build_train_args
        printf 'LOG_DIR=%q\n' "${LOG_DIR}"
        printf 'LAUNCH_LOG_DIR=%q\n' "${LAUNCH_LOG_DIR}"
        printf 'WANDB=%q/%q\n' "${WANDB_PROJECT}" "${WANDB_NAME}"
        printf 'TRAIN_ARGS='
        printf '%q ' "${TRAIN_ARGS[@]}"
        printf '\n'
        ;;

    --check-ib-ready)
        if hdr_ib_ready; then
            echo "[OK] mlx5_4:1 和 mlx5_5:1 已达到 ACTIVE/LinkUp/200G。"
            exit 0
        fi
        echo "[WARN] mlx5_4:1 和 mlx5_5:1 未同时达到 ACTIVE/LinkUp/200G。"
        print_ib_status
        exit 1
        ;;

    --check-bond-rdma-ready)
        if bond_rdma_ready; then
            echo "[OK] mlx5_bond_0:1 已达到 ACTIVE/LinkUp/至少 ${BOND_RDMA_MIN_RATE_GBPS}G。"
            exit 0
        fi
        echo "[WARN] mlx5_bond_0:1 未达到 ACTIVE/LinkUp/至少 ${BOND_RDMA_MIN_RATE_GBPS}G。"
        print_ib_status
        exit 1
        ;;

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

    --stop)
        echo "[停止] 先停止副节点，再停止主节点。"
        run_remote_script --stop-worker
        stop_node "主节点 ${MASTER_ADDR}" "${MASTER_PID_FILE}"
        ;;

    --launch)
        LAUNCH_NETWORK_MODE="$(select_launch_network_mode)"
        export NETWORK_MODE="${LAUNCH_NETWORK_MODE}"

        echo "============================================================"
        echo "2 节点 × 8 GPU 分布式训练启动器"
        echo "脚本路径：${SCRIPT_PATH}"
        echo "主节点：${MASTER_ADDR}, rank=${MASTER_RANK}"
        echo "副节点：${WORKER_USER}@${WORKER_HOST}:${SSH_PORT}, rank=${WORKER_RANK}"
        echo "Python 环境：${CONDA_ENV}"
        echo "网络模式：${NETWORK_MODE} (ib=mlx5_4/5 HDR, bond_rdma=mlx5_bond_0 RDMA, socket=bond0 TCP)"
        echo "全局 batch size：${GLOBAL_BATCH_SIZE}"
        echo "============================================================"

        echo
        echo "=== 1/4 检查主节点 ==="
        preflight_check "主节点 ${MASTER_ADDR}"

        echo
        echo "=== 2/4 检查副节点 SSH 和运行环境 ==="
        run_remote_script --check-worker

        echo
        echo "=== 3/4 启动副节点 rank ${WORKER_RANK} ==="
        run_remote_script --worker

        echo
        echo "=== 4/4 启动主节点 rank ${MASTER_RANK} ==="
        if ! launch_node \
            "${MASTER_RANK}" \
            "主节点 ${MASTER_ADDR}" \
            "${MASTER_LOG}" \
            "${MASTER_PID_FILE}"; then

            echo "主节点启动失败，尝试终止副节点进程……" >&2
            run_remote_script --stop-worker || true
            exit 1
        fi

        echo
        echo "============================================================"
        echo "2 节点 16 GPU 训练已启动"
        echo "主节点日志：${MASTER_LOG}"
        echo "副节点日志：${WORKER_LOG}"
        echo "主节点 PID 文件：${MASTER_PID_FILE}"
        echo "副节点 PID 文件：${WORKER_PID_FILE}"
        echo "网络模式：${NETWORK_MODE}"
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
        echo "  --print-train-args 打印解析后的训练参数，不启动训练" >&2
        echo "  --check-ib-ready 只检查当前节点 mlx5_4/5 是否达到 ACTIVE/LinkUp/200G" >&2
        echo "  --check-bond-rdma-ready 只检查当前节点 mlx5_bond_0 是否达到 ACTIVE/LinkUp/至少 ${BOND_RDMA_MIN_RATE_GBPS}G" >&2
        echo "  --check-worker  只检查当前节点" >&2
        echo "  --worker        在当前节点启动 worker rank" >&2
        echo "  --master        在当前节点启动 master rank" >&2
        echo "  --stop-worker   停止当前节点的 worker" >&2
        echo "  --stop-master   停止当前节点的 master" >&2
        echo "  --stop          从主节点依次停止 worker 和 master" >&2
        exit 2
        ;;
esac
