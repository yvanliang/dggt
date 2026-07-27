#!/usr/bin/env bash
export WANDB_API_KEY="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"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

# ============================================================
# 使用方式
#   在主节点 A 上执行：
#     bash pretrain_four_nodes.sh
#
# 脚本会：
#   1. 检查主节点 26 的文件、Python、PyTorch、GPU、网卡和 IB；
#   2. 通过 SSH 检查 25、30、31；
#   3. 先启动 worker ranks 1、2、3；
#   4. 再在 26 上启动 node_rank=0。
#
# 注意：
#   - 四台机器上的项目、数据、环境和模型路径统一通过 /home/wuzn/liangyy 访问；
#   - 不执行 conda activate；
#   - 始终使用 CONDA_ENV/bin/python，避免激活到其他同名环境。
# ============================================================

MODE="${1:---launch}"

# ============================================================
# 分布式配置：4 台机器 × 每台 8 GPU
# ============================================================
MASTER_ADDR="10.199.7.26"
MASTER_PORT="22229"

NNODES=4
NPROC_PER_NODE=8

MASTER_RANK=0
WORKER_USER="wuzn"
SSH_PORT=2288

NODE_HOSTS=(
    "10.199.7.26"
    "10.199.7.25"
    "10.199.7.30"
    "10.199.7.31"
)
WORKER_RANKS=(1 2 3)

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
# /mnt/vol1/liangyy_workspace/... 启动，也不会把 /mnt/vol1 前缀传给 worker。
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/pretrain_four_nodes.sh}"

# ============================================================
# 数据与模型路径
# 要求四台服务器上这些路径均存在
# ============================================================
WAYMO_DGGT_ROOT="${DATASET_ROOT}/training"
WAYMO_DGGT_VAL_ROOT="${DATASET_ROOT}/validation"
DGGT_CKPT="${PROJECT_ROOT}/pretrained/model_latest_waymo.pt"
TOKENIZER_CKPT="${PROJECT_ROOT}/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt"
FEATURE_STATS="${PROJECT_ROOT}/logs/scene_flow_pretrain_1024/feature_stats_pretrain_v3.pt"
SCENE_CAPTION_ROOT="${DATASET_ROOT}/training_captions"
SCENE_CAPTION_VAL_ROOT="${DATASET_ROOT}/validation_captions"
QWEN_TEXT_ENCODER="${QWEN_TEXT_ENCODER:-${LIANGYY_ROOT}/model/Qwen/Qwen3-0.6B}"

# 四机训练使用独立目录，避免覆盖单机/两机训练的 checkpoint、验证结果和状态文件。
LOG_DIR="${PROJECT_ROOT}/logs/scene_flow_pretrain_1024_v3"
LAUNCH_LOG_DIR="${PROJECT_ROOT}/logs/distributed_launch_four_nodes"

# 以下变量在原脚本中定义，但当前训练命令没有使用。
# 如果 train_scene_flow_pretrain.py 确实需要它们，请按实际参数名加入 TRAIN_ARGS。
SCENE_FLOW_TRAIN_MANIFEST="${DATASET_ROOT}/waymo_edit_cache/manifests/training/training_manifest.jsonl"
SCENE_FLOW_VAL_MANIFEST="${DATASET_ROOT}/waymo_edit_cache/manifests/validation/validation_manifest.jsonl"

# ============================================================
# 训练配置
# 当前全局 batch = NNODES x NPROC_PER_NODE x BATCH_SIZE_PER_GPU x GRAD_ACCUM_STEPS。
# 修改 BATCH_SIZE_PER_GPU 或 GRAD_ACCUM_STEPS 后，以启动时打印的 GLOBAL_BATCH_SIZE 为准。
# ============================================================
BATCH_SIZE_PER_GPU=1
GRAD_ACCUM_STEPS=2
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

WANDB_NAME="scene_flow_pretrain_waymo_gb64_lr1e4_v3"
GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))

# ============================================================
# NCCL 网络配置
# ============================================================

# 用于 torchrun rendezvous、Gloo 和 NCCL bootstrap。
# 26/25/30 的管理/启动网络通常走 bond0，31 走 bond4；如果当前节点没有
# 指定网卡，启动时会按到其它节点 IP 的路由自动选择。
SOCKET_IFNAME="${SOCKET_IFNAME:-bond0}"

# 大数据通信优先使用 200G HDR InfiniBand。auto 模式会在四台机器之间
# 选择共同可用的 mlx5_4、mlx5_5；没有共同 HDR 时尝试 mlx5_bond_0。
NCCL_IB_HCA_VALUE="${NCCL_IB_HCA_VALUE:-=mlx5_4:1,mlx5_5:1}"
NCCL_BOND_RDMA_HCA_VALUE="=mlx5_bond_0:1"
BOND_RDMA_MIN_RATE_GBPS=10

# auto:      四台共同 HDR -> 四台 mlx5_bond_0 RDMA；默认不允许 socket 回退。
# ib:        强制要求本节点 NCCL_IB_HCA_VALUE 中指定的 HDR HCA 可用。
# bond_rdma: 强制要求本节点 mlx5_bond_0 达到 ACTIVE/LinkUp/至少 10G。
# socket:    强制禁用 RDMA，只走 SOCKET_IFNAME。
NETWORK_MODE="${NETWORK_MODE:-auto}"

# 0: auto 模式找不到四台共同 RDMA 时直接报错，禁止某些节点悄悄退到 socket。
# 1: 明确允许最后退到 socket。当前训练要求高速网络，默认严格禁用。
ALLOW_SOCKET_FALLBACK="${ALLOW_SOCKET_FALLBACK:-0}"

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

    # 优先使用系统路由表选择到其它节点 IP 的实际网卡；这会让 26/25/30
    # 选择 bond0，而让路由不同的 31 正确选择 bond4。目标为本机时跳过 lo。
    for target in "${NODE_HOSTS[@]}"; do
        dev="$(ip -o -4 route get "${target}" 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="dev") {print $(i+1); exit}}' || true)"
        if [[ -n "${dev}" && "${dev}" != "lo" && -d "/sys/class/net/${dev}" ]]; then
            echo "${dev}"
            return 0
        fi
    done

    # 路由检测失败时才使用显式指定的网卡。
    if [[ -n "${requested}" && -d "/sys/class/net/${requested}" ]]; then
        echo "${requested}"
        return 0
    fi

    if [[ -n "${requested}" ]]; then
        echo "[警告] 当前节点不存在网卡 ${requested}，改为按常见网卡名兜底。" >&2
    fi

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

node_host_for_rank() {
    local rank="$1"

    if [[ ! "${rank}" =~ ^[0-9]+$ ]] || (( rank < 0 || rank >= NNODES )); then
        echo "[错误] node_rank 无效：${rank}" >&2
        return 1
    fi

    echo "${NODE_HOSTS[rank]}"
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

ib_ready() {
    rdma_hca_port_ready mlx5_4 1 200 && rdma_hca_port_ready mlx5_5 1 200
}

bond_rdma_ready() {
    rdma_hca_port_ready mlx5_bond_0 1 "${BOND_RDMA_MIN_RATE_GBPS}"
}

ready_ib_hcas() {
    local hca
    local ready=()

    for hca in mlx5_4 mlx5_5; do
        if rdma_hca_port_ready "${hca}" 1 200; then
            ready+=("${hca}")
        fi
    done

    echo "${ready[*]}"
}

required_ib_hcas() {
    local spec="${NCCL_IB_HCA_VALUE}"
    local item
    local hca
    local hcas=()
    local -a items

    spec="${spec#=}"
    IFS=',' read -r -a items <<< "${spec}"
    for item in "${items[@]}"; do
        hca="${item%%:*}"
        hca="${hca#=}"
        if [[ -n "${hca}" ]]; then
            hcas+=("${hca}")
        fi
    done

    echo "${hcas[*]}"
}

required_ib_ready() {
    local hca
    local hcas

    read -r -a hcas <<< "$(required_ib_hcas)"
    if [[ "${#hcas[@]}" -eq 0 ]]; then
        return 1
    fi

    for hca in "${hcas[@]}"; do
        rdma_hca_port_ready "${hca}" 1 200 || return 1
    done
}

hca_list_contains() {
    local needle="$1"
    shift
    local hca

    for hca in "$@"; do
        if [[ "${hca}" == "${needle}" ]]; then
            return 0
        fi
    done

    return 1
}

intersect_hca_lists() {
    local current_string="$1"
    local next_string="$2"
    local current
    local next
    local hca
    local intersection=()

    read -r -a current <<< "${current_string}"
    read -r -a next <<< "${next_string}"

    for hca in "${current[@]}"; do
        if hca_list_contains "${hca}" "${next[@]}"; then
            intersection+=("${hca}")
        fi
    done

    echo "${intersection[*]}"
}

hca_value_from_list() {
    local hcas=("$@")
    local value=""
    local hca

    for hca in "${hcas[@]}"; do
        if [[ -z "${value}" ]]; then
            value="=${hca}:1"
        else
            value="${value},${hca}:1"
        fi
    done

    echo "${value}"
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
            if ! required_ib_ready; then
                echo "[错误] NETWORK_MODE=ib，但本节点未满足 NCCL_IB_HCA_VALUE=${NCCL_IB_HCA_VALUE} 要求。" >&2
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
            if ib_ready; then
                echo "ib"
            elif bond_rdma_ready; then
                echo "bond_rdma"
            elif [[ "${ALLOW_SOCKET_FALLBACK}" == "1" ]]; then
                echo "socket"
            else
                echo "[错误] 本节点没有可用 RDMA，且 ALLOW_SOCKET_FALLBACK=${ALLOW_SOCKET_FALLBACK}，拒绝退到 socket。" >&2
                print_ib_status >&2
                return 1
            fi
            ;;
        *)
            echo "[错误] NETWORK_MODE 只能是 auto、ib、bond_rdma 或 socket，当前为：${NETWORK_MODE}" >&2
            return 1
            ;;
    esac
}

select_launch_network_mode() {
    local rank
    local host
    local common_hcas
    local remote_hcas
    local hcas
    local hca_value
    local all_bond_rdma_ready

    case "${NETWORK_MODE}" in
        ib)
            echo "ib ${NCCL_IB_HCA_VALUE}"
            return 0
            ;;
        bond_rdma)
            echo "bond_rdma ${NCCL_BOND_RDMA_HCA_VALUE}"
            return 0
            ;;
        socket)
            echo "socket -"
            return 0
            ;;
        auto)
            common_hcas="$(ready_ib_hcas)"

            for rank in "${WORKER_RANKS[@]}"; do
                host="$(node_host_for_rank "${rank}")"
                remote_hcas="$(ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${host}" \
                    "test -r '${SCRIPT_PATH}' && bash '${SCRIPT_PATH}' --print-ready-ib-hcas" 2>/dev/null || true)"
                common_hcas="$(intersect_hca_lists "${common_hcas}" "${remote_hcas}")"
            done

            read -r -a hcas <<< "${common_hcas}"
            if [[ "${#hcas[@]}" -gt 0 ]]; then
                hca_value="$(hca_value_from_list "${hcas[@]}")"
                echo "ib ${hca_value}"
            else
                all_bond_rdma_ready=1
                if ! bond_rdma_ready; then
                    all_bond_rdma_ready=0
                fi

                if [[ "${all_bond_rdma_ready}" -eq 1 ]]; then
                    for rank in "${WORKER_RANKS[@]}"; do
                        host="$(node_host_for_rank "${rank}")"
                        if ! ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${host}" \
                            "test -r '${SCRIPT_PATH}' && bash '${SCRIPT_PATH}' --check-bond-rdma-ready" >/dev/null 2>&1; then
                            all_bond_rdma_ready=0
                            break
                        fi
                    done
                fi

                if [[ "${all_bond_rdma_ready}" -eq 1 ]]; then
                    echo "bond_rdma ${NCCL_BOND_RDMA_HCA_VALUE}"
                elif [[ "${ALLOW_SOCKET_FALLBACK}" == "1" ]]; then
                    echo "socket -"
                else
                    echo "[错误] 四台没有共同可用的 HDR 或 mlx5_bond_0 RDMA；已禁止 socket 回退。" >&2
                    echo "       如确需临时使用 TCP，请显式设置 ALLOW_SOCKET_FALLBACK=1。" >&2
                    return 1
                fi
            fi
            ;;
        *)
            echo "[错误] NETWORK_MODE 只能是 auto、ib、bond_rdma 或 socket，当前为：${NETWORK_MODE}" >&2
            return 1
            ;;
    esac
}

verify_worker_scripts_sync() {
    local local_script="${BASH_SOURCE[0]}"
    local local_sha
    local rank
    local host
    local remote_sha

    local_sha="$(sha256sum "${local_script}" | cut -d ' ' -f 1)"
    for rank in "${WORKER_RANKS[@]}"; do
        host="$(node_host_for_rank "${rank}")"
        remote_sha="$(ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${host}" \
            "test -r '${SCRIPT_PATH}' && sha256sum '${SCRIPT_PATH}' | cut -d ' ' -f 1" 2>/dev/null || true)"
        if [[ -z "${remote_sha}" ]]; then
            echo "[错误] 无法读取 rank ${rank} 脚本或计算 SHA256：${host}:${SCRIPT_PATH}" >&2
            return 1
        fi
        if [[ "${local_sha}" != "${remote_sha}" ]]; then
            echo "[错误] rank ${rank} 启动脚本与主节点不一致，拒绝启动以避免 NCCL 配置分叉。" >&2
            echo "       主节点 ${local_script}: ${local_sha}" >&2
            echo "       rank ${rank} ${host}:${SCRIPT_PATH}: ${remote_sha}" >&2
            return 1
        fi
    done
    echo "[OK] 四台节点启动脚本 SHA256 一致：${local_sha}"
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

    # 跨节点张量通信依次使用共同 HDR、mlx5_bond_0 RDMA，最后退回 socket。
    RUNTIME_NETWORK_MODE="$(resolve_network_mode)"
    export RUNTIME_NETWORK_MODE
    case "${RUNTIME_NETWORK_MODE}" in
        ib)
            export NCCL_NET=IB
            export NCCL_IB_DISABLE=0
            export NCCL_IB_HCA="${NCCL_IB_HCA_VALUE}"
            ;;
        bond_rdma)
            export NCCL_NET=IB
            export NCCL_IB_DISABLE=0
            export NCCL_IB_HCA="${NCCL_BOND_RDMA_HCA_VALUE}"
            ;;
        socket)
            export NCCL_NET=Socket
            export NCCL_IB_DISABLE=1
            unset NCCL_IB_HCA || true
            ;;
    esac

    # NCCL 2.21+ 会动态选择 RoCE GID；清除外部 Shell 中可能残留的固定索引。
    unset NCCL_IB_GID_INDEX || true
    export NCCL_IB_ADDR_FAMILY=AF_INET
    export NCCL_IB_ROCE_VERSION_NUM=2

    # 首次启动保留 INIT/NET 信息，用于逐节点确认 NET/IB 和实际 HCA。
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS="INIT,NET,ENV"

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
    check_dir "SCENE_CAPTION_ROOT" "${SCENE_CAPTION_ROOT}"
    check_dir "SCENE_CAPTION_VAL_ROOT" "${SCENE_CAPTION_VAL_ROOT}"
    check_dir "QWEN_TEXT_ENCODER" "${QWEN_TEXT_ENCODER}"

    echo "[OK] RUNTIME_SOCKET_IFNAME: ${RUNTIME_SOCKET_IFNAME:-未设置}"
    echo "[OK] RUNTIME_NETWORK_MODE: ${RUNTIME_NETWORK_MODE:-未设置}"
    echo "[OK] NCCL_NET: ${NCCL_NET:-未设置}"
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
print(f"[{node}] NCCL: {torch.cuda.nccl.version()}")
print(f"[{node}] CUDA available: {torch.cuda.is_available()}")
print(f"[{node}] CUDA version seen by PyTorch: {torch.version.cuda}")
print(f"[{node}] visible GPU count: {count}")

if not torch.cuda.is_available():
    raise RuntimeError(f"[{node}] PyTorch 无法使用 CUDA")
if count != expected:
    raise RuntimeError(f"[{node}] 期望 {expected} 张 GPU，实际检测到 {count} 张")
PY

    if [[ "${RUNTIME_NETWORK_MODE:-}" == "ib" || "${RUNTIME_NETWORK_MODE:-}" == "bond_rdma" ]]; then
        "${PYTHON_BIN}" - <<'PY'
import ctypes
ctypes.CDLL("libibverbs.so.1")
print("[OK] libibverbs.so.1 可以由训练 Python 加载")
PY

        if command -v ibv_devinfo >/dev/null 2>&1; then
            local spec
            local item
            local hca
            local -a items
            spec="${NCCL_IB_HCA#=}"
            IFS=',' read -r -a items <<< "${spec}"
            for item in "${items[@]}"; do
                hca="${item%%:*}"
                if ! ibv_devinfo -d "${hca}" >/dev/null 2>&1; then
                    echo "[错误] libibverbs 无法打开 NCCL HCA：${hca}" >&2
                    ibv_devinfo -l >&2 || true
                    return 1
                fi
                echo "[OK] libibverbs HCA: ${hca}"
            done
        else
            echo "[警告] 未安装 ibv_devinfo，跳过 HCA userspace 诊断；NCCL_NET=IB 仍会禁止 socket 回退。" >&2
        fi
    fi

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
# 训练参数：四个节点共用同一份，避免各节点参数不一致
# ============================================================
build_train_args() {
    TRAIN_ARGS=(
        train_scene_flow_pretrain.py
        --image_dir "${WAYMO_DGGT_ROOT}"
        --val_image_dir "${WAYMO_DGGT_VAL_ROOT}"
        --dggt_ckpt_path "${DGGT_CKPT}"
        --tokenizer_ckpt_path "${TOKENIZER_CKPT}"
        --feature_stats_path "${FEATURE_STATS}"
        --resume_path "${LOG_DIR}/ckpt/pretrain_step018000.pt"
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
        --rgb_render_every 2
        --grad_clip_norm 1.0
        --seed 0
        --precision bf16
        --ddp_timeout_minutes 60
        --wandb
        --wandb_project dggt-flow
        --wandb_name "${WANDB_NAME}"
        --wandb_run_id "0bmu7zky"
        --wandb_resume must
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
log_file_for_rank() {
    local rank="$1"
    echo "${LAUNCH_LOG_DIR}/node_rank${rank}.log"
}

pid_file_for_rank() {
    local rank="$1"
    echo "${LAUNCH_LOG_DIR}/node_rank${rank}.pid"
}

stop_remote_worker() {
    local rank="$1"
    local host

    host="$(node_host_for_rank "${rank}")"
    ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${host}" \
        "NETWORK_MODE='${NETWORK_MODE}' NCCL_IB_HCA_VALUE='${NCCL_IB_HCA_VALUE}' bash '${SCRIPT_PATH}' --stop-worker '${rank}'" || true
}

MASTER_LOG="$(log_file_for_rank "${MASTER_RANK}")"
MASTER_PID_FILE="$(pid_file_for_rank "${MASTER_RANK}")"

# ============================================================
# 运行模式
# ============================================================
case "${MODE}" in
    --print-ready-ib-hcas)
        ready_ib_hcas
        exit 0
        ;;

    --check-ib-ready)
        if required_ib_ready; then
            echo "[OK] NCCL_IB_HCA_VALUE=${NCCL_IB_HCA_VALUE} 中指定的 HCA 已达到 ACTIVE/LinkUp/200G。"
            exit 0
        fi
        echo "[WARN] NCCL_IB_HCA_VALUE=${NCCL_IB_HCA_VALUE} 中指定的 HCA 未全部达到 ACTIVE/LinkUp/200G。"
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
        NODE_RANK="${2:?用法：bash ${0} --check-worker <node_rank>}"
        NODE_HOST="$(node_host_for_rank "${NODE_RANK}")"
        preflight_check "节点 rank ${NODE_RANK} ${NODE_HOST}"
        ;;

    --worker)
        NODE_RANK="${2:?用法：bash ${0} --worker <node_rank>}"
        NODE_HOST="$(node_host_for_rank "${NODE_RANK}")"
        launch_node \
            "${NODE_RANK}" \
            "节点 rank ${NODE_RANK} ${NODE_HOST}" \
            "$(log_file_for_rank "${NODE_RANK}")" \
            "$(pid_file_for_rank "${NODE_RANK}")"
        ;;

    --master)
        launch_node \
            "${MASTER_RANK}" \
            "主节点 ${MASTER_ADDR}" \
            "${MASTER_LOG}" \
            "${MASTER_PID_FILE}"
        ;;

    --stop-worker)
        NODE_RANK="${2:?用法：bash ${0} --stop-worker <node_rank>}"
        NODE_HOST="$(node_host_for_rank "${NODE_RANK}")"
        stop_node "节点 rank ${NODE_RANK} ${NODE_HOST}" "$(pid_file_for_rank "${NODE_RANK}")"
        ;;

    --stop-master)
        stop_node "主节点 ${MASTER_ADDR}" "${MASTER_PID_FILE}"
        ;;

    --launch)
        verify_worker_scripts_sync
        read -r LAUNCH_NETWORK_MODE LAUNCH_NCCL_IB_HCA_VALUE <<< "$(select_launch_network_mode)"
        export NETWORK_MODE="${LAUNCH_NETWORK_MODE}"
        if [[ "${NETWORK_MODE}" == "ib" || "${NETWORK_MODE}" == "bond_rdma" ]]; then
            export NCCL_IB_HCA_VALUE="${LAUNCH_NCCL_IB_HCA_VALUE}"
        fi
        LAUNCHED_WORKER_RANKS=()

        echo "============================================================"
        echo "4 节点 × 8 GPU 分布式训练启动器"
        echo "脚本路径：${SCRIPT_PATH}"
        echo "主节点：${MASTER_ADDR}, rank=${MASTER_RANK}"
        echo "节点列表：${NODE_HOSTS[*]}"
        echo "worker ranks：${WORKER_RANKS[*]}"
        echo "Python 环境：${CONDA_ENV}"
        echo "网络模式：${NETWORK_MODE} (ib=共同可用 mlx5 HDR, bond_rdma=四台 mlx5_bond_0 RDMA, socket=仅显式允许时使用)"
        echo "允许 socket 自动回退：${ALLOW_SOCKET_FALLBACK}"
        if [[ "${NETWORK_MODE}" == "ib" || "${NETWORK_MODE}" == "bond_rdma" ]]; then
            echo "NCCL_IB_HCA：${NCCL_IB_HCA_VALUE}"
        fi
        echo "全局 batch size：${GLOBAL_BATCH_SIZE}"
        echo "============================================================"

        echo
        echo "=== 1/4 检查主节点 ==="
        preflight_check "主节点 ${MASTER_ADDR}"

        echo
        echo "=== 2/4 检查 worker 节点 SSH 和运行环境 ==="
        for rank in "${WORKER_RANKS[@]}"; do
            host="$(node_host_for_rank "${rank}")"
            echo "[rank ${rank}] 检查 ${host}"
            ssh "${SSH_OPTS[@]}" \
                "${WORKER_USER}@${host}" \
                "test -r '${SCRIPT_PATH}' && NETWORK_MODE='${NETWORK_MODE}' NCCL_IB_HCA_VALUE='${NCCL_IB_HCA_VALUE}' bash '${SCRIPT_PATH}' --check-worker '${rank}'"
        done

        echo
        echo "=== 3/4 启动 worker ranks ${WORKER_RANKS[*]} ==="
        for rank in "${WORKER_RANKS[@]}"; do
            host="$(node_host_for_rank "${rank}")"
            echo "[rank ${rank}] 启动 ${host}"
            if ssh "${SSH_OPTS[@]}" \
                "${WORKER_USER}@${host}" \
                "NETWORK_MODE='${NETWORK_MODE}' NCCL_IB_HCA_VALUE='${NCCL_IB_HCA_VALUE}' bash '${SCRIPT_PATH}' --worker '${rank}'"; then
                LAUNCHED_WORKER_RANKS+=("${rank}")
            else
                echo "worker rank ${rank} 启动失败，尝试终止已启动 worker……" >&2
                for launched_rank in "${LAUNCHED_WORKER_RANKS[@]}"; do
                    stop_remote_worker "${launched_rank}"
                done
                exit 1
            fi
        done

        echo
        echo "=== 4/4 启动主节点 rank ${MASTER_RANK} ==="
        if ! launch_node \
            "${MASTER_RANK}" \
            "主节点 ${MASTER_ADDR}" \
            "${MASTER_LOG}" \
            "${MASTER_PID_FILE}"; then

            echo "主节点启动失败，尝试终止 worker 进程……" >&2
            for launched_rank in "${LAUNCHED_WORKER_RANKS[@]}"; do
                stop_remote_worker "${launched_rank}"
            done
            exit 1
        fi

        echo
        echo "============================================================"
        echo "4 节点 32 GPU 训练已启动"
        echo "主节点日志：${MASTER_LOG}"
        echo "主节点 PID 文件：${MASTER_PID_FILE}"
        echo "网络模式：${NETWORK_MODE}"
        if [[ "${NETWORK_MODE}" == "ib" || "${NETWORK_MODE}" == "bond_rdma" ]]; then
            echo "NCCL_IB_HCA：${NCCL_IB_HCA_VALUE}"
        fi
        for rank in "${WORKER_RANKS[@]}"; do
            echo "worker rank ${rank} 日志：$(log_file_for_rank "${rank}")"
            echo "worker rank ${rank} PID 文件：$(pid_file_for_rank "${rank}")"
        done
        echo "全局 batch size：${GLOBAL_BATCH_SIZE} = ${NNODES} nodes × ${NPROC_PER_NODE} gpu/node × ${BATCH_SIZE_PER_GPU} batch/gpu × ${GRAD_ACCUM_STEPS} accum"
        echo
        echo "查看日志："
        echo "  tail -f '${LAUNCH_LOG_DIR}'/node_rank*.log"
        echo "============================================================"
        ;;

    *)
        echo "未知参数：${MODE}" >&2
        echo "可用参数：" >&2
        echo "  --launch        从主节点启动四机训练（默认）" >&2
        echo "  --print-ready-ib-hcas 打印当前节点可用 mlx5_4/5 HCA 名称" >&2
        echo "  --check-ib-ready 只检查当前节点 mlx5_4/5 是否达到 ACTIVE/LinkUp/200G" >&2
        echo "  --check-bond-rdma-ready 只检查当前节点 mlx5_bond_0 是否达到 ACTIVE/LinkUp/至少 ${BOND_RDMA_MIN_RATE_GBPS}G" >&2
        echo "  --check-worker <rank> 只检查当前节点" >&2
        echo "  --worker <rank>        在当前节点启动 worker rank" >&2
        echo "  --master        在当前节点启动 master rank" >&2
        echo "  --stop-worker <rank>   停止当前节点的 worker" >&2
        echo "  --stop-master   停止当前节点的 master" >&2
        exit 2
        ;;
esac
