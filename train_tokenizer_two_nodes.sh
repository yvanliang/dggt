#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

# ============================================================
# JointSceneTokenizer T0 重训（Stage-A / Stage-B），2 节点 × 8 GPU
#
# 为什么要重训（2026-08-01）：
#   审计 encode/decode 往返发现 paired GS/depth = 0.796（30/30 场景 < 1），
#   即重建把 depth 放大 3%、把高斯半径缩小 17%。根因是
#     (1) gs_anchor 用一个共同 std 归一化 11 个通道，线性 scale 通道被稀释约 1.2e7 倍；
#     (2) gs_anchor 与 geom_anchor 相互独立，没有任何项约束它们的比值。
#   train_tokenizer.py 已加入 gs_channel_group_huber_loss 与 --lambda_gs_scale_sim。
#   这是损失层面的改动，架构不变，但旧 checkpoint 与新目标不可比 —— 故从 0 重训。
#
# 使用方式（在主节点 A 上执行）：
#   bash train_tokenizer_two_nodes.sh --launch-a     # Stage-A，60k step
#   bash train_tokenizer_two_nodes.sh --launch-b     # Stage-B，40k step（需 A 已完成）
#   bash train_tokenizer_two_nodes.sh --stop         # 停止两个节点
#   bash train_tokenizer_two_nodes.sh --plan-a       # 只打印 batch/lr 计划，不启动
#
# 脚本会：先检查两节点的文件/Python/GPU/网卡，再启动副节点，最后启动主节点。
# ============================================================

MODE="${1:---launch-a}"

# ============================================================
# 分布式配置
# ============================================================
MASTER_ADDR="${MASTER_ADDR:-10.199.7.26}"
MASTER_PORT="${MASTER_PORT:-22231}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

MASTER_RANK=0
WORKER_RANK=1

WORKER_HOST="${WORKER_HOST:-10.199.7.25}"
WORKER_USER="${WORKER_USER:-wuzn}"
SSH_PORT="${SSH_PORT:-2288}"

# ============================================================
# 项目与 Python 环境
# ============================================================
LIANGYY_ROOT="${LIANGYY_ROOT:-/home/wuzn/liangyy}"
PROJECT_ROOT="${PROJECT_ROOT:-${LIANGYY_ROOT}/dggt}"
DATASET_ROOT="${DATASET_ROOT:-${LIANGYY_ROOT}/waymo_processed_dggt}"

CONDA_ROOT="${CONDA_ROOT:-/home/wuzn/miniconda3}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ROOT}/envs/dggt}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"

SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/train_tokenizer_two_nodes.sh}"

# ============================================================
# 数据与模型路径
# transfer/raw/asset_root 是 WaymoEditDataset 为编辑数据保留的兼容参数。
# train_tokenizer.py 固定 clean_only=True，Stage-A 及 Stage-B 的 raw 混合分支
# 都只从 processed_root 读取，因此这些路径可以不存在。
# ============================================================
DGGT_CKPT="${DGGT_CKPT:-${PROJECT_ROOT}/pretrained/model_latest_waymo.pt}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${DATASET_ROOT}}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${LIANGYY_ROOT}/waymo_transfer}"
RAW_ROOT="${RAW_ROOT:-${LIANGYY_ROOT}/waymo}"
ASSET_ROOT="${ASSET_ROOT:-${LIANGYY_ROOT}/test_transfer/objects_ply_transformed}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${DATASET_ROOT}/waymo_edit_cache/manifests/training/training_manifest.jsonl}"

STAGE_A_LOG_DIR="${STAGE_A_LOG_DIR:-${PROJECT_ROOT}/logs/tokenizer_t0_v2_stageA}"
STAGE_B_LOG_DIR="${STAGE_B_LOG_DIR:-${PROJECT_ROOT}/logs/tokenizer_t0_v2_stageB}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/tokenizer_v2_launch}"

# Stage-B 从 Stage-A 的最终 checkpoint 初始化（只载权重，optimizer/scheduler 全新）。
STAGE_A_STEPS="${STAGE_A_STEPS:-100000}"
STAGE_B_STEPS="${STAGE_B_STEPS:-40000}"   # A + B = 100000，与「总 iter 10w」一致
STAGE_A_FINAL_CKPT="${STAGE_A_FINAL_CKPT:-${STAGE_A_LOG_DIR}/ckpt/scene_tokenizer_step_$(printf '%06d' "${STAGE_A_STEPS}").pt}"

# ============================================================
# Batch 规划
#
# 数值取自**实跑的** logs/tokenizer_t0_v2_stage{A,B}/config.json，不是 train_tokenizer.py
# docstring 里的示例 —— 那份示例的 Stage-B 写的是 batch 4 / accum 2，与实跑不符。
#   Stage-A 实跑: 2 gpu × batch 1 × accum 8 = 16
#   Stage-B 实跑: 2 gpu × batch 1 × accum 8 = 16
#
# 本次只改损失、不改 batch —— 同时改两样会让「新目标是否有效」无法归因。
# 因此按 GPU 数反推 per-GPU batch 与 accum，把全局 batch 钉在 16。
# 若 world_size 大到连 batch=1、accum=1 都会超过目标，则接受更大的全局 batch，
# 并按 sqrt 规则缩放 LR（AdamW 上比线性规则稳）。
# ============================================================
REFERENCE_GLOBAL_BATCH=16
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-16}"

# per-GPU batch 上限。两个 stage 实跑都是 1（80GB 上 batch 4 未验证过，勿臆测）。
STAGE_A_MAX_BATCH_PER_GPU="${STAGE_A_MAX_BATCH_PER_GPU:-1}"
STAGE_B_MAX_BATCH_PER_GPU="${STAGE_B_MAX_BATCH_PER_GPU:-1}"

STAGE_A_REF_LR="${STAGE_A_REF_LR:-3e-4}"
STAGE_B_REF_LR="${STAGE_B_REF_LR:-8e-5}"

# 输出：BATCH_PER_GPU / GRAD_ACCUM_STEPS / ACTUAL_GLOBAL_BATCH / SCALED_LR
derive_batch_plan() {
    local stage="$1"
    local world_size=$(( NNODES * NPROC_PER_NODE ))
    local max_per_gpu ref_lr

    case "${stage}" in
        a) max_per_gpu="${STAGE_A_MAX_BATCH_PER_GPU}"; ref_lr="${STAGE_A_REF_LR}" ;;
        b) max_per_gpu="${STAGE_B_MAX_BATCH_PER_GPU}"; ref_lr="${STAGE_B_REF_LR}" ;;
        *) echo "[错误] derive_batch_plan 只接受 a 或 b，收到 ${stage}" >&2; return 1 ;;
    esac

    # 取不超过目标的最大 per-GPU batch，最小降到 1。
    local per_gpu="${max_per_gpu}"
    while (( per_gpu > 1 && world_size * per_gpu > TARGET_GLOBAL_BATCH )); do
        per_gpu=$(( per_gpu / 2 ))
    done

    local samples_per_step=$(( world_size * per_gpu ))
    local accum=$(( (TARGET_GLOBAL_BATCH + samples_per_step - 1) / samples_per_step ))
    (( accum < 1 )) && accum=1

    BATCH_PER_GPU="${per_gpu}"
    GRAD_ACCUM_STEPS="${accum}"
    ACTUAL_GLOBAL_BATCH=$(( samples_per_step * accum ))

    # LR 按 sqrt(actual/reference) 缩放；等于参考值时保持原样。
    SCALED_LR="$(awk -v lr="${ref_lr}" -v a="${ACTUAL_GLOBAL_BATCH}" -v r="${REFERENCE_GLOBAL_BATCH}" \
        'BEGIN { printf "%.6g", lr * sqrt(a / r) }')"

    WORLD_SIZE="${world_size}"
}

print_batch_plan() {
    local stage="$1"
    derive_batch_plan "${stage}"
    echo "------------------------------------------------------------"
    echo "Stage-${stage^^} batch 规划"
    echo "  world_size          = ${NNODES} nodes × ${NPROC_PER_NODE} gpu = ${WORLD_SIZE}"
    echo "  batch/gpu           = ${BATCH_PER_GPU}"
    echo "  grad_accum_steps    = ${GRAD_ACCUM_STEPS}"
    echo "  全局 batch          = ${ACTUAL_GLOBAL_BATCH}  (目标 ${TARGET_GLOBAL_BATCH}, 参考配方 ${REFERENCE_GLOBAL_BATCH})"
    echo "  lr                  = ${SCALED_LR}"
    if (( ACTUAL_GLOBAL_BATCH != REFERENCE_GLOBAL_BATCH )); then
        echo "  [注意] 全局 batch 偏离参考配方，LR 已按 sqrt 规则缩放。"
        echo "         若想严格复现参考配方，减少 NPROC_PER_NODE 或 NNODES。"
    fi
    echo "------------------------------------------------------------"
}

# ============================================================
# NCCL 网络配置（沿用 pretrain_two_nodes26.sh 的检测逻辑）
# ============================================================
SOCKET_IFNAME="${SOCKET_IFNAME:-bond0}"
NCCL_HDR_IB_HCA_VALUE="=mlx5_4:1,mlx5_5:1"
NCCL_BOND_RDMA_HCA_VALUE="=mlx5_bond_0:1"
BOND_RDMA_MIN_RATE_GBPS=25
NETWORK_MODE="${NETWORK_MODE:-auto}"

SSH_OPTS=(
    -p "${SSH_PORT}"
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=no
)

detect_socket_ifname() {
    local requested="$1" target dev candidate

    if [[ -n "${requested}" && -d "/sys/class/net/${requested}" ]]; then
        echo "${requested}"; return 0
    fi
    if [[ -n "${requested}" ]]; then
        echo "[警告] 当前节点不存在网卡 ${requested}，改为自动检测。" >&2
    fi
    for target in "${MASTER_ADDR}" "${WORKER_HOST}"; do
        dev="$(ip -o -4 route get "${target}" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}' || true)"
        if [[ -n "${dev}" && "${dev}" != "lo" && -d "/sys/class/net/${dev}" ]]; then
            echo "${dev}"; return 0
        fi
    done
    for candidate in bond4 bond0 bond1 ib0 eth0 eno1 eno2 ens1 ens2 ens3 ens4 enp1s0 enp2s0 enp3s0 enp4s0; do
        [[ -d "/sys/class/net/${candidate}" ]] && { echo "${candidate}"; return 0; }
    done
    echo "[错误] 无法自动检测 NCCL/Gloo bootstrap 网卡。" >&2
    ip -br link >&2 || true
    return 1
}

rdma_hca_port_ready() {
    local hca="$1" port="${2:-1}" min_rate_gbps="${3:-1}"
    local base="/sys/class/infiniband/${hca}/ports/${port}"
    local state phys rate rate_gbps

    [[ -d "${base}" ]] || return 1
    state="$(cat "${base}/state" 2>/dev/null || true)"
    phys="$(cat "${base}/phys_state" 2>/dev/null || true)"
    rate="$(cat "${base}/rate" 2>/dev/null || true)"
    [[ "${state}" == *ACTIVE* ]] || return 1
    [[ "${phys}" == *LinkUp* || "${phys}" == *LINK_UP* ]] || return 1
    [[ "${rate}" =~ ([0-9]+)[[:space:]]*Gb/sec ]] || return 1
    rate_gbps="${BASH_REMATCH[1]}"
    (( rate_gbps >= min_rate_gbps ))
}

hdr_ib_ready()   { rdma_hca_port_ready mlx5_4 1 200 && rdma_hca_port_ready mlx5_5 1 200; }
bond_rdma_ready() { rdma_hca_port_ready mlx5_bond_0 1 "${BOND_RDMA_MIN_RATE_GBPS}"; }

print_ib_status() {
    local hca base port
    for hca in mlx5_4 mlx5_5 mlx5_bond_0; do
        if [[ ! -d "/sys/class/infiniband/${hca}" ]]; then
            echo "[WARN] IB HCA ${hca}: missing"; continue
        fi
        for base in "/sys/class/infiniband/${hca}/ports/"*; do
            [[ -d "${base}" ]] || continue
            port="${base##*/}"
            echo "[INFO] ${hca}:${port} state=$(cat "${base}/state" 2>/dev/null || echo NA) phys=$(cat "${base}/phys_state" 2>/dev/null || echo NA) rate=$(cat "${base}/rate" 2>/dev/null || echo NA)"
        done
    done
}

resolve_network_mode() {
    case "${NETWORK_MODE}" in
        ib)        hdr_ib_ready   || { echo "[错误] NETWORK_MODE=ib 但 mlx5_4/5 未就绪。" >&2; print_ib_status >&2; return 1; }; echo "ib" ;;
        bond_rdma) bond_rdma_ready || { echo "[错误] NETWORK_MODE=bond_rdma 但 mlx5_bond_0 未就绪。" >&2; print_ib_status >&2; return 1; }; echo "bond_rdma" ;;
        socket)    echo "socket" ;;
        auto)      if hdr_ib_ready; then echo "ib"; elif bond_rdma_ready; then echo "bond_rdma"; else echo "socket"; fi ;;
        *)         echo "[错误] NETWORK_MODE 只能是 auto/ib/bond_rdma/socket，当前为 ${NETWORK_MODE}" >&2; return 1 ;;
    esac
}

select_launch_network_mode() {
    case "${NETWORK_MODE}" in
        ib|bond_rdma|socket) echo "${NETWORK_MODE}"; return 0 ;;
        auto)
            if hdr_ib_ready && ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
                    "test -r '${SCRIPT_PATH}' && bash '${SCRIPT_PATH}' --check-ib-ready" >/dev/null 2>&1; then
                echo "ib"
            elif bond_rdma_ready && ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
                    "test -r '${SCRIPT_PATH}' && bash '${SCRIPT_PATH}' --check-bond-rdma-ready" >/dev/null 2>&1; then
                echo "bond_rdma"
            else
                echo "socket"
            fi ;;
        *) echo "[错误] NETWORK_MODE 非法：${NETWORK_MODE}" >&2; return 1 ;;
    esac
}

setup_common_env() {
    unset CONDARC CONDA_PREFIX CONDA_PREFIX_1 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER \
          CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M PYTHONHOME 2>/dev/null || true

    export CONDA_PREFIX="${CONDA_ENV}"
    export CONDA_DEFAULT_ENV="${CONDA_ENV}"
    export PATH="${CONDA_ENV}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="${PROJECT_ROOT}"
    export PYTHONUNBUFFERED=1
    export LD_LIBRARY_PATH="${CONDA_ENV}/lib:/usr/local/cuda/lib64"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    # 冻结 DGGT + 可训 tokenizer + gsplat 渲染，显存碎片较多。
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    RUNTIME_SOCKET_IFNAME="$(detect_socket_ifname "${SOCKET_IFNAME}")"
    export RUNTIME_SOCKET_IFNAME
    export GLOO_SOCKET_IFNAME="${RUNTIME_SOCKET_IFNAME}"
    export NCCL_SOCKET_IFNAME="=${RUNTIME_SOCKET_IFNAME}"
    export NCCL_SOCKET_FAMILY="AF_INET"

    RUNTIME_NETWORK_MODE="$(resolve_network_mode)"
    export RUNTIME_NETWORK_MODE
    case "${RUNTIME_NETWORK_MODE}" in
        ib)        export NCCL_IB_DISABLE=0; export NCCL_IB_HCA="${NCCL_HDR_IB_HCA_VALUE}" ;;
        bond_rdma) export NCCL_IB_DISABLE=0; export NCCL_IB_HCA="${NCCL_BOND_RDMA_HCA_VALUE}" ;;
        socket)    export NCCL_IB_DISABLE=1; unset NCCL_IB_HCA || true ;;
    esac

    export NCCL_DEBUG=WARN
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
}

# ============================================================
# 检查
# ============================================================
check_dir()  { [[ -d "$2" ]] || { echo "[错误] 目录不存在：$1"$'\n'"       $2" >&2; return 1; }; echo "[OK] $1: $2"; }
check_file() { [[ -f "$2" ]] || { echo "[错误] 文件不存在：$1"$'\n'"       $2" >&2; return 1; }; echo "[OK] $1: $2"; }
check_exec() { [[ -x "$2" ]] || { echo "[错误] 不存在或不可执行：$1"$'\n'"       $2" >&2; return 1; }; echo "[OK] $1: $2"; }

check_required_paths() {
    local node_name="$1" stage="$2"
    echo "[${node_name}] 检查文件与目录……"

    check_dir  "PROJECT_ROOT"      "${PROJECT_ROOT}"
    check_file "train_tokenizer.py" "${PROJECT_ROOT}/train_tokenizer.py"
    check_exec "PYTHON_BIN"        "${PYTHON_BIN}"
    check_file "DGGT_CKPT"         "${DGGT_CKPT}"
    check_dir  "PROCESSED_ROOT"    "${PROCESSED_ROOT}"

    if [[ "${stage}" == "b" ]]; then
        check_file "CACHE_MANIFEST"      "${CACHE_MANIFEST}"
        check_file "STAGE_A_FINAL_CKPT"  "${STAGE_A_FINAL_CKPT}"
    fi

    echo "[OK] RUNTIME_SOCKET_IFNAME: ${RUNTIME_SOCKET_IFNAME:-未设置}"
    echo "[OK] RUNTIME_NETWORK_MODE: ${RUNTIME_NETWORK_MODE:-未设置}"
    print_ib_status

    local log_dir
    [[ "${stage}" == "a" ]] && log_dir="${STAGE_A_LOG_DIR}" || log_dir="${STAGE_B_LOG_DIR}"
    mkdir -p "${log_dir}" "${LAUNCH_LOG_DIR}" || {
        echo "[错误] 创建日志目录失败：${log_dir} / ${LAUNCH_LOG_DIR}" >&2; return 1; }
    echo "[OK] LOG_DIR: ${log_dir}"
}

check_python_and_gpu() {
    local node_name="$1"
    echo "[${node_name}] 检查 Python / PyTorch / GPU……"
    EXPECTED_GPU_COUNT="${NPROC_PER_NODE}" NODE_NAME="${node_name}" "${PYTHON_BIN}" - <<'PY'
import os, sys, torch
node = os.environ["NODE_NAME"]; expected = int(os.environ["EXPECTED_GPU_COUNT"])
count = torch.cuda.device_count()
print(f"[{node}] Python: {sys.executable}")
print(f"[{node}] PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  available: {torch.cuda.is_available()}")
print(f"[{node}] visible GPU count: {count}")
if not torch.cuda.is_available():
    raise RuntimeError(f"[{node}] PyTorch 无法使用 CUDA")
if count != expected:
    raise RuntimeError(f"[{node}] 期望 {expected} 张 GPU，实际 {count} 张")
PY
    "${PYTHON_BIN}" -m torch.distributed.run --help >/dev/null
}

preflight_check() {
    local node_name="$1" stage="$2"
    setup_common_env
    check_required_paths "${node_name}" "${stage}"
    check_python_and_gpu "${node_name}"
    echo "[${node_name}] 检查通过。"
}

# ============================================================
# 训练参数
# ============================================================
build_stage_a_args() {
    derive_batch_plan a
    TRAIN_ARGS=(
        train_tokenizer.py
        --ckpt_path "${DGGT_CKPT}"
        --log_dir "${STAGE_A_LOG_DIR}"
        --processed_root "${PROCESSED_ROOT}"
        --transfer_root "${TRANSFER_ROOT}"
        --raw_root "${RAW_ROOT}"
        --asset_root "${ASSET_ROOT}"
        --views 1
        --sample_window 20
        --min_frames 10
        --max_frames 14
        --batch_size "${BATCH_PER_GPU}"
        --grad_accum_steps "${GRAD_ACCUM_STEPS}"
        --num_workers 8
        --prefetch_factor 2
        --mp_sharing_strategy file_system
        --max_steps "${STAGE_A_STEPS}"
        --save_every 5000
        --vis_every 2500
        --log_every 100
        --stats_steps 2048
        --lr "${SCALED_LR}"
        --weight_decay 0.05
        --warmup_steps 4000
        --grad_clip_norm 1.0
        --head_start_step 4000
        --head_warmup_steps 6000
        --render_start_step 10000
        --noisy_start_step 25000
        --decoder_noise_tau 0.8
        --decoder_noise_distribution uniform
        --lambda_tok_rec 1.0
        --lambda_tok_cos 0.2
        --lambda_head_anchor 0.6
        # 三项新目标。日志盯 gs_scale_sim_ratio -> 1.0 与 depth_ratio -> 1.0。
        --lambda_gs_scale_sim 0.3
        --gs_scale_sim_opacity 0.05
        --lambda_depth_log_bias 0.2
        --lambda_render_anchor 0.3
        --gt_render_ratio 1.0
        --render_dyn_alpha 6.0
        --lambda_noisy 0.15
        --lambda_lat_stat 0.05
        --lambda_dynamic_bce 0.2
        --dyn_patch_alpha 6.0
        --dyn_pixel_alpha 10.0
        --lambda_gs_lifespan 0.01
        --lambda_ghost_static 0.0
        --precision bf16
        --seed 0
        --wandb
        --wandb_project dggt-tokenizer
        --wandb_name "t0_v2_stageA_dz1024_gb${ACTUAL_GLOBAL_BATCH}"
    )
}

build_stage_b_args() {
    derive_batch_plan b
    TRAIN_ARGS=(
        train_tokenizer.py
        --init_tokenizer_path "${STAGE_A_FINAL_CKPT}"
        --ckpt_path "${DGGT_CKPT}"
        --cache_manifest_path "${CACHE_MANIFEST}"
        --cache_split training
        --log_dir "${STAGE_B_LOG_DIR}"
        --processed_root "${PROCESSED_ROOT}"
        --transfer_root "${TRANSFER_ROOT}"
        --raw_root "${RAW_ROOT}"
        --asset_root "${ASSET_ROOT}"
        --stage_b_mix_raw
        --views 1
        --sample_window 20
        --min_frames 10
        --max_frames 14
        --batch_size "${BATCH_PER_GPU}"
        --raw_batch_size "${BATCH_PER_GPU}"
        --grad_accum_steps "${GRAD_ACCUM_STEPS}"
        --num_workers 8
        --prefetch_factor 2
        --mp_sharing_strategy file_system
        --max_steps "${STAGE_B_STEPS}"
        --save_every 5000
        --vis_every 2500
        --log_every 100
        --stats_steps 512
        --lr "${SCALED_LR}"
        --weight_decay 0.05
        --warmup_steps 1000
        --grad_clip_norm 1.0
        --head_start_step 0
        --head_warmup_steps 1
        --render_start_step 0
        --noisy_start_step 0
        --decoder_noise_tau 0.8
        --decoder_noise_distribution uniform
        --lambda_tok_rec 0.5
        --lambda_tok_cos 0.1
        --lambda_head_anchor 0.8
        # Stage-B 权重更高：此时 head/render 全开，是把两个比值真正压到 1 的阶段。
        --lambda_gs_scale_sim 0.5
        --gs_scale_sim_opacity 0.05
        --lambda_depth_log_bias 0.3
        --lambda_render_anchor 0.5
        --gt_render_ratio 1.5
        --render_dyn_alpha 8.0
        --lambda_noisy 0.2
        --lambda_lat_stat 0.05
        --lambda_dynamic_bce 0.3
        --dyn_patch_alpha 8.0
        --dyn_pixel_alpha 12.0
        --lambda_gs_lifespan 0.01
        --lambda_ghost_static 0.0
        --precision bf16
        --seed 0
        --wandb
        --wandb_project dggt-tokenizer
        --wandb_name "t0_v2_stageB_dz1024_gb${ACTUAL_GLOBAL_BATCH}"
    )
}

build_train_args() {
    case "$1" in
        a) build_stage_a_args ;;
        b) build_stage_b_args ;;
        *) echo "[错误] stage 只能是 a 或 b" >&2; return 1 ;;
    esac
}

# ============================================================
# 启动 / 停止
# ============================================================
stage_pid_file()  { echo "${LAUNCH_LOG_DIR}/tokenizer_v2_stage$1_rank$2.pid"; }
stage_log_file()  { echo "${LAUNCH_LOG_DIR}/tokenizer_v2_stage$1_rank$2.log"; }

launch_node() {
    local stage="$1" node_rank="$2" node_name="$3"
    local log_file pid_file
    log_file="$(stage_log_file "${stage}" "${node_rank}")"
    pid_file="$(stage_pid_file "${stage}" "${node_rank}")"

    preflight_check "${node_name}" "${stage}"
    build_train_args "${stage}"
    print_batch_plan "${stage}"

    cd "${PROJECT_ROOT}"

    if [[ -f "${pid_file}" ]]; then
        local old_pid
        old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
            echo "[${node_name}] 已有训练进程运行，PID=${old_pid}（${pid_file}）" >&2
            exit 1
        fi
    fi

    echo "[${node_name}] 启动 torchrun，stage=${stage} node_rank=${node_rank}……"
    echo "[${node_name}] 日志：${log_file}"

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

    sleep 3
    if ! kill -0 "${launch_pid}" 2>/dev/null; then
        echo "[${node_name}] torchrun 启动后立即退出，最近日志：" >&2
        tail -n 100 "${log_file}" >&2 || true
        exit 1
    fi
    echo "[${node_name}] 已启动，PID=${launch_pid}"
}

stop_all() {
    local stage f pid
    for stage in a b; do
        for rank in "${MASTER_RANK}" "${WORKER_RANK}"; do
            f="$(stage_pid_file "${stage}" "${rank}")"
            [[ -f "${f}" ]] || continue
            pid="$(cat "${f}" 2>/dev/null || true)"
            if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
                echo "终止 stage=${stage} rank=${rank} PID=${pid}"; kill "${pid}"
            fi
            rm -f "${f}"
        done
    done
}

launch_both_nodes() {
    local stage="$1"
    LAUNCH_NETWORK_MODE="$(select_launch_network_mode)"
    export NETWORK_MODE="${LAUNCH_NETWORK_MODE}"

    echo "============================================================"
    echo "JointSceneTokenizer T0 Stage-${stage^^}  ${NNODES} 节点 × ${NPROC_PER_NODE} GPU"
    echo "主节点：${MASTER_ADDR} rank=${MASTER_RANK}"
    echo "副节点：${WORKER_USER}@${WORKER_HOST}:${SSH_PORT} rank=${WORKER_RANK}"
    echo "网络模式：${NETWORK_MODE}"
    echo "============================================================"
    print_batch_plan "${stage}"

    echo; echo "=== 1/4 检查主节点 ==="
    preflight_check "主节点 ${MASTER_ADDR}" "${stage}"

    echo; echo "=== 2/4 检查副节点 ==="
    ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
        "test -r '${SCRIPT_PATH}' && NETWORK_MODE='${NETWORK_MODE}' bash '${SCRIPT_PATH}' --check-worker-${stage}"

    echo; echo "=== 3/4 启动副节点 rank ${WORKER_RANK} ==="
    ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
        "NETWORK_MODE='${NETWORK_MODE}' bash '${SCRIPT_PATH}' --worker-${stage}"

    echo; echo "=== 4/4 启动主节点 rank ${MASTER_RANK} ==="
    if ! launch_node "${stage}" "${MASTER_RANK}" "主节点 ${MASTER_ADDR}"; then
        echo "主节点启动失败，终止副节点……" >&2
        ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
            "bash '${SCRIPT_PATH}' --stop" || true
        exit 1
    fi

    echo
    echo "============================================================"
    echo "Stage-${stage^^} 已启动"
    echo "  主节点日志：$(stage_log_file "${stage}" "${MASTER_RANK}")"
    echo "  副节点日志：$(stage_log_file "${stage}" "${WORKER_RANK}")"
    echo
    echo "必看指标（wandb / 日志）："
    echo "  gs_scale_sim_ratio  应从 ~0.80 收敛到 1.0 —— 这是本次改造的验收指标"
    echo "  gs_anchor / geom_anchor / render_anchor 不应显著变差"
    echo "============================================================"
}

# ============================================================
# 入口
# ============================================================
case "${MODE}" in
    --check-ib-ready)
        hdr_ib_ready && { echo "[OK] mlx5_4/5 已就绪。"; exit 0; }
        echo "[WARN] mlx5_4/5 未就绪。"; print_ib_status; exit 1 ;;
    --check-bond-rdma-ready)
        bond_rdma_ready && { echo "[OK] mlx5_bond_0 已就绪。"; exit 0; }
        echo "[WARN] mlx5_bond_0 未就绪。"; print_ib_status; exit 1 ;;

    --plan-a) print_batch_plan a ;;
    --plan-b) print_batch_plan b ;;

    --check-worker-a) preflight_check "副节点 ${WORKER_HOST}" a ;;
    --check-worker-b) preflight_check "副节点 ${WORKER_HOST}" b ;;

    --worker-a) launch_node a "${WORKER_RANK}" "副节点 ${WORKER_HOST}" ;;
    --worker-b) launch_node b "${WORKER_RANK}" "副节点 ${WORKER_HOST}" ;;
    --master-a) launch_node a "${MASTER_RANK}" "主节点 ${MASTER_ADDR}" ;;
    --master-b) launch_node b "${MASTER_RANK}" "主节点 ${MASTER_ADDR}" ;;

    --launch-a) launch_both_nodes a ;;
    --launch-b) launch_both_nodes b ;;

    --stop)
        stop_all
        ssh "${SSH_OPTS[@]}" "${WORKER_USER}@${WORKER_HOST}" \
            "bash '${SCRIPT_PATH}' --stop-local" || true ;;
    --stop-local) stop_all ;;

    *)
        echo "未知参数：${MODE}" >&2
        echo "可用参数：" >&2
        echo "  --launch-a / --launch-b   启动 Stage-A / Stage-B（两机）" >&2
        echo "  --plan-a   / --plan-b     只打印 batch/lr 计划" >&2
        echo "  --stop                    停止两机上的全部 stage" >&2
        echo "  --check-worker-a/-b       只检查当前节点" >&2
        echo "  --worker-a/-b, --master-a/-b  在当前节点启动指定 rank" >&2
        exit 2 ;;
esac
