#!/usr/bin/env bash
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
set -Eeuo pipefail

trap 'rc=$?; echo "[错误] 命令失败：line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

umask 000

# ============================================================
# 阿里云 PAI-DLC：JointSceneTokenizer T0 **v2** 重训（Stage-A / Stage-B）
# 每节点 16 张真武 810E PPU。
#
# DLC 控制台配置：
#   - 每节点规格：ml.gp7vf.16.40xlarge（16 PPU）
#   - 启动命令：
#       Stage-A:  bash /mnt/workspace/dggt/train_tokenizer_ppu_dlc.sh a
#       Stage-B:  bash /mnt/workspace/dggt/train_tokenizer_ppu_dlc.sh b
#       Stage-A 余弦重启延训（可选，默认关闭，见下方 EXTEND_STAGE_A 一节）：
#                 EXTEND_STAGE_A=1 bash /mnt/workspace/dggt/train_tokenizer_ppu_dlc.sh a
#
# DLC 会在每个节点分别执行同一条启动命令，并自动注入：
#   MASTER_ADDR、MASTER_PORT、WORLD_SIZE、RANK、NPROC_PER_NODE
#
# 为什么是 v2（2026-08-01）：
#   审计 encode/decode 往返（tools/retest_scene_flow_gaussian_gauge.py，90 trunk）：
#     paired GS/depth          = 0.7964   （30/30 场景 < 1）
#     depth_recon/depth_direct = 1.0307
#   根因在 train_tokenizer.py：
#     (1) gs_anchor 用一个共同 std 归一化 11 个通道，线性 scale 通道被稀释约 1.2e7 倍；
#     (2) gs_anchor 与 geom_anchor 相互独立，没有项约束它们的比值；
#     (3) geom_anchor 是绝对误差判据，容忍均匀乘性偏移。
#   对应三项新损失：gs_channel_group_huber_loss / --lambda_gs_scale_sim /
#   --lambda_depth_log_bias。架构不变，但旧 checkpoint 与新目标不可比 → 从 0 重训。
#
# 注意：
#   - WORLD_SIZE/RANK 在脚本入口表示 DLC 的节点数和节点 rank；
#     torchrun 启动后会为训练子进程重新定义同名变量；
#   - 不需要 SSH，也不要手工分别启动 master/worker；
#   - PPU PyTorch 通过 torch.cuda 兼容接口暴露设备；
#   - WANDB_API_KEY 建议通过 DLC 自定义环境变量注入。
# ============================================================

STAGE="${1:-}"
case "${STAGE}" in
    a|A) STAGE=a ;;
    b|B) STAGE=b ;;
    "")  echo "[错误] 必须指定 stage：bash $0 a   或   bash $0 b" >&2; exit 2 ;;
    *)   echo "[错误] stage 只能是 a 或 b，收到：${STAGE}" >&2; exit 2 ;;
esac

# ============================================================
# DLC 注入变量：硬性要求，不提供默认值
# 缺失时宁可失败，也不要静默退化成单机跑。
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

DLC_MASTER_ADDR="${MASTER_ADDR}"
DLC_MASTER_PORT="${MASTER_PORT}"
DLC_NNODES="${WORLD_SIZE}"
DLC_NODE_RANK="${RANK}"
DLC_NPROC_PER_NODE="${NPROC_PER_NODE}"

# 不写死节点数：本脚本对 2 节点和 4 节点同样适用，batch 会按总设备数反推。
# 若要把节点数钉死以防控制台配错，显式设置 EXPECTED_NNODES 即可。
EXPECTED_NNODES="${EXPECTED_NNODES:-${DLC_NNODES}}"
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
# 路径与 pretrain_ppu_two_nodes_dlc.sh 完全一致。
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/dggt}"
DATASETS_DIR="${DATASETS_DIR:-/mnt/workspace/datasets}"
DATASET_ROOT="${DATASET_ROOT:-${DATASETS_DIR}/waymo_processed_dggt}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/model}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# lpips 经 torchvision 的 TORCH_HOME 查找 AlexNet backbone。
# train_tokenizer.py 的 render_anchor 用到 LPIPS，因此这份权重是硬依赖。
TORCH_HOME="${TORCH_HOME:-${MODEL_ROOT}/torch}"
ALEXNET_CKPT="${TORCH_HOME}/hub/checkpoints/alexnet-owt-7be5be79.pth"

# ============================================================
# 数据、模型与日志路径
# transfer/raw/asset_root 是 WaymoEditDataset 为编辑数据保留的兼容参数。
# 本脚本固定 clean_only=True，训练样本只从 processed_root 读取，因此这些路径
# 可以不存在；保留参数仅用于兼容既有配置，不把它们列为启动前置条件。
# ============================================================
DGGT_CKPT="${DGGT_CKPT:-${PROJECT_ROOT}/pretrained/model_latest_waymo.pt}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${DATASET_ROOT}}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${DATASET_ROOT}}"
RAW_ROOT="${RAW_ROOT:-${DATASETS_DIR}/waymo}"
ASSET_ROOT="${ASSET_ROOT:-${DATASET_ROOT}/objects_ply_transformed}"
CACHE_MANIFEST="${CACHE_MANIFEST:-${DATASET_ROOT}/waymo_edit_cache/manifests/training/training_manifest.jsonl}"

STAGE_A_LOG_DIR="${STAGE_A_LOG_DIR:-${PROJECT_ROOT}/logs/tokenizer_t0_v2_stageA}"
STAGE_B_LOG_DIR="${STAGE_B_LOG_DIR:-${PROJECT_ROOT}/logs/tokenizer_t0_v2_stageB}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${PROJECT_ROOT}/logs/tokenizer_v2_ppu_dlc_launch}"

STAGE_A_STEPS="${STAGE_A_STEPS:-100000}"
STAGE_B_STEPS="${STAGE_B_STEPS:-40000}"   # A + B = 100000，与「总 iter 10w」一致
STAGE_A_FINAL_CKPT="${STAGE_A_FINAL_CKPT:-${STAGE_A_LOG_DIR}/ckpt/scene_tokenizer_step_$(printf '%06d' "${STAGE_A_STEPS}").pt}"

# ============================================================
# Stage-A 余弦重启延训（EXTEND_STAGE_A=1 时启用，默认关闭）
#
# 背景：step 100000 不是「被中断」，而是**跑完了整条 schedule**。
#   lr_lambda = warmup × 0.5(1+cos(π·step/max_steps))，max_steps=100000
#   ⇒ step 80k 时 LR 已只剩峰值的 9.5%，90k 剩 2.4%，95k 剩 0.6%，100k 精确为 0。
#   round2 评测里 80k→100k 的曲线因此是「退火收敛」，不是「还没学完」。
#
# 因此本块默认关闭：正常路径仍然是 `bash train_tokenizer_ppu_dlc.sh a`
# 从 0 重训，或直接进 Stage-B。只有在明确想做 SGDR 式暖重启时才打开：
#
#   EXTEND_STAGE_A=1 bash train_tokenizer_ppu_dlc.sh a
#
# 重要机制（已在 torch 2.7 上实测确认）：
#   --resume_path 会 scheduler.load_state_dict()，而 LambdaLR 的 state_dict
#   **包含 base_lrs**、且 lr_lambdas 存为 [None]。所以：
#     · 新传的 --lr 会被 checkpoint 里的 base_lrs 静默覆盖 → 调 --lr 无效；
#     · 新的 max_steps 会生效（lambda 是新的闭包）。
#   ⇒ 重启 LR 的唯一旋钮就是 EXTEND_EXTRA_STEPS（决定 cosine 的落点）：
#        +10000 → 峰值的 2.02%
#        +15000 → 峰值的 4.14%
#        +20000 → 峰值的 6.70%
#        +25000 → 峰值的 9.55%（≈ 回到 step 80k 当时的 LR，默认值）
#   默认取 +25000：既然要付重启的代价，就得跳得够高才可能跳出当前盆地；
#   跳太低只是白烧卡时。之后仍按 cosine 退火到 0。
#
# 另外两点：
#   · 写到独立的 EXTEND_LOG_DIR，避免覆盖原 Stage-A 的 config.json 与 ckpt 记录；
#   · 用 --feature_stats_path 复用原 run 的 feature_stats.pt，
#     否则会在新目录重算归一化统计量，latent 尺度一变，resume 就没有意义了。
# ============================================================
EXTEND_STAGE_A="${EXTEND_STAGE_A:-0}"
# 实跑的 Stage-A 日志目录（注意不是本脚本 STAGE_A_LOG_DIR 的默认值：
# round2 评测的 checkpoint_sha256 显示实跑用的是 /mnt/workspace/logs/...，
# 而非 ${PROJECT_ROOT}/logs/...）。
EXTEND_SOURCE_LOG_DIR="${EXTEND_SOURCE_LOG_DIR:-/mnt/workspace/logs/tokenizer_t0_v2_stageA}"
EXTEND_FROM_STEP="${EXTEND_FROM_STEP:-100000}"
EXTEND_EXTRA_STEPS="${EXTEND_EXTRA_STEPS:-25000}"
EXTEND_LOG_DIR="${EXTEND_LOG_DIR:-${EXTEND_SOURCE_LOG_DIR}_ext${EXTEND_EXTRA_STEPS}}"
EXTEND_WANDB_RUN_ID="${EXTEND_WANDB_RUN_ID:-wpdn1ft5}"

RESUME_PATH=""
FEATURE_STATS_PATH=""
WANDB_RUN_ID=""

if [[ "${STAGE}" == "a" ]]; then
    LOG_DIR="${STAGE_A_LOG_DIR}"
    MAX_STEPS="${STAGE_A_STEPS}"
else
    LOG_DIR="${STAGE_B_LOG_DIR}"
    MAX_STEPS="${STAGE_B_STEPS}"
fi

if [[ "${EXTEND_STAGE_A}" == "1" ]]; then
    if [[ "${STAGE}" != "a" ]]; then
        echo "[错误] EXTEND_STAGE_A=1 只能配合 stage a 使用，当前 stage=${STAGE}。" >&2
        exit 2
    fi
    for integer_value in "${EXTEND_FROM_STEP}" "${EXTEND_EXTRA_STEPS}"; do
        if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
            echo "[错误] EXTEND_FROM_STEP / EXTEND_EXTRA_STEPS 必须是非负整数，收到：${integer_value}" >&2
            exit 1
        fi
    done
    if (( EXTEND_EXTRA_STEPS < 1 )); then
        echo "[错误] EXTEND_EXTRA_STEPS 必须为正，收到：${EXTEND_EXTRA_STEPS}" >&2
        exit 1
    fi

    RESUME_PATH="${RESUME_PATH_OVERRIDE:-${EXTEND_SOURCE_LOG_DIR}/ckpt/scene_tokenizer_step_$(printf '%06d' "${EXTEND_FROM_STEP}").pt}"
    FEATURE_STATS_PATH="${FEATURE_STATS_PATH_OVERRIDE:-${EXTEND_SOURCE_LOG_DIR}/feature_stats.pt}"
    LOG_DIR="${EXTEND_LOG_DIR}"
    MAX_STEPS=$(( EXTEND_FROM_STEP + EXTEND_EXTRA_STEPS ))
    WANDB_RUN_ID="${EXTEND_WANDB_RUN_ID}"
fi

# ============================================================
# Batch 规划
#
# 数值取自**实跑的** logs/tokenizer_t0_stage{A,B}/config.json（不是 train_tokenizer.py
# docstring 里的示例 —— 那份示例 Stage-B 写的是 batch 4 / accum 2，与实跑不符）：
#   Stage-A 实跑: 2 gpu × batch 1 × accum 8 = 16
#   Stage-B 实跑: 2 gpu × batch 1 × accum 8 = 16
#
# 本次只改损失、不改 batch —— 同时改两样会让「新目标是否有效」无法归因。
# 按总设备数反推 per-device batch 与 accum，把全局 batch 尽量钉在 16。
# 2 节点 × 16 PPU = 32 设备时，即使 batch=1/accum=1 也已达 32，
# 此时接受更大的全局 batch，并按 sqrt 规则缩放 LR（AdamW 上比线性规则稳）。
# ============================================================
REFERENCE_GLOBAL_BATCH=16
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-16}"

# per-device batch 上限。两个 stage 实跑都是 1；PPU 显存小于 80GB A100，不上调。
STAGE_A_MAX_BATCH_PER_DEV="${STAGE_A_MAX_BATCH_PER_DEV:-1}"
STAGE_B_MAX_BATCH_PER_DEV="${STAGE_B_MAX_BATCH_PER_DEV:-1}"

STAGE_A_REF_LR="${STAGE_A_REF_LR:-3e-4}"
STAGE_B_REF_LR="${STAGE_B_REF_LR:-8e-5}"

derive_batch_plan() {
    local stage="$1"
    local world_size=$(( DLC_NNODES * DLC_NPROC_PER_NODE ))
    local max_per_dev ref_lr

    case "${stage}" in
        a) max_per_dev="${STAGE_A_MAX_BATCH_PER_DEV}"; ref_lr="${STAGE_A_REF_LR}" ;;
        b) max_per_dev="${STAGE_B_MAX_BATCH_PER_DEV}"; ref_lr="${STAGE_B_REF_LR}" ;;
    esac

    local per_dev="${max_per_dev}"
    while (( per_dev > 1 && world_size * per_dev > TARGET_GLOBAL_BATCH )); do
        per_dev=$(( per_dev / 2 ))
    done

    local samples_per_step=$(( world_size * per_dev ))
    local accum=$(( (TARGET_GLOBAL_BATCH + samples_per_step - 1) / samples_per_step ))
    (( accum < 1 )) && accum=1

    BATCH_PER_DEV="${per_dev}"
    GRAD_ACCUM_STEPS="${accum}"
    ACTUAL_GLOBAL_BATCH=$(( samples_per_step * accum ))
    SCALED_LR="$(awk -v lr="${ref_lr}" -v a="${ACTUAL_GLOBAL_BATCH}" -v r="${REFERENCE_GLOBAL_BATCH}" \
        'BEGIN { printf "%.6g", lr * sqrt(a / r) }')"
    WORLD_SIZE_TOTAL="${world_size}"
}

derive_batch_plan "${STAGE}"

# 与 pretrain_ppu_two_nodes_dlc.sh 一样保留一个显式护栏：
# 设置 EXPECTED_GLOBAL_BATCH_SIZE 后，反推结果必须与之一致，否则直接失败。
if [[ -n "${EXPECTED_GLOBAL_BATCH_SIZE:-}" ]]; then
    if (( ACTUAL_GLOBAL_BATCH != EXPECTED_GLOBAL_BATCH_SIZE )); then
        echo "[错误] global batch size 配置不符合预期：" >&2
        echo "       ${ACTUAL_GLOBAL_BATCH} = ${DLC_NNODES} nodes × ${DLC_NPROC_PER_NODE} ppu/node × ${BATCH_PER_DEV} batch/ppu × ${GRAD_ACCUM_STEPS} accum" >&2
        echo "       期望 EXPECTED_GLOBAL_BATCH_SIZE=${EXPECTED_GLOBAL_BATCH_SIZE}" >&2
        exit 1
    fi
fi

WANDB_PROJECT="${WANDB_PROJECT:-dggt-tokenizer}"
WANDB_NAME="${WANDB_NAME:-t0_v2_stage${STAGE^^}_dz1024_ppu_gb${ACTUAL_GLOBAL_BATCH}}"
# train_tokenizer.py 在收到 --wandb_run_id 时会自动带上 resume="allow"。
# 注意：WANDB_RUN_DESC 只是给下面 banner 打印用的展示值，并没有 export 任何
# WANDB_* 变量，不会和 wandb.init(resume=...) 冲突。
if [[ -n "${WANDB_RUN_ID}" ]]; then
    WANDB_RUN_DESC="resume=allow, id=${WANDB_RUN_ID}"
else
    WANDB_RUN_DESC="new run (resume=never)"
fi

# ============================================================
# 运行环境（PPU / ACCL-P）
# ============================================================
setup_common_env() {
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES
    export TORCH_HOME

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

    # PAI-PPU ACCL-P 官方最佳实践为 eth0、NCCL_IB_HCA=""、NCCL_IB_DISABLE=1。
    # PAI-PPU 官方镜像已内置这些值，因此优先保留镜像/平台注入值，
    # 仅在变量缺失时补齐官方默认值。
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    # PAI 官方镜像可能预置 NCCL_DEBUG=INFO；训练时固定为 WARN。
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

    # 保留 PAI-DLC/PPU 官方镜像预置的其余高性能网络配置，不在这里覆盖
    # 其他 NCCL_*、PCCL_* 或 ACCL-P 相关变量。

    # 保留 PyTorch 默认 SDPA 调度。仅将 LearnedQueryPool 按 batch 分块，
    # 避开真武 810E 对大 flattened batch 的 fused MHA kernel 限制。
    # LearnedQueryPool 就在 dggt/models/joint_scene_tokenizer.py:491,635 —
    # 也就是本次要训练的模块本身，这两个变量对 T0 是硬依赖。
    export DGGT_DEVICE_BACKEND="${DGGT_DEVICE_BACKEND:-ppu}"
    export DGGT_PPU_MHA_BATCH_CHUNK_SIZE="${DGGT_PPU_MHA_BATCH_CHUNK_SIZE:-4096}"

    unset WANDB_DISABLED || true
}

check_dir() {
    local label="$1" path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "[错误] 目录不存在：${label}" >&2; echo "       ${path}" >&2; exit 1
    fi
    echo "[OK] ${label}: ${path}"
}

check_file() {
    local label="$1" path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "[错误] 文件不存在：${label}" >&2; echo "       ${path}" >&2; exit 1
    fi
    echo "[OK] ${label}: ${path}"
}

check_required_paths() {
    echo "[DLC rank ${DLC_NODE_RANK}] 检查本节点文件和目录……"

    check_dir  "PROJECT_ROOT"       "${PROJECT_ROOT}"
    check_file "train_tokenizer.py" "${PROJECT_ROOT}/train_tokenizer.py"
    check_file "DGGT_CKPT"          "${DGGT_CKPT}"
    check_dir  "PROCESSED_ROOT"     "${PROCESSED_ROOT}"

    if [[ "${STAGE}" == "b" ]]; then
        check_file "CACHE_MANIFEST"     "${CACHE_MANIFEST}"
        check_file "STAGE_A_FINAL_CKPT" "${STAGE_A_FINAL_CKPT}"
    fi

    if [[ -n "${RESUME_PATH}" ]]; then
        check_file "RESUME_PATH"         "${RESUME_PATH}"
        # 必须复用原 run 的 feature_stats.pt：新目录下重算会改变 latent 归一化尺度。
        check_file "FEATURE_STATS_PATH"  "${FEATURE_STATS_PATH}"
    fi

    # render_anchor 用 LPIPS，两个 stage 都需要（Stage-B 从 step 0 就开着）。
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

# 读出 resume checkpoint 里真实的 global_step 与 scheduler.base_lrs，
# 据此算出重启瞬间的实际 LR。不要靠猜：resume 时 --lr 会被 base_lrs 覆盖，
# 只有把 base_lrs 打印出来，日志里的 LR 才是可信的。
inspect_resume_checkpoint() {
    RESUME_PATH="${RESUME_PATH}" \
    EXPECT_STEP="${EXTEND_FROM_STEP}" \
    EXTEND_MAX_STEPS="${MAX_STEPS}" \
    NOMINAL_LR="${SCALED_LR}" \
        "${PYTHON_BIN}" - <<'PY'
import math
import os
import torch

path = os.environ["RESUME_PATH"]
expect_step = int(os.environ["EXPECT_STEP"])
max_steps = int(os.environ["EXTEND_MAX_STEPS"])
nominal_lr = float(os.environ["NOMINAL_LR"])

payload = torch.load(path, map_location="cpu", weights_only=False)
required = {"scene_tokenizer", "optimizer", "scheduler", "global_step"}
missing = sorted(required.difference(payload))
if missing:
    raise RuntimeError(f"{path} 不是完整训练 checkpoint，缺少 {missing}，无法用于 --resume_path")

step = int(payload["global_step"])
if step != expect_step:
    raise RuntimeError(f"checkpoint global_step={step}，与 EXTEND_FROM_STEP={expect_step} 不一致")
if step >= max_steps:
    raise RuntimeError(f"global_step={step} 已达 max_steps={max_steps}，训练循环会直接退出，不会跑任何一步")

base_lr = float(payload["scheduler"]["base_lrs"][0])
factor = 0.5 * (1.0 + math.cos(math.pi * min(step / max(max_steps, 1), 1.0)))

print(f"[extend] resume checkpoint : {path}")
print(f"[extend] global_step       : {step}  →  max_steps {max_steps}（新增 {max_steps - step} 步）")
print(f"[extend] scheduler base_lr : {base_lr:.6g}   ← resume 时以它为准")
print(f"[extend] 本次传入的 --lr   : {nominal_lr:.6g}   ← 会被上面的 base_lr 覆盖，仅作记录")
print(f"[extend] 重启瞬间 LR       : {base_lr * factor:.6g}  = base_lr × {factor:.4f}（峰值的 {factor * 100:.2f}%）")
print(f"[extend] 之后按 cosine 退火，到 step {max_steps} 归零")

if abs(base_lr - nominal_lr) / max(base_lr, 1e-12) > 0.01:
    print(
        "[extend][警告] checkpoint 的 base_lr 与本次反推的 SCALED_LR 相差超过 1%，"
        "通常意味着这次的节点数/全局 batch 与原 run 不同。"
        "LR 会沿用原 run 的值，与新 batch 不匹配 —— 请把节点数改回原配置。"
    )
PY
}

# ============================================================
# 训练参数
# 除 batch/accum/lr/max_steps 与三项新损失外，逐项对齐
# logs/tokenizer_t0_stage{A,B}/config.json。
# ============================================================
build_train_args() {
    local common=(
        train_tokenizer.py
        --ckpt_path "${DGGT_CKPT}"
        --log_dir "${LOG_DIR}"
        --processed_root "${PROCESSED_ROOT}"
        --transfer_root "${TRANSFER_ROOT}"
        --raw_root "${RAW_ROOT}"
        --asset_root "${ASSET_ROOT}"
        --views 1
        --sample_window 20
        --batch_size "${BATCH_PER_DEV}"
        --grad_accum_steps "${GRAD_ACCUM_STEPS}"
        --mp_sharing_strategy file_system
        --max_steps "${MAX_STEPS}"
        --save_every 5000
        --log_every 100
        --lr "${SCALED_LR}"
        --weight_decay 0.05
        --grad_clip_norm 1.0
        --decoder_noise_tau 0.8
        --decoder_noise_distribution uniform
        --gs_scale_sim_opacity 0.05
        --lambda_lat_stat 0.05
        --lambda_gs_lifespan 0.01
        --lambda_ghost_static 0.0
        --precision bf16
        --wandb
        --wandb_project "${WANDB_PROJECT}"
        --wandb_name "${WANDB_NAME}"
    )

    if [[ -n "${RESUME_PATH}" ]]; then
        # --resume_path 会一并恢复 optimizer / scheduler / global_step，
        # wandb 的 step 因此从 100000 继续递增，不会被 wandb 当作回退而丢弃。
        common+=(
            --resume_path "${RESUME_PATH}"
            --feature_stats_path "${FEATURE_STATS_PATH}"
        )
    fi
    if [[ -n "${WANDB_RUN_ID}" ]]; then
        common+=(--wandb_run_id "${WANDB_RUN_ID}")
    fi

    if [[ "${STAGE}" == "a" ]]; then
        TRAIN_ARGS=(
            "${common[@]}"
            --min_frames 10
            --max_frames 14
            --num_workers 8
            --prefetch_factor 4
            --vis_every 2500
            --stats_steps 2048
            --warmup_steps 4000
            --head_start_step 4000
            --head_warmup_steps 6000
            --render_start_step 10000
            --noisy_start_step 25000
            --lambda_tok_rec 1.0
            --lambda_tok_cos 0.2
            --lambda_head_anchor 0.6
            --lambda_gs_scale_sim 0.3
            --lambda_depth_log_bias 0.2
            --lambda_render_anchor 0.3
            --gt_render_ratio 1.0
            --render_dyn_alpha 6.0
            --lambda_noisy 0.15
            --lambda_dynamic_bce 0.2
            --dyn_patch_alpha 6.0
            --dyn_pixel_alpha 10.0
            --seed 0
        )
    else
        TRAIN_ARGS=(
            "${common[@]}"
            --init_tokenizer_path "${STAGE_A_FINAL_CKPT}"
            --cache_manifest_path "${CACHE_MANIFEST}"
            --cache_split training
            --stage_b_mix_raw
            --min_frames 10
            --max_frames 14
            --raw_batch_size "${BATCH_PER_DEV}"
            --num_workers 8
            --prefetch_factor 4
            --vis_every 2500
            --stats_steps 512
            --warmup_steps 1000
            --head_start_step 0
            --head_warmup_steps 1
            --render_start_step 0
            --noisy_start_step 0
            --lambda_tok_rec 0.5
            --lambda_tok_cos 0.1
            --lambda_head_anchor 0.8
            --lambda_gs_scale_sim 0.5
            --lambda_depth_log_bias 0.3
            --lambda_render_anchor 0.5
            --gt_render_ratio 1.5
            --render_dyn_alpha 8.0
            --lambda_noisy 0.2
            --lambda_dynamic_bce 0.3
            --dyn_patch_alpha 8.0
            --dyn_pixel_alpha 12.0
            --seed 0
        )
    fi
}

setup_common_env
check_required_paths
check_python_and_ppu
if [[ -n "${RESUME_PATH}" ]]; then
    inspect_resume_checkpoint
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[警告] 未检测到 WANDB_API_KEY；仅当镜像中已有 wandb 登录凭证时才能正常上传。" >&2
    echo "       建议在 DLC 任务的自定义环境变量中配置 WANDB_API_KEY。" >&2
fi

cd "${PROJECT_ROOT}"
build_train_args

LAUNCH_LOG="${LAUNCH_LOG_DIR}/tokenizer_v2_stage${STAGE}_rank${DLC_NODE_RANK}.log"

echo "============================================================"
if [[ -n "${RESUME_PATH}" ]]; then
    echo "PAI-DLC ${DLC_NNODES} 节点 PPU：Tokenizer T0 v2 Stage-A **余弦重启延训**"
else
    echo "PAI-DLC ${DLC_NNODES} 节点 PPU：JointSceneTokenizer T0 v2 Stage-${STAGE^^}"
fi
echo "DLC node rank: ${DLC_NODE_RANK}/${DLC_NNODES}"
echo "master: ${DLC_MASTER_ADDR}:${DLC_MASTER_PORT}"
echo "PPU per node: ${DLC_NPROC_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "TORCH_HOME: ${TORCH_HOME}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "NCCL_IB_HCA: ${NCCL_IB_HCA}"
echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
echo "DGGT_DEVICE_BACKEND: ${DGGT_DEVICE_BACKEND}"
echo "DGGT_PPU_MHA_BATCH_CHUNK_SIZE: ${DGGT_PPU_MHA_BATCH_CHUNK_SIZE}"
echo "------------------------------------------------------------"
echo "global batch size: ${ACTUAL_GLOBAL_BATCH} = ${DLC_NNODES} nodes × ${DLC_NPROC_PER_NODE} ppu/node × ${BATCH_PER_DEV} batch/ppu × ${GRAD_ACCUM_STEPS} accum"
echo "  参考配方 ${REFERENCE_GLOBAL_BATCH}（实跑 config.json），目标 ${TARGET_GLOBAL_BATCH}"
echo "lr: ${SCALED_LR}   max_steps: ${MAX_STEPS}"
if (( ACTUAL_GLOBAL_BATCH != REFERENCE_GLOBAL_BATCH )); then
    echo "  [注意] 全局 batch 偏离参考配方，LR 已按 sqrt 规则缩放。"
    echo "         要严格复现参考配方，需把总设备数降到 ${REFERENCE_GLOBAL_BATCH} 或以下。"
fi
echo "------------------------------------------------------------"
echo "training log dir: ${LOG_DIR}"
echo "launch log: ${LAUNCH_LOG}"
echo "wandb: ${WANDB_PROJECT}/${WANDB_NAME}, ${WANDB_RUN_DESC}"
if [[ -n "${RESUME_PATH}" ]]; then
    echo "resume_path: ${RESUME_PATH}"
    echo "feature_stats（复用原 run，不重算）: ${FEATURE_STATS_PATH}"
    echo
    echo "验收指标（wandb，与 step 100000 的 round2 评测基线对比）："
    echo "  守住不许退：gs_scale_sim_ratio 与 depth_ratio 已在 0.999，任何一个跌出"
    echo "              [0.995, 1.005] 就说明重启把 v2 已经拿到的东西打坏了，应当停。"
    echo "  期望改善：  gs_axis_anisotropy（占 recovery score 的 79%，基线 0.02855）"
    echo "              以及 render_anchor / geom_anchor。"
    echo "  注意：round2 里 80k→100k 的 render_gt_psnr/ssim/lpips 与 lidar_recon_*"
    echo "        已经完全走平（甚至微降），所以这几项**不要**期待有提升；"
    echo "        延训若在这些指标上无变化，属于预期内，不代表跑错。"
else
    echo
    echo "验收指标（wandb）："
    echo "  gs_scale_sim_ratio  应从 ~0.80 收敛到 1.0"
    echo "  depth_ratio         应从 ~1.03 收敛到 1.0"
    echo "  gs_anchor / geom_anchor / render_anchor 不应显著变差"
fi
echo "============================================================"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${DLC_NNODES}" \
    --node_rank="${DLC_NODE_RANK}" \
    --nproc_per_node="${DLC_NPROC_PER_NODE}" \
    --master_addr="${DLC_MASTER_ADDR}" \
    --master_port="${DLC_MASTER_PORT}" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LAUNCH_LOG}"
