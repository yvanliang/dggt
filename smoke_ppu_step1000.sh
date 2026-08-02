#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[PPU smoke 错误] line ${LINENO}: ${BASH_COMMAND}，exit_code=${rc}" >&2' ERR

# Run the actual pretraining stack through one optimizer step and the complete
# first scheduled validation (validation loss, CFG sampling, and RGB render).
# This is intentionally a launcher around pretrain_ppu.sh so model, data,
# autocast, DDP, optimizer, EMA, and rendering behavior cannot drift from the
# production job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_TARGET_STEP="${SMOKE_TARGET_STEP:-1000}"
SMOKE_RESUME_PATH="${SMOKE_RESUME_PATH:-}"
SMOKE_WARM_START_PATH="${SMOKE_WARM_START_PATH:-}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${DGGT_DEVICE_BACKEND:-ppu}" != "ppu" ]]; then
    echo "[错误] 此脚本仅用于 PPU；DGGT_DEVICE_BACKEND 必须为 ppu。" >&2
    exit 1
fi
if [[ -n "${SMOKE_RESUME_PATH}" && -n "${SMOKE_WARM_START_PATH}" ]]; then
    echo "[错误] SMOKE_RESUME_PATH 和 SMOKE_WARM_START_PATH 不能同时设置。" >&2
    exit 1
fi
if ! [[ "${SMOKE_TARGET_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[错误] SMOKE_TARGET_STEP 必须是正整数。" >&2
    exit 1
fi

export PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
export DGGT_DEVICE_BACKEND=ppu
export WANDB_DISABLED=true
# This smoke test targets the user's one-PPU machine by default.  The shared
# production launcher remains two-card by default; these exported values only
# affect this smoke invocation and can still be overridden by the caller.
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Use the Python installed in the PPU image directly.
export PYTHON_BIN="${PYTHON_BIN:-python}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppu_step1000_smoke/${SMOKE_RUN_ID}}"
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${LOG_DIR}/launch}"
export SAVE_EVERY="${SAVE_EVERY:-1}"
export VAL_BATCHES="${VAL_BATCHES:-1}"
export VAL_LOG_IMAGES="${VAL_LOG_IMAGES:-10}"
export VAL_SAMPLE_STEPS="${VAL_SAMPLE_STEPS:-35}"
export VAL_GUIDANCE_SCALES="${VAL_GUIDANCE_SCALES:-1.0,2.0,4.0}"

if [[ -n "${SMOKE_RESUME_PATH}" ]]; then
    # Exact mode: a full step-(target-1) checkpoint performs precisely the
    # optimizer update that reaches target, followed by its scheduled val.
    expected_step=$((SMOKE_TARGET_STEP - 1))
    "${PYTHON_BIN}" -c '
import sys
import torch

path, expected_text = sys.argv[1], sys.argv[2]
payload = torch.load(path, map_location="cpu")
actual = int(payload.get("step", -1)) if isinstance(payload, dict) else -1
expected = int(expected_text)
if actual != expected:
    raise SystemExit(
        f"checkpoint step mismatch: expected {expected}, got {actual} ({path})"
    )
print(f"[OK] exact-resume checkpoint step={actual}: {path}")
' "${SMOKE_RESUME_PATH}" "${expected_step}"
    export RESUME_PATH="${SMOKE_RESUME_PATH}"
    unset WARM_START_PATH || true
    export MAX_STEPS="${SMOKE_TARGET_STEP}"
    export VAL_EVERY="${SMOKE_TARGET_STEP}"
    smoke_mode="exact resume: step ${expected_step} -> ${SMOKE_TARGET_STEP}"
else
    # Fast-equivalent mode: step 1 with val_every=1 has validation_index=0,
    # exactly like the production job's first validation at step 1000.  It
    # exercises the same data, backward/update, EMA, CFG, and RGB-render path.
    unset RESUME_PATH || true
    if [[ -n "${SMOKE_WARM_START_PATH}" ]]; then
        export WARM_START_PATH="${SMOKE_WARM_START_PATH}"
    else
        unset WARM_START_PATH || true
    fi
    export MAX_STEPS=1
    export VAL_EVERY=1
    smoke_mode="fast equivalent: step 0 -> 1, first-validation path"
fi

echo "=== PPU post-step smoke test ==="
echo "mode: ${smoke_mode}"
echo "python: ${PYTHON_BIN}"
echo "PPU topology: ${NNODES} node x ${NPROC_PER_NODE} device; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "log dir: ${LOG_DIR}"
echo "validation: batches=${VAL_BATCHES}, sample_steps=${VAL_SAMPLE_STEPS}, guidance_scales=${VAL_GUIDANCE_SCALES}"

bash "${PROJECT_ROOT}/pretrain_ppu.sh"

echo "[PASS] 完整训练 step、反向传播、optimizer/EMA 更新、CFG sampling 和 RGB render 均已完成。"
echo "日志与渲染结果：${LOG_DIR}"
