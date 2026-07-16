#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_INDEX_URL="${PPU_INDEX_URL:-https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-mirrors.aliyun.com}"
INSTALL_POINTOPS2="${INSTALL_POINTOPS2:-1}"
INSTALL_SPLATFORMER="${INSTALL_SPLATFORMER:-0}"

cd "${ROOT_DIR}"

python - <<'PY'
import sys
import torch

version = torch.__version__
if not version.startswith("2.9."):
    raise SystemExit(f"Expected preinstalled PPU torch 2.9.x, got {version}")
print("preinstalled torch:", version)
print("torch cuda:", torch.version.cuda)
PY

# Install torch-bound PPU wheels from the PPU package index only. Do not mix
# gsplat/torchvision/torchaudio with upstream PyPI CUDA wheels. Use
# --no-build-isolation because some PPU packages, including gsplat, import the
# already-installed torch while preparing build metadata.
python -m pip install \
  --index-url "${PPU_INDEX_URL}" \
  --no-build-isolation \
  --no-cache-dir \
  -r requirements_ppu/requirements-ppu-torch29-ppu-wheels.txt

if [[ "${INSTALL_SPLATFORMER}" == "1" ]]; then
  python -m pip install \
    --index-url "${PPU_INDEX_URL}" \
    --no-build-isolation \
    --no-cache-dir \
    -r requirements_ppu/requirements-ppu-torch29-optional-splatformer.txt
fi

# Install the platform-neutral dependencies from PyPI.
python -m pip install \
  --index-url "${PYPI_INDEX_URL}" \
  --trusted-host "${PYPI_TRUSTED_HOST}" \
  --no-cache-dir \
  -r requirements_ppu/requirements-ppu-torch29.txt

if [[ "${INSTALL_POINTOPS2}" == "1" ]]; then
  # PPU torch 2.9 images can be cu129/cu130-based. Respect an existing
  # CUDA_HOME, but pick common locations when it is not set.
  if [[ -z "${CUDA_HOME:-}" ]]; then
    for candidate in /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda /opt/cuda; do
      if [[ -x "${candidate}/bin/nvcc" ]]; then
        export CUDA_HOME="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA_HOME is not set to a toolkit containing bin/nvcc; cannot compile third_party/pointops2." >&2
    echo "Set CUDA_HOME to the PPU SDK CUDA toolkit path, or rerun with INSTALL_POINTOPS2=0." >&2
    exit 1
  fi

  export PATH="${CUDA_HOME}/bin:${PATH}"
  export MAX_JOBS="${MAX_JOBS:-8}"

  # Do not inherit NVIDIA-specific arch lists from the original README. The PPU
  # toolchain should choose the target for PPU-ZW810E through its cu129 setup.
  unset TORCH_CUDA_ARCH_LIST

  python -m pip install --no-build-isolation --no-cache-dir ./third_party/pointops2
fi

python - <<'PY'
import torch
import torchvision
import gsplat
import ctypes.util

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("gsplat:", getattr(gsplat, "__version__", "unknown"))
if ctypes.util.find_library("GL") is None:
    print("WARNING: libGL.so.1 was not found. Open3D/datasets imports may fail.")
    print("Ubuntu/Debian fix: apt-get update && apt-get install -y libgl1 libglib2.0-0")
PY
