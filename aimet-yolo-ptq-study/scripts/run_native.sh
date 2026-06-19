#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${PROJECT_ROOT}/.uv-python}"

PYTHONPATH_ENTRY="${PROJECT_ROOT}/.venv/lib/python3.10/site-packages"
CUDA_LIB_DIRS=(
  "${PYTHONPATH_ENTRY}/torch/lib"
  "${PYTHONPATH_ENTRY}/nvidia/cublas/lib"
  "${PYTHONPATH_ENTRY}/nvidia/cuda_runtime/lib"
  "${PYTHONPATH_ENTRY}/nvidia/cudnn/lib"
  "${PYTHONPATH_ENTRY}/nvidia/cufft/lib"
  "${PYTHONPATH_ENTRY}/nvidia/curand/lib"
)

for lib_dir in "${CUDA_LIB_DIRS[@]}"; do
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  fi
done

exec uv run --frozen "$@"
