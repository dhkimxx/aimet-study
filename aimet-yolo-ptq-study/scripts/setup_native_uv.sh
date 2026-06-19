#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${PROJECT_ROOT}/.uv-python}"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is not installed or not on PATH.

Install uv in WSL2 Ubuntu, then rerun this script:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
EOF
  exit 1
fi

uv python install 3.10 --install-dir "${UV_PYTHON_INSTALL_DIR}"
uv sync
scripts/run_native.sh python scripts/00_check_env.py
