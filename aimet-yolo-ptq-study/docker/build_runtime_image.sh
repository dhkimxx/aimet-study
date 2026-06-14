#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-aimet-yolo-onnx-gpu:2.2.0}"

echo "Project root:  ${PROJECT_ROOT}"
echo "Runtime image: ${RUNTIME_IMAGE}"

docker build \
  -f "${PROJECT_ROOT}/docker/Dockerfile.runtime" \
  -t "${RUNTIME_IMAGE}" \
  "${PROJECT_ROOT}"
