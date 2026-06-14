#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AIMET_IMAGE="${AIMET_IMAGE:-aimet-yolo-onnx-gpu:2.2.0}"
AIMET_CONTAINER_NAME="${AIMET_CONTAINER_NAME:-aimet-yolo-ptq-study}"

echo "Project root: ${PROJECT_ROOT}"
echo "AIMET image:  ${AIMET_IMAGE}"
echo "Container:    ${AIMET_CONTAINER_NAME}"

docker run --rm -it \
  --gpus all \
  --name "${AIMET_CONTAINER_NAME}" \
  --ipc=host \
  --shm-size=8G \
  --network=host \
  -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  -w "${PROJECT_ROOT}" \
  "${AIMET_IMAGE}" \
  /bin/bash
