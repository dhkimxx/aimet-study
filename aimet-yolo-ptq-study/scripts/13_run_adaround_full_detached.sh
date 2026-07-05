#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="${RUN_NAME:-aimet_adaround_a8w8_adar256_iter5000_gpu}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-1}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-256}"
ADAROUND_SAMPLES="${ADAROUND_SAMPLES:-256}"
ADAROUND_ITERATIONS="${ADAROUND_ITERATIONS:-5000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-20260614}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/results/logs}"
PID_DIR="${PID_DIR:-${PROJECT_ROOT}/results/pids}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${STAMP}.log}"
PID_FILE="${PID_FILE:-${PID_DIR}/${RUN_NAME}.pid}"

if pgrep -af "[s]cripts/06_aimet_adaround_ptq.py .*--name ${RUN_NAME}" >/dev/null; then
  echo "AdaRound run already appears to be active for name=${RUN_NAME}" >&2
  pgrep -af "[s]cripts/06_aimet_adaround_ptq.py .*--name ${RUN_NAME}" >&2
  exit 1
fi

cmd=(
  scripts/run_native.sh
  python
  scripts/06_aimet_adaround_ptq.py
  --device "${DEVICE}"
  --batch "${BATCH}"
  --calibration-samples "${CALIBRATION_SAMPLES}"
  --adaround-samples "${ADAROUND_SAMPLES}"
  --adaround-iterations "${ADAROUND_ITERATIONS}"
  --eval-samples "${EVAL_SAMPLES}"
  --eval-seed "${EVAL_SEED}"
  --name "${RUN_NAME}"
)

if [[ "${FORCE}" == "1" ]]; then
  cmd+=(--force)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  echo
  echo "PID file: ${PID_FILE}"
  echo "Log file: ${LOG_FILE}"
  exit 0
fi

{
  echo "[$(date --iso-8601=seconds)] starting detached AdaRound"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  echo
  echo
} >>"${LOG_FILE}"

nohup env PYTHONUNBUFFERED=1 "${cmd[@]}" >>"${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" >"${PID_FILE}"

echo "Started ${RUN_NAME}"
echo "PID: ${pid}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
