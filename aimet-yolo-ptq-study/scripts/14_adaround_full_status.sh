#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="${RUN_NAME:-aimet_adaround_a8w8_adar256_iter5000_gpu}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/results/logs}"
PID_DIR="${PID_DIR:-${PROJECT_ROOT}/results/pids}"
LINES="${LINES:-24}"

PID_FILE="${PID_DIR}/${RUN_NAME}.pid"
TMUX_SESSION_FILE="${PID_DIR}/${RUN_NAME}.tmux"
MODEL_PATH="${PROJECT_ROOT}/results/models/yolo26n_pretrained.aimet_adaround_int8_calib256_adar256_iter5000.onnx"
METRICS_CSV="${PROJECT_ROOT}/results/metrics_quick.csv"

latest_log="${LOG_FILE:-}"
if [[ -z "${latest_log}" ]]; then
  latest_log="$(
    find "${LOG_DIR}" -maxdepth 1 -type f -name "${RUN_NAME}_*.log" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
  )"
fi

echo "Run: ${RUN_NAME}"

if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
  echo "Log: ${latest_log}"
else
  echo "Log: not found"
fi

if [[ -f "${TMUX_SESSION_FILE}" ]]; then
  session="$(<"${TMUX_SESSION_FILE}")"
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux: ${session} (alive)"
  else
    echo "tmux: ${session} (not found or inaccessible)"
  fi
else
  echo "tmux: session file not found"
fi

if [[ -f "${PID_FILE}" ]]; then
  pid="$(<"${PID_FILE}")"
  echo "PID file: ${PID_FILE} (${pid})"
  ps -p "${pid}" -o pid,stat,etimes,pcpu,pmem,args || true
else
  echo "PID file: not found"
fi

echo
echo "Matching processes:"
pgrep -af "[s]cripts/06_aimet_adaround_ptq.py|${RUN_NAME}" || true

echo
echo "Latest meaningful log lines:"
if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
  tr '\r' '\n' <"${latest_log}" \
    | perl -pe 's/\e\[[0-9;]*[A-Za-z]//g' \
    | grep -v 'onnxruntime' \
    | grep -E 'Started Optimizing|[0-9]+%|Evaluating|Export|metrics|Traceback|Error|Exception' \
    | tail -n "${LINES}" || true
else
  echo "No log to parse."
fi

echo
echo "Expected artifact:"
if [[ -f "${MODEL_PATH}" ]]; then
  ls -lh "${MODEL_PATH}"
else
  echo "Not created yet: ${MODEL_PATH}"
fi

echo
echo "Metrics row:"
if [[ -f "${METRICS_CSV}" ]]; then
  grep -F "${RUN_NAME}" "${METRICS_CSV}" || echo "No row yet."
else
  echo "Metrics CSV not found: ${METRICS_CSV}"
fi
