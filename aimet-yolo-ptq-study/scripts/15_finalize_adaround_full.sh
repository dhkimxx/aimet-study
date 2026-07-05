#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="${RUN_NAME:-aimet_adaround_a8w8_adar256_iter5000_gpu}"
MODEL_PATH="${MODEL_PATH:-results/models/yolo26n_pretrained.aimet_adaround_int8_calib256_iter5000.onnx}"
COVERAGE_CSV="${COVERAGE_CSV:-results/quantization_coverage_adaround_full.csv}"
COVERAGE_JSON="${COVERAGE_JSON:-results/quantization_coverage_adaround_full.json}"
METRICS_CSV="${METRICS_CSV:-results/metrics_quick.csv}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing AdaRound full artifact: ${MODEL_PATH}" >&2
  echo >&2
  scripts/14_adaround_full_status.sh >&2 || true
  exit 1
fi

scripts/run_native.sh python scripts/09_quantization_coverage.py \
  --model "E256=aimet_adaround_a8w8_adar256_iter5000=${MODEL_PATH}" \
  --output-csv "${COVERAGE_CSV}" \
  --output-json "${COVERAGE_JSON}"

echo
echo "Coverage:"
echo "  ${COVERAGE_CSV}"
echo "  ${COVERAGE_JSON}"

echo
echo "Metrics row:"
if [[ -f "${METRICS_CSV}" ]] && grep -F "${RUN_NAME}" "${METRICS_CSV}"; then
  :
else
  echo "No metrics row found for ${RUN_NAME} in ${METRICS_CSV}" >&2
  exit 1
fi

echo
scripts/run_native.sh python scripts/16_summarize_adaround_full.py \
  --metrics-csv "${METRICS_CSV}" \
  --coverage-csv "${COVERAGE_CSV}"
