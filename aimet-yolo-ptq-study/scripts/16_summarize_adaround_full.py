"""Print a report-ready summary for the full AdaRound result."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", default="results/metrics_quick.csv")
    parser.add_argument("--coverage-csv", default="results/quantization_coverage_adaround_full.csv")
    parser.add_argument("--experiment-name", default="aimet_adaround_a8w8_adar256_iter5000_gpu_sample500")
    parser.add_argument("--fp32-name", default="fp32_onnx_sample500")
    parser.add_argument("--a8w8-name", default="aimet_quantsim_a8w8_gpu_sample500")
    parser.add_argument("--mid-adaround-name", default="aimet_adaround_a8w8_adar128_iter2000_gpu_sample500")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_last(rows: list[dict[str, str]], experiment_name: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("experiment_name") == experiment_name]
    if not matches:
        raise SystemExit(f"Missing row for experiment_name={experiment_name}")
    return matches[-1]


def find_coverage(rows: list[dict[str, str]], target_metric: dict[str, str]) -> dict[str, str]:
    target_sha = target_metric.get("model_sha256")
    matches = [row for row in rows if row.get("model_sha256") == target_sha]
    if matches:
        return matches[-1]
    if len(rows) == 1:
        return rows[0]
    raise SystemExit(f"Missing coverage row for model_sha256={target_sha}")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise SystemExit(f"Missing numeric field {key} in row {row.get('experiment_name')}")
    return float(value)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def delta(value: float, baseline: float) -> str:
    return f"{value - baseline:+.4f}"


def coverage_summary(row: dict[str, str]) -> str:
    def field(name: str, default: str = "-") -> str:
        return row.get(name) or default

    return (
        f"Q/DQ {field('quantize_linear_nodes')}/{field('dequantize_linear_nodes')}, "
        f"Conv weight QDQ {field('conv_weight_qdq_count')}/{field('conv_total')}, "
        f"Conv weight INT storage {field('conv_weight_int_storage_count')}/{field('conv_total')}, "
        f"effective INT storage {field('effective_conv_int_storage_pct')}%, "
        f"size {field('file_size_mb')} MB"
    )


def main() -> int:
    args = parse_args()
    metrics_path = resolve_project_path(args.metrics_csv)
    coverage_path = resolve_project_path(args.coverage_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not coverage_path.exists():
        raise FileNotFoundError(coverage_path)

    metrics_rows = read_rows(metrics_path)
    coverage_rows = read_rows(coverage_path)

    target = find_last(metrics_rows, args.experiment_name)
    fp32 = find_last(metrics_rows, args.fp32_name)
    a8w8 = find_last(metrics_rows, args.a8w8_name)
    mid = find_last(metrics_rows, args.mid_adaround_name)
    coverage = find_coverage(coverage_rows, target)

    target_map = as_float(target, "box_map_50_95")
    fp32_map = as_float(fp32, "box_map_50_95")
    a8w8_map = as_float(a8w8, "box_map_50_95")
    mid_map = as_float(mid, "box_map_50_95")

    print("## AdaRound full result snippet")
    print()
    print("| Model | mAP50-95 | mAP50 | mAP75 | Precision | Recall | vs FP32 | vs A8W8 | vs AdaRound mid |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        "| AdaRound full "
        f"| {fmt(target_map)} "
        f"| {fmt(as_float(target, 'box_map_50'))} "
        f"| {fmt(as_float(target, 'box_map_75'))} "
        f"| {fmt(as_float(target, 'precision'))} "
        f"| {fmt(as_float(target, 'recall'))} "
        f"| {delta(target_map, fp32_map)} "
        f"| {delta(target_map, a8w8_map)} "
        f"| {delta(target_map, mid_map)} |"
    )
    print()
    print(f"Coverage: {coverage_summary(coverage)}")
    print(f"Model SHA256: `{target['model_sha256']}`")
    print(f"Model path: `{target['model_path']}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
