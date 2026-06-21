"""Summarize how much of each ONNX model is quantized."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.metrics import QUANTIZATION_COVERAGE_FIELDNAMES
from aimet_yolo_study.quant_coverage import analyze_quantization_coverage


DEFAULT_MODELS = [
    ("A", "fp32_onnx", "models/yolo26n_pretrained.onnx"),
    ("B", "naive_onnx_int8", "results/models/yolo26n_pretrained.naive_int8_calib64.onnx"),
    ("C", "aimet_quantsim_ptq", "results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.onnx"),
    ("D", "aimet_cle_ptq", "results/models/yolo26n_pretrained.aimet_cle_int8_calib64.onnx"),
    (
        "E",
        "aimet_adaround_ptq",
        "results/models/yolo26n_pretrained.aimet_adaround_int8_calib64_adar8_iter50.onnx",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Optional model spec as ID=PATH or ID=NAME=PATH. Defaults to A-E study artifacts.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def parse_model_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("=", 2)
    if len(parts) == 2:
        experiment_id, model_path = parts
        return experiment_id, experiment_id, model_path
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"Invalid --model spec: {spec}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUANTIZATION_COVERAGE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in QUANTIZATION_COVERAGE_FIELDNAMES})


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    results_dir = resolve_project_path(config["paths"]["results_dir"])
    output_csv = resolve_project_path(args.output_csv) if args.output_csv else results_dir / "quantization_coverage.csv"
    output_json = resolve_project_path(args.output_json) if args.output_json else results_dir / "quantization_coverage.json"
    model_specs = [parse_model_spec(spec) for spec in args.model] if args.model else DEFAULT_MODELS

    rows = []
    for experiment_id, experiment_name, model_path in model_specs:
        resolved_model_path = resolve_project_path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Missing model for {experiment_id}: {resolved_model_path}")
        rows.append(
            analyze_quantization_coverage(
                model_path=resolved_model_path,
                experiment_id=experiment_id,
                experiment_name=experiment_name,
            )
        )

    write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)

    print(json.dumps({"rows": rows, "csv": str(output_csv), "json": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
