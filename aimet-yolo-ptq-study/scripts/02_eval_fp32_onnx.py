"""Evaluate the FP32 YOLO ONNX baseline on COCO val."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.metrics import ACCURACY_FIELDNAMES, jsonable
from aimet_yolo_study.records import append_csv_row
from aimet_yolo_study.ultralytics_eval import build_accuracy_row, run_ultralytics_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="0", help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--name", default="fp32_onnx")
    return parser.parse_args()


def require_file(path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config = config["model"]
    dataset_config = config["dataset"]
    benchmark_config = config["benchmark"]
    paths_config = config["paths"]

    model_path = resolve_project_path(model_config["path"])
    dataset_yaml = resolve_project_path(dataset_config["dataset_yaml"])
    metrics_csv = resolve_project_path(paths_config["metrics_csv"])
    results_dir = resolve_project_path(paths_config["results_dir"])
    output_dir = results_dir / "ultralytics" / args.name

    require_file(model_path, "Run: python scripts/01_prepare_yolo_onnx.py --export")
    require_file(dataset_yaml, "Run: python scripts/01_prepare_coco.py --download")

    image_size = args.imgsz or int(model_config["input_shape"][-1])
    batch_size = args.batch or int(benchmark_config["batch_size"])
    metrics = run_ultralytics_val(
        model_path=model_path,
        dataset_yaml=dataset_yaml,
        split=dataset_config["split"],
        image_size=image_size,
        batch_size=batch_size,
        device=args.device,
        output_dir=output_dir,
    )

    row = build_accuracy_row("A", "fp32_onnx", False, model_path, metrics)
    append_csv_row(metrics_csv, ACCURACY_FIELDNAMES, row)

    details_path = output_dir / "metrics_fp32_onnx.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "row": row,
        "results_dict": jsonable(getattr(metrics, "results_dict", {})),
        "output_dir": str(output_dir),
    }
    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)

    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
