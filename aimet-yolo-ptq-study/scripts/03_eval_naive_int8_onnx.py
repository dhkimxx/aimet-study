"""Run no-AIMET ONNX Runtime INT8 quantization and evaluate it on COCO val."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.metrics import ACCURACY_FIELDNAMES, jsonable
from aimet_yolo_study.ort_quant import ImageCalibrationDataReader
from aimet_yolo_study.records import append_csv_row
from aimet_yolo_study.ultralytics_eval import build_accuracy_row, run_ultralytics_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="0", help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recreate the quantized ONNX model.")
    parser.add_argument("--name", default="naive_onnx_int8")
    return parser.parse_args()


def quantize_static_onnx(
    fp32_model,
    int8_model,
    calibration_manifest,
    input_name: str,
    image_size: int,
    calibration_samples: int,
    force: bool,
) -> None:
    if int8_model.exists() and not force:
        return

    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        raise RuntimeError("Missing onnxruntime.quantization. Check the ONNX Runtime installation.") from exc

    int8_model.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageCalibrationDataReader(
        manifest_path=calibration_manifest,
        input_name=input_name,
        image_size=image_size,
        max_samples=calibration_samples,
    )
    quantize_static(
        model_input=str(fp32_model),
        model_output=str(int8_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config = config["model"]
    dataset_config = config["dataset"]
    benchmark_config = config["benchmark"]
    paths_config = config["paths"]

    fp32_model = resolve_project_path(model_config["path"])
    dataset_yaml = resolve_project_path(dataset_config["dataset_yaml"])
    calibration_manifest = resolve_project_path(dataset_config["root_dir"]) / "calibration_images.txt"
    metrics_csv = resolve_project_path(paths_config["metrics_csv"])
    results_dir = resolve_project_path(paths_config["results_dir"])
    exported_models_dir = resolve_project_path(paths_config["exported_models_dir"])
    int8_model = exported_models_dir / f"{fp32_model.stem}.naive_int8.onnx"
    output_dir = results_dir / "ultralytics" / args.name

    for required in [fp32_model, dataset_yaml, calibration_manifest]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    image_size = args.imgsz or int(model_config["input_shape"][-1])
    batch_size = args.batch or int(benchmark_config["batch_size"])
    calibration_samples = args.calibration_samples or int(dataset_config["calibration_count"])

    quantize_static_onnx(
        fp32_model=fp32_model,
        int8_model=int8_model,
        calibration_manifest=calibration_manifest,
        input_name=model_config["input_name"],
        image_size=image_size,
        calibration_samples=calibration_samples,
        force=args.force,
    )

    metrics = run_ultralytics_val(
        model_path=int8_model,
        dataset_yaml=dataset_yaml,
        split=dataset_config["split"],
        image_size=image_size,
        batch_size=batch_size,
        device=args.device,
        output_dir=output_dir,
    )
    row = build_accuracy_row("B", "naive_onnx_int8", False, int8_model, metrics)
    append_csv_row(metrics_csv, ACCURACY_FIELDNAMES, row)

    details_path = output_dir / "metrics_naive_onnx_int8.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "row": row,
        "results_dict": jsonable(getattr(metrics, "results_dict", {})),
        "quantized_model": str(int8_model),
        "calibration_samples": calibration_samples,
    }
    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)

    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
