"""Run ONNX Runtime QOperator INT8 quantization and optionally evaluate it."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

import onnx

from aimet_yolo_study.artifacts import calibration_suffix
from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.metrics import ACCURACY_FIELDNAMES, jsonable
from aimet_yolo_study.ort_quant import ImageCalibrationDataReader
from aimet_yolo_study.records import append_csv_row
from aimet_yolo_study.ultralytics_eval import (
    build_accuracy_row,
    eval_run_name,
    metrics_csv_for_eval,
    run_ultralytics_val,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="0", help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None, help="Evaluate only a reproducible image subset.")
    parser.add_argument("--eval-seed", type=int, default=20260614)
    parser.add_argument("--op-type", action="append", default=None, help="Operator type to quantize. Repeatable.")
    parser.add_argument("--no-eval", action="store_true", help="Only export the QOperator ONNX model.")
    parser.add_argument("--force", action="store_true", help="Recreate the quantized ONNX model.")
    parser.add_argument("--name", default="ort_qoperator_conv_int8")
    return parser.parse_args()


def quantize_qoperator_onnx(
    fp32_model,
    output_model,
    calibration_manifest,
    input_name: str,
    image_size: int,
    calibration_samples: int,
    op_types: list[str],
    force: bool,
) -> None:
    if output_model.exists() and not force:
        return

    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        raise RuntimeError("Missing onnxruntime.quantization. Check the ONNX Runtime installation.") from exc

    output_model.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageCalibrationDataReader(
        manifest_path=calibration_manifest,
        input_name=input_name,
        image_size=image_size,
        max_samples=calibration_samples,
    )
    # ORT 1.x adjusts Softmax ranges even when only Conv is requested. Include
    # Softmax in calibration so its range exists, then exclude it from rewrite.
    calibration_op_types = sorted(set(op_types + ["Softmax"]))
    nodes_to_exclude = [
        node.name
        for node in onnx.load(str(fp32_model), load_external_data=False).graph.node
        if node.op_type == "Softmax" and node.name
    ]
    quantize_static(
        model_input=str(fp32_model),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=calibration_op_types,
        nodes_to_exclude=nodes_to_exclude,
    )


def op_tag(op_types: list[str]) -> str:
    return "_".join(op_type.lower() for op_type in op_types)


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
    metrics_csv = metrics_csv_for_eval(resolve_project_path(paths_config["metrics_csv"]), args.eval_samples)
    results_dir = resolve_project_path(paths_config["results_dir"])
    run_name = eval_run_name(args.name, args.eval_samples)
    output_dir = results_dir / "ultralytics" / run_name

    for required in [fp32_model, dataset_yaml, calibration_manifest]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    image_size = args.imgsz or int(model_config["input_shape"][-1])
    batch_size = args.batch or int(benchmark_config["batch_size"])
    default_calibration_samples = int(dataset_config["calibration_count"])
    calibration_samples = args.calibration_samples or default_calibration_samples
    op_types = args.op_type or ["Conv"]
    exported_models_dir = resolve_project_path(paths_config["exported_models_dir"])
    output_model = exported_models_dir / (
        f"{fp32_model.stem}.ort_qoperator_int8_{op_tag(op_types)}"
        f"{calibration_suffix(calibration_samples, default_calibration_samples)}.onnx"
    )

    quantize_qoperator_onnx(
        fp32_model=fp32_model,
        output_model=output_model,
        calibration_manifest=calibration_manifest,
        input_name=model_config["input_name"],
        image_size=image_size,
        calibration_samples=calibration_samples,
        op_types=op_types,
        force=args.force,
    )

    details = {
        "quantized_model": str(output_model),
        "quant_format": "QOperator",
        "op_types_to_quantize": op_types,
        "calibration_samples": calibration_samples,
        "eval_samples": args.eval_samples,
        "eval_seed": args.eval_seed,
    }
    if not args.no_eval:
        metrics = run_ultralytics_val(
            model_path=output_model,
            dataset_yaml=dataset_yaml,
            split=dataset_config["split"],
            image_size=image_size,
            batch_size=batch_size,
            device=args.device,
            output_dir=output_dir,
            eval_samples=args.eval_samples,
            eval_seed=args.eval_seed,
        )
        row = build_accuracy_row("G", run_name, False, output_model, metrics)
        append_csv_row(metrics_csv, ACCURACY_FIELDNAMES, row)
        details["row"] = row
        details["results_dict"] = jsonable(getattr(metrics, "results_dict", {}))

    details_path = output_dir / f"metrics_{run_name}.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)

    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
