"""Run AIMET Cross-Layer Equalization followed by QuantSim PTQ."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from aimet_yolo_study.aimet_quantsim import export_quantsim_model
from aimet_yolo_study.artifacts import calibration_suffix
from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.metrics import ACCURACY_FIELDNAMES, jsonable
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
    parser.add_argument("--device", default="0", help="AIMET/Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None, help="Evaluate only a reproducible image subset.")
    parser.add_argument("--eval-seed", type=int, default=20260614)
    parser.add_argument("--name", default="aimet_cle_ptq")
    parser.add_argument("--force", action="store_true", help="Overwrite exported AIMET artifacts.")
    return parser.parse_args()


def require_file(path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")


def cle_transform_factory(equalized_model_path):
    def apply_cle(model) -> None:
        try:
            import onnx
            from aimet_onnx.cross_layer_equalization import equalize_model
        except ImportError as exc:
            raise RuntimeError("Missing AIMET ONNX CLE APIs. Install the native uv environment with: uv sync") from exc

        equalize_model(model)
        equalized_model_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(equalized_model_path))

    return apply_cle


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    quant_config = load_experiment_config("configs/quantization.yaml")
    model_config = config["model"]
    dataset_config = config["dataset"]
    benchmark_config = config["benchmark"]
    paths_config = config["paths"]

    fp32_model = resolve_project_path(model_config["path"])
    dataset_yaml = resolve_project_path(dataset_config["dataset_yaml"])
    calibration_manifest = resolve_project_path(dataset_config["root_dir"]) / "calibration_images.txt"
    metrics_csv = metrics_csv_for_eval(resolve_project_path(paths_config["metrics_csv"]), args.eval_samples)
    results_dir = resolve_project_path(paths_config["results_dir"])
    exported_models_dir = resolve_project_path(paths_config["exported_models_dir"])
    run_name = eval_run_name(args.name, args.eval_samples)
    output_dir = results_dir / "ultralytics" / run_name

    require_file(fp32_model, "Run: python scripts/01_prepare_yolo_onnx.py --export")
    require_file(dataset_yaml, "Run: python scripts/01_prepare_coco.py --download")
    require_file(calibration_manifest, "Run: python scripts/01_prepare_coco.py --download")

    image_size = args.imgsz or int(model_config["input_shape"][-1])
    batch_size = args.batch or int(benchmark_config["batch_size"])
    default_calibration_samples = int(dataset_config["calibration_count"])
    calibration_samples = args.calibration_samples or default_calibration_samples
    filename_prefix = (
        f"{fp32_model.stem}.aimet_cle_int8"
        f"{calibration_suffix(calibration_samples, default_calibration_samples)}"
    )
    equalized_model_path = exported_models_dir / f"{fp32_model.stem}.cle_equalized_fp32.onnx"

    aimet_model, encodings = export_quantsim_model(
        fp32_model=fp32_model,
        export_dir=exported_models_dir,
        filename_prefix=filename_prefix,
        input_name=model_config["input_name"],
        input_shape=model_config["input_shape"],
        image_size=image_size,
        manifest=calibration_manifest,
        quant_config=quant_config,
        calibration_samples=calibration_samples,
        device=args.device,
        force=args.force,
        model_transform=cle_transform_factory(equalized_model_path),
    )

    metrics = run_ultralytics_val(
        model_path=aimet_model,
        dataset_yaml=dataset_yaml,
        split=dataset_config["split"],
        image_size=image_size,
        batch_size=batch_size,
        device=args.device,
        output_dir=output_dir,
        eval_samples=args.eval_samples,
        eval_seed=args.eval_seed,
    )
    row = build_accuracy_row("D", run_name, True, aimet_model, metrics)
    append_csv_row(metrics_csv, ACCURACY_FIELDNAMES, row)

    details_path = output_dir / f"metrics_{run_name}.json"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "row": row,
        "results_dict": jsonable(getattr(metrics, "results_dict", {})),
        "equalized_fp32_model": str(equalized_model_path),
        "aimet_model": str(aimet_model),
        "encodings": str(encodings),
        "calibration_samples": calibration_samples,
        "eval_samples": args.eval_samples,
        "eval_seed": args.eval_seed,
    }
    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)

    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
