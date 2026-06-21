"""Build and evaluate activation QDQ ablation variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.metrics import ACCURACY_FIELDNAMES, jsonable
from aimet_yolo_study.qdq_sensitivity import sensitivity_selector, strip_selected_qdq
from aimet_yolo_study.records import append_csv_row
from aimet_yolo_study.ultralytics_eval import (
    build_accuracy_row,
    eval_run_name,
    metrics_csv_for_eval,
    run_ultralytics_val,
)


DEFAULT_VARIANTS = [
    "head_conv_outputs",
    "late_neck_20_22",
    "all_conv_outputs",
    "all_activations",
]
HEAD_VARIANTS = [
    "head_cv2_outputs",
    "head_cv3_outputs",
    "head_scale0_outputs",
    "head_scale1_outputs",
    "head_scale2_outputs",
    "head_final_outputs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--source-model",
        default="results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.onnx",
    )
    parser.add_argument("--variant", action="append", choices=DEFAULT_VARIANTS + HEAD_VARIANTS + ["graph_input"], default=[])
    parser.add_argument("--name-prefix", default="aimet_quantsim_a8w8_sensitivity")
    parser.add_argument("--device", default="0", help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None, help="Evaluate only a reproducible image subset.")
    parser.add_argument("--eval-seed", type=int, default=20260614)
    parser.add_argument("--no-eval", action="store_true", help="Only write variant ONNX models.")
    parser.add_argument("--include-tensors", action="store_true", help="Include selected tensor names in JSON output.")
    parser.add_argument("--force", action="store_true", help="Overwrite variant ONNX models.")
    return parser.parse_args()


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config = config["model"]
    dataset_config = config["dataset"]
    benchmark_config = config["benchmark"]
    paths_config = config["paths"]

    source_model = resolve_project_path(args.source_model)
    dataset_yaml = resolve_project_path(dataset_config["dataset_yaml"])
    metrics_csv = metrics_csv_for_eval(resolve_project_path(paths_config["metrics_csv"]), args.eval_samples)
    results_dir = resolve_project_path(paths_config["results_dir"])
    exported_models_dir = resolve_project_path(paths_config["exported_models_dir"])
    variants = args.variant or DEFAULT_VARIANTS

    require_file(source_model, "Run AIMET QuantSim PTQ first.")
    if not args.no_eval:
        require_file(dataset_yaml, "Run: python scripts/01_prepare_coco.py --download")

    image_size = args.imgsz or int(model_config["input_shape"][-1])
    batch_size = args.batch or int(benchmark_config["batch_size"])
    summaries = []
    for variant in variants:
        output_model = exported_models_dir / f"{source_model.stem}.sensitivity_{variant}.onnx"
        summary = strip_selected_qdq(
            source_model=source_model,
            output_model=output_model,
            variant=variant,
            selector=sensitivity_selector(variant),
            force=args.force,
        )

        summary_dict = asdict(summary)
        summary_dict["selected_tensor_count"] = len(summary.selected_tensors)
        if not args.include_tensors:
            summary_dict.pop("selected_tensors")
        details = {"summary": summary_dict}
        if not args.no_eval:
            run_name = eval_run_name(f"{args.name_prefix}_{variant}", args.eval_samples)
            output_dir = results_dir / "ultralytics" / run_name
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
            row = build_accuracy_row("S", run_name, True, output_model, metrics)
            append_csv_row(metrics_csv, ACCURACY_FIELDNAMES, row)
            details.update(
                {
                    "row": row,
                    "results_dict": jsonable(getattr(metrics, "results_dict", {})),
                    "eval_samples": args.eval_samples,
                    "eval_seed": args.eval_seed,
                }
            )

            details_path = output_dir / f"metrics_{run_name}.json"
            details_path.parent.mkdir(parents=True, exist_ok=True)
            with details_path.open("w", encoding="utf-8") as handle:
                json.dump(details, handle, indent=2)

        summaries.append(details)

    print(json.dumps({"variants": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
