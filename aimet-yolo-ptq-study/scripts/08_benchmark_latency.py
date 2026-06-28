"""Benchmark model-only and lightweight end-to-end ONNX latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aimet_yolo_study.aimet_utils import build_providers
from aimet_yolo_study.benchmark import (
    benchmark_end_to_end,
    benchmark_model_only,
    load_manifest_images,
    summarize_ms,
)
from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.images import preprocess_yolo_image
from aimet_yolo_study.metrics import LATENCY_FIELDNAMES
from aimet_yolo_study.records import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", default=None, help="ONNX model path. Defaults to configured FP32 model.")
    parser.add_argument("--experiment-id", default="A")
    parser.add_argument("--experiment-name", default="fp32_onnx")
    parser.add_argument("--device", default="0", help="ONNX Runtime device value, for example 0 or cpu.")
    parser.add_argument("--provider", choices=["cuda", "tensorrt", "cpu"], default="cuda")
    parser.add_argument("--trt-cache-dir", default=None, help="TensorRT engine cache directory.")
    parser.add_argument("--trt-fp16", action="store_true", help="Enable TensorRT FP16 mode.")
    parser.add_argument("--trt-int8", action="store_true", help="Enable TensorRT INT8 mode.")
    parser.add_argument(
        "--allow-provider-fallback",
        action="store_true",
        help="Record latency even if ONNX Runtime falls back from the requested provider.",
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--measured-runs", type=int, default=None)
    parser.add_argument("--e2e-images", type=int, default=32)
    return parser.parse_args()


def resolve_model_path(config: dict[str, object], model_arg: str | None) -> Path:
    if model_arg:
        return resolve_project_path(model_arg)
    return resolve_project_path(config["model"]["path"])


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config = config["model"]
    dataset_config = config["dataset"]
    benchmark_config = config["benchmark"]
    paths_config = config["paths"]

    model_path = resolve_model_path(config, args.model)
    calibration_manifest = resolve_project_path(dataset_config["root_dir"]) / "calibration_images.txt"
    latency_csv = resolve_project_path(paths_config["latency_csv"])
    results_dir = resolve_project_path(paths_config["results_dir"])
    output_json = results_dir / "latency" / f"{args.experiment_name}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not calibration_manifest.exists():
        raise FileNotFoundError(f"Missing calibration manifest: {calibration_manifest}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Missing onnxruntime. Run inside the configured experiment environment.") from exc

    image_size = args.imgsz or int(model_config["input_shape"][-1])
    warmup_runs = args.warmup_runs or int(benchmark_config["warmup_runs"])
    measured_runs = args.measured_runs or int(benchmark_config["measured_runs"])
    trt_cache_dir = (
        resolve_project_path(args.trt_cache_dir)
        if args.trt_cache_dir
        else results_dir / "tensorrt_cache" / args.experiment_name
    )
    if args.provider == "tensorrt":
        trt_cache_dir.mkdir(parents=True, exist_ok=True)
    providers = build_providers(
        args.device,
        provider=args.provider,
        tensorrt_cache_path=trt_cache_dir,
        tensorrt_fp16=args.trt_fp16,
        tensorrt_int8=args.trt_int8,
    )
    session = ort.InferenceSession(str(model_path), providers=providers)
    active_providers = session.get_providers()
    requested_provider = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }[args.provider]
    if not args.allow_provider_fallback and requested_provider not in active_providers:
        raise RuntimeError(
            f"Requested {requested_provider}, but active providers are {active_providers}. "
            "Use --allow-provider-fallback only when fallback latency is intentional."
        )
    input_name = session.get_inputs()[0].name
    image_paths = load_manifest_images(calibration_manifest, args.e2e_images)
    input_tensor = preprocess_yolo_image(image_paths[0], image_size)

    model_only = benchmark_model_only(session, input_name, input_tensor, warmup_runs, measured_runs)
    end_to_end = benchmark_end_to_end(
        session=session,
        input_name=input_name,
        image_paths=image_paths,
        image_size=image_size,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )

    row = {
        "experiment_id": args.experiment_id,
        "experiment_name": args.experiment_name,
        "model_path": str(model_path),
        "provider": ",".join(active_providers),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
    }
    row.update(summarize_ms(model_only, "model_only"))
    row.update(summarize_ms(end_to_end, "end_to_end"))
    append_csv_row(latency_csv, LATENCY_FIELDNAMES, row)

    details = {
        "row": row,
        "provider_request": {
            "provider": args.provider,
            "tensorrt_cache_dir": str(trt_cache_dir) if args.provider == "tensorrt" else None,
            "tensorrt_fp16": args.trt_fp16,
            "tensorrt_int8": args.trt_int8,
        },
        "model_only_samples_ms": model_only,
        "end_to_end_samples_ms": end_to_end,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)

    print(json.dumps({"row": row, "details_path": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
