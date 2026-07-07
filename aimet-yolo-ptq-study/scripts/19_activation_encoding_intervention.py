"""Build and evaluate selected activation QDQ encoding interventions."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, version_converter

import _bootstrap  # noqa: F401

from aimet_yolo_study.aimet_utils import build_providers, calibration_input_dicts
from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.hashes import sha256_file
from aimet_yolo_study.metrics import ACCURACY_FIELDNAMES, jsonable
from aimet_yolo_study.qdq_sensitivity import sensitivity_selector
from aimet_yolo_study.records import append_csv_row
from aimet_yolo_study.ultralytics_eval import (
    build_accuracy_row,
    eval_run_name,
    extract_box_metrics,
    metrics_csv_for_eval,
    run_ultralytics_val,
)


CV3_S1_2_FINAL = "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0"


@dataclass(frozen=True)
class InterventionSpec:
    target: str
    method: str
    scale_factor: float | None = None
    lower_percentile: float | None = None
    upper_percentile: float | None = None


@dataclass(frozen=True)
class QdqRecord:
    tensor_name: str
    q_node_name: str
    scale_name: str
    zero_point_name: str
    quantized_name: str


PRESET_VARIANTS: dict[str, InterventionSpec] = {
    "cv3_s1_2_final_a16": InterventionSpec(target="cv3_s1_2_final", method="a16_preserve_range"),
    "head_cv3_outputs_a16": InterventionSpec(target="head_cv3_outputs", method="a16_preserve_range"),
    "cv3_s1_2_final_scale075": InterventionSpec(
        target="cv3_s1_2_final",
        method="scale_factor",
        scale_factor=0.75,
    ),
    "cv3_s1_2_final_scale050": InterventionSpec(
        target="cv3_s1_2_final",
        method="scale_factor",
        scale_factor=0.50,
    ),
    "cv3_s1_2_final_scale125": InterventionSpec(
        target="cv3_s1_2_final",
        method="scale_factor",
        scale_factor=1.25,
    ),
    "cv3_s1_2_final_symmetric_i8": InterventionSpec(target="cv3_s1_2_final", method="symmetric_i8"),
    "cv3_s1_2_final_p999": InterventionSpec(
        target="cv3_s1_2_final",
        method="percentile_uint8",
        lower_percentile=0.1,
        upper_percentile=99.9,
    ),
}


FIELDNAMES = [
    "variant",
    "target",
    "method",
    "selected_qdq",
    "old_qdtype",
    "new_qdtype",
    "old_scale_mean",
    "new_scale_mean",
    "old_min_mean",
    "old_max_mean",
    "new_min_mean",
    "new_max_mean",
    "stats_lower_percentile",
    "stats_upper_percentile",
    "stats_samples",
    "baseline_map_50_95",
    "box_map_50_95",
    "delta_map_50_95",
    "box_map_50",
    "box_map_75",
    "precision",
    "recall",
    "output_model",
    "model_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--source-model",
        default="results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.onnx",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(PRESET_VARIANTS),
        default=[],
        help="Preset intervention to build. Defaults to a small cv3 candidate set.",
    )
    parser.add_argument("--name-prefix", default="aimet_quantsim_a8w8_encoding_intervention")
    parser.add_argument("--device", default="0", help="Ultralytics/ORT device value, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20260614)
    parser.add_argument("--baseline-map-50-95", type=float, default=None)
    parser.add_argument("--stats-samples", type=int, default=64)
    parser.add_argument("--stats-seed", type=int, default=20260708)
    parser.add_argument("--max-values-per-tensor", type=int, default=2_000_000)
    parser.add_argument("--no-eval", action="store_true", help="Only write intervention ONNX models and reports.")
    parser.add_argument("--force", action="store_true", help="Overwrite intervention ONNX models.")
    parser.add_argument("--output-csv", default="results/activation_encoding_interventions.csv")
    parser.add_argument("--output-json", default="results/activation_encoding_interventions.json")
    parser.add_argument("--output-md", default="reports/activation_encoding_interventions.md")
    return parser.parse_args()


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def qrange(dtype: np.dtype) -> tuple[int, int]:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.uint8):
        return 0, 255
    if dtype == np.dtype(np.int8):
        return -128, 127
    if dtype == np.dtype(np.uint16):
        return 0, 65535
    if dtype == np.dtype(np.int16):
        return -32768, 32767
    raise ValueError(f"Unsupported QDQ zero-point dtype: {dtype}")


def qdtype_label(array: np.ndarray) -> str:
    return str(np.asarray(array).dtype)


def model_context(model: onnx.ModelProto) -> tuple[dict[str, str], set[str], dict[str, list[onnx.NodeProto]]]:
    initializers = {initializer.name for initializer in model.graph.initializer}
    producer_op_types = {
        output_name: node.op_type
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    consumers_by_input: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers_by_input.setdefault(input_name, []).append(node)
    return producer_op_types, initializers, consumers_by_input


def target_selector(target: str):
    if target == "cv3_s1_2_final":
        return lambda tensor, _producer, _is_initializer: tensor == CV3_S1_2_FINAL
    return sensitivity_selector(target)


def selected_qdq_records(model: onnx.ModelProto, target: str) -> list[QdqRecord]:
    producer_op_types, initializers, consumers_by_input = model_context(model)
    selector = target_selector(target)
    records: list[QdqRecord] = []
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or len(node.input) < 3 or not node.output:
            continue
        tensor_name = node.input[0]
        quantized_name = node.output[0]
        consumers = consumers_by_input.get(quantized_name, [])
        if not any(consumer.op_type == "DequantizeLinear" for consumer in consumers):
            continue
        if selector(tensor_name, producer_op_types.get(tensor_name), tensor_name in initializers):
            records.append(
                QdqRecord(
                    tensor_name=tensor_name,
                    q_node_name=node.name,
                    scale_name=node.input[1],
                    zero_point_name=node.input[2],
                    quantized_name=quantized_name,
                )
            )
    return records


def initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    replacement = numpy_helper.from_array(np.asarray(array), name)
    for initializer in model.graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(replacement)
            return
    raise KeyError(f"Missing initializer {name}")


def min_max_from_qdq(scale: np.ndarray, zero_point: np.ndarray) -> tuple[float, float]:
    scale_value = float(np.asarray(scale).reshape(-1)[0])
    zero_value = int(np.asarray(zero_point).reshape(-1)[0])
    qmin, qmax = qrange(np.asarray(zero_point).dtype)
    return (qmin - zero_value) * scale_value, (qmax - zero_value) * scale_value


def new_asymmetric_params(min_value: float, max_value: float, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    qmin, qmax = qrange(dtype)
    if min_value > 0:
        min_value = 0.0
    if max_value < 0:
        max_value = 0.0
    if max_value <= min_value:
        max_value = min_value + 1e-6
    scale = (max_value - min_value) / float(qmax - qmin)
    zero_point = int(np.clip(np.rint(qmin - min_value / scale), qmin, qmax))
    return np.asarray(scale, dtype=np.float32), np.asarray(zero_point, dtype=dtype)


def new_symmetric_i8_params(old_min: float, old_max: float) -> tuple[np.ndarray, np.ndarray]:
    max_abs = max(abs(old_min), abs(old_max), 1e-6)
    scale = max_abs / 127.0
    return np.asarray(scale, dtype=np.float32), np.asarray(0, dtype=np.int8)


def apply_intervention(
    model: onnx.ModelProto,
    records: list[QdqRecord],
    spec: InterventionSpec,
    percentile_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    arrays = initializer_arrays(model)
    old_scales = []
    new_scales = []
    old_mins = []
    old_maxes = []
    new_mins = []
    new_maxes = []
    old_dtypes = []
    new_dtypes = []
    requires_opset21 = False

    for record in records:
        old_scale = arrays[record.scale_name]
        old_zero_point = arrays[record.zero_point_name]
        old_min, old_max = min_max_from_qdq(old_scale, old_zero_point)
        old_dtype = np.asarray(old_zero_point).dtype

        if spec.method == "a16_preserve_range":
            new_scale, new_zero_point = new_asymmetric_params(old_min, old_max, np.dtype(np.uint16))
            requires_opset21 = True
        elif spec.method == "scale_factor":
            if spec.scale_factor is None:
                raise ValueError("scale_factor intervention requires scale_factor")
            new_scale = np.asarray(old_scale, dtype=np.float32) * np.float32(spec.scale_factor)
            new_zero_point = np.asarray(old_zero_point)
        elif spec.method == "symmetric_i8":
            new_scale, new_zero_point = new_symmetric_i8_params(old_min, old_max)
        elif spec.method == "percentile_uint8":
            if percentile_ranges is None or record.tensor_name not in percentile_ranges:
                raise ValueError(f"Missing percentile range for {record.tensor_name}")
            new_min, new_max = percentile_ranges[record.tensor_name]
            new_scale, new_zero_point = new_asymmetric_params(new_min, new_max, np.dtype(np.uint8))
        else:
            raise ValueError(f"Unsupported intervention method: {spec.method}")

        replace_initializer(model, record.scale_name, new_scale)
        replace_initializer(model, record.zero_point_name, new_zero_point)
        new_min, new_max = min_max_from_qdq(new_scale, new_zero_point)

        old_scales.append(float(np.asarray(old_scale).reshape(-1)[0]))
        new_scales.append(float(np.asarray(new_scale).reshape(-1)[0]))
        old_mins.append(old_min)
        old_maxes.append(old_max)
        new_mins.append(new_min)
        new_maxes.append(new_max)
        old_dtypes.append(str(old_dtype))
        new_dtypes.append(str(np.asarray(new_zero_point).dtype))

    return {
        "selected_qdq": len(records),
        "old_qdtype": ",".join(sorted(set(old_dtypes))),
        "new_qdtype": ",".join(sorted(set(new_dtypes))),
        "old_scale_mean": fmt_float(float(np.mean(old_scales)) if old_scales else None),
        "new_scale_mean": fmt_float(float(np.mean(new_scales)) if new_scales else None),
        "old_min_mean": fmt_float(float(np.mean(old_mins)) if old_mins else None),
        "old_max_mean": fmt_float(float(np.mean(old_maxes)) if old_maxes else None),
        "new_min_mean": fmt_float(float(np.mean(new_mins)) if new_mins else None),
        "new_max_mean": fmt_float(float(np.mean(new_maxes)) if new_maxes else None),
        "requires_opset21": requires_opset21,
    }


def add_tensor_outputs(model: onnx.ModelProto, tensor_names: list[str]) -> onnx.ModelProto:
    existing_outputs = {output.name for output in model.graph.output}
    for tensor_name in tensor_names:
        if tensor_name in existing_outputs:
            continue
        model.graph.output.append(helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, None))
    return model


def collect_percentile_ranges(
    source_model: Path,
    tensor_names: list[str],
    manifest: Path,
    input_name: str,
    image_size: int,
    sample_count: int,
    lower_percentile: float,
    upper_percentile: float,
    max_values_per_tensor: int,
    seed: int,
    device: str,
) -> dict[str, tuple[float, float]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Missing onnxruntime. Install dependencies with: uv sync") from exc

    model = onnx.load(str(source_model), load_external_data=False)
    add_tensor_outputs(model, tensor_names)
    temp_path = Path(tempfile.gettempdir()) / f"{source_model.stem}.activation_stats.onnx"
    onnx.save(model, str(temp_path))

    session = ort.InferenceSession(str(temp_path), providers=build_providers(device))
    rng = np.random.default_rng(seed)
    per_sample_cap = max(1, max_values_per_tensor // max(1, sample_count))
    samples: dict[str, list[np.ndarray]] = {tensor_name: [] for tensor_name in tensor_names}
    counts = {tensor_name: 0 for tensor_name in tensor_names}

    for inputs in calibration_input_dicts(
        manifest_path=manifest,
        input_name=input_name,
        image_size=image_size,
        sample_count=sample_count,
    ):
        outputs = session.run(tensor_names, inputs)
        for tensor_name, output in zip(tensor_names, outputs):
            remaining = max_values_per_tensor - counts[tensor_name]
            if remaining <= 0:
                continue
            flat = np.asarray(output, dtype=np.float32).reshape(-1)
            take = min(flat.size, per_sample_cap, remaining)
            if take < flat.size:
                indices = rng.choice(flat.size, size=take, replace=False)
                flat = flat[indices]
            samples[tensor_name].append(flat)
            counts[tensor_name] += int(flat.size)

    ranges = {}
    for tensor_name, chunks in samples.items():
        if not chunks:
            raise RuntimeError(f"No activation samples collected for {tensor_name}")
        values = np.concatenate(chunks)
        ranges[tensor_name] = (
            float(np.percentile(values, lower_percentile)),
            float(np.percentile(values, upper_percentile)),
        )
    return ranges


def find_baseline_map(metrics_csv: Path, source_model: Path, eval_samples: int | None) -> float | None:
    if not metrics_csv.exists():
        return None
    source_sha = sha256_file(source_model)
    sample_tag = f"sample{eval_samples}" if eval_samples is not None else None
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = []
    for row in rows:
        if row.get("model_sha256") != source_sha or not row.get("box_map_50_95"):
            continue
        if sample_tag is not None and sample_tag not in str(row.get("experiment_name", "")):
            continue
        matches.append(row)
    if not matches:
        return None
    try:
        return float(matches[-1]["box_map_50_95"])
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def markdown_report(rows: list[dict[str, object]], eval_samples: int | None, baseline_map: float | None) -> str:
    lines = [
        "# Activation Encoding Interventions",
        "",
        "최종 업데이트: 2026-07-08",
        "",
        "A8W8 QDQ 모델에서 선택한 activation QDQ의 scale/zero-point만 바꿔 평가한 결과입니다. 목적은 QDQ를 float로 제거하지 않고도 `cv3` 민감도가 encoding/range 조정으로 회복되는지 확인하는 것입니다.",
        "",
        f"- 평가 샘플: {eval_samples if eval_samples is not None else 'full val'}",
        f"- 기준 A8W8 mAP50-95: {fmt_float(baseline_map)}",
        "",
        "## Results",
        "",
        "| Variant | Target | Method | QDQ | mAP50-95 | Delta | Old scale | New scale | Old range | New range |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        old_range = range_text(row.get("old_min_mean"), row.get("old_max_mean"))
        new_range = range_text(row.get("new_min_mean"), row.get("new_max_mean"))
        lines.append(
            "| "
            f"{row['variant']} | {row['target']} | {row['method']} | {row['selected_qdq']} | "
            f"{row.get('box_map_50_95', '')} | {row.get('delta_map_50_95', '')} | "
            f"{row.get('old_scale_mean', '')} | {row.get('new_scale_mean', '')} | "
            f"{old_range} | {new_range} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `a16_preserve_range`는 기존 min/max 범위를 유지하면서 selected activation만 uint16 QDQ로 바꿉니다. 좋아지면 bitwidth/step size가 병목이라는 신호입니다.",
            "- `scale_factor`는 기존 zero-point를 유지하고 scale만 바꿔 range clipping 또는 range expansion 효과를 봅니다.",
            "- `percentile_uint8`은 calibration activation percentile로 min/max를 다시 잡아 outlier range가 손실 원인인지 확인합니다.",
            "- 이 결과도 AIMET HW-independent accuracy/encoding 진단입니다. selected QDQ가 uint16이 되면 opset 21 QDQ 평가 모델이며 packed deployment artifact는 아닙니다.",
            "",
        ]
    )
    return "\n".join(lines)


def range_text(min_value: object, max_value: object) -> str:
    if min_value in ("", None) or max_value in ("", None):
        return ""
    return f"{min_value}..{max_value}"


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config = config["model"]
    dataset_config = config["dataset"]
    benchmark_config = config["benchmark"]
    paths_config = config["paths"]

    source_model = resolve_project_path(args.source_model)
    dataset_yaml = resolve_project_path(dataset_config["dataset_yaml"])
    calibration_manifest = resolve_project_path(dataset_config["root_dir"]) / "calibration_images.txt"
    metrics_csv = metrics_csv_for_eval(resolve_project_path(paths_config["metrics_csv"]), args.eval_samples)
    results_dir = resolve_project_path(paths_config["results_dir"])
    exported_models_dir = resolve_project_path(paths_config["exported_models_dir"])
    image_size = args.imgsz or int(model_config["input_shape"][-1])
    batch_size = args.batch or int(benchmark_config["batch_size"])
    variants = args.variant or [
        "cv3_s1_2_final_a16",
        "head_cv3_outputs_a16",
        "cv3_s1_2_final_scale075",
        "cv3_s1_2_final_scale050",
        "cv3_s1_2_final_symmetric_i8",
    ]

    require_file(source_model, "Run AIMET QuantSim PTQ first.")
    require_file(calibration_manifest, "Run: python scripts/01_prepare_coco.py --download")
    if not args.no_eval:
        require_file(dataset_yaml, "Run: python scripts/01_prepare_coco.py --download")

    baseline_map = args.baseline_map_50_95
    if baseline_map is None:
        baseline_map = find_baseline_map(metrics_csv, source_model, args.eval_samples)

    percentile_cache: dict[tuple[str, float, float], dict[str, tuple[float, float]]] = {}
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for variant in variants:
        spec = PRESET_VARIANTS[variant]
        model = onnx.load(str(source_model), load_external_data=False)
        records = selected_qdq_records(model, spec.target)
        if not records:
            raise RuntimeError(f"No selected QDQ records for {variant} target {spec.target}")

        percentile_ranges = None
        if spec.method == "percentile_uint8":
            assert spec.lower_percentile is not None
            assert spec.upper_percentile is not None
            cache_key = (spec.target, spec.lower_percentile, spec.upper_percentile)
            if cache_key not in percentile_cache:
                percentile_cache[cache_key] = collect_percentile_ranges(
                    source_model=source_model,
                    tensor_names=[record.tensor_name for record in records],
                    manifest=calibration_manifest,
                    input_name=str(model_config["input_name"]),
                    image_size=image_size,
                    sample_count=args.stats_samples,
                    lower_percentile=spec.lower_percentile,
                    upper_percentile=spec.upper_percentile,
                    max_values_per_tensor=args.max_values_per_tensor,
                    seed=args.stats_seed,
                    device=args.device,
                )
            percentile_ranges = percentile_cache[cache_key]

        summary = apply_intervention(model, records, spec, percentile_ranges)
        if bool(summary.get("requires_opset21")):
            model = version_converter.convert_version(model, 21)
        onnx.checker.check_model(model)

        output_model = exported_models_dir / f"{source_model.stem}.encoding_{variant}.onnx"
        if output_model.exists() and not args.force:
            raise FileExistsError(f"{output_model} already exists. Pass --force to rebuild it.")
        output_model.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_model))

        row: dict[str, object] = {
            "variant": variant,
            "target": spec.target,
            "method": spec.method,
            "stats_lower_percentile": fmt_float(spec.lower_percentile),
            "stats_upper_percentile": fmt_float(spec.upper_percentile),
            "stats_samples": args.stats_samples if spec.method == "percentile_uint8" else "",
            "baseline_map_50_95": fmt_float(baseline_map),
            "output_model": str(output_model),
            "model_sha256": sha256_file(output_model),
        }
        row.update({key: value for key, value in summary.items() if key in FIELDNAMES})

        detail: dict[str, object] = {
            "variant": variant,
            "spec": spec.__dict__,
            "records": [record.__dict__ for record in records],
            "summary": summary,
            "percentile_ranges": percentile_ranges,
        }
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
            accuracy_row = build_accuracy_row("I", run_name, True, output_model, metrics)
            append_csv_row(metrics_csv, ACCURACY_FIELDNAMES, accuracy_row)
            metric_values = extract_box_metrics(metrics)
            row.update({key: jsonable(value) for key, value in metric_values.items()})
            if baseline_map is not None and row.get("box_map_50_95") != "":
                row["delta_map_50_95"] = fmt_float(float(row["box_map_50_95"]) - baseline_map)
            else:
                row["delta_map_50_95"] = ""
            detail.update(
                {
                    "accuracy_row": accuracy_row,
                    "results_dict": jsonable(getattr(metrics, "results_dict", {})),
                    "eval_samples": args.eval_samples,
                    "eval_seed": args.eval_seed,
                }
            )

            details_path = output_dir / f"metrics_{run_name}.json"
            details_path.parent.mkdir(parents=True, exist_ok=True)
            with details_path.open("w", encoding="utf-8") as handle:
                json.dump(detail, handle, indent=2)
        else:
            row.update(
                {
                    "box_map_50_95": "",
                    "delta_map_50_95": "",
                    "box_map_50": "",
                    "box_map_75": "",
                    "precision": "",
                    "recall": "",
                }
            )

        rows.append(row)
        details.append(detail)

    output_csv = resolve_project_path(args.output_csv)
    output_json = resolve_project_path(args.output_json)
    output_md = resolve_project_path(args.output_md)
    write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "details": details}, handle, indent=2)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_report(rows, args.eval_samples, baseline_map), encoding="utf-8")

    print(
        json.dumps(
            {
                "csv": str(output_csv),
                "json": str(output_json),
                "markdown": str(output_md),
                "variants": variants,
                "baseline_map_50_95": baseline_map,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
