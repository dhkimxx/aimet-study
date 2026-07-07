"""Evaluate per-tensor QDQ sensitivity inside the YOLO head cv3 branch."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import onnx

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.hashes import sha256_file
from aimet_yolo_study.metrics import ACCURACY_FIELDNAMES, jsonable
from aimet_yolo_study.qdq_sensitivity import sensitivity_selector, strip_selected_qdq
from aimet_yolo_study.records import append_csv_row
from aimet_yolo_study.ultralytics_eval import (
    build_accuracy_row,
    eval_run_name,
    extract_box_metrics,
    metrics_csv_for_eval,
    run_ultralytics_val,
)


FIELDNAMES = [
    "rank_by_delta",
    "tensor_index",
    "variant",
    "scale_id",
    "layer_id",
    "is_final",
    "removed_q",
    "removed_dq",
    "encoding_bw",
    "encoding_dtype",
    "encoding_is_sym",
    "encoding_scale",
    "encoding_offset",
    "encoding_range",
    "baseline_map_50_95",
    "box_map_50_95",
    "delta_map_50_95",
    "box_map_50",
    "box_map_75",
    "precision",
    "recall",
    "tensor_name",
    "output_model",
    "model_sha256",
]

CV3_RE = re.compile(r"^/model\.23/one2one_cv3\.(?P<scale>\d)/(?P<body>.+)/Conv_output_0$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--source-model",
        default="results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.onnx",
    )
    parser.add_argument(
        "--encodings",
        default="results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.encodings",
    )
    parser.add_argument("--name-prefix", default="aimet_quantsim_a8w8_cv3_layer")
    parser.add_argument("--device", default="0", help="Ultralytics device value, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20260614)
    parser.add_argument("--baseline-map-50-95", type=float, default=None)
    parser.add_argument("--variant", action="append", default=[], help="Only evaluate selected generated variant names.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cv3 tensors.")
    parser.add_argument("--no-eval", action="store_true", help="Only write variant ONNX models and reports.")
    parser.add_argument("--force", action="store_true", help="Overwrite variant ONNX models.")
    parser.add_argument("--output-csv", default="results/head_cv3_layer_sensitivity.csv")
    parser.add_argument("--output-json", default="results/head_cv3_layer_sensitivity.json")
    parser.add_argument("--output-md", default="reports/head_cv3_layer_sensitivity.md")
    return parser.parse_args()


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")


def load_encodings(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("activation_encodings", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid activation_encodings in {path}")
    return {str(entry.get("name", "")): entry for entry in entries if isinstance(entry, dict)}


def model_context(model_path: Path) -> tuple[dict[str, str], set[str], list[str]]:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {initializer.name for initializer in model.graph.initializer}
    producer_op_types = {
        output_name: node.op_type
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    qdq_tensors = [
        node.input[0]
        for node in model.graph.node
        if node.op_type == "QuantizeLinear" and node.input
    ]
    return producer_op_types, initializers, qdq_tensors


def cv3_tensors(source_model: Path) -> list[str]:
    producer_op_types, initializers, qdq_tensors = model_context(source_model)
    selector = sensitivity_selector("head_cv3_outputs")
    tensors = [
        tensor
        for tensor in qdq_tensors
        if selector(tensor, producer_op_types.get(tensor), tensor in initializers)
    ]
    return list(dict.fromkeys(tensors))


def tensor_metadata(tensor_name: str) -> dict[str, object]:
    match = CV3_RE.match(tensor_name)
    if not match:
        return {
            "scale_id": "",
            "layer_id": "unknown",
            "is_final": False,
            "variant": f"cv3_{stable_slug(tensor_name)}",
        }

    scale_id = match.group("scale")
    body = match.group("body")
    segments = [segment for segment in body.split("/") if segment and segment != "conv"]
    layer = segments[-1] if segments else body
    prefix = f"one2one_cv3.{scale_id}."
    layer_id = layer.removeprefix(prefix).replace(".", "_")
    is_final = layer.endswith(".2")
    if is_final:
        layer_id = f"{layer_id}_final"
    variant = f"cv3_s{scale_id}_{layer_id}"
    return {
        "scale_id": scale_id,
        "layer_id": layer_id,
        "is_final": is_final,
        "variant": variant,
    }


def stable_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug[:80] or "tensor"


def first_numeric(value: object) -> float | None:
    if isinstance(value, list) and value:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def encoding_summary(entry: dict[str, Any] | None) -> dict[str, object]:
    if not entry:
        return {
            "encoding_bw": "",
            "encoding_dtype": "",
            "encoding_is_sym": "",
            "encoding_scale": "",
            "encoding_offset": "",
            "encoding_range": "",
        }

    bw = int(entry.get("bw", 0) or 0)
    scale = first_numeric(entry.get("scale"))
    offset = first_numeric(entry.get("offset"))
    q_range = scale * float((2**bw) - 1) if bw > 0 and scale is not None else None
    return {
        "encoding_bw": bw or "",
        "encoding_dtype": entry.get("dtype", ""),
        "encoding_is_sym": bool(entry.get("is_sym", False)),
        "encoding_scale": fmt_float(scale),
        "encoding_offset": fmt_float(offset),
        "encoding_range": fmt_float(q_range),
    }


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def find_baseline_map(metrics_csv: Path, source_model: Path, eval_samples: int | None) -> float | None:
    if not metrics_csv.exists():
        return None
    source_sha = sha256_file(source_model)
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample_tag = f"sample{eval_samples}" if eval_samples is not None else None
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


def ranked_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def delta(row: dict[str, object]) -> float:
        try:
            return float(row.get("delta_map_50_95", ""))
        except (TypeError, ValueError):
            return float("-inf")

    ordered = sorted(rows, key=delta, reverse=True)
    for index, row in enumerate(ordered, start=1):
        row["rank_by_delta"] = index if row.get("delta_map_50_95") != "" else ""
    return rows


def markdown_report(rows: list[dict[str, object]], eval_samples: int | None, baseline_map: float | None) -> str:
    ranked = sorted(
        rows,
        key=lambda row: float(row.get("delta_map_50_95") or "-999"),
        reverse=True,
    )
    lines = [
        "# Head cv3 Per-Layer Sensitivity",
        "",
        "최종 업데이트: 2026-07-08",
        "",
        "A8W8 QDQ 모델에서 YOLO head `cv3` branch Conv output activation QDQ를 하나씩 float로 되돌려 평가한 결과입니다. 목적은 `head_cv3_outputs` 전체 15개 중 어느 activation이 A8W8 손실 회복에 가장 크게 기여하는지 확인하는 것입니다.",
        "",
        f"- 평가 샘플: {eval_samples if eval_samples is not None else 'full val'}",
        f"- 기준 A8W8 mAP50-95: {fmt_float(baseline_map)}",
        "",
        "## Ranked By mAP50-95 Recovery",
        "",
        "| Rank | Variant | Scale | Layer | mAP50-95 | Delta | Enc scale | Enc range |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            "| "
            f"{row.get('rank_by_delta', '')} | {row['variant']} | {row['scale_id']} | {row['layer_id']} | "
            f"{row.get('box_map_50_95', '')} | {row.get('delta_map_50_95', '')} | "
            f"{row.get('encoding_scale', '')} | {row.get('encoding_range', '')} |"
        )

    lines.extend(
        [
            "",
            "## Graph Order",
            "",
            "| Index | Variant | Tensor | mAP50-95 | Delta |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: int(item["tensor_index"])):
        lines.append(
            "| "
            f"{row['tensor_index']} | {row['variant']} | `{row['tensor_name']}` | "
            f"{row.get('box_map_50_95', '')} | {row.get('delta_map_50_95', '')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 양수 delta가 클수록 해당 activation QDQ가 A8W8 정확도 손실에 더 민감하다는 뜻입니다.",
            "- 개별 tensor 제거 결과는 `head_cv3_outputs` 15개 전체 제거 결과보다 작아야 정상입니다. 전체 제거 효과는 여러 activation의 누적 오차를 포함합니다.",
            "- Encoding scale/range와 delta가 강하게 같이 움직이지 않으면, 단순 range 크기보다 branch 위치와 후속 연산 민감도가 더 큰 원인일 수 있습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config = config["model"]
    dataset_config = config["dataset"]
    benchmark_config = config["benchmark"]
    paths_config = config["paths"]

    source_model = resolve_project_path(args.source_model)
    encodings_path = resolve_project_path(args.encodings)
    dataset_yaml = resolve_project_path(dataset_config["dataset_yaml"])
    metrics_csv = metrics_csv_for_eval(resolve_project_path(paths_config["metrics_csv"]), args.eval_samples)
    results_dir = resolve_project_path(paths_config["results_dir"])
    exported_models_dir = resolve_project_path(paths_config["exported_models_dir"])

    require_file(source_model, "Run AIMET QuantSim PTQ first.")
    require_file(encodings_path, "Run AIMET QuantSim PTQ first.")
    if not args.no_eval:
        require_file(dataset_yaml, "Run: python scripts/01_prepare_coco.py --download")

    image_size = args.imgsz or int(model_config["input_shape"][-1])
    batch_size = args.batch or int(benchmark_config["batch_size"])
    encodings = load_encodings(encodings_path)
    tensors = cv3_tensors(source_model)
    if args.variant:
        wanted = set(args.variant)
        tensors = [
            tensor
            for tensor in tensors
            if str(tensor_metadata(tensor)["variant"]) in wanted
        ]
        missing = wanted - {str(tensor_metadata(tensor)["variant"]) for tensor in tensors}
        if missing:
            raise ValueError(f"Unknown --variant values: {sorted(missing)}")
    if args.limit is not None:
        tensors = tensors[: args.limit]
    if not tensors:
        raise RuntimeError(f"No head cv3 QDQ activation tensors found in {source_model}")

    baseline_map = args.baseline_map_50_95
    if baseline_map is None:
        baseline_map = find_baseline_map(metrics_csv, source_model, args.eval_samples)

    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for tensor_index, tensor_name in enumerate(tensors, start=1):
        metadata = tensor_metadata(tensor_name)
        variant = str(metadata["variant"])
        output_model = exported_models_dir / f"{source_model.stem}.sensitivity_{variant}.onnx"
        summary = strip_selected_qdq(
            source_model=source_model,
            output_model=output_model,
            variant=variant,
            selector=lambda tensor, _producer, _is_initializer, target=tensor_name: tensor == target,
            force=args.force,
        )

        row: dict[str, object] = {
            "rank_by_delta": "",
            "tensor_index": tensor_index,
            "variant": variant,
            "scale_id": metadata["scale_id"],
            "layer_id": metadata["layer_id"],
            "is_final": metadata["is_final"],
            "removed_q": summary.removed_quantize_linear_nodes,
            "removed_dq": summary.removed_dequantize_linear_nodes,
            "baseline_map_50_95": fmt_float(baseline_map),
            "tensor_name": tensor_name,
            "output_model": str(output_model),
            "model_sha256": sha256_file(output_model),
        }
        row.update(encoding_summary(encodings.get(tensor_name)))

        detail: dict[str, object] = {"summary": asdict(summary), "row": row}
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
            accuracy_row = build_accuracy_row("S", run_name, True, output_model, metrics)
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

    ranked_rows(rows)

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
                "tensor_count": len(tensors),
                "baseline_map_50_95": baseline_map,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
