"""Summarize AIMET encoding ranges by graph region."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import onnx

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import resolve_project_path
from aimet_yolo_study.hashes import sha256_file
from aimet_yolo_study.qdq_sensitivity import sensitivity_selector


DEFAULT_ARTIFACTS = [
    (
        "C64",
        "A8W8 calib64",
        "results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.encodings",
        "results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.onnx",
    ),
    (
        "C1024",
        "A8W8 calib1024",
        "results/models/yolo26n_pretrained.aimet_quantsim_int8.encodings",
        "results/models/yolo26n_pretrained.aimet_quantsim_int8.onnx",
    ),
    (
        "E128",
        "AdaRound adar128 iter2000",
        "results/models/yolo26n_pretrained.aimet_adaround_int8_calib256_adar128_iter2000.encodings",
        "results/models/yolo26n_pretrained.aimet_adaround_int8_calib256_adar128_iter2000.onnx",
    ),
    (
        "E256",
        "AdaRound adar256 iter5000",
        "results/models/yolo26n_pretrained.aimet_adaround_int8_calib256_iter5000.encodings",
        "results/models/yolo26n_pretrained.aimet_adaround_int8_calib256_iter5000.onnx",
    ),
    (
        "A16W8",
        "A16W8 calib64",
        "results/models/yolo26n_pretrained.aimet_quantsim_a16w8_calib64.encodings",
        "results/models/yolo26n_pretrained.aimet_quantsim_a16w8_calib64.onnx",
    ),
    (
        "A8W16",
        "A8W16 calib64",
        "results/models/yolo26n_pretrained.aimet_quantsim_a8w16_calib64.encodings",
        "results/models/yolo26n_pretrained.aimet_quantsim_a8w16_calib64.onnx",
    ),
    (
        "A16W16",
        "A16W16 calib64",
        "results/models/yolo26n_pretrained.aimet_quantsim_a16w16_calib64.encodings",
        "results/models/yolo26n_pretrained.aimet_quantsim_a16w16_calib64.onnx",
    ),
]

ACTIVATION_GROUPS = [
    ("qdq_activations", "QDQ-exported activations", None, True),
    ("sidecar_only_activations", "Encodings not exported to QDQ", None, False),
    ("graph_input", "Graph input", "graph_input", True),
    ("all_conv_outputs", "All Conv outputs", "all_conv_outputs", True),
    ("head_conv_outputs", "YOLO head Conv outputs", "head_conv_outputs", True),
    ("head_cv2_outputs", "YOLO head cv2 outputs", "head_cv2_outputs", True),
    ("head_cv3_outputs", "YOLO head cv3 outputs", "head_cv3_outputs", True),
    ("head_scale0_outputs", "YOLO head scale0 outputs", "head_scale0_outputs", True),
    ("head_scale1_outputs", "YOLO head scale1 outputs", "head_scale1_outputs", True),
    ("head_scale2_outputs", "YOLO head scale2 outputs", "head_scale2_outputs", True),
    ("head_final_outputs", "YOLO head final Conv outputs", "head_final_outputs", True),
    ("late_neck_20_22", "Late neck 20-22", "late_neck_20_22", True),
]

PARAM_GROUPS = [
    ("all_params", "All parameter encodings", lambda name: True),
    ("head_params", "YOLO head parameter encodings", lambda name: name.startswith("model.23.")),
    ("head_cv2_params", "YOLO head cv2 parameter encodings", lambda name: name.startswith("model.23.one2one_cv2.")),
    ("head_cv3_params", "YOLO head cv3 parameter encodings", lambda name: name.startswith("model.23.one2one_cv3.")),
]

FIELDNAMES = [
    "experiment_id",
    "experiment_name",
    "encoding_kind",
    "group",
    "group_label",
    "encoding_count",
    "bitwidths",
    "dtypes",
    "symmetric_count",
    "asymmetric_count",
    "scale_min",
    "scale_median",
    "scale_p90",
    "scale_max",
    "range_min",
    "range_median",
    "range_p90",
    "range_max",
    "offset_min",
    "offset_median",
    "offset_p90",
    "offset_max",
    "encoding_path",
    "encoding_sha256",
    "model_path",
    "model_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Optional spec ID=NAME=ENCODINGS=MODEL. Defaults to key AIMET artifacts.",
    )
    parser.add_argument("--output-csv", default="results/encoding_analysis.csv")
    parser.add_argument("--output-json", default="results/encoding_analysis.json")
    parser.add_argument("--output-md", default="reports/encoding_analysis.md")
    return parser.parse_args()


def parse_artifact_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("=", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid --artifact spec: {spec}")
    return parts[0], parts[1], parts[2], parts[3]


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_context(model_path: Path) -> tuple[dict[str, str], set[str], set[str]]:
    model = onnx.load(str(model_path), load_external_data=False)
    producer_op_types = {
        output_name: node.op_type
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    initializers = {initializer.name for initializer in model.graph.initializer}
    qdq_tensors = {
        node.input[0]
        for node in model.graph.node
        if node.op_type == "QuantizeLinear" and node.input
    }
    return producer_op_types, initializers, qdq_tensors


def encoding_values(entry: dict[str, object], key: str) -> list[float]:
    values = entry.get(key, [])
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def quantized_ranges(entry: dict[str, object]) -> list[float]:
    bitwidth = int(entry.get("bw", 0))
    if bitwidth <= 0:
        return []
    levels = float((2**bitwidth) - 1)
    return [scale * levels for scale in encoding_values(entry, "scale")]


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def counter_string(values: Iterable[object]) -> str:
    counts = Counter(values)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts, key=str))


def summarize_group(
    experiment_id: str,
    experiment_name: str,
    encoding_kind: str,
    group: str,
    group_label: str,
    entries: list[dict[str, object]],
    encoding_path: Path,
    model_path: Path,
) -> dict[str, object]:
    scales = [value for entry in entries for value in encoding_values(entry, "scale")]
    ranges = [value for entry in entries for value in quantized_ranges(entry)]
    offsets = [value for entry in entries for value in encoding_values(entry, "offset")]
    sym_count = sum(1 for entry in entries if bool(entry.get("is_sym", False)))

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "encoding_kind": encoding_kind,
        "group": group,
        "group_label": group_label,
        "encoding_count": len(entries),
        "bitwidths": counter_string(entry.get("bw", "") for entry in entries),
        "dtypes": counter_string(entry.get("dtype", "") for entry in entries),
        "symmetric_count": sym_count,
        "asymmetric_count": len(entries) - sym_count,
        "scale_min": fmt_float(percentile(scales, 0)),
        "scale_median": fmt_float(percentile(scales, 50)),
        "scale_p90": fmt_float(percentile(scales, 90)),
        "scale_max": fmt_float(percentile(scales, 100)),
        "range_min": fmt_float(percentile(ranges, 0)),
        "range_median": fmt_float(percentile(ranges, 50)),
        "range_p90": fmt_float(percentile(ranges, 90)),
        "range_max": fmt_float(percentile(ranges, 100)),
        "offset_min": fmt_float(percentile(offsets, 0)),
        "offset_median": fmt_float(percentile(offsets, 50)),
        "offset_p90": fmt_float(percentile(offsets, 90)),
        "offset_max": fmt_float(percentile(offsets, 100)),
        "encoding_path": str(encoding_path),
        "encoding_sha256": sha256_file(encoding_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
    }


def activation_rows(
    experiment_id: str,
    experiment_name: str,
    activation_entries: list[dict[str, object]],
    encoding_path: Path,
    model_path: Path,
    producer_op_types: dict[str, str],
    initializers: set[str],
    qdq_tensors: set[str],
) -> list[dict[str, object]]:
    rows = []
    for group, label, selector_name, qdq_only in ACTIVATION_GROUPS:
        if group == "sidecar_only_activations":
            selected = [entry for entry in activation_entries if str(entry.get("name", "")) not in qdq_tensors]
        else:
            selector = sensitivity_selector(selector_name) if selector_name else None
            selected = []
            for entry in activation_entries:
                name = str(entry.get("name", ""))
                if qdq_only and name not in qdq_tensors:
                    continue
                if selector and not selector(name, producer_op_types.get(name), name in initializers):
                    continue
                selected.append(entry)

        rows.append(
            summarize_group(
                experiment_id,
                experiment_name,
                "activation",
                group,
                label,
                selected,
                encoding_path,
                model_path,
            )
        )
    return rows


def param_rows(
    experiment_id: str,
    experiment_name: str,
    param_entries: list[dict[str, object]],
    encoding_path: Path,
    model_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for group, label, predicate in PARAM_GROUPS:
        selected = [entry for entry in param_entries if predicate(str(entry.get("name", "")))]
        rows.append(
            summarize_group(
                experiment_id,
                experiment_name,
                "param",
                group,
                label,
                selected,
                encoding_path,
                model_path,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def key(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row["experiment_id"]), str(row["encoding_kind"]), str(row["group"]))


def markdown_report(rows: list[dict[str, object]]) -> str:
    row_by_key = {key(row): row for row in rows}
    experiment_ids = []
    for row in rows:
        experiment_id = str(row["experiment_id"])
        if experiment_id not in experiment_ids:
            experiment_ids.append(experiment_id)

    lines = [
        "# AIMET Encoding Analysis",
        "",
        "최종 업데이트: 2026-07-08",
        "",
        "이 문서는 AIMET `.encodings` sidecar를 HW-independent PTQ 관점에서 요약합니다. QDQ ONNX로 실제 평가된 activation과 QDQ export에서 제외된 sidecar-only activation을 분리해 기록합니다. `scale`은 quantization step이므로 같은 range에서는 작을수록 더 촘촘한 양자화입니다.",
        "",
        "## Activation Group Summary",
        "",
        "| ID | 실험 | QDQ act | Act bits | sidecar-only act | QDQ scale median | Conv scale median | Head cv3 scale median | Head cv3 range median |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment_id in experiment_ids:
        qdq = row_by_key[(experiment_id, "activation", "qdq_activations")]
        sidecar = row_by_key[(experiment_id, "activation", "sidecar_only_activations")]
        conv = row_by_key[(experiment_id, "activation", "all_conv_outputs")]
        cv3 = row_by_key[(experiment_id, "activation", "head_cv3_outputs")]
        lines.append(
            "| "
            f"{experiment_id} | {qdq['experiment_name']} | {qdq['encoding_count']} | {qdq['bitwidths']} | "
            f"{sidecar['encoding_count']} | {qdq['scale_median']} | {conv['scale_median']} | "
            f"{cv3['scale_median']} | {cv3['range_median']} |"
        )

    lines.extend(
        [
            "",
            "## Parameter Summary",
            "",
            "| ID | 실험 | Param encodings | Param bits | Symmetric params | Head params | Head cv3 params | Param scale median |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for experiment_id in experiment_ids:
        params = row_by_key[(experiment_id, "param", "all_params")]
        head_params = row_by_key[(experiment_id, "param", "head_params")]
        cv3_params = row_by_key[(experiment_id, "param", "head_cv3_params")]
        lines.append(
            "| "
            f"{experiment_id} | {params['experiment_name']} | {params['encoding_count']} | {params['bitwidths']} | "
            f"{params['symmetric_count']} | {head_params['encoding_count']} | {cv3_params['encoding_count']} | "
            f"{params['scale_median']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `sidecar-only act`는 AIMET encodings에는 있지만 표준 QDQ ONNX 평가에서는 제외된 activation입니다. 현재 QDQ 평가는 graph output과 YOLO postprocess non-Conv tensor를 float로 남깁니다.",
            "- A16W8/A16W16은 activation bitwidth가 16으로 올라가 같은 tensor 수를 유지하면서 activation quantization step을 크게 줄이는 진단용 설정입니다.",
            "- A8W16/A16W16은 parameter encoding bitwidth가 16으로 올라가지만 AIMET QDQ ONNX의 Conv weight initializer storage는 별도 packed int16/int8로 접힌 배포 파일을 의미하지 않습니다.",
            "- AdaRound 중간/full 설정은 parameter rounding을 바꾸지만 activation encoding group 수와 bitwidth는 A8W8과 같으므로 activation 병목 자체를 제거하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    artifacts = [parse_artifact_spec(spec) for spec in args.artifact] if args.artifact else DEFAULT_ARTIFACTS
    rows: list[dict[str, object]] = []

    for experiment_id, experiment_name, encodings_path, model_path in artifacts:
        resolved_encodings = resolve_project_path(encodings_path)
        resolved_model = resolve_project_path(model_path)
        if not resolved_encodings.exists():
            raise FileNotFoundError(f"Missing encodings for {experiment_id}: {resolved_encodings}")
        if not resolved_model.exists():
            raise FileNotFoundError(f"Missing model for {experiment_id}: {resolved_model}")

        encodings = load_json(resolved_encodings)
        producer_op_types, initializers, qdq_tensors = model_context(resolved_model)
        activation_entries = list(encodings.get("activation_encodings", []))
        param_entries = list(encodings.get("param_encodings", []))
        rows.extend(
            activation_rows(
                experiment_id,
                experiment_name,
                activation_entries,
                resolved_encodings,
                resolved_model,
                producer_op_types,
                initializers,
                qdq_tensors,
            )
        )
        rows.extend(param_rows(experiment_id, experiment_name, param_entries, resolved_encodings, resolved_model))

    output_csv = resolve_project_path(args.output_csv)
    output_json = resolve_project_path(args.output_json)
    output_md = resolve_project_path(args.output_md)
    write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_report(rows), encoding="utf-8")

    print(json.dumps({"csv": str(output_csv), "json": str(output_json), "markdown": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
