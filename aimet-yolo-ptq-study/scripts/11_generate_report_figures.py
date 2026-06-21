"""Generate SVG figures used by the study reports."""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import resolve_project_path


@dataclass(frozen=True)
class SeriesPoint:
    key: str
    label: str
    value: float
    color: str


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    map_value: float
    latency_ms: float
    color: str


FULL_MODELS = [
    ("fp32_onnx", "FP32", "#2f6f73"),
    ("naive_onnx_int8", "Naive INT8", "#8c3b3b"),
    ("aimet_quantsim_a8w8_gpu", "A8W8 QDQ", "#596fb7"),
    ("aimet_quantsim_a16w8_gpu", "A16W8 QDQ", "#4b8b3b"),
    ("aimet_quantsim_a8w16_gpu", "A8W16 QDQ", "#b47a2a"),
    ("aimet_quantsim_a16w16_gpu", "A16W16 QDQ", "#7a5ca8"),
]

LATENCY_NAMES = {
    "FP32": "fp32_onnx_latency",
    "Naive INT8": "naive_onnx_int8_latency",
    "A8W8 QDQ": "aimet_quantsim_a8w8_qdq_latency",
    "A16W8 QDQ": "aimet_quantsim_a16w8_qdq_latency",
    "A8W16 QDQ": "aimet_quantsim_a8w16_qdq_latency",
    "A16W16 QDQ": "aimet_quantsim_a16w16_qdq_latency",
}

SENSITIVITY_MODELS = [
    ("aimet_quantsim_ptq_sample100_qdq_sample100", "A8W8 baseline", "#596fb7"),
    ("aimet_quantsim_a8w8_sensitivity_late_neck_20_22_sample100", "Late neck float", "#637383"),
    ("aimet_quantsim_a8w8_sensitivity_head_conv_outputs_sample100", "Head conv float", "#4b8b3b"),
    ("aimet_quantsim_a8w8_sensitivity_all_conv_outputs_sample100", "All conv out float", "#b47a2a"),
    ("aimet_quantsim_a8w8_sensitivity_all_activations_sample100", "All act float", "#2f6f73"),
    ("aimet_quantsim_a16w8_sample100_gpu_sample100", "A16W8 reference", "#7a5ca8"),
]

COVERAGE_MODELS = [
    ("naive_onnx_int8", "Naive INT8", "#8c3b3b"),
    ("aimet_quantsim_ptq", "A8W8 QDQ", "#596fb7"),
    ("aimet_cle_ptq", "CLE QDQ", "#4b8b3b"),
    ("aimet_adaround_ptq", "AdaRound smoke", "#7a5ca8"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", default="results/metrics.csv")
    parser.add_argument("--quick-metrics-csv", default="results/metrics_quick.csv")
    parser.add_argument("--latency-csv", default="results/latency.csv")
    parser.add_argument("--coverage-csv", default="results/quantization_coverage.csv")
    parser.add_argument("--output-dir", default="reports/figures")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for row in rows:
        values[row["experiment_name"]] = row
    return values


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing numeric field {field!r} in row {row!r}")
    return float(value)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_document(width: int, height: int, body: list[str]) -> str:
    style = """
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: #1f252b; }
    .title { font-size: 22px; font-weight: 700; }
    .subtitle { font-size: 13px; fill: #53606a; }
    .axis { font-size: 12px; fill: #53606a; }
    .label { font-size: 13px; }
    .value { font-size: 12px; font-weight: 700; }
    .grid { stroke: #d9dee3; stroke-width: 1; }
    .axis-line { stroke: #6b747c; stroke-width: 1.2; }
  </style>"""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"{style}\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def draw_full_accuracy(rows: dict[str, dict[str, str]], output_dir: Path) -> Path:
    points = [
        SeriesPoint(key=name, label=label, value=as_float(rows[name], "box_map_50_95"), color=color)
        for name, label, color in FULL_MODELS
    ]
    width, height = 960, 430
    left, top, plot_width = 190, 78, 620
    bar_h, gap = 30, 18
    x_max = 0.42
    body = [
        '<rect width="960" height="430" fill="#ffffff"/>',
        '<text x="40" y="36" class="title">Full COCO accuracy</text>',
        '<text x="40" y="58" class="subtitle">COCO val 5000, mAP50-95, CUDAExecutionProvider</text>',
    ]
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        x = left + tick / x_max * plot_width
        body.append(f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 62}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height - 38}" text-anchor="middle" class="axis">{tick:.1f}</text>')
    body.append(f'<line x1="{left}" y1="{height - 62}" x2="{left + plot_width}" y2="{height - 62}" class="axis-line"/>')
    for index, point in enumerate(points):
        y = top + index * (bar_h + gap)
        w = point.value / x_max * plot_width
        body.append(f'<text x="40" y="{y + 21}" class="label">{esc(point.label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{point.color}"/>')
        body.append(f'<text x="{left + w + 10:.1f}" y="{y + 20}" class="value">{point.value:.4f}</text>')
    body.append(f'<text x="{left + plot_width}" y="{height - 14}" text-anchor="end" class="axis">mAP50-95</text>')
    output = output_dir / "full_accuracy.svg"
    write_svg(output, svg_document(width, height, body))
    return output


def draw_accuracy_latency(
    metric_rows: dict[str, dict[str, str]],
    latency_rows: dict[str, dict[str, str]],
    output_dir: Path,
) -> Path:
    points = []
    for metric_name, label, color in FULL_MODELS:
        latency_name = LATENCY_NAMES[label]
        points.append(
            ParetoPoint(
                label=label,
                map_value=as_float(metric_rows[metric_name], "box_map_50_95"),
                latency_ms=as_float(latency_rows[latency_name], "model_only_mean_ms"),
                color=color,
            )
        )

    width, height = 960, 500
    left, top, plot_width, plot_height = 92, 70, 720, 330
    x_max, y_max = 120.0, 0.42
    body = [
        '<rect width="960" height="500" fill="#ffffff"/>',
        '<text x="40" y="36" class="title">Accuracy and model-only latency</text>',
        '<text x="40" y="58" class="subtitle">Higher is better for mAP; lower is better for latency</text>',
    ]
    for tick in [0, 30, 60, 90, 120]:
        x = left + tick / x_max * plot_width
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" class="axis">{tick}</text>')
    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        y = top + plot_height - tick / y_max * plot_height
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>')
    body.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis-line"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis-line"/>')
    label_offsets = {
        "FP32": (10, -12),
        "Naive INT8": (10, -10),
        "A8W8 QDQ": (10, 18),
        "A16W8 QDQ": (-92, -16),
        "A8W16 QDQ": (-92, 20),
        "A16W16 QDQ": (10, -2),
    }
    for point in points:
        x = left + point.latency_ms / x_max * plot_width
        y = top + plot_height - point.map_value / y_max * plot_height
        dx, dy = label_offsets[point.label]
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{point.color}" stroke="#1f252b" stroke-width="1"/>')
        body.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" class="label">{esc(point.label)}</text>')
    body.append(f'<text x="{left + plot_width}" y="{height - 36}" text-anchor="end" class="axis">model-only latency (ms)</text>')
    body.append('<text x="24" y="72" transform="rotate(-90 24 72)" class="axis">mAP50-95</text>')
    output = output_dir / "accuracy_latency_pareto.svg"
    write_svg(output, svg_document(width, height, body))
    return output


def draw_activation_sensitivity(rows: dict[str, dict[str, str]], output_dir: Path) -> Path:
    baseline_name, _, _ = SENSITIVITY_MODELS[0]
    baseline = as_float(rows[baseline_name], "box_map_50_95")
    points = [
        SeriesPoint(key=name, label=label, value=as_float(rows[name], "box_map_50_95") - baseline, color=color)
        for name, label, color in SENSITIVITY_MODELS
    ]
    width, height = 980, 430
    left, top, plot_width = 210, 78, 600
    bar_h, gap = 30, 18
    x_max = 0.03
    body = [
        '<rect width="980" height="430" fill="#ffffff"/>',
        '<text x="40" y="36" class="title">Activation QDQ sensitivity</text>',
        '<text x="40" y="58" class="subtitle">Delta mAP50-95 vs A8W8 baseline, sample100</text>',
    ]
    for tick in [0.0, 0.01, 0.02, 0.03]:
        x = left + tick / x_max * plot_width
        body.append(f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 62}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height - 38}" text-anchor="middle" class="axis">+{tick:.2f}</text>')
    body.append(f'<line x1="{left}" y1="{height - 62}" x2="{left + plot_width}" y2="{height - 62}" class="axis-line"/>')
    for index, point in enumerate(points):
        y = top + index * (bar_h + gap)
        w = max(point.value, 0.0) / x_max * plot_width
        abs_value = baseline + point.value
        body.append(f'<text x="40" y="{y + 21}" class="label">{esc(point.label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{point.color}"/>')
        body.append(
            f'<text x="{left + max(w, 2) + 10:.1f}" y="{y + 20}" class="value">'
            f'+{point.value:.4f} ({abs_value:.4f})</text>'
        )
    body.append(f'<text x="{left + plot_width}" y="{height - 14}" text-anchor="end" class="axis">delta mAP50-95</text>')
    output = output_dir / "activation_sensitivity.svg"
    write_svg(output, svg_document(width, height, body))
    return output


def draw_qdq_storage_coverage(rows: dict[str, dict[str, str]], output_dir: Path) -> Path:
    width, height = 960, 430
    left, top, plot_width = 190, 80, 610
    group_h, bar_h = 62, 22
    x_max = 100.0
    body = [
        '<rect width="960" height="430" fill="#ffffff"/>',
        '<text x="40" y="36" class="title">Conv weight QDQ vs storage</text>',
        '<text x="40" y="58" class="subtitle">QDQ coverage can be 100% while initializer storage remains FP32</text>',
    ]
    for tick in [0, 25, 50, 75, 100]:
        x = left + tick / x_max * plot_width
        body.append(f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 62}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height - 38}" text-anchor="middle" class="axis">{tick}%</text>')
    for index, (name, label, color) in enumerate(COVERAGE_MODELS):
        row = rows[name]
        y = top + index * group_h
        qdq = as_float(row, "conv_weight_qdq_pct")
        storage = as_float(row, "conv_weight_int_storage_pct")
        body.append(f'<text x="40" y="{y + 27}" class="label">{esc(label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{qdq / x_max * plot_width:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
        body.append(f'<rect x="{left}" y="{y + 28}" width="{storage / x_max * plot_width:.1f}" height="{bar_h}" rx="3" fill="#d1a84d"/>')
        body.append(f'<text x="{left + qdq / x_max * plot_width + 8:.1f}" y="{y + 16}" class="value">QDQ {qdq:.0f}%</text>')
        body.append(
            f'<text x="{left + max(storage / x_max * plot_width, 2) + 8:.1f}" y="{y + 44}" class="value">'
            f'INT storage {storage:.0f}%</text>'
        )
    body.append(f'<line x1="{left}" y1="{height - 62}" x2="{left + plot_width}" y2="{height - 62}" class="axis-line"/>')
    output = output_dir / "qdq_storage_coverage.svg"
    write_svg(output, svg_document(width, height, body))
    return output


def main() -> int:
    args = parse_args()
    metrics_rows = latest_by_name(read_rows(resolve_project_path(args.metrics_csv)))
    quick_rows = latest_by_name(read_rows(resolve_project_path(args.quick_metrics_csv)))
    latency_rows = latest_by_name(read_rows(resolve_project_path(args.latency_csv)))
    coverage_rows = latest_by_name(read_rows(resolve_project_path(args.coverage_csv)))
    output_dir = resolve_project_path(args.output_dir)

    outputs = [
        draw_full_accuracy(metrics_rows, output_dir),
        draw_accuracy_latency(metrics_rows, latency_rows, output_dir),
        draw_activation_sensitivity(quick_rows, output_dir),
        draw_qdq_storage_coverage(coverage_rows, output_dir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
