"""Latency benchmark helpers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from aimet_yolo_study.images import preprocess_yolo_image


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_ms(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": float(np.mean(values)) if values else 0.0,
        f"{prefix}_median_ms": percentile(values, 50),
        f"{prefix}_p90_ms": percentile(values, 90),
        f"{prefix}_p95_ms": percentile(values, 95),
    }


def load_manifest_images(manifest_path: str | Path, count: int) -> list[Path]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()][:count]


def confidence_filter_yolo26(output, threshold: float = 0.25):
    detections = output[0]
    if detections.ndim == 3:
        detections = detections[0]
    if detections.shape[-1] >= 5:
        return detections[detections[:, 4] > threshold]
    return detections


def timed_call(callback) -> float:
    start = time.perf_counter()
    callback()
    end = time.perf_counter()
    return (end - start) * 1000.0


def benchmark_model_only(session, input_name: str, input_tensor, warmup_runs: int, measured_runs: int) -> list[float]:
    for _ in range(warmup_runs):
        session.run(None, {input_name: input_tensor})

    timings = []
    for _ in range(measured_runs):
        timings.append(timed_call(lambda: session.run(None, {input_name: input_tensor})))
    return timings


def benchmark_end_to_end(
    session,
    input_name: str,
    image_paths: list[Path],
    image_size: int,
    warmup_runs: int,
    measured_runs: int,
) -> list[float]:
    if not image_paths:
        raise ValueError("No images available for end-to-end benchmark")

    def run_once(index: int):
        tensor = preprocess_yolo_image(image_paths[index % len(image_paths)], image_size)
        outputs = session.run(None, {input_name: tensor})
        confidence_filter_yolo26(outputs[0])

    for index in range(warmup_runs):
        run_once(index)

    timings = []
    for index in range(measured_runs):
        timings.append(timed_call(lambda idx=index: run_once(idx)))
    return timings
