"""Metric field definitions and serialization helpers."""

from __future__ import annotations

from typing import Any


ACCURACY_FIELDNAMES = [
    "experiment_id",
    "experiment_name",
    "uses_aimet",
    "model_path",
    "model_sha256",
    "box_map_50_95",
    "box_map_50",
    "box_map_75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "precision",
    "recall",
]

LATENCY_FIELDNAMES = [
    "experiment_id",
    "experiment_name",
    "model_path",
    "provider",
    "warmup_runs",
    "measured_runs",
    "model_only_mean_ms",
    "model_only_median_ms",
    "model_only_p90_ms",
    "model_only_p95_ms",
    "end_to_end_mean_ms",
    "end_to_end_median_ms",
    "end_to_end_p90_ms",
    "end_to_end_p95_ms",
]


def jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
