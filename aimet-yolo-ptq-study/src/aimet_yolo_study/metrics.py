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

QUANTIZATION_COVERAGE_FIELDNAMES = [
    "experiment_id",
    "experiment_name",
    "model_path",
    "model_sha256",
    "file_size_mb",
    "default_opset",
    "total_nodes",
    "quantize_linear_nodes",
    "dequantize_linear_nodes",
    "qdq_tensor_count",
    "graph_input_count",
    "graph_input_qdq_count",
    "graph_output_count",
    "graph_output_qdq_count",
    "conv_total",
    "conv_input_qdq_count",
    "conv_weight_qdq_count",
    "conv_output_qdq_count",
    "conv_weight_int_storage_count",
    "conv_weight_float_storage_count",
    "conv_input_qdq_pct",
    "conv_weight_qdq_pct",
    "conv_output_qdq_pct",
    "conv_weight_int_storage_pct",
    "conv_weight_float_storage_pct",
    "model23_qdq_tensor_count",
    "model23_conv_qdq_tensor_count",
    "model23_nonconv_qdq_tensor_count",
    "initializer_float_count",
    "initializer_int8_count",
    "initializer_uint8_count",
    "initializer_int16_count",
    "initializer_uint16_count",
    "initializer_int32_count",
    "qlinearconv_count",
    "convinteger_count",
    "qoperator_conv_count",
    "qoperator_conv_weight_int_storage_count",
    "qoperator_conv_weight_int_storage_pct",
    "effective_conv_quantized_count",
    "effective_conv_int_storage_count",
    "effective_conv_quantized_pct",
    "effective_conv_int_storage_pct",
    "encodings_sidecar_exists",
    "coverage_note",
]


def jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
