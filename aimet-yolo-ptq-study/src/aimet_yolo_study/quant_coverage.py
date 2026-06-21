"""ONNX quantization coverage analysis helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict

import onnx

from aimet_yolo_study.hashes import sha256_file


def analyze_quantization_coverage(model_path: Path, experiment_id: str, experiment_name: str) -> dict[str, object]:
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    node_counts = Counter(node.op_type for node in graph.node)
    initializer_dtype_counts = Counter(initializer.data_type for initializer in graph.initializer)
    node_by_output = _node_by_output(graph.node)
    consumers_by_input = _consumers_by_input(graph.node)
    graph_inputs = {value.name for value in graph.input}
    graph_outputs = {value.name for value in graph.output}
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    qdq_tensor_names = _qdq_tensor_names(graph.node)
    producer_op_types = {
        output_name: node.op_type
        for node in graph.node
        for output_name in node.output
        if output_name
    }

    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]
    conv_input_qdq_count = sum(
        _is_dequantized_tensor(node.input[0], node_by_output) for node in conv_nodes if node.input
    )
    conv_weight_qdq_count = sum(
        _is_dequantized_tensor(node.input[1], node_by_output) for node in conv_nodes if len(node.input) > 1
    )
    conv_output_qdq_count = sum(
        _is_quantized_tensor(node.output[0], consumers_by_input) for node in conv_nodes if node.output
    )
    conv_weight_storage = [
        _conv_weight_storage_kind(node, node_by_output, initializers)
        for node in conv_nodes
        if len(node.input) > 1
    ]
    conv_weight_int_storage_count = conv_weight_storage.count("int")
    conv_weight_float_storage_count = conv_weight_storage.count("float")

    model23_qdq_tensors = {name for name in qdq_tensor_names if name.startswith("/model.23/")}
    model23_conv_qdq_tensors = {
        name for name in model23_qdq_tensors if producer_op_types.get(name) == "Conv"
    }
    model23_nonconv_qdq_tensors = model23_qdq_tensors - model23_conv_qdq_tensors
    graph_input_qdq_count = sum(_is_quantized_tensor(name, consumers_by_input) for name in graph_inputs)
    graph_output_qdq_count = sum(_is_dequantized_tensor(name, node_by_output) for name in graph_outputs)

    row = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "file_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
        "default_opset": _default_opset(model),
        "total_nodes": len(graph.node),
        "quantize_linear_nodes": node_counts["QuantizeLinear"],
        "dequantize_linear_nodes": node_counts["DequantizeLinear"],
        "qdq_tensor_count": len(qdq_tensor_names),
        "graph_input_count": len(graph_inputs),
        "graph_input_qdq_count": graph_input_qdq_count,
        "graph_output_count": len(graph_outputs),
        "graph_output_qdq_count": graph_output_qdq_count,
        "conv_total": len(conv_nodes),
        "conv_input_qdq_count": conv_input_qdq_count,
        "conv_weight_qdq_count": conv_weight_qdq_count,
        "conv_output_qdq_count": conv_output_qdq_count,
        "conv_weight_int_storage_count": conv_weight_int_storage_count,
        "conv_weight_float_storage_count": conv_weight_float_storage_count,
        "conv_input_qdq_pct": _pct(conv_input_qdq_count, len(conv_nodes)),
        "conv_weight_qdq_pct": _pct(conv_weight_qdq_count, len(conv_nodes)),
        "conv_output_qdq_pct": _pct(conv_output_qdq_count, len(conv_nodes)),
        "conv_weight_int_storage_pct": _pct(conv_weight_int_storage_count, len(conv_nodes)),
        "conv_weight_float_storage_pct": _pct(conv_weight_float_storage_count, len(conv_nodes)),
        "model23_qdq_tensor_count": len(model23_qdq_tensors),
        "model23_conv_qdq_tensor_count": len(model23_conv_qdq_tensors),
        "model23_nonconv_qdq_tensor_count": len(model23_nonconv_qdq_tensors),
        "initializer_float_count": initializer_dtype_counts[onnx.TensorProto.FLOAT],
        "initializer_int8_count": initializer_dtype_counts[onnx.TensorProto.INT8],
        "initializer_uint8_count": initializer_dtype_counts[onnx.TensorProto.UINT8],
        "initializer_int16_count": initializer_dtype_counts[onnx.TensorProto.INT16],
        "initializer_uint16_count": initializer_dtype_counts[onnx.TensorProto.UINT16],
        "initializer_int32_count": initializer_dtype_counts[onnx.TensorProto.INT32],
        "encodings_sidecar_exists": model_path.with_suffix(".encodings").exists(),
    }
    row["coverage_note"] = _coverage_note(row)
    return row


def _node_by_output(nodes) -> dict[str, onnx.NodeProto]:
    return {
        output_name: node
        for node in nodes
        for output_name in node.output
        if output_name
    }


def _consumers_by_input(nodes) -> DefaultDict[str, list[onnx.NodeProto]]:
    consumers: DefaultDict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    return consumers


def _qdq_tensor_names(nodes) -> set[str]:
    return {
        node.input[0]
        for node in nodes
        if node.op_type == "QuantizeLinear" and node.input
    }


def _is_dequantized_tensor(tensor_name: str, node_by_output: dict[str, onnx.NodeProto]) -> bool:
    producer = node_by_output.get(tensor_name)
    return producer is not None and producer.op_type == "DequantizeLinear"


def _is_quantized_tensor(
    tensor_name: str,
    consumers_by_input: dict[str, list[onnx.NodeProto]],
) -> bool:
    return any(consumer.op_type == "QuantizeLinear" for consumer in consumers_by_input.get(tensor_name, []))


def _conv_weight_storage_kind(
    conv_node: onnx.NodeProto,
    node_by_output: dict[str, onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> str:
    weight_input = conv_node.input[1]
    dtype = _initializer_dtype(weight_input, node_by_output, initializers)
    if dtype in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8, onnx.TensorProto.INT16, onnx.TensorProto.UINT16}:
        return "int"
    if dtype == onnx.TensorProto.FLOAT:
        return "float"
    return "other"


def _initializer_dtype(
    tensor_name: str,
    node_by_output: dict[str, onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> int | None:
    initializer = initializers.get(tensor_name)
    if initializer is not None:
        return int(initializer.data_type)

    producer = node_by_output.get(tensor_name)
    if producer is None:
        return None
    if producer.op_type == "DequantizeLinear" and producer.input:
        return _initializer_dtype(producer.input[0], node_by_output, initializers)
    if producer.op_type == "QuantizeLinear" and producer.input:
        return _initializer_dtype(producer.input[0], node_by_output, initializers)
    return None


def _pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count * 100.0 / total, 2)


def _default_opset(model: onnx.ModelProto) -> int | None:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return None


def _coverage_note(row: dict[str, object]) -> str:
    if row["quantize_linear_nodes"] == 0 and row["dequantize_linear_nodes"] == 0:
        if row["encodings_sidecar_exists"]:
            return "encodings_sidecar_only_not_ort_int8"
        return "fp32_no_qdq"
    if row["graph_output_qdq_count"] == 0:
        return "partial_qdq_float_graph_output"
    return "standard_qdq"
