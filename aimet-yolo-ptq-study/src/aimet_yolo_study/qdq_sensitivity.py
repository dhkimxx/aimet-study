"""Helpers for building QDQ ablation models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, DefaultDict

import onnx


Selector = Callable[[str, str | None, bool], bool]


@dataclass(frozen=True)
class QdqRemovalSummary:
    variant: str
    source_model: str
    output_model: str
    removed_quantize_linear_nodes: int
    removed_dequantize_linear_nodes: int
    selected_tensors: list[str]


def strip_selected_qdq(
    source_model: Path,
    output_model: Path,
    variant: str,
    selector: Selector,
    force: bool = False,
) -> QdqRemovalSummary:
    if output_model.exists() and not force:
        raise FileExistsError(
            f"{output_model} already exists. Pass --force to rebuild the sensitivity model."
        )

    model = onnx.load(str(source_model), load_external_data=False)
    initializers = {initializer.name for initializer in model.graph.initializer}
    producer_op_types = {
        output_name: node.op_type
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }
    consumers_by_input = _consumers_by_input(model.graph.node)

    selected_quantized_outputs: dict[str, str] = {}
    selected_q_indices: set[int] = set()
    selected_tensors: list[str] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "QuantizeLinear" or not node.input or not node.output:
            continue

        tensor_name = node.input[0]
        quantized_name = node.output[0]
        if not _has_dq_consumer(quantized_name, consumers_by_input):
            continue
        if selector(tensor_name, producer_op_types.get(tensor_name), tensor_name in initializers):
            selected_quantized_outputs[quantized_name] = tensor_name
            selected_q_indices.add(index)
            selected_tensors.append(tensor_name)

    removed_dq_count = 0
    new_nodes = []
    for index, node in enumerate(model.graph.node):
        if index in selected_q_indices:
            continue
        if node.op_type == "DequantizeLinear" and node.input and node.input[0] in selected_quantized_outputs:
            new_nodes.append(
                onnx.helper.make_node(
                    "Identity",
                    inputs=[selected_quantized_outputs[node.input[0]]],
                    outputs=list(node.output),
                    name=f"{node.name}_float_identity",
                )
            )
            removed_dq_count += 1
            continue
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))

    return QdqRemovalSummary(
        variant=variant,
        source_model=str(source_model),
        output_model=str(output_model),
        removed_quantize_linear_nodes=len(selected_q_indices),
        removed_dequantize_linear_nodes=removed_dq_count,
        selected_tensors=selected_tensors,
    )


def sensitivity_selector(name: str) -> Selector:
    if name == "head_conv_outputs":
        return lambda tensor, producer, is_initializer: _is_head_conv_output(tensor, producer, is_initializer)
    if name == "head_cv2_outputs":
        return lambda tensor, producer, is_initializer: (
            _is_head_conv_output(tensor, producer, is_initializer)
            and tensor.startswith("/model.23/one2one_cv2.")
        )
    if name == "head_cv3_outputs":
        return lambda tensor, producer, is_initializer: (
            _is_head_conv_output(tensor, producer, is_initializer)
            and tensor.startswith("/model.23/one2one_cv3.")
        )
    if name == "head_scale0_outputs":
        return lambda tensor, producer, is_initializer: (
            _is_head_conv_output(tensor, producer, is_initializer)
            and tensor.startswith(("/model.23/one2one_cv2.0/", "/model.23/one2one_cv3.0/"))
        )
    if name == "head_scale1_outputs":
        return lambda tensor, producer, is_initializer: (
            _is_head_conv_output(tensor, producer, is_initializer)
            and tensor.startswith(("/model.23/one2one_cv2.1/", "/model.23/one2one_cv3.1/"))
        )
    if name == "head_scale2_outputs":
        return lambda tensor, producer, is_initializer: (
            _is_head_conv_output(tensor, producer, is_initializer)
            and tensor.startswith(("/model.23/one2one_cv2.2/", "/model.23/one2one_cv3.2/"))
        )
    if name == "head_final_outputs":
        return lambda tensor, producer, is_initializer: (
            _is_head_conv_output(tensor, producer, is_initializer)
            and tensor.endswith(".2/Conv_output_0")
        )
    if name == "late_neck_20_22":
        return lambda tensor, _producer, is_initializer: (
            not is_initializer and tensor.startswith(("/model.20/", "/model.21/", "/model.22/"))
        )
    if name == "all_conv_outputs":
        return lambda _tensor, producer, is_initializer: not is_initializer and producer == "Conv"
    if name == "all_activations":
        return lambda _tensor, _producer, is_initializer: not is_initializer
    if name == "graph_input":
        return lambda tensor, producer, is_initializer: not is_initializer and producer is None
    raise ValueError(f"Unknown sensitivity variant: {name}")


def _is_head_conv_output(tensor: str, producer: str | None, is_initializer: bool) -> bool:
    return not is_initializer and producer == "Conv" and tensor.startswith("/model.23/")


def _consumers_by_input(nodes) -> DefaultDict[str, list[onnx.NodeProto]]:
    consumers: DefaultDict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    return consumers


def _has_dq_consumer(tensor_name: str, consumers_by_input: dict[str, list[onnx.NodeProto]]) -> bool:
    return any(consumer.op_type == "DequantizeLinear" for consumer in consumers_by_input.get(tensor_name, []))
