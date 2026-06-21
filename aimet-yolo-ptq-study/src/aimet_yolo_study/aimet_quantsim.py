"""Reusable AIMET QuantSim export workflow."""

from __future__ import annotations

from collections.abc import Callable
import copy
from pathlib import Path

import numpy as np
import onnx

from aimet_yolo_study.aimet_utils import calibration_input_dicts, dummy_input, resolve_quant_scheme


ModelTransform = Callable[[object], None]
SimTransform = Callable[[object], None]


def export_quantsim_model(
    fp32_model: Path,
    export_dir: Path,
    filename_prefix: str,
    input_name: str,
    input_shape: list[int],
    image_size: int,
    manifest: Path,
    quant_config: dict[str, object],
    calibration_samples: int,
    device: str,
    force: bool,
    model_transform: ModelTransform | None = None,
    sim_transform: SimTransform | None = None,
) -> tuple[Path, Path]:
    output_model = export_dir / f"{filename_prefix}.onnx"
    output_encodings = export_dir / f"{filename_prefix}.encodings"
    if output_model.exists() and output_encodings.exists() and _onnx_has_standard_qdq(output_model) and not force:
        return output_model, output_encodings

    try:
        from aimet_onnx.quantsim import QuantizationSimModel
    except ImportError as exc:
        raise RuntimeError("Missing AIMET ONNX. Install the native uv environment with: uv sync") from exc

    model = onnx.load(str(fp32_model))
    if model_transform is not None:
        model_transform(model)

    export_dir.mkdir(parents=True, exist_ok=True)
    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input(input_name, input_shape),
        quant_scheme=resolve_quant_scheme(str(quant_config["defaults"]["quant_scheme"])),
        rounding_mode=str(quant_config["defaults"]["rounding_mode"]),
        default_param_bw=int(quant_config["defaults"]["weight_bitwidth"]),
        default_activation_bw=int(quant_config["defaults"]["activation_bitwidth"]),
        use_cuda=device.lower() != "cpu",
        device=0 if device.lower() == "cpu" else int(device),
        path=str(export_dir / "tmp"),
    )
    if sim_transform is not None:
        sim_transform(sim)

    sim.compute_encodings(_run_calibration, (manifest, input_name, image_size, calibration_samples))
    _export_encodings(sim, output_encodings)
    _export_standard_qdq_model(sim, output_model)
    return output_model, output_encodings


def _run_calibration(session, args) -> None:
    manifest, input_name, image_size, calibration_samples = args
    for inputs in calibration_input_dicts(
        manifest_path=manifest,
        input_name=input_name,
        image_size=image_size,
        sample_count=calibration_samples,
    ):
        session.run(None, inputs)


def export_adaround_quantsim_model(
    fp32_model: Path,
    export_dir: Path,
    filename_prefix: str,
    input_name: str,
    input_shape: list[int],
    image_size: int,
    manifest: Path,
    quant_config: dict[str, object],
    calibration_samples: int,
    adaround_samples: int,
    adaround_iterations: int,
    device: str,
    force: bool,
) -> tuple[Path, Path, Path]:
    output_model = export_dir / f"{filename_prefix}.onnx"
    output_encodings = export_dir / f"{filename_prefix}.encodings"
    adaround_prefix = f"{filename_prefix}.adaround"
    adaround_encodings = export_dir / f"{adaround_prefix}.encodings"
    if (
        output_model.exists()
        and output_encodings.exists()
        and adaround_encodings.exists()
        and _onnx_has_standard_qdq(output_model)
        and not force
    ):
        return output_model, output_encodings, adaround_encodings

    try:
        from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters
        from aimet_onnx.quantsim import QuantizationSimModel
    except ImportError as exc:
        raise RuntimeError("Missing AIMET ONNX. Install the native uv environment with: uv sync") from exc

    export_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(fp32_model))
    adaround_inputs = list(
        calibration_input_dicts(
            manifest_path=manifest,
            input_name=input_name,
            image_size=image_size,
            sample_count=adaround_samples,
        )
    )
    params = AdaroundParameters(
        data_loader=adaround_inputs,
        num_batches=len(adaround_inputs),
        default_num_iterations=adaround_iterations,
        forward_fn=_run_input_batches,
        forward_pass_callback_args=adaround_inputs,
    )
    adarounded_model = Adaround.apply_adaround(
        model=model,
        params=params,
        path=str(export_dir),
        filename_prefix=adaround_prefix,
        default_param_bw=int(quant_config["defaults"]["weight_bitwidth"]),
        default_quant_scheme=resolve_quant_scheme(str(quant_config["defaults"]["quant_scheme"])),
        use_cuda=device.lower() != "cpu",
        device=0 if device.lower() == "cpu" else int(device),
    )
    if hasattr(adarounded_model, "model"):
        adarounded_model = adarounded_model.model

    sim = QuantizationSimModel(
        model=adarounded_model,
        dummy_input=dummy_input(input_name, input_shape),
        quant_scheme=resolve_quant_scheme(str(quant_config["defaults"]["quant_scheme"])),
        rounding_mode=str(quant_config["defaults"]["rounding_mode"]),
        default_param_bw=int(quant_config["defaults"]["weight_bitwidth"]),
        default_activation_bw=int(quant_config["defaults"]["activation_bitwidth"]),
        use_cuda=device.lower() != "cpu",
        device=0 if device.lower() == "cpu" else int(device),
        path=str(export_dir / "tmp"),
    )
    sim.set_and_freeze_param_encodings(str(adaround_encodings))
    sim.compute_encodings(_run_calibration, (manifest, input_name, image_size, calibration_samples))
    _export_encodings(sim, output_encodings)
    _export_standard_qdq_model(sim, output_model)
    return output_model, output_encodings, adaround_encodings


def _run_input_batches(session, inputs) -> None:
    for batch in inputs:
        session.run(None, batch)


def _export_encodings(sim, output_encodings: Path) -> None:
    # AIMET ONNX 2.2.0 public export removes QcQuantizeOp nodes before saving
    # ONNX. Keep the encodings export, but save a standard QDQ model ourselves
    # for ONNX Runtime accuracy evaluation.
    sim._export_encodings(str(output_encodings))  # pylint: disable=protected-access


def _export_standard_qdq_model(sim, output_model: Path) -> None:
    model = copy.deepcopy(sim.model.model)
    quantizers = sim.get_qc_quantize_op()
    producer_op_types = {
        output_name: node.op_type
        for node in model.graph.node
        if node.op_type != "QcQuantizeOp"
        for output_name in node.output
    }
    new_initializers = list(model.graph.initializer)
    new_nodes = []
    requires_int16_qdq = False

    for node in model.graph.node:
        if node.op_type != "QcQuantizeOp":
            new_nodes.append(copy.deepcopy(node))
            continue

        tensor_name = node.input[0]
        quantizer = quantizers.get(tensor_name)
        if (
            quantizer is None
            or not quantizer.enabled
            or quantizer.get_encodings() is None
            or _should_skip_standard_qdq(tensor_name, producer_op_types.get(tensor_name))
        ):
            new_nodes.append(
                onnx.helper.make_node(
                    "Identity",
                    inputs=[tensor_name],
                    outputs=[node.output[0]],
                    name=f"{node.name}_identity",
                )
            )
            continue

        scale_name = f"{node.name}_scale"
        zero_point_name = f"{node.name}_zero_point"
        quantized_name = f"{node.output[0]}_quantized"
        scale, zero_point = _qdq_params_from_quantizer(quantizer)
        requires_int16_qdq = requires_int16_qdq or zero_point.dtype in {np.dtype(np.int16), np.dtype(np.uint16)}

        new_initializers.append(onnx.numpy_helper.from_array(scale, scale_name))
        new_initializers.append(onnx.numpy_helper.from_array(zero_point, zero_point_name))

        axis = _qdq_axis(quantizer)
        q_attrs = {"axis": axis} if axis is not None else {}
        dq_attrs = {"axis": axis} if axis is not None else {}
        new_nodes.extend(
            [
                onnx.helper.make_node(
                    "QuantizeLinear",
                    inputs=[tensor_name, scale_name, zero_point_name],
                    outputs=[quantized_name],
                    name=f"{node.name}_QuantizeLinear",
                    **q_attrs,
                ),
                onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs=[quantized_name, scale_name, zero_point_name],
                    outputs=[node.output[0]],
                    name=f"{node.name}_DequantizeLinear",
                    **dq_attrs,
                ),
            ]
        )

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)
    del model.graph.node[:]
    model.graph.node.extend(_topological_sort_nodes(model, new_nodes))
    if requires_int16_qdq:
        model = _convert_default_opset(model, 21)
    onnx.checker.check_model(model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_model))


def _qdq_params_from_quantizer(quantizer) -> tuple[np.ndarray, np.ndarray]:
    scale = quantizer._get_scale().astype(np.float32)  # pylint: disable=protected-access
    offset = quantizer._get_offset()  # pylint: disable=protected-access
    if offset is None:
        offset = np.zeros_like(scale, dtype=np.float32)
    else:
        offset = offset.astype(np.float32)

    bitwidth = int(getattr(quantizer, "bitwidth", 8))
    if bitwidth not in {8, 16}:
        raise ValueError(f"Standard QDQ export supports 8-bit and 16-bit quantizers, got {bitwidth}")

    if bool(quantizer.use_symmetric_encodings):
        zero_point_dtype = np.int8 if bitwidth == 8 else np.int16
        return scale, np.zeros(scale.shape, dtype=zero_point_dtype)

    if bitwidth == 8:
        return scale, np.clip(np.rint(-offset), 0, 255).astype(np.uint8)
    return scale, np.clip(np.rint(-offset), 0, 65535).astype(np.uint16)


def _convert_default_opset(model: onnx.ModelProto, minimum_version: int) -> onnx.ModelProto:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            if opset.version >= minimum_version:
                return model
            return onnx.version_converter.convert_version(model, minimum_version)
    return onnx.version_converter.convert_version(model, minimum_version)


def _should_skip_standard_qdq(tensor_name: str, producer_op_type: str | None) -> bool:
    if tensor_name == "output0":
        return True
    if tensor_name.startswith("/model.23/") and producer_op_type != "Conv":
        return True
    return False


def _qdq_axis(quantizer) -> int | None:
    quant_info = quantizer.quant_info
    if not bool(quant_info.usePerChannelMode):
        return None
    return int(quant_info.channelAxis)


def _topological_sort_nodes(model: onnx.ModelProto, nodes: list[onnx.NodeProto]) -> list[onnx.NodeProto]:
    known = {value.name for value in model.graph.input}
    known.update(initializer.name for initializer in model.graph.initializer)
    sorted_nodes = []
    pending = list(nodes)

    while pending:
        next_pending = []
        progressed = False
        for node in pending:
            if all(not input_name or input_name in known for input_name in node.input):
                sorted_nodes.append(node)
                known.update(output_name for output_name in node.output if output_name)
                progressed = True
            else:
                next_pending.append(node)

        if not progressed:
            sorted_nodes.extend(next_pending)
            break
        pending = next_pending

    return sorted_nodes


def _onnx_has_standard_qdq(model_path: Path) -> bool:
    model = onnx.load(str(model_path), load_external_data=False)
    op_types = {node.op_type for node in model.graph.node}
    return "QuantizeLinear" in op_types and "DequantizeLinear" in op_types
