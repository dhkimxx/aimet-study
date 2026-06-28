"""AIMET-specific helpers kept separate from scripts for easier study."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import numpy as np

from aimet_yolo_study.images import preprocess_yolo_image


def build_providers(
    device: str,
    provider: str = "cuda",
    tensorrt_cache_path: str | Path | None = None,
    tensorrt_fp16: bool = False,
    tensorrt_int8: bool = False,
) -> list[object]:
    provider = provider.lower()
    if device.lower() == "cpu" or provider == "cpu":
        return ["CPUExecutionProvider"]
    device_id = int(device)
    if provider == "tensorrt":
        options: dict[str, object] = {
            "device_id": device_id,
            "trt_engine_cache_enable": True,
        }
        if tensorrt_cache_path is not None:
            options["trt_engine_cache_path"] = str(tensorrt_cache_path)
        if tensorrt_fp16:
            options["trt_fp16_enable"] = True
        if tensorrt_int8:
            options["trt_int8_enable"] = True
        return [
            ("TensorrtExecutionProvider", options),
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ]
    if provider != "cuda":
        raise ValueError(f"Unsupported provider {provider!r}; expected cuda, tensorrt, or cpu")
    return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]


def dummy_input(input_name: str, input_shape: list[int]) -> dict[str, np.ndarray]:
    return {input_name: np.zeros(input_shape, dtype=np.float32)}


def calibration_input_dicts(
    manifest_path: str | Path,
    input_name: str,
    image_size: int,
    sample_count: int,
) -> Iterable[dict[str, np.ndarray]]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        image_paths = [Path(line.strip()) for line in handle if line.strip()]

    for image_path in image_paths[:sample_count]:
        yield {input_name: preprocess_yolo_image(image_path, image_size)}


def resolve_quant_scheme(name: str):
    """Resolve quant scheme names across AIMET 1.x and 2.x namespaces."""
    candidates = (
        "aimet_onnx.common.defs",
        "aimet_common.defs",
    )
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["QuantScheme"])
            quant_scheme = module.QuantScheme
        except Exception:
            continue

        if hasattr(quant_scheme, name):
            return getattr(quant_scheme, name)

        aliases = {
            "post_training_tf_enhanced": ("tf_enhanced", "tf-enhanced", "min_max"),
            "post_training_tf": ("tf", "min_max"),
        }
        for alias in aliases.get(name, ()):
            if hasattr(quant_scheme, alias):
                return getattr(quant_scheme, alias)

    aliases = {
        "post_training_tf_enhanced": "tf-enhanced",
        "post_training_tf": "tf",
    }
    return aliases.get(name, name)


def resolve_int8_type():
    """Return an AIMET 2.x qtype string that also works as a readable fallback."""
    return "int8"


def override_quant_bitwidths(
    quant_config: dict[str, object],
    activation_bitwidth: int | None,
    weight_bitwidth: int | None,
) -> dict[str, object]:
    """Return a quantization config with optional CLI bitwidth overrides."""
    updated_config = copy.deepcopy(quant_config)
    defaults = updated_config["defaults"]
    if activation_bitwidth is not None:
        defaults["activation_bitwidth"] = _validate_bitwidth(activation_bitwidth, "activation")
    if weight_bitwidth is not None:
        defaults["weight_bitwidth"] = _validate_bitwidth(weight_bitwidth, "weight")
    return updated_config


def _validate_bitwidth(bitwidth: int, label: str) -> int:
    if bitwidth not in {8, 16}:
        raise ValueError(f"{label} bitwidth must be 8 or 16, got {bitwidth}")
    return bitwidth
