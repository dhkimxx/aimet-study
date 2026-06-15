"""Reusable AIMET QuantSim export workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
    if output_model.exists() and output_encodings.exists() and not force:
        return output_model, output_encodings

    try:
        import onnx
        from aimet_onnx.quantsim import QuantizationSimModel
    except ImportError as exc:
        raise RuntimeError("Missing AIMET ONNX. Run inside the configured AIMET ONNX Docker container.") from exc

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
    sim.export(path=str(export_dir), filename_prefix=filename_prefix)
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
    if output_model.exists() and output_encodings.exists() and adaround_encodings.exists() and not force:
        return output_model, output_encodings, adaround_encodings

    try:
        import onnx
        from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters
        from aimet_onnx.quantsim import QuantizationSimModel
    except ImportError as exc:
        raise RuntimeError("Missing AIMET ONNX. Run inside the configured AIMET ONNX Docker container.") from exc

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
    sim.export(path=str(export_dir), filename_prefix=filename_prefix)
    return output_model, output_encodings, adaround_encodings


def _run_input_batches(session, inputs) -> None:
    for batch in inputs:
        session.run(None, batch)
