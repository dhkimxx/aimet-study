"""Helpers for naming generated model artifacts."""

from __future__ import annotations


def calibration_suffix(calibration_samples: int, default_calibration_samples: int) -> str:
    if calibration_samples == default_calibration_samples:
        return ""
    return f"_calib{calibration_samples}"


def adaround_suffix(
    calibration_samples: int,
    default_calibration_samples: int,
    adaround_samples: int,
    default_adaround_samples: int,
    adaround_iterations: int,
    default_adaround_iterations: int,
) -> str:
    parts: list[str] = []
    if calibration_samples != default_calibration_samples:
        parts.append(f"calib{calibration_samples}")
    if adaround_samples != default_adaround_samples:
        parts.append(f"adar{adaround_samples}")
    if adaround_iterations != default_adaround_iterations:
        parts.append(f"iter{adaround_iterations}")
    if not parts:
        return ""
    return "_" + "_".join(parts)
