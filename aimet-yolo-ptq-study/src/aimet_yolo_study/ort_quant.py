"""ONNX Runtime quantization helpers."""

from __future__ import annotations

from pathlib import Path

from aimet_yolo_study.images import preprocess_yolo_image


class ImageCalibrationDataReader:
    """Calibration reader that yields one preprocessed image at a time."""

    def __init__(self, manifest_path: str | Path, input_name: str, image_size: int, max_samples: int | None = None):
        self.input_name = input_name
        self.image_size = image_size
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            self.image_paths = [Path(line.strip()) for line in handle if line.strip()]
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
        self._index = 0

    def get_next(self) -> dict[str, object] | None:
        if self._index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self._index]
        self._index += 1
        return {self.input_name: preprocess_yolo_image(image_path, self.image_size)}

    def rewind(self) -> None:
        self._index = 0
