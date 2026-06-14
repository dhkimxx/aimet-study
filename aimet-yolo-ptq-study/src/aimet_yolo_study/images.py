"""Image preprocessing helpers for ONNX calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def letterbox_rgb(image: Image.Image, size: int, pad_value: int = 114) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = image.resize((resized_width, resized_height), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def preprocess_yolo_image(path: str | Path, size: int) -> np.ndarray:
    image = Image.open(path)
    image = letterbox_rgb(image, size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return array[None, ...].astype(np.float32)
