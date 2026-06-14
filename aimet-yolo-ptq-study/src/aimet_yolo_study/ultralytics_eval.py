"""Ultralytics validation wrapper used by study scripts."""

from __future__ import annotations

from pathlib import Path

from aimet_yolo_study.hashes import sha256_file
from aimet_yolo_study.metrics import jsonable


def extract_box_metrics(metrics: object) -> dict[str, object]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}

    return {
        "box_map_50_95": getattr(box, "map", ""),
        "box_map_50": getattr(box, "map50", ""),
        "box_map_75": getattr(box, "map75", ""),
        "precision": getattr(box, "mp", ""),
        "recall": getattr(box, "mr", ""),
    }


def run_ultralytics_val(
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    image_size: int,
    batch_size: int,
    device: str,
    output_dir: Path,
) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Missing ultralytics. Install dependencies with: python -m pip install -r requirements.txt") from exc

    model = YOLO(str(model_path))
    return model.val(
        data=str(dataset_yaml),
        imgsz=image_size,
        batch=batch_size,
        device=device,
        split=split,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        save_json=True,
        verbose=True,
    )


def build_accuracy_row(
    experiment_id: str,
    experiment_name: str,
    uses_aimet: bool,
    model_path: Path,
    metrics: object,
) -> dict[str, object]:
    row = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "uses_aimet": uses_aimet,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "ap_small": "",
        "ap_medium": "",
        "ap_large": "",
    }
    row.update(extract_box_metrics(metrics))
    return {key: jsonable(value) for key, value in row.items()}
