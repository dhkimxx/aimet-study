"""Ultralytics validation wrapper used by study scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from aimet_yolo_study.hashes import sha256_file
from aimet_yolo_study.metrics import jsonable

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


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


def eval_run_name(base_name: str, eval_samples: int | None) -> str:
    if eval_samples is None:
        return base_name
    return f"{base_name}_sample{eval_samples}"


def metrics_csv_for_eval(metrics_csv: Path, eval_samples: int | None) -> Path:
    if eval_samples is None:
        return metrics_csv
    return metrics_csv.with_name(f"{metrics_csv.stem}_quick{metrics_csv.suffix}")


def _load_dataset_yaml(dataset_yaml: Path) -> dict[str, Any]:
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in dataset YAML: {dataset_yaml}")
    return data


def _dataset_root(dataset_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(str(data.get("path", dataset_yaml.parent)))
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    return root


def _image_paths_from_file(manifest: Path, dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue

            candidate = Path(raw)
            if candidate.is_absolute():
                images.append(candidate)
                continue

            root_candidate = dataset_root / candidate
            if root_candidate.exists():
                images.append(root_candidate)
            else:
                images.append(manifest.parent / candidate)
    return images


def _image_paths_from_split_value(value: object, dataset_root: Path) -> list[Path]:
    if isinstance(value, (list, tuple)):
        images: list[Path] = []
        for item in value:
            images.extend(_image_paths_from_split_value(item, dataset_root))
        return images

    split_path = Path(str(value))
    if not split_path.is_absolute():
        split_path = dataset_root / split_path

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        return _image_paths_from_file(split_path, dataset_root)
    if split_path.is_file() and split_path.suffix.lower() in IMAGE_SUFFIXES:
        return [split_path]
    if split_path.is_dir():
        return sorted(path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)

    raise FileNotFoundError(f"Could not resolve evaluation split path: {split_path}")


def sample_dataset_yaml(
    dataset_yaml: Path,
    split: str,
    output_dir: Path,
    sample_count: int,
    seed: int,
) -> Path:
    if sample_count <= 0:
        raise ValueError("--eval-samples must be greater than 0")

    data = _load_dataset_yaml(dataset_yaml)
    if split not in data:
        raise KeyError(f"Dataset YAML has no split named '{split}': {dataset_yaml}")

    dataset_root = _dataset_root(dataset_yaml, data)
    images = _image_paths_from_split_value(data[split], dataset_root)
    if not images:
        raise FileNotFoundError(f"No images found for split '{split}' in {dataset_yaml}")

    selected = images[:]
    rng = random.Random(seed)
    rng.shuffle(selected)
    selected = sorted(selected[: min(sample_count, len(selected))])

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{split}_sample{len(selected)}_seed{seed}"
    manifest_path = output_dir / f"{tag}.txt"
    sample_yaml = output_dir / f"{tag}.yaml"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for image_path in selected:
            handle.write(f"{image_path.resolve()}\n")

    sample_data = dict(data)
    sample_data["path"] = str(dataset_root.resolve())
    sample_data[split] = str(manifest_path.resolve())
    with sample_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(sample_data, handle, sort_keys=False, allow_unicode=True)

    return sample_yaml


def run_ultralytics_val(
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    image_size: int,
    batch_size: int,
    device: str,
    output_dir: Path,
    eval_samples: int | None = None,
    eval_seed: int = 20260614,
) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing ultralytics. Install dependencies with: uv sync"
        ) from exc

    effective_dataset_yaml = dataset_yaml
    if eval_samples is not None:
        effective_dataset_yaml = sample_dataset_yaml(
            dataset_yaml=dataset_yaml,
            split=split,
            output_dir=output_dir,
            sample_count=eval_samples,
            seed=eval_seed,
        )

    model = YOLO(str(model_path))
    return model.val(
        data=str(effective_dataset_yaml),
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
