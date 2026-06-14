"""Prepare or validate COCO 2017 val data for the PTQ study."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from aimet_yolo_study.coco import (
    COCO80_NAMES,
    COCO_2017_ANNOTATIONS_URL,
    COCO_2017_VAL_IMAGES_URL,
    list_images,
)
from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.downloads import download_file, extract_zip
from aimet_yolo_study.hashes import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--download", action="store_true", help="Download and extract COCO val assets.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite downloaded archives or extracted files.")
    parser.add_argument("--seed", type=int, default=20260614)
    return parser.parse_args()


def write_dataset_yaml(path: Path, coco_root: Path, image_dir: Path) -> None:
    payload = {
        "path": str(coco_root.resolve()),
        "train": image_dir.name,
        "val": image_dir.name,
        "names": COCO80_NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_calibration_manifest(path: Path, images: list[Path], sample_count: int, seed: int) -> None:
    if sample_count > len(images):
        raise ValueError(f"Calibration sample count {sample_count} exceeds image count {len(images)}")

    rng = random.Random(seed)
    selected = images[:]
    rng.shuffle(selected)
    selected = sorted(selected[:sample_count])

    with path.open("w", encoding="utf-8") as handle:
        for image_path in selected:
            handle.write(f"{image_path.resolve()}\n")


def write_yolo_labels(annotation_file: Path, image_dir: Path) -> dict[str, object]:
    with annotation_file.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    categories = sorted(annotations["categories"], key=lambda item: item["id"])
    category_id_to_index = {category["id"]: index for index, category in enumerate(categories)}
    image_info = {image["id"]: image for image in annotations["images"]}
    labels_by_image_id: dict[int, list[str]] = {image_id: [] for image_id in image_info}

    for annotation in annotations["annotations"]:
        if annotation.get("iscrowd", 0):
            continue

        image = image_info[annotation["image_id"]]
        image_width = float(image["width"])
        image_height = float(image["height"])
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        if width <= 0 or height <= 0:
            continue

        x_center = (x + width / 2.0) / image_width
        y_center = (y + height / 2.0) / image_height
        norm_width = width / image_width
        norm_height = height / image_height
        class_index = category_id_to_index[annotation["category_id"]]
        labels_by_image_id[annotation["image_id"]].append(
            f"{class_index} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        )

    written = 0
    object_count = 0
    for image_id, image in image_info.items():
        label_path = image_dir / f"{Path(image['file_name']).stem}.txt"
        lines = labels_by_image_id[image_id]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
        object_count += len(lines)

    cache_path = image_dir.with_suffix(".cache")
    if cache_path.exists():
        cache_path.unlink()

    return {
        "label_files_written": written,
        "label_objects_written": object_count,
        "cache_removed": str(cache_path),
    }


def validate_coco(image_dir: Path, annotation_file: Path, expected_count: int) -> dict[str, object]:
    images = list_images(image_dir)
    if len(images) != expected_count:
        raise FileNotFoundError(
            f"Expected {expected_count} COCO val images in {image_dir}, found {len(images)}. "
            "Run with --download or check the data path."
        )

    if not annotation_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_file}")

    with annotation_file.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    annotation_image_count = len(annotations.get("images", []))
    if annotation_image_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} image entries in {annotation_file}, found {annotation_image_count}"
        )

    return {
        "image_count": len(images),
        "annotation_image_count": annotation_image_count,
        "annotation_sha256": sha256_file(annotation_file),
    }


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    dataset = config["dataset"]

    coco_root = resolve_project_path(dataset["root_dir"])
    image_dir = resolve_project_path(dataset["image_dir"])
    annotation_file = resolve_project_path(dataset["annotation_file"])
    dataset_yaml = resolve_project_path(dataset["dataset_yaml"])
    expected_count = int(dataset["sample_count"])
    calibration_count = int(dataset["calibration_count"])

    if args.download:
        downloads_dir = coco_root / "downloads"
        val_zip = download_file(COCO_2017_VAL_IMAGES_URL, downloads_dir / "val2017.zip", args.overwrite)
        ann_zip = download_file(
            COCO_2017_ANNOTATIONS_URL,
            downloads_dir / "annotations_trainval2017.zip",
            args.overwrite,
        )
        extract_zip(val_zip, coco_root, args.overwrite)
        extract_zip(ann_zip, coco_root, args.overwrite)

    summary = validate_coco(image_dir, annotation_file, expected_count)
    images = list_images(image_dir)
    write_dataset_yaml(dataset_yaml, coco_root, image_dir)
    write_calibration_manifest(coco_root / "calibration_images.txt", images, calibration_count, args.seed)
    label_summary = write_yolo_labels(annotation_file, image_dir)

    print(json.dumps({**summary, **label_summary, "dataset_yaml": str(dataset_yaml)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
