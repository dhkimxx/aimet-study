"""Export the selected Ultralytics YOLO checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from aimet_yolo_study.config import load_experiment_config, resolve_project_path
from aimet_yolo_study.hashes import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--export", action="store_true", help="Export the configured checkpoint to ONNX.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing ONNX model.")
    parser.add_argument("--opset", type=int, default=None, help="Optional ONNX opset override.")
    parser.add_argument("--simplify", action="store_true", help="Run ONNX simplification during export.")
    return parser.parse_args()


def export_onnx(source_checkpoint: str, output_path: Path, image_size: int, opset: int | None, simplify: bool) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Missing ultralytics. Install project dependencies with: python -m pip install -r requirements.txt") from exc

    model = YOLO(source_checkpoint)
    export_kwargs = {
        "format": "onnx",
        "imgsz": image_size,
        "batch": 1,
        "dynamic": False,
        "simplify": simplify,
    }
    if opset is not None:
        export_kwargs["opset"] = opset

    exported = Path(model.export(**export_kwargs))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if exported.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(exported), output_path)

    return output_path


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    model_config = config["model"]
    input_shape = model_config["input_shape"]
    image_size = int(input_shape[-1])
    output_path = resolve_project_path(model_config["path"])
    source_checkpoint = str(model_config["source_checkpoint"])

    if output_path.exists() and not args.force:
        print(
            json.dumps(
                {
                    "status": "exists",
                    "model_path": str(output_path),
                    "sha256": sha256_file(output_path),
                },
                indent=2,
            )
        )
        return 0

    if not args.export:
        raise FileNotFoundError(f"Missing {output_path}. Run again with --export to create it.")

    exported_path = export_onnx(source_checkpoint, output_path, image_size, args.opset, args.simplify)
    metadata_path = exported_path.with_suffix(".metadata.json")
    metadata = {
        "source_checkpoint": source_checkpoint,
        "model_path": str(exported_path),
        "input_shape": input_shape,
        "expected_yolo26_detection_output": "N x 300 x 6, xyxy + confidence + class_id",
        "sha256": sha256_file(exported_path),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
