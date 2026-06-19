"""Check the local experiment environment before running AIMET studies."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys


MODULES = [
    "numpy",
    "yaml",
    "requests",
    "tqdm",
    "ultralytics",
    "pycocotools",
    "onnx",
    "onnxruntime",
    "aimet_common",
    "aimet_onnx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when CUDAExecutionProvider or nvidia-smi is unavailable.",
    )
    return parser.parse_args()


def module_status(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"name": name, "ok": True, "version": getattr(module, "__version__", "unknown")}


def run_command(command: list[str]) -> dict[str, object]:
    if shutil.which(command[0]) is None:
        return {"command": command, "ok": False, "error": "command not found"}
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"command": command, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": completed.stdout.strip().splitlines()[:12],
    }


def main() -> int:
    args = parse_args()
    providers = []
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        providers = []

    payload = {
        "python_executable": sys.executable,
        "python": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uv": run_command(["uv", "--version"]),
        "modules": [module_status(name) for name in MODULES],
        "onnxruntime_providers": providers,
        "nvidia_smi": run_command(["nvidia-smi"]),
    }
    print(json.dumps(payload, indent=2))

    required_modules = set(MODULES)
    python_ok = sys.version_info[:2] == (3, 10)
    imports_ok = all(item["ok"] for item in payload["modules"] if item["name"] in required_modules)
    cuda_ok = "CUDAExecutionProvider" in providers
    nvidia_ok = bool(payload["nvidia_smi"]["ok"])
    native_ok = python_ok and imports_ok
    gpu_ok = cuda_ok and nvidia_ok
    if args.allow_cpu:
        return 0 if native_ok else 1
    return 0 if native_ok and gpu_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
