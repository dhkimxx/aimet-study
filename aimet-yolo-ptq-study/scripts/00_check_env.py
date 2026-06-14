"""Check the local experiment environment before running AIMET studies."""

from __future__ import annotations

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
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=10)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"command": command, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"command": command, "ok": True, "output": output.strip().splitlines()[:12]}


def main() -> int:
    providers = []
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        providers = []

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": [module_status(name) for name in MODULES],
        "onnxruntime_providers": providers,
        "nvidia_smi": run_command(["nvidia-smi"]),
    }
    print(json.dumps(payload, indent=2))

    required_modules = {"numpy", "yaml", "requests", "tqdm", "ultralytics", "pycocotools", "onnx", "onnxruntime"}
    imports_ok = all(item["ok"] for item in payload["modules"] if item["name"] in required_modules)
    cuda_ok = "CUDAExecutionProvider" in providers
    return 0 if imports_ok and cuda_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
