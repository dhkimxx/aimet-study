"""Verify that AIMET ONNX, ONNX Runtime, and CUDA visibility are available."""

from __future__ import annotations

import importlib
import json
import sys


def import_status(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {
            "module": module_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "module": module_name,
        "ok": True,
        "version": getattr(module, "__version__", "unknown"),
    }


def main() -> int:
    checks = [
        import_status("aimet_common"),
        import_status("aimet_onnx"),
        import_status("onnx"),
        import_status("onnxruntime"),
        import_status("numpy"),
    ]

    providers = []
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        providers = []

    payload = {
        "python": sys.version,
        "imports": checks,
        "onnxruntime_providers": providers,
        "cuda_provider_available": "CUDAExecutionProvider" in providers,
    }
    print(json.dumps(payload, indent=2))

    return 0 if all(item["ok"] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
