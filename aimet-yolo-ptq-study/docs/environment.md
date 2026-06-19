# 환경 노트

## 기본 실행 구조

현재 기본 환경은 WSL2 Ubuntu 네이티브 venv입니다.

```text
Windows host
  WSL2 Ubuntu
    uv-managed Python 3.10 .venv
      AIMET ONNX 2.2.0 GPU wheel
      ONNX Runtime CUDAExecutionProvider
```

## 필수 확인

실험을 시작하기 전에 WSL2 Ubuntu에서 다음을 확인합니다.

```bash
nvidia-smi
uv --version
```

`uv`가 없다면 먼저 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

## Native uv 환경 생성

```bash
cd /path/to/aimet-yolo-ptq-study
bash scripts/setup_native_uv.sh
```

수동으로 진행할 때는 다음 명령을 사용합니다.

```bash
uv python install 3.10
uv sync
scripts/run_native.sh python scripts/00_check_env.py
```

`scripts/setup_native_uv.sh`는 sandbox나 제한된 WSL 환경에서도 동작하도록 `UV_CACHE_DIR`을 `.uv-cache`, `UV_PYTHON_INSTALL_DIR`을 `.uv-python`으로 잡습니다.

실험 실행은 `scripts/run_native.sh python ...` 형식을 사용합니다. 이 wrapper는 `uv run --frozen`과 동일하게 lock을 따르며, ONNX Runtime CUDA provider가 필요한 CUDA 11 runtime library 경로를 `LD_LIBRARY_PATH`에 추가합니다.

`pyproject.toml`은 다음 핵심 조건을 고정합니다.

- Python `>=3.10,<3.11`
- AIMET ONNX `2.2.0+cu118`
- ONNX `1.16.*`
- ONNX Runtime GPU
- CUDA 11 runtime libraries from NVIDIA Python wheels
- PyTorch `2.1.2+cu118`, torchvision `0.16.2+cu118`

## AIMET Native Package

사용하는 AIMET wheel은 다음입니다.

```text
https://github.com/quic/aimet/releases/download/2.2.0/aimet_onnx-2.2.0+cu118-cp310-cp310-manylinux_2_34_x86_64.whl
```

AIMET 2.2.0 ONNX GPU package는 Python 3.10과 CUDA 11.x 계열 wheel을 전제로 합니다. 시스템 Python이 3.12여도 `uv`가 `.venv`에 Python 3.10을 설치해서 사용합니다.

## 검증 기준

`scripts/00_check_env.py`는 다음을 확인합니다.

- Python executable과 version
- `uv --version`
- `numpy`, `onnx`, `onnxruntime`, `aimet_common`, `aimet_onnx`, `ultralytics` 등 import
- ONNX Runtime provider 목록
- `nvidia-smi` 실행 결과

정상 GPU 환경이면 `onnxruntime_providers`에 `CUDAExecutionProvider`가 포함되어야 합니다.

## 버전 기록

각 실험은 다음 정보를 함께 기록합니다.

- AIMET version
- ONNX Runtime version
- ONNX version
- CUDA-visible provider list
- GPU name and driver version
- Model file hash
- Dataset annotation file hash
