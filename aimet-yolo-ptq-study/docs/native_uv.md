# Native WSL2 uv 환경

이 프로젝트의 기본 실행 환경은 WSL2 Ubuntu 네이티브 Python 가상환경입니다.

## 전제 조건

- WSL2 Ubuntu x86_64
- NVIDIA Windows driver가 WSL2 GPU를 노출
- `nvidia-smi`가 WSL2 Ubuntu 쉘에서 정상 실행
- `uv`

AIMET ONNX 2.2.0 GPU wheel은 Python 3.10 전용입니다. 시스템 Python이 3.12여도 `uv`가 프로젝트 전용 Python 3.10 venv를 만들도록 `.python-version`과 `pyproject.toml`을 고정했습니다.

## uv 설치

`uv`가 없다면 WSL2 Ubuntu에서 한 번 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

새 터미널을 열거나 shell rc 파일에 PATH를 반영한 뒤 확인합니다.

```bash
uv --version
```

## 프로젝트 환경 생성

```bash
cd /path/to/aimet-yolo-ptq-study
bash scripts/setup_native_uv.sh
```

위 스크립트는 다음을 수행합니다.

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
uv python install 3.10
uv sync
scripts/run_native.sh python scripts/00_check_env.py
```

수동으로 나눠 실행해도 됩니다.

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
uv python install 3.10
uv sync
scripts/run_native.sh python scripts/00_check_env.py
```

## 실행

모든 실험 명령은 `scripts/run_native.sh python`으로 실행합니다. 이 wrapper는 `uv run --frozen`을 호출하기 전에 ONNX Runtime CUDA provider가 필요한 CUDA 11 runtime library 경로를 `LD_LIBRARY_PATH`에 추가합니다.

```bash
scripts/run_native.sh python scripts/01_prepare_coco.py --download
scripts/run_native.sh python scripts/01_prepare_yolo_onnx.py --export
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --eval-samples 100
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100
```

이미 shell을 venv로 진입시켜 import만 확인하고 싶다면 다음도 가능합니다. CUDA ONNX Runtime inference는 `LD_LIBRARY_PATH` 구성이 필요하므로 실험 실행에는 wrapper를 사용합니다.

```bash
source .venv/bin/activate
python scripts/00_check_env.py
```

## GPU 체크

`scripts/00_check_env.py`가 실패하면 먼저 아래 항목을 봅니다.

- `python_version_info`가 `3.10.x`인지 확인
- `modules`에서 `aimet_onnx`, `aimet_common`, `onnxruntime` import가 성공하는지 확인
- `onnxruntime_providers`에 `CUDAExecutionProvider`가 있는지 확인
- `nvidia_smi.ok`가 `true`인지 확인

CPU만으로 import smoke check를 하고 싶을 때는 다음처럼 실행합니다.

```bash
scripts/run_native.sh python scripts/00_check_env.py --allow-cpu
```
