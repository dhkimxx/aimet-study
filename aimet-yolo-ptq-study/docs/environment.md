# 환경 노트

## Host 구성

예상 실행 구조:

```text
Windows host
  WSL2 Ubuntu
    Docker Engine
    NVIDIA Container Toolkit
    AIMET official/prebuilt ONNX GPU container
```

## 필수 확인

실험을 시작하기 전에 WSL2 Ubuntu에서 다음을 확인합니다.

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

그다음 AIMET 컨테이너를 시작합니다.

```bash
bash docker/build_runtime_image.sh
bash docker/run_aimet_onnx_gpu.sh
```

AIMET 컨테이너 내부에서는 다음 검증을 실행합니다.

```bash
python -m pip install -r requirements.txt
python docker/verify_aimet.py
python scripts/00_check_env.py
```

## AIMET Image

`docker/run_aimet_onnx_gpu.sh`에서 사용하는 기본 런타임 이미지는 다음과 같습니다.

```text
aimet-yolo-onnx-gpu:2.2.0
```

이 이미지는 AIMET 문서에서 안내하는 prebuilt AIMET development image를 기반으로 빌드합니다.

```text
artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.onnx-gpu
```

prebuilt development image에는 ONNX/CUDA 의존성 스택이 들어 있습니다. AIMET 자체는 `docker/build_runtime_image.sh` 실행 중 공식 2.2.0 ONNX GPU wheel로 설치합니다.

이미지 태그를 바꿔야 할 때는 다음처럼 실행합니다.

```bash
AIMET_IMAGE="<official-image>:<tag>" bash docker/run_aimet_onnx_gpu.sh
```

## 버전 기록

각 실험은 다음 정보를 함께 기록합니다.

- AIMET version
- ONNX Runtime version
- ONNX version
- CUDA-visible provider list
- Docker image name and tag
- GPU name and driver version
- Model file hash
- Dataset annotation file hash

## 프로젝트 의존성

공식 AIMET 이미지는 AIMET 실행에 필요한 기반을 제공합니다. 프로젝트의 `requirements.txt`는 스터디용 유틸리티를 추가합니다.

- `ultralytics`: YOLO26 모델 export와 기준선 validation
- `pycocotools`: COCO metric workflow
- `PyYAML`: 프로젝트 config 읽기
- `requests`, `tqdm`: 안정적인 다운로드와 진행률 표시
