# AIMET YOLO PTQ Study

이 프로젝트는 YOLO ONNX 모델을 대상으로 AIMET PTQ를 공부하고, 양자화 방식별 성능 변화를 재현 가능하게 비교하기 위한 실험 스캐폴드입니다.

저장소 루트 기준 경로: `aimet-yolo-ptq-study/`

핵심 질문:

> AIMET을 쓰지 않은 일반 INT8 기준선부터 QuantSim, CLE, AdaRound, AutoQuant까지 활용도를 높이면 YOLO의 정확도, 레이턴시, 메모리 사용량은 어떻게 달라지는가?

## 확정한 의사결정

| 영역 | 결정 |
| --- | --- |
| 스터디 목적 | AIMET PTQ 동작을 ablation으로 학습 |
| 모델 | 최신 Ultralytics YOLO pretrained ONNX 계열, 시작점은 YOLO26 |
| 데이터셋 | COCO 2017 val, 5천 장 |
| 입력 | batch size 1, 640x640 |
| 1차 정확도 지표 | COCO box mAP50-95 |
| 보조 지표 | mAP50, mAP75, AP small/medium/large, precision, recall |
| 레이턴시 | model-only와 end-to-end 모두 측정 |
| 환경 | Windows host, WSL2 Ubuntu, NVIDIA GPU |
| AIMET 구성 | 공식 또는 prebuilt AIMET 이미지 기반 Docker 우선 |
| 범위 | 1단계는 PTQ만 수행, QAT 제외 |

## 실험 매트릭스

| ID | 실험 | 목적 |
| --- | --- | --- |
| A | FP32 ONNX 기준선 | 정확도, 레이턴시, 메모리 기준값 확보 |
| B | no-AIMET naive ONNX INT8 | 일반 ONNX Runtime 양자화와 비교 |
| C | AIMET QuantSim PTQ | 캘리브레이션, encoding, simulated quantization 학습 |
| D | AIMET CLE + QuantSim | range equalization 효과 측정 |
| E | AIMET AdaRound + QuantSim | weight rounding 최적화 효과 측정 |
| F | AIMET AutoQuant | AIMET 자동 PTQ 워크플로와 비교 |

## 프로젝트 구조

```text
aimet-yolo-ptq-study/
  configs/
    experiment.yaml
    quantization.yaml
  docker/
    Dockerfile.runtime
    build_runtime_image.sh
    run_aimet_onnx_gpu.sh
    verify_aimet.py
  data/
    README.md
  docs/
    decision_log.md
    environment.md
  models/
    README.md
  reports/
    aimet_ptq_study.md
  results/
    README.md
  scripts/
    00_check_env.py
    01_prepare_coco.py
    01_prepare_yolo_onnx.py
    02_eval_fp32_onnx.py
    03_eval_naive_int8_onnx.py
    04_aimet_quantsim_ptq.py
    05_aimet_cle_ptq.py
    06_aimet_adaround_ptq.py
    07_aimet_autoquant.py
    08_benchmark_latency.py
  src/
    aimet_yolo_study/
```

## Quick Start

Windows PowerShell이 아니라 WSL2 Ubuntu에서 실행합니다.

```bash
cd /path/to/AIMET/aimet-yolo-ptq-study
bash docker/build_runtime_image.sh
bash docker/run_aimet_onnx_gpu.sh
```

컨테이너 내부에서 기본 검증을 실행합니다.

```bash
python -m pip install -r requirements.txt
python docker/verify_aimet.py
python scripts/00_check_env.py
```

기본 런타임 Docker 이미지는 다음 이름으로 빌드됩니다.

```text
aimet-yolo-onnx-gpu:2.2.0
```

이 이미지는 공식/prebuilt AIMET 개발 이미지를 기반으로 하며 다음을 설치합니다.

- AIMET ONNX 2.2.0 CUDA 11.8 wheel
- `requirements.txt`에 정의한 프로젝트 의존성
- `python3`를 가리키는 `python` symlink

필요하면 런타임 이미지를 바꿔 실행할 수 있습니다.

```bash
AIMET_IMAGE="<image>:<tag>" bash docker/run_aimet_onnx_gpu.sh
```

## 스터디 진행 순서

1. WSL2 GPU와 Docker 접근을 확인합니다.
2. AIMET ONNX GPU 컨테이너에 진입합니다.
3. COCO 2017 val 5천 장을 다운로드하거나 마운트합니다.
4. YOLO ONNX 모델을 export하거나 `models/` 아래에 둡니다.
5. FP32 ONNX 기준선 평가를 실행합니다.
6. no-AIMET naive ONNX INT8 평가를 실행합니다.
7. AIMET QuantSim PTQ를 실행합니다.
8. AIMET CLE, AdaRound, AutoQuant 실험을 순서대로 실행합니다.
9. model-only와 end-to-end 레이턴시를 측정합니다.
10. `reports/aimet_ptq_study.md`에 지표와 해석을 기록합니다.

## 실험 자산 준비

AIMET 컨테이너 안에서 재현 가능한 COCO validation 자산을 준비합니다.

```bash
python scripts/01_prepare_coco.py --download
```

YOLO26 nano ONNX 모델을 준비합니다.

```bash
python scripts/01_prepare_yolo_onnx.py --export
```

기본적으로 `yolo26n.pt`를 다음 경로의 ONNX 파일로 export합니다.

```text
models/yolo26n_pretrained.onnx
```

정확도 기준선과 AIMET PTQ 실험을 실행합니다.

```bash
python scripts/02_eval_fp32_onnx.py --device 0
python scripts/03_eval_naive_int8_onnx.py --device 0
python scripts/04_aimet_quantsim_ptq.py --device 0
python scripts/05_aimet_cle_ptq.py --device 0
python scripts/06_aimet_adaround_ptq.py --device 0
```

전체 COCO 평가 전에 빠른 검증을 먼저 돌릴 수 있습니다. `--eval-samples`를 주면 지정한 개수의 이미지만 재현 가능한 샘플로 평가하고, 결과는 `results/metrics_quick.csv`에 기록합니다.

```bash
python scripts/02_eval_fp32_onnx.py --device 0 --eval-samples 100
python scripts/03_eval_naive_int8_onnx.py --device 0 --calibration-samples 64 --eval-samples 100
python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100
python scripts/05_aimet_cle_ptq.py --device 0 --calibration-samples 64 --eval-samples 100
```

AdaRound는 기본 설정이 오래 걸립니다. 먼저 API와 export 경로만 확인할 때는 작은 smoke 설정을 사용합니다.

```bash
python scripts/06_aimet_adaround_ptq.py --device 0 --calibration-samples 64 --adaround-samples 8 --adaround-iterations 50 --eval-samples 100
```

내보낸 ONNX 모델의 레이턴시를 벤치마크합니다.

```bash
python scripts/08_benchmark_latency.py --experiment-id A --experiment-name fp32_onnx --device 0
python scripts/08_benchmark_latency.py --experiment-id B --experiment-name naive_onnx_int8 --model results/models/yolo26n_pretrained.naive_int8.onnx --device 0
```

## 참고 자료

- AIMET 설치 문서: https://quic.github.io/aimet-pages/releases/2.2.0/install/index.html
- AIMET Docker 설치 노트: https://github.com/quic/aimet/blob/develop/packaging/docker_install.md
- Ultralytics YOLO 모델: https://docs.ultralytics.com/models
