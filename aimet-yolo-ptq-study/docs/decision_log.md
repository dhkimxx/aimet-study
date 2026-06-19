# 의사결정 로그

## 2026-06-20

| 주제 | 결정 | 이유 |
| --- | --- | --- |
| 기본 런타임 | WSL2 Ubuntu native + `uv` + Python 3.10 venv | 현재 작업 환경이 이미 WSL2 Ubuntu 내부이므로 추가 가상화 계층 없이 직접 실험합니다. |
| AIMET 설치 | AIMET ONNX 2.2.0 GPU wheel을 `pyproject.toml` direct URL dependency로 관리 | AIMET 2.2.0 ONNX GPU package가 Python 3.10 wheel로 배포되어 있어 venv 재현성을 명시합니다. |

## 2026-06-14

| 주제 | 결정 | 이유 |
| --- | --- | --- |
| 최적화 우선순위 | 정확도 유지, 레이턴시 감소, 메모리 감소 순서 | mAP 손실을 크게 내면서 속도만 올리는 방향을 피합니다. |
| 배포 목표 | 먼저 PC GPU, 이후 Android phone | PC GPU를 개발 기준선으로 삼고, Android 제약은 export 선택에 반영합니다. |
| 모델 | 최신 Ultralytics YOLO pretrained ONNX 계열, 시작점은 YOLO26 | 현재 스터디는 프레임워크보다 ONNX 중심의 비교에 초점을 둡니다. |
| 데이터셋 | COCO 2017 val 5천 장 | 표준적이고 재현 가능한 detection benchmark입니다. |
| 정확도 지표 | COCO box mAP50-95 | COCO detection의 대표 지표이며 localization 품질에 민감합니다. |
| 레이턴시 지표 | model-only와 end-to-end 모두 측정 | 양자화된 모델 자체 속도와 전처리/후처리 비용을 분리해서 봅니다. |
| 입력 형태 | 1x3x640x640 | YOLO 비교에 흔히 쓰는 기준점입니다. |
| AIMET 범위 | PTQ만 수행 | 1단계는 학습 없이 AIMET 양자화 도구를 이해하는 데 집중합니다. |
| 산출물 형태 | 재현 가능한 실험 프로젝트 | 나중에 custom dataset을 정했을 때 그대로 재사용할 수 있습니다. |
| 런타임 | WSL2 Ubuntu + NVIDIA GPU | AIMET의 Linux/GPU 전제를 맞추면서 Windows 환경에서 실용적으로 실행합니다. |
