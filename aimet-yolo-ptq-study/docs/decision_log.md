# 의사결정 로그

## 2026-06-28

| 주제 | 결정 | 이유 |
| --- | --- | --- |
| CLE calibration 확대 | CLE + QuantSim A8W8을 calibration 1024장, full COCO val로 재평가 | QuantSim A8W8 calib1024가 0.3787 mAP50-95였고, CLE calib1024는 0.3788로 사실상 동일했습니다. QDQ/storage coverage도 Q/DQ 397/397, Conv weight QDQ 102/102, Conv weight INT storage 0/102로 같습니다. 이 YOLO ONNX에는 BatchNorm이 없어 AIMET 로그가 high-bias folding 미지원을 명시했으므로, CLE 단독으로 A8W8 손실을 회복하기 어렵다고 판단합니다. |
| AdaRound 중간 설정 | `calib256`, `adaround-samples 128`, `iterations 2000`, `sample500`으로 A8W8 AdaRound를 재평가 | mAP50-95는 0.4036으로 A8W8 QuantSim 0.4012보다 +0.0025 높았습니다. Coverage는 Q/DQ 397/397, Conv weight QDQ 102/102, Conv weight INT storage 0/102로 기존 A8W8 QDQ와 같습니다. 개선은 있으나 A8W16/A16W8/A16W16보다 작아, 현재 손실의 주원인은 weight rounding 단독이 아니라 activation QDQ 쪽이라고 판단합니다. |

## 2026-06-21

| 주제 | 결정 | 이유 |
| --- | --- | --- |
| 16비트 조합 | AIMET QuantSim/CLE/AdaRound 스크립트에 activation/weight bitwidth override를 둠 | A8W8만 보면 activation quantization error와 weight quantization error를 분리해서 설명하기 어렵습니다. A16W8, A8W16, A16W16을 같은 calibration/eval 조건으로 비교합니다. |
| 16비트 QDQ opset | int16/uint16 QDQ export는 ONNX opset 21로 변환 | ONNX opset 17의 `QuantizeLinear`는 int8/uint8만 허용합니다. int16/uint16 QDQ는 opset 21이 필요하며, 기존 YOLO 그래프는 단순 버전 숫자 변경이 아니라 ONNX version converter를 거쳐야 checker를 통과합니다. |
| 16비트 결과 해석 | 16비트 결과도 accuracy와 coverage를 같이 기록 | sample100 CUDA 기준 A16W8이 가장 높았지만, AIMET QDQ 모델은 weight storage가 FP32라 deployment artifact가 아닙니다. 정확도 경향과 배포 효율을 분리해서 봅니다. |
| Activation 민감도 | A8W8 QDQ에서 선택한 activation QDQ만 제거하는 ablation 스크립트를 추가 | A16W8이 A8W16보다 크게 회복한 원인을 확인하기 위해 weight QDQ는 유지하고 head, neck, Conv output, 전체 activation 범위별 정확도 회복을 비교합니다. |
| Head activation 세분화 | YOLO head Conv output을 branch, scale, final output 단위로 나눠 sensitivity를 기록 | head 전체 24개 QDQ 제거가 큰 회복을 보였으므로, sample500 이상에서 재확인할 후보를 `cv3`, `scale2`, final output 중심으로 좁힙니다. |
| sample500 확대 검증 | FP32, naive INT8, A8W8, 16비트 조합, head 후보를 500장으로 재평가 | naive INT8 붕괴는 유지됐고, A8W8은 FP32 대비 -0.0191 mAP50-95였습니다. 단일 축 비교에서는 A16W8이 A8W16보다 높고, head 후보 중 `cv3`가 가장 큰 회복을 보였습니다. |
| full COCO 핵심 검증 | FP32, naive INT8, A8W8, A16W8, A8W16, A16W16을 full COCO val로 재평가 | naive INT8은 full에서도 0.0000이고, A8W8은 FP32 대비 -0.0231입니다. A16W8 0.3923이 A8W16 0.3843보다 높아 activation 민감도 결론이 full에서도 유지됩니다. |
| A8W8 calibration 확대 | QuantSim A8W8을 calibration 1024장으로 full COCO 재평가 | A8W8 mAP50-95는 0.3740에서 0.3787로 +0.0047만 회복했습니다. QDQ/storage coverage는 동일하므로 calibration 부족은 일부 요인이지만 주원인으로 보기는 어렵습니다. |

## 2026-06-20

| 주제 | 결정 | 이유 |
| --- | --- | --- |
| 기본 런타임 | WSL2 Ubuntu native + `uv` + Python 3.10 venv | 현재 작업 환경이 이미 WSL2 Ubuntu 내부이므로 추가 가상화 계층 없이 직접 실험합니다. |
| AIMET 설치 | AIMET ONNX 2.2.0 GPU wheel을 `pyproject.toml` direct URL dependency로 관리 | AIMET 2.2.0 ONNX GPU package가 Python 3.10 wheel로 배포되어 있어 venv 재현성을 명시합니다. |
| AIMET 평가 산출물 | ONNX Runtime으로 평가할 AIMET 모델은 표준 QDQ ONNX여야 함 | AIMET ONNX 2.2.0 public export는 ONNX에서 quantization 노드를 제거하고 `.encodings`를 별도로 저장하므로, QDQ 없는 ONNX는 ORT/Ultralytics에서 INT8로 실행되지 않습니다. |
| YOLO postprocess 양자화 | detection postprocess의 비-Conv 텐서와 최종 `output0`은 QDQ 변환에서 제외 | postprocess까지 QDQ를 넣은 첫 시도는 sample20 기준 mAP가 0으로 떨어졌고, postprocess 제외 후 실제 QDQ 모델이 정상 mAP를 냈습니다. |
| 비교 리포트 기준 | 정확도와 함께 양자화 커버리지를 같이 기록 | input/output, Conv input/weight/output, weight storage, postprocess 양자화 범위가 다르면 같은 INT8 결과로 비교하기 어렵습니다. |

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
