# AIMET YOLO PTQ 스터디 리포트

## 요약

YOLO26 ONNX 모델을 대상으로 FP32 기준선, no-AIMET naive INT8, AIMET QuantSim/CLE/AdaRound/AutoQuant 흐름을 비교합니다. 목표는 정확도를 최대한 유지하면서 레이턴시와 메모리 사용량을 줄이는 것입니다.

정정 사항: AIMET ONNX 2.2.0 public export가 만든 `.encodings` sidecar는 ONNX Runtime 평가에 자동 적용되지 않습니다. 따라서 QDQ 노드가 없는 AIMET ONNX를 평가한 C/D/E 빠른 검증 값은 INT8 양자화 결과가 아니라 FP32 ONNX 결과였습니다. 현재 리포트의 빠른 검증 표는 `QuantizeLinear`/`DequantizeLinear` 노드가 포함된 ONNX를 다시 생성해 평가한 값입니다.

## 실험 설정

| 항목 | 값 |
| --- | --- |
| 모델 | YOLO26 pretrained ONNX |
| 데이터셋 | COCO 2017 val 5천 장 |
| 입력 | 1x3x640x640 |
| 1차 지표 | box mAP50-95 |
| 환경 | WSL2 Ubuntu native, NVIDIA GPU, uv Python 3.10 venv |

## 정확도 결과

| ID | 실험 | mAP50-95 | mAP50 | mAP75 | AP small | AP medium | AP large | mAP50-95 변화 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | FP32 ONNX | 0.3971 | 0.5512 | 0.4311 | 예정 | 예정 | 예정 | 0.0000 |
| B | no-AIMET naive ONNX INT8 | 0.0000 | 0.0000 | 0.0000 | 예정 | 예정 | 예정 | -0.3971 |
| C | AIMET QuantSim PTQ | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 |
| D | AIMET CLE + QuantSim | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 |
| E | AIMET AdaRound + QuantSim | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 |
| F | AIMET AutoQuant | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 | 예정 |

## 빠른 검증 결과

아래 결과는 COCO 전체가 아니라 `--eval-samples 100`으로 뽑은 100장 샘플 기준입니다. 절대 성능 판단용이 아니라 파이프라인 검증과 큰 방향성 확인용입니다.

상세 실행 조건과 해석은 `reports/quick_ptq_results.md`에 별도로 정리했습니다.

| ID | 실험 | 설정 | mAP50-95 | mAP50 | mAP75 | 비고 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | FP32 ONNX | sample100 | 0.5437 | 0.6657 | 0.5854 | 샘플 기준선 |
| B | no-AIMET naive ONNX INT8 | calib64, sample100 | 0.0000 | 0.0000 | 0.0000 | 단순 INT8 양자화 실패 재현 |
| C | AIMET QuantSim PTQ | calib64, sample100, QDQ | 0.5174 | 0.6524 | 0.5516 | 실제 QDQ ONNX 기준, FP32 대비 -0.0263 |
| D | AIMET CLE + QuantSim | calib64, sample100, QDQ | 0.5131 | 0.6417 | 0.5502 | 실제 QDQ ONNX 기준, FP32 대비 -0.0307 |
| E | AIMET AdaRound + QuantSim | calib64, adar8, iter50, sample100, QDQ | 0.5307 | 0.6552 | 0.5711 | 실제 QDQ ONNX 기준, FP32 대비 -0.0130 |

## 빠른 검증 산출물 확인

| ID | 실험 | QuantizeLinear | DequantizeLinear | 모델 SHA256 |
| --- | --- | ---: | ---: | --- |
| A | FP32 ONNX | 0 | 0 | `5cb19c918a1ff7a4178ab7b1f0fc878a7b67e8baf012af443c6b06372946a4f2` |
| B | no-AIMET naive ONNX INT8 | 389 | 593 | `29d4343971e6dc8609631ba302101634ce96e38e023accb4077402333a056d34` |
| C | AIMET QuantSim PTQ | 397 | 397 | `37ddc7829f1d23504762626bb9adae5511607d2a1a7218ea5d0f3d0a58180a95` |
| D | AIMET CLE + QuantSim | 397 | 397 | `00c59e859fe38d2554299d54b1c4ceb670306b8bbb84387540d0fb0ff00b2136` |
| E | AIMET AdaRound + QuantSim | 397 | 397 | `00757f981c03df09f94f1fa7239211f71b6a0ce60627ef17218055ca187a380c` |

## 레이턴시 결과

| ID | 실험 | model-only 평균 ms | model-only p95 ms | end-to-end 평균 ms | end-to-end p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| A | FP32 ONNX | 예정 | 예정 | 예정 | 예정 |
| B | no-AIMET naive ONNX INT8 | 예정 | 예정 | 예정 | 예정 |
| C | AIMET QuantSim PTQ | 예정 | 예정 | 예정 | 예정 |
| D | AIMET CLE + QuantSim | 예정 | 예정 | 예정 | 예정 |
| E | AIMET AdaRound + QuantSim | 예정 | 예정 | 예정 | 예정 |
| F | AIMET AutoQuant | 예정 | 예정 | 예정 | 예정 |

## 메모리 결과

| ID | 실험 | peak GPU MB | peak host MB |
| --- | --- | ---: | ---: |
| A | FP32 ONNX | 예정 | 예정 |
| B | no-AIMET naive ONNX INT8 | 예정 | 예정 |
| C | AIMET QuantSim PTQ | 예정 | 예정 |
| D | AIMET CLE + QuantSim | 예정 | 예정 |
| E | AIMET AdaRound + QuantSim | 예정 | 예정 |
| F | AIMET AutoQuant | 예정 | 예정 |

## 해석 메모

다음 관점으로 결과를 해석합니다.

- 어떤 AIMET 단계가 정확도를 가장 많이 회복했는지
- no-AIMET INT8 기준선이 이 모델에 대해 공정한 비교 대상인지
- AP small이 AP medium/large보다 더 민감하게 떨어지는지
- model-only 레이턴시 개선이 end-to-end 레이턴시 개선으로 이어지는지
- export 산출물이 이후 Android 실험으로 이어질 수 있는지

## 현재 메모

- FP32 ONNX는 COCO 기준 정상 mAP를 확인했습니다.
- no-AIMET naive INT8은 현재 기준으로 mAP가 0으로 떨어져, 단순 양자화가 YOLO 후처리와 잘 맞지 않는 가능성이 큽니다.
- AIMET QuantSim/CLE/AdaRound는 QDQ ONNX로 다시 내보내 평가해야 합니다. `.encodings`만 있는 AIMET export를 ORT로 평가하면 INT8 결과가 아닙니다.
- 실제 QDQ 기준 100장 빠른 검증에서는 AdaRound smoke가 C/D보다 가장 높은 mAP50-95를 보였습니다.
- 현재 QDQ export는 YOLO detection postprocess 영역의 비-Conv 텐서와 최종 `output0` QDQ를 제외합니다. postprocess까지 양자화한 첫 시도는 sample20 기준 mAP가 0으로 떨어졌습니다.
- AdaRound는 작은 smoke 설정으로 API와 export 경로를 확인했습니다. 정식 비교는 기본 또는 충분한 iteration으로 다시 평가해야 합니다.
