# AIMET YOLO PTQ 스터디 리포트

## 요약

YOLO26 ONNX 모델을 대상으로 FP32 기준선, no-AIMET naive INT8, AIMET QuantSim/CLE/AdaRound/AutoQuant 흐름을 비교합니다. 목표는 정확도를 최대한 유지하면서 레이턴시와 메모리 사용량을 줄이는 것입니다.

## 실험 설정

| 항목 | 값 |
| --- | --- |
| 모델 | YOLO26 pretrained ONNX |
| 데이터셋 | COCO 2017 val 5천 장 |
| 입력 | 1x3x640x640 |
| 1차 지표 | box mAP50-95 |
| 환경 | WSL2 Ubuntu, NVIDIA GPU, AIMET ONNX Docker |

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

아래 결과는 COCO 전체가 아니라 `--eval-samples 20`으로 뽑은 20장 샘플 기준입니다. 절대 성능 판단용이 아니라 파이프라인 검증과 큰 방향성 확인용입니다.

상세 실행 조건과 해석은 `reports/quick_ptq_results.md`에 별도로 정리했습니다.

| ID | 실험 | 설정 | mAP50-95 | mAP50 | mAP75 | 비고 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | FP32 ONNX | sample20 | 0.5575 | 0.6778 | 0.6093 | 샘플 기준선 |
| B | no-AIMET naive ONNX INT8 | calib64, sample20 | 0.0000 | 0.0000 | 0.0000 | 단순 INT8 양자화 실패 재현 |
| C | AIMET QuantSim PTQ | calib64, sample20 | 0.5575 | 0.6778 | 0.6093 | FP32와 거의 동일 |
| D | AIMET CLE + QuantSim | calib64, sample20 | 0.5575 | 0.6778 | 0.6093 | 현재 모델에서는 QuantSim과 차이 거의 없음 |
| E | AIMET AdaRound + QuantSim | calib64, adar8, iter50, sample20 | 0.5588 | 0.6783 | 0.6116 | smoke 설정이므로 정식 AdaRound 결과는 아님 |

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
- AIMET QuantSim은 20장 빠른 검증에서 FP32 수준의 정확도를 유지했습니다.
- CLE는 이 샘플과 현재 모델에서는 QuantSim 단독과 거의 같은 결과를 냈습니다.
- AdaRound는 작은 smoke 설정으로 API와 export 경로를 확인했습니다. 정식 비교는 기본 또는 충분한 iteration으로 다시 평가해야 합니다.
