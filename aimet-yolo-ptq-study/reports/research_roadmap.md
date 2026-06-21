# 최종 논문형 리포트 로드맵

목표는 AIMET ONNX PTQ를 YOLO detection 모델에 적용할 때 정확도, 양자화 범위, runtime 효율이 어떻게 달라지는지 재현 가능한 실험 리포트로 정리하는 것입니다.

## 판정 기준

최종 리포트는 다음 질문에 숫자로 답해야 합니다.

1. naive ONNX INT8이 왜 실패했는가?
2. AIMET QuantSim/CLE/AdaRound가 정확도를 얼마나 회복하는가?
3. A8W8 손실은 activation과 weight 중 어느 쪽이 지배적인가?
4. QDQ accuracy-eval 모델과 실제 배포 모델의 차이는 무엇인가?
5. 정확도-레이턴시 관점에서 현재 추천 가능한 경로는 무엇인가?

## 현재 결론

| 결론 | 근거 |
| --- | --- |
| naive INT8은 공정한 성공 기준선이 아니라 실패 기준선 | full COCO mAP50-95 0.0000, postprocess/output까지 양자화 |
| AIMET QDQ는 정확도를 유지하지만 배포 모델은 아님 | full COCO A8W8 0.3740 vs FP32 0.3971, Conv weight INT storage 0/102 |
| activation이 weight보다 민감 | full COCO A16W8 0.3923, A8W16 0.3843, sample100 all-activation-float 0.5440 |
| ORT CUDA QDQ는 현재 latency 이득 없음 | FP32 6.16ms, A8W8 QDQ 14.77ms, 16비트 QDQ 100ms+ |
| 다음 최적화 대상은 YOLO head activation | head Conv output 24개 float 변형이 0.5174에서 0.5327로 회복 |
| Head 내부 우선 후보는 `cv3` branch | sample500 `head_cv3_outputs` 0.4105, `head_scale2_outputs` 0.4055, `head_final_outputs` 0.4024 |

## 실험 큐

### P0: 리포트 결론 안정화

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu
```

상태: 완료. sample500 표와 full COCO 핵심 표는 `reports/quick_ptq_results.md`, `reports/aimet_ptq_study.md`, `reports/paper_report.md`에 반영했습니다. AdaRound 정식 설정은 별도 P1 작업으로 남깁니다.

### P0: Head activation 원인 분석

현재 `head_conv_outputs` 24개 QDQ 제거 외에 branch/scale/final output 단위 결과가 추가되었습니다. sample500에서는 `cv3`, `scale2`, final output 후보를 재확인했고, `cv3` branch가 세 후보 중 가장 큰 회복을 보였습니다.

완료 기준:

| 항목 | 기준 |
| --- | --- |
| layer group selector | `head_cv2_outputs`, `head_cv3_outputs`, `head_scale0/1/2_outputs`, `head_final_outputs` 구현 완료 |
| 결과 표 | sample100 group별 제거 QDQ 수, mAP50-95, Conv output QDQ coverage 기록 완료 |
| 확대 평가 | sample500 `cv3`, `scale2`, final output 결과 기록 완료 |
| 다음 결론 | `cv3` branch 중심으로 per-layer range, percentile, symmetric/asymmetric 후보를 확정 |

### P1: AdaRound 정식 비교

Smoke 설정은 결론용으로 부족합니다.

```bash
scripts/run_native.sh python scripts/06_aimet_adaround_ptq.py --device 0 --batch 1 --calibration-samples 256 --adaround-samples 256 --adaround-iterations 5000 --eval-samples 500 --force
```

이 설정이 너무 오래 걸리면 `adaround-samples 128`, `iterations 2000`을 중간 지점으로 둡니다.

### P1: Runtime/배포 경로 분리

현재 QDQ ONNX는 ORT CUDA에서 FP32보다 느립니다. 따라서 논문형 결론은 다음 두 층으로 분리합니다.

| 층 | 목적 | 산출물 |
| --- | --- | --- |
| Accuracy analysis | AIMET PTQ가 어디서 손실을 내는지 분석 | QDQ ONNX, coverage, sensitivity |
| Deployment analysis | 실제 속도/메모리 이득 검증 | TensorRT/QNN/ORT quantized-op export 후보 |

## 최종 산출물 구조

| 파일 | 역할 |
| --- | --- |
| `reports/paper_report.md` | 논문형 본문 초안 |
| `reports/quick_ptq_results.md` | 빠른 검증과 실험 로그 |
| `reports/aimet_ptq_study.md` | 전체 스터디 리포트 |
| `reports/research_roadmap.md` | 남은 실험 큐와 완료 기준 |
| `results/metrics_quick.csv` | quick accuracy 원천 데이터 |
| `results/quantization_coverage*.csv` | coverage 원천 데이터 |
| `results/latency.csv` | latency 원천 데이터 |
| `scripts/11_generate_report_figures.py` | report figure 재생성 |
| `reports/figures/*.svg` | 논문형 리포트 figure 산출물 |

## 최종 원고 체크리스트

- [x] sample500 이상 정확도 표
- [x] full COCO 또는 full에 준하는 대표 subset 결과
- [x] latency 표와 측정 조건 고정
- [x] QDQ coverage 표
- [x] head sensitivity 확대 평가
- [x] activation sensitivity figure
- [x] deployment artifact 한계 명시
- [x] 재현 명령과 환경 해시 정리
