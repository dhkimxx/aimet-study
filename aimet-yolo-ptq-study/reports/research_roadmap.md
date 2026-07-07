# 최종 논문형 리포트 로드맵

목표는 AIMET ONNX PTQ를 YOLO detection 모델에 적용할 때 정확도, 양자화 범위, encoding 수준, activation/weight 민감도가 어떻게 달라지는지 재현 가능한 실험 리포트로 정리하는 것입니다. Target runtime 효율은 AIMET HW-independent 결론과 분리한 배포 후속 검증으로 둡니다.

## 판정 기준

최종 리포트는 다음 질문에 숫자로 답해야 합니다.

1. naive ONNX INT8이 왜 실패했는가?
2. AIMET QuantSim/CLE/AdaRound가 정확도를 얼마나 회복하는가?
3. A8W8 손실은 activation과 weight 중 어느 쪽이 지배적인가?
4. QDQ accuracy-eval 모델과 실제 배포 모델의 차이는 무엇인가?
5. AIMET `.encodings` 기준으로 어느 group과 bitwidth가 손실 원인에 가장 가까운가?

## 현재 결론

| 결론 | 근거 |
| --- | --- |
| naive INT8은 공정한 성공 기준선이 아니라 실패 기준선 | full COCO mAP50-95 0.0000, postprocess/output까지 양자화 |
| AIMET QDQ는 정확도를 유지하지만 배포 모델은 아님 | full COCO A8W8 calib64 0.3740, QuantSim calib1024 0.3787, CLE calib1024 0.3788 vs FP32 0.3971, Conv weight INT storage 0/102 |
| AdaRound full 설정의 회복폭은 작음 | sample500 A8W8 QuantSim 0.4012, AdaRound calib256 adar128 iter2000 0.4036, AdaRound calib256 adar256 iter5000 0.4026, FP32 0.4203 |
| activation이 weight보다 민감 | full COCO A16W8 0.3923, A8W16 0.3843, sample100 all-activation-float 0.5440 |
| AIMET encoding도 activation 병목을 지지 | A8W8/AdaRound는 QDQ-exported activation 295개가 모두 8비트, A16W8/A16W16은 같은 295개 activation을 16비트로 바꾸며 scale median을 약 0.036대에서 약 0.000187로 축소 |
| ORT CUDA QDQ는 현재 latency 이득 없음 | FP32 6.16ms, A8W8 QDQ 14.77ms, 16비트 QDQ 100ms+ |
| ORT QOperator Conv-only도 ORT CUDA 배포 후보가 아님 | QLinearConv 102개, Conv weight INT storage 102/102, size 2.757MB지만 sample500 0.3486, model-only 32.40ms |
| TensorRT EP는 배포 후속 검증 | ORT provider 목록에는 보이지만 `libnvinfer.so.10` 누락으로 TensorRT 로드 실패, CUDA fallback 기록은 스크립트가 차단. AIMET HW-independent 결론의 필수 공백은 아님 |
| 다음 최적화 대상은 YOLO head activation | head Conv output 24개 float 변형이 0.5174에서 0.5327로 회복 |
| Head 내부 우선 후보는 `cv3` branch group | sample500 `head_cv3_outputs` float 0.4105, `head_scale2_outputs` 0.4055, `head_final_outputs` 0.4024. `cv3` per-layer top3에서는 `cv3_s1_2_final` float만 0.4047(+0.0036)로 양수 회복을 유지했지만, encoding intervention은 `head_cv3_outputs_a16` 0.4064(+0.0052)가 단일 tensor A16보다 안정적 |

## 실험 큐

### P0: 리포트 결론 안정화

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 1024 --name aimet_quantsim_a8w8_calib1024_gpu --force
scripts/run_native.sh python scripts/05_aimet_cle_ptq.py --device 0 --batch 1 --calibration-samples 1024 --name aimet_cle_a8w8_calib1024_gpu --force
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu
```

상태: 완료. sample500 표, full COCO 핵심 표, A8W8 calib1024 calibration ablation, CLE calib1024 ablation, AdaRound 중간/full 설정은 `reports/quick_ptq_results.md`, `reports/aimet_ptq_study.md`, `reports/paper_report.md`에 반영했습니다. AdaRound full COCO 평가는 선택적 추가 검증으로 남깁니다.

### P0: Head activation 원인 분석

현재 `head_conv_outputs` 24개 QDQ 제거 외에 branch/scale/final output 단위 결과가 추가되었습니다. sample500에서는 `cv3`, `scale2`, final output 후보를 재확인했고, `cv3` branch가 세 후보 중 가장 큰 회복을 보였습니다. 이어서 `cv3` 내부 15개 activation을 하나씩 제거해 sample100 전체 탐색과 sample500 top3 재확인을 완료했고, selected activation encoding intervention까지 완료했습니다.

완료 기준:

| 항목 | 기준 |
| --- | --- |
| layer group selector | `head_cv2_outputs`, `head_cv3_outputs`, `head_scale0/1/2_outputs`, `head_final_outputs` 구현 완료 |
| 결과 표 | sample100 group별 제거 QDQ 수, mAP50-95, Conv output QDQ coverage 기록 완료 |
| 확대 평가 | sample500 `cv3`, `scale2`, final output 결과 기록 완료 |
| per-layer 평가 | sample100 `cv3` 15개 전체와 sample500 top3 재확인 완료. `cv3_s1_2_final`만 sample500 양수 회복 유지 |
| encoding intervention | sample500 `head_cv3_outputs_a16` 0.4064(+0.0052), `cv3_s1_2_final_symmetric_i8` 0.4030(+0.0019), `cv3_s1_2_final_a16` 0.4006(-0.0005) |
| 다음 결론 | `head_cv3_outputs` group 중심으로 mixed precision, symmetric/asymmetric, per-channel 후보를 확정 |

### P0: AIMET encoding 수준 분석

완료 상태: `scripts/17_analyze_encodings.py`로 주요 AIMET `.encodings` sidecar를 분석했습니다. 결과는 `reports/encoding_analysis.md`에 요약하고, 재생성 가능한 원천 CSV/JSON은 `results/encoding_analysis.*`에 생성됩니다.

핵심 결과:

| 항목 | 결과 |
| --- | --- |
| QDQ-exported activation | 모든 주요 AIMET 산출물에서 295개 |
| sidecar-only activation | 모든 주요 AIMET 산출물에서 55개 |
| A8W8/AdaRound activation bitwidth | 8:295 |
| A16W8/A16W16 activation bitwidth | 16:295 |
| A8W8 QDQ scale median | 약 0.0367 |
| A16W8 QDQ scale median | 약 0.000187 |
| Parameter bitwidth | A8W8/AdaRound/A16W8은 8:102, A8W16/A16W16은 16:102 |

판정: AdaRound는 weight rounding을 바꾸지만 activation encoding 수와 bitwidth는 A8W8과 같습니다. A16W8이 A8W16보다 정확도를 더 회복한 이유는 activation quantization step 축소와 일관됩니다.

### P1: AdaRound 정식 비교

Smoke 설정은 결론용으로 부족합니다. 중간 설정과 full 설정을 sample500으로 완료했고, A8W8 대비 개선폭은 작았습니다.

완료한 중간 설정:

```bash
scripts/run_native.sh python scripts/06_aimet_adaround_ptq.py --device 0 --batch 1 --calibration-samples 256 --adaround-samples 128 --adaround-iterations 2000 --eval-samples 500 --name aimet_adaround_a8w8_adar128_iter2000_gpu --force
```

결과: sample500 mAP50-95 0.4036, A8W8 QuantSim 대비 +0.0025, FP32 대비 -0.0167. Coverage는 Q/DQ 397/397, Conv weight QDQ 102/102, Conv weight INT storage 0/102로 기존 AIMET A8W8 QDQ와 같습니다.

완료한 full 설정:

```bash
scripts/13_run_adaround_full_detached.sh
scripts/15_finalize_adaround_full.sh
```

결과: sample500 mAP50-95 0.4026, A8W8 QuantSim 대비 +0.0014, 중간 AdaRound 대비 -0.0011, FP32 대비 -0.0177. Coverage는 Q/DQ 397/397, Conv weight QDQ 102/102, Conv weight INT storage 0/102, effective INT storage 0.0%, size 9.786 MB입니다. 2026-06-28 foreground run은 `121/406` 모듈, 약 30% 지점에서 산출물 없이 끊겼지만, detached tmux runner로 같은 full 설정을 완료했습니다.

판정: AdaRound 설정 규모를 키워도 A8W8 손실은 거의 회복되지 않았습니다. weight rounding 단독보다는 activation QDQ/range, 특히 YOLO head activation 쪽을 다음 최적화 대상으로 봅니다. 남은 선택 작업은 같은 full 설정 산출물을 full COCO val로 평가해 sample500 결론의 일반성을 확인하는 것입니다.

### P1: Runtime/배포 경로 분리

현재 QDQ ONNX는 ORT CUDA에서 FP32보다 느립니다. 따라서 논문형 결론은 다음 두 층으로 분리합니다.

| 층 | 목적 | 산출물 |
| --- | --- | --- |
| Accuracy/encoding analysis | AIMET PTQ가 어디서 손실을 내는지 분석 | QDQ ONNX, `.encodings`, coverage, sensitivity |
| Deployment analysis | 실제 속도/메모리 이득 검증 | ORT QOperator probe 완료, TensorRT EP preflight 완료, TensorRT/QNN/target EP 측정 후보 |

완료한 ORT QOperator probe:

```bash
scripts/run_native.sh python scripts/12_eval_ort_qoperator_int8.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --name ort_qoperator_conv_int8
scripts/run_native.sh python scripts/08_benchmark_latency.py --experiment-id G --experiment-name ort_qoperator_conv_int8_latency --model results/models/yolo26n_pretrained.ort_qoperator_int8_conv_calib64.onnx --device 0 --warmup-runs 20 --measured-runs 100
```

결과: QLinearConv 102개, Conv weight INT storage 102/102, 모델 크기 2.757MB로 packed storage는 확인했습니다. 그러나 sample500 mAP50-95는 0.3486이고 model-only latency는 32.40ms라 FP32 및 AIMET A8W8 QDQ보다 불리했습니다.

TensorRT EP preflight:

```bash
scripts/run_native.sh python scripts/08_benchmark_latency.py --experiment-id T --experiment-name fp32_onnx_tensorrt --provider tensorrt --device 0 --warmup-runs 20 --measured-runs 100
```

결과: `TensorrtExecutionProvider`는 ORT provider 목록에 있지만, 현재 WSL2 환경에서 `libnvinfer.so.10`이 없어 실제 provider 로드가 실패했습니다. latency 스크립트는 요청 provider가 활성화되지 않으면 실패하므로 CUDA fallback 값을 TensorRT 결과로 기록하지 않습니다. 남은 작업은 target runtime을 정한 뒤 해당 runtime이 포함된 환경에서 FP32, A8W8 QDQ, QOperator 또는 target-friendly export 후보를 다시 측정하는 것입니다.

## 최종 산출물 구조

| 파일 | 역할 |
| --- | --- |
| `reports/paper_report.md` | 논문형 본문 |
| `reports/quick_ptq_results.md` | 빠른 검증과 실험 로그 |
| `reports/aimet_ptq_study.md` | 전체 스터디 리포트 |
| `reports/encoding_analysis.md` | AIMET `.encodings` group/bitwidth/scale 분석 |
| `reports/activation_encoding_interventions.md` | selected activation encoding intervention sample100 screening |
| `reports/activation_encoding_interventions_sample500.md` | selected activation encoding intervention sample500 재확인 |
| `reports/head_cv3_layer_sensitivity.md` | `cv3` per-layer sample100 전체 민감도 |
| `reports/head_cv3_layer_sensitivity_sample500.md` | `cv3` per-layer top3 sample500 재확인 |
| `reports/research_roadmap.md` | 완료된 실험과 후속 검증 큐 |
| `results/metrics_quick.csv` | quick accuracy 원천 데이터 |
| `results/quantization_coverage*.csv` | coverage 원천 데이터 |
| `results/latency.csv` | latency 원천 데이터 |
| `scripts/11_generate_report_figures.py` | report figure 재생성 |
| `scripts/17_analyze_encodings.py` | AIMET `.encodings` 분석 재생성 |
| `scripts/18_head_cv3_layer_sensitivity.py` | `head_cv3_outputs` 내부 per-layer 민감도 재생성 |
| `scripts/19_activation_encoding_intervention.py` | selected activation QDQ scale/zero-point intervention 재생성 |
| `reports/figures/*.svg` | 논문형 리포트 figure 산출물 |

## 최종 원고 체크리스트

- [x] sample500 이상 정확도 표
- [x] full COCO 또는 full에 준하는 대표 subset 결과
- [x] latency 표와 측정 조건 고정
- [x] QDQ coverage 표
- [x] AIMET `.encodings` group 분석
- [x] head sensitivity 확대 평가
- [x] activation sensitivity figure
- [x] deployment artifact 한계 명시
- [x] 재현 명령과 환경 해시 정리
- [x] TensorRT fallback 오기록 방지
