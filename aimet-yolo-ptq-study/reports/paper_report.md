# AIMET ONNX PTQ for YOLO: Quantization Coverage, Accuracy, and Runtime Study

작성일: 2026-06-21

상태: 논문형 리포트 초안. 현재 수치는 `sample100` 탐색, `sample500` 확대 검증, full COCO val 5천 장 핵심 결과를 포함합니다. 남은 주요 공백은 AdaRound 정식 설정, 배포형 export, figure 정리입니다.

## 초록

이 연구는 YOLO26n ONNX detection 모델을 대상으로 AIMET ONNX 2.2.0 기반 post-training quantization(PTQ)을 WSL2 Ubuntu native GPU 환경에서 재현 가능하게 비교한다. 단순 ONNX Runtime INT8 양자화, AIMET QuantSim, CLE, AdaRound, 8/16비트 조합, activation QDQ ablation을 같은 평가 파이프라인으로 비교했다. 핵심 발견은 세 가지다. 첫째, naive ONNX INT8은 postprocess와 graph output까지 공격적으로 양자화하면서 sample100, sample500, full COCO val 모두에서 mAP가 0으로 붕괴했다. 둘째, AIMET A8W8 QDQ는 full COCO에서 FP32 0.3971 대비 0.3740 mAP50-95로 정확도를 유지하지만, 현재 산출물은 Conv weight storage가 FP32인 accuracy-eval QDQ 모델이므로 배포 효율과 분리해서 해석해야 한다. 셋째, full COCO 16비트 조합과 activation-QDQ 제거 ablation은 A8W8 손실의 주된 원인이 weight보다 activation quantization error임을 보여준다.

## 연구 질문

| ID | 질문 | 현재 답 |
| --- | --- | --- |
| RQ1 | no-AIMET naive INT8은 YOLO ONNX에 유효한 기준선인가? | 정확도 기준으로는 실패 기준선입니다. full COCO val에서도 mAP50-95가 0으로 붕괴했습니다. |
| RQ2 | AIMET QDQ PTQ는 naive INT8 대비 정확도를 유지하는가? | 예. A8W8 QuantSim은 full COCO에서 mAP50-95 0.3740을 유지했습니다. |
| RQ3 | A8W8 손실은 weight와 activation 중 어디에 더 민감한가? | activation이 더 민감합니다. full COCO에서 A16W8이 A8W16보다 높고, sample100 all-activation-float ablation이 FP32 수준까지 회복했습니다. |
| RQ4 | 현재 QDQ ONNX가 배포 latency 이득을 주는가? | 아니오. ORT CUDA에서는 FP32보다 느립니다. 현재 모델은 정확도 분석용 QDQ입니다. |

## 환경

| 항목 | 값 |
| --- | --- |
| OS/runtime | WSL2 Ubuntu native |
| GPU | NVIDIA GeForce RTX 3070 |
| Python/env | uv Python 3.10 venv |
| Quantization toolkit | AIMET ONNX 2.2.0 GPU |
| Inference runtime | ONNX Runtime CUDAExecutionProvider |
| Model | Ultralytics YOLO26n pretrained ONNX |
| Dataset | COCO 2017 val 5천 장, quick result는 100 images, expanded result는 500 images |
| Input | batch 1, 640x640 |
| Primary metric | COCO box mAP50-95 |

## 방법

실험은 FP32 ONNX 기준선, ONNX Runtime static quantization 기반 naive INT8, AIMET QuantSim A8W8, CLE+QuantSim, AdaRound+QuantSim, QuantSim 16비트 조합, activation QDQ ablation으로 구성했다. 모든 AIMET accuracy 평가는 ONNX에 표준 `QuantizeLinear`/`DequantizeLinear`가 포함된 QDQ 모델로 수행했다. AIMET `.encodings` sidecar만 있는 ONNX는 ONNX Runtime/Ultralytics가 자동 적용하지 않으므로 INT8 평가로 보지 않았다.

양자화 커버리지는 Q/DQ 노드 수, graph input/output QDQ, Conv input/weight/output QDQ, Conv weight INT storage, YOLO head/postprocess QDQ 여부를 별도 스크립트로 측정했다. Latency는 동일한 CUDAExecutionProvider에서 warmup 20회, measured 100회로 model-only와 lightweight end-to-end를 분리해 측정했다.

## 정확도 결과

### sample100 탐색 결과

| 실험 | 설정 | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 ONNX | sample100 | 0.5437 | 0.6657 | 0.5854 | 0.6454 | 0.6251 |
| naive ONNX INT8 | calib64, sample100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| AIMET QuantSim | A8W8, calib64, sample100 | 0.5174 | 0.6524 | 0.5516 | 0.7065 | 0.5739 |
| AIMET CLE | A8W8, calib64, sample100 | 0.5131 | 0.6417 | 0.5502 | 0.7308 | 0.5560 |
| AIMET AdaRound | A8W8, adar8, iter50, sample100 | 0.5307 | 0.6552 | 0.5711 | 0.6644 | 0.5700 |

AdaRound는 smoke 설정이므로 정식 비교가 아니다. 현재 결과에서는 API와 export 경로가 정상임을 확인한 수준이다.

### sample500 확대 검증

| 실험 | 설정 | mAP50-95 | mAP50 | mAP75 | Precision | Recall | FP32 대비 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 ONNX | sample500 | 0.4203 | 0.5689 | 0.4551 | 0.6820 | 0.5215 | 0.0000 |
| naive ONNX INT8 | calib64, sample500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.4203 |
| AIMET QuantSim | A8W8, calib64, sample500 | 0.4012 | 0.5518 | 0.4337 | 0.6355 | 0.5111 | -0.0191 |
| AIMET QuantSim | A16W8, calib64, sample500 | 0.4143 | 0.5666 | 0.4496 | 0.6648 | 0.5158 | -0.0060 |
| AIMET QuantSim | A8W16, calib64, sample500 | 0.4074 | 0.5580 | 0.4465 | 0.6467 | 0.5093 | -0.0130 |
| AIMET QuantSim | A16W16, calib64, sample500 | 0.4170 | 0.5663 | 0.4495 | 0.6543 | 0.5142 | -0.0033 |

sample500에서는 A8W8 QDQ가 FP32 대비 -0.0191 mAP50-95에 머물렀고, 16비트 조합 중 A16W16이 FP32에 가장 가까웠다. 단일 축 비교에서는 A16W8이 A8W16보다 높아 activation quantization error가 더 큰 원인이라는 해석이 유지된다.

### full COCO val 결과

아래 결과는 COCO 2017 val 5천 장 전체를 같은 CUDAExecutionProvider 조건으로 평가한 값이다.

| 실험 | 설정 | mAP50-95 | mAP50 | mAP75 | Precision | Recall | FP32 대비 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 ONNX | full val | 0.3971 | 0.5512 | 0.4311 | 0.6689 | 0.4990 | 0.0000 |
| naive ONNX INT8 | calib64, full val | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.3971 |
| AIMET QuantSim | A8W8, calib64, full val | 0.3740 | 0.5321 | 0.4059 | 0.6450 | 0.4949 | -0.0231 |
| AIMET QuantSim | A16W8, calib64, full val | 0.3923 | 0.5490 | 0.4273 | 0.6606 | 0.5009 | -0.0049 |
| AIMET QuantSim | A8W16, calib64, full val | 0.3843 | 0.5392 | 0.4179 | 0.6543 | 0.4957 | -0.0128 |
| AIMET QuantSim | A16W16, calib64, full val | 0.3952 | 0.5489 | 0.4288 | 0.6583 | 0.5020 | -0.0019 |

full COCO에서도 sample500 결론이 유지된다. Naive INT8은 완전히 실패했고, A8W8 QDQ는 정확도를 상당 부분 유지한다. A16W8이 A8W16보다 +0.0080 mAP50-95 높아 activation bitwidth가 weight bitwidth보다 더 큰 회복을 만든다. A16W16은 FP32와 -0.0019 차이까지 접근하지만, 아래 latency 결과처럼 배포 후보가 아니라 accuracy diagnosis 상한으로 해석해야 한다.

## Bitwidth Ablation

| 실험 | Activation | Weight | Opset | sample100 mAP50-95 | sample500 mAP50-95 | full mAP50-95 | full FP32 대비 | Conv weight INT storage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QuantSim | 8 | 8 | 17 | 0.5174 | 0.4012 | 0.3740 | -0.0231 | 0/102 |
| QuantSim | 16 | 8 | 21 | 0.5449 | 0.4143 | 0.3923 | -0.0049 | 0/102 |
| QuantSim | 8 | 16 | 21 | 0.5347 | 0.4074 | 0.3843 | -0.0128 | 0/102 |
| QuantSim | 16 | 16 | 21 | 0.5374 | 0.4170 | 0.3952 | -0.0019 | 0/102 |

A16W8이 A8W16보다 더 크게 회복했다. 이는 activation quantization error가 weight quantization error보다 더 지배적일 가능성을 보인다. A16W16은 sample500/full 정확도만 보면 가장 좋지만, 네 모델 모두 Conv weight QDQ는 102/102이고 Conv weight INT storage는 0/102라 파일 내부 weight가 packed int8/int16으로 저장된 배포 산출물이 아니다.

## Activation Sensitivity

A8W8 QDQ 모델에서 선택한 activation QDQ만 제거하고 weight QDQ는 유지했다.

| 변형 | 제거한 activation QDQ | 남은 Q/DQ | mAP50-95 | 해석 |
| --- | ---: | ---: | ---: | --- |
| A8W8 기준 | 0 | 397/397 | 0.5174 | 전체 activation QDQ 유지 |
| head Conv outputs float | 24 | 373/373 | 0.5327 | YOLO head가 민감 |
| late neck 20-22 float | 33 | 364/364 | 0.5238 | 회복폭 작음 |
| all Conv outputs float | 102 | 295/295 | 0.5377 | Conv output activation 영향 큼 |
| all activations float | 295 | 102/102 | 0.5440 | weight QDQ만 남겨 FP32/A16W8에 근접 |

이 결과는 A16W8 결과와 일관된다. Activation QDQ를 제거하면 정확도가 거의 FP32 수준으로 회복되고, weight QDQ만 남아도 mAP50-95는 0.5440을 유지한다.

### Head Sensitivity

`head_conv_outputs` 24개를 branch, scale, final Conv output 기준으로 나눠 추가 ablation을 수행했다.

| 변형 | 제거한 activation QDQ | 남은 Q/DQ | mAP50-95 | 해석 |
| --- | ---: | ---: | ---: | --- |
| head cv2 outputs float | 9 | 388/388 | 0.5198 | 회복폭 작음 |
| head cv3 outputs float | 15 | 382/382 | 0.5252 | branch 기준 더 민감 |
| head scale0 outputs float | 8 | 389/389 | 0.5202 | scale별 최저 |
| head scale1 outputs float | 8 | 389/389 | 0.5235 | 중간 회복 |
| head scale2 outputs float | 8 | 389/389 | 0.5266 | scale별 최고 |
| head final outputs float | 6 | 391/391 | 0.5221 | 마지막 출력만으로 일부 회복 |

Head 내부에서는 `cv3` branch와 `scale2` 출력이 상대적으로 더 민감했다. 그러나 24개 head Conv output 전체를 float로 되돌린 0.5327에는 못 미치므로, 최종 출력만이 아니라 head 중간 activation의 누적 quantization error도 영향을 준다.

sample500에서는 세 후보만 재평가했다.

| 변형 | 제거한 activation QDQ | 남은 Q/DQ | sample500 mAP50-95 | A8W8 대비 |
| --- | ---: | ---: | ---: | ---: |
| A8W8 기준 | 0 | 397/397 | 0.4012 | 0.0000 |
| head cv3 outputs float | 15 | 382/382 | 0.4105 | +0.0094 |
| head scale2 outputs float | 8 | 389/389 | 0.4055 | +0.0043 |
| head final outputs float | 6 | 391/391 | 0.4024 | +0.0012 |

확대 검증에서는 `cv3` branch가 가장 일관된 회복을 보였고, final output 단독 개선은 작았다. 이는 head의 최종 출력만이 아니라 class branch 내부 activation quantization이 병목임을 시사한다.

## Latency Results

| 실험 | 설정 | model-only mean ms | model-only p95 ms | end-to-end mean ms | end-to-end p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| FP32 ONNX | opset17 | 6.16 | 7.27 | 13.56 | 23.80 |
| naive ONNX INT8 | int8 storage, postprocess 포함 | 15.42 | 17.16 | 21.41 | 23.34 |
| AIMET QuantSim | A8W8 QDQ | 14.77 | 15.81 | 21.25 | 23.51 |
| AIMET QuantSim | A16W8 QDQ | 109.59 | 127.29 | 123.95 | 154.13 |
| AIMET QuantSim | A8W16 QDQ | 101.40 | 121.84 | 119.97 | 151.82 |
| AIMET QuantSim | A16W16 QDQ | 110.23 | 132.79 | 119.68 | 147.78 |
| Sensitivity | A8W8, all activations float | 7.19 | 8.65 | 13.49 | 15.27 |

ORT CUDA 기준으로 현재 QDQ 모델은 FP32보다 느리다. 16비트 QDQ 모델은 ORT가 817개의 Memcpy node를 추가했고 model-only latency가 100ms 이상으로 증가했다. 따라서 16비트 결과는 정확도 원인 분석에는 유용하지만 배포 latency 후보로 해석하면 안 된다.

## 논의

Naive INT8이 mAP 0으로 붕괴한 원인은 단순히 8비트라서가 아니라 양자화 범위가 다르기 때문이다. Naive 모델은 graph output과 postprocess까지 양자화하고 weight storage까지 int8로 접는다. 반면 AIMET QDQ 모델은 postprocess와 최종 output을 float로 유지하고 Conv weight는 QDQ 경로를 거치지만 initializer storage는 FP32다.

A8W8 AIMET QDQ의 정확도 손실은 주로 activation 쪽에서 발생한다. A16W8은 full COCO에서 A8W8보다 +0.0182 mAP50-95, A8W16보다 +0.0080 높았다. sample100의 all-activation-float ablation은 weight QDQ만 남긴 상태에서도 A16W8과 거의 같은 mAP를 보였다. 특히 YOLO head Conv output 24개만 float로 되돌려도 큰 회복이 있어 head activation encoding이 다음 최적화 대상이다. Head 내부에서는 sample500 기준 `cv3` branch가 가장 강한 후보이다.

Runtime 관점에서는 정확도와 배포 효율이 분리된다. AIMET QDQ는 분석에는 유효하지만 ORT CUDA에서 QDQ 노드가 packed INT8 kernel로 충분히 접히지 않으면 FP32보다 느릴 수 있다. 최종 배포 주장은 TensorRT, QNN, 또는 ORT의 EP 친화 quantized operator 변환처럼 실제 타깃 runtime에 맞는 export를 따로 검증해야 한다.

## 한계

1. CLE와 AdaRound는 아직 full COCO 정식 설정으로 재평가하지 않았다.
2. AdaRound는 `adaround-samples 8`, `iterations 50` smoke 설정이다.
3. 현재 AIMET QDQ 모델은 packed INT8 deployment artifact가 아니다.
4. Latency는 RTX 3070 WSL2 ORT CUDA 기준이며 Android/QNN 성능을 대변하지 않는다.
5. Sensitivity ablation은 QDQ 제거로 원인을 국소화하지만, 최종 모델 개선 기법은 별도 실험이 필요하다.

## 재현 명령

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --eval-samples 100 --batch 1
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --calibration-samples 64 --eval-samples 100 --batch 1
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --batch 1

scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --eval-samples 500 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu

scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu

scripts/run_native.sh python scripts/09_quantization_coverage.py
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant all_activations --force
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 500 --variant head_cv3_outputs --variant head_scale2_outputs --variant head_final_outputs --force
scripts/run_native.sh python scripts/08_benchmark_latency.py --experiment-id C --experiment-name aimet_quantsim_a8w8_qdq_latency --model results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.onnx --device 0 --warmup-runs 20 --measured-runs 100
```

## 최종 리포트까지 남은 작업

| 우선순위 | 작업 | 완료 기준 |
| --- | --- | --- |
| P0 | full COCO val로 주요 정확도 재평가 | FP32, naive INT8, A8W8, 16비트 후보 완료. AdaRound 정식은 P1로 분리 |
| P0 | Head activation 후보 확대 검증 | sample500에서 `cv3`, `scale2`, final outputs 완료. 다음은 `cv3` 중심 per-layer/range 설정 |
| P1 | AdaRound 정식 설정 | 충분한 sample/iteration으로 A8W8 대비 개선 여부 판단 |
| P1 | Runtime 타깃 분리 | ORT CUDA QDQ와 실제 배포 후보 export를 분리한 latency 표 |
| P2 | Figure 생성 | accuracy-latency Pareto, QDQ coverage, activation sensitivity bar chart |
| P2 | 최종 원고 정리 | 초록, 방법, 결과, 논의, 한계, 재현성 체크리스트 완성 |
