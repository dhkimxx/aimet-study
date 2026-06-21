# AIMET YOLO PTQ 빠른 검증 결과

작성일: 2026-06-21

이 문서는 COCO 전체 평가 전에 `--eval-samples 100`으로 실행한 빠른 검증 결과, 핵심 후보를 `--eval-samples 500`으로 확대한 결과, 그리고 full COCO val에서 확인한 핵심 결과를 정리합니다. 목적은 no-AIMET INT8과 AIMET 기반 PTQ 경로가 실제로 어떤 차이를 보이는지 재현 가능한 숫자로 확인하는 것입니다.

중요 정정: AIMET ONNX 2.2.0의 public `QuantizationSimModel.export()`는 ONNX를 저장하기 전에 AIMET `QcQuantizeOp` 노드를 제거하고 별도 `.encodings` 파일을 저장합니다. ONNX Runtime/Ultralytics는 이 `.encodings` 파일을 자동 적용하지 않으므로, QDQ 노드가 없는 AIMET ONNX를 평가한 이전 C/D/E 값은 INT8 결과가 아니라 FP32 ONNX 평가로 봐야 합니다. 아래 표는 `QuantizeLinear`/`DequantizeLinear` 노드가 포함된 ONNX로 다시 실행한 결과입니다.

## 실행 조건

| 항목 | 값 |
| --- | --- |
| 모델 | `models/yolo26n_pretrained.onnx` |
| 데이터셋 | COCO 2017 val |
| 평가 샘플 | 100 images quick, 500 images expanded, 5000 images full |
| 입력 크기 | 640x640 |
| Batch size | 1 |
| Runtime | WSL2 native uv venv, AIMET ONNX, CUDAExecutionProvider |
| GPU | NVIDIA GeForce RTX 3070 |
| 결과 CSV | `results/metrics_quick.csv` |

## 실행 명령

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --eval-samples 100 --batch 1
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --calibration-samples 64 --eval-samples 100 --batch 1
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --batch 1 --name aimet_quantsim_ptq_sample100_qdq
scripts/run_native.sh python scripts/05_aimet_cle_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --batch 1 --name aimet_cle_ptq_sample100_qdq --force
scripts/run_native.sh python scripts/06_aimet_adaround_ptq.py --device 0 --calibration-samples 64 --adaround-samples 8 --adaround-iterations 50 --eval-samples 100 --batch 1 --name aimet_adaround_ptq_sample100_qdq --force
```

## 산출물 검증

| 모델 | Q | DQ | Input QDQ | Output QDQ | Conv input QDQ | Conv weight QDQ | Conv output QDQ | Conv weight INT storage | `/model.23` non-Conv QDQ | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 ONNX | 0 | 0 | 0/1 | 0/1 | 0/102 | 0/102 | 0/102 | 0/102 | 0 | 기준선 |
| naive ONNX INT8 | 389 | 593 | 1/1 | 1/1 | 102/102 | 102/102 | 102/102 | 102/102 | 59 | postprocess까지 양자화 |
| AIMET QuantSim PTQ | 397 | 397 | 1/1 | 0/1 | 68/102 | 102/102 | 102/102 | 0/102 | 0 | postprocess/output float 유지 |
| AIMET CLE + QuantSim | 397 | 397 | 1/1 | 0/1 | 68/102 | 102/102 | 102/102 | 0/102 | 0 | postprocess/output float 유지 |
| AIMET AdaRound + QuantSim | 397 | 397 | 1/1 | 0/1 | 68/102 | 102/102 | 102/102 | 0/102 | 0 | postprocess/output float 유지 |

`Conv weight QDQ`는 Conv weight 입력이 QDQ 경로를 통과한다는 뜻이고, `Conv weight INT storage`는 ONNX 파일 안의 weight가 실제 int8 initializer로 저장됐는지를 뜻합니다. 현재 AIMET C/D/E는 QDQ accuracy evaluation 모델이며, naive B처럼 weight storage까지 접힌 배포 최적화 모델은 아닙니다.

## 결과 요약

| ID | 실험 | AIMET 사용 | 설정 | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | FP32 ONNX | 아니오 | sample100 | 0.5437 | 0.6657 | 0.5854 | 0.6454 | 0.6251 |
| B | naive ONNX INT8 | 아니오 | calib64, sample100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C | AIMET QuantSim PTQ | 예 | calib64, sample100, QDQ | 0.5174 | 0.6524 | 0.5516 | 0.7065 | 0.5739 |
| D | AIMET CLE + QuantSim | 예 | calib64, sample100, QDQ | 0.5131 | 0.6417 | 0.5502 | 0.7308 | 0.5560 |
| E | AIMET AdaRound + QuantSim | 예 | calib64, adar8, iter50, sample100, QDQ | 0.5307 | 0.6552 | 0.5711 | 0.6644 | 0.5700 |

## sample500 확대 검증

아래 결과는 같은 seed와 CUDAExecutionProvider 조건에서 핵심 후보를 500장으로 확대한 값입니다. sample500은 아직 전체 COCO val은 아니지만, sample100에서 보인 경향이 단순 샘플 우연인지 확인하는 중간 검증입니다.

실행 명령:

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --eval-samples 500 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu
```

정확도:

| 실험 | 설정 | Runtime | mAP50-95 | mAP50 | mAP75 | Precision | Recall | FP32 대비 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 ONNX | sample500 | CUDA | 0.4203 | 0.5689 | 0.4551 | 0.6820 | 0.5215 | 0.0000 |
| naive ONNX INT8 | calib64, sample500 | CUDA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.4203 |
| AIMET QuantSim PTQ | A8W8, calib64, sample500 | CUDA | 0.4012 | 0.5518 | 0.4337 | 0.6355 | 0.5111 | -0.0191 |
| AIMET QuantSim PTQ | A16W8, calib64, sample500 | CUDA | 0.4143 | 0.5666 | 0.4496 | 0.6648 | 0.5158 | -0.0060 |
| AIMET QuantSim PTQ | A8W16, calib64, sample500 | CUDA | 0.4074 | 0.5580 | 0.4465 | 0.6467 | 0.5093 | -0.0130 |
| AIMET QuantSim PTQ | A16W16, calib64, sample500 | CUDA | 0.4170 | 0.5663 | 0.4495 | 0.6543 | 0.5142 | -0.0033 |

해석: naive INT8의 mAP 0 붕괴는 sample500에서도 그대로 재현됩니다. A8W8 QDQ는 FP32 대비 -0.0191 mAP50-95로 정확도를 유지하지만 완전한 복원은 아닙니다. 16비트 조합은 A16W16이 FP32에 가장 가깝고, 단일 축 비교에서는 A16W8이 A8W16보다 더 크게 회복합니다. 따라서 sample500에서도 activation bitwidth를 올리는 효과가 weight bitwidth를 올리는 효과보다 큽니다.

## full COCO val 확인

아래 결과는 COCO 2017 val 5천 장 전체를 `batch 1`, `CUDAExecutionProvider` 조건으로 평가한 값입니다. `--eval-samples`를 생략해 실행했습니다.

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 1024 --name aimet_quantsim_a8w8_calib1024_gpu --force
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu
```

| 실험 | 설정 | Runtime | mAP50-95 | mAP50 | mAP75 | Precision | Recall | FP32 대비 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 ONNX | full val | CUDA | 0.3971 | 0.5512 | 0.4311 | 0.6689 | 0.4990 | 0.0000 |
| naive ONNX INT8 | calib64, full val | CUDA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.3971 |
| AIMET QuantSim PTQ | A8W8, calib64, full val | CUDA | 0.3740 | 0.5321 | 0.4059 | 0.6450 | 0.4949 | -0.0231 |
| AIMET QuantSim PTQ | A8W8, calib1024, full val | CUDA | 0.3787 | 0.5346 | 0.4130 | 0.6491 | 0.4929 | -0.0184 |
| AIMET QuantSim PTQ | A16W8, calib64, full val | CUDA | 0.3923 | 0.5490 | 0.4273 | 0.6606 | 0.5009 | -0.0049 |
| AIMET QuantSim PTQ | A8W16, calib64, full val | CUDA | 0.3843 | 0.5392 | 0.4179 | 0.6543 | 0.4957 | -0.0128 |
| AIMET QuantSim PTQ | A16W16, calib64, full val | CUDA | 0.3952 | 0.5489 | 0.4288 | 0.6583 | 0.5020 | -0.0019 |

해석: full COCO에서도 sample500 결론이 유지됩니다. naive INT8은 완전히 붕괴했고, A8W8 QDQ는 정확도를 유지하되 FP32 대비 -0.0231 mAP50-95 손실이 남았습니다. Calibration sample을 64에서 1024로 늘린 A8W8은 0.3787로 +0.0047만 회복했습니다. A16W8이 A8W16보다 높아 activation 쪽 민감도가 더 크고, A16W16은 FP32에 거의 붙지만 ORT CUDA latency가 100ms 이상이라 배포 후보가 아니라 정확도 상한으로 해석합니다.

## 16비트 조합

아래 결과는 QuantSim 단독 C 경로에서 A8W8/A16W8/A8W16/A16W16을 같은 `calib64`, `CUDAExecutionProvider` 조건으로 비교한 값입니다. 목적은 activation bitwidth와 weight bitwidth 중 어느 쪽이 정확도에 더 민감한지 확인하는 것입니다.

실행 명령:

```bash
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 100 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_sample100_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 100 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_sample100_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 100 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_sample100_gpu
```

정확도:

| 실험 | 설정 | Runtime | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AIMET QuantSim PTQ | A8W8, calib64, sample100 | CUDA | 0.5174 | 0.6524 | 0.5516 | 0.7065 | 0.5739 |
| AIMET QuantSim PTQ | A16W8, calib64, sample100 | CUDA | 0.5449 | 0.6695 | 0.5862 | 0.6266 | 0.6025 |
| AIMET QuantSim PTQ | A8W16, calib64, sample100 | CUDA | 0.5347 | 0.6588 | 0.5810 | 0.6625 | 0.5813 |
| AIMET QuantSim PTQ | A16W16, calib64, sample100 | CUDA | 0.5374 | 0.6632 | 0.5783 | 0.6514 | 0.6042 |

양자화 커버리지:

| 실험 | Opset | Q/DQ | Conv weight QDQ | Conv weight INT storage | init int8 | init uint8 | init int16 | init uint16 | 모델 크기 MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A8W8 | 17 | 397/397 | 102/102 | 0/102 | 102 | 295 | 0 | 0 | 9.786 |
| A8W8 calib1024 | 17 | 397/397 | 102/102 | 0/102 | 102 | 295 | 0 | 0 | 9.786 |
| A16W8 | 21 | 397/397 | 102/102 | 0/102 | 102 | 0 | 0 | 295 | 9.882 |
| A8W16 | 21 | 397/397 | 102/102 | 0/102 | 0 | 295 | 102 | 0 | 9.890 |
| A16W16 | 21 | 397/397 | 102/102 | 0/102 | 0 | 0 | 102 | 295 | 9.890 |

16비트 QDQ는 ONNX `QuantizeLinear`/`DequantizeLinear`의 `int16/uint16` 타입 지원 때문에 opset 21이 필요합니다. 기존 YOLO ONNX는 opset 17이라 단순히 opset 숫자만 올리면 `ReduceMax` 같은 op schema가 깨지므로, export 시 ONNX version converter로 opset 21 변환을 수행합니다.

sample100 기준으로 A16W8이 가장 높았습니다. sample500과 full 기준으로는 A16W16이 가장 높지만, 단일 축 비교에서는 sample500 A16W8 0.4143 > A8W16 0.4074, full A16W8 0.3923 > A8W16 0.3843입니다. 따라서 현재 YOLO QDQ 경로에서는 weight보다 activation quantization error가 더 민감해 보입니다. 다만 A16W16은 정확도만 보면 FP32에 가장 가깝지만, 아래 latency 결과처럼 배포 후보로 해석하면 안 됩니다.

다만 16비트 QDQ 모델을 CUDAExecutionProvider로 로드할 때 ONNX Runtime이 `817 Memcpy nodes are added` 경고를 냈습니다. 정확도는 회복되지만 CUDA kernel 배치가 비효율적일 수 있다는 뜻이므로, 16비트 조합은 정확도 후보로는 의미가 있어도 레이턴시 후보로는 별도 benchmark가 필요합니다.

중요하게, 위 AIMET 모델들은 여전히 QDQ accuracy-eval 모델입니다. `Conv weight QDQ`는 102/102지만 `Conv weight INT storage`는 0/102라 ONNX 파일 안의 Conv weight 본체는 FP32 initializer입니다. 따라서 A8W16/A16W16 결과는 "16비트 weight quantization을 시뮬레이션했을 때 정확도"로 읽어야 하고, 배포 파일 크기/속도 이득을 의미하지 않습니다.

## Activation QDQ 민감도

A16W8이 A8W16보다 더 크게 회복했기 때문에, A8W8 QDQ 모델에서 activation QDQ만 선택적으로 제거한 변형을 만들었습니다. weight QDQ는 그대로 두고, 선택한 activation `QuantizeLinear -> DequantizeLinear` 쌍만 float `Identity`로 바꿔 같은 `sample100`, `CUDAExecutionProvider` 조건으로 평가했습니다.

실행 명령:

```bash
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant head_conv_outputs --force
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant late_neck_20_22 --force
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant all_conv_outputs --force
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant all_activations --force
```

정확도:

| 모델 | 제거한 activation QDQ | 남은 Q/DQ | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A8W8 기준 | 0 | 397/397 | 0.5174 | 0.6524 | 0.5516 | 0.7065 | 0.5739 |
| head Conv outputs float | 24 | 373/373 | 0.5327 | 0.6611 | 0.5728 | 0.6834 | 0.5790 |
| late neck 20-22 float | 33 | 364/364 | 0.5238 | 0.6525 | 0.5609 | 0.6820 | 0.5793 |
| all Conv outputs float | 102 | 295/295 | 0.5377 | 0.6652 | 0.5737 | 0.6520 | 0.5868 |
| all activations float | 295 | 102/102 | 0.5440 | 0.6672 | 0.5855 | 0.6260 | 0.6001 |
| A16W8 참고 | 해당 없음 | 397/397 | 0.5449 | 0.6695 | 0.5862 | 0.6266 | 0.6025 |

양자화 커버리지:

| 모델 | Input QDQ | Conv input QDQ | Conv weight QDQ | Conv output QDQ | Conv weight INT storage |
| --- | ---: | ---: | ---: | ---: | ---: |
| A8W8 기준 | 1/1 | 68/102 | 102/102 | 102/102 | 0/102 |
| head Conv outputs float | 1/1 | 68/102 | 102/102 | 78/102 | 0/102 |
| late neck 20-22 float | 1/1 | 59/102 | 102/102 | 92/102 | 0/102 |
| all Conv outputs float | 1/1 | 68/102 | 102/102 | 0/102 | 0/102 |
| all activations float | 0/1 | 0/102 | 102/102 | 0/102 | 0/102 |

해석: activation 전체를 float로 돌리고 weight QDQ만 남기면 mAP50-95가 0.5440으로 회복되어 A16W8 0.5449와 거의 같습니다. 즉 현재 A8W8 손실의 주된 원인은 weight rounding/storage가 아니라 activation QDQ입니다. 특히 YOLO head Conv output 24개만 float로 되돌려도 0.5174에서 0.5327로 크게 회복되어, head 주변 activation range와 scale 선택이 민감한 구간으로 보입니다.

`all_activations` 변형도 Conv weight QDQ 102/102를 유지하므로 완전 FP32가 아닙니다. 반대로 Conv weight INT storage는 여전히 0/102라, 이 결과도 배포용 packed INT8 성능이 아니라 fake-quant/QDQ 정확도 민감도 결과입니다.

## YOLO head activation 세분화

`head_conv_outputs` 24개 QDQ 제거가 큰 회복을 보였으므로, 같은 head 영역을 branch, scale, final Conv output 기준으로 다시 나눠 평가했습니다.

실행 명령:

```bash
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant head_cv2_outputs --variant head_cv3_outputs --variant head_scale0_outputs --variant head_scale1_outputs --variant head_scale2_outputs --variant head_final_outputs --force
```

정확도:

| 모델 | 제거한 activation QDQ | 남은 Q/DQ | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A8W8 기준 | 0 | 397/397 | 0.5174 | 0.6524 | 0.5516 | 0.7065 | 0.5739 |
| head cv2 outputs float | 9 | 388/388 | 0.5198 | 0.6540 | 0.5606 | 0.7394 | 0.5653 |
| head cv3 outputs float | 15 | 382/382 | 0.5252 | 0.6556 | 0.5653 | 0.6886 | 0.5670 |
| head scale0 outputs float | 8 | 389/389 | 0.5202 | 0.6572 | 0.5578 | 0.6807 | 0.5896 |
| head scale1 outputs float | 8 | 389/389 | 0.5235 | 0.6575 | 0.5678 | 0.7155 | 0.5651 |
| head scale2 outputs float | 8 | 389/389 | 0.5266 | 0.6591 | 0.5677 | 0.7154 | 0.5788 |
| head final outputs float | 6 | 391/391 | 0.5221 | 0.6616 | 0.5635 | 0.7015 | 0.5711 |
| all head Conv outputs float | 24 | 373/373 | 0.5327 | 0.6611 | 0.5728 | 0.6834 | 0.5790 |

해석: branch 기준으로는 `cv3` 쪽이 `cv2`보다 더 큰 회복을 보였고, scale 기준으로는 `scale2`가 가장 컸습니다. final Conv output 6개만 float로 되돌려도 A8W8 대비 mAP50-95가 0.0047 회복됩니다. 다만 `all head Conv outputs` 24개를 모두 float로 돌린 0.5327에는 못 미치므로 head 내부의 마지막 출력뿐 아니라 중간 Conv activation도 누적 영향을 줍니다.

sample500 재확인:

```bash
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 500 --variant head_cv3_outputs --variant head_scale2_outputs --variant head_final_outputs --force
```

| 모델 | 제거한 activation QDQ | 남은 Q/DQ | mAP50-95 | mAP50 | mAP75 | Precision | Recall | A8W8 대비 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A8W8 기준 | 0 | 397/397 | 0.4012 | 0.5518 | 0.4337 | 0.6355 | 0.5111 | 0.0000 |
| head cv3 outputs float | 15 | 382/382 | 0.4105 | 0.5605 | 0.4499 | 0.6609 | 0.5039 | +0.0094 |
| head scale2 outputs float | 8 | 389/389 | 0.4055 | 0.5564 | 0.4388 | 0.6360 | 0.5090 | +0.0043 |
| head final outputs float | 6 | 391/391 | 0.4024 | 0.5537 | 0.4383 | 0.6420 | 0.5074 | +0.0012 |

sample500에서도 `head_cv3_outputs`가 세 후보 중 가장 큰 회복을 보였습니다. `scale2`와 final output은 개선폭이 작아졌고, final output만으로는 A8W8 손실을 거의 설명하지 못합니다. 따라서 다음 후보는 final output 단독보다 head `cv3` branch와 더 넓은 head Conv output 범위입니다.

## 산출물 해시

| ID | 모델 SHA256 | 비고 |
| --- | --- | --- |
| A | `5cb19c918a1ff7a4178ab7b1f0fc878a7b67e8baf012af443c6b06372946a4f2` | FP32 |
| B | `29d4343971e6dc8609631ba302101634ce96e38e023accb4077402333a056d34` | naive QDQ INT8 |
| C | `37ddc7829f1d23504762626bb9adae5511607d2a1a7218ea5d0f3d0a58180a95` | AIMET QDQ |
| D | `00c59e859fe38d2554299d54b1c4ceb670306b8bbb84387540d0fb0ff00b2136` | AIMET QDQ |
| E | `00757f981c03df09f94f1fa7239211f71b6a0ce60627ef17218055ca187a380c` | AIMET QDQ |

## 해석

- FP32 ONNX는 full COCO val 기준 mAP50-95 0.3971로 정상적인 detection 성능을 보였습니다.
- no-AIMET naive ONNX INT8은 mAP가 0으로 떨어졌습니다. YOLO 모델에 일반적인 ONNX Runtime static quantization을 그대로 적용하면 출력 분포나 후처리 호환성이 깨질 수 있음을 보여줍니다.
- AIMET QuantSim PTQ를 실제 QDQ ONNX로 평가하면 full COCO 기준 A8W8 calib64는 FP32 대비 mAP50-95가 0.0231 낮았습니다. 그래도 naive ONNX INT8처럼 완전히 붕괴하지는 않았습니다.
- Calibration sample을 64장에서 1024장으로 늘린 A8W8은 full COCO mAP50-95가 0.3740에서 0.3787로 올랐습니다. 회복폭이 +0.0047이라 calibration 부족만으로 A8W8 손실을 설명하기는 어렵습니다.
- CLE + QuantSim은 현재 샘플에서는 QuantSim 단독보다 약간 낮았습니다. 이 모델은 BatchNorm 없는 ONNX로 export되어 high-bias folding도 적용되지 않았습니다.
- AdaRound 결과는 `adaround-samples 8`, `iterations 50`의 smoke 설정입니다. 이 조건에서는 QuantSim/CLE보다 높은 mAP50-95를 보였지만, 정식 AdaRound 성능으로 보려면 sample과 iteration을 늘려 다시 평가해야 합니다.
- 16비트 조합과 activation QDQ 제거 실험을 같이 보면, 현재 정확도 손실은 weight보다 activation 쪽이 더 큽니다. full COCO에서도 A16W8이 A8W16보다 높고, sample100의 `all_activations` float 변형은 weight QDQ만 남긴 상태로 A16W8과 거의 같은 mAP까지 회복했습니다.
- YOLO head 세분화에서는 sample100 기준 `cv3` branch와 `scale2` 쪽 activation이 상대적으로 더 민감했습니다. sample500에서는 `cv3`가 세 후보 중 가장 일관된 회복을 보였습니다. final output만이 아니라 head 중간 Conv activation도 함께 영향을 줍니다.
- 현재 QDQ export는 YOLO detection postprocess 영역의 비-Conv 텐서와 최종 `output0` QDQ를 제외합니다. postprocess까지 양자화하면 sample20 기준 mAP가 0으로 떨어졌기 때문입니다.
- B와 C/D/E는 양자화 수준이 다릅니다. B는 input/output/postprocess까지 더 공격적으로 양자화하고 weight storage도 int8이지만 정확도가 붕괴했습니다. C/D/E는 postprocess/output을 float로 남기고 weight도 QDQ 경로만 거치는 accuracy-eval 모델이라 정확도는 유지되지만 파일 크기/배포 효율 비교에는 아직 직접 쓰면 안 됩니다.

## 다음 실험

1. CLE는 calibration sample을 64에서 1024로 늘려 full 설정을 평가합니다.
2. AdaRound는 충분한 sample과 iteration으로 sample500 이상에서 다시 실행합니다.
3. YOLO head/postprocess 제외 정책을 더 명시적으로 설정하거나, postprocess 없는 raw-head ONNX export로 다시 비교합니다.
4. Head `cv3` branch와 wider head Conv output 범위에 대해 per-layer range, percentile, symmetric/asymmetric 설정 민감도를 확인합니다.
5. AIMET QDQ 모델의 weight를 int8 initializer로 접는 deployment export 경로를 별도로 확인합니다.
