# AIMET YOLO PTQ 빠른 검증 결과

작성일: 2026-06-20

이 문서는 COCO 전체 평가 전에 `--eval-samples 100`으로 실행한 빠른 검증 결과를 정리합니다. 목적은 최종 성능 판단이 아니라, no-AIMET INT8과 AIMET 기반 PTQ 경로가 실제로 어떤 차이를 보이는지 빠르게 확인하는 것입니다.

중요 정정: AIMET ONNX 2.2.0의 public `QuantizationSimModel.export()`는 ONNX를 저장하기 전에 AIMET `QcQuantizeOp` 노드를 제거하고 별도 `.encodings` 파일을 저장합니다. ONNX Runtime/Ultralytics는 이 `.encodings` 파일을 자동 적용하지 않으므로, QDQ 노드가 없는 AIMET ONNX를 평가한 이전 C/D/E 값은 INT8 결과가 아니라 FP32 ONNX 평가로 봐야 합니다. 아래 표는 `QuantizeLinear`/`DequantizeLinear` 노드가 포함된 ONNX로 다시 실행한 결과입니다.

## 실행 조건

| 항목 | 값 |
| --- | --- |
| 모델 | `models/yolo26n_pretrained.onnx` |
| 데이터셋 | COCO 2017 val subset |
| 평가 샘플 | 100 images |
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

## 16비트 조합

아래 결과는 QuantSim 단독 C 경로에서 A8W8/A16W8/A8W16/A16W16을 같은 `calib64`, `sample100`, `CUDAExecutionProvider` 조건으로 비교한 값입니다. 목적은 activation bitwidth와 weight bitwidth 중 어느 쪽이 정확도에 더 민감한지 확인하는 것입니다.

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
| A16W8 | 21 | 397/397 | 102/102 | 0/102 | 102 | 0 | 0 | 295 | 9.882 |
| A8W16 | 21 | 397/397 | 102/102 | 0/102 | 0 | 295 | 102 | 0 | 9.890 |
| A16W16 | 21 | 397/397 | 102/102 | 0/102 | 0 | 0 | 102 | 295 | 9.890 |

16비트 QDQ는 ONNX `QuantizeLinear`/`DequantizeLinear`의 `int16/uint16` 타입 지원 때문에 opset 21이 필요합니다. 기존 YOLO ONNX는 opset 17이라 단순히 opset 숫자만 올리면 `ReduceMax` 같은 op schema가 깨지므로, export 시 ONNX version converter로 opset 21 변환을 수행합니다.

sample100 기준으로 A16W8이 가장 높았습니다. A8W16도 A8W8보다 회복했지만 A16W8보다 낮고, A16W16은 A16W8보다 오히려 낮았습니다. 따라서 현재 YOLO QDQ 경로에서는 weight보다 activation quantization error가 더 민감해 보입니다.

다만 16비트 QDQ 모델을 CUDAExecutionProvider로 로드할 때 ONNX Runtime이 `817 Memcpy nodes are added` 경고를 냈습니다. 정확도는 회복되지만 CUDA kernel 배치가 비효율적일 수 있다는 뜻이므로, 16비트 조합은 정확도 후보로는 의미가 있어도 레이턴시 후보로는 별도 benchmark가 필요합니다.

중요하게, 위 AIMET 모델들은 여전히 QDQ accuracy-eval 모델입니다. `Conv weight QDQ`는 102/102지만 `Conv weight INT storage`는 0/102라 ONNX 파일 안의 Conv weight 본체는 FP32 initializer입니다. 따라서 A8W16/A16W16 결과는 "16비트 weight quantization을 시뮬레이션했을 때 정확도"로 읽어야 하고, 배포 파일 크기/속도 이득을 의미하지 않습니다.

## 산출물 해시

| ID | 모델 SHA256 | 비고 |
| --- | --- | --- |
| A | `5cb19c918a1ff7a4178ab7b1f0fc878a7b67e8baf012af443c6b06372946a4f2` | FP32 |
| B | `29d4343971e6dc8609631ba302101634ce96e38e023accb4077402333a056d34` | naive QDQ INT8 |
| C | `37ddc7829f1d23504762626bb9adae5511607d2a1a7218ea5d0f3d0a58180a95` | AIMET QDQ |
| D | `00c59e859fe38d2554299d54b1c4ceb670306b8bbb84387540d0fb0ff00b2136` | AIMET QDQ |
| E | `00757f981c03df09f94f1fa7239211f71b6a0ce60627ef17218055ca187a380c` | AIMET QDQ |

## 해석

- FP32 ONNX는 100장 샘플 기준으로 정상적인 detection 성능을 보였습니다.
- no-AIMET naive ONNX INT8은 mAP가 0으로 떨어졌습니다. YOLO 모델에 일반적인 ONNX Runtime static quantization을 그대로 적용하면 출력 분포나 후처리 호환성이 깨질 수 있음을 보여줍니다.
- AIMET QuantSim PTQ를 실제 QDQ ONNX로 평가하면 FP32 대비 mAP50-95가 약 0.0263 낮았습니다. 그래도 naive ONNX INT8처럼 완전히 붕괴하지는 않았습니다.
- CLE + QuantSim은 현재 샘플에서는 QuantSim 단독보다 약간 낮았습니다. 이 모델은 BatchNorm 없는 ONNX로 export되어 high-bias folding도 적용되지 않았습니다.
- AdaRound 결과는 `adaround-samples 8`, `iterations 50`의 smoke 설정입니다. 이 조건에서는 QuantSim/CLE보다 높은 mAP50-95를 보였지만, 정식 AdaRound 성능으로 보려면 sample과 iteration을 늘려 다시 평가해야 합니다.
- 현재 QDQ export는 YOLO detection postprocess 영역의 비-Conv 텐서와 최종 `output0` QDQ를 제외합니다. postprocess까지 양자화하면 sample20 기준 mAP가 0으로 떨어졌기 때문입니다.
- B와 C/D/E는 양자화 수준이 다릅니다. B는 input/output/postprocess까지 더 공격적으로 양자화하고 weight storage도 int8이지만 정확도가 붕괴했습니다. C/D/E는 postprocess/output을 float로 남기고 weight도 QDQ 경로만 거치는 accuracy-eval 모델이라 정확도는 유지되지만 파일 크기/배포 효율 비교에는 아직 직접 쓰면 안 됩니다.

## 다음 실험

1. `--eval-samples 500` 또는 전체 COCO val로 표본을 키워 quick 결과의 안정성을 확인합니다.
2. QuantSim과 CLE는 calibration sample을 64에서 1024로 늘려 full 설정을 평가합니다.
3. AdaRound는 충분한 sample과 iteration으로 다시 실행합니다.
4. YOLO head/postprocess 제외 정책을 더 명시적으로 설정하거나, postprocess 없는 raw-head ONNX export로 다시 비교합니다.
5. AIMET QDQ 모델의 weight를 int8 initializer로 접는 deployment export 경로를 별도로 확인합니다.
6. 정확도 비교가 안정화되면 `scripts/08_benchmark_latency.py`로 model-only와 end-to-end 레이턴시를 측정합니다.
