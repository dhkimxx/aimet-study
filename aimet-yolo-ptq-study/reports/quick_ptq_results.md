# AIMET YOLO PTQ 빠른 검증 결과

작성일: 2026-06-20

이 문서는 COCO 전체 평가 전에 `--eval-samples 100`으로 실행한 빠른 검증 결과를 정리합니다. 목적은 최종 성능 판단이 아니라, no-AIMET INT8과 AIMET 기반 PTQ 경로가 실제로 어떤 차이를 보이는지 빠르게 확인하는 것입니다.

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
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --batch 1
scripts/run_native.sh python scripts/05_aimet_cle_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --batch 1
scripts/run_native.sh python scripts/06_aimet_adaround_ptq.py --device 0 --calibration-samples 64 --adaround-samples 8 --adaround-iterations 50 --eval-samples 100 --batch 1
```

## 결과 요약

| ID | 실험 | AIMET 사용 | 설정 | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | FP32 ONNX | 아니오 | sample100 | 0.5437 | 0.6657 | 0.5854 | 0.6454 | 0.6251 |
| B | naive ONNX INT8 | 아니오 | calib64, sample100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C | AIMET QuantSim PTQ | 예 | calib64, sample100 | 0.5439 | 0.6660 | 0.5854 | 0.6447 | 0.6259 |
| D | AIMET CLE + QuantSim | 예 | calib64, sample100 | 0.5437 | 0.6657 | 0.5852 | 0.6453 | 0.6252 |
| E | AIMET AdaRound + QuantSim | 예 | calib64, adar8, iter50, sample100 | 0.5441 | 0.6674 | 0.5875 | 0.6454 | 0.6288 |

## 해석

- FP32 ONNX는 100장 샘플 기준으로 정상적인 detection 성능을 보였습니다.
- no-AIMET naive ONNX INT8은 mAP가 0으로 떨어졌습니다. YOLO 모델에 일반적인 ONNX Runtime static quantization을 그대로 적용하면 출력 분포나 후처리 호환성이 깨질 수 있음을 보여줍니다.
- AIMET QuantSim PTQ는 같은 calibration 64장 조건에서 FP32와 거의 같은 mAP를 유지했습니다. 이 빠른 검증에서는 AIMET encoding 기반 PTQ가 단순 INT8 대비 핵심 차이를 만든 것으로 볼 수 있습니다.
- CLE + QuantSim은 현재 샘플과 모델에서는 QuantSim 단독과 거의 같은 결과였습니다. 전체 COCO 평가나 더 큰 샘플에서 차이가 나는지 확인이 필요합니다.
- AdaRound 결과는 `adaround-samples 8`, `iterations 50`의 smoke 설정입니다. FP32보다 근소하게 높은 mAP가 나왔지만 표본과 iteration이 작으므로 정식 AdaRound 성능으로 해석하면 안 됩니다.

## 다음 실험

1. `--eval-samples 500` 또는 전체 COCO val로 표본을 키워 quick 결과의 안정성을 확인합니다.
2. QuantSim과 CLE는 calibration sample을 64에서 1024로 늘려 full 설정을 평가합니다.
3. AdaRound는 충분한 sample과 iteration으로 다시 실행합니다.
4. 정확도 비교가 안정화되면 `scripts/08_benchmark_latency.py`로 model-only와 end-to-end 레이턴시를 측정합니다.
