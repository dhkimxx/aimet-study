# AIMET YOLO PTQ Study

이 프로젝트는 YOLO ONNX 모델을 대상으로 AIMET PTQ를 공부하고, 양자화 방식별 성능 변화를 재현 가능하게 비교하기 위한 실험 스캐폴드입니다.

저장소 루트 기준 경로: `aimet-yolo-ptq-study/`

핵심 질문:

> AIMET을 쓰지 않은 일반 INT8 기준선부터 QuantSim, CLE, AdaRound, AutoQuant까지 활용도를 높이면 YOLO의 정확도, 레이턴시, 메모리 사용량은 어떻게 달라지는가?

## 확정한 의사결정

| 영역 | 결정 |
| --- | --- |
| 스터디 목적 | AIMET PTQ 동작을 ablation으로 학습 |
| 모델 | 최신 Ultralytics YOLO pretrained ONNX 계열, 시작점은 YOLO26 |
| 데이터셋 | COCO 2017 val, 5천 장 |
| 입력 | batch size 1, 640x640 |
| 1차 정확도 지표 | COCO box mAP50-95 |
| 보조 지표 | mAP50, mAP75, AP small/medium/large, precision, recall |
| 레이턴시 | model-only와 end-to-end 모두 측정 |
| 환경 | Windows host, WSL2 Ubuntu native, NVIDIA GPU |
| AIMET 구성 | `uv` 기반 Python 3.10 venv + AIMET ONNX 2.2.0 GPU wheel |
| 범위 | 1단계는 PTQ만 수행, QAT 제외 |

## 실험 매트릭스

| ID | 실험 | 목적 |
| --- | --- | --- |
| A | FP32 ONNX 기준선 | 정확도, 레이턴시, 메모리 기준값 확보 |
| B | no-AIMET naive ONNX INT8 | 일반 ONNX Runtime 양자화와 비교 |
| C | AIMET QuantSim PTQ | 캘리브레이션, encoding, simulated quantization 학습 |
| D | AIMET CLE + QuantSim | range equalization 효과 측정 |
| E | AIMET AdaRound + QuantSim | weight rounding 최적화 효과 측정 |
| F | AIMET AutoQuant | AIMET 자동 PTQ 워크플로와 비교 |

## 프로젝트 구조

```text
aimet-yolo-ptq-study/
  .python-version
  pyproject.toml
  configs/
    experiment.yaml
    quantization.yaml
  data/
    README.md
  docs/
    decision_log.md
    environment.md
    native_uv.md
  models/
    README.md
  reports/
    aimet_ptq_study.md
    paper_report.md
    research_roadmap.md
  results/
    README.md
  scripts/
    00_check_env.py
    01_prepare_coco.py
    01_prepare_yolo_onnx.py
    02_eval_fp32_onnx.py
    03_eval_naive_int8_onnx.py
    04_aimet_quantsim_ptq.py
    05_aimet_cle_ptq.py
    06_aimet_adaround_ptq.py
    07_aimet_autoquant.py
    08_benchmark_latency.py
    09_quantization_coverage.py
    10_activation_sensitivity.py
  src/
    aimet_yolo_study/
```

## Quick Start

Windows PowerShell이 아니라 WSL2 Ubuntu에서 실행합니다.

```bash
cd /path/to/AIMET/aimet-yolo-ptq-study
bash scripts/setup_native_uv.sh
```

수동으로 나눠 실행하려면 다음 순서로 진행합니다.

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
uv python install 3.10
uv sync
scripts/run_native.sh python scripts/00_check_env.py
```

AIMET ONNX 2.2.0 GPU wheel은 Python 3.10 전용입니다. 이 프로젝트는 `.python-version`과 `pyproject.toml`로 Python 3.10 venv를 고정합니다.

네이티브 환경 상세는 `docs/native_uv.md`를 참고합니다.

논문형 리포트 초안은 `reports/paper_report.md`, 최종 리포트까지의 실험 큐와 완료 기준은 `reports/research_roadmap.md`에 정리합니다.
현재 리포트는 sample100 탐색 결과, sample500 확대 검증 결과, full COCO val 핵심 결과를 함께 기록합니다.

## 스터디 진행 순서

1. WSL2 Ubuntu에서 `nvidia-smi`가 정상 실행되는지 확인합니다.
2. `uv`로 Python 3.10 venv와 AIMET ONNX GPU 의존성을 설치합니다.
3. COCO 2017 val 5천 장을 다운로드하거나 마운트합니다.
4. YOLO ONNX 모델을 export하거나 `models/` 아래에 둡니다.
5. FP32 ONNX 기준선 평가를 실행합니다.
6. no-AIMET naive ONNX INT8 평가를 실행합니다.
7. AIMET QuantSim PTQ를 실행합니다.
8. AIMET CLE, AdaRound, AutoQuant 실험을 순서대로 실행합니다.
9. model-only와 end-to-end 레이턴시를 측정합니다.
10. `reports/aimet_ptq_study.md`에 지표와 해석을 기록합니다.

## 실험 자산 준비

WSL2 Ubuntu 네이티브 venv에서 재현 가능한 COCO validation 자산을 준비합니다.

```bash
scripts/run_native.sh python scripts/01_prepare_coco.py --download
```

YOLO26 nano ONNX 모델을 준비합니다.

```bash
scripts/run_native.sh python scripts/01_prepare_yolo_onnx.py --export
```

기본적으로 `yolo26n.pt`를 다음 경로의 ONNX 파일로 export합니다.

```text
models/yolo26n_pretrained.onnx
```

정확도 기준선과 AIMET PTQ 실험을 실행합니다.

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0
scripts/run_native.sh python scripts/05_aimet_cle_ptq.py --device 0
scripts/run_native.sh python scripts/06_aimet_adaround_ptq.py --device 0
```

전체 COCO 평가 전에 빠른 검증을 먼저 돌릴 수 있습니다. `--eval-samples`를 주면 지정한 개수의 이미지만 재현 가능한 샘플로 평가하고, 결과는 `results/metrics_quick.csv`에 기록합니다.

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --eval-samples 100
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --calibration-samples 64 --eval-samples 100
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100
scripts/run_native.sh python scripts/05_aimet_cle_ptq.py --device 0 --calibration-samples 64 --eval-samples 100
```

핵심 후보는 500장 subset으로 확대 검증합니다.

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --eval-samples 500 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --eval-samples 500 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu
```

최종 정확도 표는 COCO val 5천 장 전체로 확인합니다. `--eval-samples`를 생략하면 full val 평가가 실행되고 결과는 `results/metrics.csv`에 기록됩니다.

```bash
scripts/run_native.sh python scripts/02_eval_fp32_onnx.py --device 0 --batch 1 --name fp32_onnx
scripts/run_native.sh python scripts/03_eval_naive_int8_onnx.py --device 0 --batch 1 --calibration-samples 64 --name naive_onnx_int8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --name aimet_quantsim_a8w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 8 --name aimet_quantsim_a16w8_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 8 --weight-bitwidth 16 --name aimet_quantsim_a8w16_gpu
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --batch 1 --calibration-samples 64 --activation-bitwidth 16 --weight-bitwidth 16 --name aimet_quantsim_a16w16_gpu
```

AdaRound는 기본 설정이 오래 걸립니다. 먼저 API와 export 경로만 확인할 때는 작은 smoke 설정을 사용합니다.

```bash
scripts/run_native.sh python scripts/06_aimet_adaround_ptq.py --device 0 --calibration-samples 64 --adaround-samples 8 --adaround-iterations 50 --eval-samples 100
```

AIMET ONNX 2.2.0의 public export는 `.encodings` 파일을 별도로 저장하고 ONNX에서 AIMET quantization 노드를 제거합니다. ONNX Runtime/Ultralytics 평가용 산출물은 반드시 `QuantizeLinear`/`DequantizeLinear` 노드가 들어 있는지 확인합니다.

```bash
scripts/run_native.sh python -c "import onnx; from collections import Counter; m=onnx.load('results/models/yolo26n_pretrained.aimet_quantsim_int8_calib64.onnx'); c=Counter(n.op_type for n in m.graph.node); print(c['QuantizeLinear'], c['DequantizeLinear'])"
```

모델별 양자화 범위는 별도 커버리지 스크립트로 비교합니다. 이 스크립트는 Q/DQ 노드 수, graph input/output QDQ 여부, Conv input/weight/output QDQ 여부, Conv weight가 int initializer로 저장됐는지, initializer dtype, YOLO postprocess QDQ 범위를 `results/quantization_coverage.csv`와 `.json`에 기록합니다.

```bash
scripts/run_native.sh python scripts/09_quantization_coverage.py
```

AIMET QuantSim/CLE/AdaRound 스크립트는 activation/weight bitwidth를 CLI에서 바꿀 수 있습니다. 기본값은 `configs/quantization.yaml`의 A8W8이며, 8/8은 기존 파일명 호환을 위해 `int8` tag를 유지합니다. 16비트 QDQ ONNX는 `QuantizeLinear`/`DequantizeLinear` 타입 제약 때문에 opset 21로 변환됩니다.

```bash
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --activation-bitwidth 16 --weight-bitwidth 8
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --activation-bitwidth 8 --weight-bitwidth 16
scripts/run_native.sh python scripts/04_aimet_quantsim_ptq.py --device 0 --calibration-samples 64 --eval-samples 100 --activation-bitwidth 16 --weight-bitwidth 16
```

A8W8 모델에서 activation QDQ만 선택적으로 제거해 민감한 구간을 확인할 수 있습니다. 기본 변형은 YOLO head Conv 출력, late neck 20-22, 모든 Conv 출력, 모든 activation입니다.

```bash
scripts/run_native.sh python scripts/10_activation_sensitivity.py --no-eval --force
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant head_conv_outputs --force
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant all_activations --force
```

Head Conv 출력은 branch/scale/final output 단위로도 나눠 볼 수 있습니다.

```bash
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 100 --variant head_cv3_outputs --variant head_scale2_outputs --variant head_final_outputs --force
```

sample500 확대 검증에서는 우선 후보만 다시 확인합니다.

```bash
scripts/run_native.sh python scripts/10_activation_sensitivity.py --device 0 --batch 1 --eval-samples 500 --variant head_cv3_outputs --variant head_scale2_outputs --variant head_final_outputs --force
```

내보낸 ONNX 모델의 레이턴시를 벤치마크합니다.

```bash
scripts/run_native.sh python scripts/08_benchmark_latency.py --experiment-id A --experiment-name fp32_onnx --device 0
scripts/run_native.sh python scripts/08_benchmark_latency.py --experiment-id B --experiment-name naive_onnx_int8 --model results/models/yolo26n_pretrained.naive_int8.onnx --device 0
```

## 참고 자료

- AIMET 설치 문서: https://quic.github.io/aimet-pages/releases/2.2.0/install/index.html
- Native uv 환경 노트: docs/native_uv.md
- Ultralytics YOLO 모델: https://docs.ultralytics.com/models
