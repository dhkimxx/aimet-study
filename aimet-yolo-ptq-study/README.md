# AIMET YOLO PTQ Study

This project is a reproducible study scaffold for learning AIMET through a YOLO ONNX post-training quantization ablation.

Repository root: `C:\Users\dhkim\OneDrive\문서\AIMET`

The primary question is:

> How do YOLO quantization accuracy, latency, and memory change as AIMET usage increases from no AIMET to QuantSim, CLE, AdaRound, and AutoQuant?

## Locked Decisions

| Area | Decision |
| --- | --- |
| Study focus | Learn AIMET PTQ behavior through ablation |
| Model | Latest Ultralytics YOLO pretrained ONNX, starting with YOLO26 |
| Dataset | COCO 2017 val, 5k images |
| Input | Batch size 1, 640x640 |
| Primary accuracy metric | COCO box mAP50-95 |
| Secondary metrics | mAP50, mAP75, AP small/medium/large, precision, recall |
| Latency | Both model-only and end-to-end |
| Environment | Windows host, WSL2 Ubuntu, NVIDIA GPU |
| AIMET setup | Docker-first using official/prebuilt AIMET image |
| Scope | PTQ only, no QAT in phase 1 |

## Experiment Matrix

| ID | Experiment | Purpose |
| --- | --- | --- |
| A | FP32 ONNX baseline | Accuracy, latency, and memory reference |
| B | No-AIMET naive ONNX INT8 | Compare against generic ONNX Runtime quantization |
| C | AIMET QuantSim PTQ | Learn calibration, encodings, and simulated quantization |
| D | AIMET CLE + QuantSim | Measure range equalization impact |
| E | AIMET AdaRound + QuantSim | Measure optimized weight rounding impact |
| F | AIMET AutoQuant | Compare against AIMET automated PTQ workflow |

## Project Layout

```text
aimet-yolo-ptq-study/
  configs/
    experiment.yaml
    quantization.yaml
  docker/
    Dockerfile.runtime
    build_runtime_image.sh
    run_aimet_onnx_gpu.sh
    verify_aimet.py
  data/
    README.md
  docs/
    decision_log.md
    environment.md
  models/
    README.md
  reports/
    aimet_ptq_study.md
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
  src/
    aimet_yolo_study/
```

## Quick Start

Run from WSL2 Ubuntu, not from native Windows PowerShell:

```bash
cd /path/to/AIMET/aimet-yolo-ptq-study
bash docker/build_runtime_image.sh
bash docker/run_aimet_onnx_gpu.sh
```

Inside the container:

```bash
python -m pip install -r requirements.txt
python docker/verify_aimet.py
python scripts/00_check_env.py
```

The default runtime Docker image is:

```text
aimet-yolo-onnx-gpu:2.2.0
```

It is built from the official/prebuilt AIMET development image and installs:

- AIMET ONNX 2.2.0 CUDA 11.8 wheel
- project study dependencies from `requirements.txt`
- a `python` symlink to `python3`

Override the runtime image when needed:

```bash
AIMET_IMAGE="<image>:<tag>" bash docker/run_aimet_onnx_gpu.sh
```

## Study Workflow

1. Verify WSL2 GPU and Docker access.
2. Enter the AIMET ONNX GPU container.
3. Download or mount COCO 2017 val 5k.
4. Export or place the YOLO ONNX model under `models/`.
5. Run FP32 ONNX evaluation.
6. Run no-AIMET naive ONNX INT8 evaluation.
7. Run AIMET QuantSim PTQ.
8. Run AIMET CLE, AdaRound, and AutoQuant experiments.
9. Benchmark model-only and end-to-end latency.
10. Fill `reports/aimet_ptq_study.md` with metrics and interpretation.

## Asset Preparation

From inside the AIMET container, prepare the reproducible COCO validation assets:

```bash
python scripts/01_prepare_coco.py --download
```

Prepare the YOLO26 nano ONNX model:

```bash
python scripts/01_prepare_yolo_onnx.py --export
```

By default this exports `yolo26n.pt` to:

```text
models/yolo26n_pretrained.onnx
```

Run the first two accuracy baselines:

```bash
python scripts/02_eval_fp32_onnx.py --device 0
python scripts/03_eval_naive_int8_onnx.py --device 0
python scripts/04_aimet_quantsim_ptq.py --device 0
python scripts/05_aimet_cle_ptq.py --device 0
python scripts/06_aimet_adaround_ptq.py --device 0
```

Benchmark latency for any exported ONNX model:

```bash
python scripts/08_benchmark_latency.py --experiment-id A --experiment-name fp32_onnx --device 0
python scripts/08_benchmark_latency.py --experiment-id B --experiment-name naive_onnx_int8 --model results/models/yolo26n_pretrained.naive_int8.onnx --device 0
```

## References

- AIMET installation docs: https://quic.github.io/aimet-pages/releases/2.2.0/install/index.html
- AIMET Docker install notes: https://github.com/quic/aimet/blob/develop/packaging/docker_install.md
- Ultralytics YOLO models: https://docs.ultralytics.com/models
