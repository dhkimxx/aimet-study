# AIMET Lab

This repository is the project root for AIMET optimization studies.

Current studies:

| Directory | Focus |
| --- | --- |
| `aimet-yolo-ptq-study/` | YOLO26 ONNX PTQ ablation with AIMET QuantSim, CLE, AdaRound, and no-AIMET baselines |

The repo is intentionally organized as an umbrella workspace so future AIMET studies can sit next to the YOLO work instead of being nested inside it.

Suggested future layout:

```text
AIMET/
  aimet-yolo-ptq-study/
  aimet-classification-ptq-study/
  aimet-segmentation-ptq-study/
  shared/
```

Large datasets, exported models, calibration caches, quantized artifacts, and benchmark outputs are ignored by git. Each study should keep configs, scripts, docs, and small reproducibility metadata under version control.
