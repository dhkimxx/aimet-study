# Decision Log

## 2026-06-14

| Topic | Decision | Rationale |
| --- | --- | --- |
| Optimization priority | Accuracy first, then latency, then memory | The study should avoid over-optimizing speed at the cost of losing mAP. |
| Deployment target | PC GPU first, Android phone later | PC GPU is the development baseline; Android constraints should shape export choices. |
| Model | Latest Ultralytics YOLO pretrained ONNX, starting with YOLO26 | The current study is framework-neutral and ONNX-oriented. |
| Dataset | COCO 2017 val 5k | Standard, reproducible detection benchmark. |
| Accuracy metric | COCO box mAP50-95 | Primary COCO detection metric and sensitive to localization quality. |
| Latency metrics | Model-only and end-to-end | Separates quantized model speed from preprocessing/postprocessing cost. |
| Input shape | 1x3x640x640 | Standard YOLO comparison point. |
| AIMET scope | PTQ only | Keeps phase 1 focused on AIMET quantization tools without training. |
| Output format | Reproducible experiment project | Enables reuse when a custom dataset is selected later. |
| Runtime | WSL2 Ubuntu + NVIDIA GPU + Docker | Aligns with AIMET Linux/GPU assumptions while staying practical on Windows. |
| Docker strategy | Official/prebuilt AIMET image first | Minimizes setup time and lets the study focus on AIMET behavior. |
