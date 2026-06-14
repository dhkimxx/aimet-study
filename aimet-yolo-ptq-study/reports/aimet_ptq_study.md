# AIMET YOLO PTQ Study Report

## Summary

TBD after experiments.

## Experiment Setup

| Field | Value |
| --- | --- |
| Model | YOLO26 pretrained ONNX |
| Dataset | COCO 2017 val 5k |
| Input | 1x3x640x640 |
| Primary metric | box mAP50-95 |
| Environment | WSL2 Ubuntu, NVIDIA GPU, AIMET ONNX Docker |

## Accuracy Results

| ID | Experiment | mAP50-95 | mAP50 | mAP75 | AP small | AP medium | AP large | Delta mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | FP32 ONNX | TBD | TBD | TBD | TBD | TBD | TBD | 0.00 |
| B | No-AIMET naive ONNX INT8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| C | AIMET QuantSim PTQ | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D | AIMET CLE + QuantSim | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E | AIMET AdaRound + QuantSim | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| F | AIMET AutoQuant | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Latency Results

| ID | Experiment | Model-only mean ms | Model-only p95 ms | End-to-end mean ms | End-to-end p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| A | FP32 ONNX | TBD | TBD | TBD | TBD |
| B | No-AIMET naive ONNX INT8 | TBD | TBD | TBD | TBD |
| C | AIMET QuantSim PTQ | TBD | TBD | TBD | TBD |
| D | AIMET CLE + QuantSim | TBD | TBD | TBD | TBD |
| E | AIMET AdaRound + QuantSim | TBD | TBD | TBD | TBD |
| F | AIMET AutoQuant | TBD | TBD | TBD | TBD |

## Memory Results

| ID | Experiment | Peak GPU MB | Peak host MB |
| --- | --- | ---: | ---: |
| A | FP32 ONNX | TBD | TBD |
| B | No-AIMET naive ONNX INT8 | TBD | TBD |
| C | AIMET QuantSim PTQ | TBD | TBD |
| D | AIMET CLE + QuantSim | TBD | TBD |
| E | AIMET AdaRound + QuantSim | TBD | TBD |
| F | AIMET AutoQuant | TBD | TBD |

## Interpretation Notes

Use this section to explain:

- Which AIMET stage recovered the most accuracy.
- Whether no-AIMET INT8 is a fair baseline for this model.
- Whether AP small is more sensitive than AP medium/large.
- Whether model-only latency improvements translate to end-to-end latency.
- Whether exported artifacts are suitable for later Android exploration.
