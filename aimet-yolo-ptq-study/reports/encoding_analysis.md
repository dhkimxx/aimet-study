# AIMET Encoding Analysis

최종 업데이트: 2026-07-08

이 문서는 AIMET `.encodings` sidecar를 HW-independent PTQ 관점에서 요약합니다. QDQ ONNX로 실제 평가된 activation과 QDQ export에서 제외된 sidecar-only activation을 분리해 기록합니다. `scale`은 quantization step이므로 같은 range에서는 작을수록 더 촘촘한 양자화입니다.

## Activation Group Summary

| ID | 실험 | QDQ act | Act bits | sidecar-only act | QDQ scale median | Conv scale median | Head cv3 scale median | Head cv3 range median |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| C64 | A8W8 calib64 | 295 | 8:295 | 55 | 0.036705244 | 0.065666676 | 0.16721365 | 42.63948 |
| C1024 | A8W8 calib1024 | 295 | 8:295 | 55 | 0.03661247 | 0.063839592 | 0.1713351 | 43.690451 |
| E128 | AdaRound adar128 iter2000 | 295 | 8:295 | 55 | 0.036030162 | 0.065812215 | 0.19122264 | 48.761773 |
| E256 | AdaRound adar256 iter5000 | 295 | 8:295 | 55 | 0.035939708 | 0.064491045 | 0.19155112 | 48.845535 |
| A16W8 | A16W8 calib64 | 295 | 16:295 | 55 | 0.00018731368 | 0.00033597679 | 0.0014859132 | 97.37932 |
| A8W16 | A8W16 calib64 | 295 | 8:295 | 55 | 0.037597705 | 0.066602677 | 0.192518 | 49.092089 |
| A16W16 | A16W16 calib64 | 295 | 16:295 | 55 | 0.00018694699 | 0.00033060112 | 0.0014981945 | 98.184178 |

## Parameter Summary

| ID | 실험 | Param encodings | Param bits | Symmetric params | Head params | Head cv3 params | Param scale median |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| C64 | A8W8 calib64 | 102 | 8:102 | 102 | 24 | 15 | 0.0032895871 |
| C1024 | A8W8 calib1024 | 102 | 8:102 | 102 | 24 | 15 | 0.0032895871 |
| E128 | AdaRound adar128 iter2000 | 102 | 8:102 | 102 | 24 | 15 | 0.0032895871 |
| E256 | AdaRound adar256 iter5000 | 102 | 8:102 | 102 | 24 | 15 | 0.0032895871 |
| A16W8 | A16W8 calib64 | 102 | 8:102 | 102 | 24 | 15 | 0.0032895871 |
| A8W16 | A8W16 calib64 | 102 | 16:102 | 102 | 24 | 15 | 1.2836467e-05 |
| A16W16 | A16W16 calib64 | 102 | 16:102 | 102 | 24 | 15 | 1.2836467e-05 |

## Interpretation

- `sidecar-only act`는 AIMET encodings에는 있지만 표준 QDQ ONNX 평가에서는 제외된 activation입니다. 현재 QDQ 평가는 graph output과 YOLO postprocess non-Conv tensor를 float로 남깁니다.
- A16W8/A16W16은 activation bitwidth가 16으로 올라가 같은 tensor 수를 유지하면서 activation quantization step을 크게 줄이는 진단용 설정입니다.
- A8W16/A16W16은 parameter encoding bitwidth가 16으로 올라가지만 AIMET QDQ ONNX의 Conv weight initializer storage는 별도 packed int16/int8로 접힌 배포 파일을 의미하지 않습니다.
- AdaRound 중간/full 설정은 parameter rounding을 바꾸지만 activation encoding group 수와 bitwidth는 A8W8과 같으므로 activation 병목 자체를 제거하지 않습니다.
