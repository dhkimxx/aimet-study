# Activation Encoding Interventions

최종 업데이트: 2026-07-08

A8W8 QDQ 모델에서 선택한 activation QDQ의 scale/zero-point만 바꿔 평가한 결과입니다. 목적은 QDQ를 float로 제거하지 않고도 `cv3` 민감도가 encoding/range 조정으로 회복되는지 확인하는 것입니다.

- 평가 샘플: 500
- 기준 A8W8 mAP50-95: 0.401179

## Results

| Variant | Target | Method | QDQ | mAP50-95 | Delta | Old scale | New scale | Old range | New range |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head_cv3_outputs_a16 | head_cv3_outputs | a16_preserve_range | 15 | 0.4064200517092647 | 0.0052410559 | 0.32110223 | 0.001249425 | -46.67506..35.206007 | -46.675061..35.206007 |
| cv3_s1_2_final_symmetric_i8 | cv3_s1_2_final | symmetric_i8 | 1 | 0.4030372822297452 | 0.0018582864 | 0.66714984 | 1.3395529 | -170.12321..0 | -171.46277..170.12322 |
| cv3_s1_2_final_a16 | cv3_s1_2_final | a16_preserve_range | 1 | 0.400637435692576 | -0.0005415601 | 0.66714984 | 0.0025959138 | -170.12321..0 | -170.12321..0 |

## Interpretation

- `a16_preserve_range`는 기존 min/max 범위를 유지하면서 selected activation만 uint16 QDQ로 바꿉니다. 좋아지면 bitwidth/step size가 병목이라는 신호입니다.
- `scale_factor`는 기존 zero-point를 유지하고 scale만 바꿔 range clipping 또는 range expansion 효과를 봅니다.
- `percentile_uint8`은 calibration activation percentile로 min/max를 다시 잡아 outlier range가 손실 원인인지 확인합니다.
- 이 결과도 AIMET HW-independent accuracy/encoding 진단입니다. selected QDQ가 uint16이 되면 opset 21 QDQ 평가 모델이며 packed deployment artifact는 아닙니다.
