# Head Group Mixed Precision

최종 업데이트: 2026-07-08

A8W8 QDQ 모델에서 선택한 head activation group QDQ만 uint16 preserve-range로 바꿔 평가한 결과입니다. 목적은 QDQ를 float로 제거하지 않고도 head activation 민감도가 selective A16으로 회복되는지 확인하는 것입니다.

- 평가 샘플: 100
- 기준 A8W8 mAP50-95: 0.51743571

## Results

| Variant | Target | Method | QDQ | mAP50-95 | Delta | Old scale | New scale | Old range | New range |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head_conv_outputs_a16 | head_conv_outputs | a16_preserve_range | 24 | 0.5294692046357964 | 0.012033498 | 0.22816993 | 0.00088782074 | -32.049873..26.133458 | -32.049874..26.133458 |
| head_cv2_outputs_a16 | head_cv2_outputs | a16_preserve_range | 9 | 0.5212576127693802 | 0.0038219062 | 0.073282765 | 0.00028514694 | -7.6745618..11.012543 | -7.6745619..11.012543 |
| head_cv3_outputs_a16 | head_cv3_outputs | a16_preserve_range | 15 | 0.5263898960751285 | 0.0089541895 | 0.32110223 | 0.001249425 | -46.67506..35.206007 | -46.675061..35.206007 |
| head_scale0_outputs_a16 | head_scale0_outputs | a16_preserve_range | 8 | 0.5132495698229688 | -0.0041861368 | 0.11282715 | 0.00043901615 | -16.983185..11.787738 | -16.983186..11.787738 |
| head_scale1_outputs_a16 | head_scale1_outputs | a16_preserve_range | 8 | 0.5220632830776625 | 0.0046275765 | 0.2851544 | 0.0011095502 | -39.431382..33.28299 | -39.431382..33.28299 |
| head_scale2_outputs_a16 | head_scale2_outputs | a16_preserve_range | 8 | 0.5313343136610923 | 0.013898607 | 0.28652823 | 0.0011148959 | -39.735053..33.329646 | -39.735054..33.329646 |
| head_final_outputs_a16 | head_final_outputs | a16_preserve_range | 6 | 0.5245362561190859 | 0.0071005495 | 0.23050917 | 0.00089692286 | -51.517105..7.2627337 | -51.517106..7.2627335 |

## Interpretation

- `a16_preserve_range`는 기존 min/max 범위를 유지하면서 selected activation만 uint16 QDQ로 바꿉니다. 좋아지면 bitwidth/step size가 병목이라는 신호입니다.
- `scale_factor`는 기존 zero-point를 유지하고 scale만 바꿔 range clipping 또는 range expansion 효과를 봅니다.
- `percentile_uint8`은 calibration activation percentile로 min/max를 다시 잡아 outlier range가 손실 원인인지 확인합니다.
- 이 결과도 AIMET HW-independent accuracy/encoding 진단입니다. selected QDQ가 uint16이 되면 opset 21 QDQ 평가 모델이며 packed deployment artifact는 아닙니다.
- sample100에서는 `head_scale2_outputs_a16`과 `head_conv_outputs_a16`이 가장 크게 회복했습니다. 다만 subset 변동 가능성이 있어 sample500 재확인이 필요합니다.
