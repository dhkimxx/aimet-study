# Activation Encoding Interventions

최종 업데이트: 2026-07-08

A8W8 QDQ 모델에서 선택한 activation QDQ의 scale/zero-point만 바꿔 평가한 결과입니다. 목적은 QDQ를 float로 제거하지 않고도 `cv3` 민감도가 encoding/range 조정으로 회복되는지 확인하는 것입니다.

- 평가 샘플: 100
- 기준 A8W8 mAP50-95: 0.51743571

## Results

| Variant | Target | Method | QDQ | mAP50-95 | Delta | Old scale | New scale | Old range | New range |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cv3_s1_2_final_a16 | cv3_s1_2_final | a16_preserve_range | 1 | 0.5197728585767645 | 0.002337152 | 0.66714984 | 0.0025959138 | -170.12321..0 | -170.12321..0 |
| head_cv3_outputs_a16 | head_cv3_outputs | a16_preserve_range | 15 | 0.5266639363703819 | 0.0092282298 | 0.32110223 | 0.001249425 | -46.67506..35.206007 | -46.675061..35.206007 |
| cv3_s1_2_final_scale075 | cv3_s1_2_final | scale_factor | 1 | 0.519200090415934 | 0.0017643838 | 0.66714984 | 0.5003624 | -170.12321..0 | -127.59241..0 |
| cv3_s1_2_final_scale050 | cv3_s1_2_final | scale_factor | 1 | 0.5128946935849825 | -0.004541013 | 0.66714984 | 0.33357492 | -170.12321..0 | -85.061605..0 |
| cv3_s1_2_final_scale125 | cv3_s1_2_final | scale_factor | 1 | 0.5144932876870643 | -0.0029424189 | 0.66714984 | 0.83393729 | -170.12321..0 | -212.65401..0 |
| cv3_s1_2_final_symmetric_i8 | cv3_s1_2_final | symmetric_i8 | 1 | 0.5233239051889296 | 0.0058881986 | 0.66714984 | 1.3395529 | -170.12321..0 | -171.46277..170.12322 |
| cv3_s1_2_final_p999 | cv3_s1_2_final | percentile_uint8 | 1 | 0.5093094806942542 | -0.0081262259 | 0.66714984 | 0.27138755 | -170.12321..0 | -69.203825..0 |

## Interpretation

- `a16_preserve_range`는 기존 min/max 범위를 유지하면서 selected activation만 uint16 QDQ로 바꿉니다. 좋아지면 bitwidth/step size가 병목이라는 신호입니다.
- `scale_factor`는 기존 zero-point를 유지하고 scale만 바꿔 range clipping 또는 range expansion 효과를 봅니다.
- `percentile_uint8`은 calibration activation percentile로 min/max를 다시 잡아 outlier range가 손실 원인인지 확인합니다.
- 이 결과도 AIMET HW-independent accuracy/encoding 진단입니다. selected QDQ가 uint16이 되면 opset 21 QDQ 평가 모델이며 packed deployment artifact는 아닙니다.
