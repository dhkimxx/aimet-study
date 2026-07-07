# Head cv3 Per-Layer Sensitivity

최종 업데이트: 2026-07-08

A8W8 QDQ 모델에서 YOLO head `cv3` branch Conv output activation QDQ를 하나씩 float로 되돌려 평가한 결과입니다. 목적은 `head_cv3_outputs` 전체 15개 중 어느 activation이 A8W8 손실 회복에 가장 크게 기여하는지 확인하는 것입니다.

- 평가 샘플: 500
- 기준 A8W8 mAP50-95: 0.401179

## Ranked By mAP50-95 Recovery

| Rank | Variant | Scale | Layer | mAP50-95 | Delta | Enc scale | Enc range |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | cv3_s1_2_final | 1 | 2_final | 0.4047366675878266 | 0.0035576718 | 0.66714984 | 170.12321 |
| 2 | cv3_s2_1_1 | 2 | 1_1 | 0.3991507859817844 | -0.0020282098 | 0.63858593 | 162.83941 |
| 3 | cv3_s0_0_0 | 0 | 0_0 | 0.3957802739544711 | -0.0053987218 | 0.062587723 | 15.959869 |

## Graph Order

| Index | Variant | Tensor | mAP50-95 | Delta |
| ---: | --- | --- | ---: | ---: |
| 1 | cv3_s0_0_0 | `/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv_output_0` | 0.3957802739544711 | -0.0053987218 |
| 2 | cv3_s1_2_final | `/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0` | 0.4047366675878266 | 0.0035576718 |
| 3 | cv3_s2_1_1 | `/model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv_output_0` | 0.3991507859817844 | -0.0020282098 |

## Interpretation

- 양수 delta가 클수록 해당 activation QDQ가 A8W8 정확도 손실에 더 민감하다는 뜻입니다.
- 개별 tensor 제거 결과는 `head_cv3_outputs` 15개 전체 제거 결과보다 작아야 정상입니다. 전체 제거 효과는 여러 activation의 누적 오차를 포함합니다.
- Encoding scale/range와 delta가 강하게 같이 움직이지 않으면, 단순 range 크기보다 branch 위치와 후속 연산 민감도가 더 큰 원인일 수 있습니다.
