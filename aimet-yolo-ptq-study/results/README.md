# 결과 디렉터리

생성된 metric, export된 encoding, 양자화 모델, 벤치마크 결과를 이곳에 기록합니다.
이 디렉터리의 생성 산출물은 기본적으로 git에서 제외됩니다. 문서에 기록한 sample100/sample500 수치는 `metrics_quick.csv`, full COCO 수치는 `metrics.csv`, coverage/latency 수치는 관련 CSV에서 재생성됩니다.

권장 파일과 디렉터리:

```text
metrics.csv
metrics_quick.csv
quantization_coverage.csv
quantization_coverage.json
quantization_coverage_16bit_smoke.csv
quantization_coverage_16bit_smoke.json
quantization_coverage_sensitivity.csv
quantization_coverage_sensitivity.json
quantization_coverage_head_sensitivity.csv
quantization_coverage_head_sensitivity.json
latency.csv
memory.csv
encodings/
models/
logs/
```
