# 데이터 디렉터리

COCO 관련 자산은 다음 명령으로 생성합니다.

```bash
python scripts/01_prepare_coco.py --download
```

예상 결과:

```text
data/coco/
  annotations/
    instances_val2017.json
  val2017/
    000000000139.jpg
    ...
  coco_val2017.yaml
  calibration_images.txt
```

다운로드한 이미지와 annotation 파일은 용량이 크므로 의도적으로 git에서 제외합니다.
