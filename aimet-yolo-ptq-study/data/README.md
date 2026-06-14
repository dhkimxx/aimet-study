# Data Directory

COCO assets are generated here by:

```bash
python scripts/01_prepare_coco.py --download
```

Expected result:

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

The downloaded image and annotation files are intentionally ignored by git.
