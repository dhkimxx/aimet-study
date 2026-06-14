# AIMET Lab

이 저장소는 AIMET 기반 모델 최적화 실험을 모아두는 프로젝트 루트입니다.

현재 진행 중인 스터디:

| 디렉터리 | 주제 |
| --- | --- |
| `aimet-yolo-ptq-study/` | YOLO26 ONNX PTQ 실험. AIMET QuantSim, CLE, AdaRound와 no-AIMET 기준선을 비교합니다. |

이 저장소는 우산형 작업 공간으로 구성했습니다. 나중에 YOLO가 아닌 분류, 세그멘테이션, 다른 엣지 모델 실험도 같은 루트 아래에 나란히 둘 수 있게 하기 위함입니다.

권장 확장 구조:

```text
AIMET/
  aimet-yolo-ptq-study/
  aimet-classification-ptq-study/
  aimet-segmentation-ptq-study/
  shared/
```

대용량 데이터셋, 내보낸 모델, 캘리브레이션 캐시, 양자화 산출물, 벤치마크 결과는 git에서 제외합니다. 각 스터디는 설정, 스크립트, 문서, 작은 재현성 메타데이터만 버전 관리합니다.
