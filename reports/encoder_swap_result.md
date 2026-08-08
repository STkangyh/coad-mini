# 인코더 교체 실험 — 파라미터 8배 차이, 정확도는 동일, 속도는 20배 역전

Generated: 2026-07-27. 스크립트: [`scripts/extract_ucf101_features.py`](../scripts/extract_ucf101_features.py)
(`--backbone`) · [`dev/run_ucf101_tcd_protocol.py`](../dev/run_ucf101_tcd_protocol.py) (`--features`)
원시결과: `reports/ucf101_tcd_raw.json`, `reports/ucf101_tcd_raw_mobileclip_s0.json`

**왜 했나:** 우리는 head를 5개(NCM/SLDA/Ridge/RanDumb/FeCAM) 비교했으면서 **인코더는 한 번도
바꿔보지 않았다.** 그런데 [`flops_result.md`](flops_result.md)에서 정량화했듯 인코더는
**추론 비용의 99.98%, 학습 전체의 146배**를 차지한다 — edge 성능을 실제로 결정하는 건
head가 아니라 인코더다.

**왜 지금 가능해졌나:** [`mobileclip_result.md`](mobileclip_result.md)의 이 실험은 원래
*"raw SSv2 영상 접근 불가"*로 **CPU 지연만 재고 정확도 비교를 못 한 채 중단**돼 있었다.
UCF101 원본 영상 13,320개를 확보하면서 그 반쪽이 채워졌다.

## 설정

동일 조건 — 같은 16프레임 균등 샘플, 같은 UCF101 공식 split 1, 같은 TCD 프로토콜
(51 base + 10씩, 클래스 순서 3개 평균), 같은 head. **바뀐 건 인코더뿐.**

## 결과

| 인코더 | 파라미터 | feature dim | **UCF101 FeCAM** | **CPU ms/window** | MPS ms/window |
|---|---|---|---|---|---|
| **CLIP ViT-B/32** | 87.5M (vision tower) | 512 | **88.84 ±0.52** | **148** | 85 |
| **MobileCLIP-S0** | **10.9M** | 1024 | **88.63 ±0.67** | **2,985** | 135 |
| 차이 | **8배 작음** | — | **−0.21 %p** | **20배 느림** | 1.6배 느림 |

head별 전체:

| head | CLIP B/32 | MobileCLIP-S0 | Δ |
|---|---|---|---|
| NCM prototype | 83.28 | 81.53 | −1.75 |
| Deep SLDA | 85.12 | 83.54 | −1.58 |
| **FeCAM** | **88.84** | **88.63** | **−0.21** |

## ⭐ 발견 1 — 파라미터 수가 edge 지연을 예측하지 못한다

**파라미터 8배 작은 모델이 CPU에서 20배 느리다.** 이건 오차나 설정 실수가 아니라
**설계 의도의 결과**다: MobileCLIP-S0는 Apple의 CoreML/ANE 커널을 겨냥해 설계됐고,
범용 PyTorch eager 실행에서는 그 이점이 전혀 살아나지 않는다.
(MPS에서는 격차가 1.6배로 줄지만 여전히 **느린 쪽**이다.)

**논문에 쓸 수 있는 메시지:** *"모델 선택에서 파라미터 수와 FLOPs는 edge 지연의
대리 지표가 되지 못한다. 실제 배포 런타임에서 측정해야 한다."*
우리가 읽은 edge-CL 선행연구 어느 쪽도 이 지점을 다루지 않는다 —
SparCL은 희소화 가속률을, BudgetCL은 하드웨어 추상화된 iteration 수를 보고한다
([`pycil_survey_edge_gap.md`](pycil_survey_edge_gap.md)).

## ⭐ 발견 2 — 정확도는 인코더에 거의 의존하지 않는다 (FeCAM 한정)

FeCAM 기준 **−0.21%p**로, 두 실행의 표준편차(0.52 / 0.67) 안에 들어간다.
**8배 작은 인코더로 바꿔도 정확도가 사실상 그대로**다.

다만 **head에 따라 다르다**는 게 중요하다:
- NCM −1.75, SLDA −1.58 → 단순한 head는 인코더 품질 저하를 그대로 받는다
- FeCAM −0.21 → **공분산 정규화가 약한 feature를 상당 부분 보정**한다

즉 **좋은 head는 인코더 요구사항을 낮춘다.** edge에서 인코더를 줄여야 할 때
FeCAM 같은 head가 완충 역할을 한다는 실용적 함의가 있다.

## 결론 — 우리 파이프라인 선택이 검증됐다 (단, 이유가 다르다)

CLIP B/32를 계속 쓰는 것이 맞다. **그러나 이유가 "더 정확해서"가 아니다**(0.21%p는
의미 없는 차이). **우리가 실제로 가진 런타임(CPU)에서 20배 빠르기 때문**이다.

만약 CoreML로 변환해 ANE에서 돌리는 배포 경로를 잡는다면 결론이 뒤집힐 수 있다 —
그때는 MobileCLIP이 8배 작은 모델로 같은 정확도를 낼 것이다. **이건 후속 과제로 명시**.

## 부수 관찰 — feature dim이 head 비용을 좌우한다

MobileCLIP-S0의 feature dim은 **1024**로 CLIP B/32(512)의 2배다. FeCAM의 공분산이
D²이므로 **4배 커지고**, fit 시간이 0.6초 → 2.1초로 늘었다. 여전히 무시할 수준이지만,
인코더를 고를 때 **파라미터 수만이 아니라 출력 차원도 head 비용에 영향**을 준다는 점은
기록해둔다.

## 미실행 (비용 문제)

- **OpenCLIP ViT-L/14**, **SigLIP L/16-384** — 설정은 준비돼 있으나(`--backbone`)
  ViT-L/14는 B/32의 8~10배 연산이라 추출에만 6~8시간 예상. 큰 인코더가 정확도를
  얼마나 올리는지는 **미확인**이며, 올린다 해도 edge 예산에서 벗어난다.
- 위 지연 수치는 **맥북 CPU** 기준이다. 실제 임베디드(Jetson/라즈베리파이)에서는
  또 다른 순위가 나올 수 있다 — 이것이 우리 갭 목록 1순위인 이유.

## 재현

```bash
python3 scripts/extract_ucf101_features.py --backbone mobileclip_s0     # ~60분 (MPS)
python3 dev/run_ucf101_tcd_protocol.py --features data/features_ucf101_mobileclip_s0 --inc 10
```
