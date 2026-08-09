# FeCAM은 왜 공유 공분산을 쓰는가, SLDA와 뭐가 다른가

Generated: 2026-08-08. 코드: [`src/models/fecam_head.py`](../src/models/fecam_head.py),
[`dev/run_ssv2_head_curves.py`](../dev/run_ssv2_head_curves.py) (SLDA 구현)
원문: FeCAM — Goswami et al., NeurIPS 2023, [arXiv:2309.14062](https://arxiv.org/abs/2309.14062)
(PDF 직접 확인) · SLDA — Hayes & Kanan, CVPR-W 2020

**질문:** `ssv2_head_curves_result.md`·`ap_fps_sweep_result.md`가 "FeCAM > SLDA > NCM" 순서를
반복 확인하면서 "SLDA와 FeCAM 둘 다 공유 공분산을 쓰는데 정규화 방식이 다르다"고 설명했다.
그런데 그 설명이 일반론(교과서 지식)에서 나온 거였지 원문으로 검증한 게 아니었다 — 이 리포트는
그 검증을 하고, "왜 하필 공유 공분산인가"까지 원문에서 확인해 정리한다.

---

## 1. NCM 대비 공분산을 쓰면 왜 유리한가 (배경)

클래스 평균만 쓰는 NCM은 모든 방향(차원)을 동등하게 취급한다 — 어느 축으로 떨어져 있든
그냥 거리로만 판단한다. 그런데 실제로는 어떤 방향은 클래스 안에서도 원래 값이 크게
흔들리는(분산이 큰) 방향이고, 어떤 방향은 거의 안 흔들리는(분산이 작은, 그래서 믿을 만한)
방향이다. 공분산 행렬은 이 정보를 담고 있고, 그 역행렬(precision)을 거리 계산에 곱하면
분산이 큰 방향은 눌러주고 분산이 작은 방향은 확대해서 재는 효과가 난다(Mahalanobis
distance) — "어느 방향이 진짜 구별 신호인지"를 반영해 거리를 다시 재는 것.

## 2. ⭐ FeCAM이 공유 공분산을 쓰는 이유 — 원문 확인 결과

**원문의 메인 방법은 사실 공유 공분산이 아니다.** FeCAM 논문의 기본 방법은 **클래스마다
따로** 공분산(Σ_y)을 갖는 것이다:

> *"The other alternative is to use a covariance matrix Σy for y ∈ 1,..,Y to represent the
> feature distribution of each class separately."*

**공유 공분산(Σ_{1:t})은 원문이 명시적으로 제시하는 메모리 절약형 대안이다** — 태스크가
늘어날 때마다 "지금까지 본 전체 클래스의 평균 공분산"으로 증분 갱신되는 **하나의**
행렬이다(원문 Eq. 6). 원문이 직접 밝히는 트레이드오프:

> *"[Σ1:t], storing a single covariance matrix representing all classes, already gave
> significantly better [results]... but is still far from FeCAM with a covariance matrix
> [per class]... to [get] common covariance matrix, we pay the price in memory and need to
> store covariance matrix per class [to do better]."*

즉 원문 자신의 벤치마크에서도 **클래스별 공분산이 공유 공분산보다 더 정확하다** — 그런데
클래스마다 D×D 행렬을 저장해야 해서 메모리가 **클래스 수에 비례**해 폭증한다. 우리
프로젝트는 클래스 수가 아니라 **D 하나**가 메모리를 지배하는 쪽(공유 공분산)을 택했다 —
이건 우리가 임의로 단순화한 게 아니라 **원문이 직접 벤치마크까지 해서 제시한 정식
옵션(Σ1:t)**을 가져다 쓴 것이다.

**이 선택이 이 프로젝트의 목표와 정확히 맞아떨어진다.** `memory_footprint_result.md`가
이미 실측했듯 — 256클래스 슬롯을 다 채워도 클래스 평균(means)은 4.2MB뿐이고, D×D 공분산·
역행렬 두 장이 67MB(D=2048)로 메모리를 지배한다("클래스 수가 아니라 D가 지배한다"). 만약
클래스별 공분산을 썼다면 이 67MB가 **클래스 수만큼 곱해져서**(48클래스면 3.2GB, 174클래스면
11.7GB) CPU-only 엣지 배포라는 이 프로젝트의 전제 자체가 무너진다. 공유 공분산은 "정확도를
조금 양보하고 메모리를 클래스 수와 무관하게 만드는" 선택이고, 이게 정확히 이 프로젝트가
필요로 하는 트레이드오프다.

## 3. SLDA와의 차이 — "공유냐 아니냐"가 아니라 "어떻게 정규화하냐"

**공유 공분산 자체는 FeCAM의 발명이 아니다.** SLDA(Hayes & Kanan, CVPR-W 2020, FeCAM보다
3년 먼저)가 이미 스트리밍 방식으로 하나의 공유 공분산을 유지하는 고전적 LDA 계열 방법이다.
그래서 "SLDA vs FeCAM"의 진짜 차이는 "공유하냐 마냐"가 아니라, **그 공유 공분산을 어떻게
다듬어서 쓰느냐**다.

### 3.1 SLDA — 최소한의 안전장치

```python
prec = np.linalg.inv(self.cov + self.shrink * np.eye(FEATURE_DIM))   # shrink = 0.01
```

- **대각(분산)에만** 고정 상수 0.01을 더한다. 비대각(공분산/상관 항)은 전혀 안 건드린다.
- 0.01은 **데이터 스케일과 무관한 하이퍼파라미터**다 — 특징의 실제 분산 규모가 얼마든
  항상 같은 크기만큼 더한다.
- 목적은 "역행렬이 존재하게 만드는 최소한의 장치"에 가깝다.

### 3.2 FeCAM — 데이터에서 유도한 크기로, 대각·비대각 따로

원문 Eq. 8: `Σs = Σ + γ1·V1·I + γ2·V2·(1−I)` (V1=평균 대각 분산, V2=평균 비대각 공분산).
원문은 many-shot CIL에서 **γ1=1, γ2=1**을 쓴다고 명시하는데, 우리 코드의
`SHRINK_1=1.0`·`SHRINK_2=1.0`이 정확히 이 값이다 — 지어낸 상수가 아니라 원문 그대로다.

```python
diag_mean = diag(raw).sum() / d              # V1: 그 순간 데이터의 평균 분산
off_mean = (raw.sum() - diag(raw).sum()) / (d*(d-1))  # V2: 평균 비대각 공분산
cov[대각] = diag + SHRINK_1 * diag_mean
cov[비대각] = raw + SHRINK_2 * off_mean
# 이후 상관계수 행렬로 정규화(대각=1)한 뒤 역행렬
```

세 가지가 SLDA와 다르다:

1. **비대각도 정규화한다.** SLDA는 분산만 건드리는데, FeCAM은 클래스 간 상관관계도
   같이 조정한다.
2. **정규화 크기가 데이터에서 나온다.** SLDA의 0.01은 고정값이라 D나 backbone이 바뀌면
   상대적 크기가 달라진다. FeCAM은 `diag_mean`/`off_mean`을 그 순간 데이터에서 직접
   계산해서 자동으로 스케일에 맞춘다.
3. **상관계수 공간으로 바꾼 뒤 역행렬을 푼다.** ⭐ 원문이 밝히는 이 단계의 진짜 이유는
   "노이즈 완화"가 아니라 **"클래스마다 분산 스케일이 다른 걸 맞춰서 거리를 비교
   가능하게 만드는 것"**이다:
   > *"due to the notable shift in feature distributions between the old and new classes,
   > the variances are much higher for the new classes. As a result, the Mahalanobis
   > distance of features from different classes will have different scaling factors,
   > and the distances will not be comparable."*
   즉 **증분학습 특유의 문제**(새 클래스는 옛 클래스와 특징 분포가 어긋나 있어 분산
   자체가 다르게 나옴)를 겨냥한 장치다. SLDA는 이 단계 자체가 없다 — 원래 공분산
   스케일 그대로 역행렬을 풀어, 분산이 큰 차원/클래스가 판별에 더 큰 영향을 준다.

**주의:** 원문의 correlation normalization은 원래 **클래스별로 다른 공분산 행렬들을
서로 비교 가능하게** 만들려는 장치였다(§2에서 본 메인 방법, 클래스마다 Σ_y). 우리처럼
공유 공분산(Σ1:t) 하나만 쓰는 경우엔 "비교해야 할 여러 행렬"이 애초에 없으므로, 원문의
1차 동기가 그대로 적용되진 않는다 — 다만 원문 Eq. 10이 "Σ1:t도 이 식(정규화+shrinkage
포함)에 그대로 쓸 수 있다"고 명시하므로, 우리 구현이 원문에서 벗어난 건 아니다.

## 4. few_shot_correction — 원문에 없는, 이 프로젝트의 추가 확장

`FeCAMHead`의 `few_shot_correction` 옵션(적은 샘플로 학습된 클래스의 거리 편향을 보정,
`E[Mahalanobis distance] = true_distance + tr(P·Σ)/n`)은 이번 원문 확인 과정에서
FeCAM 논문 본문에 있는지 검색했으나 **못 찾았다** — SLDA에도 없다. 이건 원문 재현이 아니라
이 프로젝트가 few-shot 등록 시나리오(`enrollment_covariance_result.md`)에서 직접 추가한
확장으로 보인다. 원문 인용이 필요한 부분과 이 프로젝트의 자체 기여를 섞지 않기 위해
명시해둔다.

## 5. 실측 결과와의 연결

| | NCM | SLDA | FeCAM |
|---|---|---|---|
| SSv2 (48cls, avg_inc) | 20.89 | 22.77 | **24.19** |
| CIFAR-100 (avg_inc, `pycil_bridge_result.md`) | 69.8 | 72.5 | **76.9** |

두 데이터셋에서 독립적으로 같은 순서가 재현된다([`ssv2_head_curves_result.md`](ssv2_head_curves_result.md)).
이 리포트가 설명한 메커니즘(공유 공분산 자체는 SLDA도 하지만, FeCAM의 정규화가 데이터
스케일에 맞춰 대각·비대각을 함께 다듬고 클래스 간 분산 이질성까지 겨냥한다는 점)이 이
일관된 우위의 근거로 보인다 — 단, 이건 메커니즘적 설명이지 이 순서차를 정량적으로
분해(decompose)한 별도 ablation은 이 프로젝트에서 수행하지 않았다(§6).

## 6. 정직한 한계

- **원문의 correlation normalization 동기(클래스 간 비교 가능성)가 공유 공분산 세팅에도
  똑같이 강하게 적용되는지는 원문에서 직접 다루지 않는다** — §3.2의 "주의" 참고. 우리
  세팅에서 이 단계가 정확히 얼마나 기여하는지는 별도 ablation(shrinkage 있음/없음,
  normalization 있음/없음)이 필요한데 이 리포트에서는 안 함.
- **"왜 SLDA보다 FeCAM이 이기는가"의 인과관계는 메커니즘 설명이지 증명이 아니다.** 대각+
  비대각 정규화, 데이터 기반 스케일링, 상관계수 정규화 세 가지가 섞여 있어 각각의
  기여도를 분리하는 ablation은 하지 않았다.
- **few_shot_correction은 원문에 없다고 결론 내렸지만, 논문 전체(부록 포함)를 정독한 건
  아니고 키워드 검색(grep) 기준이다** — 다른 표현으로 서술돼 있을 가능성은 완전히
  배제 못 한다.
- **SLDA 원 논문(Hayes & Kanan 2020)은 이번에 원문 확인을 안 했다** — SLDA 쪽 설명은
  우리 코드 구현과 표준 LDA 이론에 근거한 것이지, SLDA 논문 자체를 재확인한 건 아니다.

## 재현

```bash
python3 dev/run_ssv2_head_curves.py           # SSv2 48-class, NCM/SLDA/FeCAM 곡선
```

원문 확인 과정(재현 가능):
```bash
curl -sL -o fecam.pdf https://arxiv.org/pdf/2309.14062
pdftotext -layout fecam.pdf fecam.txt
grep -in "shrink\|correlation matrix\|common covariance" fecam.txt
```
