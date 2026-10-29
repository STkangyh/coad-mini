# FeCAM은 왜 공유 공분산을 쓰는가, SLDA와 뭐가 다른가 — ablation으로 기여도 분해

Generated: 2026-08-08 (ablation 추가로 개정). 코드: [`src/models/fecam_head.py`](../src/models/fecam_head.py),
[`dev/run_ssv2_head_curves.py`](../dev/run_ssv2_head_curves.py) (SLDA 구현),
[`dev/run_covariance_ablation.py`](../dev/run_covariance_ablation.py) (신규, ablation)
원문: FeCAM — Goswami et al., NeurIPS 2023, [arXiv:2309.14062](https://arxiv.org/abs/2309.14062)
(PDF 직접 확인) · SLDA — Hayes & Kanan, CVPR-W 2020
원시결과: `reports/covariance_ablation_raw.json`

**질문:** `ssv2_head_curves_result.md`·`ap_fps_sweep_result.md`가 "FeCAM > SLDA > NCM" 순서를
반복 확인하면서 "SLDA와 FeCAM 둘 다 공유 공분산을 쓰는데 정규화 방식이 다르다"고 설명했다.
이 리포트의 첫 버전은 그 설명을 FeCAM 원문으로 검증까지 했지만, **"어느 차이가 실제로
정확도를 얼마나 좌우하는지"는 재지 않고 메커니즘 설명으로만 남겨뒀다** — 사용자가 "ablation
돌려서 정규화 기여도 분리해봐"라고 요청해서 실제로 쟀고, **결과가 원래 가설과 정반대였다.**

---

## 1. NCM 대비 공분산을 쓰면 왜 유리한가 (배경)

클래스 평균만 쓰는 NCM은 모든 방향(차원)을 동등하게 취급한다 — 어느 축으로 떨어져 있든
그냥 거리로만 판단한다. 그런데 실제로는 어떤 방향은 클래스 안에서도 원래 값이 크게
흔들리는(분산이 큰) 방향이고, 어떤 방향은 거의 안 흔들리는(분산이 작은, 그래서 믿을 만한)
방향이다. 공분산 행렬은 이 정보를 담고 있고, 그 역행렬(precision)을 거리 계산에 곱하면
분산이 큰 방향은 눌러주고 분산이 작은 방향은 확대해서 재는 효과가 난다(Mahalanobis
distance) — "어느 방향이 진짜 구별 신호인지"를 반영해 거리를 다시 재는 것.

## 2. FeCAM이 공유 공분산을 쓰는 이유 — 원문 확인 결과

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

## 3. SLDA와 FeCAM의 코드 차이 (메커니즘 — 기여도는 §4에서 실측)

**공유 공분산 자체는 FeCAM의 발명이 아니다.** SLDA(Hayes & Kanan, CVPR-W 2020, FeCAM보다
3년 먼저)가 이미 스트리밍 방식으로 하나의 공유 공분산을 유지하는 고전적 LDA 계열 방법이다.
코드로 확인되는 차이는 크게 두 갈래다:

**A. 특징 전처리** — FeCAM의 `_prep()`은 공분산·평균을 계산하기 **전에** 모든 특징에
sign-preserving Tukey 변환(`sign(x)·|x|^0.5`, 원문 λ=0.5와 일치)을 적용하고 L2 정규화한다.
SLDA는 원시 특징을 그대로 쓴다.

**B. 공분산 정규화**(원문 Eq. 8: `Σs = Σ + γ1·V1·I + γ2·V2·(1−I)`, γ1=γ2=1이 우리 코드의
`SHRINK_1`·`SHRINK_2`와 일치):
```python
diag_mean = diag(raw).sum() / d              # V1: 평균 대각 분산
off_mean = (raw.sum() - diag(raw).sum()) / (d*(d-1))  # V2: 평균 비대각 공분산
cov[대각] = diag + SHRINK_1 * diag_mean       # SLDA: 대각만, 고정 0.01
cov[비대각] = raw + SHRINK_2 * off_mean       # SLDA: 안 건드림
# 이후 상관계수 행렬로 정규화(대각=1)한 뒤 역행렬                      # SLDA: 이 단계 없음
```

이전 버전의 이 리포트는 여기서 "B(정규화 방식 차이)가 FeCAM이 이기는 진짜 이유"라고
결론 내렸다 — **§4에서 실측한 결과, 이 결론은 틀렸다.**

## 4. ⭐ Ablation — 실제 기여도는 정반대였다

**방법:** SLDA에서 FeCAM으로 한 번에 한 요인씩만 바꾸는 5단계 사다리를 만들어 같은
8-stage SSv2 프로토콜(`ssv2_head_curves_result.md`와 동일 데이터·순서)에 각각 처음부터
끝까지 독립적으로 돌렸다. 두 끝점(step0, step3)은 **기존에 이미 나온 수치(SLDA
22.77/14.14, FeCAM 24.19/15.74)와 정확히 일치해야 한다**는 걸 스크립트 자체가 자동으로
검증하게 만들었다 — 이게 실패하면 중간 단계 숫자를 신뢰할 수 없기 때문.

**⚠️ 이 스크립트 자체가 두 번 틀렸다(교훈으로 남김):**
1. 처음엔 정규화 수식만 재구현하고 §3-A(Tukey+L2 전처리)를 통째로 빠뜨렸다 — step3가
   FeCAM(24.19)이 아니라 SLDA보다도 낮은 18.55로 나와서 바로 잡았다.
2. 전처리를 넣은 뒤에도 손으로 재구현한 정규화 수식이 실제 FeCAM과 0.2~0.35pp 차이가
   나서 원인을 못 찾다가, **아예 실제 `FeCAMHead` 객체를 가져다 `_cov_terms()`만
   단계별로 바꿔치기**하는 방식으로 재작성했다 — 그러자 step3는 실제 클래스를 안 건드린
   것이므로 **구조적으로** 정확히 일치하게 됐다.

**결과(last acc 기준, 8단계 전체 완주 후):**

| 단계 | 무엇을 더했나 | last acc | 이전 대비 |
|---|---|---|---|
| step0 | SLDA (원시 특징, 대각만, 고정 0.01) | 14.14 | — |
| **step0.5** | **+ Tukey 변환 + L2 정규화** | **15.57** | **+1.42pp** |
| step1 | + 대각 shrink를 데이터 기반 크기로(γ1·V1) | 15.74 | +0.17pp |
| step2 | + 비대각도 shrink(γ2·V2, 원시 스케일) | 15.74 | +0.00pp |
| step3 | + 상관계수 정규화(= FeCAM 그대로) | 15.74 | +0.00pp |

**FeCAM−SLDA 전체 격차 1.60pp 중 89%(1.42pp)가 §3-A(전처리)에서 나왔다.** §3-B(공분산
정규화)가 기여하는 건 **대각 shrink 강화분 0.17pp(11%)뿐**이고, 비대각 shrink와 상관계수
정규화는 이 세팅에서 **실질적으로 0**이었다 — 이전 버전이 "진짜 차이"라고 지목했던 게
정확도 격차의 대부분을 설명하지 못한다는 뜻이다.

### 4.1 왜 비대각 shrink는 미미했나

각 단계가 정말 독립 계산이었는지부터 검증했다 — step1과 step2의 원시(반올림 전) 값을
직접 비교하면 stage7에서 0.16382…(step1) vs 0.16358…(step2)로 **실제로 다르다**(복사가
아니다). 다만 8단계 누적 후 last acc에서는 그 차이가 반올림 오차 수준으로 사라진다 —
비대각 shrink 자체는 존재하지만 이 데이터·설정에서 판별 결정을 거의 안 바꾼다는 뜻이다.

### 4.2 ⭐ 상관계수 정규화는 우연히 0이 아니라 — 수학적으로 정확히 0이다

step2와 step3의 원시값은 **8단계 전부, 소수점 17자리까지 완전히 동일**했다(`0.1635787420770356`
등). 우연이라기엔 너무 정확해서 대수적으로 확인했다:

상관계수 정규화는 `Σ_corr = D⁻¹ Σ D⁻¹`로 바꾼다(D = 표준편차 대각행렬). 역행렬은
`Σ_corr⁻¹ = D·Σ⁻¹·D`가 된다. 그런데 `scores()`는 쿼리 x와 클래스 평균 μ도 **똑같은 D로
나눠서**(`x/sd`, `μ/sd`) 쓴다. 둘을 합치면:

```
(x/D)ᵀ · (D·Σ⁻¹·D) · (x/D) = xᵀ·D⁻¹D·Σ⁻¹·DD⁻¹·x = xᵀ Σ⁻¹ x
```

**D가 정확히 상쇄돼서, 상관계수 정규화를 하든 안 하든 최종 판별 점수가 수학적으로 완전히
같아진다.** 이건 데이터나 세팅과 무관한 항등식이다.

**왜 원문은 그런데도 이 단계를 쓰는가:** 원문의 상관계수 정규화는 애초에 **여러 개의
서로 다른 클래스별 공분산 행렬을 서로 비교 가능하게** 만들려던 장치다(§2의 메인 방법,
클래스마다 Σ_y). 공유 공분산(Σ1:t) **하나만** 쓰면 "비교해야 할 다른 행렬"이 원천적으로
없으므로, 이 장치가 이 세팅에서 **작동할 대상 자체가 없어서** no-op이 된 것 — 원문을
벗어난 게 아니라, 원문의 전제(여러 행렬)가 우리 세팅(행렬 하나)에서 성립하지 않는 경우다.
이전 버전이 "정직한 한계"로 남겨뒀던 의심("공유 공분산 세팅에도 원문의 동기가 그대로
적용되는지 불확실")이 여기서 **증명으로 확정**됐다.

## 5. 결론 — 진짜 이유는 정규화가 아니라 전처리였다

두 데이터셋에서 FeCAM > SLDA > NCM 순서가 재현된다는 관찰([`ssv2_head_curves_result.md`](ssv2_head_curves_result.md),
CIFAR-100 `pycil_bridge_result.md`)은 그대로 유효하다. 다만 **그 이유는 "공분산을 더
정교하게 정규화해서"가 아니라, 대부분(89%) "특징을 Tukey 변환 + L2 정규화해서 더
Gaussian에 가깝고 스케일이 고른 분포로 만든 다음 공분산을 추정하기 때문"**이다. 공분산
정규화 자체의 실질 기여는 대각 shrink 강화분 11%뿐이고, 비대각 shrink·상관계수 정규화는
이 공유-공분산 세팅에서 사실상/수학적으로 무의미하다.

## 6. few_shot_correction — 원문에 없는, 이 프로젝트의 추가 확장

`FeCAMHead`의 `few_shot_correction` 옵션(적은 샘플로 학습된 클래스의 거리 편향을 보정,
`E[Mahalanobis distance] = true_distance + tr(P·Σ)/n`)은 FeCAM 논문 본문에서 검색했으나
**못 찾았다** — SLDA에도 없다. 이 ablation은 균형 잡힌 전체 클래스 데이터로만 돌아서
이 옵션 자체가 no-op이라(문서화된 성질) 기여도 분해에 포함하지 않았다.

## 7. 정직한 한계

- **전처리(Tukey+L2)의 기여를 "묶음"으로만 쟀다.** Tukey 변환만 단독으로 켰을 때와 L2
  정규화만 단독으로 켰을 때를 분리하지 않았다 — 89%라는 숫자는 "이 두 가지를 합친 것"의
  기여지, 어느 쪽이 더 큰지는 모른다.
- **SSv2 48클래스, mean pooling, 단일 curriculum 순서 하나에서만 쟀다.** 174클래스나
  chunks4 pooling, 다른 클래스 순서에서도 89%/11%/0% 비율이 유지되는지는 확인 안 했다.
- **§4.2의 대수적 증명은 "공유 공분산 하나"라는 전제에서만 성립한다.** 클래스별 공분산을
  여러 개 쓰는 원문의 메인 방법에서는 상관계수 정규화가 실제로 기여할 수 있다 — 이
  리포트는 그 세팅을 재현하지 않았다.
- **few_shot_correction의 원문 부재는 grep 기준**이다 — 논문 전체(부록 포함)를 정독한 건
  아니라 다른 표현으로 서술돼 있을 가능성은 배제 못 한다.
- **SLDA 원 논문(Hayes & Kanan 2020)은 원문 확인을 안 했다** — SLDA 쪽 설명은 코드
  구현과 표준 LDA 이론에 근거한 것이지, SLDA 논문 자체를 재확인한 건 아니다.

## 재현

```bash
python3 dev/run_covariance_ablation.py       # 이 리포트의 ablation (step0~step3)
python3 dev/run_ssv2_head_curves.py          # SSv2 48-class, NCM/SLDA/FeCAM 곡선(비교 기준)
```

원문 확인 과정(재현 가능):
```bash
curl -sL -o fecam.pdf https://arxiv.org/pdf/2309.14062
pdftotext -layout fecam.pdf fecam.txt
grep -in "shrink\|correlation matrix\|common covariance" fecam.txt
```
