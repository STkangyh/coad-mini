# Base-heavy(TCD 스타일) split 재검증 — FeCAM vs GRU+A-GEM

Generated: 2026-07 (auto). 배경: TCD(Park et al., ICCV 2021)는 SSv2 전체(174cls)를
**84-class base + 소수 incremental session**(base:increment = 8.4~16.8배)으로 나눈 CIL
벤치마크에서 **NME(prototype 계열)가 CNN보다 못하다**고 보고했다. 우리의 핵심 결론
("FeCAM > GRU+A-GEM")과 정면으로 긴장되는 반례라, **우리 데이터로 직접 재검증**했다.
(TCD의 174-class 원본 split 자체는 이 세션에서 영상 접근이 막혀 재현 불가 — 대신
구조적 변수(base-heaviness)를 우리 48-class subset에 이식해 테스트. 상세 배경은
`reports/sota_positioning_brief.md` §1(f).)

**리터럴 재현 업데이트(2026-08-06):** SSv2 원본 영상을 확보해 174클래스 전체·84 base +
9×10 incremental(TCD의 실제 스케일)로 다시 검증했다 —
[`ssv2_video_access_result.md §8`](ssv2_video_access_result.md). 결론은 그대로다: FeCAM이
3시드 전부에서 GRU+A-GEM을 이겼다(avg_inc +3.78pp, last +2.97pp) — 여기 48클래스 근사가
낸 결론이 리터럴 스케일에서도 뒤집히지 않았다. 원본은 그대로 두고 이 문단만 추가함.

## 설정

우리 48클래스(8-stage×6class 큐레이션, 기존 실험과 동일 순서)를 두 가지 base-heavy
구조로 재편, **true class-IL 평가**(task ID 없음, 지금까지 본 전체 클래스에서 argmax):

| split | 세션 구성 | base:increment 비 | (참고) TCD 원본 |
|---|---|---|---|
| moderate | 24 + [6,6,6,6] | 4배 | — |
| aggressive | 36 + [6,6] | 6배 | 84:10=8.4배, 84:5=16.8배 |
| (참고) 우리 원래 protocol | 6×8 균등 | 1배 | — |

> ## ⚠️ 2026-07-31 정정 — baseline이 굶고 있었다
>
> 최초 실행은 A-GEM에게 **세션당 고정 50개**의 replay 예산을 줬다. 그건 균등 8×6
> 프로토콜의 설정이고 거기선 클래스당 8.33개를 뜻하지만, **36클래스 base 세션도 50개**를
> 받아 클래스당 1.39개로 쪼그라들었다. 즉 **연구 대상 변수(base 크기)가 baseline의
> 예산을 직접 깎고 있었다.**
>
> 표준 CIL은 클래스당 예산을 쓴다(TCD: SSv2 20/class, UCF101 5/class). 그래서 원래
> 비율 **8.33개/class를 유지한 채 세션 크기에만 비례**시켜 재실행했다. 6클래스 세션은
> 여전히 정확히 50개를 받으므로 **균등 프로토콜 수치는 전혀 바뀌지 않는다.**
>
> `--memory fixed`가 최초 수치를 **정확히 재현**했다(GRU 0.118/0.109, 0.124/0.084).
> 바뀐 변수는 예산 하나뿐이다.
>
> **결론 중 살아남은 것과 철회한 것은 §1에 정리했다.**

## 결과 (CLIP B/32, GRU+A-GEM은 3-seed 평균)

`scaled`가 공정한 수치이고, `fixed`는 정정 전 값을 대조용으로 남긴 것이다.

| split | method | replay 예산 | Avg Inc | Last | per-session |
|---|---|---|---|---|---|
| moderate | **FeCAM** | 사용 안 함 | **0.178** | **0.157** | 0.193 → 0.192 → 0.183 → 0.164 → 0.157 |
| moderate | GRU+A-GEM (fixed) | [50,50,50,50,50] | 0.118 | 0.109 | 0.163 → 0.125 → 0.096 → 0.095 → 0.109 |
| moderate | **GRU+A-GEM (scaled)** | **[200,50,50,50,50]** | 0.115 | 0.103 | 0.163 → 0.112 → 0.108 → 0.087 → 0.103 |
| aggressive | **FeCAM** | 사용 안 함 | **0.168** | **0.157** | 0.183 → 0.164 → 0.157 |
| aggressive | GRU+A-GEM (fixed) | [50,50,50] | 0.124 | 0.084 | **0.213 → 0.075** → 0.084 |
| aggressive | **GRU+A-GEM (scaled)** | **[300,50,50]** | 0.138 | **0.112** | 0.213 → 0.090 → 0.112 |

## 핵심 발견

### 1. 반전은 없다(유지) — 그러나 "격차가 벌어진다"는 철회한다

TCD의 발견(NME < CNN)이 우리 세팅에 적용된다면 base-heavy로 갈수록 FeCAM 우위가
줄거나 뒤집혀야 한다. **뒤집히지 않았다 — 이건 공정한 예산에서도 그대로다.**

Last acc 기준 Δ(FeCAM − GRU):

| | uniform(원래) | moderate | aggressive |
|---|---|---|---|
| 정정 전 (fixed) | +0.052 | +0.048 | **+0.073** ❌ |
| **정정 후 (scaled)** | **+0.052** | **+0.054** | **+0.045** |

(uniform은 6클래스 세션이라 두 모드가 동일 → 직접 비교 가능한 기준점이다.)

**철회:** *"aggressive에서 격차가 오히려 더 커짐(+0.073), FeCAM이 GRU의 거의 2배"* 는
**baseline을 굶긴 결과였다.** 공정한 예산에서 aggressive 격차는 **+0.045로, 균등
split의 +0.052보다 오히려 작다.** 배수도 1.87배 → **1.40배**로 줄었다.

**유지:** 세 split 전부에서 FeCAM이 앞선다. base-heavy가 순서를 뒤집는다는 TCD식 반례는
우리 세팅에서 **여전히 재현되지 않는다.** 정정된 서사는 *"격차가 커진다"* 가 아니라
**"base 구조를 바꿔도 격차가 대체로 일정하다(+0.045 ~ +0.054)"** 이다.

### 1-b. 굶주림 효과는 aggressive에서만 실재한다 (seed 분산 대조)

| split | fixed last | scaled last | 차이 | pooled sd 대비 |
|---|---|---|---|---|
| moderate | 0.109 ± 0.011 | 0.103 ± 0.020 | −0.005 | **0.3× — 노이즈** |
| aggressive | 0.084 ± 0.007 | 0.112 ± 0.010 | **+0.028** | **3.1× — 실재** |

moderate(4배)에서는 메모리를 4배(50→200) 줘도 **아무 변화가 없다**(오히려 미세하게 낮지만
노이즈 범위). aggressive(6배)에서만 seed 구간이 아예 겹치지 않는다(fixed 0.075~0.090 vs
scaled 0.100~0.119). → **예산 부족이 문제가 되는 임계점이 4배와 6배 사이에 있다.**

### 2. base 직후 붕괴는 실재하지만, 예산으로 설명되는 건 일부뿐

정정 전 이 절은 붕괴를 **전적으로** 고정 예산 탓으로 돌렸다. 예산을 고쳐 재실행하니
그 가설은 **부분적으로만 맞았다:**

| aggressive, base 직후 | 정정 전(fixed) | 정정 후(scaled) |
|---|---|---|
| 0.213 → | 0.075 (**−64.8%**) | 0.090 (**−57.7%**) |

클래스당 exemplar를 1.39 → 8.33개로 **6배** 늘렸는데 붕괴는 −64.8%에서 −57.7%로
**7.1%p 완화되는 데 그쳤다.** 즉 **붕괴의 대부분은 예산이 아니라 다른 원인**이다 —
36클래스를 한 번에 배운 뒤 6클래스를 이어 배울 때의 로짓 쏠림(backprop 계열의 고질적
문제)이 남은 몫으로 보인다. FeCAM이 같은 지점에서 0.183 → 0.164로 완만하기만 한 것과
대비된다.

**그래서 이 절의 원래 결론("A-GEM 구현의 개선점")은 유효하지만 효과는 과대평가였다.**
예산을 클래스 비례로 바꾸는 건 옳고 이제 기본값이지만(`--memory scaled`), 그것만으로
base-heavy 취약성이 해소되지는 않는다.

### 3. FeCAM은 세션 구조에 관계없이 완만하게만 열화됨

FeCAM의 마지막 세션 정확도는 **moderate·aggressive 모두 0.157로 완전히 동일**
(그리고 기존 균등-split 결과 0.157과도 일치) — 클래스별 통계가 독립 누적되므로 세션
순서/크기 구조 자체가 최종 상태에 영향을 주지 않는다는 이론적 성질이 실측으로 확인됨.
per-session 곡선도 완만한 하강(0.193→0.157 등)만 보이고 붕괴가 없음.

## 해석 — TCD의 발견과 우리 발견이 왜 다른가 (원문 확인 완료)

TCD 원문(arXiv:2203.13611)을 직접 읽어 확정함. TCD의 백본은 **ResNet-50 + TSM
(Temporal Shift Module)**, ImageNet 사전학습 가중치로 초기화한 뒤 **CIL 매 스테이지마다
SSv2 영상으로 계속 fine-tune**됨(Kinetics 사전학습은 클래스 정보 누출 우려로 일부러
배제). 즉 TCD의 NME가 계산되는 feature는 **SSv2 모션에 특화되도록 실제로 학습된
feature**다 — 그런 feature를 프레임별로 단순 평균내면 학습된 시간 정보가 파괴된다.

반면 우리는 **frozen CLIP(한 번도 SSv2로 학습되지 않은 appearance-only 표현)**을
쓴다 — 애초에 평균낼 "학습된 시간 정보"가 없으므로 mean-pool의 손실이 작다. 이는
§1(f)/Q6에서 짚은 "SSv2는 모션 중심이라 CLIP에 불리하다"는 한계와 같은 뿌리:
**TCD의 반례는 "SSv2로 학습되어 모션을 담은 feature 위에서의 mean-pool"에 대한
경고**이고, **우리는 애초에 그런 학습된 모션 정보가 없는 지점에서 시작**하므로 그
경고가 구조적으로 덜 적용된다는 설명이 확정 근거로 뒷받침됨.

### TCD 원문 정확한 수치 (Table 2, Something-Something V2)

split: **84-class base + [10개×9 session] 또는 [5개×18 session]**(174클래스 전체),
클래스 순서 3-seed(1000/1993/2021) 셔플 평균, exemplar 클래스당 20개.

| Method | 10-group CNN | 10-group NME | 5-group CNN | 5-group NME |
|---|---|---|---|---|
| UCIR | 26.84 | 17.98 | 20.69 | 12.57 |
| PODNet | 34.94 | 27.33 | 26.95 | 17.49 |
| **TCD** | **35.78** | **28.88** | **29.60** | **21.63** |

NME<CNN 격차: **−6.90%p**(10-group), **−7.97%p**(5-group). 논문 원문 인용:
> "It is noticeable that, for Something-Something V2, the performance for NME
> falls behind CNN. Since Something-Something V2 needs more temporal reasoning,
> the strategies relying on naïve averaging of the features from all frames may
> not be suitable."

(참고: CSTA(41.26%)·STSP(69.68%) 등 후속 연구 수치는 TCD 원 논문 수치가 아니라
**이후 개선된 후속 방법의 재구현/향상 결과** — TCD 자신의 보고치는 위 35.78%다.)

## 결론

1. **우리 핵심 주장(FeCAM > GRU+A-GEM)은 base-heavy 스트레스 테스트를 통과함** —
   공정한 replay 예산에서도 세 split 전부 FeCAM이 앞선다. TCD의 반례는 우리 세팅에
   직접 적용되지 않는다.
2. **단, 격차는 "커지는" 게 아니라 "일정한" 것이다**(+0.045 ~ +0.054). 정정 전의
   "aggressive에서 +0.073으로 벌어진다"는 baseline을 굶긴 결과였고 **철회한다.**
3. **A-GEM의 고정 replay 예산은 실제로 약점이지만 base-heavy 취약성의 일부만 설명한다**
   — 예산을 6배 늘려도 붕괴가 −64.8%에서 −57.7%로만 완화됐다. 나머지는 backprop 계열의
   로짓 쏠림으로 보이며 예산으로 해결되지 않는다.
4. TCD와의 차이는 backbone(frozen CLIP vs trained CNN)에서 올 가능성이 높다는 가설은
   유지 — 완전한 반증엔 TCD의 실제 174-class split·backbone 재현이 필요
   (영상 접근 차단, `reports/mobileclip_result.md`와 동일 사유).

**방법론 교훈:** 연구 대상 변수(base 크기)가 baseline의 하이퍼파라미터(replay 예산)를
동시에 바꾸고 있었다. 비교 실험을 설계할 때 **"이 변수가 baseline에게 불리하게 작용하지
않는가"** 를 먼저 점검해야 한다. 같은 실수를 UCF101 러너에서는 피했는데
([`run_ucf101_gru_agem.py:112`](../dev/run_ucf101_gru_agem.py)가 클래스 비례로 예산을
잡는다) 이 스크립트에는 반영되지 않았었다.

## 재현

```bash
python3 dev/run_base_heavy_split.py                  # scaled(공정) + fixed(대조) 둘 다
```

```bash
python3 dev/run_base_heavy_split.py --memory scaled  # 공정 예산만
```
Raw per-seed 결과: `reports/base_heavy_split_raw.json`.
