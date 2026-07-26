# 진행 사항 — 2026-07-20 이후 (미팅용 종합)

Generated: 2026-07-27. 범위: **21커밋**(`ea6cb29` 07-22 … `19c41f3` 07-27),
37개 파일, +4,266 / −31 줄. 발표용 스크립트는
[`reports/meeting_script_0726.md`](meeting_script_0726.md).

---

## 한눈에

| # | 작업 | 결과 |
|---|---|---|
| **1** | **UCF101 표준 벤치마크 진입** ⭐ | **FeCAM 88.84 — 공개 방법 중 2위**(ESSENTIAL만 위) |
| **2** | **GRU+A-GEM과 직접 비교**(같은 벤치) | **+31.45%p 우위**, 학습은 1,600배 빠름 |
| 3 | PyCIL 표준 벤치 구축 + CIFAR-100 브리지 | 서열(FeCAM>SLDA>NCM)이 다른 데이터셋에서 재현 |
| 4 | TCD 반례 재검증(base-heavy) | 반전 없음, 격차 오히려 확대 |
| 5 | SSv2-CIL 논문 4편 원문 정독 | 인용 필수 계보 확정, 우리 위치 규정 |
| 6 | edge-CL 갭 감사(3편) | **주장을 축소함** — 정직성 확보 |
| 7 | FLOPs 회계 + FeCAM 최적화 | 연산량 1,628배 적음 / 추론 248배 가속 |

**그림 3장**: [`reports/figures/`](figures/) — 미팅에 그대로 사용 가능(200dpi PNG).

**정정 2건 발생**(§정정). 테스트 68 → **70 passed**.

---

## 1. ⭐ UCF101 표준 벤치마크 — 우리 숫자를 남의 표에 올림

**왜 중요한가:** 지금까지 우리 수치는 전부 48-class SSv2 **자체 subset**이라, 남의 논문과
"서열"만 논할 수 있었고 숫자를 나란히 놓을 수 없었다. UCF101은 우리가 원문을 읽은
4편(TCD·STSP·CSTA·ESSENTIAL)이 **전부 같은 프로토콜로 보고**하는 벤치마크다.

### 프로토콜이 문자 그대로 일치함 (추정 아님, 검증함)

TCD 공개 저장소에서 확인:
- `class_list.pkl`이 `np.random.RandomState(1000).permutation(101)`과 **101개 전부 일치**
  → 우리가 쓴 클래스 순서 = **TCD가 실제로 쓴 순서**
- `ucf101_51_10.sh`가 `--init_task 51 --nb_class 10 --K 5 --budget_type class` 확인
  → base 51, 증분 10, **exemplar 클래스당 5개**

### 결과 (UCF101, 10×5 stages)

| 방법 | 백본 | exemplar | 정확도 |
|---|---|---|---|
| TCD (ICCV'21) | ResNet-34+TSM | ✔ | 74.89 |
| FrameMaker (NeurIPS'22) | TSM | ✔ | 78.13 |
| STSP (ECCV'24) | TSM | 0 | 81.15 |
| ST-prompt | CLIP | ✕ | 84.8 |
| **우리 (frozen CLIP + FeCAM)** | **frozen CLIP** | **0** | **88.84** |
| ESSENTIAL (ICCV'25 Highlight) | frozen CLIP | sparse+prompt | 95.1 |

**ESSENTIAL 다음 2위.** TCD(+14.0) · FrameMaker(+10.7) · STSP(+7.7) · ST-prompt(+4.0)를
모두 상회 — **backprop 0회, exemplar 0개, 백본 학습 0회, fit 0.3초**로.

> 📊 그림: `figures/ucf101_tcd_curves.png`

---

## 2. ⭐ 우리 자체 baseline(GRU+A-GEM)과의 직접 비교

같은 프로토콜, 같은 클래스 순서, exemplar는 TCD와 동일하게 **클래스당 5개**.

| 세션 | 0 (51cls) | 1 (61) | 2 (71) | 3 (81) | 4 (91) | 5 (101) |
|---|---|---|---|---|---|---|
| GRU + A-GEM | 84.6 | **60.1** | 49.0 | 49.3 | 49.0 | 52.3 |
| **FeCAM** | 90.1 | **90.1** | 88.9 | 88.5 | 88.1 | 87.4 |

| | avg inc | last | 학습시간 |
|---|---|---|---|
| GRU + A-GEM | 57.39±0.16 | 52.30 | 480 s/seed |
| **FeCAM** | **88.84±0.52** | **87.44** | **0.3 s** |
| 차이 | **+31.45 %p** | **+35.14 %p** | **1,600배** |

**문제는 표현력이 아니라 망각이다.** GRU는 base 51클래스를 84.6%로 잘 배운다(FeCAM과
5.6%p 차이). 그런데 **첫 증분에서 −24.5%p 붕괴**하고 50% 근처에 정체한다. FeCAM은
같은 지점에서 **−0.08%p**다.

> **정직하게 병기:** 우리 GRU+A-GEM 57.39는 위 표의 **모든 공개 방법보다 낮다**
> (최하위 iCaRL 70.6). 당연하다 — 저들은 비디오 CIL 전용 설계, 우리는 단순 GRU+A-GEM.
> 주장은 "우리 baseline이 강하다"가 아니라 **"같은 frozen feature 위에서 backprop을
> 없애는 쪽이 훨씬 낫다"**이며, 이 비교가 그걸 보여준다.

---

## 3. PyCIL 표준 벤치마크 구축 + CIFAR-100 브리지

- `external/PyCIL`(pin `f3509b8`) + **비-CUDA 패치**를 `setup_pycil.sh` 한 번으로 재현
- **브리지**: PyCIL의 **동일 스플릿**(클래스 순서를 `DataManager`에서 직접 import,
  비트 단위 일치) 위에서 우리 head 실행

| head | Avg Inc Acc | Last |
|---|---|---|
| NCM | 0.698 | 0.660 |
| Deep SLDA | 0.725 | 0.686 |
| **FeCAM** | **0.769** | **0.733** |

서열 **FeCAM > SLDA > NCM**이 SSv2 결과와 완전히 일치 → 자체 벤치 특유 현상이 아님.

**곁다리 수확:** PyCIL 자체 실행이 **마지막에 크래시**하는 상류 버그를 발견·수정했다
(정확도 행렬을 `[task+1, task+1]`로 잡는데 `init_cls != increment`면 그룹 수와 태스크 수가
달라짐 — 우리 b0=50/inc=10은 6태스크 × 10그룹). 결과값 자체엔 영향 없었지만 터미널에서
돌리면 에러를 보게 되므로 `setup_pycil.sh`에 패치로 넣었다.

> 📊 그림: `figures/cifar100_bridge.png`

---

## 4. TCD 반례 재검증 (base-heavy)

TCD가 **SSv2에서 NME(우리 FeCAM과 같은 정신)가 CNN보다 못하다**고 보고 → 우리 결론과
정면 충돌 가능. 우리 48클래스를 TCD 스타일 base-heavy로 재편해 검증:

| split | FeCAM last | GRU+A-GEM last | Δ |
|---|---|---|---|
| 균등(기존) | 0.157 | 0.105 | +0.052 |
| moderate(4배) | 0.157 | 0.109 | +0.048 |
| aggressive(6배) | 0.157 | **0.084** | **+0.073** |

**반전 없음, 격차 확대.** 부수 발견: A-GEM이 base 직후 −65% 붕괴 — replay 메모리가
stage당 고정 50개라 클래스당 exemplar가 8.33→1.39개로 급감하기 때문(우리 구현의 개선점).

---

## 5. SSv2-CIL 논문 4편 원문 정독

검색 요약이 아니라 **PDF 원문**으로 확인. **"SSv2로 CIL 하는 게 새롭다"는 주장은 틀림** —
TCD가 원조이고 Related Work에 반드시 인용해야 함.

**확정된 사실:** TCD·STSP·CSTA는 **셋 다 백본을 SSv2로 학습**한다(ESSENTIAL만 frozen CLIP).
그래서 그들의 feature엔 모션 정보가 들어있고, 프레임 평균이 그걸 파괴한다.
**우리는 한 번도 SSv2로 학습되지 않은 frozen CLIP**이라 애초에 평균내 잃을 시간 정보가 없다
→ TCD의 경고가 구조적으로 덜 적용됨.

---

## 6. edge-CL 갭 감사 — **주장을 축소함**

우리 edge 프레이밍이 이미 다뤄진 주제인지 선행연구 3편으로 확인.

- **표준 서베이**(Zhou et al., TPAMI'24, PyCIL 저자 그룹): edge 배포를 동기로 내걸지만
  **38쪽에 "CPU" 0회**, FLOPs 0, 전력 0. 유일한 시간 측정도 3090 GPU.
- **그러나 SparCL(NeurIPS'22)이 갤럭시 S20 CPU에서 학습 가속을 실측**(3.1×) →
  ❌ "edge/CPU CL 측정이 없다"는 **못 씀**.
- **BudgetCL(CVPR'23)**: 연산 예산 고정 CL을 대규모로. 단 예산을 iteration 수로 정의해
  **하드웨어를 의도적으로 추상화**.

### 살아남는 주장 (= 우리 기여)

| 주장 | 판정 |
|---|---|
| "edge/CPU CL 측정이 없다" | ❌ 못 씀 |
| "**비디오** CIL에 CPU·지연·전력 측정이 없다" | ✅ 사실 (4편 전부 미측정) |
| "backprop을 **제거**한 edge CL 비교가 없다" | ✅ 사실 (SparCL은 희소화) |
| "절대 CPU wall-clock을 보고한 연구가 없다" | ✅ 사실 |

→ **"비디오 × backprop-free × 절대 CPU 시간"의 교집합.**

**보너스:** BudgetCL의 결론이 우리와 같은 방향 — ① 연산 제약 시 어떤 CL도 Naive를 못 이김,
② 제약이 심할수록 격차 확대, ③ **frozen 사전학습 + 최소 학습이 갭을 메움**.

---

## 7. FLOPs 회계 + FeCAM 최적화

wall-clock은 하드웨어·구현 의존이라 문헌 옆에 못 놓는다. SparCL·BudgetCL이 쓰는
**FLOPs**로 같은 단위를 맞췄다(1 MAC = 2 FLOPs, train = 3×forward).

| | training FLOPs | vs GRU | 추론/윈도우 |
|---|---|---|---|
| GRU + A-GEM | 4.64 T | 1× | 18.9 M |
| **FeCAM** | **2.85 G** | **1,628×** | **636 K** |

**FeCAM 채점 최적화(적용·배포 완료):** Mahalanobis 이차형식을 전개하고 평균 의존 항을
캐시 → **윈도우당 14.4 ms → 0.058 ms (248배)**, FLOPs 40배 감소, **출력은 완전 동일**
(최대 상대오차 6e-15, argmax 100% 일치). 회귀 테스트 2개 추가, HF Space 재배포 검증.

**인코더가 지배한다:** frozen CLIP이 윈도우당 141 GFLOPs로 **학습 전체의 146배**,
추론의 **99.98%**. → head는 정확도로 고르면 되고, **실시간 성능은 인코더를 건드려야** 개선됨.

---

## 발견 — 증분 크기에 완전히 불변 (구조적 성질)

| 방법 | 10×5 → 2×25 변화 |
|---|---|
| TCD | −2.70 |
| FrameMaker | −2.36 |
| STSP | −1.90 |
| ESSENTIAL | −1.80 |
| **우리 (FeCAM)** | **0.00** |

클래스별 통계가 **독립 누적**이라 세션 경계가 무의미하다. **edge 서사에 직결** —
스마트글래스에서 **한 번에 한 클래스씩** 등록해도 페널티가 없다.

> 📊 그림: `figures/increment_sensitivity.png`

---

## 발견 — SSv2와 UCF101의 대비가 적용 범위를 규정한다

| 벤치마크 | 성격 | 우리 FeCAM |
|---|---|---|
| **UCF101** | **static-biased** (appearance로 구별) | **88.84** |
| SSv2 subset | **temporal-biased** (모션으로만 구별) | 15.7 |

"static/temporal-biased"는 **ESSENTIAL 논문이 자기 §4.1에서 쓰는 용어**다.
즉 **우리 방법의 성패가 데이터셋 성격으로 정확히 예측된다.** 약점 고백이 아니라
**적용 조건의 명시**이고, TCD의 SSv2 NME 열세 보고와도 완전히 일관된다.

논문 문장: *"appearance-discriminative한 행동에 대해서는 학습 없는 통계 head가
무거운 학습 방법을 능가하며, motion-discriminative한 경우에는 그렇지 않다.
우리는 그 경계를 정량화한다."*

---

## ⚠️ 정정 사항 (이전 보고 수치 수정)

**정확도 수치는 전부 영향 없음.**

**(1) 학습 시간이 잘못 귀속돼 있었음** — 러너 타이머 안에 평가 호출이 포함돼 있었다.

| | 기존 보고 | 실제 fit only | 평가 비중 |
|---|---|---|---|
| **FeCAM** | 8.7 s | **42.4 ms** | **99.5%** |
| Deep SLDA | 2.1 s | 1.86 s | 11% |

→ "FeCAM이 30배 빠름"은 **과소평가**였고 fit-only 기준 **약 6,400배**.

**(2) 내 이전 분석의 오독** — "ESSENTIAL이 TCD를 원문과 다르게 인용한다"고 문제 제기했으나,
Table 2를 전부 읽어보니 `TCD(ViT)`(공정 재실행)와 `TCD(TSM)`(원문 일치) **두 행이 다 있었다**.
투명하게 병기한 것이었고, 교훈을 "제3자 인용 불신" → **"표의 백본 열까지 확인"**으로 수정.

---

## 정직한 caveat (논문에 병기할 것)

1. **트랙이 다르다** — 우리는 백본을 전혀 학습하지 않는다. 위 표는 위치 파악용이지
   동일 조건 순위가 아니다.
2. **CLIP 사전학습 오염 가능성** — UCF101은 YouTube 기반, CLIP은 웹 스케일 학습.
   PTM 트랙 공통 caveat.
3. **ESSENTIAL도 같은 frozen CLIP으로 95.1** — 6.3%p 격차는 백본이 아니라 그들의
   **학습형 temporal encoder + memory retrieval**에서 온다. 개선 여지가 그만큼 있다는 뜻.
4. GRU 비교는 10×5 열만 수행(seed당 480초). 증분을 줄이면 GRU가 더 불리해지므로
   **보수적인 선택**이다.
5. **HMDB51 보류** — 영상은 구할 수 있으나 **공식 split을 구할 수 없음**(serre-lab URL이
   HTML만 반환, HF 미러는 영상만, TCD 저장소에도 없음). 자체 분할은 비교 가능성이라는
   목적 자체를 없애므로 중단.

---

## 다음 단계 (우선순위)

1. **실제 임베디드 보드 실측**(Jetson Orin Nano / 라즈베리파이) — 지금은 맥북 CPU.
   "edge"를 주장하는데 측정이 노트북인 게 가장 큰 취약점.
2. **iteration-budget 프로토콜 병기**(BudgetCL 방식) — 그들 대규모 결과와 직접 비교 가능.
3. **AUC-A/AUC-L 지표 채택**(서베이 방식) + **연산 축 추가** ← 우리 차별점.
4. **HMDB51** — split 확보 시 즉시 재개(파이프라인은 UCF101 것 재사용).
5. few-shot enrollment 정량화 — 교수님 체크리스트에서 유일하게 남은 항목.

---

## 산출물

**신규 스크립트**: `benchmarks/pycil/`(9), `scripts/get_ucf101.sh`,
`scripts/extract_ucf101_features.py`, `dev/run_base_heavy_split.py`,
`dev/run_ucf101_tcd_protocol.py`, `dev/run_ucf101_gru_agem.py`,
`dev/compute_flops.py`, `dev/plot_cil_results.py`

**신규 리포트**: `ucf101_tcd_result.md` · `base_heavy_split_result.md` ·
`pycil_bridge_result.md` · `pycil_survey_edge_gap.md` · `flops_result.md` ·
`meeting_script_0726.md` · 본 문서

**그림**: `figures/ucf101_tcd_curves.png` · `figures/cifar100_bridge.png` ·
`figures/increment_sensitivity.png`

PR: [#2](https://github.com/STkangyh/coad-mini/pull/2) (`feature/pycil-benchmarks`)
