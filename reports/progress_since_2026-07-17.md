# 진행 사항 — 2026-07-17 이후

Generated: 2026-07-26. 범위: `e364e2c`(07-16, 직전 작업) **이후 12커밋**
(`ea6cb29`…`6aa3c3a`). 23개 파일, +2,078 / −31 줄.
전체 맥락은 [`reports/meeting_prep_todo_status.md`](meeting_prep_todo_status.md),
프로젝트 프레이밍은 [`docs/project_focus.md`](../docs/project_focus.md) 참조.

---

## 한눈에

| # | 작업 | 결과 | 산출물 |
|---|---|---|---|
| 1 | **PyCIL 표준 벤치마크 구축 + 브리지** | 우리 head 서열(FeCAM>SLDA>NCM)이 CIFAR-100에서 그대로 재현 | `benchmarks/pycil/`, `pycil_bridge_result.md` |
| 2 | **TCD 반례 재검증 실험** | 반전 없음, 오히려 격차 확대(+0.052→+0.073) | `dev/run_base_heavy_split.py`, `base_heavy_split_result.md` |
| 3 | **SSv2-CIL 논문 4편 원문 정독** | "SSv2로 CIL이 새롭다"는 주장 반증, 인용 필수 계보 확정 | `sota_positioning_brief.md` §1(f) |
| 4 | **edge-CL 갭 감사(3편)** | 갭이 부분적으로만 열림 → **주장 축소** | `pycil_survey_edge_gap.md` |
| 5 | **FLOPs 회계(학습+추론)** | FeCAM이 GRU 대비 학습 1,628배 적은 연산 | `dev/compute_flops.py`, `flops_result.md` |
| 6 | **FeCAM 채점 최적화 + 재배포** | 248배 가속, 출력 불변, HF Space 반영 | `src/models/fecam_head.py` |

**정정 2건 발생** — 아래 §정정 참조. 테스트 68 → **70 passed**.

---

## 1. PyCIL 표준 벤치마크 구축 + 브리지 실험 (07-22)

**동기:** "우리 결론이 자체 SSv2 벤치에서만 성립하는 것 아니냐"는 반론 대비.

- `external/PyCIL` 체크아웃(pin `f3509b8`) + **비-CUDA 패치 3종**을
  `setup_pycil.sh` 한 번으로 재현 가능하게 자동화(MPS device, `.cuda()` → `self._device`,
  전역 `cuda_shim`)
- CIFAR-100 스모크(SimpleCIL, b0=50 inc=10): **6/6 태스크 완주**
- ImageNet-100/1000 스캐폴드(데이터는 라이선스상 수동)

**브리지 결과** — PyCIL의 **동일 스플릿**(클래스 순서를 PyCIL `DataManager`에서 직접
import, 비트 단위 일치 검증) 위에서 우리 analytic head를 frozen CLIP feature로 실행:

| head | Avg Inc Acc | Last Acc |
|---|---|---|
| NCM prototype | 0.698 | 0.660 |
| Deep SLDA | 0.725 | 0.686 |
| **FeCAM (shared cov)** | **0.769** | **0.733** |

**핵심:** 서열 **FeCAM > SLDA > NCM**이 우리 SSv2 결과(0.157 > 0.141 > 0.124)와 **완전히
일치** → 자체 벤치 특유 현상이 아님을 교차 검증.

**주의(논문에 명기):** 우리는 frozen 사전학습 CLIP = **PTM 트랙**, PyCIL 고전(iCaRL/DER)은
**from-scratch 트랙**. 한 표에서 우열 비교 금지, 트랙 라벨 병기 필수.

---

## 2. TCD 반례 재검증 실험 (07-24)

**문제:** TCD(ICCV'21)가 **SSv2에서 NME(우리 FeCAM과 같은 정신)가 CNN보다 못하다**고 보고
— 우리 핵심 결론과 정면 충돌 가능.

원본 174-class split은 영상 접근이 막혀 재현 불가 → **구조적 변수(base-heaviness)를
우리 48클래스에 이식**해 true class-IL로 재평가:

| split | 구성 | FeCAM last | GRU+A-GEM last | Δ |
|---|---|---|---|---|
| 균등(기존) | 6×8 | 0.157 | 0.105 | +0.052 |
| moderate | 24 + 6×4 (4배) | 0.157 | 0.109 | +0.048 |
| aggressive | 36 + 6×2 (6배) | 0.157 | **0.084** | **+0.073** |

**반전이 일어나지 않았고 오히려 격차가 벌어짐.**

**부수 발견 (별개 가치):** GRU+A-GEM이 base 세션 직후 **−65% 급락**(aggressive).
원인은 A-GEM의 replay 메모리가 **stage당 고정 50개**라, base 클래스가 많을수록 클래스당
exemplar가 급감(8.33 → **1.39개/class**)하기 때문. FeCAM 우위와 무관한 **우리 A-GEM
구현 자체의 개선 포인트**.

---

## 3. SSv2-CIL 논문 4편 원문 정독 (07-24 ~ 07-25)

검색 요약이 아니라 **PDF 원문을 직접 읽어** 확정. 결과: **"SSv2로 CIL 하는 게 새롭다"는
주장은 틀림** — TCD가 이 벤치마크의 원조이고 Related Work에 반드시 인용해야 함.

| 논문 | 발표 | exemplar | 백본 | SSv2 (10×9 / 5×18) |
|---|---|---|---|---|
| **TCD** | ICCV 2021 | 20/class | ResNet-50+TSM, SSv2로 fine-tune | CNN **35.78/29.60**, NME 28.88/21.63 |
| **STSP** | ECCV 2024 | **0** | ResNet-50+TSM + null-space gradient 투영 | **69.68/70.87** (최고) |
| **CSTA** | TCSVT 투고'25 | 0*/5 | TimeSformer + 공간·시간 분리 어댑터 + causal loss | 41.26 |
| **ESSENTIAL** | ICCV 2025 Highlight | sparse + prompt | **frozen CLIP** + temporal encoder | 48.9/47.5 (메모리 8.4 MiB) |

**확정된 사실 2가지:**

1. **TCD의 NME 열세 원인이 원문으로 확정** — 백본이 ImageNet에서 시작해 **CIL 내내 SSv2로
   계속 fine-tune**되므로, NME가 계산되는 feature 자체가 **학습되어 모션 정보를 담은**
   feature다. 그걸 프레임 평균내면 정보가 파괴된다. 우리는 **한 번도 SSv2로 학습되지 않은
   frozen CLIP**이라 애초에 평균내 잃을 시간 정보가 없다 → **경고가 구조적으로 덜 적용됨.**
2. **TCD·STSP·CSTA 셋 다 백본을 SSv2로 학습**한다(ESSENTIAL만 예외로 frozen CLIP 사용).
   즉 이 계보 전체와 우리를 가르는 축이 일관되게 성립.

**검증 태도:** STSP의 69.68은 TCD 대비 거의 2배라 이례적 → **제3의 독립 소스**(ESSENTIAL
Table 2가 69.7/70.9로 인용)로 교차 확인 후 채택. CSTA의 "exemplar-free"는 fine-tuning
단계에서 클래스당 5샘플을 쓴다고 원문에 명시돼 있어 caveat으로 기록.

---

## 4. edge-CL 갭 감사 — 주장을 축소함 (07-25)

우리 edge 프레이밍이 이미 다뤄진 주제인지 선행연구 3편으로 확인.

**(a) 표준 서베이** (Zhou et al., TPAMI'24 — PyCIL 저자 그룹, 38쪽 전문 키워드 전수조사):

| 키워드 | 등장 |
|---|---|
| **CPU** | **0회** |
| FLOPs / 전력 | **0회** |
| edge (실제 의미) | 8회 (74회 중 66회는 "knowl**edge**") |
| budget | 81회 ← 실제 알맹이 |

→ edge를 **메모리 바이트 문제로만** 다루고, 유일한 시간 측정도 **3090 GPU**. 연산 축은
스스로 "미래 과제"로 넘김.

**(b) 그러나 서베이가 지목한 선행연구 2편이 갭을 부분적으로 닫음:**

- **SparCL (NeurIPS'22)** — **갤럭시 S20 / Snapdragon 865 CPU에서 학습 가속 실측**
  (3.1×/2.3×, DER++ 대비 23× 적은 training FLOPs). 단 **backprop을 없애는 게 아니라
  희소화**하고, ResNet-18을 from scratch로, **이미지**(CIFAR/Tiny-ImageNet)에서.
- **BudgetCL (CVPR'23)** — 연산 예산 고정 CL을 대규모로(A100, 1500 GPU-hours). 단 예산을
  **iteration 수로 정의해 하드웨어를 의도적으로 추상화**.

### ⚠️ 결론: 쓸 수 있는 문장이 바뀜

| 주장 | 판정 |
|---|---|
| "edge/CPU CL 학습 비용을 측정한 연구가 없다" | ❌ **못 씀** (SparCL이 반박) |
| "**비디오** CIL에 CPU·지연·전력 측정이 없다" | ✅ 사실 (TCD 계보 4편 전부 미측정) |
| "backprop을 **제거**한 edge CL 비교가 없다" | ✅ 사실 (SparCL은 희소화) |
| "절대 CPU wall-clock을 보고한 연구가 없다" | ✅ 사실 (셋 다 상대 배수 또는 추상 지표) |

→ **우리 기여 = "비디오 × backprop-free × 절대 CPU 시간"의 교집합.** 재료는 각각
선행연구가 있으나 이 조합은 비어 있음.

**보너스 — 우리 결론의 독립 뒷받침:** BudgetCL이 대규모 이미지 실험에서 내린 결론이
우리 발견과 같은 방향이다: ① 연산 제약 시 **어떤 CL 알고리즘도 Naive를 못 이김**,
② 제약이 심할수록 격차 확대, ③ **frozen 사전학습 + 최소 학습이 갭을 메움**.

---

## 5. FLOPs 회계 — 학습 + 추론 (07-26)

**동기:** 지금까지 효율 주장이 wall-clock뿐이라 하드웨어·구현에 의존 → edge-CL 문헌
옆에 놓을 수 없음. SparCL·BudgetCL 모두 FLOPs를 쓰므로 같은 단위로 표현.

**규약 명시:** 1 MAC = 2 FLOPs, training = forward + backward(=3×forward).
analytic head는 `nn.Module`이 아니라 numpy 통계라 어떤 프로파일러로도 추적 불가 →
전부 손으로 유도. **GRU 파라미터 수(603,696)를 실제 모델과 `assert`로 대조**해 검증.

| 방법 | training FLOPs | vs GRU | 추론/윈도우 |
|---|---|---|---|
| GRU + A-GEM | 4.64 T | 1× | 18.9 M |
| **FeCAM** | **2.85 G** | **1,628×** | **636 K** |
| NCM prototype | 44 M | 104,791× | 59 K |

**세부 발견:**

- **A-GEM 자체 오버헤드는 전체의 11.8%뿐** — GRU가 비싼 건 A-GEM이 아니라 **backprop과
  15 epoch 반복** 때문.
- **FLOPs와 wall-clock은 다른 걸 잡아낸다** — Deep SLDA는 FeCAM보다 FLOPs가 2.3배인데
  wall-clock은 **44배**(per-sample 스트리밍 구현이라 BLAS 배칭 없음). **둘 다 보고해야 함.**
- **인코더가 모든 걸 압도** — frozen CLIP은 윈도우당 **141 GFLOPs**, 학습셋 전체 677 TFLOPs로
  **GRU+A-GEM 전체 학습의 146배**. 추론에선 **99.98% 이상**을 차지.
  → head 선택은 **정확도로** 하면 되고, 실시간 성능은 **인코더를 건드려야** 개선됨
  (MobileCLIP 시도가 CPU에서 실패한 건 별개 문제).

---

## 6. FeCAM 채점 최적화 + 재배포 (07-26)

`scores()`가 클래스별로 Python 루프를 돌며 `einsum`을 n=1로 호출 → K·D² FLOPs.
Mahalanobis 이차형식을 전개하면 `(x−m)'P(x−m) = x'Px − x'Pm − m'Px + m'Pm`이고
**m에만 의존하는 항은 캐시 가능** → 윈도우당 **D² + 2KD**로 하락.

| | 이전 | 현재 | 배수 |
|---|---|---|---|
| `predict_window` (데모 실시간) | 14.4 ms | **0.058 ms** | **248배** |
| 배치 채점 4,702개 | ~8,200 ms | **67 ms** | **122배** |
| FLOPs/윈도우 | 25.3 M | **636 K** | **40배** |

**출력 불변:** 이전 구현 대비 최대 상대오차 **6e-15**, **argmax 100% 일치**, 정확도 4개
지표 모두 마지막 자리까지 동일(task-aware 0.410 / full-48 0.157 / F1 0.144 / mAP 0.120).

교차항 2개를 `2x'Pm`으로 합치지 않고 각각 유지해, P가 완벽히 대칭이 아니어도 정확하도록 함.
회귀 테스트 2개 추가(교과서적 per-class 계산과 직접 대조 + 신규 등록 시 캐시 무효화).

**배포:** HF Space 반영(`81a0409..fafcb0d`), `/health`·`/predict_rt`로 3-way 응답 확인.

**부수 결과:** 이 최적화로 **FeCAM 추론이 GRU보다 30배 싸짐**(636 K vs 18.9 M).
최적화 전엔 오히려 1.33배 비쌌으므로, "analytic head는 학습에서만 유리"라는 중간 결론이
**"학습·추론 양쪽 모두 유리"로 뒤집힘**.

---

## ⚠️ 정정 사항 (이전 보고 수치 수정)

정직성을 위해 명시. **정확도 수치는 전부 영향 없음.**

### (1) 학습 시간이 잘못 귀속돼 있었음

러너 타이머(`dev/run_modern_cpu_methods.py:186-194`)가 stage 1의 `model.scores(Xva)`
**평가 호출을 타이머 안에 포함**하고 있었음.

| | 기존 보고 | 실제 fit only | 평가 비중 |
|---|---|---|---|
| **FeCAM** | 8.7 s | **42.4 ms** | **99.5%** |
| Deep SLDA | 2.1 s | 1.86 s | 11% |
| RanDumb | 0.9 s | 532 ms | 41% |

→ "FeCAM이 GRU보다 30배 빠름"은 **과소평가**였고, fit-only 기준 **약 6,400배**.
관련 리포트 전부 정정 완료.

### (2) 내 이전 분석의 오독 정정

07-25에 "ESSENTIAL이 TCD 수치를 원문과 다르게 인용한다"고 문제 제기했으나, **Table 2를
전부 읽어보니 오해**였음 — 그 표엔 `TCD(ViT)` 행(공정 재실행)과 `TCD(TSM)` 행(원문과
정확히 일치)이 **둘 다** 있었음. ESSENTIAL은 투명하게 병기한 것. 교훈을
"제3자 인용을 불신하라" → **"표를 볼 땐 백본 열까지 확인하라"**로 수정.

---

## 산출물

**신규 파일 (16개)**

| 경로 | 내용 |
|---|---|
| `benchmarks/pycil/` (9파일) | PyCIL 셋업·패치·config·브리지 스크립트 |
| `dev/run_base_heavy_split.py` | TCD 스타일 base-heavy 재검증 실험 |
| `dev/compute_flops.py` | 학습·추론 FLOPs 회계 |
| `reports/pycil_bridge_result.md` | CIFAR-100 교차검증 결과 |
| `reports/base_heavy_split_result.md` | TCD 반례 재검증 결과 |
| `reports/pycil_survey_edge_gap.md` | edge-CL 선행연구 3편 감사 |
| `reports/flops_result.md` | FLOPs 회계 + 발견 6가지 |
| `reports/base_heavy_split_raw.json` | seed별 원시 결과 |

**수정 파일:** `src/models/fecam_head.py`(최적화), `tests/test_fecam_head.py`(회귀 테스트 2개),
`reports/sota_positioning_brief.md`(§1(f) 문헌 지형 + Q11 신설 + §4 갱신),
`reports/cpu_friendly_methods_result.md`·`reports/meeting_prep_todo_status.md`(수치 정정),
`docs/project_focus.md`(갭 목록 갱신)

**커밋 12개**

```text
2026-07-22  ea6cb29  feat: PyCIL benchmark setup (CIFAR-100 + ImageNet) + analytic-head bridge
2026-07-24  f5d4c39  study: TCD base-heavy split re-validation — FeCAM lead holds, widens
2026-07-24  aaa5374  docs: refresh commit-history block with actual base-heavy-split hash
2026-07-24  a92cdd2  docs: verify TCD (ICCV'21) primary source, upgrade hypothesis to confirmed fact
2026-07-25  2cffe81  docs: verify CSTA and STSP primary sources, build SSv2-CIL landscape table
2026-07-25  285dc5e  docs: cross-verify STSP SSv2 numbers via ESSENTIAL, flag cross-paper TCD discrepancy
2026-07-25  0986f02  docs: verify ESSENTIAL primary source, correct prior TCD-citation misreading
2026-07-25  1b2383a  docs: audit PyCIL-group survey for CPU/resource-constrained coverage
2026-07-25  a0693d2  docs: read SparCL + BudgetCL primary sources, narrow the edge-CL gap claim
2026-07-26  c59730b  feat: report training FLOPs alongside wall-clock; fix misattributed train times
2026-07-26  fa858c1  feat: add inference FLOPs under the same convention; rename to compute_flops
2026-07-26  6aa3c3a  perf: vectorize FeCAM scoring -- 40x fewer FLOPs, 248x faster, identical output
```

PR: [#2](https://github.com/STkangyh/coad-mini/pull/2) (브랜치 `feature/pycil-benchmarks`)

---

## 다음 단계 (미착수, 우선순위 순)

1. **실제 embedded 하드웨어 실측** — 현재는 Mac CPU. Jetson Orin Nano / Raspberry Pi급에서
   학습·추론 시간과 전력을 재면 논문 설득력이 크게 오름. (갭 목록 1번)
2. **iteration-budget 프로토콜 병기**(BudgetCL 방식) — "step당 N iteration" 형식을 추가하면
   그들 대규모 결과와 직접 비교 가능. (갭 9번)
3. **AUC-A / AUC-L 지표 채택**(서베이 방식) — 예산별 성능 곡선의 면적. `benchmarks/pycil/`에
   동일 스플릿이 이미 있어 적용 비용 낮음. **여기에 연산 축을 추가하는 게 우리 차별점.** (갭 7번)
4. **원본 SSv2 영상 확보** — 막히면 TCD 174-class 원본 재현과 MobileCLIP 다운스트림 정확도
   비교가 계속 불가.
5. SLDA/Ridge/RanDumb에 precision 캐시 추가 — 데모 투입 시 선결 조건(현재 데모는 FeCAM만 사용).
