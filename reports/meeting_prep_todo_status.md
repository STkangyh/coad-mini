# 교수님 미팅 준비 — TODO 진행 상황


> ⚠️ **2026-07-31 정정**: 이 문서의 base-heavy 결론("격차가 **+0.073**으로 벌어짐", "FeCAM이 GRU의 거의 2배")은 **철회됐다.** A-GEM replay가 세션당 고정 50개여서 36클래스 base가 클래스당 1.39개만 받은 탓이다. 공정한 예산에서는 **+0.045**이며, "반전 없음"이라는 핵심 결론만 유지된다 → [`base_heavy_split_result.md`](base_heavy_split_result.md)

Generated: 2026-07 (auto). 교수님이 주신 체크리스트 3개 + 파생 작업 1개의 완료 현황 정리.
관련 코드/리포트는 각 항목에 링크.

## 요약

| # | TODO | 상태 |
|---|---|---|
| 1 | val으로 정확도 지표 산출 (precision, recall, accuracy, mAP, …) | ✅ 완료 |
| 2 | 현재 CIL 학습 세팅 확인 (initial vs continual 클래스 수) | ✅ 완료 |
| 3 | A-GEM 이후 학습 방법 조사/비교 (개선 스트레스보다 실사용 문제 해결) | ✅ 완료 |
| 4 | (파생) 데모 앱 실사용 점검 → 발견된 버그 수정 | ✅ 완료 |
| 5 | (파생) full-48-way 데이터 스케일링 정량화 ("정확도 낮은 게 데이터 부족 때문?") | ✅ 완료 |
| 6 | (파생) CPU-only 학습 방법 추가 조사 — backprop-free 스트리밍/닫힌해 계열 실측 | ✅ 완료 |
| 7 | (파생) SSv2를 CIL로 평가한 선행연구 조사 + TCD 반례 재검증 | ✅ 완료 |

전체 테스트 68 passed, 브랜치 `feature/pycil-benchmarks` (PR #2).

---

## [x] 1. val 정확도 지표 산출 (precision, recall, accuracy, mAP)

**스크립트:** [`dev/compute_val_metrics.py`](../dev/compute_val_metrics.py)
**리포트:** [`reports/val_metrics_result.md`](val_metrics_result.md)

배포된 실제 체크포인트(`baseline_48cls.pt`, `agem_48cls.pt`)로 val set 4702개 전량 평가.
**두 평가 체제를 동시에 산출**한 게 핵심 — 이유는 TODO #2와 직결됨(아래 참고):

| Regime | Metric | Baseline | A-GEM | Δ |
|---|---|---|---|---|
| **Full 48-way** (진짜 class-IL, task ID 없음, chance=2.1%) | Accuracy | 0.062 | 0.099 | +0.037 |
| Full 48-way | mAP | 0.043 | 0.104 | **+0.060** |
| Full 48-way | F1 (macro) | **0.015** | 0.068 | +0.053 |
| Task-aware 6-way (chance=16.7%, 기존 headline `avg_acc`) | Accuracy | 0.230 | 0.363 | +0.133 |
| Task-aware 6-way | F1 (macro) | 0.163 | 0.353 | +0.190 |

**발견한 것:**
- Task ID 없이(진짜 class-IL) 평가하면 정확도가 훨씬 낮음 — task-aware 수치만 보고하면 과대평가.
- **baseline의 F1(macro)가 0.015로 붕괴** — catastrophic forgetting의 전형적 지문(순차학습이 최근 클래스로 예측을 몰아주면서 클래스별 재현율이 거의 0). A-GEM은 0.068로 완화(그래도 어려운 지표이므로 절대치는 낮음, 정직하게 인정).
- A-GEM의 baseline 대비 우위는 **두 체제 모두에서 유지**.

---

## [x] 2. 현재 CIL 학습 세팅 (initial vs continual 클래스 수)

**확인 방법:** 코드 직접 추적 (`experiments/run_capacity_ablation.py::run_one`, `src/utils/gem.py::precompute_ref/apply`)

| 구간 | 클래스 수 | 비율 | 메커니즘 |
|---|---|---|---|
| **Stage 1 = initial training** | 6 클래스 | 12.5% | `gem._memory`가 비어있어 `precompute_ref`가 `g_ref=None` 반환 → `apply()`가 no-op → **순수 cross-entropy, replay 없음** |
| **Stage 2~8 = continual training** | 42 클래스 (7 stage × 6) | 87.5% | 매 스테이지 종료 시 `gem.add_stage()`가 balanced 50개를 replay 메모리에 추가 → 이후 스테이지는 growing 메모리로 A-GEM 보호받으며 학습 |

**결론:** 초기 6클래스 : 지속 42클래스 = **1 : 7 스테이지 비율**. 이 구조가 확정되면서 TODO #1의 "왜 두 체제로 평가했나"도 함께 설명됨 — 우리 `eval_stage`가 스테이지별 6-way로 argmax를 제한하는 **task-aware(task-incremental) 평가**라, chance는 1/48이 아니라 **1/6**이고 class-IL SOTA와 직접 비교 불가능함을 명시.

---

## [x] 3. A-GEM 이후 학습 방법 조사/비교

**스크립트:** [`dev/run_er_comparison.py`](../dev/run_er_comparison.py), [`dev/run_gdumb_comparison.py`](../dev/run_gdumb_comparison.py)
**리포트:** [`reports/post_agem_methods_comparison.md`](post_agem_methods_comparison.md)

### 실측 비교 (동일 스택, mem 예산 동일)

| 방법 | 메커니즘 | Avg Acc | S1 forgetting |
|---|---|---|---|
| **GDumb** (ECCV'20) | greedy balanced 버퍼 + 매 스테이지 재학습, CL 알고리즘 없음 | 0.164 ± 0.004 | +0.059 ± 0.042 |
| baseline | 메모리 없음 | 0.251 ± 0.021 | +0.115 ± 0.088 |
| plain ER | 리허설, gradient projection 없음 | 0.346 ± 0.012 | +0.020 ± 0.098 |
| **A-GEM** | ER + gradient projection | **0.387 ± 0.018** | **−0.061 ± 0.077** |

**핵심 발견 — "CL 이득이 버퍼 효과일 뿐 아니냐"는 표준 반론을 직접 검증·반증:**
GDumb(0.164)이 chance(0.167) 수준이고 **baseline보다도 낮음** → 버퍼 존재 자체가 이득을 만드는 게 아니라, **gradient projection(A-GEM) 알고리즘이 실제 가치를 더한다**(GDumb 대비 +0.223).

### 문헌 비교 (A-GEM 이후 흐름)
ER-ACE·DER/DER++·X-DER(리허설 강화), L2P·DualPrompt·RanPAC(frozen backbone + prompt/prototype), PIVOT·SMILE·ESSENTIAL(비디오 CIL SOTA, 우리와 같은 frozen-CLIP 재료 사용) 등을 표로 정리 — 상세는 리포트 참고.

**가이드 반영:** capacity·backbone·attention·SSM·data-scale·ER·GDumb까지 폭넓게 통제 실험을 마쳤고, 여기서 더 새 알고리즘을 얹는 것은 한계효용이 낮다고 판단 → **정확도 개선 탐색을 멈추고 실사용 품질(TODO #4)로 방향 전환**.

---

## [x] 4. (파생) 데모 앱 실사용 점검 → 버그 수정

TODO #3의 "개선에 스트레스 받기보다 사용하면서 안되는 것을 고치자"는 지침을 실제로 실행한 항목.

**발견:** 데모를 직접 구동해보다가 **OOD 이상탐지 배지가 기본 설정(`uvicorn app:app` 그대로, Docker/HF Spaces 배포 경로 포함)에서는 한 번도 작동한 적이 없었음**을 발견. 원인: `threshold=None`이 기본값이라 `is_anomaly`가 항상 `False`로 고정 — 크래시가 없어 눈에 안 띄던 조용한 버그.

**수정:** `app.py` 시작 시 자동 보정 로직 추가.
- 로컬(실제 val feature 있음): 실제 in-distribution 분포로 보정.
- Docker/HF Spaces(데이터 미포함): `experiments/calibrate_anomaly.py`와 동일한 synthetic in-distribution 샘플로 폴백.

**검증:** held-out 데이터로 보정 유효성 확인 — 목표 FPR 5% vs 실측 4%, feature-space OOD 입력은 42% 플래그(정상 데이터 4%와 뚜렷이 구분). 재발 방지 회귀 테스트 추가(`tests/test_app_endpoints.py::test_anomaly_threshold_auto_calibrated`).

---

## [x] 5. (파생) full-48-way 데이터 스케일링 정량화

**질문:** "full-48-way 정확도가 6~10%로 낮은 게 학습 데이터가 적어서인가?"

**스크립트:** [`dev/run_data_scale_full48way.py`](../dev/run_data_scale_full48way.py)
**리포트:** [`reports/data_scale_full48way_result.md`](data_scale_full48way_result.md)

기존 데이터 스케일링 연구(task-aware만 측정)를 **full-48-way 지표까지 확장**해 같은 (seed, fraction) 프로토콜로 재측정.

| 데이터 | Task-aware Acc | Full-48-way Acc | Full F1(macro) |
|---|---|---|---|
| 25% | 0.314 | 0.070 | 0.036 |
| 100% | 0.388 | **0.105** | **0.075** |
| 25%→100% 상대증가 | +23.6% | **+49.9%** | **+109.2%** |

**결론 (세 원인의 복합작용으로 확정):**
1. **데이터 부족은 실제로 유의미하게 기여함** — full-48-way도 상대적으로 task-aware보다 더 크게 개선(F1은 2배 이상).
2. **하지만 그것만으론 설명 안 됨** — 100% 데이터를 다 써도 accuracy 10.5%(chance의 5배)에 그치고, 50→100% 구간에서 성장이 이미 2~3배 둔화(saturation 근접).
3. → **데이터 부족 + 망각(TODO #1의 F1 0.015 붕괴) + CLIP의 구조적 모션 표현력 한계**, 세 가지가 함께 작용.

---

## [x] 6. (파생) CPU-only 학습 방법 추가 조사 — backprop-free 계열 실측

**질문:** "CPU로만 학습을 시도하는 다른 학습 방법이 더 있나?"

**스크립트:** [`dev/run_cpu_friendly_methods.py`](../dev/run_cpu_friendly_methods.py)
**리포트:** [`reports/cpu_friendly_methods_result.md`](cpu_friendly_methods_result.md)

frozen-feature CL의 표준 계열(**backprop조차 없는** 스트리밍 통계/닫힌해: NCM 프로토타입,
Deep SLDA, Ridge RLS/ACIL, RanPAC-lite)을 문헌 조사 후 동일 8-stage 프로토콜로 실측.

| 방법 (replay 버퍼 불필요) | task-aware | full-48 acc | full F1 | 학습(fit only) | train FLOPs |
|---|---|---|---|---|---|
| NCM 프로토타입 | 0.378 | 0.124 | 0.105 | 2.4 ms | 44 M |
| Deep SLDA ('20) | 0.390 | 0.141 | 0.123 | 1.9 s | 6.61 G |
| Ridge RLS (닫힌해) | 0.373 | 0.143 | 0.122 | 21 ms | 3.06 G |
| RanDumb-style RFF+SLDA (NeurIPS'24) | 0.395±0.003 | 0.145 | 0.130 | 532 ms | 64.3 G |
| **FeCAM shared-cov (NeurIPS'23)** | **0.410** | **0.157** | **0.144** | **42 ms** | **2.85 G** |
| (참고) GRU + A-GEM | 0.387 | 0.105 | 0.075 | ~270 s | 4.64 T |

**핵심 발견 (2차 확장 — 최신 방법 포함):** backprop-free 통계 계열이 GRU+A-GEM과 동률을 넘어
**최신 FeCAM(shared)은 전 지표에서 명확히 우위**(task 0.410 vs 0.387, full-48 0.157 vs 0.105,
학습 6,400배 빠르고 연산량 1,628배 적음 — `reports/flops_result.md`). 통계 계열은 클래스별 통계가 독립 누적이라 **망각이 구조적으로 없음** —
backprop 계열의 진짜 적이 망각(로짓 쏠림)이었음을 역으로 증명. RanDumb(NeurIPS'24)의 "랜덤
표현이 학습된 표현을 이긴다"도 재현. 최신 문헌 지형(AnaCP NeurIPS'25 = gradient 없이
joint-training 상한 주장, StPR ICLR'26 = exemplar-free 비디오 CIL SOTA)도 정리 — 프론티어
자체가 exemplar-free·분석적 head 방향으로 이동 중이라 우리 결론과 합류. few-shot enrollment를
프로토타입 방식으로 바꾸면 등록이 밀리초 단위가 되는 실용 시사점도 있음.

---

## [x] 7. (파생) SSv2를 CIL로 평가한 선행연구 조사 + TCD 반례 재검증

**질문:** "SSv2를 CIL 관점에서 평가한 연구들이 있는가?"

**리포트:** [`reports/sota_positioning_brief.md`](sota_positioning_brief.md) §1(f) ·
[`reports/base_heavy_split_result.md`](base_heavy_split_result.md)
**스크립트:** [`dev/run_base_heavy_split.py`](../dev/run_base_heavy_split.py)

문헌 조사 결과 **TCD(Park et al., ICCV 2021)**가 SSv2를 UCF101·HMDB51과 함께 CIL
벤치마크로 처음 구성(84-class base + base-heavy incremental session), 이후 CSTA(2025)·
STSP(ECCV'24)·ESSENTIAL(ICCV'25)이 같은 프로토콜을 계승 — **"SSv2로 CIL 하는 게
새롭다"는 주장은 부정확**, TCD를 Related Work에 반드시 인용해야 함.

**결정적 반례 발견 및 재검증:** TCD는 SSv2에서 **NME(prototype 계열, 우리 FeCAM과
정신 유사)가 CNN보다 못하다**고 보고 — 우리 결론과 정면 충돌 가능성. 우리 48클래스를
TCD 스타일 base-heavy 구조(24:6=4배, 36:6=6배)로 재편해 true class-IL로 직접 재검증:

| split | FeCAM last | GRU+A-GEM last | Δ |
|---|---|---|---|
| 균등(원래) | 0.157 | 0.105 | +0.052 |
| moderate(4배) | 0.157 | 0.109 | +0.048 |
| aggressive(6배) | 0.157 | **0.084** | **+0.073** |

**반전은 일어나지 않았고 오히려 격차가 확대됨.** 부수 발견: GRU+A-GEM이 base 세션
직후 aggressive에서 −65% 급락 — A-GEM의 stage당 고정 replay 예산(50개)이 base
클래스가 많을수록 클래스당 exemplar를 급감시키기 때문(8.33→1.39개/class). FeCAM
강점과는 별개인 A-GEM 구현 자체의 개선 포인트로 확인. TCD와 결과가 다른 이유는
backbone 차이(TCD=SSv2 직접학습 CNN, 우리=frozen CLIP) 가설이며, 완전 검증은 원본
174-class split 재현 필요(영상 접근 복구 후 후속).

---

## 종합 산출물

| 파일 | 내용 |
|---|---|
| [`reports/sota_positioning_brief.md`](sota_positioning_brief.md) | 위 7개 항목 전부 통합된 최종 미팅 브리핑 (Q&A 포함) |
| [`reports/val_metrics_result.md`](val_metrics_result.md) | TODO #1 상세 |
| [`reports/data_scale_full48way_result.md`](data_scale_full48way_result.md) | TODO #5 상세 |
| [`reports/cpu_friendly_methods_result.md`](cpu_friendly_methods_result.md) | TODO #6 상세 |
| [`reports/post_agem_methods_comparison.md`](post_agem_methods_comparison.md) | TODO #3 상세 |
| [`reports/base_heavy_split_result.md`](base_heavy_split_result.md) | TODO #7 상세 (TCD 반례 재검증) |
| [`reports/pycil_bridge_result.md`](pycil_bridge_result.md) | PyCIL 표준 벤치마크 교차검증 (CIFAR-100) |
| [`reports/measured_evidence.md`](measured_evidence.md) | 이전 라운드 실측 증거(ER/std/capacity/효율) |

## 커밋 이력 (`feature/pycil-benchmarks` 브랜치, PR #2)

```text
f5d4c39 study: TCD base-heavy split re-validation — FeCAM lead holds, widens
ea6cb29 feat: PyCIL benchmark setup (CIFAR-100 + ImageNet) + analytic-head bridge
e364e2c study: MobileCLIP-S0 edge-encoder benchmark — CPU latency inverts params
f248d8b docs: pin official project focus — edge (CPU/embedded) continual learning for video
56bd80a feat: integrate FeCAM head into the demo (3-way live captions, ms enrollment)
cfc0b4c study: modern (2023-26) backprop-free CL — FeCAM sets new best on all metrics
532d93c docs: refresh commit-history block (item 6 added)
77608f2 study: backprop-free CPU methods (NCM/SLDA/RLS) match or beat GRU+A-GEM
f38b4dc docs: refresh commit-history block in TODO status report
adfec1b study: quantify data-scarcity contribution to low full-48-way accuracy
810ac39 docs: add TODO progress status report for meeting prep checklist
6249a81 fix: OOD/anomaly badge never fired without ANOMALY_THRESHOLD env var
5a9c9ec study: GDumb ablation refutes 'CL gain = buffer effect' + post-A-GEM method survey
fec671a study: val precision/recall/F1/mAP (full 48-way vs task-aware) + CIL setup docs
```
