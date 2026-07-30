# coad-mini 연구 로그 (2026-06-05 → 2026-07-30)

25개로 흩어져 있던 리포트를 **날짜순 하나**로 통합한 문서. 각 항목은 *질문 → 한 것 →
수치 → 판정* 순이고, 전체 상세는 링크된 원본에 있다(원본은 서로/코드에서 참조되므로
그대로 둔다). 새 실험을 하면 여기에 날짜 항목을 추가한다.

- **오늘 유효한 수치만 보려면** → [§ 현재 상태](#현재-상태-2026-07-30-기준)
- **철회·정정된 주장** → [§ 정정 이력](#정정-이력-틀렸다고-확인한-것들)
- **남은 과제** → [§ 열린 과제](#열린-과제)

---

## 전체 흐름 요약

프로젝트는 세 국면을 지났다.

| 국면 | 기간 | 질문 | 결론 |
|---|---|---|---|
| **I. 아키텍처 탐색** | 6/05 – 6/07 | GRU/A-GEM을 뭘로 키워야 좋아지나? | **아무것도 안 통함.** 용량·backbone·Attention·SSM 전부 무효 또는 악화. 데이터만이 레버 |
| **II. 방향 전환** | 7/02 – 7/16 | 그럼 backprop 자체가 필요한가? | **불필요.** FeCAM(통계 head)이 GRU+A-GEM을 전 지표에서 이김, 학습 42ms |
| **III. 외부 검증·한계 규명** | 7/22 – 7/30 | 우리 벤치 밖에서도 성립하나? 한계는? | **성립.** CIFAR-100·UCF101에서 재현. 한계는 **모션 데이터**였고, 절반은 공짜로 회복 |

핵심 서사: *"frozen feature 위에서는 아키텍처 복잡도도, gradient 학습 자체도 필수가 아니다.
병목은 표현과 데이터다."*

---

# 국면 0 — 그 이전: OrthGrad 실패 → A-GEM (~2026-06-04)
📄 [`docs/research_log.md`](../docs/research_log.md) — 서술형 연구 일지(의사결정 과정 중심)

이 로그는 6/05부터의 **실험 단위** 기록이고, 그 앞의 출발점은 별도 문서에 있다. 요지만:

**첫 가설 OrthogonalGradient는 실패했다.** 이전 stage의 **평균 gradient**를 `g_ref`로 저장해
직교 투영했더니 λ가 클수록 나빠졌다(λ=0 → 0.346, λ=1.0 → 0.271).
원인: 저장된 평균 gradient는 모델이 *실제로 어디서 틀리는지*와 무관하고, SSv2 클래스는
특징 공간에서 겹치므로 그 방향을 억제하면 현재 학습까지 방해한다.

**교훈이 A-GEM으로 이어졌다** — 수식은 같고 **`g_ref`의 출처만 다르다**(평균 gradient →
실제 과거 샘플). 48 classes × 8 stages에서 정확도 **+14.1%p**, S1 forgetting 보호 **+15.7%p**.
이후 multi-seed → class scaling → memory ablation → replay ratio → balanced memory 순으로
굳혀 `best_current_h256`이 나왔고, **국면 I은 그걸 더 키울 수 있는지 묻는 데서 시작한다.**

---

# 국면 I — 아키텍처 탐색 (2026-06)

## 2026-06-05 · GRU 용량·메모리 전략 3-seed 검증
📄 [`capacity_ablation_cv_report.md`](capacity_ablation_cv_report.md) · [`next_experiment_priority.md`](next_experiment_priority.md)

**질문:** GRU를 키우거나 메모리 선택을 바꾸면 좋아지나?
**한 것:** 48-class SSv2 / 8-stage / A-GEM, 4개 config × 3 seed.

| Config | Avg Acc | S1 Drop |
|---|---|---|
| **best_current_h256** (main) | **0.388 ± 0.020** | −0.078 |
| h512 | 0.387 ± 0.012 | −0.039 |
| h512 + 2layer | 0.355 ± 0.048 | +0.008 |
| h512 + 2layer + balanced_hard | 0.272 ± 0.042 | +0.110 |

**판정:** 용량은 병목이 아니다. hard-example 메모리는 크게 실패(고손실 표본에 편중되면
A-GEM의 참조 gradient가 과거 분포를 대표하지 못함). **main 모델 h256 유지.**

## 2026-06-06 · backbone 업그레이드(OpenCLIP L/14) + Attention
📄 [`openclip_l14_result.md`](openclip_l14_result.md) · [`gru_attention_result.md`](gru_attention_result.md)

| 모델 | Avg Acc | S1 Drop |
|---|---|---|
| CLIP B/32 + GRU | 0.388 | −0.078 |
| OpenCLIP L/14 + GRU | 0.401 ± 0.033 | +0.047 |
| OpenCLIP L/14 + GRU **+ Attention** | **0.298 ± 0.019** | **+0.240** |

**판정:** backbone은 **+0.013(노이즈 범위)**. Attention은 정확도 −0.103, 망각 +0.240으로
**양쪽 다 크게 악화** — 파라미터가 늘면 A-GEM의 제약이 약해져 망각이 급증한다.

## 2026-06-07 · 데이터 규모 스케일링 + SSM
📄 [`data_scale_result.md`](data_scale_result.md) · [`ssm_result.md`](ssm_result.md)

**질문 A:** backbone 효과가 데이터가 적어서 가려진 건가?

| 데이터 | B/32 | L/14 | Δ |
|---|---|---|---|
| 25% | 0.314 | 0.339 | +0.026 |
| 100% | 0.388 | 0.401 | **+0.013** |

**가설 기각.** 격차가 데이터가 늘수록 **좁혀진다** — L/14 우위는 저데이터 현상이다.
그리고 데이터 25%→100%는 **+0.06~0.07**로 backbone(+0.013)의 3~7배.

**질문 B:** Mamba류 SSM은? (CUDA 커널 불가라 순수 PyTorch diagonal SSM으로 구현)

| 시간 모델 | Avg Acc | S1 Drop |
|---|---|---|
| GRU | **0.401** | +0.047 |
| SSM (diagonal, 2층) | 0.361 (**−0.040**) | +0.173 |
| GRU + Attention | 0.298 (−0.103) | +0.240 |

**판정:** SSM도 못 이김. 단 분산이 최소(±0.008)라 "복잡도를 더해도 안 망가지는" 수준까진 옴.
**중요 caveat:** 16프레임은 SSM의 장거리 강점이 발휘되기엔 너무 짧다 — 긴 시퀀스에서 재평가 여지.

> **국면 I 종합:** 용량 ❌ / 메모리 전략 ❌ / backbone ⚠️(노이즈) / Attention ❌ / SSM ❌ /
> **데이터 ⭕(지배적)**. 단순 GRU+A-GEM이 강건한 최적점 — 아키텍처 튜닝 중단.

---

# 국면 II — backprop을 버리다 (2026-07 전반)

## 2026-07-02 · 측정 근거 정리 + SOTA 지형
📄 [`measured_evidence.md`](measured_evidence.md) · [`sota_positioning_brief.md`](sota_positioning_brief.md)

**한 것:** 주장에 쓸 수치를 전부 실측으로 확정하고, 문헌 지형을 정리.

방법 비교(5 seed): baseline 0.251 → plain ER 0.346 → **A-GEM 0.387 ± 0.018**.
A-GEM이 plain ER를 **+4.1%p**로 이김(표준 class-IL의 통념과 반대 — 우리 task-aware·frozen 세팅 특성).

효율: GRU head **604K params**(VideoMAE-V2 ViT-g의 1/1,700), 8-stage 전체 학습 **~4.5분/seed**,
peak RAM **0.95GB**, end-to-end 추론 ~156ms/window(**CLIP이 병목**).

**SSv2-CIL 선행연구 지형** (전부 원문 확인, TCD 84 base + 10×9 프로토콜):

| 논문 | venue | exemplar | 방식 | SSv2 |
|---|---|---|---|---|
| TCD | ICCV'21 | 20/class | ResNet-50+TSM fine-tune | 35.78 / 29.60 |
| STSP | ECCV'24 | **0** | null-space gradient 투영 | **69.68 / 70.87** |
| CSTA | 2025 | 0 / 5 | TimeSformer + 시공간 어댑터 | 41.26 |
| ESSENTIAL | ICCV'25 Highlight | sparse + prompt | **frozen CLIP** + 학습형 temporal encoder | 48.9 / 47.5 |

## 2026-07-12 · 진짜 class-IL 지표 + A-GEM 이후 방법 조사
📄 [`val_metrics_result.md`](val_metrics_result.md) · [`post_agem_methods_comparison.md`](post_agem_methods_comparison.md)

**중요한 정직성 조정:** 지금까지의 headline은 **task-aware 6-way**(task ID 주어짐, chance 16.7%)였다.
task ID 없는 **full-48-way**(chance 2.1%)를 함께 측정:

| Regime | Baseline | A-GEM |
|---|---|---|
| Full 48-way Acc | 0.062 | **0.099** |
| Full 48-way F1(macro) | 0.015 | 0.068 |
| Task-aware 6-way Acc | 0.230 | 0.363 |

절대치가 훨씬 낮다. 이후 모든 비교는 **두 지표를 병기**한다.

**GDumb 반증 실험:** "CL 이득이 그냥 버퍼 효과 아니냐"는 표준 반론 검증 →
GDumb **0.164** (chance 0.167 수준, baseline 0.251보다도 낮음). **버퍼의 존재가 아니라
gradient projection이 실제 가치를 만든다**는 근거.

## 2026-07-13 · full-48-way에서 데이터 스케일링 재검증
📄 [`data_scale_full48way_result.md`](data_scale_full48way_result.md)

| 데이터 | Task-aware | **Full-48-way** | Full F1 |
|---|---|---|---|
| 25% | 0.314 | 0.070 | 0.036 |
| 100% | 0.388 | **0.105** | 0.075 |
| 상대증가 | +23.6% | **+49.9%** | **+109.2%** |

**판정:** 데이터는 진짜로 도움이 된다(상대적으로는 task-aware보다 더 크게). **하지만**
4배 늘려도 full-48-way가 10.5%(chance의 5배)에 머물고, 50%→100% 구간에서 성장이
**3배 둔화**됐다. → 원인은 데이터 부족 + 망각 + **CLIP의 모션 표현력 한계** 복합.

## 2026-07-14 · ⭐ backprop-free 계열 실측 — 전환점
📄 [`cpu_friendly_methods_result.md`](cpu_friendly_methods_result.md)

**질문:** frozen feature 위에서 gradient 학습을 아예 없애면?

| 방법 | task-aware | full-48 acc | full F1 | 학습(fit only) | 버퍼 |
|---|---|---|---|---|---|
| **FeCAM (shared cov)** | **0.410** | **0.157** | **0.144** | **42 ms** | 불필요 |
| RanDumb RFF(2000) | 0.395 | 0.145 | 0.130 | 532 ms | 불필요 |
| Deep SLDA | 0.390 | 0.141 | 0.123 | 1.9 s | 불필요 |
| NCM 프로토타입 | 0.378 | 0.124 | 0.105 | 2.4 ms | 불필요 |
| **GRU + A-GEM (기존 main)** | 0.387 | 0.105 | 0.075 | ~270 s | 필요 |
| FeCAM per-class cov | 0.320 | 0.094 | 0.092 | — | 불필요 |

**판정 — 프로젝트 방향이 바뀐 지점:**
1. **FeCAM이 전 지표 신기록.** full-48-way에서 GRU 대비 **+50%**, F1 **+92%**, 학습은 **6,400배 빠름**.
2. **학습을 아예 안 하는 NCM조차** full-48-way에서 GRU를 이김(0.124 vs 0.105).
3. **이유는 망각이 구조적으로 없기 때문** — 클래스별 통계가 독립 누적되므로 gradient 간섭이 원천 부재(S1 drop ≈ 0).
4. **시간 모델링 기여가 0임을 재확인** — 시간 순서를 완전히 버린 mean-pool이 시퀀스를 읽는 GRU와 동률/우위.
5. **FeCAM per-class는 실패** — 클래스당 100개로 512² 공분산 추정은 과적합. 작은 데이터에선 **공유 공분산이 정답**.

**후속 조치:** FeCAM head를 데모 앱에 실제 탑재(`src/models/fecam_head.py`),
few-shot 등록을 **밀리초 프로토타입 등록**으로 대체 — 망각이 구조적으로 불가능한 등록.

## 2026-07-16 · MobileCLIP-S0 인코더 지연 측정
📄 [`mobileclip_result.md`](mobileclip_result.md)

| Backbone | params | CPU(4스레드) | MPS |
|---|---|---|---|
| CLIP ViT-B/32 | 87.5M | **161 ms** | — |
| MobileCLIP-S0 | **10.9M** (8배 작음) | **2,092 ms** (13배 느림) | 125 ms |

**판정:** **파라미터·FLOPs는 실제 edge 지연을 예측하지 못한다.** MobileCLIP의 효율 주장은
CoreML/ANE 기준이고, 일반 PyTorch CPU에서 depthwise-conv 블록은 최적화된 ViT matmul보다 느리다.
스레드를 4→10으로 늘리면 오히려 더 느려짐(2.09s→2.81s).
→ **"CPU-only"는 하나가 아니다**를 우리 주장에 반영해야 함.
(다운스트림 정확도 비교는 원본 영상 부재로 보류 → **7/27 해소**.)

---

# 국면 III — 외부 검증과 한계 규명 (2026-07 후반)

## 2026-07-22 · PyCIL 표준 벤치 교차 검증 (CIFAR-100)
📄 [`pycil_bridge_result.md`](pycil_bridge_result.md)

**질문:** 우리 결론이 자체 SSv2 벤치에서만 성립하는 것 아닌가?
**한 것:** PyCIL의 `DataManager`에서 클래스 순서를 직접 import(스모크 로그와 비트 단위 일치 확인),
b0=50 / inc=10 / seed 1993, 진짜 class-IL.

| head | Avg Inc Acc | Last Acc |
|---|---|---|
| **FeCAM (shared cov)** | **0.769** | **0.733** |
| Deep SLDA | 0.725 | 0.686 |
| NCM | 0.698 | 0.660 |

**판정:** **서열이 SSv2와 완전히 일치**(FeCAM > SLDA > NCM). 자체 벤치 특유 현상이 아님을 교차 검증.
**공정성 주석(논문 필수):** 우리는 web-pretrained frozen CLIP = **PTM 트랙**, PyCIL 고전(iCaRL/DER)은
**from-scratch ResNet 트랙**. 한 표에 섞어 우열을 주장하면 안 됨.

부수 작업: PyCIL 상단 크래시 패치(`init_cls != increment`일 때 정확도 행렬 크기 불일치).

## 2026-07-24 · base-heavy split — TCD 반례 재검증
📄 [`base_heavy_split_result.md`](base_heavy_split_result.md)

**질문:** TCD는 "base가 크면 순서가 뒤집힌다"고 했다. 우리도 그런가?
**한 것:** 84-class base + 소수 incremental (base:inc = 8.4~16.8배).

**판정: 반전이 일어나지 않았다 — 오히려 격차가 벌어짐**(FeCAM 우위 +0.052 → **+0.073**).
GRU+A-GEM은 base 세션 직후 붕괴하는 메커니즘까지 확인(고정 mem_per_stage 때문).

**TCD와 왜 다른가(원문 확인):** TCD의 반례는 **SSv2로 학습되어 모션을 담은 feature 위의 mean-pool**에
대한 것이고, 우리는 **frozen CLIP feature** 위다. 전제가 다르다.

## 2026-07-25 · PyCIL 서베이 감사 — 갭 주장 축소
📄 [`pycil_survey_edge_gap.md`](pycil_survey_edge_gap.md)

**질문:** "CPU-only edge CL은 아무도 안 한다"는 우리 주장이 맞나?
**한 것:** Zhou et al. TPAMI'24 서베이 38쪽 전문 키워드 전수 검색 + 선행 2편 원문 확인.

- 서베이에 **"CPU" 0회, "FLOPs" 0회** — edge를 **메모리 예산(바이트)** 으로만 정의.
- **그러나 SparCL(NeurIPS'22)은 Galaxy S20 CPU에서 실제 학습을 측정**(3.1× 가속) — 우리 초기 주장을 반박.
- BudgetCL(CVPR'23)은 연산 예산을 **iteration 수**로 추상화.

**판정 — 주장 축소(정직성 조정):** "edge CL 최초"는 **틀렸다**. 우리 기여는
**"비디오 CIL × backprop-free × 절대 CPU wall-clock"** 삼중 교집합으로 좁혀야 한다.

## 2026-07-26 · UCF101 TCD 프로토콜 + FLOPs 회계
📄 [`ucf101_tcd_result.md`](ucf101_tcd_result.md) · [`flops_result.md`](flops_result.md)

**한 것:** 남의 논문과 **숫자를 나란히** 놓기 위해 UCF101을 TCD 원문 프로토콜(51 base + 10/5/2)로 평가.
클래스 순서가 TCD의 `class_list.pkl`과 일치함을 검증(101/101), 그룹 단위 분리도 확인(1,818/707 그룹, 중복 0).

| 방법 | UCF101 avg inc |
|---|---|
| ESSENTIAL (ICCV'25) | 95.1 |
| **우리 (frozen CLIP + FeCAM)** | **88.84 ± 0.52** |
| ST-prompt | 84.8 |
| STSP | 81.1 |
| TCD | 74.8 |
| **우리 (GRU + A-GEM)** | **57.39** |

**판정:** **ESSENTIAL 다음 2위.** 자체 baseline인 GRU+A-GEM과는 **+31.45%p** 격차.

**발견 1 — 증분 크기에 완전 불변:** inc=10/5/2 전부 88.84(변화 **0.00**). 다른 방법들은
증분이 작아질수록 떨어진다. → **스마트글래스에서 한 번에 한 클래스씩 등록해도 성능이 안 떨어진다.**

**발견 2 — 적용 범위를 정확히 규정:**

| 벤치 | 성격 | FeCAM |
|---|---|---|
| UCF101 | **static-biased**(외형으로 구별 가능) | **88.84** |
| SSv2 | **temporal-biased**(모션이 본질) | **15.7** |

세 head(NCM/SLDA/FeCAM) 전부 같은 폭으로 떨어짐 → head가 아니라 **표현의 문제**.

**FLOPs 회계**(1 MAC = 2 FLOPs, train = 3× forward): FeCAM 학습 **2.85 G**(GRU+A-GEM의 1/1,628).
**부수 발견:** 기존 리포트의 "FeCAM 학습 8.7초"는 러너 타이머가 평가 1회를 포함한 값 —
실제 fit은 **42.4 ms**(99.5%가 평가였음). 그리고 **인코더가 추론의 99.98%** 를 차지한다.

## 2026-07-27 · 인코더 교체 실험 (보류 과제 해소)
📄 [`encoder_swap_result.md`](encoder_swap_result.md)

7/16에 원본 영상 부재로 막혔던 다운스트림 비교를 UCF101 원본 영상으로 해소.

| 인코더 | UCF101 (FeCAM) | CPU 지연 |
|---|---|---|
| CLIP ViT-B/32 | **88.84** | **148 ms** |
| MobileCLIP-S0 | 88.63 (−0.21) | **2,985 ms** (20배 느림) |

**판정:** 정확도는 사실상 동일한데 CPU에서 20배 느리다. 7/16의 지연 측정 결론이
다운스트림 정확도까지 포함해 확정됨 — **인코더 교체는 "더 작은 모델" 문제가 아니라
"CPU에서 실제로 빠른 모델" 문제.** 미해결.

## 2026-07-30 · ⭐ SSv2 회복 — backprop 없이 +9.0%p
📄 [`ssv2_temporal_pooling_result.md`](ssv2_temporal_pooling_result.md)

**질문:** SSv2 15.7%를 어떻게 올리나? ESSENTIAL이 같은 frozen CLIP으로 48.9를 내는데?

**ESSENTIAL 원문 ablation(Table 3a)이 우선순위를 정해줬다:**

| 구성 | SSv2 |
|---|---|
| baseline — episodic memory만 (**학습형 temporal encoder는 있음**) | 42.1 |
| + MR 모듈 + semantic memory (논문의 실제 기여) | 48.9 (**+6.8**) |
| **우리 (frozen CLIP + mean-pool + FeCAM)** | **15.7** (baseline과 **−26.4**) |

→ **화려한 모듈(+6.8)을 흉내내는 것보다 "시간 축을 버리지 않는 것"(+26.4)이 먼저다.**

학습형 encoder는 backprop-free를 깨므로 질문을 좁혔다: **닫힌 형태로 얼마나 되찾을 수 있나?**

| pooling | 차원 | last | vs base |
|---|---|---|---|
| mean (기존) | 512 | 15.74 | — |
| mean+std | 1024 | 16.78 | +1.04 |
| mean+diff | 1024 | 20.69 | +4.96 |
| thirds | 1536 | 24.05 | +8.32 |
| **chunks3 + adjacent diffs** | 2560 | **24.71** | **+8.97** |

**판정: 파라미터 0개·gradient 0회로 +8.97%p (상대 +57%).** 격차의 약 1/3을 공짜로 메움.

**세 가지 통제 검증:**
- **순서 없는 통계는 무익** — mean+std는 +1.04뿐(std는 순열 불변). "차원이 늘어서"가 아니라 **"순서가 살아나서"** 오른 것.
- **세 head 전부 재현** — NCM +6.04, SLDA +8.36, FeCAM +8.97 → head가 아닌 **표현 수준** 개선.
- **UCF101에선 +1.17뿐** — static-biased라 시간 정보가 덜 필요. **이 비대칭이 해석을 뒷받침**(차원 증가가 원인이면 양쪽이 비슷하게 올랐어야).

## 2026-07-30 · 실시간 incremental training + FeCAM 캐시 결함
📄 [`realtime_incremental_result.md`](realtime_incremental_result.md)

**질문:** 10 fps로 incremental training이 도는가? 병목은?

| 경로 | 프레임당 | fps | 예산 대비 |
|---|---|---|---|
| 예측만 | 25.34 ms | 39.5 | 25% |
| **학습(등록)+예측** | **25.39 ms** | **39.4** | **25%** |
| 학습(공분산까지)+예측 | 34.38 ms | 29.1 | 34% |

**판정: 통과.** 학습을 켜는 한계비용 **0.05 ms**(예산의 0.1%).

**되는 이유가 자명하지 않다:** 배치 측정의 CLIP은 window당 148~159 ms라 매 프레임 재인코딩하면
**6.3 fps로 미달**. ring buffer로 **신규 1장만** 인코딩하는 구조가 이걸 가능하게 한다.
→ 배치 실험의 인코더 수치를 실시간에 대입하면 안 된다.

**병목: 인코더 91.1%, 헤드 0.4%** — backprop-free head가 실시간성에 위협이 아니라는 직접 실측 근거이자,
헤드를 더 최적화해도 얻을 게 없다는 뜻.

**발견한 실제 결함:** 처음 측정에서 학습 한계비용 8.39 ms 중 **7.97 ms(95%)가 캐시 재계산**이었다.
`observe()`가 조건 없이 캐시를 통째로 버렸는데, **등록은 공분산을 안 건드리므로 D×D 역행렬이
여전히 유효**했다. 캐시를 무효화 조건별로 분리해 등록을 **O(D³) → O(D²)** 로 수정.

**이 수정이 위 SSv2 개선을 살렸다** — chunks3+adjdiff는 D=2560이고 역행렬은 O(D³):

| pooling | D | 수정 전 | 수정 후 |
|---|---|---|---|
| mean | 512 | 8.43 ms | 0.32 ms (27×) |
| **chunks3+adjdiff** | **2560** | **315 ms** (예산 3배 초과) | **1.98 ms** (159×) |

## 2026-07-30 · chunks3+adjdiff를 기본값으로 배포
📄 커밋 `9e7a0a8` · Space [abed041](https://huggingface.co/spaces/Yhoon-3/coad-mini)

서빙 헤드 48-class val: **full-48-way 0.157 → 0.247**, task-aware 0.410 → 0.507.

- pooling을 **헤드 속성으로 만들어 체크포인트에 저장** — 예측과 등록이 어긋날 수 없고(둘 다
  `window_to_embedding` 경유), 이 필드가 없던 구 체크포인트는 `mean`으로 계속 서빙된다.
- feature_dim 512 → 2560이라 체크포인트가 52 MB가 될 뻔했으나, 공분산이 **정확히 대칭**임을
  이용해 상삼각 float32로 저장 → **13 MB**(정확도·argmax 100% 동일 검증).
- **T=16에서는 구간이 5/5/6으로 비대칭**이라 역재생 부호 반전이 정확히 성립하진 않는다
  (3의 배수일 때만). 순서 민감성 자체는 유지.

---

## 현재 상태 (2026-07-30 기준)

**서빙 구성:** frozen CLIP ViT-B/32 → chunks3+adjdiff pooling(2560-d) → FeCAM shared-cov head.
파라미터 학습 0개, gradient 0회, replay 버퍼 없음.

| 벤치 | 지표 | 값 | 비고 |
|---|---|---|---|
| SSv2 48-class | full-48-way | **0.247** | mean-pool 대비 +9.0%p |
| SSv2 48-class | task-aware 6-way | **0.507** | |
| UCF101 (TCD 프로토콜) | avg inc | **90.00** | ESSENTIAL 95.1 다음 2위 |
| CIFAR-100 (PyCIL split) | avg inc | 0.769 | 교차 검증용 |
| 실시간 | 학습+예측 | **39.4 fps** | CPU, 10 fps 예산의 25% |
| 학습 비용 | fit only | **42 ms** | GRU+A-GEM 대비 6,400배↓ |

**우리가 아직 못 이기는 축:** S1 backward-transfer(A-GEM의 −0.061은 리허설 고유 효과),
그리고 SSv2 절대치(ESSENTIAL 48.9 vs 우리 24.7).

---

## 정정 이력 (틀렸다고 확인한 것들)

연구 로그의 값어치는 틀린 걸 남기는 데 있다.

| 날짜 | 철회/정정된 주장 | 실제 |
|---|---|---|
| 7/25 | "CPU-only edge CL은 아무도 안 한다" | **틀림.** SparCL(NeurIPS'22)이 Galaxy S20 CPU 학습을 실측. 주장을 "비디오 × backprop-free × 절대 CPU wall-clock" 삼중 교집합으로 축소 |
| 7/26 | "FeCAM 학습 8.7초" | **오귀속.** 러너 타이머가 평가 1회 포함. 실제 fit **42.4 ms**(99.5%가 평가) |
| 7/26 | "ESSENTIAL이 TCD를 오인용했다" | **내 오독.** Table 2에 `TCD(ViT)`와 `TCD(TSM)` 두 행이 다 있음 |
| 7/26 | TCD의 77.2가 CNN 추론 / 41.26이 TCD 수치 | 각각 NME 77.16, **CSTA 자신의 수치** |
| 7/30 | 재계산 주기별 정확도 1.76%p 하락(7/27판) | **재현 불가.** 스크립트가 커밋 안 됨. 재현 가능한 형태로 다시 재니 **0.25%p**. 원인 미규명이라 이전 수치는 폐기 |
| 상시 | headline을 task-aware만 보고 | full-48-way 병기로 전환(7/12). 절대치가 훨씬 낮음 |

---

## 열린 과제

**막힌 것**
- **HMDB51** — 공식 split 서버가 HTML을 반환, HF 미러는 영상만. split을 지어낼 수 없어 보류.
- **인코더 교체** — MobileCLIP은 CPU에서 20배 느림. NPU 런타임(CoreML/ONNX-RT) 없이는 해결 불가.

**미실행 (우선순위 순)**
1. **실제 임베디드 보드**(Jetson/RPi) 측정 — "병목은 인코더" 결론이 강해질 뿐이지만 절대 fps는 미지.
2. **공분산 갱신 주기 튜닝** — D=2560에서 공분산 갱신은 325 ms(예산 초과). 평균은 매 프레임,
   역행렬은 가끔 갱신하는 운용이 자연스러운데, 주기가 정확도에 미치는 영향은 UCF101 800프레임·D=512
   한 조건에서만 확인됨.
3. **차원 축소와 결합** — 2560-d 공분산은 크다. random projection/PCA로 줄이면 더 많은 구간을 쓸 수 있을지도.
4. **AUC-A / AUC-L 채택** — PyCIL 서베이의 메모리-불가지론 지표.
5. **BudgetCL 방식의 iteration-budget 프로토콜**.
6. **학습형 temporal encoder를 선택적으로** — backprop-free 주장은 포기하되 상한 확인용.

---

## 원본 리포트 색인

날짜순. 위 요약보다 상세한 표·해석·재현 명령은 각 원본에 있다.

| 날짜 | 파일 |
|---|---|
| 06-05 | [capacity_ablation_cv_report.md](capacity_ablation_cv_report.md) · [next_experiment_priority.md](next_experiment_priority.md) |
| 06-06 | [openclip_l14_result.md](openclip_l14_result.md) · [gru_attention_result.md](gru_attention_result.md) |
| 06-07 | [data_scale_result.md](data_scale_result.md) · [ssm_result.md](ssm_result.md) |
| 07-02 | [measured_evidence.md](measured_evidence.md) · [sota_positioning_brief.md](sota_positioning_brief.md) |
| 07-12 | [val_metrics_result.md](val_metrics_result.md) · [post_agem_methods_comparison.md](post_agem_methods_comparison.md) |
| 07-13 | [data_scale_full48way_result.md](data_scale_full48way_result.md) · [meeting_prep_todo_status.md](meeting_prep_todo_status.md) |
| 07-14 | [cpu_friendly_methods_result.md](cpu_friendly_methods_result.md) |
| 07-16 | [mobileclip_result.md](mobileclip_result.md) |
| 07-22 | [pycil_bridge_result.md](pycil_bridge_result.md) |
| 07-24 | [base_heavy_split_result.md](base_heavy_split_result.md) |
| 07-25 | [pycil_survey_edge_gap.md](pycil_survey_edge_gap.md) |
| 07-26 | [ucf101_tcd_result.md](ucf101_tcd_result.md) · [flops_result.md](flops_result.md) · [meeting_script_0726.md](meeting_script_0726.md) |
| 07-27 | [encoder_swap_result.md](encoder_swap_result.md) · [meeting_deck_notion.md](meeting_deck_notion.md) · [progress_since_2026-07-17.md](progress_since_2026-07-17.md) |
| 07-30 | [ssv2_temporal_pooling_result.md](ssv2_temporal_pooling_result.md) · [realtime_incremental_result.md](realtime_incremental_result.md) |
