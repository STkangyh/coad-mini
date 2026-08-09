# coad-mini 연구 로그 (2026-06-05 → 2026-07-30)

25개로 흩어져 있던 리포트를 **날짜순 하나**로 통합한 문서. 각 항목은 *질문 → 한 것 →
수치 → 판정* 순이고, 전체 상세는 링크된 원본에 있다(원본은 서로/코드에서 참조되므로
그대로 둔다). 새 실험을 하면 여기에 날짜 항목을 추가한다.

- **오늘 유효한 수치만 보려면** → [§ 현재 상태](#현재-상태-2026-07-30-기준)
- **철회·정정된 주장** → [§ 정정 이력](#정정-이력-틀렸다고-확인한-것들)
- **약점·미해결 질문** → [§ 자체 감사](#자체-감사-2026-07-30--개선점-10--물음표-10)
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

**판정(2026-07-31 정정됨):** **반전은 일어나지 않았다**(유지). 단 *"격차가 벌어진다"* 는
**철회** — 아래 07-31 항목 참조.

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
| **chunks3 + adjacent diffs** | 2560 | **24.71** | **+8.97** ⚠️ |

⚠️ 이 변형 선택은 val에서 이뤄졌다 — 07-31에 held-out으로 재선택해 **+7.83**으로 정정됐다.

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

## 2026-07-30 · PyCIL 시간 측정 감사 — 툴박스에도 없다
📄 검증 대상: `external/PyCIL` @ `f3509b8`

**질문:** PyCIL이 training/inference 시간을 측정하나?

**답: 둘 다 아니다. 전 코드베이스에 `import time`이 한 줄도 없다.**

`time.time` · `perf_counter` · `timeit` · `elapsed` · `duration` · `latency` · `throughput`
전수 검색 결과 **0건**. 실행 로그 파일에서도 시간 관련 줄 0개.

| PyCIL이 재는 것 | PyCIL이 안 재는 것 |
|---|---|
| top1 / top5 정확도, grouped, 곡선 | **학습 시간** |
| Average Incremental Accuracy | **추론 시간 / latency / throughput** |
| Forgetting | **FLOPs** |
| 파라미터 수 (`count_parameters`) | **메모리(RAM/VRAM)** |

유일한 예외는 `tqdm` 진행바가 stderr에 경과시간을 띄우는 것뿐 — **기록도 보고도 되지 않는다.**

**왜 중요한가 — 7/25 서베이 감사와 정확히 같은 그림이고, 같은 그룹이다.**
PyCIL 저자는 Zhou Da-Wei · Wang Fu-Yun · Ye Han-Jia · Zhan De-Chuan (SCIS'23)이고,
우리가 7/25에 "CPU 0회, FLOPs 0회"를 확인한 그 서베이(TPAMI'24, Zhou Da-Wei 외)는
**PyCIL README가 직접 인용하는 같은 그룹의 논문**이다.

→ **이 분야의 표준 서베이와 표준 툴박스가 둘 다 시간을 재지 않는다.** 정확도와 망각,
그리고 (서베이의 경우) 메모리 바이트만 본다. 우리가 wall-clock과 FLOPs를 병기하는 게
중복이 아니라 **빈칸을 채우는 것**이라는 근거가 하나 더 생겼다.

**단, 과장 금지:** "아무도 시간을 안 잰다"가 아니다. 개별 논문(SparCL 등)은 잰다.
정확한 주장은 **"표준 평가 도구가 시간을 재도록 만들어져 있지 않아, 논문 간 시간 비교가
구조적으로 불가능하다"** 이다.

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

# 국면 IV — 배포 경로 검증 (2026-08)

## 2026-08-04 · 라이브 쿼리 윈도우가 벤치마크와 얼마나 다른가 — 물음표 4 재정의
📄 [`live_query_sim_result.md`](live_query_sim_result.md) · [`run_live_query_sim.py`](../dev/run_live_query_sim.py)

**질문의 전제가 틀렸었다.** 물음표 4는 원래 "실시간 few-shot 등록이 정확도로 이어지나"였는데,
`app.py`를 직접 읽어보니 **`/enroll`은 파일 업로드고 업로드 영상도 `np.linspace`로 전체
클립에 고르게 샘플링**한다 — 벤치마크와 완전히 같은 경로다. `enrollment_covariance_result.md`의
수치는 근사가 아니라 **실제 경로 그 자체**였다.

진짜 다른 건 **쿼리(`/predict_rt`)** 다. 브라우저는 200ms마다 캡처해 최근 16장을 ring buffer로
유지하고(`static/index.html`), 이건 **base 클래스든 방금 등록한 클래스든 모든 예측**에 적용된다.
그런데 우리가 보고한 모든 정확도는 **전체 클립에 고르게 펼친 curated 윈도우**로 쿼리한 값이다.
UCF101 평균 클립 길이 7.2초(실측) 대비 3.2초짜리 라이브 윈도우는 동작의 절반이 안 되는 구간을
임의의 시작점에서 담는다.

**측정:** 학습/등록은 그대로 두고(이미 검증됨) 쿼리 쪽만 바꿨다. test 505개 영상을 다시
디코딩해 브라우저와 같은 stride로 라이브 윈도우를 잘라내고(영상마다 fps가 25/29.97로 섞여
있어 영상별로 stride 계산), curated와 나란히 채점.

| pooling | base Δ(live−curated) | enrolled Δ |
|---|---|---|
| mean | **−2.09pp** (3시드 전부 음수, sd=0.82) | −1.33pp |
| chunks4 | **−1.57pp** (3시드 전부 음수, sd=0.40) | −2.00pp |

**base 하락은 노이즈가 아니다** — 3시드 전부 음수고 chunks4는 효과가 sd의 4배. **mean과
chunks4가 비슷하게 떨어져서**, 원인이 "구간 위치(순서) 문제"가 아니라 **"동작의 일부만 보는
정보량 손실"** 임을 시사한다. enrolled는 표본이 작아(시드당 10개) 노이즈가 크지만 방향은 같다.

**판정:** 떨어지지만 무너지지 않는다. −1.5~2.1pp는 실재하는 소폭 할인이지 벤치마크 수치와
질적으로 다른 시스템이라는 뜻은 아니다. 한계: 영상당 시작점 1개만 봤고(운 좋은/나쁜 타이밍의
분산은 미측정), UCF101만 확인(→ 08-05에 SSv2도 프록시로 확인), 캡처 화질 차이는 배제(프레임
선택만 격리).

## 2026-08-05 · ⭐ SSv2: chunks4의 이득이 부분 관측에서 역전된다
📄 [`partial_window_sim_result.md`](partial_window_sim_result.md) · [`run_partial_window_sim.py`](../dev/run_partial_window_sim.py)

08-04의 "SSv2에서는 어떨까"를 이어받았다. **SSv2 원본 영상이 이 환경에 없어** 08-04와 같은
재디코딩 실측이 불가능하다 — 대신 저장된 16장 curated 특징에서 **연속된 일부만 잘라내는
프록시**를 썼다. UCF101이 실측(재디코딩)과 프록시를 둘 다 가진 유일한 벤치라 먼저 검증:
coverage=0.44에서 프록시가 실측 대비 **15~35% 과소평가**한다(mean 프록시 −1.41 vs 실측
−2.09, chunks4 −1.33 vs −1.57). 즉 아래 SSv2 수치는 하한선으로 읽어야 한다.

**coverage를 낮출수록 (48클래스 전체 커리큘럼, 배포 설정 그대로):**

| coverage | mean | chunks4 | chunks4−mean |
|---|---|---|---|
| 1.00(배포값) | 15.74% | **23.56%** | +7.82pp |
| 0.50 | 14.64% | 17.05% | +2.41pp |
| 0.44 | 14.48% | 16.44% | +1.96pp |
| **0.25** | 13.71% | **13.45%** | **−0.26pp (역전)** |

**chunks4의 SSv2 우위가 coverage 0.25에서 mean보다 나빠진다.** UCF101의 같은 계산은
1.00→0.25에서 +1.49pp→+0.08pp로 줄기만 하고 역전은 없다(둘 다 static-biased라 순서
정보가 애초에 작은 보너스였을 뿐이라서). 메커니즘: chunks4의 구간 위치는 "동작 전체의
초반/중반/후반"을 인코딩하는데, 부분 윈도우에서는 그게 "우연히 잡힌 일부의 초반/중반/후반"이
되어 **등록 시(전체를 본 프로토타입)와 쿼리 시(부분만 본 라이브 윈도우) 사이에 구간 의미가
어긋난다** — 순서 정보가 도움이 아니라 잘못된 신호로 바뀐다. mean은 순서를 안 쓰니 이
문제가 아예 없어 완만하게만 떨어진다.

**한계 — 가장 중요한 것:** coverage=0.44는 UCF101 클립 길이(7.2초) 기준이다. **SSv2 클립이
더 짧으면 실제 커버리지 비율은 더 클 수 있고**(하락이 덜 심각), 원본 영상 없이는 정확한 값을
모른다. 역전이 일어나는 정확한 지점도 근사치다. 그럼에도 "chunks4가 mean보다 빠르게
무너진다"는 방향성은 coverage sweep의 단조적 패턴과 UCF101 대비 두 신호가 일치해 견고하다.

**판정:** 당장 pooling을 바꿀 근거는 아니지만(불확실성이 커서), **원본 영상 접근이 복구되면
최우선으로 재현해야 할 실험**이 됐다. chunks4를 SSv2에서도 계속 쓸지, 부분 관측에 더
강건한 변형(halves/thirds 등)을 따로 쓸지가 여기 달려 있다.

---

## 2026-08-06 · SSv2 원본 영상 확보 — 막혔던 항목 4개 해소

📄 [`ssv2_video_access_result.md`](ssv2_video_access_result.md)

Qualcomm 공식 배포처에서 19.4GB 직접 다운로드(220,847개 webm 전량, `COAD_VIDEO_DIR` 설정
완료). 위 08-05 항목이 최우선으로 지목했던 블로커가 해소됨. 파생 작업 4개:

1. **live_query_sim 실측 재현** — 프록시가 예측했던 "coverage 0.25 근처 역전"은 **실전에서
   안 일어남**. 실측 coverage=0.664(UCF101의 0.44보다 훨씬 큼 — SSv2가 더 짧고 12fps라서),
   그 지점에서 chunks4가 mean보다 여전히 +2.91pp 앞섬. 배포 설정(chunks4) 유지 근거 확정.
2. **174클래스 전체 subset·특징 구축** — `train.json`/`labels.json`의 브래킷 표기 불일치
   버그(전량 미스매치, 조용히 성공 종료됨) 하나 잡음.
3. **TCD 84+90 리터럴 재현** — 48클래스 근사가 냈던 "FeCAM이 GRU+A-GEM에 안 진다" 결론이
   174클래스 진짜 스케일에서도 3시드 전부 유지(avg_inc 우위 +3.78pp).
4. **48클래스 특징 재추출 + provenance 스탬핑** — 기존 특징에 메타데이터가 전무했던 것
   확인·해결. 부수 발견: SSv2 stream-sim이 "fps 메타데이터 없어 블록"이라던 이전 서술이
   **틀렸음**을 코드 재확인으로 정정(실제로는 fps를 아예 안 쓰는 함수였음).

---

## 2026-08-06 · AP-FPS 스윕 — 배포 조합이 Pareto인가 (08-08 두 차례 정정)

📄 [`ap_fps_sweep_result.md`](ap_fps_sweep_result.md)

pooling(mean/chunks4/chunks3_adjdiff) × head(NCM/SLDA/FeCAM) 9조합(clip_b32)을 mAP·FPS
평면에 처음으로 동시에 올림 — 이전엔 축마다 따로 최적화해왔음(pooling은 정확도로, head는
pooling 고정 후, 속도는 배포 조합 하나만).

- **측정 아티팩트 발견·수정:** FeCAM 원시 점수(음의 마할라노비스 거리)는 샘플별 오프셋
  편차가 ~1176인데 행 내부 편차는 4~14뿐 — macro AP가 이 오프셋에 오염돼 FeCAM을
  최하위로 잘못 보고했다(정확도는 최고인데). 행 z-score 정규화로 수정 후 **모든
  head·pooling 조합에서 FeCAM이 1위**로 뒤집힘.
- **FPS는 인코더가 95%+ 결정** — pooling·head를 아무리 바꿔도 39.4~41.6fps 범위 안,
  전부 10fps 예산의 4배 가까운 여유. 이 스코프에선 fps가 결정을 제약하지 않는다.
- chunks3+adjdiff가 val에서 근소 우위를 보이지만 **배포 변경 근거 아님**(단일 실행,
  held-out 프로토콜에서는 원래 통계적 동률).
- **08-08 정정 ①:** head를 처음엔 4개(NCM/SLDA/Ridge-RLS/FeCAM)로 돌렸는데, 사용자 확인
  없이 Ridge-RLS를 넣은 것이었다 — `cpu_friendly_methods_result.md`의 1세대 비교가
  NCM/SLDA/Ridge였다가 FeCAM 도입 후 `ssv2_head_curves_result.md`(08-08에 뒤늦게 작성 —
  전엔 스크립트·raw json만 있고 마크다운 리포트가 없었음)·`pycil_bridge_result.md`가
  이미 Ridge를 빼고 NCM/SLDA/FeCAM 3개로 정착시킨 이력이 있었는데, 그걸 확인 안 하고
  이미 은퇴한 head를 되살린 셈. Ridge 제외하고 재실행.
- **08-08 정정 ②:** backbone도 원래 clip_b32/openclip_l14 2개(24→18조합)로 비교했는데,
  L/14는 8배 느려(196ms vs 24ms) 10fps 예산 자체를 못 지켜 이 리포트의 질문("배포
  조합이 Pareto인가")과 무관한 잡음이었다 — clip_b32 단독(9조합)으로 스코프를 좁힘.
  결론(FeCAM 전승, fps는 인코더가 지배)은 두 정정 내내 그대로 유지됨.

## 2026-08-08 · CIFAR-100 PTM 문헌 대조 — PyCIL 실제 모델과의 비교는 SSv2가 아닌 여기서

📄 [`pycil_bridge_result.md §5`](pycil_bridge_result.md)

**질문:** SSv2 스윕을 "다른 모델들과 비교"해달라는 요청 → PyCIL을 SSv2에 직접 붙이는 방안
검토했으나 **PyCIL은 비디오 데이터셋 미지원**(DataManager가 CIFAR/ImageNet 전용)이라 큰
엔지니어링 비용 필요. 대신 **문헌 수치 대조**로 방향 잡고, 이미 CIFAR-100·PyCIL split으로
FeCAM을 검증해둔 07-22 리포트에 이어 붙임 — SSv2가 아니라 CIFAR-100이 겹치는 지점이라서.

SimpleCIL/APER([arXiv:2303.07338](https://arxiv.org/abs/2303.07338)), RanPAC
([arXiv:2307.02251](https://arxiv.org/abs/2307.02251)), ACIL
([arXiv:2205.14922](https://arxiv.org/abs/2205.14922)) 원문 PDF를 직접 받아 표 수치 확인:

| method | 트랙 | CIFAR-100 avg / last |
|---|---|---|
| SimpleCIL | frozen ViT-B/16-IN21K (우리와 동일 트랙) | 87.57 / 81.26 |
| APER w/ Adapter | 〃 + adapter 미세조정 | 90.65 / 85.15 |
| RanPAC | 〃 + PETL + random projection | — / 92.2(last만 보고) |
| **FeCAM(우리)** | **frozen CLIP B/32** | **76.9 / 73.3** |
| ACIL (다른 트랙) | ResNet-32 from-scratch backprop 후 analytic 증분 | 66.3(5-phase) |

- **같은 트랙 안에서 우리가 제일 낮다** — 단 backbone이 다르다(CLIP B/32 vs ViT-B/16-IN21K).
  `ap_fps_sweep_result.md`가 SSv2에서 "속도는 backbone이 결정"이라 낸 것과 같은 결로,
  **정확도 축에서도 backbone이 가장 레버리지 큰 선택**이라는 게 문헌으로 확인됨.
- **⭐ 세 논문 다 FPS/latency를 전혀 안 보고함** — 원문 전체 검색해도 없음. `pycil_survey_edge_gap.md`가
  CIL 분야 전반에 지적한 시간 미측정 공백이 PTM-frozen 트랙 대표 논문 3편에서도 그대로
  재확인됨. **문헌과의 속도 비교 자체가 불가능** — 이 축을 실측하는 것 자체가 이 프로젝트의
  차별점.
- ACIL은 트랙이 달라(backprop base 필요) 직접 비교하지 않음(`pycil_bridge_result.md §4`
  트랙 분리 원칙 그대로 적용).

## 2026-08-08 · ESSENTIAL과 직접 비교 — 정확도 절반 이하, 메모리는 근소 우위

📄 [`ssv2_video_access_result.md §9`](ssv2_video_access_result.md)

§8의 174클래스 TCD 84+9×10 인프라에 배포 설정 그대로(chunks4, few_shot_correction=True)
FeCAM을 얹어 ESSENTIAL(ICCV'25, 구조적으로 가장 가까운 선행연구)과 직접 비교:

| | avg_inc | last |
|---|---|---|
| ESSENTIAL (10×9, 원문) | **48.9%** | 47.5% |
| 우리 FeCAM(chunks4) | 20.8% | 17.8% |

**정확도는 우리가 절반 이하 — 스핀 없이 그대로 기록.** 반대로 **메모리는 근소하게 우리가
작다**(체크포인트 8.17MiB vs ESSENTIAL 8.4~8.6MiB). 해석: ESSENTIAL의 학습형 temporal
encoder+cross-attention이 backprop으로 실제 상당한 정보를 더 뽑아낸다는 뜻 —
`bounds_context_result.md`의 "backprop이 사는 건 1%p 미만"은 **같은 표현 위에서 선형
분류기와 비교했을 때**의 얘기고, **표현 자체를 학습형 temporal 모듈로 바꾸는 것**은
전혀 다른 축임을 이 결과가 보여준다. 한계: 84/90 무작위 분할, ESSENTIAL의 정확한
CLIP variant·프레임 수 미확인, ESSENTIAL은 sparse exemplar를 쓰는 반면 우리는 완전
0-메모리.

## 2026-08-08 · Ablation — FeCAM이 SLDA를 이기는 진짜 이유는 정규화가 아니었다

📄 [`fecam_vs_slda_covariance_result.md §4`](fecam_vs_slda_covariance_result.md)

`fecam_vs_slda_covariance_result.md`의 첫 버전은 "SLDA와 FeCAM의 차이는 공분산 정규화
방식"이라고 결론 냈는데, 사용자가 "ablation 돌려서 기여도 분리해봐"라고 요청해서 실측—
**결론이 정반대로 나왔다.**

SLDA→FeCAM 5단계 사다리(한 번에 한 요인만 변경, 실제 `FeCAMHead`에 `_cov_terms()`만
바꿔치기해 다른 코드는 100% 동일하게 유지 — 손으로 재구현했다가 원인 불명 오차 0.2~0.35pp가
나서 이 방식으로 다시 짬):

| 단계 | last acc | 기여 |
|---|---|---|
| SLDA | 14.14 | — |
| **+Tukey 변환+L2 정규화** | 15.57 | **+1.42pp (89%)** |
| +대각 shrink 강화(데이터 기반) | 15.74 | +0.17pp (11%) |
| +비대각 shrink | 15.74 | +0.00pp |
| +상관계수 정규화(=FeCAM) | 15.74 | +0.00pp |

**FeCAM−SLDA 격차의 89%가 "정규화"가 아니라 "특징 전처리"(Tukey+L2, 첫 버전이 완전히
빠뜨렸던 단계)에서 나왔다.** 상관계수 정규화는 대수적으로 **정확히 0**임을 증명함 —
`Σ_corr=D⁻¹ΣD⁻¹`의 역행렬과 `scores()`의 `x/D`·`μ/D` 스케일링이 정확히 상쇄돼, 공유
공분산 하나만 쓰는 세팅에서는 원문의 이 단계가 수학적 no-op이 됨(원문의 진짜 동기는
여러 클래스별 공분산을 서로 비교 가능하게 만드는 것 — 행렬이 하나뿐이면 적용 대상이
없음). 리포트 전체를 이 결과로 재작성함.

---

## 현재 상태 (2026-07-30 기준)

**서빙 구성:** frozen CLIP ViT-B/32 → chunks4 pooling(2048-d) → FeCAM shared-cov head
(`few_shot_correction` 켬 — base 예측에는 영향 없고 라이브 등록에만 작용).
파라미터 학습 0개, gradient 0회, replay 버퍼 없음.

**정확도 — 바닥·천장과 함께** (전부 배포본 chunks4, 최종 전-클래스 정확도)

| 벤치 | zero-shot(바닥) | **우리(class-IL)** | linear probe(천장) | 남은 여유 |
|---|---|---|---|---|
| UCF101 (101cls) | 69.13 | **88.74** | 89.21 | +0.48%p |
| SSv2 (48cls) | 5.21 | **23.56** | 24.61 | +1.04%p |

**증분 프로토콜 비용 = 0** (joint와 class-IL이 동일 — 항등식). **backprop이 사는 건 1%p 미만.**

| 그 외 | 지표 | 값 | 비고 |
|---|---|---|---|
| SSv2 48-class | task-aware 6-way | **0.498** | |
| UCF101 (TCD 프로토콜) | avg inc | 90.00 | 구 변형(chunks3+adjdiff) 기준 — 재측정 필요 |
| CIFAR-100 (PyCIL split) | avg inc | 0.769 | 문헌 SimpleCIL 0.876·RanPAC 0.922(last) 대비 낮음 — backbone 차이(§08-08) |
| 실시간 | 학습+예측 | **39.4 fps** | CPU, 10 fps 예산의 25% |
| 학습 비용 | fit only | **42 ms** | GRU+A-GEM 대비 6,400배↓ |

**우리가 아직 못 이기는 축:** S1 backward-transfer(A-GEM의 −0.061은 리허설 고유 효과),
그리고 SSv2 절대치(ESSENTIAL 48.9 vs 우리 23.6 — 게다가 프로토콜이 달라 직접 비교 불가).
**단 UCF101 88.7 중 69.1은 CLIP이 이미 알던 것**이므로 표에 zero-shot을 빼면 기여가 과대평가된다.

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
| 7/31 | base-heavy에서 "격차가 벌어진다(+0.073)" | **baseline을 굶긴 결과.** replay가 세션당 고정 50개라 36클래스 base가 클래스당 1.39개만 받았음. 클래스 비례로 고치니 **+0.045**로, 균등 split(+0.052)보다 오히려 작음. "반전 없음"은 유지 |
| 7/31 | 그 붕괴가 고정 예산 **때문**이다 | 예산을 6배 늘려도 −64.8% → −57.7%로 7.1%p만 완화. **대부분은 다른 원인**(로짓 쏠림) |
| 7/31 | 캐시 행-패치가 전체 재계산과 **bit-identical** | **BLAS 결합순서까지 같다고 본 게 과했다.** 1 ULP(3e-16) 차이가 numpy 2.5에서 드러남. `rtol=1e-12` + argmax 완전일치로 정정 |
| 7/31 | SSv2 pooling **+8.97%p** | **선택 편향 1.14%p 포함.** 변형·구간 수를 val에서 골랐음. held-out 선택 시 **+7.83%p**. 개선의 존재는 견고하나 *어느 변형이 최고인지*는 구분 불가 |
| 상시 | headline을 task-aware만 보고 | full-48-way 병기로 전환(7/12). 절대치가 훨씬 낮음 |

---

## 2026-07-31 · base-heavy 재실행 — 감사 개선점 1 해소, 주장 하나 철회
📄 [`base_heavy_split_result.md`](base_heavy_split_result.md)

07-30 자체 감사의 개선점 1을 처리했다. A-GEM이 **세션당 고정 50개** replay를 받고 있어서
36클래스 base 세션이 클래스당 1.39개로 굶고 있었다 — **연구 대상 변수가 baseline의 예산을
직접 깎는** 구조였다. 원래 비율(8.33개/class)을 유지한 채 세션 크기에 비례시켜 재실행.
`--memory fixed`가 최초 수치를 정확히 재현하므로 바뀐 변수는 예산 하나뿐이다.

Last acc 기준 Δ(FeCAM − GRU):

| | uniform | moderate | aggressive |
|---|---|---|---|
| 정정 전 | +0.052 | +0.048 | **+0.073** ❌ |
| **정정 후** | **+0.052** | **+0.054** | **+0.045** |

**철회:** *"aggressive에서 격차가 오히려 커진다(FeCAM이 GRU의 거의 2배)"* — 굶긴 baseline의
산물이었다. 공정한 예산에선 **+0.045로 균등 split(+0.052)보다 작고**, 배수도 1.87 → **1.40배**.

**유지:** 세 split 전부 FeCAM이 앞선다. TCD식 반전은 여전히 재현되지 않는다. 정정된 서사는
"격차가 커진다"가 아니라 **"base 구조를 바꿔도 격차가 대체로 일정하다(+0.045~+0.054)"** 이다.

**부수 발견 2개:**
- **굶주림 효과는 aggressive(6배)에서만 실재.** moderate(4배)는 메모리를 4배 줘도 변화가
  노이즈(0.3× pooled sd)인데 aggressive는 3.1× sd로 seed 구간이 아예 안 겹친다 →
  **임계점이 4배와 6배 사이**에 있다.
- **붕괴의 대부분은 예산 탓이 아니었다.** 클래스당 exemplar를 6배 늘려도 base 직후 붕괴가
  −64.8% → −57.7%로 7.1%p 완화되는 데 그쳤다. 나머지는 backprop 계열의 로짓 쏠림으로 보이며,
  이건 예산으로 해결되지 않는다. (07-24 리포트는 붕괴를 **전적으로** 예산 탓으로 돌렸는데
  그것도 과대평가였다.)

**방법론 교훈:** 같은 실수를 UCF101 러너에서는 피했다([`run_ucf101_gru_agem.py:112`](../dev/run_ucf101_gru_agem.py)가
클래스 비례). 비교 실험 설계 시 **"이 변수가 baseline에게 불리하게 작용하지 않는가"** 를
먼저 점검할 것.

## 2026-07-31 · SSv2 pooling 프로토콜 재검증 — 감사 2·3·4
📄 [`ssv2_temporal_pooling_result.md` §10](ssv2_temporal_pooling_result.md) ·
[`run_ssv2_pooling_protocol.py`](../dev/run_ssv2_pooling_protocol.py)

세 결함(시드 없음 / val에서 선택 / 자체 프로토콜)을 한 번에 고쳐 재실행했다.

### 헤드라인 정정: +8.97%p → **+7.83%p**

| | 선택 방식 | 고른 변형 | val last | mean 대비 |
|---|---|---|---|---|
| 최초 | **val에서 선택** ❌ | chunks3+adjdiff (D=2560) | 24.71 | +8.97 |
| **정정** | **held-out train에서 선택** | **chunks4 (D=2048)** | **23.56** | **+7.83** |

**선택 편향 = 1.14%p**(주장 효과의 약 13%). train을 층화로 fit(3600)/select(1200)로
나눠 val을 전혀 보지 않고 골랐더니 다른 변형이 뽑혔다.

### 더 중요한 결론: 상위 변형들은 구분되지 않는다

선택 단계 1위(chunks4 44.87)와 2위(chunks3+adjdiff 44.43) 차이가 **0.43%p인데 sd가 3.4%p**.
같은 클래스 순서로 짝지어 보면 **chunks4가 2/5 시드에서만 이긴다**(차이 +0.43 ± 1.14, 부호가
시드마다 뒤집힘).

→ **"chunks3+adjdiff가 최고"도 "chunks4가 최고"도 우리 데이터로는 뒷받침되지 않는다.**
방어 가능한 주장은 **"순서를 보존하는 pooling이면 어느 것이든 mean 대비 +7~9%p"** 이다.

### 개선 자체는 견고 — 두 구조 모두에서

| 프로토콜 | mean | chunks4 | Δ avg inc | 효과크기 |
|---|---|---|---|---|
| uniform 8×6 | 27.20 ± 2.16 | 34.12 ± 2.36 | +6.91 | 3.1× sd |
| **base-heavy 24+4×6** (TCD식) | 20.07 ± 0.97 | 27.73 ± 0.78 | **+7.66** | **8.7× sd** |

TCD가 쓰는 base-heavy 구조에서 오히려 효과가 크고 분산이 작다.

### 감사 2의 답이 바뀌었다 — `last`에는 시드가 무의미하다

`last`의 시드 분산이 모든 조건에서 **정확히 0**이다. 최종 헤드는 순서와 무관하게 같은
데이터를 다 보므로 같은 모델이 되기 때문. 클래스 순서는 **중간 평가 지점만** 움직여
`avg_inc`에만 영향을 준다. → 보고하는 +7.83%p는 **오차가 없는 값**이고, 유의성 검정은
`avg_inc`로 해야 한다. (07-24 UCF101의 "증분 불변 = 항등식"과 같은 성질.)

### 감사 4는 부분적으로만 해소 — 명시

TCD의 실제 SSv2 프로토콜은 **174클래스 중 84 base**인데, 우리는 **48클래스 feature뿐이고
나머지 원본 영상이 없다**(`mobileclip_result.md`와 동일 차단). 재현한 것은 **구조와 평가
관례**이지 클래스 수가 아니다. → **ESSENTIAL 48.9와는 여전히 직접 비교 불가.**

## 2026-07-31 · 하한·상한으로 우리 수치 괄호치기 — 감사 5
📄 [`bounds_context_result.md`](bounds_context_result.md) · [`run_bounds_context.py`](../dev/run_bounds_context.py)

우리 수치가 잘한 것인지 판단할 문맥이 없었다. **우리 특징 위에서** 바닥(CLIP zero-shot,
학습 0)과 천장(joint linear probe, backprop·전 클래스 한 번에·C는 held-out 선택)을 직접 계산.

| | zero-shot(바닥) | **배포본 class-IL** | FeCAM joint | linear probe(천장) |
|---|---|---|---|---|
| UCF101 | 69.13 | **88.74** | 88.74 | 89.21 |
| SSv2 | 5.21 | **23.56** | 23.56 | 24.61 |

**⭐ 증분 프로토콜의 비용이 정확히 0이다.** joint와 class-IL이 두 벤치·두 특징 전부에서
소수점까지 동일 — 측정이 아니라 **항등식**이다(순서 무관 합 → 문자 그대로 같은 모델).
→ 서사를 **"증분인데도 잘한다"에서 "증분은 애초에 비용이 없다"로** 바꿔야 한다. 이번 주에만
같은 항등식이 세 번째다(07-24 증분 불변, 07-31 `last` 시드 분산 0, 여기).
*성립 조건 실측:* 클래스가 한 세션에만 등장할 때(=class-IL 정의) 정확히 0. 세션에 걸치면
평균은 여전히 동일하고 공분산만 5.1e-3 차이(23.56 → 23.59).

**⭐ 천장이 코앞 — backprop이 사는 건 UCF101 +0.48%p, SSv2 +1.04%p.** 닫힌 형태 헤드가
학습형 선형 분류기의 99.5% / 95.7%를 42ms에 달성한다. "frozen feature 위에서 backprop은
불필요하다"는 주장의 **가장 직접적인 증거**이자, **분류기 쪽엔 개선 여지가 없다**는 뜻
(실시간 측정의 "병목은 인코더 91%"와 같은 결론에 정확도 축에서 도달). C를 1000까지 넓혀도
천장 불변이라 과소평가가 아니다.

**두 벤치에서 기여의 성격이 반대다.** 절대 이득은 비슷(≈19%p)한데 UCF101은 바닥이 69로 높아
**상당 부분이 CLIP이 이미 알던 것**이고(→ 표에 zero-shot 병기 필수, 물음표 2 오염 우려와 연결),
SSv2는 바닥이 chance의 2.5배뿐이라 우리가 **4.5배**를 만든다. **SSv2는 약점이 아니라 기여가
큰 쪽으로 재배치해야 한다.**

**chunks4 이득이 backprop 분류기에서도 재현.** SSv2 linear probe 16.80 → 24.61(+7.81)로
FeCAM(+7.82)과 거의 동일 → "표현 수준 개선"이 학습형 분류기까지 확장 검증됐다.

## 2026-07-31 · 등록 클래스와 공분산 — 감사 9
📄 [`enrollment_covariance_result.md`](enrollment_covariance_result.md) · [`run_enrollment_covariance.py`](../dev/run_enrollment_covariance.py)

**질문:** `enroll_class`가 `update_cov=False`라 등록 클래스가 공유 공분산에 영원히 기여하지
않는다. 해로운가?

**답: 아니다.** few-shot 등록(데모의 실제 경로)에서 공분산을 갱신했을 때의 이득이
**SSv2 +0.00, UCF101 +0.07~0.33**. 공유 공분산은 전역 통계라 클래스 몇 개로는 안 움직인다
(실시간 §6의 "역행렬 800프레임 미갱신 = 0.25%p"와 같은 성질). → **설계가 정당했다.**

### ⭐ 대신 훨씬 큰 걸 발견 — few-shot 등록 자체가 구조적으로 불리하다

5클립 등록 클래스 정확도가 **SSv2 0.14%**. 원인은 공분산이 아니라 **표본 수 비대칭**:

| 등록 샘플 | 5 | 10 | 25 | 50 | 100 | base |
|---|---|---|---|---|---|---|
| UCF101 | **66.4** | 80.0 | 87.5 | 90.3 | 91.9 | 92.7 |
| SSv2 | **0.14** | 0.92 | 5.72 | 13.70 | 25.10 | 33.2 |

base와 같은 수를 주면 동등해진다. **심각도는 클래스 구별성에 좌우** — UCF101은 base의 72%로
쓸 만하고, SSv2는 0.4%로 사용 불가.

### 편향에 닫힌 형태가 있고, 보정을 넣었다(opt-in)

E[(x−m̂)′P(x−m̂)] = 참거리 + **tr(P·Σ)/n**. 합성 데이터로 1/n 형태 확인. 이 항을 되돌리면:

| | enrolled | base | overall |
|---|---|---|---|
| UCF101 5-shot | 66.4 → **83.2** | −0.6 | **+2.4** |
| SSv2 5-shot | 0.14 → 9.5 | −8.5 | −3.3 |

**분리 가능하면 이득, 겹치면 base를 내준다.** 라이브러리 기본은 off이되 **배포본에서는 켰다**
(07-31) — 데모 용도가 UCF101형이고, 배포 체크포인트에서 base 예측 불변을 재확인(0/4,702).

**결정적 성질 — 균등 표본에서 정확히 no-op:** 모든 n이 같으면 전 클래스에 같은 상수를 더하므로
argmax 불변. SSv2(100/class 균등)에서 **argmax 변화 0/4,702, 비트 동일**. UCF101(72~121)도
10/3,783에 정확도 불변. → **보고된 벤치 수치가 하나도 영향받지 않는다.**

## 2026-07-31 · 메모리 실측 — 감사 10
📄 [`memory_footprint_result.md`](memory_footprint_result.md)

RAM 근거가 GRU 시절 학습 수치 `0.95 GB` 하나뿐이었다. 배포 경로를 분해:

| | RSS 증가 |
|---|---|
| CLIP vision tower (87.5M) | +259 MB |
| FeCAM 헤드 (D=2048) | +233 MB |
| **총 RSS** | **703 MB** |

**시간에서는 인코더가 91%였지만 메모리에서는 대등하다.** 헤드는 D×D 두 장(cov_sum +
precision, 67 MB)이 지배하고 **클래스 수는 거의 무관**(256슬롯 means가 4.2 MB) —
"클래스를 늘려도 안 깨진다"가 속도에 이어 메모리에서도 성립.

### ⭐ 임시 할당이 실제 배열의 5배였다 → 91 MB 절감

헤드 배열은 72.9 MB인데 RSS 증가는 388 MB였다. `_cov_terms()`가 **D×D 임시를 10장 넘게**
거쳐가고(장당 33.6 MB), 할당자가 아레나를 OS에 반납하지 않는다. in-place로 고치고,
특히 **`tr(P@S)`를 행렬곱 없이 `einsum("ij,ji->")`** 로 바꿔 D×D 할당 제거 + O(D³)→O(D²).

| | 전 | 후 |
|---|---|---|
| 헤드 로드 | 173.9 MB | 115.6 MB |
| 첫 scores() | 388.2 MB | **297.0 MB (−23%)** |

정확도 불변: 15시드 대조 precision 1 ULP(5.8e-16), 실데이터 4,702샘플 **argmax 100% 일치,
정확도 소수점 6자리까지 동일**.

### 물음표 9에 대한 답 — 우리는 SBC/로봇 급이지 MCU 급이 아니다

703 MB는 Jetson Orin·RPi5·폰 SoC에서는 되지만 **MCU급 글래스에서는 불가**. 헤드만 줄여도
CLIP 259 MB가 남는다. 논문에서 "AI 글래스"를 말하려면 이 수치를 병기하거나 대상을
로봇/SBC로 좁혀야 한다.

## 자체 감사 (2026-07-30) — 개선점 10 · 물음표 10

논문화 전에 스스로 약점을 찾아본 것. **⚙️ = 코드/데이터로 확인함**, **💭 = 판단**.

### 개선점 (고칠 수 있는 것)

| # | 항목 | 근거 |
|---|---|---|
| 1 | ~~**base-heavy 실험을 고친 baseline으로 재실행**~~ ✅ **07-31 완료** | ⚙️ 의심이 맞았다. 공정한 예산에서 aggressive 격차가 **+0.073 → +0.045**로 줄어 "격차가 벌어진다"를 철회했다(핵심 주장인 "반전 없음"은 유지). 위 07-31 항목 참조 |
| 2 | ~~**SSv2 실험에 seed 도입**~~ ✅ **07-31 완료** | ⚙️ 클래스 순서 5개로 재실행. **답이 바뀌었다** — `last`는 순서 무관이라 시드 분산이 정확히 0이고, 시드가 의미를 갖는 건 `avg_inc`뿐이다 |
| 3 | ~~**pooling·구간 수 선택을 held-out으로**~~ ✅ **07-31 완료** | ⚙️ 편향이 실재했다: **1.14%p**(효과의 13%). 헤드라인 +8.97 → **+7.83**. 게다가 상위 5개 변형은 통계적으로 **구분 불가** |
| 4 | **SSv2도 TCD 프로토콜로 평가** ◐ **07-31 부분 완료** | ⚙️ 구조(base-heavy)와 평가 관례(랜덤 순서)는 적용했고 개선이 거기서 더 견고했다(8.7×sd). **그러나 클래스 수는 불가** — 48클래스 feature뿐이고 원본 영상이 없다. ESSENTIAL 48.9와 여전히 직접 비교 불가 |
| 5 | ~~**zero-shot / linear-probe 상·하한 추가**~~ ✅ **07-31 완료** | ⚙️ 바닥 UCF101 69.13 / SSv2 5.21, 천장 89.21 / 24.61. **증분 비용 0(항등식)**, **backprop 여유 0.5~1%p**를 확인 |
| 6 | ~~**CI에서 테스트 실행**~~ ✅ **07-31 완료** | ⚙️ `tests.yml` 추가(push/PR, py3.11+3.13). **바로 값을 했다** — 격리 환경에서 테스트 1개가 실패해 과도한 비트-동일성 단정을 발견·수정 |
| 7 | ~~**raw JSON에 git SHA·환경 메타데이터**~~ ✅ **07-31 완료** | ⚙️ `src/utils/provenance.py` — 11개 writer 전부 전환. **`dirty` 플래그가 핵심**(트리가 더러우면 SHA가 코드를 특정 못 함 — 07-27이 정확히 그 경우). 기존 15개는 출처를 지어내지 않고 **'NO PROVENANCE'로 표시** |
| 8 | ~~**app.py 분해**~~ ✅ **07-31 완료** | ⚙️ HTML 781줄(파일의 53%)을 `static/index.html`로 분리 → app.py **1,470 → 700줄**. 서빙 내용 sha256 동일, 부분 배포 시 startup에서 명확히 실패 |
| 9 | ~~**등록 클래스의 공분산 문제 측정**~~ ✅ **07-31 완료** | ⚙️ **공분산은 무해**(few-shot 이득 +0.00~0.33). 대신 **few-shot 등록 자체가 표본 수 비대칭으로 불리**함을 발견(SSv2 5-shot 0.14%). 편향의 닫힌 형태 tr(PΣ)/n을 찾아 opt-in 보정 추가 |
| 10 | ~~**메모리를 상시 지표로**~~ ✅ **07-31 완료** | ⚙️ `--memory` 모드 추가. 총 703 MB(CLIP 259 + 헤드 233). **임시 할당이 배열의 5배**임을 발견해 91 MB 절감(정확도 불변). **SBC/로봇 급이지 MCU 급 아님**을 명시 |

### 물음표 (답을 모르는 것)

| # | 항목 | 근거 |
|---|---|---|
| 1 | **"증분 크기 불변"은 발견인가 항등식인가** | ⚙️ raw JSON에서 `last`가 세 증분 전부 **비트 단위 동일**(`0.8744382765001322`) — 순서 무관 합이라 최종 모델이 문자 그대로 같다. 리포트는 "구조적 성질"이라 정직하게 썼지만, TCD −2.70 / ESSENTIAL −1.80과 한 표에 놓으면 경쟁 우위로 읽힌다. 게다가 그들은 backbone을 학습하는 **다른 트랙** — PyCIL 비교에 붙인 트랙 주석이 여기도 필요 |
| 2 | **UCF101 88.84에 CLIP 사전학습 오염은?** ◐ 정량화됨 | ⚙️ 07-31: **zero-shot만으로 69.13** — CLIP이 UCF101을 이미 상당히 안다. 오염 여부와 무관하게 **표에 zero-shot 병기 필수**. SSv2는 5.21(chance 2.08)로 대비됨 |
| 3 | **ESSENTIAL 95.1과 "2위" 비교가 성립하나** | 💭 그쪽은 frozen CLIP + **학습형** temporal encoder + prompt, 우리는 학습 0. 같은 표에 두려면 학습 예산 열이 필요 |
| 4 | ~~**"39fps 실시간 학습"이 정확도로 이어지나**~~ ◐ **08-04 재정의·부분 완료** | ⚙️ 질문의 전제가 틀렸었다 — `/enroll`은 파일 업로드고 curated 샘플링을 쓰므로 **등록은 이미 벤치마크와 동일 경로**. 진짜 차이는 **쿼리**(`/predict_rt`의 200ms/16프레임 ring buffer)였고, 이는 base 클래스에도 적용됨. 실측: base **−1.5~2.1pp**(3시드 전부 일관, 노이즈 아님), enrolled −1.3~2.0pp(표본 작아 노이즈 큼). 상세: [`live_query_sim_result.md`](live_query_sim_result.md) |
| 5 | **합성 프레임으로 잰 fps** | ⚙️ CLIP 인코딩은 입력 무관이라 무해하지만 카메라 디코딩·리사이즈가 빠져 있다 — 종단 주장인지 부분 주장인지 애매 |
| 6 | **SSv2 나머지 18pp가 정말 학습형 encoder 몫인가** | 💭 23.6 vs ESSENTIAL baseline 42.1의 격차를 그렇게 귀속했으나 **우리 큐레이션·클래스 수**와 분리 안 됨. 07-31에 구조는 맞췄지만 **클래스 수는 여전히 불가**(48 vs 174) — 미해결 |
| 7 | **48-class 큐레이션의 대표성** | 💭 8 stages × 6개를 사람이 골랐다. 랜덤 48개와 비교한 적이 없어 큐레이션이 결론을 만들었는지 알 수 없다 |
| 8 | **T=16의 5/5/6 비대칭이 메커니즘 설명과 어긋난다** | ⚙️ 개선 근거는 "역재생하면 차분 부호가 뒤집힌다"인데 T=16은 구간이 5/5/6이라 **정확히 성립하지 않는다**(테스트 작성 중 발견, T가 3의 배수일 때만 성립). 왜 16인지 재검토 필요 |
| 9 | **체크포인트·D² 공분산이 "스마트글래스"와 맞나** ◐ 정량화됨 | ⚙️ 07-31: 총 **703 MB**(CLIP 259 + 헤드 233). RPi5·폰 SoC까지는 가능, **MCU급 글래스는 불가**. 대상을 로봇/SBC로 좁히거나 수치를 병기해야 한다 |
| 10 | **우리 고유 기여가 무엇인가 — 가장 중요** | 💭 이득의 대부분은 "frozen feature + 통계 head"에서 오는데 그건 **SimpleCIL/RanPAC/FeCAM(2023~24)이 이미 확립**한 것이고 우리는 재현·검증했다. 고유 기여 후보는 **닫힌 형태 temporal pooling**(비디오 CIL에서 backprop 없이 시간 축을 살린 것)인데 아직 **단일 실행·자체 프로토콜**이다 |

**우선순위:** 개선점 **2·3·4를 묶어서** — SSv2를 TCD 프로토콜에 seed 3개로 올리고 pooling
선택을 held-out으로 옮긴다. 이것이 물음표 10(고유 기여)을 직접 떠받치고, 물음표 6도 같이 푼다.
개선점 1은 반나절이면 되고 **기존 결론을 지키는** 일이라 그다음.

## 열린 과제

**막힌 것**
- **HMDB51** — 공식 split 서버가 HTML을 반환, HF 미러는 영상만. split을 지어낼 수 없어 보류.
- ~~**인코더 교체**~~ — MobileCLIP은 CPU에서 20배 느림. **Jetson 대여로 검증 경로가 생김**(아래).
- ~~**SSv2 원본 영상**~~ — **08-06 해결.** Qualcomm 공식 배포처에서 19.4GB 직접 다운로드
  (`COAD_VIDEO_DIR` 설정 완료, 220,847개 webm 전량 확보). 프록시가 우려했던 coverage 0.25
  근처 역전은 실측 결과 **일어나지 않음**(실제 coverage=0.664, chunks4가 live 조건에서도
  mean보다 +2.91pp 앞섬) — [`ssv2_video_access_result.md`](ssv2_video_access_result.md).
  174클래스 전체·84 base + 9×10 incremental(TCD 리터럴 스케일)로 FeCAM vs GRU+A-GEM도
  재검증 완료 — FeCAM이 3시드 전부 승리(avg_inc +3.78pp), 48클래스 근사 결론이 그대로
  유지됨 ([`ssv2_video_access_result.md §8`](ssv2_video_access_result.md)).
  ([`partial_window_sim_result.md`](partial_window_sim_result.md)). 복구되면 최우선 재현 대상.

**⭐ Jetson AGX Orin 대여 가능 — 막힌 항목 2개가 동시에 풀린다**

보드가 손에 들어오면 아래 두 가지가 한 번에 가능해진다. 우리 벤치
(`dev/bench_realtime_incremental.py`)는 플랫폼 의존 코드가 없어 **그대로 돌아간다**
(numpy/torch/PIL만 사용, `--device` 스위치 존재).

1. **임베디드 절대 fps** — 지금 수치는 Apple Silicon 1대 기준이라 절대값 근거가 없다.
   Orin의 CPU-only 모드가 우리 주장(CPU-only edge CL)에 정확히 대응하는 숫자다.
2. **MobileCLIP 가설 검증** — 7/16·7/27 결론은 *"MobileCLIP이 느린 건 depthwise conv에
   최적화 커널이 없어서지 모델이 나빠서가 아니다"* 였고, 이건 **NPU 타깃 런타임이 없어
   검증 불가**였다. Orin + TensorRT가 바로 그 런타임이다. 가설이 맞으면 역전이 나와야 한다.
3. **에너지(mJ/update)** — Orin은 `tegrastats`/INA3221 전력 레일을 노출한다. CIL 분야는
   시간조차 안 재는 상황(위 감사)이라 **에너지를 보고하면 사실상 최초**다. 보드가 있을 때만
   얻을 수 있는 수치이므로 대여 기간에 반드시 뽑을 것.

⚠️ **프레이밍 주의:** AGX Orin은 15~60W 로봇/자율주행 모듈이지 **스마트글래스 급이 아니다.**
우리 서사의 "로봇" 쪽은 뒷받침하지만 "AI 글래스"는 뒷받침하지 못한다. 글래스/폰 급 주장을
하려면 별도로 폰 CPU(SparCL이 쓴 Galaxy S20 계열)나 RPi가 필요하다.

📌 **대여 전 준비물:** 보드 시간을 디버깅에 쓰지 않도록, 한 번 실행하면 위 3개를 전부
JSON으로 떨구는 단일 스크립트를 미리 만들어 둘 것.

**미실행 (우선순위 순)**
1. ~~**공분산 갱신 주기 튜닝**~~ — **08-08 해소.** 배포 pooling(D=2048/2560) 4개 조합
   전부 실측: 정확도 손실 1pp 미만(−0.37~+0.62pp), 속도는 12.5~17.8배 향상 — D=512에서
   냈던 결론이 실제 배포 차원에서도 유지됨. `never` 운용 시 인코더 포함 17~23fps로 예산
   충분. ([`ssv2_video_access_result.md §10`](ssv2_video_access_result.md))
2. **차원 축소와 결합** — 2560-d 공분산은 크다. random projection/PCA로 줄이면 더 많은 구간을 쓸 수 있을지도.
3. **AUC-A / AUC-L 채택** — PyCIL 서베이의 메모리-불가지론 지표.
4. **BudgetCL 방식의 iteration-budget 프로토콜**.
5. **학습형 temporal encoder를 선택적으로** — backprop-free 주장은 포기하되 상한 확인용.

---

## 원본 리포트 색인

날짜순. 위 요약보다 상세한 표·해석·재현 명령은 각 원본에 있다.

### 실험이 아닌 문서 (위 본문에 항목이 없는 4개)

발표·보고용으로 만든 것들이라 실험 항목으로는 들어가 있지 않다. 내용은 전부 위 실험
항목에서 파생됐으므로 **수치의 출처는 항상 해당 실험 리포트**이고, 아래는 그 재구성이다.

| 파일 | 날짜 | 무엇 |
|---|---|---|
| [meeting_prep_todo_status.md](meeting_prep_todo_status.md) | 07-13 | 교수님 미팅 준비 TODO 7항목의 완료 상태 추적 |
| [meeting_script_0726.md](meeting_script_0726.md) | 07-26 | 미팅에서 **말로** 설명하기 위한 발표 스크립트 |
| [meeting_deck_notion.md](meeting_deck_notion.md) | 07-27 | Notion 업로드용 발표 페이지(그래프 포함) |
| [progress_since_2026-07-17.md](progress_since_2026-07-17.md) | 07-27 | 7/17 이후 진행사항만 모은 요약 |

⚠️ 이 4개는 **작성 시점에 고정**돼 있어 7/30 작업(SSv2 pooling, 실시간 측정, 배포)이
반영돼 있지 않다. 현재 수치는 [§ 현재 상태](#현재-상태-2026-07-30-기준)를 볼 것.

### 실험 리포트

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
| 07-31 | [bounds_context_result.md](bounds_context_result.md) · [enrollment_covariance_result.md](enrollment_covariance_result.md) · [memory_footprint_result.md](memory_footprint_result.md) |
| 08-04 | [live_query_sim_result.md](live_query_sim_result.md) |
| 08-05 | [partial_window_sim_result.md](partial_window_sim_result.md) |
| 08-06 | [ssv2_video_access_result.md](ssv2_video_access_result.md) · [ap_fps_sweep_result.md](ap_fps_sweep_result.md) |
| 08-08 | [ssv2_head_curves_result.md](ssv2_head_curves_result.md)(스크립트는 08-02 작성, 리포트는 뒤늦게 정리) · [fecam_vs_slda_covariance_result.md](fecam_vs_slda_covariance_result.md) |
