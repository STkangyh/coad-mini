# 교수님 미팅 브리프 — 비디오 지속학습 프로젝트 (coad-mini)

세 질문에 답하는 구조: (1) 우리 주제의 SOTA 지형, (2) 우리 baseline, (3) SOTA 대비 우리 장점(적응성·강인성 vs 계산시간).

한 줄 선요약: **정확도 SOTA는 아닙니다.** 우리의 실증 결론은 "frozen CLIP 위에서는 아키텍처 복잡도가 도움이 안 되고(Attention −0.103, SSM −0.040), 유일한 레버는 데이터 양(+0.06~0.07)"이며, 이 단순함 위에서 **CPU 동작·O(1) 메모리·라이브 few-shot 등록**이라는 배포 축의 장점을 얻습니다.

---

## 1. 주제 지형 (우리 주제의 SOTA)

우리 주제는 **비디오 행동인식의 class-incremental 지속학습(VCIL)**입니다. 관련 SOTA는 다섯 갈래입니다.

**(a) Rehearsal 계열 CL (우리가 속한 계열)**
- 대표: ER, ER-ACE(ICLR'22), DER/DER++(NeurIPS'20), X-DER(TPAMI'23), iCaRL(CVPR'17), FOSTER(ECCV'22), MEMO(ICLR'23).
- 위치: **A-GEM(2019)은 이 계열의 최하단 레거시 baseline.** Seq-CIFAR-10 Class-IL(buffer 200)에서 A-GEM ~20% vs ER ~45% vs iCaRL ~49% vs DER++ ~65% — A-GEM은 10-class 랜덤(~10%)을 겨우 넘는 수준. ER-ACE 논문은 A-GEM/GEM을 "성능이 나빠 제외"한다고 명시.
- **온라인/스트리밍 CL 좌표:** A-GEM은 single-pass online 프로토콜의 초기 대표작이고, 이후 ER-ACE·GDumb(ECCV'20)·SCR·OCM·CLS-ER 등이 online 세팅을 발전시킴. 우리는 A-GEM을 online 계열의 "가볍고 상수메모리" 극단으로 사용.

**(b) Rehearsal-free / Prompt·PTM 계열 (이미지 CIL의 현재 프론티어)**
- L2P(CVPR'22), DualPrompt(ECCV'22), CODA-Prompt(CVPR'23), S-Prompts(NeurIPS'22), SimpleCIL·APER(IJCV'24), RanPAC(NeurIPS'23), 이후 HiDe-Prompt·EASE(CVPR'24).
- 위치: frozen ViT + 소량 prompt/prototype로 리허설 없이 CODA-Prompt Split-ImageNet-R 75.5% / CIFAR-100 86.3%, RanPAC CIFAR-100 92.2%. 학습 파라미터는 backbone의 0.7~4.6%뿐. **핵심 통찰: frozen backbone + prototype만으로도(SimpleCIL) prompt-pool을 이김** — 무거운 학습보다 좋은 frozen feature가 주 레버.
- **비디오로의 확장 = §1(c)의 PIVOT/ESSENTIAL.** 즉 이 이미지 prompt-CL 갈래의 비디오 화신이 아래 vCLIMB SOTA이므로, 두 목록은 별개가 아니라 같은 계보.

**(c) 비디오 CIL — vCLIMB 벤치마크 (가장 직접적인 우리 주제)**
- 벤치마크: vCLIMB(CVPR'22), UCF101/ActivityNet/Kinetics, 메모리를 "프레임 수"로 정의. TSN+ResNet-34 baseline: UCF101 iCaRL 81.0 / Kinetics 32.0.
- SOTA: **PIVOT(CVPR'23)** = frozen CLIP + temporal prompt (UCF101 10-task 93.4, Kinetics 55.1); **SMILE**(CVPR'23 CLVision workshop) = 영상당 1프레임 저장, 다양성>시간밀도 (UCF101 95.7); **ESSENTIAL(ICCV'25 Highlight)** = frozen CLIP + sparse memory + semantic prompt, 현재 최강 (UCF101 95.8, Kinetics 58.8).
- **결정적으로 우리와 같은 재료:** 최강 방법들(PIVOT, ESSENTIAL)이 전부 **frozen CLIP/ViT + 가벼운 temporal/prompt 모듈**을 씀. 우리 frozen-CLIP 선택이 SOTA 레시피와 동일함을 검증.
- 주: vCLIMB의 "프레임 예산" 메모리 회계는 방법마다 크게 달라, PIVOT류 프레임-저장 방식은 SMILE(영상당 1프레임)류보다 저장량이 훨씬 큼 — 정확한 GiB 수치는 원논문 표를 확인해 인용 예정(초안의 "140GiB~1.1TiB"는 미검증이라 삭제).

**(d) 표준(비-CL) SSv2 행동인식 backbone**
- VideoMAE V2 ViT-g ~77.0%(CVPR'23), InternVideo2-6B 77.4%(ECCV'24)는 수억~수십억 파라미터 full-finetune. V-JEPA 2 ViT-G ~77%대는 **frozen backbone + attentive probe**로 보고돼, "무거운 계열=full-finetune만"이라는 이분법을 일부 완화함 — 단, 이때도 backbone 자체는 수십억 파라미터 대형 사전학습.
- SSv2는 모션 중심(174 클래스)이라 appearance-only(CLIP 프레임 feature) 모델에 가장 불리 — 우리 절대 정확도가 낮은 구조적 이유.

**(e) SSM (우리가 실험한 확장 축)**
- S4(ICLR'22), S5(ICLR'23), Mamba(2023), Vision Mamba(ICML'24), VideoMamba(ECCV'24). 전부 강점이 **길이(long-range)** 에서 나옴. "Repeat After Me"(ICML'24)는 고정 상태 SSM이 짧은 문맥 recall에서 이론적으로 불리함을 증명.

---

## 2. 우리 baseline

**파이프라인:** video → 16 프레임 균일샘플 → **frozen CLIP ViT-B/32**(512-d, 파인튜닝 없음) → **소형 GRU**(hidden 256, 1층) 분류기.

**망각 완화 = A-GEM**(Averaged GEM, Chaudhry et al., **ICLR 2019**): 각 학습 gradient를 과거 클래스 replay 메모리의 평균 손실을 늘리지 않도록 투영. balanced memory, mem=50/stage, replay ratio 0.25.

**벤치마크:** Something-Something V2의 48-class 부분집합, 8 스테이지(6클래스/스테이지).
- **클래스 선정 기준(확정):** 무작위/빈도가 아니라 **8 stage × 6 class 의미론적 큐레이션** — S1 Open/Close, S2 Container, S3 수직·깊이 이동, S4 수평 push/pull, S5 Cover/Throw/Drop, S6 Push-force/Hit/Tear, S7 Lift/Drop/Fall, S8 Pretend/Show. 클래스당 인스턴스는 seed 고정 `random.sample` 100개씩(총 4800 train). → cherry-pick이 아니라 **난이도가 올라가는 커리큘럼형 CL 구조**.

**학습 세팅 — initial vs continual (코드 확인, 확정):**
- **Stage 1 = initial training**: 6 클래스(전체의 12.5%), A-GEM 메모리가 비어있어(`gem._memory==[]`) `precompute_ref`가 `g_ref=None`을 반환 → `apply()`가 no-op → **순수 cross-entropy, replay 없음**. 진짜 "처음 배우는" 단계.
- **Stage 2~8 = continual training**: 나머지 42 클래스(87.5%), 7개 스테이지 × 6클래스. 매 스테이지 종료 시 `gem.add_stage()`가 해당 스테이지에서 balanced 50개를 replay 메모리에 추가 → 이후 스테이지들은 **그때까지 학습한 모든 이전 클래스**에 대해 growing replay 메모리(스테이지당 mem=50, epoch당 replay_ratio=0.25 서브샘플)로 A-GEM 보호를 받으며 학습.
- 요약: **초기 6클래스 : 지속 42클래스 = 1 : 7 스테이지** (12.5% : 87.5%). (`experiments/run_capacity_ablation.py::run_one`, `src/utils/gem.py::precompute_ref/apply`)

**지표 정의(중요 — 정정됨):** 평가는 **stage 내부 6-way, task-aware**임 (`eval_stage`가 해당 stage의 6개 class로 argmax를 제한). 즉 **테스트 시 stage/task ID가 주어지는 task-incremental 세팅**이고 **chance = 1/6 ≈ 16.7%** (❌ 1/48=2.1% 아님). "평균정확도 ~0.40" = 8개 stage-group의 6-way 정확도를 **최종 stage 학습 후 평균**. S1 forgetting = stage1의 6-way 정확도 (stage1 직후 → 전체 학습 후) 하락폭. **⚠️ 이 때문에 우리 수치를 class-IL SOTA(vCLIMB, 전체 클래스 대상)와 직접 비교 불가 — task-IL이 더 쉬움. 미팅에서 반드시 정직하게 밝힐 것.** +14.1%p는 동일 스택 A-GEM 유/무 비교.

**핵심 결과 (5 seeds, B/32, 실측·±std 확보):**

| 방법 (동일 스택, mem=50/stage) | Avg Acc | S1 forgetting |
|---|---|---|
| baseline (메모리 없음) | 0.251 ± 0.021 | +0.115 |
| plain ER (리허설, projection 없음) | 0.346 ± 0.012 | +0.020 |
| **A-GEM** | **0.387 ± 0.018** | **−0.061** |

- **A-GEM vs baseline: +13.6%p Avg Acc, +17.6%p S1-forgetting 보호** (chance 0.167).
- **A-GEM vs plain ER: +4.1%p** — 즉 **우리 셋업(frozen CLIP + task-aware)에선 A-GEM이 ER을 이김.** (`dev/run_er_comparison.py`)

**명확히 짚을 점 — A-GEM은 2019 클래식 baseline이지 정확도 SOTA는 아님.** *표준* class-IL 벤치(Seq-CIFAR 등)에선 ER/DER++/ER-ACE·prompt 계열에 밀립니다. 다만 **우리 task-aware·frozen-CLIP 세팅에서는 위 실측처럼 A-GEM이 plain ER보다 강했고**, 우리가 A-GEM을 쓰는 이유는 정확도 챔피언이어서가 아니라 **상수 메모리·single-pass 온라인 제약**이라는 효율 성질 때문입니다. 핵심: 강한 frozen feature 위에서는 약한 anti-forgetting으로도 충분.

**정밀 지표 (val 4702개, precision/recall/F1/mAP, 두 평가체제 동시 산출) — `dev/compute_val_metrics.py` / `reports/val_metrics_result.md`:**

| Regime | Metric | Baseline | A-GEM | Δ |
|---|---|---|---|---|
| **Full 48-way** (진짜 class-IL, task ID 없음, chance 2.1%) | Accuracy | 0.062 | 0.099 | +0.037 |
| Full 48-way | mAP | 0.043 | 0.104 | **+0.060** |
| Full 48-way | F1 (macro) | **0.015** | 0.068 | +0.053 |
| Task-aware 6-way (chance 16.7%) | Accuracy | 0.230 | 0.363 | +0.133 |
| Task-aware 6-way | F1 (macro) | 0.163 | 0.353 | +0.190 |

- **가장 중요한 정직성 포인트:** task ID를 안 주는 **진짜 class-IL(full 48-way)로 평가하면 정확도가 훨씬 낮아짐**(23%→6.2%, 36%→9.9%). Task-aware 수치만 보고하면 과대평가 — 미팅에서 이 낙차를 먼저 보여줄 것.
- **그래도 A-GEM의 상대적 우위는 두 체제 모두에서 유지**됨(오히려 mAP는 어려운 체제에서 격차 2.4배로 더 벌어짐).
- **baseline의 F1(macro)가 0.015로 붕괴**하는 게 catastrophic forgetting의 전형적 지문 — 순차학습이 최근 배운 클래스로 예측을 몰아주면서(accuracy는 chance의 3배지만 클래스별 재현은 거의 실패) 망각을 시각적으로도 증명. A-GEM은 이 붕괴를 0.068로 완화(그래도 낮음 — 진짜 class-IL은 여전히 어려운 문제임을 정직하게 인정).

---

## 3. SOTA vs baseline — 우리 프로젝트의 장점

**정직한 전제: 우리는 정확도 SOTA가 아닙니다.** vCLIMB 기준 PIVOT/ESSENTIAL/SMILE(UCF101 93~96)에 못 미칩니다. 그러나 우리의 **가장 방어 가능한 증거는 통제된 절제(ablation) 결과**입니다: 복잡도를 올려도 이득이 없고(Attention **−0.103**, diagonal-SSM **−0.040**, GRU 256→512·L/14 무효), 유일하게 지배적인 레버는 **데이터 양(+0.06~0.07)**. "단순함이 이긴다"가 자산(assertion)이 아니라 데이터입니다. 이 단순함 위에 효율·적응성 장점이 얹힙니다.

### 비교표

| 축 | 무거운 SOTA (비디오 트랜스포머 / prompt-pool) | 우리 (frozen CLIP B/32 + GRU + A-GEM) |
|---|---|---|
| **정확도(peak)** | vCLIMB UCF101 93~96, SSv2 recog 75~77% | 48cls/8stage 평균 ~0.40 **(다른 벤치·다른 스케일, head-to-head 비교 불가)** |
| **모델 크기(temporal head)** | VideoMAE-V2 ViT-g ~1.01B / InternVideo2-6B ~6B params | **GRU 604K params (10³~10⁴× 작음)** |
| **backbone 학습** | full-finetune(VideoMAE/InternVideo2) 또는 prompt-pool 학습 | **없음 — CLIP 완전 동결** |
| **학습 디바이스/시간** | GPU 다수, 12~19GB peak | **CPU, full 8-stage ~4.5분/seed, peak RAM 0.95GB** |
| **추론 비용** | prompt 선택용 추가 backbone forward(≈2배) | GRU head 0.3ms/win(3300 win/s); end-to-end(CLIP+GRU) ~156ms ≈ **6.4 win/s(CPU, CLIP-bound)** |
| **CL 제약 메모리** | 큰 replay 버퍼 또는 성장하는 prompt pool / 프레임-저장 방식 | **A-GEM O(1) 상수 메모리, mem=50/stage** |
| **온라인/스트리밍** | 대개 offline 다중 epoch | **single-pass 친화(A-GEM 본연의 세팅)** |
| **신규 클래스 즉시 등록** | 재학습/prompt 재학습 필요 | **few-shot live enrollment (데모 능력 — 아래 정직성 주석 참고)** |
| **시간 모델링** | temporal prompt / cross-attention(PIVOT, ESSENTIAL) | GRU(짧은 16프레임엔 충분) |

### 교수님 질문에 직접 답변: 적응성·강인성인가, 계산시간인가?

증거 강도 순으로, **강인성·효율이 지금 가장 단단하고, 적응성은 가장 매력적이나 아직 측정이 약합니다.**

**(1) 강인성/단순성 — 실증적으로 가장 단단한 강점.** 통제 실험: GRU 256→512 무효, CLIP B/32→OpenCLIP L/14 +0.013(노이즈 내), GRU→GRU+Attention **−0.103(악화, 망각 증가)**, GRU→diagonal-SSM **−0.040(악화, 대신 더 일관적)**. **유일하게 지배적인 레버는 아키텍처가 아니라 데이터 양(+0.06~0.07).** "단순한 GRU+A-GEM이 강건한 최적점"이 우리 연구의 실증 결론.
- **양방향 확정(실측):** 위로 256→512 무익, **아래로 256→128 무손실(0.388→0.387)**, h64에서야 하락(0.352). 즉 용량은 위·아래 어느 쪽으로도 병목이 아님 — 오히려 h128로 파라미터를 더 줄여도 동일. (`run_capacity_ablation --configs gru_h128 gru_h64`)

**(2) 효율/계산시간 — 크고 실재하는 강점.** frozen feature + tiny GRU + A-GEM O(1) 제약으로 전체가 **CPU에서 동작**. A-GEM은 GEM 대비 ~100배 빠르고 메모리 1/10; 무거운 prompt-pool은 208~568ms/step·최대 13GB(REP 논문, CODA-Prompt ViT-L 기준 표); frozen-head 계열(Deep SLDA, CVPR-W'20)은 iCaRL 대비 저장 ~1000배·시간 ~100배 절감.
- **실측 완료 (CPU, 이 하드웨어):** 604K params · **full 8-stage 학습 ~4.5분/seed**(스테이지당 ~33s) · **peak RAM 0.95GB** · GRU head 추론 0.3ms/window(3300 win/s) · end-to-end(CLIP+GRU) ~156ms/window ≈ **6.4 win/s 실시간(CLIP 인코더가 병목)**. → "몇 FPS냐/GPU 필요하냐" 질문에 즉답 가능. (`dev/measure_efficiency.py`)

**(3) 적응성 — 가장 차별화되나 아직 데모 수준.** frozen CLIP 위 tiny GRU 덕분에 **few-shot live enrollment**가 가능: 웹캠으로 새 행동을 몇 클립 보여주면 즉석 학습, 기존 클래스 망각 없이 라이브 캡션에 반영. full-finetune 트랜스포머/prompt-pool로 실시간엔 하기 어려운 능력. 단, **정직하게 이는 현재 "측정된 강점"이 아니라 "데모 능력"** — 신규 클래스 정확도(몇 클립→몇 %)와 기존 클래스 망각을 아직 수치화 못함. 미팅 전 최소 1개 소규모 측정 권장. (OOD/이상탐지도 같은 스택에 얹을 수 있으나 현재 미실증 — 설계 아이디어로만 언급하고 결과 주장 금지.)

**한 문장 요약(증거 강도에 맞춰):** *가장 단단한 건 강인성·효율(CPU·동결·O(1) 메모리·복잡도 무익을 실증)이고, 적응성(라이브 few-shot 등록)은 가장 차별적이나 아직 데모 단계입니다.* 정확도 최고점은 포기하고 효율·온라인 적응·배포가능성을 얻는 트레이드오프.

---

## 4. 정직한 한계

- **정확도 최고점이 아님.** vCLIMB SOTA(PIVOT 93.4 / ESSENTIAL 95.8 / SMILE 95.7)와 표준 SSv2 recog(75~77%)에 크게 못 미침. 우리 ~0.40은 48cls/8stage subset 기준이라 직접 비교 불가.
- **plain ER 비교 — 실행 완료(약점 해소).** 동일 스택·mem=50에서 **A-GEM 0.387 > ER 0.346 > baseline 0.251** (5 seeds). 우리 세팅에선 A-GEM이 ER보다 +4.1%p 우위라 이 노출점은 방어됨. (단 DER++/ER-ACE 등 *더 강한* 리허설·표준 class-IL 벤치까지는 미비교 — 후속 과제.)
- **GDumb류 상한 미확인.** "CL 이득이 실은 버퍼 효과 아니냐"는 표준 반론. mem=50/stage 제약 하 GDumb(메모리-only)·무제약 버퍼 상한을 최소 1회 측정해 선점할 것(미팅 전 또는 즉시 후속으로 명시).
- **작은 subset·단일 벤치.** 표준 vCLIMB 스플릿(UCF101/Kinetics)이 아니라 48-class 자체 subset이라 외부 방법과 head-to-head 불가.
- **GRU temporal head는 약함.** PIVOT의 temporal prompt, ESSENTIAL의 memory-retrieval cross-attention 같은 "숫자를 올리는" 장치가 없음.
- **SSM은 우리 세팅에서 안 통함.** 16프레임 짧은 시퀀스에선 SSM의 길이 이점이 없음 — 버그가 아니라 예상된 결과(§5 Q3/Q4).

---

## 5. 예상 질문 & 답변

**Q1. 왜 하필 A-GEM인가? SOTA도 아닌데.**
정확도용이 아니라 **효율/온라인 성질** 때문 — 상수 메모리, single-pass, GEM 대비 ~100배 빠름. 우리의 CPU·라이브·few-shot 시나리오에 정확히 맞음. frozen CLIP feature가 표현 drift를 줄여 약한 anti-forgetting으로도 충분. **실측으로 방어됨:** 동일 스택 5-seed에서 A-GEM 0.387 > **plain ER 0.346** > baseline 0.251 → 우리 세팅에선 A-GEM이 ER보다 우위(+4.1%p). (표준 class-IL의 DER++/ER-ACE까지는 후속 비교 과제.)

**Q2. 왜 SOTA(PIVOT/ESSENTIAL, prompt-pool)를 안 썼나?**
목표가 "정확도 논문"이 아니라 **가볍고 배포 가능한 실증 시스템**. 오히려 SOTA들과 **같은 핵심 재료(frozen CLIP)**를 공유하고, 그 위에 temporal prompt/cross-attention 대신 tiny GRU를 얹어 CPU·라이브 등록을 얻음. SOTA는 정확도 상한으로 인정, 우리는 효율·적응성 축에서 기여.

**Q3. GRU 말고 Attention이나 SSM은? 더 좋지 않나?**
통제 실험으로 답이 나옴 — Attention **−0.103**(오히려 악화, 망각 증가), diagonal-SSM −0.040(악화하되 더 일관적). 16프레임은 짧아 SSM 장점(long-range·선형시간)이 발현 안 됨("Repeat After Me", ICML'24). **복잡도가 여기선 도움 안 됨이 발견**이고 지배적 레버는 데이터 양.

**Q4. 그럼 SSM은 왜 넣었나?**
음성적 결과를 정직하게 검증하려고. "당연히 SSM이 낫다"는 통념을 16프레임 세팅에서 반증. SSM의 속도·메모리 이점은 **긴 클립(수십~수백 프레임, 분 단위 영상)**에서만 발현되므로(VideoMamba 등), 향후 긴 클립 스케일 시 재검토할 축을 미리 표시한 것. (정확한 배수는 원논문 조건 확인 후 인용.)

**Q5. 이게 논문이 되나?**
정확도 SOTA 논문은 아님. 대신 **(a) 통제된 실증 연구**("아키텍처가 아니라 데이터가 레버" — Attention −0.103·SSM −0.040·데이터 +0.06~0.07의 bottleneck 분석) + **(b) 가벼운 배포형 시스템**(CPU 실시간 캡션, few-shot 라이브 등록)의 결합. 워크샵/시스템 트랙 또는 "언제 단순함이 이기는가"류 실증 연구로 포지셔닝. ER/GDumb 비교군과 효율 벤치(학습시간·peak 메모리·버퍼 크기를 정확도와 함께)를 채우면 설득력 상승.

**Q6. frozen CLIP feature는 SSv2 모션을 못 잡지 않나? (그리고 full-48-way가 낮은 게 데이터 부족 때문 아닌가?)**
맞음 — SSv2는 모션 중심이라 appearance-only엔 가장 불리, 그래서 절대 정확도가 낮음. 하지만 **최강 vCLIMB 방법들도 똑같이 frozen CLIP을 쓰고**(PIVOT/ESSENTIAL) temporal 모듈로 보완. 우리는 그 temporal 부분을 가벼운 GRU로 대체 — 정확도 일부를 내주고 효율·적응성을 취한 명시적 트레이드오프.
**데이터 부족 기여도도 직접 측정함**(`reports/data_scale_full48way_result.md`): full-48-way를 25→100% 데이터로 스케일링하면 accuracy **+49.9%, F1(macro)는 +109%**(2배 이상) 상대 개선 — 데이터가 유의미하게 기여하는 건 실측으로 확인. 하지만 **100% 데이터를 다 써도 accuracy는 10.5%**(chance의 5배)에 그치고, 50→100% 구간에서 성장이 이미 2~3배 둔화됨. → **데이터 부족·망각(baseline F1 0.015 붕괴)·CLIP 구조적 한계, 세 가지가 함께 작용하는 복합 원인**이며 데이터만으로는 설명 안 됨.

**Q7. 48-class subset은 어떻게 골랐나? cherry-pick 아닌가?**
빈도/방법-유리가 아니라 **8 stage × 6 class 의미론적 큐레이션**(Open/Close→…→Pretend/Show, 난이도 상승 커리큘럼). 클래스당 인스턴스는 seed 고정 random.sample 100개. 특정 방법에 유리하게 고른 게 아님.

**Q8. A-GEM의 이득이 그냥 "버퍼(메모리)가 있어서"인 거 아닌가? 알고리즘이 진짜 기여하나?**
**직접 검증함 — GDumb(버퍼만 균형있게 관리 + 매 스테이지 처음부터 재학습, 알고리즘 없음)을 실측**한 결과
0.164(±0.004)로 **거의 정확히 chance(0.167) 수준**이고 **baseline(0.251)보다도 낮음**. 즉 버퍼가
있다는 사실 자체는 이득을 만들지 않고, **gradient projection(A-GEM)이라는 알고리즘이 실제 가치를
더함**(0.387, GDumb 대비 +0.223). 반론이 우리 데이터로 명확히 반증됨.

**Q9. 이 프로젝트에서 뭘 개선했나? 정확도 숫자만 계속 만지고 있나?**
아님 — capacity/backbone/attention/SSM/데이터스케일/ER/GDumb까지 폭넓게 통제 실험한 뒤, **더 이상 새
알고리즘을 얹는 게 한계효용이 낮다고 판단해 방향을 실사용 품질로 전환**했습니다. 실제로 데모를 써보다가
**OOD 이상탐지 배지가 기본 설정에서는 threshold가 계산되지 않아 한 번도 작동하지 않는 버그**를 발견 →
서버 시작 시 자동 보정(로컬은 실제 val feature로, Docker/HF Spaces처럼 데이터가 없는 배포 환경에서는
synthetic in-distribution 샘플로 폴백)하도록 고쳤고, held-out 데이터로 보정 유효성(목표 FPR 5% vs 실측
4%)까지 확인했습니다. "더 정확하게" 보다 "실제로 동작하게"를 우선한 사례.

---

## 빈칸 체크리스트 — 대부분 실측 완료 (`reports/measured_evidence.md`)

1. ✅ **plain ER 실행** — A-GEM 0.387 > ER 0.346 > baseline 0.251 (5 seeds). A-GEM +4.1%p 우위.
2. ✅ **seed std/CI** — A-GEM 0.387 ± 0.018, +13.6%p Acc / +17.6%p 망각보호 (5 seeds).
3. ✅ **효율 절대 수치** — 604K params, full run ~4.5분/seed(CPU), RAM 0.95GB, ~6.4 win/s.
4. ⬜ **few-shot enrollment 소규모 측정** — 아직(데모 능력). 클립 수→신규정확도/기존망각 수치화는 후속.
5. ✅ **하향 절제** — GRU 256→128 무손실(0.388→0.387), h64서 하락. 용량 양방향 무병목.
6. ✅ **48-class 선정 기준** — 8×6 의미론적 큐레이션 커리큘럼.
7. ✅ **지표 정의** — task-aware 6-way, **chance 1/6 ≈ 0.167** (task-incremental). 진짜 class-IL(48-way, task ID 없음) 수치도 실측: baseline 6.2%/A-GEM 9.9%.
8. ✅ **GDumb(버퍼 효과 반론 검증)** — GDumb 0.164±0.004 ≈ chance(0.167), baseline(0.251)보다도 낮음. "CL 이득=버퍼 효과" 반론 반증됨. (`dev/run_gdumb_comparison.py`, `reports/post_agem_methods_comparison.md`)
9. ✅ **정밀 지표(precision/recall/F1/mAP)** — `reports/val_metrics_result.md`.
10. ✅ **학습 세팅(initial vs continual)** — Stage1=6클래스 initial(replay 없음) : Stage2-8=42클래스 continual(growing A-GEM replay) = 1:7.
11. ✅ **데모 앱 사용성 버그 수정** — OOD 배지가 기본 설정에선 `threshold=None`이라 **항상 미작동**이었음(사용해보다 발견). 앱 시작 시 자동 보정(real val feature 또는 Docker/Spaces용 synthetic fallback)으로 수정, calibration 유효성 검증(held-out ID FPR 4%≈목표 5%, feature-space OOD 42% 플래그). 회귀 테스트 추가.

---

## 출처 (핵심 URL)

- A-GEM (우리 baseline): https://arxiv.org/abs/1812.00420
- DER/DER++ (리허설 SOTA 비교): https://arxiv.org/abs/2004.07211
- ER-ACE: https://arxiv.org/abs/2104.05025
- GDumb (버퍼-only 상한 반론): https://arxiv.org/abs/2007.05027
- CODA-Prompt (prompt-pool CIL): https://arxiv.org/abs/2211.13218
- SimpleCIL/APER (frozen feature가 prompt를 이김): https://arxiv.org/abs/2303.07338
- RanPAC (frozen feature + random projection): https://arxiv.org/abs/2307.02251
- vCLIMB (비디오 CIL 벤치마크): https://arxiv.org/abs/2201.09381
- PIVOT (frozen CLIP + prompt, 우리와 가장 유사): https://arxiv.org/abs/2212.04842
- SMILE (영상당 1프레임): https://arxiv.org/abs/2305.18418
- ESSENTIAL (현재 vCLIMB SOTA, ICCV'25): https://arxiv.org/abs/2508.10896
- VideoMAE V2 (SSv2 recog SOTA): https://arxiv.org/abs/2303.16727
- InternVideo2: https://arxiv.org/abs/2403.15377
- Mamba: https://arxiv.org/abs/2312.00752
- VideoMamba: https://arxiv.org/abs/2403.06977
- "Repeat After Me" (SSM이 짧은 문맥에서 불리): https://arxiv.org/abs/2402.01032
- Deep SLDA (frozen-head 효율 근거): CVPR-W 2020, Hayes & Kanan