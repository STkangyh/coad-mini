# edge/resource-constrained CL 선행연구 조사 — 서베이 + 지목된 2편 원문 확인

Generated: 2026-07 (auto). 조사 대상 3편(전부 PDF 원문 정독):
1. **Zhou et al., "Class-Incremental Learning: A Survey"**
   ([arXiv:2302.03648](https://arxiv.org/abs/2302.03648), v2 2024-07, **TPAMI**, 38쪽,
   [코드](https://github.com/zhoudw-zdw/CIL_Survey/)) — PyCIL을 만든 그룹의 표준 서베이
2. **SparCL: Sparse Continual Learning on the Edge**
   ([NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/80133d0f6eccaace15508f91e3c5a93c-Paper-Conference.pdf),
   [코드](https://github.com/neu-spiral/SparCL)) — 서베이가 "resource-limited" 선행연구로 지목
3. **Computationally Budgeted Continual Learning: What Does Matter?**
   ([arXiv:2303.11165](https://arxiv.org/abs/2303.11165), **CVPR 2023**,
   [코드](https://github.com/drimpossible/BudgetCL)) — 서베이가 "computational budget" 선행연구로 지목

목적: 우리 edge-CL 프레이밍(`docs/project_focus.md`)이 **이미 다뤄진 주제인지, 아니면
빈 구멍인지**를 원문 근거로 확정.

## 요약 — 갭은 "부분적으로만" 열려 있다 (초기 판단 수정)

| 주장 | 판정 |
|---|---|
| "표준 서베이가 CPU/연산 축을 안 다룬다" | ✅ **사실** — 38쪽에 "CPU" 0회, 전 실험 3090 GPU |
| "아무도 실제 기기 CPU 학습을 측정 안 했다" | ❌ **거짓** — **SparCL이 갤럭시 S20 CPU에서 학습 가속 3.1× 실측** |
| "연산 예산 고정 CL 연구가 없다" | ❌ **거짓** — CVPR'23이 대규모로 수행(1500 GPU-hours) |
| "**비디오** CIL에서 CPU/지연 측정이 없다" | ✅ **사실** — TCD/STSP/CSTA/ESSENTIAL 전부 미측정 |
| "**backprop 자체를 제거**한 edge CL 비교가 없다" | ✅ **사실** — SparCL은 backprop을 *희소화*할 뿐 |

**결론: 우리 기여는 "edge CL 최초"가 아니라 "비디오 CIL × backprop-free × 절대 CPU
wall-clock"의 교집합.** 재료는 각각 선행연구가 있으나 이 조합은 비어 있다. 아래 §5 참조.

## (1) 서베이 — edge를 "메모리 예산"으로만 정의

**이 서베이는 edge를 "메모리 예산(바이트)" 문제로만 정의하고 그 축은 정교하게 다루지만,
"연산 예산(CPU·지연·전력)" 축은 미래 과제로 넘긴 채 실험을 전혀 하지 않는다. 전 실험이
NVIDIA 3090 GPU에서 수행되고, 38쪽 전체에 "CPU"라는 단어가 0번 등장한다.**

## 1. 키워드 전수 검색 결과 (38쪽 전문)

| 키워드 | 등장 | 비고 |
|---|---|---|
| **CPU** | **0회** | 전문에 한 번도 없음 |
| GPU | 1회 | "All the running time is evaluated on a single NVIDIA 3090 GPU" |
| **FLOPs** | **0회** | 연산량 지표 없음 |
| resource-constrained | 0회 | 해당 표현 자체가 없음 |
| robot / glasses / mobile | **0회** | 우리 타깃 기기군 언급 없음 |
| edge (실제 의미) | **8회** | 74회 중 66회는 "knowl**edge**". 실제 언급은 8회뿐 |
| energy | 6회 | **전부 energy-based model**(모델링 기법). **전력 소비는 0회** |
| embedded | 1회 | 참고문헌 제목("Gaussian kernel **embedded** analytic learning") — 임베디드 기기 아님 |
| budget | 81회 | ← **여기가 이 서베이의 실제 알맹이** |

즉 **"edge"라는 단어의 존재감은 착시**(대부분 knowledge)이고, 실제 논의는 전부
**메모리 예산(budget)** 축에 몰려 있다.

## 2. 서베이가 실제로 한 것 (= 우리가 인용해야 할 것)

### (a) 문제 제기는 명확히 우리와 같은 동기 — Introduction 기여 3번

> "To boost real-world applications, CIL models **should be deployed not only on
> high-performance computers but also on edge devices.** Therefore, we advocate
> evaluating different methods holistically by emphasizing the effect of memory budgets."

### (b) dynamic network는 edge에 부적합하다고 명시 (§3 Discussions)

> "it often requires expandable memory budgets, which is **unsuitable for incremental
> learning on edge devices.** To tackle this problem, further model compression,
> decoupling, and pruning can be adopted."

### (c) 메모리 정렬 비교 방법론 (§4.4) — 바로 재사용 가능한 회계 방식

"모델 저장 공간"을 "exemplar 몇 장"으로 환산해 서로 다른 계열을 같은 자로 잰다:

> ResNet32 = 463,504 params × 4 bytes/float ÷ (3×32×32 bytes/image) ≈ **603 instances**

DER 같은 확장형은 백본을 여러 개 쌓으므로 iCaRL 대비 파라미터가 10배 — 이걸 무시한
기존 비교는 불공정하다는 게 서베이의 핵심 논증.

### (d) 메모리-불가지론 지표 AUC-A / AUC-L (§4.5)

예산을 바꿔가며 성능-메모리 곡선을 그리고 **곡선 아래 면적(AUC)**으로 "예산 확장성"을
측정. 결론: **"there is no free lunch in CIL"** — 작은 예산에선 non-dynamic 계열이,
큰 예산에선 dynamic network가 이기는 **교차점이 존재**. FOSTER·MEMO가 AUC 기준 우수.

### (e) 서베이가 정의한 "edge 예산" (Supplementary)

> "the memory size of the first group, i.e., **a single backbone with 2,000 exemplars.
> It is a relatively small budget, which can be seen as the budget for edge devices.**"

→ 즉 이들의 **"edge" = 백본 1개 + exemplar 2,000장을 저장할 수 있는 메모리**. 여전히
**3090으로 학습한다는 전제**는 그대로다.

### (f) 유일한 연산 효율 실험 (Supplementary J.2)

Figure 30에서 방법별 **running time**을 비교 — 단 **"All the running time is evaluated
on a single NVIDIA 3090 GPU."** exemplar를 늘리면 성능은 오르지만 학습 시간이 급증한다는
트레이드오프를 보이고, 다음과 같이 끝맺는다:

> "When the computational budget is strictly bounded... we need to design more
> **computationally-efficient algorithms**."

### (g) Future Directions에서 명시적으로 "미래 과제"로 넘김

"**CIL with Any Memory/Computational Budgets**" 절:

> "an algorithm should be able to learn with high-performance computers or **with edge
> devices (e.g., smartphones)**, and both scenarios are essential... future CIL methods
> are also encouraged to handle specific learning scenarios, e.g., [227] addresses
> training CIL systems under **resource-limited** scenarios. Another important
> characteristic is the **computational budget** [228]."

인용된 두 편 = 이 방향의 선행연구. **둘 다 원문 확인 완료 → §3, §4.**

## (2) SparCL (NeurIPS'22) — 진짜 폰 CPU 학습 실측이 존재함

**우리 초기 갭 주장을 부분적으로 반박하는 논문.** 반드시 인용해야 함.

- **방법**: 3종 희소화의 시너지 — **TDM**(task-aware dynamic masking, 가중치 희소성)
  + **DDR**(dynamic data removal, "쉬운" 샘플 제거) + **DGM**(dynamic gradient masking,
  중요 gradient만 업데이트)
- **여전히 backprop을 함** — Algorithm 1에 명시: *"Update θ ⊙ M_θ via backpropagation."*
  즉 backprop을 **없애는** 게 아니라 **희소하게** 만드는 접근.
- **백본**: ResNet-18, **사전학습 없이 from scratch**("without any pre-training")
- **벤치마크**: Split CIFAR-10(5 task), Split Tiny-ImageNet(10 task) — **전부 이미지, 비디오 아님**
- **⭐ 실제 기기 측정**: *"measured on the **CPU** of an off-the-shelf **Samsung Galaxy S20**
  smartphone, Qualcomm **Snapdragon 865**, Kryo 585 Octa-core CPU"*, batch 32
  → **학습 가속 3.1×**(sparsity 0.95) / **2.3×**(0.90), 메모리 풋프린트 51%/48% 절감
- **지표**: Class-IL/Task-IL 정확도 + **training FLOPs** + memory footprint
- **성과**: DER++ 대비 최대 **23× 적은 training FLOPs**, 정확도는 오히려 최대 **+1.7%**
- 논문 자체 주장: *"We are not aware of any prior CL works that explored this area and
  considered the constraints of limited resources during training."*

참고로 SparCL Table 1(Split CIFAR-10, buffer 200)의 **A-GEM Class-IL 20.04 / Task-IL 83.88**
— 우리 브리핑이 A-GEM을 "레거시 baseline"으로 규정한 근거와 독립적으로 일치.

## (3) Computationally Budgeted CL (CVPR'23) — 연산 예산을 "iteration 수"로 추상화

- **저자**: Prabhu et al.(GDumb 저자) — Oxford/KAUST/Meta AI
- **핵심 설계**: 연산 예산 **C = training iteration 수**로 정의. 명시적 이유:
  *"This avoids **hardware dependency** or suboptimal implementations when comparing
  methods."* → **의도적으로 하드웨어를 추상화**했으므로 CPU/지연 실측은 없음.
- **하드웨어**: 전부 **A100 GPU**, 총 **1500 GPU-hours**
- **모델**: ResNet-50, **ImageNet1K 사전학습**(= PTM 트랙)
- **데이터**: **ImageNet2K**(2000클래스), **CGLM**(Continual Google Landmarks V2) — 대규모
- **세팅**: data-incremental / class-incremental / **time-incremental**(업로드 타임스탬프 순)
- **예산**: ImageNet2K는 step당 400 iter(batch 1500), CGLM은 100 iter → 매 스텝 관측
  데이터의 25~50%만 학습 가능
- **⭐ 결론 1**: *"**None** of the proposed CL algorithms can outperform our simple
  baseline when computation is restricted."* — Naive(메모리에서 균등 샘플링)를
  distillation/sampling/FC-correction 계열 **전부가** 못 이김.
- **⭐ 결론 2**: *"The gap between existing CL algorithms and our baseline **becomes
  larger with harsher compute restrictions**."*
- **⭐ 결론 3 (우리에게 가장 중요)**: *"training a **minimal subset of the model** with a
  linear layer **can close the performance gap**... but **only when supported by strong
  pretrained models**."* → **frozen 사전학습 + 최소 학습**이 연산 제약 하에서 유효하다는
  독립 증거. 우리 접근의 정당성을 대규모 이미지 실험이 뒷받침.
- **경제 논거**: 저장은 싸고(구글 클라우드 2¢/GB/월) 연산은 비싸다($3/시간 A100) —
  *"computational costs for running an experiment far outweigh the costs for storing
  replay samples"*. **메모리 제약보다 연산 제약이 현실적**이라는 주장.

## 4. 우리 결론과의 정합성 (독립 corroboration)

CVPR'23의 결론 1·2는 **우리가 자체 실험에서 본 것과 같은 방향**이다:

| 그들(ImageNet2K/CGLM, A100, iteration 제약) | 우리(SSv2 48cls, CPU, backprop 제약) |
|---|---|
| 연산 제약 시 정교한 CL 알고리즘이 Naive를 못 이김 | frozen feature 위에서 FeCAM/SLDA가 GRU+A-GEM을 이김 |
| 제약이 심할수록 격차 확대 | base-heavy(예산 희석) 시 격차 확대(`base_heavy_split_result.md`) |
| frozen 사전학습 + 최소 학습이 갭을 메움 | frozen CLIP + 통계 head가 최고 성능 |

→ **우리 발견이 우리 벤치 특유의 현상이 아님**을 대규모 이미지 연구가 독립적으로 뒷받침.
`reports/pycil_bridge_result.md`(CIFAR-100 교차검증)와 함께 인용하면 강력.

## 5. 수정된 갭 주장 (논문 §Introduction용)

**쓸 수 없는 문장** ❌: "edge/CPU에서 CL 학습 비용을 측정한 연구가 없다" → SparCL이 반박.

**방어 가능한 문장** ✅:
1. **도메인**: edge CL 실측은 **이미지**에만 존재(SparCL: CIFAR/Tiny-ImageNet).
   **비디오 CIL 계보(TCD·STSP·CSTA·ESSENTIAL)에는 CPU·지연·전력 측정이 전무**
   (`sota_positioning_brief.md` §1(f) 4편 원문 확인). 비디오는 프레임 축 때문에 연산
   프로파일이 이미지와 질적으로 다르므로 별도 측정이 필요하다.
2. **메커니즘**: SparCL은 backprop을 **희소화**(2.3~3.1× 가속). 우리는 backprop을
   **제거**(닫힌해/스트리밍 통계, 자체 측정 ~30× 빠름). **다른 레버, 다른 크기의 효과**.
3. **지표**: SparCL은 *상대* 가속률, CVPR'23은 *하드웨어 추상화된* iteration 수를 보고.
   **"CPU에서 절대 몇 초에 학습되는가"를 보고하는 연구는 셋 다 아님** — 우리 절대
   wall-clock(학습 초 단위, RAM 0.95GB)이 그 칸을 채운다.
4. **표준 서베이가 이 축을 안 다룸**은 여전히 사실(§1) — 즉 이 방향이 **주류 평가
   프로토콜에 편입되지 않았다**는 문제 제기는 유효.

## 6. 즉시 채택할 것 (방법론 차용)

1. **AUC-A / AUC-L 지표**(서베이 §4.5) — 예산별 성능 곡선의 면적. `benchmarks/pycil/`에
   같은 스플릿이 이미 있어 적용 비용 낮음. **여기에 연산 축을 추가**하는 게 우리 차별점.
2. **모델→exemplar 환산 회계**(서베이 §4.4) — FeCAM head(2.1MB)·GRU(604K params)를
   "exemplar 몇 장 상당"으로 환산 보고.
3. **"no free lunch / 교차점"**(서베이) — 저예산 구간이 정당한 연구 영역임을 서베이
   문장으로 직접 뒷받침.
4. **training FLOPs 보고**(SparCL) — 우리는 지금 wall-clock만 보고 중. FLOPs를 같이
   내면 하드웨어 무관 비교가 가능해져 SparCL/CVPR'23과 나란히 놓을 수 있다. **미착수**.
5. **iteration-budget 프로토콜**(CVPR'23) — 우리 CPU 실측을 "step당 N iteration" 형식으로도
   보고하면 그들 결과와 직접 비교 가능. **미착수**.

## 7. 재현

```bash
curl -sL https://arxiv.org/pdf/2302.03648 -o survey.pdf && pdftotext survey.pdf survey.txt
grep -c "CPU" survey.txt                                   # → 0
grep -i "edge" survey.txt | grep -vi "knowledge"           # → 실제 언급 8곳
```
