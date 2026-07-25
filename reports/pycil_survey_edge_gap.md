# PyCIL 저자 그룹 서베이의 "edge/resource-constrained" 취급 — 원문 전수 조사

Generated: 2026-07 (auto). 대상: **Zhou et al., "Class-Incremental Learning: A Survey"**
([arXiv:2302.03648](https://arxiv.org/abs/2302.03648), v2 2024-07, **TPAMI**, 38쪽,
[코드](https://github.com/zhoudw-zdw/CIL_Survey/)) — PyCIL을 만든 바로 그 그룹(난징대 LAMDA
+ NTU)이 쓴 이 분야의 표준 서베이.

목적: 우리 edge-CL 프레이밍(`docs/project_focus.md`)이 **이미 다뤄진 주제인지, 아니면
빈 구멍인지**를 원문 근거로 확정. PDF 전문 텍스트를 추출해 키워드 전수 검색 + 해당 구절
정독으로 확인했다.

## 요약 (한 줄)

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

인용된 두 편 = 이 방향의 선행연구로 **우리가 반드시 확인해야 할 논문**:
- **[227] SparCL: Sparse Continual Learning on the Edge** (Wang et al., **NeurIPS 2022**)
- **[228] Computationally Budgeted Continual Learning: What does matter?**
  (Prabhu et al., **CVPR 2023** — GDumb 저자)

## 3. 우리 프로젝트에 주는 결론

### 갭이 실재함 (논문 §Introduction에 그대로 쓸 수 있음)

이 분야의 **표준 서베이(TPAMI)가 edge 배포를 명시적 동기로 내걸었으면서도**:
- 측정한 건 **메모리 바이트**뿐 — **CPU 실행, 지연, 전력은 하나도 측정 안 함**
- 유일한 시간 측정조차 **3090 GPU** 위에서 수행
- "computationally-efficient algorithms 설계가 필요하다"고 **스스로 미래 과제로 명시**

→ **우리의 CPU-only 실측(학습 초 단위, RAM 0.95GB, backprop-free)은 이 서베이가 비워둔
바로 그 칸에 들어간다.** "메모리는 정렬했지만 연산은 정렬 안 된 비교"에 **연산 축을
추가**하는 것이 우리 기여로 정당화됨.

### 즉시 채택할 것 (방법론 차용)

1. **AUC-A / AUC-L 지표** — 예산을 바꿔가며 성능-메모리 곡선을 그려 확장성을 보고하는
   방식. 우리 `benchmarks/pycil/` 브리지에 이미 같은 스플릿이 깔려 있으므로 적용 비용이 낮다.
2. **모델→exemplar 환산 회계**(§4.4) — 우리 FeCAM head(2.1MB)·GRU(604K params)를
   "exemplar 몇 장 상당"으로 환산해 보고하면 서베이 관례와 정렬된다.
3. **"no free lunch / 교차점" 프레이밍** — 서베이 스스로 "작은 예산에선 다른 계열이
   이긴다"고 인정했으므로, **우리가 겨냥하는 저예산 구간이 정당한 연구 영역**임을
   서베이 문장으로 직접 뒷받침할 수 있다.

### 후속 확인 필요 (미착수)

- **SparCL(NeurIPS'22)** 원문 — 진짜 edge CL 선행연구. 우리와 겹치는지/보완적인지 확인 필수.
- **Prabhu et al.(CVPR'23)** 원문 — "연산 예산 고정" 프로토콜. 우리 CPU 실측을 이 프로토콜에
  맞추면 비교 가능성이 생김.
- 두 편 모두 확인 후 `docs/project_focus.md`의 갭 목록을 갱신할 것.

## 4. 재현

```bash
curl -sL https://arxiv.org/pdf/2302.03648 -o survey.pdf && pdftotext survey.pdf survey.txt
grep -c "CPU" survey.txt                                   # → 0
grep -i "edge" survey.txt | grep -vi "knowledge"           # → 실제 언급 8곳
```
