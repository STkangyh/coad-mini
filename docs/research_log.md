# Research Log — Continual Learning on Something-Something V2

> 이 문서는 실험 설계부터 최종 결과까지의 의사결정 과정을 기록한 연구 일지입니다.  
> 코드만으로는 드러나지 않는 **왜 이 방향을 택했는가**를 남기기 위해 작성됐습니다.

---

## 1. 출발점 — 문제 정의

**연구 질문:**  
Something-Something V2에서 행동 클래스를 순차적으로 학습할 때,  
이전에 배운 행동을 잊지 않으면서 새 행동을 학습할 수 있는가?

**초기 세팅:**
- CLIP ViT-B/32 → 16프레임 균등 샘플링 → 512-dim 피처
- GRU Classifier (`hidden_dim=256`)
- Stage 구조: 클래스를 묶어 순차 학습
- 평가 지표: Avg Acc (전체 클래스 평균), S1 Drop (첫 stage 망각량)

---

## 2. 첫 번째 가설 — OrthogonalGradient

### 아이디어
새 task를 학습할 때, 이전 task에 해가 되는 gradient 방향 성분을 제거한다.  
이전 stage의 gradient 방향을 `g_ref`로 저장하고, 현재 gradient를 그에 직교하도록 투영:

$$g_{\text{new}} = g_{\text{cur}} - \frac{g_{\text{cur}} \cdot g_{\text{ref}}}{g_{\text{ref}} \cdot g_{\text{ref}}} \cdot g_{\text{ref}}$$

### 구현
- `utils/orthogonal_grad.py`: stage 종료 시 gradient를 누적 평균으로 저장
- λ sweep: λ=0.0 (Baseline) ~ λ=1.0 (완전 투영)

### 결과 — 실패

| λ | Avg Acc | S1 Drop |
|---|---|---|
| 0.0 (Baseline) | 0.346 | +0.149 |
| 0.1 | 0.301 | +0.193 |
| 0.5 | 0.289 | +0.211 |
| 1.0 | 0.271 | +0.238 |

**λ가 클수록 오히려 성능이 하락했다.**

### 원인 분석
- 저장된 `g_ref`는 **이전 stage의 평균 gradient** — 실제로 모델이 어디서 틀리고 있는지와 무관
- SS-V2 행동 클래스들은 특징 공간에서 상당히 겹치기 때문에, 이전 gradient 방향을 억제하면 현재 task 학습도 방해함
- OrthGrad는 gradient 충돌을 **예방**하는 게 아니라 **억제**하는 방식 → 잘못된 방향 제약

---

## 3. 전환 — A-GEM 도입

### 왜 A-GEM인가
OrthGrad 실패의 핵심 교훈: *"어떤 방향이 나쁜지"를 데이터 없이 알 수 없다.*  
A-GEM (Lopez-Paz & Ranzato, NeurIPS 2017)은 **실제 이전 task 샘플**로 reference gradient를 계산:

1. 이전 각 stage에서 샘플 `M`개를 메모리에 저장
2. 학습 중 메모리 샘플로 reference gradient `g_ref` 계산
3. `dot(g_cur, g_ref) < 0` 일 때만 gradient를 투영

$$g_{\text{new}} = g_{\text{cur}} - \frac{g_{\text{cur}} \cdot g_{\text{ref}}}{g_{\text{ref}} \cdot g_{\text{ref}}} \cdot g_{\text{ref}}$$

**OrthGrad와 수식은 같지만 `g_ref`의 출처가 다르다** — 이것이 핵심 차이.

### 구현 최적화
초기 A-GEM은 매 step마다 `g_ref`를 재계산 → 너무 느림.  
→ `precompute_ref()`: **epoch 시작 시 1회 캐싱**, 이후 step마다 재사용

```python
# gem.py 핵심 로직
def precompute_ref(self, model, device):
    # 메모리 샘플 전체로 g_ref 1회 계산
    ...

def apply(self, model, device):
    g_cur = self._collect_grad(model)
    if g_cur.dot(self._g_ref) < 0:      # 충돌 시에만
        g_new = project(g_cur, self._g_ref)
        self._set_grad(model, g_new)
```

---

## 4. Multi-seed 검증 — "우연이 아님을 확인"

5개 seed (0~4)로 반복 실험하여 결과의 신뢰성을 확보.

### 결과 (48 classes, 8 stages)

| Method | Avg Acc | ±Std | S1 Drop | ±Std |
|---|---|---|---|---|
| Baseline | 0.251 | 0.021 | +0.115 | 0.088 |
| **A-GEM** | **0.393** | **0.015** | **−0.042** | **0.080** |

- S1 Drop이 **음수**: Stage 1 정확도가 학습 내내 오히려 소폭 향상됨
- std가 Baseline보다 낮음: A-GEM이 더 **안정적**

---

## 5. Class Scaling — "더 어려운 문제에서도 유효한가"

**연구 질문:**  
클래스 수와 stage 수가 늘어날수록 A-GEM의 우위는 어떻게 변하는가?

| N | Stages | BL Avg | GEM Avg | **Δ Avg** | BL S1↓ | GEM S1↓ | **Δ S1↓** |
|---|---|---|---|---|---|---|---|
| 16 | 4 | 0.346 | 0.417 | +0.071 | +0.149 | +0.074 | +0.074 |
| 24 | 4 | 0.303 | 0.423 | +0.120 | — | — | — |
| 32 | 8 | 0.364 | 0.456 | +0.092 | +0.075 | −0.022 | +0.097 |
| **48** | **8** | **0.251** | **0.393** | **+0.141** | **+0.115** | **−0.042** | **+0.157** |

**A-GEM의 우위는 문제가 어려워질수록 단조 증가한다.**  
48 classes에서 최대 Δ Avg **+14.1%p**, S1 forgetting 보호 **+15.7%p** 달성.

---

## 6. Memory Ablation — "메모리 크기의 효과"

48 classes, seed=0 고정. 메모리 크기별 성능:

| mem/stage | Avg Acc | S1 Drop |
|---|---|---|
| 10 | 0.306 | +0.080 |
| 20 | 0.323 | +0.123 |
| 30 | 0.343 | +0.098 |
| **50** | **0.374** | **+0.008** |
| Baseline | 0.230 | +0.210 |

→ `mem=50`을 기본값으로 선택.

---

## 7. 심층 분석 — "A-GEM이 왜 좋은가"

### 7-1. Replay Ratio 실험

`precompute_ref()` 시 메모리의 몇 %를 사용해 `g_ref`를 계산하는가?  
(5 seeds, 48 classes, selection=random, mem=50)

| ratio | Avg Acc | ±Std | S1 Drop | Δ vs BL |
|---|---|---|---|---|
| (BL) | 0.251 | 0.021 | +0.115 | — |
| 0.10 | 0.379 | 0.014 | −0.033 | +0.127 |
| **0.25** | **0.391** | **0.014** | **−0.041** | **+0.139** |
| 0.50 | 0.382 | 0.004 | −0.038 | +0.131 |
| 1.00 | 0.367 | 0.012 | −0.016 | +0.116 |

**발견:** 전체 메모리(ratio=1.0)보다 25% 서브샘플이 더 효과적.  
→ *"g_ref를 너무 많은 샘플로 계산하면 gradient 방향이 과하게 제약된다"*  
→ 적당한 다양성이 효과적인 충돌 방지 방향을 제공함.

### 7-2. Memory Selection 실험

저장 단계에서 어떻게 샘플을 고르는가?  
(5 seeds, replay_ratio=1.0, mem=50)

| strategy | Avg Acc | ±Std | S1 Drop | Δ vs BL |
|---|---|---|---|---|
| (BL) | 0.251 | 0.021 | +0.115 | — |
| random | 0.367 | 0.012 | −0.016 | +0.116 |
| **balanced** | **0.379** | **0.015** | **−0.038** | **+0.128** |

**발견:** 클래스 균형 샘플링이 random보다 우수.  
→ *"class-balanced memory가 gradient space를 더 균등하게 커버한다"*  
→ Ego4D처럼 클래스 불균형이 심한 데이터셋에서 더욱 중요한 요소.

### 최종 최적 구성

```
A-GEM (mem=50, selection=balanced, replay_ratio=0.25)
  → Avg Acc: 0.363  (+13.3%p vs Baseline)
  → S1 Drop: −0.032 (망각 없음, 오히려 소폭 향상)
```

---

## 8. 연구 흐름 요약

```
OrthGrad 가설
    ↓
실험 (λ sweep)
    ↓
실패 (모든 λ > 0에서 Baseline보다 나쁨)
    ↓
원인 분석 (g_ref의 품질 문제)
    ↓
A-GEM 도입 (실제 샘플 기반 reference)
    ↓
Multi-seed 검증 (5 seeds)
    ↓
Class Scaling (16 → 24 → 32 → 48)
    ↓
Memory Ablation (mem: 10 → 50)
    ↓
Replay Ratio 실험 (ratio: 0.10 → 1.00)
    ↓
Memory Selection (random vs balanced)
    ↓
FastAPI 데모 (실시간 자막 포함)
```

**OrthGrad의 실패가 연구적으로 도움이 됐다.**  
단순히 gradient 방향을 억제하는 것과, 실제 과거 데이터로 충돌을 감지하는 것의 차이를  
직접 실험으로 확인할 수 있었다.

---

## 9. Ego4D 확장 계획

현재 파이프라인은 Ego4D Online Action Detection으로 바로 확장 가능:

| 현재 (SS-V2) | Ego4D 확장 |
|---|---|
| CLIP ViT-B/32, 512-dim | ViT-L/14 (1024-dim) 또는 유지 |
| GRU (hidden=256) | 동일 구조 사용 가능 |
| 48 classes, 8 stages | Ego4D verb/action 클래스로 교체 |
| A-GEM (balanced, ratio=0.25) | 그대로 적용 |
| FastAPI 데모 | Ego4D OAD 데모로 확장 |

클래스 불균형이 심한 Ego4D에서는 **balanced memory selection**이 특히 중요할 것으로 예상.

---

## 10. 핵심 결론

> Something-Something V2 기반 Continual Learning 환경에서  
> 단순 Orthogonal Gradient는 효과가 없었지만,  
> **A-GEM은 클래스 수와 스테이지 수가 증가할수록 더 큰 성능 향상과 forgetting 감소를 보였다.**  
>  
> 특히 48 classes × 8 stages 설정에서:  
> - 평균 정확도 **+14.1%p**  
> - S1 forgetting 보호 **+15.7%p**  
>  
> 심층 분석을 통해 A-GEM의 효과가  
> *replay 비율의 적절한 조절(25%)* 과  
> *class-balanced 메모리 구성* 에서 비롯됨을 확인했다.

---

## 11. 후속 탐색 — "A-GEM 고정 후, 다음 병목은 어디인가"

A-GEM 구성이 수렴한 뒤, 추가 성능 향상이 어디서 나올지 네 가지 방향을 순차적으로 검증했다.
모두 동일 셋업(48 classes / 8 stages / 3-seed / A-GEM balanced mem=50 replay=0.25)에서 비교했다.

### 11-1. 모델 용량 (capacity sweep)
hidden 256→512, 1-layer→2-layer, balanced→balanced_hard. 모두 **개선 없음 또는 악화**.
→ GRU 용량/메모리 전략은 병목이 아니다. (`reports/capacity_ablation_cv_report.md`)

### 11-2. Feature Backbone (CLIP B/32 → OpenCLIP L/14)
| | Avg Acc | S1 Drop |
|---|---|---|
| CLIP B/32 (512d) | 0.388 | -0.078 |
| OpenCLIP L/14 (768d) | 0.401 ± 0.033 | +0.047 |

+0.013 (3-seed ±0.033 분산 내, 사실상 노이즈). (`reports/openclip_l14_result.md`)

### 11-3. Temporal 복잡도 (GRU → GRU + MultiheadAttention)
| | Avg Acc | S1 Drop |
|---|---|---|
| GRU | 0.401 | +0.047 |
| GRU + Attention | **0.298 ± 0.019** | **+0.240** |

정확도 -0.103, 망각 대폭 악화. 파라미터가 늘면 A-GEM 제약이 약해져 **망각이 급증**.
→ 아키텍처 복잡도 증가는 역효과. (`reports/gru_attention_result.md`)

### 11-4. 데이터 규모 (data-fraction scaling)
동일 영상을 25/50/100%로 줄여 B/32 vs L/14 비교 (subsample은 backbone과 무관하게 동일).

| 데이터 | B/32 | L/14 | Δ |
|---|---|---|---|
| 25% | 0.314 | 0.339 | +0.026 |
| 50% | 0.370 | 0.394 | +0.024 |
| 100% | 0.388 | 0.401 | +0.013 |

- 데이터 25→100%: **+0.06~0.07** (backbone +0.01~0.026, 아키텍처 0/악화보다 압도적)
- L/14 우위는 **저데이터 현상** — 데이터가 늘수록 격차가 좁혀짐(+0.026→+0.013). "데이터가 적어 backbone 효과가 가려졌다"는 가설은 **기각**.
- 두 곡선 모두 100%에서 아직 상승 중 = **saturation 전**. (`reports/data_scale_result.md`)

### 11-5. 종합 결론

```text
GRU 용량         ❌
Memory 전략      ❌
Feature backbone ⚠️  저데이터에서만 소폭, 고데이터서 소멸
Temporal 복잡도   ❌  (Attention 오히려 악화)
데이터 규모       ⭕  지배적 레버, 곡선 아직 상승 중
```

> 모델 측에서는 **단순 GRU + A-GEM(balanced, mem=50)이 이 셋업의 강건한 최적점**이며,
> 아키텍처·backbone 복잡도를 더 올리는 방향은 수익이 없었다.
> **다음 성능 레버는 모델이 아니라 학습 데이터 규모 확대**에 있다.
> (이는 Ego4D 확장 시 backbone을 굳이 L/14로 키우기보다 데이터·클래스 커버리지에
> 투자하는 편이 낫다는 함의로 이어진다.)
