# GRU + Attention 결과 (negative result)

Generated: 2026-06-06 (auto)

## 설정

```text
A-GEM (balanced, mem=50, replay=0.25)
48 classes / 8 stages / 15 epochs / seeds 0 1 2
feature: OpenCLIP ViT-L/14 (768d)
비교: best_current_h256 (GRU) vs attn_h256 (GRU + MultiheadAttention)
```

## 결과

| Config | Avg Acc | ± | S1 Drop | ± |
|---|---|---|---|---|
| best_current_h256 (GRU) | **0.401** | 0.033 | +0.047 | 0.044 |
| attn_h256 (GRU+Attention) | **0.298** | 0.019 | +0.240 | 0.041 |

per-seed (attn): 0.282 / 0.324 / 0.288 — 일관되게 낮음.

## 전체 비교 (backbone + temporal 종합)

| 모델 | Avg Acc | S1 Drop |
|---|---|---|
| CLIP B/32 + GRU (기준) | 0.388 | -0.078 |
| OpenCLIP L/14 + GRU | 0.401 ±0.033 | +0.047 |
| OpenCLIP L/14 + GRU+Attention | 0.298 ±0.019 | +0.240 |

## 해석

- GRU+Attention은 정확도 **-0.103**, 망각(S1 Drop) **+0.240** 으로 양쪽 모두 크게 악화.
- 원인 추정:
  - 작은 데이터(48-class mini) + 8-stage 순차학습에서 attention의 추가 파라미터가 과적합·불안정.
  - A-GEM의 gradient projection이 늘어난 파라미터/표현을 충분히 제약하지 못해 **망각이 급증**.
  - mean-pool 기반 attention 표현이 단순 GRU last-hidden보다 continual 환경에서 덜 robust.

## 결론

```text
GRU 용량      ❌ (이전 capacity sweep)
Memory 전략   ❌
Feature 표현  ❌ (L/14, +0.013 노이즈 범위)
Temporal 복잡도 ❌ (Attention, 오히려 악화)
```

**단순 GRU + A-GEM(balanced, mem=50)이 이 셋업에서 매우 강건한 최적점**으로 보임.
아키텍처 복잡도를 더 올리는 방향은 수익이 없음 → main 모델 고정 권장.

## 다음 권고

- 현재 GRU + A-GEM을 **project main 모델로 확정**, Ego4D 전까지 아키텍처 튜닝 중단.
- 성능 점프가 더 필요하면 아키텍처가 아니라 **데이터 규모/continual 학습 레짐** 쪽을 검토.
- 실험들이 3-seed 기준 ±0.03 분산 → 작은 효과는 노이즈에 묻힘. 결론용 비교는 seed 수를 늘려야 검정력 확보.
