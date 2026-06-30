# GRU → SSM 확장 결과

Generated: 2026-06-07 (auto)

## 무엇을 했나

GRU 시간 모델을 **state-space model(SSM) 계열**로 확장 시도. 제약상(맥 CPU/MPS,
Mamba의 CUDA selective-scan 커널 사용 불가) **순수 PyTorch diagonal SSM**(S4D / LRU
스타일)을 구현해 동일 하니스의 `arch="ssm"` 스위치로 GRU와 같은 조건에서 비교.

- 모델: `src/models/ssm_detector.py` — 안정적 대각 선형 재귀 `h_t = a⊙h_{t-1} + (1−a)⊙(Wx_t)`,
  `a=σ(·)∈(0,1)`, GELU MLP + residual + LayerNorm 블록을 2층 적층, causal last-step pooling.
- 인터페이스는 `GRUDetector`와 동일(`forward(x,h)->(logits,h)`)이라 trainer/eval/A-GEM에 drop-in.

## 설정

```text
A-GEM (balanced, mem=50, replay=0.25)
48 classes / 8 stages / 15 epochs / seeds 0 1 2
feature: OpenCLIP ViT-L/14 (768d)
비교: best_current_h256 (GRU, 1-layer) vs ssm_h256 (diagonal SSM, 2-layer)
```

## 결과 (Avg Acc, 3-seed mean ± std)

| Config | Avg Acc | ± | S1 Drop | ± |
|---|---|---|---|---|
| best_current_h256 (GRU) | **0.401** | 0.033 | +0.047 | 0.044 |
| ssm_h256 (SSM) | 0.361 | **0.008** | +0.173 | 0.071 |

per-seed (SSM): 0.372 / 0.361 / 0.352 — 매우 일관적이지만 천장이 낮음.
per-seed (GRU): 0.357 / 0.410 / 0.436 — 평균은 높으나 분산 큼.

## 시간 모델 비교 (종합)

| 시간 모델 | Avg Acc | S1 Drop | vs GRU |
|---|---|---|---|
| GRU (1-layer) | 0.401 | +0.047 | — |
| **SSM** (diagonal, 2-layer) | 0.361 | +0.173 | **−0.040** |
| GRU + Attention | 0.298 | +0.240 | −0.103 |

## 해석

- **SSM은 GRU를 못 이김** (−0.040), 그리고 망각이 더 큼(S1 +0.173 vs +0.047).
  Attention과 같은 패턴: **파라미터/용량이 늘면 A-GEM 제약이 약해져 망각이 증가**.
- 단, SSM은 Attention보다 훨씬 견고하고 (**분산 ±0.008**로 최소) 정확도도 0.06 높음 —
  "복잡도를 더해도 안 망가지는" 정도까지는 옴.
- **핵심 caveat — 시퀀스 길이**: 본 셋업은 영상당 **16프레임**으로 매우 짧다. SSM의 강점은
  *긴* 시퀀스의 장거리 의존성인데 16스텝에선 그 이점이 발휘되지 않아 GRU로 충분하다.
- **구현 caveat**: 이건 실수(real) 대각 SSM이고, **full Mamba/S5(복소 고유값 + selective scan)
  가 아님**. CUDA 커널이 가능한 환경에선 더 나올 여지가 있으나 현재 하드웨어에선 검증 불가.

## 결론

```text
GRU 용량        ❌
Memory 전략     ❌
Feature backbone ⚠️ (저데이터만, 고데이터서 소멸)
Temporal: Attention ❌ (-0.103)
Temporal: SSM       ❌ (-0.040, 망각↑) — Attention보다는 견고
데이터 규모      ⭕ 지배적 레버
```

> **단순 GRU + A-GEM이 이 셋업(짧은 16프레임 시퀀스 + 작은 데이터)의 강건한 최적점**이라는
> 기존 결론을 SSM 확장도 재확인. 다만 SSM은 *안정성(낮은 분산)* 측면에서 매력이 있어,
> **더 긴 시퀀스(예: Ego4D OAD, 프레임 수 ↑)에서 재평가**하면 GRU를 넘을 가능성이 있다 —
> 향후 과제로 권장.

## 재현

```bash
python3 experiments/run_capacity_ablation.py \
  --configs best_current_h256 ssm_h256 \
  --feature-dir data/features_openclip_l14 --feature-dim 768 \
  --seeds 0 1 2 --epochs 15
```
모델/통합/테스트: `src/models/ssm_detector.py`, `experiments/run_capacity_ablation.py`(arch=ssm),
`tests/test_ssm.py` (13 passed).
