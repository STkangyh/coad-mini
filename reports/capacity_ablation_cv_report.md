# GRU Capacity and Memory Selection Cross-Seed Validation Report

Generated: 2026-06-05 23:14:18 KST

## Summary

이번 검증은 48-class Something-Something V2 subset의 8-stage continual-learning 설정에서 A-GEM 기반 모델 용량과 memory selection 변경 효과를 확인하기 위해 수행했다.

엄밀한 k-fold split이 아니라, continual-learning stage split을 고정한 상태에서 seed를 바꿔 반복한 3-seed cross-seed validation이다. 이 설정이 현재 코드 구조와 연구 질문에 더 직접적으로 맞는다.

결론은 명확하다.

- 현재 best 설정인 `hidden=256`, `1-layer GRU`, `balanced memory`가 가장 안정적이었다.
- `hidden=512`는 평균 정확도는 거의 동률이지만, S1 forgetting 방어가 약해졌다.
- `hidden=512 + 2-layer GRU + dropout=0.2`는 성능이 하락했다.
- `balanced_hard memory`는 이번 설정에서는 크게 실패했다.

따라서 현재 checkpoint 또는 demo용 best 후보는 계속 `best_current_h256`을 유지하는 것이 맞다.

## Experimental Setup

| Item | Value |
|---|---:|
| Dataset | Something-Something V2 mini subset |
| Classes | 48 |
| Stages | 8 |
| Classes per stage | 6 |
| Feature | CLIP ViT-B/32 precomputed features |
| Continual-learning method | A-GEM |
| Memory per stage | 50 |
| Replay ratio | 0.25 |
| Epochs per stage | 15 |
| Seeds | 0, 1, 2 |
| Device | CPU |

## Configurations

| Config | Hidden | Layers | Dropout | Memory selection |
|---|---:|---:|---:|---|
| `best_current_h256` | 256 | 1 | 0.0 | balanced |
| `h512` | 512 | 1 | 0.0 | balanced |
| `h512_2layer` | 512 | 2 | 0.2 | balanced |
| `h512_2layer_balanced_hard` | 512 | 2 | 0.2 | balanced_hard |

## Average Results

| Rank | Config | Avg Acc mean | Avg Acc std | S1 Drop mean | S1 Drop std |
|---:|---|---:|---:|---:|---:|
| 1 | `best_current_h256` | 0.388 | 0.020 | -0.078 | 0.060 |
| 2 | `h512` | 0.387 | 0.012 | -0.039 | 0.035 |
| 3 | `h512_2layer` | 0.355 | 0.048 | +0.008 | 0.019 |
| 4 | `h512_2layer_balanced_hard` | 0.272 | 0.042 | +0.110 | 0.028 |

`S1 Drop` is defined as stage-1 accuracy after stage 1 minus stage-1 accuracy after the final stage. Lower is better; negative values mean final-stage evaluation recovered above the initial S1 snapshot.

## Per-Seed Results

| Config | Seed | Avg Acc | S1 Drop | S2 Drop |
|---|---:|---:|---:|---:|
| `best_current_h256` | 0 | 0.363 | -0.032 | +0.077 |
| `best_current_h256` | 1 | 0.386 | -0.163 | +0.130 |
| `best_current_h256` | 2 | 0.413 | -0.040 | +0.038 |
| `h512` | 0 | 0.370 | +0.005 | +0.092 |
| `h512` | 1 | 0.400 | -0.080 | +0.072 |
| `h512` | 2 | 0.391 | -0.042 | +0.017 |
| `h512_2layer` | 0 | 0.293 | +0.007 | +0.072 |
| `h512_2layer` | 1 | 0.363 | +0.032 | -0.087 |
| `h512_2layer` | 2 | 0.409 | -0.015 | -0.013 |
| `h512_2layer_balanced_hard` | 0 | 0.287 | +0.117 | +0.055 |
| `h512_2layer_balanced_hard` | 1 | 0.215 | +0.073 | +0.135 |
| `h512_2layer_balanced_hard` | 2 | 0.315 | +0.140 | +0.105 |

## Interpretation

### 1. Hidden 512 did not clearly improve performance

`h512` achieved `0.387` average accuracy, nearly identical to `best_current_h256` at `0.388`. However, S1 forgetting defense weakened from `-0.078` to `-0.039`.

This suggests that hidden size alone is not the current bottleneck. The model may need optimizer or learning-rate retuning before the extra capacity becomes useful.

### 2. 2-layer GRU underperformed

`h512_2layer` dropped to `0.355` average accuracy. This is a meaningful decline from the current best.

The likely causes are:

- optimization became harder under the same learning rate;
- dropout may be too strong for the current feature/data regime;
- 15 epochs per stage may be insufficient for the larger recurrent model;
- stage-wise continual learning may amplify instability from the deeper GRU.

### 3. Hard example memory failed in this run

`h512_2layer_balanced_hard` dropped to `0.272` average accuracy. This is the worst result in the sweep.

The likely explanation is that selecting high-loss examples after each stage over-concentrated memory on noisy, ambiguous, or unstable samples. For A-GEM, reference gradients need to represent the previous task distribution, not only its hardest tail.

Hard memory should not be adopted in the current form.

## Recommendation

Keep the current best configuration:

```text
A-GEM
hidden_dim = 256
num_layers = 1
memory selection = balanced
memory per stage = 50
replay_ratio = 0.25
```

Do not replace it with `hidden=512`, `2-layer GRU`, or `balanced_hard memory` based on this validation.

## Next Experiments

The next most useful experiments are smaller and more targeted:

1. Test `hidden=512` with learning-rate retuning:

```text
lr = 5e-5
lr = 7.5e-5
lr = 1e-4
```

2. Test 2-layer GRU without dropout first:

```text
hidden_dim = 512
num_layers = 2
dropout = 0.0
```

3. Modify hard memory into a hybrid strategy instead of pure high-loss selection:

```text
70% class-balanced random
30% class-balanced hard
```

4. Prioritize feature upgrades before more GRU scaling:

```text
OpenCLIP ViT-L/14
SigLIP
motion-aware features
```

## Final Decision

For the current project stage, the best validated direction is not GRU scaling. The highest-confidence path is:

```text
Keep A-GEM balanced memory
Keep hidden_dim=256
Move effort toward better visual features or carefully retuned hidden=512 experiments
```

