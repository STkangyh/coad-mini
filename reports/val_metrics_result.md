# Val-set metrics — full class-IL (48-way) vs task-aware (6-way)

Generated: auto. N(val) = 4702. Checkpoints: `checkpoints/{baseline,agem}_48cls.pt`.

Two regimes are reported because they answer different questions:
- **Full 48-way** = true class-incremental / open-world eval, no task ID at test time. Chance = 1/48 ≈ 2.1%.
- **Task-aware 6-way** = current headline metric; argmax restricted to the sample's stage (task ID given). Chance = 1/6 ≈ 16.7%. Easier — this is what `avg_acc` in other reports refers to.

## baseline_48cls.pt (`method=baseline`)

### Full 48-way (class-IL, honest number)

| Metric | Value |
|---|---|
| Accuracy | 0.062 |
| Precision (macro) | 0.030 |
| Recall (macro) | 0.060 |
| F1 (macro) | 0.015 |
| Precision (weighted) | 0.030 |
| Recall (weighted) | 0.062 |
| F1 (weighted) | 0.016 |
| mAP (macro AP) | 0.043 |
| Chance level | 0.021 |

### Task-aware 6-way (current headline `avg_acc`)

| Metric | Value |
|---|---|
| Accuracy (avg_acc) | 0.230 |
| Precision (macro) | 0.254 |
| Recall (macro) | 0.236 |
| F1 (macro) | 0.163 |
| Chance level | 0.167 |

## agem_48cls.pt (`method=agem`)

### Full 48-way (class-IL, honest number)

| Metric | Value |
|---|---|
| Accuracy | 0.099 |
| Precision (macro) | 0.133 |
| Recall (macro) | 0.097 |
| F1 (macro) | 0.068 |
| Precision (weighted) | 0.134 |
| Recall (weighted) | 0.099 |
| F1 (weighted) | 0.069 |
| mAP (macro AP) | 0.104 |
| Chance level | 0.021 |

### Task-aware 6-way (current headline `avg_acc`)

| Metric | Value |
|---|---|
| Accuracy (avg_acc) | 0.363 |
| Precision (macro) | 0.404 |
| Recall (macro) | 0.364 |
| F1 (macro) | 0.353 |
| Chance level | 0.167 |

## A-GEM vs Baseline — both regimes

| Regime | Metric | Baseline | A-GEM | Δ |
|---|---|---|---|---|
| Full 48-way | Accuracy | 0.062 | 0.099 | +0.037 |
| Full 48-way | mAP | 0.043 | 0.104 | +0.060 |
| Full 48-way | F1 (macro) | 0.015 | 0.068 | +0.053 |
| Task-aware 6-way | Accuracy | 0.230 | 0.363 | +0.133 |
| Task-aware 6-way | F1 (macro) | 0.163 | 0.353 | +0.190 |

**Note:** full-48-way accuracy is much lower than the task-aware number for both models — this is the honest, harder class-IL evaluation with no task ID. A-GEM's relative improvement over baseline holds in both regimes.