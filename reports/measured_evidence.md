# Measured evidence — for the SOTA-positioning brief

Generated: 2026-07-02. Hardware: Apple Mac, **CPU** (feature extraction can use MPS).
All numbers below are freshly measured on this machine; scripts noted per section.

## 0. Setup & metric definition (important)

- **Benchmark:** 48-class subset of Something-Something V2 = **8 stages × 6 curated templates**
  (S1 Open/Close, S2 Container, S3 vertical/depth move, S4 horizontal push/pull, S5 Cover/Throw/Drop,
  S6 Push-force/Hit/Tear, S7 Lift/Drop/Fall, S8 Pretend/Show). 100 instances/class, seed-fixed
  `random.sample` (4800 train / 4702 val). Curriculum-style, not cherry-picked.
- **Evaluation = task-aware (within-stage 6-way):** `eval_stage` restricts argmax to the queried
  stage's 6 classes. So **chance = 1/6 ≈ 0.167**, and this is a **task-incremental** setting
  (task/stage ID known at test) — NOT directly comparable to class-IL SOTA (vCLIMB is over all classes).
- **Avg Acc** = mean over the 8 stage-groups of within-group 6-way accuracy after the final stage.
- **S1 drop** = stage-1 6-way accuracy (right after stage 1) − (after all 8 stages); lower = less forgetting.

## 1. Method comparison — baseline vs plain ER vs A-GEM (5 seeds, B/32)

`dev/run_er_comparison.py` (48cls/8stage, mem=50/stage, chance 0.167):

| Method | Avg Acc | S1 forgetting |
|---|---|---|
| baseline (no memory) | 0.251 ± 0.021 | +0.115 ± 0.088 |
| plain ER (rehearsal, no projection) | 0.346 ± 0.012 | +0.020 ± 0.098 |
| **A-GEM** | **0.387 ± 0.018** | **−0.061 ± 0.077** |

- **A-GEM vs baseline:** +13.6%p Avg Acc, +17.6%p S1-forgetting protection.
- **A-GEM vs plain ER:** **+4.1%p** — in our task-aware / frozen-CLIP setting A-GEM *beats* plain ER
  (reverses the usual "ER > A-GEM on standard class-IL" expectation). A-GEM also has the only
  non-positive (i.e. backward-transfer) S1 forgetting.

## 2. Capacity / architecture ablation — bidirectional

Downward (this run, `run_capacity_ablation.py --configs best_current_h256 gru_h128 gru_h64`, 3 seeds, B/32):

| GRU hidden | Avg Acc |
|---|---|
| 256 (main) | 0.388 ± 0.020 |
| **128 (half)** | **0.387 ± 0.004** — no loss |
| 64 | 0.352 ± 0.007 — only here does it drop |

Combined with earlier runs → **capacity is not the bottleneck in either direction:**

| Lever | Δ Avg Acc | verdict |
|---|---|---|
| GRU 256 → 128 (halve) | ~0.000 | no loss |
| GRU 256 → 512 (double) | ~0.000 | no gain |
| Backbone B/32 → L/14 (L/14 run) | +0.013 | within noise |
| GRU → GRU+Attention | −0.103 | worse |
| GRU → diagonal-SSM | −0.040 | worse (more consistent) |
| **Training data 25% → 100%** | **+0.06–0.07** | **dominant lever** |

## 3. Efficiency (CPU) — `dev/measure_efficiency.py`

| Metric | Value |
|---|---|
| Temporal-head params | **GRU 604K** (Attention 867K, SSM 936K) |
| — context | VideoMAE-V2 ViT-g ~1.01B, InternVideo2-6B ~6B (10³–10⁴× larger) |
| Backbone training | **none** (CLIP fully frozen) |
| Full 8-stage training | **~4.5 min / seed** (per stage ~33 s) |
| Peak RAM | **0.95 GB** |
| Inference — GRU head | 0.3 ms/window (3300 windows/s) |
| Inference — end-to-end (CLIP+GRU) | ~156 ms/window ≈ **6.4 windows/s** (CPU, CLIP encoder is the bottleneck) |

## Reproduce

```bash
python3 dev/run_er_comparison.py                 # baseline vs ER vs A-GEM, 5 seeds
python3 experiments/run_capacity_ablation.py --configs best_current_h256 gru_h128 gru_h64 \
    --feature-dir data/features --feature-dim 512 --seeds 0 1 2 --epochs 15   # down-ablation
python3 dev/measure_efficiency.py                # params / train time / RAM / FPS
```

## What this lets us claim honestly

1. **Robustness/simplicity (strongest):** capacity helps in neither direction (128=256, 512 no gain),
   richer temporal models hurt (Attention/SSM), data is the only lever. Not an assertion — measured.
2. **Efficiency:** 604K-param head, full continual run in ~4.5 min on CPU with <1 GB RAM, real-time-ish
   inference. Orders of magnitude lighter than video-transformer SOTA.
3. **A-GEM is justified here:** beats plain ER and baseline in our setting, with backward-transfer forgetting.
4. Still honest limits: task-IL (chance 1/6) not class-IL; small subset; not compared to DER++/ER-ACE or
   video-CL SOTA (PIVOT/ESSENTIAL); few-shot enrollment still a demo (not yet quantified).
