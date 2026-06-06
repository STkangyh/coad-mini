# Data-Fraction Scaling Experiment

## What it answers

Earlier ablations compared two feature backbones — CLIP **B/32** (512-d) and
OpenCLIP **L/14** (768-d) — and found only a modest gap. A natural worry is that
the gap is **masked by small data**: with few training videos per stage, the GRU
saturates before it can exploit the richer L/14 features.

`experiments/run_data_scale.py` trains the project **MAIN model** (GRU
`hidden=256` + A-GEM *balanced*, `mem=50`, `replay=0.25`) on increasing
fractions of the per-stage training data and reports final average accuracy and
stage-1 forgetting (`s1_drop`). Running it on both backbones lets us check:

1. **Does the B/32 ↔ L/14 gap widen as data grows?** Compare the per-fraction
   `avg_acc_mean` between the two backbones.
2. **Are the accuracy curves still rising at 100%?** If `avg_acc_mean` is still
   climbing from fraction 0.5 → 1.0, more data would likely help further and the
   backbone comparison at 100% is not yet saturated.

## Why the comparison is fair

The per-stage subsample is chosen by `subsample_stage(samples, fraction, seed,
stage_id)`, whose selection depends **only** on `(seed, fraction, stage_id)` and
each sample's `id` — never on the feature directory. So a B/32 run and an L/14
run with the *same* `(seed, fraction)` train on the **exact same videos**, and
any accuracy difference is attributable to the backbone, not to different data.

## Output format

Per seed (regex-friendly):

```
fraction=<f> seed=<s> avg_acc=<x> s1_drop=<y>
```

Per fraction:

```
fraction=<f> avg_acc_mean=<m> avg_acc_std=<sd> s1_drop_mean=<...> s1_drop_std=<...>
```

## Run commands (both backbones)

```
python3 experiments/run_data_scale.py --feature-dir data/features              --feature-dim 512 --fractions 0.25 0.5 1.0 --seeds 0 1 2
python3 experiments/run_data_scale.py --feature-dir data/features_openclip_l14 --feature-dim 768 --fractions 0.25 0.5 1.0 --seeds 0 1 2
```

Use the **same** `--fractions` and `--seeds` for both so the per-`(seed,
fraction)` training sets match across backbones.

## Tests

```
python3 -m pytest tests/test_data_scale.py -q
```
