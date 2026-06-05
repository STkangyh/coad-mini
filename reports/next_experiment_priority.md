# Next Experiment Priority

Generated: 2026-06-05 KST

## Current Main Model

The current model should be treated as the project main model, not just a baseline.

```text
A-GEM
memory selection = balanced
memory per stage = 50
replay ratio = 0.25

GRU
hidden_dim = 256
num_layers = 1

48 classes
8 stages
CLIP ViT-B/32 features
```

Validated 3-seed result:

```text
Avg Acc = 0.388 ± 0.020
S1 Drop = -0.078 ± 0.060
```

## Priority 1: Feature Backbone Upgrade

This is the highest-value next experiment. The previous capacity sweep suggests that GRU scaling is not the main bottleneck under the current feature representation.

Candidate backbones:

```text
OpenCLIP ViT-L/14
SigLIP
```

Extraction commands:

```bash
python scripts/extract_clip_features.py --backbone openclip_l14
python scripts/extract_clip_features.py --backbone siglip_l16_384
```

After extraction, check the generated metadata:

```bash
cat data/features_openclip_l14/feature_config.json
cat data/features_siglip_l16_384/feature_config.json
```

Then run the current main model on the new features:

```bash
python experiments/run_capacity_ablation.py \
  --configs best_current_h256 \
  --feature-dir data/features_openclip_l14 \
  --feature-dim <FEATURE_DIM_FROM_METADATA> \
  --seeds 0 1 2 \
  --epochs 15
```

```bash
python experiments/run_capacity_ablation.py \
  --configs best_current_h256 \
  --feature-dir data/features_siglip_l16_384 \
  --feature-dim <FEATURE_DIM_FROM_METADATA> \
  --seeds 0 1 2 \
  --epochs 15
```

## Priority 2: Learning Rate Sweep for Hidden 512

Only run this after or alongside feature extraction if time remains.

Commands:

```bash
python experiments/run_capacity_ablation.py --configs h512 --lr 1e-4 --seeds 0 1 2 --epochs 15
python experiments/run_capacity_ablation.py --configs h512 --lr 5e-5 --seeds 0 1 2 --epochs 15
python experiments/run_capacity_ablation.py --configs h512 --lr 2e-5 --seeds 0 1 2 --epochs 15
```

Interpretation rule:

```text
Use hidden=512 only if Avg Acc improves and S1 Drop does not degrade.
```

## Priority 3: GRU + Attention

This is a reasonable Ego4D-compatible model direction, but it should not come before feature upgrade.

Target structure:

```text
GRU
↓
MultiheadAttention
↓
Classifier
```

Use this only after the feature-backbone experiment establishes a stronger representation baseline.

## Priority 4: Ego4D Readiness

While waiting for Ego4D approval:

```text
Finalize result summary
Record demo video
Keep current A-GEM balanced model as the main model
Prepare feature-upgrade experiment
```

## Decision

Do not add more architecture complexity immediately. The next real performance opportunity is most likely in visual feature quality:

```text
CLIP ViT-B/32 -> OpenCLIP ViT-L/14 or SigLIP
```

