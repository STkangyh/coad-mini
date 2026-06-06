# GRU + MultiheadAttention temporal model

`GRUAttentionDetector` (`src/models/gru_attention.py`) is a stronger temporal
model built to test whether the accuracy bottleneck is temporal modeling rather
than the feature backbone (a backbone upgrade gave only +0.013).

## Architecture

```
x: (B, T, feature_dim)
   │
   ▼  nn.GRU (batch_first)
out: (B, T, H)            H = hidden_dim * (2 if bidirectional else 1)
   │
   ▼  nn.MultiheadAttention (self-attention, batch_first)
        query = key = value = out
attended: (B, T, H)
   │
   ▼  mean-pool over time
pooled: (B, H)
   │
   ▼  nn.Linear
logits: (B, num_classes)
```

- **GRU** encodes the frame sequence as in `GRUDetector`.
- **Self-attention** (`nn.MultiheadAttention`, `batch_first=True`) lets every
  GRU timestep attend to every other timestep, adding global temporal context
  on top of the GRU's sequential summary.
- **Pooling: mean-pool over time.** Because self-attention already mixes global
  context into every position, averaging gives each frame an equal vote and is
  robust to varying sequence length `T` (unlike taking only the last GRU step).
- **Linear classifier** maps the pooled vector to the 48 action classes.

### Interface (drop-in replacement for `GRUDetector`)

```python
logits, h = model(x, h=None)   # x: (B, T, feature_dim), logits: (B, num_classes)
```

The `(logits, h)` tuple matches `GRUDetector` exactly, so it works unchanged in
`train_epoch`, `eval_stage`, and the model-agnostic `AGEM` gradient projection.

### Constructor

```python
GRUAttentionDetector(
    feature_dim=768,
    hidden_dim=256,
    num_classes=48,
    num_layers=1,
    dropout=0.0,
    bidirectional=False,
    num_heads=4,
)
```

`num_heads` must divide the attention dim (`hidden_dim * dirs`); if it does not,
the model automatically falls back to the largest valid divisor `<= num_heads`.

## Harness integration

`ExperimentConfig` gained an `arch` field (default `"gru"`). `make_model` builds
`GRUAttentionDetector` when `arch == "attn"`, otherwise `GRUDetector`. Two new
configs are registered: `attn_h256` and `attn_h512`. All existing configs are
unchanged and still build plain GRUs.

## Running the comparison

```bash
python3 experiments/run_capacity_ablation.py \
    --configs best_current_h256 attn_h256 \
    --feature-dir data/features_openclip_l14 \
    --feature-dim 768 \
    --seeds 0 1 2 \
    --epochs 15
```

This trains the baseline GRU (`best_current_h256`) and the attention model
(`attn_h256`) under identical A-GEM continual-learning settings across 3 seeds
and prints an avg-accuracy / forgetting summary table.

## Tests

```bash
python3 -m pytest tests/test_gru_attention.py -q
```

Covers (on synthetic data): the `(logits, h)` interface and shapes across batch
and sequence-length variation, gradient flow through attention + classifier,
an A-GEM `precompute_ref` / `apply` projection cycle, and harness wiring.
