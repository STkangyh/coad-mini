# Anomaly / novel-action (OOD) detection

This module flags when the action in the prediction stream is **anomalous or
out-of-distribution** — an action the GRUDetector was never trained on, or
simply a low-confidence / novel observation. The objective is **novelty / OOD
detection**, not classification.

Code: [`src/anomaly/detector.py`](../src/anomaly/detector.py) ·
Calibration: [`experiments/calibrate_anomaly.py`](../experiments/calibrate_anomaly.py) ·
Endpoint: `POST /predict_anomaly` in [`app.py`](../app.py) ·
Tests: [`tests/test_anomaly.py`](../tests/test_anomaly.py)

## Scoring convention

Everything uses a single convention:

> **Higher anomaly score == more out-of-distribution.**

So one rule decides everything: `score >= threshold  =>  anomaly`.

Three scoring functions on the classifier logits implement this convention:

| function | definition | in-dist | OOD |
|---|---|---|---|
| `max_softmax_prob(logits)` | `-max(softmax(logits))` (negative MSP) | low (confident, MSP→1) | high (uncertain, MSP→0) |
| `predictive_entropy(logits)` | Shannon entropy of `softmax(logits)` (nats) | low (peaky) | high (uniform) |
| `energy_score(logits, T=1.0)` | `-T · logsumexp(logits / T)` | low / very negative (large logits) | high (small logits) |

All three accept a `torch.Tensor`, `np.ndarray`, or a plain sequence, reduce
the last (class) axis, and return a python `float` for a single vector or an
`np.ndarray` for a batch. `anomaly_score(logits, metric=..., T=...)` dispatches
by name (`"energy"`, `"entropy"`, `"msp"`).

`energy_score` is the default metric: it tends to separate in-dist vs OOD best
because it uses the *magnitude* of the logits, not just their normalized shape.

## `AnomalyScorer`

Holds a chosen metric, a calibrated threshold, and temporal-smoothing state.

```python
from src.anomaly.detector import AnomalyScorer

scorer = AnomalyScorer(metric="energy", smoothing="ema", ema_alpha=0.5)
scorer.calibrate(in_dist_scores, target_fpr=0.05)   # sets the threshold
out = scorer.score_window(model, feat_window)        # (16, 512) -> dict
# out = {score, raw_score, is_anomaly, top1, top1_prob, threshold}
```

- **`calibrate(in_dist_scores, target_fpr=0.05)`** sets the threshold at the
  `1 - target_fpr` quantile of the in-distribution score distribution. By
  construction only ~`target_fpr` of in-distribution windows land at/above the
  threshold and get (falsely) flagged. Default target FPR is 5%.
- **`score_window(model, feat_window, smooth=True)`** runs the GRUDetector on a
  single `(T, F)` or `(1, T, F)` window and returns
  `{score, raw_score, is_anomaly, top1, top1_prob, threshold}`. `raw_score` is
  always the unsmoothed per-window score; `score`/`is_anomaly` use the smoothed
  value when `smooth=True`.
- **Temporal smoothing** (`smoothing="ema"` or `"mean"`) is applied across a
  stream so a single noisy frame does not flip the flag. `score_stream(...)`
  scores an iterable of windows as one smoothed stream; `reset_stream()` clears
  the state between independent streams.

## Choosing the threshold (calibration)

`experiments/calibrate_anomaly.py` computes the in-distribution score
distribution for a checkpoint and picks the threshold:

```bash
# synthetic in-distribution windows (no real features required)
python3 experiments/calibrate_anomaly.py --metric energy --target-fpr 0.05

# realistic calibration from real CLIP features (read-only)
python3 experiments/calibrate_anomaly.py \
    --feature-dir data/features --metric energy --target-fpr 0.05 --out calib.json
```

It prints the chosen threshold plus in-dist / synthetic-OOD distribution
summaries, and optionally writes a JSON calibration with `--out`.

> Note: the synthetic-OOD line is a sanity check only. It is meaningful when the
> model has actually been trained on the in-distribution data (as in the test
> suite, which trains a small self-contained GRU). The shipped 48-class
> checkpoint was trained on real CLIP features, so feed it real features
> (`--feature-dir data/features`) for a meaningful in-distribution distribution.

## Demo endpoint

`POST /predict_anomaly` accepts base64 PNG frames (same payload as
`/predict_rt`) and returns:

```json
{ "top1": "Opening book", "top1_class_id": 0, "top1_prob": 0.83,
  "anomaly_score": -3.21, "is_anomaly": false,
  "metric": "energy", "threshold": null }
```

Configure at startup with env vars:

- `ANOMALY_METRIC` — `energy` (default), `entropy`, or `msp`.
- `ANOMALY_THRESHOLD` — decision threshold from calibration. If unset,
  `is_anomaly` is always `false` (the endpoint just returns the score).

```bash
ANOMALY_METRIC=energy ANOMALY_THRESHOLD=0.74 uvicorn app:app --port 8000
```
