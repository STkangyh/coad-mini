"""
Calibrate the anomaly / OOD detector
====================================

Computes the in-distribution anomaly-score distribution for a trained
GRUDetector checkpoint and picks a decision threshold at the
``(1 - target_fpr)`` quantile (see ``src/anomaly/detector.py`` for the scoring
convention: higher score == more out-of-distribution).

Usage
-----
Synthetic in-distribution windows (no real features needed)::

    python3 experiments/calibrate_anomaly.py --metric energy --target-fpr 0.05

Realistic calibration from real CLIP features (read-only)::

    python3 experiments/calibrate_anomaly.py \
        --feature-dir /Users/younghoon-kang/coad-mini/data/features \
        --metric energy --target-fpr 0.05

The script prints the chosen threshold and a short distribution summary, and
(optionally) writes the calibration to a JSON file with ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# allow running as `python3 experiments/calibrate_anomaly.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gru_detector import GRUDetector  # noqa: E402
from src.anomaly.detector import AnomalyScorer, anomaly_score  # noqa: E402

N_FRAMES = 16


# ──────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GRUDetector(
        feature_dim=ckpt.get("feature_dim", 512),
        hidden_dim=ckpt.get("hidden_dim", 256),
        num_classes=ckpt.get("n_classes", 48),
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.0),
        bidirectional=ckpt.get("bidirectional", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def iter_real_windows(feature_dir: Path, feature_dim: int, limit: int | None):
    """Yield (T, F) windows from real .npy CLIP feature files (read-only)."""
    paths = sorted(feature_dir.rglob("*.npy"))
    if limit is not None:
        paths = paths[:limit]
    n = 0
    for p in paths:
        try:
            arr = np.load(p)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[-1] != feature_dim:
            continue
        yield arr.astype(np.float32)
        n += 1
    if n == 0:
        raise RuntimeError(
            f"no usable (.npy, shape (T,{feature_dim})) features found under {feature_dir}"
        )


def synthetic_in_dist_windows(n: int, feature_dim: int, rng: np.random.Generator):
    """Generate synthetic in-distribution windows.

    We mimic L2-normalised CLIP frame features clustered around a handful of
    class-prototype directions, so the trained classifier responds confidently.
    """
    n_proto = 8
    protos = rng.standard_normal((n_proto, feature_dim)).astype(np.float32)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    windows = []
    for _ in range(n):
        c = rng.integers(n_proto)
        noise = 0.15 * rng.standard_normal((N_FRAMES, feature_dim)).astype(np.float32)
        w = protos[c][None, :] + noise
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        windows.append(w.astype(np.float32))
    return windows


def synthetic_ood_windows(n: int, feature_dim: int, rng: np.random.Generator):
    """Generate synthetic OOD windows (random / large-shift, unnormalised)."""
    windows = []
    for _ in range(n):
        shift = rng.uniform(-3.0, 3.0)
        w = (rng.standard_normal((N_FRAMES, feature_dim)).astype(np.float32) + shift)
        windows.append(w.astype(np.float32))
    return windows


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Calibrate the anomaly/OOD detector.")
    ap.add_argument("--ckpt", default="checkpoints/agem_48cls.pt",
                    help="model checkpoint (default: checkpoints/agem_48cls.pt)")
    ap.add_argument("--feature-dir", default=None,
                    help="optional read-only dir of real .npy CLIP features for "
                         "realistic in-distribution calibration")
    ap.add_argument("--metric", default="energy",
                    choices=["energy", "entropy", "msp"])
    ap.add_argument("--T", type=float, default=1.0, help="temperature for energy")
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="target in-distribution false-positive rate")
    ap.add_argument("--n-synth", type=int, default=512,
                    help="number of synthetic windows when no feature-dir")
    ap.add_argument("--limit", type=int, default=512,
                    help="max real feature files to read")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")
    model, ckpt = load_model(ckpt_path)
    feature_dim = ckpt.get("feature_dim", 512)
    print(f"Loaded {ckpt_path}  (feature_dim={feature_dim}, "
          f"n_classes={ckpt.get('n_classes', 48)})")

    scorer = AnomalyScorer(metric=args.metric, T=args.T)

    # ── in-distribution windows ───────────────────────────────────────────────
    if args.feature_dir:
        fdir = Path(args.feature_dir).expanduser()
        if not fdir.exists():
            sys.exit(f"feature-dir not found: {fdir}")
        print(f"Using REAL features (read-only) from {fdir}")
        in_windows = list(iter_real_windows(fdir, feature_dim, args.limit))
    else:
        print(f"Using SYNTHETIC in-distribution windows (n={args.n_synth})")
        in_windows = synthetic_in_dist_windows(args.n_synth, feature_dim, rng)

    # raw (unsmoothed) per-window scores for calibration
    in_scores = np.array(
        [scorer.score_window(model, w, smooth=False)["raw_score"] for w in in_windows],
        dtype=np.float64,
    )

    threshold = scorer.calibrate(in_scores, target_fpr=args.target_fpr)

    in_flag_rate = float(np.mean(in_scores >= threshold))

    # ── synthetic OOD windows for a quick sanity check ────────────────────────
    ood_windows = synthetic_ood_windows(min(len(in_windows), 256), feature_dim, rng)
    ood_scores = np.array(
        [scorer.score_window(model, w, smooth=False)["raw_score"] for w in ood_windows],
        dtype=np.float64,
    )
    ood_flag_rate = float(np.mean(ood_scores >= threshold))

    print("\n── Calibration result ──────────────────────────────────────────")
    print(f"metric            : {args.metric}  (T={args.T})")
    print(f"target FPR        : {args.target_fpr:.3f}")
    print(f"threshold         : {threshold:.4f}")
    print(f"in-dist  n={len(in_scores):4d}  mean={in_scores.mean():.4f}  "
          f"std={in_scores.std():.4f}  flagged={in_flag_rate:.3f}")
    print(f"ood(synth) n={len(ood_scores):4d}  mean={ood_scores.mean():.4f}  "
          f"std={ood_scores.std():.4f}  flagged={ood_flag_rate:.3f}")
    print(f"separation (ood_mean - in_mean) = {ood_scores.mean() - in_scores.mean():.4f}")

    if args.out:
        out = {
            "ckpt": str(ckpt_path),
            "metric": args.metric,
            "T": args.T,
            "target_fpr": args.target_fpr,
            "threshold": threshold,
            "in_dist": {
                "n": len(in_scores),
                "mean": float(in_scores.mean()),
                "std": float(in_scores.std()),
                "flag_rate": in_flag_rate,
                "source": "real" if args.feature_dir else "synthetic",
            },
            "ood_synth": {
                "n": len(ood_scores),
                "mean": float(ood_scores.mean()),
                "std": float(ood_scores.std()),
                "flag_rate": ood_flag_rate,
            },
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote calibration to {args.out}")


if __name__ == "__main__":
    main()
