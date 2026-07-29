"""
Can we recover SSv2 accuracy WITHOUT backprop, by not throwing the temporal
axis away?

Motivation (ESSENTIAL, ICCV'25, Table 3a -- read from the paper): their SSv2
ablation decomposes as

    ESSENTIAL baseline (frozen CLIP + LEARNED temporal encoder + replay)  42.1
    + MR module + semantic memory (the paper's actual contribution)       48.9

so the headline machinery is worth +6.8, while the step from a frozen encoder
to a *temporally modeled* representation is worth far more. Our FeCAM on the
same frozen CLIP is at 15.7 because we mean-pool 16 frames into one vector --
which destroys ordering entirely. TCD said the same thing in words: "naive
averaging of the features from all frames may not be suitable".

A learned temporal encoder would close much of that gap but breaks the
backprop-free property that is the whole point of our setting. So this script
asks the narrower question: how much of the temporal signal can a CLOSED-FORM
pooling recover? Every variant below is a fixed function of the (16, 512)
per-frame features -- no parameters, no gradients, no training.

The order-sensitive ones matter most: SSv2 pairs like "push left-to-right" vs
"push right-to-left" are identical under mean-pool but have opposite-signed
frame differences.

Protocol is identical to dev/run_ssv2_head_curves.py (8-stage curriculum, true
class-IL, FeCAM head) so numbers drop straight into that comparison; `mean`
reproduces the 15.74 already on record as the control.

Run: python3 dev/run_ssv2_temporal_pooling.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.models.fecam_head import FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402

FEATURE_DIR = Path("data/features")
N_CLASSES = 48
CPS = 6
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_CLASSES // CPS + 1)}


# ── closed-form poolings: (T, D) -> (D',) ────────────────────────────────────
def p_mean(f):
    """Current baseline. Order-blind by construction."""
    return f.mean(axis=0)


def p_mean_std(f):
    """Adds temporal variability, still order-blind (std is permutation-invariant)."""
    return np.concatenate([f.mean(axis=0), f.std(axis=0)])


def p_halves(f):
    """Coarsest possible ordering: what it looked like before vs after."""
    h = len(f) // 2
    return np.concatenate([f[:h].mean(axis=0), f[h:].mean(axis=0)])


def p_thirds(f):
    """Beginning / middle / end."""
    t = len(f) // 3
    return np.concatenate([f[:t].mean(axis=0), f[t:2 * t].mean(axis=0),
                           f[2 * t:].mean(axis=0)])


def p_mean_diff(f):
    """Mean plus the mean frame-to-frame delta -- the delta flips sign when the
    action is reversed, which is exactly what SSv2's confusable pairs need."""
    d = np.diff(f, axis=0)
    return np.concatenate([f.mean(axis=0), d.mean(axis=0)])


def p_diff_only(f):
    """Motion signal alone, appearance discarded."""
    return np.diff(f, axis=0).mean(axis=0)


def p_mean_diff_absdiff(f):
    """Mean + signed delta + magnitude of change (how much moved, regardless of
    direction)."""
    d = np.diff(f, axis=0)
    return np.concatenate([f.mean(axis=0), d.mean(axis=0), np.abs(d).mean(axis=0)])


def _chunks(f, k):
    """k equal temporal segments, mean-pooled independently -> (k*D,)."""
    idx = np.linspace(0, len(f), k + 1).astype(int)
    return np.concatenate([f[idx[i]:idx[i + 1]].mean(axis=0) for i in range(k)])


def p_chunks3_adjdiff(f, d=512):
    """BEST closed-form variant found. Three segment means plus the differences
    between adjacent segments -- the segments carry "what it looked like at each
    phase", the adjacent differences carry "how it changed between phases", and
    those differences flip sign under action reversal.

    A chunk sweep (reports/ssv2_chunk_sweep_raw.json) shows accuracy peaks at
    k=3-4 and then declines: more segments buy temporal resolution but inflate
    the feature dimension, and FeCAM has to estimate a DxD covariance from ~100
    samples per class. 3 segments is the sweet spot for this data size.
    """
    c = _chunks(f, 3)
    return np.concatenate([c, c[d:2 * d] - c[:d], c[2 * d:] - c[d:2 * d]])


POOLINGS = {
    "mean (baseline)": p_mean,
    "mean+std": p_mean_std,
    "halves": p_halves,
    "thirds": p_thirds,
    "mean+diff": p_mean_diff,
    "diff only": p_diff_only,
    "mean+diff+|diff|": p_mean_diff_absdiff,
    "chunks3+adjdiff (best)": p_chunks3_adjdiff,
}


def load_split(samples, split, pool):
    X, y = [], []
    d = FEATURE_DIR / split
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(pool(np.load(p)))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


def run_fecam(Xtr, ytr, Xva, yva, dim):
    head = FeCAMHead(feature_dim=dim, max_classes=N_CLASSES)
    seen, accs = [], []
    t0 = time.perf_counter()
    for cids in STAGES.values():
        m = np.isin(ytr, cids)
        head.observe(Xtr[m], ytr[m])
        seen.extend(cids)
        va = np.isin(yva, seen)
        s = head.scores(Xva[va])
        s[:, [c for c in range(N_CLASSES) if c not in seen]] = -1e18
        accs.append(float((s.argmax(axis=1) == yva[va]).mean()))
    return {"per_step": accs, "avg_inc": float(np.mean(accs)),
            "last": accs[-1], "fit_s": time.perf_counter() - t0}


def main():
    train_all = load_samples("data/subset/train_mini.json")
    val_all = load_samples("data/subset/val_mini.json")

    print("SSv2 48-class, 8-stage true class-IL, FeCAM head — closed-form poolings only\n")
    print(f"{'pooling':22s} {'dim':>6s} {'avg inc':>9s} {'last':>8s} {'vs base':>9s} {'fit(s)':>7s}")
    print("-" * 68)

    results, base_last = {}, None
    for name, pool in POOLINGS.items():
        Xtr, ytr = load_split(train_all, "train", pool)
        Xva, yva = load_split(val_all, "val", pool)
        r = run_fecam(Xtr, ytr, Xva, yva, Xtr.shape[1])
        if base_last is None:
            base_last = r["last"]
        delta = 100 * (r["last"] - base_last)
        print(f"{name:22s} {Xtr.shape[1]:6d} {100*r['avg_inc']:8.2f} "
              f"{100*r['last']:7.2f} {delta:+8.2f} {r['fit_s']:7.2f}", flush=True)
        results[name] = {**r, "dim": int(Xtr.shape[1])}

    out = Path("reports/ssv2_temporal_pooling_raw.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
