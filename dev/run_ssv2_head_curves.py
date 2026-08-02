"""
SSv2 (our 48-class subset) accuracy-vs-classes-seen curves for NCM / Deep SLDA
/ FeCAM -- the SSv2 counterpart to dev/run_ucf101_tcd_protocol.py's curves.

Why: reports/cpu_friendly_methods_result.md and ucf101_tcd_result.md only ever
compared FeCAM's number across UCF101 (static-biased) vs SSv2 (temporal-biased).
The other two heads' final numbers were on record too (NCM 12.4/SLDA 14.1 vs
FeCAM 15.7 full-48-way) but nobody had plotted the SESSION-BY-SESSION curve for
them on SSv2 -- dev/run_modern_cpu_methods.py only ever saved the final
aggregate, not a per-stage curve. This fills that gap so the UCF101 and SSv2
figures are built the same way and can sit side by side.

Protocol: identical to every other SSv2 report in this repo -- our 48 classes,
curriculum-ordered 8 stages of 6 classes each (same STAGES as
dev/run_modern_cpu_methods.py), true class-IL evaluation (no task id, argmax
over every class seen so far) after each stage. Single deterministic order
(this repo's curriculum order, not randomized -- unlike the TCD/UCF101 protocol
which averages 3 random orders) since that is the order used everywhere else
SSv2 numbers are reported in this repo.

Run: python3 dev/run_ssv2_head_curves.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.models.fecam_head import FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEATURE_DIR = Path("data/features")
FEATURE_DIM = 512
N_CLASSES = 48
CPS = 6
N_STAGES = N_CLASSES // CPS
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_STAGES + 1)}


def load_split(samples, split):
    X, y = [], []
    d = FEATURE_DIR / split
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).mean(axis=0))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


class NCM:
    name = "NCM prototype"

    def __init__(self):
        self.means = np.zeros((N_CLASSES, FEATURE_DIM))
        self.counts = np.zeros(N_CLASSES)

    def observe(self, X, y):
        for c in np.unique(y):
            m = X[y == c]
            self.means[c] = m.mean(axis=0)
            self.counts[c] = len(m)

    def scores(self, X):
        mu = self.means / (np.linalg.norm(self.means, axis=1, keepdims=True) + 1e-12)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        s = Xn @ mu.T
        s[:, self.counts == 0] = -1e9
        return s


class SLDA:
    name = "Deep SLDA"

    def __init__(self, shrink=1e-2):
        self.means = np.zeros((N_CLASSES, FEATURE_DIM))
        self.counts = np.zeros(N_CLASSES)
        self.cov = np.zeros((FEATURE_DIM, FEATURE_DIM))
        self.n = 0
        self.shrink = shrink

    def observe(self, X, y):
        for c in np.unique(y):
            m = X[y == c]
            tot = self.counts[c] + len(m)
            self.means[c] = (self.means[c] * self.counts[c] + m.sum(axis=0)) / tot
            self.counts[c] = tot
        D = X - self.means[y]
        self.cov = (self.cov * self.n + D.T @ D) / (self.n + len(X))
        self.n += len(X)

    def scores(self, X):
        prec = np.linalg.inv(self.cov + self.shrink * np.eye(FEATURE_DIM))
        W = self.means @ prec
        b = -0.5 * np.einsum("cd,cd->c", W, self.means)
        s = X @ W.T + b
        s[:, self.counts == 0] = -1e9
        return s


def make_fecam():
    h = FeCAMHead(feature_dim=FEATURE_DIM, max_classes=N_CLASSES)
    h.name = "FeCAM (shared cov)"
    return h


HEADS = {"NCM prototype": NCM, "Deep SLDA": SLDA, "FeCAM (shared cov)": make_fecam}


def run_head(make, Xtr, ytr, Xva, yva):
    head = make()
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
    print("loading features...")
    Xtr, ytr = load_split(train_all, "train")
    Xva, yva = load_split(val_all, "val")
    print(f"  train {Xtr.shape}, val {Xva.shape}, {N_STAGES} stages x {CPS} classes")

    results = {}
    print(f"{'head':22s} {'avg inc acc':>12s} {'last':>8s} {'fit(s)':>8s}  per-stage")
    for name, make in HEADS.items():
        r = run_head(make, Xtr, ytr, Xva, yva)
        curve = " ".join(f"{100*a:.1f}" for a in r["per_step"])
        print(f"{name:22s} {100*r['avg_inc']:11.2f} {100*r['last']:7.2f} "
              f"{r['fit_s']:8.2f}  {curve}")
        results[name] = r

    out = Path("reports/ssv2_head_curves_raw.json")
    save_results(out, results)
    print(f"raw -> {out}")


if __name__ == "__main__":
    main()
