"""
Classic supervised HDC (Hyperdimensional Computing) classifier vs NCM/SLDA/FeCAM.

recent_cpu_only_cl_survey_2026_result.md flagged ImageHD (arXiv:2604.21280) as a
candidate for avoiding FeCAM's O(D^3) covariance-refresh cost entirely. ImageHD
itself doesn't transplant directly -- it is UNSUPERVISED (Table I: "Sup.=Unsup."）
clustering with kMeans++ merging, and its headline 40.4x/383x numbers are an
FPGA-vs-CPU/GPU hardware comparison, not an algorithm-level speedup available in
pure CPU code. What DOES transplant is the general HDC idea ImageHD builds on:
encode features into high-dimensional bipolar hypervectors via random projection,
bundle (sum) each class's training hypervectors into one class hypervector, and
classify new samples by cosine similarity to those class hypervectors -- no
covariance, no matrix inversion, and incremental by construction (bundling a new
class is just adding to a running sum, exactly like NCM's running mean but in
hyperdimensional space).

Same 8-stage SSv2 protocol as ssv2_head_curves_result.md, so NCM/SLDA/FeCAM here
must reproduce 20.89/22.77/24.19 (avg_inc) exactly -- checked automatically.

Run: python3 dev/run_hdc_comparison.py [--hdc-dim 10000]
"""
import argparse
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


# ── reference heads (unmodified from ssv2_head_curves_result.md, used as the
#    consistency check for this script) ────────────────────────────────────
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


# ── HDC: random-projection encoding + bundled class hypervectors ───────────
class HDC:
    """Classic supervised HDC classifier.

    Encoding: h(x) = sign(R @ x), R a fixed random Gaussian projection
    (hdc_dim x feature_dim), giving a bipolar {-1,+1}^hdc_dim hypervector.
    This is the standard way to turn continuous (non-symbolic) features into
    hypervectors -- a random hyperplane projection is exactly what locality-
    sensitive hashing uses, and is the textbook HDC encoding for real-valued
    inputs (as opposed to level-hypervector encoding for scalar/pixel streams).

    Bundling: each class's hypervector is the running SUM of its members'
    encoded hypervectors (not re-binarized -- kept as a real-valued
    accumulator, matching common practical HDC implementations e.g. OnlineHD).
    Adding a class or more samples to an existing class is pure addition --
    no other class is touched, no matrix ever inverted.

    Classification: cosine similarity between the query's encoded hypervector
    and each class hypervector, argmax over active classes.
    """
    name = "HDC (bundled)"

    def __init__(self, hdc_dim=10000, seed=0):
        self.hdc_dim = hdc_dim
        rng = np.random.RandomState(seed)
        self.R = rng.standard_normal((hdc_dim, FEATURE_DIM)) / np.sqrt(FEATURE_DIM)
        self.class_hv = np.zeros((N_CLASSES, hdc_dim))
        self.counts = np.zeros(N_CLASSES)

    def _encode(self, X):
        proj = X @ self.R.T
        h = np.sign(proj)
        h[h == 0] = 1.0
        return h

    def observe(self, X, y):
        H = self._encode(X)
        for c in np.unique(y):
            self.class_hv[c] += H[y == c].sum(axis=0)
            self.counts[c] += (y == c).sum()

    def scores(self, X):
        H = self._encode(X)
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        Cn = self.class_hv / (np.linalg.norm(self.class_hv, axis=1, keepdims=True) + 1e-12)
        s = Hn @ Cn.T
        s[:, self.counts == 0] = -1e9
        return s


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdc-dim", type=int, default=10000,
                    help="hypervector dimension (10000 is HDC literature's standard default)")
    args = ap.parse_args()

    train_all = load_samples("data/subset/train_mini.json")
    val_all = load_samples("data/subset/val_mini.json")
    print("loading features...")
    Xtr, ytr = load_split(train_all, "train")
    Xva, yva = load_split(val_all, "val")
    print(f"  train {Xtr.shape}, val {Xva.shape}, {N_STAGES} stages x {CPS} classes\n")

    heads = [
        ("NCM prototype", NCM),
        ("Deep SLDA", SLDA),
        ("FeCAM (shared cov)", make_fecam),
        (f"HDC (bundled, D={args.hdc_dim})", lambda: HDC(hdc_dim=args.hdc_dim)),
    ]

    results = {}
    print(f"{'head':28s} {'avg inc acc':>12s} {'last':>8s} {'fit(s)':>8s}  per-stage")
    for name, make in heads:
        r = run_head(make, Xtr, ytr, Xva, yva)
        curve = " ".join(f"{100*a:.1f}" for a in r["per_step"])
        print(f"{name:28s} {100*r['avg_inc']:11.2f} {100*r['last']:7.2f} "
              f"{r['fit_s']:8.2f}  {curve}")
        results[name] = r

    print("\nconsistency check vs ssv2_head_curves_result.md (avg_inc): "
          f"NCM {'PASS' if abs(results['NCM prototype']['avg_inc']-0.2089)<1e-3 else 'FAIL'}  "
          f"SLDA {'PASS' if abs(results['Deep SLDA']['avg_inc']-0.2277)<1e-3 else 'FAIL'}  "
          f"FeCAM {'PASS' if abs(results['FeCAM (shared cov)']['avg_inc']-0.2419)<1e-3 else 'FAIL'}")

    out = Path("reports/hdc_comparison_raw.json")
    save_results(out, results)
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
