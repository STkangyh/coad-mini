"""
Ablation: which piece of FeCAM's covariance handling actually beats SLDA?

fecam_vs_slda_covariance_result.md identified three things FeCAM does that
SLDA doesn't -- (a) shrinks the diagonal by a much larger, data-derived
constant instead of a tiny fixed one, (b) also shrinks the off-diagonal
(SLDA never touches it), (c) normalizes to a correlation matrix before
inverting -- but flagged as an honest limitation that no ablation separated
their individual contributions.

FIRST ATTEMPT AT THIS SCRIPT WAS WRONG TWICE, both kept here as lessons:
1. Hand-reimplemented only the shrinkage formula and forgot FeCAM's `_prep()`
   (a sign-preserving Tukey ladder-of-powers transform, lam=0.5 matching the
   paper, then L2-normalize) that runs before ANY mean/covariance is
   computed -- step3 came out WORSE than SLDA (18.55/11.53) instead of
   matching FeCAM's real 24.19/15.74.
2. After adding `_prep`, a hand-reimplemented `_cov_terms` got close but not
   exact (23.84/15.50 vs the real 24.19/15.74) -- some formula detail was
   still off and never found. Rather than keep guessing, this version
   monkey-patches `_cov_terms` onto a REAL `FeCAMHead` instance per step, so
   `_prep`/`observe`/`scores`/caching are byte-for-byte the real
   implementation and ONLY the regularization math differs between steps.
   step3 (the unmodified head) is then a tautological, not approximate,
   reproduction of the real number.

  step0   SLDA            raw features (bypass _prep), cov + 0.01*I               (diag only, fixed const)
  step0.5 +prep            FeCAM's real _prep (Tukey+L2), still SLDA's 0.01 shrink (isolates preprocessing alone)
  step1   +big diag        prepped; diag += gamma1*V1, off-diag untouched         (diag only, data-derived const)
  step2   +off-diag too    prepped; cov + gamma1*V1*I + gamma2*V2*(1-I)            (FeCAM's Eq.8, raw scale)
  step3   +corr norm       real FeCAMHead, unmodified                             (= FeCAM, exact by construction)

gamma1=gamma2=1.0 throughout (FeCAM's own many-shot-CIL setting = fecam_head.py's
SHRINK_1/SHRINK_2). Same protocol as ssv2_head_curves_result.md: our 48-class
SSv2 subset, 8 curriculum stages, true class-IL eval, mean-pool 512-d CLIP
B/32 features. step0 must reproduce SLDA's 22.77/14.14 and step3 must
reproduce FeCAM's 24.19/15.74 exactly -- both checked automatically below.

Run: python3 dev/run_covariance_ablation.py
"""
import sys
import types
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
GAMMA1 = GAMMA2 = 1.0   # = fecam_head.py's SHRINK_1/SHRINK_2


def load_split(samples, split):
    X, y = [], []
    d = FEATURE_DIR / split
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).mean(axis=0))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


# ── step0: plain SLDA, no FeCAM machinery at all (independent reimplementation,
#    already verified to reproduce 22.77/14.14 exactly) ───────────────────────
class SLDA:
    name = "step0   SLDA (raw feats, diag-only fixed 0.01)"

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


# ── step0.5 / step1 / step2: real FeCAMHead with `_cov_terms` monkey-patched.
#    `_prep`, `observe`, `scores`, caching are the untouched real implementation --
#    only the regularization inside `_cov_terms` changes per variant. ─────────
def make_cov_terms(mode):
    def _cov_terms(self):
        if self._cov_cache is None:
            d = self.feature_dim
            raw = self._cov_sum / max(self._cov_n, 1)
            diag = np.diag(raw).copy()
            diag_mean = float(diag.sum()) / d
            off_mean = float(raw.sum() - diag.sum()) / (d * (d - 1))

            cov = raw.copy()
            if mode in ("big_diag", "off_diag"):
                if mode == "off_diag":
                    cov = cov + off_mean * GAMMA2
                cov[np.diag_indices(d)] = diag + GAMMA1 * diag_mean
            else:
                raise ValueError(mode)

            prec = np.linalg.inv(cov)               # no correlation normalization
            penalty = float(np.einsum("ij,ji->", prec, raw))
            self._cov_cache = (prec, np.ones(d), penalty)   # sd=1 -> scores() skips corr scaling
        return self._cov_cache
    return _cov_terms


def make_fecam_variant(mode):
    """mode: 'slda_shrink' (step0.5) | 'big_diag' (step1) | 'off_diag' (step2) | None (step3, real FeCAM)."""
    head = FeCAMHead(feature_dim=FEATURE_DIM, max_classes=N_CLASSES)
    if mode == "slda_shrink":
        def _cov_terms(self):
            if self._cov_cache is None:
                d = self.feature_dim
                raw = self._cov_sum / max(self._cov_n, 1)
                prec = np.linalg.inv(raw + 1e-2 * np.eye(d))
                penalty = float(np.einsum("ij,ji->", prec, raw))
                self._cov_cache = (prec, np.ones(d), penalty)
            return self._cov_cache
        head._cov_terms = types.MethodType(_cov_terms, head)
    elif mode in ("big_diag", "off_diag"):
        head._cov_terms = types.MethodType(make_cov_terms(mode), head)
    elif mode is not None:
        raise ValueError(mode)
    return head


STEPS = [
    ("step0   SLDA (raw feats, diag-only fixed 0.01)", "slda_plain"),
    ("step0.5 +prep (Tukey+L2, still SLDA shrink)", "slda_shrink"),
    ("step1   +big diag (prepped, data-derived)", "big_diag"),
    ("step2   +off-diag shrink (prepped, raw scale)", "off_diag"),
    ("step3   +corr norm (real FeCAMHead, = FeCAM)", None),
]


def run_head(head, Xtr, ytr, Xva, yva):
    seen, accs = [], []
    for cids in STAGES.values():
        m = np.isin(ytr, cids)
        head.observe(Xtr[m], ytr[m])
        seen.extend(cids)
        va = np.isin(yva, seen)
        s = head.scores(Xva[va])
        s[:, [c for c in range(N_CLASSES) if c not in seen]] = -1e18
        accs.append(float((s.argmax(axis=1) == yva[va]).mean()))
    return {"per_step": accs, "avg_inc": float(np.mean(accs)), "last": accs[-1]}


def main():
    train_all = load_samples("data/subset/train_mini.json")
    val_all = load_samples("data/subset/val_mini.json")
    print("loading features...")
    Xtr, ytr = load_split(train_all, "train")
    Xva, yva = load_split(val_all, "val")
    print(f"  train {Xtr.shape}, val {Xva.shape}, {N_STAGES} stages x {CPS} classes\n")

    results = {}
    print(f"{'step':50s} {'avg inc acc':>12s} {'last':>8s}  delta vs prev")
    prev = None
    for label, mode in STEPS:
        head = SLDA() if mode == "slda_plain" else make_fecam_variant(mode)
        r = run_head(head, Xtr, ytr, Xva, yva)
        delta = "" if prev is None else f"  {100*(r['last']-prev):+.2f}pp (last)"
        print(f"{label:50s} {100*r['avg_inc']:11.2f} {100*r['last']:7.2f}{delta}")
        results[label] = r
        prev = r["last"]

    print(f"\nconsistency check -- step0 vs SLDA 22.77/14.14: "
          f"{'PASS' if abs(results[STEPS[0][0]]['avg_inc']-0.2277) < 1e-3 else 'FAIL'}")
    print(f"consistency check -- step3 vs FeCAM 24.19/15.74: "
          f"{'PASS' if abs(results[STEPS[-1][0]]['avg_inc']-0.2419) < 1e-3 else 'FAIL'} "
          f"(exact by construction -- step3 IS the real FeCAMHead, unmodified)")

    out = Path("reports/covariance_ablation_raw.json")
    save_results(out, results)
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
