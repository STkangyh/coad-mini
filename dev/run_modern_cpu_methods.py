"""
Newer (2023-2024) backprop-free CL methods on frozen CLIP features — follow-up
to dev/run_cpu_friendly_methods.py (which covered NCM / Deep SLDA / ridge RLS).

  1. FeCAM (Goswami et al., NeurIPS 2023): Mahalanobis classifier with
     per-class covariance + the paper's normalization tricks — sign-preserving
     Tukey power transform, covariance shrinkage, correlation normalization.
     Exemplar-free: stores one (mean, covariance) pair per class.
  2. FeCAM-shared: same, but one covariance shared across classes (paper's
     common-covariance mode; memory = SLDA).
  3. RanDumb-style (Prabhu et al., NeurIPS 2024): fixed random Fourier
     features approximating an RBF kernel (bandwidth from the median
     heuristic on stage-1 data only), then a streaming SLDA-style linear
     classifier on top. The paper's point: a *random* representation +
     streaming linear head beats learned online-CL representations.

Same protocol as before: 8-stage arrival, CLIP B/32 features mean-pooled to
512-d, task-aware 6-way + full-48-way metrics, single pass, no replay buffer.

Run: python3 dev/run_modern_cpu_methods.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score

from src.trainer import load_samples

FEATURE_DIR = Path("data/features")
FEATURE_DIM = 512
N_CLASSES = 48
CPS = 6
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_CLASSES // CPS + 1)}

train_all = load_samples("data/subset/train_mini.json")
val_all = load_samples("data/subset/val_mini.json")


def load_split(samples, split):
    X, y = [], []
    d = FEATURE_DIR / split
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).mean(axis=0)); y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


print("loading features...")
Xtr, ytr = load_split(train_all, "train")
Xva, yva = load_split(val_all, "val")
print(f"  train {Xtr.shape}, val {Xva.shape}")


def full_48way(scores, y_true):
    y_pred = scores.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    mAP = average_precision_score(np.eye(N_CLASSES)[y_true], probs, average="macro")
    return acc, f1, mAP


def task_aware(scores, y_true):
    accs = []
    for cids in STAGES.values():
        mask = np.isin(y_true, cids)
        sub = scores[mask][:, cids]
        pred = np.array([cids[i] for i in sub.argmax(axis=1)])
        accs.append(accuracy_score(y_true[mask], pred))
    return float(np.mean(accs)), accs


# ── 1/2. FeCAM ────────────────────────────────────────────────────────────────
def tukey(X, lam=0.5):
    """Sign-preserving Tukey ladder-of-powers (CLIP features have negatives)."""
    return np.sign(X) * (np.abs(X) ** lam)


class FeCAM:
    def __init__(self, per_class=True, shrink1=1.0, shrink2=1.0, lam=0.5):
        self.per_class = per_class
        self.s1, self.s2, self.lam = shrink1, shrink2, lam
        self.name = f"FeCAM ({'per-class' if per_class else 'shared'} cov, NeurIPS'23)"
        self.means, self.covs, self.counts = {}, {}, np.zeros(N_CLASSES)
        self.shared = np.zeros((FEATURE_DIM, FEATURE_DIM))
        self.n_shared = 0

    def _prep(self, X):
        X = tukey(X, self.lam)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def observe(self, X, y):
        X = self._prep(X)
        for c in np.unique(y):
            m = X[y == c]
            self.means[c] = m.mean(axis=0)
            self.counts[c] = len(m)
            cov = np.cov(m.T, bias=False) if len(m) > 1 else np.eye(FEATURE_DIM)
            if self.per_class:
                self.covs[c] = cov
            else:
                self.shared += cov * len(m)
                self.n_shared += len(m)

    @staticmethod
    def _normalize_cov(cov, s1, s2):
        # shrinkage (paper eq: Σ + γ1·diag-mean·I + γ2·offdiag-mean·(1-I))
        d = cov.shape[0]
        diag_mean = float(np.trace(cov)) / d
        off = cov - np.diag(np.diag(cov))
        off_mean = float(off.sum()) / (d * (d - 1))
        cov = cov + s1 * diag_mean * np.eye(d) + s2 * off_mean * (1 - np.eye(d))
        # correlation normalization
        sd = np.sqrt(np.diag(cov))
        return cov / np.outer(sd, sd), sd

    def scores(self, X):
        X = self._prep(X)
        S = np.full((len(X), N_CLASSES), -1e18)
        if not self.per_class:
            corr, sd = self._normalize_cov(self.shared / max(self.n_shared, 1), self.s1, self.s2)
            prec = np.linalg.inv(corr)
        for c, mu in self.means.items():
            if self.per_class:
                corr, sd = self._normalize_cov(self.covs[c], self.s1, self.s2)
                prec = np.linalg.inv(corr)
            D = (X - mu) / sd
            S[:, c] = -np.einsum("nd,de,ne->n", D, prec, D)
        return S


# ── 3. RanDumb-style: RFF (RBF kernel) + streaming SLDA head ─────────────────
class RanDumb:
    def __init__(self, rff_dim=2000, seed=0, shrink=1e-2):
        self.name = f"RanDumb-style RFF({rff_dim}) + SLDA head (NeurIPS'24)"
        self.rff_dim, self.seed, self.shrink = rff_dim, seed, shrink
        self.W = None; self.b = None
        self.means = np.zeros((N_CLASSES, rff_dim))
        self.counts = np.zeros(N_CLASSES)
        self.cov = np.zeros((rff_dim, rff_dim))
        self.n = 0

    def _fit_rff(self, X):
        rng = np.random.default_rng(self.seed)
        # median heuristic for RBF bandwidth, on the FIRST data seen only
        idx = rng.choice(len(X), size=min(400, len(X)), replace=False)
        d2 = np.sum((X[idx, None, :] - X[None, idx, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / med
        self.W = rng.standard_normal((FEATURE_DIM, self.rff_dim)) * np.sqrt(2 * gamma)
        self.b = rng.uniform(0, 2 * np.pi, self.rff_dim)

    def _phi(self, X):
        return np.sqrt(2.0 / self.rff_dim) * np.cos(X @ self.W + self.b)

    def observe(self, X, y):
        if self.W is None:
            self._fit_rff(X)
        H = self._phi(X)
        # batched streaming updates (means + shared covariance)
        for c in np.unique(y):
            m = H[y == c]
            tot = self.counts[c] + len(m)
            self.means[c] = (self.means[c] * self.counts[c] + m.sum(axis=0)) / tot
            self.counts[c] = tot
        D = H - self.means[y]
        self.cov = (self.cov * self.n + D.T @ D) / (self.n + len(H))
        self.n += len(H)

    def scores(self, X):
        prec = np.linalg.inv(self.cov + self.shrink * np.eye(self.rff_dim))
        Wc = self.means @ prec
        b = -0.5 * np.einsum("cd,cd->c", Wc, self.means)
        s = self._phi(X) @ Wc.T + b
        s[:, self.counts == 0] = -1e9
        return s


# ── run protocol ──────────────────────────────────────────────────────────────
def run(model):
    t0 = time.perf_counter()
    s1_after_s1 = None
    for stage_id, cids in STAGES.items():
        mask = np.isin(ytr, cids)
        model.observe(Xtr[mask], ytr[mask])
        if stage_id == 1:
            _, per = task_aware(model.scores(Xva), yva)
            s1_after_s1 = per[0]
    train_s = time.perf_counter() - t0
    sc = model.scores(Xva)
    ta, per = task_aware(sc, yva)
    facc, ff1, fmap = full_48way(sc, yva)
    return {"task": ta, "s1_drop": s1_after_s1 - per[0], "facc": facc,
            "ff1": ff1, "fmap": fmap, "train_s": train_s}


def show(name, r):
    print(f"{name:52s} task={r['task']:.3f}  s1_drop={r['s1_drop']:+.3f}  "
          f"full_acc={r['facc']:.3f}  full_f1={r['ff1']:.3f}  full_mAP={r['fmap']:.3f}  "
          f"train={r['train_s']:.1f}s", flush=True)


def main():
    print(f"\n{'='*110}")
    print("  Modern (2023-24) backprop-free CL on frozen CLIP B/32 — same 8-stage protocol")
    print(f"{'='*110}")
    m = FeCAM(per_class=True);  show(m.name, run(m))
    m = FeCAM(per_class=False); show(m.name, run(m))
    rs = [run(RanDumb(rff_dim=2000, seed=s)) for s in [0, 1, 2]]
    agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    std = float(np.std([r["task"] for r in rs]))
    print(f"{'RanDumb-style RFF(2000)+SLDA head, 3 seeds':52s} task={agg['task']:.3f}±{std:.3f}  "
          f"s1_drop={agg['s1_drop']:+.3f}  full_acc={agg['facc']:.3f}  full_f1={agg['ff1']:.3f}  "
          f"full_mAP={agg['fmap']:.3f}  train={agg['train_s']:.1f}s")
    print("\nReference:")
    print("  Deep SLDA (raw 512-d):  task=0.390  full_acc=0.141  full_f1=0.123  train=2.1s")
    print("  GRU + A-GEM (backprop): task=0.387  full_acc=0.105  full_f1=0.075  train=~270s")
    print("ALL DONE")


if __name__ == "__main__":
    main()
