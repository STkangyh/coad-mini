"""
CPU-friendly, backprop-free continual learning on frozen CLIP features.

Motivation: our whole stack already trains on CPU (~4.5 min/seed for GRU+A-GEM).
This experiment asks: can we go even lighter — methods with NO backpropagation
at all, just streaming statistics with closed-form classifiers? These are the
standard "frozen-feature CL" family:

  1. NCM  (nearest class mean, SimpleCIL-style): store one mean per class,
     predict by cosine to the nearest mean. Zero training.
  2. Deep SLDA (Hayes & Kanan, CVPR-W 2020): class means + ONE shared
     covariance, streaming updates; linear discriminant at eval.
  3. Ridge / RanPAC-lite (McDonnell et al., NeurIPS 2023 without the prompt
     part): accumulate Gram statistics G += H^T H, C += H^T Y over the stream
     (optionally after a fixed random ReLU projection to expand dimension),
     solve W = (G + lambda*I)^-1 C in closed form at eval.

All three are exemplar-free (no replay buffer), single-pass, and by
construction accumulate per-class statistics — so "forgetting" can only come
from shared statistics drifting (SLDA covariance / ridge Gram), not from
gradient interference.

Protocol matches the A-GEM experiments: same 8-stage arrival of 48 classes,
CLIP B/32 features (16,512) mean-pooled over frames to one 512-d video vector.
Metrics: task-aware 6-way avg acc, full-48-way acc/F1/mAP, S1 drop, wall-clock.

Run: python3 dev/run_cpu_friendly_methods.py
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
RIDGE_LAMBDA = 1e-2
RP_DIM = 2000          # RanPAC-lite random-projection width (paper uses 10k; 2k is CPU-quick)
RP_SEEDS = [0, 1, 2]   # ridge+RP has randomness; NCM/SLDA/plain-ridge are deterministic

train_all = load_samples("data/subset/train_mini.json")
val_all = load_samples("data/subset/val_mini.json")


def video_embedding(sample, split_dir):
    p = split_dir / f"{sample['id']}.npy"
    if not p.exists():
        return None
    return np.load(p).mean(axis=0)            # (16,512) -> (512,) mean-pool


def load_split(samples, split):
    X, y = [], []
    d = FEATURE_DIR / split
    for s in samples:
        e = video_embedding(s, d)
        if e is not None:
            X.append(e); y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


print("loading features (mean-pooled video embeddings)...")
t0 = time.perf_counter()
Xtr, ytr = load_split(train_all, "train")
Xva, yva = load_split(val_all, "val")
print(f"  train {Xtr.shape}, val {Xva.shape}  ({time.perf_counter()-t0:.1f}s)")


# ── metrics ───────────────────────────────────────────────────────────────────
def full_48way(scores, y_true):
    y_pred = scores.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    # softmax for mAP comparability with the GRU report
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


# ── 1. NCM (SimpleCIL-style prototype) ────────────────────────────────────────
class NCM:
    name = "NCM prototype (SimpleCIL-style)"

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


# ── 2. Deep SLDA (streaming means + shared covariance) ────────────────────────
class SLDA:
    name = "Deep SLDA (Hayes & Kanan '20)"

    def __init__(self, shrink=1e-2):
        self.means = np.zeros((N_CLASSES, FEATURE_DIM))
        self.counts = np.zeros(N_CLASSES)
        self.cov = np.zeros((FEATURE_DIM, FEATURE_DIM))
        self.n = 0
        self.shrink = shrink

    def observe(self, X, y):
        for x, c in zip(X, y):                       # true streaming updates
            self.n += 1
            if self.counts[c] == 0:
                self.means[c] = x
            else:
                self.means[c] += (x - self.means[c]) / (self.counts[c] + 1)
            self.counts[c] += 1
            d = (x - self.means[c])[:, None]
            self.cov += (d @ d.T - self.cov) / self.n

    def scores(self, X):
        prec = np.linalg.inv(self.cov + self.shrink * np.eye(FEATURE_DIM))
        W = self.means @ prec                         # (C, D)
        b = -0.5 * np.einsum("cd,cd->c", W, self.means)
        s = X @ W.T + b
        s[:, self.counts == 0] = -1e9
        return s


# ── 3. Ridge / RanPAC-lite (recursive least squares, optional RP) ─────────────
class Ridge:
    def __init__(self, rp_seed=None):
        self.rp = None
        dim = FEATURE_DIM
        if rp_seed is not None:
            rng = np.random.default_rng(rp_seed)
            self.rp = rng.standard_normal((FEATURE_DIM, RP_DIM)) / np.sqrt(FEATURE_DIM)
            dim = RP_DIM
        self.name = f"Ridge RLS{' + random projection ' + str(RP_DIM) if self.rp is not None else ' (closed-form, ACIL-style)'}"
        self.G = np.zeros((dim, dim))
        self.C = np.zeros((dim, N_CLASSES))

    def _phi(self, X):
        return np.maximum(X @ self.rp, 0.0) if self.rp is not None else X

    def observe(self, X, y):
        H = self._phi(X)
        Y = np.eye(N_CLASSES)[y]
        self.G += H.T @ H
        self.C += H.T @ Y

    def scores(self, X):
        W = np.linalg.solve(self.G + RIDGE_LAMBDA * np.eye(self.G.shape[0]), self.C)
        return self._phi(X) @ W


# ── run the 8-stage protocol ──────────────────────────────────────────────────
def run(model):
    t0 = time.perf_counter()
    s1_after_s1 = None
    for stage_id, cids in STAGES.items():
        mask = np.isin(ytr, cids)
        model.observe(Xtr[mask], ytr[mask])
        if stage_id == 1:
            _, per_stage = task_aware(model.scores(Xva), yva)
            s1_after_s1 = per_stage[0]
    train_s = time.perf_counter() - t0
    sc = model.scores(Xva)
    ta, per_stage = task_aware(sc, yva)
    facc, ff1, fmap = full_48way(sc, yva)
    return {"task_aware": ta, "s1_drop": s1_after_s1 - per_stage[0],
            "full_acc": facc, "full_f1": ff1, "full_mAP": fmap, "train_s": train_s}


def show(name, r):
    print(f"{name:44s} task={r['task_aware']:.3f}  s1_drop={r['s1_drop']:+.3f}  "
          f"full_acc={r['full_acc']:.3f}  full_f1={r['full_f1']:.3f}  "
          f"full_mAP={r['full_mAP']:.3f}  train={r['train_s']:.1f}s")


def main():
    print(f"\n{'='*100}")
    print("  CPU-friendly backprop-free CL on frozen CLIP B/32 (8-stage, exemplar-free, single pass)")
    print(f"{'='*100}")
    show(NCM.name, run(NCM()))
    show(SLDA.name, run(SLDA()))
    show("Ridge RLS (closed-form, ACIL-style)", run(Ridge(rp_seed=None)))
    rs = [run(Ridge(rp_seed=s)) for s in RP_SEEDS]
    agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    std = float(np.std([r["task_aware"] for r in rs]))
    print(f"{'Ridge RLS + random projection '+str(RP_DIM)+f' ({len(RP_SEEDS)} seeds)':44s} "
          f"task={agg['task_aware']:.3f}±{std:.3f}  s1_drop={agg['s1_drop']:+.3f}  "
          f"full_acc={agg['full_acc']:.3f}  full_f1={agg['full_f1']:.3f}  "
          f"full_mAP={agg['full_mAP']:.3f}  train={agg['train_s']:.1f}s")
    print("\nReference (backprop, 15 epochs x 8 stages, ~4.5 min/seed):")
    print("  GRU + A-GEM: task=0.387±0.018  s1_drop=-0.061  full_acc=0.105  full_f1=0.075  full_mAP=0.105")
    print("  (full-48-way GRU numbers from reports/data_scale_full48way_result.md @100% data)")
    print("ALL DONE")


if __name__ == "__main__":
    main()
