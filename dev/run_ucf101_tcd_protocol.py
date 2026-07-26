"""
UCF101 class-incremental evaluation under the TCD protocol.

This is our first number on a benchmark other papers also report, so it can be
placed directly beside theirs. Everything before this (48-class SSv2 subset)
only supported statements about *orderings* between methods.

Protocol (TCD, Park et al. ICCV'21, Sec 4.2 -- verified from the paper):
  * UCF101, official recognition split 1 (9537 train / 3783 test, 101 classes)
  * initial model trained on 51 classes, remaining 50 arriving in groups of
    10 / 5 / 2  -> the "10 x 5", "5 x 10", "2 x 25 stages" columns in their tables
  * class order shuffled randomly; results averaged over 3 orders
    (TCD's own seeds: 1000, 1993, 2021)

    VERIFIED, not assumed: TCD ships the order it used as class_list.pkl, and
    that file is element-for-element identical (101/101) to
    np.random.RandomState(1000).permutation(101), i.e. exactly what this script
    generates. Their scripts/ucf101/ucf101_51_10.sh confirms the rest --
    --seed 1000/1993/2021, --init_task 51, --nb_class 10, --K 5
    --budget_type class, --store_frames uniform. So these runs use TCD's
    literal class orders, not merely similar random ones.
  * metric: average incremental accuracy -- mean of the accuracies measured
    after every incremental step, each over all classes seen so far (no task id)

Reported for reference (UCF101, 10x5 stages, each paper's own table):
    TCD (ICCV'21, ResNet-34+TSM)   CNN 74.89 / NME 77.16
    FrameMaker (NeurIPS'22)            78.13
    STSP (ECCV'24, exemplar-free)      81.15
    ST-prompt (CLIP)                   84.8
    ESSENTIAL (ICCV'25, frozen CLIP)   95.1
Our heads are exemplar-free and backprop-free on a frozen CLIP ViT-B/32.
NOTE these are different tracks -- ours does not train the backbone at all --
so the table is a positioning aid, not a like-for-like ranking. Say so in the
paper (same rule as reports/pycil_bridge_result.md Sec 4).

Usage:
  python3 dev/run_ucf101_tcd_protocol.py                 # all 3 increments, 3 seeds
  python3 dev/run_ucf101_tcd_protocol.py --inc 10        # one setting
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fecam_head import FeCAMHead  # noqa: E402

FEAT = ROOT / "data/features_ucf101_b32"     # overridable via --features
N_CLASSES = 101
BASE = 51                      # TCD's initial stage for UCF101
DIM = 512                                    # set from the manifest at run time
SEEDS = [1000, 1993, 2021]     # TCD's own class-order seeds
INCREMENTS = [10, 5, 2]


# ── heads (same implementations as the SSv2 study, sized for 101 classes) ─────
class NCM:
    name = "NCM prototype"

    def __init__(self):
        self.means = np.zeros((N_CLASSES, DIM))
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
    """Batched shared-covariance LDA (the SSv2 runner used a per-sample
    streaming variant; batched here since we only need the same statistics)."""
    name = "Deep SLDA"

    def __init__(self, shrink=1e-2):
        self.means = np.zeros((N_CLASSES, DIM))
        self.counts = np.zeros(N_CLASSES)
        self.cov = np.zeros((DIM, DIM))
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
        prec = np.linalg.inv(self.cov + self.shrink * np.eye(DIM))
        W = self.means @ prec
        b = -0.5 * np.einsum("cd,cd->c", W, self.means)
        s = X @ W.T + b
        s[:, self.counts == 0] = -1e9
        return s


def make_fecam():
    h = FeCAMHead(feature_dim=DIM, max_classes=N_CLASSES)
    h.name = "FeCAM (shared cov)"
    return h


HEADS = {"NCM prototype": NCM, "Deep SLDA": SLDA, "FeCAM (shared cov)": make_fecam}


# ── data ─────────────────────────────────────────────────────────────────────
def load_split(split):
    """Mean-pool each (16, 512) window to a single 512-d video embedding."""
    man = json.loads((FEAT / "manifest.json").read_text())
    X, y = [], []
    for s in man["splits"][split]["samples"]:
        p = FEAT / split / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).mean(axis=0))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


def sessions_for(order, inc):
    """51-class base, then `inc`-sized groups over the remaining 50."""
    out = [list(order[:BASE])]
    for i in range(BASE, N_CLASSES, inc):
        out.append(list(order[i:i + inc]))
    return out


def run_head(make, sessions, Xtr, ytr, Xte, yte):
    head = make()
    seen, accs = [], []
    t0 = time.perf_counter()
    for cids in sessions:
        m = np.isin(ytr, cids)
        head.observe(Xtr[m], ytr[m])
        seen.extend(cids)
        te = np.isin(yte, seen)
        s = head.scores(Xte[te])
        s[:, [c for c in range(N_CLASSES) if c not in seen]] = -1e18
        accs.append(float((s.argmax(axis=1) == yte[te]).mean()))
    return {"per_step": accs, "avg_inc": float(np.mean(accs)),
            "last": accs[-1], "fit_s": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inc", type=int, nargs="+", default=INCREMENTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--features", type=Path, default=None,
                    help="feature dir (default: CLIP B/32); use to compare encoders")
    ap.add_argument("--tag", default=None, help="label for the raw-results file")
    args = ap.parse_args()

    global FEAT, DIM
    if args.features:
        FEAT = args.features if args.features.is_absolute() else ROOT / args.features

    Xtr, ytr = load_split("train")
    Xte, yte = load_split("test")
    DIM = Xtr.shape[1]                       # encoders differ in width
    print(f"UCF101 official split 1 — train {Xtr.shape}, test {Xte.shape}, "
          f"{len(np.unique(ytr))} classes")
    print(f"TCD protocol: {BASE}-class base + increments of {args.inc}, "
          f"seeds {args.seeds}\n")

    results = {}
    for inc in args.inc:
        n_steps = 1 + (N_CLASSES - BASE + inc - 1) // inc
        print(f"=== increment {inc}  ({inc} x {n_steps - 1} stages, "
              f"{n_steps} evaluation points) ===")
        print(f"{'head':22s} {'avg inc acc':>12s} {'last':>8s} {'fit(s)':>8s}")
        for name, make in HEADS.items():
            per_seed = []
            for seed in args.seeds:
                order = np.random.RandomState(seed).permutation(N_CLASSES)
                per_seed.append(run_head(make, sessions_for(order, inc),
                                         Xtr, ytr, Xte, yte))
            avg = np.mean([r["avg_inc"] for r in per_seed])
            std = np.std([r["avg_inc"] for r in per_seed])
            last = np.mean([r["last"] for r in per_seed])
            fit = np.mean([r["fit_s"] for r in per_seed])
            print(f"{name:22s} {100*avg:9.2f}±{100*std:.2f} {100*last:7.2f} "
                  f"{fit:8.1f}")
            results[f"inc{inc}/{name}"] = {
                "avg_inc": float(avg), "avg_inc_std": float(std),
                "last": float(last), "fit_s": float(fit),
                "per_seed": per_seed,
            }
        print()

    tag = args.tag or ("" if not args.features else "_" + FEAT.name.replace("features_ucf101_", ""))
    out = ROOT / f"reports/ucf101_tcd_raw{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"raw -> {out}")


if __name__ == "__main__":
    main()
