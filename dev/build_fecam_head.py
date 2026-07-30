"""
Build the FeCAM head checkpoint for the demo app.

Fits FeCAMHead on the 48-class training features (CLIP B/32) and saves it to
checkpoints/fecam_head.npz, then sanity-checks on val (both eval regimes).

The window pooling is baked into the checkpoint, so this must be rebuilt when the
default changes -- the served head pools requests the same way it was fitted.

Run: python3 dev/build_fecam_head.py [pooling]      # default: DEFAULT_POOLING
"""
import sys, time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from src.models.fecam_head import DEFAULT_POOLING, POOLINGS, FeCAMHead
from src.trainer import load_samples

FEATURE_DIR = Path("data/features")
OUT = Path("checkpoints/fecam_head.npz")
N_CLASSES = 48
CPS = 6
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_CLASSES // CPS + 1)}


def load_split(split, pooling):
    """Pools with the same function the served head will use at request time."""
    pool = POOLINGS[pooling]
    samples = load_samples(f"data/subset/{split}_mini.json")
    X, y = [], []
    d = FEATURE_DIR / ("train" if split == "train" else "val")
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(pool(np.load(p).astype(np.float64)))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


def main():
    pooling = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POOLING
    Xtr, ytr = load_split("train", pooling)
    Xva, yva = load_split("val", pooling)
    print(f"pooling={pooling}  train {Xtr.shape}, val {Xva.shape}")

    head = FeCAMHead(feature_dim=Xtr.shape[1], max_classes=256, pooling=pooling)
    t0 = time.perf_counter()
    # stage-wise observe (equivalent to batch for means/cov; mirrors CL arrival)
    for cids in STAGES.values():
        m = np.isin(ytr, cids)
        head.observe(Xtr[m], ytr[m])
    fit_s = time.perf_counter() - t0

    S = head.scores(Xva)
    full_acc = float((S.argmax(axis=1) == yva).mean())
    task_accs = []
    for cids in STAGES.values():
        m = np.isin(yva, cids)
        sub = S[m][:, cids]
        task_accs.append(float((np.array(cids)[sub.argmax(axis=1)] == yva[m]).mean()))
    print(f"fit: {fit_s:.1f}s   task-aware={np.mean(task_accs):.3f}   full-48-way={full_acc:.3f}")

    head.save(OUT)
    print(f"saved {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
