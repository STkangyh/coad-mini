"""
Does order-preserving pooling beat mean pooling on a SECOND frozen backbone?

Every pooling ablation in the paper (segment count, adjacent-diff, head
comparison) runs on frozen CLIP ViT-B/32 only. A reviewer's fair question:
is the gain a property of the task (order information exists and matters
for SSv2's directional classes) or an artifact of CLIP specifically (a
static-image model with zero built-in temporal awareness, which is exactly
why mean pooling collapses so cleanly in our setup)?

This reuses the exact same 48-class curriculum, sample IDs, and pooling
primitive (`_segment_means`) as dev/run_diff_ablation.py, swapping only the
feature source: data/features_openclip_l14 (OpenCLIP ViT-L/14, D=768)
instead of data/features (CLIP ViT-B/32, D=512). Confirmed the same 9502
sample IDs (4800 train + 4702 val) exist under both feature directories
before writing this script.

Run: python3 dev/run_backbone_generalization_ssv2.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.models.fecam_head import FeCAMHead, _segment_means  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEATURE_DIR = Path("data/features_openclip_l14")
N_CLASSES = 48
CPS = 6
N_STAGES = N_CLASSES // CPS
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_STAGES + 1)}


def pool_mean(w):
    return w.mean(axis=0)


def pool_chunks3(w):
    return _segment_means(w, 3)


def pool_chunks3_adjdiff(w):
    d = w.shape[1]
    c = _segment_means(w, 3)
    return np.concatenate([c, c[d:2 * d] - c[:d], c[2 * d:] - c[d:2 * d]])


def pool_chunks4(w):
    return _segment_means(w, 4)


def pool_chunks4_adjdiff(w):
    d = w.shape[1]
    c = _segment_means(w, 4)
    return np.concatenate([
        c,
        c[d:2 * d] - c[:d],
        c[2 * d:3 * d] - c[d:2 * d],
        c[3 * d:] - c[2 * d:3 * d],
    ])


POOLINGS = {
    "mean": pool_mean,
    "chunks3": pool_chunks3,
    "chunks4": pool_chunks4,
    "chunks3+adjdiff": pool_chunks3_adjdiff,
    "chunks4+adjdiff": pool_chunks4_adjdiff,
}


def load_split(samples, split, pool):
    X, y, missing = [], [], 0
    d = FEATURE_DIR / split
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(pool(np.load(p)))
            y.append(s["class_id"])
        else:
            missing += 1
    return np.stack(X).astype(np.float64), np.array(y), missing


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

    print("SSv2 48-class, 8-stage true class-IL, FeCAM head")
    print("Backbone: OpenCLIP ViT-L/14 (D=768) -- NOT the paper's default CLIP B/32 (D=512)")
    print(f"{'pooling':20s} {'dim':>6s} {'avg_inc':>9s} {'last':>8s} {'fit(s)':>7s} {'missing':>8s}")
    print("-" * 68)

    results = {}
    for name, pool in POOLINGS.items():
        Xtr, ytr, miss_tr = load_split(train_all, "train", pool)
        Xva, yva, miss_va = load_split(val_all, "val", pool)
        r = run_fecam(Xtr, ytr, Xva, yva, Xtr.shape[1])
        print(f"{name:20s} {Xtr.shape[1]:6d} {100*r['avg_inc']:8.2f} "
              f"{100*r['last']:7.2f} {r['fit_s']:7.2f} {miss_tr+miss_va:8d}", flush=True)
        results[name] = {**r, "dim": int(Xtr.shape[1])}

    base_last, base_avg = results["mean"]["last"], results["mean"]["avg_inc"]
    print("\ngain over mean-pool baseline (OpenCLIP L/14):")
    for name, r in results.items():
        if name == "mean":
            continue
        print(f"  {name:20s} last {100*(r['last']-base_last):+6.2f}pp   "
              f"avg_inc {100*(r['avg_inc']-base_avg):+6.2f}pp")

    out = Path("reports/backbone_generalization_ssv2_l14_raw.json")
    save_results(out, {"backbone": "openclip_l14", "dim_native": 768, "results": results})
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
