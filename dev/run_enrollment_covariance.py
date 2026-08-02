"""
Enrolled classes never touch the shared covariance. Does that cost anything?

Audit item 9. `FeCAMHead.enroll_class` calls observe(update_cov=False), so a class
registered live in the demo contributes a mean and nothing else: the covariance
keeps whatever the 48 (or 51) base classes gave it, forever. That was a
deliberate choice -- it is what makes enrollment forgetting-proof and O(D^2)
instead of O(D^3) -- but nobody measured whether the resulting mismatch hurts the
enrolled classes. A user who registers twenty new actions is scoring all of them
against a covariance that never saw them.

Three treatments over identical data, differing only in how the covariance is
handled:

  enroll        update_cov=False  -- the deployed path
  enroll+cov    update_cov=True   -- covariance grows with each new class
  joint         everything at once, covariance from all classes (reference)

Accuracy is broken out by BASE vs ENROLLED classes, because a covariance
mismatch should hurt the enrolled ones specifically -- an overall average would
hide it behind the base classes.

Two regimes, because they stress the question differently:
  few-shot   n_shot windows per new class (what the demo actually does; the
             covariance contribution would be tiny even if it were enabled)
  full       every training video of the new class (the largest contribution
             enrollment could possibly make)

Run: python3 dev/run_enrollment_covariance.py [--bench ucf101 ssv2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fecam_head import DEFAULT_POOLING, POOLINGS, FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

OUT = ROOT / "reports/enrollment_covariance_raw.json"


def load_bench(name):
    pool = POOLINGS[DEFAULT_POOLING]
    if name == "ucf101":
        d = ROOT / "data/features_ucf101_b32"
        man = json.loads((d / "manifest.json").read_text())
        out = {}
        for split in ("train", "test"):
            X, y = [], []
            for s in man["splits"][split]["samples"]:
                p = d / split / f"{s['id']}.npy"
                if p.exists():
                    X.append(pool(np.load(p).astype(np.float64)))
                    y.append(s["class_id"])
            out[split] = (np.stack(X), np.array(y))
        return out, 101, 51
    d = ROOT / "data/features"
    out = {}
    for split, sub, mani in (("train", "train", "train"), ("test", "val", "val")):
        X, y = [], []
        for s in load_samples(f"data/subset/{mani}_mini.json"):
            p = d / sub / f"{s['id']}.npy"
            if p.exists():
                X.append(pool(np.load(p).astype(np.float64)))
                y.append(s["class_id"])
        out[split] = (np.stack(X), np.array(y))
    return out, 48, 24


def split_acc(head, Xte, yte, base_ids, enrolled_ids):
    """Overall / base-only / enrolled-only accuracy over every class seen."""
    seen = list(base_ids) + list(enrolled_ids)
    m = np.isin(yte, seen)
    s = head.scores(Xte[m])
    s[:, [c for c in range(head.max_classes) if c not in seen]] = -1e18
    pred, truth = s.argmax(axis=1), yte[m]
    ok = pred == truth
    inb = np.isin(truth, base_ids)
    ine = np.isin(truth, enrolled_ids)
    return {
        "overall": float(ok.mean()),
        "base": float(ok[inb].mean()) if inb.any() else None,
        "enrolled": float(ok[ine].mean()) if ine.any() else None,
    }


def run(Xtr, ytr, Xte, yte, n_classes, base_ids, enrolled_ids, mode, n_shot, seed):
    """mode: 'enroll' | 'enroll+cov' | 'joint'."""
    rng = np.random.RandomState(seed)
    head = FeCAMHead(feature_dim=Xtr.shape[1], max_classes=n_classes)

    def samples_for(c):
        idx = np.where(ytr == c)[0]
        if n_shot is not None and len(idx) > n_shot:
            idx = rng.choice(idx, n_shot, replace=False)
        return idx

    if mode == "joint":
        idx = np.concatenate([np.where(ytr == c)[0] for c in base_ids]
                             + [samples_for(c) for c in enrolled_ids])
        head.observe(Xtr[idx], ytr[idx])                       # covariance from all
    else:
        m = np.isin(ytr, base_ids)
        head.observe(Xtr[m], ytr[m])                           # base fit, cov from base
        for c in enrolled_ids:                                 # then register, one by one
            idx = samples_for(c)
            head.observe(Xtr[idx], ytr[idx], update_cov=(mode == "enroll+cov"))
    return split_acc(head, Xte, yte, base_ids, enrolled_ids)


def shot_sweep(Xtr, ytr, Xte, yte, n_classes, n_base, seeds, shots=(5, 10, 25, 50, 100, None)):
    """How enrolled-class accuracy depends on how many windows it was taught from.

    The covariance treatment turned out not to matter; the number of examples
    does, enormously. Base classes always keep their full training set, so this
    isolates the asymmetry a live enrollment actually faces.
    """
    print(f"\n--- enrolled-class accuracy vs windows taught (10 enrolled, "
          f"base keeps all) ---")
    print(f"{'n_shot':>8s} {'enrolled':>10s} {'base':>10s}")
    print("-" * 32)
    out = {}
    for ns in shots:
        runs = []
        for seed in seeds:
            order = np.random.RandomState(seed).permutation(n_classes)
            runs.append(run(Xtr, ytr, Xte, yte, n_classes, order[:n_base],
                            order[n_base:n_base + 10], "enroll", ns, seed))
        agg = {k: float(np.mean([r[k] for r in runs])) for k in ("enrolled", "base")}
        out["all" if ns is None else ns] = agg
        print(f"{'all' if ns is None else ns:>8} {100*agg['enrolled']:9.2f}% "
              f"{100*agg['base']:9.2f}%", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["ucf101", "ssv2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-shot", type=int, default=5, help="windows per class, few-shot regime")
    args = ap.parse_args()

    results = {}
    for bench in args.bench:
        t0 = time.perf_counter()
        data, n_classes, n_base = load_bench(bench)
        (Xtr, ytr), (Xte, yte) = data["train"], data["test"]
        print(f"\n{'='*84}\n{bench}: {n_classes} classes, {n_base} base, "
              f"{Xtr.shape[1]}-d ({DEFAULT_POOLING})")

        for regime, n_shot in (("few-shot", args.n_shot), ("full-data", None)):
            print(f"\n--- {regime} enrollment "
                  f"({'%d windows/class' % n_shot if n_shot else 'all windows'}) ---")
            print(f"{'#enrolled':>10s} {'mode':12s} {'overall':>9s} {'base':>9s} {'enrolled':>9s}")
            print("-" * 56)
            for n_enroll in (5, 10, n_classes - n_base):
                if n_enroll > n_classes - n_base:
                    continue
                per_mode = {}
                for mode in ("enroll", "enroll+cov", "joint"):
                    runs = []
                    for seed in args.seeds:
                        order = np.random.RandomState(seed).permutation(n_classes)
                        base_ids, rest = order[:n_base], order[n_base:]
                        runs.append(run(Xtr, ytr, Xte, yte, n_classes, base_ids,
                                        rest[:n_enroll], mode, n_shot, seed))
                    agg = {k: float(np.mean([r[k] for r in runs if r[k] is not None]))
                           for k in ("overall", "base", "enrolled")}
                    per_mode[mode] = agg
                    print(f"{n_enroll:10d} {mode:12s} {100*agg['overall']:8.2f} "
                          f"{100*agg['base']:8.2f} {100*agg['enrolled']:8.2f}", flush=True)
                d = 100 * (per_mode["enroll+cov"]["enrolled"] - per_mode["enroll"]["enrolled"])
                print(f"{'':10s} {'-> cov would':12s} {'':8s} {'':8s} {d:+8.2f}  "
                      f"(gain on enrolled classes)")
                results[f"{bench}/{regime}/{n_enroll}"] = per_mode
        results[f"{bench}/shot_sweep"] = shot_sweep(
            Xtr, ytr, Xte, yte, n_classes, n_base, args.seeds)
        print(f"[{time.perf_counter()-t0:.0f}s]")

    save_results(OUT, results)
    print(f"\nraw -> {OUT}")


if __name__ == "__main__":
    main()
