"""
Does order-preserving pooling help on UCF101, or only on SSv2?

Why this exists: the UCF101 numbers in the paper (Table `tab:ucf101-compare`,
88.84/88.86/88.84) were produced by dev/run_ucf101_tcd_protocol.py, whose
loader mean-pools each (16, 512) window -- i.e. the headline comparison
against the literature does NOT use this paper's proposed pooling. The paper
never states which pooling UCF101 used, while the Abstract and Conclusion
phrase the result as belonging to "the method". That has to be resolved one
way or the other, and the resolution depends on a number we did not have.

This runs the identical TCD protocol (51-class base, increments of 10/5/2,
TCD's own class-order seeds 1000/1993/2021) on the identical features
(data/features_ucf101_b32, raw 16x512 per clip), changing only the pooling
function -- reusing `_segment_means` from the deployed head so these are the
same functions Section 3.3 defines, not re-implementations.

FeCAM head only: it is the deployed head and the one in the paper's
comparison table, so sweeping NCM/SLDA here would triple the cost without
touching the question.

Run: python3 dev/run_ucf101_pooling_sweep.py
     python3 dev/run_ucf101_pooling_sweep.py --inc 10     # headline column only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fecam_head import FeCAMHead, _segment_means  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEAT = ROOT / "data/features_ucf101_b32"
N_CLASSES = 101
BASE = 51                       # TCD's initial stage for UCF101
SEEDS = [1000, 1993, 2021]      # TCD's own class-order seeds
INCREMENTS = [10, 5, 2]


# ── poolings: identical to dev/run_diff_ablation.py and the L/14 study ────────
def pool_mean(w):
    return w.mean(axis=0)


def pool_chunks3(w):
    return _segment_means(w, 3)


def pool_chunks4(w):
    return _segment_means(w, 4)


def pool_chunks3_adjdiff(w):
    d = w.shape[1]
    c = _segment_means(w, 3)
    return np.concatenate([c, c[d:2 * d] - c[:d], c[2 * d:] - c[d:2 * d]])


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
    "mean": pool_mean,                          # what the paper's table used
    "chunks3": pool_chunks3,
    "chunks4": pool_chunks4,
    "chunks3+adjdiff": pool_chunks3_adjdiff,
    "chunks4+adjdiff": pool_chunks4_adjdiff,
}


def load_split(split, pool):
    man = json.loads((FEAT / "manifest.json").read_text())
    X, y = [], []
    for s in man["splits"][split]["samples"]:
        p = FEAT / split / f"{s['id']}.npy"
        if p.exists():
            X.append(pool(np.load(p)))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


def sessions_for(order, inc):
    out = [list(order[:BASE])]
    for i in range(BASE, N_CLASSES, inc):
        out.append(list(order[i:i + inc]))
    return out


def run_fecam(sessions, Xtr, ytr, Xte, yte, dim):
    head = FeCAMHead(feature_dim=dim, max_classes=N_CLASSES)
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
    ap.add_argument("--poolings", nargs="+", default=list(POOLINGS),
                    choices=list(POOLINGS))
    args = ap.parse_args()

    print("UCF101, TCD protocol (51-class base), FeCAM head, CLIP ViT-B/32")
    print(f"increments={args.inc}  seeds={args.seeds}  poolings={args.poolings}\n")

    results = {}
    for pooling in args.poolings:
        pool = POOLINGS[pooling]
        t0 = time.perf_counter()
        Xtr, ytr = load_split("train", pool)
        Xte, yte = load_split("test", pool)
        dim = Xtr.shape[1]
        print(f"[{pooling}] D={dim}  loaded in {time.perf_counter()-t0:.0f}s "
              f"(train {Xtr.shape[0]}, test {Xte.shape[0]})", flush=True)

        for inc in args.inc:
            per_seed = []
            for seed in args.seeds:
                order = np.random.RandomState(seed).permutation(N_CLASSES)
                per_seed.append(run_fecam(sessions_for(order, inc),
                                          Xtr, ytr, Xte, yte, dim))
            avg = float(np.mean([r["avg_inc"] for r in per_seed]))
            std = float(np.std([r["avg_inc"] for r in per_seed]))
            last = float(np.mean([r["last"] for r in per_seed]))
            fit = float(np.mean([r["fit_s"] for r in per_seed]))
            print(f"    inc={inc:<3d} avg_inc {100*avg:6.2f}±{100*std:.2f}  "
                  f"last {100*last:6.2f}  ({fit:.0f}s/seed)", flush=True)
            results[f"{pooling}|inc{inc}"] = {
                "pooling": pooling, "inc": inc, "dim": int(dim),
                "avg_inc": avg, "avg_inc_std": std, "last": last,
                "fit_s": fit, "per_seed": per_seed,
            }
        print(flush=True)
        del Xtr, Xte

    # ── summary: the shape needed to decide the paper's framing ──────────────
    print("=" * 72)
    print(f"{'pooling':18s} {'D':>5s} " + " ".join(f"{'inc'+str(i):>14s}" for i in args.inc))
    print("-" * 72)
    for pooling in args.poolings:
        row = f"{pooling:18s}"
        d = results.get(f"{pooling}|inc{args.inc[0]}", {}).get("dim", 0)
        row += f" {d:5d} "
        for inc in args.inc:
            r = results.get(f"{pooling}|inc{inc}")
            row += f" {100*r['avg_inc']:8.2f}±{100*r['avg_inc_std']:4.2f}" if r else " " * 15
        print(row)

    base = {inc: results.get(f"mean|inc{inc}") for inc in args.inc}
    if all(base.values()):
        print("\ngain over mean pooling (avg_inc, pp):")
        for pooling in args.poolings:
            if pooling == "mean":
                continue
            deltas = []
            for inc in args.inc:
                r = results.get(f"{pooling}|inc{inc}")
                if r:
                    deltas.append(f"inc{inc} {100*(r['avg_inc']-base[inc]['avg_inc']):+6.2f}")
            print(f"  {pooling:18s} " + "   ".join(deltas))

    out = ROOT / "reports/ucf101_pooling_sweep_raw.json"
    save_results(out, {"protocol": {"base": BASE, "increments": args.inc,
                                    "seeds": args.seeds},
                       "results": results})
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
