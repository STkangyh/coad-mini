"""
Ablation: does the "adjacent diff" term help, isolated for k=3 AND k=4 segments?

src/models/fecam_head.py has two segment poolings in production:
  _pool_chunks4        -- 4 segment means, NO diff (the deployed default)
  _pool_chunks3_adjdiff -- 3 segment means + 2 adjacent diffs (the previous default)

Both are already measured on the real 8-stage SSv2 val protocol (23.56 / 24.71
last, ssv2_temporal_pooling_result.md Sec10), but that pairing crosses TWO
variables at once (segment count AND diff presence), so neither number in
isolation says whether diff itself helped at either k. The clean with/without
pairs were never run on this protocol:
  - chunks3 (3 segment means, no diff) only exists as "thirds" in
    run_ssv2_temporal_pooling.py -- a differently-coded pooling (floor-division
    boundaries) that happens to produce identical segment boundaries to
    _segment_means(w,3) for T=16, but was never run in the same script/table as
    chunks3_adjdiff for a direct paired comparison.
  - chunks4+adjdiff (4 segment means + 3 adjacent diffs) has NEVER been run on
    this protocol at all. It appears exactly once in the whole project, in
    ssv2_temporal_pooling_result.md Sec10(b)'s 5-seed sweep -- but that sweep
    selects on a TRAIN-internal held-out split, not val, and reports mean+-sd
    over seeds, not a single val number comparable to the others (43.83+-3.89
    avg_inc there vs the "last on val" convention used everywhere else).

This script reuses _segment_means from src/models/fecam_head.py -- the actual
deployed pooling primitive -- for all four variants, rather than reimplementing
segment logic by hand (see fecam_vs_slda_covariance_result.md Sec4 for why that
matters: a hand-reimplementation there silently diverged from the real head by
0.2-0.35pp until the script was rewritten to call the real class/helpers
directly). Harness (STAGES, FeCAMHead call, load_split) is copied verbatim from
run_ssv2_temporal_pooling.py::run_fecam so results land in the same units.

Because `last` is provably invariant to class order (the final-stage head has
seen every class regardless of which order stages arrived in -- established in
ssv2_temporal_pooling_result.md Sec10(d)), chunks3+adjdiff and chunks4's `last`
here MUST reproduce 24.71 / 23.56 exactly regardless of this script's (simple,
sequential, unshuffled) stage order -- checked automatically before trusting the
two new numbers (chunks3, chunks4+adjdiff).

Run: python3 dev/run_diff_ablation.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.models.fecam_head import FeCAMHead, _segment_means  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEATURE_DIR = Path("data/features")
N_CLASSES = 48
CPS = 6
N_STAGES = N_CLASSES // CPS
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_STAGES + 1)}


# ── poolings: real _segment_means primitive, diff added/omitted around it ──────
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
    "chunks3": pool_chunks3,
    "chunks3+adjdiff": pool_chunks3_adjdiff,
    "chunks4": pool_chunks4,
    "chunks4+adjdiff": pool_chunks4_adjdiff,
}


def load_split(samples, split, pool):
    X, y = [], []
    d = FEATURE_DIR / split
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(pool(np.load(p)))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


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

    print("SSv2 48-class, 8-stage true class-IL, FeCAM head -- diff ablation, k=3 and k=4\n")
    print(f"{'pooling':20s} {'dim':>6s} {'avg_inc':>9s} {'last':>8s} {'fit(s)':>7s}")
    print("-" * 58)

    results = {}
    for name, pool in POOLINGS.items():
        Xtr, ytr = load_split(train_all, "train", pool)
        Xva, yva = load_split(val_all, "val", pool)
        r = run_fecam(Xtr, ytr, Xva, yva, Xtr.shape[1])
        print(f"{name:20s} {Xtr.shape[1]:6d} {100*r['avg_inc']:8.2f} "
              f"{100*r['last']:7.2f} {r['fit_s']:7.2f}", flush=True)
        results[name] = {**r, "dim": int(Xtr.shape[1])}

    print("\nconsistency check vs known val-protocol 'last' values "
          "(ssv2_temporal_pooling_result.md Sec10):")

    def chk(name, got, want):
        ok = abs(got - want) < 1e-3
        print(f"  {name:20s} {100*got:6.2f}  (expect {100*want:.2f})  "
              f"{'PASS' if ok else 'FAIL'}")

    chk("chunks3+adjdiff", results["chunks3+adjdiff"]["last"], 0.2471)
    chk("chunks4", results["chunks4"]["last"], 0.2356)

    print("\nnew numbers (never measured on this protocol before):")
    print(f"  chunks3          last={100*results['chunks3']['last']:.2f}  "
          f"avg_inc={100*results['chunks3']['avg_inc']:.2f}")
    print(f"  chunks4+adjdiff  last={100*results['chunks4+adjdiff']['last']:.2f}  "
          f"avg_inc={100*results['chunks4+adjdiff']['avg_inc']:.2f}")

    out = Path("reports/diff_ablation_raw.json")
    save_results(out, results)
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
