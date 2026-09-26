"""
Order-preserving pooling on the LITERAL TCD 174-class SSv2 split.

Why this exists
---------------
The paper (reports/paper_draft.tex) tabulates the TCD/STSP/CSTA/ESSENTIAL
lineage on SSv2's standard 174-class split (Table 1, `tab:tcd-lineage`) but
reports its OWN numbers on a self-defined 48-class curriculum. A reviewer who
reads that table will ask why we do not report on the split we just tabulated.
This script closes that gap: same protocol, same class count, same session
structure as TCD, so our row can sit in the same table.

It differs from dev/run_ssv2_tcd_full_split.py, which asks the OLD question
(FeCAM vs GRU+A-GEM). This one asks the paper's question: does the
order-preserving pooling gain that we measured at 48 classes survive at 174?
So it sweeps the pooling variants of Sections 3.3-3.4 against the mean-pool
baseline, with the head held fixed.

Protocol (TCD, arXiv:2203.13611)
--------------------------------
  174 classes = 84 base + 90 incremental
  incremental delivered as 10 classes x 9 sessions  ("10x9")
                        or  5 classes x 18 sessions ("5x18")
  true class-IL evaluation: argmax over all classes seen so far, no task id

  "AxB" is A classes per session across B sessions -- the same convention the
  paper uses for UCF101 (10x5 -> 2x25 is described as splitting increments into
  progressively SMALLER groups, Section 4.2). make_sessions asserts it.

HONEST LIMITS -- state these in the paper, do not paper over them
-----------------------------------------------------------------
1. TCD does not publish WHICH 84 classes are base (we could not locate the
   list). The base/incremental partition here is random per seed, so the
   result does not hinge on one arbitrary partition -- but it reproduces
   TCD's STRUCTURE and SCALE, not its literal class assignment. Report
   mean +/- std across seeds and say so.
2. TCD/STSP/CSTA train their backbone ON SSv2; we use frozen CLIP that has
   never seen it. The comparison is therefore not like-for-like on accuracy,
   which is exactly the argument Section 2.1 already makes. This script
   supplies the number; the framing stays the paper's job.
3. Accuracy-averaging convention differs across this literature. We emit all
   three variants (see METRICS below) rather than silently picking one.

METRICS emitted per run
-----------------------
  last                 accuracy after the final session
  avg_inc_all          mean over every evaluation point, base session included
                       (this paper's convention, Table `tab:metrics`)
  avg_inc_excl_last    mean over the first S-1 points
                       (TCD/FrameMaker/ST-Prompt convention, confirmed against
                       the CSTA paper -- see `tab:ucf101-compare`'s caption)
  avg_inc_inc_only     mean over incremental sessions only, base excluded

Runtime
-------
Dominated by one D x D inverse per evaluation inside FeCAM. Rough single-run
cost on this Mac: mean (D=512) seconds, chunks4+adjdiff (D=3584) ~15s per
evaluation. Full sweep (5 poolings x 2 variants x 3 seeds) is roughly an hour.
Use --smoke first, and --poolings / --variants / --seeds to cut it down.

Run:
  python3 dev/run_ssv2_tcd_174_pooling.py --smoke
  python3 dev/run_ssv2_tcd_174_pooling.py
  python3 dev/run_ssv2_tcd_174_pooling.py --poolings mean chunks3+adjdiff --variants 10x9
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.fecam_head import FeCAMHead, _segment_means  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEATURE_DIR = Path("data/features_full")
TRAIN_JSON = "data/subset/train_full.json"
VAL_JSON = "data/subset/val_full.json"

N_CLASSES = 174
N_BASE = 84
N_INCREMENTAL = N_CLASSES - N_BASE      # 90

# TCD reports both; the paper's Table 1 has a "10x9 / 5x18" column.
# "AxB" reads as A classes per session across B sessions -- the convention the
# paper's own text depends on ("as increments are split into smaller and smaller
# groups", 10x5 -> 2x25 on UCF101, Section 4.2). Getting this backwards silently
# swaps which variant is the fine-grained one, so the name is asserted below.
VARIANTS = {"10x9": 10, "5x18": 5}      # name -> classes per incremental session


# ── poolings ──────────────────────────────────────────────────────────────────
# Built on the same _segment_means primitive the deployed head uses, so these
# are the functions the paper measured, not re-implementations of them.
def pool_mean(w):
    return w.mean(axis=0)


def _chunks_adjdiff(w, k):
    d = w.shape[1]
    c = _segment_means(w, k)
    diffs = [c[i * d:(i + 1) * d] - c[(i - 1) * d:i * d] for i in range(1, k)]
    return np.concatenate([c, *diffs])


POOLINGS = {
    "mean": pool_mean,                                  # baseline, D = 1x
    "chunks3": lambda w: _segment_means(w, 3),          # D = 3x
    "chunks4": lambda w: _segment_means(w, 4),          # D = 4x
    "chunks3+adjdiff": lambda w: _chunks_adjdiff(w, 3),  # D = 5x
    "chunks4+adjdiff": lambda w: _chunks_adjdiff(w, 4),  # D = 7x
}


# ── data ──────────────────────────────────────────────────────────────────────
def load_pooled(samples, split, pool_fn):
    """Read each feature file once, apply one pooling. Returns (X, y)."""
    d = FEATURE_DIR / split
    X, y, missing = [], [], 0
    for s in samples:
        p = d / f"{s['id']}.npy"
        if not p.exists():
            missing += 1
            continue
        X.append(pool_fn(np.load(p)))
        y.append(s["class_id"])
    if not X:
        raise SystemExit(f"no features found under {d} -- check FEATURE_DIR")
    return np.stack(X).astype(np.float64), np.array(y), missing


def group_by_class(X, y):
    return {int(c): X[y == c] for c in np.unique(y)}


def make_sessions(seed, variant):
    """[84 base classes] + incremental sessions, seeded.

    `variant` is a key of VARIANTS ("AxB" = A classes per session, B sessions).
    The session count is asserted against B so a mis-set table cannot pass.
    """
    inc_size = VARIANTS[variant]
    n_sessions = int(variant.split("x")[1])
    order = np.random.RandomState(seed).permutation(N_CLASSES)
    base, rest = list(order[:N_BASE]), order[N_BASE:]
    incs = [list(rest[i:i + inc_size]) for i in range(0, len(rest), inc_size)]
    assert sum(len(s) for s in incs) == N_INCREMENTAL, "incremental classes lost"
    assert len(incs) == n_sessions, (
        f"variant {variant!r} implies {n_sessions} sessions of {inc_size}, "
        f"but {len(incs)} were built -- VARIANTS and the name disagree")
    return [[int(c) for c in base]] + [[int(c) for c in s] for s in incs]


# ── evaluation ────────────────────────────────────────────────────────────────
def run_one(sessions, Xtr_by_class, Xva, yva, dim, few_shot_correction=False):
    """Incrementally enroll each session, evaluate true class-IL after each."""
    head = FeCAMHead(feature_dim=dim, max_classes=N_CLASSES,
                     few_shot_correction=few_shot_correction)
    seen, accs = [], []
    t0 = time.perf_counter()
    for cids in sessions:
        cids = [c for c in cids if c in Xtr_by_class]
        X = np.concatenate([Xtr_by_class[c] for c in cids])
        y = np.concatenate([np.full(len(Xtr_by_class[c]), c) for c in cids])
        head.observe(X, y)
        seen.extend(cids)

        m = np.isin(yva, seen)
        scores = head.scores(Xva[m])
        scores[:, [c for c in range(N_CLASSES) if c not in set(seen)]] = -1e18
        accs.append(float((scores.argmax(axis=1) == yva[m]).mean()))
    return {
        "per_session": accs,
        "last": accs[-1],
        "avg_inc_all": float(np.mean(accs)),
        "avg_inc_excl_last": float(np.mean(accs[:-1])) if len(accs) > 1 else accs[-1],
        "avg_inc_inc_only": float(np.mean(accs[1:])) if len(accs) > 1 else accs[-1],
        "fit_s": time.perf_counter() - t0,
    }


def agg(runs, key):
    v = [r[key] for r in runs]
    return float(np.mean(v)), float(np.std(v))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="base/incremental partitions to average over")
    ap.add_argument("--poolings", nargs="+", default=list(POOLINGS),
                    choices=list(POOLINGS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--few-shot-correction", action="store_true",
                    help="match the served head config. OFF by default so these "
                         "numbers stay comparable with the paper's 48-class "
                         "ablations (dev/run_diff_ablation.py), which leave it off.")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, mean + chunks3+adjdiff, 10x9 only -- a fast "
                         "end-to-end check before committing to the full sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.poolings, args.variants = [0], ["mean", "chunks3+adjdiff"], ["10x9"]

    train_all = load_samples(TRAIN_JSON)
    val_all = load_samples(VAL_JSON)

    print(f"SSv2 {N_CLASSES}-class, TCD protocol: {N_BASE} base + {N_INCREMENTAL} incremental")
    print(f"variants={args.variants}  poolings={args.poolings}  seeds={args.seeds}")
    print(f"few_shot_correction={args.few_shot_correction}")
    print(f"train={len(train_all)} val={len(val_all)} samples\n")

    results = {}
    for pooling in args.poolings:
        pool_fn = POOLINGS[pooling]
        t0 = time.perf_counter()
        Xtr, ytr, miss_tr = load_pooled(train_all, "train", pool_fn)
        Xva, yva, miss_va = load_pooled(val_all, "val", pool_fn)
        Xtr_by_class = group_by_class(Xtr, ytr)
        dim = Xtr.shape[1]
        print(f"[{pooling}] D={dim}  loaded in {time.perf_counter()-t0:.0f}s  "
              f"classes={len(Xtr_by_class)}/{N_CLASSES}  "
              f"missing train/val={miss_tr}/{miss_va}", flush=True)
        if len(Xtr_by_class) < N_CLASSES:
            print(f"  WARNING: only {len(Xtr_by_class)} classes have train features")

        for variant in args.variants:
            runs = []
            for seed in args.seeds:
                sessions = make_sessions(seed, variant)
                r = run_one(sessions, Xtr_by_class, Xva, yva, dim,
                            args.few_shot_correction)
                runs.append(r)
                print(f"    {variant} seed={seed}  last={100*r['last']:.2f}  "
                      f"avg_inc_all={100*r['avg_inc_all']:.2f}  "
                      f"({r['fit_s']:.0f}s, {len(sessions)} sessions)", flush=True)

            m_last, s_last = agg(runs, "last")
            m_all, s_all = agg(runs, "avg_inc_all")
            m_excl, _ = agg(runs, "avg_inc_excl_last")
            m_inc, _ = agg(runs, "avg_inc_inc_only")
            print(f"  => {pooling} / {variant}: "
                  f"last {100*m_last:.2f}+/-{100*s_last:.2f}  "
                  f"avg_inc_all {100*m_all:.2f}+/-{100*s_all:.2f}  "
                  f"avg_inc_excl_last {100*m_excl:.2f}  "
                  f"avg_inc_inc_only {100*m_inc:.2f}\n", flush=True)

            results[f"{pooling}|{variant}"] = {
                "pooling": pooling, "variant": variant, "dim": int(dim),
                "last_mean": m_last, "last_std": s_last,
                "avg_inc_all_mean": m_all, "avg_inc_all_std": s_all,
                "avg_inc_excl_last_mean": m_excl,
                "avg_inc_inc_only_mean": m_inc,
                "per_seed": runs,
            }

        del Xtr, Xva, Xtr_by_class     # the 3584-dim copies are large

    # ── summary table: the shape the paper needs ──────────────────────────────
    print("=" * 78)
    print(f"{'pooling':18s} {'variant':8s} {'D':>5s} {'last':>14s} {'avg_inc_all':>14s} {'avg_inc(S-1)':>13s}")
    print("-" * 78)
    for variant in args.variants:
        for pooling in args.poolings:
            r = results.get(f"{pooling}|{variant}")
            if not r:
                continue
            print(f"{pooling:18s} {variant:8s} {r['dim']:5d} "
                  f"{100*r['last_mean']:8.2f}+/-{100*r['last_std']:4.2f} "
                  f"{100*r['avg_inc_all_mean']:8.2f}+/-{100*r['avg_inc_all_std']:4.2f} "
                  f"{100*r['avg_inc_excl_last_mean']:12.2f}")
        print()

    # gain over the mean-pool baseline -- the paper's actual claim
    for variant in args.variants:
        base = results.get(f"mean|{variant}")
        if not base:
            continue
        print(f"gain over mean-pool baseline ({variant}):")
        for pooling in args.poolings:
            if pooling == "mean":
                continue
            r = results.get(f"{pooling}|{variant}")
            if not r:
                continue
            print(f"  {pooling:18s} last {100*(r['last_mean']-base['last_mean']):+6.2f}pp   "
                  f"avg_inc_all {100*(r['avg_inc_all_mean']-base['avg_inc_all_mean']):+6.2f}pp")
        print()

    out = Path(args.out or "reports/ssv2_tcd_174_pooling_raw.json")
    save_results(out, {
        "protocol": {"n_classes": N_CLASSES, "n_base": N_BASE,
                     "n_incremental": N_INCREMENTAL, "variants": VARIANTS},
        "config": {"seeds": args.seeds, "poolings": args.poolings,
                   "variants": args.variants,
                   "few_shot_correction": args.few_shot_correction,
                   "base_split": "random per seed (TCD's 84-class list unpublished)"},
        "results": results,
    })
    print(f"raw -> {out}")


if __name__ == "__main__":
    main()
