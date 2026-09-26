"""
Does WIDENING each temporal segment so neighbors overlap beat the existing
non-overlapping chunks3/chunks4 pooling -- at the SAME output dimension?

paper_outline_draft.md flagged overlap as a claim in the Notion outline
("overlap이 중요", ablation "overlap 유무") that the codebase has never actually
tested: src/models/fecam_head.py's _segment_means() always cuts k CONTIGUOUS,
NON-overlapping slices. This script adds the missing variant and compares it
under the exact same protocol run_ssv2_pooling_protocol.py already established
and that ssv2_temporal_pooling_result.md documents as audited:

  - selection happens entirely inside TRAIN (class-stratified fit/select split,
    val never touched) -- avoids the 1.14pp selection-bias artifact that
    ssv2_temporal_pooling_result.md §10(a) found and corrected for
  - reporting then runs only the selected config on val, over several random
    class orders, under both uniform-8x6 and TCD-style base-heavy-24+4x6

DIMENSION-MATCHED BY DESIGN: overlap widens each segment's window but keeps the
same k segments and therefore the same output dimension k*D as the existing
chunks_k baseline. ssv2_temporal_pooling_result.md §4(b) already showed that
dimension increase alone (mean+std, same D as mean+diff) buys almost nothing
(+1.04pp) while order-preservation buys most of the effect -- so if overlap
helps here, it cannot be that confound; it has to be the widening itself
(smoothing hard segment-boundary cuts) or it is genuinely not distinguishable
from noise, matching this project's other pooling-variant findings.

Overlap parameterization: for k segments of native length seg_len = T/k, each
segment is symmetrically padded by `overlap * seg_len / 2` frames on each side
before being clipped to [0, T]. overlap=0.0 reduces to (near-)the same cuts as
the existing chunks_k (small rounding differences only -- linspace-round vs
linspace-truncate); it is included in the sweep as the same-code-path anchor,
not read from the pre-existing chunks_k implementation, so the overlap delta is
never confounded by an implementation difference between two different pooling
functions.

Run: python3 dev/run_overlap_pooling.py [--seeds 0 1 2 3 4] [--select-frac 0.25]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.models.fecam_head import FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEATURE_DIR = Path("data/features")
N_CLASSES = 48
OUT = Path("reports/overlap_pooling_raw.json")


# ── poolings ───────────────────────────────────────────────────────────────
def _chunks(f, k):
    """Existing non-overlapping baseline (identical to fecam_head._segment_means)."""
    idx = np.linspace(0, len(f), k + 1).astype(int)
    return np.concatenate([f[idx[i]:idx[i + 1]].mean(axis=0) for i in range(k)])


def _chunks_overlap(f, k, overlap):
    """k segments, evenly spaced like _chunks, but each window widened by
    `overlap` (fraction of the native segment length, added to EACH side) so
    neighbors share frames. Same output dim as _chunks(f, k): k*D.
    overlap=0.0 -> same cut points as _chunks up to linspace round-vs-truncate.
    """
    T = len(f)
    idx = np.linspace(0, T, k + 1)
    seg_len = T / k
    pad = seg_len * overlap / 2.0
    segs = []
    for i in range(k):
        lo = max(0, int(round(idx[i] - pad)))
        hi = min(T, int(round(idx[i + 1] + pad)))
        if hi <= lo:
            hi = lo + 1
        segs.append(f[lo:hi].mean(axis=0))
    return np.concatenate(segs)


def chunks_k(k):
    return lambda f: _chunks(f, k)


def chunks_overlap(k, overlap):
    return lambda f: _chunks_overlap(f, k, overlap)


# k in {3,4}: the two chunk counts ssv2_temporal_pooling_result.md §5 found near
# the accuracy peak (the "sweet spot" the overlap question is actually about --
# does widening let a smaller/larger k compete better once boundary cuts are
# softened?). overlap in {0.0, 0.25, 0.5, 1.0, 2.0}: 0.0 is the same-code-path
# zero-overlap anchor; 2.0 widens each segment to 3x its native length (T=16 is
# short, so this is already close to "everything overlaps with everything").
OVERLAPS = [0.0, 0.25, 0.5, 1.0, 2.0]
CANDIDATES = {
    "mean": lambda f: f.mean(axis=0),
    "chunks3": chunks_k(3),          # existing baseline, unmodified code path
    "chunks4": chunks_k(4),          # existing baseline, unmodified code path
    **{f"chunks{k}_ov{ov}": chunks_overlap(k, ov)
       for k in (3, 4) for ov in OVERLAPS},
}
BASELINE = "mean"


# ── data: load raw windows once, pool in memory (identical to
#    run_ssv2_pooling_protocol.py) ───────────────────────────────────────────
def load_raw(split):
    samples = load_samples(f"data/subset/{split}_mini.json")
    d = FEATURE_DIR / ("train" if split == "train" else "val")
    W, y = [], []
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            W.append(np.load(p).astype(np.float32))
            y.append(s["class_id"])
    return np.stack(W), np.array(y)


def pooled(W, fn):
    return np.stack([fn(w.astype(np.float64)) for w in W])


# ── protocols (identical to run_ssv2_pooling_protocol.py) ───────────────────
def uniform_sessions(order):
    return [list(order[i:i + 6]) for i in range(0, N_CLASSES, 6)]


def base_heavy_sessions(order):
    return [list(order[:24])] + [list(order[i:i + 6]) for i in range(24, N_CLASSES, 6)]


PROTOCOLS = {"uniform-8x6": uniform_sessions, "base-heavy-24+4x6": base_heavy_sessions}


def run_cil(Xtr, ytr, Xte, yte, sessions, dim):
    head = FeCAMHead(feature_dim=dim, max_classes=N_CLASSES)
    seen, accs = [], []
    for cids in sessions:
        m = np.isin(ytr, cids)
        head.observe(Xtr[m], ytr[m])
        seen.extend(cids)
        te = np.isin(yte, seen)
        s = head.scores(Xte[te])
        s[:, [c for c in range(N_CLASSES) if c not in seen]] = -1e18
        accs.append(float((s.argmax(axis=1) == yte[te]).mean()))
    return {"per_step": accs, "avg_inc": float(np.mean(accs)), "last": accs[-1]}


def sweep(Xtr, ytr, Xte, yte, dim, seeds, protocol_fn):
    runs = [run_cil(Xtr, ytr, Xte, yte, protocol_fn(
        np.random.RandomState(s).permutation(N_CLASSES)), dim) for s in seeds]
    return {
        "avg_inc_mean": float(np.mean([r["avg_inc"] for r in runs])),
        "avg_inc_sd": float(np.std([r["avg_inc"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
        "last_mean": float(np.mean([r["last"] for r in runs])),
        "last_sd": float(np.std([r["last"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
        "runs": runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--select-frac", type=float, default=0.25,
                    help="fraction of TRAIN videos held out to choose the pooling")
    args = ap.parse_args()

    t0 = time.perf_counter()
    Wtr, ytr = load_raw("train")
    Wva, yva = load_raw("val")
    print(f"train {Wtr.shape}, val {Wva.shape}  ({time.perf_counter()-t0:.0f}s to load)\n")

    # ── phase 1: selection, inside train only (val never touched) ───────────
    rng = np.random.RandomState(12345)
    sel_mask = np.zeros(len(ytr), dtype=bool)
    for c in range(N_CLASSES):
        idx = np.where(ytr == c)[0]
        sel_mask[rng.choice(idx, max(1, int(round(args.select_frac * len(idx)))),
                            replace=False)] = True
    print(f"=== phase 1: selection on held-out TRAIN "
          f"(fit {(~sel_mask).sum()}, select {sel_mask.sum()}) — val untouched ===")
    print(f"{'pooling':22s} {'dim':>6s} {'avg inc':>16s} {'last':>16s}")
    print("-" * 64)

    selection = {}
    for name, fn in CANDIDATES.items():
        P = pooled(Wtr, fn)
        r = sweep(P[~sel_mask], ytr[~sel_mask], P[sel_mask], ytr[sel_mask],
                  P.shape[1], args.seeds, uniform_sessions)
        selection[name] = {**{k: v for k, v in r.items() if k != "runs"}, "dim": int(P.shape[1])}
        print(f"{name:22s} {P.shape[1]:6d} "
              f"{100*r['avg_inc_mean']:8.2f} ±{100*r['avg_inc_sd']:4.2f} "
              f"{100*r['last_mean']:8.2f} ±{100*r['last_sd']:4.2f}", flush=True)

    chosen = max(selection, key=lambda k: selection[k]["avg_inc_mean"])
    print(f"\n>>> selected on train alone: {chosen!r} "
          f"(dim {selection[chosen]['dim']})\n")

    # also report, for the paper's actual question, the best OVERLAPPING
    # variant specifically vs its own zero-overlap anchor at the same k -- the
    # apples-to-apples comparison, independent of which config wins overall.
    best_ov = max((n for n in selection if "_ov" in n and not n.endswith("_ov0.0")),
                  key=lambda k: selection[k]["avg_inc_mean"])
    k_of_best = best_ov.split("_ov")[0]
    anchor = f"{k_of_best}_ov0.0"
    print(f">>> best overlapping variant: {best_ov!r} ({100*selection[best_ov]['avg_inc_mean']:.2f}) "
          f"vs its own zero-overlap anchor {anchor!r} "
          f"({100*selection[anchor]['avg_inc_mean']:.2f}) "
          f"vs original {k_of_best!r} impl ({100*selection[k_of_best]['avg_inc_mean']:.2f})\n")

    # ── phase 2: report on val, both structures ──────────────────────────────
    to_report = dict.fromkeys([BASELINE, chosen, best_ov, anchor, k_of_best])
    print(f"=== phase 2: val, {len(args.seeds)} random class orders ===")
    print(f"{'protocol':20s} {'pooling':22s} {'avg inc':>16s} {'last':>16s}")
    print("-" * 78)
    report = {}
    for pname, pfn in PROTOCOLS.items():
        for name in to_report:
            fn = CANDIDATES[name]
            Ptr, Pva = pooled(Wtr, fn), pooled(Wva, fn)
            r = sweep(Ptr, ytr, Pva, yva, Ptr.shape[1], args.seeds, pfn)
            report[f"{pname}/{name}"] = {k: v for k, v in r.items() if k != "runs"}
            print(f"{pname:20s} {name:22s} "
                  f"{100*r['avg_inc_mean']:8.2f} ±{100*r['avg_inc_sd']:4.2f} "
                  f"{100*r['last_mean']:8.2f} ±{100*r['last_sd']:4.2f}", flush=True)

        def delta(a, b, label):
            ra, rb = report[f"{pname}/{a}"], report[f"{pname}/{b}"]
            d_avg = 100 * (rb["avg_inc_mean"] - ra["avg_inc_mean"])
            d_last = 100 * (rb["last_mean"] - ra["last_mean"])
            sd = 100 * np.sqrt((ra["avg_inc_sd"] ** 2 + rb["avg_inc_sd"] ** 2) / 2)
            effect = d_avg / sd if sd > 0 else None
            print(f"  -> {label:28s} {d_avg:+8.2f}pp avg_inc  {d_last:+8.2f}pp last  "
                  f"({f'{effect:.1f}x pooled sd' if effect is not None else 'single seed'})")
            return {"avg_inc_pp": d_avg, "last_pp": d_last, "avg_inc_effect_size_sd": effect}

        report[f"{pname}/delta_vs_mean"] = delta(BASELINE, chosen, f"{chosen} vs mean")
        if best_ov != chosen:
            report[f"{pname}/delta_overlap_vs_mean"] = delta(BASELINE, best_ov, f"{best_ov} vs mean")
        report[f"{pname}/delta_overlap_vs_anchor"] = delta(anchor, best_ov, f"{best_ov} vs {anchor} (overlap-only)")
        report[f"{pname}/delta_overlap_vs_original_impl"] = delta(k_of_best, best_ov, f"{best_ov} vs original {k_of_best}")

    save_results(OUT, {
        "seeds": args.seeds, "select_frac": args.select_frac,
        "selected_on_train": chosen, "best_overlap_variant": best_ov,
        "zero_overlap_anchor": anchor, "selection": selection, "report": report,
    })
    print(f"\nraw -> {OUT}   ({time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
