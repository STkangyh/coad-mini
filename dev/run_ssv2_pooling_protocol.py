"""
The closed-form pooling result (+8.97pp on SSv2), re-run under a protocol that
can actually support the claim.

Audit items 2/3/4 (reports/RESEARCH_LOG.md, 2026-07-30). The original run in
dev/run_ssv2_temporal_pooling.py has three problems, and all three inflate
confidence in the same direction:

  2. no seeds at all -- one fixed class order, so +8.97pp has no error bar
  3. the pooling variant AND the chunk count were both chosen on the same val
     set the result is reported on, so the winner is selection-biased
  4. it runs our own curated 8x6 curriculum, while being compared against
     ESSENTIAL's numbers from TCD's base-heavy protocol

What this script does instead:

  SELECTION happens entirely inside the training set. Train is split by video
  into fit/select portions; every pooling variant and chunk count is scored by
  running the full class-IL protocol fit->select. Val is never touched, so the
  chosen configuration cannot have peeked at the reported numbers.

  REPORTING then runs only that selection on val, over several random class
  orders, under two structures: the uniform 8x6 our earlier work used, and a
  TCD-style base-heavy 24+4x6.

HONEST LIMIT on item 4: TCD's actual SSv2 protocol is 84 base classes out of
174. We have features for 48 classes and no raw video for the rest (same blocker
as reports/mobileclip_result.md), so the literal protocol is out of reach. What
is reproduced here is its STRUCTURE (base-heavy sessions, small increments) and
its EVALUATION CONVENTION (random class orders, average incremental accuracy) --
not its class count. Numbers here are still not directly comparable to
ESSENTIAL's 48.9.

Run: python3 dev/run_ssv2_pooling_protocol.py [--seeds 0 1 2] [--select-frac 0.25]
"""
import argparse
import json
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
OUT = Path("reports/ssv2_pooling_protocol_raw.json")


# ── poolings (same definitions as the original study) ────────────────────────
def _chunks(f, k):
    idx = np.linspace(0, len(f), k + 1).astype(int)
    return np.concatenate([f[idx[i]:idx[i + 1]].mean(axis=0) for i in range(k)])


def p_mean(f):
    return f.mean(axis=0)


def p_mean_std(f):
    return np.concatenate([f.mean(axis=0), f.std(axis=0)])


def p_mean_diff(f):
    d = np.diff(f, axis=0)
    return np.concatenate([f.mean(axis=0), d.mean(axis=0)])


def p_mean_diff_absdiff(f):
    d = np.diff(f, axis=0)
    return np.concatenate([f.mean(axis=0), d.mean(axis=0), np.abs(d).mean(axis=0)])


def chunks_k(k):
    return lambda f: _chunks(f, k)


def chunks_adjdiff(k):
    """k segment means + differences between adjacent segments."""
    def pool(f):
        d = f.shape[1]
        c = _chunks(f, k)
        diffs = [c[(i + 1) * d:(i + 2) * d] - c[i * d:(i + 1) * d] for i in range(k - 1)]
        return np.concatenate([c] + diffs)
    return pool


# k is capped at 4: the original sweep (reports/ssv2_chunk_sweep_raw.json) already
# established that accuracy peaks at k=3-4 and declines after, and the cost is
# prohibitive anyway -- the covariance inverse is O(D^3), so chunks6+adjdiff
# (D=5632) costs ~13 s per session where chunks3+adjdiff (D=2560) costs ~1 s.
CANDIDATES = {
    "mean": p_mean,
    "mean+std": p_mean_std,
    "mean+diff": p_mean_diff,
    "mean+diff+|diff|": p_mean_diff_absdiff,
    **{f"chunks{k}": chunks_k(k) for k in (2, 3, 4)},
    **{f"chunks{k}+adjdiff": chunks_adjdiff(k) for k in (2, 3, 4)},
}
BASELINE = "mean"


# ── data: load raw windows once, pool in memory ──────────────────────────────
def load_raw(split):
    samples = load_samples(f"data/subset/{split}_mini.json")
    d = FEATURE_DIR / ("train" if split == "train" else "val")
    W, y, ids = [], [], []
    for s in samples:
        p = d / f"{s['id']}.npy"
        if p.exists():
            W.append(np.load(p).astype(np.float32))
            y.append(s["class_id"])
            ids.append(s["id"])
    return np.stack(W), np.array(y), np.array(ids)


def pooled(W, fn):
    return np.stack([fn(w.astype(np.float64)) for w in W])


# ── protocols ────────────────────────────────────────────────────────────────
def uniform_sessions(order):
    return [list(order[i:i + 6]) for i in range(0, N_CLASSES, 6)]


def base_heavy_sessions(order):
    """TCD-style: a large base session followed by small increments."""
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
    """Same pooled features, several random class orders."""
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
    Wtr, ytr, _ = load_raw("train")
    Wva, yva, _ = load_raw("val")
    print(f"train {Wtr.shape}, val {Wva.shape}  ({time.perf_counter()-t0:.0f}s to load)\n")

    # ── phase 1: selection, inside train only ────────────────────────────────
    rng = np.random.RandomState(12345)
    sel_mask = np.zeros(len(ytr), dtype=bool)
    for c in range(N_CLASSES):                    # stratified, so every class is present
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

    # ── phase 2: report on val, both structures ──────────────────────────────
    print(f"=== phase 2: val, {len(args.seeds)} random class orders ===")
    print(f"{'protocol':20s} {'pooling':22s} {'avg inc':>16s} {'last':>16s}")
    print("-" * 78)
    report = {}
    for pname, pfn in PROTOCOLS.items():
        for name in dict.fromkeys([BASELINE, chosen]):
            fn = CANDIDATES[name]
            Ptr, Pva = pooled(Wtr, fn), pooled(Wva, fn)
            r = sweep(Ptr, ytr, Pva, yva, Ptr.shape[1], args.seeds, pfn)
            report[f"{pname}/{name}"] = {k: v for k, v in r.items() if k != "runs"}
            print(f"{pname:20s} {name:22s} "
                  f"{100*r['avg_inc_mean']:8.2f} ±{100*r['avg_inc_sd']:4.2f} "
                  f"{100*r['last_mean']:8.2f} ±{100*r['last_sd']:4.2f}", flush=True)
        if chosen == BASELINE:
            continue
        b, c = report[f"{pname}/{BASELINE}"], report[f"{pname}/{chosen}"]
        d_avg = 100 * (c["avg_inc_mean"] - b["avg_inc_mean"])
        d_last = 100 * (c["last_mean"] - b["last_mean"])
        # Effect size is computed on avg_inc, not last: the final head has seen
        # every class either way and its statistics are order-independent sums, so
        # `last` is the SAME number for every seed and every protocol (sd = 0 by
        # construction, not by luck). Class order only moves the intermediate
        # evaluation points, which is exactly what avg_inc averages over.
        sd = 100 * np.sqrt((b["avg_inc_sd"] ** 2 + c["avg_inc_sd"] ** 2) / 2)
        effect = f"{d_avg / sd:.1f}x pooled sd (avg inc)" if sd > 0 else "single seed"
        print(f"{'  -> delta':20s} {'':22s} {d_avg:+8.2f}{'':7s} {d_last:+8.2f}"
              f"   ({effect})")
        report[f"{pname}/delta"] = {
            "avg_inc_pp": d_avg, "last_pp": d_last,
            "avg_inc_effect_size_sd": d_avg / sd if sd > 0 else None,
            "note": "last has zero seed variance: order-independent final model",
        }

    save_results(OUT, {
        "seeds": args.seeds, "select_frac": args.select_frac,
        "selected_on_train": chosen, "selection": selection, "report": report,
    })
    print(f"\nraw -> {OUT}   ({time.perf_counter()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
