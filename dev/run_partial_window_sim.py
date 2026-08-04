"""
Extends live_query_sim to SSv2, and validates the method it has to use to get there.

live_query_sim_result.md measured the real thing for UCF101: re-decode raw video
and slice a window at the browser's actual 200ms/16-frame cadence. That needs raw
video, and SSv2's is not available in this environment (COAD_VIDEO_DIR unset, no
common location has it -- the same blocker as mobileclip_result.md and
ssv2_pooling_protocol.py's HONEST LIMIT section). SSv2 features are (16, 512)
arrays with no frame-count or fps metadata retained, so the exact reconstruction
is not possible here.

What IS possible: the 16 stored frames already span the whole clip evenly, so a
CONTIGUOUS SUB-SPAN of those 16 approximates "only saw part of the action" --
coarser than real-time redecoding (16 samples instead of the dozens of native
frames a live window actually spans) but pointed the same direction. This script
sweeps coverage fraction on both datasets so UCF101 becomes the validity check:
if the proxy's drop at the coverage fraction matching the real UCF101 measurement
(~44%, the average live-window/clip-length ratio) lands near live_query_sim's
directly-measured -1.5 to -2.1pp, the proxy is trustworthy enough to read the
SSv2 numbers from. If it does not land close, the SSv2 numbers here should be
treated as a rough direction, not a replacement for a real measurement once raw
video access is restored.

For SSv2 there is no base/enrolled split in the deployed config (all 48 classes
are fit in one curriculum, build_fecam_head.py-style) -- so this reports the
served config's full-48-way accuracy directly, matching how it is actually
shipped, few_shot_correction=True included.

Run: python3 dev/run_partial_window_sim.py [--bench ssv2 ucf101]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fecam_head import POOLINGS, FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402
from dev.run_live_query_sim import (  # noqa: E402
    N_BASE, load_curated, load_ucf101_train, pick_subsample,
)
from scripts.extract_ucf101_features import load_class_index, load_split  # noqa: E402

COVERAGE = [1.0, 0.75, 0.5, 0.44, 0.25]   # 0.44 = UCF101's measured live-window/clip ratio
SEEDS = [0, 1, 2, 3, 4]                   # offsets, not decode-limited so we can afford more


def partial_window(w, coverage, rng):
    """Contiguous sub-span of a (T, D) window -- the 16-frame proxy for
    'the camera only caught part of the action'. A random start within
    what's left, same spirit as live_window_indices in run_live_query_sim.py
    but operating on the already-16-sampled sequence instead of native frames."""
    t = len(w)
    k = max(2, round(t * coverage))
    if k >= t:
        return w
    start = int(rng.randint(0, t - k + 1))
    return w[start:start + k]


# ── SSv2: served config, full curriculum, no base/enroll split ───────────────
N_SSV2_CLASSES = 48
CPS = 6
SSV2_STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_SSV2_CLASSES // CPS + 1)}


def load_ssv2_raw(split):
    d = ROOT / "data/features" / ("train" if split == "train" else "val")
    X, y = [], []
    for s in load_samples(f"data/subset/{split}_mini.json"):
        p = d / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).astype(np.float32))
            y.append(s["class_id"])
    return np.stack(X), np.array(y)


def eval_ssv2(pooling, coverage, seed):
    Xtr_raw, ytr = load_ssv2_raw("train")
    Xva_raw, yva = load_ssv2_raw("val")
    pool_fn = POOLINGS[pooling]
    dim = pool_fn(Xtr_raw[0].astype(np.float64)).shape[0]

    head = FeCAMHead(feature_dim=dim, max_classes=N_SSV2_CLASSES, pooling=pooling,
                     few_shot_correction=True)   # matches the served config exactly
    for cids in SSV2_STAGES.values():
        m = np.isin(ytr, cids)
        Xp = np.stack([pool_fn(x.astype(np.float64)) for x in Xtr_raw[m]])
        head.observe(Xp, ytr[m])

    rng = np.random.RandomState(seed)
    Xq = np.stack([pool_fn(partial_window(x.astype(np.float64), coverage, rng))
                   for x in Xva_raw])
    return float((head.scores(Xq).argmax(axis=1) == yva).mean())


# ── UCF101: same base(51)+enrolled(10) structure as live_query_sim, for the
#    validity check against its real decoded numbers ─────────────────────────
def eval_ucf101(pooling, coverage, seed, curated_raw, ids_by_class, Xtr_raw, ytr):
    pool_fn = POOLINGS[pooling]
    Xtr = np.stack([pool_fn(x.astype(np.float64)) for x in Xtr_raw])
    dim = Xtr.shape[1]

    order = np.random.RandomState(seed).permutation(101)
    base_ids, enroll_ids = order[:N_BASE], order[N_BASE:N_BASE + 10]

    head = FeCAMHead(feature_dim=dim, max_classes=101, pooling=pooling,
                     few_shot_correction=True)
    m = np.isin(ytr, base_ids)
    head.observe(Xtr[m], ytr[m])
    rng = np.random.RandomState(seed)
    for c in enroll_ids:
        idx = np.where(ytr == c)[0]
        shot = rng.choice(idx, min(5, len(idx)), replace=False)
        head.observe(Xtr[shot], ytr[shot], update_cov=False)

    seen = list(base_ids) + list(enroll_ids)
    q_rng = np.random.RandomState(seed + 1000)
    correct = {"base": [0, 0], "enrolled": [0, 0]}
    for cid in seen:
        group = "base" if cid in base_ids else "enrolled"
        for sid in ids_by_class.get(cid, []):
            if sid not in curated_raw:
                continue
            w = partial_window(curated_raw[sid].astype(np.float64), coverage, q_rng)
            x = pool_fn(w)[None, :]
            s = head.scores(x)[0]
            s[[c for c in range(101) if c not in seen]] = -1e18
            correct[group][1] += 1
            correct[group][0] += int(s.argmax() == cid)
    return {g: (correct[g][0] / correct[g][1] if correct[g][1] else None) for g in correct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["ucf101", "ssv2"], choices=["ucf101", "ssv2"])
    ap.add_argument("--pooling", nargs="+", default=["mean", "chunks4"])
    ap.add_argument("--per-class", type=int, default=5, help="UCF101 query subsample size")
    args = ap.parse_args()

    results = {}

    if "ucf101" in args.bench:
        print("=== UCF101 (validity check vs live_query_sim_result.md) ===")
        class_index = load_class_index()
        test_samples = load_split("test", class_index)
        sub = pick_subsample(test_samples, args.per_class, seed=0)
        ids_by_class = {}
        for s in sub:
            ids_by_class.setdefault(s["class_id"], []).append(s["id"])
        curated_raw = load_curated([s["id"] for s in sub])
        Xtr_raw, ytr = load_ucf101_train()
        print(f"subsample: {len(curated_raw)} videos, {len(ids_by_class)} classes\n")

        for pooling in args.pooling:
            print(f"--- pooling={pooling} ---")
            print(f"{'coverage':>9s} {'base':>8s} {'enrolled':>9s}")
            row = {}
            for cov in COVERAGE:
                runs = [eval_ucf101(pooling, cov, s, curated_raw, ids_by_class, Xtr_raw, ytr)
                       for s in SEEDS]
                agg = {g: float(np.mean([r[g] for r in runs if r[g] is not None]))
                      for g in ("base", "enrolled")}
                row[cov] = agg
                print(f"{cov:9.2f} {100*agg['base']:7.2f}% {100*agg['enrolled']:8.2f}%", flush=True)
            results[f"ucf101/{pooling}"] = row
            real = -2.09 if pooling == "mean" else -1.57
            proxy = 100 * (row[0.44]["base"] - row[1.0]["base"])
            print(f"  proxy Δ at coverage=0.44: {proxy:+.2f}pp  "
                  f"(live_query_sim real measurement: {real:+.2f}pp)\n")

    if "ssv2" in args.bench:
        print("=== SSv2 (served config, no raw video -- proxy only) ===")
        for pooling in args.pooling:
            print(f"--- pooling={pooling} ---")
            print(f"{'coverage':>9s} {'full-48-way':>12s}")
            row = {}
            for cov in COVERAGE:
                accs = [eval_ssv2(pooling, cov, s) for s in SEEDS]
                row[cov] = float(np.mean(accs))
                print(f"{cov:9.2f} {100*row[cov]:11.2f}%", flush=True)
            results[f"ssv2/{pooling}"] = row
            print(f"  Δ at coverage=0.44: {100*(row[0.44]-row[1.0]):+.2f}pp\n")

    save_results(ROOT / "reports/partial_window_sim_raw.json", {
        "coverage": COVERAGE, "seeds": SEEDS, "results": results,
    })
    print("raw -> reports/partial_window_sim_raw.json")


if __name__ == "__main__":
    main()
