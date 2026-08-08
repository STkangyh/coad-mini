"""
Real-decode version of partial_window_sim's SSv2 section -- now that raw SSv2
video is available (COAD_VIDEO_DIR), this replaces the "slice the stored
16-frame proxy" approximation with an actual re-decode + live-window slice,
exactly mirroring what run_live_query_sim.py already did for UCF101.

partial_window_sim_result.md flagged two honest limits on its SSv2 numbers:
1. the proxy underestimates the real drop by 15-35% (measured on UCF101, the
   only dataset where both methods were available)
2. coverage=0.44 (UCF101's measured live-window/clip ratio) was used as a
   stand-in for SSv2 because SSv2's real clip-length distribution was unknown
   without raw video

Both are resolved here: real decode gives the true drop directly, and (2) is
answered by computing each video's actual live-window/clip coverage ratio
from its real frame count and fps (SSv2 is 12fps per the official download
instructions, well under UCF101's 25/29.97fps mix, so the same 200ms/16-frame
window spans very differently).

Unlike UCF101 (base 51 + enrolled 10 split), the served SSv2 config fits all
48 classes in one curriculum (build_fecam_head.py-style, no enrollment split)
-- matching run_partial_window_sim.py's eval_ssv2 structure. So this compares
curated-vs-live query accuracy on the full 48-way task, not per base/enrolled
group.

Run: python3 dev/run_ssv2_live_query_sim.py [--per-class 10] [--seeds 0 1 2]
"""
import argparse
import sys
import time
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import VIDEO_DIR  # noqa: E402
from scripts.extract_clip_features import BACKBONES, extract  # noqa: E402
from src.models.fecam_head import POOLINGS, FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

FEATURES_DIR = ROOT / "data/features"
CACHE = ROOT / "data/features_ssv2_live_cache.npz"

# Must match static/index.html exactly (same constants run_live_query_sim.py uses).
CAPTURE_INTERVAL_MS = 200
N_FRAMES = 16

N_CLASSES = 48
CPS = 6
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_CLASSES // CPS + 1)}


def pick_subsample(samples, per_class, seed=0):
    rng = np.random.RandomState(seed)
    by_class = {}
    for s in samples:
        by_class.setdefault(s["class_id"], []).append(s)
    out = []
    for cid, group in by_class.items():
        idx = rng.choice(len(group), min(per_class, len(group)), replace=False)
        out.extend(group[i] for i in idx)
    return out


def live_window_indices(n_native_frames, fps, seed):
    """Same construction as run_live_query_sim.py -- offset+stride a real
    200ms/16-frame ring buffer would produce for this video's own fps."""
    stride = max(1, round(fps * CAPTURE_INTERVAL_MS / 1000.0))
    span = stride * (N_FRAMES - 1) + 1
    rng = np.random.RandomState(seed)
    if n_native_frames <= span:
        idx = list(range(0, n_native_frames, stride))
        while len(idx) < N_FRAMES:
            idx.append(idx[-1] if idx else 0)
        return np.array(idx[:N_FRAMES]), min(1.0, n_native_frames / span)
    offset = int(rng.randint(0, n_native_frames - span + 1))
    return offset + np.arange(N_FRAMES) * stride, span / n_native_frames


def decode_live_features(samples, model, processor, device, seed):
    """One live-style (16, 512) feature per sample, from raw video. Also
    returns each video's live-window/clip coverage ratio."""
    out, coverage = {}, {}
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        path = VIDEO_DIR / f"{s['id']}.webm"
        if not path.exists():
            continue
        container = av.open(str(path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [f.to_image() for f in container.decode(stream)]
        container.close()
        if not frames:
            continue
        idx, cov = live_window_indices(len(frames), fps, seed=hash((s["id"], seed)) & 0xFFFF)
        idx = np.clip(idx, 0, len(frames) - 1)
        chosen = [frames[j] for j in idx]
        out[s["id"]] = extract(chosen, model, processor, device)
        coverage[s["id"]] = cov
        if (i + 1) % 50 == 0:
            print(f"  decoded {i+1}/{len(samples)}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return out, coverage


def load_curated(sample_ids):
    out = {}
    for sid in sample_ids:
        p = FEATURES_DIR / "val" / f"{sid}.npy"
        if p.exists():
            out[sid] = np.load(p).astype(np.float32)
    return out


def load_ssv2_train():
    X, y = [], []
    for s in load_samples("data/subset/train_mini.json"):
        p = FEATURES_DIR / "train" / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).astype(np.float32))
            y.append(s["class_id"])
    return np.stack(X), np.array(y)


def pool_all(feat_by_id, pooling):
    fn = POOLINGS[pooling]
    return {k: fn(v.astype(np.float64)) for k, v in feat_by_id.items()}


def run_seed(seed, Xtr, ytr, pooled_curated, pooled_live, ids_by_class, pooling):
    pool_fn = POOLINGS[pooling]
    Xp = np.stack([pool_fn(x.astype(np.float64)) for x in Xtr])
    dim = Xp.shape[1]

    head = FeCAMHead(feature_dim=dim, max_classes=N_CLASSES, pooling=pooling,
                     few_shot_correction=True)  # matches served config exactly
    for cids in STAGES.values():
        m = np.isin(ytr, cids)
        head.observe(Xp[m], ytr[m])

    results = {}
    for mode, pooled in (("curated", pooled_curated), ("live", pooled_live)):
        hits, total = 0, 0
        for cid, ids in ids_by_class.items():
            for sid in ids:
                if sid not in pooled:
                    continue
                x = pooled[sid][None, :]
                pred = head.scores(x)[0].argmax()
                total += 1
                hits += int(pred == cid)
        results[mode] = hits / total if total else None
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10,
                    help="val videos per class to decode live-style")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pooling", nargs="+", default=["mean", "chunks4"])
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    if VIDEO_DIR is None:
        raise RuntimeError("COAD_VIDEO_DIR is not set.")

    val_samples = load_samples("data/subset/val_mini.json")
    sub = pick_subsample(val_samples, args.per_class, seed=0)
    ids_by_class = {}
    for s in sub:
        ids_by_class.setdefault(s["class_id"], []).append(s["id"])
    print(f"subsample: {len(sub)} val videos across "
          f"{len(ids_by_class)} classes ({args.per_class}/class)")

    curated_raw = load_curated([s["id"] for s in sub])
    print(f"curated features loaded: {len(curated_raw)}/{len(sub)}")

    if CACHE.exists() and not args.rebuild_cache:
        cached = np.load(CACHE, allow_pickle=True)
        live_raw = {k: cached[k] for k in cached.files if not k.startswith("__cov__")}
        coverage = {k[len("__cov__"):]: float(cached[k]) for k in cached.files if k.startswith("__cov__")}
        print(f"live features from cache: {len(live_raw)} "
              f"(use --rebuild-cache to redo)")
    else:
        print("decoding raw video for live-style windows "
              f"(CAPTURE_INTERVAL={CAPTURE_INTERVAL_MS}ms, N_FRAMES={N_FRAMES}, SSv2=12fps)...")
        cfg = BACKBONES["clip_b32"]
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(cfg.model_name).eval()
        processor = AutoProcessor.from_pretrained(cfg.model_name)
        device = "cpu"
        live_raw, coverage = decode_live_features(sub, model, processor, device, seed=0)
        save_blob = dict(live_raw)
        save_blob.update({f"__cov__{k}": np.float64(v) for k, v in coverage.items()})
        np.savez_compressed(CACHE, **save_blob)
        print(f"live features decoded: {len(live_raw)}, cached -> {CACHE}")

    common_ids = set(curated_raw) & set(live_raw)
    print(f"usable (both curated+live present): {len(common_ids)}")
    mean_cov = float(np.mean([coverage[k] for k in common_ids if k in coverage]))
    print(f"mean live-window/clip coverage ratio: {mean_cov:.3f}  "
          f"(vs UCF101's measured 0.44 -- SSv2 is 12fps and clips are shorter)\n")
    curated_raw = {k: v for k, v in curated_raw.items() if k in common_ids}
    live_raw = {k: v for k, v in live_raw.items() if k in common_ids}
    ids_by_class = {c: [i for i in ids if i in common_ids] for c, ids in ids_by_class.items()}

    Xtr, ytr = load_ssv2_train()
    print(f"train (curated, full 48-class curriculum): {Xtr.shape}\n")

    results = {}
    for pooling in args.pooling:
        pooled_curated = pool_all(curated_raw, pooling)
        pooled_live = pool_all(live_raw, pooling)
        print(f"=== pooling={pooling} ===")
        print(f"{'seed':>5s} {'curated':>9s} {'live':>9s} {'delta':>8s}")
        agg = {"curated": [], "live": []}
        for seed in args.seeds:
            r = run_seed(seed, Xtr, ytr, pooled_curated, pooled_live, ids_by_class, pooling)
            agg["curated"].append(r["curated"])
            agg["live"].append(r["live"])
            print(f"{seed:5d} {100*r['curated']:8.2f}% {100*r['live']:8.2f}% "
                  f"{100*(r['live']-r['curated']):+7.2f}pp")
        summary = {k: float(np.mean(v)) for k, v in agg.items()}
        print(f"\n  mean  curated {100*summary['curated']:.2f}%  live {100*summary['live']:.2f}%  "
              f"Δ {100*(summary['live']-summary['curated']):+.2f}pp\n")
        results[pooling] = {"per_seed": agg, "mean": summary}

    save_results(ROOT / "reports/ssv2_live_query_sim_raw.json", {
        "per_class": args.per_class, "seeds": args.seeds,
        "n_subsample": len(common_ids), "mean_coverage": mean_cov, "results": results,
    })
    print("raw -> reports/ssv2_live_query_sim_raw.json")


if __name__ == "__main__":
    main()
