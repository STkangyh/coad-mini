"""
Does a real live-camera query window match the curated benchmark window we
measured everything on?

Audit item (물음표 4), corrected after reading app.py rather than assuming.
The original question was "does few-shot enrollment work when learned online",
but /enroll takes uploaded video FILES and extracts them with
extract_frames_from_video(), which -- like every benchmark number in this repo
-- uniformly samples 16 frames across the WHOLE clip. So enrollment already
matches what we measured; enrollment_covariance_result.md's numbers are not an
approximation of the real path, they ARE the real path.

The live 200ms-interval, 16-frame ring buffer (static/index.html:
CAPTURE_INTERVAL=200, FRAME_BUFFER_SIZE=16) is used by /predict_rt for EVERY
prediction -- base classes and newly enrolled ones alike. That is the actual
mismatch: every accuracy number in this project (chunks4 full-48-way 0.236,
UCF101 88.74, the enrollment study's base/enrolled splits) was computed on
curated windows spanning the WHOLE clip, but the deployed system queries with
a ~3.2s slice of whatever the camera happened to be pointed at. Average UCF101
clip length is ~7.2s (measured, n=15), so a live window covers under half the
action on average and can start anywhere.

This script builds "live-style" query features by decoding raw video and
slicing a contiguous window at the same real-time cadence as the browser
(native frame stride computed per-video from its own fps, since UCF101 mixes
25fps and 29.97fps sources -- using one fixed stride would silently misalign
half the dataset), then compares accuracy against the existing curated
features for the SAME held-out videos. Training/enrollment stays exactly as
measured before -- only the query side changes, because only the query side
differs from what we benchmarked.

Caveats stated where they matter: only the classes trained-into the head are
evaluated old-style (offline curated fit + enrolled 5-shot curated windows,
matching the real /enroll and build_fecam_head.py paths exactly); the
diagnostic is the query window only. One random offset per video, not swept --
so within-video sensitivity to "when the user happened to press record" is not
characterized here.

Run: python3 dev/run_live_query_sim.py [--per-class 5] [--seeds 0 1 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_clip_features import BACKBONES, extract  # noqa: E402
from scripts.extract_ucf101_features import load_class_index, load_split  # noqa: E402
from src.models.fecam_head import POOLINGS, FeCAMHead  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

UCF_DIR = ROOT / "data/ucf101"
VIDEO_DIR = UCF_DIR / "UCF-101"
CURATED_DIR = ROOT / "data/features_ucf101_b32/test"
CACHE = ROOT / "data/features_ucf101_live_cache.npz"

# Must match static/index.html exactly -- these are not tunable parameters,
# they are the deployed browser's constants, copied here as documentation of
# what is being simulated.
CAPTURE_INTERVAL_MS = 200
N_FRAMES = 16
N_BASE = 51            # TCD split, matches run_enrollment_covariance.py


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
    """Offset + stride that a real 200ms/16-frame ring buffer would have
    produced, for a video of this length and frame rate."""
    stride = max(1, round(fps * CAPTURE_INTERVAL_MS / 1000.0))
    span = stride * (N_FRAMES - 1) + 1
    rng = np.random.RandomState(seed)
    if n_native_frames <= span:
        # Clip shorter than one live window: use every available frame at this
        # stride and pad by repeating the last one, same fallback
        # extract_frames_from_video uses for short uploads.
        idx = list(range(0, n_native_frames, stride))
        while len(idx) < N_FRAMES:
            idx.append(idx[-1] if idx else 0)
        return np.array(idx[:N_FRAMES])
    offset = int(rng.randint(0, n_native_frames - span + 1))
    return offset + np.arange(N_FRAMES) * stride


def decode_live_features(samples, model, processor, device, seed):
    """One live-style (16, 512) feature per sample, from raw video."""
    out = {}
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        path = VIDEO_DIR / s["rel"]
        container = av.open(str(path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames = [f.to_image() for f in container.decode(stream)]
        container.close()
        if not frames:
            continue
        idx = live_window_indices(len(frames), fps, seed=hash((s["id"], seed)) & 0xFFFF)
        idx = np.clip(idx, 0, len(frames) - 1)
        chosen = [frames[j] for j in idx]
        out[s["id"]] = extract(chosen, model, processor, device)
        if (i + 1) % 50 == 0:
            print(f"  decoded {i+1}/{len(samples)}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return out


def load_curated(sample_ids):
    out = {}
    for sid in sample_ids:
        p = CURATED_DIR / f"{sid}.npy"
        if p.exists():
            out[sid] = np.load(p).astype(np.float32)
    return out


def pool_all(feat_by_id, pooling):
    fn = POOLINGS[pooling]
    return {k: fn(v.astype(np.float64)) for k, v in feat_by_id.items()}


def load_ucf101_train():
    """Curated TRAIN features, same as run_enrollment_covariance.py / build_fecam_head.py."""
    d = ROOT / "data/features_ucf101_b32"
    class_index = load_class_index()
    samples = load_split("train", class_index)
    X, y = [], []
    for s in samples:
        p = d / "train" / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).astype(np.float32))
            y.append(s["class_id"])
    return np.stack(X), np.array(y)


def run_seed(seed, Xtr_raw, ytr, pooled_curated, pooled_live, ids_by_class, pooling, few_shot_correction):
    """One base/enrolled split: fit base offline (curated), enroll 10 classes
    5-shot (curated, matching the real /enroll path), then score BOTH curated
    and live query features for the same held-out videos."""
    order = np.random.RandomState(seed).permutation(101)
    base_ids, enroll_ids = order[:N_BASE], order[N_BASE:N_BASE + 10]

    pool_fn = POOLINGS[pooling]
    Xtr = np.stack([pool_fn(x.astype(np.float64)) for x in Xtr_raw])
    dim = Xtr.shape[1]

    head = FeCAMHead(feature_dim=dim, max_classes=101, pooling=pooling,
                     few_shot_correction=few_shot_correction)
    m = np.isin(ytr, base_ids)
    head.observe(Xtr[m], ytr[m])

    rng = np.random.RandomState(seed)
    for c in enroll_ids:
        idx = np.where(ytr == c)[0]
        shot = rng.choice(idx, min(5, len(idx)), replace=False)
        head.observe(Xtr[shot], ytr[shot], update_cov=False)

    seen = list(base_ids) + list(enroll_ids)
    results = {}
    for mode, pooled in (("curated", pooled_curated), ("live", pooled_live)):
        correct = {"base": [0, 0], "enrolled": [0, 0]}   # [hits, total]
        for cid in seen:
            group = "base" if cid in base_ids else "enrolled"
            for sid in ids_by_class.get(cid, []):
                if sid not in pooled:
                    continue
                x = pooled[sid][None, :]
                s = head.scores(x)[0]
                s[[c for c in range(101) if c not in seen]] = -1e18
                correct[group][1] += 1
                correct[group][0] += int(s.argmax() == cid)
        results[mode] = {
            g: (correct[g][0] / correct[g][1] if correct[g][1] else None)
            for g in ("base", "enrolled")
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5,
                    help="test videos per class to decode live-style")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pooling", nargs="+", default=["mean", "chunks4"])
    ap.add_argument("--few-shot-correction", action="store_true", default=True)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    class_index = load_class_index()
    test_samples = load_split("test", class_index)
    sub = pick_subsample(test_samples, args.per_class, seed=0)
    ids_by_class = {}
    for s in sub:
        ids_by_class.setdefault(s["class_id"], []).append(s["id"])
    print(f"subsample: {len(sub)} test videos across "
          f"{len(ids_by_class)} classes ({args.per_class}/class)")

    curated_raw = load_curated([s["id"] for s in sub])
    print(f"curated features loaded: {len(curated_raw)}/{len(sub)}")

    if CACHE.exists() and not args.rebuild_cache:
        cached = np.load(CACHE, allow_pickle=True)
        live_raw = {k: cached[k] for k in cached.files}
        print(f"live features from cache: {len(live_raw)} "
              f"(use --rebuild-cache to redo)")
    else:
        print("decoding raw video for live-style windows "
              f"(CAPTURE_INTERVAL={CAPTURE_INTERVAL_MS}ms, N_FRAMES={N_FRAMES})...")
        cfg = BACKBONES["clip_b32"]
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(cfg.model_name).eval()
        processor = AutoProcessor.from_pretrained(cfg.model_name)
        device = "cpu"
        live_raw = decode_live_features(sub, model, processor, device, seed=0)
        np.savez_compressed(CACHE, **live_raw)
        print(f"live features decoded: {len(live_raw)}, cached -> {CACHE}")

    common_ids = set(curated_raw) & set(live_raw)
    print(f"usable (both curated+live present): {len(common_ids)}\n")
    curated_raw = {k: v for k, v in curated_raw.items() if k in common_ids}
    live_raw = {k: v for k, v in live_raw.items() if k in common_ids}
    ids_by_class = {c: [i for i in ids if i in common_ids] for c, ids in ids_by_class.items()}

    Xtr_raw, ytr = load_ucf101_train()
    print(f"train (curated, for base fit + enrollment): {Xtr_raw.shape}\n")

    results = {}
    for pooling in args.pooling:
        pooled_curated = pool_all(curated_raw, pooling)
        pooled_live = pool_all(live_raw, pooling)
        print(f"=== pooling={pooling} ===")
        print(f"{'seed':>5s} {'query':8s} {'base':>8s} {'enrolled':>9s}")
        print("-" * 34)
        agg = {"curated": {"base": [], "enrolled": []}, "live": {"base": [], "enrolled": []}}
        for seed in args.seeds:
            r = run_seed(seed, Xtr_raw, ytr, pooled_curated, pooled_live,
                        ids_by_class, pooling, args.few_shot_correction)
            for mode in ("curated", "live"):
                for g in ("base", "enrolled"):
                    if r[mode][g] is not None:
                        agg[mode][g].append(r[mode][g])
                print(f"{seed:5d} {mode:8s} {100*r[mode]['base']:7.2f}% "
                      f"{100*r[mode]['enrolled']:8.2f}%")
        summary = {mode: {g: float(np.mean(v)) if v else None for g, v in gs.items()}
                  for mode, gs in agg.items()}
        print(f"\n  mean   curated  base {100*summary['curated']['base']:.2f}%  "
              f"enrolled {100*summary['curated']['enrolled']:.2f}%")
        print(f"  mean   live     base {100*summary['live']['base']:.2f}%  "
              f"enrolled {100*summary['live']['enrolled']:.2f}%")
        print(f"  Δ (live - curated)  base {100*(summary['live']['base']-summary['curated']['base']):+.2f}pp  "
              f"enrolled {100*(summary['live']['enrolled']-summary['curated']['enrolled']):+.2f}pp\n")
        results[pooling] = {"per_seed": agg, "mean": summary}

    save_results(ROOT / "reports/live_query_sim_raw.json", {
        "per_class": args.per_class, "seeds": args.seeds,
        "n_subsample": len(common_ids), "results": results,
    })
    print(f"raw -> reports/live_query_sim_raw.json")


if __name__ == "__main__":
    main()
