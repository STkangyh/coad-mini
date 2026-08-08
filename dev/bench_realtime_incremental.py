"""
Can FeCAM do incremental training in real time (10 fps), and what is the bottleneck?

Setup mirrors the demo's streaming path (app.py::frames_to_feature -> fecam_predict):
a camera delivers frames one at a time; we keep a sliding 16-frame ring buffer,
encode each ARRIVING frame (batch=1 -- streaming cannot batch the future), mean-pool
the buffer into one 512-d window embedding, optionally LEARN from it (FeCAM.observe),
then PREDICT.

Budget: 10 fps => 100 ms per frame for the whole chain.

Two modes are timed so the marginal cost of learning is isolated:
  predict-only   encode -> pool -> scores
  learn+predict  encode -> pool -> observe -> scores      <- the goal's question

The learn+predict mode has a non-obvious cost: observe() invalidates the cached
DxD correlation inverse, so the next scores() must rebuild it. In a steady
train-then-predict loop that inverse is paid EVERY frame, which never happens in
the batch experiments where we fit once and score many times. This script
measures that term explicitly rather than assuming it is negligible -- it turned
out to dominate the learning path, which is what motivated splitting the head's
cache in two (fecam_head.py::_precision).

Two learning modes are separated because only one of them is now cheap:
  enroll (update_cov=False)  the demo's path -- means move, covariance does not,
                             so the inverse survives and only one row is patched
  full   (update_cov=True)   also folds the window into the shared covariance,
                             which genuinely requires a new inverse

--dim-sweep re-times the head across the pooling dimensions we actually use,
because the inverse is O(D^3): the chunks3+adjdiff pooling that won +9.0pp on
SSv2 has D=2560, where a full update costs far more than the whole frame budget.

Run: python3 dev/bench_realtime_incremental.py [--frames 60] [--classes 48]
     python3 dev/bench_realtime_incremental.py --dim-sweep
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fecam_head import POOLING_DIM_FACTOR, FeCAMHead
from src.models.fecam_head import POOLINGS as POOLING_FNS  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

N_FRAMES = 16
DIM = 512
FPS_TARGET = 10.0
BUDGET_MS = 1000.0 / FPS_TARGET       # 100 ms per frame


def timed(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"mean": float(np.mean(ts)), "p50": ts[len(ts) // 2],
            "p95": ts[int(len(ts) * 0.95)] if len(ts) > 1 else ts[0]}


POOLINGS = [("mean (deployed)", 512), ("halves", 1024), ("thirds", 1536),
            ("chunks3+adjdiff (best SSv2)", 2560)]


def warm_head(dim, n_classes, rng):
    head = FeCAMHead(feature_dim=dim, max_classes=n_classes)
    head.observe(rng.standard_normal((n_classes * 40, dim)),
                 np.repeat(np.arange(n_classes), 40))
    head.scores(rng.standard_normal((1, dim)))      # warm both caches
    return head


def dim_sweep(n_classes):
    """Head-only cost vs feature dimension. The encoder is dimension-independent,
    so this isolates what changing the pooling does to the real-time budget."""
    rng = np.random.default_rng(0)
    print(f"head-only, {n_classes} classes, budget {BUDGET_MS:.0f} ms/frame "
          f"(encoder adds a constant ~27 ms)\n")
    print(f"{'pooling':30s} {'D':>5s} {'predict':>9s} {'+enroll':>9s} {'+full':>9s}")
    print("-" * 66)
    rows = {}
    for name, dim in POOLINGS:
        head = warm_head(dim, n_classes, rng)
        e = rng.standard_normal((1, dim))

        def predict():
            head.scores(e)

        def enroll():
            head.observe(e, np.array([0]), update_cov=False)
            head.scores(e)

        def full():
            head.observe(e, np.array([0]))
            head.scores(e)

        r = {"predict": timed(predict, 10)["mean"], "enroll": timed(enroll, 10)["mean"],
             "full": timed(full, 5)["mean"]}
        rows[name] = {**r, "dim": dim}
        print(f"{name:30s} {dim:5d} {r['predict']:8.2f}ms {r['enroll']:8.2f}ms "
              f"{r['full']:8.1f}ms")
    out = ROOT / "reports/realtime_dim_sweep_raw.json"
    save_results(out, rows)
    print(f"\nraw -> {out}")


def _rss_mb():
    """Resident set size. psutil if present, else peak-only via resource."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6, True
    except ImportError:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (peak if sys.platform == "darwin" else peak * 1024) / 1e6, False


def head_bytes(head):
    """Exact footprint of what the head holds, by component."""
    parts = {"means": head.means.nbytes, "counts": head.counts.nbytes,
             "cov_sum": head._cov_sum.nbytes}
    if head._cov_cache is not None:
        prec, sd, _ = head._cov_cache
        parts["precision"] = prec.nbytes
        parts["sd"] = sd.nbytes
    if head._mean_cache is not None:
        active, mu, mu_prec, mu_quad = head._mean_cache
        parts["mean_cache"] = mu.nbytes + mu_prec.nbytes + mu_quad.nbytes + active.nbytes
    return parts


def memory_report(n_classes=48):
    """Where the bytes actually go, and how the head scales with the pooling.

    The one RAM figure on record (0.95 GB) is from the GRU era and measured a
    training run; the deployed system is a different shape. This separates the
    encoder from the head, because only the head grows with the pooling choice.
    """
    rss0, live = _rss_mb()
    kind = "RSS" if live else "peak RSS (psutil absent)"
    print(f"process {kind}\n  {'after imports':28s} {rss0:8.1f} MB")

    from transformers import AutoModel
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    rss1, _ = _rss_mb()
    n_par = sum(p.numel() for p in model.vision_model.parameters())
    print(f"  {'+ CLIP vision tower':28s} {rss1:8.1f} MB  (+{rss1-rss0:.0f}, "
          f"{n_par/1e6:.1f}M params)")

    head = FeCAMHead.load(ROOT / "checkpoints/fecam_head.npz")
    head.scores(np.zeros((1, head.feature_dim)))          # warm both caches
    rss2, _ = _rss_mb()
    print(f"  {'+ FeCAM head (served)':28s} {rss2:8.1f} MB  (+{rss2-rss1:.0f})")

    print(f"\nhead components, D={head.feature_dim}, max_classes={head.max_classes}")
    parts = head_bytes(head)
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v/1e6:8.1f} MB")
    print(f"  {'total':28s} {sum(parts.values())/1e6:8.1f} MB")

    print(f"\nhead RAM vs pooling (max_classes={head.max_classes}, "
          f"two DxD matrices dominate)")
    print(f"{'pooling':22s} {'D':>6s} {'checkpoint':>12s} {'RAM':>9s}")
    print("-" * 54)
    rows = {}
    for name, factor in sorted(POOLING_DIM_FACTOR.items(), key=lambda kv: kv[1]):
        d = 512 * factor
        h = FeCAMHead(feature_dim=d, max_classes=head.max_classes, pooling=name)
        h.observe(np.random.default_rng(0).standard_normal((n_classes * 2, d)),
                  np.repeat(np.arange(n_classes), 2))
        h.scores(np.zeros((1, d)))
        ram = sum(head_bytes(h).values()) / 1e6
        ckpt = (d * (d + 1) // 2 * 4 + h.means.nbytes) / 1e6   # float32 triangle + means
        rows[name] = {"dim": d, "ram_mb": ram, "checkpoint_mb": ckpt}
        print(f"{name:22s} {d:6d} {ckpt:10.1f} MB {ram:7.1f} MB")

    out = {"rss_mb": {"imports": rss0, "with_clip": rss1, "with_head": rss2,
                      "live_rss": live},
           "served_head_components_bytes": parts, "by_pooling": rows}
    save_results(ROOT / "reports/realtime_memory_raw.json", out)
    print(f"\nraw -> reports/realtime_memory_raw.json")


def class_sweep(dim=DIM, counts=(48, 101, 250, 500)):
    """Does the number of enrolled classes threaten the frame budget?

    Scoring is O(K*D) and the inverse is O(D^3) -- independent of K -- so the
    expectation is "no". Pinned here because a product keeps enrolling classes.
    """
    rng = np.random.default_rng(0)
    print(f"head-only, D={dim}, budget {BUDGET_MS:.0f} ms/frame\n")
    print(f"{'classes':>8s} {'predict':>9s} {'+enroll':>9s} {'+full':>9s}")
    print("-" * 40)
    rows = {}
    for k in counts:
        head = warm_head(dim, k, rng)
        e = rng.standard_normal((1, dim))
        r = {
            "predict": timed(lambda: head.scores(e), 10)["mean"],
            "enroll": timed(lambda: (head.observe(e, np.array([0]), update_cov=False),
                                     head.scores(e)), 10)["mean"],
            "full": timed(lambda: (head.observe(e, np.array([0])), head.scores(e)), 5)["mean"],
        }
        rows[k] = r
        print(f"{k:8d} {r['predict']:8.2f}ms {r['enroll']:8.2f}ms {r['full']:8.2f}ms")
    out = ROOT / "reports/realtime_class_sweep_raw.json"
    save_results(out, rows)
    print(f"\nraw -> {out}")


UCF_DIR = ROOT / "data/features_ucf101_b32"
SSV2_DIR = ROOT / "data/features"

# stream_sim was previously UCF101-only; --dataset ssv2 was assumed blocked on
# missing fps/frame-count metadata (see ssv2_video_access_result.md), but that
# assumption was wrong -- this function never touches fps or frame count at
# all, it streams already-extracted mean-pooled feature vectors in random
# order. The real blocker was just "never pointed at the SSv2 features dir",
# which is now trivial to fix now that data/features/ has clean, provenance-
# stamped SSv2 features (also true before this session, just not used here).


def _load_stream_samples(dataset, warm, stream, pooling="mean"):
    need = warm + stream
    if dataset == "ucf101":
        man = json.loads((UCF_DIR / "manifest.json").read_text())
        samples = man["splits"]["train"]["samples"]
        feat_dir, max_classes = UCF_DIR / "train", 101
    else:
        samples = json.loads((ROOT / "data/subset/train_mini.json").read_text())
        feat_dir, max_classes = SSV2_DIR / "train", 48
    rng = np.random.default_rng(0)
    rng.shuffle(samples)
    pool_fn = POOLING_FNS[pooling]
    X, y = [], []
    for s in samples[:need]:
        p = feat_dir / f"{s['id']}.npy"
        if p.exists():
            X.append(pool_fn(np.load(p).astype(np.float64)))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y), max_classes


def stream_sim(warm=3000, stream=800, periods=(1, 5, 20, 100, 0), dataset="ucf101",
              pooling="mean"):
    """Accuracy vs how often the covariance inverse is refreshed.

    The enroll path no longer needs the inverse at all, but a full update
    genuinely changes the covariance, and at D>=1536 refreshing it every frame
    does not fit the budget (see --dim-sweep). So: how stale can the inverse get
    before accuracy suffers?

    Prequential (test-then-train) on real features (UCF101 or SSv2), pooled with
    `pooling`, streamed one sample at a time. `period=0` means never refresh
    after the warm start. Pooling matters here specifically because the
    inverse's O(D^3) cost is where refresh-period staleness actually bites --
    mean (D=512) barely notices any period, but the deployed chunks4 (D=2048)
    and chunks3_adjdiff (D=2560) are where --dim-sweep found a full refresh can
    blow the frame budget, so THIS is the condition that answers "how often
    must we actually refresh," not the D=512 case alone.
    """
    X, y, max_classes = _load_stream_samples(dataset, warm, stream, pooling)
    warm = min(warm, len(X) - 100)
    dim = X.shape[1]
    print(f"{dataset.upper()} real features | pooling={pooling} (D={dim}) | "
          f"warm start {warm}, then stream {len(X)-warm} one at a time (predict, then learn)\n")
    print(f"{'covariance refresh':22s} {'accuracy':>9s} {'head ms/frame':>14s} {'head fps':>9s}")
    print("-" * 60)

    rows = {}
    for period in periods:
        head = FeCAMHead(feature_dim=dim, max_classes=max_classes)
        head.observe(X[:warm], y[:warm])
        head.scores(X[:1])
        correct, t0 = 0, time.perf_counter()
        for i in range(warm, len(X)):
            xi = X[i:i + 1]
            correct += int(head.scores(xi).argmax() == y[i])
            saved = head._cov_cache
            head.observe(xi, y[i:i + 1])        # accumulates the covariance
            # Not a refresh step: keep using the previous inverse. The means are
            # still patched exactly -- only the D x D term goes stale.
            if period == 0 or (i - warm + 1) % period:
                head._cov_cache = saved
        dt = (time.perf_counter() - t0) / (len(X) - warm) * 1000
        acc = correct / (len(X) - warm)
        label = "every frame" if period == 1 else ("never" if period == 0
                                                   else f"every {period}")
        rows[label] = {"accuracy": acc, "ms_per_frame": dt}
        print(f"{label:22s} {100*acc:8.2f}% {dt:13.2f}ms {1000/dt:8.0f}")

    out = ROOT / f"reports/realtime_stream_sim_{dataset}_{pooling}_raw.json"
    save_results(out, rows)
    print(f"\nraw -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40, help="frames to simulate")
    ap.add_argument("--classes", type=int, default=48)
    ap.add_argument("--device", default="cpu", help="cpu is the edge-relevant case")
    ap.add_argument("--dim-sweep", action="store_true",
                    help="head-only timings across the poolings we use")
    ap.add_argument("--stream-sim", action="store_true",
                    help="accuracy vs covariance-refresh period on real features")
    ap.add_argument("--dataset", choices=["ucf101", "ssv2"], default="ucf101",
                    help="which real features --stream-sim streams")
    ap.add_argument("--pooling", choices=list(POOLING_FNS), default="mean",
                    help="pooling for --stream-sim -- D scales the refresh cost")
    ap.add_argument("--class-sweep", action="store_true",
                    help="head cost vs number of enrolled classes")
    ap.add_argument("--memory", action="store_true",
                    help="where RAM goes: encoder vs head, and head scaling")
    args = ap.parse_args()

    if args.dim_sweep:
        dim_sweep(args.classes)
        return
    if args.stream_sim:
        stream_sim(dataset=args.dataset, pooling=args.pooling)
        return
    if args.class_sweep:
        class_sweep()
        return
    if args.memory:
        memory_report(args.classes)
        return

    from transformers import AutoModel, AutoProcessor
    dev = args.device
    print(f"device: {dev} | target {FPS_TARGET:.0f} fps => budget {BUDGET_MS:.0f} ms/frame\n")
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    rng = np.random.default_rng(0)
    frame = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))

    # a warm head with `classes` enrolled, as a deployed demo would have
    head = warm_head(DIM, args.classes, rng)

    ring = rng.standard_normal((N_FRAMES, DIM))

    # ── stage 1: preprocess (PIL -> pixel tensor), one arriving frame ────────
    def preprocess():
        return proc(images=[frame], return_tensors="pt")["pixel_values"]

    px = preprocess().to(dev)

    # ── stage 2: CLIP encode, batch=1 (streaming cannot batch the future) ────
    @torch.no_grad()
    def encode():
        v = model.vision_model(pixel_values=px)
        f = model.visual_projection(v.pooler_output)
        return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ── stage 3: ring-buffer mean-pool ──────────────────────────────────────
    def pool():
        return ring.mean(axis=0)

    emb = pool()[None, :]

    # ── stage 4: incremental learn, one window, both modes ──────────────────
    def enroll():                       # means only -- the demo's path
        head.observe(emb, np.array([0]), update_cov=False)

    def learn_full():                   # also updates the shared covariance
        head.observe(emb, np.array([0]))

    # ── stage 5: predict ────────────────────────────────────────────────────
    def scores_clean():                 # no update pending: both caches valid
        return head.scores(emb)

    def scores_after_enroll():          # one mean row to patch
        head.observe(emb, np.array([0]), update_cov=False)
        return head.scores(emb)

    def scores_after_full():            # covariance moved: new DxD inverse
        head.observe(emb, np.array([0]))
        return head.scores(emb)

    # ── stage 5c: the DxD inverse alone ─────────────────────────────────────
    def rebuild_inverse():
        head._cov_cache = None
        head._cov_terms()

    n = max(args.frames, 10)
    stages = {
        "1. preprocess (PIL->tensor)": timed(preprocess, n),
        "2. CLIP encode (batch=1)": timed(encode, n),
        "3. ring-buffer mean-pool": timed(pool, n),
        "4a. FeCAM observe (enroll)": timed(enroll, n),
        "4b. FeCAM observe (full)": timed(learn_full, n),
        "5. FeCAM scores (no update)": timed(scores_clean, n),
        "5a. observe(enroll) + scores": timed(scores_after_enroll, n),
        "5b. observe(full) + scores": timed(scores_after_full, n),
        "5c. -- DxD inverse alone": timed(rebuild_inverse, n),
    }

    print(f"{'stage':34s} {'mean':>9s} {'p50':>9s} {'p95':>9s}  {'% budget':>9s}")
    print("-" * 78)
    for k, v in stages.items():
        print(f"{k:34s} {v['mean']:8.2f}ms {v['p50']:8.2f}ms {v['p95']:8.2f}ms "
              f"{100*v['mean']/BUDGET_MS:8.1f}%")

    pre = stages["1. preprocess (PIL->tensor)"]["mean"]
    enc = stages["2. CLIP encode (batch=1)"]["mean"]
    pl = stages["3. ring-buffer mean-pool"]["mean"]
    sc = stages["5. FeCAM scores (no update)"]["mean"]
    head_enroll = stages["5a. observe(enroll) + scores"]["mean"]
    head_full = stages["5b. observe(full) + scores"]["mean"]

    front = pre + enc + pl                  # everything before the head
    predict_only = front + sc
    learn_enroll = front + head_enroll
    learn_full_e2e = front + head_full

    print(f"\n{'end-to-end per frame':34s} {'total':>9s} {'fps':>9s}  {'verdict':>9s}")
    print("-" * 78)
    for name, tot in (("predict only", predict_only),
                      ("learn (enroll) + predict", learn_enroll),
                      ("learn (full cov) + predict", learn_full_e2e)):
        ok = "OK" if tot <= BUDGET_MS else "OVER"
        print(f"{name:34s} {tot:8.2f}ms {1000.0/tot:8.1f} {ok:>9s}")

    print(f"\nmarginal cost of learning (enroll): {learn_enroll - predict_only:.2f} ms/frame "
          f"({100*(learn_enroll-predict_only)/BUDGET_MS:.1f}% of budget)")
    print(f"marginal cost of learning (full):   {learn_full_e2e - predict_only:.2f} ms/frame "
          f"({100*(learn_full_e2e-predict_only)/BUDGET_MS:.1f}% of budget)")
    print(f"encoder share of learn(enroll)+predict: {100*enc/learn_enroll:.1f}%")
    print(f"head share (observe+scores):            {100*head_enroll/learn_enroll:.1f}%")
    print(f"\nheadroom at 10 fps: {BUDGET_MS - learn_enroll:.1f} ms/frame")

    out = ROOT / "reports/realtime_incremental_raw.json"
    save_results(out, {
        "device": dev, "budget_ms": BUDGET_MS, "n_classes": args.classes,
        "dim": DIM, "stages_ms": stages,
        "predict_only_ms": predict_only, "learn_enroll_ms": learn_enroll,
        "learn_full_ms": learn_full_e2e,
        "encoder_share_pct": 100 * enc / learn_enroll,
        "head_share_pct": 100 * head_enroll / learn_enroll,
    })
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
