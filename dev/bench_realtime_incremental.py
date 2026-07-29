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

The learn+predict mode has a non-obvious cost: FeCAM.observe() sets _cache = None
(fecam_head.py:80), so the next scores() must rebuild the DxD correlation inverse.
In a steady train-then-predict loop that inverse is paid EVERY frame, which never
happens in the batch experiments where we fit once and score many times. This
script measures that term explicitly rather than assuming it is negligible.

Run: python3 dev/bench_realtime_incremental.py [--frames 60] [--classes 48]
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

from src.models.fecam_head import FeCAMHead  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40, help="frames to simulate")
    ap.add_argument("--classes", type=int, default=48)
    ap.add_argument("--device", default="cpu", help="cpu is the edge-relevant case")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    dev = args.device
    print(f"device: {dev} | target {FPS_TARGET:.0f} fps => budget {BUDGET_MS:.0f} ms/frame\n")
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    rng = np.random.default_rng(0)
    frame = Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))

    # a warm head with `classes` enrolled, as a deployed demo would have
    head = FeCAMHead(feature_dim=DIM, max_classes=args.classes)
    X = rng.standard_normal((args.classes * 40, DIM))
    y = np.repeat(np.arange(args.classes), 40)
    head.observe(X, y)
    head.scores(X[:1])                      # warm the precision cache

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

    # ── stage 4: incremental learn (one window) ─────────────────────────────
    def observe():
        head.observe(emb, np.array([0]))

    # ── stage 5: predict (cache warm) ───────────────────────────────────────
    def scores_warm():
        head._cache is None and head._precision()
        return head.scores(emb)

    # ── stage 5b: predict right after an update (cache invalidated) ─────────
    def scores_cold():
        head._cache = None
        return head.scores(emb)

    # ── stage 5c: just the cache rebuild (the DxD inverse) ──────────────────
    def rebuild_cache():
        head._cache = None
        head._precision()

    n = max(args.frames, 10)
    stages = {
        "1. preprocess (PIL->tensor)": timed(preprocess, n),
        "2. CLIP encode (batch=1)": timed(encode, n),
        "3. ring-buffer mean-pool": timed(pool, n),
        "4. FeCAM observe (learn)": timed(observe, n),
        "5. FeCAM scores (cache warm)": timed(scores_warm, n),
        "5b. FeCAM scores (cache cold)": timed(scores_cold, n),
        "5c. -- of which: cache rebuild": timed(rebuild_cache, n),
    }

    print(f"{'stage':34s} {'mean':>9s} {'p50':>9s} {'p95':>9s}  {'% budget':>9s}")
    print("-" * 78)
    for k, v in stages.items():
        print(f"{k:34s} {v['mean']:8.2f}ms {v['p50']:8.2f}ms {v['p95']:8.2f}ms "
              f"{100*v['mean']/BUDGET_MS:8.1f}%")

    pre = stages["1. preprocess (PIL->tensor)"]["mean"]
    enc = stages["2. CLIP encode (batch=1)"]["mean"]
    pl = stages["3. ring-buffer mean-pool"]["mean"]
    obs = stages["4. FeCAM observe (learn)"]["mean"]
    sc_warm = stages["5. FeCAM scores (cache warm)"]["mean"]
    sc_cold = stages["5b. FeCAM scores (cache cold)"]["mean"]

    predict_only = pre + enc + pl + sc_warm
    learn_predict = pre + enc + pl + obs + sc_cold
    head_only = obs + sc_cold

    print(f"\n{'end-to-end per frame':34s} {'total':>9s} {'fps':>9s}  {'verdict':>9s}")
    print("-" * 78)
    for name, tot in (("predict only", predict_only),
                      ("learn + predict (incremental)", learn_predict)):
        fps = 1000.0 / tot
        ok = "OK" if tot <= BUDGET_MS else "OVER"
        print(f"{name:34s} {tot:8.2f}ms {fps:8.1f} {ok:>9s}")

    print(f"\nmarginal cost of learning: {learn_predict - predict_only:.2f} ms/frame "
          f"({100*(learn_predict-predict_only)/BUDGET_MS:.1f}% of budget)")
    print(f"encoder share of learn+predict: {100*enc/learn_predict:.1f}%")
    print(f"head share (observe+scores):    {100*head_only/learn_predict:.1f}%")

    out = ROOT / "reports/realtime_incremental_raw.json"
    out.write_text(json.dumps({
        "device": dev, "budget_ms": BUDGET_MS, "n_classes": args.classes,
        "stages_ms": stages,
        "predict_only_ms": predict_only, "learn_predict_ms": learn_predict,
        "encoder_share_pct": 100 * enc / learn_predict,
        "head_share_pct": 100 * head_only / learn_predict,
    }, indent=2))
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
