"""
Which backbone x pooling x head combination wins on the accuracy-vs-speed plane?

Everything in this repo so far optimized one axis at a time: pooling was chosen
on SSv2 accuracy (ssv2_pooling_protocol), heads were compared at fixed pooling
(ssv2_head_curves), and speed was measured only for the deployed config
(realtime_incremental_result). Nothing put all three axes on one plane, so
"chunks4 + FeCAM on CLIP B/32" was never actually shown to be Pareto-optimal --
it was assembled from three separate one-axis decisions.

AXES
  backbone  clip_b32 (512-d)  vs  openclip_l14 (768-d)   -- both already extracted
  pooling   mean / chunks4 / chunks3_adjdiff             -- 1x / 4x / 5x dim
  head      NCM / SLDA / Ridge-RLS / FeCAM               -- all backprop-free

METRICS
  mAP     macro average precision over the full 48-way task.

          COMPARABILITY FIX (found while running this, do not remove): macro AP
          ranks SAMPLES within each class column, so it is sensitive to
          per-sample score offsets that top-1 accuracy is blind to. FeCAM's raw
          scores are negative Mahalanobis distances whose per-sample offset
          spans ~1176 while the within-sample spread is only ~5-14 -- so raw-AP
          was measuring "is this sample close to everything" rather than "is
          this sample class c", and reported FeCAM at mAP 0.028 while its
          accuracy was the best in the sweep. Row z-scoring the score matrix
          (per sample, across classes) before computing AP moves FeCAM from
          0.028 to 0.215 and leaves argmax -- hence accuracy -- untouched.
          Every head is normalized identically, so the comparison is on
          ranking quality rather than on each head's arbitrary score scale.
          `mAP_raw` is kept alongside so the artifact stays visible.
  FPS     end-to-end streaming rate on CPU, modelled the way
          realtime_incremental_result.md established the deployed path works:
          a ring buffer means only ONE new frame is encoded per step, then the
          16-frame buffer is pooled and scored. So
              fps = 1000 / (encode_1frame_ms + pool_ms + head_scores_ms)
          CPU, batch=1 -- the edge-relevant case, not the batch throughput that
          makes encoders look 2x faster than they are in streaming.

The encoder term dominates (91% per realtime_incremental_result.md), so the
backbone axis is where FPS is really decided -- but that is the point: this
script checks whether the accuracy openclip_l14 buys is worth what it costs.

Run: python3 dev/run_ap_fps_sweep.py [--backbones clip_b32 openclip_l14] [--skip-encoder-timing]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score  # noqa: E402

from src.models.fecam_head import POOLINGS, FeCAMHead  # noqa: E402
from src.trainer import load_samples  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402

N_CLASSES = 48
CPS = 6
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_CLASSES // CPS + 1)}

BACKBONE_DIRS = {
    "clip_b32": (ROOT / "data/features", "openai/clip-vit-base-patch32", 512),
    "openclip_l14": (ROOT / "data/features_openclip_l14",
                     "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", 768),
}
POOLING_NAMES = ["mean", "chunks4", "chunks3_adjdiff"]
RIDGE_LAMBDA = 1e2


# ── heads, parameterized by dim (the repo's existing copies hardcode globals) ──
class NCM:
    """SimpleCIL-style class-mean prototype, cosine scored."""
    name = "NCM"

    def __init__(self, dim):
        self.means = np.zeros((N_CLASSES, dim))
        self.counts = np.zeros(N_CLASSES)

    def observe(self, X, y):
        for c in np.unique(y):
            m = X[y == c]
            self.means[c] = m.mean(axis=0)
            self.counts[c] = len(m)

    def scores(self, X):
        mu = self.means / (np.linalg.norm(self.means, axis=1, keepdims=True) + 1e-12)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        s = Xn @ mu.T
        s[:, self.counts == 0] = -1e9
        return s


class SLDA:
    """Deep SLDA (Hayes & Kanan '20): streaming means + one shared covariance.

    Fitted batch-wise here rather than sample-at-a-time: the repo's streaming
    copy in run_cpu_friendly_methods.py updates per sample, which is O(N*D^2)
    and takes minutes at D=3840. The resulting statistics are the same running
    mean/covariance either way -- only the update schedule differs, and this
    script measures PREDICT time, not fit time.
    """
    name = "SLDA"

    def __init__(self, dim, shrink=1e-2):
        self.dim = dim
        self.means = np.zeros((N_CLASSES, dim))
        self.counts = np.zeros(N_CLASSES)
        self.cov = np.zeros((dim, dim))
        self.n = 0
        self.shrink = shrink
        self._W = self._b = None

    def observe(self, X, y):
        for c in np.unique(y):
            m = X[y == c]
            self.means[c] = m.mean(axis=0)
            self.counts[c] = len(m)
            d = m - self.means[c]
            self.cov += d.T @ d
            self.n += len(m)
        self._W = None                      # invalidate cached solve

    def _fit(self):
        cov = self.cov / max(1, self.n)
        prec = np.linalg.inv(cov + self.shrink * np.eye(self.dim))
        self._W = self.means @ prec
        self._b = -0.5 * np.einsum("cd,cd->c", self._W, self.means)

    def scores(self, X):
        if self._W is None:
            self._fit()
        s = X @ self._W.T + self._b
        s[:, self.counts == 0] = -1e9
        return s


class Ridge:
    """Recursive least squares / ACIL-style closed-form linear readout."""
    name = "Ridge-RLS"

    def __init__(self, dim):
        self.dim = dim
        self.G = np.zeros((dim, dim))
        self.C = np.zeros((dim, N_CLASSES))
        self._W = None

    def observe(self, X, y):
        Y = np.eye(N_CLASSES)[y]
        self.G += X.T @ X
        self.C += X.T @ Y
        self._W = None

    def scores(self, X):
        if self._W is None:
            self._W = np.linalg.solve(self.G + RIDGE_LAMBDA * np.eye(self.dim), self.C)
        return X @ self._W


class FeCAM:
    """The deployed head, at its served settings (few_shot_correction=True)."""
    name = "FeCAM"

    def __init__(self, dim):
        self.h = FeCAMHead(feature_dim=dim, max_classes=N_CLASSES,
                           few_shot_correction=True)

    def observe(self, X, y):
        self.h.observe(X, y)

    def scores(self, X):
        return self.h.scores(X)


HEADS = [NCM, SLDA, Ridge, FeCAM]


# ── data ──────────────────────────────────────────────────────────────────────
def load_raw(feat_dir, split):
    X, y = [], []
    for s in load_samples(f"data/subset/{split}_mini.json"):
        p = feat_dir / ("train" if split == "train" else "val") / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).astype(np.float32))
            y.append(s["class_id"])
    return np.stack(X), np.array(y)


def pool_batch(X_raw, pooling):
    fn = POOLINGS[pooling]
    return np.stack([fn(x.astype(np.float64)) for x in X_raw])


# ── timing ────────────────────────────────────────────────────────────────────
def timed(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def encoder_ms_per_frame(model_name, n=10):
    """One frame, batch=1, CPU -- the streaming ring-buffer cost."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(model_name).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    def once():
        with torch.no_grad():
            inputs = processor(images=[img], return_tensors="pt")
            model.get_image_features(**inputs)

    ms = timed(once, n=n, warmup=2)
    del model, processor
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONE_DIRS),
                    choices=list(BACKBONE_DIRS))
    ap.add_argument("--poolings", nargs="+", default=POOLING_NAMES, choices=POOLING_NAMES)
    ap.add_argument("--skip-encoder-timing", action="store_true",
                    help="reuse literature values instead of measuring (fast dry run)")
    args = ap.parse_args()

    rows = []
    enc_ms = {}
    for bk in args.backbones:
        feat_dir, model_name, frame_dim = BACKBONE_DIRS[bk]
        if not args.skip_encoder_timing:
            print(f"timing encoder {bk} (CPU, batch=1)...", flush=True)
            enc_ms[bk] = encoder_ms_per_frame(model_name)
        else:
            enc_ms[bk] = float("nan")
        print(f"  {bk}: {enc_ms[bk]:.1f} ms/frame\n")

        Xtr_raw, ytr = load_raw(feat_dir, "train")
        Xva_raw, yva = load_raw(feat_dir, "val")
        print(f"{bk}: train {Xtr_raw.shape}, val {Xva_raw.shape}")

        for pooling in args.poolings:
            t0 = time.perf_counter()
            Xtr = pool_batch(Xtr_raw, pooling)
            Xva = pool_batch(Xva_raw, pooling)
            dim = Xtr.shape[1]
            # per-window pooling cost, measured on one raw window
            one = Xva_raw[0].astype(np.float64)
            pool_ms = timed(lambda: POOLINGS[pooling](one), n=200, warmup=20)

            for H in HEADS:
                head = H(dim)
                t1 = time.perf_counter()
                for cids in STAGES.values():
                    m = np.isin(ytr, cids)
                    head.observe(Xtr[m], ytr[m])
                fit_s = time.perf_counter() - t1

                S = head.scores(Xva)
                acc = float((S.argmax(axis=1) == yva).mean())
                onehot = np.eye(N_CLASSES)[yva]
                mAP_raw = float(average_precision_score(onehot, S, average="macro"))
                # Row z-score: removes each head's arbitrary per-sample offset
                # and scale, leaving argmax (and so accuracy) unchanged.
                Sz = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-12)
                mAP = float(average_precision_score(onehot, Sz, average="macro"))

                x1 = Xva[:1]
                head_ms = timed(lambda: head.scores(x1), n=50, warmup=5)
                total_ms = enc_ms[bk] + pool_ms + head_ms
                fps = 1000.0 / total_ms if total_ms == total_ms else float("nan")

                rows.append({
                    "backbone": bk, "pooling": pooling, "head": H.name, "dim": dim,
                    "mAP": mAP, "mAP_raw": mAP_raw, "accuracy": acc, "fps": fps,
                    "encode_ms": enc_ms[bk], "pool_ms": pool_ms, "head_ms": head_ms,
                    "fit_s": fit_s,
                })
                print(f"  {pooling:16s} {H.name:10s} D={dim:5d}  "
                      f"mAP={mAP:.4f} (raw {mAP_raw:.4f})  acc={acc:.4f}  "
                      f"head={head_ms:6.2f}ms  fps={fps:6.1f}", flush=True)
            print(f"  ({pooling} pooled+evaluated in {time.perf_counter()-t0:.0f}s)\n")

    # ── Pareto frontier on (fps, mAP): keep rows nothing else dominates ────────
    pareto = []
    for r in rows:
        if not any(o["fps"] >= r["fps"] and o["mAP"] >= r["mAP"] and o is not r
                   and (o["fps"] > r["fps"] or o["mAP"] > r["mAP"]) for o in rows):
            pareto.append(r)
    pareto.sort(key=lambda r: -r["fps"])

    print(f"\n{'='*78}\nPARETO FRONTIER (nothing is both faster AND more accurate)\n{'='*78}")
    print(f"{'backbone':14s} {'pooling':16s} {'head':10s} {'mAP':>8s} {'acc':>8s} {'fps':>8s}")
    for r in pareto:
        print(f"{r['backbone']:14s} {r['pooling']:16s} {r['head']:10s} "
              f"{r['mAP']:8.4f} {r['accuracy']:8.4f} {r['fps']:8.1f}")

    best_map = max(rows, key=lambda r: r["mAP"])
    print(f"\nbest mAP overall: {best_map['backbone']}/{best_map['pooling']}/"
          f"{best_map['head']}  mAP={best_map['mAP']:.4f}  fps={best_map['fps']:.1f}")

    save_results(ROOT / "reports/ap_fps_sweep_raw.json",
                 {"rows": rows, "pareto": pareto, "encoder_ms": enc_ms})
    print("raw -> reports/ap_fps_sweep_raw.json")


if __name__ == "__main__":
    main()
