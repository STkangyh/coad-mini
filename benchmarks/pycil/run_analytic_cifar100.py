"""
Bridge experiment: our edge-friendly analytic heads on PyCIL's CIFAR-100 CIL protocol.

Runs NCM / Deep-SLDA / FeCAM (the backprop-free heads from
reports/cpu_friendly_methods_result.md) on frozen CLIP ViT-B/32 features over the
EXACT class-incremental splits PyCIL uses (same seed-1993 class order, imported
from PyCIL's DataManager — zero reimplementation drift), so results are directly
comparable to PyCIL baselines run with the same dataset/seed/init_cls/increment.

Evaluation is TRUE class-IL: at each task, top-1 over ALL classes seen so far
(no task ID). Reports per-task accuracy, average incremental accuracy (mean over
tasks — PyCIL's headline metric), and last-task accuracy.

Honest positioning: this is the *pretrained-model-based* CIL track (SimpleCIL /
RanPAC / FeCAM literature) — a frozen web-pretrained encoder vs PyCIL classics
that train a ResNet from scratch. The comparison shows what an edge budget
(frozen encoder + statistics-only head, no backprop) buys on a standard benchmark.

Usage:
  python3 benchmarks/pycil/run_analytic_cifar100.py                # b0=50, inc=10 (simplecil.json setting)
  python3 benchmarks/pycil/run_analytic_cifar100.py --init 10 --inc 10   # standard b0=10
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external/PyCIL"))

from src.models.fecam_head import FeCAMHead  # noqa: E402

FEAT_CACHE = REPO / "data/features_cifar100_b32"
N_CLASSES = 100
FEATURE_DIM = 512
SEED = 1993


# ── PyCIL class order (single source of truth: their DataManager) ─────────────
def pycil_class_order(init_cls: int, increment: int) -> list[int]:
    import os
    from utils.data_manager import DataManager  # PyCIL
    # PyCIL's data.py loads CIFAR from the CWD-relative "./data" — chdir to the
    # PyCIL checkout (where the verified tarball lives) so it never re-downloads.
    cwd = os.getcwd()
    os.chdir(REPO / "external/PyCIL")
    try:
        dm = DataManager("cifar100", shuffle=True, seed=SEED,
                         init_cls=init_cls, increment=increment)
    finally:
        os.chdir(cwd)
    return list(dm._class_order), list(dm._increments)


# ── CLIP feature extraction (cached) ──────────────────────────────────────────
def extract_features():
    """CIFAR-100 train/test -> frozen CLIP B/32 features, cached to .npz."""
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    out_tr, out_te = FEAT_CACHE / "train.npz", FEAT_CACHE / "test.npz"
    if out_tr.exists() and out_te.exists():
        print("feature cache found, skipping extraction")
        return

    from torchvision.datasets import CIFAR100
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"extracting CLIP B/32 features on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def run(split, out_path):
        ds = CIFAR100(str(REPO / "external/PyCIL/data"), train=(split == "train"), download=True)
        feats, labels = [], []
        B = 256
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(ds.data), B):
                imgs = [Image.fromarray(a) for a in ds.data[i:i + B]]
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
                f = model.vision_model(pixel_values=px).pooler_output
                f = model.visual_projection(f)
                f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                feats.append(f.cpu().float().numpy())
                labels.extend(ds.targets[i:i + B])
                if (i // B) % 20 == 0:
                    print(f"  [{split}] {i}/{len(ds.data)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        np.savez_compressed(out_path, X=np.concatenate(feats), y=np.array(labels))
        print(f"  saved {out_path}")

    run("train", out_tr)
    run("test", out_te)


def load_features():
    tr = np.load(FEAT_CACHE / "train.npz")
    te = np.load(FEAT_CACHE / "test.npz")
    return (tr["X"].astype(np.float64), tr["y"].astype(np.int64),
            te["X"].astype(np.float64), te["y"].astype(np.int64))


# ── heads (same as dev/run_cpu_friendly_methods.py, sized for 100 classes) ────
class NCM:
    name = "NCM prototype"

    def __init__(self):
        self.means = np.zeros((N_CLASSES, FEATURE_DIM))
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
    name = "Deep SLDA"

    def __init__(self, shrink=1e-2):
        self.means = np.zeros((N_CLASSES, FEATURE_DIM))
        self.counts = np.zeros(N_CLASSES)
        self.cov = np.zeros((FEATURE_DIM, FEATURE_DIM))
        self.n = 0
        self.shrink = shrink

    def observe(self, X, y):
        for c in np.unique(y):
            m = X[y == c]
            tot = self.counts[c] + len(m)
            self.means[c] = (self.means[c] * self.counts[c] + m.sum(axis=0)) / tot
            self.counts[c] = tot
        D = X - self.means[y]
        self.cov = (self.cov * self.n + D.T @ D) / (self.n + len(X))
        self.n += len(X)

    def scores(self, X):
        prec = np.linalg.inv(self.cov + self.shrink * np.eye(FEATURE_DIM))
        W = self.means @ prec
        b = -0.5 * np.einsum("cd,cd->c", W, self.means)
        s = X @ W.T + b
        s[:, self.counts == 0] = -1e9
        return s


def make_fecam():
    h = FeCAMHead(feature_dim=FEATURE_DIM, max_classes=N_CLASSES)
    h.name = "FeCAM (shared cov)"
    return h


# ── protocol ──────────────────────────────────────────────────────────────────
def run_head(head, tasks, Xtr, ytr, Xte, yte):
    """tasks: list of lists of (mapped) class ids per task."""
    accs = []
    t0 = time.perf_counter()
    seen: list[int] = []
    for cids in tasks:
        mask = np.isin(ytr, cids)
        head.observe(Xtr[mask], ytr[mask])
        seen.extend(cids)
        te_mask = np.isin(yte, seen)
        s = head.scores(Xte[te_mask])
        s[:, [c for c in range(N_CLASSES) if c not in seen]] = -1e18
        accs.append(float((s.argmax(axis=1) == yte[te_mask]).mean()))
    return {"per_task": accs, "avg_inc": float(np.mean(accs)),
            "last": accs[-1], "train_s": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=int, default=50)
    ap.add_argument("--inc", type=int, default=10)
    args = ap.parse_args()

    extract_features()
    Xtr, ytr_raw, Xte, yte_raw = load_features()

    order, increments = pycil_class_order(args.init, args.inc)
    remap = {orig: i for i, orig in enumerate(order)}     # PyCIL maps to order-index
    ytr = np.array([remap[int(c)] for c in ytr_raw])
    yte = np.array([remap[int(c)] for c in yte_raw])
    tasks, cursor = [], 0
    for inc in increments:
        tasks.append(list(range(cursor, cursor + inc)))
        cursor += inc

    print(f"\nCIFAR-100 class-IL  b0={args.init} inc={args.inc}  "
          f"({len(tasks)} tasks, PyCIL seed={SEED} class order)")
    print(f"{'head':22s} {'avg_inc':>8} {'last':>8} {'train_s':>8}  per-task")
    for head in [NCM(), SLDA(), make_fecam()]:
        r = run_head(head, tasks, Xtr, ytr, Xte, yte)
        curve = " ".join(f"{a:.3f}" for a in r["per_task"])
        print(f"{head.name:22s} {r['avg_inc']:8.3f} {r['last']:8.3f} "
              f"{r['train_s']:8.1f}  {curve}", flush=True)


if __name__ == "__main__":
    main()
