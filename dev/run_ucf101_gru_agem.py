"""
GRU + A-GEM on UCF101 under the TCD protocol — our own baseline, standard bench.

reports/ucf101_tcd_result.md placed FeCAM at 88.84 next to the published
numbers, but our headline claim is comparative ("backprop-free heads beat
GRU+A-GEM"), and so far that comparison exists only on our 48-class SSv2 subset
and by ordering on CIFAR-100. This runs the gradient baseline on the same
UCF101 protocol so the claim is tested where the numbers are comparable.

Same protocol as dev/run_ucf101_tcd_protocol.py: official split 1, 51-class
base, 10-class increments (the headline 10x5 column), true class-IL accuracy
after each session, averaged over TCD's three class orders.

Memory budget: TCD uses 5 exemplars/class on UCF101, so we size A-GEM's replay
per session as 5 x (classes in that session) rather than a flat per-stage
count. That both matches their budget and avoids the dilution artifact we
found in reports/base_heavy_split_result.md, where a fixed per-stage buffer
collapses when the base session is large — using a flat count here would
handicap the baseline for a reason unrelated to the method.

Usage: python3 dev/run_ucf101_gru_agem.py [--epochs 15] [--seeds 1000 1993 2021]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gru_detector import GRUDetector  # noqa: E402
from src.trainer import set_seed, train_epoch  # noqa: E402
from src.utils.gem import AGEM  # noqa: E402

FEAT = ROOT / "data/features_ucf101_b32"
N_CLASSES = 101
BASE = 51
INC = 10
DIM = 512
HIDDEN = 256
EXEMPLARS_PER_CLASS = 5          # TCD's UCF101 budget
SEEDS = [1000, 1993, 2021]


def load_manifest():
    man = json.loads((FEAT / "manifest.json").read_text())
    tr = [s for s in man["splits"]["train"]["samples"]
          if (FEAT / "train" / f"{s['id']}.npy").exists()]
    te = [s for s in man["splits"]["test"]["samples"]
          if (FEAT / "test" / f"{s['id']}.npy").exists()]
    return tr, te


def sessions_for(order):
    out = [list(order[:BASE])]
    for i in range(BASE, N_CLASSES, INC):
        out.append(list(order[i:i + INC]))
    return out


@torch.no_grad()
def eval_class_il(model, seen, test_samples, device):
    """Top-1 over every class seen so far — no task id."""
    model.eval()
    seen_set = set(seen)
    mask = torch.full((N_CLASSES,), -1e18)
    mask[list(seen_set)] = 0.0
    correct = total = 0
    for s in test_samples:
        if s["class_id"] not in seen_set:
            continue
        x = torch.tensor(np.load(FEAT / "test" / f"{s['id']}.npy"),
                         dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x, None)
        pred = int((logits[0].cpu() + mask).argmax())
        total += 1
        correct += int(pred == s["class_id"])
    return correct / total if total else 0.0


def run_seed(seed, train_samples, test_samples, epochs, device):
    set_seed(seed)
    order = np.random.RandomState(seed).permutation(N_CLASSES)
    sessions = sessions_for(order)

    model = GRUDetector(feature_dim=DIM, hidden_dim=HIDDEN,
                        num_classes=N_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem = AGEM(mem_per_stage=0, selection="balanced", replay_ratio=0.25)

    by_class: dict[int, list] = {}
    for s in train_samples:
        by_class.setdefault(s["class_id"], []).append(s)

    seen, accs = [], []
    t0 = time.perf_counter()
    for si, cids in enumerate(sessions):
        cur = [s for c in cids for s in by_class.get(c, [])]
        for _ in range(epochs):
            gem.precompute_ref(model, device)
            train_epoch(model, cur, FEAT / "train", criterion, optimizer,
                        device, orth=gem)
        seen.extend(cids)
        accs.append(eval_class_il(model, seen, test_samples, device))
        # TCD-style budget: 5 exemplars per class in this session
        gem.mem_per_stage = EXEMPLARS_PER_CLASS * len(cids)
        gem.add_stage(cur, FEAT / "train", model=model, device=device)
        print(f"    session {si} (+{len(cids)}cls, {len(seen)} seen): "
              f"{100*accs[-1]:.2f}%  [{time.perf_counter()-t0:.0f}s]", flush=True)

    return {"per_step": accs, "avg_inc": float(np.mean(accs)),
            "last": accs[-1], "train_s": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    device = "cpu"          # matches how we report every other timing
    train_samples, test_samples = load_manifest()
    print(f"UCF101 split 1 — train {len(train_samples)}, test {len(test_samples)}")
    print(f"GRU+A-GEM: {BASE}-class base + {INC}-class increments, "
          f"{args.epochs} epochs/session, {EXEMPLARS_PER_CLASS} exemplars/class, "
          f"device={device}\n")

    runs = []
    for seed in args.seeds:
        print(f"  seed {seed}")
        r = run_seed(seed, train_samples, test_samples, args.epochs, device)
        print(f"  -> avg_inc {100*r['avg_inc']:.2f}  last {100*r['last']:.2f}  "
              f"({r['train_s']:.0f}s)\n", flush=True)
        runs.append(r)

    avg = np.mean([r["avg_inc"] for r in runs])
    std = np.std([r["avg_inc"] for r in runs])
    last = np.mean([r["last"] for r in runs])
    print(f"GRU + A-GEM  (10x5)  avg_inc {100*avg:.2f}±{100*std:.2f}  "
          f"last {100*last:.2f}")
    print(f"(FeCAM on the same protocol: 88.84±0.52 / 87.44)")

    out = ROOT / "reports/ucf101_gru_agem_raw.json"
    out.write_text(json.dumps({"avg_inc": float(avg), "avg_inc_std": float(std),
                               "last": float(last), "epochs": args.epochs,
                               "runs": runs}, indent=2))
    print(f"raw -> {out}")


if __name__ == "__main__":
    main()
