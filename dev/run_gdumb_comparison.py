"""
GDumb (Prabhu et al., ECCV 2020) — "greedy balanced buffer + train from scratch,
no continual-learning trick at all". The standard sanity-check baseline for
"is your CL algorithm's gain actually just the replay buffer?".

At each stage boundary: greedily rebalance a FIXED-SIZE buffer across all
classes seen so far, then RE-INITIALIZE the model and train purely on that
buffer (no A-GEM projection, no use of the full current-stage data beyond
what fits the buffer). Same eval protocol as run_er_comparison.py so results
are directly comparable to baseline / plain ER / A-GEM.

Run: python3 dev/run_gdumb_comparison.py
"""
import random, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, ".")
from src.models.gru_detector import GRUDetector
from src.trainer import load_samples, set_seed, train_epoch

SEEDS = [0, 1, 2]
NUM_EPOCHS = 15
FEATURE_DIM = 512
CPS = 6
N_CLASSES = 48
TOTAL_BUDGET = 400          # comparable to A-GEM's ~350-400 accumulated memory
TRAIN_FEAT = Path("data/features/train")
VAL_FEAT = Path("data/features/val")
device = "cpu"

train_all = load_samples("data/subset/train_mini.json")
val_all = load_samples("data/subset/val_mini.json")
stages = {s: list(range((s-1)*CPS, s*CPS)) for s in range(1, N_CLASSES//CPS+1)}


def filt(samples, cids):
    return [s for s in samples if s["class_id"] in cids]


@torch.no_grad()
def eval_stage(model, cids):
    model.eval(); correct = total = 0
    for s in val_all:
        if s["class_id"] not in cids: continue
        p = VAL_FEAT / f"{s['id']}.npy"
        if not p.exists(): continue
        x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0)
        logits, _ = model(x, None)
        pred = cids[logits[0, cids].argmax().item()]
        total += 1; correct += int(pred == s["class_id"])
    return correct / total if total else 0.0


def rebalance_buffer(buffer_by_class, new_stage_samples, classes_seen):
    """Greedy class-balanced rebalance: trim old classes, admit new ones, uniform per-class cap."""
    by_class = {}
    for s in new_stage_samples:
        by_class.setdefault(s["class_id"], []).append(s)
    for cid, pool in by_class.items():
        buffer_by_class[cid] = pool  # start with full pool for this stage's new classes
    per_class_cap = max(1, TOTAL_BUDGET // len(classes_seen))
    for cid in list(buffer_by_class.keys()):
        pool = buffer_by_class[cid]
        if len(pool) > per_class_cap:
            buffer_by_class[cid] = random.sample(pool, per_class_cap)
    return buffer_by_class


def run_one(seed):
    set_seed(seed)
    random.seed(seed)
    buffer_by_class = {}
    acc = {}
    for sid, cids in stages.items():
        stage_train = filt(train_all, cids)
        classes_seen = sorted(set(list(buffer_by_class.keys()) + cids))
        buffer_by_class = rebalance_buffer(buffer_by_class, stage_train, classes_seen)
        buffer_flat = [s for pool in buffer_by_class.values() for s in pool]

        # GDumb: fresh model each stage, trained ONLY on the rebalanced buffer.
        model = GRUDetector(feature_dim=FEATURE_DIM, hidden_dim=256, num_classes=N_CLASSES).to(device)
        crit = nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(NUM_EPOCHS):
            train_epoch(model, buffer_flat, TRAIN_FEAT, crit, opt, device, orth=None)

        acc[sid] = {j: eval_stage(model, c) for j, c in stages.items()}
    last = max(stages)
    return {"avg_acc": float(np.mean([acc[last][j] for j in stages])),
            "s1_drop": acc[1][1] - acc[last][1],
            "buffer_size_final": len(buffer_flat)}


def main():
    results = []
    for seed in SEEDS:
        r = run_one(seed)
        results.append(r)
        print(f"seed {seed}: avg_acc={r['avg_acc']:.3f} s1_drop={r['s1_drop']:+.3f} "
              f"buffer={r['buffer_size_final']}", flush=True)
    accs = [r["avg_acc"] for r in results]
    drops = [r["s1_drop"] for r in results]
    print(f"\n===== GDumb SUMMARY ({len(SEEDS)} seeds, B/32, budget={TOTAL_BUDGET}) =====")
    print(f"avg_acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"s1_drop = {np.mean(drops):+.3f} ± {np.std(drops):.3f}")


if __name__ == "__main__":
    main()
