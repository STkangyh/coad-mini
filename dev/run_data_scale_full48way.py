"""
Does more training data help the HARD metric too?

The original data-scale study (experiments/run_data_scale.py) only measured
task-aware 6-way accuracy (chance=1/6) and found it still rising at 100% data.
This script re-runs the same (seed, fraction) protocol on CLIP B/32 and
additionally evaluates FULL 48-way (true class-IL, no task ID, chance=1/48)
accuracy / precision / recall / F1(macro) / mAP at the end of each run, to see
whether the "desperately low" full-48-way numbers (baseline 6.2%, A-GEM 9.9%
at 100% data) are mainly a data-scarcity problem or something else
(forgetting-induced logit skew, frozen-CLIP's weakness on SSv2 motion verbs).

Reuses subsample_stage/MAIN_CONFIG from experiments/run_data_scale.py so the
per-stage sample selection is identical to the original study (deterministic,
backbone-independent).

Run: python3 dev/run_data_scale_full48way.py
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score

from experiments.run_capacity_ablation import (
    MEM_PER_STAGE, REPLAY_RATIO, STAGES, device, eval_stage, filter_by,
    make_model, train_all, val_all,
)
from experiments.run_data_scale import MAIN_CONFIG, subsample_stage
from src.trainer import set_seed, train_epoch
from src.utils.gem import AGEM

FRACTIONS = [0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]
EPOCHS = 15
FEATURE_DIR = Path("data/features")   # CLIP B/32
FEATURE_DIM = 512
LR = 1e-4
N_CLASSES = 48


@torch.no_grad()
def full_48way_metrics(model, feat_dir: Path) -> dict:
    model.eval()
    y_true, probs = [], []
    for s in val_all:
        p = feat_dir / "val" / f"{s['id']}.npy"
        if not p.exists():
            continue
        x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0)
        logits, _ = model(x, None)
        probs.append(torch.softmax(logits, dim=-1)[0].numpy())
        y_true.append(s["class_id"])
    y_true = np.array(y_true); probs = np.array(probs)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    mAP = average_precision_score(np.eye(N_CLASSES)[y_true], probs, average="macro")
    return {"accuracy": acc, "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m, "mAP": mAP}


def run_one(seed: int, fraction: float):
    set_seed(seed)
    model = make_model(MAIN_CONFIG, FEATURE_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    gem = AGEM(mem_per_stage=MEM_PER_STAGE, selection=MAIN_CONFIG.selection, replay_ratio=REPLAY_RATIO)

    acc_table = {}
    for stage_id, class_ids in STAGES.items():
        stage_train = filter_by(train_all, class_ids)
        stage_train = subsample_stage(stage_train, fraction, seed, stage_id)
        for _ in range(EPOCHS):
            gem.precompute_ref(model, device)
            train_epoch(model, stage_train, FEATURE_DIR / "train", criterion, optimizer, device, orth=gem)
        acc_table[stage_id] = {sid: eval_stage(model, cids, FEATURE_DIR) for sid, cids in STAGES.items()}
        gem.add_stage(stage_train, FEATURE_DIR / "train", model=model, device=device)

    last = max(STAGES)
    task_aware = float(np.mean([acc_table[last][sid] for sid in STAGES]))
    full = full_48way_metrics(model, FEATURE_DIR)
    return task_aware, full


def main():
    print(f"device: {device}  fractions={FRACTIONS} seeds={SEEDS} epochs={EPOCHS}")
    results = {f: {"task_aware": [], "full": []} for f in FRACTIONS}
    for fraction in FRACTIONS:
        print(f"\n{'='*72}\n  fraction={fraction}\n{'='*72}", flush=True)
        for seed in SEEDS:
            task_aware, full = run_one(seed, fraction)
            results[fraction]["task_aware"].append(task_aware)
            results[fraction]["full"].append(full)
            print(f"fraction={fraction:g} seed={seed} task_aware={task_aware:.4f} "
                  f"full_acc={full['accuracy']:.4f} full_mAP={full['mAP']:.4f} "
                  f"full_f1={full['f1_macro']:.4f}", flush=True)

    print(f"\n\n{'='*80}\n  Data-Scale Summary — task-aware vs full-48-way\n{'='*80}")
    print(f"  {'Fraction':>8}  {'TaskAcc':>8}  {'+/-':>6}  {'FullAcc':>8}  {'+/-':>6}  "
          f"{'FullmAP':>8}  {'+/-':>6}  {'FullF1':>8}  {'+/-':>6}")
    for fraction in FRACTIONS:
        ta = np.array(results[fraction]["task_aware"])
        fa = np.array([r["accuracy"] for r in results[fraction]["full"]])
        fm = np.array([r["mAP"] for r in results[fraction]["full"]])
        ff = np.array([r["f1_macro"] for r in results[fraction]["full"]])
        print(f"  {fraction:>8g}  {ta.mean():>8.4f}  {ta.std():>6.4f}  "
              f"{fa.mean():>8.4f}  {fa.std():>6.4f}  "
              f"{fm.mean():>8.4f}  {fm.std():>6.4f}  "
              f"{ff.mean():>8.4f}  {ff.std():>6.4f}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
