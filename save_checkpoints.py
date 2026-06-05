"""
Checkpoint 저장 스크립트
========================
Baseline과 A-GEM(best config: ratio=0.25, balanced) 모델을 각각
48 classes × 8 stages로 학습 후 checkpoints/ 에 저장.

저장 형식:
  checkpoints/
    baseline_48cls.pt    — state_dict + 메타
    agem_48cls.pt        — state_dict + 메타
    class_labels.json    — class_id → label 문자열
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from utils.gem import AGEM
from models.gru_detector import GRUDetector
from trainer import (
    load_samples, set_seed,
    train_epoch,
    TRAIN_FEAT_DIR, VAL_FEAT_DIR,
)

# ── 설정 ─────────────────────────────────────────────────────────────────────
SEED              = 0
NUM_EPOCHS        = 15
FEATURE_DIM       = 512
N_CLASSES         = 48
CLASSES_PER_STAGE = 6
MEM_PER_STAGE     = 50
REPLAY_RATIO      = 0.25    # ablation best
SELECTION         = "balanced"  # ablation best
CKPT_DIR          = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ── 클래스 레이블 ─────────────────────────────────────────────────────────────
CLASS_LABELS = [
    # Stage 1
    "Opening [something]",
    "Closing [something]",
    "Turning [something] upside down",
    "Putting [something] onto [something]",
    "Folding [something]",
    "Picking [something] up",
    # Stage 2
    "Putting [something] into [something]",
    "Taking [something] out of [something]",
    "Putting [something] on a surface",
    "Taking [something] from [somewhere]",
    "Stuffing [something] into [something]",
    "Putting [something] next to [something]",
    # Stage 3
    "Moving [something] up",
    "Moving [something] down",
    "Moving [something] towards the camera",
    "Moving [something] away from the camera",
    "Moving [something] and [something] closer to each other",
    "Moving [something] and [something] away from each other",
    # Stage 4
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Pulling [something] from left to right",
    "Pulling [something] from right to left",
    "Pushing [something] with [something]",
    "Poking [something] so lightly that it doesn't or almost doesn't move",
    # Stage 5
    "Covering [something] with [something]",
    "Uncovering [something]",
    "Throwing [something]",
    "Squeezing [something]",
    "Throwing [something] against [something]",
    "Dropping [something] onto [something]",
    # Stage 6
    "Pushing [something] so that it slightly moves",
    "Pushing [something] so that it falls off the table",
    "Hitting [something] with [something]",
    "Tearing [something] into two pieces",
    "Poking [something] so it slightly moves",
    "Tearing [something] just a little bit",
    # Stage 7
    "Lifting [something] up completely without letting it drop down",
    "Lifting up one end of [something], then letting it drop down",
    "Lifting [something] up completely, then letting it drop down",
    "Lifting up one end of [something] without letting it drop down",
    "Lifting [something] with [something] on it",
    "[Something] falling like a rock",
    # Stage 8
    "Pretending to open [something] without actually opening it",
    "Pretending to pick [something] up",
    "Pretending to put [something] on a surface",
    "Showing that [something] is empty",
    "Pretending to take [something] from [somewhere]",
    "Showing that [something] is inside [something]",
]
assert len(CLASS_LABELS) == N_CLASSES

# class_labels.json 저장
with open(CKPT_DIR / "class_labels.json", "w", encoding="utf-8") as f:
    json.dump({i: lbl for i, lbl in enumerate(CLASS_LABELS)}, f,
              ensure_ascii=False, indent=2)
print(f"class_labels.json saved ({N_CLASSES} classes)")

# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
train_all = load_samples("data/subset/train_mini.json")
val_all   = load_samples("data/subset/val_mini.json")

STAGES = {s: list(range((s-1)*CLASSES_PER_STAGE, s*CLASSES_PER_STAGE))
          for s in range(1, N_CLASSES // CLASSES_PER_STAGE + 1)}

def filter_by(samples, class_ids):
    return [s for s in samples if s["class_id"] in class_ids]

def make_model():
    return GRUDetector(
        feature_dim=FEATURE_DIM,
        hidden_dim=256,
        num_classes=N_CLASSES,
    ).to(device)

@torch.no_grad()
def eval_all_stages(model):
    model.eval()
    results = {}
    for stage_id, class_ids in STAGES.items():
        correct = total = 0
        for s in val_all:
            if s["class_id"] not in class_ids:
                continue
            p = VAL_FEAT_DIR / f"{s['id']}.npy"
            if not p.exists():
                continue
            x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(x, None)
            pred = class_ids[logits[0, class_ids].argmax().item()]
            total   += 1
            correct += int(pred == s["class_id"])
        results[stage_id] = correct / total if total else 0.0
    avg = float(np.mean(list(results.values())))
    return results, avg


def train_stages(method: str, gem=None):
    """8 stages 순차 학습 후 (model, acc_table) 반환"""
    set_seed(SEED)
    model     = make_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    acc_snapshot = {}   # acc_snapshot[stage_id] = {sid: acc} after stage_id

    for stage_id, class_ids in STAGES.items():
        stage_train = filter_by(train_all, class_ids)
        print(f"\n  [{method}] Stage {stage_id}/8  ({len(stage_train)} samples)", flush=True)

        for ep in range(NUM_EPOCHS):
            if gem is not None:
                gem.precompute_ref(model, device)
            train_epoch(
                model, stage_train, TRAIN_FEAT_DIR,
                criterion, optimizer, device,
                orth=gem,
            )
            if (ep + 1) % 5 == 0:
                _, avg = eval_all_stages(model)
                print(f"    ep {ep+1:2d}  avg_acc={avg:.3f}", flush=True)

        per_stage, avg = eval_all_stages(model)
        acc_snapshot[stage_id] = per_stage
        print(f"  → after stage {stage_id}: avg={avg:.3f}  "
              + "  ".join(f"s{k}={v:.2f}" for k, v in per_stage.items()))

        if gem is not None:
            gem.add_stage(stage_train, TRAIN_FEAT_DIR)

    return model, acc_snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Baseline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*58)
print("  Training: Baseline")
print("="*58)

bl_model, bl_acc = train_stages("baseline", gem=None)

last = max(STAGES)
bl_avg   = float(np.mean([bl_acc[last][s] for s in STAGES]))
bl_s1drop = bl_acc[1][1] - bl_acc[last][1]
print(f"\n  Baseline  avg_acc={bl_avg:.3f}  s1_drop={bl_s1drop:+.3f}")

torch.save({
    "state_dict":  bl_model.state_dict(),
    "n_classes":   N_CLASSES,
    "feature_dim": FEATURE_DIM,
    "hidden_dim":  256,
    "avg_acc":     bl_avg,
    "s1_drop":     bl_s1drop,
    "acc_table":   bl_acc,
    "class_labels": CLASS_LABELS,
    "method":      "baseline",
}, CKPT_DIR / "baseline_48cls.pt")
print(f"  ✓ saved: checkpoints/baseline_48cls.pt")


# ══════════════════════════════════════════════════════════════════════════════
# A-GEM (best config)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*58)
print(f"  Training: A-GEM  (ratio={REPLAY_RATIO}, sel={SELECTION}, mem={MEM_PER_STAGE})")
print("="*58)

gem = AGEM(mem_per_stage=MEM_PER_STAGE, selection=SELECTION, replay_ratio=REPLAY_RATIO)
gem_model, gem_acc = train_stages("agem", gem=gem)

gem_avg    = float(np.mean([gem_acc[last][s] for s in STAGES]))
gem_s1drop = gem_acc[1][1] - gem_acc[last][1]
print(f"\n  A-GEM  avg_acc={gem_avg:.3f}  s1_drop={gem_s1drop:+.3f}")

torch.save({
    "state_dict":    gem_model.state_dict(),
    "n_classes":     N_CLASSES,
    "feature_dim":   FEATURE_DIM,
    "hidden_dim":    256,
    "avg_acc":       gem_avg,
    "s1_drop":       gem_s1drop,
    "acc_table":     gem_acc,
    "class_labels":  CLASS_LABELS,
    "method":        "agem",
    "mem_per_stage": MEM_PER_STAGE,
    "replay_ratio":  REPLAY_RATIO,
    "selection":     SELECTION,
}, CKPT_DIR / "agem_48cls.pt")
print(f"  ✓ saved: checkpoints/agem_48cls.pt")


# ── 최종 비교 ─────────────────────────────────────────────────────────────────
print(f"\n\n{'='*50}")
print("  Checkpoint Summary")
print(f"{'='*50}")
print(f"  {'Method':<16}  {'Avg Acc':>8}  {'S1 Drop':>8}")
print("  " + "-"*36)
print(f"  {'Baseline':<16}  {bl_avg:>8.3f}  {bl_s1drop:>+8.3f}")
print(f"  {'A-GEM(best)':<16}  {gem_avg:>8.3f}  {gem_s1drop:>+8.3f}")
print(f"  {'Δ':<16}  {gem_avg-bl_avg:>+8.3f}  {bl_s1drop-gem_s1drop:>+8.3f}")
print(f"\n  Files:")
print(f"    checkpoints/baseline_48cls.pt")
print(f"    checkpoints/agem_48cls.pt")
print(f"    checkpoints/class_labels.json")
