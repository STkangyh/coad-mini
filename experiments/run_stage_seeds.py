"""
Multi-seed Stage Forgetting 실험
=================================
① Baseline vs A-GEM  ×  5 seeds  — N_CLASSES ∈ {16, 24, 32, 48}
   → Avg Acc ± Std,  S1 Drop ± Std

② A-GEM Memory Ablation  (mem ∈ {10, 20, 30, 50})  seed=0, N_CLASSES=48
   → Avg Acc, S1 Drop per mem size

클래스 구조: 8 stages × 6 classes = 48
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.utils.gem import AGEM
from src.models.gru_detector import GRUDetector
from src.trainer import (
    load_samples, set_seed,
    train_epoch,
    TRAIN_FEAT_DIR, VAL_FEAT_DIR,
)

# ── 공통 설정 ─────────────────────────────────────────────────────────────────
SEEDS        = [0, 1, 2, 3, 4]
NUM_EPOCHS   = 15
FEATURE_DIM  = 512
CLASSES_PER_STAGE = 6   # 8 stages × 6 = 48 classes

train_all_48 = load_samples("data/subset/train_mini.json")
val_all_48   = load_samples("data/subset/val_mini.json")
device       = "cuda" if torch.cuda.is_available() else "cpu"


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def make_stages(n_classes: int) -> dict:
    """n_classes개를 CLASSES_PER_STAGE씩 묶어 stage dict 반환"""
    cps = CLASSES_PER_STAGE
    n_stages = n_classes // cps
    return {s: list(range((s-1)*cps, s*cps)) for s in range(1, n_stages+1)}


def filter_by(samples, class_ids):
    return [s for s in samples if s["class_id"] in class_ids]


def make_model_nc(n_classes: int) -> GRUDetector:
    return GRUDetector(
        feature_dim=FEATURE_DIM,
        hidden_dim=256,
        num_classes=n_classes,
    ).to(device)


@torch.no_grad()
def eval_stage(model, val_samples, feat_dir, device, class_ids):
    model.eval()
    correct = total = 0
    for s in val_samples:
        if s["class_id"] not in class_ids:
            continue
        p = feat_dir / f"{s['id']}.npy"
        if not p.exists():
            continue
        x = torch.tensor(
            np.load(p), dtype=torch.float32
        ).unsqueeze(0).to(device)
        logits, _ = model(x, None)
        pred = class_ids[logits[0, class_ids].argmax().item()]
        total   += 1
        correct += int(pred == s["class_id"])
    return correct / total if total else 0.0


def run_one(method: str, seed: int, n_classes: int, mem_per_stage: int = 50) -> dict:
    set_seed(seed)
    stages    = make_stages(n_classes)
    train_all = filter_by(train_all_48, list(range(n_classes)))
    val_all   = filter_by(val_all_48,   list(range(n_classes)))

    model     = make_model_nc(n_classes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem       = AGEM(mem_per_stage=mem_per_stage) if method == "agem" else None

    acc_table = {}

    for stage_id, class_ids in stages.items():
        stage_train = filter_by(train_all, class_ids)

        for _ in range(NUM_EPOCHS):
            if gem is not None:
                gem.precompute_ref(model, device)
            train_epoch(
                model, stage_train, TRAIN_FEAT_DIR,
                criterion, optimizer, device,
                orth=gem,
            )

        acc_table[stage_id] = {
            sid: eval_stage(model, val_all, VAL_FEAT_DIR, device, cids)
            for sid, cids in stages.items()
        }

        if gem is not None:
            gem.add_stage(stage_train, TRAIN_FEAT_DIR)

    last    = max(stages)
    avg_acc = float(np.mean([acc_table[last][sid] for sid in stages]))
    s1_drop = acc_table[1][1] - acc_table[last][1]
    s2_drop = acc_table[2][2] - acc_table[last][2]
    return {"avg_acc": avg_acc, "s1_drop": s1_drop, "s2_drop": s2_drop}


# ══════════════════════════════════════════════════════════════════════════════
# ① Multi-seed × N_classes sweep  (mem=50 고정)
# ══════════════════════════════════════════════════════════════════════════════
CLASS_SIZES = [24, 48]   # 6 classes × {4,8} stages
                          # (16/32는 6-per-stage 구조에 맞지 않아 제외)

all_records = {}

for n_classes in CLASS_SIZES:
    n_stages = n_classes // CLASSES_PER_STAGE
    print(f"\n{'='*62}")
    print(f"  N_CLASSES={n_classes}  ({n_stages} stages × {CLASSES_PER_STAGE} classes)  mem=50")
    print(f"{'='*62}")
    records = {"baseline": [], "agem": []}

    for seed in SEEDS:
        print(f"\n  seed {seed}", flush=True)
        for method in ["baseline", "agem"]:
            r = run_one(method, seed, n_classes, mem_per_stage=50)
            records[method].append(r)
            print(f"    [{method:8s}]  avg_acc={r['avg_acc']:.3f}  "
                  f"s1_drop={r['s1_drop']:+.3f}  s2_drop={r['s2_drop']:+.3f}",
                  flush=True)

    all_records[n_classes] = records

    print(f"\n  {'Method':<12}  {'Avg Acc':>8}  {'±':>5}  {'S1 Drop':>8}  {'±':>5}")
    print("  " + "-"*48)
    for method, label in [("baseline", "Baseline"), ("agem", "A-GEM(50)")]:
        accs  = [r["avg_acc"]  for r in records[method]]
        drops = [r["s1_drop"]  for r in records[method]]
        print(f"  {label:<12}  {np.mean(accs):>8.3f}  {np.std(accs):>5.3f}"
              f"  {np.mean(drops):>+8.3f}  {np.std(drops):>5.3f}")

# ── 최종 요약 (이전 32-class 결과 포함 하드코딩) ─────────────────────────────
# 이전 run에서 얻은 32-class 결과 (4 stages × 4 classes 구조, mem=30)
prev = {
    16: {"bl_avg": 0.309, "gem_avg": 0.370, "bl_d1": +0.154, "gem_d1": +0.076},
    32: {"bl_avg": 0.338, "gem_avg": 0.445, "bl_d1": +0.016, "gem_d1": -0.058},
}

print(f"\n\n{'='*66}")
print("  Summary: Baseline vs A-GEM — Avg Acc & S1 Drop across scales")
print(f"{'='*66}")
print(f"  {'N':>4}  {'stages':>6}  {'BL Avg':>8}  {'GEM Avg':>8}  "
      f"{'Δ Avg':>7}  {'BL S1↓':>8}  {'GEM S1↓':>8}  {'Δ S1↓':>7}")
print("  " + "-"*70)

# 이전 결과 (참고)
for n in [16, 32]:
    p = prev[n]
    st = n // 4
    print(f"  {n:>4}  {st:>6}  {p['bl_avg']:>8.3f}  {p['gem_avg']:>8.3f}  "
          f"{p['gem_avg']-p['bl_avg']:>+7.3f}  {p['bl_d1']:>+8.3f}  "
          f"{p['gem_d1']:>+8.3f}  {p['bl_d1']-p['gem_d1']:>+7.3f}  (prev, 4cls/stage)")

# 신규 결과
for n_classes in CLASS_SIZES:
    r  = all_records[n_classes]
    st = n_classes // CLASSES_PER_STAGE
    bl_avg  = np.mean([x["avg_acc"] for x in r["baseline"]])
    gem_avg = np.mean([x["avg_acc"] for x in r["agem"]])
    bl_d1   = np.mean([x["s1_drop"] for x in r["baseline"]])
    gem_d1  = np.mean([x["s1_drop"] for x in r["agem"]])
    print(f"  {n_classes:>4}  {st:>6}  {bl_avg:>8.3f}  {gem_avg:>8.3f}  "
          f"{gem_avg-bl_avg:>+7.3f}  {bl_d1:>+8.3f}  "
          f"{gem_d1:>+8.3f}  {bl_d1-gem_d1:>+7.3f}  (6cls/stage)")

# ══════════════════════════════════════════════════════════════════════════════
# ② Memory Ablation (48클래스, seed=0)
# ══════════════════════════════════════════════════════════════════════════════
MEM_SIZES = [10, 20, 30, 50]

print(f"\n\n{'='*60}")
print("  Memory Ablation (A-GEM, 48 classes, 8 stages, seed=0)")
print(f"{'='*60}")
print(f"  {'mem':>5}  {'Avg Acc':>8}  {'S1 Drop':>8}  {'S2 Drop':>8}")
print("  " + "-"*38)

baseline_48 = all_records[48]["baseline"][0]  # seed=0
for mem in MEM_SIZES:
    r = run_one("agem", seed=0, n_classes=48, mem_per_stage=mem)
    print(f"  {mem:>5}  {r['avg_acc']:>8.3f}  {r['s1_drop']:>+8.3f}  {r['s2_drop']:>+8.3f}",
          flush=True)

print(f"  {'(BL)':>5}  {baseline_48['avg_acc']:>8.3f}  "
      f"{baseline_48['s1_drop']:>+8.3f}  {baseline_48['s2_drop']:>+8.3f}  ← Baseline seed=0")


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

# ── 공통 설정 ─────────────────────────────────────────────────────────────────
SEEDS        = [0, 1, 2, 3, 4]
NUM_EPOCHS   = 15
FEATURE_DIM  = 512

# 4 classes per stage, 4/6/8 stages = 16/24/32 classes
ALL_TEMPLATES = [
    # Stage 1
    "Opening [something]", "Closing [something]",
    "Turning [something] upside down", "Putting [something] onto [something]",
    # Stage 2
    "Putting [something] into [something]", "Taking [something] out of [something]",
    "Putting [something] on a surface", "Taking [something] from [somewhere]",
    # Stage 3
    "Moving [something] up", "Moving [something] down",
    "Moving [something] towards the camera", "Moving [something] away from the camera",
    # Stage 4
    "Pushing [something] from left to right", "Pushing [something] from right to left",
    "Pulling [something] from left to right", "Pulling [something] from right to left",
    # Stage 5
    "Covering [something] with [something]", "Uncovering [something]",
    "Throwing [something]", "Squeezing [something]",
    # Stage 6
    "Pushing [something] so that it slightly moves",
    "Pushing [something] so that it falls off the table",
    "Hitting [something] with [something]", "Tearing [something] into two pieces",
    # Stage 7
    "Lifting [something] up completely without letting it drop down",
    "Lifting up one end of [something], then letting it drop down",
    "Lifting [something] up completely, then letting it drop down",
    "Lifting up one end of [something] without letting it drop down",
    # Stage 8
    "Pretending to open [something] without actually opening it",
    "Pretending to pick [something] up",
    "Pretending to put [something] on a surface",
    "Showing that [something] is empty",
]

train_all_32 = load_samples("data/subset/train_mini.json")
val_all_32   = load_samples("data/subset/val_mini.json")
device       = "cuda" if torch.cuda.is_available() else "cpu"


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def make_stages(n_classes: int) -> dict:
    """n_classes개를 4개씩 묶어 stage dict 반환"""
    n_stages = n_classes // 4
    return {s: list(range((s-1)*4, s*4)) for s in range(1, n_stages+1)}


def filter_by(samples, class_ids):
    return [s for s in samples if s["class_id"] in class_ids]


def make_model_nc(n_classes: int) -> GRUDetector:
    return GRUDetector(
        feature_dim=FEATURE_DIM,
        hidden_dim=256,
        num_classes=n_classes,
    ).to(device)


@torch.no_grad()
def eval_stage(model, val_samples, feat_dir, device, class_ids):
    model.eval()
    correct = total = 0
    for s in val_samples:
        if s["class_id"] not in class_ids:
            continue
        p = feat_dir / f"{s['id']}.npy"
        if not p.exists():
            continue
        x = torch.tensor(
            np.load(p), dtype=torch.float32
        ).unsqueeze(0).to(device)
        logits, _ = model(x, None)
        pred = class_ids[logits[0, class_ids].argmax().item()]
        total   += 1
        correct += int(pred == s["class_id"])
    return correct / total if total else 0.0


def run_one(method: str, seed: int, n_classes: int, mem_per_stage: int = 30) -> dict:
    set_seed(seed)
    stages     = make_stages(n_classes)
    # n_classes에 맞는 샘플만 사용
    train_all  = filter_by(train_all_32, list(range(n_classes)))
    val_all    = filter_by(val_all_32,   list(range(n_classes)))

    model     = make_model_nc(n_classes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem       = AGEM(mem_per_stage=mem_per_stage) if method == "agem" else None

    acc_table = {}

    for stage_id, class_ids in stages.items():
        stage_train = filter_by(train_all, class_ids)

        for _ in range(NUM_EPOCHS):
            if gem is not None:
                gem.precompute_ref(model, device)
            train_epoch(
                model, stage_train, TRAIN_FEAT_DIR,
                criterion, optimizer, device,
                orth=gem,
            )

        acc_table[stage_id] = {
            sid: eval_stage(model, val_all, VAL_FEAT_DIR, device, cids)
            for sid, cids in stages.items()
        }

        if gem is not None:
            gem.add_stage(stage_train, TRAIN_FEAT_DIR)

    last = max(stages)
    avg_acc = float(np.mean([acc_table[last][sid] for sid in stages]))
    s1_drop = acc_table[1][1] - acc_table[last][1]
    s2_drop = acc_table[2][2] - acc_table[last][2]
    return {"avg_acc": avg_acc, "s1_drop": s1_drop, "s2_drop": s2_drop}


# ══════════════════════════════════════════════════════════════════════════════
# ① Multi-seed × N_classes sweep
# ══════════════════════════════════════════════════════════════════════════════
CLASS_SIZES = [16, 24, 32]

all_records = {}   # n_classes → method → [result_dict, ...]

for n_classes in CLASS_SIZES:
    print(f"\n{'='*60}")
    print(f"  N_CLASSES = {n_classes}  ({n_classes//4} stages × 4 classes)")
    print(f"{'='*60}")
    records = {"baseline": [], "agem": []}

    for seed in SEEDS:
        print(f"\n  seed {seed}", flush=True)
        for method in ["baseline", "agem"]:
            r = run_one(method, seed, n_classes)
            records[method].append(r)
            print(f"    [{method:8s}]  avg_acc={r['avg_acc']:.3f}  "
                  f"s1_drop={r['s1_drop']:+.3f}  s2_drop={r['s2_drop']:+.3f}",
                  flush=True)

    all_records[n_classes] = records

    print(f"\n  {'Method':<12}  {'Avg Acc':>8}  {'±':>5}  {'S1 Drop':>8}  {'±':>5}")
    print("  " + "-"*48)
    for method, label in [("baseline", "Baseline"), ("agem", "A-GEM")]:
        accs  = [r["avg_acc"]  for r in records[method]]
        drops = [r["s1_drop"]  for r in records[method]]
        print(f"  {label:<12}  {np.mean(accs):>8.3f}  {np.std(accs):>5.3f}"
              f"  {np.mean(drops):>+8.3f}  {np.std(drops):>5.3f}")

# ── 최종 요약 테이블 ──────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("  Summary: Baseline vs A-GEM across class sizes")
print(f"{'='*60}")
print(f"  {'N':>4}  {'BL Avg':>8}  {'GEM Avg':>8}  {'Δ Avg':>7}  "
      f"{'BL S1↓':>8}  {'GEM S1↓':>8}  {'Δ S1↓':>7}")
print("  " + "-"*62)

for n_classes in CLASS_SIZES:
    r = all_records[n_classes]
    bl_avg  = np.mean([x["avg_acc"]  for x in r["baseline"]])
    gem_avg = np.mean([x["avg_acc"]  for x in r["agem"]])
    bl_d1   = np.mean([x["s1_drop"]  for x in r["baseline"]])
    gem_d1  = np.mean([x["s1_drop"]  for x in r["agem"]])
    print(f"  {n_classes:>4}  {bl_avg:>8.3f}  {gem_avg:>8.3f}  {gem_avg-bl_avg:>+7.3f}  "
          f"  {bl_d1:>+7.3f}   {gem_d1:>+7.3f}  {bl_d1-gem_d1:>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# ② Memory Ablation (32클래스, seed=0)
# ══════════════════════════════════════════════════════════════════════════════
MEM_SIZES = [10, 20, 30, 50]

print(f"\n\n{'='*60}")
print("  Memory Ablation (A-GEM, 32 classes, seed=0)")
print(f"{'='*60}")
print(f"  {'mem':>5}  {'Avg Acc':>8}  {'S1 Drop':>8}  {'S2 Drop':>8}")
print("  " + "-"*38)

baseline_32 = all_records[32]["baseline"][0]  # seed=0
for mem in MEM_SIZES:
    r = run_one("agem", seed=0, n_classes=32, mem_per_stage=mem)
    print(f"  {mem:>5}  {r['avg_acc']:>8.3f}  {r['s1_drop']:>+8.3f}  {r['s2_drop']:>+8.3f}",
          flush=True)

print(f"  {'(BL)':>5}  {baseline_32['avg_acc']:>8.3f}  "
      f"{baseline_32['s1_drop']:>+8.3f}  {baseline_32['s2_drop']:>+8.3f}  ← Baseline seed=0")

