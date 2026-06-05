"""
A-GEM 심층 분석 실험 (48 classes × 8 stages)
=============================================

① Replay Ratio 실험
   - g_ref 계산 시 메모리의 몇 %를 사용하는가?
   - ratio ∈ {0.10, 0.25, 0.50, 1.00}  (seeds 0-4)

② Memory Selection 실험
   - 메모리를 어떻게 고를 때 더 좋은가?
   - strategy ∈ {"random", "balanced"}  (seeds 0-4)

Baseline(seed 0-4) 결과를 참조값으로 함께 출력.

실험 목적:
  "A-GEM이 왜 좋은가?" — replay 비율과 메모리 구성이 gradient 충돌 방지에
   어떤 영향을 미치는지 분석한다.
"""

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
SEEDS             = [0, 1, 2, 3, 4]
NUM_EPOCHS        = 15
FEATURE_DIM       = 512
N_CLASSES         = 48
CLASSES_PER_STAGE = 6     # 8 stages × 6 = 48
MEM_PER_STAGE     = 50    # 메모리 ablation에서 best였던 값 고정

train_all = load_samples("data/subset/train_mini.json")
val_all   = load_samples("data/subset/val_mini.json")
device    = "cuda" if torch.cuda.is_available() else "cpu"


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def make_stages() -> dict:
    cps = CLASSES_PER_STAGE
    return {s: list(range((s-1)*cps, s*cps)) for s in range(1, N_CLASSES//cps + 1)}

STAGES = make_stages()

def filter_by(samples, class_ids):
    return [s for s in samples if s["class_id"] in class_ids]

def make_model() -> GRUDetector:
    return GRUDetector(
        feature_dim=FEATURE_DIM,
        hidden_dim=256,
        num_classes=N_CLASSES,
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
        x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x, None)
        pred = class_ids[logits[0, class_ids].argmax().item()]
        total   += 1
        correct += int(pred == s["class_id"])
    return correct / total if total else 0.0


def run_one(seed: int, gem: AGEM | None) -> dict:
    """gem=None이면 Baseline"""
    set_seed(seed)
    model     = make_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    acc_table = {}
    for stage_id, class_ids in STAGES.items():
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
            for sid, cids in STAGES.items()
        }

        if gem is not None:
            gem.add_stage(stage_train, TRAIN_FEAT_DIR)

    last    = max(STAGES)
    avg_acc = float(np.mean([acc_table[last][sid] for sid in STAGES]))
    s1_drop = acc_table[1][1] - acc_table[last][1]
    s2_drop = acc_table[2][2] - acc_table[last][2]
    return {"avg_acc": avg_acc, "s1_drop": s1_drop, "s2_drop": s2_drop}


def summarise(records: list[dict]) -> dict:
    accs  = [r["avg_acc"] for r in records]
    drops = [r["s1_drop"] for r in records]
    return {
        "avg_acc_mean": np.mean(accs),
        "avg_acc_std":  np.std(accs),
        "s1_drop_mean": np.mean(drops),
        "s1_drop_std":  np.std(drops),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Baseline (공통 참조값)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  Baseline (no replay)  —  48 classes, 5 seeds")
print("="*62)
bl_records = []
for seed in SEEDS:
    print(f"  seed {seed} ...", flush=True)
    r = run_one(seed, gem=None)
    bl_records.append(r)
    print(f"    avg_acc={r['avg_acc']:.3f}  s1_drop={r['s1_drop']:+.3f}", flush=True)

bl = summarise(bl_records)
print(f"\n  Baseline  avg_acc={bl['avg_acc_mean']:.3f} ±{bl['avg_acc_std']:.3f}  "
      f"s1_drop={bl['s1_drop_mean']:+.3f} ±{bl['s1_drop_std']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# ① Replay Ratio 실험
# ══════════════════════════════════════════════════════════════════════════════
REPLAY_RATIOS = [0.10, 0.25, 0.50, 1.00]

print(f"\n\n{'='*62}")
print("  ① Replay Ratio  (selection=random, mem=50, 5 seeds)")
print(f"{'='*62}")
print(f"  {'ratio':>6}  {'Avg Acc':>8}  {'±':>5}  {'S1 Drop':>8}  {'±':>5}  {'Δ vs BL':>8}")
print("  " + "-"*54)

ratio_results = {}
for ratio in REPLAY_RATIOS:
    records = []
    print(f"\n  ratio={ratio:.2f}", flush=True)
    for seed in SEEDS:
        gem = AGEM(mem_per_stage=MEM_PER_STAGE, selection="random", replay_ratio=ratio)
        r   = run_one(seed, gem)
        records.append(r)
        print(f"    seed {seed}  avg_acc={r['avg_acc']:.3f}  s1_drop={r['s1_drop']:+.3f}",
              flush=True)
    s = summarise(records)
    ratio_results[ratio] = s
    delta = s["avg_acc_mean"] - bl["avg_acc_mean"]
    print(f"  {ratio:>6.2f}  {s['avg_acc_mean']:>8.3f}  {s['avg_acc_std']:>5.3f}"
          f"  {s['s1_drop_mean']:>+8.3f}  {s['s1_drop_std']:>5.3f}  {delta:>+8.3f}")

# 정리 출력
print(f"\n  {'ratio':>6}  {'Avg Acc':>8}  {'±':>5}  {'S1 Drop':>8}  {'±':>5}  {'Δ vs BL':>8}")
print("  " + "-"*54)
print(f"  {'(BL)':>6}  {bl['avg_acc_mean']:>8.3f}  {bl['avg_acc_std']:>5.3f}"
      f"  {bl['s1_drop_mean']:>+8.3f}  {bl['s1_drop_std']:>5.3f}  {'—':>8}")
for ratio in REPLAY_RATIOS:
    s     = ratio_results[ratio]
    delta = s["avg_acc_mean"] - bl["avg_acc_mean"]
    print(f"  {ratio:>6.2f}  {s['avg_acc_mean']:>8.3f}  {s['avg_acc_std']:>5.3f}"
          f"  {s['s1_drop_mean']:>+8.3f}  {s['s1_drop_std']:>5.3f}  {delta:>+8.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# ② Memory Selection 실험
# ══════════════════════════════════════════════════════════════════════════════
STRATEGIES = ["random", "balanced"]

print(f"\n\n{'='*62}")
print("  ② Memory Selection  (replay_ratio=1.0, mem=50, 5 seeds)")
print(f"{'='*62}")

sel_results = {}
for strategy in STRATEGIES:
    records = []
    print(f"\n  strategy={strategy}", flush=True)
    for seed in SEEDS:
        gem = AGEM(mem_per_stage=MEM_PER_STAGE, selection=strategy, replay_ratio=1.0)
        r   = run_one(seed, gem)
        records.append(r)
        print(f"    seed {seed}  avg_acc={r['avg_acc']:.3f}  s1_drop={r['s1_drop']:+.3f}",
              flush=True)
    sel_results[strategy] = summarise(records)

# 정리 출력
print(f"\n  {'strategy':>10}  {'Avg Acc':>8}  {'±':>5}  {'S1 Drop':>8}  {'±':>5}  {'Δ vs BL':>8}")
print("  " + "-"*58)
print(f"  {'(BL)':>10}  {bl['avg_acc_mean']:>8.3f}  {bl['avg_acc_std']:>5.3f}"
      f"  {bl['s1_drop_mean']:>+8.3f}  {bl['s1_drop_std']:>5.3f}  {'—':>8}")
for strategy in STRATEGIES:
    s     = sel_results[strategy]
    delta = s["avg_acc_mean"] - bl["avg_acc_mean"]
    print(f"  {strategy:>10}  {s['avg_acc_mean']:>8.3f}  {s['avg_acc_std']:>5.3f}"
          f"  {s['s1_drop_mean']:>+8.3f}  {s['s1_drop_std']:>5.3f}  {delta:>+8.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("  Final Summary — Why does A-GEM work? (48cls × 8stages, 5 seeds)")
print(f"{'='*70}")

print("\n  [1] Replay Ratio: how much of memory is used for g_ref")
print(f"  {'Config':>16}  {'Avg Acc':>8}  {'S1 Drop':>8}  {'Δ vs BL':>8}")
print("  " + "-"*48)
print(f"  {'Baseline':>16}  {bl['avg_acc_mean']:>8.3f}  {bl['s1_drop_mean']:>+8.3f}  {'—':>8}")
for ratio in REPLAY_RATIOS:
    s     = ratio_results[ratio]
    delta = s["avg_acc_mean"] - bl["avg_acc_mean"]
    tag   = "  ← best" if s["avg_acc_mean"] == max(ratio_results[r]["avg_acc_mean"] for r in ratio_results) else ""
    print(f"  {f'ratio={ratio:.2f}':>16}  {s['avg_acc_mean']:>8.3f}  {s['s1_drop_mean']:>+8.3f}  {delta:>+8.3f}{tag}")

print("\n  [2] Memory Selection: random vs class-balanced")
print(f"  {'Config':>16}  {'Avg Acc':>8}  {'S1 Drop':>8}  {'Δ vs BL':>8}")
print("  " + "-"*48)
print(f"  {'Baseline':>16}  {bl['avg_acc_mean']:>8.3f}  {bl['s1_drop_mean']:>+8.3f}  {'—':>8}")
for strategy in STRATEGIES:
    s     = sel_results[strategy]
    delta = s["avg_acc_mean"] - bl["avg_acc_mean"]
    tag   = "  ← best" if s["avg_acc_mean"] == max(sel_results[st]["avg_acc_mean"] for st in sel_results) else ""
    print(f"  {strategy:>16}  {s['avg_acc_mean']:>8.3f}  {s['s1_drop_mean']:>+8.3f}  {delta:>+8.3f}{tag}")
