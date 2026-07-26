"""
Baseline vs A-GEM vs plain ER (Experience Replay) — same stack, same memory budget.
48 classes / 8 stages, 5 seeds, CLIP B/32 features. Task-aware (within-stage 6-way) eval.

ER = same balanced memory as A-GEM (mem=50/stage) but REPLAYED into the loss
(rehearsal), with NO gradient projection. This is the missing rehearsal comparison.

Run: python3 dev/run_er_comparison.py
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, ".")
from src.utils.gem import AGEM
from src.models.gru_detector import GRUDetector
from src.trainer import load_samples, set_seed, train_epoch

SEEDS = [0, 1, 2, 3, 4]
NUM_EPOCHS = 15
FEATURE_DIM = 512
CPS = 6
N_CLASSES = 48
MEM = 50
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


def run_one(method, seed):
    set_seed(seed)
    model = GRUDetector(feature_dim=FEATURE_DIM, hidden_dim=256, num_classes=N_CLASSES).to(device)
    crit = nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem = AGEM(mem_per_stage=MEM, selection="balanced", replay_ratio=0.25) if method in ("agem", "er") else None
    acc = {}
    for sid, cids in stages.items():
        st = filt(train_all, cids)
        replay = [s for (samps, _) in (gem._memory if gem else []) for s in samps] if method == "er" else []
        for _ in range(NUM_EPOCHS):
            if method == "agem":
                gem.precompute_ref(model, device)
                train_epoch(model, st, TRAIN_FEAT, crit, opt, device, orth=gem)
            elif method == "er":
                train_epoch(model, st + replay, TRAIN_FEAT, crit, opt, device, orth=None)
            else:
                train_epoch(model, st, TRAIN_FEAT, crit, opt, device, orth=None)
        acc[sid] = {j: eval_stage(model, c) for j, c in stages.items()}
        if gem is not None:
            gem.add_stage(st, TRAIN_FEAT)
    last = max(stages)
    return {"avg_acc": float(np.mean([acc[last][j] for j in stages])),
            "s1_drop": acc[1][1] - acc[last][1]}


def main():
    methods = ["baseline", "er", "agem"]
    res = {m: [] for m in methods}
    for seed in SEEDS:
        print(f"\nseed {seed}", flush=True)
        for m in methods:
            r = run_one(m, seed); res[m].append(r)
            print(f"  {m:8s} avg_acc={r['avg_acc']:.3f} s1_drop={r['s1_drop']:+.3f}", flush=True)
    print("\n===== SUMMARY (48cls/8stage, 5 seeds, B/32, task-aware 6-way; chance=1/6=0.167) =====")
    print(f"{'method':10s} {'AvgAcc':>8} {'±std':>7} {'S1drop':>8} {'±std':>7}")
    base = res["baseline"]
    for m in methods:
        a = [r["avg_acc"] for r in res[m]]; d = [r["s1_drop"] for r in res[m]]
        print(f"{m:10s} {np.mean(a):8.3f} {np.std(a):7.3f} {np.mean(d):+8.3f} {np.std(d):7.3f}")
    ag = res["agem"]; er = res["er"]
    dacc = np.mean([r["avg_acc"] for r in ag]) - np.mean([r["avg_acc"] for r in base])
    ddrop = np.mean([r["s1_drop"] for r in base]) - np.mean([r["s1_drop"] for r in ag])
    print(f"\nA-GEM vs baseline:  +{dacc*100:.1f}%p AvgAcc,  +{ddrop*100:.1f}%p S1-forgetting-protection")
    dae = np.mean([r["avg_acc"] for r in ag]) - np.mean([r["avg_acc"] for r in er])
    print(f"A-GEM vs plain ER:  {dae*100:+.1f}%p AvgAcc  (positive => A-GEM better)")


if __name__ == "__main__":
    main()
