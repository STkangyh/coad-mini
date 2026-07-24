"""
Base-heavy (TCD-style) split re-validation on our own 48-class SSv2 subset.

Context: TCD (Park et al., ICCV 2021) and its successors (CSTA'25, STSP ECCV'24,
ESSENTIAL ICCV'25) built the standard SSv2-CIL benchmark: 174 classes, an 84-class
BASE session (~48% of classes) followed by several SMALL increments (5 or 10
classes each). TCD found NME (a prototype/mean-embedding classifier -- similar in
spirit to our FeCAM/analytic heads) UNDERPERFORMS a plain CNN classifier
specifically on SSv2, because naive frame-averaging destroys the temporal signal
SSv2 needs. This directly threatens our own finding (FeCAM > GRU+A-GEM on our
48-class subset): maybe it only holds because our subset/protocol is "easy",
and a base-heavy structure would flip it, as TCD observed.

We CANNOT reproduce TCD's literal 174-class/84-base split here (no raw-video
access to the 126 classes outside our curated 48-class subset this session --
see reports/mobileclip_result.md for the same blocker). Instead we test the
STRUCTURAL variable directly: reshape our own 48 classes (same CLIP B/32
features used everywhere else in this repo) into base-heavy sessions --
    moderate:   24-class base + four 6-class increments   (base:increment = 4x)
    aggressive: 36-class base + two 6-class increments    (base:increment = 6x)
(TCD's own ratio is 84:10 = 8.4x or 84:5 = 16.8x -- aggressive is in that range,
 moderate is a gentler stress test)

and evaluate TRUE class-incremental accuracy (no task ID; argmax over every
class seen so far) at each session boundary, for both a mean-pool analytic head
(FeCAM -- the TCD-era "NME-style" method) and our temporal GRU+A-GEM, to see
whether the ordering flips the way TCD's NME-vs-CNN finding would predict.

Run: python3 dev/run_base_heavy_split.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn

from experiments.run_capacity_ablation import (
    ExperimentConfig, MEM_PER_STAGE, REPLAY_RATIO, N_CLASSES,
    device, filter_by, make_model, train_all, val_all,
)
from src.models.fecam_head import FeCAMHead
from src.trainer import set_seed, train_epoch
from src.utils.gem import AGEM

FEATURE_DIR = Path("data/features")   # CLIP B/32, already extracted for all 48 classes
FEATURE_DIM = 512
EPOCHS = 15
SEEDS = [0, 1, 2]
MAIN_CONFIG = ExperimentConfig("best_current_h256", hidden_dim=256)

# Our 48 classes are already grouped into 8 semantic stages of 6 (unchanged,
# curriculum-curated order used everywhere else in this repo).
STAGE_CLASSES = {s: list(range((s - 1) * 6, s * 6)) for s in range(1, 9)}

SPLITS = {
    "moderate": {  # base:increment = 4x  (24-class base + four 6-class increments)
        "base": sum((STAGE_CLASSES[s] for s in (1, 2, 3, 4)), []),
        "increments": [STAGE_CLASSES[s] for s in (5, 6, 7, 8)],
    },
    "aggressive": {  # base:increment = 6x  (36-class base + two 6-class increments)
        "base": sum((STAGE_CLASSES[s] for s in (1, 2, 3, 4, 5, 6)), []),
        "increments": [STAGE_CLASSES[s] for s in (7, 8)],
    },
}


def sessions_for(split):
    return [split["base"]] + list(split["increments"])


# ── true class-IL eval: argmax over every class seen so far ───────────────────
@torch.no_grad()
def eval_gru_classIL(model, seen_classes):
    model.eval()
    correct = total = 0
    seen = set(seen_classes)
    for s in val_all:
        if s["class_id"] not in seen:
            continue
        p = FEATURE_DIR / "val" / f"{s['id']}.npy"
        if not p.exists():
            continue
        x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0)
        logits, _ = model(x, None)
        mask = torch.full((N_CLASSES,), -1e18)
        mask[list(seen)] = 0.0
        pred = int((logits[0] + mask).argmax().item())
        total += 1
        correct += int(pred == s["class_id"])
    return correct / total if total else 0.0


def load_val_features():
    X, y = [], []
    for s in val_all:
        p = FEATURE_DIR / "val" / f"{s['id']}.npy"
        if p.exists():
            X.append(np.load(p).mean(axis=0))
            y.append(s["class_id"])
    return np.stack(X).astype(np.float64), np.array(y)


def eval_fecam_classIL(head: FeCAMHead, seen_classes, Xva, yva):
    seen = set(seen_classes)
    mask = np.isin(yva, list(seen))
    if mask.sum() == 0:
        return 0.0
    s = head.scores(Xva[mask])
    not_seen = [c for c in range(N_CLASSES) if c not in seen]
    s[:, not_seen] = -1e18
    return float((s.argmax(axis=1) == yva[mask]).mean())


# ── GRU + A-GEM over a session list ───────────────────────────────────────────
def run_gru_agem(sessions, seed):
    set_seed(seed)
    model = make_model(MAIN_CONFIG, FEATURE_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem = AGEM(mem_per_stage=MEM_PER_STAGE, selection=MAIN_CONFIG.selection,
               replay_ratio=REPLAY_RATIO)

    seen: list[int] = []
    accs = []
    t0 = time.perf_counter()
    for cids in sessions:
        session_train = filter_by(train_all, cids)
        for _ in range(EPOCHS):
            gem.precompute_ref(model, device)
            train_epoch(model, session_train, FEATURE_DIR / "train",
                        criterion, optimizer, device, orth=gem)
        seen.extend(cids)
        accs.append(eval_gru_classIL(model, seen))
        gem.add_stage(session_train, FEATURE_DIR / "train", model=model, device=device)
    return {"avg_inc": float(np.mean(accs)), "last": accs[-1],
            "per_session": accs, "train_s": time.perf_counter() - t0}


# ── FeCAM over a session list ──────────────────────────────────────────────────
def run_fecam(sessions, Xtr_by_class, Xva, yva):
    head = FeCAMHead(feature_dim=FEATURE_DIM, max_classes=N_CLASSES)
    seen: list[int] = []
    accs = []
    t0 = time.perf_counter()
    for cids in sessions:
        X = np.concatenate([Xtr_by_class[c] for c in cids])
        y = np.concatenate([np.full(len(Xtr_by_class[c]), c) for c in cids])
        head.observe(X, y)
        seen.extend(cids)
        accs.append(eval_fecam_classIL(head, seen, Xva, yva))
    return {"avg_inc": float(np.mean(accs)), "last": accs[-1],
            "per_session": accs, "train_s": time.perf_counter() - t0}


def load_train_by_class():
    by_class = {c: [] for c in range(N_CLASSES)}
    for s in train_all:
        p = FEATURE_DIR / "train" / f"{s['id']}.npy"
        if p.exists():
            by_class[s["class_id"]].append(np.load(p).mean(axis=0))
    return {c: np.stack(v).astype(np.float64) for c, v in by_class.items() if v}


def main():
    print(f"device: {device}  (GRU+A-GEM training; FeCAM is CPU-only regardless)")
    Xtr_by_class = load_train_by_class()
    Xva, yva = load_val_features()

    print(f"\n{'='*90}")
    print(f"{'split':10s} {'method':16s} {'avg_inc':>8} {'last':>8} {'train_s':>8}  per-session")
    print("=" * 90)

    results = {}
    for split_name, split in SPLITS.items():
        sessions = sessions_for(split)
        sizes = [len(s) for s in sessions]
        print(f"\n[{split_name}] session sizes: {sizes}  "
              f"(base:increment = {sizes[0]//sizes[1]}x)")

        r_fecam = run_fecam(sessions, Xtr_by_class, Xva, yva)
        curve = " ".join(f"{a:.3f}" for a in r_fecam["per_session"])
        print(f"{split_name:10s} {'FeCAM':16s} {r_fecam['avg_inc']:8.3f} "
              f"{r_fecam['last']:8.3f} {r_fecam['train_s']:8.2f}  {curve}", flush=True)

        gru_runs = [run_gru_agem(sessions, seed) for seed in SEEDS]
        avg_inc = np.mean([r["avg_inc"] for r in gru_runs])
        last = np.mean([r["last"] for r in gru_runs])
        train_s = np.mean([r["train_s"] for r in gru_runs])
        curve = " ".join(f"{a:.3f}" for a in
                          np.mean([r["per_session"] for r in gru_runs], axis=0))
        print(f"{split_name:10s} {'GRU+A-GEM (3seed)':16s} {avg_inc:8.3f} "
              f"{last:8.3f} {train_s:8.1f}  {curve}", flush=True)

        results[split_name] = {"fecam": r_fecam, "gru_agem_mean": {
            "avg_inc": float(avg_inc), "last": float(last),
            "per_session": [float(x) for x in np.mean([r["per_session"] for r in gru_runs], axis=0)],
        }, "gru_agem_seeds": gru_runs}

    print("\nALL DONE")
    import json
    Path("reports").mkdir(exist_ok=True)
    with open("reports/base_heavy_split_raw.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
