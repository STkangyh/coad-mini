"""
GRU + A-GEM per-session true-class-IL curve on SSv2 (our 48-class subset),
uniform 8-stage curriculum order -- the missing series to put alongside
dev/run_ssv2_head_curves.py's NCM/SLDA/FeCAM curve, matching the shape of
dev/run_ucf101_tcd_protocol.py + dev/run_ucf101_gru_agem.py's pairing.

Same "uniform" split used everywhere else in this repo (STAGES 1..8, 6 classes
each, curriculum order) -- NOT the base-heavy splits from
dev/run_base_heavy_split.py, which reshape the sessions to stress-test TCD's
finding. Reuses that script's eval_gru_classIL/run_gru_agem machinery
unchanged, just fed the uniform session list instead of moderate/aggressive.

Sanity check: this is the same protocol whose FINAL accuracy is already on
record as GRU+A-GEM full-48-way 0.105 (reports/cpu_friendly_methods_result.md);
the last point of this curve should reproduce that number.

Run: python3 dev/run_ssv2_gru_curve.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch.nn as nn

sys.path.insert(0, ".")

from experiments.run_capacity_ablation import (  # noqa: E402
    ExperimentConfig, MEM_PER_STAGE, REPLAY_RATIO, N_CLASSES,
    device, filter_by, make_model, train_all, val_all,
)
from src.trainer import set_seed, train_epoch  # noqa: E402
from src.utils.gem import AGEM  # noqa: E402
from src.utils.provenance import save_results  # noqa: E402
from dev.run_base_heavy_split import (  # noqa: E402
    FEATURE_DIR, eval_gru_classIL,
)

EPOCHS = 15
SEEDS = [0, 1, 2]
MAIN_CONFIG = ExperimentConfig("best_current_h256", hidden_dim=256)
CPS = 6
STAGES = {s: list(range((s - 1) * CPS, s * CPS)) for s in range(1, N_CLASSES // CPS + 1)}


def run_gru_agem(sessions, seed):
    set_seed(seed)
    model = make_model(MAIN_CONFIG, 512)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = __import__("torch").optim.Adam(model.parameters(), lr=1e-4)
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
    return {"per_step": accs, "avg_inc": float(np.mean(accs)), "last": accs[-1],
            "train_s": time.perf_counter() - t0}


def main():
    sessions = list(STAGES.values())
    print(f"device: {device} | uniform 8x6 SSv2 curriculum, {EPOCHS} epochs/stage, "
          f"seeds {SEEDS}\n")

    runs = []
    for seed in SEEDS:
        r = run_gru_agem(sessions, seed)
        curve = " ".join(f"{100*a:.1f}" for a in r["per_step"])
        print(f"  seed {seed}: avg_inc={100*r['avg_inc']:.2f} last={100*r['last']:.2f} "
              f"({r['train_s']:.0f}s)  {curve}", flush=True)
        runs.append(r)

    avg_curve = np.mean([r["per_step"] for r in runs], axis=0)
    avg_inc = float(np.mean([r["avg_inc"] for r in runs]))
    last = float(np.mean([r["last"] for r in runs]))
    print(f"\nmean:  avg_inc={100*avg_inc:.2f}  last={100*last:.2f}")
    print("(reference: GRU+A-GEM full-48-way full acc on record = 0.105)")

    out = Path("reports/ssv2_gru_curve_raw.json")
    save_results(out, {
        "per_step_mean": [float(x) for x in avg_curve],
        "avg_inc": avg_inc, "last": last, "epochs": EPOCHS, "runs": runs,
    })
    print(f"raw -> {out}")


if __name__ == "__main__":
    main()
