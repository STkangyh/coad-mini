"""
Baseline vs COAD  ×  N seeds 반복 실험
결과를 mean ± std로 출력
"""

import torch
import torch.nn as nn
import numpy as np

from utils.orthogonal_grad import OrthogonalGradient
from trainer import (
    load_samples, make_model, set_seed,
    train_epoch, evaluate,
    TRAIN_FEAT_DIR, VAL_FEAT_DIR,
)

SEEDS      = [0, 1, 2, 3, 4]
NUM_EPOCHS = 10

train_samples = load_samples("data/subset/train_mini.json")
val_samples   = load_samples("data/subset/val_mini.json")
device        = "cuda" if torch.cuda.is_available() else "cpu"


def run(use_orth: bool, seed: int) -> dict:
    set_seed(seed)
    model     = make_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    orth      = OrthogonalGradient() if use_orth else None

    for _ in range(NUM_EPOCHS):
        train_epoch(
            model, train_samples, TRAIN_FEAT_DIR,
            criterion, optimizer, device, orth=orth,
        )

    vl_loss, vl_acc = evaluate(
        model, val_samples, VAL_FEAT_DIR, criterion, device
    )
    return {"val_loss": vl_loss, "val_acc": vl_acc}


# ── 실행 ────────────────────────────────────────────────────────────────────
results = {"baseline": [], "coad": []}

for seed in SEEDS:
    print(f"seed {seed} ...", flush=True)

    r_base = run(use_orth=False, seed=seed)
    r_coad = run(use_orth=True,  seed=seed)

    results["baseline"].append(r_base)
    results["coad"].append(r_coad)

    print(
        f"  baseline  val_acc={r_base['val_acc']:.4f}"
        f"  coad  val_acc={r_coad['val_acc']:.4f}"
    )

# ── 요약 ─────────────────────────────────────────────────────────────────────
print("\n=== Summary (val_acc) ===")
print(f"{'':8}  {'mean':>6}  {'std':>6}  {'min':>6}  {'max':>6}")

for name, rs in results.items():
    accs = [r["val_acc"] for r in rs]
    print(
        f"{name:8}  "
        f"{np.mean(accs):.4f}  "
        f"{np.std(accs):.4f}  "
        f"{np.min(accs):.4f}  "
        f"{np.max(accs):.4f}"
    )

delta = np.mean([r["val_acc"] for r in results["coad"]]) \
      - np.mean([r["val_acc"] for r in results["baseline"]])
print(f"\nCOAD - Baseline = {delta:+.4f}")
