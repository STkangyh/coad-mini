"""
Data-fraction scaling experiment (48 classes x 8 stages)
========================================================

Trains the project MAIN model (GRU hidden=256 + A-GEM balanced, mem=50,
replay=0.25) on increasing fractions of the per-stage training data.

Goal: test whether the feature-backbone effect (CLIP B/32 vs OpenCLIP L/14)
is masked by small data -- i.e. does the B/32 vs L/14 gap widen as the
training data grows, and are the accuracy curves still rising at 100%?

CRITICAL: the per-stage subsample depends ONLY on (seed, fraction, stage_id)
and the sample identity -- NOT on the feature backbone. This guarantees that
the B/32 and L/14 runs with the same (seed, fraction) train on the exact same
videos, so any accuracy difference is attributable to the backbone.

Examples:
  python3 experiments/run_data_scale.py --feature-dir data/features              --feature-dim 512
  python3 experiments/run_data_scale.py --feature-dir data/features_openclip_l14 --feature-dim 768 \
      --fractions 0.25 0.5 1.0 --seeds 0 1 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from experiments.run_capacity_ablation import (
    ExperimentConfig,
    MEM_PER_STAGE,
    REPLAY_RATIO,
    STAGES,
    device,
    eval_stage,
    filter_by,
    make_model,
    train_all,
    val_all,
)
from src.trainer import set_seed, train_epoch
from src.utils.gem import AGEM


# The MAIN project model: GRU hidden=256 + A-GEM balanced.
MAIN_CONFIG = ExperimentConfig("best_current_h256", hidden_dim=256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def subsample_stage(samples: list, fraction: float, seed: int, stage_id: int) -> list:
    """Deterministically subsample a stage's training list to `fraction` of it.

    The selection depends ONLY on (seed, fraction, stage_id) and each sample's
    identity ("id" field). It is INDEPENDENT of the feature backbone / feature
    directory, so two runs with the same (seed, fraction) pick the exact same
    videos regardless of which features are loaded.

    Guarantees at least 1 sample for any non-empty stage. fraction >= 1.0 keeps
    all samples (in their original order).
    """
    if not samples:
        return []
    if fraction >= 1.0:
        return list(samples)

    # Deterministic ordering by sample id so the candidate order never depends
    # on the incoming list order.
    ordered = sorted(samples, key=lambda s: str(s["id"]))

    n_keep = max(1, round(len(ordered) * fraction))

    # RNG seeded purely from (seed, fraction, stage_id) -- no feature-dir input.
    rng_seed = hash((seed, round(float(fraction), 4), stage_id)) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(len(ordered))[:n_keep]
    chosen_ids = {ordered[i]["id"] for i in idx}

    # Return chosen samples in the deterministic ordering.
    return [s for s in ordered if s["id"] in chosen_ids]


def run_one(
    seed: int,
    fraction: float,
    config: ExperimentConfig,
    epochs: int,
    feature_dir: Path,
    feature_dim: int,
    lr: float,
) -> dict:
    """Full 8-stage continual-learning loop on a `fraction` of each stage."""
    set_seed(seed)
    model = make_model(config, feature_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gem = AGEM(
        mem_per_stage=MEM_PER_STAGE,
        selection=config.selection,
        replay_ratio=REPLAY_RATIO,
    )

    acc_table = {}
    for stage_id, class_ids in STAGES.items():
        stage_train = filter_by(train_all, class_ids)
        stage_train = subsample_stage(stage_train, fraction, seed, stage_id)

        for _ in range(epochs):
            gem.precompute_ref(model, device)
            train_epoch(
                model,
                stage_train,
                feature_dir / "train",
                criterion,
                optimizer,
                device,
                orth=gem,
            )

        acc_table[stage_id] = {
            sid: eval_stage(model, cids, feature_dir)
            for sid, cids in STAGES.items()
        }
        gem.add_stage(stage_train, feature_dir / "train", model=model, device=device)

    last = max(STAGES)
    return {
        "avg_acc": float(np.mean([acc_table[last][sid] for sid in STAGES])),
        "s1_drop": acc_table[1][1] - acc_table[last][1],
    }


def summarise(records: list[dict]) -> dict:
    return {
        "avg_acc_mean": float(np.mean([r["avg_acc"] for r in records])),
        "avg_acc_std": float(np.std([r["avg_acc"] for r in records])),
        "s1_drop_mean": float(np.mean([r["s1_drop"] for r in records])),
        "s1_drop_std": float(np.std([r["s1_drop"] for r in records])),
    }


def main():
    args = parse_args()

    print(f"device: {device}")
    print(f"fractions={args.fractions} seeds={args.seeds} epochs={args.epochs}")
    print(f"feature_dir={args.feature_dir} feature_dim={args.feature_dim} lr={args.lr}")
    print(
        f"model: {MAIN_CONFIG.name} hidden={MAIN_CONFIG.hidden_dim} "
        f"selection={MAIN_CONFIG.selection} mem={MEM_PER_STAGE} replay={REPLAY_RATIO}"
    )

    summaries = {}
    for fraction in args.fractions:
        print(f"\n{'=' * 72}")
        print(f"  fraction={fraction}")
        print(f"{'=' * 72}", flush=True)
        records = []
        for seed in args.seeds:
            result = run_one(
                seed=seed,
                fraction=fraction,
                config=MAIN_CONFIG,
                epochs=args.epochs,
                feature_dir=args.feature_dir,
                feature_dim=args.feature_dim,
                lr=args.lr,
            )
            records.append(result)
            # Parseable per-seed line.
            print(
                f"fraction={fraction:g} seed={seed} "
                f"avg_acc={result['avg_acc']:.4f} s1_drop={result['s1_drop']:.4f}",
                flush=True,
            )
        summary = summarise(records)
        summaries[fraction] = summary
        # Parseable per-fraction summary line.
        print(
            f"fraction={fraction:g} "
            f"avg_acc_mean={summary['avg_acc_mean']:.4f} "
            f"avg_acc_std={summary['avg_acc_std']:.4f} "
            f"s1_drop_mean={summary['s1_drop_mean']:.4f} "
            f"s1_drop_std={summary['s1_drop_std']:.4f}",
            flush=True,
        )

    print(f"\n\n{'=' * 72}")
    print("  Data-Scale Summary")
    print(f"{'=' * 72}")
    print(f"  {'Fraction':>8}  {'Avg Acc':>8}  {'+/-':>6}  {'S1 Drop':>8}  {'+/-':>6}")
    print("  " + "-" * 46)
    for fraction, summary in summaries.items():
        print(
            f"  {fraction:>8g}  {summary['avg_acc_mean']:>8.4f}  "
            f"{summary['avg_acc_std']:>6.4f}  "
            f"{summary['s1_drop_mean']:>8.4f}  "
            f"{summary['s1_drop_std']:>6.4f}"
        )


if __name__ == "__main__":
    main()
