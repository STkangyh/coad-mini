"""
GRU capacity + hard-memory ablation (48 classes x 8 stages)
============================================================

Recommended quick sweep:
  1. hidden_dim: 256 -> 512
  2. 2-layer GRU + dropout=0.2
  3. balanced_hard memory

Examples:
  python experiments/run_capacity_ablation.py
  python experiments/run_capacity_ablation.py --seeds 0 1 2 3 4 --epochs 15
  python experiments/run_capacity_ablation.py --configs best_current_h256 --feature-dir data/features_openclip_l14 --feature-dim 768
"""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.gru_detector import GRUDetector
from src.models.gru_attention import GRUAttentionDetector
from src.models.ssm_detector import SSMDetector
from src.trainer import load_samples, set_seed, train_epoch
from src.utils.gem import AGEM


DEFAULT_FEATURE_DIM = 512
DEFAULT_FEATURE_DIR = Path("data/features")
N_CLASSES = 48
CLASSES_PER_STAGE = 6
MEM_PER_STAGE = 50
REPLAY_RATIO = 0.25

train_all = load_samples("data/subset/train_mini.json")
val_all = load_samples("data/subset/val_mini.json")
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    hidden_dim: int
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    selection: str = "balanced"
    arch: str = "gru"          # "gru" -> GRUDetector | "attn" -> GRUAttentionDetector | "ssm" -> SSMDetector
    num_heads: int = 4         # only used when arch == "attn"


CONFIGS = [
    ExperimentConfig("best_current_h256", hidden_dim=256),
    ExperimentConfig("h512", hidden_dim=512),
    ExperimentConfig("h512_2layer", hidden_dim=512, num_layers=2, dropout=0.2),
    ExperimentConfig(
        "h512_2layer_balanced_hard",
        hidden_dim=512,
        num_layers=2,
        dropout=0.2,
        selection="balanced_hard",
    ),
    ExperimentConfig("attn_h256", hidden_dim=256, arch="attn"),
    ExperimentConfig("attn_h512", hidden_dim=512, arch="attn"),
    ExperimentConfig("ssm_h256", hidden_dim=256, arch="ssm", num_layers=2),
    ExperimentConfig("ssm_h512", hidden_dim=512, arch="ssm", num_layers=2),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--configs", nargs="+", default=[c.name for c in CONFIGS])
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bidirectional", action="store_true")
    return parser.parse_args()


def make_stages() -> dict[int, list[int]]:
    return {
        s: list(range((s - 1) * CLASSES_PER_STAGE, s * CLASSES_PER_STAGE))
        for s in range(1, N_CLASSES // CLASSES_PER_STAGE + 1)
    }


STAGES = make_stages()


def filter_by(samples, class_ids):
    return [s for s in samples if s["class_id"] in class_ids]


def make_model(config: ExperimentConfig, feature_dim: int) -> nn.Module:
    if config.arch == "attn":
        return GRUAttentionDetector(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            num_classes=N_CLASSES,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            num_heads=config.num_heads,
        ).to(device)
    if config.arch == "ssm":
        return SSMDetector(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            num_classes=N_CLASSES,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        ).to(device)
    return GRUDetector(
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=N_CLASSES,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(device)


@torch.no_grad()
def eval_stage(model, class_ids, feat_dir: Path):
    model.eval()
    correct = total = 0
    for sample in val_all:
        if sample["class_id"] not in class_ids:
            continue
        feat_path = feat_dir / "val" / f"{sample['id']}.npy"
        if not feat_path.exists():
            continue
        x = torch.tensor(np.load(feat_path), dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(x, None)
        pred = class_ids[logits[0, class_ids].argmax().item()]
        total += 1
        correct += int(pred == sample["class_id"])
    return correct / total if total else 0.0


def run_one(
    seed: int,
    config: ExperimentConfig,
    epochs: int,
    feature_dir: Path,
    feature_dim: int,
    lr: float,
) -> dict:
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
        "s2_drop": acc_table[2][2] - acc_table[last][2],
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
    requested = set(args.configs)
    configs = [c for c in CONFIGS if c.name in requested]
    if len(configs) != len(requested):
        missing = sorted(requested - {c.name for c in CONFIGS})
        raise ValueError(f"Unknown config(s): {missing}")

    if args.bidirectional:
        configs = [
            ExperimentConfig(
                name=f"{c.name}_bigru",
                hidden_dim=c.hidden_dim,
                num_layers=c.num_layers,
                dropout=c.dropout,
                bidirectional=True,
                selection=c.selection,
                arch=c.arch,
                num_heads=c.num_heads,
            )
            for c in configs
        ]

    print(f"device: {device}")
    print(f"seeds={args.seeds} epochs={args.epochs}")
    print(f"feature_dir={args.feature_dir} feature_dim={args.feature_dim} lr={args.lr}")
    print(f"memory: selection varies, mem={MEM_PER_STAGE}, replay_ratio={REPLAY_RATIO}")

    results = {}
    for config in configs:
        print(f"\n{'=' * 72}")
        print(
            f"  {config.name}: hidden={config.hidden_dim}, layers={config.num_layers}, "
            f"dropout={config.dropout}, bidirectional={config.bidirectional}, "
            f"selection={config.selection}"
        )
        print(f"{'=' * 72}")
        records = []
        for seed in args.seeds:
            print(f"  seed {seed} ...", flush=True)
            result = run_one(
                seed=seed,
                config=config,
                epochs=args.epochs,
                feature_dir=args.feature_dir,
                feature_dim=args.feature_dim,
                lr=args.lr,
            )
            records.append(result)
            print(
                f"    avg_acc={result['avg_acc']:.3f}  "
                f"s1_drop={result['s1_drop']:+.3f}  "
                f"s2_drop={result['s2_drop']:+.3f}",
                flush=True,
            )
        results[config.name] = summarise(records)

    print(f"\n\n{'=' * 72}")
    print("  Capacity Ablation Summary")
    print(f"{'=' * 72}")
    print(f"  {'Config':<30}  {'Avg Acc':>8}  {'+/-':>6}  {'S1 Drop':>8}  {'+/-':>6}")
    print("  " + "-" * 66)
    for name, summary in results.items():
        print(
            f"  {name:<30}  {summary['avg_acc_mean']:>8.3f}  "
            f"{summary['avg_acc_std']:>6.3f}  "
            f"{summary['s1_drop_mean']:>+8.3f}  "
            f"{summary['s1_drop_std']:>6.3f}"
        )


if __name__ == "__main__":
    main()
