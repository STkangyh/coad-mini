"""
공통 학습/평가 루틴
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.models.gru_detector import GRUDetector
from config import FEATURE_DIM, NUM_CLASSES

TRAIN_FEAT_DIR = Path("data/features/train")
VAL_FEAT_DIR   = Path("data/features/val")


def load_samples(json_path: str | Path) -> list:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def make_model(device: str) -> GRUDetector:
    return GRUDetector(
        feature_dim=FEATURE_DIM,
        hidden_dim=256,
        num_classes=NUM_CLASSES,
    ).to(device)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 한 epoch 학습 ────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    samples: list,
    feat_dir: Path,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    orth=None,
) -> tuple[float, float]:
    import random
    model.train()
    total_loss = total_acc = count = 0

    shuffled = samples[:]
    random.shuffle(shuffled)

    for sample in shuffled:
        feat_path = feat_dir / f"{sample['id']}.npy"
        if not feat_path.exists():
            continue

        features = np.load(feat_path)                               # (T, 512)
        window   = torch.tensor(features, dtype=torch.float32
                                ).unsqueeze(0).to(device)           # (1,T,512)

        optimizer.zero_grad()
        logits, _ = model(window, None)
        n_cls  = logits.size(-1)                                    # 모델 출력 크기로 자동 결정
        target = torch.zeros(1, n_cls, dtype=torch.float32).to(device)
        target[0, sample["class_id"]] = 1.0
        loss = criterion(logits, target)
        loss.backward()

        if orth is not None:
            orth.apply(model, device)

        optimizer.step()

        # argmax 기반 분류 정확도
        pred_class = logits.argmax(dim=-1).item()
        acc        = int(pred_class == sample["class_id"])

        total_loss += loss.item()
        total_acc  += acc
        count      += 1

    return total_loss / count, total_acc / count


# ── 평가 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    samples: list,
    feat_dir: Path,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_acc = count = 0

    for sample in samples:
        feat_path = feat_dir / f"{sample['id']}.npy"
        if not feat_path.exists():
            continue

        features = np.load(feat_path)
        window   = torch.tensor(features, dtype=torch.float32
                                ).unsqueeze(0).to(device)

        target = torch.zeros(1, NUM_CLASSES, dtype=torch.float32).to(device)
        target[0, sample["class_id"]] = 1.0

        logits, _ = model(window, None)
        loss = criterion(logits, target)

        # argmax 기반 분류 정확도
        pred_class = logits.argmax(dim=-1).item()
        correct    = int(pred_class == sample["class_id"])

        total_loss += loss.item()
        total_acc  += correct
        count      += 1

    return total_loss / count, total_acc / count


# ── val 평가 (클래스별) ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    samples: list,
    feat_dir: Path,
    device: str,
    num_classes: int = NUM_CLASSES,
) -> dict[int, float]:
    model.eval()
    correct = {c: 0 for c in range(num_classes)}
    total   = {c: 0 for c in range(num_classes)}

    for sample in samples:
        feat_path = feat_dir / f"{sample['id']}.npy"
        if not feat_path.exists():
            continue

        features = np.load(feat_path)
        window   = torch.tensor(features, dtype=torch.float32
                                ).unsqueeze(0).to(device)

        logits, _ = model(window, None)
        pred_class = logits.argmax(dim=-1).item()
        true_class = sample["class_id"]

        total[true_class]   += 1
        correct[true_class] += int(pred_class == true_class)

    return {c: correct[c] / total[c] if total[c] > 0 else 0.0
            for c in range(num_classes)}
