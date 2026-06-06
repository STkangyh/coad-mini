"""Fast, deterministic tests for GRUAttentionDetector and its harness wiring."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.gru_attention import GRUAttentionDetector
from src.models.gru_detector import GRUDetector
from src.utils.gem import AGEM


FEATURE_DIM = 768
NUM_CLASSES = 48


def _seed(seed: int = 0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── interface ────────────────────────────────────────────────────────────────
def test_forward_returns_tuple_and_logits_shape():
    _seed()
    model = GRUAttentionDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    x = torch.randn(4, 16, FEATURE_DIM)
    out = model(x, None)

    assert isinstance(out, tuple)
    assert len(out) == 2
    logits, h = out
    assert logits.shape == (4, NUM_CLASSES)
    assert torch.is_tensor(h)


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_varying_batch_size(batch):
    _seed()
    model = GRUAttentionDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    x = torch.randn(batch, 16, FEATURE_DIM)
    logits, _ = model(x, None)
    assert logits.shape == (batch, NUM_CLASSES)


@pytest.mark.parametrize("T", [1, 4, 16, 32])
def test_varying_sequence_length(T):
    _seed()
    model = GRUAttentionDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    x = torch.randn(2, T, FEATURE_DIM)
    logits, _ = model(x, None)
    assert logits.shape == (2, NUM_CLASSES)


def test_drop_in_compatible_with_gru_detector_signature():
    """Same call convention as GRUDetector: forward(x, None) -> (logits, h)."""
    _seed()
    x = torch.randn(3, 16, FEATURE_DIM)

    gru = GRUDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    attn = GRUAttentionDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)

    g_logits, _ = gru(x, None)
    a_logits, _ = attn(x, None)
    assert g_logits.shape == a_logits.shape == (3, NUM_CLASSES)


# ── gradient flow ────────────────────────────────────────────────────────────
def test_backward_populates_attention_and_classifier_grads():
    _seed()
    model = GRUAttentionDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    model.train()

    x = torch.randn(4, 16, FEATURE_DIM)
    target = torch.zeros(4, NUM_CLASSES)
    target[range(4), torch.randint(0, NUM_CLASSES, (4,))] = 1.0

    criterion = nn.BCEWithLogitsLoss()
    logits, _ = model(x, None)
    loss = criterion(logits, target)
    loss.backward()

    # Attention params have grads.
    attn_grads = [p.grad for p in model.attn.parameters() if p.requires_grad]
    assert attn_grads, "attention has no parameters with grads"
    assert any(g is not None and torch.any(g != 0) for g in attn_grads)

    # Classifier params have grads.
    assert model.cls.weight.grad is not None
    assert torch.any(model.cls.weight.grad != 0)

    # GRU params have grads too (full path is differentiable).
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.gru.parameters()
    )


# ── AGEM compatibility ───────────────────────────────────────────────────────
def _write_synthetic_features(feat_dir: Path, n: int) -> list:
    """Create synthetic .npy windows + sample dicts mirroring the real pipeline."""
    feat_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n):
        sid = f"vid_{i:03d}"
        feats = np.random.randn(16, FEATURE_DIM).astype(np.float32)
        np.save(feat_dir / f"{sid}.npy", feats)
        samples.append({"id": sid, "class_id": i % NUM_CLASSES})
    return samples


def test_agem_gradient_projection_cycle(tmp_path):
    """Mirror run_one/train_epoch: precompute_ref -> backward -> apply -> step."""
    _seed()
    device = "cpu"
    feat_dir = tmp_path / "train"
    samples = _write_synthetic_features(feat_dir, n=12)

    model = GRUAttentionDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem = AGEM(mem_per_stage=4, selection="balanced", replay_ratio=1.0)

    # Stage 1: store some memory (so g_ref is non-trivial later).
    gem.add_stage(samples[:6], feat_dir, model=model, device=device)

    # Epoch start: precompute reference gradient from memory.
    gem.precompute_ref(model, device)
    assert gem._g_ref is not None  # memory was stored, ref must exist

    # In-memory mini training loop (train_epoch is file-based; this mirrors it).
    model.train()
    for s in samples[6:]:
        feats = np.load(feat_dir / f"{s['id']}.npy")
        window = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        optimizer.zero_grad()
        logits, _ = model(window, None)
        target = torch.zeros(1, logits.size(-1), dtype=torch.float32).to(device)
        target[0, s["class_id"]] = 1.0
        loss = criterion(logits, target)
        loss.backward()
        gem.apply(model, device)   # gradient projection — must run without error
        optimizer.step()

    # Sanity: parameters remained finite after the projected updates.
    for p in model.parameters():
        assert torch.all(torch.isfinite(p)), "non-finite parameter after AGEM step"


# ── harness wiring ───────────────────────────────────────────────────────────
def test_make_model_returns_correct_classes():
    import experiments.run_capacity_ablation as harness

    attn_model = harness.make_model(
        harness.ExperimentConfig("attn_h256", hidden_dim=256, arch="attn"),
        feature_dim=768,
    )
    assert isinstance(attn_model, GRUAttentionDetector)

    gru_model = harness.make_model(
        harness.ExperimentConfig("best_current_h256", hidden_dim=256),
        768,
    )
    assert isinstance(gru_model, GRUDetector)
    assert not isinstance(gru_model, GRUAttentionDetector)


def test_attn_configs_registered():
    import experiments.run_capacity_ablation as harness

    names = {c.name for c in harness.CONFIGS}
    assert "attn_h256" in names
    # Existing configs preserved (backward compatible).
    assert "best_current_h256" in names
    assert "h512_2layer_balanced_hard" in names
