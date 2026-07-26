"""Tests for the diagonal-SSM temporal model (SSMDetector) — synthetic, deterministic."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.ssm_detector import SSMDetector
from src.models.gru_detector import GRUDetector
from src.utils.gem import AGEM

FEATURE_DIM = 768
NUM_CLASSES = 12


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


# ── interface parity with GRUDetector ─────────────────────────────────────────
def test_forward_returns_tuple_and_logits_shape():
    _seed()
    m = SSMDetector(feature_dim=FEATURE_DIM, hidden_dim=64, num_classes=NUM_CLASSES)
    out = m(torch.randn(1, 16, FEATURE_DIM), None)
    assert isinstance(out, tuple) and len(out) == 2
    logits, h = out
    assert logits.shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_varying_batch_size(batch):
    m = SSMDetector(feature_dim=FEATURE_DIM, hidden_dim=48, num_classes=NUM_CLASSES)
    logits, _ = m(torch.randn(batch, 16, FEATURE_DIM), None)
    assert logits.shape == (batch, NUM_CLASSES)


@pytest.mark.parametrize("T", [1, 4, 16, 32])
def test_varying_sequence_length(T):
    m = SSMDetector(feature_dim=FEATURE_DIM, hidden_dim=48, num_classes=NUM_CLASSES)
    logits, _ = m(torch.randn(2, T, FEATURE_DIM), None)
    assert logits.shape == (2, NUM_CLASSES)


def test_drop_in_compatible_with_gru_detector_signature():
    # same constructor kwargs GRUDetector accepts must work
    m = SSMDetector(feature_dim=FEATURE_DIM, hidden_dim=64, num_classes=NUM_CLASSES,
                    num_layers=2, dropout=0.1, bidirectional=False)
    for attr in ("feature_dim", "hidden_dim", "num_classes", "num_layers"):
        assert hasattr(m, attr)


def test_backward_populates_ssm_and_classifier_grads():
    _seed()
    m = SSMDetector(feature_dim=FEATURE_DIM, hidden_dim=48, num_classes=NUM_CLASSES, num_layers=2)
    logits, _ = m(torch.randn(4, 16, FEATURE_DIM), None)
    logits.sum().backward()
    # classifier grads
    assert m.cls.weight.grad is not None and torch.any(m.cls.weight.grad != 0)
    # SSM recurrence grads (the diagonal state param)
    g = m.blocks[0].ssm.log_a.grad
    assert g is not None and torch.any(g != 0)


# ── AGEM compatibility (mirrors run_one/train_epoch) ──────────────────────────
def _write_synthetic_features(feat_dir, n):
    feat_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n):
        sid = f"vid_{i:03d}"
        np.save(feat_dir / f"{sid}.npy", np.random.randn(16, FEATURE_DIM).astype(np.float32))
        samples.append({"id": sid, "class_id": i % NUM_CLASSES})
    return samples


def test_agem_gradient_projection_cycle(tmp_path):
    _seed()
    device = "cpu"
    feat_dir = tmp_path / "train"
    samples = _write_synthetic_features(feat_dir, n=12)

    model = SSMDetector(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES, hidden_dim=48).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gem = AGEM(mem_per_stage=4, selection="balanced", replay_ratio=1.0)

    gem.add_stage(samples[:6], feat_dir, model=model, device=device)
    gem.precompute_ref(model, device)
    assert gem._g_ref is not None

    model.train()
    for s in samples[6:]:
        feats = np.load(feat_dir / f"{s['id']}.npy")
        window = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        optimizer.zero_grad()
        logits, _ = model(window, None)
        target = torch.zeros(1, logits.size(-1), dtype=torch.float32).to(device)
        target[0, s["class_id"]] = 1.0
        criterion(logits, target).backward()
        gem.apply(model, device)
        optimizer.step()

    for p in model.parameters():
        assert torch.all(torch.isfinite(p)), "non-finite parameter after AGEM step"


# ── harness wiring ────────────────────────────────────────────────────────────
def test_make_model_builds_ssm_and_keeps_gru():
    import experiments.run_capacity_ablation as harness
    ssm = harness.make_model(
        harness.ExperimentConfig("ssm_h256", hidden_dim=256, arch="ssm", num_layers=2), 768)
    assert isinstance(ssm, SSMDetector)
    gru = harness.make_model(harness.ExperimentConfig("best_current_h256", hidden_dim=256), 768)
    assert isinstance(gru, GRUDetector) and not isinstance(gru, SSMDetector)


def test_ssm_configs_registered():
    import experiments.run_capacity_ablation as harness
    names = {c.name for c in harness.CONFIGS}
    assert "ssm_h256" in names
    assert "best_current_h256" in names  # existing preserved
