"""
Tests for anomaly / novel-action (OOD) detection (src/anomaly/detector.py).

All tests are deterministic (fixed seeds) and use SYNTHETIC feature windows.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# allow `python3 -m pytest` from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.gru_detector import GRUDetector  # noqa: E402
from src.anomaly.detector import (  # noqa: E402
    max_softmax_prob,
    predictive_entropy,
    energy_score,
    anomaly_score,
    AnomalyScorer,
)

FEATURE_DIM = 512
N_FRAMES = 16
N_CLASSES = 6
SEED = 1234


# ──────────────────────────────────────────────────────────────────────────────
# fixtures: a small self-contained GRU trained on synthetic in-dist classes
# ──────────────────────────────────────────────────────────────────────────────
def _set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_prototypes(rng, n_classes=N_CLASSES, dim=FEATURE_DIM):
    """Distinct, well-separated unit-norm class prototype directions."""
    protos = rng.standard_normal((n_classes, dim)).astype(np.float32)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    return protos


def _in_dist_window(rng, protos, cls, noise=0.12):
    p = protos[cls]
    w = p[None, :] + noise * rng.standard_normal((N_FRAMES, p.shape[0])).astype(np.float32)
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return w.astype(np.float32)


def _ood_window(rng, dim=FEATURE_DIM, kind="random"):
    if kind == "random":                       # unnormalised gaussian noise
        return rng.standard_normal((N_FRAMES, dim)).astype(np.float32)
    if kind == "uniform":                      # uniform garbage
        return rng.uniform(-1.0, 1.0, (N_FRAMES, dim)).astype(np.float32)
    if kind == "shift":                        # large mean shift
        return (rng.standard_normal((N_FRAMES, dim)).astype(np.float32) + 4.0)
    raise ValueError(kind)


@pytest.fixture(scope="module")
def trained_model():
    """Train a tiny GRUDetector on synthetic in-dist classes (CPU, fast)."""
    _set_seed()
    rng = np.random.default_rng(SEED)
    protos = _make_prototypes(rng)

    model = GRUDetector(
        feature_dim=FEATURE_DIM, hidden_dim=64, num_classes=N_CLASSES
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(120):
        # mini-batch of one window per class
        xs, ys = [], []
        for c in range(N_CLASSES):
            xs.append(_in_dist_window(rng, protos, c))
            ys.append(c)
        x = torch.tensor(np.stack(xs), dtype=torch.float32)   # (B,T,F)
        y = torch.tensor(ys, dtype=torch.long)
        opt.zero_grad()
        logits, _ = model(x, None)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
    model.eval()
    return model, protos


# ──────────────────────────────────────────────────────────────────────────────
# raw scoring-function unit tests
# ──────────────────────────────────────────────────────────────────────────────
def test_scoring_convention_confident_vs_uniform():
    """Higher score == more OOD, for all three metrics."""
    confident = torch.tensor([10.0, 0.0, 0.0, 0.0])     # peaky => in-dist-like
    uniform = torch.tensor([0.0, 0.0, 0.0, 0.0])        # flat   => OOD-like

    assert max_softmax_prob(confident) < max_softmax_prob(uniform)
    assert predictive_entropy(confident) < predictive_entropy(uniform)
    # peaky logits have larger logsumexp => lower (more negative) energy
    assert energy_score(confident) < energy_score(uniform)


def test_scoring_batched_shapes():
    logits = torch.randn(7, 12)
    for fn in (max_softmax_prob, predictive_entropy, energy_score):
        out = fn(logits)
        assert isinstance(out, np.ndarray)
        assert out.shape == (7,)
    # single vector -> python float
    assert isinstance(max_softmax_prob(torch.randn(12)), float)
    assert isinstance(anomaly_score(torch.randn(12), metric="energy"), float)


def test_energy_temperature_matches_formula():
    logits = torch.tensor([1.0, 2.0, 3.0])
    T = 2.0
    expected = float(-T * torch.logsumexp(logits / T, dim=-1))
    assert energy_score(logits, T=T) == pytest.approx(expected, rel=1e-6)


def test_accepts_numpy_and_list():
    arr = np.array([3.0, 1.0, 0.5], dtype=np.float32)
    assert energy_score(arr) == pytest.approx(energy_score(torch.tensor(arr)), rel=1e-6)
    assert max_softmax_prob([3.0, 1.0, 0.5]) == pytest.approx(
        max_softmax_prob(torch.tensor([3.0, 1.0, 0.5])), rel=1e-6
    )


# ──────────────────────────────────────────────────────────────────────────────
# separation: in-dist LOW, OOD HIGH (mean OOD > mean in-dist, all 3 metrics)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("metric", ["energy", "entropy", "msp"])
def test_separation_indist_low_ood_high(trained_model, metric):
    model, protos = trained_model
    rng = np.random.default_rng(SEED + 7)
    scorer = AnomalyScorer(metric=metric, smoothing=None)

    in_scores, ood_scores = [], []
    for _ in range(40):
        c = rng.integers(N_CLASSES)
        in_scores.append(
            scorer.score_window(model, _in_dist_window(rng, protos, c), smooth=False)["raw_score"]
        )
    for kind in ("random", "uniform", "shift"):
        for _ in range(20):
            ood_scores.append(
                scorer.score_window(model, _ood_window(rng, kind=kind), smooth=False)["raw_score"]
            )

    in_mean = float(np.mean(in_scores))
    ood_mean = float(np.mean(ood_scores))
    assert ood_mean > in_mean, (
        f"metric={metric}: ood_mean={ood_mean:.4f} not > in_mean={in_mean:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# calibration: ~target_fpr of in-dist flagged, high fraction of OOD flagged
# ──────────────────────────────────────────────────────────────────────────────
def test_calibration_target_fpr(trained_model):
    model, protos = trained_model
    rng = np.random.default_rng(SEED + 99)
    scorer = AnomalyScorer(metric="energy", smoothing=None)

    in_scores = np.array([
        scorer.score_window(model, _in_dist_window(rng, protos, rng.integers(N_CLASSES)),
                            smooth=False)["raw_score"]
        for _ in range(300)
    ])
    thr = scorer.calibrate(in_scores, target_fpr=0.05)

    # held-out in-dist false-positive rate near 5%
    held = np.array([
        scorer.score_window(model, _in_dist_window(rng, protos, rng.integers(N_CLASSES)),
                            smooth=False)["raw_score"]
        for _ in range(300)
    ])
    in_fpr = float(np.mean(held >= thr))
    assert in_fpr == pytest.approx(0.05, abs=0.06)

    # OOD detection rate should be high
    ood = np.array([
        scorer.score_window(model, _ood_window(rng, kind=k), smooth=False)["raw_score"]
        for k in ("random", "uniform", "shift") for _ in range(60)
    ])
    ood_rate = float(np.mean(ood >= thr))
    assert ood_rate > 0.7, f"OOD detection rate too low: {ood_rate:.3f}"


def test_calibrate_quantile_rule():
    """calibrate sets the threshold at the (1 - fpr) quantile of in-dist scores."""
    scorer = AnomalyScorer(metric="energy", smoothing=None)
    scores = np.linspace(0.0, 1.0, 1001)
    thr = scorer.calibrate(scores, target_fpr=0.10)
    assert thr == pytest.approx(np.quantile(scores, 0.90), abs=1e-6)
    assert scorer.threshold == thr


def test_calibrate_validates_inputs():
    scorer = AnomalyScorer()
    with pytest.raises(ValueError):
        scorer.calibrate([], target_fpr=0.05)
    with pytest.raises(ValueError):
        scorer.calibrate([0.1, 0.2], target_fpr=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# integration with the real A-GEM checkpoint
# ──────────────────────────────────────────────────────────────────────────────
def _load_real_agem():
    ckpt_path = ROOT / "checkpoints" / "agem_48cls.pt"
    if not ckpt_path.exists():
        pytest.skip("agem_48cls.pt not present")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GRUDetector(
        feature_dim=ckpt.get("feature_dim", 512),
        hidden_dim=ckpt.get("hidden_dim", 256),
        num_classes=ckpt.get("n_classes", 48),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def test_integration_real_checkpoint_score_window():
    model, ckpt = _load_real_agem()
    fdim = ckpt.get("feature_dim", 512)
    n_classes = ckpt.get("n_classes", 48)
    rng = np.random.default_rng(SEED)

    scorer = AnomalyScorer(metric="energy")
    w = rng.standard_normal((N_FRAMES, fdim)).astype(np.float32)
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8

    out = scorer.score_window(model, w)
    assert set(out) == {"score", "raw_score", "is_anomaly", "top1",
                        "top1_prob", "threshold"}
    assert np.isfinite(out["score"]) and np.isfinite(out["raw_score"])
    assert 0 <= out["top1"] < n_classes
    assert 0.0 <= out["top1_prob"] <= 1.0
    assert isinstance(out["is_anomaly"], bool)
    # no threshold set -> not flagged
    assert out["is_anomaly"] is False and out["threshold"] is None


def test_integration_accepts_3d_and_tensor_windows():
    model, ckpt = _load_real_agem()
    fdim = ckpt.get("feature_dim", 512)
    scorer = AnomalyScorer(metric="entropy")

    w2d = np.random.default_rng(0).standard_normal((N_FRAMES, fdim)).astype(np.float32)
    w3d = w2d[None, :, :]                          # (1, T, F)
    s_np = scorer.score_window(model, w2d, smooth=False)["raw_score"]
    s_3d = scorer.score_window(model, w3d, smooth=False)["raw_score"]
    s_t = scorer.score_window(model, torch.tensor(w2d), smooth=False)["raw_score"]
    assert s_np == pytest.approx(s_3d, rel=1e-5)
    assert s_np == pytest.approx(s_t, rel=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# temporal smoothing reduces flicker on an alternating stream
# ──────────────────────────────────────────────────────────────────────────────
def test_temporal_smoothing_reduces_flicker(trained_model):
    model, protos = trained_model
    rng = np.random.default_rng(SEED + 3)

    # calibrate a threshold on in-dist scores
    scorer = AnomalyScorer(metric="energy", smoothing="ema", ema_alpha=0.4)
    in_scores = [
        scorer.score_window(model, _in_dist_window(rng, protos, rng.integers(N_CLASSES)),
                            smooth=False)["raw_score"]
        for _ in range(200)
    ]
    scorer.calibrate(in_scores, target_fpr=0.05)

    # alternating stream: in-dist, ood, in-dist, ood, ...
    windows = []
    for i in range(40):
        if i % 2 == 0:
            windows.append(_in_dist_window(rng, protos, rng.integers(N_CLASSES)))
        else:
            windows.append(_ood_window(rng, kind="random"))

    # count flag transitions WITHOUT smoothing
    raw_scorer = AnomalyScorer(metric="energy", smoothing=None,
                              threshold=scorer.threshold)
    raw_flags = [raw_scorer.score_window(model, w, smooth=False)["is_anomaly"]
                 for w in windows]
    raw_transitions = sum(a != b for a, b in zip(raw_flags, raw_flags[1:]))

    # count flag transitions WITH EMA smoothing across the stream
    scorer.reset_stream()
    sm_flags = [r["is_anomaly"] for r in scorer.score_stream(model, windows)]
    sm_transitions = sum(a != b for a, b in zip(sm_flags, sm_flags[1:]))

    # smoothing should not increase flicker, and should reduce it on a noisy
    # alternating stream
    assert sm_transitions <= raw_transitions
    assert sm_transitions < raw_transitions or raw_transitions == 0
