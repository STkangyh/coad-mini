"""Tests for the FeCAM classifier head — synthetic, deterministic, fast."""
import numpy as np
import pytest

from src.models.fecam_head import FeCAMHead

D = 64
RNG = np.random.default_rng(0)


def make_clusters(n_classes=4, n_per=40, spread=0.15):
    protos = RNG.standard_normal((n_classes, D))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    X, y = [], []
    for c in range(n_classes):
        X.append(protos[c] + spread * RNG.standard_normal((n_per, D)))
        y += [c] * n_per
    return np.concatenate(X), np.array(y), protos


def test_fit_and_score_shapes():
    X, y, _ = make_clusters()
    h = FeCAMHead(feature_dim=D, max_classes=16)
    h.observe(X, y)
    S = h.scores(X[:5])
    assert S.shape == (5, 16)
    assert h.n_classes == 4


def test_separable_clusters_high_accuracy():
    X, y, protos = make_clusters()
    h = FeCAMHead(feature_dim=D, max_classes=16)
    h.observe(X, y)
    Xte = np.concatenate([protos[c] + 0.15 * RNG.standard_normal((20, D)) for c in range(4)])
    yte = np.repeat(np.arange(4), 20)
    acc = (h.scores(Xte).argmax(axis=1) == yte).mean()
    assert acc > 0.9, f"accuracy {acc} too low on separable clusters"


def test_unenrolled_classes_never_predicted():
    X, y, _ = make_clusters(n_classes=3)
    h = FeCAMHead(feature_dim=D, max_classes=16)
    h.observe(X, y)
    preds = h.scores(RNG.standard_normal((50, D))).argmax(axis=1)
    assert set(preds).issubset({0, 1, 2})


def test_streaming_observe_matches_batch():
    X, y, _ = make_clusters()
    h1 = FeCAMHead(D, 16); h1.observe(X, y)
    h2 = FeCAMHead(D, 16)
    for i in range(0, len(X), 32):
        h2.observe(X[i:i+32], y[i:i+32])
    np.testing.assert_allclose(h1.means[:4], h2.means[:4], atol=1e-9)


def test_enroll_class_is_isolated_and_instant():
    """Enrolling a new class must not move existing class statistics at all."""
    X, y, protos = make_clusters()
    h = FeCAMHead(D, 16)
    h.observe(X, y)
    means_before = h.means.copy(); cov_before = h._cov_sum.copy()

    new_proto = RNG.standard_normal(D)
    windows = [np.tile(new_proto + 0.1 * RNG.standard_normal(D), (16, 1)) for _ in range(5)]
    n = h.enroll_class(9, windows)

    assert n == 5 and h.counts[9] == 5
    np.testing.assert_array_equal(h.means[:4], means_before[:4])   # untouched
    np.testing.assert_array_equal(h._cov_sum, cov_before)          # cov frozen
    # the new class is predictable
    test_w = np.tile(new_proto, (16, 1))
    top1, prob, _ = h.predict_window(test_w)
    assert top1 == 9 and 0.0 < prob <= 1.0


def test_remove_classes_reverts():
    X, y, _ = make_clusters()
    h = FeCAMHead(D, 16)
    h.observe(X, y)
    h.enroll_class(9, [RNG.standard_normal((16, D))])
    h.remove_classes([9])
    assert h.counts[9] == 0
    preds = h.scores(RNG.standard_normal((30, D))).argmax(axis=1)
    assert 9 not in set(preds)


def test_save_load_roundtrip(tmp_path):
    X, y, _ = make_clusters()
    h = FeCAMHead(D, 16)
    h.observe(X, y)
    p = tmp_path / "head.npz"
    h.save(p)
    h2 = FeCAMHead.load(p)
    Xq = RNG.standard_normal((10, D))
    np.testing.assert_allclose(h.scores(Xq), h2.scores(Xq), atol=1e-9)


def test_predict_window_accepts_1xTxD():
    X, y, _ = make_clusters()
    h = FeCAMHead(D, 16)
    h.observe(X, y)
    w = RNG.standard_normal((1, 16, D))   # torch-style (1, T, D)
    top1, prob, s = h.predict_window(w)
    assert 0 <= top1 < 16 and s.shape == (16,)


def test_scores_match_per_class_mahalanobis_reference():
    """Vectorized scoring must equal the textbook per-class quadratic form.

    scores() expands (x-m)'P(x-m) and caches the m-only term for speed; this
    pins it to a direct, obviously-correct per-class computation so the
    optimization can never silently drift.
    """
    X, y, _ = make_clusters(n_classes=5, n_per=30)
    h = FeCAMHead(feature_dim=D, max_classes=8)
    h.observe(X, y)

    Xq = X[:12]
    got = h.scores(Xq)

    prec, sd, active, _, _, _ = h._precision()
    Xp = h._prep(Xq)
    want = np.full((len(Xq), h.max_classes), -1e18)
    for c in active:
        v = (Xp - h.means[c]) / sd
        want[:, c] = -np.einsum("nd,de,ne->n", v, prec, v)

    np.testing.assert_allclose(got[:, active], want[:, active], rtol=1e-9, atol=1e-9)
    assert np.array_equal(got.argmax(axis=1), want.argmax(axis=1))


def test_scores_cache_invalidated_by_new_enrollment():
    """The cached mean-dependent terms must not survive a statistics update."""
    X, y, _ = make_clusters(n_classes=3, n_per=30)
    h = FeCAMHead(feature_dim=D, max_classes=8)
    h.observe(X, y)
    h.scores(X[:2])                       # warm the cache

    newX = RNG.standard_normal((10, D))
    h.observe(newX, np.full(10, 5))       # enroll an unseen class id
    S = h.scores(newX)

    assert h.n_classes == 4
    assert np.all(S[:, 5] > -1e17), "newly enrolled class must be scorable"
    assert S.argmax(axis=1).tolist() == [5] * 10
