"""Tests for the FeCAM classifier head — synthetic, deterministic, fast."""
import numpy as np
import pytest

from src.models.fecam_head import FeCAMHead, pooled_dim

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
    h = FeCAMHead(D, 16, pooling="mean")   # windows are (T, D): mean keeps dim
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
    h = FeCAMHead(D, 16, pooling="mean")
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
    h = FeCAMHead(D, 16, pooling="mean")
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


def test_segment_pooling_is_order_sensitive_and_mean_is_not():
    """The whole point of the default pooling: reversing a window must change it.

    SSv2's confusable pairs ("push left-to-right" vs "right-to-left") are the same
    frames in the opposite order, so an order-blind pooling cannot separate them.
    """
    w = RNG.standard_normal((16, D))
    mean_h = FeCAMHead(D, 4, pooling="mean")
    chunk_h = FeCAMHead(pooled_dim(D), 4)              # chunks4, the default

    np.testing.assert_allclose(mean_h.window_to_embedding(w),
                               mean_h.window_to_embedding(w[::-1]), atol=1e-12)
    assert not np.allclose(chunk_h.window_to_embedding(w),
                           chunk_h.window_to_embedding(w[::-1]))

    # The mechanism is positional: reversal maps segment i to segment k+1-i, so
    # the output is the same blocks in the opposite order. Exact only when the
    # segments divide evenly (T=16 into 4 does; T=16 into 3 would be 5/5/6).
    fwd = chunk_h.window_to_embedding(w)
    rev = chunk_h.window_to_embedding(w[::-1])
    for i in range(4):
        j = 3 - i
        np.testing.assert_allclose(fwd[i * D:(i + 1) * D],
                                   rev[j * D:(j + 1) * D], atol=1e-12)


def test_chunks3_adjdiff_still_flips_sign_under_reversal():
    """The previous default stays supported, with its own mechanism intact."""
    h = FeCAMHead(pooled_dim(D, "chunks3_adjdiff"), 4, pooling="chunks3_adjdiff")
    assert h.feature_dim == 5 * D
    # Equal-length segments are needed for the identity: 15 frames split 5/5/5.
    w = RNG.standard_normal((15, D))
    fwd, rev = h.window_to_embedding(w), h.window_to_embedding(w[::-1])
    np.testing.assert_allclose(fwd[3 * D:4 * D], -rev[4 * D:5 * D], atol=1e-12)
    np.testing.assert_allclose(fwd[4 * D:5 * D], -rev[3 * D:4 * D], atol=1e-12)


def test_default_head_enrolls_and_predicts_from_windows():
    """End-to-end on the deployed default: (T, D) windows in, right class out."""
    h = FeCAMHead(pooled_dim(D), 8)
    assert h.pooling == "chunks4" and h.feature_dim == 4 * D

    base = RNG.standard_normal((16, D))
    for c in range(3):
        proto = base + 3.0 * RNG.standard_normal((16, D))
        h.observe(np.stack([h.window_to_embedding(proto + 0.05 * RNG.standard_normal((16, D)))
                            for _ in range(20)]), np.full(20, c))
        if c == 0:
            target = proto
    top1, prob, _ = h.predict_window(target)
    assert top1 == 0 and 0.0 < prob <= 1.0


def test_short_windows_do_not_produce_nans():
    """A window with fewer frames than segments still yields a usable embedding."""
    h = FeCAMHead(pooled_dim(D), 4)
    for t in (1, 2, 3, 4):
        emb = h.window_to_embedding(RNG.standard_normal((t, D)))
        assert emb.shape == (4 * D,) and np.isfinite(emb).all()


def test_save_load_preserves_pooling_and_is_lossless(tmp_path):
    """The checkpoint carries its pooling, and the float32 triangle costs nothing."""
    h = FeCAMHead(pooled_dim(D), 8)
    X = RNG.standard_normal((120, 4 * D))
    h.observe(X, np.repeat(np.arange(4), 30))
    p = tmp_path / "head.npz"
    h.save(p)
    h2 = FeCAMHead.load(p)

    assert h2.pooling == "chunks4" and h2.feature_dim == 4 * D
    np.testing.assert_array_equal(h2._cov_sum, h2._cov_sum.T)   # symmetry restored
    Xq = RNG.standard_normal((10, 4 * D))
    np.testing.assert_array_equal(h.scores(Xq).argmax(axis=1),
                                  h2.scores(Xq).argmax(axis=1))
    np.testing.assert_allclose(h.scores(Xq), h2.scores(Xq), rtol=1e-5)


def test_legacy_checkpoint_without_pooling_loads_as_mean(tmp_path):
    """Checkpoints predating the pooling field must stay servable, not crash."""
    h = FeCAMHead(D, 8, pooling="mean")
    h.observe(*make_clusters(n_classes=3, n_per=30)[:2])
    p = tmp_path / "legacy.npz"
    np.savez_compressed(p, feature_dim=D, max_classes=8, means=h.means,
                        counts=h.counts, cov_sum=h._cov_sum, cov_n=h._cov_n)
    loaded = FeCAMHead.load(p)
    assert loaded.pooling == "mean" and loaded.feature_dim == D

    Xq = RNG.standard_normal((5, D))
    np.testing.assert_allclose(loaded.scores(Xq), h.scores(Xq), atol=1e-9)
    # and it still pools windows the way it was fitted
    assert loaded.window_to_embedding(RNG.standard_normal((16, D))).shape == (D,)


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        FeCAMHead(D, 4, pooling="bogus")


def test_enrollment_reuses_the_precision_matrix():
    """Enrolling must not recompute the D x D inverse.

    update_cov=False leaves the shared covariance untouched, so the precision is
    still correct; rebuilding it anyway cost 315 ms per enrolled window at
    D=2560, which alone blows a 10 fps budget (see
    reports/realtime_incremental_result.md). Identity, not equality: an equal but
    freshly-inverted matrix would mean the work was still done.
    """
    X, y, _ = make_clusters(n_classes=4, n_per=30)
    h = FeCAMHead(feature_dim=D, max_classes=16)
    h.observe(X, y)
    prec_before, sd_before, _ = h._cov_terms()

    h.observe(RNG.standard_normal((3, D)), np.full(3, 9), update_cov=False)
    h.scores(X[:2])
    prec_after, sd_after, _ = h._cov_terms()

    assert prec_after is prec_before and sd_after is sd_before

    h.observe(RNG.standard_normal((3, D)), np.full(3, 9))    # update_cov=True
    assert h._cov_terms()[0] is not prec_before, "a covariance update must rebuild"


def test_incremental_cache_matches_full_rebuild():
    """Patching single rows of the mean-dependent cache must be exact.

    Interleaves enrollments, covariance updates and removals against a head that
    is forced to rebuild everything from scratch before each scoring call.
    """
    X, y, _ = make_clusters(n_classes=5, n_per=30)
    inc = FeCAMHead(feature_dim=D, max_classes=16)
    ref = FeCAMHead(feature_dim=D, max_classes=16)
    for h in (inc, ref):
        h.observe(X, y)

    for step in range(12):
        new = RNG.standard_normal((2, D))
        cid = step % 8
        for h in (inc, ref):
            if step % 4 == 3:
                h.observe(new, np.full(2, cid))          # covariance moves too
            else:
                h.observe(new, np.full(2, cid), update_cov=False)
        if step == 7:
            for h in (inc, ref):
                h.remove_classes([1])

        ref._cov_cache = ref._mean_cache = None          # force a full rebuild
        ref._dirty.clear()
        got, want = inc.scores(X[:6]), ref.scores(X[:6])
        # Not bit-identity: patching one row computes mu[rows] @ prec, while the
        # full rebuild computes the whole mu @ prec, and BLAS sums those in a
        # different order. The gap is ~1 ULP (observed 3e-16 relative on numpy
        # 2.5, 0 on 2.3), so pin it well below anything a real bug could hide in.
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
        # Predictions must agree exactly -- that is the behavioural contract.
        np.testing.assert_array_equal(got.argmax(axis=1), want.argmax(axis=1))


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


def test_few_shot_correction_is_a_noop_when_counts_are_equal():
    """The correction adds penalty/n_c per class, so with equal n it is a constant
    shift of every score -- the argmax, and hence every balanced benchmark number,
    must be untouched."""
    X, y, _ = make_clusters(n_classes=5, n_per=30)
    off = FeCAMHead(D, 8)
    on = FeCAMHead(D, 8, few_shot_correction=True)
    for h in (off, on):
        h.observe(X, y)
    Xq = RNG.standard_normal((40, D))
    np.testing.assert_array_equal(off.scores(Xq).argmax(axis=1),
                                  on.scores(Xq).argmax(axis=1))


def test_few_shot_correction_lifts_an_undersampled_class():
    """With unequal counts it must favour the class that had fewer examples,
    which is the whole point -- otherwise a 5-clip enrollment is penalised for
    being new rather than for being wrong."""
    X, y, protos = make_clusters(n_classes=3, n_per=60)
    off = FeCAMHead(D, 8)
    on = FeCAMHead(D, 8, few_shot_correction=True)
    few = protos[0] + 0.15 * RNG.standard_normal((4, D))     # 4 examples vs 60
    for h in (off, on):
        h.observe(X, y)
        h.observe(few, np.full(4, 7), update_cov=False)

    q = protos[0] + 0.15 * RNG.standard_normal((30, D))
    s_off, s_on = off.scores(q), on.scores(q)

    # The 4-example class gains more than the 60-example ones, by exactly the
    # ratio of their counts -- that is the correction's defining behaviour.
    gain_few = (s_on[:, 7] - s_off[:, 7]).mean()
    gain_base = (s_on[:, :3] - s_off[:, :3]).mean()
    assert gain_few > gain_base > 0
    assert gain_few / gain_base == pytest.approx(60 / 4, rel=1e-6)

    # Base classes all shift by the same amount, so their ranking is preserved.
    np.testing.assert_array_equal(s_off[:, :3].argmax(axis=1), s_on[:, :3].argmax(axis=1))
    # And the undersampled class wins more often than before.
    assert (s_on.argmax(axis=1) == 7).sum() > (s_off.argmax(axis=1) == 7).sum()


def test_few_shot_correction_survives_a_save_load_round_trip(tmp_path):
    X, y, _ = make_clusters(n_classes=3, n_per=30)
    h = FeCAMHead(D, 8, few_shot_correction=True)
    h.observe(X, y)
    p = tmp_path / "h.npz"
    h.save(p)
    loaded = FeCAMHead.load(p)
    assert loaded.few_shot_correction is True
    # and a checkpoint written before the flag existed defaults to off
    import numpy as _np
    legacy = tmp_path / "legacy.npz"
    _np.savez_compressed(legacy, feature_dim=D, max_classes=8, means=h.means,
                         counts=h.counts, cov_sum=h._cov_sum, cov_n=h._cov_n)
    assert FeCAMHead.load(legacy).few_shot_correction is False
