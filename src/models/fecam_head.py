"""
FeCAM classifier head (Goswami et al., NeurIPS 2023) on frozen CLIP features.

Backprop-free, exemplar-free: stores one mean per class + ONE shared covariance.
Prediction = Mahalanobis distance with the paper's normalization tricks
(sign-preserving Tukey power transform, covariance shrinkage, correlation
normalization). Measured on our 8-stage protocol it beats the GRU+A-GEM head
on every metric (task-aware 0.410 vs 0.387, full-48-way 0.157 vs 0.105) while
"training" in seconds — see reports/cpu_friendly_methods_result.md.

Key property for the demo: enrolling a NEW class = computing one mean vector
(milliseconds), and it can never disturb existing classes (their statistics
are untouched) — forgetting-free enrollment by construction.

Input convention: a "window" is the CLIP feature array of shape (T, D)
(e.g. (16, 512)); `window_to_embedding` collapses T into one vector using this
head's `pooling`. The default is `chunks4`, which keeps temporal order and so
makes feature_dim 4x the per-frame dimension (2048 for CLIP B/32).
`chunks3_adjdiff` (5x) was the previous default and `mean` the one before that;
both are kept so existing checkpoints stay loadable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TUKEY_LAMBDA = 0.5
SHRINK_1 = 1.0   # diagonal shrinkage weight
SHRINK_2 = 1.0   # off-diagonal shrinkage weight


def _tukey(X: np.ndarray, lam: float = TUKEY_LAMBDA) -> np.ndarray:
    """Sign-preserving Tukey ladder-of-powers (CLIP features have negatives)."""
    return np.sign(X) * (np.abs(X) ** lam)


# ── window poolings: (T, D) per-frame features -> one vector ─────────────────
# A window has to become a single vector before FeCAM sees it. How that is done
# decides whether temporal order survives, which is worth far more than it looks:
# mean-pool maps "push left-to-right" and "push right-to-left" to the SAME vector.
# See reports/ssv2_temporal_pooling_result.md for the comparison.

def _pool_mean(w: np.ndarray) -> np.ndarray:
    """Order-blind average. The original default; kept for old checkpoints."""
    return w.mean(axis=0)


def _segment_means(w: np.ndarray, k: int) -> np.ndarray:
    """k equal temporal segments, each mean-pooled -> (k*D,).

    Order survives because segment i keeps its own slot in the output: reversing
    the window swaps segment 1 with k, 2 with k-1, ... and so produces a
    different vector. That is the whole mechanism -- mean-pool has one slot and
    therefore cannot tell the two apart.
    """
    if len(w) < k:                      # too short to segment: repeat the tail
        w = np.concatenate([w, np.repeat(w[-1:], k - len(w), axis=0)])
    idx = np.linspace(0, len(w), k + 1).astype(int)
    return np.concatenate([w[idx[i]:idx[i + 1]].mean(axis=0) for i in range(k)])


def _pool_chunks4(w: np.ndarray) -> np.ndarray:
    """Four temporal segment means. The deployed default (D = 4x per-frame).

    Chosen on a held-out split of TRAIN, never on val -- see
    reports/ssv2_temporal_pooling_result.md §10. Worth +7.83pp on SSv2 over
    mean-pool. Note the honest caveat from that section: the top few
    order-preserving poolings are statistically indistinguishable on our data,
    so this is "one of the best", not "the best".
    """
    return _segment_means(w, 4)


def _pool_chunks3_adjdiff(w: np.ndarray) -> np.ndarray:
    """Three segment means + differences between adjacent segments (D = 5x).

    The previous default. Kept because checkpoints written with it must stay
    loadable, and because it is the variant the original study reported. Its
    extra signal over plain segments is the adjacent differences, which flip
    sign when the action reverses; measured against `chunks4` that buys nothing
    distinguishable from noise (2 wins in 5 seeds).
    """
    d = w.shape[1]
    c = _segment_means(w, 3)
    return np.concatenate([c, c[d:2 * d] - c[:d], c[2 * d:] - c[d:2 * d]])


POOLINGS = {
    "mean": _pool_mean,
    "chunks4": _pool_chunks4,
    "chunks3_adjdiff": _pool_chunks3_adjdiff,
}
POOLING_DIM_FACTOR = {"mean": 1, "chunks4": 4, "chunks3_adjdiff": 5}
DEFAULT_POOLING = "chunks4"


def pooled_dim(frame_dim: int, pooling: str = DEFAULT_POOLING) -> int:
    """Head feature_dim implied by a per-frame dimension and a pooling."""
    return frame_dim * POOLING_DIM_FACTOR[pooling]


class FeCAMHead:
    """Shared-covariance FeCAM classifier over frozen features."""

    def __init__(self, feature_dim: int, max_classes: int = 256,
                 pooling: str = DEFAULT_POOLING, few_shot_correction: bool = False):
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}, expected one of {sorted(POOLINGS)}")
        self.feature_dim = int(feature_dim)
        self.max_classes = int(max_classes)
        # Only used by window_to_embedding: callers that pool externally and hand
        # over (N, feature_dim) vectors are unaffected by this setting.
        self.pooling = pooling
        # See `scores`. Off by default: it is a no-op when every class has the
        # same number of examples, and changing it alters live enrollment only.
        self.few_shot_correction = bool(few_shot_correction)
        self.means = np.zeros((max_classes, feature_dim), dtype=np.float64)
        self.counts = np.zeros(max_classes, dtype=np.int64)
        self._cov_sum = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        self._cov_n = 0
        # Two caches with different invalidation rules -- see `_precision`.
        self._cov_cache = None      # (precision, sd): depends only on the covariance
        self._mean_cache = None     # (active, mu, mu_prec, mu_quad): also on the means
        self._dirty: set[int] = set()   # classes whose mean moved since _mean_cache

    # ── feature prep ─────────────────────────────────────────────────────────
    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        X = _tukey(X)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def window_to_embedding(self, window: np.ndarray) -> np.ndarray:
        """(T, D) frame features -> one (feature_dim,) window embedding.

        Uses this head's `pooling`, so enrollment and prediction can never
        disagree about it -- both go through here.
        """
        w = np.asarray(window, dtype=np.float64)
        if w.ndim == 3:            # (1, T, D)
            w = w[0]
        emb = POOLINGS[self.pooling](w)
        if emb.shape[0] != self.feature_dim:
            raise ValueError(
                f"pooling {self.pooling!r} on {w.shape} frames gives dim "
                f"{emb.shape[0]}, but this head expects {self.feature_dim}")
        return emb

    # ── fitting / enrollment ─────────────────────────────────────────────────
    def observe(self, X: np.ndarray, y: np.ndarray, update_cov: bool = True):
        """Accumulate class means (and optionally the shared covariance).

        X: (N, D) raw embeddings (NOT prepped), y: (N,) int class ids.
        Streaming-safe: repeated calls accumulate exactly.
        """
        X = self._prep(X)
        y = np.asarray(y, dtype=np.int64)
        touched = np.unique(y)
        for c in touched:
            m = X[y == c]
            tot = self.counts[c] + len(m)
            self.means[c] = (self.means[c] * self.counts[c] + m.sum(axis=0)) / tot
            self.counts[c] = tot
        if update_cov:
            D = X - self.means[y]
            self._cov_sum += D.T @ D
            self._cov_n += len(X)
            self._cov_cache = None      # precision changed -> everything downstream did
            self._mean_cache = None
        self._dirty.update(int(c) for c in touched)

    def enroll_class(self, class_id: int, windows: list[np.ndarray]) -> int:
        """Instant few-shot enrollment: one mean from a few (T, D) windows.

        Does NOT touch other classes or the shared covariance -> cannot cause
        forgetting. Returns the number of examples used.
        """
        emb = np.stack([self.window_to_embedding(w) for w in windows])
        self.observe(emb, np.full(len(emb), class_id), update_cov=False)
        return len(emb)

    def remove_classes(self, class_ids: list[int]):
        for c in class_ids:
            self.means[c] = 0.0
            self.counts[c] = 0
        # The shared covariance keeps whatever those classes contributed, so the
        # precision is still valid; only the per-class terms must be rebuilt.
        self._mean_cache = None
        self._dirty.clear()

    # ── scoring ──────────────────────────────────────────────────────────────
    def _cov_terms(self):
        """Cached (precision, sd, mean_penalty). Depends ONLY on the covariance.

        `mean_penalty` = tr(P @ Sigma_scaled), the constant in the few-shot
        correction; see `scores`. It is cached here because it depends on exactly
        the same inputs as the precision.
        """
        if self._cov_cache is None:
            d = self.feature_dim
            raw = self._cov_sum / max(self._cov_n, 1)
            diag_mean = float(np.trace(raw)) / d
            off = raw - np.diag(np.diag(raw))
            off_mean = float(off.sum()) / (d * (d - 1))
            cov = raw + SHRINK_1 * diag_mean * np.eye(d) + SHRINK_2 * off_mean * (1 - np.eye(d))
            sd = np.sqrt(np.diag(cov))
            prec = np.linalg.inv(cov / np.outer(sd, sd))
            # Sampling covariance of a class mean is raw/n; measured in the same
            # sd-scaled metric the scores use.
            penalty = float(np.trace(prec @ (raw / np.outer(sd, sd))))
            self._cov_cache = (prec, sd, penalty)
        return self._cov_cache

    def _precision(self):
        """Cached (precision, sd, active, scaled means, means@prec, m'Pm).

        Split into two caches because they are invalidated by different things.
        Enrolling a class (`update_cov=False`) moves one mean but leaves the
        covariance alone, so the D x D inverse -- by far the dominant term --
        stays valid and only that class's row of the mean-dependent terms has to
        be recomputed: O(D^2) instead of O(D^3).

        That distinction is what makes train-while-predicting affordable. In a
        streaming loop the head learns and scores on the same frame, so a full
        invalidation would pay the inverse EVERY frame: 8 ms at D=512, but 318 ms
        at the D=2560 of the chunks3+adjdiff pooling -- past the 100 ms budget of
        a 10 fps loop on its own. See reports/realtime_incremental_result.md.
        """
        prec, sd, _ = self._cov_terms()
        active_now = np.where(self.counts > 0)[0]

        if self._mean_cache is not None:
            active, mu, mu_prec, mu_quad = self._mean_cache
            if np.array_equal(active, active_now):
                if self._dirty:
                    # Same classes, some means moved -> patch just those rows.
                    ids = np.fromiter(sorted(self._dirty), dtype=np.int64)
                    rows = np.searchsorted(active, ids)
                    mu[rows] = self.means[ids] / sd
                    mu_prec[rows] = mu[rows] @ prec
                    mu_quad[rows] = np.einsum("kd,kd->k", mu_prec[rows], mu[rows])
                    self._dirty.clear()
                return (prec, sd, active, mu, mu_prec, mu_quad)

        active = active_now
        mu = self.means[active] / sd                      # (K, D), a fresh copy
        mu_prec = mu @ prec                               # (K, D)
        mu_quad = np.einsum("kd,kd->k", mu_prec, mu)      # (K,)  m' P m
        self._mean_cache = (active, mu, mu_prec, mu_quad)
        self._dirty.clear()
        return (prec, sd, active, mu, mu_prec, mu_quad)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """(N, D) raw embeddings -> (N, max_classes) scores (higher = better).

        Score = negative correlation-normalized Mahalanobis distance; classes
        never enrolled get -inf-like scores.

        Computed by expanding the quadratic form rather than looping per class:
            (x-m)' P (x-m) = x'Px - x'Pm - m'Px + m'Pm
        The m-only term is cached, so cost per window drops from K*D^2 to
        D^2 + 2KD -- ~40x fewer FLOPs at 48 classes, and one BLAS call instead
        of K einsum calls. Both cross terms are kept separate (rather than
        folded into 2x'Pm) so the result stays exact if P is not perfectly
        symmetric. Verified against the previous per-class loop on the real
        val set: max relative error 6e-15, argmax agreement 100%.

        FEW-SHOT CORRECTION (opt-in, `few_shot_correction=True`). A class mean
        estimated from n examples is noisy, and that noise inflates the measured
        distance by exactly tr(P @ Sigma_scaled)/n in expectation -- so a class
        taught from 5 clips is systematically penalised against one fitted on
        100, independent of where it actually sits. Adding the term back removes
        that bias.

        It is an exact no-op when every class has the same n (the bump is then a
        constant added to every score, leaving the argmax untouched), which is
        why none of the balanced benchmark numbers move. Measured effect where
        counts differ, 10 classes enrolled from 5 clips each
        (reports/enrollment_covariance_result.md):
            UCF101   enrolled 66.4 -> 83.2, base -0.6, overall +2.4
            SSv2     enrolled  0.1 ->  9.5, base -8.5, overall -3.3
        It helps when classes are separable and trades base accuracy away when
        they are not, so the default stays off.
        """
        X = self._prep(X)
        prec, sd, active, mu, mu_prec, mu_quad = self._precision()
        S = np.full((len(X), self.max_classes), -1e18)
        if len(active) == 0:
            return S
        xs = X / sd                                           # (N, D)
        xs_prec = xs @ prec                                   # (N, D)
        x_quad = np.einsum("nd,nd->n", xs_prec, xs)           # (N,)  x' P x
        cross = xs_prec @ mu.T + xs @ mu_prec.T               # (N, K)
        S[:, active] = -(x_quad[:, None] - cross + mu_quad[None, :])
        if self.few_shot_correction:
            penalty = self._cov_terms()[2]
            S[:, active] += penalty / self.counts[active]
        return S

    def predict_window(self, window: np.ndarray) -> tuple[int, float, np.ndarray]:
        """(T, D) window -> (top1 class id, softmax prob, full score row)."""
        s = self.scores(self.window_to_embedding(window)[None, :])[0]
        active = self.counts > 0
        e = np.exp((s - s[active].max()) * 0.5)       # temperature 2 on distances
        e[~active] = 0.0
        p = e / (e.sum() + 1e-12)
        top1 = int(s.argmax())
        return top1, float(p[top1]), s

    # ── persistence ──────────────────────────────────────────────────────────
    @property
    def n_classes(self) -> int:
        return int((self.counts > 0).sum())

    def save(self, path: str | Path):
        """Stores the covariance as a float32 upper triangle: 52 MB -> 13 MB at
        D=2560, which matters once a checkpoint is committed and shipped.

        Lossless in effect, for two reasons. The matrix is exactly symmetric
        (a sum of D'D outer products), so the triangle loses nothing. And the
        float32 rounding is ~1e-7 relative on entries that `_cov_terms` then
        regularizes by adding the full mean diagonal to the diagonal (SHRINK_1)
        -- the shrinkage dwarfs the rounding. Verified on the 48-class val set:
        identical accuracy, 100% argmax agreement.
        """
        iu = np.triu_indices(self.feature_dim)
        np.savez_compressed(
            path, feature_dim=self.feature_dim, max_classes=self.max_classes,
            means=self.means, counts=self.counts,
            cov_tri=self._cov_sum[iu].astype(np.float32),
            cov_n=self._cov_n, pooling=self.pooling,
            few_shot_correction=self.few_shot_correction,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeCAMHead":
        z = np.load(path)
        # Checkpoints written before pooling was configurable are mean-pooled.
        # Honouring that keeps them servable instead of silently mixing poolings.
        pooling = str(z["pooling"]) if "pooling" in z.files else "mean"
        fsc = bool(z["few_shot_correction"]) if "few_shot_correction" in z.files else False
        head = cls(int(z["feature_dim"]), int(z["max_classes"]), pooling=pooling,
                   few_shot_correction=fsc)
        head.means = z["means"]
        head.counts = z["counts"]
        if "cov_tri" in z.files:
            d = head.feature_dim
            cov = np.zeros((d, d), dtype=np.float64)
            iu = np.triu_indices(d)
            cov[iu] = z["cov_tri"]
            head._cov_sum = cov + np.triu(cov, 1).T      # mirror, diagonal once
        else:
            head._cov_sum = z["cov_sum"]                 # legacy full matrix
        head._cov_n = int(z["cov_n"])
        return head
