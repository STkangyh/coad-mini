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
(e.g. (16, 512)); the head mean-pools over T internally.
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


class FeCAMHead:
    """Shared-covariance FeCAM classifier over frozen features."""

    def __init__(self, feature_dim: int, max_classes: int = 256):
        self.feature_dim = int(feature_dim)
        self.max_classes = int(max_classes)
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

    @staticmethod
    def window_to_embedding(window: np.ndarray) -> np.ndarray:
        """(T, D) frame features -> (D,) video embedding (mean-pool)."""
        w = np.asarray(window, dtype=np.float64)
        if w.ndim == 3:            # (1, T, D)
            w = w[0]
        return w.mean(axis=0)

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
        """Cached (precision, sd). Depends ONLY on the shared covariance."""
        if self._cov_cache is None:
            d = self.feature_dim
            cov = self._cov_sum / max(self._cov_n, 1)
            diag_mean = float(np.trace(cov)) / d
            off = cov - np.diag(np.diag(cov))
            off_mean = float(off.sum()) / (d * (d - 1))
            cov = cov + SHRINK_1 * diag_mean * np.eye(d) + SHRINK_2 * off_mean * (1 - np.eye(d))
            sd = np.sqrt(np.diag(cov))
            corr = cov / np.outer(sd, sd)
            self._cov_cache = (np.linalg.inv(corr), sd)
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
        prec, sd = self._cov_terms()
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
        np.savez_compressed(
            path, feature_dim=self.feature_dim, max_classes=self.max_classes,
            means=self.means, counts=self.counts,
            cov_sum=self._cov_sum, cov_n=self._cov_n,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeCAMHead":
        z = np.load(path)
        head = cls(int(z["feature_dim"]), int(z["max_classes"]))
        head.means = z["means"]
        head.counts = z["counts"]
        head._cov_sum = z["cov_sum"]
        head._cov_n = int(z["cov_n"])
        return head
