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
        self._cache = None          # (precision, sd) cache, invalidated on update

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
        for c in np.unique(y):
            m = X[y == c]
            tot = self.counts[c] + len(m)
            self.means[c] = (self.means[c] * self.counts[c] + m.sum(axis=0)) / tot
            self.counts[c] = tot
        if update_cov:
            D = X - self.means[y]
            self._cov_sum += D.T @ D
            self._cov_n += len(X)
        self._cache = None

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
        self._cache = None

    # ── scoring ──────────────────────────────────────────────────────────────
    def _precision(self):
        if self._cache is None:
            d = self.feature_dim
            cov = self._cov_sum / max(self._cov_n, 1)
            diag_mean = float(np.trace(cov)) / d
            off = cov - np.diag(np.diag(cov))
            off_mean = float(off.sum()) / (d * (d - 1))
            cov = cov + SHRINK_1 * diag_mean * np.eye(d) + SHRINK_2 * off_mean * (1 - np.eye(d))
            sd = np.sqrt(np.diag(cov))
            corr = cov / np.outer(sd, sd)
            self._cache = (np.linalg.inv(corr), sd)
        return self._cache

    def scores(self, X: np.ndarray) -> np.ndarray:
        """(N, D) raw embeddings -> (N, max_classes) scores (higher = better).

        Score = negative correlation-normalized Mahalanobis distance; classes
        never enrolled get -inf-like scores.
        """
        X = self._prep(X)
        prec, sd = self._precision()
        active = np.where(self.counts > 0)[0]
        S = np.full((len(X), self.max_classes), -1e18)
        for c in active:
            D = (X - self.means[c]) / sd
            S[:, c] = -np.einsum("nd,de,ne->n", D, prec, D)
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
