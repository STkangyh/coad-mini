"""
Anomaly / novel-action (OOD) detection
======================================

Goal: on the prediction stream of the GRUDetector, decide whether the current
action window is *out-of-distribution* (an action the model was never trained
on, or simply a low-confidence / novel observation). This is a NOVELTY/OOD
objective, NOT classification.

Scoring convention
------------------
All scoring functions return an **anomaly score** where

    HIGHER score == MORE anomalous (more out-of-distribution).

This convention is consistent across every metric so a single threshold rule
(`score >= threshold => anomaly`) works for all of them:

  * ``max_softmax_prob``   -> returns the *negative* max-softmax-probability
                             (-MSP). An in-distribution input is classified
                             confidently (MSP near 1, so -MSP near -1, low),
                             an OOD input is uncertain (MSP small, -MSP near 0,
                             high).
  * ``predictive_entropy`` -> Shannon entropy of the softmax distribution.
                             Confident in-dist predictions have low entropy;
                             uncertain/uniform OOD predictions have high
                             entropy.
  * ``energy_score``       -> the standard free-energy OOD score
                             ``-T * logsumexp(logits / T)``. In-distribution
                             inputs produce large logits => low (very negative)
                             energy; OOD inputs produce small logits => high
                             energy.

The raw functions accept logits as a ``torch.Tensor``, ``np.ndarray`` or a
plain sequence. They operate on the last axis (the class axis) and reduce it,
so an input of shape ``(C,)`` returns a scalar and ``(N, C)`` returns ``(N,)``.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _as_tensor(logits) -> torch.Tensor:
    """Coerce logits (Tensor / ndarray / sequence) into a float Tensor."""
    if isinstance(logits, torch.Tensor):
        t = logits.detach().to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    if t.ndim == 0:
        raise ValueError("logits must have at least one (class) dimension")
    return t


def _to_numpy_scalar_or_array(t: torch.Tensor):
    """Return a python float for a 0-d result, else a numpy array."""
    if t.ndim == 0:
        return float(t.item())
    return t.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# scoring functions (HIGHER == more OOD)
# ──────────────────────────────────────────────────────────────────────────────
def max_softmax_prob(logits) -> "float | np.ndarray":
    """Anomaly score based on the maximum softmax probability (Hendrycks &
    Gimpel, 2017). Returns ``-MSP`` so that higher == more anomalous.

    Parameters
    ----------
    logits : Tensor | ndarray | sequence, shape (..., C)

    Returns
    -------
    float (for a single vector) or np.ndarray of shape (...,).
    """
    t = _as_tensor(logits)
    probs = F.softmax(t, dim=-1)
    msp = probs.max(dim=-1).values
    return _to_numpy_scalar_or_array(-msp)


def predictive_entropy(logits) -> "float | np.ndarray":
    """Shannon entropy of the softmax distribution (nats). Higher == more
    anomalous (more uncertain / closer to uniform)."""
    t = _as_tensor(logits)
    log_probs = F.log_softmax(t, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return _to_numpy_scalar_or_array(entropy)


def energy_score(logits, T: float = 1.0) -> "float | np.ndarray":
    """Free-energy OOD score (Liu et al., 2020): ``-T * logsumexp(logits / T)``.

    Higher == more anomalous. In-distribution inputs yield large logits and
    therefore low (strongly negative) energy.
    """
    t = _as_tensor(logits)
    energy = -T * torch.logsumexp(t / T, dim=-1)
    return _to_numpy_scalar_or_array(energy)


_METRICS = {
    "msp": max_softmax_prob,
    "max_softmax_prob": max_softmax_prob,
    "entropy": predictive_entropy,
    "predictive_entropy": predictive_entropy,
    "energy": energy_score,
    "energy_score": energy_score,
}


def anomaly_score(logits, metric: str = "energy", T: float = 1.0):
    """Dispatch to one of the scoring functions by name (higher == more OOD)."""
    key = metric.lower()
    if key not in _METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; choose from {sorted(set(_METRICS))}"
        )
    fn = _METRICS[key]
    if fn is energy_score:
        return energy_score(logits, T=T)
    return fn(logits)


# ──────────────────────────────────────────────────────────────────────────────
# AnomalyScorer: metric + calibrated threshold + temporal smoothing
# ──────────────────────────────────────────────────────────────────────────────
class AnomalyScorer:
    """Holds a chosen OOD metric and a calibrated decision threshold, and runs
    inference on feature windows with optional temporal smoothing across a
    stream.

    Parameters
    ----------
    metric : str
        One of ``"energy"``, ``"entropy"``, ``"msp"`` (default ``"energy"``).
    T : float
        Temperature for the energy score.
    threshold : float | None
        Decision threshold (``score >= threshold => anomaly``). Usually set by
        :meth:`calibrate`, but may be supplied directly.
    smoothing : {"ema", "mean", None}
        Temporal smoothing applied across a stream of windows so that a single
        noisy frame does not flip the flag.
    ema_alpha : float
        EMA weight for the newest score when ``smoothing == "ema"``.
    window : int
        Rolling-window length when ``smoothing == "mean"``.
    """

    def __init__(
        self,
        metric: str = "energy",
        T: float = 1.0,
        threshold: float | None = None,
        smoothing: str | None = "ema",
        ema_alpha: float = 0.5,
        window: int = 5,
    ):
        if metric.lower() not in _METRICS:
            raise ValueError(
                f"unknown metric {metric!r}; choose from {sorted(set(_METRICS))}"
            )
        if smoothing not in (None, "ema", "mean"):
            raise ValueError("smoothing must be one of None, 'ema', 'mean'")
        self.metric = metric.lower()
        self.T = float(T)
        self.threshold = threshold
        self.smoothing = smoothing
        self.ema_alpha = float(ema_alpha)
        self.window = int(window)

        # streaming state
        self._ema: float | None = None
        self._buffer: deque[float] = deque(maxlen=self.window)

    # ── score computation ────────────────────────────────────────────────────
    def score_logits(self, logits) -> float:
        """Anomaly score for a single logit vector (shape (C,) or (1, C))."""
        t = _as_tensor(logits)
        if t.ndim > 1:
            t = t.reshape(-1, t.shape[-1])[0]
        return float(anomaly_score(t, metric=self.metric, T=self.T))

    # ── calibration ──────────────────────────────────────────────────────────
    def calibrate(self, in_dist_scores: Sequence[float], target_fpr: float = 0.05) -> float:
        """Set ``self.threshold`` at the ``(1 - target_fpr)`` quantile of the
        in-distribution anomaly scores.

        With this rule, only ~``target_fpr`` of in-distribution windows score
        at/above the threshold and are (falsely) flagged as anomalous.

        Returns the chosen threshold.
        """
        scores = np.asarray(list(in_dist_scores), dtype=np.float64)
        if scores.size == 0:
            raise ValueError("in_dist_scores is empty")
        if not (0.0 < target_fpr < 1.0):
            raise ValueError("target_fpr must be in (0, 1)")
        q = 1.0 - target_fpr
        self.threshold = float(np.quantile(scores, q))
        return self.threshold

    # ── streaming reset ──────────────────────────────────────────────────────
    def reset_stream(self) -> None:
        """Clear temporal-smoothing state (call between independent streams)."""
        self._ema = None
        self._buffer.clear()

    def _smooth(self, raw_score: float) -> float:
        """Apply temporal smoothing and return the smoothed score."""
        if self.smoothing is None:
            return raw_score
        if self.smoothing == "ema":
            if self._ema is None:
                self._ema = raw_score
            else:
                a = self.ema_alpha
                self._ema = a * raw_score + (1.0 - a) * self._ema
            return self._ema
        # rolling mean
        self._buffer.append(raw_score)
        return float(np.mean(self._buffer))

    # ── window inference ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _model_logits(self, model, feat_window) -> torch.Tensor:
        """Run the GRUDetector on a feature window and return logits (C,)."""
        if isinstance(feat_window, torch.Tensor):
            x = feat_window.detach().to(torch.float32)
        else:
            x = torch.as_tensor(np.asarray(feat_window, dtype=np.float32))
        if x.ndim == 2:          # (T, C) -> (1, T, C)
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError(
                f"feat_window must be (T, F) or (B, T, F); got shape {tuple(x.shape)}"
            )
        was_training = model.training
        model.eval()
        logits, _ = model(x, None)
        if was_training:
            model.train()
        return logits[0]

    @torch.no_grad()
    def score_window(self, model, feat_window, smooth: bool = True) -> dict:
        """Run the model on ``feat_window`` and produce an anomaly verdict.

        Parameters
        ----------
        model : GRUDetector
        feat_window : Tensor | ndarray, shape (T, F) or (1, T, F)
            A single feature window (e.g. ``(16, 512)``).
        smooth : bool
            If True (default) the returned ``score``/``is_anomaly`` use the
            temporally smoothed score, advancing the stream state. The raw
            (unsmoothed) score is always returned under ``raw_score``.

        Returns
        -------
        dict with keys: ``score``, ``raw_score``, ``is_anomaly``,
        ``top1``, ``top1_prob``, ``threshold``.
        """
        logits = self._model_logits(model, feat_window)
        probs = F.softmax(logits, dim=-1)
        top1_prob, top1 = probs.max(dim=-1)

        raw_score = float(anomaly_score(logits, metric=self.metric, T=self.T))
        score = self._smooth(raw_score) if smooth else raw_score

        if self.threshold is None:
            is_anomaly = False
        else:
            is_anomaly = bool(score >= self.threshold)

        return {
            "score": float(score),
            "raw_score": float(raw_score),
            "is_anomaly": is_anomaly,
            "top1": int(top1.item()),
            "top1_prob": float(top1_prob.item()),
            "threshold": None if self.threshold is None else float(self.threshold),
        }

    @torch.no_grad()
    def score_stream(self, model, feat_windows: Iterable, reset: bool = True) -> list[dict]:
        """Convenience: score a sequence of windows as one smoothed stream."""
        if reset:
            self.reset_stream()
        return [self.score_window(model, w, smooth=True) for w in feat_windows]
