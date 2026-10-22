"""Band-limited GCC-PHAT time-delay estimation with confidence and ambiguity measures.

:func:`gcc_phat` returns the lag (seconds) by which ``sig`` must be *advanced* to align with
``ref``: a positive lag means ``sig`` is late relative to ``ref`` (``sig(t) ≈ ref(t - lag)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LagEstimate:
    lag: float  # seconds
    confidence: float  # peak / median |correlation| (≈1 for noise, ≫1 for a real match)
    ambiguity: float  # second-best / best peak, separated by min_sep (1 = equally strong peaks)
    peak: float


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    rate: int,
    *,
    max_lag: float,
    band: tuple[float, float] = (200.0, 6000.0),
    min_sep: float = 0.05,
    eps: float = 1e-9,
) -> LagEstimate:
    """Estimate the lag of ``sig`` relative to ``ref`` within ``±max_lag`` seconds."""
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if sig.ndim != 1 or ref.ndim != 1 or sig.size == 0 or ref.size == 0:
        raise ValueError("gcc_phat needs two non-empty 1-D signals")
    if not math.isfinite(max_lag) or max_lag <= 0:
        raise ValueError("max_lag must be positive")
    if rate <= 0 or not (np.isfinite(sig).all() and np.isfinite(ref).all()):
        raise ValueError("signals must be finite and sample rate positive")
    n = _next_pow2(sig.size + ref.size)
    s = np.fft.rfft(sig - sig.mean(), n)
    r = np.fft.rfft(ref - ref.mean(), n)
    cross = s * np.conj(r)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    mag = np.abs(cross)
    whitened = np.where(keep, cross / np.maximum(mag, eps), 0.0)
    corr = np.fft.irfft(whitened, n)
    max_shift = int(max_lag * rate)
    if max_shift < 1:
        raise ValueError("max_lag is shorter than one sample")
    # Unequal inputs have an asymmetric linear lag domain. Restricting to ±n/2
    # would wrap a late match in a long reference onto a false positive lag.
    lags = np.arange(-min(max_shift, ref.size - 1), min(max_shift, sig.size - 1) + 1)
    window = corr[lags % n]
    absw = np.abs(window)
    k = int(np.argmax(absw))
    peak = float(absw[k])
    median = float(np.median(absw))
    confidence = peak / max(median, eps)
    # second-best peak at least min_sep seconds away
    sep = max(1, int(min_sep * rate))
    masked = absw.copy()
    masked[max(0, k - sep) : k + sep + 1] = 0.0
    second = float(masked.max()) if masked.size else 0.0
    ambiguity = second / max(peak, eps)
    # parabolic sub-sample refinement on the signed correlation around the peak
    frac = 0.0
    if 0 < k < window.size - 1:
        y0, y1, y2 = window[k - 1], window[k], window[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > eps:
            frac = 0.5 * (y0 - y2) / denom
            frac = max(-0.5, min(0.5, frac))
    lag = (lags[k] + frac) / rate
    if not math.isfinite(lag):
        raise ValueError("non-finite lag")
    return LagEstimate(lag=float(lag), confidence=confidence, ambiguity=ambiguity, peak=peak)
