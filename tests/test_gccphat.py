"""GCC-PHAT against signals shifted by known amounts, plus silence and periodic controls."""

import numpy as np
import pytest

from cueframe.gccphat import gcc_phat

RATE = 16000


def chirpy(rng: np.random.Generator, n: int) -> np.ndarray:
    """Broadband, non-periodic test signal: filtered noise bursts."""
    x = rng.standard_normal(n)
    env = np.zeros(n)
    for start in rng.integers(0, n - 2000, 12):
        env[start : start + 1500] += np.hanning(1500)
    return x * env


def shift(x: np.ndarray, lag_samples: float) -> np.ndarray:
    """Delay x by a (possibly fractional) number of samples using FFT phase shift."""
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    return np.fft.irfft(X * np.exp(-2j * np.pi * freqs * lag_samples), n)


@pytest.mark.parametrize("lag_samples", [0, 7, -13, 250, -2000, 3.5, -0.25])
def test_known_lag_recovered(lag_samples):
    rng = np.random.default_rng(1)
    ref = chirpy(rng, 4 * RATE)
    sig = shift(ref, lag_samples) + 0.01 * rng.standard_normal(ref.size)
    est = gcc_phat(sig, ref, RATE, max_lag=0.5)
    expected = lag_samples / RATE
    assert est.lag == pytest.approx(expected, abs=0.05e-3)  # within 0.05 ms
    assert est.confidence > 20
    assert est.ambiguity < 0.5


def test_silence_has_low_confidence():
    rng = np.random.default_rng(2)
    ref = chirpy(rng, 2 * RATE)
    sig = np.zeros_like(ref)
    est = gcc_phat(sig, ref, RATE, max_lag=0.5)
    assert est.confidence < 5


def test_noise_only_has_low_confidence():
    rng = np.random.default_rng(3)
    ref = chirpy(rng, 2 * RATE)
    sig = rng.standard_normal(ref.size)
    est = gcc_phat(sig, ref, RATE, max_lag=0.5)
    assert est.confidence < 8


def test_periodic_signal_is_ambiguous():
    """An identical burst every 0.1 s: the true lag and lag ± 0.1 s match equally well."""
    rng = np.random.default_rng(5)
    period = int(0.1 * RATE)
    burst = np.zeros(period)
    burst[:320] = rng.standard_normal(320) * np.hanning(320)
    ref = np.tile(burst, 20)
    # two recordings of the same periodic sound with independent room noise (~13 dB SNR per
    # harmonic): only the harmonic bins carry shared phase, so every period is a valid lag
    sig = np.roll(ref, int(0.03 * RATE)) + 0.3 * rng.standard_normal(ref.size)
    ref = ref + 0.3 * rng.standard_normal(ref.size)
    est = gcc_phat(sig, ref, RATE, max_lag=0.5, min_sep=0.05)
    assert est.ambiguity > 0.8
    # the reported lag is one of the equivalent solutions; it must not be presented as unique
    assert any(abs(est.lag - (0.03 + k * 0.1)) < 0.5e-3 for k in range(-5, 6))


def test_lag_beyond_max_lag_is_not_reported_as_valid():
    rng = np.random.default_rng(4)
    ref = chirpy(rng, 4 * RATE)
    sig = shift(ref, int(0.8 * RATE))  # 0.8 s, but we only search ±0.5 s
    est = gcc_phat(sig, ref, RATE, max_lag=0.5)
    assert abs(est.lag) <= 0.5
    assert est.confidence < 8 or abs(est.lag - 0.8) > 0.01


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        gcc_phat(np.zeros(0), np.zeros(10), RATE, max_lag=0.1)
    with pytest.raises(ValueError):
        gcc_phat(np.zeros(10), np.zeros(10), RATE, max_lag=0.0)
