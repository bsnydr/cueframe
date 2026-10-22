"""Synthesise a short, non-speech musical piece (pads, kicks, plucks, sweeps) with numpy only."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

TAU = 2.0 * math.pi
BEAT = 0.5  # seconds per beat at 120 BPM
EIGHTH = BEAT / 2
FADE_S = 0.030
PENTATONIC = np.array([220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33])
ARP_PATTERN = np.array([0, 2, 4, 5, 4, 2, 3, 1, 0, 3, 5, 7, 5, 3, 4, 2])
CHORD = (110.0, 164.81, 220.0, 261.63)

_Rng = np.random.Generator


def _db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def _pad(t: np.ndarray, rms: float, attack: float = 0.8) -> np.ndarray:
    """Detuned sine pad with a slow attack; ``rms`` is the steady-state RMS."""
    out = np.zeros_like(t)
    for i, f in enumerate(CHORD):
        out += np.sin(TAU * f * t + i)
        out += 0.5 * np.sin(TAU * (f + 0.35 + 0.2 * i) * t + 2 * i)
    out *= rms / math.sqrt(0.5 * len(CHORD) * (1 + 0.25))
    return out * np.minimum(1.0, t / attack) ** 2


def _kick_and_hat(t: np.ndarray, rng: _Rng) -> np.ndarray:
    tb = np.mod(t, BEAT)
    phase = TAU * (55.0 * tb + 40.0 * 0.04 * (1 - np.exp(-tb / 0.04)))
    kick = 0.7 * np.sin(phase) * np.exp(-tb / 0.09) * np.minimum(1.0, tb / 0.002)
    th = np.mod(t - BEAT / 2, BEAT)  # off-beat ticks
    noise = np.diff(rng.standard_normal(t.size + 1))  # crude high-pass
    hat = 0.12 * noise * np.exp(-th / 0.025) * np.minimum(1.0, th / 0.001)
    return kick + hat


def _arp(t: np.ndarray, level: float = 0.3) -> np.ndarray:
    idx = np.floor(t / EIGHTH).astype(np.int64)
    tn = t - idx * EIGHTH
    f = PENTATONIC[ARP_PATTERN[idx % ARP_PATTERN.size]]
    out = np.zeros_like(t)
    for h, amp in ((1, 1.0), (2, 0.45), (3, 0.25)):
        out += amp * np.sin(TAU * h * f * tn) * np.exp(-tn * (13.0 + 5.0 * h))
    return level * out * np.minimum(1.0, tn / 0.002)


def _lowpass_sweep(x: np.ndarray, sr: int, fc0: float, fc1: float) -> np.ndarray:
    """One-pole low-pass whose cutoff rises exponentially; vectorised via block recurrences."""
    block = 128
    n = x.size
    nb = -(-n // block)
    xp = np.zeros(nb * block)
    xp[:n] = x
    fc = fc0 * (fc1 / fc0) ** (np.arange(nb) / max(1, nb - 1))
    r = np.exp(-TAU * fc / sr)[:, None]  # pole per block
    a = 1.0 - r
    k = np.arange(block)[None, :]
    xb = xp.reshape(nb, block)
    # zero-state response within each block: y[k] = a * sum_j r^(k-j) x[j]
    zs = a * r**k * np.cumsum(xb * r ** (-k), axis=1)
    rb = (r[:, 0]) ** block
    carry = np.empty(nb)
    prev = 0.0
    for b in range(nb):  # scalar recurrence over blocks only
        prev = rb[b] * prev + zs[b, -1]
        carry[b] = prev
    start = np.concatenate(([0.0], carry[:-1]))[:, None]
    y = zs + start * r ** (k + 1)
    return y.reshape(-1)[:n]


def _comet(t: np.ndarray, dur: float, sr: int, rng: _Rng) -> np.ndarray:
    f0 = 220.0
    phase = TAU * f0 * dur / (2 * math.log(2)) * (2.0 ** (2.0 * t / dur) - 1.0)
    phase += 0.8 * np.sin(TAU * 5.5 * t)  # light vibrato
    out = 0.3 * np.sin(phase) * (0.4 + 0.6 * t / dur)
    n_ping = max(1, int(dur * 2.5))
    times = np.sort(rng.uniform(0.2, max(0.25, dur - 0.4), n_ping))
    freqs = rng.choice(PENTATONIC[3:] * 4, n_ping)
    for tp, f in zip(times, freqs, strict=True):
        i0 = int(tp * sr)
        m = min(int(0.4 * sr), t.size - i0)
        if m <= 0:
            continue
        tt = np.arange(m) / sr
        ping = np.sin(TAU * f * tt) + 0.4 * np.sin(TAU * 2.01 * f * tt)
        out[i0 : i0 + m] += 0.12 * ping * np.exp(-tt / 0.06) * np.minimum(1.0, tt / 0.001)
    return out


def _section(mood: str, n: int, sr: int, rng: _Rng) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    dur = n / sr
    if mood == "title":
        return _pad(t, _db(-18.0))
    if mood == "pulse":
        return _kick_and_hat(t, rng)
    if mood == "orbit":
        return _arp(t) + _pad(t, _db(-26.0))
    if mood == "bars":
        mix = _kick_and_hat(t, rng) + _arp(t, 0.35)
        return 1.2 * _lowpass_sweep(mix, sr, 400.0, min(10000.0, 0.45 * sr))
    if mood == "comet":
        return _comet(t, max(dur, 1e-3), sr, rng)
    if mood == "outro":
        return _pad(t, _db(-18.0)) * np.clip(1.0 - t / max(dur, 1e-9), 0.0, 1.0)
    raise ValueError(f"unknown mood {mood!r}")


_MOODS = ("title", "pulse", "orbit", "bars", "comet", "outro")


def render_piece(
    sections: Sequence[tuple[float, float, str]],
    *,
    sample_rate: int = 48000,
    seed: int = 7,
) -> np.ndarray:
    """Render ``sections`` of ``(start_s, end_s, mood)`` into float32 mono in [-0.9, 0.9]."""
    prev_end = 0.0
    for start, end, mood in sections:
        if mood not in _MOODS:
            raise ValueError(f"unknown mood {mood!r}; expected one of {_MOODS}")
        if start < prev_end or end <= start:
            raise ValueError("sections must be sorted, non-overlapping and have positive length")
        prev_end = end
    total = math.ceil(prev_end * sample_rate) if sections else 0
    out = np.zeros(total, dtype=np.float64)
    for i, (start, end, mood) in enumerate(sections):
        s0 = min(total, int(round(start * sample_rate)))
        s1 = min(total, max(s0, int(round(end * sample_rate))))
        n = s1 - s0
        if n == 0:
            continue
        rng = np.random.default_rng([seed, i])
        seg = _section(mood, n, sample_rate, rng)
        m = min(int(FADE_S * sample_rate), n // 2)
        if m > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)
            seg[:m] *= ramp
            seg[n - m :] *= ramp[::-1]
        out[s0:s1] = seg
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = 0.9 * np.tanh(out / 0.9)  # soft limiter: |out| < 0.9, silence stays exactly zero
    return np.clip(out, -0.9, 0.9).astype(np.float32)
