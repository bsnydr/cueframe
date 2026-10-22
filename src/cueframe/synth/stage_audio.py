"""Synthetic reference audio and per-camera audio seen through an affine clock.

The *reference recorder* track is designed chirps (no two alike) over a quiet ambience bed.
A camera's audio is the reference track read through the camera's clock: camera sample
``i`` at camera time ``i / sr`` carries the reference content at ``clock.to_ref(i / sr)``.
Each camera then gets its own tone shaping and independent noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cueframe.clock import AffineClock


@dataclass(frozen=True)
class Chirp:
    t_ref: float  # start time in reference seconds
    f0: float  # start frequency Hz
    f1: float  # end frequency Hz
    length: float  # seconds
    gain: float = 0.5


def design_chirps(events: list[float], seed: int) -> list[Chirp]:
    """A distinct sound per event: one to three back-to-back sweeps with random bands,
    directions and lengths, so no two events share the same time-frequency shape."""
    rng = np.random.default_rng(seed)
    chirps = []
    for t in events:
        t_seg = t
        for _ in range(int(rng.integers(1, 4))):
            lo = float(rng.uniform(250, 1200))
            hi = float(rng.uniform(2000, 6000))
            up = bool(rng.integers(0, 2))
            length = float(rng.uniform(0.06, 0.25))
            f0, f1 = (lo, hi) if up else (hi, lo)
            chirps.append(Chirp(t_seg, f0, f1, length, float(rng.uniform(0.3, 0.6))))
            t_seg += length + float(rng.uniform(0.0, 0.05))
    return chirps


def reference_track(
    duration: float, sample_rate: int, chirps: list[Chirp], seed: int, *, bed_rms: float = 0.01
) -> np.ndarray:
    """Chirps plus a low-level ambience bed (RMS ``bed_rms``, 0 to disable); float32 mono."""
    rng = np.random.default_rng(seed + 1)
    n = int(round(duration * sample_rate))
    out = np.zeros(n, dtype=np.float64)
    if bed_rms > 0:
        # ambience: band-limited noise with slow amplitude modulation, about -40 dBFS
        noise = rng.standard_normal(n)
        bed = _one_pole_lowpass(noise, 1200.0, sample_rate)
        bed -= _one_pole_lowpass(bed, 150.0, sample_rate)
        t = np.arange(n) / sample_rate
        mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + 1.0) * np.sin(2 * np.pi * 0.023 * t)
        bed *= mod
        bed *= bed_rms / max(float(np.sqrt(np.mean(bed * bed))), 1e-9)
        out += bed
    for c in chirps:
        m = int(round(c.length * sample_rate))
        i0 = int(round(c.t_ref * sample_rate))
        if i0 < 0 or i0 + m > n:
            continue
        tau = np.arange(m) / sample_rate
        phase = 2 * np.pi * (c.f0 * tau + 0.5 * (c.f1 - c.f0) / c.length * tau * tau)
        win = np.hanning(m)
        out[i0 : i0 + m] += c.gain * win * np.sin(phase)
    return np.clip(out, -0.95, 0.95).astype(np.float32)


def _one_pole_lowpass(x: np.ndarray, cutoff: float, sample_rate: int) -> np.ndarray:
    """One-pole low-pass as an FFT convolution with its (truncated) exponential impulse response."""
    a = float(np.exp(-2 * np.pi * cutoff / sample_rate))
    taps = int(np.ceil(np.log(1e-6) / np.log(a))) + 1
    kernel = (1.0 - a) * a ** np.arange(taps)
    n = 1 << int(np.ceil(np.log2(x.size + taps)))
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(kernel, n), n)
    return y[: x.size]


def camera_audio(
    reference: np.ndarray,
    sample_rate: int,
    clock: AffineClock,
    cam_duration: float,
    *,
    snr_db: float,
    tone: float,
    seed: int,
) -> np.ndarray:
    """Reference audio as recorded by a camera with the given clock.

    ``tone`` in [-1, 1] shifts the camera's tonal balance (negative = duller, positive =
    brighter) so cameras do not sound identical. Noise is added at ``snr_db`` relative to
    the signal RMS. Camera time outside the reference recording is silence (plus noise).
    """
    rng = np.random.default_rng(seed)
    n = int(round(cam_duration * sample_rate))
    t_cam = np.arange(n) / sample_rate
    t_ref = clock.offset + clock.rate * t_cam
    ref_idx = t_ref * sample_rate
    ref_t_axis = np.arange(reference.size)
    sig = np.interp(ref_idx, ref_t_axis, reference.astype(np.float64), left=0.0, right=0.0)
    sig[(ref_idx < 0) | (ref_idx > reference.size - 1)] = 0.0
    if tone != 0.0:
        lp = _one_pole_lowpass(sig, 2500.0, sample_rate)
        sig = sig + tone * (sig - lp) if tone > 0 else sig + tone * 0.5 * (sig - lp)
    rms = float(np.sqrt(np.mean(sig * sig))) or 1e-6
    noise = rng.standard_normal(n) * rms / (10 ** (snr_db / 20))
    out = sig + noise
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 0.95:
        out *= 0.95 / peak
    return out.astype(np.float32)


def resample_linear(x: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Plain linear resampling (adequate for analysis of band-limited chirps)."""
    if rate_in == rate_out:
        return x.astype(np.float32)
    n_out = int(round(x.size * rate_out / rate_in))
    pos = np.arange(n_out) * (rate_in / rate_out)
    return np.interp(pos, np.arange(x.size), x.astype(np.float64)).astype(np.float32)
