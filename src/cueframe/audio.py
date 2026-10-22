"""PCM WAV input/output and per-frame audio features.

Only uncompressed PCM WAV is read here (8/16/24/32-bit integer, any channel count); this
keeps the audio-only path free of ffmpeg for analysis. Compressed audio should be converted
first, e.g. ``ffmpeg -i in.m4a -ac 1 -ar 48000 out.wav``.
"""

from __future__ import annotations

import hashlib
import math
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from cueframe.timeline import frame_count


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate


def read_wav(path: str | Path) -> Audio:
    p = Path(path)
    try:
        with wave.open(str(p), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"{p}: not a PCM WAV file ({exc})") from exc
    if n == 0:
        raise AudioError(f"{p}: contains no samples")
    if channels < 1 or rate <= 0:
        raise AudioError(f"{p}: invalid header (channels={channels}, rate={rate})")
    if len(raw) != n * channels * width:
        raise AudioError(f"{p}: truncated PCM data")
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        val = np.where(val >= 1 << 23, val - (1 << 24), val)
        data = val.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise AudioError(f"{p}: unsupported sample width {width} bytes")
    if channels > 1:
        usable = (data.shape[0] // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return Audio(np.ascontiguousarray(data, dtype=np.float32), rate)


def write_wav(path: str | Path, audio: Audio) -> None:
    """Write mono 16-bit PCM."""
    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(pcm.tobytes())


def file_fingerprint(path: str | Path) -> tuple[int, int, str]:
    """(size, mtime_ns, sha256) used to prove an input file was not modified."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return st.st_size, st.st_mtime_ns, h.hexdigest()


@dataclass(frozen=True)
class Features:
    """Per-output-frame audio features, one entry per frame."""

    level: np.ndarray  # smoothed RMS normalised to the 97th percentile, 0..1
    onset: np.ndarray  # positive change in level per frame, 0..1

    @property
    def n_frames(self) -> int:
        return int(self.level.shape[0])


def frame_features(
    audio: Audio,
    fps: Fraction,
    *,
    attack: float = 0.6,
    release: float = 0.25,
) -> Features:
    """Measure one level per output frame from the exact sample window of that frame.

    Frame ``f`` uses samples ``[floor(f * sr / fps), floor((f + 1) * sr / fps))``. RMS is
    smoothed with a one-pole filter (fast attack, slower release), normalised so the 97th
    percentile maps to 1.0 (robust to a single loud transient), then clipped to [0, 1].
    """
    if not 0 < attack <= 1 or not 0 < release <= 1:
        raise AudioError("attack and release must be in (0, 1]")
    n = frame_count(audio.duration, fps)
    sr = audio.sample_rate
    edges = [math.floor(Fraction(f) * sr / fps) for f in range(n + 1)]
    edges[-1] = min(edges[-1], audio.n_samples)
    rms = np.zeros(n, dtype=np.float64)
    x = audio.samples.astype(np.float64)
    for f in range(n):
        a, b = edges[f], edges[f + 1]
        if b > a:
            seg = x[a:b]
            rms[f] = math.sqrt(float(np.mean(seg * seg)))
    smooth = np.zeros_like(rms)
    prev = 0.0
    for f in range(n):
        coef = attack if rms[f] > prev else release
        prev = prev + coef * (rms[f] - prev)
        smooth[f] = prev
    ref = float(np.percentile(smooth, 97)) if n > 1 else float(smooth.max())
    if ref <= 1e-6:
        level = np.zeros(n)
    else:
        level = np.clip(smooth / ref, 0.0, 1.0)
    onset = np.clip(np.diff(level, prepend=0.0), 0.0, 1.0)
    return Features(level.astype(np.float32), onset.astype(np.float32))
