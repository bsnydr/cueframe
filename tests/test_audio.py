"""WAV reading, sample accounting and per-frame features measured on constructed signals."""

import wave
from fractions import Fraction

import numpy as np
import pytest

from cueframe.audio import Audio, AudioError, file_fingerprint, frame_features, read_wav, write_wav


def test_write_read_roundtrip_exact_sample_count(tmp_path):
    rate = 8000
    n = 8000 + 123  # not a whole number of frames at 25 fps
    x = (0.5 * np.sin(np.arange(n) * 0.01)).astype(np.float32)
    p = tmp_path / "a.wav"
    write_wav(p, Audio(x, rate))
    a = read_wav(p)
    assert a.sample_rate == rate
    assert a.n_samples == n
    assert a.duration == pytest.approx(n / rate)
    assert np.max(np.abs(a.samples - x)) < 1.0 / 32767 + 1e-6


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_sample_widths_and_stereo_downmix(tmp_path, width):
    rate, n = 4000, 400
    p = tmp_path / f"w{width}.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        full = (1 << (8 * width - 1)) - 1
        left = np.full(n, full // 2, dtype=np.int64)
        right = np.zeros(n, dtype=np.int64)
        inter = np.empty(2 * n, dtype=np.int64)
        inter[0::2], inter[1::2] = left, right
        if width == 1:
            raw = (inter + 128).astype(np.uint8).tobytes()
        else:
            raw = b"".join(int(v).to_bytes(width, "little", signed=True) for v in inter)
        wf.writeframes(raw)
    a = read_wav(p)
    assert a.n_samples == n
    # mono downmix of (0.5, 0.0) ≈ 0.25
    assert float(np.mean(a.samples)) == pytest.approx(0.25, abs=0.01)


def test_read_rejects_non_wav(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"not a wav file at all")
    with pytest.raises(AudioError):
        read_wav(p)


def test_fingerprint_detects_change(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, Audio(np.zeros(100, dtype=np.float32), 8000))
    f1 = file_fingerprint(p)
    write_wav(p, Audio(np.ones(100, dtype=np.float32) * 0.1, 8000))
    f2 = file_fingerprint(p)
    assert f1[2] != f2[2]


def test_frame_features_silence_tone_silence():
    rate = 8000
    fps = Fraction(25)  # 320 samples per frame
    x = np.zeros(rate * 3, dtype=np.float32)  # 3 s = 75 frames
    x[rate : 2 * rate] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    feats = frame_features(Audio(x, rate), fps)
    assert feats.n_frames == 75
    assert np.all(feats.level[:25] == 0.0)
    # tone starts exactly at frame 25: first non-zero level and the only large onset there
    assert feats.level[25] > 0.3
    assert int(np.argmax(feats.onset)) == 25
    assert np.all(feats.onset[26:] < feats.onset[25])
    # release: level decays after the tone stops, never negative, ≤ 1
    assert feats.level[50] < feats.level[49]
    assert feats.level.min() >= 0.0 and feats.level.max() <= 1.0


def test_frame_features_frame_count_matches_timeline_rule():
    rate = 8000
    x = np.zeros(rate + 1, dtype=np.float32)  # 1.000125 s → 26 frames at 25 fps
    feats = frame_features(Audio(x, rate), Fraction(25))
    assert feats.n_frames == 26
    assert np.all(feats.level == 0.0)
