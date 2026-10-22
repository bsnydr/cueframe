"""Fast unit tests for the synthetic stage renderer and the synthetic music piece."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from PIL import Image

from cueframe.synth.piece import render_piece
from cueframe.synth.stage import CameraView, render_stage_frame

SR = 48000
FRAMINGS = ("wide", "left", "right")
VIEWS = [CameraView("ABC"[p], f, p) for p in range(3) for f in FRAMINGS]


def _lum(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


@pytest.mark.parametrize("size", [(320, 180), (854, 480)])
@pytest.mark.parametrize("view", VIEWS)
def test_frame_is_rgb_of_exact_size(view, size):
    img = render_stage_frame(3.1, view, size)
    assert img.mode == "RGB"
    assert img.size == size


def test_frame_deterministic():
    view = CameraView("B", "left", 1)
    a = render_stage_frame(12.34, view, (320, 180), overlay_lines=("x",))
    b = render_stage_frame(12.34, view, (320, 180), overlay_lines=("x",))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != render_stage_frame(12.84, view, (320, 180)).tobytes()


@pytest.mark.parametrize("t_ref", [0.0, 7.3, 61.9])
@pytest.mark.parametrize("view", VIEWS)
def test_flash_contrast(view, t_ref):
    assert _lum(render_stage_frame(t_ref, view, (320, 180), flash=1.0)).mean() > 200
    assert _lum(render_stage_frame(t_ref, view, (320, 180), flash=0.0)).mean() < 110


def test_overlay_lines_render():
    view = CameraView("A", "wide", 0)
    lines = ("CAM A", "frame 000123", "00:00:04.920")
    plain = render_stage_frame(1.0, view, (320, 180))
    stamped = render_stage_frame(1.0, view, (320, 180), overlay_lines=lines)
    assert stamped.size == (320, 180)
    assert plain.tobytes() != stamped.tobytes()


def test_wide_frame_not_uniform():
    img = render_stage_frame(5.0, CameraView("A", "wide", 0), (320, 180))
    assert _lum(img).std() > 10


def test_cameras_look_different():
    frames = [render_stage_frame(5.0, CameraView("X", "wide", p), (320, 180)) for p in range(3)]
    means = [np.asarray(f, dtype=np.float32).mean(axis=(0, 1)) for f in frames]
    assert not np.allclose(means[0], means[1], atol=1.0)
    assert not np.allclose(means[0], means[2], atol=1.0)


def test_stage_speed():
    view = CameraView("A", "wide", 0)
    render_stage_frame(0.0, view, (854, 480))  # warm caches
    t0 = time.perf_counter()
    for i in range(5):
        render_stage_frame(i * 0.04, view, (854, 480), overlay_lines=("a", "b", "c"))
    assert (time.perf_counter() - t0) / 5 < 0.05


def _rms_100ms(x: np.ndarray) -> np.ndarray:
    n = SR // 10
    k = x.size // n
    return np.sqrt((x[: k * n].astype(np.float64).reshape(k, n) ** 2).mean(axis=1))


def test_piece_length_dtype_range():
    secs = [(0.0, 2.5, "title"), (3.0, 5.123456, "pulse"), (5.5, 9.0, "bars")]
    x = render_piece(secs, sample_rate=SR)
    assert x.dtype == np.float32
    assert x.shape == (math.ceil(9.0 * SR),)
    assert np.abs(x).max() <= 0.9
    assert not np.isnan(x).any()
    x2 = render_piece([(0.0, 1.0, "comet")], sample_rate=44100)
    assert x2.size == 44100


def test_all_moods_audible_and_bounded():
    for mood in ("title", "pulse", "orbit", "bars", "comet", "outro"):
        x = render_piece([(0.0, 6.0, mood)])
        assert 0.01 < np.abs(x).max() <= 0.9, mood
        assert np.isfinite(x).all(), mood


def test_deterministic_and_seed_dependent():
    secs = [(0.0, 4.0, "comet")]
    assert np.array_equal(render_piece(secs, seed=3), render_piece(secs, seed=3))
    assert not np.array_equal(render_piece(secs, seed=3), render_piece(secs, seed=4))


def test_pulse_periodicity_120_bpm():
    x = render_piece([(0.0, 8.0, "pulse")])
    r = _rms_100ms(x)
    r = r - r.mean()
    ac = np.array([np.dot(r[: r.size - k], r[k:]) for k in range(10)])
    assert ac[5] > 0.5 * ac[0]
    assert 5 == 2 + int(np.argmax(ac[2:8]))


def test_title_is_steady():
    x = render_piece([(0.0, 10.0, "title")])
    r = _rms_100ms(x)[15:85]
    assert r.mean() > 0.05
    assert r.std() / r.mean() < 0.2


def test_gap_is_exact_zero():
    x = render_piece([(0.0, 2.0, "title"), (4.0, 6.0, "pulse")])
    assert np.all(x[2 * SR : 4 * SR] == 0.0)
    assert np.abs(x[:SR]).max() > 0
    assert x[0] == 0.0 or abs(x[0]) < 1e-3  # faded in


def test_unknown_mood_raises():
    with pytest.raises(ValueError):
        render_piece([(0.0, 1.0, "polka")])


def test_outro_fades_to_silence():
    x = render_piece([(0.0, 6.0, "outro")])
    r = _rms_100ms(x)
    assert r[-1] < 0.1 * r[10:30].max()


def test_30s_piece_is_fast():
    secs = [(0.0, 5.0, "title"), (5.0, 10.0, "pulse"), (10.0, 15.0, "orbit"),
            (15.0, 20.0, "bars"), (20.0, 25.0, "comet"), (25.0, 30.0, "outro")]
    t0 = time.perf_counter()
    x = render_piece(secs)
    assert time.perf_counter() - t0 < 2.0
    assert x.size == 30 * SR
