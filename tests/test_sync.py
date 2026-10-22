"""Clock estimation against planted clocks the estimator never sees, plus negative controls."""

import numpy as np
import pytest

from cueframe.clock import AffineClock
from cueframe.sync import SyncConfig, fine_centers, sync_camera, theil_sen
from cueframe.synth.generate import audio_fixture
from cueframe.synth.stage_audio import Chirp, reference_track

RATE = 16000

DEMO_CLOCKS = {
    "A": AffineClock.from_drift_ppm(1.3, 0.0),
    "B": AffineClock.from_drift_ppm(-2.7, 3200.0),
    "C": AffineClock.from_drift_ppm(4.1, -2600.0),
}


@pytest.fixture(scope="module")
def demo_audio():
    return audio_fixture(64.0, DEMO_CLOCKS, sample_rate=RATE, seed=3)


def test_theil_sen_exact_on_clean_line_with_outlier():
    x = np.arange(10, dtype=float)
    y = 0.5 + 1.002 * x
    y[4] += 3.0  # one gross outlier must not move the median fit
    a, b = theil_sen(x, y)
    assert a == pytest.approx(0.5, abs=1e-9)
    assert b == pytest.approx(1.002, abs=1e-9)


def test_fine_centers_cover_track_and_phase_shift_is_disjoint():
    cfg = SyncConfig(window=2.5, stride=4.0)
    c0 = fine_centers(30.0, cfg)
    c1 = fine_centers(30.0, cfg, phase=cfg.stride / 2)
    assert c0[0] == 1.25 and c0[-1] <= 28.75
    assert not set(c0) & set(c1)


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_demo_fixture_recovers_offset_and_drift(demo_audio, name):
    ref, cams = demo_audio
    truth = DEMO_CLOCKS[name]
    result = sync_camera(cams[name], ref, RATE, name=name)
    assert result.status == "ok", result.reason
    assert result.clock is not None
    assert abs(result.clock.offset - truth.offset) * 1000 < 1.0  # ms
    assert abs(result.clock.drift_ppm - truth.drift_ppm) < 20.0
    assert result.n_used >= 4
    assert result.residual_rms_ms < 1.0


@pytest.mark.slow
def test_realistic_drift_600s():
    """Ten minutes with tens-of-ppm drift (the magnitude real cameras show)."""
    clocks = {
        "slow": AffineClock.from_drift_ppm(3.21, 62.0),
        "fast": AffineClock.from_drift_ppm(-1.07, -38.0),
    }
    ref, cams = audio_fixture(600.0, clocks, sample_rate=RATE, seed=9, n_events=80)
    for name, truth in clocks.items():
        result = sync_camera(cams[name], ref, RATE, name=name)
        assert result.status == "ok", result.reason
        assert abs(result.clock.drift_ppm - truth.drift_ppm) < 5.0
        assert abs(result.clock.offset - truth.offset) * 1000 < 0.5


def test_silent_camera_is_reported_silent(demo_audio):
    ref, cams = demo_audio
    result = sync_camera(np.zeros(20 * RATE, dtype=np.float32), ref, RATE, name="mute")
    assert result.status == "silent"
    assert result.clock is None


def test_too_short_input(demo_audio):
    ref, cams = demo_audio
    result = sync_camera(cams["A"][: 3 * RATE], ref, RATE, name="short")
    assert result.status == "too_short"
    assert result.clock is None


def test_unrelated_audio_has_no_match(demo_audio):
    ref, _ = demo_audio
    rng = np.random.default_rng(7)
    noise = (0.1 * rng.standard_normal(30 * RATE)).astype(np.float32)
    result = sync_camera(noise, ref, RATE, name="noise")
    assert result.status in ("no_match", "insufficient", "ambiguous")
    assert result.clock is None


def test_camera_outside_reference_coverage(demo_audio):
    """A camera that recorded only after the reference stopped: no shared content."""
    ref, _ = demo_audio
    clock = AffineClock(90.0, 1.0)  # starts 26 s after the 64 s reference ends
    from cueframe.synth.stage_audio import camera_audio

    cam = camera_audio(ref, RATE, clock, 30.0, snr_db=24.0, tone=0.0, seed=1)
    result = sync_camera(cam, ref, RATE, name="late")
    assert result.status in ("silent", "no_match", "insufficient")
    assert result.clock is None


def test_periodic_reference_is_ambiguous():
    """The same chirp every second and nothing else shared: any whole-second lag matches."""
    chirps = [Chirp(t, 600.0, 3000.0, 0.2, 0.5) for t in np.arange(1.0, 29.0, 1.0)]
    ref = reference_track(30.0, RATE, chirps, seed=1, bed_rms=0.0)
    from cueframe.synth.stage_audio import camera_audio

    cam = camera_audio(ref, RATE, AffineClock(2.0, 1.0), 26.0, snr_db=20.0, tone=0.0, seed=2)
    result = sync_camera(cam, ref, RATE, name="periodic")
    assert result.status == "ambiguous", (result.status, result.reason)
    assert result.clock is None


def test_shared_ambience_resolves_periodic_content():
    """With a shared (non-periodic) room-tone bed the whitened correlation picks the true lag.

    This is why the ambiguity flag is a heuristic: it fires when nothing unique is shared.
    """
    chirps = [Chirp(t, 600.0, 3000.0, 0.2, 0.5) for t in np.arange(1.0, 29.0, 1.0)]
    ref = reference_track(30.0, RATE, chirps, seed=1, bed_rms=0.01)
    from cueframe.synth.stage_audio import camera_audio

    cam = camera_audio(ref, RATE, AffineClock(2.0, 1.0), 26.0, snr_db=30.0, tone=0.0, seed=2)
    result = sync_camera(cam, ref, RATE, name="periodic+bed")
    assert result.status == "ok"
    assert abs(result.clock.offset - 2.0) * 1000 < 1.0


def test_sync_does_not_import_truth():
    """The estimator module must not know about the generator or its manifest."""
    import cueframe.sync as sync_mod

    src = open(sync_mod.__file__, encoding="utf-8").read()  # noqa: SIM115
    for token in ("import json", "open(", "Path(", "synth", "true_clock", "flash"):
        assert token not in src, token
