"""Held-out verification has teeth: a correct clock passes, a drift-less clock fails late."""

import pytest

from cueframe.clock import AffineClock
from cueframe.sync import SyncConfig, sync_camera
from cueframe.synth.generate import audio_fixture
from cueframe.verify import heldout_centers, verify_camera

RATE = 16000
CLOCKS = {"A": AffineClock.from_drift_ppm(1.3, 0.0), "B": AffineClock.from_drift_ppm(-2.7, 3200.0)}


@pytest.fixture(scope="module")
def fixture():
    ref, cams = audio_fixture(64.0, CLOCKS, sample_rate=RATE, seed=3)
    syncs = {n: sync_camera(cams[n], ref, RATE, name=n) for n in cams}
    return ref, cams, syncs


def test_heldout_windows_are_disjoint_from_fit_windows_and_hit_start_middle_end():
    from cueframe.sync import fine_centers

    cfg = SyncConfig()
    fit = fine_centers(60.0, cfg)
    held = heldout_centers(60.0, cfg)
    assert all(abs(h - f) >= cfg.window / 4 for h in held for f in fit)
    assert held[0] <= cfg.window / 2 + cfg.stride / 2
    assert held[-1] >= 60.0 - cfg.window / 2 - cfg.stride / 2
    assert any(abs(h - 30.0) <= cfg.stride / 2 for h in held)


def test_estimated_clock_passes_heldout(fixture):
    ref, cams, syncs = fixture
    for name in cams:
        s = syncs[name]
        assert s.status == "ok"
        v = verify_camera(cams[name], ref, RATE, s.clock, name=name, tolerance_ms=2.0)
        assert v.status == "pass", v.to_json()
        assert v.n_confident >= 3
        assert v.max_abs_ms < 2.0
        # the explicit start/middle/end windows are all confident
        ts = [w.t_cam for w in v.windows if w.used]
        cam_dur = len(cams[name]) / RATE
        assert min(ts) < 4.0 and max(ts) > cam_dur - 4.0


def test_drift_zeroed_clock_fails_at_late_windows(fixture):
    """Negative control: keep the correct offset but drop the drift of camera B (3200 ppm)."""
    ref, cams, syncs = fixture
    good = syncs["B"].clock
    corrupted = AffineClock(good.offset, 1.0)
    v = verify_camera(cams["B"], ref, RATE, corrupted, name="B", tolerance_ms=2.0)
    assert v.status == "fail"
    assert v.worst_t_cam is not None and v.worst_t_cam > 30.0  # error grows with time
    assert v.max_abs_ms > 50.0


def test_no_confident_window_is_distinct_outcome(fixture):
    import numpy as np

    ref, cams, syncs = fixture
    v = verify_camera(np.zeros(30 * RATE, dtype=np.float32), ref, RATE, syncs["A"].clock,
                      name="mute")
    assert v.status == "no_confident_window"
    assert v.n_confident == 0
