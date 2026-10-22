"""Affine clock: sign and units pinned with hand-computed examples."""

import pytest

from cueframe.clock import AffineClock


def test_forward_inverse_roundtrip():
    c = AffineClock(offset=2.5, rate=1.0005)
    for t in (0.0, 1.0, 37.25, 600.0):
        assert c.to_cam(c.to_ref(t)) == pytest.approx(t, abs=1e-9)
    inv = c.inverse()
    assert inv.to_ref(c.to_ref(12.0)) == pytest.approx(12.0, abs=1e-9)


def test_drift_sign_and_units():
    # rate 1.00005 = +50 ppm: the camera clock runs slow, so one camera second covers
    # 1.00005 reference seconds; after 800 camera seconds it is 40 ms (one 25 fps frame) behind.
    c = AffineClock.from_drift_ppm(0.0, 50.0)
    assert c.rate == pytest.approx(1.00005)
    assert c.drift_ppm == pytest.approx(50.0)
    assert c.to_ref(800.0) - 800.0 == pytest.approx(0.040, abs=1e-9)
    # negative drift: camera fast, reference time lags camera time
    d = AffineClock.from_drift_ppm(-3.0, -2600.0)
    assert d.to_ref(10.0) == pytest.approx(-3.0 + 10.0 * 0.9974)


def test_offset_meaning():
    # offset is the reference time at which the camera clock reads zero
    c = AffineClock(offset=4.1, rate=1.0)
    assert c.to_ref(0.0) == 4.1
    assert c.to_cam(4.1) == 0.0


def test_clap_sync_matches_at_the_clap_and_drifts_elsewhere():
    c = AffineClock.from_drift_ppm(-2.7, 3200.0)
    clap = c.clap_sync(1.25)
    assert clap.rate == 1.0
    assert clap.to_ref(1.25) == pytest.approx(c.to_ref(1.25))
    # 60 s later the clap sync is 3200 ppm * 58.75 s = 188 ms off
    assert c.to_ref(60.0) - clap.to_ref(60.0) == pytest.approx(0.0032 * 58.75, rel=1e-6)


def test_json_roundtrip_and_validation():
    c = AffineClock(1.25, 1.0032)
    assert AffineClock.from_json(c.to_json()) == c
    assert c.to_json()["drift_ppm"] == pytest.approx(3200.0)
    with pytest.raises(ValueError):
        AffineClock(0.0, 2.5)
    with pytest.raises(ValueError):
        AffineClock(0.0, 0.0)
