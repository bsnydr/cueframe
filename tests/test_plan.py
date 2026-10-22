"""Shot plan invariants, checked without any media files."""

from fractions import Fraction

import pytest

from cueframe.clock import AffineClock
from cueframe.plan import Camera, PlanError, coverage, make_plan
from cueframe.timeline import frame_to_time


def cam(name, fps, offset, drift_ppm, duration):
    return Camera(name, f"/nonexistent/{name}.mov", Fraction(fps), duration, 640, 360,
                  AffineClock.from_drift_ppm(offset, drift_ppm))


CAMS = [cam("A", 25, 1.3, 0.0, 60.0), cam("B", 25, -2.7, 3200.0, 66.0),
        cam("C", 30, 4.1, -2600.0, 58.0)]


def test_coverage_is_intersection():
    start, end = coverage(CAMS)
    assert start == pytest.approx(4.1)  # C started last
    assert end == pytest.approx(min(c.ref_end for c in CAMS))
    assert end == pytest.approx(1.3 + 60.0)  # A ends first at reference 61.3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plan_invariants(seed):
    plan = make_plan(CAMS, fps=25, size=(640, 360), seed=seed, min_shot=1.5)
    segs = plan.timeline.segments
    assert segs[0].f0 == 0
    assert segs[-1].f1 == plan.n_frames
    assert all(a.f1 == b.f0 for a, b in zip(segs, segs[1:], strict=False))
    assert sum(s.n for s in segs) == plan.n_frames
    assert all(frame_to_time(s.n, plan.fps) >= 1.5 for s in segs)
    assert all(a.payload.cam != b.payload.cam for a, b in zip(segs, segs[1:], strict=False))
    for seg in segs:
        shot = seg.payload
        c = {x.name: x for x in CAMS}[shot.cam]
        # source start sits on the camera's own frame grid, within half a source frame
        assert shot.src_start == frame_to_time(shot.src_frame, c.fps)
        assert abs(shot.src_err_ms) <= 500.0 / float(c.fps) + 1e-9
        # fully inside the recording
        assert shot.src_start >= 0
        assert shot.src_start + frame_to_time(seg.n, plan.fps) / c.clock.rate <= c.duration
        # the reference time of the first frame maps back exactly
        assert shot.ref_t0 == pytest.approx(plan.timeline_start + frame_to_time(seg.f0, plan.fps))
        assert shot.src_t == pytest.approx(c.clock.to_cam(shot.ref_t0))
    assert {s.payload.framing for s in segs} == {"full", "tight"}


def test_30fps_camera_quantised_on_its_own_grid():
    plan = make_plan(CAMS, fps=25, size=(640, 360), seed=0)
    c_shots = [s.payload for s in plan.timeline.segments if s.payload.cam == "C"]
    assert c_shots
    for shot in c_shots:
        assert shot.src_fps == Fraction(30)
        assert (shot.src_start * 30) == pytest.approx(round(shot.src_start * 30))
        assert abs(shot.src_err_ms) <= 1000 / 30 / 2


def test_timeline_outside_coverage_is_rejected():
    with pytest.raises(PlanError, match="exceeds common coverage"):
        make_plan(CAMS, fps=25, size=(640, 360), start=0.0, duration=10.0)
    with pytest.raises(PlanError, match="exceeds common coverage"):
        make_plan(CAMS, fps=25, size=(640, 360), start=50.0, duration=20.0)


def test_non_overlapping_cameras_rejected():
    with pytest.raises(PlanError, match="do not all overlap"):
        make_plan([cam("A", 25, 0.0, 0.0, 10.0), cam("B", 25, 20.0, 0.0, 10.0)],
                  fps=25, size=(640, 360))


def test_duplicate_names_and_empty_rejected():
    with pytest.raises(PlanError, match="duplicate"):
        make_plan([CAMS[0], CAMS[0]], fps=25, size=(640, 360))
    with pytest.raises(PlanError, match="no cameras"):
        make_plan([], fps=25, size=(640, 360))


def test_plan_json_roundtrip(tmp_path):
    from cueframe.plan import Plan

    plan = make_plan(CAMS, fps=25, size=(640, 360), seed=1)
    p = tmp_path / "plan.json"
    plan.save(p)
    again = Plan.load(p)
    assert again.n_frames == plan.n_frames
    assert again.timeline_start == plan.timeline_start
    assert [(s.f0, s.f1, s.payload.cam, s.payload.src_frame) for s in again.timeline.segments] == [
        (s.f0, s.f1, s.payload.cam, s.payload.src_frame) for s in plan.timeline.segments
    ]


def test_plan_does_not_import_truth():
    import cueframe.plan as plan_mod

    src = open(plan_mod.__file__, encoding="utf-8").read()  # noqa: SIM115
    assert "manifest" not in src.lower() or "never reads the generator's manifest" in src
    assert "synth" not in src
    assert "flash" not in src.lower() or "not on any audio event" in src
