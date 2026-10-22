"""Timing units and timeline invariants, checked against hand-computed values."""

from fractions import Fraction

import pytest

from cueframe.timeline import (
    Segment,
    Timeline,
    TimelineError,
    frame_count,
    frame_to_time,
    parse_fps,
    time_to_frame,
    time_to_frame_floor,
    timeline_from_boundaries,
    timeline_from_spans,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("25", Fraction(25)),
        (30, Fraction(30)),
        ("30000/1001", Fraction(30000, 1001)),
        ("29.97", Fraction(30000, 1001)),
        (23.976, Fraction(24000, 1001)),
        ("59.94", Fraction(60000, 1001)),
        ("24", Fraction(24)),
    ],
)
def test_parse_fps(text, expected):
    assert parse_fps(text) == expected


@pytest.mark.parametrize("bad", ["0", "-25", "abc", "1/0", True, "2000"])
def test_parse_fps_rejects(bad):
    with pytest.raises(TimelineError):
        parse_fps(bad)


def test_time_to_frame_hand_values():
    fps = Fraction(25)
    # frame 0 covers [0, 0.04); boundaries round half up
    assert time_to_frame(0.0, fps) == 0
    assert time_to_frame(0.039, fps) == 1  # 0.975 frames → nearest boundary is 1
    assert time_to_frame(0.019, fps) == 0  # 0.475 → 0
    assert time_to_frame(0.02, fps) == 1  # exactly half a frame rounds up
    assert time_to_frame(3.2, fps) == 80
    assert time_to_frame(16.8, fps) == 420
    assert time_to_frame_floor(0.039, fps) == 0
    assert time_to_frame_floor(0.04, fps) == 1


def test_ntsc_rates_are_exact():
    fps = Fraction(30000, 1001)
    # 1001 frames at 29.97 take exactly 33.3666... s; floats would drift, Fractions do not
    assert frame_to_time(30000, fps) == 1001.0
    assert time_to_frame(1001.0, fps) == 30000
    assert frame_count(1001.0, fps) == 30000


def test_frame_count_ceil_with_tolerance():
    fps = Fraction(25)
    assert frame_count(1.0, fps) == 25
    assert frame_count(1.001, fps) == 26
    assert frame_count(1.0000000001, fps) == 25  # within 1 µs of a boundary
    assert frame_count(0.001, fps) == 1
    with pytest.raises(TimelineError):
        frame_count(0.0, fps)


def test_timeline_invariants_enforced():
    fps = Fraction(25)
    Timeline(fps, 10, (Segment(0, 4, "a"), Segment(4, 10, "b")))
    with pytest.raises(TimelineError, match="not 0"):
        Timeline(fps, 10, (Segment(1, 10, "a"),))
    with pytest.raises(TimelineError, match="expected 10"):
        Timeline(fps, 10, (Segment(0, 9, "a"),))
    with pytest.raises(TimelineError, match="segment 0 ends at frame 4 but segment 1 starts at 5"):
        Timeline(fps, 10, (Segment(0, 4, "a"), Segment(5, 10, "b")))
    with pytest.raises(TimelineError, match="segment 0 ends at frame 5 but segment 1 starts at 4"):
        Timeline(fps, 10, (Segment(0, 5, "a"), Segment(4, 10, "b")))
    with pytest.raises(TimelineError, match="has 0 frames"):
        Timeline(fps, 10, (Segment(0, 0, "a"), Segment(0, 10, "b")))


def test_timeline_at_and_progress():
    fps = Fraction(25)
    tl = Timeline(fps, 10, (Segment(0, 4, "a"), Segment(4, 7, None), Segment(7, 10, "c")))
    assert tl.at(0).payload == "a"
    assert tl.at(3).payload == "a"
    assert tl.at(4).payload is None
    assert tl.at(6).payload is None
    assert tl.at(7).payload == "c"
    assert tl.at(9).payload == "c"
    with pytest.raises(IndexError):
        tl.at(10)
    seg = tl.at(7)
    assert seg.progress(7) == 0.0
    assert seg.progress(9) == 1.0
    assert [s.payload for s in tl.scheduled()] == ["a", "c"]
    assert tl.duration == 0.4


def test_timeline_from_spans_hand_computed():
    fps = Fraction(25)
    # audio of 2.0 s = 50 frames; cues at 0.0-0.5, 0.5-1.1, gap, 1.5-2.0
    # 0.5 s = 12.5 frames → 13 (half up); 1.1 s = 27.5 → 28; 1.5 s = 37.5 → 38
    tl = timeline_from_spans([(0.0, 0.5, "x"), (0.5, 1.1, "y"), (1.5, 2.0, "z")], fps, 50)
    got = [(s.f0, s.f1, s.payload) for s in tl.segments]
    assert got == [(0, 13, "x"), (13, 28, "y"), (28, 38, None), (38, 50, "z")]


def test_half_frame_boundary_rule():
    """0.5 s at 25 fps is exactly frame 12.5; round-half-up puts the boundary at frame 13."""
    fps = Fraction(25)
    tl = timeline_from_spans([(0.0, 0.5, "x"), (0.5, 1.0, "y")], fps, 25)
    assert [(s.f0, s.f1) for s in tl.segments] == [(0, 13), (13, 25)]


def test_timeline_from_spans_rejections():
    fps = Fraction(25)
    with pytest.raises(TimelineError, match="overlaps"):
        timeline_from_spans([(0.0, 1.0, "a"), (0.9, 2.0, "b")], fps, 50)
    with pytest.raises(TimelineError, match="negative"):
        timeline_from_spans([(-0.1, 1.0, "a")], fps, 50)
    with pytest.raises(TimelineError, match="not after"):
        timeline_from_spans([(1.0, 1.0, "a")], fps, 50)
    with pytest.raises(TimelineError, match="beyond the last frame"):
        timeline_from_spans([(0.0, 2.04, "a")], fps, 50)
    with pytest.raises(TimelineError, match="covers no whole frame"):
        timeline_from_spans([(0.0, 0.01, "a")], fps, 50)
    with pytest.raises(TimelineError, match="non-finite"):
        timeline_from_spans([(0.0, float("nan"), "a")], fps, 50)


def test_timeline_from_boundaries():
    fps = Fraction(25)
    tl = timeline_from_boundaries([10, 20], ["a", "b", "c"], fps, 30)
    assert [(s.f0, s.f1, s.payload) for s in tl.segments] == [(0, 10, "a"), (10, 20, "b"),
                                                              (20, 30, "c")]
    with pytest.raises(TimelineError):
        timeline_from_boundaries([20, 10], ["a", "b", "c"], fps, 30)
    with pytest.raises(TimelineError):
        timeline_from_boundaries([10], ["a"], fps, 30)
