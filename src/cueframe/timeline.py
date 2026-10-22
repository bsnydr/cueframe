"""Timing units and the frame-exact timeline shared by every cueframe path.

Conventions
-----------
* Times are seconds as ``float``; frame rates are exact :class:`fractions.Fraction`.
* Frame ``f`` covers the half-open time interval ``[f / fps, (f + 1) / fps)``.
* A :class:`Timeline` is a list of contiguous :class:`Segment` objects. Continuity is a
  construction invariant: ``segments[0].f0 == 0``, ``segments[i].f1 == segments[i + 1].f0``
  and ``segments[-1].f1 == n_frames``. There are therefore no gaps and no overlaps by
  definition; "nothing scheduled here" is represented by a segment whose payload is ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, TypeVar

T = TypeVar("T")


class TimelineError(ValueError):
    """Raised for invalid timing input (bad frame rate, broken segment chain, ...)."""


def parse_fps(value: str | int | float | Fraction) -> Fraction:
    """Parse a frame rate into an exact Fraction.

    Accepts integers, floats, "25", "30000/1001", "29.97" (rounded to the nearest
    broadcast rate) and Fractions.
    """
    if isinstance(value, Fraction):
        fps = value
    elif isinstance(value, bool):
        raise TimelineError("fps must be a number, not bool")
    elif isinstance(value, int):
        fps = Fraction(value)
    elif isinstance(value, float):
        fps = _snap_broadcast(Fraction(value).limit_denominator(1001))
    elif isinstance(value, str):
        text = value.strip()
        try:
            if "/" in text:
                num, den = text.split("/")
                fps = Fraction(int(num), int(den))
            else:
                fps = _snap_broadcast(Fraction(text).limit_denominator(1001))
        except (ValueError, ZeroDivisionError) as exc:
            raise TimelineError(f"unparseable fps {value!r}") from exc
    else:
        raise TimelineError(f"unsupported fps type {type(value).__name__}")
    if fps <= 0 or fps > 1000:
        raise TimelineError(f"fps out of range: {fps}")
    return fps


def _snap_broadcast(fps: Fraction) -> Fraction:
    for exact in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001)):
        if abs(float(fps) - float(exact)) < 1e-3:
            return exact
    return fps


def time_to_frame(t: float, fps: Fraction) -> int:
    """Nearest frame boundary to time ``t`` (round half up, computed exactly)."""
    if not math.isfinite(t):
        raise TimelineError(f"non-finite time {t!r}")
    x = Fraction(t) * fps
    return math.floor(x + Fraction(1, 2))


def time_to_frame_floor(t: float, fps: Fraction) -> int:
    """Frame that contains time ``t``."""
    if not math.isfinite(t):
        raise TimelineError(f"non-finite time {t!r}")
    return math.floor(Fraction(t) * fps)


def frame_to_time(f: int, fps: Fraction) -> float:
    """Start time of frame ``f`` in seconds."""
    return float(Fraction(f) / fps)


def frame_count(duration: float, fps: Fraction) -> int:
    """Number of frames needed to cover ``duration`` seconds (ceil, with a tiny tolerance).

    A duration within one millionth of a frame boundary counts as landing on it.
    """
    if duration <= 0:
        raise TimelineError(f"duration must be positive, got {duration}")
    x = Fraction(duration) * fps
    n = math.ceil(x - Fraction(1, 1_000_000))
    return max(n, 1)


@dataclass(frozen=True)
class Segment(Generic[T]):
    """A contiguous run of output frames ``[f0, f1)`` carrying a payload."""

    f0: int
    f1: int
    payload: T | None

    @property
    def n(self) -> int:
        return self.f1 - self.f0

    def contains(self, frame: int) -> bool:
        return self.f0 <= frame < self.f1

    def progress(self, frame: int) -> float:
        """0.0 at the first frame of the segment, approaching 1.0 at the last frame."""
        if self.n <= 1:
            return 0.0
        return (frame - self.f0) / (self.n - 1)


@dataclass(frozen=True)
class Timeline(Generic[T]):
    """Contiguous segments covering exactly ``n_frames`` output frames at ``fps``."""

    fps: Fraction
    n_frames: int
    segments: tuple[Segment[T], ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_frames <= 0:
            raise TimelineError(f"n_frames must be positive, got {self.n_frames}")
        if not self.segments:
            raise TimelineError("timeline has no segments")
        if self.segments[0].f0 != 0:
            raise TimelineError(f"first segment starts at frame {self.segments[0].f0}, not 0")
        if self.segments[-1].f1 != self.n_frames:
            raise TimelineError(
                f"last segment ends at frame {self.segments[-1].f1}, expected {self.n_frames}"
            )
        for i, seg in enumerate(self.segments):
            if seg.n <= 0:
                raise TimelineError(f"segment {i} has {seg.n} frames")
            if i + 1 < len(self.segments) and seg.f1 != self.segments[i + 1].f0:
                raise TimelineError(
                    f"segment {i} ends at frame {seg.f1} but segment {i + 1} starts at "
                    f"{self.segments[i + 1].f0}"
                )

    @property
    def duration(self) -> float:
        return frame_to_time(self.n_frames, self.fps)

    def at(self, frame: int) -> Segment[T]:
        """Segment covering ``frame`` (binary search)."""
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} outside [0, {self.n_frames})")
        lo, hi = 0, len(self.segments) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.segments[mid].f1 <= frame:
                lo = mid + 1
            else:
                hi = mid
        return self.segments[lo]

    def scheduled(self) -> Iterable[Segment[T]]:
        """Segments with a payload (skips ``None`` filler)."""
        return (s for s in self.segments if s.payload is not None)


def timeline_from_spans(
    spans: Sequence[tuple[float, float, T]],
    fps: Fraction,
    n_frames: int,
) -> Timeline[T]:
    """Build a continuous timeline from (start, end, payload) spans in seconds.

    Spans must be sorted and non-overlapping (``end[i] <= start[i + 1]``) and lie within
    ``[0, n_frames / fps]``. Boundaries are quantised with :func:`time_to_frame`. Gaps
    become ``None`` segments. A span that quantises to zero frames is an error, because it
    would silently vanish from the output.
    """
    segments: list[Segment[T]] = []
    cursor = 0
    prev_end = 0.0
    for i, (start, end, payload) in enumerate(spans):
        if not (math.isfinite(start) and math.isfinite(end)):
            raise TimelineError(f"span {i}: non-finite time")
        if start < 0:
            raise TimelineError(f"span {i}: start {start} is negative")
        if end <= start:
            raise TimelineError(f"span {i}: end {end} is not after start {start}")
        if start < prev_end:
            raise TimelineError(f"span {i}: start {start} overlaps previous span ending {prev_end}")
        f0 = time_to_frame(start, fps)
        f1 = time_to_frame(end, fps)
        if f1 > n_frames:
            raise TimelineError(
                f"span {i}: end {end}s is frame {f1}, beyond the last frame {n_frames}"
            )
        if f1 <= f0:
            raise TimelineError(
                f"span {i}: {start}s-{end}s covers no whole frame at {fps} fps "
                f"(both quantise to frame {f0})"
            )
        if f0 > cursor:
            segments.append(Segment(cursor, f0, None))
        segments.append(Segment(f0, f1, payload))
        cursor = f1
        prev_end = end
    if cursor < n_frames:
        segments.append(Segment(cursor, n_frames, None))
    return Timeline(fps, n_frames, tuple(segments))


def timeline_from_boundaries(
    boundaries: Sequence[int],
    payloads: Sequence[T],
    fps: Fraction,
    n_frames: int,
) -> Timeline[T]:
    """Build a gap-free timeline from strictly increasing cut frames.

    ``boundaries`` are the frames at which a new segment starts, excluding 0 and
    ``n_frames``; ``payloads`` has one entry per segment (``len(boundaries) + 1``).
    """
    if len(payloads) != len(boundaries) + 1:
        raise TimelineError("need exactly one payload per segment")
    edges = [0, *boundaries, n_frames]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b <= a:
            raise TimelineError(f"boundaries must be strictly increasing: {a} -> {b}")
    segs = tuple(
        Segment(a, b, p) for a, b, p in zip(edges[:-1], edges[1:], payloads, strict=True)
    )
    return Timeline(fps, n_frames, segs)
