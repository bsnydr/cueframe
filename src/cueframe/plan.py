"""Shot planning: estimated camera clocks → a frame-exact :class:`Timeline` of shots.

The planner reads ``sync.json`` (estimated clocks) and the cameras' probed frame rates and
durations. It never reads the generator's manifest. Cuts fall on the planner's own beat
pattern, not on any audio event, so the pixel oracle stays independent.

Each shot maps its first output frame back to a source frame:

    ref_t0     = timeline_start + f0 / fps
    src_t      = clock.to_cam(ref_t0)
    src_frame  = round(src_t * src_fps)         # quantised to the *source* frame grid
    src_err_ms = (src_frame / src_fps - src_t) * 1000

and is rejected (not padded) if any part of it lies outside the camera's recording.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from cueframe.clock import AffineClock
from cueframe.timeline import (
    Timeline,
    TimelineError,
    frame_count,
    frame_to_time,
    parse_fps,
    time_to_frame,
    timeline_from_boundaries,
)

FRAMINGS = ("full", "tight")
TIGHT_ZOOM = 1.4


class PlanError(ValueError):
    pass


def _size(width: int, height: int) -> None:
    if (type(width) is not int or type(height) is not int
            or not (16 <= width <= 7680 and 16 <= height <= 4320)
            or width % 2 or height % 2):
        raise PlanError("size must be even integers within 16..7680 x 16..4320")


@dataclass(frozen=True)
class Camera:
    name: str
    path: str
    fps: Fraction
    duration: float  # camera seconds
    width: int
    height: int
    clock: AffineClock  # camera → reference

    def __post_init__(self) -> None:
        _size(self.width, self.height)
        if not isinstance(self.name, str) or not self.name:
            raise PlanError("camera name must be a nonempty string")
        if not isinstance(self.path, str) or not self.path:
            raise PlanError("camera path must be a nonempty string")
        parse_fps(self.fps)
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise PlanError("camera duration must be finite and positive")

    @property
    def ref_start(self) -> float:
        return self.clock.to_ref(0.0)

    @property
    def ref_end(self) -> float:
        return self.clock.to_ref(self.duration)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "fps": str(self.fps),
            "duration_s": self.duration,
            "width": self.width,
            "height": self.height,
            "clock": self.clock.to_json(),
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Camera:
        return cls(
            doc["name"],
            doc["path"],
            parse_fps(doc["fps"]),
            float(doc["duration_s"]),
            doc["width"],
            doc["height"],
            AffineClock.from_json(doc["clock"]),
        )


@dataclass(frozen=True)
class Shot:
    cam: str
    framing: str
    ref_t0: float
    src_t: float
    src_frame: int
    src_fps: Fraction

    @property
    def src_start(self) -> float:
        return frame_to_time(self.src_frame, self.src_fps)

    @property
    def src_err_ms(self) -> float:
        return (self.src_start - self.src_t) * 1000.0


@dataclass(frozen=True)
class Plan:
    fps: Fraction
    width: int
    height: int
    timeline_start: float  # reference seconds of output frame 0
    cameras: tuple[Camera, ...]
    timeline: Timeline[Shot]

    def __post_init__(self) -> None:
        _size(self.width, self.height)
        if not math.isfinite(self.timeline_start) or self.timeline_start < 0:
            raise PlanError("timeline start must be finite and nonnegative")
        if self.timeline.fps != self.fps:
            raise PlanError("plan and timeline frame rates differ")
        cams = {c.name: c for c in self.cameras}
        if not cams or len(cams) != len(self.cameras):
            raise PlanError("plan needs uniquely named cameras")
        for i, seg in enumerate(self.timeline.segments):
            shot = seg.payload
            if shot is None or shot.cam not in cams or shot.framing not in FRAMINGS:
                raise PlanError(f"shot {i}: unknown camera or framing")
            cam = cams[shot.cam]
            if type(shot.src_frame) is not int or shot.src_frame < 0 or shot.src_fps != cam.fps:
                raise PlanError(f"shot {i}: invalid source frame or frame rate")
            expected_ref = self.timeline_start + frame_to_time(seg.f0, self.fps)
            if (not math.isfinite(shot.ref_t0) or not math.isfinite(shot.src_t)
                    or abs(shot.ref_t0 - expected_ref) > 1e-5
                    or abs(shot.src_t - cam.clock.to_cam(expected_ref)) > 1e-5
                    or time_to_frame(shot.src_t, cam.fps) != shot.src_frame):
                raise PlanError(f"shot {i}: inconsistent clock mapping")
            end = shot.src_start + frame_to_time(seg.n, self.fps) / cam.clock.rate
            if end > cam.duration + 1e-9:
                raise PlanError(f"shot {i}: source range outside the recording")

    @property
    def n_frames(self) -> int:
        return self.timeline.n_frames

    @property
    def duration(self) -> float:
        return self.timeline.duration

    def to_json(self) -> dict[str, Any]:
        shots = []
        for seg in self.timeline.segments:
            s = seg.payload
            assert s is not None
            shots.append(
                {
                    "f0": seg.f0,
                    "f1": seg.f1,
                    "n": seg.n,
                    "cam": s.cam,
                    "framing": s.framing,
                    "ref_t0": s.ref_t0,
                    "src_t": s.src_t,
                    "src_frame": s.src_frame,
                    "src_fps": str(s.src_fps),
                    "src_start": round(s.src_start, 6),
                    "src_err_ms": round(s.src_err_ms, 3),
                }
            )
        return {
            "fps": str(self.fps),
            "size": [self.width, self.height],
            "n_frames": self.n_frames,
            "duration_s": round(self.duration, 6),
            "timeline_start_ref_s": self.timeline_start,
            "cameras": [c.to_json() for c in self.cameras],
            "shots": shots,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Plan:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        fps = parse_fps(doc["fps"])
        cams = tuple(Camera.from_json(c) for c in doc["cameras"])
        shots = []
        bounds = []
        cursor = 0
        for i, s in enumerate(doc["shots"]):
            if (any(type(s.get(k)) is not int for k in ("f0", "f1", "n", "src_frame"))
                    or s["f0"] != cursor or s["f1"] <= cursor
                    or s["n"] != s["f1"] - s["f0"]):
                raise PlanError(f"shot {i}: invalid or discontinuous frame boundaries")
            cursor = s["f1"]
            if i > 0:
                bounds.append(int(s["f0"]))
            shots.append(
                Shot(
                    s["cam"],
                    s["framing"],
                    float(s["ref_t0"]),
                    float(s["src_t"]),
                    s["src_frame"],
                    parse_fps(s["src_fps"]),
                )
            )
        if type(doc["n_frames"]) is not int or cursor != doc["n_frames"]:
            raise PlanError("shots do not cover the declared frame count")
        timeline = timeline_from_boundaries(bounds, shots, fps, doc["n_frames"])
        return cls(fps, doc["size"][0], doc["size"][1], float(doc["timeline_start_ref_s"]), cams,
                   timeline)


def coverage(cameras: list[Camera]) -> tuple[float, float]:
    """Reference-time interval covered by *every* camera."""
    start = max(c.ref_start for c in cameras)
    end = min(c.ref_end for c in cameras)
    if end <= start:
        raise PlanError(f"cameras do not all overlap (common coverage {start:.3f}..{end:.3f})")
    return start, end


def _beat_lengths(seed: int) -> list[float]:
    patterns = [
        [3.2, 2.4, 4.0, 2.8, 3.6],
        [2.8, 3.6, 2.4, 4.4, 3.2],
        [4.0, 2.4, 3.2, 2.8, 3.6, 2.4],
    ]
    return patterns[seed % len(patterns)]


def make_plan(
    cameras: list[Camera],
    *,
    fps: Fraction | str | int,
    size: tuple[int, int],
    start: float | None = None,
    duration: float | None = None,
    margin: float = 0.5,
    min_shot: float = 1.5,
    seed: int = 0,
) -> Plan:
    """Build a shot plan over the common coverage (or the requested ``start``/``duration``)."""
    if not cameras:
        raise PlanError("no cameras")
    _size(*size)
    if (not math.isfinite(margin) or margin < 0 or not math.isfinite(min_shot)
            or min_shot <= 0 or (start is not None and not math.isfinite(start))
            or (duration is not None and (not math.isfinite(duration) or duration <= 0))):
        raise PlanError("timing values must be finite; duration and min_shot must be positive")
    names = [c.name for c in cameras]
    if len(set(names)) != len(names):
        raise PlanError(f"duplicate camera names {names}")
    fps = parse_fps(fps)
    cov_start, cov_end = coverage(cameras)
    t0 = max(0.0, cov_start) + margin if start is None else start
    t1 = cov_end - margin if duration is None else t0 + duration
    if t0 < cov_start - 1e-9 or t1 > cov_end + 1e-9:
        raise PlanError(
            f"requested timeline {t0:.3f}..{t1:.3f} exceeds common coverage "
            f"{cov_start:.3f}..{cov_end:.3f}"
        )
    if t1 - t0 < min_shot:
        raise PlanError(f"timeline {t1 - t0:.3f}s is shorter than one shot ({min_shot}s)")
    n_frames = frame_count(t1 - t0, fps)

    # cut points on the planner's beat pattern, quantised to output frames
    lengths = _beat_lengths(seed)
    bounds: list[int] = []
    t = 0.0
    i = 0
    while True:
        t += lengths[i % len(lengths)]
        i += 1
        f = time_to_frame(t, fps)
        remaining = frame_to_time(n_frames - f, fps)
        if remaining < min_shot:
            break
        bounds.append(f)
    n_shots = len(bounds) + 1

    # camera rotation without immediate repeats; framing alternates full/tight
    order = list(range(len(cameras)))
    shots: list[Shot] = []
    prev_cam = -1
    edges = [0, *bounds, n_frames]
    for k in range(n_shots):
        cam_idx = order[(k + seed) % len(order)]
        if cam_idx == prev_cam:
            cam_idx = order[(k + seed + 1) % len(order)]
        prev_cam = cam_idx
        cam = cameras[cam_idx]
        framing = FRAMINGS[(k // 2 + seed) % len(FRAMINGS)] if len(cameras) > 1 else "full"
        f0, f1 = edges[k], edges[k + 1]
        ref_t0 = t0 + frame_to_time(f0, fps)
        src_t = cam.clock.to_cam(ref_t0)
        src_frame = time_to_frame(src_t, cam.fps)
        shot_len_cam = frame_to_time(f1 - f0, fps) / cam.clock.rate
        if src_frame < 0 or frame_to_time(src_frame, cam.fps) + shot_len_cam > cam.duration + 1e-9:
            raise PlanError(
                f"shot {k} ({cam.name}) needs source {src_t:.3f}s..{src_t + shot_len_cam:.3f}s, "
                f"outside the recording (0..{cam.duration:.3f}s)"
            )
        shots.append(Shot(cam.name, framing, ref_t0, src_t, src_frame, cam.fps))
    try:
        timeline = timeline_from_boundaries(bounds, shots, fps, n_frames)
    except TimelineError as exc:
        raise PlanError(str(exc)) from exc
    return Plan(fps, size[0], size[1], t0, tuple(cameras), timeline)
