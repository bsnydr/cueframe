"""Hand-authored cues: the audio-only producer for the timeline.

A cue file is JSON::

    {
      "fps": 25,
      "size": [960, 540],
      "crossfade": 0.0,
      "note": "scene annotations, not transcribed speech",
      "cues": [
        {"start": 0.0, "end": 3.2, "scene": "title", "text": "A quiet opening"},
        {"start": 3.2, "end": 7.0, "scene": "pulse", "text": "Rings follow the level"}
      ]
    }

Rules (all enforced by :func:`load_cues` / :func:`validate_cues`):

* ``fps`` is an exact frame rate ("25", "30000/1001", 29.97).
* Cues are sorted by ``start``, do not overlap, and satisfy
  ``0 <= start < end <= duration`` where duration comes from the audio file.
* ``scene`` is a name that must exist in the scene set in use (built-in or custom);
  JSON never names Python code, only scene names.
* Each cue must cover at least one whole output frame after quantisation.
* Gaps between cues are permitted and explicit: those frames carry no cue and are drawn
  by the scene set's ``"idle"`` scene.
* Any extra keys inside a cue are preserved in ``Cue.extra`` for custom scenes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from cueframe.timeline import Timeline, TimelineError, frame_count, parse_fps, timeline_from_spans

MAX_TEXT_LEN = 200
MAX_CUES = 10_000


class CueError(ValueError):
    """A cue file is malformed. The message names the cue index where possible."""


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    scene: str
    text: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class CueSheet:
    fps: Fraction
    size: tuple[int, int]
    cues: tuple[Cue, ...]
    crossfade: float = 0.0
    note: str = ""

    def to_timeline(self, duration: float) -> Timeline[Cue]:
        """Quantise cue boundaries to frames; ``duration`` is the audio duration in seconds."""
        n = frame_count(duration, self.fps)
        for i, cue in enumerate(self.cues):
            if cue.end > duration + 1e-9:
                raise CueError(
                    f"cue {i}: end {cue.end}s is beyond the audio duration {duration:.6f}s"
                )
        try:
            return timeline_from_spans([(c.start, c.end, c) for c in self.cues], self.fps, n)
        except TimelineError as exc:
            raise CueError(str(exc).replace("span", "cue", 1)) from exc


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CueError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise CueError(f"{what} must be finite, got {value!r}")
    return float(value)


def parse_cues(doc: Any, *, source: str = "<cues>") -> CueSheet:
    """Validate a decoded JSON document and return a :class:`CueSheet`.

    Structural validation only; scene-name validation needs the scene set and happens in
    :func:`validate_scene_names`; duration validation happens in :meth:`CueSheet.to_timeline`.
    """
    if not isinstance(doc, dict):
        raise CueError(f"{source}: top level must be an object")
    unknown = set(doc) - {"fps", "size", "crossfade", "note", "cues"}
    if unknown:
        raise CueError(f"{source}: unknown top-level keys {sorted(unknown)}")
    if "fps" not in doc or "cues" not in doc:
        raise CueError(f"{source}: 'fps' and 'cues' are required")
    try:
        fps = parse_fps(doc["fps"])
    except TimelineError as exc:
        raise CueError(f"{source}: {exc}") from exc

    size_raw = doc.get("size", [960, 540])
    if (
        not isinstance(size_raw, list)
        or len(size_raw) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in size_raw)
    ):
        raise CueError(f"{source}: size must be [width, height] integers")
    width, height = size_raw
    if not (16 <= width <= 7680 and 16 <= height <= 4320) or width % 2 or height % 2:
        raise CueError(f"{source}: size {width}x{height} must be even and within 16..7680x4320")

    crossfade = _num(doc.get("crossfade", 0.0), f"{source}: crossfade")
    if not 0 <= crossfade <= 5:
        raise CueError(f"{source}: crossfade must be within 0..5 seconds")
    note = doc.get("note", "")
    if not isinstance(note, str):
        raise CueError(f"{source}: note must be a string")

    raw_cues = doc["cues"]
    if not isinstance(raw_cues, list):
        raise CueError(f"{source}: 'cues' must be a list")
    if len(raw_cues) > MAX_CUES:
        raise CueError(f"{source}: too many cues ({len(raw_cues)} > {MAX_CUES})")

    cues: list[Cue] = []
    prev_end = 0.0
    for i, raw in enumerate(raw_cues):
        if not isinstance(raw, dict):
            raise CueError(f"cue {i}: must be an object")
        for key in ("start", "end", "scene"):
            if key not in raw:
                raise CueError(f"cue {i}: missing '{key}'")
        start = _num(raw["start"], f"cue {i}: start")
        end = _num(raw["end"], f"cue {i}: end")
        scene = raw["scene"]
        if not isinstance(scene, str) or not scene or len(scene) > 64:
            raise CueError(f"cue {i}: scene must be a non-empty name up to 64 characters")
        text = raw.get("text", "")
        if not isinstance(text, str):
            raise CueError(f"cue {i}: text must be a string")
        if len(text) > MAX_TEXT_LEN:
            raise CueError(f"cue {i}: text longer than {MAX_TEXT_LEN} characters")
        if start < 0:
            raise CueError(f"cue {i}: start {start} is negative")
        if end <= start:
            raise CueError(f"cue {i}: end {end} is not after start {start}")
        if start < prev_end:
            raise CueError(
                f"cue {i}: start {start} overlaps the previous cue (ends {prev_end}); "
                "cues must be sorted and non-overlapping"
            )
        extra = {k: v for k, v in raw.items() if k not in {"start", "end", "scene", "text"}}
        cues.append(Cue(start, end, scene, text, extra))
        prev_end = end
    return CueSheet(fps, (width, height), tuple(cues), crossfade, note)


def validate_scene_names(sheet: CueSheet, available: Sequence[str] | Mapping[str, Any]) -> None:
    names = set(available)
    if "idle" not in names:
        raise CueError("scene set has no 'idle' scene (needed for gaps between cues)")
    for i, cue in enumerate(sheet.cues):
        if cue.scene not in names:
            raise CueError(
                f"cue {i}: unknown scene {cue.scene!r}; available: {sorted(names)}"
            )


def load_cues(path: str | Path) -> CueSheet:
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CueError(f"{p}: invalid JSON: {exc}") from exc
    return parse_cues(doc, source=str(p))


def dump_cues(sheet: CueSheet, path: str | Path) -> None:
    doc = {
        "fps": str(sheet.fps) if sheet.fps.denominator != 1 else sheet.fps.numerator,
        "size": list(sheet.size),
        "crossfade": sheet.crossfade,
        "note": sheet.note,
        "cues": [
            {"start": c.start, "end": c.end, "scene": c.scene, "text": c.text, **dict(c.extra)}
            for c in sheet.cues
        ],
    }
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
