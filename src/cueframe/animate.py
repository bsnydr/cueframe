"""Audio → animation: render a cue timeline over a WAV file with scene renderers.

The scene interface
-------------------
A scene is any callable ``render(ctx: FrameContext, canvas: PIL.Image.Image) -> None`` that
draws frame ``ctx.frame`` onto ``canvas`` (an RGB image of the output size, pre-filled with
black). A *scene set* is a ``dict[str, SceneRenderer]`` that must include ``"idle"`` (used
for frames with no cue). Built-ins live in :mod:`cueframe.scenes`.

Custom scenes are loaded only from a Python file named explicitly on the command line
(``--scenes my_scenes.py``). The module must define ``SCENES`` (a dict of name → callable).
Custom scenes are trusted local code with the same privileges as cueframe itself; cue JSON
can only name scenes, never import code.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image

from cueframe.audio import Audio, Features, file_fingerprint, frame_features, read_wav
from cueframe.cues import Cue, CueError, CueSheet, validate_scene_names
from cueframe.media import check_output, mux_raw_frames, require_tools
from cueframe.timeline import Timeline, frame_to_time, time_to_frame


@dataclass(frozen=True)
class FrameContext:
    """Everything a scene may know about the frame it is drawing."""

    frame: int
    t: float
    fps: Fraction
    width: int
    height: int
    n_frames: int
    cue: Cue | None
    cue_index: int | None
    cue_progress: float  # 0.0 at the cue's first frame → 1.0 at its last frame; 0 if no cue
    cue_frame: int  # frames since the cue (or idle gap) started
    level: float  # smoothed audio level 0..1
    onset: float  # positive level change this frame, 0..1
    seed: int  # per-render seed so scenes can be deterministic yet varied

    @property
    def text(self) -> str:
        return self.cue.text if self.cue else ""

    @property
    def time_left(self) -> float:
        return frame_to_time(self.n_frames - self.frame, self.fps)


SceneRenderer = Callable[[FrameContext, Image.Image], None]
SceneSet = Mapping[str, SceneRenderer]


class SceneLoadError(ValueError):
    pass


def load_scene_module(path: str | Path) -> ModuleType:
    """Import a trusted Python file as a module and return it."""
    p = Path(path).resolve()
    if p.suffix != ".py" or not p.is_file():
        raise SceneLoadError(f"{p}: scene module must be an existing .py file")
    name = f"cueframe_user_scenes_{abs(hash(str(p)))}"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise SceneLoadError(f"{p}: cannot import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SceneLoadError(f"{p}: error while importing: {exc}") from exc
    return module


def scene_set_from_module(module: ModuleType) -> dict[str, SceneRenderer]:
    scenes = getattr(module, "SCENES", None)
    if not isinstance(scenes, dict) or not scenes:
        raise SceneLoadError(f"{module.__name__}: must define a non-empty dict SCENES")
    for name, fn in scenes.items():
        if not isinstance(name, str) or not callable(fn):
            raise SceneLoadError(f"SCENES[{name!r}] is not a callable")
    return dict(scenes)


def build_scene_set(custom_module: str | Path | None) -> dict[str, SceneRenderer]:
    """Built-in scenes, with the custom module's scenes layered on top (overrides allowed)."""
    from cueframe import scenes as builtin

    scene_set: dict[str, SceneRenderer] = dict(builtin.SCENES)
    if custom_module is not None:
        scene_set.update(scene_set_from_module(load_scene_module(custom_module)))
    return scene_set


def iter_frames(
    timeline: Timeline[Cue],
    features: Features,
    scene_set: SceneSet,
    *,
    width: int,
    height: int,
    crossfade: float = 0.0,
    seed: int = 0,
) -> Iterator[np.ndarray]:
    """Yield RGB uint8 arrays for every frame of the timeline, in order."""
    if features.n_frames != timeline.n_frames:
        raise CueError(
            f"features cover {features.n_frames} frames but the timeline has {timeline.n_frames}"
        )
    fade_frames = time_to_frame(crossfade, timeline.fps) if crossfade > 0 else 0
    segs = timeline.segments
    seg_index = {id(s): i for i, s in enumerate(segs)}
    cue_indices: dict[int, int] = {}
    k = 0
    for s in segs:
        if s.payload is not None:
            cue_indices[id(s)] = k
            k += 1

    def context_for(frame: int, seg) -> FrameContext:
        return FrameContext(
            frame=frame,
            t=frame_to_time(frame, timeline.fps),
            fps=timeline.fps,
            width=width,
            height=height,
            n_frames=timeline.n_frames,
            cue=seg.payload,
            cue_index=cue_indices.get(id(seg)),
            cue_progress=min(1.0, max(0.0, seg.progress(frame)))
            if seg.payload is not None else 0.0,
            cue_frame=frame - seg.f0,
            level=float(features.level[frame]),
            onset=float(features.onset[frame]),
            seed=seed,
        )

    def draw(frame: int, seg) -> Image.Image:
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        name = seg.payload.scene if seg.payload is not None else "idle"
        scene_set[name](context_for(frame, seg), canvas)
        if canvas.size != (width, height) or canvas.mode != "RGB":
            raise CueError(f"scene {name!r} changed the canvas size or mode")
        return canvas

    for frame in range(timeline.n_frames):
        seg = timeline.at(frame)
        img = draw(frame, seg)
        i = seg_index[id(seg)]
        into = frame - seg.f0
        if fade_frames and i > 0 and into < fade_frames:
            prev = segs[i - 1]
            # the previous scene keeps running (its own clock continues) under the new one
            prev_img = draw(frame, prev)
            alpha = (into + 1) / (fade_frames + 1)
            img = Image.blend(prev_img, img, alpha)
        yield np.asarray(img, dtype=np.uint8)


@dataclass(frozen=True)
class AnimateResult:
    out_path: Path
    n_frames: int
    fps: Fraction
    duration: float
    audio_duration: float
    sample_rate: int
    n_cues: int
    input_fingerprint: tuple[int, int, str]

    def summary(self) -> dict[str, Any]:
        return {
            "out": str(self.out_path),
            "n_frames": self.n_frames,
            "fps": str(self.fps),
            "video_duration": round(self.duration, 6),
            "audio_duration": round(self.audio_duration, 6),
            "sample_rate": self.sample_rate,
            "n_cues": self.n_cues,
            "input_sha256": self.input_fingerprint[2],
        }


def animate(
    audio_path: str | Path,
    sheet: CueSheet,
    out_path: str | Path,
    *,
    scene_set: SceneSet | None = None,
    seed: int = 0,
) -> AnimateResult:
    """Render ``sheet`` over the WAV at ``audio_path`` to ``out_path`` (MP4)."""
    require_tools()
    scene_set = scene_set if scene_set is not None else build_scene_set(None)
    validate_scene_names(sheet, scene_set)
    before = file_fingerprint(audio_path)
    audio: Audio = read_wav(audio_path)
    timeline = sheet.to_timeline(audio.duration)
    features = frame_features(audio, sheet.fps)
    width, height = sheet.size
    frames = iter_frames(
        timeline, features, scene_set, width=width, height=height,
        crossfade=sheet.crossfade, seed=seed,
    )
    mux_raw_frames(
        frames,
        width=width,
        height=height,
        fps=sheet.fps,
        n_frames=timeline.n_frames,
        audio_path=audio_path,
        out_path=out_path,
    )
    check_output(
        out_path, fps=sheet.fps, n_frames=timeline.n_frames, width=width, height=height,
        audio_rate=audio.sample_rate,
    )
    after = file_fingerprint(audio_path)
    if after != before:
        raise RuntimeError(f"input audio {audio_path} changed during rendering")
    return AnimateResult(
        out_path=Path(out_path),
        n_frames=timeline.n_frames,
        fps=sheet.fps,
        duration=timeline.duration,
        audio_duration=audio.duration,
        sample_rate=audio.sample_rate,
        n_cues=len(sheet.cues),
        input_fingerprint=before,
    )
