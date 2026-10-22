"""Cut a :class:`~cueframe.plan.Plan` into a single video with ffmpeg.

Per shot: seek the source to half a source frame *before* the planned source frame (so the
first decoded frame is exactly that frame), rebase timestamps to zero, convert to the output
frame rate with the ``fps`` filter, crop/scale, and cap the frame count. Clips are then
concatenated (stream copy) and the reference audio is muxed, trimmed to the timeline.
Every clip is probed and must contain exactly its planned number of frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from cueframe.media import (
    MediaError,
    check_output,
    probe,
    run,
    safe_path,
    video_encode_args,
)
from cueframe.plan import TIGHT_ZOOM, Plan
from cueframe.timeline import frame_to_time, parse_fps


def _even(x: float) -> int:
    v = int(round(x))
    return v if v % 2 == 0 else v - 1


def video_filter(framing: str, src_size: tuple[int, int], out_size: tuple[int, int],
                 fps_text: str, clock_rate: float = 1.0) -> str:
    """Filter graph for one shot. All values are numbers computed here, never user text."""
    sw, sh = src_size
    ow, oh = out_size
    if any(type(v) is not int or v < 16 or v > 7680 for v in (sw, sh, ow, oh)):
        raise MediaError("filter dimensions must be bounded positive integers")
    fps_text = str(parse_fps(fps_text))
    if not math.isfinite(clock_rate) or not 0.5 < clock_rate < 2:
        raise MediaError("invalid clock rate")
    # A camera second spans clock_rate reference seconds, including *within* a shot.
    pts = "PTS-STARTPTS" if clock_rate == 1 else f"(PTS-STARTPTS)*{clock_rate:.12g}"
    parts = [f"setpts={pts}", f"fps={fps_text}"]
    if framing == "tight":
        cw, ch = _even(sw / TIGHT_ZOOM), _even(sh / TIGHT_ZOOM)
        cx, cy = (sw - cw) // 2, (sh - ch) // 2
        parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
    elif framing != "full":
        raise MediaError(f"unknown framing {framing!r}")
    parts.append(f"scale={ow}:{oh}:flags=lanczos")
    parts.append("setsar=1")
    return ",".join(parts)


def concat_entry(path: Path) -> str:
    """One line of an ffmpeg concat-demuxer list; single quotes are escaped as '\\''."""
    text = safe_path(path).replace("'", "'\\''")
    return f"file '{text}'\n"


@dataclass(frozen=True)
class CutResult:
    out_path: Path
    n_frames: int
    duration: float
    clips: tuple[Path, ...]


def cut(plan: Plan, reference_audio: str | Path | None, out_path: str | Path,
        *, work_dir: str | Path | None = None) -> CutResult:
    out = Path(out_path)
    work = Path(work_dir) if work_dir is not None else out.parent / (out.stem + "_clips")
    work.mkdir(parents=True, exist_ok=True)
    cams = {c.name: c for c in plan.cameras}
    fps_text = str(plan.fps)
    clips: list[Path] = []
    for i, seg in enumerate(plan.timeline.segments):
        shot = seg.payload
        assert shot is not None
        cam = cams[shot.cam]
        half = frame_to_time(1, cam.fps) / 2
        seek = max(0.0, shot.src_start - half)
        clip = work / f"clip_{i:03d}.mp4"
        argv = [
            "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
            "-ss", f"{seek:.6f}",
            "-i", safe_path(cam.path),
            "-an",
            "-vf", video_filter(shot.framing, (cam.width, cam.height), (plan.width, plan.height),
                                fps_text, cam.clock.rate),
            "-fps_mode", "passthrough",
            "-frames:v", str(seg.n),
            *video_encode_args(plan.fps, set_rate=False),
            safe_path(clip),
        ]
        run(argv, timeout=1800)
        p = probe(clip)
        if p.nb_frames != seg.n:
            raise MediaError(f"clip {i} has {p.nb_frames} frames, planned {seg.n}")
        clips.append(clip)

    list_path = work / "concat.txt"
    list_path.write_text("".join(concat_entry(c) for c in clips), encoding="utf-8")
    argv = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", safe_path(list_path),
    ]
    if reference_audio is not None:
        argv += [
            "-ss", f"{plan.timeline_start:.6f}", "-t", f"{plan.duration:.6f}",
            "-i", safe_path(reference_audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        argv += ["-an"]
    argv += ["-c:v", "copy", "-movflags", "+faststart", "-map_metadata", "-1",
             "-t", f"{plan.duration:.6f}", safe_path(out)]
    run(argv, timeout=1800)
    audio_rate = None
    if reference_audio is not None:
        audio_rate = probe_audio_rate(reference_audio)
    check_output(out, fps=plan.fps, n_frames=plan.n_frames, width=plan.width, height=plan.height,
                 audio_rate=audio_rate)
    return CutResult(out, plan.n_frames, plan.duration, tuple(clips))


def probe_audio_rate(path: str | Path) -> int | None:
    import json

    proc = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                safe_path(path)])
    doc = json.loads(proc.stdout.decode())
    audio = [s for s in doc.get("streams", []) if s.get("codec_type") == "audio"]
    return int(audio[0]["sample_rate"]) if audio else None
