"""ffmpeg / ffprobe wrappers.

Each wrapper invocation resolves the executable to an absolute path and passes an argv list
to :mod:`subprocess` with ``shell=False``. FFmpeg receives ``-nostdin``. Media paths pass
through :func:`safe_path` as absolute paths so they cannot be read as options. Dimensions,
frame rates and clock rates are validated before construction of numeric filter values;
input text is never interpreted by a shell.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from cueframe.timeline import frame_to_time


class MediaError(RuntimeError):
    pass


def tool(name: str) -> str:
    """Absolute path of ``ffmpeg`` or ``ffprobe``; raises if missing."""
    if name not in ("ffmpeg", "ffprobe"):
        raise MediaError("only ffmpeg and ffprobe may be run through cueframe.media")
    path = shutil.which(name)
    if path is None:
        raise MediaError(f"{name} not found on PATH")
    return path


def require_tools() -> None:
    tool("ffmpeg")
    tool("ffprobe")


def safe_path(path: str | Path) -> str:
    """Absolute path string that ffmpeg cannot mistake for an option."""
    text = str(Path(path).resolve())
    if any(ord(char) < 32 for char in text):
        raise MediaError("media paths must not contain control characters")
    return text if not text.startswith("-") else "./" + text


def run(argv: Sequence[str], *, timeout: float = 600) -> subprocess.CompletedProcess[bytes]:
    """Run ffmpeg/ffprobe (argv[0] is the tool name) without a shell; raise with stderr."""
    if not argv:
        raise MediaError("empty command")
    full = [tool(argv[0]), *argv[1:]]
    try:
        proc = subprocess.run(  # noqa: S603 - absolute executable, argv list, shell=False
            full, capture_output=True, check=False, timeout=timeout, shell=False
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{argv[0]} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()[-2000:]
        raise MediaError(f"{argv[0]} failed ({proc.returncode}): {err}")
    return proc


@dataclass(frozen=True)
class Probe:
    width: int
    height: int
    fps: Fraction
    nb_frames: int
    duration: float
    audio_rate: int | None
    audio_channels: int | None
    tags: dict[str, str]
    rotation: int


def probe(path: str | Path) -> Probe:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            "-count_frames",
            safe_path(path),
        ]
    )
    doc = json.loads(proc.stdout.decode())
    video = [s for s in doc.get("streams", []) if s.get("codec_type") == "video"]
    audio = [s for s in doc.get("streams", []) if s.get("codec_type") == "audio"]
    if not video:
        raise MediaError(f"{path}: no video stream")
    v = video[0]
    fps_text = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    num, den = (int(x) for x in fps_text.split("/"))
    if den == 0:
        raise MediaError(f"{path}: unknown frame rate")
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if "rotate" in v.get("tags", {}):
        rotation = int(float(v["tags"]["rotate"]))
    fmt = doc.get("format", {})
    tags = {**fmt.get("tags", {})}
    for s in doc.get("streams", []):
        tags.update({f"stream{s.get('index')}:{k}": val for k, val in s.get("tags", {}).items()})
    nb = v.get("nb_read_frames") or v.get("nb_frames") or "0"
    return Probe(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=Fraction(num, den),
        nb_frames=int(nb),
        duration=float(fmt.get("duration", v.get("duration", 0.0))),
        audio_rate=int(audio[0]["sample_rate"]) if audio else None,
        audio_channels=int(audio[0]["channels"]) if audio else None,
        tags=tags,
        rotation=rotation,
    )


# Container/stream tags ffmpeg always writes (brand, encoder name, generic stream handler
# names, undetermined language). Anything else, e.g. a title, comment or creation time, fails.
IGNORABLE_TAGS = frozenset(
    {"major_brand", "minor_version", "compatible_brands", "encoder", "handler_name", "language",
     "vendor_id"}
)


def check_output(
    path: str | Path,
    *,
    fps: Fraction,
    n_frames: int,
    width: int,
    height: int,
    audio_rate: int | None,
) -> Probe:
    """Assert the invariants every cueframe render must satisfy."""
    p = probe(path)
    problems = []
    if (p.width, p.height) != (width, height):
        problems.append(f"size {p.width}x{p.height} != {width}x{height}")
    if p.fps != fps:
        problems.append(f"fps {p.fps} != {fps}")
    if p.nb_frames != n_frames:
        problems.append(f"frame count {p.nb_frames} != {n_frames}")
    expected = frame_to_time(n_frames, fps)
    if abs(p.duration - expected) > frame_to_time(1, fps) + 1e-6:
        problems.append(f"duration {p.duration:.4f}s != {expected:.4f}s")
    if audio_rate is not None and p.audio_rate != audio_rate:
        problems.append(f"audio rate {p.audio_rate} != {audio_rate}")
    if p.rotation:
        problems.append(f"unexpected rotation {p.rotation}")
    leftover = {k: v for k, v in p.tags.items() if k.split(":")[-1] not in IGNORABLE_TAGS}
    if leftover:
        problems.append(f"unexpected metadata tags {sorted(leftover)}")
    if problems:
        raise MediaError(f"{path}: " + "; ".join(problems))
    return p


def video_encode_args(fps: Fraction, *, set_rate: bool = True) -> list[str]:
    """libx264 settings shared by every render. ``set_rate=False`` when a filter already
    fixes the frame rate and timestamps are passed through unchanged."""
    return (["-r", str(fps)] if set_rate else []) + [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
    ]


def mux_raw_frames(
    frames: Iterator[np.ndarray],
    *,
    width: int,
    height: int,
    fps: Fraction,
    n_frames: int,
    audio_path: str | Path | None,
    out_path: str | Path,
    audio_codec: str = "aac",
    timeout: float = 1800,
) -> None:
    """Pipe exactly ``n_frames`` RGB24 frames to ffmpeg and mux with ``audio_path``.

    The audio is cut to the video length with ``-t`` from the exact frame count
    (``-shortest`` is avoided because it can drop the final frame). ``audio_codec`` is
    ``"aac"`` for deliverables or ``"pcm_s16le"`` (use a ``.mov`` output) when sample-exact
    audio timing matters, as it does for the synthetic camera files.
    """
    if audio_codec not in ("aac", "pcm_s16le"):
        raise MediaError(f"unsupported audio codec {audio_codec!r}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = frame_to_time(n_frames, fps)
    argv = [
        tool("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if audio_path is not None:
        argv += ["-i", safe_path(audio_path), "-map", "0:v:0", "-map", "1:a:0"]
        argv += ["-c:a", audio_codec] + (["-b:a", "192k"] if audio_codec == "aac" else [])
    else:
        argv += ["-an"]
    argv += video_encode_args(fps) + ["-t", f"{duration:.6f}", safe_path(out)]
    proc = subprocess.Popen(  # noqa: S603 - absolute executable, argv list, shell=False
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
    )
    assert proc.stdin is not None
    sent = 0
    err = b""
    try:
        for frame in frames:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise MediaError(f"frame {sent}: expected {height}x{width}x3 uint8 array")
            proc.stdin.write(frame.tobytes())
            sent += 1
            if sent > n_frames:
                raise MediaError("renderer produced more frames than scheduled")
        proc.stdin.close()
        # Python versions differ on flushing a closed stdin in communicate().
        proc.stdin = None
        _, err = proc.communicate(timeout=timeout)
    except BrokenPipeError as exc:
        _, err = proc.communicate(timeout=timeout)
        raise MediaError(f"ffmpeg closed the pipe: {err.decode(errors='replace')[-2000:]}") from exc
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if sent != n_frames:
        raise MediaError(f"renderer produced {sent} frames, expected {n_frames}")
    if proc.returncode != 0:
        text = err.decode(errors="replace")[-2000:]
        raise MediaError(f"ffmpeg failed ({proc.returncode}): {text}")


def extract_frame(path: str | Path, frame: int, fps: Fraction) -> np.ndarray:
    """Decode a single output frame as an RGB uint8 array (for checks and contact sheets).

    ffmpeg input seeking returns the first frame whose timestamp is >= ``-ss``; seeking a
    quarter frame early therefore yields exactly frame ``frame``.
    """
    p = probe(path)
    t = max(0.0, frame_to_time(frame, fps) - frame_to_time(1, fps) / 4)
    proc = run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-ss",
            f"{t:.6f}",
            "-i",
            safe_path(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        timeout=120,
    )
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = p.width * p.height * 3
    if arr.size < expected:
        raise MediaError(f"short frame read for frame {frame} ({arr.size} < {expected})")
    return arr[:expected].reshape(p.height, p.width, 3).copy()


def decode_audio(path: str | Path, *, rate: int) -> np.ndarray:
    """Decode the first audio stream to mono float32 at ``rate`` (used by the sync path)."""
    argv = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
        "-i", safe_path(path),
        "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "pipe:1",
    ]
    proc = run(argv, timeout=600)
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
