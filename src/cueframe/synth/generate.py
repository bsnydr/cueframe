"""Generate the synthetic multicamera stage: camera files, reference audio and a manifest.

Everything here is invented from a seed. The *manifest* holds the ground truth (planted
clocks and flash times) and is read only by the oracle and the tests; ``inputs.json`` lists
the media files and is what ``sync``/``plan`` consume.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from cueframe.audio import Audio, write_wav
from cueframe.clock import AffineClock
from cueframe.media import check_output, mux_raw_frames, require_tools
from cueframe.synth.stage import CameraView, render_stage_frame
from cueframe.synth.stage_audio import camera_audio, design_chirps, reference_track
from cueframe.timeline import frame_count, frame_to_time

FLASH_LENGTH_S = 0.08


@dataclass(frozen=True)
class CamSpec:
    name: str
    framing: str
    palette: int
    fps: Fraction
    offset_s: float  # reference time when the camera started
    drift_ppm: float
    duration_s: float  # camera seconds recorded
    snr_db: float
    tone: float

    @property
    def clock(self) -> AffineClock:
        return AffineClock.from_drift_ppm(self.offset_s, self.drift_ppm)


@dataclass(frozen=True)
class StageSpec:
    duration_s: float = 64.0  # reference recorder length
    size: tuple[int, int] = (854, 480)
    sample_rate: int = 48000
    seed: int = 1
    n_events: int = 12
    cameras: tuple[CamSpec, ...] = (
        CamSpec("A", "wide", 0, Fraction(25), 1.3, +0.0, 60.0, 30.0, 0.0),
        CamSpec("B", "left", 1, Fraction(25), -2.7, +3200.0, 66.0, 24.0, -0.5),
        CamSpec("C", "right", 2, Fraction(30), 4.1, -2600.0, 58.0, 20.0, +0.6),
    )

    @classmethod
    def quick(cls, seed: int = 1) -> StageSpec:
        """Small, fast variant for tests and CI (30 s, 320x180, same clock story compressed).

        Drift is still far beyond real cameras (thousands of ppm) so that a clap-only sync
        is visibly wrong by the end of a 30 s take.
        """
        return cls(
            duration_s=30.0,
            size=(320, 180),
            seed=seed,
            n_events=8,
            cameras=(
                CamSpec("A", "wide", 0, Fraction(25), 0.6, +0.0, 28.5, 30.0, 0.0),
                CamSpec("B", "left", 1, Fraction(25), -1.1, +5000.0, 30.5, 24.0, -0.5),
                CamSpec("C", "right", 2, Fraction(30), 1.4, -4500.0, 27.5, 20.0, +0.6),
            ),
        )


def common_coverage(spec: StageSpec) -> tuple[float, float]:
    start = max(c.clock.to_ref(0.0) for c in spec.cameras)
    end = min(c.clock.to_ref(c.duration_s) for c in spec.cameras)
    return max(start, 0.0), min(end, spec.duration_s)


def plan_events(spec: StageSpec) -> list[float]:
    """Irregularly spaced flash times inside the common coverage (never periodic)."""
    rng = np.random.default_rng(spec.seed + 11)
    lo, hi = common_coverage(spec)
    lo, hi = lo + 1.0, hi - 1.0
    min_gap = max(1.2, (hi - lo) / (spec.n_events * 1.8))
    for _ in range(200):
        times = np.sort(rng.uniform(lo, hi, spec.n_events))
        if np.all(np.diff(times) >= min_gap):
            return [round(float(t), 4) for t in times]
    # fall back to a jittered grid that still has unequal gaps
    grid = np.linspace(lo, hi, spec.n_events)
    jitter = rng.uniform(-0.3, 0.3, spec.n_events) * (grid[1] - grid[0])
    return [round(float(t), 4) for t in np.sort(grid + jitter)]


def flash_intensity(t_cam0: float, t_cam1: float, clock: AffineClock, events: list[float]) -> float:
    """1.0 if the camera frame's reference-time interval overlaps a flash event, else 0.0."""
    r0, r1 = clock.to_ref(t_cam0), clock.to_ref(t_cam1)
    for t_e in events:
        if r0 < t_e + FLASH_LENGTH_S and r1 > t_e:
            return 1.0
    return 0.0


def _timecode(frame: int, fps: Fraction) -> str:
    fps_i = math.ceil(fps)
    secs, ff = divmod(frame, fps_i)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"


def camera_frames(
    cam: CamSpec, spec: StageSpec, events: list[float], n: int
) -> Iterator[np.ndarray]:
    view = CameraView(cam.name, cam.framing, cam.palette)
    clock = cam.clock
    for k in range(n):
        t0, t1 = frame_to_time(k, cam.fps), frame_to_time(k + 1, cam.fps)
        t_mid = (t0 + t1) / 2
        t_ref = clock.to_ref(t_mid)
        img = render_stage_frame(
            t_ref,
            view,
            spec.size,
            flash=flash_intensity(t0, t1, clock, events),
            overlay_lines=(
                f"cam {cam.name}  {cam.fps} fps",
                f"frame {k:06d}",
                f"tc {_timecode(k, cam.fps)}",
                f"true ref {t_ref:9.3f}s",
            ),
        )
        yield np.asarray(img, dtype=np.uint8)


def generate_stage(spec: StageSpec, out_dir: str | Path) -> dict[str, Any]:
    """Write reference.wav, one .mov per camera, inputs.json and manifest.json."""
    require_tools()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = plan_events(spec)
    chirps = design_chirps(events, spec.seed)
    ref = reference_track(spec.duration_s, spec.sample_rate, chirps, spec.seed)
    ref_path = out / "reference.wav"
    write_wav(ref_path, Audio(ref, spec.sample_rate))

    cam_entries = []
    input_entries = []
    for i, cam in enumerate(spec.cameras):
        n = frame_count(cam.duration_s, cam.fps)
        audio = camera_audio(
            ref, spec.sample_rate, cam.clock, cam.duration_s,
            snr_db=cam.snr_db, tone=cam.tone, seed=spec.seed * 100 + i,
        )
        wav_path = out / f"cam_{cam.name}_audio.wav"
        write_wav(wav_path, Audio(audio, spec.sample_rate))
        mov_path = out / f"cam_{cam.name}.mov"
        mux_raw_frames(
            camera_frames(cam, spec, events, n),
            width=spec.size[0], height=spec.size[1], fps=cam.fps, n_frames=n,
            audio_path=wav_path, out_path=mov_path, audio_codec="pcm_s16le",
        )
        check_output(mov_path, fps=cam.fps, n_frames=n, width=spec.size[0], height=spec.size[1],
                     audio_rate=spec.sample_rate)
        wav_path.unlink()
        cam_entries.append(
            {
                "name": cam.name,
                "path": mov_path.name,
                "fps": str(cam.fps),
                "duration_s": cam.duration_s,
                "framing": cam.framing,
                "true_clock": cam.clock.to_json(),
                "snr_db": cam.snr_db,
            }
        )
        input_entries.append({"name": cam.name, "path": mov_path.name})

    manifest = {
        "note": "Ground truth planted by the generator. Read only by the oracle and tests; "
        "never by sync or plan.",
        "seed": spec.seed,
        "duration_s": spec.duration_s,
        "sample_rate": spec.sample_rate,
        "size": list(spec.size),
        "flash_length_s": FLASH_LENGTH_S,
        "flash_events_ref_s": events,
        "cameras": cam_entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    inputs = {"reference": ref_path.name, "cameras": input_entries}
    (out / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    return manifest


def audio_fixture(
    duration_s: float,
    clocks: dict[str, AffineClock],
    *,
    sample_rate: int = 16000,
    seed: int = 3,
    n_events: int | None = None,
    snr_db: float = 24.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Audio-only fixture (no video) for estimator tests at arbitrary durations."""
    rng = np.random.default_rng(seed)
    n_events = n_events or max(6, int(duration_s / 5))
    lo, hi = 1.0, duration_s - 1.0
    times = np.sort(rng.uniform(lo, hi, n_events))
    chirps = design_chirps([float(t) for t in times], seed)
    ref = reference_track(duration_s, sample_rate, chirps, seed)
    cams = {}
    for i, (name, clock) in enumerate(clocks.items()):
        cam_dur = (duration_s - clock.offset) / clock.rate
        cams[name] = camera_audio(
            ref, sample_rate, clock, cam_dur, snr_db=snr_db, tone=0.0, seed=seed * 10 + i
        )
    return ref, cams
