"""Independent pixel oracle for the multicamera demo.

The generator plants short white *flash* events at known reference times and records them in
``manifest.json``. Neither ``sync`` nor ``plan`` reads that file. This module does: for every
flash inside the edit's timeline it decodes the output frame that should show the flash and
two frames that should not, and compares mean luminance. If the clocks or the cut were wrong
by more than the flash length, the bright frame lands elsewhere and the check fails.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from cueframe.media import extract_frame, probe
from cueframe.timeline import parse_fps, time_to_frame_floor


def luminance(frame: np.ndarray) -> float:
    f = frame.astype(np.float64)
    return float(np.mean(0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]))


def flash_oracle(
    manifest_path: str | Path,
    plan_path: str | Path,
    video_path: str | Path,
    *,
    bright: float = 150.0,
    dark: float = 110.0,
    dark_offset: float = 0.3,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    fps: Fraction = parse_fps(plan["fps"])
    start = float(plan["timeline_start_ref_s"])
    n_frames = int(plan["n_frames"])
    flash_len = float(manifest["flash_length_s"])
    p = probe(video_path)
    if p.nb_frames != n_frames:
        return {"status": "fail", "reason": f"video has {p.nb_frames} frames, plan says {n_frames}"}
    events = []
    for t_event in manifest["flash_events_ref_s"]:
        mid = t_event + flash_len / 2 - start
        f_mid = time_to_frame_floor(mid, fps)
        f_before = time_to_frame_floor(mid - dark_offset, fps)
        f_after = time_to_frame_floor(mid + dark_offset, fps)
        if f_before < 0 or f_after >= n_frames:
            continue
        lum = {
            "mid": luminance(extract_frame(video_path, f_mid, fps)),
            "before": luminance(extract_frame(video_path, f_before, fps)),
            "after": luminance(extract_frame(video_path, f_after, fps)),
        }
        ok = lum["mid"] >= bright and lum["before"] <= dark and lum["after"] <= dark
        events.append(
            {
                "t_ref": t_event,
                "frame": f_mid,
                "luminance": {k: round(v, 1) for k, v in lum.items()},
                "ok": ok,
            }
        )
    n_ok = sum(e["ok"] for e in events)
    status = "pass" if events and n_ok == len(events) else "fail"
    return {
        "status": status,
        "n_events": len(events),
        "n_ok": n_ok,
        "thresholds": {"bright": bright, "dark": dark},
        "events": events,
    }
