"""Held-out verification of an estimated clock.

The fitted clock becomes the *predictor*; residuals are re-measured on windows the fit
never used: the fine grid shifted by half a stride, plus explicit windows at the start,
middle and end of the camera track. A residual is the difference between the measured
reference time of a window and the time the clock predicts for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cueframe.clock import AffineClock
from cueframe.sync import SyncConfig, WindowMeasurement, fine_centers, measure_windows


@dataclass(frozen=True)
class VerifyResult:
    name: str
    status: str  # pass | fail | no_confident_window | insufficient_coverage
    tolerance_ms: float
    windows: tuple[WindowMeasurement, ...]
    n_confident: int
    max_abs_ms: float
    rms_ms: float
    worst_t_cam: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "tolerance_ms": self.tolerance_ms,
            "n_windows": len(self.windows),
            "n_confident": self.n_confident,
            "max_abs_ms": round(self.max_abs_ms, 4),
            "rms_ms": round(self.rms_ms, 4),
            "worst_t_cam": self.worst_t_cam,
            "windows": [
                {
                    "t_cam": round(w.t_cam, 4),
                    "residual_ms": round(w.residual_ms, 4) if w.used else None,
                    "confidence": round(w.confidence, 2),
                    "ambiguity": round(w.ambiguity, 3),
                    "confident": w.used,
                }
                for w in self.windows
            ],
        }


def heldout_centers(cam_duration: float, cfg: SyncConfig) -> list[float]:
    """Fit grid shifted by half a stride, plus start/middle/end windows; never a fit centre."""
    half = cfg.window / 2
    fit = fine_centers(cam_duration, cfg)
    centers = set(fine_centers(cam_duration, cfg, phase=cfg.stride / 2))
    for t in (half, cam_duration / 2, cam_duration - half):
        t = round(t, 6)
        if any(abs(t - f) < cfg.window / 4 for f in fit):
            t = round(t + cfg.stride / 4, 6)
        centers.add(t)
    return sorted(c for c in centers if half - 1e-9 <= c <= cam_duration - half + 1e-9)


def verify_camera(
    cam: np.ndarray,
    ref: np.ndarray,
    rate: int,
    clock: AffineClock,
    *,
    name: str = "camera",
    cfg: SyncConfig | None = None,
    tolerance_ms: float = 2.0,
    min_confident: int = 3,
) -> VerifyResult:
    cfg = cfg or SyncConfig()
    if not math.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError("verification tolerance must be finite and positive")
    cam = np.asarray(cam, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    centers = heldout_centers(cam.size / rate, cfg)
    windows = tuple(measure_windows(cam, ref, rate, clock, centers, cfg))
    confident = [w for w in windows if w.used]
    if len(confident) < min_confident:
        return VerifyResult(
            name, "no_confident_window", tolerance_ms, windows, len(confident), 0.0, 0.0, None
        )
    res = np.array([w.residual_ms for w in confident])
    worst = int(np.argmax(np.abs(res)))
    max_abs = float(np.abs(res[worst]))
    lo = max(0.0, clock.to_cam(0.0))
    hi = min(cam.size / rate, clock.to_cam(ref.size / rate))
    span = max(w.t_cam for w in confident) - min(w.t_cam for w in confident)
    if max_abs > tolerance_ms:
        status = "fail"
    elif hi <= lo or span < cfg.min_span_fraction * (hi - lo):
        status = "insufficient_coverage"
    else:
        status = "pass"
    return VerifyResult(
        name,
        status,
        tolerance_ms,
        windows,
        len(confident),
        max_abs,
        float(np.sqrt(np.mean(res * res))),
        confident[worst].t_cam,
    )
