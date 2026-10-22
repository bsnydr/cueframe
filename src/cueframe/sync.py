"""Audio clock estimation: camera audio vs a reference recorder.

Two stages, both confidence-gated, followed by a robust affine fit:

1. **Coarse**: three short camera windows are each matched against the whole reference
   track (GCC-PHAT over the full lag range). The most confident gives the coarse offset.
   A strong second peak far from the first marks the audio as *ambiguous* (periodic).
2. **Fine**: many short windows along the camera track are matched against the reference
   at the time the coarse clock predicts, within a small lag range. Each yields
   ``(t_cam, t_ref, confidence, ambiguity)``.
3. **Fit**: Theil–Sen (median of pairwise slopes) over confident windows, MAD-based outlier
   rejection, least-squares refit on inliers → :class:`~cueframe.clock.AffineClock`.

The estimator never sees the generator's manifest; it only reads audio samples.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from cueframe.clock import AffineClock
from cueframe.gccphat import gcc_phat


@dataclass(frozen=True)
class SyncConfig:
    analysis_rate: int = 16000
    band: tuple[float, float] = (200.0, 6000.0)
    coarse_window: float = 8.0
    coarse_min_confidence: float = 10.0
    max_ambiguity: float = 0.6
    window: float = 2.5
    stride: float = 4.0
    fine_margin: float = 0.25
    min_confidence: float = 8.0
    min_windows: int = 4
    min_span_fraction: float = 0.5
    min_duration: float = 8.0
    silence_dbfs: float = -60.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowMeasurement:
    t_cam: float  # centre of the camera window, camera seconds
    t_ref: float  # measured reference time of that centre
    predicted_ref: float  # what the predictor clock said
    confidence: float
    ambiguity: float
    used: bool  # passed the confidence/ambiguity gate (and, after fitting, inlier)

    @property
    def residual_ms(self) -> float:
        return (self.t_ref - self.predicted_ref) * 1000.0


@dataclass(frozen=True)
class CameraSync:
    name: str
    status: str  # ok | too_short | silent | no_match | ambiguous | insufficient
    reason: str
    clock: AffineClock | None
    coarse_offset: float | None
    windows: tuple[WindowMeasurement, ...]
    n_used: int
    span_fraction: float
    residual_rms_ms: float
    residual_max_ms: float
    cam_duration: float
    config: SyncConfig = field(default_factory=SyncConfig)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "clock": self.clock.to_json() if self.clock else None,
            "coarse_offset_s": self.coarse_offset,
            "n_windows": len(self.windows),
            "n_used": self.n_used,
            "span_fraction": round(self.span_fraction, 4),
            "residual_rms_ms": round(self.residual_rms_ms, 4),
            "residual_max_ms": round(self.residual_max_ms, 4),
            "cam_duration_s": self.cam_duration,
            "windows": [
                {
                    "t_cam": round(w.t_cam, 4),
                    "t_ref": round(w.t_ref, 6),
                    "predicted_ref": round(w.predicted_ref, 6),
                    "residual_ms": round(w.residual_ms, 4),
                    "confidence": round(w.confidence, 2),
                    "ambiguity": round(w.ambiguity, 3),
                    "used": w.used,
                }
                for w in self.windows
            ],
            "config": self.config.to_json(),
        }


def _window(x: np.ndarray, start_sample: int, length: int) -> np.ndarray:
    """Exact-length slice with explicit zero padding outside the signal."""
    out = np.zeros(length, dtype=np.float64)
    a = max(start_sample, 0)
    b = min(start_sample + length, x.size)
    if b > a:
        out[a - start_sample : b - start_sample] = x[a:b]
    return out


def _warped_reference(
    ref: np.ndarray, rate: int, predictor: AffineClock, t_cam0: float, n: int
) -> np.ndarray:
    """Reference audio resampled onto the camera's sample grid through ``predictor``.

    Sample ``j`` is the reference signal at ``predictor.to_ref(t_cam0 + j / rate)``; outside
    the reference recording it is zero. With an exact predictor this equals the camera
    audio sample for sample, however large the drift.
    """
    t_cam = t_cam0 + np.arange(n) / rate
    pos = (predictor.offset + predictor.rate * t_cam) * rate
    out = np.interp(pos, np.arange(ref.size), ref, left=0.0, right=0.0)
    out[(pos < 0) | (pos > ref.size - 1)] = 0.0
    return out


def measure_windows(
    cam: np.ndarray,
    ref: np.ndarray,
    rate: int,
    predictor: AffineClock,
    centers: list[float],
    cfg: SyncConfig,
) -> list[WindowMeasurement]:
    """Measure the reference time of each camera window centre against the prediction.

    The reference is warped onto the camera grid through ``predictor`` (so the predictor's
    rate is removed from the comparison) and widened by ``fine_margin`` on each side; a lag
    of zero means the predictor was exact. Returns one measurement per centre (``used``
    reflects the confidence/ambiguity gate only).
    """
    win = int(round(cfg.window * rate))
    margin = int(round(cfg.fine_margin * rate))
    out = []
    for tc in centers:
        cam_start = int(round((tc - cfg.window / 2) * rate))
        cam_win = _window(cam, cam_start, win)
        ref_win = _warped_reference(
            ref, rate, predictor, (cam_start - margin) / rate, win + 2 * margin
        )
        predicted_center = predictor.to_ref(tc)
        if np.max(np.abs(cam_win)) < 1e-6 or np.max(np.abs(ref_win)) < 1e-6:
            out.append(WindowMeasurement(tc, math.nan, predicted_center, 0.0, 1.0, False))
            continue
        sig = np.concatenate((np.zeros(margin), cam_win, np.zeros(margin)))
        est = gcc_phat(sig, ref_win, rate, max_lag=cfg.fine_margin, band=cfg.band)
        # sig(t) ≈ ref_win(t - lag) in camera seconds; a positive lag means the camera content
        # sits later than predicted, i.e. the true reference time is earlier by lag * rate.
        measured_center = predicted_center - est.lag * predictor.rate
        used = est.confidence >= cfg.min_confidence and est.ambiguity <= cfg.max_ambiguity
        out.append(
            WindowMeasurement(
                tc, measured_center, predicted_center, est.confidence, est.ambiguity, used
            )
        )
    return out


def fine_centers(cam_duration: float, cfg: SyncConfig, phase: float = 0.0) -> list[float]:
    """Window centres along the camera track; ``phase`` shifts the grid (held-out sets)."""
    first = cfg.window / 2 + phase
    last = cam_duration - cfg.window / 2
    centers = []
    t = first
    while t <= last + 1e-9:
        centers.append(round(t, 6))
        t += cfg.stride
    return centers


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Median-of-slopes line fit: returns (intercept, slope)."""
    n = x.size
    if n < 2:
        raise ValueError("need at least two points")
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    slopes = (y[j] - y[i])[ok] / dx[ok]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return intercept, slope


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def sync_camera(
    cam: np.ndarray,
    ref: np.ndarray,
    rate: int,
    *,
    name: str = "camera",
    cfg: SyncConfig | None = None,
) -> CameraSync:
    """Estimate the camera clock from audio alone. Never raises for bad audio; see ``status``."""
    cfg = cfg or SyncConfig()
    cam = np.asarray(cam, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    cam_duration = cam.size / rate

    def fail(status: str, reason: str, coarse: float | None = None, windows=()) -> CameraSync:
        return CameraSync(
            name, status, reason, None, coarse, tuple(windows), 0, 0.0, 0.0, 0.0, cam_duration, cfg
        )

    if cam_duration < cfg.min_duration:
        return fail("too_short", f"camera audio is {cam_duration:.2f}s; need ≥ {cfg.min_duration}s")
    cam_dbfs = 20 * math.log10(max(_rms(cam), 1e-12))
    if cam_dbfs < cfg.silence_dbfs:
        return fail("silent", f"camera audio RMS {cam_dbfs:.1f} dBFS is below {cfg.silence_dbfs}")

    # --- coarse stage -------------------------------------------------------------
    # Three short camera windows against the whole reference. The best one gives the coarse
    # offset; if at least two are confident, the offsets they imply at different camera
    # times also give a rough rate, so the first fine pass is not blurred by large drift.
    cw = int(round(cfg.coarse_window * rate))
    best = None
    coarse_points: list[tuple[float, float]] = []
    for frac in (0.25, 0.5, 0.75):
        start = int(round(frac * cam.size)) - cw // 2
        start = min(max(start, 0), max(cam.size - cw, 0))
        cam_win = cam[start : start + cw]
        if cam_win.size < cw // 2:
            continue
        est = gcc_phat(
            cam_win, ref, rate, max_lag=(ref.size + cam_win.size) / rate, band=cfg.band
        )
        # cam_win(τ) ≈ ref(τ - lag) → camera time start/rate ↔ reference time -lag
        offset = -est.lag - start / rate
        if est.confidence >= cfg.coarse_min_confidence and est.ambiguity <= cfg.max_ambiguity:
            t_center = (start + cw / 2) / rate
            coarse_points.append((t_center, offset + t_center))
        if best is None or est.confidence > best[0].confidence:
            best = (est, offset, (start + cw / 2) / rate)
    if best is None:
        return fail("too_short", "no coarse window fits inside the camera audio")
    coarse_est, coarse_offset, t_best = best
    coarse_rate = 1.0
    if len(coarse_points) >= 2:
        xs = np.array([p[0] for p in coarse_points])
        ys = np.array([p[1] for p in coarse_points])
        _, slope = theil_sen(xs, ys)
        if 0.9 < slope < 1.1:
            coarse_rate = float(slope)
    if coarse_est.confidence < cfg.coarse_min_confidence:
        return fail(
            "no_match",
            f"coarse confidence {coarse_est.confidence:.1f} below {cfg.coarse_min_confidence}; "
            "no reliable match with the reference (no overlap, silence or unrelated audio)",
            coarse_offset,
        )
    if coarse_est.ambiguity > cfg.max_ambiguity:
        return fail(
            "ambiguous",
            f"coarse ambiguity {coarse_est.ambiguity:.2f} above {cfg.max_ambiguity}; "
            "a second lag matches almost as well (periodic or repeated audio)",
            coarse_offset,
        )
    coarse_clock = AffineClock(coarse_offset, 1.0)
    if coarse_rate != 1.0:
        # keep the best window's match point, apply the rough rate around it
        coarse_clock = AffineClock(coarse_offset + t_best * (1.0 - coarse_rate), coarse_rate)

    # --- fine stage: pass 1 with the coarse clock, pass 2 with the fitted clock -----------
    # Pass 2 matters when drift is large: a 2.5 s window at 3000 ppm is stretched by 7.5 ms
    # relative to the reference, which blurs the correlation peak; warping the reference
    # through the first fit removes that stretch and the residuals shrink accordingly.
    centers = fine_centers(cam_duration, cfg)
    predictor = coarse_clock
    windows: list[WindowMeasurement] = []
    intercept, slope, inlier = coarse_offset, 1.0, np.zeros(0, dtype=bool)
    for _pass in range(2):
        windows = measure_windows(cam, ref, rate, predictor, centers, cfg)
        gated = [w for w in windows if w.used]
        if len(gated) < cfg.min_windows:
            return fail(
                "insufficient",
                f"only {len(gated)} of {len(windows)} windows passed the confidence gate; "
                f"need ≥ {cfg.min_windows}",
                coarse_offset,
                windows,
            )
        x = np.array([w.t_cam for w in gated])
        y = np.array([w.t_ref for w in gated])
        intercept, slope = theil_sen(x, y)
        resid = y - (intercept + slope * x)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        tol = max(3.0 * 1.4826 * mad, 0.001)
        inlier = np.abs(resid - np.median(resid)) <= tol
        if inlier.sum() >= 2:
            slope, intercept = np.polyfit(x[inlier], y[inlier], 1)
            slope, intercept = float(slope), float(intercept)
        if not 0.5 < slope < 2.0:
            return fail("insufficient", f"implausible fitted rate {slope}", coarse_offset, windows)
        predictor = AffineClock(intercept, slope)
    final_windows = []
    k = 0
    for w in windows:
        if w.used:
            keep = bool(inlier[k])
            k += 1
        else:
            keep = False
        final_windows.append(
            WindowMeasurement(
                w.t_cam, w.t_ref, intercept + slope * w.t_cam, w.confidence, w.ambiguity, keep
            )
        )
    used = [w for w in final_windows if w.used]
    span = (max(w.t_cam for w in used) - min(w.t_cam for w in used)) / cam_duration if used else 0
    if len(used) < cfg.min_windows or span < cfg.min_span_fraction:
        return CameraSync(
            name,
            "insufficient",
            f"{len(used)} inlier windows spanning {span:.0%} of the camera; need ≥ "
            f"{cfg.min_windows} spanning ≥ {cfg.min_span_fraction:.0%}",
            None,
            coarse_offset,
            tuple(final_windows),
            len(used),
            span,
            0.0,
            0.0,
            cam_duration,
            cfg,
        )
    try:
        clock = AffineClock(intercept, slope)
    except ValueError as exc:
        return fail("insufficient", str(exc), coarse_offset, final_windows)
    res_ms = np.array([w.residual_ms for w in used])
    return CameraSync(
        name,
        "ok",
        f"{len(used)} windows, drift {clock.drift_ppm:+.1f} ppm",
        clock,
        coarse_offset,
        tuple(final_windows),
        len(used),
        span,
        _rms(res_ms),
        float(np.max(np.abs(res_ms))),
        cam_duration,
        cfg,
    )
