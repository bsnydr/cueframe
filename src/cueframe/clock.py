"""Affine clock model shared by the multicamera synth, estimator, planner and cutter.

``t_ref = offset + rate * t_cam``

* ``offset`` (seconds): the reference time at which the camera clock reads zero
  (i.e. when the camera recording starts). May be negative.
* ``rate`` (dimensionless): reference seconds per camera second. ``rate > 1`` means the
  camera clock runs *slow* (one camera second covers more than one reference second).
* ``drift_ppm = (rate - 1) * 1e6``. 50 ppm separates two clocks by 1 frame at 25 fps
  every 800 s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AffineClock:
    offset: float
    rate: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.offset):
            raise ValueError("clock offset must be finite")
        if not 0.5 < self.rate < 2.0:
            raise ValueError(f"implausible clock rate {self.rate}")

    def to_ref(self, t_cam: float) -> float:
        return self.offset + self.rate * t_cam

    def to_cam(self, t_ref: float) -> float:
        return (t_ref - self.offset) / self.rate

    @property
    def drift_ppm(self) -> float:
        return (self.rate - 1.0) * 1e6

    @classmethod
    def from_drift_ppm(cls, offset: float, drift_ppm: float) -> AffineClock:
        return cls(offset, 1.0 + drift_ppm * 1e-6)

    def inverse(self) -> AffineClock:
        """Clock mapping reference time to camera time."""
        return AffineClock(-self.offset / self.rate, 1.0 / self.rate)

    def clap_sync(self, t_cam: float) -> AffineClock:
        """Offset-only clock that agrees with this one at camera time ``t_cam``.

        This is what a single clapperboard sync at ``t_cam`` gives: exact there, and off by
        ``drift_ppm * 1e-6 * (t - t_cam)`` seconds everywhere else.
        """
        return AffineClock(self.to_ref(t_cam) - t_cam, 1.0)

    def to_json(self) -> dict[str, float]:
        return {"offset_s": self.offset, "rate": self.rate, "drift_ppm": self.drift_ppm}

    @classmethod
    def from_json(cls, doc: dict) -> AffineClock:
        return cls(float(doc["offset_s"]), float(doc["rate"]))
