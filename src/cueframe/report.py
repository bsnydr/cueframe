"""Static visual evidence: contact sheets and a lag chart, drawn with Pillow only."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cueframe.clock import AffineClock
from cueframe.media import extract_frame
from cueframe.oracle import luminance
from cueframe.sync import CameraSync
from cueframe.timeline import time_to_frame_floor
from cueframe.verify import VerifyResult

BG = (14, 15, 20)
FG = (230, 232, 240)
DIM = (120, 124, 140)
GOOD = (90, 220, 140)
BAD = (240, 90, 90)
ACCENT = (110, 170, 255)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, ValueError):
        return ImageFont.load_default()


def _thumb(arr: np.ndarray, width: int) -> Image.Image:
    img = Image.fromarray(arr)
    h = round(img.height * width / img.width)
    return img.resize((width, h), Image.Resampling.LANCZOS)


def animation_sheet(
    video: str | Path,
    fps: Fraction,
    rows: Sequence[tuple[str, int]],
    out_png: str | Path,
    *,
    cell_width: int = 320,
) -> None:
    """Grid of (label, frame) thumbnails, three per row."""
    cols = 3
    thumbs = [(label, _thumb(extract_frame(video, f, fps), cell_width)) for label, f in rows]
    if not thumbs:
        raise ValueError("no frames to show")
    th = thumbs[0][1].height
    pad, label_h = 12, 26
    n_rows = (len(thumbs) + cols - 1) // cols
    W = cols * (cell_width + pad) + pad
    H = n_rows * (th + label_h + pad) + pad
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    f = _font(14)
    for i, (label, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (cell_width + pad)
        y = pad + r * (th + label_h + pad)
        sheet.paste(im, (x, y))
        d.text((x, y + th + 5), label, font=f, fill=FG)
    sheet.save(out_png, optimize=True)


def multicam_sheet(
    cameras: Sequence[tuple[str, Path, Fraction, AffineClock, AffineClock]],
    events: Sequence[float],
    flash_len: float,
    out_png: str | Path,
    *,
    cell_width: int = 256,
) -> dict[str, list[list[float]]]:
    """Two groups of camera frames at three flash events.

    ``cameras`` holds (name, path, fps, estimated_clock, offset_only_clock). Top group maps
    the flash midpoint through the full estimated clock (offset + drift); bottom group through
    the offset-only clock (a single clap sync at the start of the take). A white cell means
    the mapping found the flash. Returns the measured luminance per cell for the caller to log.
    """
    groups = [
        ("offset + drift (estimated clock)", lambda c: c[3]),
        ("offset only (clap sync at the start, drift ignored)", lambda c: c[4]),
    ]
    pad, label_h, head_h = 10, 22, 30
    th = None
    cells: dict[str, list[list[float]]] = {}
    tiles: list[list[tuple[Image.Image, float, str]]] = []
    for gname, clock_of in groups:
        rows = []
        lums = []
        for t_e in events:
            row = []
            lum_row = []
            for name, path, fps, _clock, _coarse in cameras:
                clock = clock_of((name, path, fps, _clock, _coarse))
                t_cam = clock.to_cam(t_e + flash_len / 2)
                frame = max(0, time_to_frame_floor(t_cam, fps))
                arr = extract_frame(path, frame, fps)
                lum = luminance(arr)
                row.append((_thumb(arr, cell_width), lum, f"{name}  frame {frame}  L={lum:.0f}"))
                lum_row.append(round(lum, 1))
            rows.append(row)
            lums.append(lum_row)
        cells[gname] = lums
        tiles.append(rows)
        th = rows[0][0][0].height
    assert th is not None
    cols = len(cameras)
    W = pad + cols * (cell_width + pad) + 150
    group_h = head_h + len(events) * (th + label_h + pad)
    H = pad + len(groups) * (group_h + pad)
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    f, fh = _font(13), _font(16)
    y = pad
    for (gname, _), rows in zip(groups, tiles, strict=True):
        d.text((pad, y + 6), gname, font=fh, fill=ACCENT)
        y += head_h
        for t_e, row in zip(events, rows, strict=True):
            d.text((pad, y + th // 2 - 8), f"flash @ {t_e:6.2f}s", font=f, fill=FG)
            for c, (im, lum, label) in enumerate(row):
                x = 150 + c * (cell_width + pad)
                sheet.paste(im, (x, y))
                ok = lum >= 150
                d.rectangle((x - 1, y - 1, x + cell_width, y + th), outline=GOOD if ok else BAD)
                d.text((x, y + th + 4), label, font=f, fill=GOOD if ok else BAD)
            y += th + label_h + pad
        y += pad
    sheet.save(out_png, optimize=True)
    return cells


def _axis(p0: float, p1: float, v0: float, v1: float):
    """Linear map from value range [v0, v1] to pixel range [p0, p1]."""
    scale = (p1 - p0) / (v1 - v0)
    return lambda v: p0 + (v - v0) * scale


def lag_chart(
    syncs: Sequence[CameraSync],
    verifies: Sequence[VerifyResult],
    out_png: str | Path,
    *,
    truth: dict[str, AffineClock] | None = None,
    width: int = 960,
    panel_h: int = 230,
) -> None:
    """Per camera: measured (t_ref − t_cam − coarse offset) vs t_cam in ms, fitted line and
    held-out residuals. Filled dots = fit windows, hollow = held-out, grey = below confidence."""
    pad_l, pad_r, pad_t, pad_b = 70, 20, 34, 30
    H = len(syncs) * (panel_h + 10) + 38
    img = Image.new("RGB", (width, H), BG)
    d = ImageDraw.Draw(img)
    f, fs = _font(13), _font(11)
    vmap = {v.name: v for v in verifies}
    for i, s in enumerate(syncs):
        top = 10 + i * (panel_h + 10)
        x0, x1 = pad_l, width - pad_r
        y0, y1 = top + pad_t, top + panel_h - pad_b
        d.rectangle((x0, y0, x1, y1), outline=(50, 54, 68))
        coarse = s.coarse_offset or 0.0
        pts = [(w.t_cam, (w.t_ref - w.t_cam - coarse) * 1000, w.confidence, w.used, True)
               for w in s.windows if np.isfinite(w.t_ref)]
        v = vmap.get(s.name)
        if v is not None:
            pts += [(w.t_cam, (w.t_ref - w.t_cam - coarse) * 1000, w.confidence, w.used, False)
                    for w in v.windows if np.isfinite(w.t_ref)]
        title = f"camera {s.name}: {s.status}"
        if s.clock is not None:
            title += f"   drift {s.clock.drift_ppm:+.1f} ppm   offset {s.clock.offset:+.4f} s"
            if truth and s.name in truth:
                t = truth[s.name]
                title += (f"   (truth {t.drift_ppm:+.1f} ppm, {t.offset:+.4f} s; "
                          f"error {(s.clock.drift_ppm - t.drift_ppm):+.2f} ppm, "
                          f"{(s.clock.offset - t.offset) * 1000:+.2f} ms)")
        d.text((x0, top + 8), title, font=f, fill=FG)
        if not pts:
            d.text((x0 + 10, y0 + 10), s.reason, font=f, fill=BAD)
            continue
        tx = [p[0] for p in pts]
        ty = [p[1] for p in pts if p[3]] or [p[1] for p in pts]
        tmin, tmax = 0.0, max(s.cam_duration, max(tx))
        ymin, ymax = min(ty), max(ty)
        span = max(ymax - ymin, 2.0)
        ymin, ymax = ymin - 0.15 * span, ymax + 0.15 * span
        X = _axis(x0, x1, tmin, tmax)
        Y = _axis(y1, y0, ymin, ymax)

        for k in range(5):
            ms = ymin + (ymax - ymin) * k / 4
            d.line((x0, Y(ms), x1, Y(ms)), fill=(34, 37, 48))
            d.text((6, Y(ms) - 6), f"{ms:+.1f} ms", font=fs, fill=DIM)
        for k in range(6):
            t = tmin + (tmax - tmin) * k / 5
            d.text((X(t) - 10, y1 + 6), f"{t:.0f}s", font=fs, fill=DIM)
        if s.clock is not None:
            a = (s.clock.offset - coarse) * 1000
            b = (s.clock.rate - 1.0) * 1000
            d.line((X(tmin), Y(a + b * tmin), X(tmax), Y(a + b * tmax)), fill=ACCENT, width=2)
        for t, ms, _conf, used, is_fit in pts:
            x, y = X(t), Y(ms)
            if not used:
                d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=DIM)
            elif is_fit:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=GOOD)
            else:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 210, 90), width=2)
        if v is not None:
            txt = (f"held-out: {v.status}, {v.n_confident} windows, "
                   f"max |res| {v.max_abs_ms:.2f} ms, rms {v.rms_ms:.2f} ms "
                   f"(tolerance {v.tolerance_ms} ms)")
            d.text((x0, y1 + 16), txt, font=fs, fill=GOOD if v.status == "pass" else BAD)
    legend = "filled green = fit windows   hollow yellow = held-out   grey = below confidence   " \
             "line = fitted clock (relative to the coarse offset)"
    d.text((pad_l, H - 14), legend, font=fs, fill=DIM)
    img.save(out_png, optimize=True)
