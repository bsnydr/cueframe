"""Render a synthetic "geometric stage" as seen by virtual cameras.

The world is shared: at a given ``t_ref`` every camera sees identical shape positions in world
coordinates (a 16 x 9 unit stage), only the affine framing and colour grade differ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

WORLD_W, WORLD_H = 16.0, 9.0
HORIZON_Y = 3.0
TAU = 2.0 * math.pi

# framing -> (zoom relative to "wide", world x of view centre, world y of view centre)
_FRAMINGS = {
    "wide": (1.0, 8.0, 4.5),
    "left": (1.3, 5.8, 4.6),
    "right": (1.3, 10.2, 4.6),
}
# palette -> (per-channel gain <= 1, vignette strength, accent colour of the label)
_PALETTES = {
    0: ((1.00, 1.00, 1.00), 0.30, (235, 235, 235)),
    1: ((1.00, 0.95, 0.80), 0.45, (255, 190, 90)),
    2: ((0.80, 0.94, 1.00), 0.60, (120, 200, 255)),
}


@dataclass(frozen=True)
class CameraView:
    name: str  # short label such as "A", "B", "C"
    framing: str  # "wide" | "left" | "right"
    palette: int  # 0, 1, 2 -- selects a distinct tint/vignette


def _sines(t: float, *terms: tuple[float, float, float]) -> float:
    """Sum of amplitude * sin(2 pi t / period + phase) terms."""
    return sum(a * math.sin(TAU * t / p + ph) for a, p, ph in terms)


def _shape_state(t: float) -> dict[str, tuple[float, ...]]:
    """World-space state of the five shapes at reference time ``t``."""
    return {
        "disc_a": (
            5.0 + _sines(t, (2.9, 9.7, 0.3), (1.0, 4.13, 1.9)),
            5.6 + _sines(t, (1.0, 6.9, 1.1), (0.4, 3.11, 0.2)),
        ),
        "disc_b": (
            11.0 + _sines(t, (2.7, 13.7, 2.2), (0.9, 5.17, 0.7)),
            5.2 + _sines(t, (1.1, 8.9, 0.4), (0.4, 3.77, 2.5)),
        ),
        "bar": (
            8.0 + _sines(t, (3.2, 11.3, 4.0), (0.8, 6.37, 1.3)),
            7.0 + _sines(t, (0.5, 7.31, 2.9), (0.2, 2.93, 0.6)),
            _sines(t, (0.55, 12.3, 0.0), (0.2, 4.7, 1.0)),
        ),
        "hexagon": (
            7.0 + _sines(t, (2.5, 10.1, 5.1), (1.0, 4.49, 0.9)),
            4.6 + _sines(t, (0.8, 8.3, 1.7), (0.3, 3.53, 0.1)),
            TAU * t / 8.3,
        ),
        "square": (
            9.5 + _sines(t, (3.4, 7.9, 3.3), (1.0, 3.07, 2.0)),
            6.3 + _sines(t, (1.0, 5.67, 5.5), (0.3, 2.71, 1.4)),
            -TAU * t / 5.9,
        ),
    }


def _poly(cx: float, cy: float, radius: float, n: int, rot: float) -> list[tuple[float, float]]:
    angles = [rot + TAU * k / n for k in range(n)]
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


@lru_cache(maxsize=32)
def _base_image(size: tuple[int, int], framing: str) -> Image.Image:
    """Static floor/wall gradient (rows only, so it is separable and cheap)."""
    w, h = size
    zoom, _, cy = _FRAMINGS[framing]
    scale = w / WORLD_W * zoom
    wy = (np.arange(h, dtype=np.float32) + 0.5 - h / 2) / scale + cy  # world y of each row
    below = np.clip((wy - HORIZON_Y) / (WORLD_H - HORIZON_Y), 0.0, 1.0)[:, None]
    wall = np.array([26, 30, 46], np.float32)
    floor_near = np.array([10, 12, 18], np.float32)
    floor_far = np.array([40, 44, 60], np.float32)
    col = np.where(wy[:, None] < HORIZON_Y, wall, floor_far + (floor_near - floor_far) * below)
    img = Image.fromarray(np.broadcast_to(col[:, None, :], (h, w, 3)).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    hy = (HORIZON_Y - cy) * scale + h / 2
    d.line([(0, hy), (w, hy)], fill=(90, 100, 130), width=max(1, h // 150))
    return img


@lru_cache(maxsize=32)
def _grade_map(size: tuple[int, int], palette: int) -> Image.Image:
    """Multiplicative tint + vignette map (RGB, values <= 255)."""
    w, h = size
    gain, strength, _ = _PALETTES[palette]
    lw, lh = max(2, w // 8), max(2, h // 8)
    xs = (np.arange(lw, dtype=np.float32) + 0.5) / lw * 2 - 1
    ys = (np.arange(lh, dtype=np.float32) + 0.5) / lh * 2 - 1
    r2 = (xs[None, :] ** 2 + ys[:, None] ** 2) / 2.0
    vig = 1.0 - strength * r2**1.2
    m = vig[:, :, None] * np.array(gain, np.float32)[None, None, :]
    small = Image.fromarray(np.clip(m * 255, 0, 255).astype(np.uint8), "RGB")
    return small.resize((w, h), Image.BILINEAR)


def _spotlight(t: float, size: tuple[int, int], framing: str) -> Image.Image:
    w, h = size
    zoom, cx, cy = _FRAMINGS[framing]
    scale = w / WORLD_W * zoom
    lw, lh = max(2, w // 6), max(2, h // 6)
    wx = ((np.arange(lw, dtype=np.float32) + 0.5) * 6 - w / 2) / scale + cx
    wy = ((np.arange(lh, dtype=np.float32) + 0.5) * 6 - h / 2) / scale + cy
    sx = 8.0 + _sines(t, (5.0, 23.7, 0.5))
    sy = 5.6 + _sines(t, (0.8, 17.3, 1.0))
    d2 = ((wx[None, :] - sx) / 3.4) ** 2 + ((wy[:, None] - sy) / 1.5) ** 2
    glow = np.exp(-1.6 * d2)[:, :, None] * np.array([46, 42, 32], np.float32)
    small = Image.fromarray(np.clip(glow, 0, 255).astype(np.uint8), "RGB")
    return small.resize((w, h), Image.BILINEAR)


@lru_cache(maxsize=16)
def _font(px: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=px)
    except Exception:  # noqa: BLE001 - older Pillow or missing FreeType: bitmap fallback
        return ImageFont.load_default()


def _stamp(img: Image.Image, view: CameraView, lines: tuple[str, ...]) -> None:
    w, h = img.size
    accent = _PALETTES[view.palette][2]

    def dim_box(box: tuple[int, int, int, int]) -> None:
        box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
        region = img.crop(box)
        img.paste(Image.blend(region, Image.new("RGB", region.size, (0, 0, 0)), 0.6), box)

    d = ImageDraw.Draw(img)
    if lines:
        font = _font(max(9, round(h / 30)))
        pad = max(3, h // 90)
        boxes = [d.textbbox((0, 0), s, font=font) for s in lines]
        lh = max(b[3] for b in boxes) + max(2, pad // 2)
        bw = max(b[2] for b in boxes) + 2 * pad
        dim_box((0, 0, bw, lh * len(lines) + 2 * pad))
        for i, s in enumerate(lines):
            d.text((pad, pad + i * lh), s, font=font, fill=(225, 230, 235))
    big = _font(max(14, round(h / 7)))
    bb = d.textbbox((0, 0), view.name, font=big)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    m = max(4, h // 60)
    x0 = w - tw - 3 * m
    dim_box((x0 - m, m, w - m, th + 4 * m))
    d.text((x0 - bb[0], 2 * m - bb[1]), view.name, font=big, fill=accent)


def render_stage_frame(
    t_ref: float,
    view: CameraView,
    size: tuple[int, int],
    *,
    flash: float = 0.0,
    overlay_lines: tuple[str, ...] = (),
) -> Image.Image:
    """Render the stage at reference time ``t_ref`` as seen by ``view``."""
    if view.framing not in _FRAMINGS:
        raise ValueError(f"unknown framing {view.framing!r}")
    if view.palette not in _PALETTES:
        raise ValueError(f"unknown palette {view.palette!r}")
    w, h = size
    zoom, cx, cy = _FRAMINGS[view.framing]
    scale = w / WORLD_W * zoom

    def sc(x: float, y: float) -> tuple[float, float]:
        return (x - cx) * scale + w / 2, (y - cy) * scale + h / 2

    img = ImageChops.add(_base_image(size, view.framing), _spotlight(t_ref, size, view.framing))
    d = ImageDraw.Draw(img)

    # faint world-fixed floor grid so framing differences are obvious
    for gx in range(-2, 19, 2):
        x0, y0 = sc(gx, HORIZON_Y)
        x1, y1 = sc(8 + (gx - 8) * 1.8, WORLD_H + 3)
        d.line([(x0, y0), (x1, y1)], fill=(34, 38, 54), width=1)

    st = _shape_state(t_ref)
    outline = (235, 235, 245)
    ow = max(1, h // 120)

    def shadow(x: float, y: float, rx: float) -> None:
        sx, sy = sc(x, y + 0.85)
        ex, ey = rx * scale, 0.25 * rx * scale
        d.ellipse([sx - ex, sy - ey, sx + ex, sy + ey], fill=(8, 9, 14))

    for key, col, r in (("disc_a", (225, 70, 70), 0.8), ("disc_b", (50, 200, 190), 0.65)):
        x, y = st[key]
        shadow(x, y, r)
        sx, sy = sc(x, y)
        d.ellipse([sx - r * scale, sy - r * scale, sx + r * scale, sy + r * scale],
                  fill=col, outline=outline, width=ow)

    bx, by, ba = st["bar"]
    bar = [(bx + dx * math.cos(ba) - dy * math.sin(ba), by + dx * math.sin(ba) + dy * math.cos(ba))
           for dx, dy in ((-1.2, -0.25), (1.2, -0.25), (1.2, 0.25), (-1.2, 0.25))]
    shadow(bx, by, 1.2)
    d.polygon([sc(*p) for p in bar], fill=(245, 200, 60), outline=outline, width=ow)

    hx, hy, ha = st["hexagon"]
    shadow(hx, hy, 0.9)
    d.polygon([sc(*p) for p in _poly(hx, hy, 0.9, 6, ha)], fill=(190, 90, 235),
              outline=outline, width=ow)

    qx, qy, qa = st["square"]
    shadow(qx, qy, 0.5)
    d.polygon([sc(*p) for p in _poly(qx, qy, 0.5, 4, qa + math.pi / 4)], fill=(110, 225, 110),
              outline=outline, width=ow)

    img = ImageChops.multiply(img, _grade_map(size, view.palette))
    f = min(1.0, max(0.0, flash))
    if f > 0.0:
        img = Image.blend(img, Image.new("RGB", size, (255, 255, 255)), 0.85 * f)
    if overlay_lines:
        _stamp(img, view, tuple(overlay_lines))
    return img
