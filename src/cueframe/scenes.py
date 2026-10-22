"""Built-in scenes: dark, abstract, deterministic, and cheap enough for ~25 ms per frame.

Every scene is ``render(ctx: FrameContext, canvas: PIL.Image.Image) -> None`` and derives
everything it draws from ``ctx`` alone (no state across frames, no clock, no unseeded
randomness), so the same cue sheet, audio and ``--seed`` always give the same pixels.

Fields and gradients are computed with numpy at low resolution and upscaled; shapes are drawn
with ImageDraw on a 2x layer that is box-filtered down and added onto the canvas (cheap
anti-aliasing). Fonts are Pillow's built-in scalable font; no files are read.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont

from cueframe.animate import FrameContext, SceneRenderer

Color = tuple[int, int, int]
Painter = Callable[[ImageDraw.ImageDraw, int], None]

# --------------------------------------------------------------------------- easing


def _c01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def ease_out_cubic(x: float) -> float:
    x = _c01(x)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out_cubic(x: float) -> float:
    x = _c01(x)
    return 4.0 * x**3 if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 3 / 2.0


def ease_out_back(x: float, overshoot: float = 1.70158) -> float:
    """Ease-out that overshoots 1.0 slightly before settling."""
    x = _c01(x)
    c3 = overshoot + 1.0
    return 1.0 + c3 * (x - 1.0) ** 3 + overshoot * (x - 1.0) ** 2


# --------------------------------------------------------------------------- colour


@lru_cache(maxsize=64)
def _palette(seed: int) -> tuple[Color, ...]:
    hue = (((seed * 2654435761) & 0xFFFFFFFF) / 2**32 + 0.58) % 1.0
    # (hue offset, saturation, value): primary, secondary, accent, deep, soft
    spec = ((0.0, 0.70, 0.98), (0.10, 0.60, 0.90), (0.48, 0.55, 1.0), (0.0, 0.85, 0.40),
            (0.04, 0.20, 0.97))
    out = []
    for dh, s, v in spec:
        r, g, b = colorsys.hsv_to_rgb((hue + dh) % 1.0, s, v)
        out.append((round(r * 255), round(g * 255), round(b * 255)))
    return tuple(out)


def palette(seed: int) -> list[Color]:
    """Five consistent RGB colours for ``seed``: primary, secondary, accent, deep, soft."""
    return list(_palette(int(seed)))


def _scale(c: Color, k: float) -> Color:
    k = max(0.0, k)
    return (min(255, int(c[0] * k)), min(255, int(c[1] * k)), min(255, int(c[2] * k)))


def _mix(a: Color, b: Color, k: float) -> Color:
    k = _c01(k)
    return (int(a[0] + (b[0] - a[0]) * k), int(a[1] + (b[1] - a[1]) * k),
            int(a[2] + (b[2] - a[2]) * k))


# --------------------------------------------------------------------------- drawing helpers


@lru_cache(maxsize=64)
def font(size: int):
    """Built-in scalable font at ``size`` px (falls back to Pillow's fixed bitmap font)."""
    try:
        return ImageFont.load_default(size=max(6, int(size)))
    except (TypeError, OSError, ValueError):
        return ImageFont.load_default()


def _gradient(ctx: FrameContext, canvas: Image.Image, gain: float, speed: float = 0.12) -> None:
    """Paste a slowly drifting, vignetted two-colour gradient scaled by ``gain``."""
    pal = _palette(ctx.seed)
    sw, sh = max(8, ctx.width // 8), max(8, ctx.height // 8)
    x = np.linspace(0.0, 1.0, sw, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]
    ph = np.float32(ctx.t * speed * math.tau + (ctx.seed % 97) * 0.37)
    m = 0.5 + 0.25 * (np.sin(2.4 * x + 1.3 * y + ph) + np.sin(1.7 * y - 2.0 * x + 0.7 * ph + 1.0))
    vig = np.clip(1.15 - 1.6 * ((x - 0.5) ** 2 + (y - 0.5) ** 2), 0.0, 1.0)
    c0 = np.array(pal[3], dtype=np.float32)
    c1 = np.array(pal[1], dtype=np.float32)
    arr = (c0 + (c1 - c0) * m[..., None]) * (vig * gain)[..., None]
    _paste_field(canvas, arr)


def _paste_field(canvas: Image.Image, arr: np.ndarray) -> None:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    canvas.paste(img.resize(canvas.size, Image.Resampling.BILINEAR))


def _circle(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, s: int,
            ry: float | None = None, **kw) -> None:
    ry = r if ry is None else ry
    r, ry = max(r, 0.5), max(ry, 0.5)
    d.ellipse(((cx - r) * s, (cy - ry) * s, (cx + r) * s, (cy + ry) * s), **kw)


def _additive(canvas: Image.Image, paint: Painter, ss: int = 2) -> None:
    """Draw on a ``ss``x layer, box-filter it down and add it onto the canvas."""
    w, h = canvas.size
    layer = Image.new("RGB", (w * ss, h * ss))
    paint(ImageDraw.Draw(layer), ss)
    canvas.paste(ImageChops.add(canvas, layer.reduce(ss)))


def _wrap(text: str, fnt, max_w: float) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}" if cur else word
        if cur and fnt.getlength(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _clip(fnt, line: str, max_w: float, ellipsis: bool) -> str:
    tail = "..." if ellipsis else ""
    while line and fnt.getlength(line + tail) > max_w:
        line = line[:-1]
    return line + tail


def _fit(text: str, max_w: float, size: int, max_lines: int = 2, min_size: int = 8):
    """Largest font <= ``size`` for which ``text`` wraps to ``max_lines`` lines within ``max_w``."""
    text = " ".join(text.split())
    s = max(size, min_size)
    while True:
        fnt = font(s)
        lines = _wrap(text, fnt, max_w)
        if len(lines) <= max_lines and all(fnt.getlength(ln) <= max_w for ln in lines):
            return fnt, lines
        if s <= min_size:
            break
        s = max(min_size, s - max(1, s // 10))
    extra = len(lines) > max_lines
    lines = lines[:max_lines]
    lines = [_clip(fnt, ln, max_w, extra and i == len(lines) - 1) for i, ln in enumerate(lines)]
    return fnt, lines


def _line_metrics(fnt) -> tuple[int, int, int]:
    """(top offset, glyph height, line pitch) measured on a tall+descending sample."""
    _, t0, _, b0 = fnt.getbbox("Ag")
    th = max(1, b0 - t0)
    return t0, th, int(th * 1.3)


def _paint_lines(canvas: Image.Image, fnt, lines: list[str], cx: float, cy: float,
                 fill: Color, opacity: float = 1.0, shadow: bool = True) -> None:
    """Draw centred ``lines`` through an alpha mask so ``opacity`` is exact on any backdrop."""
    alpha = int(255 * _c01(opacity))
    if alpha <= 0 or not lines:
        return
    t0, th, pitch = _line_metrics(fnt)
    top = cy - (pitch * (len(lines) - 1) + th) / 2
    for i, line in enumerate(lines):
        bl, _, br, _ = fnt.getbbox(line)
        w = max(1, br - bl)
        mask = Image.new("L", (w + 8, th + 8), 0)
        ImageDraw.Draw(mask).text((4 - bl, 4 - t0), line, font=fnt, fill=alpha)
        x, y = round(cx - w / 2) - 4, round(top + i * pitch) - 4
        if shadow:
            canvas.paste((0, 0, 0), (x + 2, y + 2), mask)
        canvas.paste(fill, (x, y), mask)


def draw_centered_text(ctx: FrameContext, canvas: Image.Image, *, size: int,
                       opacity: float = 1.0, dy: float = 0.0, max_frac: float = 0.84,
                       fill: Color = (244, 246, 252)) -> tuple[int, int, int, int] | None:
    """Draw ``ctx.text`` centred (<= 2 lines). Returns the text block's bbox, or None."""
    text = ctx.text.strip()
    if not text:
        return None
    w, h = canvas.size
    fnt, lines = _fit(text, w * max_frac, size)
    _paint_lines(canvas, fnt, lines, w / 2, h / 2 + dy, fill, opacity)
    _, th, pitch = _line_metrics(fnt)
    bw = max(fnt.getlength(ln) for ln in lines)
    bh = pitch * (len(lines) - 1) + th
    x0, y0 = int(w / 2 - bw / 2), int(h / 2 + dy - bh / 2)
    return x0, y0, int(x0 + bw), y0 + bh


def draw_caption(ctx: FrameContext, canvas: Image.Image, *, position: str = "bottom") -> None:
    """Draw ``ctx.text`` in a rounded dark pill (<= 2 lines, within 90% of the width)."""
    text = ctx.text.strip()
    if not text:
        return
    if position not in ("bottom", "top", "center"):
        raise ValueError(f"position must be 'bottom', 'top' or 'center', not {position!r}")
    w, h = canvas.size
    pad = max(6, h // 45)
    fnt, lines = _fit(text, int(w * 0.9) - 2 * pad, max(10, h // 22))
    _, th, pitch = _line_metrics(fnt)
    pw = int(max(fnt.getlength(ln) for ln in lines)) + 2 * pad
    ph = pitch * (len(lines) - 1) + th + 2 * pad
    px = (w - pw) // 2
    py = {"bottom": h - int(h * 0.06) - ph, "top": int(h * 0.08), "center": (h - ph) // 2}[position]
    radius = min(ph // 2, 2 * pad)
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=radius, fill=190)
    canvas.paste((8, 9, 16), (px, py), mask)
    ImageDraw.Draw(canvas).rounded_rectangle(
        (px, py, px + pw - 1, py + ph - 1), radius=radius,
        outline=_scale(_palette(ctx.seed)[0], 0.35),
    )
    _paint_lines(canvas, fnt, lines, w / 2, py + ph / 2, (236, 238, 246), shadow=False)


def draw_scene_label(ctx: FrameContext, canvas: Image.Image, name: str) -> None:
    """Tiny ``scene: <name>`` tag in the top-left corner."""
    ImageDraw.Draw(canvas).text(
        (max(6, ctx.width // 96), max(4, ctx.height // 80)), f"scene: {name}",
        font=font(max(9, ctx.height // 48)), fill=(118, 122, 140),
    )


# --------------------------------------------------------------------------- scenes


def idle(ctx: FrameContext, canvas: Image.Image) -> None:
    """Very dim drifting gradient (used where no cue is active)."""
    _gradient(ctx, canvas, 0.14, 0.08)


def title(ctx: FrameContext, canvas: Image.Image) -> None:
    w, h = canvas.size
    pal = _palette(ctx.seed)
    sc = h / 540
    fps = float(ctx.fps)
    _gradient(ctx, canvas, 0.10)

    # breathing dotted field, drifting slowly, calmer toward the edges
    sp = max(12, round(h / 13.5))
    off = (ctx.t * 8.0) % sp
    gx, gy = np.meshgrid(np.arange(-sp, w + sp, sp) + off, np.arange(-sp, h + sp, sp) + off * 0.5)
    dist = np.hypot((gx - w / 2) / w, (gy - h / 2) / h)
    wave = 0.5 + 0.5 * np.sin(dist * 9.0 - ctx.t * 1.6)
    bright = np.clip((0.18 + 0.42 * wave) * np.clip(1.35 - dist * 1.8, 0, 1)
                     * (0.6 + 0.8 * ctx.level), 0, 1)
    radius = (1.3 + 1.8 * ctx.level * wave) * sc
    dots = zip(gx.ravel().tolist(), gy.ravel().tolist(), bright.ravel().tolist(),
               radius.ravel().tolist(), strict=True)

    def paint(d: ImageDraw.ImageDraw, s: int) -> None:
        for x, y, b, r in dots:
            if b > 0.02:
                _circle(d, x, y, r, s, fill=_scale(pal[1], b))

    _additive(canvas, paint)

    e = ease_out_cubic(ctx.cue_frame / (0.9 * fps))
    eb = ease_out_back(ctx.cue_frame / (1.2 * fps))
    box = draw_centered_text(ctx, canvas, size=max(14, h // 9), opacity=e, dy=(1 - e) * h * 0.05)
    if box is not None:
        x0, _, x1, y1 = box
        half = (x1 - x0) * 0.25 * eb
        ly = y1 + h * 0.03 + (1 - e) * h * 0.05
        ImageDraw.Draw(canvas).rounded_rectangle(
            (w / 2 - half, ly, w / 2 + half, ly + max(2, h // 180)), radius=2,
            fill=_scale(pal[2], e),
        )
    draw_scene_label(ctx, canvas, "title")


def pulse(ctx: FrameContext, canvas: Image.Image) -> None:
    w, h = canvas.size
    pal = _palette(ctx.seed)
    sc = h / 540
    lvl, ons = _c01(ctx.level), _c01(ctx.onset * 4.0)
    cx, cy = w / 2, h / 2
    rmax = math.hypot(cx, cy) * 0.8

    # radial glow that swells with level and flashes on onsets
    sw, sh = max(8, w // 8), max(8, h // 8)
    px = np.linspace(-0.5, 0.5, sw, dtype=np.float32)[None, :] * w
    py = np.linspace(-0.5, 0.5, sh, dtype=np.float32)[:, None] * h
    glow = np.exp(-4.0 * (px * px + py * py) / (rmax * rmax)) * min(1.0, 0.10 + 0.40 * lvl + ons)
    _paste_field(canvas, glow[..., None] * np.array(pal[0], dtype=np.float32) * 0.6)

    def paint(d: ImageDraw.ImageDraw, s: int) -> None:
        for i in range(5):  # rings born at the centre, fading as they expand
            u = (ctx.t * 0.45 + i / 5) % 1.0
            col = _scale(_mix(pal[0], pal[2], u), (1 - u) ** 1.5 * (0.30 + 0.70 * lvl))
            _circle(d, cx, cy, ease_out_cubic(u) * rmax * 0.95, s, outline=col,
                    width=max(1, round((1.5 + 4 * lvl) * sc * s)))
        r_lvl = rmax * (0.12 + 0.30 * lvl)  # level ring: radius follows the audio level
        _circle(d, cx, cy, r_lvl, s, outline=_scale(pal[4], 0.45 + 0.5 * lvl),
                width=max(1, round((2 + 6 * lvl) * sc * s)))
        if ons > 0.02:  # onset: a fresh bright ring bursts outward from the level ring
            _circle(d, cx, cy, r_lvl + rmax * 0.35 * ons, s, outline=_scale(pal[2], 0.35 + ons),
                    width=max(1, round((4 + 10 * ons) * sc * s)))
        core = rmax * (0.03 + 0.04 * lvl) * (1 + 0.8 * ons)
        _circle(d, cx, cy, core * 2.2, s, fill=_scale(pal[1], 0.15 + 0.3 * ons))
        _circle(d, cx, cy, core, s, fill=_scale(pal[4], 0.5 + 0.5 * max(lvl, ons)))

    _additive(canvas, paint)
    draw_caption(ctx, canvas)
    draw_scene_label(ctx, canvas, "pulse")


def orbit(ctx: FrameContext, canvas: Image.Image) -> None:
    w, h = canvas.size
    pal = _palette(ctx.seed)
    sc = h / 540
    cx, cy = w / 2, h / 2
    tl = ctx.cue_frame / float(ctx.fps)  # cue-local clock keeps speed changes from over-rotating
    speed = 0.4 + ctx.level
    n, trail, dt, rmax = 6, 10, 0.04, 0.42 * h
    seed_ph = (ctx.seed % 360) * math.pi / 180
    _gradient(ctx, canvas, 0.08, 0.10)

    def paint(d: ImageDraw.ImageDraw, s: int) -> None:
        radii = [rmax * (0.28 + 0.72 * i / (n - 1)) for i in range(n)]
        for r in radii:
            _circle(d, cx, cy, r * 1.5, s, ry=r * 0.95, outline=_scale(pal[4], 0.10), width=s)
        core = (8 + 6 * ctx.level) * sc
        _circle(d, cx, cy, core * 2.4, s, fill=_scale(pal[1], 0.12))
        _circle(d, cx, cy, core, s, fill=_scale(pal[4], 0.55 + 0.45 * ctx.level))
        for i, r in enumerate(radii):
            omega = (0.55 + 0.23 * i) * (1 if i % 2 == 0 else -1)
            phase = i * 2.1 + seed_ph
            col = pal[i % 3]
            rad = (5 + 0.9 * i + 4 * ctx.level) * sc
            for k in range(trail, -1, -1):  # oldest trail point first, head last
                ang = phase + omega * speed * (tl - k * dt)
                f = 1 - k / (trail + 1)
                x, y = cx + r * 1.5 * math.cos(ang), cy + r * 0.95 * math.sin(ang)
                if k == 0:
                    _circle(d, x, y, rad * 2.4, s, fill=_scale(col, 0.22))
                    _circle(d, x, y, rad, s, fill=col)
                else:
                    _circle(d, x, y, rad * (0.3 + 0.7 * f), s, fill=_scale(col, 0.7 * f**1.5))

    _additive(canvas, paint)
    draw_caption(ctx, canvas)
    draw_scene_label(ctx, canvas, "orbit")


def bars(ctx: FrameContext, canvas: Image.Image) -> None:
    w, h = canvas.size
    pal = _palette(ctx.seed)
    n = 40
    x0, slot = w * 0.08, w * 0.84 / n
    bw = slot * 0.62
    base, max_h = h * 0.88, h * 0.38
    _gradient(ctx, canvas, 0.08, 0.10)

    i = np.arange(n)
    ph = (ctx.seed % 628) / 100
    wave = (0.5 + 0.25 * np.sin(i * 0.37 - ctx.t * 3.1 + ph)
            + 0.25 * np.sin(i * 0.91 + ctx.t * 2.3 + 2 * ph))
    frac = np.clip(0.05 + 0.10 * wave + ctx.level * (0.25 + 0.70 * wave), 0.04, 1.0).tolist()

    def paint(d: ImageDraw.ImageDraw, s: int) -> None:
        for k, f in enumerate(frac):
            bh = max(bw, f * max_h)
            x = x0 + k * slot + (slot - bw) / 2
            col = _scale(_mix(pal[0], pal[2], k / (n - 1)), 0.55 + 0.45 * f)
            d.rounded_rectangle((x * s, (base - bh) * s, (x + bw) * s, base * s),
                                radius=bw / 2 * s, fill=col)
            d.rounded_rectangle((x * s, (base + 6) * s, (x + bw) * s, (base + 6 + bh * 0.18) * s),
                                radius=bw / 2 * s, fill=_scale(col, 0.18))

    _additive(canvas, paint)
    th = max(2, h // 135)
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, h - th, w, h), fill=(24, 26, 34))
    d.rectangle((0, h - th, w * ctx.frame / max(1, ctx.n_frames), h), fill=pal[0])
    draw_caption(ctx, canvas, position="top")
    draw_scene_label(ctx, canvas, "bars")


def outro(ctx: FrameContext, canvas: Image.Image) -> None:
    w, h = canvas.size
    pal = _palette(ctx.seed)
    k = 1.0 - _c01(ctx.cue_progress)  # 1 -> 0 over the cue
    _gradient(ctx, canvas, 0.14, 0.06)

    def paint(d: ImageDraw.ImageDraw, s: int) -> None:
        for i in range(3):  # slow rings that converge on the centre
            u = (ctx.t * 0.12 + i / 3) % 1.0
            r = (1 - u) * h * 0.42
            _circle(d, w / 2, h / 2, r, s, outline=_scale(pal[1], 0.35 * math.sin(math.pi * u)),
                    width=s)

    _additive(canvas, paint)
    canvas.paste(ImageEnhance.Brightness(canvas).enhance(k))
    draw_centered_text(ctx, canvas, size=max(10, h // 22), opacity=k, fill=(200, 204, 216))
    draw_scene_label(ctx, canvas, "outro")


SCENES: dict[str, SceneRenderer] = {
    "idle": idle,
    "title": title,
    "pulse": pulse,
    "orbit": orbit,
    "bars": bars,
    "outro": outro,
}

SCENE_DESCRIPTIONS: dict[str, str] = {
    "idle": "Very dim, slowly drifting gradient used for frames with no cue; draws no text.",
    "title": "Title card: large centred text eases in over a dotted field that breathes "
             "with the audio level.",
    "pulse": "Concentric rings radiate from the centre, following the audio level and "
             "flashing on onsets.",
    "orbit": "Glowing discs with short trails orbit a core at speeds that scale with "
             "the audio level.",
    "bars": "A rounded audio-reactive bar strip that ripples over time, with a progress "
            "bar along the bottom edge.",
    "outro": "A slow ring fade toward black over the cue with small centred text.",
}
