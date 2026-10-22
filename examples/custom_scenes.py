"""Example custom scenes for cueframe -- copy this file and edit it to make your own.

How it is used
--------------
Scenes are *trusted local Python*. cueframe only ever loads a scene module that you name
explicitly on the command line::

    python -m cueframe animate audio.wav cues.json out.mp4 --scenes examples/custom_scenes.py

Cue JSON never names code, only scene names: a cue with ``"scene": "comet"`` is drawn by
the ``comet`` function below. Scenes from this module are layered on top of the built-ins
(same name = override), and the scene set must always contain ``"idle"`` (the built-in one
is kept unless you define your own).

What a scene is
---------------
A scene is a function ``render(ctx, canvas)`` that draws onto ``canvas``, a Pillow RGB image
that is already filled black and has size ``ctx.width`` x ``ctx.height``. Draw in place;
never replace the image or change its size or mode. The module must expose them in a dict
called ``SCENES`` mapping name -> function.

Fields of ``ctx`` (a ``cueframe.animate.FrameContext``)
-------------------------------------------------------
* ``frame``, ``n_frames`` -- this frame's index and the total frame count
* ``t``                   -- time of this frame in seconds (``frame / fps``)
* ``fps``                 -- exact frame rate as a ``fractions.Fraction``
* ``width``, ``height``   -- output size in pixels
* ``cue``, ``cue_index``  -- the active ``Cue`` (start, end, scene, text, extra) or None
* ``text``                -- shortcut for the cue text ("" when there is no cue)
* ``cue_progress``        -- 0.0 at the cue's first frame to 1.0 at its last frame
* ``cue_frame``           -- frames since the cue started
* ``level``               -- smoothed audio level, 0..1
* ``onset``               -- positive level change on this frame, 0..1
* ``seed``                -- the per-render ``--seed``, for deterministic variation
* ``time_left``           -- seconds until the end of the video

Rules of thumb
--------------
* Be deterministic: the same ``ctx`` must give the same pixels. Derive everything from the
  fields above. If you need randomness, build ``numpy.random.default_rng(ctx.seed)`` inside
  the function; do not use ``time``, global ``random`` or state kept between frames.
* Be fast: aim for under ~25 ms per frame at 960x540. Use numpy for fields and gradients,
  ImageDraw for shapes, and avoid per-pixel Python loops.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from cueframe.animate import FrameContext
from cueframe.scenes import draw_caption, draw_scene_label, ease_in_out_cubic, palette

MARKER_COLOR = (220, 40, 120)


def marker(ctx: FrameContext, canvas: Image.Image) -> None:
    """Solid colour on every pixel and nothing else -- handy for checking which scene is live."""
    canvas.paste(MARKER_COLOR, (0, 0, canvas.width, canvas.height))


def comet(ctx: FrameContext, canvas: Image.Image) -> None:
    """A comet sweeping along an eased, curved path over the cue, above a twinkling star field."""
    w, h = canvas.size
    sc = h / 540  # scale sizes so the look is the same at any resolution
    pal = palette(ctx.seed)
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, w, h), fill=(5, 7, 15))

    # Stars: fixed positions and phases come from the seed; twinkle comes from the frame number.
    rng = np.random.default_rng(ctx.seed)
    n = 90
    xs, ys = rng.random(n) * w, rng.random(n) * h
    phase, rate = rng.random(n) * math.tau, 0.05 + rng.random(n) * 0.12
    twinkle = 0.5 + 0.5 * np.sin(ctx.frame * rate + phase)
    for x, y, tw in zip(xs.tolist(), ys.tolist(), twinkle.tolist(), strict=True):
        r = (0.7 + 1.1 * tw) * sc
        v = int(55 + 170 * tw)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(v, v, min(255, v + 30)))

    def path(progress: float) -> tuple[float, float]:
        """Left to right along an arch; eased so the comet accelerates, then settles."""
        q = ease_in_out_cubic(progress)
        return w * (0.08 + 0.84 * q), h * (0.78 - 0.5 * math.sin(math.pi * q) ** 0.8)

    # Tail: circles at earlier path positions, oldest first so the head is drawn on top.
    tail = 28
    for k in range(tail, 0, -1):
        f = 1 - k / (tail + 1)  # 0 at the tail tip -> ~1 next to the head
        x, y = path(max(0.0, ctx.cue_progress - k * 0.012))
        r = (2 + 9 * f) * sc
        c = pal[1] if f < 0.6 else pal[0]
        d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v * 0.8 * f**2) for v in c))

    hx, hy = path(ctx.cue_progress)
    for radius, color in ((26 * sc, tuple(v // 6 for v in pal[0])),
                          (15 * sc, tuple(v // 2 for v in pal[0])),
                          (8 * sc, pal[4])):
        d.ellipse((hx - radius, hy - radius, hx + radius, hy + radius), fill=color)

    draw_caption(ctx, canvas)  # text from the cue (nothing is drawn if it is empty)
    draw_scene_label(ctx, canvas, "comet")


SCENES = {
    "marker": marker,
    "comet": comet,
}
