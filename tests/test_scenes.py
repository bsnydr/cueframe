"""Unit tests for the built-in scenes and the example custom scene module (no ffmpeg, no I/O)."""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cueframe.animate import FrameContext, load_scene_module, scene_set_from_module
from cueframe.cues import Cue
from cueframe.scenes import (
    SCENE_DESCRIPTIONS,
    SCENES,
    draw_caption,
    ease_in_out_cubic,
    ease_out_back,
    ease_out_cubic,
    palette,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "custom_scenes.py"


def make_ctx(scene="title", *, frame=10, size=(160, 90), level=0.3, onset=0.0, progress=0.5,
             text="Hello world", seed=0, n_frames=100, cue_frame=None):
    cue = None if scene == "idle" else Cue(0.0, 4.0, scene, text)
    return FrameContext(
        frame=frame, t=frame / 25, fps=Fraction(25), width=size[0], height=size[1],
        n_frames=n_frames, cue=cue, cue_index=None if cue is None else 0,
        cue_progress=0.0 if cue is None else progress,
        cue_frame=frame if cue_frame is None else cue_frame,
        level=level, onset=onset, seed=seed,
    )


def render(fn, ctx):
    canvas = Image.new("RGB", (ctx.width, ctx.height), (0, 0, 0))
    fn(ctx, canvas)
    return canvas


def mean(img):
    return float(np.asarray(img, dtype=np.float64).mean())


def test_scene_set_contents():
    expected = {"idle", "title", "pulse", "orbit", "bars", "outro"}
    assert set(SCENES) == expected
    assert set(SCENE_DESCRIPTIONS) == expected
    assert all(isinstance(v, str) and v for v in SCENE_DESCRIPTIONS.values())


@pytest.mark.parametrize("name", sorted(SCENES))
def test_renders_without_changing_canvas(name):
    for frame in (0, 1, 37, 99):
        ctx = make_ctx(name, frame=frame, progress=frame / 99, level=(frame % 5) / 4, onset=0.2)
        img = render(SCENES[name], ctx)
        assert img.size == (160, 90)
        assert img.mode == "RGB"


@pytest.mark.parametrize("name", sorted(SCENES))
@pytest.mark.parametrize("text", ["", "x" * 200, "A fairly long caption that has to wrap " * 4])
def test_handles_odd_text_and_sizes(name, text):
    for size in ((160, 90), (64, 48)):
        img = render(SCENES[name], make_ctx(name, text=text, size=size))
        assert img.size == size


@pytest.mark.parametrize("name", sorted(SCENES))
def test_deterministic(name):
    ctx = make_ctx(name, frame=33, level=0.6, onset=0.3, progress=0.4, seed=7)
    assert render(SCENES[name], ctx).tobytes() == render(SCENES[name], ctx).tobytes()


def test_seed_changes_look():
    a = render(SCENES["bars"], make_ctx("bars", seed=0)).tobytes()
    b = render(SCENES["bars"], make_ctx("bars", seed=1)).tobytes()
    assert a != b


def test_idle_is_dim():
    for frame in (0, 50, 99):
        assert mean(render(SCENES["idle"], make_ctx("idle", frame=frame, level=1.0))) < 40


def test_pulse_reacts_to_level_and_onset():
    quiet = render(SCENES["pulse"], make_ctx("pulse", level=0.0, onset=0.0))
    loud = render(SCENES["pulse"], make_ctx("pulse", level=1.0, onset=0.5))
    assert mean(loud) > mean(quiet)


def test_outro_fades_to_black():
    early = render(SCENES["outro"], make_ctx("outro", progress=0.05))
    late = render(SCENES["outro"], make_ctx("outro", progress=0.95))
    assert mean(late) < mean(early)


def test_caption_pill_stays_within_width():
    ctx = make_ctx("pulse", text="word " * 60, size=(320, 180))
    canvas = Image.new("RGB", (320, 180), (0, 0, 0))
    draw_caption(ctx, canvas)
    cols = np.asarray(canvas).sum(axis=(0, 2))
    lit = np.nonzero(cols)[0]
    assert lit.min() >= 320 * 0.05 - 1 and lit.max() <= 320 * 0.95 + 1


def test_caption_rejects_bad_position():
    with pytest.raises(ValueError):
        draw_caption(make_ctx("pulse"), Image.new("RGB", (160, 90)), position="left")


def test_easing_and_palette():
    for fn in (ease_out_cubic, ease_in_out_cubic, ease_out_back):
        assert fn(0.0) == pytest.approx(0.0)
        assert fn(1.0) == pytest.approx(1.0)
        assert fn(-1.0) == pytest.approx(0.0)
    assert ease_out_back(0.8) > 1.0
    assert ease_out_cubic(0.5) > 0.5 > ease_in_out_cubic(0.4)
    assert palette(3) == palette(3)
    assert palette(3) != palette(4)
    assert 4 <= len(palette(0)) <= 5
    assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in palette(-5))


def test_example_marker_fills_every_pixel():
    scenes = scene_set_from_module(load_scene_module(EXAMPLE))
    assert set(scenes) == {"marker", "comet"}
    ctx = make_ctx("marker", text="should not be drawn")
    arr = np.asarray(render(scenes["marker"], ctx))
    assert (arr == np.array([220, 40, 120], dtype=np.uint8)).all()


def test_example_comet_is_deterministic():
    scenes = scene_set_from_module(load_scene_module(EXAMPLE))
    ctx = make_ctx("comet", frame=20, progress=0.3)
    first = render(scenes["comet"], ctx)
    assert first.size == (160, 90) and first.mode == "RGB"
    assert first.tobytes() == render(scenes["comet"], ctx).tobytes()


@pytest.mark.parametrize("name", ["title", "pulse", "orbit", "bars", "outro"])
def test_render_speed_at_960x540(name):
    ctxs = [
        make_ctx(name, frame=f, size=(960, 540), progress=f / 10, level=0.5, onset=0.1,
                 text="A reasonably sized title for speed")
        for f in range(5)
    ]
    render(SCENES[name], ctxs[0])  # warm font and import caches
    start = time.perf_counter()
    for ctx in ctxs:
        render(SCENES[name], ctx)
    assert (time.perf_counter() - start) / len(ctxs) < 0.08
