import json
from pathlib import Path

import pytest
from PIL import Image

from examples.hk2.motion import (
    FPS,
    Captions,
    Illustration,
    Type,
    active_word,
    card_geom,
    on,
    page_at,
)

ROOT = Path(__file__).resolve().parents[1]
PREPARED, INPUTS = ROOT / "work/hk2/prepared", ROOT / "work/hk2/inputs"
FONTS = (INPUTS / "SpaceGrotesk-Regular.ttf", INPUTS / "SpaceGrotesk-Medium.ttf")


def test_event_is_frame_centred():
    assert on(10.0 + 0.5 / FPS, 10.0)
    assert not on(10.0 + 0.5 / FPS + 1e-6, 10.0)


def test_low_confidence_current_word_suppresses_highlight_without_falling_back():
    words = [{"text": "a", "body_start_s": 1.0, "reliable": True},
             {"text": "hey,", "body_start_s": 1.5, "reliable": False},
             {"text": "b", "body_start_s": 2.0, "reliable": True}]
    page = {"word_indices": [0, 1, 2]}
    assert active_word(page, words, 29) is None  # before the first word (0.983 s centre)
    assert active_word(page, words, 30) == 0
    assert active_word(page, words, 45) is None  # 'hey,' is current but unreliable
    assert active_word(page, words, 60) == 2


def test_caption_pages_are_half_open_on_the_body_clock():
    pages = [{"page": 1, "body_frames": [304, 329]}, {"page": 2, "body_frames": [329, 340]}]
    assert page_at(pages, 303) is None
    assert page_at(pages, 328)["page"] == 1
    assert page_at(pages, 329)["page"] == 2
    assert page_at(pages, 340) is None


def test_card_steps_back_monotonically_and_lands_on_the_schematic():
    full, s1 = (170, 300, 740, 840), (355, 140, 370, 420)
    assert card_geom(15.89, 15.9, 0.6, full, s1) == (full, 0.0)
    widths = [card_geom(15.9 + k / FPS, 15.9, 0.6, full, s1)[0][2] for k in range(19)]
    assert widths == sorted(widths, reverse=True)
    assert card_geom(16.5, 15.9, 0.6, full, s1) == (s1, 1.0)


@pytest.mark.skipif(not all(p.exists() for p in (PREPARED / "plan.json", *FONTS)),
                    reason="local HK #2 media is not present")
def test_local_port_reproduces_recipe_geometry_and_final_gate_state():
    plan = json.loads((PREPARED / "plan.json").read_text(encoding="utf-8"))
    ty = Type(*FONTS)
    captions, ill = Captions(plan, ty), Illustration(plan, ty)  # metric/geometry cross-checks
    offset = plan["origins"]["body_frames"][0]
    first, last = plan["motion"]["excerpt_frames"][0], plan["frames"] - 1

    fr, info = ill.frame(first + offset)
    assert info == {"card": list(plan["motion"]["geometry"]["card_full"])}
    assert captions.draw(fr, first + offset, paper=True)[0] is not None

    fr, info = ill.frame(last + offset)
    assert info["gate"] == "Lawyer review required" and info["cells"] == 6 and info["dest"]
    assert info["card"] == plan["motion"]["geometry"]["card_schematic"]
    assert fr.size == (1080, 1920)

    blank = Image.new("RGBA", (1080, 1920))
    assert captions.draw(blank, offset, paper=False) == (None, None)  # handle frame 0
    hey = next(w for w in plan["captions"]["words"] if w["text"] == "hey,")
    frame = round(hey["body_start_s"] * FPS) + 1
    assert captions.draw(blank, frame, paper=False)[1] is None
