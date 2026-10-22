"""Cue file validation: every rejection names the offending cue; gaps are explicit."""

import json
from fractions import Fraction

import pytest

from cueframe.cues import CueError, dump_cues, load_cues, parse_cues, validate_scene_names


def doc(**over):
    base = {
        "fps": 25,
        "size": [320, 180],
        "cues": [
            {"start": 0.0, "end": 1.0, "scene": "title", "text": "a"},
            {"start": 1.0, "end": 2.0, "scene": "pulse", "text": "b"},
        ],
    }
    base.update(over)
    return base


def test_parse_ok_and_timeline():
    sheet = parse_cues(doc())
    assert sheet.fps == Fraction(25)
    assert sheet.size == (320, 180)
    tl = sheet.to_timeline(2.0)
    assert [(s.f0, s.f1, s.payload.scene if s.payload else None) for s in tl.segments] == [
        (0, 25, "title"),
        (25, 50, "pulse"),
    ]


def test_gap_is_explicit_none_segment():
    sheet = parse_cues(doc(cues=[
        {"start": 0.0, "end": 0.8, "scene": "title"},
        {"start": 1.2, "end": 2.0, "scene": "pulse"},
    ]))
    tl = sheet.to_timeline(2.4)
    assert [(s.f0, s.f1, s.payload is None) for s in tl.segments] == [
        (0, 20, False), (20, 30, True), (30, 50, False), (50, 60, True),
    ]


@pytest.mark.parametrize(
    ("cues", "pattern"),
    [
        ([{"start": 0.0, "end": 1.0, "scene": "a"}, {"start": 0.5, "end": 2.0, "scene": "a"}],
         r"cue 1: .*overlaps"),
        ([{"start": 1.0, "end": 2.0, "scene": "a"}, {"start": 0.0, "end": 0.5, "scene": "a"}],
         r"cue 1: .*overlaps"),
        ([{"start": -0.1, "end": 1.0, "scene": "a"}], r"cue 0: start -0.1 is negative"),
        ([{"start": 1.0, "end": 1.0, "scene": "a"}], r"cue 0: end 1.0 is not after"),
        ([{"start": 1.0, "end": 0.5, "scene": "a"}], r"cue 0: end 0.5 is not after"),
        ([{"start": 0.0, "end": 1.0}], r"cue 0: missing 'scene'"),
        ([{"start": "0", "end": 1.0, "scene": "a"}], r"cue 0: start must be a number"),
        ([{"start": 0.0, "end": float("inf"), "scene": "a"}], r"cue 0: end must be finite"),
        ([{"start": 0.0, "end": 1.0, "scene": ""}], r"cue 0: scene must be"),
        ([{"start": 0.0, "end": 1.0, "scene": "a", "text": "x" * 201}], r"cue 0: text longer"),
        (["not an object"], r"cue 0: must be an object"),
    ],
)
def test_structural_rejections_name_the_cue(cues, pattern):
    with pytest.raises(CueError, match=pattern):
        parse_cues(doc(cues=cues))


def test_top_level_rejections():
    with pytest.raises(CueError, match="top level must be an object"):
        parse_cues([])
    with pytest.raises(CueError, match="unknown top-level keys"):
        parse_cues(doc(renderer="evil.module"))
    with pytest.raises(CueError, match="required"):
        parse_cues({"cues": []})
    with pytest.raises(CueError, match="size"):
        parse_cues(doc(size=[321, 180]))
    with pytest.raises(CueError, match="size"):
        parse_cues(doc(size=[10, 10]))
    with pytest.raises(CueError, match="crossfade"):
        parse_cues(doc(crossfade=-1))
    with pytest.raises(CueError, match="fps"):
        parse_cues(doc(fps="0"))


def test_timing_rejections_against_audio_duration():
    sheet = parse_cues(doc())
    with pytest.raises(CueError, match=r"cue 1: end 2.0s is beyond the audio duration"):
        sheet.to_timeline(1.9)
    tiny = parse_cues(doc(cues=[{"start": 0.0, "end": 0.01, "scene": "a"}]))
    with pytest.raises(CueError, match=r"cue 0: .*covers no whole frame"):
        tiny.to_timeline(1.0)


def test_scene_name_validation():
    sheet = parse_cues(doc())
    validate_scene_names(sheet, {"idle": None, "title": None, "pulse": None})
    with pytest.raises(CueError, match=r"cue 1: unknown scene 'pulse'"):
        validate_scene_names(sheet, {"idle": None, "title": None})
    with pytest.raises(CueError, match="no 'idle' scene"):
        validate_scene_names(sheet, {"title": None, "pulse": None})


def test_extra_keys_preserved_and_roundtrip(tmp_path):
    sheet = parse_cues(doc(cues=[{"start": 0.0, "end": 1.0, "scene": "a", "colour": "#fff"}]))
    assert sheet.cues[0].extra == {"colour": "#fff"}
    path = tmp_path / "cues.json"
    dump_cues(sheet, path)
    again = load_cues(path)
    assert again == sheet
    assert json.loads(path.read_text())["cues"][0]["colour"] == "#fff"


def test_load_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(CueError, match="invalid JSON"):
        load_cues(p)
