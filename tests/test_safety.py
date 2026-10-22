"""Subprocess and path safety: argv lists only, no shell, inputs never become options."""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cueframe import media
from cueframe.cut import concat_entry, video_filter


def test_safe_path_neutralises_leading_dash(tmp_path):
    p = tmp_path / "-i.wav"
    s = media.safe_path(p)
    assert not s.startswith("-")
    assert s.startswith("./") or s.startswith("/")
    assert s.endswith("-i.wav")
    weird = tmp_path / "it's a 'quoted' name; rm -rf.mov"
    assert media.safe_path(weird) == str(weird.resolve())


def test_only_ffmpeg_and_ffprobe_may_run():
    with pytest.raises(media.MediaError):
        media.run(["sh", "-c", "echo hi"])
    with pytest.raises(media.MediaError):
        media.tool("bash")
    with pytest.raises(media.MediaError):
        media.run([])


def test_concat_entry_escapes_single_quotes(tmp_path):
    p = tmp_path / "clip 'one'.mp4"
    line = concat_entry(p)
    assert line.startswith("file '") and line.endswith("'\n")
    assert "'\\''" in line
    # round trip the escaping rule used by ffmpeg's concat demuxer
    inner = line[len("file '"):-2]
    assert inner.replace("'\\''", "'") == str(p.resolve())


def test_video_filter_is_numeric_only():
    f = video_filter("tight", (854, 480), (854, 480), "25")
    assert re.fullmatch(r"[a-zA-Z0-9=:,./_-]+", f)
    assert "crop=610:342:122:69" in f
    assert f.startswith("setpts=PTS-STARTPTS,fps=25")
    with pytest.raises(media.MediaError):
        video_filter("zoom", (854, 480), (854, 480), "25")


def test_mux_rejects_wrong_frame_shape(tmp_path):
    pytest.importorskip("numpy")
    if not media.__dict__.get("shutil").which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    bad = iter([np.zeros((10, 10, 3), dtype=np.uint8)])
    with pytest.raises(media.MediaError, match="expected 16x32x3"):
        media.mux_raw_frames(bad, width=32, height=16, fps=Fraction(25), n_frames=1,
                             audio_path=None, out_path=tmp_path / "x.mp4")


def test_subprocess_is_confined_to_media_module():
    src_dir = Path(media.__file__).parent
    offenders = []
    for py in src_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if py.name != "media.py" and ("subprocess" in text or "os.system" in text):
            offenders.append(py.name)
        assert "shell=True" not in text, py
    assert offenders == [], offenders


def test_cue_json_cannot_name_code():
    from cueframe.cues import CueError, parse_cues

    with pytest.raises(CueError, match="unknown top-level keys"):
        parse_cues({"fps": 25, "cues": [], "module": "os"})
    # a scene name is just a dictionary key; it is never imported
    sheet = parse_cues({"fps": 25, "cues": [{"start": 0, "end": 1, "scene": "os.system"}]})
    from cueframe.cues import validate_scene_names

    with pytest.raises(CueError, match="unknown scene"):
        validate_scene_names(sheet, {"idle": None})
