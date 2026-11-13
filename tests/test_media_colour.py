"""Check the encoded stream's colour declaration, not only FFmpeg arguments."""

import json
from fractions import Fraction

import numpy as np
import pytest

from cueframe.media import mux_raw_frames, run, safe_path


@pytest.mark.ffmpeg
def test_raw_rgb_encode_retains_bt709_in_stream_and_decoded_frames(tmp_path):
    out = tmp_path / "colour.mp4"
    frames = (np.full((32, 64, 3), level, dtype=np.uint8) for level in (20, 220))
    mux_raw_frames(frames, width=64, height=32, fps=Fraction(30), n_frames=2,
                   audio_path=None, out_path=out)
    fields = "color_space,color_primaries,color_transfer"
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                  f"stream={fields}:frame={fields}", "-of", "json", safe_path(out)])
    info = json.loads(result.stdout)
    assert len(info["streams"]) == 1 and len(info["frames"]) == 2
    for entry in info["streams"] + info["frames"]:
        assert [entry.get(key) for key in fields.split(",")] == ["bt709"] * 3
