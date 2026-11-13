from fractions import Fraction

import pytest

from cueframe.cues import parse_cues
from examples.hk2.run import (
    captions,
    cue_document,
    lobby_range,
    pcm_range,
    sample_bounds,
    write_voice,
)


def test_pcm_sample_range_preserves_24_bit_stereo_bytes(tmp_path):
    path = tmp_path / "source.wav"
    fmt = (48000, 2, 3)
    # Three frame periods, with a distinct byte pattern per sample frame.
    chunks = [bytes(range(6)), bytes(range(6, 12)), bytes(range(12, 18))]
    write_voice(path, fmt, b"".join(c * 1600 for c in chunks))
    start, stop = sample_bounds(1, 3, Fraction(30), fmt[0])
    assert (start, stop) == (1600, 4800)
    assert pcm_range(path, start, stop, fmt) == (chunks[1] * 1600 + chunks[2] * 1600)
    with pytest.raises(ValueError, match="invalid PCM format or range"):
        pcm_range(path, start, stop + 1, fmt)


def test_fractional_sample_boundary_is_not_rounded():
    with pytest.raises(ValueError, match="whole sample frames"):
        sample_bounds(1, 2, Fraction(30000, 1001), 48000)


def test_lobby_handle_uses_previous_source_frame_and_checks_bounds():
    shot = {"src_in": 0.2, "start_f": 304, "end_f": 399}
    assert lobby_range(shot, 303, Fraction(30), 428) == (5, 101)
    with pytest.raises(ValueError, match="exceeds source"):
        lobby_range(shot, 297, Fraction(30), 428)
    with pytest.raises(ValueError, match="exceeds source"):
        lobby_range(shot, 303, Fraction(30), 100)


def test_caption_rebase_excludes_prior_and_next_pages_and_clips_final_hold():
    word = {
        "text": "Finish.", "start": 10.14, "end": 10.25,
        "src_start": 364.34, "src_end": 364.45, "asr_probability": 0.4, "reliable": False,
    }
    source = {
        "captions": {"font_px": 111, "pages": [
            {"page": 1, "lines": ["Before."], "f0": 300, "f1": 303},
            {"page": 2, "lines": ["Finish."], "f0": 304, "f1": 318},
            {"page": 3, "lines": ["After."], "f0": 319, "f1": 322},
        ]},
        "words": [{}, word, {}],
    }
    words = [{}, word | {"start": 12.14, "end": 12.25}, {}]
    srt = [None, ("00:00:12,133", "00:00:12,633", ["Finish."]), None]
    kwargs = dict(origin=303, stop=318, offset=60, fps=Fraction(30), page_range=(2, 2))
    result = captions(source, words, srt, **kwargs)
    assert result["pages"] == [{
        "page": 2, "lines": ["Finish."], "body_frames": [304, 319],
        "frames": [1, 15], "word_indices": [0],
    }]
    assert result["words"] == [word | {
        "start": 0.04, "end": 0.15, "body_start_s": 10.14, "body_end_s": 10.25,
    }]
    assert result["style"] == {"font_px": 111}
    words[1]["reliable"] = True
    with pytest.raises(ValueError, match="disagrees with the final source records"):
        captions(source, words, srt, **kwargs)


def test_prepared_cue_document_keeps_exact_cut_and_no_crossfade():
    plan = {
        "fps": "30", "size": [1080, 1920], "duration_s": 13.5, "shots": [
            {"scene": "lobby", "frames": [0, 96]},
            {"scene": "agreement", "frames": [96, 405]},
        ],
    }
    cues = parse_cues(cue_document(plan))
    timeline = cues.to_timeline(13.5)
    assert cues.crossfade == 0
    assert timeline.n_frames == 405
    assert timeline.at(95).payload.scene == "lobby"
    assert timeline.at(96).payload.scene == "agreement"
    assert timeline.at(404).payload.scene == "agreement"
