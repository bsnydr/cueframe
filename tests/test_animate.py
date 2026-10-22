"""Audio → animation integration: hand-computed boundary frames, custom scene, unchanged input."""

from fractions import Fraction

import numpy as np
import pytest

from cueframe.animate import animate, build_scene_set, iter_frames
from cueframe.audio import Audio, file_fingerprint, frame_features, read_wav, write_wav
from cueframe.cues import CueError, parse_cues
from cueframe.media import extract_frame, probe

pytestmark = pytest.mark.ffmpeg

MARKER = np.array([220, 40, 120])


def make_wav(path, seconds=2.0, rate=8000):
    n = int(seconds * rate)
    x = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / rate
    x[n // 4 :] = 0.4 * np.sin(2 * np.pi * 330 * t[n // 4 :])
    write_wav(path, Audio(x, rate))
    return path


def sheet(crossfade=0.0):
    return parse_cues({
        "fps": 25,
        "size": [160, 90],
        "crossfade": crossfade,
        "cues": [
            {"start": 0.0, "end": 0.6, "scene": "title", "text": "T"},
            {"start": 0.6, "end": 1.2, "scene": "marker"},
            {"start": 1.5, "end": 2.0, "scene": "pulse", "text": "P"},
        ],
    })


def is_marker(frame, tol=14):
    med = np.median(frame.reshape(-1, 3), axis=0)
    return np.all(np.abs(med - MARKER) <= tol)


def test_custom_scene_boundaries_match_hand_computed_frames(tmp_path, repo_root):
    wav = make_wav(tmp_path / "in.wav")
    before = file_fingerprint(wav)
    scenes = build_scene_set(repo_root / "examples" / "custom_scenes.py")
    out = tmp_path / "out.mp4"
    result = animate(wav, sheet(), out, scene_set=scenes)
    # 2.0 s at 25 fps = 50 frames; marker occupies frames [15, 30); gap is [30, 38)
    assert result.n_frames == 50
    p = probe(out)
    assert p.nb_frames == 50 and p.fps == Fraction(25) and (p.width, p.height) == (160, 90)
    assert abs(p.duration - 2.0) <= 0.04 + 0.05
    assert p.audio_rate == 8000
    fps = Fraction(25)
    assert not is_marker(extract_frame(out, 14, fps))
    assert is_marker(extract_frame(out, 15, fps))
    assert is_marker(extract_frame(out, 29, fps))
    gap = extract_frame(out, 30, fps)
    assert not is_marker(gap)
    assert gap.mean() < 40  # idle scene is dim
    assert not is_marker(extract_frame(out, 37, fps))
    assert not is_marker(extract_frame(out, 38, fps))
    assert file_fingerprint(wav) == before
    assert result.input_fingerprint == before
    assert result.audio_duration == pytest.approx(2.0)


def test_crossfade_blends_only_the_first_frames(tmp_path, repo_root):
    wav = make_wav(tmp_path / "in.wav")
    scenes = build_scene_set(repo_root / "examples" / "custom_scenes.py")
    audio = read_wav(wav)
    s = sheet(crossfade=0.2)  # 5 frames
    tl = s.to_timeline(audio.duration)
    feats = frame_features(audio, s.fps)
    frames = list(iter_frames(tl, feats, scenes, width=160, height=90, crossfade=0.2))
    assert len(frames) == 50
    assert not is_marker(frames[15])  # blended with the title scene
    assert is_marker(frames[20])  # fade over
    assert is_marker(frames[29])


def test_unknown_scene_rejected_before_rendering(tmp_path):
    wav = make_wav(tmp_path / "in.wav")
    out = tmp_path / "out.mp4"
    with pytest.raises(CueError, match="unknown scene 'marker'"):
        animate(wav, sheet(), out)  # built-ins only: no 'marker'
    assert not out.exists()


def test_cue_beyond_audio_rejected(tmp_path):
    wav = make_wav(tmp_path / "in.wav", seconds=1.0)
    with pytest.raises(CueError, match="beyond the audio duration"):
        animate(wav, sheet(), tmp_path / "out.mp4", scene_set=build_scene_set(None) | {
            "marker": lambda ctx, canvas: None})


def test_non_whole_frame_audio_duration_rounds_up(tmp_path):
    rate = 8000
    wav = tmp_path / "odd.wav"
    write_wav(wav, Audio(np.zeros(rate + 1, dtype=np.float32), rate))  # 1.000125 s → 26 frames
    s = parse_cues({"fps": 25, "size": [160, 90],
                    "cues": [{"start": 0.0, "end": 1.0, "scene": "title", "text": "x"}]})
    out = tmp_path / "out.mp4"
    result = animate(wav, s, out)
    assert result.n_frames == 26
    assert probe(out).nb_frames == 26


def test_scene_module_loading_errors(tmp_path):
    from cueframe.animate import SceneLoadError, load_scene_module, scene_set_from_module

    with pytest.raises(SceneLoadError):
        load_scene_module(tmp_path / "missing.py")
    bad = tmp_path / "bad.py"
    bad.write_text("SCENES = {'x': 42}\n")
    with pytest.raises(SceneLoadError, match="not a callable"):
        scene_set_from_module(load_scene_module(bad))
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n")
    with pytest.raises(SceneLoadError, match="SCENES"):
        scene_set_from_module(load_scene_module(empty))
