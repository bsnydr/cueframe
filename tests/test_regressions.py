"""Independent regressions from adversarial review: rates, schemas and sparse evidence."""

import json
from fractions import Fraction

import numpy as np
import pytest

from cueframe.animate import build_scene_set, iter_frames
from cueframe.audio import Audio, AudioError, frame_features, read_wav, write_wav
from cueframe.cli import main
from cueframe.clock import AffineClock
from cueframe.cues import parse_cues
from cueframe.cut import concat_entry, cut, video_filter
from cueframe.gccphat import gcc_phat
from cueframe.media import MediaError, extract_frame, mux_raw_frames
from cueframe.plan import Camera, Plan, PlanError, make_plan
from cueframe.sync import sync_camera
from cueframe.verify import verify_camera


def test_short_camera_matches_late_in_long_reference():
    rate = 16000
    ref = np.random.default_rng(41).normal(0, 0.1, 100 * rate)
    cam = ref[75 * rate:95 * rate].copy()
    lag = gcc_phat(cam[:8 * rate], ref, rate, max_lag=108)
    assert lag.lag == pytest.approx(-75, abs=1 / rate)
    result = sync_camera(cam, ref, rate)
    assert result.status == "ok", result.reason
    assert result.clock.offset == pytest.approx(75, abs=0.001)


def test_verification_rejects_only_early_matching_audio():
    rate = 16000
    rng = np.random.default_rng(5)
    ref = rng.normal(0, 0.1, 40 * rate)
    cam = rng.normal(0, 0.1, ref.size)
    cam[:12 * rate] = ref[:12 * rate]
    result = verify_camera(cam, ref, rate, AffineClock(0))
    assert result.status == "insufficient_coverage"


def test_custom_comet_crossfade_progress_is_bounded(repo_root):
    sheet = parse_cues({"fps": 25, "size": [160, 90], "crossfade": 0.4, "cues": [
        {"start": 0, "end": 1, "scene": "comet"},
        {"start": 1, "end": 2, "scene": "outro"},
    ]})
    audio = Audio(np.zeros(16000, dtype=np.float32), 8000)
    scenes = build_scene_set(repo_root / "examples/custom_scenes.py")
    original = scenes["comet"]

    def checked(ctx, canvas):
        assert 0 <= ctx.cue_progress <= 1
        original(ctx, canvas)

    scenes["comet"] = checked
    frames = list(iter_frames(sheet.to_timeline(2), frame_features(audio, sheet.fps), scenes,
                              width=160, height=90, crossfade=0.4))
    assert len(frames) == 50


@pytest.mark.parametrize("mutate", [
    lambda d: d["size"].__setitem__(0, "32:16,hflip,scale=32"),
    lambda d: d["shots"][0].__setitem__("f0", 1),
    lambda d: d["shots"][0].__setitem__("f1", 1),
    lambda d: d["shots"][0].__setitem__("src_frame", 5000),
    lambda d: d["shots"][0].__setitem__("src_fps", "30"),
    lambda d: d["shots"][0].__setitem__("cam", "missing"),
])
def test_loaded_plan_validates_executable_numbers_and_source_bounds(tmp_path, mutate):
    cam = Camera("A", "camera.mov", Fraction(25), 10, 160, 90, AffineClock(0))
    doc = make_plan([cam], fps=25, size=(160, 90), duration=4).to_json()
    mutate(doc)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(PlanError):
        Plan.load(path)


@pytest.mark.ffmpeg
@pytest.mark.parametrize("rate", [0.8, 1.25])
def test_cut_applies_clock_rate_within_shot(tmp_path, rate):
    # Source pixel value is 10 + source-frame index: a hand-computable time ruler.
    source = tmp_path / "ramp.mp4"
    fps = Fraction(25)
    frames = (np.full((90, 160, 3), 10 + f, dtype=np.uint8) for f in range(200))
    mux_raw_frames(frames, width=160, height=90, fps=fps, n_frames=200,
                   audio_path=None, out_path=source)
    camera = Camera("A", str(source), fps, 8, 160, 90, AffineClock(0, rate))
    plan = make_plan([camera], fps=fps, size=(160, 90), start=0, duration=4, seed=0)
    out = tmp_path / "cut.mp4"
    cut(plan, None, out)
    for output_frame in (0, 25, 50, 75):
        expected = 10 + output_frame / rate
        measured = float(np.median(extract_frame(out, output_frame, fps)))
        assert measured == pytest.approx(expected, abs=3)


@pytest.mark.parametrize("offset,rate", [(float("nan"), 1), (float("inf"), 1), (0, float("nan"))])
def test_clock_rejects_nonfinite_values(offset, rate):
    with pytest.raises(ValueError):
        AffineClock(offset, rate)


def test_saved_plan_preserves_precision_at_quantization_boundary(tmp_path):
    camera = Camera("A", "a.mov", Fraction(25), 10, 160, 90, AffineClock(0))
    plan = make_plan([camera], fps=25, size=(160, 90), start=1.0199999, duration=2)
    path = tmp_path / "plan.json"
    plan.save(path)
    assert Plan.load(path).timeline.segments[0].payload.src_frame == 25


def test_filter_api_rejects_text_in_numeric_fields():
    with pytest.raises(MediaError):
        video_filter("full", (160, 90), ("32:16,hflip,scale=32", 90), "25")


def test_truncated_pcm_is_rejected(tmp_path):
    path = tmp_path / "truncated.wav"
    write_wav(path, Audio(np.zeros(800, dtype=np.float32), 8000))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(AudioError, match="truncated"):
        read_wav(path)


@pytest.mark.ffmpeg
def test_cli_verify_fails_if_any_camera_is_unsynced(tmp_path, capsys):
    wav = tmp_path / "noise.wav"
    write_wav(wav, Audio(np.random.default_rng(9).normal(0, 0.1, 20 * 16000), 16000))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"reference": wav.name, "cameras": [
        {"name": "good", "path": wav.name}, {"name": "bad", "path": wav.name}]}))
    sync = tmp_path / "sync.json"
    sync.write_text(json.dumps({"inputs": str(inputs), "cameras": [
        {"name": "good", "status": "ok", "clock": AffineClock(0).to_json()},
        {"name": "bad", "status": "silent", "clock": None}]}))
    assert main(["verify", str(sync)]) == 1
    out = capsys.readouterr()
    assert json.loads(out.out)["cameras"][0]["status"] == "pass"
    assert "bad: cannot verify" in out.err


def test_concat_rejects_control_characters_in_paths(tmp_path):
    with pytest.raises(MediaError, match="control"):
        concat_entry(tmp_path / "name\nfile.mp4")


def test_cli_sync_empty_camera_list_is_invalid(tmp_path, capsys):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"reference": "ref.wav", "cameras": []}))
    assert main(["sync", str(path)]) == 2
    assert "nonempty cameras" in capsys.readouterr().err
