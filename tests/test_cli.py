"""CLI: help text, exit codes, and the two quick demos end to end."""

import json

import pytest

from cueframe.cli import build_parser, main


@pytest.mark.parametrize("cmd", ["scenes", "animate", "stage", "sync", "verify", "plan", "cut",
                                 "oracle", "demo"])
def test_help_for_every_subcommand(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        main([cmd, "--help"])
    assert exc.value.code == 0
    assert cmd in capsys.readouterr().out


def test_no_command_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_scenes_lists_builtins_and_custom(capsys, repo_root):
    assert main(["scenes"]) == 0
    out = capsys.readouterr().out
    for name in ("idle", "title", "pulse", "orbit", "bars", "outro"):
        assert name in out
    assert main(["scenes", "--scenes", str(repo_root / "examples" / "custom_scenes.py")]) == 0
    assert "comet" in capsys.readouterr().out


def test_bad_cues_exit_2(tmp_path, capsys):
    cues = tmp_path / "c.json"
    cues.write_text(json.dumps({"fps": 25, "cues": [{"start": 1, "end": 0, "scene": "x"}]}))
    assert main(["animate", str(tmp_path / "missing.wav"), str(cues), str(tmp_path / "o.mp4")]) == 2
    assert "cue 0" in capsys.readouterr().err


@pytest.mark.ffmpeg
def test_demo_animation_quick(tmp_path, capsys):
    assert main(["demo", "animation", "--out", str(tmp_path / "anim"), "--quick"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert (tmp_path / "anim" / "animation.mp4").is_file()
    assert (tmp_path / "anim" / "animation_sheet.png").is_file()
    assert (tmp_path / "anim" / "cues.json").is_file()
    assert summary["render"]["n_frames"] == 225  # 9 s at 25 fps
    names = [s["scene"] for s in summary["segments"]]
    assert names == ["title", "pulse", "orbit", "idle (gap)", "bars", "comet", "outro"]


@pytest.mark.ffmpeg
@pytest.mark.slow
def test_demo_multicam_quick(tmp_path, capsys):
    assert main(["demo", "multicam", "--out", str(tmp_path / "mc"), "--quick"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["oracle"]["status"] == "pass"
    for row in summary["truth_vs_estimate"]:
        assert row["status"] == "ok" and row["heldout_status"] == "pass"
    for name in ("edit.mp4", "lag_chart.png", "multicam_sheet.png", "sync.json", "verify.json",
                 "plan.json", "oracle.json", "manifest.json", "inputs.json"):
        assert (tmp_path / "mc" / name).is_file()
    # sync.json is an estimate, not a copy of the manifest
    sync_doc = json.loads((tmp_path / "mc" / "sync.json").read_text())
    manifest = json.loads((tmp_path / "mc" / "manifest.json").read_text())
    for est, tru in zip(sync_doc["cameras"], manifest["cameras"], strict=True):
        assert est["clock"]["offset_s"] != tru["true_clock"]["offset_s"]
        assert "true_clock" not in est


@pytest.mark.ffmpeg
def test_cli_pipeline_commands_on_quick_stage(quick_stage, tmp_path, capsys):
    out = tmp_path
    assert main(["sync", str(quick_stage / "inputs.json"), "--out", str(out / "sync.json")]) == 0
    assert main(["verify", str(out / "sync.json"), "--out", str(out / "verify.json")]) == 0
    assert main(["plan", str(out / "sync.json"), "--out", str(out / "plan.json"),
                 "--size", "320x180", "--duration", "6"]) == 0
    assert main(["cut", str(out / "plan.json"), "--audio", str(quick_stage / "reference.wav"),
                 "--out", str(out / "edit.mp4")]) == 0
    code = main(["oracle", str(quick_stage / "manifest.json"), str(out / "plan.json"),
                 str(out / "edit.mp4")])
    assert code == 0
