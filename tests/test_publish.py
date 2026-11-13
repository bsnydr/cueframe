import sys

import pytest

from scripts import check_publish


@pytest.mark.parametrize(
    ("rel", "size", "diagnostic", "checked_as_mp4"),
    [
        ("docs/hk2_demo.mp4", 8_439_656, None, True),
        ("docs/hk2_demo.mp4", 10_000_001, "exceeds media limit 10000000", True),
        ("docs/alternate/hk2_demo.mp4", 8_439_656, "exceeds media limit 1500000", True),
        ("docs/another.mp4", 8_439_656, "exceeds media limit 1500000", True),
        ("docs/camera.MOV", 100, "not in the publish allowlist", False),
        ("work/hk2/inputs/voice.wav", 100, "artifact inside the clean publication tree", False),
    ],
)
def test_release_guard_limits_large_media_to_the_exact_demo_path(
    tmp_path, monkeypatch, capsys, rel, size, diagnostic, checked_as_mp4
):
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        stream.truncate(size)
    checked = []
    monkeypatch.setattr(check_publish, "ROOT", tmp_path)
    monkeypatch.setattr(check_publish, "check_mp4", lambda path, _: checked.append(path))
    monkeypatch.setattr(sys, "argv", ["check_publish.py", "--strict-clean"])

    assert check_publish.main() == (1 if diagnostic else 0)
    assert checked == ([path] if checked_as_mp4 else [])
    if diagnostic:
        assert diagnostic in capsys.readouterr().out
