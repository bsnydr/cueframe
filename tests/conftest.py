import shutil
from pathlib import Path

import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not on PATH")
    for item in items:
        if "ffmpeg" in item.keywords and not HAVE_FFMPEG:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def quick_stage(tmp_path_factory) -> Path:
    """A small generated stage (20 s, 320x180, three cameras) shared by the multicam tests."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    from cueframe.synth.generate import StageSpec, generate_stage

    out = tmp_path_factory.mktemp("stage")
    generate_stage(StageSpec.quick(seed=1), out)
    return out


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
