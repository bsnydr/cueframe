"""The audio-only path must not drag in the multicamera machinery."""

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")

PROBE = r"""
import sys
import cueframe.animate, cueframe.cues, cueframe.audio, cueframe.scenes
loaded = sorted(m for m in sys.modules if m.startswith("cueframe"))
print("\n".join(loaded))
"""


def test_animation_imports_exclude_sync_plan_cut_and_synth():
    env = {**os.environ, "PYTHONPATH": SRC}  # same source tree pytest itself is running
    proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                          check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    loaded = set(proc.stdout.split())
    forbidden = {"cueframe.sync", "cueframe.verify", "cueframe.plan", "cueframe.cut",
                 "cueframe.gccphat", "cueframe.clock", "cueframe.oracle", "cueframe.synth",
                 "cueframe.synth.stage", "cueframe.synth.generate", "cueframe.report"}
    assert not loaded & forbidden, sorted(loaded & forbidden)
    assert {"cueframe.animate", "cueframe.cues", "cueframe.audio", "cueframe.timeline",
            "cueframe.media"} <= loaded
