#!/usr/bin/env python3
"""One-shot local gate: lint, tests, quick demos, publish check. Same steps as CI.

    python scripts/check.py            # everything
    python scripts/check.py --fast     # skip the slow end-to-end tests and demos
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(name: str, argv: list[str]) -> bool:
    print(f"\n== {name}: {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=ROOT, check=False)  # noqa: S603 - fixed argv, no shell
    print(f"== {name}: {'ok' if proc.returncode == 0 else f'FAILED ({proc.returncode})'}")
    return proc.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    ok = step("ruff", [py, "-m", "ruff", "check", "."])
    pytest_args = [py, "-m", "pytest", "-q"]
    if args.fast:
        pytest_args += ["-m", "not slow"]
    ok &= step("pytest", pytest_args)
    if not args.fast:
        with tempfile.TemporaryDirectory() as tmp:
            ok &= step("demos (quick)", [py, "scripts/demo.py", "--out", tmp, "--quick"])
    ok &= step("publish check", [py, "scripts/check_publish.py"])
    print("\nALL OK" if ok else "\nFAILURES above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
