#!/usr/bin/env python3
"""Run both demos and (optionally) refresh the small showcase assets under docs/.

    python scripts/demo.py --out out            # full demos into out/animation, out/multicam
    python scripts/demo.py --out out --docs docs  # also write docs/*.png, docs/*_preview.mp4
    python scripts/demo.py --out out --quick    # fast, small variants (what CI runs)

Works from a source checkout without installing: if ``cueframe`` is not importable the
``src/`` directory next to this script is added to ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import cueframe  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "src"))

from cueframe.demo import demo_animation, demo_multicam  # noqa: E402
from cueframe.media import check_output, probe, run, safe_path  # noqa: E402
from cueframe.timeline import frame_count  # noqa: E402

PREVIEW_MAX_BYTES = 1_500_000


def preview(src: Path, dst: Path, *, start: float, duration: float | None, height: int,
            crf: int) -> None:
    """Small, metadata-free excerpt for the README."""
    argv = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", "-ss", f"{start:.3f}",
            "-i", safe_path(src)]
    if duration is not None:
        argv += ["-t", f"{duration:.3f}"]
    p = probe(src)
    argv += ["-vf", f"scale=-2:{height}:flags=lanczos", "-r", str(p.fps),
             "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-map_metadata", "-1",
             safe_path(dst)]
    run(argv, timeout=900)
    q = probe(dst)
    expected_dur = duration if duration is not None else p.duration - start
    check_output(dst, fps=p.fps, n_frames=frame_count(expected_dur, p.fps) if duration else
                 q.nb_frames, width=q.width, height=q.height, audio_rate=q.audio_rate)
    size = dst.stat().st_size
    if size > PREVIEW_MAX_BYTES:
        raise SystemExit(f"{dst} is {size} bytes, over the {PREVIEW_MAX_BYTES} byte limit")


def markdown_table(rows: list[dict]) -> str:
    cols = [("camera", "camera"), ("true_drift_ppm", "true drift (ppm)"),
            ("est_drift_ppm", "estimated drift (ppm)"), ("drift_error_ppm", "drift error (ppm)"),
            ("offset_error_ms", "offset error (ms)"), ("fit_windows", "fit windows"),
            ("heldout_windows", "held-out windows"),
            ("heldout_max_abs_ms", "held-out max absolute residual (ms)")]
    out = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c[0], "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs", type=Path, help="copy showcase assets here")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", choices=["animation", "multicam"])
    args = ap.parse_args()

    summaries = {}
    if args.only in (None, "animation"):
        summaries["animation"] = demo_animation(args.out / "animation", quick=args.quick,
                                                seed=args.seed)
        print(f"animation: {summaries['animation']['video']}", file=sys.stderr)
    if args.only in (None, "multicam"):
        summaries["multicam"] = demo_multicam(args.out / "multicam", quick=args.quick,
                                              seed=args.seed)
        print(f"multicam: {summaries['multicam']['video']} "
              f"(oracle {summaries['multicam']['oracle']['status']})", file=sys.stderr)

    if args.docs:
        docs = args.docs
        docs.mkdir(parents=True, exist_ok=True)
        results = {}
        if "animation" in summaries:
            a = summaries["animation"]
            shutil.copyfile(a["sheet"], docs / "animation_sheet.png")
            preview(Path(a["video"]), docs / "animation_preview.mp4", start=0.0, duration=None,
                    height=270, crf=28)
            results["animation"] = {"render": a["render"], "segments": a["segments"]}
        if "multicam" in summaries:
            m = summaries["multicam"]
            shutil.copyfile(m["lag_chart"], docs / "lag_chart.png")
            shutil.copyfile(m["sheet"], docs / "multicam_sheet.png")
            preview(Path(m["video"]), docs / "multicam_preview.mp4", start=0.0, duration=12.0,
                    height=270, crf=27)
            results["multicam"] = {k: m[k] for k in ("plan", "truth_vs_estimate", "oracle",
                                                     "sheet_luminance", "seconds")}
            print(markdown_table(m["truth_vs_estimate"]))
        (docs / "results.json").write_text(json.dumps(results, indent=2) + "\n",
                                           encoding="utf-8")
    ok = all(s.get("ok") for s in summaries.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
