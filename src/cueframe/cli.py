"""Command line interface: ``python -m cueframe <command>``.

Exit codes: 0 success; 1 a check failed (sync not ok, verify failed, oracle failed);
2 invalid input or usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cueframe import __version__
from cueframe.audio import AudioError
from cueframe.cues import CueError
from cueframe.media import MediaError
from cueframe.plan import PlanError
from cueframe.timeline import TimelineError

USER_ERRORS = (CueError, AudioError, PlanError, TimelineError, MediaError, ValueError,
               FileNotFoundError, OSError, KeyError, TypeError)


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"size must be WxH, got {text!r}") from exc


def _json_out(obj, path: Path | None) -> None:
    text = json.dumps(obj, indent=2) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


# ----------------------------------------------------------------------------- animate


def cmd_scenes(args: argparse.Namespace) -> int:
    from cueframe.animate import build_scene_set
    from cueframe.scenes import SCENE_DESCRIPTIONS

    scenes = build_scene_set(args.scenes)
    for name in sorted(scenes):
        desc = SCENE_DESCRIPTIONS.get(name, "(custom scene)" if args.scenes else "")
        print(f"{name:10s} {desc}")
    return 0


def cmd_animate(args: argparse.Namespace) -> int:
    from cueframe.animate import animate, build_scene_set
    from cueframe.cues import load_cues

    sheet = load_cues(args.cues)
    scene_set = build_scene_set(args.scenes)
    result = animate(args.audio, sheet, args.out, scene_set=scene_set, seed=args.seed)
    summary = result.summary()
    if args.sheet:
        from cueframe.audio import read_wav
        from cueframe.report import animation_sheet

        timeline = sheet.to_timeline(read_wav(args.audio).duration)
        rows = []
        for i, seg in enumerate(timeline.segments):
            name = seg.payload.scene if seg.payload else "idle (gap)"
            rows.append((f"{i}: {name}  frame {seg.f0}", seg.f0))
            mid = seg.f0 + seg.n // 2
            if seg.n > 2:
                rows.append((f"{i}: {name}  frame {mid}", mid))
        animation_sheet(args.out, sheet.fps, rows, args.sheet)
        summary["sheet"] = str(args.sheet)
    _json_out(summary, None)
    return 0


# ----------------------------------------------------------------------------- multicam


def _load_inputs(path: Path) -> tuple[Path, list[tuple[str, Path]]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(doc, dict) or not isinstance(doc.get("reference"), str)
            or not doc["reference"] or not isinstance(doc.get("cameras"), list)
            or not doc["cameras"]):
        raise ValueError("inputs need a reference path and a nonempty cameras list")
    names = set()
    for camera in doc["cameras"]:
        if (not isinstance(camera, dict) or not isinstance(camera.get("name"), str)
                or not camera["name"] or camera["name"] in names
                or not isinstance(camera.get("path"), str) or not camera["path"]):
            raise ValueError("cameras need unique nonempty names and local paths")
        names.add(camera["name"])
    base = path.parent
    ref = base / doc["reference"]
    cams = [(c["name"], base / c["path"]) for c in doc["cameras"]]
    return ref, cams


def cmd_stage(args: argparse.Namespace) -> int:
    from cueframe.synth.generate import StageSpec, generate_stage

    spec = StageSpec.quick(args.seed) if args.quick else StageSpec(seed=args.seed)
    manifest = generate_stage(spec, args.out)
    print(f"stage written to {args.out}: {len(manifest['cameras'])} cameras, "
          f"{len(manifest['flash_events_ref_s'])} flash events, {manifest['duration_s']}s")
    return 0


def _decode_all(ref_path: Path, cams: list[tuple[str, Path]], rate: int):
    from cueframe.media import decode_audio

    ref = decode_audio(ref_path, rate=rate)
    return ref, {name: decode_audio(p, rate=rate) for name, p in cams}


def cmd_sync(args: argparse.Namespace) -> int:
    from cueframe.sync import SyncConfig, sync_camera

    ref_path, cams = _load_inputs(args.inputs)
    cfg = SyncConfig()
    ref, cam_audio = _decode_all(ref_path, cams, cfg.analysis_rate)
    results = [sync_camera(cam_audio[name], ref, cfg.analysis_rate, name=name, cfg=cfg)
               for name, _ in cams]
    doc = {
        "inputs": str(args.inputs),
        "analysis_rate": cfg.analysis_rate,
        "cameras": [
            {**r.to_json(), "path": str(path)} for r, (_, path) in zip(results, cams, strict=True)
        ],
    }
    _json_out(doc, args.out)
    for r in results:
        line = f"{r.name}: {r.status}"
        if r.clock:
            line += (f"  offset {r.clock.offset:+.4f}s  drift {r.clock.drift_ppm:+.1f} ppm  "
                     f"windows {r.n_used}/{len(r.windows)}  rms {r.residual_rms_ms:.2f} ms")
        else:
            line += f"  ({r.reason})"
        print(line, file=sys.stderr)
    return 0 if all(r.status == "ok" for r in results) else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from cueframe.clock import AffineClock
    from cueframe.sync import SyncConfig
    from cueframe.verify import verify_camera

    sync_doc = json.loads(args.sync.read_text(encoding="utf-8"))
    ref_path, cams = _load_inputs(Path(sync_doc["inputs"]))
    cfg = SyncConfig()
    ref, cam_audio = _decode_all(ref_path, cams, cfg.analysis_rate)
    results = []
    incomplete = False
    for entry in sync_doc["cameras"]:
        if entry.get("status") != "ok" or entry.get("clock") is None:
            incomplete = True
            print(f"{entry.get('name', 'camera')}: cannot verify an unsynced camera",
                  file=sys.stderr)
            continue
        clock = AffineClock.from_json(entry["clock"])
        results.append(
            verify_camera(cam_audio[entry["name"]], ref, cfg.analysis_rate, clock,
                          name=entry["name"], cfg=cfg, tolerance_ms=args.tolerance_ms)
        )
    _json_out({"cameras": [r.to_json() for r in results]}, args.out)
    for r in results:
        print(f"{r.name}: {r.status}  confident {r.n_confident}/{len(r.windows)}  "
              f"max |res| {r.max_abs_ms:.2f} ms  rms {r.rms_ms:.2f} ms", file=sys.stderr)
    ok = not incomplete and bool(results) and all(r.status == "pass" for r in results)
    return 0 if ok else 1


def cameras_from_sync(sync_path: Path):
    from cueframe.clock import AffineClock
    from cueframe.media import probe
    from cueframe.plan import Camera

    doc = json.loads(sync_path.read_text(encoding="utf-8"))
    cams = []
    for entry in doc["cameras"]:
        if entry["status"] != "ok":
            raise PlanError(f"camera {entry['name']} has sync status {entry['status']}; "
                            "cannot plan with an unsynced camera")
        p = probe(entry["path"])
        cams.append(Camera(entry["name"], str(Path(entry["path"]).resolve()), p.fps,
                           float(p.nb_frames / p.fps), p.width, p.height,
                           AffineClock.from_json(entry["clock"])))
    return cams


def cmd_plan(args: argparse.Namespace) -> int:
    from cueframe.plan import make_plan

    cams = cameras_from_sync(args.sync)
    plan = make_plan(cams, fps=args.fps, size=args.size, duration=args.duration, seed=args.seed)
    plan.save(args.out)
    print(f"{len(plan.timeline.segments)} shots, {plan.n_frames} frames at {plan.fps} fps, "
          f"timeline starts at reference {plan.timeline_start:.3f}s", file=sys.stderr)
    return 0


def cmd_cut(args: argparse.Namespace) -> int:
    from cueframe.cut import cut
    from cueframe.plan import Plan

    plan = Plan.load(args.plan)
    result = cut(plan, args.audio, args.out)
    print(f"{result.out_path}: {result.n_frames} frames, {result.duration:.3f}s", file=sys.stderr)
    return 0


def cmd_oracle(args: argparse.Namespace) -> int:
    from cueframe.oracle import flash_oracle

    report = flash_oracle(args.manifest, args.plan, args.video)
    _json_out(report, args.out)
    print(f"oracle: {report['status']} ({report.get('n_ok', 0)}/{report.get('n_events', 0)} "
          "flash events landed on the predicted frame)", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


# ----------------------------------------------------------------------------- demos


def cmd_demo(args: argparse.Namespace) -> int:
    from cueframe.demo import demo_animation, demo_multicam

    if args.which == "animation":
        summary = demo_animation(args.out, quick=args.quick, seed=args.seed)
    else:
        summary = demo_multicam(args.out, quick=args.quick, seed=args.seed)
    _json_out(summary, None)
    return 0 if summary.get("ok", False) else 1


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cueframe", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=f"cueframe {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scenes", help="list available scene names")
    s.add_argument("--scenes", type=Path, help="custom scene module (.py) to include")
    s.set_defaults(func=cmd_scenes)

    s = sub.add_parser("animate", help="render cues over a WAV file to MP4")
    s.add_argument("audio", type=Path, help="PCM WAV input (left unchanged)")
    s.add_argument("cues", type=Path, help="cue JSON file")
    s.add_argument("out", type=Path, help="output .mp4")
    s.add_argument("--scenes", type=Path, help="trusted Python file defining SCENES")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sheet", type=Path, help="also write a contact sheet PNG")
    s.set_defaults(func=cmd_animate)

    s = sub.add_parser("stage", help="generate the synthetic multicamera stage")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--quick", action="store_true", help="20 s, 320x180 (tests)")
    s.set_defaults(func=cmd_stage)

    s = sub.add_parser("sync", help="estimate camera clocks from audio")
    s.add_argument("inputs", type=Path, help="inputs.json (reference + camera files)")
    s.add_argument("--out", type=Path, help="write sync.json here (default: stdout)")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("verify", help="re-measure residuals on held-out windows")
    s.add_argument("sync", type=Path, help="sync.json")
    s.add_argument("--tolerance-ms", type=float, default=2.0)
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("plan", help="build a frame-exact shot plan from sync.json")
    s.add_argument("sync", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--fps", default="25")
    s.add_argument("--size", type=_size, default=(854, 480))
    s.add_argument("--duration", type=float, help="seconds (default: common coverage)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("cut", help="render a plan with ffmpeg")
    s.add_argument("plan", type=Path)
    s.add_argument("--audio", type=Path, help="reference audio to mux (trimmed to the timeline)")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_cut)

    s = sub.add_parser("oracle", help="check the cut against the stage manifest (truth)")
    s.add_argument("manifest", type=Path)
    s.add_argument("plan", type=Path)
    s.add_argument("video", type=Path)
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_oracle)

    s = sub.add_parser("demo", help="run an end-to-end example")
    s.add_argument("which", choices=["animation", "multicam"])
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--quick", action="store_true")
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
