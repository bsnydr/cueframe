"""End-to-end examples: ``python -m cueframe demo animation|multicam --out DIR``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cueframe.audio import Audio, read_wav, write_wav
from cueframe.cues import CueSheet, dump_cues, parse_cues

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

DEMO_CUES: list[tuple[float, float, str, str, str]] = [
    # start, end, scene, mood (audio), annotation text
    (0.0, 4.0, "title", "title", "cueframe: audio cues become a visual timeline"),
    (4.0, 10.0, "pulse", "pulse", "Level and onsets are measured per frame; rings follow them"),
    (10.0, 16.0, "orbit", "orbit", "Orbit speed follows the audio level"),
    # 16.0-16.8 is a deliberate gap: no cue, silent audio, idle scene
    (16.8, 23.0, "bars", "bars", "Forty bars, one timeline, no hidden state between frames"),
    (23.0, 30.0, "comet", "comet", "A custom scene loaded from examples/custom_scenes.py"),
    (30.0, 36.0, "outro", "outro", "All text here is scene annotation, not transcribed speech"),
]


def demo_cue_sheet(*, quick: bool) -> tuple[CueSheet, list[tuple[float, float, str]]]:
    scale = 0.25 if quick else 1.0
    cues = []
    sections = []
    for start, end, scene, mood, text in DEMO_CUES:
        cues.append({"start": round(start * scale, 3), "end": round(end * scale, 3),
                     "scene": scene, "text": text})
        sections.append((round(start * scale, 3), round(end * scale, 3), mood))
    doc = {
        "fps": 25,
        "size": [320, 180] if quick else [960, 540],
        "crossfade": 0.4 * scale,
        "note": "Generated demo. Cue text is invented scene annotation, not transcribed speech; "
        "the audio is synthesised (no voices, no recordings).",
        "cues": cues,
    }
    return parse_cues(doc, source="demo"), sections


def demo_animation(out_dir: str | Path, *, quick: bool = False, seed: int = 1) -> dict[str, Any]:
    from cueframe.animate import animate, build_scene_set
    from cueframe.report import animation_sheet
    from cueframe.synth.piece import render_piece

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    sheet, sections = demo_cue_sheet(quick=quick)
    rate = 48000
    samples = render_piece(sections, sample_rate=rate, seed=seed)
    wav_path = out / "piece.wav"
    write_wav(wav_path, Audio(samples, rate))
    cues_path = out / "cues.json"
    dump_cues(sheet, cues_path)

    custom = EXAMPLES_DIR / "custom_scenes.py"
    if not custom.is_file():
        # A normal wheel install lives in site-packages; examples stay in the checkout.
        custom = Path.cwd() / "examples" / "custom_scenes.py"
    if not custom.is_file():
        raise FileNotFoundError(
            f"{custom} not found; the demo needs the examples/ directory of a source checkout"
        )
    scene_set = build_scene_set(custom)
    video = out / "animation.mp4"
    result = animate(wav_path, sheet, video, scene_set=scene_set, seed=seed)

    audio = read_wav(wav_path)
    timeline = sheet.to_timeline(audio.duration)
    rows = []
    boundaries = []
    for i, seg in enumerate(timeline.segments):
        name = seg.payload.scene if seg.payload else "idle (gap)"
        boundaries.append({"segment": i, "scene": name, "f0": seg.f0, "f1": seg.f1})
        rows.append((f"{name}  frame {seg.f0}", seg.f0))
        if seg.n > 2:
            rows.append((f"{name}  frame {seg.f0 + seg.n // 2}", seg.f0 + seg.n // 2))
    sheet_png = out / "animation_sheet.png"
    animation_sheet(video, sheet.fps, rows, sheet_png)

    summary = {
        "ok": True,
        "kind": "animation",
        "out_dir": str(out),
        "audio": str(wav_path),
        "cues": str(cues_path),
        "video": str(video),
        "sheet": str(sheet_png),
        "custom_scenes": str(custom),
        "render": result.summary(),
        "segments": boundaries,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def demo_multicam(out_dir: str | Path, *, quick: bool = False, seed: int = 1) -> dict[str, Any]:
    from cueframe.clock import AffineClock
    from cueframe.cut import cut
    from cueframe.media import decode_audio, probe
    from cueframe.oracle import flash_oracle
    from cueframe.plan import Camera, make_plan
    from cueframe.report import lag_chart, multicam_sheet
    from cueframe.sync import SyncConfig, sync_camera
    from cueframe.synth.generate import StageSpec, generate_stage
    from cueframe.verify import verify_camera

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    spec = StageSpec.quick(seed) if quick else StageSpec(seed=seed)
    manifest = generate_stage(spec, out)
    t_stage = time.perf_counter()

    # --- sync + verify (audio only; never reads manifest.json) -------------------------
    inputs = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    cfg = SyncConfig()
    ref = decode_audio(out / inputs["reference"], rate=cfg.analysis_rate)
    cams_audio = {c["name"]: decode_audio(out / c["path"], rate=cfg.analysis_rate)
                  for c in inputs["cameras"]}
    syncs = [sync_camera(cams_audio[c["name"]], ref, cfg.analysis_rate, name=c["name"], cfg=cfg)
             for c in inputs["cameras"]]
    sync_doc = {
        "inputs": str(out / "inputs.json"),
        "analysis_rate": cfg.analysis_rate,
        "cameras": [{**s.to_json(), "path": str(out / c["path"])}
                    for s, c in zip(syncs, inputs["cameras"], strict=True)],
    }
    (out / "sync.json").write_text(json.dumps(sync_doc, indent=2) + "\n", encoding="utf-8")
    verifies = [
        verify_camera(cams_audio[s.name], ref, cfg.analysis_rate, s.clock, name=s.name, cfg=cfg)
        for s in syncs if s.clock is not None
    ]
    (out / "verify.json").write_text(
        json.dumps({"cameras": [v.to_json() for v in verifies]}, indent=2) + "\n",
        encoding="utf-8",
    )
    t_sync = time.perf_counter()

    # --- plan + cut ------------------------------------------------------------------------
    cameras = []
    for s, c in zip(syncs, inputs["cameras"], strict=True):
        if s.clock is None:
            continue
        p = probe(out / c["path"])
        cameras.append(Camera(s.name, str((out / c["path"]).resolve()), p.fps,
                              float(p.nb_frames / p.fps), p.width, p.height, s.clock))
    plan = make_plan(cameras, fps=25, size=spec.size, seed=seed)
    plan.save(out / "plan.json")
    cut(plan, out / inputs["reference"], out / "edit.mp4")
    t_cut = time.perf_counter()

    # --- proof: oracle, sheet, chart (the only steps that read the manifest) ------------------
    oracle = flash_oracle(out / "manifest.json", out / "plan.json", out / "edit.mp4")
    (out / "oracle.json").write_text(json.dumps(oracle, indent=2) + "\n", encoding="utf-8")
    truth = {c["name"]: AffineClock.from_json(c["true_clock"]) for c in manifest["cameras"]}
    lag_chart(syncs, verifies, out / "lag_chart.png", truth=truth)
    events = manifest["flash_events_ref_s"]
    picks = [events[0], events[len(events) // 2], events[-1]]
    sheet_cams = [
        (s.name, out / c["path"], probe(out / c["path"]).fps, s.clock,
         s.clock.clap_sync(min(w.t_cam for w in s.windows if w.used)))
        for s, c in zip(syncs, inputs["cameras"], strict=True) if s.clock is not None
    ]
    sheet_lums = multicam_sheet(sheet_cams, picks, manifest["flash_length_s"],
                                out / "multicam_sheet.png")

    table = []
    for s in syncs:
        t = truth[s.name]
        row = {"camera": s.name, "status": s.status, "true_offset_s": t.offset,
               "true_drift_ppm": round(t.drift_ppm, 2)}
        if s.clock is not None:
            row.update({
                "est_offset_s": round(s.clock.offset, 6),
                "est_drift_ppm": round(s.clock.drift_ppm, 2),
                "offset_error_ms": round((s.clock.offset - t.offset) * 1000, 3),
                "drift_error_ppm": round(s.clock.drift_ppm - t.drift_ppm, 3),
                "fit_windows": s.n_used,
                "fit_rms_ms": round(s.residual_rms_ms, 3),
            })
        v = next((v for v in verifies if v.name == s.name), None)
        if v is not None:
            row.update({"heldout_status": v.status, "heldout_max_abs_ms": round(v.max_abs_ms, 3),
                        "heldout_rms_ms": round(v.rms_ms, 3), "heldout_windows": v.n_confident})
        table.append(row)

    ok = (all(s.status == "ok" for s in syncs) and all(v.status == "pass" for v in verifies)
          and oracle["status"] == "pass")
    summary = {
        "ok": ok,
        "kind": "multicam",
        "out_dir": str(out),
        "video": str(out / "edit.mp4"),
        "lag_chart": str(out / "lag_chart.png"),
        "sheet": str(out / "multicam_sheet.png"),
        "plan": {"n_shots": len(plan.timeline.segments), "n_frames": plan.n_frames,
                 "fps": str(plan.fps), "timeline_start_ref_s": plan.timeline_start,
                 "max_abs_src_err_ms": max(abs(seg.payload.src_err_ms)
                                           for seg in plan.timeline.segments)},
        "truth_vs_estimate": table,
        "oracle": {k: oracle[k] for k in ("status", "n_events", "n_ok")},
        "sheet_luminance": sheet_lums,
        "seconds": {"stage": round(t_stage - t_start, 1), "sync_verify": round(t_sync - t_stage, 1),
                    "plan_cut": round(t_cut - t_sync, 1),
                    "proof": round(time.perf_counter() - t_cut, 1)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
