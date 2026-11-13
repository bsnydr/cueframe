#!/usr/bin/env python3
"""Prepare, render and verify the local HK #2 Cueframe example."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from cueframe.cues import parse_cues
from cueframe.media import MediaError, probe, run, safe_path, tool
from cueframe.timeline import Segment, Timeline, frame_to_time, parse_fps, time_to_frame

HERE = Path(__file__).resolve().parent
ARTIFACTS = ("voice.wav", "lobby-clean.mkv", "cues.json", "plan.json")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact(value) -> Fraction:
    # Source JSON timings are decimal records, not binary float arithmetic.
    return Fraction(str(value))


def clipped_frames(start: int, end: int, origin: int, stop: int) -> tuple[int, int] | None:
    """Intersect half-open frame spans and express the result on the excerpt clock."""
    if not (0 <= start < end and 0 <= origin < stop):
        raise ValueError("invalid frame interval")
    a, b = max(start, origin), min(end, stop)
    return (a - origin, b - origin) if a < b else None


def pcm_range(path: Path, start: int, end: int, fmt: tuple[int, int, int]) -> bytes:
    """Read an exact half-open range of sample frames without decoding or conversion."""
    rate, channels, width = fmt
    with wave.open(str(path), "rb") as reader:
        actual = (reader.getframerate(), reader.getnchannels(), reader.getsampwidth())
        if actual != fmt or not 0 <= start < end <= reader.getnframes():
            raise ValueError(f"{path.name}: invalid PCM format or range [{start}, {end})")
        reader.setpos(start)
        data = reader.readframes(end - start)
    if len(data) != (end - start) * channels * width:
        raise ValueError(f"{path.name}: truncated PCM")
    return data


def sample_bounds(start: int, end: int, fps: Fraction, rate: int) -> tuple[int, int]:
    a, b = start * rate / fps, end * rate / fps
    if not 0 <= start < end or a.denominator != 1 or b.denominator != 1:
        raise ValueError("frame interval does not map to whole sample frames")
    return int(a), int(b)


def srt_cues(path: Path) -> list[tuple[str, str, list[str]]]:
    cues = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        index, bounds, *lines = block.splitlines()
        if int(index) != len(cues) + 1:
            raise ValueError("invalid SRT numbering")
        start, end = bounds.split(" --> ")
        cues.append((start, end, lines))
    return cues


def srt_time(frame: int, fps: Fraction) -> str:
    ms = round(frame_to_time(frame, fps) * 1000)
    return f"00:00:{ms // 1000:02},{ms % 1000:03}"


def captions(series, final_words, final_srt, *, origin, stop, offset, fps, page_range):
    # Reattach page-to-word indexes from the source's exact line grouping.
    pages, words = [], []
    cursor = 0
    for p in series["captions"]["pages"]:
        expected_text = " ".join(p["lines"]).split()
        indices = range(cursor, cursor + len(expected_text))
        cursor += len(expected_text)
        if not page_range[0] <= p["page"] <= page_range[1]:
            continue
        bounds = clipped_frames(p["f0"], p["f1"] + 1, origin, stop)
        if bounds is None:
            raise ValueError(f"selected caption page {p['page']} is outside the passage")
        final_bounds = (
            srt_time(p["f0"] + offset, fps), srt_time(p["f1"] + 1 + offset, fps), p["lines"]
        )
        if final_srt[p["page"] - 1] != final_bounds:
            raise ValueError(f"caption page {p['page']} disagrees with the final SRT")
        excerpt_indices = []
        for i, txt in zip(indices, expected_text, strict=True):
            w, fw = series["words"][i], final_words[i]
            if w["text"] != txt or any(
                w[k] != fw[k]
                for k in ("text", "src_start", "src_end", "reliable", "asr_probability")
            ) or any(exact(w[k]) + offset / fps != exact(fw[k]) for k in ("start", "end")):
                raise ValueError(f"word {i} disagrees with the final source records")
            if not origin / fps <= exact(w["start"]) < exact(w["end"]) <= stop / fps:
                raise ValueError(f"selected word {txt!r} is outside the passage")
            excerpt_indices.append(len(words))
            words.append(w | {
                "body_start_s": w["start"], "body_end_s": w["end"],
                "start": float(exact(w["start"]) - origin / fps),
                "end": float(exact(w["end"]) - origin / fps),
            })
        pages.append({
            "page": p["page"], "lines": p["lines"], "frames": list(bounds),
            "body_frames": [p["f0"], p["f1"] + 1], "word_indices": excerpt_indices,
        })
    if len(pages) != page_range[1] - page_range[0] + 1:
        raise ValueError("missing caption pages")
    # Use Timeline to confirm exact coverage after the optional blank lead-in.
    segments = ([Segment(0, pages[0]["frames"][0], None)] if pages[0]["frames"][0] else [])
    segments += [Segment(*p["frames"], p["page"]) for p in pages]
    Timeline(fps, stop - origin, tuple(segments))
    return {"pages": pages, "words": words, "style": {
        k: v for k, v in series["captions"].items() if k != "pages"
    }}


def video_stream(path: Path) -> dict:
    probe = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", (
            "stream=width,height,r_frame_rate,avg_frame_rate,start_time,nb_frames,"
            "color_space,color_range,color_transfer,color_primaries:stream_side_data=rotation"
        ), "-of", "json", safe_path(path),
    ])
    return json.loads(probe.stdout)["streams"][0]


def checked_input_paths(manifest, inputs: Path, reference: Path) -> dict[str, Path]:
    paths = {}
    for name, digest in manifest["files"].items():
        directory, basename = name.split("/")
        root = {"inputs": inputs, "reference": reference}[directory]
        path = root / basename
        if file_hash(path) != digest:
            raise ValueError(f"{name}: SHA-256 differs from the checked input manifest")
        paths[name] = path
    return paths


def lobby_range(shot, origin: int, fps: Fraction, source_frames: int) -> tuple[int, int]:
    accepted_first = time_to_frame(exact(shot["src_in"]), fps)
    start = accepted_first + origin - shot["start_f"]
    end = accepted_first + shot["end_f"] - shot["start_f"]
    if not 0 <= start < end <= source_frames:
        raise ValueError(f"lobby frame range [{start}, {end}) exceeds source ({source_frames})")
    return start, end


def build_plan(manifest, paths):
    series = read_json(paths["reference/series-edit.json"])
    animation = read_json(paths["reference/animation-edit.json"])["timeline_spec"]
    final_words = read_json(paths["reference/final-word-timings.json"])
    final_srt = srt_cues(paths["reference/final-captions.srt"])
    final_verification = read_json(paths["reference/final-verification.json"])
    if final_verification["sha256"] != manifest["files"]["inputs/hk2-final-reference.mp4"]:
        raise ValueError("wrong comparison master")
    fps = parse_fps(manifest["fps"])
    width, height = manifest["size"]
    origin, stop = manifest["body_frames"]
    offset = manifest["final_offset_frames"]
    count = stop - origin
    shot = next(s for s in series["picture"]["shots"] if s["id"] == "lobby_a")
    motion_start, motion_stop = animation["segments_frames"][0]
    if motion_start != shot["end_f"] or not origin < motion_start < stop <= motion_stop:
        raise ValueError("passage does not span the expected lobby/illustration cut")
    if series["output"]["fps"] != fps or [series["output"][k] for k in ("w", "h")] != [
        width, height
    ]:
        raise ValueError("timing and picture metadata disagree")

    info = video_stream(paths[manifest["lobby"]["source"]])
    rotation = next(sd["rotation"] for sd in info["side_data_list"] if "rotation" in sd)
    if not (
        [info["width"], info["height"]] == manifest["lobby"]["source_size"]
        and rotation == manifest["lobby"]["recorded_rotation"]
        and parse_fps(info["avg_frame_rate"]) == parse_fps(info["r_frame_rate"]) == fps
        and exact(info["start_time"]) == 0
        and [info[k] for k in ("color_space", "color_range", "color_transfer", "color_primaries")]
        == ["bt2020nc", "tv", "arib-std-b67", "bt2020"]
    ):
        raise ValueError("lobby source frame grid, rotation, or HLG tags do not match")
    lobby_start, lobby_end = lobby_range(shot, origin, fps, int(info["nb_frames"]))
    cw, ch, x, y = shot["crop_px_wh_xy"]
    if not (0 <= x < x + cw <= info["height"] and 0 <= y < y + ch <= info["width"]):
        raise ValueError("crop exceeds the rotated lobby source")

    motion_f0 = motion_start - origin
    shots = Timeline(fps, count, (
        Segment(0, motion_f0, {
            "scene": "lobby", "source": manifest["lobby"]["source"],
            "source_frames": [lobby_start, lobby_end],
            "crop_px_wh_xy": [cw, ch, x, y], "grade": manifest["lobby"]["grade"],
            "accepted_lobby_first_frame": shot["start_f"] - origin,
        }),
        Segment(motion_f0, count, {"scene": "agreement"}),
    ))
    prepared_captions = captions(
        series, final_words, final_srt, origin=origin, stop=stop, offset=offset, fps=fps,
        page_range=manifest["caption_pages"],
    )
    main = series["speech"]["main"]
    rate = manifest["audio"]["sample_rate"]
    start_sample, end_sample = sample_bounds(origin, stop, fps, rate)
    if main["placed_at"] != 0 or stop / fps > exact(main["out"]) - exact(main["in"]):
        raise ValueError("selection extends beyond the continuous voice passage")

    def rebase(body_s) -> float:
        return float(exact(body_s) - origin / fps)

    original_motion_range = (motion_start / fps, motion_stop / fps)
    gate_words = []
    for w in animation["gate_words"]:
        if exact(w["in_s"]) < stop / fps:
            gate_words.append({
                "word": w["word"], "body_in_s": w["in_s"], "in_s": rebase(w["in_s"]),
                "body_delete_s": w["delete_s"], "delete_s": None,
            })
    return {
        "version": 1,
        "fps": str(fps),
        "size": [width, height],
        "frames": count,
        "duration_s": shots.duration,
        "origins": {
            "body_frames": [origin, stop], "comparison_frames": [origin + offset, stop + offset],
            "comparison_source": "inputs/hk2-final-reference.mp4", "matched_picture_from_frame": 1,
        },
        "audio": {
            **manifest["audio"], "prepared": "voice.wav",
            "processed_source": "inputs/approved-voice.wav",
            "processed_source_samples": [start_sample, end_sample],
            "camera_context_source": "inputs/IMG_3133.MOV",
            "camera_context_seconds": [
                float(exact(main["in"]) + origin / fps),
                float(exact(main["in"]) + stop / fps),
            ],
            "continuous_source_end_s": float(exact(main["out"]) - exact(main["in"])),
            "continuous_source_pcm_sha256": series["audio"]["first_32.51s_pcm_sha256"],
        },
        "shots": [{"frames": [s.f0, s.f1], **s.payload} for s in shots.segments],
        "captions": prepared_captions,
        "motion": {
            "body_frames": [motion_start, motion_stop],
            "excerpt_frames": [motion_f0, count],
            "anchors_s": {
                k: {"body": v, "excerpt": rebase(v)}
                for k, v in animation["anchors_s"].items()
                if original_motion_range[0] <= exact(v) < stop / fps
            },
            "beats": [
                {**b, "excerpt_s": rebase(b["t"])}
                for b in animation["beats"]
                if original_motion_range[0] <= exact(b["t"]) < stop / fps
            ],
            "gate_words": gate_words,
            "geometry": {
                k: v for k, v in animation["geometry"].items() if k != "not_reviewed_chip"
            },
            "durations_s": {
                k: v for k, v in animation["durations_s"].items()
                if k not in {"reframe", "fall_away"}
            },
            "colour_roles": animation["colour_roles"],
            "copy_on_screen": animation["copy_on_screen"][:7],
        },
        "source_sha256": manifest["files"],
    }


def cue_document(plan) -> dict:
    fps = parse_fps(plan["fps"])
    cues = [{
        "start": frame_to_time(s["frames"][0], fps),
        "end": frame_to_time(s["frames"][1], fps),
        "scene": s["scene"],
    } for s in plan["shots"]]
    doc = {"fps": plan["fps"], "size": plan["size"], "crossfade": 0.0, "cues": cues}
    actual = parse_cues(doc).to_timeline(plan["duration_s"])
    if [(s.f0, s.f1) for s in actual.segments] != [tuple(s["frames"]) for s in plan["shots"]]:
        raise ValueError("cue timestamps do not reproduce the exact shot boundaries")
    return doc


def verified_pcm(plan, paths) -> tuple[tuple[int, int, int], bytes]:
    audio = plan["audio"]
    fmt = (audio["sample_rate"], audio["channels"], audio["sample_width_bytes"])
    source = paths[audio["processed_source"]]
    continuity_end = int(exact(audio["continuous_source_end_s"]) * fmt[0])
    pcm = pcm_range(source, 0, continuity_end, fmt)
    if hashlib.sha256(pcm).hexdigest() != audio["continuous_source_pcm_sha256"]:
        raise ValueError("processed voice continuity hash disagrees with original series record")
    start, end = audio["processed_source_samples"]
    return fmt, pcm_range(source, start, end, fmt)


def write_voice(path: Path, fmt: tuple[int, int, int], pcm: bytes) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setframerate(fmt[0])
        writer.setnchannels(fmt[1])
        writer.setsampwidth(fmt[2])
        writer.writeframes(pcm)


def prepare_lobby(path: Path, plan, paths) -> None:
    shot = plan["shots"][0]
    grade = shot["grade"]
    source = paths[shot["source"]]
    lut = paths[grade["lut"]]
    a, b = shot["source_frames"]
    cw, ch, x, y = shot["crop_px_wh_xy"]
    w, h = plan["size"]
    # Run in the LUT directory so FFmpeg filter syntax only sees the fixed local filename,
    # even when the checkout's path contains shell/filter punctuation.
    filters = (
        f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS,"
        f"scale=in_color_matrix={grade['input_matrix']}:in_range={grade['input_range']}:"
        f"out_range={grade['rgb_range']}:flags=bicubic+accurate_rnd+full_chroma_int,"
        f"format=gbrp16le,lut3d=file={lut.name}:interp={grade['interpolation']},format=gbrp,"
        f"crop={cw}:{ch}:{x}:{y},scale={w}:{h}:flags=lanczos+accurate_rnd,setsar=1,"
        f"format=gbrp,format=bgr0,setparams=range=full:colorspace=gbr:"
        f"color_primaries={grade['output_primaries']}:color_trc={grade['output_transfer']}"
    )
    proc = subprocess.run(  # noqa: S603 - resolved executable, argv list, shell=False
        [tool("ffmpeg"), "-v", "error", "-nostdin", "-xerror", "-i", safe_path(source),
         "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters, "-frames:v", str(b - a),
         "-c:v", "ffv1", "-pix_fmt", "bgr0", "-map_metadata", "-1", "-map_chapters", "-1",
         safe_path(path)],
        cwd=lut.parent, capture_output=True, timeout=180, shell=False,
    )
    if proc.returncode:
        raise MediaError(proc.stderr.decode(errors="replace").strip()[-2000:])


def verify_lobby(path: Path, plan) -> None:
    w, h = plan["size"]
    a, b = plan["shots"][0]["frames"]
    info = probe(path)
    if (info.width, info.height, info.nb_frames, info.fps, info.rotation, info.audio_rate) != (
        w, h, b - a, parse_fps(plan["fps"]), 0, None
    ):
        raise ValueError("prepared lobby has unexpected picture or audio streams")
    if any(k.split(":")[-1].lower() not in {"encoder", "duration"} for k in info.tags):
        raise ValueError("prepared lobby has unexpected metadata")
    stream = video_stream(path)
    keys = ("color_space", "color_range", "color_transfer", "color_primaries")
    if [stream.get(k) for k in keys] != ["gbr", "pc", "bt709", "bt709"]:
        raise ValueError(f"prepared lobby must be full-range BT.709 RGB: {stream}")


def prepare_or_verify(args) -> dict:
    manifest = read_json(HERE / "input-manifest.json")
    inputs = args.inputs.resolve()
    reference = args.reference.resolve() if args.reference else inputs.with_name("reference")
    out = args.out.resolve()
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in (inputs, reference)):
        raise ValueError("prepared directory must be separate from the input and reference folders")
    paths = checked_input_paths(manifest, inputs, reference)
    plan = build_plan(manifest, paths)
    cues = cue_document(plan)
    fmt, pcm = verified_pcm(plan, paths)
    plan["audio"]["selected_pcm_sha256"] = hashlib.sha256(pcm).hexdigest()

    if args.command == "prepare":
        out.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=out) as temp:
            staging = Path(temp)
            write_voice(staging / "voice.wav", fmt, pcm)
            prepare_lobby(staging / "lobby-clean.mkv", plan, paths)
            write_json(staging / "cues.json", cues)
            verify_lobby(staging / "lobby-clean.mkv", plan)
            plan["prepared_sha256"] = {name: file_hash(staging / name) for name in ARTIFACTS[:-1]}
            write_json(staging / "plan.json", plan)
            checked_input_paths(manifest, inputs, reference)
            for name in ARTIFACTS:
                (staging / name).replace(out / name)
    else:
        prepared = read_json(out / "plan.json")
        plan["prepared_sha256"] = {name: file_hash(out / name) for name in ARTIFACTS[:-1]}
        if plan != prepared or cues != read_json(out / "cues.json"):
            raise ValueError("prepared plan, cues, or media hash differs; run prepare again")
        voice = out / "voice.wav"
        with wave.open(str(voice), "rb") as reader:
            length = reader.getnframes()
        if length != len(pcm) // (fmt[1] * fmt[2]) or pcm_range(voice, 0, length, fmt) != pcm:
            raise ValueError("prepared voice differs from the exact source PCM interval")
        verify_lobby(out / "lobby-clean.mkv", plan)
        checked_input_paths(manifest, inputs, reference)
    return {
        "status": "prepared" if args.command == "prepare" else "prepared inputs verified",
        "frames": plan["frames"], "duration_s": plan["duration_s"],
        "selected_pcm_sha256": plan["audio"]["selected_pcm_sha256"],
        "next": "listening, motion/caption rendering and review pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-prepared", "render", "verify"))
    parser.add_argument("--inputs", type=Path, required=True,
                        help="source inputs (prepare) or the prepared folder (render/verify)")
    parser.add_argument("--reference", type=Path, help="defaults to the input directory's sibling")
    parser.add_argument("--sources", type=Path,
                        help="render/verify: checked fonts and final reference; defaults to "
                             "the prepared folder's sibling 'inputs'")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command in ("render", "verify"):
            try:
                from . import review
            except ImportError:  # script mode
                import review
            prepared = args.inputs.resolve()
            sources = (args.sources or prepared.with_name("inputs")).resolve()
            out = args.out.resolve()
            if any(out.is_relative_to(p) or p.is_relative_to(out) for p in (prepared, sources)):
                raise ValueError("review folder must be separate from prepared and source folders")
            if args.command == "render":
                print(json.dumps(review.render(prepared, sources, out), indent=2))
                return
            summary, ok = review.verify(prepared, sources, out)
            print(json.dumps(summary, indent=2))
            if not ok:
                parser.exit(1, "hk2: automated verification failed; see the report\n")
            return
        print(json.dumps(prepare_or_verify(args), indent=2))
    except (OSError, ValueError, KeyError, MediaError, wave.Error) as exc:
        parser.exit(1, f"hk2: {exc}\n")


if __name__ == "__main__":
    main()
