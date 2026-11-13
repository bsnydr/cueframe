# ruff: noqa: E501 - inline HTML template and check descriptions
"""Render the HK #2 Cueframe movie, then verify it and build a local review page.

``render`` binds the ported scene callbacks (``motion.py``) to Cueframe ``animate`` over
the prepared voice and cue sheet. ``verify`` decodes the result, checks frames, colour,
timing, audio lineage and hashes, compares matched frames with the original final film
and writes review media under the output folder (all ignored local media).
"""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cueframe.animate import animate
from cueframe.cues import load_cues
from cueframe.media import MediaError, run, safe_path, tool, video_encode_args

try:  # package import (tests) or script import (examples/hk2/run.py)
    from .motion import FPS, H, Hk2Scenes, W
except ImportError:  # pragma: no cover - script mode
    from motion import FPS, H, Hk2Scenes, W

MOVIE, RECORD, REPORT = "hk2-cueframe.mp4", "render-record.json", "verify-report.json"
FONTS = ("inputs/SpaceGrotesk-Regular.ttf", "inputs/SpaceGrotesk-Medium.ttf")
REFERENCE = "inputs/hk2-final-reference.mp4"
ATTRIBUTION = {
    "original": "Original HK #2 film: bespoke Python/Pillow/FFmpeg production (series and "
                "animation recipes), final master used here only as the comparison reference.",
    "new": "This movie: Cueframe animate with example-specific scene callbacks that port the "
           "first accepted illustration segment and the caption typesetting from that production.",
}
# Thresholds for matched decoded frames (two independent lossy encodes of the same picture).
PSNR_MEDIAN_MIN, PSNR_MIN, CHANGED_LEVEL, CHANGED_MAX = 30.0, 24.0, 64, 0.0025


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_inputs(prepared: Path, sources: Path, *, reference: bool) -> tuple[dict, dict]:
    plan = read_json(prepared / "plan.json")
    wanted = {name: (prepared / name, h) for name, h in plan["prepared_sha256"].items()}
    names = FONTS + ((REFERENCE,) if reference else ())
    wanted |= {n: (sources / n.split("/", 1)[1], plan["source_sha256"][n]) for n in names}
    hashes = {}
    for name, (path, digest) in wanted.items():
        hashes[name] = file_hash(path)
        if hashes[name] != digest:
            raise ValueError(f"{name}: SHA-256 differs from the prepared plan")
    hashes["plan.json"] = file_hash(prepared / "plan.json")
    return plan, hashes


def render(prepared: Path, sources: Path, out: Path) -> dict:
    plan, hashes = checked_inputs(prepared, sources, reference=False)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f".partial-{MOVIE}"
    scenes = Hk2Scenes(plan, prepared, *(sources / n.split("/", 1)[1] for n in FONTS))
    try:
        result = animate(prepared / "voice.wav", load_cues(prepared / "cues.json"), partial,
                         scene_set=scenes.scene_set())
        scenes.lobby.finish()
    finally:
        scenes.lobby.close()
    if result.n_frames != plan["frames"] or len(scenes.records) != plan["frames"]:
        raise MediaError(f"rendered {result.n_frames} frames, expected {plan['frames']}")
    if checked_inputs(prepared, sources, reference=False)[1] != hashes:
        raise MediaError("inputs changed during rendering")
    partial.replace(out / MOVIE)
    summary = result.summary() | {"out": MOVIE, "sha256": file_hash(out / MOVIE)}
    write_json(out / RECORD, {"attribution": ATTRIBUTION, "result": summary, "inputs_sha256": hashes,
                              "frames": scenes.records})
    return {"status": "rendered; run verify for checks and review media", **summary}


# ---------------------------------------------------------------- decoding helpers
def frames(path: Path, count: int, *, skip: int = 0, exact: bool = True):
    """Yield ``count`` RGB frames after ``skip``; with ``exact`` require the stream to end there."""
    proc = subprocess.Popen(  # noqa: S603 - resolved executable, argv list
        [tool("ffmpeg"), "-v", "error", "-nostdin", "-i", safe_path(path), "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False,
    )
    size = W * H * 3
    try:
        for i in range(skip + count):
            data = proc.stdout.read(size)
            if len(data) != size:
                raise MediaError(f"{path.name}: only {i} decodable frames")
            if i >= skip:
                yield np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)
        if exact and (proc.stdout.read(size) or proc.wait(timeout=60)):
            raise MediaError(f"{path.name}: more than {skip + count} frames or decode error")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def pcm24(path: Path) -> tuple[np.ndarray, str]:
    with wave.open(str(path), "rb") as r:
        if r.getsampwidth() != 3:
            raise ValueError("prepared voice must be 24-bit PCM")
        channels, data = r.getnchannels(), r.readframes(r.getnframes())
    b = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
    v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return v.reshape(-1, channels) / float(1 << 23), hashlib.sha256(data).hexdigest()


def best_lag(decoded: np.ndarray, source: np.ndarray, window: int = 4800) -> int:
    a, b = decoded.mean(axis=1), source.mean(axis=1)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lags = np.r_[np.arange(window + 1), np.arange(-window, 0)]
    return int(lags[np.argmax(np.r_[c[: window + 1], c[-window:]])])


def stream_info(path: Path) -> list[dict]:
    keys = ("codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,"
            "duration,color_space,color_transfer,color_primaries,color_range,sample_rate,channels")
    out = run(["ffprobe", "-v", "error", "-show_entries", f"stream={keys}", "-of", "json",
               safe_path(path)])
    return json.loads(out.stdout)["streams"]


# ---------------------------------------------------------------- verify
def verify(prepared: Path, sources: Path, out: Path) -> tuple[dict, bool]:
    plan, hashes = checked_inputs(prepared, sources, reference=True)
    movie = out / MOVIE
    if not (movie.exists() and (out / RECORD).exists()):
        raise ValueError(f"{movie} or {RECORD} is missing; run render first")
    record = read_json(out / RECORD)
    n, fps = plan["frames"], Fraction(plan["fps"])
    checks = []

    def check(name, ok, **detail):
        checks.append({"check": name, "pass": bool(ok), **detail})

    movie_hash = file_hash(movie)
    check("output hash matches render record", movie_hash == record["result"]["sha256"],
          sha256=movie_hash)
    check("inputs unchanged since render", all(record["inputs_sha256"][k] == v for k, v in hashes.items()
                                               if k != REFERENCE), inputs_sha256=hashes)

    streams = stream_info(movie)
    v = [s for s in streams if s["codec_type"] == "video"]
    a = [s for s in streams if s["codec_type"] == "audio"]
    vs = v[0] if len(v) == 1 else {}
    asr = a[0] if len(a) == 1 else {}
    check("one H.264 1080x1920 yuv420p video stream", len(v) == 1 and (
        vs.get("codec_name"), vs.get("width"), vs.get("height"), vs.get("pix_fmt")
    ) == ("h264", W, H, "yuv420p"), stream=vs)
    check("constant 30 fps, 405 frames, 13.5 s", Fraction(vs.get("r_frame_rate", "0/1")) == fps
          == Fraction(vs.get("avg_frame_rate", "0/1")) and int(vs.get("nb_frames", -1)) == n
          and abs(float(vs.get("duration", 0)) - n / fps) < 0.5 / float(fps),
          expected_frames=n, expected_s=float(n / fps))
    check("SDR BT.709 colour tags", [vs.get(k) for k in ("color_space", "color_transfer",
          "color_primaries")] == ["bt709"] * 3, color_range=vs.get("color_range"))
    check("one AAC 48 kHz stereo audio stream", len(a) == 1 and (
        asr.get("codec_name"), asr.get("sample_rate"), asr.get("channels")) == ("aac", "48000", 2),
        stream=asr)
    run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", safe_path(movie), "-f", "null", "-"])
    check("full decode without errors", True)

    # Audio lineage: exact PCM selection -> prepared voice -> lossy AAC in the movie.
    source, pcm_hash = pcm24(prepared / "voice.wav")
    raw = run(["ffmpeg", "-v", "error", "-nostdin", "-i", safe_path(movie), "-map", "0:a:0",
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", "48000", "-"]).stdout
    decoded = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
    lag = best_lag(decoded, source)
    m = min(len(source) - max(-lag, 0), len(decoded) - max(lag, 0))
    ref = source[max(-lag, 0): max(-lag, 0) + m]
    err = decoded[max(lag, 0): max(lag, 0) + m] - ref
    snr = 10 * np.log10(np.sum(ref**2) / max(np.sum(err**2), 1e-20))
    check("prepared voice is the selected processed PCM", pcm_hash == plan["audio"]["selected_pcm_sha256"],
          sha256=pcm_hash)
    check("AAC aligned to prepared PCM", lag == 0 and abs(len(decoded) - len(source)) <= 1024
          and snr >= 15.0, lag_samples=lag, decoded_samples=len(decoded),
          source_samples=len(source), aligned_snr_db=round(float(snr), 2),
          note="lossy AAC: waveform correspondence only; no listening performed")

    # Motion/caption state recorded by the scene callbacks.
    recs = record["frames"]
    words = plan["captions"]["words"]
    unreliable = {w["text"] for w in words if not w["reliable"]}
    cut = plan["motion"]["excerpt_frames"][0]
    last = recs[-1] if recs else {}
    check("scene per frame follows the cue sheet", [r["frame"] for r in recs] == list(range(n))
          and all(r["scene"] == ("lobby" if r["frame"] < cut else "agreement") for r in recs),
          hard_cut_frame=cut)
    check("captions: none on handle frame 0, every later frame captioned",
          recs[0]["page"] is None and all(r["page"] is not None for r in recs[1:]))
    check("low-confidence words never highlighted", not {r["active"] for r in recs} & unreliable,
          unreliable=sorted(unreliable))
    check("final frame holds the paused lawyer gate", last.get("gate") == "Lawyer review required"
          and last.get("cells") == 6 and last.get("dest") and last.get("card")
          == plan["motion"]["geometry"]["card_schematic"], final_state=last)

    # Matched decoded frames: new frame f <-> original final frame f + comparison origin.
    c0 = plan["origins"]["comparison_frames"][0]
    match0 = plan["origins"]["matched_picture_from_frame"]
    keep = transition_frames(plan)
    sheet_frames = sorted({0, 48, 95, 96, 150, 183, 213, 246, 290, 345, 380, n - 1})
    stats, kept, sheet = [], {}, {}
    pairs = zip(frames(movie, n), frames(sources / REFERENCE.split("/", 1)[1], n, skip=c0,
                                          exact=False), strict=True)
    for f, (new, old) in enumerate(pairs):
        d = np.abs(new.astype(np.int16) - old.astype(np.int16))
        mse = float(np.mean(d.astype(np.float32) ** 2))
        psnr = 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)
        changed = float(np.mean(d.max(axis=2) > CHANGED_LEVEL))
        stats.append({"frame": f, "reference_frame": f + c0, "psnr_db": round(psnr, 2),
                      "changed_fraction": round(changed, 5)})
        if f in keep:
            kept[f] = (old.copy(), new.copy())
        if f in sheet_frames:
            sheet[f] = new.copy()
    check("decoded new movie has exactly 405 frames", len(stats) == n)
    regions = {"lobby": (match0, cut), "illustration": (cut, n)}
    for name, (f0, f1) in regions.items():
        sel = stats[f0:f1]
        ps = [s["psnr_db"] for s in sel]
        worst = sorted(sel, key=lambda s: s["psnr_db"])[:3]
        check(f"matched {name} frames {f0}-{f1 - 1} agree with the original final film",
              ps and np.median(ps) >= PSNR_MEDIAN_MIN and min(ps) >= PSNR_MIN
              and max(s["changed_fraction"] for s in sel) <= CHANGED_MAX,
              median_psnr_db=round(float(np.median(ps)), 2) if ps else None,
              min_psnr_db=min(ps) if ps else None,
              max_changed_fraction=max((s["changed_fraction"] for s in sel), default=None),
              worst=worst, thresholds={"median_psnr": PSNR_MEDIAN_MIN, "min_psnr": PSNR_MIN,
                                       "changed_level": CHANGED_LEVEL, "changed_max": CHANGED_MAX})
    stats0 = stats[0] if stats else None

    review = write_review(plan, sources, out, kept, sheet)
    ok = all(c["pass"] for c in checks)
    report = {
        "status": "automated checks passed" if ok else "automated checks FAILED",
        "human_review": "pending: no listening, visual acceptance or editorial sign-off performed",
        "attribution": ATTRIBUTION, "movie": MOVIE, "movie_sha256": movie_hash,
        "checks": checks, "frame_0_handle_vs_original": stats0, "frame_stats": stats,
        "review_files": review,
    }
    write_json(out / REPORT, report)
    write_index(plan, out, report, record)
    summary = {"status": report["status"], "failed": [c["check"] for c in checks if not c["pass"]],
               "report": str(out / REPORT), "index": str(out / "index.html")}
    return summary, ok


def transition_frames(plan) -> dict[int, str]:
    cut = plan["motion"]["excerpt_frames"][0]
    out = {cut - 1: "last lobby frame", cut: "hard cut to illustration", plan["frames"] - 1: "final frame"}
    for name, a in plan["motion"]["anchors_s"].items():
        f = round(a["excerpt"] * FPS) + 6
        out.setdefault(min(f, plan["frames"] - 1), f"{name} + 0.2 s")
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- review media
def label_font(sources: Path, size: int):
    return ImageFont.truetype(str(sources / FONTS[0].split("/", 1)[1]), size)


def write_review(plan, sources: Path, out: Path, kept, sheet) -> dict:
    font, small = label_font(sources, 44), label_font(sources, 22)
    # Labelled reference excerpt (picture + audio of the original final film).
    c0, c1 = plan["origins"]["comparison_frames"]
    fps = Fraction(plan["fps"])
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle((0, 0, W, 132), fill=(0, 0, 0, 170))
    d.text((32, 22), "REFERENCE: original bespoke production", font=font, fill=(255, 255, 255, 255))
    d.text((32, 82), f"final film frames {c0}-{c1 - 1} | not the Cueframe render", font=label_font(sources, 32),
           fill=(255, 200, 150, 255))
    label = out / ".reference-label.png"
    band.save(label)
    excerpt = out / "reference-excerpt.mp4"
    try:
        run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", safe_path(sources / REFERENCE.split("/", 1)[1]),
             "-i", safe_path(label), "-filter_complex",
             f"[0:v]trim=start_frame={c0}:end_frame={c1},setpts=PTS-STARTPTS[v];"
             f"[v][1:v]overlay=0:0,format=yuv420p[o];"
             f"[0:a]atrim=start={float(c0 / fps):.6f}:end={float(c1 / fps):.6f},asetpts=PTS-STARTPTS[a]",
             "-map", "[o]", "-map", "[a]", *video_encode_args(fps, set_rate=False),
             "-c:a", "aac", "-b:a", "192k", safe_path(excerpt)])
    finally:
        label.unlink(missing_ok=True)

    tw, th = 216, 384
    names = transition_frames(plan)
    tr = Image.new("RGB", (60 + 3 * (tw + 12), 50 + len(kept) * (th + 56)), (250, 250, 250))
    d = ImageDraw.Draw(tr)
    for i, head in enumerate(("original final film", "Cueframe render", "|difference| x4")):
        d.text((60 + i * (tw + 12), 12), head, font=small, fill=(17, 17, 17))
    for row, (f, (old, new)) in enumerate(sorted(kept.items())):
        y = 50 + row * (th + 56)
        diff = np.clip(np.abs(new.astype(np.int16) - old.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        for i, arr in enumerate((old, new, diff)):
            tr.paste(Image.fromarray(arr).resize((tw, th), Image.LANCZOS), (60 + i * (tw + 12), y))
        d.text((60, y + th + 6), f"frame {f} ({f / FPS:.2f} s) <-> original {f + c0}: {names[f]}",
               font=small, fill=(17, 17, 17))
    tr.save(out / "transitions.jpg", quality=85)

    cols, sw, sh = 6, 180, 320
    rows = (len(sheet) + cols - 1) // cols
    cs = Image.new("RGB", (cols * (sw + 8) + 8, rows * (sh + 36) + 8), (250, 250, 250))
    d = ImageDraw.Draw(cs)
    for i, (f, arr) in enumerate(sorted(sheet.items())):
        x, y = 8 + (i % cols) * (sw + 8), 8 + (i // cols) * (sh + 36)
        cs.paste(Image.fromarray(arr).resize((sw, sh), Image.LANCZOS), (x, y))
        d.text((x, y + sh + 4), f"f{f}  {f / FPS:.2f}s", font=small, fill=(17, 17, 17))
    cs.save(out / "contact-sheet.jpg", quality=85)
    return {"reference_excerpt": excerpt.name, "transitions": "transitions.jpg",
            "contact_sheet": "contact-sheet.jpg"}


def write_index(plan, out: Path, report, record) -> None:
    e = html.escape
    fps = Fraction(plan["fps"])
    c0 = plan["origins"]["comparison_frames"][0]
    b0 = plan["origins"]["body_frames"][0]
    cam0 = plan["audio"]["camera_context_seconds"][0]
    lobby = next(s for s in plan["shots"] if s["scene"] == "lobby")
    words = plan["captions"]["words"]
    rows = []
    for f, name in transition_frames(plan).items():
        src = (f"{lobby['source']} frame {lobby['source_frames'][0] + f}" if f < lobby["frames"][1]
               else "illustration (drawn)")
        rows.append(f"<tr><td>{f}</td><td>{float(f / fps):.3f}</td><td>{f + b0}</td><td>{f + c0}</td>"
                    f"<td>{cam0 + float(f / fps):.3f}</td><td>{e(src)}</td><td>{e(name)}</td></tr>")
    pages = []
    for p in plan["captions"]["pages"]:
        f0, f1 = p["frames"]
        off = [words[k]["text"] for k in p["word_indices"] if not words[k]["reliable"]]
        pages.append(f"<tr><td>{p['page']}</td><td>{e(' / '.join(p['lines']))}</td><td>{f0}-{f1 - 1}</td>"
                     f"<td>{p['body_frames'][0]}-{p['body_frames'][1] - 1}</td>"
                     f"<td>{f0 + c0}-{f1 - 1 + c0}</td><td>{e(', '.join(off)) or '-'}</td></tr>")
    checks = "".join(
        f"<tr><td>{'PASS' if c['pass'] else '<b>FAIL</b>'}</td><td>{e(c['check'])}</td></tr>"
        for c in report["checks"])
    a = report["attribution"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HK #2 Cueframe review</title>
<style>body{{font:16px/1.45 system-ui,sans-serif;margin:0 auto;max-width:720px;padding:12px;color:#111}}
video{{display:block;width:auto;max-width:100%;height:min(70vh,760px);aspect-ratio:9/16;object-fit:contain;margin:auto;background:#000}}
img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid #ddd;padding:4px;text-align:left;vertical-align:top}}
.s{{padding:10px;background:#f4f4f4;border-left:4px solid #ff7300}}.w{{overflow-x:auto}}</style></head><body>
<h1>HK #2 — Cueframe render</h1>
<p>Local review draft · {plan['duration_s']} seconds. The continuous agreement-and-lawyer passage,
recreated from the original footage, speech and editable animation. Nothing published.</p>
<video controls playsinline preload="metadata" src="{MOVIE}"></video>
<p class="s"><b>Automated checks:</b> {e(report['status'])}<br><b>Human review:</b> {e(report['human_review'])}</p>
<p>Review the opening “When you tell your agent…” and final “not a lawyer” at normal speed,
then compare the card-to-route transition with the original below. The cut boundaries still need a listening check.</p>
<h2>Attribution</h2><p>{e(a['new'])}</p><p>{e(a['original'])}</p>
<h2>Reference excerpt (original)</h2>
<video controls playsinline preload="metadata" src="{report['review_files']['reference_excerpt']}"></video>
<details><summary>Frame comparisons, source mapping and verification details</summary>
<p>Output SHA-256 <code>{e(report['movie_sha256'])}</code>; 1080×1920, 30 fps, {plan['frames']} frames,
{plan['duration_s']} s. Frame 0 is an extra lobby handle; picture correspondence starts at frame
{plan['origins']['matched_picture_from_frame']} (original {c0 + 1}).</p>
<h2>Transitions: original | Cueframe | difference</h2><img src="transitions.jpg" alt="transition comparison">
<h2>Contact sheet (decoded Cueframe frames)</h2><img src="contact-sheet.jpg" alt="contact sheet">
<h2>Automated checks</h2><div class="w"><table>{checks}</table></div>
<p>Details: <a href="{REPORT}">{REPORT}</a>, render record <a href="{RECORD}">{RECORD}</a>.</p>
<h2>Source time mapping</h2><div class="w"><table><tr><th>frame</th><th>s</th><th>body frame</th>
<th>original final frame</th><th>camera audio s ({e(plan['audio']['camera_context_source'])})</th>
<th>picture source</th><th>moment</th></tr>{''.join(rows)}</table></div>
<h2>Caption mapping</h2><div class="w"><table><tr><th>page</th><th>lines</th><th>frames</th>
<th>body frames</th><th>original frames</th><th>never highlighted</th></tr>{''.join(pages)}</table></div>
<p>Render summary: <code>{e(json.dumps(record['result']))}</code></p></details></body></html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")
