# ruff: noqa: E501 - dense drawing recipe kept line-for-line comparable with the original
"""Cueframe scene callbacks for HK #2: the accepted illustration and captions, ported.

This is a port of two editable recipes from the original bespoke Python/Pillow/FFmpeg
production (kept locally, never imported or executed here):

* ``animation-segment.py`` — only its first segment, body frames [399, 709), cut at the
  excerpt end (body 708): one persistent agreement card -> note -> step back into a
  schematic -> route -> handoff squares -> counterparty -> paused at the lawyer gate.
* ``series-produce.py`` — caption typesetting: Space Grotesk Medium, 78 px cap height,
  last baseline 1400; footage white / current #ff7300 / black 2.5 px edge, paper ink /
  current #9a3d00 / no edge; low-confidence words are never highlighted.

Timings come from the prepared ``plan.json`` on the original body clock
(``ctx.frame + 303``) so every comparison and easing is evaluated exactly as the original.
Geometry the plan carries is cross-checked against the recipe so a different font or
plan fails loudly instead of drifting.
"""

from __future__ import annotations

import math
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from cueframe.media import MediaError, safe_path, tool

W, H, FPS = 1080, 1920, 30
WHITE, OFF, INK = (255, 255, 255), (250, 250, 250), (17, 17, 17)
ORANGE, ORANGE_TXT, DIM, BLACK = (255, 115, 0), (154, 61, 0), (85, 85, 85), (0, 0, 0)
HAIR, RULE, LINE = (226, 226, 226), (229, 229, 229), (226, 226, 222)
K = 3  # supersampling factor of the original type renderer
CAP, LAST_BASELINE, EDGE = 78, 1400, 2.5
G_GAP = (762, 848)  # route gap behind the gate line (recipe constant, not in the plan)


# ---------------------------------------------------------------- pure timing helpers
def clamp(v):
    return max(0.0, min(1.0, v))


def ease_out(v):
    return 1 - (1 - clamp(v)) ** 3


def ease_in_out(v):
    v = clamp(v)
    return 4 * v**3 if v < 0.5 else 1 - (-2 * v + 2) ** 3 / 2


def lerp(a, b, u):
    return tuple(round(p + (q - p) * u) for p, q in zip(a, b, strict=True))


def on(te, t):
    """Frame-centred event test shared by captions and motion in the original."""
    return te <= t + 0.5 / FPS


def page_at(pages, body_frame):
    """Caption page covering ``body_frame`` (half-open body frame spans), or None."""
    return next((p for p in pages if p["body_frames"][0] <= body_frame < p["body_frames"][1]), None)


def active_word(page, words, body_frame):
    """Index of the highlighted word: latest started word, unless it is low-confidence.

    As in the original, a low-confidence current word suppresses the highlight entirely
    rather than leaving the previous word lit.
    """
    t, cur = body_frame / FPS, None
    for k in page["word_indices"]:
        if words[k]["body_start_s"] <= t + 0.5 / FPS:
            cur = k
    return cur if cur is not None and words[cur]["reliable"] else None


def card_geom(t, morph_t, morph_s, full, schematic):
    """Agreement card rectangle: full size until ``morph_t`` then eased to the schematic."""
    if t < morph_t:
        return full, 0.0
    u = ease_in_out((t - morph_t) / morph_s)
    a, b = full, schematic
    s = (a[2] + (b[2] - a[2]) * u) / full[2]
    x, y = round(a[0] + (b[0] - a[0]) * u), round(a[1] + (b[1] - a[1]) * u)
    return (x, y, round(full[2] * s), round(full[3] * s)), u


@lru_cache(None)
def shadow(w_, h_, blur=18, op=0.12, dy=8):
    m = Image.new("L", (w_ + 4 * blur, h_ + 4 * blur), 0)
    ImageDraw.Draw(m).rectangle(
        (2 * blur, 2 * blur + dy, 2 * blur + w_ - 1, 2 * blur + dy + h_ - 1), fill=round(255 * op)
    )
    m = m.filter(ImageFilter.GaussianBlur(blur / 2))
    s = Image.new("RGBA", m.size, BLACK + (0,))
    s.putalpha(m)
    return s, 2 * blur


# ---------------------------------------------------------------- type (original r4/r5 recipe)
class Type:
    def __init__(self, regular: Path, medium: Path):
        self.reg, self.med = str(regular), str(medium)
        self.font = lru_cache(None)(ImageFont.truetype)
        box = self.font(self.reg, 1000).getbbox("H")
        self.capr = (box[3] - box[1]) / 1000
        self.cpx = round(CAP / self.capr)
        self.clead = round(self.cpx * 1.12)
        self.rimg = lru_cache(4096)(self._rimg)

    def _rimg(self, runs, size, fp, stroke=0.0):
        f = self.font(fp, size * K)
        space, xs, x = f.getlength(" "), [], 0.0
        for i, (text, _) in enumerate(runs):
            xs.append(x)
            x += f.getlength(text) + (space if i < len(runs) - 1 else 0)
        asc, dsc = f.getmetrics()
        pad = K * math.ceil(stroke * 2 + 4)
        top = pad + ((-(pad + asc)) % K)
        wd, ht = math.ceil((x + 2 * pad) / K) * K, math.ceil((top + asc + dsc + pad) / K) * K
        big = Image.new("RGBA", (wd, ht), (0, 0, 0, 0))
        if stroke > 0:
            edge = Image.new("RGBA", (wd, ht), (0, 0, 0, 0))
            ed = ImageDraw.Draw(edge)
            for (text, _), wx in zip(runs, xs, strict=True):
                ed.text((pad + wx, top + asc), text, font=f, fill=BLACK + (255,), anchor="ls",
                        stroke_width=round(stroke * K), stroke_fill=BLACK + (255,))
            big.alpha_composite(edge)
        d = ImageDraw.Draw(big)
        for (text, colour), wx in zip(runs, xs, strict=True):
            d.text((pad + wx, top + asc), text, font=f, fill=tuple(colour) + (255,), anchor="ls")
        small = big.convert("RGBa").resize((wd // K, ht // K), Image.LANCZOS).convert("RGBA")
        return small, (top + asc) // K, x / K, pad // K

    def put(self, frame, runs, size, base_y, fp=None, cx=None, x=None, stroke=0.0):
        im, bo, adv, pd = self.rimg(tuple(runs), size, fp or self.reg, stroke)
        x0 = round(cx - adv / 2) - pd if cx is not None else x - pd
        frame.alpha_composite(im, (x0, base_y - bo))
        return [x0 + pd, base_y - round(size * self.capr), x0 + pd + round(adv), base_y]

    def adv(self, text, size):
        return self.rimg(((text, INK),), size, self.reg)[2]


class Captions:
    """Burned-in captions from the prepared pages/words (original line wrapping)."""

    def __init__(self, plan, ty: Type):
        style = plan["captions"]["style"]
        got = (ty.cpx, ty.clead, LAST_BASELINE, CAP, EDGE)
        want = tuple(style[k] for k in ("font_px", "lead_px", "last_baseline_y", "cap_px", "edge_px"))
        if got != want:
            raise ValueError(f"caption metrics {got} differ from the original {want}")
        self.ty, self.pages, self.words = ty, plan["captions"]["pages"], plan["captions"]["words"]

    def draw(self, frame, body_frame, *, paper):
        p = page_at(self.pages, body_frame)
        if p is None:
            return None, None
        aw, k, n = active_word(p, self.words, body_frame), p["word_indices"][0], len(p["lines"])
        for j, line in enumerate(p["lines"]):
            runs = []
            for w in line.split(" "):
                if self.words[k]["text"] != w:
                    raise ValueError(f"caption page {p['page']} line does not match its words")
                current = k == aw
                runs.append((w, (ORANGE_TXT if paper else ORANGE) if current else (INK if paper else WHITE)))
                k += 1
            self.ty.put(frame, runs, self.ty.cpx, LAST_BASELINE - (n - 1 - j) * self.ty.clead,
                        fp=self.ty.med, cx=W / 2, stroke=0 if paper else EDGE)
        return p["page"], (self.words[aw]["text"] if aw is not None else None)


# ---------------------------------------------------------------- the illustration (segment 1)
class Illustration:
    def __init__(self, plan, ty: Type):
        m, g = plan["motion"], plan["motion"]["geometry"]
        self.ty = ty
        self.T = {k: v["body"] for k, v in m["anchors_s"].items()}
        self.full, self.s1 = tuple(g["card_full"]), tuple(g["card_schematic"])
        self.rx, self.r_top = g["route_x"], self.s1[1] + self.s1[3]
        c = g["cells"]
        self.cell, self.pitch, self.ncell, self.cell0, self.cell_step = (
            c["size"], c["pitch"], c["n"], c["first_top"], c["step_s"])
        d = m["durations_s"]
        self.morph_s, self.rail_s = d["morph"], d["route_draw"]
        self.recentre_s, self.decay_s = d["recentre"], d["accent_decay"]
        self.g_size, self.g_base = g["gate"]["font_px"], g["gate"]["baseline"]
        self.br_w, self.br_h, self.br_gap = g["gate"]["bracket_wh_gap"]
        self.gate_words = [(w["word"], w["body_in_s"], w["body_delete_s"]) for w in m["gate_words"]]
        self.events = sorted({e for _, ti, td in self.gate_words for e in (ti, td)})
        end_s = plan["origins"]["body_frames"][1] / FPS
        recipe = (
            self.cell0 == self.r_top + 16,
            (self.br_w, self.br_h, self.br_gap) == tuple(round(self.g_size * f) for f in (.2, .44, .17)),
            [w for w, *_ in self.gate_words] == ["Lawyer", "review", "required"],
            all(td >= end_s for *_, td in self.gate_words),
            set(self.T) >= {"note", "morph", "rail", "send", "dest", "gate", "reviewed", "lawyer"},
        )
        if not all(recipe):
            raise ValueError(f"plan motion disagrees with the accepted recipe: {recipe}")
        self.dest = self.chip_geom("Counterparty", 50, cx=self.rx, y=950)
        if self.dest != g["dest_chip"]:
            raise ValueError(f"destination chip {self.dest} differs from {g['dest_chip']}")
        self.card = {False: self.card_layer(False), True: self.card_layer(True)}
        self.card_scaled = lru_cache(64)(
            lambda w, h: self.card[True].convert("RGB").resize((w, h), Image.LANCZOS).convert("RGBA"))

    def chip_geom(self, text, size, cx, y):
        cap = round(size * self.ty.capr)
        h, pad, sq = round(2.9 * cap), round(1.1 * cap), round(0.62 * cap)
        w = round(self.ty.adv(text, size) + 2 * pad)
        x0 = round(cx - w / 2)
        return {"text": text, "size": size, "x": x0, "y": y, "w": w, "h": h, "pad": pad, "sq": sq,
                "base": y + round(h / 2 + cap / 2), "cap": cap}

    def draw_chip(self, frame, g, t0, t, col=INK):
        if not on(t0, t):
            return False
        d, x, y = ImageDraw.Draw(frame), g["x"], g["y"]
        if not on(t0 + 2 / FPS, t):
            d.rectangle((x, y, x + g["sq"] - 1, y + g["sq"] - 1), fill=ORANGE + (255,))
            return True
        sh, o = shadow(g["w"], g["h"])
        frame.alpha_composite(sh, (x - o, y - o))
        d.rectangle((x, y, x + g["w"] - 1, y + g["h"] - 1), fill=WHITE + (255,))
        d.rectangle((x, y, x + g["sq"] - 1, y + g["sq"] - 1), fill=ORANGE + (255,))
        self.ty.put(frame, ((g["text"], col),), g["size"], g["base"], x=x + g["pad"])
        return True

    def draw_note(self, im, ox, oy, t):
        """Note on the card bottom; ``t=None`` draws it complete, else its reveal."""
        d, nx, ny = ImageDraw.Draw(im), ox + 56, oy + self.full[3] - 170
        if t is None:
            hb, lab, txt = 104, True, True
        else:
            n0 = self.T["note"]
            if not on(n0, t):
                return
            hb = round(104 * ease_out((t - n0) / 0.15))
            lab, txt = on(n0 + 2 / FPS, t), on(n0 + 4 / FPS, t)
        if hb > 0:
            d.rectangle((nx, ny, nx + 5, ny + hb), fill=ORANGE + (255,))
        if lab:
            self.ty.put(im, (("Note", ORANGE_TXT),), 30, ny + 40, x=nx + 30)
        if txt:
            self.ty.put(im, (("Have it reviewed by a lawyer.", INK),), 34, ny + 88, x=nx + 30)

    def card_layer(self, note):
        cw, ch = self.full[2:]
        im = Image.new("RGBA", (cw, ch), WHITE + (255,))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 30, 30), fill=ORANGE + (255,))
        self.ty.put(im, (("Agreement", INK),), 46, 96, x=56)
        self.ty.put(im, (("Draft", DIM),), 30, 140, x=56)
        d.rectangle((56, 167, cw - 56, 168), fill=RULE + (255,))
        for i in range(10):
            ly, frac = 210 + i * 46, (0.92 if i % 4 != 3 else 0.55)
            d.rectangle((56, ly, 56 + round((cw - 112) * frac), ly + 14), fill=LINE + (255,))
        if note:
            self.draw_note(im, 0, 0, None)
        return im

    def draw_card(self, frame, t):
        (x, y, w, h), _ = card_geom(t, self.T["morph"], self.morph_s, self.full, self.s1)
        sh, o = shadow(w, h)
        frame.alpha_composite(sh, (x - o, y - o))
        if (w, h) == self.full[2:]:
            frame.alpha_composite(self.card[False], (x, y))
            self.draw_note(frame, x, y, t if t < self.T["morph"] else None)
        else:
            frame.alpha_composite(self.card_scaled(w, h), (x, y))
        self.ty.put(frame, (("Illustration", DIM),), 30, y - 34, x=x)
        return (x, y, w, h)

    def gate_vis(self, t):
        return [(w, ti) for w, ti, td in self.gate_words if on(ti, t) and not on(td, t)]

    def gate_layout(self, vis):
        return round(self.rx - self.ty.adv(" ".join(w for w, _ in vis), self.g_size) / 2) if vis else None

    def gate_line(self, frame, t):
        d, vis = ImageDraw.Draw(frame), self.gate_vis(t)
        yb = self.g_base - round(self.g_size * 0.05)
        if not vis:
            for xx in (self.rx - self.br_w - 2, self.rx + 2):
                d.rectangle((xx, yb - self.br_h, xx + self.br_w - 1, yb - 1), fill=INK + (255,))
            return "closed"
        left = anchor = self.gate_layout(vis)
        past = [e for e in self.events if on(e, t)]
        if past:
            te = past[-1]
            prev = self.gate_vis(te - 1 / FPS)
            names = [w for w, _ in prev]
            if prev and vis[0][0] in names:
                k = names.index(vis[0][0])
                xp = self.gate_layout(prev) + (
                    self.ty.adv(" ".join(names[:k]) + " ", self.g_size) if k else 0)
                anchor = round(xp + (left - xp) * ease_out((t - te) / self.recentre_s))
        runs = tuple((w, lerp(ORANGE_TXT, INK, ease_out((t - ti) / self.decay_s))) for w, ti in vis)
        bb = self.ty.put(frame, runs, self.g_size, self.g_base, x=anchor)
        for xx in (bb[0] - self.br_gap - self.br_w, bb[2] + self.br_gap):
            d.rectangle((xx, yb - self.br_h, xx + self.br_w - 1, yb - 1), fill=INK + (255,))
        return " ".join(w for w, _ in vis)

    def diagram(self, t):
        """Route, handoff squares, destination and gate on one transparent layer."""
        T, rx = self.T, self.rx
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d, st = ImageDraw.Draw(lay), {}
        if on(T["rail"], t):
            y_end = round(self.r_top + (self.dest["y"] - self.r_top) * ease_out((t - T["rail"]) / self.rail_s))
            segs = ([(self.r_top, y_end)] if not on(T["gate"], t)
                    else [(self.r_top, min(y_end, G_GAP[0])), (G_GAP[1], y_end)])
            for a, b in segs:
                if b > a:
                    d.rectangle((rx - 2, a, rx + 1, b - 1), fill=HAIR + (255,))
            st["route"] = True
        n = sum(1 for k in range(self.ncell) if on(T["send"] + self.cell_step * k, t))
        for k in range(n):
            yy = self.cell0 + self.pitch * k
            d.rectangle((rx - self.cell // 2, yy, rx + self.cell // 2 - 1, yy + self.cell - 1),
                        fill=ORANGE + (255,))
        st["cells"] = n
        dcol = lerp(INK, DIM, ease_out((t - T["gate"]) / 0.3)) if on(T["gate"], t) else INK
        if self.draw_chip(lay, self.dest, T["dest"], t, dcol):
            st["dest"] = True
        if on(T["gate"], t):
            st["gate"] = self.gate_line(lay, t)
        return lay, st

    def frame(self, body_frame):
        t = body_frame / FPS
        fr = Image.new("RGBA", (W, H), OFF + (255,))
        info = {}
        if on(self.T["morph"], t):
            lay, st = self.diagram(t)
            fr.alpha_composite(lay)
            info |= st
        info["card"] = list(self.draw_card(fr, t))
        return fr, info


# ---------------------------------------------------------------- lobby footage stream
class LobbyReader:
    """Decode the prepared lossless lobby clip one RGB frame at a time, strictly in order."""

    def __init__(self, path: Path, frames: int):
        self.path, self.frames, self.next = path, frames, 0
        self.proc = None

    def read(self, index: int) -> np.ndarray:
        if index != self.next or index >= self.frames:
            raise MediaError(f"lobby frame {index} requested out of order (expected {self.next})")
        if self.proc is None:
            self.proc = subprocess.Popen(  # noqa: S603 - resolved executable, argv list
                [tool("ffmpeg"), "-v", "error", "-nostdin", "-i", safe_path(self.path),
                 "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
            )
        size = W * H * 3
        data = self.proc.stdout.read(size)
        if len(data) != size:
            raise MediaError(f"lobby clip ended before frame {index}")
        self.next += 1
        return np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)

    def finish(self) -> None:
        if self.proc is None or self.next != self.frames:
            raise MediaError(f"lobby used {self.next} of {self.frames} prepared frames")
        extra = self.proc.stdout.read()
        err = self.proc.stderr.read()
        if self.proc.wait(timeout=60) or extra or err:
            raise MediaError(f"lobby decode: {len(extra)} extra bytes, {err.decode(errors='replace')}")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


class Hk2Scenes:
    """Cueframe scene set: ``lobby`` (footage + captions), ``agreement`` (paper + captions)."""

    def __init__(self, plan, prepared: Path, regular: Path, medium: Path):
        self.offset = plan["origins"]["body_frames"][0]
        self.shots = {s["scene"]: s["frames"] for s in plan["shots"]}
        ty = Type(regular, medium)
        self.captions, self.illustration = Captions(plan, ty), Illustration(plan, ty)
        lobby = self.shots["lobby"]
        self.lobby = LobbyReader(prepared / "lobby-clean.mkv", lobby[1] - lobby[0])
        self.records = []

    def scene_set(self):
        return {"idle": self.idle, "lobby": self.lobby_scene, "agreement": self.agreement_scene}

    def idle(self, ctx, canvas):
        raise MediaError(f"HK #2 cue sheet has no idle gaps (frame {ctx.frame})")

    def _record(self, ctx, body, page, active, **state):
        self.records.append({"frame": ctx.frame, "t": round(ctx.t, 6), "scene": ctx.cue.scene,
                             "cue_frame": ctx.cue_frame, "body_frame": body,
                             "page": page, "active": active, **state})

    def lobby_scene(self, ctx, canvas):
        body = ctx.frame + self.offset
        img = Image.fromarray(self.lobby.read(ctx.cue_frame), "RGB").convert("RGBA")
        page, active = self.captions.draw(img, body, paper=False)
        canvas.paste(img.convert("RGB"))
        self._record(ctx, body, page, active)

    def agreement_scene(self, ctx, canvas):
        body = ctx.frame + self.offset
        fr, info = self.illustration.frame(body)
        page, active = self.captions.draw(fr, body, paper=True)
        canvas.paste(fr.convert("RGB"))
        self._record(ctx, body, page, active, **info)
