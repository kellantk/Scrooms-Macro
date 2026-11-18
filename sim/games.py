"""The fake minigames. Sizes are fractions of the window height, the same
ones the detectors measure in the real game; every piece of state carries a
time stamp so a frame can be drawn for any moment (the screen lags the game).
"""
import heapq
import math

import numpy as np
from PIL import Image, ImageDraw

from .core import background

WHITE = (250, 250, 250)


def _dim(view, alpha, color=(0, 0, 0)):
    view[:] = (view.astype(np.float32) * (1 - alpha) + np.asarray(color, np.float32) * alpha).astype(np.uint8)


def _diamond(d, x, y, a, fill, outline=None):
    d.polygon([(x - a, y), (x, y - a), (x + a, y), (x, y + a)], fill=fill, outline=outline)


class Stage:
    """One part of a cast (ring, count-in, a minigame)."""
    name = "?"
    pad = 0

    def __init__(self, w, h, bg, rng):
        self.w, self.h, self.bg, self.rng = w, h, bg, rng
        self.t0 = None
        self.results = []
        self._base = None

    def start(self, t):
        self.t0 = t

    def box(self):
        raise NotImplementedError

    def _box(self):
        x0, y0, x1, y1 = self.box()
        return max(0, int(x0)), max(0, int(y0)), min(self.w, int(x1)), min(self.h, int(y1))

    def base(self):
        """The parts that never change, drawn once."""
        if self._base is None:
            x0, y0, x1, y1 = self._box()
            arr = self.bg[y0:y1, x0:x1].copy()
            self.static(arr, x0, y0)
            self._base = arr
        return self._base

    def static(self, arr, ox, oy):
        pass

    def draw(self, frame, t):
        x0, y0, x1, y1 = self._box()
        img = Image.fromarray(self.base().copy())
        self.dynamic(ImageDraw.Draw(img), img, x0, y0, t)
        frame[y0:y1, x0:x1] = np.asarray(img)

    def visible(self, t):
        return self.t0 is not None and self.t0 <= t < self.end()

    def end(self):
        return math.inf

    def on_input(self, t, kind, arg):
        pass

    def tick(self, t):
        pass

    def close(self):
        """Called once when the stage is over: judge what was never hit."""


# ---------------------------------------------------------------- ring
class Ring(Stage):
    """The cast ring: a needle spins, click when it's on the green."""
    name = "ring"

    def __init__(self, w, h, bg, rng, omega=None, center=None, zone_half=30.0, perfect_half=6.0, freeze=0.55):
        super().__init__(w, h, bg, rng)
        self.R = 100 * h / 1360
        self.cx, self.cy = center or (0.62 * w, 0.5 * h)
        om = omega if omega is not None else float(rng.choice([-1, 1]) * rng.uniform(285, 305))
        self.omega = math.radians(om)
        self.zone_half, self.perfect_half, self.freeze = zone_half, perfect_half, freeze
        self.zone_mid = float(rng.uniform(-math.pi, math.pi))
        self.perfect = self.zone_mid + math.radians(float(rng.uniform(-14, 14)))
        self.phase = float(rng.uniform(-math.pi, math.pi))
        self.clicked = None

    def box(self):
        m = 1.5 * self.R
        return self.cx - m, self.cy - m, self.cx + m, self.cy + m

    def end(self):
        if self.clicked is not None:
            return self.clicked + self.freeze
        return self.t0 + 9.0

    def angle(self, t):
        tt = t if self.clicked is None else min(t, self.clicked)
        return self.phase + self.omega * (tt - self.t0)

    def on_input(self, t, kind, arg):
        if kind == "down" and arg == "left" and self.clicked is None:
            self.clicked = t
            d = (self.angle(t) - self.perfect + math.pi) % (2 * math.pi) - math.pi
            err = d / self.omega * 1000                  # + = late
            off = abs(math.degrees(d))
            zone = abs(math.degrees((self.angle(t) - self.zone_mid + math.pi) % (2 * math.pi) - math.pi))
            verdict = "perfect" if off <= self.perfect_half else "good" if zone <= self.zone_half else "miss"
            self.results.append({"stage": "ring", "err": err, "verdict": verdict})

    def close(self):
        if self.clicked is None:
            self.results.append({"stage": "ring", "err": None, "verdict": "miss"})

    def dynamic(self, d, img, ox, oy, t):
        R, cx, cy = self.R, self.cx - ox, self.cy - oy
        if self.clicked is not None and self.clicked <= t < self.clicked + 0.3:
            # the click flashes the ring away; it comes back for a moment with
            # the needle frozen where the click landed
            d.ellipse([cx - 1.3 * R, cy - 1.3 * R, cx + 1.3 * R, cy + 1.3 * R], fill=(250, 250, 240))
            return
        bb = [cx - 1.18 * R, cy - 1.18 * R, cx + 1.18 * R, cy + 1.18 * R]
        wid = max(3, int(round(0.36 * R)))
        d.ellipse(bb, outline=(20, 38, 22), width=wid)
        zm, zh = math.degrees(self.zone_mid), self.zone_half
        d.arc(bb, zm - zh, zm + zh, fill=(188, 246, 8), width=wid)
        pm = math.degrees(self.perfect)
        d.arc(bb, pm - self.perfect_half, pm + self.perfect_half, fill=(22, 246, 24), width=wid)
        k = 0.2 * R
        d.rounded_rectangle([cx - k, cy - k, cx + k, cy + k], radius=3, fill=(40, 40, 44))
        a = self.angle(t)
        pts = []
        for da, rr in ((-3.0, 0.84), (3.0, 0.84), (5.5, 1.16), (-5.5, 1.16)):
            b = a + math.radians(da)
            pts.append((cx + rr * R * math.cos(b), cy + rr * R * math.sin(b)))
        d.polygon(pts, fill=(255, 255, 255))


# ------------------------------------------------------------ count-in
class CountIn(Stage):
    """Rings shrink onto a circle; click as each one touches it."""
    name = "countin"

    def __init__(self, w, h, bg, rng, speed=None, gap=None, n=4, glow=True):
        super().__init__(w, h, bg, rng)
        self.r0 = 0.0705 * h
        # (not dead center: the search for it works on a grid, and a thin
        # outline must be found wherever it falls between the grid's points)
        self.cx, self.cy = 0.5 * w + float(rng.uniform(-9, 9)), 0.5 * h + float(rng.uniform(-9, 9))
        self.speed = (speed if speed is not None else float(rng.uniform(1.0, 1.45))) * self.r0
        self.gap = gap if gap is not None else float(rng.uniform(0.47, 0.66))
        self.n, self.glow = n, glow
        self.r_start = 2.3 * self.r0
        self.target = 1.04 * self.r0        # Perfect: the ring lying on the circle (1.0-1.08 radii on recordings)
        self.rings = []
        self.clicks = []

    def start(self, t):
        self.t0 = t
        for k in range(self.n):
            spawn = t + 0.9 + k * self.gap
            self.rings.append({"spawn": spawn, "T": spawn + (self.r_start - self.target) / self.speed, "hit": None})

    def box(self):
        m = 2.5 * self.r0
        return self.cx - m, self.cy - m, self.cx + m, self.cy + m

    def end(self):
        return self.rings[-1]["T"] + 0.9

    def on_input(self, t, kind, arg):
        if kind != "down" or arg != "left":
            return
        self.clicks.append(t)
        free = [r for r in self.rings if r["hit"] is None and abs(t - r["T"]) < 0.22]
        if not free:
            self.results.append({"stage": "countin", "err": None, "verdict": "stray"})
            return
        r = min(free, key=lambda r: abs(t - r["T"]))
        r["hit"] = t
        err = (t - r["T"]) * 1000
        off = abs(err) / 1000 * self.speed / self.r0          # how far the ring was from the circle, in radii
        r["verdict"] = "perfect" if off <= 0.04 else "good" if off <= 0.11 else "miss"
        self.results.append({"stage": "countin", "err": err, "verdict": r["verdict"]})

    def close(self):
        for r in self.rings:
            if r["hit"] is None:
                self.results.append({"stage": "countin", "err": None, "verdict": "miss"})

    def dynamic(self, d, img, ox, oy, t):
        r0, cx, cy = self.r0, self.cx - ox, self.cy - oy
        grow = min(1.0, 0.6 + 0.4 * (t - self.t0) / 0.3)           # the circle grows in
        r = r0 * grow
        last = max((c for c in self.clicks if c <= t), default=None)
        if last is not None and t < last + 0.18:                   # the click's flash hides it all
            d.ellipse([cx - 1.7 * r0, cy - 1.7 * r0, cx + 1.7 * r0, cy + 1.7 * r0], fill=(252, 252, 252))
            return
        # The circle and the rings are thin, soft lines, as on recordings: about
        # 1.6 px wide on a 720p window (one or two pixels of it bright enough
        # to count as white). Drawn 3x bigger and scaled down for that.
        S = 3
        W, H = img.size
        lines = Image.new("L", (W * S, H * S), 0)
        dl = ImageDraw.Draw(lines)
        wl = max(2, int(round(self.h / 450 * S)))

        def circle(rad, width=wl):                                 # a line whose middle is `rad` from the center
            ro = rad * S + width / 2
            dl.ellipse([cx * S - ro, cy * S - ro, cx * S + ro, cy * S + ro], outline=255, width=int(width))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 40, 24))
        circle(r)
        bad = [r["hit"] for r in self.rings if r["hit"] is not None and r["hit"] <= t and r.get("verdict") != "perfect"]
        if self.glow and bad and t < max(bad) + 0.65:
            # ...and for a moment after, the circle is puffed up and glowing
            circle(1.08 * r0, 0.17 * r0 * S)
        d.ellipse([cx - 0.42 * r, cy - 0.5 * r, cx + 0.42 * r, cy + 0.45 * r], fill=WHITE)      # the scroom icon
        bx, by = cx + 0.71 * r, cy + 0.71 * r
        d.ellipse([bx - 0.26 * r, by - 0.26 * r, bx + 0.26 * r, by + 0.26 * r], fill=WHITE, outline=(0, 0, 0))
        for k, ring in enumerate(self.rings):
            x = cx + (k - 1.5) * 0.5 * r0
            done = ring["hit"] is not None and ring["hit"] <= t
            col = {"perfect": (250, 235, 130), "good": (150, 235, 180), "miss": (240, 110, 90)}.get(ring.get("verdict"))
            d.ellipse([x - 0.1 * r0, cy + 1.3 * r0, x + 0.1 * r0, cy + 1.55 * r0],
                      fill=col if done else (90, 100, 150), outline=(0, 0, 0))
            if t < ring["spawn"] or done:
                continue
            rr = self.r_start - self.speed * (t - ring["spawn"])
            if rr > 0.99 * r0:
                circle(rr)
        img.paste(WHITE, mask=lines.resize((W, H), Image.BOX))


# ------------------------------------------------------- shared: notes
def make_notes(rng, lanes, n, t_first, gap=(0.28, 0.75), holds=0.0, chords=0.0, hold_len=(0.35, 0.9)):
    """A random chart: list of dicts lane/T/hold (seconds)."""
    notes, t, last = [], t_first, {}
    for _ in range(n):
        lane = int(rng.integers(lanes))
        hold = float(rng.uniform(*hold_len)) if rng.random() < holds else 0.0
        t = max(t, last.get(lane, -9) + 0.3)
        notes.append({"lane": lane, "T": t, "hold": hold, "hit": None, "up": None})
        last[lane] = t + hold
        if lanes > 1 and rng.random() < chords:
            other = int((lane + 1 + rng.integers(lanes - 1)) % lanes)
            if t >= last.get(other, -9) + 0.3:
                notes.append({"lane": other, "T": t, "hold": 0.0, "hit": None, "up": None})
                last[other] = t
        t += hold + float(rng.uniform(*gap))
    return notes


class LaneGame(Stage):
    """Common rules for strings and keys: a key per lane, taps and holds."""
    keys = "asd"
    perfect_at = 0.0        # the game's "Perfect" moment, s after the note's middle is on the spot
    windows = (0.045, 0.10, 0.14)      # perfect / good / counts at all

    def judge_down(self, t, lane):
        self.presses.append((t, lane))
        free = [n for n in self.notes if n["lane"] == lane and n["hit"] is None and not n.get("ghost")
                and abs(t - n["T"] - self.perfect_at) <= self.windows[2]]
        if not free:
            self.results.append({"stage": self.name, "err": None, "verdict": "stray"})
            return
        n = min(free, key=lambda n: abs(t - n["T"]))
        n["hit"] = t
        err = (t - n["T"] - self.perfect_at) * 1000
        self.results.append({"stage": self.name, "err": err, "hold": n["hold"] > 0,
                             "verdict": "perfect" if abs(err) <= self.windows[0] * 1000 else
                             "good" if abs(err) <= self.windows[1] * 1000 else "miss"})

    def judge_up(self, t, lane):
        for n in self.notes:
            if n["lane"] == lane and n["hold"] > 0 and n["hit"] is not None and n["up"] is None:
                n["up"] = t
                err = (t - n["T"] - n["hold"]) * 1000
                self.results.append({"stage": self.name + "-release", "err": err,
                                     "verdict": "perfect" if -120 <= err <= 250 else "miss"})

    def on_input(self, t, kind, arg):
        if kind in ("kdown", "kup") and arg in self.keys:
            lane = self.keys.index(arg)
            (self.judge_down if kind == "kdown" else self.judge_up)(t, lane)
        elif kind == "kdown":
            self.results.append({"stage": self.name, "err": None, "verdict": "stray"})

    def close(self):
        for n in self.notes:
            if n["T"] > self.t_close:
                continue                                  # cut short before this one came
            if n["hit"] is None and not n.get("ghost"):
                self.results.append({"stage": self.name, "err": None, "verdict": "miss", "hold": n["hold"] > 0})
            elif n["hold"] > 0 and n["hit"] is not None and n["up"] is None and n["T"] + n["hold"] <= self.t_close:
                self.results.append({"stage": self.name + "-release", "err": None, "verdict": "miss"})

    def last_time(self):
        return max(n["T"] + n["hold"] for n in self.notes)

    def lit(self, lane, t):
        return any(l == lane and p <= t < p + 0.12 for p, l in self.presses)


# ------------------------------------------------------------- strings
class Strings(LaneGame):
    """3-lane note highway in perspective, keys A S D."""
    name = "strings"
    keys = "asd"
    perfect_at = 0.0
    windows = (0.05, 0.11, 0.15)
    COLORS = ((64, 214, 116), (236, 92, 104), (246, 214, 70))

    def __init__(self, w, h, bg, rng, speed=1.0, n=12, holds=0.2, chords=0.1, ghost=True, keys=None, notes=None,
                 cut_at=None):
        super().__init__(w, h, bg, rng)
        self.cut_at = cut_at                  # the minigame ends this long after it started, whatever is going on
        self.keys = (keys or "asd").lower()
        self.r = 0.0323 * h
        self.cy = 0.811 * h
        self.cxs = [w / 2 + k * 2.93 * self.r for k in (-1, 0, 1)]
        self.xv, self.yv_lines = w / 2, -0.012 * h       # where the lane lines meet
        # how the notes move, measured on recordings: 1/(y - yv) is a straight
        # line in time, with yv about 20 circle-radii above the circles
        self.yv = self.cy - 20.3 * self.r
        self.S0 = self.cy - self.yv
        self.c = 0.38 * speed * h / self.S0
        self.n, self.holds, self.chords, self.ghost = n, holds, chords, ghost
        self.notes = notes
        self.presses = []

    def start(self, t):
        self.t0 = t
        if self.notes is None:
            self.notes = make_notes(self.rng, 3, self.n, t + 2.2, holds=self.holds, chords=self.chords)
        else:
            self.notes = [dict(n, T=n["T"] + t, hit=None, up=None) for n in self.notes]
        self.t_close = self.last_time() + 1.0 if self.cut_at is None else t + self.cut_at
        if self.ghost:      # a leftover note that never arrives: it fades out up the lane as the highway closes
            self.notes.append({"lane": int(self.rng.integers(3)), "T": self.t_close + 0.55, "hold": 0.0,
                               "hit": None, "up": None, "ghost": True})

    CLOSE = 0.45           # how long the closing animation takes, s

    def end(self):
        return self.t_close + self.CLOSE

    def box(self):
        return (self.cxs[0] - 5 * self.r, self.cy - 10.5 * self.r, self.cxs[2] + 5 * self.r, self.cy + 3 * self.r)

    def pos(self, lane, t, T):
        x = 1 + self.c * (T - t)
        s = self.S0 / x if x > 0.25 else self.S0 * 4
        y = self.yv + s
        k = (y - self.yv_lines) / (self.cy - self.yv_lines)      # size and sideways: along the lane lines
        return self.xv + (self.cxs[lane] - self.xv) * k, y, k

    def static(self, arr, ox, oy):
        H, W = arr.shape[:2]
        # dark backdrop, fading out toward its edges
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        fx = np.clip(1.25 - np.abs(xx - W / 2) / (W / 2) * 1.25, 0, 1)
        fy = np.clip(1.6 - np.abs(yy - H * 0.55) / (H / 2) * 1.6, 0, 1)
        a = (0.86 * np.minimum(1, fx * 1.6) * np.minimum(1, fy * 1.6))[..., None]
        arr[:] = (arr.astype(np.float32) * (1 - a) + np.array([8, 10, 26], np.float32) * a).astype(np.uint8)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        r, cy = self.r, self.cy - oy
        lw = max(1, int(round(self.h / 720)))
        for k in (-1.5, -0.5, 0.5, 1.5):                 # lane lines, toward the vanishing point
            xb = self.w / 2 + k * 2.93 * r - ox
            xt = self.xv - ox + (xb - (self.xv - ox)) * 0.42
            d.line([(xt, cy - 9.8 * r), (xb + (xb - xt) * 0.12, cy + 1.6 * r)], fill=(235, 235, 235), width=lw)
        for i, cx in enumerate(self.cxs):
            d.ellipse([cx - ox - r, cy - r, cx - ox + r, cy + r], fill=(22, 26, 34), outline=self.COLORS[i],
                      width=max(2, int(round(0.075 * r))))
        arr[:] = np.asarray(img)

    def dynamic(self, d, img, ox, oy, t):
        r, cy = self.r, self.cy - oy
        for i, cx in enumerate(self.cxs):
            if self.lit(i, t):
                d.ellipse([cx - ox - 0.93 * r, cy - 0.93 * r, cx - ox + 0.93 * r, cy + 0.93 * r], fill=self.COLORS[i])
        for n in sorted(self.notes, key=lambda n: -n["T"]):
            lane, T, col = n["lane"], n["T"], self.COLORS[n["lane"]]
            hit = n["hit"] is not None and n["hit"] <= t
            if not hit and not n.get("ghost") and t > T + self.perfect_at + self.windows[2]:
                continue                                  # missed: it's gone
            if n["hold"] > 0:
                # the bar: from the head back to where the hold ends
                gone = hit and n["up"] is not None and n["up"] <= t
                if not gone and t < T + n["hold"] + 0.1:
                    pts = []
                    for f in np.linspace(0, 1, 10):
                        x, y, k = self.pos(lane, t, T + f * n["hold"])
                        if hit and y - oy > cy:
                            x, y, k = self.cxs[lane], self.cy, 1.0
                        if y > self.cy - 10 * r:
                            pts.append((x - ox, y - oy, k))
                    for (xa, ya, ka), (xb, yb, kb) in zip(pts, pts[1:]):
                        d.line([(xa, ya), (xb, yb)], fill=col, width=max(2, int(round(0.26 * r * (ka + kb) / 2))))
            if hit:
                continue
            x, y, k = self.pos(lane, t, T)
            if y < self.cy - 9.6 * r or y > self.cy + 1.4 * r:
                continue
            x, y = x - ox, y - oy
            wd, ht = 1.1 * r * k, 0.85 * r * k
            d.rounded_rectangle([x - wd / 2, y - ht / 2, x + wd / 2, y + ht / 2], radius=ht / 2, fill=col)
            c = max(1.5, 0.2 * r * k)
            d.ellipse([x - c, y - c, x + c, y + c], fill=(252, 252, 252))
        if t >= self.t_close:
            # closing, as on recordings: the whole highway (with the notes left
            # on it) swells by a fifth around a point under the circles for a
            # good quarter of a second, then collapses
            f = (t - self.t_close) / self.CLOSE
            k = 1 + 0.2 * min(1.0, f / 0.62) if f < 0.8 else max(0.3, 1.1 - 0.8 * (f - 0.8) / 0.2)
            W, H = img.size
            big = img.resize((max(2, int(W * k)), max(2, int(H * k))), Image.BILINEAR)
            x0, y0, x1, y1 = self._box()
            ax, ay = self.w / 2 - x0, self.cy + 1.8 * r - y0
            img.paste(Image.fromarray(self.bg[y0:y1, x0:x1]))
            img.paste(big, (int(round(ax - ax * k)), int(round(ay - ay * k))))


# ---------------------------------------------------------------- keys
class Keys(LaneGame):
    """Piano tiles: 4 columns, keys A S D F, tiles fall at a steady speed."""
    name = "keys"
    keys = "asdf"
    perfect_at = 0.0

    def __init__(self, w, h, bg, rng, speed=1.0, n=12, holds=0.2, chords=0.1, ghost=True, keys=None, notes=None,
                 cut_at=None):
        super().__init__(w, h, bg, rng)
        self.cut_at = cut_at
        self.keys = (keys or "asdf").lower()
        self.L = h / 3
        self.x0 = (w - self.L) / 2
        self.line_y = round(0.813 * h)
        self.top = self.line_y - 1.28 * self.L
        self.v = 0.22 * speed * h
        self.slot_y = self.line_y + 0.05 * self.L         # a tile is hit lying on its lettered key
        # the game judges by where the tile is: Perfect on the key, Good from
        # the row of boxes above the line down
        self.windows = (0.025 * self.L / self.v, 0.16 * self.L / self.v, 0.2 * self.L / self.v)
        self.n, self.holds, self.chords, self.ghost = n, holds, chords, ghost
        self.notes = notes
        self.presses = []

    def start(self, t):
        self.t0 = t
        if self.notes is None:
            self.notes = make_notes(self.rng, 4, self.n, t + 2.6, holds=self.holds, chords=self.chords)
        else:
            self.notes = [dict(n, T=n["T"] + t, hit=None, up=None) for n in self.notes]
        self.t_close = self.last_time() + 1.0 if self.cut_at is None else t + self.cut_at
        if self.ghost:
            self.notes.append({"lane": int(self.rng.integers(4)), "T": self.t_close + 0.5, "hold": 0.0,
                               "hit": None, "up": None, "ghost": True})

    CLOSE = 0.43           # how long the closing animation takes, s

    def end(self):
        return self.t_close + self.CLOSE

    def box(self):
        return self.x0 - 0.5 * self.L, self.top - 0.35 * self.L, self.x0 + 1.1 * self.L, self.line_y + 0.2 * self.L

    def static(self, arr, ox, oy):
        L, x0, ly, top = self.L, int(self.x0 - ox), int(self.line_y - oy), int(self.top - oy)
        _dim(arr[max(0, top):ly + int(0.15 * L), x0:x0 + int(L)], 0.62, (0, 8, 2))
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        lw = max(1, int(round(self.h / 720)))
        grey = (150, 156, 150)
        for k in range(5):
            x = x0 + k * L / 4
            d.line([(x, top), (x, ly + 0.15 * L)], fill=grey, width=lw)
        d.line([(x0, top), (x0 + L, top)], fill=grey, width=lw)
        for k in range(4):
            xa, xb = x0 + k * L / 4 + 3, x0 + (k + 1) * L / 4 - 3
            d.rounded_rectangle([xa, ly - 0.146 * L, xb, ly - 2], radius=4, outline=(105, 110, 105))
            d.rounded_rectangle([xa, ly + 4, xb, ly + 0.132 * L], radius=4, fill=(62, 58, 54), outline=(120, 126, 120))
        d.line([(x0, ly), (x0 + L, ly)], fill=(236, 238, 236), width=max(2, lw))
        arr[:] = np.asarray(img)

    def dynamic(self, d, img, ox, oy, t):
        L, x0, ly = self.L, self.x0 - ox, self.line_y - oy
        lane_w = L / 4
        wT, hT = 0.88 * lane_w, 0.12 * L
        shift = 0.0
        for i in range(4):
            if self.lit(i, t):
                d.rounded_rectangle([x0 + i * lane_w + 3, ly + 4, x0 + (i + 1) * lane_w - 3, ly + 0.132 * L], radius=4,
                                    fill=(136, 129, 122), outline=(170, 210, 170))
        clip_lo, clip_hi = self.top - oy, ly + 0.147 * L
        for n in self.notes:
            hit = n["hit"] is not None and n["hit"] <= t
            if not hit and not n.get("ghost") and t > n["T"] + self.windows[2] + 0.1:
                continue                                  # missed: it's gone
            xc = x0 + (n["lane"] + 0.5) * lane_w
            y = self.slot_y - oy - self.v * (n["T"] - t) + shift
            if n["hold"] > 0:
                ya, yb = y - hT / 2 - self.v * n["hold"], y - hT / 2
                if hit:
                    yb = min(yb, self.slot_y - oy)
                gone = hit and n["up"] is not None and n["up"] <= t
                ya, yb = max(ya, clip_lo), min(yb, clip_hi)
                if not gone and yb - ya > 1:
                    d.rectangle([xc - 0.29 * lane_w, ya, xc + 0.29 * lane_w, yb], fill=(126, 121, 116))
            if hit:
                continue
            ya, yb = max(y - hT / 2, clip_lo), min(y + hT / 2, clip_hi)
            if yb - ya < 2:
                continue
            d.rounded_rectangle([xc - wT / 2, ya, xc + wT / 2, yb], radius=min(6, (yb - ya) / 2), fill=(240, 242, 248))
            if yb - ya > 8:
                d.rectangle([xc - wT / 2 + 2, yb - 0.3 * (yb - ya), xc + wT / 2 - 2, yb - 2], fill=(214, 218, 226))
        if t >= self.t_close:
            # closing, as on recordings: the whole board (with the tiles left on
            # it) swells by an eighth around its bottom middle for a quarter of
            # a second, then collapses. Seen from where the board used to be,
            # the tiles drift up and then shoot down.
            f = (t - self.t_close) / self.CLOSE
            k = 1 + 0.13 * min(1.0, f / 0.55) if f < 0.78 else max(0.3, 1.13 - 0.83 * (f - 0.78) / 0.22)
            W, H = img.size
            big = img.resize((max(2, int(W * k)), max(2, int(H * k))), Image.BILINEAR)
            bx0, by0, bx1, by1 = self._box()
            ax, ay = self.x0 + L / 2 - bx0, self.line_y + 0.16 * L - by0
            img.paste(Image.fromarray(self.bg[by0:by1, bx0:bx1]))
            img.paste(big, (int(round(ax - ax * k)), int(round(ay - ay * k))))


# ---------------------------------------------------------- percussion
class Perc(Stage):
    """One lane, notes slide left to the scroom: white = left click, blue =
    right click, big two-color = both."""
    name = "percussion"
    windows = (0.045, 0.10, 0.14)

    def __init__(self, w, h, bg, rng, speed=1.0, n=12, big=0.2, notes=None):
        super().__init__(w, h, bg, rng)
        self.L = 0.69 * h
        self.x0 = round((w - self.L) / 2)
        self.y = round(0.81 * h)
        self.gap = 0.17 * self.L
        self.tx = self.x0 + 0.0864 * self.L
        self.v = 0.424 * speed * h
        self.n, self.big = n, big
        self.notes = notes
        self.half = []        # big notes waiting for their second button

    def start(self, t):
        self.t0 = t
        if self.notes is None:
            self.notes, tt = [], t + 2.4
            for _ in range(self.n):
                kind = "B" if self.rng.random() < self.big else ("L" if self.rng.random() < 0.5 else "R")
                self.notes.append({"kind": kind, "T": tt, "hit": None, "first": None})
                tt += float(self.rng.uniform(0.3, 0.8))
        else:
            self.notes = [dict(n, T=n["T"] + t, hit=None, first=None) for n in self.notes]

    def end(self):
        return max(n["T"] for n in self.notes) + 1.2

    def box(self):
        return self.x0 - 0.06 * self.L, self.y - 0.9 * self.gap, self.x0 + 1.06 * self.L, self.y + 0.9 * self.gap

    def on_input(self, t, kind, arg):
        if kind != "down":
            return
        btn = "L" if arg == "left" else "R"
        cands = [n for n in self.notes if n["hit"] is None and abs(t - n["T"]) <= self.windows[2]]
        if not cands:
            self.results.append({"stage": self.name, "err": None, "verdict": "stray"})
            return
        n = min(cands, key=lambda n: abs(t - n["T"]))
        if n["kind"] == "B":
            if n["first"] is None or n["first"][1] == btn:
                n["first"] = (t, btn)
                return
            if t - n["first"][0] > 0.08:
                n["first"] = (t, btn)
                return
        elif n["kind"] != btn:
            self.results.append({"stage": self.name, "err": None, "verdict": "wrong"})
            return
        n["hit"] = t
        err = (t - n["T"]) * 1000
        self.results.append({"stage": self.name, "err": err, "big": n["kind"] == "B",
                             "verdict": "perfect" if abs(err) <= self.windows[0] * 1000 else
                             "good" if abs(err) <= self.windows[1] * 1000 else "miss"})

    def close(self):
        for n in self.notes:
            if n["hit"] is None:
                self.results.append({"stage": self.name, "err": None, "verdict": "miss", "big": n["kind"] == "B"})

    def static(self, arr, ox, oy):
        L, x0, y, g = self.L, int(self.x0 - ox), int(self.y - oy), self.gap
        ya, yb = int(y - g / 2), int(y + g / 2)
        _dim(arr[ya - 4:yb + 5, max(0, x0 - int(0.02 * L)):x0 + int(1.02 * L)], 0.55, (10, 22, 66))
        _dim(arr[ya:yb, x0:x0 + int(0.218 * L)], 0.25, (150, 160, 190))
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        lw = max(2, int(round(self.h / 480)))
        d.line([(x0, ya), (x0 + L, ya)], fill=(240, 240, 244), width=lw)
        d.line([(x0, yb), (x0 + L, yb)], fill=(240, 240, 244), width=lw)
        d.line([(x0 + 0.218 * L, ya), (x0 + 0.218 * L, yb)], fill=(226, 226, 232), width=max(1, lw - 1))
        d.line([(x0 + 0.218 * L, y), (x0 + L, y)], fill=(70, 82, 120), width=1)
        arr[:] = np.asarray(img)

    def dynamic(self, d, img, ox, oy, t):
        L, y = self.L, self.y - oy
        right = self.x0 + L - ox
        for n in sorted(self.notes, key=lambda n: -n["T"]):
            if n["hit"] is not None and n["hit"] <= t:
                continue
            x = self.tx + self.v * (n["T"] - t) - ox
            a = (0.06 if n["kind"] == "B" else 0.047) * L
            if x - a > right or x < self.tx - ox - 0.03 * L:
                continue
            white, blue = (228, 230, 236), (62, 132, 250)
            if n["kind"] == "B":
                d.polygon([(x - a, y), (x, y - a), (x, y + a)], fill=white)
                d.polygon([(x + a, y), (x, y - a), (x, y + a)], fill=blue)
            else:
                _diamond(d, x, y, a, white if n["kind"] == "L" else blue)
                d.rounded_rectangle([x - 0.012 * L, y - 0.02 * L, x + 0.012 * L, y + 0.02 * L], radius=3,
                                    fill=(120, 124, 136) if n["kind"] == "L" else (30, 70, 170))
        x = self.tx - ox                                   # the scroom's diamond sits on top
        _diamond(d, x, y, 0.062 * L, (34, 44, 70), outline=(190, 196, 210))
        d.ellipse([x - 0.02 * L, y - 0.028 * L, x + 0.02 * L, y + 0.028 * L], fill=WHITE)


# ---------------------------------------------------------------- wind
class Wind(Stage):
    """A 5-line staff: steer the scroom with the mouse's height, hold left
    click while a note's bar passes. Some notes slide from one line to another."""
    name = "wind"
    LINES = (0.142, 0.321, 0.5, 0.679, 0.858)

    def __init__(self, w, h, bg, rng, speed=1.0, n=8, slides=0.25, follow=0.07, gain=1.0, offset=0.0, notes=None):
        super().__init__(w, h, bg, rng)
        self.L = round(0.697 * h)
        self.D = 0.328 * self.L
        self.top = round(0.64 * h)
        self.x0 = round((w - self.L) / 2)
        self.sx = self.x0 + 0.084 * self.L
        self.v = 0.43 * speed * self.L
        self.lines = [self.top + f * self.D for f in self.LINES]
        self.n, self.slides = n, slides
        self.follow, self.gain, self.offset = follow, gain, offset
        self.notes = notes
        self.hist_t, self.hist_y = [], []
        self.y_s = self.lines[2]
        self.cursor_y = self.lines[2]
        self.down = []          # [t_down, t_up or None]

    def start(self, t):
        self.t0 = t
        self._t = t
        if self.notes is None:
            self.notes, tt, a = [], t + 2.6, 2
            for _ in range(self.n):
                a = int(np.clip(a + self.rng.integers(-2, 3), 0, 4))
                b = a
                if self.rng.random() < self.slides:
                    b = int(np.clip(a + self.rng.choice([-1, 1]), 0, 4))
                dur = float(self.rng.uniform(0.35, 1.3))
                self.notes.append({"a": a, "b": b, "T": tt, "dur": dur})
                tt += dur + float(self.rng.uniform(0.18, 0.7))
                a = b
        else:
            self.notes = [dict(n, T=n["T"] + t) for n in self.notes]

    def end(self):
        return max(n["T"] + n["dur"] for n in self.notes) + 1.2

    def box(self):
        return self.x0 - 0.06 * self.L, self.top - 0.3 * self.D, self.x0 + 1.06 * self.L, self.top + 1.3 * self.D

    def on_input(self, t, kind, arg):
        if kind == "move":
            self.tick(t)
            self.cursor_y = arg[1]
        elif kind == "down" and arg == "left":
            self.down.append([t, None])
        elif kind == "up" and arg == "left" and self.down and self.down[-1][1] is None:
            self.down[-1][1] = t

    def tick(self, t):
        """The scroom eases toward the cursor's height."""
        mid = self.lines[2]
        while self._t < t:
            dt = min(1 / 240, t - self._t)
            want = float(np.clip(mid + self.gain * (self.cursor_y - mid) + self.offset,
                                 self.top + 0.05 * self.D, self.top + 0.95 * self.D))
            self.y_s += (want - self.y_s) * (1 - math.exp(-dt / self.follow))
            self._t += dt
            self.hist_t.append(self._t)
            self.hist_y.append(self.y_s)

    def scroom(self, t):
        if not self.hist_t:
            return self.lines[2]
        return float(np.interp(t, self.hist_t, self.hist_y))

    def held(self, t):
        return any(a <= t and (b is None or t < b) for a, b in self.down)

    def bar_y(self, n, t):
        f = float(np.clip((t - n["T"]) / n["dur"], 0, 1))
        return self.lines[n["a"]] + f * (self.lines[n["b"]] - self.lines[n["a"]])

    def close(self):
        tol = 0.5 * (self.lines[1] - self.lines[0])
        for n in self.notes:
            ts = np.arange(n["T"], n["T"] + n["dur"], 0.005)
            ok = [self.held(t) and abs(self.scroom(t) - self.bar_y(n, t)) <= tol for t in ts]
            on = [self.held(t) for t in ts]
            cover = float(np.mean(ok))
            first = next((a for a, b in self.down if n["T"] - 0.3 < a < n["T"] + n["dur"]), None)
            self.results.append({"stage": self.name, "err": None if first is None else (first - n["T"]) * 1000,
                                 "cover": cover, "held": float(np.mean(on)), "slide": n["a"] != n["b"],
                                 "verdict": "perfect" if cover >= 0.9 else "good" if cover >= 0.7 else "miss"})

    def static(self, arr, ox, oy):
        L, x0, top, D = self.L, int(self.x0 - ox), int(self.top - oy), self.D
        _dim(arr[top - 3:top + int(D) + 4, max(0, x0 - 4):x0 + L + 4], 0.72, (4, 8, 30))
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        lw = max(2, int(round(self.h / 480)))
        d.line([(x0, top), (x0 + L, top)], fill=(242, 242, 246), width=lw)
        d.line([(x0, top + D), (x0 + L, top + D)], fill=(242, 242, 246), width=lw)
        sx = self.sx - ox
        for y in self.lines:
            d.line([(sx, y - oy), (x0 + L, y - oy)], fill=(96, 102, 120), width=max(1, lw - 1))
        d.line([(sx, self.lines[0] - oy), (sx, self.lines[4] - oy)], fill=(96, 102, 120), width=max(1, lw - 1))
        arr[:] = np.asarray(img)

    def dynamic(self, d, img, ox, oy, t):
        L, D = self.L, self.D
        left, right = self.x0 - ox, self.x0 + L - ox
        th = max(3, int(round(0.058 * D)))
        for n in self.notes:
            hx = self.sx + self.v * (n["T"] - t) - ox
            tx = hx + self.v * n["dur"]
            ya, yb = self.lines[n["a"]] - oy, self.lines[n["b"]] - oy
            if tx < left or hx > right + 0.04 * L:
                continue
            bx = hx + 0.024 * L                       # the bar starts just behind the head
            xa, xb = max(bx, left), min(tx, right)
            if xb > xa:
                span = max(tx - hx, 1e-6)
                fa, fb = (xa - hx) / span, (xb - hx) / span
                d.line([(xa, ya + fa * (yb - ya)), (xb, ya + fb * (yb - ya))], fill=(186, 240, 216), width=th)
                if tx <= right:
                    d.ellipse([tx - th / 2, yb - th / 2, tx + th / 2, yb + th / 2], fill=(186, 240, 216))
            if left <= hx <= right:
                _diamond(d, hx, ya, 0.026 * L, (222, 222, 228), outline=(120, 124, 132))
        sx, sy = self.sx - ox, self.scroom(t) - oy
        _diamond(d, sx, sy, 0.05 * L, (96, 100, 116), outline=(170, 174, 186))
        d.ellipse([sx - 0.016 * L, sy - 0.08 * D, sx + 0.016 * L, sy + 0.08 * D], fill=WHITE)


STAGES = {"strings": Strings, "keys": Keys, "percussion": Perc, "wind": Wind}


# ------------------------------------------------------------- session
class Session:
    """A whole fishing trip: cast -> ring -> count-in -> minigame, `rounds`
    times. `direct=True` skips cast/ring/count-in (the minigame is just up)."""

    def __init__(self, w, h, scene="night", minigame="strings", rounds=1, seed=0, direct=False,
                 count_in=True, opts=None, ring_opts=None, countin_opts=None):
        self.w, self.h = w, h
        self.rng = np.random.default_rng(seed)
        self.bg = background(scene, w, h, seed)
        self.minigame, self.rounds, self.direct, self.count_in = minigame, rounds, direct, count_in
        self.opts, self.ring_opts, self.countin_opts = opts or {}, ring_opts or {}, countin_opts or {}
        self.stages = []            # in order; the last one is the current one
        self.results = []
        self.done_rounds = 0
        self.idle_since = 0.0
        self._q = []
        self._seq = 0
        self.inputs = []            # (time the game got it, kind, arg)
        self.t = 0.0
        self.t_done = None
        if direct:
            self._add(self._minigame(), 0.6)

    def _minigame(self):
        name = self.minigame if isinstance(self.minigame, str) else self.minigame[self.done_rounds % len(self.minigame)]
        return STAGES[name](self.w, self.h, self.bg, self.rng, **self.opts)

    def _add(self, stage, t):
        stage.start(t)
        self.stages.append(stage)

    def queue(self, t, kind, arg):
        self._seq += 1
        self.inputs.append((t, kind, arg))
        heapq.heappush(self._q, (t, self._seq, kind, arg))

    def current(self):
        return self.stages[-1] if self.stages and not getattr(self.stages[-1], "closed", False) else None

    def advance(self, t):
        while True:
            t_ev = self._q[0][0] if self._q else math.inf
            cur = self.current()
            t_end = cur.end() if cur is not None else math.inf
            if min(t_ev, t_end) > t:
                break
            if t_end <= t_ev:                      # the current stage is over: what's next?
                cur.tick(t_end)
                cur.close()
                cur.closed = True
                self.results += cur.results
                if cur.name == "ring":
                    if cur.clicked is None or cur.results[-1]["verdict"] == "miss":
                        self.idle_since = t_end                 # missed: cast again
                    elif self.count_in:
                        self._add(CountIn(self.w, self.h, self.bg, self.rng, **self.countin_opts), t_end + 0.5)
                    else:
                        self._add(self._minigame(), t_end + 0.8)
                elif cur.name == "countin":
                    self._add(self._minigame(), t_end + 0.3)
                else:
                    self.done_rounds += 1
                    self.idle_since = t_end
                    if self.done_rounds >= self.rounds:
                        self.t_done = t_end + 1.5
                    elif self.direct:
                        self._add(self._minigame(), t_end + 4.0)
                continue
            _, _, kind, arg = heapq.heappop(self._q)
            if cur is not None and t_ev >= cur.t0:
                cur.tick(t_ev)
                cur.on_input(t_ev, kind, arg)
            elif cur is None and kind == "down" and arg == "left" and not self.direct and self.t_done is None:
                self._add(Ring(self.w, self.h, self.bg, self.rng, **self.ring_opts), t_ev + 0.7)   # a cast
        cur = self.current()
        if cur is not None and t >= cur.t0:
            cur.tick(t)
        self.t = t

    def finished(self, t):
        return self.t_done is not None and t >= self.t_done

    def frame(self, t):
        frame = self.bg.copy()
        for st in self.stages[-2:]:
            if st.visible(t):
                st.draw(frame, t)
        return frame
