"""Screen detectors for the scroom catching minigames - any window size.

Nothing is looked up at fixed pixel positions. Each panel is *found* on screen
by its white track lines, and those lines also give the scale:

  frequency panel  two track lines 1224 long, 59.5 apart (+ the pin above them)
  reel bar         top + bottom lines 955 long, 185 apart (+ the gray box)

(all "reference" numbers were measured on a 3440x1360 window). Detectors
return x positions converted back to reference pixels, so the controller's
physics numbers work at every size.

Each Finder remembers where its panel was last seen and asks the bot to grab
just that area next time; if the panel moves or disappears it searches the
whole lower part of the window again.
"""
from dataclasses import dataclass

import numpy as np

REF_H = 1360

# frequency panel, reference px
FREQ_X0, FREQ_LEN = 1108, 1224        # track line start, length
FREQ_SPACING = 59.5                   # upper -> lower track line
FREQ_PIN_DY = -85.5                   # pin ball row, relative to the upper track line
FREQ_DOT_DY = 29.5                    # target dot row, relative to the upper track line
FREQ_WAVE_DY = 131                    # line above the track (top of the wave area)

# reel bar, reference px
REEL_X0, REEL_LEN = 1243, 955         # top/bottom line start, length
REEL_SPACING = 185                    # top line -> bottom line
REEL_EDGE_DY = (40, 176)              # rows the box's side edges span (from top line)
REEL_FISH_DY = (88.5, 158.5)          # rows of the scroom icon
REEL_DOT_DY = 45.5                    # row of the "< o >" dot
REEL_BOX_W = 323                      # gray box width
REEL_PROG_DY = 33.5                   # progress bar row, below the bottom line
REEL_PROG_X = (1345, 2095)

RING_R = 100                          # cast ring mid radius

# latest measured UI scale (vertical), shared with the ring finder
SCALE = {"s": None}


def _runs(mask):
    """Return (start, end) index pairs of consecutive True runs in a 1D mask."""
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _clamp_region(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


@dataclass
class HLine:
    y: float     # row (image-local)
    x0: int
    x1: int
    rows: int = 1   # thickness

    @property
    def length(self):
        return self.x1 - self.x0


def line_ridge(v, gap=4, contrast=30):
    """v: 2D brightness. True where a pixel is a thin bright line: bright AND
    clearly brighter than the pixels `gap` rows above and below. Bright
    backgrounds (lit grass, snow...) are bright everywhere, so they don't count."""
    v = v.astype(np.int16)
    pad = np.pad(v, ((gap, gap), (0, 0)), mode="edge")
    around = np.maximum(pad[:-2 * gap], pad[2 * gap:])
    return v - around >= contrast


def find_hlines(img, thresh, min_len, step=2, channel="min", ridge=False):
    """Long, bright, horizontal lines (allowing small gaps, e.g. where the pin
    crosses a track line)."""
    sub = img[:, ::step]
    # channel="min": white lines only. channel="max": bright in any color (rarer
    # scrooms tint the reel bar purple etc.)
    if channel == "max":
        # any color, but it has to stand out from what's right above/below it
        v = sub.max(axis=2)
        bright = (v >= thresh) & line_ridge(v)
    else:
        bright = sub.min(axis=2) >= thresh
        if ridge:   # must also stand out from above/below (not just bright glare/snow)
            bright &= line_ridge(sub.max(axis=2))
    need = min_len / step
    cand = np.flatnonzero(bright.sum(axis=1) >= need * 0.8)
    rows = []
    gap = max(3, int(40 / step))
    for y in cand:
        spans = []
        for a, b in _runs(bright[y]):
            if spans and a - spans[-1][1] <= gap:
                spans[-1][1] = b
            else:
                spans.append([a, b])
        a, b = max(spans, key=lambda s: s[1] - s[0])
        if b - a >= need and bright[y, a:b].mean() >= 0.85:
            rows.append((y, a * step, b * step))
    lines = []
    for y, a, b in rows:
        if lines and y - lines[-1][-1][0] <= 2 and abs(a - lines[-1][-1][1]) < 12 and abs(b - lines[-1][-1][2]) < 12:
            lines[-1].append((y, a, b))
        else:
            lines.append([(y, a, b)])
    return [HLine(float(np.mean([r[0] for r in g])), int(np.median([r[1] for r in g])),
                  int(np.median([r[2] for r in g])), len(g)) for g in lines]


def _pairs(lines, lo, hi, overlap=None):
    """Pairs of lines whose spacing/length ratio is in [lo, hi]. By default both
    ends must match; with `overlap`, the shorter line only has to cover that
    fraction of the longer one (a line can fade out at one end)."""
    out = []
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            top, bot = (a, b) if a.y < b.y else (b, a)
            L = (top.length + bot.length) / 2
            if L <= 0:
                continue
            if overlap is None:
                if abs(top.x0 - bot.x0) > 0.03 * L or abs(top.x1 - bot.x1) > 0.03 * L:
                    continue
            else:
                longer = max(top.length, bot.length)
                ov = min(top.x1, bot.x1) - max(top.x0, bot.x0)
                if ov < overlap * longer:
                    continue
                L = longer
            if lo <= (bot.y - top.y) / L <= hi:
                out.append((top, bot))
    return out


class Finder:
    """Common caching: tight region when we know where the panel is."""

    def __init__(self):
        self.geo = None     # panel geometry in window-client coordinates
        self.misses = 0
        self.reads = 0
        self.refreshing = False

    def search_area(self, w, h):
        return _clamp_region(0.03 * w, 0.45 * h, 0.97 * w, h, w, h)

    def tight_area(self, w, h):
        raise NotImplementedError


    REFRESH_EVERY = 30   # re-find the panel from scratch this often (reads), so a
                         # position locked while it was still sliding in can't stick

    def region(self, w, h):
        if self.geo is not None and self.reads >= self.REFRESH_EVERY:
            self.refreshing, self.reads = True, 0
        if self.refreshing or not self.geo:
            return self.search_area(w, h)
        return self.tight_area(w, h)

    def find(self, img, ox, oy, w, h):
        if self.refreshing:
            # re-find from scratch, but never throw away a working position
            # just because the search failed (bright backgrounds etc.)
            self.refreshing = False
            geo = self.locate(img, ox, oy, w, h)
            if geo is not None:
                old, self.geo = self.geo, geo
                res = self.read(img, ox, oy)
                if res is not None:
                    return res
                self.geo = old
            if self.geo is not None:
                res = self.read(img, ox, oy)
                if res is not None:
                    return res
        if self.geo is not None:
            res = self.read(img, ox, oy)
            if res is not None:
                self.misses = 0
                self.reads += 1
                return res
            self.misses += 1
            if self.misses >= 2:      # moved or gone - search again next time
                self.geo = None
                self.misses = 0
            return None
        geo = self.locate(img, ox, oy, w, h)
        if geo is None:
            return None
        self.geo = geo
        res = self.read(img, ox, oy)
        if res is None:
            self.geo = None
        return res


# ------------------------------------------------------------ frequency phase

class FreqFinder(Finder):
    """Returns (pin_x, dot_x) in reference px (dot_x may be None)."""

    def __init__(self):
        super().__init__()
        self.last_pin = None
        self.pin_jumps = 0

    def locate(self, img, ox, oy, w, h):
        lines = find_hlines(img, 225, 0.2 * w, ridge=True)
        for top, bot in sorted(_pairs(lines, 0.025, 0.09), key=lambda p: -p[1].y):
            L = (top.length + bot.length) / 2
            sy = (bot.y - top.y) / FREQ_SPACING
            sx = L / FREQ_LEN
            if not 0.5 <= sx / sy <= 2:
                continue
            # the panel also has a line above the track (top of the wave
            # area); light beams / streaks don't
            want = top.y - FREQ_WAVE_DY * sy
            if not any(abs(l.y - want) < 8 * sy and abs(l.x0 - top.x0) < 0.03 * L and abs(l.x1 - top.x1) < 0.03 * L
                       for l in lines):
                continue
            return {"x0": ox + (top.x0 + bot.x0) / 2, "len": L, "top": oy + top.y, "bot": oy + bot.y}
        return None

    def tight_area(self, w, h):
        g = self.geo
        sy = (g["bot"] - g["top"]) / FREQ_SPACING
        sx = g["len"] / FREQ_LEN
        return _clamp_region(g["x0"] - 40 * sx, g["top"] + (FREQ_PIN_DY - 30) * sy,
                             g["x0"] + g["len"] + 40 * sx, g["bot"] + 20 * sy, w, h)

    def read(self, img, ox, oy):
        g = self.geo
        sx = g["len"] / FREQ_LEN
        sy = (g["bot"] - g["top"]) / FREQ_SPACING
        H, W = img.shape[:2]
        xa = int(round(g["x0"] - ox))
        xb = int(round(g["x0"] + g["len"] - ox))
        if xa < 0 or xb > W:
            return None
        # the pin can sit a little past either end of the track
        pa, pb = max(0, xa - int(40 * sx)), min(W, xb + int(40 * sx))

        def rows(y_abs, half, a=xa, b=xb):
            y = int(round(y_abs - oy))
            if y - half < 0 or y + half >= H:
                return None
            return img[y - half:y + half + 1, a:b]

        # both track lines still where we left them?
        for y_abs in (g["top"], g["bot"]):
            band = rows(y_abs, max(1, int(round(2 * sy))))
            if band is None or (band.max(axis=0).min(axis=1) >= 225).mean() < 0.8:
                return None
        SCALE["s"] = sy

        def to_ref(x):   # x is relative to pa
            return FREQ_X0 + (x + pa - xa) / sx

        pin = rows(g["top"] + FREQ_PIN_DY * sy, max(1, int(round(sy))), pa, pb)
        if pin is None:
            return None
        # pin ball: a small white blob. Snowflakes/sparkles look the same, so a
        # candidate only counts if the pin's needle hangs below it.
        cands = [r for r in _runs(pin.min(axis=0).min(axis=1) >= 245)
                 if max(5, 12 * sx) <= r[1] - r[0] <= 70 * sx]
        if not cands:
            return None
        ya = int(round(g["top"] + (FREQ_PIN_DY + 25) * sy - oy))
        yb2 = int(round(g["top"] - 8 * sy - oy))
        side = max(4, int(round(8 * sx)))

        def has_needle(r):
            c0 = int((r[0] + r[1]) / 2) + pa
            if yb2 - ya < 3 or ya < 0 or c0 - side < 0 or c0 + side >= W:
                return True          # can't check here - don't reject
            col = img[ya:yb2, c0 - side:c0 + side + 1].max(axis=2).astype(np.int16)
            center = col[:, side - 1:side + 2].mean(axis=1)
            sides = (col[:, 0] + col[:, -1]) / 2
            return (np.abs(center - sides) >= 20).mean() >= 0.5

        good = [r for r in cands if has_needle(r)] or (cands if len(cands) == 1 else [])
        if not good:
            return None
        if self.last_pin is not None and len(good) > 1:
            a, b = min(good, key=lambda r: abs(to_ref((r[0] + r[1]) / 2) - self.last_pin))
        else:
            a, b = max(good, key=lambda r: r[1] - r[0])
        pin_x = to_ref((a + b) / 2)
        # the pin can't teleport: a far jump from the last reading is a sparkle
        # (give up on the old position after a few such frames in a row)
        if self.last_pin is not None and abs(pin_x - self.last_pin) > 300 and self.pin_jumps < 4:
            self.pin_jumps += 1
            return None
        self.pin_jumps = 0
        self.last_pin = pin_x

        dot = rows(g["top"] + FREQ_DOT_DY * sy, max(1, int(round(sy))), pa, pb)
        dot_runs = [] if dot is None else \
            [r for r in _runs(dot.min(axis=0).min(axis=1) >= 245) if max(5, 12 * sx) <= r[1] - r[0] <= 45 * sx]
        if not dot_runs:
            return pin_x, None
        a, b = max(dot_runs, key=lambda r: r[1] - r[0])
        return pin_x, to_ref((a + b) / 2)


# ----------------------------------------------------------------- reel phase

class ReelFinder(Finder):
    """Returns (box_left, box_right, fish_x, dot_x, progress) in reference px.
    Any of the values may be None except that a None box means "not sure"."""

    def __init__(self):
        super().__init__()
        self.last_box_w = None

    def locate(self, img, ox, oy, w, h):
        lines = find_hlines(img, 150, 0.15 * w, channel="max")
        # track lines are thin (~2-4 px); the progress bar under the reel is a
        # thick bright bar once it fills up - never pair with that
        thin_max = max(6, 0.008 * h)
        thick = [l for l in lines if l.rows > thin_max]
        # ...including thin slivers of that bar's edges right next to it
        lines = [l for l in lines if l.rows <= thin_max and not any(
            abs(l.y - t.y) <= t.rows / 2 + 12 and min(l.x1, t.x1) - max(l.x0, t.x0) > 0.5 * l.length
            for t in thick)]
        # the lines can fade out at one end, so the pair only has to mostly
        # overlap; take the bar's extent from the longer line
        for top, bot in sorted(_pairs(lines, 0.13, 0.4, overlap=0.7), key=lambda p: -p[1].y):
            full = top if top.length >= bot.length else bot
            L = full.length
            if not 0.5 <= (L / REEL_LEN) / ((bot.y - top.y) / REEL_SPACING) <= 2:
                continue
            geo = {"x0": ox + full.x0, "len": L, "top": oy + top.y, "bot": oy + bot.y, "full": oy + full.y}
            saved, self.geo = self.geo, geo
            res = self.read(img, ox, oy)
            self.geo = saved
            if res is not None and res[0] is not None:   # the gray box must be there
                return geo
        return None

    def tight_area(self, w, h):
        g = self.geo
        sx = g["len"] / REEL_LEN
        sy = (g["bot"] - g["top"]) / REEL_SPACING
        return _clamp_region(g["x0"] - 30 * sx, g["top"] - 15 * sy,
                             g["x0"] + g["len"] + 30 * sx, g["bot"] + (REEL_PROG_DY + 15) * sy, w, h)

    def read(self, img, ox, oy):
        g = self.geo
        sx = g["len"] / REEL_LEN
        sy = (g["bot"] - g["top"]) / REEL_SPACING
        H, W = img.shape[:2]
        xa = int(round(g["x0"] - ox))
        xb = int(round(g["x0"] + g["len"] - ox))
        if xa < 0 or xb > W:
            return None

        def y_of(dy):
            return int(round(g["top"] + dy - oy))

        # bottom line still there?
        yb = int(round(g["bot"] - oy))
        hb = max(1, int(round(2 * sy)))
        if yb - hb < 0 or yb + hb >= H:
            return None
        def ridge_row(y_c):
            """per-column: is there a thin bright line at row y_c (+-hb)?"""
            lo, hi = y_c - hb - 5, y_c + hb + 6
            if lo < 0 or hi > H:
                return None
            v = img[lo:hi].max(axis=2)
            m = (v >= 140) & line_ridge(v)
            return m[5:-5].any(axis=0)

        bottom = ridge_row(yb)
        if bottom is None or bottom[xa:xb].mean() < 0.75:
            return None
        # the full-length line must start and end exactly where we think
        # (catches a position locked while the bar was sliding in or out)
        yt = int(round(g.get("full", g["top"]) - oy))
        top = ridge_row(yt)                                         # whole row width
        if top is None:
            return None
        if top[xa:xb].mean() < 0.85:
            return None
        # coarse on purpose: this is for big misalignments (a bar locked while
        # sliding in is off by ~75 ref px), not a few pixels of measuring error
        e0, e1 = max(4, int(round(15 * sx))), max(10, int(round(45 * sx)))
        # a position locked in the wrong place makes the line stick out on ONE
        # side; a bright background is bright on both sides, so that's fine
        past = [seg.mean() > 0.5 for seg in (top[max(0, xa - e1):max(0, xa - e0)], top[xb + e0:xb + e1])
                if seg.size]
        if len(past) == 2 and past[0] != past[1]:
            return None
        if top[xa:xa + e1].mean() < 0.4 or top[max(xa, xb - e1):xb].mean() < 0.4:
            return None                            # ...or stops well short of it

        def to_ref(x):
            return REEL_X0 + x / sx

        # gray box: its two side edges are thin light vertical lines
        y0, y1 = y_of(REEL_EDGE_DY[0] * sy), y_of(REEL_EDGE_DY[1] * sy)
        lum = img[max(y0, 0):min(y1, H), xa:xb].astype(np.int16).sum(axis=2)
        box = None
        if lum.shape[0] > 4 and lum.shape[1] > 20:
            d = max(2, int(round(5 * sx)))
            mid = lum[:, d:-d]
            ridge = (mid > lum[:, :-2 * d] + 45) & (mid > lum[:, 2 * d:] + 45)
            score = np.zeros(lum.shape[1])
            score[d:-d] = ridge.mean(axis=0)
            score = np.maximum.reduce([np.roll(score, k) for k in (-1, 0, 1)])
            # the box's width depends on the player's gear, so don't assume it:
            # take the two strongest full-height edges a sensible distance apart
            peaks = []
            for a, b in _runs(score >= 0.5):
                i = a + int(score[a:b].argmax())
                peaks.append((i, float(score[i])))
            n = len(score)
            best = 0.0
            for i, (pa_, sa) in enumerate(peaks):
                for pb_, sb in peaks[i + 1:]:
                    width = pb_ - pa_
                    if not 0.12 * n <= width <= 0.75 * n:
                        continue
                    s_ = sa + sb
                    if self.last_box_w:   # small preference for the size we saw last
                        s_ -= 0.3 * abs(width / sx - self.last_box_w) / self.last_box_w
                    if s_ > best:
                        best, box = s_, (pa_, pb_)
            if best < 1.1:
                box = None
            if box is not None:
                self.last_box_w = (box[1] - box[0]) / sx
        if box is not None:
            box = (to_ref(box[0]), to_ref(box[1]))
            SCALE["s"] = sy

        # the little "< o >" dot above the icon (always white)
        y = y_of(REEL_DOT_DY * sy)
        dot_px = None
        if 1 <= y < H - 1:
            dot_row = img[y - 1:y + 2, xa:xb].min(axis=0).min(axis=1) >= 240
            runs = [r for r in _runs(dot_row) if max(5, 14 * sx) <= (r[1] - r[0]) <= 45 * sx]
            if runs:
                a, b = max(runs, key=lambda r: r[1] - r[0])
                dot_px = (a + b) / 2
        dot = to_ref(dot_px) if dot_px is not None else None

        # scroom icon: bright in any color - plain scrooms are white, rarer ones
        # are tinted (purple...). 175 is above the glow (~160) and the gray box (~136).
        y0, y1 = y_of(REEL_FISH_DY[0] * sy), y_of(REEL_FISH_DY[1] * sy)
        bright = img[max(y0, 0):min(y1, H), xa:xb].max(axis=2) >= 175
        cols = bright.sum(axis=0)
        fish = None
        if cols.size and cols.sum() >= 60 * sx * sy:
            groups = _runs(cols >= max(2, cols.max() * 0.25))   # ignores scattered sparkles
            # the scroom always stays close to the dot (it chases it), so other
            # bright things on the bar (glows from other players...) are ignored
            if dot_px is not None:
                near = [g for g in groups if abs((g[0] + g[1]) / 2 - dot_px) < 150 * sx]
                groups = near or groups
            if groups:
                a, b = max(groups, key=lambda r: cols[r[0]:r[1]].sum())
                fish = to_ref(float(np.average(np.arange(a, b), weights=cols[a:b])))
        # catch progress bar (white fill from the left)
        y = int(round(g["bot"] + REEL_PROG_DY * sy - oy))
        pa = int(round(g["x0"] + (REEL_PROG_X[0] - REEL_X0) * sx - ox))
        pb = int(round(g["x0"] + (REEL_PROG_X[1] - REEL_X0) * sx - ox))
        progress = None
        if 1 <= y < H - 1 and 0 <= pa < pb <= W:
            bar = img[y - 1:y + 2, pa:pb].min(axis=0).max(axis=1)   # fill can be tinted too
            progress = float((bar >= 150).mean())

        return (box[0] if box else None, box[1] if box else None, fish, dot, progress)


# ---------------------------------------------------------------- ring (cast)

@dataclass
class Ring:
    cx: float           # ring center (window-client px)
    cy: float
    r: float            # mid radius
    needle: float | None  # needle angle, radians
    target: float       # angle to aim for (perfect zone if visible)
    half_width: float   # half angular width of the whole green zone
    zone_mid: float = 0.0  # middle of the whole green zone


def _circ_mean(a):
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def _ang_diff(a, b):
    """Signed smallest difference a - b, in (-pi, pi]."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


class RingFinder:
    """The ring floats next to the character, so it's searched for by color."""

    def __init__(self):
        self.last = None    # (cx, cy, r) in client coords
        self.misses = 0

    def scale(self, h):
        # the ring's on-screen size follows the window height (radius 100 at
        # 1360 tall, 75 at 1020 tall in your recordings)
        return h / REF_H

    def region(self, w, h):
        if self.last:
            cx, cy, r = self.last
            return _clamp_region(cx - 2.2 * r, cy - 2.2 * r, cx + 2.2 * r, cy + 2.2 * r, w, h)
        return _clamp_region(0.15 * w, 0.08 * h, 0.85 * w, 0.92 * h, w, h)

    def find(self, img, ox, oy, w, h):
        res = self._find(img, ox, oy, h)
        if res is None:
            self.misses += 1
            if self.misses >= 2:
                self.last = None
        else:
            self.misses = 0
            self.last = (res.cx, res.cy, res.r)
        return res

    def _find(self, img, ox, oy, h):
        s = self.scale(h)
        step = 2 if s > 0.6 else 1
        sub = img[::step, ::step].astype(np.int16)
        R, G, B = sub[..., 0], sub[..., 1], sub[..., 2]
        green = (G >= 240) & (B < 70) & (R < 235)
        ys, xs = np.nonzero(green)
        if len(xs) < max(8, 40 * s * s / (step * step)):
            return None
        # keep the biggest cluster: pixels near the median (ignores stray grass)
        mx, my = np.median(xs), np.median(ys)
        keep = (np.abs(xs - mx) < 90 * s / step) & (np.abs(ys - my) < 90 * s / step)
        perfect = (R < 120)[green][keep]
        xs, ys = xs[keep].astype(float) * step, ys[keep].astype(float) * step
        if len(xs) < max(8, 30 * s * s / (step * step)):
            return None
        full = img.astype(np.int16)
        H, W = full.shape[:2]

        # The green zone is too short to fit a circle to. The center lies one
        # radius away along the zone's normal: try both sides and a range of
        # radii, keep whichever circle runs along the dark ring track.
        ax, ay = xs.mean(), ys.mean()
        _, _, vt = np.linalg.svd(np.column_stack([xs - ax, ys - ay]), full_matrices=False)
        nx, ny = -vt[0, 1], vt[0, 0]
        t = np.linspace(-np.pi, np.pi, 90, endpoint=False)
        best = None
        # only a narrow range of sizes: in dark scenes almost any circle looks like
        # the dark ring track, so a wide search locks onto the wrong size
        for f in np.linspace(0.85, 1.15, 7):
            r = RING_R * s * f
            for sign in (1, -1):
                cx, cy = ax + sign * nx * r, ay + sign * ny * r
                px = (cx + r * np.cos(t)).astype(int)
                py = (cy + r * np.sin(t)).astype(int)
                inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                if inside.mean() < 0.7:
                    continue
                ring_px = full[py[inside], px[inside]]
                # the track is dark on normal backgrounds, but over snow it's
                # mid-gray - so also accept "clearly darker than either side"
                v = ring_px.max(axis=1)
                nb = []
                for k in (0.62, 1.38):
                    qx = np.clip((cx + k * r * np.cos(t)).astype(int), 0, W - 1)[inside]
                    qy = np.clip((cy + k * r * np.sin(t)).astype(int), 0, H - 1)[inside]
                    nb.append(full[qy, qx].max(axis=1))
                dark = (v < 60) | (v <= np.maximum(nb[0], nb[1]) - 45)
                grn = (ring_px[:, 1] >= 240) & (ring_px[:, 2] < 70)
                score = (dark | grn).mean()
                if best is None or score > best[0]:
                    best = (score, cx, cy, r)
        if best is None or best[0] < 0.6:
            return None
        _, cx, cy, r = best

        ang = np.arctan2(ys - cy, xs - cx)
        zone_mid = _circ_mean(ang)
        half_width = float(np.max(np.abs([_ang_diff(a, zone_mid) for a in ang])))
        target = _circ_mean(ang[perfect]) if perfect.sum() >= max(3, 5 * s * s / (step * step)) else zone_mid

        # needle: white wedge on the ring band (it's partly gray-shaded, and
        # small on small windows, hence the lower threshold)
        needle = None
        x0, x1 = int(max(cx - 1.4 * r, 0)), int(min(cx + 1.4 * r, W))
        y0, y1 = int(max(cy - 1.4 * r, 0)), int(min(cy + 1.4 * r, H))
        win_min = full[y0:y1, x0:x1].min(axis=2) if (x1 > x0 and y1 > y0) else None
        # strict pure white first (ignores most sparkle effects); fall back to a
        # looser threshold for the gray-shaded needle on small windows
        for thresh in (245, 232):
            if win_min is None or needle is not None:
                break
            wy, wx = np.nonzero(win_min >= thresh)
            wx, wy = wx + x0, wy + y0
            d = np.hypot(wx - cx, wy - cy)
            # only the track itself - just outside it can be white snow
            band = (d > 0.8 * r) & (d < 1.1 * r)
            if band.sum() >= max(3, 6 * s * s):
                # strongest narrow spike, so other white stuff near the ring
                # (level-up sparkles...) doesn't drag it
                a = np.arctan2(wy[band] - cy, wx[band] - cx)
                bins = ((a + np.pi) / (2 * np.pi) * 72).astype(int) % 72
                hist = np.bincount(bins, minlength=72).astype(float)
                hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
                peak = (int(hist.argmax()) + 0.5) / 72 * 2 * np.pi - np.pi
                near = np.abs([_ang_diff(x, peak) for x in a]) < np.radians(10)
                if near.sum() >= max(3, 4 * s * s):
                    needle = _circ_mean(a[near])
        return Ring(cx + ox, cy + oy, r, needle, target, half_width, zone_mid)
