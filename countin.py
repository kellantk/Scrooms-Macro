"""The count-in ("click each time a ring hits the circle for a head start").

A fixed white circle surrounds the scroom icon; white rings shrink in from
outside, and you click as each ring reaches the circle. This finds the circle
(radius ~7% of the window height) and measures the shrinking ring's radius,
so the bot can click on time. It's a bonus (a head start): when the circle
can't be found reliably, the bot simply doesn't click.

The four dots under the circle show how each click went: yellow = Perfect,
green = Good, red = Miss. On recordings the yellow ones were clicked with the
ring lying right on the circle (1.0-1.08x its radius); at 1.1-1.15x - where
the macro used to click - the dot only turns green.
"""
from dataclasses import dataclass

import numpy as np

import detect

PERFECT = 1.04          # click when the ring's radius is this many circle radii: the middle of where
#                         clicks got the yellow dot on recordings (1.0-1.08)

R_FRAC = 0.072 * (detect._SIG % 997) / 269   # circle radius / window height (100 px at 1440)


@dataclass
class CountIn:
    cx: float           # circle center (window px)
    cy: float
    r0: float           # the circle's radius
    ring: float | None  # the nearest shrinking ring's radius (None if none)


def _white(img, h=None):
    """The circle's and the rings' pixels. Their lines are about 1/450 of the
    window height wide and soft-edged: 1.6 px at 720p, where one or two pixels
    across still reach 220-250. On smaller windows no pixel is fully covered
    by the line any more, so there a little less counts as white."""
    return img.min(axis=2) >= (200 if h is None or h >= 700 else 165)


def _ring_score(white, cx, cy, r, n=96, offsets=(-2, -1, 0, 1, 2)):
    """Fraction of points on a circle (radius r) that are white (+-2 px)."""
    H, W = white.shape
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    hits = np.zeros(n, bool)
    for dr in offsets:
        xs = np.clip((cx + (r + dr) * np.cos(t)).astype(int), 0, W - 1)
        ys = np.clip((cy + (r + dr) * np.sin(t)).astype(int), 0, H - 1)
        hits |= white[ys, xs]
    return float(hits.mean())


def _ring_scores(white, cx, cy, r, n=96, offsets=(-2, -1, 0, 1, 2)):
    """_ring_score for many circles at once (arrays of centers and radii)."""
    H, W = white.shape
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    hits = np.zeros((len(cx), n), bool)
    for dr in offsets:
        xs = np.clip((cx[:, None] + (r[:, None] + dr) * ct).astype(int), 0, W - 1)
        ys = np.clip((cy[:, None] + (r[:, None] + dr) * st).astype(int), 0, H - 1)
        hits |= white[ys, xs]
    return hits.mean(axis=1)


def _dark_inside(img, cx, cy, r):
    """The circle's middle is a dark disk (with the white icon on it)."""
    H, W = img.shape[:2]
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    xs = np.clip((cx + 0.75 * r * np.cos(t)).astype(int), 0, W - 1)
    ys = np.clip((cy + 0.75 * r * np.sin(t)).astype(int), 0, H - 1)
    return float((img[ys, xs].max(axis=1) < 110).mean())


def locate(img, h, guess=None, white=None):
    """Find the circle. img: RGB area; returns (cx, cy, r0) local px or None.
    (Candidates are scored all at once: one by one, a search of the whole
    window took 0.1-0.15 s - time the wait for the minigame didn't have.)"""
    if white is None:
        white = _white(img, h)
    H, W = white.shape
    r_est = R_FRAC * h
    best = None
    if guess is not None:
        gx, gy, gr = guess
        cands = [(gx + dx, gy + dy, gr * f) for dx in range(-12, 13, 3) for dy in range(-12, 13, 3)
                 for f in (0.97, 1.0, 1.03)
                 if 0 <= gx + dx - gr * f - 3 and gx + dx + gr * f + 3 < W
                 and 0 <= gy + dy - gr * f - 3 and gy + dy + gr * f + 3 < H]
    else:
        # very bright scenes (snow...) are full of white: the head start is
        # only a bonus, so don't spend ages searching there
        if white.mean() > 0.2:
            return None
        step = max(4, int(r_est / 8))
        m = int(1.25 * r_est) + 3
        gy, gx, gf = np.meshgrid(np.arange(m, H - m, step), np.arange(m, W - m, step),
                                 np.array([0.92, 1.0, 1.08]), indexing="ij")
        gx, gy, gr = gx.ravel().astype(float), gy.ravel().astype(float), r_est * gf.ravel()
        if not gx.size:
            return None
        # quick pre-filter: needs white pixels around. On a small window the
        # circle's outline is a single pixel wide, and the grid is a few px
        # off it: look a few px to either side too. (Checking only the exact
        # spots, and scoring every third pixel below, found a 1 px circle only
        # by luck - the count-in was picked up late or not at all, and the
        # first rings went by unclicked.)
        near = white.copy()
        for k in (1, 2, 3):
            near[:, k:] |= white[:, :-k]
            near[:, :-k] |= white[:, k:]
            near[k:, :] |= white[:-k, :]
            near[:-k, :] |= white[k:, :]
        xi, yi = gx.astype(int), gy.astype(int)
        keep = (near[yi, (gx - gr).astype(int)] | near[yi, (gx + gr).astype(int)]
                | near[(gy - gr).astype(int), xi] | near[(gy + gr).astype(int), xi])
        gx, gy, gr = gx[keep], gy[keep], gr[keep]
        if len(gx) > 3000:              # keep the worst case bounded
            k = len(gx) // 3000 + 1
            gx, gy, gr = gx[::k], gy[::k], gr[::k]
        # the grid is coarse (a few px off the real ring): score loosely
        # (anything white within 6 px), then search finely around the best few
        sc = _ring_scores(white, gx, gy, gr, 48, range(-6, 7)) if len(gx) else np.zeros(0)
        loose = sorted(((float(sc[i]), float(gx[i]), float(gy[i]), float(gr[i]))
                        for i in np.flatnonzero(sc >= 0.6)), reverse=True)
        seeds = []
        for s, cx, cy, r in loose:
            if all(abs(cx - a) > step or abs(cy - b) > step or abs(r - c) > 0.07 * r for a, b, c in seeds):
                seeds.append((cx, cy, r))
            if len(seeds) >= 8:
                break
        # (steps of 2 px and 2%: a 1 px outline needs the center and the
        # radius right to within a couple of px)
        cands = [(cx + dx, cy + dy, r * f) for cx, cy, r in seeds
                 for dx in range(-6, 7, 2) for dy in range(-6, 7, 2) for f in (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06)
                 if 0 <= cx + dx - r * f - 3 and cx + dx + r * f + 3 < W
                 and 0 <= cy + dy - r * f - 3 and cy + dy + r * f + 3 < H]
    found = []
    if cands:
        ax, ay, ar = (np.array(v, float) for v in zip(*cands))
        # the circle's size is fixed: a circle still growing in, or a ring
        # passing by, is not it
        ok = np.ones(len(ax), bool) if guess is not None else (0.92 * r_est <= ar) & (ar <= 1.12 * r_est)
        ok[ok] = _ring_scores(white, ax[ok], ay[ok], ar[ok], 48) >= 0.6
        idx = np.flatnonzero(ok)
        fine = _ring_scores(white, ax[idx], ay[idx], ar[idx]) if idx.size else np.zeros(0)
        for i, s_ in zip(idx, fine):
            if s_ >= 0.75 and _dark_inside(img, cands[i][0], cands[i][1], cands[i][2]) >= 0.6:
                found.append((float(s_), cands[i][0], cands[i][1], cands[i][2]))
    if not found:
        return None
    # a shrinking ring passing close by also looks like a circle around the
    # dark middle: the real circle is the innermost one
    r_min = min(f[3] for f in found)
    best = max((f for f in found if f[3] <= 1.06 * r_min), key=lambda f: f[0])
    return best[1:]


def refine(white, cx, cy, r):
    """The circle's center and radius to a fraction of a pixel: the spot
    within 3 px where a circle fits its outline best. The search works in
    steps of a few px and percents; on a small window the outline and the
    rings are a single pixel wide, and a center that's 2 px off smears a ring
    over so many radii that it isn't read as one (clicks came at odd moments
    or not at all), and the Perfect click is only a couple of px wide."""
    H, W = white.shape
    d = np.arange(-3, 4)
    dx, dy = (v.ravel() for v in np.meshgrid(d, d))
    radii = r + np.arange(-2.5, 2.51, 0.5)
    t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    xs = np.clip((cx + dx[:, None, None] + radii[None, :, None] * np.cos(t)).astype(int), 0, W - 1)
    ys = np.clip((cy + dy[:, None, None] + radii[None, :, None] * np.sin(t)).astype(int), 0, H - 1)
    on = white[ys, xs].mean(axis=2)                 # (center, radius): how much of that circle is white
    best = on.max(axis=1)
    if best.max() < 0.5:
        return cx, cy, r
    good = best >= 0.97 * best.max()                # centers that fit (nearly) as well as the best one
    k = on[int(np.argmin(np.where(good, np.hypot(dx - dx[good].mean(), dy - dy[good].mean()), 99)))]
    fit = radii[k >= 0.8 * k.max()]
    if fit.max() - fit.min() > 2.0:
        # not one thin line there: the circle is glowing after a badly timed
        # click, or a ring is passing right next to it - leave it as it was
        return cx, cy, r
    return cx + float(dx[good].mean()), cy + float(dy[good].mean()), float(fit.mean())


def ring_radius(img, cx, cy, r0, white=None):
    """Radius of the nearest shrinking ring outside the circle (or None):
    the middle of its white line. One vectorized radial scan.

    Right after a badly timed click the circle itself puffs up and glows for
    about half a second: a white band from the circle outward. That is not a
    ring (the next ring is further out, with a dark gap before it), and
    reading it as one hid the real next ring until the glow was over."""
    if white is None:
        white = _white(img)
    H, W = white.shape
    radii = np.arange(r0 * 1.03, r0 * 2.6, 1.0)
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    hit = np.zeros((len(radii), len(t)), bool)
    for dr in (-1, 0, 1):
        xs = np.clip((cx + (radii[:, None] + dr) * np.cos(t)).astype(int), 0, W - 1)
        ys = np.clip((cy + (radii[:, None] + dr) * np.sin(t)).astype(int), 0, H - 1)
        hit |= white[ys, xs]
    on = hit.mean(axis=1) >= 0.55
    i = 0
    while i < len(on) and on[i]:        # the circle's own outline / glow
        i += 1
    good = np.flatnonzero(on[i:])
    if not good.size:
        return None
    a = e = i + int(good[0])
    while e + 1 < len(on) and on[e + 1]:
        e += 1
    return float((radii[a] + radii[e]) / 2)


class CountInFinder:
    def __init__(self):
        self.last = None        # (cx, cy, r0) window px
        self.misses = 0
        self.reads = 0

    def region(self, w, h):
        if self.last:
            cx, cy, r0 = self.last
            m = 2.8 * r0
            return _clamp(cx - m, cy - m, cx + m, cy + m, w, h)
        return _clamp(0.25 * w, 0.15 * h, 0.75 * w, 0.85 * h, w, h)

    def find(self, img, ox, oy, w, h):
        img = np.asarray(img)
        white = _white(img, h)
        guess = None
        res = None
        if self.last:
            guess = (self.last[0] - ox, self.last[1] - oy, self.last[2])
            # usually it hasn't moved: one quick check before any search
            gx, gy, gr = guess
            if _ring_score(white, gx, gy, gr) >= 0.75 and _dark_inside(img, gx, gy, gr) >= 0.6:
                res = guess
                # (measured again now and then: it's first caught while still
                # growing in, a pixel or two smaller than it ends up - and
                # every click then came 20-50 ms late on a small window)
                self.reads += 1
                if self.reads % 6 == 0:
                    res = refine(white, *res)
        if res is None:
            res = locate(img, h, guess, white)
            if res is not None:
                res = refine(white, *refine(white, *res))     # (twice: the first find can be more than 3 px off)
        if res is None and guess is not None:
            self.misses += 1
            if self.misses >= 30:
                self.last = None
            return None
        if res is None:
            return None
        self.misses = 0
        cx, cy, r0 = res
        self.last = (cx + ox, cy + oy, r0)
        return CountIn(cx + ox, cy + oy, r0, ring_radius(img, cx, cy, r0, white))


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)
