"""The count-in ("click each time a ring hits the circle for a head start").

A fixed white circle surrounds the scroom icon; white rings shrink in from
outside, and you click as each ring reaches the circle. This finds the circle
(radius ~7% of the window height) and measures the shrinking ring's radius,
so the bot can click on time. It's a bonus (a head start): when the circle
can't be found reliably, the bot simply doesn't click.
"""
from dataclasses import dataclass

import numpy as np

R_FRAC = 0.072          # circle radius / window height (100 px at 1440)


@dataclass
class CountIn:
    cx: float           # circle center (window px)
    cy: float
    r0: float           # the circle's radius
    ring: float | None  # the nearest shrinking ring's radius (None if none)


def _white(img):
    return img.min(axis=2) >= 200


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


def _dark_inside(img, cx, cy, r):
    """The circle's middle is a dark disk (with the white icon on it)."""
    H, W = img.shape[:2]
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    xs = np.clip((cx + 0.75 * r * np.cos(t)).astype(int), 0, W - 1)
    ys = np.clip((cy + 0.75 * r * np.sin(t)).astype(int), 0, H - 1)
    return float((img[ys, xs].max(axis=1) < 110).mean())


def locate(img, h, guess=None):
    """Find the circle. img: RGB area; returns (cx, cy, r0) local px or None."""
    white = _white(img)
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
        cands = [(x, y, r_est * f) for y in range(m, H - m, step)
                 for x in range(m, W - m, step) for f in (0.92, 1.0, 1.08)]
        # quick pre-filter: needs white pixels around
        cands = [c for c in cands if white[int(c[1]), int(c[0] - c[2])] or white[int(c[1]), int(c[0] + c[2])]
                 or white[int(c[1] - c[2]), int(c[0])] or white[int(c[1] + c[2]), int(c[0])]]
        if len(cands) > 3000:           # keep the worst case bounded
            cands = cands[:: len(cands) // 3000 + 1]
        # the grid is coarse (a few px off the real ring): score loosely
        # (+-6 px), then search finely around the best few
        loose = []
        for cx, cy, r in cands:
            s = _ring_score(white, cx, cy, r, 48, (-6, -3, 0, 3, 6))
            if s >= 0.6:
                loose.append((s, cx, cy, r))
        loose.sort(reverse=True)
        seeds = []
        for s, cx, cy, r in loose:
            if all(abs(cx - a) > step or abs(cy - b) > step or abs(r - c) > 0.07 * r for a, b, c in seeds):
                seeds.append((cx, cy, r))
            if len(seeds) >= 8:
                break
        cands = [(cx + dx, cy + dy, r * f) for cx, cy, r in seeds
                 for dx in range(-6, 7, 3) for dy in range(-6, 7, 3) for f in (0.94, 0.97, 1.0, 1.03, 1.06)
                 if 0 <= cx + dx - r * f - 3 and cx + dx + r * f + 3 < W
                 and 0 <= cy + dy - r * f - 3 and cy + dy + r * f + 3 < H]
    found = []
    for cx, cy, r in cands:
        # the circle's size is fixed: a circle still growing in, or a ring
        # passing by, is not it
        if guess is None and not 0.92 * r_est <= r <= 1.12 * r_est:
            continue
        s = _ring_score(white, cx, cy, r, 48)
        if s < 0.6:
            continue
        s = _ring_score(white, cx, cy, r)
        if s >= 0.75 and _dark_inside(img, cx, cy, r) >= 0.6:
            found.append((s, cx, cy, r))
    if not found:
        return None
    # a shrinking ring passing close by also looks like a circle around the
    # dark middle: the real circle is the innermost one
    r_min = min(f[3] for f in found)
    best = max((f for f in found if f[3] <= 1.06 * r_min), key=lambda f: f[0])
    return best[1:]


def ring_radius(img, cx, cy, r0):
    """Radius of the nearest shrinking ring outside the circle (or None).
    One vectorized radial scan."""
    white = _white(img)
    H, W = white.shape
    radii = np.arange(r0 * 1.12, r0 * 2.6, 1.5)
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    hit = np.zeros((len(radii), len(t)), bool)
    for dr in (-1.5, 0, 1.5):
        xs = np.clip((cx + (radii[:, None] + dr) * np.cos(t)).astype(int), 0, W - 1)
        ys = np.clip((cy + (radii[:, None] + dr) * np.sin(t)).astype(int), 0, H - 1)
        hit |= white[ys, xs]
    good = np.flatnonzero(hit.mean(axis=1) >= 0.55)
    return float(radii[good[0]]) if good.size else None


class CountInFinder:
    def __init__(self):
        self.last = None        # (cx, cy, r0) window px
        self.misses = 0

    def region(self, w, h):
        if self.last:
            cx, cy, r0 = self.last
            m = 2.8 * r0
            return _clamp(cx - m, cy - m, cx + m, cy + m, w, h)
        return _clamp(0.25 * w, 0.15 * h, 0.75 * w, 0.85 * h, w, h)

    def find(self, img, ox, oy, w, h):
        img = np.asarray(img)
        guess = None
        res = None
        if self.last:
            guess = (self.last[0] - ox, self.last[1] - oy, self.last[2])
            # usually it hasn't moved: one quick check before any search
            gx, gy, gr = guess
            if _ring_score(_white(img), gx, gy, gr) >= 0.75 and _dark_inside(img, gx, gy, gr) >= 0.6:
                res = guess
        if res is None:
            res = locate(img, h, guess)
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
        return CountIn(cx + ox, cy + oy, r0, ring_radius(img, cx, cy, r0))


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)
