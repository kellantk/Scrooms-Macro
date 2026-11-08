"""Strings family minigame: a 3-lane note highway (keys A / S / D).

Notes are filled pills in their lane's color (green = A, red = S,
yellow = D) that slide down to a target circle (an outline of the same
color with a dark middle). Press the lane's key when the note reaches its
circle.

The finder locates the three circles (any window size), then reports, per
lane, how far above its circle the next note is.
"""
from dataclasses import dataclass

import numpy as np

LANES = ("A", "S", "D")


def lane_masks(img):
    """Boolean masks of each lane's color (int16 RGB image)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return {
        "A": (g > 190) & (r < 200) & (g - r > 45) & (g - b > 60),      # green (ring is paler)
        "S": (r > 200) & (g < 160) & (r - g > 70),                     # red
        "D": (r > 220) & (g > 195) & (b < 190) & (r - b > 50),         # yellow
    }


@dataclass
class Highway:
    cx: tuple          # circle centers x (A, S, D), window-client px
    cy: float          # circles' y
    r: float           # circle radius


def _ring_center(mask, h):
    """Find a ring (circle outline) in a mask: densest cluster, then check the
    pixels form an outline around a gap (not a filled note)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    cell = max(8, int(0.03 * h))
    gx, gy = xs // cell, ys // cell
    key = gx * 10000 + gy
    vals, counts = np.unique(key, return_counts=True)
    best = None
    for k in vals[np.argsort(counts)[::-1][:8]]:
        kx, ky = k // 10000, k % 10000
        cx, cy = (kx + 0.5) * cell, (ky + 0.5) * cell
        win = 0.045 * h
        # refine: re-center on the pixels around us a few times, so a ring
        # found by its edge (or with text next to it) still centers properly
        for _ in range(4):
            near = (np.abs(xs - cx) < win) & (np.abs(ys - cy) < win)
            if near.sum() < 60:
                break
            x0, x1 = xs[near].min(), xs[near].max()
            y0, y1 = ys[near].min(), ys[near].max()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            win = max(0.03 * h, 0.65 * ((x1 - x0) + (y1 - y0)) / 2)
        else:
            near = (np.abs(xs - cx) < win) & (np.abs(ys - cy) < win)
        if near.sum() < 60:
            continue
        x0, x1 = xs[near].min(), xs[near].max()
        y0, y1 = ys[near].min(), ys[near].max()
        rad = ((x1 - x0) + (y1 - y0)) / 4
        if not (0.018 * h < rad < 0.05 * h):
            continue
        # outline: few pixels near the middle, many on the rim
        d = np.hypot(xs[near] - cx, ys[near] - cy)
        rim = ((d > 0.7 * rad) & (d < 1.25 * rad)).mean()
        if rim < 0.7:
            continue
        score = near.sum() * rim
        if best is None or score > best[0]:
            best = (score, cx, cy, rad)
    return best[1:] if best else None


def find_highway(img, ox, oy, h):
    """img: RGB area (int16) at offset (ox, oy). Returns Highway or None."""
    masks = lane_masks(img)
    found = {}
    for lane in LANES:
        res = _ring_center(masks[lane], h)
        if res is None:
            return None
        found[lane] = res
    (ax, ay, ar), (sx, sy, sr), (dx, dy, dr) = (found[k] for k in LANES)
    # three circles in a row, left to right, evenly spaced, same size
    if not (ax < sx < dx):
        return None
    if max(ay, sy, dy) - min(ay, sy, dy) > 0.02 * h:
        return None
    g1, g2 = sx - ax, dx - sx
    if g1 <= 0 or abs(g1 - g2) > 0.25 * max(g1, g2):
        return None
    r = (ar + sr + dr) / 3
    if not (2.2 * r < g1 < 6 * r):
        return None
    return Highway((ax + ox, sx + ox, dx + ox), (ay + sy + dy) / 3 + oy, r)


def next_notes(img, ox, oy, hw, masks=None):
    """img is int16 RGB covering the highway's lower part. Returns
    (notes, holds, bar_near), each keyed by lane:
      notes:    y (window px) of the lowest note still above its circle, or None
      holds:    True when that note is a hold note (a bar trails above it)
      bar_near: how much of the strip just above the circle a hold bar covers
                (0..1) - the key stays down until the bar has passed"""
    if masks is None:
        masks = lane_masks(img)
    notes, holds, bar_near = {}, {}, {}
    r = hw.r
    y_lo = int(hw.cy - 9 * r - oy)            # how far up to look
    y_hi = int(hw.cy - 1.15 * r - oy)         # stop above the circle's own outline
    thick = max(4, 0.35 * r)                  # a note's head is a solid pill
    thin = max(2, 0.08 * r)                   # a hold bar is a thin line
    for i, lane in enumerate(LANES):
        x = hw.cx[i] - ox
        mask = masks[lane]
        s = mask[max(0, int(hw.cy - 2.2 * r - oy)):max(0, y_hi), max(0, int(x - 0.5 * r)):max(0, int(x + 0.5 * r))]
        rows_near = s.sum(axis=1)
        bar_near[lane] = float((rows_near >= thin).mean()) if rows_near.size else 0.0
        # the lane leans a little (perspective): allow some sideways room
        xa = max(0, int(x - 1.6 * r))
        m = mask[max(0, y_lo):max(0, y_hi), xa:max(0, int(x + 2.2 * r))]
        rows = m.sum(axis=1)
        filled = np.flatnonzero(rows >= thick)
        holds[lane] = False
        if filled.size == 0:
            notes[lane] = None
            continue
        # lowest pill: last run of filled rows
        bottom = filled[-1]
        top = bottom
        while top - 1 >= 0 and rows[top - 1] >= thick:
            top -= 1
        if bottom >= len(rows) - 2:
            # the note is sliding under the circle's outline: only part of it
            # shows, so its middle would seem to slow down. Report it hidden
            # and let the bot predict where it is from the earlier readings
            # (for a hold note the bar behind it would even stall the reading)
            holds[lane] = bottom - top > 1.3 * r
            notes[lane] = None
            continue
        if bottom - top > 1.3 * r:
            # a hold bar gets wider near the circle and merges with its head:
            # the head is the bottom part
            center = bottom - 0.4 * r
            holds[lane] = True
        else:
            center = (top + bottom) / 2
            # a thin bar right above the head = a hold note
            a0, a1 = top - int(1.5 * r), top - int(0.1 * r)
            if a0 >= 0 and a1 > a0:
                cols = np.flatnonzero(m[top:bottom + 1].any(axis=0))
                hx = (cols.min() + cols.max()) / 2 if cols.size else x - xa
                strip = m[a0:a1, max(0, int(hx - 0.5 * r)):max(0, int(hx + 0.5 * r))].sum(axis=1)
                holds[lane] = bool(strip.size) and (strip >= thin).mean() >= 0.6
        notes[lane] = max(0, y_lo) + center + oy
    return notes, holds, bar_near


def _rim(mask, cx, cy, r, n=48):
    """Fraction of a circle's outline (+-2 px) that is lit in mask."""
    H, W = mask.shape
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    hit = np.zeros(n, bool)
    for dr in (-2, 0, 2):
        xs = np.clip((cx + (r + dr) * np.cos(t)).astype(int), 0, W - 1)
        ys = np.clip((cy + (r + dr) * np.sin(t)).astype(int), 0, H - 1)
        hit |= mask[ys, xs]
    return float(hit.mean())


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


class StringsFinder:
    """Same interface as the other finders: region() to grab, find() to read.
    find() returns (Highway, {lane: note_y or None}) or None."""

    def __init__(self):
        self.hw = None
        self.misses = 0
        self.holds = {}         # lane -> the next note is a hold note
        self.bar_near = {}      # lane -> hold bar coverage just above the circle

    def region(self, w, h):
        if self.hw:
            r = self.hw.r
            return _clamp(self.hw.cx[0] - 5 * r, self.hw.cy - 10 * r, self.hw.cx[2] + 5 * r, self.hw.cy + 2 * r, w, h)
        return _clamp(0.2 * w, 0.4 * h, 0.8 * w, 0.97 * h, w, h)

    def find(self, img, ox, oy, w, h):
        img = img.astype(np.int16)
        masks = lane_masks(img)
        hw = find_highway(img, ox, oy, h)
        if hw is None:
            self.misses += 1
            # the highway doesn't move; a note sliding into a circle (or a long
            # hold bar lying over it) hides its outline for a while - keep the
            # saved position while the other circles are still there
            if self.hw is not None:
                old = self.hw
                seen = sum(_rim(masks[k], old.cx[i] - ox, old.cy - oy, old.r) >= 0.5
                           for i, k in enumerate(LANES))
                if self.misses >= 90 and seen < 2:
                    self.hw = None
            if self.hw is None:
                return None
            hw = self.hw          # brief miss (e.g. a hit flash): keep the last position
        else:
            self.misses = 0
            self.hw = hw
        notes, self.holds, self.bar_near = next_notes(img, ox, oy, hw, masks)
        return hw, notes
