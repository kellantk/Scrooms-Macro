"""Percussion family minigame: one horizontal lane (Taiko style).

Notes slide right-to-left toward the scroom's diamond on the left:
  white diamond  -> left click
  blue diamond   -> right click
  big half/half  -> left + right click together
The panel has a long line above and below the lane, and a vertical divider
line ~22% along. Positions below are fractions of the line length L
(measured on a 3440x1440 recording).
"""
from dataclasses import dataclass

import numpy as np

import detect

DIVIDER = 0.218      # vertical divider, along the lines
TARGET = 0.0864      # scroom diamond center, along the lines
SPACING = (0.15, 0.20)   # line gap / L (measured 0.17)


@dataclass
class Lane:
    x0: float        # line start (window px)
    L: float         # line length
    y: float         # lane center y
    gap: float       # distance between the two lines

    @property
    def target_x(self):
        return self.x0 + TARGET * self.L


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def find_lane(img, ox, oy, w):
    lines = detect.find_hlines(img, 200, 0.2 * w, ridge=True)
    H, W = img.shape[:2]
    for top, bot in sorted(detect._pairs(lines, *SPACING), key=lambda p: -p[1].y):
        L = (top.length + bot.length) / 2
        x0 = (top.x0 + bot.x0) / 2
        y0, y1 = int(top.y + 0.1 * (bot.y - top.y)), int(bot.y - 0.1 * (bot.y - top.y))
        # the divider: a bright vertical line between the two lines, ~22% along
        best = 0.0
        for dx in range(int(-0.012 * L), int(0.012 * L) + 1):
            x = int(x0 + DIVIDER * L + dx)
            if 0 <= x < W and y1 > y0:
                col = img[y0:y1, x].max(axis=1)
                best = max(best, float((col >= 150).mean()))
        if best >= 0.6:
            return Lane(x0 + ox, L, (top.y + bot.y) / 2 + oy, bot.y - top.y)
    return None


def read_notes(img, ox, oy, lane):
    """Notes in the lane, left to right: list of (x_center, kind) with kind
    'L' (white), 'R' (blue) or 'B' (big: both). Window px."""
    L = lane.L
    half = 0.04 * L
    ya, yb = int(lane.y - half - oy), int(lane.y + half - oy)
    xa = int(lane.target_x + 0.07 * L - ox)          # right of the scroom's own diamond
    xb = int(lane.x0 + L - ox)
    H, W = img.shape[:2]
    if ya < 0 or yb > H or xa < 0 or xb > W or xb <= xa:
        return []
    band = img[ya:yb, xa:xb].astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    need = max(4, int(0.008 * L))
    blue = ((b > 180) & (b - r > 90)).sum(axis=0) >= need
    white = ((band.min(axis=2) > 185) & (band.max(axis=2) - band.min(axis=2) < 40)).sum(axis=0) >= need
    minw = max(5, 0.015 * L)
    blues = [((a + e) / 2, e - a) for a, e in detect._runs(blue) if e - a >= minw]
    whites = [((a + e) / 2, e - a) for a, e in detect._runs(white) if e - a >= minw]
    notes = []
    used = set()
    for bx, bw in blues:
        # big note: a white half right next to a blue half
        partner = next((i for i, (wx, ww) in enumerate(whites)
                        if i not in used and 0 < bx - wx < 0.09 * L), None)
        if partner is not None:
            used.add(partner)
            notes.append(((whites[partner][0] + bx) / 2 + xa + ox, "B"))
        else:
            notes.append((bx + xa + ox, "R"))
    for i, (wx, ww) in enumerate(whites):
        if i not in used:
            notes.append((wx + xa + ox, "L"))
    return sorted(notes)


class PercussionFinder:
    """region() to grab, find() -> (Lane, notes) or None."""

    def __init__(self):
        self.lane = None
        self.misses = 0

    def region(self, w, h):
        if self.lane:
            ln = self.lane
            return _clamp(ln.x0 - 0.05 * ln.L, ln.y - 0.8 * ln.gap, ln.x0 + 1.05 * ln.L, ln.y + 0.8 * ln.gap, w, h)
        return _clamp(0.03 * w, 0.45 * h, 0.97 * w, h, w, h)

    def find(self, img, ox, oy, w, h):
        lane = find_lane(img, ox, oy, w)
        if lane is None:
            self.misses += 1
            if self.misses >= 90:      # ~1.5 s: notes passing can hide the lines briefly
                self.lane = None
            if self.lane is None:
                return None
            lane = self.lane
        else:
            self.misses = 0
            self.lane = lane
        return lane, read_notes(img, ox, oy, lane)
