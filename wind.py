"""Wind family minigame: a 5-line staff.

Move the mouse up and down to steer the scroom (a diamond on the left) from
line to line. Notes slide in from the right on the lines: a grey diamond head
with a mint bar trailing behind it. When a note's head reaches the scroom,
hold left click and stay on its line until the bar has passed. Staying on the
line fills the progress bar under the staff.

The staff is a long white line above and below the 5 lines. Positions are
fractions of the white lines' length L and of the gap D between them
(measured on 3440x1440 and 1766x720 recordings).
"""
from dataclasses import dataclass

import numpy as np

import detect

LINES = (0.142, 0.321, 0.5, 0.679, 0.858)   # the 5 lines, top to bottom, as a fraction of D
SCROOM_X = 0.084                            # the scroom's column, along L
SPACING = (0.29, 0.38)                      # D / L (measured 0.335)
HEAD = 0.024                                # a note's head is this far left of where its bar starts (L)


@dataclass
class Staff:
    x0: float        # white lines' left end (window px)
    L: float         # their length
    top: float       # upper white line y
    D: float         # gap down to the lower white line

    @property
    def sx(self):
        return self.x0 + SCROOM_X * self.L

    @property
    def lines(self):
        return [self.top + f * self.D for f in LINES]

    @property
    def spacing(self):
        return (LINES[1] - LINES[0]) * self.D


@dataclass
class Note:
    line: int        # 0 = top line
    head: float      # x of the head (window px); can be left of the scroom
    tail: float      # x where its bar ends


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def find_staff(img, ox, oy, h):
    lines = detect.find_hlines(img, 170, 0.4 * h, ridge=True)
    for top, bot in sorted(detect._pairs(lines, *SPACING, overlap=0.8), key=lambda p: -p[1].y):
        L = max(top.length, bot.length)
        if not (0.5 * h < L < 0.9 * h):
            continue
        x0 = min(top.x0, bot.x0)
        st = Staff(x0 + ox, L, top.y + oy, bot.y - top.y)
        # the 5 inner lines are dim grey, between dark gaps: check a few
        ok = 0
        xa, xb = int(x0 + 0.75 * L), int(x0 + 0.95 * L)
        for y in st.lines:
            yy = int(round(y - oy))
            if 2 <= yy < img.shape[0] - 3 and xb > xa:
                v = img[yy - 1:yy + 2, xa:xb].max(axis=2).max(axis=0).astype(np.int16)
                around = np.maximum(img[yy - 3 - int(0.03 * st.D), xa:xb].max(axis=1),
                                    img[yy + 3 + int(0.03 * st.D), xa:xb].max(axis=1)).astype(np.int16)
                ok += (v - around >= 12).mean() >= 0.4
        if ok >= 3:
            return st
    return None


def scroom_y(img, ox, oy, st):
    """y of the scroom (its white icon) in its column, or None."""
    half = int(0.035 * st.L)
    x = int(st.sx - ox)
    ya, yb = int(st.top + 0.04 * st.D - oy), int(st.top + 0.96 * st.D - oy)
    if ya < 0 or yb > img.shape[0] or x - half < 0 or x + half > img.shape[1]:
        return None
    col = img[ya:yb, x - half:x + half]
    rows = (col.min(axis=2) > 232).sum(axis=1) >= 2
    runs = [(a, b) for a, b in detect._runs(rows)]
    if not runs:
        return None
    # the icon can show as a few pieces: join pieces closer than its size
    groups = []
    for a, b in runs:
        if groups and a - groups[-1][1] <= 0.08 * st.D:
            groups[-1][1] = b
        else:
            groups.append([a, b])
    a, b = max(groups, key=lambda g: g[1] - g[0])
    if b - a < 0.04 * st.D:
        return None
    return ya + (a + b) / 2 + oy


def read_notes(img, ox, oy, st):
    """Notes on the staff (head + tail x, window px), left to right."""
    notes = []
    half = max(2, int(0.035 * st.D))
    xa, xb = int(st.x0 - ox), int(st.x0 + st.L - ox)
    if xa < 0 or xb > img.shape[1]:
        return notes
    # the scroom's own diamond covers the bar where it sits
    s0, s1 = int(st.sx - 0.045 * st.L - ox) - xa, int(st.sx + 0.045 * st.L - ox) - xa
    gap = max(3, int(0.012 * st.L))
    for i, y in enumerate(st.lines):
        yy = int(round(y - oy))
        if yy - half < 0 or yy + half >= img.shape[0]:
            continue
        band = img[yy - half:yy + half + 1, xa:xb].astype(np.int16)
        r, g, b = band[..., 0], band[..., 1], band[..., 2]
        mint = ((g > 180) & (g - r > 15) & (g - b > 2) & (r > 110)).sum(axis=0) >= 2
        runs = []
        for a, e in detect._runs(mint):
            # join pieces split by a small gap or by the scroom's diamond
            if runs and (a - runs[-1][1] <= gap or (runs[-1][1] >= s0 - gap and a <= s1 + gap)):
                runs[-1][1] = e
            else:
                runs.append([a, e])
        for a, e in runs:
            if e - a < 0.012 * st.L:
                continue
            notes.append(Note(i, a + xa + ox - HEAD * st.L, e + xa + ox))
    return sorted(notes, key=lambda n: n.head)


class WindFinder:
    """region() to grab, find() -> (Staff, scroom y or None, notes) or None."""

    def __init__(self):
        self.staff = None
        self.misses = 0

    def region(self, w, h):
        if self.staff:
            st = self.staff
            return _clamp(st.x0 - 0.05 * st.L, st.top - 0.3 * st.D, st.x0 + 1.05 * st.L, st.top + 1.3 * st.D, w, h)
        return _clamp(0.1 * w, 0.4 * h, 0.9 * w, 0.98 * h, w, h)

    def find(self, img, ox, oy, w, h):
        st = find_staff(img, ox, oy, h)
        if st is None:
            self.misses += 1
            if self.misses >= 60:      # the white lines can hide briefly (hit flashes)
                self.staff = None
            if self.staff is None:
                return None
            st = self.staff
        else:
            self.misses = 0
            self.staff = st
        return st, scroom_y(img, ox, oy, st), read_notes(img, ox, oy, st)
