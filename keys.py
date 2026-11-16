"""Keys family minigame: Piano-Tiles style, 4 columns (keys A S D F).

White tiles fall straight down; press a column's key when its tile reaches
the bottom slot row. The board is a frame with 5 evenly spaced vertical lines
and a horizontal line under the slot row. Sizes below are fractions of the
board width L (measured on a 3440x1440 recording: L = 480).
"""
from dataclasses import dataclass

import numpy as np

import detect

LANES = ("A", "S", "D", "F")
SLOT = 0.0375        # slot-row center, above the bottom line
TOP = 1.28           # board top, above the bottom line


@dataclass
class Board:
    x0: float
    L: float
    line_y: float

    @property
    def cy(self):          # where tiles are hit (slot row center)
        return self.line_y - SLOT * self.L

    @property
    def r(self):           # a "note size" used by the shared lane logic
        return 0.1 * self.L

    def col_x(self, k):
        return self.x0 + (k + 0.5) * self.L / 4


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _vertical_line(img, x, ya, yb, search):
    """Fraction of rows where there's a thin bright vertical line near x."""
    H, W = img.shape[:2]
    best = 0.0
    for dx in range(-search, search + 1):
        c = int(x + dx)
        if c - 5 < 0 or c + 5 >= W or yb <= ya:
            continue
        v = img[ya:yb, c - 5:c + 6].max(axis=2).astype(np.int16)
        ridge = v[:, 5] - np.maximum(v[:, 1], v[:, 9])
        best = max(best, float((ridge >= 15).mean()))
    return best


def find_board(img, ox, oy, w):
    lines = detect.find_hlines(img, 150, 0.08 * w, channel="max")
    H = img.shape[0]
    for ln in sorted(lines, key=lambda l: -l.y):
        L = ln.length
        if L < 40:
            continue
        ya, yb = int(ln.y - 1.15 * L), int(ln.y - 0.2 * L)
        if ya < 0:
            continue
        search = max(2, int(0.01 * L))
        # 5 evenly spaced vertical lines (board edges + 3 column separators)
        if all(_vertical_line(img, ln.x0 + k * L / 4, ya, yb, search) >= 0.6 for k in range(5)):
            return Board(ln.x0 + ox, L, ln.y + oy)
    return None


def next_tiles(img, ox, oy, board):
    """Returns (tiles, holds, bar_near), each keyed by column:
      tiles:    y (window px) of the lowest tile's center, or None
      holds:    True when that tile is a hold note (a grey bar trails above it)
      bar_near: how much of the strip just above the slot row a hold bar covers
    """
    out, holds, bar_near = {}, {}, {}
    L = board.L
    ya = int(board.line_y - TOP * L - oy)
    yb = int(board.line_y - 0.01 * L - oy)          # stop above the bottom line
    half = int(0.33 * L / 4)
    y0 = max(0, ya)
    blue_end = int(board.line_y - 0.1 * L - oy) - y0     # the slot row can glow when pressed
    for k, lane in enumerate(LANES):
        x = int(board.col_x(k) - ox)
        sub = img[y0:max(0, yb), max(0, x - half):x + half].astype(np.int16)
        r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
        white = sub.min(axis=2) >= 200
        # a hold note's bar: see-through grey, narrower than a tile - grey in
        # the middle of the column but not at its sides
        mx, mn = sub.max(axis=2), sub.min(axis=2)
        grey = (mn >= 85) & (mn < 200) & (mx - mn <= 30)
        c = grey.shape[1] // 2
        q = max(2, int(0.18 * L / 4))
        bar = grey[:, c - q:c + q].mean(axis=1) >= 0.8
        for sx in (x - int(0.43 * L / 4), x + int(0.43 * L / 4)):
            if 0 <= sx < img.shape[1]:
                col = img[y0:max(0, yb), sx].astype(np.int16)
                bar &= ~((col.min(axis=1) >= 85) & (col.max(axis=1) - col.min(axis=1) <= 30))
        na, nb = int(board.line_y - 0.22 * L - oy) - y0, int(board.line_y - 0.09 * L - oy) - y0
        strip = bar[max(0, na):max(0, nb)]
        bar_near[lane] = float(strip.mean()) if strip.size else 0.0
        holds[lane] = False
        # special notes (instrument abilities): a blue icon tile ...
        blue = (b >= 160) & (b - r >= 60) & (b - g >= 20)
        blue[max(0, blue_end):] = False
        rows = (white | blue).sum(axis=1) >= max(4, half)
        runs = [(a, e) for a, e in detect._runs(rows) if e - a >= max(4, 0.03 * L)]
        # ... and a thin white sword, narrower than a tile but tall
        thin = white.sum(axis=1) >= max(2, int(0.08 * L / 4))
        runs += [(a, e) for a, e in detect._runs(thin) if e - a >= 0.15 * L]
        if runs:
            a, e = max(runs, key=lambda s: s[1])          # the lowest one
            out[lane] = y0 + (a + e) / 2 + oy
            above = bar[max(0, a - int(0.1 * L)):max(0, a - 2)]
            holds[lane] = above.size >= 4 and float(above.mean()) >= 0.7
        else:
            out[lane] = None
    return out, holds, bar_near


class KeysFinder:
    """region() to grab, find() -> (Board, {lane: tile_y or None}) or None."""

    def __init__(self):
        self.board = None
        self.misses = 0
        self.holds = {}         # column -> the next tile is a hold note
        self.bar_near = {}      # column -> hold bar coverage just above the slot row

    def region(self, w, h):
        if self.board:
            b = self.board
            return _clamp(b.x0 - 0.1 * b.L, b.line_y - 1.35 * b.L, b.x0 + 1.1 * b.L, b.line_y + 0.1 * b.L, w, h)
        return _clamp(0.2 * w, 0.1 * h, 0.8 * w, 0.97 * h, w, h)

    def find(self, img, ox, oy, w, h):
        board = find_board(img, ox, oy, w)
        if board is None:
            self.misses += 1
            if self.misses >= 90:
                self.board = None
            if self.board is None:
                return None
            board = self.board
        else:
            self.misses = 0
            self.board = board
        tiles, self.holds, self.bar_near = next_tiles(img, ox, oy, board)
        return board, tiles
