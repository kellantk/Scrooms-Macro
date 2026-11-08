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
    """For each column: y (window px) of the lowest tile's center, or None."""
    out = {}
    L = board.L
    ya = int(board.line_y - TOP * L - oy)
    yb = int(board.line_y - 0.01 * L - oy)          # stop above the bottom line
    half = int(0.33 * L / 4)
    for k, lane in enumerate(LANES):
        x = int(board.col_x(k) - ox)
        m = img[max(0, ya):max(0, yb), max(0, x - half):x + half].min(axis=2) >= 200
        rows = m.sum(axis=1) >= max(4, half)
        runs = [(a, e) for a, e in detect._runs(rows) if e - a >= max(4, 0.03 * L)]
        out[lane] = (max(0, ya) + (runs[-1][0] + runs[-1][1]) / 2 + oy) if runs else None
    return out


class KeysFinder:
    """region() to grab, find() -> (Board, {lane: tile_y or None}) or None."""

    def __init__(self):
        self.board = None
        self.misses = 0

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
        return board, next_tiles(img, ox, oy, board)
