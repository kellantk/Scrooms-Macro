"""Keys family minigame: Piano-Tiles style, 4 columns (keys A S D F).

White tiles fall straight down onto a row of lettered keys; press a column's
key when its tile lies on that key. The board is a frame with 5 evenly spaced
vertical lines and a white horizontal line along the top of the key row.
Sizes below are fractions of the board width L (measured on 3440x1440 and
1920x1080 recordings: L = a third of the window height).

Where to hit, measured on recordings with the game's own Perfect/Good labels:
a tile taken while it's still in the row of empty boxes above the white line
only gets "Good" (and nothing at all when the tiles are fast); the Perfect
ones were taken with the tile on the lettered key, just below the line.
"""
from dataclasses import dataclass

import numpy as np

import detect

LANES = ("A", "S", "D", "F")
KEY = 0.05           # where a tile's middle is hit: on the lettered key row (which spans 0 .. 0.134
#                      below the white line). The Perfect hits on recordings had the tile's middle
#                      0.03-0.06 below the line; one at ~0.02-0.03 only got Good
TILE_H = 0.12        # a tile's height
TOP = 1.28           # board top, above the white line


@dataclass
class Board:
    x0: float
    L: float
    line_y: float

    @property
    def cy(self):          # where a tile's middle is when it's hit: on its key
        return self.line_y + KEY * self.L

    @property
    def r(self):           # a "note size" used by the shared lane logic
        return 0.1 * self.L

    @property
    def hold_end(self):    # a hold bar's top end is this far above cy when the hold is over
        return TILE_H * self.L / 2

    def col_x(self, k):
        return self.x0 + (k + 0.5) * self.L / 4


def _clamp(x0, y0, x1, y1, w, h):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), w), min(int(y1), h)
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _vertical_line(img, x, ya, yb, search, inside=0):
    """Fraction of rows where there's a thin bright vertical line near x.
    inside: +1 / -1 for the board's left / right edge - those are compared
    with the board's (dimmed) inside only. What's outside is the scenery: on
    snow or a pale sky it's as bright as the line, and the board wasn't found."""
    H, W = img.shape[:2]
    # every column within `search` of x, in one go (c = the line, compared
    # with the columns 4 px to its left and right)
    lo, hi = max(int(x) - search, 5), min(int(x) + search, W - 6)
    if hi < lo or yb <= ya:
        return 0.0
    n = hi - lo + 1
    v = img[ya:yb, lo - 5:hi + 6].max(axis=2).astype(np.int16)
    c, left, right = v[:, 5:5 + n], v[:, 1:1 + n], v[:, 9:9 + n]
    if inside > 0:
        ridge = c - right
    elif inside < 0:
        ridge = c - left
    else:
        ridge = c - np.maximum(left, right)
    return float((ridge >= 15).mean(axis=0).max())


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
        if all(_vertical_line(img, ln.x0 + k * L / 4, ya, yb, search, (1, 0, 0, 0, -1)[k]) >= 0.6
               for k in range(5)):
            return Board(ln.x0 + ox, L, ln.y + oy)
    return None


def board_there(img, ox, oy, board):
    """Is the board still standing where it was found? Its five vertical
    lines are there and so is (most of) the white line. Unlike find_board
    this doesn't mind a tile or a hold bar lying across the white line - and
    it fails as soon as the board moves or shrinks (the minigame closing)."""
    H, W = img.shape[:2]
    L = board.L
    x0, ly = board.x0 - ox, board.line_y - oy
    ya, yb = int(ly - 1.15 * L), int(ly - 0.2 * L)
    if ya < 0 or yb > H or x0 < 0 or x0 + L > W or ly + 3 > H:
        return False
    search = max(2, int(0.01 * L))
    if not all(_vertical_line(img, x0 + k * L / 4, ya, yb, search, (1, 0, 0, 0, -1)[k]) >= 0.6 for k in range(5)):
        return False
    row = img[int(ly) - 2:int(ly) + 3, int(x0):int(x0 + L)].max(axis=2).max(axis=0)
    return float((row >= 150).mean()) >= 0.4


def next_tiles(img, ox, oy, board):
    """Returns (tiles, holds, bar_near, bar_top), each keyed by column:
      tiles:    y (window px) of the lowest tile's middle, or None
      holds:    True when that tile is a hold note (a grey bar trails above it)
      bar_near: how much of the strip just above the key row a hold bar covers
      bar_top:  y of the top end of the lowest hold bar, or None
    A tile is placed by its top edge: that edge is sharp and stays in view
    all the way down, while the tile's lower part is shaded darker and ends
    up over the line and the key's letter."""
    out, holds, bar_near, bar_top = {}, {}, {}, {}
    L = board.L
    ya = int(board.line_y - TOP * L - oy)
    yb = int(board.line_y + 0.14 * L - oy)          # through the key row
    half = int(0.33 * L / 4)
    y0 = max(0, ya)
    line = int(board.line_y - 0.01 * L - oy) - y0     # rows from here on: the white line and the keys
    for k, lane in enumerate(LANES):
        x = int(board.col_x(k) - ox)
        sub = img[y0:max(0, yb), max(0, x - half):x + half]
        mx, mn = sub.max(axis=2), sub.min(axis=2)
        white = mn >= 200
        # a hold note's bar: see-through grey, narrower than a tile - grey in
        # the middle of the column and clearly brighter there than at the
        # column's sides (which show the board's dim backdrop; over snow that
        # backdrop is grey too, just darker)
        grey = (mn >= 85) & (mn < 200) & (mx - mn <= 30)
        c = grey.shape[1] // 2
        q = max(2, int(0.18 * L / 4))
        bar = grey[:, c - q:c + q].mean(axis=1) >= 0.8
        mid = mn[:, c - q:c + q].mean(axis=1)
        # (sides: between the bar's edge at 0.3 of the column and the outlines
        # of the boxes it passes over at ~0.45)
        for sx in (x - int(round(0.37 * L / 4)), x + int(round(0.37 * L / 4))):
            if 0 <= sx < img.shape[1]:
                bar &= mid - img[y0:max(0, yb), sx].min(axis=1).astype(np.int16) >= 18
        na, nb = int(board.line_y - 0.13 * L - oy) - y0, line
        strip = bar[max(0, na):max(0, nb)]
        bar_near[lane] = float(strip.mean()) if strip.size else 0.0
        holds[lane] = False
        # the far (top) end of the lowest hold bar: the hold is over when it
        # gets down to the key
        # (the outlines of the boxes it passes over cut it into pieces: join them)
        bars = []
        for a, e in detect._runs(bar):
            if bars and a - bars[-1][1] <= max(3, 0.02 * L):
                bars[-1][1] = e
            else:
                bars.append([a, e])
        bars = [s for s in bars if s[1] - s[0] >= 0.04 * L]
        low = max(bars, key=lambda s: s[1]) if bars else None
        # (a bar that runs out of the top of the board: its end isn't in view yet)
        bar_top[lane] = y0 + low[0] + oy if low is not None and low[0] > 1 else None
        # special notes (instrument abilities): a blue icon tile ...
        b = sub[..., 2]
        blue = b >= 160
        if blue.any():
            b = b.astype(np.int16)
            blue &= (b - sub[..., 0] >= 60) & (b - sub[..., 1] >= 20)
            blue[max(0, line):] = False
        rows = (white | blue).sum(axis=1) >= max(4, half)
        # (top edge, middle) of every tile; the white line itself is too thin to count
        tiles = [(a, a + TILE_H * L / 2) for a, e in detect._runs(rows) if e - a >= max(4, 0.03 * L)]
        # ... and a thin white sword, narrower than a tile but tall
        thin = white.sum(axis=1) >= max(2, int(0.08 * L / 4))
        thin[max(0, line):] = False
        tiles += [(a, (a + e) / 2) for a, e in detect._runs(thin) if e - a >= 0.15 * L]
        if tiles:
            a, mid = max(tiles)                          # the lowest one
            out[lane] = y0 + mid + oy
            above = bar[max(0, a - int(0.1 * L)):max(0, a - 2)]
            holds[lane] = above.size >= 4 and float(above.mean()) >= 0.7
        else:
            out[lane] = None
    return out, holds, bar_near, bar_top


class KeysFinder:
    """region() to grab, find() -> (Board, {lane: tile_y or None}) or None."""

    def __init__(self):
        self.board = None
        self.misses = 0
        self.changed = 0        # reads in a row that showed a different board
        self.reads = 0
        self.present = False    # the board was really there in the last read (not just remembered)
        self.holds = {}         # column -> the next tile is a hold note
        self.bar_near = {}      # column -> hold bar coverage just above the key row
        self.bar_top = {}       # column -> y of the top end of the lowest hold bar
        self.label = None       # pixel counts of the verdict text ("Perfect"...) left of the board
        self.label_size = 1.0

    def region(self, w, h):
        if self.board:
            b = self.board
            return _clamp(b.x0 - 0.45 * b.L, b.line_y - 1.35 * b.L, b.x0 + 1.1 * b.L, b.line_y + 0.18 * b.L, w, h)
        return _clamp(0.2 * w, 0.1 * h, 0.8 * w, 0.97 * h, w, h)

    def find(self, img, ox, oy, w, h):
        # Standing where it was (its five vertical lines and the white line
        # are there)? Then it isn't searched for again - the search is half
        # the cost of a read - except now and then, to follow it if it was
        # first caught a pixel or two off while still sliding in.
        there = self.board is not None and board_there(img, ox, oy, self.board)
        self.reads += 1
        board = None
        if not there or self.reads % 20 == 0:
            board = find_board(img, ox, oy, w)
            if board is not None and self.board is not None:
                # the board doesn't move while tiles fall. A tile lying on the white
                # line cuts the line short for a moment, which reads as a narrower
                # board: don't believe a different board until it has stayed
                # (the window was resized)
                old = self.board
                if (abs(board.L - old.L) > 0.02 * old.L or abs(board.x0 - old.x0) > 0.02 * old.L
                        or abs(board.line_y - old.line_y) > 0.02 * old.L):
                    self.changed += 1
                    if self.changed < 25:
                        board = None
        if board is None and there:
            board = self.board
        # (a tile or a hold bar across the white line hides the board from the
        # search for up to a second or more; `there` still sees it. Neither:
        # it has moved or gone - the minigame is closing)
        self.present = board is not None
        if board is None:
            self.misses += 1
            if self.misses >= 90:
                self.board = None
            if self.board is None:
                return None
            board = self.board
        else:
            self.misses = 0
            self.changed = 0
            self.board = board
        tiles, self.holds, self.bar_near, self.bar_top = next_tiles(img, ox, oy, board)
        # the game's verdict for the last tile shows left of the board, halfway up
        L = board.L
        self.label_size = 0.1 * L
        self.label = detect.label_counts(
            img[max(0, int(board.line_y - 0.58 * L - oy)):max(0, int(board.line_y - 0.43 * L - oy)),
                max(0, int(board.x0 - 0.42 * L - ox)):max(0, int(board.x0 - 0.02 * L - ox))], 0.1 * L)
        return board, tiles
