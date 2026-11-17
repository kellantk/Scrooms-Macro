"""Practice range for the macro: fake SCROOMS minigames it can really play.

The bot only talks to an `io` object, so here the io is a small simulated
game: it draws frames that look like the real minigames (same sizes, colors
and quirks the detectors were built on), moves the notes on a virtual clock,
takes the bot's clicks and key presses, and knows exactly when each note
*should* have been hit. That gives what recordings can't: how many ms off
every single press was, at any window size, frame rate, PC speed, input delay
and scenery.

Nothing here ships in the app - it's for testing changes to bot.py.
"""
import math

import numpy as np
from PIL import Image


class Done(BaseException):
    """The simulated session is over."""


# sceneries: (base color, texture strength). The bright single-color ones
# are the nasty cases (yellow sand hides the yellow strings circle...)
SCENES = {
    "night": ((14, 22, 60), 10),
    "dusk": ((112, 104, 150), 18),
    "sand": ((246, 204, 72), 8),
    "grass": ((60, 205, 62), 14),
    "lava": ((228, 82, 58), 14),
    "snow": ((238, 242, 247), 5),
}


def background(kind, w, h, seed=0):
    """A textured backdrop, uint8 (h, w, 3)."""
    base, amp = SCENES[kind]
    rng = np.random.default_rng(seed)
    small = rng.normal(0, 1, (max(2, h // 40), max(2, w // 40))).astype(np.float32)
    tex = np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR), dtype=np.float32)
    fine = rng.normal(0, 0.25, (h, w)).astype(np.float32)
    img = np.asarray(base, np.float32)[None, None, :] + (tex + fine)[..., None] * amp
    return np.clip(img, 0, 255).astype(np.uint8)


class SimIO:
    """The bot's window, mouse and keyboard - all simulated.

    fps         the game's frame rate: the screen only changes on frames and
                the game only notices input on frames
    loop_ms     what one screen read + analysis costs the bot (a slow PC: 30+)
    input_ms    click/key -> the game gets it
    display_ms  game state -> visible on screen
    hiccup      chance per read of a 20-60 ms stall (other programs, GC...)
    """

    def __init__(self, game, w, h, fps=60, loop_ms=7.0, input_ms=12.0, display_ms=20.0,
                 hiccup=0.01, seed=0, t_end=120.0):
        self.game, self.w, self.h = game, w, h
        self.fps, self.loop_ms, self.input_ms, self.display_ms = fps, loop_ms, input_ms, display_ms
        self.hiccup = hiccup
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.t_end = t_end
        self.cursor = (w // 2, h // 2)
        self._tick = None
        self._frame = None
        self.grabs = 0

    # --- clock
    def now(self):
        self.t += 2e-5
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.0) + abs(self.rng.normal(0, 0.0004))
        self._check()

    def wait_until(self, deadline):
        self.t = max(self.t, deadline) + 5e-5
        self._check()

    def _check(self):
        self.game.advance(self.t)
        if self.t > self.t_end or self.game.finished(self.t):
            raise Done

    # --- screen
    def client_size(self):
        return self.w, self.h

    def focused(self):
        return True

    def prepare(self, cfg):
        pass

    def grab(self, x0, y0, x1, y1):
        cost = self.loop_ms * (0.85 + 0.3 * self.rng.random()) / 1000
        if self.rng.random() < self.hiccup:
            cost += self.rng.uniform(0.02, 0.06)
        t_cap = self.t + 0.15 * cost           # the screen is copied early on; the rest is analysis
        self.t += cost
        self._check()
        tick = math.floor((t_cap - self.display_ms / 1000) * self.fps)
        if tick != self._tick:
            self._tick = tick
            self._frame = self.game.frame(tick / self.fps)
        self.grabs += 1
        return self._frame[y0:y1, x0:x1]

    # --- input: the game gets it input_ms later, on its next frame
    def _send(self, kind, arg=None):
        t = self.t + (self.input_ms + self.rng.normal(0, 0.8)) / 1000
        t = math.ceil(t * self.fps) / self.fps
        self.game.queue(t, kind, arg)

    def mouse_down(self): self._send("down", "left")
    def mouse_up(self): self._send("up", "left")
    def button_down(self, b): self._send("down", b)
    def button_up(self, b): self._send("up", b)
    def key_down(self, k): self._send("kdown", k.lower())
    def key_up(self, k): self._send("kup", k.lower())

    def key(self, k):
        self._send("kdown", k.lower())
        self.sleep(0.05)
        self._send("kup", k.lower())

    def move_to(self, x, y):
        self.cursor = (x, y)
        self._send("move", (x, y))


def stats(errs):
    """Summary of timing errors (ms)."""
    if not len(errs):
        return {"n": 0, "mean": 0.0, "std": 0.0, "p95": 0.0, "worst": 0.0}
    e = np.asarray(errs, float)
    return {"n": len(e), "mean": float(e.mean()), "std": float(e.std()),
            "p95": float(np.percentile(np.abs(e - e.mean()), 95)), "worst": float(np.abs(e).max())}
