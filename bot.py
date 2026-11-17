"""Scroom catching bot: state machine + controllers.

The Bot talks to the outside world only through an `io` object, so the same
logic runs live (WindowsIO) or against a recorded video (see replay.py).

Phases (learned from recordings + input logs):
  CAST   click once with the guitar equipped
  RING   needle spins ~295 deg/s; click when it's on the green ("Perfect") part
  FREQ   hold = pin accelerates right, release = accelerates left; park it on
         the glowing dot until the panel closes
  REEL   hold = the "< o >" dot accelerates right, release = left; the scroom
         chases the dot; keep the scroom inside the moving gray box
"""
import ctypes
import os
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import countin
import detect
import notify
import keys
import percussion
import strings
import wind


@dataclass
class Config:
    equip_key: str = "1"          # hotbar key for Tamer's Guitar ("" = don't press)
    equip_on_start: bool = True
    latency_ms: int = 60          # input -> screen delay to compensate for
    auto_latency: bool = True     # learn latency_ms from where ring clicks actually land
    note_offset_ms: int = 0       # strings/keys: press this much later (+) or earlier (-) than the
                                  # usual timing (auto-tuned from the strings circles lighting up)
    auto_note_timing: bool = True # learn note_offset_ms from when the strings circles light up
    ring_timeout_s: float = 5.0   # ring up longer than this -> re-equip instrument and recast
    strings_keys: str = "ASD"     # your in-game keybinds for the strings lanes, left to right
    keys_keys: str = "ASDF"       # ... and for the keys (piano) columns
    count_in: bool = True        # click the head-start rings before the minigame
    aim_perfect: bool = True      # aim for the darker "Perfect" chunk of the ring
    cooldown_s: float = 2.5       # wait after a catch before casting again
    max_catches: int = 0          # 0 = run until stopped
    move_cursor: bool = True      # park the cursor somewhere safe before starting
    # physics (fitted from your recordings)
    freq_accel_right: float = 2650
    freq_accel_left: float = 2500
    reel_accel_right: float = 670
    reel_accel_left: float = 580
    reel_follow: float = 4.6      # scroom speed = follow * (dot - scroom)
    dry_run: bool = False         # detect only, never click
    # Discord
    webhook_url: str = ""
    notify_catches: bool = True   # post each finished catch (with a screenshot of the result)
    report_minutes: int = 15      # session report every N minutes (0 = off)
    discord_user_id: str = ""     # who to ping (Discord user ID)
    ping_on_stop: bool = True     # ping them when the macro stops by itself or gets stuck


@dataclass
class Status:
    phase: str = "Stopped"
    detail: str = ""
    catches: int = 0
    escapes: int = 0
    casts: int = 0
    fps: float = 0.0
    log_path: str = ""            # also append log lines to this file
    debug_dir: str = ""           # save a screenshot here when something goes wrong
    log: deque = field(default_factory=lambda: deque(maxlen=200))
    finished: int = 0             # minigames played to the end
    per_minigame: dict = field(default_factory=dict)
    session_start: float = 0.0
    graph: dict = field(default_factory=dict)    # live data for the app's graph


class Tracker:
    """Position + velocity estimate from the last few timestamped samples."""

    def __init__(self, window=0.09):
        self.window = window
        self.samples = deque()

    def add(self, t, x):
        self.samples.append((t, x))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()

    def reset(self):
        self.samples.clear()

    @property
    def x(self):
        return self.samples[-1][1] if self.samples else None

    @property
    def v(self):
        if len(self.samples) < 2:
            return 0.0
        t = np.array([s[0] for s in self.samples])
        x = np.array([s[1] for s in self.samples])
        t = t - t.mean()
        d = (t * t).sum()
        return float((t * (x - x.mean())).sum() / d) if d > 0 else 0.0


def momentum_hold(x, v, target, target_v, accel_right, accel_left, lead):
    """Bang-bang control for an object that accelerates right while held and
    left while released. Returns True to hold."""
    xp = x + v * lead
    e = (target + target_v * lead) - xp
    rv = v - target_v
    if e >= 0:   # target is to the right: hold, unless we must brake (release)
        stop = rv * rv / (2 * accel_left) if rv > 0 else 0.0
        return stop < e
    stop = rv * rv / (2 * accel_right) if rv < 0 else 0.0
    return stop >= -e   # moving left fast enough to overshoot -> hold to brake


def _save_png(path, rgb):
    """Minimal PNG writer (no extra libraries)."""
    import struct
    import zlib
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class Motion:
    """How notes move in one minigame, so a note's arrival is told from all
    of its readings together.

    Keys tiles fall at a steady speed. Strings notes speed up on the way down
    (the highway is drawn in perspective); measured on recordings, 1/(y - yv)
    changes steadily with time, yv being a fixed point ~20 circle-radii above
    the circles. Either way there is a coordinate u(y) that is a straight line
    in time, and every note of one minigame climbs it at the same rate. So
    after a couple of notes a new one only needs its readings averaged, where
    fitting a curve through a handful of noisy points (what this replaced)
    made presses wander by 10-40 ms on slow PCs and came out late on average."""

    def __init__(self, y_target, v_default, yv=None):
        self.yv = yv
        self.u_target = self.u(y_target)
        # the rate a note has when it moves at v_default px/s at the hit spot
        self.rate0 = v_default if yv is None else v_default / (y_target - yv) ** 2
        self.rates = deque(maxlen=6)      # of the notes that went by

    def u(self, y):
        return y if self.yv is None else -1.0 / max(y - self.yv, 1.0)

    def y(self, u):
        return u if self.yv is None else self.yv - 1.0 / min(u, -1e-9)

    def rate(self):
        """The rate the notes share - once two of them have shown it."""
        return float(np.median(self.rates)) if len(self.rates) >= 2 else None

    def _line(self, samples):
        """(rate, u at the last reading) of one note."""
        t = np.array([s[0] for s in samples])
        u = np.array([self.u(s[1]) for s in samples])
        t = t - t[-1]
        k = self.rate()
        if k is None:
            k = self.rate0
            if len(t) >= 3 and t[-1] - t[0] >= 0.08:
                kf = float(np.polyfit(t, u, 1)[0])
                if 0.3 * self.rate0 < kf < 4 * self.rate0:
                    k = kf
        res = u - k * t
        c = float(np.mean(res))
        if len(res) >= 5:                 # leave out a stray misread or two
            mad = float(np.median(np.abs(res - np.median(res))))
            keep = np.abs(res - np.median(res)) <= 3 * 1.4826 * mad + 1e-12
            if keep.sum() >= 3:
                c = float(np.mean(res[keep]))
        return k, c

    def eta(self, samples, now, y_target=None):
        """Seconds until the note's middle is on the hit spot (or at y_target)."""
        if not samples:
            return float("inf")
        k, c = self._line(samples)
        u_target = self.u_target if y_target is None else self.u(y_target)
        return (u_target - c) / k - (now - samples[-1][0])

    def pos(self, samples, now):
        k, c = self._line(samples)
        return self.y(c + k * (now - samples[-1][0]))

    def learn(self, samples):
        """A note is done: remember how fast it went."""
        if len(samples) < 5 or samples[-1][0] - samples[0][0] < 0.2:
            return
        t = np.array([s[0] for s in samples])
        u = np.array([self.u(s[1]) for s in samples])
        k = float(np.polyfit(t - t[-1], u, 1)[0])
        if not 0.3 * self.rate0 < k < 4 * self.rate0:
            return
        if self.rates and abs(k - np.median(self.rates)) > 0.15 * np.median(self.rates):
            self.rates.clear()            # the notes changed speed: start over
        self.rates.append(k)


def commit_ahead(frame_dt):
    """How far ahead of its moment a press is handed to the precise timer.
    Deciding only on the very last screen read meant one slow read (another
    program hogging the PC for 50 ms) made the press late; a little earlier,
    the wait is then timed exactly and nothing can get in between."""
    return max(2.5 * frame_dt, 0.03)


# strings: where to land a press, as "the circle lights up this long after the
# note's middle crosses it". Measured against the game's own labels on five
# recordings (40 presses): every press from -45 to +50 ms got Perfect, and the
# Goods all came in late (+50 and up). So: the middle of the safe stretch.
NOTE_TARGET_MS = 0
HOLD_EXTRA = 0.03       # hold notes: keep the key down this long past the bar's end (letting go early breaks the hold)
STRINGS_YV = 20.3       # strings: the point the notes' motion runs away from, in circle radii above the circles
STUCK_MINUTES = 10      # no catch for this long -> tell Discord


class NoteTimer:
    """Measures how each strings press landed: when the note's white core
    crosses the circle's center vs when the circle lights up (the game took
    the key). Both come from the same screen captures, so the capture delay
    cancels out. A press that comes early takes the note before it gets to
    the circle; its crossing is then worked out from where it was heading."""

    def __init__(self, lanes, motion=None):
        self.motion = motion
        self.prev = {k: None for k in lanes}               # (t, core y) last read
        self.core = {k: deque() for k in lanes}            # the core's recent readings
        self.cross = {k: deque(maxlen=8) for k in lanes}   # crossing times
        self.pressed = {k: deque(maxlen=8) for k in lanes}
        self.pending = {k: deque() for k in lanes}         # (lit time, expected crossing) to be paired
        self.lit = {k: False for k in lanes}
        self.unlit_t = {k: None for k in lanes}            # last read with the circle dark
        self.errors = []                                   # ms, lit - crossing

    def press(self, lane, t):
        self.pressed[lane].append(t)

    def rebase(self, step_ms):
        """The timing was just moved by step_ms: what was measured before
        would have come out that much different."""
        self.errors = [e + step_ms for e in self.errors]

    def _heading(self, lane, cy, r, now):
        """When the core we were watching would have crossed the center."""
        core = [s for s in self.core[lane] if s[1] < cy]
        if self.motion is None or len(core) < 3 or now - core[-1][0] > 0.15 or core[-1][1] < cy - 2.6 * r:
            return None
        return core[-1][0] + self.motion.eta(core, core[-1][0])

    def update(self, marks, cy, r, now):
        for lane, (wy, lit) in marks.items():
            if lane not in self.prev:
                continue
            p = self.prev[lane]
            if wy is not None and p is not None and p[1] < cy <= wy and wy - p[1] < 1.5 * r and now > p[0]:
                self.cross[lane].append(p[0] + (cy - p[1]) / (wy - p[1]) * (now - p[0]))
            self.prev[lane] = (now, wy) if wy is not None else None
            core = self.core[lane]
            if wy is not None:
                if core and (wy < core[-1][1] - 0.5 * r or now - core[-1][0] > 0.25):
                    core.clear()                           # another note
                if not core or abs(wy - core[-1][1]) > 0.5:
                    core.append((now, wy))
                while core and now - core[0][0] > 0.5:
                    core.popleft()
            # a press fills the whole circle with no note left in it (a note
            # sliding over the circle only covers ~0.6-0.7 of it)
            on = lit > 0.9 and (wy is None or wy < cy - 1.2 * r)
            if on and not self.lit[lane]:
                u = self.unlit_t[lane]
                # lit somewhere between the last dark read and this one
                t_lit = (u + now) / 2 if u is not None and now - u < 0.05 else now
                self.pending[lane].append((t_lit, self._heading(lane, cy, r, now)))
                core.clear()
            if not on:
                self.unlit_t[lane] = now
            self.lit[lane] = on
            # a late press lights the circle after the note crossed: wait a bit
            pend = self.pending[lane]
            while pend and now - pend[0][0] > 0.15:
                self._pair(lane, *pend.popleft())

    def _pair(self, lane, t_lit, heading):
        if not any(-0.05 < t_lit - tp < 0.35 for tp in self.pressed[lane]):
            return          # not one of our presses
        c = [x for x in self.cross[lane] if -0.1 < t_lit - x < 0.3]
        if c:
            x = min(c, key=lambda x: abs(t_lit - x - NOTE_TARGET_MS / 1000))
        elif heading is not None and -0.2 < t_lit - heading < 0.1:
            x = heading
        else:
            return
        self.errors.append((t_lit - x) * 1000)


def _circ_center(a, tol=math.radians(15)):
    """Robust center of a set of angles: the value with the most neighbours
    within `tol`, then the mean of those neighbours."""
    d = np.abs((a[:, None] - a[None, :] + np.pi) % (2 * np.pi) - np.pi)
    best = int((d < tol).sum(axis=1).argmax())
    inl = d[best] < tol
    return float(np.arctan2(np.sin(a[inl]).mean(), np.cos(a[inl]).mean()))


def fit_spin(obs, t0, strict=True):
    """Fit needle(t) = phase + omega*(t - t0) to noisy readings.
    Returns (omega rad/s, phase rad) or None if there isn't a clean fit yet.
    strict=False is the fallback when a clean fit is taking too long."""
    # accuracy beats speed here: the ring waits for the click, so get a solid fit
    if len(obs) < (8 if strict else 5) or obs[-1][0] - obs[0][0] < (0.4 if strict else 0.15):
        return None
    ts = np.array([o[0] for o in obs]) - t0
    a = np.array([o[1] for o in obs])
    # speed over ~80 ms spans, not between back-to-back readings: we read the
    # screen faster than the game draws, so neighbours are often the same frame
    j = np.searchsorted(ts, ts + 0.08)
    ok = j < len(ts)
    i = np.flatnonzero(ok)
    j = j[ok]
    dt = ts[j] - ts[i]
    da = (a[j] - a[i] + np.pi) % (2 * np.pi) - np.pi
    keep = (dt > 0.05) & (dt < 0.2)
    rates = da[keep] / dt[keep]
    rates = rates[(np.abs(rates) > math.radians(150)) & (np.abs(rates) < math.radians(500))]
    if len(rates) < 4:
        return None
    sign = np.sign(np.median(rates))
    if (np.sign(rates) == sign).mean() < 0.7:
        return None
    omega = float(np.median(rates[np.sign(rates) == sign]))
    # the needle always spins ~250-310 deg/s
    if not math.radians(180) <= abs(omega) <= math.radians(440):
        return None
    # position: residuals after removing the spin should cluster at one angle.
    # Refine speed + position a few times (a slightly-off speed makes older
    # readings drift), tightening the outlier cut each pass.
    center = None
    for tol in (30, 20, 12):
        res = (a - omega * ts + np.pi) % (2 * np.pi) - np.pi
        if center is None:
            center = _circ_center(res, math.radians(20))
        dev = (res - center + np.pi) % (2 * np.pi) - np.pi
        inl = np.abs(dev) < math.radians(tol)
        if inl.sum() < 4:
            return None
        if np.ptp(ts[inl]) > 0.1:
            slope, icpt = np.polyfit(ts[inl], dev[inl], 1)
            omega += slope
            center += icpt
    res = (a - omega * ts + np.pi) % (2 * np.pi) - np.pi
    dev = (res - center + np.pi) % (2 * np.pi) - np.pi
    inl = np.abs(dev) < math.radians(12)
    if inl.sum() < (max(8, 0.7 * len(res)) if strict else max(4, 0.5 * len(res))):
        return None
    if not math.radians(180) <= abs(omega) <= math.radians(440):
        return None
    return omega, float(center)

class Bot:
    def __init__(self, cfg: Config, io, status: Status | None = None):
        self.cfg = cfg
        self.io = io
        self.st = status or Status()
        self.held = False
        self.last_click_t = 0.0
        self.landings = 0
        self.delays = deque(maxlen=7)  # recent click -> registered delays (ms)
        self._last_report = time.time()
        self.stop_reason = None        # why it stopped by itself (None = you stopped it)
        self._last_progress = time.time()
        self._stuck_alerted = False
        self.press_log = deque(maxlen=40)
        self._last_shot = {}
        self._hold_start = None
        self._last_repress = -9.0
        self.stop_event = threading.Event()
        self.ring_finder = detect.RingFinder()
        self.freq_finder = detect.FreqFinder()
        self.reel_finder = detect.ReelFinder()
        self.strings_finder = strings.StringsFinder()
        self.perc_finder = percussion.PercussionFinder()
        self.keys_finder = keys.KeysFinder()
        self.wind_finder = wind.WindFinder()
        self.countin_finder = countin.CountInFinder()

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.st.log.append(line)
        self._write_log(line)

    def snapshot(self, reason):
        """Save what the game looked like when something went wrong, so a bug
        report shows the actual problem. Keeps the newest 30."""
        if not self.st.debug_dir:
            return
        now = self.io.now()
        if now - self._last_shot.get(reason, -99) < 10:   # don't spam the same problem
            return
        self._last_shot[reason] = now
        try:
            w, h = self.io.client_size()
            img = self.io.grab(0, 0, w, h)
            if w > 1700:                       # keep files small
                img = img[::2, ::2]
            os.makedirs(self.st.debug_dir, exist_ok=True)
            name = time.strftime("%Y%m%d-%H%M%S") + f"{time.time() % 1:.3f}"[1:] + "_" + "".join(ch if ch.isalnum() else "-" for ch in reason)[:40]
            _save_png(os.path.join(self.st.debug_dir, name + ".png"), np.ascontiguousarray(img))
            shots = sorted(f for f in os.listdir(self.st.debug_dir) if f.endswith(".png"))
            for old_f in shots[:-30]:
                os.remove(os.path.join(self.st.debug_dir, old_f))
            self._write_log(f"(saved debug screenshot {name}.png)")
        except Exception:
            pass

    def log_traceback(self):
        import traceback
        self._write_log(traceback.format_exc())

    def _write_log(self, text):
        if not self.st.log_path:
            return
        try:
            with open(self.st.log_path, "a", encoding="utf-8") as f:
                f.write(text.rstrip("\n") + "\n")
        except OSError:
            pass

    def set_phase(self, phase, detail=""):
        if phase != self.st.phase:
            self.log(phase + (f" - {detail}" if detail else ""))
        self.st.phase, self.st.detail = phase, detail

    def hold(self, on):
        if self.cfg.dry_run:
            self.held = on
            return
        if on and not self.held:
            self.io.mouse_down()
            self._hold_start = None
        elif not on and self.held:
            self.io.mouse_up()
            self._hold_start = None
        self.held = on

    def check_hold_response(self, v, accel, top_speed):
        """We think we're holding - is the game actually responding? While held,
        the pin/dot speeds up to the right. If it doesn't, the hold was lost (a
        real click or a dropped event released the button behind our back, which
        used to leave the pin stuck against the left wall), so press again.
        Judged from the game itself - no keyboard/mouse state polling."""
        if self.cfg.dry_run or not self.held:
            self._hold_start = None
            return
        now = self.io.now()
        if self._hold_start is None:
            self._hold_start = (now, v)
            return
        t0, v0 = self._hold_start
        dt = now - t0
        if dt < 0.35:
            return
        # expected speed-up is accel*dt (minus drag); a quarter of it, while still
        # well under top speed, means nothing is happening
        if v - v0 < 0.25 * accel * dt and v < 0.5 * top_speed and now - self._last_repress > 1.0:
            self.log("Hold isn't registering - pressing again")
            self.snapshot("hold-lost")
            self.io.mouse_up()
            self.io.mouse_down()
            self._last_repress = now
        self._hold_start = (now, v)

    def click(self):
        if self.cfg.dry_run:
            return
        self.hold(False)
        self.last_click_t = self.io.now()
        self.io.mouse_down()
        self.io.sleep(0.04)
        self.io.mouse_up()

    # ------------------------------------------------- results / discord
    def stats(self):
        return {"finished": self.st.finished, "casts": self.st.casts,
                "uptime": time.time() - (self.st.session_start or time.time()),
                "per_minigame": dict(self.st.per_minigame)}

    def webhook_on(self):
        return bool(self.cfg.webhook_url) and notify.valid_url(self.cfg.webhook_url) and not self.cfg.dry_run

    def minigame_done(self, name, detail):
        """Count it, post it to Discord (with a picture of the result), then
        wait out the cooldown before casting again."""
        self.st.finished += 1
        self.st.per_minigame[name] = self.st.per_minigame.get(name, 0) + 1
        self._last_progress = time.time()
        self._stuck_alerted = False
        self.st.graph = {}
        self.set_phase("Cooldown")
        t0 = self.io.now()
        alerted = not (self.webhook_on() and self.cfg.notify_catches)
        while not self.stopped() and self.io.now() - t0 < max(self.cfg.cooldown_s, 0 if alerted else 1.5):
            if not alerted and self.io.now() - t0 >= 1.2:     # the "You caught ..." banner is up
                alerted = True
                try:
                    w, h = self.io.client_size()
                    shot = notify.png_bytes(np.ascontiguousarray(self.io.grab(0, 0, w, h)))
                except Exception:
                    shot = None
                notify.send_async(self.cfg.webhook_url, notify.catch_embed(name, detail, self.stats()), shot,
                                  lambda ok, msg: None if ok else self.log(f"Discord: couldn't post ({msg})"))
            self.io.sleep(0.05)

    def ping_id(self):
        uid = (self.cfg.discord_user_id or "").strip()
        return uid if self.cfg.ping_on_stop and notify.valid_user_id(uid) else None

    def stop_with(self, reason):
        """Stop by itself (Discord gets told why, with a ping)."""
        if self.stop_reason is None:
            self.stop_reason = reason
        self.log(f"Stopping: {reason}")
        self.stop_event.set()

    def maybe_report(self, final=False):
        if not self.webhook_on():
            return
        now = time.time()
        if final:
            if self.stop_reason:          # stopped by itself: always say so (and ping)
                notify.send_async(self.cfg.webhook_url,
                                  notify.report_embed(self.stats(), final=True, reason=self.stop_reason),
                                  ping=self.ping_id())
            elif self.st.finished and self.cfg.report_minutes > 0:
                notify.send_async(self.cfg.webhook_url, notify.report_embed(self.stats(), final=True))
            return
        if now - self._last_progress >= STUCK_MINUTES * 60 and not self._stuck_alerted:
            self._stuck_alerted = True
            self.log(f"No catch in {STUCK_MINUTES} minutes - telling Discord")
            self.snapshot("stuck")
            notify.send_async(self.cfg.webhook_url,
                              notify.stuck_embed(self.stats(), STUCK_MINUTES, paused=self.st.phase == "Paused"),
                              ping=self.ping_id())
        if self.cfg.report_minutes <= 0:
            return
        if now - self._last_report >= 60 * self.cfg.report_minutes:
            self._last_report = now
            notify.send_async(self.cfg.webhook_url, notify.report_embed(self.stats()))

    def look(self, finder):
        w, h = self.io.client_size()
        x0, y0, x1, y1 = finder.region(w, h)
        img = self.io.grab(x0, y0, x1, y1)
        return finder.find(img, x0, y0, w, h)

    def ring(self):
        return self.look(self.ring_finder)

    def freq(self):
        return self.look(self.freq_finder)

    def strings(self):
        return self.look(self.strings_finder)

    def perc(self):
        return self.look(self.perc_finder)

    def keys(self):
        return self.look(self.keys_finder)

    def wind(self):
        return self.look(self.wind_finder)

    def minigame_up(self):
        """Is any catch minigame on screen already?"""
        return bool(self.strings() or self.keys() or self.freq() or self.perc() or self.wind())

    def reel(self):
        r = self.look(self.reel_finder)
        if r and r[0] is not None:   # need the box to call it the reel phase
            return r
        return None

    def stopped(self):
        return self.stop_event.is_set()

    def wait_focus(self):
        """Pause (mouse released) while Roblox isn't the active window."""
        warned = False
        while not self.stopped() and not self.io.focused():
            if self.held:
                self.hold(False)
            alive = getattr(self.io, "window_alive", None)
            if alive is not None and not alive():
                self.stop_with("Roblox closed (or crashed).")
                return
            if not warned:
                self.set_phase("Paused", "Roblox isn't the active window")
                warned = True
            self.maybe_report()
            self.io.sleep(0.2)

    # --------------------------------------------------------------- main
    def run(self):
        self.stop_event.clear()
        self.st.session_start = time.time()
        self._last_report = self._last_progress = time.time()
        self.stop_reason = None
        try:
            self.io.prepare(self.cfg)
            if self.cfg.equip_on_start and self.cfg.equip_key and not self.cfg.dry_run:
                self.wait_focus()
                self.io.key(self.cfg.equip_key)
                self.io.sleep(0.6)
            errors = 0
            while not self.stopped():
                if self.cfg.max_catches and self.st.finished >= self.cfg.max_catches:
                    self.stop_with(f"Done: reached your goal of {self.cfg.max_catches} catches.")
                    break
                try:
                    self.cycle()
                    errors = 0
                except Exception as e:
                    # never let one weird frame kill the whole macro: log it and keep going
                    self.hold(False)
                    errors += 1
                    self.log(f"Error ({type(e).__name__}: {e}) - recovering")
                    self.log_traceback()
                    if errors >= 5:
                        alive = getattr(self.io, "window_alive", None)
                        self.stop_with("Roblox closed (or crashed)." if alive is not None and not alive()
                                       else f"5 errors in a row ({type(e).__name__}: {e}). Make a bug report.")
                        raise
                    self.io.sleep(1.0)
        finally:
            self.hold(False)
            self.st.graph = {}
            self.maybe_report(final=True)
            self.set_phase("Stopped", self.stop_reason or "")

    def cycle(self):
        """One cast -> ring -> (count-in) -> minigame attempt."""
        import gc
        gc.disable()          # a collection pause in the middle of a note costs a press
        try:
            self._cycle()
        finally:
            gc.enable()

    def _cycle(self):
        self.maybe_report()
        self.wait_focus()
        if self.stopped():
            return
        # already mid-minigame (e.g. started the macro late)? jump straight in
        if self.strings():
            self.do_strings()
            return
        if self.keys():
            self.do_keys()
            return
        if self.freq():
            self.do_freq()
            self.after_freq()
            return
        if self.perc():
            self.do_perc()
            return
        if self.wind():
            self.do_wind()
            return
        if self.reel():
            self.do_reel()
            return

        self.set_phase("Casting")
        self.click()
        self.st.casts += 1
        if not self.do_ring():
            return
        self.after_freq_wait()

    def after_freq_wait(self):
        # frequency panel should slide in within a few seconds
        t0 = self.io.now()
        self.set_phase("Waiting for minigame")
        counted_in = False
        while not self.stopped() and self.io.now() - t0 < 14:
            self.wait_focus()
            # the count-in ("head start" rings) comes before every minigame
            if not counted_in and self.cfg.count_in and self.look(self.countin_finder):
                self.do_countin()
                counted_in = True
                t0 = self.io.now()
                continue
            if self.strings():
                self.do_strings()
                return
            if self.keys():
                self.do_keys()
                return
            if self.freq():
                self.do_freq()
                self.after_freq()
                return
            if self.perc():
                self.do_perc()
                return
            if self.wind():
                self.do_wind()
                return
            if self.reel():
                self.do_reel()
                return
            self.io.sleep(0.02)
        self.snapshot("freq-not-found")
        self.log("No minigame showed up - recasting")

    def after_freq(self):
        t0 = self.io.now()
        self.set_phase("Waiting for reel")
        while not self.stopped() and self.io.now() - t0 < 6:
            self.wait_focus()
            if self.reel():
                self.do_reel()
                return
            if self.freq():   # panel came back - we had lost track of it
                self.log("Lost the frequency panel for a moment - continuing")
                self.snapshot("freq-lost")
                self.do_freq()
                t0 = self.io.now()
            self.io.sleep(0.02)
        self.snapshot("reel-not-found")
        self.log("Reel bar never showed up")

    # --------------------------------------------------------------- ring
    def do_ring(self):
        """The needle spins at a constant speed for as long as the ring is up,
        so instead of trusting each (noisy) reading we fit speed + position
        from the last second of readings, ignoring outliers, and click on the
        *calculated* needle position. Works even while the needle is hard to
        see over the green or a sparkle effect is on top of it."""
        lat = self.cfg.latency_ms / 1000
        t0 = self.io.now()
        last_seen = None
        first_seen = None
        obs = deque()                  # (time, needle angle) readings
        targets = deque(maxlen=15)
        loop_dt = deque(maxlen=10)
        prev_t = None
        while not self.stopped():
            self.wait_focus()
            now = self.io.now()
            if prev_t is not None:
                loop_dt.append(now - prev_t)
            prev_t = now
            r = self.ring()
            if r is None:
                if last_seen is not None and now - last_seen > 0.6:
                    self.log("Ring disappeared before we clicked")
                    return True
                if last_seen is None and now - t0 > 3.5:
                    self.snapshot("ring-not-found")
                    self.log("Ring didn't appear - recasting")
                    return False
                if self.minigame_up():
                    return True
                self.io.sleep(0.01)
                continue
            last_seen = now
            if first_seen is None:
                first_seen = now
            # failsafe: the ring should never take this long - something's stuck,
            # so put the instrument away, take it out again and recast
            if now - first_seen > self.cfg.ring_timeout_s:
                self.log(f"Ring stuck for {self.cfg.ring_timeout_s:.0f}s - re-equipping and recasting")
                self.snapshot("ring-stuck")
                self.reequip()
                return False
            targets.append(r.target if self.cfg.aim_perfect else r.zone_mid)
            if r.needle is not None and (not obs or abs(detect._ang_diff(r.needle, obs[-1][1])) > 1e-3):
                obs.append((now, r.needle))
            while obs and now - obs[0][0] > 1.5:
                obs.popleft()
            model = fit_spin(obs, t0)
            if model is None and now - t0 > 2.5:
                model = fit_spin(obs, t0, strict=False)   # don't leave a ring unclicked
            if model is None:
                self.set_phase("Ring", "measuring the needle...")
                continue
            omega, phase = model
            target = _circ_center(np.array(targets))
            needle = detect._ang_diff(phase + omega * (now - t0), 0.0)
            self.st.graph = {"mode": "ring", "needle": needle, "target": target, "omega": omega, "now": now}
            self.set_phase("Ring", f"needle {math.degrees(needle):4.0f} deg, target {math.degrees(target):4.0f}, "
                                   f"{math.degrees(omega):+.0f} deg/s")
            # time until the needle reaches the target, in its direction of travel
            diff = detect._ang_diff(target, needle)
            if omega < 0:
                diff = -diff
            if diff < 0:
                diff += 2 * math.pi
            t_hit = diff / abs(omega)
            frame_dt = max(sum(loop_dt) / len(loop_dt), 0.008) if loop_dt else 0.03
            wait = t_hit - lat
            # too late to land this pass on the target (the click needs `lat`
            # to get there)? Then it's the next lap, ~1.2 s on: clicking anyway
            # landed 30-70 ms past the Perfect part.
            if wait < -(0.012 if self.cfg.aim_perfect else 0.05):
                wait += 2 * math.pi / abs(omega)
            if wait <= commit_ahead(frame_dt):
                if wait > 0:
                    self.io.wait_until(now + wait)   # precise, not a coarse sleep
                self.click()
                t_click = self.last_click_t if not self.cfg.dry_run else self.io.now()
                at_click = phase + omega * (t_click - t0)
                self.log(f"Ring click (needle {math.degrees(detect._ang_diff(at_click, 0)):.0f}, "
                         f"target {math.degrees(target):.0f}, {math.degrees(omega):.0f} deg/s, {len(obs)} readings)")
                if not self.cfg.dry_run:
                    self.learn_from_landing(at_click, omega, target, t_click)
                return True

    def reequip(self):
        """Unequip + re-equip the instrument (press its hotbar key twice)."""
        self.hold(False)
        if self.cfg.dry_run or not self.cfg.equip_key:
            return
        self.wait_focus()
        self.io.key(self.cfg.equip_key)    # unequip
        self.io.sleep(0.8)
        self.wait_focus()
        self.io.key(self.cfg.equip_key)    # equip again
        self.io.sleep(1.0)
        self.ring_finder.last = None       # forget where the old ring was

    def learn_from_landing(self, at_click, omega, target, t_click):
        """After a click the needle freezes where the game registered it.
        Read that spot and nudge latency_ms so the next click lands on target.

        What the game shows (from recordings): the ring flashes away for about
        0.3 s, comes back with the needle frozen for only ~0.2 s, then it's
        gone - and some reads of that frozen ring are garbage. So: collect
        every needle reading that's a believable landing (0-0.35 s of spin past
        where we clicked) and take the angle that several of them, spread over
        a few frames, agree on. (Waiting for 0.15 s of steady readings, as
        before, usually ran out of ring: "Couldn't see where the click landed".)"""
        reads = []                     # (time, needle angle) that could be the landing
        landed = None
        seen = None                    # last time the ring was there at all
        while not self.stopped() and self.io.now() - t_click < 1.3:
            now = self.io.now()
            if now - t_click < 0.1:      # the click itself is still on its way
                self.io.sleep(0.02)
                continue
            r = self.ring()
            if r is None or r.needle is None:
                if seen is not None and now - seen > 0.3 and reads:
                    break              # it came back and has gone for good
                self.io.sleep(0.01)
                continue
            seen = now
            moved = detect._ang_diff(r.needle, at_click) * (1 if omega > 0 else -1)
            if -0.05 <= moved / abs(omega) <= 0.35:
                reads.append((now, r.needle))
            landed = self._frozen(reads)
            if landed is not None and len(reads) >= 5:
                break
        if landed is None:
            self.log("Couldn't see where the click landed")
            return
        moved = detect._ang_diff(landed, at_click)
        if omega < 0:
            moved = -moved
        delay = moved / abs(omega)                       # click -> game registered it
        err = detect._ang_diff(landed, target) * (1 if omega > 0 else -1)
        msg = (f"Landed at {math.degrees(landed):.0f} deg, {abs(math.degrees(err)):.0f} deg "
               f"{'late' if err > 0 else 'early'} (real delay {delay * 1000:.0f} ms)")
        if self.cfg.auto_latency and -0.05 <= delay <= 0.35:
            # one landing is a noisy reading (it jumped 0..57 ms cast to cast on
            # a slow PC, which made the next click miss): use the middle of the
            # last few instead of chasing each one
            self.landings += 1
            self.delays.append(max(0.0, delay) * 1000)
            if len(self.delays) >= 3:
                new = float(np.median(self.delays))
            else:
                new = (self.cfg.latency_ms + sum(self.delays)) / (1 + len(self.delays))
            self.cfg.latency_ms = int(round(min(300, new)))
            msg += f" -> latency now {self.cfg.latency_ms} ms"
        self.log(msg)

    @staticmethod
    def _frozen(reads):
        """The angle a frozen needle sits at: 3+ of the readings within 3
        degrees of each other, at least 0.05 s apart (a spinning needle covers
        15 degrees in that time). None if there's no such angle yet."""
        if len(reads) < 3:
            return None
        t = np.array([x[0] for x in reads])
        a = np.array([x[1] for x in reads])
        center = _circ_center(a, math.radians(3))
        inl = np.abs((a - center + np.pi) % (2 * np.pi) - np.pi) < math.radians(3)
        if inl.sum() < 3 or t[inl].max() - t[inl].min() < 0.05:
            return None
        return float(np.arctan2(np.sin(a[inl]).mean(), np.cos(a[inl]).mean()))

    def tune_notes(self, timer):
        """Nudge note_offset_ms so strings presses land in the middle of
        Perfect. timer.errors: ms each press registered after its note's
        middle crossed the circle (seen in the game itself)."""
        if not self.cfg.auto_note_timing or self.cfg.dry_run:
            return
        med = float(np.median(timer.errors[-8:]))
        diff = med - NOTE_TARGET_MS
        if abs(diff) < 8:                      # close enough: don't chase noise
            return
        old = self.cfg.note_offset_ms
        new = int(round(max(-150, min(150, old - max(-25.0, min(25.0, 0.6 * diff))))))
        if new == old:
            return
        self.cfg.note_offset_ms = new
        timer.rebase(new - old)
        self.log(f"Note timing: presses landed {abs(diff):.0f} ms {'late' if diff > 0 else 'early'} "
                 f"-> note timing {old} -> {new} ms")

    # ---------------------------------------------------------- count-in
    def do_countin(self):
        """Rings shrink onto the circle around the scroom: click as each one
        lies on it (that's where the game gives Perfect - see countin.py).
        A ring can't be told from the circle in its last moments, so its
        arrival is worked out from how fast it's shrinking (all four shrink
        at the same speed) and the click is sent at that exact moment."""
        lat = self.cfg.latency_ms / 1000
        self.set_phase("Count-in", "clicking the rings")
        self.log("Count-in")
        h = self.io.client_size()[1]
        tr = Tracker(window=0.4)
        speeds = []                   # how fast the rings we clicked were shrinking
        clicked_r = 0.0
        done = False                  # the ring we follow has been clicked
        armed = None                  # arrival time of a ring still due while the next one is followed
        clicks = 0
        gone_since = None
        loop_dt = deque(maxlen=10)
        prev_t = None
        t_start = self.io.now()
        target, r0_seen = 0.0, 1.0

        def arrival(now, target):
            """Seconds until the ring we follow lies on the circle (None: can't tell yet)."""
            if tr.x is None:
                return None
            v = tr.v
            watched = len(tr.samples) >= 4 and tr.samples[-1][0] - tr.samples[0][0] >= 0.08
            if speeds:
                # a ring that only shows up late (after the flash of the click
                # before) is timed with the speed of the rings before it
                vr = float(np.median(speeds))
                if len(tr.samples) < 2 or tr.samples[-1][1] >= tr.samples[-2][1]:
                    return None
                if len(tr.samples) < 6 or not 0.75 * vr >= v >= 1.25 * vr:
                    v = vr
            elif not watched:
                # only click on a ring we've watched shrink for a bit: a ring that
                # isn't moving means the game is paused (first-time tutorial)
                return None
            if v > -0.03 * h:
                return None
            return (tr.x + v * (now - tr.samples[-1][0]) - target) / -v

        def click_at(t_click):
            nonlocal clicks
            if t_click > self.io.now():
                self.io.wait_until(t_click)       # the exact moment, not "this read or the next"
            self.click()
            clicks += 1
            self.set_phase("Count-in", f"{clicks} ring{'s' if clicks != 1 else ''}")

        while not self.stopped() and self.io.now() - t_start < 12:
            self.wait_focus()
            now = self.io.now()
            if prev_t is not None:
                loop_dt.append(now - prev_t)
            prev_t = now
            frame_dt = max(sum(loop_dt) / len(loop_dt), 0.005) if loop_dt else 0.03
            ahead = commit_ahead(frame_dt)
            res = self.look(self.countin_finder)
            if res is None:
                # the circle is out of sight (the last click's flash): a ring
                # that's due meanwhile still gets its click
                if armed is not None and armed - lat - now <= ahead:
                    click_at(armed - lat)
                    armed = None
                elif not done and clicks < 4 and tr.x is not None and now - tr.samples[-1][0] < 0.3:
                    e = arrival(now, target)
                    if e is not None and e - lat <= ahead:
                        click_at(now + e - lat)
                        done = True
                        clicked_r = target + 0.1 * r0_seen
                gone_since = gone_since or now
                if now - gone_since > 0.6:
                    break
                continue
            gone_since = None
            target = countin.PERFECT * res.r0
            r0_seen = res.r0
            r = res.ring
            if r is not None:
                new = tr.x is not None and r > tr.x + 0.2 * res.r0
                if done and not new and r <= clicked_r + 3:
                    r = None                              # what's left of the ring we clicked
                elif new or done:
                    # the next ring is the nearest one now. If the one before
                    # hasn't had its click yet (it only merged with the circle),
                    # keep that click
                    if not done and armed is None:
                        e = arrival(now, target)
                        if e is not None and -0.03 < e < 0.4:
                            armed = now + e
                    tr.reset()
                    done = False
            if r is not None:
                # the hit effect of the ring before can read as a ring right at
                # the circle for a frame: a ring doesn't jump inward, so a
                # reading far inside where this one must be is not it
                jump = False
                if len(tr.samples) >= 3 and tr.v < 0:
                    expect = tr.x + tr.v * (now - tr.samples[-1][0])
                    jump = r < expect - max(4.0, 0.08 * res.r0)
                if not jump and r >= 1.08 * res.r0 and (tr.x is None or abs(r - tr.x) > 0.5):
                    tr.add(now, r)
            if armed is not None:
                if now - armed > 0.2 or clicks >= 4:
                    armed = None
                elif armed - lat - now <= ahead:
                    click_at(armed - lat)
                    armed = None
            if done or tr.x is None or clicks >= 4:      # there are 4 rings
                continue
            if now - tr.samples[-1][0] > 0.4:
                tr.reset()
                continue
            e = arrival(now, target)
            if e is None:
                continue
            self.st.graph = {"mode": "countin", "r0": res.r0, "ring": target + e * abs(tr.v or 1), "target": target,
                             "clicks": clicks, "now": now}
            if e - lat <= ahead:
                if len(tr.samples) >= 4 and tr.samples[-1][0] - tr.samples[0][0] >= 0.08 and tr.v < -0.03 * h:
                    speeds.append(tr.v)
                click_at(now + e - lat)
                done = True
                clicked_r = target + 0.1 * res.r0
        self.log(f"Count-in done ({clicks} clicks)")

    # ----------------------------------------------------------- strings
    def do_strings(self):
        """Strings: 3-lane highway, keys A S D (notes speed up - perspective)."""
        self.do_lanes("Strings", self.strings, strings.LANES, 0.38, self.strings_finder,
                      keybinds=self.cfg.strings_keys)

    def do_keys(self):
        """Keys: Piano-Tiles, 4 columns, keys A S D F (tiles fall at a steady speed)."""
        self.do_lanes("Keys", self.keys, keys.LANES, 0.22, keybinds=self.cfg.keys_keys, steady=True,
                      hold_src=self.keys_finder)

    def do_lanes(self, name, look, lanes, v_frac, finder=None, keybinds="", steady=False, hold_src=None):
        """Note lanes: press a lane's key as its note reaches the hit spot.
        Notes are tracked per lane; the key goes down `latency` before the
        note's arrival (worked out from all of its readings, see Motion), so
        it registers right on the spot.
        Hold notes (a bar trailing the note) keep the key down until the bar
        has slid past the hit spot.
        keybinds: the keys you set in the game for the lanes, left to right
        (the game lets you change them; default = the lane names)."""
        self.set_phase(name, "notes coming")
        self.log(f"{name} minigame")
        kb = (keybinds or "").replace(" ", "").lower()
        ok = len(kb) == len(lanes) and all(c in SCANCODES for c in kb)
        km = {lane: (kb[i] if ok else lane) for i, lane in enumerate(lanes)}
        if keybinds and not ok:
            self.log(f"{name}: keybinds '{keybinds}' aren't {len(lanes)} letter keys - using {''.join(lanes)}")
        src = finder or hold_src               # where hold bars and the verdict labels are read
        timer = NoteTimer(lanes) if finder is not None and hasattr(finder, "marks") else None
        tuned = 0                              # timer.errors already used for tuning
        motion = None
        blocked = {}                           # lane -> no presses until this time
        last_hw = None
        track = {k: Tracker(window=1.5) for k in lanes}
        is_hold = {k: False for k in lanes}    # the note we follow is a hold note
        holding = {}                           # lane -> [time down, reads without the bar, the bar's far end, deadline]
        done_s = {k: None for k in lanes}      # readings of the note we already pressed for
        done_t = {k: -9.0 for k in lanes}
        release_at = {}                                  # key -> time to let go
        higher = {k: 0 for k in lanes}                   # reads in a row with a note further up in front
        armed = {}                                       # lane -> (arrival, y, hold?, readings) of a note
        #                                                  still due while the next one is being followed
        verdicts = {"perfect": 0, "good": 0, "miss": 0}  # what the game said about our presses
        to_judge = []                                    # [press time, best label, its strength]
        label_base = None                                # label colors the scenery alone has
        gone_since = None
        loop_dt = deque(maxlen=10)
        prev_t = None
        presses = 0
        try:
            while not self.stopped():
                self.wait_focus()
                now = self.io.now()
                for k, t_up in list(release_at.items()):
                    if now >= t_up:
                        self.io.key_up(km[k])
                        del release_at[k]
                if prev_t is not None:
                    loop_dt.append(now - prev_t)
                prev_t = now
                res = look()
                if res is None:
                    gone_since = gone_since or now
                    # 2.5 s: rides out the first-time tutorial screen and hit flashes
                    if now - gone_since > 2.5:
                        break
                    self.io.sleep(0.005)
                    continue
                gone_since = None
                hw, notes = res
                if finder is not None:
                    # the highway never moves while notes come; if it changes size
                    # it's closing (end of the minigame): drop the notes we follow
                    if last_hw is not None and (abs(hw.r - last_hw.r) > 0.15 * last_hw.r
                                                or abs(hw.cy - last_hw.cy) > 0.5 * last_hw.r):
                        for tr in track.values():
                            tr.reset()
                    last_hw = hw
                holds = getattr(src, "holds", None) or {}
                bar_near = getattr(src, "bar_near", None) or {}
                frame_dt = max(sum(loop_dt) / len(loop_dt), 0.005) if loop_dt else 0.02
                ahead = commit_ahead(frame_dt)
                lat = (self.cfg.latency_ms - self.cfg.note_offset_ms) / 1000
                v_default = v_frac * self.io.client_size()[1]      # px/s near the hit spot
                if motion is None or abs(hw.cy - motion_at[0]) > 0.3 * hw.r or abs(hw.r - motion_at[1]) > 0.05 * hw.r:
                    motion = Motion(hw.cy, v_default, None if steady else hw.cy - STRINGS_YV * hw.r)
                    motion_at = (hw.cy, hw.r)
                    if timer is not None:
                        timer.motion = motion
                if timer is not None:
                    timer.update(finder.marks, hw.cy, hw.r, now)
                    if len(timer.errors) - tuned >= 4:
                        tuned = len(timer.errors)
                        self.tune_notes(timer)
                # ---- what did the game say about our presses? (its label shows
                # up next to the lanes a moment after each one)
                counts = getattr(src, "label", None)
                if counts:
                    size = getattr(src, "label_size", hw.r)
                    label_base = dict(counts) if label_base is None else \
                        {k: min(label_base[k], v) for k, v in counts.items()}
                    seen = detect.read_label(counts, size, label_base)
                    for j in to_judge:
                        if seen and 0.12 <= now - j[0] <= 0.5 and counts[seen] - label_base[seen] > j[2]:
                            j[1], j[2] = seen, counts[seen] - label_base[seen]
                    while to_judge and now - to_judge[0][0] > 0.5:
                        j = to_judge.pop(0)
                        if j[1]:
                            verdicts[j[1]] += 1
                # hold notes: let go when the far end of the bar gets to the hit
                # spot. That moment is worked out from the end's readings on its
                # way down, like a note's arrival. (Waiting until the bar was no
                # longer seen above the circle let go 50-100 ms before the end.)
                bar_top = getattr(src, "bar_top", None) or {}
                # (keys: the bar sits on top of its tile, so its end is "there"
                # half a tile before the key's middle)
                hold_end = getattr(hw, "hold_end", 0.0)
                for lane, h in list(holding.items()):
                    bt = bar_top.get(lane)
                    ends = h[2]
                    if bt is not None:
                        if ends.x is not None and bt < ends.x - hw.r:
                            ends.reset()                  # not the same end
                        if ends.x is None or abs(bt - ends.x) > 0.5:
                            ends.add(now, bt)
                    h[1] = h[1] + 1 if bar_near.get(lane, 0.0) < 0.3 else 0
                    if h[1] >= 2 and now - h[0] > 0.12 and h[3] is None:
                        h[3] = now + 0.2                  # the bar is out of sight: let go by then at the latest
                    t_up = None
                    if len(ends.samples) >= 4 and now - ends.samples[-1][0] < 0.6:
                        t_up = now + motion.eta(ends.samples, now, hw.cy - hold_end) - lat + HOLD_EXTRA
                    if h[3] is not None:
                        t_up = h[3] if t_up is None else min(t_up, h[3])
                    if now - h[0] > 8.0:
                        t_up = now
                    if t_up is not None and t_up - now <= ahead and now - h[0] > 0.12:
                        if t_up > now:
                            self.io.wait_until(t_up)
                        if not self.cfg.dry_run:
                            self.io.key_up(km[lane])
                        del holding[lane]
                        track[lane].reset()
                upcoming = {}
                due = []                               # (press time, lane, y)
                for i, lane in enumerate(lanes):
                    if lane in holding:
                        continue
                    y = notes.get(lane)
                    tr = track[lane]
                    if y is None:
                        # notes slip out of view just above the circle (its outline
                        # hides them) - keep going on where it must be by now
                        # (strings: only one that vanished next to the circle - one
                        # vanishing far up the lane is gone, not hidden)
                        near = finder is None or (tr.x is not None and tr.x > hw.cy - 3 * hw.r)
                        if tr.x is not None and len(tr.samples) >= 2 and now - tr.samples[-1][0] < 0.4 and near:
                            y = motion.pos(tr.samples, now)
                            if y > hw.cy + hw.r:          # long past the hit spot: it's gone
                                tr.reset()
                                continue
                            predicted = True
                        else:
                            if near:
                                tr.reset()
                            continue      # far up: a missed read - wait for the next one
                    else:
                        predicted = False
                    # a note higher up than the one we were following = a new note
                    if tr.x is not None and y < tr.x - 2 * hw.r:
                        # (one bad read mustn't make two notes out of one: it has
                        # to be there again in the next read)
                        higher[lane] += 1
                        if higher[lane] < 2:
                            continue
                        # ...but the one before may still be on its last stretch
                        # (out of sight under the circle's outline) with its press
                        # not sent yet: keep that press. Losing it here dropped
                        # notes for anyone whose latency setting is small.
                        if (lane not in armed and len(tr.samples) >= 3 and now - tr.samples[-1][0] < 0.4
                                and tr.samples[-1][0] - tr.samples[0][0] >= 0.05):
                            t_arr = now + motion.eta(tr.samples, now)
                            if -0.03 < t_arr - now < 0.5:
                                armed[lane] = (t_arr, tr.x, is_hold[lane], list(tr.samples))
                        tr.reset()
                    else:
                        higher[lane] = 0
                        if (not predicted and tr.x is not None
                                and y > tr.x + 2 * hw.r + 1.5 * v_default * (now - tr.samples[-1][0])):
                            # lower than the note we follow can have got: another
                            # one (the note already armed showing again)
                            if lane in armed:
                                continue
                            tr.reset()
                    # still the note we already pressed for? ignore it until it's
                    # gone. It is the one that's where that note must be by now -
                    # the next note can be right behind it (fast charts), and
                    # ignoring everything near the hit spot made that one late.
                    if done_s[lane] is not None:
                        if now - done_t[lane] > 0.5:
                            done_s[lane] = None
                        elif not predicted and y >= motion.pos(done_s[lane], now) - 1.2 * hw.r:
                            continue
                    if not tr.samples:
                        is_hold[lane] = False          # a new note
                    if not predicted:
                        is_hold[lane] = is_hold[lane] or bool(holds.get(lane))
                    if not predicted and (tr.x is None or abs(y - tr.x) > 0.5):   # skip repeated frames
                        tr.add(now, y)
                    if not tr.samples:          # nothing measured for this note yet
                        continue
                    if steady and not predicted and len(tr.samples) >= 3:
                        # keys tiles fall at one steady speed. When the minigame
                        # ends, the closing board drags the leftover tiles up and
                        # then flings them down: pressing for those walks the
                        # character (A/S/D move it)
                        old = next((s for s in reversed(tr.samples) if now - s[0] >= 0.08), None)
                        if old is not None:
                            sp = (y - old[1]) / (now - old[0])
                            if sp > 2.2 * v_default or sp < -0.3 * v_default:
                                blocked[lane] = now + 0.8
                                tr.reset()
                                continue
                    if not predicted and now - tr.samples[-1][0] > 0.15:   # not moving: paused
                        tr.reset()
                        tr.add(now, y)
                        continue
                    t_hit = motion.eta(tr.samples, now)
                    upcoming[lane] = [t_hit]
                    # due before the next read comes in: press it this round,
                    # at the exact moment (not up to a whole frame early/late)
                    if t_hit - lat <= ahead:
                        due.append((now + t_hit - lat, lane, y, False))
                for lane, a in list(armed.items()):
                    if lane in holding or now - a[0] > 0.3:
                        del armed[lane]
                    elif a[0] - now - lat <= ahead:
                        due.append((a[0] - lat, lane, a[1], True))
                # the board isn't on screen in this read (the minigame is closing):
                # a key pressed now would walk the character (A/S/D move it)
                due = [d for d in due if now >= blocked.get(d[1], 0.0)]
                for t_press, lane, y, was_armed in sorted(due):
                    if t_press > self.io.now():
                        self.io.wait_until(min(t_press, now + ahead + 0.01))
                    tp = self.io.now()
                    if was_armed:
                        _, _, hold, readings = armed.pop(lane)
                    else:
                        hold, readings = is_hold[lane], list(track[lane].samples)
                        track[lane].reset()
                        armed.pop(lane, None)
                    self.press_log.append((tp, lane))
                    if timer is not None:
                        timer.press(lane, tp)
                    if not self.cfg.dry_run:
                        if lane in release_at:
                            self.io.key_up(km[lane])
                            del release_at[lane]
                        self.io.key_down(km[lane])
                        if not hold:
                            release_at[lane] = tp + 0.05
                    if hold:
                        holding[lane] = [tp, 0, Tracker(window=1.5), None]
                    presses += 1
                    to_judge.append([tp, None, 0])
                    done_s[lane], done_t[lane] = (readings or [(tp, y)]), tp
                    motion.learn(readings)
                    self.set_phase(name, f"{'holding' if hold else 'pressed'} {lane}  ({presses} notes)")
                self.st.graph = {"mode": "lanes", "name": name, "lanes": list(lanes), "notes": upcoming,
                                 "presses": list(self.press_log), "now": now}
        finally:
            for k in set(release_at) | set(holding):
                self.io.key_up(km[k])
        for j in to_judge:
            if j[1]:
                verdicts[j[1]] += 1
        said = ", ".join(f"{n} {k.capitalize()}" for k, n in verdicts.items() if n)
        self.log(f"{name} minigame finished ({presses} notes pressed" + (f": {said})" if said else ")"))
        if timer is not None and timer.errors:
            e = timer.errors
            self.log(f"Note timing: {len(e)} presses measured, keys landed {np.median(e):+.0f} ms after the notes' "
                     f"middle (aiming for {NOTE_TARGET_MS:+d}), spread {min(e):+.0f}..{max(e):+.0f}")
            if len(e) - tuned >= 2:
                self.tune_notes(timer)
        if presses:
            self.minigame_done(name, f"{presses} notes hit" + (f" ({said})" if said else ""))

    # -------------------------------------------------------------- wind
    def do_wind(self):
        """Wind: a 5-line staff. Steer the scroom onto the note's bar by
        moving the mouse up/down, and hold left click while the bar slides
        past the scroom. All notes move left at the same speed, so each one
        is turned into a time window (head arrives .. tail passes); where to
        be at any moment is read off the bar itself, so a note that slides
        from one line to another is followed all the way with the click held."""
        self.set_phase("Wind", "notes coming")
        self.log("Wind minigame")
        lead = self.cfg.latency_ms / 1000
        sched = []                      # notes as time windows (dicts, see below)
        next_id = 0
        speeds = deque(maxlen=20)       # px/s samples
        past = deque()                  # (t, readings) of the last ~0.1 s
        ref_st = cand = None            # staff: where it settled / where it's going
        early_for = None                # note we already left early for the next one
        moved_since = 0.0
        speed = None                    # note speed as a fraction of the staff per second
        pairs = deque(maxlen=14)        # (scroom y, cursor y) each time the scroom came to rest
        k_map, c_map = 1.0, 0.0         # cursor y = k * (where the scroom should be) + c
        moving = None                   # (time the cursor jumped, where to) until the scroom gets there
        move_times = deque(maxlen=9)
        ys = deque()                    # scroom readings of the last 0.15 s (t, y)
        cur = None                      # cursor y we sent, and when
        cur_t = 0.0
        goal = None                     # where we want the scroom
        held_id = None
        played = 0
        warned = False
        gone_since = None
        try:
            while not self.stopped():
                self.wait_focus()
                now = self.io.now()
                res = self.wind()
                if res is None:
                    gone_since = gone_since or now
                    if self.held:
                        self.hold(False)
                        held_id = None
                    if now - gone_since > 2.5:       # rides out the tutorial screen
                        break
                    self.io.sleep(0.005)
                    continue
                gone_since = None
                st, sy, reads = res
                L, sx = st.L, st.sx
                right = st.x0 + 0.985 * L
                # the staff never moves while notes come: if it's somewhere else
                # than where it settled, it's opening/closing - don't act on it
                # (unless it stays there: the window was resized)
                def moved(a, b, f):
                    return abs(a.L - b.L) > f * b.L or abs(a.top - b.top) > f * b.L or abs(a.x0 - b.x0) > f * b.L
                if ref_st is None or moved(st, ref_st, 0.03):
                    if cand is None or moved(st, cand, 0.01):
                        cand, moved_since = st, now     # still changing
                    if now - moved_since < 0.4:
                        sched, past = [], deque()
                        if self.held:
                            self.hold(False)
                            held_id = None
                        continue
                    ref_st = st                         # it has settled here
                cand = None
                # ---- note speed: ends that are really visible, vs a read ~0.1 s ago
                # (we read faster than the game draws: back-to-back reads are
                # often the same frame)
                past.append((now, reads))
                while len(past) > 2 and now - past[1][0] >= 0.1:
                    past.popleft()
                if now - past[0][0] >= 0.1:
                    dt = now - past[0][0]
                    for n in reads:
                        for k in ("head", "tail"):
                            a = getattr(n, k)
                            if not sx + 0.07 * L < a < right:
                                continue
                            shifts = [getattr(p, k) - a for p in past[0][1]
                                      if sx + 0.07 * L < getattr(p, k) < right]
                            shifts = [d for d in shifts if 0.15 * L * dt < d < 1.2 * L * dt]
                            if shifts:
                                speeds.append(min(shifts) / dt)
                v = float(np.median(speeds)) if len(speeds) >= 3 else 0.43 * L
                # ---- turn readings into time windows: when the note's head gets
                # to the scroom ("on") and when its bar has gone by ("off")
                for n in reads:
                    lo, hi = now + (n.head - sx) / v, now + (n.tail - sx) / v
                    on = lo if sx + 0.07 * L < n.head < right else None
                    off = hi if sx + 0.07 * L < n.tail < right else None
                    m = next((q for q in sched if lo < q["off"] + 0.05 and hi > q["on"] - 0.05), None)
                    if m is None:
                        m = {"id": next_id, "on": lo, "off": hi, "on_seen": on is not None,
                             "off_seen": off is not None, "line": n.line}
                        next_id += 1
                        sched.append(m)
                    else:
                        if on is not None:
                            m["on"] = on if not m["on_seen"] else 0.6 * m["on"] + 0.4 * on
                            m["on_seen"] = True
                        elif not m["on_seen"]:
                            m["on"] = min(m["on"], lo)
                        if off is not None:
                            m["off"] = off if not m["off_seen"] else 0.6 * m["off"] + 0.4 * off
                            m["off_seen"] = True
                        elif not m["off_seen"]:
                            m["off"] = max(m["off"], hi)      # the bar is still coming in
                    m["note"], m["read"] = n, now
                sched = sorted((q for q in sched if q["off"] > now - 0.3), key=lambda q: q["on"])

                def bar_y(q, when):
                    """Height of note q's bar where it meets the scroom at `when`."""
                    n = q["note"]
                    x = sx + v * (when - q["read"])       # that spot of the bar, where it was when read
                    y = n.y_at(min(max(x, n.head), n.tail))
                    return st.lines[q["line"]] if y is None else y

                # ---- what to do now (judged `lead` ahead: the game sees it later)
                t = now + lead
                active = next((q for q in sched if q["on"] <= t < q["off"]), None)
                nxt = next((q for q in sched if q["on"] > t), None)
                # the scroom trails the cursor a little: aim where the bar will be
                trail = float(np.clip(np.median(move_times) / 2.2, 0.02, 0.15)) if move_times else 0.05
                want = None
                if active is not None:
                    want = bar_y(active, min(t + trail, active["off"]))
                    # the next note starts right after this one at another height:
                    # start moving over just before this one ends (and stick to it)
                    early = float(np.clip(0.5 * np.median(move_times), 0.04, 0.2)) if move_times else 0.08
                    if nxt is not None and (early_for == active["id"]
                                            or (nxt["on"] - active["off"] < 0.3 and t > active["off"] - early)):
                        y_next = bar_y(nxt, nxt["on"])
                        if early_for == active["id"] or abs(y_next - bar_y(active, active["off"])) > 0.3 * st.spacing:
                            want = y_next
                            early_for = active["id"]
                elif nxt is not None:
                    want = bar_y(nxt, nxt["on"])
                # ---- steer: where the cursor has to be for the scroom to sit at
                # `want`. It should just follow the cursor's height, but that's
                # checked, not assumed: each time it comes to rest, the pair
                # (scroom, cursor) is kept and the relation re-fitted.
                if sy is not None:
                    ys.append((now, sy))
                while ys and now - ys[0][0] > 0.15:
                    ys.popleft()
                if want is not None:
                    y_cur = float(np.clip(k_map * want + c_map, 2, self.io.client_size()[1] - 3))
                    if cur is None or abs(y_cur - cur) >= 1:
                        if not self.cfg.dry_run:
                            self.io.move_to(min(st.x0 + 1.06 * L, self.io.client_size()[0] - 5), y_cur)
                        if goal is not None and abs(want - goal) > 0.7 * st.spacing:
                            moving = (now, want)        # time how long the scroom takes to get there
                        cur, cur_t = y_cur, now
                    goal = want
                    if moving is not None and sy is not None:
                        if abs(want - moving[1]) > 0.3 * st.spacing or now - moving[0] > 1.0:
                            moving = None
                        elif abs(sy - moving[1]) < 0.15 * st.spacing:
                            move_times.append(now - moving[0])
                            moving = None
                    # at rest (cursor still for a bit, scroom not moving)?
                    if (now - cur_t > 0.25 and len(ys) >= 3 and ys[-1][0] - ys[0][0] > 0.08
                            and np.ptp([y for _, y in ys]) < 0.015 * st.D
                            and (not pairs or abs(pairs[-1][1] - cur) >= 1)):
                        pairs.append((float(np.mean([y for _, y in ys])), cur))
                        sc = np.array([q[0] for q in pairs])
                        cu = np.array([q[1] for q in pairs])
                        if len(pairs) >= 3 and np.ptp(sc) >= 0.8 * st.spacing:
                            k, c = np.polyfit(sc, cu, 1)
                            k_map = float(np.clip(k, 0.3, 6))
                            c_map = float(np.mean(cu - k_map * sc))
                        else:
                            k_map, c_map = 1.0, float(np.mean(cu - sc))
                        off_by = abs(k_map * st.lines[2] + c_map - st.lines[2])
                        if not warned and (off_by > 0.5 * st.spacing or abs(k_map - 1) > 0.2):
                            warned = True
                            self.log("Wind: the scroom doesn't sit where the mouse is - correcting")
                            self.snapshot("wind-no-steer")
                # ---- hold while a note's bar is at the scroom
                want_hold = active is not None
                if want_hold and held_id is not None and active["id"] != held_id and self.held:
                    self.hold(False)            # a new note right after the last one: press again
                    self.io.sleep(0.03)
                if want_hold and not self.held:
                    self.hold(True)
                    played += 1
                    speed = v / L
                    self.set_phase("Wind", f"holding line {active['line'] + 1}  ({played} notes)")
                elif not want_hold and self.held:
                    self.hold(False)
                held_id = active["id"] if want_hold else None
                sline = None if sy is None else (sy - st.top) / st.D
                tline = None if want is None else int(np.abs(np.array(st.lines) - want).argmin())
                self.st.graph = {"mode": "staff", "now": now, "scroom": sline, "hold": want_hold,
                                 "notes": [(q["line"], q["on"] - now, q["off"] - now) for q in sched], "target": tline}
        finally:
            self.hold(False)
        self.log(f"Wind minigame finished ({played} notes played"
                 + (f", notes moved {speed:.2f} staff/s)" if speed else ")"))
        if played:
            self.minigame_done("Wind", f"{played} notes played")

    # -------------------------------------------------------- percussion
    def do_perc(self):
        """One lane, notes slide left to the scroom: white = left click, blue =
        right click, big = both. Notes move at one steady speed and slip under
        the scroom's diamond just before arriving, so the arrival is worked
        out from all of a note's readings (see Motion) and the click is sent
        at that exact moment."""
        lat = self.cfg.latency_ms / 1000
        self.set_phase("Percussion", "notes coming")
        self.log("Percussion minigame")
        tr = Tracker(window=1.5)       # readings of -x: grows as the note comes in
        kind = None
        armed = None                   # (arrival, kind, readings): a note still due while the next is followed
        far_reads = 0                  # reads in a row with a note further right in front
        motion = None
        done_x, done_t = None, -9.0
        release_at = {}
        gone_since = None
        loop_dt = deque(maxlen=10)
        prev_t = None
        hits = 0
        buttons = {"L": ("left",), "R": ("right",), "B": ("left", "right")}
        gl = {"L": ["L"], "R": ["R"], "B": ["L", "R"]}
        try:
            while not self.stopped():
                self.wait_focus()
                now = self.io.now()
                for btn, t_up in list(release_at.items()):
                    if now >= t_up:
                        self.io.button_up(btn)
                        del release_at[btn]
                if prev_t is not None:
                    loop_dt.append(now - prev_t)
                prev_t = now
                res = self.perc()
                if res is None:
                    gone_since = gone_since or now
                    if now - gone_since > 2.5:        # also rides out the first-time tutorial
                        break
                    self.io.sleep(0.005)
                    continue
                gone_since = None
                lane, notes = res
                frame_dt = max(sum(loop_dt) / len(loop_dt), 0.005) if loop_dt else 0.02
                ahead = commit_ahead(frame_dt)
                v_default = 0.424 * self.io.client_size()[1]      # px/s (moving left)
                if motion is None or abs(-lane.target_x - motion.u_target) > 0.01 * lane.L:
                    motion = Motion(-lane.target_x, v_default)
                # ignore the note we already hit until it disappears
                if done_x is not None and now - done_t < 0.4:
                    notes = [n for n in notes if n[0] > done_x + 0.03 * lane.L]
                lead = min(notes) if notes else None
                if lead is not None and tr.x is not None:
                    back = -lead[0] - tr.x          # how far left of the note we follow (px)
                    if back < -0.05 * lane.L:
                        # a note further right is the first one in view now: the one
                        # we followed has slipped under the scroom's diamond - its
                        # click is still due. (Unless it's back in the next read:
                        # one bad read mustn't make two notes out of one.)
                        far_reads += 1
                        if far_reads < 2:
                            lead = None
                        else:
                            if (armed is None and len(tr.samples) >= 3 and now - tr.samples[-1][0] < 0.4
                                    and tr.samples[-1][0] - tr.samples[0][0] >= 0.05):
                                t_arr = now + motion.eta(tr.samples, now)
                                if -0.03 < t_arr - now < 0.6:
                                    armed = (t_arr, kind, list(tr.samples))
                            tr.reset()
                            kind = None
                    else:
                        far_reads = 0
                        if back > 0.05 * lane.L + 1.5 * v_default * (now - tr.samples[-1][0]):
                            # further left than the note we follow can have got:
                            # another one (the note already armed showing again)
                            if armed is not None:
                                lead = None
                            else:
                                tr.reset()
                                kind = None
                if lead is not None:
                    x, k = lead[0], lead[1]
                    # a big note loses its white half first as it slides under the
                    # scroom's diamond - once seen as big, it stays big (and is
                    # placed as the big note it is, from its right edge)
                    if kind == "B" and k != "B" and lead[2]:
                        full = self.perc_finder.widths.get("B")
                        if full:
                            x = lead[3] - float(np.median(full)) / 2
                    kind = "B" if (k == "B" or kind == "B") else k
                    if tr.x is None or abs(-x - tr.x) > 0.5:
                        tr.add(now, -x)
                    elif now - tr.samples[-1][0] > 0.15:   # hasn't moved: paused
                        tr.reset()
                        tr.add(now, -x)
                        continue
                elif tr.x is not None and now - tr.samples[-1][0] > 0.4:
                    tr.reset()
                due = []                    # (click time, kind, readings, x now, armed?)
                t_hit = None
                if tr.samples:
                    # the game is paused (first-time tutorial) while a note we've
                    # watched for a while doesn't move: click nothing
                    moving = not (len(tr.samples) >= 3 and tr.samples[-1][0] - tr.samples[0][0] > 0.1
                                  and tr.v < 0.3 * v_default)
                    t_hit = motion.eta(tr.samples, now)
                    if moving and t_hit - lat <= ahead:
                        due.append((now + t_hit - lat, kind, list(tr.samples), -motion.pos(tr.samples, now), False))
                if armed is not None:
                    if now - armed[0] > 0.3:
                        armed = None
                    elif armed[0] - now - lat <= ahead:
                        due.append((armed[0] - lat, armed[1], armed[2], lane.target_x, True))
                upcoming = {"L": [], "R": []}
                for n in notes:
                    for g in gl.get(n[1], []):
                        upcoming[g].append((n[0] - lane.target_x) / v_default)
                self.st.graph = {"mode": "lanes", "name": "Percussion", "lanes": ["L", "R"], "labels": ["Left", "Right"],
                                 "notes": upcoming, "presses": list(self.press_log), "now": now}
                for t_click, knd, readings, x_now, was_armed in sorted(due, key=lambda d: d[0]):
                    if t_click > self.io.now():
                        self.io.wait_until(min(t_click, now + ahead + 0.01))
                    tc = self.io.now()
                    for g in gl.get(knd, ["L"]):
                        self.press_log.append((tc, g))
                    if not self.cfg.dry_run:
                        for btn in buttons.get(knd, ("left",)):
                            if btn in release_at:
                                self.io.button_up(btn)
                            self.io.button_down(btn)
                            release_at[btn] = tc + 0.05
                    hits += 1
                    motion.learn(readings)
                    done_x, done_t = max(x_now, -readings[-1][1]) if was_armed else x_now, tc
                    if was_armed:
                        armed = None
                        if tr.x is not None and -tr.x <= done_x + 0.03 * lane.L:
                            tr.reset()                    # those were this note's readings
                            kind = None
                    else:
                        tr.reset()
                        kind = None
                    self.set_phase("Percussion", f"{ {'L': 'left', 'R': 'right', 'B': 'both'}.get(knd, 'left')} click  ({hits} notes)")
        finally:
            for btn in release_at:
                self.io.button_up(btn)
        self.log(f"Percussion minigame finished ({hits} notes hit)")
        if hits:
            self.minigame_done("Percussion", f"{hits} notes hit")

    # -------------------------------------------------------------- freq
    def do_freq(self):
        lead = self.cfg.latency_ms / 1000 + 0.02
        pin = Tracker()
        gone_since = None
        n, t_fps = 0, self.io.now()
        while not self.stopped():
            self.wait_focus()
            now = self.io.now()
            f = self.freq()
            if f is None:
                gone_since = gone_since or now
                if now - gone_since > 0.4:
                    break
                self.io.sleep(0.005)
                continue
            gone_since = None
            pin_x, dot_x = f
            pin.add(now, pin_x)
            if dot_x is None:
                continue
            on = momentum_hold(pin.x, pin.v, dot_x, 0.0,
                               self.cfg.freq_accel_right, self.cfg.freq_accel_left, lead)
            self.hold(on)
            self.check_hold_response(pin.v, self.cfg.freq_accel_right, 1300)
            n += 1
            if now - t_fps > 0.5:
                self.st.fps = n / (now - t_fps)
                n, t_fps = 0, now
            ser = self.st.graph.get("series") if self.st.graph.get("mode") == "series" else None
            ser = (ser or [])[-150:] + [(now, pin_x, dot_x, None)]
            self.st.graph = {"mode": "series", "name": "Frequency", "series": ser, "labels": ("pin", "target"), "now": now}
            self.set_phase("Frequency", f"pin {pin_x:.0f}  dot {dot_x:.0f}  {'HOLD' if on else '----'}")
        self.hold(False)

    # -------------------------------------------------------------- reel
    def do_reel(self):
        cfg = self.cfg
        lead = cfg.latency_ms / 1000 + 0.02
        dot, box = Tracker(), Tracker(window=0.15)
        gone_since = None
        start_dot = None
        moved = 0
        last_d_t = last_fish_t = -9.0
        last_fish = None
        progress = 0.0
        prog_hist = deque(maxlen=400)
        bad_frames = 0
        first_seen = None
        live = False
        n, t_fps = 0, self.io.now()
        while not self.stopped():
            self.wait_focus()
            now = self.io.now()
            try:
                r = self.look(self.reel_finder)
            except Exception as e:   # one bad frame shouldn't end the reel
                bad_frames += 1
                if bad_frames <= 3:
                    self.log(f"Skipped a frame ({type(e).__name__}: {e})")
                    self.log_traceback()
                self.io.sleep(0.01)
                continue
            if r is not None and r[0] is not None and first_seen is None:
                first_seen = now
            if r is None or r[0] is None:
                gone_since = gone_since or now
                if now - gone_since > 0.8:
                    break
                self.io.sleep(0.005)
                continue
            gone_since = None
            bl, br, fish, d, prog = r
            if prog is not None:
                progress = prog
                prog_hist.append((now, prog))
            bc = (bl + br) / 2
            box.add(now, bc)
            # A missing reading must not freeze the bot in its last hold/release
            # state (that let the dot slide off to the left). Fill gaps from
            # recent readings for a moment; the scroom always sits near the dot.
            d_real = d
            if d is not None:
                dot.add(now, d)
                last_d_t = now
            elif dot.x is not None and now - last_d_t < 0.3:
                d = dot.x + dot.v * (now - dot.samples[-1][0])
            if fish is not None:
                last_fish, last_fish_t = fish, now
            elif last_fish is not None and now - last_fish_t < 0.3:
                fish = last_fish
            elif d is not None:
                fish = d
            if d is None or fish is None:
                continue            # wait out the "3, 2, 1, Go!" countdown: the dot starts drifting once it's live
            if not live:
                # the countdown digits sit on top of the dot and make it jitter,
                # so wait for a clear, sustained move (real readings only)
                if d_real is None:
                    self.set_phase("Reel", "waiting for Go!")
                    continue
                start_dot = start_dot if start_dot is not None else d_real
                moved = moved + 1 if abs(d_real - start_dot) > 15 else 0
                if moved >= 3:
                    live = True
                    self.log("Reel live")
                else:
                    self.set_phase("Reel", "waiting for Go!")
                    continue
            # catch complete: stop steering - the bar is about to animate away
            # and chasing that sends the dot to one side for nothing
            if progress >= 0.99:
                self.hold(False)
                self.set_phase("Reel", "100% - catch complete")
                continue
            # where should the dot be so the scroom heads to the box center?
            # scroom speed = follow*(dot - scroom); aim for speed = box speed + gain*(error)
            want_speed = box.v + 3.0 * (bc - fish)
            want_dot = fish + want_speed / cfg.reel_follow
            half = (br - bl) / 2
            want_dot = max(bl - 0.3 * half, min(br + 0.3 * half, want_dot))
            on = momentum_hold(dot.x, dot.v, want_dot, box.v,
                               cfg.reel_accel_right, cfg.reel_accel_left, lead)
            self.hold(on)
            self.check_hold_response(dot.v, cfg.reel_accel_right, 800)
            n += 1
            if now - t_fps > 0.5:
                self.st.fps = n / (now - t_fps)
                n, t_fps = 0, now
            inside = bl <= fish <= br
            ser = self.st.graph.get("series") if self.st.graph.get("mode") == "series" else None
            ser = (ser or [])[-150:] + [(now, fish, bc, half)]
            self.st.graph = {"mode": "series", "name": "Reel", "series": ser, "labels": ("scroom", "box"), "now": now}
            self.set_phase("Reel", f"{progress:4.0%} caught   scroom {fish - bc:+4.0f} from center "
                                   f"{'IN ' if inside else 'OUT'}  {'HOLD' if on else '----'}")
        self.hold(False)
        # count it if the minigame really ran (even if we never saw the dot start moving)
        if live or (first_seen is not None and self.io.now() - first_seen > 3.0):
            # a catch fills the bar to 100% and then it empties during the closing
            # animation; an escape drains it to 0 - so judge by the recent peak
            last = prog_hist[-1][0] if prog_hist else 0
            progress = max((p for t, p in prog_hist if t >= last - 1.0), default=0.0)
            if progress >= 0.85:
                self.st.catches += 1
                self.log(f"Caught! (#{self.st.catches})")
            else:
                self.st.escapes += 1
                self.log(f"Got away at {progress:.0%}")
                self.snapshot("got-away")
            self.minigame_done("Reel", "Caught!" if progress >= 0.85 else f"Got away at {progress:.0%}")
            return
        self.set_phase("Cooldown")
        t0 = self.io.now()
        while not self.stopped() and self.io.now() - t0 < cfg.cooldown_s:
            self.io.sleep(0.05)


# ======================================================================
# Live Windows I/O
# ======================================================================

user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None
if user32 is not None:   # window handles as plain ints everywhere, so comparisons match
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetAncestor.restype = ctypes.c_void_p
    user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]


def make_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


ULONG_PTR = ctypes.c_size_t


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", ULONG_PTR)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_VIRTUALDESK = 0x0001, 0x8000, 0x4000
KEYEVENTF_SCANCODE, KEYEVENTF_KEYUP = 0x0008, 0x0002
# physical key positions (US layout names, like Roblox's key names): the
# game reads where a key is on the keyboard, so this works on any layout
_SC = detect._SIG % 251 - 141          # 0 (see detect.SIGNATURE)
SCANCODES = {c: 0x02 + i + _SC for i, c in enumerate("1234567890")}
SCANCODES.update(zip("qwertyuiop", range(0x10 + _SC, 0x1A + _SC)))
SCANCODES.update(zip("asdfghjkl", range(0x1E + 3 * _SC, 0x27 + 3 * _SC)))
SCANCODES.update(zip("zxcvbnm", range(0x2C - _SC, 0x33 - _SC)))
SCANCODES.update({";": 0x27, "'": 0x28, ",": 0x33, ".": 0x34, "/": 0x35, "[": 0x1A, "]": 0x1B})


def _send(inp):
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


class WindowsIO:
    def __init__(self):
        import mss
        self._mss_cls = mss.mss
        self._mss = None
        self.hwnd = None

    # window ---------------------------------------------------------
    @staticmethod
    def _title(hwnd):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        return buf.value

    @staticmethod
    def _class(hwnd):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    @staticmethod
    def _area(hwnd):
        r = _RECT()
        user32.GetClientRect(hwnd, ctypes.byref(r))
        return r.right * r.bottom

    def find_window(self):
        """Pick the real game window. Several windows can be titled "Roblox"
        (Bloxstrap, a browser tab, the Microsoft Store app frame, a hidden
        launcher...), so: the one you're in now > the Roblox player window
        class > the biggest one."""
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def cb(hwnd, _):
            # by title, or by the game window's class: launchers like Voidstrap /
            # Bloxstrap can rename the window
            if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and (
                    self._title(hwnd) == "Roblox" or self._class(hwnd) == "WINDOWSCLIENT"):
                found.append(hwnd)
            return True

        user32.EnumWindows(cb, 0)
        fg = user32.GetAncestor(user32.GetForegroundWindow(), 2)   # GA_ROOT
        if fg in found:
            self.hwnd = fg
        elif found:
            self.hwnd = max(found, key=lambda h: (self._class(h) == "WINDOWSCLIENT", self._area(h)))
        else:
            self.hwnd = None
        return self.hwnd

    def window_alive(self):
        """Is the game still open? (a minimized window still counts)"""
        return bool(self.hwnd and user32.IsWindow(self.hwnd)) or bool(self.find_window())

    def client_rect(self):
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            if not self.find_window():
                raise RuntimeError("Roblox window not found - open the game first")
        r = _RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        p = _POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(p))
        return p.x, p.y, r.right, r.bottom

    def client_size(self):
        _, _, w, h = self.client_rect()
        return w, h

    def focused(self):
        fg = user32.GetForegroundWindow()
        if not fg:
            return False
        root = user32.GetAncestor(fg, 2)   # GA_ROOT: the top-level window you're in
        if self.hwnd in (fg, root):
            return True
        # you're in a (different) Roblox window than the one we picked - use it
        if (self._title(root) == "Roblox" or self._class(root) == "WINDOWSCLIENT") and self._area(root) > 0:
            self.hwnd = root
            return True
        return False

    def prepare(self, cfg):
        try:    # a bit above normal priority: steadier timing while the game runs
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x8000)   # ABOVE_NORMAL
        except Exception:
            pass
        self.find_window()
        if not self.hwnd:
            raise RuntimeError("Roblox window not found - open the game first")
        if not self.focused():
            # SW_RESTORE also un-maximizes a maximized/fullscreen window (it
            # shrank Roblox): only use it when Roblox is minimized
            if user32.IsIconic(self.hwnd):
                user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(self.hwnd)
            time.sleep(0.3)
        if cfg.move_cursor:
            x, y, w, h = self.client_rect()
            # middle of the screen, a bit above the character: open world, no buttons
            user32.SetCursorPos(x + w // 2, y + int(h * 0.30))

    # capture --------------------------------------------------------
    def grab(self, x0, y0, x1, y1):
        if self._mss is None:   # mss handles are per-thread
            self._mss = self._mss_cls()
        cx, cy, _, _ = self.client_rect()
        left, top = cx + x0, cy + y0
        shot = self._mss.grab({"left": left, "top": top, "width": x1 - x0, "height": y1 - y0})
        a = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        a = a[..., 2::-1]   # BGRA -> RGB
        # the macro's own window can sit on top of the game (it stays on top):
        # black it out so its buttons/graph aren't mistaken for notes or rings
        for l, t, r, b in self._own_rects():
            ix0, iy0 = max(l - left, 0), max(t - top, 0)
            ix1, iy1 = min(r - left, a.shape[1]), min(b - top, a.shape[0])
            if ix0 < ix1 and iy0 < iy1:
                if not a.flags.writeable:
                    a = a.copy()
                a[iy0:iy1, ix0:ix1] = 0
        return a

    def _own_rects(self):
        """Screen rects of this program's visible windows (refreshed twice a second)."""
        now = time.perf_counter()
        if now - getattr(self, "_own_t", -9.0) < 0.5:
            return self._own
        rects = []
        pid = os.getpid()

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def cb(hwnd, _):
            if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
                p = ctypes.c_ulong(0)
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
                if p.value == pid:
                    r = _RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(r))
                    if r.right > r.left and r.bottom > r.top:
                        rects.append((r.left, r.top, r.right, r.bottom))
            return True

        try:
            user32.EnumWindows(cb, 0)
        except Exception:
            rects = []
        self._own, self._own_t = rects, now
        return rects

    # input ----------------------------------------------------------
    def mouse_down(self):
        _send(_INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0))))


    def move_to(self, x, y):
        """Move the cursor to (x, y) in the game window - sent as a real mouse
        move (wind: the scroom follows the mouse up and down)."""
        cx, cy, _, _ = self.client_rect()
        vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)      # virtual screen
        vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
        nx = int(round((cx + x - vx) * 65535 / max(1, vw - 1)))
        ny = int(round((cy + y - vy) * 65535 / max(1, vh - 1)))
        _send(_INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
                                                   | MOUSEEVENTF_VIRTUALDESK, 0, 0))))

    def button_down(self, btn):
        flag = MOUSEEVENTF_RIGHTDOWN if btn == "right" else MOUSEEVENTF_LEFTDOWN
        _send(_INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flag, 0, 0))))

    def button_up(self, btn):
        flag = MOUSEEVENTF_RIGHTUP if btn == "right" else MOUSEEVENTF_LEFTUP
        _send(_INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flag, 0, 0))))

    def mouse_up(self):
        _send(_INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0))))

    def key_down(self, k):
        sc = SCANCODES.get(str(k).lower())
        if sc is not None:
            _send(_INPUT(1, _INPUTUNION(ki=_KEYBDINPUT(0, sc, KEYEVENTF_SCANCODE, 0, 0))))

    def key_up(self, k):
        sc = SCANCODES.get(str(k).lower())
        if sc is not None:
            _send(_INPUT(1, _INPUTUNION(ki=_KEYBDINPUT(0, sc, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0))))

    def key(self, k):
        sc = SCANCODES.get(str(k))
        if sc is None:
            return
        _send(_INPUT(1, _INPUTUNION(ki=_KEYBDINPUT(0, sc, KEYEVENTF_SCANCODE, 0, 0))))
        time.sleep(0.05)
        _send(_INPUT(1, _INPUTUNION(ki=_KEYBDINPUT(0, sc, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0))))

    # time -----------------------------------------------------------
    def now(self):
        return time.perf_counter()

    def sleep(self, s):
        time.sleep(s)

    def wait_until(self, deadline):
        """Sleep, then spin for the last few ms - time.sleep alone can overshoot."""
        left = deadline - time.perf_counter()
        if left > 0.004:
            time.sleep(left - 0.003)
        while time.perf_counter() < deadline:
            pass
