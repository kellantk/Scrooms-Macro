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
    note_offset_ms: int = -30     # strings/keys: press this much later (+) or earlier (-) than the
                                  # usual timing (measured: the macro landed ~30 ms after Perfect's middle)
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


def _note_fit(tr, v_default):
    """Notes speed up as they come closer (the highway is in perspective):
    fit y = c + b*t + a*t^2 to the note's recent readings (t relative to the
    last one). Falls back to a straight line / default speed."""
    if len(tr.samples) < 2:
        return None
    t = np.array([s[0] for s in tr.samples])
    y = np.array([s[1] for s in tr.samples])
    t = t - t[-1]
    if len(t) >= 4 and t[-1] - t[0] >= 0.15:
        a, b, c0 = np.polyfit(t, y, 2)
        if b > 0.1 * v_default and a > -0.5 * v_default:
            return max(a, 0.0), b, c0
    b, c0 = np.polyfit(t, y, 1) if t[-1] - t[0] > 0.05 else (v_default, y[-1])
    return 0.0, (b if b > 0.1 * v_default else v_default), c0


def note_pos(tr, now, v_default):
    fit = _note_fit(tr, v_default)
    dt = now - tr.samples[-1][0]
    if fit is None:
        return tr.x + v_default * dt
    a, b, c0 = fit
    return c0 + b * dt + a * dt * dt


def time_to_reach(tr, now, y_target, v_default):
    """Seconds until the note's center reaches y_target."""
    if not tr.samples:
        return float("inf")
    fit = _note_fit(tr, v_default)
    if fit is None:
        return (y_target - tr.x) / v_default
    a, b, c0 = fit
    dt = now - tr.samples[-1][0]
    # solve a*T^2 + b*T + c0 = y_target for T >= dt
    if a > 1e-6:
        disc = b * b + 4 * a * (y_target - c0)
        T = (-b + np.sqrt(max(disc, 0.0))) / (2 * a)
    else:
        T = (y_target - c0) / b
    return float(T - dt)


# strings: a Perfect press lights the circle about this long after the note's
# middle crosses it (measured on manual Perfect hits: +33..+65 ms)
NOTE_PERFECT_MS = 45
STUCK_MINUTES = 10      # no catch for this long -> tell Discord


class NoteTimer:
    """Measures how late each strings press registered: when the note's white
    core crosses the circle's center vs when the circle lights up (the game
    took the key). Both are read from the same screen captures, so the
    capture delay cancels out."""

    def __init__(self, lanes):
        self.prev = {k: None for k in lanes}               # (t, core y) last read
        self.cross = {k: deque(maxlen=8) for k in lanes}   # crossing times
        self.pressed = {k: deque(maxlen=8) for k in lanes}
        self.pending = {k: deque() for k in lanes}         # lit times waiting for their crossing
        self.lit = {k: False for k in lanes}
        self.unlit_t = {k: None for k in lanes}            # last read with the circle dark
        self.errors = []                                   # ms, lit - crossing

    def press(self, lane, t):
        self.pressed[lane].append(t)

    def update(self, marks, cy, r, now):
        for lane, (wy, lit) in marks.items():
            if lane not in self.prev:
                continue
            p = self.prev[lane]
            if wy is not None and p is not None and p[1] < cy <= wy and wy - p[1] < 1.5 * r and now > p[0]:
                self.cross[lane].append(p[0] + (cy - p[1]) / (wy - p[1]) * (now - p[0]))
            self.prev[lane] = (now, wy) if wy is not None else None
            # a press fills the whole circle with no note left in it (a note
            # sliding over the circle only covers ~0.6-0.7 of it)
            on = lit > 0.9 and (wy is None or wy < cy - 1.2 * r)
            if on and not self.lit[lane]:
                u = self.unlit_t[lane]
                # lit somewhere between the last dark read and this one
                self.pending[lane].append((u + now) / 2 if u is not None and now - u < 0.05 else now)
            if not on:
                self.unlit_t[lane] = now
            self.lit[lane] = on
            # an early press lights the circle before the note crosses: wait a bit
            pend = self.pending[lane]
            while pend and now - pend[0] > 0.15:
                self._pair(lane, pend.popleft())

    def _pair(self, lane, t_lit):
        if not any(-0.05 < t_lit - tp < 0.35 for tp in self.pressed[lane]):
            return          # not one of our presses
        c = [x for x in self.cross[lane] if -0.1 < t_lit - x < 0.3]
        if c:
            x = min(c, key=lambda x: abs(t_lit - x - NOTE_PERFECT_MS / 1000))
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
            if diff < -0.15:
                diff += 2 * math.pi
            t_hit = diff / abs(omega)
            frame_dt = max(sum(loop_dt) / len(loop_dt), 0.008) if loop_dt else 0.03
            wait = t_hit - lat
            if wait <= frame_dt:
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
        Read that spot and nudge latency_ms so the next click lands on target."""
        # frozen = the needle stays put for 0.15 s+ (a spinning one covers ~45 deg
        # in that time). Repeated identical frames alone don't count.
        reads = deque()
        landed = None
        while not self.stopped() and self.io.now() - t_click < 1.4:
            now = self.io.now()
            if now - t_click < 0.2:      # ring flashes right after the click
                self.io.sleep(0.02)
                continue
            r = self.ring()
            if r is None or r.needle is None:
                if self.freq():
                    break
                self.io.sleep(0.01)
                continue
            reads.append((now, r.needle))
            while reads and now - reads[0][0] > 0.2:
                reads.popleft()
            if len(reads) >= 3 and reads[-1][0] - reads[0][0] >= 0.15:
                a = np.array([x[1] for x in reads])
                center = _circ_center(a, math.radians(4))
                if np.abs((a - center + np.pi) % (2 * np.pi) - np.pi).max() < math.radians(3):
                    landed = center
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

    def tune_notes(self, errs):
        """Nudge note_offset_ms so strings presses land where Perfect is.
        errs: recent presses, ms the key registered after the note's middle."""
        if not self.cfg.auto_note_timing or self.cfg.dry_run:
            return
        med = float(np.median(errs))
        diff = med - NOTE_PERFECT_MS
        if abs(diff) < 8:                      # already in the Perfect window
            return
        old = self.cfg.note_offset_ms
        step = max(-25.0, min(25.0, 0.6 * diff))
        self.cfg.note_offset_ms = int(round(max(-150, min(150, old - step))))
        self.log(f"Note timing: presses landed {abs(diff):.0f} ms {'late' if diff > 0 else 'early'} "
                 f"-> note timing {old} -> {self.cfg.note_offset_ms} ms")

    # ---------------------------------------------------------- count-in
    def do_countin(self):
        """Rings shrink onto the circle around the scroom: click as each one
        touches it (radius ~1.12x the circle, where the game gives Perfect).
        The ring is hard to see in the last moment, so its arrival is predicted
        from how fast it's shrinking."""
        lat = self.cfg.latency_ms / 1000
        self.set_phase("Count-in", "clicking the rings")
        self.log("Count-in")
        h = self.io.client_size()[1]
        tr = Tracker(window=0.3)
        done = False
        clicks = 0
        gone_since = None
        loop_dt = deque(maxlen=10)
        prev_t = None
        t_start = self.io.now()
        while not self.stopped() and self.io.now() - t_start < 12:
            self.wait_focus()
            now = self.io.now()
            if prev_t is not None:
                loop_dt.append(now - prev_t)
            prev_t = now
            res = self.look(self.countin_finder)
            if res is None:
                gone_since = gone_since or now
                if now - gone_since > 0.6:
                    break
                continue
            gone_since = None
            target = 1.12 * res.r0
            r = res.ring
            if r is not None:
                if tr.x is not None and r > tr.x + 0.2 * res.r0:      # a new ring came in
                    tr.reset()
                    done = False
                if tr.x is None or abs(r - tr.x) > 0.5:
                    tr.add(now, r)
            if done or tr.x is None or clicks >= 4:      # there are 4 rings
                continue
            if now - tr.samples[-1][0] > 0.4:
                tr.reset()
                continue
            # only click on a ring we've watched shrink for a bit: a ring that
            # isn't moving means the game is paused (first-time tutorial)
            if len(tr.samples) < 4 or tr.samples[-1][0] - tr.samples[0][0] < 0.1:
                continue
            v = tr.v
            if v > -0.03 * h:
                continue
            r_now = tr.x + v * (now - tr.samples[-1][0])
            t_hit = (r_now - target) / -v
            self.st.graph = {"mode": "countin", "r0": res.r0, "ring": r_now, "target": target, "clicks": clicks, "now": now}
            frame_dt = max(sum(loop_dt) / len(loop_dt), 0.005) if loop_dt else 0.03
            if t_hit - lat <= frame_dt * 0.5:
                self.click()
                clicks += 1
                done = True
                self.set_phase("Count-in", f"{clicks} ring{'s' if clicks != 1 else ''}")
        self.log(f"Count-in done ({clicks} clicks)")

    # ----------------------------------------------------------- strings
    def do_strings(self):
        """Strings: 3-lane highway, keys A S D (notes speed up - perspective)."""
        self.do_lanes("Strings", self.strings, strings.LANES, 0.38, self.strings_finder,
                      keybinds=self.cfg.strings_keys)

    def do_keys(self):
        """Keys: Piano-Tiles, 4 columns, keys A S D F (tiles fall at a steady speed)."""
        self.do_lanes("Keys", self.keys, keys.LANES, 0.22, keybinds=self.cfg.keys_keys, steady=True)

    def do_lanes(self, name, look, lanes, v_frac, finder=None, keybinds="", steady=False):
        """Note lanes: press a lane's key as its note reaches the hit spot.
        Notes are tracked per lane; the key goes down `latency` before the
        note's predicted arrival, so it registers right on the spot.
        Hold notes (strings: a bar trailing the note) keep the key down until
        the bar has slid past the circle.
        keybinds: the keys you set in the game for the lanes, left to right
        (the game lets you change them; default = the lane names)."""
        self.set_phase(name, "notes coming")
        self.log(f"{name} minigame")
        kb = (keybinds or "").replace(" ", "").lower()
        ok = len(kb) == len(lanes) and all(c in SCANCODES for c in kb)
        km = {lane: (kb[i] if ok else lane) for i, lane in enumerate(lanes)}
        if keybinds and not ok:
            self.log(f"{name}: keybinds '{keybinds}' aren't {len(lanes)} letter keys - using {''.join(lanes)}")
        timer = NoteTimer(lanes) if finder is not None and hasattr(finder, "marks") else None
        tuned = 0                              # timer.errors already used for tuning
        blocked = {}                           # lane -> no presses until this time
        last_hw = None
        track = {k: Tracker(window=0.6) for k in lanes}
        is_hold = {k: False for k in lanes}    # the note we follow is a hold note
        holding = {}                           # lane -> [time down, reads without the bar]
        done_y = {k: None for k in lanes}      # the note we already pressed for
        done_t = {k: -9.0 for k in lanes}
        release_at = {}                                  # key -> time to let go
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
                holds = getattr(finder, "holds", None) or {}
                bar_near = getattr(finder, "bar_near", None) or {}
                frame_dt = max(sum(loop_dt) / len(loop_dt), 0.005) if loop_dt else 0.02
                lat = (self.cfg.latency_ms - self.cfg.note_offset_ms) / 1000
                if timer is not None:
                    timer.update(finder.marks, hw.cy, hw.r, now)
                    if len(timer.errors) - tuned >= 4:
                        tuned = len(timer.errors)
                        self.tune_notes(timer.errors[-8:])
                v_default = v_frac * self.io.client_size()[1]      # px/s near the hit spot
                # hold notes: let go once the bar has passed (2 reads in a row
                # without it, so one bad frame doesn't drop the hold)
                for lane, h in list(holding.items()):
                    h[1] = h[1] + 1 if bar_near.get(lane, 0.0) < 0.3 else 0
                    if (h[1] >= 2 and now - h[0] > 0.12) or now - h[0] > 8.0:
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
                            y = note_pos(tr, now, v_default)
                            if y > hw.cy + hw.r:          # long past the circle: it's gone
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
                        tr.reset()
                    # still the note we already pressed for? ignore it until it's gone
                    if done_y[lane] is not None and y >= done_y[lane] - 1.5 * hw.r and now - done_t[lane] < 0.5:
                        continue
                    done_y[lane] = None
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
                    t_hit = time_to_reach(tr, now, hw.cy, v_default)
                    upcoming[lane] = [t_hit]
                    # due before the next read comes in: press it this round,
                    # at the exact moment (not up to a whole frame early/late)
                    if t_hit - lat <= frame_dt:
                        due.append((now + t_hit - lat, lane, y))
                # the board isn't on screen in this read (the minigame is closing):
                # a key pressed now would walk the character (A/S/D move it)
                due = [d for d in due if now >= blocked.get(d[1], 0.0)]
                for t_press, lane, y in sorted(due):
                    if t_press > self.io.now():
                        self.io.wait_until(min(t_press, now + 2 * frame_dt))
                    tp = self.io.now()
                    self.press_log.append((tp, lane))
                    if timer is not None:
                        timer.press(lane, tp)
                    if not self.cfg.dry_run:
                        if lane in release_at:
                            self.io.key_up(km[lane])
                            del release_at[lane]
                        self.io.key_down(km[lane])
                        if not is_hold[lane]:
                            release_at[lane] = tp + 0.05
                    if is_hold[lane]:
                        holding[lane] = [tp, 0]
                    presses += 1
                    done_y[lane], done_t[lane] = y, tp
                    track[lane].reset()
                    self.set_phase(name, f"{'holding' if is_hold[lane] else 'pressed'} {lane}  ({presses} notes)")
                self.st.graph = {"mode": "lanes", "name": name, "lanes": list(lanes), "notes": upcoming,
                                 "presses": list(self.press_log), "now": now}
        finally:
            for k in set(release_at) | set(holding):
                self.io.key_up(km[k])
        self.log(f"{name} minigame finished ({presses} notes pressed)")
        if timer is not None and timer.errors:
            e = timer.errors
            self.log(f"Note timing: {len(e)} presses measured, keys landed {np.median(e):+.0f} ms after the notes' "
                     f"middle (Perfect is about {NOTE_PERFECT_MS:+d}), spread {min(e):+.0f}..{max(e):+.0f}")
            if len(e) - tuned >= 2:
                self.tune_notes(e[-8:])
        if presses:
            self.minigame_done(name, f"{presses} notes hit")

    # -------------------------------------------------------------- wind
    def do_wind(self):
        """Wind: a 5-line staff. Steer the scroom onto the next note's line by
        moving the mouse up/down, and hold left click while that note's bar
        slides past the scroom. All notes move left at the same speed, so each
        one is turned into a time window (head arrives .. tail passes)."""
        self.set_phase("Wind", "notes coming")
        self.log("Wind minigame")
        lead = self.cfg.latency_ms / 1000
        sched = []                      # [line, on, off, on_seen, off_seen, id]
        next_id = 0
        speeds = deque(maxlen=20)       # px/s samples
        past = deque()                  # (t, readings) of the last ~0.1 s
        ref_st = cand = None            # staff: where it settled / where it's going
        early_for = None                # note we already left early for the next one
        moved_since = 0.0
        speed = None                    # note speed as a fraction of the staff per second
        learned = {}                    # line -> cursor y that put the scroom on it
        nudges = 0
        cur_line = None
        moving = None                   # (time the cursor moved, line) until the scroom gets there
        move_times = deque(maxlen=9)
        ys = deque()                    # scroom readings of the last 0.15 s (t, y)
        cur = None                      # cursor y we sent, and when
        cur_t = 0.0
        line = None
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
                            shifts = [getattr(p, k) - a for p in past[0][1] if p.line == n.line
                                      and sx + 0.07 * L < getattr(p, k) < right]
                            shifts = [d for d in shifts if 0.15 * L * dt < d < 1.2 * L * dt]
                            if shifts:
                                speeds.append(min(shifts) / dt)
                v = float(np.median(speeds)) if len(speeds) >= 3 else 0.43 * L
                # ---- turn readings into time windows
                for n in reads:
                    lo, hi = now + (n.head - sx) / v, now + (n.tail - sx) / v
                    on = lo if sx + 0.07 * L < n.head < right else None
                    off = hi if sx + 0.07 * L < n.tail < right else None
                    m = next((s for s in sched if s[0] == n.line and lo < s[2] + 0.1 and hi > s[1] - 0.1), None)
                    if m is None:
                        sched.append([n.line, lo, hi, on is not None, off is not None, next_id])
                        next_id += 1
                        continue
                    if on is not None:
                        m[1] = on if not m[3] else 0.6 * m[1] + 0.4 * on
                        m[3] = True
                    elif not m[3]:
                        m[1] = min(m[1], lo)
                    if off is not None:
                        m[2] = off if not m[4] else 0.6 * m[2] + 0.4 * off
                        m[4] = True
                    elif not m[4]:
                        m[2] = max(m[2], hi)          # the bar is still coming in
                sched = sorted((s for s in sched if s[2] > now - 0.3), key=lambda s: s[1])
                # ---- what to do now (judged `lead` ahead: the game sees it later)
                t = now + lead
                active = next((s for s in sched if s[1] <= t < s[2]), None)
                nxt = next((s for s in sched if s[1] > t), None)
                if active is not None:
                    line = active[0]
                    # the next note starts right after this one on another line:
                    # start moving over just before this one ends (and stick to it)
                    early = float(np.clip(0.5 * np.median(move_times), 0.04, 0.2)) if move_times else 0.08
                    if nxt is not None and nxt[0] != line and (
                            early_for == active[5] or (nxt[1] - active[2] < 0.3 and t > active[2] - early)):
                        line = nxt[0]
                        early_for = active[5]
                elif nxt is not None:
                    line = nxt[0]
                # ---- steer: cursor y -> scroom y, learned from where it settles
                if sy is not None:
                    ys.append((now, sy))
                while ys and now - ys[0][0] > 0.15:
                    ys.popleft()
                if line is not None:
                    want = st.lines[line]
                    # cursor y for this line: what worked before, else a guess from
                    # the lines learned so far (the scroom should just follow the
                    # cursor's height, but that's checked, not assumed)
                    known = sorted(learned.items())
                    if line in learned:
                        y_cur = learned[line]
                    elif len(known) >= 2:
                        k, c = np.polyfit([st.lines[i] for i, _ in known], [c for _, c in known], 1)
                        y_cur = k * want + c
                    elif known:
                        y_cur = want + known[0][1] - st.lines[known[0][0]]
                    else:
                        y_cur = want
                    y_cur = float(np.clip(y_cur, 2, self.io.client_size()[1] - 3))
                    if cur is None or abs(y_cur - cur) >= 1:
                        if not self.cfg.dry_run:
                            self.io.move_to(min(st.x0 + 1.06 * L, self.io.client_size()[0] - 5), y_cur)
                        if cur_line is not None and line != cur_line:
                            moving = (now, line)        # time how long the scroom takes to get there
                        cur, cur_t, cur_line = y_cur, now, line
                    if moving is not None and sy is not None:
                        if moving[1] != line or now - moving[0] > 1.0:
                            moving = None
                        elif abs(sy - st.lines[moving[1]]) < 0.15 * st.spacing:
                            move_times.append(now - moving[0])
                            moving = None
                    # settled (cursor still for a bit, scroom not moving)? check it
                    # sits on the line; if not, nudge the cursor and remember
                    if (cur_line == line and now - cur_t > 0.25 and len(ys) >= 3 and ys[-1][0] - ys[0][0] > 0.08
                            and np.ptp([y for _, y in ys]) < 0.015 * st.D):
                        err = want - ys[-1][1]
                        if abs(err) <= 0.06 * st.spacing:
                            learned[line] = cur
                        else:
                            slope = 1.0      # cursor px per scroom px
                            if len(known) >= 2:
                                k = np.polyfit([st.lines[i] for i, _ in known], [c for _, c in known], 1)[0]
                                slope = float(np.clip(k, 0.3, 6))
                            learned[line] = float(np.clip(cur + slope * err, 2, self.io.client_size()[1] - 3))
                            nudges += 1
                            if nudges == 6:
                                self.log("Wind: the scroom doesn't sit where the mouse is - correcting")
                            if nudges == 30 and not warned:
                                warned = True
                                self.log("Wind: the scroom doesn't follow the mouse")
                                self.snapshot("wind-no-steer")
                # ---- hold while a note's bar is at the scroom
                want_hold = active is not None
                if want_hold and held_id is not None and active[5] != held_id and self.held:
                    self.hold(False)            # a new note right after the last one: press again
                    self.io.sleep(0.03)
                if want_hold and not self.held:
                    self.hold(True)
                    played += 1
                    speed = v / L
                    self.set_phase("Wind", f"holding line {active[0] + 1}  ({played} notes)")
                elif not want_hold and self.held:
                    self.hold(False)
                held_id = active[5] if want_hold else None
                sline = None if sy is None else (sy - st.top) / st.D
                self.st.graph = {"mode": "staff", "now": now, "scroom": sline, "hold": want_hold,
                                 "notes": [(s[0], s[1] - now, s[2] - now) for s in sched], "target": line}
        finally:
            self.hold(False)
        self.log(f"Wind minigame finished ({played} notes played"
                 + (f", notes moved {speed:.2f} staff/s)" if speed else ")"))
        if played:
            self.minigame_done("Wind", f"{played} notes played")

    # -------------------------------------------------------- percussion
    def do_perc(self):
        """One lane, notes slide left to the scroom: white = left click, blue =
        right click, big = both. Notes move at a steady speed and slip under
        the scroom's diamond just before arriving, so the arrival is predicted."""
        lat = self.cfg.latency_ms / 1000
        self.set_phase("Percussion", "notes coming")
        self.log("Percussion minigame")
        tr = Tracker(window=0.4)
        kind = None
        done_x, done_t = None, -9.0
        release_at = {}
        gone_since = None
        loop_dt = deque(maxlen=10)
        prev_t = None
        hits = 0
        buttons = {"L": ("left",), "R": ("right",), "B": ("left", "right")}
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
                v_default = -0.424 * self.io.client_size()[1]      # px/s (moving left)
                # ignore the note we already hit until it disappears
                if done_x is not None and now - done_t < 0.4:
                    notes = [n for n in notes if n[0] > done_x + 0.03 * lane.L]
                lead = min(notes) if notes else None
                # the note we follow just slipped under the scroom's diamond, and the
                # next one (further right) is now the first visible - keep following
                # the hidden one until we've hit it
                if (lead is not None and tr.x is not None and lead[0] > tr.x + 0.05 * lane.L
                        and tr.x < lane.target_x + 0.15 * lane.L and now - tr.samples[-1][0] < 0.4):
                    lead = None
                if lead is not None:
                    x, k = lead
                    if tr.x is not None and x > tr.x + 0.05 * lane.L:   # a new note
                        tr.reset()
                        kind = None
                    if tr.x is None or abs(x - tr.x) > 0.5:
                        tr.add(now, x)
                    elif now - tr.samples[-1][0] > 0.15:   # hasn't moved: paused
                        tr.reset()
                        tr.add(now, x)
                        continue
                    # a big note loses its white half first as it slides under the
                    # scroom's diamond - once seen as big, it stays big
                    kind = "B" if (k == "B" or kind == "B") else k
                elif tr.x is None or now - tr.samples[-1][0] > 0.4:
                    tr.reset()
                    continue
                # where is it now (it slips under the scroom's diamond near the end)?
                if len(tr.samples) >= 3 and tr.samples[-1][0] - tr.samples[0][0] > 0.1:
                    v = tr.v
                    if v > 0.3 * v_default:      # (v_default is negative) not moving:
                        continue                 # the game is paused (tutorial) - press nothing
                else:
                    v = v_default
                x_now = tr.x + v * (now - tr.samples[-1][0])
                t_hit = (x_now - lane.target_x) / -v
                gl = {"L": ["L"], "R": ["R"], "B": ["L", "R"]}
                upcoming = {"L": [], "R": []}
                for nx, nk in notes:
                    for g in gl.get(nk, []):
                        upcoming[g].append((nx - lane.target_x) / -v)
                for g in gl.get(kind, []):
                    upcoming[g].append(t_hit)
                self.st.graph = {"mode": "lanes", "name": "Percussion", "lanes": ["L", "R"], "labels": ["Left", "Right"],
                                 "notes": upcoming, "presses": list(self.press_log), "now": now}
                if t_hit - lat <= frame_dt * 0.5:
                    for g in gl.get(kind, ["L"]):
                        self.press_log.append((now, g))
                    if not self.cfg.dry_run:
                        for btn in buttons.get(kind, ("left",)):
                            if btn in release_at:
                                self.io.button_up(btn)
                            self.io.button_down(btn)
                            release_at[btn] = now + 0.05
                    hits += 1
                    done_x, done_t = x_now, now
                    tr.reset()
                    kind = None
                    self.set_phase("Percussion", f"{ {'L': 'left', 'R': 'right', 'B': 'both'}.get(kind)} click  ({hits} notes)")
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
SCANCODES = {"1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
             "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B}
SCANCODES.update(zip("qwertyuiop", range(0x10, 0x1A)))
SCANCODES.update(zip("asdfghjkl", range(0x1E, 0x27)))
SCANCODES.update(zip("zxcvbnm", range(0x2C, 0x33)))
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
