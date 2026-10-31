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

import detect


@dataclass
class Config:
    equip_key: str = "1"          # hotbar key for Tamer's Guitar ("" = don't press)
    equip_on_start: bool = True
    latency_ms: int = 60          # input -> screen delay to compensate for
    auto_latency: bool = True     # learn latency_ms from where ring clicks actually land
    ring_timeout_s: float = 5.0   # ring up longer than this -> re-equip instrument and recast
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
        self._last_shot = {}
        self._hold_start = None
        self._last_repress = -9.0
        self.stop_event = threading.Event()
        self.ring_finder = detect.RingFinder()
        self.freq_finder = detect.FreqFinder()
        self.reel_finder = detect.ReelFinder()

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

    def look(self, finder):
        w, h = self.io.client_size()
        x0, y0, x1, y1 = finder.region(w, h)
        img = self.io.grab(x0, y0, x1, y1)
        return finder.find(img, x0, y0, w, h)

    def ring(self):
        return self.look(self.ring_finder)

    def freq(self):
        return self.look(self.freq_finder)

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
            if not warned:
                self.set_phase("Paused", "Roblox isn't the active window")
                warned = True
            self.io.sleep(0.2)

    # --------------------------------------------------------------- main
    def run(self):
        self.stop_event.clear()
        try:
            self.io.prepare(self.cfg)
            if self.cfg.equip_on_start and self.cfg.equip_key and not self.cfg.dry_run:
                self.wait_focus()
                self.io.key(self.cfg.equip_key)
                self.io.sleep(0.6)
            errors = 0
            while not self.stopped():
                if self.cfg.max_catches and self.st.catches >= self.cfg.max_catches:
                    self.log(f"Reached {self.cfg.max_catches} catches")
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
                        self.log("5 errors in a row - stopping")
                        raise
                    self.io.sleep(1.0)
        finally:
            self.hold(False)
            self.set_phase("Stopped")

    def cycle(self):
        """One cast -> ring -> frequency -> reel attempt."""
        self.wait_focus()
        if self.stopped():
            return
        # already mid-minigame (e.g. started the macro late)? jump straight in
        if self.freq():
            self.do_freq()
            self.after_freq()
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
        self.set_phase("Waiting for frequency")
        while not self.stopped() and self.io.now() - t0 < 7:
            self.wait_focus()
            if self.freq():
                self.do_freq()
                self.after_freq()
                return
            if self.reel():
                self.do_reel()
                return
            self.io.sleep(0.02)
        self.snapshot("freq-not-found")
        self.log("Frequency panel never showed up - recasting")

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
                if self.freq():
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
            # first landing of the session: trust it fully; after that, average
            gain = 1.0 if self.landings == 0 else 0.4
            self.landings += 1
            new = (1 - gain) * self.cfg.latency_ms + gain * max(0.0, delay) * 1000
            self.cfg.latency_ms = int(round(min(300, new)))
            msg += f" -> latency now {self.cfg.latency_ms} ms"
        self.log(msg)

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
KEYEVENTF_SCANCODE, KEYEVENTF_KEYUP = 0x0008, 0x0002
SCANCODES = {"1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
             "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B}


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
            if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and self._title(hwnd) == "Roblox":
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
        if self._title(root) == "Roblox" and self._area(root) > 0:
            self.hwnd = root
            return True
        return False

    def prepare(self, cfg):
        self.find_window()
        if not self.hwnd:
            raise RuntimeError("Roblox window not found - open the game first")
        if not self.focused():
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
        shot = self._mss.grab({"left": cx + x0, "top": cy + y0, "width": x1 - x0, "height": y1 - y0})
        a = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        return a[..., 2::-1]   # BGRA -> RGB

    # input ----------------------------------------------------------
    def mouse_down(self):
        _send(_INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0))))


    def mouse_up(self):
        _send(_INPUT(0, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0))))

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
