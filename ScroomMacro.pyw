"""Scroom Macro - the app window (status, live graph, settings)."""
import json
import math
import os
import sys
import threading
import time
import tkinter as tk
import webbrowser
from dataclasses import asdict
from tkinter import messagebox, ttk

import startup_check

startup_check.run()   # before numpy loads: safe settings + clear errors

import bot  # noqa: E402
import hotkeys  # noqa: E402
import notify  # noqa: E402
import updater  # noqa: E402

APP = "Scroom Macro"
VERSION = "2.0"
REPO_URL = "https://github.com/kellantk/Scrooms-Macro"
BASE = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
SETTINGS = os.path.join(BASE, "settings.json")

# ---------------------------------------------------------------- theme
BG = "#101217"
CARD = "#181b22"
FIELD = "#20242d"
BORDER = "#2a2f3a"
TEXT = "#e9ebf1"
MUTED = "#8a91a0"
ACCENT = "#8b5cf6"
ACCENT_HI = "#a78bfa"
GOOD = "#22c55e"
BAD = "#ef4444"
WARN = "#f59e0b"
FONT = "Segoe UI"
LANE_COLORS = {
    "Strings": {"A": "#4ade80", "S": "#f87171", "D": "#facc15"},
    "Keys": {"A": "#e5e7eb", "S": "#e5e7eb", "D": "#e5e7eb", "F": "#e5e7eb"},
    "Percussion": {"L": "#e5e7eb", "R": "#60a5fa"},
}
PHASE_COLORS = {"Stopped": MUTED, "Paused": WARN, "Error": BAD, "Cooldown": MUTED,
                "Casting": ACCENT_HI, "Waiting for minigame": ACCENT_HI}


def load_settings():
    cfg, hotkey = bot.Config(), "f6"
    try:
        with open(SETTINGS) as f:
            data = json.load(f)
        hotkey = data.pop("hotkey", hotkey)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
    except (OSError, ValueError, TypeError):
        pass
    return cfg, hotkey


def save_settings(cfg, hotkey):
    try:
        data = asdict(cfg)
        data.pop("dry_run", None)
        with open(SETTINGS, "w") as f:
            json.dump({**data, "hotkey": hotkey}, f, indent=2)
    except OSError:
        pass


def setup_style(root):
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=BG, foreground=TEXT, font=(FONT, 10), bordercolor=BORDER,
                lightcolor=BORDER, darkcolor=BORDER, troughcolor=FIELD, focuscolor=ACCENT)
    s.configure("Card.TFrame", background=CARD)
    s.configure("TFrame", background=BG)
    s.configure("TLabel", background=CARD, foreground=TEXT)
    s.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=(FONT, 9))
    s.configure("TCheckbutton", background=CARD, foreground=TEXT, indicatorbackground=FIELD,
                indicatorforeground=TEXT, padding=(0, 4))
    s.map("TCheckbutton", background=[("active", CARD)], indicatorbackground=[("selected", ACCENT), ("active", BORDER)])
    for w in ("TEntry", "TSpinbox", "TCombobox"):
        s.configure(w, fieldbackground=FIELD, foreground=TEXT, background=FIELD, arrowcolor=MUTED,
                    insertcolor=TEXT, padding=5, selectbackground=ACCENT, selectforeground="white")
        s.map(w, fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
              bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)], darkcolor=[("focus", ACCENT)])
    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
    s.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(14, 7), borderwidth=0, font=(FONT, 10))
    s.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", TEXT)],
          lightcolor=[("selected", CARD)], bordercolor=[("selected", CARD)])
    s.configure("Ghost.TButton", background=FIELD, foreground=TEXT, borderwidth=0, padding=(10, 6), font=(FONT, 9))
    s.map("Ghost.TButton", background=[("active", BORDER), ("pressed", BORDER)])
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.font", (FONT, 10))


class FlatButton(tk.Label):
    """A big flat button (tk buttons can't be styled cleanly on Windows)."""

    def __init__(self, master, text, command, bg=ACCENT, hover=ACCENT_HI, **kw):
        super().__init__(master, text=text, bg=bg, fg="white", font=(FONT, 12, "bold"),
                         pady=11, cursor="hand2", **kw)
        self.command, self.base, self.hover = command, bg, hover
        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", lambda e: self.configure(bg=self.hover))
        self.bind("<Leave>", lambda e: self.configure(bg=self.base))

    def set_colors(self, bg, hover):
        self.base, self.hover = bg, hover
        self.configure(bg=bg)


class App:
    def __init__(self, root):
        self.root = root
        self.cfg, self.hotkey = load_settings()
        self.status = bot.Status()
        self.bot = None
        self.thread = None
        self.toggle_requested = False
        self.hk = hotkeys.GlobalHotkey(self.request_toggle)
        self.log_win = None
        self.update_info = None

        root.title(APP)
        root.resizable(False, False)
        try:
            root.iconbitmap(os.path.join(getattr(sys, "_MEIPASS", BASE), "scroom.ico"))
        except tk.TclError:
            pass
        setup_style(root)

        outer = tk.Frame(root, bg=BG, padx=16, pady=14)
        outer.pack(fill="both", expand=True)

        # ---- header ----------------------------------------------------
        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text=APP, bg=BG, fg=TEXT, font=(FONT, 15, "bold")).pack(side="left")
        tk.Label(head, text=f"v{VERSION}", bg=BORDER, fg=MUTED, font=(FONT, 8, "bold"), padx=6).pack(side="left", padx=8, pady=(4, 0))
        self.update_pill = tk.Label(head, text="", bg=BG, fg=GOOD, font=(FONT, 9, "bold"), cursor="hand2")
        self.update_pill.pack(side="right")
        self.update_pill.bind("<Button-1>", lambda e: self.update_info and webbrowser.open(self.update_info[1]))

        # ---- status card -----------------------------------------------
        card = tk.Frame(outer, bg=CARD, padx=14, pady=12, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x", pady=(12, 0))
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x")
        self.dot = tk.Canvas(row, width=12, height=12, bg=CARD, highlightthickness=0)
        self.dot.pack(side="left", pady=(2, 0))
        self.dot_id = self.dot.create_oval(1, 1, 11, 11, fill=MUTED, outline="")
        self.l_phase = tk.Label(row, text="Stopped", bg=CARD, fg=TEXT, font=(FONT, 13, "bold"))
        self.l_phase.pack(side="left", padx=(8, 0))
        self.l_fps = tk.Label(row, text="", bg=CARD, fg=MUTED, font=(FONT, 9))
        self.l_fps.pack(side="right")
        self.l_detail = tk.Label(card, text="Press Start or your hotkey while Roblox is open", bg=CARD, fg=MUTED,
                                 font=(FONT, 9), anchor="w")
        self.l_detail.pack(fill="x", pady=(2, 10))

        tiles = tk.Frame(card, bg=CARD)
        tiles.pack(fill="x")
        self.tiles = {}
        for i, (key, label) in enumerate((("finished", "Catches"), ("casts", "Casts"), ("rate", "Per hour"), ("uptime", "Running"))):
            t = tk.Frame(tiles, bg=FIELD, padx=10, pady=7)
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
            tiles.columnconfigure(i, weight=1, uniform="t")
            v = tk.Label(t, text="0", bg=FIELD, fg=TEXT, font=(FONT, 15, "bold"))
            v.pack(anchor="w")
            tk.Label(t, text=label, bg=FIELD, fg=MUTED, font=(FONT, 8)).pack(anchor="w")
            self.tiles[key] = v

        self.graph = tk.Canvas(card, width=470, height=150, bg=FIELD, highlightthickness=0)
        self.graph.pack(fill="x", pady=(10, 0))

        # ---- start button ----------------------------------------------
        self.b_start = FlatButton(outer, "", self.toggle)
        self.b_start.pack(fill="x", pady=(12, 12))

        # ---- settings tabs ---------------------------------------------
        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True)
        self._build_general(self._tab(nb, "General"))
        self._build_timing(self._tab(nb, "Timing"))
        self._build_discord(self._tab(nb, "Discord"))
        self._build_more(self._tab(nb, "More"))

        self.update_start_button()
        self.apply_hotkey()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        threading.Thread(target=self._check_update, daemon=True).start()
        self.poll()

    # ------------------------------------------------------------- tabs
    def _tab(self, nb, title):
        f = tk.Frame(nb, bg=CARD, padx=14, pady=10)
        nb.add(f, text=title)
        f.columnconfigure(1, weight=1)
        return f

    def _row(self, parent, r, label, widget, hint=None):
        tk.Label(parent, text=label, bg=CARD, fg=TEXT, font=(FONT, 10)).grid(row=r, column=0, sticky="w", pady=5)
        widget.grid(row=r, column=1, sticky="e", pady=5)
        if hint:
            tk.Label(parent, text=hint, bg=CARD, fg=MUTED, font=(FONT, 8)).grid(row=r + 1, column=0, columnspan=2,
                                                                               sticky="w", pady=(0, 4))

    def _check(self, parent, r, text, var):
        ttk.Checkbutton(parent, text=text, variable=var).grid(row=r, column=0, columnspan=2, sticky="w")

    def _build_general(self, f):
        self.v_key = tk.StringVar(value=self.cfg.equip_key or "1")
        self.v_equip = tk.BooleanVar(value=self.cfg.equip_on_start)
        self.v_countin = tk.BooleanVar(value=self.cfg.count_in)
        self.v_aim = tk.StringVar(value="Perfect" if self.cfg.aim_perfect else "Middle")
        self.v_repeat_n = tk.IntVar(value=self.cfg.max_catches)
        self.v_cursor = tk.BooleanVar(value=self.cfg.move_cursor)
        self._row(f, 0, "Instrument hotbar slot", ttk.Combobox(f, textvariable=self.v_key, width=5, state="readonly",
                                                           values=[str(i) for i in range(1, 10)] + ["0"]))
        self._check(f, 1, "Equip it when starting", self.v_equip)
        self._check(f, 2, "Count-in: click the head-start rings", self.v_countin)
        self._row(f, 3, "Cast circle aim", ttk.Combobox(f, textvariable=self.v_aim, width=9, state="readonly",
                                                     values=["Perfect", "Middle"]))
        self._row(f, 4, "Stop after (catches)", ttk.Spinbox(f, from_=0, to=9999, width=7, textvariable=self.v_repeat_n),
                  "0 = keep going until you stop it")
        self._check(f, 6, "Move the cursor to a safe spot when starting", self.v_cursor)

    def _build_timing(self, f):
        self.v_latency = tk.IntVar(value=self.cfg.latency_ms)
        self.v_auto = tk.BooleanVar(value=self.cfg.auto_latency)
        self.v_cooldown = tk.DoubleVar(value=self.cfg.cooldown_s)
        self._row(f, 0, "Latency (ms)", ttk.Spinbox(f, from_=0, to=300, increment=5, width=7, textvariable=self.v_latency),
                  "How early to press, to make up for your PC's delay")
        self._check(f, 2, "Auto-tune latency from where circle clicks land", self.v_auto)
        self._row(f, 3, "Wait between catches (s)",
                  ttk.Spinbox(f, from_=0, to=10, increment=0.5, width=7, textvariable=self.v_cooldown))

    def _build_discord(self, f):
        self.v_webhook = tk.StringVar(value=self.cfg.webhook_url)
        self.v_notify = tk.BooleanVar(value=self.cfg.notify_catches)
        self.v_report = tk.IntVar(value=self.cfg.report_minutes)
        tk.Label(f, text="Webhook URL", bg=CARD, fg=TEXT, font=(FONT, 10)).grid(row=0, column=0, columnspan=2, sticky="w")
        e = ttk.Entry(f, textvariable=self.v_webhook, show="•", width=52)
        e.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 2))
        tk.Label(f, text="Discord: channel settings > Integrations > Webhooks. It's only ever sent to Discord.",
                 bg=CARD, fg=MUTED, font=(FONT, 8)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._check(f, 3, "Post every catch, with a screenshot of the result", self.v_notify)
        self._row(f, 4, "Session report every (min)", ttk.Spinbox(f, from_=0, to=240, width=7, textvariable=self.v_report),
                  "0 = off. A final report is also sent when you stop.")
        bar = tk.Frame(f, bg=CARD)
        bar.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(bar, text="Send test message", style="Ghost.TButton", command=self.test_webhook).pack(side="left")
        self.l_webhook = tk.Label(bar, text="", bg=CARD, fg=MUTED, font=(FONT, 9))
        self.l_webhook.pack(side="left", padx=10)

    def _build_more(self, f):
        self.b_hotkey = ttk.Button(f, text=f"Start/Stop key: {self.hotkey.upper()}", style="Ghost.TButton",
                                   command=self.hotkey_dialog)
        btns = [(self.b_hotkey, None),
                (ttk.Button(f, text="Test detection (no clicks)", style="Ghost.TButton",
                            command=lambda: self.start(dry=True)), None),
                (ttk.Button(f, text="Show log", style="Ghost.TButton", command=self.show_log), None),
                (ttk.Button(f, text="Make bug report", style="Ghost.TButton", command=self.bug_report), None),
                (ttk.Button(f, text="Check for updates", style="Ghost.TButton", command=self.check_update_now), None),
                (ttk.Button(f, text="Source code on GitHub", style="Ghost.TButton",
                            command=lambda: webbrowser.open(REPO_URL)), None)]
        for i, (b, _) in enumerate(btns):
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0 if i % 2 == 0 else 6, 0), pady=4)
        f.columnconfigure(0, weight=1, uniform="m")
        f.columnconfigure(1, weight=1, uniform="m")
        self.l_update = tk.Label(f, text="", bg=CARD, fg=MUTED, font=(FONT, 9))
        self.l_update.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------ config
    def read_cfg(self, quiet=False):
        c = self.cfg
        try:
            c.latency_ms = int(self.v_latency.get())
            c.cooldown_s = float(self.v_cooldown.get())
            c.max_catches = max(0, int(self.v_repeat_n.get()))
            c.report_minutes = max(0, int(self.v_report.get()))
        except (tk.TclError, ValueError):
            if not quiet:
                messagebox.showerror(APP, "Please enter numbers in the number boxes.")
            return None
        c.auto_latency = self.v_auto.get()
        c.equip_key = self.v_key.get()
        c.equip_on_start = self.v_equip.get()
        c.count_in = self.v_countin.get()
        c.aim_perfect = self.v_aim.get() == "Perfect"
        c.move_cursor = self.v_cursor.get()
        c.webhook_url = self.v_webhook.get().strip()
        c.notify_catches = self.v_notify.get()
        save_settings(c, self.hotkey)
        return c

    # ----------------------------------------------------------- running
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, dry=False):
        if self.running():
            return
        cfg = self.read_cfg()
        if cfg is None:
            return
        if cfg.webhook_url and not notify.valid_url(cfg.webhook_url):
            messagebox.showwarning(APP, "The Discord webhook URL doesn't look right - Discord posts are off until it's fixed.")
        cfg.dry_run = dry
        self.status = bot.Status(log_path=os.path.join(BASE, "macro_log.txt"), debug_dir=os.path.join(BASE, "debug"))
        try:
            io = bot.WindowsIO()
            if not io.find_window():
                messagebox.showwarning(APP, "Couldn't find the Roblox window.\nOpen the game first, then press Start.")
                return
        except Exception as e:   # mss / ctypes problems
            messagebox.showerror(APP, f"Couldn't start: {e}")
            return
        self.bot = bot.Bot(cfg, io, self.status)
        self.bot.log(f"{APP} v{VERSION} " + ("test detection mode - nothing will be clicked" if dry else "started"))

        def run():
            try:
                self.bot.run()
            except Exception as e:
                self.status.log.append(f"Error: {e}")
                self.status.phase, self.status.detail = "Error", str(e)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.update_start_button(True)

    def stop(self):
        if self.bot:
            self.bot.stop_event.set()

    def request_toggle(self):
        self.toggle_requested = True      # from the hotkey thread; handled in poll()

    def toggle(self):
        if self.running():
            self.stop()
        else:
            self.start()

    def update_start_button(self, running=None):
        running = self.running() if running is None else running
        k = self.hotkey.upper()
        if running:
            self.b_start.configure(text=f"■  Stop   ({k})")
            self.b_start.set_colors(BAD, "#f87171")
        else:
            self.b_start.configure(text=f"▶  Start   ({k})")
            self.b_start.set_colors(ACCENT, ACCENT_HI)

    # ------------------------------------------------------------- poll
    def poll(self):
        if self.toggle_requested:
            self.toggle_requested = False
            self.toggle()
        st = self.status
        running = self.running()
        phase = st.phase if (running or st.phase in ("Error",)) else "Stopped"
        self.l_phase.config(text=phase)
        self.dot.itemconfigure(self.dot_id, fill=PHASE_COLORS.get(phase, GOOD))
        self.l_detail.config(text=(st.detail or ("Press Start or your hotkey while Roblox is open" if not running else ""))[:80])
        self.l_fps.config(text=f"{st.fps:.0f} fps" if running and st.fps else "")
        up = time.time() - st.session_start if st.session_start else 0
        self.tiles["finished"].config(text=str(st.finished))
        self.tiles["casts"].config(text=str(st.casts))
        self.tiles["rate"].config(text=f"{st.finished / (up / 3600):.0f}" if up > 60 else "-")
        self.tiles["uptime"].config(text=notify.fmt_duration(up) if st.session_start else "-")
        if running and self.bot is not None:
            try:   # show the latency the bot has learned so far
                if int(self.v_latency.get()) != self.bot.cfg.latency_ms:
                    self.v_latency.set(self.bot.cfg.latency_ms)
            except (tk.TclError, ValueError):
                pass
        if not running and self.b_start.base == BAD:
            self.update_start_button(False)
        self.draw_graph(st.graph if running else {})
        if self.log_win is not None and self.log_win.winfo_exists():
            lines = list(st.log)
            if lines != self._log_shown:
                self._log_text.config(state="normal")
                self._log_text.delete("1.0", "end")
                self._log_text.insert("end", "\n".join(lines))
                self._log_text.see("end")
                self._log_text.config(state="disabled")
                self._log_shown = lines
        self.root.after(50, self.poll)

    # ------------------------------------------------------------ graph
    def draw_graph(self, g):
        c = self.graph
        c.delete("all")
        W = int(c.winfo_width()) if c.winfo_width() > 1 else 470
        H = 150
        mode = g.get("mode")
        age = time.perf_counter() - g["now"] if g.get("now") else 0.0
        if not mode:
            for x in range(0, W, 40):
                c.create_line(x, 0, x, H, fill="#262a34")
            c.create_text(W / 2, H / 2, text="Live minigame view" if self.running() else "Not running",
                          fill=MUTED, font=(FONT, 10))
            return
        if mode == "lanes":
            self._draw_lanes(c, g, W, H, age)
        elif mode == "ring":
            self._draw_ring(c, g, W, H, age)
        elif mode == "countin":
            self._draw_countin(c, g, W, H)
        elif mode == "series":
            self._draw_series(c, g, W, H)

    def _draw_lanes(self, c, g, W, H, age):
        lanes = g.get("lanes", [])
        labels = g.get("labels", lanes)
        colors = LANE_COLORS.get(g.get("name"), {})
        n = max(1, len(lanes))
        top, bot_ = 22, H - 8
        rowh = (bot_ - top) / n
        left = 58
        hitx = left + 34
        scale = (W - hitx - 12) / 1.6          # px per second of "time until hit"
        c.create_text(10, 11, text=f"{g.get('name', '')}  -  notes coming in", anchor="w", fill=MUTED, font=(FONT, 8))
        c.create_line(hitx, top - 4, hitx, bot_, fill=ACCENT, width=2)
        for i, lane in enumerate(lanes):
            y = top + rowh * (i + 0.5)
            col = colors.get(lane, TEXT)
            c.create_line(left, y, W - 8, y, fill="#2b303a")
            c.create_text(12, y, text=labels[i] if i < len(labels) else lane, anchor="w", fill=col, font=(FONT, 9, "bold"))
            for t in g.get("notes", {}).get(lane, []):
                x = hitx + (t - age) * scale
                if left <= x <= W - 6:
                    c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=col, outline="")
            for tp, pl in g.get("presses", []):
                if pl != lane:
                    continue
                since = g["now"] + age - tp
                x = hitx - since * scale * 0.6
                if since < 0.25:
                    c.create_oval(hitx - 11, y - 11, hitx + 11, y + 11, outline=col, width=2)
                if left <= x <= hitx:
                    c.create_line(x, y - 7, x, y + 7, fill=col, width=3)

    def _draw_ring(self, c, g, W, H, age):
        cx, cy, r = W / 2, H / 2 + 6, 52
        c.create_text(10, 11, text="Cast circle", anchor="w", fill=MUTED, font=(FONT, 8))
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#2f3440", width=10)
        tgt = g.get("target", 0.0)
        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=-math.degrees(tgt) - 9, extent=18, style="arc",
                     outline=GOOD, width=10)
        needle = g.get("needle", 0.0) + g.get("omega", 0.0) * age
        c.create_line(cx, cy, cx + (r + 6) * math.cos(needle), cy + (r + 6) * math.sin(needle), fill="white", width=3)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="white", outline="")

    def _draw_countin(self, c, g, W, H):
        cx, cy, base = W / 2, H / 2 + 6, 34
        c.create_text(10, 11, text=f"Count-in  -  {g.get('clicks', 0)}/4 rings", anchor="w", fill=MUTED, font=(FONT, 8))
        r0 = g.get("r0") or 1
        c.create_oval(cx - base, cy - base, cx + base, cy + base, outline="white", width=3)
        tr = base * g.get("target", r0) / r0
        c.create_oval(cx - tr, cy - tr, cx + tr, cy + tr, outline=GOOD, dash=(3, 4))
        rr = base * (g.get("ring") or r0) / r0
        if rr > base:
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=ACCENT_HI, width=2)

    def _draw_series(self, c, g, W, H):
        ser = g.get("series") or []
        if len(ser) < 2:
            return
        la, lb = g.get("labels", ("a", "b"))
        t1 = ser[-1][0]
        pts = [p for p in ser if t1 - p[0] <= 5]
        vals = [v for p in pts for v in (p[1], p[2]) if v is not None]
        vals += [p[2] + s * p[3] for p in pts if p[3] is not None for s in (-1, 1)]
        lo, hi = min(vals) - 20, max(vals) + 20
        top, bot_ = 22, H - 8

        def xy(t, v):
            return 10 + (W - 20) * (1 - (t1 - t) / 5), bot_ - (bot_ - top) * (v - lo) / max(1, hi - lo)
        if pts[-1][3] is not None:      # the reel box: draw it as a band
            band = []
            for p in pts:
                band.append(xy(p[0], p[2] + p[3]))
            for p in reversed(pts):
                band.append(xy(p[0], p[2] - p[3]))
            c.create_polygon(*[v for pt in band for v in pt], fill="#2a2440", outline="")
        c.create_line(*[v for p in pts for v in xy(p[0], p[2])], fill=MUTED, width=2)
        c.create_line(*[v for p in pts for v in xy(p[0], p[1])], fill=ACCENT_HI, width=2)
        c.create_text(10, 11, text=f"{g.get('name', '')}  -  {la} vs {lb}", anchor="w", fill=MUTED, font=(FONT, 8))

    # ---------------------------------------------------------- discord
    def test_webhook(self):
        url = self.v_webhook.get().strip()
        if not notify.valid_url(url):
            self.l_webhook.config(text="That isn't a Discord webhook URL", fg=BAD)
            return
        self.l_webhook.config(text="Sending...", fg=MUTED)
        embed = {"title": "Scroom Macro is connected", "description": "Catch alerts and session reports will show up here.",
                 "color": notify.COLOR}
        notify.send_async(url, embed, on_done=lambda ok, msg: self.root.after(
            0, lambda: self.l_webhook.config(text="Sent! Check your channel" if ok else f"Failed: {msg}"[:60],
                                             fg=GOOD if ok else BAD)))

    # ---------------------------------------------------------- updates
    def _check_update(self, manual=False):
        newer, url = updater.check(VERSION)
        self.root.after(0, lambda: self._show_update(newer, url, manual))

    def _show_update(self, newer, url, manual):
        if newer:
            self.update_info = (newer, url)
            self.update_pill.config(text=f"Update v{newer} available  >", bg=BG)
            self.l_update.config(text=f"v{newer} is out - click the green text at the top to download it.", fg=GOOD)
        elif manual:
            self.l_update.config(text=f"You're on the latest version (v{VERSION}).", fg=MUTED)

    def check_update_now(self):
        self.l_update.config(text="Checking...", fg=MUTED)
        threading.Thread(target=self._check_update, args=(True,), daemon=True).start()

    # ------------------------------------------------------------ hotkey
    def apply_hotkey(self):
        if not self.hk.set(self.hotkey):
            messagebox.showwarning(APP, f"Couldn't register {self.hotkey.upper()} as the hotkey - another app may "
                                        f"be using it. Pick a different one under More > Start/Stop key.")

    def hotkey_dialog(self):
        win = tk.Toplevel(self.root, bg=CARD, padx=18, pady=16)
        win.title("Start/Stop key")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        tk.Label(win, text="Click the box, then press the key you want", bg=CARD, fg=TEXT).grid(row=0, column=0, columnspan=2, sticky="w")
        btn = ttk.Button(win, text=self.hotkey.upper(), style="Ghost.TButton", width=18)
        btn.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        new = {"key": self.hotkey, "listening": False}

        def on_key(event):
            if not new["listening"]:       # only listens while this dialog has focus
                return
            name = hotkeys.name_from_keysym(event.keysym)
            if hotkeys.vk_for(name) is None:
                btn.config(text="Unsupported key")
                return
            new["key"], new["listening"] = name, False
            btn.config(text=name.upper())

        def capture():
            new["listening"] = True
            btn.config(text="Press a key...")
            btn.focus_set()

        btn.config(command=capture)
        win.bind("<KeyPress>", on_key)

        def ok():
            self.hotkey = new["key"]
            self.update_start_button()
            self.b_hotkey.config(text=f"Start/Stop key: {self.hotkey.upper()}")
            self.apply_hotkey()
            save_settings(self.cfg, self.hotkey)
            win.destroy()

        ttk.Button(win, text="Save", style="Ghost.TButton", command=ok).grid(row=2, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(win, text="Cancel", style="Ghost.TButton", command=win.destroy).grid(row=2, column=1, sticky="ew")
        tk.Label(win, text="F1-F12, letters, numbers, Insert, Home, End, Page Up/Down, Pause",
                 bg=CARD, fg=MUTED, font=(FONT, 8)).grid(row=3, column=0, columnspan=2, pady=(10, 0))

    # --------------------------------------------------------- bug report
    def bug_report(self):
        """Zip the log, settings and debug screenshots into one file to send."""
        import glob
        import subprocess
        import zipfile
        path = os.path.join(BASE, time.strftime("bug_report_%Y%m%d-%H%M%S.zip"))
        files = [os.path.join(BASE, f) for f in ("macro_log.txt", "settings.json", "startup_error.txt")]
        files += sorted(glob.glob(os.path.join(BASE, "debug", "*.png")))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            messagebox.showinfo(APP, "Nothing to report yet - run the macro until the problem happens, then try again.")
            return
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("version.txt", f"{APP} v{VERSION}\n")
                for f in files:
                    if f.endswith("settings.json"):      # never share the webhook URL
                        try:
                            data = json.load(open(f))
                            data["webhook_url"] = "(removed)" if data.get("webhook_url") else ""
                            z.writestr("settings.json", json.dumps(data, indent=2))
                        except (OSError, ValueError):
                            pass
                        continue
                    z.write(f, os.path.relpath(f, BASE))
        except OSError as e:
            messagebox.showerror(APP, f"Couldn't make the bug report: {e}")
            return
        subprocess.Popen(["explorer", "/select,", path])
        messagebox.showinfo(APP, f"Bug report saved:\n{path}\n\nSend this file to the macro's creator.")

    # --------------------------------------------------------------- log
    def show_log(self):
        if self.log_win is not None and self.log_win.winfo_exists():
            self.log_win.lift()
            return
        self.log_win = tk.Toplevel(self.root, bg=BG)
        self.log_win.title(f"{APP} - log")
        self._log_text = tk.Text(self.log_win, width=80, height=22, font=("Consolas", 9), state="disabled",
                                 bg=CARD, fg=TEXT, insertbackground=TEXT, relief="flat", padx=10, pady=8)
        self._log_text.pack(fill="both", expand=True)
        self._log_shown = None

    def quit(self):
        self.stop()
        self.hk.stop()
        self.read_cfg(quiet=True)
        self.root.destroy()


def main():
    bot.make_dpi_aware()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
