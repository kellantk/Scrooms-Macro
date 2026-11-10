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
VERSION = "2.0.1"
REPO_URL = "https://github.com/kellantk/Scrooms-Macro"
BASE = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
SETTINGS = os.path.join(BASE, "settings.json")

# ---------------------------------------------------------------- theme
BG = "#0f1115"
CARD = "#171a21"
FIELD = "#1f232c"
HOVER = "#2a2f3b"
BORDER = "#262b35"
TEXT = "#eceef3"
MUTED = "#8b93a3"
DIM = "#5d6474"
ACCENT = "#8b5cf6"
ACCENT_HI = "#a78bfa"
GOOD = "#22c55e"
BAD = "#ef4444"
WARN = "#f59e0b"
FONT = "Segoe UI"
FONT_SB = "Segoe UI Semibold"
ICONS = "Segoe MDL2 Assets"          # Windows 10/11 icon font
IC_PIN, IC_PINNED, IC_UP, IC_DOWN = "", "", "", ""
LANE_COLORS = {
    "Strings": {"A": "#4ade80", "S": "#f87171", "D": "#facc15"},
    "Keys": {"A": "#e5e7eb", "S": "#e5e7eb", "D": "#e5e7eb", "F": "#e5e7eb"},
    "Percussion": {"L": "#e5e7eb", "R": "#60a5fa"},
}
PHASE_COLORS = {"Stopped": MUTED, "Paused": WARN, "Error": BAD, "Cooldown": MUTED,
                "Casting": ACCENT_HI, "Waiting for minigame": ACCENT_HI}
UI_KEYS = {"hotkey": "f6", "pin_on_top": False, "compact": False, "tab": "General"}


def work_area():
    """The screen's usable area (without the taskbar): left, top, right, bottom."""
    try:
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(r), 0):   # SPI_GETWORKAREA
            return r.left, r.top, r.right, r.bottom
    except Exception:
        pass
    return 0, 0, 1920, 1040


def load_settings():
    cfg, ui = bot.Config(), dict(UI_KEYS)
    try:
        with open(SETTINGS) as f:
            data = json.load(f)
        for k in UI_KEYS:
            if k in data:
                ui[k] = type(UI_KEYS[k])(data.pop(k))
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
    except (OSError, ValueError, TypeError):
        pass
    return cfg, ui


def save_settings(cfg, ui):
    try:
        data = asdict(cfg)
        data.pop("dry_run", None)
        with open(SETTINGS, "w") as f:
            json.dump({**data, **ui}, f, indent=2)
    except OSError:
        pass


def setup_style(root):
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=CARD, foreground=TEXT, font=(FONT, 10), bordercolor=BORDER,
                lightcolor=BORDER, darkcolor=BORDER, troughcolor=FIELD, focuscolor=ACCENT)
    for w in ("TEntry", "TSpinbox", "TCombobox"):
        s.configure(w, fieldbackground=FIELD, foreground=TEXT, background=FIELD, arrowcolor=MUTED,
                    insertcolor=TEXT, padding=(8, 5), selectbackground=ACCENT, selectforeground="white",
                    bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD)
        s.map(w, fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
              bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)], darkcolor=[("focus", ACCENT)],
              arrowcolor=[("active", TEXT)])
    s.configure("Dark.Vertical.TScrollbar", background=BORDER, troughcolor=CARD, bordercolor=CARD,
                lightcolor=BORDER, darkcolor=BORDER, arrowcolor=MUTED, gripcount=0)
    s.map("Dark.Vertical.TScrollbar", background=[("active", HOVER)])
    s.configure("Ghost.TButton", background=FIELD, foreground=TEXT, borderwidth=0, padding=(10, 6), font=(FONT, 9))
    s.map("Ghost.TButton", background=[("active", HOVER), ("pressed", HOVER)])
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.font", (FONT, 10))


# ---------------------------------------------------------------- widgets
class FlatButton(tk.Label):
    """A big flat button (tk buttons can't be styled cleanly on Windows)."""

    def __init__(self, master, text, command, bg=ACCENT, hover=ACCENT_HI, fg="white", font=(FONT, 12, "bold"),
                 pady=11, **kw):
        super().__init__(master, text=text, bg=bg, fg=fg, font=font, pady=pady, cursor="hand2", **kw)
        self.command, self.base, self.hover = command, bg, hover
        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", lambda e: self.configure(bg=self.hover))
        self.bind("<Leave>", lambda e: self.configure(bg=self.base))

    def set_colors(self, bg, hover):
        self.base, self.hover = bg, hover
        self.configure(bg=bg)


def ghost(master, text, command):
    return FlatButton(master, text, command, bg=FIELD, hover=HOVER, fg=TEXT, font=(FONT, 9), pady=7, padx=10)


class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.win, self.job = widget, text, None, None
        widget.bind("<Enter>", self._later, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Button-1>", self._hide, add="+")

    def _later(self, _):
        self.job = self.widget.after(450, self._show)

    def _show(self):
        if self.win:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.win = tk.Toplevel(self.widget, bg=BORDER, padx=1, pady=1)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        tk.Label(self.win, text=self.text() if callable(self.text) else self.text, bg=FIELD, fg=TEXT,
                 font=(FONT, 8), padx=8, pady=4).pack()
        self.win.update_idletasks()
        self.win.geometry(f"+{x - self.win.winfo_width() // 2}+{y}")

    def _hide(self, _=None):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None
        if self.win:
            self.win.destroy()
            self.win = None


class IconButton(tk.Label):
    """A small square icon button (Windows icon font) that can show an on state."""

    def __init__(self, master, glyph, command, tip, bg=BG):
        super().__init__(master, text=glyph, font=(ICONS, 11), bg=bg, fg=MUTED, width=3, pady=4, cursor="hand2")
        self.base = bg
        self.on = False
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self.configure(bg=FIELD, fg=TEXT if not self.on else ACCENT_HI))
        self.bind("<Leave>", lambda e: self.configure(bg=self.base, fg=ACCENT_HI if self.on else MUTED))
        Tooltip(self, tip)

    def set(self, glyph, on):
        self.on = on
        self.configure(text=glyph, fg=ACCENT_HI if on else MUTED)


class Toggle(tk.Canvas):
    """An on/off switch bound to a BooleanVar."""
    W, H = 38, 22

    def __init__(self, master, var, bg=CARD):
        super().__init__(master, width=self.W, height=self.H, bg=bg, highlightthickness=0, cursor="hand2")
        self.var = var
        self.bind("<Button-1>", lambda e: self.flip())
        var.trace_add("write", lambda *a: self.draw())
        self.draw()

    def flip(self):
        self.var.set(not self.var.get())

    def draw(self):
        self.delete("all")
        on = bool(self.var.get())
        col = ACCENT if on else "#3a404d"
        h = self.H - 2
        self.create_oval(1, 1, 1 + h, 1 + h, fill=col, outline=col)
        self.create_oval(self.W - 1 - h, 1, self.W - 1, 1 + h, fill=col, outline=col)
        self.create_rectangle(1 + h / 2, 1, self.W - 1 - h / 2, 1 + h, fill=col, outline=col)
        x = self.W - 1 - h / 2 if on else 1 + h / 2
        self.create_oval(x - h / 2 + 3, 4, x + h / 2 - 3, self.H - 4, fill="white", outline="")


class Tabs(tk.Frame):
    """Underlined text tabs over one card; all pages share the card's size."""

    def __init__(self, master, on_change=None):
        super().__init__(master, bg=BG)
        self.bar = tk.Frame(self, bg=BG)
        self.bar.pack(fill="x")
        # the pages sit in a canvas so they can scroll when the screen is too
        # short to show everything (small laptops: it ran under the taskbar)
        self.wrap = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        self.wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.wrap, bg=CARD, highlightthickness=0, bd=0)
        self.sb = ttk.Scrollbar(self.wrap, orient="vertical", command=self.canvas.yview, style="Dark.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=CARD)
        self.win = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.body.grid_rowconfigure(0, weight=1)
        self.body.grid_columnconfigure(0, weight=1)
        self.max_h = None
        self.body.bind("<Configure>", lambda e: self._sync())
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.wrap.bind("<Enter>", lambda e: self.wrap.bind_all("<MouseWheel>", self._wheel))
        self.wrap.bind("<Leave>", lambda e: self.wrap.unbind_all("<MouseWheel>"))
        self.pages, self.heads = {}, {}
        self.current = None
        self.on_change = on_change

    def _sync(self):
        w, h = self.body.winfo_reqwidth(), self.body.winfo_reqheight()
        shown = h if self.max_h is None else min(h, self.max_h)
        self.canvas.configure(width=w, height=shown, scrollregion=(0, 0, w, h))
        if shown < h:
            if not self.sb.winfo_ismapped():
                self.sb.pack(side="right", fill="y")
        elif self.sb.winfo_ismapped():
            self.sb.pack_forget()
            self.canvas.yview_moveto(0)

    def set_max_height(self, h):
        self.max_h = h
        self._sync()

    def _wheel(self, e):
        if self.sb.winfo_ismapped():
            self.canvas.yview_scroll(int(-e.delta / 120) * 3, "units")

    def add(self, title):
        page = tk.Frame(self.body, bg=CARD, padx=16, pady=6)
        page.grid(row=0, column=0, sticky="nsew")
        head = tk.Frame(self.bar, bg=BG, cursor="hand2")
        head.pack(side="left", padx=(0, 4))
        lbl = tk.Label(head, text=title, bg=BG, fg=MUTED, font=(FONT_SB, 10), padx=12, pady=6, cursor="hand2")
        lbl.pack()
        line = tk.Frame(head, bg=BG, height=2)
        line.pack(fill="x")
        for w in (head, lbl):
            w.bind("<Button-1>", lambda e, t=title: self.select(t))
        lbl.bind("<Enter>", lambda e, t=title: t != self.current and lbl.configure(fg=TEXT))
        lbl.bind("<Leave>", lambda e, t=title: t != self.current and lbl.configure(fg=MUTED))
        self.pages[title], self.heads[title] = page, (lbl, line)
        if self.current is None:
            self.select(title)
        return page

    def select(self, title):
        if title not in self.pages:
            return
        self.current = title
        for t, (lbl, line) in self.heads.items():
            lbl.configure(fg=TEXT if t == title else MUTED)
            line.configure(bg=ACCENT if t == title else BG)
        self.pages[title].tkraise()
        self.canvas.yview_moveto(0)
        if self.on_change:
            self.on_change(title)


# ------------------------------------------------------------------- app
class App:
    def __init__(self, root):
        self.root = root
        self.cfg, self.ui = load_settings()
        self.hotkey = self.ui["hotkey"]
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

        outer = tk.Frame(root, bg=BG, padx=16, pady=12)
        outer.pack(fill="both", expand=True)

        # ---- header ----------------------------------------------------
        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text=APP, bg=BG, fg=TEXT, font=(FONT, 15, "bold")).pack(side="left")
        tk.Label(head, text=f"v{VERSION}", bg=FIELD, fg=MUTED, font=(FONT, 8, "bold"), padx=7, pady=1).pack(
            side="left", padx=8, pady=(5, 0))
        self.b_pin = IconButton(head, IC_PIN, self.toggle_pin,
                                lambda: "Stop keeping this window on top" if self.ui["pin_on_top"] else "Keep this window on top")
        self.b_pin.pack(side="right")
        self.b_compact = IconButton(head, IC_UP, self.toggle_compact,
                                    lambda: "Show settings" if self.ui["compact"] else "Mini view (hide settings)")
        self.b_compact.pack(side="right", padx=(0, 2))
        self.update_pill = tk.Label(head, text="", bg=BG, fg=GOOD, font=(FONT, 9, "bold"), cursor="hand2")
        self.update_pill.pack(side="right", padx=(0, 8))
        self.update_pill.bind("<Button-1>", lambda e: self.update_info and webbrowser.open(self.update_info[1]))

        # ---- status card -----------------------------------------------
        card = tk.Frame(outer, bg=CARD, padx=14, pady=12, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x", pady=(10, 0))
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x")
        self.dot = tk.Canvas(row, width=12, height=12, bg=CARD, highlightthickness=0)
        self.dot.pack(side="left", pady=(2, 0))
        self.dot_id = self.dot.create_oval(1, 1, 11, 11, fill=MUTED, outline="")
        self.l_phase = tk.Label(row, text="Stopped", bg=CARD, fg=TEXT, font=(FONT, 13, "bold"))
        self.l_phase.pack(side="left", padx=(8, 0))
        self.l_fps = tk.Label(row, text="", bg=CARD, fg=DIM, font=(FONT, 9))
        self.l_fps.pack(side="right")
        self.l_detail = tk.Label(card, text="", bg=CARD, fg=MUTED, font=(FONT, 9), anchor="w")
        self.l_detail.pack(fill="x", pady=(2, 10))

        tiles = tk.Frame(card, bg=CARD)
        tiles.pack(fill="x")
        self.tiles = {}
        for i, (key, label) in enumerate((("finished", "Catches"), ("casts", "Casts"), ("rate", "Per hour"),
                                          ("uptime", "Running"))):
            t = tk.Frame(tiles, bg=FIELD, padx=10, pady=7)
            t.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
            tiles.columnconfigure(i, weight=1, uniform="t")
            v = tk.Label(t, text="0", bg=FIELD, fg=TEXT, font=(FONT, 15, "bold"))
            v.pack(anchor="w")
            tk.Label(t, text=label, bg=FIELD, fg=MUTED, font=(FONT, 8)).pack(anchor="w")
            self.tiles[key] = v

        self.graph = tk.Canvas(card, width=460, height=118, bg=FIELD, highlightthickness=0)
        self.graph.pack(fill="x", pady=(10, 0))

        # ---- start button ----------------------------------------------
        self.b_start = FlatButton(outer, "", self.toggle)
        self.b_start.pack(fill="x", pady=(12, 12))

        # ---- settings --------------------------------------------------
        self.tabs = Tabs(outer, on_change=self._tab_changed)
        self.tabs.pack(fill="both", expand=True)
        self._build_general(self.tabs.add("General"))
        self._build_timing(self.tabs.add("Timing"))
        self._build_discord(self.tabs.add("Discord"))
        self._build_more(self.tabs.add("More"))
        self.tabs.select(self.ui.get("tab", "General"))

        self.update_start_button()
        self.apply_hotkey()
        self.set_pin(self.ui["pin_on_top"])
        self.set_compact(self.ui["compact"])
        self.fit_screen(place=True)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        threading.Thread(target=self._check_update, daemon=True).start()
        self.poll()

    # ------------------------------------------------------- settings ui
    def _setting(self, page, title, control=None, hint=None, toggle=None):
        """One settings row: title (+ hint) on the left, its control on the right.
        control: function(parent) -> widget. toggle: a BooleanVar (makes a switch)."""
        if page.winfo_children():
            tk.Frame(page, bg=BORDER, height=1).pack(fill="x")
        row = tk.Frame(page, bg=CARD)
        row.pack(fill="x", pady=6)
        left = tk.Frame(row, bg=CARD)
        left.pack(side="left", fill="x", expand=True)
        t = tk.Label(left, text=title, bg=CARD, fg=TEXT, font=(FONT_SB, 10), anchor="w")
        t.pack(fill="x")
        self._last_hint = None
        if hint:
            self._last_hint = tk.Label(left, text=hint, bg=CARD, fg=MUTED, font=(FONT, 8), anchor="w", justify="left",
                                       wraplength=320)
            self._last_hint.pack(fill="x")
        if toggle is not None:
            sw = Toggle(row, toggle)
            sw.pack(side="right", padx=(12, 0))
            for w in (row, left, t, *left.winfo_children()):
                w.bind("<Button-1>", lambda e: sw.flip())
                w.configure(cursor="hand2")
        elif control is not None:
            control(row).pack(side="right", padx=(12, 0))
        return row

    def _spin(self, var, lo, hi, step=1, width=6):
        return lambda p: ttk.Spinbox(p, from_=lo, to=hi, increment=step, width=width, textvariable=var)

    def _combo(self, var, values, width=8):
        return lambda p: ttk.Combobox(p, textvariable=var, width=width, state="readonly", values=values)

    def _build_general(self, f):
        self.v_key = tk.StringVar(value=self.cfg.equip_key or "1")
        self.v_equip = tk.BooleanVar(value=self.cfg.equip_on_start)
        self.v_countin = tk.BooleanVar(value=self.cfg.count_in)
        self.v_aim = tk.StringVar(value="Perfect" if self.cfg.aim_perfect else "Middle")
        self.v_repeat_n = tk.IntVar(value=self.cfg.max_catches)
        self.v_cursor = tk.BooleanVar(value=self.cfg.move_cursor)
        self._setting(f, "Instrument hotbar slot", self._combo(self.v_key, [str(i) for i in range(1, 10)] + ["0"], 4))
        self._setting(f, "Equip it when starting", toggle=self.v_equip)
        self._setting(f, "Count-in", hint="Click the head-start rings before each minigame", toggle=self.v_countin)
        self._setting(f, "Cast circle aim", self._combo(self.v_aim, ["Perfect", "Middle"]))
        self._setting(f, "Stop after", self._spin(self.v_repeat_n, 0, 9999),
                      "Catches. 0 = keep going until you stop it")
        self._setting(f, "Move the cursor to a safe spot", toggle=self.v_cursor)
        self.v_skeys = tk.StringVar(value=self.cfg.strings_keys)
        self.v_kkeys = tk.StringVar(value=self.cfg.keys_keys)
        self._setting(f, "Minigame keybinds", lambda p: self._keybind_boxes(p),
                      "Strings / Keys, left to right, if you changed them")

    def _keybind_boxes(self, p):
        box = tk.Frame(p, bg=CARD)
        ttk.Entry(box, textvariable=self.v_skeys, width=5, justify="center").pack(side="left")
        tk.Label(box, text="/", bg=CARD, fg=MUTED).pack(side="left", padx=4)
        ttk.Entry(box, textvariable=self.v_kkeys, width=6, justify="center").pack(side="left")
        return box

    def _build_timing(self, f):
        self.v_latency = tk.IntVar(value=self.cfg.latency_ms)
        self.v_auto = tk.BooleanVar(value=self.cfg.auto_latency)
        self.v_note_off = tk.IntVar(value=self.cfg.note_offset_ms)
        self.v_auto_note = tk.BooleanVar(value=self.cfg.auto_note_timing)
        self.v_cooldown = tk.DoubleVar(value=self.cfg.cooldown_s)
        self._setting(f, "Latency (ms)", self._spin(self.v_latency, 0, 300, 5),
                      "How early to press, to make up for your PC's delay")
        self._setting(f, "Auto-tune latency", hint="Learns it from where cast circle clicks land", toggle=self.v_auto)
        self._setting(f, "Note timing (ms)", self._spin(self.v_note_off, -150, 150, 5),
                      "Strings/Keys: higher = press later")
        self._setting(f, "Auto-tune note timing", hint="Learns it from when the strings circles light up",
                      toggle=self.v_auto_note)
        self._setting(f, "Wait between catches (s)", self._spin(self.v_cooldown, 0, 10, 0.5))

    def _build_discord(self, f):
        self.v_webhook = tk.StringVar(value=self.cfg.webhook_url)
        self.v_uid = tk.StringVar(value=self.cfg.discord_user_id)
        self.v_ping = tk.BooleanVar(value=self.cfg.ping_on_stop)
        self.v_notify = tk.BooleanVar(value=self.cfg.notify_catches)
        self.v_report = tk.IntVar(value=self.cfg.report_minutes)
        row = self._setting(f, "Webhook URL", lambda p: ghost(p, "Send test", self.test_webhook),
                            self.WEBHOOK_HINT)
        self.l_webhook = self._last_hint          # the hint line also shows the test result
        ttk.Entry(row.master, textvariable=self.v_webhook, show="•").pack(fill="x", pady=(0, 8))
        self._setting(f, "Your Discord user ID", lambda p: ttk.Entry(p, textvariable=self.v_uid, width=21),
                      "Developer Mode on > right-click your name > Copy User ID")
        self._setting(f, "Ping me if it stops or gets stuck", toggle=self.v_ping,
                      hint=f"Roblox closed, errors, goal reached, no catch in {bot.STUCK_MINUTES} min")
        self._setting(f, "Post every catch", hint="With a screenshot of the result", toggle=self.v_notify)
        self._setting(f, "Session report every (min)", self._spin(self.v_report, 0, 240), "0 = off")

    WEBHOOK_HINT = "Channel settings > Integrations > Webhooks > Copy URL"

    def _build_more(self, f):
        grid = tk.Frame(f, bg=CARD)
        grid.pack(fill="x", pady=(6, 0))
        self.b_hotkey = ghost(grid, f"Start/Stop key: {self.hotkey.upper()}", self.hotkey_dialog)
        btns = [self.b_hotkey,
                ghost(grid, "Test detection (no clicks)", lambda: self.start(dry=True)),
                ghost(grid, "Show log", self.show_log),
                ghost(grid, "Make bug report", self.bug_report),
                ghost(grid, "Check for updates", self.check_update_now),
                ghost(grid, "Source code on GitHub", lambda: webbrowser.open(REPO_URL))]
        for i, b in enumerate(btns):
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0 if i % 2 == 0 else 6, 0), pady=3)
        grid.columnconfigure(0, weight=1, uniform="m")
        grid.columnconfigure(1, weight=1, uniform="m")
        self.l_update = tk.Label(f, text="", bg=CARD, fg=MUTED, font=(FONT, 9), anchor="w")
        self.l_update.pack(fill="x", pady=(8, 0))
        tk.Label(f, text="Tip: the pin (top right) keeps this window on top, the arrow shrinks it.",
                 bg=CARD, fg=DIM, font=(FONT, 8), anchor="w").pack(fill="x", pady=(2, 6))

    def _tab_changed(self, title):
        self.ui["tab"] = title

    # ------------------------------------------------------- pin / mini
    def set_pin(self, on):
        self.ui["pin_on_top"] = bool(on)
        self.root.attributes("-topmost", bool(on))
        self.b_pin.set(IC_PINNED if on else IC_PIN, bool(on))

    def toggle_pin(self):
        self.set_pin(not self.ui["pin_on_top"])
        self.read_cfg(quiet=True)

    def set_compact(self, on):
        self.ui["compact"] = bool(on)
        if on:
            self.graph.pack_forget()
            self.tabs.pack_forget()
            self.b_start.pack_configure(pady=(12, 0))
        else:
            self.graph.pack(fill="x", pady=(10, 0))
            self.b_start.pack_configure(pady=(12, 12))
            self.tabs.pack(fill="both", expand=True)
        self.b_compact.set(IC_DOWN if on else IC_UP, False)
        if hasattr(self, "graph") and self.root.winfo_ismapped():
            self.fit_screen(place=True)

    def fit_screen(self, place=False):
        """Keep the whole window above the taskbar: on short screens the
        settings get a scrollbar instead of running off the bottom."""
        left, top, right, bottom = work_area()
        root = self.root
        self.tabs.set_max_height(None)
        self.graph.configure(height=118)
        root.update_idletasks()
        avail = bottom - top - 45                  # the title bar + a little room
        over = root.winfo_reqheight() - avail
        if over > 0 and not self.ui["compact"]:
            # settings scroll first, then the live view gets shorter
            body = self.tabs.body.winfo_reqheight()
            cut = min(over, body - 140)
            self.tabs.set_max_height(body - max(cut, 0))
            over -= max(cut, 0)
            if over > 0:
                self.graph.configure(height=max(60, 118 - over))
            root.update_idletasks()
        if place:
            h = root.winfo_reqheight() + 40
            y = root.winfo_y() if root.winfo_ismapped() else (top + max(0, (bottom - top - h) // 3))
            if y + h > bottom or y < top:
                y = max(top, bottom - h)
            x = root.winfo_x() if root.winfo_ismapped() else max(left, (right - left - root.winfo_reqwidth()) // 2)
            root.geometry(f"+{x}+{y}")

    def toggle_compact(self):
        self.set_compact(not self.ui["compact"])
        self.read_cfg(quiet=True)

    # ------------------------------------------------------------ config
    def read_cfg(self, quiet=False):
        c = self.cfg
        try:
            c.latency_ms = int(self.v_latency.get())
            c.cooldown_s = float(self.v_cooldown.get())
            c.note_offset_ms = int(self.v_note_off.get())
            c.max_catches = max(0, int(self.v_repeat_n.get()))
            c.report_minutes = max(0, int(self.v_report.get()))
        except (tk.TclError, ValueError):
            if not quiet:
                messagebox.showerror(APP, "Please enter numbers in the number boxes.")
            return None
        c.auto_latency = self.v_auto.get()
        c.auto_note_timing = self.v_auto_note.get()
        c.equip_key = self.v_key.get()
        c.equip_on_start = self.v_equip.get()
        c.count_in = self.v_countin.get()
        c.aim_perfect = self.v_aim.get() == "Perfect"
        c.move_cursor = self.v_cursor.get()
        c.strings_keys = self.v_skeys.get().replace(" ", "").upper()[:3] or "ASD"
        c.keys_keys = self.v_kkeys.get().replace(" ", "").upper()[:4] or "ASDF"
        c.webhook_url = self.v_webhook.get().strip()
        c.discord_user_id = self.v_uid.get().strip()
        c.ping_on_stop = self.v_ping.get()
        c.notify_catches = self.v_notify.get()
        self.ui["hotkey"] = self.hotkey
        save_settings(c, self.ui)
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
        elif cfg.webhook_url and cfg.discord_user_id and not notify.valid_user_id(cfg.discord_user_id):
            messagebox.showwarning(APP, "Your Discord user ID doesn't look right (it's a long number) - "
                                        "pings are off until it's fixed.")
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
        idle = f"Open Roblox, then press Start or {self.hotkey.upper()}"
        self.l_detail.config(text=(st.detail or (idle if not running else ""))[:80],
                             fg=WARN if phase == "Stopped" and st.detail else MUTED)
        self.l_fps.config(text=f"{st.fps:.0f} fps" if running and st.fps else "")
        up = time.time() - st.session_start if st.session_start else 0
        self.tiles["finished"].config(text=str(st.finished))
        self.tiles["casts"].config(text=str(st.casts))
        self.tiles["rate"].config(text=f"{st.finished / (up / 3600):.0f}" if up > 60 else "-")
        self.tiles["uptime"].config(text=notify.fmt_duration(up) if st.session_start else "-")
        if running and self.bot is not None:
            try:   # show the timing the bot has learned so far
                if int(self.v_latency.get()) != self.bot.cfg.latency_ms:
                    self.v_latency.set(self.bot.cfg.latency_ms)
                if int(self.v_note_off.get()) != self.bot.cfg.note_offset_ms:
                    self.v_note_off.set(self.bot.cfg.note_offset_ms)
            except (tk.TclError, ValueError):
                pass
        if not running and self.b_start.base == BAD:
            self.update_start_button(False)
        if not self.ui["compact"]:
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
        W = int(c.winfo_width()) if c.winfo_width() > 1 else 460
        H = int(c.winfo_height()) if c.winfo_height() > 1 else 140
        mode = g.get("mode")
        age = time.perf_counter() - g["now"] if g.get("now") else 0.0
        if not mode:
            for x in range(20, W, 40):
                for y in range(20, H, 40):
                    c.create_oval(x - 1, y - 1, x + 1, y + 1, fill="#2b303b", outline="")
            c.create_text(W / 2, H / 2, text="Live minigame view" if self.running() else "Not running",
                          fill=DIM, font=(FONT, 10))
            return
        if mode == "lanes":
            self._draw_lanes(c, g, W, H, age)
        elif mode == "ring":
            self._draw_ring(c, g, W, H, age)
        elif mode == "countin":
            self._draw_countin(c, g, W, H)
        elif mode == "series":
            self._draw_series(c, g, W, H)
        elif mode == "staff":
            self._draw_staff(c, g, W, H, age)

    def _draw_staff(self, c, g, W, H, age):
        top, bot_ = 24, H - 8
        gap = (bot_ - top) / 5
        ys = [top + gap * (i + 0.5) for i in range(5)]
        left, hitx = 30, 64
        scale = (W - hitx - 10) / 2.0          # px per second of "time until it arrives"
        c.create_text(10, 11, text="Wind  -  notes coming in", anchor="w", fill=MUTED, font=(FONT, 8))
        for y in ys:
            c.create_line(left, y, W - 8, y, fill="#2b303a")
        c.create_line(hitx, top - 4, hitx, bot_, fill=ACCENT, width=2)
        for ln, on, off in g.get("notes", []):
            if not 0 <= ln < 5:
                continue
            xa, xb = max(left, hitx + (on - age) * scale), min(W - 8, hitx + (off - age) * scale)
            if xb > xa:
                c.create_line(xa, ys[ln], xb, ys[ln], fill="#a7f3d0", width=5, capstyle="round")
        tgt = g.get("target")
        if tgt is not None and 0 <= tgt < 5:
            c.create_text(14, ys[tgt], text="▸", fill=ACCENT_HI, font=(FONT, 9))
        s = g.get("scroom")
        if s is not None:
            y = ys[0] + (s - 0.142) / 0.179 * gap
            d = 8
            c.create_polygon(hitx, y - d, hitx + d, y, hitx, y + d, hitx - d, y,
                             fill=ACCENT if g.get("hold") else FIELD, outline="white", width=2)

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
        cx, cy, r = W / 2, H / 2 + 6, min(52, H / 2 - 16)
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
        uid = self.v_uid.get().strip()
        if not notify.valid_url(url):
            self.l_webhook.config(text="That isn't a Discord webhook URL", fg=BAD)
            return
        if uid and not notify.valid_user_id(uid):
            self.l_webhook.config(text="The user ID should be a long number", fg=BAD)
            return
        ping = uid if uid and self.v_ping.get() else None
        self.l_webhook.config(text="Sending...", fg=MUTED)
        embed = {"title": "Scroom Macro is connected",
                 "description": "Catch alerts and session reports will show up here."
                                + (" You'll get pinged like this if the macro stops or gets stuck." if ping else ""),
                 "color": notify.COLOR}
        notify.send_async(url, embed, ping=ping, on_done=lambda ok, msg: self.root.after(
            0, lambda: self.l_webhook.config(text=("Sent! Check your channel" + (" (and your ping)" if ping else ""))
                                             if ok else f"Failed: {msg}"[:60], fg=GOOD if ok else BAD)))

    # ---------------------------------------------------------- updates
    def _check_update(self, manual=False):
        newer, url = updater.check(VERSION)
        self.root.after(0, lambda: self._show_update(newer, url, manual))

    def _show_update(self, newer, url, manual):
        if newer:
            self.update_info = (newer, url)
            self.update_pill.config(text=f"Update v{newer}  →")
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
            self.read_cfg(quiet=True)
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
                    if f.endswith("settings.json"):      # never share the webhook URL or your Discord ID
                        try:
                            data = json.load(open(f))
                            for k in ("webhook_url", "discord_user_id"):
                                data[k] = "(removed)" if data.get(k) else ""
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
        self.log_win.transient(self.root)
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
