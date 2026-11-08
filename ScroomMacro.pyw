"""Scroom Macro - OP Auto Clicker style front end for bot.py."""
import json
import os
import sys
import threading
import tkinter as tk
from dataclasses import asdict
from tkinter import messagebox, ttk

import startup_check

startup_check.run()   # before numpy loads: safe settings + clear errors

import bot  # noqa: E402
import hotkeys  # noqa: E402

APP = "Scroom Macro"
VERSION = "1.9.3"
BASE = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
SETTINGS = os.path.join(BASE, "settings.json")


def load_settings():
    cfg, hotkey = bot.Config(), "f6"
    try:
        with open(SETTINGS) as f:
            data = json.load(f)
        hotkey = data.pop("hotkey", hotkey)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
    except (OSError, ValueError):
        pass
    return cfg, hotkey


def save_settings(cfg, hotkey):
    try:
        with open(SETTINGS, "w") as f:
            json.dump({**asdict(cfg), "hotkey": hotkey}, f, indent=2)
    except OSError:
        pass


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

        root.title(f"{APP} {VERSION}")
        root.resizable(False, False)
        try:
            root.iconbitmap(os.path.join(getattr(sys, "_MEIPASS", BASE), "scroom.ico"))
        except tk.TclError:
            pass
        style = ttk.Style()
        for theme in ("vista", "winnative", "clam"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        root.option_add("*Font", ("Segoe UI", 9))

        pad = {"padx": 6, "pady": 4}
        outer = ttk.Frame(root, padding=8)
        outer.grid(sticky="nsew")

        # ---- Timing ------------------------------------------------------
        timing = ttk.LabelFrame(outer, text="Timing", padding=(8, 4))
        timing.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)
        self.v_latency = tk.IntVar(value=self.cfg.latency_ms)
        self.v_cooldown = tk.DoubleVar(value=self.cfg.cooldown_s)
        ttk.Spinbox(timing, from_=0, to=300, increment=5, width=6,
                    textvariable=self.v_latency).grid(row=0, column=0)
        ttk.Label(timing, text="latency (ms)").grid(row=0, column=1, sticky="w", padx=(4, 18))
        ttk.Spinbox(timing, from_=0, to=10, increment=0.5, width=6,
                    textvariable=self.v_cooldown).grid(row=0, column=2)
        ttk.Label(timing, text="secs between catches").grid(row=0, column=3, sticky="w", padx=4)
        self.v_auto = tk.BooleanVar(value=self.cfg.auto_latency)
        ttk.Checkbutton(timing, text="Auto-tune latency from where ring clicks land",
                        variable=self.v_auto).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ---- Cast options / Catch repeat --------------------------------
        cast = ttk.LabelFrame(outer, text="Cast options", padding=(8, 4))
        cast.grid(row=1, column=0, sticky="nsew", **pad)
        ttk.Label(cast, text="Guitar slot:").grid(row=0, column=0, sticky="w")
        self.v_key = tk.StringVar(value=self.cfg.equip_key or "1")
        ttk.Combobox(cast, textvariable=self.v_key, width=4, state="readonly",
                     values=[str(i) for i in range(1, 10)] + ["0"]).grid(row=0, column=1, sticky="w", pady=2)
        self.v_equip = tk.BooleanVar(value=self.cfg.equip_on_start)
        ttk.Checkbutton(cast, text="Equip on start", variable=self.v_equip).grid(row=1, column=0, columnspan=2,
                                                                               sticky="w")
        ttk.Label(cast, text="Ring aim:").grid(row=2, column=0, sticky="w")
        self.v_aim = tk.StringVar(value="Perfect" if self.cfg.aim_perfect else "Middle")
        ttk.Combobox(cast, textvariable=self.v_aim, width=8, state="readonly",
                     values=["Perfect", "Middle"]).grid(row=2, column=1, sticky="w", pady=2)

        repeat = ttk.LabelFrame(outer, text="Catch repeat", padding=(8, 4))
        repeat.grid(row=1, column=1, sticky="nsew", **pad)
        self.v_repeat_mode = tk.StringVar(value="count" if self.cfg.max_catches else "forever")
        self.v_repeat_n = tk.IntVar(value=self.cfg.max_catches or 10)
        r1 = ttk.Frame(repeat)
        r1.grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(r1, text="Repeat", value="count", variable=self.v_repeat_mode).pack(side="left")
        ttk.Spinbox(r1, from_=1, to=9999, width=5, textvariable=self.v_repeat_n).pack(side="left", padx=4)
        ttk.Label(r1, text="catches").pack(side="left")
        ttk.Radiobutton(repeat, text="Repeat until stopped", value="forever",
                        variable=self.v_repeat_mode).grid(row=1, column=0, sticky="w", pady=(6, 0))

        # ---- Cursor position --------------------------------------------
        cursor = ttk.LabelFrame(outer, text="Cursor position", padding=(8, 4))
        cursor.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)
        self.v_cursor = tk.StringVar(value="safe" if self.cfg.move_cursor else "current")
        ttk.Radiobutton(cursor, text="Move to safe spot", value="safe",
                        variable=self.v_cursor).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(cursor, text="Current location", value="current",
                        variable=self.v_cursor).grid(row=0, column=1, sticky="w", padx=24)

        # ---- Status -----------------------------------------------------
        stat = ttk.LabelFrame(outer, text="Status", padding=(8, 4))
        stat.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)
        self.l_phase = ttk.Label(stat, text="Stopped", font=("Segoe UI", 10, "bold"), width=40)
        self.l_phase.grid(row=0, column=0, sticky="w")
        self.l_detail = ttk.Label(stat, text="", foreground="#555", width=48)
        self.l_detail.grid(row=1, column=0, sticky="w")
        self.l_count = ttk.Label(stat, text="Catches: 0    Casts: 0")
        self.l_count.grid(row=2, column=0, sticky="w")

        # ---- Buttons ----------------------------------------------------
        btns = ttk.Frame(outer)
        btns.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))
        btns.columnconfigure((0, 1), weight=1, uniform="b")
        self.b_start = ttk.Button(btns, command=self.start)
        self.b_stop = ttk.Button(btns, command=self.stop, state="disabled")
        self.b_start.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2, ipady=4)
        self.b_stop.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2, ipady=4)
        ttk.Button(btns, text="Hotkey setting", command=self.hotkey_dialog).grid(
            row=1, column=0, sticky="ew", padx=(0, 3), pady=2, ipady=4)
        ttk.Button(btns, text="Test detection", command=lambda: self.start(dry=True)).grid(
            row=1, column=1, sticky="ew", padx=(3, 0), pady=2, ipady=4)
        log_link = ttk.Label(outer, text="Show log", foreground="#0066cc", cursor="hand2")
        log_link.grid(row=5, column=1, sticky="e", padx=6)
        log_link.bind("<Button-1>", lambda e: self.show_log())
        bug_link = ttk.Label(outer, text="Make bug report", foreground="#0066cc", cursor="hand2")
        bug_link.grid(row=5, column=0, sticky="w", padx=6)
        bug_link.bind("<Button-1>", lambda e: self.bug_report())

        self.update_hotkey_labels()
        self.apply_hotkey()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.poll()

    # ------------------------------------------------------------ config
    def read_cfg(self):
        c = self.cfg
        try:
            c.latency_ms = int(self.v_latency.get())
            c.cooldown_s = float(self.v_cooldown.get())
            c.max_catches = int(self.v_repeat_n.get()) if self.v_repeat_mode.get() == "count" else 0
        except (tk.TclError, ValueError):
            messagebox.showerror(APP, "Please enter numbers in the Timing / Repeat boxes.")
            return None
        c.auto_latency = self.v_auto.get()
        c.equip_key = self.v_key.get()
        c.equip_on_start = self.v_equip.get()
        c.aim_perfect = self.v_aim.get() == "Perfect"
        c.move_cursor = self.v_cursor.get() == "safe"
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
        cfg.dry_run = dry
        self.status = bot.Status(log_path=os.path.join(BASE, "macro_log.txt"),
                                 debug_dir=os.path.join(BASE, "debug"))
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
        self.b_start.state(["disabled"])
        self.b_stop.state(["!disabled"])

    def stop(self):
        if self.bot:
            self.bot.stop_event.set()

    def request_toggle(self):
        # called from the hotkey thread; the Tk loop picks it up in poll()
        self.toggle_requested = True

    def toggle(self):
        if self.running():
            self.stop()
        else:
            self.start()

    def poll(self):
        if self.toggle_requested:
            self.toggle_requested = False
            self.toggle()
        st = self.status
        self.l_phase.config(text=st.phase)
        self.l_detail.config(text=st.detail[:60])
        fps = f"    {st.fps:.0f} fps" if self.running() and st.fps else ""
        self.l_count.config(text=f"Catches: {st.catches}    Got away: {st.escapes}    Casts: {st.casts}{fps}")
        if self.running() and self.bot is not None:
            try:   # show the latency the bot has learned so far
                if int(self.v_latency.get()) != self.bot.cfg.latency_ms:
                    self.v_latency.set(self.bot.cfg.latency_ms)
            except (tk.TclError, ValueError):
                pass
        if not self.running():
            self.b_start.state(["!disabled"])
            self.b_stop.state(["disabled"])
        if self.log_win is not None and self.log_win.winfo_exists():
            lines = list(st.log)
            if lines != self._log_shown:
                self._log_text.config(state="normal")
                self._log_text.delete("1.0", "end")
                self._log_text.insert("end", "\n".join(lines))
                self._log_text.see("end")
                self._log_text.config(state="disabled")
                self._log_shown = lines
        self.root.after(100, self.poll)

    # ------------------------------------------------------------ hotkey
    def update_hotkey_labels(self):
        k = self.hotkey.upper()
        self.b_start.config(text=f"Start ({k})")
        self.b_stop.config(text=f"Stop ({k})")

    def apply_hotkey(self):
        """(Re)register the start/stop hotkey with Windows."""
        if not self.hk.set(self.hotkey):
            messagebox.showwarning(APP, f"Couldn't register {self.hotkey.upper()} as the hotkey - another app may "
                                        f"be using it. Pick a different one under Hotkey setting.")

    def hotkey_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Hotkey setting")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        frm = ttk.Frame(win, padding=12)
        frm.pack()
        ttk.Label(frm, text="Start / Stop").grid(row=0, column=0, padx=(0, 8))
        btn = ttk.Button(frm, text=self.hotkey.upper(), width=14)
        btn.grid(row=0, column=1)
        new = {"key": self.hotkey, "listening": False}

        def on_key(event):
            # only listens while this dialog has focus - nothing system-wide
            if not new["listening"]:
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
            self.update_hotkey_labels()
            self.apply_hotkey()
            save_settings(self.cfg, self.hotkey)
            win.destroy()

        bar = ttk.Frame(frm)
        bar.grid(row=1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(bar, text="Ok", command=ok).pack(side="left", padx=4)
        ttk.Button(bar, text="Cancel", command=win.destroy).pack(side="left", padx=4)
        ttk.Label(frm, text="F1-F12, letters, numbers, Insert, Home, End, Page Up/Down, Pause",
                  foreground="#666").grid(row=2, column=0, columnspan=2, pady=(8, 0))
    # --------------------------------------------------------- bug report
    def bug_report(self):
        """Zip the log, settings and debug screenshots into one file to send."""
        import glob
        import time
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
                    z.write(f, os.path.relpath(f, BASE))
        except OSError as e:
            messagebox.showerror(APP, f"Couldn't make the bug report: {e}")
            return
        import subprocess
        subprocess.Popen(["explorer", "/select,", path])
        messagebox.showinfo(APP, f"Bug report saved:\n{path}\n\nSend this file to the macro's creator.")

    # --------------------------------------------------------------- log
    def show_log(self):
        if self.log_win is not None and self.log_win.winfo_exists():
            self.log_win.lift()
            return
        self.log_win = tk.Toplevel(self.root)
        self.log_win.title(f"{APP} - log")
        self._log_text = tk.Text(self.log_win, width=70, height=20, font=("Consolas", 9), state="disabled")
        self._log_text.pack(fill="both", expand=True)
        self._log_shown = None

    def quit(self):
        self.stop()
        self.hk.stop()
        self.read_cfg()
        self.root.destroy()


def main():
    bot.make_dpi_aware()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
