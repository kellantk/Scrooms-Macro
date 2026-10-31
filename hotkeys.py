"""Start/stop hotkey via Windows' RegisterHotKey.

Windows only tells us about the one key we registered - no system-wide
keyboard hook (that's what keyloggers use, and why antivirus flags apps
that do it).
"""
import ctypes
import threading
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_NOREPEAT = 0x4000

_SPECIAL = {
    "space": 0x20, "page_up": 0x21, "page_down": 0x22, "end": 0x23, "home": 0x24,
    "insert": 0x2D, "delete": 0x2E, "pause": 0x13, "scroll_lock": 0x91,
}


def vk_for(name):
    """Virtual-key code for a hotkey name like 'f6', 'g', '7', 'insert'."""
    name = (name or "").lower()
    if name.startswith("f") and name[1:].isdigit() and 1 <= int(name[1:]) <= 24:
        return 0x70 + int(name[1:]) - 1
    if len(name) == 1 and (name.isdigit() or "a" <= name <= "z"):
        return ord(name.upper())
    return _SPECIAL.get(name)


def name_from_keysym(keysym):
    """Tk keysym -> our hotkey name."""
    k = keysym.lower()
    return {"prior": "page_up", "next": "page_down"}.get(k, k)


class GlobalHotkey:
    def __init__(self, on_press):
        self.on_press = on_press
        self.thread = None
        self.thread_id = None

    def set(self, name):
        """Register `name` as the hotkey (replacing any previous one).
        Returns False if Windows refused (e.g. another app owns that key)."""
        self.stop()
        vk = vk_for(name)
        if vk is None:
            return False
        ready = threading.Event()
        result = {"ok": False}

        def run():
            self.thread_id = kernel32.GetCurrentThreadId()
            result["ok"] = bool(user32.RegisterHotKey(None, 1, MOD_NOREPEAT, vk))
            ready.set()
            if not result["ok"]:
                return
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    self.on_press()
            user32.UnregisterHotKey(None, 1)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        ready.wait(2)
        return result["ok"]

    def stop(self):
        if self.thread and self.thread.is_alive() and self.thread_id:
            user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
            self.thread.join(1)
        self.thread = None
        self.thread_id = None
