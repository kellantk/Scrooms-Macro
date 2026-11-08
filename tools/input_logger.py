"""Records mouse clicks, mouse MOVEMENT and key presses with UTC timestamps,
for lining up with Roblox recordings (their file name is the UTC end time).

Writes inputs_<start time>.csv next to this script. Ctrl+C (or close the
window) to stop. Mouse moves are recorded at most ~100 times a second.
"""
import csv
import os
import time
from datetime import datetime, timezone

from pynput import keyboard, mouse

here = os.path.dirname(os.path.abspath(__file__))
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H_%M_%SZ")
path = os.path.join(here, f"inputs_{stamp}.csv")

out = open(path, "w", newline="")
w = csv.writer(out)
w.writerow(["wall_clock_utc", "t_seconds", "device", "event", "detail", "x", "y"])
out.flush()
t0 = time.perf_counter()
last_move = [0.0]


def log(device, event, detail="", x="", y=""):
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    w.writerow([now, f"{time.perf_counter() - t0:.3f}", device, event, detail, x, y])


def on_move(x, y):
    t = time.perf_counter()
    if t - last_move[0] >= 0.01:
        last_move[0] = t
        log("mouse", "move", "", x, y)


def on_click(x, y, button, pressed):
    log("mouse", "down" if pressed else "up", button.name, x, y)
    out.flush()


def on_scroll(x, y, dx, dy):
    log("mouse", "scroll", f"{dy:+d}", x, y)


def key_name(k):
    return getattr(k, "char", None) or str(k).replace("Key.", "")


held = set()


def on_press(k):
    name = key_name(k)
    if name in held:
        return
    held.add(name)
    log("key", "down", name)
    out.flush()


def on_release(k):
    name = key_name(k)
    held.discard(name)
    log("key", "up", name)


mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll).start()
keyboard.Listener(on_press=on_press, on_release=on_release).start()
print(f"Logging inputs (incl. mouse movement) to:\n  {path}\n")
print("Record in Roblox now. Press Ctrl+C here (or close this window) when done.")
try:
    while True:
        time.sleep(0.5)
        out.flush()
except KeyboardInterrupt:
    pass
finally:
    out.close()
    print(f"\nSaved {path}")
