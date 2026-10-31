# Scroom Macro

A Windows app that plays the scroom-catching minigame in a Fisch-style Roblox game for you:
it casts, clicks the ring, holds the frequency pin on the dot, and keeps the scroom inside
the box during the reel.

It works by **looking at the screen** (screenshots of the Roblox window) and **sending normal
mouse clicks**. It does not read or change the game's memory, inject anything into Roblox, or
use any exploit. All the code is here so you can check exactly what it does.

> Using macros can be against Roblox's or a game's rules. Use it at your own risk.

## Download

Get the latest `ScroomMacro_vX.X.X.zip` from the **Releases** page, then:

1. Right-click the zip → **Extract All…** → **Extract** (don't run it from inside the zip).
2. Open the extracted folder → **App** → double-click **ScroomMacro.exe**.
3. If Windows says *"Windows protected your PC"*: **More info** → **Run anyway**
   (the app isn't code-signed).
4. In Roblox, stand where you fish and press **F6** to start / stop.

Problems? Run it until the problem happens, click **Make bug report** in the app, and send
the zip it creates.

## How it works

| File | What it does |
|---|---|
| `ScroomMacro.pyw` | The app window (OP Auto Clicker style), settings, hotkey, bug reports |
| `bot.py` | The macro: cast → ring → frequency → reel, timing/physics, failsafes, Windows input & screen capture |
| `detect.py` | Finds each minigame on screen at any window size and reads positions from it |
| `hotkeys.py` | The start/stop hotkey (Windows `RegisterHotKey` - no keyboard hook) |
| `startup_check.py` | Checks the app's files on startup and shows a clear error if something's missing |
| `make_version.py`, `build_exe.bat` | Build the `.exe` and release zip |

**Each minigame:**
- **Ring:** finds the ring by its green zone, measures the needle's speed and direction, and
  clicks on the calculated position. After each click it reads where the needle froze and
  learns your PC's delay (*auto-tune latency*).
- **Frequency:** holding speeds the pin up to the right, releasing to the left. A
  bang-bang controller using the pin's measured acceleration brakes it onto the dot.
- **Reel:** your clicks move the `< o >` dot, the scroom chases the dot, and the box moves on
  its own. The bot steers the dot so the scroom stays centered in the box.

Detection is based on contrast and shapes, not fixed colors or positions, so it handles
different window sizes, rarer (purple) scrooms, dark areas and snow.

What it touches on your PC: screenshots of the Roblox window only, mouse clicks, your
instrument's hotbar key, and one registered hotkey. Settings, a log and debug screenshots are
saved next to the exe.

## Run / build from source

Needs Windows and Python 3.12.

```bash
pip install numpy mss pyinstaller
python ScroomMacro.pyw
```

To build the app folder and a release zip: run `build_exe.bat` (update `VERSION` in
`ScroomMacro.pyw` and `PATCH_NOTES.txt` first).

## License

[MIT](LICENSE) © 2026 Kellan
