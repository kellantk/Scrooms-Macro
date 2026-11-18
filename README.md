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
2. Open the extracted folder → double-click **ScroomMacro.exe** (keep the `_internal` folder next to it).
3. If Windows says *"Windows protected your PC"*: **More info** → **Run anyway**
   (the app isn't code-signed).
4. In Roblox, stand where you fish and press **F6** to start / stop.

Problems? Run it until the problem happens, click **Make bug report** in the app, and send
the zip it creates.

## How it works

| File | What it does |
|---|---|
| `ScroomMacro.pyw` | The app window: status, live view, settings, hotkey, pin-on-top / mini view, bug reports |
| `bot.py` | The macro: cast → ring → frequency → reel, timing/physics, failsafes, Windows input & screen capture |
| `detect.py` | Finds each minigame on screen at any window size and reads positions from it |
| `hotkeys.py` | The start/stop hotkey (Windows `RegisterHotKey` - no keyboard hook) |
| `startup_check.py` | Checks the app's files on startup and shows a clear error if something's missing |
| `make_version.py`, `build_exe.bat` | Build the `.exe` and release zip |
| `sim/` | Practice range: simulated minigames to test the macro on (not part of the app) |

**Supported instruments (Crescendo Valley update):** Strings, Keys, Percussion, Wind.

| File | Minigame |
|---|---|
| `strings.py` | Strings: 3-lane note highway, keys A/S/D |
| `keys.py` | Keys: Piano-Tiles board, keys A/S/D/F |
| `percussion.py` | Percussion: one lane, left/right/both clicks |
| `wind.py` | Wind: 5-line staff - steer the scroom with the mouse, hold left click through each note |
| `countin.py` | The count-in: clicks each head-start ring as it reaches the circle |
| `notify.py` | Discord webhook: catch alerts with a screenshot, session reports |
| `updater.py` | Checks this repo's Releases for a newer version (only opens the page - never installs anything) |

**Each minigame:**
- **Ring:** finds the ring by its green zone, measures the needle's speed and direction, and
  clicks on the calculated position. After each click it reads where the needle froze and
  learns your PC's delay (*auto-tune latency*).
- **Count-in:** tracks each shrinking ring and clicks when it lies on the circle (the dot under
  the circle turns yellow = Perfect).
- **Wind:** turns every note on the staff into a time window from its bar's ends and the measured
  scroll speed, moves the mouse so the scroom sits on the bar (following it when it slides to
  another line, and learning how the scroom follows the cursor), and holds left click while the
  bar passes.
- **Strings / Keys / Percussion:** every note of one minigame moves the same way, so a note's
  arrival is worked out from all of its readings together and the key or click is sent at that
  exact moment. Keys tiles are hit on the lettered key row; hold notes are released when the
  bar's end arrives.
- **Strings auto-tune:** it times when the circle lights up against when the note crossed it,
  and nudges the timing toward the middle of Perfect (*auto-tune note timing*).
- **Where "Perfect" is** was measured on recordings, from the game's own Perfect / Good / Miss
  labels; the log reports them after each strings or keys minigame.
- **Frequency:** holding speeds the pin up to the right, releasing to the left. A
  bang-bang controller using the pin's measured acceleration brakes it onto the dot.
- **Reel:** your clicks move the `< o >` dot, the scroom chases the dot, and the box moves on
  its own. The bot steers the dot so the scroom stays centered in the box.

Detection is based on contrast and shapes, not fixed colors or positions, so it handles
different window sizes, rarer (purple) scrooms, dark areas and snow. Note speed isn't assumed
either: it's measured in each minigame, so the much faster notes of rare scrooms are played the
same way.

The macro can't see what's under its own window when that's pinned on top of the game, so it
moves out of the middle of the game when you press Start.

What it touches on your PC: screenshots of the Roblox window only, mouse clicks (and mouse
moves inside the game window for wind), your instrument's hotbar key and A/S/D/F, and one
registered hotkey. Settings, a log and debug screenshots are
saved next to the exe.

## Discord alerts

Paste a channel webhook URL in the app's **Discord** tab. Every catch is posted with a screenshot
of the result, plus a session report every N minutes. Add your Discord user ID to get **pinged**
when the macro stops by itself (Roblox closed, errors, catch goal reached) or gets stuck (no catch
in 10 minutes). The URL and your ID are only ever sent to Discord, and they're removed from bug
reports.

## Run / build from source

Needs Windows and Python 3.12.

```bash
pip install numpy mss pyinstaller
python ScroomMacro.pyw
```

To build the app folder and a release zip: run `build_exe.bat` (update `VERSION` in
`ScroomMacro.pyw` and `PATCH_NOTES.txt` first).

### Practice range

`sim/` has simulated versions of the cast ring, the count-in and the four
minigames. The macro plays them through the same code it uses on the real
game, and the simulator knows exactly when every note should have been hit -
so a change to `bot.py` can be checked in minutes, at any window size, frame
rate, PC speed, input delay and scenery, without opening Roblox.

```bash
pip install pillow
python -m sim.run                          # all four minigames (a few minutes)
python -m sim.run --matrix                 # ... under every condition (slow PC, 30/144 fps, snow...)
python -m sim.run keys --fps 30 --loop 30  # one minigame, slow PC, 30 fps game
python -m sim.run keys --speed 3           # a rare scroom's fast tiles
python -m sim.run wind --shot wind.png     # save a frame to look at
```

It prints, per minigame, how many notes were Perfect / Good / missed and how
many ms off the presses were. The timing rules in it (where Perfect is) were
measured on recordings of the real game; it doesn't ship in the app.

## License

[MIT](LICENSE) © 2026 Kellan
