SCROOM MACRO
============

HOW TO INSTALL (Windows PC only)
  1. Download the ScroomMacro_vX.X.zip file.
  2. Right-click the zip -> "Extract All..." -> "Extract".
     Don't double-click the app while you're still looking inside the zip -
     it has to be extracted first.
  3. Extract somewhere short and simple, like Desktop or Documents
     (very long folder paths can stop it from starting).
  4. Open the extracted folder -> open the "App" folder -> double-click
     ScroomMacro.exe. Leave the "_internal" folder next to it - the app needs it.
  5. If Windows says "Windows protected your PC": click "More info" ->
     "Run anyway".
  6. If you get an error when opening it, the app tells you what to do and
     saves startup_error.txt next to ScroomMacro.exe - send that file to the
     creator if it doesn't help.

HAVING A PROBLEM?
  Run the macro until the problem happens, then click "Make bug report"
  (bottom left of the app) and send the .zip file it makes to the creator.
  It contains the log and screenshots of what went wrong.

SUPPORTED INSTRUMENTS (Crescendo Valley update)
  Strings, Keys, Percussion: yes.   Wind: not yet.
  First time you use an instrument the game shows a tutorial - click its
  "Continue" button yourself; the macro waits.

DISCORD ALERTS (optional)
  In Discord: channel settings > Integrations > Webhooks > New Webhook > Copy URL.
  Paste it in the app's Discord tab and click "Send test message".

HOW TO USE
1. Open Roblox and stand where you normally fish, with nothing open
   (no backpack, no shop, no menus).
2. Press F6 (or click Start). The macro switches to Roblox, equips your
   guitar slot, and starts casting. Press F6 again to stop.
3. If you click out of Roblox, the macro lets go of the mouse and pauses
   until Roblox is the active window again.

What it does each cast
  Cast       clicks once
  Ring       waits for the needle to reach the green ("Perfect") part, then clicks
  Frequency  holds/releases to park the pin on the glowing dot
  Reel       holds/releases to move the dot so the scroom stays in the gray box
  Then it waits "secs between catches" and casts again.
  Failsafe   if the ring is up for more than 5 seconds without a good click,
             it presses the guitar slot key twice (unequip + re-equip) and recasts.

Settings
  latency (ms)        How early to react. With "Auto-tune latency" ticked (default)
                      the macro learns this itself: after each ring click it reads where
                      the needle froze and corrects the timing (see macro_log.txt).
                      Manually: if ring clicks land late (past the
                      green), raise it in steps of 10-20. If they land early,
                      lower it. The frequency pin and reel dot use it too.
  Guitar slot         Hotbar key for Tamer's Guitar.
  Ring aim            Perfect = darker green chunk, Middle = middle of the green.
  Cursor position     "Move to safe spot" puts the cursor at the top-middle of the
                      game so the hold never lands on the Cancel button.
  Test detection      Runs the detectors without clicking. Play by hand and
                      watch the Status box / Show log to check it sees each phase.

Settings are saved to App\settings.json. Everything the macro does is
also written to App\macro_log.txt - send that file if something goes wrong.

Files
  App\               the app (ScroomMacro.exe + its _internal support folder)
  ScroomMacro.pyw     app window source  (python ScroomMacro.pyw also works)
  bot.py / detect.py  macro logic and screen detection
  hotkeys.py          the start/stop hotkey
  build_exe.bat       rebuilds the App folder and makes releases\ScroomMacro_v<version>.zip
  make_version.py     used by build_exe.bat

Works at any Roblox window size: the macro finds the frequency panel and reel bar
by their track lines and measures the scale from them. Tested on 3440x1360 and
2560x1020 recordings, plus resized copies down to 1280x720.

Antivirus / "Windows protected your PC"
  The app isn't code-signed, so Windows SmartScreen may warn the first time:
  click "More info" -> "Run anyway". It only listens for its one hotkey
  (registered with Windows, no keyboard hook), captures the Roblox window to
  read the minigames, and sends mouse clicks / the guitar slot key. All the
  source code is included. If Defender still flags it, report it as a false
  positive at https://www.microsoft.com/wdsi/filesubmission