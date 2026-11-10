SCROOM MACRO
============

INSTALL (Windows PC only)
  1. Right-click the zip -> "Extract All..." -> "Extract".
     (Don't run the app from inside the zip - extract it first.)
  2. Extract somewhere short and simple, like Desktop or Documents.
  3. Open the extracted folder and double-click ScroomMacro.exe.
     Keep the "_internal" folder next to it - the app needs it.
  4. If Windows says "Windows protected your PC": click "More info" -> "Run anyway"
     (the app isn't code-signed).

USE
  1. Open Roblox and stand where you fish, with no menus open.
  2. Press F6 (or click Start). The macro equips your instrument and starts
     casting. Press F6 again to stop.
  3. If you click out of Roblox, the macro lets go and pauses until you're back.
  The first time you use an instrument, click "Continue" on the game's
  tutorial yourself - the macro waits for it.

SUPPORTED INSTRUMENTS (Crescendo Valley update)
  Strings, Keys, Percussion, Wind: all supported.
  Wind: the macro moves your mouse up and down to steer the scroom -
  don't touch the mouse while it plays.

WINDOW
  Pin (top right)    keeps the app on top of the game.
  Arrow (top right)  mini view: only the status and the Start button.

TIMING
  Latency and Note timing are learned automatically while it plays
  ("Auto-tune" in the Timing tab). Getting "Good" instead of "Perfect"
  on strings/keys? Note timing: higher = press later.

DISCORD ALERTS (optional)
  1. Discord: channel settings > Integrations > Webhooks > New Webhook > Copy URL.
     Paste it in the app's Discord tab.
  2. To get pinged when the macro stops by itself or gets stuck:
     Discord Settings > Advanced > turn on Developer Mode, then right-click your
     name > Copy User ID, and paste it in "Your Discord user ID".
  3. Click "Send test".

HAVING A PROBLEM?
  Run the macro until the problem happens, then go to More > "Make bug report"
  and send the .zip it makes to the creator.

ANTIVIRUS
  The app only captures the Roblox window, sends mouse clicks and your
  instrument / A S D F keys, and listens for its one hotkey. All the source code
  is on GitHub: https://github.com/kellantk/Scrooms-Macro
  If Defender flags it, report a false positive at
  https://www.microsoft.com/wdsi/filesubmission

Settings (settings.json) and the log (macro_log.txt) are saved next to ScroomMacro.exe.
