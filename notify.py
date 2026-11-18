"""Discord webhook notifications: catch alerts (with a screenshot of the
result) and session reports. Uses the webhook URL the user pastes in the app;
nothing is sent anywhere else. Sending happens on a background thread so the
macro never waits on the network."""
import io
import json
import struct
import threading
import time
import urllib.request
import uuid
import zlib

import numpy as np

USER_AGENT = "ScroomMacro (+https://github.com/kellantk/Scrooms-Macro)"
COLOR = 0x7C5CFF
RED = 0xEF4444
AMBER = 0xF59E0B


def png_bytes(rgb, max_w=1280):
    """Encode an RGB numpy image as PNG (downscaled to max_w wide)."""
    h, w = rgb.shape[:2]
    if w > max_w:
        step = int(np.ceil(w / max_w))
        rgb = rgb[::step, ::step]
        h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def valid_url(url):
    url = (url or "").strip()
    return url.startswith(("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/",
                           "https://ptb.discord.com/api/webhooks/", "https://canary.discord.com/api/webhooks/"))


def valid_user_id(uid):
    """A Discord user ID: 17-20 digits (Developer Mode > right-click > Copy User ID)."""
    uid = (uid or "").strip()
    return uid.isdigit() and 15 <= len(uid) <= 21


def send(url, embed, image=None, timeout=10, ping=None):
    """POST one embed (+ optional PNG) to a Discord webhook. Returns (ok, message).
    ping: a Discord user ID to mention (only that user can be pinged)."""
    if not valid_url(url):
        return False, "That isn't a Discord webhook URL."
    payload = {"username": "Scroom Macro", "embeds": [embed], "allowed_mentions": {"parse": []}}
    if ping and valid_user_id(ping):
        payload["content"] = f"<@{ping.strip()}>"
        payload["allowed_mentions"] = {"parse": [], "users": [ping.strip()]}
    if image is not None:
        embed["image"] = {"url": "attachment://catch.png"}
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
                   f"Content-Type: application/json\r\n\r\n".encode())
        body.write(json.dumps(payload).encode())
        body.write(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"files[0]\"; filename=\"catch.png\"\r\n"
                   f"Content-Type: image/png\r\n\r\n".encode())
        body.write(image)
        body.write(f"\r\n--{boundary}--\r\n".encode())
        data, ctype = body.getvalue(), f"multipart/form-data; boundary={boundary}"
    else:
        data, ctype = json.dumps(payload).encode(), "application/json"
    req = urllib.request.Request(url.strip(), data=data, method="POST",
                                 headers={"Content-Type": ctype, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300, f"HTTP {r.status}"
    except Exception as e:      # network problems must never break the macro
        return False, str(e)


_sending = []       # posts on their way


def send_async(url, embed, image=None, on_done=None, ping=None):
    def run():
        res = send(url, embed, image, ping=ping)
        if on_done:
            on_done(*res)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    _sending[:] = [x for x in _sending if x.is_alive()] + [t]


def flush(timeout=3.0):
    """Wait (at most `timeout` s) for posts that are on their way - the app
    is closing, and they'd be cut off."""
    end = time.time() + timeout
    for t in list(_sending):
        t.join(max(0.0, end - time.time()))


def fmt_duration(s):
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s % 60:02d}s"


def catch_embed(minigame, detail, stats):
    return {
        "title": f"Catch finished - {minigame}",
        "description": detail,
        "color": COLOR,
        "fields": [
            {"name": "This session", "value": f"{stats['finished']} finished - {stats['casts']} casts", "inline": True},
            {"name": "Running for", "value": fmt_duration(stats["uptime"]), "inline": True},
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def stuck_embed(stats, minutes, paused=False):
    why = ("It's paused because Roblox isn't the active window." if paused else
           "It keeps trying - check on Roblox (disconnected? a menu open?).")
    return {
        "title": "Macro might be stuck",
        "description": f"No catch in the last {minutes} minutes. {why}",
        "color": AMBER,
        "fields": [
            {"name": "This session", "value": f"{stats['finished']} finished - {stats['casts']} casts", "inline": True},
            {"name": "Running for", "value": fmt_duration(stats["uptime"]), "inline": True},
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def report_embed(stats, final=False, reason=None):
    """Session report. reason: why the macro stopped by itself (shown red)."""
    per = stats.get("per_minigame") or {}
    lines = "\n".join(f"**{k}**: {v}" for k, v in sorted(per.items(), key=lambda kv: -kv[1])) or "nothing yet"
    rate = stats["finished"] / max(stats["uptime"] / 3600, 1e-9)
    # how the game itself scored the strings and keys notes (its Perfect / Good / Miss labels)
    sc = stats.get("scored") or {}
    total = sum(sc.values())
    scored = ([{"name": "Notes (strings & keys)", "inline": False,
                "value": " - ".join(f"{sc[k]} {k.capitalize()}" for k in ("perfect", "good", "miss") if sc.get(k))
                         + f"  ({100 * sc.get('perfect', 0) / total:.0f}% Perfect)"}] if total else [])
    return {
        "title": "Macro stopped" if reason else "Session report" + (" (stopped)" if final else ""),
        **({"description": reason} if reason else {}),
        "color": RED if reason else COLOR,
        "fields": [
            {"name": "Finished", "value": str(stats["finished"]), "inline": True},
            {"name": "Casts", "value": str(stats["casts"]), "inline": True},
            {"name": "Per hour", "value": f"{rate:.0f}", "inline": True},
            {"name": "Running for", "value": fmt_duration(stats["uptime"]), "inline": True},
            {"name": "By instrument", "value": lines, "inline": False},
        ] + scored,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
