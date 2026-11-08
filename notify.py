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


def send(url, embed, image=None, timeout=10):
    """POST one embed (+ optional PNG) to a Discord webhook. Returns (ok, message)."""
    if not valid_url(url):
        return False, "That isn't a Discord webhook URL."
    payload = {"username": "Scroom Macro", "embeds": [embed]}
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


def send_async(url, embed, image=None, on_done=None):
    def run():
        res = send(url, embed, image)
        if on_done:
            on_done(*res)
    threading.Thread(target=run, daemon=True).start()


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


def report_embed(stats, final=False):
    per = stats.get("per_minigame") or {}
    lines = "\n".join(f"**{k}**: {v}" for k, v in sorted(per.items(), key=lambda kv: -kv[1])) or "nothing yet"
    rate = stats["finished"] / max(stats["uptime"] / 3600, 1e-9)
    return {
        "title": "Session report" + (" (stopped)" if final else ""),
        "color": COLOR,
        "fields": [
            {"name": "Finished", "value": str(stats["finished"]), "inline": True},
            {"name": "Casts", "value": str(stats["casts"]), "inline": True},
            {"name": "Per hour", "value": f"{rate:.0f}", "inline": True},
            {"name": "Running for", "value": fmt_duration(stats["uptime"]), "inline": True},
            {"name": "By instrument", "value": lines, "inline": False},
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
