"""Checks GitHub for a newer release. Only reads the public releases list -
it never downloads or installs anything by itself: the app shows a button
that opens the release page in your browser."""
import json
import re
import urllib.request

REPO = "kellantk/Scrooms-Macro"
USER_AGENT = "ScroomMacro-UpdateCheck"


def _ver(s):
    return tuple(int(x) for x in re.findall(r"\d+", s or "")[:4]) or (0,)


def check(current, timeout=8):
    """Returns (newer_version, url) if a newer release exists, else (None, None).
    Network errors also give (None, None)."""
    for api, pick in (
        (f"https://api.github.com/repos/{REPO}/releases/latest", lambda d: (d.get("tag_name"), d.get("html_url"))),
        (f"https://api.github.com/repos/{REPO}/tags", lambda d: (d[0]["name"], f"https://github.com/{REPO}/releases") if d else (None, None)),
    ):
        try:
            req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                tag, url = pick(json.loads(r.read().decode()))
            if tag:
                return (tag.lstrip("vV"), url) if _ver(tag) > _ver(current) else (None, None)
        except Exception:
            continue
    return None, None
