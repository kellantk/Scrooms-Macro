"""Let the bot play the simulated minigames and report how well it did.

    python -m sim.run                    all four minigames, normal conditions
    python -m sim.run --matrix           ... and slow PC, 30/144 fps, laggy input, other sizes, snow (slow)
    python -m sim.run strings            one minigame
    python -m sim.run keys --fps 30 --loop 25 --size 1280x720 --scene sand -v
    python -m sim.run wind --shot out.png --at 6.5      save a frame to look at
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot  # noqa: E402
from sim.core import Done, SimIO, stats  # noqa: E402
from sim.games import Session  # noqa: E402


def play(minigame="strings", size=(1280, 720), scene="night", fps=60, loop_ms=7.0, input_ms=12.0,
         display_ms=20.0, hiccup=0.01, rounds=1, seed=0, direct=False, count_in=True, latency_ms=None,
         auto_latency=True, opts=None, ring_opts=None, countin_opts=None, cfg=None, verbose=False, t_end=None):
    """One simulated session. Returns (results, bot log lines, bot)."""
    w, h = size
    game = Session(w, h, scene=scene, minigame=minigame, rounds=rounds, seed=seed, direct=direct,
                   count_in=count_in, opts=opts, ring_opts=ring_opts, countin_opts=countin_opts)
    io = SimIO(game, w, h, fps=fps, loop_ms=loop_ms, input_ms=input_ms, display_ms=display_ms,
               hiccup=hiccup, seed=seed + 1, t_end=t_end or (40.0 * rounds + 20))
    c = bot.Config(equip_on_start=False, cooldown_s=1.0, move_cursor=False)
    # what the bot should have learned: its click reaches the game input_ms
    # later (+ half a frame on average, the game only looks on frames), and
    # what it sees is display_ms old (+ half a frame: the screen holds a frame)
    c.latency_ms = int(round(input_ms + display_ms + 1000 / fps)) if latency_ms is None else latency_ms
    c.auto_latency = auto_latency
    for k, v in (cfg or {}).items():
        setattr(c, k, v)
    b = bot.Bot(c, io, bot.Status())
    lines = []
    b.log = lambda m: (lines.append(f"[{io.t:7.2f}] {m}"), print(lines[-1]) if verbose else None)
    b._write_log = lambda text: lines.append("          " + text.rstrip())
    try:
        b.run()
    except Done:
        pass
    cur = game.current()
    if cur is not None:                 # ran out of time mid-stage: judge what there is
        cur.close()
        game.results += cur.results
    return game.results, lines, b


def summarize(results):
    """{stage: {n, perfect, good, miss, stray, mean, std, worst}}"""
    out = {}
    for r in results:
        s = out.setdefault(r["stage"], {"errs": [], "perfect": 0, "good": 0, "miss": 0, "stray": 0, "wrong": 0})
        s[r["verdict"]] = s.get(r["verdict"], 0) + 1
        if r.get("err") is not None and r["verdict"] in ("perfect", "good", "miss") and r["stage"] != "wind":
            s["errs"].append(r["err"])
        if "cover" in r:
            s.setdefault("cover", []).append(r["cover"])
    for s in out.values():
        s.update(stats(s.pop("errs")))
        if "cover" in s:
            s["cover"] = float(np.mean(s["cover"]))
    return out


def line(name, s):
    notes = s["perfect"] + s["good"] + s["miss"]
    txt = f"{name:<18} {s['perfect']:>3}/{notes:<3} perfect  {s['good']:>2} good  {s['miss']:>2} miss  {s['stray'] + s['wrong']:>2} stray"
    if s["n"]:
        txt += f"   timing {s['mean']:+6.1f} ms  +-{s['std']:4.1f}  worst {s['worst']:5.1f}"
    if "cover" in s:
        txt += f"   on the bar {100 * s['cover']:5.1f}%"
    return txt


MATRIX = [
    # label, kwargs
    ("normal PC, 60 fps", {}),
    ("slow PC (30 ms reads)", {"loop_ms": 30, "hiccup": 0.03}),
    ("30 fps game", {"fps": 30}),
    ("144 fps game", {"fps": 144, "loop_ms": 5}),
    ("laggy input (60 ms)", {"input_ms": 60}),
    ("small window", {"size": (1024, 576)}),
    ("big window", {"size": (2560, 1080)}),
    ("fast notes", {"opts": {"speed": 1.5}}),
    ("on snow", {"scene": "snow"}),
    ("on sand", {"scene": "sand"}),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("minigame", nargs="?", default=None)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--scene", default="night")
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--loop", type=float, default=7.0, help="ms per screen read")
    ap.add_argument("--input", type=float, default=12.0, help="input delay ms")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated minigames")
    ap.add_argument("--matrix", action="store_true", help="run every condition in MATRIX (slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--direct", action="store_true", help="skip cast, ring and count-in")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--shot", help="save one frame as PNG instead of playing")
    ap.add_argument("--at", type=float, default=5.0)
    a = ap.parse_args()
    size = tuple(int(x) for x in a.size.lower().split("x"))
    if a.shot:
        from PIL import Image
        g = Session(size[0], size[1], scene=a.scene, minigame=a.minigame or "strings", seed=a.seed, direct=True,
                    opts={"speed": a.speed})
        g.advance(a.at)
        Image.fromarray(g.frame(a.at)).save(a.shot)
        return
    games = [a.minigame] if a.minigame else (a.only.split(",") if a.only else ["strings", "keys", "percussion", "wind"])
    if a.minigame:
        res, _, _ = play(a.minigame, size=size, scene=a.scene, fps=a.fps, loop_ms=a.loop, input_ms=a.input,
                         rounds=a.rounds, seed=a.seed, direct=a.direct, opts={"speed": a.speed}, verbose=a.verbose)
        for name, s in summarize(res).items():
            print(line(name, s))
        return
    for label, kw in (MATRIX if a.matrix else MATRIX[:1]):
        print(f"\n== {label}", flush=True)
        total = {}
        for g in games:
            res, _, _ = play(g, rounds=a.rounds, seed=a.seed, **kw)
            for name, s in summarize(res).items():
                if name in ("ring", "countin"):
                    t = total.setdefault(name, [])
                    t.append(s)
                else:
                    print("  " + line(name, s), flush=True)
        for name, parts in total.items():
            agg = {k: sum(p.get(k, 0) for p in parts) for k in ("perfect", "good", "miss", "stray", "wrong")}
            agg.update({"n": 0})
            print("  " + line(name, agg))


if __name__ == "__main__":
    main()
