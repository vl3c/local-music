"""Command-line front end for the local-music server (standard library only).

The server must be running (start.cmd). Everything stays on 127.0.0.1.

    python scripts/music.py "warm lo-fi hip hop, dusty piano, vinyl crackle" --bpm 84    # 4 takes, ranked
    python scripts/music.py "epic orchestral trailer music" --duration 120 --out D:/game/music
    python scripts/music.py "chiptune boss battle" --bpm 160 --loop-bars 8 --out D:/game/music
    python scripts/music.py "solo nylon guitar, flamenco" --stem --key "E minor"
    python scripts/music.py --extend 20261010-120000-ab12:1234 --seconds 60       # continue a take
    python scripts/music.py --vary 20261010-120000-ab12:1234 --strength 0.3        # a variation
    python scripts/music.py --inpaint 20261010-120000-ab12:1234 --span 20-28 "a drum fill"
    python scripts/music.py --master 20261010-120000-ab12 --fade-out 4 --lufs -16  # re-master, no regeneration
    python scripts/music.py --import D:/audio/demo.mp3 "a slow acoustic folk song"    # your own audio, as a take
    python scripts/music.py --vary D:/audio/demo.mp3 "the same melody as a string quartet" --strength 0.5
    python scripts/music.py --combine A:1 B:7@20-60 C:3 --bridge 8                  # one song, bridged
    python scripts/music.py --mix SONG:1,vol=-8 VOICE:0,at=4                         # a voice over a song
    python scripts/music.py --show-prompt "dark synthwave" --bpm 100
    python scripts/music.py --show 20261010-120000-ab12                             # an earlier request's takes
    python scripts/music.py --save-defaults --n 6 --lufs -16                         # your defaults (page and API too)

A take is '<request id>:<seed>' (printed as `ref`); --extend, --vary and --inpaint also take
an audio file, which is imported first. Delivered files are named after the
request (warm-lofi-hip-hop_v01.wav, .mp3 and a .json sidecar) and never overwrite anything.

Exit status: 0 done, 1 failed or no server, 2 bad arguments, 3 every take was rejected.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8192"
# Never through a proxy: urllib honours HTTP_PROXY even for 127.0.0.1 unless NO_PROXY
# says otherwise, which would hand the prompts to whatever the proxy is.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OUTPUT = Path(__file__).resolve().parent.parent / "output"
NAME = re.compile(r"^[A-Za-z0-9-]+$")
SPAN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
PART = re.compile(r"^(?P<ref>[0-9]{8}-[0-9]{6}-[0-9a-f]{4}:[0-9]+)(?:@(?P<a>\d+(?:\.\d+)?)?-(?P<b>\d+(?:\.\d+)?)?)?$")
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")


def call(path: str, payload: dict | None = None, timeout: float = 3600, method: str | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"},
                                 method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("detail")
        except Exception:
            detail = exc.reason
        if isinstance(detail, list):  # pydantic validation errors
            detail = "; ".join(f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
        print(f"server said {exc.code}: {detail}", file=sys.stderr)
        sys.exit(2 if exc.code in (400, 404, 422) else 1)
    except (OSError, http.client.HTTPException) as exc:
        # Refused, reset or timed out: the server is down, restarting or was stopped mid-request.
        print(f"lost the server at {SERVER}: {exc}", file=sys.stderr)
        sys.exit(1)


def upload(path: Path, text: str | None = None, name: str | None = None, check: bool | None = None,
           kind: str | None = None) -> dict:
    """POST an audio file's bytes to /api/import: it comes back as a request with one take (seed 0)."""
    q = urllib.parse.urlencode(given({"filename": path.name, "text": text, "name": name, "kind": kind,
                                      "check": None if check is None else str(check).lower()}))
    req = urllib.request.Request(f"{SERVER}/api/import?{q}", data=path.read_bytes(),
                                 headers={"Content-Type": "application/octet-stream"}, method="POST")
    try:
        with _OPENER.open(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("detail")
        except Exception:
            detail = exc.reason
        print(f"server said {exc.code}: {detail}", file=sys.stderr)
        sys.exit(2 if exc.code in (400, 404, 413, 422) else 1)
    except (OSError, http.client.HTTPException) as exc:
        print(f"lost the server at {SERVER}: {exc}", file=sys.stderr)
        sys.exit(1)


def parse_part(spec: str) -> dict:
    """REF or REF@START-END (either end may be left out: REF@30- is from 30 s to the end)."""
    m = PART.fullmatch(spec.strip())
    if not m:
        raise ValueError(f"a part is REQ:SEED or REQ:SEED@START-END (seconds), not {spec!r}")
    return given({"take": m["ref"], "start": float(m["a"]) if m["a"] else None,
                  "end": float(m["b"]) if m["b"] else None})


def parse_layer(spec: str) -> dict:
    """REF[@START-END][,vol=DB][,at=SECONDS]: a take (or a span of it) at a volume, starting at a time."""
    head, *opts = [x.strip() for x in spec.split(",")]
    layer = parse_part(head)
    for o in opts:
        k, _, v = o.partition("=")
        try:
            if k in ("vol", "gain"):
                layer["gain_db"] = float(v.lower().removesuffix("db"))
            elif k == "at":
                layer["at"] = float(v.lower().removesuffix("s"))
            else:
                raise ValueError
        except ValueError:
            raise ValueError(f"layer option {o!r}: use vol=DB (e.g. vol=-8) or at=SECONDS (e.g. at=4)") from None
    return layer


def is_audio_file(arg: str | None) -> bool:
    return bool(arg) and arg.lower().endswith(AUDIO_EXT) and Path(arg).is_file()


def wait_for_server(seconds: float) -> bool:
    deadline = time.time() + seconds
    while True:
        try:
            _OPENER.open(SERVER + "/api/health", timeout=5)
            return True
        except OSError:
            if time.time() >= deadline:
                return False
            time.sleep(2)


def play(path: Path) -> None:
    """Play a WAV on this machine's default output device, blocking until it ends."""
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    players = [["afplay"]] if sys.platform == "darwin" else [
        ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]]
    for cmd in players:
        if shutil.which(cmd[0]):
            subprocess.run([*cmd, str(path)], check=True)
            return
    raise RuntimeError("no audio player found (tried " + ", ".join(c[0] for c in players) + ")")


# --------------------------------------------------------------------------- delivery

def _used_versions(out: Path, base: str) -> set[int]:
    """Version numbers already taken in `out` for this base name (case-insensitive, like Windows)."""
    rx = re.compile(re.escape(base) + r"_v(\d+)\.(wav|mp3|json|png)", re.I)
    return {int(m.group(1)) for p in out.iterdir() if (m := rx.fullmatch(p.name))}


def sidecar(res: dict, take: dict) -> dict:
    keys = ("seed", "rank", "rank_score", "clap", "aes", "music", "fails", "warnings", "post", "final", "signal")
    return {"request": res["id"], "ref": take.get("ref"), "op": res.get("op"), "source": res.get("source"),
            "text": res.get("text"), "prompt": res.get("prompt"), "bpm": res.get("bpm"), "key": res.get("key"),
            "engine": res.get("engine"), "model": res.get("model"), "steps": res.get("steps"),
            **{k: take[k] for k in keys if k in take}}


def deliver(res: dict, takes: list[dict], out: Path, fmt: str, png: bool, name: str | None = None) -> list[Path]:
    """Copy takes to `out` as <name>_vNN.wav / .mp3 (+ .json sidecar, + .png with --png).

    Never overwrites: numbering continues after whatever is there (compared case-insensitively),
    and each file is created exclusively, so even a file appearing meanwhile isn't clobbered.
    """
    if name is not None and not NAME.fullmatch(name):
        raise ValueError(f"--name takes letters, digits and hyphens only, not {name!r}")
    out.mkdir(parents=True, exist_ok=True)
    base = name or res["name"]
    exts = {"all": ["wav", "mp3"], "wav": ["wav"], "mp3": ["mp3"]}[fmt]
    used = _used_versions(out, base)
    v = 1
    while any((v + k) in used for k in range(len(takes))):
        v += 1
    paths = []
    for t in takes:
        while True:
            stem = f"{base}_v{v:02d}"
            try:
                first = out / f"{stem}.{exts[0]}"
                with open(first, "xb") as fh, open(OUTPUT / t["files"][exts[0]], "rb") as fs:
                    shutil.copyfileobj(fs, fh)
                break
            except FileExistsError:
                v += 1
        paths.append(first)
        for ext in exts[1:]:
            shutil.copyfile(OUTPUT / t["files"][ext], out / f"{stem}.{ext}")
            paths.append(out / f"{stem}.{ext}")
        (out / f"{stem}.json").write_text(json.dumps(sidecar(res, t), indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        if png:
            shutil.copyfile(OUTPUT / t["files"]["png"], out / f"{stem}.png")
            paths.append(out / f"{stem}.png")
        v += 1
    return paths


# --------------------------------------------------------------------------- printing

def describe(t: dict) -> str:
    bits = [f"#{t.get('rank', '?')}", f"ref {t.get('ref')}", f"score {t.get('rank_score', 0):.2f}"]
    aes = t.get("aes") or {}
    if aes:
        bits.append("  ".join(f"{k} {aes[k]:.1f}" for k in ("CE", "PQ") if k in aes))
    if t.get("clap") is not None:
        bits.append(f"CLAP {t['clap']:.2f}")
    mus = t.get("music", {})
    tempo, key = mus.get("tempo", {}), mus.get("key", {})
    if tempo.get("bpm"):
        bits.append(f"{tempo['bpm']:.0f} BPM" + (f" ({tempo['match']})" if tempo.get("match") else ""))
    if key.get("key"):
        bits.append(key["key"] + (f" ({key['match']})" if key.get("match") else ""))
    p = t.get("post", {})
    if "length_s" in p:
        bits.append(f"{p['length_s']:.1f}s")
    if "loop" in p:
        lp = p["loop"]
        bits.append(f"loop {lp['bars']} bars @ {lp['bpm']:g} BPM from {lp['start_s']:.1f}s, seam "
                    + ("ok" if p.get("seam_ok") else "CHECK"))
    line = "  ".join(bits)
    if t.get("rejected"):
        line += "  REJECTED: " + "; ".join(t["fails"])
    for w in t.get("warnings", []):
        line += f"\n      warning: {w}"
    for note in (tempo.get("note"), key.get("note")):
        if note:
            line += f"\n      note: {note}"
    return line


def print_request(res: dict) -> None:
    what = res.get("op", "generate") + (f" of {res['source']}" if res.get("source") else "")
    if res.get("layers"):
        what += " of " + " + ".join(f"{ly['take']} {ly['gain_db']:+g} dB" + (f" at {ly['at']:g} s" if ly.get("at") else "")
                                    for ly in res["layers"])
    if res.get("parts"):
        what += " of " + " + ".join(pt["take"] + (f"@{pt['start']:g}-{pt['end']:g}" if pt.get("start") else "")
                                    for pt in res["parts"]) + f" (bridges of {res.get('bridge', 0):g} s)"
    print(f"request {res['id']}  {what}  {res.get('engine')}  {res.get('name')}")
    if res.get("prompt"):
        print(f"prompt:   {res['prompt']}")
    if "negative" in res.get("ignored", []):
        print("negative: ignored (the post-trained Stable Audio 3 models don't use it)")
    if "gen_s" in res:
        print(f"{len(res['takes'])} take(s) of {res.get('gen_length_s', 0):g} s in {res['gen_s']}s "
              f"+ {res.get('score_s', 0)}s checks; {res['kept']} kept")
    for t in res["takes"]:
        print("  " + describe(t))
        print(f"      {t['paths']['wav']}")
        print(f"      {t['paths']['png']}")
    for w in res.get("warnings", []):
        print(f"WARNING: {w}")


# --------------------------------------------------------------------------- main

def given(d: dict) -> dict:
    """Drop options left unset, so the server's (or the user's saved) defaults apply."""
    return {k: v for k, v in d.items() if v is not None}


def master_fields(a) -> dict:
    out = given({"length": a.length, "fade_in": a.fade_in, "fade_out": a.fade_out, "lufs": a.lufs,
                 "peak_dbtp": a.peak, "loop_bars": a.loop_bars, "beats_per_bar": a.beats_per_bar})
    if a.no_loop:
        out["loop_bars"] = None  # an explicit null clears it on the server
    if a.full_length:
        out["length"] = None
    return out


def shape_fields(a) -> dict:
    out = given({"n": a.n, "engine": a.engine, "steps": a.steps, "check": a.check})
    if a.seeds:
        out["seeds"] = [int(s) for s in a.seeds.split(",") if s.strip()]
    elif a.seed is not None:
        out["seed"] = None if a.seed.lower() == "random" else int(a.seed)
    return out


def main() -> int:
    if not sys.stdout.isatty():  # piped (as agents run it): Windows would default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="the music: genre, instruments, mood, energy, era, production "
                    "(with --extend / --vary / --inpaint: a new prompt for that part; default the source's)")
    g = ap.add_argument_group("generation")
    g.add_argument("--duration", type=float, help="seconds, 2-380 on medium, 2-120 on small (default 30)")
    g.add_argument("--bpm", type=float, help="tempo: added to the prompt and checked in the result")
    g.add_argument("--key", help="key, e.g. 'A minor', 'F# major', 'Bbm': added to the prompt and checked")
    g.add_argument("--stem", action="store_true", help="one instrument or part (a drum loop, a solo piano), "
                   "not a full arrangement")
    g.add_argument("--n", type=int, help="takes to make and rank (best-of-N; default 4)")
    g.add_argument("--seed", help="start seed (takes use seed, seed+1, ...), or 'random' (the default)")
    g.add_argument("--seeds", help="exact seeds, comma-separated (e.g. to remake one take: --seeds 1234)")
    g.add_argument("--engine", choices=["medium", "small"], help="medium (Stable Audio 3 Medium, the default) or "
                   "small (Small-Music: lighter, lower quality, max 120 s; install with -WithSmall)")
    g.add_argument("--raw", action="store_true", default=None, help="send the text as written: no tags, "
                   "tempo or key added")
    g.add_argument("--negative", help="negative prompt (ignored by the post-trained models; kept for -base ones)")
    g.add_argument("--steps", type=int, help="sampling steps (default 8; more doesn't help the post-trained models)")
    g.add_argument("--check", dest="check", action="store_true", default=None,
                   help="CLAP + Audiobox Aesthetics scores (default on)")
    g.add_argument("--no-check", dest="check", action="store_false", help="skip the model scores (the signal, "
                   "tempo and key checks still run)")
    s = ap.add_argument_group("from earlier takes or your own audio (REF is REQ:SEED)")
    s.add_argument("--extend", metavar="REF", help="continue the take: the result is the whole track, longer")
    s.add_argument("--seconds", type=float, help="with --extend: how much new music (default 30)")
    s.add_argument("--vary", metavar="REF", help="a variation of the take (audio to audio)")
    s.add_argument("--strength", type=float, help="with --vary: 0.05 (nearly the same) .. 1 (unrelated); default 0.5")
    s.add_argument("--inpaint", metavar="REF", help="regenerate a span of the take, keep the rest")
    s.add_argument("--span", help="with --inpaint: START-END in seconds, e.g. 20-28")
    s.add_argument("--import", dest="import_file", metavar="FILE", type=Path,
                   help="your own audio (WAV, FLAC, OGG, MP3) as a take; the text describes it. --extend, --vary "
                        "and --inpaint also take a file directly")
    s.add_argument("--combine", nargs="+", metavar="PART", help="takes in order as one song, with a bridge written "
                   "between each two: REF or REF@START-END; the text describes the bridges")
    s.add_argument("--bridge", type=float, help="with --combine: seconds of new music at each join (default 2 bars, "
                   "or 4 s without a tempo; 0 = blend the parts straight into each other)")
    s.add_argument("--blend", type=float, help="with --combine: seconds of each part regenerated beside a bridge "
                   "(default 1)")
    s.add_argument("--mix", nargs="+", metavar="LAYER", help="takes on top of each other (a voice over a song): "
                   "REF[@START-END][,vol=DB][,at=SECONDS]; each layer at its own volume (default 0 dB), "
                   "starting at its time (default 0)")
    m = ap.add_argument_group("mastering (also with --master REQ)")
    m.add_argument("--length", type=float, help="cut the delivered track to this many seconds (with a fade)")
    m.add_argument("--full-length", action="store_true", help="with --master: undo an earlier --length")
    m.add_argument("--fade-in", type=float, help="seconds (default 0)")
    m.add_argument("--fade-out", type=float, help="seconds (default: automatic, a 3 s fade only when the take stops "
                   "at full level; 0 keeps a hard cut)")
    m.add_argument("--lufs", type=float, help="loudness target (default -14, streaming level; -16..-18 for "
                   "background music under dialogue)")
    m.add_argument("--peak", type=float, help="true-peak ceiling in dBTP (default -1)")
    m.add_argument("--loop-bars", type=int, help="deliver a seamless loop of this many bars, cut on the beat "
                   "(give --bpm when generating, so the take is long enough)")
    m.add_argument("--beats-per-bar", type=int, help="for loops and the plot's grid (default 4)")
    m.add_argument("--no-loop", action="store_true", help="with --master: back to the whole track")
    o = ap.add_argument_group("other actions")
    o.add_argument("--master", metavar="REQ", help="re-master a request's takes with the options above; no regeneration")
    o.add_argument("--show-prompt", action="store_true", help="print the prompt the engine would get, and exit")
    o.add_argument("--list", action="store_true", help="recent requests")
    o.add_argument("--show", metavar="REQ", help="an earlier request: its prompt and ranked takes, with their files")
    o.add_argument("--settings", action="store_true", help="show the defaults in effect")
    o.add_argument("--save-defaults", action="store_true", help="save the given --n, --duration, --engine, --steps, "
                   "--seed, --raw/--no-check, --lufs, --peak, --fade-in, --fade-out, --strength as your defaults "
                   "(config.local.json; the page and the API use them too); others stay as saved")
    o.add_argument("--reset-defaults", action="store_true", help="forget your saved defaults: back to the built-in ones")
    o.add_argument("--engines", action="store_true", help="engines: installed / running")
    o.add_argument("--delete", metavar="REQ", help="delete a request's files from output/")
    d = ap.add_argument_group("delivery")
    d.add_argument("--out", type=Path, help="copy results here as <name>_vNN.wav/.mp3 + .json (never overwrites)")
    d.add_argument("--format", choices=["all", "wav", "mp3"], default="all", help="with --out: all (WAV + MP3, "
                   "the default), wav or mp3")
    d.add_argument("--best", type=int, default=1, help="with --out: how many of the best kept takes (default 1)")
    d.add_argument("--all", action="store_true", help="with --out: every kept take")
    d.add_argument("--png", action="store_true", help="with --out: copy the waveform/spectrogram PNGs too")
    d.add_argument("--name", help="file name for delivered files (letters, digits, hyphens; default from the text)")
    d.add_argument("--play", action="store_true", help="play the best take on this machine's speakers")
    d.add_argument("--no-save", action="store_true", help="delete the request from output/ when done (after --out/--play)")
    d.add_argument("--json", action="store_true", help="print only JSON (the server's response plus `delivered`)")
    d.add_argument("--wait", type=float, default=5, help="seconds to wait for the server to come up")
    a = ap.parse_args()

    if a.seed is not None and a.seed.lower() != "random" and not a.seed.isdigit():
        ap.error("--seed takes a number or 'random'")
    sources = [x for x in (a.extend, a.vary, a.inpaint, a.combine, a.mix, a.import_file) if x]
    if len(sources) > 1:
        ap.error("use one of --extend, --vary, --inpaint, --combine, --mix, --import at a time")
    if not (a.text or sources or a.master or a.list or a.show or a.settings or a.save_defaults or a.reset_defaults
            or a.engines or a.delete):
        ap.error("give the music to make, or --extend / --vary / --inpaint REF, --import FILE, --combine PARTS, "
                 "--mix LAYERS, --master REQ, --list, --show REQ, --settings, --save-defaults, --reset-defaults, "
                 "--engines or --delete")
    layers = None
    if a.mix:
        if len(a.mix) < 2:
            ap.error("--mix needs at least two layers")
        try:
            layers = [parse_layer(x) for x in a.mix]
        except ValueError as exc:
            ap.error(str(exc))
    if a.import_file is not None and not a.import_file.is_file():
        ap.error(f"--import: no file {a.import_file}")
    parts = None
    if a.combine:
        if len(a.combine) < 2:
            ap.error("--combine needs at least two parts")
        try:
            parts = [parse_part(x) for x in a.combine]
        except ValueError as exc:
            ap.error(str(exc))
    if a.inpaint and not a.span:
        ap.error("--inpaint needs --span START-END (seconds)")
    span = None
    if a.span:
        if not (mm := SPAN.fullmatch(a.span)):
            ap.error("--span takes START-END in seconds, e.g. 20-28")
        span = (float(mm.group(1)), float(mm.group(2)))
    if a.name is not None and not NAME.fullmatch(a.name):
        ap.error("--name takes letters, digits and hyphens only")
    if not wait_for_server(a.wait):
        print(f"No server at {SERVER}. Start it with start.cmd (agents: start.cmd --no-browser).", file=sys.stderr)
        return 1

    if a.reset_defaults:
        call("/api/settings", method="DELETE")
        print("saved defaults forgotten: the built-in ones apply")
        a.settings = True
    if a.save_defaults:
        new = given({"n": a.n, "duration": a.duration, "engine": a.engine, "steps": a.steps, "raw": a.raw,
                     "check": a.check, "lufs": a.lufs, "peak_dbtp": a.peak, "fade_in": a.fade_in,
                     "fade_out": a.fade_out, "strength": a.strength})
        if a.seed is not None:
            new["seed"] = None if a.seed.lower() == "random" else int(a.seed)
        if not new:
            ap.error("--save-defaults saves the options given with it, e.g. --save-defaults --n 6 --lufs -16")
        call("/api/settings", {**call("/api/settings")["saved"], **new})
        print("saved: " + ", ".join(f"{k} {v}" for k, v in new.items()))
        a.settings = True
    if a.settings:
        st = call("/api/settings")
        print("in effect: " + (f"your saved defaults ({st['file']}) over the built-in ones" if st["file"]
                               else "the built-in defaults"))
        for k, v in st["effective"].items():
            shown = "random" if (k == "seed" and v is None) else "engine default" if (v is None) else v
            print(f"  {k:10s} {shown}{'  (saved)' if k in st['saved'] else ''}")
        return 0
    if a.engines:
        for k, e in call("/api/engines").items():
            print(f"{k:7s} {e['label']:28s} {'installed' if e['installed'] else 'NOT installed'}"
                  f"{', running' if e['running'] else ''}")
        return 0
    if a.show:
        res = call(f"/api/requests/{a.show}")
        if a.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print_request(res)
        return 0
    if a.list:
        for r in call("/api/requests?limit=30"):
            print(f"{r['id']}  {r.get('op') or 'generate':8s} {r.get('duration') or 0:6.1f}s  {r['takes']} take(s)  "
                  f"{r.get('name')}: {(r.get('text') or '')[:60]}")
        return 0
    if a.delete:
        r = call(f"/api/requests/{a.delete}", method="DELETE")
        print(f"deleted request {r['id']} ({r['deleted']} files)")
        return 0
    if a.show_prompt and sources:
        ap.error("--show-prompt shows a new request's prompt: use it without --extend/--vary/--inpaint/--combine/--mix/--import")
    if a.show_prompt and a.text:
        q = urllib.parse.urlencode(given({"text": a.text, "bpm": a.bpm, "key": a.key,
                                          "kind": "stem" if a.stem else "music", "raw": "true" if a.raw else None}))
        r = call(f"/api/prompt?{q}")
        print(json.dumps(r, indent=2) if a.json else f"prompt: {r['prompt']}\nfile:   {r['name']}_v01.wav")
        return 0

    # An audio file given to --extend / --vary / --inpaint is imported first (undescribed: the
    # text, if any, is the prompt for the new music, not a description of the file).
    imported = []
    for attr in ("extend", "vary", "inpaint"):
        val = getattr(a, attr)
        if val and not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}[:#/][0-9]+", val):
            if not is_audio_file(val):
                ap.error(f"--{attr} takes a take's ref (REQ:SEED) or an audio file (WAV, FLAC, OGG, MP3), not {val!r}")
            imp = upload(Path(val), check=False)
            imported.append(imp["id"])
            if not a.json:
                print(f"imported {val} as {imp['takes'][0]['ref']}")
            setattr(a, attr, imp["takes"][0]["ref"])

    text = {"text": a.text} if a.text else {}
    if a.import_file is not None:
        res = upload(a.import_file, a.text, a.name, a.check, "stem" if a.stem else None)
    elif a.mix:
        res = call("/api/mix", {"layers": layers, **master_fields(a), **given({"name": a.name, "check": a.check})})
    elif a.combine:
        res = call("/api/arrange", {"parts": parts, **text, **shape_fields(a), **master_fields(a),
                                    **given({"bridge": a.bridge, "blend": a.blend, "name": a.name})})
    elif a.master:
        res = call("/api/master", {"request": a.master, **master_fields(a), **given({"bpm": a.bpm, "check": a.check})})
    elif a.extend:
        res = call("/api/extend", {"take": a.extend, **text, **shape_fields(a), **master_fields(a),
                                   **given({"seconds": a.seconds, "raw": a.raw})})
    elif a.vary:
        res = call("/api/vary", {"take": a.vary, **text, **shape_fields(a), **master_fields(a),
                                 **given({"strength": a.strength, "raw": a.raw})})
    elif a.inpaint:
        res = call("/api/inpaint", {"take": a.inpaint, **text, "start": span[0], "end": span[1],
                                    **shape_fields(a), **master_fields(a), **given({"raw": a.raw})})
    else:
        res = call("/api/generate", {"text": a.text, "kind": "stem" if a.stem else "music", **shape_fields(a),
                                     **master_fields(a), **given({"duration": a.duration, "bpm": a.bpm, "key": a.key,
                                                                  "raw": a.raw, "negative": a.negative,
                                                                  "name": a.name})})

    kept = [t for t in res["takes"] if not t.get("rejected")]
    delivered: list[Path] = []
    if a.out:
        pick = kept if a.all else kept[:max(1, a.best)]
        try:
            delivered = deliver(res, pick, a.out, a.format, png=a.png, name=a.name)
        except ValueError as exc:
            ap.error(str(exc))

    if a.json:
        print(json.dumps({**res, "delivered": [str(p) for p in delivered]}, indent=2, ensure_ascii=False))
    else:
        print_request(res)
        if delivered:
            print("delivered:")
            for p in delivered:
                print(f"  {p}")

    status = 0 if kept else 3
    try:
        if a.play:
            target = next((p for p in delivered if p.suffix == ".wav"), None) or \
                OUTPUT / (kept[0] if kept else res["takes"][0])["files"]["wav"]
            try:
                sys.stdout.flush()
                play(target)
                if not a.json:
                    print(f"played {target.name}")
            except Exception as exc:  # the take exists; only the playback failed
                print(f"could not play {target}: {exc}", file=sys.stderr)
                status = 1
    finally:
        if a.no_save and res.get("id") and not a.master:
            for rid in [res["id"], *imported]:  # the file imported on the way, too
                call(f"/api/requests/{rid}", method="DELETE")
            if not a.json:
                print(f"not saved: removed request {res['id']}" + (f" and the import {imported[0]}" if imported else "")
                      + " from output/")
    if status == 3 and not a.json:
        print("every take was rejected: try other seeds, or rephrase the prompt", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
