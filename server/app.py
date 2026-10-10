"""local-music server: Stable Audio 3 instrumental music + automatic checks + mastering.

Loopback only (127.0.0.1:8192). The engine runs as a child process (server/workers/), one
per model, so stopping it frees all of its VRAM; this process holds the checkers (CLAP and
Audiobox Aesthetics) and does all analysis and mastering, so every take is judged the same way.

    start.cmd   (the server runs only in the console window start.cmd opens)

A request makes N takes (best-of-N, default 4) on consecutive seeds, checks and scores each,
rejects the broken ones and returns them all ranked. Its files live in output/<id>/:

    request.json         the request, the prompt actually sent, and every take's scores
    <seed>_raw.wav       the engine's output, as generated (44.1 kHz float stereo)
    <seed>.wav / .mp3    the delivered track: 48 kHz 24-bit, mastered (or the loop, for loops),
                         with prompt, seed, model and scores in the WAV's INFO chunk
    <seed>.png           waveform, loudness and spectrogram with the bar grid

Kinds of request: generate (text to music), extend (continue a take), vary (a variation
of a take: audio to audio), inpaint (regenerate a span of a take), import (your own audio
file, as a take the others can start from), arrange (several takes in order, with a bridge
written between each two) and mix (takes on top of each other, each at its own volume). /api/master re-masters a request's takes (length, fades,
loudness, loop) without regenerating.
"""

from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
OUTPUT = REPO / "output"
LOGS = REPO / "logs"
PID_FILE = OUTPUT / ".server.pid"
SETTINGS_FILE = REPO / "config.local.json"
sys.path.insert(0, str(REPO))
# Everything is loaded from models/: never let a library reach for the network.
for _k, _v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
               "DO_NOT_TRACK": "1", "TRANSFORMERS_VERBOSITY": "error",
               # The Intel Fortran runtime that comes with torch's maths libraries installs its own
               # console handler, which aborts the process when the window closes ("forrtl: error
               # (200)") before anything else can run. Off: the server's own handler cleans up instead.
               "FOR_DISABLE_CONSOLE_CTRL_HANDLER": "1"}.items():
    os.environ[_k] = _v

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

import serverlog  # noqa: E402  (server/, on sys.path as the script's folder)
from music import __version__, analysis, audio, checks, plot, post, prompts  # noqa: E402

PORT = 8192
WINDOW_MARK = "LOCAL_MUSIC_WINDOW"  # set by start.cmd for the console window it opens, and only there
REQ_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.(wav|mp3|png|json)$")
TAKE_REF = re.compile(r"^(?P<rid>[0-9]{8}-[0-9]{6}-[0-9a-f]{4})[:#/](?P<seed>[0-9]+)$")
RAW_SR = 44100  # what the engine writes

app = FastAPI(title="local-music", docs_url=None, redoc_url=None, openapi_url=None)
# Refuse any Host but our own: a web page that DNS-rebinds its name to 127.0.0.1
# must not be able to read takes or drive the server.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def _same_origin_only(request, call_next):
    """Refuse writes that a web page on another site makes the user's browser send.

    Browsers label those with Origin / Sec-Fetch-Site; the CLI and curl send neither, and
    the server's own page sends its own origin. Also forbid framing the page elsewhere.
    """
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin is not None and origin not in _own_origins(request)):
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    return response


def _own_origins(request) -> set[str]:
    port = request.url.port or 80
    return {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


@app.exception_handler(PermissionError)
async def _file_in_use(request, exc):
    # Windows won't replace or delete a file another handle has open: the browser still
    # streaming a take's MP3, an editor holding a WAV. audio.replace() retries for a while first.
    return JSONResponse({"detail": f"a file is in use ({exc.filename or exc}): stop playing or downloading it, "
                                   "then try again"}, status_code=409)


_lock = threading.Lock()  # one GPU job at a time: generation and scoring share the card
_file_lock = threading.Lock()  # settings: a small file that mustn't wait for a GPU job


# --------------------------------------------------------------------------- engines

ENGINES = {
    "medium": {"weights": "sa3-medium/model.safetensors", "label": "Stable Audio 3 Medium"},
    "small": {"weights": "sa3-small-music/model.safetensors", "label": "Stable Audio 3 Small-Music"},
}
EngineName = Literal["medium", "small"]


def engine_installed(name: str) -> bool:
    return (REPO / ".venv" / "Scripts" / "python.exe").is_file() and (MODELS / ENGINES[name]["weights"]).is_file() \
        and (MODELS / "sa3-medium" / "t5gemma-b-b-ul2" / "model.safetensors").is_file()


class Worker:
    """A model in its own process, speaking workers/protocol.py."""

    START_TIMEOUT = 600.0  # loads ~10 GB and warms up
    JOB_TIMEOUT = 3600.0

    def __init__(self, name: str):
        if not engine_installed(name):
            raise HTTPException(503, f"engine {name!r} isn't installed: re-run install.ps1"
                                     + (" -WithSmall" if name == "small" else ""))
        self.name = name
        t0 = time.time()
        # The worker's stderr (its logs, progress bars) goes through this process's log
        # and console filter, like everything else the server prints.
        self.proc = subprocess.Popen(
            [str(REPO / ".venv" / "Scripts" / "python.exe"), str(Path(__file__).parent / "workers" / "sa3_worker.py"),
             name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO),
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        # Reader threads, so a hung worker times out instead of holding _lock forever,
        # and its stderr pipe never fills up and blocks it.
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        try:
            hello = self._read(self.START_TIMEOUT)
            if not hello.get("ready"):
                raise RuntimeError(f"{name} worker failed to start: {hello.get('error', hello)}")
        except Exception:
            self.close()
            raise
        self.info = hello
        print(f"[local-music] {ENGINES[name]['label']} ready in {time.time() - t0:.1f}s", flush=True)

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put("")  # EOF

    def _pump_err(self) -> None:
        for line in self.proc.stderr:
            sys.stderr.write(line)

    def _read(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"{self.name} worker gave no answer in {timeout:.0f}s") from None
        if not line:
            raise RuntimeError(f"{self.name} worker exited (code {self.proc.poll()}); see logs/server.log")
        return json.loads(line)

    def call(self, req: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            reply = self._read(self.JOB_TIMEOUT)
        except Exception:
            # Hung, dead or out of sync: kill it, so the next request starts a fresh one.
            self.close()
            raise
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "unknown worker error"))
        return reply

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()


_workers: dict[str, Worker] = {}


def worker(name: str) -> Worker:
    """The engine's worker, started (or restarted, if it died) on demand. Call with _lock held."""
    if name not in ENGINES:
        raise HTTPException(400, f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    w = _workers.get(name)
    if w is None or not w.alive():
        _workers[name] = w = Worker(name)
    return w


def stop_worker(name: str) -> bool:
    w = _workers.pop(name, None)
    if w is not None:
        w.close()
        return True
    return False


@atexit.register
def _stop_workers() -> None:
    for w in list(_workers.values()):
        w.close()


# --------------------------------------------------------------------------- scorers

_clap = _aes = None


def clap():
    global _clap
    if _clap is None:
        from music.scoring import Clap
        if not (MODELS / "clap" / "pytorch_model.bin").is_file():
            raise HTTPException(503, "CLAP weights are missing: run scripts/download_models.py")
        _clap = Clap(MODELS / "clap")
    return _clap


def aes():
    global _aes
    if _aes is None:
        from music.scoring import Aesthetics
        if not (MODELS / "aesthetics" / "model.safetensors").is_file():
            raise HTTPException(503, "Audiobox Aesthetics weights are missing: run scripts/download_models.py")
        _aes = Aesthetics(MODELS / "aesthetics")
    return _aes


def _release_cache() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- settings

# The built-in defaults. The user's own, saved from the page, live in config.local.json
# (gitignored) and override them for the page and for every request that leaves a key out.
REPO_DEFAULTS = {
    "engine": "medium",
    "n": 4,                   # best-of-N
    "seed": None,             # None: a random start seed; takes use seed, seed+1, ...
    "steps": None,            # None: the engine's default (8)
    "duration": 30.0,         # seconds
    "raw": False,             # True: send the text as written, no tags
    "check": True,            # CLAP + Audiobox Aesthetics scores
    "lufs": -14.0,            # integrated loudness of every delivered track
    "peak_dbtp": -1.0,        # true-peak ceiling
    "fade_in": 0.0,
    "fade_out": None,         # None: auto (a 3 s fade when a take stops at full level); 0: never
    "strength": 0.5,          # vary: 0 = a copy, 1 = unrelated
}


class Settings(BaseModel):
    """What can be saved. Omitted keys fall back to the repo defaults; seed null = random."""
    model_config = {"extra": "forbid"}
    engine: EngineName | None = None
    n: int | None = Field(None, ge=1, le=8)
    seed: int | None = Field(None, ge=0, lt=2**31)
    steps: int | None = Field(None, ge=1, le=100)
    duration: float | None = Field(None, ge=2, le=380)
    raw: bool | None = None
    check: bool | None = None
    lufs: float | None = Field(None, ge=-30, le=-6)
    peak_dbtp: float | None = Field(None, ge=-12, le=0)
    fade_in: float | None = Field(None, ge=0, le=30)
    fade_out: float | None = Field(None, ge=0, le=30)
    strength: float | None = Field(None, ge=0.05, le=1.0)


def _keep(data: dict) -> dict:
    """Drop empty values: only `seed` may be null (it means "random"). A cleared field on the
    page arrives as null and must not override the defaults with nothing."""
    return {k: v for k, v in data.items() if v is not None or k == "seed"}


def local_settings() -> dict:
    """The user's saved settings, validated; {} if none (or the file is unreadable)."""
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return _keep(Settings(**raw).model_dump(exclude_unset=True))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # a hand-edited file with a typo: say so, use the repo defaults
        print(f"[local-music] ignoring {SETTINGS_FILE.name}: {exc}", flush=True)
        return {}


def effective_settings(saved: dict | None = None) -> dict:
    return {**REPO_DEFAULTS, **(local_settings() if saved is None else saved)}


def _resolve(req: BaseModel, saved: dict | None = None) -> dict:
    """Every setting: the request's own value where it gave one, the saved/repo default otherwise.

    An explicit null counts as "not given", except for the seed, where null means random
    (overriding a saved seed).
    """
    eff = effective_settings(saved)
    out = {}
    for k in eff:
        v = getattr(req, k, None)
        given = k in req.model_fields_set and (v is not None or k == "seed")
        out[k] = v if given else eff[k]
    return out


# --------------------------------------------------------------------------- requests

class MasterFields(BaseModel):
    """Mastering knobs shared by every request and by re-mastering. None = default / keep."""
    length: float | None = Field(None, gt=0, le=4000)       # cut the delivered track to this many seconds
    fade_in: float | None = Field(None, ge=0, le=30)
    fade_out: float | None = Field(None, ge=0, le=30)
    lufs: float | None = Field(None, ge=-30, le=-6)
    peak_dbtp: float | None = Field(None, ge=-12, le=0)
    loop_bars: int | None = Field(None, ge=1, le=64)        # deliver a seamless loop of this many bars
    beats_per_bar: int | None = Field(None, ge=1, le=12)


class ShapeFields(MasterFields):
    """What every generating request shares: how many takes, which seeds, which engine."""
    n: int | None = Field(None, ge=1, le=8)
    seed: int | None = Field(None, ge=0, lt=2**31)  # omitted: the saved default (random unless saved); null: random
    seeds: list[int] | None = Field(None, max_length=8)
    engine: EngineName | None = None
    steps: int | None = Field(None, ge=1, le=100)
    check: bool | None = None

    @field_validator("seeds")
    @classmethod
    def _seed_range(cls, v):
        if v is not None and (not v or any(not 0 <= s < 2**31 for s in v) or len(set(v)) != len(v)):
            raise ValueError("seeds must be distinct numbers from 0 to 2^31-1")
        return v


def _clean_text(v: str | None) -> str | None:
    if v is None:
        return v
    v = " ".join(v.split())
    if not v:
        raise ValueError("text is empty")
    return v


def _check_key(v: str | None) -> str | None:
    if v is not None:
        pc, mode = analysis.parse_key(v)
        return analysis.key_name(pc, mode)
    return v


class GenReq(ShapeFields):
    text: str = Field(min_length=1, max_length=1000)
    duration: float | None = Field(None, ge=2, le=380)
    bpm: float | None = Field(None, ge=30, le=300)
    key: str | None = Field(None, max_length=12)
    kind: Literal["music", "stem"] = "music"
    raw: bool | None = None
    negative: str | None = Field(None, max_length=500)
    name: str | None = Field(None, max_length=60, pattern=r"^[A-Za-z0-9-]*$")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        return _clean_text(v)

    @field_validator("key")
    @classmethod
    def _known_key(cls, v: str | None) -> str | None:
        return _check_key(v)


class FromTakeReq(ShapeFields):
    take: str                                              # "<request id>:<seed>"
    text: str | None = Field(None, max_length=1000)        # default: the source's
    raw: bool | None = None

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str | None) -> str | None:
        return _clean_text(v)


class ExtendReq(FromTakeReq):
    seconds: float = Field(30.0, ge=2, le=360)             # how much new music


class VaryReq(FromTakeReq):
    strength: float | None = Field(None, ge=0.05, le=1.0)  # 0 = a copy, 1 = unrelated


class InpaintReq(FromTakeReq):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def req_dir(rid: str) -> Path:
    if not REQ_ID.fullmatch(rid):
        raise HTTPException(400, f"bad request id {rid!r}")
    return OUTPUT / rid


def load_request(rid: str) -> dict:
    p = req_dir(rid) / "request.json"
    if not p.is_file():
        raise HTTPException(404, f"no request {rid}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_request(rec: dict) -> None:
    p = req_dir(rec["id"]) / "request.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    audio.replace(tmp, p)


def parse_take(ref: str) -> tuple[str, int]:
    m = TAKE_REF.fullmatch(ref.strip())
    if not m:
        raise HTTPException(400, f"a take is '<request id>:<seed>', e.g. 20261010-120000-ab12:1234; got {ref!r}")
    return m["rid"], int(m["seed"])


def find_take(rec: dict, seed: int) -> dict:
    for t in rec["takes"]:
        if t["seed"] == seed:
            return t
    raise HTTPException(404, f"request {rec['id']} has no take with seed {seed}")


def master_opts(base: dict, req: MasterFields | None, saved: dict | None = None) -> post.PostOpts:
    """Mastering options: the request's values over `base` (a stored request's, or the defaults).

    For re-mastering, an explicit null clears length / loop_bars (back to the whole track)."""
    o = dict(base)
    if req is not None:
        for k, v in req.model_dump(exclude_unset=True).items():
            key = "duration" if k == "length" else k
            if key in o and (v is not None or k in ("length", "loop_bars")):
                o[key] = v
    try:
        return post.PostOpts(**o)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


def default_master(opt: dict, bpm: float | None) -> dict:
    return post.PostOpts(fade_in=opt["fade_in"], fade_out=opt["fade_out"], lufs=opt["lufs"],
                         peak_dbtp=opt["peak_dbtp"], bpm=bpm).asdict()


def _info(rec: dict, take: dict) -> dict:
    aes_ = take.get("aes") or {}
    scores = (f"CLAP {take['clap']:.3f}" if take.get("clap") is not None else "") + \
             "".join(f" {k} {aes_[k]:.1f}" for k in ("CE", "PQ") if k in aes_)
    return {"title": rec["name"], "genre": rec.get("kind", "music"),
            "comment": f"{rec['prompt']} | seed {take['seed']} | {rec['model']} | {scores.strip()} | "
                       f"rank {take.get('rank', '?')}/{len(rec['takes'])} | request {rec['id']}",
            "software": f"local-music {__version__}", "date": rec["created"][:10],
            "keywords": f"{rec.get('op', 'generate')}, {rec['engine']}, seed {take['seed']}"}


def analyse(rec: dict, take: dict, x: np.ndarray, sr: int) -> None:
    """Signal and musical checks on the raw take; stores the beat grid for loops and plots."""
    take["signal"] = checks.signal_report(x, sr, rec.get("duration"))
    mus = checks.musical_report(x, sr, rec.get("bpm"), rec.get("key"))
    take["beats"] = mus.pop("beat_times")
    take["music"] = mus


def master_take(rec: dict, take: dict, opts: post.PostOpts) -> tuple[np.ndarray, dict]:
    """Raw -> mastered audio and its report, for one take. No side effects (so a request's takes
    can all be mastered before anything is written, and a failure leaves them as they were);
    thread-safe (numpy, soxr)."""
    raw, sr = audio.read(req_dir(rec["id"]) / take["raw"])
    return post.process(raw, sr, opts, take.get("beats"))


def commit_take(rec: dict, take: dict, y: np.ndarray, rep: dict) -> None:
    """Store a mastered take: its report in the record, its MP3 on disk (the WAV is written by
    judge(), with the scores in its INFO chunk). Thread-safe (libsndfile)."""
    rep = dict(rep)
    take["final"] = {"true_peak_dbtp": rep.pop("true_peak_dbtp"), "lufs": rep.pop("lufs")}
    take["post"] = rep
    audio.write_mp3(req_dir(rec["id"]) / take["mp3"], y)


def draw(rec: dict, take: dict, opts: post.PostOpts, y: np.ndarray) -> None:
    loop = opts.loop_bars is not None
    shift = take["post"].get("trim_start_s", 0.0)
    beats = None if loop else [b - shift for b in take.get("beats", []) if b >= shift]
    tempo = take.get("music", {}).get("tempo", {})
    key = take.get("music", {}).get("key", {})
    title = (f"{rec['name']}  seed {take['seed']}  {rec['engine']}  |  "
             f"{tempo.get('bpm') or '?'} BPM, {key.get('key') or '?'}  |  {rec['prompt'][:100]}")
    plot.draw(y, audio.SR, req_dir(rec["id"]) / take["png"], title=title, beats=beats,
              beats_per_bar=opts.beats_per_bar, seam=loop)


def render_all(rec: dict, opts: post.PostOpts, prepare=None) -> list[np.ndarray]:
    """Master every take of a request in parallel and only then write anything, so one take
    failing to master (a loop longer than it, say) leaves every take as it was; then the MP3s
    and the plots. (A file locked while they're written, the browser playing it, still stops
    part-way: a 409, some takes rewritten; running it again finishes the job.) prepare(take),
    if given, runs first in the take's thread (a new take's analysis)."""
    def one(t):
        if prepare is not None:
            prepare(t)
        return master_take(rec, t, opts)
    n = min(4, len(rec["takes"])) or 1
    with ThreadPoolExecutor(max_workers=n) as ex:
        done = list(ex.map(one, rec["takes"]))  # raises before anything is written
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(lambda tr: commit_take(rec, tr[0], *tr[1]), zip(rec["takes"], done)))
    finals = [y for y, _ in done]
    for t, y in zip(rec["takes"], finals):
        draw(rec, t, opts, y)
    return finals


def judge(rec: dict, takes: list[dict], finals: list[np.ndarray], check: bool) -> None:
    """Scores, fails and warnings for each take; then rank and write the WAVs with their INFO chunk."""
    from music.scoring import PQ_FAIL, PQ_WARN, rank_score
    if check:
        try:
            # An inpainted take is mostly its source: judge it against the source's description.
            # Nothing to match without one (an imported track nobody described).
            text = prompts.describe(rec.get("clap_prompt") or rec["prompt"]) if (rec.get("clap_prompt") or
                                                                                rec["prompt"]) else ""
            cl = clap().score(finals, audio.SR, text) if text else [{} for _ in finals]
            ae = aes().score(finals, audio.SR)
            for t, c, a in zip(takes, cl, ae):
                t.update(c)
                t["aes"] = a
        except HTTPException:
            raise
        except Exception as exc:  # the takes exist either way: say the scoring failed
            rec.setdefault("warnings", []).append(f"scoring failed: {type(exc).__name__}: {exc}")
    for t in takes:
        t["fails"] = list(t["signal"]["fails"])
        t["warnings"] = list(t["signal"]["warnings"]) + list(t.get("music", {}).get("warnings", []))
        pq = (t.get("aes") or {}).get("PQ")
        if pq is not None and pq < PQ_FAIL:
            t["fails"].append(f"low production quality (Audiobox PQ {pq:.1f} < {PQ_FAIL})")
        elif pq is not None and pq < PQ_WARN:
            t["warnings"].append(f"so-so production quality (Audiobox PQ {pq:.1f})")
        lp = t.get("post", {})
        if t["signal"].get("abrupt_end") and "loop" not in lp and not lp.get("fade_out_s"):
            t["warnings"].append("ends abruptly at full level: fade it out (fade_out) or extend it")
        if "seam_ok" in lp and not lp["seam_ok"]:
            t["warnings"].append(f"the loop's start is {lp['seam_level_db']:.0f} dB off what would follow its end: "
                                 "the repeat may be heard")
        if lp.get("loop", {}).get("seam_similarity", 1.0) < 0.85:
            t["warnings"].append(f"the music changes across the loop point (similarity "
                                 f"{lp['loop']['seam_similarity']:.2f}): the repeat may be noticeable")
        if lp.get("limiter_db", 0) > 6:
            t["warnings"].append(f"limited hard ({lp['limiter_db']:.1f} dB) to reach {rec['post']['lufs']:g} LUFS: "
                                 "a lower lufs target keeps more punch")
        t["rejected"] = bool(t["fails"])
        t["rank_score"] = rank_score(t)
    order = sorted(range(len(takes)), key=lambda i: (-takes[i]["rank_score"], takes[i]["seed"]))
    ranked = [takes[i] for i in order]
    for r, t in enumerate(ranked, 1):
        t["rank"] = r
    rec["takes"] = ranked
    d = req_dir(rec["id"])
    for t, y in zip(takes, finals):
        audio.write_wav(d / t["wav"], y, info=_info(rec, t))


def take_view(rec: dict, take: dict) -> dict:
    """A take as the API returns it: its ref, file names and paths, and every score."""
    d = req_dir(rec["id"])
    out = {k: v for k, v in take.items() if k not in ("raw", "wav", "mp3", "png", "beats")}
    out["ref"] = f"{rec['id']}:{take['seed']}"
    out["files"] = {k: f"{rec['id']}/{take[k]}" for k in ("wav", "mp3", "png", "raw")}
    out["paths"] = {k: str(d / take[k]) for k in ("wav", "mp3", "png")}
    return out


def request_view(rec: dict) -> dict:
    keep = ("id", "created", "op", "kind", "text", "prompt", "negative", "raw", "bpm", "key", "name", "engine",
            "model", "seeds", "steps", "duration", "source", "seconds", "strength", "span", "parts", "bridge",
            "blend", "layers", "post", "gen_length_s", "gen_s", "score_s", "warnings")
    out = {k: rec[k] for k in keep if k in rec}
    out["takes"] = [take_view(rec, t) for t in rec["takes"]]
    out["kept"] = sum(not t.get("rejected") for t in rec["takes"])
    out["ignored"] = rec.get("ignored", [])
    return out


def _seeds(req: ShapeFields, opt: dict) -> list[int]:
    if req.seeds:
        return list(req.seeds)
    start = opt["seed"] if opt["seed"] is not None else secrets.randbelow(2**31 - 64)
    return [(start + k) % 2**31 for k in range(opt["n"])]


def _start(engine: str) -> Worker:
    try:
        return worker(engine)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"engine {engine} failed to start: {exc} (see logs/server.log)") from exc


def _call(rec: dict, job: dict, w: Worker) -> dict:
    """Send one job to the engine. On failure the request's folder goes and it's a 500."""
    try:
        return w.call(job)
    except Exception as exc:
        shutil.rmtree(req_dir(rec["id"]), ignore_errors=True)
        print(f"[local-music] generation FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(500, f"generation failed: {type(exc).__name__}: {exc}") from exc


def _run(rec: dict, job: dict, w: Worker, opt: dict, stitch=None) -> dict:
    """Send the job, turn the worker's files into checked, mastered, ranked takes. Call with _lock held.

    stitch(raw) -> raw: for extensions longer than the engine's window, puts the earlier part
    of the source back in front of what the engine returned.
    """
    t0 = time.time()
    reply = _call(rec, job, w)
    rec["gen_s"] = round(time.time() - t0, 2)
    return _finish(rec, reply, opt, stitch)


def _finish(rec: dict, reply: dict, opt: dict, stitch=None) -> dict:
    """The engine's files ({"files": [{"seed", "path"}]}) -> checked, mastered, ranked takes,
    saved; the request as the API returns it. Call with _lock held."""
    d = req_dir(rec["id"])
    try:
        t1 = time.time()
        opts = post.PostOpts(**rec["post"])
        for f in reply["files"]:
            seed = int(f["seed"])
            raw_name = f"{seed}_raw.wav"
            x, sr = audio.read(f["path"])
            if stitch is not None:
                x = stitch(x, sr)
            audio.write_wav(d / raw_name, x, sr, subtype="FLOAT")
            Path(f["path"]).unlink(missing_ok=True)
            rec["takes"].append({"seed": seed, "raw": raw_name, "wav": f"{seed}.wav", "mp3": f"{seed}.mp3",
                                 "png": f"{seed}.png"})

        def prepare(take: dict) -> None:
            analyse(rec, take, *audio.read(d / take["raw"]))
        finals = render_all(rec, opts, prepare)
        judge(rec, rec["takes"], finals, bool(opt["check"]))
        rec["score_s"] = round(time.time() - t1, 2)
        save_request(rec)
        _release_cache()
    except Exception as exc:  # don't leave a half-made request folder behind
        shutil.rmtree(d, ignore_errors=True)
        if isinstance(exc, HTTPException):
            raise
        status = 400 if isinstance(exc, ValueError) else 500
        print(f"[local-music] checks FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(status, f"mastering or checks failed: {type(exc).__name__}: {exc}") from exc
    kept = sum(not t["rejected"] for t in rec["takes"])
    top = rec["takes"][0]
    print(f"[local-music] {time.strftime('%H:%M:%S')} {rec['id']} {rec['op']} {rec['engine']}: "
          f"{len(rec['takes'])} take(s) of {rec['name']} ({rec['gen_length_s']:.0f} s) in {rec['gen_s']:.1f}s "
          f"+ {rec['score_s']:.1f}s checks, {kept} kept; best seed {top['seed']} (score {top['rank_score']})",
          flush=True)
    return request_view(rec)


def _new_record(op: str, opt: dict, info: dict, engine: str, seeds: list[int], **fields) -> dict:
    rid = new_id()
    (OUTPUT / rid).mkdir(parents=True)
    return {"id": rid, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "op": op, "engine": engine,
            "model": info["model"], "seeds": seeds, "steps": opt["steps"] or info["default_steps"], "takes": [],
            **fields}


@app.post("/api/generate")
def generate(req: GenReq):
    saved = local_settings()  # read once: a save mid-request can't mix old and new
    opt = _resolve(req, saved)
    duration = float(req.duration or opt["duration"])
    prompt = prompts.build_prompt(req.text, req.bpm, req.key, req.kind, raw=bool(opt["raw"]))
    seeds = _seeds(req, opt)
    po = master_opts(default_master(opt, req.bpm), req)
    length = duration
    if po.loop_bars and req.bpm:
        # Room for the loop after the opening bars, so it can start where the music has settled.
        need = po.loop_bars * po.beats_per_bar * 60.0 / req.bpm + 0.5
        length = max(length, need + 4.0)
    with _lock:
        w = _start(opt["engine"])
        info = w.info
        if length > info["max_duration"]:
            raise HTTPException(400, f"{opt['engine']} makes at most {info['max_duration']:.0f} s"
                                + (f" (a {po.loop_bars}-bar loop at {req.bpm:g} BPM needs ~{length:.0f} s)"
                                   if po.loop_bars else "") + "; extend a take for longer music")
        rec = _new_record("generate", opt, info, opt["engine"], seeds, kind=req.kind, text=req.text, prompt=prompt,
                          negative=req.negative, raw=bool(opt["raw"]), bpm=req.bpm, key=req.key,
                          name=req.name or prompts.track_name(req.text), duration=duration,
                          post=po.asdict(), gen_length_s=round(length, 2),
                          ignored=[k for k in info.get("ignores", []) if k == "negative" and req.negative])
        job = {"op": "generate", "prompt": prompt, "negative": req.negative, "duration": length, "seeds": seeds,
               "steps": rec["steps"], "out_dir": str(req_dir(rec["id"])), "stem": "gen"}
        return _run(rec, job, w, opt)


def _source(ref: str) -> tuple[dict, dict, Path]:
    rid, seed = parse_take(ref)
    src = load_request(rid)
    take = find_take(src, seed)
    return src, take, req_dir(rid) / take["raw"]


def _raw_offset(src: dict, take: dict, ref: str) -> float:
    """Where the delivered track (what the player and the plot show) starts on the raw take's
    timeline: spans are given in the former and cut from the latter. A loop has no single
    mapping, so spans on one are refused."""
    if take.get("post", {}).get("loop") is not None:
        raise HTTPException(400, f"{ref} is delivered as a loop: give a span only on a whole track "
                                 "(re-master it with loop_bars null first)")
    return float(take.get("post", {}).get("trim_start_s", 0.0))


def _from_take(req: FromTakeReq, op: str, **fields):
    """The shared part of extend / vary / inpaint: resolve settings, the source and its prompt."""
    saved = local_settings()
    opt = _resolve(req, saved)
    if "engine" not in req.model_fields_set or req.engine is None:
        opt["engine"] = None  # filled from the source below
    src, take, raw_path = _source(req.take)
    text = req.text or src["text"]
    raw = bool(opt["raw"]) if req.raw is not None else bool(src.get("raw"))
    prompt = prompts.build_prompt(text, src.get("bpm"), src.get("key"), src.get("kind", "music"), raw=raw) \
        if req.text else src["prompt"]
    base = {**src["post"]}
    base.pop("duration", None)  # a cut length belonged to the source, not to what's made from it
    base["loop_bars"] = None
    po = master_opts(base, req)
    common = dict(kind=src.get("kind", "music"), text=text, prompt=prompt, raw=raw, bpm=src.get("bpm"),
                  key=src.get("key"), name=src["name"], source=req.take, post=po.asdict(), **fields)
    return opt, src, raw_path, common


@app.post("/api/extend")
def extend(req: ExtendReq):
    """Continue a take: the engine hears it (its last part, past the engine's 380 s window) and
    writes what comes next. The result is the whole track: the source exactly as it was, then
    the new music, crossfaded over 0.1 s where they meet."""
    opt, src, raw_path, common = _from_take(req, "extend")
    x, sr = audio.read(raw_path)
    # Continue from the music's body: given a fade-out or silence to continue, the engine
    # continues it (a silent extension), and the gap would stay inside the result.
    end = analysis.body(x, sr)[1]
    trimmed = x.shape[1] - end
    if trimmed > int(0.5 * sr):
        x = x[:, :end]
    else:
        trimmed = 0
    src_s = x.shape[1] / sr
    with _lock:
        engine = opt["engine"] or src["engine"]
        w = _start(engine)
        info = w.info
        if req.seconds > info["max_duration"] - 10:
            raise HTTPException(400, f"extend by at most {info['max_duration'] - 10:.0f} s at a time on {engine}")
        context_s = min(src_s, info["max_duration"] - req.seconds)
        seeds = _seeds(req, opt)
        rec = _new_record("extend", opt, info, engine, seeds, seconds=req.seconds,
                          duration=round(src_s + req.seconds, 2), gen_length_s=round(context_s + req.seconds, 2),
                          **common)
        if trimmed:
            rec["warnings"] = [f"the source's last {trimmed / sr:.1f} s (a fade-out or silence) were left out: "
                               "the new music continues from where the source was still playing"]
        d = req_dir(rec["id"])
        # The engine hears the last context_s of the source (all of it, if it fits) and writes
        # what follows. It re-renders what it heard, too: the source is put back in front.
        cut_n = x.shape[1] - int(round(context_s * sr))
        head, ctx = x[:, :cut_n], x[:, cut_n:]
        ctx_path = raw_path
        if cut_n > 0:
            ctx_path = d / "context.wav"
            audio.write_wav(ctx_path, ctx, sr, subtype="FLOAT")
        xf = int(post.KEEP_XFADE_S * sr)

        def stitch(y: np.ndarray, ysr: int, head=head, ctx=ctx, xf=xf) -> np.ndarray:
            n = ctx.shape[1] + int(round(req.seconds * ysr))
            return np.concatenate([head, post.keep_outside(ctx, _fit(y, n), ctx.shape[1], n, xf)], axis=1)
        job = {"op": "generate", "prompt": rec["prompt"], "duration": context_s + req.seconds, "seeds": seeds,
               "steps": rec["steps"], "out_dir": str(d), "stem": "gen", "inpaint_audio": str(ctx_path),
               "mask_start": context_s, "mask_end": context_s + req.seconds}
        out = _run(rec, job, w, opt, stitch)
        (d / "context.wav").unlink(missing_ok=True)
        return out


@app.post("/api/vary")
def vary(req: VaryReq):
    """A variation of a take: the engine starts from it instead of from noise. strength 0.1 is
    a close variation; 0.5 keeps the timbre and style but rewrites melody and rhythm."""
    opt, src, raw_path, common = _from_take(req, "vary")
    strength = req.strength if req.strength is not None else opt["strength"]
    x, sr = audio.read(raw_path)
    with _lock:
        engine = opt["engine"] or src["engine"]
        w = _start(engine)
        seeds = _seeds(req, opt)
        length = min(x.shape[1] / sr, w.info["max_duration"])
        rec = _new_record("vary", opt, w.info, engine, seeds, strength=strength, duration=round(length, 2),
                          gen_length_s=round(length, 2), **common)
        init = raw_path
        if x.shape[1] / sr > length:  # longer than the engine's window: vary its first part
            init = req_dir(rec["id"]) / "init.wav"
            audio.write_wav(init, x[:, :int(length * sr)], sr, subtype="FLOAT")
            rec["warnings"] = [f"the source is {x.shape[1] / sr:.0f} s: only its first {length:.0f} s (the engine's "
                               "window) were varied"]
        job = {"op": "generate", "prompt": rec["prompt"], "duration": length, "seeds": seeds, "steps": rec["steps"],
               "out_dir": str(req_dir(rec["id"])), "stem": "gen", "init_audio": str(init),
               "init_strength": strength}
        out = _run(rec, job, w, opt)
        (req_dir(rec["id"]) / "init.wav").unlink(missing_ok=True)
        return out


@app.post("/api/inpaint")
def inpaint(req: InpaintReq):
    """Regenerate start..end (seconds, as the take plays) and keep the rest as it was."""
    opt, src, raw_path, common = _from_take(req, "inpaint")
    x, sr = audio.read(raw_path)
    total = x.shape[1] / sr
    off = _raw_offset(src, find_take(src, parse_take(req.take)[1]), req.take)
    start, end = req.start + off, req.end + off
    if not start < end <= total + 1e-6:
        raise HTTPException(400, f"the span must lie within the take (0..{total - off:.1f} s), start before end")
    with _lock:
        engine = opt["engine"] or src["engine"]
        w = _start(engine)
        if total > w.info["max_duration"]:
            raise HTTPException(400, f"{engine} inpaints tracks of up to {w.info['max_duration']:.0f} s")
        seeds = _seeds(req, opt)
        rec = _new_record("inpaint", opt, w.info, engine, seeds, span=[req.start, req.end], duration=round(total, 2),
                          gen_length_s=round(total, 2), clap_prompt=src["prompt"], **common)
        job = {"op": "generate", "prompt": rec["prompt"], "duration": total, "seeds": seeds, "steps": rec["steps"],
               "out_dir": str(req_dir(rec["id"])), "stem": "gen", "inpaint_audio": str(raw_path),
               "mask_start": start, "mask_end": end}
        m0, m1, xf = int(round(start * sr)), int(round(end * sr)), int(post.KEEP_XFADE_S * sr)

        def stitch(y: np.ndarray, ysr: int) -> np.ndarray:
            # Only the span is the engine's: everything else is the take as it was.
            return post.keep_outside(x, _fit(y, x.shape[1]), m0, m1, xf)
        return _run(rec, job, w, opt, stitch)


# --------------------------------------------------------------------------- your own audio

MAX_IMPORT_BYTES = 300 * 2**20
MAX_IMPORT_S = 1200.0


@app.post("/api/import")
async def import_audio(request: Request, filename: str = Query(min_length=1, max_length=200),
                       text: str | None = Query(None, max_length=1000), kind: Literal["music", "stem"] = "music",
                       name: str | None = Query(None, max_length=60, pattern=r"^[A-Za-z0-9-]*$"),
                       check: bool | None = None):
    """Your own audio file (the request body, as raw bytes: WAV, FLAC, OGG or MP3) becomes a
    take (seed 0) of an "import" request: checked, measured and mastered like any other, so
    extend, vary, inpaint, arrange and master all work from its ref. `text` describes it
    (it becomes the prompt that work continues from; without it the engine is led by the
    audio alone)."""
    size = int(request.headers.get("content-length") or 0)
    if size > MAX_IMPORT_BYTES:
        raise HTTPException(413, f"the file is over {MAX_IMPORT_BYTES // 2**20} MB")
    data = bytearray()
    async for chunk in request.stream():  # counted as it arrives: a chunked upload has no length
        data += chunk
        if len(data) > MAX_IMPORT_BYTES:
            raise HTTPException(413, f"the file is over {MAX_IMPORT_BYTES // 2**20} MB")
    if not data:
        raise HTTPException(400, "no audio in the request body")
    data = bytes(data)
    return await run_in_threadpool(_import, data, filename, text, kind, name, check)


def _import(data: bytes, filename: str, text: str | None, kind: str, name: str | None, check: bool | None) -> dict:
    import soundfile as sf
    base = Path(filename.replace("\\", "/")).name or "audio"
    warnings = []
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            sr = f.samplerate
            keep = int(MAX_IMPORT_S * sr)
            if f.frames > keep:  # decode only what's kept: a long FLAC is gigabytes as float
                warnings.append(f"only the first {MAX_IMPORT_S / 60:.0f} minutes were imported")
            # In blocks, two channels kept: an 8-channel 384 kHz file mustn't decode whole first.
            blocks, got = [], 0
            want = min(f.frames, keep) if f.frames > 0 else keep
            while got < want:
                blk = f.read(min(10 * sr, want - got), dtype="float32", always_2d=True)
                if not len(blk):
                    break
                blocks.append(blk[:, :2].copy())
                got += len(blk)
            x = np.concatenate(blocks) if blocks else np.zeros((0, 1), np.float32)
    except Exception as exc:
        raise HTTPException(400, f"can't read {base!r} as audio ({exc}). WAV, FLAC, OGG and MP3 work; "
                                 "convert other formats (M4A, AAC...) first") from exc
    if not np.all(np.isfinite(x)):
        raise HTTPException(400, f"{base!r} has samples that aren't numbers (NaN or infinity): it's damaged")
    x = np.ascontiguousarray(x.T)
    x = np.vstack([x, x]) if x.shape[0] == 1 else x[:2]
    if x.shape[1] < sr:
        raise HTTPException(400, f"{base!r} is shorter than a second")
    x = audio.resample(x, sr, RAW_SR)
    text = " ".join(text.split()) if text else ""
    opt = effective_settings()
    if check is not None:
        opt["check"] = check
    prompt = prompts.build_prompt(text, kind=kind) if text else ""
    with _lock:
        rid = new_id()
        d = OUTPUT / rid
        d.mkdir(parents=True)
        rec = {"id": rid, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "op": "import", "engine": opt["engine"],
               "model": "imported", "seeds": [0], "steps": None, "takes": [], "kind": kind, "text": text,
               "prompt": prompt, "raw": False, "bpm": None, "key": None,
               "name": name or prompts.file_name(Path(base).stem), "source": base,
               "duration": round(x.shape[1] / RAW_SR, 2), "gen_length_s": round(x.shape[1] / RAW_SR, 2), "gen_s": 0.0,
               "post": default_master(opt, None), "warnings": warnings}
        path = d / "import.wav"
        audio.write_wav(path, x, RAW_SR, subtype="FLOAT")
        return _finish(rec, {"files": [{"seed": 0, "path": str(path)}]}, opt)


# --------------------------------------------------------------------------- arranging takes

ARRANGE_CONTEXT_S = 20.0  # how much of each part the engine hears either side of a join


class PartItem(BaseModel):
    take: str
    start: float | None = Field(None, ge=0)  # seconds on the take's raw timeline; default its start
    end: float | None = Field(None, gt=0)    # default its end


class ArrangeReq(ShapeFields):
    parts: list[PartItem] = Field(min_length=2, max_length=12)
    bridge: float | None = Field(None, ge=0, le=30)  # new music between parts; default 2 bars (4 s without a tempo)
    blend: float = Field(1.0, ge=0.25, le=6)         # seconds of each part regenerated beside the bridge
    text: str | None = Field(None, max_length=1000)  # what the bridges should be; default the first part's prompt
    name: str | None = Field(None, max_length=60, pattern=r"^[A-Za-z0-9-]*$")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str | None) -> str | None:
        return _clean_text(v)


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    """y cut or zero-padded to n samples (the engine rounds lengths to whole samples its own way)."""
    return y[:, :n] if y.shape[1] >= n else np.pad(y, ((0, 0), (0, n - y.shape[1])))


def _arrangement_warnings(parts: list[dict]) -> list[str]:
    """Neighbouring parts at different tempos or in unrelated keys: the bridge has to change them."""
    out = []
    for p, q in zip(parts, parts[1:]):
        tp, tq = (z["take"].get("music", {}).get("tempo", {}) for z in (p, q))
        if tp.get("bpm") and tq.get("bpm") and min(tp.get("confidence", 0), tq.get("confidence", 0)) \
                >= checks.TEMPO_CONFIDENT and analysis.tempo_matches(tq["bpm"], tp["bpm"], tol=0.05) is None:
            out.append(f"{p['ref']} is at {tp['bpm']:.0f} BPM and {q['ref']} at {tq['bpm']:.0f}: the bridge has to "
                       "change tempo, which may sound abrupt")
        kp, kq = (z["take"].get("music", {}).get("key", {}) for z in (p, q))
        if kp.get("key") and kq.get("key") and min(kp.get("strength", 0), kq.get("strength", 0)) >= checks.KEY_CONFIDENT:
            pp, pm = analysis.parse_key(kp["key"])
            qp, qm = analysis.parse_key(kq["key"])
            if analysis.key_relation(qp, qm, pp, pm) is None:
                out.append(f"{p['ref']} is in {kp['key']} and {q['ref']} in {kq['key']}: the bridge has to modulate")
    return out


@app.post("/api/arrange")
def arrange(req: ArrangeReq):
    """Several takes (or spans of them, in seconds as they play), in order, as one new song: at
    each join the engine hears up to 20 s of the part before and after, and writes `bridge`
    seconds of new music between them, rewriting `blend` seconds either side so the parts flow
    into it. Everything else is the parts' own audio. Each take of the result is the whole song,
    its bridges made with one seed."""
    saved = local_settings()
    opt = _resolve(req, saved)
    parts = []
    last = len(req.parts) - 1
    for i, item in enumerate(req.parts):
        src, take, raw_path = _source(item.take)
        x, sr = audio.read(raw_path)
        total = x.shape[1] / sr
        # At a join the engine must hear music, not a fade-out and silence (given those, it
        # writes a silent bridge): an inner edge starts or ends with the part's body, unless the
        # span sets it. The song's very start and end stay as they are.
        tp = take.get("post", {})
        looped = tp.get("loop") is not None
        off = 0.0 if looped else float(tp.get("trim_start_s", 0.0))
        if item.start is not None or item.end is not None:
            off = _raw_offset(src, take, item.take)
        # What the take delivers (a loop: its whole raw), on the raw timeline.
        lo = 0.0 if looped else off
        hi = total if looped else (off + tp["cut_at_s"] if tp.get("cut_at_s") else total - tp.get("trim_end_s", 0.0))
        b0, b1 = analysis.body(x[:, int(lo * sr):int(hi * sr)], sr)
        a = item.start + off if item.start is not None else (lo + b0 / sr if i > 0 else lo)
        b = min(item.end + off, total) if item.end is not None else (lo + b1 / sr if i < last else hi)
        if not 0 <= a < b or b - a < 4.0:
            raise HTTPException(400, f"part {item.take}: the span must be at least 4 s inside the take "
                                     f"(0..{hi - lo:.1f} s as it plays)")
        # Each part at its delivered loudness (its mastering gain; the song is mastered again as a
        # whole): a quiet recording joined to a generated take would otherwise be 30 dB under it.
        g = 10 ** (float(tp.get("gain_db", 0.0)) / 20)
        parts.append({"src": src, "take": take, "ref": item.take, "x": x[:, int(a * sr):int(b * sr)] * g,
                      "start": round(max(0.0, a - off), 2), "end": round(b - off, 2),
                      "dropped_s": round((b0 / sr if i > 0 and item.start is None else 0.0)
                                         + ((hi - lo) - b1 / sr if i < last and item.end is None else 0.0), 1)})
    first = parts[0]["src"]
    tempo = parts[0]["take"].get("music", {}).get("tempo", {})
    bpm = first.get("bpm") or (tempo.get("bpm") if tempo.get("confidence", 0) >= checks.TEMPO_CONFIDENT else None)
    bridge = req.bridge if req.bridge is not None else (8 * 60.0 / bpm if bpm else 4.0)
    prompt = (prompts.build_prompt(req.text, first.get("bpm"), first.get("key"), first.get("kind", "music"),
                                   raw=bool(first.get("raw"))) if req.text else first["prompt"])
    with _lock:
        engine = req.engine if "engine" in req.model_fields_set and req.engine else first["engine"]
        w = _start(engine)
        seeds = _seeds(req, opt)
        base = {**first["post"]}
        base.pop("duration", None)
        base["loop_bars"] = None
        po = master_opts(base, req)
        total_s = sum(p["x"].shape[1] for p in parts) / RAW_SR + bridge * (len(parts) - 1)
        name = req.name or (prompts.track_name(req.text) if req.text else f"{first['name']}-arranged")
        rec = _new_record("arrange", opt, w.info, engine, seeds, kind=first.get("kind", "music"),
                          text=req.text or first.get("text", ""), prompt=prompt, raw=bool(first.get("raw")),
                          bpm=first.get("bpm"), key=first.get("key"), name=name[:60],
                          parts=[{"take": p["ref"], "name": p["src"]["name"], "start": p["start"], "end": p["end"]}
                                 for p in parts],  # the spans as used
                          bridge=round(bridge, 2), blend=req.blend, duration=round(total_s, 2), post=po.asdict(),
                          clap_prompt=first["prompt"] or None,
                          warnings=_arrangement_warnings(parts) + [
                              f"{p['ref']}: {p['dropped_s']:g} s of fade or quiet edge left out at its join(s) "
                              "(give a span to keep it)" for p in parts if p["dropped_s"] >= 3])
        d = req_dir(rec["id"])
        t0 = time.time()
        joins = []
        for j, (p, q) in enumerate(zip(parts, parts[1:])):
            # Each part's two joins share it: neither may hear more than half of it.
            ctx_n = int(min(ARRANGE_CONTEXT_S, p["x"].shape[1] / RAW_SR / 2, q["x"].shape[1] / RAW_SR / 2) * RAW_SR)
            gap = int(bridge * RAW_SR)
            ctx = np.concatenate([p["x"][:, -ctx_n:], np.zeros((2, gap), np.float32), q["x"][:, :ctx_n]], axis=1)
            path = d / f"join{j}.wav"
            audio.write_wav(path, ctx, RAW_SR, subtype="FLOAT")
            ctx_s = ctx_n / RAW_SR
            blend = max(0.0, min(req.blend, ctx_s - 2 * post.KEEP_XFADE_S))
            m0, m1 = ctx_s - blend, ctx_s + bridge + blend
            job = {"op": "generate", "prompt": prompt, "duration": ctx.shape[1] / RAW_SR, "seeds": seeds,
                   "steps": rec["steps"], "out_dir": str(d), "stem": f"join{j}", "inpaint_audio": str(path),
                   "mask_start": m0, "mask_end": m1}
            reply = _call(rec, job, w)
            joins.append({"ctx_n": ctx_n, "n": ctx.shape[1], "ctx": ctx, "m0": int(round(m0 * RAW_SR)),
                          "m1": int(round(m1 * RAW_SR)), "renders": {int(f["seed"]): f["path"] for f in reply["files"]}})
            path.unlink(missing_ok=True)
        rec["gen_s"] = round(time.time() - t0, 2)
        rec["gen_length_s"] = round(sum(jn["n"] for jn in joins) / RAW_SR, 2)
        xf = int(post.KEEP_XFADE_S * RAW_SR)
        files = []
        try:
            for seed in seeds:
                # Each join's rendering, with the parts put back outside its bridge and blend
                # (the engine re-renders the context it heard), starts and ends with the parts'
                # own samples: the song is the parts and the joins laid end to end.
                song = [parts[0]["x"][:, :parts[0]["x"].shape[1] - joins[0]["ctx_n"]]]
                for j, jn in enumerate(joins):
                    r, _ = audio.read(jn["renders"][seed])
                    song.append(post.keep_outside(jn["ctx"], _fit(r, jn["n"]), jn["m0"], jn["m1"], xf))
                    q = parts[j + 1]["x"]
                    end = q.shape[1] - joins[j + 1]["ctx_n"] if j + 1 < len(joins) else q.shape[1]
                    song.append(q[:, jn["ctx_n"]:end])
                path = d / f"gen_{seed}.wav"
                audio.write_wav(path, np.concatenate(song, axis=1), RAW_SR, subtype="FLOAT")
                files.append({"seed": seed, "path": str(path)})
            for jn in joins:
                for f in jn["renders"].values():
                    Path(f).unlink(missing_ok=True)
        except Exception as exc:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(500, f"stitching failed: {type(exc).__name__}: {exc}") from exc
        return _finish(rec, {"files": files}, opt)


# --------------------------------------------------------------------------- mixing takes

class MixLayer(BaseModel):
    take: str
    gain_db: float = Field(0.0, ge=-60, le=12)  # its volume in the mix
    at: float = Field(0.0, ge=0, le=1200)       # where it starts in the mix, seconds
    start: float | None = Field(None, ge=0)     # a span of the take (its delivered timeline)
    end: float | None = Field(None, gt=0)


class MixReq(MasterFields):
    layers: list[MixLayer] = Field(min_length=2, max_length=8)
    name: str | None = Field(None, max_length=60, pattern=r"^[A-Za-z0-9-]*$")
    check: bool | None = None


@app.post("/api/mix")
def mix(req: MixReq):
    """Takes on top of each other, as one new take: each layer is a take's delivered track (what
    you hear: mastered, or its loop) at its own volume (gain_db), starting `at` seconds in.
    Nothing is levelled or ducked automatically. The sum is mastered like any take (so the mix's
    overall loudness is the usual target; the volumes set the balance), and is a take itself
    (seed 0), so it can be extended, looped or re-mastered."""
    opt = effective_settings()
    if req.check is not None:
        opt["check"] = req.check
    layers, srcs, desc = [], [], []
    for item in req.layers:
        src, take, _ = _source(item.take)
        y, sr = audio.read(req_dir(src["id"]) / take["wav"])
        total = y.shape[1] / sr
        a = item.start or 0.0
        b = total if item.end is None else min(item.end, total)
        if not 0 <= a or b - a < 0.1:
            raise HTTPException(400, f"layer {item.take}: the span must lie inside the take (0..{total:.1f} s) "
                                     "and last at least 0.1 s")
        layers.append(post.Layer(audio=y[:, int(a * sr):int(b * sr)], gain_db=item.gain_db, offset_s=item.at))
        srcs.append(src)
        desc.append({"take": item.take, "name": src["name"], "gain_db": item.gain_db, "at": item.at,
                     "start": round(a, 2), "end": round(b, 2)})
    base = srcs[0]  # the first layer's prompt, tempo and key carry on (extending the mix uses them)
    with _lock:
        y, rep = post.mix_layers(layers, audio.SR)
        raw = audio.resample(y, audio.SR, RAW_SR)
        rid = new_id()
        d = OUTPUT / rid
        d.mkdir(parents=True)
        po = master_opts(default_master(opt, base.get("bpm")), req)
        rec = {"id": rid, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "op": "mix", "engine": base["engine"],
               "model": "mix", "seeds": [0], "steps": None, "takes": [], "kind": base.get("kind", "music"),
               "text": base.get("text", ""), "prompt": base["prompt"], "raw": bool(base.get("raw")),
               "bpm": base.get("bpm"), "key": base.get("key"),
               "name": (req.name or f"{base['name']}-mix")[:60], "layers": desc, "duration": rep["length_s"], "gen_length_s": rep["length_s"], "gen_s": 0.0, "post": po.asdict(),
               "warnings": []}
        path = d / "mix.wav"
        audio.write_wav(path, raw, RAW_SR, subtype="FLOAT")
        return _finish(rec, {"files": [{"seed": 0, "path": str(path)}]}, opt)


class MasterReq(MasterFields):
    request: str = Field(pattern=REQ_ID.pattern)
    bpm: float | None = Field(None, ge=30, le=300)  # loops: the tempo to cut at (default: the request's or measured)
    check: bool | None = None


@app.post("/api/master")
def remaster(req: MasterReq):
    """Re-master every take of a request (length, fades, loudness, loop); no regeneration."""
    check = req.check if req.check is not None else effective_settings()["check"]
    with _lock:  # read, change and save the record under the lock, so no other change is lost
        rec = load_request(req.request)
        opts = master_opts(rec["post"], req)
        if req.bpm is not None:
            opts.bpm = req.bpm
        rec["post"] = opts.asdict()
        try:
            finals = render_all(rec, opts)
        except ValueError as exc:  # e.g. a loop longer than the take
            raise HTTPException(400, str(exc)) from exc
        judge(rec, rec["takes"], finals, check)
        save_request(rec)
        _release_cache()
    print(f"[local-music] {time.strftime('%H:%M:%S')} re-mastered {rec['id']}"
          + (f" as a {opts.loop_bars}-bar loop" if opts.loop_bars else ""), flush=True)
    return request_view(rec)


@app.get("/api/requests/{rid}")
def get_request(rid: str):
    return request_view(load_request(rid))


@app.get("/api/requests")
def list_requests(limit: int = 20):
    out = []
    if OUTPUT.is_dir():
        for d in sorted((p for p in OUTPUT.iterdir() if p.is_dir() and REQ_ID.fullmatch(p.name)), reverse=True):
            try:
                rec = json.loads((d / "request.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": rec["id"], "op": rec.get("op"), "text": rec.get("text"), "engine": rec.get("engine"),
                        "name": rec.get("name"), "takes": len(rec.get("takes", [])),
                        "duration": rec.get("duration")})
            if len(out) >= max(1, min(limit, 200)):
                break
    return out


@app.delete("/api/requests/{rid}")
def delete_request(rid: str):
    d = req_dir(rid)
    if not d.is_dir():
        raise HTTPException(404, f"no request {rid}")
    with _lock:
        n = sum(1 for p in d.iterdir() if p.is_file())
        for attempt in range(20):  # a file still being streamed to the browser holds the folder
            try:
                shutil.rmtree(d)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)
    return {"id": rid, "deleted": n}


@app.get("/api/prompt")
def preview_prompt(text: str, bpm: float | None = None, key: str | None = None, kind: str = "music",
                   raw: bool = False):
    """The prompt and file name a request would use. Nothing is generated."""
    try:
        key = _check_key(key)
        return {"prompt": prompts.build_prompt(text, bpm, key, kind, raw=raw), "name": prompts.track_name(text)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/engines")
def engines():
    return {k: {"label": e["label"], "installed": engine_installed(k),
                "running": k in _workers and _workers[k].alive(),
                "info": _workers[k].info if k in _workers and _workers[k].alive() else None}
            for k, e in ENGINES.items()}


@app.post("/api/engines/{name}/stop")
def stop_engine(name: str):
    if name not in ENGINES:
        raise HTTPException(400, f"unknown engine {name!r}")
    with _lock:
        stopped = stop_worker(name)
    return {"engine": name, "stopped": stopped}


def gpu_memory() -> tuple[float, float]:
    """(used, total) GB on the whole card. nvidia-smi sees every process, the engine worker
    included; torch's mem_get_info on Windows doesn't (it read 2.6 GB with 8.5 GB in use)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.splitlines()[0]
        used, total = (float(v) / 1024 for v in out.split(","))
        return round(used, 2), round(total, 2)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        import torch
        free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        return round((total - free) / 2**30, 2), round(total / 2**30, 2)


@app.get("/api/health")
def health():
    used, total = gpu_memory()
    return {"ok": True, "version": __version__,
            "engines": {k: (k in _workers and _workers[k].alive()) for k in ENGINES},
            "clap_loaded": _clap is not None, "aesthetics_loaded": _aes is not None, "busy": _lock.locked(),
            "vram_used_gb": used, "vram_total_gb": total}


@app.get("/api/settings")
def get_settings():
    local = local_settings()
    return {"effective": {**REPO_DEFAULTS, **local}, "saved": local, "repo": REPO_DEFAULTS,
            "file": SETTINGS_FILE.name if SETTINGS_FILE.is_file() else None}


@app.post("/api/settings")
def save_settings(new: Settings):
    # Only what differs from the repo's defaults is stored: if those change in an update,
    # anything the user never chose follows along.
    given = _keep(new.model_dump(exclude_unset=True))
    data = {k: v for k, v in given.items() if v != REPO_DEFAULTS.get(k)}
    tmp = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
    with _file_lock:  # a reader never sees a half-written file
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audio.replace(tmp, SETTINGS_FILE)
    return get_settings()


@app.delete("/api/settings")
def reset_settings():
    SETTINGS_FILE.unlink(missing_ok=True)
    return get_settings()


@app.get("/files/{rid}/{name}")
def files(rid: str, name: str):
    if not FILE_NAME.fullmatch(name):
        raise HTTPException(404)
    p = req_dir(rid) / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "page.html").read_text(encoding="utf-8")


_ctrl_handler = None  # kept referenced: Windows calls it from its own thread


def _clean_up_on_console_close() -> None:
    """Closing the server's window ends the process without running atexit: delete the PID file
    and stop the engine worker first, then let Windows end it as usual."""
    if sys.platform != "win32":
        return
    import ctypes
    global _ctrl_handler

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
    def handler(event: int) -> int:
        if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            PID_FILE.unlink(missing_ok=True)
            _stop_workers()
        return 0  # not handled: the default handler ends the process
    _ctrl_handler = handler
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler, True)


_instance_lock = None


def _single_instance() -> bool:
    """Hold an exclusive lock on output/.server.lock for this process's lifetime."""
    global _instance_lock
    OUTPUT.mkdir(exist_ok=True)
    fh = open(OUTPUT / ".server.lock", "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    _instance_lock = fh
    return True


def main() -> None:
    # Always in its own visible window, so a server holding ~8 GB of VRAM can't be left running
    # out of sight: start.cmd opens that window and marks it; any other start is refused.
    if os.environ.get(WINDOW_MARK) != "1":
        sys.exit("local-music runs only in its own visible window: start it with start.cmd "
                 "(close that window, or run stop.cmd, to stop it).")
    ap = argparse.ArgumentParser(description="local-music server")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--engines", default=None,
                    help="comma-separated engines to start now (default: the default engine); others start on demand")
    ap.add_argument("--lazy", action="store_true", help="start nothing until the first request")
    args = ap.parse_args()
    # Fail fast if the port is taken (most likely by another copy of this server)
    # rather than loading models onto the GPU first and dying at bind time.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            sys.exit(f"127.0.0.1:{args.port} is in use: is local-music already running? (stop.cmd stops it)")
    # The port is only bound once the models are loaded; until then this lock stops a second copy.
    if not _single_instance():
        sys.exit("local-music is already starting up (output/.server.lock is held); stop.cmd stops it")
    log_path = serverlog.install(LOGS)
    print(f"[local-music] full log: {log_path}", flush=True)
    # stop.cmd stops exactly this process tree (and the engine workers under it)
    # by PID, whatever path - real folder or skills junction - started it.
    OUTPUT.mkdir(exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))
    _clean_up_on_console_close()
    if not args.lazy:
        names = [s.strip() for s in (args.engines or effective_settings()["engine"]).split(",") if s.strip()]
        with _lock:
            for n in names:
                try:
                    worker(n)
                except Exception as exc:  # the server still serves the page and the other engine
                    print(f"[local-music] could not start {n}: {getattr(exc, 'detail', exc)}", flush=True)
            try:
                # A first call pays for CUDA kernel selection: do it now, not on a request.
                noise = (np.random.default_rng(0).standard_normal((2, 5 * audio.SR)) * 0.1).astype(np.float32)
                clap().score([noise], audio.SR, "noise")
                aes().score([noise], audio.SR)
                _release_cache()
            except Exception as exc:
                print(f"[local-music] checkers not loaded: {getattr(exc, 'detail', exc)}", flush=True)
    # Loopback only, always: nothing here is meant to be reachable from the network.
    print(f"[local-music] ready on http://127.0.0.1:{args.port}/  (stop: close this window, or stop.cmd)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
