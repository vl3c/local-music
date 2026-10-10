"""The line protocol between the server and an engine worker (standard library only).

Each engine runs in its own venv (their dependencies clash: SA3 wants torch 2.7 and
transformers 5, MOSS torch 2.9 and transformers 4.57), as a child process of the server.
They talk over stdin/stdout, one JSON object per line:

    <- {"ready": true, "engine": ..., "sample_rate": ..., "max_duration": ..., "features": [...]}
    -> {"op": "generate", "prompt": ..., "negative": ..., "duration": 3.5, "seeds": [1, 2],
        "steps": 8, "cfg": 1.0, "out_dir": "...", "stem": "...",
        "init_audio": path | null, "init_strength": 0.5}
    <- {"ok": true, "files": [{"seed": 1, "path": "..."}], "sample_rate": 44100, "gen_s": 1.2}
    <- {"ok": false, "error": "..."}

Import this module first: it takes the real stdout for the protocol and points file
descriptor 1 at stderr, so anything else printed (by Python or native code: CUDA, progress
bars) lands in the server's log instead of corrupting the stream.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
os.dup2(2, 1)
sys.stdout = sys.stderr

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"

# Everything is loaded from models/: never let a library reach for the network.
for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
             "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error",
             "DO_NOT_TRACK": "1", "GRADIO_ANALYTICS_ENABLED": "False"}.items():
    os.environ[k] = v


def log(name: str, *a) -> None:
    print(f"[{name}]", *a, file=sys.stderr, flush=True)


def send(obj: dict) -> None:
    _proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _proto.flush()


def serve(name: str, load, handle) -> None:
    """load() -> (engine, ready_info); handle(engine, request) -> reply dict. Runs until stdin closes."""
    sys.stdin.reconfigure(encoding="utf-8")
    t0 = time.time()
    try:
        engine, info = load()
    except Exception as exc:
        log(name, traceback.format_exc())
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    log(name, f"loaded in {time.time() - t0:.1f}s")
    send({"ready": True, **info})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "ping":
                send({"ok": True})
                continue
            t1 = time.time()
            reply = handle(engine, req)
            send({"ok": True, "gen_s": round(time.time() - t1, 2), **reply})
        except Exception as exc:  # report and keep serving
            log(name, traceback.format_exc())
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
