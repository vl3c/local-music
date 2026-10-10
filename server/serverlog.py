"""Server output: everything to logs/server.log, only what matters to the console window.

The engine's libraries are chatty at startup (warmup steps, progress bars, compiler
notes). The window should show milestones, one line per take and anything that goes
wrong; the full record goes to the log, timestamped, for when something does.

This works at the Python level (sys.stdout / sys.stderr), never by redirecting the
process's file descriptors: in a console window that breaks Python's own writes.
"""

from __future__ import annotations

import io
import re
import sys
import threading
import time
from pathlib import Path

# Known, harmless engine chatter: logged, never shown in the window.
NOISE = re.compile("|".join([
    r"flash_attn", r"No module named 'flash_attn'", r"disabling (Flash Attention|varlen attention)",
    r"Loading checkpoint shards", r"Loading weights", r"^\s*\d+%\|", r"it/s\]$", r"s/it\]$",
    r"Pipeline assembled on", r"^\s*(Loaded|Loading) ", r"Mismatch dtype between input and weight",
    r"return torch\.rms_norm", r"WeightNorm\.apply", r"weight_norm` is deprecated",
    r"is deprecated(!| for | and will be removed)",  # library API-change notices
    r"^\[(medium|small)\] loaded in", r"^\[(medium|small)\] \d+ take",
    r"^\S+\.py:\d+: \w*Warning:", r"^\s*warnings\.warn\(",  # library deprecation notices
    r"^\s*(ai|ti) = self\.proc\(",
]))


class Tee(io.TextIOBase):
    """A text stream that writes every line to the log and the non-noise ones to `console`."""

    def __init__(self, console, log, lock: threading.Lock):
        self.console, self.log, self.lock = console, log, lock
        self._buf = ""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False  # so progress bars don't animate into the log

    def fileno(self) -> int:
        return self.console.fileno()

    @property
    def encoding(self) -> str:
        return "utf-8"

    def write(self, s: str) -> int:
        with self.lock:
            self._buf += s
            while True:
                cut = [i for i in (self._buf.find("\n"), self._buf.find("\r")) if i != -1]
                if not cut:
                    break
                i = min(cut)
                line, end, self._buf = self._buf[:i], self._buf[i], self._buf[i + 1:]
                if end == "\r" and not self._buf.startswith("\n"):
                    continue  # a progress-bar redraw: only its final state matters
                self._emit(line)
        return len(s)

    def flush(self) -> None:
        with self.lock:
            self.log.flush()
            self.console.flush()

    def _emit(self, line: str) -> None:
        if not line.strip():
            return
        self.log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        self.log.flush()
        if not NOISE.search(line):
            self.console.write(line + "\n")
            self.console.flush()


def install(log_dir: Path) -> Path:
    """Route this process's stdout and stderr through the log. Returns the log's path.

    The previous run's log is kept as server.previous.log, so a crash's details survive
    the next start.
    """
    log_dir.mkdir(exist_ok=True)
    path = log_dir / "server.log"
    if path.exists():
        path.replace(log_dir / "server.previous.log")
    log = open(path, "a", encoding="utf-8", errors="replace")
    log.write(f"=== local-music server, started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    lock = threading.Lock()
    sys.stdout = Tee(sys.stdout, log, lock)
    sys.stderr = Tee(sys.stderr, log, lock)
    return path
