r"""The server's console filter and log.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import io
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

import serverlog  # noqa: E402


def tee():
    console, log = io.StringIO(), io.StringIO()
    return serverlog.Tee(console, log, threading.Lock()), console, log


def test_noise_goes_to_the_log_only():
    t, console, log = tee()
    t.write("No module named 'flash_attn'\n")
    t.write("flash_attn not installed, disabling Flash Attention\n")
    t.write("[medium] 3 take(s), 30s each: warm lo-fi hip hop\n")
    t.write("[local-music] ready on http://127.0.0.1:8192/\n")
    assert console.getvalue() == "[local-music] ready on http://127.0.0.1:8192/\n"
    assert "flash_attn" in log.getvalue() and "[medium] 3 take(s)" in log.getvalue()


def test_errors_and_unknown_output_reach_the_window():
    t, console, _ = tee()
    t.write("Traceback (most recent call last):\n  File \"app.py\", line 1\nRuntimeError: boom\n")
    t.write("something the filter has never seen\n")
    shown = console.getvalue()
    assert "Traceback" in shown and "RuntimeError: boom" in shown and "never seen" in shown


def test_progress_bar_redraws_collapse_to_the_final_state():
    t, console, log = tee()
    for i in (0, 4, 8):
        t.write(f"\r{i * 12}%|#####     | {i}/8 [00:00<00:00, 31.11it/s]")
    t.write("\n")
    assert log.getvalue().count("it/s]") == 1 and "8/8" in log.getvalue()
    assert console.getvalue() == ""  # a known noisy bar: log only


def test_partial_writes_are_joined_into_lines():
    t, console, _ = tee()
    t.write("[local-music] 20261010-120000-aaaa generate medium: ")
    t.write("3 take(s) of warm-lofi-hip-hop\n")
    assert console.getvalue() == "[local-music] 20261010-120000-aaaa generate medium: 3 take(s) of warm-lofi-hip-hop\n"


def test_install_keeps_the_previous_log(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    (tmp_path / "server.log").write_text("old run\n", encoding="utf-8")
    path = serverlog.install(tmp_path)
    print("[local-music] hello")
    sys.stdout.flush()
    assert (tmp_path / "server.previous.log").read_text(encoding="utf-8") == "old run\n"
    assert "[local-music] hello" in path.read_text(encoding="utf-8")
