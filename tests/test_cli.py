r"""scripts/music.py without a server: delivery, request building and printing. And the
downloader's reuse rules (scripts/download_models.py).

    .venv\Scripts\python.exe -m pytest tests -q
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import download_models  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# scripts/music.py shares its name with the music package: load it under another one.
cli = _load("music_cli", SCRIPTS / "music.py")


def fake_result(tmp_path, monkeypatch, seeds=(5, 6)) -> dict:
    out = tmp_path / "output"
    monkeypatch.setattr(cli, "OUTPUT", out)
    rid = "20261010-120000-abcd"
    (out / rid).mkdir(parents=True)
    takes = []
    for r, s in enumerate(seeds, 1):
        for ext in ("wav", "mp3", "png"):
            (out / rid / f"{s}.{ext}").write_bytes(f"{ext}{s}".encode())
        takes.append({"ref": f"{rid}:{s}", "seed": s, "rank": r, "rank_score": 0.8, "warnings": [],
                      "files": {k: f"{rid}/{s}.{k}" for k in ("wav", "mp3", "png")},
                      "paths": {k: str(out / rid / f"{s}.{k}") for k in ("wav", "mp3", "png")},
                      "aes": {"CE": 7.1, "PQ": 7.9}, "clap": 0.41,
                      "music": {"tempo": {"bpm": 120.0, "match": "exact"}, "key": {"key": "C major"}},
                      "post": {"length_s": 30.0}})
    return {"id": rid, "op": "generate", "name": "warm-lofi", "prompt": "p", "text": "t", "engine": "medium",
            "takes": takes, "kept": len(takes)}


def test_delivery_numbers_files_and_never_overwrites(tmp_path, monkeypatch):
    res = fake_result(tmp_path, monkeypatch)
    dest = tmp_path / "deliver"
    first = cli.deliver(res, res["takes"][:1], dest, "all", png=False)
    assert [p.name for p in first] == ["warm-lofi_v01.wav", "warm-lofi_v01.mp3"]
    again = cli.deliver(res, res["takes"], dest, "all", png=True)
    assert [p.name for p in again] == ["warm-lofi_v02.wav", "warm-lofi_v02.mp3", "warm-lofi_v02.png",
                                       "warm-lofi_v03.wav", "warm-lofi_v03.mp3", "warm-lofi_v03.png"]
    assert (dest / "warm-lofi_v01.wav").read_bytes() == b"wav5"  # untouched
    side = json.loads((dest / "warm-lofi_v03.json").read_text(encoding="utf-8"))
    assert side["ref"] == res["takes"][1]["ref"] and side["aes"]["PQ"] == 7.9
    mp3 = cli.deliver(res, res["takes"][:1], dest, "mp3", png=False, name="Boss-Theme")
    assert [p.name for p in mp3] == ["Boss-Theme_v01.mp3"]
    with pytest.raises(ValueError):
        cli.deliver(res, res["takes"][:1], dest, "all", png=False, name="no spaces")


def test_describe_shows_scores_tempo_key_and_warnings(tmp_path, monkeypatch):
    t = fake_result(tmp_path, monkeypatch)["takes"][0]
    t["warnings"] = ["so-so production quality"]
    t["music"]["key"]["note"] = "the relative key"
    line = cli.describe(t)
    for bit in ("#1", "CE 7.1", "PQ 7.9", "CLAP 0.41", "120 BPM (exact)", "C major", "30.0s",
                "warning: so-so production quality", "note: the relative key"):
        assert bit in line


def ns(**kw):
    base = dict(length=None, fade_in=None, fade_out=None, lufs=None, peak=None, loop_bars=None, beats_per_bar=None,
                no_loop=False, full_length=False, n=None, engine=None, steps=None, check=None, seeds=None, seed=None)
    return argparse.Namespace(**{**base, **kw})


def test_unset_options_are_left_to_the_server():
    assert cli.master_fields(ns()) == {} and cli.shape_fields(ns()) == {}


def test_flags_map_to_the_request():
    assert cli.master_fields(ns(fade_out=0.0, lufs=-16.0, loop_bars=8)) == {"fade_out": 0.0, "lufs": -16.0,
                                                                            "loop_bars": 8}
    assert cli.master_fields(ns(no_loop=True, full_length=True)) == {"loop_bars": None, "length": None}
    assert cli.shape_fields(ns(seed="random")) == {"seed": None}
    assert cli.shape_fields(ns(seed="7", seeds="3,4", n=2)) == {"n": 2, "seeds": [3, 4]}


# ------------------------------------------------------------------ the downloader's reuse rules

def test_reuse_finds_identical_files_under_other_folders_only_when_checksummed(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "no-cache"))
    other = tmp_path / "other" / "models"
    (other / "sa3-small-sfx" / "t5gemma-b-b-ul2").mkdir(parents=True)
    enc = other / "sa3-small-sfx" / "t5gemma-b-b-ul2" / "model.safetensors"
    enc.write_bytes(b"x")
    entry = {"name": "sa3-medium", "repo": "stabilityai/stable-audio-3-medium"}
    rel = "t5gemma-b-b-ul2/model.safetensors"
    assert download_models.candidates(entry, rel, [other], any_folder=True) == [enc]
    assert download_models.candidates(entry, rel, [other], any_folder=False) == []
    (other / "sa3-medium").mkdir()
    same = other / "sa3-medium" / "model_config.json"
    same.write_text("{}")
    assert download_models.candidates(entry, "model_config.json", [other]) == [same]


def test_manifest_is_consistent():
    repo = Path(__file__).resolve().parent.parent
    entries = json.loads((repo / "models.json").read_text(encoding="utf-8"))["repos"]
    lock = json.loads((repo / "models.lock.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in entries]
    assert len(set(names)) == len(names)
    for e in entries:
        assert len(e["revision"]) == 40 and e["licence"] and e["files"]
        if e["name"] in lock:
            assert lock[e["name"]]["revision"] == e["revision"]
            assert all(f["sha256"] and f["bytes"] > 0 for f in lock[e["name"]]["files"])
    assert set(lock) <= set(names)
