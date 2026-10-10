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


def test_parts_for_combine():
    ref = "20261010-120000-abcd:5"
    assert cli.parse_part(ref) == {"take": ref}
    assert cli.parse_part(f"{ref}@20-60") == {"take": ref, "start": 20.0, "end": 60.0}
    assert cli.parse_part(f"{ref}@30.5-") == {"take": ref, "start": 30.5}
    assert cli.parse_part(f"{ref}@-12") == {"take": ref, "end": 12.0}
    for bad in ("a drum break", f"{ref}@x-y", "20261010-120000-abcd"):
        with pytest.raises(ValueError):
            cli.parse_part(bad)


def test_audio_files_are_recognised(tmp_path):
    f = tmp_path / "demo.MP3"
    f.write_bytes(b"x")
    assert cli.is_audio_file(str(f)) and not cli.is_audio_file(str(tmp_path / "missing.wav"))
    assert not cli.is_audio_file("20261010-120000-abcd:5") and not cli.is_audio_file(None)


def test_layers_for_mix():
    ref = "20261010-120000-abcd:5"
    assert cli.parse_layer(ref) == {"take": ref}
    assert cli.parse_layer(f"{ref},vol=-8,at=4") == {"take": ref, "gain_db": -8.0, "at": 4.0}
    assert cli.parse_layer(f"{ref}@10-40, vol=-6dB, at=2.5s") == {"take": ref, "start": 10.0, "end": 40.0,
                                                                  "gain_db": -6.0, "at": 2.5}
    for bad in (f"{ref},duck", f"{ref},vol=loud", "voice.mp3,vol=-3", f"{ref},reverb=garage", f"{ref},boost=4-6",
                f"{ref},voice=yes"):
        with pytest.raises(ValueError):
            cli.parse_layer(bad)


def test_layer_edits_and_files_for_mix(tmp_path):
    ref = "20261010-120000-abcd:5"
    assert cli.parse_layer(f"{ref},at=4,voice,reverb=room,boost=60.9-63.5:+4,boost=10-12:-6dB") == {
        "take": ref, "at": 4.0, "voice": True, "reverb": "room",
        "spans": [{"start": 60.9, "end": 63.5, "gain_db": 4.0}, {"start": 10.0, "end": 12.0, "gain_db": -6.0}]}
    f = tmp_path / "voice.wav"
    f.write_bytes(b"x")
    assert cli.parse_layer(f"{f},at=4,voice") == {"file": str(f), "at": 4.0, "voice": True}


def test_combine_parts_take_a_volume():
    ref = "20261010-120000-abcd:5"
    assert cli.parse_combine_part(f"{ref}@0-35.2,vol=-6") == {"take": ref, "start": 0.0, "end": 35.2, "gain_db": -6.0}
    assert cli.parse_combine_part(ref) == {"take": ref}
    with pytest.raises(ValueError):
        cli.parse_combine_part(f"{ref},at=4")


def test_export_copies_the_take_and_everything_it_was_made_from(tmp_path, monkeypatch):
    res = fake_result(tmp_path, monkeypatch, seeds=(0,))  # the mix: 20261010-120000-abcd:0
    out = tmp_path / "output"
    views = {res["id"]: {**res, "op": "mix", "layers": [{"take": "20261010-110000-0001:7"},
                                                        {"take": "20261010-110000-0002:0"}]},
             "20261010-110000-0001": {"id": "20261010-110000-0001", "op": "inpaint", "source": "20261010-100000-0003:7",
                                      "takes": []},
             "20261010-110000-0002": {"id": "20261010-110000-0002", "op": "import", "source": "voice.wav", "takes": []},
             "20261010-100000-0003": {"id": "20261010-100000-0003", "op": "generate", "takes": []}}
    for rid in list(views)[1:]:
        (out / rid).mkdir(parents=True)
        (out / rid / "request.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "call", lambda path, *a, **k: calls.append(path) or views[path.rsplit("/", 1)[1]])
    dest = tmp_path / "keep"
    files, chain = cli.export(f"{res['id']}:0", dest, "wav", png=False, name="final")
    assert [f.name for f in files] == ["final_v01.wav"]
    assert set(chain) == set(views) and all((dest / "requests" / r).is_dir() for r in views)
    man = json.loads((dest / "local-music-export.json").read_text(encoding="utf-8"))
    assert man["take"] == f"{res['id']}:0" and len(man["requests"]) == 4
    assert "copy the folders in requests/" in (dest / "README.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        cli.export(f"{res['id']}:9", tmp_path / "other", "wav", png=False, name=None)


def test_sources_of_a_request():
    assert cli.sources_of({"source": "20261010-100000-0003:7"}) == ["20261010-100000-0003:7"]
    assert cli.sources_of({"source": "voice.wav", "layers": [{"take": "20261010-100000-0003:1"}]}) ==         ["20261010-100000-0003:1"]
    assert cli.sources_of({"parts": [{"take": "a:1"}, {"take": "b:2"}]}) == ["a:1", "b:2"]
