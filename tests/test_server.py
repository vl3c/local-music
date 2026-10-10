r"""The server with a fake engine and fake scorers: every endpoint, the records, the guards.

The fake worker writes synthetic music (conftest.band) where the engine would, and records
the jobs it was sent, so the tests check both what the server asks of the engine and what it
makes of the answer. No GPU and no weights are needed.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import SR, band

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))


class FakeWorker:
    def __init__(self, name="medium", max_duration=380.0):
        self.name = name
        self.info = {"engine": name, "model": f"fake-{name}", "sample_rate": SR, "channels": 2,
                     "max_duration": max_duration, "default_steps": 8, "features": ["init_audio", "inpaint", "seed"],
                     "ignores": ["negative", "cfg"]}
        self.jobs = []

    def call(self, job):
        from music import audio
        self.jobs.append(job)
        files = []
        for s in job["seeds"]:
            x = band(job.get("bpm_fake", 120.0), job["duration"], seed=s)
            p = Path(job["out_dir"]) / f"{job['stem']}_{s}.wav"
            audio.write_wav(p, x, SR, subtype="FLOAT")
            files.append({"seed": s, "path": str(p)})
        return {"ok": True, "files": files, "sample_rate": SR}

    def alive(self):
        return True


class FakeClap:
    def score(self, audios, sr, prompt):
        self.prompt = prompt
        return [{"clap": 0.4 + 0.01 * i} for i in range(len(audios))]


class FakeAes:
    def score(self, audios, sr):
        return [{"CE": 7.0 - 0.1 * i, "CU": 7.0, "PC": 5.0, "PQ": 7.5} for i in range(len(audios))]


@pytest.fixture()
def server(tmp_path, monkeypatch):
    import app as server  # noqa: PLC0415  (FastAPI and the music package; loads no models)
    monkeypatch.setattr(server, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(server, "SETTINGS_FILE", tmp_path / "config.local.json")
    (tmp_path / "output").mkdir()
    fake = FakeWorker()
    monkeypatch.setattr(server, "worker", lambda name: fake)
    clap, aes = FakeClap(), FakeAes()
    monkeypatch.setattr(server, "clap", lambda: clap)
    monkeypatch.setattr(server, "aes", lambda: aes)
    monkeypatch.setattr(server, "_release_cache", lambda: None)
    server._fake, server._fake_clap = fake, clap
    return server


@pytest.fixture()
def client(server):
    from fastapi.testclient import TestClient
    return TestClient(server.app, base_url="http://127.0.0.1:8192")


def gen(client, **kw):
    body = {"text": "steady test groove", "duration": 12, "n": 2, "seed": 100, **kw}
    r = client.post("/api/generate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ generate

def test_generate_makes_ranked_checked_takes(client, server):
    res = gen(client, bpm=120, key="C major")
    assert res["op"] == "generate" and res["kept"] == 2 and [t["rank"] for t in res["takes"]] == [1, 2]
    job = server._fake.jobs[-1]
    assert job["seeds"] == [100, 101] and job["duration"] == 12 and job["steps"] == 8
    assert job["prompt"] == "TrackType: Music, VocalType: Instrumental, steady test groove, in C major, 120 BPM"
    assert server._fake_clap.prompt == "steady test groove, in C major, 120 BPM"  # CLAP hears no tags
    t = res["takes"][0]
    assert t["music"]["tempo"]["match"] == "exact" and t["aes"]["CE"] == 7.0
    assert t["final"]["lufs"] == pytest.approx(-14.0, abs=0.6) and t["final"]["true_peak_dbtp"] <= -0.95
    d = server.OUTPUT / res["id"]
    for k in ("wav", "mp3", "png"):
        assert Path(t["paths"][k]).is_file()
    assert (d / f"{t['seed']}_raw.wav").is_file() and not list(d.glob("gen_*.wav"))
    rec = json.loads((d / "request.json").read_text(encoding="utf-8"))
    assert rec["takes"][0]["beats"]  # stored for loops


def test_a_take_that_stops_gets_faded_automatically(client):
    t = gen(client)["takes"][0]
    assert t["post"]["auto_fade_out_s"] > 0 and not any("abruptly" in w for w in t["warnings"])
    t0 = gen(client, fade_out=0)["takes"][0]
    assert "auto_fade_out_s" not in t0["post"] and any("abruptly" in w for w in t0["warnings"])


def test_loop_generation_asks_for_enough_music(client, server):
    res = gen(client, bpm=120, loop_bars=8, duration=10)
    assert server._fake.jobs[-1]["duration"] >= 8 * 4 * 0.5 + 4  # 16 s of loop + room to start it
    lp = res["takes"][0]["post"]
    assert lp["loop"]["bars"] == 8 and lp["length_s"] == pytest.approx(16.0, abs=0.02)


def test_too_long_is_refused_before_anything_is_made(client, server):
    server._fake.info["max_duration"] = 120.0
    r = client.post("/api/generate", json={"text": "x", "duration": 200})
    assert r.status_code == 400 and "at most 120" in r.json()["detail"]
    assert not list(server.OUTPUT.iterdir())


@pytest.mark.parametrize("body", [{"text": "  "}, {"text": "x", "key": "H minor"}, {"text": "x", "duration": 1},
                                  {"text": "x", "bpm": 500}, {"text": "x", "seeds": [1, 1]}, {"text": "x", "n": 9},
                                  {"text": "x", "engine": "large"}, {"text": "x", "name": "bad name"}])
def test_bad_requests_are_4xx(client, body):
    assert client.post("/api/generate", json=body).status_code == 422


def test_a_failed_generation_leaves_nothing_behind(client, server, monkeypatch):
    def boom(job):
        raise RuntimeError("worker died")
    monkeypatch.setattr(server._fake, "call", boom)
    r = client.post("/api/generate", json={"text": "x", "duration": 5})
    assert r.status_code == 500 and "worker died" in r.json()["detail"]
    assert not list(server.OUTPUT.iterdir())


# ------------------------------------------------------------------ from a take

def test_extend_continues_the_take(client, server):
    src = gen(client, n=1)
    ref = src["takes"][0]["ref"]
    r = client.post("/api/extend", json={"take": ref, "seconds": 20, "n": 1}).json()
    job = server._fake.jobs[-1]
    assert job["inpaint_audio"].endswith("_raw.wav") and job["mask_start"] == pytest.approx(12.0, abs=0.01)
    assert job["mask_end"] == pytest.approx(32.0, abs=0.01) and job["duration"] == pytest.approx(32.0, abs=0.01)
    assert r["op"] == "extend" and r["source"] == ref and r["name"] == src["name"] and r["prompt"] == src["prompt"]


def test_extend_past_the_window_hears_the_end_and_keeps_the_start(client, server):
    server._fake.info["max_duration"] = 40.0
    src = gen(client, n=1, duration=30)
    r = client.post("/api/extend", json={"take": src["takes"][0]["ref"], "seconds": 20, "n": 1}).json()
    job = server._fake.jobs[-1]
    assert job["inpaint_audio"].endswith("context.wav") and job["duration"] == pytest.approx(40.0, abs=0.01)
    assert job["mask_start"] == pytest.approx(20.0, abs=0.01)
    assert r["takes"][0]["signal"]["duration_s"] == pytest.approx(50.0, abs=0.05)  # 10 s kept + 40 s from the engine
    assert not (server.OUTPUT / r["id"] / "context.wav").exists()


def test_vary_and_inpaint_send_the_source(client, server):
    src = gen(client, n=1)
    ref = src["takes"][0]["ref"]
    client.post("/api/vary", json={"take": ref, "strength": 0.3, "n": 1})
    job = server._fake.jobs[-1]
    assert job["init_audio"].endswith("_raw.wav") and job["init_strength"] == 0.3
    r = client.post("/api/inpaint", json={"take": ref, "start": 4, "end": 8, "text": "a drum fill", "n": 1}).json()
    job = server._fake.jobs[-1]
    assert (job["mask_start"], job["mask_end"]) == (4, 8) and "a drum fill" in job["prompt"]
    assert server._fake_clap.prompt == "steady test groove"  # judged as the source it mostly is
    assert r["span"] == [4, 8]
    bad = client.post("/api/inpaint", json={"take": ref, "start": 8, "end": 40})
    assert bad.status_code == 400


def test_unknown_takes_are_404_and_malformed_refs_400(client):
    assert client.post("/api/vary", json={"take": "20261010-120000-abcd:1"}).status_code == 404
    assert client.post("/api/vary", json={"take": "nope"}).status_code == 400


# ------------------------------------------------------------------ re-master

def test_master_relevels_and_loops_without_regenerating(client, server):
    res = gen(client, bpm=120, duration=30)
    n_jobs = len(server._fake.jobs)
    r = client.post("/api/master", json={"request": res["id"], "lufs": -18, "loop_bars": 4}).json()
    assert len(server._fake.jobs) == n_jobs
    t = r["takes"][0]
    assert t["final"]["lufs"] == pytest.approx(-18, abs=0.6) and t["post"]["loop"]["bars"] == 4
    back = client.post("/api/master", json={"request": res["id"], "loop_bars": None}).json()
    assert "loop" not in back["takes"][0]["post"] and back["post"]["lufs"] == -18


def test_master_a_loop_too_long_is_400(client):
    res = gen(client, bpm=120)
    r = client.post("/api/master", json={"request": res["id"], "loop_bars": 32})
    assert r.status_code == 400 and "too short" in r.json()["detail"]


# ------------------------------------------------------------------ the rest of the API

def test_list_get_files_and_delete(client, server):
    res = gen(client, n=1)
    assert client.get("/api/requests").json()[0]["id"] == res["id"]
    assert client.get(f"/api/requests/{res['id']}").json()["takes"][0]["ref"] == res["takes"][0]["ref"]
    f = res["takes"][0]["files"]
    assert client.get(f"/files/{f['mp3']}").status_code == 200
    assert client.get(f"/files/{res['id']}/..%2Fconfig.local.json").status_code == 404
    assert client.get("/files/../x/y.wav").status_code == 404
    assert client.delete(f"/api/requests/{res['id']}").json()["deleted"] >= 5
    assert not (server.OUTPUT / res["id"]).exists()


def test_prompt_preview(client):
    r = client.get("/api/prompt", params={"text": "dark synthwave", "bpm": 100, "key": "dm"}).json()
    assert r["prompt"].endswith("dark synthwave, in D minor, 100 BPM") and r["name"] == "dark-synthwave"
    assert client.get("/api/prompt", params={"text": "x", "key": "Q"}).status_code == 400


def test_settings_save_only_differences_and_apply(client, server):
    r = client.post("/api/settings", json={"n": 2, "engine": "medium", "duration": 20, "lufs": -16}).json()
    assert r["saved"] == {"n": 2, "duration": 20, "lufs": -16}
    res = client.post("/api/generate", json={"text": "x"}).json()
    assert len(res["takes"]) == 2 and server._fake.jobs[-1]["duration"] == 20
    assert res["takes"][0]["final"]["lufs"] == pytest.approx(-16, abs=0.6)
    assert client.delete("/api/settings").json()["saved"] == {}
    assert client.post("/api/settings", json={"bogus": 1}).status_code == 422


def test_engines_and_page(client):
    assert set(client.get("/api/engines").json()) == {"medium", "small"}
    page = client.get("/").text
    assert "<title>local-music</title>" in page and "/api/generate" in page


# ------------------------------------------------------------------ guards

def test_cross_site_writes_are_refused(client):
    r = client.post("/api/generate", json={"text": "x"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/generate", json={"text": "x"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert client.get("/api/requests").headers["X-Frame-Options"] == "DENY"


def test_other_hosts_are_refused(server):
    from fastapi.testclient import TestClient
    c = TestClient(server.app, base_url="http://attacker.example:8192")
    assert c.get("/api/requests").status_code == 400


def test_the_server_runs_only_in_its_window(server, monkeypatch):
    monkeypatch.delenv(server.WINDOW_MARK, raising=False)
    with pytest.raises(SystemExit, match="start.cmd"):
        server.main()


# ------------------------------------------------------------------ your own audio

def wav_bytes(x, sr=SR) -> bytes:
    import io

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, x.T, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_import_makes_a_take_the_rest_can_start_from(client, server):
    data = wav_bytes(band(100, 15.0, stereo=False, sr=48000), 48000)
    r = client.post("/api/import", params={"filename": "C:\music\My Song.wav"}, content=data,
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    res = r.json()
    t = res["takes"][0]
    assert res["op"] == "import" and res["source"] == "My Song.wav" and res["name"] == "my-song"
    assert t["ref"].endswith(":0") and res["prompt"] == "" and "clap" not in t and t["aes"]["PQ"] == 7.5
    assert t["music"]["tempo"]["bpm"] == pytest.approx(100, abs=1) and t["signal"]["duration_s"] == pytest.approx(15, abs=0.01)
    assert not server._fake.jobs  # nothing generated
    client.post("/api/vary", json={"take": t["ref"], "strength": 0.4, "n": 1})
    job = server._fake.jobs[-1]
    assert job["prompt"] == "" and job["init_audio"].endswith("0_raw.wav")  # led by the audio alone
    client.post("/api/extend", json={"take": t["ref"], "seconds": 10, "text": "the same, with strings", "n": 1})
    assert "the same, with strings" in server._fake.jobs[-1]["prompt"]


def test_import_with_a_description_is_scored_against_it(client, server):
    from music import audio
    p = server.OUTPUT / "x.mp3"
    audio.write_mp3(p, band(120, 6.0, sr=48000))
    r = client.post("/api/import", params={"filename": "groove.mp3", "text": "a funky drum groove"},
                    content=p.read_bytes())
    res = r.json()
    assert res["prompt"].endswith("a funky drum groove") and res["takes"][0]["clap"] == 0.4
    assert server._fake_clap.prompt == "a funky drum groove"


@pytest.mark.parametrize("data,code", [(b"", 400), (b"not audio at all" * 100, 400)])
def test_import_refuses_what_isnt_audio(client, data, code):
    r = client.post("/api/import", params={"filename": "x.m4a"}, content=data)
    assert r.status_code == code and ("audio" in r.json()["detail"])


def test_import_refuses_a_cross_site_upload(client):
    r = client.post("/api/import", params={"filename": "a.wav"}, content=wav_bytes(band(100, 3.0)),
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


# ------------------------------------------------------------------ arranging

def test_arrange_bridges_parts_into_one_song(client, server):
    a = gen(client, n=1, duration=30, bpm=120)["takes"][0]["ref"]
    b = gen(client, n=1, duration=30, bpm=120)["takes"][0]["ref"]
    c = gen(client, n=1, duration=20, bpm=120)["takes"][0]["ref"]
    before = len(server._fake.jobs)
    r = client.post("/api/arrange", json={"parts": [{"take": a}, {"take": b, "start": 5, "end": 25}, {"take": c}],
                                         "n": 2, "seed": 7})
    assert r.status_code == 200, r.text
    res = r.json()
    jobs = server._fake.jobs[before:]
    assert len(jobs) == 2 and all(j["seeds"] == [7, 8] for j in jobs)  # one engine call per join
    assert res["bridge"] == pytest.approx(4.0)  # 2 bars at 120 BPM
    # join 0 hears 10 s of each side (half of part b's 20 s span), join 1 hears 10 s of b and of c
    assert jobs[0]["duration"] == pytest.approx(10 + 4 + 10, abs=0.01)
    assert (jobs[0]["mask_start"], jobs[0]["mask_end"]) == (pytest.approx(9.0), pytest.approx(15.0))
    assert res["op"] == "arrange" and [p["take"] for p in res["parts"]] == [a, b, c]
    for t in res["takes"]:
        assert t["signal"]["duration_s"] == pytest.approx(30 + 20 + 20 + 2 * 4.0, abs=0.05)
    assert not list((server.OUTPUT / res["id"]).glob("join*")) and not list((server.OUTPUT / res["id"]).glob("gen_*"))


def test_arrange_warns_about_a_tempo_change_and_checks_spans(client, server):
    a = gen(client, n=1, duration=20)["takes"][0]["ref"]
    server._fake.call = (lambda orig: lambda job: orig({**job, "bpm_fake": 90.0}))(server._fake.call)
    b = gen(client, n=1, duration=20)["takes"][0]["ref"]
    res = client.post("/api/arrange", json={"parts": [{"take": a}, {"take": b}], "bridge": 2, "n": 1}).json()
    assert any("BPM" in w and "change tempo" in w for w in res["warnings"])
    assert client.post("/api/arrange", json={"parts": [{"take": a}, {"take": b, "start": 18}]}).status_code == 400
    assert client.post("/api/arrange", json={"parts": [{"take": a}]}).status_code == 422
