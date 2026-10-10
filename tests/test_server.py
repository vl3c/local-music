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
    r = client.post("/api/import", params={"filename": r"C:\music\My Song.wav"}, content=data,
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
    assert all(p["name"] == "steady-test-groove" for p in res["parts"])
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


# ------------------------------------------------------------------ mixing

def test_mix_puts_a_voice_over_the_music_at_set_volumes(client, server):
    song = gen(client, n=1, duration=20, bpm=120)["takes"][0]["ref"]
    voice_x = np.zeros((1, 6 * 48000), np.float32)
    voice_x[0, 48000:4 * 48000] = 0.4 * np.sin(np.arange(3 * 48000) / 48000 * 2 * np.pi * 180)
    voice = client.post("/api/import", params={"filename": "voice.wav"}, content=wav_bytes(voice_x, 48000)).json()
    vref = voice["takes"][0]["ref"]
    n_jobs = len(server._fake.jobs)
    r = client.post("/api/mix", json={"layers": [{"take": song, "gain_db": -9}, {"take": vref, "at": 5}]})
    assert r.status_code == 200, r.text
    res = r.json()
    t = res["takes"][0]
    assert res["op"] == "mix" and len(server._fake.jobs) == n_jobs  # nothing generated
    assert [(ly["take"], ly["gain_db"], ly["at"]) for ly in res["layers"]] == [(song, -9, 0), (vref, 0, 5)]
    assert [ly["name"] for ly in res["layers"]] == ["steady-test-groove", "voice"]
    assert t["signal"]["duration_s"] == pytest.approx(20.0, abs=0.05) and t["ref"].endswith(":0")
    assert res["prompt"] == client.get(f"/api/requests/{song.split(':')[0]}").json()["prompt"]  # the first layer's
    # The mix is a take like any other: it can be extended.
    client.post("/api/extend", json={"take": t["ref"], "seconds": 10, "n": 1})
    assert server._fake.jobs[-1]["inpaint_audio"].endswith("0_raw.wav")


def test_mix_runs_as_long_as_its_longest_layer_and_validates(client):
    a = gen(client, n=1, duration=10)["takes"][0]["ref"]
    b = gen(client, n=1, duration=8)["takes"][0]["ref"]
    res = client.post("/api/mix", json={"layers": [{"take": a}, {"take": b, "at": 6, "start": 2}]}).json()
    assert res["takes"][0]["signal"]["duration_s"] == pytest.approx(12.0, abs=0.1)
    assert client.post("/api/mix", json={"layers": [{"take": a}, {"take": b, "start": 30}]}).status_code == 400
    assert client.post("/api/mix", json={"layers": [{"take": a}, {"take": b, "gain_db": 40}]}).status_code == 422
    assert client.post("/api/mix", json={"layers": [{"take": a}]}).status_code == 422


# ------------------------------------------------------------------ what the engine writes, and what it doesn't

def raw_of(server, ref):
    from music import audio
    rid, seed = ref.split(":")
    rec = json.loads((server.OUTPUT / rid / "request.json").read_text(encoding="utf-8"))
    t = next(t for t in rec["takes"] if t["seed"] == int(seed))
    return audio.read(server.OUTPUT / rid / t["raw"])[0], t


def noisy_engine(server, monkeypatch):
    """The fake engine writes loud noise everywhere it was asked for audio: whatever of the
    source survives in the result, the server put back."""
    from music import audio

    def call(job):
        server._fake.jobs.append(job)
        files = []
        for s in job["seeds"]:
            x = np.random.default_rng(s).standard_normal((2, int(job["duration"] * SR))).astype(np.float32) * 0.3
            p = Path(job["out_dir"]) / f"{job['stem']}_{s}.wav"
            audio.write_wav(p, x, SR, subtype="FLOAT")
            files.append({"seed": s, "path": str(p)})
        return {"ok": True, "files": files, "sample_rate": SR}
    monkeypatch.setattr(server._fake, "call", call)


def test_inpaint_keeps_everything_outside_the_span(client, server, monkeypatch):
    src = gen(client, n=1, duration=20, fade_out=0)["takes"][0]["ref"]
    x, _ = raw_of(server, src)
    noisy_engine(server, monkeypatch)
    res = client.post("/api/inpaint", json={"take": src, "start": 8, "end": 12, "n": 1}).json()
    y, _ = raw_of(server, res["takes"][0]["ref"])
    xf = int(0.1 * SR)
    assert y.shape == x.shape
    assert np.array_equal(y[:, :8 * SR - xf], x[:, :8 * SR - xf]) and np.array_equal(y[:, 12 * SR + xf:], x[:, 12 * SR + xf:])
    assert not np.allclose(y[:, 9 * SR:11 * SR], x[:, 9 * SR:11 * SR])


def test_extend_keeps_the_source_in_front(client, server, monkeypatch):
    src = gen(client, n=1, duration=15)["takes"][0]["ref"]
    x, _ = raw_of(server, src)
    noisy_engine(server, monkeypatch)
    res = client.post("/api/extend", json={"take": src, "seconds": 5, "n": 1}).json()
    y, _ = raw_of(server, res["takes"][0]["ref"])
    xf = int(0.1 * SR)
    assert y.shape[1] == x.shape[1] + 5 * SR
    assert np.array_equal(y[:, :x.shape[1] - xf], x[:, :x.shape[1] - xf])
    # past the window: the head and the context are both the source
    server._fake.info["max_duration"] = 18.0  # hears 13 s of the 15 s source
    res = client.post("/api/extend", json={"take": src, "seconds": 5, "n": 1}).json()
    y, _ = raw_of(server, res["takes"][0]["ref"])
    assert server._fake.jobs[-1]["duration"] == pytest.approx(18.0, abs=0.01)
    assert y.shape[1] == x.shape[1] + 5 * SR and np.array_equal(y[:, :x.shape[1] - xf], x[:, :x.shape[1] - xf])


def test_arrange_keeps_the_parts_and_writes_only_the_joins(client, server, monkeypatch):
    a = gen(client, n=1, duration=20, fade_out=0)["takes"][0]["ref"]
    b = gen(client, n=1, duration=20, fade_out=0)["takes"][0]["ref"]
    xa, ta = raw_of(server, a)
    xb, tb = raw_of(server, b)
    xa = xa * 10 ** (ta["post"]["gain_db"] / 20)  # each part joins at its delivered loudness
    xb = xb * 10 ** (tb["post"]["gain_db"] / 20)
    noisy_engine(server, monkeypatch)
    res = client.post("/api/arrange", json={"parts": [{"take": a, "start": 0, "end": 20},
                                                     {"take": b, "start": 0, "end": 20}],
                                           "bridge": 2, "blend": 1, "n": 1}).json()
    y, _ = raw_of(server, res["takes"][0]["ref"])
    xf = int(0.1 * SR)
    assert y.shape[1] == 42 * SR
    cut = 19 * SR - xf  # part a is itself up to its blend (1 s before its end) minus the crossfade
    assert np.allclose(y[:, :cut], xa[:, :cut], atol=1e-6)
    tail = 42 * SR - (23 * SR + xf)  # part b is itself from 1 s past the bridge (+ crossfade)
    assert np.allclose(y[:, -tail:], xb[:, -tail:], atol=1e-6)
    assert not np.allclose(y[:, 20 * SR:22 * SR], 0)  # the bridge is the engine's


def test_spans_follow_the_player_not_the_raw_take(client, server):
    x = np.concatenate([np.zeros((2, int(1.5 * 48000)), np.float32), band(100, 12.0, sr=48000)], axis=1)
    imp = client.post("/api/import", params={"filename": "late.wav"}, content=wav_bytes(x, 48000)).json()
    t = imp["takes"][0]
    assert t["post"]["trim_start_s"] == pytest.approx(1.5, abs=0.12)
    client.post("/api/inpaint", json={"take": t["ref"], "start": 2, "end": 3, "n": 1})
    job = server._fake.jobs[-1]
    off = t["post"]["trim_start_s"]
    assert job["mask_start"] == pytest.approx(2 + off) and job["mask_end"] == pytest.approx(3 + off)


def test_spans_on_a_loop_are_refused(client):
    res = gen(client, n=1, duration=30, bpm=120)
    client.post("/api/master", json={"request": res["id"], "loop_bars": 4})
    r = client.post("/api/inpaint", json={"take": res["takes"][0]["ref"], "start": 1, "end": 2})
    assert r.status_code == 400 and "loop" in r.json()["detail"]


def test_a_failed_remaster_changes_nothing(client, server):
    res = gen(client, n=2, duration=12, bpm=120)
    d = server.OUTPUT / res["id"]
    rec = json.loads((d / "request.json").read_text(encoding="utf-8"))
    rec["takes"][1]["beats"] = []  # this take can't be looped
    (d / "request.json").write_text(json.dumps(rec), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in d.iterdir() if p.suffix in (".mp3", ".wav", ".png", ".json")}
    r = client.post("/api/master", json={"request": res["id"], "loop_bars": 2})
    assert r.status_code == 400
    after = {p.name: p.read_bytes() for p in d.iterdir() if p.suffix in (".mp3", ".wav", ".png", ".json")}
    assert after == before


def test_odd_imports_are_4xx_or_rejected_not_500(client):
    nan = band(100, 3.0, sr=48000)
    nan[0, 100] = np.nan
    import io

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, nan.T, 48000, format="WAV", subtype="FLOAT")
    assert client.post("/api/import", params={"filename": "nan.wav"}, content=buf.getvalue()).status_code == 400
    quiet = np.zeros((1, 3 * 48000), np.float32)
    quiet[0, 1000] = 0.004  # one click at -48 dBFS
    r = client.post("/api/import", params={"filename": "quiet.wav"}, content=wav_bytes(quiet, 48000))
    assert r.status_code == 200 and r.json()["takes"][0]["rejected"]


def test_a_mix_span_too_short_is_400(client):
    a = gen(client, n=1, duration=10)["takes"][0]["ref"]
    r = client.post("/api/mix", json={"layers": [{"take": a}, {"take": a, "start": 1.0, "end": 1.00001}]})
    assert r.status_code == 400


def test_arrange_joins_parts_at_their_delivered_loudness(client, server):
    from music import analysis
    a = gen(client, n=1, duration=20, fade_out=0)["takes"][0]["ref"]
    quiet = band(110, 20.0, sr=48000) * 0.01  # about -40 dB under the generated part
    q = client.post("/api/import", params={"filename": "quiet.wav"}, content=wav_bytes(quiet, 48000)).json()
    res = client.post("/api/arrange", json={"parts": [{"take": a, "start": 0, "end": 20},
                                                     {"take": q["takes"][0]["ref"], "start": 0, "end": 20}],
                                           "bridge": 2, "n": 1}).json()
    y, _ = raw_of(server, res["takes"][0]["ref"])
    first = analysis.lufs(y[:, 2 * SR:16 * SR], SR)
    second = analysis.lufs(y[:, 26 * SR:40 * SR], SR)
    assert abs(first - second) < 3.0  # was ~30 dB apart before


def test_extend_continues_from_the_music_not_its_trailing_silence(client, server):
    x = np.concatenate([band(100, 12.0, sr=48000), np.zeros((2, 6 * 48000), np.float32)], axis=1)
    imp = client.post("/api/import", params={"filename": "ends-early.wav"}, content=wav_bytes(x, 48000)).json()
    res = client.post("/api/extend", json={"take": imp["takes"][0]["ref"], "seconds": 10, "n": 1}).json()
    job = server._fake.jobs[-1]
    assert job["mask_start"] < 12.5  # the silence isn't part of what it continues
    assert any("left out" in w for w in res.get("warnings", []))
    assert res["takes"][0]["signal"]["duration_s"] < 23.0


def test_arrange_records_spans_as_the_take_plays(client):
    x = np.concatenate([np.zeros((2, 3 * 48000), np.float32), band(100, 20.0, sr=48000)], axis=1)
    imp = client.post("/api/import", params={"filename": "late.wav"}, content=wav_bytes(x, 48000)).json()
    a = gen(client, n=1, duration=20)["takes"][0]["ref"]
    res = client.post("/api/arrange", json={"parts": [{"take": imp["takes"][0]["ref"], "start": 2, "end": 14},
                                                     {"take": a}], "n": 1}).json()
    assert (res["parts"][0]["start"], res["parts"][0]["end"]) == (pytest.approx(2.0, abs=0.01), pytest.approx(14.0, abs=0.01))


# ------------------------------------------------------------------ mixing: layers before their limiter, edits, the voice check

def voice_import(client, seconds=8.0, loud_end=False, trim=True):
    """A 'voice': 1 kHz syllables (in the speech band) after 1 s of silence."""
    n = int(seconds * 48000)
    t = np.arange(n) / 48000
    x = (0.1 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 3 * t) > -0.6)).astype(np.float32)[None, :]
    x[:, :48000] = 0.0
    if loud_end:  # sharp, loud peaks in the last second (high crest, little loudness): what a limiter flattens
        for k in range(10):
            i = n - 48000 + k * 4800
            x[:, i:i + 480] = 0.9
    params = {"filename": "voice.wav"} if trim else {"filename": "voice.wav", "trim": "false"}
    r = client.post("/api/import", params=params, content=wav_bytes(x, 48000))
    assert r.status_code == 200, r.text
    return r.json()


def test_a_layer_joins_the_mix_before_its_own_limiter(client, server):
    from music import audio
    v = voice_import(client, loud_end=True)
    t = v["takes"][0]
    assert t["post"]["limiter_db"] > 3  # delivered on its own, the loud end is limited
    song = gen(client, n=1, duration=12)["takes"][0]["ref"]
    res = client.post("/api/mix", json={"layers": [{"take": song, "gain_db": -40}, {"take": t["ref"]}],
                                        "lufs": -30}).json()  # quiet enough that the mix itself isn't limited
    assert res["takes"][0]["post"]["limiter_db"] == 0.0
    y, _ = audio.read(server.OUTPUT / res["id"] / "0.wav")
    delivered, _ = audio.read(server.OUTPUT / v["id"] / "0.wav")
    n = delivered.shape[1]  # the voice starts the mix: its last second is the mix's n-48000..n
    rise = lambda z: 20 * np.log10(np.max(np.abs(z[:, n - 48000:n])) / np.max(np.abs(z[:, 48000:4 * 48000])))  # noqa: E731
    assert rise(y) > rise(delivered) + 3  # the peaks come through as they were, not as the import's limiter left them
    assert rise(y) == pytest.approx(20 * np.log10(0.9 / 0.1), abs=1.0)
    ly = res["layers"][1]
    assert ly["lufs"] == pytest.approx(t["final"]["lufs"], abs=0.5)  # how loud it plays (its target, before its limiter)
    # an excerpt plays as loud as it is, not as the whole take: its quiet half
    x = (0.3 * np.sin(2 * np.pi * 1000 * np.arange(8 * 48000) / 48000)).astype(np.float32)[None, :]
    x[:, :4 * 48000] *= 0.1  # -20 dB for the first 4 s
    two = client.post("/api/import", params={"filename": "two.wav"}, content=wav_bytes(x, 48000)).json()["takes"][0]
    whole = client.post("/api/mix", json={"layers": [{"take": song, "gain_db": -40}, {"take": two["ref"]}]}).json()
    part = client.post("/api/mix", json={"layers": [{"take": song, "gain_db": -40},
                                                    {"take": two["ref"], "start": 0.5, "end": 3.5}]}).json()
    assert part["layers"][1]["lufs"] < whole["layers"][1]["lufs"] - 10


def test_mix_layer_spans_reverb_and_the_voice_check(client, server):
    v = voice_import(client, seconds=10.0)["takes"][0]["ref"]
    song = gen(client, n=1, duration=12)["takes"][0]["ref"]
    body = {"layers": [{"take": song, "gain_db": 6}, {"take": v, "voice": True, "reverb": "room",
                                                        "spans": [{"start": 2, "end": 4, "gain_db": 4}]}]}
    res = client.post("/api/mix", json=body).json()
    ly = res["layers"][1]
    assert ly["reverb"] == "room" and ly["spans"] == [{"start": 2.0, "end": 4.0, "gain_db": 4.0}] and ly["voice"]
    m = ly["speech_margin"]
    assert m["median_db"] is not None and "speech_margin" not in res["layers"][0]
    loud = any("voice layer" in w for w in res["warnings"])
    body["layers"][0]["gain_db"] = -30
    quiet = client.post("/api/mix", json=body).json()
    assert quiet["layers"][1]["speech_margin"]["median_db"] > m["median_db"] + 20
    assert not any("voice layer" in w for w in quiet["warnings"]) and loud
    body["layers"][1]["spans"] = [{"start": 50, "end": 60, "gain_db": 4}]
    assert client.post("/api/mix", json=body).status_code == 400  # outside the layer
    body["layers"][1]["spans"] = [{"start": 4, "end": 2, "gain_db": 4}]
    assert client.post("/api/mix", json=body).status_code == 422
    body["layers"][1]["spans"], body["layers"][1]["reverb"] = [], "garage"
    assert client.post("/api/mix", json=body).status_code == 422


def test_import_can_keep_the_files_timeline(client):
    trimmed = voice_import(client)["takes"][0]
    kept = voice_import(client, trim=False)["takes"][0]
    assert trimmed["post"]["trim_start_s"] == pytest.approx(1.0, abs=0.12)
    assert kept["post"].get("trim_start_s", 0.0) == 0.0
    assert kept["signal"]["duration_s"] == pytest.approx(8.0, abs=0.01)


def test_arrange_part_volumes(client, server):
    from music import analysis
    a = gen(client, n=1, duration=20, fade_out=0)["takes"][0]["ref"]
    b = gen(client, n=1, duration=20, fade_out=0)["takes"][0]["ref"]
    res = client.post("/api/arrange", json={"parts": [{"take": a, "start": 0, "end": 20, "gain_db": -8},
                                                     {"take": b, "start": 0, "end": 20}], "bridge": 2, "n": 1}).json()
    assert res["parts"][0]["gain_db"] == -8 and "gain_db" not in res["parts"][1]
    y, _ = raw_of(server, res["takes"][0]["ref"])
    first, second = analysis.lufs(y[:, 2 * SR:16 * SR], SR), analysis.lufs(y[:, 26 * SR:40 * SR], SR)
    assert second - first == pytest.approx(8.0, abs=1.5)


def test_a_request_others_are_made_from_isnt_deleted_unless_forced(client, server):
    src = gen(client, n=1, duration=12)
    a = src["takes"][0]["ref"]
    mixed = client.post("/api/mix", json={"layers": [{"take": a}, {"take": a, "at": 2}]}).json()
    ext = client.post("/api/extend", json={"take": a, "seconds": 5, "n": 1}).json()
    r = client.delete(f"/api/requests/{src['id']}")
    assert r.status_code == 409 and mixed["id"] in r.json()["detail"] and ext["id"] in r.json()["detail"]
    assert (server.OUTPUT / src["id"]).is_dir()
    assert client.delete(f"/api/requests/{mixed['id']}").status_code == 200  # nothing is made from the mix
    assert client.delete(f"/api/requests/{src['id']}?force=true").status_code == 200
    assert not (server.OUTPUT / src["id"]).exists()


def test_names_may_have_underscores(client):
    assert gen(client, n=1, name="boss_theme-v2")["name"] == "boss_theme-v2"
    assert client.post("/api/generate", json={"text": "x", "name": "no spaces"}).status_code == 422



def test_a_mix_keeps_its_timeline_and_spans_land_where_asked(client, server):
    from music import audio
    v = voice_import(client, seconds=8.0, trim=False)["takes"][0]  # 1 s of silence, then the voice
    song = gen(client, n=1, duration=12)["takes"][0]["ref"]
    body = {"layers": [{"take": song, "gain_db": -60, "at": 3}, {"take": v["ref"], "at": 2, "start": 1.0,
                                                                  "spans": [{"start": 4, "end": 5, "gain_db": 12}]}],
            "lufs": -30, "fade_out": 0}
    res = client.post("/api/mix", json=body).json()
    assert res["takes"][0]["post"].get("trim_start_s", 0.0) == 0.0  # every layer starts late: nothing trimmed
    y, _ = audio.read(server.OUTPUT / res["id"] / "0.wav")
    rms = lambda a, b: np.sqrt(np.mean(y[:, int(a * 48000):int(b * 48000)] ** 2))  # noqa: E731
    # the voice's 4-5 s (as the take plays) sits at 2 + (4 - 1) = 5..6 s in the mix, 12 dB up
    assert 20 * np.log10(rms(5.1, 5.9) / rms(3.1, 3.9)) == pytest.approx(12.0, abs=1.5)
    assert rms(0.0, 1.9) < 1e-4  # silence until the first layer starts
