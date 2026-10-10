r"""Signal and musical checks, prompts, and the WAV / MP3 files.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import numpy as np
import pytest
import soundfile as sf
from conftest import A_MINOR, C_MAJOR, SR, band

from music import audio, checks, prompts
from music.scoring import rank_score, windows


# ------------------------------------------------------------------ signal

def test_clean_music_passes():
    rep = checks.signal_report(band(120, 10.0), SR, 10.0)
    assert rep["fails"] == [] and rep["warnings"] == []
    assert rep["dropouts"] == [] and rep["clipped_pct"] == 0.0 and rep["abrupt_end"]


def test_silence_fails():
    rep = checks.signal_report(np.zeros((2, SR * 3), np.float32), SR)
    assert rep["fails"] and "silent" in rep["fails"][0]


def test_clipping_is_flat_tops_not_loudness():
    x = band(120, 6.0)
    loud = x / np.max(np.abs(x)) * 1.4          # over full scale (the float raw can be) but not clipped
    assert checks.clipped_fraction(loud) == 0.0
    clipped = np.clip(x * 8, -1, 1)
    assert checks.clipped_fraction(clipped) > 0.01
    assert any("clipped" in f for f in checks.signal_report(clipped, SR)["fails"])


def test_dropouts_inside_the_music_are_found():
    x = band(120, 12.0)
    x[:, 5 * SR:7 * SR] = 0.0
    rep = checks.signal_report(x, SR)
    assert len(rep["dropouts"]) == 1 and 4.9 < rep["dropouts"][0][0] < 5.2
    assert any("drops to silence" in w for w in rep["warnings"])


def test_an_early_end_and_a_late_start_are_reported():
    x = band(120, 10.0)
    padded = np.concatenate([np.zeros((2, 2 * SR), np.float32), x, np.zeros((2, 4 * SR), np.float32)], axis=1)
    rep = checks.signal_report(padded, SR, 16.0)
    assert any("ends at" in w for w in rep["warnings"]) and any("before the music starts" in w for w in rep["warnings"])


def test_out_of_phase_stereo_warns():
    x = band(120, 6.0)
    x[1] = -x[0]
    assert any("out of phase" in w for w in checks.signal_report(x, SR)["warnings"])


# ------------------------------------------------------------------ musical

def test_tempo_and_key_as_asked():
    rep = checks.musical_report(band(128, 16.0, A_MINOR), SR, bpm=128, key="A minor")
    assert rep["tempo"]["match"] == "exact" and rep["warnings"] == []
    assert rep["key"]["match"] in ("exact", "relative")
    assert len(rep["beat_times"]) > 30


def test_wrong_tempo_warns():
    # (Half and double time are the same grid: tempo_matches covers them. The hint steers the
    # tracker towards the asked level when the music allows both, which this one does.)
    rep = checks.musical_report(band(100, 16.0, C_MAJOR), SR, bpm=140)
    assert rep["tempo"]["match"] is None and any("not the 140" in w for w in rep["warnings"])


def test_wrong_key_warns():
    rep = checks.musical_report(band(110, 16.0, C_MAJOR), SR, key="F# major")
    assert rep["key"]["match"] is None and any("not the F# major" in w for w in rep["warnings"])


# ------------------------------------------------------------------ prompts

def test_prompt_gets_the_guide_tags_tempo_and_key():
    p = prompts.build_prompt("Warm lo-fi hip hop, dusty piano.", bpm=84, key="A minor")
    assert p == "TrackType: Music, VocalType: Instrumental, Warm lo-fi hip hop, dusty piano, in A minor, 84 BPM"
    assert prompts.build_prompt("solo cello", kind="stem").startswith("TrackType: Instrument, solo cello")


def test_prompt_doesnt_repeat_what_the_text_says():
    p = prompts.build_prompt("TrackType: Music, techno at 130 BPM in F minor", bpm=130, key="F minor")
    assert p.count("TrackType") == 1 and p.count("BPM") == 1 and p.count("F minor") == 1


def test_raw_prompt_is_sent_as_written():
    assert prompts.build_prompt("  just   this ", bpm=90, key="C major", raw=True) == "just this"
    with pytest.raises(ValueError):
        prompts.build_prompt("x", kind="vocals")


def test_describe_strips_the_tags_for_clap():
    p = "TrackType: Music, VocalType: Instrumental, Warm lo-fi hip hop, 84 BPM"
    assert prompts.describe(p) == "Warm lo-fi hip hop, 84 BPM"
    assert prompts.describe("TrackType: Instrument") == "TrackType: Instrument"  # nothing else left: keep it


def test_track_names():
    assert prompts.track_name("A warm lo-fi hip hop beat with dusty piano, 84 BPM") == "warm-lofi-hip-hop"
    assert prompts.track_name("8-bit chiptune boss battle") == "8bit-chiptune-boss-battle"
    assert prompts.track_name("!!!") == "track"


# ------------------------------------------------------------------ scores

def test_rank_score_orders_by_quality_and_match():
    good = {"aes": {"CE": 7.8, "PQ": 8.2}, "clap": 0.5, "warnings": []}
    meh = {"aes": {"CE": 6.0, "PQ": 6.2}, "clap": 0.3, "warnings": ["x"]}
    failed = {**good, "fails": ["silent"]}
    assert rank_score(good) > rank_score(meh) > rank_score(failed) == -1.0
    assert 0 < rank_score({"warnings": []}) < rank_score(good)  # no scores: neutral, not zero


def test_clap_windows_cover_the_track_evenly():
    x = np.arange(48000 * 95, dtype=np.float32)[None]
    w = windows(x, 48000)
    assert len(w) == 6 and all(len(v) == 480000 for v in w)
    assert w[0][0] == 0 and w[-1][-1] == x[0, -1]
    assert len(windows(x[:, :48000 * 4], 48000)) == 1


# ------------------------------------------------------------------ files

def test_wav_info_round_trip(tmp_path):
    x = band(120, 2.0)
    p = audio.write_wav(tmp_path / "t.wav", x, SR, info={"title": "warm-lofi", "comment": "prompt | seed 1"})
    info = audio.read_info(p)
    assert info["title"] == "warm-lofi" and info["comment"] == "prompt | seed 1"
    y, sr = audio.read(p)
    assert sr == SR and y.shape == x.shape and np.max(np.abs(y - x)) < 1e-4


def test_mp3_is_stereo_and_about_320_kbps(tmp_path):
    x = band(120, 4.0, sr=48000)
    p = audio.write_mp3(tmp_path / "t.mp3", x)
    info = sf.info(str(p))
    assert info.channels == 2 and info.samplerate == 48000
    assert 280 < p.stat().st_size * 8 / 4.0 / 1000 < 360


def test_float_wavs_keep_peaks_over_full_scale(tmp_path):
    x = band(120, 1.0) * 6  # peaks around 1.5
    y, _ = audio.read(audio.write_wav(tmp_path / "f.wav", x, SR, subtype="FLOAT"))
    assert np.max(np.abs(y)) > 1.2 and checks.clipped_fraction(y) == 0.0
    z, _ = audio.read(audio.write_wav(tmp_path / "i.wav", x, SR))  # 24-bit: clamped, not wrapped
    assert np.max(np.abs(z)) <= 1.0
