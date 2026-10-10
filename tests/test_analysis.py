r"""Tempo, beats, key, loudness and the ending measurement, on synthetic music.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import numpy as np
import pytest
from conftest import A_MINOR, C_MAJOR, SR, band

from music import analysis


@pytest.mark.parametrize("bpm", [84.0, 126.0, 160.0])
def test_tempo_is_measured_to_a_fraction_of_a_bpm(bpm):
    t = analysis.tempo_and_beats(band(bpm, 20.0), SR, bpm_hint=bpm)
    assert abs(t["bpm"] - bpm) < 0.6  # single beat intervals are 23 ms frames apart: the fit isn't
    assert t["confidence"] > 0.8
    assert len(t["beats"]) > bpm / 60 * 20 * 0.8


def test_silence_has_no_tempo():
    t = analysis.tempo_and_beats(np.zeros((2, SR * 5), np.float32), SR)
    assert t["bpm"] is None and t["beats"] == []


@pytest.mark.parametrize("measured,wanted,expect", [
    (120.4, 120, "exact"), (60.2, 120, "half"), (241, 120, "double"), (100, 120, None), (None, 120, None)])
def test_tempo_matches(measured, wanted, expect):
    assert analysis.tempo_matches(measured, wanted) == expect


@pytest.mark.parametrize("chords,key", [(C_MAJOR, "C major"), (A_MINOR, "A minor")])
def test_key_of_a_clear_progression(chords, key):
    k = analysis.estimate_key(band(110, 16.0, chords), SR)
    want_pc, want_mode = analysis.parse_key(key)
    # A diatonic progression is ambiguous only between a key and its relative: either is right here.
    assert analysis.key_relation(k["pc"], k["mode"], want_pc, want_mode) in ("exact", "relative")
    assert k["strength"] > 0.5


@pytest.mark.parametrize("text,expect", [
    ("A minor", (9, "minor")), ("Am", (9, "minor")), ("f# min", (6, "minor")), ("Bb major", (10, "major")),
    ("Eb", (3, "major")), ("C", (0, "major")), ("Db Minor", (1, "minor")), (" g  MAJOR ", (7, "major"))])
def test_parse_key(text, expect):
    assert analysis.parse_key(text) == expect


@pytest.mark.parametrize("bad", ["H minor", "A dorian", "", "C##"])
def test_parse_key_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        analysis.parse_key(bad)


def test_key_relations():
    a_min, c_maj = (9, "minor"), (0, "major")
    assert analysis.key_relation(*a_min, *a_min) == "exact"
    assert analysis.key_relation(*c_maj, *a_min) == "relative"
    assert analysis.key_relation(*a_min, *c_maj) == "relative"
    assert analysis.key_relation(9, "major", 9, "minor") == "parallel"
    assert analysis.key_relation(7, "major", 0, "major") == "fifth"
    assert analysis.key_relation(1, "major", 0, "major") is None


def test_ending_tells_a_stop_from_a_fade():
    x = band(120, 10.0)
    stop = analysis.ending(x, SR)
    faded = x * np.linspace(1, 0, x.shape[1]) ** 3
    end = analysis.ending(faded.astype(np.float32), SR)
    assert stop["end_db"] > -6 and stop["tail_s"] < 0.1
    assert end["end_db"] < -20


def test_a_stop_followed_by_silence_is_still_a_stop():
    x = np.concatenate([band(120, 8.0), np.zeros((2, int(0.8 * SR)), np.float32)], axis=1)
    e = analysis.ending(x, SR)
    assert e["end_db"] > -6 and 0.6 < e["tail_s"] < 1.0


def test_loudness_and_short_term_loudness():
    x = band(120, 12.0)
    lufs = analysis.lufs(x, SR)
    t, st = analysis.short_term_lufs(x, SR, hop_s=1.0)
    assert len(t) == len(st) >= 9
    assert abs(float(np.median(st)) - lufs) < 2.0


def test_true_peak_is_at_least_the_sample_peak():
    x = band(120, 4.0)
    assert analysis.true_peak_db(x, SR) >= analysis.sample_peak_db(x) - 1e-6


@pytest.mark.parametrize("seconds,stereo", [(0.2, True), (3.0, False), (17.3, True)])
def test_loudness_matches_pyloudnorm(seconds, stereo):
    import pyloudnorm
    x = band(97, seconds, stereo=stereo) * np.linspace(0.2, 1, int(seconds * SR))  # a crescendo: gating matters
    want = analysis.lufs(x.astype(np.float32), SR)
    ref_in = x if x.shape[1] >= int(0.4 * SR) + 1 else np.tile(x, (1, int(0.4 * SR) // x.shape[1] + 2))
    ref = pyloudnorm.Meter(SR).integrated_loudness(ref_in.T.astype(np.float64))
    assert want == pytest.approx(ref, abs=0.02)
