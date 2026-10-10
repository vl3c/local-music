r"""Mastering, fades, trimming and seamless loops.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import numpy as np
import pytest
from conftest import SR, band

from music import analysis, post
from music.audio import SR as OUT_SR


def test_master_hits_the_loudness_target_under_the_ceiling():
    x = band(120, 12.0) * 3.0  # hot: needs limiting to reach -9 under -1 dBTP
    for target in (-16.0, -14.0, -9.0):
        y, rep = post.master(x, SR, target, -1.0)
        assert analysis.true_peak_db(y, SR) <= -1.0 + 0.05
        assert abs(analysis.lufs(y, SR) - target) < 1.0, rep


def test_limiter_leaves_quiet_audio_alone():
    x = band(120, 4.0) * 0.1
    y, gr = post.limit(x, SR, -1.0)
    assert gr == 0.0 and np.array_equal(y, x)


def test_limiter_catches_a_lone_peak_before_it_arrives():
    x = np.zeros((2, SR), np.float32)
    x[:, SR // 2] = 1.0
    y, gr = post.limit(x, SR, -6.0)
    assert np.max(np.abs(y)) <= 10 ** (-6 / 20) + 1e-4
    assert gr == pytest.approx(6.0, abs=0.01)


def test_fades_are_at_least_anti_click():
    x = np.ones((2, SR), np.float32)
    y = post.fade(x, SR, 0.0, 0.0)
    assert y[0, 0] == 0.0 and y[0, -1] == pytest.approx(0.0, abs=1e-6) and y[0, SR // 2] == 1.0
    z = post.fade(x, SR, 0.5, 0.25)
    assert z[0, int(0.25 * SR)] < 0.8 and z[0, -int(0.1 * SR)] < 0.7


def test_trim_silence_cuts_only_real_gaps():
    x = band(120, 6.0)
    lead, tail = np.zeros((2, SR), np.float32), np.zeros((2, 3 * SR), np.float32)
    y, rep = post.trim_silence(np.concatenate([lead, x, tail], axis=1), SR)
    assert 0.9 < rep["trim_start_s"] < 1.0 and 2.0 < rep["trim_end_s"] < 2.6
    short = np.concatenate([x, np.zeros((2, SR // 2), np.float32)], axis=1)
    _, rep2 = post.trim_silence(short, SR)
    assert rep2 == {"trim_start_s": 0.0, "trim_end_s": 0.0}


def test_process_fades_an_abrupt_stop_automatically():
    x = band(120, 20.0)
    y, rep = post.process(x, SR, post.PostOpts())
    assert rep["auto_fade_out_s"] == pytest.approx(2.0)  # min(3 s, a tenth of 20 s)
    assert analysis.ending(y, OUT_SR)["end_db"] < -10
    y0, rep0 = post.process(x, SR, post.PostOpts(fade_out=0.0))  # 0 = keep the cut
    assert "auto_fade_out_s" not in rep0 and rep0["fade_out_s"] == 0.0


def test_process_cuts_to_length_with_a_fade():
    y, rep = post.process(band(120, 20.0), SR, post.PostOpts(duration=10.0))
    assert rep["cut_at_s"] == 10.0 and rep["length_s"] == pytest.approx(10.0, abs=0.01)
    assert rep["fade_out_s"] >= 2.0


@pytest.mark.parametrize("bpm,bars", [(120.0, 4), (90.0, 4), (150.0, 8)])  # whole 4-bar progressions
def test_loops_are_exactly_n_bars_and_seamless(bpm, bars):
    x = band(bpm, 40.0)
    beats = analysis.tempo_and_beats(x, SR, bpm)["beats"]
    y, rep = post.process(x, SR, post.PostOpts(loop_bars=bars, bpm=bpm), beats)
    want = bars * 4 * 60.0 / bpm
    assert rep["length_s"] == pytest.approx(want, abs=0.02)
    assert rep["loop"]["bpm"] == pytest.approx(bpm, abs=0.5)
    assert rep["seam_ok"] and rep["loop"]["seam_similarity"] > 0.9
    # Played twice in a row, the wrap is as smooth as anywhere inside the loop.
    twice = np.concatenate([y, y], axis=1).mean(axis=0)
    step = np.abs(np.diff(twice))
    assert step[y.shape[1] - 1] < np.percentile(step, 99.9) + 1e-6


def test_half_a_progression_is_flagged_as_a_noticeable_repeat():
    # The chords change every bar over 4 bars: a 2-bar loop's end is followed by other chords
    # than its start, and the similarity says so (the server warns under 0.85).
    x = band(90, 40.0)
    beats = analysis.tempo_and_beats(x, SR, 90)["beats"]
    _, rep = post.process(x, SR, post.PostOpts(loop_bars=2, bpm=90), beats)
    assert rep["loop"]["seam_similarity"] < 0.85


def test_a_loop_longer_than_the_take_says_so():
    x = band(120, 10.0)
    beats = analysis.tempo_and_beats(x, SR, 120)["beats"]
    with pytest.raises(ValueError, match="too short"):
        post.process(x, SR, post.PostOpts(loop_bars=16), beats)


def test_no_beat_no_loop():
    with pytest.raises(ValueError, match="no steady beat"):
        post.process(band(120, 10.0), SR, post.PostOpts(loop_bars=2), [])


def test_cut_loop_crossfades_the_continuation_into_the_start():
    x = np.vstack([np.arange(10000, dtype=np.float32)] * 2) / 10000
    y = post.cut_loop(x, 1000, start=1000, length=4000, xfade_s=0.1)
    assert y.shape == (2, 4000)
    assert y[0, 0] == pytest.approx(x[0, 5000])      # the wrap continues where the end left off
    assert y[0, 99] == pytest.approx(x[0, 1099], abs=0.02)  # and arrives at the loop's own start


@pytest.mark.parametrize("bad", [dict(duration=0), dict(fade_in=-1), dict(fade_out=-1), dict(loop_bars=0),
                                 dict(beats_per_bar=0)])
def test_options_are_validated(bad):
    with pytest.raises(ValueError):
        post.PostOpts(**bad)


def test_mix_places_and_scales_layers():
    sr = 1000
    a = np.ones((2, 5 * sr), np.float32)
    b = np.ones((1, 2 * sr), np.float32)
    m, rep = post.mix_layers([post.Layer(a, gain_db=-6.0206), post.Layer(b, offset_s=4)], sr)
    assert m.shape == (2, 6 * sr) and rep["length_s"] == 6.0
    assert m[0, 100] == pytest.approx(0.5, abs=1e-3) and m[1, 4500] == pytest.approx(1.5, abs=1e-3)
    assert m[0, 5500] == pytest.approx(1.0)


def test_keep_outside_puts_the_original_back_around_the_new_part():
    n = 10000
    x = np.vstack([np.linspace(0, 1, n, dtype=np.float32)] * 2)
    r = np.full((2, n), 5.0, np.float32)  # "the engine": different everywhere
    out = post.keep_outside(x, r, 4000, 6000, 100)
    assert np.array_equal(out[:, :3900], x[:, :3900]) and np.array_equal(out[:, 6100:], x[:, 6100:])
    assert np.all(out[:, 4000:6000] == 5.0)
    assert 0 < out[0, 3950] - x[0, 3950] < 5  # crossfading in...
    # identical signals across a linear crossfade stay identical (no +3 dB bump)
    same = post.keep_outside(x, x.copy(), 4000, 6000, 100)
    assert np.allclose(same, x, atol=1e-6)


def test_keep_outside_extends_where_the_original_ends():
    x = np.ones((2, 1000), np.float32)
    r = np.full((2, 1500), 2.0, np.float32)
    out = post.keep_outside(x, r, 1000, 1500, 50)  # an extension: the mask runs to the end
    assert out.shape == (2, 1500) and np.all(out[:, :950] == 1.0) and np.all(out[:, 1000:] == 2.0)
    assert 1.0 < out[0, 975] < 2.0


# ------------------------------------------------------------------ mix layers

def test_a_layer_can_be_mastered_without_its_limiter():
    x = band(120, 8.0) * 3.0  # at -9 LUFS the limiter takes several dB off its peaks
    limited, rl = post.master(x, SR, -9.0, -1.0)
    free, rf = post.master(x, SR, -9.0, -1.0, limit_peaks=False)
    assert rl["limiter_db"] > 1.0 and rf["limiter_db"] == 0.0
    assert abs(analysis.lufs(free, SR) - -9.0) < 0.1  # the same loudness...
    assert np.max(np.abs(free)) > np.max(np.abs(limited))  # ...with its peaks intact
    assert np.allclose(free, x * 10 ** (rf["gain_db"] / 20), rtol=2e-3, atol=1e-6)  # nothing but the gain (rounded)


def test_gain_spans_ramp_in_and_add_up():
    x = np.ones((2, 4 * SR), np.float32)
    y = post.gain_spans(x, SR, [(1.0, 2.0, 6.0), (1.5, 3.0, -6.0)])
    assert y[0, int(0.5 * SR)] == 1.0 and y[0, int(3.5 * SR)] == 1.0  # untouched outside
    assert y[0, int(1.25 * SR)] == pytest.approx(10 ** (6 / 20), rel=1e-4)
    assert y[0, int(1.75 * SR)] == pytest.approx(1.0, rel=1e-4)  # +6 and -6 overlap
    assert y[0, int(2.5 * SR)] == pytest.approx(10 ** (-6 / 20), rel=1e-4)
    assert y[0, SR] == pytest.approx(1.0, abs=1e-3)  # the ramp starts at the span's edge: no step
    assert np.array_equal(post.gain_spans(x, SR, []), x)


def test_reverb_adds_a_tail_at_the_same_loudness():
    x = np.zeros((1, 3 * SR), np.float32)
    x[0, :SR] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
    for kind, rt60 in (("room", 0.6), ("cathedral", 3.5)):
        y = post.add_reverb(x, SR, kind)
        assert y.shape[0] == 2 and y.shape[1] > x.shape[1]
        assert abs(analysis.lufs(y, SR) - analysis.lufs(np.vstack([x, x]), SR)) < 0.3  # as loud as it was, in stereo
        tail = 20 * np.log10(np.sqrt(np.mean(y[:, SR + int(0.2 * SR):SR + int(0.4 * SR)] ** 2)) + 1e-12)
        assert tail > -60  # it rings on after the sound stops
    assert np.array_equal(post.reverb_ir("room", SR), post.reverb_ir("room", SR))  # the same room every time


def test_speech_margin_finds_where_a_voice_is_buried():
    n = 12 * SR
    t = np.arange(n) / SR
    voice = np.zeros((2, n), np.float32)
    voice[:, :] = 0.2 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 3 * t) > -0.6)  # syllables, 1 kHz
    music = np.zeros((2, n), np.float32)
    rng = np.random.default_rng(0)
    music[:, 6 * SR:9 * SR] = rng.standard_normal((2, 3 * SR)) * 0.5  # a loud burst at 6-9 s
    m = post.speech_margin(voice, music, SR)
    assert m["median_db"] > 20
    assert len(m["under"]) == 1
    a, b, worst = m["under"][0]
    assert 5.5 <= a <= 6.5 and 8.5 <= b <= 9.5 and worst < 0
    quiet = post.speech_margin(voice, music * 0.01, SR)
    assert quiet["under"] == [] and quiet["p10_db"] > post.MARGIN_P10_WARN_DB



def test_speech_margin_of_a_short_voice_in_a_long_mix():
    n = 120 * SR
    voice = np.zeros((2, n), np.float32)
    t = np.arange(4 * SR) / SR
    voice[:, 10 * SR:14 * SR] = 0.2 * np.sin(2 * np.pi * 1000 * t)  # 4 s of a 2-minute mix
    rest = np.random.default_rng(0).standard_normal((2, n)).astype(np.float32) * 0.002
    m = post.speech_margin(voice, rest, SR)
    assert m["median_db"] > 20 and m["under"] == []  # judged where it speaks, not where it's silent
    assert post.speech_margin(np.zeros((2, n), np.float32), rest, SR)["median_db"] is None
