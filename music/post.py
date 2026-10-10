"""From the engine's raw output to a delivered track: trim, loop, fade, master.

Everything here is deterministic numpy/scipy on (channels, samples) float32 arrays, so a
take can be re-mastered any number of times from its raw file without regenerating.

    master(raw, sr, opts) -> (48 kHz audio, report)

Mastering is loudness normalisation (ITU-R BS.1770 integrated loudness, -14 LUFS by
default: what streaming services play at) followed by a look-ahead peak limiter, with the
true peak checked afterwards and the gain trimmed if 4x oversampling still finds an overshoot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import analysis
from .audio import SR, as2d, resample

ANTI_CLICK_S = 0.005   # every track starts and ends with at least this fade
SILENCE_DB = -60.0     # quieter than this (short-term) counts as silence for trimming
LOOP_XFADE_S = 0.05    # the loop seam's crossfade
AUTO_FADE_S = 3.0      # fade_out=None: a take that stops at full level fades out over this


@dataclass
class PostOpts:
    duration: float | None = None        # cut to at most this long (seconds); None keeps it all
    fade_in: float = 0.0                 # seconds
    fade_out: float | None = None        # seconds; None = auto: AUTO_FADE_S if the take stops at
                                         # full level, else none. A cut (duration) gets >= CUT_FADE
    lufs: float = -14.0                  # integrated loudness target
    peak_dbtp: float = -1.0              # true-peak ceiling
    loop_bars: int | None = None         # make a seamless loop of this many bars
    beats_per_bar: int = 4
    bpm: float | None = None             # the tempo to loop at (default: the measured one)
    trim_silence: bool = True            # trim silence at the start and the end

    CUT_FADE = 2.0

    def __post_init__(self):
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.fade_in < 0 or (self.fade_out is not None and self.fade_out < 0):
            raise ValueError("fades can't be negative")
        if self.loop_bars is not None and not 1 <= self.loop_bars <= 64:
            raise ValueError("loop_bars must be 1..64")
        if not 1 <= self.beats_per_bar <= 12:
            raise ValueError("beats_per_bar must be 1..12")

    def asdict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- pieces

def fade(x: np.ndarray, sr: int, fade_in: float, fade_out: float) -> np.ndarray:
    """Equal-power (sine) fades; at least ANTI_CLICK_S at both ends."""
    x = as2d(x).copy()
    n = x.shape[1]
    for secs, at_start in ((max(fade_in, ANTI_CLICK_S), True), (max(fade_out, ANTI_CLICK_S), False)):
        k = min(n, int(round(secs * sr)))
        if k < 2:
            continue
        ramp = np.sin(np.linspace(0.0, np.pi / 2, k, dtype=np.float32))
        if at_start:
            x[:, :k] *= ramp
        else:
            x[:, n - k:] *= ramp[::-1]
    return x


def ends_abruptly(x: np.ndarray, sr: int) -> bool:
    """True when the music stops at full level instead of ending (its last half second within
    6 dB of its median level). The engine was trained on crops of longer tracks, so many takes do."""
    return analysis.ending(x, sr)["end_db"] > -6.0


def trim_silence(x: np.ndarray, sr: int, keep_s: float = 0.02, tail_keep_s: float = 0.5) -> tuple[np.ndarray, dict]:
    """Cut silence before the music starts and after it ends (the engine pads short pieces
    with silence when it finishes early). Keeps keep_s before the first sound and tail_keep_s
    after the last (a reverb tail fades out inside it)."""
    x = as2d(x)
    env = analysis.envelope_db(x, sr)  # 10 ms hops
    hop = max(1, int(round(analysis.HOP_S * sr)))
    loud = np.nonzero(env > max(SILENCE_DB, float(env.max()) - 60))[0]
    if not len(loud):
        return x, {"trim_start_s": 0.0, "trim_end_s": 0.0}
    a = max(0, int(loud[0]) * hop - int(keep_s * sr))
    b = min(x.shape[1], (int(loud[-1]) + 1) * hop + int(tail_keep_s * sr))
    # Only trim real gaps: a few tens of ms is the music's own attack or decay.
    if a < int(0.1 * sr):
        a = 0
    if x.shape[1] - b < int(1.0 * sr):
        b = x.shape[1]
    return x[:, a:b], {"trim_start_s": round(a / sr, 3), "trim_end_s": round((x.shape[1] - b) / sr, 3)}


def limit(x: np.ndarray, sr: int, ceiling_db: float, lookahead_s: float = 0.0015,
          release_s: float = 0.08) -> tuple[np.ndarray, float]:
    """Look-ahead peak limiter: the gain falls *before* each peak and recovers over release_s.

    The gain reduction a sample needs is held over +-lookahead (a max filter), then smoothed by
    a moving average of the same length, so the gain is fully down by the time the peak
    arrives; a second max + average pass of release_s makes the recovery gradual instead of
    pumping. Returns (audio, the largest reduction in dB).
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    x = as2d(x)
    ceil = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(x), axis=0).astype(np.float64)
    need = np.maximum(0.0, 20 * np.log10(np.maximum(peak, 1e-12) / ceil))  # dB per sample
    if not np.any(need > 0):
        return x, 0.0
    la = max(1, int(round(lookahead_s * sr)))
    gr = uniform_filter1d(maximum_filter1d(need, size=2 * la + 1), size=la)
    h = max(1, int(round(release_s * sr / 2)))
    # Release: hold each reduction for release_s after it (a centred max filter, shifted so
    # it only looks back), then smooth; the max with the attack curve keeps the peaks covered.
    c = maximum_filter1d(gr, size=2 * h + 1)
    held = np.concatenate([np.full(h, c[0]), c[:-h]]) if len(c) > h else c
    gr = np.maximum(gr, uniform_filter1d(held, size=2 * h + 1))
    y = (x * (10 ** (-gr / 20))[None, :]).astype(np.float32)
    return y, float(gr.max())


def master(x: np.ndarray, sr: int, lufs: float, peak_dbtp: float, limit_peaks: bool = True) -> tuple[np.ndarray, dict]:
    """Normalise to `lufs` integrated loudness, then limit to the `peak_dbtp` true-peak ceiling.

    Limiting takes loudness off the top, so the gain is re-measured and corrected after it, up
    to three times (a dense track pushed 4 dB into the limiter otherwise lands 1 dB short).
    limit_peaks=False stops after the gain: the track at its loudness with its peaks intact
    (a mix layer, which is limited once, as part of the mix).
    """
    x = as2d(x)
    before = analysis.lufs(x, sr)
    if before <= -70:  # silence: nothing to normalise
        return x, {"lufs_before": round(before, 1), "gain_db": 0.0, "limiter_db": 0.0, "lufs": round(before, 1),
                   "true_peak_dbtp": round(analysis.true_peak_db(x, sr), 2)}
    gain = lufs - before
    if not limit_peaks:
        y = (x * 10 ** (gain / 20)).astype(np.float32)
        return y, {"lufs_before": round(before, 1), "gain_db": round(gain, 2), "limiter_db": 0.0,
                   "lufs": round(lufs, 1), "true_peak_dbtp": round(analysis.true_peak_db(y, sr), 2)}
    for _ in range(3):
        # Sample peaks are limited 0.3 dB under the ceiling; inter-sample peaks can still poke
        # over, which the true-peak trim below takes care of.
        y, gr = limit((x * 10 ** (gain / 20)).astype(np.float32), sr, peak_dbtp - 0.3)
        now = analysis.lufs(y, sr) if gr > 0.0 else before + gain  # unlimited: the gain is exact
        err = lufs - now
        if abs(err) < 0.2 or gr == 0.0:
            break
        gain += err
    tp = analysis.true_peak_db(y, sr)
    trim = 0.0
    if tp > peak_dbtp:
        trim = peak_dbtp - tp - 0.05
        y = (y * 10 ** (trim / 20)).astype(np.float32)
    # The trim is a plain gain, so the final loudness and peak follow without measuring again.
    return y, {"lufs_before": round(before, 1), "gain_db": round(gain + trim, 2), "limiter_db": round(gr, 2),
               "lufs": round(now + trim, 1), "true_peak_dbtp": round(tp + trim, 2)}


# --------------------------------------------------------------------------- loops

def _mel(x: np.ndarray, sr: int):
    import librosa
    m = analysis._mono_for_analysis(x, sr)
    hop = 512
    S = librosa.feature.melspectrogram(y=m, sr=analysis.ANALYSIS_SR, hop_length=hop, n_mels=64)
    return np.log1p(S), analysis.ANALYSIS_SR / hop  # (n_mels, frames), frames per second


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else 0.0


def find_loop(x: np.ndarray, sr: int, beats: list[float], bars: int, beats_per_bar: int = 4,
              skip_s: float = 1.0) -> dict:
    """The best place to cut a seamless loop of `bars` bars, on the beat grid.

    Every beat after skip_s (past the opening) is a candidate start; the loop ends exactly
    bars * beats_per_bar beats later, per the tracked grid. A candidate scores by how alike the
    music is around its two ends (log-mel cosine over one beat either side: what plays after
    the end should sound like what plays after the start). The winner's end is then nudged by
    up to +-10 ms to the best waveform match, so the seam lines up to the sample. Candidates
    whose two ends differ in level just after the beat lose 0.02 per dB.
    """
    n_beats = bars * beats_per_bar
    b = np.asarray(beats, dtype=np.float64)
    total = as2d(x).shape[1] / sr
    xf = LOOP_XFADE_S
    cands = [i for i in range(len(b) - n_beats) if b[i] >= skip_s and b[i + n_beats] + xf + 0.01 < total]
    if not cands:
        cands = [i for i in range(len(b) - n_beats) if b[i + n_beats] + xf + 0.01 < total]
    if not cands:
        need = n_beats * (float(np.median(np.diff(b))) if len(b) > 1 else 0.5)
        raise ValueError(f"the track is too short for a {bars}-bar loop (needs ~{need + skip_s:.0f} s with "
                         f"beats tracked across it): generate a longer one or ask for fewer bars")
    S, fps = _mel(x, sr)
    beat_frames = max(1, int(round(float(np.median(np.diff(b))) * fps)))

    def frames(t: float) -> int:
        return int(round(t * fps))

    env = analysis.envelope_db(x, sr)  # 10 ms hops

    def level_after(t: float) -> float:
        i = int(round(t / analysis.HOP_S)) + int(xf / analysis.HOP_S)
        return float(np.mean(env[i:i + 10])) if i + 10 <= len(env) else float(env[-1])

    best = None
    for i in cands:
        s, e = frames(b[i]), frames(b[i + n_beats])
        after = min(beat_frames, S.shape[1] - e)   # what follows each end...
        before = min(beat_frames, s)               # ...and what leads into it
        sims = []
        if after > 0:
            sims.append(_cos(S[:, s:s + after], S[:, e:e + after]))
        if before > 0:
            sims.append(_cos(S[:, s - before:s], S[:, e - before:e]))
        sim = float(np.mean(sims)) if sims else 0.0
        # After the wrap the loop's start plays where the end's continuation would have: their
        # levels over 100 ms should match too (a downbeat landing where a rest was is heard).
        jump_db = abs(level_after(b[i]) - level_after(b[i + n_beats]))
        score = sim - 0.02 * jump_db
        if best is None or score > best[0]:
            best = (score, i, sim)
    _, i, sim = best
    x2 = as2d(x)
    start = int(round(b[i] * sr))
    length = int(round((b[i + n_beats] - b[i]) * sr))
    m = x2.mean(axis=0)
    ref = m[start:start + 2048]
    span = int(0.01 * sr)
    best_l, best_c = length, -2.0
    for L in range(max(1, length - span), length + span + 1, 4):  # 4-sample steps, then refine
        seg = m[start + L:start + L + len(ref)]
        if len(seg) < len(ref) or start + L + int(xf * sr) >= x2.shape[1]:
            continue
        c = _cos(ref, seg)
        if c > best_c:
            best_c, best_l = c, L
    for L in range(best_l - 3, best_l + 4):
        seg = m[start + L:start + L + len(ref)]
        if len(seg) == len(ref) and start + L + int(xf * sr) < x2.shape[1] and (c := _cos(ref, seg)) > best_c:
            best_c, best_l = c, L
    return {"start": start, "length": best_l, "start_s": round(start / sr, 3), "length_s": round(best_l / sr, 4),
            "bars": bars, "beats": n_beats, "bpm": round(n_beats * 60 * sr / best_l, 2),
            "seam_similarity": round(sim, 3), "seam_correlation": round(best_c, 3)}


def cut_loop(x: np.ndarray, sr: int, start: int, length: int, xfade_s: float = LOOP_XFADE_S) -> np.ndarray:
    """x[start:start+length] made seamless: its first xfade_s is an equal-power crossfade from
    what follows the loop's end into its own start, so playing it end-to-start continues."""
    x = as2d(x)
    k = int(round(xfade_s * sr))
    if start + length + k > x.shape[1]:
        raise ValueError("not enough audio after the loop's end for the crossfade")
    y = x[:, start:start + length].copy()
    t = np.linspace(0.0, np.pi / 2, k, dtype=np.float32)
    y[:, :k] = x[:, start:start + k] * np.sin(t) + x[:, start + length:start + length + k] * np.cos(t)
    return y


def seam_report(y: np.ndarray, sr: int, follow: np.ndarray) -> dict:
    """How the loop's wrap compares with the source it was cut from.

    The wrap itself is continuous by construction (cut_loop crossfades the source's own
    continuation into the loop's start), so what can be heard is the music: after the
    crossfade the loop plays its start where the source would have played `follow`.
    seam_level_db is how much louder or quieter that is over the next 100 ms; seam_ok is
    under 3 dB. (find_loop's seam_similarity covers what is playing.)
    """
    m = as2d(y).mean(axis=0)
    f = as2d(follow).mean(axis=0)
    a, k = int(LOOP_XFADE_S * sr), int(0.1 * sr)
    if len(f) < a + k or len(m) < a + k:
        return {"seam_level_db": None, "seam_ok": True}
    rms = lambda v: 20 * np.log10(np.sqrt(np.mean(np.square(v, dtype=np.float64))) + 1e-9)  # noqa: E731
    level = float(abs(rms(m[a:a + k]) - rms(f[a:a + k])))
    return {"seam_level_db": round(level, 1), "seam_ok": level < 3.0}


# --------------------------------------------------------------------------- the whole chain

def process(raw: np.ndarray, sr: int, opts: PostOpts, beats: list[float] | None = None,
            limit_peaks: bool = True) -> tuple[np.ndarray, dict]:
    """Raw engine output -> the delivered 48 kHz track, and a report of what was done.

    `beats` (seconds, on the raw audio's timeline) are needed for loops; the caller measures
    them once per take (analysis.tempo_and_beats). limit_peaks=False: the same track before
    its limiter (see master).
    """
    x = resample(as2d(raw), sr, SR)
    rep: dict = {}
    if opts.loop_bars:
        if not beats or len(beats) < 4:
            raise ValueError("no steady beat was found in this take, so it can't be looped on the bar")
        lp = find_loop(x, SR, beats, opts.loop_bars, opts.beats_per_bar)
        y = cut_loop(x, SR, lp["start"], lp["length"])
        rep["loop"] = {k: v for k, v in lp.items() if k not in ("start", "length")}
        follow = x[:, lp["start"] + lp["length"]:lp["start"] + lp["length"] + int(0.2 * SR)]
        rep.update(seam_report(y, SR, follow))  # before mastering: the same gain on both sides
        # No fades on a loop: they would be heard at every repeat.
        y, m = master(y, SR, opts.lufs, opts.peak_dbtp, limit_peaks)
        rep.update(m)
    else:
        if opts.trim_silence:
            x, t = trim_silence(x, SR)
            rep.update(t)
        fade_out = opts.fade_out or 0.0
        if opts.duration is not None and x.shape[1] > int(opts.duration * SR):
            x = x[:, :int(opts.duration * SR)]
            fade_out = max(fade_out, min(PostOpts.CUT_FADE, opts.duration / 4))
            rep["cut_at_s"] = opts.duration
        elif opts.fade_out is None and ends_abruptly(x, SR):
            fade_out = min(AUTO_FADE_S, x.shape[1] / SR / 10)
            rep["auto_fade_out_s"] = round(fade_out, 2)
        rep["fade_out_s"] = round(fade_out, 2)
        y = fade(x, SR, opts.fade_in, fade_out)
        y, m = master(y, SR, opts.lufs, opts.peak_dbtp, limit_peaks)
        rep.update(m)
    rep["length_s"] = round(y.shape[1] / SR, 3)
    return y, rep


# --------------------------------------------------------------------------- mixing layers

@dataclass
class Layer:
    audio: np.ndarray        # (channels, n) at the mix's sample rate
    gain_db: float = 0.0     # its volume in the mix
    offset_s: float = 0.0    # where it starts in the mix


def mix_layers(layers: list[Layer], sr: int) -> tuple[np.ndarray, dict]:
    """Layers on top of each other, each at its volume and offset; as long as the longest.
    Nothing is levelled or ducked: the volumes are the caller's. Unmastered (the caller
    masters the sum). Returns (mix, report)."""
    if not layers:
        raise ValueError("nothing to mix")
    starts = [int(round(ly.offset_s * sr)) for ly in layers]
    n = max(s + as2d(ly.audio).shape[1] for s, ly in zip(starts, layers))
    out = np.zeros((2, n), np.float32)
    for s, ly in zip(starts, layers):
        a = as2d(ly.audio)
        a = np.vstack([a, a]) if a.shape[0] == 1 else a[:2]
        out[:, s:s + a.shape[1]] += a * 10 ** (ly.gain_db / 20)
    return out, {"length_s": round(n / sr, 3)}


SPAN_RAMP_S = 0.03  # a span's gain fades in and out over this, inside the span: no clicks

# A space for a dry layer (a voice recorded or generated without a room). Convolution with a
# seeded, exponentially decaying stereo noise tail whose highs die faster than its lows:
# (RT60 seconds, wet, dry, pre-delay ms). The names, and roughly the rooms, are local-voice's.
REVERBS = {
    "room": (0.6, 0.16, 0.92, 0.0),
    "hall": (1.8, 0.26, 0.86, 20.0),
    "cathedral": (3.5, 0.24, 0.88, 40.0),
}


def gain_spans(x: np.ndarray, sr: int, spans: list[tuple[float, float, float]]) -> np.ndarray:
    """x with each (start_s, end_s, gain_db) span raised or lowered; overlapping spans add up.
    The change ramps in and out over SPAN_RAMP_S inside the span."""
    x = as2d(x)
    if not spans:
        return x
    n = x.shape[1]
    db = np.zeros(n, np.float64)
    for a, b, g in spans:
        i, j = max(0, int(round(a * sr))), min(n, int(round(b * sr)))
        if j <= i:
            continue
        env = np.ones(j - i)
        r = min(int(SPAN_RAMP_S * sr), (j - i) // 2)
        if r > 0:
            env[:r] = np.linspace(0.0, 1.0, r)
            env[-r:] = np.linspace(1.0, 0.0, r)
        db[i:j] += g * env
    return (x * (10 ** (db / 20))[None, :]).astype(np.float32)


def reverb_ir(kind: str, sr: int) -> np.ndarray:
    """The impulse response of a REVERBS space: (2, n), unit energy per channel."""
    from scipy.signal import lfilter
    rt60, _, _, pre_ms = REVERBS[kind]
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)  # the same room every time
    out = []
    for _ in range(2):  # decorrelated left and right: width
        bright = rng.standard_normal(n) * np.exp(-6.9 * t / (rt60 * 0.5))
        dark = lfilter([0.15], [1.0, -0.85], rng.standard_normal(n)) * np.exp(-6.9 * t / rt60)
        ir = bright + 2.5 * dark
        ir[: int(0.005 * sr)] *= np.linspace(0.0, 1.0, int(0.005 * sr))
        ir = np.concatenate([np.zeros(int(pre_ms / 1000 * sr)), ir])
        out.append(ir / np.sqrt(np.sum(ir ** 2)))
    return np.vstack(out).astype(np.float32)


def add_reverb(x: np.ndarray, sr: int, kind: str) -> np.ndarray:
    """x in a REVERBS space, its tail added at the end, at the same integrated loudness as x
    (so a layer's volume in a mix means the same with or without the room)."""
    from scipy.signal import fftconvolve
    x = as2d(x)
    x = np.vstack([x, x]) if x.shape[0] == 1 else x[:2]
    _, wet_l, dry_l, _ = REVERBS[kind]
    ir = reverb_ir(kind, sr)
    wet = np.vstack([fftconvolve(x[c], ir[c]) for c in range(2)])
    y = wet * wet_l
    y[:, : x.shape[1]] += x * dry_l
    before, after = analysis.lufs(x, sr), analysis.lufs(y, sr)
    if before > -70 and after > -70:
        y *= 10 ** ((before - after) / 20)
    return y.astype(np.float32)


SPEECH_BAND = (300.0, 3400.0)  # where the words are: what decides whether a voice is followed
MARGIN_WIN_S = 0.5
MARGIN_MIN_RUN_S = 1.0
MARGIN_P10_WARN_DB = 6.0  # a voice whose quietest tenth sits closer than this to the music loses words


def speech_margin(voice: np.ndarray, rest: np.ndarray, sr: int) -> dict:
    """How far a voice sits above everything else in the speech band, while it speaks.

    Both on the mix's timeline, at their mix volumes. Per 0.5 s window where the voice is
    speaking (within 20 dB of its loudest), the voice's band energy over the rest's, in dB.
    Returns the median and the 10th percentile, and the spans (seconds) of at least
    MARGIN_MIN_RUN_S where the voice stays under the rest (margin < 0 dB), with the worst margin
    in each: where words get lost. (A single quiet syllable or breath dips under the music at any
    sensible balance: those aren't spans.) A proxy, calibrated against Whisper on real mixes.
    """
    from scipy.signal import butter, sosfilt
    sos = butter(4, SPEECH_BAND, btype="band", fs=sr, output="sos")
    v = sosfilt(sos, as2d(voice).mean(axis=0).astype(np.float64))
    o = sosfilt(sos, as2d(rest).mean(axis=0).astype(np.float64))
    w = int(MARGIN_WIN_S * sr)
    k = min(len(v), len(o)) // w
    if k == 0:
        return {"median_db": None, "p10_db": None, "under": []}
    ev = 10 * np.log10(np.mean(v[: k * w].reshape(k, w) ** 2, axis=1) + 1e-12)
    eo = 10 * np.log10(np.mean(o[: k * w].reshape(k, w) ** 2, axis=1) + 1e-12)
    present = ev > -90.0  # where the voice layer has any sound at all (it's zeros outside itself)
    if not np.any(present):
        return {"median_db": None, "p10_db": None, "under": []}
    active = present & (ev > np.percentile(ev[present], 95) - 20.0)
    if not np.any(active):
        return {"median_db": None, "p10_db": None, "under": []}
    m = ev - eo
    under, run = [], None
    for i in range(k):
        if active[i] and m[i] < 0:
            if run is None:
                run = [i, i, m[i]]
            else:
                run[1], run[2] = i, min(run[2], m[i])
        elif run is not None and (active[i] or i - run[1] > 1):  # a pause of one window doesn't split a run
            under.append(run)
            run = None
    if run is not None:
        under.append(run)
    return {"median_db": round(float(np.median(m[active])), 1), "p10_db": round(float(np.percentile(m[active], 10)), 1),
            "under": [[round(a * MARGIN_WIN_S, 1), round((b + 1) * MARGIN_WIN_S, 1), round(float(g), 1)]
                      for a, b, g in under if (b + 1 - a) * MARGIN_WIN_S >= MARGIN_MIN_RUN_S]}


# --------------------------------------------------------------------------- keeping originals

KEEP_XFADE_S = 0.1  # original <-> the engine's audio, just outside the region it was asked to write


def keep_outside(x: np.ndarray, r: np.ndarray, m0: int, m1: int, xfade: int) -> np.ndarray:
    """The engine's output r with the original x put back outside r's masked region [m0, m1).

    The engine re-renders everything it hears, not only what it was asked to write: the same
    music, but not the same waveform (measured 14-23 dB apart), and a voice comes back as the
    model's imitation of one. So only [m0, m1) is taken from r; x is kept everywhere else it
    exists, crossfading linearly over `xfade` samples just outside the mask on each side
    (linear: both sides there are renderings of the same moment, so an equal-power fade would
    bump the level by up to 3 dB). x and r share their timeline from sample 0; where x ends
    (an extension), r carries on. Returns an array as long as r.
    """
    x, r = as2d(x), as2d(r)
    n = r.shape[1]
    have = min(x.shape[1], n)
    out = r.copy()
    a = max(0, min(m0, have) - xfade)
    out[:, :a] = x[:, :a]
    if min(m0, have) > a:
        k = min(m0, have) - a
        t = np.linspace(0.0, 1.0, k, dtype=np.float32)
        out[:, a:a + k] = x[:, a:a + k] * (1 - t) + r[:, a:a + k] * t
    if m1 < have:
        b = min(m1 + xfade, have)
        t = np.linspace(0.0, 1.0, b - m1, dtype=np.float32)
        out[:, m1:b] = r[:, m1:b] * (1 - t) + x[:, m1:b] * t
        out[:, b:have] = x[:, b:have]
    return out
