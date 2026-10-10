"""Measurements shared by mastering, loops and the checks: levels, loudness, tempo, beats and key."""

from __future__ import annotations

import numpy as np

from .audio import as2d

HOP_S = 0.01  # 10 ms envelope resolution: plenty for silences and dropouts in music


def db(v: float | np.ndarray) -> float | np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(v), 1e-10))


def envelope_db(x: np.ndarray, sr: int, hop_s: float = HOP_S) -> np.ndarray:
    """RMS level per hop (dBFS), of the channel sum's power."""
    x = as2d(x)
    hop = max(1, int(round(hop_s * sr)))
    n = x.shape[1] // hop
    if n == 0:
        return np.full(1, -200.0)
    p = np.mean(np.square(x[:, : n * hop].astype(np.float64)), axis=0).reshape(n, hop).mean(axis=1)
    return 10 * np.log10(p + 1e-20)


def noise_floor_db(env: np.ndarray) -> float:
    """The quiet end of the level distribution (10th percentile), dBFS."""
    return float(np.percentile(env, 10))


def ending(x: np.ndarray, sr: int, window_s: float = 0.5) -> dict:
    """How the music ends: the level of its last window_s of sound relative to its median level
    (dB; near 0 = it stops at full level, -25 or lower = it ends or fades), and the silence after it.

    "Sound" is anything within 35 dB of the median, so a short silence after an abrupt stop
    doesn't hide the stop.
    """
    env = envelope_db(x, sr)
    level = float(np.median(env))
    loud = np.nonzero(env >= max(level - 35.0, -70.0))[0]
    if not len(loud):
        return {"end_db": -200.0, "tail_s": round(len(env) * HOP_S, 2)}
    last = int(loud[-1])
    k = max(1, int(window_s / HOP_S))
    end = float(np.mean(env[max(0, last - k + 1):last + 1]))
    return {"end_db": round(end - level, 1), "tail_s": round((len(env) - 1 - last) * HOP_S, 2)}


def true_peak_db(x: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770-style true peak: the sample peak after 4x oversampling (dBTP)."""
    x = as2d(x)
    if x.size == 0:
        return -200.0
    import soxr
    up = soxr.resample(x.T.astype(np.float64), sr, sr * 4, quality="HQ")
    return float(db(max(np.max(np.abs(up)), np.max(np.abs(x)))))


def sample_peak_db(x: np.ndarray) -> float:
    return float(db(np.max(np.abs(as2d(x))))) if np.size(x) else -200.0


def k_weighted(x: np.ndarray, sr: int) -> np.ndarray:
    """The ITU-R BS.1770 K-weighting (pyloudnorm's two filter stages), float64 (channels, n)."""
    import pyloudnorm
    from scipy.signal import lfilter
    y = as2d(x).astype(np.float64)
    for f in pyloudnorm.Meter(sr)._filters.values():
        y = f.passband_gain * lfilter(f.b, f.a, y, axis=1)
    return y


def _block_power(p: np.ndarray, sr: int, block_s: float, step: float) -> np.ndarray:
    """Mean K-weighted power of each block (block_s long, every step * block_s), with the same
    block bounds as pyloudnorm, computed from a running sum instead of a loop."""
    c = np.concatenate([[0.0], np.cumsum(p)])
    n = len(p)
    blocks = int(np.round((n / sr - block_s) / (block_s * step))) + 1
    j = np.arange(max(blocks, 0))
    lo = np.minimum((block_s * (j * step) * sr).astype(np.int64), n)
    hi = np.minimum((block_s * (j * step + 1) * sr).astype(np.int64), n)
    return (c[hi] - c[lo]) / (block_s * sr)


def lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness (ITU-R BS.1770-4, gated), LUFS. Clips under 0.4 s are measured looped
    to 0.4 s. The same numbers as pyloudnorm (tests check), ~10x faster on long tracks."""
    x = as2d(x)
    need = int(0.4 * sr) + 1
    if x.shape[1] < need:
        x = np.tile(x, (1, need // max(1, x.shape[1]) + 1))
    p = np.sum(np.square(k_weighted(x, sr)), axis=0)  # channel weights are 1 for L and R
    z = _block_power(p, sr, 0.4, 0.25)
    with np.errstate(divide="ignore"):
        lj = -0.691 + 10 * np.log10(z)
        gated = z[lj >= -70.0]
        if not len(gated):
            return -200.0
        rel = -0.691 + 10 * np.log10(np.mean(gated)) - 10.0
        final = z[(lj > rel) & (lj > -70.0)]
        v = -0.691 + 10 * np.log10(np.mean(final)) if len(final) else -np.inf
    return float(v) if np.isfinite(v) else -200.0


def stereo_correlation(x: np.ndarray) -> float | None:
    """Pearson correlation of L and R (1 = mono, 0 = wide/unrelated, < 0 = phase problems)."""
    x = as2d(x)
    if x.shape[0] < 2:
        return None
    l, r = x[0].astype(np.float64), x[1].astype(np.float64)
    den = np.sqrt(np.sum(l * l) * np.sum(r * r))
    return float(np.sum(l * r) / den) if den > 1e-12 else None


def zero_crossings(v: np.ndarray) -> np.ndarray:
    """Indices i where v changes sign between i and i+1."""
    s = np.signbit(v)
    return np.nonzero(s[1:] != s[:-1])[0]


# --------------------------------------------------------------------------- loudness over time

def short_term_lufs(x: np.ndarray, sr: int, window_s: float = 3.0, hop_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """EBU R128 short-term loudness (ungated 3 s windows), every hop_s: (times at window centres,
    LUFS). Silent windows read -70 (the R128 absolute gate) instead of -inf."""
    x = as2d(x)
    win, hop = int(window_s * sr), max(1, int(hop_s * sr))
    if x.shape[1] < int(0.4 * sr):
        return np.array([]), np.array([])
    win = min(win, x.shape[1])
    c = np.concatenate([[0.0], np.cumsum(np.sum(np.square(k_weighted(x, sr)), axis=0))])
    starts = np.arange(0, x.shape[1] - win + 1, hop)
    z = (c[starts + win] - c[starts]) / win
    with np.errstate(divide="ignore"):
        v = -0.691 + 10 * np.log10(z)
    return (starts + win / 2) / sr, np.maximum(np.nan_to_num(v, nan=-70.0, neginf=-70.0), -70.0)


# --------------------------------------------------------------------------- tempo and beats

ANALYSIS_SR = 22050


def _mono_for_analysis(x: np.ndarray, sr: int) -> np.ndarray:
    from .audio import resample
    return resample(as2d(x), sr, ANALYSIS_SR).mean(axis=0)


def tempo_and_beats(x: np.ndarray, sr: int, bpm_hint: float | None = None) -> dict:
    """Estimated tempo and beat times (librosa's dynamic-programming beat tracker).

    A hint (the tempo that was asked for) centres the tracker's tempo prior on it, which
    settles the usual half/double-time ambiguity without forcing the answer: a track that
    really sits at another tempo still comes out there. Returns {"bpm", "beats" (seconds),
    "confidence"}: confidence is how evenly the beats are spaced, 1 for a perfect grid.
    """
    import librosa
    y = _mono_for_analysis(x, sr)
    if len(y) < ANALYSIS_SR * 2 or float(np.max(np.abs(y))) < 1e-4:
        return {"bpm": None, "beats": [], "confidence": 0.0}
    onset = librosa.onset.onset_strength(y=y, sr=ANALYSIS_SR)
    kw = {"start_bpm": float(bpm_hint)} if bpm_hint else {}
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=ANALYSIS_SR, trim=False, **kw)
    beats = librosa.frames_to_time(frames, sr=ANALYSIS_SR)
    bpm = float(np.atleast_1d(tempo)[0])
    if len(beats) >= 8:
        ibi = np.diff(beats)
        med = float(np.median(ibi))
        # The grid's own tempo: a straight line through every beat. Each beat sits on a 23 ms
        # analysis frame, so single intervals are coarse (126 BPM reads as 123 or 129), but
        # the slope over the whole track is good to a fraction of a BPM.
        slope = float(np.polyfit(np.arange(len(beats)), beats, 1)[0])
        bpm = 60.0 / slope if slope > 0 else 60.0 / med
        conf = float(np.clip(1.0 - np.std(ibi) / med * 4, 0.0, 1.0))
    else:
        conf = 0.0
    return {"bpm": round(bpm, 1), "beats": [round(float(b), 4) for b in beats], "confidence": round(conf, 2)}


def tempo_matches(measured: float | None, wanted: float, tol: float = 0.04) -> str | None:
    """'exact', 'half' or 'double' when the measured tempo is wanted (or half / twice it)
    within tol (relative); None when it's something else."""
    if not measured:
        return None
    for label, f in (("exact", 1.0), ("half", 0.5), ("double", 2.0)):
        if abs(measured - wanted * f) <= tol * wanted * f:
            return label
    return None


# --------------------------------------------------------------------------- key

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
# Krumhansl-Kessler key profiles (probe-tone ratings), C major and C minor.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def parse_key(text: str) -> tuple[int, str]:
    """'A minor', 'Am', 'f# min', 'Bb major', 'Eb' -> (pitch class, 'major' | 'minor')."""
    import re
    m = re.fullmatch(r"\s*([A-Ga-g])([#b♯♭]?)\s*((?i:maj(?:or)?|min(?:or)?|m))?\s*", text)
    if not m:
        raise ValueError(f"not a key: {text!r} (e.g. 'A minor', 'F# major', 'Bbm')")
    name = m.group(1).upper() + m.group(2).replace("♯", "#").replace("♭", "b")
    name = FLATS.get(name, name)
    mode = "minor" if (m.group(3) or "").lower() in ("m", "min", "minor") else "major"
    return NOTES.index(name), mode


def key_name(pc: int, mode: str) -> str:
    return f"{NOTES[pc % 12]} {mode}"


def estimate_key(x: np.ndarray, sr: int) -> dict:
    """The most likely key (Krumhansl-Schmuckler on the track's average chroma).

    Returns {"key": "A minor", "pc", "mode", "strength" (the winning correlation, ~0.5-0.9 for
    tonal music), "runner_up"}. A low strength (< 0.5) means the music is atonal, drum-only
    or ambiguous between keys: treat the answer as a guess.
    """
    import librosa
    y = _mono_for_analysis(x, sr)
    if len(y) < ANALYSIS_SR or float(np.max(np.abs(y))) < 1e-4:
        return {"key": None, "pc": None, "mode": None, "strength": 0.0, "runner_up": None}
    chroma = librosa.feature.chroma_cqt(y=y, sr=ANALYSIS_SR).mean(axis=1)
    scores = []
    for pc in range(12):
        for mode, prof in (("major", MAJOR), ("minor", MINOR)):
            scores.append((float(np.corrcoef(chroma, np.roll(prof, pc))[0, 1]), pc, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return {"key": key_name(best[1], best[2]), "pc": best[1], "mode": best[2], "strength": round(best[0], 3),
            "runner_up": key_name(second[1], second[2])}


def key_relation(found_pc: int, found_mode: str, want_pc: int, want_mode: str) -> str | None:
    """How a detected key relates to the wanted one: 'exact', 'relative' (same notes: A minor
    / C major), 'parallel' (same tonic, other mode), 'fifth' (a neighbour on the circle of
    fifths, a common confusion of the detector), or None."""
    if (found_pc, found_mode) == (want_pc, want_mode):
        return "exact"
    if found_mode != want_mode:
        rel = (want_pc + 9) % 12 if want_mode == "major" else (want_pc + 3) % 12
        if found_pc == rel:
            return "relative"
        if found_pc == want_pc:
            return "parallel"
    elif (found_pc - want_pc) % 12 in (5, 7):
        return "fifth"
    return None
