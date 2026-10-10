"""Checks on a generated track, for a caller that can't listen.

signal_report: is it broken? Silence, clipping, dropouts, an early or abrupt ending, phase
and DC problems. `fails` reject the take; `warnings` lower its rank and are shown.

musical_report: is it what was asked? The tempo and the key, measured and compared with the
request's `bpm` / `key` when it gave them.
"""

from __future__ import annotations

import numpy as np

from . import analysis
from .audio import as2d

SILENT_PEAK_DB = -50.0
DROPOUT_S = 1.5          # a silence this long inside the music is a dropout
EARLY_END_FRAC = 0.15    # trailing silence longer than this share of the take: it ended early
CLIP_NEAR_PEAK = 0.98    # a flat top is within 2% of the take's own peak...
CLIP_RUN = 3             # ...and this many consecutive samples hold exactly the same value
TEMPO_CONFIDENT = 0.35   # below this the beat grid is too irregular to judge the tempo by
KEY_CONFIDENT = 0.55     # below this the key estimate is a guess


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of the True runs in a 1-D bool array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def clipped_fraction(x: np.ndarray) -> float:
    """Share of samples in flat tops: CLIP_RUN or more consecutive samples holding exactly the
    same value near the peak. That is what clipping leaves, whatever the level it happened at
    (the float raw can peak above 1.0 without being clipped; a clipped file can peak at -3 dB)."""
    x = as2d(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return 0.0
    n = 0
    for ch in x:
        flat = np.zeros(len(ch), dtype=bool)
        same = (ch[1:] == ch[:-1]) & (np.abs(ch[1:]) >= CLIP_NEAR_PEAK * peak)
        flat[1:] |= same
        flat[:-1] |= same
        for a, b in _runs(flat):
            if b - a >= CLIP_RUN:
                n += b - a
    return n / max(1, x.size)


def signal_report(x: np.ndarray, sr: int, requested_s: float | None = None) -> dict:
    x = as2d(x)
    fails: list[str] = []
    warnings: list[str] = []
    dur = x.shape[1] / sr
    peak = analysis.sample_peak_db(x)
    rep: dict = {"duration_s": round(dur, 2), "peak_db": round(peak, 1)}
    if peak < SILENT_PEAK_DB:
        fails.append(f"silent (peak {peak:.0f} dBFS)")
        return {**rep, "fails": fails, "warnings": warnings}

    env = analysis.envelope_db(x, sr)  # 10 ms hops
    hop_s = analysis.HOP_S
    # Silence relative to the music itself (a quiet piece is not a dropout) and in absolute terms.
    level = float(np.percentile(env, 50))
    quiet = env < max(level - 35.0, -70.0)
    loud = np.nonzero(~quiet)[0]
    if not len(loud):  # a few clicks in silence: nothing that counts as music
        fails.append(f"silent (no sound above {max(level - 35.0, -70.0):.0f} dBFS beyond a peak of {peak:.0f})")
        return {**rep, "fails": fails, "warnings": warnings}
    first, last = int(loud[0]), int(loud[-1])
    lead_s, tail_s = first * hop_s, (len(env) - 1 - last) * hop_s
    rep.update(lead_silence_s=round(lead_s, 2), tail_silence_s=round(tail_s, 2))
    if tail_s > EARLY_END_FRAC * dur and tail_s > 2.0:
        warnings.append(f"the music ends at {dur - tail_s:.1f} s and the rest is silence "
                        f"(trimmed off when mastered)")
    if lead_s > 1.0:
        warnings.append(f"{lead_s:.1f} s of silence before the music starts (trimmed off when mastered)")
    gaps = [(a, b) for a, b in _runs(quiet[first:last + 1]) if (b - a) * hop_s >= DROPOUT_S]
    if gaps:
        where = ", ".join(f"{(first + a) * hop_s:.1f}-{(first + b) * hop_s:.1f} s" for a, b in gaps[:3])
        warnings.append(f"drops to silence inside the music ({where})")
    rep["dropouts"] = [[round((first + a) * hop_s, 2), round((first + b) * hop_s, 2)] for a, b in gaps]

    clip = clipped_fraction(x)
    rep["clipped_pct"] = round(100 * clip, 4)
    if clip > 1e-3:
        fails.append(f"clipped ({100 * clip:.2f}% of samples pinned at full scale)")
    elif clip > 1e-5:
        warnings.append(f"some clipping ({100 * clip:.3f}% of samples)")

    # An abrupt ending: the last half-second of sound is about as loud as the music's middle.
    rep["end_level_db"] = analysis.ending(x, sr)["end_db"]
    rep["abrupt_end"] = bool(rep["end_level_db"] > -6.0)

    corr = analysis.stereo_correlation(x)
    rep["stereo_correlation"] = None if corr is None else round(corr, 3)
    if corr is not None and corr < 0.0:
        warnings.append(f"stereo channels are out of phase (correlation {corr:.2f}): it will thin out in mono")
    dc = float(np.max(np.abs(x.mean(axis=1))))
    if dc > 0.01:
        warnings.append(f"DC offset ({dc:.3f})")
    if requested_s and dur < 0.9 * requested_s:
        warnings.append(f"shorter than asked ({dur:.1f} s of {requested_s:g} s)")
    return {**rep, "fails": fails, "warnings": warnings}


def musical_report(x: np.ndarray, sr: int, bpm: float | None = None, key: str | None = None) -> dict:
    """Tempo, beats and key, measured; compared with the requested bpm / key when given."""
    warnings: list[str] = []
    t = analysis.tempo_and_beats(x, sr, bpm_hint=bpm)
    tempo = {"bpm": t["bpm"], "confidence": t["confidence"], "beats": len(t["beats"])}
    if bpm:
        tempo["asked"] = bpm
        rel = analysis.tempo_matches(t["bpm"], bpm)
        tempo["match"] = rel
        if t["bpm"] is None or t["confidence"] < TEMPO_CONFIDENT:
            tempo["match"] = "unclear"
        elif rel is None:
            warnings.append(f"tempo is {t['bpm']:.0f} BPM, not the {bpm:g} asked")
        elif rel != "exact":
            tempo["note"] = f"{t['bpm']:.0f} BPM: {rel} time of the {bpm:g} asked (the same grid)"
    k = analysis.estimate_key(x, sr)
    keyrep = {"key": k["key"], "strength": k["strength"], "runner_up": k["runner_up"]}
    if key:
        want_pc, want_mode = analysis.parse_key(key)
        keyrep["asked"] = analysis.key_name(want_pc, want_mode)
        if k["key"] is None or k["strength"] < KEY_CONFIDENT:
            keyrep["match"] = "unclear"
        else:
            rel = analysis.key_relation(k["pc"], k["mode"], want_pc, want_mode)
            keyrep["match"] = rel
            if rel is None:
                warnings.append(f"sounds in {k['key']}, not the {keyrep['asked']} asked")
            elif rel != "exact":
                keyrep["note"] = f"{k['key']}: the {rel} key of the {keyrep['asked']} asked"
    return {"tempo": tempo, "key": keyrep, "beat_times": t["beats"], "warnings": warnings}
