"""Synthetic music for the tests. No models are loaded anywhere in tests/."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 44100  # what the engine writes

# Triads as semitones above C. A I-V-vi-IV progression in C major, and i-VI-III-VII in A minor.
C_MAJOR = [(0, 4, 7), (7, 11, 14), (9, 12, 16), (5, 9, 12)]
A_MINOR = [(9, 12, 16), (5, 9, 12), (0, 4, 7), (7, 11, 14)]


def band(bpm: float = 120.0, seconds: float = 20.0, chords=C_MAJOR, sr: int = SR, seed: int = 0,
         stereo: bool = True, root_hz: float = 261.63) -> np.ndarray:
    """A steady little band: a kick on every beat, a hi-hat on the off-beats, and sustained
    chords changing every bar (4 beats). (2, n) or (1, n) float32 around -12 dBFS peak."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    beat = 60.0 / bpm
    kick_len = int(0.15 * sr)
    kt = np.arange(kick_len) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 18)
    hat_len = int(0.04 * sr)
    hat = rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / sr * 120) * 0.3
    k = 0
    while k * beat < seconds:
        i = int(k * beat * sr)
        j = min(n, i + kick_len)
        x[i:j] += kick[:j - i]
        h = int((k + 0.5) * beat * sr)
        if h < n:
            e = min(n, h + hat_len)
            x[h:e] += hat[:e - h]
        k += 1
    bar = 4 * beat
    pad = np.zeros(n)
    for b in range(int(np.ceil(seconds / bar))):
        a, z = int(b * bar * sr), min(n, int((b + 1) * bar * sr))
        for semi in chords[b % len(chords)]:
            f = root_hz * 2 ** (semi / 12)
            pad[a:z] += 0.12 * np.sin(2 * np.pi * f * t[a:z]) + 0.04 * np.sin(2 * np.pi * 2 * f * t[a:z])
    x = x * 0.5 + pad
    x = 0.25 * x / np.max(np.abs(x))
    if not stereo:
        return x[None, :].astype(np.float32)
    return np.vstack([x, 0.97 * x]).astype(np.float32)


@pytest.fixture()
def sr():
    return SR
