"""Waveform, short-term loudness and spectrogram PNG with the bar grid, for an agent to look at.

An agent can't listen; this picture shows the things a listener would notice: where the
music builds and drops (loudness), what's playing (spectrogram: bass, kick, cymbals, pads),
silences, an abrupt end, and whether the beat grid is steady. For a loop, a dashed line marks
the seam in a copy rotated so the wrap sits in the middle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import analysis
from .audio import as2d

WIDTH_PX = 1440


def _envelope(m: np.ndarray, sr: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, min, max) per bin: draws a minutes-long waveform without plotting every sample."""
    n = len(m) // bins * bins if len(m) >= bins else len(m)
    if n < bins:
        t = np.arange(len(m)) / sr
        return t, m, m
    blocks = m[:n].reshape(bins, -1)
    t = (np.arange(bins) + 0.5) * blocks.shape[1] / sr
    return t, blocks.min(axis=1), blocks.max(axis=1)


def draw(x: np.ndarray, sr: int, path: Path | str, title: str = "", beats: list[float] | None = None,
         beats_per_bar: int = 4, seam: bool = False) -> Path:
    """Write the PNG. `beats` (seconds, on this audio's timeline) draw the bar grid: a line
    every bar, numbered, and fainter beat lines when they're far enough apart to see."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = as2d(x)
    if seam:
        x = np.roll(x, x.shape[1] // 2, axis=1)
        beats = None  # the rotated copy's grid would mislead
    m = x.mean(axis=0)
    dur = len(m) / sr
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(WIDTH_PX / 80, 6.6), height_ratios=[1, 0.7, 2.2],
                                        sharex=True)
    t, lo, hi = _envelope(m, sr, 4000)
    ax1.fill_between(t, lo, hi, color="#1f3b5a", lw=0)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("amplitude")
    ax1.set_title(title, fontsize=9, loc="left", pad=14)

    lt, lv = analysis.short_term_lufs(x, sr, hop_s=0.5 if dur < 120 else 1.0)
    if len(lt):
        ax2.plot(lt, lv, color="#b5482f", lw=1.2)
        ax2.set_ylim(max(-60, float(np.min(lv)) - 3), min(0, float(np.max(lv)) + 3))
    ax2.set_ylabel("LUFS (3 s)")
    ax2.grid(axis="y", color="#ddd", lw=0.5)

    nfft = 4096
    ax3.specgram(m + 1e-9, NFFT=nfft, Fs=sr, noverlap=nfft // 2, cmap="magma", vmin=-130)
    ax3.set_yscale("symlog", linthresh=400)
    ax3.set_ylim(30, sr / 2)
    ax3.set_ylabel("Hz")
    ax3.set_xlabel("seconds" + (f"  (grid: bars of {beats_per_bar} beats)" if beats else ""))

    if beats:
        b = np.asarray(beats)
        bars = b[::beats_per_bar]
        px_per_beat = WIDTH_PX * float(np.median(np.diff(b))) / dur if len(b) > 1 else 0
        label_every = max(1, int(np.ceil(len(bars) / 40)))  # at most ~40 numbers across
        for k, bt in enumerate(bars):
            strong = k % label_every == 0
            for ax in (ax1, ax2, ax3):
                ax.axvline(bt, color="#888" if strong else "#bbb", lw=0.6 if strong else 0.4,
                           alpha=0.9 if ax is not ax3 else 0.35, zorder=0)
            if strong:
                ax1.text(bt, 1.02, str(k + 1), fontsize=7, ha="center", va="bottom", color="#555",
                         transform=ax1.get_xaxis_transform())
        if px_per_beat >= 6:
            for bt in b:
                ax1.axvline(bt, color="#ddd", lw=0.3, zorder=0)
    if seam:
        for ax in (ax1, ax2, ax3):
            ax.axvline(dur / 2, color="cyan", lw=1.0, ls="--")
        ax1.text(dur / 2, -1.0, " loop seam (end -> start)", color="#08a", fontsize=7, va="bottom")
    ax1.set_xlim(0, dur)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
