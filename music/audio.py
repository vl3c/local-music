"""Audio arrays, resampling and WAV files with an INFO chunk.

Audio is float32 shaped (channels, samples) everywhere in this package: mono is (1, n).
Delivered files are 48 kHz 24-bit PCM WAV (every editor and DAW imports it), with the
prompt, seed, model and scores in the RIFF INFO chunk so they travel with the file, plus
an MP3 for sharing.
"""

from __future__ import annotations

import io
import os
import struct
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000  # every delivered file


def replace(src: Path | str, dst: Path | str, tries: int = 40, wait_s: float = 0.05) -> None:
    """os.replace, retried for a couple of seconds: Windows refuses to replace a file another
    handle has open (the browser streaming a take, a reader mid-read). Raises PermissionError
    if it never frees up."""
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(wait_s)


def as2d(x: np.ndarray) -> np.ndarray:
    """(channels, samples) float32, whatever came in: (n,), (n, ch) from soundfile, or (ch, n)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {x.shape}")
    # soundfile gives (n, ch); a real (ch, n) array has far fewer rows than columns.
    return np.ascontiguousarray(x.T) if x.shape[0] > x.shape[1] else x


def read(path: Path | str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """High-quality resample (soxr VHQ)."""
    x = as2d(x)
    if sr_in == sr_out:
        return x
    import soxr
    return np.ascontiguousarray(soxr.resample(x.T, sr_in, sr_out, quality="VHQ").T.astype(np.float32))


def mono(x: np.ndarray) -> np.ndarray:
    """Fold to mono. If the channels are out of phase (summing would cancel), keep the louder one."""
    x = as2d(x)
    if x.shape[0] == 1:
        return x
    l, r = x[0], x[1]
    s = 0.5 * (l + r)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v, dtype=np.float64)) + 1e-20))  # noqa: E731
    if rms(s) < 0.5 * max(rms(l), rms(r)):  # more than ~6 dB lost to cancellation
        return (l if rms(l) >= rms(r) else r)[None, :].copy()
    return s[None, :].astype(np.float32)


def stereo(x: np.ndarray) -> np.ndarray:
    x = as2d(x)
    return np.vstack([x, x]) if x.shape[0] == 1 else x[:2]


# --------------------------------------------------------------------------- WAV + INFO

# RIFF INFO ids this package writes. Values are short text; the JSON sidecar has the rest.
INFO_KEYS = {
    "title": b"INAM",      # the track's name
    "comment": b"ICMT",    # prompt, seed, model, scores
    "software": b"ISFT",
    "date": b"ICRD",
    "genre": b"IGNR",
    "keywords": b"IKEY",
    "artist": b"IART",
    "copyright": b"ICOP",
}


def _info_chunk(info: dict[str, str]) -> bytes:
    body = b"INFO"
    for key, value in info.items():
        if not value or key not in INFO_KEYS:
            continue
        data = str(value).encode("utf-8", "replace")[:4000] + b"\0"
        if len(data) % 2:
            data += b"\0"  # RIFF chunks are word-aligned
        body += INFO_KEYS[key] + struct.pack("<I", len(data)) + data
    return b"LIST" + struct.pack("<I", len(body)) + body


def _chunks(riff: bytes) -> list[tuple[bytes, bytes]]:
    if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    out, i = [], 12
    while i + 8 <= len(riff):
        cid, size = riff[i:i + 4], struct.unpack("<I", riff[i + 4:i + 8])[0]
        out.append((cid, riff[i + 8:i + 8 + size]))
        i += 8 + size + (size & 1)
    return out


def write_wav(path: Path | str, x: np.ndarray, sr: int = SR, info: dict[str, str] | None = None,
              subtype: str = "PCM_24") -> Path:
    """Write a WAV (24-bit PCM by default) with an INFO chunk placed before the audio data.

    Integer formats are clamped to [-1, 1] first (they would wrap around otherwise); FLOAT keeps
    every value, so a raw take's peaks over full scale survive until mastering.
    """
    x = as2d(x) if subtype == "FLOAT" else np.clip(as2d(x), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, x.T, sr, format="WAV", subtype=subtype)
    chunks = [(c, d) for c, d in _chunks(buf.getvalue()) if c != b"LIST"]
    body = b"WAVE"
    for cid, data in chunks:
        if cid == b"data" and info:
            body += _info_chunk(info)
        body += cid + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    replace(tmp, path)
    return path


def read_info(path: Path | str) -> dict[str, str]:
    """The INFO chunk of a WAV, as {"title": ..., "comment": ...}."""
    names = {v: k for k, v in INFO_KEYS.items()}
    out: dict[str, str] = {}
    for cid, data in _chunks(Path(path).read_bytes()):
        if cid != b"LIST" or data[:4] != b"INFO":
            continue
        i = 4
        while i + 8 <= len(data):
            sid, size = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
            text = data[i + 8:i + 8 + size].rstrip(b"\0").decode("utf-8", "replace")
            out[names.get(sid, sid.decode("ascii", "replace"))] = text
            i += 8 + size + (size & 1)
    return out


def write_mp3(path: Path | str, x: np.ndarray, sr: int = SR, kbps: int = 320) -> Path:
    """An MP3 (libsndfile's LAME encoder, constant bitrate) next to the WAV, for sharing."""
    x = np.clip(as2d(x), -1.0, 1.0)
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    # compression_level: 0 = highest bitrate, 1 = lowest; libsndfile maps it onto LAME's range.
    with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=x.shape[0], format="MP3", subtype="MPEG_LAYER_III",
                      compression_level=min(1.0, max(0.0, (320 - kbps) / (320 - 32))),
                      bitrate_mode="CONSTANT") as f:
        f.write(x.T)
    replace(tmp, path)
    return path
