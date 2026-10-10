"""From what the caller asked for to the prompt the engine gets.

Stable Audio 3 was trained on AudioSparx and Freesound with their metadata, and its prompt
guide (engine/stable-audio-3/docs/guides/prompting.md) recommends:

- for music: a description of genre, instruments, mood and energy, and the tempo as
  "120 BPM"; the AudioSparx tags "TrackType: Music, VocalType: Instrumental" give "higher
  quality, more semantically coherent outputs";
- for a solo instrument or stem: start with "TrackType: Instrument".

build_prompt adds those tags and the tempo / key unless the caller sends the text raw.
"""

from __future__ import annotations

import re

MUSIC_TAGS = "TrackType: Music, VocalType: Instrumental"
STEM_TAGS = "TrackType: Instrument"
KINDS = ("music", "stem")


def build_prompt(text: str, bpm: float | None = None, key: str | None = None, kind: str = "music",
                 raw: bool = False) -> str:
    """The engine's prompt. `kind`: 'music' (a full instrumental track) or 'stem' (one instrument
    or part, e.g. a drum loop or a solo piano). raw: send `text` exactly as written."""
    text = " ".join(text.split())
    if raw:
        return text
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    parts = []
    if "tracktype:" not in text.lower():
        parts.append(MUSIC_TAGS if kind == "music" else STEM_TAGS)
    body = text.rstrip(" .")
    if key and not re.search(r"\b[A-G][#b]?\s*(major|minor|maj|min)\b", body, re.I):
        body += f", in {key}"
    if bpm and not re.search(r"\b\d{2,3}(\.\d+)?\s*bpm\b", body, re.I):
        body += f", {bpm:g} BPM"
    parts.append(body)
    return ", ".join(parts) if len(parts) > 1 else parts[0]


TAG = re.compile(r"\b(TrackType|VocalType|Genre|Instruments|Format)\s*:\s*[^,.]*[,.]?\s*", re.I)


def describe(prompt: str) -> str:
    """The prompt without its AudioSparx tags: what CLAP is asked to match (it was trained on
    plain captions, so 'TrackType: Music' would only add noise)."""
    return " ".join(TAG.sub("", prompt).split()).strip(" ,.") or prompt


STOP = {"a", "an", "the", "and", "with", "of", "in", "on", "for", "to", "at", "by", "from", "some", "very",
        "that", "this", "is", "are", "bpm", "music", "track", "instrumental", "tracktype", "vocaltype"}


def track_name(text: str, words: int = 4) -> str:
    """A short file-name stem from the request: 'Warm lo-fi hip hop, dusty piano, 84 BPM' ->
    'warm-lofi-hip-hop'. Letters, digits and hyphens only."""
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower().replace("-", "")) if t not in STOP and not t.isdigit()]
    return "-".join(toks[:words]) or "track"
