---
name: local-music
description: Generate instrumental music locally with Stable Audio 3 Medium on this machine's GPU: full tracks or single-instrument stems from a description (genre, instruments, mood, tempo, key), up to 6 min 20 s, for games, videos, animations, podcasts and menus; or start from the user's own audio file (continue it, restyle it), or join several takes into one song with bridges written between them, or mix takes on top of each other (a voice over a song) at set volumes. Every take is checked automatically (silence, clipping, dropouts, abrupt ends; tempo and key against what was asked; CLAP prompt match; Audiobox enjoyment and production quality), ranked best-of-N and mastered to -14 LUFS / -1 dBTP; takes can be extended, varied, partly regenerated, re-mastered or cut into seamless loops on the bar, and are delivered as WAV + MP3 to a folder the user names. Use when the user asks for music, a soundtrack, a background track, a loop, a jingle bed or a stem made locally or without a cloud service. Not for songs with lyrics, speech or sound effects.
---

# local-music: instrumental music on the local GPU

You describe the music; the server makes N takes (default 4), checks, scores and ranks them,
masters them, and hands back WAV + MP3 files with a picture of each (waveform, loudness over
time, spectrogram with the bar grid).

**You cannot hear the music.** You judge takes from the checks, the scores and the PNG (Read
it). The user is the only judge of how it *sounds*: deliver early, say what you looked at,
and let them steer with concrete knobs.

## Quick start

The tool is installed at `~/.claude/skills/local-music` (this skill's folder). Use the
commands as written: `~` works in Bash, and `cygpath -w` gives `cmd` the Windows form.

```bash
curl -s http://127.0.0.1:8192/api/health                                           # JSON = the server is up
cmd //c "$(cygpath -w ~/.claude/skills/local-music/start.cmd)" --no-browser        # if not: its own window, ~60 s
PY=~/.claude/skills/local-music/.venv/Scripts/python.exe
M=~/.claude/skills/local-music/scripts/music.py
$PY $M "A warm lo-fi hip hop beat with dusty piano chords, soft vinyl crackle and a laid-back boom bap groove" --bpm 84 --duration 60 --wait 180 --out <dir>
```

That makes 4 takes, prints them ranked with scores, tempo, key and PNG paths, and copies the
best to `<dir>` as `warm-lofi-hip-hop_v01.wav` + `.mp3` + a `.json` sidecar. Read the PNG of
the top take. Send files with SendUserFile if you have it (the MP3 plays inline); otherwise
give the paths.

## How a request goes

"I need calm exploration music for my game's forest level, and a short loop for the menu."

1. **GPU and server** (section 1). Start the server if needed.
2. **Write the prompt** (section 3): genre, instruments and how they sound, mood and energy,
   tempo. One piece per request.
3. **Generate**: `"Calm fantasy exploration music, gentle harp arpeggios, warm strings, soft
   woodwinds, a sense of wonder" --duration 120 --bpm 80`; the menu loop: `... --bpm 90
   --loop-bars 8`.
4. **Read the ranking** (section 4): failed takes are listed but not delivered. Check the
   warnings, the tempo and key against what you asked, and Read the top take's PNG.
5. **Iterate** (section 5): more takes, a rephrased prompt, `--vary` the best one, `--extend`
   it, `--inpaint` a weak spot, `--master` to change length, loudness or loop without
   regenerating.
6. **Deliver** to the folder the user named (`--out`), and **report**: paths, each take's ref
   and seed, scores, tempo/key, warnings, and that the server still holds GPU memory.

**When to ask the user:**
- How it sounds: you can't tell. Offer concrete knobs ("slower", "less busy drums", "darker",
  "more strings", "no piano", "a real ending") rather than asking open-endedly.
- Before stopping another program that is using the GPU: it may be in use.
- Where the files go, if they didn't say. Never write into their project unasked.

## Ground rules

- **Everything stays local.** The server listens on 127.0.0.1 only and never touches the
  network. Don't fall back to a cloud music service without asking.
- **Instrumental only.** The model makes no intelligible vocals or lyrics; for a song with
  words, say so rather than trying.
- **Licence** (`LICENSE-MODELS.md`): Stability AI Community License. The user owns what they
  generate; commercial use needs a (free) registration with Stability and is limited to
  individuals and organisations under US$1M a year in revenue. Mention it once if they plan
  commercial use.
- `README.md` has the full details for humans.

## 1. GPU and server

```bash
curl -s http://127.0.0.1:8192/api/health            # engines loaded, busy, VRAM
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

This needs about 7 GB of a 16 GB card loaded and up to ~12 GB while making long tracks, so
other large GPU programs can't run alongside it. If the `local-gpu` skill is installed, use it
to see what holds the GPU and to switch (it asks before stopping anything):
`python ~/.claude/skills/local-gpu/gpu.py switch local-music`. Otherwise check `nvidia-smi`
and **ask the user** before stopping other GPU programs.

**Start it in its own window** (it returns at once; the window is the user's to close):

```bash
cmd //c "$(cygpath -w ~/.claude/skills/local-music/start.cmd)" --no-browser
```

Ready in ~45-60 s; `--wait 180` on the first `music.py` call covers it. If it doesn't come
up, read `~/.claude/skills/local-music/logs/server.log` (previous run: `server.previous.log`).
If `.venv` or `models/` is missing, the tool isn't installed: `install.ps1` downloads ~11 GB
and needs a Hugging Face login that has accepted the Stable Audio 3 licence, so ask the user.

**Stop it** when the user is done with music or needs the GPU, after asking:
`cmd //c "$(cygpath -w ~/.claude/skills/local-music/stop.cmd)"`.

## 2. Commands

The CLI is standard-library Python; run it with the tool's own interpreter (plain `python` may
be the Microsoft Store stub):

```bash
$PY $M "Driving progressive house, rolling bassline, euphoric supersaw chords" --bpm 126 --duration 90 --out <dir>
$PY $M "A gentle solo piano ballad, intimate and tender" --stem --key "E minor" --out <dir>
$PY $M "8-bit chiptune boss battle, fast arpeggios, square wave lead" --bpm 160 --loop-bars 8 --out <dir>
$PY $M "Calm ambient soundscape, evolving warm pads, distant chimes" --duration 240 --lufs -18 --fade-in 4 --out <dir>
$PY $M --extend <ref> --seconds 60 --out <dir>                 # continue a take: the whole, longer track
$PY $M --vary <ref> --strength 0.3 --n 4                        # variations of a take
$PY $M --inpaint <ref> --span 20-28 "a drum fill"              # redo that span, keep the rest
$PY $M --master <request id> --loop-bars 4                      # re-master: no regeneration
$PY $M --master <request id> --no-loop --fade-out 5 --lufs -16
$PY $M --import <file> "a slow acoustic folk song"            # the user's audio as a take (ref <id>:0)
$PY $M "a warm music box melody" --vary <file> --strength 0.5    # files work directly too
$PY $M "a big drum fill and a riser" --combine <intro ref> <main ref>@0-40 <outro ref>   # one song
$PY $M --mix <song ref>,vol=-9 <voice ref>,at=4                 # a voice over a song (import the voice first)
$PY $M --show-prompt "dark synthwave" --bpm 100                 # what the engine would get
$PY $M --list          $PY $M --settings          $PY $M --engines
```

A **ref** is `<request id>:<seed>`, printed for every take. `--json` prints the server's full
response (plus `delivered`): parse that rather than the text.

| Flag | Meaning |
|---|---|
| `--duration S` | 2-380 s (default 30). Pick what the use needs: a menu loop 20-40 s, a level track 2-4 min. |
| `--bpm N`, `--key K` | Added to the prompt and checked. Ask for a tempo whenever it matters (loops, editing to picture). |
| `--stem` | One instrument or part ("a drum loop", "solo cello") instead of a full arrangement. |
| `--n N` | Takes (default 4, max 8). Generation is cheap (seconds); checks take a few seconds per minute of music. |
| `--seed N` / `--seeds A,B` | Reproduce takes. |
| `--length S`, `--fade-in S`, `--fade-out S` | Cut, fades. `--fade-out` unset = automatic (3 s, only if the take stops at full level); `0` keeps a hard cut. |
| `--lufs X`, `--peak X` | Loudness (default -14 LUFS: music on its own; -16 to -20 under dialogue or as a background bed) and true-peak ceiling (default -1 dBTP). |
| `--loop-bars N` | A seamless loop of N bars (4/4 unless `--beats-per-bar`). Give `--bpm` when generating. |
| `--out DIR` | Deliver the best (`--best N`, `--all`) as `<name>_vNN.wav/.mp3/.json` (`--format`, `--png`, `--name`). Never overwrites. |
| `--play` | Play the best take on the user's speakers (only if they asked to hear it). |
| `--no-save` | Delete the request from `output/` afterwards. |

Exit codes: 0 done, 1 no server or failed, 2 bad arguments, 3 every take rejected.

## 3. Writing prompts

Stable Audio 3 was trained on library music (AudioSparx) and Freesound with their
descriptions: write like a music-library catalogue entry, not like a chat request.

- **Genre and era** first: "1980s synthwave", "baroque chamber music", "UK garage".
- **Instruments, and how they sound**: "dusty Rhodes piano", "gated reverb drums", "warm
  analog bass", "pizzicato strings", "overdriven guitars".
- **Mood and energy**: "melancholic", "triumphant", "tense and pulsing", "laid-back".
- **Production / space** helps: "lo-fi tape warmth", "close-mic'd", "big cinematic hall".
- **Tempo** with `--bpm` (or "120 BPM" in the text), **key** with `--key`.
- A **structure** hint can work for longer pieces ("builds to a climax", "with a quiet middle
  section"), loosely.
- Don't ask for lyrics, vocals, artist or song names; describe the style instead.

The tool adds `TrackType: Music, VocalType: Instrumental` (or `TrackType: Instrument` with
`--stem`), the tempo and the key. `--raw` turns that off.

## 4. Reading the results

Each take prints (and `--json` returns):

| | Good | Look again |
|---|---|---|
| **CE** (Audiobox content enjoyment, 1-10) | 7-8 | under 6 |
| **PQ** (production quality) | 7-8.5 clean productions; 6-7 is normal for deliberately lo-fi music | under 6 warns, under 5 fails |
| **CLAP** (match with the description) | 0.4-0.55 | under 0.25: it may not be what was asked |
| **tempo** | `(exact)`, or `half`/`double` (the same grid) | a warning: another tempo |
| **key** | `(exact)`; `relative`/`fifth`/`parallel` are near misses | a warning: another key; `unclear` = the detector can't tell (busy or jazzy harmony): not the model's fault |

**Warnings and what to do:**
- *drops to silence inside the music*: a gap; if it wasn't asked for (a "quiet middle section"
  can legitimately drop), use another take, or `--inpaint` the span.
- *ends abruptly at full level*: only with `--fade-out 0`; otherwise it was faded automatically
  (`post.auto_fade_out_s`). For a real ending: `--inpaint <ref> --span <last 8 s> "<the
  prompt>, coming to a gentle ending on a final sustained chord" --n 4 --fade-out 0`; about
  half the takes end properly (`signal.end_level_db` around -20), the rest still carry this
  warning and rank lower.
- *the music ends at X s and the rest is silence*: trimmed off when mastering; the delivered
  file is shorter than asked. Ask for less, or extend.
- *tempo is N BPM, not the M asked* / *sounds in K, not the K asked*: use another take, or
  regenerate; mention it if every take misses.
- *limited hard*: the track is dense; a lower `--lufs` keeps more punch.
- *loop... may be heard* / *the music changes across the loop point*: try another take, or a
  different number of bars (a 4-bar phrase loops better as 4 or 8 bars than as 2).

**Read the PNG** of the takes you deliver: the top strip is the waveform, the middle the
short-term loudness (builds, drops, the ending), the bottom the spectrogram (bass, drums,
pads, brightness), with numbered bars. Look for gaps, sudden ends, a flat loudness line where
the user wanted a build, or an empty band (no bass, no highs) that contradicts the prompt.

## 5. Iterating

- **Not what was asked**: rephrase the prompt (more specific instruments, mood words), more
  takes (`--n 6`), or a new seed.
- **Close but not quite**: `--vary <ref> --strength 0.2-0.4` keeps the sound and style and
  rewrites details; 0.5-0.7 rewrites the notes; a new text steers the variation.
- **Longer**: `--extend <ref> --seconds S` (repeatable; past 380 s it continues from the last
  part). The result is the whole track.
- **One bad section**: `--inpaint <ref> --span A-B` (seconds on the take's timeline), with a
  new text for that part if wanted ("a drum break", "the strings drop out").
- **Different length, loudness, fades, or a loop**: `--master <request id> ...`, no
  regeneration.
- **From the user's own audio**: `--import <file> "what it is"`, then work from its ref; or pass
  the file to `--extend` / `--vary` / `--inpaint`. A melody on another instrument: `--vary` at
  0.4-0.6 with the new instrument as the text. Don't import recordings the user has no right to
  rework without saying so.
- **A song from sections**: make each part separately with the **same `--bpm` and `--key`**
  (an intro without drums, a main part, a breakdown, an outro), then `"<what the bridges
  are>" --combine A B C` (text first). Bridges default to 2 bars; `--bridge 0` blends parts
  straight into each other. Parts are cut to their body at the joins (fades and silence would
  make silent bridges); `REF@START-END` picks a span yourself. Read the result's PNG at each
  join: the bar grid should run through, the loudness shouldn't fall into a hole.
- **A voice (or any layer) over a song**: `--import voice.mp3`, then `--mix <song ref>,vol=-9
  <voice ref>,at=4`. Each layer is at the volume you give (default 0 dB, both at the same
  loudness); nothing is ducked or levelled for you: the user decides the balance, so ask, or
  offer two or three balances (song at -6 / -9 / -12) rather than picking one silently.

## 6. Delivering

`--out <dir>` copies the best take (or `--best N` / `--all`) as WAV (48 kHz 24-bit) + MP3
(320 kbps) + a JSON sidecar with the prompt, seed, model, scores, tempo and key. Report to the
user: the delivered paths; each take's ref, seed, CE/PQ/CLAP, tempo and key; any warnings
you didn't resolve; what you looked at in the PNG; and that the server is still holding GPU
memory (offer to stop it when they're done).

Everything a request made stays in `~/.claude/skills/local-music/output/<request id>/`
(raw take, master, MP3, PNG, `request.json`); `--no-save` or `--delete <id>` removes it.
