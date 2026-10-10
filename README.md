# local-music

**Powered by Stability AI** (Stable Audio 3).

A private, fully local **instrumental music generator**, built for AI agents (and people) to
call. Describe the music (genre, instruments, mood, tempo, key) and get back mastered
tracks: checked, scored and ranked best-of-N, as 48 kHz WAVs and MP3s, with a waveform,
loudness and spectrogram picture of each so an agent that can't listen can still judge them.
Continue a track, make variations, redo a section, or cut a seamless loop on the bar; start
from your own audio file; join several takes into one song, with new music written at each
join; or mix takes on top of each other (a voice over a song), each at its own volume.

It runs **Stable Audio 3 Medium**, the best open-weights model on the
[Artificial Analysis instrumental leaderboard](https://artificialanalysis.ai/music/leaderboard/instrumental)
that fits a 16 GB card (see [LICENSE-MODELS.md](LICENSE-MODELS.md) for the comparison), on
your own GPU. Up to 6 min 20 s of stereo music per take, in seconds. No accounts, no cloud:
prompts and audio never leave the machine.

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -AcceptModelLicenses   # ~17 GB: weights + venv
.\start.cmd                                                                  # opens http://127.0.0.1:8192/
```

> **Licences, plainly.** This repo's code is MIT. Stable Audio 3 is under the **Stability AI
> Community License**: free for non-commercial use and for commercial use by individuals and
> organisations under US$1M a year in revenue (register with Stability first). As between you and
> Stability, you own what you generate, to the extent the law allows; its Acceptable Use Policy
> applies, and you may not use the model or its outputs to train other foundation models. It is
> gated on Hugging Face: accept it there and run `hf auth login` before installing. `-AcceptModelLicenses` says you've read [LICENSE-MODELS.md](LICENSE-MODELS.md).
> Windows and an NVIDIA GPU are required.

## For AI agents (Claude Code skill)

`SKILL.md` is a Claude Code skill: how to prompt, judge takes without hearing them, iterate
and deliver. `install.ps1` links this folder as `~\.claude\skills\local-music` (`-NoSkillLink`
to skip). Then ask Claude Code for "a 2-minute calm piano piece for my game's menu" or "an
8-bar chiptune loop at 160 BPM".

Any other agent can use the CLI (`scripts/music.py`, standard library only) or the HTTP API
below:

```powershell
python scripts\music.py "warm lo-fi hip hop, dusty piano chords, vinyl crackle" --bpm 84 --out D:\music
python scripts\music.py "8-bit chiptune boss battle" --bpm 160 --loop-bars 8 --out D:\music
python scripts\music.py --extend 20261010-153631-080e:84123544 --seconds 60
python scripts\music.py "a warm music box melody" --vary D:\audio\demo.wav --strength 0.5
python scripts\music.py "a drum fill and a riser" --combine <intro ref> <main ref> <outro ref>
```

## How it works

1. **The prompt.** Your description gets the tags Stable Audio 3's prompt guide recommends
   (`TrackType: Music, VocalType: Instrumental`, or `TrackType: Instrument` for a single
   instrument with `--stem`), plus the tempo ("84 BPM") and key ("in A minor") if you give
   them. `--raw` sends the text as written; `--show-prompt` shows what would be sent.
2. **The takes.** The engine makes N takes (default 4) on consecutive seeds, at 44.1 kHz stereo.
   Its output is kept in full float (the library clamps to full scale, so the worker gives
   the decoder 12 dB of headroom first and takes it back after; no peaks are clipped).
3. **The checks**, on every take, for a caller that can't listen:
   - *signal*: silence, clipping (flat tops), dropouts inside the music, an early end (the rest
     silence), an abrupt end at full level, out-of-phase stereo, DC offset;
   - *musical*: the tempo (librosa's beat tracker, then a straight line through every beat:
     within a fraction of a BPM) and the key (Krumhansl-Schmuckler on the chroma), compared
     with what was asked, allowing half/double time and noting relative/parallel keys;
   - *scores*: Audiobox Aesthetics (content enjoyment CE, production quality PQ, complexity PC,
     usefulness CU, each 1-10, over 10 s windows) and CLAP's match between the music and
     your description (cosine, averaged over up to six 10 s windows).

   A take fails on silence, heavy clipping or PQ under 5; warnings cost rank. Takes are ranked
   by CE and PQ (35% each) and CLAP (30%).
4. **Mastering.** Silence before and after the music is trimmed; a take that stops at full
   level (many do: the model learned from crops of longer tracks) gets a 3 s fade-out
   automatically (`--fade-out 0` keeps the cut); then loudness normalisation to **-14 LUFS**
   (ITU-R BS.1770 integrated, what streaming services play at) and a look-ahead limiter to a
   **-1 dBTP** true-peak ceiling, re-measured after limiting so it lands on target. Delivered
   as 48 kHz 24-bit WAV (prompt, seed, model and scores in its INFO chunk) and 320 kbps MP3.
5. **Loops** (`--loop-bars N`): the take's beat grid is tracked, every beat after the first
   second is a candidate start, the loop ends exactly N bars later, and the candidate whose
   two ends sound most alike (log-mel similarity over a beat either side, and matching level)
   wins; its end is nudged to the sample with the best waveform match, and what follows the
   end is crossfaded into the start (50 ms), so playing it end-to-start just continues. The
   take reports the loop's BPM, its seam similarity and the level match.

### From an earlier take

| | What the engine does | Result |
|---|---|---|
| **extend** `--extend REF --seconds S` | hears the take and writes the next S seconds (inpainting past its end) | the whole, longer track |
| **vary** `--vary REF --strength X` | starts from the take instead of from noise (audio-to-audio): 0.1 is close, 0.5 keeps the style and sound but rewrites the notes, 1 is unrelated | variations, same length |
| **inpaint** `--inpaint REF --span 20-28 "a drum fill"` | regenerates that span (seconds as the take plays), joins it musically | the whole track, the rest exactly as it was |

A new text replaces the prompt for that part (default: the source's). Extending past the
engine's 380 s window works too: it hears the last part of the track. A fade-out or silence at
the end of the source is left out first (with a warning): given those, the engine would
continue the silence.

**Only the new part is the engine's.** The engine re-renders everything it hears, not just
what it was asked to write: the same music, but not the same waveform (measured 14-23 dB apart),
and a voice would come back as the model's imitation of one. So the server puts the original
back everywhere outside the region the engine was asked to write (after the source for extend,
the span for inpaint, the bridge and blend for joins), with 0.1 s linear crossfades where they
meet. Tests check this sample for sample. A vary is the exception by nature: it's all new.

### Your own audio

`--import FILE ["what it is"]` (`--stem` if it's one instrument; or **Start from your own audio** on the page) turns a WAV, FLAC,
OGG or MP3 into a take of an `import` request (seed 0): checked, measured and mastered like a
generated one, nothing generated. Everything above works from its ref: extend it into a longer
piece, vary it (timbre or style transfer: a melody played on another instrument at strength
~0.4-0.6), inpaint a span, loop it, master it, or put it in a song. The description becomes the
prompt that later work continues from; without one, the engine is led by the audio alone (and
CLAP has nothing to score). `--extend`, `--vary`, `--inpaint` and `--mix` take a file path directly
too (imported first). Silence at the file's start and end is trimmed like a generated take's;
`--keep-silence` (API `trim=false`) keeps it, so the take keeps the file's own timeline (a voice
whose word timings come from elsewhere; files given to `--mix` always keep it). Up to 20 minutes
and 300 MB; other formats (M4A, AAC) need converting first.
What you may do with someone else's recording is up to its licence: the tool doesn't check.

Measured: Windows' `Alarm01.wav` (5.6 s) extended by 40 s as "an upbeat tropical house track
built around a bright marimba melody" came back as music (CE 7.4, PQ 8.4); varied at 0.5 as
"a warm music box melody", its CLAP match to that description rose from 0.37 to 0.45.

### Mixing takes on top of each other

`--mix SONG:1,vol=-8 voice.wav,at=4,voice` (or **Mix on top** in the page's Song tray) layers
takes: each is the take's delivered track (what you hear: its fades, cut or loop, at its
loudness), at its own volume (`vol=`, dB, default 0) starting at its own time (`at=`, seconds,
default 0), optionally a span (`REF@10-40`). A layer can be an audio file: it's imported first,
silence and all, so `at` and the spans below line up with the file's own timeline. The mix runs
as long as its longest layer. Nothing levels or ducks the layers: the balance is yours. The
sum is mastered like any take (so the overall loudness is the usual -14 LUFS and the volumes
set the balance between layers), and it's a take itself: extend it, loop it, re-master it, or
mix it again.

- **vol=0 means "as loud as it's delivered"**: each layer's loudness in the mix is printed
  (`plays at -17 LUFS`). A song delivered at -17 LUFS and a voice imported at -14 sit 3 dB
  apart at vol=0; a voice over a busy song typically wants the song another -6 to -12.
- **Layers are mixed before their own limiter**: the mix is limited once, as a whole, so a
  voice's loud moments (a shouted or sung last word) come through as they were.
- **Edits per layer**: `boost=START-END:DB` raises or lowers it there (seconds on the layer's
  own timeline; repeatable; 30 ms ramps), e.g. `boost=60.9-63.5:+4` for a held last word;
  `reverb=room|hall|cathedral` puts a dry layer in a space (a light room: `room`), keeping its
  loudness, so its `vol` means the same.
- **`voice` marks speech**: the mix measures, while it speaks, how far the voice sits above
  everything else in the speech band (300-3400 Hz), and warns where it falls under the rest
  for a second or more, or when its quieter tenth is less than 6 dB above it: words there may
  be lost (calibrated against Whisper on real mixes). The numbers are in the layer's
  `speech_margin`. A proxy: listening, or transcribing the mix, is the real test.

### Joining takes into one song

`--combine A B C` (or **Add to song** on takes, then **Combine** on the page) joins takes in
order. At each join the engine hears up to 20 s of the part before and after, writes `--bridge`
seconds of new music between them (default 2 bars at the first part's tempo, 4 s without one;
0 blends the parts straight into each other) and rewrites `--blend` seconds (default 1) of each
part beside it so they flow into it. Everything else is the parts' own audio, sample for
sample, each brought to its delivered loudness first (an imported recording joins as loud as it
plays on its own; `,vol=DB` sets a part quieter or louder than that: `REF,vol=-6` for a quiet
intro before a loud main part). The text describes the bridges
("a big drum fill and a rising synth sweep"; default: the first part's prompt).

- A part can be a span of a take: `REF@20-60` (seconds as the take plays, as on its picture;
  `REF@30-` from 30 s to the end). Spans can't be taken from a take delivered as a loop.
- At inner joins a part is cut to its *body*: a fade-out, a final chord dying away or silence
  before a join (and a fade-in after one) would tell the engine the music had ended, and it
  wrote silent bridges. A span you give sets that edge yourself; the song's very start and end
  stay as they are.
- Parts at different tempos or in unrelated keys get a warning: the bridge has to change
  them, and may sound abrupt. Generating the parts with the same `--bpm` and `--key` avoids it.
- Each take of the result is the whole song, its bridges made with one seed (a few seconds per
  join), then checked, scored and mastered like any take.

Measured: an intro (pads, no drums), a main part (drums, bass, lead) and an outro, each made
separately at 100 BPM, joined with "a big drum fill and a rising synth sweep": 109 s at 100.0
BPM, CE 7.4, PQ 8.2, no warnings; the drums build in across the first bridge, and the bar grid
runs through both joins unbroken.

### Re-mastering

`--master REQ` re-runs mastering on every take of a request, with no regeneration: length,
fades, loudness, peak ceiling, or a loop (`--master REQ --loop-bars 4`; `--no-loop` for the
whole track again). It takes a second or two per take.

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11 |
| GPU | NVIDIA, 16 GB: the server holds ~6.8 GB loaded and peaks around 12 GB making three 380 s takes. RTX 50-series supported (CUDA 12.8 wheels) |
| Disk | ~17 GB (weights ~11 GB, venv ~6.3 GB) plus pip cache |
| Driver | NVIDIA driver 570 or newer (for the CUDA 12.8 PyTorch wheels) |
| Software | Python 3.10+ (3.13 by default, via the `py` launcher), Git |
| Hugging Face | an account that has accepted the Stable Audio 3 licence, logged in with `hf auth login` |

FlashAttention isn't needed: Medium's autoencoder uses sliding-window attention, which the
engine computes with chunked SDPA when FlashAttention (no Windows wheels) and Triton are
missing. The worker skips the engine's flex-attention path in that case: without Triton,
torch would run it eagerly and materialise the whole attention matrix (fine for seconds of
audio, out of memory for minutes).

## Install

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -AcceptModelLicenses
```

It clones the stable-audio-3 inference code at a pinned commit into `engine\`, builds `.venv\`
with PyTorch 2.7.1 + CUDA 12.8 (what the library pins), downloads the weights into `models\`
and links the Claude Code skill. Re-running is safe: finished steps are skipped and
downloads resume.

**Weights are pinned:** `models.json` fixes each Hugging Face repo to a revision, and
`models.lock.json` every file's size and SHA256, so an upstream re-upload fails loudly instead
of silently changing what you get.

| Model | Size | Role | Licence |
|---|---|---|---|
| [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium) | 10.4 GB | the engine (with its T5Gemma text encoder) | Stability AI Community |
| [CLAP larger_clap_general](https://huggingface.co/laion/larger_clap_general) | 0.74 GB | prompt match | Apache-2.0 |
| [Audiobox Aesthetics](https://huggingface.co/facebook/audiobox-aesthetics) | 0.4 GB | enjoyment and production quality | CC-BY-4.0 |
| [Stable Audio 3 Small-Music](https://huggingface.co/stabilityai/stable-audio-3-small-music) (optional, `-WithSmall`) | 2.3 GB | a lighter engine, up to 120 s | Stability AI Community |

(`laion/larger_clap_music` would be the obvious CLAP for music, but its text tower is broken:
every text gets the same embedding. The general one was trained on music too.)

Options: `-WithSmall`, `-SkipModels` (no weights; no licence needed), `-ReuseModels <other
install>\models` (hard-link matching weights instead of downloading; checksums decide, and a
file may come from any model folder there: other Stable Audio 3 installs carry the same
T5Gemma encoder), `-PythonVersion 3.12`, `-CudaIndex cu130`, `-NoSkillLink`, `-NoPipCache`.
Files already in your Hugging Face cache are reused automatically.

Verify the weights at any time without downloading anything:
`.venv\Scripts\python.exe scripts\download_models.py --verify`

## Using it

**`.\start.cmd`** loads the engine (~35 s) and the checkers, then opens the page. It always
runs in its own console window titled "local-music server - close this window to stop it",
and `start.cmd` returns at once, so a server is never running out of sight; closing the window
stops it and frees the GPU. `start.cmd --no-browser` does the same without opening the page
(that's how agents start it); `--lazy` loads nothing until the first request. Started any other
way (`python server\app.py`, a hidden background shell) the server refuses to run. A second
start while one is running exits at once ("port in use"). `.\stop.cmd` stops the server
recorded in `output\.server.pid` and its engine worker, however it was started.

### Command line (what agents use)

```powershell
python scripts\music.py "Driving progressive house, rolling bassline, euphoric supersaw chords" --bpm 126 --duration 90
python scripts\music.py "A gentle solo piano ballad, intimate and tender" --stem --key "E minor" --out D:\music
python scripts\music.py "Calm ambient soundscape, evolving warm pads" --duration 240 --lufs -18 --fade-in 4
python scripts\music.py --vary 20261010-153631-080e:84123544 --strength 0.3 --n 4
python scripts\music.py --inpaint 20261010-153631-080e:84123544 --span 10-18 "a short drum solo"
python scripts\music.py --master 20261010-153631-080e --loop-bars 4
python scripts\music.py --list
python scripts\music.py --settings
```

| Flag | Meaning |
|---|---|
| `--duration S` | 2-380 s on Medium, 2-120 on Small (default 30). |
| `--bpm N` / `--key K` | Added to the prompt and checked in the result. Keys: `A minor`, `F# major`, `Bbm`... The model follows the key on clearly tonal music (a solo piano in E minor came out in E minor 3 times out of 3); busy or jazzy harmony reads as "unclear". |
| `--stem` | One instrument or part instead of a full arrangement. |
| `--n N` | Takes to make and rank (default 4, max 8). |
| `--seed N` / `--seeds A,B` | Start seed (takes use N, N+1...) or exact seeds; `random` by default. |
| `--engine medium\|small` | Medium (default) or Small-Music (installed with `-WithSmall`). |
| `--raw` | Send the text as written: no tags, tempo or key added. |
| `--no-check` | Skip CLAP and Audiobox (the signal, tempo and key checks still run). |
| `--length S` | Cut the delivered track to S seconds, with a fade. |
| `--fade-in S` / `--fade-out S` | Fades. `--fade-out` unset means auto (3 s, only when the take stops at full level); `0` keeps a hard cut. |
| `--lufs X` / `--peak X` | Loudness target (default -14; -16 to -18 sits under dialogue) and true-peak ceiling (default -1 dBTP). |
| `--loop-bars N` / `--beats-per-bar N` | Deliver a seamless loop of N bars (give `--bpm` when generating so the take is long enough); 4 beats per bar by default. |
| `--extend REF --seconds S`, `--vary REF --strength X`, `--inpaint REF --span A-B` | From an earlier take, or an audio file (above). |
| `--import FILE`, `--keep-silence` | Your own audio as a take; the text describes it. `--keep-silence` keeps the file's timeline. |
| `--combine PART PART...`, `--bridge S`, `--blend S` | Join takes into one song; parts are `REF` or `REF@START-END`, optionally `,vol=DB`. Put the text before `--combine`. |
| `--mix LAYER LAYER...` | Takes or audio files on top of each other: `REF[@START-END]` or a file, then `,vol=DB`, `,at=SECONDS`, `,boost=START-END:DB`, `,reverb=room\|hall\|cathedral`, `,voice`. |
| `--master REQ` | Re-master a request with the mastering flags; `--no-loop`, `--full-length` undo a loop or a cut. |
| `--out DIR` | Copy the best take (`--best N`, `--all`, or `--pick SEEDS` for those very takes) as `<name>_vNN.wav` + `.mp3` + `.json` sidecar (`--format wav\|mp3`, `--png` for the plots, `--name` to rename). Never overwrites. |
| `--show REQ:SEED --out DIR` | Deliver one take of an earlier request, whatever its rank. |
| `--export REQ:SEED --out DIR` | The take and every request it was made from (sources, parts, layers, all the way back), copied to DIR with a manifest: copy its `requests/` folders back into `output/` and every ref works again. Do this before cleaning up `output/`. |
| `--play`, `--no-save`, `--json`, `--wait S` | Play the best take; delete the request afterwards; print the server's JSON; wait for the server. |
| `--show-prompt`, `--list`, `--show REQ`, `--settings`, `--engines`, `--delete REQ` | Inspect and tidy up (`--show`: an earlier request's takes and files). `--delete` refuses a request another one was made from (a mix of it, an extension...); `--force` deletes it anyway. |
| `--save-defaults` / `--reset-defaults` | Save the options given with it (`--n`, `--duration`, `--lufs`, `--fade-out`...) as your defaults, merged with what's saved; or forget them. The page's **Save as my defaults** does the same. |

A take's **ref** is `<request id>:<seed>`. Exit codes: 0 done, 1 no server or failed, 2 bad
arguments, 3 every take was rejected.

**Your own defaults.** On the page, **Save as my defaults** stores the panel's settings in
`config.local.json` (local, gitignored); they apply to the page and to every API/CLI request
that leaves a setting out. **Reset to built-in** deletes it. `music.py --settings` shows what's
in effect.

**The server's log.** The window shows milestones, one line per request and any errors; the
full output goes to `logs\server.log`, timestamped. Each start keeps the previous run as
`logs\server.previous.log`.

### HTTP API (127.0.0.1:8192)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/generate` | `{"text", "duration"?, "bpm"?, "key"?, "kind"?: "music"\|"stem", "n"?, "seed"?, "seeds"?, "engine"?, "steps"?, "raw"?, "negative"?, "name"?, "check"?, + mastering}` | the request: id, prompt, and its takes ranked, each with `ref`, `files`/`paths` (wav, mp3, png), `signal`, `music` (tempo, key), `aes`, `clap`, `post`, `final` (LUFS, true peak), `fails`, `warnings`, `rank_score` |
| `POST /api/extend` | `{"take", "seconds"?, "text"?, + shape and mastering}` | the same, for the longer track |
| `POST /api/vary` | `{"take", "strength"?, "text"?, ...}` | variations |
| `POST /api/inpaint` | `{"take", "start", "end", "text"?, ...}` | the take with that span redone |
| `POST /api/import?filename=&text=&kind=&name=&check=&trim=` | the file's bytes (`application/octet-stream`) | an `import` request with one take (seed 0) |
| `POST /api/mix` | `{"layers": [{"take", "gain_db"?, "at"?, "start"?, "end"?, "spans"?: [{"start", "end", "gain_db"}], "reverb"?, "voice"?}, ...], "name"?, "check"?, + mastering}` | the mix, as one take (seed 0); each layer with its `lufs` in the mix and, for a voice, its `speech_margin` |
| `POST /api/arrange` | `{"parts": [{"take", "start"?, "end"?, "gain_db"?}, ...], "bridge"?, "blend"?, "text"?, "name"?, + shape and mastering}` | the song, as takes |
| `POST /api/master` | `{"request", "length"?, "fade_in"?, "fade_out"?, "lufs"?, "peak_dbtp"?, "loop_bars"?, "beats_per_bar"?, "bpm"?}` | the request, re-mastered (`null` clears `length` / `loop_bars`) |
| `GET /api/requests`, `GET /api/requests/<id>`, `DELETE /api/requests/<id>?force=` | | list, one, delete (409 while another request was made from it, unless `force=true`) |
| `GET /api/prompt?text=&bpm=&key=&kind=&raw=` | | the prompt and file name a request would use |
| `GET /api/engines`, `POST /api/engines/<name>/stop` | | installed / running; stop one to free its VRAM |
| `GET` / `POST` / `DELETE /api/settings` | | effective, saved and built-in defaults |
| `GET /api/health` | | engines loaded, `busy`, VRAM |
| `GET /files/<id>/<name>` | | a WAV, MP3, PNG or JSON from `output/` |

Mastering fields on every generating request: `length`, `fade_in`, `fade_out`, `lufs`,
`peak_dbtp`, `loop_bars`, `beats_per_bar`. Bad input is a 4xx with a message, never a 500.
Requests must use Host `127.0.0.1` or `localhost`, and writes from other web pages are refused.

### Files

Every request is a folder `output\<id>\` (clear it any time):

| File | What |
|---|---|
| `request.json` | the request, the prompt sent, every take's checks and scores, the beat grid |
| `<seed>_raw.wav` | the engine's output: 44.1 kHz float stereo, unmastered |
| `<seed>.wav` / `.mp3` | the delivered track (or loop), 48 kHz 24-bit / 320 kbps |
| `<seed>.png` | waveform, short-term loudness and spectrogram with the bar grid (a loop: rotated, seam marked) |

### The page

http://127.0.0.1:8192/ does the same: the description, seconds, BPM, key and kind up front,
mastering and fine-tuning folded away, then one card per take with its scores, tempo and key,
player, picture, downloads, **Add to song**, and **Extend / Vary / Inpaint**; per request,
**Re-master**, **Make loops** and **delete**. On the left, **Start from your own audio** imports
a file, and **Song** lists the takes you added: **Join in order** (reorder, trim each to a
span, set the bridge, **Combine**) or **Mix on top** (a volume slider and a start time per
layer, double-click a slider for 0 dB, **Mix**). `Ctrl+Enter` generates.

### Speed (RTX 5080)

| | |
|---|---|
| Start | ~45-60 s to ready (engine ~35 s, checkers ~10 s) |
| Generation | 3 takes of 30 s in ~1.5 s; of 60 s in ~3 s; of 380 s in ~13 s |
| Checks and mastering | 3 takes of 60 s in ~5 s; of 380 s in ~30 s (Audiobox, CLAP, tempo, key, mastering, MP3 and plot) |
| Extend / vary / inpaint | about the same as generating the result's length |

## Notes and gotchas

- **Endings.** Many takes stop rather than end (the training data was crops of longer tracks):
  that's what the automatic fade-out is for. For a real ending, inpaint the last seconds with
  `--fade-out 0` and a prompt that asks for one ("...coming to a gentle ending on a final
  sustained chord"): on a lo-fi take that stopped dead, 2 of 4 such takes ended properly
  (-20 dB at the end); the other two still stop, are flagged "ends abruptly" and rank lower.
- **Key** is followed on clearly tonal material and only loosely on dense or jazzy harmony, where
  the detector also struggles. Use it to pick between takes, not as a guarantee.
- **Vocals.** The model makes no intelligible vocals; it sometimes adds wordless vocal
  textures. `VocalType: Instrumental` (added by default) discourages them.
- **Negative prompts and `cfg`** do nothing on the post-trained checkpoints Medium and
  Small-Music ship as; the API accepts them and says they were ignored.
- **The scores are relative.** CLAP and Audiobox rank takes of the same request; they don't
  say whether music is good. Deliberately lo-fi music scores PQ 6-7, clean productions 7-8.5.
- Startup prints `flash_attn not installed` and a `weight_norm` deprecation: both harmless.

## Privacy

The server binds to 127.0.0.1 only (hard-coded) and forces Hugging Face offline, so after
installation nothing contacts the network. `output\` holds every take (gitignored).

## Before pushing

```powershell
python scripts\check_repo_safe.py            # what git would publish
python scripts\check_repo_safe.py --history  # and every past commit
```

It fails on weights, audio, images, large files, local paths, emails, IP addresses and
credential-shaped strings. It can't judge prose, so read your own diff too. CI runs it and the
test suite (`pytest tests`, no GPU or weights needed) on every push and pull request.

## Repo layout

```
install.ps1 / start.cmd / stop.cmd   setup, launch, shutdown
models.json, models.lock.json        pinned, checksummed weight manifest
requirements.txt                     the venv (server, checks and engine)
server/app.py, server/page.html      the loopback server and its page
server/workers/                      the engine worker (Stable Audio 3) and its line protocol
server/serverlog.py                  console filter + logs/server.log
music/                               analysis (tempo, key, loudness), checks, scoring,
                                     mastering and loops, plots, prompts, WAV/MP3 files
scripts/music.py                     CLI (standard library only)
scripts/download_models.py           resumable, verifying downloader
scripts/check_repo_safe.py           pre-publish scan
tests/                               pytest, with synthetic music and a fake engine
SKILL.md                             the Claude Code skill
engine/, models/, .venv/, output/, logs/   created at install / run time, gitignored
```
