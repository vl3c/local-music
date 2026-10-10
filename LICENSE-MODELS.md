# Model licences

**Powered by Stability AI.** (The Stability AI Community License asks products and services
built on its models to say so prominently.)

This repo's own code is MIT (see `LICENSE`). It does **not** include or redistribute any
model weights or third-party code: `install.ps1` fetches them from their upstream sources,
and they stay under their own licences. Check them yourself before any commercial use.

| Component | Source | Licence | What it means here |
|---|---|---|---|
| Stable Audio 3 Medium weights | [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium) (gated) | [Stability AI Community License](https://stability.ai/license) (copy in `models/sa3-medium/LICENSE.md` after install) | Free for non-commercial use, and for commercial use by individuals and organisations under the revenue threshold the licence sets (US$1M a year; registration required); above it, an Enterprise licence. Not OSI open source. You accept it on Hugging Face before downloading. You own the music you generate ("Ownership of Outputs"), to the extent the law allows. Use must follow Stability's [Acceptable Use Policy](https://stability.ai/use-policy), and neither the model nor its outputs may be used to train other foundation models. |
| Stable Audio 3 Small-Music weights (optional) | [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music) (gated) | Stability AI Community License | Same terms. Only downloaded with `-WithSmall`. |
| T5Gemma text encoder (bundled with Stable Audio 3) | Google | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (`models/sa3-medium/LICENSE_GEMMA.md`) | Covers the encoder inside the Stable Audio 3 download. |
| stable-audio-3 inference code | [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) (cloned into `engine/`) | MIT | |
| CLAP (larger_clap_general) | [laion/larger_clap_general](https://huggingface.co/laion/larger_clap_general) | Apache-2.0 | Used only to score takes against their prompt. |
| Audiobox Aesthetics | [facebook/audiobox-aesthetics](https://huggingface.co/facebook/audiobox-aesthetics), [code](https://github.com/facebookresearch/audiobox-aesthetics) | CC-BY-4.0 (WavLM parts MIT) | Used only to score enjoyment and production quality. |
| librosa | [librosa/librosa](https://github.com/librosa/librosa) | ISC | Tempo, beats and key. Installed as a library, not redistributed. |
| pyloudnorm | [csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | MIT | Its BS.1770 K-weighting filters. |

## Why these models

Chosen from the [Artificial Analysis instrumental music leaderboard](https://artificialanalysis.ai/music/leaderboard/instrumental)
(October 2026) as the best model there whose weights can be downloaded and that runs on a
16 GB card:

| Model | Elo | Why not / why |
|---|---|---|
| Suno, Mureka, Lyria, Eleven Music, MiniMax 2.x | 1002-1140 | Closed: API only. |
| Stable Audio 3 Large | 1017 | API / enterprise only; no public weights. |
| MiniMax Music 3.0 | 1000 | Open weights (MiniMax-Music3 Community License), but an ~11B song model built for lyrics; ~22 GB of VRAM at full precision (CPU offloading below that) and 57 GB of files. Statistically tied with Medium on this board. |
| **Stable Audio 3 Medium** | **998** | **Open weights, 1.4B, ~7 GB, up to 380 s, generates minutes of music in seconds.** |
| Stable Audio 3 Small-Music | 948 | Open weights, smaller and lower quality: the optional second engine. |
| MusicGen | 881 | CC-BY-NC weights, far behind. |
