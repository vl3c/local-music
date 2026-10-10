"""Stable Audio 3 worker: one model per process (`sa3_worker.py medium` or `... small`).
See protocol.py for the protocol.

The official stable-audio-3 library, loaded from models/ instead of the Hub: its
from_pretrained only takes Hub names (upstream issue #35), so this builds the model the way
from_pretrained does, with the text encoder pointed at the local folder. Small-Music uses
the same T5Gemma encoder as Medium (identical files), so it reads Medium's copy.

One job, three modes:
    text to music         prompt + duration
    audio to audio        + init_audio, init_strength (0 = a copy, 1 = unrelated): a variation
    inpaint / continue    + inpaint_audio, mask_start, mask_end (seconds): regenerate that span;
                          past the source's end it continues the music

Notes from upstream: the post-trained checkpoints ignore cfg_scale and the negative prompt
(only the -base checkpoints use them); 8 steps is their default, and more doesn't help.
Medium's autoencoder (SAME-L) uses sliding-window attention: without FlashAttention 2 (no
Windows wheels) the library computes it with chunked SDPA, which gives the same result.
"""

from __future__ import annotations

import protocol  # noqa: F401  (first: takes stdout for the protocol)

import json
import sys
from pathlib import Path

from protocol import MODELS, log, serve

MODELS_BY_NAME = {
    "medium": {"dir": "sa3-medium", "model": "stable-audio-3-medium", "max_duration": 380.0},
    "small": {"dir": "sa3-small-music", "model": "stable-audio-3-small-music", "max_duration": 120.0},
}
T5GEMMA = MODELS / "sa3-medium" / "t5gemma-b-b-ul2"
# generate() clamps the decoded audio to [-1, 1], and Medium decodes right up to full scale:
# ~0.05% of samples came out pinned at the rail. The decoder's output is scaled down by this
# before the clamp and back up after it, so peaks survive into the float WAV; mastering sets
# the delivered level anyway.
HEADROOM = 0.25
NAME = sys.argv[1] if len(sys.argv) > 1 else "medium"


def _no_flex_without_triton() -> None:
    """Without Triton (none on Windows by default) torch.compile can't build flex_attention, and
    torch falls back to *eager* flex attention, which materialises the whole score matrix:
    fine for a few seconds of audio, out of memory for minutes. Skip it, so the library goes
    straight to its chunked-halo SDPA for the sliding-window attention (same maths, linear memory)."""
    try:
        import triton  # noqa: F401
        return
    except ImportError:
        pass
    from stable_audio_3.models import transformer
    transformer.flex_attention_compiled = None
    log(NAME, "no Triton: sliding-window attention runs as chunked SDPA")


def load():
    import torch
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel

    _no_flex_without_triton()
    spec = MODELS_BY_NAME[NAME]
    model_dir = MODELS / spec["dir"]
    cfg = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    for c in cfg["model"]["conditioning"]["configs"]:
        if c.get("type") == "t5gemma":
            c["config"].pop("repo_id", None)
            c["config"].pop("subfolder", None)
            c["config"]["model_path"] = str(T5GEMMA)
    model = load_diffusion_cond(cfg, str(model_dir / "model.safetensors"), device="cuda", model_half=True)
    model.use_lora, model.lora_names = False, []
    decode = model.pretransform.decode
    model.pretransform.decode = lambda *a, **k: decode(*a, **k) * HEADROOM
    sam = StableAudioModel(model, cfg, "cuda", True)
    sr = int(cfg["sample_rate"])
    max_duration = min(spec["max_duration"], int(cfg["sample_size"]) / sr)
    # The first generation pays for CUDA kernel selection and allocator growth; a short one
    # here moves that into startup.
    sam.generate(prompt="a short piano chord", duration=4.0, steps=2, seed=0, batch_size=1,
                 sample_size=int(cfg["sample_size"]))
    torch.cuda.empty_cache()
    return sam, {"engine": NAME, "model": spec["model"], "sample_rate": sr,
                 "channels": int(cfg.get("audio_channels", 2)), "max_duration": max_duration,
                 "default_steps": 8, "features": ["init_audio", "inpaint", "seed"],
                 "ignores": ["negative", "cfg"]}


def _read(path: str):
    import soundfile as sf
    import torch
    a, a_sr = sf.read(path, dtype="float32", always_2d=True)
    return a_sr, torch.from_numpy(a.T.copy())


def handle(sam, req: dict) -> dict:
    import soundfile as sf
    import torch

    cfg = sam.model_config
    sr = int(cfg["sample_rate"])
    duration = float(req["duration"])
    out_dir = Path(req["out_dir"])
    base = dict(prompt=req["prompt"], negative_prompt=req.get("negative") or None, duration=duration,
                steps=int(req.get("steps") or 8), cfg_scale=float(req.get("cfg") or 1.0), batch_size=1,
                sample_size=int(cfg["sample_size"]))
    if req.get("init_audio"):
        base.update(init_audio=_read(req["init_audio"]), init_noise_level=float(req.get("init_strength", 0.5)))
    if req.get("inpaint_audio"):
        base.update(inpaint_audio=_read(req["inpaint_audio"]),
                    inpaint_mask_start_seconds=float(req["mask_start"]),
                    inpaint_mask_end_seconds=float(req["mask_end"]))
    files = []
    for seed in req["seeds"]:
        audio = sam.generate(seed=int(seed) % 2**31, **base) / HEADROOM  # (1, C, T) float32
        path = out_dir / f"{req['stem']}_{seed}.wav"
        sf.write(str(path), audio[0].float().cpu().numpy().T, sr, subtype="FLOAT")
        files.append({"seed": seed, "path": str(path)})
    torch.cuda.empty_cache()
    log(NAME, f"{len(files)} take(s), {duration:.0f}s each: {req['prompt'][:70]}")
    return {"files": files, "sample_rate": sr}


if __name__ == "__main__":
    if NAME not in MODELS_BY_NAME:
        sys.exit(f"unknown model {NAME!r}; choose from {sorted(MODELS_BY_NAME)}")
    serve(NAME, load, handle)
