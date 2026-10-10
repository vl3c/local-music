"""Model-based scores, run in the server process (on the GPU).

- CLAP (laion/larger_clap_general, trained on music, speech and sounds; the music-only
  larger_clap_music has a broken text tower): how well a track matches its prompt, as the cosine between
  the prompt's and the music's embeddings. CLAP hears 10 s at a time, so a track is cut into
  up to MAX_WINDOWS evenly spaced 10 s windows and their embeddings averaged (its processor
  would otherwise take one random 10 s crop). A *relative* score, for ranking takes of the same
  request: never an absolute pass/fail.
- Audiobox Aesthetics (facebook/audiobox-aesthetics): content enjoyment (CE), content
  usefulness (CU), production complexity (PC) and production quality (PQ), each ~1-10,
  averaged over 10 s windows by the library itself. CE and PQ carry the ranking; PQ is the
  noise / artefact gate.

Every class loads from models/<name> with the Hub offline.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .audio import as2d, resample

# PQ below this is rejected as broken (noise, heavy artefacts); below WARN it's flagged.
# Calibrated on Stable Audio 3 Medium takes: clean orchestral, house, rock, ambient and piano
# scored 7.1-8.5; deliberately lo-fi hip hop (vinyl crackle, dusty samples) 6.2-6.9.
PQ_FAIL = 5.0
PQ_WARN = 6.0
MAX_WINDOWS = 6
WINDOW_S = 10.0


def _features(out):
    """transformers 5 returns the projected, normalised features as `pooler_output` of an
    output object; 4.x returned the tensor itself."""
    return getattr(out, "pooler_output", out)


def windows(x: np.ndarray, sr: int, window_s: float = WINDOW_S, max_windows: int = MAX_WINDOWS) -> list[np.ndarray]:
    """Up to max_windows evenly spaced, non-overlapping-where-possible slices of window_s seconds
    (mono). A track shorter than one window is returned whole."""
    m = as2d(x).mean(axis=0)
    win = int(window_s * sr)
    if len(m) <= win:
        return [m]
    k = min(max_windows, int(np.ceil(len(m) / win)))
    starts = np.linspace(0, len(m) - win, k).astype(int)
    return [m[s:s + win] for s in starts]


class Clap:
    SR = 48000

    def __init__(self, model_dir: Path):
        import torch
        from transformers import ClapModel, ClapProcessor
        t0 = time.time()
        self.torch = torch
        self.model = ClapModel.from_pretrained(str(model_dir)).to("cuda").eval()
        self.proc = ClapProcessor.from_pretrained(str(model_dir))
        print(f"[local-music] CLAP loaded in {time.time() - t0:.1f}s", flush=True)

    def score(self, audios: list[np.ndarray], sr: int, prompt: str) -> list[dict]:
        """For each track: {"clap": cosine between the prompt and the track (~0.1 poor, 0.3 fair,
        0.45+ good)}."""
        torch = self.torch
        with torch.inference_mode():
            ti = self.proc(text=[prompt], return_tensors="pt", padding=True).to("cuda")
            t = _features(self.model.get_text_features(**ti))
            t = t / t.norm(dim=-1, keepdim=True)
            out = []
            for a in audios:
                wins = windows(resample(as2d(a), sr, self.SR), self.SR)
                ai = self.proc(audio=wins, sampling_rate=self.SR, return_tensors="pt").to("cuda")
                e = _features(self.model.get_audio_features(**ai))
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(dim=0, keepdim=True)
                e = e / e.norm(dim=-1, keepdim=True)
                out.append({"clap": round(float((e @ t.T).item()), 4)})
        return out


class Aesthetics:
    def __init__(self, model_dir: Path):
        import torch
        from audiobox_aesthetics.infer import AXES_NAME, AesPredictor, Normalize
        from audiobox_aesthetics.model.aes import AesMultiOutput
        t0 = time.time()
        pred = AesPredictor.__new__(AesPredictor)  # skip its __post_init__, which loads from the Hub
        pred.checkpoint_pth, pred.precision, pred.batch_size = None, "bf16", 1
        pred.data_col, pred.sample_rate, pred.device = "path", 16000, torch.device("cuda")
        model = AesMultiOutput.from_pretrained(str(model_dir)).to(pred.device).eval()
        pred.model = model
        pred.target_transform = {ax: Normalize(mean=model.target_transform[ax]["mean"],
                                               std=model.target_transform[ax]["std"]) for ax in AXES_NAME}
        self.pred, self.torch = pred, torch
        print(f"[local-music] Audiobox Aesthetics loaded in {time.time() - t0:.1f}s", flush=True)

    def score(self, audios: list[np.ndarray], sr: int) -> list[dict]:
        torch = self.torch
        out = []
        for a in audios:  # one at a time: a 6-minute track is 38 windows on its own
            rows = self.pred.forward([{"path": torch.from_numpy(as2d(a).copy()), "sample_rate": sr}])
            out.append({k: round(float(v), 2) for k, v in rows[0].items()})
        return out


def rank_score(take: dict) -> float:
    """One number to sort takes of the same request by. Higher is better.

    Audiobox content enjoyment (CE) and production quality (PQ) carry most of it (each scaled
    4..8.5 -> 0..1), CLAP's prompt match the rest (0.45 earns full marks). Each warning costs a
    little; failed takes sort last regardless.
    """
    if take.get("fails"):
        return -1.0
    aes = take.get("aes") or {}
    scale = lambda v: 0.5 if v is None else min(1.0, max(0.0, (v - 4.0) / 4.5))  # noqa: E731
    clap = take.get("clap")
    c = 0.5 if clap is None else min(1.0, max(0.0, clap / 0.45))
    return round(0.35 * scale(aes.get("CE")) + 0.35 * scale(aes.get("PQ")) + 0.3 * c
                 - 0.04 * len(take.get("warnings", [])), 4)
