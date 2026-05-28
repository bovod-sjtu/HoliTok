from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path, target_sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return audio.unsqueeze(0), sample_rate


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 3:
        audio = audio[0]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio.detach().cpu(), sample_rate)
