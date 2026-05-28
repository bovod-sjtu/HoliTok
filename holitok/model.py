from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .checkpoint import load_ckpt
from .config import load_model_config, to_hparams
from .hub import resolve_checkpoint
from .presets import is_preset, normalize_preset_name
from .runtime import BigVGANVAEVQ, BigVGANVAEVQDS


MODEL_TYPES = {
    "BigVGANVAEVQ": BigVGANVAEVQ,
    "BigVGANVAEVQDS": BigVGANVAEVQDS,
}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_audio_tensor(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(1)
    elif audio.dim() != 3:
        raise ValueError(f"audio must have shape [T], [B,T], or [B,1,T], got {tuple(audio.shape)}")
    if audio.size(1) != 1:
        raise ValueError(f"audio must be mono with channel dimension 1, got {tuple(audio.shape)}")
    return audio


def _model_class(model_type: str):
    if model_type in MODEL_TYPES:
        return MODEL_TYPES[model_type]
    raise ValueError(f"Unsupported model type: {model_type}. Supported: {sorted(MODEL_TYPES)}")


class HoliTok(nn.Module):
    """Inference wrapper for HoliTok checkpoints."""

    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        checkpoint: str | Path | None = None,
        *,
        basic_config: str | Path | Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        sanitize_runtime: bool = True,
        remove_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        model_name = normalize_preset_name(config) if isinstance(config, str) and is_preset(config) else None
        model_config = load_model_config(
            config,
            basic_config=basic_config,
            sanitize_runtime=sanitize_runtime,
        )
        if checkpoint is None:
            checkpoint = model_config.get("model_path")
        checkpoint = resolve_checkpoint(
            model_name,
            checkpoint,
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if checkpoint is None:
            raise ValueError("checkpoint is required unless config.model_path is set")

        self.model_config = model_config
        self.hparams = to_hparams(model_config)
        self.device = _resolve_device(device)

        cls = _model_class(self.hparams.type)
        model = cls(self.hparams)
        model = load_ckpt(model, str(checkpoint), map_location="cpu")
        if remove_weight_norm:
            model.remove_weight_norm()
        model.eval()
        model.to(self.device)
        if dtype is not None:
            model.to(dtype=dtype)
        self.model = model

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    @property
    def hop_size(self) -> int:
        return int(self.model.hop_size)

    @property
    def encoder_hop_size(self) -> int:
        return int(getattr(self.model, "encoder_hop_size", self.hop_size))

    @property
    def latent_dim(self) -> int:
        return int(self.hparams.latent_dim)

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "HoliTok":
        return cls(*args, **kwargs)

    def encode_posterior(
        self,
        audio: torch.Tensor,
        *,
        deg_sr: int = 16000,
    ) -> torch.Tensor:
        """Return channel-first VAE posterior ``[B, 2*D, T]``."""

        audio = _as_audio_tensor(audio).to(self.device)
        with torch.no_grad():
            return self.model.extract_latents(audio, deg_sr=deg_sr, do_sample=False)

    def encode(
        self,
        audio: torch.Tensor,
        *,
        sample: bool = False,
        deg_sr: int = 16000,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Return ``[B, D, T]`` latents, using posterior mean unless sampled."""

        posterior = self.encode_posterior(audio, deg_sr=deg_sr)
        if sample:
            return self.sample_posterior(posterior, noise_scale=noise_scale)
        return self.posterior_mean(posterior)

    def decode(
        self,
        latents: torch.Tensor,
        *,
        do_sample: bool = False,
        noise_scale: float = 1.0,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """Decode channel-first ``[B, D, T]`` latents to ``[B, 1, samples]``."""

        if channel_last:
            latents = latents.transpose(1, 2)
        latents = latents.to(self.device)
        with torch.no_grad():
            return self.model.inference_from_latents(
                latents,
                do_sample=do_sample,
                noise_scale=noise_scale,
            )

    def reconstruct(
        self,
        audio: torch.Tensor,
        *,
        sample: bool = True,
        deg_sr: int = 16000,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Encode then decode audio using the same latent flow as Audiokit stats."""

        posterior = self.encode_posterior(audio, deg_sr=deg_sr)
        if sample:
            latents = self.sample_posterior(posterior, noise_scale=noise_scale)
        else:
            latents = self.posterior_mean(posterior)
        return self.decode(latents, do_sample=False)

    def extract_vq_latents(self, latents: torch.Tensor, *args, **kwargs):
        latents = latents.to(self.device)
        with torch.no_grad():
            return self.model.extract_vq_latents(latents, *args, **kwargs)

    def inference_from_vq_latents(self, vq_latents: torch.Tensor) -> torch.Tensor:
        vq_latents = vq_latents.to(self.device)
        with torch.no_grad():
            return self.model.inference_from_vq_latents(vq_latents)

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.reconstruct(audio, **kwargs)

    def remove_weight_norm(self) -> None:
        self.model.remove_weight_norm()

    @staticmethod
    def posterior_mean(posterior: torch.Tensor) -> torch.Tensor:
        mean, _ = posterior.chunk(2, dim=1)
        return mean

    @staticmethod
    def sample_posterior(posterior: torch.Tensor, *, noise_scale: float = 1.0) -> torch.Tensor:
        mean, log_std = posterior.chunk(2, dim=1)
        return mean + torch.randn_like(mean) * torch.exp(log_std) * noise_scale
