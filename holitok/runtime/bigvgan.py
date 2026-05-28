from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .layers import Conv1d, Decoder, Encoder, SLSTM, high_order_resample_torch


class _RuntimeBigVGANVAE(nn.Module):
    """Minimal Audiokit BigVGAN VAE graph needed for latent encode/decode."""

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.encoder_hop_size = int(np.prod(h.downsample_rates))
        self.hop_size = int(np.prod(h.upsample_rates))
        self.sample_rate = h.sample_rate
        self.encoder_sample_rate = h.get("encoder_sample_rate", None)
        if self.encoder_sample_rate is not None and self.encoder_sample_rate != self.sample_rate:
            self.encoder_resample = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate,
                new_freq=self.encoder_sample_rate,
            )
        else:
            self.encoder_resample = None

        self.quantizer_type = "VAEVQ"
        self.codebook_size = h.get("codebook_size", 16384)
        self.vq_downsample_factor = h.get("vq_downsample_factor", 1)
        self.use_invertible_downsample = h.get("use_invertible_downsample", False)
        self.quantizer = None

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim,
            down_sample_factors=h.downsample_rates,
            channels=h.downsample_channels,
            causal=h.get("causal_encoder", False),
            lookahead=h.get("num_encoder_lookahead", 0),
        )

        if h.get("use_vae", False):
            self.mi_num_layers = h.get("mi_num_layers", 2)
            self.mi_skip = h.get("mi_skip", True)
            intermediate_size = h.latent_dim * 4
            self.enc_mi_layer = nn.Sequential(
                nn.Linear(h.latent_dim, intermediate_size),
                SLSTM(intermediate_size, num_layers=self.mi_num_layers, skip=self.mi_skip),
                nn.Linear(intermediate_size, h.latent_dim),
            )
            self.dec_mi_layer = nn.Sequential(
                nn.Linear(h.latent_dim, intermediate_size),
                SLSTM(intermediate_size, num_layers=self.mi_num_layers, skip=self.mi_skip),
                nn.Linear(intermediate_size, h.latent_dim),
            )
            self.pre_proj = Conv1d(
                in_channels=h.latent_dim,
                out_channels=h.latent_dim * 2,
                kernel_size=1,
                stride=1,
            )
            self.post_proj = Conv1d(
                in_channels=h.latent_dim,
                out_channels=h.latent_dim,
                kernel_size=1,
                stride=1,
            )

        self.decoder = Decoder(h)

    def _preprocess_audio(self, x: torch.Tensor, deg_sr: int) -> torch.Tensor:
        x = x.float()
        if self.h.get("dynamic_sr", False):
            with torch.autocast(enabled=False, device_type=x.device.type):
                x = high_order_resample_torch(
                    x,
                    orig_sr=self.h.dynamic_sr_orig_sr,
                    target_sr=deg_sr,
                )
                x = high_order_resample_torch(
                    x,
                    orig_sr=deg_sr,
                    target_sr=self.h.dynamic_sr_orig_sr,
                )
        if self.encoder_resample is not None:
            with torch.autocast(enabled=False, device_type=x.device.type):
                x = self.encoder_resample(x)
        return x

    def _posterior_from_audio(self, x: torch.Tensor, deg_sr: int) -> torch.Tensor:
        x = self._preprocess_audio(x, deg_sr)
        x = self.audio_encoder(x)
        if self.h.get("use_vae", False):
            x = x.permute(0, 2, 1)
            x = self.enc_mi_layer(x)
            x = x.permute(0, 2, 1)
            x = self.pre_proj(x)
        return x

    def extract_latents(self, x: torch.Tensor, deg_sr: int = 16000, do_sample: bool = False) -> torch.Tensor:
        x = self._posterior_from_audio(x, deg_sr)
        if self.h.get("use_vae", False) and do_sample:
            mean, log_std = torch.split(x, self.h.latent_dim, dim=1)
            x = mean + torch.randn_like(mean) * torch.exp(log_std)
        return x

    def _decode_latents(self, x: torch.Tensor) -> torch.Tensor:
        if self.h.get("use_vae", False):
            x = self.post_proj(x)
            x = x.permute(0, 2, 1)
            x = self.dec_mi_layer(x)
            x = x.permute(0, 2, 1)
        return self.decoder(x)

    def inference_from_latents(
        self,
        x: torch.Tensor,
        do_sample: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        if self.h.get("use_vae", False) and do_sample:
            assert x.size(1) == self.h.latent_dim * 2, f"Input must be like [B, D, H], got {x.shape}"
            mean, log_std = torch.split(x, self.h.latent_dim, dim=1)
            x = mean + torch.randn_like(mean) * torch.exp(log_std) * noise_scale
        else:
            assert x.size(1) == self.h.latent_dim, f"Input must be like [B, D, H], got {x.shape}"
        return self._decode_latents(x)

    def extract_vq_latents(self, latents, *args, **kwargs):
        raise NotImplementedError("VQ token extraction is not part of the lean HoliTok inference runtime.")

    def inference_from_vq_latents(self, vq_latents, *args, **kwargs):
        raise NotImplementedError("VQ token decoding is not part of the lean HoliTok inference runtime.")

    def remove_weight_norm(self) -> None:
        self.decoder.remove_weight_norm()


class BigVGANVAEVQ(_RuntimeBigVGANVAE):
    """Lean runtime equivalent of Audiokit ``BigVGANVAEVQ`` encode/decode paths."""

    def __init__(self, h):
        self.direct_vq = h.get("direct_vq", True)
        self.vq_downsample_mode = h.get("vq_downsample_mode", "conv")
        super().__init__(h)


class BigVGANVAEVQDS(_RuntimeBigVGANVAE):
    """Lean runtime equivalent of Audiokit ``BigVGANVAEVQDS`` encode/decode paths."""

    def __init__(self, h):
        self.vq_downstream_tasks = h.get("vq_downstream_tasks", [])
        self.quant_mean = h.get("quant_mean", False)
        self.quant_after_supervise_encoder = h.get("quant_after_supervise_encoder", False)
        self.use_interpolation = h.get("use_interpolation", False)
        self.add_fm_noise = h.get("add_fm_noise", False)
        self.min_var = h.get("min_var", 0.0)
        self.vae_type = h.get("vae_type", "flow")
        super().__init__(h)

    def extract_latents(self, x: torch.Tensor, deg_sr: int = 16000, do_sample: bool = False) -> torch.Tensor:
        x = self._posterior_from_audio(x, deg_sr)
        if self.vae_type == "gq":
            mu, logvar = torch.split(x, self.h.latent_dim, dim=1)
            x = torch.cat([mu, logvar / 2], dim=1)
        if self.h.get("use_vae", False) and do_sample:
            mean, log_std = torch.split(x, self.h.latent_dim, dim=1)
            x = mean + torch.randn_like(mean) * torch.exp(log_std)
        return x

    def inference_from_latents(
        self,
        x: torch.Tensor,
        do_sample: bool = True,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        return super().inference_from_latents(x, do_sample=do_sample, noise_scale=noise_scale)
