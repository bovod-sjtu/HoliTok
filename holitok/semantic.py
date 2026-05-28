from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .checkpoint import clean_state_dict
from .config import JsonHParams
from .hub import resolve_semantic_checkpoint
from .runtime import SuperviseEncoder
from .model import HoliTok, _resolve_device
from .presets import is_preset, normalize_preset_name


def _clean_state_dict(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    return clean_state_dict(str(checkpoint), map_location="cpu")


class SemanticModule(nn.Module):
    """Runtime loader for ``supervise_net.in_proj`` + ``supervise_net.encoder``."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_heads: int = 16,
        causal: bool = True,
    ) -> None:
        super().__init__()
        clean = _clean_state_dict(checkpoint)
        encoder_params = {
            k[len("supervise_net.encoder.") :]: v
            for k, v in clean.items()
            if k.startswith("supervise_net.encoder.")
        }
        in_proj_params = {
            k[len("supervise_net.in_proj.") :]: v
            for k, v in clean.items()
            if k.startswith("supervise_net.in_proj.")
        }
        if not encoder_params:
            raise RuntimeError(f"No supervise_net.encoder params found in {checkpoint}")

        hidden_size = self._infer_hidden_size(encoder_params)
        num_layers = self._infer_num_layers(encoder_params)
        ffn_hidden_size = self._infer_ffn_hidden_size(encoder_params, hidden_size)
        encoder_config = JsonHParams(
            type="transformer",
            hidden_size=hidden_size,
            num_heads=num_heads,
            ffn_hidden_size=ffn_hidden_size,
            num_layers=num_layers,
            causal=causal,
            learn_speaker_emb=False,
        )

        self.encoder = SuperviseEncoder(encoder_config)
        mismatch = self.encoder.load_state_dict(encoder_params, strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            print(f"Semantic encoder load mismatch: {mismatch}")

        self.in_proj = None
        if in_proj_params:
            in_dim = in_proj_params["weight"].shape[1]
            out_dim = in_proj_params["weight"].shape[0]
            self.in_proj = nn.Linear(in_dim, out_dim)
            self.in_proj.load_state_dict(in_proj_params)

        self.device = _resolve_device(device)
        self.to(self.device)
        if dtype is not None:
            self.to(dtype=dtype)
        self.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        checkpoint: str | Path | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_heads: int = 16,
        causal: bool = True,
    ) -> "SemanticModule":
        resolved = resolve_semantic_checkpoint(
            model_name,
            checkpoint,
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if resolved is None:
            raise ValueError(f"semantic checkpoint is required for non-preset model: {model_name}")
        return cls(resolved, device=device, dtype=dtype, num_heads=num_heads, causal=causal)

    @staticmethod
    def _infer_hidden_size(params: dict[str, torch.Tensor]) -> int:
        for key, value in params.items():
            if "o_proj.weight" in key or "out_proj.weight" in key:
                return int(value.shape[0])
        raise RuntimeError("Cannot infer semantic hidden_size from checkpoint")

    @staticmethod
    def _infer_num_layers(params: dict[str, torch.Tensor]) -> int:
        num_layers = 0
        for key in params:
            if key.startswith("layers."):
                num_layers = max(num_layers, int(key.split(".")[1]) + 1)
        if num_layers == 0:
            raise RuntimeError("Cannot infer semantic num_layers from checkpoint")
        return num_layers

    @staticmethod
    def _infer_ffn_hidden_size(params: dict[str, torch.Tensor], hidden_size: int) -> int:
        for key, value in params.items():
            if "fc1.weight" in key or "mlp.fc1.weight" in key:
                return int(value.shape[0])
        return hidden_size * 4

    @torch.no_grad()
    def forward(
        self,
        latents: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *,
        channel_last: bool = True,
    ) -> torch.Tensor:
        if not channel_last:
            latents = latents.transpose(1, 2)
        x = latents.to(self.device)
        if lengths is not None:
            lengths = lengths.to(self.device)
        if self.in_proj is not None:
            x = self.in_proj(x)
        content, _ = self.encoder(x, lengths)
        return content


class UnifiedTokenizer(nn.Module):
    """Compatibility wrapper for VAE latents and semantic features."""

    def __init__(
        self,
        config,
        checkpoint: str | Path | int | None = None,
        *,
        basic_config=None,
        semantic_checkpoint: str | Path | None = None,
        semantic_feature_dim: int | None = None,
        device: str | torch.device | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        sanitize_runtime: bool = True,
        remove_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(checkpoint, int) and semantic_feature_dim is None:
            semantic_feature_dim = checkpoint
            checkpoint = None
        model_name = normalize_preset_name(config) if isinstance(config, str) and is_preset(config) else None
        self.vocoder = HoliTok(
            config,
            checkpoint,
            basic_config=basic_config,
            device=device,
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            sanitize_runtime=sanitize_runtime,
            remove_weight_norm=remove_weight_norm,
        )
        self.hop_size = self.vocoder.hop_size
        self.sample_rate = self.vocoder.sample_rate
        self.vq_downsample_factor = getattr(self.vocoder.model, "vq_downsample_factor", 1)
        self.quantizer = getattr(self.vocoder.model, "quantizer", None)

        self.semantic_module = None
        self.semantic_feature_dim = semantic_feature_dim
        if semantic_feature_dim is not None or semantic_checkpoint is not None:
            semantic_checkpoint = semantic_checkpoint or self.vocoder.model_config.get("semantic_ckpt_path")
            if semantic_checkpoint is None and model_name is not None:
                semantic_checkpoint = resolve_semantic_checkpoint(
                    model_name,
                    None,
                    repo_id=repo_id,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            semantic_checkpoint = semantic_checkpoint or checkpoint or self.vocoder.model_config.get("model_path")
            if semantic_checkpoint is None:
                raise ValueError("semantic_checkpoint is required when semantic loading is enabled")
            self.semantic_module = SemanticModule(semantic_checkpoint, device=self.vocoder.device)

    @torch.no_grad()
    def extract_latents(self, x, **kwargs):
        return self.vocoder.model.extract_latents(x.to(self.vocoder.device), **kwargs)

    @torch.no_grad()
    def extract_vq_latents(self, latents, **kwargs):
        return self.vocoder.extract_vq_latents(latents, **kwargs)

    @torch.no_grad()
    def inference_from_latents(self, x, **kwargs):
        return self.vocoder.decode(x, **kwargs)

    def extract_semantic_features_from_latents(self, latents_sampled):
        if self.semantic_module is None:
            raise RuntimeError("Semantic module was not loaded")
        return self.semantic_module(latents_sampled, channel_last=True)

    def eval(self):
        super().eval()
        self.vocoder.eval()
        if self.semantic_module is not None:
            self.semantic_module.eval()
        return self


UnifiedTokenizerWrapper = UnifiedTokenizer
