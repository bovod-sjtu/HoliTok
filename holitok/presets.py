from __future__ import annotations

from copy import deepcopy


_HOLITOK_48K_CONFIG = {
    "type": "BigVGANVAEVQDS",
    "sample_rate": 48000,
    "upsample_rates": [10, 6, 4, 2, 2, 2],
    "upsample_kernel_sizes": [20, 12, 8, 4, 4, 4],
    "upsample_initial_channel": 1536,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "downsample_rates": [2, 2, 2, 4, 6, 10],
    "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
    "activation": "snakebeta",
    "snake_logscale": True,
    "latent_dim": 128,
    "use_vae": True,
    "use_flow": True,
    "flow_hidden_channels": 256,
    "causal": True,
    "causal_encoder": True,
    "num_encoder_lookahead": 2,
    "num_decoder_lookahead": 2,
    "fixed_filter": True,
    "use_bias_at_final": False,
    "use_tanh_at_final": False,
    "use_vq": False,
    "quant_after_supervise_encoder": False,
    "quant_mean": True,
    "vq_downsample_factor": 1,
    "use_invertible_downsample": False,
    "mi_num_layers": 4,
}


PRESETS = {
    "HoliTok-Base": _HOLITOK_48K_CONFIG,
    "HoliTok-Unite": _HOLITOK_48K_CONFIG,
}

CHECKPOINT_FILENAMES = {
    "HoliTok-Base": "HoliTok-Base/model.pt",
    "HoliTok-Unite": "HoliTok-Unite/model.pt",
}

CHECKPOINT_SHA256 = {
    "HoliTok-Base": "bf69dd89c2fbcd511602377c9ac8bf559179667269f9820811a9c17cdc8d53a0",
    "HoliTok-Unite": "93a7cb4cd39ad99e50ecbb03f48b97dc81da966e4ffcd76f8ce988ddf04c2dcb",
}

SEMANTIC_CHECKPOINT_FILENAMES = {
    "HoliTok-Base": "HoliTok-Base/semantic.pt",
    "HoliTok-Unite": "HoliTok-Unite/semantic.pt",
}

SEMANTIC_CHECKPOINT_SHA256 = {
    "HoliTok-Base": "ffbc2ddadbcf057da82267ea61e3fb6562877b79658d4f59149cd939358a8b91",
    "HoliTok-Unite": "0db70c5f827aa847dbcc9799816600d46fe868a2e3119407236ea163eb3d9058",
}

_ALIASES = {
    "base": "HoliTok-Base",
    "holitok-base": "HoliTok-Base",
    "unite": "HoliTok-Unite",
    "holitok-unite": "HoliTok-Unite",
}


def preset_names() -> tuple[str, ...]:
    return tuple(PRESETS)


def normalize_preset_name(name: str) -> str:
    key = name.strip()
    return _ALIASES.get(key.lower(), key)


def is_preset(name: str) -> bool:
    return normalize_preset_name(name) in PRESETS


def get_preset_config(name: str) -> dict:
    preset_name = normalize_preset_name(name)
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown HoliTok preset: {name}. Available: {', '.join(preset_names())}")
    return deepcopy(PRESETS[preset_name])


def checkpoint_filename(name: str) -> str:
    preset_name = normalize_preset_name(name)
    if preset_name not in CHECKPOINT_FILENAMES:
        raise ValueError(f"Unknown HoliTok preset: {name}. Available: {', '.join(preset_names())}")
    return CHECKPOINT_FILENAMES[preset_name]


def checkpoint_sha256(name: str) -> str:
    preset_name = normalize_preset_name(name)
    if preset_name not in CHECKPOINT_SHA256:
        raise ValueError(f"Unknown HoliTok preset: {name}. Available: {', '.join(preset_names())}")
    return CHECKPOINT_SHA256[preset_name]


def semantic_checkpoint_filename(name: str) -> str:
    preset_name = normalize_preset_name(name)
    if preset_name not in SEMANTIC_CHECKPOINT_FILENAMES:
        raise ValueError(f"Unknown HoliTok preset: {name}. Available: {', '.join(preset_names())}")
    return SEMANTIC_CHECKPOINT_FILENAMES[preset_name]


def semantic_checkpoint_sha256(name: str) -> str:
    preset_name = normalize_preset_name(name)
    if preset_name not in SEMANTIC_CHECKPOINT_SHA256:
        raise ValueError(f"Unknown HoliTok preset: {name}. Available: {', '.join(preset_names())}")
    return SEMANTIC_CHECKPOINT_SHA256[preset_name]
