from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import json5

from .presets import get_preset_config, is_preset


RUNTIME_DISABLED_FLAGS = {
    "distill_wavlm": False,
    "distill_pitch": False,
    "distill_intensity": False,
    "distill_xvector": False,
    "sid_classifier": False,
    "use_supervise": False,
    "add_fm_noise": False,
}


class JsonHParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = JsonHParams(**value)
            if isinstance(value, str) and value.lower() in ["non", "none", "nil", "null"]:
                value = None
            self[key] = value

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, JsonHParams) else value
        return result

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)

    def pop(self, key):
        return self.__dict__.pop(key)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def set(self, key, value):
        return setattr(self, key, value)

    def exist(self, key):
        return hasattr(self, key)


def override_config(basic_config, new_config):
    merged = deepcopy(dict(basic_config))
    for key, value in new_config.items():
        if isinstance(value, dict):
            merged[key] = override_config(merged.get(key, {}), value)
        else:
            merged[key] = value
    return merged


def load_json5(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json5.load(f)


def _as_dict(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return deepcopy(config.to_dict())
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    if hasattr(config, "items"):
        return deepcopy(dict(config.items()))
    return deepcopy(dict(config))


def to_hparams(config: Mapping[str, Any] | Any) -> JsonHParams:
    return JsonHParams(**_as_dict(config))


def sanitize_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Disable training-only branches that do not affect VAE encode/decode."""

    model_config = deepcopy(dict(config))
    model_config.update(RUNTIME_DISABLED_FLAGS)
    return model_config


def load_model_config(
    config: str | Path | Mapping[str, Any] | Any,
    *,
    basic_config: str | Path | Mapping[str, Any] | Any | None = None,
    sanitize_runtime: bool = True,
) -> dict[str, Any]:
    """Load and optionally merge Audiokit JSON5 configs.

    ``config`` may be either a full experiment config with a top-level
    ``model`` key, a model-only config, or a built-in preset name such as
    ``HoliTok-Base``. When ``basic_config`` is supplied, it is merged with
    ``config`` using Audiokit's recursive override rules.
    """

    if isinstance(config, str) and is_preset(config):
        if basic_config is not None:
            raise ValueError("basic_config cannot be used with built-in HoliTok presets")
        exp = get_preset_config(config)
    elif isinstance(config, (str, Path)):
        exp = load_json5(config)
    else:
        exp = _as_dict(config)

    if basic_config is not None:
        if isinstance(basic_config, (str, Path)):
            base = load_json5(basic_config)
        else:
            base = _as_dict(basic_config)
        merged = override_config(base, exp)
    else:
        merged = exp

    model_config = merged["model"] if "model" in merged else merged
    model_config = deepcopy(dict(model_config))
    if sanitize_runtime:
        model_config = sanitize_model_config(model_config)
    return model_config
