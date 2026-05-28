from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch


EXCLUDED_RUNTIME_PREFIXES = (
    "flow.",
    "gq.",
    "quantizer.",
    "supervise_net.",
    "wavlm_",
    "xvector_",
    "pitch_",
    "intensity_",
    "sid_mlp.",
    "self_distill_proj.",
    "vq_ds_proj.",
    "vq_us_proj.",
    "vq_recon_",
    "before_vq_proj.",
    "after_vq_proj.",
    "code_activity",
    "decoder_buckets",
)


def clean_state_dict(checkpoint: str, map_location="cpu") -> OrderedDict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint, map_location=map_location, weights_only=False)
    model_state = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
    clean = OrderedDict()
    for key, value in model_state.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def load_ckpt(model, model_path: str, map_location="cpu", return_iteration: bool = False):
    print(f"Loading model from {model_path}")
    clean_dict = clean_state_dict(model_path, map_location=map_location)
    model_dict = model.state_dict()
    filtered_dict = OrderedDict()
    skipped_keys = []
    for key, value in clean_dict.items():
        if key in model_dict and model_dict[key].shape != value.shape:
            skipped_keys.append(key)
        else:
            filtered_dict[key] = value
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys due to shape mismatch: {skipped_keys[:8]}")
    mismatch = model.load_state_dict(filtered_dict, strict=False)
    expected_unused = [
        key for key in mismatch.unexpected_keys if key.startswith(EXCLUDED_RUNTIME_PREFIXES)
    ]
    unknown_unexpected = [
        key for key in mismatch.unexpected_keys if not key.startswith(EXCLUDED_RUNTIME_PREFIXES)
    ]
    if expected_unused:
        print(f"Ignored {len(expected_unused)} checkpoint keys outside the lean inference graph.")
    if mismatch.missing_keys or unknown_unexpected:
        print(
            "Checkpoint load mismatch: "
            f"{len(mismatch.missing_keys)} missing, {len(unknown_unexpected)} unexpected"
        )
    if not return_iteration:
        return model
    try:
        iteration = int(os.path.basename(model_path).split("-")[-1].split(".")[0])
    except Exception:
        iteration = 0
    return model, iteration


def num_params(net) -> float:
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    return sum(np.prod(p.size()) for p in parameters) / 1024 / 1024
