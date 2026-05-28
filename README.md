# HoliTok

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.8%2C%3C2.9-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Checkpoints](https://img.shields.io/badge/checkpoints-Hugging%20Face-yellow.svg)](https://huggingface.co/bovod-sjtu/HoliTok)

HoliTok is a compact inference runtime for 48 kHz VAE audio tokenization,
reconstruction, and semantic feature extraction.

Public presets:

- `HoliTok-Base`
- `HoliTok-Unite`

The presets contain architecture parameters only. Checkpoints are resolved from
the public checkpoint source by default.

## Install

Use Python 3.10 or newer. Install a CUDA-enabled PyTorch wheel first, then
install HoliTok:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

Runtime dependencies are `torch`, `torchaudio`, `numpy`, `json5`,
`huggingface_hub`, and `soundfile`.

## Python API

```python
import torch
from holitok import HoliTok, SemanticModule

model = HoliTok.from_pretrained("HoliTok-Unite", device="cuda:0")
audio = torch.randn(1, 1, 48000, device="cuda:0")

# [B, 2 * latent_dim, T], concat(mu, log_std)
posterior = model.encode_posterior(audio)

# [B, latent_dim, T]
latents = model.posterior_mean(posterior)

# [B, 1, samples]
recon = model.decode(latents)

semantic = SemanticModule.from_pretrained("HoliTok-Unite", device="cuda:0")
features = semantic(latents.transpose(1, 2))  # [B, T, 1536]
```

`model.reconstruct(audio, sample=True)` follows the latent-stats reconstruction
flow: extract posterior, sample `mu + randn * exp(log_std)`, then decode with
`do_sample=False`.

## CLI

```bash
holitok encode \
  --model HoliTok-Unite \
  --input input.wav \
  --output latents.pt \
  --mode posterior

holitok semantic \
  --model HoliTok-Unite \
  --input-latents latents.pt \
  --output semantic_features.pt \
  --posterior-mode mean

holitok reconstruct \
  --model HoliTok-Unite \
  --input input.wav \
  --output recon.wav \
  --mode sample \
  --seed 1234
```

For custom internal configs, pass `--config <experiment-or-model.json5>` and,
if needed, `--basic-config <base.json>`.

## Scripts

The `scripts/` wrappers use environment variables so they are easy to call from
batch jobs.

Extract VAE latents from audio:

```bash
MODEL=HoliTok-Unite \
INPUT=input.wav \
OUTPUT=latents.pt \
MODE=posterior \
scripts/extract_latent.sh
```

Extract semantic features from a latent file:

```bash
MODEL=HoliTok-Unite \
LATENTS=latents.pt \
OUTPUT=semantic_features.pt \
POSTERIOR_MODE=mean \
scripts/extract_semantic_feature.sh
```

Extract semantic features directly from audio:

```bash
MODEL=HoliTok-Unite \
INPUT=input.wav \
OUTPUT=semantic_features.pt \
MODE=mean \
scripts/extract_semantic_feature.sh
```

Reconstruct audio:

```bash
MODEL=HoliTok-Unite \
INPUT=input.wav \
OUTPUT=recon.wav \
MODE=sample \
scripts/reconstruct.sh
```

Common optional variables for the wrappers:

- `PYTHON=/path/to/python`
- `DEVICE=cuda:0`
- `CHECKPOINT=/path/to/model.pt`
- `SEMANTIC_CHECKPOINT=/path/to/semantic.pt`
- `CHECKPOINT_SOURCE=https://...`
- `CACHE_DIR=/path/to/cache`
- `LOCAL_FILES_ONLY=1`

## Checkpoint Source

Built-in presets download `model.pt` and `semantic.pt` from the configured
checkpoint source. Override it with `HOLITOK_CHECKPOINT_SOURCE`, `--repo-id`, or
`CHECKPOINT_SOURCE` in the shell wrappers.

Checkpoint repository:

`https://huggingface.co/bovod-sjtu/HoliTok`

Local checkpoints are supported with `checkpoint="path/to/model.pt"` in Python,
`--checkpoint path/to/model.pt` in the CLI, or `CHECKPOINT=...` in scripts.

## Outputs

`holitok encode` and `scripts/extract_latent.sh` save a `.pt` dictionary with:

- `latents`: posterior, mean, or sampled latents depending on `MODE`
- `mode`: `posterior`, `mean`, or `sample`
- `sample_rate`, `hop_size`, `latent_dim`

`holitok semantic` and `scripts/extract_semantic_feature.sh` save a `.pt`
dictionary with:

- `features`: semantic features with shape `[B, T, 1536]`
- `model`
- `latent_mode`
- `semantic_dim`
- `source`
- `metadata`

## Layout

- `holitok/model.py`: public `HoliTok` API.
- `holitok/presets.py`: `HoliTok-Base` and `HoliTok-Unite` architecture presets.
- `holitok/semantic.py`: semantic feature encoder.
- `holitok/runtime/`: distilled inference-only model architecture.
- `holitok/cli.py`: encode, semantic feature, decode, and reconstruct commands.
- `scripts/`: shell wrappers for latent extraction, semantic feature extraction,
  and reconstruction.
