#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT:?set OUTPUT=/path/to/semantic_features.pt}"

MODEL="${MODEL:-HoliTok-Unite}"
CONFIG="${CONFIG:-}"
BASIC_CONFIG="${BASIC_CONFIG:-}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-${REPO_ID:-}}"
SEMANTIC_CHECKPOINT="${SEMANTIC_CHECKPOINT:-}"
CACHE_DIR="${CACHE_DIR:-}"
INPUT="${INPUT:-}"
LATENTS="${LATENTS:-}"
MODE="${MODE:-sample}"
POSTERIOR_MODE="${POSTERIOR_MODE:-sample}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-}"
CHANNEL_LAST="${CHANNEL_LAST:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-}"
PYTHON="${PYTHON:-python}"

if [[ -z "$INPUT" && -z "$LATENTS" ]]; then
  echo "set either INPUT=/path/to/input.wav or LATENTS=/path/to/latents.pt" >&2
  exit 2
fi

if [[ -n "$INPUT" && -n "$LATENTS" ]]; then
  echo "set only one of INPUT or LATENTS" >&2
  exit 2
fi

args=(
  semantic
  --model "$MODEL"
  --output "$OUTPUT"
  --mode "$MODE"
  --posterior-mode "$POSTERIOR_MODE"
  --noise-scale "$NOISE_SCALE"
)

if [[ -n "$INPUT" ]]; then
  args+=(--input-audio "$INPUT")
fi

if [[ -n "$LATENTS" ]]; then
  args+=(--input-latents "$LATENTS")
fi

if [[ -n "$CONFIG" ]]; then
  args+=(--config "$CONFIG")
fi

if [[ -n "$BASIC_CONFIG" ]]; then
  args+=(--basic-config "$BASIC_CONFIG")
fi

if [[ -n "$CHECKPOINT" ]]; then
  args+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "$CHECKPOINT_SOURCE" ]]; then
  args+=(--repo-id "$CHECKPOINT_SOURCE")
fi

if [[ -n "$SEMANTIC_CHECKPOINT" ]]; then
  args+=(--semantic-checkpoint "$SEMANTIC_CHECKPOINT")
fi

if [[ -n "$CACHE_DIR" ]]; then
  args+=(--cache-dir "$CACHE_DIR")
fi

if [[ -n "$DEVICE" ]]; then
  args+=(--device "$DEVICE")
fi

if [[ -n "$SEED" ]]; then
  args+=(--seed "$SEED")
fi

if [[ "$CHANNEL_LAST" == "1" || "$CHANNEL_LAST" == "true" ]]; then
  args+=(--channel-last)
fi

if [[ "$LOCAL_FILES_ONLY" == "1" || "$LOCAL_FILES_ONLY" == "true" ]]; then
  args+=(--local-files-only)
fi

"$PYTHON" -m holitok.cli "${args[@]}"
