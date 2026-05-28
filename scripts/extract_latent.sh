#!/usr/bin/env bash
set -euo pipefail

: "${INPUT:?set INPUT=/path/to/input.wav}"
: "${OUTPUT:?set OUTPUT=/path/to/latents.pt}"

MODEL="${MODEL:-HoliTok-Unite}"
CONFIG="${CONFIG:-}"
BASIC_CONFIG="${BASIC_CONFIG:-}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-${REPO_ID:-}}"
CACHE_DIR="${CACHE_DIR:-}"
MODE="${MODE:-posterior}"
DEVICE="${DEVICE:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-}"
PYTHON="${PYTHON:-python}"

args=(
  encode
  --model "$MODEL"
  --input "$INPUT"
  --output "$OUTPUT"
  --mode "$MODE"
)

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

if [[ -n "$CACHE_DIR" ]]; then
  args+=(--cache-dir "$CACHE_DIR")
fi

if [[ -n "$DEVICE" ]]; then
  args+=(--device "$DEVICE")
fi

if [[ "$LOCAL_FILES_ONLY" == "1" || "$LOCAL_FILES_ONLY" == "true" ]]; then
  args+=(--local-files-only)
fi

"$PYTHON" -m holitok.cli "${args[@]}"
