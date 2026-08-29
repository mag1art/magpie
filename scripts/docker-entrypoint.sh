#!/bin/bash
# Entrypoint for the Magpie + llama.cpp container.
# Model selection is done via environment variables (see docker-compose.yml).
set -e

# Model selection
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3.5-4B-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen3.5-4B-Q4_K_M.gguf}"
TOTAL_PROMPTS="${TOTAL_PROMPTS:-1000}"
INS_TOPP="${INS_TOPP:-1}"
INS_TEMP="${INS_TEMP:-0.7}"
RES_TOPP="${RES_TOPP:-1}"
RES_TEMP="${RES_TEMP:-0}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

echo "[entrypoint] MODEL_NAME=$MODEL_NAME"
echo "[entrypoint] GGUF_REPO=$GGUF_REPO"
echo "[entrypoint] GGUF_FILE=$GGUF_FILE"
echo "[entrypoint] TOTAL_PROMPTS=$TOTAL_PROMPTS"
echo "[entrypoint] N_GPU_LAYERS=$N_GPU_LAYERS"

cd /app/magpie/scripts

exec ./magpie-qwen3.6-35b-a3b-gguf.sh \
    "$MODEL_NAME" "$GGUF_REPO" "$GGUF_FILE" \
    "$TOTAL_PROMPTS" "$INS_TOPP" "$INS_TEMP" "$RES_TOPP" "$RES_TEMP" \
    "$N_GPU_LAYERS"
