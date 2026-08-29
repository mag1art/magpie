#!/bin/bash
# Entrypoint for the Magpie + OpenAI-compatible API container.
# Model selection and API endpoint are set via environment variables (see docker-compose.yml).
set -e

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
API_URL="${API_URL:-http://host.docker.internal:8000}"
API_KEY="${API_KEY:-}"
TOTAL_PROMPTS="${TOTAL_PROMPTS:-1000}"
INS_TOPP="${INS_TOPP:-1}"
INS_TEMP="${INS_TEMP:-0.7}"
RES_TOPP="${RES_TOPP:-1}"
RES_TEMP="${RES_TEMP:-0}"
N="${N:-200}"
BATCH_SIZE="${BATCH_SIZE:-200}"

echo "[entrypoint] MODEL_NAME=$MODEL_NAME"
echo "[entrypoint] API_URL=$API_URL"
echo "[entrypoint] TOTAL_PROMPTS=$TOTAL_PROMPTS"
echo "[entrypoint] N=$N BATCH_SIZE=$BATCH_SIZE"

timestamp=$(date +%s)
job_name="${MODEL_NAME##*/}_topp${INS_TOPP}_temp${INS_TEMP}_${timestamp}"
job_path="../data/${job_name}"
mkdir -p "$job_path"

cd /app/magpie/scripts

echo "[entrypoint] Generating instructions via ${API_URL}/v1/completions ..."
python ../exp/gen_ins.py \
    --model_name "$MODEL_NAME" \
    --api_url "${API_URL}/v1/completions" \
    --api_key "$API_KEY" \
    --total_prompts "$TOTAL_PROMPTS" \
    --top_p "$INS_TOPP" \
    --temperature "$INS_TEMP" \
    --n "$N" \
    --job_name "$job_name" \
    --timestamp "$timestamp"

echo "[entrypoint] Generating responses via ${API_URL}/v1/chat/completions ..."
python ../exp/gen_res.py \
    --model_name "$MODEL_NAME" \
    --api_url "${API_URL}/v1/chat/completions" \
    --api_key "$API_KEY" \
    --batch_size "$BATCH_SIZE" \
    --top_p "$RES_TOPP" \
    --temperature "$RES_TEMP" \
    --repetition_penalty 1 \
    --input_file "${job_path}/Magpie_${MODEL_NAME##*/}_${TOTAL_PROMPTS}_${timestamp}_ins.json"

echo "[entrypoint] Done. Data written to ${job_path}"
