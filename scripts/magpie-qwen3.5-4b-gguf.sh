#!/bin/bash
# Magpie for Qwen3.5-4B using llama.cpp + GGUF (Q4_K_M)
#
# Usage:
#   ./magpie-qwen3.5-4b-gguf.sh [model_name] [gguf_repo] [gguf_file] [total_prompts] [ins_topp] [ins_temp] [res_topp] [res_temp] [n_gpu_layers]
#
# Defaults:
#   model_name  = Qwen/Qwen3.5-4B   (HF id, used for config lookup / reasoning detection)
#   gguf_repo   = unsloth/Qwen3.5-4B-GGUF
#   gguf_file   = Qwen3.5-4B-Q4_K_M.gguf  (~2.7 GB)
#
# Requires: llama-cpp-python (with CUDA build for GPU offload), transformers, torch

model_name=${1:-"Qwen/Qwen3.5-4B"}
gguf_repo=${2:-"unsloth/Qwen3.5-4B-GGUF"}
gguf_file=${3:-"Qwen3.5-4B-Q4_K_M.gguf"}
total_prompts=${4:-1000}
ins_topp=${5:-1}
ins_temp=${6:-0.7}
res_topp=${7:-1}
res_temp=${8:-0}
res_rep=1
n_gpu_layers=${9:--1}
n=200
batch_size=200

# Local GGUF path
model_dir="models"
model_path="${model_dir}/${gguf_file}"

# Download GGUF if not present
if [ ! -f "$model_path" ]; then
    echo "[magpie-gguf] Downloading ${gguf_file} from ${gguf_repo} ..."
    mkdir -p "$model_dir"
    python - <<PYEOF
import urllib.request, os
url = "https://huggingface.co/${gguf_repo}/resolve/main/${gguf_file}"
dest = "${model_path}"
print(f"[magpie-gguf] {url} -> {dest}")
urllib.request.urlretrieve(url, dest)
print("[magpie-gguf] Download complete.")
PYEOF
else
    echo "[magpie-gguf] GGUF already present: ${model_path}"
fi

# Get Current Time
timestamp=$(date +%s)

# Generate Pretty Name
job_name="${model_name##*/}_topp${ins_topp}_temp${ins_temp}_${timestamp}"

### Setup Logging
log_dir="data"
if [ ! -d "../${log_dir}" ]; then
    mkdir -p "../${log_dir}"
fi

job_path="../${log_dir}/${job_name}"

mkdir -p $job_path
exec > >(tee -a "$job_path/${job_name}.log") 2>&1
echo "[magpie-gguf] Model Name: $model_name"
echo "[magpie-gguf] GGUF: $model_path"
echo "[magpie-gguf] Pretty name: $job_name"
echo "[magpie-gguf] Total Prompts: $total_prompts"
echo "[magpie-gguf] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp"
echo "[magpie-gguf] Response Generation Config: temp=$res_temp, top_p=$res_topp, rep=$res_rep"
echo "[magpie-gguf] System Config: n=$n, batch_size=$batch_size, n_gpu_layers=$n_gpu_layers"
echo "[magpie-gguf] Timestamp: $timestamp"
echo "[magpie-gguf] Job Name: $job_name"

echo "[magpie-gguf] Start Generating Instructions..."
python ../exp/gen_ins.py \
    --engine llamacpp \
    --model_path $model_path \
    --model_name $model_name \
    --total_prompts $total_prompts \
    --top_p $ins_topp \
    --temperature $ins_temp \
    --n_gpu_layers $n_gpu_layers \
    --n $n \
    --job_name $job_name \
    --timestamp $timestamp

echo "[magpie-gguf] Finish Generating Instructions!"

echo "[magpie-gguf] Start Generating Responses..."
python ../exp/gen_res.py \
    --engine llamacpp \
    --model_path $model_path \
    --model_name $model_name \
    --batch_size $batch_size \
    --top_p $res_topp \
    --temperature $res_temp \
    --repetition_penalty $res_rep \
    --n_gpu_layers $n_gpu_layers \
    --input_file $job_path/Magpie_${model_name##*/}_${total_prompts}_${timestamp}_ins.json \
    --use_tokenizer_template

echo "[magpie-gguf] Finish Generating Responses!"
