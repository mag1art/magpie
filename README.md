# 🐦 Magpie — OpenAI-compatible API generation

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2406.08464) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a fork of [Magpie](https://github.com/magpie-align/magpie) (ICLR 2025, *"Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing"*) that generates synthetic alignment data **exclusively through an OpenAI-compatible API**.

Magpie generates high-quality alignment data by prompting aligned LLMs with their pre-query templates. It uses the prompt template of an aligned LLM to generate both the user query and an LLM response — no prompt engineering or seed questions required.

This fork **does not run any model locally**. It calls an external OpenAI-compatible server (e.g. a [vLLM](https://docs.vllm.ai/) or [llama.cpp](https://github.com/ggml-org/llama.cpp) server) that can aggregate GPUs across one or more machines. The container needs no CUDA, no GPU, and no model weights.

## Features

- **API-only generation** — no vLLM / llama.cpp / torch / transformers runtime.
- **Two-stage pipeline**:
  - `gen_ins.py` — generates instructions via `/v1/completions` (raw prompt continuation).
  - `gen_res.py` — generates responses via `/v1/chat/completions`.
- **Reasoning-model support** — strips ` thinking...response` markers for Qwen3-style models (`extract_final_answer`).
- **Quality filters** — drops truncated/too-short instructions; saves JSON with readable Unicode (`ensure_ascii=False`).
- **Docker** — lightweight image (no GPU) that connects to your API server.

## Installation

```bash
git clone <your-fork-url> magpie
cd magpie
pip install -r requirements.txt
```

Only two dependencies are required: `requests` and `tqdm`.

## Quick Start

Point the scripts at any OpenAI-compatible server that exposes `/v1/completions` and `/v1/chat/completions`:

```bash
# 1. Generate instructions
python exp/gen_ins.py \
    --model_name deepseek-v4-flash:0731-cloud \
    --api_url http://localhost:8000/v1/completions \
    --api_key "" \
    --total_prompts 1000 \
    --top_p 1 \
    --temperature 0.7 \
    --n 200

# 2. Generate responses for the instructions
python exp/gen_res.py \
    --model_name deepseek-v4-flash:0731-cloud \
    --api_url http://localhost:8000/v1/chat/completions \
    --api_key "" \
    --batch_size 200 \
    --top_p 1 \
    --temperature 0 \
    --input_file data/Magpie_deepseek-v4-flash:0731-cloud_1000_<timestamp>_ins.json
```

Output is written to `data/`:
- `Magpie_<model>_<n>_<timestamp>_ins.json` — generated instructions.
- `Magpie_<model>_<n>_<timestamp>_ins_res.json` — instructions + responses.

## Docker

The container does not run a model — it only calls your API server.

```bash
# Build and run with defaults
API_URL=http://<server>:<port> docker compose up --build

# Or run interactively
docker compose run --rm magpie-api bash
```

Configuration is done via environment variables (see `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://host.docker.internal:8000` | Base URL of the OpenAI-compatible server |
| `API_KEY` | *(empty)* | API key (optional for local servers) |
| `MODEL_NAME` | `deepseek-v4-flash:0731-cloud` | Model id (config lookup + sent to the API) |
| `TOTAL_PROMPTS` | `1000` | Number of instructions to generate |
| `INS_TOPP` / `INS_TEMP` | `1` / `0.7` | Instruction sampling params |
| `RES_TOPP` / `RES_TEMP` | `1` / `0` | Response sampling params |
| `N` / `BATCH_SIZE` | `200` | Samples per request / batch size |

Generated data is written to `./data` on the host.

## Model Configuration

This fork ships with configs for three Qwen3 reasoning models (all use the same Qwen3 chat template) plus a DeepSeek cloud model:

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.8-27B`
- `deepseek-v4-flash:0731-cloud` (default)

Each model needs an entry in [`configs/model_configs.json`](configs/model_configs.json) with its chat template and stop tokens. Example for Qwen3.5-4B:

```json
"Qwen/Qwen3.5-4B": {
  "model_name": "Qwen/Qwen3.5-4B",
  "stop_tokens": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
  "stop_token_ids": [248045, 248046, 248044],
  "stop_tokens_assistant": ["Assistant", "assistant"],
  "pre_query_template": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
}
```

To add a new model, add its config and pass its id via `--model_name`.

## Reasoning Models (Qwen3)

Qwen3-style reasoning models emit a ` thinking...response` block before the final answer. The scripts automatically strip this via `extract_final_answer` (in `exp/str_utils.py`), returning only the final answer. Detection is based on `"qwen3"` appearing in the model id.

## Project Structure

```
exp/
  gen_ins.py        # instruction generation via /v1/completions
  gen_res.py        # response generation via /v1/chat/completions
  str_utils.py      # reasoning-marker stripping, filters
  utils.py          # file I/O, API request helpers
configs/
  model_configs.json  # chat templates + stop tokens per model
scripts/
  docker-entrypoint.sh  # container entrypoint (gen_ins + gen_res)
Dockerfile
docker-compose.yml
```

## Citation

If you find the model, data, or code useful, please cite the original Magpie paper 🤩:

```bibtex
@article{xu2024magpie,
  title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
  author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
  journal={ArXiv},
  year={2024},
  volume={abs/2406.08464},
  url={https://api.semanticscholar.org/CorpusID:270391432}
}
```
