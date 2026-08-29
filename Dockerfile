# Magpie + llama.cpp (GGUF) runtime
#
# Build:
#   docker build -t magpie-llamacpp:latest .
#
# Base: CUDA 13.0 runtime (Ubuntu 22.04, Python 3.10)
# CUDA 13.0 is required for Blackwell GPUs (RTX 50-series, sm_120), e.g. RTX 5060 Ti.
# CUDA 12.4 (used previously) does NOT support Blackwell.
# llama.cpp does the GPU inference; torch is only needed for the tokenizer,
# so we install the CPU build of torch to keep the image smaller.
FROM nvidia/cuda:13.0.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

# llama.cpp bindings with CUDA 13.0 support (prebuilt wheel, Blackwell-compatible)
# Direct wheel URL ensures the CUDA build is used (not the PyPI CPU wheel).
RUN pip install --no-cache-dir \
    https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu130/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl

# Python deps: torch (CPU) + transformers (tokenizer) + misc
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install transformers sentencepiece numpy tqdm requests

# Copy the magpie codebase
WORKDIR /app
COPY . /app/magpie

WORKDIR /app/magpie/scripts
RUN chmod +x docker-entrypoint.sh magpie-qwen3.6-35b-a3b-gguf.sh

ENTRYPOINT ["/app/magpie/scripts/docker-entrypoint.sh"]
