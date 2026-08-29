# Magpie + OpenAI-compatible API runtime
#
# Build:
#   docker build -t magpie-api:latest .
#
# This image does NOT run a model locally. It generates synthetic data by
# calling an external OpenAI-compatible API (e.g. a vLLM / llama.cpp server
# that aggregates GPUs across one or more machines).
#
# Base: Ubuntu 22.04 with Python 3.10 (no CUDA / GPU needed).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

# Python deps: requests (API calls) + tqdm
RUN pip install --no-cache-dir requests tqdm

# Copy the magpie codebase
WORKDIR /app
COPY . /app/magpie

WORKDIR /app/magpie/scripts
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["/app/magpie/scripts/docker-entrypoint.sh"]
