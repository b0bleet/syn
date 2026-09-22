# RunPod serverless image for the decision scorer.
#
#   docker build -f deploy/runpod/Dockerfile -t <registry>/syn-scorer .
#
# The worker loads SYN_MODEL at init. Qwen/Qwen3.5-4B in bfloat16 fits a 24 GB card.
# Qwen3-8B needs 24 GB, Qwen3-14B needs 48 GB, Qwen3-32B needs 80 GB. Set the
# endpoint's Model field to SYN_MODEL so RunPod caches the weights on the host:
# HF_HOME below is where that cache (or a network volume) is mounted, so workers
# load the weights from disk instead of downloading them on every cold start.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[local]" "runpod>=1.7,<2" "hf_transfer>=0.1,<1"

COPY deploy/runpod/handler.py ./handler.py

ENV SYN_BACKEND=local \
    SYN_READOUT=letters \
    SYN_MODEL=Qwen/Qwen3.5-4B \
    SYN_ORDERINGS=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python", "-u", "handler.py"]
