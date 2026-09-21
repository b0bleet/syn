# RunPod serverless image for the decision scorer.
#
#   docker build -f deploy/runpod/Dockerfile -t <registry>/syn-scorer .
#
# The worker loads SYN_MODEL at init. Qwen/Qwen3-8B in bfloat16 needs a 24 GB card
# (RTX 4090, L40S, A5000); Qwen3-14B needs 48 GB; Qwen3-32B needs 80 GB. Attach a
# network volume so the weights download once per volume, not once per worker;
# HF_HOME below lands in it automatically when one is mounted at /runpod-volume.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[local]" "runpod>=1.7,<2" "hf_transfer>=0.1,<1"

COPY handler.py ./handler.py

ENV SYN_BACKEND=local \
    SYN_READOUT=letters \
    SYN_MODEL=Qwen/Qwen3-8B \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python", "-u", "handler.py"]
