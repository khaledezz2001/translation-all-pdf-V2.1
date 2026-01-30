FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------
# System dependencies
# -----------------------------------------------------
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Python dependencies
# IMPORTANT: do NOT install torch
# -----------------------------------------------------
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --no-deps -r /requirements.txt

# -----------------------------------------------------
# Pre-download models (offline)
# -----------------------------------------------------
RUN python3 - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/models/hf/qwen",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="Helsinki-NLP/opus-mt-ru-en",
    local_dir="/models/hf/marian-ru-en",
    local_dir_use_symlinks=False
)
EOF

# -----------------------------------------------------
# Hugging Face offline configuration
# -----------------------------------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# -----------------------------------------------------
# CUDA arch safety (4090 + 5090)
# -----------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# -----------------------------------------------------
# App
# -----------------------------------------------------
WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "handler.py"]
