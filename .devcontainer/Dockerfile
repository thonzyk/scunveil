FROM tensorflow/tensorflow:2.21.0-gpu-jupyter

RUN python -m pip install --no-cache-dir --no-deps \
    --target /opt/cudnn9 "nvidia-cudnn-cu12==9.3.0.75"

ENV LD_LIBRARY_PATH=/opt/cudnn9/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
