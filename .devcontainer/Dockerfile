FROM tensorflow/tensorflow:2.17.0-gpu

RUN apt-get update && apt-get install -y git

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      "numpy>=1.23.5,<2" \
      "pandas>=2.0,<2.3" \
      "scipy>=1.10,<1.14" \
      "h5py>=3.8,<4" \
      "anndata>=0.10,<0.12" \
      "tqdm>=4.66,<5"\
      "huggingface_hub>=0.23"