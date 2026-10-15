# Approximation of the environment used for the GPU-based analyses
# (scripts/xai_freq_occlusion.py, scripts/xai_dft_lrp.py,
#  scripts/female_finetune_band_shift.py):
#   Ubuntu 24.04, Python 3.12, PyTorch 2.9, CUDA 13, NVIDIA T4 GPUs,
#   numpy 2.1, scikit-learn 1.7.1, librosa 0.11.0, zennit (version not recorded).
# The original image is not available; this file has not been built or tested
# against the published numbers. CPU-only parts of the repo use requirements.txt.
#
#   docker build -t speaker-sex-gpu .
#   docker run --gpus all -it -v "$PWD":/work -w /work speaker-sex-gpu bash
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130 \
 && pip install --no-cache-dir \
        "numpy==2.1.*" \
        scikit-learn==1.7.1 \
        librosa==0.11.0 \
        zennit \
        pandas==2.2.2 \
        scipy==1.16.1 \
        matplotlib==3.10.6 \
        pyarrow soundfile seaborn tqdm PyYAML

WORKDIR /work
