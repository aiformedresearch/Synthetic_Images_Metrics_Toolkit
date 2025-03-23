# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Install Python 3.10 and set as default
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    gcc g++ make \
    && apt-get clean
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Ensure pip is up to date
RUN python3 -m ensurepip --upgrade
RUN python3 -m pip install --upgrade pip

# Install required libraries
RUN pip install --no-cache-dir \
    tensorflow==2.12.0 \
    matplotlib==3.5.3 \
    tqdm==4.64.1 \
    scikit-learn==1.0.2 \
    click==7.1.2 \
    requests==2.24 \
    pyspng==0.1.1 \
    imageio-ffmpeg==0.4.3 \
    psutil==5.7.2 \
    reportlab==4.3.1
# Install libraries for data loading (NIfTI and CSV files):
RUN pip install --no-cache-dir \
    nibabel==4.0.2 \
    pandas==1.3.5 
# Install libraries for computing metrics for Mediffusion:
RUN pip install --no-cache-dir \
#   typing-extensions==4.12.2 \
    mediffusion==0.7.1 \
    bkh_pytorch_utils==0.9.3 \
    torchextractor==0.3.0 \
    OmegaConf==2.3.0

# Create the required directories
RUN mkdir -p /Synthetic_Images_Metrics_Toolkit/outputs

# Copy project files into the container
COPY ./metrics/ /Synthetic_Images_Metrics_Toolkit/metrics/
COPY ./representations/ /Synthetic_Images_Metrics_Toolkit/representations/
COPY ./torch_utils/ /Synthetic_Images_Metrics_Toolkit/torch_utils/
COPY ./dnnlib/ /Synthetic_Images_Metrics_Toolkit/dnnlib/
COPY legacy.py /Synthetic_Images_Metrics_Toolkit/
COPY ./dataset.py /Synthetic_Images_Metrics_Toolkit/
COPY ./calc_metrics.py /Synthetic_Images_Metrics_Toolkit/

# Set the working directory
WORKDIR /Synthetic_Images_Metrics_Toolkit