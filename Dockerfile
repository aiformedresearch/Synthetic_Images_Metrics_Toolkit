# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Install Python 3.10 and system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common libgl1 gcc g++ make libaec-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    apt-get clean && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade pip

# Install required libraries
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    tensorflow==2.12.0 \
    keras==2.12.0 \
    matplotlib==3.5.3 \
    tqdm==4.64.1 \
    scikit-learn==1.0.2 \
    click==8.1.8 \
    requests==2.24 \
    pyspng==0.1.1 \
    imageio-ffmpeg==0.4.3 \
    psutil==5.7.2 \
    reportlab==4.3.1 \
    seaborn==0.13.2 \
    protobuf==4.25.6 \
    scipy==1.15.2 \
# Install libraries for data loading (NIfTI, TIFF and CSV files):
    nibabel==4.0.2 \
    pandas==1.3.5 \
    tifffile==2025.3.13 \
    imagecodecs==2025.3.30 \
    opencv-python==4.11.0.86 

# Optional: Install libraries for computing metrics for Mediffusion:
# RUN pip install --no-cache-dir \
#   typing-extensions==4.12.2 \
    # mediffusion==0.7.1 \
    # bkh_pytorch_utils==0.9.3 \
    # torchextractor==0.3.0 \
    # OmegaConf==2.3.0 

# Create the required directories
RUN mkdir -p /Synthetic_Images_Metrics_Toolkit/outputs

# Copy project files into the container
COPY ./metrics/ /Synthetic_Images_Metrics_Toolkit/metrics/
COPY ./representations/ /Synthetic_Images_Metrics_Toolkit/representations/
COPY ./torch_utils/ /Synthetic_Images_Metrics_Toolkit/torch_utils/
COPY ./dnnlib/ /Synthetic_Images_Metrics_Toolkit/dnnlib/
COPY legacy.py /Synthetic_Images_Metrics_Toolkit/
COPY ./dataset.py /Synthetic_Images_Metrics_Toolkit/
COPY ./dataset3D_BIDS.py /Synthetic_Images_Metrics_Toolkit/
COPY ./calc_metrics.py /Synthetic_Images_Metrics_Toolkit/

# Set the working directory
WORKDIR /Synthetic_Images_Metrics_Toolkit