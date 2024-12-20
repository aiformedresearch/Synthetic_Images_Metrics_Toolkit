# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Install Python 3.7 and set as default
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
# RUN update-alternatives --set python3 /usr/bin/python3.7

# Ensure pip is up to date
RUN python3 -m ensurepip --upgrade
RUN python3 -m pip install --upgrade pip

# Install required libraries
RUN pip install --no-cache-dir \
    pandas==1.3.5 \
    nibabel==4.0.2 \
    tensorflow==2.4.1 \
    matplotlib==3.5.3 \
    tqdm==4.64.1 \
    scikit-learn==1.0.2 \
    click==7.1.2 \
    requests==2.24 \
    pyspng==0.1.1 \
    imageio-ffmpeg==0.4.3 \
    psutil==5.7.2

# Create the required directories
RUN mkdir -p /Metrics_Toolkit/outputs

# Copy project files into the container
COPY ./metrics/ /Metrics_Toolkit/metrics/
COPY ./training/ /Metrics_Toolkit/training/
COPY ./representations/ /Metrics_Toolkit/representations/
COPY ./torch_utils/ /Metrics_Toolkit/torch_utils/
COPY ./dnnlib/ /Metrics_Toolkit/dnnlib/
COPY legacy.py /Metrics_Toolkit/
COPY ./calc_metrics_demo.py /Metrics_Toolkit/
COPY ./calc_metrics_StyleGAN.py /Metrics_Toolkit/

# Set the working directory
WORKDIR /Metrics_Toolkit