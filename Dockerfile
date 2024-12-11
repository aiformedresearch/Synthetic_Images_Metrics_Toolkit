# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN conda install pandas==1.3.5
RUN conda install -c conda-forge nibabel==4.0.2
RUN conda install tensorflow=2.4.1
RUN conda install matplotlib=3.5.3 tqdm=4.64.1
RUN conda install anaconda::scikit-learn=1.0.2
RUN pip install click==7.1.2 requests==2.24 pyspng==0.1.1 imageio-ffmpeg==0.4.3
RUN pip install psutil==5.7.2 legacy

RUN apt-get update
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN mkdir /Metrics_Toolkit
RUN mkdir /Metrics_Toolkit/outputs

COPY ./metrics/ /Metrics_Toolkit/metrics/
COPY ./training/ /Metrics_Toolkit/training/
COPY ./representations/ /Metrics_Toolkit/representations/
COPY ./torch_utils/ /Metrics_Toolkit/torch_utils/
COPY ./dnnlib/ /Metrics_Toolkit/dnnlib/
COPY legacy.py /Metrics_Toolkit/
COPY ./calc_metrics_demo.py /Metrics_Toolkit/
COPY ./calc_metrics_StyleGAN.py /Metrics_Toolkit/

WORKDIR /Metrics_Toolkit