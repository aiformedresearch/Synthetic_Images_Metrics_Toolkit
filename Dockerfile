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
RUN python -m pip install "sim_toolkit[torch,tf,nifti,tiff,jpeg,png,dcm]==4.0.0"

# Workspace for mounting data & runs
WORKDIR /workspace

# Default sanity check
CMD ["python", "-c", "import sim_toolkit as sim; print('SIM Toolkit version:', getattr(sim, '__version__', 'unknown'))"]