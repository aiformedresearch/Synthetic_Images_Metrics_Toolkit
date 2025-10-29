<!--
SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>

SPDX-License-Identifier: NPOSL-3.0
-->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14845935.svg)](https://doi.org/10.5281/zenodo.14845935) [<img src="https://img.shields.io/badge/  -Dockerhub-blue.svg?logo=docker&logoColor=white">](<https://hub.docker.com/r/aiformedresearch/metrics_toolkit>) [<img src="https://img.shields.io/badge/Jupyter_Notebook-orange.svg?logo=jupyter&logoColor=white">](https://colab.research.google.com/drive/1EyO8hAu6sJw_gbE3bsHID-5IzUBjhm6B?usp=sharing)

# Synthetic_Images_Metrics_Toolkit

<p align="center">
  <img src="Images/logo.png" width="300" title="metrics">
</p>

The **Synthetic Images Metrics (SIM) Toolkit** provides a comprehensive collection of state-of-the-art metrics for evaluating the quality of **2D** and **3D** synthetic images. 

These metrics enable the assessment of:
- **Fidelity**: the realism of synthetic data;
- **Diversity**: the coverage of the real data distribution;
- **Generalization**: the generation of authentic, non-memorized images. 

### 📊 Automated Report Generation
The toolkit produces a comprehensive PDF report with quantitative metrics and qualitative analysis.

➡️ **Example report:** 📄 [report_metrics_toolkit.pdf](https://drive.google.com/file/d/1K_H0KCjjqr6rfi1tHYk03Gy3WhdcyKjk/view?usp=sharing)

## Installation
Before proceeding, ensure that [CUDA](https://developer.nvidia.com/cuda-downloads) is installed. CUDA 11.0 or later is recommended.

### 🔧 Option A: Conda + local source (recommended for development)
Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) for your operating system.

 ```bash
conda create -n sim_toolkit python=3.10 -y
conda activate sim_toolkit

# Clone and install (editable)
git clone https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit.git
cd Synthetic_Images_Metrics_Toolkit
pip install -e .
```
Add extras as needed:

```bash
# PyTorch backend (CPU by default; see CUDA note below)
pip install ".[torch]"

# TensorFlow backend (auto-selects the right wheel for your OS/arch)
pip install ".[tf]"

# Data formats / IO
pip install ".[nifti,opencv,tiff,csv,dicom]"
```

### 🐳 Option B: Use Docker
0. Install [Docker](https://docs.docker.com/get-docker/) for your operating system.

1. Pull the Docker image
    ```
    docker pull aiformedresearch/metrics_toolkit:3.1
    ```

2. Run the Docker container
    ```
    docker run -it --gpus all \
      -v /absolute/path/to/real_data:/Synthetic_Images_Metrics_Toolkit/data \
      -v /absolute/path/to/config_file:/Synthetic_Images_Metrics_Toolkit/configs \
      -v /absolute/path/to/local_output_directory:/Synthetic_Images_Metrics_Toolkit/outputs \
      aiformedresearch/metrics_toolkit
    ```
      - The `--gpus all` flag enables GPU support. Specify a GPU if needed, e.g., `--gpus 0`.
      - The `-v` flag is used to mount the local directories to the working directory `Synthetic_Images_Metrics_Toolkit` inside the container. 

Refer to the [Usage](#usage) section for detailed instructions about running the main script. 


## Usage (NEW)
> 🚀 **Last Update:** The SIM Toolkit now runs directly via Python API.
No Python configuration file is required — pass paths and flags programmatically.

### A) From image files 
Load and analyze pre-existing synthetic images stored in files or directories.

Currently, the SIM toolkit support NIfTI, DICOM, TIF, JPG, and PNG formats. If you need to work with different data types, you can open an issue or implement it from `sim_toolkit/datasets`.

```python
import sim_toolkit as sim

sim.compute(
    metrics=["fid", "kid", "is_", "prdc", "pr_auth", "knn"],
    run_dir="./runs/exp1",
    num_gpus=1,              # set 0 to force CPU
    batch_size=64,
    data_type="2D",          # or "3D"
    use_cache=True,
    padding=False,

    ## Real data
    real_dataset="auto",     # "nifti" | "dicom" | "tiff" | "jpeg" | "png" | "auto"
    real_params={
        "path_data": "data/real_images",
        "path_labels": None, 
        "use_labels": False, 
        "size_dataset": None # (int) if None, using all 
        },

    ## Synthetic data (from files)
    synth_dataset="auto",    # "nifti" | "dicom" | "tiff" | "jpeg" | "png" | "auto"
    synth_params={
        "path_data": "data/synt_images",
        "path_labels": None, 
        "use_labels": False, 
        "size_dataset": None # (int) if None, using all 
        },
    )
```
📖 Find here a [short tutorial](https://colab.research.google.com/drive/14ebfSXuMn--heFF-AyjT23MB2QFPgEU3?usp=sharing) to use the SIM Toolkit with your data.

### B) From a pre-trained generator (no synthetic files)
Generate synthetic images on-the-fly using a pretrained generative model.

If you want to generate synthetic images dynamically using a pretrained generator, you need to define functions to:
- Load the pretrained generator
- Generate new samples

```python
import sim_toolkit as sim

def load_network(pkl_path):
    # user-provided loader returning a torch.nn.Module (G)
    ...

def run_generator(G, labels, num_images, dataset, device):
    # custom generator sampling
    ...

sim.compute(
    metrics=["fid", "kid", "is_", "prdc", "pr_auth", "knn"],
    run_dir="./runs/gen",

    ## Real data
    real_dataset="auto", # "nifti" | "jpeg" | "tiff" | "dicom" | "auto"
    real_params={"path_data": "data/real_images_simulation.nii.gz"},

    ## Synthetic data (from pre-trained generator)
    use_pretrained_generator=True,
    network_path="checkpoints/G.pkl",
    load_network=load_network,
    run_generator=run_generator,  
    num_gen=50000, # how many synthetic images to generate
)
```
📖 Find here a [short tutorial](https://colab.research.google.com/drive/1TMELn54mEmmFAErgOorhHWeXn9dvNLQM?usp=sharing) to use the SIM Toolkit with your pre-trained model.

All metric results, figures, and the final report are written under `run_dir/`.

## Metrics overview

<p align="center">
  <img src="Images/Metrics.png" width="400" title="metrics">
</p>

### Quantitative metrics
The following quantitative metrics are available:

| Metric flag      | Description | Original implementation |
| :-----        | :-----: | :---------- |
| `fid` | Fr&eacute;chet inception distance<sup>[1]</sup> against the full dataset | [Karras et al.](https://github.com/NVlabs/stylegan2-ada-pytorch)
| `kid` | Kernel inception distance<sup>[2]</sup> against the full dataset         | [Karras et al.](https://github.com/NVlabs/stylegan2-ada-pytorch)
| `is_`       | Inception score<sup>[3]</sup> against the full dataset (only 2D)                            | [Karras et al.](https://github.com/NVlabs/stylegan2-ada-pytorch)
| `prdc`    |  Precision, recall<sup>[4]</sup>, density, and coverage<sup>[5]</sup>  against the full dataset                    | [Naeem et al.](https://github.com/clovaai/generative-evaluation-prdc)
| `pr_auth`    |  	$\alpha$-precision, 	$\beta$-recall, and authenticity<sup>[6]</sup> against the full dataset  | [Alaa et al.](https://github.com/vanderschaarlab/evaluating-generative-models)

> ⚠️ **3D metrics** uses a 3D-ResNet50 feature extractor from [MedicalNet](https://github.com/Tencent/MedicalNet/tree/master), pre-trained on 23 medical imaging datasets. Ensure your domain is compatible; otherwise embeddings (and thus metrics) may not be meaningful.

References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016
4. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
5. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018
6. [Reliable Fidelity and Diversity Metrics for Generative Models](https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf), Naeem et al., 2020
7. [How Faithful is your Synthetic Data?
Sample-level Metrics for Evaluating and Auditing Generative Models](https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf), Alaa et al., 2022

### Qualitative metrics
The SIM Toolkit automatically generates **t-SNE** and **PCA** visualizations of real and synthetic image embeddings for each metric, to qualitatively assess the fidelity and diversity of the synthetic images. These plots are included in the final report, which also provides detailed guidance to help users interpret the resulting metric scores.
You can view an example report [here](https://drive.google.com/file/d/1K_H0KCjjqr6rfi1tHYk03Gy3WhdcyKjk/view?usp=sharing).

In addition, the toolkit supports **k-NN analysis**, enabling users to qualitatively evaluate how well synthetic samples generalize with respect to real data:

| Metric flag      | Description | Original implementation |
| :-----        | :-----: | :---------- |
| `knn` | k-nearest neighbors (k-NN) analysis, to assess potential memorization of the model | [Lai et al.](https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit) |

<p align="center">
  <img src="Images/knn_analysis.png" width="600" title="knn-analysis">
</p>

The k-NN analysis identifies and visualizes the `top_n` real images most similar to any synthetic sample (from a set of 50,000 generated samples). For each real image, the visualization displays the top `k` synthetic images ranked by their cosine similarity to the corresponding real image.

By default, `k=5` and `top_n=3`. These parameters can be customized in the *Metrics configurations* section of the configuration file.

## 🚧To-do list:

- [x] 3D data support;

- [x] Implement PCA and t-SNE to qualitatively assess diversity.

- [x] Simple Python API (no configuration file)

## Licenses
This repository complies with the [REUSE Specification](https://reuse.software/). All source files are annotated with SPDX license identifiers, and full license texts are included in the `LICENSES` directory.

### Licenses Used

1. **LicenseRef-NVIDIA-1.0**: Applies to code reused from NVIDIA's StyleGAN2-ADA repository: https://github.com/NVlabs/stylegan2-ada-pytorch, under the [NVIDIA Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).
2. **MIT**:  For code reused from:
    - https://github.com/vanderschaarlab/evaluating-generative-models; 
    - https://github.com/clovaai/generative-evaluation-prdc.
3. **BSD-3-Clause**: Applies to two scripts reused from https://github.com/vanderschaarlab/evaluating-generative-models;
4. **NPOSL-3.0**: Applies to the code developed specifically for this repository.

For detailed license texts, see the `LICENSES` directory.

## Aknowledgments
This repository builds on NVIDIA's StyleGAN2-ADA repository: https://github.com/NVlabs/stylegan2-ada-pytorch.
