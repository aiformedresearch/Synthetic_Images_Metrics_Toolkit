<!--
SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>

SPDX-License-Identifier: NPOSL-3.0
-->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14845934.svg)](https://doi.org/10.5281/zenodo.14845934)
[![PyPI - Version](https://img.shields.io/pypi/v/sim_toolkit.svg?logo=pypi&logoColor=white)](https://pypi.org/project/sim-toolkit/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white)](https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit)
[<img src="https://img.shields.io/badge/  -Dockerhub-blue.svg?logo=docker&logoColor=white">](<https://hub.docker.com/r/aiformedresearch/metrics_toolkit>)
[<img src="https://img.shields.io/badge/Jupyter_Notebook-orange.svg?logo=jupyter&logoColor=white">](https://colab.research.google.com/drive/1EyO8hAu6sJw_gbE3bsHID-5IzUBjhm6B?usp=sharing)


# Synthetic Images Metrics Toolkit (SIM Toolkit)

<p align="center">
  <img src="https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit/raw/main/sim_toolkit/Images/logo.png" width="300" alt="SIM Toolkit logo">
</p>


The **Synthetic Images Metrics (SIM) Toolkit** is a Python library for evaluating the quality of **2D** and **3D** synthetic images. 

It provides metrics to assess:
- **Fidelity**: how realistic synthetic images are;
- **Diversity**: how well they cover the real data distribution ;
- **Generalization**: whether the model generates new samples instead of memorizing the training set. 

The toolkit automatically generates a **PDF report** with metric values, plots, and qualitative analyses.


📄 **Example report:**  
[report_metrics_toolkit.pdf](https://drive.google.com/file/d/1K_H0KCjjqr6rfi1tHYk03Gy3WhdcyKjk/view?usp=sharing)

## Installation
### 🐍 Python and compatibility
- **Tested / supported**: Python **3.10 – 3.12**
- **Recommended**: Python **3.10 or 3.11** (most robust with deep learning dependencies).
- Works on **CPU and GPU**.  
  For GPU acceleration, install a compatible [CUDA](https://developer.nvidia.org/cuda-downloads) version (CUDA ≥ 11 recommended) and use the matching PyTorch wheels.

### 🔧 Basic install
```bash
pip install sim_toolkit
```
This installs the core library (no heavy backends).

#### ➕ Optional extras

Install only what you need:
- **PyTorch backend** (required for core metrics):
    ```bash
    pip install "sim_toolkit[torch]"
    ```
- **TensorFlow backend** (only needed for 2D `pr_auth`, `prdc`, and `knn`):
    ```bash
    pip install "sim_toolkit[tf]"
    ```
    ⚠️ Officially supported on Python 3.10–3.11.  
    On Python 3.12, this extra may not resolve a compatible TF wheel; if TF is already installed and working (e.g. Colab), SIM Toolkit will detect and use it.  
- **File format / dataset support**:
    ```bash
    pip install "sim_toolkit[nifti]"        # NIfTI (.nii/.nii.gz)
    pip install "sim_toolkit[dcm]"          # DICOM (.dcm)
    pip install "sim_toolkit[tiff]"         # TIFF
    pip install "sim_toolkit[opencv]"       # JPEG/PNG via OpenCV, etc.
    pip install "sim_toolkit[csv]"          # CSV-based labels/metadata
    ```
You can combine extras, for example:
```bash
pip install "sim_toolkit[torch,nifti]"
```
If a required backend is missing at runtime, SIM Toolkit will raise a clear, short error with the exact `pip install` command to run.

### 🐳 Docker
You can run the SIM Toolkit in a fully isolated, ready-to-use Docker environment.

1. Pull the image:
    ```bash
    docker pull aiformedresearch/metrics_toolkit:4.1.1
    ```

2. Run the Docker container
    ```
    docker run -it --gpus all \
      -v /absolute/path/to/real_data:/workspace/data/real \
      -v /absolute/path/to/synt_data:/workspace/data/synt \
      -v /absolute/path/to/runs:/workspace/runs \
      aiformedresearch/metrics_toolkit:4.1.1
    ```
      - The `--gpus all` flag enables GPU support. Specify a GPU if needed, e.g., `--gpus 0`.
      - The `-v host/container` mount the local directories to the working directory `/workspace` inside the container. 

Refer to the [Usage](#usage) section for detailed instructions about running the main script. 

## Usage (Python API)
> 🚀 **Last Update**: No config file needed — everything is passed as function arguments via `sim.compute(...)`.

To run SIM Toolkit you only need to define how to load:

- **Real data** → choose a built-in dataset tag or define a small custom loader (👉 A).
- **Synthetic data**:
    - from files → same as real data (👉 A);
    - from a pretrained generator → on-the-fly synthesis (👉 B).

### A) From image files 
Load and evaluate synthetic images directly from files or directories.

**Supported built-in dataset tags**: `nifti`, `dcm`, `tiff`, `jpeg`, `png`, or `auto` (infers format from `path_data`).

**Custom format or custom folder structure?**  
You can plug in your own loader **without modifying SIM Toolkit**:  
- define a small dataset class inheriting from `sim_toolkit.datasets.base.BaseDataset`
- then either:
  - pass the class directly: `real_dataset=MyDataset`, or  
  - point to a file: `real_dataset="path/to/my_dataset.py:MyDataset"`

📄 See: [sim_toolkit/datasets](https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit/blob/main/sim_toolkit/datasets/README.md)

**Basic example**
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
    real_dataset="auto",     # "nifti" | "dcm" | "tiff" | "jpeg" | "png" | "auto"
    real_params={
        "path_data": "data/real_images",
        "path_labels": None, 
        "use_labels": False, 
        "size_dataset": None # (int) if None, using all 
        },

    ## Synthetic data (from files)
    synth_dataset="auto",    # "nifti" | "dcm" | "tiff" | "jpeg" | "png" | "auto"
    synth_params={
        "path_data": "data/synt_images",
        "path_labels": None, 
        "use_labels": False, 
        "size_dataset": None # (int) if None, using all 
        },
    )
```

📖 Tutorial (file-based usage):  
[Colab – SIM Toolkit with your data](https://colab.research.google.com/drive/14ebfSXuMn--heFF-AyjT23MB2QFPgEU3?usp=sharing)

### B) From a pre-trained generator (no synthetic files)
Generate synthetic images on-the-fly using a pretrained generative model.

You provide:
- `load_network(network_path)` → loads your pretrained model
- `run_generator(z, c, opts)` → uses that model to generate a batch of images

Real data loading is identical to section A (built-in or custom dataset).

```python
import sim_toolkit as sim

def load_network(network_path):
    # user-provided loader returning a torch.nn.Module (G)
    ...

def run_generator(z, c, opts):
    """
    Args:
    - z:    Latent input for the generator.
              - For GANs: typically a tensor of shape (N, latent_dim).
              - For diffusion / other models: can be any shape your model expects.
    - c:    (optional) class labels, for conditional generation.
    - opts:  Helper object from the SIM Toolkit.
              - opts.G :      the loaded generator/model (e.g., torch.nn.Module)
              - opts.device : torch.device to run generation on

    Must return a tensor of shape:
      - (N, C, H, W)    for 2D data
      - (N, C, H, W, D) for 3D data
    """
    ## Example:
    # img = opts.G(z, c)
    # return img
    ...

sim.compute(
    metrics=["fid", "kid", "is_", "prdc", "pr_auth", "knn"],
    run_dir="./runs/gen",

    ## Real data
    real_dataset="auto", # "nifti" | "dcm" | "tiff" | "jpeg" | "png" | "auto"
    real_params={"path_data": "data/real_images_simulation.nii.gz"},

    ## Synthetic data (from pre-trained generator)
    use_pretrained_generator=True,
    network_path="checkpoints/G.pkl",
    load_network=load_network,
    run_generator=run_generator,  
    num_gen=50000, # how many synthetic images to generate
)
```
📖 Tutorial (generator-based usage):  
[Colab – SIM Toolkit with your pre-trained model](https://colab.research.google.com/drive/1TMELn54mEmmFAErgOorhHWeXn9dvNLQM?usp=sharing)

All metric values, plots, and the final PDF report are saved under: `run_dir/`.

## Metrics

<p align="center">
  <img src="https://github.com/aiformedresearch/Synthetic_Images_Metrics_Toolkit/raw/main/sim_toolkit/Images/Metrics.png" width="300" alt="SIM Toolkit metrics">
</p>


### Quantitative 
| Flag      | Description | Source | Reference |
| :-----        | :-----: | :---------- | :---------- |
| `fid` | Fr&eacute;chet inception distance against the full dataset | [Karras et al.](https://github.com/NVlabs/stylegan2-ada-pytorch) | [Heusel et al. 2017](https://arxiv.org/abs/1706.08500)
| `kid` | Kernel inception distance against the full dataset         | [Karras et al.](https://github.com/NVlabs/stylegan2-ada-pytorch) | [Bi&nacute;kowski et al. 2018](https://arxiv.org/abs/1801.01401)
| `is_`       | Inception score against the full dataset (only 2D)                            | [Karras et al.](https://github.com/NVlabs/stylegan2-ada-pytorch) | [Salimans et al. 2016](https://arxiv.org/abs/1606.03498)
| `prdc`    |  Precision, recall, density, and coverage  against the full dataset                    | [Naeem et al.](https://github.com/clovaai/generative-evaluation-prdc) | [Kynk&auml;&auml;nniemi et al. 2019](https://arxiv.org/abs/1904.06991); [Naeem et al., 2020](https://proceedings.mlr.press/v119/naeem20a/naeem20a.pdf)
| `pr_auth`    |  	$\alpha$-precision, 	$\beta$-recall, and authenticity against the full dataset  | [Alaa et al.](https://github.com/vanderschaarlab/evaluating-generative-models) | [Alaa et al., 2022](https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf)

> ⚠️ **3D setup** 3D metrics use a 3D-ResNet50 feature extractor from [MedicalNet](https://github.com/Tencent/MedicalNet/tree/master), pre-trained on 23 medical imaging datasets. Ensure your domain is compatible; otherwise embeddings (and thus metrics) may not be meaningful.

### Qualitative
The toolkit automatically generates:  
- Grids of real and synthetic samples
- **PCA** and **t-SNE** visualizations of real vs. synthetic embeddings
- Summary plots embedded into the final PDF report

Additionally, you can enable:

- **k-NN analysis** (`knn` flag):

  The k-nearest neighbour (k-NN) visualization shows:
  - The `knn_num_real` real images most similar to *any* synthetic image *(default: `knn_num_real=3`)*.
  - For each of those real images, the `knn_num_synth` most similar synthetic samples, ranked by their cosine similarity *(default: `knn_num_synth=5`)*.

  These can be configured directly in `sim.compute(...)`:
  ```python
  sim.compute(
      metrics=["knn"],
      run_dir="./runs/knn_example",
      knn_num_real=3,
      knn_num_synth=5,
      # other args...
  )
📄 Example report:  
[report_sim_toolkit.pdf](https://drive.google.com/file/d/1K_H0KCjjqr6rfi1tHYk03Gy3WhdcyKjk/view?usp=sharing)

## Licenses
This project complies with the [REUSE Specification](https://reuse.software/). All source files are annotated with SPDX license identifiers, and full license texts are included in the `LICENSES` directory.

- **LicenseRef-NVIDIA-1.0**: Applies to code reused from NVIDIA's StyleGAN2-ADA repository: https://github.com/NVlabs/stylegan2-ada-pytorch, under the [NVIDIA Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).
- **MIT**:  For code reused from:
    - https://github.com/vanderschaarlab/evaluating-generative-models; 
    - https://github.com/clovaai/generative-evaluation-prdc.
- **BSD-3-Clause**: Applies to two scripts reused from https://github.com/vanderschaarlab/evaluating-generative-models;
- **NPOSL-3.0**: Applies to the code developed specifically for this repository.

For detailed license texts, see the `LICENSES` directory.

## Aknowledgments
This toolkit builds upon and adapts components from:
- [NVIDIA StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) – core project layout, `dnnlib` utilities, dataset and metric infrastructure, and metric implementations (used under the NVIDIA Source Code License).
- [evaluating-generative-models](https://github.com/vanderschaarlab/evaluating-generative-models) – metric implementations.
- [generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc) – metric implementations.
- [MedicalNet](https://github.com/Tencent/MedicalNet) – pre-trained 3D feature extractor.

## Citation

If you use the SIM Toolkit in your research or publications, please cite:

```bibtex
@article{lai25,
  title   = {Generating Brain MRI with StyleGAN2-ADA: The Effect of the Training Set Size on the Quality of Synthetic Images},
  author = {Lai, Matteo and Mascalchi, Mario and Tessa, Carlo and Diciotti, Stefano},
  journal = {Journal of Imaging Informatics in Medicine},
  year    = {2025},
  issn = {2948-2933},
  url = {https://doi.org/10.1007/s10278-025-01536-0},
  doi = {10.1007/s10278-025-01536-0},
}
```
