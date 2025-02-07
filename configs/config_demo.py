# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

"""
Configuration file for Synthetic Image Metrics Toolkit.
Defines metrics, dataset, and generator configurations.
"""

# -------------------------------- Metrics --------------------------------

# Define the metrics to compute.
# Available options: fid50k,kid50k,pr50k3,ppl_zfull,pr_auth,prdc,knn
METRICS = [
    # QUANTITATIVE METRICS:
        "fid",             # Frechét Inception distance (FID) (Karras et al.)
        "kid",             # Kernel Inception distance (KID) (Karras et al.)
        "is_",             # Inception score (IS) (Karras et al.)
        "pr",              # Improved Precision & Recall (Karras et al.)
        "pr_auth",         # alpha-precision, beta-recall, authenticity (Alaa et al.)
        "prdc",            # Precision, recall, density, coverage (Naeem et al.)
    
    # QUALITATIVE METRICS:
        "knn",             # k-NN analysis (Lai et al.)
          ]

# -------------------------------- Runtime configurations --------------------------------

CONFIGS = {

    # Define the path where the results should be saved.
    # This is the directory where all metric computations will be stored.
    "RUN_DIR": "Synthetic_Images_Metrics_Toolkit/metrics_output",

    # Define the number of synthetic images to generate for computing the metrics.
    # Default: 50,000
    "NUM_SYNTH": 50000,

    # Define the number of GPUs to use for computation.
    "NUM_GPUS": 1,

    # Set verbosity for logging and debugging.
    # If True, the script will print logs and progress updates.
    "VERBOSE": True,

    # Path to an optional Outlier Classifier (OC) detector model.
    # If None, the OC detector will be trained during metric computation.
    "OC_DETECTOR_PATH": None
}

# ----------------------------- Real dataset configuration ----------------------------

DATASET = {
    # The module where the dataset class is implemented (./training/dataset_mediffusion.py).
    "module": "training.dataset",

    # The class name of the dataset.
    # The script will dynamically import and instantiate this class.
    "class_name": "NiftiDataset",

    # Additional parameters required for loading the dataset.
    "params": 
    {
        # Path to the dataset file containing the medical images (in NIfTI format).
        "path_data": "path_to/data", # Be sure to change "path_data" with the actual attribute provided in the dataset

        # Path to an optional labels file.
        # If None, the dataset will be treated as unlabelled.
        "path_labels": "path_to/labels",

        # Flag to enable label usage.
        "use_labels": False
    }
}

# ----------------------------- Generator configuration -----------------------------

# 0. Import necessary packages.
import ...

# 1. Define a function to load the generator network.
def load_network(network_path):
    """
    Input: network_path (str): path to the pre-trained generator

    Output: pre-trained generator model
    """
    pass

# 2. Define a custom function to generate images using the generator.
def run_generator(z, c, opts):
    
    # Generate images
    img = opts.G(...)
    
    return img # [batch_size, n_channels, img_resolution, img_resolution]

# 3. Define the path to the pre-trained generator
network_path = "path_to/pretrained/generator"

# 4. Generator configuration dictionary
GENERATOR = {
    "network_path": network_path,

    "load_network": load_network,

    "run_generator": run_generator

    }