# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

"""
Configuration file for Synthetic Image Metrics Toolkit.
Defines metrics, dataset, and generator configurations.
"""
from dataset import BaseDataset

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

# 0. Import necessary packages for data loading.
import ...

# 1. Define the function(s) to load data and, optionally, labels.
class CustomDataset(BaseDataset):
    def _load_files(self):
        """
        Load input data file(s) and return a NumPy array.
        Expects images to be in (N, C, H, W) format.
        """
        pass

    def _load_raw_labels(self):
        """
        (Optional) Function to load the labels, for multi-class datasets.
        Expects Numpy array of shape: (N,) - e.g.,: array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0])
        """
        pass

# 2. Define the path(s) to the data file and, optionally, path to the label file and additional settings.
DATASET = {

    # Class definition, to load your data
    "class": CustomDataset,

    # Additional parameters required for loading the dataset.
    "params": 
    {
        # Path to the dataset file containing the medical images (in NIfTI format).
        "path_data": "path_to/data", # Be sure to change "path_data" with the actual attribute provided in the dataset

        # Path to an optional labels file.
        # If None, the dataset will be treated as unlabelled.
        "path_labels": "path_to/labels",

        # Flag to enable label usage.
        "use_labels": False,

        # Set if you want to upload a subset of data from the dataset.
        # If None, the whole dataset will be loaded
        "size_dataset": None
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
    # Load the pre-trained network
    G = ...

    # Define latent dimension and number of classes
    G.z_dim = ...
    G.c_dim = ... # 1 for single class dataset
    return G

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