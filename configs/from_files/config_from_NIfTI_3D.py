# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

"""
Configuration file for Synthetic Image Metrics Toolkit.
Defines metrics, dataset, and generator configurations.
"""
from dataset3D_BIDS import BidsDataset

# ----------------------------------- Metrics --------------------------------------

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
    "RUN_DIR": "Synthetic_Images_Metrics_Toolkit/EXPERIMENT_metrics",
    # Define the number of GPUs to use for computation.
    # Set 0 for CPU mode.
    "NUM_GPUS": 1,
    # Set the batch size to use while computing the embeddings of real and synthetic images
    "BATCH_SIZE": 64,
    # Set data type ('2D' or '3D')
    "DATA_TYPE": '3D',
    # Enable or disable caching of feature statistics. When True, cached data is reused (if available).
    "USE_CACHE": True,
    # Set verbosity for logging and debugging.
    "VERBOSE": True,
    # Path to an optional Outlier Classifier (OC) detector model, for the computation of pr_auth.
    # If None, the OC detector will be trained during metric computation.
    "OC_DETECTOR_PATH": None
}

# ---------------------------- Metrics  configurations -----------------------------

METRICS_CONFIGS = {

    # Some metrics are computed estimating data manifold through k-Nearest Neighbor.
    # Set the size of k (nhood_size):
    "nhood_size":
        {
            "pr": 2,      # For improved Precision & Recall (Karras et al.)
            "prdc": 2,    # For precision, recall, density, coverage (Naeem et al.)
            "pr_auth": 2  # For alpha-precision, beta-recall, authenticity (Alaa et al.)
        },

    # If you are perfroming the k-NN analysis, set the number of images to visualize in a num_real x num_synth grid.
    "K-NN_configs":
        {
            "num_real": 2, # Number of real images to visualize (the closest to any synthetic images).
            "num_synth": 3 # Number of synthetic images to visualize, ranked by similarity, next to each real image.
        },

    # The computation of some metrics require the resize of the images to the size of the input required by a pre-trained model
    # If False, images are resized with the PIL.BICUBIC resizer. If True, zero-padding is performed (ideal if the image has black background, such as the brain MRI)
    "padding": True
}

# ----------------------------- Real data configuration ----------------------------

DATASET = {

    # Class definition, to load real data
    "class": "<defined below: NiftiDataset>",

    # Additional parameters required for loading the dataset.
    "params": 
    {
        # Path to the dataset file containing the real images (in NIfTI format).
        "path_data": r"data\OpenBHB\OpenBHB_3D_BIDS",
        # Path to an optional labels file. If None, the dataset will be treated as unlabelled.
        "path_labels": None,
        # Flag to enable label usage.
        "use_labels": False,
        # Number of real images to use, if None using all
        "size_dataset": None,
    }
}

# ----------------------------- Synthetic data configuration -----------------------------

# Flag to determine the mode of operation
USE_PRETRAINED_MODEL = False  # Set to False to load synthetic images from files

SYNTHETIC_DATA = {

    # Configuration for direct synthetic images mode
    "from_files": 
        {
        # Class definition, to load synthetic data
        "class": "<defined below: NiftiDataset>",
        
        "params":
            {
            # Path to directory or file containing synthetic images
            "path_data": r"data\OpenBHB\OpenBHB_3D_BIDS",
            # Path to an optional labels file. If None, the dataset will be treated as unlabelled.
            "path_labels": None,
            # Flag to enable label usage.
            "use_labels": False,
            # Number of synthetic images to use, if None using all
            "size_dataset": None,  
            }
        }
}

# ----------------------------- Functions and classes definition -----------------------------

# -> For data loading

import nibabel as nib
import numpy as np

class NiftiDataset(BidsDataset):
    def _load_files(self, input):
        """
        Load a 3D NIfTI file and return a NumPy array.
        Expects images to be in (C, H, W, D) format.
        """
        data = nib.load(input).get_fdata()

        data = np.rot90(data, k=3, axes=(0, 1))
        data = np.rot90(data, k=1, axes=(0, 2))
        data = np.rot90(data, k=1, axes=(1, 2))

        data = np.expand_dims(data, axis=0)

        # Normalize to the range [0, 255]
        min_val = data.min()
        max_val = data.max()
        data = (data - min_val) / (max_val - min_val)
        data *= 255

        return data.copy() # [n_channels, img_resolution, img_resolution, img_resolution]

    def _load_raw_labels(self):
        pass

DATASET["class"] = NiftiDataset
SYNTHETIC_DATA["from_files"]["class"] = NiftiDataset