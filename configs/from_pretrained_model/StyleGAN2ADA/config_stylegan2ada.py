# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

"""
Configuration file for Synthetic Image Metrics Toolkit.
Defines metrics, dataset, and generator configurations.
"""
from dataset import BaseDataset

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
    "RUN_DIR": "Synthetic_Images_Metrics_Toolkit/EXPERIMENT_StyleGAN2-ADA",
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
            "pr": 5,      # For improved Precision & Recall (Karras et al.)
            "prdc": 5,    # For precision, recall, density, coverage (Naeem et al.)
            "pr_auth": 5  # For alpha-precision, beta-recall, authenticity (Alaa et al.)
        },

    # If you are perfroming the k-NN analysis, set the number of images to visualize in a num_real x num_synth grid.
    "K-NN_configs":
        {
            "num_real": 3, # Number of real images to visualize (the closest to any synthetic images).
            "num_synth": 5 # Number of synthetic images to visualize, ranked by similarity, next to each real image.
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
        "path_data": "data/real_images_simulation.nii.gz",
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
USE_PRETRAINED_MODEL = True  # Set to False to load synthetic images from files

SYNTHETIC_DATA = {

    # Configuration for pre-trained model mode
    "pretrained_model": 
        {
        # Path to the pre-trained generator
        "network_path": "Synthetic_Images_Metrics_Toolkit/configs/StyleGAN2ADA/pre-trained_generator.pkl",
        # Function to load the pre-trained generator (below in this script)
        "load_network": lambda network_path: _load_network(network_path),
        # Function to generate synthetic images from the pre-trained generator (below in this script)
        "run_generator": lambda z, opts: _run_generator(z, opts),
        # Number of images you want to generate
        "NUM_SYNTH": 500
        },
        
}

# ----------------------------- Functions and classes definition -----------------------------

# -> For data loading

import nibabel as nib
import numpy as np

class NiftiDataset(BaseDataset):
    def _load_files(self):
        """
        Load a NIfTI file and return a NumPy array.
        Expects images to be in (N, C, H, W) format.
        """
        data = nib.load(self.path_data)
        data = np.asanyarray(data.dataobj) # numpy array of shape (W,H,C,N)
        data = np.float64(data)

        W, H, C, N = data.shape
        assert W==H

        # Swap axes to have the format (N,C,W,H)
        data = np.swapaxes(data, 0,3)
        data = np.swapaxes(data, 1,2)   # after swapping axes, array shape (N,C,H,W)

        return data # [batch_size, n_channels, img_resolution, img_resolution]

    def _load_raw_labels(self):
        pass

DATASET["class"] = NiftiDataset

#  -------    -------    -------    -------    -------    -------    -------    -------    -------

# -> for synthetic data generation:
import dnnlib
import legacy

def _load_network(network_path):
    with dnnlib.util.open_url(network_path, verbose=CONFIGS["VERBOSE"]) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'] # subclass of torch.nn.Module
    
    return G

def _run_generator(z, opts):
    img = opts.G(z=z, c=None)

    # Normalize pixel values to the standard [0, 255] range for image representation.
    img = (img.float() * 127.5 + 128).clamp(0, 255)#.to(torch.uint8)
    
    return img # [batch_size, n_channels, img_resolution, img_resolution]
