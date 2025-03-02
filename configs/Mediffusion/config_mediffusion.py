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
        "knn",              # k-NN analysis (Lai et al.)
           ]

# -------------------------------- Runtime configurations --------------------------------

CONFIGS = {

    # Define the path where the results should be saved.
    # This is the directory where all metric computations will be stored.
    "RUN_DIR": "Synthetic_Images_Metrics_Toolkit/EXPERIMENT_Mediffusion2D",

    # Define the number of synthetic images to generate for computing the metrics.
    # Default: 50,000
    "NUM_SYNTH": 500,

    # If you are perfroming the k-NN analysis, set the number of images to visualize in the grid.
    "K-NN_configs":
        {
            # Number of real images to visualize (the closest to any synthetic images).
            "num_real": 2,

            # Number of synthetic images to visualize, ranked by similarity, next to each real image.
            "num_synth": 4
        },

    # Define the number of GPUs to use for computation.
    # Set 0 for CPU mode.
    "NUM_GPUS": 0,

    # Set verbosity for logging and debugging.
    # If True, the script will print logs and progress updates.
    "VERBOSE": True,

    # Path to an optional Outlier Classifier (OC) detector model, for the computation of pr_auth.
    # If None, the OC detector will be trained during metric computation.
    "OC_DETECTOR_PATH": None
}

# ----------------------------- Real dataset configuration ----------------------------

# 0. Import necessary packages for data loading.
import nibabel as nib
import numpy as np

# 1. Define the function(s) to load data and, optionally, labels.
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

# 2. Define the path(s) to the data file and, optionally, path to the label file and additional settings.
DATASET = {

    # Class definition, to load your data
    "class": NiftiDataset,

    # Additional parameters required for loading the dataset.
    "params": 
    {
        # Path to the dataset file containing the medical images (in NIfTI format).
        "path_data": "data/real_images_simulation.nii.gz",

        # Path to an optional labels file.
        # If None, the dataset will be treated as unlabelled.
        "path_labels": None,

        # Flag to enable label usage.
        "use_labels": False,

        # Set if you want to upload a subset of data from the dataset.
        # If None, the whole dataset will be loaded
        "size_dataset": None
    }
}

# ----------------------------- Generator configuration -----------------------------

# 0. Import necessary packages.
import torch
from mediffusion import DiffusionModule

# 1. Define the path to the pre-trained generator
network_path = "Synthetic_Images_Metrics_Toolkit/configs/Mediffusion/pre-trained_generator.ckpt"

# 2. Define a function to load the generator network.
config_path= "Synthetic_Images_Metrics_Toolkit/configs/Mediffusion/config.yaml"
def load_network(network_path):
    G = DiffusionModule(config_path)
    network_dict = torch.load(network_path)['state_dict']
    G.load_state_dict(network_dict)
    G.eval().cuda().half()

    # Define latent dimension and number of classes
    G.z_dim = [1, 256, 256]
    G.c_dim = 1
    return G

# 3. Define a custom function to generate images using the generator.
def run_generator(z, opts):

    # Generate images using the specified inference protocol.
    img = opts.G.predict(z, inference_protocol="DDIM100") # List of images
    
    # Convert the list to a single torch tensor.
    img = torch.stack(img, dim=0)                    # torch tensor of shape [batch_size, 1, 256, 256]

    # Normalize pixel values to the standard [0, 255] range for image representation.
    img = (img.float() * 255.0).clamp(0, 255)
    
    return img # [batch_size, n_channels, img_resolution, img_resolution]

# 4. Generator configuration dictionary
GENERATOR = {
    # Path to the pre-trained generator checkpoint file.
    "network_path": network_path,

    "load_network": load_network,

    "run_generator": run_generator

    }