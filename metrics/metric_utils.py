# SPDX-FileCopyrightText: 2024 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: LicenseRef-NVIDIA-1.0
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. 
# Modifications copyright (c) 2024, Matteo Lai
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import torch.nn as nn
import dnnlib
import platform

from representations.OneClass import OneClassLayer
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import re
from PIL import Image
from pathlib import Path
import inspect
from metrics import metric_main
from representations import resnet3d
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
import PIL.Image
from matplotlib import gridspec
from tqdm import tqdm
import random

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, run_dir, batch_size, data_type, use_pretrained_generator, run_generator, network_pkl, num_gen, nhood_size, knn_config, padding, oc_detector_path, train_OC, cache, seed, G=None, G_kwargs={}, dataset_kwargs={}, dataset_synt_kwargs={}, num_gpus=1, rank=0, device=None, progress=None):
        assert 0 <= rank <= num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.dataset_synt_kwargs = dnnlib.EasyDict(dataset_synt_kwargs) if dataset_synt_kwargs is not None else None
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.run_dir        = run_dir
        self.gen_path       = network_pkl
        self.data_path      = dataset_kwargs.path_data
        self.max_size       = dataset_kwargs.size_dataset
        self.num_gen        = num_gen
        self.nhood_size     = nhood_size
        self.knn_config     = knn_config
        self.padding        = padding
        self.oc_detector_path = oc_detector_path
        self.train_OC       = train_OC
        self.run_generator  = run_generator
        self.use_pretrained_generator = use_pretrained_generator
        self.data_type      = data_type
        self.batch_size     = batch_size
        self.seed           = seed
        self.OC_params  = dict({"rep_dim": 32, 
                    "num_layers": 3, 
                    "num_hidden": 128, 
                    "activation": "ReLU",
                    "dropout_prob": 0.5, 
                    "dropout_active": False,
                    "LossFn": "SoftBoundary",
                    "lr": 2e-3,
                    "epochs": 2000,
                    "warm_up_epochs" : 10,
                    "train_prop" : 0.8,
                    "weight_decay": 1e-2}   
                    )   

        self.OC_hyperparams = dict({"Radius": 1, "nu": 1e-2})
#----------------------------------------------------------------------------

_feature_detector_cache = dict()
_feature_detector_3d_cache = dict()

class ResNet3DEmbedder(nn.Module):
    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.device = device
        self._feature_output = None
        self.model = self._load_model(checkpoint_path)
        self._register_hook()

    def _load_model(self, checkpoint_path):
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        model = resnet3d.resnet50(
            shortcut_type='B',
            no_cuda=not use_cuda)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model.to(self.device).eval()

    def _register_hook(self):
        def hook_fn(_, __, output):
            self._feature_output = output.detach()

        # Register hook on layer4 only once
        self.model.layer4.register_forward_hook(hook_fn)

    def forward(self, x):
        self._feature_output = None
        with torch.no_grad():
            _ = self.model(x)

        if self._feature_output is None:
            raise RuntimeError("Feature hook did not capture output from layer4.")

        # Global Average Pooling over 3D spatial dimensions
        embedding = self._feature_output.mean(dim=(-1, -2, -3))
        return embedding

def download_pretrained_model(url, destination_path):
    """Downloads a pre-trained model from a URL if it doesn't exist locally."""
    if not os.path.exists(destination_path):
        print(f"Downloading pre-trained model from: {url} to {destination_path}")
        try:
            state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location='cpu')
            # Save the downloaded state_dict
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            torch.save({'state_dict': state_dict}, destination_path)
            print("Pre-trained model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading pre-trained model: {e}")
            raise
    else:
        print(f"Pre-trained model found at: {destination_path}")

# def get_feature_detector_name(url):
#     return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector_name(url):
    """
    Function added to manage the different types of detectors:
    - "url" is a string with the path to the detector (NVIDIA pretrained models)
    - "url" is a dictionary with the model name (to exploit tf pretrained models)
    """
    if type(url)== tuple:
        detector_name = os.path.splitext(url[0].split('/')[-1])[0]
    elif type(url)==dict:
        detector_name = url['model']
    return detector_name

    
def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank <= num_gpus
    key = (url[0], device)
    if type(url)== tuple and url[1] == '3d':
        if key not in _feature_detector_3d_cache:
            is_leader = (rank == 0)
            if not is_leader and num_gpus > 1:
                torch.distributed.barrier() # leader goes first

            # Define a local path for the downloaded checkpoint
            pretrained_dir = dnnlib.make_cache_dir_path('pretrained_models')
            os.makedirs(pretrained_dir, exist_ok=True)
            filename = os.path.basename(url[0].split('?')[0])
            checkpoint_path = os.path.join(pretrained_dir, filename)
            download_pretrained_model(url[0], checkpoint_path)
            model = ResNet3DEmbedder(checkpoint_path, device).eval().to(device)
            _feature_detector_3d_cache[key] = model
            if is_leader and num_gpus > 1:
                torch.distributed.barrier() # others follow
        return _feature_detector_3d_cache[key]        
    else:
        if key not in _feature_detector_cache:
            is_leader = (rank == 0)
            if not is_leader and num_gpus > 1:
                torch.distributed.barrier() # leader goes first
            with dnnlib.util.open_url(url[0], verbose=(verbose and is_leader)) as f:
                _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
            if is_leader and num_gpus > 1:
                torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank <= num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().detach().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

# --------------------------------------------------------------------------------------

def validate_config(config):
    """Validates the configuration parameters in config.py."""

    errors = []

    def check_required_keys(dictionary, required_keys, name):
        for key in required_keys:
            if key not in dictionary:
                errors.append(f"Missing key in {name}: {key}")

    def validate_metrics(metrics):
        if not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics):
            errors.append("METRICS must be a list of metric names (strings).")
        else:
            valid_metrics = metric_main.list_valid_metrics()
            invalid_metrics = [m for m in metrics if m not in valid_metrics]
            if invalid_metrics:
                errors.append(f"Invalid metric(s) found: {invalid_metrics}. Allowed options: {valid_metrics}")

    def validate_metrics_configs(metrics_configs):
        # Validate 'nhood_size'
        if "nhood_size" not in metrics_configs:
            errors.append("Missing 'nhood_size' in METRICS_CONFIGS.")
        else:
            nhood_size = metrics_configs["nhood_size"]
            required_keys = ["prdc"]
            for key in required_keys:
                if key not in nhood_size:
                    errors.append(f"Missing key in METRICS_CONFIGS['nhood_size']: {key}")
                elif not isinstance(nhood_size[key], int) or nhood_size[key] <= 0:
                    errors.append(f"METRICS_CONFIGS['nhood_size']['{key}'] must be a positive integer.")

        # Validate 'K-NN_configs'
        if "K-NN_configs" not in metrics_configs:
            errors.append("Missing 'K-NN_configs' in METRICS_CONFIGS.")
        else:
            knn_configs = metrics_configs["K-NN_configs"]
            required_keys = ["num_real", "num_synth"]
            for key in required_keys:
                if key not in knn_configs:
                    errors.append(f"Missing key in METRICS_CONFIGS['K-NN_configs']: {key}")
                elif not isinstance(knn_configs[key], int) or knn_configs[key] <= 0:
                    errors.append(f"METRICS_CONFIGS['K-NN_configs']['{key}'] must be a positive integer.")

    def validate_configs(configs):
        required_keys = ["RUN_DIR", "DATA_TYPE", "USE_CACHE", "NUM_GPUS", "VERBOSE", "OC_DETECTOR_PATH"]
        check_required_keys(configs, required_keys, "CONFIGS")

        if not isinstance(configs["RUN_DIR"], (str, Path)):
            errors.append("RUN_DIR must be a string or Path object.")

        allowed_values = ['2d', '2D', '3d', '3D']
        if not isinstance(configs["DATA_TYPE"], str):
            errors.append("DATA_TYPE must be a string.")
        elif not configs["DATA_TYPE"] in allowed_values:
            errors.append(f"DATA_TYPE allowed values: {allowed_values}")

        if not isinstance(configs["USE_CACHE"], bool):
            errors.append("USE_CACHE must be a boolean (True/False).")

        if not isinstance(configs["NUM_GPUS"], int) or configs["NUM_GPUS"] < 0:
            errors.append("NUM_GPUS must be an integer greater than or equal to 0.")

        if not isinstance(configs["VERBOSE"], bool):
            errors.append("VERBOSE must be a boolean (True/False).")

    def validate_dataset(dataset):
        if not isinstance(dataset, dict):
            errors.append("DATASET must be a dictionary.")
        else:
            required_keys = ["class", "params"]
            check_required_keys(dataset, required_keys, "DATASET")

            if not inspect.isclass(dataset["class"]):
                errors.append("DATASET 'class' must be a class.")

            if not isinstance(dataset["params"], dict):
                errors.append("DATASET params must be a dictionary.")

            path_data = dataset["params"].get("path_data")
            if path_data and not Path(path_data).exists():
                errors.append(f"Dataset file not found: {path_data}")

    def validate_synthetic_data_config(synthetic_data_config, use_pretrained_model):

        # Validate pretrained_model configuration
        if use_pretrained_model:
            pretrained_model_config = synthetic_data_config.get("pretrained_model", {})
            required_keys = ["network_path", "load_network", "run_generator", "NUM_SYNTH"]
            for key in required_keys:
                if key not in pretrained_model_config:
                    errors.append(f"Missing key in pretrained_model config: {key}")

            if not Path(pretrained_model_config["network_path"]).exists():
                errors.append(f"Pre-trained generator file not found: {pretrained_model_config['network_path']}")

            if not callable(pretrained_model_config["load_network"]):
                errors.append("load_network must be a callable function.")

            if not callable(pretrained_model_config["run_generator"]):
                errors.append("run_generator must be a callable function.")

            if not isinstance(pretrained_model_config["NUM_SYNTH"], int) or pretrained_model_config["NUM_SYNTH"] <= 0:
                errors.append("NUM_SYNTH must be a positive integer.")

        # Validate from_files configuration
        else:
            from_files_config = synthetic_data_config.get("from_files", {})
            required_keys = ["class", "params"]
            for key in required_keys:
                if key not in from_files_config:
                    errors.append(f"Missing key in from_files config: {key}")

            if not inspect.isclass(from_files_config["class"]):
                errors.append("from_files 'class' must be a class.")

            params = from_files_config.get("params", {})
            required_params_keys = ["path_data", "path_labels", "use_labels", "size_dataset"]
            for key in required_params_keys:
                if key not in params:
                    errors.append(f"Missing key in from_files params: {key}")

            if not Path(params["path_data"]).exists():
                errors.append(f"Synthetic images file not found: {params['path_data']}")

            if params["path_labels"] is not None and not Path(params["path_labels"]).exists():
                errors.append(f"Labels file not found: {params['path_labels']}")

            if not isinstance(params["use_labels"], bool):
                errors.append("use_labels must be a boolean.")

            if params["size_dataset"] is not None and (not isinstance(params["size_dataset"], int) or params["size_dataset"] <= 0):
                errors.append("size_dataset must be a positive integer or None.")

    def validate_is(config):
        if config.CONFIGS["DATA_TYPE"].lower() == '3d':
            if 'is_' in config.METRICS:
                config.METRICS.remove('is_')
                print("Warning: Inception Score cannot be computed for 3D data because the ResNet-3D used as feature extractor lacks a classification layer.")
    # Validate each section
    validate_metrics(config.METRICS)
    validate_configs(config.CONFIGS)
    validate_is(config)
    validate_metrics_configs(config.METRICS_CONFIGS)
    validate_dataset(config.DATASET)
    validate_synthetic_data_config(config.SYNTHETIC_DATA, config.USE_PRETRAINED_MODEL)

    # Print errors if any
    if errors:
        raise ValueError(f"{len(errors)} error(s) in the configuration file:\n- "+"\n- ".join(errors))


#----------------------------------------------------------------------------

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_unique_filename(base_figname):
    """
    Check if a file already exists. If so, add a suffix to create a unique filename.
    """
    if not os.path.exists(base_figname):
        return base_figname
    
    filename, ext = os.path.splitext(base_figname)
    counter = 1
    
    while os.path.exists(f"{filename}_{counter}{ext}"):
        counter += 1
    
    return f"{filename}_{counter}{ext}"

def get_latest_figure(file_path):
    """
    Find the figure most recent in the folder, based on the greater suffix
    """

    directory, filename = os.path.split(file_path)
    basename, ext = os.path.splitext(filename)
    
    pattern = re.compile(rf"^{re.escape(basename)}(?:_(\d+))?{re.escape(ext)}$")
    
    max_index = -1
    latest_file = None
    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            # No number means index 0
            index = int(match.group(1)) if match.group(1) is not None else 0
            if index > max_index:
                max_index = index
                latest_file = f

    if latest_file:
        return os.path.join(directory, latest_file)
    else:
        return None

def pad_image(batch, output_shape, input_shape='nhwc'):
    # Calculate the amount of padding needed
    pad_height = output_shape[1] - batch.shape[1]
    pad_width = output_shape[2] - batch.shape[2]

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Zero-padding to obtain an array of the required shape
    if input_shape == 'nhwc':
        padded_batch = np.pad(batch, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
    elif input_shape == 'chw':
        padded_batch = np.pad(batch, pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return padded_batch

def setup_grid_slices(images_3d, grid_size, drange, line_thickness=2):
    n, c, d, h, w = images_3d.shape
    assert c == 1, "Only single-channel 3D volumes supported"
    lo, hi = drange
    images_3d = np.asarray(images_3d, dtype=np.float32)
    images_3d = (images_3d - lo) * (255 / (hi - lo))
    images_3d = np.rint(images_3d).clip(0, 255).astype(np.uint8)
    assert images_3d.min() >= 0 and images_3d.max() <= 255

    gw, gh = grid_size
    third = max(1, int(gh / 3))  # vertical thirds

    slices = []

    for i in range(gh * gw):
        image_3d = images_3d[i % n, 0]  # cycle through volumes if needed
        row = i // gw

        if row < third:
            slice_img = image_3d[d // 2, :, :]      # axial: [H, W]
        elif row < 2 * third:
            slice_img = image_3d[:, h // 2, :]      # coronal: [D, W]
        else:
            slice_img = image_3d[:, :, w // 2]      # sagittal: [D, H]

        slice_img = np.expand_dims(slice_img, axis=0)  # shape: [1, H, W]
        slices.append(slice_img)

    # Pad all slices to same shape
    max_h = max(img.shape[1] for img in slices)
    max_w = max(img.shape[2] for img in slices)
    target_shape = (1, max_h, max_w)
    slices = [pad_image(img, target_shape, input_shape='chw') for img in slices]

    # Overwrite white horizontal lines at third boundaries
    for row in [third, 2 * third]:
        if row >= gh:
            continue
        for col in range(gw):
            idx = row * gw + col
            if idx < len(slices):
                img = slices[idx]
                img[:, :line_thickness, :] = 255  # top few rows white
                slices[idx] = img

    # Final check
    assert len(slices) == gh * gw, f"Expected {gh*gw} slices, got {len(slices)}"
    return np.stack(slices, axis=0)  # [N, 1, H, W]

def setup_snapshot_image_grid(args, dataset, random_seed=0):
    rnd = np.random.RandomState(random_seed)

    n = 1 if args.data_type=="2d" or args.data_type=="2D" else 3
    grid_size = min(int(np.round(np.sqrt(len(dataset)*n))), 32)
    gw = np.clip(7680 // dataset._raw_shape[2], 7, grid_size)
    gh = np.clip(4320 // dataset._raw_shape[3], 4, grid_size)

    # No labels => show random subset of training samples.
    if not dataset._use_labels:
        all_indices = list(range(len(dataset)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(dataset)):
            label = tuple(dataset.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    images, labels = zip(*[dataset[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

def setup_grid_generated(args, G, labels, grid_size, num_images, real_dataset, device):
    # Latent vectors
    grid_z = torch.randn([labels.shape[0], *(G.z_dim if isinstance(G.z_dim, (list, tuple)) else [G.z_dim])], device=device)
    grid_c = torch.from_numpy(labels).to(device) if real_dataset._use_labels else None

    # Batching
    num_batches = (num_images + args.batch_size - 1) // args.batch_size
    print(f"Generating a grid of {grid_size[0]} x {grid_size[1]} synthetic images...")
    
    image_tensors = []
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start = i * args.batch_size
            end = min(start + args.batch_size, num_images)

            z_batch = grid_z[start:end]
            if grid_c is not None:
                c_batch = grid_c[start:end]
                out = args.run_generator(z_batch, c_batch, args)
            else:
                out = args.run_generator(z_batch, args)

            image_tensors.append(out)

    # Combine
    images_synt = torch.cat(image_tensors, dim=0).cpu().numpy()
    return images_synt

#----------------------------------------------------------------------------

def plot_image_grid(args, img, drange, grid_size, group, rank=0, verbose=True):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    fig_dir = os.path.join(args.run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    save_path_base = os.path.join(fig_dir, f"samples_{group}.png")
    save_path = get_unique_filename(save_path_base)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(save_path)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(save_path)
    if rank == 0 and verbose:
        print(f"Saved grid of {group} samples in {save_path}")

def reset_weights(model):
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
    for k, initializer in layer.__dict__.items():
        if "initializer" not in k:
            continue
      # find the corresponding variable
        var = getattr(layer, k.replace("_initializer", ""))
        var.assign(initializer(var.shape, var.dtype))
    return model

def remove_layer(model):
        new_input = model.input
        hidden_layer = model.layers[-2].output
        return Model(new_input, hidden_layer)   

def load_embedder(embedding):
    """
    Load embedder to compute density and coverage metrics
    """
    if embedding['model'] == 'vgg16' or embedding['model'] == 'vgg':
        model = tf.keras.applications.VGG16(include_top = True, weights='imagenet')
        model = remove_layer(model)
        model.trainable = False
          
    elif embedding['model'] == 'inceptionv3' or embedding['model'] == 'inception':
        model = tf.keras.applications.InceptionV3(include_top = True, weights='imagenet')
        model = remove_layer(model)
        model.trainable = False
    
    if embedding['randomise']:
        model = reset_weights(model)
        if embedding['dim64']:
            # removes last layer and inserts 64 output
            model = remove_layer(model)
            new_input = model.input
            hidden_layer = tf.keras.layers.Dense(64)(model.layers[-2].output)
            model = Model(new_input, hidden_layer)   
    model.run_eagerly = True
    return model
 
def adjust_size_embedder(opts, embedder, embedding, batch):
    
    if embedding['model'] == 'vgg16' or embedding['model'] == 'vgg':
        input_shape = 224
    elif embedder.input.shape[2]==299:
        input_shape = 299

    # Adjust input shape from [N, C, H, W] to [N, W, H, C]
    n,c,h,w = batch.shape
    batch = batch.permute(0, 2, 3, 1)
    if c==1:
        batch = batch.repeat(1, 1, 1, 3)

    # Desired output shape
    output_shape = (None, input_shape, input_shape, 3)

    if opts.padding and (h<input_shape or w<input_shape):
        batch = pad_image(batch.cpu(), output_shape)
    else:
        # Resize the image
        resized_batch = []
        for img in batch:
            pil_img = Image.fromarray((img.cpu().numpy()).astype(np.uint8))
            resized_img = pil_img.resize((input_shape, input_shape), resample=Image.BICUBIC) 
            resized_batch.append(np.array(resized_img))
        batch = np.array(resized_batch)

    return batch

def extract_features_from_detector(opts, images, detector, detector_url, detector_kwargs):
    if type(detector_url)==dict:
        images = adjust_size_embedder(opts, detector, detector_url, images)
        features = detector(images)                                   # tf.EagerTensor
        features = torch.from_numpy(features.numpy()).to(opts.device) # torch.Tensor
    elif type(detector_url)==tuple and detector_url[1]=='2d':
        features = detector(images.to(opts.device), **detector_kwargs)
    elif type(detector_url)==tuple and detector_url[1]=='3d':
        features = detector(images.to(opts.device))
    return features

def define_detector(opts, detector_url, progress):
    if type(detector_url)==dict:
        detector = load_embedder(detector_url)
    else:
        detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    return detector

def get_OC_model(opts, X=None, OC_params=None, OC_hyperparams=None):

    if opts.train_OC or not os.path.exists(opts.oc_detector_path):
        
        OC_params['input_dim'] = X.shape[1]

        if OC_params['rep_dim'] is None:
            OC_params['rep_dim'] = X.shape[1]
        # Check center definition !
        OC_hyperparams['center'] = torch.ones(OC_params['rep_dim'])*10
        
        OC_model = OneClassLayer(params=OC_params, 
                                 hyperparams=OC_hyperparams,
                                 seed=opts.seed)
        OC_model.fit(X,verbosity=True)
        
        # Check that the folder exists
        if not os.path.exists(os.path.dirname(opts.oc_detector_path)):
            os.makedirs(os.path.dirname(opts.oc_detector_path))

        # Save the OC model
        pickle.dump((OC_model, OC_params, OC_hyperparams),open(opts.oc_detector_path,'wb'))
    
    else:
        OC_model, OC_params, OC_hyperparams = pickle.load(open(opts.oc_detector_path,'rb'))
    
    OC_model.to(opts.device)
    if opts.rank == 0:
        print(OC_params)
        print(OC_hyperparams)
    OC_model.eval()
    return OC_model, OC_params, OC_hyperparams

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, dataset, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, dataset_kwargs=None, data_loader_kwargs=None, max_items=None, return_imgs=False, item_subset=None, **stats_kwargs):
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2) if platform.system() != 'Windows' else dict(pin_memory=True, num_workers=0)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag and not return_imgs:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    print('Extracting features from dataset...')
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = define_detector(opts, detector_url, progress)

    # Main loop.
    if item_subset is None:
        if opts.num_gpus>0:
            item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
        else:
            item_subset = list(range(num_items))
        
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=opts.batch_size, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(opts.seed), **data_loader_kwargs):
        if images.shape[1] == 1 and opts.data_type in ['2d', '2D']:
            images = images.repeat([1, 3, 1, 1])
        features = extract_features_from_detector(opts, images, detector, detector_url, detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    if return_imgs:
        return stats, images[:,0,:,:]
    else:
        return stats
#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_gen=None, jit=False, return_imgs=False, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(opts.batch_size, 4)
    assert opts.batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # JIT.
    if jit:
        z = torch.zeros([batch_gen, *(G.z_dim if isinstance(G.z_dim, (list, tuple)) else [G.z_dim])], device=opts.device)
        if dataset._use_labels:
            c = torch.zeros([batch_gen, *G.c_dim], device=opts.device)
            run_generator = torch.jit.trace(opts.run_generator, [z, c, opts], check_trace=False)
        else:
            run_generator = torch.jit.trace(opts.run_generator, [z, opts], check_trace=False)
        
    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = define_detector(opts, detector_url, progress)

    # Main loop.
    while not stats.is_full():
        images = []
        for _i in tqdm(range(opts.batch_size // batch_gen), desc="Generating images", unit="batch"):
            z = torch.randn([batch_gen, *(G.z_dim if isinstance(G.z_dim, (list, tuple)) else [G.z_dim])], device=opts.device)
            if dataset._use_labels:
                c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
                images.append(opts.run_generator(z, c, opts))
            else:
                images.append(opts.run_generator(z, opts))
        images = torch.cat(images).to(opts.device)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = extract_features_from_detector(opts, images, detector, detector_url, detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    if return_imgs:
        return stats, images[:,0,:,:]
    else:
        return stats

#----------------------------------------------------------------------------

def compute_feature_stats_synthetic(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, **stats_kwargs):
    if opts.use_pretrained_generator:
        gen_features = compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs, 
            rel_lo=rel_lo, rel_hi=rel_hi, **stats_kwargs)
    else:
        gen_features = compute_feature_stats_for_dataset(
            opts=opts, dataset=dnnlib.util.construct_class_by_name(**opts.dataset_synt_kwargs),
            detector_url=detector_url, detector_kwargs=detector_kwargs, 
            rel_lo=rel_lo, rel_hi=rel_hi, **stats_kwargs)
        
    return gen_features

#----------------------------------------------------------------------------
# Functions for k-NN analysis visualization

def visualize_grid(opts, real_images, synthetic_images, top_n_real_indices, closest_synthetic_indices, fig_path, top_n, k):
    fig, axes = plt.subplots(top_n, k+1, figsize=(5 * k, 5 * top_n))
    base_fontsize = max(35, 30 - k) 

    for row_idx in range(top_n):
        # Show the real image in the first column
        image = real_images[row_idx][0,:,:].cpu()
        axes[row_idx, 0].imshow(image, cmap='gray')
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(f"Real Image", fontsize=base_fontsize)
        
         # Add index annotation below the real image
        axes[row_idx, 0].text(
            0.5, -0.1, str(top_n_real_indices[row_idx]),
            fontsize=base_fontsize - 10, color='black', ha='center', va='bottom',
            transform=axes[row_idx, 0].transAxes
        )
               
        # Show the top k synthetic images in the next columns
        img_gray = synthetic_images[0][0].shape[0] != 3
        for col_idx in range(k):
            if img_gray:
                image = synthetic_images[row_idx][col_idx][:,:]
            else:
                image = synthetic_images[row_idx][col_idx][0,:,:]
            axes[row_idx, col_idx+1].imshow(image, cmap='gray')
            axes[row_idx, col_idx+1].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx+1].set_title(f"Synth {col_idx+1}", fontsize=base_fontsize)

            if not opts.use_pretrained_generator:
                # Add annotation of the index of the synthetic image
                axes[row_idx, col_idx+1].text(
                    0.5, -0.1, str(closest_synthetic_indices[top_n_real_indices[row_idx]][col_idx]),
                    fontsize=base_fontsize - 10, color='black', ha='center', va='bottom',
                    transform=axes[row_idx, col_idx+1].transAxes
                )

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def extract_slices(volume):
    """Extract axial, coronal, sagittal central slices from 3D volume (C, D, H, W)."""
    d, h, w = volume.shape
    return [
        volume[d // 2, :, :],     # Axial (XY)
        volume[:, h // 2, :],     # Coronal (XZ)
        volume[:, :, w // 2],     # Sagittal (YZ)
    ]

def visualize_grid_3d(opts, real_volumes, synthetic_volumes, top_n_real_indices, closest_synthetic_indices, fig_path, top_n, k):
    n_slices = 3  # Axial, Coronal, Sagittal
    total_cols = k + 1
    base_fontsize = max(20, 30 - k)

    fig = plt.figure(figsize=(3 * total_cols, 3 * top_n * n_slices))
    outer = gridspec.GridSpec(top_n, total_cols, wspace=0.2, hspace=0.05)
    
    for row_idx in range(top_n):
        real_volume = real_volumes[row_idx][0].cpu()
        real_slices = extract_slices(real_volume)  # List of 3 slices: axial, coronal, sagittal

        for col_idx in range(total_cols):
            # Determine whether it's real or synthetic
            if col_idx == 0:
                volume = real_volume
                slices = real_slices
                is_real = True
                title = f"Real {top_n_real_indices[row_idx]}"
            else:
                volume = synthetic_volumes[row_idx][col_idx - 1]
                slices = extract_slices(volume)
                is_real = False
                title = f"Synth {col_idx}"

            inner = gridspec.GridSpecFromSubplotSpec(n_slices, 1, subplot_spec=outer[row_idx, col_idx], hspace=0)
            for slice_idx in range(n_slices):
                ax = plt.Subplot(fig, inner[slice_idx])
                ax.imshow(slices[slice_idx], cmap='gray')
                ax.axis('off')

                # Only label the top slice of each volume
                if slice_idx == 0:
                    ax.set_title(title, fontsize=base_fontsize)

                # Add real/synthetic index below the bottom slice
                if slice_idx == 2:
                    if is_real:
                        # Real index below real volume
                        ax.text(
                            0.5, -0.2,
                            str(top_n_real_indices[row_idx]),
                            fontsize=base_fontsize - 8,
                            color='black',
                            ha='center', va='bottom',
                            transform=ax.transAxes
                        )
                    elif not opts.use_pretrained_generator:
                        # Synthetic index below synthetic volume
                        ax.text(
                            0.5, -0.2,
                            str(closest_synthetic_indices[top_n_real_indices[row_idx]][col_idx - 1]),
                            fontsize=base_fontsize - 8,
                            color='black',
                            ha='center', va='bottom',
                            transform=ax.transAxes
                        )

                fig.add_subplot(ax)

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def select_top_n_real_images(closest_similarities, top_n=6):
    """
    Select the top-n real images based on the highest similarity.
    """
    # Compute the highest similarity for each real image (first column of closest_similarities)
    max_similarities = {idx: similarities[0] for idx, similarities in closest_similarities.items()}

    # Sort real images by highest similarity and select top-n
    sorted_real_indices = sorted(max_similarities, key=max_similarities.get, reverse=True)[:top_n]
    
    return sorted_real_indices


def visualize_top_k(opts, closest_images, closest_indices, top_n_real_indices, fig_path, top_n=6, k=8):
    """
    Visualize the top-k closest synthetic images for the selected real images.
    """
    # Create a dataset and DataLoader for the real images
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Use the indices of the closest synthetic images to load the real images from the dataset
    real_images, _ = next(iter(torch.utils.data.DataLoader(dataset=dataset, sampler=top_n_real_indices, batch_size=opts.batch_size, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(opts.seed))))

    # Collect the synthetic images corresponding to each real image from closest_images
    synthetic_images_to_visualize = [closest_images[real_idx][:k] for real_idx in top_n_real_indices]

    # Now visualize the real image and the k closest synthetic images
    if opts.data_type.lower() == '2d':
        visualize_grid(opts, real_images, synthetic_images_to_visualize, top_n_real_indices, closest_indices, fig_path, top_n, k)
    elif opts.data_type.lower() == '3d':
        visualize_grid_3d(opts, real_images, synthetic_images_to_visualize, top_n_real_indices, closest_indices, fig_path, top_n, k)

#----------------------------------------------------------------------------
# Functions for qualitative assessment

def project_gaussian(mu, sigma, pca):
    """Project high-dim mean and covariance through PCA to 2D."""
    mu_2d = pca.transform(mu.reshape(1, -1))[0]
    sigma_2d = pca.components_ @ sigma @ pca.components_.T
    return mu_2d, sigma_2d

def plot_gaussian_ellipse(ax, mu, sigma, n_std=2, edgecolor='black', facecolor='none', label=None, alpha=0.2):
    """Draw a covariance ellipse given 2D mean and covariance."""
    from scipy.linalg import eigh

    # Compute eigenvalues and eigenvectors
    vals, vecs = eigh(sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta,
                    edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, linewidth=2, label=label)
    ax.add_patch(ellipse)


def plot_pca(metric, real_features, gen_features, mu_real=None, sigma_real=None, mu_gen=None, sigma_gen=None, circle_info=None, fig_path=None):
    # --- PCA to 2D ---
    all_features = np.vstack([real_features, gen_features]) if real_features is not None else gen_features
    pca = PCA(n_components=2)
    pca.fit(all_features)

    if real_features is not None:
        real_2d = pca.transform(real_features)
    gen_2d = pca.transform(gen_features)

    if all(x is not None for x in [mu_real, sigma_real, mu_gen, sigma_gen]):
        if real_features is not None:
            mu_real_2d, sigma_real_2d = project_gaussian(mu_real, sigma_real, pca)
        mu_gen_2d, sigma_gen_2d = project_gaussian(mu_gen, sigma_gen, pca)

    explained_var = pca.explained_variance_ratio_.sum() * 100

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plots
    if real_features is not None:
        ax.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.5, s=50, c='#1f77b4', label='Real')
    ax.scatter(gen_2d[:, 0], gen_2d[:, 1], alpha=0.5, s=50, c='#ff7f0e', label='Synthetic')

    # Gaussian ellipses
    if all(x is not None for x in [mu_real, sigma_real, mu_gen, sigma_gen]):
        plot_gaussian_ellipse(ax, mu_real_2d, sigma_real_2d,
                            edgecolor='#1f77b4', facecolor='#1f77b4', alpha=0.2, label='Real Gaussian')
        plot_gaussian_ellipse(ax, mu_gen_2d, sigma_gen_2d,
                            edgecolor='#ff7f0e', facecolor='#ff7f0e', alpha=0.2, label='Synthetic Gaussian')

    if circle_info is not None:
        center, Radii = circle_info
        center_2d = pca.transform(center.reshape(1, -1))[0]
        from matplotlib.patches import Circle
        circle = Circle(center_2d, Radii, color='black', fill=False)
        ax.add_patch(circle)
        ax.set_xlim(min(center_2d[0],real_2d[:,0].min(),gen_2d[:,0].min()) - Radii - .025, max(center_2d[0],real_2d[:,0].max(),gen_2d[:,0].max()) + Radii + .025)
        ax.set_ylim(min(center_2d[1],real_2d[:,1].min(),gen_2d[:,1].min()) - Radii - .025,max(center_2d[1],real_2d[:,1].max(),gen_2d[:,1].max()) + Radii + .025)

    # Final touches
    ax.set_title(f"PCA of the embeddings for {metric}\n(Explained variance: {explained_var:.2f}%)", fontsize=22)
    ax.set_xlabel("PCA Component 1", fontsize=20)
    ax.set_ylabel("PCA Component 2", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_tsne(metric, real_features, gen_features, fig_path):
    all_embeddings = np.vstack([real_features, gen_features]) if real_features is not None else gen_features

    # Create labels: 0 = real, 1 = synthetic
    labels = np.array(['Real'] * len(real_features) + ['Synthetic'] * len(gen_features)) if real_features is not None else np.array(['Synthetic'] * len(gen_features))

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(all_embeddings)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    palette = {"Real": "#1f76b4", "Synthetic": "#ff7f0e"}

    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels, palette=palette, alpha=0.6,
        s=60, edgecolor='k'
    )
    plt.title(f"t-SNE of the embeddings for {metric}", fontsize=22)
    plt.legend(fontsize=20)
    plt.xlabel("t-SNE Component 1", fontsize=20)
    plt.ylabel("t-SNE Component 2", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()