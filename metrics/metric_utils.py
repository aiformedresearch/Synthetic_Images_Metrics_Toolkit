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
import dnnlib

from representations.OneClass import OneClassLayer
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import re
from PIL import Image
from pathlib import Path
import inspect
from metrics import metric_main

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, run_dir, use_pretrained_generator, run_generator, network_pkl, num_gen, nhood_size, knn_config, padding, oc_detector_path, train_OC, G=None, G_kwargs={}, dataset_kwargs={}, dataset_synt_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True):
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

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

# def get_feature_detector_name(url):
#     return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector_name(url):
    """
    Function added to manage the different types of detectors:
    - "url" is a string with the path to the detector (NVIDIA pretrained models)
    - "url" is a dictionary with the model name (to exploit tf pretrained models)
    """
    if type(url)==str and url.startswith('https://') and url.endswith('.pt'):
        detector_name = os.path.splitext(url.split('/')[-1])[0]
    elif type(url)==dict:
        detector_name = url['model']
    return detector_name

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank <= num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
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
            required_keys = ["pr", "prdc", "pr_auth"]
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
        required_keys = ["RUN_DIR", "NUM_GPUS", "VERBOSE", "OC_DETECTOR_PATH"]
        check_required_keys(configs, required_keys, "CONFIGS")

        if not isinstance(configs["RUN_DIR"], (str, Path)):
            errors.append("RUN_DIR must be a string or Path object.")

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

        # Validate mode
        expected_mode = "pretrained_model" if use_pretrained_model else "from_files"
        if synthetic_data_config.get("mode") != expected_mode:
            errors.append(f"Mode mismatch: expected '{expected_mode}' but got '{synthetic_data_config.get('mode')}'.")

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

    # Validate each section
    validate_metrics(config.METRICS)
    validate_metrics_configs(config.METRICS_CONFIGS)
    validate_configs(config.CONFIGS)
    validate_dataset(config.DATASET)
    validate_synthetic_data_config(config.SYNTHETIC_DATA, config.USE_PRETRAINED_MODEL)

    # Print errors if any
    if errors:
        raise ValueError(f"{len(errors)} error(s) in the configuration file:\n- "+"\n- ".join(errors))


#----------------------------------------------------------------------------
# Functions added to manage the different types of detectors
#--------------------------------------------------------------------------------

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

def visualize_ex_samples(args, device, rank=0, verbose=True):

    def load_img(ds_kwargs):
        dataset = dnnlib.util.construct_class_by_name(**ds_kwargs)
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        item_subset = [0]
        batch_size = 1
        dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs)    
        image, label = next(iter(dataloader))
        sample = image[0,0,:,:]
        return sample, label, dataset._use_labels

    # Load a real image
    real_sample, label, use_labels = load_img(args.dataset_kwargs)

    if args.use_pretrained_generator:
        # Generate a synthetic image
        args.G.to(device)
        c = label.to(device)
        z = torch.from_numpy(np.random.RandomState(42).randn(1, *(args.G.z_dim if isinstance(args.G.z_dim, (list, tuple)) else [args.G.z_dim]))).to(device)
        if use_labels:
            img_synth = args.run_generator(z, c, args).to(device)
        else:
            img_synth = args.run_generator(z, args).to(device)
        synth_sample = img_synth[0,0,:,:].cpu().detach().numpy()
    else:
        # Load a synthetic image
        synth_sample, _, _ = load_img(args.dataset_synt_kwargs)

    # Plot the images in a 2x1 subplot
    fig, axs = plt.subplots(1, 2, figsize=(20, 12))

    axs[0].imshow(real_sample, cmap='gray')
    axs[0].set_title(f'Real Sample\n∈ [{real_sample.min()}, {real_sample.max()}]\ndtype: {real_sample.dtype}', fontsize=25)
    axs[0].axis('off')

    axs[1].imshow(synth_sample, cmap='gray')
    axs[1].set_title(f'Synthetic Sample\n∈ [{synth_sample.min()}, {synth_sample.max()}]\ndtype: {synth_sample.dtype}', fontsize=25)
    axs[1].axis('off')

    plt.tight_layout()
    fig_dir = os.path.join(args.run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    save_path_base = os.path.join(fig_dir, "samples_visualization.png")
    save_path = get_unique_filename(save_path_base)

    plt.savefig(save_path)
    if rank == 0 and verbose:
        print(f"Saved samples from real and synthetic datasets in {save_path}")

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
        # Calculate the amount of padding needed
        pad_height = output_shape[1] - batch.shape[1]
        pad_width = output_shape[2] - batch.shape[2]

        # Calculate padding for each side
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Zero-padding to obtain an array of the required shape
        batch = np.pad(batch.cpu(), ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
    else:
        # Resize the image
        resized_batch = []
        for img in batch:
            pil_img = Image.fromarray((img.cpu().numpy()).astype(np.uint8))
            resized_img = pil_img.resize((input_shape, input_shape), resample=Image.BICUBIC) 
            resized_batch.append(np.array(resized_img))
        batch = np.array(resized_batch)

    return batch

def get_activation_from_dataset(opts, dataset, max_img, embedding, embedder=None, verbose=True, save_act=False, batch_size=64*2):
    
    act_filename = f'{opts.run_dir}/act_{embedding["model"]}_{embedding["dim64"]}_{embedding["randomise"]}'
    # Check if embeddings are already available
    if os.path.exists(f'{act_filename}.npz'):
        if verbose:
            print('Loaded activations from', act_filename)
            print(act_filename)
        data = np.load(f'{act_filename}.npz',allow_pickle=True)
        act, _ = data['act'], data['embedding']
    # Otherwise compute embeddings
    else:
        if verbose:
            print('Calculating activations')
        n_imgs = min(len(dataset), max_img) if max_img is not None else len(dataset)
        if batch_size is None:
            batch_size = n_imgs
        elif batch_size > n_imgs:
            print("warning: batch size is bigger than the data size. setting batch size to data size")
            batch_size = n_imgs
        n_batches = (n_imgs + batch_size - 1) // batch_size
    
        pred_arr = np.empty((n_imgs,embedder.output.shape[-1]))
        if opts.num_gpus>0:
            item_subset = [(i * opts.num_gpus + opts.rank) % n_imgs for i in range((n_imgs - 1) // opts.num_gpus + 1)]
        else:
            item_subset = list(range(n_imgs))
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        i=0
        for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            if verbose:
                print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
            start = i*batch_size
            i+=1
            if start+batch_size < n_imgs:
                end = start+batch_size
            else:
                end = n_imgs

            batch = adjust_size_embedder(opts, embedder, embedding, images)

            batch_embedding = embedder(batch)

            # Convert to numpy array:
            pred_arr[start:end] = np.stack(list(batch_embedding))
            del batch #clean up memory        
        if verbose:
            print(" done")
        act = pred_arr
        # Save embeddings
        if save_act:
            np.savez(f'{act_filename}', act=act,embedding=embedding)
    return act

def get_activation_from_generator(opts, num_gen, embedding, embedder=None, verbose=True, batch_size=64*2):
    act_filename = f'{opts.run_dir}/act_{embedding["model"]}_{embedding["dim64"]}_{embedding["randomise"]}'
    # Check if embeddings are already available
    if os.path.exists(f'{act_filename}.npz'):
        if verbose:
            print('Loaded activations from', act_filename)
            print(act_filename)
        data = np.load(f'{act_filename}.npz',allow_pickle=True)
        act, _ = data['act'], data['embedding']
    # Otherwise compute embeddings
    else:
        if verbose:
            print('Calculating activations')

        # Define the generator model
        G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        
        n_imgs = opts.num_gen 
        print(f"Generating {n_imgs} synthetic images...")
        n_generated = 0

        n_imgs = num_gen
        if batch_size is None:
            batch_size = n_imgs
        elif batch_size > n_imgs:
            print("warning: batch size is bigger than the data size. setting batch size to data size")
            batch_size = n_imgs
        n_batches = (n_imgs + batch_size - 1) // batch_size

        pred_arr = np.empty((n_imgs,embedder.output.shape[-1]))

        for i in range(n_batches):
            if verbose:
                print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
            start = i*batch_size
            if start+batch_size < n_imgs:
                end = start+batch_size
            else:
                end = n_imgs
            n_to_generate = end-start
            z = torch.randn([n_to_generate, G.z_dim], device=opts.device)
            # define c as a vector with batch_size elements sampled from 0 and 1:
            half_n = n_to_generate // 2
            c = [torch.tensor([1, 0]) for _ in range(half_n)] + [torch.tensor([0, 1]) for _ in range(half_n)]       
            if n_to_generate % 2 != 0:
                c.append(torch.tensor([1, 0]) if torch.randint(0, 2, (1,)).item() == 1 else torch.tensor([0, 1]))
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            if dataset._use_labels:
                batch = opts.run_generator(z, c, opts).to(opts.device)
            else:
                batch = opts.run_generator(z, opts).to(opts.device)
            n_generated += n_to_generate
            print(f'\rGenerated {n_generated}/{n_imgs} images')
            batch = adjust_size_embedder(opts, embedder, embedding, batch)

            batch_embedding = embedder(batch)
            # Convert to numpy array:
            pred_arr[start:end] = np.stack(list(batch_embedding))
            del batch #clean up memory
    if verbose:
        print(" done")
    return pred_arr


def get_activation_synthetic(opts, num_gen, detector_url, embedder, verbose=True):
    if opts.use_pretrained_generator:
        gen_features = get_activation_from_generator(
            opts, num_gen, detector_url, embedder=embedder, verbose=verbose)
    else:
        gen_features = get_activation_from_dataset(
            opts, dataset=dnnlib.util.construct_class_by_name(**opts.dataset_synt_kwargs), 
                                max_img=opts.num_gen, embedding=detector_url, embedder=embedder, verbose=verbose)
        
    return gen_features

def extract_features_from_detector(opts, images, detector, detector_url, detector_kwargs):
    if type(detector_url)==dict:
        images = adjust_size_embedder(opts, detector, detector_url, images)
        features = detector(images)                                   # tf.EagerTensor
        features = torch.from_numpy(features.numpy()).to(opts.device) # torch.Tensor
    else:
        features = detector(images.to(opts.device), **detector_kwargs)
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
                                 hyperparams=OC_hyperparams)
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

def compute_feature_stats_for_dataset(opts, dataset, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, dataset_kwargs=None, data_loader_kwargs=None, max_items=None, return_imgs=False, item_subset=None, **stats_kwargs):
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

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
        
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
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

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, jit=False, return_imgs=False, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

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
        for _i in range(batch_size // batch_gen):
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
# Functions added for k-NN analysis visualization

def visualize_grid(real_images, synthetic_images, top_n_real_indices, closest_synthetic_indices, fig_path, top_n, k):
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

            # Add index annotation below the synthetic image
            axes[row_idx, col_idx+1].text(
                0.5, -0.1, str(closest_synthetic_indices[top_n_real_indices[row_idx]][col_idx]),
                fontsize=base_fontsize - 10, color='black', ha='center', va='bottom',
                transform=axes[row_idx, col_idx+1].transAxes
            )

    plt.tight_layout()
    plt.savefig(fig_path)

def select_top_n_real_images(closest_similarities, top_n=6):
    """
    Select the top-n real images based on the highest similarity.
    """
    # Compute the highest similarity for each real image (first column of closest_similarities)
    max_similarities = {idx: similarities[0] for idx, similarities in closest_similarities.items()}

    # Sort real images by highest similarity and select top-n
    sorted_real_indices = sorted(max_similarities, key=max_similarities.get, reverse=True)[:top_n]
    
    return sorted_real_indices


def visualize_top_k(opts, closest_images, closest_indices, top_n_real_indices, fig_path, batch_size, top_n=6, k=8):
    """
    Visualize the top-k closest synthetic images for the selected real images.
    """
    # Create a dataset and DataLoader for the real images
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Use the indices of the closest synthetic images to load the real images from the dataset
    real_images, _ = next(iter(torch.utils.data.DataLoader(dataset=dataset, sampler=top_n_real_indices, batch_size=batch_size)))

    # Collect the synthetic images corresponding to each real image from closest_images
    synthetic_images_to_visualize = [closest_images[real_idx][:k] for real_idx in top_n_real_indices]

    # Now visualize the real image and the k closest synthetic images
    visualize_grid(real_images, synthetic_images_to_visualize, top_n_real_indices, closest_indices, fig_path, top_n, k)
   