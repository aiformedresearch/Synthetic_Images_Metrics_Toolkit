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

import legacy
import nibabel as nib
from representations.OneClass import OneClassLayer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend 
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, run_dir, run_generator, network_pkl, num_gen, oc_detector_path, train_OC, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.run_dir        = run_dir
        self.gen_path       = network_pkl
        self.data_path      = dataset_kwargs.path_data
        self.num_gen        = num_gen
        self.oc_detector_path = oc_detector_path
        self.train_OC       = train_OC
        self.run_generator  = run_generator

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
    assert 0 <= rank < num_gpus
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
        assert 0 <= rank < num_gpus
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

def visualize_ex_samples(args, device, rank=0, verbose=True):

    # Load a real image
    dataset = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    item_subset = [0]
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs)
    real_image, label = next(iter(dataloader))
    real_sample = real_image[0,0,:,:]

    # Generate a synthetic image
    args.G.to(device)
    c = label.to(device)
    z = torch.from_numpy(np.random.RandomState(42).randn(1, *(args.G.z_dim if isinstance(args.G.z_dim, (list, tuple)) else [args.G.z_dim]))).to(device)
    if dataset._use_labels:
        img_synth = args.run_generator(z, c, args).to(device)
    else:
        img_synth = args.run_generator(z, args).to(device)
    synth_sample = img_synth[0,0,:,:].cpu().detach().numpy()

    # #POI RIMUOVI
    # real_sample = np.rot90(real_sample, k=3)
    # synth_sample = np.rot90(synth_sample, k=3)

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

def adjust_size_embedder(embedder, embedding, batch):
    if embedder.input.shape[2]==299:

        # Desired output shape
        output_shape = (None, 299, 299, 3)

        # Calculate the amount of padding needed
        pad_height = output_shape[1] - batch.shape[1]
        pad_width = output_shape[2] - batch.shape[2]

        # Calculate padding for each side
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the array to obtain a 299x299 array
        batch = np.pad(batch, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
    
    if embedding['model'] == 'vgg16' or embedding['model'] == 'vgg':
        if batch.shape[2]!=224:
            batch = tf.image.resize(batch, [224, 224])
    return batch

def get_activations_from_nifti(opts, synth_file, embedder, embedding, batch_size=None, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- synth_file      : a) path to the NIFTI file containing the images or
                       b) path to the generator model that generates the images
    -- embedding        : embedding type (Inception, VGG16, None)
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, embedding_size) that contains the
       activations of the given tensor when feeding e.g. inception with the query tensor.
    """
      
    # Check if synth_file is a .nii.gz format:
    if synth_file.endswith('.nii.gz'):
        # Load the NIFTI file
        img_nifti = nib.load(synth_file)
        img_data = img_nifti.get_fdata()

        # Transpose dimensions to match the expected input shape of the embedder
        img_data = np.transpose(img_data, (3, 0, 1, 2))  # Adjust dimensions to (num_images, height, width, channels)
        if embedder is not None:
            if embedder.input.shape[-1]==3:
                img_data = np.repeat(img_data, 3, axis=-1)

        print(f"Image data shape: {img_data.shape}")

        n_imgs = img_data.shape[0]
        print(f"Number of images: {n_imgs}")

    else:
        # Define the generator model
        G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

        n_imgs = opts.num_gen 
        print(f"Generating {n_imgs} synthetic images...")
        n_generated = 0
    
    if batch_size is None:
        batch_size = n_imgs
    elif batch_size > n_imgs:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_imgs

    n_batches = (n_imgs + batch_size - 1) // batch_size
    
    pred_arr = np.empty((n_imgs,embedder.output.shape[-1]))
    input_shape = embedder.input.shape[1]
    
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_imgs:
            end = start+batch_size
        else:
            end = n_imgs
        
        if synth_file.endswith('.nii.gz'):
            batch = img_data[start:end,:,:,:]
            batch = adjust_size_embedder(embedder, embedding, batch)
            batch_embedding = embedder(batch)
            # Convert to numpy array:
            pred_arr[start:end] = np.stack(list(batch_embedding))
            del batch #clean up memory

        else:
            n_to_generate = end-start
            z = torch.randn([n_to_generate, *(G.z_dim if isinstance(G.z_dim, (list, tuple)) else [G.z_dim])], device=opts.device)
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
            batch = batch.cpu().detach().numpy()
            n_generated += n_to_generate
            print(f'\rGenerated {n_generated}/{n_imgs} images')
            batch = np.transpose(batch, (0, 2, 3, 1))
            batch = np.repeat(batch, 3, axis=-1)
            batch = adjust_size_embedder(embedder, embedding, batch)

            batch_embedding = embedder(batch)
            # Convert to numpy array:
            pred_arr[start:end] = np.stack(list(batch_embedding))
            del batch #clean up memory
        
    if verbose:
        print(" done")
    
    return pred_arr

def get_activation(opts, path, embedding, embedder=None, verbose=True, save_act=False):
# Check if folder exists
    if not os.path.exists(path):
        raise RuntimeError("Invalid path: %s" % path)
    # Don't embed data if no embedding is given 
    if embedding is None:
        act = load_all_images_nifti(path)    

    else:
        act_filename = f'{opts.run_dir}/act_{embedding["model"]}_{embedding["dim64"]}_{embedding["randomise"]}'
        # Check if embeddings are already available
        if os.path.exists(f'{act_filename}.npz'):
            print('Loaded activations from', act_filename)
            print(act_filename)
            data = np.load(f'{act_filename}.npz',allow_pickle=True)
            act, _ = data['act'], data['embedding']
        # Otherwise compute embeddings
        else:
            # if load_act:
            #     print('Could not find activation file', act_filename)
            print('Calculating activations')
            act = get_activations_from_nifti(opts, path, embedder, embedding, batch_size=64*2, verbose=verbose)
            # Save embeddings
            if save_act:
                np.savez(f'{act_filename}', act=act,embedding=embedding)
    return act

def extract_features_from_detector(opts, images, detector, detector_url, detector_kwargs):
    if type(detector_url)==str and detector_url.startswith('https://') and detector_url.endswith('.pt'):
        features = detector(images.to(opts.device), **detector_kwargs)
    elif type(detector_url)==dict:
        if detector.input_shape[-1]==3:
            images = images.permute(0,2,3,1)
        if images.shape[1] != detector.input_shape[1]:
            res_shape = detector.input_shape[1]
            images = images.cpu()
            images = tf.image.resize(images, (res_shape, res_shape))

        features = detector(tf.convert_to_tensor(images))
        features = torch.from_numpy(features.numpy()).to(opts.device)
    return features

def define_detector(opts, detector_url, progress):
    if type(detector_url)==str and detector_url.startswith('https://') and detector_url.endswith('.pt'):
        detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    else:
        detector = load_embedder(detector_url)
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

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
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
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    print('Extracting features from real data...')
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = define_detector(opts, detector_url, progress)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
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
        images = torch.cat(images)
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
# Functions added for k-NN analysis visualization

def visualize_grid(real_images, synthetic_images, fig_path, top_n, k):
    fig, axes = plt.subplots(top_n, k+1, figsize=(5 * k, 5 * top_n))
    base_fontsize = max(35, 30 - k) 

    for row_idx in range(top_n):
        # Show the real image in the first column
        image = real_images[row_idx][0,:,:].cpu()
        axes[row_idx, 0].imshow(image, cmap='gray')
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(f"Real Image", fontsize=base_fontsize)
        
        
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


def visualize_top_k(opts, closest_images, top_n_real_indices, fig_path, batch_size, top_n=6, k=8):
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
    visualize_grid(real_images, synthetic_images_to_visualize, fig_path, top_n, k)
   