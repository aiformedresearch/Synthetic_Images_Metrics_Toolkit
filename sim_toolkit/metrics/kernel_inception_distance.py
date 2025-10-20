# SPDX-License-Identifier: LicenseRef-NVIDIA-1.0
# 
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Kernel Inception Distance (KID) from the paper "Demystifying MMD
GANs". Matches the original implementation by Binkowski et al. at
https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py"""

import numpy as np
from .. import dnnlib
from . import metric_utils

#----------------------------------------------------------------------------

def compute_kid(opts, max_real, num_gen, num_subsets, max_subset_size):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    if opts.data_type.lower() == '2d':
        detector_url = ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt', '2d')
    elif opts.data_type.lower() == '3d':
        detector_url = ('https://zenodo.org/records/15234379/files/resnet_50_23dataset_cpu.pth?download=1', '3d')
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, dataset=dnnlib.util.construct_class_by_name(**opts.dataset_kwargs), 
        detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, dataset_kwargs=opts.dataset_kwargs, capture_all=True, max_items=max_real).get_all()

    gen_features = metric_utils.compute_feature_stats_synthetic(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all()
   
    if opts.rank != 0:
        return float('nan')

    # Visualize t-SNE
    fig_path = opts.run_dir + '/figures/tsne_kid.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_tsne('KID', real_features=real_features, gen_features=gen_features, fig_path=fig_path)

    # Visualize PCA
    fig_path = opts.run_dir + '/figures/pca_kid.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_pca('KID', real_features=real_features, gen_features=gen_features, fig_path=fig_path)
    
    # Compute KID
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)

#----------------------------------------------------------------------------
