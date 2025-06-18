# SPDX-License-Identifier: LicenseRef-NVIDIA-1.0
# 
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils
import dnnlib
import torch

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    if opts.data_type.lower() == '2d':
        detector_url = ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt', '2d')
    elif opts.data_type.lower() == '3d':
        detector_url = ('https://zenodo.org/records/15234379/files/resnet_50_23dataset_cpu.pth?download=1', '3d')
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    real_feature_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, dataset=dnnlib.util.construct_class_by_name(**opts.dataset_kwargs),
        detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, dataset_kwargs=opts.dataset_kwargs, capture_all=True, capture_mean_cov=True, max_items=max_real)
    real_features = real_feature_stats.get_all_torch().to(torch.float32).to(opts.device)
    mu_real, sigma_real = real_feature_stats.get_mean_cov()

    gen_feature_stats = metric_utils.compute_feature_stats_synthetic(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, capture_mean_cov=True, max_items=num_gen)
    gen_features = gen_feature_stats.get_all_torch().to(torch.float32).to(opts.device)
    mu_gen, sigma_gen = gen_feature_stats.get_mean_cov()

    if opts.rank != 0:
        return float('nan')
    
    # Visualize t-SNE
    fig_path = opts.run_dir + '/figures/tsne_fid.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_tsne('FID', real_features=real_features.cpu(), gen_features=gen_features.cpu(), fig_path=fig_path)

    # Visualize PCA
    fig_path = opts.run_dir + '/figures/pca_fid.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_pca('FID', real_features=real_features.cpu(), gen_features=gen_features.cpu(), 
             mu_real=mu_real, sigma_real=sigma_real, mu_gen=mu_gen, sigma_gen=sigma_gen,
             fig_path=fig_path)
    
    # Compute FID
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
