﻿# SPDX-License-Identifier: LicenseRef-NVIDIA-1.0
# 
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Precision/Recall (PR) from the paper "Improved Precision and Recall
Metric for Assessing Generative Models". Matches the original implementation
by Kynkaanniemi et al. at
https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py"""

import torch
import dnnlib
from . import metric_utils

#----------------------------------------------------------------------------

def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

#----------------------------------------------------------------------------

def compute_pr(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):
    if opts.data_type.lower() == '2d':
        detector_url = ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt', '2d')
    elif opts.data_type.lower() == '3d':
        detector_url = ('https://zenodo.org/records/15234379/files/resnet_50_23dataset_cpu.pth?download=1', '3d')
    detector_kwargs = dict(return_features=True)

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, dataset=dnnlib.util.construct_class_by_name(**opts.dataset_kwargs), 
        detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, dataset_kwargs=opts.dataset_kwargs, capture_all=True, max_items=max_real).get_all_torch().to(torch.float32).to(opts.device)

    gen_features = metric_utils.compute_feature_stats_synthetic(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_torch().to(torch.float32).to(opts.device)

    # Visualize t-SNE
    fig_path = opts.run_dir + '/figures/tsne_pr.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_tsne('Precision & Recall', real_features=real_features.cpu(), gen_features=gen_features.cpu(), fig_path=fig_path)

    # Visualize PCA
    fig_path = opts.run_dir + '/figures/pca_pr.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_pca('Precision & Recall', real_features=real_features.cpu(), gen_features=gen_features.cpu(), fig_path=fig_path)
    
    # Compute P&R
    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if opts.rank == 0 else None)
        kth = torch.cat(kth) if opts.rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else 'nan')
    return results['precision'], results['recall']

#----------------------------------------------------------------------------
