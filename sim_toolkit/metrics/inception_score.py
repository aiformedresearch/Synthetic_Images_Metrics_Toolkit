# SPDX-License-Identifier: LicenseRef-NVIDIA-1.0
# 
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Inception Score (IS) from the paper "Improved techniques for training
GANs". Matches the original implementation by Salimans et al. at
https://github.com/openai/improved-gan/blob/master/inception_score/model.py"""

import numpy as np
import matplotlib.pyplot as plt
from . import metric_utils

#----------------------------------------------------------------------------

def compute_is(opts, num_gen, num_splits):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt', '2d')
    detector_kwargs = dict(no_output_bias=True) # Match the original implementation by not applying bias in the softmax layer.

    gen_probs = metric_utils.compute_feature_stats_synthetic(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_all=True, max_items=num_gen).get_all()

    # Visualize t-SNE
    fig_path = opts.run_dir + '/figures/tsne_is.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_tsne('IS', real_features=None, gen_features=gen_probs, fig_path=fig_path)

    # Visualize PCA
    fig_path = opts.run_dir + '/figures/pca_is.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_pca('IS', real_features=None, gen_features=gen_probs, fig_path=fig_path)

    # --- Compute values ---
    max_probs = np.max(gen_probs, axis=1)
    top1_classes = np.argmax(gen_probs, axis=1)

    # --- Create plots ---

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Histogram of max probabilities
    ax[0].hist(max_probs, bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax[0].set_title('Max Softmax Probabilities', fontsize=16)
    ax[0].set_xlabel('Max Probability', fontsize=14)
    ax[0].set_ylabel('Frequency', fontsize=14)
    ax[0].grid(True)

    # 2. Top-1 class prediction distribution
    ax[1].hist(top1_classes, bins=100, color='orange', edgecolor='black')
    ax[1].set_title('Top-1 Predicted Class', fontsize=16)
    ax[1].set_xlabel('Class Index', fontsize=14)
    ax[1].set_ylabel('Frequency', fontsize=14)
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(opts.run_dir + '/figures/is_probs.png')
    plt.close()


    num_gen = len(gen_probs) if num_gen is None else num_gen
        
    if opts.rank != 0:
        return float('nan'), float('nan')

    scores = []
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))

#----------------------------------------------------------------------------
