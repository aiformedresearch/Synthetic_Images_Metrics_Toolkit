"""
MIO CODICE

Taken from https://github.com/clovaai/generative-evaluation-prdc

prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import metric_utils
import sklearn.metrics

#----------------------------------------------------------------------------

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x.cpu(), data_y.cpu(), metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k=5):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii

# ----------------------------------------------------------------------------

def compute_prdc(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):

    # detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    detector_url = {'model': 'vgg16', 'randomise': False, 'dim64': False}
    detector_kwargs = dict(return_features=True)

    # Compute the embedding from pre-trained detector

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all_torch().to(torch.float16).to(opts.device)

    gen_features = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_torch().to(torch.float16).to(opts.device)

    # Define the value of k for the kth nearest neighbour
    nearest_k = 5

    # Compute pairwise distances between real features and fake features
    # Compute the kth nearest neighbour distances for both real features
    # -> this gives a threshold distance for each real feature vector
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        gen_features, nearest_k)
    # Comparison: check if each element in distance_real_fake is less than the corresponding 
    #  threshold distance from real_nearest_neighbour_distances
    distance_real_fake = compute_pairwise_distance(
        real_features, gen_features)

    # Compute the PRDC metrics
    # Precision is calculated as the mean of a boolean array where each element indicates 
    #  if the corresponding generated feature vector is within the l-th nearest distance of  any real feature vectors
    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    print()
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Density: ', density)
    print('Coverage: ', coverage)
    print()
    return precision, recall, density, coverage