# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0
 
import numpy as np
import torch
import dnnlib
from sklearn.metrics.pairwise import cosine_similarity

from metrics import metric_utils

# Function to update the list of closest images
def update_closest_images(closest_images, closest_similarities, closest_indices, batch_synthetic_images, similarities, batch_indices, k=8):
    """
    Update the list of closest images for each real image by comparing the similarities, 
    and store the indices of the real images with the closest synthetic matches.
    """
    batch_synthetic_images_np = batch_synthetic_images.cpu().numpy()

    for real_idx in range(similarities.shape[0]):  # Iterate over real images
        if len(closest_images[real_idx]) == 0:  # If no images stored yet for this real image
            # Initialize with the current batch's top-k images
            top_k_indices = np.argsort(similarities[real_idx])[-k:][::-1]  # Get indices of top-k most similar
            closest_images[real_idx] = batch_synthetic_images_np[top_k_indices]
            closest_similarities[real_idx] = similarities[real_idx][top_k_indices]
            closest_indices[real_idx] = batch_indices[top_k_indices]

        else:
            # Combine current closest images and similarities with the new batch
            combined_similarities = np.concatenate([closest_similarities[real_idx], similarities[real_idx]])
            combined_images = np.concatenate([closest_images[real_idx], batch_synthetic_images_np])
            combined_indices = np.concatenate([closest_indices[real_idx], batch_indices])
            
            # Sort by similarity and keep the top k
            sorted_indices = np.argsort(combined_similarities)[-k:][::-1]  # Get top-k most similar
            closest_images[real_idx] = combined_images[sorted_indices]
            closest_similarities[real_idx] = combined_similarities[sorted_indices]
            closest_indices[real_idx] = combined_indices[sorted_indices]

    return closest_images, closest_similarities, closest_indices

# Generate batches of synthetic images and compare to real embeddings
def process_batches_and_find_closest(opts, real_embeddings, detector_url, detector_kwargs, num_gen, k=8):
    num_real_images = real_embeddings.shape[0]

    if not opts.use_pretrained_generator:
        synt_dataset = dnnlib.util.construct_class_by_name(**opts.dataset_synt_kwargs)
        if num_gen is None:
            num_gen = len(synt_dataset)

    # Initialize lists to store the closest synthetic images and their similarities for each real image
    closest_images = {i: [] for i in range(num_real_images)}  # Dictionary to store closest synthetic images
    closest_similarities = {i: [] for i in range(num_real_images)}  # To store similarity scores
    closest_indices = {i: [] for i in range(num_real_images)}

    # Loop over synthetic image batches
    batch_size = min(num_gen, opts.batch_size)
    num_batches = num_gen // batch_size

    for batch_idx in range(num_batches):
        batch_indices = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        if opts.use_pretrained_generator:
            batch_embeddings, batch_synthetic_images = metric_utils.compute_feature_stats_for_generator(
                opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
                rel_lo=0, rel_hi=1, return_imgs=True, capture_all=True, max_items=batch_size)
        else:
            batch_embeddings, batch_synthetic_images = metric_utils.compute_feature_stats_for_dataset(
                opts=opts, dataset=synt_dataset,
            detector_url=detector_url, detector_kwargs=detector_kwargs, 
            rel_lo=0, rel_hi=1, dataset_kwargs=opts.dataset_synt_kwargs, return_imgs=True, item_subset=batch_indices, capture_all=True, max_items=batch_size)
        batch_embeddings = batch_embeddings.get_all_torch().to(torch.float16).to(opts.device)
        batch_synthetic_images = batch_synthetic_images.to(opts.device)

        # Compute similarity between real embeddings and this batch of synthetic embeddings
        similarities = cosine_similarity(real_embeddings.cpu(), batch_embeddings.cpu())
        
        # Update the closest images and similarities for each real image
        closest_images, closest_similarities, closest_indices = update_closest_images(
            closest_images, closest_similarities, closest_indices, batch_synthetic_images, similarities, batch_indices, k=k)
    
    return closest_images, closest_similarities, closest_indices # Dictionaries with shape: [num_real_images x k x C x H x W]


# Main function to compute embeddings, find k-NN in batches, and visualize results
def plot_knn(opts, max_real, num_gen, k=8, top_n=6):
    #detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    if opts.data_type.lower() == '2d':
        detector_url = ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt', '2d')
    elif opts.data_type.lower() == '3d':
        detector_url = ('https://zenodo.org/records/15234379/files/resnet_50_23dataset_cpu.pth?download=1', '3d')
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    # Step 1: Get embeddings for real images
    real_embeddings = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, dataset=dnnlib.util.construct_class_by_name(**opts.dataset_kwargs), 
        detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, dataset_kwargs=opts.dataset_kwargs, capture_all=True, max_items=max_real).get_all_torch().to(torch.float16).to(opts.device)

    # Step 2: Process synthetic images in batches and find the closest synthetic images
    closest_images, closest_similarities, closest_indices = process_batches_and_find_closest(
        opts, real_embeddings, detector_url, detector_kwargs, num_gen=num_gen, k=k
    )

    # Step 3: Select the top_n real images with the smallest distance to any synthetic image
    fig_path = opts.run_dir + '/figures/knn_analysis.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    top_n_real_indices = metric_utils.select_top_n_real_images(closest_similarities, top_n=top_n)
    metric_utils.visualize_top_k(opts, closest_images, closest_indices, top_n_real_indices, fig_path, top_n=top_n, k=k)

    return fig_path