# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0
 
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from metrics import metric_utils

# Function to update the list of closest images
def update_closest_images(closest_images, closest_similarities, batch_synthetic_images, similarities, k=8):
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
        else:
            # Combine current closest images and similarities with the new batch
            combined_similarities = np.concatenate([closest_similarities[real_idx], similarities[real_idx]])
            combined_images = np.concatenate([closest_images[real_idx], batch_synthetic_images_np])
            
            # Sort by similarity and keep the top k
            sorted_indices = np.argsort(combined_similarities)[-k:][::-1]  # Get top-k most similar
            closest_images[real_idx] = combined_images[sorted_indices]
            closest_similarities[real_idx] = combined_similarities[sorted_indices]

# Generate batches of synthetic images and compare to real embeddings
def process_batches_and_find_closest(opts, real_embeddings, detector_url, detector_kwargs, num_gen, batch_size=64, k=8):
    num_real_images = real_embeddings.shape[0]
    
    # Initialize lists to store the closest synthetic images and their similarities for each real image
    closest_images = {i: [] for i in range(num_real_images)}  # Dictionary to store closest synthetic images
    closest_similarities = {i: [] for i in range(num_real_images)}  # To store similarity scores

    # Loop over synthetic image batches
    G = opts.G.eval().requires_grad_(False).to(opts.device)  # Generator
    num_batches = num_gen // batch_size

    for batch_idx in range(num_batches):
        batch_embeddings, batch_synthetic_images = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, return_imgs=True, capture_all=True, max_items=batch_size)
        batch_embeddings = batch_embeddings.get_all_torch().to(torch.float16).to(opts.device)
        batch_synthetic_images = batch_synthetic_images.to(opts.device)

        # Compute similarity between real embeddings and this batch of synthetic embeddings
        similarities = cosine_similarity(real_embeddings.cpu(), batch_embeddings.cpu())
        
        # Update the closest images and similarities for each real image
        update_closest_images(closest_images, closest_similarities, batch_synthetic_images, similarities, k=k)
    
    return closest_images, closest_similarities # Dictionaries with shape: [num_real_images x k x C x H x W]


# Main function to compute embeddings, find k-NN in batches, and visualize results
def plot_knn(opts, max_real, num_gen, batch_size=64, k=8, top_n=6):
    #detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    # Step 1: Get embeddings for real images
    real_embeddings = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all_torch().to(torch.float16).to(opts.device)

    # Step 2: Process synthetic images in batches and find the closest synthetic images
    closest_images, closest_similarities = process_batches_and_find_closest(
        opts, real_embeddings, detector_url, detector_kwargs, num_gen=num_gen, batch_size=batch_size, k=k
    )

    # Step 3: Select the top 6 real images with the smallest distance to any synthetic image
    fig_path = opts.run_dir + '/figures/knn_analysis.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    top_n_real_indices = metric_utils.select_top_n_real_images(closest_similarities, top_n=top_n)
    metric_utils.visualize_top_k(opts, closest_images, top_n_real_indices, fig_path, batch_size, top_n=top_n, k=k)

    return fig_path