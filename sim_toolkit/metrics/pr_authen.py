# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: MIT

"""
alpha-Precision, beta-Recall and authenticity from the paper "How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models". 
Matches the original implementation by Alaa et al. at https://github.com/vanderschaarlab/evaluating-generative-models
"""

import torch
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from . import metric_utils
from .. import dnnlib

#----------------------------------------------------------------------------

def plot_curves(opts, alphas, alpha_precision_curve, beta_coverage_curve, authenticity_values, Delta_precision_alpha, AUC_alpha_precision, Delta_coverage_beta, AUC_beta_coverage, authen):
    plt.figure(figsize=(10, 6))
    
    # Plot alpha precision curve
    plt.plot(alphas, alpha_precision_curve, label='α-precision curve', marker='o')
    
    # Plot beta coverage curve
    plt.plot(alphas, beta_coverage_curve, label='β-recall curve', marker='s')
    plt.plot([0, 1], [0, 1], "k--", label="Optimal performance")
    
    # Add titles and labels
    plt.title('α-precision and β-recall curves', fontsize=18)
    plt.xlabel('α, β', fontsize=15)
    plt.ylabel('Value', fontsize=15)
    
    # Add legend
    plt.legend(fontsize=12)

    textstr = '\n'.join((
        f'$\Delta$ α-precision: {Delta_precision_alpha:.3f}',
        f'AUC α-precision: {AUC_alpha_precision:.3f}',
        f'$\Delta$ β-recall: {Delta_coverage_beta:.3f}',
        f'AUC β-recall: {AUC_beta_coverage:.3f}'
    ))

    # These are matplotlib.patches.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in upper left in axes coords
    plt.text(0.02, 0.75, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
        
    # Display the plot
    plt.grid(True)
    fig_dir = os.path.join(opts.run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    base_figname = os.path.join(fig_dir, f'alpha_precision_beta_recall_curves.png')
    figname = metric_utils.get_unique_filename(base_figname)
    plt.savefig(figname)
    plt.close()

    # Plot authenticity
    plt.figure(figsize=(10, 6))
    plt.hist(authenticity_values, bins=30, alpha=0.75)
    plt.axvline(x=authen, color='r', linestyle='--', label='Authenticity score')
    plt.title("Distribution of Authenticity Values", fontsize=18)
    plt.xlabel("Authenticity for each batch", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.legend(fontsize=12)
    base_figname = os.path.join(fig_dir, f'authenticity_distribution.png')
    figname = metric_utils.get_unique_filename(base_figname)
    plt.savefig(figname)
    plt.close() 

def compute_authenticity_in_batches(real_data, synthetic_data, batch_size=1024):

    authenticity_values = []

    # Determine the batch size
    batch_size = min(batch_size, real_data.shape[0], synthetic_data.shape[0])

    # Fit the NearestNeighbors model on real data once
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _ = nbrs_real.kneighbors(real_data)
    real_to_real = torch.from_numpy(real_to_real[:, 1].squeeze())  # Closest other real image (excluding itself)

    # Shuffle synthetic data once and split into batches without replacement
    idx = torch.randperm(synthetic_data.size(0))
    synthetic_data = synthetic_data[idx]
    num_batches = int(np.ceil(synthetic_data.shape[0] / batch_size))

    for i in range(num_batches):
        # Select batch of synthetic images without replacement
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, synthetic_data.shape[0])
        subset_synth_data = synthetic_data[start_idx:end_idx].cpu().numpy()

        # Compute nearest neighbors for this batch of synthetic data
        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(subset_synth_data)
        real_to_synth_auth, real_to_synth_args_auth = nbrs_synth.kneighbors(real_data)

        real_to_synth_auth = torch.from_numpy(real_to_synth_auth.squeeze())
        real_to_synth_args_auth = real_to_synth_args_auth.squeeze()

        # Find the closest real point to any real point (excluding itself)
        authen = real_to_real[real_to_synth_args_auth] < real_to_synth_auth
        batch_authenticity = authen.float().mean().item()
        authenticity_values.append(batch_authenticity)

    # Compute and return the average authenticity
    authenticity = np.mean(authenticity_values)
    return authenticity_values, authenticity

def compute_alpha_precision(opts, real_data, synthetic_data, emb_center):

    n_steps = 30
    alphas  = np.linspace(0, 1, n_steps)
        
    Radii   = np.quantile(torch.sqrt(torch.sum((real_data.float() - emb_center) ** 2, dim=1)).cpu().numpy(), alphas)
    
    synth_center          = synthetic_data.float().mean(dim=0)
    
    alpha_precision_curve = []
    beta_coverage_curve   = []
    
    synth_to_center       = torch.sqrt(torch.sum((synthetic_data.float() - emb_center) ** 2, dim=1))
      
    real_data_np = real_data.cpu().numpy()
    nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(real_data_np)
    real_to_real, _       = nbrs_real.kneighbors(real_data_np)
    
    nbrs_synth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(synthetic_data.cpu().numpy())
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data_np)

    # To compute authenticity, select a subset of fake images of the same number of real images
    smaller_population = min(real_data.shape[0], synthetic_data.shape[0])
    subset_synth_data = synthetic_data[np.random.choice(synthetic_data.shape[0], smaller_population, replace=False)].cpu().numpy()
    nbrs_synth_auth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(subset_synth_data)
    real_to_synth_auth, real_to_synth_args_auth = nbrs_synth_auth.kneighbors(real_data_np)

    # Let us find closest real point to any real point, excluding itself
    real_to_real          = torch.from_numpy(real_to_real[:,-1].squeeze()).to(opts.device) # Use the k-th neighbor distance
    real_to_synth         = torch.from_numpy(real_to_synth[:,-1].squeeze()).to(opts.device) # Use the k-th neighbor distance
    real_to_synth_auth    = torch.from_numpy(real_to_synth_auth.squeeze())
    real_to_synth_args    = real_to_synth_args[:,-1].squeeze()
    real_to_synth_args_auth = real_to_synth_args_auth.squeeze()

    real_synth_closest    = synthetic_data[real_to_synth_args].to(opts.device)
    
    real_synth_closest_d  = torch.sqrt(torch.sum((real_synth_closest.float()- synth_center) ** 2, dim=1))
    closest_synth_Radii   = np.quantile(real_synth_closest_d.cpu().numpy(), alphas)

    # Compute authenticity
    authenticity_values, authenticity = compute_authenticity_in_batches(real_data_np, synthetic_data)


    # Compute alpha precision and beta recall   
    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).float()
        alpha_precision      = precision_audit_mask.mean().item()

        beta_coverage        = ((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).float().mean().item()
 
        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)

    # Original metrics
    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    Delta_coverage_beta  = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) * (alphas[1] - alphas[0])
    
    # New AUC metrics
    AUC_alpha_precision = 2 * np.trapz(alpha_precision_curve, alphas)
    AUC_beta_coverage = 2 * np.trapz(beta_coverage_curve, alphas)

    return alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, AUC_alpha_precision, AUC_beta_coverage, authenticity_values, authenticity

#----------------------------------------------------------------------------

def compute_pr_a(opts, max_real, num_gen):

    # Load embedder function
    if opts.data_type.lower() == '2d':
        detector_url = {'model': 'vgg16', 'randomise': False, 'dim64': False}
    elif opts.data_type.lower() == '3d':
        detector_url = ('https://zenodo.org/records/15234379/files/resnet_50_23dataset_cpu.pth?download=1', '3d')
    detector_kwargs = dict(return_features=True)
    
    if detector_url is not None and opts.data_type.lower() == '2d':
        import tensorflow as tf
        embedder = metric_utils.load_embedder(detector_url)
        print('Checking if embedder is using GPU')
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        print(sess)
    
    # Compute the embedding from pre-trained detector
    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, dataset=dnnlib.util.construct_class_by_name(**opts.dataset_kwargs), 
        detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, dataset_kwargs=opts.dataset_kwargs, capture_all=True, max_items=max_real).get_all_torch().to(torch.float32).to(opts.device)

    gen_features = metric_utils.compute_feature_stats_synthetic(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_torch().to(torch.float32).to(opts.device)

    # Visualize t-SNE pre OC embedding
    fig_path = opts.run_dir + '/figures/tsne_pr_auth.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_tsne('α-precision, β-recall, & authenticity', real_features=real_features.cpu(), gen_features=gen_features.cpu(), fig_path=fig_path)

    # Visualize PCA pre OC embedding
    fig_path = opts.run_dir + '/figures/pca_pr_auth.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_pca('α-precision, β-recall, & authenticity', real_features=real_features.cpu(), gen_features=gen_features.cpu(), fig_path=fig_path)
    
    # Get the OC model (and eventually train it on the real features)
    OC_model, OC_params, OC_hyperparams = metric_utils.get_OC_model(opts, real_features, opts.OC_params, opts.OC_hyperparams)
    if opts.rank == 0:
        print(OC_params)
        print(OC_hyperparams)
    OC_model.eval()
     
    # Compute the metrics considering two different centers for the OC representation
    results = dict()

    # Embed the data into the OC representation
    if opts.rank == 0:
        print('Embedding data into OC representation')
    OC_model.to(opts.device)
    with torch.no_grad():
        real_features_OC = OC_model(real_features.float().to(opts.device))
        gen_features_OC = OC_model(gen_features.float().to(opts.device))
    
    if opts.rank == 0:
        print('Done embedding')
        print('real_features: mean, std - ', real_features_OC.mean(), real_features_OC.std(unbiased=False))
        print('gen_features:  mean, std - ', gen_features_OC.mean(), gen_features_OC.std(unbiased=False))

    emb_center = OC_model.c.to(opts.device)

    # Compute the metrics
    OC_res = compute_alpha_precision(opts, real_features_OC, gen_features_OC, emb_center)
    alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, AUC_alpha_precision, AUC_beta_coverage, authenticity_values, authen = OC_res

    # Visualize t-SNE
    fig_path = opts.run_dir + '/figures/tsne_pr_auth_OC.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    metric_utils.plot_tsne('α-precision, β-recall, & authenticity (OC)', real_features_OC.cpu(), gen_features_OC.cpu(), fig_path)

    # Visualize PCA
    fig_path = opts.run_dir + '/figures/pca_pr_auth_OC.png'
    fig_path = metric_utils.get_unique_filename(fig_path)
    center = OC_model.c.cpu().float().detach().numpy()
    Radii   = np.quantile(torch.sqrt(torch.sum((real_features_OC.float() - emb_center) ** 2, dim=1)).cpu().numpy(), 1 - OC_model.nu)
    circle_info = [center, Radii]
    metric_utils.plot_pca('α-precision, β-recall, & authenticity (OC)', real_features_OC.cpu(), gen_features_OC.cpu(), circle_info=circle_info, fig_path=fig_path)          

    results[f'alphas'] = alphas
    results[f'alpha_pc'] = alpha_precision_curve
    results[f'beta_cv'] = beta_coverage_curve
    results[f'auten'] = authen
    results[f'Dap'] = Delta_precision_alpha
    results[f'Dbr'] = Delta_coverage_beta
    results[f'AUC_ap'] = AUC_alpha_precision
    results[f'AUC_br'] = AUC_beta_coverage
    results[f'Daut'] = np.mean(authen)
    if opts.rank == 0:
        print('OneClass: Delta_alpha-precision', results[f'Dap'])
        print('OneClass: AUC_alpha-precision', results[f'AUC_ap'])
        print('OneClass: Delta_beta-recall  ', results[f'Dbr'])
        print('OneClass: AUC_beta-recall  ', results[f'AUC_br'])
        print('OneClass: Delta_autenticity    ', results[f'Daut'])

    # Plot the curves
    if opts.rank == 0:
        plot_curves(opts, alphas, alpha_precision_curve, beta_coverage_curve, authenticity_values, Delta_precision_alpha, AUC_alpha_precision, Delta_coverage_beta, AUC_beta_coverage, authen)

    return results['AUC_ap'], results['AUC_br'], results['Daut']


