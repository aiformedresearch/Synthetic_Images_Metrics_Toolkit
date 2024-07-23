"""
MIO CODICE

alpha-Precision, beta-Recall and authenticity from the paper "How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models". 
Matches the original implementation by Alaa et al. at https://github.com/vanderschaarlab/evaluating-generative-models
"""

import torch
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import metric_utils
import matplotlib.pyplot as plt

import tensorflow as tf

#----------------------------------------------------------------------------
def plot_curves(alphas, alpha_precision_curve, beta_coverage_curve, run_dir, emb):
    plt.figure(figsize=(10, 6))
    
    # Plot alpha precision curve
    plt.plot(alphas, alpha_precision_curve, label='Alpha Precision Curve', marker='o')
    
    # Plot beta coverage curve
    plt.plot(alphas, beta_coverage_curve, label='Beta Coverage Curve', marker='s')
    plt.plot([0, 1], [0, 1], "k--", label="Optimal performance")
    
    # Add titles and labels
    plt.title('Alpha Precision and Beta Coverage Curves')
    plt.xlabel('alpha, beta')
    plt.ylabel('Value')
    
    # Add legend
    plt.legend()
    
    # Display the plot
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, f'alpha_precision_beta_coverage_curves{emb}.png'))

def compute_alpha_precision(opts, real_data, synthetic_data, emb_center):
    
    emb_center = torch.tensor(emb_center, device='cpu')

    n_steps = 30
    nn_size = 2
    alphas  = np.linspace(0, 1, n_steps)
        
    
    Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)
    
    synth_center          = torch.tensor(np.mean(synthetic_data, axis=0)).float()
    
    alpha_precision_curve = []
    beta_coverage_curve   = []
    
    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))
    
    
    nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _       = nbrs_real.kneighbors(real_data)
    
    nbrs_synth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(synthetic_data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data)

    # To compute authenticity, select a subset of fake images of the same number of real images
    subset_synth_data = synthetic_data[np.random.choice(synthetic_data.shape[0], real_data.shape[0], replace=False)]
    nbrs_synth_auth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(subset_synth_data)
    real_to_synth_auth, real_to_synth_args_auth = nbrs_synth_auth.kneighbors(real_data)

    # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
    real_to_real          = torch.from_numpy(real_to_real[:,1].squeeze())
    real_to_synth         = torch.from_numpy(real_to_synth.squeeze())
    real_to_synth_auth    = torch.from_numpy(real_to_synth_auth.squeeze())
    real_to_synth_args    = real_to_synth_args.squeeze()
    real_to_synth_args_auth = real_to_synth_args_auth.squeeze()

    real_synth_closest    = synthetic_data[real_to_synth_args]
    
    real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float()- synth_center) ** 2, dim=1))
    closest_synth_Radii   = np.quantile(real_synth_closest_d, alphas)


    
    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)

        beta_coverage        = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).detach().float().numpy())
 
        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)
    

    # See which one is bigger
    
    authen = real_to_real[real_to_synth_args_auth] < real_to_synth_auth
    authenticity = np.mean(authen.numpy())

    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    Delta_coverage_beta  = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) * (alphas[1] - alphas[0])
    
    return alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authenticity

#----------------------------------------------------------------------------

def compute_pr_a(opts, oc_detector_path, train_OC, run_dir, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):
    OC_params  = dict({"rep_dim": 32, 
                "num_layers": 3, 
                "num_hidden": 128, 
                "activation": "ReLU",
                "dropout_prob": 0.5, 
                "dropout_active": False,
                "LossFn": "SoftBoundary",
                "lr": 2e-3,
                "epochs": 2000,
                "warm_up_epochs" : 10,
                "train_prop" : 0.8,
                "weight_decay": 1e-2}   
                )   

    OC_hyperparams = dict({"Radius": 1, "nu": 1e-2})

    # # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    # detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    # # Compute the embedding from pre-trained detector
    # real_features = metric_utils.compute_feature_stats_for_dataset(
    #     opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
    #     rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all()

    # gen_features = metric_utils.compute_feature_stats_for_generator(
    #     opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
    #     rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all()


    # Load embedder function
    detector_url = embedding = {'model': 'inceptionv3', 'randomise': False, 'dim64': False}
    if embedding is not None:
        embedder = metric_utils.load_embedder(embedding)
        print('Checking of embedder is using GPU')
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        print(sess)
    
    # Compute the embedding from pre-trained detector
    real_features = metric_utils.get_activation(opts, opts.data_path, embedding, embedder=embedder, verbose=True)
    gen_features = metric_utils.get_activation(opts, opts.gen_path, embedding, embedder=embedder, verbose=True)
    

    # Get the OC model (and eventually train it on the real features)
    OC_model, OC_params, OC_hyperparams = metric_utils.get_OC_model(opts, oc_detector_path, train_OC, real_features, OC_params, OC_hyperparams)
    print(OC_params)
    print(OC_hyperparams)
    OC_model.eval()
     
    # Compute the metrics considering two different centers for the OC representation
    results = dict()
    for emb_index in [0,1]:

        if emb_index == 1:
            # Embed the data into the OC representation
            print('Computing metrics for OC embedding')
            print('Embedding data into OC representation')
            OC_model.to(opts.device)
            with torch.no_grad():
                real_features = OC_model(torch.tensor(real_features).float().to(opts.device)).cpu().detach().numpy()
                gen_features = OC_model(torch.tensor(gen_features).float().to(opts.device)).cpu().detach().numpy()
            print('Done embedding')
            print('real_features: mean, std - ', np.mean(real_features), np.std(real_features))
            print('gen_features:  mean, std - ', np.mean(gen_features), np.std(gen_features))
        else:
            print('Computing metrics for no additional OneClass embedding')

        if emb_index==1:
            emb = '_c'
            emb_center = OC_model.c
            print('\n-> with embedding centered in c=10:')
        else:
            emb = '_mean'
            emb_center = np.mean(real_features,axis=0)
            print('\n-> with as center the data center:')

        # Compute the metrics
        OC_res = compute_alpha_precision(opts, real_features, gen_features, emb_center)
        alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authen = OC_res
        
        results[f'alphas{emb}'] = alphas
        results[f'alpha_pc{emb}'] = alpha_precision_curve
        results[f'beta_cv{emb}'] = beta_coverage_curve
        results[f'auten{emb}'] = authen
        results[f'Dpa{emb}'] = Delta_precision_alpha
        results[f'Dcb{emb}'] = Delta_coverage_beta
        results[f'Daut{emb}'] = np.mean(authen)
        print('OneClass: Delta_precision_alpha', results[f'Dpa{emb}'])
        print('OneClass: Delta_coverage_beta  ', results[f'Dcb{emb}'])
        print('OneClass: Delta_autenticity    ', results[f'Daut{emb}'])
    
        # Plot the curves
        plot_curves(alphas, alpha_precision_curve, beta_coverage_curve, run_dir, emb)
    
    return results['Dpa_c'], results['Dcb_c'], results['Daut_c'], results['Dpa_mean'], results['Dcb_mean'], results['Daut_mean']


