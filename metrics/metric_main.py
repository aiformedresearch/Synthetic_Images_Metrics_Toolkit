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
import json
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score
from . import pr_authen
from . import density_coverage
from . import knn_analysis

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, run_generator, num_gen, oc_detector_path, train_OC, snapshot_pkl, run_dir, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(run_dir, run_generator, snapshot_pkl, num_gen, oc_detector_path, train_OC, **kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        print(f'Saving metrics in {run_dir}')
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# Legacy metrics, from Karras et al.

@register_metric
def is_(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=opts.num_gen, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

@register_metric
def fid(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=opts.num_gen)
    return dict(fid50k=fid)

@register_metric
def kid(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=opts.num_gen, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def pr(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=opts.num_gen, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)

@register_metric
def ppl_zfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=opts.num_gen, epsilon=1e-4, space='z', sampling='full', crop=True, batch_size=2)
    return dict(ppl_zfull=ppl)

@register_metric
def ppl_wfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=opts.num_gen, epsilon=1e-4, space='w', sampling='full', crop=True, batch_size=2)
    return dict(ppl_wfull=ppl)

@register_metric
def ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=opts.num_gen, epsilon=1e-4, space='z', sampling='end', crop=True, batch_size=2)
    return dict(ppl_zend=ppl)

@register_metric
def ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=opts.num_gen, epsilon=1e-4, space='w', sampling='end', crop=True, batch_size=2)
    return dict(ppl_wend=ppl)

#----------------------------------------------------------------------------
# Extra metrics, from Lai et al.

@register_metric
def pr_auth(opts):
    opts.dataset_kwargs.update(max_size=None)
    a_precision_c, b_recall_c, authenticity_c  = pr_authen.compute_pr_a(opts, max_real=50000, num_gen=opts.num_gen, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(a_precision_c=a_precision_c, b_recall_c=b_recall_c, authenticity_c=authenticity_c)

@register_metric
def prdc(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall, density, coverage  = density_coverage.compute_prdc(opts, max_real=50000, num_gen=opts.num_gen, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)

@register_metric
def knn(opts):
    opts.dataset_kwargs.update(max_size=None)
    path_to_img = knn_analysis.plot_knn(opts, max_real=50000, num_gen=opts.num_gen, batch_size=64, k=5, top_n=3)
    return dict(path_to_knn=path_to_img)
