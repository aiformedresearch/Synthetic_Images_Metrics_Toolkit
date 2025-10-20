﻿# SPDX-FileCopyrightText: 2024 NVIDIA CORPORATION
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
import csv
import torch
from .. import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import inception_score
from . import pr_authen
from . import pr_dc
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

def calc_metric(metric, use_pretrained_generator, run_generator, num_gen, nhood_size, knn_configs, padding, oc_detector_path, train_OC, snapshot_pkl, run_dir, batch_size, data_type, cache, seed, comp_metrics, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(run_dir, batch_size, data_type, use_pretrained_generator, run_generator, snapshot_pkl, num_gen, nhood_size, knn_configs, padding, oc_detector_path, train_OC, cache, seed, comp_metrics, **kwargs)

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

def report_metric(result_dict, run_dir=None, real_source=None, synt_source=None):
    metric = result_dict['metric']
    results = result_dict['results']
    assert isinstance(results, dict)

    # Build rows
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    base_row = {
        'flag': metric,
        'real_source': real_source,
        'synt_source': synt_source,
        'timestamp': timestamp,
        'total_time': round(result_dict.get('total_time', 0), 2),
        'total_time_str': result_dict['total_time_str'],
        'num_gpus': result_dict.get('num_gpus', 1),
    }

    rows = []
    for k, v in results.items():
        row = base_row.copy()
        row['metric'] = k
        row['score'] = v
        rows.append(row)

    # Save to CSV
    if run_dir is not None and os.path.isdir(run_dir):
        csv_path = os.path.join(run_dir, 'metrics.csv')
        write_header = not os.path.isfile(csv_path)
        print(f"Saving metrics in {csv_path}")
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['flag', 'metric', 'score', 'real_source', 'synt_source', 'timestamp', 'total_time', 'total_time_str', 'num_gpus']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

#----------------------------------------------------------------------------
# Legacy metrics, from Karras et al.

@register_metric
def is_(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=opts.num_gen, num_splits=10)
    return dict(is_mean=mean, is_std=std)

@register_metric
def fid(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=opts.max_size, num_gen=opts.num_gen)
    return dict(fid=fid)

@register_metric
def kid(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=opts.max_size, num_gen=opts.num_gen, num_subsets=100, max_subset_size=1000)
    return dict(kid=kid)

#----------------------------------------------------------------------------
# Extra metrics, from Lai et al.

@register_metric
def pr_auth(opts):
    opts.dataset_kwargs.update(max_size=None)
    a_precision, b_recall, authenticity  = pr_authen.compute_pr_a(opts, max_real=opts.max_size, num_gen=opts.num_gen)
    return dict(a_precision=a_precision, b_recall=b_recall, authenticity=authenticity)

@register_metric
def prdc(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall, density, coverage  = pr_dc.compute_prdc(opts, max_real=opts.max_size, num_gen=opts.num_gen)
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)

@register_metric
def knn(opts):
    opts.dataset_kwargs.update(max_size=None)
    knn_analysis.plot_knn(opts, max_real=opts.max_size, num_gen=opts.num_gen, k=opts.knn_config["num_synth"], top_n=opts.knn_config["num_real"])
    return {}
