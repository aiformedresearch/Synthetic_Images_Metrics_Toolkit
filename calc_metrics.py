# SPDX-License-Identifier: LicenseRef-NVIDIA-1.0
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import click
import tempfile
import torch
import dnnlib

from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops

import importlib 
from metrics.create_report import generate_metrics_report

import sys
if not sys.warnoptions:
    import warnings 
    warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------

def load_config_from_path(config_path):
    """Dynamically loads a config.py file from the given path."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    return config

#----------------------------------------------------------------------------

def validate_config(config):
    """Validates the configuration parameters in config.py."""
    
    errors = []

    # Validate METRICS
    if not isinstance(config.METRICS, list) or not all(isinstance(m, str) for m in config.METRICS):
        errors.append("METRICS must be a list of metric names (strings).")
    elif not all(metric_main.is_valid_metric(m) for m in config.METRICS):
        valid_metrics = metric_main.list_valid_metrics()
        errors.append(f"Invalid metric(s) found. Allowed options: {valid_metrics}")

    # Validate CONFIGS
    required_config_keys = ["RUN_DIR", "NUM_SYNTH", "NUM_GPUS", "VERBOSE", "OC_DETECTOR_PATH"]
    for key in required_config_keys:
        if key not in config.CONFIGS:
            errors.append(f"Missing key in CONFIGS: {key}")

    if not isinstance(config.CONFIGS["RUN_DIR"], str):
        errors.append("RUN_DIR must be a string (path).")

    if not isinstance(config.CONFIGS["NUM_SYNTH"], int) or config.CONFIGS["NUM_SYNTH"] <= 0:
        errors.append("NUM_SYNTH must be a positive integer.")

    if not isinstance(config.CONFIGS["NUM_GPUS"], int) or config.CONFIGS["NUM_GPUS"] < 0:
        errors.append("NUM_GPUS must be an integer greater than or equal to 0.")

    if not isinstance(config.CONFIGS["VERBOSE"], bool):
        errors.append("VERBOSE must be a boolean (True/False).")

    # Validate DATASET
    dataset = config.DATASET
    if not isinstance(dataset, dict):
        errors.append("DATASET must be a dictionary.")
    else:
        required_dataset_keys = ["module", "class_name", "params"]
        for key in required_dataset_keys:
            if key not in dataset:
                errors.append(f"Missing key in DATASET: {key}")

        if not isinstance(dataset["module"], str) or not isinstance(dataset["class_name"], str):
            errors.append("DATASET module and class_name must be strings.")

        if not isinstance(dataset["params"], dict):
            errors.append("DATASET params must be a dictionary.")

        if "path_data" in dataset["params"] and dataset["params"]["path_data"] is not None:
            if not os.path.exists(dataset["params"]["path_data"]):
                errors.append(f"Dataset file not found: {dataset['params']['path_data']}")

    # Validate GENERATOR
    generator = config.GENERATOR
    if not isinstance(generator, dict):
        errors.append("GENERATOR must be a dictionary.")
    else:
        required_generator_keys = ["network_path", "load_network", "run_generator"]
        for key in required_generator_keys:
            if key not in generator:
                errors.append(f"Missing key in GENERATOR: {key}")

        if not isinstance(generator["network_path"], str) or not os.path.exists(generator["network_path"]):
            errors.append(f"Generator checkpoint file not found: {generator['network_path']}")

        if not callable(generator["load_network"]):
            errors.append("load_network must be a callable function.")

        if not callable(generator["run_generator"]):
            errors.append("run_generator must be a callable function.")

    # Print errors if any
    if errors:
        raise ValueError("\n".join(errors))

#----------------------------------------------------------------------------

def print_config(config):
    """Prints the loaded configuration in a readable format."""
    print("\nLoaded Configuration:")
    print(f"  METRICS: {config.METRICS}")
    print("\n  CONFIGS:")
    for key, value in config.CONFIGS.items():
        print(f"    {key}: {value}")
    print("\n  GENERATOR:")
    for key, value in config.GENERATOR.items():
        print(f"    {key}: {value}")
    print("\n  DATASET:")
    for key, value in config.DATASET.items():
        print(f"    {key}: {value}")
    print()

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)
    # define device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() and args.num_gpus > 0 else 'cpu')
    
    # Init torch.distributed.
    if args.num_gpus > 1 and torch.cuda.is_available():
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Visualize one sample for real and generated data
    if rank == 0:
        metric_utils.visualize_ex_samples(args, device=device, rank=rank, verbose=args.verbose)

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
 
        # Set the path to the OC detector:
        train_OC = False if args.oc_detector_path is not None else True
        oc_detector_path = args.oc_detector_path if args.oc_detector_path is not None else args.run_dir+'/oc_detector.pkl'
        
        result_dict = metric_main.calc_metric(
            metric=metric,
            run_generator=args.generator["run_generator"], 
            num_gen=args.num_gen, 
            knn_configs = args.knn_configs,
            oc_detector_path=oc_detector_path, 
            train_OC=train_OC, 
            snapshot_pkl=args.generator['network_path'], 
            run_dir=args.run_dir, 
            G=args.G, 
            dataset_kwargs=args.dataset_kwargs,
            num_gpus=args.num_gpus, 
            rank=rank, device=device, 
            progress=progress
            )
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.generator['network_path'])
        if rank == 0 and args.verbose:
            print()

    # Create the final report.
    generate_metrics_report(args)

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--config', help='Path to config.py', metavar='PATH', required=True)

def calc_metrics(ctx, config):

    dnnlib.util.Logger(should_flush=True)

    # Load configuration dynamically
    print(f"Reading configuration from {config}...")
    config = load_config_from_path(config)

    # Validate configuration
    validate_config(config)

    args = dnnlib.EasyDict({
        'metrics': config.METRICS,
        'run_dir': config.CONFIGS["RUN_DIR"],
        'knn_configs': config.CONFIGS["K-NN_configs"],
        'num_gpus': config.CONFIGS["NUM_GPUS"],
        'verbose': config.CONFIGS["VERBOSE"],
        'num_gen': config.CONFIGS["NUM_SYNTH"],
        'oc_detector_path': config.CONFIGS["OC_DETECTOR_PATH"],
        'generator': config.GENERATOR,
        'run_generator': config.GENERATOR["run_generator"],
        'dataset': config.DATASET,
    })

    # Print configuration values
    if args.verbose:
        print_config(config)

    # Load the pre-trained generator
    if args.verbose:
        print(f'Loading network from "{args.generator["network_path"]}"...')
    args.G = args.generator["load_network"](args.generator["network_path"])

    # Initialize dataset options.
    if args.dataset['params']['path_data'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(
            class_name=args.dataset["module"] + "." + args.dataset["class_name"],
            **args.dataset["params"]
                )
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus <= 1:
            if args.num_gpus==0:
                print("Running in CPU mode...")
            else:
                print("Running in single GPU mode...")
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            print(f"Spawning {args.num_gpus} processes...")
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
