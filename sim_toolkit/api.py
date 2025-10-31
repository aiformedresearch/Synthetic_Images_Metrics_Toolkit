# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Any, Callable

from copy import deepcopy
import random
import numpy as np

from . import dnnlib
from ._deps import require_backends


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_global_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _class_path_from_obj(cls) -> str:
    return f"{cls.__module__}.{cls.__name__}"

def _infer_dataset_from_path(path: str) -> str:
    p = str(path).lower()
    if p.endswith((".nii", ".nii.gz")):
        return "nifti"
    if p.endswith((".jpg", ".jpeg")):
        return "jpeg"
    if p.endswith((".png")):
        return "png"
    if p.endswith((".tif", ".tiff")):
        return "tiff"
    # fallback
    return "image_folder"

def _dataset_class_name(
    name_or_obj,
    *,
    data_type: str,                 # "2D" or "3D"
    path_data: str | None = None,
) -> str:
    """
    Return a fully-qualified class path for dnnlib.util.construct_class_by_name().

    Accepts:
      - short names: "nifti", "jpeg", "tiff", "image_folder", "auto"
      - fully-qualified strings: "pkg.mod.Class" or "pkg.mod:Class"
      - a class object (e.g., NiftiDataset2D)
    Uses `data_type` to pick 2D vs 3D variants where applicable (NIfTI).
    """
    # class object -> use as-is
    if hasattr(name_or_obj, "__mro__"):
        return _class_path_from_obj(name_or_obj)

    name = str(name_or_obj).strip()
    dtype = str(data_type).lower()

    if ":" in name:
        name = name.replace(":", ".")

    # If it's already fully-qualified, trust it and don't override based on data_type
    if "." in name and name not in {"nifti", "jpeg", "tiff", "image_folder", "auto"}:
        return name

    # auto-detect base dataset by extension
    if name == "auto":
        if not path_data:
            raise ValueError("dataset='auto' requires path_data to infer the format.")
        name = _infer_dataset_from_path(path_data)

    # Per-data_type mapping
    mapping_2d = {
        "nifti":        "sim_toolkit.datasets.nifti.NiftiDataset2D",       # 2D
        "png":          "sim_toolkit.datasets.png.PNGDataset",
        "jpeg":         "sim_toolkit.datasets.jpeg.JPEGDataset",
        "tiff":         "sim_toolkit.datasets.tiff.TifDataset",
        "dcm":          "sim_toolkit.datasets.dcm.DicomDataset2D",
    }
    mapping_3d = {
        "nifti":        "sim_toolkit.datasets.nifti.NiftiDataset3D",     # 3D
        "tiff":         "sim_toolkit.datasets.tiff.TifDataset",
        "dcm":          "sim_toolkit.datasets.dcm.DicomDataset3D",
    }

    if dtype == "3d":
        if name in mapping_3d:
            return mapping_3d[name]
        if name in mapping_2d:
            raise ValueError(
                f"Dataset '{name}' has no 3D loader. "
                f"Choose a supported 3D dataset (e.g., 'nifti') or switch data_type='2D'."
            )
        raise ValueError(f"Unknown dataset '{name}' for 3D.")
    else:
        # default 2D
        if name in mapping_2d:
            return mapping_2d[name]
        raise ValueError(f"Unknown dataset '{name}' for 2D.")

_DEFAULT_DS_PARAMS: Dict[str, Any] = {
    "path_data": None,     # REQUIRED later for file-based mode
    "path_labels": None,   # default
    "use_labels": False,   # default
    "size_dataset": None,  # default
}

def _normalize_params(params: Optional[Dict[str, Any]],
                      *,
                      require_path: bool,
                      who: str) -> Dict[str, Any]:
    """
    Merge user-supplied dataset params with defaults and validate.
    """
    merged = {**_DEFAULT_DS_PARAMS, **(params or {})}
    if require_path and not merged.get("path_data"):
        raise ValueError(f"{who}: 'path_data' is required for file-based mode.")
    return merged

def _mk_dataset_kwargs(dataset, params: dict | None, *, data_type: str) -> dnnlib.EasyDict:
    class_name = _dataset_class_name(dataset, data_type=data_type, path_data=(params or {}).get("path_data"))
    return dnnlib.EasyDict(class_name=class_name, **(params or {}))

# ---------------------------------------------------------------------
# Core worker (single rank)
# ---------------------------------------------------------------------

def _subprocess_fn(rank: int, args: dnnlib.EasyDict, temp_dir: str):
    import torch
    import torch.distributed as dist
    from .torch_utils import training_stats
    try:
        from .torch_utils import custom_ops
    except Exception:
        custom_ops = None
    from .metrics import metric_main, metric_utils
    from .metrics.create_report import generate_metrics_report

    dnnlib.util.Logger(should_flush=True)

    # Seed & device
    set_global_seed(args.seed)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")

    # Init torch.distributed
    if args.num_gpus > 1 and torch.cuda.is_available():
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(
                backend="gloo", init_method=init_method, rank=rank, world_size=args.num_gpus
            )
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(
                backend="nccl", init_method=init_method, rank=rank, world_size=args.num_gpus
            )

    # Init torch_utils
    sync_device = torch.device("cuda", rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if custom_ops is not None:
        if rank != 0 or not args.verbose:
            custom_ops.verbosity = "none"
    else:
        if rank == 0 and args.verbose:
            print("[SIM] custom_ops unavailable; running without compiled extensions.")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Visualize samples (rank 0 only)
    if rank == 0:
        # Real
        real_dataset = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)
        drange_real = [real_dataset._min, real_dataset._max]
        grid_size, images_real, labels = metric_utils.setup_snapshot_image_grid(args, real_dataset)
        if args.data_type.lower() == "3d":
            images_real = metric_utils.setup_grid_slices(images_real, grid_size, drange_real)
        metric_utils.plot_image_grid(
            args, images_real, drange=drange_real, grid_size=grid_size, group="real", rank=rank, verbose=args.verbose
        )

        # Synthetic
        if args.use_pretrained_generator:
            G = deepcopy(args.G).eval().to(torch.device("cuda:0"))  # keep G on cuda:0 for snapshots
            device_ = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            num_images = labels.shape[0]
            images_synt = metric_utils.setup_grid_generated(args, G, labels, grid_size, num_images, real_dataset, device_)
        else:
            synt_dataset = dnnlib.util.construct_class_by_name(**args.dataset_synt_kwargs)
            grid_size, images_synt, _ = metric_utils.setup_snapshot_image_grid(args, synt_dataset)
        drange_synt = [images_synt.min(), images_synt.max()]
        if args.data_type.lower() == "3d":
            images_synt = metric_utils.setup_grid_slices(images_synt, grid_size, drange_synt)
        metric_utils.plot_image_grid(
            args, images_synt, drange=drange_synt, grid_size=grid_size, group="synt", rank=rank, verbose=args.verbose
        )

    if args.num_gpus > 1 and dist.is_initialized():
        dist.barrier()

    # Compute metrics
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f"Calculating {metric}...")
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)

        # OC detector
        train_OC = False if args.oc_detector_path is not None else True
        oc_detector_path = args.oc_detector_path if args.oc_detector_path is not None else os.path.join(args.run_dir, "oc_detector.pkl")

        result_dict = metric_main.calc_metric(
            metric=metric,
            use_pretrained_generator=args.use_pretrained_generator,
            run_generator=args.run_generator,
            num_gen=args.num_gen,
            nhood_size=args.nhood_size,
            knn_configs=args.knn_configs,
            padding=args.padding,
            oc_detector_path=oc_detector_path,
            train_OC=train_OC,
            snapshot_pkl=args.network_path,
            run_dir=args.run_dir,
            batch_size=args.batch_size,
            data_type=args.data_type,
            cache=args.cache,
            seed=args.seed,
            comp_metrics=args.metrics,
            G=args.G,
            dataset_kwargs=args.dataset_kwargs,
            dataset_synt_kwargs=args.dataset_synt_kwargs,
            num_gpus=args.num_gpus,
            rank=rank,
            device=device,
            progress=progress,
        )

        if rank == 0:
            synt_source = args.network_path if args.use_pretrained_generator else args.dataset_synt_kwargs["path_data"]
            metric_main.report_metric(
                result_dict, run_dir=args.run_dir, real_source=args.dataset_kwargs["path_data"], synt_source=synt_source
            )
        if rank == 0 and args.verbose:
            print()

    # Final report
    generate_metrics_report(args)

    # Done
    if rank == 0 and args.verbose:
        print("Exiting...")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def compute(
    *,
    # What to compute
    metrics: List[str],
    # Runtime
    run_dir: str,
    num_gpus: int = 1,
    batch_size: int = 64,
    data_type: str = "2D",                 # "2D" | "3D"
    use_cache: bool = True,
    verbose: bool = True,
    padding: bool = False,
    seed: int = 42,
    oc_detector_path: Optional[str] = None,
    # Metrics params
    prdc_nhood_size: int = 5,
    knn_num_real: int = 3,
    knn_num_synth: int = 5,
    # REAL data
    real_dataset: str = "auto",                         # "nifti" | "tif" | "png" | "jpg" | "image_folder" | fully-qualified class
    real_params: Optional[Dict[str, Any]] = None,       # must include path_data
    # SYNTH data (choose one mode)
    # (A) from files:
    synth_dataset: Optional[str] = "auto",               # "nifti" | "tif" | "png" | "jpg" | "image_folder" | fully-qualified class
    synth_params: Optional[Dict[str, Any]] = None,       # must include path_data
    # (B) pretrained generator:
    use_pretrained_generator: bool = False,
    network_path: Optional[str] = None,                  # path to .pkl (or your format)
    load_network: Optional[Callable[[str], object]] = None,
    run_generator: Optional[Callable[..., object]] = None,
    num_gen: Optional[int] = None,                       # number of images to synthesize
) -> Dict[str, float]:
    """
    Programmatic entry-point. Mirrors the old CLI/config flow but takes Python args.

    Minimal examples:
    -----------------
    # From files (NIfTI)
    compute(
        metrics=["fid","kid","prdc", "pr_auth", "knn"],
        run_dir="./runs/exp1",
        real_dataset="nifti",
        real_params={"path_data": "data/real.nii.gz"},
        synth_dataset="nifti",
        synth_params={"path_data": "data/synth.nii.gz"},
    )

    # From a pretrained generator
    compute(
        metrics=["fid","kid", "prdc", "pr_auth", "knn"],
        run_dir="./runs/exp2",
        real_dataset="nifti",
        real_params={"path_data": "data/real.nii.gz"},
        use_pretrained_generator=True,
        network_path="path/to/network.pkl",
        load_network=my_loader_fn,          # def my_loader_fn(network_path) -> G
        run_generator=my_runner_fn,         # def my_runner_fn(opts) -> Tensor[N,H,W] or Tensor[N,D,H,W]
        num_gen=5000,
    )
    """
    os.makedirs(run_dir, exist_ok=True)

    metrics_norm = {m.lower() for m in metrics}
    dtype_norm = str(data_type).lower()

    # Always need torch; tf only for the computation of 2D pr_auth and prdc
    need_torch = True
    need_tf = (dtype_norm == "2d") and bool({"pr_auth", "prdc"} & metrics_norm)

    require_backends(
        need_torch=need_torch,
        need_tf=need_tf,
        reason=("core pipeline" if need_torch and not need_tf else "metrics 'pr_auth' and 'prdc'")
    )

    import torch
    set_global_seed(seed)
    
    # Validate mode selection
    if use_pretrained_generator:
        if network_path is None or load_network is None:
            raise ValueError("When use_pretrained_generator=True, you must provide network_path and load_network().")
        if synth_dataset is not None or synth_params:
            # Ignore file-based synth if generator mode is on
            synth_dataset = None
            synth_params = None
    else:
        synth_params = _normalize_params(synth_params, require_path=True, who="synth_params")
        if not (synth_dataset and synth_params and synth_params.get("path_data")):
            raise ValueError("When using 'from_files' mode, provide synth_dataset and synth_params['path_data'].")

    real_params = _normalize_params(real_params, require_path=True, who="real_params")
    if not (real_dataset and real_params and real_params.get("path_data")):
        raise ValueError("Please provide real_dataset and real_params['path_data'].")

    # Default configs
    nhood_size = prdc_nhood_size
    knn_configs = {"num_real": knn_num_real, "num_synth": knn_num_synth}

    # Build dnnlib args
    args = dnnlib.EasyDict({
        "metrics": metrics,
        "run_dir": run_dir,
        "batch_size": batch_size,
        "data_type": data_type,
        "cache": use_cache,
        "knn_configs": knn_configs,
        "nhood_size": nhood_size,
        "padding": padding,
        "num_gpus": num_gpus,
        "verbose": verbose,
        "oc_detector_path": oc_detector_path,
        "use_pretrained_generator": use_pretrained_generator,
        "seed": seed,
    })

    # Real dataset
    args.dataset_kwargs = _mk_dataset_kwargs(real_dataset, real_params, data_type=args.data_type)

    # Synthetic source
    if use_pretrained_generator:
        if verbose:
            print(f'Loading network from "{network_path}"...')
        args.G = load_network(network_path)
        args.network_path = network_path
        args.load_network = load_network
        args.run_generator = run_generator
        args.num_gen = int(num_gen) if num_gen is not None else None
        args.dataset_synt_kwargs = None
    else:
        args.G = None
        args.dataset_synt_kwargs = _mk_dataset_kwargs(synth_dataset, synth_params, data_type=args.data_type)
        args.num_gen = synth_params.get("size_dataset") 
        args.network_path = None
        args.load_network = None
        args.run_generator = None

    # Launch
    if verbose:
        print("Launching processes...")
    torch.multiprocessing.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if num_gpus <= 1:
            if num_gpus == 0:
                print("Running in CPU mode...")
            else:
                print("Running in single GPU mode...")
            _subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            print(f"Spawning {num_gpus} processes...")
            torch.multiprocessing.spawn(fn=_subprocess_fn, args=(args, temp_dir), nprocs=num_gpus)

    # The detailed metric values are written by metric_main.report_metric().
    # For convenience, return the last metrics.json if present.
    results_json = os.path.join(run_dir, "report", "metrics_summary.json")
    if os.path.exists(results_json):
        try:
            import json
            with open(results_json, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}