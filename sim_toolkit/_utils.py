# SPDX-FileCopyrightText: 2025 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import os
import numpy as np
import threading
import hashlib
import importlib.util
import inspect
from types import ModuleType
import sys
from typing import Any, Dict, Optional

from . import dnnlib
try:
    from torch.utils.data import get_worker_info  # optional at runtime
except Exception:  # torch may not be installed yet at import time
    get_worker_info = None  # type: ignore[attr-defined]

# ---------- rank/worker helpers ----------

def is_dist_rank0() -> bool:
    """True if process looks like rank 0 (DDP-friendly, env-based)."""
    r = os.getenv("RANK")
    lr = os.getenv("LOCAL_RANK")
    return (r in (None, "", "0")) and (lr in (None, "", "0"))

def get_worker_id() -> Optional[int]:
    """Return DataLoader worker id or None if single-process/no torch."""
    if get_worker_info is None:
        return None
    wi = get_worker_info()
    return None if wi is None else wi.id

def is_worker0() -> bool:
    """True if DataLoader worker is id==0 or if no workers."""
    wid = get_worker_id()
    return wid in (None, 0)

def is_main_process() -> bool:
    """Convenience: rank0 AND worker0."""
    return is_dist_rank0() and is_worker0()

# ---------- print/warn once per process ----------

def print_once(msg: str) -> None:
    """Print once per process and only from main process (rank0/worker0)."""
    if not is_main_process():
        return
    if getattr(print_once, "_done", False):
        return
    print(msg)
    setattr(print_once, "_done", True)

__once_keys = set()
__once_lock = threading.Lock()

def warn_once(msg: str, *, key: str | None = None, print_fn=print) -> None:
    k = key or msg
    with __once_lock:
        if k in __once_keys:
            return
        __once_keys.add(k)
    print_fn(f"[SIM Toolkit] {msg}")

# ---------- array shape helpers ----------
def _to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Convert a single image array to CHW (C,H,W).
    Accepts:
      - (H, W) -> (1, H, W)
      - (H, W, C) with C in {1,3,4} -> (C, H, W)
      - (C, H, W) with C in {1,3,4} -> (C, H, W)
    Raises on ambiguous shapes (e.g., volumetric).
    """
    if arr.ndim == 2:
        return arr[np.newaxis, ...]  # (1,H,W)
    if arr.ndim == 3:
        # Treat last-dim-small as channels (H,W,C)
        if arr.shape[-1] in (1, 3, 4) and arr.shape[0] > 8 and arr.shape[1] > 8:
            return np.moveaxis(arr, -1, 0)  # (C,H,W)
        # Already CHW?
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
            return arr  # (C,H,W)
        # Looks like a volume or multi-slice (H,W,D) with D not small
        raise ValueError(f"Ambiguous 3D shape {arr.shape}: looks like a volume/multi-slice; use a 3D loader.")

    raise ValueError(f"Unsupported array shape {arr.shape} (expected 2D image or 2D+channels).")

# ---------------------------------------------------------------------
# Utilities dataset/model handling
# ---------------------------------------------------------------------

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
      - short names: "nifti", "jpeg", "tiff", "png", "dicom", "auto"
      - fully-qualified strings: "pkg.mod.Class" or "pkg.mod:Class"
      - a class object (e.g., NiftiDataset2D)
      - a filesystem path to a .py file (optionally with a class):
            "C:/path/my_ds.py"
            "C:/path/my_ds.py:MyDataset"
    Uses `data_type` to pick 2D vs 3D variants where applicable (NIfTI, DICOM).
    """
    # class object -> use as-is
    if hasattr(name_or_obj, "__mro__"):
        return _class_path_from_obj(name_or_obj)

    name = str(name_or_obj).strip()
    dtype = str(data_type).lower()

    maybe_py, maybe_cls = None, None
    if ":" in name:
        left, right = name.rsplit(":", 1)
        if left.lower().endswith(".py") and os.path.exists(os.path.abspath(left)):
            maybe_py, maybe_cls = left, right.strip()

    # Case A: "path/to/file.py:Class"
    # Case B: "path/to/file.py" (no ':Class')
    if name.lower().endswith(".py") and os.path.exists(os.path.abspath(name)):
        from .datasets.base import BaseDataset
        return _class_path_from_file_spec(name, BaseDatasetRef=BaseDataset)
    if maybe_py is not None:
        from .datasets.base import BaseDataset
        return _class_path_from_file_spec(f"{maybe_py}:{maybe_cls}", BaseDatasetRef=BaseDataset)

    if ":" in name:
        name = name.replace(":", ".")

    # If it's already fully-qualified, trust it and don't override based on data_type
    if "." in name and name not in {"nifti", "jpeg", "tiff", "png", "dicom", "auto"}:
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


# --- helper: load a module from a .py file and register it under a unique name ---
def _load_module_from_file(py_path: str) -> ModuleType:
    """Load a Python file as a module and register it under a unique name."""
    abspath = os.path.abspath(py_path)
    if not os.path.isfile(abspath) or not abspath.lower().endswith(".py"):
        raise ValueError(f"Expected a .py file, got: {py_path}")

    # Make a repeatable unique name so it can be imported by dnnlib.construct_class_by_name
    stem = os.path.splitext(os.path.basename(abspath))[0]
    digest = hashlib.sha1(abspath.encode("utf-8")).hexdigest()[:10]
    module_name = f"_sim_userds_{stem}_{digest}"

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for file: {py_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# --- helper: find a BaseDataset subclass inside a module ---
def _find_dataset_class_in_module(module: ModuleType, *, class_name: str | None, BaseDatasetRef) -> type:
    """
    If class_name is provided, return it (after validation).
    Else, find the first subclass of BaseDataset in the module.
    """
    if class_name:
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in module '{module.__name__}'.")
        cls = getattr(module, class_name)
        if not (inspect.isclass(cls) and issubclass(cls, BaseDatasetRef)):
            raise TypeError(f"'{class_name}' is not a subclass of BaseDataset.")
        return cls

    # autodetect: first subclass of BaseDataset (excluding BaseDataset itself)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and issubclass(obj, BaseDatasetRef) and obj.__name__ != "BaseDataset":
            return obj

    raise RuntimeError(
        f"No subclass of BaseDataset was found in module '{module.__name__}'. "
        f"Define e.g. 'class MyDataset(BaseDataset): ...' and try again."
    )

# --- helper: turn "path/to/file.py[:ClassName]" into "importable.module.ClassName" ---
def _class_path_from_file_spec(path_spec: str, *, BaseDatasetRef) -> str:
    """
    Accepts:
      - 'C:/path/custom_ds.py'
      - 'C:/path/custom_ds.py:MyDataset'
    Returns: 'registered.unique.module.MyDataset'
    """
    # split optional ":ClassName"
    drive, rest = os.path.splitdrive(path_spec)
    if ":" in rest:
        py_path, cls_name = rest.split(":", 1)
        py_path = os.path.join(drive, py_path)
        cls_name = cls_name.strip()
    else:
        py_path = os.path.join(drive, rest)
        cls_name = None

    module = _load_module_from_file(py_path)
    cls = _find_dataset_class_in_module(module, class_name=cls_name, BaseDatasetRef=BaseDatasetRef)
    return f"{module.__name__}.{cls.__name__}"

