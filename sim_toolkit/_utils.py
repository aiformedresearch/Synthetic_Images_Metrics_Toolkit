# SPDX-FileCopyrightText: 2025 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import os
import numpy as np
import threading
from typing import Optional
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
