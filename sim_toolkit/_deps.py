# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

from __future__ import annotations
import importlib.util, platform, sys

def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

def has_torch() -> bool:
    return _has("torch")

def has_tf() -> bool:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return _has("tensorflow") or _has("tensorflow_macos") or _has("tensorflow-macos")
    return _has("tensorflow")

def suggest_install(torch_needed: bool, tf_needed: bool) -> str:
    if torch_needed and tf_needed:
        return 'pip install "sim_toolkit[torch,tf]"'
    if torch_needed:
        return 'pip install "sim_toolkit[torch]"'
    if tf_needed:
        return 'pip install "sim_toolkit[tf]"'
    return "pip install sim_toolkit"

def require_backends(*, need_torch: bool, need_tf: bool, reason: str = "") -> None:
    missing = []
    t_missing = need_torch and not has_torch()
    f_missing = need_tf and not has_tf()
    if t_missing: missing.append("PyTorch")
    if f_missing: missing.append("TensorFlow")
    if missing:
        cmd = suggest_install(t_missing, f_missing)
        msg = "Missing required backend(s): " + ", ".join(missing) + "."
        if reason: msg += f" Required for: {reason}."
        msg += f"\nInstall with: \n\n{cmd}\n\n" \
               "Tip (CUDA): choose a CUDA wheel from https://pytorch.org/get-started/locally/ if you need GPU."
        raise RuntimeError(msg)
