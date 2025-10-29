# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

__version__ = "4.0.0"
__all__ = ["compute", "__version__"]

def compute(*args, **kwargs):
    # Lazy import to avoid heavy deps & circulars at package import time
    from .api import compute as _compute
    return _compute(*args, **kwargs)