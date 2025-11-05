# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import warnings
from .base import BidsDataset  # re-export

__all__ = ["BidsDataset"]

warnings.warn(
    "sim_toolkit.datasets.base3D is deprecated; import BidsDataset from "
    "sim_toolkit.datasets.base instead.",
    DeprecationWarning,
    stacklevel=2,
)