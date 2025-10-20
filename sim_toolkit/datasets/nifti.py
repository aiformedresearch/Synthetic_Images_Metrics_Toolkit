# SPDX-FileCopyrightText: 2025 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import numpy as np
from .base import BaseDataset
from .base3D import BidsDataset

def _require_nibabel():
    try:
        import nibabel as nib  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "NIfTI support requires 'nibabel'. Install with: pip install 'sim_toolkit[nifti]'"
        ) from e

class NiftiDataset2D(BaseDataset):
    def _load_files(self):
        """
        Load a NIfTI file and return a NumPy array.
        Expects images to be in (N, C, H, W) format.
        """
        _require_nibabel()
        import nibabel as nib

        data = nib.load(self.path_data)
        data = np.asanyarray(data.dataobj) # numpy array of shape (W,H,C,N)
        data = np.float64(data)

        W, H, C, N = data.shape
        assert W==H

        # Swap axes to have the format (N,C,W,H)
        data = np.swapaxes(data, 0,3)
        data = np.swapaxes(data, 1,2)   # after swapping axes, array shape (N,C,H,W)

        return data # [batch_size, n_channels, img_resolution, img_resolution]

    def _load_raw_labels(self):
        pass

class NiftiDataset3D(BidsDataset):
    def _load_files(self, input):
        """
        Load a 3D NIfTI file and return a NumPy array.
        Expects images to be in (C, H, W, D) format.
        """
        _require_nibabel()
        import nibabel as nib
        
        data = nib.load(input).get_fdata()

        data = np.rot90(data, k=3, axes=(0, 1))
        data = np.rot90(data, k=1, axes=(0, 2))
        data = np.rot90(data, k=1, axes=(1, 2))

        data = np.expand_dims(data, axis=0)

        # Normalize to the range [0, 255]
        min_val = data.min()
        max_val = data.max()
        data = (data - min_val) / (max_val - min_val)
        data *= 255

        return data.copy() # [n_channels, img_resolution, img_resolution, img_resolution]

    def _load_raw_labels(self):
        pass