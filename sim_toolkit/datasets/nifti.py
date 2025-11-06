# SPDX-FileCopyrightText: 2025 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import os
import numpy as np
from glob import glob

from .base import BaseDataset, BidsDataset
from .._utils import  _to_chw

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
        Load 2D NIfTI data and return NumPy array in (N, C, H, W) format.

        - If `path_data` is a file (.nii / .nii.gz):
            Assumes data stored as (W, H, C, N) and converts to (N, C, H, W).
        - If `path_data` is a folder:
            Loads all *.nii / *.nii.gz in the folder. Each file must contain a
            single 2D image (H,W) or (H,W,C) with C in {1,3,4}. Stacks to (N,C,H,W).
        """
        _require_nibabel()
        import nibabel as nib

        p = os.path.abspath(self.path_data)

        # --- Case A: single file ---
        if os.path.isfile(p) and (p.endswith(".nii") or p.endswith(".nii.gz")):
            data = nib.load(p)
            data = np.asanyarray(data.dataobj) # numpy array of shape (W,H,C,N)
            data = np.float64(data)

            if data.ndim != 4:
                raise RuntimeError(f"Expected a 4D NIfTI array shaped like (W,H,C,N); got shape {data.shape}.")
            warn_once(f"Assuming NIfTI files stored as (W, H, C, N) format. Your data has shape: {data.shape}.",
                      key="nifti2d.image_format")

            # Swap axes to have the format (N,C,W,H)
            data = np.swapaxes(data, 0,3)
            data = np.swapaxes(data, 1,2)   # after swapping axes, array shape (N,C,H,W)

            return data # [batch_size, n_channels, img_resolution, img_resolution]

        # --- Case B: folder of files ---
        if os.path.isdir(p):
            file_paths = sorted(
                glob(os.path.join(p, "*.nii")) + glob(os.path.join(p, "*.nii.gz"))
            )
            if not file_paths:
                raise RuntimeError(
                    f"No NIfTI files (.nii/.nii.gz) found under directory: {p}"
                )

            images = []
            bad = []

            for fp in file_paths:
                try:
                    img = nib.load(fp)
                    arr = np.asanyarray(img.dataobj)

                    # Allow 2D or 2D+channels images only
                    if arr.ndim == 4:
                        # Many NIfTI files store (H,W,C,N) with N=1 for a single image.
                        # Accept this special case and squeeze N if feasible.
                        if arr.shape[-1] == 1 and arr.shape[2] in (1, 3, 4):
                            arr = arr[..., 0]  # (H,W,C)
                        else:
                            raise ValueError(f"{os.path.basename(fp)}: 4D array {arr.shape} not supported for 2D dataset.")

                    arr = arr.astype(np.float32, copy=False)
                    chw = _to_chw(arr)  # (C,H,W)
                    images.append(chw)
                except Exception as e:
                    bad.append((os.path.basename(fp), str(e)))

            if not images:
                lines = [f"No readable 2D NIfTI images loaded from {p}."]
                if bad:
                    lines.append("Some reasons:")
                    for name, reason in bad[:5]:
                        lines.append(f"  - {name}: {reason}")
                raise RuntimeError("\n".join(lines))

            # Ensure consistent (C,H,W)
            shapes = {im.shape for im in images}
            if len(shapes) != 1:
                raise ValueError(
                    f"Inconsistent image shapes in folder: {sorted(shapes)}. "
                    "Please resample/crop to a uniform (C,H,W) shape."
                )

            data = np.stack(images, axis=0)  # (N,C,H,W)
            return data

        # --- Neither file nor folder ---
        raise RuntimeError(
            f"`path_data` must be a NIfTI file (.nii/.nii.gz) or a directory; got: {p}"
        )

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