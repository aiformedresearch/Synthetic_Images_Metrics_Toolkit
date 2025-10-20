# SPDX-FileCopyrightText: 2025 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

from glob import glob
import os
import numpy as np

from datasets.base import BaseDataset

def _require_opencv():
    try:
        import cv2  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "TIFF support requires OpenCV. Install with: pip install 'sim_toolkit[opencv]'"
        ) from e

class TifDataset(BaseDataset):
    def _load_files(self):
        """
        Load all TIFF images from the directory and return a NumPy array.
        Expects images to be in (N, C, H, W) format.
        """
        _require_opencv()
        import cv2

        image_paths = sorted(glob(os.path.join(self.path_data, "*.tif")) + glob(os.path.join(self.path_data, "*.tiff")))
        images = []

        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)  # Ensure single-channel images have (H, W, 1)
                images.append(image)
            else:
                print(f"Warning: Could not load {path}")

        if not images:
            raise RuntimeError(f"No TIFF images found in {self.path_data}")

        data = np.stack(images, axis=0)  # Shape: (N, H, W, C)
        data = np.moveaxis(data, -1, 1)  # Convert to (N, C, H, W) format
        return data # [batch_size, n_channels, img_resolution, img_resolution]

    def _load_raw_labels(self):
        pass