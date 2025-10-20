# SPDX-FileCopyrightText: 2025 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import os
from glob import glob
import numpy as np

from datasets.base import BaseDataset

def _require_pillow():
    try:
        from PIL import Image  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "JPEG support requires 'Pillow'. Install with: pip install pillow"
        ) from e

class PNGDataset(BaseDataset):
    def _load_files(self):
        """
        Load a PNG file and return a NumPy array.
        Expects images to be in (N, C, H, W) format.
        """
        _require_pillow()
        from PIL import Image, ImageOps
        
        image_paths = sorted(glob(os.path.join(self.path_data, "*.png")))
        images = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img) # Correct orientation

                    # Convert to either grayscale or RGB
                    if img.mode not in ["L", "RGB"]:
                        img = img.convert("RGB")

                    img_np = np.array(img)  # Shape: (H, W) or (H, W, C)

                    # Add channel dimension if grayscale
                    if img_np.ndim == 2:
                        img_np = img_np[np.newaxis, :, :]  # (1, H, W)
                    elif img_np.ndim == 3:
                        img_np = np.transpose(img_np, (2, 0, 1))  # (C, H, W)

                    images.append(img_np)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

        if not images:
            raise RuntimeError(f"No PNG images found in {self.path_data}")

        data = np.stack(images, axis=0)        # Shape: (N, C, H, W)

        return data  # [batch_size, n_channels, H, W]

    def _load_raw_labels(self):
        pass
