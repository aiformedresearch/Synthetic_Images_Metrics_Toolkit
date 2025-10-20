# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0
import os
import numpy as np
import torch
import torch.utils.data as data
from glob import glob

class BidsDataset(data.Dataset):
    def __init__(self, 
            path_data,              # Path to the dataset
            path_labels=None,       # (optional) Path to the labels
            use_labels=False,       # Enable conditioning labels? False = label dimension is zero.
            size_dataset=None,      # Max size of the dataset
            random_seed = 0,        # Random seed to use when applying max_size.
            structure='sub-*/anat/*_T1w.nii.gz', # Define the structure of the NIfTI folder
            **kwargs):
        self.path_data = path_data
        self.path_labels = path_labels
        self.structure = structure
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Get the list of all subject directories (sub-<id>)
        self.inputfiles = sorted(glob(os.path.join(self.path_data, self.structure)))

        # Early, user-friendly error if nothing matches
        if not self.inputfiles:
            # Optionally: show a few available NIfTI files (helps users find the right structure)
            candidates = sorted(glob(os.path.join(self.path_data, "**", "*.nii*"), recursive=True))[:5]
            hint = ""
            if candidates:
                shown = "\n  - " + "\n  - ".join(os.path.relpath(p, self.path_data) for p in candidates)
                hint = (
                    "\n\nHere are a few NIfTI files I did find under the dataset root; "
                    "you can use these to identify the correct structure pattern:\n" + shown +
                    "\n\nExamples of alternative patterns you might try:\n"
                    "  - 'sub-*/ses-*/anat/*_T1w.nii.gz'\n"
                    "  - 'sub-*/anat/*.nii.gz'\n"
                    "  - '**/*.nii.gz'  (recursive)"
                )

            # Default BIDS-like structure used by the dataset
            structure_default = 'sub-*/anat/*_T1w.nii.gz'

            raise FileNotFoundError(
                "No files were found in the dataset directory that match the expected structure.\n"
                f"  Dataset root:   {os.path.abspath(self.path_data)}\n"
                f"  Structure used: {self.structure}\n\n"
                "This likely means your data are organized differently. "
                "Try specifying a different structure (glob pattern) using the 'structure' parameter.\n\n"
                "📁 **Note:** NIfTI 3D files are treated as organized according to the BIDS structure, "
                f"which by default is structured as:\n"
                f"    {structure_default}\n"
                "but you can change it by providing a different value for the 'structure' parameter.\n\n"
                "💡 **Tip:** When defining your dataset parameters, you typically need to specify:\n"
                '    params = {"path_data": "/path/to_data",\n'
                '              "path_labels": "/path/to_labels",\n'
                '              "use_labels": False,\n'
                '              "size_dataset": None}\n\n'
                "For 3D NIfTI files organized according to the BIDS structure, "
                "you also need to include the additional parameter:\n"
                f'    params = {{ ..., "structure": "{structure_default}" }}\n\n'
                f"Note: '{structure_default}' is the default structure, "
                "but you can modify it if your dataset uses a different organization." + hint
            )


        
        # Load dataset
        #self._data = self._load_files(self.path_data)
        self._labels = self._load_raw_labels() if use_labels and path_labels else None

        # Store dataset metadata
        self.name = os.path.basename(path_data)
        example_img = self._load_files(self.inputfiles[0])
        self._raw_shape = [len(self.inputfiles)] + list(example_img.shape)
        self._dtype = example_img.dtype
        self._min = example_img.min()
        self._max = example_img.max()

        # Apply max_size.
        self._raw_idx = np.arange(len(self.inputfiles), dtype=np.int64)
        if size_dataset and len(self._raw_idx) > size_dataset:
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:size_dataset])

    def update_minmax(self, image):
        if image.min() < self._min:
            self._min = image.min()
        if image.max() > self._max:
            self._max = image.max()

    def __len__(self):
        return len(self._raw_idx)

    def __getitem__(self, idx):
        inputfile = self.inputfiles[self._raw_idx[idx]]
        image = self._load_files(inputfile)
        self.update_minmax(image)
        label = self._labels[idx] if self._labels is not None else -1
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels
    
    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        return label.copy()

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)
        
    def _load_files(self):
        """
        Users must implement this function in subclasses.
        Should return a NumPy array of shape (N, C, H, W).
        """
        raise NotImplementedError

    def _load_raw_labels(self):
        """
        Users must implement this function in subclasses if labels are used.
        """
        raise NotImplementedError


