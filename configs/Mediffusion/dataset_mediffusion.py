# SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size=None,          # Artificially limit the size of the dataset. None = no limit.
        use_labels=False,       # Enable conditioning labels? False = label dimension is zero.
        xflip=False,            # Artificially double the size of the dataset via x-flips.
        random_seed=0,          # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

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

    def _load_raw_image(self, raw_idx):
        raise NotImplementedError  # To be implemented in subclasses.

    def _load_raw_labels(self):
        raise NotImplementedError  # To be implemented in subclasses.

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]
    
    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class NiftiDataset(Dataset):
    def __init__(self,
        path_data,                   # Path to nifti file.
        path_labels=None,       # Path to labels file (CSV).
        sel_ids=None,           # Selected indices for training/validation split.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path_d = path_data
        self._path_l = path_labels
        self._sel_ids = sel_ids if sel_ids else []

        # Load data and determine raw shape.
        data = self._load_nifti_file()
        raw_shape = list(data.shape)

        # Initialize the base Dataset class.
        super().__init__(name=os.path.basename(path_data), raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_nifti_file(self):
        # Load nifti data.
        data = nib.load(self._path_d)
        data = np.asanyarray(data.dataobj) # numpy array of shape (W,H,C,N)
        data = np.float64(data)

        W, H, C, N = data.shape # Best format for visualization in ImageJ: (W,H,C,N)
        assert W==H

        # Swap axes to have the format (N,C,W,H)
        data = np.swapaxes(data, 0,3)
        data = np.swapaxes(data, 1,2)   # after swapping axes, array shape (N,C,H,W)
        return data

    def _load_raw_image(self, raw_idx):
        data = self._load_nifti_file()
        image = data[raw_idx,:,:,:]

        # If the image values are in the range [-1, 1], convert the image data in the range [0, 255]
        if np.min(image) < 0.0 and np.max(image) <= 1.0:
            image = (image + 1.0) * 127.5
        # If the image values are in the range [0, 1], convert the image data in the range [0, 255]
        if np.min(image) >= 0.0 and np.max(image) <= 1.0:
            image = image * 255.0
        #image = image.astype(np.uint8)
        return image

    def _load_raw_labels(self):
        # Load labels from file csv
        labels = pd.read_csv(self._path_l, delimiter=',')
        # if sel_ids==[]:
        #     # If sel_ids (selected indexes) are not inserted as input, consider all the indexes
        #     sel_ids = labels['ID'].values.astype(np.int)
        #sel_ids = labels['ID'].values.astype(int)
        labels = labels['Group'].map({'CN':0, 'AD':1}).values.astype(int)#[sel_ids]
        # labels = labels['Label'].values.astype(int)[sel_ids]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels