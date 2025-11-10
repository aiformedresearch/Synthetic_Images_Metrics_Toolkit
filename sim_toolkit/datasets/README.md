# Load data from a custom dataset

The SIM Toolkit ships with ready-to-use loaders for:

- NIfTI (`nifti`)
- DICOM (`dcm`)
- TIFF (`tiff`)
- JPEG (`jpeg`)
- PNG (`png`)

For many use cases, you can simply select one of the built-in formats:

```python
real_dataset = "nifti"      # or "dcm", "tiff", "jpeg", "png"
synth_dataset = "auto"      # let SIM infer from path_data
```

If your file format or folder layout is different, you can still use the SIM Toolkit **without modifying its source code** by defining a small custom dataset class.

## 🧩 Defining a custom dataset class

Create a Python class that inherits from `sim_toolkit.datasets.base.BaseDataset`.

You can keep it in your own script or module and either:
- pass the **class object directly** (if it’s importable / in scope), or
- reference it via a **`.py` file path**.

```python
# my_dataset.py
import numpy as np
from sim_toolkit.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def _load_files(self):
        """
        Load your data and return a NumPy array with shape:
          - 2D: np.ndarray of shape (N, C, H, W)
          - 3D: np.ndarray of shape (N, C, H, W, D)

        You can use:
          - self.path_data
          - self.path_labels
          - self._use_labels
          - self.size_dataset (via params)
          """
        data = ...  # build your array here
        return data.astype(np.float32)

    def _load_raw_labels(self):
        """
        Optional — only needed if you set use_labels=True.

        Should return a NumPy array with one label entry per sample
        (e.g. shape (N,) or (N, K)), or None.

        If use_labels=True but this returns None (or is not implemented),
        SIM Toolkit will emit a warning once and continue **without** labels.
        """
        return None
```

### Using the class directly
If `MyDataset` is defined in the same script (or properly imported), you can pass the class itself:

```python
import sim_toolkit as sim
from my_dataset import MyDataset


sim.compute(
    metrics=["fid", "kid"],
    run_dir="./runs/exp1",
    data_type="2D",
    num_gpus=1,

    real_dataset=MyDataset,
    real_params={
        "path_data": "/path/to/real_data",
        "use_labels": False,
    },

    synth_dataset=MyDataset, 
    synth_params={
        "path_data": "/path/to/synth_data",
        "use_labels": False,
    },
)

```

### Using a `.py` file path
Alternatively, you can reference the implementation file as a string:
```python
import sim_toolkit as sim

sim.compute(
    metrics=["fid", "kid"],
    run_dir="./runs/exp1",
    data_type="2D",
    num_gpus=1,

    real_dataset="path/to/my_dataset.py:MyDataset",
    real_params={"path_data": "/path/to/real_data", "use_labels": False},

    synth_dataset="path/to/my_dataset.py:MyDataset",
    synth_params={"path_data": "/path/to/synth_data", "use_labels": False},
)

```
If you omit `:MyDataset`, SIM Toolkit will automatically use the first class in the file that inherits from `BaseDataset`.

## ✅ Requirements:

- Your class **must inherit** from `sim_toolkit.datasets.base.BaseDataset`.
- `_load_files()` must return:
    - `(N, C, H, W)` for 2D data, or
    - `(N, C, H, W, D)` for 3D data.
- `_load_raw_labels()`:
    - Implement only if `use_labels=True`.
    - If missing or returning `None`, labels are silently ignored after a warning.

## 📌 Conventions:

- All dataset configuration is passed via `real_params` / `synth_params`, e.g.:
   ```python
    real_params = {
        "path_data": "/path/to/data",
        "path_labels": "/path/to/labels",  # optional
        "use_labels": True,
        "size_dataset": None,
    }
   ```
- These are exposed inside your dataset as: 
    - `self.path_data`, 
    - `self.path_labels`, 
    - `self.use_labels`,  
    - `self.size_dataset`,
    - plus any extra keys you pass in `*_params`.
- Custom datasets are supported for both **real** and **synthetic** data in file-based mode.
- When using a **pre-trained generator**, this mechanism is used only for the real dataset; synthetic samples are drawn from your `run_generator` function.
