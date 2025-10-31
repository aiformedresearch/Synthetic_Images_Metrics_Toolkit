# SPDX-FileCopyrightText: 2025 Matteo Lai <matteo.lai3@unibo.it>
# SPDX-License-Identifier: NPOSL-3.0

import os
from glob import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from .base import BaseDataset
from .._utils import warn_once

class UserError(RuntimeError):
    """Compact, user-facing error (suppresses long trace)."""

class MissingExtraError(UserError):
    """Required optional dependency is missing (e.g., pydicom)."""

class DataLoadError(UserError):
    """Dataset could not be loaded/decoded."""

# -------- lazy deps --------
def _require_pydicom():
    try:
        import pydicom  # noqa: F401
    except Exception:
        raise MissingExtraError(
            "\nDICOM support requires pydicom.\n\n"
            "Install with: pip install \"sim_toolkit[dicom]\""
        ) from None

# -------- helpers --------
def _read_dicom_pixel(path: str) -> Tuple[np.ndarray, dict]:
    """
    Read a single DICOM file and return (H, W) float32 pixel array plus selected tags.
    Applies RescaleSlope/Intercept and PhotometricInterpretation when possible.
    """
    _require_pydicom()
    import pydicom

    ds = pydicom.dcmread(path, force=True)  # force=True tolerates some quirks

    # Skip non-image SOP classes early (optional)
    if not hasattr(ds, "PixelData"):
        raise ValueError(f"No PixelData in DICOM: {path}")

    # Get pixel array; pydicom handles many transfer syntaxes
    arr = ds.pixel_array  # dtype varies

    # Handle MONOCHROME1 (invert)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric.upper() == "MONOCHROME1":
        # MONOCHROME1: higher values are displayed as darker, invert it to match MONOCHROME2
        arr = arr.max() - arr

    arr = arr.astype(np.float32, copy=False)

    # Rescale
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or inter != 0.0:
        arr = arr * slope + inter

    # Collect a few tags used for grouping/sorting
    meta = {
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
        "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
        "InstanceNumber": getattr(ds, "InstanceNumber", None),
        "ImagePositionPatient": getattr(ds, "ImagePositionPatient", None),
        "ImageOrientationPatient": getattr(ds, "ImageOrientationPatient", None),
        "SliceLocation": getattr(ds, "SliceLocation", None),
        "SeriesDescription": getattr(ds, "SeriesDescription", None),
        "ImageType": [s.upper() for s in getattr(ds, "ImageType", [])],
    }
    return arr, meta

def _is_derived(meta: dict) -> bool:
    """Heuristic to skip derived/reformatted images."""
    img_type = meta.get("ImageType") or []
    if any(tok in img_type for tok in ("DERIVED", "SECONDARY", "LOCALIZER")):
        return True
    # Some localizers are marked in the description
    desc = (meta.get("SeriesDescription") or "").lower()
    if "localizer" in desc or "scout" in desc:
        return True
    return False

def _sort_slices(metas: List[dict]) -> List[int]:
    """
    Determine slice order using ImagePositionPatient (IPP) if present,
    else fall back to InstanceNumber, else identity.
    Returns indices that sort the list.
    """
    # Prefer IPP (z coordinate along slice normal)
    has_ipp = [m.get("ImagePositionPatient") is not None for m in metas]
    if all(has_ipp):
        # Use the third coordinate (z). If orientation is available, you could project onto slice normal.
        zs = [float(m["ImagePositionPatient"][2]) for m in metas]
        return sorted(range(len(metas)), key=lambda i: zs[i])
    # Fallback: InstanceNumber
    has_in = [m.get("InstanceNumber") is not None for m in metas]
    if all(has_in):
        ins = [int(m["InstanceNumber"]) for m in metas]
        return sorted(range(len(metas)), key=lambda i: ins[i])
    # Last resort: keep as is
    return list(range(len(metas)))

# ===============================
#           2D DATASET
# ===============================
class DicomDataset2D(BaseDataset):
    """
    Loads ALL DICOM files from a folder (optionally recursive) as individual 2D images.
    Returns data with shape (N, C, H, W), where C is 1 (grayscale) or 3/4 (RGB/RGBA).
    """

    def __init__(self, path_data: str, recursive: bool = False, normalize: bool = False,
                 allow_derived: bool = False,  # new: optionally keep derived/localizer
                 **kwargs):
        self.recursive = recursive
        self.normalize = normalize
        self.allow_derived = allow_derived
        super().__init__(path_data=path_data, **kwargs)

    def _iter_dicom_paths(self):
        # Match common extensions & case variants
        pats = ["*.dcm", "*.DCM", "*.dicom", "*.DICOM"]
        for pat in pats:
            pattern = os.path.join(self.path_data, "**", pat) if self.recursive else os.path.join(self.path_data, pat)
            for p in glob(pattern, recursive=self.recursive):
                yield p

    def _load_files(self):
        paths = sorted(set(self._iter_dicom_paths()))
        if not paths:
            raise DataLoadError(
                f"No DICOM files found in {os.path.abspath(self.path_data)} "
                f"(extensions: .dcm/.DCM/.dicom; recursive={self.recursive})."
            ) from None

        images: List[np.ndarray] = []
        bad: List[Tuple[str, str]] = []  # (path, reason)

        for p in paths:
            try:
                arr, meta = _read_dicom_pixel(p)
                if not self.allow_derived and _is_derived(meta):
                    continue  # skip scouts/derived
            except MissingExtraError as e:
                raise e 
            except Exception as e:
                # Try to give a helpful Transfer Syntax hint
                try:
                    _require_pydicom()
                    import pydicom
                    ds_hdr = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                    ts = getattr(ds_hdr.file_meta, "TransferSyntaxUID", "unknown")
                    bad.append((p, f"decode-failed (TransferSyntax={ts}): {e.__class__.__name__}"))
                except Exception:
                    bad.append((p, f"decode-failed: {e.__class__.__name__}"))
                continue

            # Accept true 2D, RGB/RGBA, and CHW
            if arr.ndim == 2:
                chw = arr[None, ...]  # (1, H, W)
            elif arr.ndim == 3:
                # Color image? (H, W, C) with small C
                if arr.shape[-1] in (3, 4):         # HWC -> CHW
                    chw = np.moveaxis(arr, -1, 0)
                elif arr.shape[0] in (3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:  # already CHW
                    chw = arr
                else:
                    # Likely multi-frame (frames, H, W) or ambiguous 3D → direct the user to 3D loader
                    bad.append((p, f"multi-frame or 3D shape {arr.shape} (use a 3D DICOM dataset)"))
                    continue
            else:
                bad.append((p, f"unsupported shape {arr.shape}"))
                continue

            # Optional per-image min–max normalization to [0,1]
            chw = chw.astype(np.float32, copy=False)
            if self.normalize:
                vmin, vmax = float(chw.min()), float(chw.max())
                if vmax > vmin:
                    chw = (chw - vmin) / (vmax - vmin)

            images.append(chw)

        if not images:
            # Summarize why nothing was loaded and offer concrete fixes
            raise DataLoadError(
                "No readable 2D DICOM images were loaded.\n"
                "\nCommon fixes:\n"
                "  • If TransferSyntax is JPEG-Lossless/JPEG-LS/JPEG2000, install a decoder:\n"
                "      - conda install -c conda-forge gdcm [recommanded]\n"
                "      - or: pip install pylibjpeg pylibjpeg-libjpeg\n"
                "  • If shapes look like (frames,H,W), use the 3D DICOM dataset.\n"
                "  • To include DERIVED/LOCALIZER series, set allow_derived=True."
            ) from None

        # Ensure consistent spatial size across loaded images
        shapes = {im.shape for im in images}  # shapes like (C, H, W)
        if len(shapes) != 1:
            raise DataLoadError(
                f"Inconsistent image shapes {sorted(shapes)}. "
                "Please resample/crop to uniform (C,H,W)."
            ) from None

        data = np.stack(images, axis=0).astype(np.float32, copy=False)  # (N, C, H, W)
        return data

    def _load_raw_labels(self):
        if self.path_labels is not None and self._use_labels:
            warn_once(
                (
                    f"Labels were requested (use_labels=True, path_labels='{self.path_labels}'), "
                    "but a label loader is not provided by default.\n"
                    "→ Labels will be ignored for this run.\n"
                    "To enable labels, implement `_load_raw_labels(self)` in "
                    " `sim_toolkit/datasets/dcm.py` for `DicomDataset2D` and return a NumPy "
                    "array of shape (N,) or (N, K) aligned with your images.\n"
                    "Set `use_labels=False` or `path_labels=None` to silence this warning."
                ),
                key="dcm2D.labels.unimplemented",
            )
        pass

# ===============================
#           3D DATASET
# ===============================
class DicomDataset3D(BaseDataset):
    """
    Groups .dcm files by SeriesInstanceUID into 3D volumes.
    Returns data with shape (N, C, H, W, D), where C=1 (intensity).
    Designed for folders that contain mixed series; each series becomes one sample.

    Notes:
      - This class is folder-based (non-BIDS). If you have BIDS-wrapped DICOM (rare), use a dedicated indexer.
      - We skip derived/localizer series by default (_is_derived).
    """

    def __init__(self, path_data: str, recursive: bool = True, normalize: bool = False, **kwargs):
        self.recursive = recursive
        self.normalize = normalize
        super().__init__(path_data=path_data, **kwargs)

    def _load_files(self):
        pattern = "**/*.dcm" if self.recursive else "*.dcm"
        paths = sorted(glob(os.path.join(self.path_data, pattern), recursive=self.recursive))
        if not paths:
            raise DataLoadError(
                f"No DICOM files (.dcm) found in {os.path.abspath(self.path_data)}"
            ) from None

        # 1) read all slices, bucket by SeriesInstanceUID
        series_pixels: Dict[str, List[np.ndarray]] = defaultdict(list)
        series_meta: Dict[str, List[dict]] = defaultdict(list)
        dropped = 0

        for p in paths:
            try:
                arr, meta = _read_dicom_pixel(p)
            except Exception:
                dropped += 1
                continue

            sid = meta.get("SeriesInstanceUID")
            if sid is None:
                dropped += 1
                continue
            if _is_derived(meta):
                continue

            series_pixels[sid].append(arr)
            series_meta[sid].append(meta)

        if not series_pixels:
            raise DataLoadError(
                "No valid DICOM series found (after filtering derived/localizer)."
            ) from None

        # 2) sort and stack each series into (H, W, D)
        volumes: List[np.ndarray] = []
        for sid, slices in series_pixels.items():
            metas = series_meta[sid]
            order = _sort_slices(metas)

            if len(order) != len(slices) or len(slices) == 0:
                continue

            sorted_slices = [slices[i] for i in order]

            # Ensure all slices have same size
            hs = {s.shape for s in sorted_slices}
            if len(hs) != 1:
                # Skip inconsistent series
                continue

            vol = np.stack(sorted_slices, axis=-1).astype(np.float32, copy=False)  # (H, W, D)

            if self.normalize:
                vmin, vmax = float(vol.min()), float(vol.max())
                if vmax > vmin:
                    vol = (vol - vmin) / (vmax - vmin)

            volumes.append(vol[None, ...])  # (1, H, W, D)

        if not volumes:
            raise DataLoadError(
                "No stackable DICOM series with consistent geometry."
            ) from None

        # 3) ensure consistent spatial sizes across series (H, W, D)
        shapes = {v.shape for v in volumes}
        if len(shapes) != 1:
            raise DataLoadError(
                f"Inconsistent volume shapes across series: {sorted(shapes)}. "
                "Please resample/crop to uniform (H,W,D)."
            ) from None

        data = np.stack(volumes, axis=0).astype(np.float32, copy=False)  # (N, 1, H, W, D)
        return data

    def _load_raw_labels(self):
        if self.path_labels is not None and self._use_labels:
            warn_once(
                (
                    f"Labels were requested (use_labels=True, path_labels='{self.path_labels}'), "
                    "but a label loader is not provided by default.\n"
                    "→ Labels will be ignored for this run.\n"
                    "To enable labels, implement `_load_raw_labels(self)` in "
                    " `sim_toolkit/datasets/dcm.py` for `DicomDataset3D` and return a NumPy "
                    "array of shape (N,) or (N, K) aligned with your images.\n"
                    "Set `use_labels=False` or `path_labels=None` to silence this warning."
                ),
                key="dcm3D.labels.unimplemented",
            )
        pass
