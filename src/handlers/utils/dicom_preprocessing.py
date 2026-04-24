"""DICOM preprocessing utilities for MedGemma 3D volume input."""

from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

MAX_SLICES = 32  #! must equal VLLMHandler.limit_mm_per_prompt["image"]


def load_dicom_series(dicom_dir: Path) -> list[pydicom.Dataset]:
    """Load and sort DICOM files from a directory by instance number."""
    dcm_files = list(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")
    slices = [pydicom.dcmread(f) for f in dcm_files]
    slices.sort(key=lambda s: float(s.InstanceNumber))
    return slices


def sample_equidistant_indices(n: int, max_slices: int = MAX_SLICES) -> list[int]:
    """Return equidistant indices into a series of length n, capped at max_slices."""
    if n <= max_slices:
        return list(range(n))
    return np.linspace(0, n - 1, max_slices, dtype=int).tolist()


def apply_ct_windowing(pixel_array: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Map raw CT pixel data to RGB using multi-channel HU windowing.

    R: HU -1024 to 1024  (morphological boundaries)
    G: HU -135  to 215   (soft tissue)
    B: HU 0     to 80    (brain parenchyma / hemorrhage)
    """
    hu = pixel_array.astype(np.float32) * slope + intercept

    r = np.clip((hu - (-1024)) / (1024 - (-1024)) * 255, 0, 255)
    g = np.clip((hu - (-135)) / (215 - (-135)) * 255, 0, 255)
    b = np.clip((hu - 0) / (80 - 0) * 255, 0, 255)

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def apply_mri_normalization(
    pixel_array: np.ndarray, vol_min: float, vol_max: float
) -> np.ndarray:
    """Min-max normalize MRI pixel data, identical across R/G/B."""
    if vol_max == vol_min:
        normalized = np.zeros_like(pixel_array, dtype=np.uint8)
    else:
        normalized = np.clip(
            (pixel_array.astype(np.float32) - vol_min) / (vol_max - vol_min) * 255, 0, 255
        ).astype(np.uint8)
    return np.stack([normalized, normalized, normalized], axis=-1)


def preprocess_dicom_series(
    dicom_dir: Path,
    slice_range: tuple[int, int] | None = None,
) -> tuple[list[Image.Image], int, list[int]]:
    """Load a DICOM series and return preprocessed 896x896 PIL images.

    Args:
        dicom_dir: Directory containing .dcm files.
        slice_range: Optional (start, end) 0-indexed inclusive range into the
            sorted series. If omitted, falls back to equidistant sampling.

    Returns:
        Tuple of (images, total_slices, used_indices). used_indices are the
        0-indexed positions in the sorted series that were selected.
    """
    slices = load_dicom_series(dicom_dir)
    total_slices = len(slices)

    modality = getattr(slices[0], "Modality", "CT").upper()

    # For MRI, compute volume-wide min/max over the full series so contrast
    # stays consistent regardless of which slices are selected.
    vol_min, vol_max = 0.0, 0.0
    if modality == "MR":
        all_pixels = np.concatenate([s.pixel_array.flatten() for s in slices])
        vol_min, vol_max = float(all_pixels.min()), float(all_pixels.max())

    if slice_range is not None:
        start, end = slice_range
        if not (0 <= start <= end < total_slices):
            raise ValueError(
                f"slice_range [{start}, {end}] out of bounds for series with {total_slices} slices"
            )
        count = end - start + 1
        if count > MAX_SLICES:
            raise ValueError(
                f"slice_range [{start}, {end}] selects {count} slices; max is {MAX_SLICES}"
            )
        used_indices = list(range(start, end + 1))
    else:
        used_indices = sample_equidistant_indices(total_slices)

    images = []
    for idx in used_indices:
        dcm = slices[idx]
        pixel_array = dcm.pixel_array

        if modality == "CT":
            intercept = float(getattr(dcm, "RescaleIntercept", 0))
            slope = float(getattr(dcm, "RescaleSlope", 1))
            rgb = apply_ct_windowing(pixel_array, intercept, slope)
        else:
            rgb = apply_mri_normalization(pixel_array, vol_min, vol_max)

        img = Image.fromarray(rgb).resize((896, 896), Image.BILINEAR)
        images.append(img)

    return images, total_slices, used_indices
