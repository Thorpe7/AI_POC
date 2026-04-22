"""DICOM preprocessing utilities for MedGemma 3D volume input."""

from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

MAX_SLICES = 85


def load_dicom_series(dicom_dir: Path) -> list[pydicom.Dataset]:
    """Load and sort DICOM files from a directory by instance number."""
    dcm_files = list(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")
    slices = [pydicom.dcmread(f) for f in dcm_files]
    slices.sort(key=lambda s: float(s.InstanceNumber))
    return slices


def sample_equidistant(slices: list, max_slices: int = MAX_SLICES) -> list:
    """Sample slices equidistantly if the volume exceeds max_slices."""
    if len(slices) <= max_slices:
        return slices
    indices = np.linspace(0, len(slices) - 1, max_slices, dtype=int)
    return [slices[i] for i in indices]


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


def preprocess_dicom_series(dicom_dir: Path) -> tuple[list[Image.Image], int, int]:
    """Load a DICOM series and return preprocessed 896x896 PIL images.

    Returns:
        Tuple of (images, total_slices, used_slices) so the caller can
        surface the sampling info to the user.
    """
    slices = load_dicom_series(dicom_dir)
    total_slices = len(slices)

    modality = getattr(slices[0], "Modality", "CT").upper()

    # For MRI, compute volume-wide min/max before sampling
    vol_min, vol_max = 0.0, 0.0
    if modality == "MR":
        all_pixels = np.concatenate([s.pixel_array.flatten() for s in slices])
        vol_min, vol_max = float(all_pixels.min()), float(all_pixels.max())

    sampled = sample_equidistant(slices)

    images = []
    for dcm in sampled:
        pixel_array = dcm.pixel_array

        if modality == "CT":
            intercept = float(getattr(dcm, "RescaleIntercept", 0))
            slope = float(getattr(dcm, "RescaleSlope", 1))
            rgb = apply_ct_windowing(pixel_array, intercept, slope)
        else:
            rgb = apply_mri_normalization(pixel_array, vol_min, vol_max)

        img = Image.fromarray(rgb).resize((896, 896), Image.BILINEAR)
        images.append(img)

    return images, total_slices, len(sampled)
