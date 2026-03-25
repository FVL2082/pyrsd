"""
pyrsd/core/filters.py
optional image filtering and denoising for displacement fields.
apply before passing to solvers if your images are noisy.
"""

import numpy as np

def gaussian_filter(field: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    invalid = np.isnan(field)
    field_filled = field.copy()
    field_filled[invalid] = 0.0
    field_filled = _gaussian_filter(field_filled, sigma)
    field_filled[invalid] = np.nan
    return field_filled

def median_filter(field: np.ndarray, size: int = 3) -> np.ndarray:
    from scipy.ndimage import median_filter as _median_filter
    invalid = np.isnan(field)
    field_filled = field.copy()
    field_filled[invalid] = 0.0
    field_filled = _median_filter(field_filled, size)
    field_filled[invalid] = np.nan
    return field_filled

def bilateral_filter(field: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    from cv2 import bilateralFilter
    invalid = np.isnan(field)
    field_filled = field.copy()
    field_filled[invalid] = 0.0
    filtered = bilateralFilter(field_filled.astype(np.float32), d, sigma_color, sigma_space)
    result = filtered.astype(np.float64)
    result[invalid] = np.nan
    return result

def tv_denoise(field: np.ndarray, weight: float = 0.1) -> np.ndarray:
    from skimage.restoration import denoise_tv_chambolle
    invalid = np.isnan(field)
    field_filled = field.copy()
    field_filled[invalid] = 0.0
    field_filled = denoise_tv_chambolle(field_filled, weight)
    field_filled[invalid] = np.nan
    return field_filled