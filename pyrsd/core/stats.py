"""
pyrsd/core/stats.py
statistical analysis of flow field stacks.
"""
import numpy as np

def ensemble_mean(stack: np.ndarray) -> np.ndarray:
    """returns mean"""
    return np.nanmean(stack, axis=0)

def ensemble_std(stack: np.ndarray) -> np.ndarray:
    """returns standard deviation"""
    return np.nanstd(stack, axis=0)

def turbulence_intensity(stack: np.ndarray) -> np.ndarray:
    """returns tubulence intensity"""
    std = np.nanstd(stack, axis=0)
    mean = np.abs(np.nanmean(stack, axis=0))
    ti = std / mean
    ti[mean < 1e-10] = np.nan
    return ti

def rms_fluctuation(stack: np.ndarray) -> np.ndarray:
    """returns root mean square fluctuations"""
    mean = np.nanmean(stack, axis=0)
    fluctuations = stack - mean
    return np.sqrt(np.nanmean(fluctuations**2, axis=0))

def reynolds_decompose(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """returns mean field and fluctuating field stack"""
    mean = np.nanmean(stack, axis=0)
    fluctuations = stack - mean
    return mean, fluctuations

def spatial_profile(field: np.ndarray, axis: int, reduce: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """returns spatial profile by reducing an axis"""
    if field.ndim < 2:
        raise ValueError("field must be atleast 2D")
    if axis >= field.ndim:
        raise ValueError(f"There are only {field.ndim} axes. Given axis: {axis}")
    if reduce == "mean":
        profile = np.nanmean(field, axis=axis)
    elif reduce == "max":
        profile = np.nanmax(field, axis=axis)
    elif reduce == "min":
        profile = np.nanmin(field, axis=axis)
    else:
        raise ValueError(f"Unsupported reduction '{reduce}'")
    pixel_indices = np.arange(profile.size)
    return pixel_indices, profile

def spatial_correlation(field: np.ndarray, ref_point: tuple[int, int]) -> np.ndarray:
    """returns normalized two point spatial correlation of single field snapshot with a reference pixel"""
    if field.ndim != 2:
        raise ValueError("spatial_correlation expects a 2D field.")    
    ref_row, ref_col = ref_point
    ref = field[ref_row, ref_col]
    if np.isnan(ref):
        raise ValueError(f"reference pixel ({ref_row},{ref_col}) is NaN.")
    mean = float(np.nanmean(field))
    std = float(np.nanstd(field))
    if std == 0:
        return np.zeros_like(field)
    ref_fluct = ref - mean
    return (field - mean) * ref_fluct/(std**2)