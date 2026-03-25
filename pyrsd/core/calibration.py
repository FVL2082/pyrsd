"""
pyrsd/core/calibration.py
extracts data for hue displacement calibration curve from sequential filter calibration images
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from pyrsd.utils.io import (find_images, load_image, image_to_hue_field, load_json, sequence_number)

def mean_hue_in_roi(hue_field: np.ndarray, roi_size: int) -> float:
    h, w = hue_field.shape[:2]
    mid = roi_size//2
    cy, cx = h//2, w//2
    top, bottom = max(0,cy-mid), min(h,cy+mid)
    left, right = max(0,cx-mid), min(w,cx+mid) 
    roi = hue_field[top:bottom, left:right]
    valid = roi[~np.isnan(roi)]
    if valid.size == 0:
        raise ValueError("No valid pixels in ROI")
    angles = np.deg2rad(valid)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float(np.rad2deg(mean_angle) % 360)

def build_calibration_data(image_folder: str, step_size_mm: float, roi_size: int) -> list[dict]:
    """loads images and returns hue and displacement value data"""
    files = find_images(image_folder)
    results = []
    for path in files:
        seq = sequence_number(path)
        if seq is None:
            continue
        img = load_image(path)
        hue = image_to_hue_field(img)
        mean_hue = mean_hue_in_roi(hue, roi_size)
        displacement = float((seq) * step_size_mm)
        results.append({"hue":mean_hue,"displacement_mm":displacement})
    return results

def fit_spline(hue: np.ndarray, displacement: np.ndarray, fit: str = "spline", hue_min: float = 0.0, hue_max: float = 360.0):
    """performs curve fitting using univariate spline"""
    if len(hue) < 2:
        raise RuntimeError(f"At least 2 calibration are required")
    
    valid = (hue >= hue_min) & (hue <= hue_max)
    hue, displacement = hue[valid], displacement[valid]

    if len(hue) < 2:
        raise ValueError(f"Fewer than 2 points remain after trimming to hue range [{hue_min}, {hue_max}]. Check your hue_min and hue_max values.")

    order = np.argsort(hue)
    hue, displacement = hue[order], displacement[order]

    kwargs = {"linear":{"k":1,"s":0},"cubic":{"k":3,"s":0},"spline":{"k":3}}
    if fit not in kwargs:
        raise ValueError(f"fit must be 'linear', 'cubic', or 'spline'. {fit} is not recognized")
    
    return UnivariateSpline(hue, displacement, **kwargs[fit])

def build_calibration_json(data: list[dict], image_folder: str, step_size_mm: float, roi_size: int, hue_min: float, hue_max: float, fit: str) -> dict:
    """builds content of calibration json file"""
    return {
        "header": {
            "image_folder": image_folder,
            "step_size_mm": step_size_mm,
            "roi_size_px":  roi_size,
            "n_points":     len(data),
        },
        "valid_hue_range": [hue_min, hue_max],
        "data": data,
    }

def load_spline_from_json(calib_path: str, fit: str = "spline", hue_min: float = 0.0, hue_max: float = 360.0):
    """takes calibration file and loads calibration curve"""
    filter_data = load_json(calib_path)
    
    hue = np.array([d["hue"] for d in filter_data["data"]])
    displacement = np.array([d["displacement_mm"] for d in filter_data["data"]])

    min_hue, max_hue = map(float,filter_data["valid_hue_range"])

    hue_min = max(min_hue,hue_min)
    hue_max = min(max_hue,hue_max)

    return fit_spline(hue, displacement, fit, hue_min, hue_max)
