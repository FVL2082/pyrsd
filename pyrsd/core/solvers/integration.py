"""
pyrsd/core/solvers/integration.py
performs 1D integration in the flow field.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

def integrate_1d(gradient_field: np.ndarray, dr_mm: float, ref_value: float, axis: int = 0, ref_side: str = "start", ref_index: int|None = None, bidirectional: bool = False) -> np.ndarray:
      """Performs 1 D culumative integration of a gradient field
      Parameters:
      ------------
      gradient_field: float64 ndarray 
      dr_mm: pixel spacing in mm along the integration axis
      ref_value: anchor value at the reference pixel 
      axis: 0 = integrates along rows (along y axis)
            1 = integrates along columns (along x axis)
      ref_side: 'start' - selects the first valid pixel of each line.
                  'end' - selects the last valid pixel of each line.
                  'index' - anchor at the pixel specified on ref_index.
      ref_index: pixel index used when ref_side='index' 

      Returns: 
      Scalar_Field : float64 ndarray, of same shape as gradient_field
      """

      gradient = np.moveaxis(gradient_field.astype(np.float64), axis, 0)
      result = np.full_like(gradient, np.nan)

      for col in range(gradient.shape[1]):
            line = gradient[:,col]
            valid_idx = np.where(~np.isnan(line))[0]
            if valid_idx.size == 0:
                  continue

            i_start, i_end = int(valid_idx[0]), int(valid_idx[-1])
            segment = line[i_start:i_end+1].copy()

            nan_mask = np.isnan(segment)
            if nan_mask.any():
                  x = np.arange(len(segment))
                  segment[nan_mask] = np.interp(x[nan_mask],x[~nan_mask],segment[~nan_mask])
            
            integrated = cumulative_trapezoid(segment, dx=dr_mm, initial=0.0)
            if bidirectional:
                  forward = integrated
                  backward = -cumulative_trapezoid(segment[::-1],dx=dr_mm, initial=0)[::-1]
                  integrated = 0.5*(forward+backward)
            seg_len = len(integrated)
            anchor = seg_len-1

            if ref_side == "start":
                  anchor = 0  
            elif ref_side == "index" and ref_index is not None:
                  anchor = max(0, min(ref_index - i_start, anchor))

            result[i_start:i_end+1, col] = integrated + (ref_value - integrated[anchor])

      return np.moveaxis(result, 0, axis)
