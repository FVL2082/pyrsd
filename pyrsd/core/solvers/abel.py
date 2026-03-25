"""
pyrsd/core/solvers/abel.py
inverse Abel transform for axisymmetric flows.
"""
import numpy as np

def inverse_abel(projection: np.ndarray, dr: float, method: str = "hansenlaw", axis: int = 0, symmetry_axis: int | None = None) -> np.ndarray:
    """
    Performs Inverse Abel transform on a 2D projection data
    Parameters:
    ----------
    projection: ndarray 2D projection data
    dr: float Radial spacing (change in distance corresponding to unit change in column of projection matrix)
    method: str Abel inversion method like hansenlaw, basex
    axis: int  Axis corresponding to radial direction 0=along the rows or y axis, 1= along the columns or x axis, normal to symmtery_axis
    symmetry_axis: int|none Axis about which the projection and field is symmetric, normal to axis, if None automatically detected
    
    Returns:
    -------
    ndarray Reconstructed radial field
    """
    try:
        import abel
    except ImportError as e:
        raise RuntimeError("PyAbel not installed.\n pip install PyAbel") from e

    if projection.ndim != 2:
        raise ValueError("Projection must be 2D")

    proj = np.moveaxis(projection.astype(np.float64), axis, 1)
    invalid = np.isnan(proj)
    proj_filled = proj.copy()
    proj_filled[invalid] = 0.0

    if symmetry_axis is None:
        # full projection — axis at centre
        origin = (None, proj_filled.shape[1] // 2)
    elif symmetry_axis == 0:
        # half projection — axis at left edge, use hansenlaw directly
        # PyAbel hansenlaw accepts half-image natively
        origin = "none"
    else:
        origin = (None, symmetry_axis)

    transform = abel.Transform(proj_filled, direction="inverse",
                                method=method, origin=origin)
    result = transform.transform / dr
    result[invalid] = np.nan
    return np.moveaxis(result, 1, axis)