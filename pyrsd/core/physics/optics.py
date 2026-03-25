"""
pyrsd/core/physics/optics.py
computes refractive index gradient from delta displacement
"""

import numpy as np

def displacement_to_deflection(displacement: np.ndarray, n0: float, f2: float) -> np.ndarray:
    """takes focal plane displcement field, focal length of decollimating optics, and ambient refractive index returns deflection field"""
    epsilon = displacement / f2
    return epsilon * n0 

def displacement_to_gradient(displacement: np.ndarray, n0: float, f2: float, L: float):
    """takes focal plane displcement field, focal length of decollimating optics, length of test section, and ambient refractive index returns refractive index gradient field"""
    return (displacement / f2) * (n0 / L) 

def setup_constant(n0: float, f2_mm: float, L_mm: float, K: float) -> float:
    """directly computes the setup constant C which can be multiplied with displacement field to get density gradient instead of refractive index gradient"""
    return n0 / (f2_mm * L_mm * K)