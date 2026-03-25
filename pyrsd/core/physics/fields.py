"""
pyrsd/core/physics/fields.py
converts refractive index gradient fields to physical quantities.
density, temperature, pressure.
"""

import numpy as np
from pyrsd.core.physics.optics import displacement_to_deflection, displacement_to_gradient
from pyrsd.core.solvers.integration import integrate_1d
from pyrsd.core.solvers.poisson import poisson_sparse, poisson_iterative
from pyrsd.core.solvers.abel import inverse_abel

def density_from_gradient_1d(displacement: np.ndarray, dr: float, ref_rho: float, K: float, n0: float, f2: float, L: float, axis: int = 0, ref_side: str = "start") -> np.ndarray:
    """
    calculates density field from 1D refractive index gradient
    1D integration so reduced accuracy
    """
    grad = displacement_to_gradient(displacement, n0, f2, L)  
    drho_dr = grad/K
    return integrate_1d(drho_dr, dr, ref_rho, axis, ref_side)

def density_from_gradient_2d(disp_x: np.ndarray, disp_y: np.ndarray, dx: float, dy: float, dirichlet_coord: tuple[int,int], ref_rho: float, K: float, n0: float, f2: float, L: float, solver: str = "sparse", tol: float = 1e-8) -> np.ndarray:
    """calculates density field from 2D refractive index gradient using poisson solver"""
    grad_x = displacement_to_gradient(disp_x, n0, f2, L)
    grad_y = displacement_to_gradient(disp_y, n0, f2, L)
    drho_dx = grad_x / K
    drho_dy = grad_y / K
    if solver=="sparse":
        return poisson_sparse(drho_dx,drho_dy,dx,dy,dirichlet_coord,ref_rho)
    else:
        return poisson_iterative(drho_dx,drho_dy,dx,dy,dirichlet_coord,ref_rho,solver,tol)

def density_from_gradient_abel(displacement: np.ndarray, dr: float, ref_rho: float, K: float, n0: float, f2: float, method: str = "hansenlaw", axis: int = 0, symmetry_axis: int|None =None, ref_side: str = "end") -> np.ndarray:
    """calculates density field from 1D displacement field"""
    grad = displacement_to_deflection(displacement, n0, f2)
    drho_dr = inverse_abel(grad, dr, method, axis, symmetry_axis)/K
    return integrate_1d(drho_dr, dr, ref_rho, axis, ref_side)

def temperature_ideal_gas(rho: np.ndarray, P_pa: float = 101325.0, R: float = 287.058) -> np.ndarray:
    """calculates temperature using ideal gas equation"""
    return P_pa / (rho * R)

def temperature_isobaric(rho: np.ndarray, rho_ref: float, T_ref: float) -> np.ndarray:
    """calculates temperature using isobaric relation"""
    return T_ref * rho_ref / rho

def temperature_boussinesq(rho: np.ndarray, rho_ref: float, T_ref: float, beta: float|None=None) -> np.ndarray:
    """calculates temperature using boussinesq approximation"""
    if beta is None:
        beta = 1.0/T_ref
    return T_ref - (rho-rho_ref)/(rho_ref*beta)

def temperature_isentropic(rho: np.ndarray, rho_ref: float, T_ref: float, gamma: float = 1.4) -> np.ndarray:
    """calculates temperature using isentropic relation"""
    return T_ref * (rho / rho_ref) ** (gamma-1)

def pressure_isentropic(rho: np.ndarray, rho_ref: float, P_ref: float, gamma: float = 1.4) -> np.ndarray:
    """calculates pressure using isentropic relation"""
    return P_ref * (rho / rho_ref) ** gamma
