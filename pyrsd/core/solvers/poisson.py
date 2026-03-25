"""
pyrsd/core/solvers/poisson.py
solves 2D Poisson equation to compute scalar field from gradient fields.
"""

import numpy as np
from scipy.sparse import csr_matrix

def assembly(grad_x: np.ndarray, grad_y: np.ndarray, dx: float, dy: float, dirichlet_coords: tuple[int, int], dirichlet_values: float) -> np.ndarray:
    """Assembles the discretized poisson equation into a linear system"""
    if grad_x.shape != grad_y.shape:
        raise ValueError(f"Components of gradient field are not of equal size. {grad_x.shape}!={grad_y.shape}")
    
    h, w  = grad_x.shape
    cx, cy = 1.0/dx**2, 1.0/dy**2

    valid = ~np.isnan(grad_x) & ~np.isnan(grad_y)
    idx = np.full((h,w),-1, dtype=int)
    idx[valid] = np.arange(valid.sum())
    
    RHS = (np.gradient(grad_x, dx, axis=1) + np.gradient(grad_y, dy, axis=0))
    RHS_flat = RHS[valid].copy()

    rows_v, cols_v = np.where(valid)
    data_list, row_list, col_list = [], [], []

    def _off_diag(mask, dr, dc, coeff):
        i = idx[rows_v[mask], cols_v[mask]]
        j = idx[rows_v[mask]+dr, cols_v[mask]+dc]
        row_list.append(i)
        col_list.append(j)
        data_list.append(np.full(mask.sum(), coeff))

    has_N = (rows_v > 0)   & valid[np.maximum(rows_v-1, 0),   cols_v]
    has_S = (rows_v < h-1) & valid[np.minimum(rows_v+1, h-1), cols_v]
    has_W = (cols_v > 0)   & valid[rows_v, np.maximum(cols_v-1, 0)]
    has_E = (cols_v < w-1) & valid[rows_v, np.minimum(cols_v+1, w-1)]

    _off_diag(has_N, -1, 0, cy)
    _off_diag(has_S,  1, 0, cy)
    _off_diag(has_W,  0,-1, cx)
    _off_diag(has_E,  0, 1, cx)  
    
    diag = -((has_N*cy + has_S*cy + has_W*cx + has_E*cx).astype(np.float64))
    i_all = idx[rows_v, cols_v]
    row_list.append(i_all)
    col_list.append(i_all)
    data_list.append(diag)

    A = csr_matrix((np.concatenate(data_list),(np.concatenate(row_list),np.concatenate(col_list))),shape=(valid.sum(),valid.sum()))
    A = A.tolil()

    r, c = dirichlet_coords 
    d_id = idx[r, c]
    if d_id==-1:
        raise ValueError(f"{dirichlet_coords} is outside valid region.")
    A[d_id,:] = 0
    A[d_id,d_id] = 1
    RHS_flat[d_id] = dirichlet_values

    A = A.tocsr()
    return A,RHS_flat,valid

def poisson_sparse(grad_x: np.ndarray, grad_y: np.ndarray, dx: float, dy: float, dirichlet_coords: tuple[int, int], dirichlet_values: float) -> np.ndarray:
    """
    Reconstructs a scalar field from x and y gradient components
    Solves laplacian of phi = partial x of Gx + partial y of Gy with one dirichlet boundary condition
    parameters
    ----------
    grad_x: float64 ndarray x gradient
    grad_y: float64 ndarray y gradient
    dx_mm: spacing in x
    dy_mm: spacing in y
    dirichlet_coord: (row,col) of element with known value
    dirichlet_value: known scalar at dirichlet_coord
    Returns
    Scalar_field: float64 ndarray 
    """
    from scipy.sparse.linalg import spsolve

    A,b,valid=assembly(grad_x,grad_y,dx,dy,dirichlet_coords,dirichlet_values)
    solution = spsolve(A,b)

    phi = np.full(grad_x.shape,np.nan,dtype=np.float64)
    phi[valid] = solution
    return phi

def poisson_iterative(grad_x, grad_y, dx, dy, dirichlet_coords, dirichlet_values, solver="cg", tol=1e-8):
    """
    Reconstructs a scalar field from x and y gradient components
    Solves laplacian of phi = partial x of Gx + partial y of Gy with one dirichlet boundary condition
    parameters
    ----------
    grad_x: float64 ndarray x gradient
    grad_y: float64 ndarray y gradient
    dx_mm: spacing in x
    dy_mm: spacing in y
    dirichlet_coord: (row,col) of element with known value
    dirichlet_value: known scalar at dirichlet_coord
    Returns
    Scalar_field: float64 ndarray 
    """
    A,b,valid=assembly(grad_x,grad_y,dx,dy,dirichlet_coords,dirichlet_values)
    if solver=="cg":
        from scipy.sparse.linalg import cg
        X, info = cg(A,b, rtol=tol)
    elif solver=="bicgstab":
        from scipy.sparse.linalg import bicgstab
        X, info = bicgstab(A,b, rtol=tol)
    else:
        raise RuntimeError(f"{solver} is not supported")
    if info != 0:
        import warnings
        warnings.warn(f"{solver} did not converge. info={info}", RuntimeWarning)

    phi = np.full(grad_x.shape,np.nan,dtype=np.float64)
    phi[valid] = X
    return phi