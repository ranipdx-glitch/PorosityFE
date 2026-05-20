"""Gauss quadrature points and weights."""

from typing import Tuple

import numpy as np

# ============================================================
# SECTION 7c: GAUSS QUADRATURE
# ============================================================

def gauss_points_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """1D Gauss-Legendre points and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of points (1, 2, or 3).

    Returns
    -------
    points : np.ndarray, shape (n,)
    weights : np.ndarray, shape (n,)
    """
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    elif n == 2:
        g = 1.0 / np.sqrt(3.0)
        return np.array([-g, g]), np.array([1.0, 1.0])
    elif n == 3:
        g = np.sqrt(3.0 / 5.0)
        return np.array([-g, 0.0, g]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError(f"Only n=1, 2, 3 supported, got n={n}.")


def gauss_points_hex(order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """3D Gauss-Legendre quadrature for a hexahedron [-1,1]^3.

    Parameters
    ----------
    order : int
        Points per axis (default 2 for 2x2x2 = 8 points).

    Returns
    -------
    points : np.ndarray, shape (n_points, 3)
    weights : np.ndarray, shape (n_points,)
    """
    pts_1d, wts_1d = gauss_points_1d(order)
    xi, eta, zeta = np.meshgrid(pts_1d, pts_1d, pts_1d, indexing='ij')
    wi, wj, wk = np.meshgrid(wts_1d, wts_1d, wts_1d, indexing='ij')
    points = np.column_stack([xi.ravel(), eta.ravel(), zeta.ravel()])
    weights = (wi * wj * wk).ravel()
    return points, weights
