"""Coordinate transforms (3D rotations of stress / strain / stiffness)."""

import numpy as np

# ============================================================
# SECTION 7b: COORDINATE TRANSFORMS
# ============================================================

def rotation_matrix_3d(angle_rad: float, axis: str = 'z') -> np.ndarray:
    """3x3 rotation matrix for rotation about a principal axis.

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.
    axis : str
        'z' for ply orientation, 'y' for wrinkle/waviness misalignment.

    Returns
    -------
    np.ndarray
        Shape (3, 3) rotation matrix.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    if axis == 'z':
        return np.array([
            [ c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])
    elif axis == 'y':
        return np.array([
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0,  c],
        ])
    else:
        raise ValueError(f"Unsupported axis '{axis}'. Use 'z' or 'y'.")


def stress_transformation_3d(angle_rad: float, axis: str = 'z') -> np.ndarray:
    """6x6 stress transformation matrix in Voigt notation.

    Voigt ordering: [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.
    axis : str
        'z' or 'y'.

    Returns
    -------
    np.ndarray
        Shape (6, 6) stress transformation matrix.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    c2 = c * c
    s2 = s * s
    sc = s * c

    if axis == 'z':
        return np.array([
            [ c2,   s2,  0.0,  0.0,  0.0,  2.0 * sc],
            [ s2,   c2,  0.0,  0.0,  0.0, -2.0 * sc],
            [0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,    c,   -s,  0.0],
            [0.0,  0.0,  0.0,    s,    c,  0.0],
            [-sc,   sc,  0.0,  0.0,  0.0,  c2 - s2],
        ])
    elif axis == 'y':
        return np.array([
            [ c2,  0.0,  s2,  0.0, -2.0 * sc, 0.0],
            [0.0,  1.0, 0.0,  0.0,  0.0,      0.0],
            [ s2,  0.0,  c2,  0.0,  2.0 * sc,  0.0],
            [0.0,  0.0, 0.0,    c,  0.0,         s],
            [ sc,  0.0, -sc,  0.0,  c2 - s2,   0.0],
            [0.0,  0.0, 0.0,   -s,  0.0,         c],
        ])
    else:
        raise ValueError(f"Unsupported axis '{axis}'. Use 'z' or 'y'.")


def strain_transformation_3d(angle_rad: float, axis: str = 'z') -> np.ndarray:
    """6x6 engineering strain transformation matrix.

    Related to stress transformation via the Reuter matrix:
        T_epsilon = R @ T_sigma @ R_inv

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.
    axis : str
        'z' or 'y'.

    Returns
    -------
    np.ndarray
        Shape (6, 6) strain transformation matrix.
    """
    T_sigma = stress_transformation_3d(angle_rad, axis=axis)
    R = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    R_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    return R @ T_sigma @ R_inv


def rotate_stiffness_3d(C: np.ndarray, angle_rad: float, axis: str = 'z') -> np.ndarray:
    """Rotate a 6x6 stiffness matrix to a new coordinate system.

    C_bar = T_sigma_inv @ C @ T_epsilon

    Parameters
    ----------
    C : np.ndarray
        Shape (6, 6) stiffness matrix in material coordinates.
    angle_rad : float
        Rotation angle in radians.
    axis : str
        'z' or 'y'.

    Returns
    -------
    np.ndarray
        Shape (6, 6) rotated stiffness matrix.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError(f"Stiffness matrix must be 6x6, got {C.shape}.")
    T_sigma = stress_transformation_3d(angle_rad, axis=axis)
    T_epsilon = strain_transformation_3d(angle_rad, axis=axis)
    T_sigma_inv = np.linalg.inv(T_sigma)
    return T_sigma_inv @ C @ T_epsilon
