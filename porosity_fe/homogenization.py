"""Mori-Tanaka homogenization, MT-cache, and CLT effective moduli."""

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

from .materials import MaterialProperties
from .transforms import rotate_stiffness_3d
from .void_geometry import VOID_SHAPES

# ============================================================
# SECTION 7d: MT effective stiffness + LRU cache + composite degradation
# ============================================================

_MT_CACHE_MAXSIZE = 4096
_mt_cache: 'OrderedDict[tuple, np.ndarray]' = OrderedDict()


def _mt_cache_key(C_m: np.ndarray, Vp: float, void_shape_radii: Tuple,
                  nu_m: float) -> tuple:
    # An isotropic 6x6 stiffness is fully determined by (lam+2mu, mu); use
    # both diagonal slots as a content fingerprint so different materials
    # never collide. Rounding tolerances are tighter than typical FP noise
    # but loose enough that two physically-identical inputs collapse.
    return (
        round(float(Vp), 6),
        tuple(round(float(r), 6) for r in void_shape_radii),
        round(float(nu_m), 8),
        round(float(C_m[0, 0]), 4),
        round(float(C_m[3, 3]), 4),
    )


def _mt_cache_clear() -> None:
    """Drop the MT effective-stiffness cache. Test/diagnostic helper."""
    _mt_cache.clear()


def _mt_effective_stiffness(C_m: np.ndarray, Vp: float,
                            void_shape_radii: Tuple,
                            nu_m: float) -> np.ndarray:
    """Mori-Tanaka effective stiffness for void inclusions (C_inclusion = 0).

    Memoized by a content-fingerprint cache (#42). The cached result is
    copied before return so callers can mutate it freely.

    Standalone Eshelby-tensor-based Mori-Tanaka calculation for use inside
    Hex8Element.

    The Eshelby tensor is computed in a canonical frame (symmetry axis along
    x_1) and then permuted to align with the actual void axis. The previous
    implementation used ``ar = max/min`` as the aspect ratio, which is
    always >= 1, leaving the ``else: # Oblate`` branch unreachable — oblate
    voids (penny shape) were silently routed to the prolate formulas. See
    issue #32.

    Parameters
    ----------
    C_m : np.ndarray
        Shape (6, 6) isotropic matrix stiffness.
    Vp : float
        Void volume fraction (0 to 1).
    void_shape_radii : tuple
        (a1, a2, a3) radii defining void shape for Eshelby tensor.
    nu_m : float
        Matrix Poisson's ratio.

    Returns
    -------
    np.ndarray
        Shape (6, 6) effective stiffness matrix.
    """
    if Vp < 1e-12:
        return C_m.copy()
    if Vp > 0.99:
        return np.zeros((6, 6))

    # Cache check (#42). The compute path below is ~200x more expensive
    # than the fingerprint+lookup, so the hit case is essentially free.
    cache_key = _mt_cache_key(C_m, Vp, void_shape_radii, nu_m)
    cached = _mt_cache.get(cache_key)
    if cached is not None:
        _mt_cache.move_to_end(cache_key)  # LRU touch
        return cached.copy()

    nu = nu_m
    r = list(void_shape_radii)
    sphere_tol = 0.01

    S = np.zeros((6, 6))

    if (max(r) - min(r)) / max(r) < sphere_tol:
        # All three radii within 1% — treat as sphere.
        S[0, 0] = S[1, 1] = S[2, 2] = (7 - 5 * nu) / (15 * (1 - nu))
        S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = \
            (5 * nu - 1) / (15 * (1 - nu))
        S[3, 3] = S[4, 4] = S[5, 5] = (4 - 5 * nu) / (15 * (1 - nu))
    else:
        # Axisymmetric: find the symmetry axis (the radius that differs
        # from the other two equal radii).
        def _close(a, b):
            return abs(a - b) / max(a, b) < sphere_tol

        if _close(r[1], r[2]):
            idx_axis = 0  # a_1 is the unique axis
        elif _close(r[0], r[2]):
            idx_axis = 1
        elif _close(r[0], r[1]):
            idx_axis = 2
        else:
            # Triaxial: no axisymmetric closed form. Approximate by
            # treating the largest axis as the symmetry axis (prolate
            # fallback). Documented limitation; acceptable because the
            # default VOID_SHAPES are all axisymmetric.
            idx_axis = r.index(max(r))

        a_axis = r[idx_axis]
        a_eq = r[(idx_axis + 1) % 3]  # equatorial radius
        alpha = a_axis / a_eq
        a2 = alpha ** 2

        # g-function with correct branch:
        #   prolate (alpha > 1): cosh-based form
        #   oblate  (alpha < 1): arccos-based form
        # The S-tensor formulas below are identical in structure for both;
        # only the g-function and the sign of (alpha^2 - 1) differ.
        if alpha > 1.0:
            g = alpha / (a2 - 1) ** 1.5 * (
                alpha * np.sqrt(a2 - 1) - np.arccosh(alpha))
        else:
            g = alpha / (1 - a2) ** 1.5 * (
                np.arccos(alpha) - alpha * np.sqrt(1 - a2))

        # Eshelby tensor in canonical frame (symmetry axis along x_1).
        S[0, 0] = (1.0 / (2 * (1 - nu))) * (
            1 - 2 * nu + (3 * a2 - 1) / (a2 - 1) - (1 - 2 * nu + 3 * a2 / (a2 - 1)) * g)
        S[1, 1] = S[2, 2] = (3.0 / (8 * (1 - nu))) * a2 / (a2 - 1) + \
            (1.0 / (4 * (1 - nu))) * (1 - 2 * nu - 9.0 / (4 * (a2 - 1))) * g
        S[0, 1] = S[0, 2] = -(1.0 / (2 * (1 - nu))) * a2 / (a2 - 1) + \
            (1.0 / (4 * (1 - nu))) * (3 * a2 / (a2 - 1) - (1 - 2 * nu)) * g
        S[1, 0] = S[2, 0] = -(1.0 / (2 * (1 - nu))) * 1.0 / (a2 - 1) + \
            (1.0 / (4 * (1 - nu))) * (3.0 / (a2 - 1) - (1 - 2 * nu)) * g
        S[1, 2] = S[2, 1] = (1.0 / (4 * (1 - nu))) * (
            a2 / (2 * (a2 - 1)) - (1 - 2 * nu + 3.0 / (4 * (a2 - 1))) * g)
        S[3, 3] = (1.0 / (4 * (1 - nu))) * (
            a2 / (2 * (a2 - 1)) + (1 - 2 * nu - 3.0 / (4 * (a2 - 1))) * g)
        S[4, 4] = S[5, 5] = (1.0 / (4 * (1 - nu))) * (
            1 - 2 * nu - (a2 + 1) / (a2 - 1) -
            0.5 * (1 - 2 * nu - 3 * (a2 + 1) / (a2 - 1)) * g)

        # Permute the Voigt tensor to align the symmetry axis with the
        # actual unique-radius axis. For a swap x_1 <-> x_k the Voigt
        # permutation is:
        #   x_1 <-> x_2: [1, 0, 2, 4, 3, 5]
        #   x_1 <-> x_3: [2, 1, 0, 5, 4, 3]
        if idx_axis != 0:
            if idx_axis == 1:
                perm = [1, 0, 2, 4, 3, 5]
            else:
                perm = [2, 1, 0, 5, 4, 3]
            P = np.eye(6)[perm]
            S = P @ S @ P.T

    I6 = np.eye(6)
    inner = I6 - (1 - Vp) * S
    # The Mori-Tanaka concentration tensor `inner` becomes singular as Vp -> 1
    # for high-aspect-ratio (oblate) voids even before the Vp > 0.99 early-out
    # above is reached. Falling back to pinv keeps the result finite instead of
    # propagating NaN/inf through the assembled stiffness.
    try:
        inner_inv = np.linalg.inv(inner)
    except np.linalg.LinAlgError:
        inner_inv = np.linalg.pinv(inner)
    C_eff = C_m @ (I6 - Vp * inner_inv)
    if not np.all(np.isfinite(C_eff)):
        # Last-ditch: return the void-saturated zero stiffness rather than
        # silently emitting NaN/inf from a near-singular inner. Skip the
        # cache for this path — it's a defensive fallback, not a result
        # we want to look up later.
        return np.zeros((6, 6))

    # Store in the LRU cache (#42); evict the oldest entry if full. Cache
    # a defensive copy so callers can mutate the returned array.
    _mt_cache[cache_key] = C_eff.copy()
    if len(_mt_cache) > _MT_CACHE_MAXSIZE:
        _mt_cache.popitem(last=False)
    return C_eff


def _degraded_composite_stiffness(Vp: float, void_shape_radii: Tuple,
                                  mat: 'MaterialProperties') -> np.ndarray:
    """Build porosity-degraded composite stiffness via component-wise micromechanics.

    Rather than applying a single scalar degradation ratio, this function:
    1. Uses Mori-Tanaka to get degraded matrix modulus E_m* and G_m*
    2. Re-applies rule-of-mixtures with degraded matrix to get degraded
       composite engineering constants (E11*, E22*, G12*, etc.)
    3. Builds the full 6x6 stiffness from those degraded constants.

    This correctly captures that porosity (matrix voids) barely affects
    fiber-dominated E11 but strongly degrades matrix-dominated E22 and G12.

    Parameters
    ----------
    Vp : float
        Local void volume fraction (0 to 1).
    void_shape_radii : tuple
        (a1, a2, a3) for Eshelby tensor.
    mat : MaterialProperties
        Material with constituent properties (E_f, E_m, V_f, etc.).

    Returns
    -------
    np.ndarray
        Shape (6, 6) degraded composite stiffness in material coordinates.
    """
    if Vp < 1e-12:
        return mat.get_stiffness_matrix()
    if Vp > 0.99:
        return np.zeros((6, 6))

    E_m = mat.matrix_modulus
    nu_m = mat.matrix_poisson
    G_m = E_m / (2.0 * (1.0 + nu_m))
    E_f = mat.fiber_modulus
    Vf = mat.fiber_volume_fraction
    Vm = 1.0 - Vf

    # --- Step 1: Degraded matrix properties from Mori-Tanaka ---
    C_m = mat.get_isotropic_matrix_stiffness()
    C_eff = _mt_effective_stiffness(C_m, Vp, void_shape_radii, nu_m)

    # Extract degraded isotropic matrix moduli
    mu_eff = C_eff[3, 3]
    lam_eff = C_eff[0, 1]
    denom = lam_eff + mu_eff
    G_m_eff = max(mu_eff, 1.0)
    E_m_eff = mu_eff * (3.0 * lam_eff + 2.0 * mu_eff) / denom if denom > 1e-12 else 1.0
    E_m_eff = max(E_m_eff, 1.0)
    nu_m_eff = lam_eff / (2.0 * denom) if denom > 1e-12 else nu_m

    # --- Step 2: Compute degradation RATIOS via micromechanics ---
    # Use Halpin-Tsai with pristine and degraded matrix to get ratios,
    # then apply ratios to actual measured composite properties.
    # This avoids mismatch between micromechanics predictions and actual data.

    nu_f = 0.2  # typical carbon fiber Poisson's ratio
    G_f = E_f / (2.0 * (1.0 + nu_f))

    # E11 ratio (Rule of Mixtures — fiber-dominated, barely affected)
    E11_rom_prist = Vf * E_f + Vm * E_m
    E11_rom_deg = Vf * E_f + Vm * E_m_eff
    r_E11 = E11_rom_deg / E11_rom_prist  # ~0.999 for 5% porosity

    # E22 ratio (Halpin-Tsai — matrix-dominated, strongly affected)
    xi_HT = 2.0
    def _halpin_tsai(Ef, Em, xi, vf):
        ratio = Ef / Em
        eta = (ratio - 1.0) / (ratio + xi)
        return Em * (1.0 + xi * eta * vf) / (1.0 - eta * vf)

    E22_HT_prist = _halpin_tsai(E_f, E_m, xi_HT, Vf)
    E22_HT_deg = _halpin_tsai(E_f, E_m_eff, xi_HT, Vf)
    r_E22 = E22_HT_deg / E22_HT_prist

    # G12 ratio (Halpin-Tsai — matrix-dominated)
    xi_G = 1.0
    G12_HT_prist = _halpin_tsai(G_f, G_m, xi_G, Vf)
    G12_HT_deg = _halpin_tsai(G_f, G_m_eff, xi_G, Vf)
    r_G12 = G12_HT_deg / G12_HT_prist

    # G23 ratio (Halpin-Tsai — matrix-dominated)
    G23_HT_prist = _halpin_tsai(G_f, G_m, xi_G, Vf)
    G23_HT_deg = _halpin_tsai(G_f, G_m_eff, xi_G, Vf)
    r_G23 = G23_HT_deg / G23_HT_prist

    # nu12 ratio (Rule of Mixtures — weakly affected)
    nu12_rom_prist = Vf * nu_f + Vm * nu_m
    nu12_rom_deg = Vf * nu_f + Vm * nu_m_eff
    r_nu12 = nu12_rom_deg / nu12_rom_prist if abs(nu12_rom_prist) > 1e-12 else 1.0

    # --- Step 3: Apply ratios to actual measured properties ---
    E11_deg = mat.E11 * r_E11
    E22_deg = mat.E22 * r_E22
    E33_deg = mat.E33 * r_E22  # same as E22 (transverse isotropy)
    G12_deg = mat.G12 * r_G12
    G13_deg = mat.G13 * r_G12  # same as G12
    G23_deg = mat.G23 * r_G23
    nu12_deg = mat.nu12 * r_nu12
    nu13_deg = mat.nu13 * r_nu12
    nu23_deg = mat.nu23  # weakly affected, keep constant

    # --- Step 4: Build compliance matrix from degraded constants ---
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E11_deg
    S[1, 1] = 1.0 / E22_deg
    S[2, 2] = 1.0 / E33_deg
    S[0, 1] = S[1, 0] = -nu12_deg / E11_deg
    S[0, 2] = S[2, 0] = -nu13_deg / E11_deg
    S[1, 2] = S[2, 1] = -nu23_deg / E22_deg
    S[3, 3] = 1.0 / G23_deg
    S[4, 4] = 1.0 / G13_deg
    S[5, 5] = 1.0 / G12_deg

    return np.linalg.inv(S)


# ============================================================
# CLT effective moduli (uses _degraded_composite_stiffness from above)
# ============================================================

def compute_clt_effective_modulus(material: MaterialProperties,
                                  ply_angles: List[float]) -> float:
    """Compute effective laminate longitudinal modulus using CLT (ABD matrix).

    Builds the full A-matrix (in-plane stiffness) from Classical Lamination
    Theory, then computes the effective Ex from the A-matrix inverse.

    Parameters
    ----------
    material : MaterialProperties
        Material with orthotropic ply-level properties.
    ply_angles : list of float
        Ply orientation angles in degrees (one per ply).

    Returns
    -------
    float
        Effective longitudinal modulus E_x (MPa).
    """
    C_base = material.get_stiffness_matrix()
    n_plies = len(ply_angles)
    t_ply = material.t_ply
    h_total = n_plies * t_ply

    # Build A-matrix: A_ij = sum over plies of Q_bar_ij * t_ply
    A = np.zeros((3, 3))
    for angle_deg in ply_angles:
        angle_rad = np.radians(float(angle_deg))
        if abs(angle_rad) > 1e-15:
            C_rot = rotate_stiffness_3d(C_base, angle_rad, axis='z')
        else:
            C_rot = C_base
        # Extract in-plane Q-bar (reduced stiffness) from 6x6:
        # Q_bar = C_rot[0:3, 0:3] is the membrane portion in plane-stress
        # For CLT, use the plane-stress reduced stiffness
        # Q_bar_ij = C_ij - C_i3*C_j3/C_33  (i,j = 1,2,6 -> indices 0,1,5)
        idx = [0, 1, 5]  # 11, 22, 12 in Voigt
        Q_bar = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ii, jj = idx[i], idx[j]
                if abs(C_rot[2, 2]) > 1e-12:
                    Q_bar[i, j] = C_rot[ii, jj] - C_rot[ii, 2] * C_rot[jj, 2] / C_rot[2, 2]
                else:
                    Q_bar[i, j] = C_rot[ii, jj]
        A += Q_bar * t_ply

    # Effective modulus: E_x = (A11*A22 - A12^2) / (A22 * h)
    # From a_ij = A_inv, E_x = 1 / (h * a_11)
    A_inv = np.linalg.inv(A)
    E_x = 1.0 / (h_total * A_inv[0, 0])
    return float(E_x)


def _build_clt_abd(material: MaterialProperties, ply_angles: List[float],
                   C_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build CLT A (membrane) and D (bending) matrices from a 6x6 stiffness."""
    n_plies = len(ply_angles)
    t_ply = material.t_ply
    h_total = n_plies * t_ply

    A_mat = np.zeros((3, 3))
    D_mat = np.zeros((3, 3))

    idx = [0, 1, 5]  # 11, 22, 12 in Voigt
    z_k_prev = -h_total / 2.0

    for angle_deg in ply_angles:
        z_k = z_k_prev + t_ply
        z_mid = (z_k_prev + z_k) / 2.0

        angle_rad = np.radians(float(angle_deg))
        if abs(angle_rad) > 1e-15:
            C_rot = rotate_stiffness_3d(C_base, angle_rad, axis='z')
        else:
            C_rot = C_base

        # Plane-stress reduced stiffness Q_bar
        Q_bar = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ii, jj = idx[i], idx[j]
                if abs(C_rot[2, 2]) > 1e-12:
                    Q_bar[i, j] = C_rot[ii, jj] - C_rot[ii, 2] * C_rot[jj, 2] / C_rot[2, 2]
                else:
                    Q_bar[i, j] = C_rot[ii, jj]

        A_mat += Q_bar * t_ply
        D_mat += Q_bar * (t_ply * z_mid**2 + t_ply**3 / 12.0)

        z_k_prev = z_k

    return A_mat, D_mat


def compute_degraded_clt_moduli(material: MaterialProperties,
                                ply_angles: List[float],
                                Vp: float,
                                method: str = 'mori_tanaka') -> Dict[str, float]:
    """Compute effective laminate in-plane moduli (Ex, Ey, Gxy) with porosity.

    Uses Mori-Tanaka-degraded ply stiffness via _degraded_composite_stiffness,
    then builds the CLT A-matrix to extract effective moduli.

    Parameters
    ----------
    material : MaterialProperties
    ply_angles : list of float  (degrees)
    Vp : float  void volume fraction (0–1)
    method : str  ignored (kept for API symmetry); always uses Mori-Tanaka.

    Returns
    -------
    dict with keys: 'Ex', 'Ey', 'Gxy' (all in MPa)
    """
    void_shape_radii = VOID_SHAPES['spherical']  # spherical default
    C_deg = _degraded_composite_stiffness(Vp, void_shape_radii, material)

    n_plies = len(ply_angles)
    t_ply = material.t_ply
    h_total = n_plies * t_ply

    A_mat, _ = _build_clt_abd(material, ply_angles, C_deg)

    a_inv = np.linalg.inv(A_mat)
    Ex = 1.0 / (h_total * a_inv[0, 0])
    Ey = 1.0 / (h_total * a_inv[1, 1])
    Gxy = 1.0 / (h_total * a_inv[2, 2])

    return {'Ex': float(Ex), 'Ey': float(Ey), 'Gxy': float(Gxy)}


def compute_degraded_clt_flexural_modulus(material: MaterialProperties,
                                          ply_angles: List[float],
                                          Vp: float,
                                          method: str = 'mori_tanaka') -> Dict[str, float]:
    """Compute effective laminate flexural modulus Ef_x with porosity.

    Uses the CLT D-matrix (bending stiffness) to compute the flexural modulus.

    Parameters
    ----------
    material : MaterialProperties
    ply_angles : list of float  (degrees)
    Vp : float  void volume fraction (0–1)
    method : str  ignored; always uses Mori-Tanaka.

    Returns
    -------
    dict with key: 'Ef_x' (MPa)
    """
    void_shape_radii = VOID_SHAPES['spherical']
    C_deg = _degraded_composite_stiffness(Vp, void_shape_radii, material)

    n_plies = len(ply_angles)
    t_ply = material.t_ply
    h_total = n_plies * t_ply

    _, D_mat = _build_clt_abd(material, ply_angles, C_deg)

    d_inv = np.linalg.inv(D_mat)
    # Flexural modulus: Ef_x = 12 / (h^3 * d_11_inv)
    Ef_x = 12.0 / (h_total**3 * d_inv[0, 0])

    return {'Ef_x': float(Ef_x)}
