"""Hex8 isoparametric element with porosity degradation."""

from typing import Tuple

import numpy as np

from ..gauss import gauss_points_hex
from ..homogenization import _degraded_composite_stiffness, _mt_effective_stiffness
from ..materials import MaterialProperties
from ..transforms import rotate_stiffness_3d

# ============================================================
# SECTION 7d: HEX8 ELEMENT WITH POROSITY DEGRADATION
# ============================================================

# Natural coordinates of 8 hex nodes
_NODE_COORDS_REF = np.array([
    [-1.0, -1.0, -1.0],  # 0
    [+1.0, -1.0, -1.0],  # 1
    [+1.0, +1.0, -1.0],  # 2
    [-1.0, +1.0, -1.0],  # 3
    [-1.0, -1.0, +1.0],  # 4
    [+1.0, -1.0, +1.0],  # 5
    [+1.0, +1.0, +1.0],  # 6
    [-1.0, +1.0, +1.0],  # 7
], dtype=float)


class Hex8Element:
    """8-node isoparametric hexahedral element with porosity degradation.

    Instead of wrinkle-angle rotation (as in WrinkleFE), this element
    degrades the stiffness matrix at each Gauss point using the local
    porosity via Mori-Tanaka homogenization.

    Parameters
    ----------
    node_coords : np.ndarray
        Shape (8, 3) physical coordinates of the 8 nodes (mm).
    C_base : np.ndarray
        Shape (6, 6) base stiffness matrix (pristine composite).
    ply_angle_deg : float
        Ply orientation angle in degrees.
    node_porosities : np.ndarray
        Shape (8,) porosity volume fraction at each node.
    void_shape_radii : tuple
        (a1, a2, a3) void shape radii for Eshelby tensor.
    nu_m : float
        Matrix Poisson's ratio.
    C_m : np.ndarray
        Shape (6, 6) isotropic matrix stiffness for Mori-Tanaka.
    """

    # Near-zero stiffness for void elements (Pa, not MPa — ~6 orders softer)
    VOID_MODULUS = 1.0  # MPa (effectively zero vs composite E11 ~ 161,000 MPa)

    def __init__(self, node_coords: np.ndarray, C_base: np.ndarray,
                 ply_angle_deg: float, node_porosities: np.ndarray,
                 void_shape_radii: Tuple, nu_m: float,
                 C_m: np.ndarray, is_void: bool = False,
                 material: 'MaterialProperties' = None) -> None:
        self.node_coords = np.asarray(node_coords, dtype=float)
        if self.node_coords.shape != (8, 3):
            raise ValueError(f"node_coords must be (8,3), got {self.node_coords.shape}.")
        self.C_base = np.asarray(C_base, dtype=float)
        self.ply_angle_deg = ply_angle_deg
        self.node_porosities = np.asarray(node_porosities, dtype=float)
        if self.node_porosities.shape != (8,):
            raise ValueError(f"node_porosities must be (8,), got {self.node_porosities.shape}.")
        if not np.all(np.isfinite(self.node_porosities)):
            raise ValueError(
                "node_porosities must be finite; "
                "received NaN/inf values would propagate as NaN through "
                "the assembled stiffness."
            )
        # Allow a small fp overshoot (~1e-9) and clip back into [0, 1]; reject
        # anything beyond that as a clear unit/percent confusion.
        eps = 1e-9
        too_low = self.node_porosities < -eps
        too_high = self.node_porosities > 1.0 + eps
        if np.any(too_low) or np.any(too_high):
            bad = self.node_porosities[too_low | too_high]
            hint = ""
            if np.any(too_high) and np.max(bad) >= 1.0 + 1e-3:
                hint = " (Pass a fraction in [0, 1], not a percent.)"
            raise ValueError(
                f"node_porosities must be a fraction in [0, 1] (per node), "
                f"got out-of-range values {bad.tolist()}.{hint}"
            )
        self.node_porosities = np.clip(self.node_porosities, 0.0, 1.0)
        self.void_shape_radii = void_shape_radii
        self.nu_m = nu_m
        self.C_m = np.asarray(C_m, dtype=float)
        self.material = material
        self.is_void = is_void

        self._gauss_points, self._gauss_weights = gauss_points_hex(order=2)

        # Pre-compute void stiffness (isotropic, near-zero modulus)
        if self.is_void:
            E_void = self.VOID_MODULUS
            nu_void = 0.3
            lam = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
            mu = E_void / (2 * (1 + nu_void))
            self._void_C = np.zeros((6, 6))
            self._void_C[0, 0] = self._void_C[1, 1] = self._void_C[2, 2] = lam + 2 * mu
            self._void_C[0, 1] = self._void_C[0, 2] = self._void_C[1, 0] = lam
            self._void_C[1, 2] = self._void_C[2, 0] = self._void_C[2, 1] = lam
            self._void_C[3, 3] = self._void_C[4, 4] = self._void_C[5, 5] = mu

        # Cache: if all node porosities are the same, pre-compute C_eff once
        self._uniform_porosity = None
        if not self.is_void and np.allclose(self.node_porosities, self.node_porosities[0], atol=1e-12):
            self._uniform_porosity = float(self.node_porosities[0])

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Evaluate 8 trilinear shape functions at natural coordinates.

        Returns
        -------
        np.ndarray
            Shape (8,).
        """
        N = np.empty(8)
        for i in range(8):
            N[i] = (
                0.125
                * (1.0 + _NODE_COORDS_REF[i, 0] * xi)
                * (1.0 + _NODE_COORDS_REF[i, 1] * eta)
                * (1.0 + _NODE_COORDS_REF[i, 2] * zeta)
            )
        return N

    @staticmethod
    def shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Derivatives of shape functions w.r.t. natural coordinates.

        Returns
        -------
        np.ndarray
            Shape (3, 8): dN[i, j] = dN_j / d(xi_i).
        """
        dN = np.empty((3, 8))
        for j in range(8):
            xi_j, eta_j, zeta_j = _NODE_COORDS_REF[j]
            dN[0, j] = 0.125 * xi_j * (1.0 + eta_j * eta) * (1.0 + zeta_j * zeta)
            dN[1, j] = 0.125 * (1.0 + xi_j * xi) * eta_j * (1.0 + zeta_j * zeta)
            dN[2, j] = 0.125 * (1.0 + xi_j * xi) * (1.0 + eta_j * eta) * zeta_j
        return dN

    def jacobian(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Jacobian matrix (3x3) mapping natural to physical coordinates."""
        dN = self.shape_derivatives(xi, eta, zeta)
        return dN @ self.node_coords

    def B_matrix(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Strain-displacement matrix (6x24) in Voigt notation.

        Strain ordering: [eps_11, eps_22, eps_33, gamma_23, gamma_13, gamma_12]
        DOF ordering: [u1x, u1y, u1z, u2x, u2y, u2z, ..., u8x, u8y, u8z]

        Notes
        -----
        Voigt order: ``[11, 22, 33, 23, 13, 12]``. Shear rows produce
        **engineering** strain (``gamma_ij = 2 * eps_ij = du_i/dx_j +
        du_j/dx_i``), which is the convention paired with
        :meth:`MaterialProperties.get_stiffness_matrix` so that ``sigma = C @
        (B @ u)`` is dimensionally consistent. Apply the engineering-strain
        transformation (:func:`strain_transformation_3d`) — not the tensor
        form — when rotating ``B @ u`` between coordinate frames.
        """
        dN_dxi = self.shape_derivatives(xi, eta, zeta)
        J = dN_dxi @ self.node_coords
        J_inv = np.linalg.inv(J)
        dN_dx = J_inv @ dN_dxi  # (3, 8)

        B = np.zeros((6, 24))
        for i in range(8):
            col = 3 * i
            dNi_dx = dN_dx[0, i]
            dNi_dy = dN_dx[1, i]
            dNi_dz = dN_dx[2, i]
            B[0, col] = dNi_dx
            B[1, col + 1] = dNi_dy
            B[2, col + 2] = dNi_dz
            B[3, col + 1] = dNi_dz
            B[3, col + 2] = dNi_dy
            B[4, col] = dNi_dz
            B[4, col + 2] = dNi_dx
            B[5, col] = dNi_dy
            B[5, col + 1] = dNi_dx
        return B

    def _degraded_stiffness(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Compute porosity-degraded and ply-rotated stiffness at a point.

        Steps:
        1. Interpolate porosity at this point from nodal values.
        2. Degrade individual composite engineering constants (E11, E22, G12, etc.)
           via Mori-Tanaka + micromechanics rule-of-mixtures, so that porosity in
           0-degree plies correctly yields different laminate stiffness reduction
           than porosity in 90-degree plies.
        3. Rotate by ply angle about z-axis.

        Returns
        -------
        np.ndarray
            Shape (6, 6) degraded and rotated stiffness.
        """
        # VOID ELEMENTS: use near-zero isotropic stiffness (explicit inclusion)
        if self.is_void:
            return self._void_C

        # NON-VOID ELEMENTS: degrade by distributed microporosity via Mori-Tanaka
        # 1. Interpolate porosity at this Gauss point
        if self._uniform_porosity is not None:
            Vp = self._uniform_porosity
        else:
            N = self.shape_functions(xi, eta, zeta)
            Vp = float(N @ self.node_porosities)
        Vp = max(0.0, min(Vp, 0.99))

        # 2. Component-wise degradation: degrade E11, E22, G12, etc. individually
        #    This correctly captures that E11 (fiber-dominated) is barely affected
        #    while E22/G12 (matrix-dominated) are strongly reduced by porosity.
        if self.material is not None:
            C_degraded = _degraded_composite_stiffness(
                Vp, self.void_shape_radii, self.material)
        else:
            # Fallback: scalar degradation (legacy behavior)
            C_eff_mt = _mt_effective_stiffness(self.C_m, Vp, self.void_shape_radii, self.nu_m)
            diag_pristine = np.diag(self.C_m)
            diag_degraded = np.diag(C_eff_mt)
            mask = diag_pristine > 1e-12
            if np.any(mask):
                avg_ratio = np.mean(diag_degraded[mask] / diag_pristine[mask])
            else:
                avg_ratio = 1.0
            avg_ratio = max(0.0, min(avg_ratio, 1.0))
            C_degraded = self.C_base * avg_ratio

        # 3. Rotate by ply angle
        ply_rad = np.radians(self.ply_angle_deg)
        if abs(ply_rad) > 1e-15:
            C_degraded = rotate_stiffness_3d(C_degraded, ply_rad, axis='z')

        return C_degraded

    def stiffness_matrix(self) -> np.ndarray:
        """Element stiffness matrix (24x24) via 2x2x2 Gauss quadrature.

        ``Ke = sum over GPs of: B^T @ C_bar @ B * det(J) * w``

        Raises
        ------
        ValueError
            If the Jacobian determinant is non-positive at any Gauss point —
            this signals a degenerate or inverted element whose contribution
            would corrupt the assembled global stiffness with a wrong-sign
            block. Catching here makes failures legible instead of silent.
        """
        Ke = np.zeros((24, 24))
        for gp_idx in range(len(self._gauss_weights)):
            xi, eta, zeta = self._gauss_points[gp_idx]
            w = self._gauss_weights[gp_idx]
            B = self.B_matrix(xi, eta, zeta)
            C_bar = self._degraded_stiffness(xi, eta, zeta)
            J = self.jacobian(xi, eta, zeta)
            detJ = np.linalg.det(J)
            if not np.isfinite(detJ) or detJ <= 0.0:
                raise ValueError(
                    f"Element has non-positive Jacobian determinant "
                    f"(detJ={detJ!r}) at Gauss point "
                    f"(xi={xi}, eta={eta}, zeta={zeta}). The element is "
                    f"degenerate or has inverted node ordering — its "
                    f"contribution would silently corrupt the assembled "
                    f"stiffness."
                )
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                Ke_contrib = (B.T @ C_bar @ B) * detJ * w
            # Protect against overflow in void elements
            if np.any(~np.isfinite(Ke_contrib)):
                Ke_contrib = np.where(np.isfinite(Ke_contrib), Ke_contrib, 0.0)
            Ke += Ke_contrib
        return Ke

    def stress_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute stress at all Gauss points.

        Parameters
        ----------
        u_elem : np.ndarray
            Shape (24,) element nodal displacement vector.

        Returns
        -------
        np.ndarray
            Shape (n_gp, 6) stress in Voigt notation.
        """
        u_elem = np.asarray(u_elem, dtype=float)
        n_gp = len(self._gauss_weights)
        stresses = np.empty((n_gp, 6))
        for gp_idx in range(n_gp):
            xi, eta, zeta = self._gauss_points[gp_idx]
            B = self.B_matrix(xi, eta, zeta)
            C_bar = self._degraded_stiffness(xi, eta, zeta)
            stresses[gp_idx] = C_bar @ (B @ u_elem)
        return stresses

    def strain_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute strain at all Gauss points.

        Parameters
        ----------
        u_elem : np.ndarray
            Shape (24,) element nodal displacement vector.

        Returns
        -------
        np.ndarray
            Shape (n_gp, 6) engineering strain in Voigt notation.
        """
        u_elem = np.asarray(u_elem, dtype=float)
        n_gp = len(self._gauss_weights)
        strains = np.empty((n_gp, 6))
        for gp_idx in range(n_gp):
            xi, eta, zeta = self._gauss_points[gp_idx]
            B = self.B_matrix(xi, eta, zeta)
            strains[gp_idx] = B @ u_elem
        return strains

    @property
    def volume(self) -> float:
        """Element volume via Gauss quadrature.

        Uses ``abs(det(J))`` so an inverted-but-otherwise-valid element
        still reports a sensible (positive) volume. ``stiffness_matrix``
        rejects inverted elements at assembly time, so the negative-volume
        case is only reachable via direct ``.volume`` lookup on a degenerate
        element constructed manually.
        """
        vol = 0.0
        for gp_idx in range(len(self._gauss_weights)):
            xi, eta, zeta = self._gauss_points[gp_idx]
            w = self._gauss_weights[gp_idx]
            J = self.jacobian(xi, eta, zeta)
            vol += abs(np.linalg.det(J)) * w
        return float(vol)

