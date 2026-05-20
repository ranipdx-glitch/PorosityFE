"""Discrete ellipsoidal void geometry."""

from typing import Tuple

import numpy as np

# ============================================================
# SECTION 2: VOID GEOMETRY MODEL
# ============================================================

VOID_SHAPES = {
    'spherical':   (1.0, 1.0, 1.0),
    'cylindrical': (3.0, 1.0, 1.0),
    'penny':       (3.0, 3.0, 0.3),
}


class VoidGeometry:
    """Single discrete void parameterized as an oriented ellipsoid.

    A void is the locus of points satisfying

    .. math::

        \\left(\\frac{x_\\ell}{a}\\right)^2
        + \\left(\\frac{y_\\ell}{b}\\right)^2
        + \\left(\\frac{z_\\ell}{c}\\right)^2 \\le 1,

    where ``(x_l, y_l, z_l)`` are coordinates in the **void-local** frame:
    coordinates are first translated so the void centroid is at the origin
    and then rotated by ``-orientation`` about the global +z axis. The
    semi-axes ``(a, b, c)`` correspond to the world x / y / z directions
    in that local frame. This class is the porosity-model counterpart of
    ``WrinkleGeometry`` in the WrinkleFE codebase.

    Parameters
    ----------
    center : tuple of 3 float
        Centroid coordinates ``(x, y, z)`` in the global frame, in mm.
        Must be finite.
    radii : tuple of 3 float
        Semi-axes ``(a, b, c)`` of the ellipsoid in mm, ordered along the
        **local** x / y / z axes. All three must be positive and finite
        (they appear as ``1 / r`` in the containment test).
    orientation : float, optional
        Rotation about the global +z axis, in **radians** (default 0).
        Positive values rotate the local x-axis toward the global y-axis.

    Attributes
    ----------
    center : np.ndarray
        Shape ``(3,)`` float array; the void centroid in mm.
    radii : np.ndarray
        Shape ``(3,)`` float array of positive semi-axes ``(a, b, c)``
        in mm.
    orientation : float
        In-plane rotation angle in radians.
    aspect_ratio : float
        Read-only property: ``max(radii) / min(radii)``, used to pick a
        spherical / cylindrical / penny SCF regime in
        :meth:`stress_concentration_factor`.

    Examples
    --------
    A 1 mm-radius spherical void at the coupon midpoint:

    >>> v = VoidGeometry(center=(25.0, 10.0, 2.2),
    ...                  radii=(1.0, 1.0, 1.0))
    >>> bool(v.contains(25.0, 10.0, 2.2))
    True
    >>> round(v.volume(), 4)
    4.1888

    A penny-shaped void rotated 30 deg about z:

    >>> import math
    >>> v = VoidGeometry(center=(25.0, 10.0, 2.2),
    ...                  radii=(3.0, 3.0, 0.3),
    ...                  orientation=math.radians(30.0))
    >>> round(v.aspect_ratio, 2)
    10.0
    """

    def __init__(self, center: Tuple, radii: Tuple, orientation: float = 0.0):
        self.center = np.array(center, dtype=float)
        self.radii = np.array(radii, dtype=float)
        if self.center.shape != (3,):
            raise ValueError(
                f"VoidGeometry.center must have 3 components (x, y, z), "
                f"got shape {self.center.shape}."
            )
        if self.radii.shape != (3,):
            raise ValueError(
                f"VoidGeometry.radii must have 3 components (a, b, c), "
                f"got shape {self.radii.shape}."
            )
        if not np.all(np.isfinite(self.radii)) or np.any(self.radii <= 0):
            raise ValueError(
                f"VoidGeometry.radii must be 3 positive finite numbers "
                f"(used as 1/r in the ellipsoid containment test), "
                f"got {self.radii.tolist()}."
            )
        if not np.all(np.isfinite(self.center)):
            raise ValueError(
                f"VoidGeometry.center must be finite, got {self.center.tolist()}."
            )
        if not np.isfinite(orientation):
            raise ValueError(
                f"VoidGeometry.orientation must be a finite angle (radians), "
                f"got {orientation!r}."
            )
        self.orientation = orientation

    def _to_local(self, x, y, z):
        """Transform world coordinates to void-local (translated + rotated)."""
        dx = np.asarray(x, dtype=float) - self.center[0]
        dy = np.asarray(y, dtype=float) - self.center[1]
        dz = np.asarray(z, dtype=float) - self.center[2]
        c, s = np.cos(self.orientation), np.sin(self.orientation)
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        z_loc = dz
        return x_loc, y_loc, z_loc

    def contains(self, x, y, z) -> np.ndarray:
        x_l, y_l, z_l = self._to_local(x, y, z)
        val = (x_l / self.radii[0])**2 + (y_l / self.radii[1])**2 + (z_l / self.radii[2])**2
        return val <= 1.0

    def distance_field(self, x, y, z) -> np.ndarray:
        x_l, y_l, z_l = self._to_local(x, y, z)
        val = np.sqrt((x_l / self.radii[0])**2 + (y_l / self.radii[1])**2 + (z_l / self.radii[2])**2)
        r_eff = np.sqrt(x_l**2 + y_l**2 + z_l**2)
        # At the void center (r_eff ~ 0), val is also ~0 causing 0/0.
        # Return -1.0 (clearly inside) for those points.
        eps = 1e-12
        at_center = r_eff < eps
        r_eff = np.maximum(r_eff, eps)
        val_safe = np.maximum(val, eps)
        result = r_eff * (val - 1.0) / val_safe
        result = np.where(at_center, -1.0, result)
        return result

    def stress_concentration_factor(self) -> dict:
        ar = self.aspect_ratio
        # 'transverse_tension' is matrix-dominated; pair its SCF with the
        # in-plane shear value (issue #35).
        if ar < 1.2:  # Spherical
            return {'compression': 2.0, 'tension': 2.0, 'shear': 1.5,
                    'ilss': 1.8, 'transverse_tension': 2.0}
        elif self.radii[0] > self.radii[2]:
            if self.radii[1] < self.radii[0] * 0.5:  # Cylindrical (prolate)
                return {'compression': 1.5 + 0.5 * ar, 'tension': 1.5 + 0.5 * ar,
                        'shear': 1.3 + 0.3 * ar, 'ilss': 1.5 + 0.4 * ar,
                        'transverse_tension': 1.5 + 0.5 * ar}
            else:  # Penny (oblate)
                return {'compression': 2.0 + 1.0 * ar, 'tension': 2.0 + 1.5 * ar,
                        'shear': 1.5 + 0.8 * ar, 'ilss': 2.0 + 1.2 * ar,
                        'transverse_tension': 2.0 + 1.5 * ar}
        else:
            return {'compression': 2.0, 'tension': 2.0, 'shear': 1.5,
                    'ilss': 1.8, 'transverse_tension': 2.0}

    def volume(self) -> float:
        return (4.0 / 3.0) * np.pi * self.radii[0] * self.radii[1] * self.radii[2]

    @property
    def aspect_ratio(self) -> float:
        return float(np.max(self.radii) / np.min(self.radii))

    def __repr__(self) -> str:
        return (f"VoidGeometry(center={self.center.tolist()}, "
                f"radii={self.radii.tolist()}, "
                f"orientation={self.orientation:.3f}, "
                f"aspect_ratio={self.aspect_ratio:.2f})")
