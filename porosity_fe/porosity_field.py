"""Continuous porosity field with optional discrete-void superposition."""

from typing import List, Optional, Tuple, Union

import numpy as np

from .materials import MaterialProperties
from .void_geometry import VOID_SHAPES, VoidGeometry

# ============================================================
# SECTION 3: POROSITY FIELD MODEL
# ============================================================

POROSITY_CONFIGS = {
    'uniform_spherical': {
        'distribution': 'uniform',
        'void_shape': 'spherical',
    },
    'uniform_cylindrical': {
        'distribution': 'uniform',
        'void_shape': 'cylindrical',
    },
    'clustered_midplane': {
        'distribution': 'clustered',
        'void_shape': 'spherical',
        'cluster_location': 'midplane',
    },
    'clustered_surface': {
        'distribution': 'clustered',
        'void_shape': 'spherical',
        'cluster_location': 'surface',
    },
    'interface_penny': {
        'distribution': 'interface',
        'void_shape': 'penny',
    },
}


class PorosityField:
    """Continuous through-thickness porosity profile plus discrete voids.

    A ``PorosityField`` produces a scalar void volume fraction
    ``Vp(x, y, z) in [0, 1]`` at any point inside the laminate by
    superposing a smooth distributed component (one of three closed-form
    through-thickness profiles) with an optional list of explicit
    :class:`VoidGeometry` ellipsoids. The distributed profile is
    renormalized so its mean over ``z in [0, Lz]`` equals ``Vp``. This is
    the porosity input consumed by :class:`CompositeMesh` (sampled at
    every node) and by :class:`Hex8Element` (the Mori-Tanaka stiffness
    degradation reads the per-node values from here).

    Parameters
    ----------
    material : MaterialProperties
        Composite material — supplies ``t_ply`` and ``n_plies`` (and
        hence ``Lz = t_ply * n_plies``) used by the through-thickness
        profile.
    void_volume_fraction : float
        Mean void volume fraction ``Vp`` over the laminate, as a
        fraction in ``[0, 1]`` (e.g. ``0.02`` for 2 %). Passing a percent
        such as ``2.0`` raises ``ValueError`` with a unit hint.
    distribution : {'uniform', 'clustered', 'interface'}, optional
        Through-thickness profile shape (default ``'uniform'``).

        - ``'uniform'`` — constant ``Vp`` at every ``z``.
        - ``'clustered'`` — Gaussian bump centered at
          ``Lz * _CLUSTER_OFFSETS[cluster_location]`` with
          ``sigma = Lz / 6``.
        - ``'interface'`` — sum of Gaussians at each ply interface
          ``z = k * t_ply`` with ``sigma = 0.35 * t_ply``.
    void_shape : str or tuple of 3 float, optional
        Either a key of :data:`VOID_SHAPES`
        (``'spherical'``, ``'cylindrical'`` or ``'penny'``) or an
        explicit ``(a1, a2, a3)`` shape-radii tuple. Used as the Eshelby
        inclusion shape for the Mori-Tanaka homogenization in
        :class:`Hex8Element`. Default ``'spherical'``.
    cluster_location : {'midplane', 'surface', 'quarter'}, optional
        Location of the Gaussian bump for ``distribution='clustered'``,
        expressed as a fraction of ``Lz``. ``'midplane'`` -> ``0.5``,
        ``'surface'`` -> ``0.0``, ``'quarter'`` -> ``0.25``. Ignored for
        the other distributions. Default ``'midplane'``.
    discrete_voids : list of VoidGeometry, optional
        Explicit ellipsoidal voids superposed on the smooth profile
        (their contribution is taken via ``max``, then clipped at 1.0).
    seed : int, optional
        Recorded into provenance for a future stochastic placement mode
        (#55). The current pipeline is RNG-free, so this argument has no
        effect on the produced field.

    Attributes
    ----------
    material : MaterialProperties
        Stored reference to the input material.
    Vp : float
        Mean void volume fraction (after the snap-to-1.0 tolerance).
    distribution : str
        One of the three names listed above.
    cluster_location : str
        One of ``'midplane'``, ``'surface'``, ``'quarter'``.
    void_shape_radii : tuple of 3 float
        Resolved Eshelby ``(a1, a2, a3)`` shape radii.
    discrete_voids : list of VoidGeometry
        Discrete inclusions (empty list if none were supplied).
    Lz : float
        Total laminate thickness (mm), copied from
        ``material.total_thickness``.
    seed : int or None
        The seed argument, kept for provenance.

    Examples
    --------
    A 2 % uniformly-distributed spherical porosity field:

    >>> mat = MATERIALS['T800_epoxy']
    >>> field = PorosityField(mat, void_volume_fraction=0.02,
    ...                       distribution='uniform',
    ...                       void_shape='spherical')
    >>> bool(0.0 < field.Vp <= 1.0)
    True

    A midplane-clustered field with penny-shaped voids:

    >>> field = PorosityField(mat, void_volume_fraction=0.03,
    ...                       distribution='clustered',
    ...                       cluster_location='midplane',
    ...                       void_shape='penny')
    """

    _CLUSTER_OFFSETS = {'midplane': 0.5, 'surface': 0.0, 'quarter': 0.25}
    _DISTRIBUTIONS = ('uniform', 'clustered', 'interface')

    def __init__(self, material: MaterialProperties, void_volume_fraction: float,
                 distribution: str = 'uniform', void_shape: Union[str, Tuple] = 'spherical',
                 cluster_location: str = 'midplane',
                 discrete_voids: Optional[List[VoidGeometry]] = None,
                 seed: Optional[int] = None):
        if void_volume_fraction is None:
            raise ValueError("void_volume_fraction is None; expected a finite float in [0, 1].")
        if not isinstance(void_volume_fraction, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"void_volume_fraction must be a numeric type (int, float, or numpy scalar), "
                f"got {type(void_volume_fraction).__name__}."
            )
        Vp = float(void_volume_fraction)
        # Snap to 1.0 for inputs in (1.0, 1.0 + 1e-9] — accommodates upstream
        # numerical noise (e.g. np.mean across an element) without rejecting it.
        if 1.0 < Vp <= 1.0 + 1e-9:
            Vp = 1.0
        if not np.isfinite(Vp) or not (0.0 <= Vp <= 1.0):
            # Suppress the percent-confusion hint for values just above 1.0
            # (likely numerical noise rather than a percent mistake).
            show_percent_hint = (np.isfinite(Vp) and Vp >= 1.0 + 1e-3)
            hint = (f" Did you pass a percent? Use {Vp / 100:.4f} instead of {Vp}."
                    if show_percent_hint else "")
            raise ValueError(
                f"void_volume_fraction must be a finite fraction in [0, 1], "
                f"got {void_volume_fraction!r}.{hint}"
            )

        if distribution not in self._DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {distribution!r}. "
                f"Use one of {list(self._DISTRIBUTIONS)}."
            )
        if cluster_location not in self._CLUSTER_OFFSETS:
            raise ValueError(
                f"Unknown cluster_location {cluster_location!r}. "
                f"Use one of {sorted(self._CLUSTER_OFFSETS)}."
            )

        self.material = material
        self.Vp = Vp
        self.distribution = distribution
        self.cluster_location = cluster_location
        self.discrete_voids = discrete_voids or []
        # The pipeline is RNG-free today; `seed` is recorded into provenance
        # so a future stochastic void-placement mode has a determinism
        # contract to honor (#55).
        self.seed = seed
        self.Lz = material.total_thickness

        # Resolve void shape
        if isinstance(void_shape, str):
            if void_shape not in VOID_SHAPES:
                raise ValueError(
                    f"Unknown void_shape {void_shape!r}. "
                    f"Use one of {sorted(VOID_SHAPES)}."
                )
            self.void_shape_radii = VOID_SHAPES[void_shape]
        else:
            self.void_shape_radii = tuple(void_shape)

    def _compute_normalization(self, distribution: str, cluster_location: str) -> float:
        """Compute normalization factor over the full domain so average equals Vp."""
        z_ref = np.linspace(0, self.Lz, 1000)
        if distribution == 'clustered':
            z0 = self.Lz * self._CLUSTER_OFFSETS[cluster_location]
            sigma = self.Lz / 6
            profile_ref = np.exp(-0.5 * ((z_ref - z0) / sigma)**2)
        elif distribution == 'interface':
            t = self.material.t_ply
            n = self.material.n_plies
            profile_ref = np.zeros_like(z_ref)
            for k in range(1, n):
                z_int = k * t
                profile_ref += np.exp(-0.5 * ((z_ref - z_int) / (t * 0.35))**2)
        else:
            return 1.0
        mean_val = np.mean(profile_ref)
        return mean_val if mean_val > 0 else 1.0

    def _distributed_porosity(self, z: np.ndarray) -> np.ndarray:
        """Through-thickness distributed porosity profile."""
        z = np.asarray(z, dtype=float)
        if self.distribution == 'uniform':
            return np.full_like(z, self.Vp)
        elif self.distribution == 'clustered':
            z0 = self.Lz * self._CLUSTER_OFFSETS[self.cluster_location]
            sigma = self.Lz / 6
            profile = np.exp(-0.5 * ((z - z0) / sigma)**2)
            norm = self._compute_normalization('clustered', self.cluster_location)
            return self.Vp * profile / norm
        elif self.distribution == 'interface':
            t = self.material.t_ply
            n = self.material.n_plies
            profile = np.zeros_like(z)
            for k in range(1, n):
                z_int = k * t
                profile += np.exp(-0.5 * ((z - z_int) / (t * 0.35))**2)
            norm = self._compute_normalization('interface', self.cluster_location)
            return self.Vp * profile / norm
        else:
            raise ValueError(
                f"Unknown distribution {self.distribution!r}. "
                f"Use one of {list(self._DISTRIBUTIONS)}."
            )

    def local_porosity(self, x, y, z) -> np.ndarray:
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        Vp_dist = self._distributed_porosity(z)
        Vp_discrete = np.zeros_like(Vp_dist)
        for void in self.discrete_voids:
            Vp_discrete = np.maximum(Vp_discrete,
                                      void.contains(x, y, z).astype(float))
        return np.minimum(Vp_dist + Vp_discrete, 1.0)

    def local_stiffness_reduction(self, x, y, z) -> np.ndarray:
        Vp_local = self.local_porosity(x, y, z)
        return 1.0 - Vp_local

    def get_void_locations(self) -> list:
        return [(v.center.tolist(), v.radii.tolist()) for v in self.discrete_voids]

    def effective_porosity_profile(self, nz: int = 100) -> tuple:
        """Through-thickness profile including discrete void contributions."""
        z_coords = np.linspace(0, self.Lz, nz)
        x_mid = np.full(nz, 25.0)  # Sample at domain center
        y_mid = np.full(nz, 10.0)
        Vp_vals = self.local_porosity(x_mid, y_mid, z_coords)
        return z_coords, Vp_vals

    def __repr__(self) -> str:
        return (f"PorosityField(Vp={self.Vp:.4f}, "
                f"distribution='{self.distribution}', "
                f"void_shape={self.void_shape_radii}, "
                f"n_discrete_voids={len(self.discrete_voids)})")
