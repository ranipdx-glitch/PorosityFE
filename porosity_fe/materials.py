"""Material properties dataclass and built-in presets."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# ============================================================
# SECTION 1: MATERIAL PROPERTIES AND CONSTANTS
# ============================================================

@dataclass
class MaterialProperties:
    """Composite material properties with constituent data for micromechanics.

    Bundles the lamina-level orthotropic stiffness, the longitudinal /
    transverse / shear strength allowables (used by the Tsai-Wu failure
    criterion in :class:`Hex8Element`), the ply / laminate geometry and the
    constituent (matrix + fiber) data needed by the Mori-Tanaka
    homogenization for porosity degradation. All stress / modulus inputs are
    in **MPa** and all lengths are in **mm**; Poisson ratios and the fiber
    volume fraction are dimensionless fractions.

    The dataclass is validated by :meth:`__post_init__`: stiffness moduli
    and strengths must be positive finite floats; Poisson ratios must lie
    in ``(-1, 0.5)``; ``n_plies`` must be a positive integer; and
    ``fiber_volume_fraction`` must be a fraction in ``(0, 1)`` (a percent
    such as ``60`` is rejected with a hint).

    Parameters
    ----------
    E11, E22, E33 : float
        Lamina orthotropic Young's moduli along the fiber (1), transverse
        (2) and through-thickness (3) directions, in MPa.
    G12, G13, G23 : float
        Lamina shear moduli (in-plane, interlaminar and transverse shear),
        in MPa.
    nu12, nu13, nu23 : float
        Major (12), through-thickness (13) and transverse (23) Poisson's
        ratios. Each must lie in ``(-1, 0.5)``.
    sigma_1c, sigma_1t : float
        Longitudinal compression and tension allowables, in MPa.
    sigma_2t, sigma_2c : float
        Transverse tension and compression allowables, in MPa.
    tau_12 : float
        In-plane shear allowable, in MPa.
    tau_ilss : float
        Interlaminar (short-beam) shear allowable, in MPa.
    t_ply : float
        Ply thickness, in mm.
    n_plies : int
        Number of plies in the laminate (positive integer).
    matrix_modulus : float
        Matrix (resin) Young's modulus ``E_m``, in MPa, used by the
        Mori-Tanaka homogenization in :class:`Hex8Element`.
    matrix_poisson : float
        Matrix Poisson's ratio ``nu_m`` (dimensionless, in
        ``(-1, 0.5)``).
    fiber_modulus : float
        Fiber longitudinal Young's modulus ``E_f``, in MPa.
    fiber_volume_fraction : float
        Pristine fiber volume fraction ``V_f``, as a fraction in
        ``(0, 1)`` (e.g. ``0.60`` for a 60 % fiber laminate).

    Attributes
    ----------
    E11, E22, E33 : float
        Orthotropic Young's moduli (MPa).
    G12, G13, G23 : float
        Shear moduli (MPa).
    nu12, nu13, nu23 : float
        Poisson's ratios (dimensionless).
    sigma_1c, sigma_1t, sigma_2t, sigma_2c : float
        Normal-direction strengths (MPa).
    tau_12, tau_ilss : float
        Shear strengths (MPa).
    t_ply : float
        Ply thickness (mm).
    n_plies : int
        Number of plies.
    matrix_modulus, matrix_poisson : float
        Constituent matrix elasticity (MPa, dimensionless).
    fiber_modulus, fiber_volume_fraction : float
        Constituent fiber modulus (MPa) and pristine fiber volume
        fraction (dimensionless, in ``(0, 1)``).
    total_thickness : float
        Read-only property: ``t_ply * n_plies`` (mm). Used as ``L_z`` by
        :class:`CompositeMesh`.

    Examples
    --------
    Build a T800/epoxy ply (the same values are pre-baked in
    :data:`MATERIALS`):

    >>> mat = MaterialProperties(
    ...     E11=161000.0, E22=11380.0, E33=11380.0,
    ...     G12=5170.0, G13=5170.0, G23=3980.0,
    ...     nu12=0.32, nu13=0.32, nu23=0.40,
    ...     sigma_1c=1500.0, sigma_1t=2800.0,
    ...     sigma_2t=80.0, sigma_2c=250.0,
    ...     tau_12=100.0, tau_ilss=90.0,
    ...     t_ply=0.183, n_plies=24,
    ...     matrix_modulus=3500.0, matrix_poisson=0.35,
    ...     fiber_modulus=294000.0, fiber_volume_fraction=0.60,
    ... )
    >>> round(mat.total_thickness, 4)
    4.392
    """
    # Lamina-level orthotropic properties
    E11: float          # Longitudinal modulus (MPa)
    E22: float          # Transverse modulus (MPa)
    E33: float          # Through-thickness modulus (MPa)
    G12: float          # In-plane shear modulus (MPa)
    G13: float          # Interlaminar shear modulus (MPa)
    G23: float          # Transverse shear modulus (MPa)
    nu12: float         # Major Poisson's ratio
    nu13: float         # Through-thickness Poisson's ratio
    nu23: float         # Transverse Poisson's ratio

    # Longitudinal strengths
    sigma_1c: float     # Longitudinal compression strength (MPa)
    sigma_1t: float     # Longitudinal tension strength (MPa)
    # Transverse strengths
    sigma_2t: float     # Transverse tension strength (MPa)
    sigma_2c: float     # Transverse compression strength (MPa)
    # Shear strengths
    tau_12: float       # In-plane shear strength (MPa)
    tau_ilss: float     # Interlaminar shear strength (MPa)

    # Geometric
    t_ply: float        # Ply thickness (mm)
    n_plies: int        # Number of plies

    # Constituent properties (for micromechanics)
    matrix_modulus: float         # E_m (MPa)
    matrix_poisson: float         # nu_m
    fiber_modulus: float          # E_f (MPa)
    fiber_volume_fraction: float  # V_f (pristine)

    # ----------------------------------------------------------------
    # Hygrothermal conditioning (issue #59).
    #
    # All five fields default to "no environmental effect": when any of
    # ``T_service`` / ``M_service`` / ``T_g_dry`` is ``None``,
    # :meth:`environment_knockdown` returns 1.0 (back-compat no-op).
    # ----------------------------------------------------------------
    T_service: Optional[float] = None   # Service temperature (deg C)
    M_service: Optional[float] = None   # Service moisture content (wt %)
    T_ref: float = 23.0                 # Reference / RT (deg C); RTD baseline
    M_ref: float = 0.0                  # Reference moisture (wt %)
    T_g_dry: Optional[float] = None     # Dry glass-transition temperature (deg C)

    def __post_init__(self):
        # Stiffness moduli must be positive finite (non-zero for 1/E in compliance).
        for name in ('E11', 'E22', 'E33', 'G12', 'G13', 'G23',
                     'matrix_modulus', 'fiber_modulus'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"MaterialProperties.{name} must be a positive finite number "
                    f"(MPa), got {value!r}."
                )
        # Poisson ratios must be in (-1, 0.5) for an isotropic-stable matrix
        # (the matrix stiffness uses (1 - 2*nu_m) in the denominator) and for
        # well-posed orthotropic compliance entries.
        for name in ('nu12', 'nu13', 'nu23', 'matrix_poisson'):
            value = getattr(self, name)
            if not np.isfinite(value) or not (-1.0 < value < 0.5):
                raise ValueError(
                    f"MaterialProperties.{name} must be a finite Poisson's ratio "
                    f"in (-1, 0.5), got {value!r}."
                )
        # Strengths must be positive finite (used as 1/X in Tsai-Wu).
        for name in ('sigma_1c', 'sigma_1t', 'sigma_2t', 'sigma_2c',
                     'tau_12', 'tau_ilss'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"MaterialProperties.{name} must be a positive finite "
                    f"strength (MPa), got {value!r}."
                )
        # Geometry
        if not np.isfinite(self.t_ply) or self.t_ply <= 0:
            raise ValueError(
                f"MaterialProperties.t_ply must be a positive ply thickness (mm), "
                f"got {self.t_ply!r}."
            )
        if not isinstance(self.n_plies, (int, np.integer)) or self.n_plies <= 0:
            raise ValueError(
                f"MaterialProperties.n_plies must be a positive integer, "
                f"got {self.n_plies!r}."
            )
        # Fiber volume fraction
        if (not np.isfinite(self.fiber_volume_fraction)
                or not (0.0 < self.fiber_volume_fraction < 1.0)):
            raise ValueError(
                f"MaterialProperties.fiber_volume_fraction must be a fraction in "
                f"(0, 1), got {self.fiber_volume_fraction!r}. "
                f"(Pass a fraction such as 0.60, not a percent.)"
            )

        # Hygrothermal conditioning (issue #59). Optional scalars; reject
        # only explicit non-finite or nonsensical values so the default
        # ``None`` no-op path is unaffected.
        for name in ('T_service', 'M_service', 'T_g_dry'):
            value = getattr(self, name)
            if value is not None:
                if not np.isfinite(float(value)):
                    raise ValueError(
                        f"MaterialProperties.{name} must be a finite number or "
                        f"None, got {value!r}."
                    )
        for name in ('T_ref', 'M_ref'):
            value = getattr(self, name)
            if not np.isfinite(float(value)):
                raise ValueError(
                    f"MaterialProperties.{name} must be a finite number, "
                    f"got {value!r}."
                )
        if self.M_service is not None and float(self.M_service) < 0.0:
            raise ValueError(
                f"MaterialProperties.M_service (moisture wt%) must be >= 0, "
                f"got {self.M_service!r}."
            )
        if float(self.M_ref) < 0.0:
            raise ValueError(
                f"MaterialProperties.M_ref (moisture wt%) must be >= 0, "
                f"got {self.M_ref!r}."
            )

    @property
    def total_thickness(self) -> float:
        return self.t_ply * self.n_plies

    def get_compliance_matrix(self) -> np.ndarray:
        """6x6 compliance matrix [S] for orthotropic material.

        Notes
        -----
        Voigt order: ``[11, 22, 33, 23, 13, 12]`` (normals first, then shears in
        the 23 / 13 / 12 order). The shear rows/columns assume **engineering**
        strain (``gamma_ij = 2 * eps_ij``), i.e. ``S[5, 5] = 1 / G12`` maps a
        single ``tau_12`` directly to ``gamma_12 = tau_12 / G12`` without a
        factor of two. Stress is in MPa; strain is dimensionless.
        """
        S = np.zeros((6, 6))
        S[0, 0] = 1.0 / self.E11
        S[1, 1] = 1.0 / self.E22
        S[2, 2] = 1.0 / self.E33
        S[0, 1] = S[1, 0] = -self.nu12 / self.E11
        S[0, 2] = S[2, 0] = -self.nu13 / self.E11
        S[1, 2] = S[2, 1] = -self.nu23 / self.E22
        S[3, 3] = 1.0 / self.G23
        S[4, 4] = 1.0 / self.G13
        S[5, 5] = 1.0 / self.G12
        return S

    def get_stiffness_matrix(self) -> np.ndarray:
        """6x6 stiffness matrix [C] = [S]^-1.

        Notes
        -----
        Voigt order: ``[11, 22, 33, 23, 13, 12]`` (normals first, then shears in
        the 23 / 13 / 12 order), matching :meth:`get_compliance_matrix`. Shear
        components are **engineering** strain (``gamma_ij = 2 * eps_ij``), so
        ``C[5, 5] = G12`` maps ``gamma_12`` directly to ``tau_12 = G12 *
        gamma_12``. Stress is in MPa; strain is dimensionless.
        """
        return np.linalg.inv(self.get_compliance_matrix())

    def get_isotropic_matrix_stiffness(self) -> np.ndarray:
        """6x6 isotropic stiffness tensor C_m from matrix_modulus and matrix_poisson."""
        E_m = self.matrix_modulus
        nu_m = self.matrix_poisson
        lam = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))
        mu = E_m / (2 * (1 + nu_m))
        C_m = np.zeros((6, 6))
        C_m[0, 0] = C_m[1, 1] = C_m[2, 2] = lam + 2 * mu
        C_m[0, 1] = C_m[0, 2] = C_m[1, 0] = C_m[1, 2] = C_m[2, 0] = C_m[2, 1] = lam
        C_m[3, 3] = C_m[4, 4] = C_m[5, 5] = mu
        return C_m

    # Modes whose strength is matrix-/interface-dominated and therefore
    # sensitive to hygrothermal conditioning. Mirrors the analogous frozenset
    # on :class:`EmpiricalSolver` but lives here so callers that only have a
    # ``MaterialProperties`` (no solver) can still query the knockdown.
    _HYGROTHERMAL_MATRIX_DOMINATED_MODES = frozenset({
        'transverse_tension', 'ilss', 'shear', 'compression',
    })

    # Springer / Chamis empirical slope for the dry -> wet shift of the
    # glass-transition temperature: T_g_wet ~= T_g_dry - 25 * M, with M in
    # wt% moisture content. See Springer (1981, "Environmental Effects on
    # Composite Materials") and Chamis (NASA-TM-83320, 1983).
    _SPRINGER_MOISTURE_TG_SLOPE = 25.0  # deg C per wt% moisture

    def environment_knockdown(self, mode: str,
                              T: Optional[float] = None,
                              M: Optional[float] = None) -> float:
        """Hygrothermal (T / M) knockdown factor for the requested mode.

        Implements the standard Chamis / Springer matrix-property ratio::

            F_env = sqrt((T_g_wet - T) / (T_g_dry - T_ref))

        where ``T_g_wet ~= T_g_dry - 25 * M`` (the Springer rule of thumb
        for epoxy matrices, with moisture ``M`` in wt%). The square-root
        form was proposed by Chamis (NASA-TM-83320, 1983) for matrix
        modulus / strength retention as the service temperature approaches
        the wet glass transition.

        Fiber-dominated modes (``'tension'``) are largely insensitive to
        hygrothermal conditioning at engineering relevant temperatures and
        return ``1.0`` unconditionally. Matrix- and matrix/interface-
        dominated modes (``'transverse_tension'``, ``'ilss'``, ``'shear'``)
        get the full Chamis/Springer ratio. ``'compression'`` is
        treated as matrix-dominated here because fiber microbuckling is
        gated by matrix shear stiffness — a defensible aerospace-screening
        choice, but conservative compared to a true fiber-failure mode.

        Parameters
        ----------
        mode : str
            Loading mode name (see
            :attr:`EmpiricalSolver.PRISTINE_STRENGTH_KEY`).
        T : float, optional
            Service temperature in degrees Celsius. Falls back to
            :attr:`T_service` when ``None``.
        M : float, optional
            Service moisture content in wt%. Falls back to
            :attr:`M_service` when ``None``.

        Returns
        -------
        float
            Multiplicative knockdown in ``(0, 1]``. Returns ``1.0`` (no
            effect) whenever any of ``T`` / ``M`` / :attr:`T_g_dry` is
            unspecified — the back-compat no-op path.

        Notes
        -----
        This is a screening-level model. Production design allowables
        should still come from a fully populated test matrix per the
        applicable spec (e.g. CMH-17 Vol. 2 hygrothermal conditioning).
        The factor is clamped to ``[0.01, 1.0]`` to keep downstream
        knockdown composition well-behaved when the service temperature
        is set extremely close to (or above) the wet ``T_g``.
        """
        # Resolve T / M from arguments or attribute defaults. ``None`` from
        # both sides -> graceful no-op.
        T_eff = T if T is not None else self.T_service
        M_eff = M if M is not None else self.M_service
        if T_eff is None or M_eff is None or self.T_g_dry is None:
            return 1.0

        # Fiber-dominated modes are insensitive to T / M at engineering
        # relevant temperatures.
        if mode not in self._HYGROTHERMAL_MATRIX_DOMINATED_MODES:
            return 1.0

        T_eff = float(T_eff)
        M_eff = float(M_eff)
        T_g_dry = float(self.T_g_dry)
        T_ref = float(self.T_ref)

        T_g_wet = T_g_dry - self._SPRINGER_MOISTURE_TG_SLOPE * M_eff
        denom = T_g_dry - T_ref
        if denom <= 0.0:
            # Pathological calibration (T_ref above T_g_dry); the matrix is
            # already above its dry transition at the reference. Refuse to
            # scale rather than divide by zero.
            return 1.0

        numer = T_g_wet - T_eff
        if numer <= 0.0:
            # Service temperature has reached (or exceeded) the wet T_g —
            # matrix has effectively lost its load-carrying capability.
            # Clamp to a small floor so downstream multiplications stay
            # finite and so callers can spot the regime via the value.
            return 0.01

        ratio = numer / denom
        # Square-root form per Chamis; clamp the final factor to <= 1.0 so
        # cool / dry conditioning (numer > denom) does not synthesise
        # strength above the RTD allowable.
        return float(min(np.sqrt(ratio), 1.0))

    # Fields a UQ driver is allowed to perturb. Geometry (t_ply, n_plies) and
    # Poisson ratios are excluded by default: the empirical knockdown models
    # only consume strengths/moduli, and perturbing a bounded Poisson ratio or
    # an integer ply count is rarely the intent. Callers may still target any
    # of these via an explicit `covs`/`spec` key if needed.
    PERTURBABLE_FIELDS = (
        'E11', 'E22', 'E33', 'G12', 'G13', 'G23',
        'sigma_1c', 'sigma_1t', 'sigma_2t', 'sigma_2c', 'tau_12', 'tau_ilss',
        'matrix_modulus', 'fiber_modulus', 'fiber_volume_fraction',
    )

    def _perturbed_value(self, name: str, unit_draw: float,
                         dist: str, params) -> float:
        """Map one unit draw (a standard-normal or U(0,1) variate) to a
        perturbed value of field ``name``.

        ``dist`` is one of:
          - ``'lognormal'`` (default): multiplicative truncated-lognormal so a
            positive quantity stays positive. ``params`` is the coefficient of
            variation (CoV, std/mean of the underlying value). ``unit_draw`` is
            a standard-normal variate.
          - ``'normal'``: additive Gaussian. ``params`` is the CoV; the std is
            ``cov * |nominal|``. ``unit_draw`` is a standard-normal variate.
          - ``'uniform'``: ``params`` is the fractional half-width ``h``; the
            value is drawn uniformly on ``nominal * [1 - h, 1 + h]``.
            ``unit_draw`` is a U(0, 1) variate.
        """
        nominal = float(getattr(self, name))
        if dist == 'lognormal':
            cov = float(params)
            if cov <= 0.0:
                return nominal
            # Median-preserving lognormal with the requested CoV.
            sigma_ln = np.sqrt(np.log1p(cov * cov))
            return float(nominal * np.exp(sigma_ln * unit_draw))
        if dist == 'normal':
            cov = float(params)
            if cov <= 0.0:
                return nominal
            return float(nominal + cov * abs(nominal) * unit_draw)
        if dist == 'uniform':
            h = float(params)
            if h <= 0.0:
                return nominal
            return float(nominal * (1.0 - h + 2.0 * h * unit_draw))
        raise ValueError(
            f"Unknown distribution {dist!r} for field {name!r}. "
            f"Use one of 'lognormal', 'normal', 'uniform'."
        )

    def perturb(self, draws: Dict[str, float],
                spec: Dict[str, Tuple[str, float]]) -> "MaterialProperties":
        """Return a new ``MaterialProperties`` with the fields in ``spec``
        perturbed using the per-field unit draws in ``draws``.

        ``spec`` maps ``field -> (distribution, params)`` (see
        :meth:`_perturbed_value`). ``draws`` maps the same field names to a
        single unit variate. Fields absent from ``spec`` are left at nominal.
        The returned dataclass is re-validated by ``__post_init__``.
        """
        updates = {}
        for name, (dist, params) in spec.items():
            updates[name] = self._perturbed_value(
                name, float(draws[name]), dist, params)
        from dataclasses import replace as _dc_replace
        return _dc_replace(self, **updates)

    def __repr__(self) -> str:
        return (f"MaterialProperties(E11={self.E11}, E22={self.E22}, "
                f"G12={self.G12}, nu12={self.nu12}, "
                f"sigma_1c={self.sigma_1c}, sigma_1t={self.sigma_1t}, "
                f"n_plies={self.n_plies}, t_ply={self.t_ply}, "
                f"Vf={self.fiber_volume_fraction})")


MATERIALS = {
    'T800_epoxy': MaterialProperties(
        E11=161000.0, E22=11380.0, E33=11380.0,
        G12=5170.0, G13=5170.0, G23=3980.0,
        nu12=0.32, nu13=0.32, nu23=0.40,
        sigma_1c=1500.0, sigma_1t=2800.0, sigma_2t=80.0, sigma_2c=250.0,
        tau_12=100.0, tau_ilss=90.0,
        t_ply=0.183, n_plies=24,
        matrix_modulus=3500.0, matrix_poisson=0.35,
        fiber_modulus=294000.0, fiber_volume_fraction=0.60,
    ),
    'T700_epoxy': MaterialProperties(
        E11=132000.0, E22=10300.0, E33=10300.0,
        G12=4700.0, G13=4700.0, G23=3500.0,
        nu12=0.30, nu13=0.30, nu23=0.40,
        sigma_1c=1200.0, sigma_1t=2400.0, sigma_2t=65.0, sigma_2c=200.0,
        tau_12=85.0, tau_ilss=80.0,
        t_ply=0.125, n_plies=24,
        matrix_modulus=3200.0, matrix_poisson=0.35,
        fiber_modulus=230000.0, fiber_volume_fraction=0.58,
    ),
    'glass_epoxy': MaterialProperties(
        E11=45000.0, E22=12000.0, E33=12000.0,
        G12=5500.0, G13=5500.0, G23=4000.0,
        nu12=0.28, nu13=0.28, nu23=0.40,
        sigma_1c=600.0, sigma_1t=1100.0, sigma_2t=40.0, sigma_2c=140.0,
        tau_12=70.0, tau_ilss=55.0,
        t_ply=0.200, n_plies=24,
        matrix_modulus=3500.0, matrix_poisson=0.35,
        fiber_modulus=73000.0, fiber_volume_fraction=0.55,
    ),
    'IM7_8551_epoxy': MaterialProperties(
        E11=172000.0, E22=10000.0, E33=10000.0,
        G12=5500.0, G13=5500.0, G23=3800.0,
        nu12=0.30, nu13=0.30, nu23=0.45,
        sigma_1c=1600.0, sigma_1t=3100.0, sigma_2t=90.0, sigma_2c=260.0,
        tau_12=110.0, tau_ilss=100.0,
        t_ply=0.125, n_plies=24,
        matrix_modulus=3700.0, matrix_poisson=0.35,
        fiber_modulus=276000.0, fiber_volume_fraction=0.60,
    ),
    'T300_934_epoxy': MaterialProperties(
        E11=131000.0, E22=8500.0, E33=8500.0,
        G12=4600.0, G13=4600.0, G23=3000.0,
        nu12=0.28, nu13=0.28, nu23=0.42,
        sigma_1c=1200.0, sigma_1t=1900.0, sigma_2t=55.0, sigma_2c=200.0,
        tau_12=75.0, tau_ilss=85.0,
        t_ply=0.127, n_plies=16,
        matrix_modulus=3400.0, matrix_poisson=0.35,
        fiber_modulus=230000.0, fiber_volume_fraction=0.60,
    ),
    'CF_PEEK': MaterialProperties(
        E11=140000.0, E22=10000.0, E33=10000.0,
        G12=5200.0, G13=5200.0, G23=3500.0,
        nu12=0.32, nu13=0.32, nu23=0.45,
        sigma_1c=1100.0, sigma_1t=2200.0, sigma_2t=85.0, sigma_2c=180.0,
        tau_12=105.0, tau_ilss=95.0,
        t_ply=0.14, n_plies=8,
        matrix_modulus=3800.0, matrix_poisson=0.38,
        fiber_modulus=240000.0, fiber_volume_fraction=0.60,
    ),
    # AS4/3501-6 (Hercules/Hexcel) — an IM-class carbon/untoughened epoxy
    # system. Nominal lamina properties from Soden, Hinton & Kaddour
    # (Worldwide Failure Exercise, WWFE-I, Compos. Sci. Technol. 1998/2002)
    # and Daniel & Ishai, "Engineering Mechanics of Composite Materials"
    # (2nd ed., 2006, Table A.4). 3501-6 neat-resin modulus from Hexcel
    # technical datasheet (E_m ≈ 4.27 GPa, nu_m ≈ 0.34). AS4 fibre modulus
    # from Hexcel HexTow AS4 datasheet (E_f ≈ 235 GPa). Used for Ghiorse 1993
    # (AS4/3501-6 unidirectional) and Jeong 1997 (AS4 fabric/3501-6).
    'AS4_3501_6_epoxy': MaterialProperties(
        E11=142000.0, E22=10300.0, E33=10300.0,
        G12=7200.0, G13=7200.0, G23=3800.0,
        nu12=0.27, nu13=0.27, nu23=0.40,
        sigma_1c=1440.0, sigma_1t=2280.0, sigma_2t=57.0, sigma_2c=228.0,
        tau_12=71.0, tau_ilss=95.0,
        t_ply=0.125, n_plies=24,
        matrix_modulus=4270.0, matrix_poisson=0.34,
        fiber_modulus=235000.0, fiber_volume_fraction=0.60,
    ),
    # HTA 24k / EHkF 420 epoxy — Tenax HTA (Toho Tenax) high-tenacity
    # carbon fibre with a toughened aerospace epoxy system. Lamina
    # properties from the Stamopoulos et al. (2016) baseline tabulation and
    # the Tenax HTA fibre datasheet (E_f ≈ 238 GPa). The matrix modulus
    # (E_m ≈ 3.4 GPa, nu_m ≈ 0.35) is a typical aerospace toughened-epoxy
    # value used in the absence of an EHkF 420 datasheet entry. Strengths
    # are scaled to a standard HTA/epoxy unidirectional with Vf ≈ 0.60.
    'HTA_EHkF420_epoxy': MaterialProperties(
        E11=130000.0, E22=9000.0, E33=9000.0,
        G12=4500.0, G13=4500.0, G23=3200.0,
        nu12=0.32, nu13=0.32, nu23=0.42,
        sigma_1c=1200.0, sigma_1t=2100.0, sigma_2t=60.0, sigma_2c=200.0,
        tau_12=70.0, tau_ilss=75.0,
        t_ply=0.127, n_plies=16,
        matrix_modulus=3400.0, matrix_poisson=0.35,
        fiber_modulus=238000.0, fiber_volume_fraction=0.60,
    ),
}
