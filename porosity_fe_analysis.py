#!/usr/bin/env python3
"""
POROSITY DEFECT ANALYSIS FOR COMPOSITE LAMINATES
==================================================

Evaluates the effects of porosity (distributed microporosity and discrete
macrovoids) on composite laminate strength under multiple loading modes
(compression, tension, shear, ILSS).

Supports five porosity configurations across three material presets.
Two solver tiers: empirical (Judd-Wright, power law, linear) and
finite element with Mori-Tanaka-degraded element stiffness.

Based on:
- Judd & Wright - Empirical porosity-strength relationships
- Eshelby (1957) - Inclusion theory for void stress concentration
- Tsai-Wu - 3D failure criterion

Dependencies:
    pip install numpy scipy matplotlib
"""

import argparse
import concurrent.futures
import dataclasses
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

logger = logging.getLogger(__name__)

# ============================================================
# PLOT STYLE — applied once at import time (#53)
# ============================================================
#
# Centralizing rcParams + label text here keeps the Streamlit app
# (``app.py``), the static-PNG path (``FEVisualizer``), and the
# validation runner (``validation/validate_all.py``) from drifting
# apart on font size, DPI, colormap, or axis units.

# Axis / colorbar label text. Importable as module-level constants so
# call sites can do ``from porosity_fe_analysis import LABEL_X_MM`` and
# the GUI / PNG / validation paths share a single source of truth.
LABEL_POROSITY_PCT = "Porosity (%)"
LABEL_POROSITY_VP = "Porosity Vp (%)"
LABEL_X_MM = "x (mm)"
LABEL_Y_MM = "y (mm)"
LABEL_Z_MM = "z (mm)"
LABEL_STIFFNESS_RETENTION = "Stiffness Retention (%)"
LABEL_STIFFNESS_RETENTION_FRAC = "Stiffness Retention (-)"
LABEL_KNOCKDOWN = "Knockdown Factor (-)"
LABEL_SCF = "Stress Concentration Factor (-)"
LABEL_STRESS_MPA = "Stress (MPa)"
LABEL_MAE_PCT = "MAE (%)"

# Legacy dict kept so anything outside this module that still does
# ``LABELS['knockdown_factor']`` keeps working. New code should use the
# ``LABEL_*`` constants above.
LABELS = {
    'porosity_pct': LABEL_POROSITY_PCT,
    'x_mm': LABEL_X_MM,
    'y_mm': LABEL_Y_MM,
    'z_mm': LABEL_Z_MM,
    'stiffness_retention_pct': LABEL_STIFFNESS_RETENTION,
    'knockdown_factor': LABEL_KNOCKDOWN,
    'scf': LABEL_SCF,
}


def _configure_matplotlib_style(style: str = 'default') -> None:
    """Set shared matplotlib rcParams for all plots in the project (#53).

    Parameters
    ----------
    style : {'default', 'publication'}
        ``'default'`` (the import-time setting) is the screen/README
        raster style: 11pt body, 14pt titles, ``savefig.dpi=300``,
        ``image.cmap='cividis'`` (perceptually-uniform + colorblind-
        safe; matches #51).

        ``'publication'`` bumps fonts (~+2pt) for use in figures
        embedded in papers. PNG is retained as the default
        ``savefig.format`` because some downstream consumers
        (Streamlit, GitHub README previews) cannot render PDF inline;
        callers that want vector output should pass an explicit
        ``.pdf`` extension to ``plt.savefig``.
    """
    import matplotlib

    base = {
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        # Colorblind-safe perceptually-uniform colormap; matches the
        # damage-contour fix in #51 so cividis is now the project-wide
        # default. Do NOT switch back to 'viridis'.
        'image.cmap': 'cividis',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    }
    if style == 'publication':
        base.update({
            'font.size': 13,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 11,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.titlesize': 18,
        })
    matplotlib.rcParams.update(base)


# Backwards-compatible alias for callers that imported the old helper.
_apply_plot_style = _configure_matplotlib_style

# Apply at import so any module that imports ``porosity_fe_analysis``
# (the Streamlit app, validation runner, tests) inherits the same style.
_configure_matplotlib_style()


try:
    import importlib.metadata as _ilm
    __version__ = _ilm.version("porosity-fe")
except Exception:
    # Source checkout that isn't pip-installed. Keep in sync with
    # pyproject.toml on each release.
    __version__ = "1.2.0"


def _json_default(o):
    """json.dump ``default=`` hook: make numpy scalars/arrays serializable.

    The science payload is already float()-wrapped, but user-supplied
    fields (e.g. ndarray ply_angles, dataclass configs, datetime stamps)
    would otherwise raise TypeError (#20).
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    # Plain dataclass instances (e.g. MaterialProperties) — accept on a
    # best-effort basis so callers can stash a dataclass field in the
    # config dict without an explicit asdict() at the call site.
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    raise TypeError(
        f"Object of type {type(o).__name__} is not JSON serializable"
    )

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

# ============================================================
# SECTION 4: MESH GENERATION
# ============================================================

class CompositeMesh:
    """3D structured hexahedral mesh of a composite coupon with porosity.

    Builds a regular grid of 8-node hexahedral elements over a
    rectangular coupon of dimensions ``L_x x L_y x L_z``, samples the
    porosity field at every node, assigns a ply id (and optional ply
    angle in degrees) to each element from its centroid z, and flags
    elements whose centroid falls inside any explicit
    :class:`VoidGeometry` for the explicit-inclusion solver path.

    The in-plane coupon size is **hard-coded** to ``L_x = 50.0`` mm and
    ``L_y = 20.0`` mm (a standard ASTM-style coupon). Through-thickness
    ``L_z`` is taken from ``material.total_thickness``
    (``t_ply * n_plies``). To analyze a different coupon size, set
    ``self.L_x`` / ``self.L_y`` on the instance and call
    :meth:`generate_mesh` again.

    Parameters
    ----------
    porosity_field : PorosityField
        Source of nodal porosity values. Sampled by
        :meth:`generate_mesh` at every node coordinate.
    material : MaterialProperties
        Composite material; supplies ``total_thickness`` (``L_z``) and
        ``n_plies`` for the per-element ply id assignment.
    nx, ny, nz : int, optional
        Number of elements along each axis (defaults
        ``nx=50``, ``ny=20``, ``nz=24``). Each must be a positive
        integer not greater than ``_MAX_ELEMENTS_PER_AXIS`` (10 000).
    ply_angles : list of float or {'QI', 'UD'}, optional
        Per-ply orientation in degrees, OR a string sentinel — ``'QI'``
        (default, expands to the 8-ply quasi-isotropic baseline
        ``[0, 90, 45, -45]_s``) or ``'UD'`` (all-zero unidirectional).
        Explicit lists shorter than ``n_plies`` are tiled. Passing
        ``None`` is deprecated and is currently resolved to all-zero
        plies (the historical default) with a
        :class:`DeprecationWarning` (#44 item 2); this back-compat path
        will be removed in a future major version.

    Attributes
    ----------
    porosity_field : PorosityField
        Stored input.
    material : MaterialProperties
        Stored input.
    nx, ny, nz : int
        Element counts along each axis.
    L_x, L_y, L_z : float
        Coupon dimensions in mm. ``L_x = 50.0`` and ``L_y = 20.0`` are
        defaults; ``L_z = material.total_thickness``.
    nodes : np.ndarray
        Shape ``(n_nodes, 3)`` float array of node coordinates (mm),
        populated by :meth:`generate_mesh`.
    elements : np.ndarray
        Shape ``(n_elem, 8)`` int array of node indices per element
        (VTK hexahedron ordering).
    porosity : np.ndarray
        Shape ``(n_nodes,)`` nodal porosity values in ``[0, 1]``.
    stiffness_reduction : np.ndarray
        Shape ``(n_nodes,)`` complementary ``1 - Vp`` field.
    ply_ids : np.ndarray
        Shape ``(n_nodes,)`` per-node ply id (``0`` to ``n_plies - 1``).
    elem_ply_ids : np.ndarray
        Shape ``(n_elem,)`` per-element ply id from the centroid z.
    ply_angles : np.ndarray
        Shape ``(n_elem,)`` per-element ply orientation in degrees.
    void_elements : np.ndarray
        Int indices of elements whose centroid lies inside any discrete
        void.
    void_element_set : set of int
        Same content as ``void_elements`` for O(1) membership tests.
    n_nodes, n_elements, n_dof : int
        Read-only sizes.

    Examples
    --------
    A default-size coupon with the T800 ply and 2 % uniform porosity:

    >>> mat = MATERIALS['T800_epoxy']
    >>> field = PorosityField(mat, void_volume_fraction=0.02,
    ...                       distribution='uniform')
    >>> mesh = CompositeMesh(field, mat, nx=10, ny=4, nz=4)
    >>> mesh.L_x, mesh.L_y
    (50.0, 20.0)
    >>> mesh.n_elements
    160

    Override the in-plane coupon size to 80 mm x 25 mm:

    >>> mesh = CompositeMesh(field, mat, nx=8, ny=4, nz=4)
    >>> mesh.L_x = 80.0
    >>> mesh.L_y = 25.0
    >>> mesh.generate_mesh()

    Notes
    -----
    ``ply_angles`` defaults — ``'QI'`` is the standardised default across
    :class:`EmpiricalSolver`, :class:`CompositeMesh`, and :class:`FESolver`
    (#44 item 2). The string sentinels expand to canonical baselines
    (``'QI'`` -> ``[0, 90, 45, -45]_s``; ``'UD'`` -> all-zero plies);
    explicit lists pass through unchanged. Pass ``ply_angles='UD'`` to
    reproduce the pre-#44 behaviour of leaving every element at
    ``ply_angle = 0``.
    """

    # Cap mesh dimensions to prevent accidental memory blowup. A million-element
    # mesh is already ~100x what the GUI spinboxes allow; an order of magnitude
    # above that is almost certainly a typo or unit confusion.
    _MAX_ELEMENTS_PER_AXIS = 10_000

    def __init__(self, porosity_field: PorosityField, material: MaterialProperties,
                 nx: int = 50, ny: int = 20, nz: int = 24,
                 ply_angles: Optional[Union[List[float], str]] = 'QI'):
        for axis_name, value in (('nx', nx), ('ny', ny), ('nz', nz)):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(
                    f"CompositeMesh.{axis_name} must be a positive integer "
                    f"(elements per axis), got {value!r}."
                )
            if value > self._MAX_ELEMENTS_PER_AXIS:
                raise ValueError(
                    f"CompositeMesh.{axis_name}={value} exceeds the "
                    f"{self._MAX_ELEMENTS_PER_AXIS} per-axis cap. "
                    f"Such a fine mesh would exhaust memory; "
                    f"reduce or split the analysis."
                )

        self.porosity_field = porosity_field
        self.material = material
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.L_x = 50.0
        self.L_y = 20.0
        self.L_z = material.total_thickness

        self.nodes = None
        self.elements = None
        self.porosity = None
        self.stiffness_reduction = None
        self.ply_ids = None
        self.ply_angles = None  # Per-element ply orientation angles (degrees)
        self.void_elements = None

        # Resolve the ply_angles sentinel (#44 item 2). ``None`` is the
        # deprecated path and emits a DeprecationWarning inside
        # ``_resolve_ply_angles``.
        self._input_ply_angles = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='CompositeMesh.ply_angles')
        self.generate_mesh()

    def generate_mesh(self):
        x = np.linspace(0, self.L_x, self.nx + 1)
        y = np.linspace(0, self.L_y, self.ny + 1)
        z = np.linspace(0, self.L_z, self.nz + 1)

        nodes = []

        for zk in z:
            for yj in y:
                for xi in x:
                    nodes.append([xi, yj, zk])

        self.nodes = np.array(nodes)

        # Sample porosity at all nodes
        self.porosity = self.porosity_field.local_porosity(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])
        self.stiffness_reduction = self.porosity_field.local_stiffness_reduction(
            self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2])

        # Ply IDs
        z_normalized = self.nodes[:, 2] / self.L_z
        self.ply_ids = np.clip((z_normalized * self.material.n_plies).astype(int),
                               0, self.material.n_plies - 1)

        # Hex element connectivity
        elements = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n0 = k * (self.ny + 1) * (self.nx + 1) + j * (self.nx + 1) + i
                    n1 = n0 + 1
                    n2 = n0 + (self.nx + 1) + 1
                    n3 = n0 + (self.nx + 1)
                    n4 = n0 + (self.ny + 1) * (self.nx + 1)
                    n5 = n4 + 1
                    n6 = n4 + (self.nx + 1) + 1
                    n7 = n4 + (self.nx + 1)
                    elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

        self.elements = np.array(elements)

        # Identify void elements: check if element centroid falls inside
        # any discrete void geometry (explicit inclusion modeling)
        elem_centers = np.mean(self.nodes[self.elements], axis=1)  # (n_elem, 3)
        void_mask = np.zeros(len(self.elements), dtype=bool)
        for void in self.porosity_field.discrete_voids:
            inside = void.contains(elem_centers[:, 0], elem_centers[:, 1], elem_centers[:, 2])
            void_mask |= inside
        self.void_elements = np.where(void_mask)[0]
        # Also create a set for O(1) lookup
        self.void_element_set = set(self.void_elements.tolist())

        # Assign per-element ply angles (degrees)
        # Element ply_id is determined by the centroid z-coordinate
        elem_centroids_z = np.mean(self.nodes[self.elements][:, :, 2], axis=1)
        elem_ply_ids = np.clip(
            (elem_centroids_z / self.L_z * self.material.n_plies).astype(int),
            0, self.material.n_plies - 1)
        self.elem_ply_ids = elem_ply_ids

        if self._input_ply_angles is not None:
            angle_list = list(self._input_ply_angles)
            if len(angle_list) < self.material.n_plies:
                # Repeat to fill all plies
                angle_list = (angle_list * (self.material.n_plies // len(angle_list) + 1))[:self.material.n_plies]
            self.ply_angles = np.array([angle_list[pid] for pid in elem_ply_ids], dtype=float)
        else:
            # Default: all 0-degree plies
            self.ply_angles = np.zeros(len(self.elements), dtype=float)

        logger.info("Mesh generated: %d nodes, %d elements",
                    len(self.nodes), len(self.elements))
        logger.info("  Domain: %.1f x %.1f x %.2f mm",
                    self.L_x, self.L_y, self.L_z)
        logger.info("  Void elements: %d", len(self.void_elements))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def domain_size(self) -> Tuple[float, float, float]:
        return (self.L_x, self.L_y, self.L_z)

    def nodes_on_face(self, face: str) -> np.ndarray:
        """Return node indices on the specified face.

        Parameters
        ----------
        face : str
            One of 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.

        Returns
        -------
        np.ndarray
            1-D array of node indices on that face.
        """
        tol = 1e-8
        coords = self.nodes
        if face == 'x_min':
            return np.where(np.abs(coords[:, 0] - coords[:, 0].min()) < tol)[0]
        elif face == 'x_max':
            return np.where(np.abs(coords[:, 0] - coords[:, 0].max()) < tol)[0]
        elif face == 'y_min':
            return np.where(np.abs(coords[:, 1] - coords[:, 1].min()) < tol)[0]
        elif face == 'y_max':
            return np.where(np.abs(coords[:, 1] - coords[:, 1].max()) < tol)[0]
        elif face == 'z_min':
            return np.where(np.abs(coords[:, 2] - coords[:, 2].min()) < tol)[0]
        elif face == 'z_max':
            return np.where(np.abs(coords[:, 2] - coords[:, 2].max()) < tol)[0]
        else:
            raise ValueError(f"Unknown face '{face}'. Use x_min/x_max/y_min/y_max/z_min/z_max.")

    def find_nodes_near(self, x: Optional[float] = None,
                        y: Optional[float] = None,
                        z: Optional[float] = None,
                        tol: Optional[float] = None) -> np.ndarray:
        """Return node indices within ``tol`` of the specified target coords.

        Any of ``x``, ``y``, ``z`` may be ``None``, in which case that axis
        is not used in the distance computation (i.e. the search becomes a
        line/plane match rather than a point match). Distances are computed
        with ``np.linalg.norm`` on the subset of axes that were specified.

        Parameters
        ----------
        x, y, z : float or None
            Target coordinate per axis. Pass ``None`` to ignore an axis.
        tol : float or None
            Distance tolerance. If ``None``, defaults to half of a typical
            element edge length (``0.5 * min(L_x/nx, L_y/ny, L_z/nz)``).

        Returns
        -------
        np.ndarray
            Sorted 1-D array of node indices whose distance to the target
            (restricted to the specified axes) is ``<= tol``.

        Notes
        -----
        Used by ILSS short-beam BCs to locate midspan-top loading nodes
        even when ``Lx / 2`` does not coincide with a mesh node.
        """
        if x is None and y is None and z is None:
            raise ValueError(
                "find_nodes_near: at least one of x/y/z must be specified."
            )
        if tol is None:
            dxs = []
            if self.nx > 0:
                dxs.append(self.L_x / self.nx)
            if self.ny > 0:
                dxs.append(self.L_y / self.ny)
            if self.nz > 0:
                dxs.append(self.L_z / self.nz)
            tol = 0.5 * min(dxs)

        coords = self.nodes
        targets = []
        cols = []
        if x is not None:
            targets.append(float(x))
            cols.append(0)
        if y is not None:
            targets.append(float(y))
            cols.append(1)
        if z is not None:
            targets.append(float(z))
            cols.append(2)

        diffs = coords[:, cols] - np.asarray(targets, dtype=float)
        dist = np.linalg.norm(diffs, axis=1)
        return np.where(dist <= tol)[0]

    def __repr__(self) -> str:
        return (f"CompositeMesh(nx={self.nx}, ny={self.ny}, nz={self.nz}, "
                f"n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
                f"domain={self.L_x:.1f}x{self.L_y:.1f}x{self.L_z:.2f}mm, "
                f"void_elements={len(self.void_elements)})")


def check_mesh_quality(mesh: CompositeMesh, verbose: bool = False) -> Dict:
    """Check mesh quality: element aspect ratios and Jacobian determinants.

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh to check.
    verbose : bool
        Print detailed quality report.

    Returns
    -------
    dict
        Quality metrics: min/max aspect ratio, min Jacobian determinant,
        number of inverted elements, number of highly distorted elements.

    Raises
    ------
    Warning messages are printed for inverted or highly distorted elements.
    """
    import warnings

    n_elem = mesh.n_elements
    aspect_ratios = np.empty(n_elem)
    min_detJ_per_elem = np.empty(n_elem)

    for e in range(n_elem):
        node_ids = mesh.elements[e]
        coords = mesh.nodes[node_ids]  # (8, 3)

        # Aspect ratio: ratio of max edge length to min edge length
        # Check all 12 edges of a hexahedron
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]
        edge_lengths = np.array([np.linalg.norm(coords[a] - coords[b])
                                  for a, b in edges])
        min_len = edge_lengths.min()
        max_len = edge_lengths.max()
        aspect_ratios[e] = max_len / min_len if min_len > 1e-15 else np.inf

        # Jacobian at element center
        dN = Hex8Element.shape_derivatives(0.0, 0.0, 0.0)
        J = dN @ coords
        min_detJ_per_elem[e] = np.linalg.det(J)

    n_inverted = int(np.sum(min_detJ_per_elem < 0))
    n_distorted = int(np.sum(aspect_ratios > 20.0))

    result = {
        'min_aspect_ratio': float(np.min(aspect_ratios)),
        'max_aspect_ratio': float(np.max(aspect_ratios)),
        'mean_aspect_ratio': float(np.mean(aspect_ratios)),
        'min_jacobian_det': float(np.min(min_detJ_per_elem)),
        'n_inverted': n_inverted,
        'n_distorted': n_distorted,
        'n_elements': n_elem,
    }

    if verbose:
        logger.info("  Mesh quality: %d elements", n_elem)
        logger.info(
            "    Aspect ratio: min=%.2f, max=%.2f, mean=%.2f",
            result['min_aspect_ratio'],
            result['max_aspect_ratio'],
            result['mean_aspect_ratio'],
        )
        logger.info("    Min Jacobian det: %.6e", result['min_jacobian_det'])
        if n_inverted > 0:
            logger.warning(
                "    WARNING: %d inverted elements (negative Jacobian)!",
                n_inverted,
            )
        if n_distorted > 0:
            logger.warning(
                "    WARNING: %d highly distorted elements (aspect ratio > 20)!",
                n_distorted,
            )

    if n_inverted > 0:
        warnings.warn(
            f"Mesh has {n_inverted} inverted elements (negative Jacobian determinant).",
            stacklevel=2,
        )
    if n_distorted > 0:
        warnings.warn(
            f"Mesh has {n_distorted} highly distorted elements (aspect ratio > 20).",
            stacklevel=2,
        )

    return result


# ============================================================
# SECTION 5: EMPIRICAL SOLVER
# ============================================================

# ---- API consistency dataclasses (#44) ----
#
# These dataclasses unify the previously divergent return shapes from the
# empirical and FE solvers, and slim the return value of
# :func:`compare_configurations` so batch loops do not retain heavyweight
# live objects (mesh / solver / field) when only the headline numbers are
# needed.
#
# Back-compat: callers that historically treated the returned object as a
# dict (e.g. ``result['failure_stress']``) keep working because each
# dataclass exposes ``__getitem__`` mapping the old keys to attribute
# access. New code should prefer attribute access (``result.failure_stress``).
# The dict shim will be removed in a future major version.

#: Canonical QI baseline layup (8-ply symmetric ``[0/90/45/-45]_s``).
#: Used to expand the ``ply_angles='QI'`` sentinel (#44 item 2).
_PLY_ANGLES_QI: Tuple[float, ...] = (0.0, 90.0, 45.0, -45.0, -45.0, 45.0, 90.0, 0.0)

#: Canonical UD baseline (4 plies, all 0 deg). Used to expand the
#: ``ply_angles='UD'`` sentinel; the FE / empirical scaling only cares about
#: the angle distribution, so a short list is fine.
_PLY_ANGLES_UD: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


def _resolve_ply_angles(
    ply_angles: Optional[Union[List[float], Tuple[float, ...], str]],
    *,
    none_means: str = 'QI',
    caller: str = 'ply_angles',
) -> Optional[List[float]]:
    """Resolve the ``ply_angles`` sentinel to a concrete list of angles.

    Unifies the three previously divergent ``ply_angles=None`` defaults
    across :class:`EmpiricalSolver`, :class:`CompositeMesh` and
    :class:`FESolver` (#44 item 2). The resolved value is:

    - ``'QI'`` -> ``[0, 90, 45, -45, -45, 45, 90, 0]`` (8-ply symmetric
      quasi-isotropic). The standardised default for new code.
    - ``'UD'`` -> ``[0, 0, 0, 0]`` (unidirectional).
    - A list / tuple of floats -> returned verbatim (as a list).
    - ``None`` -> resolved to ``none_means`` (default ``'QI'``) and emits a
      :class:`DeprecationWarning`. Class-specific call sites override
      ``none_means`` if they need to preserve the prior class-specific
      default during the deprecation window.

    Returns
    -------
    list of float or None
        ``None`` is returned for :class:`CompositeMesh`'s historical
        ``None`` -> all-zero behaviour when ``none_means='UD_legacy'``; the
        empirical / FE paths always get an explicit list back.
    """
    if isinstance(ply_angles, str):
        key = ply_angles.upper()
        if key == 'QI':
            return list(_PLY_ANGLES_QI)
        if key == 'UD':
            return list(_PLY_ANGLES_UD)
        raise ValueError(
            f"{caller} string sentinel must be 'QI' or 'UD', got {ply_angles!r}."
        )
    if ply_angles is None:
        # Back-compat shim — emit a DeprecationWarning and resolve to the
        # standardised default. Planned removal in a future major version.
        warnings.warn(
            f"Passing {caller}=None is deprecated; pass {none_means!r} (or "
            f"'UD', or an explicit list of ply angles) instead. None is "
            f"resolved to {none_means!r} for back-compat and will be removed "
            "in a future major version (#44).",
            DeprecationWarning,
            stacklevel=3,
        )
        if none_means == 'QI':
            return list(_PLY_ANGLES_QI)
        if none_means == 'UD':
            return list(_PLY_ANGLES_UD)
        # 'UD_legacy' preserves the historical "None means literal zero array"
        # behaviour for CompositeMesh — same as 'UD' angle-wise.
        if none_means == 'UD_legacy':
            return None
        raise ValueError(
            f"Internal: unsupported none_means={none_means!r}."
        )
    # Concrete sequence — convert to list of floats for hashability /
    # reproducibility and validate entries.
    angle_list = [float(a) for a in ply_angles]
    return angle_list


@dataclass
class FailureResult:
    """Unified failure-load summary returned by empirical and FE solvers.

    Adds API consistency across :meth:`EmpiricalSolver.get_failure_load`
    (historically dict-returning) and :meth:`FESolver.solve` (returns the
    richer :class:`FieldResults`, distilled here via
    :meth:`FieldResults.summary`). Callers can now treat the two solvers
    polymorphically (#44 item 1).

    Attributes
    ----------
    failure_stress : float
        Failure stress magnitude in MPa. For the empirical solver this is
        ``knockdown * sigma_pristine`` at the specimen-average ``Vp``; for
        the FE solver (distilled via :meth:`FieldResults.summary`) it is
        ``knockdown * sigma_pristine`` evaluated with the loading-mode-
        specific pristine strength. Reported as a positive magnitude
        regardless of loading sign (compression strengths are stored as
        positive numbers in :class:`MaterialProperties`).
    knockdown : float
        Knockdown factor in ``(0, 1]``. Same definition the source solver
        used; bit-identical to what the legacy dict / float returns
        produced.
    model : str
        Knockdown model label (``'judd_wright'`` / ``'power_law'`` /
        ``'linear'`` / ``'user_callable'`` for the empirical solver, and
        ``'fe_<criterion>'`` for the FE solver's summary).
    details : dict
        Free-form solver-specific extras (e.g. ``'critical_location'`` for
        the empirical path, or ``'max_failure_index'`` /
        ``'failure_criterion'`` from the FE solver). Always JSON-friendly.

    Notes
    -----
    Back-compat: callers that historically accessed
    ``result['failure_stress']`` / ``result['knockdown']`` / ``result['model']``
    / ``result['critical_location']`` keep working via the ``__getitem__``
    shim below. The shim maps the four documented dict keys to attribute /
    ``details`` access; any other key raises :class:`KeyError`. New code
    should use attribute access. The dict shim will be removed in a future
    major version.
    """
    failure_stress: float
    knockdown: float
    model: str
    details: dict = field(default_factory=dict)

    # Dict keys served by the back-compat shim. ``critical_location`` is
    # routed through ``details`` because not every distilled FailureResult
    # has a meaningful crack location (the FE summary uses a max-FI
    # element index instead).
    _DICT_KEYS_DIRECT = ('failure_stress', 'knockdown', 'model')

    def __getitem__(self, key: str):
        """Back-compat dict-style access (deprecated; will be removed)."""
        if key in self._DICT_KEYS_DIRECT:
            return getattr(self, key)
        if key in self.details:
            return self.details[key]
        raise KeyError(
            f"{key!r} is not a known FailureResult field. "
            f"Known: {sorted(set(self._DICT_KEYS_DIRECT) | set(self.details))}."
        )

    def __contains__(self, key: object) -> bool:
        return key in self._DICT_KEYS_DIRECT or key in self.details

    def get(self, key: str, default=None):
        """Dict-style :meth:`dict.get` for the back-compat shim."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Iterable of known dict-style keys (back-compat shim)."""
        return list(self._DICT_KEYS_DIRECT) + list(self.details.keys())

    def to_dict(self) -> dict:
        """Return a plain dict matching the legacy
        :meth:`EmpiricalSolver.get_failure_load` return shape."""
        out = {k: getattr(self, k) for k in self._DICT_KEYS_DIRECT}
        out.update(self.details)
        return out


@dataclass
class ConfigArtifacts:
    """Live solver objects retained for one configuration in a sweep.

    Returned in the second slot of the :func:`compare_configurations` tuple
    when ``return_artifacts=True``. Keeping these out of the default
    :class:`ConfigResult` lets batch loops over many sweeps hold only the
    headline numbers in memory (#44 item 3).

    Attributes
    ----------
    mesh : CompositeMesh
        The mesh used for the empirical / FE solves.
    empirical_solver : EmpiricalSolver
        The empirical solver constructed for this configuration.
    porosity_field : PorosityField
        The porosity field used for this configuration.
    field_results : FieldResults or None
        Populated only if the FE solver was run for this configuration
        (``compare_configurations`` is empirical-only today, so this is
        ``None`` from that path).
    """
    mesh: 'CompositeMesh'
    empirical_solver: 'EmpiricalSolver'
    porosity_field: 'PorosityField'
    field_results: Optional['FieldResults'] = None


@dataclass
class ConfigResult:
    """Lightweight (numbers-only) per-configuration result from
    :func:`compare_configurations`.

    Holds only JSON-friendly scalars plus the nested ``empirical`` dict
    (the headline knockdown / failure_stress tables already produced by
    :meth:`EmpiricalSolver.get_all_failure_loads`). Heavy live objects
    (mesh, empirical_solver, porosity_field) are kept on the parallel
    :class:`ConfigArtifacts` mapping, returned only when
    ``compare_configurations(..., return_artifacts=True)`` is requested.

    Attributes
    ----------
    Vp : float
        Specimen-average void volume fraction in [0, 1].
    config_name : str
        Configuration name (key in :data:`POROSITY_CONFIGS`).
    config : dict
        :class:`PorosityField` constructor kwargs for this configuration.
    failure_stress : float
        Headline compression failure stress (Judd-Wright model) in MPa.
        Convenience scalar; the full per-mode/per-model table is on
        :attr:`empirical`.
    knockdown : float
        Headline compression knockdown (Judd-Wright). Convenience scalar.
    model : str
        Label of the headline knockdown model. Always ``'judd_wright'``
        for the default sweep; preserved as a field so future overrides
        flow through.
    empirical : dict
        The nested empirical-knockdown table from
        :meth:`EmpiricalSolver.get_all_failure_loads`; structure is
        ``{mode: {model: FailureResult}}``. Kept on the result so the
        existing JSON exporter, plot helpers, and tests can continue to
        read ``cfg['empirical']['compression']['judd_wright']['knockdown']``
        without touching the artifacts dict.
    seed : int or None
        The reproducibility seed recorded on the underlying
        :class:`PorosityField` (mirrors the input to
        :func:`compare_configurations`). Carried on the lightweight
        result so the JSON exporter's provenance block can recover it
        without holding the live ``porosity_field`` (#55 / #44 item 3).

    Notes
    -----
    Back-compat: this object supports dict-style item access for the
    documented keys above (``'Vp'``, ``'config'``, ``'config_name'``,
    ``'failure_stress'``, ``'knockdown'``, ``'model'``, ``'empirical'``,
    ``'seed'``) so legacy callers do not break. Any *other* key —
    notably the legacy ``'mesh'`` / ``'empirical_solver'`` /
    ``'porosity_field'`` keys — raises :class:`KeyError` with a hint
    pointing at ``return_artifacts=True``. The dict shim will be removed
    in a future major version.
    """
    Vp: float
    config_name: str
    config: dict
    failure_stress: float
    knockdown: float
    model: str
    empirical: dict
    seed: Optional[int] = None

    _DICT_KEYS_DIRECT = (
        'Vp', 'config_name', 'config', 'failure_stress',
        'knockdown', 'model', 'empirical', 'seed',
    )
    # Old keys callers used to find via the dict — surface a helpful
    # KeyError now that they live on :class:`ConfigArtifacts`.
    _ARTIFACT_KEYS = ('mesh', 'empirical_solver', 'porosity_field', 'field_results')

    def __getitem__(self, key: str):
        if key in self._DICT_KEYS_DIRECT:
            return getattr(self, key)
        if key in self._ARTIFACT_KEYS:
            raise KeyError(
                f"{key!r} is no longer carried on the default "
                f"compare_configurations result (#44). Re-run with "
                f"`return_artifacts=True` and read it from the parallel "
                f"artifacts dict."
            )
        raise KeyError(
            f"{key!r} is not a known ConfigResult field. "
            f"Known: {sorted(self._DICT_KEYS_DIRECT)}."
        )

    def __contains__(self, key: object) -> bool:
        return key in self._DICT_KEYS_DIRECT

    def get(self, key: str, default=None):
        """Dict-style :meth:`dict.get` for the back-compat shim."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Iterable of known dict-style keys (back-compat shim)."""
        return list(self._DICT_KEYS_DIRECT)


# ============================================================
# Fatigue (S-N) knockdown surface (issue #59).
#
# Log-linear (Mandell) fatigue knockdown::
#
#     S_N / S_0 = max(floor, 1 - b * log10(N))
#
# with mode-keyed slopes ``b``. The table below is calibrated against
# representative CFRP S-N data at ``R = 0.1`` (tension-tension): the
# tension/compression slopes follow the Mandell 1991 review value of
# b ~ 0.1 per decade for IM-class CFRP; ILSS / matrix-dominated shear
# is slightly shallower (b ~ 0.08) as reported by Curtis (1989) and the
# WWFE-III fatigue exercise. These are screening-level values; production
# allowables should use a fully populated S-N matrix per CMH-17 Vol. 2.
#
# References:
# - Mandell, J. F., "Fatigue Behavior of Fiber-Resin Composites,"
#   Developments in Reinforced Plastics 2, 1991.
# - Curtis, P. T., "The fatigue behaviour of fibrous composite materials,"
#   J. Strain Analysis, 1989.
# ============================================================
_FATIGUE_B_QI: Dict[str, float] = {
    'tension': 0.10,
    'compression': 0.10,
    'shear': 0.08,
    'ilss': 0.08,
    'transverse_tension': 0.10,
}

# Floor clamp for the log-linear formula: at very large N the linear
# extrapolation predicts a negative knockdown. Clamp to 1% of static so
# downstream multiplications stay well-behaved, and emit a warning so the
# caller knows they are off the calibration range.
_FATIGUE_KD_FLOOR = 0.01


@dataclass
class FatigueModel:
    """S-N (cycles-to-failure) knockdown surface.

    Implements the log-linear (Mandell-style) form::

        S_N / S_0 = max(floor, 1 - b * log10(N))

    where ``b`` is a mode-keyed slope (see :data:`_FATIGUE_B_QI`),
    ``N`` is the number of load cycles, and ``floor`` (default 0.01)
    is a small lower clamp that prevents the linear extrapolation from
    going negative at very large ``N``.

    The default slopes are calibrated for quasi-isotropic CFRP at
    ``R = 0.1`` (typical tension-tension fatigue). The ``R`` argument is
    currently informational only — future revisions can wrap the base
    formula with a Goodman / Walker R-correction.

    Attributes
    ----------
    b : dict[str, float], optional
        Mode-keyed slope override. Modes absent from the override fall
        back to :data:`_FATIGUE_B_QI`.

    Notes
    -----
    For ``cycles = None`` callers should bypass this class entirely (the
    :meth:`EmpiricalSolver.get_failure_load` path returns ``1.0`` in that
    case). The model is screening-level; production allowables should
    come from a fully populated test matrix per the applicable spec
    (e.g. CMH-17 Vol. 2 fatigue protocols).
    """
    b: Optional[Dict[str, float]] = None

    def _slope(self, mode: str) -> float:
        if mode not in _FATIGUE_B_QI:
            raise ValueError(
                f"Unknown fatigue mode {mode!r}. "
                f"Use one of {sorted(_FATIGUE_B_QI)}."
            )
        if self.b is not None and mode in self.b:
            return float(self.b[mode])
        return float(_FATIGUE_B_QI[mode])

    def knockdown_factor(self, mode: str, cycles: float,
                         R: float = 0.1) -> float:
        """Multiplicative fatigue knockdown for the given mode.

        Parameters
        ----------
        mode : str
            Loading mode (see :data:`_FATIGUE_B_QI`).
        cycles : float
            Number of load cycles ``N``. Must be a positive finite
            value; ``cycles = 1`` corresponds to the static (one-cycle)
            allowable and returns ``1.0``.
        R : float, optional
            Stress ratio ``sigma_min / sigma_max``. Currently
            informational (default 0.1, tension-tension); reserved
            for a future Goodman / Walker R-correction.

        Returns
        -------
        float
            Knockdown factor in ``[floor, 1.0]``. When the linear
            extrapolation would go below the floor (``floor = 0.01``),
            the value is clamped and a :class:`UserWarning` is emitted.
        """
        # Reserved for future R-correction; today it is purely
        # informational. Validate finiteness so a stray nan can't slip
        # through silently.
        if not np.isfinite(float(R)):
            raise ValueError(
                f"FatigueModel.knockdown_factor: R must be finite, got {R!r}."
            )

        N = float(cycles)
        if not np.isfinite(N) or N < 1.0:
            raise ValueError(
                f"FatigueModel.knockdown_factor: cycles must be a finite "
                f"value >= 1, got {cycles!r}."
            )

        b = self._slope(mode)
        raw = 1.0 - b * np.log10(N)
        if raw < _FATIGUE_KD_FLOOR:
            warnings.warn(
                f"Fatigue knockdown for mode={mode!r}, cycles={N:.3g}, "
                f"R={R!r} extrapolates to {raw:.3g} (<= floor "
                f"{_FATIGUE_KD_FLOOR}); clamping. The log-linear model is "
                f"off its calibration range — consider a richer S-N model.",
                UserWarning,
                stacklevel=2,
            )
            return _FATIGUE_KD_FLOOR
        return float(min(raw, 1.0))


class EmpiricalSolver:
    """Fast analytical solver using empirical porosity-strength models.

    Coefficients are calibrated against quasi-isotropic data and scaled by
    a layup-dependent matrix-dominated fraction so that fiber-dominated
    layups (e.g. UD [0]_n) see a smaller porosity penalty than QI layups.

    Knockdowns are evaluated at the specimen-average porosity (Vp_mean),
    matching how the original correlations were calibrated — not at the
    local peak Vp that clustered distributions produce.
    """

    # QI-calibrated coefficients (Elhajjar 2025, Sci. Rep. 15:25977).
    # `_F_MD_REF = 0.5` below is the LAYUP-SCALING reference (scale = 1.0 at
    # f_md = 0.5), NOT a property of the Elhajjar coupon layup itself
    # (`[0/45/90/-45/0]_s`, which the binning rule below puts at f_md = 0.4).
    # The coefficients were tuned with the layup-scaling already applied,
    # so they represent the model's effective f_md = 0.5 baseline rather
    # than the raw fit on a single layup.
    # See README "Empirical Strength Knockdown" for definitions, units (alpha, n
    # are dimensionless when Vp is a fraction in [0, 1]), validity bounds, and
    # the calibration recipe for custom materials.
    # Modes: 'compression' (sigma_1c, fiber+matrix), 'tension' (sigma_1t,
    # fiber-dominated), 'shear' (tau_12, in-plane, matrix-dominated),
    # 'ilss' (tau_ilss, short-beam, matrix/interface-dominated),
    # 'transverse_tension' (sigma_2t, in-plane transverse, matrix-dominated;
    # alpha matched to ilss because both fail by matrix/interface-dominated
    # mechanisms — see issue #35).
    _JUDD_WRIGHT_ALPHA_QI = {
        'compression': 6.9, 'tension': 3.9, 'shear': 8.0, 'ilss': 10.0,
        'transverse_tension': 10.0,
    }
    _POWER_LAW_N_QI = {
        'compression': 2.8, 'tension': 1.8, 'shear': 3.5, 'ilss': 4.5,
        'transverse_tension': 4.5,
    }
    _LINEAR_BETA_QI = {
        'compression': 5.5, 'tension': 3.5, 'shear': 7.0, 'ilss': 9.0,
        'transverse_tension': 9.0,
    }
    PRISTINE_STRENGTH_KEY = {
        'compression': 'sigma_1c', 'tension': 'sigma_1t',
        'shear': 'tau_12', 'ilss': 'tau_ilss',
        'transverse_tension': 'sigma_2t',
    }
    # Modes whose porosity sensitivity is matrix-/interface-dominated even in
    # UD layups (where the longitudinal-fiber metric would otherwise drive
    # f_md to ~0).  These modes use the elevated ``_F_MD_FLOOR_ILSS`` floor.
    _MATRIX_DOMINATED_MODES = frozenset({'ilss', 'transverse_tension'})
    # QI reference fraction and minimum floor
    _F_MD_REF = 0.5    # f_md for the QI layup used in calibration
    _F_MD_FLOOR = 0.15  # even UD has some matrix sensitivity
    _F_MD_FLOOR_ILSS = 0.80  # ILSS / transverse-tension are always matrix-dominated

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 ply_angles: Optional[Union[List[float], str]] = 'QI',
                 *,
                 judd_wright_alpha: Optional[Dict[str, float]] = None,
                 power_law_n: Optional[Dict[str, float]] = None,
                 linear_beta: Optional[Dict[str, float]] = None):
        """Empirical knockdown solver.

        Parameters
        ----------
        mesh : CompositeMesh
            Mesh whose nodal porosity drives the knockdown.
        material : MaterialProperties
            Composite material; supplies pristine strengths.
        ply_angles : list of float or {'QI', 'UD'}, optional
            Per-ply orientation in degrees, OR a string sentinel:

            - ``'QI'`` (default) -> ``[0, 90, 45, -45]_s`` quasi-isotropic
              baseline (``f_md = 0.5``, matches the calibration basis).
            - ``'UD'`` -> ``[0, 0, 0, 0]`` unidirectional baseline.
            - explicit list of floats -> used verbatim.

            Passing ``None`` is deprecated and currently resolved to
            ``'QI'`` with a :class:`DeprecationWarning` (#44 item 2);
            this back-compat path will be removed in a future major
            version. ``judd_wright_alpha`` / ``power_law_n`` /
            ``linear_beta`` are optional partial overrides for the
            QI-calibrated coefficients (see README "Empirical Strength
            Knockdown"). Each accepts a dict keyed by mode
            (``'compression'`` / ``'tension'`` / ``'shear'`` / ``'ilss'``);
            modes that are absent fall back to the QI defaults. Override
            values are layup-scaled exactly like the defaults: at
            ``f_md = 0.5`` the scale is 1.0, so a passed-in ``alpha`` is
            the value used directly.

        Notes
        -----
        ``ply_angles`` defaults — ``'QI'`` is the standardised default
        across :class:`EmpiricalSolver`, :class:`CompositeMesh`, and
        :class:`FESolver` (#44 item 2). The string sentinels expand to
        canonical baselines; explicit lists pass through unchanged.
        """
        self.mesh = mesh
        self.material = material
        self.nodal_knockdown = None

        # Resolve the ply_angles sentinel (#44 item 2). ``None`` is the
        # deprecated path and emits a DeprecationWarning inside
        # ``_resolve_ply_angles``.
        ply_angles_resolved = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='EmpiricalSolver.ply_angles')

        # Resolve coefficient dicts: per-mode merge of class default with override.
        alpha_qi = self._merge_coefficient_override(
            self._JUDD_WRIGHT_ALPHA_QI, judd_wright_alpha, 'judd_wright_alpha')
        n_qi = self._merge_coefficient_override(
            self._POWER_LAW_N_QI, power_law_n, 'power_law_n')
        beta_qi = self._merge_coefficient_override(
            self._LINEAR_BETA_QI, linear_beta, 'linear_beta')

        # Compute layup-dependent scaling
        self.f_md = self._matrix_dominated_fraction(ply_angles_resolved)

        # Build scaled coefficient dicts
        self.JUDD_WRIGHT_ALPHA = {}
        self.POWER_LAW_N = {}
        self.LINEAR_BETA = {}
        for mode in ['compression', 'tension', 'shear', 'ilss',
                     'transverse_tension']:
            s = self._layup_scale(mode)
            self.JUDD_WRIGHT_ALPHA[mode] = alpha_qi[mode] * s
            self.POWER_LAW_N[mode] = max(n_qi[mode] * s, 0.1)
            self.LINEAR_BETA[mode] = beta_qi[mode] * s

    @classmethod
    def _merge_coefficient_override(cls, defaults: Dict[str, float],
                                    override: Optional[Dict[str, float]],
                                    name: str) -> Dict[str, float]:
        """Validate ``override`` and merge it onto ``defaults``."""
        if override is None:
            return dict(defaults)
        if not isinstance(override, dict):
            raise TypeError(
                f"{name} must be a dict mapping mode -> coefficient, "
                f"got {type(override).__name__}."
            )
        valid_modes = set(cls.PRISTINE_STRENGTH_KEY)
        unknown = set(override) - valid_modes
        if unknown:
            raise ValueError(
                f"{name} has unknown mode keys {sorted(unknown)}. "
                f"Use a subset of {sorted(valid_modes)}."
            )
        for mode, value in override.items():
            if not isinstance(value, (int, float, np.floating, np.integer)):
                raise TypeError(
                    f"{name}[{mode!r}] must be a number, "
                    f"got {type(value).__name__}."
                )
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name}[{mode!r}] must be a positive finite number, got {value!r}."
                )
        return {**defaults, **override}

    @staticmethod
    def _matrix_dominated_fraction(ply_angles: Optional[List[float]]) -> float:
        """Fraction of matrix-dominated plies in the layup (0 to 1).

        - 0-degree plies contribute 0 (fiber-dominated)
        - +/-45-degree plies contribute 0.5 (intermediate)
        - 90-degree plies contribute 1.0 (matrix-dominated)

        Returns 0.5 (QI reference) if ply_angles is None.
        """
        if ply_angles is None or len(ply_angles) == 0:
            return 0.5  # default = QI reference
        total = 0.0
        for angle in ply_angles:
            a = abs(angle) % 180
            if a <= 10:        # near 0°
                total += 0.0
            elif a >= 80:      # near 90°
                total += 1.0
            else:              # off-axis (30°, 45°, 60°, etc.)
                total += 0.5
        return total / len(ply_angles)

    def _layup_scale(self, mode: str) -> float:
        """Scaling factor for empirical coefficients based on layup.

        Maps f_md to a coefficient multiplier:
        - f_md = f_md_ref (0.5, QI) -> scale = 1.0 (unchanged)
        - f_md = 0 (UD) -> scale = floor (0.15 for most modes, 0.80 for ILSS)
        - f_md > f_md_ref -> scale > 1.0 (more matrix-dominated than QI)
        """
        floor = (self._F_MD_FLOOR_ILSS if mode in self._MATRIX_DOMINATED_MODES
                 else self._F_MD_FLOOR)
        ref = self._F_MD_REF
        if ref < 1e-12:
            return 1.0
        raw = self.f_md / ref
        return max(raw, floor)

    @staticmethod
    def _check_internal_Vp(Vp: float) -> float:
        # Defensive: tolerate fp overshoot (~1e-15) from element-mean averaging
        # by clipping to [0, 1]; reject non-finite outright.
        if not np.isfinite(Vp):
            raise ValueError(f"Internal Vp is non-finite: {Vp!r}")
        return float(np.clip(Vp, 0.0, 1.0))

    @staticmethod
    def _validate_user_kd_callable(model_func: Callable[[float, str], float],
                                   mode: str,
                                   max_Vp: float = 1.0,
                                   n_grid: int = 11) -> None:
        """Validate that a user-supplied knockdown callable is well-behaved.

        The callable contract is ``model(Vp: float, mode: str) -> float in
        [0, 1]``. We sample ``Vp`` on a uniform grid over ``[0, max_Vp]`` and
        check that the return is finite and in the closed unit interval.
        Reuses :meth:`_check_internal_Vp` so the same overshoot/finite policy
        applies as the built-in models.

        Raises
        ------
        TypeError
            If ``model_func`` is not callable.
        ValueError
            If the callable returns a non-finite value, or a value outside
            ``[0, 1]``, on any grid point.
        """
        if not callable(model_func):
            raise TypeError(
                f"User knockdown model must be callable; got "
                f"{type(model_func).__name__}."
            )
        grid = np.linspace(0.0, float(max_Vp), int(n_grid))
        for Vp in grid:
            Vp = EmpiricalSolver._check_internal_Vp(float(Vp))
            try:
                kd = model_func(Vp, mode)
            except Exception as exc:
                raise ValueError(
                    f"User knockdown model raised {type(exc).__name__} at "
                    f"Vp={Vp:.3f}, mode={mode!r}: {exc}"
                ) from exc
            kd_f = float(kd)
            if not np.isfinite(kd_f):
                raise ValueError(
                    f"User knockdown model returned non-finite value "
                    f"{kd_f!r} at Vp={Vp:.3f}, mode={mode!r}; "
                    f"expected a finite float in [0, 1]."
                )
            if kd_f < 0.0 or kd_f > 1.0:
                raise ValueError(
                    f"User knockdown model returned {kd_f!r} at Vp={Vp:.3f}, "
                    f"mode={mode!r}; expected a value in [0, 1]."
                )

    def _judd_wright(self, Vp: float, mode: str) -> float:
        """Judd-Wright knockdown: KD = exp(-alpha * Vp).

        Vp is a void volume fraction in [0, 1]. ``alpha`` is the
        layup-scaled, mode-specific sensitivity coefficient (see
        ``JUDD_WRIGHT_ALPHA`` and the README "Empirical Strength
        Knockdown" section for definitions and ranges).
        """
        Vp = self._check_internal_Vp(Vp)
        alpha = self.JUDD_WRIGHT_ALPHA[mode]
        return float(np.exp(-alpha * Vp))

    def _power_law(self, Vp: float, mode: str) -> float:
        """Power-law knockdown: KD = (1 - Vp)**n.

        Vp is a void volume fraction in [0, 1]. ``n`` is the
        layup-scaled, mode-specific exponent (see ``POWER_LAW_N`` and
        the README "Empirical Strength Knockdown" section for
        definitions and ranges).
        """
        Vp = self._check_internal_Vp(Vp)
        n = self.POWER_LAW_N[mode]
        return float((1.0 - Vp)**n)

    def _linear(self, Vp: float, mode: str) -> float:
        Vp = self._check_internal_Vp(Vp)
        beta = self.LINEAR_BETA[mode]
        return float(max(1.0 - beta * Vp, 0.0))

    def _get_pristine_strength(self, mode: str) -> float:
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        return getattr(self.material, self.PRISTINE_STRENGTH_KEY[mode])

    def _environment_knockdown_factor(self, mode: str,
                                      environment: Optional[Dict[str, float]]
                                      ) -> float:
        """Resolve the hygrothermal knockdown factor (issue #59).

        Returns ``1.0`` (back-compat no-op) when ``environment`` is
        ``None``. Otherwise delegates to
        :meth:`MaterialProperties.environment_knockdown`, pulling ``T``
        and ``M`` out of the dict (either key is optional — a missing
        key falls back to the material's ``T_service`` / ``M_service``).
        """
        if environment is None:
            return 1.0
        if not isinstance(environment, dict):
            raise TypeError(
                f"environment must be a dict (e.g. {{'T': 80.0, 'M': 1.2}}), "
                f"got {type(environment).__name__}."
            )
        allowed_keys = {'T', 'M'}
        unknown = set(environment) - allowed_keys
        if unknown:
            raise ValueError(
                f"environment has unknown keys {sorted(unknown)}. "
                f"Use a subset of {sorted(allowed_keys)}."
            )
        T = environment.get('T')
        M = environment.get('M')
        return float(self.material.environment_knockdown(mode, T=T, M=M))

    def _fatigue_knockdown_factor(self, mode: str,
                                  cycles: Optional[float],
                                  R: Optional[float]) -> float:
        """Resolve the S-N fatigue knockdown factor (issue #59).

        Returns ``1.0`` (back-compat no-op) when ``cycles`` is ``None``.
        Otherwise instantiates a default :class:`FatigueModel` and
        evaluates :meth:`FatigueModel.knockdown_factor` at the requested
        ``cycles`` and ``R`` (``R = 0.1`` if not specified).
        """
        if cycles is None:
            return 1.0
        R_eff = 0.1 if R is None else float(R)
        return float(FatigueModel().knockdown_factor(mode, cycles, R_eff))

    def _apply_discrete_void_scf(self, base_knockdown: np.ndarray, mode: str) -> np.ndarray:
        kd = base_knockdown.copy()
        for void in self.mesh.porosity_field.discrete_voids:
            scf_dict = void.stress_concentration_factor()
            scf = scf_dict.get(mode, 1.0)
            dist = void.distance_field(self.mesh.nodes[:, 0],
                                        self.mesh.nodes[:, 1],
                                        self.mesh.nodes[:, 2])
            influence = np.exp(-np.maximum(dist, 0) / max(void.radii))
            kd *= (1.0 - influence * (1.0 - 1.0 / scf))
        return kd

    def _resolve_knockdown_model(
            self, model: Union[str, Callable[[float, str], float]], mode: str
    ) -> Tuple[Callable[[float, str], float], bool]:
        """Resolve ``model`` to a ``(callable, is_user_supplied)`` pair.

        Built-in string names dispatch to the corresponding ``_judd_wright``
        / ``_power_law`` / ``_linear`` bound method (and therefore go through
        the existing layup-scaled coefficient table). A user-supplied
        callable is validated on a ``Vp ∈ [0, 1]`` grid (so the contract
        ``model(Vp, mode) -> float in [0, 1]`` is enforced once at dispatch
        time, not silently propagated to downstream nodal arrays) and
        returned as-is — the caller owns the coefficients, so the layup
        scaling is bypassed (#62).
        """
        if isinstance(model, str):
            _MODEL_FUNCS = {'judd_wright': self._judd_wright,
                            'power_law': self._power_law,
                            'linear': self._linear}
            if model not in _MODEL_FUNCS:
                raise ValueError(
                    f"Unknown knockdown model {model!r}. "
                    f"Use one of {sorted(_MODEL_FUNCS)} or pass a callable."
                )
            return _MODEL_FUNCS[model], False
        # User-supplied callable.
        self._validate_user_kd_callable(model, mode)
        return model, True

    def apply_loading(self, mode: str = 'compression',
                      model: Union[str, Callable[[float, str], float]] = 'judd_wright',
                      *,
                      cycles: Optional[float] = None,
                      environment: Optional[Dict[str, float]] = None,
                      R: Optional[float] = None):
        """Compute per-node knockdown for a given loading mode and model.

        Populates ``self.nodal_knockdown`` (shape ``(n_nodes,)``, values in
        ``(0, 1]``) by evaluating the empirical model at each node's local
        ``Vp`` and folding in any discrete-void stress concentration factors.
        When ``environment`` and / or ``cycles`` are supplied, the hygrothermal
        and / or S-N fatigue knockdowns are composed multiplicatively into
        the per-node field (issue #59).

        Parameters
        ----------
        mode : {'compression', 'tension', 'shear', 'ilss', 'transverse_tension'}
            Loading mode that selects the pristine strength and the
            mode-specific empirical coefficient.
        model : str or callable
            Empirical knockdown form. A string in
            ``{'judd_wright', 'power_law', 'linear'}`` dispatches to the
            built-in model with its layup-scaled coefficient. Alternatively,
            a callable matching the contract
            ``model(Vp: float, mode: str) -> float ∈ [0, 1]`` plugs in a
            user-defined knockdown law. User callables own their own
            coefficients, so the internal layup scaling is bypassed; the
            discrete-void SCF post-step still applies so caller-defined
            knockdowns and explicit voids compose the same way as the
            built-in models.
        cycles : int or float, optional
            Number of load cycles ``N`` for an S-N fatigue knockdown
            (issue #59). When ``None`` (default) no fatigue effect is
            applied. When supplied, the :class:`FatigueModel` log-linear
            knockdown is composed multiplicatively. Requires ``N >= 1``.
        environment : dict, optional
            Mapping with optional keys ``'T'`` (service temperature, deg C)
            and ``'M'`` (moisture content, wt%) for the hygrothermal
            knockdown (issue #59). When ``None`` (default) no environment
            effect is applied. The hygrothermal factor comes from
            :meth:`MaterialProperties.environment_knockdown`.
        R : float, optional
            Stress ratio for the fatigue knockdown. Currently
            informational; default (``None``) falls back to ``R = 0.1``
            (tension-tension).

        Notes
        -----
        Sign convention: ``mode='compression'`` and ``mode='ilss'`` return a
        **positive magnitude** failure stress (and a knockdown in ``(0, 1]``),
        not a signed value. Tension and compression strengths are stored on
        :class:`MaterialProperties` as positive numbers (``sigma_1c``,
        ``sigma_1t``), and the downstream
        ``failure_stress_MPa = KD * sigma_0`` is reported with the same
        positive-magnitude sign. Distinguish modes by the ``mode`` field of
        the returned dict, not by the sign of the stress. This is the
        empirical-solver convention only — :class:`FieldResults` stores
        **signed** FE stresses and strains in Voigt order
        ``[11, 22, 33, 23, 13, 12]`` (engineering shear).
        """
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        # #115: vectorize the built-in knockdown evaluation. The scalar list
        # comprehension was ~60x slower than NumPy on the per-node Vp array
        # (4400-element typical mesh). User-supplied callables still get the
        # scalar path so the (Vp, mode) -> float contract from #62 holds.
        Vp_arr = self.mesh.porosity
        if isinstance(model, str):
            # Mode validation already done above; this picks built-ins or
            # raises a clear error before we touch the array.
            if model == 'judd_wright':
                kd = np.exp(-self.JUDD_WRIGHT_ALPHA[mode] * Vp_arr)
            elif model == 'power_law':
                kd = (1.0 - Vp_arr) ** self.POWER_LAW_N[mode]
            elif model == 'linear':
                kd = np.maximum(1.0 - self.LINEAR_BETA[mode] * Vp_arr, 0.0)
            else:
                raise ValueError(
                    f"Unknown knockdown model {model!r}. "
                    f"Use one of ['judd_wright', 'power_law', 'linear'] "
                    f"or pass a callable."
                )
        else:
            self._validate_user_kd_callable(model, mode)
            kd = np.array([model(Vp, mode) for Vp in Vp_arr])
        kd = self._apply_discrete_void_scf(kd, mode)
        env_kd = self._environment_knockdown_factor(mode, environment)
        fat_kd = self._fatigue_knockdown_factor(mode, cycles, R)
        if env_kd != 1.0:
            kd = kd * env_kd
        if fat_kd != 1.0:
            kd = kd * fat_kd
        self.nodal_knockdown = kd  # type: ignore[assignment]  # lazy-init attr starts None

    def get_failure_load(self, mode: str = 'compression',
                         model: Union[str, Callable[[float, str], float]]
                         = 'judd_wright',
                         *,
                         cycles: Optional[float] = None,
                         environment: Optional[Dict[str, float]] = None,
                         R: Optional[float] = None) -> 'FailureResult':
        """Compute failure load using specimen-average porosity.

        The knockdown is evaluated at the mean Vp (matching how the original
        correlations were calibrated), not at the local peak.  Per-node
        knockdown is still computed for visualization via apply_loading().
        Optional hygrothermal (``environment``) and S-N fatigue
        (``cycles`` / ``R``) knockdowns compose multiplicatively with the
        porosity knockdown (issue #59).

        Parameters
        ----------
        mode : str
            Loading mode; see :meth:`apply_loading`.
        model : str or callable
            Either a built-in name (``'judd_wright'``, ``'power_law'``,
            ``'linear'``) or a user-supplied callable with signature
            ``model(Vp: float, mode: str) -> float ∈ [0, 1]``. User
            callables bypass the layup-scaled coefficient table (the caller
            owns their model).
        cycles : int or float, optional
            Number of load cycles ``N`` for an S-N fatigue knockdown.
            When ``None`` (default) returns ``1.0`` for the fatigue
            factor (no fatigue effect). When supplied, the
            :class:`FatigueModel` log-linear knockdown composes
            multiplicatively with the porosity knockdown. The result
            ``details`` dict gains a ``'fatigue_knockdown'`` entry so the
            breakdown is auditable.
        environment : dict, optional
            Hygrothermal conditioning dict with optional keys ``'T'``
            (service temperature, deg C) and ``'M'`` (moisture content,
            wt%). When ``None`` (default) no hygrothermal knockdown is
            applied. Composed multiplicatively with porosity and fatigue;
            the result ``details`` dict gains an ``'environment_knockdown'``
            entry when active.
        R : float, optional
            Stress ratio for the fatigue knockdown. Currently informational
            (default ``None`` -> ``R = 0.1``).

        Returns
        -------
        FailureResult
            Unified failure summary (#44 item 1) with ``failure_stress``,
            ``knockdown``, ``model`` attributes plus a ``details`` dict
            carrying the legacy ``'critical_location'`` extra and the new
            ``'environment_knockdown'`` / ``'fatigue_knockdown'`` entries
            (when active). Back-compat dict-style access
            (``result['failure_stress']``, ``result['critical_location']``,
            etc.) is preserved via the :class:`FailureResult`
            ``__getitem__`` shim and will be removed in a future major
            version — prefer attribute access.
        """
        self.apply_loading(mode, model,
                           cycles=cycles, environment=environment, R=R)
        sigma_0 = self._get_pristine_strength(mode)

        # Use specimen-average Vp for knockdown (matches calibration basis)
        Vp_mean = self.mesh.porosity_field.Vp
        model_func, _is_user = self._resolve_knockdown_model(model, mode)
        porosity_kd = float(model_func(Vp_mean, mode))

        # Hygrothermal and fatigue knockdowns compose multiplicatively at
        # the same point as the porosity knockdown so the final
        # `failure_stress` carries all three effects, mirroring how the
        # layup scaling is folded into the empirical coefficients upstream.
        env_kd = self._environment_knockdown_factor(mode, environment)
        fat_kd = self._fatigue_knockdown_factor(mode, cycles, R)
        mean_kd = porosity_kd * env_kd * fat_kd

        # Record a JSON-friendly label even when the caller passes a
        # callable (lambdas/closures don't round-trip through json.dumps).
        model_label = model if isinstance(model, str) else getattr(
            model, '__name__', 'user_callable')

        details: Dict[str, Any] = {
            'critical_location': [0.0, 0.0, 0.0],
            'mode': mode,
        }
        # Surface the per-knockdown breakdown in ``details`` so callers
        # (e.g. the #65 tornado sensitivities) can audit the composition.
        if environment is not None:
            details['environment_knockdown'] = float(env_kd)
        if cycles is not None:
            details['fatigue_knockdown'] = float(fat_kd)

        return FailureResult(
            failure_stress=float(sigma_0 * mean_kd),
            knockdown=float(mean_kd),
            model=str(model_label),
            details=details,
        )

    def get_all_failure_loads(
            self,
            extra_models: Optional[Dict[str, Callable[[float, str], float]]] = None,
            *,
            cycles: Optional[float] = None,
            environment: Optional[Dict[str, float]] = None,
            R: Optional[float] = None,
    ) -> dict:
        """Compute failure loads for all modes against all built-in models.

        Parameters
        ----------
        extra_models : dict[str, callable], optional
            User-supplied knockdown callables to evaluate alongside the
            built-ins. Each entry's key is the label used in the result
            dict; its value must be a callable matching the contract
            ``model(Vp: float, mode: str) -> float ∈ [0, 1]``. The built-in
            three models are always included.
        cycles : int or float, optional
            Number of load cycles ``N`` for an S-N fatigue knockdown
            (issue #59). Threaded into every :meth:`get_failure_load` call.
            ``None`` -> no fatigue effect.
        environment : dict, optional
            Hygrothermal conditioning dict (see :meth:`apply_loading`).
            Threaded into every :meth:`get_failure_load` call. ``None`` ->
            no hygrothermal effect.
        R : float, optional
            Stress ratio for the fatigue knockdown (informational).
        """
        results: Dict[str, Dict[str, FailureResult]] = {}
        all_models: List[Tuple[str, Union[str, Callable[[float, str], float]]]] = [
            ('judd_wright', 'judd_wright'),
            ('power_law', 'power_law'),
            ('linear', 'linear'),
        ]
        if extra_models:
            for label, fn in extra_models.items():
                all_models.append((str(label), fn))
        for mode in ['compression', 'tension', 'shear', 'ilss',
                     'transverse_tension']:
            results[mode] = {}
            for label, model in all_models:
                results[mode][label] = self.get_failure_load(
                    mode, model,
                    cycles=cycles, environment=environment, R=R,
                )
        return results

    def local_sensitivities(self, mode: str = 'compression',
                            model: str = 'judd_wright',
                            Vp: Optional[float] = None) -> Dict[str, float]:
        """Closed-form local sensitivities of the empirical knockdown.

        Returns the analytic partials ``dKD/dVp`` and ``dKD/dcoef`` (the
        layup-scaled coefficient for the chosen model) at the supplied
        porosity ``Vp``.  The three knockdown laws are closed form, so the
        partials are exact, dimensionless, and effectively free to compute
        (no FD, no sampling).

        Partial-derivative table (with ``c`` denoting the layup-scaled
        coefficient — ``alpha`` for Judd-Wright, ``n`` for power-law,
        ``beta`` for linear):

        =============  ===========================  ===========================
        Model          ``dKD/dVp``                  ``dKD/dcoef``
        =============  ===========================  ===========================
        judd_wright    ``-alpha * KD``              ``-Vp * KD``
        power_law      ``-n * (1 - Vp)**(n-1)``     ``(1-Vp)**n * ln(1-Vp)``
        linear         ``-beta`` (or 0 if clipped)  ``-Vp``  (or 0 if clipped)
        =============  ===========================  ===========================

        Parameters
        ----------
        mode:
            Loading mode (same keys as :meth:`get_failure_load`).
        model:
            Empirical knockdown law: ``'judd_wright'``, ``'power_law'``,
            or ``'linear'``.
        Vp:
            Specimen-average porosity at which to evaluate.  Defaults to
            ``self.mesh.porosity_field.Vp`` — the same value
            :meth:`get_failure_load` uses.

        Returns
        -------
        dict
            ``{'KD': float, 'dKD_dVp': float, 'dKD_dcoef': float}``.
            ``dKD_dcoef`` is the partial with respect to the layup-scaled
            coefficient (alpha/n/beta) that the solver actually applied;
            it already reflects the layup scaling from
            :meth:`_layup_scale`.
        """
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        if Vp is None:
            Vp = self.mesh.porosity_field.Vp
        Vp = self._check_internal_Vp(Vp)
        if model == 'judd_wright':
            alpha = self.JUDD_WRIGHT_ALPHA[mode]
            kd = float(np.exp(-alpha * Vp))
            return {
                'KD': kd,
                'dKD_dVp': float(-alpha * kd),
                'dKD_dcoef': float(-Vp * kd),
            }
        if model == 'power_law':
            n = self.POWER_LAW_N[mode]
            one_minus = 1.0 - Vp
            kd = float(one_minus**n)
            # Guard the log when Vp = 1 (degenerate edge): KD is 0 there
            # and the d/dn partial collapses to 0 because KD * ln(1-Vp)
            # is 0 * (-inf) in the limit.  We pin it to 0.0 explicitly so
            # callers see a finite value.
            if one_minus <= 0.0:
                d_dcoef = 0.0
                d_dVp = 0.0
            else:
                d_dcoef = float(kd * np.log(one_minus))
                d_dVp = float(-n * one_minus**(n - 1.0))
            return {
                'KD': kd,
                'dKD_dVp': d_dVp,
                'dKD_dcoef': d_dcoef,
            }
        if model == 'linear':
            beta = self.LINEAR_BETA[mode]
            raw = 1.0 - beta * Vp
            kd = float(max(raw, 0.0))
            # The linear law is clipped at 0: once raw < 0, the
            # piecewise-constant 0 floor has zero gradient.
            if raw <= 0.0:
                d_dVp = 0.0
                d_dcoef = 0.0
            else:
                d_dVp = float(-beta)
                d_dcoef = float(-Vp)
            return {
                'KD': kd,
                'dKD_dVp': d_dVp,
                'dKD_dcoef': d_dcoef,
            }
        raise ValueError(
            f"Unknown knockdown model {model!r}. "
            f"Use one of ['judd_wright', 'linear', 'power_law']."
        )

    def sensitivity_fd(self, mode: str = 'compression',
                       model: str = 'judd_wright',
                       param: str = 'Vp',
                       h: float = 1e-4) -> float:
        """Central-difference fallback for the local knockdown sensitivity.

        Useful for paths where the gradient is not closed form (FE or
        Mori-Tanaka couplings).  For the bundled empirical models this
        matches :meth:`local_sensitivities` to ~1e-7 and is shipped as a
        cross-check / drop-in for non-analytic models.

        Parameters
        ----------
        mode:
            Loading mode (same keys as :meth:`get_failure_load`).
        model:
            Empirical knockdown law.
        param:
            Either ``'Vp'`` (porosity) or ``'coef'`` (the layup-scaled
            coefficient that the model actually applied — alpha/n/beta).
        h:
            Step size for the central difference.

        Returns
        -------
        float
            ``(KD(x+h) - KD(x-h)) / (2*h)`` evaluated at the same
            ``Vp_mean`` :meth:`get_failure_load` uses.
        """
        if mode not in self.PRISTINE_STRENGTH_KEY:
            raise ValueError(
                f"Unknown loading mode {mode!r}. "
                f"Use one of {sorted(self.PRISTINE_STRENGTH_KEY)}."
            )
        if model not in ('judd_wright', 'power_law', 'linear'):
            raise ValueError(
                f"Unknown knockdown model {model!r}. "
                f"Use one of ['judd_wright', 'linear', 'power_law']."
            )
        if param not in ('Vp', 'coef'):
            raise ValueError(
                f"param must be 'Vp' or 'coef', got {param!r}."
            )
        Vp0 = float(self.mesh.porosity_field.Vp)

        # Select the analytic functional form so we can perturb the
        # parameter without mutating solver state.
        if model == 'judd_wright':
            coef0 = float(self.JUDD_WRIGHT_ALPHA[mode])
            def f(Vp_val, coef_val):
                return float(np.exp(-coef_val * Vp_val))
        elif model == 'power_law':
            coef0 = float(self.POWER_LAW_N[mode])
            def f(Vp_val, coef_val):
                return float((1.0 - Vp_val)**coef_val)
        else:  # linear
            coef0 = float(self.LINEAR_BETA[mode])
            def f(Vp_val, coef_val):
                return float(max(1.0 - coef_val * Vp_val, 0.0))

        if param == 'Vp':
            return float((f(Vp0 + h, coef0) - f(Vp0 - h, coef0)) / (2.0 * h))
        # param == 'coef'
        return float((f(Vp0, coef0 + h) - f(Vp0, coef0 - h)) / (2.0 * h))


# ============================================================
# SECTION 6b: UNCERTAINTY PROPAGATION (MONTE CARLO / LHS)
# ============================================================

# Default percentiles reported by the UQ helpers (a 5%/50%/95% band, the
# spread an A-/B-basis workflow typically wants to see first).
_UQ_DEFAULT_PERCENTILES = (5.0, 50.0, 95.0)
_UQ_METHODS = ('monte_carlo', 'lhs')
# Distributions that consume a standard-normal unit draw vs. a U(0,1) draw.
_UQ_NORMAL_DISTS = ('lognormal', 'normal')
_UQ_UNIFORM_DISTS = ('uniform',)


def _normalize_uq_spec(material: 'MaterialProperties',
                       covs: Optional[Dict[str, float]],
                       spec: Optional[Dict[str, Tuple[str, float]]]
                       ) -> "OrderedDict":
    """Resolve the user-facing uncertainty description into a canonical
    ``OrderedDict{field: (dist, params)}``.

    ``covs`` is the convenience form: ``{field: cov}`` -> truncated-lognormal
    with that coefficient of variation. ``spec`` is the explicit form:
    ``{field: (dist, params)}``. They may both be given (``spec`` wins on a
    field collision). Fields with a non-positive CoV are dropped so a
    zero-CoV request is exactly the deterministic pipeline.
    """
    resolved: OrderedDict = OrderedDict()
    valid = set(MaterialProperties.PERTURBABLE_FIELDS)

    def _check_field(name: str) -> None:
        if name not in valid:
            raise ValueError(
                f"Unknown / non-perturbable material field {name!r}. "
                f"Use one of {sorted(valid)}."
            )

    for name, cov in (covs or {}).items():
        _check_field(name)
        cov = float(cov)
        if not np.isfinite(cov) or cov < 0.0:
            raise ValueError(
                f"CoV for {name!r} must be a finite non-negative number, "
                f"got {cov!r}."
            )
        if cov > 0.0:
            resolved[name] = ('lognormal', cov)

    for name, ds in (spec or {}).items():
        _check_field(name)
        if not (isinstance(ds, (tuple, list)) and len(ds) == 2):
            raise ValueError(
                f"spec[{name!r}] must be a (distribution, params) pair, "
                f"got {ds!r}."
            )
        dist, params = ds
        if dist not in _UQ_NORMAL_DISTS + _UQ_UNIFORM_DISTS:
            raise ValueError(
                f"spec[{name!r}] has unknown distribution {dist!r}. "
                f"Use one of {sorted(_UQ_NORMAL_DISTS + _UQ_UNIFORM_DISTS)}."
            )
        params = float(params)
        if not np.isfinite(params) or params < 0.0:
            raise ValueError(
                f"spec[{name!r}] params must be a finite non-negative "
                f"number, got {params!r}."
            )
        if params > 0.0:
            resolved[name] = (dist, params)
        else:
            resolved.pop(name, None)
    return resolved


def _draw_unit_samples(n_vars: int, n_samples: int, method: str,
                       rng: np.random.Generator) -> np.ndarray:
    """Return an ``(n_samples, n_vars)`` array of U(0, 1) variates.

    ``method='monte_carlo'`` uses ``rng.random``; ``method='lhs'`` uses
    ``scipy.stats.qmc.LatinHypercube`` seeded from the same ``rng`` so the
    whole helper is reproducible from a single seed.
    """
    if n_vars == 0:
        return np.empty((n_samples, 0))
    if method == 'monte_carlo':
        return rng.random((n_samples, n_vars))
    if method == 'lhs':
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_vars, seed=rng)
        return sampler.random(n=n_samples)
    raise ValueError(
        f"Unknown sampling method {method!r}. Use one of {sorted(_UQ_METHODS)}."
    )


def _unit_to_draw(u: np.ndarray, dist: str) -> np.ndarray:
    """Map U(0, 1) variates to the unit variate the target distribution
    expects: a standard normal for normal/lognormal, the U(0,1) untouched
    (clipped off the 0/1 endpoints) for uniform."""
    if dist in _UQ_NORMAL_DISTS:
        from scipy.stats import norm
        return norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    return u


def propagate_uncertainty(void_volume_fraction: float,
                          material: Union[str, 'MaterialProperties'] = 'T800_epoxy',
                          mode: str = 'compression',
                          model: str = 'judd_wright',
                          *,
                          covs: Optional[Dict[str, float]] = None,
                          spec: Optional[Dict[str, Tuple[str, float]]] = None,
                          vp_cov: float = 0.0,
                          n_samples: int = 1000,
                          method: str = 'monte_carlo',
                          seed: Optional[int] = None,
                          percentiles: Tuple[float, ...] = _UQ_DEFAULT_PERCENTILES,
                          ply_angles: Optional[Union[List[float], str]] = 'QI',
                          config: Optional[Dict] = None) -> Dict:
    """Propagate input uncertainty through ``EmpiricalSolver.get_failure_load``.

    Perturbs uncertain ``MaterialProperties`` fields (and, optionally, the
    specimen-average porosity ``Vp``) and reports summary statistics of the
    knockdown and failure stress. The base deterministic pipeline is
    untouched; this is a strictly additive wrapper.

    Parameters
    ----------
    void_volume_fraction : float
        Nominal mean porosity fraction in [0, 1].
    material : str or MaterialProperties
        A ``MATERIALS`` preset name or an explicit dataclass instance.
    mode, model : str
        Forwarded to :meth:`EmpiricalSolver.get_failure_load`.
    covs : dict, optional
        Convenience uncertainty spec ``{field: cov}`` -> truncated-lognormal
        with that coefficient of variation (std/mean).
    spec : dict, optional
        Explicit uncertainty spec ``{field: (dist, params)}`` where ``dist``
        is ``'lognormal'`` / ``'normal'`` (params = CoV) or ``'uniform'``
        (params = fractional half-width). Wins over ``covs`` on a collision.
    vp_cov : float
        CoV of the mean porosity itself (truncated-lognormal, clipped to
        [0, 1]). 0.0 (default) holds Vp fixed at ``void_volume_fraction``.
    n_samples : int
        Number of draws.
    method : {'monte_carlo', 'lhs'}
        ``'monte_carlo'`` -> ``numpy.random.default_rng``;
        ``'lhs'`` -> ``scipy.stats.qmc.LatinHypercube`` (both seeded from
        ``seed`` so results are reproducible).
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Echoed into the result. With a
        fixed seed the summary is bit-for-bit reproducible.
    percentiles : tuple of float
        Percentiles to report (default 5/50/95).
    ply_angles : list of float, optional
        Forwarded to ``EmpiricalSolver`` (layup scaling).
    config : dict, optional
        Forwarded to ``PorosityField`` (distribution / void_shape / ...).

    Returns
    -------
    dict
        ``{'failure_stress': {'mean','std','min','max','percentiles': {...}},
        'knockdown': {... same ...}, 'nominal': {...},
        'samples': {'failure_stress': np.ndarray, 'knockdown': np.ndarray},
        'seed', 'n_samples', 'method', 'mode', 'model', 'spec', 'vp_cov'}``.
    """
    if isinstance(material, str):
        if material not in MATERIALS:
            raise ValueError(
                f"Unknown material {material!r}. "
                f"Available presets: {sorted(MATERIALS)}."
            )
        material_name = material
        mat = MATERIALS[material]
    else:
        material_name = getattr(material, '__class__', type(material)).__name__
        mat = material

    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(
            f"n_samples must be a positive integer, got {n_samples!r}."
        )
    if method not in _UQ_METHODS:
        raise ValueError(
            f"Unknown sampling method {method!r}. "
            f"Use one of {sorted(_UQ_METHODS)}."
        )
    vp_cov = float(vp_cov)
    if not np.isfinite(vp_cov) or vp_cov < 0.0:
        raise ValueError(
            f"vp_cov must be a finite non-negative number, got {vp_cov!r}."
        )
    pcts = tuple(float(p) for p in percentiles)
    if any((not np.isfinite(p)) or p < 0.0 or p > 100.0 for p in pcts):
        raise ValueError(
            f"percentiles must lie in [0, 100], got {percentiles!r}."
        )

    resolved = _normalize_uq_spec(mat, covs, spec)
    field_names = list(resolved.keys())
    # The porosity variable, if active, is the last sampling dimension.
    sample_vp = vp_cov > 0.0
    n_vars = len(field_names) + (1 if sample_vp else 0)

    config = config or {}

    import contextlib
    import io

    def _build_solver(material_obj: 'MaterialProperties',
                      vp_value: float) -> 'EmpiricalSolver':
        # CompositeMesh prints a banner on construction; the sampling loop
        # builds one mesh per draw, so silence it (additive: we do not touch
        # CompositeMesh itself).
        with contextlib.redirect_stdout(io.StringIO()):
            pf = PorosityField(material_obj, vp_value, **config)
            msh = CompositeMesh(pf, material_obj, nx=4, ny=3, nz=3,
                                ply_angles=ply_angles)
            return EmpiricalSolver(msh, material_obj, ply_angles=ply_angles)

    # Deterministic nominal (no perturbation): the mean must land near this.
    nominal = _build_solver(mat, float(void_volume_fraction)).get_failure_load(
        mode, model)

    rng = np.random.default_rng(seed)
    unit = _draw_unit_samples(n_vars, int(n_samples), method, rng)

    fs_samples = np.empty(int(n_samples), dtype=float)
    kd_samples = np.empty(int(n_samples), dtype=float)

    # Pre-compute per-field column index and the unit-variate mapping.
    col_for_field = {name: i for i, name in enumerate(field_names)}
    vp_col = len(field_names) if sample_vp else None
    nominal_vp = float(void_volume_fraction)
    if sample_vp:
        sigma_ln_vp = np.sqrt(np.log1p(vp_cov * vp_cov))

    for s in range(int(n_samples)):
        draws = {}
        for name in field_names:
            dist, _ = resolved[name]
            u = unit[s, col_for_field[name]]
            draws[name] = float(_unit_to_draw(np.array([u]), dist)[0])
        sampled_mat = mat.perturb(draws, resolved) if field_names else mat

        if sample_vp:
            z = float(_unit_to_draw(np.array([unit[s, vp_col]]),
                                    'lognormal')[0])
            vp_value = nominal_vp * np.exp(sigma_ln_vp * z)
            vp_value = float(np.clip(vp_value, 0.0, 1.0))
        else:
            vp_value = nominal_vp

        res = _build_solver(sampled_mat, vp_value).get_failure_load(mode, model)
        fs_samples[s] = res['failure_stress']
        kd_samples[s] = res['knockdown']

    def _summary(arr: np.ndarray) -> Dict:
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'percentiles': {
                f'p{p:g}': float(np.percentile(arr, p)) for p in pcts
            },
        }

    return {
        'failure_stress': _summary(fs_samples),
        'knockdown': _summary(kd_samples),
        'nominal': {
            'failure_stress': float(nominal['failure_stress']),
            'knockdown': float(nominal['knockdown']),
        },
        'samples': {
            'failure_stress': fs_samples,
            'knockdown': kd_samples,
        },
        'seed': seed,
        'n_samples': int(n_samples),
        'method': method,
        'mode': mode,
        'model': model,
        'material': material_name,
        'void_volume_fraction': float(void_volume_fraction),
        'vp_cov': vp_cov,
        'percentiles': list(pcts),
        'spec': {k: list(v) for k, v in resolved.items()},
    }


# ============================================================
# SECTION 7: VISUALIZATION
# ============================================================

class FEVisualizer:
    """Publication-quality plotting for porosity analysis."""

    @staticmethod
    def plot_porosity_field(porosity_field: PorosityField, save_path: str = None):
        """Single panel: through-thickness porosity profile."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        z, Vp = porosity_field.effective_porosity_profile(nz=200)
        ax.plot(Vp * 100, z, 'b-', linewidth=2)
        ax.set_xlabel(LABEL_POROSITY_PCT)
        ax.set_ylabel(LABEL_Z_MM)
        ax.set_title('Through-Thickness Porosity Profile')
        ax.set_xlim(left=0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_mesh_3d(mesh: CompositeMesh, save_path: str = None):
        """3D hex mesh wireframe with void elements highlighted."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot top and bottom surface grids
        nx, ny = mesh.nx, mesh.ny
        n_per_layer = (nx + 1) * (ny + 1)

        for layer_idx in [0, mesh.nz]:
            start = layer_idx * n_per_layer
            end = start + n_per_layer
            layer_nodes = mesh.nodes[start:end]
            X = layer_nodes[:, 0].reshape(ny + 1, nx + 1)
            Y = layer_nodes[:, 1].reshape(ny + 1, nx + 1)
            Z = layer_nodes[:, 2].reshape(ny + 1, nx + 1)
            ax.plot_wireframe(X, Y, Z, alpha=0.3, color='gray', linewidth=0.5)

        # Highlight void elements as red wireframe hex boxes
        hex_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        if len(mesh.void_elements) > 0:
            for eidx in mesh.void_elements[:50]:  # limit for performance
                corners = mesh.nodes[mesh.elements[eidx]]  # (8, 3)
                for i1, i2 in hex_edges:
                    ax.plot3D(
                        *zip(corners[i1], corners[i2]),
                        color='red', linewidth=1.5, alpha=0.8, zorder=6,
                    )

        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Y_MM)
        ax.set_zlabel(LABEL_Z_MM)
        ax.set_title('3D Mesh with Porosity')

        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_mesh_detail(mesh: CompositeMesh, save_path: str = None):
        """Cross-section with porosity contour + single hex element."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: cross-section at mid-y
        ny_mid = mesh.ny // 2
        nx1 = mesh.nx + 1
        ny1 = mesh.ny + 1
        indices = []
        for k in range(mesh.nz + 1):
            for i in range(mesh.nx + 1):
                idx = k * ny1 * nx1 + ny_mid * nx1 + i
                indices.append(idx)
        indices = np.array(indices)  # type: ignore[assignment]  # list rebound to ndarray
        X = mesh.nodes[indices, 0].reshape(mesh.nz + 1, mesh.nx + 1)
        Z = mesh.nodes[indices, 2].reshape(mesh.nz + 1, mesh.nx + 1)
        P = mesh.porosity[indices].reshape(mesh.nz + 1, mesh.nx + 1)

        im = axes[0].contourf(X, Z, P * 100, levels=20, cmap='YlOrRd')
        plt.colorbar(im, ax=axes[0], label=LABEL_POROSITY_PCT)
        axes[0].set_xlabel(LABEL_X_MM)
        axes[0].set_ylabel(LABEL_Z_MM)
        axes[0].set_title('Cross-Section Porosity')
        axes[0].set_aspect('equal')

        # Right: single hex element diagram
        ax = axes[1]
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            pts = corners[list(e)]
            ax.plot(pts[:, 0] + pts[:, 1]*0.3, pts[:, 2] + pts[:, 1]*0.3,
                   'b-', linewidth=1.5)
        for idx, c in enumerate(corners):
            ax.plot(c[0] + c[1]*0.3, c[2] + c[1]*0.3, 'ko', markersize=6)
            ax.annotate(str(idx), (c[0] + c[1]*0.3 + 0.05, c[2] + c[1]*0.3 + 0.05),
                       fontweight='bold')
        ax.set_title('8-Node Hexahedral Element')
        # Use the same (mm) units as the sibling cross-section panel so the
        # hex-element diagram is not ambiguous within the same figure (#53).
        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Z_MM)
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_damage_contour(mesh: CompositeMesh, solver, save_path: str = None):
        """2D stiffness reduction map at midplane."""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Get midplane slice
        nz_mid = mesh.nz // 2
        nx1 = mesh.nx + 1
        ny1 = mesh.ny + 1
        start = nz_mid * ny1 * nx1
        end = start + ny1 * nx1
        X = mesh.nodes[start:end, 0].reshape(ny1, nx1)
        Y = mesh.nodes[start:end, 1].reshape(ny1, nx1)

        if solver.nodal_knockdown is not None:
            kd = solver.nodal_knockdown[start:end].reshape(ny1, nx1)
        else:
            kd = mesh.stiffness_reduction[start:end].reshape(ny1, nx1)

        im = ax.contourf(X, Y, kd, levels=20, cmap='cividis')
        # GUI version uses "Stiffness Retention (%)"; static PNG was using
        # "Stiffness Retention (fraction)" and a 0..1 scale. The two paths
        # plot the same physical quantity (``stiffness_reduction`` is a
        # 0..1 retention fraction), so report it consistently as a
        # percentage and the GUI/PNG units cannot drift again (#53).
        plt.colorbar(im, ax=ax, label=LABEL_STIFFNESS_RETENTION_FRAC)
        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Y_MM)
        ax.set_title('Stiffness Reduction at Midplane')
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_void_scf(void_geometry: VoidGeometry, save_path: str = None):
        """Stress concentration field around a single void."""
        fig, ax = plt.subplots(figsize=(8, 8))

        r_max = 3 * max(void_geometry.radii)
        x = np.linspace(-r_max, r_max, 200)
        y = np.linspace(-r_max, r_max, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        dist = void_geometry.distance_field(X.ravel(), Y.ravel(), Z.ravel())
        dist = dist.reshape(X.shape)

        scf = void_geometry.stress_concentration_factor()
        scf_max = scf['compression']
        field = np.where(dist < 0, 0, 1.0 + (scf_max - 1) * np.exp(-dist / max(void_geometry.radii)))

        im = ax.contourf(X, Y, field, levels=30, cmap='magma')
        plt.colorbar(im, ax=ax, label=LABEL_SCF)
        ax.set_xlabel(LABEL_X_MM)
        ax.set_ylabel(LABEL_Y_MM)
        ax.set_title(
            f'SCF Field (aspect ratio={void_geometry.aspect_ratio:.1f})'
        )
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_knockdown_curves(results_by_porosity: dict, save_path: str = None):
        """Strength vs porosity % for all loading modes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        modes = ['compression', 'tension', 'shear', 'ilss']
        colors = {'judd_wright': 'blue', 'power_law': 'red', 'linear': 'green'}

        for idx, mode in enumerate(modes):
            ax = axes[idx]
            Vp_vals = sorted([float(k.replace('pct', '')) for k in results_by_porosity.keys()])

            for config_name in list(list(results_by_porosity.values())[0].keys()):
                # Empirical models
                for model in ['judd_wright', 'power_law', 'linear']:
                    kd_vals = []
                    for Vp_label in sorted(results_by_porosity.keys()):
                        r = results_by_porosity[Vp_label][config_name]['empirical']
                        kd_vals.append(r[mode][model]['knockdown'])
                    ax.plot(Vp_vals, kd_vals, color=colors[model],
                           linestyle='-' if 'uniform' in config_name else '--',
                           alpha=0.7, linewidth=1.5)

            ax.set_xlabel(LABEL_POROSITY_PCT)
            ax.set_ylabel(LABEL_KNOCKDOWN)
            ax.set_title(mode.upper())
            ax.set_ylim(0, 1.1)

        plt.suptitle('Porosity Knockdown Curves')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig

    @staticmethod
    def plot_model_comparison(results: dict, save_path: str = None):
        """Empirical model comparison bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        configs = list(results.keys())
        x = np.arange(len(configs))
        width = 0.2

        # Left: compression knockdown
        for i, model in enumerate(['judd_wright', 'power_law', 'linear']):
            vals = [results[c]['empirical']['compression'][model]['knockdown'] for c in configs]
            axes[0].bar(x + i * width, vals, width, label=model.replace('_', ' ').title())
        axes[0].set_xticks(x + width)
        # Tick labels intentionally smaller than rcParams.xtick.labelsize
        # because the config names are long and wrap to two lines.
        axes[0].set_xticklabels(
            [c.replace('_', '\n') for c in configs], fontsize=8,
        )
        axes[0].set_ylabel(LABEL_KNOCKDOWN)
        axes[0].set_title('Compression')
        axes[0].legend()
        axes[0].grid(True, axis='y')

        # Right: ILSS knockdown
        for i, model in enumerate(['judd_wright', 'power_law', 'linear']):
            vals = [results[c]['empirical']['ilss'][model]['knockdown'] for c in configs]
            axes[1].bar(x + i * width, vals, width, label=model.replace('_', ' ').title())
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(
            [c.replace('_', '\n') for c in configs], fontsize=8,
        )
        axes[1].set_ylabel(LABEL_KNOCKDOWN)
        axes[1].set_title('ILSS')
        axes[1].legend()
        axes[1].grid(True, axis='y')

        plt.suptitle('Model Comparison')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info("Saved: %s", save_path)
        return fig


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


# Bounded LRU cache for _mt_effective_stiffness results (#42). The FE stress
# recovery loop calls this once per (element, Gauss point) with a Vp that
# only varies element-to-element, so within an element the same key recurs
# 8x. Across an FE run with a few thousand elements, the number of unique
# (Vp, shape, nu_m, C_m-fingerprint) tuples is small enough that an LRU of
# a few thousand entries gives a high hit rate at low memory cost.
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


# ============================================================
# SECTION 7e: GLOBAL ASSEMBLER
# ============================================================

class GlobalAssembler:
    """Assembles global stiffness matrix from Hex8Element contributions.

    Uses COO format for assembly, converts to CSC for solving.

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh.
    material : MaterialProperties
        Material properties.
    porosity_field : PorosityField
        Porosity field for degradation.
    """

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 porosity_field: PorosityField) -> None:
        self.mesh = mesh
        self.material = material
        self.porosity_field = porosity_field
        self._C_base = material.get_stiffness_matrix()
        self._C_m = material.get_isotropic_matrix_stiffness()
        self._nu_m = material.matrix_poisson
        self._void_shape = porosity_field.void_shape_radii
        self._ke_cache: Dict[tuple, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def create_element(self, elem_idx: int) -> Hex8Element:
        """Create a Hex8Element for the given element index.

        Elements whose centroid falls inside a discrete void get is_void=True,
        which sets their stiffness to near-zero (~1 MPa), creating an explicit
        void inclusion in the mesh.
        """
        node_ids = self.mesh.elements[elem_idx]
        node_coords = self.mesh.nodes[node_ids]
        ply_angle = float(self.mesh.ply_angles[elem_idx])
        node_porosities = self.mesh.porosity[node_ids]
        is_void = elem_idx in self.mesh.void_element_set
        return Hex8Element(
            node_coords=node_coords,
            C_base=self._C_base,
            ply_angle_deg=ply_angle,
            node_porosities=node_porosities,
            void_shape_radii=self._void_shape,
            nu_m=self._nu_m,
            C_m=self._C_m,
            is_void=is_void,
            material=self.material,
        )

    def _element_cache_key(self, elem_idx: int) -> Optional[tuple]:
        """Return a cache key if this element can reuse a cached Ke.

        Elements can share stiffness matrices when they have:
        - Same ply angle
        - Same uniform porosity at all 8 nodes
        - Same element geometry (all 8 node positions relative to centroid)
        - Same void status
        - Same base stiffness matrix C_base

        The geometry fingerprint uses all 8 node coordinates relative to the
        element centroid (rounded to 8 decimal places) so that skewed, rotated,
        or otherwise non-rectilinear elements are never incorrectly coalesced
        with axis-aligned elements that share the same bounding-box extents.
        C_base is included so elements with identical shape but different
        material properties do not share a cached stiffness matrix.
        """
        node_ids = self.mesh.elements[elem_idx]
        node_porosities = self.mesh.porosity[node_ids]
        # Only cache if all nodes have the same porosity
        if not np.allclose(node_porosities, node_porosities[0], atol=1e-12):
            return None
        is_void = elem_idx in self.mesh.void_element_set
        ply_angle = float(self.mesh.ply_angles[elem_idx])
        porosity_val = round(float(node_porosities[0]), 10)
        # Encode the full element shape: 8 node positions relative to the
        # centroid, rounded to 8 decimal places.  This correctly distinguishes
        # skewed/non-rectilinear elements from axis-aligned ones that happen to
        # share the same (dx, dy, dz) bounding-box extents.
        coords = self.mesh.nodes[node_ids]
        centroid = coords.mean(axis=0)
        rel_coords = np.round(coords - centroid, 8)
        geom_key = tuple(rel_coords.ravel())
        # Include a hash of C_base so elements with the same geometry but
        # different material stiffness do not share a cached matrix.
        c_key = hash(self._C_base.tobytes())
        return (ply_angle, porosity_val, is_void, geom_key, c_key)

    def _cache_uniform_elements(self) -> None:
        """Pre-compute stiffness matrices for elements that share properties.

        For uniform porosity distributions on structured meshes, many elements
        differ only in ply angle. This method identifies unique element types
        and caches their stiffness matrices.
        """
        self._ke_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

        for e in range(self.mesh.n_elements):
            key = self._element_cache_key(e)
            if key is not None and key not in self._ke_cache:
                elem = self.create_element(e)
                Ke = elem.stiffness_matrix()
                # Symmetrize: Ke = B^T C B * |J| * w is mathematically
                # symmetric, but the void-overflow finite-mask in
                # Hex8Element.stiffness_matrix can break this for elements
                # crossing the void modulus boundary, and FP accumulation
                # across 8 Gauss points adds further drift. Iterative
                # solvers (CG/MINRES) warn or fail on asymmetric K, so we
                # enforce K = K^T at the source (issue #57).
                self._ke_cache[key] = 0.5 * (Ke + Ke.T)

    def element_dof_indices(self, elem_idx: int) -> np.ndarray:
        """Global DOF indices (24,) for an element's 8 nodes."""
        node_ids = self.mesh.elements[elem_idx]
        dofs = np.empty(24, dtype=np.intp)
        for i, nid in enumerate(node_ids):
            base = 3 * nid
            dofs[3 * i] = base
            dofs[3 * i + 1] = base + 1
            dofs[3 * i + 2] = base + 2
        return dofs

    def assemble_stiffness(self, verbose: bool = False) -> scipy.sparse.csc_matrix:
        """Assemble global stiffness matrix K in CSC format.

        Uses COO pre-allocation: n_elem * 576 entries.
        Elements with identical properties (same ply angle, porosity, geometry)
        reuse cached stiffness matrices for faster assembly.
        """
        # Pre-compute cache for uniform elements
        self._cache_uniform_elements()

        n_elem = self.mesh.n_elements
        n_dof = self.mesh.n_dof
        entries_per_elem = 24 * 24  # 576

        total_entries = n_elem * entries_per_elem
        coo_rows = np.empty(total_entries, dtype=np.intp)
        coo_cols = np.empty(total_entries, dtype=np.intp)
        coo_vals = np.empty(total_entries, dtype=np.float64)

        local_ii, local_jj = np.meshgrid(np.arange(24), np.arange(24), indexing='ij')
        local_ii = local_ii.ravel()
        local_jj = local_jj.ravel()

        for e in range(n_elem):
            if verbose and e % 500 == 0:
                logger.info(
                    "  Assembling element %d/%d (%.1f%%)",
                    e, n_elem, 100.0 * e / n_elem,
                )

            # Try cache first
            key = self._element_cache_key(e)
            if key is not None and key in self._ke_cache:
                Ke = self._ke_cache[key]
                self._cache_hits += 1
            else:
                elem = self.create_element(e)
                Ke = elem.stiffness_matrix()
                # Match the symmetrization applied in
                # _cache_uniform_elements so cache-miss and cache-hit
                # paths yield identical entries (issue #57).
                Ke = 0.5 * (Ke + Ke.T)
                self._cache_misses += 1

            dofs = self.element_dof_indices(e)

            offset = e * entries_per_elem
            coo_rows[offset:offset + entries_per_elem] = dofs[local_ii]
            coo_cols[offset:offset + entries_per_elem] = dofs[local_jj]
            coo_vals[offset:offset + entries_per_elem] = Ke.ravel()

        if verbose:
            n_void = len(self.mesh.void_elements)
            logger.info(
                "  Assembling element %d/%d (100.0%%) -- done.", n_elem, n_elem)
            logger.info(
                "  Void inclusion elements: %d (E ~ %s MPa)",
                n_void, Hex8Element.VOID_MODULUS,
            )
            logger.info(
                "  Ke cache: %d unique, %d hits, %d misses",
                len(self._ke_cache), self._cache_hits, self._cache_misses,
            )
            logger.info(
                "  Building sparse matrix: %d DOFs, %d COO entries",
                n_dof, total_entries,
            )

        K_coo = scipy.sparse.coo_matrix(
            (coo_vals, (coo_rows, coo_cols)),
            shape=(n_dof, n_dof),
        )
        K_csc = K_coo.tocsc()

        if verbose:
            logger.info(
                "  CSC matrix: %d stored entries (%.1f per DOF)",
                K_csc.nnz, K_csc.nnz / n_dof,
            )

        return K_csc


# ============================================================
# SECTION 7f: BOUNDARY HANDLER
# ============================================================

class BoundaryHandler:
    """Handles boundary conditions for the porosity FE model.

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh.
    """

    def __init__(self, mesh: CompositeMesh) -> None:
        self.mesh = mesh

    def nodes_on_face(self, face: str) -> np.ndarray:
        """Return node indices on the specified face."""
        return self.mesh.nodes_on_face(face)

    def compression_bcs(self, applied_strain: float = -0.01
                        ) -> Tuple[Dict[int, float], np.ndarray]:
        """Standard uniaxial compression boundary conditions.

        - x_min: ux = 0
        - x_max: ux = applied_strain * Lx
        - y_min: uy = 0
        - one corner fixed in z

        Parameters
        ----------
        applied_strain : float
            Applied nominal strain (negative for compression).

        Returns
        -------
        constrained_dofs : dict
            {global_dof: prescribed_value}
        F : np.ndarray
            Shape (n_dof,) force vector (zeros for displacement-controlled).
        """
        n_dof = self.mesh.n_dof
        Lx = self.mesh.L_x
        prescribed_disp = applied_strain * Lx

        constrained: Dict[int, float] = {}

        # Fix ux on x_min
        for nid in self.mesh.nodes_on_face('x_min'):
            constrained[3 * int(nid)] = 0.0

        # Prescribe ux on x_max
        for nid in self.mesh.nodes_on_face('x_max'):
            constrained[3 * int(nid)] = prescribed_disp

        # Fix uy on y_min (symmetry)
        for nid in self.mesh.nodes_on_face('y_min'):
            constrained[3 * int(nid) + 1] = 0.0

        # Fix uz on one corner node (rigid body)
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        ymin_nodes = self.mesh.nodes_on_face('y_min')
        zmin_nodes = self.mesh.nodes_on_face('z_min')
        corner = np.intersect1d(np.intersect1d(xmin_nodes, ymin_nodes), zmin_nodes)
        if corner.size > 0:
            constrained[3 * int(corner[0]) + 2] = 0.0
        else:
            constrained[3 * int(xmin_nodes[0]) + 2] = 0.0

        F = np.zeros(n_dof, dtype=np.float64)
        return constrained, F

    def tension_bcs(self, applied_strain: float = 0.01
                    ) -> Tuple[Dict[int, float], np.ndarray]:
        """Uniaxial tension boundary conditions (same structure, positive strain)."""
        return self.compression_bcs(applied_strain=applied_strain)

    def shear_bcs(self, applied_strain: float = 0.01
                  ) -> Tuple[Dict[int, float], np.ndarray]:
        """Pure shear boundary conditions for engineering shear strain gamma_12.

        Prescribes the deformation field consistent with pure shear on ALL four
        side faces (±x and ±y) so that no spurious bending or traction-free
        condition biases the computed shear modulus G12.

        For engineering shear strain gamma = applied_strain, the displacement
        field is::

            u(x, y) = (gamma / 2) * y
            v(x, y) = (gamma / 2) * x
            w       = 0

        This gives ε_xy = gamma/2 (Voigt ε_6 = gamma), with all normal strains
        and out-of-plane shear strains exactly zero.

        BCs applied
        -----------
        - x_min (x = 0): ux = 0,              uy = (gamma/2) * y_node
        - x_max (x = Lx): ux = (gamma/2)*y_node, uy = (gamma/2)*Lx
        - y_min (y = 0): ux = 0,              uy = (gamma/2)*x_node
        - y_max (y = Ly): ux = (gamma/2)*Ly,  uy = (gamma/2)*x_node
        - uz = 0 pinned at the (x_min, y_min, z_min) corner to remove rigid-body
          motion in z.
        """
        n_dof = self.mesh.n_dof
        gamma = applied_strain
        nodes = self.mesh.nodes  # shape (n_nodes, 3)

        constrained: Dict[int, float] = {}

        # ±x faces: u = gamma/2 * y_node,  v = gamma/2 * x_node
        for face in ('x_min', 'x_max'):
            for nid in self.mesh.nodes_on_face(face):
                nid = int(nid)
                x_n = float(nodes[nid, 0])
                y_n = float(nodes[nid, 1])
                constrained[3 * nid]     = (gamma / 2.0) * y_n   # ux
                constrained[3 * nid + 1] = (gamma / 2.0) * x_n   # uy

        # ±y faces: u = gamma/2 * y_node,  v = gamma/2 * x_node
        for face in ('y_min', 'y_max'):
            for nid in self.mesh.nodes_on_face(face):
                nid = int(nid)
                x_n = float(nodes[nid, 0])
                y_n = float(nodes[nid, 1])
                constrained[3 * nid]     = (gamma / 2.0) * y_n   # ux
                constrained[3 * nid + 1] = (gamma / 2.0) * x_n   # uy

        # Fix uz at one corner to prevent rigid-body translation in z
        xmin_nodes = self.mesh.nodes_on_face('x_min')
        ymin_nodes = self.mesh.nodes_on_face('y_min')
        zmin_nodes = self.mesh.nodes_on_face('z_min')
        corner = np.intersect1d(np.intersect1d(xmin_nodes, ymin_nodes), zmin_nodes)
        if corner.size > 0:
            constrained[3 * int(corner[0]) + 2] = 0.0
        else:
            constrained[3 * int(xmin_nodes[0]) + 2] = 0.0

        F = np.zeros(n_dof, dtype=np.float64)
        return constrained, F

    def ilss_bcs(self, applied_load: float = -10.0
                 ) -> Tuple[Dict[int, float], np.ndarray]:
        """ILSS (interlaminar short-beam shear) boundary conditions, ASTM D2344.

        Three-point short-beam-shear setup:

        - Two simple supports at the bottom face (``z_min``), one along the
          ``x_min`` edge and one along the ``x_max`` edge. All three
          translational DOFs are pinned at the support nodes so the beam
          is fully simply-supported in the FE sense.
        - A downward (``-z``) midspan load applied as a nodal force on
          the top face (``z_max``) at ``x = L_x / 2``. The total load is
          ``applied_load`` (typically negative for "downward"); it is
          distributed equally across the midspan-top nodes.

        Unlike the compression/tension/shear BCs which are *displacement*
        controlled, ILSS is **force controlled** — the returned ``F``
        vector carries the load directly rather than being routed through
        ``apply_penalty``. The penalty path is still used for the support
        DOF constraints.

        Parameters
        ----------
        applied_load : float
            Total midspan load in the ``z`` direction (negative = downward).

        Returns
        -------
        constrained_dofs : dict
            ``{global_dof: prescribed_value}`` — all-zero values at the
            two support edges (left/right of the bottom face).
        F : np.ndarray
            Shape ``(n_dof,)`` force vector with the midspan load applied
            to the top-face nodes near ``x = Lx/2``.

        Notes
        -----
        This implementation assumes the **three-point bend** geometry of
        ASTM D2344. The standard four-point bend variant (ASTM D7264)
        requires a different BC method — either a new ``ilss_4pt_bcs``
        with two upper load rollers, or the empirical-only path. The
        Tsai-Wu / strength-recovery code already handles the multi-axial
        stress state recovered from this solve.
        """
        n_dof = self.mesh.n_dof
        Lx = self.mesh.L_x
        Lz = self.mesh.L_z

        zmin = self.mesh.nodes_on_face('z_min')
        xmin = self.mesh.nodes_on_face('x_min')
        xmax = self.mesh.nodes_on_face('x_max')

        support_left = np.intersect1d(zmin, xmin)
        support_right = np.intersect1d(zmin, xmax)

        if support_left.size == 0 or support_right.size == 0:
            raise RuntimeError(
                "ilss_bcs: failed to locate bottom-face support edges "
                "(intersection of z_min with x_min / x_max is empty). "
                "Check mesh generation."
            )

        constrained: Dict[int, float] = {}
        for nid in np.concatenate([support_left, support_right]):
            nid = int(nid)
            constrained[3 * nid]     = 0.0  # ux
            constrained[3 * nid + 1] = 0.0  # uy
            constrained[3 * nid + 2] = 0.0  # uz

        F = np.zeros(n_dof, dtype=np.float64)
        # Tolerance must be wide enough to bracket *some* x-column when the
        # exact midspan does not fall on a mesh node. Use half the x-edge
        # plus a small fp epsilon so odd nx values still resolve a column.
        dx = Lx / max(self.mesh.nx, 1)
        dz = Lz / max(self.mesh.nz, 1)
        tol = 0.5 * np.hypot(dx, dz) + 1e-9
        midspan_top = self.mesh.find_nodes_near(x=Lx / 2.0, z=Lz, tol=tol)
        if midspan_top.size == 0:
            raise RuntimeError(
                "ilss_bcs: no top-face nodes found near midspan "
                f"(x = {Lx / 2.0}, z = {Lz}). Refine the mesh."
            )
        F[3 * midspan_top + 2] = applied_load / float(midspan_top.size)
        return constrained, F

    @staticmethod
    def apply_penalty(K: scipy.sparse.csc_matrix, F: np.ndarray,
                      constrained_dofs: Dict[int, float],
                      penalty_factor: float = 1e6
                      ) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
        """Apply penalty method for prescribed displacements.

        For each constrained DOF ``i`` with value ``v``,
        ``K[i, i] += alpha`` and ``F[i] = alpha * v``, where
        ``alpha = penalty_factor * max(diag(K))``.

        Parameters
        ----------
        K : scipy.sparse.csc_matrix
            Global stiffness matrix.
        F : np.ndarray
            Global force vector.
        constrained_dofs : dict
            {dof_index: prescribed_value}.
        penalty_factor : float
            Multiplier for max diagonal entry. Defaults to ``1e6`` (six
            decades of BC enforcement), lowered from the historical
            ``1e8`` because the latter pushed ``cond(K_mod)`` to ~2.4e9
            and capped LU-vs-CG agreement at ~3e-6 even when the
            iterative residual was at machine precision (issue #60).

        Returns
        -------
        K_mod : scipy.sparse.csc_matrix
        F_mod : np.ndarray
        """
        if not constrained_dofs:
            return K, F

        # Ensure K is in a sparse format that supports conversion to LIL
        if not scipy.sparse.issparse(K):
            K = scipy.sparse.csc_matrix(K)
        K_lil = K.tolil()
        F_mod = F.copy()

        diag_max = np.abs(K.diagonal()).max()
        alpha = penalty_factor * max(diag_max, 1.0)

        for dof, val in constrained_dofs.items():
            K_lil[dof, dof] += alpha
            F_mod[dof] = alpha * val

        return K_lil.tocsc(), F_mod


# ============================================================
# SECTION 7g: FE SOLVER AND FIELD RESULTS
# ============================================================

@dataclass
class FieldResults:
    """Results from a finite element solve.

    Attributes
    ----------
    displacement : np.ndarray
        Shape (n_nodes, 3) nodal displacements.
    stress_global : np.ndarray
        Shape (n_elem, n_gp, 6) stress in global coordinates.
    stress_local : np.ndarray
        Shape (n_elem, n_gp, 6) stress in local (material) coordinates.
    strain_global : np.ndarray
        Shape (n_elem, n_gp, 6) strain in global coordinates.
    strain_local : np.ndarray
        Shape (n_elem, n_gp, 6) strain in local coordinates.
    max_failure_index : float
        Maximum failure index across all Gauss points (criterion-dependent).
    knockdown : float
        Stiffness knockdown factor (modulus ratio: E_porous/E_pristine).
    per_element_failure_index : np.ndarray or None
        Shape (n_elem,) max-over-Gauss-point failure index per element.
        Optional (defaults to ``None`` for back-compatibility with callers
        that construct ``FieldResults`` directly); populated by
        ``FESolver.solve`` and consumed by the VTK export so failure
        hot-spots can be sliced in ParaView.
    failure_criterion : str
        Which failure criterion was used to produce ``max_failure_index`` and
        ``per_element_failure_index``. One of ``'tsai_wu'`` (default for
        back-compat), ``'hashin'``, or ``'max_stress'``.
    failure_mode_indices : dict or None
        Per-mode breakdown of the maximum failure index across the model with
        keys ``'fiber_t'``, ``'fiber_c'``, ``'matrix_t'``, ``'matrix_c'``,
        ``'shear'`` (plus ``'max_fi'``). For Tsai-Wu the per-mode entries are
        ``NaN`` (the polynomial does not separate modes); for ``max_stress``
        the unused entries are zero. Lets the GUI and JSON exporter report
        the dominant failure mode, not just severity.

    Notes
    -----
    Voigt order for both stress and strain: ``[11, 22, 33, 23, 13, 12]``
    (normals first, then 23 / 13 / 12 shears). The last three strain
    components are **engineering** strain (``gamma_ij = 2 * eps_ij``); the
    last three stress components are the matching shear stresses
    ``[tau_23, tau_13, tau_12]``. Sign convention: tensile normals are
    positive, compressive normals are negative — these arrays are signed
    (unlike the empirical ``failure_stress_MPa`` returned by
    :meth:`EmpiricalSolver.apply_loading`, which is a positive magnitude).
    Use :func:`strain_transformation_3d` / :func:`stress_transformation_3d`
    to rotate these arrays between frames.
    """
    displacement: np.ndarray
    stress_global: np.ndarray
    stress_local: np.ndarray
    strain_global: np.ndarray
    strain_local: np.ndarray
    max_failure_index: float
    knockdown: float
    per_element_failure_index: Optional[np.ndarray] = None
    failure_criterion: str = 'tsai_wu'
    failure_mode_indices: Optional[Dict[str, float]] = None

    def __repr__(self) -> str:
        n_nodes = self.displacement.shape[0] if self.displacement is not None else 0
        n_elem = self.stress_global.shape[0] if self.stress_global is not None else 0
        return (f"FieldResults(n_nodes={n_nodes}, n_elements={n_elem}, "
                f"max_FI={self.max_failure_index:.4f}, "
                f"knockdown={self.knockdown:.4f})")

    def summary(self, sigma_pristine: Optional[float] = None,
                model_label: Optional[str] = None) -> 'FailureResult':
        """Distill the field result into a :class:`FailureResult`.

        Unifies the FE return shape with the empirical solver (#44 item 1)
        so callers can treat the two solver outputs polymorphically.

        Parameters
        ----------
        sigma_pristine : float, optional
            Pristine reference stress (MPa) used to compute
            ``failure_stress = knockdown * sigma_pristine``. If ``None``
            (default), the ``failure_stress`` field is set to
            ``knockdown`` itself (unit knockdown — the bare ratio); pass
            the loading-mode-specific pristine strength
            (``material.sigma_1c`` for compression, ``material.tau_ilss``
            for ILSS, etc.) to get a meaningful magnitude.
        model_label : str, optional
            Label used for the ``model`` field. Defaults to
            ``f"fe_{self.failure_criterion}"`` so the FE summary is
            self-describing and distinguishable from the empirical labels.

        Returns
        -------
        FailureResult
            Unified summary with the FE ``knockdown``, derived
            ``failure_stress``, FE-criterion-tagged ``model`` and a
            ``details`` dict carrying ``max_failure_index``,
            ``failure_criterion`` and ``failure_mode_indices`` for
            downstream consumers that need the richer field-result data.
        """
        kd = float(self.knockdown)
        sigma_ref = float(sigma_pristine) if sigma_pristine is not None else 1.0
        return FailureResult(
            failure_stress=kd * sigma_ref,
            knockdown=kd,
            model=str(model_label) if model_label is not None
            else f"fe_{self.failure_criterion}",
            details={
                'max_failure_index': float(self.max_failure_index),
                'failure_criterion': self.failure_criterion,
                'failure_mode_indices': (
                    dict(self.failure_mode_indices)
                    if self.failure_mode_indices is not None else None
                ),
            },
        )

    def to_vtk(self, mesh: 'CompositeMesh', filename: str) -> None:
        """Write the hex mesh and per-element FE fields to a legacy ASCII VTK
        file (``UNSTRUCTURED_GRID``) for inspection in ParaView / VisIt / PyVista.

        The writer is dependency-free: it emits the legacy VTK 3.0 ASCII
        format by hand. The 8-node hex connectivity already stored in
        ``mesh.elements`` follows the standard VTK hexahedron ordering
        (bottom face CCW then top face CCW, see ``_NODE_COORDS_REF``), so the
        cells are written verbatim with cell type 12 (``VTK_HEXAHEDRON``).

        Point data
        -----------
        - ``displacement`` (3-vector), and the scalars ``porosity``,
          ``stiffness_reduction``, ``ply_id`` if present on the mesh.

        Cell data
        ---------
        - element-averaged ``von_mises`` and the six global stress
          (``sigma_xx`` .. ``tau_xy``) and strain (``eps_xx`` .. ``gamma_xy``)
          components (Gauss points reduced by mean),
        - ``tsai_wu_index`` (max-over-GP per element, if available),
        - ``Vp_elem`` (mean nodal porosity over the 8 corners),
        - ``ply_id``, ``ply_angle_deg``, ``is_void`` and ``knockdown`` where
          available.

        Parameters
        ----------
        mesh : CompositeMesh
            The mesh that produced these results (supplies geometry,
            connectivity, porosity and ply metadata).
        filename : str
            Output ``.vtk`` file path.
        """
        nodes = np.asarray(mesh.nodes, dtype=float)
        elements = np.asarray(mesh.elements, dtype=np.int64)
        n_nodes = nodes.shape[0]
        n_elem = elements.shape[0]

        if elements.shape[1] != 8:
            raise ValueError(
                f"to_vtk only supports 8-node hexahedra; got connectivity "
                f"of width {elements.shape[1]}."
            )
        if self.displacement is not None and \
                self.displacement.shape[0] != n_nodes:
            raise ValueError(
                f"displacement has {self.displacement.shape[0]} rows but the "
                f"mesh has {n_nodes} nodes; results do not match this mesh."
            )

        # Gauss-point-averaged global stress/strain -> (n_elem, 6)
        sig = np.mean(self.stress_global, axis=1)
        eps = np.mean(self.strain_global, axis=1)

        # Element-averaged von Mises from the averaged stress tensor.
        sxx, syy, szz = sig[:, 0], sig[:, 1], sig[:, 2]
        tyz, txz, txy = sig[:, 3], sig[:, 4], sig[:, 5]
        von_mises = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2)
        )

        def _fmt(values) -> str:
            return "\n".join(repr(float(v)) for v in np.asarray(values).ravel())

        def _fmt_xyz(rows) -> str:
            return "\n".join(
                f"{float(r[0])!r} {float(r[1])!r} {float(r[2])!r}"
                for r in rows
            )

        lines = [
            "# vtk DataFile Version 3.0",
            "PorosityFE results (hex mesh + per-element fields)",
            "ASCII",
            "DATASET UNSTRUCTURED_GRID",
            f"POINTS {n_nodes} float",
        ]
        lines.append(_fmt_xyz(nodes))

        # CELLS: each line is "8 n0 n1 ... n7"; total size = n_elem * 9.
        lines.append(f"CELLS {n_elem} {n_elem * 9}")
        lines.append(
            "\n".join(
                "8 " + " ".join(str(int(i)) for i in conn) for conn in elements
            )
        )
        lines.append(f"CELL_TYPES {n_elem}")
        lines.append("\n".join("12" for _ in range(n_elem)))

        # ---- POINT_DATA ----
        lines.append(f"POINT_DATA {n_nodes}")
        if self.displacement is not None:
            disp = np.asarray(self.displacement, dtype=float)
            lines.append("VECTORS displacement float")
            lines.append(_fmt_xyz(disp))

        def _point_scalar(name: str, arr) -> None:
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.shape[0] != n_nodes:
                return
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.append(_fmt(arr))

        if getattr(mesh, 'porosity', None) is not None:
            _point_scalar("porosity", mesh.porosity)
        if getattr(mesh, 'stiffness_reduction', None) is not None:
            _point_scalar("stiffness_reduction", mesh.stiffness_reduction)
        if getattr(mesh, 'ply_ids', None) is not None:
            _point_scalar("ply_id", mesh.ply_ids)

        # ---- CELL_DATA ----
        lines.append(f"CELL_DATA {n_elem}")

        def _cell_scalar(name: str, arr) -> None:
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.shape[0] != n_elem:
                return
            lines.append(f"SCALARS {name} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.append(_fmt(arr))

        _cell_scalar("von_mises", von_mises)
        for idx, comp in enumerate(
                ("sigma_xx", "sigma_yy", "sigma_zz",
                 "tau_yz", "tau_xz", "tau_xy")):
            _cell_scalar(comp, sig[:, idx])
        for idx, comp in enumerate(
                ("eps_xx", "eps_yy", "eps_zz",
                 "gamma_yz", "gamma_xz", "gamma_xy")):
            _cell_scalar(comp, eps[:, idx])

        if self.per_element_failure_index is not None:
            _cell_scalar("tsai_wu_index", self.per_element_failure_index)

        # Element-averaged nodal porosity over the 8 corner nodes.
        if getattr(mesh, 'porosity', None) is not None:
            vp_elem = np.mean(
                np.asarray(mesh.porosity, dtype=float)[elements], axis=1)
            _cell_scalar("Vp_elem", vp_elem)
        if getattr(mesh, 'elem_ply_ids', None) is not None:
            _cell_scalar("ply_id", mesh.elem_ply_ids)
        if getattr(mesh, 'ply_angles', None) is not None:
            _cell_scalar("ply_angle_deg", mesh.ply_angles)
        if getattr(mesh, 'void_elements', None) is not None:
            is_void = np.zeros(n_elem, dtype=float)
            void_idx = np.asarray(mesh.void_elements, dtype=np.int64).ravel()
            if void_idx.size:
                is_void[void_idx] = 1.0
            _cell_scalar("is_void", is_void)

        _cell_scalar(
            "knockdown",
            np.full(n_elem, float(self.knockdown), dtype=float))

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            f.write("\n")
        logger.info("Saved FE results (VTK): %s", filename)


class FESolver:
    """Linear static FE solver for porosity-degraded composite laminates.

    Workflow:
    1. Assemble K via GlobalAssembler
    2. Build BCs via BoundaryHandler
    3. Apply penalty method
    4. Solve K*u = F via spsolve
    5. Recover stresses at Gauss points
    6. Evaluate Tsai-Wu failure at each GP
    7. Compute knockdown factor

    Parameters
    ----------
    mesh : CompositeMesh
        The finite element mesh.
    material : MaterialProperties
        Material properties.
    porosity_field : PorosityField
        Porosity field for stiffness degradation.
    ply_angles : list of float or {'QI', 'UD'}, optional
        Optional list of ply angles (degrees), OR a string sentinel —
        ``'QI'`` (default, ``[0, 90, 45, -45]_s``) or ``'UD'`` (all-zero
        unidirectional). When provided, the resolved angle list is
        forwarded into the underlying :class:`CompositeMesh` per-element
        ``ply_angles`` array via :meth:`CompositeMesh.generate_mesh` only
        if the mesh has not yet been laid up with it — the mesh's
        ``ply_angles`` field remains the authoritative source for the
        per-element transformations used during solve. Passing ``None``
        is deprecated and resolved to ``'QI'`` with a
        :class:`DeprecationWarning` (#44 item 2).

    Notes
    -----
    ``ply_angles`` defaults — ``'QI'`` is the standardised default across
    :class:`EmpiricalSolver`, :class:`CompositeMesh`, and
    :class:`FESolver` (#44 item 2). The string sentinels expand to
    canonical baselines; explicit lists pass through unchanged. The
    constructor stores the resolved value on ``self.ply_angles`` so
    callers can introspect what layup the solver was built for; the
    actual per-element angles used during ``solve()`` come from
    ``self.mesh.ply_angles``, which is set when the mesh was constructed
    (passing ``ply_angles`` here does *not* relayup the mesh).
    """

    #: Supported failure criteria for :meth:`solve`. Used both at runtime
    #: (for validation) and as the documented enumeration.
    SUPPORTED_FAILURE_CRITERIA: Tuple[str, ...] = (
        'tsai_wu', 'hashin', 'max_stress')

    def __init__(self, mesh: CompositeMesh, material: MaterialProperties,
                 porosity_field: PorosityField,
                 ply_angles: Optional[Union[List[float], str]] = 'QI',
                 failure_criterion: Literal[
                     'tsai_wu', 'hashin', 'max_stress'] = 'tsai_wu') -> None:
        self.mesh = mesh
        self.material = material
        self.porosity_field = porosity_field
        # Resolve the ply_angles sentinel (#44 item 2). The resolved value
        # is stored for introspection; the per-element angles consumed by
        # solve() come from ``mesh.ply_angles`` (set by the mesh's own
        # constructor). Previously ``ply_angles`` was stored-but-unused;
        # documenting the intentional decoupling here so the field has an
        # explicit contract.
        self.ply_angles = _resolve_ply_angles(
            ply_angles, none_means='QI', caller='FESolver.ply_angles')
        self.assembler = GlobalAssembler(mesh, material, porosity_field)
        self.bc_handler = BoundaryHandler(mesh)
        if failure_criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {failure_criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )
        self.failure_criterion = failure_criterion

    def solve(self, loading: str = 'compression',
              applied_strain: float = -0.01,
              applied_load: float = -10.0,
              verbose: bool = False,
              failure_criterion: Optional[Literal[
                  'tsai_wu', 'hashin', 'max_stress']] = None,
              solver: Literal['direct', 'cg', 'minres'] = 'direct',
              rtol: float = 1e-9,
              diag_scale: bool = False,
              penalty_factor: float = 1e6) -> FieldResults:
        """Solve the static FE problem.

        Parameters
        ----------
        loading : str
            'compression', 'tension', 'shear', or 'ilss'.
        applied_strain : float
            Applied nominal strain (negative for compression). Used by the
            displacement-controlled modes ('compression', 'tension',
            'shear').
        applied_load : float
            Total midspan load (force) used by the force-controlled ILSS
            short-beam-shear mode (ASTM D2344). Ignored for the other
            modes.
        verbose : bool
            Print progress information.
        failure_criterion : {'tsai_wu', 'hashin', 'max_stress'}, optional
            Per-call override for the failure criterion. Defaults to the
            value passed to :meth:`__init__` (``'tsai_wu'`` if unset). When
            ``'tsai_wu'`` the result is bit-identical to the historical
            behavior; ``'hashin'`` and ``'max_stress'`` populate the
            per-mode breakdown on :class:`FieldResults`.
        solver : {'direct', 'cg', 'minres'}
            Linear solver to use for ``K u = F``. ``'direct'`` (default)
            uses :func:`scipy.sparse.linalg.spsolve` (sparse LU). For
            large meshes the LU fill-in dominates RAM; the penalty-modified
            matrix is SPD, so ``'cg'`` (conjugate gradient) with a Jacobi
            preconditioner is a memory-light alternative. ``'minres'``
            is offered for completeness when the matrix is symmetric but
            not strictly positive definite. Auto-switching is intentionally
            *not* performed — callers select the path explicitly
            (issue #57).
        rtol : float
            Relative-residual tolerance for the iterative solvers. Ignored
            when ``solver='direct'``.
        diag_scale : bool, optional
            If ``True``, symmetrically Jacobi-pre-scale the penalty-
            modified system before solving:
            ``(D^{-1/2} K_mod D^{-1/2}) y = D^{-1/2} F_mod``,
            ``u = D^{-1/2} y``, where ``D = diag(K_mod)``. The math is
            unchanged but the diagonal-conditioning ratio is reduced by
            2-3 decades on graded/voided meshes, which improves both LU
            backward error and CG/MINRES convergence. Defaults to
            ``False`` to preserve bit-identical legacy behavior; opt in
            when conditioning is a concern (issue #60).
        penalty_factor : float, optional
            Multiplier on ``max(diag(K))`` used by
            :meth:`BoundaryHandler.apply_penalty` to enforce Dirichlet
            BCs. Lowered from ``1e8`` to ``1e6`` (default) in issue #60
            to keep ``cond(K_mod)`` well below the float64 ceiling while
            still enforcing BCs to six decades. Tune higher only if BC
            slack is a problem; tune lower if conditioning is.

        Returns
        -------
        FieldResults
            Complete solution data.

        Raises
        ------
        ValueError
            If ``solver`` is not one of ``'direct'``, ``'cg'``, ``'minres'``.
        RuntimeError
            If the iterative solver fails to converge to ``rtol``, or if
            the direct solve produces non-finite values / a residual above
            ``1e-6``.
        """
        import time
        t0 = time.perf_counter()
        criterion = failure_criterion if failure_criterion is not None \
            else self.failure_criterion
        if criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure_criterion {criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )

        # 0. Mesh quality check
        check_mesh_quality(self.mesh, verbose=verbose)

        # 1. Assemble global stiffness
        if verbose:
            logger.info("Assembling global stiffness matrix...")
        K = self.assembler.assemble_stiffness(verbose=verbose)

        if verbose:
            t1 = time.perf_counter()
            logger.info("  Assembly time: %.2f s", t1 - t0)

        # 2. Build BCs
        if loading == 'compression':
            constrained, F = self.bc_handler.compression_bcs(applied_strain)
        elif loading == 'tension':
            constrained, F = self.bc_handler.tension_bcs(applied_strain)
        elif loading == 'shear':
            constrained, F = self.bc_handler.shear_bcs(applied_strain)
        elif loading == 'ilss':
            constrained, F = self.bc_handler.ilss_bcs(applied_load)
        else:
            raise ValueError(
                f"Unknown loading '{loading}'. "
                "Use compression/tension/shear/ilss."
            )

        if verbose:
            logger.info("  Applied %d displacement BCs", len(constrained))

        # 3. Apply penalty
        K_mod, F_mod = BoundaryHandler.apply_penalty(
            K, F, constrained, penalty_factor=penalty_factor,
        )

        # 3a. Conditioning diagnostic (issue #60). The diagonal ratio is
        # an inexpensive proxy for cond(K_mod) — full condest is O(n^2)
        # for sparse matrices and we want this on every solve. Warn the
        # user well before float64's ~1e16 headroom is exhausted.
        _diag = K_mod.diagonal()
        _diag_abs = np.abs(_diag)
        _diag_min = float(_diag_abs[_diag_abs > 0.0].min()) \
            if np.any(_diag_abs > 0.0) else 0.0
        _diag_max = float(_diag_abs.max()) if _diag_abs.size else 0.0
        cond_diag_ratio = (_diag_max / _diag_min) if _diag_min > 0.0 \
            else float('inf')
        logger.info(
            "Matrix conditioning: cond_diag_ratio=%.4e "
            "(penalty_factor=%.2e, diag_scale=%s)",
            cond_diag_ratio, penalty_factor, diag_scale,
        )
        if cond_diag_ratio > 1e12:
            logger.warning(
                "Matrix conditioning near float64 limit "
                "(cond_diag_ratio=%.2e); consider lowering "
                "penalty_factor or enabling diag_scale.",
                cond_diag_ratio,
            )

        # 4. Solve
        if solver not in ('direct', 'cg', 'minres'):
            raise ValueError(
                f"Unknown solver '{solver}'. "
                "Use 'direct', 'cg', or 'minres'."
            )
        if verbose:
            logger.info(
                "Solving system (%d DOFs) with solver='%s'...",
                self.mesh.n_dof, solver,
            )

        # 4a. Optional symmetric Jacobi pre-scaling (issue #60).
        # Replace (K_mod, F_mod) with (K_scaled, F_scaled) for the solve;
        # after solving, unscale y -> u via u = d_inv_sqrt * y.
        if diag_scale:
            _d = K_mod.diagonal()
            if not np.all(_d > 0):
                raise RuntimeError(
                    "Cannot apply diag_scale: K_mod has a non-positive "
                    "diagonal entry. Check assembly / penalty."
                )
            d_inv_sqrt = 1.0 / np.sqrt(_d)
            _D_is = scipy.sparse.diags(d_inv_sqrt)
            K_solve = (_D_is @ K_mod) @ _D_is
            F_solve = d_inv_sqrt * F_mod
            # Log the post-scaling diagonal ratio so the user can see
            # what the rescaling bought them.
            _d_scaled = K_solve.diagonal()
            _d_scaled_abs = np.abs(_d_scaled)
            _ds_min = float(_d_scaled_abs[_d_scaled_abs > 0.0].min()) \
                if np.any(_d_scaled_abs > 0.0) else 0.0
            _ds_max = float(_d_scaled_abs.max()) \
                if _d_scaled_abs.size else 0.0
            cond_diag_ratio_scaled = (_ds_max / _ds_min) \
                if _ds_min > 0.0 else float('inf')
            logger.info(
                "Matrix conditioning after diag_scale: "
                "cond_diag_ratio=%.4e (was %.4e)",
                cond_diag_ratio_scaled, cond_diag_ratio,
            )
        else:
            K_solve = K_mod
            F_solve = F_mod
            d_inv_sqrt = None

        if solver == 'direct':
            y = scipy.sparse.linalg.spsolve(K_solve, F_solve)

            # Hygiene checks on the solution vector
            if not np.isfinite(y).all():
                raise RuntimeError(
                    "spsolve produced non-finite values (NaN or Inf) in the solution "
                    "vector. Check matrix conditioning and boundary conditions."
                )
            _r = K_solve @ y - F_solve
            _rel_res = np.linalg.norm(_r) / max(np.linalg.norm(F_solve), 1.0)  # type: ignore[call-overload,operator]
            if _rel_res >= 1e-6:
                raise RuntimeError(
                    f"spsolve residual {_rel_res:.4e} exceeds tolerance 1e-6. "
                    "Check matrix conditioning or penalty factor."
                )
        else:
            # Jacobi (diagonal) preconditioner: K is SPD after penalty,
            # diag(K) is strictly positive.
            diag = K_solve.diagonal()
            if not np.all(diag > 0):
                raise RuntimeError(
                    "Cannot build Jacobi preconditioner: K_mod has a "
                    "non-positive diagonal entry. Check assembly / penalty."
                )
            M = scipy.sparse.diags(1.0 / diag)

            if solver == 'cg':
                y, info = scipy.sparse.linalg.cg(
                    K_solve, F_solve, M=M, rtol=rtol,
                )
            else:  # solver == 'minres'
                y, info = scipy.sparse.linalg.minres(
                    K_solve, F_solve, M=M, rtol=rtol,
                )

            _r = K_solve @ y - F_solve
            _norm_b = float(np.linalg.norm(F_solve))  # type: ignore[call-overload]
            _rel_res = float(
                np.linalg.norm(_r) / _norm_b if _norm_b > 0.0 else 0.0  # type: ignore[call-overload,operator]
            )
            # Compare the achieved relative residual against the user-
            # requested rtol directly. SciPy's iterative solvers can
            # report info=0 while still bouncing off the machine-
            # precision floor — if the user asked for sub-eps tolerance
            # they will (correctly) get a non-convergence error.
            _converged = info == 0 and _rel_res <= rtol * 10.0
            if not _converged:
                raise RuntimeError(
                    f"{solver} failed to converge: info={info}, "
                    f"achieved relative residual {_rel_res:.4e} "
                    f"(requested rtol={rtol:.4e})."
                )
            logger.info(
                "%s converged: relative residual %.4e (rtol=%.4e)",
                solver, _rel_res, rtol,
            )

        # 4b. Unscale if we Jacobi-pre-scaled. ``y`` solves the scaled
        # system; the physical displacement is ``u = D^{-1/2} y``.
        if diag_scale:
            u = d_inv_sqrt * y
        else:
            u = y

        if verbose:
            t2 = time.perf_counter()
            logger.info(
                "  Solve time: %.2f s, residual: %.4e", t2 - t1, _rel_res)
            t1 = t2

        # 5. Recover stresses and strains
        if verbose:
            logger.info("Recovering element stresses and strains...")

        n_elem = self.mesh.n_elements
        n_gp = 8  # 2x2x2

        stress_global = np.empty((n_elem, n_gp, 6))
        stress_local = np.empty((n_elem, n_gp, 6))
        strain_global = np.empty((n_elem, n_gp, 6))
        strain_local = np.empty((n_elem, n_gp, 6))

        for e in range(n_elem):
            if verbose and e % 500 == 0:
                logger.info(
                    "  Post-processing element %d/%d (%.1f%%)",
                    e, n_elem, 100.0 * e / n_elem,
                )

            dofs = self.assembler.element_dof_indices(e)
            u_elem = u[dofs]
            elem = self.assembler.create_element(e)

            sig_g = elem.stress_at_gauss_points(u_elem)
            eps_g = elem.strain_at_gauss_points(u_elem)

            stress_global[e] = sig_g
            strain_global[e] = eps_g

            # Transform to local coordinates. Stress uses T_sigma; engineering
            # strain (with gamma_ij = 2*eps_ij in slots 3-5) uses T_epsilon —
            # T_sigma applied to engineering strain leaves the shear components
            # off by 2x.
            ply_rad = np.radians(float(self.mesh.ply_angles[e]))
            T_sigma = stress_transformation_3d(ply_rad, axis='z')
            T_eps = strain_transformation_3d(ply_rad, axis='z')

            for g in range(n_gp):
                stress_local[e, g] = T_sigma @ sig_g[g]
                strain_local[e, g] = T_eps @ eps_g[g]

        # 6. Evaluate the selected failure criterion at each GP.
        #    per_elem_fi[e] is the max-over-GP failure index for element e
        #    (0.0 for skipped void elements); the scalar max_fi is its
        #    overall maximum. mode_indices captures the per-mode breakdown
        #    (NaN entries for Tsai-Wu, which does not separate modes).
        max_fi, per_elem_fi, mode_indices = self._evaluate_failure(
            stress_local, criterion=criterion)

        # 7. Compute knockdown as average-stress ratio (porous / pristine)
        # Both numerator and denominator use the same 3D FE framework so that
        # dimensional/mesh effects cancel.  For each element we compute what
        # the dominant stress component *would* be with pristine stiffness
        # at the same strain, then average. This avoids the CLT-vs-3D
        # mismatch that caused knockdown > 1.
        #
        # For ILSS short-beam shear the dominant component is tau_xz
        # (Voigt index 4); for the other modes it is sigma_xx (index 0).
        if loading == 'ilss':
            comp_idx = 4
        else:
            comp_idx = 0

        avg_sigma = np.mean(stress_global[:, :, comp_idx])

        # Pristine reference: compute the same Voigt component using the
        # rotated pristine stiffness applied to the recovered strain field.
        C_base = self.material.get_stiffness_matrix()
        pristine_sigma_sum = 0.0
        pristine_count = 0
        for e in range(n_elem):
            ply_rad = np.radians(float(self.mesh.ply_angles[e]))
            if abs(ply_rad) > 1e-15:
                C_prist_rot = rotate_stiffness_3d(C_base, ply_rad, axis='z')
            else:
                C_prist_rot = C_base
            for g in range(n_gp):
                eps = strain_global[e, g]
                pristine_sig = float(C_prist_rot[comp_idx, :] @ eps)
                pristine_sigma_sum += pristine_sig
                pristine_count += 1

        pristine_avg = pristine_sigma_sum / pristine_count if pristine_count > 0 else 1.0

        if abs(pristine_avg) > 1e-12:
            knockdown = abs(avg_sigma) / abs(pristine_avg)
        else:
            knockdown = 1.0
        knockdown = min(knockdown, 1.0)

        displacement = u.reshape(-1, 3)

        if verbose:
            t3 = time.perf_counter()
            logger.info("  Post-processing time: %.2f s", t3 - t1)
            logger.info("Total solve time: %.2f s", t3 - t0)
            logger.info("  Max %s FI: %.4f", criterion, max_fi)
            logger.info("  Knockdown factor: %.4f", knockdown)

        return FieldResults(
            displacement=displacement,
            stress_global=stress_global,
            stress_local=stress_local,
            strain_global=strain_global,
            strain_local=strain_local,
            max_failure_index=max_fi,
            knockdown=knockdown,
            per_element_failure_index=per_elem_fi,
            failure_criterion=criterion,
            failure_mode_indices=mode_indices,
        )

    #: Empty per-mode failure-index dict used when an element is skipped
    #: (void) or for criteria that do not populate a particular mode.
    _EMPTY_MODE_FI: Dict[str, float] = {
        'max_fi': 0.0,
        'fiber_t': 0.0,
        'fiber_c': 0.0,
        'matrix_t': 0.0,
        'matrix_c': 0.0,
        'shear': 0.0,
    }

    def _degraded_strengths(self, elem_Vp: float
                            ) -> Tuple[float, float, float, float, float, float]:
        """Return per-element porosity-degraded ply strengths.

        Implements the strength-degradation block shared by all failure
        criteria. Fiber-direction strengths (``Xt``, ``Xc``) follow the rule-
        of-mixtures fiber ratio (matrix porosity has only a weak indirect
        effect via ``E_m_eff``); transverse and shear strengths
        (``Yt``, ``Yc``, ``S12``, ``S23``) follow the Mori-Tanaka matrix
        stiffness ratio. Strengths are clamped to a small numerical floor so
        the per-criterion polynomial cannot divide by zero.

        Parameters
        ----------
        elem_Vp : float
            Element-average void volume fraction in [0, 1].

        Returns
        -------
        (Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s) : tuple of floats
            Floor-clamped degraded strengths in MPa.
        """
        mat = self.material
        C_m_pristine = mat.get_isotropic_matrix_stiffness()
        if elem_Vp > 1e-12:
            C_eff = _mt_effective_stiffness(
                C_m_pristine, elem_Vp,
                self.porosity_field.void_shape_radii,
                mat.matrix_poisson)
            # Matrix stiffness degradation ratio (matrix-dominated)
            r_matrix = np.sqrt(max(C_eff[0, 0] / C_m_pristine[0, 0], 0.0))
            # Fiber-direction ratio: scale by ROM ratio (much weaker effect)
            E_m = mat.matrix_modulus
            E_m_eff_approx = E_m * max(C_eff[0, 0] / C_m_pristine[0, 0], 0.0)
            Vf = mat.fiber_volume_fraction
            Vm = 1.0 - Vf
            r_fiber = (Vf * mat.fiber_modulus + Vm * E_m_eff_approx) / \
                      (Vf * mat.fiber_modulus + Vm * E_m)
            r_fiber = np.sqrt(max(r_fiber, 0.0))  # sqrt for strength vs stiffness
        else:
            r_matrix = 1.0
            r_fiber = 1.0

        Xt = mat.sigma_1t * r_fiber
        Xc = mat.sigma_1c * r_fiber
        Yt = mat.sigma_2t * r_matrix
        Yc = mat.sigma_2c * r_matrix
        S12 = mat.tau_12 * r_matrix
        S23 = mat.tau_ilss * r_matrix

        # Strengths approaching zero make the 1/X reciprocals overflow to
        # inf; clamp to a numerical floor so a heavily-degraded element
        # produces a large-but-finite failure index instead of poisoning the
        # global max with inf/NaN.
        strength_floor = 1e-3  # MPa
        return (max(Xt, strength_floor),
                max(Xc, strength_floor),
                max(Yt, strength_floor),
                max(Yc, strength_floor),
                max(S12, strength_floor),
                max(S23, strength_floor))

    def _evaluate_failure(self, stress_local: np.ndarray,
                          criterion: str = 'tsai_wu'
                          ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Evaluate the chosen failure criterion at every Gauss point.

        Dispatches to :meth:`_evaluate_tsai_wu`, :meth:`_evaluate_hashin`,
        or :meth:`_evaluate_max_stress` element by element. Per-element
        strength degradation is computed once via :meth:`_degraded_strengths`
        and reused by the per-criterion polynomials.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape (n_elem, n_gp, 6) local stresses.
        criterion : {'tsai_wu', 'hashin', 'max_stress'}
            Failure criterion to apply.

        Returns
        -------
        max_fi : float
            Overall maximum failure index.
        per_elem_fi : np.ndarray
            Shape (n_elem,) max-over-Gauss-point failure index per element
            (0.0 for skipped void elements).
        mode_indices : dict
            Per-mode breakdown at the element/GP where ``max_fi`` is
            attained. Keys: ``'max_fi'``, ``'fiber_t'``, ``'fiber_c'``,
            ``'matrix_t'``, ``'matrix_c'``, ``'shear'``. For Tsai-Wu the
            per-mode entries are ``NaN`` (the coupled polynomial does not
            separate modes).
        """
        if criterion not in self.SUPPORTED_FAILURE_CRITERIA:
            raise ValueError(
                f"Unknown failure criterion {criterion!r}. "
                f"Use one of {list(self.SUPPORTED_FAILURE_CRITERIA)}."
            )

        n_elem, _, _ = stress_local.shape
        per_elem_fi = np.zeros(n_elem, dtype=float)
        # Defense in depth: non-finite porosity silently corrupts elem_Vp.
        if not np.all(np.isfinite(self.mesh.porosity)):  # type: ignore[call-overload]
            raise ValueError(
                f"mesh.porosity contains non-finite values; refusing to evaluate "
                f"{criterion} on a corrupted porosity field."
            )

        # #114: hoist the per-element mean computation outside the loop.
        # The old `np.mean(porosity[elements[e]])` per iteration was an
        # O(n_elem) Python loop where O(1) vectorized NumPy works; this gives
        # ~143x on the inner step and ~1-2 s on a typical 5x5 sweep.
        elem_Vp_all = np.clip(
            np.mean(self.mesh.porosity[self.mesh.elements], axis=1),
            0.0, 1.0,
        )

        max_fi = 0.0
        best_mode_indices: Dict[str, float] = dict(self._EMPTY_MODE_FI)
        if criterion == 'tsai_wu':
            # Tsai-Wu polynomial couples all components — per-mode breakdown
            # is undefined. Surface NaN sentinels so downstream consumers see
            # "criterion did not separate modes" rather than spurious zeros.
            best_mode_indices = {
                'max_fi': 0.0,
                'fiber_t': float('nan'),
                'fiber_c': float('nan'),
                'matrix_t': float('nan'),
                'matrix_c': float('nan'),
                'shear': float('nan'),
            }

        for e in range(n_elem):
            elem_Vp = float(elem_Vp_all[e])

            # Skip void elements (carry no meaningful load)
            if elem_Vp > 0.95:
                continue

            strengths = self._degraded_strengths(elem_Vp)
            s_all = stress_local[e]  # (n_gp, 6)

            if criterion == 'tsai_wu':
                fi_per_gp = self._evaluate_tsai_wu(s_all, strengths, e, elem_Vp)
                elem_max = float(fi_per_gp.max())
                per_elem_fi[e] = elem_max
                if elem_max > max_fi:
                    max_fi = elem_max
                    best_mode_indices['max_fi'] = elem_max
            else:
                if criterion == 'hashin':
                    mode_fi_per_gp = self._evaluate_hashin(s_all, strengths)
                else:  # 'max_stress'
                    mode_fi_per_gp = self._evaluate_max_stress(s_all, strengths)
                # mode_fi_per_gp is a dict of arrays, each shape (n_gp,).
                fi_per_gp = mode_fi_per_gp['max_fi']
                if not np.all(np.isfinite(fi_per_gp)):
                    bad_g = int(np.argmax(~np.isfinite(fi_per_gp)))
                    raise ValueError(
                        f"{criterion} failure index is non-finite at element "
                        f"{e}, Gauss point {bad_g} (Vp={elem_Vp:.4f}, "
                        f"stress={s_all[bad_g].tolist()})."
                    )
                elem_max = float(fi_per_gp.max())
                per_elem_fi[e] = elem_max
                if elem_max > max_fi:
                    max_fi = elem_max
                    g_max = int(np.argmax(fi_per_gp))
                    best_mode_indices = {
                        'max_fi': elem_max,
                        'fiber_t': float(mode_fi_per_gp['fiber_t'][g_max]),
                        'fiber_c': float(mode_fi_per_gp['fiber_c'][g_max]),
                        'matrix_t': float(mode_fi_per_gp['matrix_t'][g_max]),
                        'matrix_c': float(mode_fi_per_gp['matrix_c'][g_max]),
                        'shear': float(mode_fi_per_gp['shear'][g_max]),
                    }

        return float(max_fi), per_elem_fi, best_mode_indices

    def _evaluate_tsai_wu(self, s_all: np.ndarray,
                          strengths: Tuple[float, float, float, float, float, float],
                          e: int, elem_Vp: float) -> np.ndarray:
        """Tsai-Wu polynomial evaluated for one element's Gauss points.

        Bit-identical to the historical implementation. Returns the per-GP
        failure index array (shape ``(n_gp,)``).
        """
        Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            F1 = 1.0 / Xt_s - 1.0 / Xc_s
            F2 = 1.0 / Yt_s - 1.0 / Yc_s
            F3 = F2
            F11 = 1.0 / (Xt_s * Xc_s)
            F22 = 1.0 / (Yt_s * Yc_s)
            F33 = F22
            F44 = 1.0 / S23_s**2
            F55 = 1.0 / S12_s**2
            F66 = 1.0 / S12_s**2
            # F12, F23 use sqrt of a product. Guard against negative
            # products in case future refactors break the F11/F22/F33 sign.
            F11_F22 = max(F11 * F22, 0.0)
            F22_F33 = max(F22 * F33, 0.0)
            F12 = -0.5 * np.sqrt(F11_F22)
            F13 = F12
            F23 = -0.5 * np.sqrt(F22_F33)

        # Vectorize across all Gauss points of this element (#41).
        fi_per_gp = (
            F1 * s_all[:, 0] + F2 * s_all[:, 1] + F3 * s_all[:, 2]
            + F11 * s_all[:, 0]**2 + F22 * s_all[:, 1]**2 + F33 * s_all[:, 2]**2
            + F44 * s_all[:, 3]**2 + F55 * s_all[:, 4]**2 + F66 * s_all[:, 5]**2
            + 2 * F12 * s_all[:, 0] * s_all[:, 1]
            + 2 * F13 * s_all[:, 0] * s_all[:, 2]
            + 2 * F23 * s_all[:, 1] * s_all[:, 2]
        )
        if not np.all(np.isfinite(fi_per_gp)):
            bad_g = int(np.argmax(~np.isfinite(fi_per_gp)))
            raise ValueError(
                f"Tsai-Wu failure index is non-finite at element {e}, "
                f"Gauss point {bad_g} (Vp={elem_Vp:.4f}, "
                f"stress={s_all[bad_g].tolist()}). This usually indicates "
                f"a degenerate stiffness or strength matrix; refine the "
                f"mesh or check input bounds."
            )
        return fi_per_gp

    def _evaluate_hashin(self, s_all: np.ndarray,
                         strengths: Tuple[float, float, float, float, float, float]
                         ) -> Dict[str, np.ndarray]:
        """2D Hashin failure indices for unidirectional plies.

        Implements the standard four-mode Hashin criterion (Hashin, 1980).
        Indices are computed per Gauss point on the in-plane local stresses
        ``(σ_11, σ_22, τ_12)``; ``σ_33`` and out-of-plane shears are ignored
        because the standard formulation is 2D. The ``shear`` slot returns
        the in-plane ``(τ_12 / S_12)^2`` contribution for completeness.

        Returns a dict of per-GP arrays:
        ``{'max_fi', 'fiber_t', 'fiber_c', 'matrix_t', 'matrix_c', 'shear'}``.
        """
        Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
        sigma_11 = s_all[:, 0]
        sigma_22 = s_all[:, 1]
        tau_12 = s_all[:, 5]

        # Fiber tension: σ_11 >= 0
        ft = (sigma_11 / Xt_s) ** 2 + (tau_12 / S12_s) ** 2
        ft = np.where(sigma_11 >= 0.0, ft, 0.0)

        # Fiber compression: σ_11 < 0
        fc = (sigma_11 / Xc_s) ** 2
        fc = np.where(sigma_11 < 0.0, fc, 0.0)

        # Matrix tension: σ_22 >= 0
        mt = (sigma_22 / Yt_s) ** 2 + (tau_12 / S12_s) ** 2
        mt = np.where(sigma_22 >= 0.0, mt, 0.0)

        # Matrix compression: σ_22 < 0
        mc_term = ((Yc_s / (2.0 * S23_s)) ** 2 - 1.0) * (sigma_22 / Yc_s)
        mc = (sigma_22 / (2.0 * S23_s)) ** 2 + mc_term + (tau_12 / S12_s) ** 2
        mc = np.where(sigma_22 < 0.0, mc, 0.0)

        shear = (tau_12 / S12_s) ** 2

        max_fi = np.maximum.reduce([ft, fc, mt, mc])
        return {
            'max_fi': max_fi,
            'fiber_t': ft,
            'fiber_c': fc,
            'matrix_t': mt,
            'matrix_c': mc,
            'shear': shear,
        }

    def _evaluate_max_stress(self, s_all: np.ndarray,
                             strengths: Tuple[float, float, float, float, float, float]
                             ) -> Dict[str, np.ndarray]:
        """Maximum-stress failure indices.

        ``FI_i = |σ_i| / X_i_allowable`` per component (signed split for
        normals: tensile vs compressive allowable). The reported ``max_fi``
        is the maximum across all five mode/component buckets. Returns the
        same per-GP dict shape as :meth:`_evaluate_hashin`; unused entries
        are zeroed (rather than NaN) since each mode is well-defined for
        max-stress.
        """
        Xt_s, Xc_s, Yt_s, Yc_s, S12_s, S23_s = strengths
        sigma_11 = s_all[:, 0]
        sigma_22 = s_all[:, 1]
        sigma_33 = s_all[:, 2]
        tau_23 = s_all[:, 3]
        tau_13 = s_all[:, 4]
        tau_12 = s_all[:, 5]

        ft = np.where(sigma_11 >= 0.0, sigma_11 / Xt_s, 0.0)
        fc = np.where(sigma_11 < 0.0, -sigma_11 / Xc_s, 0.0)
        # Matrix uses worst of σ_22 and σ_33 (transverse normals share the
        # same in-plane transverse strength).
        mt_22 = np.where(sigma_22 >= 0.0, sigma_22 / Yt_s, 0.0)
        mt_33 = np.where(sigma_33 >= 0.0, sigma_33 / Yt_s, 0.0)
        mt = np.maximum(mt_22, mt_33)
        mc_22 = np.where(sigma_22 < 0.0, -sigma_22 / Yc_s, 0.0)
        mc_33 = np.where(sigma_33 < 0.0, -sigma_33 / Yc_s, 0.0)
        mc = np.maximum(mc_22, mc_33)
        # Shear: worst of all three engineering shear components against the
        # appropriate allowable (S_23 for the 23 plane, S_12 for 12 / 13).
        shear = np.maximum.reduce([
            np.abs(tau_12) / S12_s,
            np.abs(tau_13) / S12_s,
            np.abs(tau_23) / S23_s,
        ])

        max_fi = np.maximum.reduce([ft, fc, mt, mc, shear])
        return {
            'max_fi': max_fi,
            'fiber_t': ft,
            'fiber_c': fc,
            'matrix_t': mt,
            'matrix_c': mc,
            'shear': shear,
        }

    @staticmethod
    def export_results(field_results: 'FieldResults', filename: str,
                       fmt: str = 'json',
                       mesh: Optional['CompositeMesh'] = None,
                       include_raw: bool = False) -> None:
        """Export FE results to a JSON summary or a VTK field file.

        With ``fmt='json'`` (the default, unchanged legacy behavior) this
        saves displacement statistics, stress/strain summaries, failure data,
        and knockdown factor; large arrays are summarized (min/max/mean/std)
        rather than stored in full.

        With ``fmt='vtk'`` it delegates to :meth:`FieldResults.to_vtk` and
        writes the full hex mesh plus per-element fields as a legacy ASCII
        ``UNSTRUCTURED_GRID`` for ParaView / VisIt / PyVista. The richer
        per-element/per-node API lives on ``FieldResults.to_vtk`` directly;
        this ``fmt='vtk'`` path is a convenience shim for callers that
        already hold an ``FESolver``.

        Parameters
        ----------
        field_results : FieldResults
            Results from FESolver.solve().
        filename : str
            Output file path (``.json`` or ``.vtk``).
        fmt : str
            ``'json'`` (default) or ``'vtk'``.
        mesh : CompositeMesh, optional
            Required when ``fmt='vtk'`` (supplies geometry/connectivity).
        include_raw : bool
            When ``True`` (and ``fmt='json'``), also write a sidecar
            ``<filename>.npz`` containing the raw displacement/stress/strain
            arrays so a full audit can re-derive the per-key summary
            statistics. Default ``False`` so existing outputs are not
            bloated (#55).
        """
        fmt = str(fmt).lower()
        if fmt == 'vtk':
            if mesh is None:
                raise ValueError(
                    "export_results(fmt='vtk') requires the `mesh` argument "
                    "(pass the CompositeMesh used by the solver)."
                )
            field_results.to_vtk(mesh, filename)
            return
        if fmt != 'json':
            raise ValueError(
                f"Unknown export format {fmt!r}. Use 'json' or 'vtk'."
            )

        def _array_stats(arr: np.ndarray) -> dict:
            """Compute summary statistics for an array."""
            return {
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
            }

        results_data = {
            'displacement': {
                'n_nodes': int(field_results.displacement.shape[0]),
                'ux': _array_stats(field_results.displacement[:, 0]),
                'uy': _array_stats(field_results.displacement[:, 1]),
                'uz': _array_stats(field_results.displacement[:, 2]),
            },
            'stress_global': {
                'n_elements': int(field_results.stress_global.shape[0]),
                'n_gauss_points': int(field_results.stress_global.shape[1]),
                'sigma_11': _array_stats(field_results.stress_global[:, :, 0]),
                'sigma_22': _array_stats(field_results.stress_global[:, :, 1]),
                'sigma_33': _array_stats(field_results.stress_global[:, :, 2]),
                'tau_23': _array_stats(field_results.stress_global[:, :, 3]),
                'tau_13': _array_stats(field_results.stress_global[:, :, 4]),
                'tau_12': _array_stats(field_results.stress_global[:, :, 5]),
            },
            'stress_local': {
                'sigma_11': _array_stats(field_results.stress_local[:, :, 0]),
                'sigma_22': _array_stats(field_results.stress_local[:, :, 1]),
                'tau_12': _array_stats(field_results.stress_local[:, :, 5]),
            },
            'strain_global': {
                'eps_11': _array_stats(field_results.strain_global[:, :, 0]),
                'eps_22': _array_stats(field_results.strain_global[:, :, 1]),
                'gamma_12': _array_stats(field_results.strain_global[:, :, 5]),
            },
            'failure': {
                'max_tsai_wu_index': float(field_results.max_failure_index),
                'max_failure_index': float(field_results.max_failure_index),
                'criterion': str(getattr(field_results,
                                          'failure_criterion', 'tsai_wu')),
                'mode_indices': (
                    {k: float(v) for k, v in field_results.failure_mode_indices.items()}
                    if field_results.failure_mode_indices is not None
                    else None
                ),
                'knockdown_factor': float(field_results.knockdown),
            },
        }

        output = {
            'schema_version': JSON_SCHEMA_VERSION,
            'format': FORMAT_FE_FIELDS,
            'provenance': _build_provenance(),
            **results_data,
        }
        if include_raw:
            # Sidecar file path lives next to the JSON so users see them
            # together; ``np.savez`` will append ``.npz`` if missing.
            npz_path = f"{filename}.npz"
            arrays = {
                'displacement': np.asarray(field_results.displacement),
                'stress_global': np.asarray(field_results.stress_global),
                'stress_local': np.asarray(field_results.stress_local),
                'strain_global': np.asarray(field_results.strain_global),
                'strain_local': np.asarray(field_results.strain_local),
            }
            if field_results.per_element_failure_index is not None:
                arrays['per_element_failure_index'] = np.asarray(
                    field_results.per_element_failure_index)
            np.savez(npz_path, **arrays)
            output['raw_sidecar'] = os.path.basename(npz_path)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=_json_default)
        logger.info("Saved FE results: %s", filename)


# ============================================================
# SECTION 8: ANALYSIS PIPELINE
# ============================================================

def _analyze_one(Vp: float,
                 name: str,
                 config: Dict,
                 material_name: str,
                 applied_stress: float,
                 seed: Optional[int] = None) -> Tuple[float, str, Dict]:
    """Build PorosityField/CompositeMesh/EmpiricalSolver for one (Vp, config).

    Top-level (picklable) helper so this can be dispatched to a
    :class:`concurrent.futures.ProcessPoolExecutor` from
    :func:`compare_configurations` for a ~Nx speedup on the 5 x 5 sweep
    (#52). Each call is fully independent — no shared mutable state — so
    the result of the parallel execution is order-invariant.

    The returned dict carries both the live solver objects (mesh /
    empirical_solver / porosity_field) and the headline empirical
    knockdown table; the public-facing :func:`compare_configurations`
    splits this into :class:`ConfigResult` / :class:`ConfigArtifacts`
    (#44 item 3). Keeping the worker dict intact preserves the
    parallel-sweep pickle contract (#52).

    Parameters
    ----------
    Vp : float
        Void volume fraction in [0, 1].
    name : str
        Configuration name (key in ``POROSITY_CONFIGS``).
    config : dict
        Porosity-field constructor kwargs.
    material_name : str
        Material preset name. Resolved inside the worker so the parent
        process doesn't need to pickle the :class:`MaterialProperties`
        dataclass across the boundary (it's keyed by name anyway).
    applied_stress : float
        Reserved for downstream solver hooks. Accepted for parity with the
        ``compare_configurations`` signature even though the empirical
        knockdown does not currently consume it.
    seed : int, optional
        Recorded into the porosity field for reproducibility provenance.

    Returns
    -------
    (Vp, name, result_dict)
        Tuple keyed on ``(Vp, name)`` so the caller can deterministically
        re-assemble results even when the worker pool reorders completion.
    """
    material = MATERIALS[material_name]
    porosity_field = PorosityField(material, Vp, seed=seed, **config)
    mesh = CompositeMesh(porosity_field, material, nx=30, ny=10, nz=12)
    empirical = EmpiricalSolver(mesh, material)
    emp_results = empirical.get_all_failure_loads()

    result = {
        'config': config,
        'mesh': mesh,
        'porosity_field': porosity_field,
        'empirical_solver': empirical,
        'empirical': emp_results,
    }
    return (Vp, name, result)


def _build_config_result(name: str, Vp: float, raw: Dict) -> 'ConfigResult':
    """Distill the worker-dict shape into a lightweight :class:`ConfigResult`.

    Reads the headline compression / Judd-Wright knockdown from the inner
    ``empirical`` table so the convenience scalars on the result match
    what the existing rankings code prints (#44 item 3). The nested
    ``empirical`` dict is carried verbatim so existing callers (JSON
    exporter, plot helpers, tests reading
    ``cfg['empirical']['compression'][model]['knockdown']``) keep working.
    """
    emp = raw['empirical']
    headline = emp['compression']['judd_wright']
    # Carry the seed off the PorosityField so the JSON exporter's
    # provenance block can recover it without holding the live field.
    pf = raw.get('porosity_field')
    seed_val = getattr(pf, 'seed', None) if pf is not None else None
    # Headline is now a FailureResult; the dict-style shim keeps the
    # legacy ``['failure_stress']`` access working too.
    return ConfigResult(
        Vp=float(Vp),
        config_name=str(name),
        config=raw['config'],
        failure_stress=float(headline['failure_stress']),
        knockdown=float(headline['knockdown']),
        model=str(headline['model']),
        empirical=emp,
        seed=seed_val,
    )


def _build_config_artifacts(raw: Dict) -> 'ConfigArtifacts':
    """Bundle the live worker objects into a :class:`ConfigArtifacts`."""
    return ConfigArtifacts(
        mesh=raw['mesh'],
        empirical_solver=raw['empirical_solver'],
        porosity_field=raw['porosity_field'],
        field_results=raw.get('field_results'),
    )


def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Normalise ``n_jobs`` to a positive worker count.

    ``None``/``0``/``-1`` map to ``os.cpu_count() or 1`` so callers can
    request "all cores" without having to look up the count themselves.
    ``1`` preserves the serial path for reproducibility / debugging.
    """
    if n_jobs is None or n_jobs <= 0:
        return os.cpu_count() or 1
    return int(n_jobs)


def compare_configurations(void_volume_fraction: float,
                           material_name: str = 'T800_epoxy',
                           applied_stress: float = -1500.0,
                           configs: Optional[Dict] = None,
                           seed: Optional[int] = None,
                           n_jobs: int = 1,
                           return_artifacts: bool = False):
    """Main analysis function — loops through porosity configurations.

    Parameters
    ----------
    void_volume_fraction : float
        Specimen-average void volume fraction in [0, 1].
    material_name : str
        Material preset name; validated against :data:`MATERIALS`.
    applied_stress : float
        Reserved for downstream solver hooks (empirical knockdown does
        not currently consume it).
    configs : dict, optional
        Mapping of configuration name -> :class:`PorosityField` kwargs.
        Defaults to the bundled :data:`POROSITY_CONFIGS`.
    seed : int, optional
        Recorded into provenance and threaded into each
        :class:`PorosityField` for reproducibility (#55). The pipeline is
        deterministic, so this does not alter results today.
    n_jobs : int, optional
        Number of worker processes to use for the per-configuration sweep
        (#52). ``1`` (default) runs serially — bit-for-bit identical to
        the legacy behaviour, useful for tests / debugging. ``N > 1``
        dispatches the (Vp, config) calls to a
        :class:`concurrent.futures.ProcessPoolExecutor` of that size.
        ``0`` / ``-1`` / ``None`` resolve to :func:`os.cpu_count`. Results
        are deterministically re-assembled by ``(Vp, name)`` regardless
        of completion order, so the returned dict is independent of ``N``.
    return_artifacts : bool, optional
        If ``False`` (default), returns ``Dict[str, ConfigResult]`` —
        numbers only, JSON-friendly, safe to retain in long batch loops
        (#44 item 3). If ``True``, returns a tuple
        ``(Dict[str, ConfigResult], Dict[str, ConfigArtifacts])`` so
        callers that need the live ``mesh`` / ``empirical_solver`` /
        ``porosity_field`` objects (plot helpers, the GUI, the
        ``--plots`` CLI path) can still get them. Existing callers that
        accessed ``results[name]['mesh']`` need to switch to the
        artifacts dict; the legacy keys now raise :class:`KeyError` with
        a hint pointing to ``return_artifacts=True``.

    Returns
    -------
    Dict[str, ConfigResult]
        When ``return_artifacts=False`` (default).
    Tuple[Dict[str, ConfigResult], Dict[str, ConfigArtifacts]]
        When ``return_artifacts=True``.
    """
    if material_name not in MATERIALS:
        raise ValueError(
            f"Unknown material {material_name!r}. "
            f"Available presets: {sorted(MATERIALS)}."
        )
    configs = configs or POROSITY_CONFIGS
    workers = _resolve_n_jobs(n_jobs)

    _bar = '=' * 70
    logger.info("\n%s", _bar)
    logger.info("POROSITY ANALYSIS: Vp = %.1f%%", void_volume_fraction * 100)
    logger.info("Material: %s", material_name)
    logger.info("%s", _bar)

    # Build the (Vp, name, config, ...) task list once. We always iterate
    # the original ``configs`` dict so the assembled output preserves the
    # caller's configuration ordering (Python dicts are insertion-ordered)
    # regardless of which worker finishes first.
    tasks = [
        (void_volume_fraction, name, config, material_name, applied_stress, seed)
        for name, config in configs.items()
    ]

    raw_results: Dict[Tuple[float, str], Dict] = {}
    if workers == 1 or len(tasks) <= 1:
        # Serial path — preserves the legacy behaviour byte-for-byte and
        # avoids the ProcessPoolExecutor fork cost for trivially small
        # sweeps. The per-config "Configuration: ..." log lines fire here
        # too, mirroring the original CLI UX.
        for Vp, name, config, mat, stress, sd in tasks:
            logger.info("\n  Configuration: %s", name)
            Vp_out, name_out, result = _analyze_one(
                Vp, name, config, mat, stress, sd)
            raw_results[(Vp_out, name_out)] = result
            comp_kd = result['empirical']['compression']['judd_wright']['knockdown']
            ilss_kd = result['empirical']['ilss']['judd_wright']['knockdown']
            logger.info("    Compression KD (J-W): %.3f", comp_kd)
            logger.info("    ILSS KD (J-W):        %.3f", ilss_kd)
            # Issue #65: surface the closed-form local sensitivities at
            # the same Vp_mean used for the headline KD. This is a free
            # diagnostic — the partials are analytic.
            s = result['empirical_solver'].local_sensitivities(
                mode='compression', model='judd_wright')
            logger.info(
                "    Tornado [%s]: dKD/dVp=%.3g, dKD/dcoef=%.3g",
                name_out, s['dKD_dVp'], s['dKD_dcoef'])
    else:
        logger.info("Parallel sweep: %d task(s) across %d worker process(es)",
                    len(tasks), workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_analyze_one, *task) for task in tasks]
            for fut in concurrent.futures.as_completed(futures):
                Vp_out, name_out, result = fut.result()
                raw_results[(Vp_out, name_out)] = result
                comp_kd = result['empirical']['compression']['judd_wright']['knockdown']
                ilss_kd = result['empirical']['ilss']['judd_wright']['knockdown']
                logger.info("  Configuration %s done — "
                            "compression KD (J-W) %.3f, ILSS KD (J-W) %.3f",
                            name_out, comp_kd, ilss_kd)
                s = result['empirical_solver'].local_sensitivities(
                    mode='compression', model='judd_wright')
                logger.info(
                    "    Tornado [%s]: dKD/dVp=%.3g, dKD/dcoef=%.3g",
                    name_out, s['dKD_dVp'], s['dKD_dcoef'])

    # Re-assemble in the original config insertion order so callers see a
    # deterministic dict regardless of which worker finished first.
    # Split the worker dict into the public-facing lightweight
    # ConfigResult (numbers + nested empirical table) and the parallel
    # ConfigArtifacts (live mesh / solver / field), per #44 item 3.
    results: Dict[str, ConfigResult] = {}
    artifacts: Dict[str, ConfigArtifacts] = {}
    for name in configs:
        raw = raw_results[(void_volume_fraction, name)]
        results[name] = _build_config_result(name, void_volume_fraction, raw)
        artifacts[name] = _build_config_artifacts(raw)

    logger.info("\n%s", _bar)
    logger.info("RANKINGS (by compression strength, Judd-Wright)")
    logger.info("%s", _bar)
    ranked = sorted(
        results.keys(),
        key=lambda c: results[c].failure_stress,
        reverse=True,
    )
    for i, name in enumerate(ranked, 1):
        logger.info("  %d. %s: %.1f MPa", i, name, results[name].failure_stress)

    if return_artifacts:
        return results, artifacts
    return results


# JSON output schema (#20). Bump the major when an incompatible change
# to the payload structure ships; bump the minor for additive changes.
JSON_SCHEMA_VERSION = "1.0"
FORMAT_EMPIRICAL_SWEEP = "porosity-fe.empirical-sweep"
FORMAT_FE_FIELDS = "porosity-fe.fe-fields"
FORMAT_NCR = "porosity-fe.ncr"
_KNOWN_FORMATS = {FORMAT_EMPIRICAL_SWEEP, FORMAT_FE_FIELDS, FORMAT_NCR}


def _build_provenance(seed: Optional[int] = None) -> dict:
    """Return a provenance metadata dict for JSON output reproducibility.

    Captures software versions, platform, timestamp, optional git commit,
    and the run ``seed`` so that any JSON output can be traced back to the
    exact environment used (#55).

    Field names use two parallel conventions for back-compat: the original
    ``*_version`` / ``timestamp_utc`` / ``git_commit`` keys plus the shorter
    ``python`` / ``numpy`` / ``scipy`` / ``git_sha`` / ``generated_utc`` /
    ``package_version`` aliases from the #55 reproducibility contract.

    The optional ``hostname`` field is opt-in via the
    ``POROSITY_FE_INCLUDE_HOSTNAME`` env var (set to ``1``/``true``/``yes``)
    so the default JSON output does not leak workstation names.
    """
    try:
        import importlib.metadata as _ilm
        pfe_version: Optional[str] = _ilm.version("porosity-fe")
    except Exception:
        # Source checkout not pip-installed: use the importable module
        # attribute (defined at the top of this file).
        pfe_version = __version__

    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"

    def _pkg_version(module_name: str) -> Optional[str]:
        mod = sys.modules.get(module_name)
        return getattr(mod, "__version__", None) if mod else None

    try:
        # Run git from the directory containing this module so a CLI invoked
        # from somewhere else still resolves the repo SHA. Graceful fallback
        # to ``None`` for wheel/sdist installs or untracked checkouts.
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        git_commit: Optional[str] = result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.CalledProcessError, FileNotFoundError, Exception):
        git_commit = None

    numpy_v = _pkg_version("numpy")
    scipy_v = _pkg_version("scipy")
    generated_utc = datetime.datetime.utcnow().isoformat() + "Z"

    prov = {
        # Envelope schema version, repeated inside the provenance block so a
        # consumer holding just the provenance dict can still tell what
        # contract it was emitted under (#55).
        "schema_version": JSON_SCHEMA_VERSION,
        # Existing keys (kept for back-compat with the published JSON schema
        # and downstream consumers).
        "porosity_fe_version": pfe_version,
        "python_version": python_version,
        "platform": platform.platform(),
        "numpy_version": numpy_v,
        "scipy_version": scipy_v,
        "matplotlib_version": _pkg_version("matplotlib"),
        "timestamp_utc": generated_utc,
        "seed": seed,
        "git_commit": git_commit,
        # #55 aliases (short names from the reproducibility contract).
        "package_version": pfe_version,
        "python": python_version,
        "numpy": numpy_v,
        "scipy": scipy_v,
        "generated_utc": generated_utc,
        "git_sha": git_commit,
    }

    # Hostname is opt-in to avoid leaking workstation names in shared
    # artifacts. Default off (#55).
    if os.environ.get("POROSITY_FE_INCLUDE_HOSTNAME", "").lower() in (
            "1", "true", "yes", "on"):
        try:
            prov["hostname"] = platform.node() or None
        except Exception:
            prov["hostname"] = None

    return prov


def save_results_to_json(results: Dict, filename: str,
                         artifacts: Optional[Dict[str, 'ConfigArtifacts']] = None):
    """Export numerical results to JSON.

    Parameters
    ----------
    results : dict
        Either the new ``Dict[str, ConfigResult]`` returned by
        :func:`compare_configurations`, or the legacy worker-dict shape
        (``Dict[str, dict]``). Both keep the same on-disk JSON shape so
        the published JSON schema is unchanged (#44 item 3).
    filename : str
        Output JSON path.
    artifacts : dict, optional
        Parallel ``Dict[str, ConfigArtifacts]`` from
        ``compare_configurations(..., return_artifacts=True)``. Used
        only to recover the seed for the provenance block when
        ``results`` is the lightweight ``ConfigResult`` shape (no live
        ``porosity_field`` carried). Optional; the JSON is still written
        if absent, just with ``seed=None`` in provenance.
    """
    # All configs in a sweep share one seed; record it iff unambiguous.
    # Source the seed from (in priority order):
    #   1. ConfigResult.seed (the new lightweight path),
    #   2. the parallel artifacts dict's porosity_field.seed, or
    #   3. the legacy worker-dict shape (back-compat).
    # (#44 item 3 / #55).
    seeds: set = set()
    for entry in results.values():
        if isinstance(entry, ConfigResult):
            if entry.seed is not None or artifacts is None:
                seeds.add(entry.seed)
            else:
                art = artifacts.get(entry.config_name) if artifacts else None
                pf = getattr(art, 'porosity_field', None) if art is not None else None
                seeds.add(getattr(pf, 'seed', None))
        elif isinstance(entry, dict):
            pf = entry.get('porosity_field')
            if pf is not None:
                seeds.add(getattr(pf, 'seed', None))
    seed = seeds.pop() if len(seeds) == 1 else None

    output = {
        'schema_version': JSON_SCHEMA_VERSION,
        'format': FORMAT_EMPIRICAL_SWEEP,
        'provenance': _build_provenance(seed=seed),
    }
    for name, data in results.items():
        if name in ('schema_version', 'format'):
            # Defensive: a user-named config that collides with envelope
            # keys would silently overwrite them. Skip with a clear error.
            raise ValueError(
                f"Configuration name {name!r} collides with the JSON "
                f"envelope keys ('schema_version', 'format')."
            )
        # Resolve the void_volume_fraction and config dict for both the
        # legacy dict shape and the new ConfigResult shape. The legacy
        # path reads ``data['porosity_field'].Vp`` and ``data['config']``;
        # the new path reads ``data.Vp`` / ``data.config``.
        if isinstance(data, ConfigResult):
            vp_value = float(data.Vp)
            cfg_dict = data.config
            emp_table = data.empirical
        else:
            vp_value = float(data['porosity_field'].Vp)
            cfg_dict = data['config']
            emp_table = data['empirical']

        entry = {
            'config': cfg_dict,
            'void_volume_fraction': vp_value,
            'empirical': {},
        }
        for mode in emp_table:
            entry['empirical'][mode] = {}
            for model in emp_table[mode]:
                r = emp_table[mode][model]
                # ``r`` is now a FailureResult (with dict-style back-compat
                # shim) for the empirical path, but legacy callers may
                # still hand in raw dicts.
                entry['empirical'][mode][model] = {
                    'failure_stress_MPa': r['failure_stress'],
                    'knockdown': r['knockdown'],
                }
        output[name] = entry

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("Saved: %s", filename)


def load_results_from_json(filename: str) -> Dict:
    """Round-trip loader for save_results_to_json / export_results outputs.

    Validates schema_version compatibility and format identifier. Raises
    ValueError on missing or incompatible envelope so callers don't silently
    consume the wrong shape.
    """
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{filename}: expected a JSON object at the top level.")
    sv = data.get('schema_version')
    if sv is None:
        raise ValueError(
            f"{filename}: missing 'schema_version'. "
            f"This file was likely written by a pre-1.0 build of porosity-fe."
        )
    major = sv.split('.', 1)[0]
    expected_major = JSON_SCHEMA_VERSION.split('.', 1)[0]
    if major != expected_major:
        raise ValueError(
            f"{filename}: schema_version {sv} is incompatible with this "
            f"loader (expects {expected_major}.x)."
        )
    fmt = data.get('format')
    if fmt not in _KNOWN_FORMATS:
        raise ValueError(
            f"{filename}: unknown format {fmt!r}. "
            f"Known formats: {sorted(_KNOWN_FORMATS)}."
        )
    return data


# Default Vp sweep — preserves the historical hardcoded behavior so that
# `porosity-analyze` with no arguments reproduces today's analysis range.
DEFAULT_POROSITY_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08]


def _vp_label(Vp: float) -> str:
    """Stable, filesystem-safe label for a void fraction.

    Integer-percent fractions keep the legacy ``Npct`` form (e.g. 0.03 ->
    ``3pct``); non-integer fractions fall back to a decimal-derived form
    (e.g. 0.025 -> ``2p5pct``) so distinct sweeps never collide.
    """
    pct = Vp * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}pct"
    return f"{pct:.4f}".rstrip('0').rstrip('.').replace('.', 'p') + "pct"


def _build_arg_parser() -> 'argparse.ArgumentParser':
    """Construct the argparse driver for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        prog="porosity-analyze",
        description=(
            "Run the porosity-degraded composite laminate analysis over one "
            "or more void volume fractions and write JSON results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--material",
        default="T800_epoxy",
        help="Material preset name (validated against the built-in presets).",
    )
    parser.add_argument(
        "--vp",
        type=float,
        nargs="+",
        default=list(DEFAULT_POROSITY_LEVELS),
        metavar="VP",
        help=(
            "One or more void volume fractions in [0, 1]. Defaults to the "
            "historical sweep."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write JSON results into (created if missing).",
    )
    parser.add_argument(
        "--applied-stress",
        type=float,
        default=-1500.0,
        help="Applied stress (MPa) passed to compare_configurations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Recorded in JSON provenance for reproducibility. The pipeline "
            "is deterministic (RNG-free), so this does not alter results."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Also render the heavy matplotlib figures (PNG). Off by default "
            "to keep CI / batch runs fast."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress progress output; only warnings and errors are shown. "
            "Mutually exclusive with --verbose."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Show debug-level progress output in addition to the default "
            "INFO progress. Mutually exclusive with --quiet."
        ),
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="List the available material presets and exit.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of worker processes for the per-configuration sweep "
            "in compare_configurations (#52). 1 (default) runs serially "
            "(deterministic, byte-identical to legacy behaviour); N>1 "
            "parallelises the (Vp, config) calls across N processes; 0 or "
            "-1 uses os.cpu_count(). Results are deterministically "
            "re-assembled regardless of N."
        ),
    )
    return parser


class _DynamicStdoutHandler(logging.StreamHandler):
    """StreamHandler that resolves ``sys.stdout`` on every emit.

    Plain ``StreamHandler(sys.stdout)`` caches the stream reference at
    construction time, which breaks pytest's ``capsys`` fixture (it
    rebinds ``sys.stdout`` per-test). Looking it up lazily keeps the
    formatting identical to the old bare-``print`` output while
    remaining capturable.
    """

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value):
        # logging.StreamHandler.__init__ assigns ``self.stream`` -- accept
        # and ignore so the dynamic property above wins.
        pass


def _configure_cli_logging(*, quiet: bool, verbose: bool) -> None:
    """Wire the module ``logger`` for CLI use.

    Routes progress through the module logger so ``--quiet`` actually
    silences the run (issue #78). Attaches a stdout stream handler the
    first time the CLI is invoked so the output looks like the old
    bare-``print`` style.

    Parameters
    ----------
    quiet : bool
        Suppress INFO/DEBUG; only warnings and errors are surfaced.
    verbose : bool
        Lower the threshold to DEBUG.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Don't propagate to the root logger -- pytest's caplog and library
    # consumers (the Streamlit app) configure their own handlers and we
    # don't want duplicated lines.
    logger.propagate = False
    logger.setLevel(level)

    # Reuse any handler we already attached; otherwise add a simple
    # stdout stream so the formatting matches the old print()-based UX.
    if not any(getattr(h, "_porosity_cli", False) for h in logger.handlers):
        handler = _DynamicStdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._porosity_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def main(argv: Optional[List[str]] = None) -> int:
    """Argparse-driven entry point.

    Returns
    -------
    int
        ``0`` on success, ``2`` on bad input, ``3`` on a solver failure.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_materials:
        for name in sorted(MATERIALS):
            print(name)
        return 0

    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose are mutually exclusive.")

    _configure_cli_logging(quiet=args.quiet, verbose=args.verbose)

    if args.material not in MATERIALS:
        parser.error(
            f"Unknown material {args.material!r}. "
            f"Available presets: {sorted(MATERIALS)}."
        )

    for Vp in args.vp:
        if not (0.0 <= Vp <= 1.0) or Vp != Vp:  # NaN-safe range check
            parser.error(
                f"--vp value {Vp!r} is out of range; expected a finite "
                f"float in [0, 1] (a void *fraction*, not a percentage)."
            )

    output_dir = args.output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {output_dir!r}: {exc}",
              file=sys.stderr)
        return 2

    # Back-compat shim: a few callers may still reference _log. Now that
    # progress is routed through ``logger``, this just forwards to
    # logger.info so --quiet (WARNING level) silences these too.
    def _log(msg: str) -> None:
        logger.info("%s", msg)

    all_results = {}
    for Vp in args.vp:
        Vp_label = _vp_label(Vp)
        try:
            # ``return_artifacts=True`` because the --plots path needs the
            # live mesh / empirical_solver / porosity_field objects for
            # the FEVisualizer calls below (#44 item 3 migration).
            results, artifacts = compare_configurations(
                Vp,
                material_name=args.material,
                applied_stress=args.applied_stress,
                seed=args.seed,
                n_jobs=args.jobs,
                return_artifacts=True,
            )
        except ValueError as exc:
            print(f"ERROR: bad input for Vp={Vp}: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - surface as solver failure
            print(f"ERROR: solver failure for Vp={Vp}: {exc}", file=sys.stderr)
            return 3
        all_results[Vp_label] = results

        if args.plots:
            for name in results:
                art = artifacts[name]
                FEVisualizer.plot_porosity_field(
                    art.porosity_field,
                    save_path=os.path.join(
                        output_dir, f"porosity_profile_{name}_{Vp_label}.png"))
                FEVisualizer.plot_mesh_3d(
                    art.mesh,
                    save_path=os.path.join(
                        output_dir, f"porosity_mesh_3d_{name}_{Vp_label}.png"))
                FEVisualizer.plot_mesh_detail(
                    art.mesh,
                    save_path=os.path.join(
                        output_dir, f"porosity_mesh_detail_{name}_{Vp_label}.png"))
                FEVisualizer.plot_damage_contour(
                    art.mesh,
                    art.empirical_solver,
                    save_path=os.path.join(
                        output_dir, f"porosity_damage_{name}_{Vp_label}.png"))
            FEVisualizer.plot_model_comparison(
                results,
                save_path=os.path.join(
                    output_dir, f"porosity_comparison_{Vp_label}.png"))

        out_path = os.path.join(
            output_dir, f"porosity_analysis_results_{Vp_label}.json")
        save_results_to_json(results, out_path, artifacts=artifacts)

    if args.plots and all_results:
        FEVisualizer.plot_knockdown_curves(
            all_results,
            save_path=os.path.join(output_dir, "porosity_knockdown_curves.png"))

    _bar = "=" * 70
    logger.info("\n%s", _bar)
    logger.info("COMPLETE ANALYSIS FINISHED")
    logger.info("%s", _bar)
    logger.info("Material: %s", args.material)
    logger.info("Porosity levels analyzed: %s",
                [f"{v*100:.2f}%" for v in args.vp])
    logger.info("Configurations: %s", list(POROSITY_CONFIGS.keys()))
    logger.info("Output directory: %s", os.path.abspath(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
