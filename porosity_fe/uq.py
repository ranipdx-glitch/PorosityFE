"""Uncertainty propagation (Monte Carlo / LHS) over the empirical solver."""

import contextlib
import io
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .empirical import EmpiricalSolver
from .materials import MATERIALS, MaterialProperties
from .mesh import CompositeMesh
from .porosity_field import PorosityField

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
