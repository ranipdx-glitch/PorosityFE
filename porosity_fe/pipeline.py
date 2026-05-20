"""Sweep orchestrator: ``_analyze_one`` worker + ``compare_configurations``."""

import concurrent.futures
import logging
import os
from typing import Dict, Optional, Tuple

from .empirical import EmpiricalSolver
from .materials import MATERIALS
from .mesh import CompositeMesh
from .porosity_field import POROSITY_CONFIGS, PorosityField
from .results import ConfigArtifacts, ConfigResult

logger = logging.getLogger("porosity_fe_analysis")

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
