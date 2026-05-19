#!/usr/bin/env python3
"""Master validation runner: loads all datasets and runs model predictions."""

import json
import logging
import os
import sys
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Append (not insert(0, ...)) so a file dropped next to this script can't
# shadow a stdlib / site-packages module of the same name on import (#29).
# pip-installing the package makes this unnecessary; it only matters for
# source-layout / direct `python validation/validate_all.py` runs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import jsonschema


class ValidationError(Exception):
    """Raised when a dataset fails schema validation."""
    pass


_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'schemas', 'validation_dataset_schema.json')
_SCHEMA = None


def _get_schema() -> Dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        try:
            with open(_SCHEMA_PATH, encoding='utf-8') as f:
                _SCHEMA = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.exception("Failed to load validation schema")
            raise ValueError(
                f"Failed to load validation schema from {_SCHEMA_PATH!r}: "
                f"{type(e).__name__}: {e}"
            ) from e
    return _SCHEMA


def load_dataset(path: str) -> Dict[str, Any]:
    """Load and validate a validation dataset JSON file.

    Raises ValidationError if the file fails schema validation.
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    try:
        jsonschema.validate(instance=data, schema=_get_schema())
    except jsonschema.ValidationError as e:
        raise ValidationError(f"Dataset {path} failed schema: {e.message}") from e
    return data


import dataclasses
from porosity_fe_analysis import MATERIALS, MaterialProperties


# Mapping from a dataset's (fiber, matrix) pair to a named preset in
# ``porosity_fe_analysis.MATERIALS``.  Every dataset shipped under
# ``validation/datasets/`` MUST have an entry here; ``resolve_material`` now
# raises ``KeyError`` on a missing key rather than silently defaulting to
# ``T700_epoxy`` (issue #34).  When adding a new dataset, add the matching
# preset to ``MATERIALS`` first, then map it here.
#
# Generic "Carbon / epoxy" labels (Almeida 1994, Wang 2022) are mapped to
# ``T700_epoxy`` only because the source papers do not name the fibre or
# matrix system explicitly; this is the closest defensible generic carbon
# preset.  Mapping AS4/3501-6 to the dedicated ``AS4_3501_6_epoxy`` preset
# (rather than ``T700_epoxy``) corrects a systematic IM-class vs.
# standard-modulus mismatch flagged in issue #34.
_FIBER_MATRIX_TO_PRESET = {
    ('T700', 'TDE85 epoxy'): 'T700_epoxy',
    ('T700GC-12K-31E', '#2510 epoxy'): 'T700_epoxy',
    ('T700', 'epoxy'): 'T700_epoxy',
    ('HTA 24k', 'EHkF 420 epoxy'): 'HTA_EHkF420_epoxy',
    ('IM7', '8551-7 epoxy'): 'IM7_8551_epoxy',
    ('T300', '924 epoxy'): 'T300_934_epoxy',
    ('T300', '976 epoxy'): 'T300_934_epoxy',
    ('T300', '934 epoxy'): 'T300_934_epoxy',
    ('T300', '914 epoxy'): 'T300_934_epoxy',
    ('Carbon fiber (PEEK-CF60)', 'PEEK (thermoplastic)'): 'CF_PEEK',
    ('AS4', '3501-6 epoxy'): 'AS4_3501_6_epoxy',
    ('AS4 fabric', '3501-6 epoxy'): 'AS4_3501_6_epoxy',
    # Generic-carbon datasets where the source paper does not name the
    # fibre/matrix system. T700_epoxy is the closest defensible match
    # for a standard-modulus carbon/epoxy.
    ('Carbon', 'epoxy'): 'T700_epoxy',
    ('Carbon fiber', 'epoxy'): 'T700_epoxy',
}


def resolve_material(dataset: Dict[str, Any], strict: bool = True) -> MaterialProperties:
    """Build a MaterialProperties instance from a dataset's material block.

    Selects the closest preset from ``MATERIALS`` based on fiber/matrix, then
    overrides ``n_plies`` and ``fiber_volume_fraction`` from the dataset.

    Behaviour as of issue #34: a missing (fiber, matrix) entry in
    ``_FIBER_MATRIX_TO_PRESET`` is now always a hard ``KeyError`` — the prior
    silent fallback to ``'T700_epoxy'`` masked at least three material
    mismatches (AS4/3501-6, HTA/EHkF420, generic Carbon/epoxy) that
    systematically inflated the reported MAE on Ghiorse_1993 and similar
    datasets.  Callers wanting different physics should pass an explicit
    ``MaterialProperties`` instance, or add a preset+mapping entry.

    Parameters
    ----------
    dataset:
        A validated dataset dict containing a ``'material'`` block.
    strict:
        Retained for backward compatibility.  As of #34 the function raises
        ``KeyError`` on an unknown ``(fiber, matrix)`` pair regardless of
        this flag; ``strict=False`` is therefore a no-op and is kept only so
        existing call sites do not break.

    Raises
    ------
    KeyError
        When the ``(fiber, matrix)`` pair is not present in
        ``_FIBER_MATRIX_TO_PRESET``.  The message names the dataset, the
        missing key, and the available presets so the fix is mechanical:
        add a ``MATERIALS`` entry in ``porosity_fe_analysis.py`` and map it
        in ``_FIBER_MATRIX_TO_PRESET``.
    """
    m = dataset['material']
    key = (m['fiber'], m['matrix'])
    if key not in _FIBER_MATRIX_TO_PRESET:
        dataset_name = dataset.get('reference', str(key))
        raise KeyError(
            f"No material preset for (fiber={key[0]!r}, matrix={key[1]!r}) "
            f"in dataset {dataset_name!r}. Add an entry to "
            f"validation.validate_all._FIBER_MATRIX_TO_PRESET, or supply a "
            f"custom MaterialProperties. Available presets: "
            f"{sorted(MATERIALS)}."
        )
    preset_name = _FIBER_MATRIX_TO_PRESET[key]
    base = MATERIALS[preset_name]
    return dataclasses.replace(
        base,
        n_plies=m['n_plies'],
        fiber_volume_fraction=m['fiber_volume_fraction'],
    )


from porosity_fe_analysis import (
    PorosityField, CompositeMesh, EmpiricalSolver,
)


_PROPERTY_TO_MODE = {
    'compression_strength': 'compression',
    'tensile_strength': 'tension',
    # 'transverse_tensile_strength' is intentionally excluded: EmpiricalSolver
    # supports only longitudinal modes ('tension', 'compression', 'shear',
    # 'ilss').  Routing a transverse property to the longitudinal 'tension' mode
    # produces physically incorrect alpha/n coefficients (fiber-dominated vs.
    # matrix-dominated failure).  Until a calibrated 'transverse_tension' mode
    # is added to EmpiricalSolver, transverse_tensile_strength is skipped in
    # MAE calculations.  See GitHub issue #35.
    'shear_strength': 'shear',
    'ilss': 'ilss',
}

# Properties that cannot currently be predicted by EmpiricalSolver due to
# missing calibrated failure modes.  They are skipped with a logged warning
# rather than silently misrouted to the wrong physics.
_UNSUPPORTED_STRENGTH_PROPS = {'transverse_tensile_strength'}


def predict_strength(dataset: Dict[str, Any], prop_key: str,
                     vp_pcts) -> list:
    """Predict normalized strength at each porosity level via Judd-Wright.

    Renormalized to the dataset's baseline_porosity_pct.

    Raises
    ------
    ValueError
        If *prop_key* is in ``_UNSUPPORTED_STRENGTH_PROPS`` (e.g.
        ``'transverse_tensile_strength'``), because EmpiricalSolver has no
        calibrated failure mode for it and silently misrouting it would yield
        incorrect physics.  The caller (``run_all_datasets``) logs a warning and
        records an error entry instead of raising to the user.
    """
    if prop_key in _UNSUPPORTED_STRENGTH_PROPS:
        msg = (
            f"Property '{prop_key}' is not supported by EmpiricalSolver: no "
            "calibrated transverse failure mode exists.  Skipping MAE calculation "
            "to avoid physically incorrect predictions.  See issue #35."
        )
        logger.warning(msg)
        raise ValueError(msg)
    mat = resolve_material(dataset)
    ply_angles = dataset['material']['ply_angles']
    mode = _PROPERTY_TO_MODE[prop_key]
    baseline_vp = dataset.get('baseline_porosity_pct', 0.0) / 100.0

    def _kd(vp_frac):
        pf = PorosityField(mat, vp_frac, distribution='uniform',
                           void_shape='spherical')
        mesh = CompositeMesh(pf, mat, nx=10, ny=5,
                             nz=mat.n_plies, ply_angles=ply_angles)
        emp = EmpiricalSolver(mesh, mat, ply_angles=ply_angles)
        return emp.get_failure_load(mode=mode, model='judd_wright')['knockdown']

    kd_base = _kd(baseline_vp) if baseline_vp > 1e-9 else 1.0
    return [float(_kd(vp / 100.0) / kd_base) for vp in vp_pcts]


from porosity_fe_analysis import (
    compute_degraded_clt_moduli,
    compute_degraded_clt_flexural_modulus,
)


def predict_modulus(dataset: Dict[str, Any], prop_key: str,
                    vp_pcts, method: str = 'mori_tanaka') -> list:
    """Predict normalized modulus at each porosity level via CLT.

    For flexural_modulus, uses D-matrix (bending) formulation.
    Otherwise uses A-matrix (membrane).
    """
    mat = resolve_material(dataset)
    ply_angles = dataset['material']['ply_angles']
    baseline_vp = dataset.get('baseline_porosity_pct', 0.0) / 100.0

    if prop_key == 'flexural_modulus':
        def compute_fn(vp):
            return compute_degraded_clt_flexural_modulus(
                mat, ply_angles, vp, method=method)['Ef_x']
    elif prop_key in ('transverse_tensile_modulus', 'shear_modulus',
                      'tensile_modulus'):
        key_map = {
            'tensile_modulus': 'Ex',
            'transverse_tensile_modulus': 'Ey',
            'shear_modulus': 'Gxy',
        }
        extract = key_map[prop_key]

        def compute_fn(vp):
            return compute_degraded_clt_moduli(
                mat, ply_angles, vp, method=method)[extract]
    else:
        raise ValueError(f"Unknown modulus property: {prop_key}")

    base_val = compute_fn(baseline_vp) if baseline_vp > 1e-9 else compute_fn(0.0)
    return [float(compute_fn(vp / 100.0) / base_val) for vp in vp_pcts]


import numpy as np


def compute_mae(predicted, experimental) -> float:
    """Mean absolute error in percent between predicted and experimental."""
    predicted = np.asarray(predicted, dtype=float)
    experimental = np.asarray(experimental, dtype=float)
    errs = np.abs(predicted - experimental) / np.maximum(np.abs(experimental), 1e-12) * 100.0
    return float(np.mean(errs))


def summarize_mae(results: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate per-(paper, property) MAE numbers into overall summary.

    The property-weighted form gives each (paper, property) entry equal
    weight — this is what the README headline traditionally reports. The
    point-weighted form weights each individual (Vp, normalized) measurement
    equally — the more standard convention in regression-error reporting.
    They typically differ by ~0.5 percentage points because datasets carry
    very different numbers of points. See issue #36.

    Returns a dict with both numbers, the count of entries, and the
    individual extremes.
    """
    per_entry_mae = []
    per_entry_npts = []
    for ds_results in results.values():
        if 'error' in ds_results:
            continue
        for r in ds_results.values():
            if 'mae' in r and 'n_points' in r:
                per_entry_mae.append(r['mae'])
                per_entry_npts.append(r['n_points'])
    if not per_entry_mae:
        return {
            'property_weighted_mae': float('nan'),
            'point_weighted_mae': float('nan'),
            'n_entries': 0,
            'n_points': 0,
            'best_mae': float('nan'),
            'worst_mae': float('nan'),
        }
    mae_arr = np.asarray(per_entry_mae, dtype=float)
    npts_arr = np.asarray(per_entry_npts, dtype=float)
    return {
        'property_weighted_mae': float(np.mean(mae_arr)),
        'point_weighted_mae': float(np.average(mae_arr, weights=npts_arr)),
        'n_entries': int(mae_arr.size),
        'n_points': int(npts_arr.sum()),
        'best_mae': float(mae_arr.min()),
        'worst_mae': float(mae_arr.max()),
    }


import glob
from concurrent.futures import ProcessPoolExecutor

_MODULUS_PROPS = {'tensile_modulus', 'transverse_tensile_modulus',
                  'flexural_modulus', 'shear_modulus'}


def _run_one_dataset(path: str):
    """Compute the per-dataset MAE results for a single dataset JSON file.

    Returns ``(name, dataset_results)`` where ``dataset_results`` has the same
    shape the serial loop produced: either an ``{'error', 'error_type'}`` dict
    on a load failure, or a ``{prop_key: {...}}`` mapping.

    This is a *top-level* function (not a closure) so it is picklable and can
    be dispatched to a ``ProcessPoolExecutor`` worker.  It contains zero
    cross-dataset state, so running it in any order / any process yields
    byte-identical results to the serial path (the pipeline is deterministic
    and RNG-free).
    """
    name = os.path.basename(path).replace('.json', '')
    try:
        data = load_dataset(path)
    except ValidationError as e:
        logger.warning("Skipping dataset %s: %s", name, e)
        return name, {
            'error': str(e),
            'error_type': type(e).__name__,
        }

    dataset_results = {}
    for prop_key, prop_data in data['properties'].items():
        vp = prop_data['void_content_pct']
        exp = prop_data['normalized_values']
        if prop_key in _UNSUPPORTED_STRENGTH_PROPS:
            skip_msg = (
                f"Skipping '{prop_key}' for dataset '{name}': no calibrated "
                "EmpiricalSolver mode available (see issue #35)."
            )
            logger.warning(skip_msg)
            dataset_results[prop_key] = {'skipped': skip_msg}
            continue
        try:
            if prop_key in _MODULUS_PROPS:
                pred = predict_modulus(data, prop_key, vp)
            else:
                pred = predict_strength(data, prop_key, vp)
            mae = compute_mae(pred, exp)
            dataset_results[prop_key] = {
                'vp_pcts': list(vp),
                'experimental': list(exp),
                'predicted': pred,
                'mae': mae,
                'n_points': len(vp),
            }
        except Exception as e:
            # logger.exception() captures the traceback so a downstream
            # debug log (logging level DEBUG/INFO) shows *why* the
            # property prediction failed, rather than just the bare
            # string we surface in the JSON results (#19).
            logger.exception("Prediction failed for %s/%s", name, prop_key)
            dataset_results[prop_key] = {
                'error': str(e),
                'error_type': type(e).__name__,
            }
    return name, dataset_results


def _resolve_n_jobs(n_jobs: int) -> int:
    """Map the public ``n_jobs`` contract onto a concrete worker count.

    ``n_jobs <= 0`` (covers ``-1`` and ``0``) means "use all cores".
    ``os.cpu_count()`` can return ``None`` on exotic platforms; fall back to
    ``1`` (serial) there so we never crash on a missing CPU count.
    """
    if n_jobs <= 0:
        return os.cpu_count() or 1
    return n_jobs


def run_all_datasets(datasets_dir: str = None,
                      n_jobs: int = 1) -> Dict[str, Any]:
    """Run predictions for all datasets, return nested MAE results.

    Parameters
    ----------
    datasets_dir:
        Directory of dataset JSON files (defaults to the bundled
        ``validation/datasets/``).
    n_jobs:
        Process-level parallelism for the per-dataset work.

        * ``1`` (default) keeps the original *serial* code path verbatim —
          zero behaviour change, fully deterministic, back-compatible.
        * ``>1`` distributes per-dataset work across that many worker
          processes via ``concurrent.futures.ProcessPoolExecutor``.
        * ``-1`` or ``0`` uses ``os.cpu_count()`` workers.

        Every ``(dataset)`` task is fully independent and the pipeline is
        RNG-free, so the returned dict is identical (same keys, same values)
        regardless of ``n_jobs``.  Results are always assembled in sorted
        dataset order, so dict iteration order is deterministic too.
    """
    if datasets_dir is None:
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'datasets')

    paths = sorted(glob.glob(os.path.join(datasets_dir, '*.json')))
    workers = _resolve_n_jobs(n_jobs)

    if workers <= 1 or len(paths) <= 1:
        # Serial path: byte-identical to the historical implementation.
        return {name: res for name, res in (_run_one_dataset(p)
                                            for p in paths)}

    # Parallel path: dispatch independent per-dataset tasks across processes,
    # then reassemble in sorted (== input) order so the result dict is
    # deterministic and identical to the serial path.
    by_name = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for name, dataset_results in executor.map(_run_one_dataset, paths):
            by_name[name] = dataset_results
    return {os.path.basename(p).replace('.json', ''):
            by_name[os.path.basename(p).replace('.json', '')]
            for p in paths}


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_master_report(results: Dict[str, Any], output_dir: str = None):
    """Generate master validation report (PNG plot + Markdown table)."""
    if output_dir is None:
        # Default to cwd, not the package directory: when running inside a
        # PyInstaller-frozen bundle (macOS .app / Windows .exe) the package
        # path is read-only and writing the PNG/MD there crashes.
        output_dir = os.getcwd()

    by_property = {}
    for ds_name, ds_results in results.items():
        if 'error' in ds_results:
            continue
        for prop, prop_result in ds_results.items():
            if 'error' in prop_result or 'skipped' in prop_result:
                continue
            by_property.setdefault(prop, []).append({
                'dataset': ds_name,
                'mae': prop_result['mae'],
                'n': prop_result['n_points'],
            })

    fig, ax = plt.subplots(figsize=(14, 8))
    labels = []
    values = []
    colors = []
    for prop, entries in sorted(by_property.items()):
        for e in entries:
            labels.append(f"{e['dataset']}\n{prop}")
            values.append(e['mae'])
            if e['mae'] < 5:
                colors.append('#5cb85c')
            elif e['mae'] < 10:
                colors.append('#f0ad4e')
            else:
                colors.append('#d9534f')

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel('MAE (%)', fontsize=12)
    ax.set_title('Master Validation Report: MAE across all papers/properties',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'validation_master_report.png')
    # dpi=300 to match every other static PNG in the project (#53);
    # bbox/dpi defaults also come from porosity_fe_analysis._apply_plot_style.
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    md_lines = ['# Master Validation Report', '']

    # Overall MAE summary — report both aggregation forms (#36).
    summary = summarize_mae(results)
    if summary['n_entries']:
        md_lines.extend([
            '## Overall MAE',
            '',
            f"- Property-weighted: **{summary['property_weighted_mae']:.2f}%** "
            f"(n={summary['n_entries']} paper-property entries)",
            f"- Point-weighted:    **{summary['point_weighted_mae']:.2f}%** "
            f"(n={summary['n_points']} individual data points)",
            '',
            '_The two aggregations weight datasets differently (entries vs. data points); '
            'point-weighted is the standard convention in regression-error reporting._',
            '',
        ])

    md_lines.extend([
        '## Per-(dataset, property) breakdown',
        '',
        '| Dataset | Property | N points | MAE (%) |',
        '|---|---|---|---|',
    ])
    for ds_name, ds_results in sorted(results.items()):
        if 'error' in ds_results:
            md_lines.append(f"| {ds_name} | LOAD ERROR | - | - |")
            continue
        for prop in sorted(ds_results.keys()):
            r = ds_results[prop]
            if 'skipped' in r:
                md_lines.append(f"| {ds_name} | {prop} | - | SKIPPED (no model) |")
            elif 'error' in r:
                md_lines.append(f"| {ds_name} | {prop} | - | ERROR |")
            else:
                md_lines.append(f"| {ds_name} | {prop} | {r['n_points']} | {r['mae']:.2f} |")

    md_path = os.path.join(output_dir, 'validation_detail_report.md')
    with open(md_path, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(md_lines))

    return plot_path, md_path


if __name__ == "__main__":
    results = run_all_datasets()
    plot, md = generate_master_report(results)
    print(f"Plot: {plot}")
    print(f"Markdown: {md}")
    summary = summarize_mae(results)
    if summary['n_entries']:
        print(
            f"\nOverall MAE (property-weighted): {summary['property_weighted_mae']:.2f}%  "
            f"(n={summary['n_entries']} entries)"
        )
        print(
            f"Overall MAE (point-weighted):    {summary['point_weighted_mae']:.2f}%  "
            f"(n={summary['n_points']} points)"
        )
