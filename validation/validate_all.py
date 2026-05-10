#!/usr/bin/env python3
"""Master validation runner: loads all datasets and runs model predictions."""

import json
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        with open(_SCHEMA_PATH, encoding='utf-8') as f:
            _SCHEMA = json.load(f)
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


_FIBER_MATRIX_TO_PRESET = {
    ('T700', 'TDE85 epoxy'): 'T700_epoxy',
    ('T700GC-12K-31E', '#2510 epoxy'): 'T700_epoxy',
    ('T700', 'epoxy'): 'T700_epoxy',
    ('HTA 24k', 'EHkF 420 epoxy'): 'T700_epoxy',
    ('IM7', '8551-7 epoxy'): 'IM7_8551_epoxy',
    ('T300', '924 epoxy'): 'T300_934_epoxy',
    ('T300', '976 epoxy'): 'T300_934_epoxy',
    ('T300', '934 epoxy'): 'T300_934_epoxy',
    ('T300', '914 epoxy'): 'T300_934_epoxy',
    ('Carbon fiber (PEEK-CF60)', 'PEEK (thermoplastic)'): 'CF_PEEK',
    ('AS4', '3501-6 epoxy'): 'T700_epoxy',
    ('AS4 fabric', '3501-6 epoxy'): 'T700_epoxy',
    ('Carbon', 'epoxy'): 'T700_epoxy',
    ('Carbon fiber', 'epoxy'): 'T700_epoxy',
}


def resolve_material(dataset: Dict[str, Any]) -> MaterialProperties:
    """Build a MaterialProperties instance from a dataset's material block.

    Selects the closest preset from MATERIALS based on fiber/matrix, then
    overrides n_plies and fiber_volume_fraction from the dataset.
    """
    m = dataset['material']
    key = (m['fiber'], m['matrix'])
    preset_name = _FIBER_MATRIX_TO_PRESET.get(key, 'T700_epoxy')
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
    'transverse_tensile_strength': 'tension',
    'shear_strength': 'shear',
    'ilss': 'ilss',
}


def predict_strength(dataset: Dict[str, Any], prop_key: str,
                     vp_pcts) -> list:
    """Predict normalized strength at each porosity level via Judd-Wright.

    Renormalized to the dataset's baseline_porosity_pct.
    """
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


import glob

_MODULUS_PROPS = {'tensile_modulus', 'transverse_tensile_modulus',
                  'flexural_modulus', 'shear_modulus'}


def run_all_datasets(datasets_dir: str = None) -> Dict[str, Any]:
    """Run predictions for all datasets, return nested MAE results."""
    if datasets_dir is None:
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'datasets')

    all_results = {}
    for path in sorted(glob.glob(os.path.join(datasets_dir, '*.json'))):
        name = os.path.basename(path).replace('.json', '')
        try:
            data = load_dataset(path)
        except ValidationError as e:
            all_results[name] = {'error': str(e)}
            continue

        dataset_results = {}
        for prop_key, prop_data in data['properties'].items():
            vp = prop_data['void_content_pct']
            exp = prop_data['normalized_values']
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
                dataset_results[prop_key] = {'error': str(e)}
        all_results[name] = dataset_results
    return all_results


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_master_report(results: Dict[str, Any], output_dir: str = None):
    """Generate master validation report (PNG plot + Markdown table)."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    by_property = {}
    for ds_name, ds_results in results.items():
        if 'error' in ds_results:
            continue
        for prop, prop_result in ds_results.items():
            if 'error' in prop_result:
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
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    md_lines = ['# Master Validation Report', '',
                '| Dataset | Property | N points | MAE (%) |',
                '|---|---|---|---|']
    for ds_name, ds_results in sorted(results.items()):
        if 'error' in ds_results:
            md_lines.append(f"| {ds_name} | LOAD ERROR | - | - |")
            continue
        for prop in sorted(ds_results.keys()):
            r = ds_results[prop]
            if 'error' in r:
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
    total_maes = []
    for ds_results in results.values():
        if 'error' in ds_results:
            continue
        for r in ds_results.values():
            if 'mae' in r:
                total_maes.append(r['mae'])
    if total_maes:
        print(f"\nOverall MAE: {np.mean(total_maes):.2f}% (n={len(total_maes)})")
