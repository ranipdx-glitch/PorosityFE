#!/usr/bin/env python3
"""Tests for validation dataset schema and loader."""

import json
import os
import tempfile

import jsonschema
import pytest


SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'validation', 'schemas', 'validation_dataset_schema.json')


def test_schema_file_exists():
    assert os.path.exists(SCHEMA_PATH), f"Schema file missing at {SCHEMA_PATH}"


def test_schema_is_valid_jsonschema():
    with open(SCHEMA_PATH, encoding='utf-8') as f:
        schema = json.load(f)
    jsonschema.Draft7Validator.check_schema(schema)


def test_load_dataset_function_exists():
    from validation.validate_all import load_dataset
    assert callable(load_dataset)


def test_load_dataset_rejects_invalid_json():
    from validation.validate_all import load_dataset, ValidationError
    # Use mkstemp so the descriptor is closed before load_dataset reopens
    # the path — NamedTemporaryFile in text mode on Windows can hold a lock
    # that blocks the reopen even with delete=False.
    fd, tmppath = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({"reference": "too short"}, f)
        with pytest.raises(ValidationError):
            load_dataset(tmppath)
    finally:
        os.unlink(tmppath)


def test_elhajjar_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'elhajjar_2025.json')
    data = load_dataset(path)
    assert 'compression_strength' in data['properties']
    assert 'tensile_strength' in data['properties']
    assert data['material']['n_plies'] == 10


def test_liu_2006_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'liu_2006.json')
    data = load_dataset(path)
    assert len(data['properties']) == 4
    assert data['material']['layup_name'] == '[0/90]3s'


def test_stamopoulos_2016_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'stamopoulos_2016.json')
    data = load_dataset(path)
    assert len(data['properties']) == 6


def test_ghiorse_1993_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'ghiorse_1993.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_olivier_1995_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'olivier_1995.json')
    data = load_dataset(path)
    assert 'tensile_strength' in data['properties']


def test_almeida_1994_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'almeida_1994.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_tang_1987_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'tang_1987.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_bowles_1992_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'bowles_1992.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_jeong_1997_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'jeong_1997.json')
    data = load_dataset(path)
    assert 'ilss' in data['properties']


def test_liu_2018_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'liu_2018.json')
    data = load_dataset(path)
    assert 'tensile_strength' in data['properties']


def test_zhang_peek_2025_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'zhang_peek_2025.json')
    data = load_dataset(path)
    assert 'transverse_tensile_strength' in data['properties']


def test_wen_2023_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'wen_2023.json')
    data = load_dataset(path)
    assert 'compression_strength' in data['properties']


def test_wang_2022_dataset_loads():
    from validation.validate_all import load_dataset
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'validation', 'datasets', 'wang_2022.json')
    data = load_dataset(path)
    assert 'tensile_strength' in data['properties']


def test_resolve_material_from_dataset():
    from validation.validate_all import resolve_material
    dataset = {
        'material': {
            'fiber': 'T700',
            'matrix': 'TDE85 epoxy',
            'fiber_volume_fraction': 0.60,
            'n_plies': 12
        }
    }
    mat = resolve_material(dataset)
    assert mat.n_plies == 12
    assert abs(mat.fiber_volume_fraction - 0.60) < 1e-6


def test_resolve_material_uses_im7_for_8551():
    from validation.validate_all import resolve_material
    dataset = {
        'material': {
            'fiber': 'IM7',
            'matrix': '8551-7 epoxy',
            'fiber_volume_fraction': 0.60,
            'n_plies': 24
        }
    }
    mat = resolve_material(dataset)
    assert 170000 <= mat.E11 <= 180000


def test_predict_strength_returns_normalized_values():
    from validation.validate_all import predict_strength
    dataset = {
        'material': {
            'fiber': 'T700', 'matrix': 'TDE85 epoxy',
            'fiber_volume_fraction': 0.60, 'n_plies': 12,
            'ply_angles': [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]
        },
        'baseline_porosity_pct': 0.6,
    }
    vp_pcts = [0.6, 1.0, 2.0, 3.0]
    pred = predict_strength(dataset, 'tensile_strength', vp_pcts)
    assert len(pred) == 4
    assert abs(pred[0] - 1.0) < 0.01  # baseline normalizes to ~1
    assert pred[3] < pred[0]  # strength decreases with porosity


def test_predict_modulus_returns_normalized():
    from validation.validate_all import predict_modulus
    dataset = {
        'material': {
            'fiber': 'T700', 'matrix': 'TDE85 epoxy',
            'fiber_volume_fraction': 0.60, 'n_plies': 12,
            'ply_angles': [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]
        },
        'baseline_porosity_pct': 0.6,
    }
    pred = predict_modulus(dataset, 'tensile_modulus', [0.6, 1.0, 2.0, 3.0])
    assert len(pred) == 4
    assert pred[3] < pred[0]


def test_compute_mae():
    from validation.validate_all import compute_mae
    exp = [1.0, 0.9, 0.8]
    pred = [1.0, 0.85, 0.75]
    mae = compute_mae(pred, exp)
    expected = (0 + 5.56 + 6.25) / 3
    assert abs(mae - expected) < 0.5


def test_summarize_mae_returns_both_weightings():
    """Regression for #36: summary must distinguish property- vs. point-weighted."""
    from validation.validate_all import summarize_mae
    # Two papers: paper_A has 1 property with 10 points and MAE 10%;
    # paper_B has 1 property with 1 point and MAE 0%. Property-weighted
    # average is 5%; point-weighted average should be (10*10 + 1*0)/11 ≈ 9.09%.
    fake_results = {
        'paper_A': {'tensile_strength': {'mae': 10.0, 'n_points': 10}},
        'paper_B': {'tensile_strength': {'mae': 0.0, 'n_points': 1}},
    }
    s = summarize_mae(fake_results)
    assert abs(s['property_weighted_mae'] - 5.0) < 1e-9
    assert abs(s['point_weighted_mae'] - (10.0 * 10 + 0.0 * 1) / 11) < 1e-9
    assert s['n_entries'] == 2
    assert s['n_points'] == 11
    assert s['best_mae'] == 0.0
    assert s['worst_mae'] == 10.0

def test_summarize_mae_handles_empty():
    from validation.validate_all import summarize_mae
    import math
    s = summarize_mae({})
    assert s['n_entries'] == 0
    assert s['n_points'] == 0
    assert math.isnan(s['property_weighted_mae'])
    assert math.isnan(s['point_weighted_mae'])


def test_run_all_produces_per_dataset_mae():
    from validation.validate_all import run_all_datasets
    results = run_all_datasets()
    assert 'elhajjar_2025' in results
    assert 'liu_2006' in results
    liu = results['liu_2006']
    assert 'ilss' in liu
    assert 0 <= liu['ilss']['mae'] <= 100


def test_elhajjar_validation_matches_existing():
    """Regression: Elhajjar compression MAE is in expected range."""
    from validation.validate_all import run_all_datasets
    results = run_all_datasets()
    elh = results.get('elhajjar_2025', {})
    assert 'compression_strength' in elh
    # Historical Elhajjar compression MAE was ~6.9% (pre-migration)
    assert abs(elh['compression_strength']['mae'] - 6.9) < 1.5


def test_liu_2006_validation_matches_existing():
    from validation.validate_all import run_all_datasets
    results = run_all_datasets()
    liu = results.get('liu_2006', {})
    assert 'ilss' in liu
    # Historical Liu ILSS MAE was ~1.8%
    assert liu['ilss']['mae'] < 5.0


# ---------------------------------------------------------------------------
# Issue #34 — Unknown material mapping fires a UserWarning
# ---------------------------------------------------------------------------

def test_resolve_material_warns_on_unknown_fiber_matrix():
    """Unknown (fiber, matrix) should emit a UserWarning with the missing key
    and fall back to the default preset rather than silently proceeding."""
    import warnings
    from validation.validate_all import resolve_material

    dataset = {
        'reference': 'fake_unknown_2099',
        'material': {
            'fiber': 'UnknownFiberXYZ',
            'matrix': 'UnknownMatrixABC',
            'fiber_volume_fraction': 0.55,
            'n_plies': 8,
        }
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mat = resolve_material(dataset)
        # At least one UserWarning should have been raised
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, (
            "Expected a UserWarning for unknown (fiber, matrix) but got none"
        )
        msg = str(user_warnings[0].message)
        assert 'UnknownFiberXYZ' in msg, f"Warning did not mention fiber: {msg}"
        assert 'UnknownMatrixABC' in msg, f"Warning did not mention matrix: {msg}"
        assert 'fake_unknown_2099' in msg, f"Warning did not mention dataset name: {msg}"

    # Fallback should still return a valid MaterialProperties object
    assert mat is not None
    assert mat.n_plies == 8
    assert abs(mat.fiber_volume_fraction - 0.55) < 1e-6


def test_resolve_material_strict_raises_on_unknown():
    """strict=True should raise KeyError instead of warning and falling back."""
    from validation.validate_all import resolve_material

    dataset = {
        'reference': 'strict_test_dataset',
        'material': {
            'fiber': 'UnknownFiberXYZ',
            'matrix': 'UnknownMatrixABC',
            'fiber_volume_fraction': 0.55,
            'n_plies': 8,
        }
    }

    import pytest
    with pytest.raises(KeyError, match='UnknownFiberXYZ'):
        resolve_material(dataset, strict=True)


def test_resolve_material_no_warning_for_known_fiber_matrix():
    """Known (fiber, matrix) combinations must NOT emit any warning."""
    import warnings
    from validation.validate_all import resolve_material

    dataset = {
        'reference': 'known_material_test',
        'material': {
            'fiber': 'T700',
            'matrix': 'TDE85 epoxy',
            'fiber_volume_fraction': 0.60,
            'n_plies': 10,
        }
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        resolve_material(dataset)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, (
            f"Unexpected UserWarning for known material: {[str(x.message) for x in user_warnings]}"
        )


# ---------------------------------------------------------------------------
# Issue #35 — transverse_tensile_strength is skipped, not misrouted
# ---------------------------------------------------------------------------

def test_transverse_tensile_strength_is_skipped_in_run_all():
    """Datasets with transverse_tensile_strength should show 'skipped' rather
    than a spurious MAE computed via the wrong (longitudinal) failure mode."""
    from validation.validate_all import run_all_datasets
    results = run_all_datasets()

    datasets_with_transverse = ['zhang_peek_2025', 'liu_2018', 'stamopoulos_2016']
    for ds_name in datasets_with_transverse:
        if ds_name not in results:
            continue  # dataset might not be present in all environments
        ds = results[ds_name]
        if 'transverse_tensile_strength' not in ds:
            continue
        entry = ds['transverse_tensile_strength']
        assert 'skipped' in entry, (
            f"Expected 'skipped' for {ds_name}/transverse_tensile_strength "
            f"but got: {entry}"
        )
        assert 'mae' not in entry, (
            f"transverse_tensile_strength for {ds_name} should not have an MAE "
            "(would be computed via wrong physics)"
        )


def test_predict_strength_raises_for_transverse_tensile():
    """predict_strength should raise ValueError for transverse_tensile_strength."""
    import pytest
    from validation.validate_all import predict_strength

    dataset = {
        'reference': 'transverse_test',
        'material': {
            'fiber': 'T700',
            'matrix': 'TDE85 epoxy',
            'fiber_volume_fraction': 0.60,
            'n_plies': 12,
            'ply_angles': [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0],
        },
        'baseline_porosity_pct': 0.0,
    }

    with pytest.raises(ValueError, match='transverse_tensile_strength'):
        predict_strength(dataset, 'transverse_tensile_strength', [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Issue #48 (item 3) — regression-pin per-dataset/property MAE
#
# The pipeline is RNG-free and deterministic (#55), so these baselines are
# stable. Tolerance is rel=10% OR abs=0.5 (whichever is larger) so tiny-MAE
# entries don't go flaky while real model drift on any dataset fails CI.
# Regenerate with: python -c "from validation.validate_all import
# run_all_datasets as r; ..." (see PR for the one-liner) only on an
# intentional model change, and call it out in the PR.
# ---------------------------------------------------------------------------

_MAE_BASELINES = {
    ('almeida_1994', 'ilss'): 6.897,
    ('bowles_1992', 'ilss'): 3.019,
    ('elhajjar_2025', 'compression_strength'): 5.643,
    ('elhajjar_2025', 'tensile_strength'): 4.331,
    ('ghiorse_1993', 'flexural_modulus'): 16.284,
    ('ghiorse_1993', 'ilss'): 10.155,
    ('jeong_1997', 'ilss'): 4.156,
    ('liu_2006', 'flexural_modulus'): 7.707,
    ('liu_2006', 'ilss'): 1.987,
    ('liu_2006', 'tensile_modulus'): 1.808,
    ('liu_2006', 'tensile_strength'): 1.581,
    ('liu_2018', 'tensile_modulus'): 0.864,
    ('liu_2018', 'tensile_strength'): 5.26,
    ('liu_2018', 'transverse_tensile_modulus'): 3.286,
    ('olivier_1995', 'flexural_modulus'): 10.55,
    ('olivier_1995', 'ilss'): 1.554,
    ('olivier_1995', 'tensile_strength'): 15.667,
    ('stamopoulos_2016', 'flexural_modulus'): 2.314,
    ('stamopoulos_2016', 'ilss'): 2.523,
    ('stamopoulos_2016', 'shear_modulus'): 15.387,
    ('stamopoulos_2016', 'shear_strength'): 4.353,
    ('stamopoulos_2016', 'transverse_tensile_modulus'): 1.022,
    ('tang_1987', 'flexural_modulus'): 7.933,
    ('tang_1987', 'ilss'): 8.374,
    ('tang_1987', 'tensile_strength'): 10.923,
    ('wang_2022', 'tensile_modulus'): 1.336,
    ('wang_2022', 'tensile_strength'): 12.801,
    ('wen_2023', 'compression_strength'): 17.236,
    ('wen_2023', 'ilss'): 5.704,
    ('wen_2023', 'shear_strength'): 22.644,
    ('wen_2023', 'tensile_strength'): 6.583,
    ('zhang_peek_2025', 'transverse_tensile_modulus'): 5.96,
}


@pytest.fixture(scope="module")
def _all_results():
    """run_all_datasets is expensive (~30s); compute once for all 32 cases."""
    import warnings
    from validation.validate_all import run_all_datasets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_all_datasets()


@pytest.mark.parametrize("key", sorted(_MAE_BASELINES),
                         ids=lambda k: f"{k[0]}:{k[1]}")
def test_per_dataset_mae_regression_pinned(key, _all_results):
    dataset, prop = key
    baseline = _MAE_BASELINES[key]
    assert dataset in _all_results, f"dataset {dataset!r} missing from results"
    ds = _all_results[dataset]
    assert 'error' not in ds, f"{dataset} errored: {ds.get('error')}"
    assert prop in ds, f"{dataset!r} missing property {prop!r}"
    mae = ds[prop]['mae']
    assert mae == pytest.approx(baseline, rel=0.10, abs=0.5), (
        f"{dataset}/{prop} MAE drifted: {mae:.3f} vs baseline {baseline:.3f}"
    )
