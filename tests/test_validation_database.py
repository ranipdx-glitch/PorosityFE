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
# Issue #34 — Unknown material mapping is a hard KeyError, not a silent
# fallback to T700_epoxy. The prior silent default masked at least three
# material mismatches (AS4/3501-6, HTA/EHkF420, generic Carbon/epoxy) in the
# shipped validation datasets and inflated the reported MAE.
# ---------------------------------------------------------------------------

def test_resolve_material_raises_on_unknown_fiber_matrix():
    """Unknown (fiber, matrix) must raise KeyError with the missing key, the
    dataset name, and the list of available presets — never silently fall
    back to a default preset."""
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

    with pytest.raises(KeyError) as exc_info:
        resolve_material(dataset)
    msg = str(exc_info.value)
    assert 'UnknownFiberXYZ' in msg, f"KeyError did not mention fiber: {msg}"
    assert 'UnknownMatrixABC' in msg, f"KeyError did not mention matrix: {msg}"
    assert 'fake_unknown_2099' in msg, f"KeyError did not mention dataset name: {msg}"
    # The error message should also point the caller at the fix.
    assert '_FIBER_MATRIX_TO_PRESET' in msg or 'MaterialProperties' in msg


def test_resolve_material_strict_kwarg_is_backward_compatible():
    """The ``strict`` kwarg is retained as a no-op for backward compatibility
    (issue #34 made the loud KeyError unconditional). Passing strict=False
    must still raise on an unknown (fiber, matrix) pair."""
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
    with pytest.raises(KeyError, match='UnknownFiberXYZ'):
        resolve_material(dataset, strict=True)
    with pytest.raises(KeyError, match='UnknownFiberXYZ'):
        resolve_material(dataset, strict=False)


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


def test_resolve_material_as4_3501_uses_dedicated_preset():
    """Ghiorse_1993 / Jeong_1997 materials (AS4/3501-6) must now resolve to
    the dedicated AS4_3501_6_epoxy preset, not silently fall back to T700
    (issue #34). AS4 is an IM-class fibre with higher E11 than T700."""
    from validation.validate_all import resolve_material

    for fiber in ('AS4', 'AS4 fabric'):
        dataset = {
            'reference': f'{fiber.lower().replace(" ", "_")}_test',
            'material': {
                'fiber': fiber,
                'matrix': '3501-6 epoxy',
                'fiber_volume_fraction': 0.60,
                'n_plies': 32,
            }
        }
        mat = resolve_material(dataset)
        # AS4/3501-6 E11 is ~142 GPa (significantly above T700's 132 GPa).
        assert 138000 <= mat.E11 <= 148000, (
            f"AS4/3501-6 should map to AS4_3501_6_epoxy with E11≈142 GPa, "
            f"got {mat.E11}"
        )


def test_resolve_material_hta_uses_dedicated_preset():
    """Stamopoulos_2016 material (HTA 24k / EHkF 420) must now resolve to
    the dedicated HTA_EHkF420_epoxy preset, not silently fall back to T700
    (issue #34)."""
    from validation.validate_all import resolve_material

    dataset = {
        'reference': 'hta_test',
        'material': {
            'fiber': 'HTA 24k',
            'matrix': 'EHkF 420 epoxy',
            'fiber_volume_fraction': 0.60,
            'n_plies': 16,
        }
    }
    mat = resolve_material(dataset)
    # HTA/EHkF420 fibre modulus is ~238 GPa (Toho Tenax HTA datasheet),
    # distinct from T700 (~230 GPa) and AS4 (~235 GPa). Sanity-check via
    # the preset's named fiber_modulus rather than the lamina E11.
    assert 236000 <= mat.fiber_modulus <= 240000, (
        f"HTA/EHkF420 should map to HTA_EHkF420_epoxy with E_f≈238 GPa, "
        f"got {mat.fiber_modulus}"
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


# ---------------------------------------------------------------------------
# Issue #56 — parallel per-dataset walk must be identical to the serial path
# ---------------------------------------------------------------------------

# Run the (slow) full serial walk exactly once and share it across the
# parallelism tests so we don't duplicate the expensive computation.
@pytest.fixture(scope="module")
def serial_results():
    from validation.validate_all import run_all_datasets
    return run_all_datasets(n_jobs=1)


def test_n_jobs_default_is_serial_path(serial_results):
    """n_jobs=1 (default) must equal an explicit serial run — back-compat."""
    from validation.validate_all import run_all_datasets
    assert run_all_datasets() == serial_results


def test_run_all_datasets_parallel_matches_serial(serial_results):
    """run_all_datasets(n_jobs=2) must return a result *equal* to the serial
    path: same keys, same order, same values (issue #56)."""
    from validation.validate_all import run_all_datasets
    parallel_results = run_all_datasets(n_jobs=2)
    # Identical keys in identical (sorted) order.
    assert list(parallel_results.keys()) == list(serial_results.keys())
    # Deep-equal values (MAE floats, predicted lists, error dicts, skips).
    assert parallel_results == serial_results


def test_run_all_datasets_n_jobs_all_cores_matches_serial(serial_results):
    """n_jobs=-1 (all cores) must also be identical to the serial path."""
    from validation.validate_all import run_all_datasets
    assert run_all_datasets(n_jobs=-1) == serial_results


def test_resolve_n_jobs_contract():
    """-1 and 0 map to os.cpu_count(); positive values pass through."""
    import os
    from validation.validate_all import _resolve_n_jobs
    expected_all = os.cpu_count() or 1
    assert _resolve_n_jobs(-1) == expected_all
    assert _resolve_n_jobs(0) == expected_all
    assert _resolve_n_jobs(1) == 1
    assert _resolve_n_jobs(4) == 4


def test_run_one_dataset_is_picklable():
    """The ProcessPool worker must be a top-level (picklable) function."""
    import pickle
    from validation.validate_all import _run_one_dataset
    assert pickle.loads(pickle.dumps(_run_one_dataset)) is _run_one_dataset
